/*! Presentation.

## Lifecycle

Whenever a submission detects the use of any surface texture, it adds it to the device
tracker for the duration of the submission (temporarily, while recording).
It's added with `UNINITIALIZED` state and transitioned into `empty()` state.
When this texture is presented, we remove it from the device tracker as well as
extract it from the hub.
!*/

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::mem::ManuallyDrop;

#[cfg(feature = "trace")]
use crate::device::trace::{Action, IntoTrace};
use crate::{
    conv,
    device::{queue::Queue, Device, DeviceError, MissingDownlevelFlags, WaitIdleError},
    hal_label,
    instance::Surface,
    resource::{self, Labeled},
};

use thiserror::Error;
use wgt::{
    error::{ErrorType, WebGpuError},
    PresentationFeedbackError, PresentationFeedbackResult, SurfaceStatus as Status,
};

const FRAME_TIMEOUT_MS: u32 = 1000;

#[derive(Debug)]
pub(crate) struct Presentation {
    pub(crate) device: Arc<Device>,
    pub(crate) config: wgt::SurfaceConfiguration<Vec<wgt::TextureFormat>>,
    pub(crate) acquired_texture: Option<Arc<resource::Texture>>,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum SurfaceError {
    #[error("Surface is invalid")]
    Invalid,
    #[error("Surface is not configured for presentation")]
    NotConfigured,
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Surface image is already acquired")]
    AlreadyAcquired,
    #[error("No surface image is currently acquired to present")]
    NothingToPresent,
    #[error("Texture has been destroyed")]
    TextureDestroyed,
    #[error("Surface image does not match the exact acquired texture")]
    AcquiredTextureMismatch,
}

impl WebGpuError for SurfaceError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::Invalid
            | Self::NotConfigured
            | Self::AlreadyAcquired
            | Self::NothingToPresent
            | Self::TextureDestroyed
            | Self::AcquiredTextureMismatch => ErrorType::Validation,
        }
    }
}

enum PresentationFeedbackGateState {
    Closed {
        callback: Option<hal::PresentationFeedbackCallback>,
        result: Option<PresentationFeedbackResult>,
    },
    Open {
        callback: Option<hal::PresentationFeedbackCallback>,
    },
    Complete,
}

struct PresentationFeedbackGate {
    state: wgpu_sync::Mutex<PresentationFeedbackGateState>,
}

impl PresentationFeedbackGate {
    fn new(callback: hal::PresentationFeedbackCallback) -> Arc<Self> {
        Arc::new(Self {
            state: wgpu_sync::Mutex::new(PresentationFeedbackGateState::Closed {
                callback: Some(callback),
                result: None,
            }),
        })
    }

    fn complete(&self, result: PresentationFeedbackResult) {
        let callback = {
            let mut state = self.state.lock();
            match &mut *state {
                PresentationFeedbackGateState::Closed {
                    result: pending, ..
                } => {
                    if pending.is_none() {
                        *pending = Some(result);
                    } else {
                        log::warn!("presentation feedback completed more than once");
                    }
                    None
                }
                PresentationFeedbackGateState::Open { callback } => {
                    let callback = callback.take();
                    *state = PresentationFeedbackGateState::Complete;
                    callback
                }
                PresentationFeedbackGateState::Complete => {
                    log::warn!("presentation feedback completed more than once");
                    None
                }
            }
        };

        if let Some(callback) = callback {
            callback(result);
        }
    }

    fn open(&self) {
        let completion = {
            let mut state = self.state.lock();
            match &mut *state {
                PresentationFeedbackGateState::Closed { callback, result } => {
                    if let Some(result) = result.take() {
                        let callback = callback.take();
                        *state = PresentationFeedbackGateState::Complete;
                        callback.map(|callback| (callback, result))
                    } else {
                        let callback = callback.take();
                        *state = PresentationFeedbackGateState::Open { callback };
                        None
                    }
                }
                PresentationFeedbackGateState::Open { .. }
                | PresentationFeedbackGateState::Complete => None,
            }
        };

        if let Some((callback, result)) = completion {
            callback(result);
        }
    }
}

fn feedback_error(error: &SurfaceError) -> PresentationFeedbackError {
    match error {
        SurfaceError::Device(DeviceError::Lost) => PresentationFeedbackError::DeviceLost,
        SurfaceError::Device(DeviceError::OutOfMemory) => PresentationFeedbackError::OutOfMemory,
        SurfaceError::Device(DeviceError::DeviceMismatch(_))
        | SurfaceError::Invalid
        | SurfaceError::NotConfigured
        | SurfaceError::AlreadyAcquired
        | SurfaceError::NothingToPresent
        | SurfaceError::TextureDestroyed
        | SurfaceError::AcquiredTextureMismatch => PresentationFeedbackError::Validation,
    }
}

fn take_exact_acquisition<T>(
    acquired: &mut Option<Arc<T>>,
    expected: Option<&Arc<T>>,
) -> Result<Arc<T>, SurfaceError> {
    let current = acquired.as_ref().ok_or(SurfaceError::NothingToPresent)?;
    if expected.is_some_and(|expected| !Arc::ptr_eq(current, expected)) {
        return Err(SurfaceError::AcquiredTextureMismatch);
    }
    Ok(acquired.take().unwrap())
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum ConfigureSurfaceError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Invalid surface")]
    InvalidSurface,
    #[error("The view format {0:?} is not compatible with texture format {1:?}, only changing srgb-ness is allowed.")]
    InvalidViewFormat(wgt::TextureFormat, wgt::TextureFormat),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
    #[error("The `SurfaceOutput` returned by `get_current_texture` must be dropped before re-configuring via `configure` or  retrieving a new texture via `get_current_texture`.")]
    PreviousOutputExists,
    #[error("Failed to wait for GPU to come idle before reconfiguring the Surface")]
    GpuWaitTimeout,
    #[error("Both `Surface` width and height must be non-zero. Wait to recreate the `Surface` until the window has non-zero area.")]
    ZeroArea,
    #[error("`Surface` width and height must be within the maximum supported texture size. Requested was ({width}, {height}), maximum extent for either dimension is {max_texture_dimension_2d}.")]
    TooLarge {
        width: u32,
        height: u32,
        max_texture_dimension_2d: u32,
    },
    #[error("Surface does not support the adapter's queue family")]
    UnsupportedQueueFamily,
    #[error("Requested format {requested:?} is not in list of supported formats: {available:?}")]
    UnsupportedFormat {
        requested: wgt::TextureFormat,
        available: Vec<wgt::TextureFormat>,
    },
    #[error("Requested color space {requested:?} is not in the list of color spaces supported for format {format:?}: {available:?}")]
    UnsupportedColorSpace {
        requested: wgt::SurfaceColorSpace,
        format: wgt::TextureFormat,
        available: wgt::SurfaceColorSpaces,
    },
    #[error("Requested present mode {requested:?} is not in the list of supported present modes: {available:?}")]
    UnsupportedPresentMode {
        requested: wgt::PresentMode,
        available: Vec<wgt::PresentMode>,
    },
    #[error("Requested alpha mode {requested:?} is not in the list of supported alpha modes: {available:?}")]
    UnsupportedAlphaMode {
        requested: wgt::CompositeAlphaMode,
        available: Vec<wgt::CompositeAlphaMode>,
    },
    #[error("Requested usage {requested:?} is not in the list of supported usages: {available:?}")]
    UnsupportedUsage {
        requested: wgt::TextureUses,
        available: wgt::TextureUses,
    },
}

impl From<WaitIdleError> for ConfigureSurfaceError {
    fn from(e: WaitIdleError) -> Self {
        match e {
            WaitIdleError::Device(d) => ConfigureSurfaceError::Device(d),
            WaitIdleError::WrongSubmissionIndex(..) => unreachable!(),
            WaitIdleError::Timeout => ConfigureSurfaceError::GpuWaitTimeout,
        }
    }
}

impl WebGpuError for ConfigureSurfaceError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::MissingDownlevelFlags(e) => e.webgpu_error_type(),
            Self::InvalidSurface
            | Self::InvalidViewFormat(..)
            | Self::PreviousOutputExists
            | Self::GpuWaitTimeout
            | Self::ZeroArea
            | Self::TooLarge { .. }
            | Self::UnsupportedQueueFamily
            | Self::UnsupportedFormat { .. }
            | Self::UnsupportedColorSpace { .. }
            | Self::UnsupportedPresentMode { .. }
            | Self::UnsupportedAlphaMode { .. }
            | Self::UnsupportedUsage { .. } => ErrorType::Validation,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SurfaceOutput<T = Arc<resource::Texture>> {
    pub status: Status,
    pub texture: Option<T>,
}

impl Surface {
    pub fn get_current_texture(self: &Arc<Self>) -> Result<SurfaceOutput, SurfaceError> {
        let output = self.get_current_texture_inner();
        #[cfg(feature = "trace")]
        if let Some(present) = self.presentation.lock().as_ref() {
            if let Some(ref mut trace) = *present.device.trace.lock() {
                if let Some(texture) = present.acquired_texture.as_ref() {
                    trace.add(Action::GetSurfaceTexture {
                        id: texture.to_trace(),
                        parent: self.to_trace(),
                    });
                }
            }
        }
        output
    }

    pub(crate) fn get_current_texture_inner(&self) -> Result<SurfaceOutput, SurfaceError> {
        profiling::scope!("Surface::get_current_texture");

        let (device, config) = if let Some(ref present) = *self.presentation.lock() {
            present.device.check_is_valid()?;
            (present.device.clone(), present.config.clone())
        } else {
            return Err(SurfaceError::NotConfigured);
        };

        let suf = self.raw(device.backend()).unwrap();
        let (texture, status) = match unsafe {
            suf.acquire_texture(
                Some(core::time::Duration::from_millis(FRAME_TIMEOUT_MS as u64)),
                device.fence.as_ref(),
            )
        } {
            Ok(ast) => {
                let texture_desc = wgt::TextureDescriptor {
                    label: hal_label(
                        Some(alloc::borrow::Cow::Borrowed("<Surface Texture>")),
                        device.instance_flags,
                    ),
                    size: wgt::Extent3d {
                        width: config.width,
                        height: config.height,
                        depth_or_array_layers: 1,
                    },
                    sample_count: 1,
                    mip_level_count: 1,
                    format: config.format,
                    dimension: wgt::TextureDimension::D2,
                    usage: config.usage,
                    view_formats: config.view_formats,
                };
                let format_features = wgt::TextureFormatFeatures {
                    allowed_usages: wgt::TextureUsages::RENDER_ATTACHMENT,
                    flags: wgt::TextureFormatFeatureFlags::MULTISAMPLE_X4
                        | wgt::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE,
                };
                let hal_usage = conv::map_texture_usage(
                    config.usage,
                    config.format.into(),
                    format_features.flags,
                );
                let clear_view_desc = hal::TextureViewDescriptor {
                    label: hal_label(
                        Some("(wgpu internal) clear surface texture view"),
                        device.instance_flags,
                    ),
                    format: config.format,
                    dimension: wgt::TextureViewDimension::D2,
                    usage: wgt::TextureUses::COLOR_TARGET,
                    range: wgt::ImageSubresourceRange::default(),
                };
                let clear_view = unsafe {
                    device
                        .raw()
                        .create_texture_view(ast.texture.as_ref().borrow(), &clear_view_desc)
                }
                .map_err(|e| device.handle_hal_error(e))?;

                let mut presentation = self.presentation.lock();
                let present = presentation.as_mut().unwrap();
                let texture = resource::Texture::new(
                    &device,
                    resource::TextureInner::Surface { raw: ast.texture },
                    hal_usage,
                    &texture_desc,
                    format_features,
                    resource::TextureClearMode::Surface {
                        clear_view: ManuallyDrop::new(clear_view),
                    },
                    true,
                );

                let texture = Arc::new(texture);

                device
                    .trackers
                    .lock()
                    .textures
                    .insert_single(&texture, wgt::TextureUses::UNINITIALIZED);

                if present.acquired_texture.is_some() {
                    return Err(SurfaceError::AlreadyAcquired);
                }
                present.acquired_texture = Some(texture.clone());

                let status = if ast.suboptimal {
                    Status::Suboptimal
                } else {
                    Status::Good
                };
                (Some(texture), status)
            }
            Err(err) => (
                None,
                match err {
                    hal::SurfaceError::Timeout => Status::Timeout,
                    hal::SurfaceError::Occluded => Status::Occluded,
                    hal::SurfaceError::Lost => Status::Lost,
                    hal::SurfaceError::Device(err) => {
                        return Err(device.handle_hal_error(err).into());
                    }
                    hal::SurfaceError::Outdated => Status::Outdated,
                    hal::SurfaceError::Other(msg) => {
                        log::error!("acquire error: {msg}");
                        Status::Lost
                    }
                },
            ),
        };

        Ok(SurfaceOutput { status, texture })
    }

    pub fn present(self: &Arc<Self>) -> Result<Status, SurfaceError> {
        self.present_inner()
    }

    pub(crate) fn present_inner(&self) -> Result<Status, SurfaceError> {
        profiling::scope!("Surface::present");

        let presentation = self.presentation.lock();
        let present = match presentation.as_ref() {
            Some(present) => present,
            None => return Err(SurfaceError::NotConfigured),
        };

        present.device.check_is_valid()?;
        let queue = present
            .device
            .get_queue()
            .ok_or(SurfaceError::Device(DeviceError::Lost))?;
        drop(presentation);

        queue.present(self)
    }
}

impl Queue {
    pub fn present(&self, surface: &Surface) -> Result<Status, SurfaceError> {
        profiling::scope!("Queue::present");

        let texture = self.take_surface_texture(surface, None)?;

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *self.device.trace.lock() {
            trace.add(Action::Present {
                surface: unsafe { crate::device::trace::to_trace(surface) },
                texture: texture.to_trace(),
            });
        }

        self.present_surface_texture(surface, texture)
    }

    #[doc(hidden)]
    pub fn present_acquired(
        &self,
        surface: &Surface,
        expected: &Arc<resource::Texture>,
    ) -> Result<Status, SurfaceError> {
        profiling::scope!("Queue::present_acquired");

        let texture = self.take_surface_texture(surface, Some(expected))?;

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *self.device.trace.lock() {
            trace.add(Action::Present {
                surface: unsafe { crate::device::trace::to_trace(surface) },
                texture: texture.to_trace(),
            });
        }

        self.present_surface_texture(surface, texture)
    }

    #[doc(hidden)]
    pub fn present_acquired_with_feedback(
        &self,
        surface: &Surface,
        expected: &Arc<resource::Texture>,
        callback: hal::PresentationFeedbackCallback,
    ) -> Result<Status, SurfaceError> {
        profiling::scope!("Queue::present_acquired_with_feedback");

        let gate = PresentationFeedbackGate::new(callback);
        let result = self.present_acquired_with_feedback_inner(surface, expected, &gate);
        if let Err(error) = &result {
            gate.complete(Err(feedback_error(error)));
        }
        gate.open();
        result
    }

    fn take_surface_texture(
        &self,
        surface: &Surface,
        expected: Option<&Arc<resource::Texture>>,
    ) -> Result<Arc<resource::Texture>, SurfaceError> {
        {
            let mut presentation = surface.presentation.lock();
            let present = match presentation.as_mut() {
                Some(present) => present,
                None => return Err(SurfaceError::NotConfigured),
            };

            let device = &self.device;

            // Check the surface is configured for this device.
            if !Arc::ptr_eq(&present.device, device) {
                return Err(SurfaceError::Device(DeviceError::DeviceMismatch(Box::new(
                    crate::device::DeviceMismatch {
                        res: self.error_ident(),
                        res_device: device.error_ident(),
                        target: None,
                        target_device: present.device.error_ident(),
                    },
                ))));
            }

            take_exact_acquisition(&mut present.acquired_texture, expected)
        }
    }

    fn present_surface_texture(
        &self,
        surface: &Surface,
        texture: Arc<resource::Texture>,
    ) -> Result<Status, SurfaceError> {
        // If the texture was never rendered to, clear it and transition to
        // PRESENT state before presenting.
        // Fixes <https://github.com/gfx-rs/wgpu/issues/6748>
        self.prepare_surface_texture_for_present(&texture)?;

        let device = &self.device;

        let mut exclusive_snatch_guard = device.snatchable_lock.write();
        let inner = texture
            .state()
            .ok()
            .and_then(|state| state.inner.snatch(&mut exclusive_snatch_guard));
        drop(exclusive_snatch_guard);

        let result = match inner {
            None => return Err(SurfaceError::TextureDestroyed),
            Some(resource::TextureInner::Surface { raw }) => {
                let raw_surface = surface.raw(device.backend()).unwrap();
                let raw_queue = self.raw();
                // [`wgpu_hal::Queue::present`] requires the queue to be synchronized with submit calls and
                // other present calls. Locking command indices prevents submits which must increment the
                // submission index, and by `write`ing prevents other present calls.
                let _command_indices = device.command_indices.write();
                unsafe { raw_queue.present(raw_surface, raw) }
            }
            _ => unreachable!(),
        };

        match result {
            Ok(()) => Ok(Status::Good),
            Err(err) => match err {
                hal::SurfaceError::Timeout => Ok(Status::Timeout),
                hal::SurfaceError::Occluded => Ok(Status::Occluded),
                hal::SurfaceError::Lost => Ok(Status::Lost),
                hal::SurfaceError::Device(err) => {
                    Err(SurfaceError::from(device.handle_hal_error(err)))
                }
                hal::SurfaceError::Outdated => Ok(Status::Outdated),
                hal::SurfaceError::Other(msg) => {
                    log::error!("present error: {msg}");
                    Err(SurfaceError::Invalid)
                }
            },
        }
    }

    fn present_acquired_with_feedback_inner(
        &self,
        surface: &Surface,
        expected: &Arc<resource::Texture>,
        gate: &Arc<PresentationFeedbackGate>,
    ) -> Result<Status, SurfaceError> {
        let texture = self.take_surface_texture(surface, Some(expected))?;

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *self.device.trace.lock() {
            trace.add(Action::Present {
                surface: unsafe { crate::device::trace::to_trace(surface) },
                texture: texture.to_trace(),
            });
        }

        self.prepare_surface_texture_for_present(&texture)?;

        let device = &self.device;
        let mut exclusive_snatch_guard = device.snatchable_lock.write();
        let inner = texture
            .state()
            .ok()
            .and_then(|state| state.inner.snatch(&mut exclusive_snatch_guard));
        drop(exclusive_snatch_guard);

        let result = match inner {
            None => return Err(SurfaceError::TextureDestroyed),
            Some(resource::TextureInner::Surface { raw }) => {
                let raw_surface = surface.raw(device.backend()).unwrap();
                let raw_queue = self.raw();
                let command_indices = device.command_indices.write();
                let hal_gate = Arc::clone(gate);
                let result = unsafe {
                    raw_queue.present_with_feedback(
                        raw_surface,
                        raw,
                        Box::new(move |result| hal_gate.complete(result)),
                    )
                };
                drop(command_indices);
                result
            }
            _ => unreachable!(),
        };

        match result {
            Ok(()) => Ok(Status::Good),
            Err(err) => {
                let feedback = match &err {
                    hal::SurfaceError::Timeout | hal::SurfaceError::Occluded => {
                        Ok(wgt::PresentationFeedback::NotPresented)
                    }
                    hal::SurfaceError::Lost | hal::SurfaceError::Outdated => {
                        Err(PresentationFeedbackError::SurfaceLost)
                    }
                    hal::SurfaceError::Device(hal::DeviceError::OutOfMemory) => {
                        Err(PresentationFeedbackError::OutOfMemory)
                    }
                    hal::SurfaceError::Device(
                        hal::DeviceError::Lost | hal::DeviceError::Unexpected,
                    ) => Err(PresentationFeedbackError::DeviceLost),
                    hal::SurfaceError::Other(_) => Err(PresentationFeedbackError::ProtocolFailure),
                };
                gate.complete(feedback);

                match err {
                    hal::SurfaceError::Timeout => Ok(Status::Timeout),
                    hal::SurfaceError::Occluded => Ok(Status::Occluded),
                    hal::SurfaceError::Lost => Ok(Status::Lost),
                    hal::SurfaceError::Device(err) => {
                        Err(SurfaceError::from(device.handle_hal_error(err)))
                    }
                    hal::SurfaceError::Outdated => Ok(Status::Outdated),
                    hal::SurfaceError::Other(msg) => {
                        log::error!("present error: {msg}");
                        Err(SurfaceError::Invalid)
                    }
                }
            }
        }
    }
}

impl Surface {
    pub fn discard(self: &Arc<Self>) -> Result<(), SurfaceError> {
        self.discard_inner(None)
    }

    #[doc(hidden)]
    pub fn discard_acquired(&self, expected: &Arc<resource::Texture>) -> Result<(), SurfaceError> {
        self.discard_inner(Some(expected))
    }

    pub(crate) fn discard_inner(
        &self,
        expected: Option<&Arc<resource::Texture>>,
    ) -> Result<(), SurfaceError> {
        profiling::scope!("Surface::discard");

        let mut presentation = self.presentation.lock();
        let present = match presentation.as_mut() {
            Some(present) => present,
            None => return Err(SurfaceError::NotConfigured),
        };

        let device = &present.device;

        device.check_is_valid()?;

        let texture = take_exact_acquisition(&mut present.acquired_texture, expected)?;

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *device.trace.lock() {
            trace.add(Action::DiscardSurfaceTexture {
                surface: unsafe { crate::device::trace::to_trace(self) },
                texture: texture.to_trace(),
            });
        }

        let mut exclusive_snatch_guard = device.snatchable_lock.write();
        let inner = texture
            .state()
            .ok()
            .and_then(|state| state.inner.snatch(&mut exclusive_snatch_guard));
        drop(exclusive_snatch_guard);

        match inner {
            None => return Err(SurfaceError::TextureDestroyed),
            Some(resource::TextureInner::Surface { raw }) => {
                let raw_surface = self.raw(device.backend()).unwrap();
                unsafe { raw_surface.discard_texture(raw) };
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    pub fn release(self: &Arc<Self>) -> Result<(), SurfaceError> {
        self.release_inner(None)
    }

    #[doc(hidden)]
    pub fn release_acquired(&self, expected: &Arc<resource::Texture>) -> Result<(), SurfaceError> {
        self.release_inner(Some(expected))
    }

    /// Like `discard`, drops the inner texture reference, but skips the
    /// HAL `discard_texture` call. Safe to call during unwinding
    pub(crate) fn release_inner(
        &self,
        expected: Option<&Arc<resource::Texture>>,
    ) -> Result<(), SurfaceError> {
        profiling::scope!("Surface::release");

        let mut presentation = self.presentation.lock();
        let Some(present) = presentation.as_mut() else {
            return Err(SurfaceError::NotConfigured);
        };

        let texture = take_exact_acquisition(&mut present.acquired_texture, expected)?;

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *present.device.trace.lock() {
            trace.add(Action::ReleaseSurfaceTexture {
                surface: unsafe { crate::device::trace::to_trace(self) },
                texture: texture.to_trace(),
            });
        }

        // `texture` is dropped here, decrementing the refcount of
        // Arc<SwapchainAcquireSemaphore>. If this was the last Arc, the Texture
        // is freed, which drops NativeSurfaceTextureMetadata and
        // its Arc<SwapchainAcquireSemaphore>.
        drop(texture);

        Ok(())
    }
}

#[cfg(test)]
mod presentation_feedback_tests {
    use super::*;

    #[test]
    fn synchronous_completion_waits_until_gate_opens() {
        let results = Arc::new(wgpu_sync::Mutex::new(Vec::new()));
        let captured = Arc::clone(&results);
        let gate = PresentationFeedbackGate::new(Box::new(move |result| {
            captured.lock().push(result);
        }));

        gate.complete(Ok(wgt::PresentationFeedback::NotPresented));
        assert!(results.lock().is_empty());
        gate.open();
        assert_eq!(
            *results.lock(),
            [Ok(wgt::PresentationFeedback::NotPresented)]
        );
    }

    #[test]
    fn asynchronous_completion_after_open_is_exactly_once() {
        let results = Arc::new(wgpu_sync::Mutex::new(Vec::new()));
        let captured = Arc::clone(&results);
        let gate = PresentationFeedbackGate::new(Box::new(move |result| {
            captured.lock().push(result);
        }));

        gate.open();
        gate.complete(Ok(wgt::PresentationFeedback::NotPresented));
        gate.complete(Err(PresentationFeedbackError::ProtocolFailure));
        assert_eq!(
            *results.lock(),
            [Ok(wgt::PresentationFeedback::NotPresented)]
        );
    }

    #[test]
    fn stale_acquisition_cannot_consume_a_later_texture() {
        let first = Arc::new(());
        let second = Arc::new(());
        let mut acquired = Some(Arc::clone(&second));

        assert!(matches!(
            take_exact_acquisition(&mut acquired, Some(&first)),
            Err(SurfaceError::AcquiredTextureMismatch)
        ));
        assert!(Arc::ptr_eq(acquired.as_ref().unwrap(), &second));
        assert!(Arc::ptr_eq(
            &take_exact_acquisition(&mut acquired, Some(&second)).unwrap(),
            &second
        ));
        assert!(matches!(
            take_exact_acquisition(&mut acquired, Some(&second)),
            Err(SurfaceError::NothingToPresent)
        ));
    }
}
