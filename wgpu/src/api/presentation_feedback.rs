use alloc::{boxed::Box, sync::Arc};
use core::{
    fmt,
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
};

use crate::{dispatch, util::Mutex, PresentationFeedbackResult};

enum PresentationFeedbackState {
    Pending { waker: Option<Waker> },
    Ready(PresentationFeedbackResult),
}

struct SharedPresentationFeedback {
    state: Mutex<PresentationFeedbackState>,
}

impl SharedPresentationFeedback {
    fn complete(&self, result: PresentationFeedbackResult) {
        let waker = {
            let mut state = self.state.lock();
            match &mut *state {
                PresentationFeedbackState::Pending { waker } => {
                    let waker = waker.take();
                    *state = PresentationFeedbackState::Ready(result);
                    waker
                }
                PresentationFeedbackState::Ready(_) => {
                    log::warn!("presentation feedback completed more than once");
                    None
                }
            }
        };

        if let Some(waker) = waker {
            wake_without_unwinding(waker);
        }
    }
}

fn wake_without_unwinding(waker: Waker) {
    #[cfg(std)]
    {
        let _ = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| waker.wake()));
    }
    #[cfg(not(std))]
    waker.wake();
}

/// Future terminal feedback for one exact surface presentation.
///
/// Dropping this future abandons observation. It does not cancel or relabel the presentation.
/// Backends that cannot produce terminal evidence resolve this future with
/// [`PresentationFeedbackError::Unsupported`](crate::PresentationFeedbackError::Unsupported).
/// The registered task waker may be invoked from a platform presentation callback thread and must
/// therefore return promptly; the future body itself runs only when its executor polls it.
#[must_use = "presentation feedback does nothing unless awaited or polled"]
pub struct PresentationFeedbackFuture {
    shared: Arc<SharedPresentationFeedback>,
}

impl PresentationFeedbackFuture {
    pub(crate) fn pending() -> (Self, dispatch::PresentationFeedbackCallback) {
        let shared = Arc::new(SharedPresentationFeedback {
            state: Mutex::new(PresentationFeedbackState::Pending { waker: None }),
        });
        let weak = Arc::downgrade(&shared);
        let callback = Box::new(move |result| {
            if let Some(shared) = weak.upgrade() {
                shared.complete(result);
            }
        });
        (Self { shared }, callback)
    }
}

impl fmt::Debug for PresentationFeedbackFuture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PresentationFeedbackFuture")
            .finish_non_exhaustive()
    }
}

impl Future for PresentationFeedbackFuture {
    type Output = PresentationFeedbackResult;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.shared.state.lock();
        match &mut *state {
            PresentationFeedbackState::Ready(result) => Poll::Ready(*result),
            PresentationFeedbackState::Pending { waker } => {
                if waker
                    .as_ref()
                    .is_none_or(|registered| !registered.will_wake(cx.waker()))
                {
                    *waker = Some(cx.waker().clone());
                }
                Poll::Pending
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;
    use core::{future::Future as _, task::Waker};

    use super::*;
    use crate::{PresentationFeedback, PresentationTimestamp};

    #[test]
    fn future_completes_exactly_once() {
        let (mut future, callback) = PresentationFeedbackFuture::pending();
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        assert_eq!(Pin::new(&mut future).poll(&mut cx), Poll::Pending);

        callback(Ok(PresentationFeedback::Presented {
            timestamp: PresentationTimestamp(42),
        }));

        assert_eq!(
            Pin::new(&mut future).poll(&mut cx),
            Poll::Ready(Ok(PresentationFeedback::Presented {
                timestamp: PresentationTimestamp(42),
            }))
        );
    }

    #[test]
    fn dropping_future_abandons_observation() {
        let (future, callback) = PresentationFeedbackFuture::pending();
        drop(future);
        callback(Err(crate::PresentationFeedbackError::Cancelled));
    }

    #[test]
    fn hostile_waker_cannot_unwind_completion() {
        struct PanickingWake;
        impl alloc::task::Wake for PanickingWake {
            fn wake(self: Arc<Self>) {
                panic!("hostile waker");
            }
        }

        let (mut future, callback) = PresentationFeedbackFuture::pending();
        let waker = Waker::from(Arc::new(PanickingWake));
        let mut cx = Context::from_waker(&waker);
        assert_eq!(Pin::new(&mut future).poll(&mut cx), Poll::Pending);
        callback(Ok(PresentationFeedback::NotPresented));
        assert_eq!(
            Pin::new(&mut future).poll(&mut cx),
            Poll::Ready(Ok(PresentationFeedback::NotPresented))
        );
    }
}
