use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

#[derive(Debug)]
struct TrainingFrame {
    token: Rc<()>,
    training: bool,
}

#[derive(Debug)]
struct TrainingState {
    frames: Vec<TrainingFrame>,
}

impl TrainingState {
    const fn new() -> Self {
        Self { frames: Vec::new() }
    }

    fn push(&mut self, training: bool) -> Rc<()> {
        let token = Rc::new(());
        self.frames.push(TrainingFrame {
            token: Rc::clone(&token),
            training,
        });
        token
    }

    fn remove(&mut self, token: &Rc<()>) {
        if let Some(position) = self
            .frames
            .iter()
            .position(|frame| Rc::ptr_eq(&frame.token, token))
        {
            self.frames.remove(position);
        }
    }

    fn is_training(&self) -> bool {
        self.frames.last().is_some_and(|frame| frame.training)
    }
}

thread_local! {
    static TRAINING: RefCell<TrainingState> = const { RefCell::new(TrainingState::new()) };
}

/// Scoped ambient training mode for source-facing graph compositions.
///
/// The default is evaluation mode. Dropping a guard removes its frame and
/// restores the newest remaining frame, including during unwinding and when
/// guards are dropped out of order. Guards are deliberately thread-affine.
/// Existing APIs that accept an explicit mode remain independent of this context.
#[derive(Debug)]
#[must_use = "ambient training mode remains active until this guard is dropped"]
pub struct TrainingContext {
    token: Rc<()>,
    _thread_affine: PhantomData<Rc<()>>,
}

impl TrainingContext {
    /// Enters `training` until the returned guard is dropped.
    pub fn enter(training: bool) -> Self {
        let token = TRAINING.with_borrow_mut(|state| state.push(training));
        Self {
            token,
            _thread_affine: PhantomData,
        }
    }

    /// Enters training mode until the returned guard is dropped.
    pub fn training() -> Self {
        Self::enter(true)
    }

    /// Enters evaluation mode until the returned guard is dropped.
    pub fn evaluation() -> Self {
        Self::enter(false)
    }

    /// Returns the current thread's ambient training mode.
    pub fn is_training() -> bool {
        TRAINING.with_borrow(TrainingState::is_training)
    }
}

impl Drop for TrainingContext {
    fn drop(&mut self) {
        TRAINING.with_borrow_mut(|state| state.remove(&self.token));
    }
}

#[cfg(test)]
mod tests {
    use super::TrainingContext;

    #[test]
    fn scoped_training_mode_nests_and_restores_after_panic() {
        assert!(!TrainingContext::is_training());
        let outer = TrainingContext::training();
        assert!(TrainingContext::is_training());
        {
            let _evaluation = TrainingContext::evaluation();
            assert!(!TrainingContext::is_training());
        }
        assert!(TrainingContext::is_training());
        let result = std::panic::catch_unwind(|| {
            let _evaluation = TrainingContext::evaluation();
            assert!(!TrainingContext::is_training());
            panic!("restore ambient training mode");
        });
        assert!(result.is_err());
        assert!(TrainingContext::is_training());
        drop(outer);
        assert!(!TrainingContext::is_training());
    }

    #[test]
    fn scoped_training_mode_is_thread_local() {
        let _training = TrainingContext::training();
        assert!(TrainingContext::is_training());
        std::thread::spawn(|| {
            assert!(!TrainingContext::is_training());
            let _training = TrainingContext::training();
            assert!(TrainingContext::is_training());
        })
        .join()
        .unwrap();
        assert!(TrainingContext::is_training());
    }

    #[test]
    fn scoped_training_mode_tolerates_out_of_order_guard_drops() {
        assert!(!TrainingContext::is_training());
        let training = TrainingContext::training();
        let evaluation = TrainingContext::evaluation();
        let inner_training = TrainingContext::training();
        assert!(TrainingContext::is_training());

        drop(evaluation);
        assert!(TrainingContext::is_training());
        drop(inner_training);
        assert!(TrainingContext::is_training());
        drop(training);
        assert!(!TrainingContext::is_training());
    }
}
