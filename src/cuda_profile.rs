#![allow(dead_code)] // crate-private Phase A is consumed by the next CUDA adapter phase.
//! Driver-free ownership foundation for future CUDA timing adapters.
//!
//! The recorder lock protects only trace mutation. Completion is supplied by a
//! future adapter outside that lock; samples retain an `Arc` sentinel until
//! collected, failed, or abandoned. Records are ordered by submission sequence.

use crate::DeviceId;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OperationKind {
    Kernel,
    HtoD,
    DtoH,
    DtoD,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Completion {
    Pending,
    Ready(Option<u64>),
    Collected,
    Failed(String),
    Abandoned,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Metadata {
    pub kind: OperationKind,
    pub name: String,
    pub owner: usize,
    pub device: DeviceId,
    pub stream: usize,
    pub bytes: Option<usize>,
    pub geometry: Option<([u32; 3], [u32; 3])>,
    pub source_key: Option<String>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Record {
    pub sequence: u64,
    pub metadata: Metadata,
    pub completion: Completion,
}
#[derive(Clone)]
pub(crate) enum ProfilingSession {
    Disabled,
    Enabled(Arc<Inner>),
}
pub(crate) struct Inner {
    owner: usize,
    device: DeviceId,
    next: AtomicU64,
    trace: Mutex<Vec<Record>>,
}
pub(crate) struct PendingSample {
    session: Arc<Inner>,
    sequence: u64,
    retained: Option<Arc<dyn Send + Sync>>,
    state: Completion,
}
impl ProfilingSession {
    pub(crate) fn disabled() -> Self {
        Self::Disabled
    }
    pub(crate) fn enabled(owner: usize, device: DeviceId) -> Self {
        Self::Enabled(Arc::new(Inner {
            owner,
            device,
            next: AtomicU64::new(0),
            trace: Mutex::new(vec![]),
        }))
    }
    pub(crate) fn submit(
        &self,
        mut metadata: Metadata,
        retained: Arc<dyn Send + Sync>,
    ) -> Result<Option<PendingSample>, ()> {
        let Self::Enabled(inner) = self else {
            return Ok(None);
        };
        if metadata.owner != inner.owner || metadata.device != inner.device {
            return Err(());
        };
        let sequence = inner.next.fetch_add(1, Ordering::AcqRel);
        inner.trace.lock().unwrap().push(Record {
            sequence,
            metadata: {
                metadata.owner = inner.owner;
                metadata
            },
            completion: Completion::Pending,
        });
        Ok(Some(PendingSample {
            session: inner.clone(),
            sequence,
            retained: Some(retained),
            state: Completion::Pending,
        }))
    }
    pub(crate) fn records(&self) -> Vec<Record> {
        match self {
            Self::Disabled => vec![],
            Self::Enabled(x) => x.trace.lock().unwrap().clone(),
        }
    }
}
impl PendingSample {
    fn transition(&mut self, state: Completion) {
        self.state = state.clone();
        let mut trace = self.session.trace.lock().unwrap();
        trace[self.sequence as usize].completion = state;
    }
    pub(crate) fn ready(&mut self, duration_ns: Option<u64>) {
        self.transition(Completion::Ready(duration_ns));
    }
    pub(crate) fn collect(&mut self) {
        self.transition(Completion::Collected);
        self.retained.take();
    }
    pub(crate) fn fail(&mut self, error: String) {
        self.transition(Completion::Failed(error));
        self.retained.take();
    }
}
impl Drop for PendingSample {
    fn drop(&mut self) {
        if matches!(self.state, Completion::Pending | Completion::Ready(_)) {
            self.transition(Completion::Abandoned);
            self.retained.take();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    fn meta(owner: usize, stream: usize) -> Metadata {
        Metadata {
            kind: OperationKind::HtoD,
            name: "copy".into(),
            owner,
            device: DeviceId(0),
            stream,
            bytes: Some(4),
            geometry: None,
            source_key: None,
        }
    }
    #[test]
    fn disabled_is_empty_and_enabled_orders_samples() {
        let off = ProfilingSession::disabled();
        assert!(off.submit(meta(1, 1), Arc::new(())).unwrap().is_none());
        let on = ProfilingSession::enabled(1, DeviceId(0));
        let a = on.submit(meta(1, 2), Arc::new(())).unwrap().unwrap();
        let b = on.submit(meta(1, 3), Arc::new(())).unwrap().unwrap();
        assert_eq!(
            on.records().iter().map(|x| x.sequence).collect::<Vec<_>>(),
            vec![0, 1]
        );
        drop(a);
        drop(b);
    }
    #[test]
    fn transitions_failure_and_abandonment_release() {
        let on = ProfilingSession::enabled(1, DeviceId(0));
        let held = Arc::new(AtomicBool::new(false));
        let sentinel = Arc::new(Sentinel(held.clone()));
        let mut sample = on.submit(meta(1, 1), sentinel).unwrap().unwrap();
        sample.ready(Some(12));
        sample.collect();
        assert_eq!(on.records()[0].completion, Completion::Collected);
        let doomed = on.submit(meta(1, 1), Arc::new(())).unwrap().unwrap();
        drop(doomed);
        assert_eq!(on.records()[1].completion, Completion::Abandoned);
        assert!(held.load(Ordering::Acquire));
    }
    struct Sentinel(Arc<AtomicBool>);
    impl Drop for Sentinel {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }
    #[test]
    fn owner_mismatch_is_rejected() {
        assert!(
            ProfilingSession::enabled(1, DeviceId(0))
                .submit(meta(2, 1), Arc::new(()))
                .is_err()
        );
    }
    fn assert_send_sync<T: Send + Sync>() {}
    #[test]
    fn session_is_send_sync() {
        assert_send_sync::<ProfilingSession>();
    }
}
