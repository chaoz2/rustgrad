#![allow(dead_code)] // crate-private profiling foundation awaits submission integration.
//! Backend-neutral ownership core plus an isolated CUDA event timing adapter.
//!
//! The recorder lock protects only trace mutation. Completion is supplied by a
//! future adapter outside that lock; samples retain an `Arc` sentinel until
//! collected, failed, or abandoned. Records are ordered by submission sequence.

use crate::{CudaError, DeviceId, Event, PrimaryContext, Stream};
use std::fmt;
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
    pub(crate) fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled(_))
    }
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
    pub(crate) fn abandon(&mut self) {
        if matches!(self.state, Completion::Pending | Completion::Ready(_)) {
            self.transition(Completion::Abandoned);
            self.retained.take();
        }
    }
}
impl Drop for PendingSample {
    fn drop(&mut self) {
        self.abandon();
    }
}

/// Structured failure from the CUDA-only timing boundary. The neutral recorder
/// stores a diagnostic string, but callers retain the original CUDA error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TimingError {
    Cuda(CudaError),
    OwnerMismatch,
    StreamMismatch,
    EndNotRecorded,
    EndAlreadyRecorded,
    InvalidElapsed,
}
impl From<CudaError> for TimingError {
    fn from(error: CudaError) -> Self {
        Self::Cuda(error)
    }
}
impl fmt::Display for TimingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => write!(f, "CUDA timing error: {error}"),
            Self::OwnerMismatch => write!(f, "profiling primary owner does not match the stream"),
            Self::StreamMismatch => write!(f, "timing pair used with a different stream"),
            Self::EndNotRecorded => write!(f, "timing end event has not been recorded"),
            Self::EndAlreadyRecorded => write!(f, "timing end event has already been recorded"),
            Self::InvalidElapsed => write!(f, "CUDA event elapsed time is invalid"),
        }
    }
}

/// Start/end events bound to one retained primary context and stream. Both use
/// `CU_EVENT_DEFAULT` through `PrimaryContext::timing_event`, deliberately
/// leaving CUDA elapsed timing enabled. No recorder lock is held while this
/// adapter creates, records, queries, synchronizes, or destroys its events.
struct TimingPair<'a> {
    stream: &'a Stream,
    start: Event,
    end: Event,
    end_recorded: bool,
}
impl<'a> TimingPair<'a> {
    fn new(primary: &PrimaryContext, stream: &'a Stream) -> Result<Self, TimingError> {
        if !stream.belongs_to_primary(primary) {
            return Err(TimingError::OwnerMismatch);
        }
        let start = primary.timing_event()?;
        let end = primary.timing_event()?;
        start.record(stream)?;
        Ok(Self {
            stream,
            start,
            end,
            end_recorded: false,
        })
    }
    fn record_end(&mut self, stream: &Stream) -> Result<(), TimingError> {
        if !self.stream.same_stream(stream) {
            return Err(TimingError::StreamMismatch);
        }
        if self.end_recorded {
            return Err(TimingError::EndAlreadyRecorded);
        }
        self.end.record(stream)?;
        self.end_recorded = true;
        Ok(())
    }
    fn query(&self) -> Result<bool, TimingError> {
        if !self.end_recorded {
            return Err(TimingError::EndNotRecorded);
        }
        Ok(self.end.query()?)
    }
    fn wait(&self) -> Result<(), TimingError> {
        if !self.end_recorded {
            return Err(TimingError::EndNotRecorded);
        }
        Ok(self.end.synchronize()?)
    }
    fn elapsed_ns(&self) -> Result<u64, TimingError> {
        let milliseconds = Event::elapsed_ms(&self.start, &self.end)?;
        let nanoseconds = milliseconds * 1_000_000.0;
        if !nanoseconds.is_finite() || nanoseconds < 0.0 || nanoseconds > u64::MAX as f32 {
            return Err(TimingError::InvalidElapsed);
        }
        Ok(nanoseconds as u64)
    }
}

/// Completion hook for a recorder sample. It is deliberately not connected to
/// kernel or copy submission yet; callers record the end event after their own
/// submission, then use `query`, `wait`, or `collect` explicitly.
pub(crate) struct TimedSample<'a> {
    timing: Option<TimingPair<'a>>,
    pending: PendingSample,
    ready_ns: Option<u64>,
}
impl<'a> TimedSample<'a> {
    pub(crate) fn begin(
        session: &ProfilingSession,
        metadata: Metadata,
        primary: &PrimaryContext,
        stream: &'a Stream,
        retained: Arc<dyn Send + Sync>,
    ) -> Result<Option<Self>, TimingError> {
        if !stream.belongs_to_primary(primary) {
            return Err(TimingError::OwnerMismatch);
        }
        let Some(mut pending) = session
            .submit(metadata, retained)
            .map_err(|()| TimingError::OwnerMismatch)?
        else {
            return Ok(None);
        };
        match TimingPair::new(primary, stream) {
            Ok(timing) => Ok(Some(Self {
                timing: Some(timing),
                pending,
                ready_ns: None,
            })),
            Err(error) => {
                pending.fail(error.to_string());
                Err(error)
            }
        }
    }
    pub(crate) fn record_end(&mut self, stream: &Stream) -> Result<(), TimingError> {
        match self.timing_mut().record_end(stream) {
            Ok(()) => Ok(()),
            Err(error @ TimingError::Cuda(_)) => self.fail(error),
            Err(error) => Err(error),
        }
    }
    pub(crate) fn query(&mut self) -> Result<Option<u64>, TimingError> {
        if let Some(duration) = self.ready_ns {
            return Ok(Some(duration));
        }
        if !self.timing().query().or_else(|error| self.fail(error))? {
            return Ok(None);
        }
        self.finish_elapsed().map(Some)
    }
    pub(crate) fn wait(&mut self) -> Result<u64, TimingError> {
        if let Some(duration) = self.ready_ns {
            return Ok(duration);
        }
        self.timing().wait().or_else(|error| self.fail(error))?;
        self.finish_elapsed()
    }
    pub(crate) fn collect(mut self) -> Result<u64, TimingError> {
        let duration = self.wait()?;
        self.pending.collect();
        self.timing.take();
        Ok(duration)
    }
    pub(crate) fn fail_due_to(&mut self, error: TimingError) {
        let _ = self.fail::<()>(error);
    }
    fn timing(&self) -> &TimingPair<'a> {
        self.timing
            .as_ref()
            .expect("timing pair retained until completion")
    }
    fn timing_mut(&mut self) -> &mut TimingPair<'a> {
        self.timing
            .as_mut()
            .expect("timing pair retained until completion")
    }
    fn finish_elapsed(&mut self) -> Result<u64, TimingError> {
        match self.timing().elapsed_ns() {
            Ok(duration) => {
                self.pending.ready(Some(duration));
                self.ready_ns = Some(duration);
                Ok(duration)
            }
            Err(error) => self.fail(error),
        }
    }
    fn fail<T>(&mut self, error: TimingError) -> Result<T, TimingError> {
        self.pending.fail(error.to_string());
        self.timing.take();
        Err(error)
    }
}
impl Drop for TimedSample<'_> {
    fn drop(&mut self) {
        self.pending.abandon();
        self.timing.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Driver, cuda::tests::Mock};
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

    fn timing_fixture() -> (Arc<Mock>, PrimaryContext, Stream, ProfilingSession) {
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let primary = driver
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let stream = primary.stream().unwrap();
        let session = ProfilingSession::enabled(primary.identity(), primary.device());
        (mock, primary, stream, session)
    }

    #[test]
    fn timing_records_start_then_end_and_queries_without_syncing() {
        let (mock, primary, stream, session) = timing_fixture();
        mock.set_elapsed_support(true);
        let mut sample = TimedSample::begin(
            &session,
            meta(primary.identity(), 1),
            &primary,
            &stream,
            Arc::new(()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            mock.calls()
                .into_iter()
                .filter(|call| *call == "event_create" || *call == "event_record")
                .collect::<Vec<_>>(),
            vec!["event_create", "event_create", "event_record"]
        );
        sample.record_end(&stream).unwrap();
        assert_eq!(sample.query().unwrap(), None);
        assert!(!mock.calls().contains(&"event_sync"));
        mock.set_event_ready(true);
        assert_eq!(sample.query().unwrap(), Some(1_500_000));
        assert_eq!(
            session.records()[0].completion,
            Completion::Ready(Some(1_500_000))
        );
        assert_eq!(sample.collect().unwrap(), 1_500_000);
        assert_eq!(session.records()[0].completion, Completion::Collected);
    }

    #[test]
    fn timing_wait_synchronizes_end_before_collecting() {
        let (mock, primary, stream, session) = timing_fixture();
        mock.set_elapsed_support(true);
        let mut sample = TimedSample::begin(
            &session,
            meta(primary.identity(), 1),
            &primary,
            &stream,
            Arc::new(()),
        )
        .unwrap()
        .unwrap();
        sample.record_end(&stream).unwrap();
        assert_eq!(sample.wait().unwrap(), 1_500_000);
        let calls = mock.calls();
        assert!(
            calls.iter().position(|x| *x == "event_sync").unwrap()
                < calls.iter().position(|x| *x == "event_elapsed").unwrap()
        );
    }

    #[test]
    fn timing_reports_missing_symbol_driver_failure_and_invalid_elapsed() {
        let (mock, primary, stream, session) = timing_fixture();
        let mut missing = TimedSample::begin(
            &session,
            meta(primary.identity(), 1),
            &primary,
            &stream,
            Arc::new(()),
        )
        .unwrap()
        .unwrap();
        missing.record_end(&stream).unwrap();
        mock.set_event_ready(true);
        assert!(matches!(
            missing.query(),
            Err(TimingError::Cuda(CudaError::MissingSymbol(
                "cuEventElapsedTime"
            )))
        ));

        mock.set_elapsed_support(true);
        mock.set_elapsed_result(2);
        let mut failed = TimedSample::begin(
            &session,
            meta(primary.identity(), 1),
            &primary,
            &stream,
            Arc::new(()),
        )
        .unwrap()
        .unwrap();
        failed.record_end(&stream).unwrap();
        assert!(matches!(
            failed.query(),
            Err(TimingError::Cuda(CudaError::Driver { code: 2, .. }))
        ));

        mock.set_elapsed_result(0);
        for invalid in [f32::NAN, -1.0] {
            mock.set_elapsed_millis(invalid);
            let mut sample = TimedSample::begin(
                &session,
                meta(primary.identity(), 1),
                &primary,
                &stream,
                Arc::new(()),
            )
            .unwrap()
            .unwrap();
            sample.record_end(&stream).unwrap();
            assert_eq!(sample.query(), Err(TimingError::InvalidElapsed));
        }
        assert!(
            session
                .records()
                .iter()
                .all(|record| matches!(record.completion, Completion::Failed(_)))
        );
    }

    #[test]
    fn timing_rejects_owner_and_stream_mismatches_before_event_calls() {
        let (mock, primary, stream, session) = timing_fixture();
        let other = primary.stream().unwrap();
        let foreign = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let calls_before = mock.calls().len();
        assert!(matches!(
            TimedSample::begin(
                &session,
                meta(primary.identity(), 1),
                &foreign,
                &stream,
                Arc::new(())
            ),
            Err(TimingError::OwnerMismatch)
        ));
        assert_eq!(mock.calls().len(), calls_before);

        let mut sample = TimedSample::begin(
            &session,
            meta(primary.identity(), 1),
            &primary,
            &stream,
            Arc::new(()),
        )
        .unwrap()
        .unwrap();
        let calls_before = mock.calls().len();
        assert_eq!(sample.record_end(&other), Err(TimingError::StreamMismatch));
        assert_eq!(mock.calls().len(), calls_before);
    }

    #[test]
    fn timing_abandonment_releases_retention_and_events_before_primary_release() {
        let (mock, primary, stream, session) = timing_fixture();
        let released = Arc::new(AtomicBool::new(false));
        let sentinel = Arc::new(Sentinel(released.clone()));
        let sample = TimedSample::begin(
            &session,
            meta(primary.identity(), 1),
            &primary,
            &stream,
            sentinel,
        )
        .unwrap()
        .unwrap();
        drop(sample);
        assert_eq!(session.records()[0].completion, Completion::Abandoned);
        assert!(released.load(Ordering::Acquire));
        drop(stream);
        drop(primary);
        let calls = mock.calls();
        assert!(
            calls.iter().position(|x| *x == "event_destroy").unwrap()
                < calls.iter().position(|x| *x == "primary_release").unwrap()
        );
    }
}
