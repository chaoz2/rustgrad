//! Typed values exchanged across the compiled-training runtime boundary.
//!
//! This module owns batch declarations, detached evaluation and training
//! results, checkpoint values, and partial-window outcomes. Graph lowering,
//! recurrent transactions, and backend execution remain in the parent module.

use super::observation::CompiledTrainingObservationValue;
use super::*;

/// Detached result of one successfully committed compiled training step.
#[derive(Clone, Debug)]
pub struct CompiledTrainingStepResult {
    pub(super) loss: TensorData,
    pub(super) loss_aggregation_weight: u64,
    pub(super) outputs: BTreeMap<String, TensorData>,
    pub(super) step: u64,
    pub(super) capture_identity: u64,
    pub(super) observations: Vec<CompiledTrainingObservationValue>,
}

/// Detached outputs from one read-only evaluation of the live compiled
/// parameter frontier.
#[derive(Clone, Debug)]
pub struct CompiledEvaluationResult {
    pub(super) loss: TensorData,
    pub(super) outputs: BTreeMap<String, TensorData>,
    pub(super) loss_weight: u64,
    pub(super) capture_identity: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CompiledInputPolicy {
    External,
    HostToken,
}

/// One fixed external input declaration supplied by a [`CompiledInputBatch`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledInputSpec {
    pub(super) name: &'static str,
    pub(super) shape: &'static [usize],
    pub(super) dtype: DType,
    pub(super) policy: CompiledInputPolicy,
}

impl CompiledInputSpec {
    /// Declares one ordinary fixed-shape external input.
    pub const fn new(name: &'static str, shape: &'static [usize], dtype: DType) -> Self {
        Self {
            name,
            shape,
            dtype,
            policy: CompiledInputPolicy::External,
        }
    }

    /// Declares one fixed rank-two I32 token input eligible for the compiled
    /// training capture's authenticated host-index policy.
    pub const fn host_token(name: &'static str, shape: &'static [usize]) -> Self {
        Self {
            name,
            shape,
            dtype: DType::I32,
            policy: CompiledInputPolicy::HostToken,
        }
    }

    pub const fn name(self) -> &'static str {
        self.name
    }

    pub const fn shape(self) -> &'static [usize] {
        self.shape
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }
}

/// Converts one workload-owned batch into the exact named external inputs of a
/// compiled training or evaluation capture.
///
/// The declarative schema is shared by compilation and replay. Implementing
/// this trait once for a domain batch keeps names, shapes, dtypes, and host-token
/// policy at the workload boundary. Recurrent parameter, optimizer, and
/// workload state remain runtime-owned, and
/// [`CompiledTrainingRuntime::step_batch`] supplies the learning rate through
/// its separate typed scalar argument.
pub trait CompiledInputBatch {
    fn schema() -> &'static [CompiledInputSpec];

    fn into_compiled_inputs(self) -> Result<BTreeMap<String, TensorData>>;
}

impl CompiledEvaluationResult {
    pub fn loss(&self) -> &TensorData {
        &self.loss
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        &self.outputs
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.outputs.get(name)
    }

    /// Exact aggregation weight for this normalized evaluation loss.
    ///
    /// Ordinary scalar objectives report one. Compiler-owned token-mean
    /// objectives report the validated number of live tokens in this batch.
    pub fn loss_weight(&self) -> u64 {
        self.loss_weight
    }

    pub fn capture_identity(&self) -> u64 {
        self.capture_identity
    }
}

/// Backend-neutral view of one successful read-only compiled evaluation.
pub trait CompiledEvaluation {
    fn loss(&self) -> &TensorData;

    fn outputs(&self) -> &BTreeMap<String, TensorData>;

    fn output(&self, name: &str) -> Option<&TensorData> {
        self.outputs().get(name)
    }

    /// Exact aggregation weight for this normalized evaluation loss.
    /// Historical and custom scalar evaluation results retain weight one.
    fn loss_weight(&self) -> u64 {
        1
    }

    fn capture_identity(&self) -> u64;
}

impl CompiledEvaluation for CompiledEvaluationResult {
    fn loss(&self) -> &TensorData {
        self.loss()
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.outputs()
    }

    fn loss_weight(&self) -> u64 {
        self.loss_weight()
    }

    fn capture_identity(&self) -> u64 {
        self.capture_identity()
    }
}

/// Successful strict-Metal evaluation plus its exact stateless run evidence.
pub struct MetalCompiledEvaluationResult {
    pub(super) inner: CompiledEvaluationResult,
    pub(super) report: MetalDeviceRunReport,
}

impl MetalCompiledEvaluationResult {
    pub fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.inner.output(name)
    }

    pub fn loss_weight(&self) -> u64 {
        self.inner.loss_weight()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn report(&self) -> &MetalDeviceRunReport {
        &self.report
    }
}

impl CompiledEvaluation for MetalCompiledEvaluationResult {
    fn loss(&self) -> &TensorData {
        self.loss()
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.outputs()
    }

    fn loss_weight(&self) -> u64 {
        self.loss_weight()
    }

    fn capture_identity(&self) -> u64 {
        self.capture_identity()
    }
}

/// Read-only strict-native CPU evaluation plus its replay evidence.
pub struct NativeCpuCompiledEvaluationResult {
    pub(super) inner: CompiledEvaluationResult,
    pub(super) report: NativeCpuRunReport,
}

impl NativeCpuCompiledEvaluationResult {
    pub fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.inner.output(name)
    }

    pub fn loss_weight(&self) -> u64 {
        self.inner.loss_weight()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn report(&self) -> &NativeCpuRunReport {
        &self.report
    }
}

impl CompiledEvaluation for NativeCpuCompiledEvaluationResult {
    fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    fn loss_weight(&self) -> u64 {
        self.inner.loss_weight()
    }

    fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }
}

/// Optional read-only evaluation capability for a compiled training session.
/// Evaluation observes the current parameter frontier without advancing or
/// mutating training, optimizer, accumulation, or workload state.
pub trait CompiledEvaluationRuntime {
    type Evaluation: CompiledEvaluation;

    fn evaluate(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation>;

    /// Evaluates a domain batch after converting only its external bindings.
    fn evaluate_batch<B>(&mut self, batch: B) -> Result<Self::Evaluation>
    where
        Self: Sized,
        B: CompiledInputBatch,
    {
        self.evaluate(batch.into_compiled_inputs()?)
    }

    fn evaluation_capture_identity(&self) -> Option<u64>;
}

impl CompiledTrainingStepResult {
    pub fn loss(&self) -> &TensorData {
        &self.loss
    }

    /// Exact weight of this normalized loss when aggregating across batches.
    pub fn loss_aggregation_weight(&self) -> u64 {
        self.loss_aggregation_weight
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        &self.outputs
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.outputs.get(name)
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn capture_identity(&self) -> u64 {
        self.capture_identity
    }
}

pub type CompiledMomentumSgdStepResult = CompiledTrainingStepResult;

/// Backend- and optimizer-neutral view of one committed compiled training step.
///
/// Concrete optimizer results may expose additional progress, while device
/// results may retain execution reports. Generic training loops can still
/// consume loss plus its exact aggregation weight, named outputs, replay
/// progress, and capture identity without selecting either concern through an
/// enum.
pub trait CompiledTrainingStep {
    fn loss(&self) -> &TensorData;

    /// Exact weight of this normalized loss when aggregating across batches.
    ///
    /// Ordinary scalar objectives use one. Compiler-owned token-mean
    /// objectives override this with their validated number of contributing
    /// tokens, independently of the optimizer that consumed the objective.
    fn loss_aggregation_weight(&self) -> u64 {
        1
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData>;

    fn output(&self, name: &str) -> Option<&TensorData> {
        self.outputs().get(name)
    }

    fn step(&self) -> u64;

    fn capture_identity(&self) -> u64;
}

impl CompiledTrainingStep for CompiledTrainingStepResult {
    fn loss(&self) -> &TensorData {
        CompiledTrainingStepResult::loss(self)
    }

    fn loss_aggregation_weight(&self) -> u64 {
        CompiledTrainingStepResult::loss_aggregation_weight(self)
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        CompiledTrainingStepResult::outputs(self)
    }

    fn step(&self) -> u64 {
        CompiledTrainingStepResult::step(self)
    }

    fn capture_identity(&self) -> u64 {
        CompiledTrainingStepResult::capture_identity(self)
    }
}

/// Optimizer-neutral accumulation-window progress after one committed replay.
pub trait CompiledTrainingWindowStep: CompiledTrainingStep {
    /// Microbatches retained toward the next complete gradient window.
    fn pending_microbatch_count(&self) -> u64;

    /// Whether this replay closed its gradient window.
    fn did_close_gradient_window(&self) -> bool {
        self.pending_microbatch_count() == 0
    }
}

/// One committed replay of a compiled AdamW program.
///
/// `step` counts microbatch replays. `optimizer_step` advances only when the
/// configured accumulation window commits, and `accumulation_index` reports
/// the number of retained microbatches toward the next update. `loss_weight`
/// is one for ordinary scalar-loss programs and the exact valid-token count for
/// compiler-owned token-mean programs.
#[derive(Clone, Debug)]
pub struct CompiledAdamWStepResult {
    pub(super) inner: CompiledTrainingStepResult,
    pub(super) optimizer_step: u64,
    pub(super) accumulation_index: u64,
    pub(super) clip_report: Option<CompiledAdamWClipReport>,
    pub(super) window_loss_report: Option<CompiledAdamWWindowLossReport>,
}

/// Completed-window evidence for compiled global gradient clipping.
///
/// Both values are exact F32 results from the captured graph. The scale is
/// one when clipping is disabled or the pre-clip norm is at or below the
/// configured limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledAdamWClipReport {
    pub(super) pre_clip_global_norm_bits: u32,
    pub(super) applied_scale_bits: u32,
}

/// Exact aggregate loss for one committed AdamW accumulation window.
///
/// `mean_loss` is the captured F32 recurrence result. `loss_weight` is the
/// microbatch count for ordinary scalar objectives and the valid-token count
/// for compiler-owned token means. Accumulation-only steps, discarded windows,
/// and empty flushes produce no report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledAdamWWindowLossReport {
    pub(super) mean_loss_bits: u32,
    pub(super) loss_weight: u64,
    pub(super) microbatch_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CompiledAdamWWindowLossValue {
    pub(super) mean_loss_bits: u32,
    pub(super) loss_weight: u64,
}

impl CompiledAdamWWindowLossReport {
    pub(super) fn new(value: CompiledAdamWWindowLossValue, microbatch_count: u64) -> Self {
        Self {
            mean_loss_bits: value.mean_loss_bits,
            loss_weight: value.loss_weight,
            microbatch_count,
        }
    }

    /// Exact F32 mean produced by the captured recurrence.
    pub fn mean_loss(&self) -> f32 {
        f32::from_bits(self.mean_loss_bits)
    }

    /// Sum of microbatch or validated-token weights in this window.
    pub fn loss_weight(&self) -> u64 {
        self.loss_weight
    }

    /// Number of replays committed by this window.
    pub fn microbatch_count(&self) -> u64 {
        self.microbatch_count
    }

    /// Whether the captured aggregate mean is finite.
    pub fn is_finite(&self) -> bool {
        self.mean_loss().is_finite()
    }
}

impl CompiledAdamWClipReport {
    pub(super) fn new(pre_clip_global_norm: f32, applied_scale: f32) -> Self {
        Self {
            pre_clip_global_norm_bits: pre_clip_global_norm.to_bits(),
            applied_scale_bits: applied_scale.to_bits(),
        }
    }

    pub fn pre_clip_global_norm(&self) -> f32 {
        f32::from_bits(self.pre_clip_global_norm_bits)
    }

    pub fn applied_scale(&self) -> f32 {
        f32::from_bits(self.applied_scale_bits)
    }

    /// Whether both captured values are finite.
    pub fn is_finite(&self) -> bool {
        self.pre_clip_global_norm().is_finite() && self.applied_scale().is_finite()
    }

    /// Whether clipping reduced this completed window's gradient.
    ///
    /// A non-finite report has no meaningful clipping outcome and returns
    /// `None` while preserving its exact captured F32 bits for diagnosis.
    pub fn did_clip(&self) -> Option<bool> {
        self.is_finite().then(|| self.applied_scale() < 1.0)
    }
}

impl CompiledAdamWStepResult {
    pub fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.inner.output(name)
    }

    pub fn step(&self) -> u64 {
        self.inner.step()
    }

    pub fn optimizer_step(&self) -> u64 {
        self.optimizer_step
    }

    pub fn accumulation_index(&self) -> u64 {
        self.accumulation_index
    }

    /// Exact weight of this step's normalized loss in an aggregate mean.
    ///
    /// Ordinary scalar-loss programs use one. Compiler-owned token-mean
    /// programs use the validated number of non-padding tokens in this replay.
    pub fn loss_weight(&self) -> u64 {
        self.inner.loss_aggregation_weight()
    }

    /// Completed-window clipping evidence when reporting was requested.
    pub fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        self.clip_report.as_ref()
    }

    /// Aggregate loss for the full window committed by this replay.
    pub fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        self.window_loss_report.as_ref()
    }

    pub fn did_update(&self) -> bool {
        self.accumulation_index == 0
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }
}

/// AdamW-specific progress for one successfully committed training step.
///
/// Concrete runtimes may retain additional execution evidence. For example,
/// [`MetalCompiledAdamWStepResult`] also exposes its exact device run report.
pub trait CompiledAdamWStep: CompiledTrainingStep {
    fn optimizer_step(&self) -> u64;

    fn accumulation_index(&self) -> u64;

    /// Exact weight of this step's loss in the optimizer's aggregate mean.
    ///
    /// The default preserves ordinary scalar-loss and existing external
    /// implementations. Token-mean CPU results override it with the validated
    /// number of non-padding tokens in the replay.
    fn loss_weight(&self) -> u64 {
        1
    }

    /// Completed-window clipping evidence. Existing implementations and
    /// programs compiled without the opt-in report return `None`.
    fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        None
    }

    /// Aggregate loss for a full window committed by this replay, when enabled.
    fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        None
    }

    fn did_update(&self) -> bool {
        self.accumulation_index() == 0
    }
}

impl<S> CompiledTrainingWindowStep for S
where
    S: CompiledAdamWStep + ?Sized,
{
    fn pending_microbatch_count(&self) -> u64 {
        CompiledAdamWStep::accumulation_index(self)
    }
}

impl CompiledTrainingStep for CompiledAdamWStepResult {
    fn loss(&self) -> &TensorData {
        CompiledAdamWStepResult::loss(self)
    }

    fn loss_aggregation_weight(&self) -> u64 {
        CompiledAdamWStepResult::loss_weight(self)
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        CompiledAdamWStepResult::outputs(self)
    }

    fn step(&self) -> u64 {
        CompiledAdamWStepResult::step(self)
    }

    fn capture_identity(&self) -> u64 {
        CompiledAdamWStepResult::capture_identity(self)
    }
}

impl CompiledAdamWStep for CompiledAdamWStepResult {
    fn optimizer_step(&self) -> u64 {
        CompiledAdamWStepResult::optimizer_step(self)
    }

    fn accumulation_index(&self) -> u64 {
        CompiledAdamWStepResult::accumulation_index(self)
    }

    fn loss_weight(&self) -> u64 {
        CompiledAdamWStepResult::loss_weight(self)
    }

    fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        CompiledAdamWStepResult::clip_report(self)
    }

    fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        CompiledAdamWStepResult::window_loss_report(self)
    }
}

/// One committed strict-native CPU training step and its replay evidence.
///
/// This result is intentionally optimizer-neutral. Optimizers with additional
/// progress or policy reports expose a more specific result type.
pub struct NativeCpuCompiledTrainingStepResult {
    pub(super) inner: CompiledTrainingStepResult,
    pub(super) report: NativeCpuRunReport,
}

impl NativeCpuCompiledTrainingStepResult {
    pub fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    pub fn loss_aggregation_weight(&self) -> u64 {
        self.inner.loss_aggregation_weight()
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.inner.output(name)
    }

    pub fn step(&self) -> u64 {
        self.inner.step()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn report(&self) -> &NativeCpuRunReport {
        &self.report
    }
}

impl CompiledTrainingStep for NativeCpuCompiledTrainingStepResult {
    fn loss(&self) -> &TensorData {
        NativeCpuCompiledTrainingStepResult::loss(self)
    }

    fn loss_aggregation_weight(&self) -> u64 {
        NativeCpuCompiledTrainingStepResult::loss_aggregation_weight(self)
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        NativeCpuCompiledTrainingStepResult::outputs(self)
    }

    fn step(&self) -> u64 {
        NativeCpuCompiledTrainingStepResult::step(self)
    }

    fn capture_identity(&self) -> u64 {
        NativeCpuCompiledTrainingStepResult::capture_identity(self)
    }
}

/// Strict-native CPU momentum-SGD step result.
pub type NativeCpuCompiledMomentumSgdStepResult = NativeCpuCompiledTrainingStepResult;

/// One committed strict-native CPU AdamW step and its replay evidence.
pub struct NativeCpuCompiledAdamWStepResult {
    pub(super) inner: CompiledAdamWStepResult,
    pub(super) report: NativeCpuRunReport,
}

impl NativeCpuCompiledAdamWStepResult {
    pub fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    pub fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.inner.output(name)
    }

    pub fn step(&self) -> u64 {
        self.inner.step()
    }

    pub fn optimizer_step(&self) -> u64 {
        self.inner.optimizer_step()
    }

    pub fn accumulation_index(&self) -> u64 {
        self.inner.accumulation_index()
    }

    pub fn loss_weight(&self) -> u64 {
        self.inner.loss_weight()
    }

    pub fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        self.inner.clip_report()
    }

    /// Aggregate loss for the full window committed by this replay, when enabled.
    pub fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        self.inner.window_loss_report()
    }

    pub fn did_update(&self) -> bool {
        self.inner.did_update()
    }

    pub fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }

    pub fn report(&self) -> &NativeCpuRunReport {
        &self.report
    }
}

impl CompiledTrainingStep for NativeCpuCompiledAdamWStepResult {
    fn loss(&self) -> &TensorData {
        self.inner.loss()
    }

    fn loss_aggregation_weight(&self) -> u64 {
        self.inner.loss_weight()
    }

    fn outputs(&self) -> &BTreeMap<String, TensorData> {
        self.inner.outputs()
    }

    fn step(&self) -> u64 {
        self.inner.step()
    }

    fn capture_identity(&self) -> u64 {
        self.inner.capture_identity()
    }
}

impl CompiledAdamWStep for NativeCpuCompiledAdamWStepResult {
    fn optimizer_step(&self) -> u64 {
        self.inner.optimizer_step()
    }

    fn accumulation_index(&self) -> u64 {
        self.inner.accumulation_index()
    }

    fn loss_weight(&self) -> u64 {
        self.inner.loss_weight()
    }

    fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        self.inner.clip_report()
    }

    /// Aggregate loss for the full window committed by this replay, when enabled.
    fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        self.inner.window_loss_report()
    }
}

/// Outcome of explicitly discarding a compiled partial gradient window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledTrainingWindowReset {
    pub(super) discarded_microbatches: u64,
}

impl CompiledTrainingWindowReset {
    /// Number of previously retained microbatches removed by this call.
    pub fn discarded_microbatches(&self) -> u64 {
        self.discarded_microbatches
    }

    /// Whether this call published a new recurrent frontier.
    pub fn did_discard(&self) -> bool {
        self.discarded_microbatches != 0
    }
}

/// AdamW compatibility name for [`CompiledTrainingWindowReset`].
pub type CompiledAdamWZeroGradResult = CompiledTrainingWindowReset;

/// Optimizer-neutral outcome of committing a retained partial gradient window.
///
/// Concrete optimizer and backend results may expose additional update or
/// execution evidence. A generic training loop only needs to know how many
/// retained microbatches were committed and whether the call published a new
/// recurrent frontier. Empty windows are exact no-ops.
pub trait CompiledTrainingWindowCommit {
    /// Number of retained microbatches committed by this call.
    fn committed_microbatches(&self) -> u64;

    /// Whether this call published a new recurrent frontier.
    fn did_commit_window(&self) -> bool {
        self.committed_microbatches() != 0
    }
}

/// Outcome of explicitly committing a non-full compiled AdamW window on CPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledAdamWFlushResult {
    pub(super) flushed_microbatches: u64,
    pub(super) optimizer_step: u64,
    pub(super) clip_report: Option<CompiledAdamWClipReport>,
    pub(super) window_loss_report: Option<CompiledAdamWWindowLossReport>,
}

impl CompiledAdamWFlushResult {
    /// Number of retained microbatches averaged by this call.
    pub fn flushed_microbatches(&self) -> u64 {
        self.flushed_microbatches
    }

    /// Whether this call published an AdamW update.
    pub fn did_update(&self) -> bool {
        self.flushed_microbatches != 0
    }

    /// Optimizer step after the call. Empty windows preserve the prior step.
    pub fn optimizer_step(&self) -> u64 {
        self.optimizer_step
    }

    /// Clipping evidence for a committed nonempty partial window.
    pub fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        self.clip_report.as_ref()
    }

    pub fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        self.window_loss_report.as_ref()
    }
}

/// Optimizer progress produced by one optional partial-window flush.
pub trait CompiledAdamWFlush {
    fn flushed_microbatches(&self) -> u64;

    fn did_update(&self) -> bool;

    fn optimizer_step(&self) -> u64;

    fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        None
    }

    fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        None
    }
}

impl<T> CompiledTrainingWindowCommit for T
where
    T: CompiledAdamWFlush + ?Sized,
{
    fn committed_microbatches(&self) -> u64 {
        self.flushed_microbatches()
    }
}

impl CompiledAdamWFlush for CompiledAdamWFlushResult {
    fn flushed_microbatches(&self) -> u64 {
        CompiledAdamWFlushResult::flushed_microbatches(self)
    }

    fn did_update(&self) -> bool {
        CompiledAdamWFlushResult::did_update(self)
    }

    fn optimizer_step(&self) -> u64 {
        CompiledAdamWFlushResult::optimizer_step(self)
    }

    fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        CompiledAdamWFlushResult::clip_report(self)
    }

    fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        CompiledAdamWFlushResult::window_loss_report(self)
    }
}

/// One strict-native CPU partial-window flush. Empty windows execute no
/// native program and therefore carry no run report.
pub struct NativeCpuCompiledAdamWFlushResult {
    pub(super) inner: CompiledAdamWFlushResult,
    pub(super) report: Option<NativeCpuRunReport>,
}

impl NativeCpuCompiledAdamWFlushResult {
    pub fn flushed_microbatches(&self) -> u64 {
        self.inner.flushed_microbatches()
    }

    pub fn did_update(&self) -> bool {
        self.inner.did_update()
    }

    pub fn optimizer_step(&self) -> u64 {
        self.inner.optimizer_step()
    }

    pub fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        self.inner.clip_report()
    }

    pub fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        self.inner.window_loss_report()
    }

    pub fn report(&self) -> Option<&NativeCpuRunReport> {
        self.report.as_ref()
    }
}

impl CompiledAdamWFlush for NativeCpuCompiledAdamWFlushResult {
    fn flushed_microbatches(&self) -> u64 {
        self.inner.flushed_microbatches()
    }

    fn did_update(&self) -> bool {
        self.inner.did_update()
    }

    fn optimizer_step(&self) -> u64 {
        self.inner.optimizer_step()
    }

    fn clip_report(&self) -> Option<&CompiledAdamWClipReport> {
        self.inner.clip_report()
    }

    fn window_loss_report(&self) -> Option<&CompiledAdamWWindowLossReport> {
        self.inner.window_loss_report()
    }
}
