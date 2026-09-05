//! Fixed-shape, device-resident dense Llama token execution on Metal.

use super::{
    LlamaLinearWeight, LlamaModel, LlamaOutputBinding, LlamaQkNorm, OUTPUT_NORM, OUTPUT_WEIGHT,
    ROPE_FREQS, TOKEN_EMBEDDING,
    layer::{add_bias, permute_rope_projection, rms_norm},
};
use crate::runtime::metal::{
    MetalAppendStateInferencePlan, MetalDevice, MetalDeviceRunReport, MetalDeviceSession,
    MetalDeviceSessionSummary, MetalError, MetalRenderer, MetalScoreboardError,
    MetalSessionScoreboard, RenderedMetal,
};
use crate::{
    AttentionOptions, CapturedAppendStateInference, CapturedInferenceError, CapturedSchedule,
    CompareOp, DType, Error, ExecutionPlanSummary, Graph, InferenceAppendStateLink, NodeId,
    ReplayInput, Scalar, Shape, TensorData,
    engine::capture::QuantizedCaptureBinding,
    gguf::{GgmlType, QuantizedTensorData},
};
use std::{collections::BTreeMap, error, fmt, num::NonZeroUsize};

const TOKEN_INPUT: &str = "llama.token";
const POSITION_INPUT: &str = "llama.position";
const PREFILL_TOKEN_INPUT: &str = "llama.tokens";
const PREFILL_POSITIONS_INPUT: &str = "llama.positions";
const ROPE_TABLE: &str = "llama.rope.table";
const ATTENTION_POSITIONS: &str = "llama.attention.positions";

/// Resource-free deployment of one dense F32, batch-one Llama token body.
///
/// The graph owns fixed-capacity K/V state and is captured exactly once. Token
/// is the only caller-supplied per-run input; the session seals and synthesizes
/// its committed scalar position, from which the row-shaped append index is
/// derived on device. All GGUF weights and precomputed position tables are
/// immutable named residents.
pub struct LlamaMetalStepPlan {
    inner: MetalAppendStateInferencePlan,
    max_context: usize,
    vocab_size: usize,
    layer_count: usize,
    output_binding: LlamaOutputBinding,
}

/// Resource-free token-step plan whose sole public output is a greedy I32 token
/// or the impossible negative sentinel when any logit is nonfinite.
pub struct LlamaMetalGreedyStepPlan {
    inner: MetalAppendStateInferencePlan,
    max_context: usize,
    vocab_size: usize,
    layer_count: usize,
    output_binding: LlamaOutputBinding,
}

/// Resource-free, state-only fixed-span Llama program used by the private
/// multi-program Metal session coordinator.
pub(crate) struct LlamaMetalPrefillPlan {
    inner: MetalAppendStateInferencePlan,
    span_rows: NonZeroUsize,
    max_context: usize,
    vocab_size: usize,
    layer_count: usize,
    output_binding: LlamaOutputBinding,
}

/// Persistent device-resident Llama token session whose position advances only
/// after every K/V row and the public logits commit successfully.
pub struct LlamaMetalStepSession {
    inner: MetalDeviceSession,
    max_context: usize,
    vocab_size: usize,
    scoreboard: Option<LlamaMetalScoreboardObserver>,
}

/// Persistent device-resident token session with typed greedy-only output.
pub struct LlamaMetalGreedyStepSession {
    inner: MetalDeviceSession,
    max_context: usize,
    vocab_size: usize,
}

struct LlamaMetalScoreboardObserver {
    recorder: MetalSessionScoreboard,
    first_error: Option<MetalScoreboardError>,
    #[cfg(test)]
    record_attempts: usize,
}

/// One successfully committed token invocation.
pub struct LlamaMetalStep {
    logits: TensorData,
    position: usize,
    report: MetalDeviceRunReport,
}

/// One committed device-greedy token.
pub struct LlamaMetalGreedyStep {
    token: u32,
    position: usize,
    report: MetalDeviceRunReport,
}

impl LlamaMetalGreedyStep {
    pub const fn token(&self) -> u32 {
        self.token
    }

    pub const fn position(&self) -> usize {
        self.position
    }

    pub const fn report(&self) -> &MetalDeviceRunReport {
        &self.report
    }

    pub fn into_parts(self) -> (u32, MetalDeviceRunReport) {
        (self.token, self.report)
    }
}

/// One successfully committed token whose logits remained device-local.
pub struct LlamaMetalTokenCommit {
    position: usize,
    report: MetalDeviceRunReport,
}

impl LlamaMetalTokenCommit {
    /// Returns the zero-based position consumed by this invocation.
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Returns the exact successful underlying Metal run report.
    pub const fn report(&self) -> &MetalDeviceRunReport {
        &self.report
    }

    /// Consumes the commit into its position and run report.
    pub fn into_parts(self) -> (usize, MetalDeviceRunReport) {
        (self.position, self.report)
    }
}

impl LlamaMetalStep {
    /// Returns the `[1, vocab]` F32 logits downloaded for this token.
    pub const fn logits(&self) -> &TensorData {
        &self.logits
    }

    /// Returns the zero-based position consumed by this invocation.
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Returns the exact successful underlying Metal run report.
    pub const fn report(&self) -> &MetalDeviceRunReport {
        &self.report
    }

    /// Consumes the step into detached logits and its run report.
    pub fn into_parts(self) -> (TensorData, MetalDeviceRunReport) {
        (self.logits, self.report)
    }
}

/// Dense-Llama planning, binding, graph, or strict Metal failure.
#[derive(Clone, Debug, PartialEq)]
pub enum LlamaMetalStepError {
    Graph(Error),
    Capture(CapturedInferenceError),
    Metal(MetalError),
    PackedTensor(String),
    Dimension(&'static str),
    TokenOutOfRange { token: u32, vocab_size: usize },
    ContextExhausted { position: usize, maximum: usize },
    NonFiniteLogits,
}

impl fmt::Display for LlamaMetalStepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llama Metal step error: {self:?}")
    }
}

impl error::Error for LlamaMetalStepError {}

impl From<Error> for LlamaMetalStepError {
    fn from(value: Error) -> Self {
        Self::Graph(value)
    }
}

impl From<CapturedInferenceError> for LlamaMetalStepError {
    fn from(value: CapturedInferenceError) -> Self {
        Self::Capture(value)
    }
}

impl From<MetalError> for LlamaMetalStepError {
    fn from(value: MetalError) -> Self {
        Self::Metal(value)
    }
}

impl LlamaMetalStepPlan {
    /// Builds, captures, and renders one reusable fixed-capacity token graph.
    /// Packed GGUF tensors and dimensions that cannot be represented by the
    /// I32 runtime ABI reject before any Metal resource is created.
    pub fn new(model: &LlamaModel, renderer: MetalRenderer) -> Result<Self, LlamaMetalStepError> {
        let parts = build_step_plan(model, renderer, StepOutputContract::HostLogits)?;
        Ok(Self {
            inner: parts.inner,
            max_context: parts.max_context,
            vocab_size: parts.vocab_size,
            layer_count: parts.layer_count,
            output_binding: parts.output_binding,
        })
    }

    /// Returns the capture plus exact resident/state payload identity.
    pub const fn deployment_identity(&self) -> u64 {
        self.inner.deployment_identity()
    }

    /// Returns the exact authenticated token-step capture.
    pub fn capture(&self) -> &CapturedSchedule {
        self.inner.capture()
    }

    /// Returns backend-neutral logical schedule and memory facts.
    pub const fn execution_plan(&self) -> &ExecutionPlanSummary {
        self.inner.execution_plan()
    }

    /// Returns deterministic Metal resource and execution planning facts.
    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.inner.summary()
    }

    /// Returns exact immutable model-weight and position-table schemas.
    pub fn resident_inputs(&self) -> &[ReplayInput] {
        self.inner.resident_inputs()
    }

    /// Returns the ordered per-layer K/V state-input schemas.
    pub fn state_inputs(&self) -> &[ReplayInput] {
        self.inner.state_inputs()
    }

    /// Returns the token-only caller transient schema.
    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.inner.transient_inputs()
    }

    /// Returns the sealed scalar-position schema synthesized per invocation.
    pub fn runtime_control_inputs(&self) -> &[ReplayInput] {
        self.inner.runtime_control_inputs()
    }

    /// Returns every rendered schedule item for inspection.
    pub fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.inner.rendered_items()
    }

    /// Returns the fixed K/V capacity.
    pub const fn max_context(&self) -> usize {
        self.max_context
    }

    /// Returns the exact GGUF vocabulary row count.
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns the transformer layer count.
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }

    /// Returns whether logits use an explicit or tied output weight.
    pub const fn output_binding(&self) -> LlamaOutputBinding {
        self.output_binding
    }

    /// Creates persistent resources, uploads every immutable resident once,
    /// and initializes the single physical K/V state bank at position zero.
    pub fn prepare(
        self,
        device: MetalDevice,
    ) -> Result<LlamaMetalStepSession, LlamaMetalStepError> {
        Ok(LlamaMetalStepSession {
            inner: self.inner.prepare(device)?,
            max_context: self.max_context,
            vocab_size: self.vocab_size,
            scoreboard: None,
        })
    }

    pub(crate) const fn append_state_plan(&self) -> &MetalAppendStateInferencePlan {
        &self.inner
    }
}

impl LlamaMetalGreedyStepPlan {
    /// Builds and captures one strict device-greedy token body.
    pub fn new(model: &LlamaModel, renderer: MetalRenderer) -> Result<Self, LlamaMetalStepError> {
        let parts = build_step_plan(model, renderer, StepOutputContract::DeviceGreedy)?;
        Ok(Self {
            inner: parts.inner,
            max_context: parts.max_context,
            vocab_size: parts.vocab_size,
            layer_count: parts.layer_count,
            output_binding: parts.output_binding,
        })
    }

    pub const fn deployment_identity(&self) -> u64 {
        self.inner.deployment_identity()
    }

    pub fn capture(&self) -> &CapturedSchedule {
        self.inner.capture()
    }

    pub const fn execution_plan(&self) -> &ExecutionPlanSummary {
        self.inner.execution_plan()
    }

    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.inner.summary()
    }

    pub fn resident_inputs(&self) -> &[ReplayInput] {
        self.inner.resident_inputs()
    }

    pub fn state_inputs(&self) -> &[ReplayInput] {
        self.inner.state_inputs()
    }

    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.inner.transient_inputs()
    }

    pub fn runtime_control_inputs(&self) -> &[ReplayInput] {
        self.inner.runtime_control_inputs()
    }

    pub fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.inner.rendered_items()
    }

    pub const fn max_context(&self) -> usize {
        self.max_context
    }

    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub const fn output_binding(&self) -> LlamaOutputBinding {
        self.output_binding
    }

    pub fn prepare(
        self,
        device: MetalDevice,
    ) -> Result<LlamaMetalGreedyStepSession, LlamaMetalStepError> {
        Ok(LlamaMetalGreedyStepSession {
            inner: self.inner.prepare(device)?,
            max_context: self.max_context,
            vocab_size: self.vocab_size,
        })
    }

    pub(crate) const fn append_state_plan(&self) -> &MetalAppendStateInferencePlan {
        &self.inner
    }
}

impl LlamaMetalPrefillPlan {
    /// Builds and captures one state-only fixed-span program without creating
    /// Metal resources. The scalar append position is sealed; exact token and
    /// position vectors remain typed inputs for the private session coordinator.
    /// That coordinator alone may consume this plan, and must validate before
    /// driver work that `positions[j] == scalar_position + j` for every row.
    pub(crate) fn new(
        model: &LlamaModel,
        renderer: MetalRenderer,
        span_rows: NonZeroUsize,
    ) -> Result<Self, LlamaMetalStepError> {
        let config = model.config();
        let schema = config.schema();
        let rows = span_rows.get();
        if schema.vocab_size() > i32::MAX as usize {
            return Err(LlamaMetalStepError::Dimension("vocabulary exceeds I32"));
        }
        if config.max_context() > i32::MAX as usize {
            return Err(LlamaMetalStepError::Dimension("context exceeds I32"));
        }
        if rows == 1 {
            return Err(LlamaMetalStepError::Dimension(
                "prefill span must exceed the existing token-step row",
            ));
        }
        if rows > config.max_context() || rows > i32::MAX as usize {
            return Err(LlamaMetalStepError::Dimension(
                "prefill span exceeds the fixed I32 context",
            ));
        }
        let built = build_prefill_graph(model, span_rows)?;
        let host_gathers = if built.packed_embedding {
            &[PREFILL_POSITIONS_INPUT][..]
        } else {
            &[PREFILL_TOKEN_INPUT, PREFILL_POSITIONS_INPUT][..]
        };
        let captured = CapturedAppendStateInference::from_graph_residents(
            &built.graph,
            &[],
            &built.state_links,
            built.initial_state,
            built.residents,
            &built.quantized,
            host_gathers,
        )?
        .seal_committed_position()?;
        let inner = MetalAppendStateInferencePlan::new(captured, renderer)?;
        if inner.summary().fallback_count != 0 {
            return Err(LlamaMetalStepError::Metal(MetalError::Unsupported(
                "Llama prefill plan admitted a fallback".into(),
            )));
        }
        if inner.append_span_rows() != rows || inner.summary().requested_output_count != 0 {
            return Err(LlamaMetalStepError::Metal(MetalError::InvalidBinding(
                "Llama prefill plan does not retain its exact state-only span".into(),
            )));
        }
        Ok(Self {
            inner,
            span_rows,
            max_context: config.max_context(),
            vocab_size: schema.vocab_size(),
            layer_count: config.layer_count(),
            output_binding: model.output_binding(),
        })
    }

    pub(crate) const fn deployment_identity(&self) -> u64 {
        self.inner.deployment_identity()
    }

    pub(crate) fn capture(&self) -> &CapturedSchedule {
        self.inner.capture()
    }

    pub(crate) const fn execution_plan(&self) -> &ExecutionPlanSummary {
        self.inner.execution_plan()
    }

    pub(crate) fn summary(&self) -> &MetalDeviceSessionSummary {
        self.inner.summary()
    }

    #[cfg(test)]
    pub(crate) fn resident_inputs(&self) -> &[ReplayInput] {
        self.inner.resident_inputs()
    }

    #[cfg(test)]
    pub(crate) fn state_inputs(&self) -> &[ReplayInput] {
        self.inner.state_inputs()
    }

    #[cfg(test)]
    pub(crate) fn quantized_input_names(&self) -> &BTreeMap<u64, String> {
        self.inner.quantized_input_names()
    }

    #[cfg(test)]
    pub(crate) fn transient_inputs(&self) -> &[ReplayInput] {
        self.inner.transient_inputs()
    }

    pub(crate) fn token_input(&self) -> &ReplayInput {
        self.inner
            .transient_inputs()
            .iter()
            .find(|input| input.name == PREFILL_TOKEN_INPUT)
            .expect("prefill capture authenticates its token vector")
    }

    pub(crate) fn position_vector_input(&self) -> &ReplayInput {
        self.inner
            .transient_inputs()
            .iter()
            .find(|input| input.name == PREFILL_POSITIONS_INPUT)
            .expect("prefill capture authenticates its position vector")
    }

    #[cfg(test)]
    pub(crate) fn runtime_control_inputs(&self) -> &[ReplayInput] {
        self.inner.runtime_control_inputs()
    }

    pub(crate) fn scalar_position_input(&self) -> &ReplayInput {
        let [position] = self.inner.runtime_control_inputs() else {
            unreachable!("prefill capture seals one scalar position")
        };
        position
    }

    pub(crate) fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.inner.rendered_items()
    }

    pub(crate) const fn span_rows(&self) -> NonZeroUsize {
        self.span_rows
    }

    pub(crate) const fn max_context(&self) -> usize {
        self.max_context
    }

    pub(crate) const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub(crate) const fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub(crate) const fn output_binding(&self) -> LlamaOutputBinding {
        self.output_binding
    }

    /// Borrows the sealed append plan for the sibling generation coordinator.
    pub(super) const fn append_state_plan(&self) -> &MetalAppendStateInferencePlan {
        &self.inner
    }

    /// Transfers the sealed append plan to the sibling generation coordinator.
    pub(super) fn into_append_state_plan(self) -> MetalAppendStateInferencePlan {
        self.inner
    }
}

impl LlamaMetalStepSession {
    /// Returns the number of tokens atomically committed to device K/V state.
    pub fn position(&self) -> usize {
        self.inner
            .committed_state_position()
            .expect("Llama plans always use append-state sessions")
    }

    /// Returns the fixed K/V capacity authenticated by this session.
    pub const fn max_context(&self) -> usize {
        self.max_context
    }

    /// Returns the exact GGUF vocabulary size authenticated by this session.
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns true after the final valid context position commits.
    pub fn is_full(&self) -> bool {
        self.position() == self.max_context
    }

    /// Returns the strict session for resource, metric, and kernel inspection.
    pub const fn metal_session(&self) -> &MetalDeviceSession {
        &self.inner
    }

    pub(crate) fn bind_execution_scoreboard(
        &mut self,
        mut recorder: MetalSessionScoreboard,
    ) -> Result<(), MetalScoreboardError> {
        recorder.bind(&self.inner)?;
        self.scoreboard = Some(LlamaMetalScoreboardObserver {
            recorder,
            first_error: None,
            #[cfg(test)]
            record_attempts: 0,
        });
        Ok(())
    }

    pub(crate) fn execution_scoreboard(&self) -> Option<&MetalSessionScoreboard> {
        self.scoreboard.as_ref().map(|state| &state.recorder)
    }

    pub(crate) fn scoreboard_recording_error(&self) -> Option<&MetalScoreboardError> {
        self.scoreboard
            .as_ref()
            .and_then(|state| state.first_error.as_ref())
    }

    #[cfg(test)]
    pub(crate) fn inject_scoreboard_recording_error(&mut self, error: MetalScoreboardError) {
        if let Some(state) = &mut self.scoreboard
            && state.first_error.is_none()
        {
            state.first_error = Some(error);
        }
    }

    #[cfg(test)]
    pub(crate) fn scoreboard_record_attempts(&self) -> Option<usize> {
        self.scoreboard.as_ref().map(|state| state.record_attempts)
    }

    /// Runs exactly one token. Invalid tokens, a full context, and failed
    /// device transactions preserve both position and the prior committed K/V rows.
    pub fn run_token(&mut self, token: u32) -> Result<LlamaMetalStep, LlamaMetalStepError> {
        let position = self.position();
        let invocation = self.inner.successful_run_count().checked_add(1).ok_or(
            LlamaMetalStepError::Dimension("invocation counter overflow"),
        )?;
        self.run_token_at(token, position, invocation)
    }

    pub(crate) fn run_token_at(
        &mut self,
        token: u32,
        position: usize,
        invocation: u64,
    ) -> Result<LlamaMetalStep, LlamaMetalStepError> {
        let inputs = self.token_inputs_at(token, position)?;
        let run = self.inner.run_at(&inputs, position)?;
        self.observe_run(&run);
        let (mut outputs, mut report) = run.into_parts();
        report.successful_invocation = invocation;
        report.first_successful_run = invocation == 1;
        debug_assert_eq!(outputs.len(), 1);
        let logits = outputs
            .pop()
            .expect("capture authenticates one Llama output");
        debug_assert_eq!(logits.shape().dims(), [1, self.vocab_size]);
        debug_assert_eq!(logits.dtype(), DType::F32);
        Ok(LlamaMetalStep {
            logits,
            position,
            report,
        })
    }

    /// Commits exactly one prompt token while retaining its computed logits on
    /// device. This advances the same atomic K/V position as [`Self::run_token`].
    pub fn commit_token(
        &mut self,
        token: u32,
    ) -> Result<LlamaMetalTokenCommit, LlamaMetalStepError> {
        let position = self.position();
        let invocation = self.inner.successful_run_count().checked_add(1).ok_or(
            LlamaMetalStepError::Dimension("invocation counter overflow"),
        )?;
        self.commit_token_at(token, position, invocation)
    }

    pub(crate) fn commit_token_at(
        &mut self,
        token: u32,
        position: usize,
        invocation: u64,
    ) -> Result<LlamaMetalTokenCommit, LlamaMetalStepError> {
        let inputs = self.token_inputs_at(token, position)?;
        let run = self.inner.run_without_host_outputs_at(&inputs, position)?;
        self.observe_run(&run);
        debug_assert!(run.outputs().is_empty());
        let (_, mut report) = run.into_parts();
        report.successful_invocation = invocation;
        report.first_successful_run = invocation == 1;
        Ok(LlamaMetalTokenCommit { position, report })
    }

    fn token_inputs_at(
        &self,
        token: u32,
        position: usize,
    ) -> Result<BTreeMap<String, TensorData>, LlamaMetalStepError> {
        if token > i32::MAX as u32 || token as usize >= self.vocab_size {
            return Err(LlamaMetalStepError::TokenOutOfRange {
                token,
                vocab_size: self.vocab_size,
            });
        }
        if position >= self.max_context {
            return Err(LlamaMetalStepError::ContextExhausted {
                position,
                maximum: self.max_context,
            });
        }
        Ok(BTreeMap::from([(
            TOKEN_INPUT.to_owned(),
            TensorData::from_scalars([1, 1], DType::I32, [Scalar::I(i64::from(token))])?,
        )]))
    }

    fn observe_run(&mut self, run: &crate::runtime::metal::MetalDeviceRun) {
        let Some(state) = &mut self.scoreboard else {
            return;
        };
        if state.first_error.is_some() {
            return;
        }
        #[cfg(test)]
        {
            state.record_attempts += 1;
        }
        if let Err(error) = state.recorder.record(run) {
            state.first_error = Some(error);
        }
    }
}

impl LlamaMetalGreedyStepSession {
    pub fn position(&self) -> usize {
        self.inner
            .committed_state_position()
            .expect("Llama plans always use append-state sessions")
    }

    pub const fn max_context(&self) -> usize {
        self.max_context
    }

    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn is_full(&self) -> bool {
        self.position() == self.max_context
    }

    pub const fn metal_session(&self) -> &MetalDeviceSession {
        &self.inner
    }

    /// Runs one token, downloads only the guarded I32 greedy token, and commits
    /// position only after the result is nonnegative.
    pub fn run_token(&mut self, token: u32) -> Result<LlamaMetalGreedyStep, LlamaMetalStepError> {
        let position = self.position();
        let invocation = self.inner.successful_run_count().checked_add(1).ok_or(
            LlamaMetalStepError::Dimension("invocation counter overflow"),
        )?;
        self.run_token_at(token, position, invocation)
    }

    pub(crate) fn run_token_at(
        &mut self,
        token: u32,
        position: usize,
        invocation: u64,
    ) -> Result<LlamaMetalGreedyStep, LlamaMetalStepError> {
        let inputs = self.token_inputs_at(token, position)?;
        let run = self
            .inner
            .run_at_requiring_bounded_i32(&inputs, position, 0, self.vocab_size)
            .map_err(|error| match error {
                MetalError::InvalidDeviceProof(_) => LlamaMetalStepError::NonFiniteLogits,
                error => LlamaMetalStepError::Metal(error),
            })?;
        let (outputs, mut report) = run.into_parts();
        report.successful_invocation = invocation;
        report.first_successful_run = invocation == 1;
        let [token_output] = outputs.as_slice() else {
            return Err(LlamaMetalStepError::Dimension(
                "device-greedy output count changed",
            ));
        };
        if token_output.dtype() != DType::I32 || token_output.len() != 1 {
            return Err(LlamaMetalStepError::Dimension(
                "device-greedy output schema changed",
            ));
        }
        let selected = token_output.scalar_at(0).as_i64();
        let token = u32::try_from(selected)
            .ok()
            .filter(|token| (*token as usize) < self.vocab_size)
            .ok_or(LlamaMetalStepError::Dimension(
                "device-greedy token is outside the vocabulary",
            ))?;
        Ok(LlamaMetalGreedyStep {
            token,
            position,
            report,
        })
    }

    pub fn commit_token(
        &mut self,
        token: u32,
    ) -> Result<LlamaMetalTokenCommit, LlamaMetalStepError> {
        let position = self.position();
        let invocation = self.inner.successful_run_count().checked_add(1).ok_or(
            LlamaMetalStepError::Dimension("invocation counter overflow"),
        )?;
        self.commit_token_at(token, position, invocation)
    }

    pub(crate) fn commit_token_at(
        &mut self,
        token: u32,
        position: usize,
        invocation: u64,
    ) -> Result<LlamaMetalTokenCommit, LlamaMetalStepError> {
        let inputs = self.token_inputs_at(token, position)?;
        let run = self.inner.run_without_host_outputs_at(&inputs, position)?;
        debug_assert!(run.outputs().is_empty());
        let (_, mut report) = run.into_parts();
        report.successful_invocation = invocation;
        report.first_successful_run = invocation == 1;
        Ok(LlamaMetalTokenCommit { position, report })
    }

    fn token_inputs_at(
        &self,
        token: u32,
        position: usize,
    ) -> Result<BTreeMap<String, TensorData>, LlamaMetalStepError> {
        if token > i32::MAX as u32 || token as usize >= self.vocab_size {
            return Err(LlamaMetalStepError::TokenOutOfRange {
                token,
                vocab_size: self.vocab_size,
            });
        }
        if position >= self.max_context {
            return Err(LlamaMetalStepError::ContextExhausted {
                position,
                maximum: self.max_context,
            });
        }
        Ok(BTreeMap::from([(
            TOKEN_INPUT.to_owned(),
            TensorData::from_scalars([1, 1], DType::I32, [Scalar::I(i64::from(token))])?,
        )]))
    }
}

struct BuiltStepGraph {
    graph: Graph,
    residents: BTreeMap<String, (NodeId, TensorData)>,
    quantized: Vec<QuantizedCaptureBinding>,
    initial_state: BTreeMap<String, TensorData>,
    state_links: Vec<InferenceAppendStateLink>,
    packed_embedding: bool,
    logits: NodeId,
}

#[derive(Clone, Copy)]
enum StepOutputContract {
    HostLogits,
    DeviceGreedy,
}

struct BuiltStepPlan {
    inner: MetalAppendStateInferencePlan,
    max_context: usize,
    vocab_size: usize,
    layer_count: usize,
    output_binding: LlamaOutputBinding,
}

fn build_step_plan(
    model: &LlamaModel,
    renderer: MetalRenderer,
    output: StepOutputContract,
) -> Result<BuiltStepPlan, LlamaMetalStepError> {
    let config = model.config();
    let schema = config.schema();
    if schema.vocab_size() > i32::MAX as usize {
        return Err(LlamaMetalStepError::Dimension("vocabulary exceeds I32"));
    }
    if config.max_context() > i32::MAX as usize {
        return Err(LlamaMetalStepError::Dimension("context exceeds I32"));
    }
    let mut built = build_step_graph(model)?;
    let requested = match output {
        StepOutputContract::HostLogits => vec![built.logits],
        StepOutputContract::DeviceGreedy => {
            vec![guarded_greedy_token(&mut built.graph, built.logits)?]
        }
    };
    let host_gathers = if built.packed_embedding {
        &[POSITION_INPUT][..]
    } else {
        &[TOKEN_INPUT, POSITION_INPUT][..]
    };
    let captured = CapturedAppendStateInference::from_graph_residents(
        &built.graph,
        &requested,
        &built.state_links,
        built.initial_state,
        built.residents,
        &built.quantized,
        host_gathers,
    )?
    .seal_committed_position()?;
    let inner = MetalAppendStateInferencePlan::new(captured, renderer)?;
    if inner.summary().fallback_count != 0 {
        return Err(LlamaMetalStepError::Metal(MetalError::Unsupported(
            "Llama token plan admitted a fallback".into(),
        )));
    }
    if inner.summary().requested_output_count != 1 {
        return Err(LlamaMetalStepError::Metal(MetalError::InvalidBinding(
            "Llama token plan changed its typed output contract".into(),
        )));
    }
    Ok(BuiltStepPlan {
        inner,
        max_context: config.max_context(),
        vocab_size: schema.vocab_size(),
        layer_count: config.layer_count(),
        output_binding: model.output_binding(),
    })
}

fn guarded_greedy_token(graph: &mut Graph, logits: NodeId) -> Result<NodeId, Error> {
    let finite = finite_logit_lanes(graph, logits)?;
    let finite = graph.all_default(finite)?;
    let token = graph.argmax_with_axis(logits, Some(-1), false)?;
    let invalid = graph.constant(TensorData::scalar_with_dtype(Scalar::I(-1), DType::I32));
    graph.select(finite, token, invalid)
}

fn finite_logit_lanes(graph: &mut Graph, logits: NodeId) -> Result<NodeId, Error> {
    // Keep the predicate inside the exact Metal scalar-lane subset. Direct
    // ordered comparisons are false for NaN; the conjunction excludes both
    // infinities while accepting every finite F32, including both extrema.
    let minimum = graph.constant(TensorData::scalar_with_dtype(
        Scalar::F(-f64::from(f32::MAX)),
        DType::F32,
    ));
    let maximum = graph.constant(TensorData::scalar_with_dtype(
        Scalar::F(f64::from(f32::MAX)),
        DType::F32,
    ));
    let at_least_minimum = graph.compare(CompareOp::Ge, logits, minimum)?;
    let at_most_maximum = graph.compare(CompareOp::Le, logits, maximum)?;
    graph.logical_and(at_least_minimum, at_most_maximum)
}

struct BuiltPrefillGraph {
    graph: Graph,
    residents: BTreeMap<String, (NodeId, TensorData)>,
    quantized: Vec<QuantizedCaptureBinding>,
    initial_state: BTreeMap<String, TensorData>,
    state_links: Vec<InferenceAppendStateLink>,
    packed_embedding: bool,
}

fn build_prefill_graph(
    model: &LlamaModel,
    span_rows: NonZeroUsize,
) -> Result<BuiltPrefillGraph, LlamaMetalStepError> {
    let config = model.config();
    let schema = config.schema();
    let rows = span_rows.get();
    let mut graph = Graph::new();
    let tokens = graph.input_dtype_requires_grad(PREFILL_TOKEN_INPUT, [1, rows], DType::I32, false);
    let position = graph.input_dtype_requires_grad(POSITION_INPUT, [1], DType::I32, false);
    let positions =
        graph.input_dtype_requires_grad(PREFILL_POSITIONS_INPUT, [1, rows], DType::I32, false);
    let terminal_layer_prefix = config
        .layer_count()
        .checked_sub(1)
        .map(|layer| format!("blk.{layer}."));
    let terminal_only = |name: &str| {
        let Some(suffix) = terminal_layer_prefix
            .as_deref()
            .and_then(|prefix| name.strip_prefix(prefix))
        else {
            return false;
        };
        matches!(
            suffix,
            "attn_q.weight"
                | "attn_q.bias"
                | "attn_q_norm.weight"
                | "attn_output.weight"
                | "ffn_norm.weight"
                | "ffn_gate.weight"
                | "ffn_up.weight"
                | "ffn_down.weight"
        )
    };
    let mut residents = BTreeMap::new();
    let mut nodes = BTreeMap::new();
    let mut packed = BTreeMap::new();
    insert_weight(
        &mut graph,
        &mut residents,
        &mut nodes,
        &mut packed,
        TOKEN_EMBEDDING,
        model.embedding_weight(),
    )?;
    for (name, value) in model.dense_state().iter().filter(|(name, _)| {
        name.as_str() != ROPE_FREQS && name.as_str() != OUTPUT_NORM && !terminal_only(name)
    }) {
        insert_resident(&mut graph, &mut residents, &mut nodes, name, value)?;
    }
    for (name, weight) in model
        .linear_weights()
        .iter()
        .filter(|(name, _)| name.as_str() != OUTPUT_WEIGHT && !terminal_only(name))
    {
        insert_weight(
            &mut graph,
            &mut residents,
            &mut nodes,
            &mut packed,
            name,
            weight,
        )?;
    }
    let rope = rope_table(config.max_context(), schema.rope_dim(), config.rope_theta())?;
    insert_resident(&mut graph, &mut residents, &mut nodes, ROPE_TABLE, &rope)?;
    let attention_positions = attention_position_table(config.max_context())?;
    insert_resident_with_dtype(
        &mut graph,
        &mut residents,
        &mut nodes,
        ATTENTION_POSITIONS,
        &attention_positions,
        DType::I32,
        "attention position resident must be I32",
    )?;

    let mut quantized = Vec::new();
    let mut x = lookup_prefill_embedding(
        &mut graph,
        nodes[TOKEN_EMBEDDING],
        tokens,
        rows,
        schema.vocab_size(),
        schema.embedding_dim(),
    )?;
    let packed_embedding = packed.contains_key(TOKEN_EMBEDDING);
    if let Some(weight) = packed.get(TOKEN_EMBEDDING) {
        quantized.push(QuantizedCaptureBinding::RowGather {
            output: x,
            indices: tokens,
            weight: nodes[TOKEN_EMBEDDING],
            value: weight.clone(),
        });
    }
    let rope_rows = lookup_prefill_rope_rows(
        &mut graph,
        nodes[ROPE_TABLE],
        positions,
        rows,
        config.max_context(),
        schema.rope_dim(),
    )?;
    let absolute_positions = nodes[ATTENTION_POSITIONS];
    let absolute_positions = graph.reshape(absolute_positions, [1, 1, 1, config.max_context()])?;
    let query_positions = graph.reshape(positions, [1, 1, rows, 1])?;
    let attention_mask = graph.le(absolute_positions, query_positions)?;

    let cache_shape = Shape::new([
        1,
        schema.kv_heads(),
        config.max_context(),
        schema.head_dim(),
    ]);
    let mut initial_state = BTreeMap::new();
    let state_count = config
        .layer_count()
        .checked_mul(2)
        .ok_or(LlamaMetalStepError::Dimension("KV state count overflow"))?;
    let mut state_links = Vec::with_capacity(state_count);
    let mut append_index = None;
    for layer in 0..config.layer_count() {
        let key_name = format!("llama.state.{layer}.key");
        let value_name = format!("llama.state.{layer}.value");
        let past_key =
            graph.input_dtype_requires_grad(&key_name, cache_shape.clone(), DType::F32, false);
        let past_value =
            graph.input_dtype_requires_grad(&value_name, cache_shape.clone(), DType::F32, false);
        initial_state.insert(
            key_name,
            TensorData::zeros_with_dtype(cache_shape.clone(), DType::F32)?,
        );
        initial_state.insert(
            value_name,
            TensorData::zeros_with_dtype(cache_shape.clone(), DType::F32)?,
        );
        let built = PrefillLayerBuildContext {
            graph: &mut graph,
            nodes: &nodes,
            packed: &packed,
            quantized: &mut quantized,
            config,
            rope_rows,
            append_index,
            position,
            attention_mask,
            rows,
            needs_output: layer + 1 < config.layer_count(),
        }
        .append(x, layer, past_key, past_value)?;
        append_index = Some(built.append_index);
        if let Some(output) = built.output {
            x = output;
        }
        state_links.extend([
            InferenceAppendStateLink::new(
                past_key,
                built.key,
                position,
                built.append_index,
                built.key_update,
                2,
            ),
            InferenceAppendStateLink::new(
                past_value,
                built.value,
                position,
                built.append_index,
                built.value_update,
                2,
            ),
        ]);
    }

    Ok(BuiltPrefillGraph {
        graph,
        residents,
        quantized,
        initial_state,
        state_links,
        packed_embedding,
    })
}

fn build_step_graph(model: &LlamaModel) -> Result<BuiltStepGraph, LlamaMetalStepError> {
    let config = model.config();
    let schema = config.schema();
    let mut graph = Graph::new();
    let token = graph.input_dtype_requires_grad(TOKEN_INPUT, [1, 1], DType::I32, false);
    let position = graph.input_dtype_requires_grad(POSITION_INPUT, [1], DType::I32, false);
    let append_index_shape = Shape::new([1, schema.kv_heads(), 1, schema.head_dim()]);
    let append_index = graph.reshape(position, vec![1; append_index_shape.rank()])?;
    let append_index = graph.expand(append_index, append_index_shape)?;
    let mut residents = BTreeMap::new();
    let mut nodes = BTreeMap::new();
    let mut packed = BTreeMap::new();
    insert_weight(
        &mut graph,
        &mut residents,
        &mut nodes,
        &mut packed,
        TOKEN_EMBEDDING,
        model.embedding_weight(),
    )?;
    for (name, value) in model
        .dense_state()
        .iter()
        .filter(|(name, _)| name.as_str() != ROPE_FREQS)
    {
        insert_resident(&mut graph, &mut residents, &mut nodes, name, value)?;
    }
    for (name, weight) in model.linear_weights() {
        insert_weight(
            &mut graph,
            &mut residents,
            &mut nodes,
            &mut packed,
            name,
            weight,
        )?;
    }
    let rope = rope_table(config.max_context(), schema.rope_dim(), config.rope_theta())?;
    insert_resident(&mut graph, &mut residents, &mut nodes, ROPE_TABLE, &rope)?;
    let attention_positions = attention_position_table(config.max_context())?;
    insert_resident_with_dtype(
        &mut graph,
        &mut residents,
        &mut nodes,
        ATTENTION_POSITIONS,
        &attention_positions,
        DType::I32,
        "attention position resident must be I32",
    )?;

    let mut quantized = Vec::new();
    let mut x = lookup_embedding(
        &mut graph,
        nodes[TOKEN_EMBEDDING],
        token,
        schema.vocab_size(),
        schema.embedding_dim(),
    )?;
    let packed_embedding = packed.contains_key(TOKEN_EMBEDDING);
    if let Some(weight) = packed.get(TOKEN_EMBEDDING) {
        quantized.push(QuantizedCaptureBinding::RowGather {
            output: x,
            indices: token,
            weight: nodes[TOKEN_EMBEDDING],
            value: weight.clone(),
        });
    }
    let rope_row = lookup_rope_row(&mut graph, nodes[ROPE_TABLE], position, schema.rope_dim())?;
    let positions = nodes[ATTENTION_POSITIONS];
    let positions = graph.reshape(positions, [1, 1, 1, config.max_context()])?;
    let position_mask = graph.reshape(position, [1, 1, 1, 1])?;
    let attention_mask = graph.le(positions, position_mask)?;

    let cache_shape = Shape::new([
        1,
        schema.kv_heads(),
        config.max_context(),
        schema.head_dim(),
    ]);
    let mut initial_state = BTreeMap::new();
    let state_count = config
        .layer_count()
        .checked_mul(2)
        .ok_or(LlamaMetalStepError::Dimension("KV state count overflow"))?;
    let mut state_links = Vec::with_capacity(state_count);
    for layer in 0..config.layer_count() {
        let key_name = format!("llama.state.{layer}.key");
        let value_name = format!("llama.state.{layer}.value");
        let past_key =
            graph.input_dtype_requires_grad(&key_name, cache_shape.clone(), DType::F32, false);
        let past_value =
            graph.input_dtype_requires_grad(&value_name, cache_shape.clone(), DType::F32, false);
        initial_state.insert(
            key_name,
            TensorData::zeros_with_dtype(cache_shape.clone(), DType::F32)?,
        );
        initial_state.insert(
            value_name,
            TensorData::zeros_with_dtype(cache_shape.clone(), DType::F32)?,
        );
        let built = StepLayerBuildContext {
            graph: &mut graph,
            nodes: &nodes,
            packed: &packed,
            quantized: &mut quantized,
            config,
            rope_row,
            append_index,
            attention_mask,
        }
        .append(x, layer, past_key, past_value)?;
        x = built.output;
        state_links.extend([
            InferenceAppendStateLink::new(
                past_key,
                built.key,
                position,
                append_index,
                built.key_update,
                2,
            ),
            InferenceAppendStateLink::new(
                past_value,
                built.value,
                position,
                append_index,
                built.value_update,
                2,
            ),
        ]);
    }

    let normalized = rms_norm(
        &mut graph,
        x,
        nodes[OUTPUT_NORM],
        schema.embedding_dim(),
        config.norm_eps(),
    )?;
    let output_name = match model.output_binding() {
        LlamaOutputBinding::Explicit => OUTPUT_WEIGHT,
        LlamaOutputBinding::TiedToTokenEmbedding => TOKEN_EMBEDDING,
    };
    let logits = model_linear(
        &mut graph,
        normalized,
        output_name,
        &nodes,
        &packed,
        &mut quantized,
    )?;
    let logits = graph.reshape(logits, [1, schema.vocab_size()])?;
    Ok(BuiltStepGraph {
        graph,
        residents,
        quantized,
        initial_state,
        state_links,
        packed_embedding,
        logits,
    })
}

fn insert_weight(
    graph: &mut Graph,
    residents: &mut BTreeMap<String, (NodeId, TensorData)>,
    nodes: &mut BTreeMap<String, NodeId>,
    packed: &mut BTreeMap<String, QuantizedTensorData>,
    name: &str,
    weight: &LlamaLinearWeight,
) -> Result<(), LlamaMetalStepError> {
    match weight {
        LlamaLinearWeight::Dense(value) => insert_resident(graph, residents, nodes, name, value),
        LlamaLinearWeight::Quantized(value) => {
            if !matches!(
                value.descriptor().ggml_type,
                GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q6K
            ) {
                return Err(LlamaMetalStepError::PackedTensor(format!(
                    "{name} ({:?})",
                    value.descriptor().ggml_type
                )));
            }
            value
                .validate()
                .map_err(|error| LlamaMetalStepError::PackedTensor(format!("{name} ({error})")))?;
            if residents.contains_key(name) || packed.contains_key(name) {
                return Err(LlamaMetalStepError::Dimension(
                    "duplicate Llama resident name",
                ));
            }
            let node = graph.input_dtype_requires_grad(
                format!("llama.packed.{name}"),
                value.descriptor().logical_shape.clone(),
                DType::F32,
                false,
            );
            nodes.insert(name.to_owned(), node);
            packed.insert(name.to_owned(), value.clone());
            Ok(())
        }
    }
}

fn insert_resident(
    graph: &mut Graph,
    residents: &mut BTreeMap<String, (NodeId, TensorData)>,
    nodes: &mut BTreeMap<String, NodeId>,
    name: &str,
    value: &TensorData,
) -> Result<(), LlamaMetalStepError> {
    insert_resident_with_dtype(
        graph,
        residents,
        nodes,
        name,
        value,
        DType::F32,
        "dense Llama residents must be F32",
    )
}

fn insert_resident_with_dtype(
    graph: &mut Graph,
    residents: &mut BTreeMap<String, (NodeId, TensorData)>,
    nodes: &mut BTreeMap<String, NodeId>,
    name: &str,
    value: &TensorData,
    expected_dtype: DType,
    dtype_error: &'static str,
) -> Result<(), LlamaMetalStepError> {
    if value.dtype() != expected_dtype {
        return Err(LlamaMetalStepError::Dimension(dtype_error));
    }
    if residents.contains_key(name) {
        return Err(LlamaMetalStepError::Dimension(
            "duplicate Llama resident name",
        ));
    }
    let node = graph.input_dtype_requires_grad(name, value.shape().clone(), expected_dtype, false);
    residents.insert(name.to_owned(), (node, value.clone()));
    nodes.insert(name.to_owned(), node);
    Ok(())
}

// Both scalar sources are host-validated before any driver work. Capture then
// authenticates their value-preserving reshape/expand lineage into raw Gather.
fn lookup_embedding(
    graph: &mut Graph,
    embedding: NodeId,
    token: NodeId,
    vocab_size: usize,
    embedding_dim: usize,
) -> Result<NodeId, Error> {
    let embedding = graph.reshape(embedding, [1, vocab_size, embedding_dim])?;
    let index = graph.reshape(token, [1, 1, 1])?;
    let index = graph.expand(index, [1, 1, embedding_dim])?;
    graph.gather(embedding, index, 1)
}

fn lookup_rope_row(
    graph: &mut Graph,
    table: NodeId,
    position: NodeId,
    rope_dim: usize,
) -> Result<NodeId, Error> {
    let index = graph.reshape(position, [1, 1])?;
    let index = graph.expand(index, [1, rope_dim])?;
    graph.gather(table, index, 0)
}

fn lookup_prefill_embedding(
    graph: &mut Graph,
    embedding: NodeId,
    tokens: NodeId,
    rows: usize,
    vocab_size: usize,
    embedding_dim: usize,
) -> Result<NodeId, Error> {
    let embedding = graph.reshape(embedding, [1, vocab_size, embedding_dim])?;
    let index = graph.reshape(tokens, [1, rows, 1])?;
    let index = graph.expand(index, [1, rows, embedding_dim])?;
    graph.gather(embedding, index, 1)
}

fn lookup_prefill_rope_rows(
    graph: &mut Graph,
    table: NodeId,
    positions: NodeId,
    rows: usize,
    max_context: usize,
    rope_dim: usize,
) -> Result<NodeId, Error> {
    let table = graph.reshape(table, [1, max_context, rope_dim])?;
    let index = graph.reshape(positions, [1, rows, 1])?;
    let index = graph.expand(index, [1, rows, rope_dim])?;
    graph.gather(table, index, 1)
}

fn fixed_span_append_index(
    graph: &mut Graph,
    position: NodeId,
    update: NodeId,
    axis: usize,
) -> Result<NodeId, Error> {
    let shape = graph.shape(update)?.clone();
    let expanded_position = graph.reshape(position, vec![1; shape.rank()])?;
    let expanded_position = graph.expand(expanded_position, shape.clone())?;
    let iota = graph.shape_iota(update, axis)?;
    let mut iota_shape = vec![1; shape.rank()];
    iota_shape[axis] = shape.dims()[axis];
    let expanded_iota = graph.reshape(iota, iota_shape)?;
    let expanded_iota = graph.expand(expanded_iota, shape)?;
    graph.add(expanded_position, expanded_iota)
}

// Keep the exact materialized row and raw Scatter boundary isolated so the
// append-state capture can authenticate one device-produced dense update.
fn append_cache_row(
    graph: &mut Graph,
    state: NodeId,
    index: NodeId,
    value: NodeId,
) -> Result<(NodeId, NodeId), Error> {
    let update = graph.contiguous(value)?;
    let output = graph.scatter(state, index, update, 2)?;
    Ok((update, output))
}

fn rope_table(
    max_context: usize,
    rope_dim: usize,
    theta: f64,
) -> Result<TensorData, LlamaMetalStepError> {
    let half = rope_dim / 2;
    let mut values = Vec::with_capacity(
        max_context
            .checked_mul(rope_dim)
            .ok_or(LlamaMetalStepError::Dimension("RoPE table overflow"))?,
    );
    for position in 0..max_context {
        let angles = (0..half)
            .map(|index| {
                let frequency = 1.0 / theta.powf((2 * index) as f64 / rope_dim as f64);
                position as f64 * frequency
            })
            .collect::<Vec<_>>();
        values.extend(angles.iter().map(|angle| angle.cos() as f32));
        values.extend(angles.iter().map(|angle| angle.sin() as f32));
    }
    Ok(TensorData::new([max_context, rope_dim], values)?)
}

fn attention_position_table(max_context: usize) -> Result<TensorData, LlamaMetalStepError> {
    Ok(TensorData::from_scalars(
        [max_context],
        DType::I32,
        (0..max_context).map(|value| Scalar::I(value as i64)),
    )?)
}

struct StepLayerNodes {
    output: NodeId,
    key: NodeId,
    value: NodeId,
    key_update: NodeId,
    value_update: NodeId,
}

struct PrefillLayerNodes {
    output: Option<NodeId>,
    key: NodeId,
    value: NodeId,
    key_update: NodeId,
    value_update: NodeId,
    append_index: NodeId,
}

struct PrefillLayerBuildContext<'a> {
    graph: &'a mut Graph,
    nodes: &'a BTreeMap<String, NodeId>,
    packed: &'a BTreeMap<String, QuantizedTensorData>,
    quantized: &'a mut Vec<QuantizedCaptureBinding>,
    config: &'a super::LlamaModelConfig,
    rope_rows: NodeId,
    append_index: Option<NodeId>,
    position: NodeId,
    attention_mask: NodeId,
    rows: usize,
    needs_output: bool,
}

impl PrefillLayerBuildContext<'_> {
    fn append(
        self,
        mut x: NodeId,
        layer: usize,
        past_key: NodeId,
        past_value: NodeId,
    ) -> Result<PrefillLayerNodes, LlamaMetalStepError> {
        let Self {
            graph,
            nodes,
            packed,
            quantized,
            config,
            rope_rows,
            append_index,
            position,
            attention_mask,
            rows,
            needs_output,
        } = self;
        let schema = config.schema();
        let name = |suffix: &str| format!("blk.{layer}.{suffix}");
        let attn_norm = rms_norm(
            graph,
            x,
            nodes[&name("attn_norm.weight")],
            schema.embedding_dim(),
            config.norm_eps(),
        )?;
        let mut key = model_linear(
            graph,
            attn_norm,
            &name("attn_k.weight"),
            nodes,
            packed,
            quantized,
        )?;
        let value = model_linear(
            graph,
            attn_norm,
            &name("attn_v.weight"),
            nodes,
            packed,
            quantized,
        )?;
        key = permute_rope_projection(
            graph,
            key,
            schema.kv_heads(),
            schema.head_dim(),
            schema.rope_dim(),
            false,
        )?;
        key = add_bias(
            graph,
            key,
            config.qkv_bias().then(|| nodes[&name("attn_k.bias")]),
        )?;
        let value = add_bias(
            graph,
            value,
            config.qkv_bias().then(|| nodes[&name("attn_v.bias")]),
        )?;
        if config.qk_norm() == LlamaQkNorm::PerProjection {
            key = rms_norm(
                graph,
                key,
                nodes[&name("attn_k_norm.weight")],
                schema.kv_heads() * schema.head_dim(),
                config.norm_eps(),
            )?;
        }
        key = graph.reshape(key, [1, rows, schema.kv_heads(), schema.head_dim()])?;
        key = graph.permute(key, vec![0, 2, 1, 3])?;
        let mut value = graph.reshape(value, [1, rows, schema.kv_heads(), schema.head_dim()])?;
        value = graph.permute(value, vec![0, 2, 1, 3])?;
        if config.qk_norm() == LlamaQkNorm::PerHead {
            key = rms_norm(
                graph,
                key,
                nodes[&name("attn_k_norm.weight")],
                schema.head_dim(),
                config.norm_eps(),
            )?;
        }
        key = apply_prefill_rope(graph, key, rope_rows, rows, schema)?;
        let key_update = graph.contiguous(key)?;
        let append_index = append_index
            .map(Ok)
            .unwrap_or_else(|| fixed_span_append_index(graph, position, key_update, 2))?;
        let key = graph.scatter(past_key, append_index, key_update, 2)?;
        let value_update = graph.contiguous(value)?;
        let value = graph.scatter(past_value, append_index, value_update, 2)?;

        if !needs_output {
            return Ok(PrefillLayerNodes {
                output: None,
                key,
                value,
                key_update,
                value_update,
                append_index,
            });
        }

        let mut query = model_linear(
            graph,
            attn_norm,
            &name("attn_q.weight"),
            nodes,
            packed,
            quantized,
        )?;
        query = permute_rope_projection(
            graph,
            query,
            schema.query_heads(),
            schema.head_dim(),
            schema.rope_dim(),
            true,
        )?;
        query = add_bias(
            graph,
            query,
            config.qkv_bias().then(|| nodes[&name("attn_q.bias")]),
        )?;
        if config.qk_norm() == LlamaQkNorm::PerProjection {
            query = rms_norm(
                graph,
                query,
                nodes[&name("attn_q_norm.weight")],
                schema.query_heads() * schema.head_dim(),
                config.norm_eps(),
            )?;
        }
        query = graph.reshape(query, [1, rows, schema.query_heads(), schema.head_dim()])?;
        query = graph.permute(query, vec![0, 2, 1, 3])?;
        if config.qk_norm() == LlamaQkNorm::PerHead {
            query = rms_norm(
                graph,
                query,
                nodes[&name("attn_q_norm.weight")],
                schema.head_dim(),
                config.norm_eps(),
            )?;
        }
        query = apply_prefill_rope(graph, query, rope_rows, rows, schema)?;
        let attended = graph.scaled_dot_product_attention(
            query,
            key,
            value,
            Some(attention_mask),
            AttentionOptions {
                enable_gqa: true,
                ..AttentionOptions::default()
            },
        )?;
        let attended = graph.permute(attended, vec![0, 2, 1, 3])?;
        let attended = graph.reshape(
            attended,
            [1, rows, schema.query_heads() * schema.head_dim()],
        )?;
        let attended = model_linear(
            graph,
            attended,
            &name("attn_output.weight"),
            nodes,
            packed,
            quantized,
        )?;
        x = graph.add(x, attended)?;
        let normalized = rms_norm(
            graph,
            x,
            nodes[&name("ffn_norm.weight")],
            schema.embedding_dim(),
            config.norm_eps(),
        )?;
        let gate = model_linear(
            graph,
            normalized,
            &name("ffn_gate.weight"),
            nodes,
            packed,
            quantized,
        )?;
        let gate = graph.silu(gate)?;
        let up = model_linear(
            graph,
            normalized,
            &name("ffn_up.weight"),
            nodes,
            packed,
            quantized,
        )?;
        let gated = graph.mul(gate, up)?;
        let down = model_linear(
            graph,
            gated,
            &name("ffn_down.weight"),
            nodes,
            packed,
            quantized,
        )?;
        x = graph.add(x, down)?;
        Ok(PrefillLayerNodes {
            output: Some(x),
            key,
            value,
            key_update,
            value_update,
            append_index,
        })
    }
}

struct StepLayerBuildContext<'a> {
    graph: &'a mut Graph,
    nodes: &'a BTreeMap<String, NodeId>,
    packed: &'a BTreeMap<String, QuantizedTensorData>,
    quantized: &'a mut Vec<QuantizedCaptureBinding>,
    config: &'a super::LlamaModelConfig,
    rope_row: NodeId,
    append_index: NodeId,
    attention_mask: NodeId,
}

impl StepLayerBuildContext<'_> {
    fn append(
        self,
        mut x: NodeId,
        layer: usize,
        past_key: NodeId,
        past_value: NodeId,
    ) -> Result<StepLayerNodes, LlamaMetalStepError> {
        let Self {
            graph,
            nodes,
            packed,
            quantized,
            config,
            rope_row,
            append_index,
            attention_mask,
        } = self;
        let schema = config.schema();
        let name = |suffix: &str| format!("blk.{layer}.{suffix}");
        let attn_norm = rms_norm(
            graph,
            x,
            nodes[&name("attn_norm.weight")],
            schema.embedding_dim(),
            config.norm_eps(),
        )?;
        let mut query = model_linear(
            graph,
            attn_norm,
            &name("attn_q.weight"),
            nodes,
            packed,
            quantized,
        )?;
        let mut key = model_linear(
            graph,
            attn_norm,
            &name("attn_k.weight"),
            nodes,
            packed,
            quantized,
        )?;
        let value = model_linear(
            graph,
            attn_norm,
            &name("attn_v.weight"),
            nodes,
            packed,
            quantized,
        )?;
        query = permute_rope_projection(
            graph,
            query,
            schema.query_heads(),
            schema.head_dim(),
            schema.rope_dim(),
            true,
        )?;
        key = permute_rope_projection(
            graph,
            key,
            schema.kv_heads(),
            schema.head_dim(),
            schema.rope_dim(),
            false,
        )?;
        query = add_bias(
            graph,
            query,
            config.qkv_bias().then(|| nodes[&name("attn_q.bias")]),
        )?;
        key = add_bias(
            graph,
            key,
            config.qkv_bias().then(|| nodes[&name("attn_k.bias")]),
        )?;
        let value = add_bias(
            graph,
            value,
            config.qkv_bias().then(|| nodes[&name("attn_v.bias")]),
        )?;
        if config.qk_norm() == LlamaQkNorm::PerProjection {
            query = rms_norm(
                graph,
                query,
                nodes[&name("attn_q_norm.weight")],
                schema.query_heads() * schema.head_dim(),
                config.norm_eps(),
            )?;
            key = rms_norm(
                graph,
                key,
                nodes[&name("attn_k_norm.weight")],
                schema.kv_heads() * schema.head_dim(),
                config.norm_eps(),
            )?;
        }
        query = graph.reshape(query, [1, 1, schema.query_heads(), schema.head_dim()])?;
        query = graph.permute(query, vec![0, 2, 1, 3])?;
        key = graph.reshape(key, [1, 1, schema.kv_heads(), schema.head_dim()])?;
        key = graph.permute(key, vec![0, 2, 1, 3])?;
        let mut value = graph.reshape(value, [1, 1, schema.kv_heads(), schema.head_dim()])?;
        value = graph.permute(value, vec![0, 2, 1, 3])?;
        if config.qk_norm() == LlamaQkNorm::PerHead {
            query = rms_norm(
                graph,
                query,
                nodes[&name("attn_q_norm.weight")],
                schema.head_dim(),
                config.norm_eps(),
            )?;
            key = rms_norm(
                graph,
                key,
                nodes[&name("attn_k_norm.weight")],
                schema.head_dim(),
                config.norm_eps(),
            )?;
        }
        let (query, key) = apply_resident_rope(graph, query, key, rope_row, schema)?;
        let (key_update, next_key) = append_cache_row(graph, past_key, append_index, key)?;
        let (value_update, next_value) = append_cache_row(graph, past_value, append_index, value)?;
        let attended = graph.scaled_dot_product_attention(
            query,
            next_key,
            next_value,
            Some(attention_mask),
            AttentionOptions {
                enable_gqa: true,
                ..AttentionOptions::default()
            },
        )?;
        let attended = graph.permute(attended, vec![0, 2, 1, 3])?;
        let attended = graph.reshape(attended, [1, 1, schema.query_heads() * schema.head_dim()])?;
        let attended = model_linear(
            graph,
            attended,
            &name("attn_output.weight"),
            nodes,
            packed,
            quantized,
        )?;
        x = graph.add(x, attended)?;

        let normalized = rms_norm(
            graph,
            x,
            nodes[&name("ffn_norm.weight")],
            schema.embedding_dim(),
            config.norm_eps(),
        )?;
        let gate = model_linear(
            graph,
            normalized,
            &name("ffn_gate.weight"),
            nodes,
            packed,
            quantized,
        )?;
        let gate = graph.silu(gate)?;
        let up = model_linear(
            graph,
            normalized,
            &name("ffn_up.weight"),
            nodes,
            packed,
            quantized,
        )?;
        let gated = graph.mul(gate, up)?;
        let down = model_linear(
            graph,
            gated,
            &name("ffn_down.weight"),
            nodes,
            packed,
            quantized,
        )?;
        x = graph.add(x, down)?;
        Ok(StepLayerNodes {
            output: x,
            key: next_key,
            value: next_value,
            key_update,
            value_update,
        })
    }
}

fn linear(graph: &mut Graph, input: NodeId, weight: NodeId) -> Result<NodeId, Error> {
    let weight = graph.permute(weight, vec![1, 0])?;
    graph.matmul(input, weight)
}

fn model_linear(
    graph: &mut Graph,
    input: NodeId,
    name: &str,
    nodes: &BTreeMap<String, NodeId>,
    packed: &BTreeMap<String, QuantizedTensorData>,
    quantized: &mut Vec<QuantizedCaptureBinding>,
) -> Result<NodeId, LlamaMetalStepError> {
    let weight = nodes[name];
    let output = linear(graph, input, weight)?;
    if let Some(value) = packed.get(name) {
        quantized.push(QuantizedCaptureBinding::Matmul {
            output,
            activation: input,
            weight,
            value: value.clone(),
        });
    }
    Ok(output)
}

fn apply_resident_rope(
    graph: &mut Graph,
    query: NodeId,
    key: NodeId,
    row: NodeId,
    schema: super::LlamaDecoderSchema,
) -> Result<(NodeId, NodeId), LlamaMetalStepError> {
    let half = schema.rope_dim() / 2;
    let cos = graph.shrink(row, vec![(0, 1), (0, half)])?;
    let sin = graph.shrink(row, vec![(0, 1), (half, schema.rope_dim())])?;
    let cos = graph.reshape(cos, [1, 1, 1, half])?;
    let sin = graph.reshape(sin, [1, 1, 1, half])?;
    Ok((
        rotate(graph, query, cos, sin, schema.rope_dim(), schema.head_dim())?,
        rotate(graph, key, cos, sin, schema.rope_dim(), schema.head_dim())?,
    ))
}

fn apply_prefill_rope(
    graph: &mut Graph,
    input: NodeId,
    rows: NodeId,
    span_rows: usize,
    schema: super::LlamaDecoderSchema,
) -> Result<NodeId, LlamaMetalStepError> {
    let half = schema.rope_dim() / 2;
    let cos = graph.shrink(rows, vec![(0, 1), (0, span_rows), (0, half)])?;
    let sin = graph.shrink(
        rows,
        vec![(0, 1), (0, span_rows), (half, schema.rope_dim())],
    )?;
    let cos = graph.reshape(cos, [1, 1, span_rows, half])?;
    let sin = graph.reshape(sin, [1, 1, span_rows, half])?;
    Ok(rotate(
        graph,
        input,
        cos,
        sin,
        schema.rope_dim(),
        schema.head_dim(),
    )?)
}

fn rotate(
    graph: &mut Graph,
    input: NodeId,
    cos: NodeId,
    sin: NodeId,
    rope_dim: usize,
    head_dim: usize,
) -> Result<NodeId, Error> {
    let shape = graph.shape(input)?.dims().to_vec();
    let half = rope_dim / 2;
    let mut first_bounds = shape
        .iter()
        .copied()
        .map(|dim| (0, dim))
        .collect::<Vec<_>>();
    first_bounds[3] = (0, half);
    let first = graph.shrink(input, first_bounds)?;
    let mut second_bounds = shape
        .iter()
        .copied()
        .map(|dim| (0, dim))
        .collect::<Vec<_>>();
    second_bounds[3] = (half, rope_dim);
    let second = graph.shrink(input, second_bounds)?;
    let first_cos = graph.mul(first, cos)?;
    let second_sin = graph.mul(second, sin)?;
    let rotated_first = graph.sub(first_cos, second_sin)?;
    let second_cos = graph.mul(second, cos)?;
    let first_sin = graph.mul(first, sin)?;
    let rotated_second = graph.add(second_cos, first_sin)?;
    let mut parts = vec![rotated_first, rotated_second];
    if rope_dim != head_dim {
        let mut tail_bounds = shape
            .iter()
            .copied()
            .map(|dim| (0, dim))
            .collect::<Vec<_>>();
        tail_bounds[3] = (rope_dim, head_dim);
        parts.push(graph.shrink(input, tail_bounds)?);
    }
    graph.concat(parts, 3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::metal::MetalCapabilities;
    use crate::{Backend, CpuBackend, LogicalOp, Op, UnaryOp};
    use std::collections::{BTreeSet, HashMap};

    fn renderer() -> MetalRenderer {
        MetalRenderer::new(
            8,
            MetalCapabilities {
                max_buffer_length: 1 << 24,
                unified_memory: true,
                family: "MockApple9".into(),
            },
        )
        .unwrap()
    }

    fn assert_close(actual: &TensorData, expected: &TensorData) {
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.dtype(), expected.dtype());
        for (&actual, &expected) in actual.values().iter().zip(expected.values()) {
            assert!(
                (actual - expected).abs() <= 3e-5,
                "{actual} differs from {expected}"
            );
        }
    }

    fn state_name(graph: &Graph, link: InferenceAppendStateLink) -> String {
        let Op::Input { name } = graph.op(link.input()).unwrap() else {
            unreachable!("append state input is authenticated by name")
        };
        name.clone()
    }

    fn execute_step_state(
        built: &BuiltStepGraph,
        states: &BTreeMap<String, TensorData>,
        token: u32,
        position: usize,
    ) -> BTreeMap<String, TensorData> {
        let mut bindings = built
            .residents
            .iter()
            .map(|(name, (_, value))| (name.clone(), value.clone()))
            .chain(states.clone())
            .collect::<HashMap<_, _>>();
        bindings.insert(
            TOKEN_INPUT.into(),
            TensorData::from_scalars([1, 1], DType::I32, [Scalar::I(i64::from(token))]).unwrap(),
        );
        bindings.insert(
            POSITION_INPUT.into(),
            TensorData::from_scalars([1], DType::I32, [Scalar::I(position as i64)]).unwrap(),
        );
        built
            .state_links
            .iter()
            .map(|link| {
                (
                    state_name(&built.graph, *link),
                    CpuBackend
                        .execute(&built.graph, link.output(), &bindings)
                        .unwrap(),
                )
            })
            .collect()
    }

    fn execute_step_tokens(
        built: &BuiltStepGraph,
        mut states: BTreeMap<String, TensorData>,
        tokens: &[u32],
        start: usize,
    ) -> BTreeMap<String, TensorData> {
        for (offset, token) in tokens.iter().copied().enumerate() {
            states = execute_step_state(built, &states, token, start + offset);
        }
        states
    }

    fn execute_prefill_state(
        built: &BuiltPrefillGraph,
        states: &BTreeMap<String, TensorData>,
        tokens: &[u32],
        start: usize,
    ) -> BTreeMap<String, TensorData> {
        let rows = tokens.len();
        let mut bindings = built
            .residents
            .iter()
            .map(|(name, (_, value))| (name.clone(), value.clone()))
            .chain(states.clone())
            .collect::<HashMap<_, _>>();
        bindings.insert(
            PREFILL_TOKEN_INPUT.into(),
            TensorData::from_scalars(
                [1, rows],
                DType::I32,
                tokens.iter().map(|token| Scalar::I(i64::from(*token))),
            )
            .unwrap(),
        );
        bindings.insert(
            POSITION_INPUT.into(),
            TensorData::from_scalars([1], DType::I32, [Scalar::I(start as i64)]).unwrap(),
        );
        bindings.insert(
            PREFILL_POSITIONS_INPUT.into(),
            TensorData::from_scalars(
                [1, rows],
                DType::I32,
                (start..start + rows).map(|position| Scalar::I(position as i64)),
            )
            .unwrap(),
        );
        built
            .state_links
            .iter()
            .map(|link| {
                (
                    state_name(&built.graph, *link),
                    CpuBackend
                        .execute(&built.graph, link.output(), &bindings)
                        .unwrap(),
                )
            })
            .collect()
    }

    fn assert_state_close(
        actual: &BTreeMap<String, TensorData>,
        expected: &BTreeMap<String, TensorData>,
    ) {
        assert_eq!(
            actual.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>()
        );
        for (name, actual) in actual {
            assert_close(actual, &expected[name]);
        }
    }

    #[test]
    fn guarded_greedy_token_uses_first_tie_and_rejects_nonfinite_logits() {
        let mut predicate = Graph::new();
        let predicate_logits = predicate.input_dtype("predicate_logits", [1, 4], DType::F32);
        let finite = finite_logit_lanes(&mut predicate, predicate_logits).unwrap();
        let Op::Logical {
            op: LogicalOp::And,
            lhs: lower,
            rhs: Some(upper),
        } = predicate.op(finite).unwrap()
        else {
            panic!("finite lanes must join two ordered bounds")
        };
        assert!(matches!(
            predicate.op(*lower).unwrap(),
            Op::Compare {
                op: CompareOp::Ge,
                lhs,
                ..
            } if *lhs == predicate_logits
        ));
        assert!(matches!(
            predicate.op(*upper).unwrap(),
            Op::Compare {
                op: CompareOp::Le,
                lhs,
                ..
            } if *lhs == predicate_logits
        ));
        assert_eq!(
            (0..predicate.node_count())
                .filter(|&index| matches!(
                    predicate.op(NodeId::from_index(index)).unwrap(),
                    Op::Compare { .. }
                ))
                .count(),
            2
        );
        assert!((0..predicate.node_count()).all(|index| !matches!(
            predicate.op(NodeId::from_index(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Abs
                    | UnaryOp::Sign
                    | UnaryOp::IsInf
                    | UnaryOp::IsNan
                    | UnaryOp::IsFinite,
                ..
            }
        )));
        assert!((0..predicate.node_count()).all(|index| matches!(
            predicate.dtype(NodeId::from_index(index)).unwrap(),
            DType::F32 | DType::Bool
        )));

        let mut graph = Graph::new();
        let logits = graph.input_dtype("logits", [1, 4], DType::F32);
        let token = guarded_greedy_token(&mut graph, logits).unwrap();
        assert!((0..graph.node_count()).all(|index| !matches!(
            graph.op(NodeId::from_index(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Abs
                    | UnaryOp::Sign
                    | UnaryOp::IsInf
                    | UnaryOp::IsNan
                    | UnaryOp::IsFinite,
                ..
            }
        )));
        assert!((0..graph.node_count()).all(|index| matches!(
            graph.dtype(NodeId::from_index(index)).unwrap(),
            DType::F32 | DType::Bool | DType::I32
        )));
        for (values, expected) in [
            (vec![1.0, 7.0, 7.0, 2.0], 1),
            (vec![1.0, f32::MAX, 3.0, 2.0], 1),
            (vec![1.0, 3.0, -f32::MAX, 2.0], 1),
            (vec![1.0, f32::NAN, 3.0, 2.0], -1),
            (vec![1.0, f32::INFINITY, 3.0, 2.0], -1),
            (vec![1.0, f32::NEG_INFINITY, 3.0, 2.0], -1),
        ] {
            let actual = CpuBackend
                .execute(
                    &graph,
                    token,
                    &HashMap::from([(
                        "logits".to_owned(),
                        TensorData::new([1, 4], values).unwrap(),
                    )]),
                )
                .unwrap();
            assert_eq!(actual.dtype(), DType::I32);
            assert_eq!(actual.len(), 1);
            assert_eq!(actual.scalar_at(0).as_i64(), expected);
        }
    }

    #[test]
    fn greedy_step_plans_publish_one_bounded_token_without_fallback() {
        let (dense, _, _) = super::super::model_tests::make_variant_model(4);
        let (packed, _, _, _) = super::super::packed_metal_fixture_models();
        for plan in [
            LlamaMetalGreedyStepPlan::new(&dense, renderer()).unwrap(),
            LlamaMetalGreedyStepPlan::new(&packed, renderer()).unwrap(),
        ] {
            assert_eq!(plan.summary().requested_output_count, 1);
            assert_eq!(plan.summary().fallback_count, 0);
            assert_eq!(
                plan.capture().requested.len(),
                plan.state_inputs().len() + 1
            );
            assert!(
                plan.capture()
                    .items
                    .iter()
                    .all(|item| item.boundary.is_none())
            );
            let iotas = plan
                .rendered_items()
                .filter(|item| item.source.contains("b0[gid] = ((int)((ulong)gid));"))
                .collect::<Vec<_>>();
            assert!(
                iotas.iter().any(|item| item.extent == plan.vocab_size()),
                "greedy plan renders its vocabulary-width I32 ShapeIota"
            );
            for iota in iotas {
                assert_eq!(iota.buffers.len(), 1);
                assert_eq!(iota.buffers[0].dtype, DType::I32);
                assert!(!iota.source.contains("device long*"));
            }
        }
    }

    #[test]
    fn dense_gqa_step_graph_uses_raw_gathers_and_owned_append_rows() {
        let (model, _, _) = super::super::model_tests::make_variant_model(4);
        let built = build_step_graph(&model).unwrap();
        assert_eq!(
            (0..built.graph.node_count())
                .filter(|&index| matches!(
                    built.graph.op(NodeId::from_index(index)).unwrap(),
                    Op::Gather { .. }
                ))
                .count(),
            2
        );
        assert_eq!(
            (0..built.graph.node_count())
                .filter(|&index| matches!(
                    built.graph.op(NodeId::from_index(index)).unwrap(),
                    Op::Scatter { add: false, .. }
                ))
                .count(),
            model.config().layer_count() * 2
        );
        assert!(built.state_links.iter().all(|link| {
            matches!(built.graph.op(link.output()).unwrap(), Op::Scatter { .. })
                && matches!(
                    built.graph.op(link.updates()).unwrap(),
                    Op::Contiguous { .. }
                )
                && link.index()
                    == built
                        .state_links
                        .first()
                        .expect("Llama has K/V state")
                        .index()
        }));
        let index = built.state_links[0].index();
        let Op::Expand {
            input: reshaped, ..
        } = built.graph.op(index).unwrap()
        else {
            panic!("append index must be the shared scalar expansion")
        };
        assert!(matches!(
            built.graph.op(*reshaped).unwrap(),
            Op::Reshape { input, .. } if *input == built.state_links[0].position()
        ));
        let mut states = built.initial_state.clone();
        let mut oracle = super::super::LlamaModelCache::new(model.config().clone());
        for (position, token) in [3u32, 4, 5].into_iter().enumerate() {
            let mut bindings = built
                .residents
                .iter()
                .map(|(name, (_, value))| (name.clone(), value.clone()))
                .chain(states.clone())
                .collect::<HashMap<_, _>>();
            bindings.insert(
                TOKEN_INPUT.into(),
                TensorData::from_scalars([1, 1], DType::I32, [Scalar::I(i64::from(token))])
                    .unwrap(),
            );
            bindings.insert(
                POSITION_INPUT.into(),
                TensorData::from_scalars([1], DType::I32, [Scalar::I(position as i64)]).unwrap(),
            );
            let actual = CpuBackend
                .execute(&built.graph, built.logits, &bindings)
                .unwrap();
            let expected = oracle.forward(&model, &[token]).unwrap();
            assert_close(&actual, &expected);
            states = built
                .state_links
                .iter()
                .map(|link| {
                    let Op::Input { name } = built.graph.op(link.input()).unwrap() else {
                        unreachable!()
                    };
                    (
                        name.clone(),
                        CpuBackend
                            .execute(&built.graph, link.output(), &bindings)
                            .unwrap(),
                    )
                })
                .collect();
        }
    }

    #[test]
    fn plans_tied_and_explicit_outputs_with_exact_resident_state_ownership() {
        let (tied, _, _) = super::super::model_tests::make_model(4);
        let tied_plan = LlamaMetalStepPlan::new(&tied, renderer()).unwrap();
        assert_eq!(
            tied_plan.output_binding(),
            LlamaOutputBinding::TiedToTokenEmbedding
        );
        assert_eq!(
            tied_plan.state_inputs().len(),
            tied.config().layer_count() * 2
        );
        assert_eq!(
            tied_plan.summary().state_pair_count,
            tied.config().layer_count() * 2
        );
        assert_eq!(tied_plan.summary().fallback_count, 0);
        assert_eq!(tied_plan.summary().requested_output_count, 1);
        assert_eq!(tied_plan.summary().state_bank_count, 1);
        assert_eq!(
            tied_plan.summary().append_state_work_items,
            tied.config().layer_count()
                * 2
                * tied.config().schema().kv_heads()
                * tied.config().schema().head_dim()
        );
        assert_eq!(
            tied_plan
                .transient_inputs()
                .iter()
                .map(|input| (
                    input.name.as_str(),
                    input.desc.dtype,
                    input.desc.shape.dims().to_vec(),
                ))
                .collect::<Vec<_>>(),
            [(TOKEN_INPUT, DType::I32, vec![1, 1])]
        );
        assert_eq!(
            tied_plan.summary().runtime_control_input_names,
            [POSITION_INPUT]
        );
        assert_eq!(tied_plan.summary().runtime_control_input_bytes, 4);
        assert_eq!(
            tied_plan
                .resident_inputs()
                .iter()
                .filter(|input| input.name == TOKEN_EMBEDDING)
                .count(),
            1
        );
        assert!(
            tied_plan
                .resident_inputs()
                .iter()
                .any(|input| input.name == ROPE_TABLE)
        );
        assert!(
            tied_plan
                .rendered_items()
                .all(|item| item.extent == 0 || !item.source.is_empty())
        );
        assert!(
            tied_plan
                .capture()
                .items
                .iter()
                .all(|item| item.boundary.is_none())
        );
        assert_eq!(
            tied_plan
                .rendered_items()
                .filter(|item| item.indexed_movement().is_some())
                .count(),
            0
        );
        assert_eq!(
            tied_plan
                .rendered_items()
                .filter(|item| item.source.contains("rg_metal_host_gather_f32_i32"))
                .count(),
            2
        );

        let explicit = super::super::model_tests::make_explicit_model(4);
        let explicit_plan = LlamaMetalStepPlan::new(&explicit, renderer()).unwrap();
        assert_eq!(explicit_plan.output_binding(), LlamaOutputBinding::Explicit);
        assert!(
            explicit_plan
                .resident_inputs()
                .iter()
                .any(|input| input.name == OUTPUT_WEIGHT)
        );
        assert_ne!(
            tied_plan.deployment_identity(),
            explicit_plan.deployment_identity()
        );
    }

    #[test]
    fn authenticated_gathers_change_append_deployment_not_graph_capture_identity() {
        let (model, _, _) = super::super::model_tests::make_variant_model(2);
        let built = build_step_graph(&model).unwrap();
        let unchecked = CapturedAppendStateInference::from_graph_residents(
            &built.graph,
            &[built.logits],
            &built.state_links,
            built.initial_state.clone(),
            built.residents.clone(),
            &built.quantized,
            &[],
        )
        .unwrap();
        let authenticated = CapturedAppendStateInference::from_graph_residents(
            &built.graph,
            &[built.logits],
            &built.state_links,
            built.initial_state,
            built.residents,
            &built.quantized,
            &[TOKEN_INPUT, POSITION_INPUT],
        )
        .unwrap()
        .seal_committed_position()
        .unwrap();
        assert_eq!(
            unchecked.capture().identity,
            authenticated.capture().identity
        );
        assert_ne!(
            unchecked.deployment_identity(),
            authenticated.deployment_identity()
        );
        assert_eq!(authenticated.transient_inputs().len(), 1);
        assert_eq!(authenticated.transient_inputs()[0].name, TOKEN_INPUT);
    }

    #[test]
    fn fixed_prefill_capture_is_state_only_with_exact_typed_schemas() {
        let (model, _, _) = super::super::model_tests::make_variant_model(6);
        let plan =
            LlamaMetalPrefillPlan::new(&model, renderer(), NonZeroUsize::new(3).unwrap()).unwrap();
        assert_eq!(plan.span_rows().get(), 3);
        assert_eq!(plan.max_context(), 6);
        assert_eq!(plan.vocab_size(), model.config().schema().vocab_size());
        assert_eq!(plan.layer_count(), model.config().layer_count());
        assert_eq!(plan.output_binding(), model.output_binding());
        assert_eq!(plan.append_state_plan().append_span_rows(), 3);
        assert_eq!(plan.summary().requested_output_count, 0);
        assert_eq!(plan.capture().requested.len(), plan.state_inputs().len());
        assert_eq!(plan.summary().fallback_count, 0);
        assert_eq!(
            plan.summary().append_state_work_items,
            model.config().layer_count()
                * 2
                * 3
                * model.config().schema().kv_heads()
                * model.config().schema().head_dim()
        );
        assert_eq!(
            plan.transient_inputs()
                .iter()
                .map(|input| (
                    input.name.as_str(),
                    input.desc.dtype,
                    input.desc.shape.dims().to_vec(),
                ))
                .collect::<Vec<_>>(),
            [
                (PREFILL_POSITIONS_INPUT, DType::I32, vec![1, 3]),
                (PREFILL_TOKEN_INPUT, DType::I32, vec![1, 3]),
            ]
        );
        assert_eq!(plan.token_input().desc.shape.dims(), [1, 3]);
        assert_eq!(plan.position_vector_input().desc.shape.dims(), [1, 3]);
        assert_eq!(plan.runtime_control_inputs().len(), 1);
        assert_eq!(plan.scalar_position_input().name, POSITION_INPUT);
        assert_eq!(plan.scalar_position_input().desc.shape.dims(), [1]);
        assert_eq!(plan.state_inputs().len(), model.config().layer_count() * 2);
        assert!(
            plan.resident_inputs()
                .iter()
                .all(|input| input.name != OUTPUT_NORM && input.name != OUTPUT_WEIGHT)
        );
        assert!(
            plan.capture()
                .items
                .iter()
                .all(|item| item.boundary.is_none())
        );
        assert!(
            plan.rendered_items()
                .all(|item| item.extent == 0 || !item.source.is_empty())
        );
        assert_eq!(
            plan.rendered_items()
                .filter(|item| item.source.contains("rg_metal_host_gather_fixed_f32_i32"))
                .count(),
            2
        );
        assert_ne!(plan.deployment_identity(), 0);
        assert!(!plan.execution_plan().items.is_empty());
        let inner = plan.into_append_state_plan();
        assert_eq!(inner.append_span_rows(), 3);
    }

    #[test]
    fn fixed_prefill_graph_authenticates_one_shared_position_plus_iota_index() {
        let (model, _, _) = super::super::model_tests::make_variant_model(6);
        let built = build_prefill_graph(&model, NonZeroUsize::new(3).unwrap()).unwrap();
        assert_eq!(built.state_links.len(), model.config().layer_count() * 2);
        let first = built.state_links[0];
        assert!(
            built
                .state_links
                .iter()
                .all(|link| link.index() == first.index() && link.position() == first.position())
        );
        let Op::Binary {
            op: crate::BinaryOp::Add,
            lhs,
            rhs,
        } = built.graph.op(first.index()).unwrap()
        else {
            panic!("fixed prefill append index must add position and ShapeIota")
        };
        let Op::Expand {
            input: position_reshape,
            shape,
        } = built.graph.op(*lhs).unwrap()
        else {
            panic!("fixed prefill position must be expanded")
        };
        assert_eq!(
            shape.dims(),
            [
                1,
                model.config().schema().kv_heads(),
                3,
                model.config().schema().head_dim(),
            ]
        );
        assert!(matches!(
            built.graph.op(*position_reshape).unwrap(),
            Op::Reshape { input, .. } if *input == first.position()
        ));
        let Op::Expand {
            input: iota_reshape,
            shape,
        } = built.graph.op(*rhs).unwrap()
        else {
            panic!("fixed prefill ShapeIota must be expanded over update lanes")
        };
        assert_eq!(shape, built.graph.shape(first.updates()).unwrap());
        let Op::Reshape { input: iota, shape } = built.graph.op(*iota_reshape).unwrap() else {
            panic!("fixed prefill ShapeIota must be axis-aligned")
        };
        assert_eq!(shape.dims(), [1, 1, 3, 1]);
        assert!(matches!(
            built.graph.op(*iota).unwrap(),
            Op::ShapeIota { source, axis: 2 } if *source == first.updates()
        ));
        assert!((0..built.graph.node_count()).any(|index| {
            let node = NodeId::from_index(index);
            matches!(built.graph.op(node), Ok(Op::Compare { .. }))
                && built
                    .graph
                    .shape(node)
                    .is_ok_and(|shape| shape.dims() == [1, 1, 3, 6])
        }));
    }

    #[test]
    fn fixed_prefill_state_matches_sequential_t1_at_zero_and_nonzero_prefix() {
        let (model, _, _) = super::super::model_tests::make_variant_model(8);
        let step = build_step_graph(&model).unwrap();
        let prefill = build_prefill_graph(&model, NonZeroUsize::new(3).unwrap()).unwrap();

        let initial = step.initial_state.clone();
        let expected = execute_step_tokens(&step, initial.clone(), &[3, 4, 5], 0);
        let actual = execute_prefill_state(&prefill, &initial, &[3, 4, 5], 0);
        assert_state_close(&actual, &expected);

        let prefix = execute_step_tokens(&step, initial, &[1, 2], 0);
        let expected = execute_step_tokens(&step, prefix.clone(), &[3, 4, 5], 2);
        let actual = execute_prefill_state(&prefill, &prefix, &[3, 4, 5], 2);
        assert_state_close(&actual, &expected);
    }

    #[test]
    fn fixed_prefill_dense_inventory_is_exact_and_typed() {
        let (model, _, _) = super::super::model_tests::make_variant_model(6);
        let plan =
            LlamaMetalPrefillPlan::new(&model, renderer(), NonZeroUsize::new(3).unwrap()).unwrap();
        let expected = BTreeSet::from([
            ATTENTION_POSITIONS,
            TOKEN_EMBEDDING,
            ROPE_TABLE,
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "blk.0.attn_q.bias",
            "blk.0.attn_k.bias",
            "blk.0.attn_v.bias",
            "blk.1.attn_norm.weight",
            "blk.1.attn_k.weight",
            "blk.1.attn_v.weight",
            "blk.1.attn_k_norm.weight",
            "blk.1.attn_k.bias",
            "blk.1.attn_v.bias",
        ]);
        let actual = plan
            .resident_inputs()
            .iter()
            .map(|input| input.name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(actual, expected);
        for input in plan.resident_inputs() {
            let expected_dtype = if input.name == ATTENTION_POSITIONS {
                DType::I32
            } else {
                DType::F32
            };
            assert_eq!(input.desc.dtype, expected_dtype, "{}", input.name);
            assert_eq!(
                input.desc.bytes,
                input.desc.shape.numel().unwrap() * expected_dtype.itemsize()
            );
        }
    }

    #[test]
    fn fixed_prefill_captures_every_supported_packed_format_without_logits() {
        let (packed, _, _, _) = super::super::packed_metal_fixture_models();
        let step = LlamaMetalStepPlan::new(&packed, renderer()).unwrap();
        let plan =
            LlamaMetalPrefillPlan::new(&packed, renderer(), NonZeroUsize::new(3).unwrap()).unwrap();
        assert!(
            plan.quantized_input_names().len()
                < step.append_state_plan().quantized_input_names().len(),
            "state-only prefill must be allowed to share a strict subset of token-step weights"
        );
        plan.append_state_plan()
            .authenticate_shared_from(step.append_state_plan())
            .unwrap();
        let formats = plan
            .capture()
            .quantized_constants
            .values()
            .map(|value| value.descriptor().ggml_type)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            formats,
            BTreeSet::from([GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q6K])
        );
        assert_eq!(plan.summary().requested_output_count, 0);
        assert_eq!(plan.capture().requested.len(), plan.state_inputs().len());
        assert_eq!(plan.summary().fallback_count, 0);
        assert_eq!(plan.transient_inputs().len(), 2);
        assert_eq!(plan.runtime_control_inputs().len(), 1);
        assert!(plan.summary().quantized_constant_count > 0);
        let expected_names = [
            TOKEN_EMBEDDING,
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.1.attn_k.weight",
            "blk.1.attn_v.weight",
        ]
        .into_iter()
        .map(|name| format!("llama.packed.{name}"))
        .collect::<BTreeSet<_>>();
        assert_eq!(
            plan.quantized_input_names()
                .values()
                .cloned()
                .collect::<BTreeSet<_>>(),
            expected_names
        );
        assert_eq!(
            plan.quantized_input_names().len(),
            plan.capture().quantized_constants.len()
        );
        for (id, name) in plan.quantized_input_names() {
            let packed = &plan.capture().quantized_constants[id];
            let bindings = plan
                .capture()
                .items
                .iter()
                .flat_map(|item| item.ordered_quantized_inputs())
                .filter(|binding| binding.input_node.index() as u64 == *id)
                .collect::<Vec<_>>();
            assert!(!bindings.is_empty(), "{name}");
            assert!(
                bindings
                    .iter()
                    .all(|binding| &binding.desc == packed.descriptor()),
                "{name}"
            );
            assert!(
                plan.resident_inputs()
                    .iter()
                    .all(|dense| dense.desc.id != *id),
                "{name}"
            );
        }
    }

    #[test]
    fn fixed_prefill_rejects_t1_and_capacity_overflow_without_changing_t1_plan() {
        let (model, _, _) = super::super::model_tests::make_variant_model(4);
        assert!(matches!(
            LlamaMetalPrefillPlan::new(&model, renderer(), NonZeroUsize::new(1).unwrap()),
            Err(LlamaMetalStepError::Dimension(
                "prefill span must exceed the existing token-step row"
            ))
        ));
        assert!(matches!(
            LlamaMetalPrefillPlan::new(&model, renderer(), NonZeroUsize::new(5).unwrap()),
            Err(LlamaMetalStepError::Dimension(
                "prefill span exceeds the fixed I32 context"
            ))
        ));
        let first = LlamaMetalStepPlan::new(&model, renderer()).unwrap();
        let second = LlamaMetalStepPlan::new(&model, renderer()).unwrap();
        assert_eq!(first.deployment_identity(), second.deployment_identity());
        assert_eq!(first.capture().identity, second.capture().identity);
        assert_eq!(first.summary(), second.summary());
        assert_eq!(
            first.summary().rendered_cache_keys,
            second.summary().rendered_cache_keys
        );
        assert_eq!(first.summary().requested_output_count, 1);
        assert_eq!(first.transient_inputs()[0].desc.shape.dims(), [1, 1]);
        assert_eq!(first.runtime_control_inputs()[0].desc.shape.dims(), [1]);
        assert!(
            first
                .rendered_items()
                .all(|item| !item.source.contains("rg_metal_host_gather_fixed_f32_i32"))
        );
    }
}
