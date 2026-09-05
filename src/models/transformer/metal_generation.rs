//! Typed prompt-to-tokens orchestration over one persistent Metal Llama session.

use super::{
    LlamaChatError, LlamaChatMessage, LlamaChatTemplate, LlamaGeneration, LlamaGenerationError,
    LlamaMetalStep, LlamaMetalStepError, LlamaMetalStepPlan, LlamaMetalStepSession,
    LlamaPromptWorkflow, LlamaSampling,
    generation::{select_last, validate_sampling},
    metal_step::LlamaMetalPrefillPlan,
    metal_workload_evidence::{LlamaMetalWorkloadEvidence, LlamaMetalWorkloadPhase},
};
use crate::{
    CapturedSchedule, DType, ExecutionPlanSummary, ReplayInput, Scalar, TensorData,
    runtime::metal::{
        MetalDevice, MetalDeviceInfo, MetalDevicePreparationReport, MetalDeviceRunReport,
        MetalDeviceSessionSummary, MetalPlanOptions, MetalScoreboardContext, MetalScoreboardError,
        MetalSessionScoreboard, MetalSharedAppendSession, RenderedMetal,
    },
    tokenizer::{SimpleTokenizer, TokenizerError},
};
use std::{collections::BTreeMap, error, fmt, num::NonZeroUsize};

/// Resource-free, GGUF-bound deployment plan for one selected Metal device.
///
/// The contained tokenizer, chat template, and captured model all originate
/// from one [`LlamaPromptWorkflow`]. The source host model is not retained.
pub struct LlamaMetalPlan {
    step: LlamaMetalStepPlan,
    prefill: Option<LlamaMetalPrefillPlan>,
    tokenizer: SimpleTokenizer,
    chat_template: LlamaChatTemplate,
    selected_device: MetalDevice,
}

/// Persistent single-sequence Metal generation session.
pub struct LlamaMetalSession {
    step: LlamaMetalStepSession,
    prefill: Option<LlamaMetalPrefillSession>,
    token_step_deployment_identity: u64,
    fixed_prefill_deployment_identity: Option<u64>,
    committed_position: usize,
    successful_invocations: u64,
    tokenizer: SimpleTokenizer,
    chat_template: LlamaChatTemplate,
}

pub(super) struct LlamaMetalPrefillSession {
    inner: MetalSharedAppendSession,
    span_rows: NonZeroUsize,
    token_input_name: String,
    position_input_name: String,
}

/// Successful prompt ingestion with only its final logits downloaded.
#[derive(Debug)]
pub struct LlamaMetalPrefill {
    logits: TensorData,
    reports: Vec<MetalDeviceRunReport>,
    start_position: usize,
    workload_evidence: LlamaMetalWorkloadEvidence,
}

/// Successful Metal generation plus an inspectable report for every committed
/// prompt or generated-token invocation.
#[derive(Debug)]
pub struct LlamaMetalGeneration {
    generation: LlamaGeneration,
    reports: Vec<MetalDeviceRunReport>,
    workload_evidence: LlamaMetalWorkloadEvidence,
}

/// Rendered prompt plus its successful generation.
pub struct LlamaMetalPromptOutput {
    rendered_prompt: String,
    generation: LlamaMetalGeneration,
}

/// Durable successful prefix accompanying a failure after Metal work began.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaMetalProgress {
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    reports: Vec<MetalDeviceRunReport>,
    start_position: usize,
    committed_position: usize,
}

/// Token-execution phase recorded by a partial failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlamaMetalGenerationStage {
    Prompt,
    Decode,
}

/// Typed planning, preflight, execution, or post-execution decoding failure.
#[derive(Clone, Debug, PartialEq)]
pub enum LlamaMetalGenerationError {
    Step(LlamaMetalStepError),
    Scoreboard(MetalScoreboardError),
    Generation(LlamaGenerationError),
    Tokenizer(TokenizerError),
    Chat(LlamaChatError),
    FreshSessionRequired {
        position: usize,
    },
    Execution {
        progress: Box<LlamaMetalProgress>,
        stage: LlamaMetalGenerationStage,
        token_offset: usize,
        token: u32,
        source: LlamaMetalStepError,
    },
    PrefillChunkExecution {
        progress: Box<LlamaMetalProgress>,
        token_offset: usize,
        span_rows: usize,
        source: LlamaMetalStepError,
    },
    Decode {
        progress: Box<LlamaMetalProgress>,
        stage: LlamaMetalGenerationStage,
        token_offset: usize,
        token: u32,
        source: TokenizerError,
    },
    PostExecution {
        progress: Box<LlamaMetalProgress>,
        stage: LlamaMetalGenerationStage,
        token_offset: usize,
        source: LlamaGenerationError,
    },
}

impl fmt::Display for LlamaMetalGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llama Metal generation error: {self:?}")
    }
}

impl error::Error for LlamaMetalGenerationError {}

impl From<LlamaMetalStepError> for LlamaMetalGenerationError {
    fn from(value: LlamaMetalStepError) -> Self {
        Self::Step(value)
    }
}

impl From<MetalScoreboardError> for LlamaMetalGenerationError {
    fn from(value: MetalScoreboardError) -> Self {
        Self::Scoreboard(value)
    }
}

impl From<LlamaGenerationError> for LlamaMetalGenerationError {
    fn from(value: LlamaGenerationError) -> Self {
        Self::Generation(value)
    }
}

impl From<TokenizerError> for LlamaMetalGenerationError {
    fn from(value: TokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}

impl From<LlamaChatError> for LlamaMetalGenerationError {
    fn from(value: LlamaChatError) -> Self {
        Self::Chat(value)
    }
}

impl LlamaMetalPlan {
    /// Consumes one GGUF-bound workflow into an inspectable resource-free plan
    /// for the explicitly selected Metal device.
    pub fn from_workflow(
        workflow: LlamaPromptWorkflow,
        device: &MetalDevice,
        options: MetalPlanOptions,
    ) -> Result<Self, LlamaMetalGenerationError> {
        let (model, tokenizer, chat_template) = workflow.into_parts();
        let renderer = device
            .renderer(options.local_size)
            .map_err(LlamaMetalStepError::Metal)?;
        let step = LlamaMetalStepPlan::new(&model, renderer)?;
        Ok(Self {
            step,
            prefill: None,
            tokenizer,
            chat_template,
            selected_device: device.clone(),
        })
    }

    /// Builds the exact token-step deployment plus one state-only fixed-span
    /// prompt program. The latter is used only for complete chunks preceding
    /// the prompt's final token.
    pub fn from_workflow_with_prefill_span(
        workflow: LlamaPromptWorkflow,
        device: &MetalDevice,
        options: MetalPlanOptions,
        span_rows: NonZeroUsize,
    ) -> Result<Self, LlamaMetalGenerationError> {
        if span_rows.get() == 1 {
            return Self::from_workflow(workflow, device, options);
        }
        let (model, tokenizer, chat_template) = workflow.into_parts();
        let renderer = device
            .renderer(options.local_size)
            .map_err(LlamaMetalStepError::Metal)?;
        let step = LlamaMetalStepPlan::new(&model, renderer.clone())?;
        let prefill = LlamaMetalPrefillPlan::new(&model, renderer, span_rows)?;
        authenticate_prefill_pair(&step, &prefill)?;
        Ok(Self {
            step,
            prefill: Some(prefill),
            tokenizer,
            chat_template,
            selected_device: device.clone(),
        })
    }

    /// Returns the opt-in fixed prompt span. `None` retains the exact T1-only
    /// deployment and execution path.
    pub fn prefill_span_rows(&self) -> Option<NonZeroUsize> {
        self.prefill.as_ref().map(LlamaMetalPrefillPlan::span_rows)
    }

    /// Returns the state-only prefill capture when fixed-span prefill is enabled.
    pub fn prefill_capture(&self) -> Option<&CapturedSchedule> {
        self.prefill.as_ref().map(LlamaMetalPrefillPlan::capture)
    }

    /// Returns the fixed-span program's resource and execution facts.
    pub fn prefill_summary(&self) -> Option<&MetalDeviceSessionSummary> {
        self.prefill.as_ref().map(LlamaMetalPrefillPlan::summary)
    }

    /// Returns the fixed-span program's deployment identity when enabled.
    pub fn prefill_deployment_identity(&self) -> Option<u64> {
        self.prefill
            .as_ref()
            .map(LlamaMetalPrefillPlan::deployment_identity)
    }

    /// Returns backend-neutral schedule and memory facts for the state-only
    /// fixed-span program.
    pub fn prefill_execution_plan(&self) -> Option<&ExecutionPlanSummary> {
        self.prefill
            .as_ref()
            .map(LlamaMetalPrefillPlan::execution_plan)
    }

    /// Returns rendered state-only prefill items when fixed-span prefill is enabled.
    pub fn prefill_rendered_items(&self) -> Option<impl ExactSizeIterator<Item = &RenderedMetal>> {
        self.prefill
            .as_ref()
            .map(LlamaMetalPrefillPlan::rendered_items)
    }

    /// Returns immutable selected-device information without a raw handle.
    pub fn selected_device_info(&self) -> &MetalDeviceInfo {
        self.selected_device.info()
    }

    /// Returns the stable selected-device owner identity rechecked at prepare.
    pub fn selected_device_owner_id(&self) -> u64 {
        self.selected_device.owner_id()
    }

    /// Returns the compute-step deployment identity. Tokenizer and chat policy
    /// are deliberately outside this capture/cache identity.
    pub fn step_deployment_identity(&self) -> u64 {
        self.step.deployment_identity()
    }

    /// Returns the authenticated token-step capture.
    pub fn capture(&self) -> &CapturedSchedule {
        self.step.capture()
    }

    /// Returns backend-neutral schedule and memory-plan facts.
    pub fn execution_plan(&self) -> &ExecutionPlanSummary {
        self.step.execution_plan()
    }

    /// Returns immutable model and RoPE resident schemas.
    pub fn resident_inputs(&self) -> &[ReplayInput] {
        self.step.resident_inputs()
    }

    /// Returns the ordered K/V state-input schemas.
    pub fn state_inputs(&self) -> &[ReplayInput] {
        self.step.state_inputs()
    }

    /// Returns the token-only caller input schema.
    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.step.transient_inputs()
    }

    /// Returns the sealed position schema synthesized by the session.
    pub fn runtime_control_inputs(&self) -> &[ReplayInput] {
        self.step.runtime_control_inputs()
    }

    /// Returns deterministic resource and execution planning facts.
    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.step.summary()
    }

    /// Returns the fixed K/V capacity.
    pub fn max_context(&self) -> usize {
        self.step.max_context()
    }

    /// Returns the exact GGUF vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.step.vocab_size()
    }

    /// Returns every rendered token-step item for inspection.
    pub fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.step.rendered_items()
    }

    /// Creates the persistent device session. No host model or weight object is
    /// retained after this transition.
    pub fn prepare(self) -> Result<LlamaMetalSession, LlamaMetalGenerationError> {
        let token_step_deployment_identity = self.step.deployment_identity();
        let fixed_prefill_deployment_identity = self
            .prefill
            .as_ref()
            .map(LlamaMetalPrefillPlan::deployment_identity);
        let shared = self
            .prefill
            .as_ref()
            .map(|prefill| {
                prefill
                    .append_state_plan()
                    .authenticate_shared_from(self.step.append_state_plan())
            })
            .transpose()
            .map_err(LlamaMetalStepError::Metal)?;
        let step = self.step.prepare(self.selected_device.clone())?;
        let prefill = match (self.prefill, shared) {
            (Some(prefill), Some(proof)) => {
                let span_rows = prefill.span_rows();
                let token_input_name = prefill.token_input().name.clone();
                let position_input_name = prefill.position_vector_input().name.clone();
                let inner = prefill
                    .into_append_state_plan()
                    .prepare_shared(self.selected_device, step.metal_session(), proof)
                    .map_err(LlamaMetalStepError::Metal)?;
                Some(LlamaMetalPrefillSession::new(
                    inner,
                    span_rows,
                    token_input_name,
                    position_input_name,
                ))
            }
            (None, None) => None,
            _ => unreachable!("fixed prefill proof follows the optional plan"),
        };
        Ok(LlamaMetalSession {
            step,
            prefill,
            token_step_deployment_identity,
            fixed_prefill_deployment_identity,
            committed_position: 0,
            successful_invocations: 0,
            tokenizer: self.tokenizer,
            chat_template: self.chat_template,
        })
    }

    /// Creates the persistent device session with an opt-in v4 recorder bound
    /// before the first token can execute.
    pub fn prepare_with_scoreboard(
        self,
        context: MetalScoreboardContext,
    ) -> Result<LlamaMetalSession, LlamaMetalGenerationError> {
        if let Some(prefill) = &self.prefill {
            return Err(MetalScoreboardError::UnsupportedAppendSpan {
                span_rows: prefill.span_rows().get(),
            }
            .into());
        }
        let recorder = MetalSessionScoreboard::try_new_append_state_v4(
            context,
            self.step.append_state_plan(),
        )?;
        let token_step_deployment_identity = self.step.deployment_identity();
        let mut step = self.step.prepare(self.selected_device)?;
        step.bind_execution_scoreboard(recorder)?;
        Ok(LlamaMetalSession {
            step,
            prefill: None,
            token_step_deployment_identity,
            fixed_prefill_deployment_identity: None,
            committed_position: 0,
            successful_invocations: 0,
            tokenizer: self.tokenizer,
            chat_template: self.chat_template,
        })
    }
}

impl LlamaMetalPrefill {
    /// Returns the final prompt token's `[1,vocab]` logits.
    pub fn logits(&self) -> &TensorData {
        &self.logits
    }

    /// Returns one successful device report per device invocation. A fixed
    /// prompt chunk contributes one report for all rows in that chunk.
    pub fn reports(&self) -> &[MetalDeviceRunReport] {
        &self.reports
    }

    /// Returns the committed position before prompt ingestion began.
    pub const fn start_position(&self) -> usize {
        self.start_position
    }

    /// Returns host-observed plan, preparation, prompt-prefill, and first-run
    /// evidence for this completed prompt ingestion.
    pub fn workload_evidence(&self) -> &LlamaMetalWorkloadEvidence {
        &self.workload_evidence
    }

    /// Consumes the result into final logits and ordered reports.
    pub fn into_parts(self) -> (TensorData, Vec<MetalDeviceRunReport>) {
        (self.logits, self.reports)
    }
}

impl LlamaMetalGeneration {
    pub(super) fn from_parts_with_evidence(
        generation: LlamaGeneration,
        reports: Vec<MetalDeviceRunReport>,
        workload_evidence: LlamaMetalWorkloadEvidence,
    ) -> Self {
        Self {
            generation,
            reports,
            workload_evidence,
        }
    }

    /// Returns the backend-independent generation result.
    pub fn generation(&self) -> &LlamaGeneration {
        &self.generation
    }

    /// Returns the encoded prompt IDs.
    pub fn prompt_ids(&self) -> &[u32] {
        self.generation.prompt_ids()
    }

    /// Returns selected IDs, including a final EOS/EOT when stopped.
    pub fn generated_ids(&self) -> &[u32] {
        self.generation.generated_ids()
    }

    /// Returns incremental tokenizer decoding of the selected IDs.
    pub fn decoded(&self) -> &str {
        self.generation.decoded()
    }

    /// Returns whether EOS/EOT stopped generation before the bound.
    pub const fn stopped(&self) -> bool {
        self.generation.stopped()
    }

    /// Returns reports for successful token invocations only.
    pub fn reports(&self) -> &[MetalDeviceRunReport] {
        &self.reports
    }

    /// Returns host-observed plan, preparation, prompt-prefill, first-run,
    /// and steady-decode evidence for this completed generation.
    pub fn workload_evidence(&self) -> &LlamaMetalWorkloadEvidence {
        &self.workload_evidence
    }

    /// Consumes the result into generation and ordered reports.
    pub fn into_parts(self) -> (LlamaGeneration, Vec<MetalDeviceRunReport>) {
        (self.generation, self.reports)
    }
}

impl LlamaMetalPromptOutput {
    pub(super) fn from_parts(rendered_prompt: String, generation: LlamaMetalGeneration) -> Self {
        Self {
            rendered_prompt,
            generation,
        }
    }

    /// Returns the plain or chat-rendered prompt submitted for tokenization.
    pub fn rendered_prompt(&self) -> &str {
        &self.rendered_prompt
    }

    /// Returns selected tokens, decoded text, and device reports.
    pub fn generation(&self) -> &LlamaMetalGeneration {
        &self.generation
    }

    /// Consumes the output into its rendered prompt and generation.
    pub fn into_parts(self) -> (String, LlamaMetalGeneration) {
        (self.rendered_prompt, self.generation)
    }
}

impl LlamaMetalProgress {
    /// Returns the complete validated prompt supplied to the call.
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }

    /// Returns every token selected successfully before failure.
    pub fn generated_ids(&self) -> &[u32] {
        &self.generated_ids
    }

    /// Returns reports for token invocations committed before failure.
    pub fn reports(&self) -> &[MetalDeviceRunReport] {
        &self.reports
    }

    /// Returns the device position at the start of this operation.
    pub const fn start_position(&self) -> usize {
        self.start_position
    }

    /// Returns the device position retained after failure.
    pub const fn committed_position(&self) -> usize {
        self.committed_position
    }
}

impl LlamaMetalSession {
    /// Returns the bound token-execution recorder, or `None` for ordinary
    /// preparation. Recording never changes a successful execution result.
    pub fn execution_scoreboard(&self) -> Option<&MetalSessionScoreboard> {
        self.step.execution_scoreboard()
    }

    /// Returns the first fail-soft recording error, if one occurred.
    pub fn scoreboard_recording_error(&self) -> Option<&MetalScoreboardError> {
        self.step.scoreboard_recording_error()
    }

    #[cfg(test)]
    pub(crate) fn inject_scoreboard_recording_error(&mut self, error: MetalScoreboardError) {
        self.step.inject_scoreboard_recording_error(error);
    }

    #[cfg(test)]
    pub(crate) fn scoreboard_record_attempts(&self) -> Option<usize> {
        self.step.scoreboard_record_attempts()
    }

    /// Returns immutable selected-device information without a raw handle.
    pub fn device_info(&self) -> &MetalDeviceInfo {
        self.step.metal_session().device_info()
    }

    /// Returns the stable selected-device owner identity.
    pub fn device_owner_id(&self) -> u64 {
        self.step.metal_session().device_owner_id()
    }

    /// Returns deterministic resource and execution planning facts.
    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.step.metal_session().summary()
    }

    /// Returns one-time resource and pipeline preparation facts.
    pub fn preparation_report(&self) -> &MetalDevicePreparationReport {
        self.step.metal_session().preparation_report()
    }

    /// Returns the fixed-span program's one-time preparation report.
    pub fn prefill_preparation_report(&self) -> Option<&MetalDevicePreparationReport> {
        self.prefill
            .as_ref()
            .map(|prefill| prefill.inner.preparation_report())
    }

    /// Returns the prepared fixed-span program's deterministic plan facts.
    pub fn prefill_summary(&self) -> Option<&MetalDeviceSessionSummary> {
        self.prefill.as_ref().map(|prefill| prefill.inner.summary())
    }

    /// Returns the authenticated token-step deployment identity.
    pub const fn token_step_deployment_identity(&self) -> u64 {
        self.token_step_deployment_identity
    }

    /// Returns the authenticated fixed-span deployment identity when enabled.
    pub const fn fixed_prefill_deployment_identity(&self) -> Option<u64> {
        self.fixed_prefill_deployment_identity
    }

    /// Returns the configured fixed prompt span.
    pub fn prefill_span_rows(&self) -> Option<NonZeroUsize> {
        self.prefill.as_ref().map(|prefill| prefill.span_rows)
    }

    /// Returns compiled state-only prefill kernels when fixed-span prefill is enabled.
    pub fn compiled_prefill_kernels(&self) -> Option<impl Iterator<Item = &RenderedMetal>> {
        self.prefill
            .as_ref()
            .map(|prefill| prefill.inner.compiled_kernels())
    }

    /// Returns the nonzero compiled kernels retained by this session.
    pub fn compiled_kernels(&self) -> impl Iterator<Item = &RenderedMetal> {
        self.step.metal_session().compiled_kernels()
    }

    /// Returns the authenticated token-step capture.
    pub fn capture(&self) -> &CapturedSchedule {
        self.step.metal_session().capture()
    }

    /// Returns immutable model and RoPE resident schemas.
    pub fn resident_inputs(&self) -> &[ReplayInput] {
        self.step.metal_session().resident_inputs()
    }

    /// Returns the ordered K/V state-input schemas.
    pub fn state_inputs(&self) -> &[ReplayInput] {
        self.step.metal_session().state_inputs()
    }

    /// Returns the token-only caller input schema.
    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.step.metal_session().transient_inputs()
    }

    /// Returns the sealed position schema synthesized by the session.
    pub fn runtime_control_inputs(&self) -> &[ReplayInput] {
        self.step.metal_session().runtime_control_inputs()
    }

    /// Returns the next K/V row to commit.
    pub fn position(&self) -> usize {
        self.committed_position
    }

    /// Returns the number of successful device invocations across both the
    /// fixed-span prefill and token-step programs.
    pub fn successful_invocation_count(&self) -> u64 {
        self.successful_invocations
    }

    /// Returns the fixed K/V capacity.
    pub fn max_context(&self) -> usize {
        self.step.max_context()
    }

    /// Returns the exact GGUF vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.step.vocab_size()
    }

    /// Returns whether no further token can be committed.
    pub fn is_full(&self) -> bool {
        self.committed_position == self.step.max_context()
    }

    /// Executes and commits one token while retaining its logits.
    pub fn run_token(&mut self, token: u32) -> Result<LlamaMetalStep, LlamaMetalGenerationError> {
        let invocation = self.next_invocation()?;
        let step = self
            .step
            .run_token_at(token, self.committed_position, invocation)?;
        self.committed_position =
            step.position()
                .checked_add(1)
                .ok_or(LlamaMetalStepError::Dimension(
                    "committed position overflow",
                ))?;
        self.successful_invocations = invocation;
        Ok(step)
    }

    /// Validates the complete prefix before the first driver call, then retains
    /// only the final token's logits on the host.
    pub fn prefill_ids(
        &mut self,
        prompt_ids: &[u32],
    ) -> Result<LlamaMetalPrefill, LlamaMetalGenerationError> {
        self.preflight_tokens(prompt_ids, 0)?;
        let start_position = self.position();
        let mut reports = Vec::with_capacity(prompt_ids.len());
        let prefix_end = prompt_ids.len() - 1;
        let mut offset = 0usize;
        if let Some(span) = self.prefill.as_ref().map(|prefill| prefill.span_rows.get()) {
            while offset
                .checked_add(span)
                .is_some_and(|end| end <= prefix_end)
            {
                let end = offset + span;
                let start = self.committed_position;
                let invocation = self.next_invocation()?;
                let report = self
                    .prefill
                    .as_mut()
                    .expect("fixed-span loop requires the configured prefill program")
                    .run(&prompt_ids[offset..end], start, invocation)
                    .map_err(|source| LlamaMetalGenerationError::PrefillChunkExecution {
                        progress: Box::new(progress(
                            prompt_ids,
                            &[],
                            &reports,
                            start_position,
                            self.committed_position,
                        )),
                        token_offset: offset,
                        span_rows: span,
                        source,
                    })?;
                self.committed_position = start
                    .checked_add(span)
                    .ok_or(LlamaGenerationError::ContextOverflow)?;
                self.successful_invocations = invocation;
                reports.push(report);
                offset = end;
            }
        }
        for (tail_offset, &token) in prompt_ids[offset..prefix_end].iter().enumerate() {
            let token_offset = offset + tail_offset;
            let invocation = self.next_invocation()?;
            let commit = self
                .step
                .commit_token_at(token, self.committed_position, invocation)
                .map_err(|source| LlamaMetalGenerationError::Execution {
                    progress: Box::new(progress(
                        prompt_ids,
                        &[],
                        &reports,
                        start_position,
                        self.position(),
                    )),
                    stage: LlamaMetalGenerationStage::Prompt,
                    token_offset,
                    token,
                    source,
                })?;
            let (_, report) = commit.into_parts();
            self.committed_position = self
                .committed_position
                .checked_add(1)
                .ok_or(LlamaGenerationError::ContextOverflow)?;
            self.successful_invocations = invocation;
            reports.push(report);
        }
        let offset = prefix_end;
        let token = prompt_ids[offset];
        let invocation = self.next_invocation()?;
        let step = self
            .step
            .run_token_at(token, self.committed_position, invocation)
            .map_err(|source| LlamaMetalGenerationError::Execution {
                progress: Box::new(progress(
                    prompt_ids,
                    &[],
                    &reports,
                    start_position,
                    self.position(),
                )),
                stage: LlamaMetalGenerationStage::Prompt,
                token_offset: offset,
                token,
                source,
            })?;
        let (logits, report) = step.into_parts();
        self.committed_position = self
            .committed_position
            .checked_add(1)
            .ok_or(LlamaGenerationError::ContextOverflow)?;
        self.successful_invocations = invocation;
        reports.push(report);
        Ok(LlamaMetalPrefill {
            logits,
            workload_evidence: self.workload_evidence(
                self.first_prompt_invocation_token_count(prompt_ids.len()),
                prompt_ids.len(),
                &reports,
                0,
                &[],
            ),
            reports,
            start_position,
        })
    }

    /// Generates from explicit IDs on a fresh session. Successful prefix steps
    /// remain committed if a later device transaction fails; the failing token
    /// never advances the committed position.
    pub fn generate_ids(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        sampling: LlamaSampling<'_>,
    ) -> Result<LlamaMetalGeneration, LlamaMetalGenerationError> {
        self.require_fresh()?;
        self.preflight_tokens(prompt_ids, max_new_tokens)?;
        validate_sampling(sampling, max_new_tokens, self.step.vocab_size())?;
        if max_new_tokens == 0 {
            return Ok(LlamaMetalGeneration {
                generation: LlamaGeneration::from_parts(
                    prompt_ids.to_vec(),
                    Vec::new(),
                    String::new(),
                    false,
                ),
                reports: Vec::new(),
                workload_evidence: self.workload_evidence(0, 0, &[], 0, &[]),
            });
        }

        let prefill = self.prefill_ids(prompt_ids)?;
        let start_position = prefill.start_position();
        let LlamaMetalPrefill {
            mut logits,
            mut reports,
            ..
        } = prefill;
        let prompt_report_count = reports.len();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut decoded = String::new();
        let mut decoder = self.tokenizer.stream_decoder();
        let mut stopped = false;
        for step_index in 0..max_new_tokens {
            let token = select_last(&logits, sampling, step_index, self.step.vocab_size())
                .map_err(|source| LlamaMetalGenerationError::PostExecution {
                    progress: Box::new(progress(
                        prompt_ids,
                        &generated,
                        &reports,
                        start_position,
                        self.position(),
                    )),
                    stage: LlamaMetalGenerationStage::Decode,
                    token_offset: prompt_ids.len() + step_index,
                    source,
                })?;
            generated.push(token);
            decoded.push_str(&decoder.push(token).map_err(|source| {
                LlamaMetalGenerationError::Decode {
                    progress: Box::new(progress(
                        prompt_ids,
                        &generated,
                        &reports,
                        start_position,
                        self.position(),
                    )),
                    stage: LlamaMetalGenerationStage::Decode,
                    token_offset: prompt_ids.len() + step_index,
                    token,
                    source,
                }
            })?);
            if self.tokenizer.is_end(token) {
                stopped = true;
                break;
            }
            if step_index + 1 < max_new_tokens {
                let offset = prompt_ids.len() + step_index;
                let invocation = self.next_invocation()?;
                let next = self
                    .step
                    .run_token_at(token, self.committed_position, invocation)
                    .map_err(|source| LlamaMetalGenerationError::Execution {
                        progress: Box::new(progress(
                            prompt_ids,
                            &generated,
                            &reports,
                            start_position,
                            self.position(),
                        )),
                        stage: LlamaMetalGenerationStage::Decode,
                        token_offset: offset,
                        token,
                        source,
                    })?;
                let (next_logits, report) = next.into_parts();
                self.committed_position = self
                    .committed_position
                    .checked_add(1)
                    .ok_or(LlamaGenerationError::ContextOverflow)?;
                self.successful_invocations = invocation;
                logits = next_logits;
                reports.push(report);
            }
        }
        decoded.push_str(&decoder.finish());
        Ok(LlamaMetalGeneration {
            generation: LlamaGeneration::from_parts(
                prompt_ids.to_vec(),
                generated,
                decoded,
                stopped,
            ),
            workload_evidence: self.workload_evidence(
                self.first_prompt_invocation_token_count(prompt_ids.len()),
                prompt_ids.len(),
                &reports[..prompt_report_count],
                reports.len() - prompt_report_count,
                &reports[prompt_report_count..],
            ),
            reports,
        })
    }

    /// Encodes and generates from one plain-text prompt on a fresh session.
    pub fn generate_text(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        sampling: LlamaSampling<'_>,
    ) -> Result<LlamaMetalPromptOutput, LlamaMetalGenerationError> {
        let prompt_ids = self.encode_prompt(prompt)?;
        let generation = self.generate_ids(&prompt_ids, max_new_tokens, sampling)?;
        Ok(LlamaMetalPromptOutput {
            rendered_prompt: prompt.to_owned(),
            generation,
        })
    }

    /// Renders the checked chat template with a generation prompt, then
    /// generates on a fresh session.
    pub fn generate_chat(
        &mut self,
        messages: &[LlamaChatMessage],
        max_new_tokens: usize,
        sampling: LlamaSampling<'_>,
    ) -> Result<LlamaMetalPromptOutput, LlamaMetalGenerationError> {
        let rendered_prompt = self.chat_template.render(&self.tokenizer, messages, true)?;
        let prompt_ids = self.encode_prompt(&rendered_prompt)?;
        let generation = self.generate_ids(&prompt_ids, max_new_tokens, sampling)?;
        Ok(LlamaMetalPromptOutput {
            rendered_prompt,
            generation,
        })
    }

    fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>, LlamaMetalGenerationError> {
        let mut prompt_ids = self.tokenizer.encode(prompt)?;
        if let Some(bos) = self.tokenizer.bos_id() {
            prompt_ids.insert(0, bos);
        }
        Ok(prompt_ids)
    }

    fn require_fresh(&self) -> Result<(), LlamaMetalGenerationError> {
        let position = self.position();
        if position != 0 {
            return Err(LlamaMetalGenerationError::FreshSessionRequired { position });
        }
        Ok(())
    }

    fn next_invocation(&self) -> Result<u64, LlamaMetalGenerationError> {
        self.successful_invocations
            .checked_add(1)
            .ok_or_else(|| LlamaMetalStepError::Dimension("invocation counter overflow").into())
    }

    fn workload_evidence(
        &self,
        first_successful_run_token_count: usize,
        prompt_token_count: usize,
        prompt_reports: &[MetalDeviceRunReport],
        decode_token_count: usize,
        decode_reports: &[MetalDeviceRunReport],
    ) -> LlamaMetalWorkloadEvidence {
        build_workload_evidence(LlamaMetalEvidenceInputs {
            token_step_deployment_identity: self.token_step_deployment_identity,
            fixed_prefill_deployment_identity: self.fixed_prefill_deployment_identity,
            plan: self.summary(),
            fixed_prefill_plan: self.prefill_summary(),
            token_step_preparation: self.preparation_report(),
            fixed_prefill_preparation: self.prefill_preparation_report(),
            first_successful_run_token_count,
            prompt_token_count,
            prompt_reports,
            decode_token_count,
            decode_reports,
        })
    }

    fn first_prompt_invocation_token_count(&self, prompt_token_count: usize) -> usize {
        let prefix_token_count = prompt_token_count.saturating_sub(1);
        self.prefill_span_rows()
            .map(NonZeroUsize::get)
            .filter(|span_rows| *span_rows <= prefix_token_count)
            .unwrap_or(1)
    }

    fn preflight_tokens(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<(), LlamaMetalGenerationError> {
        if prompt_ids.is_empty() {
            return Err(LlamaGenerationError::EmptyPrompt.into());
        }
        let requested = self
            .position()
            .checked_add(prompt_ids.len())
            .and_then(|value| value.checked_add(max_new_tokens))
            .ok_or(LlamaGenerationError::ContextOverflow)?;
        if requested > self.step.max_context() {
            return Err(LlamaGenerationError::ContextLength {
                requested,
                maximum: self.step.max_context(),
            }
            .into());
        }
        if let Some(&token) = prompt_ids
            .iter()
            .find(|&&token| token as usize >= self.step.vocab_size())
        {
            return Err(LlamaMetalStepError::TokenOutOfRange {
                token,
                vocab_size: self.step.vocab_size(),
            }
            .into());
        }
        Ok(())
    }
}

fn authenticate_prefill_pair(
    step: &LlamaMetalStepPlan,
    prefill: &LlamaMetalPrefillPlan,
) -> Result<(), LlamaMetalStepError> {
    let [step_position] = step.runtime_control_inputs() else {
        return Err(LlamaMetalStepError::Metal(
            crate::runtime::metal::MetalError::InvalidBinding(
                "token-step Llama position control is not an exact singleton".into(),
            ),
        ));
    };
    if prefill.max_context() != step.max_context()
        || prefill.vocab_size() != step.vocab_size()
        || prefill.layer_count() != step.layer_count()
        || prefill.output_binding() != step.output_binding()
        || prefill.summary().requested_output_count != 0
        || prefill.scalar_position_input().name != step_position.name
    {
        return Err(LlamaMetalStepError::Metal(
            crate::runtime::metal::MetalError::InvalidBinding(
                "fixed-span and token-step Llama deployments differ".into(),
            ),
        ));
    }
    Ok(())
}

impl LlamaMetalPrefillSession {
    pub(super) fn new(
        inner: MetalSharedAppendSession,
        span_rows: NonZeroUsize,
        token_input_name: String,
        position_input_name: String,
    ) -> Self {
        Self {
            inner,
            span_rows,
            token_input_name,
            position_input_name,
        }
    }

    pub(super) const fn span_rows(&self) -> NonZeroUsize {
        self.span_rows
    }

    pub(super) fn summary(&self) -> &MetalDeviceSessionSummary {
        self.inner.summary()
    }

    pub(super) fn preparation_report(&self) -> &MetalDevicePreparationReport {
        self.inner.preparation_report()
    }

    pub(super) fn compiled_kernels(&self) -> impl Iterator<Item = &RenderedMetal> {
        self.inner.compiled_kernels()
    }

    pub(super) fn run(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        invocation: u64,
    ) -> Result<MetalDeviceRunReport, LlamaMetalStepError> {
        let span = self.span_rows.get();
        if tokens.len() != span {
            return Err(LlamaMetalStepError::Dimension(
                "fixed prefill token span differs from its plan",
            ));
        }
        let token_values = tokens
            .iter()
            .map(|token| Scalar::I(i64::from(*token)))
            .collect::<Vec<_>>();
        let position_values = (0..span)
            .map(|offset| {
                start_position
                    .checked_add(offset)
                    .and_then(|position| i32::try_from(position).ok())
                    .map(|position| Scalar::I(i64::from(position)))
                    .ok_or(LlamaMetalStepError::Dimension(
                        "fixed prefill position exceeds I32",
                    ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let inputs = BTreeMap::from([
            (
                self.token_input_name.clone(),
                TensorData::from_scalars([1, span], DType::I32, token_values)?,
            ),
            (
                self.position_input_name.clone(),
                TensorData::from_scalars([1, span], DType::I32, position_values)?,
            ),
        ]);
        let run = self
            .inner
            .run_without_host_outputs_at(&inputs, start_position)?;
        debug_assert!(run.outputs().is_empty());
        let (_, mut report) = run.into_parts();
        report.successful_invocation = invocation;
        report.first_successful_run = invocation == 1;
        Ok(report)
    }
}

pub(super) fn progress(
    prompt_ids: &[u32],
    generated_ids: &[u32],
    reports: &[MetalDeviceRunReport],
    start_position: usize,
    committed_position: usize,
) -> LlamaMetalProgress {
    LlamaMetalProgress {
        prompt_ids: prompt_ids.to_vec(),
        generated_ids: generated_ids.to_vec(),
        reports: reports.to_vec(),
        start_position,
        committed_position,
    }
}

pub(super) struct LlamaMetalEvidenceInputs<'a> {
    pub token_step_deployment_identity: u64,
    pub fixed_prefill_deployment_identity: Option<u64>,
    pub plan: &'a MetalDeviceSessionSummary,
    pub fixed_prefill_plan: Option<&'a MetalDeviceSessionSummary>,
    pub token_step_preparation: &'a MetalDevicePreparationReport,
    pub fixed_prefill_preparation: Option<&'a MetalDevicePreparationReport>,
    pub first_successful_run_token_count: usize,
    pub prompt_token_count: usize,
    pub prompt_reports: &'a [MetalDeviceRunReport],
    pub decode_token_count: usize,
    pub decode_reports: &'a [MetalDeviceRunReport],
}

pub(super) fn build_workload_evidence(
    input: LlamaMetalEvidenceInputs<'_>,
) -> LlamaMetalWorkloadEvidence {
    let first_successful_run = input
        .prompt_reports
        .iter()
        .chain(input.decode_reports)
        .find(|report| report.first_successful_run)
        .map(|report| {
            LlamaMetalWorkloadPhase::from_reports(
                input.first_successful_run_token_count,
                std::slice::from_ref(report),
            )
        });
    LlamaMetalWorkloadEvidence {
        token_step_deployment_identity: input.token_step_deployment_identity,
        fixed_prefill_deployment_identity: input.fixed_prefill_deployment_identity,
        plan: input.plan.clone(),
        fixed_prefill_plan: input.fixed_prefill_plan.cloned(),
        token_step_preparation: input.token_step_preparation.clone(),
        fixed_prefill_preparation: input.fixed_prefill_preparation.cloned(),
        first_successful_run,
        prompt_prefill: LlamaMetalWorkloadPhase::from_reports(
            input.prompt_token_count,
            input.prompt_reports,
        ),
        steady_decode: LlamaMetalWorkloadPhase::from_reports(
            input.decode_token_count,
            input.decode_reports,
        ),
    }
}
