//! Typed prompt-to-tokens orchestration over one persistent Metal Llama session.

use super::{
    LlamaChatError, LlamaChatMessage, LlamaChatTemplate, LlamaGeneration, LlamaGenerationError,
    LlamaMetalStep, LlamaMetalStepError, LlamaMetalStepPlan, LlamaMetalStepSession,
    LlamaPromptWorkflow, LlamaSampling,
    generation::{select_last, validate_sampling},
};
use crate::{
    CapturedSchedule, ExecutionPlanSummary, ReplayInput, TensorData,
    runtime::metal::{
        MetalDevice, MetalDeviceInfo, MetalDevicePreparationReport, MetalDeviceRunReport,
        MetalDeviceSessionSummary, MetalPlanOptions, MetalScoreboardContext, MetalScoreboardError,
        MetalSessionScoreboard, RenderedMetal,
    },
    tokenizer::{SimpleTokenizer, TokenizerError},
};
use std::{error, fmt};

/// Resource-free, GGUF-bound deployment plan for one selected Metal device.
///
/// The contained tokenizer, chat template, and captured model all originate
/// from one [`LlamaPromptWorkflow`]. The source host model is not retained.
pub struct LlamaMetalPlan {
    step: LlamaMetalStepPlan,
    tokenizer: SimpleTokenizer,
    chat_template: LlamaChatTemplate,
    selected_device: MetalDevice,
}

/// Persistent single-sequence Metal generation session.
pub struct LlamaMetalSession {
    step: LlamaMetalStepSession,
    tokenizer: SimpleTokenizer,
    chat_template: LlamaChatTemplate,
}

/// Successful prompt ingestion with only its final logits downloaded.
#[derive(Debug)]
pub struct LlamaMetalPrefill {
    logits: TensorData,
    reports: Vec<MetalDeviceRunReport>,
    start_position: usize,
}

/// Successful Metal generation plus an inspectable report for every committed
/// prompt or generated-token invocation.
#[derive(Debug)]
pub struct LlamaMetalGeneration {
    generation: LlamaGeneration,
    reports: Vec<MetalDeviceRunReport>,
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
            tokenizer,
            chat_template,
            selected_device: device.clone(),
        })
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
        Ok(LlamaMetalSession {
            step: self.step.prepare(self.selected_device)?,
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
        let recorder = MetalSessionScoreboard::try_new_append_state_v4(
            context,
            self.step.append_state_plan(),
        )?;
        let mut step = self.step.prepare(self.selected_device)?;
        step.bind_execution_scoreboard(recorder)?;
        Ok(LlamaMetalSession {
            step,
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

    /// Returns one successful device report per ingested prompt token.
    pub fn reports(&self) -> &[MetalDeviceRunReport] {
        &self.reports
    }

    /// Returns the committed position before prompt ingestion began.
    pub const fn start_position(&self) -> usize {
        self.start_position
    }

    /// Consumes the result into final logits and ordered reports.
    pub fn into_parts(self) -> (TensorData, Vec<MetalDeviceRunReport>) {
        (self.logits, self.reports)
    }
}

impl LlamaMetalGeneration {
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

    /// Consumes the result into generation and ordered reports.
    pub fn into_parts(self) -> (LlamaGeneration, Vec<MetalDeviceRunReport>) {
        (self.generation, self.reports)
    }
}

impl LlamaMetalPromptOutput {
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
        self.step.position()
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
        self.step.is_full()
    }

    /// Executes and commits one token while retaining its logits.
    pub fn run_token(&mut self, token: u32) -> Result<LlamaMetalStep, LlamaMetalGenerationError> {
        self.step.run_token(token).map_err(Into::into)
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
        for (offset, &token) in prompt_ids[..prompt_ids.len() - 1].iter().enumerate() {
            let commit = self.step.commit_token(token).map_err(|source| {
                LlamaMetalGenerationError::Execution {
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
                }
            })?;
            let (_, report) = commit.into_parts();
            reports.push(report);
        }
        let offset = prompt_ids.len() - 1;
        let token = prompt_ids[offset];
        let step =
            self.step
                .run_token(token)
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
        reports.push(report);
        Ok(LlamaMetalPrefill {
            logits,
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
            });
        }

        let prefill = self.prefill_ids(prompt_ids)?;
        let start_position = prefill.start_position();
        let (mut logits, mut reports) = prefill.into_parts();
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
                let next = self.step.run_token(token).map_err(|source| {
                    LlamaMetalGenerationError::Execution {
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
                    }
                })?;
                let (next_logits, report) = next.into_parts();
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

fn progress(
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
