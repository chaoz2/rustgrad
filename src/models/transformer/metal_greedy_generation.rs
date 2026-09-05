//! Typed device-greedy generation over one persistent Metal Llama session.

use super::{
    LlamaChatMessage, LlamaChatTemplate, LlamaGeneration, LlamaGenerationError,
    LlamaMetalGeneration, LlamaMetalGenerationError, LlamaMetalGenerationStage,
    LlamaMetalPromptOutput, LlamaPromptWorkflow,
    metal_generation::{
        LlamaMetalEvidenceInputs, LlamaMetalPrefillSession, build_workload_evidence, progress,
    },
    metal_step::{
        LlamaMetalGreedyStepPlan, LlamaMetalGreedyStepSession, LlamaMetalPrefillPlan,
        LlamaMetalStepError,
    },
    metal_workload_evidence::LlamaMetalWorkloadEvidence,
};
use crate::{
    CapturedSchedule, ExecutionPlanSummary, ReplayInput,
    runtime::metal::{
        MetalDevice, MetalDeviceInfo, MetalDevicePreparationReport, MetalDeviceRunReport,
        MetalDeviceSessionSummary, MetalPlanOptions, RenderedMetal,
    },
    tokenizer::SimpleTokenizer,
};
use std::num::NonZeroUsize;

/// Resource-free Llama deployment whose token-step capture publishes only a
/// finite-guarded greedy I32 token.
pub struct LlamaMetalGreedyPlan {
    step: LlamaMetalGreedyStepPlan,
    prefill: Option<LlamaMetalPrefillPlan>,
    tokenizer: SimpleTokenizer,
    chat_template: LlamaChatTemplate,
    selected_device: MetalDevice,
}

/// Persistent greedy-only generation session. Host/Gumbel sampling remains on
/// [`super::LlamaMetalPlan`] and cannot be selected through this typed surface.
pub struct LlamaMetalGreedySession {
    step: LlamaMetalGreedyStepSession,
    prefill: Option<LlamaMetalPrefillSession>,
    token_step_deployment_identity: u64,
    fixed_prefill_deployment_identity: Option<u64>,
    committed_position: usize,
    successful_invocations: u64,
    tokenizer: SimpleTokenizer,
    chat_template: LlamaChatTemplate,
}

impl LlamaMetalGreedyPlan {
    pub fn from_workflow(
        workflow: LlamaPromptWorkflow,
        device: &MetalDevice,
        options: MetalPlanOptions,
    ) -> Result<Self, LlamaMetalGenerationError> {
        let (model, tokenizer, chat_template) = workflow.into_parts();
        let renderer = device
            .renderer(options.local_size)
            .map_err(LlamaMetalStepError::Metal)?;
        let step = LlamaMetalGreedyStepPlan::new(&model, renderer)?;
        Ok(Self {
            step,
            prefill: None,
            tokenizer,
            chat_template,
            selected_device: device.clone(),
        })
    }

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
        let step = LlamaMetalGreedyStepPlan::new(&model, renderer.clone())?;
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

    pub fn prefill_span_rows(&self) -> Option<NonZeroUsize> {
        self.prefill.as_ref().map(LlamaMetalPrefillPlan::span_rows)
    }

    pub fn prefill_capture(&self) -> Option<&CapturedSchedule> {
        self.prefill.as_ref().map(LlamaMetalPrefillPlan::capture)
    }

    pub fn prefill_summary(&self) -> Option<&MetalDeviceSessionSummary> {
        self.prefill.as_ref().map(LlamaMetalPrefillPlan::summary)
    }

    pub fn prefill_deployment_identity(&self) -> Option<u64> {
        self.prefill
            .as_ref()
            .map(LlamaMetalPrefillPlan::deployment_identity)
    }

    pub fn prefill_execution_plan(&self) -> Option<&ExecutionPlanSummary> {
        self.prefill
            .as_ref()
            .map(LlamaMetalPrefillPlan::execution_plan)
    }

    pub fn prefill_rendered_items(&self) -> Option<impl ExactSizeIterator<Item = &RenderedMetal>> {
        self.prefill
            .as_ref()
            .map(LlamaMetalPrefillPlan::rendered_items)
    }

    pub fn selected_device_info(&self) -> &MetalDeviceInfo {
        self.selected_device.info()
    }

    pub fn selected_device_owner_id(&self) -> u64 {
        self.selected_device.owner_id()
    }

    pub fn step_deployment_identity(&self) -> u64 {
        self.step.deployment_identity()
    }

    pub fn capture(&self) -> &CapturedSchedule {
        self.step.capture()
    }

    pub fn execution_plan(&self) -> &ExecutionPlanSummary {
        self.step.execution_plan()
    }

    pub fn resident_inputs(&self) -> &[ReplayInput] {
        self.step.resident_inputs()
    }

    pub fn state_inputs(&self) -> &[ReplayInput] {
        self.step.state_inputs()
    }

    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.step.transient_inputs()
    }

    pub fn runtime_control_inputs(&self) -> &[ReplayInput] {
        self.step.runtime_control_inputs()
    }

    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.step.summary()
    }

    pub fn max_context(&self) -> usize {
        self.step.max_context()
    }

    pub fn vocab_size(&self) -> usize {
        self.step.vocab_size()
    }

    pub fn rendered_items(&self) -> impl ExactSizeIterator<Item = &RenderedMetal> {
        self.step.rendered_items()
    }

    pub fn prepare(self) -> Result<LlamaMetalGreedySession, LlamaMetalGenerationError> {
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
        Ok(LlamaMetalGreedySession {
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
}

impl LlamaMetalGreedySession {
    pub fn device_info(&self) -> &MetalDeviceInfo {
        self.step.metal_session().device_info()
    }

    pub fn device_owner_id(&self) -> u64 {
        self.step.metal_session().device_owner_id()
    }

    pub fn summary(&self) -> &MetalDeviceSessionSummary {
        self.step.metal_session().summary()
    }

    pub fn prefill_summary(&self) -> Option<&MetalDeviceSessionSummary> {
        self.prefill.as_ref().map(LlamaMetalPrefillSession::summary)
    }

    pub fn preparation_report(&self) -> &MetalDevicePreparationReport {
        self.step.metal_session().preparation_report()
    }

    pub fn prefill_preparation_report(&self) -> Option<&MetalDevicePreparationReport> {
        self.prefill
            .as_ref()
            .map(LlamaMetalPrefillSession::preparation_report)
    }

    pub const fn token_step_deployment_identity(&self) -> u64 {
        self.token_step_deployment_identity
    }

    pub const fn fixed_prefill_deployment_identity(&self) -> Option<u64> {
        self.fixed_prefill_deployment_identity
    }

    pub fn prefill_span_rows(&self) -> Option<NonZeroUsize> {
        self.prefill
            .as_ref()
            .map(LlamaMetalPrefillSession::span_rows)
    }

    pub fn compiled_kernels(&self) -> impl Iterator<Item = &RenderedMetal> {
        self.step.metal_session().compiled_kernels()
    }

    pub fn compiled_prefill_kernels(&self) -> Option<impl Iterator<Item = &RenderedMetal>> {
        self.prefill
            .as_ref()
            .map(LlamaMetalPrefillSession::compiled_kernels)
    }

    pub fn capture(&self) -> &CapturedSchedule {
        self.step.metal_session().capture()
    }

    pub fn resident_inputs(&self) -> &[ReplayInput] {
        self.step.metal_session().resident_inputs()
    }

    pub fn state_inputs(&self) -> &[ReplayInput] {
        self.step.metal_session().state_inputs()
    }

    pub fn transient_inputs(&self) -> &[ReplayInput] {
        self.step.metal_session().transient_inputs()
    }

    pub fn runtime_control_inputs(&self) -> &[ReplayInput] {
        self.step.metal_session().runtime_control_inputs()
    }

    pub const fn position(&self) -> usize {
        self.committed_position
    }

    pub const fn successful_invocation_count(&self) -> u64 {
        self.successful_invocations
    }

    pub fn max_context(&self) -> usize {
        self.step.max_context()
    }

    pub fn vocab_size(&self) -> usize {
        self.step.vocab_size()
    }

    pub fn is_full(&self) -> bool {
        self.committed_position == self.step.max_context()
    }

    pub fn generate_ids(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<LlamaMetalGeneration, LlamaMetalGenerationError> {
        self.require_fresh()?;
        self.preflight_tokens(prompt_ids, max_new_tokens)?;
        if max_new_tokens == 0 {
            return Ok(LlamaMetalGeneration::from_parts_with_evidence(
                LlamaGeneration::from_parts(prompt_ids.to_vec(), Vec::new(), String::new(), false),
                Vec::new(),
                self.workload_evidence(0, 0, &[], 0, &[]),
            ));
        }

        let start_position = self.position();
        let (mut token, mut reports) = self.prefill_for_token(prompt_ids, start_position)?;
        let prompt_report_count = reports.len();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut decoded = String::new();
        let mut decoder = self.tokenizer.stream_decoder();
        let mut stopped = false;
        for step_index in 0..max_new_tokens {
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
                        token_offset: prompt_ids.len() + step_index,
                        token,
                        source,
                    })?;
                let (next_token, report) = next.into_parts();
                self.committed_position = self
                    .committed_position
                    .checked_add(1)
                    .ok_or(LlamaGenerationError::ContextOverflow)?;
                self.successful_invocations = invocation;
                reports.push(report);
                token = next_token;
            }
        }
        decoded.push_str(&decoder.finish());
        let evidence = self.workload_evidence(
            self.first_prompt_invocation_token_count(prompt_ids.len()),
            prompt_ids.len(),
            &reports[..prompt_report_count],
            reports.len() - prompt_report_count,
            &reports[prompt_report_count..],
        );
        Ok(LlamaMetalGeneration::from_parts_with_evidence(
            LlamaGeneration::from_parts(prompt_ids.to_vec(), generated, decoded, stopped),
            reports,
            evidence,
        ))
    }

    pub fn generate_text(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<LlamaMetalPromptOutput, LlamaMetalGenerationError> {
        let prompt_ids = self.encode_prompt(prompt)?;
        let generation = self.generate_ids(&prompt_ids, max_new_tokens)?;
        Ok(LlamaMetalPromptOutput::from_parts(
            prompt.to_owned(),
            generation,
        ))
    }

    pub fn generate_chat(
        &mut self,
        messages: &[LlamaChatMessage],
        max_new_tokens: usize,
    ) -> Result<LlamaMetalPromptOutput, LlamaMetalGenerationError> {
        let rendered_prompt = self.chat_template.render(&self.tokenizer, messages, true)?;
        let prompt_ids = self.encode_prompt(&rendered_prompt)?;
        let generation = self.generate_ids(&prompt_ids, max_new_tokens)?;
        Ok(LlamaMetalPromptOutput::from_parts(
            rendered_prompt,
            generation,
        ))
    }

    fn prefill_for_token(
        &mut self,
        prompt_ids: &[u32],
        start_position: usize,
    ) -> Result<(u32, Vec<MetalDeviceRunReport>), LlamaMetalGenerationError> {
        let mut reports = Vec::with_capacity(prompt_ids.len());
        let prefix_end = prompt_ids.len() - 1;
        let mut offset = 0usize;
        if let Some(span) = self
            .prefill
            .as_ref()
            .map(|prefill| prefill.span_rows().get())
        {
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
        let token_offset = prefix_end;
        let token = prompt_ids[token_offset];
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
                token_offset,
                token,
                source,
            })?;
        let (selected, report) = step.into_parts();
        self.committed_position = self
            .committed_position
            .checked_add(1)
            .ok_or(LlamaGenerationError::ContextOverflow)?;
        self.successful_invocations = invocation;
        reports.push(report);
        Ok((selected, reports))
    }

    fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>, LlamaMetalGenerationError> {
        let mut prompt_ids = self.tokenizer.encode(prompt)?;
        if let Some(bos) = self.tokenizer.bos_id() {
            prompt_ids.insert(0, bos);
        }
        Ok(prompt_ids)
    }

    fn require_fresh(&self) -> Result<(), LlamaMetalGenerationError> {
        if self.position() != 0 {
            return Err(LlamaMetalGenerationError::FreshSessionRequired {
                position: self.position(),
            });
        }
        Ok(())
    }

    fn next_invocation(&self) -> Result<u64, LlamaMetalGenerationError> {
        self.successful_invocations
            .checked_add(1)
            .ok_or_else(|| LlamaMetalStepError::Dimension("invocation counter overflow").into())
    }

    fn preflight_tokens(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<(), LlamaMetalGenerationError> {
        if prompt_ids.is_empty() {
            return Err(LlamaGenerationError::EmptyPrompt.into());
        }
        let requested = prompt_ids
            .len()
            .checked_add(max_new_tokens)
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
}

fn authenticate_prefill_pair(
    step: &LlamaMetalGreedyStepPlan,
    prefill: &LlamaMetalPrefillPlan,
) -> Result<(), LlamaMetalStepError> {
    let [step_position] = step.runtime_control_inputs() else {
        return Err(LlamaMetalStepError::Metal(
            crate::runtime::metal::MetalError::InvalidBinding(
                "device-greedy Llama position control is not an exact singleton".into(),
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
                "fixed-span and device-greedy Llama deployments differ".into(),
            ),
        ));
    }
    Ok(())
}
