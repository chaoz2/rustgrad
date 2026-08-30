use super::{
    LlamaBatchGenerationError, LlamaBatchNativeCache, LlamaBatchSampling, LlamaChatError,
    LlamaChatMessage, LlamaChatTemplate, LlamaGenerationError, LlamaModel, LlamaNativeCache,
    LlamaNativeError, LlamaNativeStageTrace, LlamaSampling,
    batch_generation::validate_sampling as validate_batch_sampling,
    generation::{select_last, select_row, validate_sampling},
};
use crate::tokenizer::SimpleTokenizer;
use std::{error, fmt};

/// One successful native prefill or decode step and its compile/cache evidence.
#[derive(Clone, Debug)]
pub struct LlamaNativeGenerationStepTrace {
    input_lengths: Vec<usize>,
    cache_before: Vec<usize>,
    cache_after: Vec<usize>,
    compile_cache_before: usize,
    compile_cache_after: usize,
    stages: Vec<LlamaNativeStageTrace>,
}

impl LlamaNativeGenerationStepTrace {
    pub fn input_lengths(&self) -> &[usize] {
        &self.input_lengths
    }
    pub fn cache_before(&self) -> &[usize] {
        &self.cache_before
    }
    pub fn cache_after(&self) -> &[usize] {
        &self.cache_after
    }
    pub const fn compile_cache_before(&self) -> usize {
        self.compile_cache_before
    }
    pub const fn compile_cache_after(&self) -> usize {
        self.compile_cache_after
    }
    pub fn stages(&self) -> &[LlamaNativeStageTrace] {
        &self.stages
    }
}

/// Deterministic native generation output and its complete staged trace.
#[derive(Clone, Debug)]
pub struct LlamaNativeGeneration {
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    decoded: String,
    stopped: bool,
    trace: Vec<LlamaNativeGenerationStepTrace>,
}

impl LlamaNativeGeneration {
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }
    pub fn generated_ids(&self) -> &[u32] {
        &self.generated_ids
    }
    pub fn decoded(&self) -> &str {
        &self.decoded
    }
    pub const fn stopped(&self) -> bool {
        self.stopped
    }
    pub fn trace(&self) -> &[LlamaNativeGenerationStepTrace] {
        &self.trace
    }
}

/// Reusable single-sequence native generator with whole-call cache commit.
pub struct LlamaNativeGenerator<'a> {
    model: &'a LlamaModel,
    tokenizer: &'a SimpleTokenizer,
    cache: LlamaNativeCache,
    #[cfg(test)]
    fail_after_stage: Option<usize>,
}

impl<'a> LlamaNativeGenerator<'a> {
    pub fn new(model: &'a LlamaModel, tokenizer: &'a SimpleTokenizer) -> Self {
        Self {
            model,
            tokenizer,
            cache: LlamaNativeCache::new(model.config().clone()),
            #[cfg(test)]
            fail_after_stage: None,
        }
    }

    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }
    #[cfg(test)]
    pub(super) fn inject_stage_failure(&mut self, stage: Option<usize>) {
        self.fail_after_stage = stage;
    }

    pub fn generate_text(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        sampling: LlamaSampling<'_>,
    ) -> Result<LlamaNativeGeneration, LlamaNativeGenerationError> {
        let mut prompt_ids = self.tokenizer.encode(prompt)?;
        if let Some(bos) = self.model.config().token_ids().bos() {
            prompt_ids.insert(0, bos);
        }
        self.generate_ids(&prompt_ids, max_new_tokens, sampling)
    }

    pub fn generate_chat(
        &mut self,
        template: LlamaChatTemplate,
        messages: &[LlamaChatMessage],
        max_new_tokens: usize,
        sampling: LlamaSampling<'_>,
    ) -> Result<LlamaNativeGeneration, LlamaNativeGenerationError> {
        let rendered = template.render(self.tokenizer, messages, true)?;
        let prompt_ids = self.tokenizer.encode(&rendered)?;
        self.generate_ids(&prompt_ids, max_new_tokens, sampling)
    }

    pub fn generate_ids(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        sampling: LlamaSampling<'_>,
    ) -> Result<LlamaNativeGeneration, LlamaNativeGenerationError> {
        if prompt_ids.is_empty() {
            return Err(LlamaGenerationError::EmptyPrompt.into());
        }
        let requested = prompt_ids
            .len()
            .checked_add(max_new_tokens)
            .ok_or(LlamaGenerationError::ContextOverflow)?;
        if requested > self.model.config().max_context() {
            return Err(LlamaGenerationError::ContextLength {
                requested,
                maximum: self.model.config().max_context(),
            }
            .into());
        }
        self.model
            .validate_token_ids(prompt_ids)
            .map_err(LlamaGenerationError::from)?;
        let vocab = self.model.config().schema().vocab_size();
        validate_sampling(sampling, max_new_tokens, vocab)?;
        if max_new_tokens == 0 {
            return Ok(LlamaNativeGeneration {
                prompt_ids: prompt_ids.to_vec(),
                generated_ids: Vec::new(),
                decoded: String::new(),
                stopped: false,
                trace: Vec::new(),
            });
        }

        let mut staged = LlamaNativeCache::new(self.model.config().clone());
        #[cfg(test)]
        staged.inject_stage_failure(self.fail_after_stage);
        let mut trace = Vec::with_capacity(max_new_tokens);
        let mut logits = forward_single(&mut staged, self.model, prompt_ids, &mut trace)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut stopped = false;
        for step in 0..max_new_tokens {
            let token = select_last(logits.logits(), sampling, step, vocab)?;
            generated.push(token);
            if self.model.config().token_ids().is_stop(token) {
                stopped = true;
                break;
            }
            if step + 1 < max_new_tokens {
                logits = forward_single(&mut staged, self.model, &[token], &mut trace)?;
            }
        }
        let decoded = self.tokenizer.decode(&generated)?;
        self.cache = staged;
        Ok(LlamaNativeGeneration {
            prompt_ids: prompt_ids.to_vec(),
            generated_ids: generated,
            decoded,
            stopped,
            trace,
        })
    }
}

fn forward_single(
    cache: &mut LlamaNativeCache,
    model: &LlamaModel,
    input: &[u32],
    trace: &mut Vec<LlamaNativeGenerationStepTrace>,
) -> Result<super::LlamaNativeExecution, LlamaNativeGenerationError> {
    let before = cache.len();
    let compile_before = cache.compile_cache_len();
    let execution = cache.forward(model, input)?;
    trace.push(LlamaNativeGenerationStepTrace {
        input_lengths: vec![input.len()],
        cache_before: vec![before],
        cache_after: vec![cache.len()],
        compile_cache_before: compile_before,
        compile_cache_after: cache.compile_cache_len(),
        stages: execution.trace().to_vec(),
    });
    Ok(execution)
}

/// One row of native fixed-batch generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaBatchNativeSequence {
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    decoded: String,
    stopped: bool,
}

impl LlamaBatchNativeSequence {
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }
    pub fn generated_ids(&self) -> &[u32] {
        &self.generated_ids
    }
    pub fn decoded(&self) -> &str {
        &self.decoded
    }
    pub const fn stopped(&self) -> bool {
        self.stopped
    }
}

#[derive(Clone, Debug)]
pub struct LlamaBatchNativeGeneration {
    sequences: Vec<LlamaBatchNativeSequence>,
    trace: Vec<LlamaNativeGenerationStepTrace>,
}

impl LlamaBatchNativeGeneration {
    pub fn sequences(&self) -> &[LlamaBatchNativeSequence] {
        &self.sequences
    }
    pub fn trace(&self) -> &[LlamaNativeGenerationStepTrace] {
        &self.trace
    }
}

/// Fixed-batch native generator with independent stop state and atomic commit.
pub struct LlamaBatchNativeGenerator<'a> {
    model: &'a LlamaModel,
    tokenizer: &'a SimpleTokenizer,
    cache: LlamaBatchNativeCache,
    #[cfg(test)]
    fail_after_stage: Option<usize>,
}

impl<'a> LlamaBatchNativeGenerator<'a> {
    pub fn new(
        model: &'a LlamaModel,
        tokenizer: &'a SimpleTokenizer,
        batch_size: usize,
    ) -> Result<Self, LlamaNativeGenerationError> {
        Ok(Self {
            model,
            tokenizer,
            cache: LlamaBatchNativeCache::new(model.config().clone(), batch_size)?,
            #[cfg(test)]
            fail_after_stage: None,
        })
    }

    pub fn cache_lengths(&self) -> &[usize] {
        self.cache.lengths()
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }
    #[cfg(test)]
    pub(super) fn inject_stage_failure(&mut self, stage: Option<usize>) {
        self.fail_after_stage = stage;
    }

    pub fn generate_texts(
        &mut self,
        prompts: &[&str],
        max_new_tokens: usize,
        sampling: LlamaBatchSampling<'_>,
    ) -> Result<LlamaBatchNativeGeneration, LlamaNativeGenerationError> {
        let mut rows = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let mut ids = self.tokenizer.encode(prompt)?;
            if let Some(bos) = self.model.config().token_ids().bos() {
                ids.insert(0, bos);
            }
            rows.push(ids);
        }
        self.generate_ids(&rows, max_new_tokens, sampling)
    }

    pub fn generate_ids(
        &mut self,
        prompts: &[Vec<u32>],
        max_new_tokens: usize,
        sampling: LlamaBatchSampling<'_>,
    ) -> Result<LlamaBatchNativeGeneration, LlamaNativeGenerationError> {
        if prompts.len() != self.cache.lengths().len() {
            return Err(LlamaBatchGenerationError::BatchSize {
                expected: self.cache.lengths().len(),
                actual: prompts.len(),
            }
            .into());
        }
        for (row, prompt) in prompts.iter().enumerate() {
            if prompt.is_empty() {
                return Err(LlamaBatchGenerationError::EmptyPrompt { row }.into());
            }
            let requested = prompt
                .len()
                .checked_add(max_new_tokens)
                .ok_or(LlamaBatchGenerationError::ContextOverflow)?;
            if requested > self.model.config().max_context() {
                return Err(LlamaBatchGenerationError::ContextLength {
                    row,
                    requested,
                    maximum: self.model.config().max_context(),
                }
                .into());
            }
        }
        self.model
            .validate_batch_token_ids(prompts)
            .map_err(LlamaBatchGenerationError::from)?;
        let batch = prompts.len();
        let vocab = self.model.config().schema().vocab_size();
        validate_batch_sampling(sampling, max_new_tokens, batch, vocab)?;
        if max_new_tokens == 0 {
            return finish_batch(
                self.tokenizer,
                prompts,
                vec![Vec::new(); batch],
                vec![false; batch],
                Vec::new(),
            );
        }

        let mut staged = LlamaBatchNativeCache::new(self.model.config().clone(), batch)?;
        #[cfg(test)]
        staged.inject_stage_failure(self.fail_after_stage);
        let mut trace = Vec::with_capacity(max_new_tokens);
        let mut logits = forward_batch(&mut staged, self.model, prompts, &mut trace)?;
        let mut generated = vec![Vec::with_capacity(max_new_tokens); batch];
        let mut stopped = vec![false; batch];
        for step in 0..max_new_tokens {
            let mut next = vec![Vec::new(); batch];
            for row in 0..batch {
                if stopped[row] {
                    continue;
                }
                let values = logits.rows()[row].values();
                if values.len() < vocab {
                    return Err(LlamaGenerationError::InvalidLogits.into());
                }
                let last = &values[values.len() - vocab..];
                let token = match sampling {
                    LlamaBatchSampling::Greedy => select_row(last, None, None),
                    LlamaBatchSampling::GumbelMax {
                        temperature,
                        uniforms,
                    } => {
                        let offset = (step * batch + row) * vocab;
                        select_row(
                            last,
                            Some(temperature),
                            Some(&uniforms[offset..offset + vocab]),
                        )
                    }
                }?;
                generated[row].push(token);
                stopped[row] = self.model.config().token_ids().is_stop(token);
                if !stopped[row] {
                    next[row].push(token);
                }
            }
            if step + 1 < max_new_tokens && stopped.iter().any(|value| !value) {
                logits = forward_batch(&mut staged, self.model, &next, &mut trace)?;
            }
        }
        let result = finish_batch(self.tokenizer, prompts, generated, stopped, trace)?;
        self.cache = staged;
        Ok(result)
    }
}

fn forward_batch(
    cache: &mut LlamaBatchNativeCache,
    model: &LlamaModel,
    input: &[Vec<u32>],
    trace: &mut Vec<LlamaNativeGenerationStepTrace>,
) -> Result<super::LlamaBatchNativeExecution, LlamaNativeGenerationError> {
    let before = cache.lengths().to_vec();
    let compile_before = cache.compile_cache_len();
    let execution = cache.forward(model, input)?;
    trace.push(LlamaNativeGenerationStepTrace {
        input_lengths: input.iter().map(Vec::len).collect(),
        cache_before: before,
        cache_after: cache.lengths().to_vec(),
        compile_cache_before: compile_before,
        compile_cache_after: cache.compile_cache_len(),
        stages: execution.trace().to_vec(),
    });
    Ok(execution)
}

fn finish_batch(
    tokenizer: &SimpleTokenizer,
    prompts: &[Vec<u32>],
    generated: Vec<Vec<u32>>,
    stopped: Vec<bool>,
    trace: Vec<LlamaNativeGenerationStepTrace>,
) -> Result<LlamaBatchNativeGeneration, LlamaNativeGenerationError> {
    let sequences = prompts
        .iter()
        .zip(generated)
        .zip(stopped)
        .map(|((prompt, generated_ids), stopped)| {
            Ok(LlamaBatchNativeSequence {
                prompt_ids: prompt.clone(),
                decoded: tokenizer.decode(&generated_ids)?,
                generated_ids,
                stopped,
            })
        })
        .collect::<Result<Vec<_>, LlamaNativeGenerationError>>()?;
    Ok(LlamaBatchNativeGeneration { sequences, trace })
}

impl LlamaModel {
    pub fn generate_native(
        &self,
        tokenizer: &SimpleTokenizer,
        prompt: &str,
        max_new_tokens: usize,
        sampling: LlamaSampling<'_>,
    ) -> Result<LlamaNativeGeneration, LlamaNativeGenerationError> {
        LlamaNativeGenerator::new(self, tokenizer).generate_text(prompt, max_new_tokens, sampling)
    }

    pub fn generate_chat_native(
        &self,
        tokenizer: &SimpleTokenizer,
        template: LlamaChatTemplate,
        messages: &[LlamaChatMessage],
        max_new_tokens: usize,
        sampling: LlamaSampling<'_>,
    ) -> Result<LlamaNativeGeneration, LlamaNativeGenerationError> {
        LlamaNativeGenerator::new(self, tokenizer).generate_chat(
            template,
            messages,
            max_new_tokens,
            sampling,
        )
    }

    pub fn generate_batch_native(
        &self,
        tokenizer: &SimpleTokenizer,
        prompts: &[&str],
        max_new_tokens: usize,
        sampling: LlamaBatchSampling<'_>,
    ) -> Result<LlamaBatchNativeGeneration, LlamaNativeGenerationError> {
        LlamaBatchNativeGenerator::new(self, tokenizer, prompts.len())?.generate_texts(
            prompts,
            max_new_tokens,
            sampling,
        )
    }
}

/// Structured tokenizer/chat/sampling/native-stage failure.
#[derive(Debug)]
pub enum LlamaNativeGenerationError {
    Native(LlamaNativeError),
    Generation(LlamaGenerationError),
    Batch(LlamaBatchGenerationError),
    Chat(LlamaChatError),
}

impl fmt::Display for LlamaNativeGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llama native generation error: {self:?}")
    }
}
impl error::Error for LlamaNativeGenerationError {}
impl From<LlamaNativeError> for LlamaNativeGenerationError {
    fn from(value: LlamaNativeError) -> Self {
        Self::Native(value)
    }
}
impl From<LlamaGenerationError> for LlamaNativeGenerationError {
    fn from(value: LlamaGenerationError) -> Self {
        Self::Generation(value)
    }
}
impl From<LlamaBatchGenerationError> for LlamaNativeGenerationError {
    fn from(value: LlamaBatchGenerationError) -> Self {
        Self::Batch(value)
    }
}
impl From<LlamaChatError> for LlamaNativeGenerationError {
    fn from(value: LlamaChatError) -> Self {
        Self::Chat(value)
    }
}
impl From<crate::tokenizer::TokenizerError> for LlamaNativeGenerationError {
    fn from(value: crate::tokenizer::TokenizerError) -> Self {
        Self::Generation(LlamaGenerationError::Tokenizer(value))
    }
}
