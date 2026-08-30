use super::{
    LlamaBatchCache, LlamaGenerationError, LlamaModel, LlamaModelError, generation::select_row,
};
use crate::tokenizer::{SimpleTokenizer, TokenizerError};
use std::{error, fmt};

/// Deterministic batch sampling with an explicit `[step, batch, vocabulary]`
/// row-major uniform tape.
#[derive(Clone, Copy, Debug)]
pub enum LlamaBatchSampling<'a> {
    /// Selects the lowest token ID among maximum-logit ties independently per row.
    Greedy,
    /// Applies the checked tinygrad Gumbel-max transform to each active row.
    GumbelMax {
        temperature: f32,
        uniforms: &'a [f32],
    },
}

/// One row of a completed batch generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaBatchSequence {
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    decoded: String,
    stopped: bool,
}

impl LlamaBatchSequence {
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

/// Per-sequence deterministic output in input batch order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaBatchGeneration {
    sequences: Vec<LlamaBatchSequence>,
}

impl LlamaBatchGeneration {
    pub fn sequences(&self) -> &[LlamaBatchSequence] {
        &self.sequences
    }
}

/// Fixed-size batch generator with an atomically committed multi-layer cache.
#[derive(Clone, Debug)]
pub struct LlamaBatchGenerator<'a> {
    model: &'a LlamaModel,
    tokenizer: &'a SimpleTokenizer,
    cache: LlamaBatchCache,
}

impl<'a> LlamaBatchGenerator<'a> {
    pub fn new(
        model: &'a LlamaModel,
        tokenizer: &'a SimpleTokenizer,
        batch_size: usize,
    ) -> Result<Self, LlamaBatchGenerationError> {
        Ok(Self {
            model,
            tokenizer,
            cache: LlamaBatchCache::new(model.config().clone(), batch_size)?,
        })
    }

    pub fn cache_lengths(&self) -> &[usize] {
        self.cache.lengths()
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn generate_texts(
        &mut self,
        prompts: &[&str],
        max_new_tokens: usize,
        sampling: LlamaBatchSampling<'_>,
    ) -> Result<LlamaBatchGeneration, LlamaBatchGenerationError> {
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

    /// Generates every row with independent stop state while staging the entire
    /// batch cache until every row, sample, and decode succeeds.
    pub fn generate_ids(
        &mut self,
        prompts: &[Vec<u32>],
        max_new_tokens: usize,
        sampling: LlamaBatchSampling<'_>,
    ) -> Result<LlamaBatchGeneration, LlamaBatchGenerationError> {
        if prompts.len() != self.cache.lengths().len() {
            return Err(LlamaBatchGenerationError::BatchSize {
                expected: self.cache.lengths().len(),
                actual: prompts.len(),
            });
        }
        for (row, prompt) in prompts.iter().enumerate() {
            if prompt.is_empty() {
                return Err(LlamaBatchGenerationError::EmptyPrompt { row });
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
                });
            }
        }
        self.model.validate_batch_token_ids(prompts)?;
        let batch = prompts.len();
        let vocab = self.model.config().schema().vocab_size();
        validate_sampling(sampling, max_new_tokens, batch, vocab)?;
        if max_new_tokens == 0 {
            return finish(
                self.tokenizer,
                prompts,
                vec![Vec::new(); batch],
                vec![false; batch],
            );
        }

        let mut staged = LlamaBatchCache::new(self.model.config().clone(), batch)?;
        let mut logits = staged.forward(self.model, prompts)?;
        let mut generated = vec![Vec::with_capacity(max_new_tokens); batch];
        let mut stopped = vec![false; batch];
        for step in 0..max_new_tokens {
            let mut next = vec![Vec::new(); batch];
            for row in 0..batch {
                if stopped[row] {
                    continue;
                }
                let values = logits[row].values();
                if values.len() < vocab {
                    return Err(LlamaBatchGenerationError::Sampling(
                        LlamaGenerationError::InvalidLogits,
                    ));
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
                logits = staged.forward(self.model, &next)?;
            }
        }
        let result = finish(self.tokenizer, prompts, generated, stopped)?;
        self.cache = staged;
        Ok(result)
    }
}

fn finish(
    tokenizer: &SimpleTokenizer,
    prompts: &[Vec<u32>],
    generated: Vec<Vec<u32>>,
    stopped: Vec<bool>,
) -> Result<LlamaBatchGeneration, LlamaBatchGenerationError> {
    let sequences = prompts
        .iter()
        .zip(generated)
        .zip(stopped)
        .map(|((prompt, generated_ids), stopped)| {
            Ok(LlamaBatchSequence {
                prompt_ids: prompt.clone(),
                decoded: tokenizer.decode(&generated_ids)?,
                generated_ids,
                stopped,
            })
        })
        .collect::<Result<Vec<_>, LlamaBatchGenerationError>>()?;
    Ok(LlamaBatchGeneration { sequences })
}

pub(super) fn validate_sampling(
    sampling: LlamaBatchSampling<'_>,
    steps: usize,
    batch: usize,
    vocab: usize,
) -> Result<(), LlamaBatchGenerationError> {
    let LlamaBatchSampling::GumbelMax {
        temperature,
        uniforms,
    } = sampling
    else {
        return Ok(());
    };
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(LlamaBatchGenerationError::Sampling(
            LlamaGenerationError::InvalidTemperature,
        ));
    }
    let required = steps
        .checked_mul(batch)
        .and_then(|value| value.checked_mul(vocab))
        .ok_or(LlamaBatchGenerationError::ContextOverflow)?;
    if uniforms.len() < required {
        return Err(LlamaBatchGenerationError::UniformTapeLength {
            required,
            actual: uniforms.len(),
        });
    }
    if let Some(index) = uniforms[..required]
        .iter()
        .position(|value| !value.is_finite() || *value < 0.0 || *value >= 1.0)
    {
        return Err(LlamaBatchGenerationError::InvalidUniform { index });
    }
    Ok(())
}

/// Structured batch prompt, tape, context, model, or tokenizer failure.
#[derive(Clone, Debug, PartialEq)]
pub enum LlamaBatchGenerationError {
    Model(LlamaModelError),
    Sampling(LlamaGenerationError),
    Tokenizer(TokenizerError),
    BatchSize {
        expected: usize,
        actual: usize,
    },
    EmptyPrompt {
        row: usize,
    },
    ContextOverflow,
    ContextLength {
        row: usize,
        requested: usize,
        maximum: usize,
    },
    UniformTapeLength {
        required: usize,
        actual: usize,
    },
    InvalidUniform {
        index: usize,
    },
}

impl fmt::Display for LlamaBatchGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llama batch generation error: {self:?}")
    }
}
impl error::Error for LlamaBatchGenerationError {}
impl From<LlamaModelError> for LlamaBatchGenerationError {
    fn from(value: LlamaModelError) -> Self {
        Self::Model(value)
    }
}
impl From<LlamaGenerationError> for LlamaBatchGenerationError {
    fn from(value: LlamaGenerationError) -> Self {
        Self::Sampling(value)
    }
}
impl From<TokenizerError> for LlamaBatchGenerationError {
    fn from(value: TokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}
