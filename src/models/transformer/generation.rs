use super::{LlamaModel, LlamaModelCache, LlamaModelError};
use crate::tokenizer::{SimpleTokenizer, TokenizerError};
use std::{error, fmt};

/// Deterministic token selection supported by the checked-in tinygrad path.
///
/// `GumbelMax` uses tinygrad's exact score transform, while randomness is an
/// explicit row-major `[step, vocabulary]` uniform tape. This makes replay and
/// tape consumption fully specified without claiming tinygrad RNG-state parity.
#[derive(Clone, Copy, Debug)]
pub enum LlamaSampling<'a> {
    /// Selects the lowest token ID among maximum-logit ties.
    Greedy,
    /// Applies tinygrad's Gumbel-max score transform using explicit uniforms.
    GumbelMax {
        temperature: f32,
        uniforms: &'a [f32],
    },
}

/// Generated IDs and their source-compatible tokenizer decoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaGeneration {
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    decoded: String,
    stopped: bool,
}

impl LlamaGeneration {
    /// Returns the exact encoded prompt, including an inserted BOS if enabled.
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }
    /// Returns generated IDs, including a final EOS/EOT when one stopped generation.
    pub fn generated_ids(&self) -> &[u32] {
        &self.generated_ids
    }
    /// Returns tokenizer decoding of `generated_ids`.
    pub fn decoded(&self) -> &str {
        &self.decoded
    }
    /// Returns whether EOS or EOT stopped generation before its token bound.
    pub const fn stopped(&self) -> bool {
        self.stopped
    }
}

/// Single-sequence generator with an atomically committed per-layer KV cache.
#[derive(Clone, Debug)]
pub struct LlamaGenerator<'a> {
    model: &'a LlamaModel,
    tokenizer: &'a SimpleTokenizer,
    cache: LlamaModelCache,
}

impl<'a> LlamaGenerator<'a> {
    /// Creates an empty generator bound to one model and its GGUF tokenizer.
    pub fn new(model: &'a LlamaModel, tokenizer: &'a SimpleTokenizer) -> Self {
        Self {
            model,
            tokenizer,
            cache: LlamaModelCache::new(model.config().clone()),
        }
    }

    /// Returns the committed prefix length (the final generated token is not cached).
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }
    /// Drops all committed per-layer cache state.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Encodes a prompt (including configured BOS), generates up to the exact
    /// requested bound, stops on EOS/EOT, and decodes generated IDs. The cache
    /// changes only if the complete call succeeds.
    pub fn generate_text(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        sampling: LlamaSampling<'_>,
    ) -> Result<LlamaGeneration, LlamaGenerationError> {
        let mut prompt_ids = self.tokenizer.encode(prompt)?;
        if let Some(bos) = self.model.config().token_ids().bos() {
            prompt_ids.insert(0, bos);
        }
        self.generate_ids(&prompt_ids, max_new_tokens, sampling)
    }

    /// Generates from explicit prompt IDs with all cache mutation staged.
    pub fn generate_ids(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        sampling: LlamaSampling<'_>,
    ) -> Result<LlamaGeneration, LlamaGenerationError> {
        if prompt_ids.is_empty() {
            return Err(LlamaGenerationError::EmptyPrompt);
        }
        let requested = prompt_ids
            .len()
            .checked_add(max_new_tokens)
            .ok_or(LlamaGenerationError::ContextOverflow)?;
        if requested > self.model.config().max_context() {
            return Err(LlamaGenerationError::ContextLength {
                requested,
                maximum: self.model.config().max_context(),
            });
        }
        let vocab = self.model.config().schema().vocab_size;
        validate_sampling(sampling, max_new_tokens, vocab)?;
        if max_new_tokens == 0 {
            return Ok(LlamaGeneration {
                prompt_ids: prompt_ids.to_vec(),
                generated_ids: Vec::new(),
                decoded: String::new(),
                stopped: false,
            });
        }

        let mut staged = LlamaModelCache::new(self.model.config().clone());
        let mut logits = staged.forward(self.model, prompt_ids)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut stopped = false;
        for step in 0..max_new_tokens {
            let token = select_last(&logits, sampling, step, vocab)?;
            generated.push(token);
            if self.model.config().token_ids().is_stop(token) {
                stopped = true;
                break;
            }
            if step + 1 < max_new_tokens {
                logits = staged.forward(self.model, &[token])?;
            }
        }
        let decoded = self.tokenizer.decode(&generated)?;
        self.cache = staged;
        Ok(LlamaGeneration {
            prompt_ids: prompt_ids.to_vec(),
            generated_ids: generated,
            decoded,
            stopped,
        })
    }
}

/// Structured prompt, sampling, context, model, or decoding failure.
#[derive(Clone, Debug, PartialEq)]
pub enum LlamaGenerationError {
    Model(LlamaModelError),
    Tokenizer(TokenizerError),
    EmptyPrompt,
    ContextOverflow,
    ContextLength { requested: usize, maximum: usize },
    InvalidTemperature,
    UniformTapeLength { required: usize, actual: usize },
    InvalidUniform { index: usize },
    InvalidLogits,
}

impl fmt::Display for LlamaGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llama generation error: {self:?}")
    }
}
impl error::Error for LlamaGenerationError {}
impl From<LlamaModelError> for LlamaGenerationError {
    fn from(value: LlamaModelError) -> Self {
        Self::Model(value)
    }
}
impl From<TokenizerError> for LlamaGenerationError {
    fn from(value: TokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}

fn validate_sampling(
    sampling: LlamaSampling<'_>,
    steps: usize,
    vocab: usize,
) -> Result<(), LlamaGenerationError> {
    let LlamaSampling::GumbelMax {
        temperature,
        uniforms,
    } = sampling
    else {
        return Ok(());
    };
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(LlamaGenerationError::InvalidTemperature);
    }
    let required = steps
        .checked_mul(vocab)
        .ok_or(LlamaGenerationError::ContextOverflow)?;
    if uniforms.len() < required {
        return Err(LlamaGenerationError::UniformTapeLength {
            required,
            actual: uniforms.len(),
        });
    }
    if let Some(index) = uniforms[..required]
        .iter()
        .position(|value| !value.is_finite() || *value < 0.0 || *value >= 1.0)
    {
        return Err(LlamaGenerationError::InvalidUniform { index });
    }
    Ok(())
}

fn select_last(
    logits: &crate::TensorData,
    sampling: LlamaSampling<'_>,
    step: usize,
    vocab: usize,
) -> Result<u32, LlamaGenerationError> {
    if logits.dtype() != crate::DType::F32
        || logits.shape().rank() != 2
        || logits.shape().dims()[1] != vocab
    {
        return Err(LlamaGenerationError::InvalidLogits);
    }
    let sequence = logits.shape().dims()[0];
    if sequence == 0 {
        return Err(LlamaGenerationError::InvalidLogits);
    }
    let row = &logits.values()[(sequence - 1) * vocab..sequence * vocab];
    let offset = step
        .checked_mul(vocab)
        .ok_or(LlamaGenerationError::ContextOverflow)?;
    match sampling {
        LlamaSampling::Greedy => select_row(row, None, None),
        LlamaSampling::GumbelMax {
            temperature,
            uniforms,
        } => select_row(
            row,
            Some(temperature),
            Some(&uniforms[offset..offset + vocab]),
        ),
    }
}

pub(super) fn select_row(
    row: &[f32],
    temperature: Option<f32>,
    uniforms: Option<&[f32]>,
) -> Result<u32, LlamaGenerationError> {
    if row.is_empty() || temperature.is_some() != uniforms.is_some() {
        return Err(LlamaGenerationError::InvalidLogits);
    }
    let mut best = None;
    for (token, &logit) in row.iter().enumerate() {
        if !logit.is_finite() {
            return Err(LlamaGenerationError::InvalidLogits);
        }
        let score = match (temperature, uniforms) {
            (None, None) => logit,
            (Some(temperature), Some(uniforms)) if uniforms.len() == row.len() => {
                let uniform = uniforms[token].max(1e-12);
                logit / temperature.max(1e-12) - (-uniform.ln()).ln()
            }
            _ => return Err(LlamaGenerationError::InvalidLogits),
        };
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((token, score));
        }
    }
    u32::try_from(best.ok_or(LlamaGenerationError::InvalidLogits)?.0)
        .map_err(|_| LlamaGenerationError::InvalidLogits)
}
