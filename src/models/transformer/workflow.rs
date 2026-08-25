//! Bounded public composition for one local GGUF Llama prompt-to-output call.

use super::{
    LlamaChatError, LlamaChatMessage, LlamaChatTemplate, LlamaGeneration, LlamaGenerationError,
    LlamaGenerator, LlamaModel, LlamaModelError, LlamaSampling,
};
use crate::{
    gguf::{GgufError, read_gguf},
    tokenizer::SimpleTokenizer,
};
use std::{error, fmt, path::Path};

/// A validated local GGUF Llama model with its source-compatible tokenizer and
/// the one checked chat-template contract. It owns no device resources and
/// every request executes the existing CPU graph/generation path.
#[derive(Clone, Debug)]
pub struct LlamaPromptWorkflow {
    model: LlamaModel,
    tokenizer: SimpleTokenizer,
    chat_template: LlamaChatTemplate,
}

impl LlamaPromptWorkflow {
    /// Parses and validates one bounded in-memory GGUF before binding the
    /// fixed supported Llama schema, tokenizer, and chat template.
    pub fn from_gguf_bytes(bytes: &[u8]) -> Result<Self, LlamaPromptWorkflowError> {
        let file = read_gguf(bytes)?;
        let (model, tokenizer) = LlamaModel::from_gguf(&file)?;
        let chat_template = LlamaChatTemplate::from_gguf(&file)?;
        Ok(Self {
            model,
            tokenizer,
            chat_template,
        })
    }

    /// Reads a local GGUF file and applies the same fail-closed validation as
    /// [`Self::from_gguf_bytes`]. No network, device selection, or fallback is
    /// involved.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, LlamaPromptWorkflowError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|error| LlamaPromptWorkflowError::ReadFile {
            path: path.display().to_string(),
            kind: error.kind(),
        })?;
        Self::from_gguf_bytes(&bytes)
    }

    /// Runs a plain text prompt with deterministic greedy selection through
    /// the existing transactional CPU generator.
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<LlamaPromptOutput, LlamaPromptWorkflowError> {
        let generation = LlamaGenerator::new(&self.model, &self.tokenizer).generate_text(
            prompt,
            max_new_tokens,
            LlamaSampling::Greedy,
        )?;
        Ok(LlamaPromptOutput {
            rendered_prompt: prompt.to_owned(),
            generation,
        })
    }

    /// Renders the checked Llama chat template then runs the same greedy CPU
    /// generation path. Unsupported Jinja/templates reject before execution.
    pub fn generate_chat(
        &self,
        messages: &[LlamaChatMessage],
        max_new_tokens: usize,
    ) -> Result<LlamaPromptOutput, LlamaPromptWorkflowError> {
        let rendered_prompt = self.chat_template.render(&self.tokenizer, messages, true)?;
        let generation = LlamaGenerator::new(&self.model, &self.tokenizer).generate_text(
            &rendered_prompt,
            max_new_tokens,
            LlamaSampling::Greedy,
        )?;
        Ok(LlamaPromptOutput {
            rendered_prompt,
            generation,
        })
    }

    /// Returns the validated, immutable model configuration.
    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    /// Returns the GGUF-bound tokenizer used by both prompt forms.
    pub fn tokenizer(&self) -> &SimpleTokenizer {
        &self.tokenizer
    }

    /// Begins an isolated stateful conversation over this immutable model.
    pub fn conversation(&self) -> super::LlamaConversation<'_> {
        super::LlamaConversation::new(self)
    }

    pub(crate) fn render_chat(
        &self,
        messages: &[LlamaChatMessage],
    ) -> Result<String, LlamaChatError> {
        self.chat_template.render(&self.tokenizer, messages, true)
    }
}

/// Observable result of a deterministic prompt-to-output request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaPromptOutput {
    rendered_prompt: String,
    generation: LlamaGeneration,
}

impl LlamaPromptOutput {
    /// Returns the exact text submitted to tokenizer encoding.
    pub fn rendered_prompt(&self) -> &str {
        &self.rendered_prompt
    }

    /// Returns token IDs, decoded text, and stop status from generation.
    pub fn generation(&self) -> &LlamaGeneration {
        &self.generation
    }
}

/// Typed local-file, GGUF, schema, tokenizer/template, or generation failure.
#[derive(Clone, Debug, PartialEq)]
pub enum LlamaPromptWorkflowError {
    ReadFile {
        path: String,
        kind: std::io::ErrorKind,
    },
    Gguf(GgufError),
    Model(LlamaModelError),
    Chat(LlamaChatError),
    Generation(LlamaGenerationError),
}

impl fmt::Display for LlamaPromptWorkflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llama prompt workflow error: {self:?}")
    }
}

impl error::Error for LlamaPromptWorkflowError {}

impl From<GgufError> for LlamaPromptWorkflowError {
    fn from(value: GgufError) -> Self {
        Self::Gguf(value)
    }
}
impl From<LlamaModelError> for LlamaPromptWorkflowError {
    fn from(value: LlamaModelError) -> Self {
        Self::Model(value)
    }
}
impl From<LlamaChatError> for LlamaPromptWorkflowError {
    fn from(value: LlamaChatError) -> Self {
        Self::Chat(value)
    }
}
impl From<LlamaGenerationError> for LlamaPromptWorkflowError {
    fn from(value: LlamaGenerationError) -> Self {
        Self::Generation(value)
    }
}
