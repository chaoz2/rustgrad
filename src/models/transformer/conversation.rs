//! Stateful transcript composition over the released Llama generator.

use super::{
    LlamaChatError, LlamaChatMessage, LlamaChatRole, LlamaGeneration, LlamaGenerationError,
    LlamaGenerator, LlamaPromptWorkflow,
};
use std::{error, fmt};

/// One successfully committed user/assistant exchange.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaConversationTurn {
    user: LlamaChatMessage,
    assistant: LlamaChatMessage,
    generation: LlamaGeneration,
}

impl LlamaConversationTurn {
    pub fn user(&self) -> &LlamaChatMessage {
        &self.user
    }
    pub fn assistant(&self) -> &LlamaChatMessage {
        &self.assistant
    }
    pub fn generation(&self) -> &LlamaGeneration {
        &self.generation
    }
}

/// A conversation borrows one immutable validated workflow and owns only its
/// committed transcript plus the existing generator/cache state.
#[derive(Debug)]
pub struct LlamaConversation<'a> {
    workflow: &'a LlamaPromptWorkflow,
    generator: LlamaGenerator<'a>,
    history: Vec<LlamaChatMessage>,
}

impl<'a> LlamaConversation<'a> {
    pub(crate) fn new(workflow: &'a LlamaPromptWorkflow) -> Self {
        Self {
            workflow,
            generator: LlamaGenerator::new(workflow.model(), workflow.tokenizer()),
            history: Vec::new(),
        }
    }

    pub fn history(&self) -> &[LlamaChatMessage] {
        &self.history
    }
    pub fn cache_len(&self) -> usize {
        self.generator.cache_len()
    }
    pub fn reset(&mut self) {
        self.history.clear();
        self.generator.clear();
    }

    /// Renders and generates against a staged transcript. Visible history is
    /// updated only after all template/token/model/cache work succeeds.
    pub fn send(
        &mut self,
        content: impl Into<String>,
        max_new_tokens: usize,
    ) -> Result<LlamaConversationTurn, LlamaConversationError> {
        let user = LlamaChatMessage::new(LlamaChatRole::User, content)?;
        if user.content().is_empty() {
            return Err(LlamaConversationError::EmptyInput);
        }
        let mut staged = self.history.clone();
        staged.push(user.clone());
        let prompt = self.workflow.render_chat(&staged)?;
        let generation =
            self.generator
                .generate_text(&prompt, max_new_tokens, super::LlamaSampling::Greedy)?;
        let assistant = LlamaChatMessage::new(LlamaChatRole::Assistant, generation.decoded())?;
        self.history.push(user.clone());
        self.history.push(assistant.clone());
        Ok(LlamaConversationTurn {
            user,
            assistant,
            generation,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LlamaConversationError {
    EmptyInput,
    Chat(LlamaChatError),
    Generation(LlamaGenerationError),
}
impl fmt::Display for LlamaConversationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llama conversation error: {self:?}")
    }
}
impl error::Error for LlamaConversationError {}
impl From<LlamaChatError> for LlamaConversationError {
    fn from(value: LlamaChatError) -> Self {
        Self::Chat(value)
    }
}
impl From<LlamaGenerationError> for LlamaConversationError {
    fn from(value: LlamaGenerationError) -> Self {
        Self::Generation(value)
    }
}
