use crate::{
    gguf::{GgufFile, GgufMetadataAccessError},
    tokenizer::{SimpleTokenizer, TokenizerError, TokenizerPreset},
};
use std::{error, fmt};

const CHAT_TEMPLATE_KEY: &str = "tokenizer.chat_template";
const MAX_TEMPLATE_BYTES: usize = 64 * 1024;
const MAX_MESSAGE_BYTES: usize = 1024 * 1024;
const MAX_CHAT_BYTES: usize = 4 * 1024 * 1024;
const HEADER_START: &str = "<|start_header_id|>";
const HEADER_END: &str = "<|end_header_id|>\n\n";

/// The one Jinja source string accepted from GGUF metadata. Its semantics are
/// exactly the checked-in tinygrad `FallbackTemplate` for the Llama preset.
pub const LLAMA_SIMPLE_CHAT_TEMPLATE: &str = "{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + eos_token }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}";

/// Roles supported by the source-compatible plain-text Llama formatter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlamaChatRole {
    System,
    User,
    Assistant,
}

impl LlamaChatRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// One validated string-content message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaChatMessage {
    role: LlamaChatRole,
    content: String,
}

impl LlamaChatMessage {
    /// Creates a bounded string-content message; tool and multimodal content are unsupported.
    pub fn new(role: LlamaChatRole, content: impl Into<String>) -> Result<Self, LlamaChatError> {
        let content = content.into();
        if content.len() > MAX_MESSAGE_BYTES {
            return Err(LlamaChatError::MessageTooLong {
                bytes: content.len(),
                maximum: MAX_MESSAGE_BYTES,
            });
        }
        Ok(Self { role, content })
    }
    pub const fn role(&self) -> LlamaChatRole {
        self.role
    }
    pub fn content(&self) -> &str {
        &self.content
    }
}

/// Checked Llama chat formatter selected from absent or exactly supported GGUF metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaChatTemplate {
    metadata_present: bool,
}

impl LlamaChatTemplate {
    /// Accepts an absent template (tinygrad fallback) or the exact supported
    /// Llama template. Other Jinja/control syntax is never approximated.
    pub fn from_gguf(file: &GgufFile<'_>) -> Result<Self, LlamaChatError> {
        let Some(template) = file.metadata_string(CHAT_TEMPLATE_KEY)? else {
            return Ok(Self {
                metadata_present: false,
            });
        };
        if template.len() > MAX_TEMPLATE_BYTES {
            return Err(LlamaChatError::TemplateTooLong {
                bytes: template.len(),
                maximum: MAX_TEMPLATE_BYTES,
            });
        }
        if template != LLAMA_SIMPLE_CHAT_TEMPLATE {
            return Err(
                if template.contains("{{") || template.contains("{%") || template.contains("{#") {
                    LlamaChatError::UnsupportedJinja
                } else {
                    LlamaChatError::UnsupportedTemplate
                },
            );
        }
        Ok(Self {
            metadata_present: true,
        })
    }

    /// Returns whether the exact supported template was present rather than fallback-selected.
    pub const fn metadata_present(self) -> bool {
        self.metadata_present
    }

    /// Applies the exact checked-in Llama fallback semantics.
    pub fn render(
        self,
        tokenizer: &SimpleTokenizer,
        messages: &[LlamaChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, LlamaChatError> {
        if tokenizer.preset() != TokenizerPreset::Llama3 {
            return Err(LlamaChatError::UnsupportedPreset(tokenizer.preset()));
        }
        let bos = tokenizer
            .bos_id()
            .map_or_else(|| Ok(String::new()), |id| tokenizer.decode(&[id]))?;
        let eos = tokenizer.decode(&[tokenizer.eos_id()])?;
        let mut required = bos.len();
        for message in messages {
            required = required
                .checked_add(HEADER_START.len())
                .and_then(|value| value.checked_add(message.role.as_str().len()))
                .and_then(|value| value.checked_add(HEADER_END.len()))
                .and_then(|value| value.checked_add(message.content.len()))
                .and_then(|value| value.checked_add(eos.len()))
                .ok_or(LlamaChatError::ChatOverflow)?;
        }
        if add_generation_prompt {
            required = required
                .checked_add(HEADER_START.len())
                .and_then(|value| value.checked_add("assistant".len()))
                .and_then(|value| value.checked_add(HEADER_END.len()))
                .ok_or(LlamaChatError::ChatOverflow)?;
        }
        if required > MAX_CHAT_BYTES {
            return Err(LlamaChatError::ChatTooLong {
                bytes: required,
                maximum: MAX_CHAT_BYTES,
            });
        }
        let mut output = String::with_capacity(required);
        output.push_str(&bos);
        for message in messages {
            output.push_str(HEADER_START);
            output.push_str(message.role.as_str());
            output.push_str(HEADER_END);
            output.push_str(&message.content);
            output.push_str(&eos);
        }
        if add_generation_prompt {
            output.push_str(HEADER_START);
            output.push_str("assistant");
            output.push_str(HEADER_END);
        }
        Ok(output)
    }
}

/// Structured metadata, tokenizer, bound, or unsupported-template failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LlamaChatError {
    Metadata(GgufMetadataAccessError),
    Tokenizer(TokenizerError),
    TemplateTooLong { bytes: usize, maximum: usize },
    UnsupportedJinja,
    UnsupportedTemplate,
    UnsupportedPreset(TokenizerPreset),
    MessageTooLong { bytes: usize, maximum: usize },
    ChatTooLong { bytes: usize, maximum: usize },
    ChatOverflow,
}

impl fmt::Display for LlamaChatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llama chat template error: {self:?}")
    }
}
impl error::Error for LlamaChatError {}
impl From<GgufMetadataAccessError> for LlamaChatError {
    fn from(value: GgufMetadataAccessError) -> Self {
        Self::Metadata(value)
    }
}
impl From<TokenizerError> for LlamaChatError {
    fn from(value: TokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}
