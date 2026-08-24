//! Source-compatible tokenizer support for tinygrad's checked-in GGUF LLM path.
//!
//! This is deliberately the small `SimpleTokenizer` contract used by
//! `tinygrad/llm/cli.py`: GPT-2 byte-alphabet decoding, its Unicode
//! pre-tokenizer, rank-ordered greedy merges, and ordered special tokens. It is
//! not a general SentencePiece or Hugging Face tokenizer implementation.

use crate::gguf::{GgufFile, GgufMetadataAccessError};
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    error, fmt,
    sync::LazyLock,
};

const TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const TOKEN_TYPES_KEY: &str = "tokenizer.ggml.token_type";
const PRE_KEY: &str = "tokenizer.ggml.pre";
const BOS_KEY: &str = "tokenizer.ggml.bos_token_id";
const EOS_KEY: &str = "tokenizer.ggml.eos_token_id";
const EOT_KEY: &str = "tokenizer.ggml.eot_token_id";
const ADD_BOS_KEY: &str = "tokenizer.ggml.add_bos_token";

// `ucat_range` in the checked-in Python source includes Unicode general
// categories only through U+323AF. Regex class intersection preserves that
// source boundary rather than silently expanding with newer Unicode tables.
const LETTER: &str = r"[\p{L}&&\x{0}-\x{323af}]";
const NUMBER: &str = r"[\p{N}&&\x{0}-\x{323af}]";
const SEPARATOR: &str = r"[\p{Z}&&\x{0}-\x{323af}]";

static WORD_PREFIX: LazyLock<Regex> = LazyLock::new(|| {
    let whitespace = format!(r"\t\n\x0B\x0C\r\x{{85}}{SEPARATOR}");
    let pattern = format!(
        r"^(?:(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n{NUMBER}{LETTER}]?{LETTER}+|{NUMBER}{{1,3}}| ?[^{whitespace}{NUMBER}{LETTER}]+[\r\n]*|[{whitespace}]*[\r\n]+)"
    );
    Regex::new(&pattern).expect("the checked-in SimpleTokenizer pattern is valid")
});

static WHITESPACE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(r"^[\t\n\x0B\x0C\r\x{{85}}{SEPARATOR}]+"))
        .expect("the checked-in SimpleTokenizer whitespace class is valid")
});

/// The pre-tokenizer presets accepted by tinygrad's checked-in tokenizer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TokenizerPreset {
    Llama3,
    LlamaV3,
    LlamaBpe,
    Qwen2,
    Olmo,
    KimiK2,
    Tekken,
    Glm4,
}

impl TokenizerPreset {
    /// Parses a GGUF `tokenizer.ggml.pre` value. The qwen3.5 aliases are
    /// normalized exactly as in tinygrad.
    pub fn parse(value: &str) -> Result<Self, TokenizerError> {
        Ok(match value {
            "llama3" => Self::Llama3,
            "llama-v3" => Self::LlamaV3,
            "llama-bpe" => Self::LlamaBpe,
            "qwen2" | "qwen35" | "qwen35moe" => Self::Qwen2,
            "olmo" => Self::Olmo,
            "kimi-k2" => Self::KimiK2,
            "tekken" => Self::Tekken,
            "glm4" => Self::Glm4,
            _ => {
                return Err(TokenizerError::new(TokenizerErrorKind::UnsupportedPreset(
                    value.to_owned(),
                )));
            }
        })
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Llama3 => "llama3",
            Self::LlamaV3 => "llama-v3",
            Self::LlamaBpe => "llama-bpe",
            Self::Qwen2 => "qwen2",
            Self::Olmo => "olmo",
            Self::KimiK2 => "kimi-k2",
            Self::Tekken => "tekken",
            Self::Glm4 => "glm4",
        }
    }
}

/// Explicit special-token configuration for a [`SimpleTokenizer`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerConfig {
    pub preset: TokenizerPreset,
    pub bos_id: Option<u32>,
    pub eos_id: u32,
    pub eot_id: Option<u32>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            preset: TokenizerPreset::Llama3,
            bos_id: None,
            eos_id: 0,
            eot_id: None,
        }
    }
}

/// Structured tokenizer construction or coding failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenizerErrorKind {
    UnsupportedPreset(String),
    MissingMetadata(&'static str),
    MalformedMetadata(GgufMetadataAccessError),
    MetadataLengthMismatch { tokens: usize, token_types: usize },
    TokenIdOutOfRange { key: &'static str, value: u64 },
    InvalidNormalTokenCharacter { token_id: u32, character: char },
    EmptySpecialToken { token_id: u32 },
    TokenNotFound(Vec<u8>),
    UnknownTokenId(u32),
}

/// A tokenizer error whose kind can be matched without parsing display text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerError {
    kind: TokenizerErrorKind,
}

impl TokenizerError {
    const fn new(kind: TokenizerErrorKind) -> Self {
        Self { kind }
    }

    pub const fn kind(&self) -> &TokenizerErrorKind {
        &self.kind
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tokenizer error: {:?}", self.kind)
    }
}

impl error::Error for TokenizerError {}

impl From<GgufMetadataAccessError> for TokenizerError {
    fn from(value: GgufMetadataAccessError) -> Self {
        Self::new(TokenizerErrorKind::MalformedMetadata(value))
    }
}

/// The bounded tokenizer used by tinygrad's checked-in GGUF command-line LLM.
#[derive(Clone, Debug)]
pub struct SimpleTokenizer {
    normal_tokens: HashMap<Vec<u8>, u32>,
    special_tokens: Vec<(String, u32)>,
    tok2bytes: HashMap<u32, Vec<u8>>,
    config: TokenizerConfig,
}

impl SimpleTokenizer {
    /// Constructs a tokenizer from the same byte-alphabet token strings used
    /// by tinygrad. Duplicate token strings retain their first special-token
    /// match position and their last ID, matching Python dictionary behavior.
    pub fn new(
        normal_tokens: impl IntoIterator<Item = (String, u32)>,
        special_tokens: impl IntoIterator<Item = (String, u32)>,
        config: TokenizerConfig,
    ) -> Result<Self, TokenizerError> {
        let byte_decoder = byte_decoder();
        let mut normal = HashMap::new();
        let mut normal_order = Vec::new();
        let mut seen_normal = HashSet::new();
        for (token, token_id) in normal_tokens {
            let mut bytes = Vec::with_capacity(token.len());
            for character in token.chars() {
                bytes.push(*byte_decoder.get(&character).ok_or_else(|| {
                    TokenizerError::new(TokenizerErrorKind::InvalidNormalTokenCharacter {
                        token_id,
                        character,
                    })
                })?);
            }
            if seen_normal.insert(bytes.clone()) {
                normal_order.push(bytes.clone());
            }
            normal.insert(bytes, token_id);
        }

        let mut ordered_specials: Vec<(String, u32)> = Vec::new();
        let mut special_positions: HashMap<String, usize> = HashMap::new();
        for (token, token_id) in special_tokens {
            if token.is_empty() {
                return Err(TokenizerError::new(TokenizerErrorKind::EmptySpecialToken {
                    token_id,
                }));
            }
            if let Some(position) = special_positions.get(&token).copied() {
                ordered_specials[position].1 = token_id;
            } else {
                special_positions.insert(token.clone(), ordered_specials.len());
                ordered_specials.push((token, token_id));
            }
        }

        let mut tok2bytes = HashMap::new();
        for token in normal_order {
            tok2bytes.insert(normal[&token], token);
        }
        for (token, token_id) in &ordered_specials {
            tok2bytes.insert(*token_id, token.as_bytes().to_vec());
        }
        Ok(Self {
            normal_tokens: normal,
            special_tokens: ordered_specials,
            tok2bytes,
            config,
        })
    }

    /// Constructs the exact tokenizer described by a validated GGUF file.
    pub fn from_gguf(file: &GgufFile<'_>) -> Result<Self, TokenizerError> {
        let tokens = file
            .metadata_strings(TOKENS_KEY)?
            .ok_or_else(|| TokenizerError::new(TokenizerErrorKind::MissingMetadata(TOKENS_KEY)))?;
        let token_types = file.metadata_integers(TOKEN_TYPES_KEY)?.ok_or_else(|| {
            TokenizerError::new(TokenizerErrorKind::MissingMetadata(TOKEN_TYPES_KEY))
        })?;
        if tokens.len() != token_types.len() {
            return Err(TokenizerError::new(
                TokenizerErrorKind::MetadataLengthMismatch {
                    tokens: tokens.len(),
                    token_types: token_types.len(),
                },
            ));
        }
        let preset =
            TokenizerPreset::parse(file.metadata_string(PRE_KEY)?.ok_or_else(|| {
                TokenizerError::new(TokenizerErrorKind::MissingMetadata(PRE_KEY))
            })?)?;

        let mut normal_tokens = Vec::new();
        let mut special_tokens = Vec::new();
        for (index, (token, token_type)) in tokens.into_iter().zip(token_types).enumerate() {
            let token_id = u32::try_from(index).map_err(|_| {
                TokenizerError::new(TokenizerErrorKind::TokenIdOutOfRange {
                    key: TOKENS_KEY,
                    value: index as u64,
                })
            })?;
            if token_type == 1 {
                normal_tokens.push((token.to_owned(), token_id));
            } else {
                special_tokens.push((token.to_owned(), token_id));
            }
        }

        let add_bos = file.metadata_bool(ADD_BOS_KEY)?.unwrap_or(true);
        let bos_id = if add_bos {
            optional_token_id(file, BOS_KEY)?
        } else {
            None
        };
        let eos_id = optional_token_id(file, EOS_KEY)?.unwrap_or(0);
        let special_im_end = special_tokens
            .iter()
            .rev()
            .find_map(|(token, id)| (token == "<|im_end|>").then_some(*id));
        let eot_id = optional_token_id(file, EOT_KEY)?.or(special_im_end);
        Self::new(
            normal_tokens,
            special_tokens,
            TokenizerConfig {
                preset,
                bos_id,
                eos_id,
                eot_id,
            },
        )
    }

    pub const fn preset(&self) -> TokenizerPreset {
        self.config.preset
    }

    pub const fn bos_id(&self) -> Option<u32> {
        self.config.bos_id
    }

    pub const fn eos_id(&self) -> u32 {
        self.config.eos_id
    }

    pub const fn eot_id(&self) -> Option<u32> {
        self.config.eot_id
    }

    /// Encodes text with ordered special-token splitting and tinygrad's
    /// source-exact Unicode word pre-tokenizer.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut tokens = Vec::new();
        let mut position = 0;
        while let Some((start, end, token_id)) = self.next_special(text, position) {
            self.encode_sentence(&text[position..start], &mut tokens)?;
            tokens.push(token_id);
            position = end;
        }
        self.encode_sentence(&text[position..], &mut tokens)?;
        Ok(tokens)
    }

    /// Decodes IDs by concatenating token bytes and replacing malformed UTF-8,
    /// matching Python's `decode(errors="replace")` behavior.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();
        for &token_id in ids {
            bytes.extend_from_slice(self.token_bytes(token_id)?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn stream_decoder(&self) -> StreamingDecoder<'_> {
        StreamingDecoder {
            tokenizer: self,
            pending: Vec::new(),
            finished: false,
        }
    }

    pub fn is_end(&self, token_id: u32) -> bool {
        token_id == self.config.eos_id || self.config.eot_id == Some(token_id)
    }

    fn token_bytes(&self, token_id: u32) -> Result<&[u8], TokenizerError> {
        self.tok2bytes
            .get(&token_id)
            .map(Vec::as_slice)
            .ok_or_else(|| TokenizerError::new(TokenizerErrorKind::UnknownTokenId(token_id)))
    }

    fn next_special(&self, text: &str, position: usize) -> Option<(usize, usize, u32)> {
        self.special_tokens
            .iter()
            .enumerate()
            .filter_map(|(order, (token, token_id))| {
                text[position..]
                    .find(token)
                    .map(|relative| (position + relative, order, token.len(), *token_id))
            })
            .min_by_key(|(start, order, _, _)| (*start, *order))
            .map(|(start, _, len, token_id)| (start, start + len, token_id))
    }

    fn encode_sentence(&self, sentence: &str, output: &mut Vec<u32>) -> Result<(), TokenizerError> {
        let mut remaining = sentence;
        while !remaining.is_empty() {
            let word_len = if let Some(found) = WORD_PREFIX.find(remaining) {
                debug_assert_eq!(found.start(), 0);
                found.end()
            } else if let Some(whitespace) = WHITESPACE.find(remaining) {
                let run = &remaining[..whitespace.end()];
                if whitespace.end() == remaining.len() {
                    whitespace.end()
                } else {
                    let last_len = run.chars().next_back().expect("non-empty match").len_utf8();
                    if run.chars().count() > 1 {
                        whitespace.end() - last_len
                    } else {
                        whitespace.end()
                    }
                }
            } else {
                remaining
                    .chars()
                    .next()
                    .expect("non-empty input")
                    .len_utf8()
            };
            self.encode_word(&remaining.as_bytes()[..word_len], output)?;
            remaining = &remaining[word_len..];
        }
        Ok(())
    }

    fn encode_word(&self, word: &[u8], output: &mut Vec<u32>) -> Result<(), TokenizerError> {
        if let Some(token_id) = self.normal_tokens.get(word) {
            output.push(*token_id);
            return Ok(());
        }
        let mut parts: Vec<Vec<u8>> = word.iter().map(|byte| vec![*byte]).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for position in 0..parts.len().saturating_sub(1) {
                let mut merged = parts[position].clone();
                merged.extend_from_slice(&parts[position + 1]);
                if let Some(&rank) = self.normal_tokens.get(&merged) {
                    let candidate = (rank, position);
                    if best.is_none_or(|current| candidate < current) {
                        best = Some(candidate);
                    }
                }
            }
            let Some((_, position)) = best else {
                break;
            };
            let right = parts.remove(position + 1);
            parts[position].extend_from_slice(&right);
        }
        for part in parts {
            output.push(*self.normal_tokens.get(&part).ok_or_else(|| {
                TokenizerError::new(TokenizerErrorKind::TokenNotFound(part.clone()))
            })?);
        }
        Ok(())
    }
}

/// Incremental UTF-8 decoder retaining an incomplete final code point between
/// tokens and replacing malformed input deterministically.
pub struct StreamingDecoder<'a> {
    tokenizer: &'a SimpleTokenizer,
    pending: Vec<u8>,
    finished: bool,
}

impl StreamingDecoder<'_> {
    /// Decodes one token, retaining only an incomplete trailing UTF-8 sequence.
    pub fn push(&mut self, token_id: u32) -> Result<String, TokenizerError> {
        if self.finished {
            return Ok(String::new());
        }
        self.pending
            .extend_from_slice(self.tokenizer.token_bytes(token_id)?);
        Ok(drain_utf8(&mut self.pending, false))
    }

    /// Flushes an incomplete sequence with the Unicode replacement character.
    pub fn finish(&mut self) -> String {
        self.finished = true;
        drain_utf8(&mut self.pending, true)
    }
}

fn optional_token_id(
    file: &GgufFile<'_>,
    key: &'static str,
) -> Result<Option<u32>, TokenizerError> {
    file.metadata_u64(key)?
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                TokenizerError::new(TokenizerErrorKind::TokenIdOutOfRange { key, value })
            })
        })
        .transpose()
}

fn byte_decoder() -> HashMap<char, u8> {
    let direct = (33u16..127)
        .chain(161..173)
        .chain(174..256)
        .map(|byte| byte as u8)
        .collect::<Vec<_>>();
    let mut decoder = direct
        .iter()
        .copied()
        .map(|byte| (char::from(byte), byte))
        .collect::<HashMap<_, _>>();
    for (index, byte) in (0u16..256)
        .map(|byte| byte as u8)
        .filter(|byte| !direct.contains(byte))
        .enumerate()
    {
        decoder.insert(
            char::from_u32(256 + index as u32).expect("byte alphabet code point"),
            byte,
        );
    }
    decoder
}

fn drain_utf8(pending: &mut Vec<u8>, final_chunk: bool) -> String {
    let mut output = String::new();
    let mut consumed = 0;
    while consumed < pending.len() {
        match std::str::from_utf8(&pending[consumed..]) {
            Ok(valid) => {
                output.push_str(valid);
                consumed = pending.len();
            }
            Err(error) => {
                let valid_end = consumed + error.valid_up_to();
                output.push_str(
                    std::str::from_utf8(&pending[consumed..valid_end])
                        .expect("valid_up_to is valid UTF-8"),
                );
                match error.error_len() {
                    Some(invalid_len) => {
                        output.push('\u{fffd}');
                        consumed = valid_end + invalid_len;
                    }
                    None if final_chunk => {
                        output.push('\u{fffd}');
                        consumed = pending.len();
                    }
                    None => {
                        consumed = valid_end;
                        break;
                    }
                }
            }
        }
    }
    pending.drain(..consumed);
    output
}

#[cfg(test)]
mod tests;
