use super::*;
use crate::gguf::{GgufMetadataAccessError, GgufMetadataExpectation, GgufMetadataType, read_gguf};

fn byte_encoder() -> HashMap<u8, char> {
    byte_decoder()
        .into_iter()
        .map(|(character, byte)| (byte, character))
        .collect()
}

fn encoded_bytes(bytes: &[u8]) -> String {
    let encoder = byte_encoder();
    bytes.iter().map(|byte| encoder[byte]).collect()
}

fn byte_fallback_vocab() -> Vec<(String, u32)> {
    (0u16..256)
        .map(|byte| {
            let byte = byte as u8;
            (encoded_bytes(&[byte]), 1_000 + u32::from(byte))
        })
        .collect()
}

fn tokenizer_with(
    extra_normal: impl IntoIterator<Item = (&'static [u8], u32)>,
    specials: impl IntoIterator<Item = (&'static str, u32)>,
) -> SimpleTokenizer {
    let mut normal = byte_fallback_vocab();
    normal.extend(
        extra_normal
            .into_iter()
            .map(|(bytes, id)| (encoded_bytes(bytes), id)),
    );
    SimpleTokenizer::new(
        normal,
        specials
            .into_iter()
            .map(|(token, id)| (token.to_owned(), id)),
        TokenizerConfig::default(),
    )
    .unwrap()
}

#[test]
fn exact_word_token_wins_before_ranked_merges() {
    let tokenizer = tokenizer_with([(b"ab" as &[u8], 5), (b"bc", 4), (b"abc", 99)], []);
    assert_eq!(tokenizer.encode("abc").unwrap(), vec![99]);

    let tokenizer = tokenizer_with([(b"ab" as &[u8], 5), (b"bc", 4)], []);
    assert_eq!(
        tokenizer.encode("abc").unwrap(),
        vec![1_000 + u32::from(b'a'), 4]
    );
}

#[test]
fn unicode_word_splitting_matches_checked_in_source_cases() {
    let chunks: &[&[u8]] = &[
        b"hello",
        b" world",
        " 한국어".as_bytes(),
        " 中文".as_bytes(),
        " текст".as_bytes(),
        b" ",
        "١٢٣".as_bytes(),
        b"123",
        " 😊\n".as_bytes(),
        b"  ",
        b"\ttoday",
        b"\n",
        "'équivalent".as_bytes(),
        "²³".as_bytes(),
        "№".as_bytes(),
    ];
    let tokenizer = tokenizer_with(
        chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| (*chunk, index as u32 + 1)),
        [],
    );
    let text = "hello world 한국어 中文 текст ١٢٣ 123 😊\n  \ttoday\n'équivalent ²³№ ";
    assert_eq!(
        tokenizer.encode(text).unwrap(),
        vec![1, 2, 3, 4, 5, 6, 7, 6, 8, 9, 10, 11, 12, 13, 6, 14, 15, 6]
    );
    assert_eq!(
        tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap(),
        text
    );
}

#[test]
fn trailing_whitespace_lookahead_behavior_is_preserved() {
    let tokenizer = tokenizer_with([(b"  " as &[u8], 1), (b" hello", 2), (b"123", 3)], []);
    assert_eq!(tokenizer.encode("   hello").unwrap(), vec![1, 2]);
    assert_eq!(
        tokenizer.encode("  1234").unwrap(),
        vec![
            1_000 + u32::from(b' '),
            1_000 + u32::from(b' '),
            3,
            1_000 + u32::from(b'4')
        ]
    );
}

#[test]
fn specials_use_source_order_and_override_normal_sentence_coding() {
    let tokenizer = tokenizer_with([], [("<x>", 20), ("<x>y", 21)]);
    assert_eq!(
        tokenizer.encode("a<x>yb").unwrap(),
        vec![
            1_000 + u32::from(b'a'),
            20,
            1_000 + u32::from(b'y'),
            1_000 + u32::from(b'b')
        ]
    );
    assert_eq!(tokenizer.decode(&[20]).unwrap(), "<x>");
}

#[test]
fn byte_fallback_round_trips_unicode_and_streams_incomplete_utf8() {
    let tokenizer = tokenizer_with([], []);
    let text = "a😊실\u{85}z";
    let ids = tokenizer.encode(text).unwrap();
    assert_eq!(tokenizer.decode(&ids).unwrap(), text);

    let tokenizer = SimpleTokenizer::new(
        [
            (encoded_bytes(b" \xf0\x9f\x98"), 25_677),
            (encoded_bytes(b"\x8a"), 138),
        ],
        [],
        TokenizerConfig::default(),
    )
    .unwrap();
    let mut decoder = tokenizer.stream_decoder();
    assert_eq!(decoder.push(25_677).unwrap(), " ");
    assert_eq!(decoder.push(138).unwrap(), "😊");
    assert_eq!(decoder.finish(), "");
}

#[test]
fn coding_failures_are_structured() {
    let tokenizer =
        SimpleTokenizer::new([(encoded_bytes(b"a"), 1)], [], TokenizerConfig::default()).unwrap();
    assert_eq!(
        tokenizer.encode("b").unwrap_err().kind(),
        &TokenizerErrorKind::TokenNotFound(vec![b'b'])
    );
    assert_eq!(
        tokenizer.decode(&[99]).unwrap_err().kind(),
        &TokenizerErrorKind::UnknownTokenId(99)
    );
    assert!(matches!(
        SimpleTokenizer::new([("🙂".to_owned(), 1)], [], TokenizerConfig::default())
            .unwrap_err()
            .kind(),
        TokenizerErrorKind::InvalidNormalTokenCharacter { .. }
    ));
}

#[derive(Clone)]
enum Metadata<'a> {
    String(&'a str),
    Bool(bool),
    U32(u32),
    Strings(&'a [&'a str]),
    I32s(&'a [i32]),
}

fn push_string(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value.as_bytes());
}

fn gguf_fixture(entries: &[(&str, Metadata<'_>)]) -> Vec<u8> {
    let mut output = b"GGUF".to_vec();
    output.extend_from_slice(&3u32.to_le_bytes());
    output.extend_from_slice(&0u64.to_le_bytes());
    output.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (key, value) in entries {
        push_string(&mut output, key);
        match value {
            Metadata::String(value) => {
                output.extend_from_slice(&8u32.to_le_bytes());
                push_string(&mut output, value);
            }
            Metadata::Bool(value) => {
                output.extend_from_slice(&7u32.to_le_bytes());
                output.push(u8::from(*value));
            }
            Metadata::U32(value) => {
                output.extend_from_slice(&4u32.to_le_bytes());
                output.extend_from_slice(&value.to_le_bytes());
            }
            Metadata::Strings(values) => {
                output.extend_from_slice(&9u32.to_le_bytes());
                output.extend_from_slice(&8u32.to_le_bytes());
                output.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for value in *values {
                    push_string(&mut output, value);
                }
            }
            Metadata::I32s(values) => {
                output.extend_from_slice(&9u32.to_le_bytes());
                output.extend_from_slice(&5u32.to_le_bytes());
                output.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for value in *values {
                    output.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
    }
    output
}

fn tokenizer_fixture(
    tokens: &[&str],
    token_types: &[i32],
    preset: &str,
    extra: &[(&str, Metadata<'_>)],
) -> Vec<u8> {
    let mut entries = vec![
        (TOKENS_KEY, Metadata::Strings(tokens)),
        (TOKEN_TYPES_KEY, Metadata::I32s(token_types)),
        (PRE_KEY, Metadata::String(preset)),
    ];
    entries.extend_from_slice(extra);
    gguf_fixture(&entries)
}

#[test]
fn gguf_constructor_binds_types_special_ids_and_presets() {
    let hello = encoded_bytes(b"hello");
    let tokens = ["<s>", "</s>", "<|im_end|>", "<|im_end|>", hello.as_str()];
    let bytes = tokenizer_fixture(
        &tokens,
        &[3, 3, 3, 3, 1],
        "qwen35",
        &[(BOS_KEY, Metadata::U32(0)), (EOS_KEY, Metadata::U32(1))],
    );
    let file = read_gguf(&bytes).unwrap();
    let tokenizer = SimpleTokenizer::from_gguf(&file).unwrap();
    assert_eq!(tokenizer.preset(), TokenizerPreset::Qwen2);
    assert_eq!(tokenizer.bos_id(), Some(0));
    assert_eq!(tokenizer.eos_id(), 1);
    assert_eq!(tokenizer.eot_id(), Some(3));
    assert!(tokenizer.is_end(1));
    assert!(tokenizer.is_end(3));
    assert_eq!(tokenizer.encode("hello<|im_end|>").unwrap(), vec![4, 3]);
    assert_eq!(tokenizer.decode(&[0, 4, 1]).unwrap(), "<s>hello</s>");
}

#[test]
fn disabled_bos_does_not_require_or_read_bos_id() {
    let token = encoded_bytes(b"a");
    let tokens = [token.as_str()];
    let bytes = tokenizer_fixture(
        &tokens,
        &[1],
        "tekken",
        &[
            (ADD_BOS_KEY, Metadata::Bool(false)),
            (BOS_KEY, Metadata::String("not an integer")),
        ],
    );
    let file = read_gguf(&bytes).unwrap();
    let tokenizer = SimpleTokenizer::from_gguf(&file).unwrap();
    assert_eq!(tokenizer.bos_id(), None);
    assert_eq!(tokenizer.preset(), TokenizerPreset::Tekken);
}

#[test]
fn malformed_gguf_tokenizer_metadata_fails_before_construction() {
    let token = encoded_bytes(b"a");
    let tokens = [token.as_str()];
    let wrong_length = tokenizer_fixture(&tokens, &[], "llama3", &[]);
    let file = read_gguf(&wrong_length).unwrap();
    assert_eq!(
        SimpleTokenizer::from_gguf(&file).unwrap_err().kind(),
        &TokenizerErrorKind::MetadataLengthMismatch {
            tokens: 1,
            token_types: 0
        }
    );

    let wrong_type = gguf_fixture(&[
        (TOKENS_KEY, Metadata::Strings(&tokens)),
        (TOKEN_TYPES_KEY, Metadata::Strings(&["1"])),
        (PRE_KEY, Metadata::String("llama3")),
    ]);
    let file = read_gguf(&wrong_type).unwrap();
    assert_eq!(
        SimpleTokenizer::from_gguf(&file).unwrap_err().kind(),
        &TokenizerErrorKind::MalformedMetadata(GgufMetadataAccessError::ArrayElementTypeMismatch {
            key: TOKEN_TYPES_KEY.to_owned(),
            expected: GgufMetadataExpectation::IntegerArray,
            actual: GgufMetadataType::String,
        })
    );

    let unsupported = tokenizer_fixture(&tokens, &[1], "sentencepiece", &[]);
    let file = read_gguf(&unsupported).unwrap();
    assert_eq!(
        SimpleTokenizer::from_gguf(&file).unwrap_err().kind(),
        &TokenizerErrorKind::UnsupportedPreset("sentencepiece".to_owned())
    );
}

#[test]
fn gguf_control_token_ids_must_reference_the_vocabulary() {
    let token = encoded_bytes(b"a");
    let tokens = ["<s>", "</s>", token.as_str()];
    let types = [3, 3, 1];
    let valid = tokenizer_fixture(
        &tokens,
        &types,
        "llama3",
        &[(BOS_KEY, Metadata::U32(0)), (EOS_KEY, Metadata::U32(1))],
    );
    let tokenizer = SimpleTokenizer::from_gguf(&read_gguf(&valid).unwrap()).unwrap();
    assert_eq!(tokenizer.decode(&tokenizer.encode("a").unwrap()).unwrap(), "a");

    for key in [BOS_KEY, EOS_KEY, EOT_KEY] {
        let malformed = tokenizer_fixture(
            &tokens,
            &types,
            "llama3",
            &[(key, Metadata::U32(3))],
        );
        let error = SimpleTokenizer::from_gguf(&read_gguf(&malformed).unwrap()).unwrap_err();
        assert_eq!(
            error.kind(),
            &TokenizerErrorKind::ConfiguredTokenIdOutOfRange {
                key,
                token_id: 3,
                token_count: 3,
            }
        );
    }
}
