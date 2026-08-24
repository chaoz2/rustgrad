use super::*;
use crate::{DType, Storage, TensorData};

fn checksum(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (0u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

fn envelope(payload: &[u8]) -> Vec<u8> {
    let mut bytes = b"RGFZ".to_vec();
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(&checksum(payload).to_le_bytes());
    bytes
}

#[test]
fn generated_cases_are_valid_bounded_and_order_independent() {
    let forward = (0..128)
        .map(|index| generate_case(0x1234, index))
        .collect::<Vec<_>>();
    assert!(forward.iter().all(|case| case.validate().is_ok()));
    let mut reverse = (0..128)
        .rev()
        .map(|index| (index, generate_case(0x1234, index)))
        .collect::<Vec<_>>();
    reverse.sort_by_key(|(index, _)| *index);
    assert_eq!(
        forward,
        reverse
            .into_iter()
            .map(|(_, case)| case)
            .collect::<Vec<_>>()
    );
    assert_ne!(
        forward,
        (0..128)
            .map(|index| generate_case(0x1235, index))
            .collect::<Vec<_>>()
    );
}

#[test]
fn fixed_campaigns_match_interpreter_and_strict_native() {
    let interpreter = run_campaign(FuzzConfig {
        seed: 7,
        cases: 64,
        native: false,
    })
    .unwrap();
    assert_eq!(interpreter.generated, 64);
    assert_eq!(interpreter.interpreter_matches, 64);
    assert!(interpreter.failures.is_empty());
    assert_eq!(interpreter.native_matches, 0);

    let native = run_campaign(FuzzConfig {
        seed: 11,
        cases: 16,
        native: true,
    })
    .unwrap();
    assert_eq!(native.interpreter_matches, 16);
    assert_eq!(native.native_matches, 16);
    assert_eq!(native.native_unsupported, 0);
    assert!(native.failures.is_empty());
    assert!(
        run_campaign(FuzzConfig {
            seed: 0,
            cases: 4097,
            native: false
        })
        .is_err()
    );
}

#[test]
fn unsupported_native_cases_remain_explicit() {
    let mut unsupported = 0;
    for (index, case) in regression_cases().iter().enumerate() {
        for comparison in run_case(3, index as u64, case, true).unwrap() {
            if matches!(comparison, FuzzComparison::Unsupported { .. }) {
                unsupported += 1;
            }
        }
    }
    assert!(unsupported > 0);
}

#[test]
fn regression_corpus_covers_edges_and_retains_genuine_concat_failure() {
    let cases = regression_cases();
    assert_eq!(cases.len(), 9);
    let mut failures = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        for comparison in run_case(0xfeed, index as u64, case, false).unwrap() {
            if let FuzzComparison::Failure(failure) = comparison {
                failures.push(*failure);
            }
        }
    }
    assert_eq!(failures.len(), 1);
    assert!(matches!(failures[0].case, FuzzCase::Concat { .. }));
    assert!(replay_failure(&failures[0]).unwrap());
}

#[test]
fn failure_artifact_is_deterministic_bounded_and_fail_closed() {
    let concat = regression_cases()
        .into_iter()
        .find(|case| matches!(case, FuzzCase::Concat { .. }))
        .unwrap();
    let failure = run_case(9, 3, &concat, false)
        .unwrap()
        .into_iter()
        .find_map(|comparison| match comparison {
            FuzzComparison::Failure(failure) => Some(*failure),
            _ => None,
        })
        .unwrap();
    let first = failure.to_bytes().unwrap();
    let second = failure.to_bytes().unwrap();
    assert_eq!(first, second);
    let decoded = FuzzFailureArtifact::from_bytes(&first).unwrap();
    assert_eq!(decoded, failure);
    assert_eq!(decoded.to_bytes().unwrap(), first);

    let mut corrupt = first.clone();
    corrupt[12] ^= 0x20;
    assert!(matches!(
        FuzzFailureArtifact::from_bytes(&corrupt),
        Err(FuzzArtifactError::Checksum)
    ));
    assert!(matches!(
        FuzzFailureArtifact::from_bytes(&first[..8]),
        Err(FuzzArtifactError::Truncated)
    ));
    let mut trailing = first.clone();
    trailing.push(0);
    assert!(matches!(
        FuzzFailureArtifact::from_bytes(&trailing),
        Err(FuzzArtifactError::Trailing)
    ));
    let mut version = first.clone();
    version[4..6].copy_from_slice(&2u16.to_le_bytes());
    assert!(matches!(
        FuzzFailureArtifact::from_bytes(&version),
        Err(FuzzArtifactError::Version(2))
    ));

    let mut value = serde_json::to_value(&failure).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(1));
    let unknown = envelope(&serde_json::to_vec(&value).unwrap());
    assert!(matches!(
        FuzzFailureArtifact::from_bytes(&unknown),
        Err(FuzzArtifactError::Json(_))
    ));
    let mut wrong_identity = serde_json::to_value(&failure).unwrap();
    wrong_identity.as_object_mut().unwrap().insert(
        "identity".into(),
        serde_json::json!(failure.identity.wrapping_add(1)),
    );
    assert!(matches!(
        FuzzFailureArtifact::from_bytes(&envelope(&serde_json::to_vec(&wrong_identity).unwrap())),
        Err(FuzzArtifactError::Identity)
    ));
    let mut invalid_case = serde_json::to_value(&failure).unwrap();
    invalid_case["case"]["rhs"]["dtype"] = serde_json::json!("Bool");
    assert!(matches!(
        FuzzFailureArtifact::from_bytes(&envelope(&serde_json::to_vec(&invalid_case).unwrap())),
        Err(FuzzArtifactError::Invalid(_))
    ));
    assert!(matches!(
        FuzzFailureArtifact::from_bytes(&vec![0; (1 << 20) + 15]),
        Err(FuzzArtifactError::TooLarge)
    ));
}

#[test]
fn checked_in_failure_corpus_decodes_and_reproduces() {
    let bytes = include_bytes!("../../tests/fuzz_corpus/failure-a30335b03b77b166.rgfz");
    let artifact = FuzzFailureArtifact::from_bytes(bytes).unwrap();
    assert!(matches!(artifact.case, FuzzCase::Concat { .. }));
    assert_eq!(artifact.actual_path, FuzzPath::CapturedInterpreter);
    assert!(replay_failure(&artifact).unwrap());
    assert_eq!(artifact.to_bytes().unwrap(), bytes);
}

#[test]
fn portable_tensor_raw_bits_round_trip_every_dtype() {
    let fixtures = vec![
        Storage::Bool(vec![false, true]),
        Storage::I8(vec![i8::MIN, i8::MAX]),
        Storage::U8(vec![0, u8::MAX]),
        Storage::I16(vec![i16::MIN, i16::MAX]),
        Storage::U16(vec![0, u16::MAX]),
        Storage::I32(vec![i32::MIN, i32::MAX]),
        Storage::U32(vec![0, u32::MAX]),
        Storage::I64(vec![i64::MIN, i64::MAX]),
        Storage::U64(vec![0, u64::MAX]),
        Storage::F16(vec![0x8000, 0x7e01]),
        Storage::BF16(vec![0x8000, 0x7fc1]),
        Storage::F32(vec![
            f32::from_bits(0x8000_0000),
            f32::from_bits(0x7fc0_0001),
        ]),
        Storage::F64(vec![
            f64::from_bits(0x8000_0000_0000_0000),
            f64::from_bits(0x7ff8_0000_0000_0001),
        ]),
    ];
    for storage in fixtures {
        let value = TensorData::from_storage([2], storage).unwrap();
        let portable = FuzzTensor::from_tensor(&value);
        assert_eq!(
            FuzzTensor::from_tensor(&portable.to_tensor().unwrap()),
            portable,
            "{:?}",
            value.dtype()
        );
    }
    let malformed = FuzzTensor {
        shape: vec![2],
        dtype: DType::Bool,
        bytes: vec![0, 2],
    };
    assert!(malformed.validate().is_err());
}

#[test]
fn minimizer_is_deterministic_and_never_loses_reproduction() {
    let original = regression_cases().remove(0);
    let first = minimize_case(&original, |candidate| candidate != &original);
    let second = minimize_case(&original, |candidate| candidate != &original);
    assert_eq!(first, second);
    assert_ne!(first, original);
    let unchanged = minimize_case(&original, |_| false);
    assert_eq!(unchanged, original);
}
