use super::*;
use crate::{Backend, CpuBackend, DType, Scalar, Storage, TensorData};
use std::{
    fs::{self, File},
    sync::atomic::{AtomicU64, Ordering},
};

static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn test_directory(label: &str) -> std::path::PathBuf {
    let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rustgrad-fuzz-{label}-{}-{sequence}",
        std::process::id()
    ))
}

fn historical_concat_failure() -> FuzzFailureArtifact {
    let concat = regression_cases()
        .into_iter()
        .find(|case| matches!(case, FuzzCase::Concat { .. }))
        .unwrap();
    let built = concat.build().unwrap();
    let expected = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
    FuzzFailureArtifact::new(
        9,
        3,
        concat,
        FuzzPath::CapturedInterpreter,
        FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&expected),
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "historical movement dispatch failure".into(),
        },
    )
    .unwrap()
}

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
            if matches!(
                comparison,
                FuzzComparison::Unsupported {
                    path: FuzzPath::NativeScalar | FuzzPath::NativeVector,
                    ..
                }
            ) {
                unsupported += 1;
            }
        }
    }
    assert!(unsupported > 0);
}

#[test]
fn regression_cases_cover_edges_without_current_failures() {
    let cases = regression_cases();
    assert_eq!(cases.len(), 9);
    for (index, case) in cases.iter().enumerate() {
        for comparison in run_case(0xfeed, index as u64, case, false).unwrap() {
            assert!(matches!(
                comparison,
                FuzzComparison::Match {
                    path: FuzzPath::CapturedInterpreter,
                    ..
                }
            ));
        }
    }
}

#[test]
fn failure_artifact_is_deterministic_bounded_and_fail_closed() {
    let failure = historical_concat_failure();
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
    assert_eq!(
        replay_failure(&failure).unwrap(),
        FuzzReplayStatus::Resolved
    );
}

#[test]
fn nested_unknown_fields_and_equal_outcomes_are_rejected() {
    let failure = historical_concat_failure();
    for path in [["case"].as_slice(), ["expected"].as_slice()] {
        let mut value = serde_json::to_value(&failure).unwrap();
        let mut target = &mut value;
        for component in path {
            target = &mut target[*component];
        }
        target
            .as_object_mut()
            .unwrap()
            .insert("unknown_nested".into(), serde_json::json!(true));
        let bytes = envelope(&serde_json::to_vec(&value).unwrap());
        assert!(matches!(
            FuzzFailureArtifact::from_bytes(&bytes),
            Err(FuzzArtifactError::Json(_))
        ));
    }

    let case = regression_cases().remove(0);
    let expected = FuzzOutcome::value(&TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F32));
    let actual = FuzzOutcome::value(&TensorData::scalar_with_dtype(Scalar::F(2.0), DType::F32));
    let float_failure = FuzzFailureArtifact::new(
        1,
        2,
        case.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::FloatTolerance {
            absolute_bits: 1e-6f64.to_bits(),
            relative_bits: 1e-6f64.to_bits(),
        },
        expected.clone(),
        actual,
    )
    .unwrap();
    let mut policy_unknown = serde_json::to_value(&float_failure).unwrap();
    policy_unknown["policy"]["float_tolerance"]
        .as_object_mut()
        .unwrap()
        .insert("unknown_nested".into(), serde_json::json!(true));
    assert!(matches!(
        FuzzFailureArtifact::from_bytes(&envelope(&serde_json::to_vec(&policy_unknown).unwrap())),
        Err(FuzzArtifactError::Json(_))
    ));

    assert!(matches!(
        FuzzFailureArtifact::new(
            1,
            2,
            case,
            FuzzPath::NativeScalar,
            FuzzComparisonPolicy::ExactBytes,
            expected.clone(),
            expected,
        ),
        Err(FuzzArtifactError::Invalid(_))
    ));
}

#[test]
fn replay_status_distinguishes_reproduced_changed_resolved_and_unsupported() {
    use super::execute::{PathError, replay_failure_with};

    let failure = historical_concat_failure();
    let expected = failure.expected.clone();
    let recorded_actual = failure.actual.clone();
    let mut reproduced = |_: &FuzzCase, path: FuzzPath| match path {
        FuzzPath::CpuOracle => Ok(expected.clone()),
        _ => Ok(recorded_actual.clone()),
    };
    assert_eq!(
        replay_failure_with(&failure, &mut reproduced).unwrap(),
        FuzzReplayStatus::Reproduced
    );

    let mut resolved = |_: &FuzzCase, _: FuzzPath| Ok(expected.clone());
    assert_eq!(
        replay_failure_with(&failure, &mut resolved).unwrap(),
        FuzzReplayStatus::Resolved
    );

    let mut changed = |_: &FuzzCase, path: FuzzPath| match path {
        FuzzPath::CpuOracle => Ok(expected.clone()),
        _ => Ok(FuzzOutcome::Error {
            class: "execute".into(),
            detail: "different failure".into(),
        }),
    };
    assert_eq!(
        replay_failure_with(&failure, &mut changed).unwrap(),
        FuzzReplayStatus::Changed
    );

    let mut unsupported = |_: &FuzzCase, path: FuzzPath| match path {
        FuzzPath::CpuOracle => Ok(expected.clone()),
        _ => Err(PathError::Unsupported("not supported".into())),
    };
    assert!(matches!(
        replay_failure_with(&failure, &mut unsupported).unwrap(),
        FuzzReplayStatus::Unsupported {
            path: FuzzPath::CapturedInterpreter,
            ..
        }
    ));
}

#[test]
fn minimization_never_blesses_unsupported_as_a_mismatch() {
    use super::execute::{PathError, compare_path_with};

    let original = regression_cases().remove(0);
    let built = original.build().unwrap();
    let expected = FuzzOutcome::value(
        &CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap(),
    );
    let mut execute = |candidate: &FuzzCase, path: FuzzPath| match path {
        FuzzPath::CpuOracle => Ok(expected.clone()),
        _ if candidate == &original => Err(PathError::Failed {
            class: "execute",
            detail: "stable failure".into(),
        }),
        _ => Err(PathError::Unsupported("candidate unsupported".into())),
    };
    let comparison =
        compare_path_with(7, 0, &original, FuzzPath::CapturedInterpreter, &mut execute).unwrap();
    let FuzzComparison::Failure(failure) = comparison else {
        panic!("expected a preserved failure");
    };
    assert_eq!(failure.case, original);
    assert!(matches!(
        failure.actual,
        FuzzOutcome::Error { ref class, .. } if class == "execute"
    ));
}

#[test]
fn campaign_accounting_rejects_interpreter_unsupported_only() {
    use super::execute::record_comparison;

    let mut report = FuzzCampaign {
        seed: 1,
        generated: 1,
        interpreter_matches: 0,
        native_matches: 0,
        native_unsupported: 0,
        failures: vec![],
    };
    assert!(
        record_comparison(
            &mut report,
            0,
            FuzzComparison::Unsupported {
                path: FuzzPath::CapturedInterpreter,
                reason: "coverage hole".into(),
            },
        )
        .is_err()
    );
    record_comparison(
        &mut report,
        0,
        FuzzComparison::Unsupported {
            path: FuzzPath::NativeScalar,
            reason: "native policy".into(),
        },
    )
    .unwrap();
    assert_eq!(report.native_unsupported, 1);
}

#[test]
fn replay_output_requires_exactly_one_value() {
    use super::execute::{PathError, exact_single_output};

    assert!(matches!(
        exact_single_output(vec![]),
        Err(PathError::Failed {
            class: "output_count",
            ..
        })
    ));
    let value = TensorData::scalar_with_dtype(Scalar::I(1), DType::I32);
    assert!(matches!(
        exact_single_output(vec![value.clone(), value]),
        Err(PathError::Failed {
            class: "output_count",
            ..
        })
    ));
}

#[test]
fn corpus_inventory_reports_and_explicitly_prunes_resolved_artifacts() {
    let directory = test_directory("resolved-corpus");
    let failure = historical_concat_failure();
    assert!(write_failure_artifact_atomic(&directory, &failure).unwrap());
    assert!(!write_failure_artifact_atomic(&directory, &failure).unwrap());

    let checked = reconcile_regression_corpus(&directory, FuzzCorpusMode::Check).unwrap();
    assert_eq!(checked.inventoried, 1);
    assert_eq!(checked.resolved, 1);
    assert_eq!(checked.pruned, 0);
    assert!(!checked.is_clean());

    let pruned =
        reconcile_regression_corpus(&directory, FuzzCorpusMode::WriteAndPruneResolved).unwrap();
    assert_eq!(pruned.resolved, 1);
    assert_eq!(pruned.pruned, 1);
    assert!(pruned.is_clean());
    assert!(fs::read_dir(&directory).unwrap().next().is_none());
    fs::remove_dir(&directory).unwrap();
}

#[test]
fn artifact_file_cap_is_enforced_before_bulk_read() {
    let directory = test_directory("oversized");
    fs::create_dir(&directory).unwrap();
    let path = directory.join("oversized.rgfz");
    let file = File::create(&path).unwrap();
    file.set_len(MAX_FUZZ_ARTIFACT_FILE_BYTES as u64 + 1)
        .unwrap();
    let error = read_failure_artifact(&path).unwrap_err();
    assert!(error.contains("exceeds"));
    fs::remove_file(path).unwrap();
    fs::remove_dir(directory).unwrap();
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
