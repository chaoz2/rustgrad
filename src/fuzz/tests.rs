use super::*;
use crate::{
    Backend, CapturedSchedule, CompareOp, CpuBackend, CpuJit, DType, LogicalOp, Op, Scalar,
    MovementKernelKind, MovementKernelPlan, Storage, TensorData, UArg, UOpKind, UnaryOp,
    schedule,
};
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
fn generated_concat_cases_are_valid_diverse_and_deterministic() {
    let mut found = false;
    let mut lhs_zero = false;
    let mut rhs_zero = false;
    let mut lhs_nonzero = false;
    let mut rhs_nonzero = false;
    let mut axis_zero = false;
    let mut axis_one = false;
    let mut f32 = false;
    let mut i32 = false;
    let mut f16 = false;
    let mut boolean = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..256 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::Concat { lhs, rhs, axis } = case else {
                continue;
            };
            found = true;
            assert!(!lhs.shape.is_empty());
            assert_eq!(lhs.dtype, rhs.dtype);
            assert_eq!(lhs.shape.len(), rhs.shape.len());
            assert!(axis < lhs.shape.len());
            for dimension in 0..lhs.shape.len() {
                if dimension != axis {
                    assert_eq!(lhs.shape[dimension], rhs.shape[dimension]);
                }
            }
            lhs_zero |= lhs.shape[axis] == 0;
            rhs_zero |= rhs.shape[axis] == 0;
            lhs_nonzero |= lhs.shape[axis] != 0;
            rhs_nonzero |= rhs.shape[axis] != 0;
            axis_zero |= axis == 0;
            axis_one |= lhs.shape.len() >= 2 && axis == 1;
            match lhs.dtype {
                DType::F32 => f32 = true,
                DType::I32 => i32 = true,
                DType::F16 => f16 = true,
                DType::Bool => boolean = true,
                _ => unreachable!("concat generator selects only native-supported dtypes"),
            }
        }
    }

    assert!(found);
    assert!(lhs_zero && rhs_zero && lhs_nonzero && rhs_nonzero);
    assert!(axis_zero && axis_one);
    assert!(f32 && i32 && f16 && boolean);
}

#[test]
fn generated_unary_cases_are_valid_diverse_and_deterministic() {
    let mut found = false;
    let mut neg = false;
    let mut abs = false;
    let mut f32 = false;
    let mut i32 = false;
    let mut scalar = false;
    let mut empty = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..256 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::Unary { op, input } = case else {
                continue;
            };
            found = true;
            neg |= op == FuzzUnaryOp::Neg;
            abs |= op == FuzzUnaryOp::Abs;
            f32 |= input.dtype == DType::F32;
            i32 |= input.dtype == DType::I32;
            scalar |= input.shape.is_empty();
            empty |= input.shape.iter().any(|extent| *extent == 0);
            assert!(matches!(input.dtype, DType::F32 | DType::I32));
        }
    }

    assert!(found);
    assert!(neg && abs);
    assert!(f32 && i32);
    assert!(scalar && empty);
}

#[test]
fn generated_compare_cases_are_valid_diverse_and_deterministic() {
    let mut found = false;
    let mut ops = std::collections::BTreeSet::new();
    let mut f32 = false;
    let mut i32 = false;
    let mut scalar = false;
    let mut empty = false;
    let mut scalar_rhs = false;
    let mut matching_rhs = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..512 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::Compare { op, lhs, rhs } = case else {
                continue;
            };
            found = true;
            ops.insert(op);
            f32 |= lhs.dtype == DType::F32;
            i32 |= lhs.dtype == DType::I32;
            scalar |= lhs.shape.is_empty();
            empty |= lhs.shape.iter().any(|extent| *extent == 0);
            scalar_rhs |= rhs.shape.is_empty();
            matching_rhs |= rhs.shape == lhs.shape;
            assert_eq!(lhs.dtype, rhs.dtype);
            assert!(matches!(lhs.dtype, DType::F32 | DType::I32));
        }
    }

    assert!(found);
    assert_eq!(ops.len(), 6);
    assert!(f32 && i32 && scalar && empty && scalar_rhs && matching_rhs);
}

#[test]
fn generated_logical_cases_are_valid_diverse_and_deterministic() {
    let mut found = false;
    let mut and = false;
    let mut or = false;
    let mut true_lane = false;
    let mut false_lane = false;
    let mut scalar = false;
    let mut empty = false;
    let mut scalar_rhs = false;
    let mut matching_rhs = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..512 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::Logical { op, lhs, rhs } = case else {
                continue;
            };
            found = true;
            and |= op == FuzzLogicalOp::And;
            or |= op == FuzzLogicalOp::Or;
            true_lane |= lhs.bytes.contains(&1) || rhs.bytes.contains(&1);
            false_lane |= lhs.bytes.contains(&0) || rhs.bytes.contains(&0);
            scalar |= lhs.shape.is_empty();
            empty |= lhs.shape.iter().any(|extent| *extent == 0);
            scalar_rhs |= rhs.shape.is_empty();
            matching_rhs |= rhs.shape == lhs.shape;
            assert_eq!(lhs.dtype, DType::Bool);
            assert_eq!(rhs.dtype, DType::Bool);
        }
    }

    assert!(found);
    assert!(and && or && true_lane && false_lane);
    assert!(scalar && empty && scalar_rhs && matching_rhs);
}

#[test]
fn generated_logical_not_cases_are_valid_diverse_and_deterministic() {
    let mut found = false;
    let mut boolean = false;
    let mut i32 = false;
    let mut f32 = false;
    let mut zero = false;
    let mut nonzero = false;
    let mut scalar = false;
    let mut empty = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..512 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::LogicalNot { input } = case else {
                continue;
            };
            found = true;
            scalar |= input.shape.is_empty();
            empty |= input.shape.iter().any(|extent| *extent == 0);
            match input.to_tensor().unwrap().storage() {
                Storage::Bool(values) => {
                    boolean = true;
                    zero |= values.iter().any(|value| !*value);
                    nonzero |= values.iter().any(|value| *value);
                }
                Storage::I32(values) => {
                    i32 = true;
                    zero |= values.contains(&0);
                    nonzero |= values.iter().any(|value| *value != 0);
                }
                Storage::F32(values) => {
                    f32 = true;
                    zero |= values.iter().any(|value| *value == 0.0);
                    nonzero |= values.iter().any(|value| *value != 0.0);
                }
                _ => unreachable!("logical-not generator selects only Bool/I32/F32"),
            }
        }
    }

    assert!(found);
    assert!(boolean && i32 && f32 && zero && nonzero && scalar && empty);
}

#[test]
fn generated_tensor_t_cases_are_valid_diverse_and_deterministic() {
    let mut found = false;
    let mut f32 = false;
    let mut i32 = false;
    let mut f16 = false;
    let mut boolean = false;
    let mut square = false;
    let mut rectangular = false;
    let mut zero_rows = false;
    let mut zero_columns = false;
    let mut all_zero = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..512 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::TensorT { input } = case else {
                continue;
            };
            found = true;
            assert_eq!(input.shape.len(), 2);
            square |= input.shape[0] == input.shape[1];
            rectangular |= input.shape[0] != input.shape[1];
            zero_rows |= input.shape[0] == 0;
            zero_columns |= input.shape[1] == 0;
            all_zero |= input.shape == vec![0, 0];
            match input.dtype {
                DType::F32 => f32 = true,
                DType::I32 => i32 = true,
                DType::F16 => f16 = true,
                DType::Bool => boolean = true,
                _ => unreachable!("Tensor.T generator selects movement dtypes only"),
            }
        }
    }

    assert!(found);
    assert!(f32 && i32 && f16 && boolean);
    assert!(square && rectangular && zero_rows && zero_columns && all_zero);
}

#[test]
fn generated_pad_cases_are_valid_diverse_and_deterministic() {
    let mut found = false;
    let mut f32 = false;
    let mut i32 = false;
    let mut f16 = false;
    let mut boolean = false;
    let mut scalar = false;
    let mut empty = false;
    let mut before = false;
    let mut after = false;
    let mut asymmetric = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..512 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::Pad {
                input,
                padding,
                fill,
            } = case
            else {
                continue;
            };
            found = true;
            assert_eq!(padding.len(), input.shape.len());
            assert!(fill.shape.is_empty());
            assert_eq!(fill.dtype, input.dtype);
            scalar |= input.shape.is_empty();
            empty |= input.shape.iter().any(|extent| *extent == 0);
            before |= padding.iter().any(|(before, _)| *before != 0);
            after |= padding.iter().any(|(_, after)| *after != 0);
            asymmetric |= padding.iter().any(|(before, after)| before != after);
            match input.dtype {
                DType::F32 => f32 = true,
                DType::I32 => i32 = true,
                DType::F16 => f16 = true,
                DType::Bool => boolean = true,
                _ => unreachable!("Pad generator selects movement dtypes only"),
            }
        }
    }

    assert!(found);
    assert!(f32 && i32 && f16 && boolean);
    assert!(scalar && empty && before && after && asymmetric);
}

#[test]
fn pad_cases_round_trip_minimize_and_capture_as_movement_plans() {
    let pad = FuzzCase::Pad {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage([2, 2], Storage::F32(vec![1.0, 2.0, 3.0, 4.0])).unwrap(),
        ),
        padding: vec![(1, 0), (0, 2)],
        fill: FuzzTensor::from_tensor(&TensorData::from_storage([], Storage::F32(vec![-0.0])).unwrap()),
    };
    let value = serde_json::to_value(&pad).unwrap();
    assert_eq!(value["kind"], "pad");
    assert_eq!(serde_json::from_value::<FuzzCase>(value.clone()).unwrap(), pad);
    let mut unknown = value;
    unknown.as_object_mut().unwrap().insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    for case in [
        pad.clone(),
        FuzzCase::Pad {
            input: FuzzTensor::from_tensor(&TensorData::from_storage([0, 2], Storage::I32(vec![])).unwrap()),
            padding: vec![(1, 1), (1, 0)],
            fill: FuzzTensor::from_tensor(&TensorData::from_storage([], Storage::I32(vec![-7])).unwrap()),
        },
        FuzzCase::Pad {
            input: FuzzTensor::from_tensor(&TensorData::from_storage([], Storage::Bool(vec![true])).unwrap()),
            padding: vec![],
            fill: FuzzTensor::from_tensor(&TensorData::from_storage([], Storage::Bool(vec![false])).unwrap()),
        },
    ] {
        let built = case.build().unwrap();
        assert_eq!(built.ordered.len(), 1, "Pad fill is plan metadata, not an input binding");
        let Op::Pad { padding, .. } = built.graph.op(built.output).unwrap() else {
            panic!("raw Pad case must retain its Pad root");
        };
        let FuzzCase::Pad { padding: expected, .. } = &case else {
            unreachable!("constructed as Pad")
        };
        assert_eq!(padding, expected);
        let plan = MovementKernelPlan::from_graph(&built.graph, built.output).unwrap();
        let MovementKernelKind::Pad { padding: planned, fill_bits, .. } = &plan.kind else {
            panic!("Pad root must use a Pad movement plan");
        };
        assert_eq!(planned, expected);
        if matches!(&case, FuzzCase::Pad { input, .. } if input.dtype == DType::F32) {
            assert_eq!(*fill_bits, 0x8000_0000);
        }
        let scheduled = schedule(&built.graph, built.output).unwrap();
        assert_eq!(scheduled.items.len(), 1);
        let item = &scheduled.items[0];
        assert!(item.boundary.is_none());
        assert!(matches!(item.kernel.kind(), UOpKind::Movement));
        assert!(matches!(item.kernel.arg(), UArg::Movement(plan) if matches!(&plan.kind, MovementKernelKind::Pad { .. })));
        assert!(CpuJit::render(&item.kernel).is_ok());
        assert!(CpuJit::render_vectorized(&item.kernel).is_ok());
        let captured = CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        assert_eq!(CapturedSchedule::from_bytes(&bytes).unwrap().to_bytes().unwrap(), bytes);
    }

    let built = pad.build().unwrap();
    let output = CpuBackend.execute(&built.graph, built.output, &built.oracle).unwrap();
    let expected = TensorData::from_storage(
        [3, 4],
        Storage::F32(vec![-0.0, -0.0, -0.0, -0.0, 1.0, 2.0, -0.0, -0.0, 3.0, 4.0, -0.0, -0.0]),
    ).unwrap();
    assert_eq!(FuzzTensor::from_tensor(&output), FuzzTensor::from_tensor(&expected));
    let artifact = FuzzFailureArtifact::new(
        12, 16, pad.clone(), FuzzPath::NativeScalar, FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&output),
        FuzzOutcome::Error { class: "execute".into(), detail: "synthetic Pad mismatch".into() },
    ).unwrap();
    assert_eq!(FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(), artifact);
    let zeroed = minimize_case(&pad, |candidate| {
        matches!(candidate, FuzzCase::Pad { input, fill, .. }
            if input.bytes == vec![0; 16] && fill.bytes == vec![0; 4])
    });
    assert!(matches!(zeroed, FuzzCase::Pad { ref input, ref fill, .. }
        if input.bytes == vec![0; 16] && fill.bytes == vec![0; 4]));

    let nan_fill = FuzzCase::Pad {
        input: FuzzTensor::from_tensor(&TensorData::from_storage([1], Storage::F32(vec![1.0])).unwrap()),
        padding: vec![(1, 1)],
        fill: FuzzTensor::from_tensor(&TensorData::from_storage([], Storage::F32(vec![f32::from_bits(0x7fc0_0001)])).unwrap()),
    };
    let nan_built = nan_fill.build().unwrap();
    let MovementKernelKind::Pad { fill_bits, .. } = MovementKernelPlan::from_graph(&nan_built.graph, nan_built.output).unwrap().kind else {
        unreachable!("Pad plan")
    };
    assert!(f32::from_bits(fill_bits as u32).is_nan());

    let bad_shape = FuzzCase::Pad {
        input: FuzzTensor::from_tensor(&TensorData::from_storage([1], Storage::F32(vec![1.0])).unwrap()),
        padding: vec![(0, 1)],
        fill: FuzzTensor::from_tensor(&TensorData::from_storage([1], Storage::F32(vec![0.0])).unwrap()),
    };
    let bad_dtype = FuzzCase::Pad {
        input: FuzzTensor::from_tensor(&TensorData::from_storage([1], Storage::F32(vec![1.0])).unwrap()),
        padding: vec![(0, 1)],
        fill: FuzzTensor::from_tensor(&TensorData::from_storage([], Storage::I32(vec![0])).unwrap()),
    };
    let bad_padding = FuzzCase::Pad {
        input: FuzzTensor::from_tensor(&TensorData::from_storage([1], Storage::F32(vec![1.0])).unwrap()),
        padding: vec![],
        fill: FuzzTensor::from_tensor(&TensorData::from_storage([], Storage::F32(vec![0.0])).unwrap()),
    };
    assert!(bad_shape.validate().is_err());
    assert!(bad_dtype.validate().is_err());
    assert!(bad_padding.validate().is_err());
}

#[test]
fn generated_gather_cases_are_valid_diverse_and_deterministic() {
    let mut found = false;
    let mut f32 = false;
    let mut i32 = false;
    let mut f16 = false;
    let mut boolean = false;
    let mut index_i32 = false;
    let mut index_i64 = false;
    let mut axes = std::collections::BTreeSet::new();
    let mut empty = false;
    let mut duplicate = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index_number in 0..512 {
            let case = generate_case(seed, index_number);
            assert_eq!(case, generate_case(seed, index_number));
            case.validate().unwrap();
            let FuzzCase::Gather { input, index, axis } = case else {
                continue;
            };
            found = true;
            assert!((1..=3).contains(&input.shape.len()));
            assert_eq!(index.shape.len(), input.shape.len());
            assert!(axis < input.shape.len());
            assert!(matches!(index.dtype, DType::I32 | DType::I64));
            for (dimension, (&source, &selected)) in input.shape.iter().zip(&index.shape).enumerate() {
                if dimension != axis {
                    assert!(selected <= source);
                }
            }
            let lanes = index.to_tensor().unwrap();
            let mut values = std::collections::BTreeSet::new();
            for position in 0..lanes.len() {
                let Scalar::I(value) = lanes.scalar_at(position) else {
                    panic!("generated gather indices are signed integer lanes")
                };
                assert!(value >= 0 && (value as usize) < input.shape[axis]);
                duplicate |= !values.insert(value);
            }
            empty |= input.shape.iter().any(|extent| *extent == 0) || index.shape.iter().any(|extent| *extent == 0);
            axes.insert((input.shape.len(), axis));
            match input.dtype {
                DType::F32 => f32 = true,
                DType::I32 => i32 = true,
                DType::F16 => f16 = true,
                DType::Bool => boolean = true,
                _ => unreachable!("Gather generator selects movement dtypes only"),
            }
            index_i32 |= index.dtype == DType::I32;
            index_i64 |= index.dtype == DType::I64;
        }
    }

    assert!(found);
    assert!(f32 && i32 && f16 && boolean && index_i32 && index_i64);
    assert!(empty && duplicate);
    assert!(axes.contains(&(1, 0)) && axes.iter().any(|(rank, axis)| *rank >= 2 && *axis == 1));
}

#[test]
fn gather_cases_round_trip_minimize_and_capture_as_movement_plans() {
    let gather = FuzzCase::Gather {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage([2, 4], Storage::F32(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])).unwrap(),
        ),
        index: FuzzTensor::from_tensor(
            &TensorData::from_storage([2, 3], Storage::I32(vec![3, 1, 1, 0, 2, 2])).unwrap(),
        ),
        axis: 1,
    };
    let value = serde_json::to_value(&gather).unwrap();
    assert_eq!(value["kind"], "gather");
    assert_eq!(serde_json::from_value::<FuzzCase>(value.clone()).unwrap(), gather);
    let mut unknown = value;
    unknown.as_object_mut().unwrap().insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    for case in [
        gather.clone(),
        FuzzCase::Gather {
            input: FuzzTensor::from_tensor(&TensorData::from_storage([3], Storage::I32(vec![10, 20, 30])).unwrap()),
            index: FuzzTensor::from_tensor(&TensorData::from_storage([3], Storage::I64(vec![2, 0, 1])).unwrap()),
            axis: 0,
        },
        FuzzCase::Gather {
            input: FuzzTensor::from_tensor(&TensorData::from_storage([2, 0], Storage::F16(vec![])).unwrap()),
            index: FuzzTensor::from_tensor(&TensorData::from_storage([2, 0], Storage::I32(vec![])).unwrap()),
            axis: 1,
        },
    ] {
        let built = case.build().unwrap();
        let Op::Gather { axis, .. } = built.graph.op(built.output).unwrap() else {
            panic!("raw Gather case must retain its Gather root");
        };
        let FuzzCase::Gather { axis: expected, .. } = &case else {
            unreachable!("constructed as Gather")
        };
        assert_eq!(axis, expected);
        let plan = MovementKernelPlan::from_graph(&built.graph, built.output).unwrap();
        let MovementKernelKind::Gather { axis: planned, input, index } = &plan.kind else {
            panic!("Gather root must use a Gather movement plan");
        };
        assert_eq!(planned, expected);
        assert_eq!(plan.output_shape, index.shape);
        assert_eq!(plan.dtype, input.dtype);
        let scheduled = schedule(&built.graph, built.output).unwrap();
        assert_eq!(scheduled.items.len(), 1);
        let item = &scheduled.items[0];
        assert!(item.boundary.is_none());
        assert!(matches!(item.kernel.kind(), UOpKind::Movement));
        assert!(matches!(item.kernel.arg(), UArg::Movement(plan) if matches!(&plan.kind, MovementKernelKind::Gather { .. })));
        let scalar = CpuJit::render(&item.kernel).unwrap();
        assert!(scalar.source.contains("rg_selected < 0") && scalar.source.contains("failure[1]=3"));
        assert!(CpuJit::render_vectorized(&item.kernel).is_ok());
        let captured = CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        assert_eq!(CapturedSchedule::from_bytes(&bytes).unwrap().to_bytes().unwrap(), bytes);
    }

    let built = gather.build().unwrap();
    let output = CpuBackend.execute(&built.graph, built.output, &built.oracle).unwrap();
    assert_eq!(
        FuzzTensor::from_tensor(&output),
        FuzzTensor::from_tensor(&TensorData::from_storage([2, 3], Storage::F32(vec![3.0, 1.0, 1.0, 4.0, 6.0, 6.0])).unwrap()),
    );
    let artifact = FuzzFailureArtifact::new(
        13, 17, gather.clone(), FuzzPath::NativeScalar, FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&output),
        FuzzOutcome::Error { class: "execute".into(), detail: "synthetic Gather mismatch".into() },
    ).unwrap();
    assert_eq!(FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(), artifact);
    let zeroed = minimize_case(&gather, |candidate| {
        matches!(candidate, FuzzCase::Gather { input, index, axis }
            if input.bytes == vec![0; 32] && index.bytes == vec![0; 24] && *axis == 1)
    });
    assert!(matches!(zeroed, FuzzCase::Gather { ref input, ref index, axis }
        if input.bytes == vec![0; 32] && index.bytes == vec![0; 24] && axis == 1));

    let ieee = FuzzCase::Gather {
        input: FuzzTensor::from_tensor(&TensorData::from_storage([3], Storage::F32(vec![f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_0001), f32::INFINITY])).unwrap()),
        index: FuzzTensor::from_tensor(&TensorData::from_storage([3], Storage::I32(vec![1, 0, 2])).unwrap()),
        axis: 0,
    };
    let ieee_built = ieee.build().unwrap();
    let ieee_output = CpuBackend.execute(&ieee_built.graph, ieee_built.output, &ieee_built.oracle).unwrap();
    assert_eq!(
        FuzzTensor::from_tensor(&ieee_output),
        FuzzTensor::from_tensor(&TensorData::from_storage([3], Storage::F32(vec![f32::from_bits(0x7fc0_0001), -0.0, f32::INFINITY])).unwrap()),
    );

    let bad_dtype = FuzzCase::Gather {
        input: FuzzTensor::from_tensor(&TensorData::from_storage([2], Storage::F32(vec![1.0, 2.0])).unwrap()),
        index: FuzzTensor::from_tensor(&TensorData::from_storage([1], Storage::I16(vec![0])).unwrap()),
        axis: 0,
    };
    let bad_index = FuzzCase::Gather {
        input: FuzzTensor::from_tensor(&TensorData::from_storage([2], Storage::F32(vec![1.0, 2.0])).unwrap()),
        index: FuzzTensor::from_tensor(&TensorData::from_storage([1], Storage::I32(vec![2])).unwrap()),
        axis: 0,
    };
    assert!(bad_dtype.validate().is_err());
    assert!(bad_index.validate().is_err());
}

#[test]
fn generated_scatter_cases_are_valid_diverse_and_deterministic() {
    let mut found = false;
    let mut replace_f32 = false;
    let mut replace_i32 = false;
    let mut replace_f16 = false;
    let mut replace_bool = false;
    let mut add_f32 = false;
    let mut index_i32 = false;
    let mut index_i64 = false;
    let mut zero_axis = false;
    let mut empty = false;
    let mut duplicate = false;
    let mut axes = std::collections::BTreeSet::new();

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index_number in 0..512 {
            let case = generate_case(seed, index_number);
            assert_eq!(case, generate_case(seed, index_number));
            case.validate().unwrap();
            let FuzzCase::Scatter {
                base,
                index,
                updates,
                axis,
                op,
            } = case
            else {
                continue;
            };
            found = true;
            assert!((1..=3).contains(&base.shape.len()));
            assert_eq!(index.shape.len(), base.shape.len());
            assert_eq!(updates.shape.len(), index.shape.len());
            assert!(axis < base.shape.len());
            assert_eq!(base.dtype, updates.dtype);
            assert!(matches!(index.dtype, DType::I32 | DType::I64));
            for (dimension, ((&base, &index), &updates)) in base
                .shape
                .iter()
                .zip(&index.shape)
                .zip(&updates.shape)
                .enumerate()
            {
                if dimension != axis {
                    assert!(index <= base);
                }
                assert!(updates >= index);
            }
            let lanes = index.to_tensor().unwrap();
            let mut values = std::collections::BTreeSet::new();
            for position in 0..lanes.len() {
                let Scalar::I(value) = lanes.scalar_at(position) else {
                    panic!("generated scatter indices are signed integer lanes")
                };
                assert!(value >= 0 && (value as usize) < base.shape[axis]);
                duplicate |= !values.insert(value);
            }
            empty |= base.shape.iter().any(|extent| *extent == 0)
                || index.shape.iter().any(|extent| *extent == 0);
            zero_axis |= base.shape[axis] == 0;
            axes.insert((base.shape.len(), axis));
            index_i32 |= index.dtype == DType::I32;
            index_i64 |= index.dtype == DType::I64;
            match op {
                FuzzScatterOp::Replace => match base.dtype {
                    DType::F32 => replace_f32 = true,
                    DType::I32 => replace_i32 = true,
                    DType::F16 => replace_f16 = true,
                    DType::Bool => replace_bool = true,
                    _ => unreachable!("replacement generator selects movement dtypes only"),
                },
                FuzzScatterOp::Add => {
                    assert_eq!(base.dtype, DType::F32);
                    add_f32 = true;
                }
            }
        }
    }

    assert!(found);
    assert!(replace_f32 && replace_i32 && replace_f16 && replace_bool && add_f32);
    assert!(index_i32 && index_i64 && zero_axis && empty && duplicate);
    assert!(axes.contains(&(1, 0)) && axes.iter().any(|(rank, axis)| *rank >= 2 && *axis == 1));
}

#[test]
fn scatter_cases_round_trip_minimize_and_capture_as_movement_plans() {
    let replace = FuzzCase::Scatter {
        base: FuzzTensor::from_tensor(
            &TensorData::from_storage([1, 4], Storage::F32(vec![10.0, 20.0, 30.0, 40.0]))
                .unwrap(),
        ),
        index: FuzzTensor::from_tensor(
            &TensorData::from_storage([1, 3], Storage::I32(vec![2, 1, 2])).unwrap(),
        ),
        updates: FuzzTensor::from_tensor(
            &TensorData::from_storage([1, 3], Storage::F32(vec![1.0, 2.0, 3.0])).unwrap(),
        ),
        axis: 1,
        op: FuzzScatterOp::Replace,
    };
    let value = serde_json::to_value(&replace).unwrap();
    assert_eq!(value["kind"], "scatter");
    assert_eq!(serde_json::from_value::<FuzzCase>(value.clone()).unwrap(), replace);
    let mut unknown = value;
    unknown.as_object_mut().unwrap().insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    for case in [
        replace.clone(),
        FuzzCase::Scatter {
            base: FuzzTensor::from_tensor(
                &TensorData::from_storage([1, 3], Storage::F32(vec![1.0, 10.0, 100.0])).unwrap(),
            ),
            index: FuzzTensor::from_tensor(
                &TensorData::from_storage([1, 3], Storage::I64(vec![1, 1, 1])).unwrap(),
            ),
            updates: FuzzTensor::from_tensor(
                &TensorData::from_storage([1, 3], Storage::F32(vec![0.25, 0.5, 4.0])).unwrap(),
            ),
            axis: 1,
            op: FuzzScatterOp::Add,
        },
        FuzzCase::Scatter {
            base: FuzzTensor::from_tensor(&TensorData::from_storage([2, 0], Storage::F16(vec![])).unwrap()),
            index: FuzzTensor::from_tensor(&TensorData::from_storage([2, 0], Storage::I32(vec![])).unwrap()),
            updates: FuzzTensor::from_tensor(&TensorData::from_storage([2, 0], Storage::F16(vec![])).unwrap()),
            axis: 1,
            op: FuzzScatterOp::Replace,
        },
    ] {
        let built = case.build().unwrap();
        let Op::Scatter { axis, add, .. } = built.graph.op(built.output).unwrap() else {
            panic!("raw Scatter case must retain its Scatter root");
        };
        let FuzzCase::Scatter { axis: expected, op, .. } = &case else {
            unreachable!("constructed as Scatter")
        };
        assert_eq!(axis, expected);
        assert_eq!(*add, *op == FuzzScatterOp::Add);
        let plan = MovementKernelPlan::from_graph(&built.graph, built.output).unwrap();
        let MovementKernelKind::Scatter { axis: planned, add: planned_add, .. } = &plan.kind else {
            panic!("Scatter root must use a Scatter movement plan");
        };
        assert_eq!(planned, expected);
        assert_eq!(*planned_add, *op == FuzzScatterOp::Add);
        let scheduled = schedule(&built.graph, built.output).unwrap();
        assert_eq!(scheduled.items.len(), 1);
        let item = &scheduled.items[0];
        assert!(item.boundary.is_none());
        assert!(matches!(item.kernel.kind(), UOpKind::Movement));
        assert!(matches!(item.kernel.arg(), UArg::Movement(plan) if matches!(&plan.kind, MovementKernelKind::Scatter { .. })));
        let scalar = CpuJit::render(&item.kernel).unwrap();
        assert!(scalar.source.contains("memcpy(") && scalar.source.contains("rg_selected < 0") && scalar.source.contains("failure[1]=3"));
        if *op == FuzzScatterOp::Add {
            assert!(scalar.source.contains("] += ((const"));
        } else {
            assert!(scalar.source.contains("] = ((const"));
        }
        assert!(CpuJit::render_vectorized(&item.kernel).is_ok());
        let captured = CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        assert_eq!(CapturedSchedule::from_bytes(&bytes).unwrap().to_bytes().unwrap(), bytes);
    }

    let built = replace.build().unwrap();
    let output = CpuBackend.execute(&built.graph, built.output, &built.oracle).unwrap();
    assert_eq!(
        FuzzTensor::from_tensor(&output),
        FuzzTensor::from_tensor(
            &TensorData::from_storage([1, 4], Storage::F32(vec![10.0, 2.0, 3.0, 40.0])).unwrap(),
        ),
    );
    let add = FuzzCase::Scatter {
        base: FuzzTensor::from_tensor(
            &TensorData::from_storage([1, 3], Storage::F32(vec![1.0, 10.0, 100.0])).unwrap(),
        ),
        index: FuzzTensor::from_tensor(
            &TensorData::from_storage([1, 3], Storage::I32(vec![1, 1, 1])).unwrap(),
        ),
        updates: FuzzTensor::from_tensor(
            &TensorData::from_storage([1, 3], Storage::F32(vec![0.25, 0.5, 4.0])).unwrap(),
        ),
        axis: 1,
        op: FuzzScatterOp::Add,
    };
    let add_built = add.build().unwrap();
    let add_output = CpuBackend.execute(&add_built.graph, add_built.output, &add_built.oracle).unwrap();
    assert_eq!(
        FuzzTensor::from_tensor(&add_output),
        FuzzTensor::from_tensor(
            &TensorData::from_storage([1, 3], Storage::F32(vec![1.0, 14.75, 100.0])).unwrap(),
        ),
    );
    let artifact = FuzzFailureArtifact::new(
        14, 19, replace.clone(), FuzzPath::NativeScalar, FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&output),
        FuzzOutcome::Error { class: "execute".into(), detail: "synthetic Scatter mismatch".into() },
    ).unwrap();
    assert_eq!(FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(), artifact);
    let zeroed = minimize_case(&replace, |candidate| {
        matches!(candidate, FuzzCase::Scatter { base, index, updates, axis, op }
            if base.bytes == vec![0; 16] && index.bytes == vec![0; 12]
                && updates.bytes == vec![0; 12] && *axis == 1 && *op == FuzzScatterOp::Replace)
    });
    assert!(matches!(zeroed, FuzzCase::Scatter { ref base, ref index, ref updates, axis, op }
        if base.bytes == vec![0; 16] && index.bytes == vec![0; 12]
            && updates.bytes == vec![0; 12] && axis == 1 && op == FuzzScatterOp::Replace));

    let ieee = FuzzCase::Scatter {
        base: FuzzTensor::from_tensor(&TensorData::from_storage([3], Storage::F32(vec![0.0, 1.0, 2.0])).unwrap()),
        index: FuzzTensor::from_tensor(&TensorData::from_storage([3], Storage::I32(vec![2, 0, 1])).unwrap()),
        updates: FuzzTensor::from_tensor(&TensorData::from_storage([3], Storage::F32(vec![f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_0001), f32::INFINITY])).unwrap()),
        axis: 0,
        op: FuzzScatterOp::Replace,
    };
    let ieee_built = ieee.build().unwrap();
    let ieee_output = CpuBackend.execute(&ieee_built.graph, ieee_built.output, &ieee_built.oracle).unwrap();
    assert_eq!(
        FuzzTensor::from_tensor(&ieee_output),
        FuzzTensor::from_tensor(&TensorData::from_storage([3], Storage::F32(vec![f32::from_bits(0x7fc0_0001), f32::INFINITY, -0.0])).unwrap()),
    );

    let bad_add = FuzzCase::Scatter {
        base: FuzzTensor::from_tensor(&TensorData::from_storage([2], Storage::I32(vec![1, 2])).unwrap()),
        index: FuzzTensor::from_tensor(&TensorData::from_storage([1], Storage::I32(vec![0])).unwrap()),
        updates: FuzzTensor::from_tensor(&TensorData::from_storage([1], Storage::I32(vec![3])).unwrap()),
        axis: 0,
        op: FuzzScatterOp::Add,
    };
    let bad_index = FuzzCase::Scatter {
        base: FuzzTensor::from_tensor(&TensorData::from_storage([2], Storage::F32(vec![1.0, 2.0])).unwrap()),
        index: FuzzTensor::from_tensor(&TensorData::from_storage([1], Storage::I64(vec![-1])).unwrap()),
        updates: FuzzTensor::from_tensor(&TensorData::from_storage([1], Storage::F32(vec![3.0])).unwrap()),
        axis: 0,
        op: FuzzScatterOp::Replace,
    };
    assert!(bad_add.validate().is_err());
    assert!(bad_index.validate().is_err());
}

#[test]
fn tensor_t_cases_round_trip_minimize_and_capture_as_affine_permute() {
    let tensor_t = FuzzCase::TensorT {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage([2, 3], Storage::F32(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
                .unwrap(),
        ),
    };
    let value = serde_json::to_value(&tensor_t).unwrap();
    assert_eq!(value["kind"], "tensor_t");
    assert_eq!(serde_json::from_value::<FuzzCase>(value.clone()).unwrap(), tensor_t);
    let mut unknown = value;
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    for input in [
        tensor_t_input(&tensor_t),
        FuzzTensor::from_tensor(
            &TensorData::from_storage([2, 2], Storage::I32(vec![0, 1, 2, 3])).unwrap(),
        ),
        FuzzTensor::from_tensor(&TensorData::from_storage([0, 3], Storage::F16(vec![])).unwrap()),
    ] {
        let case = FuzzCase::TensorT { input };
        let built = case.build().unwrap();
        let Op::Permute {
            input: source,
            axes,
        } = built.graph.op(built.output).unwrap()
        else {
            panic!("Tensor.T must retain its literal Permute root");
        };
        assert_eq!(axes, &vec![1, 0]);
        assert_eq!(
            built.graph.shape(built.output).unwrap().dims(),
            &[built.graph.shape(*source).unwrap().dims()[1], built.graph.shape(*source).unwrap().dims()[0]]
        );
        let scheduled = schedule(&built.graph, built.output).unwrap();
        for item in &scheduled.items {
            assert!(item.boundary.is_none());
            assert!(item.kernel.topological().unwrap().iter().any(|node| {
                matches!(node.arg(), UArg::ViewBufferIndex { .. })
            }));
            assert!(CpuJit::render(&item.kernel).is_ok());
            assert!(CpuJit::render_vectorized(&item.kernel).is_ok());
        }
        let captured = CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
        assert_eq!(decoded.items.len(), scheduled.items.len());
    }

    let built = tensor_t.build().unwrap();
    let output = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
    assert_eq!(output.storage(), &Storage::F32(vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]));
    let expected = FuzzOutcome::value(&output);
    let artifact = FuzzFailureArtifact::new(
        11,
        15,
        tensor_t.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        expected,
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic Tensor.T mismatch".into(),
        },
    )
    .unwrap();
    assert_eq!(FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(), artifact);
    let zeroed = minimize_case(&tensor_t, |candidate| {
        matches!(candidate, FuzzCase::TensorT { input }
            if input.shape == vec![2, 3] && input.bytes == vec![0; 24])
    });
    assert!(matches!(zeroed, FuzzCase::TensorT { ref input }
        if input.shape == vec![2, 3] && input.bytes == vec![0; 24]));
    assert_eq!(minimize_case(&tensor_t, |_| true), zeroed);

    let malformed = FuzzCase::TensorT {
        input: FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F32)),
    };
    assert!(malformed.validate().is_err());
}

fn tensor_t_input(case: &FuzzCase) -> FuzzTensor {
    match case {
        FuzzCase::TensorT { input } => input.clone(),
        _ => unreachable!("constructed as Tensor.T"),
    }
}

#[test]
fn logical_not_cases_round_trip_minimize_and_capture_source_composition() {
    let logical_not = FuzzCase::LogicalNot {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [7],
                Storage::F32(vec![
                    0.0,
                    f32::from_bits(0x8000_0000),
                    f32::from_bits(0x7fc0_0001),
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    0.5,
                    -0.5,
                ]),
            )
            .unwrap(),
        ),
    };
    let value = serde_json::to_value(&logical_not).unwrap();
    assert_eq!(value["kind"], "logical_not");
    assert_eq!(
        serde_json::from_value::<FuzzCase>(value.clone()).unwrap(),
        logical_not
    );
    let mut unknown = value;
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    for input in [
        logical_not_input(&logical_not),
        FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::I(-1), DType::I32)),
        FuzzTensor::from_tensor(
            &TensorData::from_storage([2], Storage::Bool(vec![false, true])).unwrap(),
        ),
    ] {
        let case = FuzzCase::LogicalNot { input };
        let built = case.build().unwrap();
        let Op::Compare {
            op: CompareOp::Ne,
            lhs,
            rhs,
        } = built.graph.op(built.output).unwrap()
        else {
            panic!("logical_not must be a source Ne root");
        };
        assert!(matches!(
            built.graph.op(*lhs).unwrap(),
            Op::Cast { dtype: DType::Bool, .. }
        ));
        assert!(matches!(
            built.graph.op(*rhs).unwrap(),
            Op::Constant(value)
                if value.shape().dims().is_empty()
                    && value.dtype() == DType::Bool
                    && value.storage() == &Storage::Bool(vec![true])
        ));
        let scheduled = schedule(&built.graph, built.output).unwrap();
        let assert_kernel = |kernel: &crate::UOp| {
            let nodes = kernel.topological().unwrap();
            assert!(nodes.iter().any(|node| {
                matches!(node.kind(), UOpKind::Cast)
                    && node.ty().is_some_and(|ty| ty.scalar == DType::Bool)
            }));
            assert!(nodes.iter().any(|node| {
                matches!(node.kind(), UOpKind::GraphCompare(CompareOp::Ne))
            }));
            assert!(!nodes.iter().any(|node| {
                matches!(node.kind(), UOpKind::GraphLogical(crate::LogicalOp::Not))
            }));
        };
        for item in &scheduled.items {
            assert_kernel(&item.kernel);
        }
        let captured = CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        for item in &captured.items {
            assert_kernel(&item.kernel);
        }
    }

    let built = logical_not.build().unwrap();
    let output = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
    assert_eq!(
        output.storage(),
        &Storage::Bool(vec![true, true, false, false, false, false, false])
    );
    let expected = FuzzOutcome::value(&output);
    let artifact = FuzzFailureArtifact::new(
        10,
        14,
        logical_not.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        expected,
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic logical-not mismatch".into(),
        },
    )
    .unwrap();
    assert_eq!(FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(), artifact);
    let zeroed = minimize_case(&logical_not, |candidate| {
        matches!(candidate, FuzzCase::LogicalNot { input } if input.bytes == vec![0; 28])
    });
    assert!(matches!(zeroed, FuzzCase::LogicalNot { ref input } if input.bytes == vec![0; 28]));
    let scalarized = minimize_case(&logical_not, |_| true);
    assert!(matches!(scalarized, FuzzCase::LogicalNot { ref input } if input.shape.is_empty()));
    scalarized.validate().unwrap();
}

fn logical_not_input(case: &FuzzCase) -> FuzzTensor {
    match case {
        FuzzCase::LogicalNot { input } => input.clone(),
        _ => unreachable!("constructed as logical-not"),
    }
}

#[test]
fn logical_cases_round_trip_minimize_and_capture_as_graph_logical() {
    let logical = FuzzCase::Logical {
        op: FuzzLogicalOp::Or,
        lhs: FuzzTensor::from_tensor(
            &TensorData::from_storage([2], Storage::Bool(vec![false, true])).unwrap(),
        ),
        rhs: FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::Bool(true), DType::Bool)),
    };
    let value = serde_json::to_value(&logical).unwrap();
    assert_eq!(value["kind"], "logical");
    assert_eq!(serde_json::from_value::<FuzzCase>(value.clone()).unwrap(), logical);
    let mut unknown = value;
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    for (op, graph_op, lhs, rhs) in [
        (
            FuzzLogicalOp::And,
            LogicalOp::And,
            FuzzTensor::from_tensor(
                &TensorData::from_storage([2], Storage::Bool(vec![true, false])).unwrap(),
            ),
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::Bool(true), DType::Bool)),
        ),
        (
            FuzzLogicalOp::Or,
            LogicalOp::Or,
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::Bool(false), DType::Bool)),
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::Bool(true), DType::Bool)),
        ),
    ] {
        let case = FuzzCase::Logical { op, lhs, rhs };
        let built = case.build().unwrap();
        assert!(matches!(
            built.graph.op(built.output).unwrap(),
            Op::Logical { op: actual, rhs: Some(_), .. } if *actual == graph_op
        ));
        let scheduled = schedule(&built.graph, built.output).unwrap();
        assert!(scheduled.items.iter().any(|item| {
            item.kernel
                .topological()
                .unwrap()
                .iter()
                .any(|uop| matches!(uop.kind(), UOpKind::GraphLogical(actual) if *actual == graph_op))
        }));
        let captured = CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        assert!(captured.items.iter().any(|item| {
            item.kernel
                .topological()
                .unwrap()
                .iter()
                .any(|uop| matches!(uop.kind(), UOpKind::GraphLogical(actual) if *actual == graph_op))
        }));
    }

    let built = logical.build().unwrap();
    let expected = FuzzOutcome::value(
        &CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap(),
    );
    let artifact = FuzzFailureArtifact::new(
        9,
        13,
        logical.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        expected,
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic logical mismatch".into(),
        },
    )
    .unwrap();
    assert_eq!(FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(), artifact);
    let zeroed = minimize_case(&logical, |candidate| {
        matches!(candidate, FuzzCase::Logical { lhs, rhs, .. }
            if lhs.bytes == vec![0; 2] && rhs.bytes == vec![0])
    });
    assert!(matches!(zeroed, FuzzCase::Logical { ref lhs, ref rhs, .. }
        if lhs.bytes == vec![0; 2] && rhs.bytes == vec![0]));
    let scalarized = minimize_case(&logical, |_| true);
    assert!(matches!(scalarized, FuzzCase::Logical { ref lhs, ref rhs, .. }
        if lhs.shape.is_empty() && rhs.shape.is_empty()));
    scalarized.validate().unwrap();
}

#[test]
fn compare_cases_round_trip_minimize_and_capture_as_graph_compare() {
    let compare = FuzzCase::Compare {
        op: FuzzCompareOp::Ge,
        lhs: FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [2],
                Storage::F32(vec![f32::from_bits(0x7fc0_0001), f32::from_bits(0x8000_0000)]),
            )
            .unwrap(),
        ),
        rhs: FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::F(0.0), DType::F32)),
    };
    let value = serde_json::to_value(&compare).unwrap();
    assert_eq!(value["kind"], "compare");
    assert_eq!(serde_json::from_value::<FuzzCase>(value.clone()).unwrap(), compare);
    let mut unknown = value;
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    let legacy = regression_cases().remove(0);
    let legacy_value = serde_json::to_value(&legacy).unwrap();
    assert_eq!(legacy_value["kind"], "binary");
    assert_eq!(serde_json::from_value::<FuzzCase>(legacy_value).unwrap(), legacy);

    for (op, graph_op, lhs, rhs) in [
        (
            FuzzCompareOp::Eq,
            CompareOp::Eq,
            FuzzTensor::from_tensor(
                &TensorData::from_storage([2], Storage::F32(vec![f32::NAN, -0.0])).unwrap(),
            ),
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::F(0.0), DType::F32)),
        ),
        (
            FuzzCompareOp::Ne,
            CompareOp::Ne,
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::I(1), DType::I32)),
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::I(2), DType::I32)),
        ),
        (
            FuzzCompareOp::Lt,
            CompareOp::Lt,
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::I(1), DType::I32)),
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::I(2), DType::I32)),
        ),
        (
            FuzzCompareOp::Le,
            CompareOp::Le,
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::I(1), DType::I32)),
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::I(2), DType::I32)),
        ),
        (
            FuzzCompareOp::Gt,
            CompareOp::Gt,
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::I(2), DType::I32)),
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::I(1), DType::I32)),
        ),
        (
            FuzzCompareOp::Ge,
            CompareOp::Ge,
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::I(2), DType::I32)),
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::I(1), DType::I32)),
        ),
    ] {
        let case = FuzzCase::Compare { op, lhs, rhs };
        let built = case.build().unwrap();
        assert!(matches!(
            built.graph.op(built.output).unwrap(),
            Op::Compare { .. }
        ));
        let scheduled = schedule(&built.graph, built.output).unwrap();
        assert!(scheduled.items.iter().any(|item| {
            item.kernel
                .topological()
                .unwrap()
                .iter()
                .any(|uop| matches!(uop.kind(), UOpKind::GraphCompare(actual) if *actual == graph_op))
        }));
        let captured = CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        assert_eq!(captured.items.len(), scheduled.items.len());
        assert!(captured.items.iter().any(|item| {
            item.kernel
                .topological()
                .unwrap()
                .iter()
                .any(|uop| matches!(uop.kind(), UOpKind::GraphCompare(actual) if *actual == graph_op))
        }));
    }

    let built = compare.build().unwrap();
    assert!(matches!(
        built.graph.op(built.output).unwrap(),
        Op::Compare {
            op: CompareOp::Ge,
            ..
        }
    ));
    let expected = FuzzOutcome::value(
        &CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap(),
    );
    let artifact = FuzzFailureArtifact::new(
        8,
        12,
        compare.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        expected,
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic compare mismatch".into(),
        },
    )
    .unwrap();
    assert_eq!(FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(), artifact);
    let zeroed = minimize_case(&compare, |candidate| {
        matches!(candidate, FuzzCase::Compare { lhs, rhs, .. }
            if lhs.bytes == vec![0; 8] && rhs.bytes == vec![0; 4])
    });
    assert!(matches!(zeroed, FuzzCase::Compare { ref lhs, ref rhs, .. }
        if lhs.bytes == vec![0; 8] && rhs.bytes == vec![0; 4]));
    let scalarized = minimize_case(&compare, |_| true);
    assert!(matches!(scalarized, FuzzCase::Compare { ref lhs, ref rhs, .. }
        if lhs.shape.is_empty() && rhs.shape.is_empty()));
    scalarized.validate().unwrap();
}

#[test]
fn unary_cases_round_trip_minimize_and_build_as_direct_graph_unaries() {
    let unary = FuzzCase::Unary {
        op: FuzzUnaryOp::Abs,
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [2],
                Storage::F32(vec![f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_0001)]),
            )
            .unwrap(),
        ),
    };
    let value = serde_json::to_value(&unary).unwrap();
    assert_eq!(value["kind"], "unary");
    assert_eq!(serde_json::from_value::<FuzzCase>(value.clone()).unwrap(), unary);
    let mut unknown = value;
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    let legacy = regression_cases().remove(0);
    let legacy_value = serde_json::to_value(&legacy).unwrap();
    assert_eq!(legacy_value["kind"], "binary");
    assert_eq!(serde_json::from_value::<FuzzCase>(legacy_value).unwrap(), legacy);

    let built = unary.build().unwrap();
    assert!(matches!(
        built.graph.op(built.output).unwrap(),
        Op::Unary {
            op: UnaryOp::Abs,
            ..
        }
    ));
    let negated = FuzzCase::Unary {
        op: FuzzUnaryOp::Neg,
        input: match &unary {
            FuzzCase::Unary { input, .. } => input.clone(),
            _ => unreachable!("constructed as unary"),
        },
    };
    let negated = negated.build().unwrap();
    assert!(matches!(
        negated.graph.op(negated.output).unwrap(),
        Op::Unary {
            op: UnaryOp::Neg,
            ..
        }
    ));
    let expected = FuzzOutcome::value(
        &CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap(),
    );
    let artifact = FuzzFailureArtifact::new(
        7,
        11,
        unary.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        expected,
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic unary mismatch".into(),
        },
    )
    .unwrap();
    assert_eq!(FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(), artifact);

    let zeroed = minimize_case(&unary, |candidate| {
        matches!(candidate, FuzzCase::Unary { input, .. }
            if input.shape.as_slice() == [2] && input.bytes == vec![0; 8])
    });
    assert!(matches!(
        zeroed,
        FuzzCase::Unary {
            op: FuzzUnaryOp::Abs,
            ref input,
        } if input.bytes == vec![0; 8]
    ));
    let scalarized = minimize_case(&unary, |_| true);
    assert!(matches!(
        scalarized,
        FuzzCase::Unary {
            op: FuzzUnaryOp::Abs,
            ref input,
        } if input.shape.is_empty() && input.bytes == vec![0; 4]
    ));
    scalarized.validate().unwrap();
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
    assert_eq!(cases.len(), 31);
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
