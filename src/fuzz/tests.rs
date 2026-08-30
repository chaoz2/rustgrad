use super::*;
use crate::uop::Ternary;
use crate::{
    Backend, CapturedSchedule, CompareOp, CpuBackend, CpuJit, DType, LogicalOp, MovementKernelKind,
    MovementKernelPlan, Op, ReduceKind, Scalar, Storage, TensorData, UArg, UOpKind, UnaryOp,
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
fn generated_reduction_cases_cover_portable_kinds_and_geometry() {
    let mut kinds = std::collections::BTreeSet::new();
    let mut ranks = std::collections::BTreeSet::new();
    let mut dtypes = std::collections::BTreeSet::new();
    let mut nonzero_axis = false;
    let mut zero_domain = false;
    let mut scalar_output = false;
    let mut keepdim = false;
    let mut dropdim = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..512 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::Reduction {
                input,
                reduction,
                axis,
                keepdim: case_keepdim,
            } = case
            else {
                continue;
            };
            assert!((1..=3).contains(&input.shape.len()));
            assert!(axis < input.shape.len());
            if matches!(reduction, FuzzReduction::Max | FuzzReduction::Min) {
                assert_ne!(input.shape[axis], 0);
            }
            let built = FuzzCase::Reduction {
                input: input.clone(),
                reduction,
                axis,
                keepdim: case_keepdim,
            }
            .build()
            .unwrap();
            kinds.insert(reduction);
            ranks.insert(input.shape.len());
            dtypes.insert(input.dtype);
            nonzero_axis |= input.shape[axis] != 0;
            zero_domain |= input.shape[axis] == 0;
            scalar_output |= built.graph.shape(built.output).unwrap().rank() == 0;
            keepdim |= case_keepdim;
            dropdim |= !case_keepdim;
        }
    }

    assert_eq!(
        kinds,
        std::collections::BTreeSet::from([
            FuzzReduction::Sum,
            FuzzReduction::Mean,
            FuzzReduction::Product,
            FuzzReduction::Max,
            FuzzReduction::Min,
        ])
    );
    assert_eq!(ranks, std::collections::BTreeSet::from([1, 2, 3]));
    assert_eq!(dtypes.len(), 13);
    assert!(nonzero_axis && zero_domain && scalar_output && keepdim && dropdim);
}

#[test]
fn generated_matmul_cases_cover_portable_generalized_geometry() {
    let mut f32 = false;
    let mut f64 = false;
    let mut vector_vector = false;
    let mut matrix_vector = false;
    let mut vector_matrix = false;
    let mut matrix_matrix = false;
    let mut batched = false;
    let mut zero_k = false;
    let mut zero_geometry = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..4096 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::Matmul { lhs, rhs } = case else {
                continue;
            };
            assert_eq!(lhs.dtype, rhs.dtype);
            assert!(matches!(lhs.dtype, DType::F32 | DType::F64));
            f32 |= lhs.dtype == DType::F32;
            f64 |= lhs.dtype == DType::F64;
            vector_vector |= lhs.shape.len() == 1 && rhs.shape.len() == 1;
            matrix_vector |= lhs.shape.len() == 2 && rhs.shape.len() == 1;
            vector_matrix |= lhs.shape.len() == 1 && rhs.shape.len() == 2;
            matrix_matrix |= lhs.shape.len() == 2 && rhs.shape.len() == 2;
            batched |= lhs.shape.len() > 2 || rhs.shape.len() > 2;
            zero_k |= lhs.shape.last() == Some(&0);
            zero_geometry |= lhs.shape.contains(&0) || rhs.shape.contains(&0);
        }
    }

    assert!(f32 && f64);
    assert!(vector_vector && matrix_vector && vector_matrix && matrix_matrix && batched);
    assert!(zero_k && zero_geometry);
}

#[test]
fn matmul_cases_round_trip_capture_and_keep_f32_storage_rounding() {
    let f32 = FuzzCase::Matmul {
        lhs: FuzzTensor::from_tensor(
            &TensorData::from_storage([1, 3], Storage::F32(vec![1.0e10, 1.0, -1.0e10])).unwrap(),
        ),
        rhs: FuzzTensor::from_tensor(
            &TensorData::from_storage([3, 1], Storage::F32(vec![1.0; 3])).unwrap(),
        ),
    };
    let f64_vector = FuzzCase::Matmul {
        lhs: FuzzTensor::from_tensor(
            &TensorData::from_storage([3], Storage::F64(vec![1.0, -2.0, 0.5])).unwrap(),
        ),
        rhs: FuzzTensor::from_tensor(
            &TensorData::from_storage([3], Storage::F64(vec![4.0, 8.0, 16.0])).unwrap(),
        ),
    };
    let batched = FuzzCase::Matmul {
        lhs: FuzzTensor::from_tensor(
            &TensorData::from_storage([2, 1, 1, 2], Storage::F64(vec![1.0, 2.0, 3.0, 4.0]))
                .unwrap(),
        ),
        rhs: FuzzTensor::from_tensor(
            &TensorData::from_storage([3, 2, 1], Storage::F64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
                .unwrap(),
        ),
    };

    let value = serde_json::to_value(&f32).unwrap();
    assert_eq!(value["kind"], "matmul");
    assert_eq!(
        serde_json::from_value::<FuzzCase>(value.clone()).unwrap(),
        f32
    );
    let mut unknown = value;
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    for case in [f32.clone(), f64_vector, batched] {
        let built = case.build().unwrap();
        assert!(matches!(
            built.graph.op(built.output).unwrap(),
            Op::Matmul { .. }
        ));
        let plan = crate::MatmulKernelPlan::from_graph(&built.graph, built.output).unwrap();
        assert_eq!(
            plan.output_shape,
            built.graph.shape(built.output).unwrap().clone()
        );
        assert!(plan.lhs_vector == (plan.lhs_shape.rank() == 1));
        assert!(plan.rhs_vector == (plan.rhs_shape.rank() == 1));
        let scheduled = schedule(&built.graph, built.output).unwrap();
        assert_eq!(scheduled.items.len(), 1);
        let item = &scheduled.items[0];
        assert!(matches!(item.kernel.kind(), UOpKind::Matmul));
        let rendered_plan = item
            .kernel
            .arg()
            .matmul_plan()
            .expect("all static Matmul payloads retain their normalized base plan");
        assert_eq!(rendered_plan.m, plan.m);
        assert_eq!(rendered_plan.n, plan.n);
        assert_eq!(rendered_plan.k, plan.k);
        assert_eq!(rendered_plan.batch_shape, plan.batch_shape);
        let rendered = CpuJit::render(&item.kernel).unwrap();
        if plan.dtype == DType::F32 {
            assert!(rendered.source.contains("float rg_acc=0.0f;"));
            assert!(rendered.source.contains("float rg_product=(float)"));
        } else {
            assert!(rendered.source.contains("double rg_acc=0.0;"));
        }
        // Matmul's vector request intentionally has a scalar contraction
        // plan, but it remains a supported strict-native render path.
        assert_eq!(
            rendered.source,
            CpuJit::render_vectorized(&item.kernel).unwrap().source
        );
        let captured =
            CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        assert_eq!(
            CapturedSchedule::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
    }

    let built = f32.build().unwrap();
    let output = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
    let Scalar::F(value) = output.scalar_at(0) else {
        panic!("F32 matmul output")
    };
    assert_eq!((value as f32).to_bits(), 0.0f32.to_bits());
    let artifact = FuzzFailureArtifact::new(
        16,
        24,
        f32.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&output),
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic F32 matmul rounding mismatch".into(),
        },
    )
    .unwrap();
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(),
        artifact
    );
    let zeroed = minimize_case(&f32, |candidate| {
        matches!(candidate, FuzzCase::Matmul { lhs, rhs }
            if lhs.bytes == vec![0; 12] && rhs.bytes == vec![0; 12])
    });
    assert!(matches!(zeroed, FuzzCase::Matmul { ref lhs, ref rhs }
        if lhs.bytes == vec![0; 12] && rhs.bytes == vec![0; 12]));
}

#[test]
fn reduction_cases_round_trip_capture_render_and_preserve_extrema_payloads() {
    let cases = [
        (FuzzReduction::Sum, ReduceKind::Sum, true),
        (FuzzReduction::Mean, ReduceKind::Mean, false),
        (FuzzReduction::Product, ReduceKind::Product, true),
        (FuzzReduction::Max, ReduceKind::Max, false),
        (FuzzReduction::Min, ReduceKind::Min, true),
    ]
    .map(|(reduction, _kind, keepdim)| FuzzCase::Reduction {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage([2, 3], Storage::F32(vec![1.0, -0.0, 2.0, 3.0, 4.0, 5.0]))
                .unwrap(),
        ),
        reduction,
        axis: 1,
        keepdim,
    });
    let value = serde_json::to_value(&cases[2]).unwrap();
    assert_eq!(value["kind"], "reduction");
    assert_eq!(
        serde_json::from_value::<FuzzCase>(value.clone()).unwrap(),
        cases[2]
    );
    let mut unknown = value;
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    for (case, expected_kind) in cases.iter().cloned().zip([
        ReduceKind::Sum,
        ReduceKind::Mean,
        ReduceKind::Product,
        ReduceKind::Max,
        ReduceKind::Min,
    ]) {
        let built = case.build().unwrap();
        let Op::Reduce {
            kind,
            axes,
            keepdim,
            ..
        } = built.graph.op(built.output).unwrap()
        else {
            panic!("raw reduction case must retain its Reduce root");
        };
        assert_eq!(*kind, expected_kind);
        assert_eq!(axes, &vec![1]);
        let FuzzCase::Reduction {
            keepdim: expected_keepdim,
            ..
        } = &case
        else {
            unreachable!("constructed as Reduction")
        };
        assert_eq!(*keepdim, *expected_keepdim);
        let scheduled = schedule(&built.graph, built.output).unwrap();
        assert_eq!(scheduled.items.len(), 1);
        let item = &scheduled.items[0];
        assert!(item.boundary.is_none());
        assert!(
            item.kernel
                .topological()
                .unwrap()
                .iter()
                .any(|uop| { matches!(uop.kind(), UOpKind::ReduceFinalize) })
        );
        assert!(CpuJit::render(&item.kernel).is_ok());
        assert!(CpuJit::render_vectorized(&item.kernel).is_ok());
        let captured =
            CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        assert_eq!(
            CapturedSchedule::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
    }

    for reduction in [
        FuzzReduction::Sum,
        FuzzReduction::Mean,
        FuzzReduction::Product,
    ] {
        let empty = FuzzCase::Reduction {
            input: FuzzTensor::from_tensor(
                &TensorData::from_storage([2, 0], Storage::F32(vec![])).unwrap(),
            ),
            reduction,
            axis: 1,
            keepdim: false,
        };
        let built = empty.build().unwrap();
        let scheduled = schedule(&built.graph, built.output).unwrap();
        assert!(CpuJit::render(&scheduled.items[0].kernel).is_ok());
    }

    let product = cases[2].clone();
    let product_built = product.build().unwrap();
    let product_output = CpuBackend
        .execute(
            &product_built.graph,
            product_built.output,
            &product_built.oracle,
        )
        .unwrap();
    let artifact = FuzzFailureArtifact::new(
        15,
        23,
        product.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&product_output),
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic Product mismatch".into(),
        },
    )
    .unwrap();
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(),
        artifact
    );
    let zeroed = minimize_case(&product, |candidate| {
        matches!(candidate, FuzzCase::Reduction { input, reduction: FuzzReduction::Product, axis: 1, keepdim: true }
            if input.bytes == vec![0; 24])
    });
    assert!(
        matches!(zeroed, FuzzCase::Reduction { ref input, reduction: FuzzReduction::Product, axis: 1, keepdim: true }
        if input.bytes == vec![0; 24])
    );

    let max = FuzzCase::Reduction {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [5],
                Storage::F32(vec![f32::NEG_INFINITY, f32::NAN, -0.0, 0.0, f32::INFINITY]),
            )
            .unwrap(),
        ),
        reduction: FuzzReduction::Max,
        axis: 0,
        keepdim: false,
    };
    let max_built = max.build().unwrap();
    let max_value = CpuBackend
        .execute(&max_built.graph, max_built.output, &max_built.oracle)
        .unwrap();
    assert_eq!(max_value.scalar_at(0), Scalar::F(f32::INFINITY as f64));
    let min = FuzzCase::Reduction {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [3],
                Storage::F32(vec![f32::from_bits(0x8000_0000), 0.0, f32::INFINITY]),
            )
            .unwrap(),
        ),
        reduction: FuzzReduction::Min,
        axis: 0,
        keepdim: false,
    };
    let min_built = min.build().unwrap();
    let min_value = CpuBackend
        .execute(&min_built.graph, min_built.output, &min_built.oracle)
        .unwrap();
    let Scalar::F(minimum) = min_value.scalar_at(0) else {
        panic!("F32 min output")
    };
    assert_eq!((minimum as f32).to_bits(), 0x8000_0000);

    for (dtype, reduction, storage) in [
        (
            DType::F32,
            FuzzReduction::Max,
            Storage::F32(vec![f32::from_bits(0x7fc0_0001), f32::INFINITY]),
        ),
        (
            DType::F64,
            FuzzReduction::Min,
            Storage::F64(vec![-0.0, 0.0, f64::INFINITY]),
        ),
        (
            DType::F16,
            FuzzReduction::Max,
            Storage::F16(vec![0x8000, 0x0000, 0x7c00]),
        ),
        (
            DType::BF16,
            FuzzReduction::Min,
            Storage::BF16(vec![0x8000, 0x0000, 0x7f80]),
        ),
    ] {
        let case = FuzzCase::Reduction {
            input: FuzzTensor::from_tensor(
                &TensorData::from_storage([storage.len()], storage).unwrap(),
            ),
            reduction,
            axis: 0,
            keepdim: false,
        };
        let built = case.build().unwrap();
        let oracle = CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap();
        assert_eq!(
            crate::execute_elementwise(&built.graph, built.output, &built.oracle)
                .unwrap()
                .to_le_bytes()
                .unwrap(),
            oracle.to_le_bytes().unwrap(),
            "{dtype:?} {reduction:?} special-lane ordering",
        );
    }
}

#[test]
fn raw_reduction_dtype_matrix_preserves_output_policy_and_portable_execution_paths() {
    let dtypes = [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];
    let reductions = [
        (FuzzReduction::Sum, ReduceKind::Sum),
        (FuzzReduction::Mean, ReduceKind::Mean),
        (FuzzReduction::Product, ReduceKind::Product),
        (FuzzReduction::Max, ReduceKind::Max),
        (FuzzReduction::Min, ReduceKind::Min),
    ];

    for dtype in dtypes {
        let values = (0..6).map(|index| match dtype {
            DType::Bool => Scalar::Bool(index % 2 != 0),
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => Scalar::U((index % 3) as u64),
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => Scalar::I((index % 3) as i64 - 1),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 => {
                Scalar::F((index % 3) as f64 - 1.0)
            }
            _ => unreachable!("float8 reduction fuzz is not generated"),
        });
        let input =
            FuzzTensor::from_tensor(&TensorData::from_scalars([2, 3], dtype, values).unwrap());

        for (reduction, kind) in reductions {
            let case = FuzzCase::Reduction {
                input: input.clone(),
                reduction,
                axis: 1,
                keepdim: reduction != FuzzReduction::Mean,
            };
            let encoded = serde_json::to_value(&case).unwrap();
            assert_eq!(encoded["kind"], "reduction");
            assert_eq!(serde_json::from_value::<FuzzCase>(encoded).unwrap(), case);
            let built = case.build().unwrap();
            let expected_dtype = match reduction {
                FuzzReduction::Sum => match dtype {
                    DType::Bool | DType::I8 | DType::I16 => DType::I32,
                    DType::U8 | DType::U16 => DType::U32,
                    DType::F16 | DType::BF16 => DType::F32,
                    _ => dtype,
                },
                FuzzReduction::Mean if !dtype.is_float() => DType::F32,
                FuzzReduction::Mean
                | FuzzReduction::Product
                | FuzzReduction::Max
                | FuzzReduction::Min => dtype,
            };
            assert_eq!(
                built.graph.dtype(built.output).unwrap(),
                expected_dtype,
                "{dtype:?} {reduction:?}"
            );
            let Op::Reduce { kind: actual, .. } = built.graph.op(built.output).unwrap() else {
                panic!("raw fuzz reduction must retain an Op::Reduce root");
            };
            assert_eq!(*actual, kind);
            let oracle = CpuBackend
                .execute(&built.graph, built.output, &built.oracle)
                .unwrap();
            assert_eq!(
                crate::execute_elementwise(&built.graph, built.output, &built.oracle)
                    .unwrap()
                    .storage(),
                oracle.storage(),
                "captured {dtype:?} {reduction:?}",
            );
            let scheduled = schedule(&built.graph, built.output).unwrap();
            assert!(
                scheduled.items[0]
                    .kernel
                    .topological()
                    .unwrap()
                    .iter()
                    .any(|node| matches!(node.kind(), UOpKind::ReduceFinalize))
            );
            assert!(
                CpuJit::render(&scheduled.items[0].kernel).is_ok(),
                "{dtype:?} {reduction:?}"
            );
            assert!(
                CpuJit::render_vectorized(&scheduled.items[0].kernel).is_ok(),
                "{dtype:?} {reduction:?}"
            );
            let captured =
                CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
            let bytes = captured.to_bytes().unwrap();
            assert_eq!(
                CapturedSchedule::from_bytes(&bytes)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
        }

        for reduction in [
            FuzzReduction::Sum,
            FuzzReduction::Mean,
            FuzzReduction::Product,
        ] {
            let empty = FuzzCase::Reduction {
                input: FuzzTensor::from_tensor(
                    &TensorData::from_scalars([2, 0], dtype, std::iter::empty::<Scalar>()).unwrap(),
                ),
                reduction,
                axis: 1,
                keepdim: false,
            };
            let built = empty.build().unwrap();
            let oracle = CpuBackend
                .execute(&built.graph, built.output, &built.oracle)
                .unwrap();
            assert_eq!(
                crate::execute_elementwise(&built.graph, built.output, &built.oracle)
                    .unwrap()
                    .to_le_bytes()
                    .unwrap(),
                oracle.to_le_bytes().unwrap(),
                "empty {dtype:?} {reduction:?}",
            );
            let scheduled = schedule(&built.graph, built.output).unwrap();
            assert!(CpuJit::render(&scheduled.items[0].kernel).is_ok());
            assert!(CpuJit::render_vectorized(&scheduled.items[0].kernel).is_ok());
        }
    }
}

#[test]
fn generated_concat_cases_are_valid_diverse_and_deterministic() {
    let mut found = false;
    let mut arities = std::collections::BTreeSet::new();
    let mut zero_width = false;
    let mut nonzero_width = false;
    let mut axis_zero = false;
    let mut axis_one = false;
    let mut dtypes = std::collections::BTreeSet::new();
    let mut zero_non_axis = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..1024 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::ConcatMany { inputs, axis } = case else {
                continue;
            };
            found = true;
            assert!((2..=4).contains(&inputs.len()));
            let first = &inputs[0];
            assert!((1..=3).contains(&first.shape.len()));
            assert!(axis < first.shape.len());
            for input in &inputs {
                assert_eq!(input.dtype, first.dtype);
                assert_eq!(input.shape.len(), first.shape.len());
                for dimension in 0..first.shape.len() {
                    if dimension != axis {
                        assert_eq!(input.shape[dimension], first.shape[dimension]);
                    }
                }
                zero_width |= input.shape[axis] == 0;
                nonzero_width |= input.shape[axis] != 0;
            }
            arities.insert(inputs.len());
            axis_zero |= axis == 0;
            axis_one |= first.shape.len() >= 2 && axis == 1;
            zero_non_axis |= first
                .shape
                .iter()
                .enumerate()
                .any(|(dimension, extent)| dimension != axis && *extent == 0);
            dtypes.insert(first.dtype);
        }
    }

    assert!(found);
    assert_eq!(arities, std::collections::BTreeSet::from([2, 3, 4]));
    assert!(zero_width && nonzero_width && zero_non_axis);
    assert!(axis_zero && axis_one);
    assert_eq!(dtypes.len(), 13);
}

#[test]
fn concat_many_cases_preserve_arity_order_and_raw_payloads() {
    let many = FuzzCase::ConcatMany {
        inputs: vec![
            FuzzTensor::from_tensor(
                &TensorData::from_storage([1, 2], Storage::F16(vec![0x8000, 0x7e01])).unwrap(),
            ),
            FuzzTensor::from_tensor(
                &TensorData::from_storage([1, 0], Storage::F16(vec![])).unwrap(),
            ),
            FuzzTensor::from_tensor(
                &TensorData::from_storage([1, 2], Storage::F16(vec![0x7c00, 0x3c00])).unwrap(),
            ),
        ],
        axis: 1,
    };
    let encoded = serde_json::to_value(&many).unwrap();
    assert_eq!(encoded["kind"], "concat_many");
    assert_eq!(
        serde_json::from_value::<FuzzCase>(encoded.clone()).unwrap(),
        many
    );
    let mut unknown = encoded;
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    // The original two-input tag remains decodable without schema migration.
    let legacy = FuzzCase::Concat {
        lhs: FuzzTensor::from_tensor(
            &TensorData::from_storage([1], Storage::U64(vec![7])).unwrap(),
        ),
        rhs: FuzzTensor::from_tensor(
            &TensorData::from_storage([1], Storage::U64(vec![9])).unwrap(),
        ),
        axis: 0,
    };
    assert_eq!(
        serde_json::from_value::<FuzzCase>(serde_json::to_value(&legacy).unwrap()).unwrap(),
        legacy
    );
    let too_few = FuzzCase::ConcatMany {
        inputs: vec![FuzzTensor::from_tensor(
            &TensorData::from_storage([1], Storage::Bool(vec![true])).unwrap(),
        )],
        axis: 0,
    };
    assert!(too_few.validate().is_err());
    let mismatched = FuzzCase::ConcatMany {
        inputs: vec![
            FuzzTensor::from_tensor(
                &TensorData::from_storage([1, 1], Storage::I32(vec![1])).unwrap(),
            ),
            FuzzTensor::from_tensor(
                &TensorData::from_storage([2, 1], Storage::I32(vec![2, 3])).unwrap(),
            ),
        ],
        axis: 1,
    };
    assert!(mismatched.validate().is_err());

    let built = many.build().unwrap();
    assert_eq!(
        built.ordered.keys().cloned().collect::<Vec<_>>(),
        vec![
            "input0".to_string(),
            "input1".to_string(),
            "input2".to_string()
        ]
    );
    let Op::Concat { inputs, axis } = built.graph.op(built.output).unwrap() else {
        panic!("concat_many must retain raw Concat")
    };
    assert_eq!(*axis, 1);
    assert_eq!(inputs.len(), 3);
    assert_eq!(
        built.graph.shape(built.output).unwrap(),
        &crate::Shape::from([1, 4])
    );
    assert_eq!(built.graph.dtype(built.output).unwrap(), DType::F16);
    let plan = MovementKernelPlan::from_graph(&built.graph, built.output).unwrap();
    let MovementKernelKind::Concat {
        inputs: planned,
        axis,
    } = &plan.kind
    else {
        panic!("raw Concat must use a movement plan")
    };
    assert_eq!(*axis, 1);
    assert_eq!(planned.len(), 3);
    let scheduled = schedule(&built.graph, built.output).unwrap();
    assert_eq!(scheduled.items.len(), 1);
    assert!(
        matches!(scheduled.items[0].kernel.arg(), UArg::Movement(plan) if matches!(&plan.kind, MovementKernelKind::Concat { inputs, .. } if inputs.len() == 3))
    );
    let scalar = CpuJit::render(&scheduled.items[0].kernel).unwrap();
    assert!(scalar.source.contains("else if"));
    assert!(scalar.source.contains("uint16_t"));
    assert!(CpuJit::render_vectorized(&scheduled.items[0].kernel).is_ok());
    let captured = CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
    let bytes = captured.to_bytes().unwrap();
    assert_eq!(
        CapturedSchedule::from_bytes(&bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );

    let output = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
    assert_eq!(
        FuzzTensor::from_tensor(&output),
        FuzzTensor::from_tensor(
            &TensorData::from_storage([1, 4], Storage::F16(vec![0x8000, 0x7e01, 0x7c00, 0x3c00]))
                .unwrap()
        ),
    );
    let artifact = FuzzFailureArtifact::new(
        17,
        29,
        many.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&output),
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic concat_many mismatch".into(),
        },
    )
    .unwrap();
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(),
        artifact
    );
    let zeroed = minimize_case(&many, |candidate| {
        matches!(candidate, FuzzCase::ConcatMany { inputs, axis: 1 }
            if inputs.len() == 3 && inputs.iter().all(|input| input.bytes.iter().all(|byte| *byte == 0)))
    });
    assert!(matches!(zeroed, FuzzCase::ConcatMany { ref inputs, axis: 1 } if inputs.len() == 3));
}

#[test]
fn generated_unary_cases_are_valid_diverse_and_deterministic() {
    let all_dtypes = [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];
    let mut found = false;
    let mut ops = std::collections::BTreeSet::new();
    let mut dtypes = std::collections::BTreeSet::new();
    let mut dtype_operations = std::collections::BTreeSet::new();
    let mut scalar = false;
    let mut empty = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..12_288 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::Unary { op, input } = case else {
                continue;
            };
            found = true;
            ops.insert(op);
            dtypes.insert(input.dtype);
            dtype_operations.insert((op, input.dtype));
            scalar |= input.shape.is_empty();
            empty |= input.shape.contains(&0);
        }
    }

    assert!(found);
    assert_eq!(ops.len(), 34);
    assert_eq!(dtypes.len(), 13);
    assert_eq!(dtype_operations.len(), 253);
    for op in [
        FuzzUnaryOp::Neg,
        FuzzUnaryOp::Abs,
        FuzzUnaryOp::Exp,
        FuzzUnaryOp::Exp2,
        FuzzUnaryOp::Relu,
        FuzzUnaryOp::Step,
        FuzzUnaryOp::Reciprocal,
        FuzzUnaryOp::Square,
        FuzzUnaryOp::Sqrt,
        FuzzUnaryOp::Rsqrt,
        FuzzUnaryOp::Log2,
        FuzzUnaryOp::Sin,
        FuzzUnaryOp::Cos,
        FuzzUnaryOp::Tan,
        FuzzUnaryOp::Log,
        FuzzUnaryOp::Sinh,
        FuzzUnaryOp::Cosh,
        FuzzUnaryOp::Tanh,
        FuzzUnaryOp::Erf,
        FuzzUnaryOp::Erfc,
        FuzzUnaryOp::Asin,
        FuzzUnaryOp::Acos,
        FuzzUnaryOp::Atan,
        FuzzUnaryOp::Asinh,
        FuzzUnaryOp::Acosh,
        FuzzUnaryOp::Atanh,
        FuzzUnaryOp::Floor,
        FuzzUnaryOp::Ceil,
        FuzzUnaryOp::Trunc,
        FuzzUnaryOp::Round,
        FuzzUnaryOp::Sign,
        FuzzUnaryOp::IsNan,
        FuzzUnaryOp::IsInf,
        FuzzUnaryOp::IsFinite,
    ] {
        assert!(ops.contains(&op), "missing generated {op:?}");
    }
    assert_eq!(all_dtypes.len(), 13);
    assert!(scalar && empty);
}

#[test]
fn generated_compare_cases_are_valid_diverse_and_deterministic() {
    let mut found = false;
    let mut ops = std::collections::BTreeSet::new();
    let mut dtypes = std::collections::BTreeSet::new();
    let mut scalar = false;
    let mut empty = false;
    let mut scalar_rhs = false;
    let mut matching_rhs = false;
    let mut right_aligned_rhs = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..4096 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::Compare { op, lhs, rhs } = case else {
                continue;
            };
            found = true;
            ops.insert(op);
            dtypes.insert(lhs.dtype);
            scalar |= lhs.shape.is_empty();
            empty |= lhs.shape.contains(&0);
            scalar_rhs |= rhs.shape.is_empty();
            matching_rhs |= rhs.shape == lhs.shape;
            right_aligned_rhs |=
                !rhs.shape.is_empty() && rhs.shape != lhs.shape && lhs.shape.ends_with(&rhs.shape);
            assert_eq!(lhs.dtype, rhs.dtype);
        }
    }

    assert!(found);
    assert_eq!(ops.len(), 6);
    assert_eq!(dtypes.len(), 13);
    assert!(scalar && empty && scalar_rhs && matching_rhs && right_aligned_rhs);
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
            empty |= lhs.shape.contains(&0);
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
            empty |= input.shape.contains(&0);
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
                    zero |= values.contains(&0.0);
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
fn generated_permute_cases_cover_passthrough_and_affine_geometry() {
    let mut found = false;
    let mut scalar = false;
    let mut identity = false;
    let mut non_identity = false;
    let mut zero = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..1024 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::Permute { input, axes } = case else {
                continue;
            };
            found = true;
            assert_eq!(axes.len(), input.shape.len());
            assert_eq!(
                axes.iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>(),
                (0..axes.len()).collect()
            );
            scalar |= input.shape.is_empty();
            identity |= axes.iter().copied().eq(0..axes.len());
            non_identity |= !axes.iter().copied().eq(0..axes.len());
            zero |= input.shape.contains(&0);
        }
    }

    assert!(found && scalar && identity && non_identity && zero);
}

#[test]
fn generated_stride_cases_cover_all_dtypes_and_signed_geometry() {
    let mut dtypes = std::collections::BTreeSet::new();
    let mut scalar = false;
    let mut zero = false;
    let mut positive_step = false;
    let mut negative_step = false;
    let mut bounded = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..2048 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::Stride { input, slices } = case else {
                continue;
            };
            assert_eq!(slices.len(), input.shape.len());
            dtypes.insert(input.dtype);
            scalar |= input.shape.is_empty();
            zero |= input.shape.contains(&0);
            positive_step |= slices.iter().any(|slice| slice.step > 1);
            negative_step |= slices.iter().any(|slice| slice.step < 0);
            bounded |= slices
                .iter()
                .any(|slice| slice.start.is_some() || slice.stop.is_some());
        }
    }

    assert_eq!(dtypes.len(), 17);
    assert!(scalar && zero && positive_step && negative_step && bounded);
}

#[test]
fn generated_select_cases_cover_homogeneous_dtypes_and_broadcasts() {
    let mut dtypes = std::collections::BTreeSet::new();
    let mut scalar_condition = false;
    let mut scalar_branch = false;
    let mut aligned_condition = false;
    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..1024 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::Select {
                condition,
                on_true,
                on_false,
            } = case
            else {
                continue;
            };
            assert_eq!(on_true.dtype, on_false.dtype);
            assert_eq!(condition.dtype, DType::Bool);
            dtypes.insert(on_true.dtype);
            scalar_condition |= condition.shape.is_empty();
            scalar_branch |= on_false.shape.is_empty();
            aligned_condition |= condition.shape.len() == 2 && condition.shape[0] == 1;
            let built = FuzzCase::Select {
                condition,
                on_true,
                on_false,
            }
            .build()
            .unwrap();
            assert!(matches!(
                built.graph.op(built.output).unwrap(),
                Op::Select { .. }
            ));
        }
    }
    assert_eq!(dtypes.len(), 13);
    assert!(scalar_condition && scalar_branch && aligned_condition);
}

#[test]
fn select_cases_round_trip_capture_all_dtypes_and_vector_fallbacks() {
    let dtypes = [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];
    for dtype in dtypes {
        let mut graph = crate::Graph::new();
        let condition = graph.input_dtype("condition", crate::Shape::from([2]), DType::Bool);
        let on_true = graph.input_dtype("on_true", crate::Shape::from([2]), dtype);
        let on_false = graph.input_dtype("on_false", crate::Shape::from([]), dtype);
        let output = graph.select(condition, on_true, on_false).unwrap();
        assert!(matches!(graph.op(output).unwrap(), Op::Select { .. }));
        assert_eq!(graph.dtype(output).unwrap(), dtype);
        let uop = crate::lower_graph_elementwise(&graph, output).unwrap();
        assert!(matches!(uop.kind(), UOpKind::Sink));
        assert!(CpuJit::render(&uop).is_ok());
        let vector = CpuJit::render_vectorized(&uop).unwrap();
        if matches!(dtype, DType::F16 | DType::BF16) {
            assert!(!vector.source.contains("B2 VectorProgram"));
        } else if matches!(dtype, DType::F32 | DType::I32) {
            assert!(vector.source.contains("B2 VectorProgram"));
        }
        let scheduled = schedule(&graph, output).unwrap();
        assert_eq!(scheduled.items.len(), 1);
        assert_eq!(
            scheduled.items[0]
                .kernel
                .topological()
                .unwrap()
                .iter()
                .filter(|node| matches!(node.kind(), UOpKind::Ternary(Ternary::Where)))
                .count(),
            1
        );
        let captured = CapturedSchedule::capture(&graph, &scheduled, &[output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        assert_eq!(
            CapturedSchedule::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
    }
    let case = FuzzCase::Select {
        condition: FuzzTensor::from_tensor(
            &TensorData::from_storage([3], Storage::Bool(vec![true, false, true])).unwrap(),
        ),
        on_true: FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [3],
                Storage::F32(vec![f32::from_bits(0x8000_0000), f32::INFINITY, 3.0]),
            )
            .unwrap(),
        ),
        on_false: FuzzTensor::from_tensor(
            &TensorData::from_storage([], Storage::F32(vec![f32::from_bits(0x7fc0_0001)])).unwrap(),
        ),
    };
    let encoded = serde_json::to_value(&case).unwrap();
    assert_eq!(serde_json::from_value::<FuzzCase>(encoded).unwrap(), case);
    let built = case.build().unwrap();
    let output = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
    assert_eq!(
        FuzzTensor::from_tensor(&output),
        FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [3],
                Storage::F32(vec![
                    f32::from_bits(0x8000_0000),
                    f32::from_bits(0x7fc0_0001),
                    3.0
                ])
            )
            .unwrap()
        )
    );
    let artifact = FuzzFailureArtifact::new(
        31,
        37,
        case.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&output),
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic select mismatch".into(),
        },
    )
    .unwrap();
    let artifact_bytes = artifact.to_bytes().unwrap();
    let decoded_artifact = FuzzFailureArtifact::from_bytes(&artifact_bytes).unwrap();
    assert_eq!(decoded_artifact, artifact);
    assert_eq!(decoded_artifact.to_bytes().unwrap(), artifact_bytes);
    let zeroed = minimize_case(
        &case,
        |candidate| matches!(candidate, FuzzCase::Select { condition, on_true, on_false } if condition.shape == vec![3] && on_true.shape == vec![3] && on_false.shape.is_empty() && condition.dtype == DType::Bool && on_true.dtype == DType::F32 && on_false.dtype == DType::F32 && condition.bytes.iter().all(|byte| *byte == 0) && on_true.bytes.iter().all(|byte| *byte == 0) && on_false.bytes.iter().all(|byte| *byte == 0)),
    );
    assert!(
        matches!(zeroed, FuzzCase::Select { ref condition, ref on_true, ref on_false }
        if condition.shape == vec![3]
            && on_true.shape == vec![3]
            && on_false.shape.is_empty()
            && condition.dtype == DType::Bool
            && on_true.dtype == DType::F32
            && on_false.dtype == DType::F32)
    );
    let FuzzCase::Select {
        on_true, on_false, ..
    } = &case
    else {
        unreachable!("fixture is a Select case")
    };
    let malformed = FuzzCase::Select {
        condition: FuzzTensor::from_tensor(
            &TensorData::from_storage([2], Storage::I32(vec![1, 0])).unwrap(),
        ),
        on_true: on_true.clone(),
        on_false: on_false.clone(),
    };
    assert!(malformed.validate().is_err());
}

#[test]
fn cast_cases_cover_the_safe_all_dtype_matrix_without_claiming_undefined_c_casts() {
    const DTYPES: [DType; 13] = [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];

    for from in DTYPES {
        for to in DTYPES {
            // 0 and 1 are exactly representable and in range for every pair.
            // This intentionally does not claim non-finite/out-of-range
            // float-to-int or implementation-defined signed-overflow parity.
            let source = TensorData::from_scalars([2], from, [Scalar::I(0), Scalar::I(1)]).unwrap();
            let case = FuzzCase::Cast {
                input: FuzzTensor::from_tensor(&source),
                to,
            };
            let built = case.build().unwrap();
            assert!(matches!(
                built.graph.op(built.output).unwrap(),
                Op::Cast { dtype, .. } if *dtype == to
            ));
            let oracle = CpuBackend
                .execute(&built.graph, built.output, &built.oracle)
                .unwrap();
            assert_eq!(
                FuzzTensor::from_tensor(&oracle),
                FuzzTensor::from_tensor(&source.cast(to))
            );

            let scheduled = schedule(&built.graph, built.output).unwrap();
            assert_eq!(scheduled.items.len(), 1);
            assert_eq!(
                scheduled.items[0]
                    .kernel
                    .topological()
                    .unwrap()
                    .iter()
                    .filter(|node| matches!(node.kind(), UOpKind::Cast))
                    .count(),
                1,
            );
            let captured =
                CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
            let capture_bytes = captured.to_bytes().unwrap();
            assert_eq!(
                CapturedSchedule::from_bytes(&capture_bytes)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                capture_bytes
            );

            let uop = crate::lower_graph_elementwise(&built.graph, built.output).unwrap();
            assert!(CpuJit::render(&uop).is_ok(), "{from:?} -> {to:?}");
            let vector = CpuJit::render_vectorized(&uop).unwrap();
            if matches!(from, DType::F16 | DType::BF16) || matches!(to, DType::F16 | DType::BF16) {
                assert!(
                    !vector.source.contains("B2 VectorProgram"),
                    "{from:?} -> {to:?}"
                );
                assert!(vector.source.contains("C11 ABI v2"), "{from:?} -> {to:?}");
            }
        }
    }

    // Representative non-half pairs still use B2, while every half endpoint
    // remains on the v17 legacy scalar-per-lane path rather than claiming a
    // tagged half-vector ABI.
    let mut b2_graph = crate::Graph::new();
    let b2_input = b2_graph.input_dtype("input", [2], DType::F32);
    let b2_output = b2_graph.cast(b2_input, DType::I32).unwrap();
    let b2_uop = crate::lower_graph_elementwise(&b2_graph, b2_output).unwrap();
    assert!(
        CpuJit::render_vectorized(&b2_uop)
            .unwrap()
            .source
            .contains("B2 VectorProgram")
    );

    // Finite fractional truncation and unsigned conversion stay in the safe
    // nonnegative domain. Arbitrary half NaN payload identity is not claimed.
    for to in [DType::I32, DType::U32] {
        let input = TensorData::from_storage([3], Storage::F32(vec![0.0, 1.5, 2.5])).unwrap();
        let case = FuzzCase::Cast {
            input: FuzzTensor::from_tensor(&input),
            to,
        };
        let built = case.build().unwrap();
        let output = CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap();
        assert_eq!(
            FuzzTensor::from_tensor(&output),
            FuzzTensor::from_tensor(&input.cast(to))
        );
    }

    let artifact_case = FuzzCase::Cast {
        input: FuzzTensor::from_tensor(
            &TensorData::from_scalars([2], DType::BF16, [Scalar::I(0), Scalar::I(1)]).unwrap(),
        ),
        to: DType::U64,
    };
    let built = artifact_case.build().unwrap();
    let expected = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
    let artifact = FuzzFailureArtifact::new(
        41,
        43,
        artifact_case.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&expected),
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic safe cast mismatch".into(),
        },
    )
    .unwrap();
    let artifact_bytes = artifact.to_bytes().unwrap();
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact_bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
        artifact_bytes
    );
    let minimized = minimize_case(&artifact_case, |candidate| {
        matches!(candidate,
            FuzzCase::Cast { input, to: DType::U64 }
                if input.dtype == DType::BF16
                    && input.shape == vec![2]
                    && input.bytes.iter().all(|byte| *byte == 0)
        )
    });
    assert!(
        matches!(minimized, FuzzCase::Cast { ref input, to: DType::U64 }
        if input.dtype == DType::BF16 && input.shape == vec![2])
    );
}

#[test]
fn generated_cast_cases_reach_every_concrete_dtype_on_the_safe_domain() {
    let mut sources = std::collections::BTreeSet::new();
    let mut targets = std::collections::BTreeSet::new();
    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..16_384 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            let FuzzCase::Cast { input, to } = case else {
                continue;
            };
            input.validate().unwrap();
            let values = input.to_tensor().unwrap();
            for lane in 0..values.len() {
                assert!([0.0, 1.0, 2.0].contains(&values.scalar_at(lane).as_f64()));
            }
            sources.insert(input.dtype);
            targets.insert(to);
        }
    }
    assert_eq!(sources.len(), 13);
    assert_eq!(targets.len(), 13);
}

#[test]
fn binary_cases_cover_all_homogeneous_dtypes_and_raw_storage_boundaries() {
    const DTYPES: [DType; 13] = [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];
    let ops = [
        (FuzzBinaryOp::Add, crate::BinaryOp::Add),
        (FuzzBinaryOp::Sub, crate::BinaryOp::Sub),
        (FuzzBinaryOp::Mul, crate::BinaryOp::Mul),
        (FuzzBinaryOp::Maximum, crate::BinaryOp::Maximum),
    ];

    for (op_index, (fuzz_op, raw_op)) in ops.into_iter().enumerate() {
        for (dtype_index, dtype) in DTYPES.into_iter().enumerate() {
            let shape = match (op_index + dtype_index) % 3 {
                0 => vec![],
                1 => vec![0],
                _ => vec![2],
            };
            let rhs_shape = if (op_index + dtype_index) % 2 == 0 {
                vec![]
            } else {
                shape.clone()
            };
            let values = (0..shape.iter().product::<usize>()).map(|lane| match dtype {
                DType::Bool => Scalar::Bool(lane != 0),
                DType::F16 | DType::BF16 | DType::F32 | DType::F64 => Scalar::F(lane as f64),
                _ => Scalar::I(lane as i64),
            });
            let rhs_values = (0..rhs_shape.iter().product::<usize>()).map(|_| match dtype {
                DType::Bool => Scalar::Bool(true),
                DType::F16 | DType::BF16 | DType::F32 | DType::F64 => Scalar::F(1.0),
                _ => Scalar::I(1),
            });
            let case = FuzzCase::Binary {
                op: fuzz_op,
                lhs: FuzzTensor::from_tensor(
                    &TensorData::from_scalars(shape.clone(), dtype, values).unwrap(),
                ),
                rhs: FuzzTensor::from_tensor(
                    &TensorData::from_scalars(rhs_shape, dtype, rhs_values).unwrap(),
                ),
            };
            let built = case.build().unwrap();
            assert!(
                matches!(built.graph.op(built.output).unwrap(), Op::Binary { op, .. } if *op == raw_op)
            );
            assert_eq!(built.graph.dtype(built.output).unwrap(), dtype);
            assert_eq!(
                built.graph.shape(built.output).unwrap().dims(),
                shape.as_slice()
            );
            let output = CpuBackend
                .execute(&built.graph, built.output, &built.oracle)
                .unwrap();
            assert_eq!(output.dtype(), dtype);

            let scheduled = schedule(&built.graph, built.output).unwrap();
            assert_eq!(scheduled.items.len(), 1);
            assert_eq!(
                scheduled.items[0]
                    .kernel
                    .topological()
                    .unwrap()
                    .iter()
                    .filter(|node| matches!(node.kind(), UOpKind::GraphBinary(op) if *op == raw_op))
                    .count(),
                1,
            );
            let captured =
                CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
            let bytes = captured.to_bytes().unwrap();
            assert_eq!(
                CapturedSchedule::from_bytes(&bytes)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );

            let uop = crate::lower_graph_elementwise(&built.graph, built.output).unwrap();
            assert!(CpuJit::render(&uop).is_ok(), "{fuzz_op:?} {dtype:?}");
            let vector = CpuJit::render_vectorized(&uop).unwrap();
            if matches!(dtype, DType::F16 | DType::BF16) || fuzz_op == FuzzBinaryOp::Maximum {
                assert!(
                    !vector.source.contains("B2 VectorProgram"),
                    "{fuzz_op:?} {dtype:?}"
                );
            }
        }
    }

    for (dtype, storage, expected) in [
        (
            DType::I8,
            Storage::I8(vec![i8::MAX]),
            Storage::I8(vec![i8::MIN]),
        ),
        (DType::U8, Storage::U8(vec![u8::MAX]), Storage::U8(vec![0])),
        (
            DType::I16,
            Storage::I16(vec![i16::MAX]),
            Storage::I16(vec![i16::MIN]),
        ),
        (
            DType::U16,
            Storage::U16(vec![u16::MAX]),
            Storage::U16(vec![0]),
        ),
        (
            DType::I32,
            Storage::I32(vec![i32::MAX]),
            Storage::I32(vec![i32::MIN]),
        ),
        (
            DType::U32,
            Storage::U32(vec![u32::MAX]),
            Storage::U32(vec![0]),
        ),
        (
            DType::I64,
            Storage::I64(vec![i64::MAX]),
            Storage::I64(vec![i64::MIN]),
        ),
        (
            DType::U64,
            Storage::U64(vec![u64::MAX]),
            Storage::U64(vec![0]),
        ),
    ] {
        let lhs = TensorData::from_storage([1], storage).unwrap();
        let rhs = TensorData::from_scalars([1], dtype, [Scalar::I(1)]).unwrap();
        let case = FuzzCase::Binary {
            op: FuzzBinaryOp::Add,
            lhs: FuzzTensor::from_tensor(&lhs),
            rhs: FuzzTensor::from_tensor(&rhs),
        };
        let built = case.build().unwrap();
        let output = CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap();
        assert_eq!(
            FuzzTensor::from_tensor(&output),
            FuzzTensor::from_tensor(&TensorData::from_storage([1], expected).unwrap())
        );
    }

    // Finite/signed-zero/infinity arithmetic is storage-exact here. Arbitrary
    // half-NaN arithmetic or payload identity remains intentionally unclaimed.
    for (dtype, lhs, rhs, expected) in [
        (
            DType::F16,
            Storage::F16(vec![0x8000, 0x7c00]),
            Storage::F16(vec![0x8000, 0x3c00]),
            Storage::F16(vec![0x8000, 0x7c00]),
        ),
        (
            DType::BF16,
            Storage::BF16(vec![0x8000, 0x7f80]),
            Storage::BF16(vec![0x8000, 0x3f80]),
            Storage::BF16(vec![0x8000, 0x7f80]),
        ),
        (
            DType::F32,
            Storage::F32(vec![-0.0, f32::INFINITY]),
            Storage::F32(vec![-0.0, 1.0]),
            Storage::F32(vec![-0.0, f32::INFINITY]),
        ),
        (
            DType::F64,
            Storage::F64(vec![-0.0, f64::INFINITY]),
            Storage::F64(vec![-0.0, 1.0]),
            Storage::F64(vec![-0.0, f64::INFINITY]),
        ),
    ] {
        let case = FuzzCase::Binary {
            op: FuzzBinaryOp::Add,
            lhs: FuzzTensor::from_tensor(&TensorData::from_storage([2], lhs).unwrap()),
            rhs: FuzzTensor::from_tensor(&TensorData::from_storage([2], rhs).unwrap()),
        };
        let built = case.build().unwrap();
        let output = CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap();
        assert_eq!(
            FuzzTensor::from_tensor(&output),
            FuzzTensor::from_tensor(&TensorData::from_storage([2], expected).unwrap())
        );
        assert_eq!(output.dtype(), dtype);
    }

    for (dtype, lhs, rhs, expected) in [
        (
            DType::F32,
            Storage::F32(vec![f32::from_bits(0x7fc0_0001), -0.0]),
            Storage::F32(vec![1.0, 0.0]),
            Storage::F32(vec![f32::from_bits(0x7fc0_0001), -0.0]),
        ),
        (
            DType::F64,
            Storage::F64(vec![f64::from_bits(0x7ff8_0000_0000_0001), -0.0]),
            Storage::F64(vec![1.0, 0.0]),
            Storage::F64(vec![f64::from_bits(0x7ff8_0000_0000_0001), -0.0]),
        ),
    ] {
        let case = FuzzCase::Binary {
            op: FuzzBinaryOp::Maximum,
            lhs: FuzzTensor::from_tensor(&TensorData::from_storage([2], lhs).unwrap()),
            rhs: FuzzTensor::from_tensor(&TensorData::from_storage([2], rhs).unwrap()),
        };
        let built = case.build().unwrap();
        let output = CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap();
        assert_eq!(
            FuzzTensor::from_tensor(&output),
            FuzzTensor::from_tensor(&TensorData::from_storage([2], expected).unwrap())
        );
        assert_eq!(output.dtype(), dtype);
    }

    let bool_lhs =
        TensorData::from_storage([4], Storage::Bool(vec![true, true, false, false])).unwrap();
    let bool_rhs =
        TensorData::from_storage([4], Storage::Bool(vec![true, false, true, false])).unwrap();
    for (op, expected) in [
        (FuzzBinaryOp::Add, vec![true, true, true, false]),
        (FuzzBinaryOp::Sub, vec![false, true, true, false]),
        (FuzzBinaryOp::Mul, vec![true, false, false, false]),
        (FuzzBinaryOp::Maximum, vec![true, true, true, false]),
    ] {
        let case = FuzzCase::Binary {
            op,
            lhs: FuzzTensor::from_tensor(&bool_lhs),
            rhs: FuzzTensor::from_tensor(&bool_rhs),
        };
        let built = case.build().unwrap();
        let output = CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap();
        assert_eq!(
            FuzzTensor::from_tensor(&output),
            FuzzTensor::from_tensor(
                &TensorData::from_storage([4], Storage::Bool(expected)).unwrap()
            )
        );
    }

    for dtype in [DType::F32, DType::I32] {
        let mut b2_graph = crate::Graph::new();
        let b2_lhs = b2_graph.input_dtype("lhs", [2], dtype);
        let b2_rhs = b2_graph.input_dtype("rhs", [2], dtype);
        for op in [
            crate::BinaryOp::Add,
            crate::BinaryOp::Sub,
            crate::BinaryOp::Mul,
        ] {
            let output = b2_graph.binary(op, b2_lhs, b2_rhs).unwrap();
            assert!(
                CpuJit::render_vectorized(
                    &crate::lower_graph_elementwise(&b2_graph, output).unwrap()
                )
                .unwrap()
                .source
                .contains("B2 VectorProgram")
            );
        }
    }

    let artifact_case = FuzzCase::Binary {
        op: FuzzBinaryOp::Sub,
        lhs: FuzzTensor::from_tensor(
            &TensorData::from_storage([2], Storage::I16(vec![3, 1])).unwrap(),
        ),
        rhs: FuzzTensor::from_tensor(&TensorData::from_storage([], Storage::I16(vec![1])).unwrap()),
    };
    let json = serde_json::to_value(&artifact_case).unwrap();
    assert_eq!(
        serde_json::from_value::<FuzzCase>(json).unwrap(),
        artifact_case
    );
    let built = artifact_case.build().unwrap();
    let expected = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
    let artifact = FuzzFailureArtifact::new(
        47,
        53,
        artifact_case.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&expected),
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic raw binary mismatch".into(),
        },
    )
    .unwrap();
    let artifact_bytes = artifact.to_bytes().unwrap();
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact_bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
        artifact_bytes
    );
    let minimized = minimize_case(&artifact_case, |candidate| {
        matches!(candidate,
            FuzzCase::Binary { op: FuzzBinaryOp::Sub, lhs, rhs }
                if lhs.dtype == DType::I16 && rhs.dtype == DType::I16
                    && lhs.shape == vec![2] && rhs.shape.is_empty()
                    && lhs.bytes.iter().all(|byte| *byte == 0)
                    && rhs.bytes.iter().all(|byte| *byte == 0)
        )
    });
    assert!(
        matches!(minimized, FuzzCase::Binary { op: FuzzBinaryOp::Sub, ref lhs, ref rhs }
        if lhs.dtype == DType::I16 && rhs.dtype == DType::I16
            && lhs.shape == vec![2] && rhs.shape.is_empty())
    );
}

#[test]
fn generated_binary_cases_reach_all_ops_dtypes_and_broadcast_geometries() {
    let mut pairs = std::collections::BTreeSet::new();
    let mut scalar_rhs = false;
    let mut scalar = false;
    let mut empty = false;
    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..16_384 {
            let case = generate_case(seed, index);
            assert_eq!(case, generate_case(seed, index));
            case.validate().unwrap();
            let FuzzCase::Binary { op, lhs, rhs } = case else {
                continue;
            };
            pairs.insert((op, lhs.dtype));
            scalar_rhs |= rhs.shape.is_empty();
            scalar |= lhs.shape.is_empty();
            empty |= lhs.shape.contains(&0);
        }
    }
    assert_eq!(pairs.len(), 52);
    assert!(scalar_rhs && scalar && empty);
}

#[test]
fn generated_pad_cases_are_valid_diverse_and_deterministic() {
    let mut found = false;
    let mut dtypes = std::collections::BTreeSet::new();
    let mut scalar = false;
    let mut empty = false;
    let mut before = false;
    let mut after = false;
    let mut asymmetric = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index in 0..1024 {
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
            empty |= input.shape.contains(&0);
            before |= padding.iter().any(|(before, _)| *before != 0);
            after |= padding.iter().any(|(_, after)| *after != 0);
            asymmetric |= padding.iter().any(|(before, after)| before != after);
            dtypes.insert(input.dtype);
        }
    }

    assert!(found);
    assert_eq!(dtypes.len(), 13);
    assert!(scalar && empty && before && after && asymmetric);
}

#[test]
fn pad_cases_round_trip_minimize_and_capture_as_movement_plans() {
    let pad = FuzzCase::Pad {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage([2, 2], Storage::F32(vec![1.0, 2.0, 3.0, 4.0])).unwrap(),
        ),
        padding: vec![(1, 0), (0, 2)],
        fill: FuzzTensor::from_tensor(
            &TensorData::from_storage([], Storage::F32(vec![-0.0])).unwrap(),
        ),
    };
    let value = serde_json::to_value(&pad).unwrap();
    assert_eq!(value["kind"], "pad");
    assert_eq!(
        serde_json::from_value::<FuzzCase>(value.clone()).unwrap(),
        pad
    );
    let mut unknown = value;
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    for case in [
        pad.clone(),
        FuzzCase::Pad {
            input: FuzzTensor::from_tensor(
                &TensorData::from_storage([0, 2], Storage::I32(vec![])).unwrap(),
            ),
            padding: vec![(1, 1), (1, 0)],
            fill: FuzzTensor::from_tensor(
                &TensorData::from_storage([], Storage::I32(vec![-7])).unwrap(),
            ),
        },
        FuzzCase::Pad {
            input: FuzzTensor::from_tensor(
                &TensorData::from_storage([], Storage::Bool(vec![true])).unwrap(),
            ),
            padding: vec![],
            fill: FuzzTensor::from_tensor(
                &TensorData::from_storage([], Storage::Bool(vec![false])).unwrap(),
            ),
        },
    ] {
        let built = case.build().unwrap();
        assert_eq!(
            built.ordered.len(),
            1,
            "Pad fill is plan metadata, not an input binding"
        );
        let Op::Pad { padding, .. } = built.graph.op(built.output).unwrap() else {
            panic!("raw Pad case must retain its Pad root");
        };
        let FuzzCase::Pad {
            padding: expected, ..
        } = &case
        else {
            unreachable!("constructed as Pad")
        };
        assert_eq!(padding, expected);
        let plan = MovementKernelPlan::from_graph(&built.graph, built.output).unwrap();
        let MovementKernelKind::Pad {
            padding: planned,
            fill_bits,
            ..
        } = &plan.kind
        else {
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
        assert!(
            matches!(item.kernel.arg(), UArg::Movement(plan) if matches!(&plan.kind, MovementKernelKind::Pad { .. }))
        );
        assert!(CpuJit::render(&item.kernel).is_ok());
        assert!(CpuJit::render_vectorized(&item.kernel).is_ok());
        let captured =
            CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        assert_eq!(
            CapturedSchedule::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
    }

    let built = pad.build().unwrap();
    let output = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
    let expected = TensorData::from_storage(
        [3, 4],
        Storage::F32(vec![
            -0.0, -0.0, -0.0, -0.0, 1.0, 2.0, -0.0, -0.0, 3.0, 4.0, -0.0, -0.0,
        ]),
    )
    .unwrap();
    assert_eq!(
        FuzzTensor::from_tensor(&output),
        FuzzTensor::from_tensor(&expected)
    );
    let artifact = FuzzFailureArtifact::new(
        12,
        16,
        pad.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&output),
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic Pad mismatch".into(),
        },
    )
    .unwrap();
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(),
        artifact
    );
    let zeroed = minimize_case(&pad, |candidate| {
        matches!(candidate, FuzzCase::Pad { input, fill, .. }
            if input.bytes == vec![0; 16] && fill.bytes == vec![0; 4])
    });
    assert!(matches!(zeroed, FuzzCase::Pad { ref input, ref fill, .. }
        if input.bytes == vec![0; 16] && fill.bytes == vec![0; 4]));

    let nan_fill = FuzzCase::Pad {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage([1], Storage::F32(vec![1.0])).unwrap(),
        ),
        padding: vec![(1, 1)],
        fill: FuzzTensor::from_tensor(
            &TensorData::from_storage([], Storage::F32(vec![f32::from_bits(0x7fc0_0001)])).unwrap(),
        ),
    };
    let nan_built = nan_fill.build().unwrap();
    let MovementKernelKind::Pad { fill_bits, .. } =
        MovementKernelPlan::from_graph(&nan_built.graph, nan_built.output)
            .unwrap()
            .kind
    else {
        unreachable!("Pad plan")
    };
    assert!(f32::from_bits(fill_bits as u32).is_nan());

    // A raw Pad copies finite input lanes at their storage width. Its scalar
    // fill is deliberately a separate commitment through `scalar_at` and
    // `MovementKernelPlan::fill_bits`, so raw half-NaN input/fill payload
    // identity is not claimed here.
    let dtype_cases = vec![
        (
            DType::Bool,
            Storage::Bool(vec![false]),
            Storage::Bool(vec![true]),
            1,
            "uint8_t",
        ),
        (
            DType::I8,
            Storage::I8(vec![i8::MIN]),
            Storage::I8(vec![-1]),
            0xff,
            "int8_t",
        ),
        (
            DType::U8,
            Storage::U8(vec![u8::MAX]),
            Storage::U8(vec![u8::MAX]),
            0xff,
            "uint8_t",
        ),
        (
            DType::I16,
            Storage::I16(vec![i16::MIN]),
            Storage::I16(vec![-1]),
            0xffff,
            "int16_t",
        ),
        (
            DType::U16,
            Storage::U16(vec![u16::MAX]),
            Storage::U16(vec![u16::MAX]),
            0xffff,
            "uint16_t",
        ),
        (
            DType::I32,
            Storage::I32(vec![i32::MIN]),
            Storage::I32(vec![-1]),
            0xffff_ffff,
            "int32_t",
        ),
        (
            DType::U32,
            Storage::U32(vec![u32::MAX]),
            Storage::U32(vec![u32::MAX]),
            0xffff_ffff,
            "uint32_t",
        ),
        (
            DType::I64,
            Storage::I64(vec![i64::MIN]),
            Storage::I64(vec![-1]),
            u64::MAX,
            "int64_t",
        ),
        (
            DType::U64,
            Storage::U64(vec![u64::MAX]),
            Storage::U64(vec![u64::MAX]),
            u64::MAX,
            "uint64_t",
        ),
        (
            DType::F16,
            Storage::F16(vec![0x3c00]),
            Storage::F16(vec![0x8000]),
            0x8000,
            "uint16_t",
        ),
        (
            DType::BF16,
            Storage::BF16(vec![0x3f80]),
            Storage::BF16(vec![0x8000]),
            0x8000,
            "uint16_t",
        ),
        (
            DType::F32,
            Storage::F32(vec![1.0]),
            Storage::F32(vec![-0.0]),
            0x8000_0000,
            "float",
        ),
        (
            DType::F64,
            Storage::F64(vec![1.0]),
            Storage::F64(vec![-0.0]),
            0x8000_0000_0000_0000,
            "double",
        ),
    ];
    for (dtype, input_storage, fill_storage, expected_bits, native_type) in dtype_cases {
        let input = FuzzTensor::from_tensor(&TensorData::from_storage([1], input_storage).unwrap());
        let fill = FuzzTensor::from_tensor(&TensorData::from_storage([], fill_storage).unwrap());
        let case = FuzzCase::Pad {
            input: input.clone(),
            padding: vec![(1, 0)],
            fill,
        };
        let built = case.build().unwrap();
        let MovementKernelKind::Pad { fill_bits, .. } =
            MovementKernelPlan::from_graph(&built.graph, built.output)
                .unwrap()
                .kind
        else {
            unreachable!("Pad plan")
        };
        assert_eq!(built.graph.dtype(built.output).unwrap(), dtype);
        assert_eq!(fill_bits, expected_bits);
        let scheduled = schedule(&built.graph, built.output).unwrap();
        let scalar = CpuJit::render(&scheduled.items[0].kernel).unwrap();
        assert!(scalar.source.contains(native_type));
        match dtype {
            DType::Bool => assert!(scalar.source.contains("((uint8_t)1)")),
            DType::F16 | DType::BF16 => assert!(scalar.source.contains("0x8000u")),
            DType::F32 => assert!(scalar.source.contains("0x80000000u")),
            DType::F64 => assert!(scalar.source.contains("0x8000000000000000")),
            _ => assert!(scalar.source.contains("UINT64_C")),
        }
        assert!(CpuJit::render_vectorized(&scheduled.items[0].kernel).is_ok());
        let captured =
            CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        assert_eq!(
            CapturedSchedule::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        let output = CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap();
        let output = FuzzTensor::from_tensor(&output);
        assert_eq!(&output.bytes[dtype.itemsize()..], input.bytes.as_slice());
    }

    for (dtype, fill, expect_nan) in [
        (DType::F16, 0x7e01_u16, true),
        (DType::F16, 0x7c00_u16, false),
        (DType::BF16, 0x7fc1_u16, true),
        (DType::BF16, 0x7f80_u16, false),
    ] {
        let input = FuzzTensor::from_tensor(
            &TensorData::from_scalars([1], dtype, [Scalar::F(1.0)]).unwrap(),
        );
        let fill = FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [],
                if dtype == DType::F16 {
                    Storage::F16(vec![fill])
                } else {
                    Storage::BF16(vec![fill])
                },
            )
            .unwrap(),
        );
        let built = FuzzCase::Pad {
            input,
            padding: vec![(1, 0)],
            fill,
        }
        .build()
        .unwrap();
        let MovementKernelKind::Pad { fill_bits, .. } =
            MovementKernelPlan::from_graph(&built.graph, built.output)
                .unwrap()
                .kind
        else {
            unreachable!("Pad plan")
        };
        let committed = if dtype == DType::F16 {
            crate::f16_to_f32(fill_bits as u16)
        } else {
            crate::bf16_to_f32(fill_bits as u16)
        };
        assert!(if expect_nan {
            committed.is_nan()
        } else {
            committed.is_infinite()
        });
    }

    for (dtype, fill, expect_nan) in [
        (
            DType::F32,
            Storage::F32(vec![f32::from_bits(0x7fc0_0001)]),
            true,
        ),
        (DType::F32, Storage::F32(vec![f32::INFINITY]), false),
        (
            DType::F64,
            Storage::F64(vec![f64::from_bits(0x7ff8_0000_0000_0001)]),
            true,
        ),
        (DType::F64, Storage::F64(vec![f64::INFINITY]), false),
    ] {
        let input = FuzzTensor::from_tensor(
            &TensorData::from_scalars([1], dtype, [Scalar::F(1.0)]).unwrap(),
        );
        let fill = FuzzTensor::from_tensor(&TensorData::from_storage([], fill).unwrap());
        let built = FuzzCase::Pad {
            input,
            padding: vec![(1, 0)],
            fill,
        }
        .build()
        .unwrap();
        let MovementKernelKind::Pad { fill_bits, .. } =
            MovementKernelPlan::from_graph(&built.graph, built.output)
                .unwrap()
                .kind
        else {
            unreachable!("Pad plan")
        };
        let committed = if dtype == DType::F32 {
            f32::from_bits(fill_bits as u32).is_nan()
        } else {
            f64::from_bits(fill_bits).is_nan()
        };
        assert_eq!(committed, expect_nan);
        if !expect_nan {
            assert_eq!(
                fill_bits,
                if dtype == DType::F32 {
                    f32::INFINITY.to_bits() as u64
                } else {
                    f64::INFINITY.to_bits()
                }
            );
        }
    }

    let bad_shape = FuzzCase::Pad {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage([1], Storage::F32(vec![1.0])).unwrap(),
        ),
        padding: vec![(0, 1)],
        fill: FuzzTensor::from_tensor(
            &TensorData::from_storage([1], Storage::F32(vec![0.0])).unwrap(),
        ),
    };
    let bad_dtype = FuzzCase::Pad {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage([1], Storage::F32(vec![1.0])).unwrap(),
        ),
        padding: vec![(0, 1)],
        fill: FuzzTensor::from_tensor(
            &TensorData::from_storage([], Storage::I32(vec![0])).unwrap(),
        ),
    };
    let bad_padding = FuzzCase::Pad {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage([1], Storage::F32(vec![1.0])).unwrap(),
        ),
        padding: vec![],
        fill: FuzzTensor::from_tensor(
            &TensorData::from_storage([], Storage::F32(vec![0.0])).unwrap(),
        ),
    };
    assert!(bad_shape.validate().is_err());
    assert!(bad_dtype.validate().is_err());
    assert!(bad_padding.validate().is_err());
}

#[test]
fn generated_gather_cases_are_valid_diverse_and_deterministic() {
    let mut found = false;
    let mut dtypes = std::collections::BTreeSet::new();
    let mut index_i32 = false;
    let mut index_i64 = false;
    let mut axes = std::collections::BTreeSet::new();
    let mut empty = false;
    let mut duplicate = false;

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index_number in 0..1024 {
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
            for (dimension, (&source, &selected)) in
                input.shape.iter().zip(&index.shape).enumerate()
            {
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
            empty |= input.shape.contains(&0) || index.shape.contains(&0);
            axes.insert((input.shape.len(), axis));
            dtypes.insert(input.dtype);
            index_i32 |= index.dtype == DType::I32;
            index_i64 |= index.dtype == DType::I64;
        }
    }

    assert!(found);
    assert_eq!(dtypes.len(), 13);
    assert!(index_i32 && index_i64);
    assert!(empty && duplicate);
    assert!(axes.contains(&(1, 0)) && axes.iter().any(|(rank, axis)| *rank >= 2 && *axis == 1));
}

#[test]
fn gather_cases_round_trip_minimize_and_capture_as_movement_plans() {
    let gather = FuzzCase::Gather {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [2, 4],
                Storage::F32(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            )
            .unwrap(),
        ),
        index: FuzzTensor::from_tensor(
            &TensorData::from_storage([2, 3], Storage::I32(vec![3, 1, 1, 0, 2, 2])).unwrap(),
        ),
        axis: 1,
    };
    let value = serde_json::to_value(&gather).unwrap();
    assert_eq!(value["kind"], "gather");
    assert_eq!(
        serde_json::from_value::<FuzzCase>(value.clone()).unwrap(),
        gather
    );
    let mut unknown = value;
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    for case in [
        gather.clone(),
        FuzzCase::Gather {
            input: FuzzTensor::from_tensor(
                &TensorData::from_storage([3], Storage::I32(vec![10, 20, 30])).unwrap(),
            ),
            index: FuzzTensor::from_tensor(
                &TensorData::from_storage([3], Storage::I64(vec![2, 0, 1])).unwrap(),
            ),
            axis: 0,
        },
        FuzzCase::Gather {
            input: FuzzTensor::from_tensor(
                &TensorData::from_storage([2, 0], Storage::F16(vec![])).unwrap(),
            ),
            index: FuzzTensor::from_tensor(
                &TensorData::from_storage([2, 0], Storage::I32(vec![])).unwrap(),
            ),
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
        let MovementKernelKind::Gather {
            axis: planned,
            input,
            index,
        } = &plan.kind
        else {
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
        assert!(
            matches!(item.kernel.arg(), UArg::Movement(plan) if matches!(&plan.kind, MovementKernelKind::Gather { .. }))
        );
        let scalar = CpuJit::render(&item.kernel).unwrap();
        if index.shape.numel().unwrap() != 0 {
            assert!(
                scalar.source.contains("rg_selected < 0") && scalar.source.contains("failure[1]=3")
            );
        } else {
            assert!(scalar.source.contains("empty gather domain"));
        }
        assert!(CpuJit::render_vectorized(&item.kernel).is_ok());
        let captured =
            CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        assert_eq!(
            CapturedSchedule::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
    }

    let built = gather.build().unwrap();
    let output = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
    assert_eq!(
        FuzzTensor::from_tensor(&output),
        FuzzTensor::from_tensor(
            &TensorData::from_storage([2, 3], Storage::F32(vec![3.0, 1.0, 1.0, 4.0, 6.0, 6.0]))
                .unwrap()
        ),
    );
    let artifact = FuzzFailureArtifact::new(
        13,
        17,
        gather.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&output),
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic Gather mismatch".into(),
        },
    )
    .unwrap();
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(),
        artifact
    );
    let zeroed = minimize_case(&gather, |candidate| {
        matches!(candidate, FuzzCase::Gather { input, index, axis }
            if input.bytes == vec![0; 32] && index.bytes == vec![0; 24] && *axis == 1)
    });
    assert!(
        matches!(zeroed, FuzzCase::Gather { ref input, ref index, axis }
        if input.bytes == vec![0; 32] && index.bytes == vec![0; 24] && axis == 1)
    );

    let ieee = FuzzCase::Gather {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [3],
                Storage::F32(vec![
                    f32::from_bits(0x8000_0000),
                    f32::from_bits(0x7fc0_0001),
                    f32::INFINITY,
                ]),
            )
            .unwrap(),
        ),
        index: FuzzTensor::from_tensor(
            &TensorData::from_storage([3], Storage::I32(vec![1, 0, 2])).unwrap(),
        ),
        axis: 0,
    };
    let ieee_built = ieee.build().unwrap();
    let ieee_output = CpuBackend
        .execute(&ieee_built.graph, ieee_built.output, &ieee_built.oracle)
        .unwrap();
    assert_eq!(
        FuzzTensor::from_tensor(&ieee_output),
        FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [3],
                Storage::F32(vec![f32::from_bits(0x7fc0_0001), -0.0, f32::INFINITY])
            )
            .unwrap()
        ),
    );

    // Raw Gather selects storage lanes directly through MovementKernelPlan;
    // unlike scalar helpers, no value commitment occurs between input and
    // output. These finite lanes also retain exact CPU-oracle payloads.
    let dtype_cases = vec![
        (
            DType::Bool,
            Storage::Bool(vec![true, false, true]),
            "uint8_t",
        ),
        (DType::I8, Storage::I8(vec![i8::MIN, -1, i8::MAX]), "int8_t"),
        (DType::U8, Storage::U8(vec![0, 1, u8::MAX]), "uint8_t"),
        (
            DType::I16,
            Storage::I16(vec![i16::MIN, -1, i16::MAX]),
            "int16_t",
        ),
        (DType::U16, Storage::U16(vec![0, 1, u16::MAX]), "uint16_t"),
        (
            DType::I32,
            Storage::I32(vec![i32::MIN, -1, i32::MAX]),
            "int32_t",
        ),
        (DType::U32, Storage::U32(vec![0, 1, u32::MAX]), "uint32_t"),
        (
            DType::I64,
            Storage::I64(vec![i64::MIN, -1, i64::MAX]),
            "int64_t",
        ),
        (DType::U64, Storage::U64(vec![0, 1, u64::MAX]), "uint64_t"),
        (
            DType::F16,
            Storage::F16(vec![0x3c00, 0x4000, 0x4200]),
            "uint16_t",
        ),
        (
            DType::BF16,
            Storage::BF16(vec![0x3f80, 0x4000, 0x4040]),
            "uint16_t",
        ),
        (DType::F32, Storage::F32(vec![1.0, 2.0, 3.0]), "float"),
        (DType::F64, Storage::F64(vec![1.0, 2.0, 3.0]), "double"),
    ];
    for (position, (dtype, storage, native_type)) in dtype_cases.into_iter().enumerate() {
        let input = FuzzTensor::from_tensor(&TensorData::from_storage([3], storage).unwrap());
        let index = FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [3],
                if position % 2 == 0 {
                    Storage::I32(vec![2, 0, 1])
                } else {
                    Storage::I64(vec![2, 0, 1])
                },
            )
            .unwrap(),
        );
        let case = FuzzCase::Gather {
            input: input.clone(),
            index: index.clone(),
            axis: 0,
        };
        let built = case.build().unwrap();
        let plan = MovementKernelPlan::from_graph(&built.graph, built.output).unwrap();
        let MovementKernelKind::Gather {
            input: planned,
            index: planned_index,
            ..
        } = &plan.kind
        else {
            unreachable!("Gather plan")
        };
        assert_eq!(plan.dtype, dtype);
        assert_eq!(planned.dtype, dtype);
        assert_eq!(planned_index.dtype, index.dtype);
        let scheduled = schedule(&built.graph, built.output).unwrap();
        let scalar = CpuJit::render(&scheduled.items[0].kernel).unwrap();
        assert!(scalar.source.contains(native_type));
        assert!(scalar.source.contains("rg_selected < 0"));
        assert!(CpuJit::render_vectorized(&scheduled.items[0].kernel).is_ok());
        let captured =
            CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        assert_eq!(
            CapturedSchedule::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        let selected = plan
            .execute(&[input.to_tensor().unwrap(), index.to_tensor().unwrap()])
            .unwrap();
        let oracle = CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap();
        let expected = [
            &input.bytes[2 * dtype.itemsize()..3 * dtype.itemsize()],
            &input.bytes[..dtype.itemsize()],
            &input.bytes[dtype.itemsize()..2 * dtype.itemsize()],
        ]
        .concat();
        assert_eq!(FuzzTensor::from_tensor(&selected).bytes, expected);
        assert_eq!(FuzzTensor::from_tensor(&oracle).bytes, expected);
    }

    for (dtype, storage) in [
        (DType::F16, Storage::F16(vec![0x8000, 0x7e01, 0x7c00])),
        (DType::BF16, Storage::BF16(vec![0x8000, 0x7fc1, 0x7f80])),
        (
            DType::F32,
            Storage::F32(vec![
                f32::from_bits(0x8000_0000),
                f32::from_bits(0x7fc0_0001),
                f32::INFINITY,
            ]),
        ),
        (
            DType::F64,
            Storage::F64(vec![
                f64::from_bits(0x8000_0000_0000_0000),
                f64::from_bits(0x7ff8_0000_0000_0001),
                f64::INFINITY,
            ]),
        ),
    ] {
        let input = FuzzTensor::from_tensor(&TensorData::from_storage([3], storage).unwrap());
        let index = FuzzTensor::from_tensor(
            &TensorData::from_storage([3], Storage::I64(vec![1, 0, 2])).unwrap(),
        );
        let built = FuzzCase::Gather {
            input: input.clone(),
            index: index.clone(),
            axis: 0,
        }
        .build()
        .unwrap();
        let plan = MovementKernelPlan::from_graph(&built.graph, built.output).unwrap();
        let selected = plan
            .execute(&[input.to_tensor().unwrap(), index.to_tensor().unwrap()])
            .unwrap();
        let expected = [
            &input.bytes[dtype.itemsize()..2 * dtype.itemsize()],
            &input.bytes[..dtype.itemsize()],
            &input.bytes[2 * dtype.itemsize()..3 * dtype.itemsize()],
        ]
        .concat();
        assert_eq!(FuzzTensor::from_tensor(&selected).bytes, expected);
    }

    let bad_dtype = FuzzCase::Gather {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage([2], Storage::F32(vec![1.0, 2.0])).unwrap(),
        ),
        index: FuzzTensor::from_tensor(
            &TensorData::from_storage([1], Storage::I16(vec![0])).unwrap(),
        ),
        axis: 0,
    };
    let bad_index = FuzzCase::Gather {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage([2], Storage::F32(vec![1.0, 2.0])).unwrap(),
        ),
        index: FuzzTensor::from_tensor(
            &TensorData::from_storage([1], Storage::I32(vec![2])).unwrap(),
        ),
        axis: 0,
    };
    assert!(bad_dtype.validate().is_err());
    assert!(bad_index.validate().is_err());
}

#[test]
fn generated_scatter_cases_are_valid_diverse_and_deterministic() {
    let mut found = false;
    let mut replace_dtypes = std::collections::BTreeSet::new();
    let mut add_f32 = false;
    let mut add_f64 = false;
    let mut index_i32 = false;
    let mut index_i64 = false;
    let mut zero_axis = false;
    let mut empty = false;
    let mut duplicate = false;
    let mut axes = std::collections::BTreeSet::new();

    for seed in [0, 0x1234, 0xfeed_cafe] {
        for index_number in 0..1024 {
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
            empty |= base.shape.contains(&0) || index.shape.contains(&0);
            zero_axis |= base.shape[axis] == 0;
            axes.insert((base.shape.len(), axis));
            index_i32 |= index.dtype == DType::I32;
            index_i64 |= index.dtype == DType::I64;
            match op {
                FuzzScatterOp::Replace => {
                    replace_dtypes.insert(base.dtype);
                }
                FuzzScatterOp::Add => {
                    assert!(matches!(base.dtype, DType::F32 | DType::F64));
                    add_f32 |= base.dtype == DType::F32;
                    add_f64 |= base.dtype == DType::F64;
                }
            }
        }
    }

    assert!(found);
    assert_eq!(replace_dtypes.len(), 13);
    assert!(add_f32 && add_f64);
    assert!(index_i32 && index_i64 && zero_axis && empty && duplicate);
    assert!(axes.contains(&(1, 0)) && axes.iter().any(|(rank, axis)| *rank >= 2 && *axis == 1));
}

#[test]
fn scatter_cases_round_trip_minimize_and_capture_as_movement_plans() {
    let replace = FuzzCase::Scatter {
        base: FuzzTensor::from_tensor(
            &TensorData::from_storage([1, 4], Storage::F32(vec![10.0, 20.0, 30.0, 40.0])).unwrap(),
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
    assert_eq!(
        serde_json::from_value::<FuzzCase>(value.clone()).unwrap(),
        replace
    );
    let mut unknown = value;
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(true));
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
            base: FuzzTensor::from_tensor(
                &TensorData::from_storage([2], Storage::F64(vec![1.0, 10.0])).unwrap(),
            ),
            index: FuzzTensor::from_tensor(
                &TensorData::from_storage([2], Storage::I64(vec![1, 1])).unwrap(),
            ),
            updates: FuzzTensor::from_tensor(
                &TensorData::from_storage([2], Storage::F64(vec![0.5, 4.0])).unwrap(),
            ),
            axis: 0,
            op: FuzzScatterOp::Add,
        },
        FuzzCase::Scatter {
            base: FuzzTensor::from_tensor(
                &TensorData::from_storage([2, 0], Storage::F16(vec![])).unwrap(),
            ),
            index: FuzzTensor::from_tensor(
                &TensorData::from_storage([2, 0], Storage::I32(vec![])).unwrap(),
            ),
            updates: FuzzTensor::from_tensor(
                &TensorData::from_storage([2, 0], Storage::F16(vec![])).unwrap(),
            ),
            axis: 1,
            op: FuzzScatterOp::Replace,
        },
    ] {
        let built = case.build().unwrap();
        let Op::Scatter { axis, add, .. } = built.graph.op(built.output).unwrap() else {
            panic!("raw Scatter case must retain its Scatter root");
        };
        let FuzzCase::Scatter {
            axis: expected, op, ..
        } = &case
        else {
            unreachable!("constructed as Scatter")
        };
        assert_eq!(axis, expected);
        assert_eq!(*add, *op == FuzzScatterOp::Add);
        let plan = MovementKernelPlan::from_graph(&built.graph, built.output).unwrap();
        let MovementKernelKind::Scatter {
            axis: planned,
            add: planned_add,
            index,
            ..
        } = &plan.kind
        else {
            panic!("Scatter root must use a Scatter movement plan");
        };
        assert_eq!(planned, expected);
        assert_eq!(*planned_add, *op == FuzzScatterOp::Add);
        let scheduled = schedule(&built.graph, built.output).unwrap();
        assert_eq!(scheduled.items.len(), 1);
        let item = &scheduled.items[0];
        assert!(item.boundary.is_none());
        assert!(matches!(item.kernel.kind(), UOpKind::Movement));
        assert!(
            matches!(item.kernel.arg(), UArg::Movement(plan) if matches!(&plan.kind, MovementKernelKind::Scatter { .. }))
        );
        let scalar = CpuJit::render(&item.kernel).unwrap();
        assert!(scalar.source.contains("memcpy("));
        if index.shape.numel().unwrap() == 0 {
            assert!(!scalar.source.contains("rg_selected"));
        } else {
            assert!(
                scalar.source.contains("rg_selected < 0") && scalar.source.contains("failure[1]=3")
            );
            if *op == FuzzScatterOp::Add {
                assert!(scalar.source.contains("] += ((const"));
            } else {
                assert!(scalar.source.contains("] = ((const"));
            }
        }
        assert!(CpuJit::render_vectorized(&item.kernel).is_ok());
        let captured =
            CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        assert_eq!(
            CapturedSchedule::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
    }

    let built = replace.build().unwrap();
    let output = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
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
    let add_output = CpuBackend
        .execute(&add_built.graph, add_built.output, &add_built.oracle)
        .unwrap();
    assert_eq!(
        FuzzTensor::from_tensor(&add_output),
        FuzzTensor::from_tensor(
            &TensorData::from_storage([1, 3], Storage::F32(vec![1.0, 14.75, 100.0])).unwrap(),
        ),
    );
    let artifact = FuzzFailureArtifact::new(
        14,
        19,
        replace.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&output),
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic Scatter mismatch".into(),
        },
    )
    .unwrap();
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(),
        artifact
    );
    let zeroed = minimize_case(&replace, |candidate| {
        matches!(candidate, FuzzCase::Scatter { base, index, updates, axis, op }
            if base.bytes == vec![0; 16] && index.bytes == vec![0; 12]
                && updates.bytes == vec![0; 12] && *axis == 1 && *op == FuzzScatterOp::Replace)
    });
    assert!(
        matches!(zeroed, FuzzCase::Scatter { ref base, ref index, ref updates, axis, op }
        if base.bytes == vec![0; 16] && index.bytes == vec![0; 12]
            && updates.bytes == vec![0; 12] && axis == 1 && op == FuzzScatterOp::Replace)
    );

    let ieee = FuzzCase::Scatter {
        base: FuzzTensor::from_tensor(
            &TensorData::from_storage([3], Storage::F32(vec![0.0, 1.0, 2.0])).unwrap(),
        ),
        index: FuzzTensor::from_tensor(
            &TensorData::from_storage([3], Storage::I32(vec![2, 0, 1])).unwrap(),
        ),
        updates: FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [3],
                Storage::F32(vec![
                    f32::from_bits(0x8000_0000),
                    f32::from_bits(0x7fc0_0001),
                    f32::INFINITY,
                ]),
            )
            .unwrap(),
        ),
        axis: 0,
        op: FuzzScatterOp::Replace,
    };
    let ieee_built = ieee.build().unwrap();
    let ieee_output = CpuBackend
        .execute(&ieee_built.graph, ieee_built.output, &ieee_built.oracle)
        .unwrap();
    assert_eq!(
        FuzzTensor::from_tensor(&ieee_output),
        FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [3],
                Storage::F32(vec![f32::from_bits(0x7fc0_0001), f32::INFINITY, -0.0])
            )
            .unwrap()
        ),
    );

    let bad_add = FuzzCase::Scatter {
        base: FuzzTensor::from_tensor(
            &TensorData::from_storage([2], Storage::I32(vec![1, 2])).unwrap(),
        ),
        index: FuzzTensor::from_tensor(
            &TensorData::from_storage([1], Storage::I32(vec![0])).unwrap(),
        ),
        updates: FuzzTensor::from_tensor(
            &TensorData::from_storage([1], Storage::I32(vec![3])).unwrap(),
        ),
        axis: 0,
        op: FuzzScatterOp::Add,
    };
    let bad_index = FuzzCase::Scatter {
        base: FuzzTensor::from_tensor(
            &TensorData::from_storage([2], Storage::F32(vec![1.0, 2.0])).unwrap(),
        ),
        index: FuzzTensor::from_tensor(
            &TensorData::from_storage([1], Storage::I64(vec![-1])).unwrap(),
        ),
        updates: FuzzTensor::from_tensor(
            &TensorData::from_storage([1], Storage::F32(vec![3.0])).unwrap(),
        ),
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
    assert_eq!(
        serde_json::from_value::<FuzzCase>(value.clone()).unwrap(),
        tensor_t
    );
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
            &[
                built.graph.shape(*source).unwrap().dims()[1],
                built.graph.shape(*source).unwrap().dims()[0]
            ]
        );
        let scheduled = schedule(&built.graph, built.output).unwrap();
        for item in &scheduled.items {
            assert!(item.boundary.is_none());
            assert!(
                item.kernel
                    .topological()
                    .unwrap()
                    .iter()
                    .any(|node| { matches!(node.arg(), UArg::ViewBufferIndex { .. }) })
            );
            assert!(CpuJit::render(&item.kernel).is_ok());
            assert!(CpuJit::render_vectorized(&item.kernel).is_ok());
        }
        let captured =
            CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
        assert_eq!(decoded.items.len(), scheduled.items.len());
    }

    let built = tensor_t.build().unwrap();
    let output = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
    assert_eq!(
        output.storage(),
        &Storage::F32(vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0])
    );
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
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(),
        artifact
    );
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

#[test]
fn general_permute_captures_identity_and_affine_views_for_all_dtypes() {
    let dtypes = [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];
    for (position, dtype) in dtypes.into_iter().enumerate() {
        let axes = if position % 2 == 0 {
            vec![0, 1]
        } else {
            vec![1, 0]
        };
        let case = FuzzCase::Permute {
            input: FuzzTensor::from_tensor(&TensorData::zeros_with_dtype([2, 3], dtype).unwrap()),
            axes: axes.clone(),
        };
        let built = case.build().unwrap();
        let expected_shape = if axes == [0, 1] { [2, 3] } else { [3, 2] };
        assert_eq!(
            built.graph.shape(built.output).unwrap().dims(),
            &expected_shape
        );
        let scheduled = schedule(&built.graph, built.output).unwrap();
        if axes == [0, 1] {
            assert!(matches!(
                built.graph.op(built.output).unwrap(),
                Op::Input { .. }
            ));
            assert!(scheduled.items.is_empty());
        } else {
            assert!(matches!(
                built.graph.op(built.output).unwrap(),
                Op::Permute { axes, .. } if axes == &vec![1, 0]
            ));
            assert_eq!(scheduled.items.len(), 1);
            let nodes = scheduled.items[0].kernel.topological().unwrap();
            assert!(
                nodes
                    .iter()
                    .any(|node| matches!(node.arg(), UArg::ViewBufferIndex { .. }))
            );
            assert!(CpuJit::render(&scheduled.items[0].kernel).is_ok());
            assert!(CpuJit::render_vectorized(&scheduled.items[0].kernel).is_ok());
        }
        let capture = CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        let bytes = capture.to_bytes().unwrap();
        let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
        let oracle = CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap();
        let replay = crate::CapturedReplayExecutor::default()
            .replay(
                &decoded,
                &built.ordered,
                crate::CapturedReplayOptions {
                    backend: crate::CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        assert_eq!(replay.outputs[0].storage(), oracle.storage());
        assert_eq!(replay.trace.items.is_empty(), axes == [0, 1]);
    }

    let scalar = FuzzCase::Permute {
        input: FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::I(7), DType::I32)),
        axes: vec![],
    };
    let scalar_built = scalar.build().unwrap();
    assert!(
        schedule(&scalar_built.graph, scalar_built.output)
            .unwrap()
            .items
            .is_empty()
    );

    let artifact_case = FuzzCase::Permute {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [2, 1, 3],
                Storage::F32(vec![-0.0, 1.0, f32::INFINITY, 3.0, 4.0, f32::NAN]),
            )
            .unwrap(),
        ),
        axes: vec![2, 0, 1],
    };
    let value = serde_json::to_value(&artifact_case).unwrap();
    assert_eq!(value["kind"], "permute");
    assert_eq!(
        serde_json::from_value::<FuzzCase>(value).unwrap(),
        artifact_case
    );
    let artifact_built = artifact_case.build().unwrap();
    let expected = CpuBackend
        .execute(
            &artifact_built.graph,
            artifact_built.output,
            &artifact_built.oracle,
        )
        .unwrap();
    let artifact = FuzzFailureArtifact::new(
        17,
        29,
        artifact_case.clone(),
        FuzzPath::NativeVector,
        FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&expected),
        FuzzOutcome::Error {
            class: "actual".into(),
            detail: "synthetic permute mismatch".into(),
        },
    )
    .unwrap();
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(),
        artifact
    );
    let zeroed = minimize_case(&artifact_case, |candidate| {
        matches!(candidate, FuzzCase::Permute { input, axes }
            if input.shape == vec![2, 1, 3]
                && input.bytes == vec![0; 24]
                && axes == &vec![2, 0, 1])
    });
    assert!(matches!(zeroed, FuzzCase::Permute { ref input, ref axes }
        if input.bytes == vec![0; 24] && axes == &vec![2, 0, 1]));

    let malformed = FuzzCase::Permute {
        input: FuzzTensor::from_tensor(&TensorData::zeros_with_dtype([2, 3], DType::F32).unwrap()),
        axes: vec![0, 0],
    };
    assert!(malformed.validate().is_err());
}

#[test]
fn signed_stride_captures_affine_views_for_all_dtypes_and_preserves_raw_order() {
    let dtypes = [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F8E4M3,
        DType::F8E5M2,
        DType::F8E4M3FNUZ,
        DType::F8E5M2FNUZ,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];
    let slices = vec![
        FuzzSlice {
            start: None,
            stop: None,
            step: -1,
        },
        FuzzSlice {
            start: None,
            stop: None,
            step: 2,
        },
    ];
    for dtype in dtypes {
        let case = FuzzCase::Stride {
            input: FuzzTensor::from_tensor(&TensorData::zeros_with_dtype([2, 3], dtype).unwrap()),
            slices: slices.clone(),
        };
        let built = case.build().unwrap();
        assert_eq!(built.graph.shape(built.output).unwrap().dims(), &[2, 2]);
        assert_eq!(built.graph.dtype(built.output).unwrap(), dtype);
        assert!(matches!(
            built.graph.op(built.output).unwrap(),
            Op::Stride { slices: actual, .. }
                if actual.iter().map(|slice| slice.step).eq([-1, 2])
        ));
        let scheduled = schedule(&built.graph, built.output).unwrap();
        assert_eq!(scheduled.items.len(), 1);
        let nodes = scheduled.items[0].kernel.topological().unwrap();
        assert!(
            nodes
                .iter()
                .any(|node| matches!(node.arg(), UArg::ViewBufferIndex { .. }))
        );
        assert!(CpuJit::render(&scheduled.items[0].kernel).is_ok());
        assert!(CpuJit::render_vectorized(&scheduled.items[0].kernel).is_ok());
        let captured =
            CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
        let oracle = CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap();
        let replay = crate::CapturedReplayExecutor::default()
            .replay(
                &decoded,
                &built.ordered,
                crate::CapturedReplayOptions {
                    backend: crate::CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        assert_eq!(replay.outputs[0].storage(), oracle.storage());
    }

    let special = FuzzCase::Stride {
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [2, 4],
                Storage::F32(vec![
                    f32::from_bits(0x8000_0000),
                    f32::from_bits(0x7fc0_0001),
                    f32::INFINITY,
                    1.0,
                    f32::NEG_INFINITY,
                    2.0,
                    3.0,
                    4.0,
                ]),
            )
            .unwrap(),
        ),
        slices,
    };
    let value = serde_json::to_value(&special).unwrap();
    assert_eq!(value["kind"], "stride");
    assert_eq!(serde_json::from_value::<FuzzCase>(value).unwrap(), special);
    let built = special.build().unwrap();
    let output = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
    let Storage::F32(values) = output.storage() else {
        panic!("stride fixture must remain F32");
    };
    assert_eq!(
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![
            f32::NEG_INFINITY.to_bits(),
            3.0f32.to_bits(),
            0x8000_0000,
            f32::INFINITY.to_bits(),
        ]
    );
    let artifact = FuzzFailureArtifact::new(
        23,
        37,
        special.clone(),
        FuzzPath::NativeVector,
        FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&output),
        FuzzOutcome::Error {
            class: "actual".into(),
            detail: "synthetic stride mismatch".into(),
        },
    )
    .unwrap();
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(),
        artifact
    );
    let zeroed = minimize_case(&special, |candidate| {
        matches!(candidate, FuzzCase::Stride { input, slices }
            if input.shape == vec![2, 4]
                && input.bytes == vec![0; 32]
                && slices.len() == 2)
    });
    assert!(matches!(zeroed, FuzzCase::Stride { ref input, ref slices }
        if input.bytes == vec![0; 32] && slices.len() == 2));

    let malformed_rank = FuzzCase::Stride {
        input: FuzzTensor::from_tensor(&TensorData::zeros_with_dtype([2, 3], DType::F32).unwrap()),
        slices: vec![FuzzSlice {
            start: None,
            stop: None,
            step: 1,
        }],
    };
    assert!(malformed_rank.validate().is_err());
    let malformed_step = FuzzCase::Stride {
        input: FuzzTensor::from_tensor(&TensorData::zeros_with_dtype([2], DType::F32).unwrap()),
        slices: vec![FuzzSlice {
            start: None,
            stop: None,
            step: 0,
        }],
    };
    assert!(malformed_step.validate().is_err());
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
            Op::Cast {
                dtype: DType::Bool,
                ..
            }
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
            assert!(
                nodes
                    .iter()
                    .any(|node| { matches!(node.kind(), UOpKind::GraphCompare(CompareOp::Ne)) })
            );
            assert!(!nodes.iter().any(|node| {
                matches!(node.kind(), UOpKind::GraphLogical(crate::LogicalOp::Not))
            }));
        };
        for item in &scheduled.items {
            assert_kernel(&item.kernel);
        }
        let captured =
            CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
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
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(),
        artifact
    );
    let zeroed = minimize_case(
        &logical_not,
        |candidate| matches!(candidate, FuzzCase::LogicalNot { input } if input.bytes == vec![0; 28]),
    );
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
        rhs: FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(
            Scalar::Bool(true),
            DType::Bool,
        )),
    };
    let value = serde_json::to_value(&logical).unwrap();
    assert_eq!(value["kind"], "logical");
    assert_eq!(
        serde_json::from_value::<FuzzCase>(value.clone()).unwrap(),
        logical
    );
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
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(
                Scalar::Bool(true),
                DType::Bool,
            )),
        ),
        (
            FuzzLogicalOp::Or,
            LogicalOp::Or,
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(
                Scalar::Bool(false),
                DType::Bool,
            )),
            FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(
                Scalar::Bool(true),
                DType::Bool,
            )),
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
            item.kernel.topological().unwrap().iter().any(
                |uop| matches!(uop.kind(), UOpKind::GraphLogical(actual) if *actual == graph_op),
            )
        }));
        let captured =
            CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        assert!(captured.items.iter().any(|item| {
            item.kernel.topological().unwrap().iter().any(
                |uop| matches!(uop.kind(), UOpKind::GraphLogical(actual) if *actual == graph_op),
            )
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
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(),
        artifact
    );
    let zeroed = minimize_case(&logical, |candidate| {
        matches!(candidate, FuzzCase::Logical { lhs, rhs, .. }
            if lhs.bytes == vec![0; 2] && rhs.bytes == vec![0])
    });
    assert!(matches!(zeroed, FuzzCase::Logical { ref lhs, ref rhs, .. }
        if lhs.bytes == vec![0; 2] && rhs.bytes == vec![0]));
    let scalarized = minimize_case(&logical, |_| true);
    assert!(
        matches!(scalarized, FuzzCase::Logical { ref lhs, ref rhs, .. }
        if lhs.shape.is_empty() && rhs.shape.is_empty())
    );
    scalarized.validate().unwrap();
}

#[test]
fn compare_cases_round_trip_minimize_and_capture_as_graph_compare() {
    let compare = FuzzCase::Compare {
        op: FuzzCompareOp::Ge,
        lhs: FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [2],
                Storage::F32(vec![
                    f32::from_bits(0x7fc0_0001),
                    f32::from_bits(0x8000_0000),
                ]),
            )
            .unwrap(),
        ),
        rhs: FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(Scalar::F(0.0), DType::F32)),
    };
    let value = serde_json::to_value(&compare).unwrap();
    assert_eq!(value["kind"], "compare");
    assert_eq!(
        serde_json::from_value::<FuzzCase>(value.clone()).unwrap(),
        compare
    );
    let mut unknown = value;
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    let legacy = regression_cases().remove(0);
    let legacy_value = serde_json::to_value(&legacy).unwrap();
    assert_eq!(legacy_value["kind"], "binary");
    assert_eq!(
        serde_json::from_value::<FuzzCase>(legacy_value).unwrap(),
        legacy
    );

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
            item.kernel.topological().unwrap().iter().any(
                |uop| matches!(uop.kind(), UOpKind::GraphCompare(actual) if *actual == graph_op),
            )
        }));
        let captured =
            CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
        assert_eq!(captured.items.len(), scheduled.items.len());
        assert!(captured.items.iter().any(|item| {
            item.kernel.topological().unwrap().iter().any(
                |uop| matches!(uop.kind(), UOpKind::GraphCompare(actual) if *actual == graph_op),
            )
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
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(),
        artifact
    );
    let zeroed = minimize_case(&compare, |candidate| {
        matches!(candidate, FuzzCase::Compare { lhs, rhs, .. }
            if lhs.bytes == vec![0; 8] && rhs.bytes == vec![0; 4])
    });
    assert!(matches!(zeroed, FuzzCase::Compare { ref lhs, ref rhs, .. }
        if lhs.bytes == vec![0; 8] && rhs.bytes == vec![0; 4]));
    let scalarized = minimize_case(&compare, |_| true);
    assert!(
        matches!(scalarized, FuzzCase::Compare { ref lhs, ref rhs, .. }
        if lhs.shape.is_empty() && rhs.shape.is_empty())
    );
    scalarized.validate().unwrap();
}

#[test]
fn raw_compare_dtype_matrix_retains_typed_ordering_broadcast_and_renderer_paths() {
    let dtypes = [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];
    let operations = [
        (FuzzCompareOp::Eq, CompareOp::Eq),
        (FuzzCompareOp::Ne, CompareOp::Ne),
        (FuzzCompareOp::Lt, CompareOp::Lt),
        (FuzzCompareOp::Le, CompareOp::Le),
        (FuzzCompareOp::Gt, CompareOp::Gt),
        (FuzzCompareOp::Ge, CompareOp::Ge),
    ];

    for dtype in dtypes {
        let lane_values = |right| -> Vec<Scalar> {
            match dtype {
                DType::Bool => (if right {
                    [true, false, true]
                } else {
                    [false, true, false]
                })
                .into_iter()
                .cycle()
                .take(if right { 3 } else { 6 })
                .map(Scalar::Bool)
                .collect(),
                DType::I8 | DType::I16 | DType::I32 => {
                    (if right { [0, -1, 1] } else { [-1, 0, 1] })
                        .into_iter()
                        .cycle()
                        .take(if right { 3 } else { 6 })
                        .map(Scalar::I)
                        .collect()
                }
                DType::U8 | DType::U16 | DType::U32 => {
                    (if right { [1_u64, 0, 2] } else { [0, 1, 2] })
                        .into_iter()
                        .cycle()
                        .take(if right { 3 } else { 6 })
                        .map(Scalar::U)
                        .collect()
                }
                DType::I64 => (if right {
                    [-(1_i64 << 53), -((1_i64 << 53) + 1), i64::MIN]
                } else {
                    [-((1_i64 << 53) + 1), -(1_i64 << 53), i64::MAX]
                })
                .into_iter()
                .cycle()
                .take(if right { 3 } else { 6 })
                .map(Scalar::I)
                .collect(),
                DType::U64 => (if right {
                    [1_u64 << 53, (1_u64 << 53) + 1, 0]
                } else {
                    [(1_u64 << 53) + 1, 1_u64 << 53, u64::MAX]
                })
                .into_iter()
                .cycle()
                .take(if right { 3 } else { 6 })
                .map(Scalar::U)
                .collect(),
                DType::F16 | DType::BF16 | DType::F32 | DType::F64 => (if right {
                    [f64::NAN, 0.0, f64::NEG_INFINITY]
                } else {
                    [f64::NAN, -0.0, f64::INFINITY]
                })
                .into_iter()
                .cycle()
                .take(if right { 3 } else { 6 })
                .map(Scalar::F)
                .collect(),
                _ => unreachable!("float8 comparison fuzz is not generated"),
            }
        };
        let lhs = FuzzTensor::from_tensor(
            &TensorData::from_scalars([2, 1, 3], dtype, lane_values(false)).unwrap(),
        );
        let rhs = FuzzTensor::from_tensor(
            &TensorData::from_scalars([1, 3], dtype, lane_values(true)).unwrap(),
        );

        for (op, graph_op) in operations {
            let case = FuzzCase::Compare {
                op,
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            };
            let value = serde_json::to_value(&case).unwrap();
            assert_eq!(value["kind"], "compare");
            assert_eq!(serde_json::from_value::<FuzzCase>(value).unwrap(), case);
            let built = case.build().unwrap();
            assert_eq!(built.graph.dtype(built.output).unwrap(), DType::Bool);
            assert_eq!(
                built.graph.shape(built.output).unwrap(),
                &crate::Shape::from([2, 1, 3])
            );
            assert!(
                matches!(built.graph.op(built.output).unwrap(), Op::Compare { op: actual, .. } if *actual == graph_op)
            );
            let oracle = CpuBackend
                .execute(&built.graph, built.output, &built.oracle)
                .unwrap();
            assert_eq!(
                crate::execute_elementwise(&built.graph, built.output, &built.oracle)
                    .unwrap()
                    .storage(),
                oracle.storage(),
                "captured {dtype:?} {op:?}",
            );
            let scheduled = schedule(&built.graph, built.output).unwrap();
            assert!(scheduled.items[0].kernel.topological().unwrap().iter().any(
                |node| matches!(node.kind(), UOpKind::GraphCompare(actual) if *actual == graph_op)
            ));
            let scalar = CpuJit::render(&scheduled.items[0].kernel).unwrap();
            let vector = CpuJit::render_vectorized(&scheduled.items[0].kernel).unwrap();
            if matches!(dtype, DType::F16 | DType::BF16) {
                assert!(!vector.source.contains("B2 VectorProgram"), "{dtype:?}");
                assert!(scalar.source.contains("rg_f"), "{dtype:?}");
            } else if matches!(dtype, DType::F32 | DType::I32) {
                assert_eq!(vector.abi.buffers.last().unwrap().dtype, DType::Bool);
            }
            let captured =
                CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
            let bytes = captured.to_bytes().unwrap();
            assert_eq!(
                CapturedSchedule::from_bytes(&bytes)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
        }
    }

    let artifact_case = FuzzCase::Compare {
        op: FuzzCompareOp::Gt,
        lhs: FuzzTensor::from_tensor(
            &TensorData::from_scalars(
                [2],
                DType::U64,
                [Scalar::U((1_u64 << 53) + 1), Scalar::U(0)],
            )
            .unwrap(),
        ),
        rhs: FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(
            Scalar::U(1_u64 << 53),
            DType::U64,
        )),
    };
    let built = artifact_case.build().unwrap();
    let expected = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
    let artifact = FuzzFailureArtifact::new(
        19,
        31,
        artifact_case.clone(),
        FuzzPath::NativeVector,
        FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&expected),
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic wide comparison mismatch".into(),
        },
    )
    .unwrap();
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(),
        artifact
    );
    let zeroed = minimize_case(&artifact_case, |candidate| {
        matches!(candidate, FuzzCase::Compare { lhs, rhs, op: FuzzCompareOp::Gt }
            if lhs.bytes == vec![0; 16] && rhs.bytes == vec![0; 8])
    });
    assert!(
        matches!(zeroed, FuzzCase::Compare { ref lhs, ref rhs, op: FuzzCompareOp::Gt }
        if lhs.dtype == DType::U64 && rhs.dtype == DType::U64
            && lhs.shape == vec![2] && rhs.shape.is_empty())
    );
}

#[test]
fn unary_cases_round_trip_minimize_and_build_as_direct_graph_unaries() {
    let unary = FuzzCase::Unary {
        op: FuzzUnaryOp::Abs,
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage(
                [2],
                Storage::F32(vec![
                    f32::from_bits(0x8000_0000),
                    f32::from_bits(0x7fc0_0001),
                ]),
            )
            .unwrap(),
        ),
    };
    let value = serde_json::to_value(&unary).unwrap();
    assert_eq!(value["kind"], "unary");
    assert_eq!(
        serde_json::from_value::<FuzzCase>(value.clone()).unwrap(),
        unary
    );
    let mut unknown = value;
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<FuzzCase>(unknown).is_err());

    let legacy = regression_cases().remove(0);
    let legacy_value = serde_json::to_value(&legacy).unwrap();
    assert_eq!(legacy_value["kind"], "binary");
    assert_eq!(
        serde_json::from_value::<FuzzCase>(legacy_value).unwrap(),
        legacy
    );

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
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact.to_bytes().unwrap()).unwrap(),
        artifact
    );

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
fn unary_cases_cover_every_concrete_dtype_and_public_bool_negation() {
    let dtypes = [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];
    for dtype in dtypes {
        for op in [FuzzUnaryOp::Neg, FuzzUnaryOp::Abs] {
            let source =
                TensorData::from_scalars([2], dtype, [Scalar::I(0), Scalar::I(1)]).unwrap();
            let case = FuzzCase::Unary {
                op,
                input: FuzzTensor::from_tensor(&source),
            };
            let encoded = serde_json::to_value(&case).unwrap();
            assert_eq!(serde_json::from_value::<FuzzCase>(encoded).unwrap(), case);
            let built = case.build().unwrap();
            assert_eq!(built.graph.dtype(built.output).unwrap(), dtype);
            assert_eq!(built.graph.shape(built.output).unwrap().dims(), &[2]);
            if dtype == DType::Bool && op == FuzzUnaryOp::Neg {
                assert!(matches!(
                    built.graph.op(built.output).unwrap(),
                    Op::Compare {
                        op: CompareOp::Ne,
                        ..
                    }
                ));
                assert!((0..built.graph.node_count()).any(|index| {
                    matches!(
                        built.graph.op(crate::NodeId(index)).unwrap(),
                        Op::Cast {
                            dtype: DType::Bool,
                            ..
                        }
                    )
                }));
            } else {
                let expected = match op {
                    FuzzUnaryOp::Neg => UnaryOp::Neg,
                    FuzzUnaryOp::Abs => UnaryOp::Abs,
                    _ => unreachable!("loop contains only Neg and Abs"),
                };
                assert!(matches!(
                    built.graph.op(built.output).unwrap(),
                    Op::Unary { op, .. } if *op == expected
                ));
            }

            let scheduled = schedule(&built.graph, built.output).unwrap();
            assert_eq!(scheduled.items.len(), 1);
            let kernel = &scheduled.items[0].kernel;
            let topological = kernel.topological().unwrap();
            if dtype == DType::Bool && op == FuzzUnaryOp::Neg {
                assert!(
                    topological
                        .iter()
                        .any(|node| matches!(node.kind(), UOpKind::Cast))
                );
                assert!(
                    topological
                        .iter()
                        .any(|node| matches!(node.kind(), UOpKind::GraphCompare(CompareOp::Ne)))
                );
            } else {
                let expected = match op {
                    FuzzUnaryOp::Neg => UnaryOp::Neg,
                    FuzzUnaryOp::Abs => UnaryOp::Abs,
                    _ => unreachable!("loop contains only Neg and Abs"),
                };
                assert!(topological.iter().any(|node| {
                    matches!(node.kind(), UOpKind::GraphUnary(actual) if *actual == expected)
                }));
            }
            assert!(CpuJit::render(kernel).is_ok(), "{op:?} {dtype:?}");
            let vector = CpuJit::render_vectorized(kernel).unwrap();
            if matches!(dtype, DType::F16 | DType::BF16) {
                assert!(
                    !vector.source.contains("B2 VectorProgram"),
                    "{op:?} {dtype:?}"
                );
            } else if matches!(dtype, DType::F32 | DType::I32) {
                assert!(
                    vector.source.contains("B2 VectorProgram"),
                    "{op:?} {dtype:?}"
                );
            }
            let captured =
                CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
            let capture_bytes = captured.to_bytes().unwrap();
            assert_eq!(
                CapturedSchedule::from_bytes(&capture_bytes)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                capture_bytes
            );
            let output = CpuBackend
                .execute(&built.graph, built.output, &built.oracle)
                .unwrap();
            assert_eq!(output.dtype(), dtype);
            assert_eq!(output.shape().dims(), &[2]);
            if op == FuzzUnaryOp::Abs
                && (dtype == DType::Bool
                    || matches!(dtype.category(), crate::DTypeCategory::Unsigned))
            {
                assert_eq!(output, source);
            }
        }
    }

    // Exact storage edge cases are deliberately separate from generator
    // values: min lanes must wrap, U64 must not take an f64 detour, and full
    // float lanes preserve the observable IEEE behavior. Half NaN payload
    // identity remains outside this assertion because decode/re-encode is the
    // established storage boundary.
    let edges = [
        (
            FuzzUnaryOp::Abs,
            TensorData::from_storage([1], Storage::I8(vec![i8::MIN])).unwrap(),
        ),
        (
            FuzzUnaryOp::Abs,
            TensorData::from_storage([1], Storage::I16(vec![i16::MIN])).unwrap(),
        ),
        (
            FuzzUnaryOp::Abs,
            TensorData::from_storage([1], Storage::I32(vec![i32::MIN])).unwrap(),
        ),
        (
            FuzzUnaryOp::Abs,
            TensorData::from_storage([1], Storage::I64(vec![i64::MIN])).unwrap(),
        ),
        (
            FuzzUnaryOp::Neg,
            TensorData::from_storage([1], Storage::U64(vec![(1u64 << 53) + 1])).unwrap(),
        ),
        (
            FuzzUnaryOp::Neg,
            TensorData::from_storage([3], Storage::F32(vec![-0.0, f32::NAN, f32::INFINITY]))
                .unwrap(),
        ),
        (
            FuzzUnaryOp::Abs,
            TensorData::from_storage([3], Storage::F64(vec![-0.0, f64::NAN, f64::NEG_INFINITY]))
                .unwrap(),
        ),
        (
            FuzzUnaryOp::Abs,
            TensorData::from_storage([3], Storage::F16(vec![0x8000, 0x7e01, 0x7c00])).unwrap(),
        ),
        (
            FuzzUnaryOp::Abs,
            TensorData::from_storage([3], Storage::BF16(vec![0x8000, 0x7fc1, 0x7f80])).unwrap(),
        ),
    ];
    for (op, input) in edges {
        let case = FuzzCase::Unary {
            op,
            input: FuzzTensor::from_tensor(&input),
        };
        let built = case.build().unwrap();
        let output = CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .unwrap();
        assert_eq!(output.dtype(), input.dtype());
        assert_eq!(output.shape(), input.shape());
        match (op, input.dtype()) {
            (FuzzUnaryOp::Abs, DType::I8 | DType::I16 | DType::I32 | DType::I64) => {
                assert_eq!(output, input);
            }
            (FuzzUnaryOp::Neg, DType::U64) => {
                assert_eq!(output.scalar_at(0).as_u64(), u64::MAX - (1u64 << 53));
            }
            (FuzzUnaryOp::Neg, DType::F32) => {
                assert!(output.scalar_at(0).as_f64().is_sign_positive());
                assert!(output.scalar_at(1).as_f64().is_nan());
                assert!(output.scalar_at(2).as_f64().is_infinite());
            }
            (FuzzUnaryOp::Abs, DType::F64) => {
                assert!(output.scalar_at(0).as_f64().is_sign_positive());
                assert!(output.scalar_at(1).as_f64().is_nan());
                assert!(output.scalar_at(2).as_f64().is_infinite());
            }
            (FuzzUnaryOp::Abs, DType::F16 | DType::BF16) => {
                assert!(output.scalar_at(0).as_f64().is_sign_positive());
                assert!(output.scalar_at(1).as_f64().is_nan());
                assert!(output.scalar_at(2).as_f64().is_infinite());
            }
            _ => unreachable!("constructed exact unary edge"),
        }
    }

    let bool_neg = FuzzCase::Unary {
        op: FuzzUnaryOp::Neg,
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage([2], Storage::Bool(vec![false, true])).unwrap(),
        ),
    };
    let built = bool_neg.build().unwrap();
    let output = CpuBackend
        .execute(&built.graph, built.output, &built.oracle)
        .unwrap();
    assert_eq!(
        output,
        TensorData::from_storage([2], Storage::Bool(vec![true, false])).unwrap()
    );
    let artifact = FuzzFailureArtifact::new(
        41,
        43,
        bool_neg.clone(),
        FuzzPath::NativeScalar,
        FuzzComparisonPolicy::ExactBytes,
        FuzzOutcome::value(&output),
        FuzzOutcome::Error {
            class: "execute".into(),
            detail: "synthetic Bool unary mismatch".into(),
        },
    )
    .unwrap();
    let artifact_bytes = artifact.to_bytes().unwrap();
    assert_eq!(
        FuzzFailureArtifact::from_bytes(&artifact_bytes).unwrap(),
        artifact
    );
    let minimized = minimize_case(&bool_neg, |candidate| {
        matches!(candidate, FuzzCase::Unary { op: FuzzUnaryOp::Neg, input }
            if input.dtype == DType::Bool && input.shape == vec![2] && input.bytes == vec![0, 0])
    });
    assert!(
        matches!(minimized, FuzzCase::Unary { op: FuzzUnaryOp::Neg, ref input }
        if input.dtype == DType::Bool && input.shape == vec![2] && input.bytes == vec![0, 0])
    );
}

#[test]
fn portable_float_unaries_retain_graph_capture_and_native_contracts() {
    let operations = [
        (FuzzUnaryOp::Exp, UnaryOp::Exp, "exp("),
        (FuzzUnaryOp::Exp2, UnaryOp::Exp2, "exp2("),
        (FuzzUnaryOp::Reciprocal, UnaryOp::Reciprocal, "1.0/("),
        (FuzzUnaryOp::Sqrt, UnaryOp::Sqrt, "sqrt("),
        (FuzzUnaryOp::Rsqrt, UnaryOp::Rsqrt, "1.0/sqrt("),
        (FuzzUnaryOp::Log2, UnaryOp::Log2, "log2("),
        (FuzzUnaryOp::Sin, UnaryOp::Sin, "sin("),
        (FuzzUnaryOp::Cos, UnaryOp::Cos, "cos("),
        (FuzzUnaryOp::Tan, UnaryOp::Tan, "tan("),
        (FuzzUnaryOp::Log, UnaryOp::Log, "log("),
        (FuzzUnaryOp::Sinh, UnaryOp::Sinh, "sinh("),
        (FuzzUnaryOp::Cosh, UnaryOp::Cosh, "cosh("),
        (FuzzUnaryOp::Tanh, UnaryOp::Tanh, "tanh("),
        (FuzzUnaryOp::Erf, UnaryOp::Erf, "rg_erf("),
        (FuzzUnaryOp::Erfc, UnaryOp::Erfc, "1.0-rg_erf("),
        (FuzzUnaryOp::Asin, UnaryOp::Asin, "asin("),
        (FuzzUnaryOp::Acos, UnaryOp::Acos, "acos("),
        (FuzzUnaryOp::Atan, UnaryOp::Atan, "atan("),
        (FuzzUnaryOp::Asinh, UnaryOp::Asinh, "asinh("),
        (FuzzUnaryOp::Acosh, UnaryOp::Acosh, "acosh("),
        (FuzzUnaryOp::Atanh, UnaryOp::Atanh, "atanh("),
    ];

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        for (fuzz_op, graph_op, source_token) in operations {
            let input = TensorData::from_scalars(
                [3],
                dtype,
                [Scalar::F(0.25), Scalar::F(1.0), Scalar::F(2.0)],
            )
            .unwrap();
            let case = FuzzCase::Unary {
                op: fuzz_op,
                input: FuzzTensor::from_tensor(&input),
            };
            let encoded = serde_json::to_vec(&case).unwrap();
            assert_eq!(serde_json::from_slice::<FuzzCase>(&encoded).unwrap(), case);

            let built = case.build().unwrap();
            assert_eq!(built.graph.dtype(built.output).unwrap(), dtype);
            assert_eq!(built.graph.shape(built.output).unwrap().dims(), &[3]);
            assert!(matches!(
                built.graph.op(built.output).unwrap(),
                Op::Unary { op, .. } if *op == graph_op
            ));

            let oracle = CpuBackend
                .execute(&built.graph, built.output, &built.oracle)
                .unwrap();
            let interpreted =
                crate::execute_elementwise(&built.graph, built.output, &built.oracle).unwrap();
            assert_eq!(
                FuzzTensor::from_tensor(&interpreted),
                FuzzTensor::from_tensor(&oracle),
                "captured {dtype:?} {fuzz_op:?}"
            );

            let scheduled = schedule(&built.graph, built.output).unwrap();
            assert_eq!(scheduled.items.len(), 1);
            assert!(scheduled.items[0].kernel.topological().unwrap().iter().any(
                |node| matches!(node.kind(), UOpKind::GraphUnary(actual) if *actual == graph_op)
            ));
            let scalar = CpuJit::render(&scheduled.items[0].kernel).unwrap();
            assert!(
                scalar.source.contains(source_token),
                "{dtype:?} {fuzz_op:?}"
            );
            let vector = CpuJit::render_vectorized(&scheduled.items[0].kernel).unwrap();
            assert!(
                vector.source.contains(source_token),
                "{dtype:?} {fuzz_op:?}"
            );
            assert!(
                !vector.source.contains("B2 VectorProgram"),
                "portable B2 does not yet admit {dtype:?} {fuzz_op:?}"
            );

            let captured =
                CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
            let bytes = captured.to_bytes().unwrap();
            assert_eq!(
                CapturedSchedule::from_bytes(&bytes)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
        }
    }

    let log2 = FuzzCase::Unary {
        op: FuzzUnaryOp::Log2,
        input: FuzzTensor::from_tensor(
            &TensorData::from_storage([2], Storage::F32(vec![0.5, 2.0])).unwrap(),
        ),
    };
    let minimized = minimize_case(&log2, |candidate| {
        matches!(candidate, FuzzCase::Unary { op: FuzzUnaryOp::Log2, input }
            if input.dtype == DType::F32 && input.shape == vec![2] && input.bytes == vec![0; 8])
    });
    assert!(matches!(
        minimized,
        FuzzCase::Unary {
            op: FuzzUnaryOp::Log2,
            ref input,
        } if input.dtype == DType::F32 && input.shape == vec![2] && input.bytes == vec![0; 8]
    ));
}

#[test]
fn portable_storage_unaries_cover_every_concrete_dtype_and_native_fallback() {
    let dtypes = [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];
    let operations = [
        (FuzzUnaryOp::Relu, UnaryOp::Relu),
        (FuzzUnaryOp::Step, UnaryOp::Step),
        (FuzzUnaryOp::Square, UnaryOp::Square),
        (FuzzUnaryOp::Floor, UnaryOp::Floor),
        (FuzzUnaryOp::Ceil, UnaryOp::Ceil),
        (FuzzUnaryOp::Trunc, UnaryOp::Trunc),
        (FuzzUnaryOp::Round, UnaryOp::Round),
        (FuzzUnaryOp::Sign, UnaryOp::Sign),
        (FuzzUnaryOp::IsNan, UnaryOp::IsNan),
        (FuzzUnaryOp::IsInf, UnaryOp::IsInf),
        (FuzzUnaryOp::IsFinite, UnaryOp::IsFinite),
    ];

    for dtype in dtypes {
        for (fuzz_op, graph_op) in operations {
            let input =
                TensorData::from_scalars([3], dtype, [Scalar::I(-2), Scalar::I(0), Scalar::I(3)])
                    .unwrap();
            let case = FuzzCase::Unary {
                op: fuzz_op,
                input: FuzzTensor::from_tensor(&input),
            };
            let built = case.build().unwrap();
            let predicate = matches!(
                fuzz_op,
                FuzzUnaryOp::IsNan | FuzzUnaryOp::IsInf | FuzzUnaryOp::IsFinite
            );
            assert_eq!(
                built.graph.dtype(built.output).unwrap(),
                if predicate { DType::Bool } else { dtype },
                "{dtype:?} {fuzz_op:?}"
            );
            assert!(matches!(
                built.graph.op(built.output).unwrap(),
                Op::Unary { op, .. } if *op == graph_op
            ));

            let oracle = CpuBackend
                .execute(&built.graph, built.output, &built.oracle)
                .unwrap();
            let interpreted =
                crate::execute_elementwise(&built.graph, built.output, &built.oracle).unwrap();
            assert_eq!(interpreted, oracle, "captured {dtype:?} {fuzz_op:?}");

            let scheduled = schedule(&built.graph, built.output).unwrap();
            assert_eq!(scheduled.items.len(), 1);
            assert!(scheduled.items[0].kernel.topological().unwrap().iter().any(
                |node| matches!(node.kind(), UOpKind::GraphUnary(actual) if *actual == graph_op)
            ));
            let scalar = CpuJit::render(&scheduled.items[0].kernel).unwrap();
            let vector = CpuJit::render_vectorized(&scheduled.items[0].kernel).unwrap();
            assert!(scalar.source.contains(crate::cpu_jit::RENDERER_VERSION));
            assert!(vector.source.contains(crate::cpu_jit::RENDERER_VERSION));
            assert!(
                !vector.source.contains("B2 VectorProgram"),
                "storage unary remains on the exact scalar-per-lane path: {dtype:?} {fuzz_op:?}"
            );
            if fuzz_op == FuzzUnaryOp::Square && dtype.is_integer() {
                assert!(scalar.source.contains("(uint64_t)"), "{dtype:?}");
            }
            if fuzz_op == FuzzUnaryOp::Round && dtype.is_float() {
                assert!(scalar.source.contains("rg_round_ties_even("), "{dtype:?}");
            }
            if fuzz_op == FuzzUnaryOp::IsNan && dtype.is_float() {
                assert!(scalar.source.contains("isnan("), "{dtype:?}");
            }
            if fuzz_op == FuzzUnaryOp::IsInf && dtype.is_float() {
                assert!(scalar.source.contains("isinf("), "{dtype:?}");
            }

            let captured =
                CapturedSchedule::capture(&built.graph, &scheduled, &[built.output]).unwrap();
            let bytes = captured.to_bytes().unwrap();
            assert_eq!(
                CapturedSchedule::from_bytes(&bytes)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
        }
    }
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
fn regression_native_cases_remain_explicit_and_portable() {
    let mut unsupported = 0;
    let mut native_matches = 0;
    for (index, case) in regression_cases().iter().enumerate() {
        for comparison in run_case(3, index as u64, case, true).unwrap() {
            match comparison {
                FuzzComparison::Match {
                    path: FuzzPath::NativeScalar | FuzzPath::NativeVector,
                    ..
                } => native_matches += 1,
                FuzzComparison::Unsupported {
                    path: FuzzPath::NativeScalar | FuzzPath::NativeVector,
                    ..
                } => unsupported += 1,
                FuzzComparison::Failure(failure) => {
                    panic!("regression native failure: {failure:?}")
                }
                _ => {}
            }
        }
    }
    assert!(native_matches > 0);
    assert_eq!(unsupported, 0);
}

#[test]
fn regression_cases_cover_edges_without_current_failures() {
    let cases = regression_cases();
    assert_eq!(cases.len(), 95);
    for (index, case) in cases.iter().enumerate() {
        for comparison in run_case(0xfeed, index as u64, case, false).unwrap() {
            assert!(
                matches!(
                    comparison,
                    FuzzComparison::Match {
                        path: FuzzPath::CapturedInterpreter,
                        ..
                    }
                ),
                "regression case {index}: {comparison:?}"
            );
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
