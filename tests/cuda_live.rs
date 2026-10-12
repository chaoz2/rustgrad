//! Opt-in native CUDA acceptance; no mock dispatch or CPU fallback.

use rustgrad::{
    Backend, CapturedSchedule, ConcurrentPtxCache, CpuBackend, DType, DeviceId, Driver, Graph,
    MatmulValue, NodeId, Operation, PreparedCudaGraphPrefix, PtxError, PtxRenderer, Scalar,
    TensorData, schedule,
};
use std::collections::{BTreeMap, HashMap};

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn prepare(
    graph: &Graph,
    output: NodeId,
) -> Result<rustgrad::runtime::cuda_graph::CapturedCudaGraphPrefix> {
    let driver = Driver::load()?;
    let ordinal = std::env::var("RUSTGRAD_CUDA_DEVICE")
        .unwrap_or_else(|_| "0".into())
        .parse()?;
    let device = driver.device(DeviceId(ordinal))?;
    let capability = device.capability()?;
    eprintln!("native CUDA: {} sm_{}", capability.name, capability.sm());
    let scheduled = schedule(graph, output)?;
    let capture = CapturedSchedule::capture(graph, &scheduled, &[output])?;
    let renderer = PtxRenderer::new(capability.sm())?;
    if std::env::var_os("RUSTGRAD_CUDA_DUMP_PTX").is_some() {
        for item in &capture.items {
            eprintln!("{}", renderer.render(&item.kernel)?.source);
        }
    }
    Ok(PreparedCudaGraphPrefix::prepare_capture(
        device.retain_primary_context()?,
        &capture,
        renderer,
        &ConcurrentPtxCache::new(),
    )?)
}

fn assert_close(case: &str, actual: &TensorData, expected: &TensorData) {
    assert_eq!(actual.shape(), expected.shape(), "{case}: shape");
    assert_eq!(actual.dtype(), expected.dtype(), "{case}: dtype");
    for (lane, (actual, expected)) in actual
        .to_vec_f64()
        .into_iter()
        .zip(expected.to_vec_f64())
        .enumerate()
    {
        // Native GPU contractions may round differently from the CPU oracle.
        // These small finite F32 fixtures use dyadic inputs. This is a bounded
        // numerical check, not a broad bitwise-parity or performance claim.
        let tolerance = 1.0e-5 * expected.abs().max(1.0);
        assert!(
            actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= tolerance,
            "{case} lane {lane}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn check_replay(
    case: &str,
    graph: &Graph,
    output: NodeId,
    prepared: &mut rustgrad::runtime::cuda_graph::CapturedCudaGraphPrefix,
    inputs: BTreeMap<String, TensorData>,
) -> Result {
    let oracle_inputs: HashMap<_, _> = inputs.clone().into_iter().collect();
    let expected = CpuBackend.execute(graph, output, &oracle_inputs)?;
    eprintln!("CUDA replay: {case}");
    let actual = prepared.execute(&inputs)?;
    assert_eq!(actual.len(), 1, "{case}: output count");
    assert_close(case, &actual[0], &expected);
    Ok(())
}

#[test]
#[ignore = "requires a live NVIDIA GPU and CUDA driver"]
fn live_cuda_elementwise_reduction_replay_and_rejected_input_retry() -> Result {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 3], DType::F32);
    let squared = graph.square(x)?;
    let bias = graph.constant(TensorData::new([3], vec![1.0_f32, -2.0, 0.5])?);
    let shifted = graph.add(squared, bias)?;
    let output = graph.sum(shifted, 1)?;
    let mut prepared = prepare(&graph, output)?;
    for (case, values) in [
        ("first replay", vec![1.0_f32, -2.0, 3.0, 0.0, 0.5, -1.0]),
        ("changed input", vec![-3.0, 2.0, 0.0, 4.0, -0.5, 1.0]),
        ("repeat first input", vec![1.0, -2.0, 3.0, 0.0, 0.5, -1.0]),
    ] {
        check_replay(
            case,
            &graph,
            output,
            &mut prepared,
            BTreeMap::from([("x".into(), TensorData::new([2, 3], values)?)]),
        )?;
    }
    let malformed = BTreeMap::from([("x".into(), TensorData::new([3], vec![0.0_f32; 3])?)]);
    assert!(matches!(
        prepared.execute(&malformed),
        Err(PtxError::InvalidBinding(_))
    ));
    check_replay(
        "retry after rejected shape",
        &graph,
        output,
        &mut prepared,
        BTreeMap::from([("x".into(), TensorData::new([2, 3], vec![2.0_f32; 6])?)]),
    )
}

#[test]
#[ignore = "requires a live NVIDIA GPU and CUDA driver"]
fn live_cuda_matmul_serial_tails_and_empty_domains() -> Result {
    for (case, lhs_shape, rhs_shape) in [
        ("vector matrix", vec![3], vec![3, 2]),
        ("matrix tails", vec![9, 7], vec![7, 11]),
        ("broadcast batch", vec![2, 1, 3, 4], vec![1, 3, 4, 2]),
        ("zero contraction", vec![2, 0], vec![0, 2]),
        ("zero output", vec![0, 3], vec![3, 2]),
    ] {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", lhs_shape.clone(), DType::F32);
        let rhs = graph.input_dtype("rhs", rhs_shape.clone(), DType::F32);
        let output = graph.matmul(lhs, rhs)?;
        let mut prepared = prepare(&graph, output)?;
        for phase in [0, 1, 0] {
            let tensor = |shape: Vec<usize>, offset: i32| {
                let count = shape.iter().product::<usize>();
                TensorData::from_scalars(
                    shape,
                    DType::F32,
                    (0..count).map(|i| Scalar::F(((i % 7) as i32 - 3 + offset) as f64 * 0.25)),
                )
            };
            check_replay(
                case,
                &graph,
                output,
                &mut prepared,
                BTreeMap::from([
                    ("lhs".into(), tensor(lhs_shape.clone(), phase)?),
                    ("rhs".into(), tensor(rhs_shape.clone(), -phase)?),
                ]),
            )?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an sm_80+ NVIDIA GPU and CUDA driver"]
fn live_cuda_tensor_core_f16_bf16_exact_fixtures() -> Result {
    for dtype in [DType::F16, DType::BF16] {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 16, 32], dtype);
        let rhs = graph.input_dtype("rhs", [1, 32, 16], dtype);
        let output = graph.matmul(lhs, rhs)?;
        let scheduled = schedule(&graph, output)?;
        assert!(matches!(
            scheduled.items[0].kernel.operation(),
            Operation::Matmul(MatmulValue::TensorCore(_))
        ));
        let mut prepared = prepare(&graph, output)?;
        for phase in [0, 1, 0] {
            // Small integral products and sums are exactly representable in
            // both narrow formats; exact storage equality is justified only
            // for this fixture, not arbitrary tensor-core contractions.
            let lhs = TensorData::from_scalars(
                [2, 16, 32],
                dtype,
                (0..1024).map(|i| Scalar::F((i % 3 - 1 + phase) as f64)),
            )?;
            let rhs = TensorData::from_scalars(
                [1, 32, 16],
                dtype,
                (0..512).map(|i| Scalar::F((i % 3 - 1) as f64)),
            )?;
            let inputs = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
            let oracle_inputs: HashMap<_, _> = inputs.clone().into_iter().collect();
            let expected = CpuBackend.execute(&graph, output, &oracle_inputs)?;
            let actual = prepared.execute(&inputs)?;
            assert_eq!(actual.len(), 1);
            assert_eq!(actual[0].shape(), expected.shape());
            assert_eq!(actual[0].dtype(), dtype);
            assert_eq!(
                actual[0].to_le_bytes()?,
                expected.to_le_bytes()?,
                "{dtype:?} replay {phase}"
            );
        }
    }
    Ok(())
}
