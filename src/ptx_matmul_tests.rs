//! Primary-context acceptance tests for static Matmul PTX lowering.

use super::{ConcurrentPtxCache, PtxBinding, PtxError, PtxRenderer};
use crate::{
    Backend, CapturedSchedule, CpuBackend, CudaError, DType, Driver, Graph, MatmulKernelPlan,
    Scalar, TensorData, UArg, UOpKind,
};
use std::{collections::HashMap, num::NonZeroUsize, sync::Arc};

fn primary(mock: &Arc<crate::cuda::tests::Mock>) -> crate::PrimaryContext {
    Driver::from_dispatch(mock.clone())
        .unwrap()
        .device(crate::DeviceId(0))
        .unwrap()
        .retain_primary_context()
        .unwrap()
}

fn tensor(shape: Vec<usize>, dtype: DType, values: &[f64]) -> TensorData {
    TensorData::from_scalars(shape, dtype, values.iter().copied().map(Scalar::F)).unwrap()
}

fn lease(primary: &crate::PrimaryContext, bytes: &[u8]) -> crate::PrimaryBufferLease {
    let result = primary
        .allocator()
        .allocate(NonZeroUsize::new(bytes.len().max(1)).unwrap())
        .unwrap();
    if !bytes.is_empty() {
        result.view().unwrap().copy_from(0, bytes).unwrap();
    }
    result
}

#[test]
fn matmul_primary_cache_launches_owner_scoped_mock_semantics() {
    struct Case {
        name: &'static str,
        dtype: DType,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
        lhs: Vec<f64>,
        rhs: Vec<f64>,
    }

    let cases = vec![
        Case {
            name: "f32 vector matrix",
            dtype: DType::F32,
            lhs_shape: vec![3],
            rhs_shape: vec![3, 2],
            lhs: vec![2.0, -1.0, 3.0],
            rhs: vec![1.0, 4.0, 2.0, -3.0, 5.0, 6.0],
        },
        Case {
            name: "f64 broadcast batch",
            dtype: DType::F64,
            lhs_shape: vec![2, 1, 2, 3],
            rhs_shape: vec![1, 4, 3, 2],
            lhs: (0..12).map(|x| x as f64 - 5.0).collect(),
            rhs: (0..24).map(|x| x as f64 * 0.25 - 2.0).collect(),
        },
        Case {
            name: "f32 zero k",
            dtype: DType::F32,
            lhs_shape: vec![2, 0],
            rhs_shape: vec![0, 2],
            lhs: vec![],
            rhs: vec![],
        },
        Case {
            name: "f32 tiled broadcast tails",
            dtype: DType::F32,
            lhs_shape: vec![2, 1, 9, 7],
            rhs_shape: vec![1, 3, 7, 11],
            lhs: (0..2 * 9 * 7)
                .map(|index| index as f64 * 0.125 - 4.0)
                .collect(),
            rhs: (0..3 * 7 * 11)
                .map(|index| index as f64 * -0.0625 + 2.0)
                .collect(),
        },
    ];

    let mock = Arc::new(crate::cuda::tests::Mock::default());
    let first = primary(&mock);
    // Distinct owners deliberately retain the mock's same raw primary context.
    let second = primary(&mock);
    assert_ne!(first.identity(), second.identity());
    let stream = first.stream().unwrap();
    let cache = ConcurrentPtxCache::new();
    let mut first_rendered = None;
    let mut first_kernel = None;
    let mut first_lhs = None;
    let mut first_rhs = None;
    let mut first_output = None;
    let mut tiled_rendered = None;
    let mut tiled_kernel = None;

    for case in cases {
        let mut graph = Graph::new();
        let lhs_node = graph.input_dtype("lhs", case.lhs_shape.clone(), case.dtype);
        let rhs_node = graph.input_dtype("rhs", case.rhs_shape.clone(), case.dtype);
        let output_node = graph.matmul(lhs_node, rhs_node).unwrap();
        let plan = MatmulKernelPlan::from_graph(&graph, output_node).unwrap();
        let schedule = crate::schedule(&graph, output_node).unwrap();
        let captured = CapturedSchedule::capture(&graph, &schedule, &[output_node]).unwrap();
        let artifact = CapturedSchedule::from_bytes(&captured.to_bytes().unwrap()).unwrap();
        let kernel = &artifact.items[0].kernel;
        let UOpKind::Matmul = kernel.kind() else {
            panic!("{} did not retain a matmul payload", case.name);
        };
        let shared_plan = kernel.arg().matmul_plan().unwrap();
        assert_eq!(shared_plan, &plan, "{} shared plan", case.name);
        if case.name == "f32 tiled broadcast tails" {
            let UArg::TiledMatmul(payload) = kernel.arg() else {
                panic!("eligible matrix case did not retain tiled payload");
            };
            assert!(payload.tile.tails.m && payload.tile.tails.n && payload.tile.tails.k);
            assert!(payload.tile.tails.broadcast_batch);
        }
        let lhs = tensor(case.lhs_shape, case.dtype, &case.lhs);
        let rhs = tensor(case.rhs_shape, case.dtype, &case.rhs);
        let plan_expected = plan.execute(&lhs, &rhs).unwrap();
        let cpu_expected = CpuBackend
            .execute(
                &graph,
                output_node,
                &HashMap::from([("lhs".into(), lhs.clone()), ("rhs".into(), rhs.clone())]),
            )
            .unwrap();
        assert_eq!(
            plan_expected.to_le_bytes().unwrap(),
            cpu_expected.to_le_bytes().unwrap(),
            "{} plan",
            case.name
        );

        let rendered = PtxRenderer::new(80).unwrap().render(kernel).unwrap();
        let lhs_lease = lease(&first, &lhs.to_le_bytes().unwrap());
        let rhs_lease = lease(&first, &rhs.to_le_bytes().unwrap());
        let output_lease = lease(
            &first,
            &vec![0xA5; plan_expected.to_le_bytes().unwrap().len()],
        );
        let kernel = cache.get_or_load(&first, rendered.clone(), 32).unwrap();
        kernel
            .launch(
                &stream,
                &[
                    PtxBinding {
                        buffer: lhs_lease.view().unwrap(),
                        dtype: case.dtype,
                        mutable: false,
                    },
                    PtxBinding {
                        buffer: rhs_lease.view().unwrap(),
                        dtype: case.dtype,
                        mutable: false,
                    },
                    PtxBinding {
                        buffer: output_lease.view().unwrap(),
                        dtype: case.dtype,
                        mutable: true,
                    },
                ],
                true,
            )
            .unwrap();
        let mut actual = vec![0; plan_expected.to_le_bytes().unwrap().len()];
        output_lease
            .view()
            .unwrap()
            .copy_to(0, &mut actual)
            .unwrap();
        assert_eq!(
            actual,
            plan_expected.to_le_bytes().unwrap(),
            "{} mock",
            case.name
        );

        if first_rendered.is_none() {
            first_rendered = Some(rendered.clone());
            first_kernel = Some(kernel.clone());
            first_lhs = Some(lhs_lease);
            first_rhs = Some(rhs_lease);
            first_output = Some(output_lease);
        }
        if case.name == "f32 tiled broadcast tails" {
            tiled_rendered = Some(rendered);
            tiled_kernel = Some(kernel);
        }
    }

    assert_eq!(mock.generic_kernel_count(), 4);
    let tiled_rendered = tiled_rendered.take().unwrap();
    let tiled_repeated = cache
        .get_or_load(&first, tiled_rendered.clone(), 256)
        .unwrap();
    assert!(Arc::ptr_eq(tiled_kernel.as_ref().unwrap(), &tiled_repeated));
    assert_eq!(cache.len(), 4);
    let mut malformed_launch = tiled_rendered;
    let crate::PtxLaunchGeometry::Exact(mut launch) = malformed_launch.launch else {
        panic!("tiled launch geometry is not exact");
    };
    launch.shared_bytes -= 4;
    malformed_launch.launch = crate::PtxLaunchGeometry::Exact(launch);
    assert!(matches!(
        cache.get_or_load(&first, malformed_launch, 32),
        Err(PtxError::InvalidBinding(_))
    ));
    assert_eq!(cache.len(), 4);
    let rendered = first_rendered.take().unwrap();
    let kernel = first_kernel.take().unwrap();
    let repeated = cache.get_or_load(&first, rendered.clone(), 32).unwrap();
    assert!(Arc::ptr_eq(&kernel, &repeated));
    assert_eq!(mock.generic_kernel_count(), 4);

    let second_kernel = cache.get_or_load(&second, rendered, 32).unwrap();
    assert!(!Arc::ptr_eq(&kernel, &second_kernel));
    assert_eq!(cache.len(), 5);
    assert_eq!(mock.generic_kernel_count(), 5);

    let lhs = first_lhs.take().unwrap();
    let rhs = first_rhs.take().unwrap();
    let output = first_output.take().unwrap();
    let sentinel = vec![0x5A; output.view().unwrap().len()];
    output.view().unwrap().copy_from(0, &sentinel).unwrap();
    assert!(matches!(
        kernel.launch(
            &stream,
            &[
                PtxBinding {
                    buffer: lhs.view().unwrap(),
                    dtype: DType::F64,
                    mutable: false
                },
                PtxBinding {
                    buffer: rhs.view().unwrap(),
                    dtype: DType::F32,
                    mutable: false
                },
                PtxBinding {
                    buffer: output.view().unwrap(),
                    dtype: DType::F32,
                    mutable: true
                },
            ],
            true,
        ),
        Err(PtxError::InvalidBinding(_))
    ));
    let mut unchanged = vec![0; sentinel.len()];
    output.view().unwrap().copy_to(0, &mut unchanged).unwrap();
    assert_eq!(unchanged, sentinel);

    let other_output = lease(&second, &sentinel);
    assert!(matches!(
        kernel.launch(
            &stream,
            &[
                PtxBinding {
                    buffer: lhs.view().unwrap(),
                    dtype: DType::F32,
                    mutable: false
                },
                PtxBinding {
                    buffer: rhs.view().unwrap(),
                    dtype: DType::F32,
                    mutable: false
                },
                PtxBinding {
                    buffer: other_output.view().unwrap(),
                    dtype: DType::F32,
                    mutable: true
                },
            ],
            true,
        ),
        Err(PtxError::Cuda(CudaError::ContextMismatch))
    ));
    let mut other_unchanged = vec![0; sentinel.len()];
    other_output
        .view()
        .unwrap()
        .copy_to(0, &mut other_unchanged)
        .unwrap();
    assert_eq!(other_unchanged, sentinel);

    drop(second_kernel);
    drop(tiled_repeated);
    drop(tiled_kernel);
    drop(repeated);
    drop(kernel);
    drop(cache);
    assert_eq!(mock.generic_kernel_count(), 0);
}

#[test]
fn tensor_core_primary_cache_uses_fragment_simulator_and_exact_launch() {
    let mock = Arc::new(crate::cuda::tests::Mock::default());
    let primary = primary(&mock);
    let stream = primary.stream().unwrap();
    let cache = ConcurrentPtxCache::new();
    for dtype in [DType::F16, DType::BF16] {
        let mut graph = Graph::new();
        let lhs_node = graph.input_dtype("lhs", [2, 16, 32], dtype);
        let rhs_node = graph.input_dtype("rhs", [1, 32, 16], dtype);
        let output_node = graph.matmul(lhs_node, rhs_node).unwrap();
        let schedule = crate::schedule(&graph, output_node).unwrap();
        let captured = CapturedSchedule::capture(&graph, &schedule, &[output_node]).unwrap();
        let artifact = CapturedSchedule::from_bytes(&captured.to_bytes().unwrap()).unwrap();
        let kernel_uop = &artifact.items[0].kernel;
        let UArg::TensorCoreMatmul(payload) = kernel_uop.arg() else {
            panic!("eligible narrow artifact did not retain tensor-core payload");
        };
        let lhs = tensor(
            vec![2, 16, 32],
            dtype,
            &(0..2 * 16 * 32)
                .map(|index| (index % 7) as f64 - 3.0)
                .collect::<Vec<_>>(),
        );
        let rhs = tensor(
            vec![1, 32, 16],
            dtype,
            &(0..32 * 16)
                .map(|index| (index % 5) as f64 - 2.0)
                .collect::<Vec<_>>(),
        );
        let expected = payload.simulate(&lhs, &rhs).unwrap();
        let rendered = PtxRenderer::new(80).unwrap().render(kernel_uop).unwrap();
        assert!(rendered.source.contains("mma.sync.aligned.m16n8k16"));
        let lhs_lease = lease(&primary, &lhs.to_le_bytes().unwrap());
        let rhs_lease = lease(&primary, &rhs.to_le_bytes().unwrap());
        let output_lease = lease(&primary, &vec![0xa5; expected.to_le_bytes().unwrap().len()]);
        let loaded = cache.get_or_load(&primary, rendered.clone(), 32).unwrap();
        loaded
            .launch(
                &stream,
                &[
                    PtxBinding {
                        buffer: lhs_lease.view().unwrap(),
                        dtype,
                        mutable: false,
                    },
                    PtxBinding {
                        buffer: rhs_lease.view().unwrap(),
                        dtype,
                        mutable: false,
                    },
                    PtxBinding {
                        buffer: output_lease.view().unwrap(),
                        dtype,
                        mutable: true,
                    },
                ],
                true,
            )
            .unwrap();
        let mut actual = vec![0; expected.to_le_bytes().unwrap().len()];
        output_lease
            .view()
            .unwrap()
            .copy_to(0, &mut actual)
            .unwrap();
        assert_eq!(actual, expected.to_le_bytes().unwrap());
        let repeated = cache.get_or_load(&primary, rendered.clone(), 32).unwrap();
        assert!(Arc::ptr_eq(&loaded, &repeated));

        let sentinel = vec![0x5a; expected.to_le_bytes().unwrap().len()];
        output_lease
            .view()
            .unwrap()
            .copy_from(0, &sentinel)
            .unwrap();
        assert!(matches!(
            loaded.launch(
                &stream,
                &[
                    PtxBinding {
                        buffer: lhs_lease.view().unwrap(),
                        dtype: DType::F32,
                        mutable: false,
                    },
                    PtxBinding {
                        buffer: rhs_lease.view().unwrap(),
                        dtype,
                        mutable: false,
                    },
                    PtxBinding {
                        buffer: output_lease.view().unwrap(),
                        dtype,
                        mutable: true,
                    },
                ],
                true,
            ),
            Err(PtxError::InvalidBinding(_))
        ));
        let mut unchanged = vec![0; sentinel.len()];
        output_lease
            .view()
            .unwrap()
            .copy_to(0, &mut unchanged)
            .unwrap();
        assert_eq!(unchanged, sentinel);

        let mut malformed = rendered;
        let crate::PtxLaunchGeometry::Exact(mut launch) = malformed.launch else {
            panic!("tensor-core launch is not exact");
        };
        launch.block = [64, 1, 1];
        malformed.launch = crate::PtxLaunchGeometry::Exact(launch);
        assert!(matches!(
            cache.get_or_load(&primary, malformed, 32),
            Err(PtxError::InvalidBinding(_))
        ));
    }
    assert_eq!(cache.len(), 2);
    assert_eq!(mock.generic_kernel_count(), 2);
}
