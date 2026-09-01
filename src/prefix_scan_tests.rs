use crate::{Backend, CpuBackend, DType, Error, Graph, Shape, TensorData};
use std::collections::HashMap;

fn execute(graph: &Graph, output: crate::NodeId, input: TensorData) -> TensorData {
    CpuBackend
        .execute(graph, output, &HashMap::from([("x".into(), input)]))
        .unwrap()
}

#[test]
fn cumsum_matches_tinygrad_values_for_signed_axes_and_empty_extents() {
    let cases = [
        (
            Shape::new([2, 3]),
            1,
            vec![1, 2, 3, 4, 5, 6],
            vec![1, 3, 6, 4, 9, 15],
        ),
        (
            Shape::new([2, 3]),
            -2,
            vec![1, 2, 3, 4, 5, 6],
            vec![1, 2, 3, 5, 7, 9],
        ),
    ];
    for (shape, axis, input, expected) in cases {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", shape.clone(), DType::I16);
        let output = graph.cumsum(x, axis).unwrap();
        let actual = execute(
            &graph,
            output,
            TensorData::from_scalars(shape, DType::I16, input.into_iter().map(crate::Scalar::I))
                .unwrap(),
        );
        assert_eq!(actual.dtype(), DType::I32);
        assert_eq!(
            actual.to_vec_f64(),
            expected.into_iter().map(f64::from).collect::<Vec<_>>()
        );
    }

    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 0, 3], DType::I8);
    let output = graph.cumsum(x, -2).unwrap();
    let actual = execute(
        &graph,
        output,
        TensorData::from_scalars([2, 0, 3], DType::I8, []).unwrap(),
    );
    assert_eq!(actual.shape(), &Shape::new([2, 0, 3]));
    assert_eq!(actual.dtype(), DType::I32);
}

#[test]
fn cumsum_dtype_scalar_trace_and_artifact_are_canonical() {
    let dtype_cases = [
        (DType::Bool, DType::I32),
        (DType::I8, DType::I32),
        (DType::U8, DType::U32),
        (DType::F16, DType::F16),
        (DType::F32, DType::F32),
    ];
    for (input_dtype, output_dtype) in dtype_cases {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2], input_dtype);
        let output = graph.cumsum(x, 0).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), output_dtype);
    }

    let mut graph = Graph::new();
    let scalar = graph.input_dtype("x", [], DType::I8);
    let output = graph.cumsum(scalar, -1).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([]));
    assert_eq!(graph.dtype(output).unwrap(), DType::I32);
    assert_eq!(
        execute(
            &graph,
            output,
            TensorData::from_scalars([], DType::I8, [crate::Scalar::I(7)]).unwrap(),
        )
        .to_vec_f64(),
        vec![7.0]
    );
    let trace = graph.trace(output).unwrap().to_string();
    assert!(trace.contains("cumsum(%"));
    assert!(trace.contains("axis=0"));
    assert!(trace.contains("[] I32"));
    let lowered = crate::lower_graph_prefix_scan(&graph, output).unwrap();
    lowered.validate().unwrap();
    let bytes = crate::uop::artifact::encode(&lowered).unwrap();
    assert_eq!(crate::uop::artifact::decode(&bytes).unwrap(), lowered);
    assert_eq!(crate::uop::artifact::encode(&lowered).unwrap(), bytes);
}

#[test]
fn cumsum_commits_the_tinygrad_work_dtype_at_each_prefix_boundary() {
    let mut f32_graph = Graph::new();
    let f32_input = f32_graph.input_dtype("x", [3], DType::F32);
    let f32_output = f32_graph.cumsum(f32_input, 0).unwrap();
    let f32 = execute(
        &f32_graph,
        f32_output,
        TensorData::from_storage(
            [3],
            crate::Storage::F32(vec![16_777_216.0, 1.0, -16_777_216.0]),
        )
        .unwrap(),
    );
    assert_eq!(
        f32.storage(),
        &crate::Storage::F32(vec![16_777_216.0, 16_777_216.0, 0.0])
    );

    let mut f16_graph = Graph::new();
    let f16_input = f16_graph.input_dtype("x", [3], DType::F16);
    let f16_output = f16_graph.cumsum(f16_input, 0).unwrap();
    let f16_source = TensorData::from_scalars(
        [3],
        DType::F16,
        [2048.0, 1.0, -2048.0].into_iter().map(crate::Scalar::F),
    )
    .unwrap();
    let f16 = execute(&f16_graph, f16_output, f16_source);
    assert_eq!(f16.to_vec_f64(), vec![2048.0, 2048.0, 1.0]);
}

#[test]
fn cumsum_rejects_invalid_axes_without_graph_mutation() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 3]);
    let before = graph.trace(x).unwrap();
    assert!(matches!(
        graph.cumsum(x, 2),
        Err(Error::InvalidReductionAxes { node, rank: 2, .. }) if node == x
    ));
    assert_eq!(graph.trace(x).unwrap(), before);

    let scalar = graph.input("scalar", []);
    let before = graph.trace(scalar).unwrap();
    assert!(matches!(
        graph.cumsum(scalar, 1),
        Err(Error::InvalidReductionAxes { node, rank: 0, .. }) if node == scalar
    ));
    assert_eq!(graph.trace(scalar).unwrap(), before);
}

#[test]
fn prefix_scan_artifact_rejects_malformed_static_geometry() {
    let malformed = crate::UOp::from_operation(
        crate::Operation::PrefixScan(crate::PrefixScanValue {
            input: crate::NodeId::from_index(0),
            destination: crate::NodeId::from_index(1),
            input_shape: Shape::new([2]),
            output_shape: Shape::new([3]),
            axis: 0,
            kind: crate::PrefixScanKind::Sum,
            output: crate::PrefixScanOutput::Values,
            input_dtype: DType::I32,
            dtype: DType::I32,
        }),
        Some(crate::UType::scalar(DType::I32)),
        vec![],
    );
    assert!(crate::uop::artifact::encode(&malformed).is_err());
}

#[test]
fn native_prefix_scan_plan_carries_exact_source_result_abi_and_value_kinds() {
    for input_dtype in DType::ALL {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3], input_dtype);
        let outputs = [
            graph.cumsum(input, 1).unwrap(),
            graph.cumprod(input, 1).unwrap(),
            graph.cummax(input, 1).unwrap().0,
            graph.cummin(input, 1).unwrap().0,
        ];
        for output in outputs {
            let kernel = crate::lower_graph_prefix_scan(&graph, output).unwrap();
            let crate::Operation::PrefixScan(plan) = kernel.operation() else {
                panic!("prefix scan must retain its typed semantic root")
            };
            assert_eq!(plan.input, input);
            assert_eq!(plan.destination, output);
            assert_eq!(plan.input_dtype, input_dtype);
            assert_eq!(plan.dtype, graph.dtype(output).unwrap());
            let rendered = crate::CpuJit::render(&kernel).unwrap();
            assert_eq!(rendered.abi.buffers.len(), 2);
            assert_eq!(rendered.abi.buffers[0].id, input.index() as u64);
            assert_eq!(rendered.abi.buffers[0].dtype, input_dtype);
            assert!(!rendered.abi.buffers[0].mutable);
            assert_eq!(rendered.abi.buffers[1].id, output.index() as u64);
            assert_eq!(rendered.abi.buffers[1].dtype, plan.dtype);
            assert!(rendered.abi.buffers[1].mutable);
            assert!(rendered.source.contains("prefix-scan"));
            assert!(rendered.source.contains("rg_axis"));
            assert_eq!(
                crate::CpuJit::render_vectorized(&kernel).unwrap().cache_key,
                rendered.cache_key
            );
            let bytes = crate::uop::artifact::encode(&kernel).unwrap();
            assert_eq!(bytes[4], 19);
            assert_eq!(crate::uop::artifact::decode(&bytes).unwrap(), kernel);
        }
    }
}

#[test]
fn portable_prefix_scan_projection_authenticates_common_dtype_kind_and_launch_domains() {
    for dtype in [DType::Bool, DType::I32, DType::U32, DType::F32] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3], dtype);
        let sum = graph.cumsum(input, 1).unwrap();
        let product = graph.cumprod(input, 1).unwrap();
        let (maximum, maximum_indices) = graph.cummax(input, 1).unwrap();
        let (minimum, minimum_indices) = graph.cummin(input, 1).unwrap();
        for output in [
            sum,
            product,
            maximum,
            maximum_indices,
            minimum,
            minimum_indices,
        ] {
            let kernel = crate::lower_graph_prefix_scan(&graph, output).unwrap();
            let crate::Operation::PrefixScan(value) = kernel.operation() else {
                unreachable!()
            };
            let portable = crate::prefix_scan_native::PortablePrefixScan::new(value).unwrap();
            assert_eq!(portable.plan().elements, 6);
            assert_eq!(portable.launch_extent(), 2);
        }
    }

    let mut empty = Graph::new();
    let input = empty.input_dtype("x", [2, 0, 3], DType::F32);
    let output = empty.cumsum(input, 1).unwrap();
    let kernel = crate::lower_graph_prefix_scan(&empty, output).unwrap();
    let crate::Operation::PrefixScan(value) = kernel.operation() else {
        unreachable!()
    };
    let portable = crate::prefix_scan_native::PortablePrefixScan::new(value).unwrap();
    assert_eq!(portable.plan().elements, 0);
    assert_eq!(portable.launch_extent(), 0);

    let mut zero_inner = Graph::new();
    let input = zero_inner.input_dtype("x", [2, 3, 0], DType::F32);
    let output = zero_inner.cumsum(input, 1).unwrap();
    let kernel = crate::lower_graph_prefix_scan(&zero_inner, output).unwrap();
    let crate::Operation::PrefixScan(value) = kernel.operation() else {
        unreachable!()
    };
    let portable = crate::prefix_scan_native::PortablePrefixScan::new(value).unwrap();
    assert_eq!(portable.plan().inner, 0);
    assert_eq!(portable.launch_extent(), 0);

    let mut narrow = Graph::new();
    let input = narrow.input_dtype("x", [2], DType::F16);
    let output = narrow.cumsum(input, 0).unwrap();
    let kernel = crate::lower_graph_prefix_scan(&narrow, output).unwrap();
    let crate::Operation::PrefixScan(value) = kernel.operation() else {
        unreachable!()
    };
    assert!(matches!(
        crate::prefix_scan_native::PortablePrefixScan::new(value),
        Err(crate::prefix_scan_native::PortablePrefixScanError::Unsupported(_))
    ));
}

#[test]
fn native_prefix_scan_indices_share_exact_abi_and_tampered_source_dtype_fails_closed() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [3], DType::I16);
    let (values, indices) = graph.cummax(input, 0).unwrap();
    let values_kernel = crate::lower_graph_prefix_scan(&graph, values).unwrap();
    let crate::Operation::PrefixScan(values_plan) = values_kernel.operation() else {
        unreachable!()
    };
    let mut tampered = values_plan.clone();
    tampered.input_dtype = DType::F32;
    let malformed = crate::UOp::from_operation(
        crate::Operation::PrefixScan(tampered),
        values_kernel.ty(),
        vec![],
    );
    assert!(malformed.validate().is_err());
    assert!(crate::CpuJit::render(&malformed).is_err());

    let indices_kernel = crate::lower_graph_prefix_scan(&graph, indices).unwrap();
    let indices_rendered = crate::CpuJit::render(&indices_kernel).unwrap();
    assert_eq!(indices_rendered.abi.buffers[0].dtype, DType::I16);
    assert_eq!(indices_rendered.abi.buffers[1].dtype, DType::I32);
    assert!(indices_rendered.source.contains("rg_index"));
    assert!(indices_rendered.source.contains("int32_t rg_index = 3;"));
    assert!(indices_rendered.source.contains("rg_strict"));
    assert!(indices_rendered.source.contains("rg_index == 3"));
}

#[test]
fn native_prefix_scan_first_match_state_is_canonical() {
    let update = crate::prefix_scan_native::first_match_index;
    let sentinel = 4;

    assert_eq!(update(sentinel, sentinel, 0, false, true), 0);
    assert_eq!(update(0, sentinel, 1, false, true), 0);
    assert_eq!(update(0, sentinel, 2, true, true), 2);
    assert_eq!(update(sentinel, sentinel, 0, false, false), sentinel);
}

#[test]
fn native_prefix_scan_executes_representative_storage_and_result_contracts() {
    use crate::{JitBuffer, PrefixScanKind as Kind, PrefixScanOutput as Output, Scalar};

    for (dtype, kind, result) in [
        (DType::Bool, Kind::Product, Output::Values),
        (DType::Bool, Kind::Max, Output::Values),
        (DType::Bool, Kind::Max, Output::Indices),
        (DType::Bool, Kind::Min, Output::Values),
        (DType::Bool, Kind::Min, Output::Indices),
        (DType::I8, Kind::Sum, Output::Values),
        (DType::U16, Kind::Sum, Output::Values),
        (DType::I64, Kind::Product, Output::Values),
        (DType::U64, Kind::Max, Output::Indices),
        (DType::F8E4M3, Kind::Sum, Output::Values),
        (DType::F8E5M2FNUZ, Kind::Product, Output::Values),
        (DType::F16, Kind::Sum, Output::Values),
        (DType::BF16, Kind::Product, Output::Values),
        (DType::F32, Kind::Max, Output::Values),
        (DType::F32, Kind::Max, Output::Indices),
        (DType::F64, Kind::Min, Output::Values),
        (DType::F8E4M3, Kind::Max, Output::Values),
        (DType::F8E4M3, Kind::Max, Output::Indices),
        (DType::F8E4M3, Kind::Min, Output::Values),
        (DType::F8E4M3, Kind::Min, Output::Indices),
        (DType::F8E5M2, Kind::Max, Output::Values),
        (DType::F8E5M2, Kind::Max, Output::Indices),
        (DType::F8E5M2, Kind::Min, Output::Values),
        (DType::F8E5M2, Kind::Min, Output::Indices),
        (DType::F8E4M3FNUZ, Kind::Max, Output::Values),
        (DType::F8E4M3FNUZ, Kind::Max, Output::Indices),
        (DType::F8E4M3FNUZ, Kind::Min, Output::Values),
        (DType::F8E4M3FNUZ, Kind::Min, Output::Indices),
        (DType::F8E5M2FNUZ, Kind::Max, Output::Values),
        (DType::F8E5M2FNUZ, Kind::Max, Output::Indices),
        (DType::F8E5M2FNUZ, Kind::Min, Output::Values),
        (DType::F8E5M2FNUZ, Kind::Min, Output::Indices),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [3], dtype);
        let output = match (kind, result) {
            (Kind::Sum, Output::Values) => graph.cumsum(input, 0).unwrap(),
            (Kind::Product, Output::Values) => graph.cumprod(input, 0).unwrap(),
            (Kind::Max, Output::Values) => graph.cummax(input, 0).unwrap().0,
            (Kind::Max, Output::Indices) => graph.cummax(input, 0).unwrap().1,
            (Kind::Min, Output::Values) => graph.cummin(input, 0).unwrap().0,
            (Kind::Min, Output::Indices) => graph.cummin(input, 0).unwrap().1,
            _ => unreachable!("public scan result contract"),
        };
        let values = if dtype == DType::Bool {
            [Scalar::Bool(true), Scalar::Bool(false), Scalar::Bool(true)]
        } else if dtype.is_float() {
            [Scalar::F(0.5), Scalar::F(-0.0), Scalar::F(2.0)]
        } else {
            [Scalar::I(1), Scalar::I(2), Scalar::I(1)]
        };
        let source = TensorData::from_scalars([3], dtype, values).unwrap();
        let expected = execute(&graph, output, source.clone());
        let kernel =
            crate::CpuJit::compile(&crate::lower_graph_prefix_scan(&graph, output).unwrap())
                .unwrap();
        let mut buffers = [
            JitBuffer::from_tensor(&source, false),
            JitBuffer::zeroed(expected.dtype(), expected.len(), true),
        ];
        kernel.call(&mut buffers, &[]).unwrap();
        let actual = buffers[1]
            .clone()
            .into_tensor(expected.shape().clone())
            .unwrap();
        assert_eq!(
            actual.storage(),
            expected.storage(),
            "{dtype:?} {kind:?} {result:?}"
        );
    }
}

#[test]
fn native_scalar_prefix_scans_copy_source_bits_and_emit_zero_indices() {
    let sources = [
        TensorData::from_storage([], crate::Storage::F32(vec![f32::from_bits(0x7fc0_1234)]))
            .unwrap(),
        TensorData::from_storage(
            [],
            crate::Storage::Float8(crate::Float8Storage::from_raw(
                crate::Float8Format::E4M3,
                vec![0xff],
            )),
        )
        .unwrap(),
        TensorData::from_storage([], crate::Storage::Bool(vec![true])).unwrap(),
        TensorData::from_storage([], crate::Storage::I16(vec![-7])).unwrap(),
    ];
    for source in sources {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [], source.dtype());
        let sum = graph.cumsum(input, 0).unwrap();
        let product = graph.cumprod(input, 0).unwrap();
        let (maximum, maximum_indices) = graph.cummax(input, 0).unwrap();
        let (minimum, minimum_indices) = graph.cummin(input, 0).unwrap();
        for output in [
            sum,
            product,
            maximum,
            minimum,
            maximum_indices,
            minimum_indices,
        ] {
            let expected = execute(&graph, output, source.clone());
            let lowered = crate::lower_graph_prefix_scan(&graph, output).unwrap();
            let rendered = crate::CpuJit::render(&lowered).unwrap();
            assert!(!rendered.source.contains("for (size_t rg_axis"));
            if output != maximum_indices
                && output != minimum_indices
                && source.dtype() == expected.dtype()
            {
                assert!(rendered.source.contains("memcpy(buffers[1], buffers[0]"));
            }
            let kernel = crate::CpuJit::compile(&lowered).unwrap();
            let mut buffers = [
                crate::JitBuffer::from_tensor(&source, false),
                crate::JitBuffer::zeroed(expected.dtype(), 1, true),
            ];
            kernel.call(&mut buffers, &[]).unwrap();
            let expected_bytes = expected.to_le_bytes().unwrap();
            assert_eq!(
                buffers[1].bytes(),
                expected_bytes.as_slice(),
                "{:?} scalar {:?}",
                source.dtype(),
                lowered.operation()
            );

            if source.dtype() == DType::F32 {
                let ptx = crate::PtxRenderer::new(80)
                    .unwrap()
                    .render(&lowered)
                    .unwrap();
                assert!(!ptx.source.contains("SCAN_LOOP"));
                if output == maximum_indices || output == minimum_indices {
                    assert!(ptx.source.contains("mov.u32 %r7, 0;"));
                    assert!(ptx.source.contains("st.global.s32 [%rd1], %r7;"));
                } else {
                    assert!(ptx.source.contains("ld.global.b32 %r10, [%rd0];"));
                    assert!(ptx.source.contains("st.global.b32 [%rd1], %r10;"));
                }
            }
        }
    }
}

#[test]
fn ptx_prefix_scan_values_and_indices_share_the_typed_plan_and_fail_closed_by_capability() {
    for dtype in [DType::Bool, DType::I32, DType::U32, DType::F32] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3], dtype);
        let sum = graph.cumsum(input, 1).unwrap();
        let product = graph.cumprod(input, 1).unwrap();
        let (maximum, max_indices) = graph.cummax(input, 1).unwrap();
        let (minimum, min_indices) = graph.cummin(input, 1).unwrap();
        for output in [sum, product, maximum, max_indices, minimum, min_indices] {
            let kernel = crate::lower_graph_prefix_scan(&graph, output).unwrap();
            let rendered = crate::PtxRenderer::new(80)
                .unwrap()
                .render(&kernel)
                .unwrap();
            assert_eq!(rendered.buffers.len(), 2);
            assert_eq!(rendered.buffers[0].dtype, dtype);
            assert_eq!(rendered.buffers[1].dtype, graph.dtype(output).unwrap());
            assert_eq!(rendered.extent, 2);
            assert!(rendered.source.contains("SCAN_LOOP"));
            assert!(rendered.source.contains(crate::ptx::PTX_RENDERER_VERSION));
            if output == max_indices || output == min_indices {
                assert!(rendered.source.contains("mov.u32 %r7, 3;"));
                assert!(rendered.source.contains("setp.eq.u32 %p6, %r7, 3;"));
                assert!(rendered.source.contains("or.pred %p7, %p7, %p3;"));
            }
        }
    }

    for dtype in [DType::Bool, DType::I32, DType::U32, DType::F32] {
        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], dtype);
        let sum = scalar.cumsum(input, 0).unwrap();
        let product = scalar.cumprod(input, 0).unwrap();
        let (maximum, maximum_indices) = scalar.cummax(input, 0).unwrap();
        let (minimum, minimum_indices) = scalar.cummin(input, 0).unwrap();
        for output in [
            sum,
            product,
            maximum,
            minimum,
            maximum_indices,
            minimum_indices,
        ] {
            let rendered = crate::PtxRenderer::new(80)
                .unwrap()
                .render(&crate::lower_graph_prefix_scan(&scalar, output).unwrap())
                .unwrap();
            assert!(!rendered.source.contains("SCAN_LOOP"));
            if output == maximum_indices || output == minimum_indices {
                assert!(rendered.source.contains("mov.u32 %r7, 0;"));
                assert!(rendered.source.contains("st.global.s32 [%rd1], %r7;"));
            } else if dtype == DType::Bool && output == sum {
                assert!(rendered.source.contains("ld.global.u8 %r10, [%rd0];"));
                assert!(rendered.source.contains("st.global.s32 [%rd1], %r10;"));
            } else if dtype == DType::Bool {
                assert!(rendered.source.contains("ld.global.u8 %r10, [%rd0];"));
                assert!(rendered.source.contains("st.global.u8 [%rd1], %r10;"));
            } else {
                assert!(rendered.source.contains("ld.global.b32 %r10, [%rd0];"));
                assert!(rendered.source.contains("st.global.b32 [%rd1], %r10;"));
            }
        }
    }

    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2], DType::F16);
    let output = graph.cumsum(input, 0).unwrap();
    let kernel = crate::lower_graph_prefix_scan(&graph, output).unwrap();
    assert!(matches!(
        crate::PtxRenderer::new(80).unwrap().render(&kernel),
        Err(crate::PtxError::Unsupported(reason))
            if reason == "PTX prefix scan requires a 32-bit Bool/I32/U32/F32 domain"
    ));
}

#[test]
fn cumprod_matches_tinygrad_values_dtypes_and_empty_scalar_contracts() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 3], DType::I16);
    let output = graph.cumprod(x, -1).unwrap();
    let actual = execute(
        &graph,
        output,
        TensorData::from_scalars(
            [2, 3],
            DType::I16,
            [2, 3, 4, -1, 2, 3].into_iter().map(crate::Scalar::I),
        )
        .unwrap(),
    );
    assert_eq!(actual.dtype(), DType::I16);
    assert_eq!(actual.to_vec_f64(), vec![2., 6., 24., -1., -2., -6.]);

    let mut boolean_graph = Graph::new();
    let boolean = boolean_graph.input_dtype("x", [3], DType::Bool);
    let boolean_output = boolean_graph.cumprod(boolean, 0).unwrap();
    assert_eq!(boolean_graph.dtype(boolean_output).unwrap(), DType::Bool);
    assert_eq!(
        execute(
            &boolean_graph,
            boolean_output,
            TensorData::from_scalars(
                [3],
                DType::Bool,
                [true, false, true].into_iter().map(crate::Scalar::Bool),
            )
            .unwrap(),
        )
        .to_vec_f64(),
        vec![1., 0., 0.]
    );

    let mut empty_graph = Graph::new();
    let empty = empty_graph.input_dtype("x", [2, 0], DType::U8);
    let empty_output = empty_graph.cumprod(empty, 1).unwrap();
    let empty_value = execute(
        &empty_graph,
        empty_output,
        TensorData::from_scalars([2, 0], DType::U8, []).unwrap(),
    );
    assert_eq!(empty_value.shape(), &Shape::new([2, 0]));
    assert_eq!(empty_value.dtype(), DType::U8);

    let mut scalar_graph = Graph::new();
    let scalar = scalar_graph.input_dtype("x", [], DType::I8);
    let scalar_output = scalar_graph.cumprod(scalar, -1).unwrap();
    assert_eq!(scalar_graph.shape(scalar_output).unwrap(), &Shape::new([]));
    assert_eq!(scalar_graph.dtype(scalar_output).unwrap(), DType::I8);
    assert_eq!(
        execute(
            &scalar_graph,
            scalar_output,
            TensorData::from_scalars([], DType::I8, [crate::Scalar::I(-3)]).unwrap(),
        )
        .to_vec_f64(),
        vec![-3.]
    );
    let trace = scalar_graph.trace(scalar_output).unwrap().to_string();
    assert!(trace.contains("cumprod(%"));
}

#[test]
fn cumprod_artifact_round_trip_and_invalid_axis_leave_graph_unchanged() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 3], DType::I32);
    let before = graph.trace(x).unwrap();
    assert!(matches!(
        graph.cumprod(x, 2),
        Err(Error::InvalidReductionAxes { node, rank: 2, .. }) if node == x
    ));
    assert_eq!(graph.trace(x).unwrap(), before);

    let output = graph.cumprod(x, 0).unwrap();
    let lowered = crate::lower_graph_prefix_scan(&graph, output).unwrap();
    lowered.validate().unwrap();
    let bytes = crate::uop::artifact::encode(&lowered).unwrap();
    assert_eq!(crate::uop::artifact::decode(&bytes).unwrap(), lowered);
}

#[test]
fn cumulative_extrema_match_tinygrad_first_match_indices_and_static_edges() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 4], DType::I32);
    let before = graph.trace(x).unwrap();
    assert!(matches!(
        graph.cummax(x, 2),
        Err(Error::InvalidReductionAxes { node, rank: 2, .. }) if node == x
    ));
    assert_eq!(graph.trace(x).unwrap(), before);
    let (maximum, max_indices) = graph.cummax(x, -1).unwrap();
    let (minimum, min_indices) = graph.cummin(x, 1).unwrap();
    let input = TensorData::from_scalars(
        [2, 4],
        DType::I32,
        [1, 3, 3, 2, 4, 2, 2, 5].into_iter().map(crate::Scalar::I),
    )
    .unwrap();
    assert_eq!(
        execute(&graph, maximum, input.clone()).to_vec_f64(),
        vec![1., 3., 3., 3., 4., 4., 4., 5.]
    );
    assert_eq!(
        execute(&graph, max_indices, input.clone()).to_vec_f64(),
        vec![0., 1., 1., 1., 0., 0., 0., 3.]
    );
    assert_eq!(
        execute(&graph, minimum, input.clone()).to_vec_f64(),
        vec![1., 1., 1., 1., 4., 2., 2., 2.]
    );
    assert_eq!(
        execute(&graph, min_indices, input).to_vec_f64(),
        vec![0., 0., 0., 0., 0., 1., 1., 1.]
    );
    assert_eq!(graph.dtype(max_indices).unwrap(), DType::I32);
    let trace = graph.trace(max_indices).unwrap().to_string();
    assert!(trace.contains("cummax_indices(%"));
    assert!(trace.contains("axis=1"));
    let kernel = crate::lower_graph_prefix_scan(&graph, max_indices).unwrap();
    let bytes = crate::uop::artifact::encode(&kernel).unwrap();
    assert_eq!(crate::uop::artifact::decode(&bytes).unwrap(), kernel);

    for (kind, input, expected_values, expected_indices) in [
        (
            crate::PrefixScanKind::Max,
            [false, false, true, true, false],
            [false, false, true, true, true],
            [0., 0., 2., 2., 2.],
        ),
        (
            crate::PrefixScanKind::Min,
            [true, true, false, false, true],
            [true, true, false, false, false],
            [0., 0., 2., 2., 2.],
        ),
    ] {
        let mut boolean_graph = Graph::new();
        let source = boolean_graph.input_dtype("x", [5], DType::Bool);
        let (values, indices) = match kind {
            crate::PrefixScanKind::Max => boolean_graph.cummax(source, 0).unwrap(),
            crate::PrefixScanKind::Min => boolean_graph.cummin(source, 0).unwrap(),
            _ => unreachable!(),
        };
        let input =
            TensorData::from_scalars([5], DType::Bool, input.into_iter().map(crate::Scalar::Bool))
                .unwrap();
        assert_eq!(
            execute(&boolean_graph, values, input.clone()).storage(),
            &crate::Storage::Bool(expected_values.to_vec())
        );
        assert_eq!(
            execute(&boolean_graph, indices, input).to_vec_f64(),
            expected_indices
        );
    }

    let mut empty_graph = Graph::new();
    let empty = empty_graph.input_dtype("x", [0], DType::F32);
    let (values, indices) = empty_graph.cummax(empty, 0).unwrap();
    assert_eq!(empty_graph.shape(values).unwrap(), &Shape::new([0]));
    assert_eq!(empty_graph.dtype(indices).unwrap(), DType::I32);

    let mut scalar_graph = Graph::new();
    let scalar = scalar_graph.input_dtype("x", [], DType::I16);
    let (value, index) = scalar_graph.cummin(scalar, -1).unwrap();
    let input = TensorData::from_scalars([], DType::I16, [crate::Scalar::I(-7)]).unwrap();
    assert_eq!(
        execute(&scalar_graph, value, input.clone()).to_vec_f64(),
        vec![-7.]
    );
    assert_eq!(execute(&scalar_graph, index, input).to_vec_f64(), vec![0.]);

    // tinygrad's Ops.MAX uses left-biased `max`: NaNs do not replace a finite
    // prefix and an equal positive zero does not replace an earlier negative zero.
    let mut float_graph = Graph::new();
    let float = float_graph.input_dtype("x", [4], DType::F32);
    let (values, indices) = float_graph.cummax(float, 0).unwrap();
    let input = TensorData::from_scalars(
        [4],
        DType::F32,
        [
            crate::Scalar::F(-0.0),
            crate::Scalar::F(0.0),
            crate::Scalar::F(f64::NAN),
            crate::Scalar::F(-1.0),
        ],
    )
    .unwrap();
    let actual = execute(&float_graph, values, input.clone());
    let crate::Storage::F32(actual) = actual.storage() else {
        panic!("expected F32 cumulative maximum")
    };
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![(-0.0f32).to_bits(); 4]
    );
    assert_eq!(
        execute(&float_graph, indices, input).to_vec_f64(),
        vec![0., 0., 0., 0.]
    );

    let mut sentinel_graph = Graph::new();
    let source = sentinel_graph.input_dtype("x", [2], DType::F32);
    let (values, indices) = sentinel_graph.cummax(source, 0).unwrap();
    let leading_nan = TensorData::from_storage(
        [2],
        crate::Storage::F32(vec![f32::from_bits(0x7fc0_0123), 1.0]),
    )
    .unwrap();
    let actual = execute(&sentinel_graph, values, leading_nan.clone());
    let crate::Storage::F32(actual) = actual.storage() else {
        panic!("expected F32 cumulative maximum")
    };
    assert_eq!(actual[0], f32::NEG_INFINITY);
    assert_eq!(actual[1], 1.0);
    assert_eq!(
        execute(&sentinel_graph, indices, leading_nan).to_vec_f64(),
        vec![2., 1.]
    );

    let identity = TensorData::from_storage(
        [2],
        crate::Storage::F32(vec![f32::NEG_INFINITY, f32::NEG_INFINITY]),
    )
    .unwrap();
    assert_eq!(
        execute(&sentinel_graph, indices, identity).to_vec_f64(),
        vec![0., 0.]
    );

    let (values, indices) = sentinel_graph.cummin(source, 0).unwrap();
    let leading_nan = TensorData::from_storage(
        [2],
        crate::Storage::F32(vec![f32::from_bits(0x7fc0_0123), 1.0]),
    )
    .unwrap();
    assert_eq!(
        execute(&sentinel_graph, values, leading_nan.clone()).storage(),
        &crate::Storage::F32(vec![f32::INFINITY, 1.0])
    );
    assert_eq!(
        execute(&sentinel_graph, indices, leading_nan).to_vec_f64(),
        vec![2., 1.]
    );
}

#[test]
fn float8_prefix_scans_commit_typed_identities_and_execute_every_kind() {
    for (dtype, maximum_identity, minimum_identity, unordered_identity) in [
        // tinygrad's dtype constant first truncates -/+infinity through E4M3,
        // then exposes the decoded canonical NaN used by the recurrence.
        (DType::F8E4M3, 0x7fu8, 0x7fu8, true),
        (DType::F8E5M2, 0xfcu8, 0x7cu8, false),
        (DType::F8E4M3FNUZ, 0x80u8, 0x80u8, true),
        (DType::F8E5M2FNUZ, 0x80u8, 0x80u8, true),
    ] {
        let identity_raw = |kind| {
            let scalar = crate::prefix_scan_native::scan_identity(dtype, dtype, kind);
            let data = TensorData::from_scalars([1], dtype, [scalar]).unwrap();
            let crate::Storage::Float8(values) = data.storage() else {
                unreachable!("Float8 identity storage")
            };
            values.as_raw()[0]
        };
        assert_eq!(identity_raw(crate::PrefixScanKind::Max), maximum_identity);
        assert_eq!(identity_raw(crate::PrefixScanKind::Min), minimum_identity);

        let mut graph = Graph::new();
        let source = graph.input_dtype("x", [3], dtype);
        let sum = graph.cumsum(source, 0).unwrap();
        let product = graph.cumprod(source, 0).unwrap();
        let (maximum, maximum_indices) = graph.cummax(source, 0).unwrap();
        let (minimum, minimum_indices) = graph.cummin(source, 0).unwrap();
        let maximum_source =
            crate::CpuJit::render(&crate::lower_graph_prefix_scan(&graph, maximum).unwrap())
                .unwrap()
                .source;
        let minimum_source =
            crate::CpuJit::render(&crate::lower_graph_prefix_scan(&graph, minimum).unwrap())
                .unwrap()
                .source;
        if unordered_identity {
            assert!(maximum_source.contains("0x7fc00000u"));
            assert!(minimum_source.contains("0x7fc00000u"));
        } else {
            assert!(maximum_source.contains("-INFINITY"));
            assert!(minimum_source.contains("INFINITY"));
        }
        let input =
            TensorData::from_scalars([3], dtype, [1.0, 2.0, 1.0].map(crate::Scalar::F)).unwrap();
        for (output, expected) in [(sum, [1.0, 3.0, 4.0]), (product, [1.0, 2.0, 2.0])] {
            assert_eq!(
                execute(&graph, output, input.clone()).storage(),
                TensorData::from_scalars([3], dtype, expected.map(crate::Scalar::F))
                    .unwrap()
                    .storage(),
                "{dtype:?} arithmetic scan"
            );
        }
        if unordered_identity {
            for (output, raw) in [(maximum, maximum_identity), (minimum, minimum_identity)] {
                let actual = execute(&graph, output, input.clone());
                let crate::Storage::Float8(values) = actual.storage() else {
                    unreachable!("Float8 extrema storage")
                };
                assert_eq!(values.as_raw(), &[raw; 3], "{dtype:?} extrema identity");
            }
            assert_eq!(
                execute(&graph, maximum_indices, input.clone()).to_vec_f64(),
                vec![3., 3., 3.]
            );
            assert_eq!(
                execute(&graph, minimum_indices, input).to_vec_f64(),
                vec![3., 3., 3.]
            );
        } else {
            assert_eq!(
                execute(&graph, maximum, input.clone()).to_vec_f64(),
                vec![1., 2., 2.]
            );
            assert_eq!(
                execute(&graph, maximum_indices, input.clone()).to_vec_f64(),
                vec![0., 1., 1.]
            );
            assert_eq!(
                execute(&graph, minimum, input.clone()).to_vec_f64(),
                vec![1., 1., 1.]
            );
            assert_eq!(
                execute(&graph, minimum_indices, input).to_vec_f64(),
                vec![0., 0., 0.]
            );
        }
    }
}
