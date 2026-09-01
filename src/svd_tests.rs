use crate::{
    CapturedSchedule, DType, Error, Graph, NodeId, Op, RealizationPolicy, Scalar, Shape, TensorData,
};
use std::collections::HashMap;

fn f32_data(shape: impl Into<Shape>, values: impl IntoIterator<Item = f32>) -> TensorData {
    TensorData::from_scalars(
        shape,
        DType::F32,
        values.into_iter().map(|v| Scalar::F(v as f64)),
    )
    .unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "lane {index}: expected {expected}, got {actual}"
        );
    }
}

fn matmul_2x2(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    (0..2)
        .flat_map(|row| {
            (0..2).map(move |column| {
                (0..2)
                    .map(|inner| lhs[row * 2 + inner] * rhs[inner * 2 + column])
                    .sum()
            })
        })
        .collect()
}

fn transpose_2x2(value: &[f64]) -> Vec<f64> {
    vec![value[0], value[2], value[1], value[3]]
}

fn numeric_svd_graph(shape: impl Into<Shape>, full_matrices: bool) -> (Graph, [NodeId; 4]) {
    let shape = shape.into();
    let rank = shape.rank();
    let m = shape.dims()[rank - 2];
    let n = shape.dims()[rank - 1];
    let k = m.min(n);
    let batch = shape.dims()[..rank - 2].to_vec();
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", shape, DType::F32);
    let (u, singular, vt) = graph.svd(input, full_matrices).unwrap();
    let eye = graph.eye(k, Some(k), DType::F32).unwrap();
    let mut eye_shape = batch.clone();
    eye_shape.extend([k, k]);
    let eye = graph.expand(eye, Shape::new(eye_shape)).unwrap();
    let singular_row = graph.unsqueeze(singular, -2).unwrap();
    let diagonal = graph.mul(singular_row, eye).unwrap();
    let diagonal = if full_matrices {
        let padding = batch
            .iter()
            .map(|_| (0, 0))
            .chain([(0, m - k), (0, n - k)])
            .collect::<Vec<_>>();
        graph.pad(diagonal, padding, Scalar::I(0)).unwrap()
    } else {
        diagonal
    };
    let reconstructed = graph.dot_default(u, diagonal).unwrap();
    let reconstructed = graph.dot_default(reconstructed, vt).unwrap();
    (graph, [u, singular, vt, reconstructed])
}

fn assert_orthonormal_rows(
    values: &[f64],
    batches: usize,
    rows: usize,
    columns: usize,
    tolerance: f64,
) {
    for batch in 0..batches {
        let offset = batch * rows * columns;
        for left in 0..rows {
            for right in 0..rows {
                let actual = (0..columns)
                    .map(|column| {
                        values[offset + left * columns + column]
                            * values[offset + right * columns + column]
                    })
                    .sum::<f64>();
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "batch {batch} row product ({left}, {right}): expected {expected}, got {actual}"
                );
            }
        }
    }
}

fn assert_orthonormal_columns(
    values: &[f64],
    batches: usize,
    rows: usize,
    columns: usize,
    tolerance: f64,
) {
    for batch in 0..batches {
        let offset = batch * rows * columns;
        for left in 0..columns {
            for right in 0..columns {
                let actual = (0..rows)
                    .map(|row| {
                        values[offset + row * columns + left]
                            * values[offset + row * columns + right]
                    })
                    .sum::<f64>();
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "batch {batch} column product ({left}, {right}): expected {expected}, got {actual}"
                );
            }
        }
    }
}

fn assert_numeric_svd(
    graph: &Graph,
    outputs: [NodeId; 4],
    shape: impl Into<Shape>,
    input_values: &[f32],
    full_matrices: bool,
    orthogonal: bool,
    tolerance: f64,
) -> Vec<f64> {
    let shape = shape.into();
    let rank = shape.rank();
    let m = shape.dims()[rank - 2];
    let n = shape.dims()[rank - 1];
    let k = m.min(n);
    let batches = shape.dims()[..rank - 2].iter().copied().product::<usize>();
    let realized = crate::realize_graph(
        graph,
        &outputs,
        &HashMap::from([(
            "x".into(),
            f32_data(shape.clone(), input_values.iter().copied()),
        )]),
        RealizationPolicy::Interpreter,
    )
    .unwrap();
    let u = realized.outputs[0].to_vec_f64();
    let singular = realized.outputs[1].to_vec_f64();
    let vt = realized.outputs[2].to_vec_f64();
    let reconstructed = realized.outputs[3].to_vec_f64();
    assert!(u.iter().chain(&singular).chain(&vt).all(|v| v.is_finite()));
    assert_close(
        &reconstructed,
        &input_values
            .iter()
            .map(|&value| value as f64)
            .collect::<Vec<_>>(),
        tolerance,
    );
    for values in singular.chunks(k) {
        assert!(values.iter().all(|&value| value >= 0.0));
        assert!(values.windows(2).all(|pair| pair[0] + tolerance >= pair[1]));
    }
    if orthogonal {
        let u_columns = if full_matrices { m } else { k };
        if full_matrices {
            assert_orthonormal_rows(&u, batches, m, u_columns, tolerance);
        } else {
            assert_orthonormal_columns(&u, batches, m, u_columns, tolerance);
        }
        let vt_rows = if full_matrices { n } else { k };
        assert_orthonormal_rows(&vt, batches, vt_rows, n, tolerance);
    }
    singular
}

#[test]
fn source_svd_uses_two_barrier_jacobi_sort_composition() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [3, 3], DType::F32);
    let (u, singular, vt) = graph.svd(input, false).unwrap();
    assert_eq!(graph.shape(u).unwrap(), &Shape::new([3, 3]));
    assert_eq!(graph.shape(singular).unwrap(), &Shape::new([3]));
    assert_eq!(graph.shape(vt).unwrap(), &Shape::new([3, 3]));

    let operations = (0..graph.node_count())
        .map(|index| graph.op(NodeId::from_index(index)).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        operations
            .iter()
            .filter(|op| matches!(op, Op::Contiguous { .. }))
            .count(),
        2,
        "the source's R and expanded-eye barriers must both remain visible"
    );
    let sort_selectors = operations
        .iter()
        .filter_map(|op| match op {
            Op::Sort { pair, .. } => Some(*pair),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(sort_selectors.len(), 2);
    assert_eq!(sort_selectors[0], sort_selectors[1]);
    assert!(operations.iter().all(|op| !matches!(op, Op::Matmul { .. })));

    let schedule = crate::schedule_many(&graph, &[u, singular, vt]).unwrap();
    let sort_item = schedule
        .items
        .iter()
        .find(|item| matches!(item.kernel.operation(), crate::Operation::Sort(_)))
        .expect("SVD schedule must retain its coupled Sort producer");
    assert_eq!(sort_item.outputs.len(), 2);
    assert!(graph.trace(u).unwrap().to_string().contains("contiguous(%"));
}

#[test]
fn source_svd_shapes_cover_full_reduced_batch_empty_and_source_singleton_rejection() {
    let mut tall = Graph::new();
    let x = tall.input_dtype("x", [5, 2], DType::F32);
    let (u, s, vt) = tall.svd_default(x).unwrap();
    assert_eq!(tall.shape(u).unwrap(), &Shape::new([5, 5]));
    assert_eq!(tall.shape(s).unwrap(), &Shape::new([2]));
    assert_eq!(tall.shape(vt).unwrap(), &Shape::new([2, 2]));

    let mut wide = Graph::new();
    let x = wide.input_dtype("x", [2, 5], DType::F64);
    let (u, s, vt) = wide.svd(x, false).unwrap();
    assert_eq!(wide.shape(u).unwrap(), &Shape::new([2, 2]));
    assert_eq!(wide.shape(s).unwrap(), &Shape::new([2]));
    assert_eq!(wide.shape(vt).unwrap(), &Shape::new([2, 5]));

    let mut batched = Graph::new();
    let x = batched.input_dtype("x", [2, 3, 2], DType::I16);
    let (u, s, vt) = batched.svd(x, false).unwrap();
    assert_eq!(batched.shape(u).unwrap(), &Shape::new([2, 3, 2]));
    assert_eq!(batched.shape(s).unwrap(), &Shape::new([2, 2]));
    assert_eq!(batched.shape(vt).unwrap(), &Shape::new([2, 2, 2]));
    assert_eq!(batched.dtype(u).unwrap(), DType::F32);
    assert_eq!(batched.dtype(s).unwrap(), DType::F32);
    assert_eq!(batched.dtype(vt).unwrap(), DType::F32);

    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [0, 2], DType::F32);
    let (u, s, vt) = empty.svd_default(x).unwrap();
    assert_eq!(empty.shape(u).unwrap(), &Shape::new([0, 0]));
    assert_eq!(empty.shape(s).unwrap(), &Shape::new([0]));
    assert_eq!(empty.shape(vt).unwrap(), &Shape::new([2, 2]));

    let mut singleton = Graph::new();
    let x = singleton.input_dtype("x", [1, 2], DType::F32);
    let before = singleton.node_count();
    assert!(matches!(
        singleton.svd_default(x),
        Err(Error::InvalidSplit { .. })
    ));
    assert_eq!(singleton.node_count(), before);
}

#[test]
fn source_svd_cpu_interpreter_reconstructs_and_orders_singular_values() {
    let mut graph = Graph::new();
    let input = graph.input_dtype_requires_grad("x", [2, 2], DType::F32, true);
    let (u, singular, vt) = graph.svd_default(input).unwrap();
    let eye = graph.eye(2, Some(2), DType::F32).unwrap();
    let singular_row = graph.unsqueeze(singular, -2).unwrap();
    let diagonal = graph.mul(singular_row, eye).unwrap();
    let u_diagonal = graph.dot_default(u, diagonal).unwrap();
    let reconstructed = graph.dot_default(u_diagonal, vt).unwrap();

    let values = f32_data([2, 2], [3.0, 0.0, 0.0, 2.0]);
    let realized = crate::realize_graph(
        &graph,
        &[u, singular, vt, reconstructed],
        &HashMap::from([("x".into(), values)]),
        RealizationPolicy::Interpreter,
    )
    .unwrap();
    let u_values = realized.outputs[0].to_vec_f64();
    let singular_values = realized.outputs[1].to_vec_f64();
    let vt_values = realized.outputs[2].to_vec_f64();
    assert_close(&singular_values, &[3.0, 2.0], 1.0e-4);
    assert_close(
        &realized.outputs[3].to_vec_f64(),
        &[3.0, 0.0, 0.0, 2.0],
        1.0e-4,
    );
    assert_close(
        &matmul_2x2(&u_values, &transpose_2x2(&u_values)),
        &[1.0, 0.0, 0.0, 1.0],
        1.0e-4,
    );
    assert_close(
        &matmul_2x2(&vt_values, &transpose_2x2(&vt_values)),
        &[1.0, 0.0, 0.0, 1.0],
        1.0e-4,
    );
}

#[test]
fn source_svd_realizes_well_conditioned_vjp_and_matches_finite_difference() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", Shape::new([2, 2]), DType::F32);
    let (_, singular, _) = graph.svd_default(input).unwrap();
    let loss = graph.sum_all(singular).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.shape(gradient).unwrap(), &Shape::new([2, 2]));
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F32);

    // Shrink is used by the source Jacobi/QR composition. Its exact static
    // ScatterPositions adjoint is an owned movement item rather than a hidden
    // host-oracle prerequisite.
    let schedule = crate::schedule_many(&graph, &[loss, gradient]).unwrap();
    schedule.validate().unwrap();
    let scatter_item = schedule
        .items
        .iter()
        .find(|item| matches!(graph.op(item.node), Ok(Op::ScatterPositions { .. })))
        .expect("SVD VJP must retain its static movement adjoint");
    assert!(scatter_item.boundary.is_none());
    assert!(matches!(
        scatter_item.kernel.operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(plan))
            if matches!(&plan.kind, crate::MovementKernelKind::ScatterPositions { .. })
    ));
    let crate::Operation::Movement(crate::MovementValue::Plan(plan)) =
        scatter_item.kernel.operation()
    else {
        unreachable!();
    };
    let crate::MovementKernelKind::ScatterPositions { input, .. } = &plan.kind else {
        unreachable!();
    };
    assert_eq!(scatter_item.input_bindings.len(), 1);
    assert_eq!(scatter_item.input_bindings[0].input_node, input.node);
    assert_eq!(
        scatter_item.input_bindings[0].desc.id,
        input.node.index() as u64
    );
    crate::PtxRenderer::new(80)
        .unwrap()
        .render(&scatter_item.kernel)
        .unwrap();
    crate::runtime::opencl::OpenClRenderer::default()
        .render(&scatter_item.kernel)
        .unwrap();
    crate::runtime::metal::MetalRenderer::new(
        8,
        crate::runtime::metal::MetalCapabilities {
            max_buffer_length: 1 << 20,
            unified_memory: true,
            family: "MockApple9".into(),
        },
    )
    .unwrap()
    .render(&scatter_item.kernel)
    .unwrap();
    crate::runtime::webgpu::WgslRenderer::new(
        8,
        crate::runtime::webgpu::WebGpuCapabilities {
            max_buffer_size: 1 << 20,
            max_storage_buffers_per_shader_stage: 8,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroups_per_dimension: 65_535,
            timestamp_query: false,
            shader_f16: false,
        },
    )
    .unwrap()
    .render(&scatter_item.kernel)
    .unwrap();
    let producer = schedule
        .items
        .iter()
        .find(|item| item.node == input.node)
        .expect("a rejected scalar-fusion rehearsal must retain the movement input producer");
    assert!(scatter_item.dependencies.contains(&producer.id));
    // The realized multi-root schedule is also the ownership regression for
    // reduction-epilogue selection: every computed load retained by a final
    // normalized kernel must have an earlier producer and dependency. This
    // specifically covers the Select-through-view source shared across the
    // source-composed SVD epilogues without relying on arena indices.
    for item in &schedule.items {
        for binding in &item.input_bindings {
            if item.external_materializations.contains(&binding.input_node)
                || matches!(
                    graph.op(binding.input_node),
                    Ok(Op::Input { .. } | Op::Constant(_))
                )
            {
                continue;
            }
            let producer = schedule
                .items
                .iter()
                .find(|candidate| {
                    candidate.id < item.id
                        && candidate
                            .outputs
                            .iter()
                            .any(|output| output.id == binding.desc.id)
                })
                .unwrap_or_else(|| {
                    panic!(
                        "computed binding {:?} for item {} must retain an earlier producer",
                        binding.input_node, item.id
                    )
                });
            assert!(item.dependencies.contains(&producer.id));
        }
    }
    crate::MemoryPlan::from_schedule(&schedule, &[loss, gradient], true).unwrap();

    let values = [3.0f32, 0.25, -0.5, 2.0];
    let realization_inputs = HashMap::from([("x".into(), f32_data([2, 2], values))]);
    let realized = crate::realize(
        &graph,
        &schedule,
        &[loss, gradient],
        &realization_inputs,
        RealizationPolicy::Interpreter,
    )
    .unwrap();
    let analytic = realized.outputs[1].to_vec_f64();
    let epsilon = 1.0e-3f32;
    let mut finite = Vec::new();
    for lane in 0..values.len() {
        let mut plus = values;
        plus[lane] += epsilon;
        let plus = crate::realize_graph(
            &graph,
            &[loss],
            &HashMap::from([("x".into(), f32_data([2, 2], plus))]),
            RealizationPolicy::Interpreter,
        )
        .unwrap()
        .outputs[0]
            .scalar_at(0)
            .as_f64();
        let mut minus = values;
        minus[lane] -= epsilon;
        let minus = crate::realize_graph(
            &graph,
            &[loss],
            &HashMap::from([("x".into(), f32_data([2, 2], minus))]),
            RealizationPolicy::Interpreter,
        )
        .unwrap()
        .outputs[0]
            .scalar_at(0)
            .as_f64();
        finite.push((plus - minus) / f64::from(2.0 * epsilon));
    }
    assert_close(&analytic, &finite, 2.0e-2);
}

#[test]
fn source_svd_cpu_covers_rectangular_batch_and_degenerate_reconstruction() {
    let tall_values = [3.0, 1.0, 0.0, 2.0, 1.0, -1.0];
    for full_matrices in [true, false] {
        let (graph, outputs) = numeric_svd_graph([3, 2], full_matrices);
        assert_numeric_svd(
            &graph,
            outputs,
            [3, 2],
            &tall_values,
            full_matrices,
            true,
            2.0e-3,
        );
    }

    let wide_values = [3.0, 0.0, 1.0, 0.0, 2.0, 1.0];
    for full_matrices in [true, false] {
        let (graph, outputs) = numeric_svd_graph([2, 3], full_matrices);
        assert_numeric_svd(
            &graph,
            outputs,
            [2, 3],
            &wide_values,
            full_matrices,
            true,
            2.0e-3,
        );
    }

    let (batched, outputs) = numeric_svd_graph([2, 2, 2], false);
    assert_numeric_svd(
        &batched,
        outputs,
        [2, 2, 2],
        &[3.0, 0.0, 0.0, 2.0, 4.0, 1.0, 1.0, 3.0],
        false,
        true,
        2.0e-3,
    );

    let (special, outputs) = numeric_svd_graph([2, 2], true);
    let identity_s = assert_numeric_svd(
        &special,
        outputs,
        [2, 2],
        &[1.0, 0.0, 0.0, 1.0],
        true,
        true,
        1.0e-4,
    );
    assert_close(&identity_s, &[1.0, 1.0], 1.0e-4);
    let zero_s = assert_numeric_svd(
        &special,
        outputs,
        [2, 2],
        &[0.0, 0.0, 0.0, 0.0],
        true,
        false,
        1.0e-4,
    );
    assert_close(&zero_s, &[0.0, 0.0], 1.0e-4);
    let rank_one_s = assert_numeric_svd(
        &special,
        outputs,
        [2, 2],
        &[1.0, 1.0, 2.0, 2.0],
        true,
        false,
        2.0e-3,
    );
    assert_close(&rank_one_s, &[10.0f64.sqrt(), 0.0], 2.0e-3);
}

#[test]
fn source_svd_rejects_invalid_descriptors_atomically() {
    let mut rank_one = Graph::new();
    let input = rank_one.input_dtype("x", [4], DType::F32);
    let before = rank_one.node_count();
    assert!(matches!(
        rank_one.svd_default(input),
        Err(Error::InvalidMatmul { .. })
    ));
    assert_eq!(rank_one.node_count(), before);

    let mut overflow = Graph::new();
    let input = overflow.input_dtype("x", [usize::MAX, 2], DType::F32);
    let before = overflow.node_count();
    assert!(matches!(
        overflow.svd(input, false),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), before);

    let mut unknown = Graph::new();
    let before = unknown.node_count();
    assert!(matches!(
        unknown.svd_default(NodeId::from_index(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(unknown.node_count(), before);
}

#[test]
fn nontrivial_source_svd_keeps_sort_capture_and_native_boundaries() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 2], DType::F32);
    let outputs = graph.svd_default(input).unwrap();
    let requested = [outputs.0, outputs.1, outputs.2];
    let schedule = crate::schedule_many(&graph, &requested).unwrap();
    let capture = CapturedSchedule::capture(&graph, &schedule, &requested).unwrap();
    assert!(matches!(
        capture.to_bytes(),
        Err(crate::ReplayError::Unsupported(_))
    ));

    let inputs = HashMap::from([("x".into(), f32_data([2, 2], [3.0, 0.0, 0.0, 2.0]))]);
    assert!(matches!(
        crate::realize_graph(
            &graph,
            &requested,
            &inputs,
            RealizationPolicy::CpuJit {
                fallback_to_interpreter: false,
            },
        ),
        Err(crate::RealizationError::Unsupported(reason))
            if reason.contains("sort pairs are CPU-interpreter only")
    ));
    assert!(
        crate::realize_graph(&graph, &requested, &inputs, RealizationPolicy::Interpreter,).is_ok()
    );
}
