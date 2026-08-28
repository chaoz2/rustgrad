use super::{shape::normalize_axes, Graph, NodeId, Op, RandomKind, RandomStream};
use crate::random::reserve;
use crate::{DType, Error, ExpandExtent, ReshapeExtent, Result, Scalar, Shape, TensorData};
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

#[derive(Default)]
struct StreamRegistry {
    seed: u64,
    counters: BTreeMap<u32, [u32; 2]>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, SplitSections};
    use std::collections::HashMap;

    fn execute(graph: &Graph, output: NodeId, input: TensorData) -> TensorData {
        CpuBackend
            .execute(graph, output, &HashMap::from([("x".into(), input)]))
            .unwrap()
    }

    #[test]
    fn chunk_matches_tinygrad_uneven_tail_and_preserves_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 5]);
        let outputs = graph.chunk(input, 3, -1).unwrap();
        assert_eq!(outputs.len(), 3);
        assert_eq!(
            outputs
                .iter()
                .map(|&output| graph.shape(output).unwrap().clone())
                .collect::<Vec<_>>(),
            vec![Shape::from([2, 2]), Shape::from([2, 2]), Shape::from([2, 1])]
        );
        let loss = graph.sum_all(outputs[1]).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 5], (0..10).map(|x| x as f32).collect()).unwrap();

        assert_eq!(
            execute(&graph, outputs[2], values.clone()),
            TensorData::new([2, 1], vec![4., 9.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 5], vec![0., 0., 1., 1., 0., 0., 0., 1., 1., 0.]).unwrap()
        );
    }

    #[test]
    fn chunk_of_a_zero_axis_returns_exactly_requested_empty_views() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 0]);
        let outputs = graph.chunk(input, 3, 1).unwrap();
        assert_eq!(outputs.len(), 3);
        for output in outputs {
            assert_eq!(graph.shape(output).unwrap(), &Shape::from([2, 0]));
        }
    }

    #[test]
    fn chunk_rejects_invalid_count_or_axis_without_graph_growth() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let node_count = graph.node_count();

        assert!(graph.chunk(input, 0, 0).is_err());
        assert_eq!(graph.node_count(), node_count);
        assert!(graph.chunk(input, 2, 2).is_err());
        assert_eq!(graph.node_count(), node_count);
    }

    #[test]
    fn triangular_helpers_match_tinygrad_diagonals_and_select_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let upper = graph.triu(input, 1).unwrap();
        let lower = graph.tril(input, -1).unwrap();
        let loss = graph.sum_all(upper).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();

        assert_eq!(
            execute(&graph, upper, values.clone()),
            TensorData::new([2, 3], vec![0., 2., 3., 0., 0., 6.]).unwrap()
        );
        assert_eq!(
            execute(&graph, lower, values.clone()),
            TensorData::new([2, 3], vec![0., 0., 0., 4., 0., 0.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 3], vec![0., 1., 1., 0., 0., 1.]).unwrap()
        );

        let mut empty = Graph::new();
        let input = empty.input_dtype("x", [2, 0], DType::I8);
        let output = empty.tril(input, 0).unwrap();
        assert_eq!(empty.dtype(output).unwrap(), DType::I8);
        assert!(execute(
            &empty,
            output,
            TensorData::from_scalars([2, 0], DType::I8, []).unwrap(),
        )
        .to_vec_f64()
        .is_empty());
    }

    #[test]
    fn triangular_helpers_preflight_rank_extent_and_diagonal_before_nodes() {
        let mut graph = Graph::new();
        let vector = graph.input("vector", [3]);
        let before = graph.node_count();
        assert!(graph.triu(vector, 0).is_err());
        assert_eq!(graph.node_count(), before);

        let overflow = graph.input("overflow", [usize::MAX, 2]);
        let before = graph.node_count();
        assert!(graph.tril(overflow, 0).is_err());
        assert_eq!(graph.node_count(), before);

        let matrix = graph.input("matrix", [2, 2]);
        let before = graph.node_count();
        assert!(graph.tril(matrix, i64::MAX).is_err());
        assert_eq!(graph.node_count(), before);
    }

    #[test]
    fn diagonal_matches_tinygrad_offset_signed_dimensions_and_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [3, 4]);
        let diagonal = graph.diagonal(input, 1, 0, 1).unwrap();
        let loss = graph.sum_all(diagonal).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new(
            [3, 4],
            (1..=12).map(|value| value as f32).collect(),
        )
        .unwrap();
        assert_eq!(
            execute(&graph, diagonal, values.clone()),
            TensorData::new([3], vec![2., 7., 12.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new(
                [3, 4],
                vec![0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
            )
            .unwrap()
        );

        let mut signed = Graph::new();
        let input = signed.input("x", [2, 2, 3]);
        let diagonal = signed.diagonal(input, 1, -1, -3).unwrap();
        assert_eq!(signed.shape(diagonal).unwrap(), &Shape::from([2, 1]));
        assert_eq!(
            execute(
                &signed,
                diagonal,
                TensorData::new([2, 2, 3], (0..12).map(|value| value as f32).collect()).unwrap(),
            ),
            TensorData::new([2, 1], vec![6., 9.]).unwrap()
        );

        let mut empty = Graph::new();
        let input = empty.input_dtype("x", [2, 3], DType::I8);
        let diagonal = empty.diagonal(input, 3, 0, 1).unwrap();
        assert_eq!(empty.shape(diagonal).unwrap(), &Shape::from([0]));
        assert_eq!(empty.dtype(diagonal).unwrap(), DType::I8);
    }

    #[test]
    fn diagonal_preflights_axes_offsets_and_extents_before_nodes() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let before = graph.node_count();
        assert!(graph.diagonal(input, 0, 0, 0).is_err());
        assert_eq!(graph.node_count(), before);
        assert!(graph.diagonal(input, 4, 0, 1).is_err());
        assert_eq!(graph.node_count(), before);
        assert!(graph.diagonal(input, 0, 2, 1).is_err());
        assert_eq!(graph.node_count(), before);

        let overflow = graph.input("overflow", [usize::MAX, 2]);
        let before = graph.node_count();
        assert!(graph.diagonal(overflow, 0, 0, 1).is_err());
        assert_eq!(graph.node_count(), before);
    }

    #[test]
    fn roll_matches_tinygrad_signed_shift_axis_dtype_and_vjp_contracts() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 4]);
        let rolled = graph.roll(input, -1, -1).unwrap();
        let loss = graph.sum_all(rolled).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 4], (1..=8).map(|value| value as f32).collect()).unwrap();
        assert_eq!(
            execute(&graph, rolled, values.clone()),
            TensorData::new([2, 4], vec![2., 3., 4., 1., 6., 7., 8., 5.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 4], vec![1.; 8]).unwrap()
        );

        let mut integer = Graph::new();
        let input = integer.input_dtype("x", [3], DType::I8);
        let rolled = integer.roll(input, 7, 0).unwrap();
        assert_eq!(integer.dtype(rolled).unwrap(), DType::I8);
        assert_eq!(
            execute(
                &integer,
                rolled,
                TensorData::from_scalars([3], DType::I8, [Scalar::I(1), Scalar::I(2), Scalar::I(3)])
                    .unwrap(),
            ),
            TensorData::from_scalars([3], DType::I8, [Scalar::I(3), Scalar::I(1), Scalar::I(2)])
                .unwrap()
        );

        let mut empty = Graph::new();
        let input = empty.input("x", [2, 0]);
        assert_eq!(empty.roll(input, i64::MIN, -1).unwrap(), input);
    }

    #[test]
    fn roll_preflights_scalar_axis_and_extent_before_nodes() {
        let mut graph = Graph::new();
        let scalar = graph.input("scalar", []);
        let before = graph.node_count();
        assert!(graph.roll(scalar, 1, 0).is_err());
        assert_eq!(graph.node_count(), before);

        let input = graph.input("input", [2, 3]);
        let before = graph.node_count();
        assert!(graph.roll(input, 1, 2).is_err());
        assert_eq!(graph.node_count(), before);

        let overflow = graph.input("overflow", [usize::MAX]);
        let before = graph.node_count();
        assert!(graph.roll(overflow, 1, 0).is_err());
        assert_eq!(graph.node_count(), before);
    }

    #[test]
    fn flattened_roll_matches_tinygrad_default_form_and_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let rolled = graph.roll_flattened(input, -1).unwrap();
        let loss = graph.sum_all(rolled).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 3], (1..=6).map(|value| value as f32).collect()).unwrap();
        assert_eq!(
            execute(&graph, rolled, values.clone()),
            TensorData::new([2, 3], vec![2., 3., 4., 5., 6., 1.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 3], vec![1.; 6]).unwrap()
        );

        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], DType::I8);
        assert_eq!(scalar.roll_flattened(input, i64::MIN).unwrap(), input);

        let mut empty = Graph::new();
        let input = empty.input_dtype("x", [0, 2], DType::F16);
        assert_eq!(empty.roll_flattened(input, i64::MAX).unwrap(), input);
    }

    #[test]
    fn flattened_roll_preflights_extent_before_nodes() {
        let mut graph = Graph::new();
        let overflow = graph.input("overflow", [usize::MAX, 2]);
        let before = graph.node_count();
        assert!(graph.roll_flattened(overflow, 1).is_err());
        assert_eq!(graph.node_count(), before);
    }

    #[test]
    fn flatten_matches_tinygrad_scalar_identity_and_signed_spans() {
        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], DType::F16);
        let flattened = scalar.flatten(input, 0, -1).unwrap();
        assert_eq!(scalar.shape(flattened).unwrap(), &Shape::from([1]));
        assert_eq!(scalar.dtype(flattened).unwrap(), DType::F16);

        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3, 4]);
        assert_eq!(graph.flatten(input, -2, -2).unwrap(), input);
        let flattened = graph.flatten(input, -3, -2).unwrap();
        assert_eq!(
            graph.shape(flattened).unwrap(),
            &Shape::from([6, 4])
        );
        let loss = graph.sum_all(flattened).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 3, 4], vec![1f32; 24]).unwrap();
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 3, 4], vec![1f32; 24]).unwrap()
        );
    }

    #[test]
    fn flatten_preflights_invalid_scalar_axes_and_extents() {
        let mut scalar = Graph::new();
        let input = scalar.input("x", []);
        let before = scalar.node_count();
        assert!(scalar.flatten(input, 1, 0).is_err());
        assert_eq!(scalar.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow.flatten(input, 0, 1).is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn squeeze_matches_tinygrad_scalar_and_identity_views() {
        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], DType::BF16);
        assert_eq!(scalar.squeeze(input, None).unwrap(), input);
        assert_eq!(scalar.squeeze(input, Some(-1)).unwrap(), input);
        assert_eq!(scalar.squeeze(input, Some(0)).unwrap(), input);

        let mut graph = Graph::new();
        let input = graph.input("x", [2, 0, 3]);
        assert_eq!(graph.squeeze(input, None).unwrap(), input);
        assert_eq!(graph.squeeze(input, Some(-1)).unwrap(), input);

        let mut singleton_graph = Graph::new();
        let singleton = singleton_graph.input("x", [2, 1, 3]);
        let squeezed = singleton_graph.squeeze(singleton, Some(-2)).unwrap();
        assert_eq!(singleton_graph.shape(squeezed).unwrap(), &Shape::from([2, 3]));
        let loss = singleton_graph.sum_all(squeezed).unwrap();
        let gradient = singleton_graph.grad(loss, singleton).unwrap();
        let values = TensorData::new([2, 1, 3], vec![1f32; 6]).unwrap();
        assert_eq!(
            execute(&singleton_graph, gradient, values),
            TensorData::new([2, 1, 3], vec![1f32; 6]).unwrap()
        );
    }

    #[test]
    fn squeeze_preflights_invalid_scalar_axis_and_extent() {
        let mut scalar = Graph::new();
        let input = scalar.input("x", []);
        let before = scalar.node_count();
        assert!(scalar.squeeze(input, Some(1)).is_err());
        assert_eq!(scalar.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow.squeeze(input, None).is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn unsqueeze_matches_tinygrad_single_signed_axis_and_vjp() {
        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], DType::F16);
        let trailing = scalar.unsqueeze(input, -1).unwrap();
        let leading = scalar.unsqueeze(input, 0).unwrap();
        assert_eq!(scalar.shape(trailing).unwrap(), &Shape::from([1]));
        assert_eq!(scalar.shape(leading).unwrap(), &Shape::from([1]));

        let mut graph = Graph::new();
        let input = graph.input("x", [2, 0, 3]);
        let unsqueezed = graph.unsqueeze(input, -2).unwrap();
        assert_eq!(graph.shape(unsqueezed).unwrap(), &Shape::from([2, 0, 1, 3]));
        let loss = graph.sum_all(unsqueezed).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        assert_eq!(
            execute(&graph, gradient, TensorData::new([2, 0, 3], Vec::<f32>::new()).unwrap()),
            TensorData::new([2, 0, 3], Vec::<f32>::new()).unwrap()
        );
    }

    #[test]
    fn unsqueeze_preflights_invalid_axis_and_extent() {
        let mut scalar = Graph::new();
        let input = scalar.input("x", []);
        let before = scalar.node_count();
        assert!(scalar.unsqueeze(input, 1).is_err());
        assert_eq!(scalar.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow.unsqueeze(input, 0).is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn permute_signed_matches_tinygrad_identity_scalar_and_vjp() {
        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], DType::F16);
        assert_eq!(scalar.permute_signed(input, Vec::<isize>::new()).unwrap(), input);

        let mut graph = Graph::new();
        let input = graph.input("x", [2, 2, 3]);
        assert_eq!(graph.permute_signed(input, [0, 1, 2]).unwrap(), input);
        let permuted = graph.permute_signed(input, [-1, -3, -2]).unwrap();
        assert_eq!(graph.shape(permuted).unwrap(), &Shape::from([3, 2, 2]));
        let loss = graph.sum_all(permuted).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 2, 3], vec![1f32; 12]).unwrap();
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 2, 3], vec![1f32; 12]).unwrap()
        );

        let repeated = graph.input("repeated", [2, 2]);
        assert_ne!(graph.permute_signed(repeated, [1, 0]).unwrap(), repeated);
    }

    #[test]
    fn permute_signed_preflights_invalid_axes_and_extents() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let before = graph.node_count();
        assert!(graph.permute_signed(input, [0, 0]).is_err());
        assert_eq!(graph.node_count(), before);
        assert!(graph.permute_signed(input, [isize::MIN, 1]).is_err());
        assert_eq!(graph.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow.permute_signed(input, [1, 0]).is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn transpose_matches_tinygrad_defaults_equal_axes_and_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 2]);
        let transposed = graph.transpose_default(input).unwrap();
        assert_ne!(transposed, input);
        assert_eq!(graph.shape(transposed).unwrap(), &Shape::from([2, 2]));
        assert_eq!(graph.transpose(input, -1, -1).unwrap(), input);
        let loss = graph.sum_all(transposed).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 2], vec![1f32; 4]).unwrap();
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 2], vec![1f32; 4]).unwrap()
        );
    }

    #[test]
    fn transpose_default_preflights_rank_and_extent() {
        let mut vector = Graph::new();
        let input = vector.input("x", [2]);
        let before = vector.node_count();
        assert!(vector.transpose_default(input).is_err());
        assert_eq!(vector.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow.transpose_default(input).is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn reshape_with_extents_matches_tinygrad_infer_copy_identity_and_vjp() {
        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], DType::BF16);
        let reshaped = scalar.reshape_with_extents(input, [ReshapeExtent::Infer]).unwrap();
        assert_eq!(scalar.shape(reshaped).unwrap(), &Shape::from([1]));

        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3], DType::F16);
        assert_eq!(
            graph
                .reshape_with_extents(input, [ReshapeExtent::Copy, ReshapeExtent::Copy])
                .unwrap(),
            input
        );
        let reshaped = graph
            .reshape_with_extents(input, [ReshapeExtent::Exact(3), ReshapeExtent::Infer])
            .unwrap();
        assert_eq!(graph.shape(reshaped).unwrap(), &Shape::from([3, 2]));
        let loss = graph.sum_all(reshaped).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 3], vec![1f32; 6]).unwrap();
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 3], vec![1f32; 6]).unwrap()
        );

        let mut empty = Graph::new();
        let input = empty.input("x", [0, 3]);
        let reshaped = empty
            .reshape_with_extents(input, [ReshapeExtent::Exact(3), ReshapeExtent::Infer])
            .unwrap();
        assert_eq!(empty.shape(reshaped).unwrap(), &Shape::from([3, 0]));
    }

    #[test]
    fn reshape_with_extents_preflights_source_errors_without_nodes() {
        let mut graph = Graph::new();
        let input = graph.input("x", [0, 3]);
        let before = graph.node_count();
        assert!(graph
            .reshape_with_extents(input, [ReshapeExtent::Exact(0), ReshapeExtent::Infer])
            .is_err());
        assert_eq!(graph.node_count(), before);
        assert!(graph
            .reshape_with_extents(input, [ReshapeExtent::Copy, ReshapeExtent::Copy, ReshapeExtent::Copy])
            .is_err());
        assert_eq!(graph.node_count(), before);
        assert!(graph
            .reshape_with_extents(input, [ReshapeExtent::Infer, ReshapeExtent::Infer])
            .is_err());
        assert_eq!(graph.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow
            .reshape_with_extents(input, [ReshapeExtent::Infer])
            .is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn expand_with_extents_matches_tinygrad_copy_alignment_identity_and_vjp() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [1, 3], DType::F16);
        assert_eq!(graph.expand(input, [3]).unwrap(), input);
        assert_eq!(
            graph
                .expand_with_extents(input, [ExpandExtent::Copy])
                .unwrap(),
            input
        );
        let expanded = graph
            .expand_with_extents(input, [ExpandExtent::Exact(2), ExpandExtent::Copy])
            .unwrap();
        assert_eq!(graph.shape(expanded).unwrap(), &Shape::from([2, 3]));
        let loss = graph.sum_all(expanded).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([1, 3], vec![1f32; 3]).unwrap();
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([1, 3], vec![2f32; 3]).unwrap()
        );

        let mut empty = Graph::new();
        let input = empty.input("x", [1, 0]);
        let expanded = empty.expand_with_extents(input, [ExpandExtent::Exact(2), ExpandExtent::Copy]).unwrap();
        assert_eq!(empty.shape(expanded).unwrap(), &Shape::from([2, 0]));
    }

    #[test]
    fn expand_preflights_invalid_broadcast_and_extent() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let before = graph.node_count();
        assert!(graph.expand(input, [3]).is_err());
        assert_eq!(graph.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow
            .expand_with_extents(input, [ExpandExtent::Copy, ExpandExtent::Copy])
            .is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn split_preserves_explicit_sections_uniform_tails_and_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 5]);
        let explicit = graph
            .split(input, SplitSections::Explicit(vec![1, 3, 1]), -1)
            .unwrap();
        let uniform = graph.split(input, SplitSections::Uniform(2), 1).unwrap();
        assert_eq!(explicit.len(), 3);
        assert_eq!(uniform.len(), 3);
        assert_eq!(
            explicit
                .iter()
                .map(|&output| graph.shape(output).unwrap().clone())
                .collect::<Vec<_>>(),
            vec![Shape::from([2, 1]), Shape::from([2, 3]), Shape::from([2, 1])]
        );
        let loss = graph.sum_all(explicit[1]).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 5], (0..10).map(|x| x as f32).collect()).unwrap();

        assert_eq!(
            execute(&graph, uniform[2], values.clone()),
            TensorData::new([2, 1], vec![4., 9.]).unwrap()
        );
        assert_eq!(
            execute(&graph, explicit[1], values.clone()),
            TensorData::new([2, 3], vec![1., 2., 3., 6., 7., 8.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 5], vec![0., 1., 1., 1., 0., 0., 1., 1., 1., 0.]).unwrap()
        );
    }

    #[test]
    fn split_preserves_tinygrad_zero_axis_forms() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 0]);
        let uniform = graph.split(input, SplitSections::Uniform(0), 1).unwrap();
        let explicit = graph
            .split(input, SplitSections::Explicit(vec![0, 0]), 1)
            .unwrap();
        assert_eq!(uniform.len(), 1);
        assert_eq!(explicit.len(), 2);
        for output in uniform.into_iter().chain(explicit) {
            assert_eq!(graph.shape(output).unwrap(), &Shape::from([2, 0]));
        }
    }

    #[test]
    fn split_rejects_bad_sections_before_graph_growth() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 5]);
        let node_count = graph.node_count();

        assert!(graph
            .split(input, SplitSections::Uniform(0), 1)
            .is_err());
        assert_eq!(graph.node_count(), node_count);
        assert!(graph
            .split(input, SplitSections::Explicit(vec![2, 2]), 1)
            .is_err());
        assert_eq!(graph.node_count(), node_count);
        assert!(graph
            .split(input, SplitSections::Explicit(vec![usize::MAX, 1]), 1)
            .is_err());
        assert_eq!(graph.node_count(), node_count);
        assert!(graph
            .split(input, SplitSections::Uniform(1), isize::MIN)
            .is_err());
        assert_eq!(graph.node_count(), node_count);
    }

    #[test]
    fn flip_uses_signed_axes_and_preserves_stride_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let flipped = graph.flip(input, [0isize, -1]).unwrap();
        let selected = graph.shrink(flipped, [(0, 1), (0, 2)]).unwrap();
        let loss = graph.sum_all(selected).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();

        assert_eq!(
            execute(&graph, flipped, values.clone()),
            TensorData::new([2, 3], vec![6., 5., 4., 3., 2., 1.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 3], vec![0., 0., 0., 0., 1., 1.]).unwrap()
        );
    }

    #[test]
    fn flip_empty_axes_is_a_scalar_noop_and_bad_axes_do_not_grow_the_graph() {
        let mut graph = Graph::new();
        let scalar = graph.input("scalar", []);
        let node_count = graph.node_count();
        assert_eq!(graph.flip(scalar, Vec::<isize>::new()).unwrap(), scalar);
        assert_eq!(graph.node_count(), node_count);

        let input = graph.input("x", [2, 3]);
        let node_count = graph.node_count();
        assert!(graph.flip(input, [1isize, -1]).is_err());
        assert_eq!(graph.node_count(), node_count);
        assert!(graph.flip(input, [isize::MIN]).is_err());
        assert_eq!(graph.node_count(), node_count);
    }

    #[test]
    fn stack_preflights_all_inputs_before_constructing_unsqueezes() {
        let mut graph = Graph::new();
        let left = graph.input("left", [2]);
        let right = graph.input("right", [3]);
        let node_count = graph.node_count();

        assert!(graph.stack([left, right], 0).is_err());
        assert_eq!(graph.node_count(), node_count);

        let first = graph.input("first", [2]);
        let second = graph.input("second", [2]);
        let stacked = graph.stack([first, second], -1).unwrap();
        let loss = graph.sum_all(stacked).unwrap();
        let gradient = graph.grad(loss, first).unwrap();
        assert_eq!(graph.shape(stacked).unwrap(), &Shape::from([2, 2]));
        let bindings = HashMap::from([
            ("left".into(), TensorData::new([2], vec![0., 0.]).unwrap()),
            ("right".into(), TensorData::new([3], vec![0., 0., 0.]).unwrap()),
            ("first".into(), TensorData::new([2], vec![1., 2.]).unwrap()),
            ("second".into(), TensorData::new([2], vec![3., 4.]).unwrap()),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, stacked, &bindings).unwrap(),
            TensorData::new([2, 2], vec![1., 3., 2., 4.]).unwrap()
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &bindings).unwrap(),
            TensorData::new([2], vec![1., 1.]).unwrap()
        );
    }
}

static STREAM_REGISTRY: OnceLock<Mutex<StreamRegistry>> = OnceLock::new();

fn stream_registry() -> &'static Mutex<StreamRegistry> {
    STREAM_REGISTRY.get_or_init(|| Mutex::new(StreamRegistry::default()))
}

fn stream_words(shape: &Shape, dtype: DType, multiplier: usize) -> Result<u64> {
    let elements = shape
        .numel()?
        .checked_mul(multiplier)
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    let bytes = elements
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    Ok(bytes.div_ceil(4) as u64)
}

fn checked_initializer_tail_fan(shape: &Shape) -> Result<usize> {
    shape.dims()[1..].iter().try_fold(1usize, |fan, &dimension| {
        fan.checked_mul(dimension)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    })
}

fn reserve_implicit_stream(device: u32, words: u64) -> RandomStream {
    // A mutex deliberately serializes implicit construction. Every node stores
    // the reservation it received, so later execution is schedule-independent.
    let mut registry = stream_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let start = reserve(registry.counters.entry(device).or_insert([0, 0]), words);
    RandomStream {
        device,
        // This is SHA256(0u32-be) narrowed to U32, matching tinygrad's first
        // device key. Further numeric devices use a deterministic distinct
        // derivation until RustGrad grows canonical backend device names.
        key: [device_key(device), registry.seed as u32],
        counter: start,
    }
}

fn device_key(device: u32) -> u32 {
    if device == 0 {
        0x14B8_1119
    } else {
        device.wrapping_mul(0x9E37_79B9).rotate_left(13) ^ 0xA5A5_5A5A
    }
}

impl Graph {
    pub fn unsqueeze(&mut self, input: NodeId, axis: isize) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let dtype = self.dtype(input)?;
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let mut dims = shape.dims().to_vec();
        let rank = isize::try_from(dims.len())
            .ok()
            .and_then(|rank| rank.checked_add(1))
            .ok_or(Error::InvalidAxis {
                node: input,
                axis: usize::MAX,
                rank: usize::MAX,
            })?;
        let axis = if axis < 0 {
            axis.checked_add(rank).ok_or(Error::InvalidAxis {
                node: input,
                axis: usize::MAX,
                rank: rank as usize,
            })?
        } else {
            axis
        };
        if axis < 0 || axis >= rank {
            return Err(Error::InvalidAxis {
                node: input,
                axis: usize::MAX,
                rank: rank as usize,
            });
        }
        dims.insert(axis as usize, 1);
        let output_shape = Shape::new(dims);
        output_shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        self.reshape(input, output_shape)
    }

    pub fn squeeze(&mut self, input: NodeId, axis: Option<isize>) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let dtype = self.dtype(input)?;
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let mut dims = shape.dims().to_vec();
        if let Some(axis) = axis {
            // Tensor._resolve_dim accepts -1 and 0 for scalars, and the
            // explicit scalar path is a no-op.
            if dims.is_empty() {
                if matches!(axis, -1 | 0) {
                    return Ok(input);
                }
                return Err(Error::InvalidRandom {
                    reason: "invalid squeeze axis",
                });
            }
            let rank = isize::try_from(dims.len()).map_err(|_| Error::InvalidRandom {
                reason: "invalid squeeze axis",
            })?;
            let axis = if axis < 0 {
                axis.checked_add(rank).ok_or(Error::InvalidRandom {
                    reason: "invalid squeeze axis",
                })?
            } else {
                axis
            };
            if axis < 0 || axis >= rank {
                return Err(Error::InvalidRandom {
                    reason: "invalid squeeze axis",
                });
            }
            if dims[axis as usize] != 1 {
                return Ok(input);
            }
            dims.remove(axis as usize);
        } else {
            dims.retain(|dim| *dim != 1);
        }
        let output_shape = Shape::new(dims);
        output_shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        // Tensor.reshape returns self for both non-singleton explicit axes
        // and all-axis squeezes that leave the shape unchanged.
        if output_shape == shape {
            Ok(input)
        } else {
            self.reshape(input, output_shape)
        }
    }

    pub fn flatten(&mut self, input: NodeId, start: isize, end: isize) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let dtype = self.dtype(input)?;
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let invalid = || Error::InvalidRandom {
            reason: "invalid flatten dimensions",
        };
        let rank = isize::try_from(shape.rank()).map_err(|_| invalid())?;
        // tinygrad resolves scalar dimensions against `max(1, ndim)`: every
        // accepted scalar span is empty and therefore reshapes `[]` to `[1]`.
        let output_shape = if rank == 0 {
            if !matches!(start, -1 | 0) || !matches!(end, -1 | 0) {
                return Err(invalid());
            }
            Shape::new([1])
        } else {
            let start = if start < 0 {
                start.checked_add(rank).ok_or_else(invalid)?
            } else {
                start
            };
            let end = if end < 0 {
                end.checked_add(rank).ok_or_else(invalid)?
            } else {
                end
            };
            if start < 0 || end < start || end >= rank {
                return Err(invalid());
            }
            let mut dims = shape.dims()[..start as usize].to_vec();
            dims.push(
                shape.dims()[start as usize..=end as usize]
                    .iter()
                    .try_fold(1usize, |n, d| n.checked_mul(*d))
                    .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?,
            );
            dims.extend_from_slice(&shape.dims()[end as usize + 1..]);
            Shape::new(dims)
        };
        output_shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        // Tensor.reshape returns self when the view leaves the shape unchanged.
        if output_shape == shape {
            Ok(input)
        } else {
            self.reshape(input, output_shape)
        }
    }

    pub fn stack(&mut self, inputs: impl Into<Vec<NodeId>>, axis: isize) -> Result<NodeId> {
        let inputs = inputs.into();
        if inputs.is_empty() {
            return Err(Error::InvalidRandom {
                reason: "stack requires at least one tensor",
            });
        }
        let shapes = inputs
            .iter()
            .map(|&input| Ok(self.shape(input)?.clone()))
            .collect::<Result<Vec<_>>>()?;
        let rank = shapes[0].rank() as isize + 1;
        let axis = if axis < 0 {
            axis.checked_add(rank).ok_or(Error::InvalidAxis {
                node: inputs[0],
                axis: usize::MAX,
                rank: rank as usize,
            })?
        } else {
            axis
        };
        if axis < 0 || axis >= rank {
            return Err(Error::InvalidAxis {
                node: inputs[0],
                axis: usize::MAX,
                rank: rank as usize,
            });
        }
        if shapes.iter().any(|shape| shape != &shapes[0]) {
            return Err(Error::InvalidConcat {
                axis: axis as usize,
                shapes,
            });
        }
        let mut expanded = Vec::with_capacity(inputs.len());
        for input in inputs {
            expanded.push(self.unsqueeze(input, axis)?);
        }
        self.concat(expanded, axis as usize)
    }

    pub fn one_hot(&mut self, input: NodeId, classes: usize) -> Result<NodeId> {
        if !self.dtype(input)?.is_integer() {
            return Err(Error::InvalidRandom {
                reason: "one_hot requires integer indices",
            });
        }
        let class_end = i64::try_from(classes).map_err(|_| Error::InvalidRandom {
            reason: "one_hot class count exceeds the supported i64 range",
        })?;
        let mut value_dims = self.shape(input)?.dims().to_vec();
        value_dims.push(1);
        let value_shape = Shape::new(value_dims.clone());
        value_shape.numel()?;
        let mut class_dims = vec![1; value_dims.len()];
        *class_dims.last_mut().unwrap() = classes;
        let class_shape = Shape::new(class_dims);
        class_shape.numel()?;
        let mut output_dims = self.shape(input)?.dims().to_vec();
        output_dims.push(classes);
        Shape::new(output_dims).numel()?;

        let values = self.reshape(input, value_shape)?;
        let classes_node = self.arange(0, class_end, 1)?;
        let classes_node = self.reshape(classes_node, class_shape)?;
        let equal = self.eq(values, classes_node)?;
        let one = self.constant(TensorData::scalar_with_dtype(Scalar::I(1), DType::I32));
        let zero = self.constant(TensorData::scalar_with_dtype(Scalar::I(0), DType::I32));
        self.select(equal, one, zero)
    }

    pub fn meshgrid(
        &mut self,
        inputs: impl Into<Vec<NodeId>>,
        indexing: &str,
    ) -> Result<Vec<NodeId>> {
        let inputs = inputs.into();
        if !(indexing == "ij" || indexing == "xy") {
            return Err(Error::InvalidRandom {
                reason: "meshgrid indexing must be ij or xy",
            });
        }
        if inputs.len() <= 1 {
            return Ok(inputs);
        }
        let mut lengths = Vec::new();
        for input in &inputs {
            let shape = self.shape(*input)?;
            if shape.rank() > 1 {
                return Err(Error::InvalidRandom {
                    reason: "meshgrid inputs must be scalars or vectors",
                });
            }
            lengths.push(if shape.rank() == 0 {
                1
            } else {
                shape.dims()[0]
            });
        }
        let mut output = lengths.clone();
        if indexing == "xy" {
            output.swap(0, 1);
        }
        inputs
            .into_iter()
            .enumerate()
            .map(|(index, input)| {
                let axis = if indexing == "xy" && index < 2 {
                    1 - index
                } else {
                    index
                };
                let mut shape = vec![1; output.len()];
                shape[axis] = lengths[index];
                let input = if self.shape(input)?.rank() == 0 {
                    self.unsqueeze(input, 0)?
                } else {
                    input
                };
                let input = self.reshape(input, Shape::new(shape))?;
                self.expand(input, Shape::new(output.clone()))
            })
            .collect()
    }
    /// Resets all implicit per-device Threefry streams. Existing graph nodes
    /// retain their captured reservations; only subsequently constructed nodes
    /// observe the new sequence.
    pub fn manual_seed(seed: u64) {
        let mut registry = stream_registry()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        registry.seed = seed;
        registry.counters.clear();
    }
    pub fn full(&mut self, shape: impl Into<Shape>, value: f32) -> Result<NodeId> {
        Ok(self.constant(TensorData::full(shape, value)?))
    }

    pub fn full_with_dtype(
        &mut self,
        shape: impl Into<Shape>,
        value: Scalar,
        dtype: DType,
    ) -> Result<NodeId> {
        Ok(self.constant(TensorData::full_with_dtype(shape, value, dtype)?))
    }

    pub fn zeros(&mut self, shape: impl Into<Shape>) -> Result<NodeId> {
        Ok(self.constant(TensorData::zeros(shape)?))
    }

    pub fn zeros_with_dtype(&mut self, shape: impl Into<Shape>, dtype: DType) -> Result<NodeId> {
        Ok(self.constant(TensorData::zeros_with_dtype(shape, dtype)?))
    }

    pub fn ones(&mut self, shape: impl Into<Shape>) -> Result<NodeId> {
        Ok(self.constant(TensorData::ones(shape)?))
    }

    pub fn arange(&mut self, start: i64, end: i64, step: i64) -> Result<NodeId> {
        Ok(self.constant(TensorData::arange(start, end, step)?))
    }

    pub fn empty(&mut self, shape: impl Into<Shape>, dtype: DType) -> Result<NodeId> {
        Ok(self.constant(TensorData::empty(shape, dtype)?))
    }

    pub fn linspace(
        &mut self,
        start: f64,
        stop: f64,
        steps: isize,
        dtype: DType,
    ) -> Result<NodeId> {
        Ok(self.constant(TensorData::linspace(start, stop, steps, dtype)?))
    }

    pub fn eye(&mut self, rows: usize, columns: Option<usize>, dtype: DType) -> Result<NodeId> {
        Ok(self.constant(TensorData::eye(rows, columns, dtype)?))
    }

    /// Returns the upper triangular part of `input`, zeroing entries below
    /// `diagonal` in its final two dimensions.
    pub fn triu(&mut self, input: NodeId, diagonal: i64) -> Result<NodeId> {
        self.triangular(input, diagonal, false)
    }

    /// Returns the lower triangular part of `input`, zeroing entries above
    /// `diagonal` in its final two dimensions.
    pub fn tril(&mut self, input: NodeId, diagonal: i64) -> Result<NodeId> {
        self.triangular(input, diagonal, true)
    }

    /// The shared checked `Tensor._tri(...).where(...)` composition used by
    /// tinygrad's public triangular helpers. Every rank, index extent,
    /// diagonal shift, and broadcast is validated before this appends its I64
    /// index constants, comparison, zero, or select nodes.
    fn triangular(&mut self, input: NodeId, diagonal: i64, lower: bool) -> Result<NodeId> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        shape.numel()?;
        if shape.rank() < 2 {
            return Err(Error::InvalidMovementRank {
                op: "triangular",
                expected: 2,
                actual: shape.rank(),
            });
        }
        let rows = shape.dims()[shape.rank() - 2];
        let columns = shape.dims()[shape.rank() - 1];
        let rows_i64 = i64::try_from(rows).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        let columns_i64 =
            i64::try_from(columns).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        let shift = if lower {
            diagonal
                .checked_add(1)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?
        } else {
            diagonal
        };
        if rows != 0 {
            (rows_i64 - 1)
                .checked_add(shift)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        }
        let mask_shape = Shape::new([rows, columns]);
        mask_shape.numel()?;
        if mask_shape.broadcast_with(&shape).as_ref() != Ok(&shape) {
            return Err(Error::InvalidExpand {
                from: mask_shape,
                to: shape,
            });
        }

        let row = self.reshape(self.arange(0, rows_i64, 1)?, Shape::new([rows, 1]))?;
        let column = self.reshape(
            self.arange(0, columns_i64, 1)?,
            Shape::new([1, columns]),
        )?;
        let shift = self.full_with_dtype([], Scalar::I(shift), DType::I64)?;
        let outside = self.le(self.add(row, shift)?, column)?;
        let zero = self.zeros_with_dtype(shape, dtype)?;
        if lower {
            self.select(outside, zero, input)
        } else {
            self.select(outside, input, zero)
        }
    }

    /// Extracts an offset diagonal from two signed dimensions.
    ///
    /// This is tinygrad's movement-only `diagonal(offset, dim1, dim2)`
    /// composition. Axis normalization, crop bounds, every intermediate
    /// extent, and the final output shape are checked before it appends a
    /// permutation, movement node, or zero pad.
    pub fn diagonal(
        &mut self,
        input: NodeId,
        offset: i64,
        dim1: isize,
        dim2: isize,
    ) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        shape.numel()?;
        let rank = shape.rank();
        let dim1 = normalize_axes(input, rank, Some(vec![dim1]))?[0];
        let dim2 = normalize_axes(input, rank, Some(vec![dim2]))?[0];
        if dim1 == dim2 {
            return Err(Error::InvalidRandom {
                reason: "diagonal dimensions must differ",
            });
        }
        let rows = shape.dims()[dim1];
        let columns = shape.dims()[dim2];
        let (row_start, column_start) = if offset >= 0 {
            let column_start =
                usize::try_from(offset).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
            if column_start > columns {
                return Err(Error::InvalidBounds {
                    axis: dim2,
                    start: column_start,
                    end: columns,
                    dim: columns,
                });
            }
            (0, column_start)
        } else {
            let row_start = offset
                .checked_neg()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            if row_start > rows {
                return Err(Error::InvalidBounds {
                    axis: dim1,
                    start: row_start,
                    end: rows,
                    dim: rows,
                });
            }
            (row_start, 0)
        };
        let cropped_rows = rows - row_start;
        let cropped_columns = columns - column_start;
        let diagonal_extent = cropped_rows.min(cropped_columns);
        let mut order = (0..rank)
            .filter(|&axis| axis != dim1 && axis != dim2)
            .collect::<Vec<_>>();
        let leading_dims = order
            .iter()
            .map(|&axis| shape.dims()[axis])
            .collect::<Vec<_>>();
        order.extend([dim1, dim2]);

        let mut cropped_dims = leading_dims.clone();
        cropped_dims.extend([cropped_rows, cropped_columns]);
        Shape::new(cropped_dims).numel()?;
        let mut output_dims = leading_dims.clone();
        output_dims.push(diagonal_extent);
        let output_shape = Shape::new(output_dims);
        output_shape.numel()?;

        let unflatten_shape = if diagonal_extent == 0 {
            None
        } else {
            let square_extent = diagonal_extent
                .checked_mul(diagonal_extent)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            let padded_extent = square_extent
                .checked_add(diagonal_extent)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            let diagonal_plus_one = diagonal_extent
                .checked_add(1)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            let mut padded_dims = leading_dims.clone();
            padded_dims.push(padded_extent);
            Shape::new(padded_dims).numel()?;
            let mut unflatten_dims = leading_dims.clone();
            unflatten_dims.extend([diagonal_extent, diagonal_plus_one]);
            let unflatten_shape = Shape::new(unflatten_dims);
            unflatten_shape.numel()?;
            Some(unflatten_shape)
        };

        let permuted = self.permute(input, order)?;
        let mut crop_bounds = leading_dims
            .iter()
            .map(|&extent| (0, extent))
            .collect::<Vec<_>>();
        crop_bounds.extend([(row_start, rows), (column_start, columns)]);
        let cropped = self.shrink(permuted, crop_bounds)?;
        if diagonal_extent == 0 {
            return self.reshape(cropped, output_shape);
        }

        let mut square_bounds = leading_dims
            .iter()
            .map(|&extent| (0, extent))
            .collect::<Vec<_>>();
        square_bounds.extend([(0, diagonal_extent), (0, diagonal_extent)]);
        let square = self.shrink(cropped, square_bounds)?;
        let flattened = self.flatten(square, -2, -1)?;
        let mut padding = vec![(0, 0); leading_dims.len()];
        padding.push((0, diagonal_extent));
        let padded = self.pad(flattened, padding, Scalar::I(0))?;
        let unflattened = self.reshape(
            padded,
            unflatten_shape.expect("nonempty diagonal has a checked unflatten shape"),
        )?;
        let mut diagonal_bounds = leading_dims
            .iter()
            .map(|&extent| (0, extent))
            .collect::<Vec<_>>();
        diagonal_bounds.extend([(0, diagonal_extent), (0, 1)]);
        let diagonal = self.shrink(unflattened, diagonal_bounds)?;
        self.squeeze(diagonal, Some(-1))
    }

    /// Circularly rolls `input` by a signed shift along one signed axis.
    ///
    /// This is the one-axis branch of tinygrad's public `roll` helper. Its
    /// signed axis, empty-tensor no-op, and Euclidean shift normalization are
    /// resolved before the two source views or their concat are appended.
    pub fn roll(&mut self, input: NodeId, shift: i64, axis: isize) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        shape.numel()?;
        if shape.rank() == 0 {
            return Err(Error::InvalidMovementRank {
                op: "roll",
                expected: 1,
                actual: 0,
            });
        }
        let axis = normalize_axes(input, shape.rank(), Some(vec![axis]))?[0];
        if shape.dims().contains(&0) {
            return Ok(input);
        }
        let extent = shape.dims()[axis];
        let extent_i64 = i64::try_from(extent).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        let normalized = shift.rem_euclid(extent_i64) as usize;
        if normalized == 0 {
            return Ok(input);
        }
        let split = extent - normalized;
        let tail = shape
            .dims()
            .iter()
            .enumerate()
            .map(|(dimension, &size)| {
                if dimension == axis {
                    (split, size)
                } else {
                    (0, size)
                }
            })
            .collect::<Vec<_>>();
        let head = shape
            .dims()
            .iter()
            .enumerate()
            .map(|(dimension, &size)| {
                if dimension == axis {
                    (0, split)
                } else {
                    (0, size)
                }
            })
            .collect::<Vec<_>>();

        let tail = self.shrink(input, tail)?;
        let head = self.shrink(input, head)?;
        self.concat(vec![tail, head], axis)
    }

    /// Circularly rolls the flattened logical tensor, then restores its shape.
    ///
    /// This is tinygrad's public `roll(shifts)` form with `dims=None`, kept
    /// distinct from the explicit-axis API. Its flattened extent and signed
    /// shift are checked before flattening can append a movement node.
    pub fn roll_flattened(&mut self, input: NodeId, shift: i64) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let elements = shape.numel()?;
        if shape.rank() == 0 || elements == 0 {
            return Ok(input);
        }
        let elements_i64 =
            i64::try_from(elements).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        if shift.rem_euclid(elements_i64) == 0 {
            return Ok(input);
        }
        let end = isize::try_from(shape.rank() - 1)
            .map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        let flattened = self.flatten(input, 0, end)?;
        let rolled = self.roll(flattened, shift, 0)?;
        self.reshape(rolled, shape)
    }

    /// Uniform `[0, 1)` values from an explicit Threefry stream key.
    pub fn rand(&mut self, shape: impl Into<Shape>, dtype: DType, seed: u64) -> Result<NodeId> {
        if !dtype.is_float() {
            return Err(Error::InvalidRandom {
                reason: "rand requires a floating point dtype",
            });
        }
        self.uniform(shape, 0.0, 1.0, dtype, seed)
    }

    pub fn rand_implicit(&mut self, shape: impl Into<Shape>, dtype: DType) -> Result<NodeId> {
        self.rand_implicit_on_device(shape, dtype, 0)
    }

    /// Implicit `rand` from an isolated numeric device stream. Device `0` is
    /// the CPU-compatible default; accelerator lowering is not implemented.
    pub fn rand_implicit_on_device(
        &mut self,
        shape: impl Into<Shape>,
        dtype: DType,
        device: u32,
    ) -> Result<NodeId> {
        if !dtype.is_float() {
            return Err(Error::InvalidRandom {
                reason: "rand requires a floating point dtype",
            });
        }
        let shape = shape.into();
        let stream = reserve_implicit_stream(device, stream_words(&shape, dtype, 1)?);
        self.random_stream(
            shape,
            dtype,
            RandomKind::Uniform {
                low: 0.0,
                high: 1.0,
            },
            stream,
        )
    }
    pub fn randn_implicit(&mut self, shape: impl Into<Shape>, dtype: DType) -> Result<NodeId> {
        self.randn_implicit_on_device(shape, dtype, 0)
    }

    pub fn randn_implicit_on_device(
        &mut self,
        shape: impl Into<Shape>,
        dtype: DType,
        device: u32,
    ) -> Result<NodeId> {
        if !dtype.is_float() {
            return Err(Error::InvalidRandom {
                reason: "normal requires a floating point dtype",
            });
        }
        let shape = shape.into();
        // tinygrad's Box-Muller path consumes two F32 uniforms per output.
        let stream = reserve_implicit_stream(device, stream_words(&shape, DType::F32, 2)?);
        self.random_stream(
            shape,
            dtype,
            RandomKind::Normal {
                mean: 0.0,
                std: 1.0,
            },
            stream,
        )
    }

    pub fn uniform(
        &mut self,
        shape: impl Into<Shape>,
        low: f64,
        high: f64,
        dtype: DType,
        seed: u64,
    ) -> Result<NodeId> {
        if !(low.is_finite() && high.is_finite() && low < high) {
            return Err(Error::InvalidRandom {
                reason: "uniform requires finite low < high",
            });
        }
        self.random(shape.into(), dtype, RandomKind::Uniform { low, high }, seed)
    }

    pub fn randn(&mut self, shape: impl Into<Shape>, dtype: DType, seed: u64) -> Result<NodeId> {
        self.normal(shape, 0.0, 1.0, dtype, seed)
    }

    pub fn normal(
        &mut self,
        shape: impl Into<Shape>,
        mean: f64,
        std: f64,
        dtype: DType,
        seed: u64,
    ) -> Result<NodeId> {
        if !dtype.is_float() {
            return Err(Error::InvalidRandom {
                reason: "normal requires a floating point dtype",
            });
        }
        if !(mean.is_finite() && std.is_finite() && std >= 0.0) {
            return Err(Error::InvalidRandom {
                reason: "normal requires finite mean and non-negative std",
            });
        }
        self.random(shape.into(), dtype, RandomKind::Normal { mean, std }, seed)
    }

    pub fn randint(
        &mut self,
        shape: impl Into<Shape>,
        low: i64,
        high: i64,
        dtype: DType,
        seed: u64,
    ) -> Result<NodeId> {
        if !dtype.is_integer() {
            return Err(Error::InvalidRandom {
                reason: "randint requires an integer dtype",
            });
        }
        if low >= high {
            return Err(Error::InvalidRandom {
                reason: "randint requires low < high",
            });
        }
        if high.checked_sub(low).is_none() {
            return Err(Error::InvalidRandom {
                reason: "randint range overflows i64",
            });
        }
        self.random(shape.into(), dtype, RandomKind::RandInt { low, high }, seed)
    }

    pub fn randint_implicit(
        &mut self,
        shape: impl Into<Shape>,
        low: i64,
        high: i64,
        dtype: DType,
    ) -> Result<NodeId> {
        self.randint_implicit_on_device(shape, low, high, dtype, 0)
    }

    pub fn randint_implicit_on_device(
        &mut self,
        shape: impl Into<Shape>,
        low: i64,
        high: i64,
        dtype: DType,
        device: u32,
    ) -> Result<NodeId> {
        if !dtype.is_integer() {
            return Err(Error::InvalidRandom {
                reason: "randint requires an integer dtype",
            });
        }
        if low >= high {
            return Err(Error::InvalidRandom {
                reason: "randint requires low < high",
            });
        }
        if high.checked_sub(low).is_none() {
            return Err(Error::InvalidRandom {
                reason: "randint range overflows i64",
            });
        }
        let shape = shape.into();
        let stream = reserve_implicit_stream(device, stream_words(&shape, DType::F32, 1)?);
        self.random_stream(shape, dtype, RandomKind::RandInt { low, high }, stream)
    }

    pub fn full_like(
        &mut self,
        input: NodeId,
        value: Scalar,
        dtype: Option<DType>,
    ) -> Result<NodeId> {
        self.full_with_dtype(
            self.shape(input)?.clone(),
            value,
            dtype.unwrap_or(self.dtype(input)?),
        )
    }
    pub fn zeros_like(&mut self, input: NodeId, dtype: Option<DType>) -> Result<NodeId> {
        self.full_like(input, Scalar::I(0), dtype)
    }
    pub fn ones_like(&mut self, input: NodeId, dtype: Option<DType>) -> Result<NodeId> {
        self.full_like(input, Scalar::I(1), dtype)
    }
    pub fn empty_like(&mut self, input: NodeId, dtype: Option<DType>) -> Result<NodeId> {
        self.empty(
            self.shape(input)?.clone(),
            dtype.unwrap_or(self.dtype(input)?),
        )
    }
    pub fn rand_like(&mut self, input: NodeId, dtype: Option<DType>, seed: u64) -> Result<NodeId> {
        self.rand(
            self.shape(input)?.clone(),
            dtype.unwrap_or(self.dtype(input)?),
            seed,
        )
    }
    pub fn randn_like(&mut self, input: NodeId, dtype: Option<DType>, seed: u64) -> Result<NodeId> {
        self.randn(
            self.shape(input)?.clone(),
            dtype.unwrap_or(self.dtype(input)?),
            seed,
        )
    }

    pub fn randperm(&mut self, count: usize, dtype: DType, seed: u64) -> Result<NodeId> {
        if !dtype.is_integer() {
            return Err(Error::InvalidRandom {
                reason: "randperm requires an integer dtype",
            });
        }
        Ok(self.push(Op::RandomPermutation { seed }, Shape::new([count]), dtype))
    }
    pub fn randperm_implicit(&mut self, count: usize, dtype: DType) -> Result<NodeId> {
        if !dtype.is_integer() {
            return Err(Error::InvalidRandom {
                reason: "randperm requires an integer dtype",
            });
        }
        // `RandomPermutation` predates captured streams. Reserve the same F32
        // domain as tinygrad's `rand(n).argsort()` and derive its legacy seed
        // from that immutable reservation until permutation receives typed IR.
        let stream = reserve_implicit_stream(0, stream_words(&Shape::new([count]), DType::F32, 1)?);
        let seed = (u64::from(stream.counter[1]) << 32 | u64::from(stream.counter[0]))
            ^ (u64::from(stream.key[1]) << 1)
            ^ u64::from(stream.key[0]);
        self.randperm(count, dtype, seed)
    }

    pub fn scaled_uniform(
        &mut self,
        shape: impl Into<Shape>,
        dtype: DType,
        seed: u64,
    ) -> Result<NodeId> {
        let shape = shape.into();
        let bound = (shape.numel()? as f64).sqrt().recip();
        self.uniform(shape, -bound, bound, dtype, seed)
    }
    pub fn glorot_uniform(
        &mut self,
        shape: impl Into<Shape>,
        dtype: DType,
        seed: u64,
    ) -> Result<NodeId> {
        let shape = shape.into();
        if shape.rank() == 0 {
            return Err(Error::InvalidRandom {
                reason: "glorot_uniform requires rank at least one",
            });
        }
        let fan = shape.dims()[0]
            .checked_add(checked_initializer_tail_fan(&shape)?)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        self.uniform(
            shape,
            -(6.0 / fan as f64).sqrt(),
            (6.0 / fan as f64).sqrt(),
            dtype,
            seed,
        )
    }
    pub fn kaiming_uniform(
        &mut self,
        shape: impl Into<Shape>,
        a: f64,
        dtype: DType,
        seed: u64,
    ) -> Result<NodeId> {
        let shape = shape.into();
        if shape.rank() < 2 {
            return Err(Error::InvalidRandom {
                reason: "kaiming initializer requires rank at least two",
            });
        }
        let fan = checked_initializer_tail_fan(&shape)?;
        let b = (6.0 / (1.0 + a * a) / fan as f64).sqrt();
        self.uniform(shape, -b, b, dtype, seed)
    }
    pub fn kaiming_normal(
        &mut self,
        shape: impl Into<Shape>,
        a: f64,
        dtype: DType,
        seed: u64,
    ) -> Result<NodeId> {
        let shape = shape.into();
        if shape.rank() < 2 {
            return Err(Error::InvalidRandom {
                reason: "kaiming initializer requires rank at least two",
            });
        }
        let fan = checked_initializer_tail_fan(&shape)?;
        self.normal(
            shape,
            0.0,
            (2.0 / (1.0 + a * a) / fan as f64).sqrt(),
            dtype,
            seed,
        )
    }

    fn random(
        &mut self,
        shape: Shape,
        dtype: DType,
        kind: RandomKind,
        seed: u64,
    ) -> Result<NodeId> {
        self.random_stream(
            shape,
            dtype,
            kind,
            RandomStream {
                device: 0,
                key: [0, seed as u32],
                counter: [0, 0],
            },
        )
    }

    fn random_stream(
        &mut self,
        shape: Shape,
        dtype: DType,
        kind: RandomKind,
        stream: RandomStream,
    ) -> Result<NodeId> {
        shape.numel()?;
        Ok(self.push(Op::Random { kind, stream }, shape, dtype))
    }
}
