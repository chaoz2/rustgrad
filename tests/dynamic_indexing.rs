use rustgrad::{CpuBackend, DType, DynamicAllocationTarget, Graph, Scalar, TensorData};
use std::collections::HashMap;

fn ints(values: [i64; 4]) -> TensorData {
    TensorData::from_scalars([2, 2], DType::I64, values.map(Scalar::I)).unwrap()
}

#[test]
fn nonzero_realizes_row_major_coordinates_with_fresh_runtime_shapes() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 2], DType::I64);
    let output = graph.nonzero(input).unwrap();
    let backend = CpuBackend;
    let some = backend
        .execute_dynamic(
            &graph,
            output,
            &HashMap::from([("input".into(), ints([0, 2, -3, 0]))]),
        )
        .unwrap();
    assert_eq!(some.shape.shape(), &[2, 2].into());
    assert_eq!(
        some.output,
        TensorData::from_scalars(
            [2, 2],
            DType::I32,
            [Scalar::I(0), Scalar::I(1), Scalar::I(1), Scalar::I(0)]
        )
        .unwrap()
    );
    let none = backend
        .execute_dynamic(
            &graph,
            output,
            &HashMap::from([("input".into(), ints([0, 0, 0, 0]))]),
        )
        .unwrap();
    assert_eq!(none.shape.shape(), &[0, 2].into());
    assert_eq!(none.output.dtype(), DType::I32);
    assert_eq!(none.output.len(), 0);
}

#[test]
fn dynamic_masked_select_broadcasts_and_changes_concrete_extent() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 2]);
    let mask = graph.input_dtype("mask", [1, 2], DType::Bool);
    let output = graph.masked_select_dynamic(input, mask).unwrap();
    let plan = graph.dynamic_allocation_plan(output).unwrap();
    assert_eq!(plan.bindings().len(), 2);
    assert_eq!(plan.allocation_for_count(0).unwrap().bytes, 0);
    assert!(
        plan.validate_target(DynamicAllocationTarget::NativeCpuJit)
            .is_err()
    );
    let backend = CpuBackend;
    let values = TensorData::new([2, 2], vec![10., 20., 30., 40.]).unwrap();
    let selected = backend
        .execute_dynamic(
            &graph,
            output,
            &HashMap::from([
                ("input".into(), values.clone()),
                (
                    "mask".into(),
                    TensorData::from_scalars(
                        [1, 2],
                        DType::Bool,
                        [Scalar::Bool(true), Scalar::Bool(false)],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(selected.shape.shape(), &[2].into());
    assert_eq!(
        selected.output,
        TensorData::new([2], vec![10., 30.]).unwrap()
    );
    let empty = backend
        .execute_dynamic(
            &graph,
            output,
            &HashMap::from([
                ("input".into(), values),
                (
                    "mask".into(),
                    TensorData::from_scalars(
                        [1, 2],
                        DType::Bool,
                        [Scalar::Bool(false), Scalar::Bool(false)],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(empty.shape.shape(), &[0].into());
    assert_eq!(empty.output.len(), 0);

    let gradient = backend
        .execute_dynamic_masked_select_vjp(
            &graph,
            output,
            &TensorData::new([2], vec![2., 3.]).unwrap(),
            &HashMap::from([
                (
                    "input".into(),
                    TensorData::new([2, 2], vec![10., 20., 30., 40.]).unwrap(),
                ),
                (
                    "mask".into(),
                    TensorData::from_scalars(
                        [1, 2],
                        DType::Bool,
                        [Scalar::Bool(true), Scalar::Bool(false)],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(
        gradient,
        TensorData::new([2, 2], vec![2., 0., 3., 0.]).unwrap()
    );
}

#[test]
fn dynamic_sum_loss_backpropagates_through_broadcast_mask_with_varying_extent() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 2]);
    let mask = graph.input_dtype("mask", [1, 2], DType::Bool);
    let selected = graph.masked_select_dynamic(input, mask).unwrap();
    let loss = graph.dynamic_sum(selected).unwrap();
    let backend = CpuBackend;
    let values = TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap();
    let result = backend
        .execute_dynamic_gradient(
            &graph,
            loss,
            input,
            &HashMap::from([
                ("input".into(), values.clone()),
                (
                    "mask".into(),
                    TensorData::from_scalars(
                        [1, 2],
                        DType::Bool,
                        [Scalar::Bool(true), Scalar::Bool(false)],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(result.loss.output, TensorData::scalar(4.0f32));
    assert_eq!(
        result.gradient,
        TensorData::new([2, 2], vec![1., 0., 1., 0.]).unwrap()
    );
    let empty = backend
        .execute_dynamic_gradient(
            &graph,
            loss,
            input,
            &HashMap::from([
                ("input".into(), values),
                (
                    "mask".into(),
                    TensorData::from_scalars(
                        [1, 2],
                        DType::Bool,
                        [Scalar::Bool(false), Scalar::Bool(false)],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(empty.loss.output, TensorData::scalar(0.0f32));
    assert_eq!(
        empty.gradient,
        TensorData::new([2, 2], vec![0., 0., 0., 0.]).unwrap()
    );
}
