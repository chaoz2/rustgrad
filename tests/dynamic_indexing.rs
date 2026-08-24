use rustgrad::{CpuBackend, DType, Graph, Scalar, TensorData};
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
            DType::I64,
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
    assert_eq!(none.output.len(), 0);
}
