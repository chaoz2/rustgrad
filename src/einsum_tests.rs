use crate::{Backend, CpuBackend, DType, Error, Graph, Scalar, Shape, TensorData};
use std::collections::HashMap;

fn data(shape: impl Into<Shape>, values: &[i64]) -> TensorData {
    TensorData::from_scalars(shape, DType::I32, values.iter().copied().map(Scalar::I)).unwrap()
}

fn run(equation: &str, inputs: Vec<TensorData>) -> TensorData {
    let mut graph = Graph::new();
    let ids = inputs
        .into_iter()
        .map(|input| graph.constant(input))
        .collect::<Vec<_>>();
    let output = graph.einsum(equation, &ids).unwrap();
    CpuBackend.execute(&graph, output, &HashMap::new()).unwrap()
}

#[test]
fn einsum_forward_fixtures() {
    let scalar = run(",->", vec![data([], &[3]), data([], &[4])]);
    assert_eq!(scalar.to_vec_f64(), vec![12.0]);
    assert_eq!(run("", vec![data([], &[7])]).to_vec_f64(), vec![7.0]);
    assert_eq!(
        run("i,i->", vec![data([3], &[1, 2, 3]), data([3], &[4, 5, 6])]).to_vec_f64(),
        vec![32.0]
    );
    assert_eq!(
        run("i,j->ij", vec![data([2], &[2, 3]), data([2], &[4, 5])]).to_vec_f64(),
        vec![8., 10., 12., 15.]
    );
    assert_eq!(
        run(
            "ij,jk->ik",
            vec![
                data([2, 3], &[1, 2, 3, 4, 5, 6]),
                data([3, 2], &[7, 8, 9, 10, 11, 12])
            ]
        )
        .to_vec_f64(),
        vec![58., 64., 139., 154.]
    );
    assert_eq!(
        run("ii->", vec![data([3, 3], &[1, 2, 3, 4, 5, 6, 7, 8, 9])]).to_vec_f64(),
        vec![15.]
    );
    assert_eq!(
        run("ii->i", vec![data([3, 3], &[1, 2, 3, 4, 5, 6, 7, 8, 9])]).to_vec_f64(),
        vec![1., 5., 9.]
    );
    // No explicit output follows NumPy's alphabetical singleton ordering.
    assert_eq!(
        run("ji", vec![data([2, 3], &[1, 2, 3, 4, 5, 6])]).shape(),
        &Shape::from([3, 2])
    );
    assert_eq!(
        run(
            "...ij,...jk->...ik",
            vec![data([2, 1, 2], &[1, 2, 3, 4]), data([1, 2, 1], &[5, 6])]
        )
        .to_vec_f64(),
        vec![17., 39.]
    );
    assert_eq!(
        run(
            "ij,ij,ij->",
            vec![
                data([1, 2], &[2, 3]),
                data([1, 2], &[4, 5]),
                data([1, 2], &[6, 7])
            ]
        )
        .to_vec_f64(),
        vec![153.]
    );
}

#[test]
fn einsum_preserves_promoted_exact_storage_and_empty_domains() {
    let mut graph = Graph::new();
    let lhs = graph.constant(
        TensorData::from_scalars([2], DType::U64, [Scalar::U(1 << 40), Scalar::U(2)]).unwrap(),
    );
    let rhs = graph
        .constant(TensorData::from_scalars([2], DType::I64, [Scalar::I(3), Scalar::I(4)]).unwrap());
    let out = graph.einsum("i,i->", &[lhs, rhs]).unwrap();
    let result = CpuBackend.execute(&graph, out, &HashMap::new()).unwrap();
    assert_eq!(result.dtype(), DType::F64); // U64 + I64 promotes to F64 in RustGrad's lattice.
    assert_eq!(result.to_vec_f64(), vec![(1u64 << 40) as f64 * 3.0 + 8.0]);
    let empty = run("ij,j->i", vec![data([2, 0], &[]), data([0], &[])]);
    assert_eq!(empty.shape(), &Shape::from([2]));
    assert_eq!(empty.to_vec_f64(), vec![0., 0.]);
    let zero_output = run("ij->i", vec![data([0, 3], &[])]);
    assert_eq!(zero_output.len(), 0);
}

#[test]
fn einsum_errors_trace_and_gradient_prerequisite() {
    let mut graph = Graph::new();
    let a = graph.input("a", [2, 3]);
    let b = graph.input("b", [3, 4]);
    let output = graph.einsum("ij,jk->ik", &[a, b]).unwrap();
    let trace = graph.trace(output).unwrap().to_string();
    assert!(trace.contains(
        "einsum([NodeId(0), NodeId(1)], output=[Named('i'), Named('k')], contract=[Named('j')])"
    ));
    assert_eq!(
        graph.grad(output, a),
        Err(Error::NonScalarLoss(Shape::from([2, 4])))
    );
    let loss = graph.einsum("ij,jk->", &[a, b]).unwrap();
    assert_eq!(graph.grad(loss, a), Err(Error::EinsumGradientPending));
    assert!(graph.einsum("ij,jk->ik", &[a]).is_err());
    let c = graph.input("c", [4, 5]);
    assert!(graph.einsum("ij,jk", &[a, c]).is_err());
}
