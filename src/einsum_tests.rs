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

fn run_with_dtype(equation: &str, inputs: Vec<TensorData>, dtype: DType) -> TensorData {
    let mut graph = Graph::new();
    let ids = inputs
        .into_iter()
        .map(|input| graph.constant(input))
        .collect::<Vec<_>>();
    let output = graph.einsum_with_dtype(equation, &ids, dtype).unwrap();
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
fn einsum_dtype_override_casts_products_before_requested_reduction() {
    let half = |values: &[f64]| {
        TensorData::from_scalars([2], DType::F16, values.iter().copied().map(Scalar::F)).unwrap()
    };
    let half_dot = run_with_dtype(
        "i,i->",
        vec![half(&[1.5, 2.0]), half(&[2.0, 1.0])],
        DType::F32,
    );
    assert_eq!(half_dot.dtype(), DType::F32);
    assert_eq!(half_dot.to_vec_f64(), vec![5.0]);

    let rounded_product = run_with_dtype(
        "i,i->",
        vec![half(&[1.000_976_562_5, 1.0]), half(&[1.000_976_562_5, 0.0])],
        DType::F32,
    );
    assert_eq!(rounded_product.to_vec_f64(), vec![1.001_953_125]);

    let rounded_sum = run_with_dtype(
        "i,i->",
        vec![
            TensorData::new([3], vec![16_777_216.0, 1.0, -16_777_216.0]).unwrap(),
            TensorData::new([3], vec![1.0, 1.0, 1.0]).unwrap(),
        ],
        DType::F32,
    );
    assert_eq!(rounded_sum.to_vec_f64(), vec![0.0]);

    let mixed = run_with_dtype(
        "i,i->",
        vec![
            TensorData::from_scalars(
                [2],
                DType::F16,
                [Scalar::F(2.0), Scalar::F(3.0)],
            )
            .unwrap(),
            TensorData::from_scalars(
                [2],
                DType::BF16,
                [Scalar::F(4.0), Scalar::F(5.0)],
            )
            .unwrap(),
        ],
        DType::F64,
    );
    assert_eq!(mixed.dtype(), DType::F64);
    assert_eq!(mixed.to_vec_f64(), vec![23.0]);

    let exact = run_with_dtype(
        "i,i->",
        vec![data([2], &[2, 3]), data([2], &[4, 5])],
        DType::F32,
    );
    assert_eq!(exact.dtype(), DType::F32);
    assert_eq!(exact.to_vec_f64(), vec![23.0]);
}

#[test]
fn einsum_dtype_override_handles_scalar_outer_and_empty_domains() {
    let scalar = run_with_dtype(
        ",->",
        vec![
            TensorData::from_scalars([], DType::F16, [Scalar::F(3.0)]).unwrap(),
            data([], &[4]),
        ],
        DType::F64,
    );
    assert_eq!(scalar.dtype(), DType::F64);
    assert_eq!(scalar.to_vec_f64(), vec![12.0]);

    let outer = run_with_dtype(
        "i,j->ij",
        vec![
            TensorData::from_scalars(
                [2],
                DType::F16,
                [Scalar::F(2.0), Scalar::F(3.0)],
            )
            .unwrap(),
            TensorData::from_scalars(
                [2],
                DType::F16,
                [Scalar::F(4.0), Scalar::F(5.0)],
            )
            .unwrap(),
        ],
        DType::F64,
    );
    assert_eq!(outer.dtype(), DType::F64);
    assert_eq!(outer.to_vec_f64(), vec![8.0, 10.0, 12.0, 15.0]);

    let diagonal = run_with_dtype(
        "ii->i",
        vec![
            TensorData::from_scalars(
                [3, 3],
                DType::F16,
                (1..=9).map(|value| Scalar::F(f64::from(value))),
            )
            .unwrap(),
        ],
        DType::F64,
    );
    assert_eq!(diagonal.dtype(), DType::F64);
    assert_eq!(diagonal.to_vec_f64(), vec![1.0, 5.0, 9.0]);

    let empty = run_with_dtype(
        "ij,j->i",
        vec![
            TensorData::from_scalars([2, 0], DType::F16, []).unwrap(),
            TensorData::from_scalars([0], DType::F16, []).unwrap(),
        ],
        DType::F32,
    );
    assert_eq!(empty.dtype(), DType::F32);
    assert_eq!(empty.to_vec_f64(), vec![0.0, 0.0]);
}

#[test]
fn einsum_dtype_override_is_traceable_and_validated_before_mutation() {
    let mut graph = Graph::new();
    let a = graph.input_dtype("a", [2], DType::F16);
    let b = graph.input_dtype("b", [2], DType::F16);
    let output = graph.einsum_with_dtype("i,i->", &[a, b], DType::F32).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert!(graph.trace(output).unwrap().to_string().contains(
        "einsum([NodeId(0), NodeId(1)], output=[], contract=[Named('i')], product=f16, accumulate=f32)"
    ));

    let before = graph.node_count();
    assert_eq!(
        graph.einsum_with_dtype("i,i->", &[a, b], DType::I32),
        Err(Error::InvalidEinsum {
            equation: "i,i->".into(),
            reason: "einsum dtype overrides currently support only f32 or f64",
        })
    );
    assert_eq!(graph.node_count(), before);
    assert_eq!(
        graph.einsum_with_dtype("i,i->", &[a, b], DType::F8E4M3),
        Err(Error::InvalidEinsum {
            equation: "i,i->".into(),
            reason: "einsum dtype overrides currently support only f32 or f64",
        })
    );

    let mut float8_graph = Graph::new();
    let f8 = float8_graph.input_dtype("f8", [2], DType::F8E4M3);
    assert_eq!(
        float8_graph.einsum_with_dtype("i->", &[f8], DType::F32),
        Err(Error::InvalidEinsum {
            equation: "i->".into(),
            reason: "einsum dtype overrides for float8 products are not implemented",
        })
    );
    assert_eq!(float8_graph.node_count(), 1);
}

#[test]
fn einsum_dtype_override_rejects_gradients_until_cast_aware_vjp_exists() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2], DType::F16);
    let y = graph.input_dtype("y", [2], DType::F16);
    let loss = graph.einsum_with_dtype("i,i->", &[x, y], DType::F32).unwrap();
    assert_eq!(
        graph.grad(loss, x),
        Err(Error::NonDifferentiableIndexing(
            "einsum dtype overrides require cast-aware gradients",
        ))
    );
}

#[test]
fn einsum_dtype_override_keeps_existing_gradients_without_a_cast_boundary() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2]);
    let y = graph.input("y", [2]);
    let loss = graph.einsum_with_dtype("i,i->", &[x, y], DType::F32).unwrap();
    let gradient = graph.grad(loss, x).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F32);
    assert!(graph
        .trace(gradient)
        .unwrap()
        .to_string()
        .contains("einsum_grad("));
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
    let gradient = graph.grad(loss, a).unwrap();
    assert_eq!(graph.shape(gradient).unwrap(), &Shape::from([2, 3]));
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F32);
    assert!(
        graph
            .trace(gradient)
            .unwrap()
            .to_string()
            .contains("einsum_grad(%4, target=0)")
    );
    assert!(graph.einsum("ij,jk->ik", &[a]).is_err());
    let c = graph.input("c", [4, 5]);
    assert!(graph.einsum("ij,jk", &[a, c]).is_err());
}

fn fdata(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
    TensorData::new(shape, values.to_vec()).unwrap()
}

fn gradient(equation: &str, inputs: &[TensorData], target: usize) -> TensorData {
    let mut graph = Graph::new();
    let ids = inputs
        .iter()
        .enumerate()
        .map(|(i, data)| graph.input(format!("x{i}"), data.shape().clone()))
        .collect::<Vec<_>>();
    let output = graph.einsum(equation, &ids).unwrap();
    let loss = if output == ids[target] {
        output
    } else {
        graph
            .reduce(output, crate::ReduceKind::Sum, None, false)
            .unwrap()
    };
    let derivative = graph.grad(loss, ids[target]).unwrap();
    let bindings = inputs
        .iter()
        .enumerate()
        .map(|(i, data)| (format!("x{i}"), data.clone()))
        .collect();
    CpuBackend.execute(&graph, derivative, &bindings).unwrap()
}

fn loss(equation: &str, inputs: &[TensorData]) -> f64 {
    let mut graph = Graph::new();
    let ids = inputs
        .iter()
        .map(|data| graph.constant(data.clone()))
        .collect::<Vec<_>>();
    let output = graph.einsum(equation, &ids).unwrap();
    let output = graph
        .reduce(output, crate::ReduceKind::Sum, None, false)
        .unwrap();
    CpuBackend
        .execute(&graph, output, &HashMap::new())
        .unwrap()
        .to_vec_f64()[0]
}

fn assert_finite_difference(equation: &str, inputs: &[TensorData], target: usize) {
    let analytic = gradient(equation, inputs, target).to_vec_f64();
    let mut plus = inputs.to_vec();
    let mut minus = inputs.to_vec();
    let eps = 1e-3_f32;
    for index in 0..analytic.len() {
        let mut p = plus[target].to_vec_f64();
        let mut m = minus[target].to_vec_f64();
        p[index] += f64::from(eps);
        m[index] -= f64::from(eps);
        plus[target] = fdata(
            inputs[target].shape().clone(),
            &p.iter().map(|v| *v as f32).collect::<Vec<_>>(),
        );
        minus[target] = fdata(
            inputs[target].shape().clone(),
            &m.iter().map(|v| *v as f32).collect::<Vec<_>>(),
        );
        let numeric = (loss(equation, &plus) - loss(equation, &minus)) / f64::from(2.0 * eps);
        assert!(
            (analytic[index] - numeric).abs() < 2e-2,
            "{equation} target {target} index {index}: analytic={} numeric={numeric}",
            analytic[index]
        );
        plus[target] = inputs[target].clone();
        minus[target] = inputs[target].clone();
    }
}

#[test]
fn einsum_gradients_cover_scatter_contracts() {
    let dot = [fdata([3], &[1., 2., 3.]), fdata([3], &[4., 5., 6.])];
    assert_eq!(gradient("i,i->", &dot, 0).to_vec_f64(), vec![4., 5., 6.]);
    assert_eq!(gradient("i,i->", &dot, 1).to_vec_f64(), vec![1., 2., 3.]);
    let matrix = [
        fdata([2, 2], &[1., 2., 3., 4.]),
        fdata([2, 2], &[5., 6., 7., 8.]),
    ];
    assert_eq!(
        gradient("ij,jk->", &matrix, 0).to_vec_f64(),
        vec![11., 15., 11., 15.]
    );
    assert_eq!(
        gradient("ij,jk->", &matrix, 1).to_vec_f64(),
        vec![4., 4., 6., 6.]
    );
    let diagonal = [fdata([3, 3], &[1., 2., 3., 4., 5., 6., 7., 8., 9.])];
    assert_eq!(
        gradient("ii->", &diagonal, 0).to_vec_f64(),
        vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]
    );
    let broadcast = [
        fdata([1, 2], &[2., 3.]),
        fdata([3, 2], &[1., 2., 3., 4., 5., 6.]),
    ];
    assert_eq!(
        gradient("ij,ij->", &broadcast, 0).to_vec_f64(),
        vec![9., 12.]
    );
    assert_eq!(
        gradient("ij,ij->", &broadcast, 1).to_vec_f64(),
        vec![2., 3., 2., 3., 2., 3.]
    );
    let repeated = [fdata([2, 2], &[1., 2., 3., 4.]), fdata([2], &[5., 7.])];
    assert_eq!(
        gradient("ii,j->", &repeated, 0).to_vec_f64(),
        vec![12., 0., 0., 12.]
    );
    let non_target_repeated = [
        fdata([2, 2], &[1., 2., 3., 4.]),
        fdata([2, 2], &[5., 6., 7., 8.]),
    ];
    assert_eq!(
        gradient("ij,kk->", &non_target_repeated, 0).to_vec_f64(),
        vec![13., 13., 13., 13.]
    );
    let scalar = [fdata([], &[2.]), fdata([3], &[3., 4., 5.])];
    assert_eq!(gradient(",i->", &scalar, 0).to_vec_f64(), vec![12.]);
    let zero = [fdata([2, 0], &[]), fdata([0], &[])];
    assert_eq!(gradient("ij,j->i", &zero, 0).len(), 0);
}

#[test]
fn einsum_gradients_match_finite_differences() {
    let matrix = [
        fdata([2, 2], &[1., 2., 3., 4.]),
        fdata([2, 2], &[5., 6., 7., 8.]),
    ];
    assert_finite_difference("ij,jk->ik", &matrix, 0);
    let ellipsis = [
        fdata([2, 1, 2], &[1., 2., 3., 4.]),
        fdata([1, 2, 1], &[5., 6.]),
    ];
    assert_finite_difference("...ij,...jk->...ik", &ellipsis, 0);
    let three = [
        fdata([2], &[1., 2.]),
        fdata([2], &[3., 4.]),
        fdata([2], &[5., 6.]),
    ];
    assert_finite_difference("i,i,i->", &three, 1);
}

#[test]
fn einsum_gradient_accumulates_multiple_paths() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2]);
    let y = graph.input("y", [2]);
    let product = graph.einsum("i,i->", &[x, y]).unwrap();
    let loss = graph.add(product, product).unwrap();
    let dx = graph.grad(loss, x).unwrap();
    let values = HashMap::from([
        ("x".into(), fdata([2], &[1., 2.])),
        ("y".into(), fdata([2], &[3., 4.])),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&graph, dx, &values)
            .unwrap()
            .to_vec_f64(),
        vec![6., 8.]
    );
}

#[test]
fn einsum_gradients_keep_target_float_dtype_and_skip_exact_inputs() {
    let mut graph = Graph::new();
    let half = graph.input_dtype("half", [2], DType::F16);
    let float = graph.input("float", [2]);
    let loss = graph.einsum("i,i->", &[half, float]).unwrap();
    let dhalf = graph.grad(loss, half).unwrap();
    assert_eq!(graph.dtype(dhalf).unwrap(), DType::F16);
    let values = HashMap::from([
        (
            "half".into(),
            TensorData::from_scalars([2], DType::F16, [Scalar::F(1.0), Scalar::F(2.0)]).unwrap(),
        ),
        ("float".into(), fdata([2], &[3., 4.])),
    ]);
    assert_eq!(
        CpuBackend.execute(&graph, dhalf, &values).unwrap().dtype(),
        DType::F16
    );

    let mut exact_graph = Graph::new();
    let exact = exact_graph.input_dtype("exact", [2], DType::I32);
    let floating = exact_graph.input("floating", [2]);
    let exact_loss = exact_graph.einsum("i,i->", &[exact, floating]).unwrap();
    assert_eq!(
        exact_graph.grad(exact_loss, exact),
        Err(Error::NoGradient(exact))
    );
}
