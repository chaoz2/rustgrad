use super::schema::{axes_usize, const_i64, reshape_dims};
use super::tensor::tensor_data;
use super::*;
use crate::{DType, Scalar};
fn vi(mut id: u32, out: &mut Vec<u8>) {
    loop {
        let b = (id & 127) as u8;
        id >>= 7;
        out.push(if id == 0 { b } else { b | 128 });
        if id == 0 {
            return;
        }
    }
}
fn field(out: &mut Vec<u8>, id: u32, data: &[u8]) {
    vi(id << 3 | 2, out);
    vi(data.len() as u32, out);
    out.extend_from_slice(data)
}
fn var(out: &mut Vec<u8>, id: u32, n: u32) {
    vi(id << 3, out);
    vi(n, out)
}
fn vi64(mut n: u64, out: &mut Vec<u8>) {
    loop {
        let b = (n & 127) as u8;
        n >>= 7;
        out.push(if n == 0 { b } else { b | 128 });
        if n == 0 {
            return;
        }
    }
}
fn int64_attr(name: &str, value: i64) -> Vec<u8> {
    let mut a = vec![];
    text(&mut a, 1, name);
    vi(3 << 3, &mut a);
    vi64(value as u64, &mut a);
    a
}
fn text(out: &mut Vec<u8>, id: u32, s: &str) {
    field(out, id, s.as_bytes())
}
fn ints_attr(name: &str, values: &[u32]) -> Vec<u8> {
    let mut a = vec![];
    text(&mut a, 1, name);
    let mut packed = vec![];
    for &value in values {
        vi(value, &mut packed);
    }
    field(&mut a, 8, &packed);
    a
}
fn int_attr(name: &str, value: u32) -> Vec<u8> {
    let mut a = vec![];
    text(&mut a, 1, name);
    var(&mut a, 3, value);
    a
}
fn string_attr(name: &str, value: &str) -> Vec<u8> {
    let mut a = vec![];
    text(&mut a, 1, name);
    text(&mut a, 4, value);
    a
}
fn value(name: &str, dims: &[u32]) -> Vec<u8> {
    value_dtype(name, dims, 1)
}
fn value_dtype(name: &str, dims: &[u32], dtype: u32) -> Vec<u8> {
    let mut shape = vec![];
    for &d in dims {
        let mut dm = vec![];
        var(&mut dm, 1, d);
        field(&mut shape, 1, &dm)
    }
    let mut ten = vec![];
    var(&mut ten, 1, dtype);
    field(&mut ten, 2, &shape);
    let mut ty = vec![];
    field(&mut ty, 1, &ten);
    let mut x = vec![];
    text(&mut x, 1, name);
    field(&mut x, 2, &ty);
    x
}
fn tensor(name: &str, dims: &[u32], data: &[f32]) -> Vec<u8> {
    let mut x = vec![];
    let mut packed = vec![];
    for &d in dims {
        vi(d, &mut packed)
    }
    field(&mut x, 1, &packed);
    var(&mut x, 2, 1);
    text(&mut x, 8, name);
    let raw: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    field(&mut x, 9, &raw);
    x
}
fn raw_tensor(name: &str, dims: &[u32], dtype: u32, raw: &[u8]) -> Vec<u8> {
    let mut x = vec![];
    let mut packed = vec![];
    for &d in dims {
        vi(d, &mut packed)
    }
    field(&mut x, 1, &packed);
    var(&mut x, 2, dtype);
    if !name.is_empty() {
        text(&mut x, 8, name);
    }
    field(&mut x, 9, raw);
    x
}
fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn typed_i64_tensor(dims: &[u32], values: &[i64]) -> Vec<u8> {
    let mut x = vec![];
    let mut packed_dims = vec![];
    for &d in dims {
        vi(d, &mut packed_dims)
    }
    if !packed_dims.is_empty() {
        field(&mut x, 1, &packed_dims);
    }
    var(&mut x, 2, 7);
    let mut packed_values = vec![];
    for &value in values {
        vi64(value as u64, &mut packed_values);
    }
    field(&mut x, 7, &packed_values);
    x
}
fn tensor_attr(name: &str, tensor: &[u8]) -> Vec<u8> {
    let mut a = vec![];
    text(&mut a, 1, name);
    field(&mut a, 5, tensor);
    a
}
fn node(op: &str, ins: &[&str], out: &str) -> Vec<u8> {
    let mut x = vec![];
    for i in ins {
        text(&mut x, 1, i)
    }
    text(&mut x, 2, out);
    text(&mut x, 4, op);
    x
}
fn model_proto(initializers: &[Vec<u8>], nodes: &[Vec<u8>], outputs: &[Vec<u8>]) -> Vec<u8> {
    let mut graph = vec![];
    for initializer in initializers {
        field(&mut graph, 5, initializer);
    }
    for node in nodes {
        field(&mut graph, 1, node);
    }
    for output in outputs {
        field(&mut graph, 12, output);
    }
    let mut opset = vec![];
    var(&mut opset, 2, 13);
    let mut model = vec![];
    field(&mut model, 7, &graph);
    field(&mut model, 8, &opset);
    model
}
fn fattr(name: &str, value: f32) -> Vec<u8> {
    let mut a = vec![];
    text(&mut a, 1, name);
    vi(2 << 3 | 5, &mut a);
    a.extend_from_slice(&value.to_le_bytes());
    a
}
fn mlp() -> Vec<u8> {
    let mut g = vec![];
    field(&mut g, 11, &value("x", &[1, 2]));
    field(&mut g, 12, &value("y", &[1, 2]));
    field(&mut g, 5, &tensor("w", &[2, 2], &[1., 2., 3., 4.]));
    field(&mut g, 5, &tensor("b", &[1, 2], &[1., -10.]));
    field(&mut g, 1, &node("MatMul", &["x", "w"], "m"));
    field(&mut g, 1, &node("Add", &["m", "b"], "a"));
    field(&mut g, 1, &node("Relu", &["a"], "y"));
    let mut op = vec![];
    var(&mut op, 2, 13);
    let mut m = vec![];
    field(&mut m, 7, &g);
    field(&mut m, 8, &op);
    m
}
#[test]
fn imports_static_mlp_and_rejects_schema() {
    let model = import_onnx(&mlp()).unwrap();
    let out = model
        .run(HashMap::from([(
            "x".into(),
            TensorData::new([1, 2], vec![1., 2.]).unwrap(),
        )]))
        .unwrap();
    assert_eq!(out["y"].values(), &[8., 0.]);
    let mut bad = mlp();
    bad[0] = 0xff;
    assert!(import_onnx(&bad).is_err());
}
#[test]
fn imports_additional_static_activations() {
    let mut bytes = mlp();
    let at = bytes.windows(4).position(|x| x == b"Relu").unwrap();
    bytes[at..at + 4].copy_from_slice(b"Tanh");
    let out = import_onnx(&bytes)
        .unwrap()
        .run(HashMap::from([(
            "x".into(),
            TensorData::new([1, 2], vec![1., 2.]).unwrap(),
        )]))
        .unwrap();
    assert!(out["y"].values()[0] > 0.999 && out["y"].values()[1].abs() < 1e-6);
}
#[test]
fn static_movement_shape_and_axis_contracts_are_checked() {
    assert_eq!(reshape_dims(&[2, 3], &[3, -1]).unwrap().dims(), &[3, 2]);
    assert!(reshape_dims(&[2, 3], &[0, 0, 0]).is_err());
    assert!(reshape_dims(&[2, 3], &[-1, -1]).is_err());
    assert_eq!(axes_usize(&[-1, 0], 2).unwrap(), vec![1, 0]);
    assert!(axes_usize(&[2], 2).is_err());
    let constants = BTreeMap::from([(
        "shape".into(),
        TensorData::from_scalars([2], DType::I64, [crate::Scalar::I(3), crate::Scalar::I(2)])
            .unwrap(),
    )]);
    assert_eq!(const_i64(&constants, "shape").unwrap(), vec![3, 2]);
}

#[test]
fn flatten_rejects_unknown_attributes_before_publication() {
    let mut g = Graph::new();
    let x = g.input("x", [2, 3]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = g.node_count();
    let mut invalid = node("Flatten", &["x"], "out");
    field(&mut invalid, 5, &int_attr("keepdims", 1));
    assert!(lower(
        &mut g,
        Msg::new(&invalid),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(g.node_count(), before_nodes);

    let mut valid = node("Flatten", &["x"], "flat");
    field(&mut valid, 5, &int_attr("axis", 1));
    lower(&mut g, Msg::new(&valid), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["flat"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2, 3]);
    assert_eq!(output.values(), &[1., 2., 3., 4., 5., 6.]);
}

#[test]
fn unsqueeze_supports_sorted_signed_axes_and_preflights_them_together() {
    let mut g = Graph::new();
    let x = g.input("x", [2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([(
        "axes".into(),
        TensorData::from_scalars([2], DType::I64, [Scalar::I(-1), Scalar::I(0)]).unwrap(),
    )]);
    lower(
        &mut g,
        Msg::new(&node("Unsqueeze", &["x", "axes"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([("x".into(), TensorData::new([2], vec![2., 3.]).unwrap())]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[1, 2, 1]);
    assert_eq!(output.values(), &[2., 3.]);

    let mut malformed = Graph::new();
    let x = malformed.input("x", [2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([(
        "axes".into(),
        TensorData::from_scalars([1], DType::I64, [Scalar::I(i64::MAX)]).unwrap(),
    )]);
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = malformed.node_count();
    assert!(lower(
        &mut malformed,
        Msg::new(&node("Unsqueeze", &["x", "axes"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(malformed.node_count(), before_nodes);
}

#[test]
fn squeeze_sorts_signed_axes_and_preflights_the_full_sequence() {
    let mut g = Graph::new();
    let x = g.input("x", [1, 2, 1]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([(
        "axes".into(),
        TensorData::from_scalars([2], DType::I64, [Scalar::I(2), Scalar::I(0)]).unwrap(),
    )]);
    lower(
        &mut g,
        Msg::new(&node("Squeeze", &["x", "axes"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([1, 2, 1], vec![2., 3.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2]);
    assert_eq!(output.values(), &[2., 3.]);

    let mut malformed = Graph::new();
    let x = malformed.input("x", [1, 1]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([(
        "axes".into(),
        TensorData::from_scalars([2], DType::I64, [Scalar::I(2), Scalar::I(0)]).unwrap(),
    )]);
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = malformed.node_count();
    assert!(lower(
        &mut malformed,
        Msg::new(&node("Squeeze", &["x", "axes"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(malformed.node_count(), before_nodes);
}

#[test]
fn reshape_preflights_allowzero_before_publication() {
    let mut g = Graph::new();
    let x = g.input("x", [2, 3]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let before_values = values.clone();
    let before_nodes = g.node_count();
    let mut constants = BTreeMap::from([(
        "shape".into(),
        TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(3)]).unwrap(),
    )]);
    let before_constants = constants.clone();

    for (case, attribute) in [
        ("allowzero", int_attr("allowzero", 1)),
        ("unknown", int_attr("axis", 0)),
    ] {
        let mut invalid = node("Reshape", &["x", "shape"], "out");
        field(&mut invalid, 5, &attribute);
        assert!(
            lower(
                &mut g,
                Msg::new(&invalid),
                &mut values,
                &mut constants,
            )
            .is_err(),
            "{case}"
        );
        assert_eq!(values, before_values, "{case}");
        assert_eq!(constants, before_constants, "{case}");
        assert_eq!(g.node_count(), before_nodes, "{case}");
    }

    let mut valid = node("Reshape", &["x", "shape"], "valid");
    field(&mut valid, 5, &int_attr("allowzero", 0));
    lower(&mut g, Msg::new(&valid), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["valid"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2, 3]);
    assert_eq!(output.values(), &[1., 2., 3., 4., 5., 6.]);
}

#[test]
fn transpose_preflights_closed_attributes_and_permutation_before_publication() {
    let mut g = Graph::new();
    let x = g.input("x", [2, 3]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let before_values = values.clone();
    let before_nodes = g.node_count();

    for (case, attribute) in [
        ("unknown", int_attr("axis", 0)),
        ("duplicate", ints_attr("perm", &[0, 0])),
    ] {
        let mut invalid = node("Transpose", &["x"], "out");
        field(&mut invalid, 5, &attribute);
        assert!(
            lower(&mut g, Msg::new(&invalid), &mut values, &mut BTreeMap::new()).is_err(),
            "{case}"
        );
        assert_eq!(values, before_values, "{case}");
        assert_eq!(g.node_count(), before_nodes, "{case}");
    }

    let mut valid = node("Transpose", &["x"], "valid");
    field(&mut valid, 5, &ints_attr("perm", &[1, 0]));
    lower(&mut g, Msg::new(&valid), &mut values, &mut BTreeMap::new()).unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["valid"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[3, 2]);
    assert_eq!(output.values(), &[1., 4., 2., 5., 3., 6.]);
}

#[test]
fn slice_rejects_duplicate_axes_with_unit_steps_before_publication() {
    let mut g = Graph::new();
    let x = g.input("x", [4]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let before_values = values.clone();
    let before_nodes = g.node_count();
    let mut invalid_constants = BTreeMap::from([
        (
            "starts".into(),
            TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(1)]).unwrap(),
        ),
        (
            "ends".into(),
            TensorData::from_scalars([2], DType::I64, [Scalar::I(3), Scalar::I(4)]).unwrap(),
        ),
        (
            "axes".into(),
            TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(0)]).unwrap(),
        ),
        (
            "steps".into(),
            TensorData::from_scalars([2], DType::I64, [Scalar::I(1), Scalar::I(1)]).unwrap(),
        ),
    ]);
    let before_constants = invalid_constants.clone();
    assert!(
        lower(
            &mut g,
            Msg::new(&node("Slice", &["x", "starts", "ends", "axes", "steps"], "out")),
            &mut values,
            &mut invalid_constants,
        )
        .is_err()
    );
    assert_eq!(values, before_values);
    assert_eq!(invalid_constants, before_constants);
    assert_eq!(g.node_count(), before_nodes);

    let mut constants = BTreeMap::from([
        (
            "starts".into(),
            TensorData::from_scalars([1], DType::I64, [Scalar::I(1)]).unwrap(),
        ),
        (
            "ends".into(),
            TensorData::from_scalars([1], DType::I64, [Scalar::I(3)]).unwrap(),
        ),
        (
            "axes".into(),
            TensorData::from_scalars([1], DType::I64, [Scalar::I(0)]).unwrap(),
        ),
        (
            "steps".into(),
            TensorData::from_scalars([1], DType::I64, [Scalar::I(1)]).unwrap(),
        ),
    ]);
    lower(
        &mut g,
        Msg::new(&node("Slice", &["x", "starts", "ends", "axes", "steps"], "valid")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["valid"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([4], vec![1., 2., 3., 4.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.values(), &[2., 3.]);
}

#[test]
fn softmax_family_preflights_closed_attribute_surface_before_publication() {
    let mut g = Graph::new();
    let x = g.input("x", [1, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let before_values = values.clone();
    let before_nodes = g.node_count();
    for op in ["Softmax", "LogSoftmax"] {
        let mut invalid = node(op, &["x"], "out");
        field(&mut invalid, 5, &int_attr("keepdims", 1));
        assert!(
            lower(&mut g, Msg::new(&invalid), &mut values, &mut BTreeMap::new()).is_err(),
            "{op}"
        );
        assert_eq!(values, before_values, "{op}");
        assert_eq!(g.node_count(), before_nodes, "{op}");
    }

    let mut valid = node("Softmax", &["x"], "valid");
    field(&mut valid, 5, &int_attr("axis", 1));
    lower(&mut g, Msg::new(&valid), &mut values, &mut BTreeMap::new()).unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["valid"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([1, 2], vec![0., 0.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.values(), &[0.5, 0.5]);
}

#[test]
fn gemm_and_softmax_lower_through_cpu_graph() {
    let mut g = Graph::new();
    let a = g.input("a", [1, 2]);
    let b = g.input("b", [2, 2]);
    let c = g.input("c", [1, 2]);
    let mut values = BTreeMap::from([("a".into(), a), ("b".into(), b), ("c".into(), c)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut g,
        Msg::new(&node("Gemm", &["a", "b", "c"], "m")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    lower(
        &mut g,
        Msg::new(&node("Softmax", &["m"], "y")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let out = CpuBackend
        .execute(
            &g,
            values["y"],
            &HashMap::from([
                ("a".into(), TensorData::new([1, 2], vec![1., 2.]).unwrap()),
                (
                    "b".into(),
                    TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
                ),
                ("c".into(), TensorData::new([1, 2], vec![1., 0.]).unwrap()),
            ]),
        )
        .unwrap();
    assert!(out.values()[1] > out.values()[0]);
    assert!((out.values()[0] + out.values()[1] - 1.).abs() < 1e-6);
}
#[test]
fn gemm_finite_scales_are_compositional() {
    let mut g = Graph::new();
    let a = g.input("a", [1, 1]);
    let b = g.input("b", [1, 1]);
    let c = g.input("c", [1, 1]);
    let mut values = BTreeMap::from([("a".into(), a), ("b".into(), b), ("c".into(), c)]);
    let mut n = node("Gemm", &["a", "b", "c"], "y");
    field(&mut n, 5, &fattr("alpha", 2.));
    field(&mut n, 5, &fattr("beta", 3.));
    lower(&mut g, Msg::new(&n), &mut values, &mut BTreeMap::new()).unwrap();
    let out = CpuBackend
        .execute(
            &g,
            values["y"],
            &HashMap::from([
                ("a".into(), TensorData::new([1, 1], vec![2.]).unwrap()),
                ("b".into(), TensorData::new([1, 1], vec![4.]).unwrap()),
                ("c".into(), TensorData::new([1, 1], vec![5.]).unwrap()),
            ]),
        )
        .unwrap();
    assert_eq!(out.values(), &[31.]);
}

#[test]
fn gemm_preflights_closed_attributes_and_binary_transpose_flags() {
    let mut g = Graph::new();
    let a = g.input("a", [2, 1]);
    let b = g.input("b", [2, 1]);
    let mut values = BTreeMap::from([("a".into(), a), ("b".into(), b)]);
    let before_values = values.clone();
    let before_nodes = g.node_count();

    for (case, attribute) in [
        ("unknown", int_attr("broadcast", 1)),
        ("trans_a", int_attr("transA", 2)),
        ("trans_b", int_attr("transB", 2)),
    ] {
        let mut invalid = node("Gemm", &["a", "b"], "out");
        field(&mut invalid, 5, &attribute);
        assert!(
            lower(&mut g, Msg::new(&invalid), &mut values, &mut BTreeMap::new()).is_err(),
            "{case}"
        );
        assert_eq!(values, before_values, "{case}");
        assert_eq!(g.node_count(), before_nodes, "{case}");
    }

    let mut valid = node("Gemm", &["a", "b"], "valid");
    field(&mut valid, 5, &int_attr("transA", 1));
    field(&mut valid, 5, &int_attr("transB", 0));
    lower(&mut g, Msg::new(&valid), &mut values, &mut BTreeMap::new()).unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["valid"],
            &HashMap::from([
                ("a".into(), TensorData::new([2, 1], vec![2., 3.]).unwrap()),
                ("b".into(), TensorData::new([2, 1], vec![5., 7.]).unwrap()),
            ]),
        )
        .unwrap();
    assert_eq!(output.values(), &[31.]);
}

#[test]
fn gemm_preflights_optional_bias_before_constructing_transposes_or_matmul() {
    let mut g = Graph::new();
    let a = g.input("a", [1, 2]);
    let b = g.input("b", [2, 2]);
    let bad_c = g.input("bad_c", [3]);
    let mut values = BTreeMap::from([
        ("a".into(), a),
        ("b".into(), b),
        ("bad_c".into(), bad_c),
    ]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = g.node_count();

    for inputs in [["a", "b", "missing"], ["a", "b", "bad_c"]] {
        assert!(lower(
            &mut g,
            Msg::new(&node("Gemm", &inputs, "out")),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(g.node_count(), before_nodes);
    }
}

#[test]
fn typed_payloads_match_raw_bits_including_u64() {
    let raw = tensor(
        "f",
        &[2],
        &[f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234)],
    );
    let mut typed = vec![];
    let mut dims = vec![];
    vi(2, &mut dims);
    field(&mut typed, 1, &dims);
    var(&mut typed, 2, 1);
    text(&mut typed, 8, "f");
    vi(4 << 3 | 5, &mut typed);
    typed.extend_from_slice(&0x8000_0000u32.to_le_bytes());
    vi(4 << 3 | 5, &mut typed);
    typed.extend_from_slice(&0x7fc0_1234u32.to_le_bytes());
    assert_eq!(
        super::tensor(Msg::new(&raw))
            .unwrap()
            .1
            .to_le_bytes()
            .unwrap(),
        super::tensor(Msg::new(&typed))
            .unwrap()
            .1
            .to_le_bytes()
            .unwrap()
    );
    let mut u = vec![];
    field(&mut u, 1, &[1]);
    var(&mut u, 2, 13);
    text(&mut u, 8, "u");
    let mut packed = vec![0xff; 9];
    packed.push(1);
    field(&mut u, 11, &packed);
    assert_eq!(
        super::tensor(Msg::new(&u))
            .unwrap()
            .1
            .to_le_bytes()
            .unwrap(),
        u64::MAX.to_le_bytes()
    );
}
#[test]
fn typed_payload_acceptance_and_rejection_matrix() {
    fn msg(dtype: u32, fid: u32, payload: Vec<u8>) -> Vec<u8> {
        let mut x = vec![];
        field(&mut x, 1, &[1]);
        var(&mut x, 2, dtype);
        text(&mut x, 8, "x");
        field(&mut x, fid, &payload);
        x
    }
    let cases = [
        (9, 5, vec![1], vec![1]),
        (3, 5, vec![127], vec![127]),
        (2, 5, vec![0xff, 1], vec![0xff]),
        (5, 5, vec![123], vec![123, 0]),
        (4, 5, vec![0xff, 0xff, 3], vec![0xff, 0xff]),
        (
            6,
            5,
            vec![0xff, 0xff, 0xff, 0xff, 0x0f],
            (-1i32).to_le_bytes().to_vec(),
        ),
        (
            12,
            5,
            vec![0xff, 0xff, 0xff, 0xff, 0x0f],
            u32::MAX.to_le_bytes().to_vec(),
        ),
        (7, 7, vec![0x7f], 127i64.to_le_bytes().to_vec()),
        (10, 5, vec![0xff, 0xff, 3], vec![0xff, 0xff]),
        (16, 5, vec![0x81, 0xfc, 3], vec![0x01, 0xfe]),
    ];
    for (dtype, field, payload, expect) in cases {
        assert_eq!(
            super::tensor(Msg::new(&msg(dtype, field, payload)))
                .unwrap()
                .1
                .to_le_bytes()
                .unwrap(),
            expect,
            "dtype {dtype}"
        )
    }
    let mut bad = msg(9, 5, vec![2]);
    assert!(super::tensor(Msg::new(&bad)).is_err());
    bad = msg(2, 5, vec![0x80, 0x02]);
    assert!(super::tensor(Msg::new(&bad)).is_err());
    bad = msg(1, 5, vec![1]);
    assert!(super::tensor(Msg::new(&bad)).is_err());
    let mut conflict = msg(1, 4, 0u32.to_le_bytes().to_vec());
    field(&mut conflict, 9, &0f32.to_le_bytes());
    assert!(super::tensor(Msg::new(&conflict)).is_err());
}
#[test]
fn default_nchw_conv_lowers_through_cpu_graph() {
    let mut g = Graph::new();
    let x = g.input("x", [1, 1, 2, 2]);
    let w = g.input("w", [1, 1, 1, 1]);
    let mut values = BTreeMap::from([("x".into(), x), ("w".into(), w)]);
    lower(
        &mut g,
        Msg::new(&node("Conv", &["x", "w"], "y")),
        &mut values,
        &mut BTreeMap::new(),
    )
    .unwrap();
    let out = CpuBackend
        .execute(
            &g,
            values["y"],
            &HashMap::from([
                (
                    "x".into(),
                    TensorData::new([1, 1, 2, 2], vec![1., 2., 3., 4.]).unwrap(),
                ),
                ("w".into(), TensorData::new([1, 1, 1, 1], vec![2.]).unwrap()),
            ]),
        )
        .unwrap();
    assert_eq!(out.values(), &[2., 4., 6., 8.]);
}
#[test]
fn conv_attributes_cover_grouped_asymmetric_and_same_padding() {
    let mut g = Graph::new();
    let x = g.input("x", [1, 2, 3, 4]);
    let w = g.input("w", [2, 1, 2, 2]);
    let mut values = BTreeMap::from([("x".into(), x), ("w".into(), w)]);
    let mut n = node("Conv", &["x", "w"], "y");
    for a in [
        int_attr("group", 2),
        ints_attr("strides", &[2, 1]),
        ints_attr("dilations", &[1, 2]),
        ints_attr("pads", &[1, 2, 0, 1]),
    ] {
        field(&mut n, 5, &a);
    }
    lower(&mut g, Msg::new(&n), &mut values, &mut BTreeMap::new()).unwrap();
    let out = CpuBackend
        .execute(
            &g,
            values["y"],
            &HashMap::from([
                (
                    "x".into(),
                    TensorData::new([1, 2, 3, 4], vec![1.; 24]).unwrap(),
                ),
                (
                    "w".into(),
                    TensorData::new([2, 1, 2, 2], vec![1.; 8]).unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(out.shape().dims(), &[1, 2, 2, 5]);

    let mut g = Graph::new();
    let x = g.input("x", [1, 1, 3, 3]);
    let w = g.input("w", [1, 1, 2, 2]);
    let mut values = BTreeMap::from([("x".into(), x), ("w".into(), w)]);
    let mut n = node("Conv", &["x", "w"], "y");
    field(&mut n, 5, &string_attr("auto_pad", "SAME_LOWER"));
    field(&mut n, 5, &ints_attr("strides", &[2, 2]));
    lower(&mut g, Msg::new(&n), &mut values, &mut BTreeMap::new()).unwrap();
    let out = CpuBackend
        .execute(
            &g,
            values["y"],
            &HashMap::from([
                (
                    "x".into(),
                    TensorData::new([1, 1, 3, 3], vec![1.; 9]).unwrap(),
                ),
                (
                    "w".into(),
                    TensorData::new([1, 1, 2, 2], vec![1.; 4]).unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(out.shape().dims(), &[1, 1, 2, 2]);
}
#[test]
fn conv_attributes_reject_bad_lengths_and_pad_conflicts() {
    let mut g = Graph::new();
    let x = g.input("x", [1, 1, 2, 2]);
    let w = g.input("w", [1, 1, 1, 1]);
    let mut values = BTreeMap::from([("x".into(), x), ("w".into(), w)]);
    let mut n = node("Conv", &["x", "w"], "y");
    field(&mut n, 5, &ints_attr("strides", &[1]));
    assert!(lower(&mut g, Msg::new(&n), &mut values, &mut BTreeMap::new()).is_err());
    let mut n = node("Conv", &["x", "w"], "z");
    field(&mut n, 5, &string_attr("auto_pad", "VALID"));
    field(&mut n, 5, &ints_attr("pads", &[0, 0, 0, 0]));
    assert!(lower(&mut g, Msg::new(&n), &mut values, &mut BTreeMap::new()).is_err());
}

#[test]
fn conv_preflights_closed_attributes_and_weight_kernel_identity() {
    let mut g = Graph::new();
    let x = g.input("x", [1, 1, 1, 1]);
    let w = g.input("w", [1, 1, 1, 1]);
    let mut values = BTreeMap::from([("x".into(), x), ("w".into(), w)]);
    let before_values = values.clone();
    let before_nodes = g.node_count();

    for (case, attribute) in [
        ("unknown", int_attr("output_padding", 1)),
        ("kernel", ints_attr("kernel_shape", &[2, 1])),
    ] {
        let mut invalid = node("Conv", &["x", "w"], "out");
        field(&mut invalid, 5, &attribute);
        assert!(
            lower(&mut g, Msg::new(&invalid), &mut values, &mut BTreeMap::new()).is_err(),
            "{case}"
        );
        assert_eq!(values, before_values, "{case}");
        assert_eq!(g.node_count(), before_nodes, "{case}");
    }

    let mut valid = node("Conv", &["x", "w"], "valid");
    field(&mut valid, 5, &ints_attr("kernel_shape", &[1, 1]));
    lower(&mut g, Msg::new(&valid), &mut values, &mut BTreeMap::new()).unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["valid"],
            &HashMap::from([
                ("x".into(), TensorData::new([1, 1, 1, 1], vec![3.]).unwrap()),
                ("w".into(), TensorData::new([1, 1, 1, 1], vec![2.]).unwrap()),
            ]),
        )
        .unwrap();
    assert_eq!(output.values(), &[6.]);
}

#[test]
fn batch_norm_and_global_average_pool_lower_through_cpu_graph() {
    let mut g = Graph::new();
    let x = g.input("x", [1, 2, 2, 2]);
    let scale = g.input("scale", [2]);
    let bias = g.input("bias", [2]);
    let mean = g.input("mean", [2]);
    let variance = g.input("variance", [2]);
    let mut values = BTreeMap::from([
        ("x".into(), x),
        ("scale".into(), scale),
        ("bias".into(), bias),
        ("mean".into(), mean),
        ("variance".into(), variance),
    ]);
    let mut bn = node(
        "BatchNormalization",
        &["x", "scale", "bias", "mean", "variance"],
        "bn",
    );
    field(&mut bn, 5, &fattr("epsilon", 0.));
    lower(&mut g, Msg::new(&bn), &mut values, &mut BTreeMap::new()).unwrap();
    lower(
        &mut g,
        Msg::new(&node("Relu", &["bn"], "relu")),
        &mut values,
        &mut BTreeMap::new(),
    )
    .unwrap();
    lower(
        &mut g,
        Msg::new(&node("GlobalAveragePool", &["relu"], "y")),
        &mut values,
        &mut BTreeMap::new(),
    )
    .unwrap();
    let out = CpuBackend
        .execute(
            &g,
            values["y"],
            &HashMap::from([
                (
                    "x".into(),
                    TensorData::new([1, 2, 2, 2], vec![1., 2., 3., 4., 5., 6., 7., 8.]).unwrap(),
                ),
                ("scale".into(), TensorData::new([2], vec![2., 1.]).unwrap()),
                ("bias".into(), TensorData::new([2], vec![0., -1.]).unwrap()),
                ("mean".into(), TensorData::new([2], vec![1., 5.]).unwrap()),
                (
                    "variance".into(),
                    TensorData::new([2], vec![1., 1.]).unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(out.shape().dims(), &[1, 2, 1, 1]);
    assert_eq!(out.values(), &[3., 0.75]);
}
#[test]
fn batch_norm_rejects_training_outputs_and_bad_parameter_contracts() {
    let mut g = Graph::new();
    let x = g.input("x", [1, 2, 1, 1]);
    let p = g.input("p", [2]);
    let mut values = BTreeMap::from([("x".into(), x), ("p".into(), p)]);
    let mut n = node("BatchNormalization", &["x", "p", "p", "p", "p"], "y");
    field(&mut n, 5, &int_attr("training_mode", 1));
    assert!(lower(&mut g, Msg::new(&n), &mut values, &mut BTreeMap::new()).is_err());
    let mut g = Graph::new();
    let x = g.input("x", [1, 2]);
    let p = g.input("p", [1]);
    let mut values = BTreeMap::from([("x".into(), x), ("p".into(), p)]);
    assert!(
        lower(
            &mut g,
            Msg::new(&node("BatchNormalization", &["x", "p", "p", "p", "p"], "y")),
            &mut values,
            &mut BTreeMap::new()
        )
        .is_err()
    );
    assert!(
        lower(
            &mut g,
            Msg::new(&node("GlobalAveragePool", &["x"], "z")),
            &mut values,
            &mut BTreeMap::new()
        )
        .is_err()
    );
}
#[test]
fn static_pools_lower_with_border_and_same_geometry() {
    let mut g = Graph::new();
    let x = g.input("x", [1, 1, 2, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut max = node("MaxPool", &["x"], "max");
    field(&mut max, 5, &ints_attr("kernel_shape", &[2, 2]));
    lower(&mut g, Msg::new(&max), &mut values, &mut BTreeMap::new()).unwrap();
    let mut avg = node("AveragePool", &["x"], "avg");
    field(&mut avg, 5, &ints_attr("kernel_shape", &[2, 2]));
    field(&mut avg, 5, &ints_attr("pads", &[1, 1, 1, 1]));
    lower(&mut g, Msg::new(&avg), &mut values, &mut BTreeMap::new()).unwrap();
    let mut same = node("MaxPool", &["x"], "same");
    field(&mut same, 5, &ints_attr("kernel_shape", &[2, 2]));
    field(&mut same, 5, &string_attr("auto_pad", "SAME_UPPER"));
    lower(&mut g, Msg::new(&same), &mut values, &mut BTreeMap::new()).unwrap();
    let inputs = HashMap::from([(
        "x".into(),
        TensorData::new([1, 1, 2, 2], vec![1., 2., 3., 4.]).unwrap(),
    )]);
    assert_eq!(
        CpuBackend
            .execute(&g, values["max"], &inputs)
            .unwrap()
            .values(),
        &[4.]
    );
    assert_eq!(
        CpuBackend
            .execute(&g, values["avg"], &inputs)
            .unwrap()
            .values(),
        &[1., 1.5, 2., 2., 2.5, 3., 3., 3.5, 4.]
    );
    assert_eq!(
        CpuBackend
            .execute(&g, values["same"], &inputs)
            .unwrap()
            .shape()
            .dims(),
        &[1, 1, 2, 2]
    );
}

#[test]
fn average_pool_accepts_dilation_and_preflights_storage_order() {
    let mut g = Graph::new();
    let x = g.input("x", [1, 1, 3, 3]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut valid = node("AveragePool", &["x"], "out");
    field(&mut valid, 5, &ints_attr("kernel_shape", &[2, 2]));
    field(&mut valid, 5, &ints_attr("dilations", &[2, 2]));
    lower(&mut g, Msg::new(&valid), &mut values, &mut BTreeMap::new()).unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new(
                    [1, 1, 3, 3],
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[1, 1, 1, 1]);
    assert_eq!(output.values(), &[5.0]);

    let mut invalid = Graph::new();
    let x = invalid.input("x", [1, 1, 3, 3]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let before_values = values.clone();
    let before_nodes = invalid.node_count();
    let mut malformed = node("AveragePool", &["x"], "out");
    field(&mut malformed, 5, &ints_attr("kernel_shape", &[2, 2]));
    field(&mut malformed, 5, &int_attr("storage_order", 1));
    assert!(
        lower(
            &mut invalid,
            Msg::new(&malformed),
            &mut values,
            &mut BTreeMap::new(),
        )
        .is_err()
    );
    assert_eq!(values, before_values);
    assert_eq!(invalid.node_count(), before_nodes);
}

#[test]
fn pools_reject_missing_bad_and_indices_contracts() {
    let mut g = Graph::new();
    let x = g.input("x", [1, 1, 2, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    assert!(
        lower(
            &mut g,
            Msg::new(&node("MaxPool", &["x"], "a")),
            &mut values,
            &mut BTreeMap::new()
        )
        .is_err()
    );
    let mut bad = node("AveragePool", &["x"], "b");
    field(&mut bad, 5, &ints_attr("kernel_shape", &[2, 2]));
    field(&mut bad, 5, &int_attr("storage_order", 1));
    assert!(lower(&mut g, Msg::new(&bad), &mut values, &mut BTreeMap::new()).is_err());
    let mut indexed = node("MaxPool", &["x"], "c");
    text(&mut indexed, 2, "indices");
    field(&mut indexed, 5, &ints_attr("kernel_shape", &[2, 2]));
    assert!(
        lower(
            &mut g,
            Msg::new(&indexed),
            &mut values,
            &mut BTreeMap::new()
        )
        .is_err()
    );
}

#[test]
fn max_pool_rejects_average_only_padding_control_before_publication() {
    let mut g = Graph::new();
    let x = g.input("x", [1, 1, 2, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let before_values = values.clone();
    let before_nodes = g.node_count();

    let mut invalid = node("MaxPool", &["x"], "out");
    field(&mut invalid, 5, &ints_attr("kernel_shape", &[2, 2]));
    field(&mut invalid, 5, &int_attr("count_include_pad", 1));
    assert!(lower(&mut g, Msg::new(&invalid), &mut values, &mut BTreeMap::new()).is_err());
    assert_eq!(values, before_values);
    assert_eq!(g.node_count(), before_nodes);

    let mut valid = node("MaxPool", &["x"], "valid");
    field(&mut valid, 5, &ints_attr("kernel_shape", &[2, 2]));
    field(&mut valid, 5, &int_attr("ceil_mode", 0));
    lower(&mut g, Msg::new(&valid), &mut values, &mut BTreeMap::new()).unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["valid"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([1, 1, 2, 2], vec![1., 2., 3., 4.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.values(), &[4.]);
}

#[test]
fn static_predicates_math_clip_and_inference_dropout_lower() {
    let mut g = Graph::new();
    let x = g.input("x", [2]);
    let y = g.input("y", [2]);
    let lo = TensorData::scalar(-1.0f32);
    let hi = TensorData::scalar(1.0f32);
    let ratio = TensorData::scalar(0.0f32);
    let training = TensorData::scalar_with_dtype(crate::Scalar::Bool(false), DType::Bool);
    let mut constants = BTreeMap::from([
        ("lo".into(), lo.clone()),
        ("hi".into(), hi.clone()),
        ("ratio".into(), ratio.clone()),
        ("training".into(), training.clone()),
    ]);
    let mut values = BTreeMap::from([("x".into(), x), ("y".into(), y)]);
    for (name, value) in [
        ("lo", lo),
        ("hi", hi),
        ("ratio", ratio),
        ("training", training),
    ] {
        values.insert(name.into(), g.constant(value));
    }
    lower(
        &mut g,
        Msg::new(&node("Greater", &["x", "y"], "p")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    lower(
        &mut g,
        Msg::new(&node("Where", &["p", "x", "y"], "w")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let mut leaky = node("LeakyRelu", &["w"], "l");
    field(&mut leaky, 5, &fattr("alpha", 0.5));
    lower(&mut g, Msg::new(&leaky), &mut values, &mut constants).unwrap();
    lower(
        &mut g,
        Msg::new(&node("Clip", &["l", "lo", "hi"], "c")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    lower(
        &mut g,
        Msg::new(&node("Dropout", &["c", "ratio", "training"], "d")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let out = CpuBackend
        .execute(
            &g,
            values["d"],
            &HashMap::from([
                ("x".into(), TensorData::new([2], vec![-4., 2.]).unwrap()),
                ("y".into(), TensorData::new([2], vec![3., 1.]).unwrap()),
            ]),
        )
        .unwrap();
    assert_eq!(out.values(), &[1., 1.]);
}

#[test]
fn pow_restores_integer_base_dtype_before_publication() {
    let mut g = Graph::new();
    let base = g.input_dtype("base", [2], DType::I32);
    let exponent = g.input("exponent", [2]);
    let mut values = BTreeMap::from([("base".into(), base), ("exponent".into(), exponent)]);
    lower(
        &mut g,
        Msg::new(&node("Pow", &["base", "exponent"], "out")),
        &mut values,
        &mut BTreeMap::new(),
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([
                (
                    "base".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::I32,
                        [Scalar::I(2), Scalar::I(3)],
                    )
                    .unwrap(),
                ),
                (
                    "exponent".into(),
                    TensorData::new([2], vec![2.0, 0.5]).unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::I32);
    assert_eq!(output.to_vec_f64(), vec![4.0, 2.0]);

    let mut malformed = Graph::new();
    let base = malformed.input_dtype("base", [2], DType::I32);
    let exponent = malformed.input("exponent", [3]);
    let mut values = BTreeMap::from([("base".into(), base), ("exponent".into(), exponent)]);
    let before_values = values.clone();
    let before_nodes = malformed.node_count();
    assert!(lower(
        &mut malformed,
        Msg::new(&node("Pow", &["base", "exponent"], "out")),
        &mut values,
        &mut BTreeMap::new(),
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(malformed.node_count(), before_nodes);
}

#[test]
fn div_rejects_fmod_attribute_before_publication() {
    let mut g = Graph::new();
    let lhs = g.input_dtype("lhs", [2], DType::I32);
    let rhs = g.input_dtype("rhs", [2], DType::I32);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    lower(
        &mut g,
        Msg::new(&node("Div", &["lhs", "rhs"], "out")),
        &mut values,
        &mut BTreeMap::new(),
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::I32,
                        [Scalar::I(-7), Scalar::I(7)],
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::I32,
                        [Scalar::I(3), Scalar::I(3)],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::I32);
    assert_eq!(output.to_vec_f64(), vec![-2.0, 2.0]);

    let mut invalid = Graph::new();
    let lhs = invalid.input_dtype("lhs", [2], DType::I32);
    let rhs = invalid.input_dtype("rhs", [2], DType::I32);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let before_values = values.clone();
    let before_nodes = invalid.node_count();
    let mut node = node("Div", &["lhs", "rhs"], "out");
    field(&mut node, 5, &int_attr("fmod", 1));
    assert!(
        lower(
            &mut invalid,
            Msg::new(&node),
            &mut values,
            &mut BTreeMap::new(),
        )
        .is_err()
    );
    assert_eq!(values, before_values);
    assert_eq!(invalid.node_count(), before_nodes);
}

#[test]
fn leaky_relu_keeps_fractional_alpha_for_integer_input() {
    let mut g = Graph::new();
    let x = g.input_dtype("x", [2], DType::I32);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut valid = node("LeakyRelu", &["x"], "out");
    field(&mut valid, 5, &fattr("alpha", 0.5));
    lower(&mut g, Msg::new(&valid), &mut values, &mut BTreeMap::new()).unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::from_scalars([2], DType::I32, [Scalar::I(-2), Scalar::I(2)])
                    .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.to_vec_f64(), vec![-1.0, 2.0]);

    let mut invalid = Graph::new();
    let x = invalid.input_dtype("x", [2], DType::I32);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let before_values = values.clone();
    let before_nodes = invalid.node_count();
    let mut malformed = node("LeakyRelu", &["x"], "out");
    field(&mut malformed, 5, &fattr("alpha", f32::NAN));
    assert!(
        lower(
            &mut invalid,
            Msg::new(&malformed),
            &mut values,
            &mut BTreeMap::new(),
        )
        .is_err()
    );
    assert_eq!(values, before_values);
    assert_eq!(invalid.node_count(), before_nodes);
}

#[test]
fn gather_normalizes_constant_negative_scalar_index_before_lowering() {
    let mut g = Graph::new();
    let x = g.input("x", [1, 3, 3]);
    let indices = TensorData::from_scalars([], DType::I64, [Scalar::I(-2)]).unwrap();
    let index = g.constant(indices.clone());
    let mut values = BTreeMap::from([("x".into(), x), ("indices".into(), index)]);
    let mut constants = BTreeMap::from([("indices".into(), indices)]);
    let mut valid = node("Gather", &["x", "indices"], "out");
    field(&mut valid, 5, &int_attr("axis", 1));
    lower(&mut g, Msg::new(&valid), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([1, 3, 3], (0..9).map(|x| x as f32).collect()).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[1, 3]);
    assert_eq!(output.values(), &[3.0, 4.0, 5.0]);

    let mut invalid = Graph::new();
    let x = invalid.input("x", [1, 3, 3]);
    let indices = TensorData::from_scalars([], DType::I64, [Scalar::I(-4)]).unwrap();
    let index = invalid.constant(indices.clone());
    let mut values = BTreeMap::from([("x".into(), x), ("indices".into(), index)]);
    let mut constants = BTreeMap::from([("indices".into(), indices)]);
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = invalid.node_count();
    let mut malformed = node("Gather", &["x", "indices"], "out");
    field(&mut malformed, 5, &int_attr("axis", 1));
    assert!(
        lower(
            &mut invalid,
            Msg::new(&malformed),
            &mut values,
            &mut constants,
        )
        .is_err()
    );
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(invalid.node_count(), before_nodes);
}

#[test]
fn concat_rejects_unknown_attributes_before_publication() {
    let mut g = Graph::new();
    let lhs = g.input("lhs", [1, 2]);
    let rhs = g.input("rhs", [1, 1]);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let mut valid = node("Concat", &["lhs", "rhs"], "out");
    field(&mut valid, 5, &int_attr("axis", 1));
    lower(&mut g, Msg::new(&valid), &mut values, &mut BTreeMap::new()).unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([
                ("lhs".into(), TensorData::new([1, 2], vec![1.0, 2.0]).unwrap()),
                ("rhs".into(), TensorData::new([1, 1], vec![3.0]).unwrap()),
            ]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[1, 3]);
    assert_eq!(output.values(), &[1.0, 2.0, 3.0]);

    let mut invalid = Graph::new();
    let lhs = invalid.input("lhs", [1, 2]);
    let rhs = invalid.input("rhs", [1, 1]);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let before_values = values.clone();
    let before_nodes = invalid.node_count();
    let mut malformed = node("Concat", &["lhs", "rhs"], "out");
    field(&mut malformed, 5, &int_attr("axis", 1));
    field(&mut malformed, 5, &int_attr("unexpected", 0));
    assert!(
        lower(
            &mut invalid,
            Msg::new(&malformed),
            &mut values,
            &mut BTreeMap::new(),
        )
        .is_err()
    );
    assert_eq!(values, before_values);
    assert_eq!(invalid.node_count(), before_nodes);

    let mut overflow = Graph::new();
    let lhs = overflow.input("lhs", [usize::MAX, 2]);
    let rhs = overflow.input("rhs", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let before_values = values.clone();
    let before_nodes = overflow.node_count();
    let mut oversized = node("Concat", &["lhs", "rhs"], "out");
    field(&mut oversized, 5, &int_attr("axis", 1));
    assert!(
        lower(
            &mut overflow,
            Msg::new(&oversized),
            &mut values,
            &mut BTreeMap::new(),
        )
        .is_err()
    );
    assert_eq!(values, before_values);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn matmul_rejects_attributes_before_publication() {
    let mut g = Graph::new();
    let lhs = g.input("lhs", [1, 2, 3]);
    let rhs = g.input("rhs", [3, 2]);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    lower(
        &mut g,
        Msg::new(&node("MatMul", &["lhs", "rhs"], "out")),
        &mut values,
        &mut BTreeMap::new(),
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::new([1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                        .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::new([3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
                        .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[1, 2, 2]);
    assert_eq!(output.values(), &[58.0, 64.0, 139.0, 154.0]);

    let mut invalid = Graph::new();
    let lhs = invalid.input("lhs", [1, 2, 3]);
    let rhs = invalid.input("rhs", [3, 2]);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let before_values = values.clone();
    let before_nodes = invalid.node_count();
    let mut malformed = node("MatMul", &["lhs", "rhs"], "out");
    field(&mut malformed, 5, &int_attr("unexpected", 0));
    assert!(
        lower(
            &mut invalid,
            Msg::new(&malformed),
            &mut values,
            &mut BTreeMap::new(),
        )
        .is_err()
    );
    assert_eq!(values, before_values);
    assert_eq!(invalid.node_count(), before_nodes);
}

#[test]
fn static_phase_four_rejects_dynamic_clip_and_dropout_training() {
    let mut g = Graph::new();
    let x = g.input("x", [1]);
    let b = g.input("b", []);
    let mut values = BTreeMap::from([("x".into(), x), ("b".into(), b)]);
    assert!(
        lower(
            &mut g,
            Msg::new(&node("Clip", &["x", "b"], "c")),
            &mut values,
            &mut BTreeMap::new()
        )
        .is_err()
    );
    assert!(
        lower(
            &mut g,
            Msg::new(&node("Dropout", &["x", "b"], "d")),
            &mut values,
            &mut BTreeMap::new()
        )
        .is_err()
    );
}

#[test]
fn clip_without_bounds_is_identity_and_malformed_bounds_do_not_publish() {
    let mut g = Graph::new();
    let x = g.input("x", [2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let before_nodes = g.node_count();
    lower(
        &mut g,
        Msg::new(&node("Clip", &["x"], "identity")),
        &mut values,
        &mut BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(values["identity"], x);
    assert_eq!(g.node_count(), before_nodes);
    let output = CpuBackend
        .execute(
            &g,
            values["identity"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2], vec![-2., 3.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.values(), &[-2., 3.]);

    let before_values = values.clone();
    let mut constants = BTreeMap::from([(
        "bad".into(),
        TensorData::from_scalars([], DType::I64, [Scalar::I(1)]).unwrap(),
    )]);
    let before_constants = constants.clone();
    for (case, invalid) in [
        ("dtype", node("Clip", &["x", "bad"], "out")),
        ("attribute", {
            let mut node = node("Clip", &["x"], "out");
            field(&mut node, 5, &int_attr("axis", 0));
            node
        }),
    ] {
        assert!(
            lower(
                &mut g,
                Msg::new(&invalid),
                &mut values,
                &mut constants,
            )
            .is_err(),
            "{case}"
        );
        assert_eq!(values, before_values, "{case}");
        assert_eq!(constants, before_constants, "{case}");
        assert_eq!(g.node_count(), before_nodes, "{case}");
    }
}
#[test]
fn embedded_tensor_attributes_may_be_unnamed_but_initializers_may_not() {
    let mut unnamed = vec![];
    var(&mut unnamed, 2, 1);
    field(&mut unnamed, 9, &3.5f32.to_le_bytes());
    assert_eq!(tensor_data(Msg::new(&unnamed)).unwrap().values(), &[3.5]);
    assert!(super::tensor(Msg::new(&unnamed)).is_err());
    let named = tensor("named", &[], &[3.5]);
    assert!(super::tensor(Msg::new(&named)).is_ok());
}

#[test]
fn constant_and_cast_reject_duplicate_attribute_values_before_publication() {
    let embedded = tensor("embedded", &[1], &[3.5]);
    let mut constant_value = tensor_attr("value", &embedded);
    field(&mut constant_value, 5, &embedded);
    let mut invalid_constant = node("Constant", &[], "constant");
    field(&mut invalid_constant, 5, &constant_value);
    let mut constant_graph = Graph::new();
    let mut constant_values = BTreeMap::new();
    let mut constants = BTreeMap::new();
    assert!(
        lower(
            &mut constant_graph,
            Msg::new(&invalid_constant),
            &mut constant_values,
            &mut constants,
        )
        .is_err()
    );
    assert!(constant_values.is_empty());
    assert!(constants.is_empty());
    assert_eq!(constant_graph.node_count(), 0);

    let mut cast_to = int_attr("to", 6);
    var(&mut cast_to, 3, 11);
    let mut invalid_cast = node("Cast", &["x"], "out");
    field(&mut invalid_cast, 5, &cast_to);
    let mut cast_graph = Graph::new();
    let x = cast_graph.input("x", [1]);
    let mut cast_values = BTreeMap::from([("x".into(), x)]);
    let before_values = cast_values.clone();
    let before_nodes = cast_graph.node_count();
    assert!(
        lower(
            &mut cast_graph,
            Msg::new(&invalid_cast),
            &mut cast_values,
            &mut BTreeMap::new(),
        )
        .is_err()
    );
    assert_eq!(cast_values, before_values);
    assert_eq!(cast_graph.node_count(), before_nodes);

    let mut valid_constant = node("Constant", &[], "constant");
    field(&mut valid_constant, 5, &tensor_attr("value", &embedded));
    lower(
        &mut constant_graph,
        Msg::new(&valid_constant),
        &mut constant_values,
        &mut constants,
    )
    .unwrap();
    let mut valid_cast = node("Cast", &["constant"], "cast");
    field(&mut valid_cast, 5, &int_attr("to", 6));
    lower(
        &mut constant_graph,
        Msg::new(&valid_cast),
        &mut constant_values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(&constant_graph, constant_values["cast"], &HashMap::new())
        .unwrap();
    assert_eq!(output.dtype(), DType::I32);
    assert_eq!(output.scalar_at(0).as_i64(), 3);
}
#[test]
fn reductions_and_arg_reject_dynamic_and_last_tie_controls() {
    let mut g = Graph::new();
    let x = g.input("x", [2, 2]);
    let axes = g.input_dtype("axes", [1], DType::I64);
    let mut values = BTreeMap::from([("x".into(), x), ("axes".into(), axes)]);
    assert!(
        lower(
            &mut g,
            Msg::new(&node("ReduceSum", &["x", "axes"], "s")),
            &mut values,
            &mut BTreeMap::new()
        )
        .is_err()
    );
    let mut arg = node("ArgMax", &["x"], "a");
    field(&mut arg, 5, &int_attr("select_last_index", 1));
    assert!(lower(&mut g, Msg::new(&arg), &mut values, &mut BTreeMap::new()).is_err());
}

#[test]
fn reductions_preflight_ranked_axes_before_publication() {
    let mut g = Graph::new();
    let x = g.input("x", [2, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let before_values = values.clone();
    let before_nodes = g.node_count();

    for (case, axes) in [
        (
            "scalar",
            TensorData::from_scalars([], DType::I64, [Scalar::I(1)]).unwrap(),
        ),
        (
            "matrix",
            TensorData::from_scalars([1, 1], DType::I64, [Scalar::I(1)]).unwrap(),
        ),
    ] {
        let mut constants = BTreeMap::from([("axes".into(), axes)]);
        let before_constants = constants.clone();
        assert!(
            lower(
                &mut g,
                Msg::new(&node("ReduceSum", &["x", "axes"], "out")),
                &mut values,
                &mut constants,
            )
            .is_err(),
            "{case}"
        );
        assert_eq!(values, before_values, "{case}");
        assert_eq!(constants, before_constants, "{case}");
        assert_eq!(g.node_count(), before_nodes, "{case}");
    }

    let axes = TensorData::from_scalars([1], DType::I64, [Scalar::I(-1)]).unwrap();
    let mut constants = BTreeMap::from([("axes".into(), axes)]);
    lower(
        &mut g,
        Msg::new(&node("ReduceSum", &["x", "axes"], "sum")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["sum"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2]);
    assert_eq!(output.values(), &[3., 7.]);
}
#[test]
fn static_reductions_and_args_have_checked_cpu_numerics() {
    let cases = [
        ("ReduceSum", 10.),
        ("ReduceMean", 2.5),
        ("ReduceProd", 24.),
        ("ReduceMin", 1.),
        ("ReduceMax", 4.),
    ];
    for (op, expected) in cases {
        let mut g = Graph::new();
        let x = g.input("x", [2, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        lower(
            &mut g,
            Msg::new(&node(op, &["x"], "y")),
            &mut values,
            &mut BTreeMap::new(),
        )
        .unwrap();
        let y = CpuBackend
            .execute(
                &g,
                values["y"],
                &HashMap::from([(
                    "x".into(),
                    TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(y.values(), &[expected], "{op}");
    }
    let mut g = Graph::new();
    let x = g.input("x", [2, 2]);
    let axes = TensorData::from_scalars([1], DType::I64, [Scalar::I(-1)]).unwrap();
    let mut constants = BTreeMap::from([("axes".into(), axes.clone())]);
    let mut values = BTreeMap::from([("x".into(), x), ("axes".into(), g.constant(axes))]);
    lower(
        &mut g,
        Msg::new(&node("ReduceSum", &["x", "axes"], "sum")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    lower(
        &mut g,
        Msg::new(&node("ArgMax", &["x"], "arg")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let input = HashMap::from([(
        "x".into(),
        TensorData::new([2, 2], vec![2., 2., 1., 1.]).unwrap(),
    )]);
    assert_eq!(
        CpuBackend
            .execute(&g, values["sum"], &input)
            .unwrap()
            .values(),
        &[4., 2.]
    );
    let arg = CpuBackend.execute(&g, values["arg"], &input).unwrap();
    assert_eq!(arg.dtype(), DType::I64);
    assert_eq!(arg.scalar_at(0).as_i64(), 0);
}

#[test]
fn model_proto_constant_of_shape_defaults_and_typed_scalar_are_exact() {
    let shape = raw_tensor("shape", &[2], 7, &i64_bytes(&[2, 1]));
    let default = node("ConstantOfShape", &["shape"], "default");
    let mut typed = node("ConstantOfShape", &["shape"], "typed");
    field(
        &mut typed,
        5,
        &tensor_attr("value", &typed_i64_tensor(&[], &[9])),
    );
    let bytes = model_proto(
        &[shape],
        &[default, typed],
        &[
            value_dtype("default", &[2, 1], 1),
            value_dtype("typed", &[2, 1], 7),
        ],
    );

    let outputs = import_onnx(&bytes).unwrap().run(HashMap::new()).unwrap();
    let default = &outputs["default"];
    assert_eq!(default.shape().dims(), &[2, 1]);
    assert_eq!(default.dtype(), DType::F32);
    assert_eq!(default.values(), &[0., 0.]);
    let typed = &outputs["typed"];
    assert_eq!(typed.shape().dims(), &[2, 1]);
    assert_eq!(typed.dtype(), DType::I64);
    assert_eq!(typed.scalar_at(0).as_i64(), 9);
    assert_eq!(typed.scalar_at(1).as_i64(), 9);
}

#[test]
fn model_proto_constant_of_shape_rejects_bad_embedded_value_count_and_type() {
    let shape = raw_tensor("shape", &[1], 7, &i64_bytes(&[2]));
    let cases = [
        (
            "count",
            typed_i64_tensor(&[2], &[3, 4]),
            "ConstantOfShape value must contain one element",
        ),
        (
            "type",
            raw_tensor("", &[], 8, b"x"),
            "unsupported ONNX dtype",
        ),
    ];
    for (case, embedded, expected) in cases {
        let mut constant = node("ConstantOfShape", &["shape"], "y");
        field(&mut constant, 5, &tensor_attr("value", &embedded));
        let bytes = model_proto(
            std::slice::from_ref(&shape),
            &[constant],
            &[value_dtype("y", &[2], 1)],
        );
        match import_onnx(&bytes) {
            Err(Error::ModelIo { reason }) => assert_eq!(reason, expected, "{case}"),
            Err(error) => panic!("{case}: unexpected error {error}"),
            Ok(_) => panic!("{case}: malformed embedded value was accepted"),
        }
    }
}

#[test]
fn model_proto_arg_reductions_keep_first_ties_normalize_axis_and_return_i64() {
    let x = raw_tensor("x", &[2, 3], 1, &f32_bytes(&[3., 3., 1., -2., -2., 4.]));
    let mut maximum = node("ArgMax", &["x"], "maximum");
    field(&mut maximum, 5, &int64_attr("axis", -1));
    field(&mut maximum, 5, &int_attr("keepdims", 0));
    let mut minimum = node("ArgMin", &["x"], "minimum");
    field(&mut minimum, 5, &int64_attr("axis", -1));
    field(&mut minimum, 5, &int_attr("keepdims", 1));
    let bytes = model_proto(
        &[x],
        &[maximum, minimum],
        &[
            value_dtype("maximum", &[2], 7),
            value_dtype("minimum", &[2, 1], 7),
        ],
    );

    let outputs = import_onnx(&bytes).unwrap().run(HashMap::new()).unwrap();
    let maximum = &outputs["maximum"];
    assert_eq!(maximum.shape().dims(), &[2]);
    assert_eq!(maximum.dtype(), DType::I64);
    assert_eq!(maximum.scalar_at(0).as_i64(), 0);
    assert_eq!(maximum.scalar_at(1).as_i64(), 2);
    let minimum = &outputs["minimum"];
    assert_eq!(minimum.shape().dims(), &[2, 1]);
    assert_eq!(minimum.dtype(), DType::I64);
    assert_eq!(minimum.scalar_at(0).as_i64(), 2);
    assert_eq!(minimum.scalar_at(1).as_i64(), 0);
}

#[test]
fn model_proto_reduction_boundaries_cover_noop_nan_and_zero_domains() {
    let x = raw_tensor("x", &[2, 2], 1, &f32_bytes(&[1., 2., 3., 4.]));
    let empty_axes = raw_tensor("axes", &[0], 7, &[]);
    let mut noop = node("ReduceSum", &["x", "axes"], "y");
    field(&mut noop, 5, &int_attr("noop_with_empty_axes", 1));
    let noop_bytes = model_proto(&[x, empty_axes], &[noop], &[value_dtype("y", &[2, 2], 1)]);
    let noop = import_onnx(&noop_bytes)
        .unwrap()
        .run(HashMap::new())
        .unwrap();
    assert_eq!(noop["y"].shape().dims(), &[2, 2]);
    assert_eq!(noop["y"].values(), &[1., 2., 3., 4.]);

    let nan_x = raw_tensor("x", &[3], 1, &f32_bytes(&[f32::NAN, 2., -1.]));
    let nan_bytes = model_proto(
        &[nan_x],
        &[
            node("ReduceMin", &["x"], "minimum"),
            node("ReduceMax", &["x"], "maximum"),
        ],
        &[
            value_dtype("minimum", &[], 1),
            value_dtype("maximum", &[], 1),
        ],
    );
    let extrema = import_onnx(&nan_bytes)
        .unwrap()
        .run(HashMap::new())
        .unwrap();
    assert_eq!(extrema["minimum"].values(), &[-1.]);
    assert_eq!(extrema["maximum"].values(), &[2.]);

    let empty = raw_tensor("x", &[2, 0], 1, &[]);
    let axis = raw_tensor("axis", &[1], 7, &i64_bytes(&[1]));
    let zero_bytes = model_proto(
        &[empty, axis],
        &[
            node("ReduceSum", &["x", "axis"], "sum"),
            node("ReduceMean", &["x", "axis"], "mean"),
            node("ReduceProd", &["x", "axis"], "product"),
        ],
        &[
            value_dtype("sum", &[2], 1),
            value_dtype("mean", &[2], 1),
            value_dtype("product", &[2], 1),
        ],
    );
    let zero = import_onnx(&zero_bytes)
        .unwrap()
        .run(HashMap::new())
        .unwrap();
    assert_eq!(zero["sum"].values(), &[0., 0.]);
    assert!(zero["mean"].values().iter().all(|x| x.is_nan()));
    assert_eq!(zero["product"].values(), &[1., 1.]);
}

#[test]
fn model_proto_rejects_duplicate_initializers_and_invalid_axes_tensors() {
    let duplicate = raw_tensor("duplicate", &[1], 1, &f32_bytes(&[1.]));
    let bytes = model_proto(
        &[duplicate.clone(), duplicate],
        &[],
        &[value_dtype("duplicate", &[1], 1)],
    );
    match import_onnx(&bytes) {
        Err(Error::ModelIo { reason }) => assert_eq!(reason, "duplicate ONNX initializer"),
        Err(error) => panic!("unexpected duplicate initializer error: {error}"),
        Ok(_) => panic!("duplicate initializer was accepted"),
    }

    let x = raw_tensor("x", &[2, 2], 1, &f32_bytes(&[1., 2., 3., 4.]));
    let invalid_axes = [
        (
            "duplicate",
            raw_tensor("axes", &[2], 7, &i64_bytes(&[0, -2])),
            "duplicate Reduce axis",
        ),
        (
            "dtype",
            raw_tensor("axes", &[1], 6, &1i32.to_le_bytes()),
            "ONNX shape/axes constant must be I64",
        ),
        (
            "count",
            raw_tensor("axes", &[2], 7, &i64_bytes(&[1])),
            "invalid ONNX tensor data",
        ),
    ];
    for (case, axes, expected) in invalid_axes {
        let bytes = model_proto(
            &[x.clone(), axes],
            &[node("ReduceSum", &["x", "axes"], "y")],
            &[value_dtype("y", &[2], 1)],
        );
        match import_onnx(&bytes) {
            Err(Error::ModelIo { reason }) => {
                assert!(reason.contains(expected), "{case}: {reason}")
            }
            Err(error) => panic!("{case}: unexpected axes error {error}"),
            Ok(_) => panic!("{case}: invalid axes tensor was accepted"),
        }
    }
}

#[test]
fn pad_supports_signed_constant_crop_and_preflights_pads_rank() {
    let mut g = Graph::new();
    let x = g.input("x", [2, 3]);
    let pads = TensorData::from_scalars(
        [4],
        DType::I64,
        [Scalar::I(-1), Scalar::I(1), Scalar::I(1), Scalar::I(-1)],
    )
    .unwrap();
    let fill = TensorData::scalar_with_dtype(Scalar::F(9.), DType::F32);
    let pads_node = g.constant(pads.clone());
    let fill_node = g.constant(fill.clone());
    let mut values = BTreeMap::from([
        ("x".into(), x),
        ("pads".into(), pads_node),
        ("fill".into(), fill_node),
    ]);
    let mut constants = BTreeMap::from([("pads".into(), pads), ("fill".into(), fill)]);
    let valid = node("Pad", &["x", "pads", "fill"], "out");
    lower(&mut g, Msg::new(&valid), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2, 3]);
    assert_eq!(output.values(), &[9., 4., 5., 9., 9., 9.]);

    let mut malformed = Graph::new();
    let x = malformed.input("x", [2, 3]);
    let pads = TensorData::from_scalars(
        [2, 2],
        DType::I64,
        [Scalar::I(0), Scalar::I(0), Scalar::I(0), Scalar::I(0)],
    )
    .unwrap();
    let pads_node = malformed.constant(pads.clone());
    let mut values = BTreeMap::from([("x".into(), x), ("pads".into(), pads_node)]);
    let mut constants = BTreeMap::from([("pads".into(), pads)]);
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = malformed.node_count();
    let invalid = node("Pad", &["x", "pads"], "out");
    assert!(lower(
        &mut malformed,
        Msg::new(&invalid),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(malformed.node_count(), before_nodes);
}

#[test]
fn expand_aligns_leading_rank_and_preflights_shape_rank() {
    let mut g = Graph::new();
    let x = g.input("x", [2, 1]);
    let shape = TensorData::from_scalars([1], DType::I64, [Scalar::I(3)]).unwrap();
    let shape_node = g.constant(shape.clone());
    let mut values = BTreeMap::from([("x".into(), x), ("shape".into(), shape_node)]);
    let mut constants = BTreeMap::from([("shape".into(), shape)]);
    lower(
        &mut g,
        Msg::new(&node("Expand", &["x", "shape"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 1], vec![1., 2.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2, 3]);
    assert_eq!(output.values(), &[1., 1., 1., 2., 2., 2.]);

    let mut malformed = Graph::new();
    let x = malformed.input("x", [2, 1]);
    let shape = TensorData::from_scalars([1, 1], DType::I64, [Scalar::I(2)]).unwrap();
    let shape_node = malformed.constant(shape.clone());
    let mut values = BTreeMap::from([("x".into(), x), ("shape".into(), shape_node)]);
    let mut constants = BTreeMap::from([("shape".into(), shape)]);
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = malformed.node_count();
    assert!(lower(
        &mut malformed,
        Msg::new(&node("Expand", &["x", "shape"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(malformed.node_count(), before_nodes);
}

#[test]
fn tile_preserves_repeat_order_and_preflights_repeats_rank() {
    let mut g = Graph::new();
    let x = g.input("x", [2, 2]);
    let repeats = TensorData::from_scalars([2], DType::I64, [Scalar::I(1), Scalar::I(2)]).unwrap();
    let repeats_node = g.constant(repeats.clone());
    let mut values = BTreeMap::from([("x".into(), x), ("repeats".into(), repeats_node)]);
    let mut constants = BTreeMap::from([("repeats".into(), repeats)]);
    lower(
        &mut g,
        Msg::new(&node("Tile", &["x", "repeats"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2, 4]);
    assert_eq!(output.values(), &[1., 2., 1., 2., 3., 4., 3., 4.]);

    let mut scalar = Graph::new();
    let input = scalar.input("scalar", []);
    let repeats = TensorData::from_scalars([0], DType::I64, []).unwrap();
    let repeats_node = scalar.constant(repeats.clone());
    let mut scalar_values = BTreeMap::from([
        ("scalar".into(), input),
        ("repeats".into(), repeats_node),
    ]);
    let mut scalar_constants = BTreeMap::from([("repeats".into(), repeats)]);
    lower(
        &mut scalar,
        Msg::new(&node("Tile", &["scalar", "repeats"], "out")),
        &mut scalar_values,
        &mut scalar_constants,
    )
    .unwrap();
    assert_eq!(scalar_values["out"], input);

    let mut malformed = Graph::new();
    let x = malformed.input("x", [2, 2]);
    let repeats = TensorData::from_scalars(
        [1, 2],
        DType::I64,
        [Scalar::I(1), Scalar::I(2)],
    )
    .unwrap();
    let repeats_node = malformed.constant(repeats.clone());
    let mut values = BTreeMap::from([("x".into(), x), ("repeats".into(), repeats_node)]);
    let mut constants = BTreeMap::from([("repeats".into(), repeats)]);
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = malformed.node_count();
    assert!(lower(
        &mut malformed,
        Msg::new(&node("Tile", &["x", "repeats"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(malformed.node_count(), before_nodes);
}

#[test]
fn constant_of_shape_preflights_shape_and_fill_broadcast_before_publication() {
    let mut g = Graph::new();
    let shape = TensorData::from_scalars([1], DType::I64, [Scalar::I(2)]).unwrap();
    let shape_node = g.constant(shape.clone());
    let mut values = BTreeMap::from([("shape".into(), shape_node)]);
    let mut constants = BTreeMap::from([("shape".into(), shape)]);
    let mut valid = node("ConstantOfShape", &["shape"], "out");
    field(
        &mut valid,
        5,
        &tensor_attr("value", &typed_i64_tensor(&[1], &[9])),
    );
    lower(&mut g, Msg::new(&valid), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(&g, values["out"], &HashMap::new())
        .unwrap();
    assert_eq!(output.shape().dims(), &[2]);
    assert_eq!(output.dtype(), DType::I64);
    assert_eq!(output.to_vec_f64(), vec![9., 9.]);

    let mut malformed = Graph::new();
    let shape = TensorData::from_scalars([1], DType::I64, [Scalar::I(2)]).unwrap();
    let shape_node = malformed.constant(shape.clone());
    let mut values = BTreeMap::from([("shape".into(), shape_node)]);
    let mut constants = BTreeMap::from([("shape".into(), shape)]);
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = malformed.node_count();
    let mut invalid = node("ConstantOfShape", &["shape"], "out");
    field(
        &mut invalid,
        5,
        &tensor_attr("value", &typed_i64_tensor(&[1, 1], &[9])),
    );
    assert!(lower(
        &mut malformed,
        Msg::new(&invalid),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(malformed.node_count(), before_nodes);
}

#[test]
fn shape_clamps_signed_endpoints_and_preflights_i64_dimensions() {
    let mut g = Graph::new();
    let x = g.input("x", [2, 3]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    let mut valid = node("Shape", &["x"], "out");
    field(&mut valid, 5, &int64_attr("start", -100));
    field(&mut valid, 5, &int64_attr("end", 100));
    lower(&mut g, Msg::new(&valid), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(&g, values["out"], &HashMap::new())
        .unwrap();
    assert_eq!(output.dtype(), DType::I64);
    assert_eq!(output.shape().dims(), &[2]);
    assert_eq!(output.to_vec_f64(), vec![2., 3.]);

    let mut malformed = Graph::new();
    let x = malformed.input("x", [usize::MAX]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = malformed.node_count();
    assert!(lower(
        &mut malformed,
        Msg::new(&node("Shape", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(malformed.node_count(), before_nodes);
}

#[test]
fn size_is_static_i64_and_preflights_before_publication() {
    for (shape, expected) in [(vec![2, 3], 6), (vec![], 1), (vec![2, 0], 0)] {
        let mut g = Graph::new();
        let x = g.input("x", shape);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        lower(
            &mut g,
            Msg::new(&node("Size", &["x"], "out")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        let output = CpuBackend
            .execute(&g, values["out"], &HashMap::new())
            .unwrap();
        assert_eq!(output.shape().dims(), &[]);
        assert_eq!(output.dtype(), DType::I64);
        assert_eq!(output.scalar_at(0).as_i64(), expected);
    }

    let mut attr = node("Size", &["x"], "out");
    field(&mut attr, 5, &int_attr("keepdims", 1));
    for invalid in [node("Size", &[], "out"), attr] {
        let mut g = Graph::new();
        let x = g.input("x", [2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = g.node_count();
        assert!(lower(&mut g, Msg::new(&invalid), &mut values, &mut constants).is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(g.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let x = overflow.input("x", [usize::MAX]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Size", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn not_matches_tinygrad_bool_cast_and_preflights_before_publication() {
    let mut g = Graph::new();
    let x = g.input("x", [2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut g,
        Msg::new(&node("Not", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([("x".into(), TensorData::new([2], vec![0., -2.]).unwrap())]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::Bool);
    assert!(output.scalar_at(0).as_bool());
    assert!(!output.scalar_at(1).as_bool());

    let mut attribute = node("Not", &["x"], "out");
    field(&mut attribute, 5, &int_attr("keepdims", 1));
    for invalid in [node("Not", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let x = overflow.input("x", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Not", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn isinf_matches_tinygrad_sign_selection_and_preflights_before_publication() {
    let mut g = Graph::new();
    let x = g.input("x", [3]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    let mut valid = node("IsInf", &["x"], "out");
    field(&mut valid, 5, &int_attr("detect_positive", 1));
    field(&mut valid, 5, &int_attr("detect_negative", 0));
    lower(&mut g, Msg::new(&valid), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([3], vec![f32::NEG_INFINITY, f32::INFINITY, 0.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::Bool);
    assert!(!output.scalar_at(0).as_bool());
    assert!(output.scalar_at(1).as_bool());
    assert!(!output.scalar_at(2).as_bool());

    let mut attribute = node("IsInf", &["x"], "out");
    field(&mut attribute, 5, &int_attr("keepdims", 1));
    for invalid in [node("IsInf", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let x = overflow.input("x", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("IsInf", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn isnan_matches_tinygrad_and_preflights_before_publication() {
    let mut g = Graph::new();
    let x = g.input("x", [2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut g,
        Msg::new(&node("IsNaN", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2], vec![f32::NAN, 1.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::Bool);
    assert!(output.scalar_at(0).as_bool());
    assert!(!output.scalar_at(1).as_bool());

    let mut integers = Graph::new();
    let x = integers.input_dtype("x", [1], DType::I32);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integers,
        Msg::new(&node("IsNaN", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &integers,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::from_scalars([1], DType::I32, [Scalar::I(7)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::Bool);
    assert!(!output.scalar_at(0).as_bool());

    let mut attribute = node("IsNaN", &["x"], "out");
    field(&mut attribute, 5, &int_attr("keepdims", 1));
    for invalid in [node("IsNaN", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let x = overflow.input("x", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("IsNaN", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn reciprocal_matches_tinygrad_and_preflights_before_publication() {
    let mut g = Graph::new();
    let x = g.input("x", [2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut g,
        Msg::new(&node("Reciprocal", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([("x".into(), TensorData::new([2], vec![2., 0.]).unwrap())]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.scalar_at(0).as_f64(), 0.5);
    assert!(output.scalar_at(1).as_f64().is_infinite());

    let mut attribute = node("Reciprocal", &["x"], "out");
    field(&mut attribute, 5, &int_attr("keepdims", 1));
    for invalid in [node("Reciprocal", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let x = overflow.input("x", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Reciprocal", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn xor_matches_tinygrad_bool_cast_and_preflights_before_publication() {
    let mut g = Graph::new();
    let lhs = g.input("lhs", [2, 1]);
    let rhs = g.input("rhs", [2]);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut g,
        Msg::new(&node("Xor", &["lhs", "rhs"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([
                ("lhs".into(), TensorData::new([2, 1], vec![0., 2.]).unwrap()),
                ("rhs".into(), TensorData::new([2], vec![0., 3.]).unwrap()),
            ]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::Bool);
    assert_eq!(output.shape().dims(), &[2, 2]);
    assert!(!output.scalar_at(0).as_bool());
    assert!(output.scalar_at(1).as_bool());
    assert!(output.scalar_at(2).as_bool());
    assert!(!output.scalar_at(3).as_bool());

    let mut attribute = node("Xor", &["lhs", "rhs"], "out");
    field(&mut attribute, 5, &int_attr("keepdims", 1));
    for invalid in [node("Xor", &["lhs"], "out"), attribute] {
        let mut malformed = Graph::new();
        let lhs = malformed.input("lhs", [2]);
        let rhs = malformed.input("rhs", [2]);
        let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut mismatch = Graph::new();
    let lhs = mismatch.input("lhs", [2]);
    let rhs = mismatch.input("rhs", [3]);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = mismatch.node_count();
    assert!(lower(
        &mut mismatch,
        Msg::new(&node("Xor", &["lhs", "rhs"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(mismatch.node_count(), before_nodes);

    let mut overflow = Graph::new();
    let lhs = overflow.input("lhs", [usize::MAX, 2]);
    let rhs = overflow.input("rhs", [1, 2]);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Xor", &["lhs", "rhs"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn and_matches_tinygrad_value_select_and_preflights_before_publication() {
    let mut g = Graph::new();
    let lhs = g.input("lhs", [2, 1]);
    let rhs = g.input("rhs", [2]);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut g,
        Msg::new(&node("And", &["lhs", "rhs"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([
                ("lhs".into(), TensorData::new([2, 1], vec![1., 2.]).unwrap()),
                ("rhs".into(), TensorData::new([2], vec![1., 3.]).unwrap()),
            ]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.shape().dims(), &[2, 2]);
    assert_eq!(output.values(), &[1., 0., 0., 0.]);

    let mut boolean = Graph::new();
    let lhs = boolean.input_dtype("lhs", [2], DType::Bool);
    let rhs = boolean.input_dtype("rhs", [2], DType::Bool);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut boolean,
        Msg::new(&node("And", &["lhs", "rhs"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &boolean,
            values["out"],
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::Bool,
                        [Scalar::Bool(true), Scalar::Bool(false)],
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::Bool,
                        [Scalar::Bool(true), Scalar::Bool(true)],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::Bool);
    assert!(output.scalar_at(0).as_bool());
    assert!(!output.scalar_at(1).as_bool());

    let mut attribute = node("And", &["lhs", "rhs"], "out");
    field(&mut attribute, 5, &int_attr("keepdims", 1));
    for invalid in [node("And", &["lhs"], "out"), attribute] {
        let mut malformed = Graph::new();
        let lhs = malformed.input("lhs", [2]);
        let rhs = malformed.input("rhs", [2]);
        let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut mismatch = Graph::new();
    let lhs = mismatch.input("lhs", [2]);
    let rhs = mismatch.input("rhs", [3]);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = mismatch.node_count();
    assert!(lower(
        &mut mismatch,
        Msg::new(&node("And", &["lhs", "rhs"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(mismatch.node_count(), before_nodes);

    let mut overflow = Graph::new();
    let lhs = overflow.input("lhs", [usize::MAX, 2]);
    let rhs = overflow.input("rhs", [1, 2]);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("And", &["lhs", "rhs"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn or_matches_tinygrad_value_select_and_preflights_before_publication() {
    let mut g = Graph::new();
    let lhs = g.input("lhs", [2, 1]);
    let rhs = g.input("rhs", [2]);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut g,
        Msg::new(&node("Or", &["lhs", "rhs"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &g,
            values["out"],
            &HashMap::from([
                ("lhs".into(), TensorData::new([2, 1], vec![2., 4.]).unwrap()),
                ("rhs".into(), TensorData::new([2], vec![2., 7.]).unwrap()),
            ]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.shape().dims(), &[2, 2]);
    assert_eq!(output.values(), &[2., 1., 1., 1.]);

    let mut boolean = Graph::new();
    let lhs = boolean.input_dtype("lhs", [2], DType::Bool);
    let rhs = boolean.input_dtype("rhs", [2], DType::Bool);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut boolean,
        Msg::new(&node("Or", &["lhs", "rhs"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &boolean,
            values["out"],
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::Bool,
                        [Scalar::Bool(true), Scalar::Bool(false)],
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::Bool,
                        [Scalar::Bool(true), Scalar::Bool(true)],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::Bool);
    assert!(output.scalar_at(0).as_bool());
    assert!(output.scalar_at(1).as_bool());

    let mut attribute = node("Or", &["lhs", "rhs"], "out");
    field(&mut attribute, 5, &int_attr("keepdims", 1));
    for invalid in [node("Or", &["lhs"], "out"), attribute] {
        let mut malformed = Graph::new();
        let lhs = malformed.input("lhs", [2]);
        let rhs = malformed.input("rhs", [2]);
        let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut mismatch = Graph::new();
    let lhs = mismatch.input("lhs", [2]);
    let rhs = mismatch.input("rhs", [3]);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = mismatch.node_count();
    assert!(lower(
        &mut mismatch,
        Msg::new(&node("Or", &["lhs", "rhs"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(mismatch.node_count(), before_nodes);

    let mut overflow = Graph::new();
    let lhs = overflow.input("lhs", [usize::MAX, 2]);
    let rhs = overflow.input("rhs", [1, 2]);
    let mut values = BTreeMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Or", &["lhs", "rhs"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn identity_aliases_its_input_without_graph_growth_and_rejects_attributes() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 3], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_nodes = graph.node_count();
    lower(
        &mut graph,
        Msg::new(&node("Identity", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    assert_eq!(values["out"], input);
    assert_eq!(graph.dtype(values["out"]).unwrap(), DType::I64);
    assert_eq!(graph.shape(values["out"]).unwrap().dims(), &[2, 3]);
    assert_eq!(graph.node_count(), before_nodes);
    assert!(constants.is_empty());

    let mut attribute = node("Identity", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    for invalid in [node("Identity", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }
}

#[test]
fn cast_like_uses_only_static_target_dtype_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2], DType::F32);
    let target = graph.input_dtype("target", [3], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input), ("target".into(), target)]);
    let mut constants = BTreeMap::new();
    let mut cast_like = node("CastLike", &["input", "target"], "out");
    field(&mut cast_like, 5, &int_attr("saturate", 0));
    lower(
        &mut graph,
        Msg::new(&cast_like),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([
                ("input".into(), TensorData::new([2], vec![1.9, -2.1]).unwrap()),
                (
                    "target".into(),
                    TensorData::from_scalars(
                        [3],
                        DType::I64,
                        [Scalar::I(7), Scalar::I(8), Scalar::I(9)],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::I64);
    assert_eq!(output.scalar_at(0), Scalar::I(1));
    assert_eq!(output.scalar_at(1), Scalar::I(-2));

    let mut same = Graph::new();
    let input = same.input_dtype("input", [2], DType::F32);
    let target = same.input_dtype("target", [], DType::F32);
    let mut values = BTreeMap::from([("input".into(), input), ("target".into(), target)]);
    let mut constants = BTreeMap::new();
    let before_nodes = same.node_count();
    lower(
        &mut same,
        Msg::new(&node("CastLike", &["input", "target"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    assert_eq!(values["out"], input);
    assert_eq!(same.node_count(), before_nodes);
    assert!(constants.is_empty());

    let mut attribute = node("CastLike", &["input", "target"], "out");
    field(&mut attribute, 5, &int_attr("unknown", 1));
    for invalid in [node("CastLike", &["input"], "out"), attribute] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let target = malformed.input_dtype("target", [], DType::I64);
        let mut values = BTreeMap::from([("input".into(), input), ("target".into(), target)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let target = overflow.input_dtype("target", [], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input), ("target".into(), target)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("CastLike", &["input", "target"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn variadic_sum_matches_tinygrad_left_fold_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let first = graph.input_dtype("first", [2, 1], DType::F32);
    let second = graph.input_dtype("second", [2], DType::I64);
    let third = graph.input_dtype("third", [], DType::F32);
    let mut values = BTreeMap::from([
        ("first".into(), first),
        ("second".into(), second),
        ("third".into(), third),
    ]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Sum", &["first", "second", "third"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([
                (
                    "first".into(),
                    TensorData::new([2, 1], vec![1., 2.]).unwrap(),
                ),
                (
                    "second".into(),
                    TensorData::from_scalars([2], DType::I64, [Scalar::I(3), Scalar::I(4)])
                        .unwrap(),
                ),
                ("third".into(), TensorData::from_scalars([], DType::F32, [Scalar::F(0.5)]).unwrap()),
            ]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.shape().dims(), &[2, 2]);
    assert_eq!(output.values(), &[4.5, 5.5, 5.5, 6.5]);

    let mut single = Graph::new();
    let input = single.input("input", [2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_nodes = single.node_count();
    lower(
        &mut single,
        Msg::new(&node("Sum", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    assert_eq!(values["out"], input);
    assert_eq!(single.node_count(), before_nodes);
    assert!(constants.is_empty());

    let mut attribute = node("Sum", &["first"], "out");
    field(&mut attribute, 5, &int_attr("keepdims", 1));
    for invalid in [node("Sum", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let first = malformed.input("first", [2]);
        let mut values = BTreeMap::from([("first".into(), first)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut mismatch = Graph::new();
    let first = mismatch.input("first", [2]);
    let second = mismatch.input("second", [3]);
    let mut values = BTreeMap::from([("first".into(), first), ("second".into(), second)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = mismatch.node_count();
    assert!(lower(
        &mut mismatch,
        Msg::new(&node("Sum", &["first", "second"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(mismatch.node_count(), before_nodes);

    let mut overflow = Graph::new();
    let first = overflow.input("first", [usize::MAX, 2]);
    let second = overflow.input("second", [1, 2]);
    let mut values = BTreeMap::from([("first".into(), first), ("second".into(), second)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Sum", &["first", "second"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn variadic_mean_matches_tinygrad_sum_then_true_division_and_preflights() {
    let mut graph = Graph::new();
    let first = graph.input_dtype("first", [2, 1], DType::F32);
    let second = graph.input_dtype("second", [2], DType::I64);
    let third = graph.input_dtype("third", [], DType::F32);
    let mut values = BTreeMap::from([
        ("first".into(), first),
        ("second".into(), second),
        ("third".into(), third),
    ]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Mean", &["first", "second", "third"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([
                (
                    "first".into(),
                    TensorData::new([2, 1], vec![1., 2.]).unwrap(),
                ),
                (
                    "second".into(),
                    TensorData::from_scalars([2], DType::I64, [Scalar::I(3), Scalar::I(4)])
                        .unwrap(),
                ),
                ("third".into(), TensorData::from_scalars([], DType::F32, [Scalar::F(0.5)]).unwrap()),
            ]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.shape().dims(), &[2, 2]);
    assert_eq!(output.values(), &[1.5, 11. / 6., 11. / 6., 13. / 6.]);

    let mut single = Graph::new();
    let input = single.input_dtype("input", [2], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut single,
        Msg::new(&node("Mean", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &single,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([2], DType::I64, [Scalar::I(2), Scalar::I(4)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values(), &[2., 4.]);

    let mut attribute = node("Mean", &["first"], "out");
    field(&mut attribute, 5, &int_attr("keepdims", 1));
    for invalid in [node("Mean", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let first = malformed.input("first", [2]);
        let mut values = BTreeMap::from([("first".into(), first)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut mismatch = Graph::new();
    let first = mismatch.input("first", [2]);
    let second = mismatch.input("second", [3]);
    let mut values = BTreeMap::from([("first".into(), first), ("second".into(), second)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = mismatch.node_count();
    assert!(lower(
        &mut mismatch,
        Msg::new(&node("Mean", &["first", "second"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(mismatch.node_count(), before_nodes);

    let mut overflow = Graph::new();
    let first = overflow.input("first", [usize::MAX, 2]);
    let second = overflow.input("second", [1, 2]);
    let mut values = BTreeMap::from([("first".into(), first), ("second".into(), second)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Mean", &["first", "second"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn exp_preserves_graph_unary_semantics_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [3]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Exp", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::new([3], vec![0., 1., f32::INFINITY]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values()[0], 1.);
    assert_eq!(output.values()[1], std::f32::consts::E);
    assert!(output.values()[2].is_infinite());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Exp", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &integer,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([1], DType::I64, [Scalar::I(1)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values(), &[std::f32::consts::E]);

    let mut attribute = node("Exp", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    for invalid in [node("Exp", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Exp", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn floor_preserves_graph_unary_semantics_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [4]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Floor", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::new([4], vec![-1.25, -0.0, f32::INFINITY, f32::NEG_INFINITY])
                    .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values()[0], -2.0);
    assert_eq!(output.values()[1].to_bits(), (-0.0f32).to_bits());
    assert!(output.values()[2].is_infinite() && output.values()[2].is_sign_positive());
    assert!(output.values()[3].is_infinite() && output.values()[3].is_sign_negative());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [2], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Floor", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &integer,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([2], DType::I64, [Scalar::I(-3), Scalar::I(4)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::I64);
    assert_eq!(output.scalar_at(0).as_i64(), -3);
    assert_eq!(output.scalar_at(1).as_i64(), 4);

    let mut attribute = node("Floor", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    let mut multiple_outputs = node("Floor", &["input"], "out");
    text(&mut multiple_outputs, 2, "other");
    for invalid in [node("Floor", &[], "out"), multiple_outputs, attribute] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Floor", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn log_preserves_graph_unary_semantics_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [3]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Log", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::new([3], vec![1., 0., -1.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values()[0], 0.);
    assert!(output.values()[1].is_infinite() && output.values()[1].is_sign_negative());
    assert!(output.values()[2].is_nan());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Log", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &integer,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([1], DType::I64, [Scalar::I(1)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values(), &[0.]);

    let mut attribute = node("Log", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    for invalid in [node("Log", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Log", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn sqrt_preserves_graph_unary_semantics_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [4]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Sqrt", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::new([4], vec![4., -0., f32::INFINITY, -1.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values()[0], 2.);
    assert_eq!(output.values()[1], -0.);
    assert!(output.values()[1].is_sign_negative());
    assert!(output.values()[2].is_infinite());
    assert!(output.values()[3].is_nan());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Sqrt", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &integer,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([1], DType::I64, [Scalar::I(4)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values(), &[2.]);

    let mut attribute = node("Sqrt", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    for invalid in [node("Sqrt", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Sqrt", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn sin_matches_tinygrad_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [4]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Sin", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::new(
                    [4],
                    vec![0., std::f32::consts::FRAC_PI_2, -std::f32::consts::FRAC_PI_2, f32::INFINITY],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values()[0], 0.);
    assert!((output.values()[1] - 1.).abs() < 1e-6);
    assert!((output.values()[2] + 1.).abs() < 1e-6);
    assert!(output.values()[3].is_nan());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Sin", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &integer,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([1], DType::I64, [Scalar::I(0)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values(), &[0.]);

    let mut attribute = node("Sin", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    for invalid in [node("Sin", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Sin", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn cos_matches_tinygrad_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [4]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Cos", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::new(
                    [4],
                    vec![0., std::f32::consts::PI, std::f32::consts::FRAC_PI_2, f32::INFINITY],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert!((output.values()[0] - 1.).abs() < 1e-6);
    assert!((output.values()[1] + 1.).abs() < 1e-6);
    assert!(output.values()[2].abs() < 1e-6);
    assert!(output.values()[3].is_nan());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Cos", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &integer,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([1], DType::I64, [Scalar::I(0)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values(), &[1.]);

    let mut attribute = node("Cos", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    for invalid in [node("Cos", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Cos", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn tan_matches_tinygrad_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [3]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Tan", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::new([3], vec![0., std::f32::consts::FRAC_PI_4, f32::INFINITY]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values()[0], 0.);
    assert!((output.values()[1] - 1.).abs() < 1e-6);
    assert!(output.values()[2].is_nan());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Tan", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &integer,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([1], DType::I64, [Scalar::I(0)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values(), &[0.]);

    let mut attribute = node("Tan", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    for invalid in [node("Tan", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Tan", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn asin_matches_tinygrad_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [4]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Asin", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::new([4], vec![0., 1., -1., 2.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values()[0], 0.);
    assert!((output.values()[1] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    assert!((output.values()[2] + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    assert!(output.values()[3].is_nan());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Asin", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &integer,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([1], DType::I64, [Scalar::I(0)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values(), &[0.]);

    let mut attribute = node("Asin", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    for invalid in [node("Asin", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Asin", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn acos_matches_tinygrad_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [3]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Acos", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::new([3], vec![1., -1., 2.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values()[0], 0.);
    assert!((output.values()[1] - std::f32::consts::PI).abs() < 1e-6);
    assert!(output.values()[2].is_nan());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Acos", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &integer,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([1], DType::I64, [Scalar::I(1)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values(), &[0.]);

    let mut attribute = node("Acos", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    for invalid in [node("Acos", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Acos", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn atan_matches_tinygrad_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [3]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Atan", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::new([3], vec![0., 1., f32::INFINITY]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values()[0], 0.);
    assert!((output.values()[1] - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    assert!((output.values()[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Atan", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &integer,
            values["out"],
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([1], DType::I64, [Scalar::I(0)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values(), &[0.]);

    let mut attribute = node("Atan", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    for invalid in [node("Atan", &[], "out"), attribute] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Atan", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}
