use super::schema::{axes_usize, const_i64, reshape_dims};
use super::tensor::tensor_data;
use super::*;
use crate::{DType, Scalar, Shape};
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

#[test]
fn lrn_matches_tinygrad_fixed_channel_divisor_and_preflights() {
    let lrn = |attrs: &[Vec<u8>]| {
        let mut encoded = node("LRN", &["x"], "out");
        for attr in attrs {
            field(&mut encoded, 5, attr);
        }
        encoded
    };
    let mut graph = Graph::new();
    let x = graph.input("x", [1, 3, 1, 1]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    // alpha=1, beta=1, bias=0 exposes the source's fixed size-three
    // border divisor: [1,2,3] / ([1,5,13]/3), not a variable-count mean.
    lower(
        &mut graph,
        Msg::new(&lrn(&[
            typed_int_attr("size", 3),
            float_attr("alpha", 1.0),
            float_attr("beta", 1.0),
            float_attr("bias", 0.0),
        ])),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([1, 3, 1, 1], vec![1., 2., 3.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.shape().dims(), &[1, 3, 1, 1]);
    for (actual, expected) in output.values().iter().zip([0.6, 3. / 7., 9. / 13.]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    // LRN intentionally contains Pad, which remains fail-closed outside CPU.
    assert!(crate::schedule(&graph, values["out"]).is_err());

    for invalid in [
        lrn(&[]),
        lrn(&[float_attr("size", 3.0)]),
        lrn(&[typed_int_attr("size", 0)]),
        lrn(&[typed_int_attr("size", 3), float_attr("unknown", 1.0)]),
        lrn(&[typed_int_attr("size", 3), float_attr("alpha", 1.0), float_attr("alpha", 2.0)]),
    ] {
        let mut malformed = Graph::new();
        let input = malformed.input("x", [1, 3, 1, 1]);
        let mut values = BTreeMap::from([("x".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
        assert_eq!(malformed.node_count(), before_nodes);
        assert_eq!(values["x"], input);
        assert!(!values.contains_key("out"));
    }

    let mut empty = Graph::new();
    let input = empty.input_dtype("x", [1, 0, 2, 2], DType::I32);
    let mut values = BTreeMap::from([("x".into(), input)]);
    lower(
        &mut empty,
        Msg::new(&lrn(&[typed_int_attr("size", 3)])),
        &mut values,
        &mut BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(empty.dtype(values["out"]).unwrap(), DType::F32);
}

#[test]
fn gelu_uses_closed_typed_modes_and_preflights() {
    let gelu = |attrs: &[Vec<u8>]| {
        let mut encoded = node("Gelu", &["x"], "out");
        for attr in attrs { field(&mut encoded, 5, attr); }
        encoded
    };
    for attrs in [Vec::new(), vec![typed_string_attr("approximate", "none")], vec![typed_string_attr("approximate", "tanh")]] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2], DType::F16);
        let mut values = BTreeMap::from([("x".into(), input)]);
        lower(&mut graph, Msg::new(&gelu(&attrs)), &mut values, &mut BTreeMap::new()).unwrap();
        assert_eq!(graph.dtype(values["out"]).unwrap(), DType::F16);
        assert_eq!(graph.shape(values["out"]).unwrap().dims(), &[2]);
    }
    for invalid in [
        gelu(&[typed_string_attr("approximate", "fast")]),
        gelu(&[int_attr("approximate", 1)]),
        gelu(&[typed_string_attr("approximate", "none"), typed_string_attr("approximate", "tanh")]),
        gelu(&[typed_string_attr("other", "none")]),
        node("Gelu", &[], "out"),
    ] {
        let mut graph = Graph::new();
        let input = graph.input("x", [2]);
        let mut values = BTreeMap::from([("x".into(), input)]);
        let before = graph.node_count();
        assert!(lower(&mut graph, Msg::new(&invalid), &mut values, &mut BTreeMap::new()).is_err());
        assert_eq!(graph.node_count(), before);
        assert!(!values.contains_key("out"));
    }
    let mut empty = Graph::new();
    let input = empty.input_dtype("x", [0], DType::I32);
    let mut values = BTreeMap::from([("x".into(), input)]);
    lower(&mut empty, Msg::new(&gelu(&[])), &mut values, &mut BTreeMap::new()).unwrap();
    assert_eq!(empty.dtype(values["out"]).unwrap(), DType::F32);
}

#[test]
fn elu_uses_strict_source_branches_and_preflights() {
    let elu = |attrs: &[Vec<u8>]| {
        let mut encoded = node("Elu", &["x"], "out");
        for attr in attrs { field(&mut encoded, 5, attr); }
        encoded
    };
    let mut graph = Graph::new();
    let input = graph.input("x", [3]);
    let mut values = BTreeMap::from([("x".into(), input)]);
    lower(&mut graph, Msg::new(&elu(&[float_attr("alpha", f32::NAN)])), &mut values, &mut BTreeMap::new()).unwrap();
    assert_eq!(graph.dtype(values["out"]).unwrap(), DType::F32);
    for invalid in [elu(&[int_attr("alpha", 1)]), elu(&[float_attr("other", 1.0)]), elu(&[float_attr("alpha", 1.0), float_attr("alpha", 2.0)]), node("Elu", &[], "out")] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [1]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let before = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut BTreeMap::new()).is_err());
        assert_eq!(malformed.node_count(), before);
        assert!(!values.contains_key("out"));
    }
    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [0], DType::I32);
    let mut values = BTreeMap::from([("x".into(), x)]);
    lower(&mut empty, Msg::new(&elu(&[])), &mut values, &mut BTreeMap::new()).unwrap();
    assert_eq!(empty.dtype(values["out"]).unwrap(), DType::F32);
}

#[test]
fn selu_uses_closed_typed_attributes_and_empty_promotion() {
    let selu = |attrs: &[Vec<u8>]| {
        let mut encoded = node("Selu", &["x"], "out");
        for attr in attrs { field(&mut encoded, 5, attr); }
        encoded
    };
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [1], DType::BF16);
    let mut values = BTreeMap::from([("x".into(), x)]);
    lower(&mut graph, Msg::new(&selu(&[float_attr("alpha", f32::NAN), float_attr("gamma", f32::INFINITY)])), &mut values, &mut BTreeMap::new()).unwrap();
    assert_eq!(graph.dtype(values["out"]).unwrap(), DType::BF16);
    for invalid in [selu(&[int_attr("alpha", 1)]), selu(&[float_attr("other", 1.0)]), selu(&[float_attr("gamma", 1.0), float_attr("gamma", 2.0)]), node("Selu", &[], "out")] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [1]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let before = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut BTreeMap::new()).is_err());
        assert_eq!(malformed.node_count(), before);
        assert!(!values.contains_key("out"));
    }
}

#[test]
fn swish_uses_typed_exp2_reciprocal_path_and_preflights() {
    let swish = |attrs: &[Vec<u8>]| {
        let mut encoded = node("Swish", &["x"], "out");
        for attr in attrs { field(&mut encoded, 5, attr); }
        encoded
    };
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [1], DType::F16);
    let mut values = BTreeMap::from([("x".into(), x)]);
    lower(&mut graph, Msg::new(&swish(&[float_attr("alpha", f32::INFINITY)])), &mut values, &mut BTreeMap::new()).unwrap();
    assert_eq!(graph.dtype(values["out"]).unwrap(), DType::F16);
    for invalid in [swish(&[int_attr("alpha", 1)]), swish(&[float_attr("other", 1.0)]), swish(&[float_attr("alpha", 1.0), float_attr("alpha", 2.0)]), node("Swish", &[], "out")] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [1]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let before = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut BTreeMap::new()).is_err());
        assert_eq!(malformed.node_count(), before);
        assert!(!values.contains_key("out"));
    }
}

#[test]
fn mod_uses_typed_fmod_selector_and_constant_zero_preflight() {
    let modu = |attrs: &[Vec<u8>]| {
        let mut encoded = node("Mod", &["x", "y"], "out");
        for attr in attrs { field(&mut encoded, 5, attr); }
        encoded
    };
    for attrs in [Vec::new(), vec![typed_int_attr("fmod", 0)], vec![typed_int_attr("fmod", -1)]] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2, 1], DType::I32);
        let y = graph.input_dtype("y", [1, 2], DType::I32);
        let mut values = BTreeMap::from([("x".into(), x), ("y".into(), y)]);
        lower(&mut graph, Msg::new(&modu(&attrs)), &mut values, &mut BTreeMap::new()).unwrap();
        assert_eq!(graph.shape(values["out"]).unwrap().dims(), &[2, 2]);
    }
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [1], DType::I32);
    let y = graph.input_dtype("y", [1], DType::I32);
    let mut values = BTreeMap::from([("x".into(), x), ("y".into(), y)]);
    let mut constants = BTreeMap::from([("y".into(), TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap())]);
    let before = graph.node_count();
    assert!(lower(&mut graph, Msg::new(&modu(&[])), &mut values, &mut constants).is_err());
    assert_eq!(graph.node_count(), before);
    assert!(!values.contains_key("out"));
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
fn typed_int_attr(name: &str, value: i64) -> Vec<u8> {
    let mut a = vec![];
    text(&mut a, 1, name);
    vi(3 << 3, &mut a);
    vi64(value as u64, &mut a);
    var(&mut a, 20, 2);
    a
}
fn typed_ints_attr(name: &str, values: &[i64]) -> Vec<u8> {
    let mut a = vec![];
    text(&mut a, 1, name);
    let mut packed = vec![];
    for &value in values {
        vi64(value as u64, &mut packed);
    }
    field(&mut a, 8, &packed);
    var(&mut a, 20, 7);
    a
}
fn float_attr(name: &str, value: f32) -> Vec<u8> {
    let mut a = vec![];
    text(&mut a, 1, name);
    vi(2 << 3 | 5, &mut a);
    a.extend_from_slice(&value.to_le_bytes());
    var(&mut a, 20, 1);
    a
}
fn string_attr(name: &str, value: &str) -> Vec<u8> {
    let mut a = vec![];
    text(&mut a, 1, name);
    text(&mut a, 4, value);
    a
}
fn typed_string_attr(name: &str, value: &str) -> Vec<u8> {
    let mut a = string_attr(name, value);
    var(&mut a, 20, 3);
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
fn cumsum_node(attrs: &[Vec<u8>]) -> Vec<u8> {
    let mut x = node("CumSum", &["x", "axis"], "out");
    for attr in attrs {
        field(&mut x, 5, attr);
    }
    x
}
fn lower_cumsum(
    graph: &mut Graph,
    x: NodeId,
    axis: TensorData,
    attrs: &[Vec<u8>],
) -> NodeId {
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axis".into(), axis)]);
    lower(
        graph,
        Msg::new(&cumsum_node(attrs)),
        &mut values,
        &mut constants,
    )
    .unwrap();
    values["out"]
}
fn trilu_node(k: Option<&str>, attrs: &[Vec<u8>]) -> Vec<u8> {
    let inputs = k.map_or_else(|| vec!["x"], |k| vec!["x", k]);
    let mut x = node("Trilu", &inputs, "out");
    for attr in attrs {
        field(&mut x, 5, attr);
    }
    x
}
fn lower_trilu(
    graph: &mut Graph,
    x: NodeId,
    k: Option<TensorData>,
    attrs: &[Vec<u8>],
) -> NodeId {
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    if let Some(k) = k {
        constants.insert("k".into(), k);
    }
    let node = trilu_node(constants.contains_key("k").then_some("k"), attrs);
    lower(&mut *graph, Msg::new(&node), &mut values, &mut constants).unwrap();
    values["out"]
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
fn global_max_pool_matches_tinygrad_trailing_max_and_empty_identities() {
    // tinygrad uses max over range(2, rank), so scalar, vector, and matrix
    // inputs have an empty axis tuple and retain the exact input NodeId.
    for shape in [Shape::new(vec![]), Shape::new(vec![2]), Shape::new(vec![1, 2])] {
        let mut graph = Graph::new();
        let x = graph.input("x", shape.clone());
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        let before_nodes = graph.node_count();
        lower(
            &mut graph,
            Msg::new(&node("GlobalMaxPool", &["x"], "out")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        assert_eq!(values["out"], x);
        assert_eq!(graph.node_count(), before_nodes);
        assert!(constants.is_empty());
    }

    // Every trailing spatial axis is reduced and kept, not just the 2-D pool
    // surface handled by MaxPool.
    for (shape, data, expected_shape, expected) in [
        (
            Shape::new(vec![1, 2, 2]),
            vec![1., 2., 3., 4.],
            Shape::new(vec![1, 2, 1]),
            vec![2., 4.],
        ),
        (
            Shape::new(vec![1, 1, 2, 2]),
            vec![1., 2., 3., 4.],
            Shape::new(vec![1, 1, 1, 1]),
            vec![4.],
        ),
        (
            Shape::new(vec![1, 1, 2, 1, 2]),
            vec![1., 2., 3., 4.],
            Shape::new(vec![1, 1, 1, 1, 1]),
            vec![4.],
        ),
    ] {
        let mut graph = Graph::new();
        let x = graph.input("x", shape.clone());
        let mut values = BTreeMap::from([("x".into(), x)]);
        lower(
            &mut graph,
            Msg::new(&node("GlobalMaxPool", &["x"], "out")),
            &mut values,
            &mut BTreeMap::new(),
        )
        .unwrap();
        let output = CpuBackend
            .execute(
                &graph,
                values["out"],
                &HashMap::from([("x".into(), TensorData::new(shape, data).unwrap())]),
        )
        .unwrap();
        assert_eq!(output.shape(), &expected_shape);
        assert_eq!(output.values(), expected.as_slice());
    }

    // Existing extrema semantics are source-aligned: NaNs are ignored, strict
    // ties retain the first non-NaN lane (including signed zero), and infinities
    // participate normally.
    let mut graph = Graph::new();
    let x = graph.input("x", [1, 1, 1, 3]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    lower(
        &mut graph,
        Msg::new(&node("GlobalMaxPool", &["x"], "out")),
        &mut values,
        &mut BTreeMap::new(),
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([1, 1, 1, 3], vec![f32::NAN, -0., 0.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.values()[0].to_bits(), (-0.0f32).to_bits());

    let mut graph = Graph::new();
    let x = graph.input("x", [1, 1, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    lower(
        &mut graph,
        Msg::new(&node("GlobalMaxPool", &["x"], "out")),
        &mut values,
        &mut BTreeMap::new(),
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([1, 1, 2], vec![f32::NAN, f32::NAN]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.values(), &[f32::NEG_INFINITY]);

    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([1, 1, 2], vec![f32::NEG_INFINITY, f32::INFINITY]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.values(), &[f32::INFINITY]);

    // A zero spatial domain is a tinygrad MAX identity at the source dtype,
    // while zero batch/channel extent remains an unpopulated empty result.
    for (dtype, identity) in [
        (DType::Bool, Scalar::Bool(false)),
        (DType::I8, Scalar::I(i8::MIN.into())),
        (DType::U8, Scalar::U(0)),
        (DType::I16, Scalar::I(i16::MIN.into())),
        (DType::U16, Scalar::U(0)),
        (DType::I32, Scalar::I(i32::MIN.into())),
        (DType::U32, Scalar::U(0)),
        (DType::I64, Scalar::I(i64::MIN)),
        (DType::U64, Scalar::U(0)),
        (DType::F16, Scalar::F(f64::NEG_INFINITY)),
        (DType::BF16, Scalar::F(f64::NEG_INFINITY)),
        (DType::F32, Scalar::F(f64::NEG_INFINITY)),
        (DType::F64, Scalar::F(f64::NEG_INFINITY)),
    ] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [1, 1, 0], dtype);
        let mut values = BTreeMap::from([("x".into(), x)]);
        lower(
            &mut graph,
            Msg::new(&node("GlobalMaxPool", &["x"], "out")),
            &mut values,
            &mut BTreeMap::new(),
        )
        .unwrap();
        let output = CpuBackend
            .execute(
                &graph,
                values["out"],
                &HashMap::from([(
                    "x".into(),
                    TensorData::from_scalars([1, 1, 0], dtype, []).unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(
            output,
            TensorData::from_scalars([1, 1, 1], dtype, [identity]).unwrap(),
            "{dtype:?}"
        );
    }

    let mut graph = Graph::new();
    let x = graph.input("x", [0, 1, 0]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    lower(
        &mut graph,
        Msg::new(&node("GlobalMaxPool", &["x"], "out")),
        &mut values,
        &mut BTreeMap::new(),
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([("x".into(), TensorData::new([0, 1, 0], vec![]).unwrap())]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[0, 1, 1]);
    assert!(output.values().is_empty());

    let mut unknown = node("GlobalMaxPool", &["x"], "out");
    field(&mut unknown, 5, &int_attr("axis", 0));
    for invalid in [
        node("GlobalMaxPool", &[], "out"),
        node("GlobalMaxPool", &["x", "extra"], "out"),
        node("GlobalMaxPool", &["missing"], "out"),
        unknown,
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [1, 1, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let x = overflow.input("x", [usize::MAX, 2, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("GlobalMaxPool", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn cumsum_matches_tinygrad_static_axis_flags_and_fail_closed_pad_boundary() {
    // Both permitted constant representations resolve to the same signed
    // axis.  Tinygrad treats every nonzero flag value as true.
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 3]);
    let forward = lower_cumsum(
        &mut graph,
        x,
        TensorData::from_scalars([], DType::I32, [Scalar::I(-1)]).unwrap(),
        &[],
    );
    let leading_axis = lower_cumsum(
        &mut graph,
        x,
        TensorData::from_scalars([1], DType::I64, [Scalar::I(-2)]).unwrap(),
        &[],
    );
    let reverse_exclusive = lower_cumsum(
        &mut graph,
        x,
        TensorData::from_scalars([1], DType::I64, [Scalar::I(-1)]).unwrap(),
        &[int64_attr("exclusive", -7), int64_attr("reverse", 9)],
    );
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
    )]);
    assert_eq!(
        CpuBackend.execute(&graph, forward, &bindings).unwrap().values(),
        &[1., 3., 6., 4., 9., 15.]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, leading_axis, &bindings)
            .unwrap()
            .values(),
        &[1., 2., 3., 5., 7., 9.]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, reverse_exclusive, &bindings)
            .unwrap()
            .values(),
        &[5., 3., 0., 11., 6., 0.]
    );
    // The literal source-exclusive form has an Op::Pad.  It is intentionally
    // outside the generic schedule inventory, so native work cannot begin.
    assert!(crate::schedule(&graph, reverse_exclusive).is_err());

    // Sum defaults remain the public cumsum contract, including widened
    // small integers, narrow float accumulation with final narrowing, and
    // wrapping retained-width U64 output.
    for (dtype, values, expected_dtype, expected) in [
        (
            DType::Bool,
            vec![Scalar::Bool(true), Scalar::Bool(true)],
            DType::I32,
            vec![Scalar::I(1), Scalar::I(2)],
        ),
        (
            DType::I8,
            vec![Scalar::I(120), Scalar::I(120)],
            DType::I32,
            vec![Scalar::I(120), Scalar::I(240)],
        ),
        (
            DType::U16,
            vec![Scalar::U(2), Scalar::U(3)],
            DType::U32,
            vec![Scalar::U(2), Scalar::U(5)],
        ),
        (
            DType::F16,
            vec![Scalar::F(0.5), Scalar::F(0.25)],
            DType::F16,
            vec![Scalar::F(0.5), Scalar::F(0.75)],
        ),
        (
            DType::BF16,
            vec![Scalar::F(0.5), Scalar::F(0.25)],
            DType::BF16,
            vec![Scalar::F(0.5), Scalar::F(0.75)],
        ),
        (
            DType::U64,
            vec![Scalar::U(u64::MAX), Scalar::U(1)],
            DType::U64,
            vec![Scalar::U(u64::MAX), Scalar::U(0)],
        ),
        (
            DType::F64,
            vec![Scalar::F(1.0), Scalar::F(2.0)],
            DType::F64,
            vec![Scalar::F(1.0), Scalar::F(3.0)],
        ),
    ] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2], dtype);
        let output = lower_cumsum(
            &mut graph,
            x,
            TensorData::from_scalars([], DType::I64, [Scalar::I(0)]).unwrap(),
            &[],
        );
        assert_eq!(graph.dtype(output).unwrap(), expected_dtype, "{dtype:?}");
        assert_eq!(
            CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([(
                        "x".into(),
                        TensorData::from_scalars([2], dtype, values).unwrap(),
                    )]),
                )
                .unwrap(),
            TensorData::from_scalars([2], expected_dtype, expected).unwrap(),
            "{dtype:?}"
        );
    }

    let mut specials = Graph::new();
    let x = specials.input("x", [4]);
    let output = lower_cumsum(
        &mut specials,
        x,
        TensorData::from_scalars([], DType::I64, [Scalar::I(0)]).unwrap(),
        &[],
    );
    let special = CpuBackend
        .execute(
            &specials,
            output,
            &HashMap::from([(
                "x".into(),
                TensorData::new([4], vec![-0.0, 0.0, f32::NAN, f32::INFINITY]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(special.values()[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(special.values()[1].to_bits(), 0.0f32.to_bits());
    assert!(special.values()[2].is_nan());
    assert!(special.values()[3].is_nan());

    // Inclusive scalar cumsum uses the Graph's scalar typed-Sum path; source
    // reverse/exclusive scalar requests fail before any Graph mutation.
    let mut scalar = Graph::new();
    let x = scalar.input_dtype("x", [], DType::I8);
    let output = lower_cumsum(
        &mut scalar,
        x,
        TensorData::from_scalars([], DType::I64, [Scalar::I(-1)]).unwrap(),
        &[],
    );
    assert_eq!(scalar.dtype(output).unwrap(), DType::I32);
    assert_eq!(
        CpuBackend
            .execute(
                &scalar,
                output,
                &HashMap::from([(
                    "x".into(),
                    TensorData::from_scalars([], DType::I8, [Scalar::I(5)]).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        vec![5.]
    );

    for attrs in [vec![int_attr("reverse", 1)], vec![int_attr("exclusive", 1)]] {
        let mut graph = Graph::new();
        let x = graph.input("x", []);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::from([(
            "axis".into(),
            TensorData::from_scalars([], DType::I64, [Scalar::I(0)]).unwrap(),
        )]);
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = graph.node_count();
        assert!(lower(
            &mut graph,
            Msg::new(&cumsum_node(&attrs)),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(graph.node_count(), before_nodes);
    }

    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [1, 0], DType::F16);
    let output = lower_cumsum(
        &mut empty,
        x,
        TensorData::from_scalars([1], DType::I64, [Scalar::I(-1)]).unwrap(),
        &[int_attr("exclusive", 1)],
    );
    let data = CpuBackend
        .execute(
            &empty,
            output,
            &HashMap::from([(
                "x".into(),
                TensorData::from_scalars([1, 0], DType::F16, []).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(data.dtype(), DType::F16);
    assert_eq!(data.shape().dims(), &[1, 0]);
    assert!(data.is_empty());

    for axis in [
        TensorData::from_scalars([], DType::F32, [Scalar::F(0.0)]).unwrap(),
        TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(1)]).unwrap(),
        TensorData::from_scalars([1, 1], DType::I64, [Scalar::I(0)]).unwrap(),
        TensorData::from_scalars([], DType::I64, [Scalar::I(i64::MAX)]).unwrap(),
        TensorData::from_scalars([], DType::I64, [Scalar::I(i64::MIN)]).unwrap(),
    ] {
        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::from([("axis".into(), axis)]);
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = graph.node_count();
        assert!(lower(
            &mut graph,
            Msg::new(&cumsum_node(&[])),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(graph.node_count(), before_nodes);
    }

    for invalid in [
        node("CumSum", &["x"], "out"),
        node("CumSum", &["x", "axis", "extra"], "out"),
        node("CumSum", &["missing", "axis"], "out"),
        cumsum_node(&[int_attr("unknown", 1)]),
        cumsum_node(&[int_attr("exclusive", 1), int_attr("exclusive", 1)]),
        cumsum_node(&[string_attr("reverse", "yes")]),
    ] {
        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::from([(
            "axis".into(),
            TensorData::from_scalars([], DType::I64, [Scalar::I(0)]).unwrap(),
        )]);
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = graph.node_count();
        assert!(lower(&mut graph, Msg::new(&invalid), &mut values, &mut constants).is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(graph.node_count(), before_nodes);
    }

    let mut overflow = Graph::new();
    let x = overflow.input("x", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([(
        "axis".into(),
        TensorData::from_scalars([], DType::I64, [Scalar::I(0)]).unwrap(),
    )]);
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&cumsum_node(&[])),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn trilu_matches_tinygrad_masks_and_saturates_extreme_diagonals() {
    // Default k=0/upper=1, an I32 scalar k, and truthy upper=0 lower mode.
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 3]);
    let upper = lower_trilu(&mut graph, x, None, &[]);
    let lower = lower_trilu(
        &mut graph,
        x,
        Some(TensorData::from_scalars([], DType::I32, [Scalar::I(0)]).unwrap()),
        &[int64_attr("upper", 0)],
    );
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
    )]);
    assert_eq!(
        CpuBackend.execute(&graph, upper, &bindings).unwrap().values(),
        &[1., 2., 3., 0., 5., 6.]
    );
    assert_eq!(
        CpuBackend.execute(&graph, lower, &bindings).unwrap().values(),
        &[1., 0., 0., 4., 5., 0.]
    );

    // The final two dimensions are the only matrix axes; leading dimensions
    // broadcast the same mask.
    let mut batched = Graph::new();
    let x = batched.input("x", [2, 2, 2]);
    let output = lower_trilu(
        &mut batched,
        x,
        Some(TensorData::from_scalars([1], DType::I64, [Scalar::I(1)]).unwrap()),
        &[int64_attr("upper", -3)],
    );
    assert_eq!(
        CpuBackend
            .execute(
                &batched,
                output,
                &HashMap::from([(
                    "x".into(),
                    TensorData::new([2, 2, 2], vec![1., 2., 3., 4., 5., 6., 7., 8.])
                        .unwrap(),
                )]),
            )
            .unwrap()
            .values(),
        &[0., 2., 0., 0., 0., 6., 0., 0.]
    );

    // Source saturation is observable at the diagonal boundaries, including
    // I64 extremes which the generic helpers deliberately reject internally.
    for (upper, diagonal, identity) in [
        (true, -1, true),
        (true, 3, false),
        (true, i64::MIN, true),
        (true, i64::MAX, false),
        (false, -2, false),
        (false, 2, true),
        (false, i64::MIN, false),
        (false, i64::MAX, true),
    ] {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 3]);
        let before = graph.node_count();
        let output = lower_trilu(
            &mut graph,
            x,
            Some(TensorData::from_scalars([1], DType::I64, [Scalar::I(diagonal)]).unwrap()),
            &[int64_attr("upper", upper as i64)],
        );
        let data = CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([(
                    "x".into(),
                    TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
                )]),
            )
            .unwrap();
        if identity {
            assert_eq!(output, x);
            assert_eq!(graph.node_count(), before);
            assert_eq!(data.values(), &[1., 2., 3., 4., 5., 6.]);
        } else {
            assert_eq!(data.values(), &[0.; 6]);
        }
    }

    // Saturated zero uses the source dtype for every supported storage class.
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [1, 1], dtype);
        let output = lower_trilu(
            &mut graph,
            x,
            Some(TensorData::from_scalars([], DType::I64, [Scalar::I(i64::MAX)]).unwrap()),
            &[],
        );
        let data = TensorData::from_scalars([1, 1], dtype, [Scalar::I(1)]).unwrap();
        let output_data = CpuBackend
            .execute(&graph, output, &HashMap::from([("x".into(), data)]))
            .unwrap();
        assert_eq!(output_data.dtype(), dtype, "{dtype:?}");
        assert_eq!(output_data.to_vec_f64(), vec![0.], "{dtype:?}");
    }

    let mut special = Graph::new();
    let x = special.input("x", [2, 2]);
    let output = lower_trilu(&mut special, x, None, &[]);
    let data = CpuBackend
        .execute(
            &special,
            output,
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 2], vec![-0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY])
                    .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(data.values()[0].to_bits(), (-0.0f32).to_bits());
    assert!(data.values()[1].is_nan());
    assert_eq!(data.values()[2].to_bits(), 0.0f32.to_bits());
    assert_eq!(data.values()[3], f32::NEG_INFINITY);

    for shape in [Shape::new(vec![0, 2, 2]), Shape::new(vec![1, 0, 3])] {
        let mut graph = Graph::new();
        let x = graph.input("x", shape);
        let before = graph.node_count();
        let output = lower_trilu(&mut graph, x, None, &[]);
        assert_eq!(output, x);
        assert_eq!(graph.node_count(), before);
    }

    for k in [
        TensorData::from_scalars([], DType::F32, [Scalar::F(0.0)]).unwrap(),
        TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(1)]).unwrap(),
        TensorData::from_scalars([1, 1], DType::I64, [Scalar::I(0)]).unwrap(),
    ] {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::from([("k".into(), k)]);
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = graph.node_count();
        assert!(lower(
            &mut graph,
            Msg::new(&trilu_node(Some("k"), &[])),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(graph.node_count(), before_nodes);
    }

    for invalid in [
        node("Trilu", &[], "out"),
        node("Trilu", &["x", "k", "extra"], "out"),
        node("Trilu", &["missing"], "out"),
        trilu_node(None, &[int_attr("unknown", 1)]),
        trilu_node(None, &[string_attr("upper", "yes")]),
    ] {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = graph.node_count();
        assert!(lower(&mut graph, Msg::new(&invalid), &mut values, &mut constants).is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(graph.node_count(), before_nodes);
    }

    let mut rank = Graph::new();
    let x = rank.input("x", [2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    let before_nodes = rank.node_count();
    assert!(lower(
        &mut rank,
        Msg::new(&trilu_node(None, &[])),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(rank.node_count(), before_nodes);

    let mut overflow = Graph::new();
    let x = overflow.input("x", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&trilu_node(None, &[])),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn one_hot_matches_tinygrad_live_values_axis_and_negative_index_contract() {
    let one_hot = |attrs: &[Vec<u8>]| {
        let mut encoded = node("OneHot", &["indices", "depth", "values"], "out");
        for attr in attrs {
            field(&mut encoded, 5, attr);
        }
        encoded
    };

    // Only depth is static. A fractional finite depth follows Python's int
    // conversion; live [off, on, ..] values retain their trailing broadcast.
    let mut graph = Graph::new();
    let indices = graph.input_dtype("indices", [4], DType::F32);
    let values_input = graph.input("values", [2, 1]);
    let mut values = BTreeMap::from([
        ("indices".into(), indices),
        ("values".into(), values_input),
    ]);
    let mut constants = BTreeMap::from([("depth".into(), TensorData::scalar(3.7))]);
    lower(
        &mut graph,
        Msg::new(&one_hot(&[])),
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
                    "indices".into(),
                    TensorData::new([4], vec![-1., -4., 3., 1.]).unwrap(),
                ),
                (
                    "values".into(),
                    TensorData::new([2, 1], vec![-0.0, f32::NAN]).unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[4, 3]);
    assert_eq!(output.dtype(), DType::F32);
    assert!(output.values()[2].is_nan()); // -1 adjusts once to class 2.
    assert!(output.values()[10].is_nan()); // class 1 of the final input.
    assert!(output.values()[3..6]
        .iter()
        .all(|value| value.to_bits() == (-0.0f32).to_bits()));
    assert!(output.values()[6..9]
        .iter()
        .all(|value| value.to_bits() == (-0.0f32).to_bits()));

    // Both permitted empty-depth source forms create a zero class axis.
    for depth in [-1, 0] {
        let mut empty = Graph::new();
        let indices = empty.input_dtype("indices", [0], DType::I64);
        let values_input = empty.input("values", [2]);
        let mut bindings = BTreeMap::from([
            ("indices".into(), indices),
            ("values".into(), values_input),
        ]);
        let mut constants = BTreeMap::from([(
            "depth".into(),
            TensorData::scalar_with_dtype(Scalar::I(depth), DType::I64),
        )]);
        lower(
            &mut empty,
            Msg::new(&one_hot(&[])),
            &mut bindings,
            &mut constants,
        )
        .unwrap();
        let output = CpuBackend
            .execute(
                &empty,
                bindings["out"],
                &HashMap::from([
                    (
                        "indices".into(),
                        TensorData::from_scalars([0], DType::I64, []).unwrap(),
                    ),
                    (
                        "values".into(),
                        TensorData::new([2], vec![1., 2.]).unwrap(),
                    ),
                ]),
            )
            .unwrap();
        assert_eq!(output.shape().dims(), &[0, 0]);
        assert!(output.values().is_empty());
    }

    // Axis bounds are rank + 1 bounds, and the cast/select construction
    // admits every locally supported index and value storage dtype.
    for axis in [-2, 1] {
        let mut graph = Graph::new();
        let indices = graph.input_dtype("indices", [2], DType::U64);
        let values_input = graph.input_dtype("values", [2], DType::BF16);
        let mut bindings = BTreeMap::from([
            ("indices".into(), indices),
            ("values".into(), values_input),
        ]);
        let mut constants = BTreeMap::from([(
            "depth".into(),
            TensorData::scalar_with_dtype(Scalar::I(2), DType::I32),
        )]);
        lower(
            &mut graph,
            Msg::new(&one_hot(&[int64_attr("axis", axis)])),
            &mut bindings,
            &mut constants,
        )
        .unwrap();
        assert_eq!(graph.shape(bindings["out"]).unwrap().dims(), &[2, 2]);
        assert_eq!(graph.dtype(bindings["out"]).unwrap(), DType::BF16);
    }

    for dtype in [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        let mut graph = Graph::new();
        let indices = graph.input_dtype("indices", [], dtype);
        let values_input = graph.input_dtype("values", [2], dtype);
        let mut bindings = BTreeMap::from([
            ("indices".into(), indices),
            ("values".into(), values_input),
        ]);
        let mut constants = BTreeMap::from([(
            "depth".into(),
            TensorData::scalar_with_dtype(Scalar::I(1), DType::I32),
        )]);
        lower(
            &mut graph,
            Msg::new(&one_hot(&[])),
            &mut bindings,
            &mut constants,
        )
        .unwrap();
        assert_eq!(graph.dtype(bindings["out"]).unwrap(), dtype);
    }

    // Every descriptor/static failure is rejected before output publication.
    for (invalid, depth, values_shape) in [
        (node("OneHot", &["indices", "depth"], "out"), TensorData::scalar(2.), vec![2]),
        (one_hot(&[int_attr("unknown", 1)]), TensorData::scalar(2.), vec![2]),
        (one_hot(&[]), TensorData::scalar(f32::NAN), vec![2]),
        (one_hot(&[]), TensorData::scalar(f32::INFINITY), vec![2]),
        (one_hot(&[]), TensorData::scalar(-2.), vec![2]),
        (
            one_hot(&[]),
            TensorData::scalar_with_dtype(Scalar::U(u64::MAX), DType::U64),
            vec![2],
        ),
        (one_hot(&[]), TensorData::scalar(2.), vec![]),
        (one_hot(&[]), TensorData::scalar(2.), vec![1]),
        (one_hot(&[]), TensorData::scalar(2.), vec![2, 3]),
    ] {
        let mut malformed = Graph::new();
        let indices = malformed.input("indices", [2]);
        let values_input = malformed.input("values", values_shape);
        let mut bindings = BTreeMap::from([
            ("indices".into(), indices),
            ("values".into(), values_input),
        ]);
        let mut constants = BTreeMap::from([("depth".into(), depth)]);
        let before_values = bindings.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut bindings,
            &mut constants,
        )
        .is_err());
        assert_eq!(bindings, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }
}

#[test]
fn eye_like_matches_tinygrad_rank_two_padding_and_preflights() {
    let eye_like = |attrs: &[Vec<u8>]| {
        let mut encoded = node("EyeLike", &["x"], "out");
        for attr in attrs {
            field(&mut encoded, 5, attr);
        }
        encoded
    };
    let run = |shape: [usize; 2], attrs: &[Vec<u8>]| {
        let mut graph = Graph::new();
        let x = graph.input("x", shape);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        lower(&mut graph, Msg::new(&eye_like(attrs)), &mut values, &mut constants).unwrap();
        let output = CpuBackend
            .execute(
                &graph,
                values["out"],
                &HashMap::from([(
                    "x".into(),
                    TensorData::new(shape, (0..shape[0] * shape[1]).map(|_| 9.).collect()).unwrap(),
                )]),
            )
            .unwrap();
        (graph, values, output)
    };

    let (_, _, square) = run([2, 2], &[typed_int_attr("k", i64::MAX)]);
    assert_eq!(square.dtype(), DType::F32);
    assert_eq!(square.values(), &[1., 0., 0., 1.]);

    let (_, _, wide) = run([2, 4], &[typed_int_attr("k", 1)]);
    assert_eq!(wide.values(), &[0., 1., 0., 0., 0., 0., 1., 0.]);
    let (_, _, wide_negative) = run([2, 4], &[typed_int_attr("k", -1)]);
    assert_eq!(wide_negative.values(), &[0., 0., 0., 0., 1., 0., 0., 0.]);
    let (_, _, tall) = run([4, 2], &[typed_int_attr("k", 1)]);
    assert_eq!(tall.values(), &[0., 0., 1., 0., 0., 1., 0., 0.]);
    let (_, _, just_inside) = run([2, 4], &[typed_int_attr("k", 3)]);
    assert_eq!(just_inside.values(), &[0., 0., 0., 1., 0., 0., 0., 0.]);
    for k in [-2, 4] {
        let (_, _, endpoint) = run([2, 4], &[typed_int_attr("k", k)]);
        assert_eq!(endpoint.values(), &[0.; 8]);
    }

    for shape in [[0, 3], [3, 0]] {
        let (_, _, empty) = run(shape, &[typed_int_attr("k", 0)]);
        assert_eq!(empty.shape().dims(), &shape);
        assert!(empty.is_empty());
    }

    for (code, dtype) in [
        (1, DType::F32), (2, DType::U8), (3, DType::I8), (4, DType::U16),
        (5, DType::I16), (6, DType::I32), (7, DType::I64), (9, DType::Bool),
        (10, DType::F16), (11, DType::F64), (12, DType::U32), (13, DType::U64),
        (16, DType::BF16),
    ] {
        let (_, _, output) = run([1, 1], &[typed_int_attr("dtype", code)]);
        assert_eq!(output.dtype(), dtype);
        assert_eq!(output.scalar_at(0).as_f64(), 1.);
    }

    let mut default_dtype = Graph::new();
    let x = default_dtype.input_dtype("x", [1, 1], DType::F64);
    let mut values = BTreeMap::from([("x".into(), x)]);
    lower(&mut default_dtype, Msg::new(&eye_like(&[])), &mut values, &mut BTreeMap::new()).unwrap();
    assert_eq!(default_dtype.dtype(values["out"]).unwrap(), DType::F64);

    let mut duplicate = eye_like(&[]);
    field(&mut duplicate, 5, &typed_int_attr("k", 0));
    field(&mut duplicate, 5, &typed_int_attr("k", 1));
    let mut wrong_float = eye_like(&[]);
    field(&mut wrong_float, 5, &float_attr("k", 1.));
    let mut untyped = eye_like(&[]);
    field(&mut untyped, 5, &int64_attr("dtype", 1));
    let mut unknown = eye_like(&[]);
    field(&mut unknown, 5, &typed_int_attr("other", 1));
    let mut invalid_dtype = eye_like(&[]);
    field(&mut invalid_dtype, 5, &typed_int_attr("dtype", 8));
    for invalid in [
        node("EyeLike", &[], "out"),
        node("EyeLike", &["x", "extra"], "out"),
        duplicate,
        wrong_float,
        untyped,
        unknown,
        invalid_dtype,
        eye_like(&[typed_int_attr("k", 5)]),
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [2, 4]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }
    for shape in [Vec::new(), vec![2], vec![1, 1, 1]] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", shape);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&eye_like(&[])), &mut values, &mut BTreeMap::new()).is_err());
        assert_eq!(malformed.node_count(), before_nodes);
        assert_eq!(values["x"], x);
        assert!(!values.contains_key("out"));
    }
    let mut missing = Graph::new();
    let mut values = BTreeMap::new();
    let before_nodes = missing.node_count();
    assert!(lower(&mut missing, Msg::new(&node("EyeLike", &["missing"], "out")), &mut values, &mut BTreeMap::new()).is_err());
    assert_eq!(missing.node_count(), before_nodes);

    let mut overflow = Graph::new();
    let x = overflow.input("x", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let before_values = values.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(&mut overflow, Msg::new(&eye_like(&[])), &mut values, &mut BTreeMap::new()).is_err());
    assert_eq!(values, before_values);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn space_to_depth_matches_tinygrad_hblock_wblock_channel_order_and_preflights() {
    let space_to_depth = |attrs: &[Vec<u8>]| {
        let mut encoded = node("SpaceToDepth", &["x"], "out");
        for attr in attrs {
            field(&mut encoded, 5, attr);
        }
        encoded
    };
    let run = |shape: Vec<usize>, dtype: DType, blocksize: i64, data: Vec<f32>| {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", shape.clone(), dtype);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        lower(
            &mut graph,
            Msg::new(&space_to_depth(&[typed_int_attr("blocksize", blocksize)])),
            &mut values,
            &mut constants,
        )
        .unwrap();
        let output = CpuBackend
            .execute(
                &graph,
                values["out"],
                &HashMap::from([("x".into(), TensorData::new(shape, data).unwrap())]),
            )
            .unwrap();
        (graph, values, constants, output)
    };

    // `h1, w1, c` is observable: the two original channels alternate within
    // each spatial block rather than forming four C-sized channel groups.
    let (_, _, _, ordered) = run(
        vec![1, 2, 2, 2],
        DType::F32,
        2,
        (0..8).map(|value| value as f32).collect(),
    );
    assert_eq!(ordered.shape().dims(), &[1, 8, 1, 1]);
    assert_eq!(ordered.values(), &[0., 4., 1., 5., 2., 6., 3., 7.]);

    let (_, _, _, rectangular) = run(
        vec![1, 1, 4, 6],
        DType::F32,
        2,
        (0..24).map(|value| value as f32).collect(),
    );
    assert_eq!(rectangular.shape().dims(), &[1, 4, 2, 3]);
    assert_eq!(rectangular.values(), &[0., 2., 4., 12., 14., 16., 1., 3., 5., 13., 15., 17., 6., 8., 10., 18., 20., 22., 7., 9., 11., 19., 21., 23.]);

    let (identity_graph, identity_values, identity_constants, identity) = run(
        vec![1, 2, 2, 2],
        DType::F32,
        1,
        (0..8).map(|value| value as f32).collect(),
    );
    assert_eq!(identity_values["out"], identity_values["x"]);
    assert_eq!(identity_graph.node_count(), 1);
    assert!(identity_constants.is_empty());
    assert_eq!(identity.values(), &[0., 1., 2., 3., 4., 5., 6., 7.]);

    for shape in [vec![0, 2, 2, 2], vec![1, 0, 2, 2], vec![1, 2, 0, 2], vec![1, 2, 2, 0]] {
        let (graph, values, constants, output) = run(shape.clone(), DType::F32, 2, vec![]);
        let expected = match shape.as_slice() {
            [0, 2, 2, 2] => vec![0, 8, 1, 1],
            [1, 0, 2, 2] => vec![1, 0, 1, 1],
            [1, 2, 0, 2] => vec![1, 8, 0, 1],
            [1, 2, 2, 0] => vec![1, 8, 1, 0],
            _ => unreachable!(),
        };
        assert_eq!(output.shape().dims(), expected.as_slice());
        assert_eq!(graph.dtype(values["out"]).unwrap(), DType::F32);
        assert!(constants.is_empty());
    }

    for dtype in [
        DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16, DType::I32,
        DType::U32, DType::I64, DType::U64, DType::F16, DType::BF16, DType::F32,
        DType::F64,
    ] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [1, 1, 2, 2], dtype);
        let mut values = BTreeMap::from([("x".into(), x)]);
        lower(
            &mut graph,
            Msg::new(&space_to_depth(&[typed_int_attr("blocksize", 2)])),
            &mut values,
            &mut BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(graph.dtype(values["out"]).unwrap(), dtype);
        assert_eq!(graph.shape(values["out"]).unwrap().dims(), &[1, 4, 1, 1]);
    }

    let mut duplicate = space_to_depth(&[]);
    field(&mut duplicate, 5, &typed_int_attr("blocksize", 2));
    field(&mut duplicate, 5, &typed_int_attr("blocksize", 3));
    let mut wrong_float = space_to_depth(&[]);
    field(&mut wrong_float, 5, &float_attr("blocksize", 2.));
    let mut untyped = space_to_depth(&[]);
    field(&mut untyped, 5, &int64_attr("blocksize", 2));
    let mut unknown = space_to_depth(&[]);
    field(&mut unknown, 5, &typed_int_attr("other", 2));
    for invalid in [
        space_to_depth(&[]),
        duplicate,
        wrong_float,
        untyped,
        unknown,
        space_to_depth(&[typed_int_attr("blocksize", 0)]),
        space_to_depth(&[typed_int_attr("blocksize", -1)]),
        node("SpaceToDepth", &[], "out"),
        node("SpaceToDepth", &["x", "extra"], "out"),
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [1, 1, 2, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }
    for shape in [
        vec![], vec![1, 2, 2], vec![1, 2, 2, 2, 1], vec![1, 1, 3, 2], vec![1, 1, 2, 3],
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", shape);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let before_values = values.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&space_to_depth(&[typed_int_attr("blocksize", 2)])),
            &mut values,
            &mut BTreeMap::new(),
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut missing = Graph::new();
    let mut values = BTreeMap::new();
    let before_nodes = missing.node_count();
    assert!(lower(
        &mut missing,
        Msg::new(&space_to_depth(&[typed_int_attr("blocksize", 2)])),
        &mut values,
        &mut BTreeMap::new(),
    )
    .is_err());
    assert_eq!(missing.node_count(), before_nodes);

    // Product and byte limits are both checked before the first reshape.
    for (shape, dtype) in [
        (vec![1, usize::MAX, 0, 2], DType::U8),
        (vec![usize::MAX, 1, 1, 1], DType::F32),
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input_dtype("x", shape, dtype);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let before_values = values.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&space_to_depth(&[typed_int_attr("blocksize", 2)])),
            &mut values,
            &mut BTreeMap::new(),
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(malformed.node_count(), before_nodes);
    }
}

#[test]
fn depth_to_space_matches_tinygrad_modes_and_source_empty_preflight() {
    let depth_to_space = |attrs: &[Vec<u8>]| {
        let mut encoded = node("DepthToSpace", &["x"], "out");
        for attr in attrs {
            field(&mut encoded, 5, attr);
        }
        encoded
    };
    let run = |shape: Vec<usize>, dtype: DType, attrs: &[Vec<u8>], data: Vec<f32>| {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", shape.clone(), dtype);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        lower(&mut graph, Msg::new(&depth_to_space(attrs)), &mut values, &mut constants).unwrap();
        let output = CpuBackend
            .execute(
                &graph,
                values["out"],
                &HashMap::from([("x".into(), TensorData::new(shape, data).unwrap())]),
            )
            .unwrap();
        (graph, values, constants, output)
    };
    let block_two = typed_int_attr("blocksize", 2);

    // Default DCR treats the input channel order as (h1, w1, c).
    let (_, _, _, dcr) = run(
        vec![1, 8, 1, 1],
        DType::F32,
        &[block_two.clone()],
        (0..8).map(|value| value as f32).collect(),
    );
    assert_eq!(dcr.shape().dims(), &[1, 2, 2, 2]);
    assert_eq!(dcr.values(), &[0., 2., 4., 6., 1., 3., 5., 7.]);

    // CRD instead treats the source channel order as (c, h1, w1).
    let (_, _, _, crd) = run(
        vec![1, 8, 1, 1],
        DType::F32,
        &[block_two.clone(), typed_string_attr("mode", "CRD")],
        (0..8).map(|value| value as f32).collect(),
    );
    assert_eq!(crd.shape().dims(), &[1, 2, 2, 2]);
    assert_eq!(crd.values(), &[0., 1., 2., 3., 4., 5., 6., 7.]);
    let (_, _, _, arbitrary_mode) = run(
        vec![1, 8, 1, 1],
        DType::F32,
        &[block_two.clone(), typed_string_attr("mode", "not-crd")],
        (0..8).map(|value| value as f32).collect(),
    );
    assert_eq!(arbitrary_mode.values(), dcr.values());

    let block_one = typed_int_attr("blocksize", 1);
    let (identity_graph, identity_values, identity_constants, identity) = run(
        vec![1, 2, 2, 3],
        DType::F32,
        &[block_one],
        (0..12).map(|value| value as f32).collect(),
    );
    assert_eq!(identity_values["out"], identity_values["x"]);
    assert_eq!(identity_graph.node_count(), 1);
    assert!(identity_constants.is_empty());
    let expected_identity: Vec<f32> = (0..12).map(|value| value as f32).collect();
    assert_eq!(identity.values(), expected_identity.as_slice());

    // A zero channel extent has a nonzero inferred-shape denominator and is
    // accepted by tinygrad; B/H/W zero makes that denominator zero instead.
    let mut zero_channel = Graph::new();
    let x = zero_channel.input_dtype("x", [1, 0, 2, 3], DType::F32);
    let mut values = BTreeMap::from([("x".into(), x)]);
    lower(
        &mut zero_channel,
        Msg::new(&depth_to_space(&[block_two.clone()])),
        &mut values,
        &mut BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(zero_channel.shape(values["out"]).unwrap().dims(), &[1, 0, 4, 6]);

    for dtype in [
        DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16, DType::I32,
        DType::U32, DType::I64, DType::U64, DType::F16, DType::BF16, DType::F32,
        DType::F64,
    ] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [1, 4, 2, 2], dtype);
        let mut values = BTreeMap::from([("x".into(), x)]);
        lower(
            &mut graph,
            Msg::new(&depth_to_space(&[block_two.clone()])),
            &mut values,
            &mut BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(graph.dtype(values["out"]).unwrap(), dtype);
        assert_eq!(graph.shape(values["out"]).unwrap().dims(), &[1, 1, 4, 4]);
    }

    let mut duplicate = depth_to_space(&[]);
    field(&mut duplicate, 5, &typed_int_attr("blocksize", 2));
    field(&mut duplicate, 5, &typed_int_attr("blocksize", 3));
    let mut duplicate_mode = depth_to_space(&[]);
    field(&mut duplicate_mode, 5, &typed_int_attr("blocksize", 2));
    field(&mut duplicate_mode, 5, &typed_string_attr("mode", "DCR"));
    field(&mut duplicate_mode, 5, &typed_string_attr("mode", "CRD"));
    let mut wrong_mode_type = depth_to_space(&[]);
    field(&mut wrong_mode_type, 5, &typed_int_attr("blocksize", 2));
    field(&mut wrong_mode_type, 5, &typed_int_attr("mode", 1));
    let mut invalid_utf8 = depth_to_space(&[]);
    field(&mut invalid_utf8, 5, &typed_int_attr("blocksize", 2));
    let mut invalid_mode_attr = vec![];
    text(&mut invalid_mode_attr, 1, "mode");
    field(&mut invalid_mode_attr, 4, &[0xff]);
    var(&mut invalid_mode_attr, 20, 3);
    field(&mut invalid_utf8, 5, &invalid_mode_attr);
    let mut unknown = depth_to_space(&[]);
    field(&mut unknown, 5, &typed_int_attr("blocksize", 2));
    field(&mut unknown, 5, &typed_int_attr("other", 1));
    for invalid in [
        depth_to_space(&[]),
        duplicate,
        duplicate_mode,
        wrong_mode_type,
        invalid_utf8,
        unknown,
        depth_to_space(&[int64_attr("blocksize", 2)]),
        depth_to_space(&[float_attr("blocksize", 2.)]),
        depth_to_space(&[block_two.clone(), string_attr("mode", "DCR")]),
        depth_to_space(&[typed_int_attr("blocksize", 0)]),
        depth_to_space(&[typed_int_attr("blocksize", -1)]),
        node("DepthToSpace", &[], "out"),
        node("DepthToSpace", &["x", "extra"], "out"),
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [1, 4, 2, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }
    for shape in [
        vec![], vec![1, 4, 2], vec![1, 4, 2, 2, 1], vec![1, 3, 2, 2],
        vec![0, 4, 2, 2], vec![1, 4, 0, 2], vec![1, 4, 2, 0],
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", shape);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let before_values = values.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&depth_to_space(&[block_two.clone()])),
            &mut values,
            &mut BTreeMap::new(),
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut missing = Graph::new();
    let mut values = BTreeMap::new();
    let before_nodes = missing.node_count();
    assert!(lower(
        &mut missing,
        Msg::new(&depth_to_space(&[block_two.clone()])),
        &mut values,
        &mut BTreeMap::new(),
    )
    .is_err());
    assert_eq!(missing.node_count(), before_nodes);

    for (shape, blocksize) in [
        (vec![1, 0, 1, 1], i64::MAX),
        (vec![1, 0, usize::MAX, 1], 2),
        (vec![usize::MAX, 4, 1, 1], 2),
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input_dtype("x", shape, DType::F32);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let before_values = values.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&depth_to_space(&[typed_int_attr("blocksize", blocksize)])),
            &mut values,
            &mut BTreeMap::new(),
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(malformed.node_count(), before_nodes);
    }
}

#[test]
fn center_crop_pad_matches_tinygrad_zip_ranges_and_fail_closed_pad_boundary() {
    let center_crop_pad = |attrs: &[Vec<u8>]| {
        let mut encoded = node("CenterCropPad", &["x", "shape"], "out");
        for attr in attrs {
            field(&mut encoded, 5, attr);
        }
        encoded
    };
    let shape_data = |values: &[i64]| {
        TensorData::from_scalars(
            [values.len()],
            DType::I64,
            values.iter().copied().map(Scalar::I),
        )
        .unwrap()
    };
    let run = |shape: Vec<usize>, attrs: &[Vec<u8>], target: &[i64], data: Vec<f32>| {
        let mut graph = Graph::new();
        let x = graph.input("x", shape.clone());
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::from([("shape".into(), shape_data(target))]);
        lower(&mut graph, Msg::new(&center_crop_pad(attrs)), &mut values, &mut constants).unwrap();
        let output = CpuBackend
            .execute(
                &graph,
                values["out"],
                &HashMap::from([("x".into(), TensorData::new(shape, data).unwrap())]),
            )
            .unwrap();
        (graph, values, constants, output)
    };

    // Axis 1 has an odd crop (five to two: one low, two high); axis 2 has
    // an odd pad (four to seven: one low, two high).
    let axes = typed_ints_attr("axes", &[1, 2]);
    let (mixed_graph, mixed_values, _, mixed) = run(
        vec![1, 5, 4],
        &[axes],
        &[2, 7],
        (0..20).map(|value| value as f32).collect(),
    );
    assert_eq!(mixed.shape().dims(), &[1, 2, 7]);
    assert_eq!(mixed.values(), &[0., 4., 5., 6., 7., 0., 0., 0., 8., 9., 10., 11., 0., 0.]);
    // The planned Pad remains outside generic scheduling, before any live
    // work/cache path can begin.
    assert!(crate::schedule(&mixed_graph, mixed_values["out"]).is_err());

    let (_, _, _, default_axes) = run(
        vec![1, 5, 4],
        &[],
        &[2],
        (0..20).map(|value| value as f32).collect(),
    );
    assert_eq!(default_axes.shape().dims(), &[2, 5, 4]);
    let (_, _, _, empty_axes) = run(
        vec![1, 5, 4],
        &[typed_ints_attr("axes", &[])],
        &[2],
        (0..20).map(|value| value as f32).collect(),
    );
    assert_eq!(empty_axes.shape().dims(), &[2, 5, 4]);
    let (_, _, _, negative_axis) = run(
        vec![1, 5, 4],
        &[typed_ints_attr("axes", &[-1])],
        &[2],
        (0..20).map(|value| value as f32).collect(),
    );
    assert_eq!(negative_axis.shape().dims(), &[1, 5, 2]);
    let (_, _, _, duplicate_axis) = run(
        vec![1, 5, 4],
        &[typed_ints_attr("axes", &[1, 1])],
        &[2, 3],
        (0..20).map(|value| value as f32).collect(),
    );
    assert_eq!(duplicate_axis.shape().dims(), &[1, 3, 4]);
    let (_, _, _, unequal) = run(
        vec![1, 5, 4],
        &[typed_ints_attr("axes", &[1])],
        &[2, -1],
        (0..20).map(|value| value as f32).collect(),
    );
    assert_eq!(unequal.shape().dims(), &[1, 2, 4]);
    let (_, _, _, extra_axes) = run(
        vec![1, 5, 4],
        &[typed_ints_attr("axes", &[1, 2])],
        &[2],
        (0..20).map(|value| value as f32).collect(),
    );
    assert_eq!(extra_axes.shape().dims(), &[1, 2, 4]);

    let mut scalar = Graph::new();
    let scalar_x = scalar.input("x", []);
    let mut scalar_values = BTreeMap::from([("x".into(), scalar_x)]);
    let mut scalar_constants = BTreeMap::from([("shape".into(), shape_data(&[]))]);
    lower(
        &mut scalar,
        Msg::new(&center_crop_pad(&[])),
        &mut scalar_values,
        &mut scalar_constants,
    )
    .unwrap();
    assert_eq!(scalar_values["out"], scalar_x);
    assert_eq!(scalar.node_count(), 1);

    let (_, _, _, empty_padded) = run(vec![0, 2], &[], &[2, 2], vec![]);
    assert_eq!(empty_padded.shape().dims(), &[2, 2]);
    assert_eq!(empty_padded.values(), &[0.; 4]);
    let (_, _, _, zero_target) = run(
        vec![1, 2],
        &[typed_ints_attr("axes", &[1])],
        &[0],
        vec![1., 2.],
    );
    assert_eq!(zero_target.shape().dims(), &[1, 0]);

    for dtype in [
        DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16, DType::I32,
        DType::U32, DType::I64, DType::U64, DType::F16, DType::BF16, DType::F32,
        DType::F64,
    ] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [1, 2], dtype);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::from([("shape".into(), shape_data(&[1, 3]))]);
        lower(&mut graph, Msg::new(&center_crop_pad(&[])), &mut values, &mut constants).unwrap();
        assert_eq!(graph.dtype(values["out"]).unwrap(), dtype);
        assert_eq!(graph.shape(values["out"]).unwrap().dims(), &[1, 3]);
    }

    let mut duplicate = center_crop_pad(&[]);
    field(&mut duplicate, 5, &typed_ints_attr("axes", &[1]));
    field(&mut duplicate, 5, &typed_ints_attr("axes", &[0]));
    let mut wrong_scalar = center_crop_pad(&[]);
    field(&mut wrong_scalar, 5, &typed_int_attr("axes", 1));
    let mut untyped = center_crop_pad(&[]);
    field(&mut untyped, 5, &ints_attr("axes", &[1]));
    let mut unknown = center_crop_pad(&[]);
    field(&mut unknown, 5, &typed_ints_attr("other", &[1]));
    for invalid in [
        node("CenterCropPad", &["x"], "out"),
        node("CenterCropPad", &["x", "shape", "extra"], "out"),
        duplicate,
        wrong_scalar,
        untyped,
        unknown,
        center_crop_pad(&[typed_ints_attr("axes", &[9])]),
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [1, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::from([("shape".into(), shape_data(&[1]))]);
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }
    for shape in [
        TensorData::scalar_with_dtype(Scalar::I(1), DType::I64),
        TensorData::from_scalars([1, 1], DType::I64, [Scalar::I(1), Scalar::I(2)]).unwrap(),
        TensorData::from_scalars([1], DType::I32, [Scalar::I(1)]).unwrap(),
        shape_data(&[-1]),
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [1, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::from([("shape".into(), shape)]);
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&center_crop_pad(&[typed_ints_attr("axes", &[1])])),
            &mut values,
            &mut constants,
        )
        .is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut missing = Graph::new();
    let mut values = BTreeMap::new();
    assert!(lower(
        &mut missing,
        Msg::new(&center_crop_pad(&[])),
        &mut values,
        &mut BTreeMap::new(),
    )
    .is_err());
    assert!(!values.contains_key("out"));

    for (input_shape, target) in [
        (vec![usize::MAX, 1], vec![1, 1]),
        (vec![0, i64::MAX as usize], vec![i64::MAX, i64::MAX]),
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input_dtype("x", input_shape, DType::F32);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::from([("shape".into(), shape_data(&target))]);
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&center_crop_pad(&[])), &mut values, &mut constants).is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }
}

#[test]
fn hardmax_matches_tinygrad_first_ties_and_leading_nan_sentinel() {
    let hardmax = |attrs: &[Vec<u8>]| {
        let mut encoded = node("Hardmax", &["x"], "out");
        for attr in attrs {
            field(&mut encoded, 5, attr);
        }
        encoded
    };

    let mut graph = Graph::new();
    let x = graph.input("x", [7, 3]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&hardmax(&[])),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new(
                    [7, 3],
                    vec![
                        1., 3., 3., // first equal maximum
                        -0.0, 0.0, -1., // first signed-zero tie
                        0.0, -0.0, -1., // reverse signed-zero order
                        f32::NAN, 2., 3., // leading NaN -> sentinel/all zero
                        2., f32::NAN, 3., // later NaN is ignored
                        f32::NAN, f32::NAN, f32::NAN, // all NaN -> all zero
                        f32::INFINITY, 1., f32::NEG_INFINITY,
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[7, 3]);
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(
        output.values(),
        &[
            0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
            0., 0.,
        ]
    );
    // ArgReduce remains deliberately outside generic scheduling, so this
    // importer cannot create native/JIT work or a cache entry.
    assert!(crate::schedule(&graph, values["out"]).is_err());

    // Axis normalization is against the original rank, not the restored
    // one-hot rank, and every storage dtype is restored after the bool mask.
    for axis in [-2, 0] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2, 3], DType::I16);
        let mut bindings = BTreeMap::from([("x".into(), x)]);
        lower(
            &mut graph,
            Msg::new(&hardmax(&[int64_attr("axis", axis)])),
            &mut bindings,
            &mut BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(graph.shape(bindings["out"]).unwrap().dims(), &[2, 3]);
        assert_eq!(graph.dtype(bindings["out"]).unwrap(), DType::I16);
    }
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [1], dtype);
        let mut bindings = BTreeMap::from([("x".into(), x)]);
        lower(
            &mut graph,
            Msg::new(&hardmax(&[])),
            &mut bindings,
            &mut BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(graph.shape(bindings["out"]).unwrap().dims(), &[1]);
        assert_eq!(graph.dtype(bindings["out"]).unwrap(), dtype);
    }

    // Empty layouts are source-identities after all static shape/axis facts
    // are checked; no ArgReduce, range, or constants are appended.
    for shape in [[0, 2], [2, 0]] {
        let mut empty = Graph::new();
        let x = empty.input("x", shape);
        let mut bindings = BTreeMap::from([("x".into(), x)]);
        let before = empty.node_count();
        lower(
            &mut empty,
            Msg::new(&hardmax(&[])),
            &mut bindings,
            &mut BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(bindings["out"], x);
        assert_eq!(empty.node_count(), before);
    }

    for invalid in [
        node("Hardmax", &[], "out"),
        node("Hardmax", &["x", "extra"], "out"),
        hardmax(&[int_attr("unknown", 1)]),
        hardmax(&[int64_attr("axis", -2)]),
        hardmax(&[int64_attr("axis", 1)]),
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [2]);
        let mut bindings = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        let before_values = bindings.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(
            &mut malformed,
            Msg::new(&invalid),
            &mut bindings,
            &mut constants,
        )
        .is_err());
        assert_eq!(bindings, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }
    let mut scalar = Graph::new();
    let x = scalar.input("x", []);
    let mut bindings = BTreeMap::from([("x".into(), x)]);
    let before = scalar.node_count();
    assert!(lower(
        &mut scalar,
        Msg::new(&hardmax(&[])),
        &mut bindings,
        &mut BTreeMap::new(),
    )
    .is_err());
    assert_eq!(scalar.node_count(), before);
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
    lower(
        &mut g,
        Msg::new(&node("GlobalAveragePool", &["x"], "z")),
        &mut values,
        &mut BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(g.shape(values["z"]).unwrap().dims(), &[1, 2]);
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
fn reduce_sum_square_matches_tinygrad_typed_sum_and_preflights() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("ReduceSumSquare", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 2], vec![1., -2., 3., -4.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[]);
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values(), &[30.]);

    let axes = TensorData::from_scalars([1], DType::I64, [Scalar::I(-1)]).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), axes)]);
    let mut keep = node("ReduceSumSquare", &["x", "axes"], "out");
    field(&mut keep, 5, &int_attr("keepdims", 1));
    lower(&mut graph, Msg::new(&keep), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 2], vec![1., -2., 3., -4.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2, 1]);
    assert_eq!(output.values(), &[5., 25.]);

    let empty_axes = TensorData::from_scalars([0], DType::I64, []).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), empty_axes)]);
    let mut noop = node("ReduceSumSquare", &["x", "axes"], "out");
    field(&mut noop, 5, &int_attr("noop_with_empty_axes", 1));
    lower(&mut graph, Msg::new(&noop), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2], vec![-2., 3.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2]);
    assert_eq!(output.values(), &[4., 9.]);

    for (dtype, data, expected) in [
        (
            DType::Bool,
            TensorData::from_scalars([], DType::Bool, [Scalar::Bool(true)]).unwrap(),
            DType::I32,
        ),
        (
            DType::I8,
            TensorData::from_scalars([], DType::I8, [Scalar::I(2)]).unwrap(),
            DType::I32,
        ),
        (
            DType::I32,
            TensorData::from_scalars([], DType::I32, [Scalar::I(2)]).unwrap(),
            DType::I32,
        ),
        (
            DType::U8,
            TensorData::from_scalars([], DType::U8, [Scalar::U(2)]).unwrap(),
            DType::U32,
        ),
        (
            DType::U32,
            TensorData::from_scalars([], DType::U32, [Scalar::U(2)]).unwrap(),
            DType::U32,
        ),
        (
            DType::I64,
            TensorData::from_scalars([], DType::I64, [Scalar::I(2)]).unwrap(),
            DType::I64,
        ),
        (
            DType::U64,
            TensorData::from_scalars([], DType::U64, [Scalar::U(2)]).unwrap(),
            DType::U64,
        ),
        (
            DType::F16,
            TensorData::from_scalars([], DType::F16, [Scalar::F(2.)]).unwrap(),
            DType::F16,
        ),
        (
            DType::BF16,
            TensorData::from_scalars([], DType::BF16, [Scalar::F(2.)]).unwrap(),
            DType::BF16,
        ),
    ] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [], dtype);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        lower(
            &mut graph,
            Msg::new(&node("ReduceSumSquare", &["x"], "out")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        let output = CpuBackend
            .execute(&graph, values["out"], &HashMap::from([("x".into(), data)]))
            .unwrap();
        assert_eq!(output.dtype(), expected);
        assert_eq!(output.values(), &[if dtype == DType::Bool { 1. } else { 4. }]);
    }

    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [], DType::I8);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("ReduceSumSquare", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::from_scalars([], DType::I8, [Scalar::I(16)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::I32);
    assert_eq!(output.values(), &[0.]);

    let mut graph = Graph::new();
    let x = graph.input("x", [1]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("ReduceSumSquare", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([("x".into(), TensorData::new([1], vec![-0.0]).unwrap())]),
        )
        .unwrap();
    assert_eq!(output.values()[0].to_bits(), 0.0f32.to_bits());

    let mut graph = Graph::new();
    let x = graph.input("x", [3]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("ReduceSumSquare", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([3], vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY]).unwrap(),
            )]),
        )
        .unwrap();
    assert!(output.values()[0].is_nan());

    let axes = TensorData::from_scalars([1], DType::I64, [Scalar::I(1)]).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 0]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), axes)]);
    lower(
        &mut graph,
        Msg::new(&node("ReduceSumSquare", &["x", "axes"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([("x".into(), TensorData::new([2, 0], vec![]).unwrap())]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2]);
    assert_eq!(output.values(), &[0., 0.]);

    let mut unknown = node("ReduceSumSquare", &["x"], "out");
    field(&mut unknown, 5, &int_attr("axis", 0));
    let mut bad_keep = node("ReduceSumSquare", &["x"], "out");
    field(&mut bad_keep, 5, &int_attr("keepdims", 2));
    let mut bad_noop = node("ReduceSumSquare", &["x"], "out");
    field(&mut bad_noop, 5, &int_attr("noop_with_empty_axes", 2));
    let duplicate_axes = TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(-2)]).unwrap();
    let rank_zero_axes = TensorData::from_scalars([], DType::I64, [Scalar::I(0)]).unwrap();
    let wrong_dtype_axes = TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap();
    for (invalid, axes) in [
        (node("ReduceSumSquare", &[], "out"), None),
        (node("ReduceSumSquare", &["x", "axes", "extra"], "out"), None),
        (unknown, None),
        (bad_keep, None),
        (bad_noop, None),
        (node("ReduceSumSquare", &["x", "missing"], "out"), None),
        (node("ReduceSumSquare", &["x", "axes"], "out"), Some(duplicate_axes)),
        (node("ReduceSumSquare", &["x", "axes"], "out"), Some(rank_zero_axes)),
        (node("ReduceSumSquare", &["x", "axes"], "out"), Some(wrong_dtype_axes)),
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [2, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = axes.map(|axes| BTreeMap::from([("axes".into(), axes)])).unwrap_or_default();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
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
        Msg::new(&node("ReduceSumSquare", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn reduce_l1_matches_tinygrad_abs_then_typed_sum_and_preflights() {
    // Default axes reduce all dimensions after tinygrad's source-level
    // `x * x.sign()` absolute-value composition.
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("ReduceL1", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 2], vec![1., -2., 3., -4.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[]);
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values(), &[10.]);

    // Signed axes and keepdims share the opset-13 ReductionPlan.
    let axes = TensorData::from_scalars([1], DType::I64, [Scalar::I(-1)]).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), axes)]);
    let mut keep = node("ReduceL1", &["x", "axes"], "out");
    field(&mut keep, 5, &int_attr("keepdims", 1));
    lower(&mut graph, Msg::new(&keep), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 2], vec![1., -2., 3., -4.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2, 1]);
    assert_eq!(output.values(), &[3., 7.]);

    // Unlike UnaryOp::Abs, the source composition leaves negative zero alone
    // when empty axes request the noop path.
    let empty_axes = TensorData::from_scalars([0], DType::I64, []).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [4]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), empty_axes)]);
    let mut noop = node("ReduceL1", &["x", "axes"], "out");
    field(&mut noop, 5, &int_attr("noop_with_empty_axes", 1));
    lower(&mut graph, Msg::new(&noop), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([4], vec![-0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY])
                    .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[4]);
    assert_eq!(output.values()[0].to_bits(), (-0.0f32).to_bits());
    assert!(output.values()[1].is_nan());
    assert_eq!(output.values()[2], f32::INFINITY);
    assert_eq!(output.values()[3], f32::INFINITY);

    // Sum's existing typed accumulator/output policy remains intact after
    // the shape- and dtype-preserving absolute-value composition.
    for (dtype, data, expected_dtype, expected) in [
        (
            DType::Bool,
            TensorData::from_scalars([], DType::Bool, [Scalar::Bool(true)]).unwrap(),
            DType::I32,
            1.0,
        ),
        (
            DType::I8,
            TensorData::from_scalars([], DType::I8, [Scalar::I(-2)]).unwrap(),
            DType::I32,
            2.0,
        ),
        (
            DType::I16,
            TensorData::from_scalars([], DType::I16, [Scalar::I(-2)]).unwrap(),
            DType::I32,
            2.0,
        ),
        (
            DType::I32,
            TensorData::from_scalars([], DType::I32, [Scalar::I(-2)]).unwrap(),
            DType::I32,
            2.0,
        ),
        (
            DType::U8,
            TensorData::from_scalars([], DType::U8, [Scalar::U(2)]).unwrap(),
            DType::U32,
            2.0,
        ),
        (
            DType::U16,
            TensorData::from_scalars([], DType::U16, [Scalar::U(2)]).unwrap(),
            DType::U32,
            2.0,
        ),
        (
            DType::U32,
            TensorData::from_scalars([], DType::U32, [Scalar::U(2)]).unwrap(),
            DType::U32,
            2.0,
        ),
        (
            DType::I64,
            TensorData::from_scalars([], DType::I64, [Scalar::I(-2)]).unwrap(),
            DType::I64,
            2.0,
        ),
        (
            DType::U64,
            TensorData::from_scalars([], DType::U64, [Scalar::U(2)]).unwrap(),
            DType::U64,
            2.0,
        ),
        (
            DType::F16,
            TensorData::from_scalars([], DType::F16, [Scalar::F(-2.)]).unwrap(),
            DType::F16,
            2.0,
        ),
        (
            DType::BF16,
            TensorData::from_scalars([], DType::BF16, [Scalar::F(-2.)]).unwrap(),
            DType::BF16,
            2.0,
        ),
    ] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [], dtype);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        lower(
            &mut graph,
            Msg::new(&node("ReduceL1", &["x"], "out")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        let output = CpuBackend
            .execute(&graph, values["out"], &HashMap::from([("x".into(), data)]))
            .unwrap();
        assert_eq!(output.dtype(), expected_dtype, "{dtype:?}");
        assert_eq!(output.values(), &[expected], "{dtype:?}");
    }

    // Two's-complement signed minima wrap through the same multiply-by-sign
    // path as tinygrad rather than saturating.
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [], DType::I8);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("ReduceL1", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::from_scalars([], DType::I8, [Scalar::I(-128)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::I32);
    assert_eq!(output.values(), &[-128.]);

    let mut graph = Graph::new();
    let x = graph.input("x", [3]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("ReduceL1", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([3], vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY]).unwrap(),
            )]),
        )
        .unwrap();
    assert!(output.values()[0].is_nan());

    // Empty reduction domains keep typed Sum's additive zero.
    let axes = TensorData::from_scalars([1], DType::I64, [Scalar::I(1)]).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 0]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), axes)]);
    lower(
        &mut graph,
        Msg::new(&node("ReduceL1", &["x", "axes"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([("x".into(), TensorData::new([2, 0], vec![]).unwrap())]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2]);
    assert_eq!(output.values(), &[0., 0.]);

    let mut unknown = node("ReduceL1", &["x"], "out");
    field(&mut unknown, 5, &int_attr("axis", 0));
    let mut bad_keep = node("ReduceL1", &["x"], "out");
    field(&mut bad_keep, 5, &int_attr("keepdims", 2));
    let mut bad_noop = node("ReduceL1", &["x"], "out");
    field(&mut bad_noop, 5, &int_attr("noop_with_empty_axes", 2));
    let duplicate_axes = TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(-2)]).unwrap();
    let rank_zero_axes = TensorData::from_scalars([], DType::I64, [Scalar::I(0)]).unwrap();
    let wrong_dtype_axes = TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap();
    for (invalid, axes) in [
        (node("ReduceL1", &[], "out"), None),
        (node("ReduceL1", &["x", "axes", "extra"], "out"), None),
        (unknown, None),
        (bad_keep, None),
        (bad_noop, None),
        (node("ReduceL1", &["x", "missing"], "out"), None),
        (node("ReduceL1", &["x", "axes"], "out"), Some(duplicate_axes)),
        (node("ReduceL1", &["x", "axes"], "out"), Some(rank_zero_axes)),
        (node("ReduceL1", &["x", "axes"], "out"), Some(wrong_dtype_axes)),
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [2, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = axes.map(|axes| BTreeMap::from([("axes".into(), axes)])).unwrap_or_default();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
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
        Msg::new(&node("ReduceL1", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn reduce_l2_matches_tinygrad_widen_square_sum_sqrt_and_preflights() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("ReduceL2", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([("x".into(), TensorData::new([2], vec![3., 4.]).unwrap())]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[]);
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values(), &[5.]);

    let axes = TensorData::from_scalars([1], DType::I64, [Scalar::I(-1)]).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), axes)]);
    let mut keep = node("ReduceL2", &["x", "axes"], "out");
    field(&mut keep, 5, &int_attr("keepdims", 1));
    lower(&mut graph, Msg::new(&keep), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 2], vec![3., 4., 5., 12.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2, 1]);
    assert_eq!(output.values(), &[5., 13.]);

    // no-op still performs square then sqrt, so negative zero becomes +0 and
    // the ordinary floating nonfinite path remains visible elementwise.
    let empty_axes = TensorData::from_scalars([0], DType::I64, []).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [4]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), empty_axes)]);
    let mut noop = node("ReduceL2", &["x", "axes"], "out");
    field(&mut noop, 5, &int_attr("noop_with_empty_axes", 1));
    lower(&mut graph, Msg::new(&noop), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([4], vec![-0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY])
                    .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[4]);
    assert_eq!(output.values()[0].to_bits(), 0.0f32.to_bits());
    assert!(output.values()[1].is_nan());
    assert_eq!(output.values()[2], f32::INFINITY);
    assert_eq!(output.values()[3], f32::INFINITY);

    // F16/BF16 are widened before `work * work`: 255 squared would overflow
    // at F16 storage width, but the source composition performs it in F32 and
    // narrows only after sqrt.
    for dtype in [DType::F16, DType::BF16] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [], dtype);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        lower(
            &mut graph,
            Msg::new(&node("ReduceL2", &["x"], "out")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        let output = CpuBackend
            .execute(
                &graph,
                values["out"],
                &HashMap::from([(
                    "x".into(),
                    TensorData::from_scalars([], dtype, [Scalar::F(255.)]).unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(output.dtype(), dtype);
        assert_eq!(output.values(), &[255.]);
    }

    // Integer and Bool square at their source storage width, Sum with the
    // standard accumulator, then sqrt and cast all the way back to source.
    for (dtype, data, expected) in [
        (
            DType::Bool,
            TensorData::from_scalars([], DType::Bool, [Scalar::Bool(true)]).unwrap(),
            1.0,
        ),
        (
            DType::I8,
            TensorData::from_scalars([], DType::I8, [Scalar::I(-2)]).unwrap(),
            2.0,
        ),
        (
            DType::I16,
            TensorData::from_scalars([], DType::I16, [Scalar::I(-2)]).unwrap(),
            2.0,
        ),
        (
            DType::I32,
            TensorData::from_scalars([], DType::I32, [Scalar::I(-2)]).unwrap(),
            2.0,
        ),
        (
            DType::U8,
            TensorData::from_scalars([], DType::U8, [Scalar::U(2)]).unwrap(),
            2.0,
        ),
        (
            DType::U16,
            TensorData::from_scalars([], DType::U16, [Scalar::U(2)]).unwrap(),
            2.0,
        ),
        (
            DType::U32,
            TensorData::from_scalars([], DType::U32, [Scalar::U(2)]).unwrap(),
            2.0,
        ),
        (
            DType::I64,
            TensorData::from_scalars([], DType::I64, [Scalar::I(-2)]).unwrap(),
            2.0,
        ),
        (
            DType::U64,
            TensorData::from_scalars([], DType::U64, [Scalar::U(2)]).unwrap(),
            2.0,
        ),
    ] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [], dtype);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        lower(
            &mut graph,
            Msg::new(&node("ReduceL2", &["x"], "out")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        let output = CpuBackend
            .execute(&graph, values["out"], &HashMap::from([("x".into(), data)]))
            .unwrap();
        assert_eq!(output.dtype(), dtype, "{dtype:?}");
        assert_eq!(output.values(), &[expected], "{dtype:?}");
    }

    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [], DType::I8);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("ReduceL2", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::from_scalars([], DType::I8, [Scalar::I(-128)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::I8);
    assert_eq!(output.values(), &[0.]);

    let axes = TensorData::from_scalars([1], DType::I64, [Scalar::I(1)]).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 0]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), axes)]);
    lower(
        &mut graph,
        Msg::new(&node("ReduceL2", &["x", "axes"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([("x".into(), TensorData::new([2, 0], vec![]).unwrap())]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2]);
    assert_eq!(output.values(), &[0., 0.]);

    let mut unknown = node("ReduceL2", &["x"], "out");
    field(&mut unknown, 5, &int_attr("axis", 0));
    let mut bad_keep = node("ReduceL2", &["x"], "out");
    field(&mut bad_keep, 5, &int_attr("keepdims", 2));
    let mut bad_noop = node("ReduceL2", &["x"], "out");
    field(&mut bad_noop, 5, &int_attr("noop_with_empty_axes", 2));
    let duplicate_axes = TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(-2)]).unwrap();
    let rank_zero_axes = TensorData::from_scalars([], DType::I64, [Scalar::I(0)]).unwrap();
    let wrong_dtype_axes = TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap();
    for (invalid, axes) in [
        (node("ReduceL2", &[], "out"), None),
        (node("ReduceL2", &["x", "axes", "extra"], "out"), None),
        (unknown, None),
        (bad_keep, None),
        (bad_noop, None),
        (node("ReduceL2", &["x", "missing"], "out"), None),
        (node("ReduceL2", &["x", "axes"], "out"), Some(duplicate_axes)),
        (node("ReduceL2", &["x", "axes"], "out"), Some(rank_zero_axes)),
        (node("ReduceL2", &["x", "axes"], "out"), Some(wrong_dtype_axes)),
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [2, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = axes.map(|axes| BTreeMap::from([("axes".into(), axes)])).unwrap_or_default();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
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
        Msg::new(&node("ReduceL2", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn reduce_log_sum_matches_tinygrad_typed_sum_log2_ln2_and_preflights() {
    // The source definition is ReduceSum(...).log(), and tinygrad defines
    // log as log2 multiplied by a weak ln(2) literal at the concrete log
    // dtype.  This also covers the default all-axes form.
    let mut graph = Graph::new();
    let x = graph.input("x", [2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("ReduceLogSum", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([("x".into(), TensorData::new([2], vec![1., 3.]).unwrap())]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[]);
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values(), &[4.0f32.log2() * std::f32::consts::LN_2]);

    // Signed axes and keepdims use the same pure plan before the Sum node is
    // made, and retain the reduction's concrete output shape.
    let axes = TensorData::from_scalars([1], DType::I64, [Scalar::I(-1)]).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), axes)]);
    let mut keep = node("ReduceLogSum", &["x", "axes"], "out");
    field(&mut keep, 5, &int_attr("keepdims", 1));
    lower(&mut graph, Msg::new(&keep), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 2], vec![1., 3., 1., 7.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2, 1]);
    assert_eq!(
        output.values(),
        &[
            4.0f32.log2() * std::f32::consts::LN_2,
            8.0f32.log2() * std::f32::consts::LN_2,
        ]
    );

    // Empty axes with noop leaves the input unreduced, but still runs the
    // source log2*ln(2) composition: -0 -> -inf, negatives -> NaN.
    let empty_axes = TensorData::from_scalars([0], DType::I64, []).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [5]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), empty_axes)]);
    let mut noop = node("ReduceLogSum", &["x", "axes"], "out");
    field(&mut noop, 5, &int_attr("noop_with_empty_axes", 1));
    lower(&mut graph, Msg::new(&noop), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new(
                    [5],
                    vec![-0., -1., f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[5]);
    assert_eq!(output.values()[0], f32::NEG_INFINITY);
    assert!(output.values()[1].is_nan());
    assert!(output.values()[2].is_nan());
    assert_eq!(output.values()[3], f32::INFINITY);
    assert!(output.values()[4].is_nan());

    // Typed Sum determines the log2 width.  Narrow floats keep their storage
    // width after the typed Sum; all integer/Bool accumulator widths enter
    // log2 as F32.
    for (dtype, data, expected_dtype) in [
        (
            DType::F16,
            TensorData::from_scalars([], DType::F16, [Scalar::F(4.)]).unwrap(),
            DType::F16,
        ),
        (
            DType::BF16,
            TensorData::from_scalars([], DType::BF16, [Scalar::F(4.)]).unwrap(),
            DType::BF16,
        ),
        (
            DType::Bool,
            TensorData::from_scalars([], DType::Bool, [Scalar::Bool(true)]).unwrap(),
            DType::F32,
        ),
        (
            DType::I8,
            TensorData::from_scalars([], DType::I8, [Scalar::I(4)]).unwrap(),
            DType::F32,
        ),
        (
            DType::I16,
            TensorData::from_scalars([], DType::I16, [Scalar::I(4)]).unwrap(),
            DType::F32,
        ),
        (
            DType::I32,
            TensorData::from_scalars([], DType::I32, [Scalar::I(4)]).unwrap(),
            DType::F32,
        ),
        (
            DType::U8,
            TensorData::from_scalars([], DType::U8, [Scalar::U(4)]).unwrap(),
            DType::F32,
        ),
        (
            DType::U16,
            TensorData::from_scalars([], DType::U16, [Scalar::U(4)]).unwrap(),
            DType::F32,
        ),
        (
            DType::U32,
            TensorData::from_scalars([], DType::U32, [Scalar::U(4)]).unwrap(),
            DType::F32,
        ),
        (
            DType::I64,
            TensorData::from_scalars([], DType::I64, [Scalar::I(4)]).unwrap(),
            DType::F32,
        ),
        (
            DType::U64,
            TensorData::from_scalars([], DType::U64, [Scalar::U(4)]).unwrap(),
            DType::F32,
        ),
        (
            DType::F64,
            TensorData::from_scalars([], DType::F64, [Scalar::F(4.)]).unwrap(),
            DType::F64,
        ),
    ] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [], dtype);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        lower(
            &mut graph,
            Msg::new(&node("ReduceLogSum", &["x"], "out")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        let output = CpuBackend
            .execute(&graph, values["out"], &HashMap::from([("x".into(), data)]))
            .unwrap();
        assert_eq!(output.dtype(), expected_dtype, "{dtype:?}");
        assert!(output.values()[0].is_finite(), "{dtype:?}");
    }

    // Reducing an empty domain uses typed Sum's zero neutral element, so the
    // subsequent logarithm exposes -infinity at every retained position.
    let axes = TensorData::from_scalars([1], DType::I64, [Scalar::I(1)]).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 0]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), axes)]);
    lower(
        &mut graph,
        Msg::new(&node("ReduceLogSum", &["x", "axes"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([("x".into(), TensorData::new([2, 0], vec![]).unwrap())]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2]);
    assert_eq!(output.values(), &[f32::NEG_INFINITY, f32::NEG_INFINITY]);

    let mut unknown = node("ReduceLogSum", &["x"], "out");
    field(&mut unknown, 5, &int_attr("axis", 0));
    let mut bad_keep = node("ReduceLogSum", &["x"], "out");
    field(&mut bad_keep, 5, &int_attr("keepdims", 2));
    let mut bad_noop = node("ReduceLogSum", &["x"], "out");
    field(&mut bad_noop, 5, &int_attr("noop_with_empty_axes", 2));
    let duplicate_axes =
        TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(-2)]).unwrap();
    let rank_zero_axes = TensorData::from_scalars([], DType::I64, [Scalar::I(0)]).unwrap();
    let wrong_dtype_axes = TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap();
    for (invalid, axes) in [
        (node("ReduceLogSum", &[], "out"), None),
        (node("ReduceLogSum", &["x", "axes", "extra"], "out"), None),
        (unknown, None),
        (bad_keep, None),
        (bad_noop, None),
        (node("ReduceLogSum", &["x", "missing"], "out"), None),
        (node("ReduceLogSum", &["x", "axes"], "out"), Some(duplicate_axes)),
        (node("ReduceLogSum", &["x", "axes"], "out"), Some(rank_zero_axes)),
        (node("ReduceLogSum", &["x", "axes"], "out"), Some(wrong_dtype_axes)),
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [2, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = axes
            .map(|axes| BTreeMap::from([("axes".into(), axes)]))
            .unwrap_or_default();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
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
        Msg::new(&node("ReduceLogSum", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn reduce_log_sum_exp_matches_tinygrad_direct_exp_sum_log_and_preflights() {
    // This ONNX dispatcher path is deliberately not Tensor.logsumexp: it is
    // direct exp, typed Sum, then Tensor.log's log2*ln(2) composition.
    let mut graph = Graph::new();
    let x = graph.input("x", [2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("ReduceLogSumExp", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2], vec![0., std::f32::consts::LN_2]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[]);
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(
        output.values(),
        &[(1.0f32 + std::f32::consts::LN_2.exp()).log2() * std::f32::consts::LN_2]
    );

    // Signed axes and keepdims are frozen by the shared ReducePlan before
    // the first exp node is appended.
    let axes = TensorData::from_scalars([1], DType::I64, [Scalar::I(-1)]).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 2]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), axes)]);
    let mut keep = node("ReduceLogSumExp", &["x", "axes"], "out");
    field(&mut keep, 5, &int_attr("keepdims", 1));
    lower(&mut graph, Msg::new(&keep), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new([2, 2], vec![0., std::f32::consts::LN_2, 0., 0.]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2, 1]);
    assert_eq!(
        output.values(),
        &[
            (1.0f32 + std::f32::consts::LN_2.exp()).log2() * std::f32::consts::LN_2,
            2.0f32.log2() * std::f32::consts::LN_2,
        ]
    );

    // Empty noop axes still execute exp then log; they are not an identity.
    // This exposes exp underflow, both infinities, and NaN propagation.
    let empty_axes = TensorData::from_scalars([0], DType::I64, []).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [5]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), empty_axes)]);
    let mut noop = node("ReduceLogSumExp", &["x", "axes"], "out");
    field(&mut noop, 5, &int_attr("noop_with_empty_axes", 1));
    lower(&mut graph, Msg::new(&noop), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([(
                "x".into(),
                TensorData::new(
                    [5],
                    vec![-1000., 0., f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[5]);
    assert_eq!(output.values()[0], f32::NEG_INFINITY);
    assert_eq!(output.values()[1].to_bits(), 0.0f32.to_bits());
    assert!(output.values()[2].is_nan());
    assert_eq!(output.values()[3], f32::INFINITY);
    assert_eq!(output.values()[4], f32::NEG_INFINITY);

    // This is intentionally the direct, overflow-sensitive dispatcher form,
    // rather than the finite stable-max result of Graph::logsumexp.
    let mut graph = Graph::new();
    let x = graph.input("x", []);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("ReduceLogSumExp", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([("x".into(), TensorData::scalar(1000.))]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[]);
    assert_eq!(output.values(), &[f32::INFINITY]);

    // Exp sets the calculation width: narrow floats return their post-exp
    // storage width before typed Sum narrows again, while every integer/Bool
    // input promotes to F32 before both Sum and log.
    for (dtype, data, expected_dtype) in [
        (
            DType::F16,
            TensorData::from_scalars([], DType::F16, [Scalar::F(0.)]).unwrap(),
            DType::F16,
        ),
        (
            DType::BF16,
            TensorData::from_scalars([], DType::BF16, [Scalar::F(0.)]).unwrap(),
            DType::BF16,
        ),
        (
            DType::Bool,
            TensorData::from_scalars([], DType::Bool, [Scalar::Bool(false)]).unwrap(),
            DType::F32,
        ),
        (
            DType::I8,
            TensorData::from_scalars([], DType::I8, [Scalar::I(0)]).unwrap(),
            DType::F32,
        ),
        (
            DType::I16,
            TensorData::from_scalars([], DType::I16, [Scalar::I(0)]).unwrap(),
            DType::F32,
        ),
        (
            DType::I32,
            TensorData::from_scalars([], DType::I32, [Scalar::I(0)]).unwrap(),
            DType::F32,
        ),
        (
            DType::I64,
            TensorData::from_scalars([], DType::I64, [Scalar::I(0)]).unwrap(),
            DType::F32,
        ),
        (
            DType::U8,
            TensorData::from_scalars([], DType::U8, [Scalar::U(0)]).unwrap(),
            DType::F32,
        ),
        (
            DType::U16,
            TensorData::from_scalars([], DType::U16, [Scalar::U(0)]).unwrap(),
            DType::F32,
        ),
        (
            DType::U32,
            TensorData::from_scalars([], DType::U32, [Scalar::U(0)]).unwrap(),
            DType::F32,
        ),
        (
            DType::U64,
            TensorData::from_scalars([], DType::U64, [Scalar::U(0)]).unwrap(),
            DType::F32,
        ),
        (
            DType::F64,
            TensorData::from_scalars([], DType::F64, [Scalar::F(0.)]).unwrap(),
            DType::F64,
        ),
    ] {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [], dtype);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = BTreeMap::new();
        lower(
            &mut graph,
            Msg::new(&node("ReduceLogSumExp", &["x"], "out")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        let output = CpuBackend
            .execute(&graph, values["out"], &HashMap::from([("x".into(), data)]))
            .unwrap();
        assert_eq!(output.dtype(), expected_dtype, "{dtype:?}");
        assert_eq!(output.values()[0].to_bits(), 0.0f32.to_bits(), "{dtype:?}");
    }

    // Empty reduced domains use Sum's zero neutral value after the elementwise
    // exp, which the final log exposes as -infinity.
    let axes = TensorData::from_scalars([1], DType::I64, [Scalar::I(1)]).unwrap();
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 0]);
    let mut values = BTreeMap::from([("x".into(), x)]);
    let mut constants = BTreeMap::from([("axes".into(), axes)]);
    lower(
        &mut graph,
        Msg::new(&node("ReduceLogSumExp", &["x", "axes"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([("x".into(), TensorData::new([2, 0], vec![]).unwrap())]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[2]);
    assert_eq!(output.values(), &[f32::NEG_INFINITY, f32::NEG_INFINITY]);

    let mut unknown = node("ReduceLogSumExp", &["x"], "out");
    field(&mut unknown, 5, &int_attr("axis", 0));
    let mut bad_keep = node("ReduceLogSumExp", &["x"], "out");
    field(&mut bad_keep, 5, &int_attr("keepdims", 2));
    let mut bad_noop = node("ReduceLogSumExp", &["x"], "out");
    field(&mut bad_noop, 5, &int_attr("noop_with_empty_axes", 2));
    let duplicate_axes =
        TensorData::from_scalars([2], DType::I64, [Scalar::I(0), Scalar::I(-2)]).unwrap();
    let rank_zero_axes = TensorData::from_scalars([], DType::I64, [Scalar::I(0)]).unwrap();
    let wrong_dtype_axes = TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap();
    for (invalid, axes) in [
        (node("ReduceLogSumExp", &[], "out"), None),
        (node("ReduceLogSumExp", &["x", "axes", "extra"], "out"), None),
        (unknown, None),
        (bad_keep, None),
        (bad_noop, None),
        (node("ReduceLogSumExp", &["x", "missing"], "out"), None),
        (node("ReduceLogSumExp", &["x", "axes"], "out"), Some(duplicate_axes)),
        (node("ReduceLogSumExp", &["x", "axes"], "out"), Some(rank_zero_axes)),
        (node("ReduceLogSumExp", &["x", "axes"], "out"), Some(wrong_dtype_axes)),
    ] {
        let mut malformed = Graph::new();
        let x = malformed.input("x", [2, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut constants = axes
            .map(|axes| BTreeMap::from([("axes".into(), axes)]))
            .unwrap_or_default();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
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
        Msg::new(&node("ReduceLogSumExp", &["x"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
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
fn ceil_preserves_graph_unary_semantics_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [4]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Ceil", &["input"], "out")),
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
    assert_eq!(output.values()[0], -1.0);
    assert_eq!(output.values()[1].to_bits(), (-0.0f32).to_bits());
    assert!(output.values()[2].is_infinite() && output.values()[2].is_sign_positive());
    assert!(output.values()[3].is_infinite() && output.values()[3].is_sign_negative());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [2], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Ceil", &["input"], "out")),
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

    let mut attribute = node("Ceil", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    let mut multiple_outputs = node("Ceil", &["input"], "out");
    text(&mut multiple_outputs, 2, "other");
    for invalid in [node("Ceil", &[], "out"), multiple_outputs, attribute] {
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
        Msg::new(&node("Ceil", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn sign_matches_tinygrad_special_values_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [7]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Sign", &["input"], "out")),
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
                    [7],
                    vec![-2.0, 3.0, -0.0, 0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values()[0], -1.0);
    assert_eq!(output.values()[1], 1.0);
    assert_eq!(output.values()[2].to_bits(), 0.0f32.to_bits());
    assert_eq!(output.values()[3].to_bits(), 0.0f32.to_bits());
    assert_eq!(output.values()[4], 1.0);
    assert_eq!(output.values()[5], 1.0);
    assert_eq!(output.values()[6], -1.0);

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [3], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Sign", &["input"], "out")),
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
                TensorData::from_scalars(
                    [3],
                    DType::I64,
                    [Scalar::I(-3), Scalar::I(0), Scalar::I(4)],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::I64);
    assert_eq!(output.scalar_at(0).as_i64(), -1);
    assert_eq!(output.scalar_at(1).as_i64(), 0);
    assert_eq!(output.scalar_at(2).as_i64(), 1);

    let mut attribute = node("Sign", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    let mut multiple_outputs = node("Sign", &["input"], "out");
    text(&mut multiple_outputs, 2, "other");
    for invalid in [node("Sign", &[], "out"), multiple_outputs, attribute] {
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
        Msg::new(&node("Sign", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn round_matches_ties_to_even_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [10]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Round", &["input"], "out")),
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
                    [10],
                    vec![
                        0.5,
                        1.5,
                        2.5,
                        -1.5,
                        -2.5,
                        1.25,
                        -1.25,
                        -0.0,
                        f32::INFINITY,
                        f32::NAN,
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values()[0], 0.0);
    assert_eq!(output.values()[1], 2.0);
    assert_eq!(output.values()[2], 2.0);
    assert_eq!(output.values()[3], -2.0);
    assert_eq!(output.values()[4], -2.0);
    assert_eq!(output.values()[5], 1.0);
    assert_eq!(output.values()[6], -1.0);
    assert_eq!(output.values()[7].to_bits(), (-0.0f32).to_bits());
    assert!(output.values()[8].is_infinite() && output.values()[8].is_sign_positive());
    assert!(output.values()[9].is_nan());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [2], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Round", &["input"], "out")),
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

    let mut attribute = node("Round", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    let mut multiple_outputs = node("Round", &["input"], "out");
    text(&mut multiple_outputs, 2, "other");
    for invalid in [node("Round", &[], "out"), multiple_outputs, attribute] {
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
        Msg::new(&node("Round", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn erf_preserves_graph_special_values_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [7]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Erf", &["input"], "out")),
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
                    [7],
                    vec![0.0, -0.0, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    // The deterministic A&S approximation has a bounded residual at zero;
    // its sign still follows the input zero sign.
    assert!(output.values()[0].abs() < 1e-6 && output.values()[0].is_sign_positive());
    assert!(output.values()[1].abs() < 1e-6 && output.values()[1].is_sign_negative());
    assert!((output.values()[2] - 0.8427008).abs() < 1e-5);
    assert!((output.values()[3] + 0.8427008).abs() < 1e-5);
    assert_eq!(output.values()[4], 1.0);
    assert_eq!(output.values()[5], -1.0);
    assert!(output.values()[6].is_nan());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Erf", &["input"], "out")),
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
    assert!((output.values()[0] - 0.8427008).abs() < 1e-5);

    let mut attribute = node("Erf", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    let mut multiple_outputs = node("Erf", &["input"], "out");
    text(&mut multiple_outputs, 2, "other");
    for invalid in [node("Erf", &[], "out"), multiple_outputs, attribute] {
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
        Msg::new(&node("Erf", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn sinh_preserves_graph_special_values_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [7]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Sinh", &["input"], "out")),
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
                    [7],
                    vec![1.0, -1.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert!((output.values()[0] - 1.1752012).abs() < 1e-6);
    assert!((output.values()[1] + 1.1752012).abs() < 1e-6);
    assert_eq!(output.values()[2].to_bits(), (-0.0f32).to_bits());
    assert!(output.values()[3].is_infinite() && output.values()[3].is_sign_positive());
    assert!(output.values()[4].is_infinite() && output.values()[4].is_sign_negative());
    assert!(output.values()[5].is_nan());
    assert_eq!(output.values()[6].to_bits(), 0.0f32.to_bits());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Sinh", &["input"], "out")),
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
    assert!((output.values()[0] - 1.1752012).abs() < 1e-6);

    let mut attribute = node("Sinh", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    let mut multiple_outputs = node("Sinh", &["input"], "out");
    text(&mut multiple_outputs, 2, "other");
    for invalid in [node("Sinh", &[], "out"), multiple_outputs, attribute] {
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
        Msg::new(&node("Sinh", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn cosh_preserves_graph_special_values_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [7]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Cosh", &["input"], "out")),
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
                    [7],
                    vec![1.0, -1.0, -0.0, 0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert!((output.values()[0] - 1.5430806).abs() < 1e-6);
    assert!((output.values()[1] - 1.5430806).abs() < 1e-6);
    assert_eq!(output.values()[2].to_bits(), 1.0f32.to_bits());
    assert_eq!(output.values()[3].to_bits(), 1.0f32.to_bits());
    assert!(output.values()[4].is_infinite() && output.values()[4].is_sign_positive());
    assert!(output.values()[5].is_infinite() && output.values()[5].is_sign_positive());
    assert!(output.values()[6].is_nan());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Cosh", &["input"], "out")),
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
    assert!((output.values()[0] - 1.5430806).abs() < 1e-6);

    let mut attribute = node("Cosh", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    let mut multiple_outputs = node("Cosh", &["input"], "out");
    text(&mut multiple_outputs, 2, "other");
    for invalid in [node("Cosh", &[], "out"), multiple_outputs, attribute] {
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
        Msg::new(&node("Cosh", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn asinh_preserves_graph_special_values_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [7]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Asinh", &["input"], "out")),
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
                    [7],
                    vec![1.0, -1.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert!((output.values()[0] - 0.8813736).abs() < 1e-6);
    assert!((output.values()[1] + 0.8813736).abs() < 1e-6);
    assert_eq!(output.values()[2].to_bits(), (-0.0f32).to_bits());
    assert!(output.values()[3].is_infinite() && output.values()[3].is_sign_positive());
    assert!(output.values()[4].is_infinite() && output.values()[4].is_sign_negative());
    assert!(output.values()[5].is_nan());
    assert_eq!(output.values()[6].to_bits(), 0.0f32.to_bits());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Asinh", &["input"], "out")),
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
    assert!((output.values()[0] - 0.8813736).abs() < 1e-6);

    let mut attribute = node("Asinh", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    let mut multiple_outputs = node("Asinh", &["input"], "out");
    text(&mut multiple_outputs, 2, "other");
    for invalid in [node("Asinh", &[], "out"), multiple_outputs, attribute] {
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
        Msg::new(&node("Asinh", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn acosh_preserves_graph_domain_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [7]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Acosh", &["input"], "out")),
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
                TensorData::new([7], vec![1.0, 2.0, 4.0, 0.5, -1.0, f32::INFINITY, f32::NAN])
                    .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values()[0], 0.0);
    assert!((output.values()[1] - 1.3169579).abs() < 1e-6);
    assert!((output.values()[2] - 2.063437).abs() < 1e-6);
    assert!(output.values()[3].is_nan());
    assert!(output.values()[4].is_nan());
    assert!(output.values()[5].is_infinite() && output.values()[5].is_sign_positive());
    assert!(output.values()[6].is_nan());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Acosh", &["input"], "out")),
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
                TensorData::from_scalars([1], DType::I64, [Scalar::I(2)]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert!((output.values()[0] - 1.3169579).abs() < 1e-6);

    let mut attribute = node("Acosh", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    let mut multiple_outputs = node("Acosh", &["input"], "out");
    text(&mut multiple_outputs, 2, "other");
    for invalid in [node("Acosh", &[], "out"), multiple_outputs, attribute] {
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
        Msg::new(&node("Acosh", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn atanh_preserves_graph_domain_and_preflights_before_publication() {
    let mut graph = Graph::new();
    let input = graph.input("input", [7]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Atanh", &["input"], "out")),
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
                TensorData::new([7], vec![0.5, -0.5, -0.0, 1.0, -1.0, 2.0, f32::NAN])
                    .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert!((output.values()[0] - 0.54930615).abs() < 1e-6);
    assert!((output.values()[1] + 0.54930615).abs() < 1e-6);
    assert_eq!(output.values()[2].to_bits(), (-0.0f32).to_bits());
    assert!(output.values()[3].is_infinite() && output.values()[3].is_sign_positive());
    assert!(output.values()[4].is_infinite() && output.values()[4].is_sign_negative());
    assert!(output.values()[5].is_nan());
    assert!(output.values()[6].is_nan());

    let mut integer = Graph::new();
    let input = integer.input_dtype("input", [1], DType::I64);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut integer,
        Msg::new(&node("Atanh", &["input"], "out")),
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
    assert_eq!(output.values()[0].to_bits(), 0.0f32.to_bits());

    let mut attribute = node("Atanh", &["input"], "out");
    field(&mut attribute, 5, &int_attr("unused", 1));
    let mut multiple_outputs = node("Atanh", &["input"], "out");
    text(&mut multiple_outputs, 2, "other");
    for invalid in [node("Atanh", &[], "out"), multiple_outputs, attribute] {
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
        Msg::new(&node("Atanh", &["input"], "out")),
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
fn hard_sigmoid_uses_typed_float_attributes_and_strict_select_clamping() {
    let mut graph = Graph::new();
    let input = graph.input("input", [6]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&node("HardSigmoid", &["input"], "out")), &mut values, &mut constants).unwrap();
    let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([(
        "input".into(), TensorData::new([6], vec![-3., 0., 3., f32::NAN, f32::INFINITY, f32::NEG_INFINITY]).unwrap(),
    )])).unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(&output.values()[0..3], &[0., 0.5, 1.]);
    assert!(output.values()[3].is_nan());
    assert_eq!(&output.values()[4..], &[1., 0.]);

    let mut custom = node("HardSigmoid", &["input"], "out");
    field(&mut custom, 5, &float_attr("alpha", 0.25));
    field(&mut custom, 5, &float_attr("beta", 0.25));
    let mut graph = Graph::new();
    let input = graph.input("input", [2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&custom), &mut values, &mut constants).unwrap();
    let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([(
        "input".into(), TensorData::new([2], vec![0., 2.]).unwrap(),
    )])).unwrap();
    assert_eq!(output.values(), &[0.25, 0.75]);

    for (dtype, data) in [
        (DType::I32, TensorData::from_scalars([], DType::I32, [Scalar::I(0)]).unwrap()),
        (DType::F16, TensorData::from_scalars([], DType::F16, [Scalar::F(0.)]).unwrap()),
        (DType::BF16, TensorData::from_scalars([], DType::BF16, [Scalar::F(0.)]).unwrap()),
        (DType::F32, TensorData::from_scalars([], DType::F32, [Scalar::F(0.)]).unwrap()),
        (DType::F64, TensorData::from_scalars([], DType::F64, [Scalar::F(0.)]).unwrap()),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [], dtype);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        lower(&mut graph, Msg::new(&node("HardSigmoid", &["input"], "out")), &mut values, &mut constants).unwrap();
        let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([("input".into(), data)])).unwrap();
        assert_eq!(output.shape().dims(), &[]);
        assert_eq!(output.dtype(), if dtype.is_float() { dtype } else { DType::F32 });
    }

    let mut ties = node("HardSigmoid", &["input"], "out");
    field(&mut ties, 5, &float_attr("alpha", 1.));
    field(&mut ties, 5, &float_attr("beta", 0.));
    let mut graph = Graph::new();
    let input = graph.input("input", [3]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&ties), &mut values, &mut constants).unwrap();
    let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([(
        "input".into(), TensorData::new([3], vec![0., 1., -1.]).unwrap(),
    )])).unwrap();
    assert_eq!(output.values(), &[0., 1., 0.]);

    let mut signed_zero = node("HardSigmoid", &["input"], "out");
    field(&mut signed_zero, 5, &float_attr("alpha", 0.));
    field(&mut signed_zero, 5, &float_attr("beta", -0.));
    let mut graph = Graph::new();
    let input = graph.input("input", []);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&signed_zero), &mut values, &mut constants).unwrap();
    let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([(
        "input".into(), TensorData::scalar(-1.),
    )])).unwrap();
    assert_eq!(output.values()[0].to_bits(), (-0.0f32).to_bits());

    let mut nan_attr = node("HardSigmoid", &["input"], "out");
    field(&mut nan_attr, 5, &float_attr("alpha", f32::NAN));
    let mut graph = Graph::new();
    let input = graph.input("input", []);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&nan_attr), &mut values, &mut constants).unwrap();
    let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([(
        "input".into(), TensorData::scalar(0.),
    )])).unwrap();
    assert!(output.values()[0].is_nan());

    for (beta, expected) in [(f32::INFINITY, 1.), (f32::NEG_INFINITY, 0.)] {
        let mut infinite_attr = node("HardSigmoid", &["input"], "out");
        field(&mut infinite_attr, 5, &float_attr("alpha", 0.));
        field(&mut infinite_attr, 5, &float_attr("beta", beta));
        let mut graph = Graph::new();
        let input = graph.input("input", []);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        lower(&mut graph, Msg::new(&infinite_attr), &mut values, &mut constants).unwrap();
        let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([(
            "input".into(), TensorData::scalar(0.),
        )])).unwrap();
        assert_eq!(output.values(), &[expected]);
    }

    let mut multiple_outputs = node("HardSigmoid", &["input"], "out");
    text(&mut multiple_outputs, 2, "extra");
    let mut wrong_int = node("HardSigmoid", &["input"], "out");
    field(&mut wrong_int, 5, &int_attr("alpha", 1));
    let mut wrong_string = node("HardSigmoid", &["input"], "out");
    field(&mut wrong_string, 5, &string_attr("beta", "bad"));
    let mut wrong_tensor = node("HardSigmoid", &["input"], "out");
    field(&mut wrong_tensor, 5, &tensor_attr("alpha", &tensor("", &[], &[1.])));
    let mut duplicate = node("HardSigmoid", &["input"], "out");
    field(&mut duplicate, 5, &float_attr("alpha", 0.2));
    field(&mut duplicate, 5, &float_attr("alpha", 0.3));
    let mut unknown = node("HardSigmoid", &["input"], "out");
    field(&mut unknown, 5, &float_attr("other", 1.));
    for invalid in [node("HardSigmoid", &[], "out"), multiple_outputs, wrong_int, wrong_string, wrong_tensor, duplicate, unknown] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
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
    assert!(lower(&mut overflow, Msg::new(&node("HardSigmoid", &["input"], "out")), &mut values, &mut constants).is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn shrink_activation_preserves_tinygrad_mask_products_and_preflights() {
    let mut graph = Graph::new();
    let input = graph.input("input", [8]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&node("Shrink", &["input"], "out")), &mut values, &mut constants).unwrap();
    let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([(
        "input".into(),
        TensorData::new(
            [8],
            vec![-2., -0.5, -0.0, 0.0, 0.5, 2., f32::NAN, f32::INFINITY],
        )
        .unwrap(),
    )]))
    .unwrap();
    assert_eq!(&output.values()[0..2], &[-2., 0.]);
    assert_eq!(output.values()[2].to_bits(), 0.0f32.to_bits());
    assert_eq!(output.values()[3].to_bits(), 0.0f32.to_bits());
    assert_eq!(&output.values()[4..6], &[0., 2.]);
    // The source is multiplication, not Select: NaN and infinity poison a
    // false-mask branch through IEEE 0 * NaN/infinity.
    assert!(output.values()[6].is_nan());
    assert!(output.values()[7].is_nan());

    let mut negative_lambda = node("Shrink", &["input"], "out");
    field(&mut negative_lambda, 5, &float_attr("bias", 0.25));
    field(&mut negative_lambda, 5, &float_attr("lambd", -1.0));
    let mut graph = Graph::new();
    let input = graph.input("input", [2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&negative_lambda), &mut values, &mut constants).unwrap();
    let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([(
        "input".into(),
        TensorData::new([2], vec![-0.5, 0.5]).unwrap(),
    )]))
    .unwrap();
    assert_eq!(output.values(), &[-1., 1.]);

    for (lambd, input_value, expected) in [
        (f32::INFINITY, 2., 0.),
        (f32::NEG_INFINITY, 2., 4.),
    ] {
        let mut special = node("Shrink", &["input"], "out");
        field(&mut special, 5, &float_attr("lambd", lambd));
        let mut graph = Graph::new();
        let input = graph.input("input", []);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        lower(&mut graph, Msg::new(&special), &mut values, &mut constants).unwrap();
        let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([(
            "input".into(), TensorData::scalar(input_value),
        )])).unwrap();
        assert_eq!(output.values(), &[expected]);
    }
    let mut nan_bias = node("Shrink", &["input"], "out");
    field(&mut nan_bias, 5, &float_attr("bias", f32::NAN));
    let mut graph = Graph::new();
    let input = graph.input("input", []);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&nan_bias), &mut values, &mut constants).unwrap();
    let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([(
        "input".into(), TensorData::scalar(0.),
    )])).unwrap();
    assert!(output.values()[0].is_nan());

    for (dtype, data, expected_dtype) in [
        (DType::Bool, TensorData::from_scalars([], DType::Bool, [Scalar::Bool(true)]).unwrap(), DType::F32),
        (DType::I32, TensorData::from_scalars([], DType::I32, [Scalar::I(-2)]).unwrap(), DType::F32),
        (DType::U64, TensorData::from_scalars([], DType::U64, [Scalar::U(2)]).unwrap(), DType::F32),
        (DType::F16, TensorData::from_scalars([], DType::F16, [Scalar::F(2.)]).unwrap(), DType::F16),
        (DType::BF16, TensorData::from_scalars([], DType::BF16, [Scalar::F(2.)]).unwrap(), DType::BF16),
        (DType::F32, TensorData::from_scalars([], DType::F32, [Scalar::F(2.)]).unwrap(), DType::F32),
        (DType::F64, TensorData::from_scalars([], DType::F64, [Scalar::F(2.)]).unwrap(), DType::F64),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [], dtype);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        lower(&mut graph, Msg::new(&node("Shrink", &["input"], "out")), &mut values, &mut constants).unwrap();
        let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([("input".into(), data)])).unwrap();
        assert_eq!(output.shape().dims(), &[]);
        assert_eq!(output.dtype(), expected_dtype);
    }

    let mut graph = Graph::new();
    let input = graph.input("input", [0]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&node("Shrink", &["input"], "out")), &mut values, &mut constants).unwrap();
    let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([(
        "input".into(), TensorData::new([0], vec![]).unwrap(),
    )])).unwrap();
    assert_eq!(output.shape().dims(), &[0]);
    assert_eq!(output.dtype(), DType::F32);

    let mut multiple_outputs = node("Shrink", &["input"], "out");
    text(&mut multiple_outputs, 2, "extra");
    let mut wrong_int = node("Shrink", &["input"], "out");
    field(&mut wrong_int, 5, &int_attr("bias", 1));
    let mut wrong_string = node("Shrink", &["input"], "out");
    field(&mut wrong_string, 5, &string_attr("lambd", "bad"));
    let mut wrong_tensor = node("Shrink", &["input"], "out");
    field(&mut wrong_tensor, 5, &tensor_attr("bias", &tensor("", &[], &[1.])));
    let mut duplicate = node("Shrink", &["input"], "out");
    field(&mut duplicate, 5, &float_attr("bias", 0.));
    field(&mut duplicate, 5, &float_attr("bias", 1.));
    let mut unknown = node("Shrink", &["input"], "out");
    field(&mut unknown, 5, &float_attr("other", 1.));
    for invalid in [
        node("Shrink", &[], "out"),
        node("Shrink", &["input", "extra"], "out"),
        multiple_outputs,
        wrong_int,
        wrong_string,
        wrong_tensor,
        duplicate,
        unknown,
    ] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mut missing = Graph::new();
    let mut values = BTreeMap::new();
    let mut constants = BTreeMap::new();
    let before_nodes = missing.node_count();
    assert!(lower(&mut missing, Msg::new(&node("Shrink", &["missing"], "out")), &mut values, &mut constants).is_err());
    assert!(values.is_empty());
    assert!(constants.is_empty());
    assert_eq!(missing.node_count(), before_nodes);

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(&mut overflow, Msg::new(&node("Shrink", &["input"], "out")), &mut values, &mut constants).is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn thresholded_relu_matches_tinygrad_weak_scalars_and_preflights() {
    let mut graph = Graph::new();
    let input = graph.input("input", [7]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("ThresholdedRelu", &["input"], "out")),
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
                    [7],
                    vec![
                        -1.,
                        -0.,
                        1.,
                        2.,
                        f32::NAN,
                        f32::INFINITY,
                        f32::NEG_INFINITY,
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(&output.values()[0..4], &[0., 0., 0., 2.]);
    assert_eq!(output.values()[4].to_bits(), 0.0f32.to_bits());
    assert_eq!(output.values()[5], f32::INFINITY);
    assert_eq!(output.values()[6], 0.);

    let mut custom = node("ThresholdedRelu", &["input"], "out");
    field(&mut custom, 5, &float_attr("alpha", -1.));
    let mut graph = Graph::new();
    let input = graph.input("input", []);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&custom), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([("input".into(), TensorData::scalar(-0.0))]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[]);
    assert_eq!(output.values()[0].to_bits(), (-0.0f32).to_bits());

    for alpha in [f32::NAN, f32::INFINITY] {
        let mut attributed = node("ThresholdedRelu", &["input"], "out");
        field(&mut attributed, 5, &float_attr("alpha", alpha));
        let mut graph = Graph::new();
        let input = graph.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        lower(&mut graph, Msg::new(&attributed), &mut values, &mut constants).unwrap();
        let output = CpuBackend
            .execute(
                &graph,
                values["out"],
                &HashMap::from([(
                    "input".into(),
                    TensorData::new([2], vec![f32::NAN, f32::INFINITY]).unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(output.values(), &[0., 0.]);
    }
    let mut negative_infinite = node("ThresholdedRelu", &["input"], "out");
    field(
        &mut negative_infinite,
        5,
        &float_attr("alpha", f32::NEG_INFINITY),
    );
    let mut graph = Graph::new();
    let input = graph.input("input", [3]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&negative_infinite),
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
                TensorData::new([3], vec![f32::NEG_INFINITY, 0., f32::INFINITY]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.values(), &[0., 0., f32::INFINITY]);

    for (dtype, data, expected) in [
        (
            DType::I32,
            TensorData::from_scalars([], DType::I32, [Scalar::I(2)]).unwrap(),
            DType::I32,
        ),
        (
            DType::Bool,
            TensorData::from_scalars([], DType::Bool, [Scalar::Bool(true)]).unwrap(),
            DType::I32,
        ),
        (
            DType::F16,
            TensorData::from_scalars([], DType::F16, [Scalar::F(2.)]).unwrap(),
            DType::F16,
        ),
        (
            DType::BF16,
            TensorData::from_scalars([], DType::BF16, [Scalar::F(2.)]).unwrap(),
            DType::BF16,
        ),
        (
            DType::F32,
            TensorData::from_scalars([], DType::F32, [Scalar::F(2.)]).unwrap(),
            DType::F32,
        ),
        (
            DType::F64,
            TensorData::from_scalars([], DType::F64, [Scalar::F(2.)]).unwrap(),
            DType::F64,
        ),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [], dtype);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        lower(
            &mut graph,
            Msg::new(&node("ThresholdedRelu", &["input"], "out")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        let output = CpuBackend
            .execute(&graph, values["out"], &HashMap::from([("input".into(), data)]))
            .unwrap();
        assert_eq!(output.shape().dims(), &[]);
        assert_eq!(output.dtype(), expected);
    }

    let mut empty_graph = Graph::new();
    let input = empty_graph.input("input", [0]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut empty_graph,
        Msg::new(&node("ThresholdedRelu", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &empty_graph,
            values["out"],
            &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[0]);
    assert!(output.values().is_empty());

    let mut multiple_outputs = node("ThresholdedRelu", &["input"], "out");
    text(&mut multiple_outputs, 2, "extra");
    let mut wrong_int = node("ThresholdedRelu", &["input"], "out");
    field(&mut wrong_int, 5, &int_attr("alpha", 1));
    let mut wrong_string = node("ThresholdedRelu", &["input"], "out");
    field(&mut wrong_string, 5, &string_attr("alpha", "bad"));
    let mut wrong_tensor = node("ThresholdedRelu", &["input"], "out");
    field(
        &mut wrong_tensor,
        5,
        &tensor_attr("alpha", &tensor("", &[], &[1.])),
    );
    let mut duplicate = node("ThresholdedRelu", &["input"], "out");
    field(&mut duplicate, 5, &float_attr("alpha", 0.));
    field(&mut duplicate, 5, &float_attr("alpha", 1.));
    let mut unknown = node("ThresholdedRelu", &["input"], "out");
    field(&mut unknown, 5, &float_attr("other", 1.));
    for invalid in [
        node("ThresholdedRelu", &[], "out"),
        multiple_outputs,
        wrong_int,
        wrong_string,
        wrong_tensor,
        duplicate,
        unknown,
    ] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let missing = node("ThresholdedRelu", &["missing"], "out");
    let mut malformed = Graph::new();
    let input = malformed.input("input", [2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = malformed.node_count();
    assert!(lower(&mut malformed, Msg::new(&missing), &mut values, &mut constants).is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(malformed.node_count(), before_nodes);

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("ThresholdedRelu", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn binarizer_matches_tinygrad_strict_float_output_and_preflights() {
    let mut graph = Graph::new();
    let input = graph.input("input", [7]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&node("Binarizer", &["input"], "out")),
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
                    [7],
                    vec![
                        -1.,
                        -0.,
                        0.,
                        1.,
                        f32::NAN,
                        f32::INFINITY,
                        f32::NEG_INFINITY,
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(&output.values()[0..5], &[0., 0., 0., 1., 0.]);
    assert_eq!(output.values()[1].to_bits(), 0.0f32.to_bits());
    assert_eq!(output.values()[2].to_bits(), 0.0f32.to_bits());
    assert_eq!(output.values()[5], 1.);
    assert_eq!(output.values()[6], 0.);

    let mut custom = node("Binarizer", &["input"], "out");
    field(&mut custom, 5, &float_attr("threshold", -0.5));
    let mut graph = Graph::new();
    let input = graph.input("input", []);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&custom), &mut values, &mut constants).unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            values["out"],
            &HashMap::from([("input".into(), TensorData::scalar(-0.0))]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[]);
    assert_eq!(output.values(), &[1.]);

    for threshold in [f32::NAN, f32::INFINITY] {
        let mut attributed = node("Binarizer", &["input"], "out");
        field(&mut attributed, 5, &float_attr("threshold", threshold));
        let mut graph = Graph::new();
        let input = graph.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        lower(&mut graph, Msg::new(&attributed), &mut values, &mut constants).unwrap();
        let output = CpuBackend
            .execute(
                &graph,
                values["out"],
                &HashMap::from([(
                    "input".into(),
                    TensorData::new([2], vec![f32::NAN, f32::INFINITY]).unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(output.values(), &[0., 0.]);
    }
    let mut negative_infinite = node("Binarizer", &["input"], "out");
    field(
        &mut negative_infinite,
        5,
        &float_attr("threshold", f32::NEG_INFINITY),
    );
    let mut graph = Graph::new();
    let input = graph.input("input", [3]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut graph,
        Msg::new(&negative_infinite),
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
                TensorData::new([3], vec![f32::NEG_INFINITY, 0., f32::INFINITY]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(output.values(), &[0., 1., 1.]);

    for (dtype, data) in [
        (
            DType::I32,
            TensorData::from_scalars([], DType::I32, [Scalar::I(1)]).unwrap(),
        ),
        (
            DType::U64,
            TensorData::from_scalars([], DType::U64, [Scalar::U(1)]).unwrap(),
        ),
        (
            DType::Bool,
            TensorData::from_scalars([], DType::Bool, [Scalar::Bool(true)]).unwrap(),
        ),
        (
            DType::F16,
            TensorData::from_scalars([], DType::F16, [Scalar::F(1.)]).unwrap(),
        ),
        (
            DType::BF16,
            TensorData::from_scalars([], DType::BF16, [Scalar::F(1.)]).unwrap(),
        ),
        (
            DType::F32,
            TensorData::from_scalars([], DType::F32, [Scalar::F(1.)]).unwrap(),
        ),
        (
            DType::F64,
            TensorData::from_scalars([], DType::F64, [Scalar::F(1.)]).unwrap(),
        ),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [], dtype);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        lower(
            &mut graph,
            Msg::new(&node("Binarizer", &["input"], "out")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        let output = CpuBackend
            .execute(&graph, values["out"], &HashMap::from([("input".into(), data)]))
            .unwrap();
        assert_eq!(output.shape().dims(), &[]);
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.values(), &[1.]);
    }

    let mut empty_graph = Graph::new();
    let input = empty_graph.input("input", [0]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    lower(
        &mut empty_graph,
        Msg::new(&node("Binarizer", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &empty_graph,
            values["out"],
            &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[0]);
    assert_eq!(output.dtype(), DType::F32);
    assert!(output.values().is_empty());

    let mut multiple_outputs = node("Binarizer", &["input"], "out");
    text(&mut multiple_outputs, 2, "extra");
    let mut wrong_int = node("Binarizer", &["input"], "out");
    field(&mut wrong_int, 5, &int_attr("threshold", 1));
    let mut wrong_string = node("Binarizer", &["input"], "out");
    field(&mut wrong_string, 5, &string_attr("threshold", "bad"));
    let mut wrong_tensor = node("Binarizer", &["input"], "out");
    field(
        &mut wrong_tensor,
        5,
        &tensor_attr("threshold", &tensor("", &[], &[1.])),
    );
    let mut duplicate = node("Binarizer", &["input"], "out");
    field(&mut duplicate, 5, &float_attr("threshold", 0.));
    field(&mut duplicate, 5, &float_attr("threshold", 1.));
    let mut unknown = node("Binarizer", &["input"], "out");
    field(&mut unknown, 5, &float_attr("other", 1.));
    for invalid in [
        node("Binarizer", &[], "out"),
        multiple_outputs,
        wrong_int,
        wrong_string,
        wrong_tensor,
        duplicate,
        unknown,
    ] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let mut values = BTreeMap::from([("input".into(), input)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let missing = node("Binarizer", &["missing"], "out");
    let mut malformed = Graph::new();
    let input = malformed.input("input", [2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = malformed.node_count();
    assert!(lower(&mut malformed, Msg::new(&missing), &mut values, &mut constants).is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(malformed.node_count(), before_nodes);

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let mut values = BTreeMap::from([("input".into(), input)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(
        &mut overflow,
        Msg::new(&node("Binarizer", &["input"], "out")),
        &mut values,
        &mut constants,
    )
    .is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(overflow.node_count(), before_nodes);
}

#[test]
fn prelu_matches_tinygrad_strict_branching_and_preflights() {
    let mut graph = Graph::new();
    let input = graph.input("input", [7]);
    let slope = graph.input("slope", []);
    let mut values = BTreeMap::from([("input".into(), input), ("slope".into(), slope)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&node("PRelu", &["input", "slope"], "out")), &mut values, &mut constants).unwrap();
    let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([
        ("input".into(), TensorData::new([7], vec![-1., -0., 0., 1., f32::NAN, f32::INFINITY, f32::NEG_INFINITY]).unwrap()),
        ("slope".into(), TensorData::scalar(2.)),
    ])).unwrap();
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.values()[0], -2.);
    assert_eq!(output.values()[1].to_bits(), (-0.0f32).to_bits());
    assert_eq!(output.values()[2].to_bits(), 0.0f32.to_bits());
    assert_eq!(output.values()[3], 1.);
    assert!(output.values()[4].is_nan());
    assert_eq!(output.values()[5], f32::INFINITY);
    assert_eq!(output.values()[6], f32::NEG_INFINITY);

    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3]);
    let slope = graph.input("slope", [3]);
    let mut values = BTreeMap::from([("input".into(), input), ("slope".into(), slope)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&node("PRelu", &["input", "slope"], "out")), &mut values, &mut constants).unwrap();
    let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([
        ("input".into(), TensorData::new([2, 3], vec![-2., -2., -2., 1., 1., 1.]).unwrap()),
        ("slope".into(), TensorData::new([3], vec![0., 0.5, 1.]).unwrap()),
    ])).unwrap();
    assert_eq!(output.values(), &[0., -1., -2., 1., 1., 1.]);

    let mut graph = Graph::new();
    let input = graph.input("input", [2, 1, 3]);
    let slope = graph.input("slope", [1, 4, 1]);
    let mut values = BTreeMap::from([("input".into(), input), ("slope".into(), slope)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&node("PRelu", &["input", "slope"], "out")), &mut values, &mut constants).unwrap();
    assert_eq!(graph.shape(values["out"]).unwrap().dims(), &[2, 4, 3]);

    for (x_dtype, slope_dtype, x_data, slope_data, expected) in [
        (DType::I32, DType::F32, TensorData::from_scalars([], DType::I32, [Scalar::I(-2)]).unwrap(), TensorData::scalar(0.5), DType::F32),
        (DType::F16, DType::F16, TensorData::from_scalars([], DType::F16, [Scalar::F(-2.)]).unwrap(), TensorData::from_scalars([], DType::F16, [Scalar::F(0.5)]).unwrap(), DType::F16),
        (DType::BF16, DType::BF16, TensorData::from_scalars([], DType::BF16, [Scalar::F(-2.)]).unwrap(), TensorData::from_scalars([], DType::BF16, [Scalar::F(0.5)]).unwrap(), DType::BF16),
        (DType::U64, DType::I64, TensorData::from_scalars([], DType::U64, [Scalar::U(2)]).unwrap(), TensorData::from_scalars([], DType::I64, [Scalar::I(-1)]).unwrap(), DType::F32),
        (DType::I64, DType::U64, TensorData::from_scalars([], DType::I64, [Scalar::I(-2)]).unwrap(), TensorData::from_scalars([], DType::U64, [Scalar::U(1)]).unwrap(), DType::F32),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [], x_dtype);
        let slope = graph.input_dtype("slope", [], slope_dtype);
        let mut values = BTreeMap::from([("input".into(), input), ("slope".into(), slope)]);
        let mut constants = BTreeMap::new();
        lower(&mut graph, Msg::new(&node("PRelu", &["input", "slope"], "out")), &mut values, &mut constants).unwrap();
        let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([("input".into(), x_data), ("slope".into(), slope_data)])).unwrap();
        assert_eq!(output.dtype(), expected);
    }

    let mut graph = Graph::new();
    let input = graph.input("input", [2]);
    let slope = graph.input("slope", []);
    let mut values = BTreeMap::from([("input".into(), input), ("slope".into(), slope)]);
    let mut constants = BTreeMap::new();
    lower(&mut graph, Msg::new(&node("PRelu", &["input", "slope"], "out")), &mut values, &mut constants).unwrap();
    let output = CpuBackend.execute(&graph, values["out"], &HashMap::from([
        ("input".into(), TensorData::new([2], vec![1., -1.]).unwrap()),
        ("slope".into(), TensorData::scalar(f32::NAN)),
    ])).unwrap();
    assert_eq!(output.values()[0], 1.);
    assert!(output.values()[1].is_nan());

    let mut attributed = node("PRelu", &["input", "slope"], "out");
    field(&mut attributed, 5, &int_attr("unused", 1));
    let mut multiple_outputs = node("PRelu", &["input", "slope"], "out");
    text(&mut multiple_outputs, 2, "extra");
    for invalid in [
        node("PRelu", &["input"], "out"),
        node("PRelu", &["input", "missing"], "out"),
        attributed,
        multiple_outputs,
    ] {
        let mut malformed = Graph::new();
        let input = malformed.input("input", [2]);
        let slope = malformed.input("slope", [3]);
        let mut values = BTreeMap::from([("input".into(), input), ("slope".into(), slope)]);
        let mut constants = BTreeMap::new();
        let before_values = values.clone();
        let before_constants = constants.clone();
        let before_nodes = malformed.node_count();
        assert!(lower(&mut malformed, Msg::new(&invalid), &mut values, &mut constants).is_err());
        assert_eq!(values, before_values);
        assert_eq!(constants, before_constants);
        assert_eq!(malformed.node_count(), before_nodes);
    }

    let mismatch = node("PRelu", &["input", "slope"], "out");
    let mut malformed = Graph::new();
    let input = malformed.input("input", [2, 3]);
    let slope = malformed.input("slope", [2, 2]);
    let mut values = BTreeMap::from([("input".into(), input), ("slope".into(), slope)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = malformed.node_count();
    assert!(lower(&mut malformed, Msg::new(&mismatch), &mut values, &mut constants).is_err());
    assert_eq!(values, before_values);
    assert_eq!(constants, before_constants);
    assert_eq!(malformed.node_count(), before_nodes);

    let mut overflow = Graph::new();
    let input = overflow.input("input", [usize::MAX, 2]);
    let slope = overflow.input("slope", []);
    let mut values = BTreeMap::from([("input".into(), input), ("slope".into(), slope)]);
    let mut constants = BTreeMap::new();
    let before_values = values.clone();
    let before_constants = constants.clone();
    let before_nodes = overflow.node_count();
    assert!(lower(&mut overflow, Msg::new(&node("PRelu", &["input", "slope"], "out")), &mut values, &mut constants).is_err());
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
