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
    field(&mut bad, 5, &ints_attr("dilations", &[1, 1]));
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
