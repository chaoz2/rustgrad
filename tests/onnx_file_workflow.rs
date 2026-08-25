//! Public local ONNX + named NPY CPU workflow acceptance.

use rustgrad::interop::host::{load_npy_file, save_npy_file};
use rustgrad::onnx::{
    NamedPaths, OnnxFileError, OnnxReadLimits, OnnxWorkflowError, OnnxWorkflowLimits,
    run_onnx_files,
};
use rustgrad::{DType, TensorData};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

fn dir() -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let p = std::env::temp_dir().join(format!(
        "rustgrad-onnx-file-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&p).unwrap();
    p
}
fn vi(mut n: u32, o: &mut Vec<u8>) {
    loop {
        let b = (n & 127) as u8;
        n >>= 7;
        o.push(if n == 0 { b } else { b | 128 });
        if n == 0 {
            return;
        }
    }
}
fn field(o: &mut Vec<u8>, id: u32, b: &[u8]) {
    vi(id << 3 | 2, o);
    vi(b.len() as u32, o);
    o.extend_from_slice(b)
}
fn var(o: &mut Vec<u8>, id: u32, n: u32) {
    vi(id << 3, o);
    vi(n, o)
}
fn text(o: &mut Vec<u8>, id: u32, s: &str) {
    field(o, id, s.as_bytes())
}
fn value(name: &str, dims: &[u32]) -> Vec<u8> {
    let mut shape = vec![];
    for &d in dims {
        let mut dim = vec![];
        var(&mut dim, 1, d);
        field(&mut shape, 1, &dim)
    }
    let mut tensor = vec![];
    var(&mut tensor, 1, 1);
    field(&mut tensor, 2, &shape);
    let mut ty = vec![];
    field(&mut ty, 1, &tensor);
    let mut v = vec![];
    text(&mut v, 1, name);
    field(&mut v, 2, &ty);
    v
}
fn tensor(name: &str, dims: &[u32], values: &[f32]) -> Vec<u8> {
    let mut t = vec![];
    let mut ds = vec![];
    for &d in dims {
        vi(d, &mut ds)
    }
    field(&mut t, 1, &ds);
    var(&mut t, 2, 1);
    text(&mut t, 8, name);
    field(
        &mut t,
        9,
        &values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    t
}
fn node(op: &str, ins: &[&str], out: &str) -> Vec<u8> {
    let mut n = vec![];
    for i in ins {
        text(&mut n, 1, i)
    }
    text(&mut n, 2, out);
    text(&mut n, 4, op);
    n
}
fn fixture() -> Vec<u8> {
    let mut g = vec![];
    field(&mut g, 11, &value("x", &[1, 2]));
    field(&mut g, 12, &value("y", &[1, 2]));
    field(&mut g, 5, &tensor("w", &[2, 2], &[1., 2., 3., 4.]));
    field(&mut g, 5, &tensor("b", &[1, 2], &[1., -10.]));
    for n in [
        node("MatMul", &["x", "w"], "m"),
        node("Add", &["m", "b"], "a"),
        node("Relu", &["a"], "y"),
    ] {
        field(&mut g, 1, &n)
    }
    let mut op = vec![];
    var(&mut op, 2, 13);
    let mut m = vec![];
    field(&mut m, 7, &g);
    field(&mut m, 8, &op);
    m
}
fn paths(items: &[(&str, PathBuf)]) -> NamedPaths {
    NamedPaths::new(items.iter().map(|(n, p)| (n.to_string(), p.clone()))).unwrap()
}

#[test]
fn local_model_named_npy_input_and_output_are_exact_and_repeatable() {
    let d = dir();
    let model = d.join("model.onnx");
    let input = d.join("x.npy");
    let output = d.join("y.npy");
    fs::write(&model, fixture()).unwrap();
    save_npy_file(
        &input,
        &TensorData::from_le_bytes([1, 2], DType::F32, &[0, 0, 0x80, 0x3f, 0, 0, 0, 0x40]).unwrap(),
    )
    .unwrap();
    let inputs = paths(&[("x", input)]);
    let outputs = paths(&[("y", output.clone())]);
    let first = run_onnx_files(&model, &inputs, &outputs, OnnxWorkflowLimits::default()).unwrap();
    assert_eq!(
        first["y"].to_le_bytes().unwrap(),
        vec![0, 0, 0, 0x41, 0, 0, 0, 0]
    );
    let bytes = fs::read(&output).unwrap();
    run_onnx_files(&model, &inputs, &outputs, OnnxWorkflowLimits::default()).unwrap();
    assert_eq!(bytes, fs::read(&output).unwrap());
    assert_eq!(
        load_npy_file(&output).unwrap().to_le_bytes().unwrap(),
        first["y"].to_le_bytes().unwrap()
    );
    fs::remove_dir_all(d).unwrap()
}

#[test]
fn workflow_rejects_limits_and_named_preflight_before_execution() {
    let d = dir();
    let model = d.join("model.onnx");
    let input = d.join("x.npy");
    fs::write(&model, fixture()).unwrap();
    fs::write(&input, b"not read for extra names").unwrap();
    let out = d.join("y.npy");
    assert!(matches!(
        run_onnx_files(
            &model,
            &paths(&[("extra", input.clone())]),
            &paths(&[("y", out.clone())]),
            OnnxWorkflowLimits::default()
        ),
        Err(OnnxWorkflowError::MissingInput(_))
    ));
    assert!(matches!(
        run_onnx_files(
            &model,
            &paths(&[("x", input)]),
            &paths(&[("missing", out)]),
            OnnxWorkflowLimits::default()
        ),
        Err(OnnxWorkflowError::UnknownOutput(_))
    ));
    assert!(matches!(
        run_onnx_files(
            &model,
            &NamedPaths::default(),
            &NamedPaths::default(),
            OnnxWorkflowLimits {
                onnx: OnnxReadLimits { max_model_bytes: 1 },
                ..OnnxWorkflowLimits::default()
            }
        ),
        Err(OnnxWorkflowError::Model(_))
    ));
    let truncated = d.join("truncated.onnx");
    fs::write(&truncated, &fixture()[..3]).unwrap();
    assert!(matches!(
        run_onnx_files(
            &truncated,
            &NamedPaths::default(),
            &NamedPaths::default(),
            OnnxWorkflowLimits::default()
        ),
        Err(OnnxWorkflowError::Model(OnnxFileError::Model(_)))
    ));
    let valid = d.join("valid.npy");
    save_npy_file(
        &valid,
        &TensorData::from_le_bytes([1, 2], DType::F32, &[0; 8]).unwrap(),
    )
    .unwrap();
    let wrong_shape = d.join("wrong-shape.npy");
    save_npy_file(
        &wrong_shape,
        &TensorData::from_le_bytes([2], DType::F32, &[0; 8]).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        run_onnx_files(
            &model,
            &paths(&[("x", wrong_shape)]),
            &paths(&[("y", d.join("shape.npy"))]),
            OnnxWorkflowLimits::default()
        ),
        Err(OnnxWorkflowError::Run(_))
    ));
    let wrong_dtype = d.join("wrong-dtype.npy");
    save_npy_file(
        &wrong_dtype,
        &TensorData::from_le_bytes([1, 2], DType::I32, &[0; 8]).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        run_onnx_files(
            &model,
            &paths(&[("x", wrong_dtype)]),
            &paths(&[("y", d.join("dtype.npy"))]),
            OnnxWorkflowLimits::default()
        ),
        Err(OnnxWorkflowError::Run(_))
    ));
    assert!(
        NamedPaths::new(vec![
            ("y".into(), d.join("first.npy")),
            ("y".into(), d.join("second.npy")),
        ])
        .is_err()
    );
    let output_directory = d.join("output-directory.npy");
    fs::create_dir(&output_directory).unwrap();
    assert!(matches!(
        run_onnx_files(
            &model,
            &paths(&[("x", valid)]),
            &paths(&[("y", output_directory)]),
            OnnxWorkflowLimits::default()
        ),
        Err(OnnxWorkflowError::Output { .. })
    ));
    fs::remove_dir_all(d).unwrap()
}
