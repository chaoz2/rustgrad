//! Public strict safetensors-to-module CPU inference acceptance.

use rustgrad::nn::{Linear, ReLU, Sequential};
use rustgrad::{
    DType, Module, ModuleStateDict, TensorData, infer_module_cpu, save_safetensors_file,
};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

fn directory() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "rustgrad-strict-module-state-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&directory).unwrap();
    directory
}

fn linear(in_features: usize, out_features: usize) -> Linear {
    Linear::new_static(in_features, out_features, true, 7).unwrap()
}

fn f32(shape: impl Into<rustgrad::Shape>, values: &[f32]) -> TensorData {
    TensorData::from_le_bytes(
        shape,
        DType::F32,
        &values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

#[test]
fn local_safetensors_strictly_loads_a_fresh_linear_for_cpu_inference() {
    let directory = directory();
    let path = directory.join("linear.safetensors");
    let source = linear(2, 1);
    let mut state = BTreeMap::new();
    state.insert("weight".into(), f32([1, 2], &[2., 3.]));
    state.insert("bias".into(), f32([1], &[1.]));
    save_safetensors_file(&path, &state, &BTreeMap::new()).unwrap();

    let target = linear(2, 1);
    let report = target.load_safetensors_file_strict(&path).unwrap();
    assert_eq!(report.loaded_keys, ["bias", "weight"]);
    let input = f32([2, 2], &[1., 2., 3., 4.]);
    let first = infer_module_cpu(&target, input.clone()).unwrap();
    let second = infer_module_cpu(&target, input).unwrap();
    assert_eq!(first.output().to_vec_f64(), [9., 19.]);
    assert_eq!(first.output(), second.output());
    assert_eq!(first.trace(), second.trace());
    assert_eq!(
        first.parameter_versions(),
        &BTreeMap::from([("bias".into(), 1), ("weight".into(), 1),])
    );
    assert_ne!(source.weight.id(), target.weight.id());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn strict_state_loads_fresh_sequential_for_graph_free_inference() {
    let directory = directory();
    let path = directory.join("sequential.safetensors");
    let mut state = BTreeMap::new();
    state.insert("0.weight".into(), f32([2, 2], &[1., 0., 0., 1.]));
    state.insert("0.bias".into(), f32([2], &[0., 0.]));
    state.insert("2.weight".into(), f32([1, 2], &[2., 3.]));
    state.insert("2.bias".into(), f32([1], &[1.]));
    save_safetensors_file(&path, &state, &BTreeMap::new()).unwrap();

    let mut model = Sequential::default();
    model.push(linear(2, 2));
    model.push(ReLU::new());
    model.push(linear(2, 1));
    model.load_safetensors_file_strict(&path).unwrap();
    let first = infer_module_cpu(&model, f32([2, 2], &[-1., 2., 3., 4.])).unwrap();
    let second = infer_module_cpu(&model, f32([2, 2], &[-1., 2., 3., 4.])).unwrap();
    assert_eq!(first.output().to_vec_f64(), [7., 19.]);
    assert_eq!(first.output(), second.output());
    assert_eq!(first.trace(), second.trace());
    assert_eq!(
        first.parameter_versions(),
        &BTreeMap::from([
            ("0.bias".into(), 1),
            ("0.weight".into(), 1),
            ("2.bias".into(), 1),
            ("2.weight".into(), 1),
        ])
    );
    let empty = infer_module_cpu(&model, f32([0, 2], &[])).unwrap();
    assert_eq!(empty.output().shape().dims(), &[0, 1]);
    assert_eq!(empty.parameter_versions(), first.parameter_versions());
    let before = model.state_dict().unwrap();
    let mut unexpected = before.clone().into_tensors();
    unexpected.insert("1.weight".into(), f32([1], &[1.]));
    assert!(
        model
            .load_state_dict_strict(&ModuleStateDict::from(unexpected))
            .is_err()
    );
    assert_eq!(model.state_dict().unwrap(), before);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn strict_state_rejects_every_mismatch_without_a_visible_update() {
    let target = linear(2, 1);
    let before = target.state_dict().unwrap();
    let mut cases = Vec::new();

    let mut missing = before.clone().into_tensors();
    missing.remove("weight");
    cases.push(("missing", ModuleStateDict::from(missing)));
    let mut unexpected = before.clone().into_tensors();
    unexpected.insert("extra".into(), f32([1], &[1.]));
    cases.push(("unexpected", ModuleStateDict::from(unexpected)));
    let mut shape = before.clone().into_tensors();
    shape.insert("weight".into(), f32([1], &[1.]));
    cases.push(("shape", ModuleStateDict::from(shape)));
    let mut dtype = before.clone().into_tensors();
    dtype.insert(
        "weight".into(),
        TensorData::from_le_bytes([1, 2], DType::I32, &[1, 0, 0, 0, 2, 0, 0, 0]).unwrap(),
    );
    cases.push(("dtype", ModuleStateDict::from(dtype)));

    for (name, state) in cases {
        assert!(target.load_state_dict_strict(&state).is_err(), "{name}");
        assert_eq!(target.state_dict().unwrap(), before, "{name}");
    }

    assert!(
        target
            .load_safetensors_strict(b"not a safetensors file")
            .is_err()
    );
    assert_eq!(target.state_dict().unwrap(), before);
    assert!(
        target
            .load_safetensors_strict_with_limits(
                b"1234",
                rustgrad::StrictStateLoadLimits {
                    max_safetensors_bytes: 3
                }
            )
            .is_err()
    );
    assert_eq!(target.state_dict().unwrap(), before);

    let directory = directory();
    let oversized = directory.join("oversized.safetensors");
    fs::write(&oversized, b"1234").unwrap();
    assert!(
        target
            .load_safetensors_file_strict_with_limits(
                &oversized,
                rustgrad::StrictStateLoadLimits {
                    max_safetensors_bytes: 3,
                },
            )
            .is_err()
    );
    assert_eq!(target.state_dict().unwrap(), before);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn ordered_maps_are_deterministic_when_loading_fresh_identities() {
    let forward = ModuleStateDict::from(BTreeMap::from([
        ("weight".into(), f32([1, 2], &[2., 3.])),
        ("bias".into(), f32([1], &[1.])),
    ]));
    let reverse = ModuleStateDict::from(BTreeMap::from([
        ("bias".into(), f32([1], &[1.])),
        ("weight".into(), f32([1, 2], &[2., 3.])),
    ]));
    let first = linear(2, 1);
    let second = linear(2, 1);
    first.load_state_dict_strict(&forward).unwrap();
    second.load_state_dict_strict(&reverse).unwrap();
    assert_eq!(first.state_dict().unwrap(), second.state_dict().unwrap());
    assert_ne!(first.weight.id(), second.weight.id());
}
