//! A deterministic local safetensors → strict `Linear` → CPU inference route.

use rustgrad::nn::Linear;
use rustgrad::{Backend, CpuBackend, Graph, Module, TensorData, save_safetensors_file};
use std::{collections::BTreeMap, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::temp_dir().join(format!(
        "rustgrad-strict-state-example-{}",
        std::process::id()
    ));
    fs::create_dir_all(&directory)?;
    let path = directory.join("linear.safetensors");
    let state = BTreeMap::from([
        ("weight".into(), TensorData::new([1, 2], vec![2.0f32, 3.0])?),
        ("bias".into(), TensorData::new([1], vec![1.0f32])?),
    ]);
    save_safetensors_file(&path, &state, &BTreeMap::new())?;

    let model = Linear::new(&mut Graph::new(), 2, 1, true, 7)?;
    model.load_safetensors_file_strict(&path)?;
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 2]);
    let output = model.forward(&mut graph, input)?;
    let mut bindings = model.input_bindings(&graph)?;
    bindings.insert(
        "input".into(),
        TensorData::new([2, 2], vec![1., 2., 3., 4.])?,
    );
    assert_eq!(
        CpuBackend.execute(&graph, output, &bindings)?.to_vec_f64(),
        [9., 19.]
    );
    fs::remove_dir_all(directory)?;
    Ok(())
}
