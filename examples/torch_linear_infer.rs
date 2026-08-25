//! Strictly load a restricted local Torch state dictionary into a Linear CPU model.

use rustgrad::nn::Linear;
use rustgrad::{
    Backend, CpuBackend, DType, Graph, Module, TensorData, load_torch_state_file_strict,
};
use std::env;

fn input(values: [f32; 2]) -> rustgrad::Result<TensorData> {
    TensorData::from_le_bytes(
        [1, 2],
        DType::F32,
        &values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let state_path = args.next().ok_or(
        "usage: torch_linear_infer STATE.pt X0 X1 (restricted CPU FloatStorage state for Linear(2, 1))",
    )?;
    let values = [
        args.next().ok_or("missing X0")?.parse::<f32>()?,
        args.next().ok_or("missing X1")?.parse::<f32>()?,
    ];
    if args.next().is_some() {
        return Err("expected exactly STATE.pt X0 X1".into());
    }

    let linear = Linear::new(&mut Graph::new(), 2, 1, true, 7)?;
    load_torch_state_file_strict(&linear, state_path)?;
    let input = input(values)?;
    let mut graph = Graph::new();
    let input_node = graph.input("input", input.shape().clone());
    let output = linear.forward(&mut graph, input_node)?;
    let mut bindings = linear.input_bindings(&graph)?;
    bindings.insert("input".into(), input);
    println!(
        "{:?}",
        CpuBackend.execute(&graph, output, &bindings)?.to_vec_f64()
    );
    Ok(())
}
