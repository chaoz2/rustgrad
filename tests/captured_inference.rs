use rustgrad::nn::{Linear, Module, ModuleForward, Parameter, StateKind};
use rustgrad::{CapturedInference, DType, Graph, NodeId, Result, TensorData};

struct Identity;

impl Module for Identity {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}

impl ModuleForward for Identity {
    fn forward(&self, _: &mut Graph, input: NodeId) -> Result<NodeId> {
        Ok(input)
    }
}

#[test]
fn public_capture_separates_frozen_module_state_from_transient_inputs() {
    let model = Linear::new_static(2, 1, true, 901).unwrap();
    model
        .weight
        .replace(TensorData::new([1, 2], vec![2., -1.]).unwrap())
        .unwrap();
    model
        .bias
        .as_ref()
        .unwrap()
        .replace(TensorData::new([1], vec![0.5]).unwrap())
        .unwrap();

    let mut graph = Graph::new();
    let input = graph.input_dtype("features", [3, 2], DType::F32);
    let output = model.forward(&mut graph, input).unwrap();
    let captured = CapturedInference::from_module_graph(&model, &graph, &[output]).unwrap();

    assert_eq!(captured.capture().requested, [output.index() as u64]);
    assert_eq!(captured.execution_plan().requested_outputs.len(), 1);
    assert_eq!(captured.resident_bindings().len(), 2);
    assert_eq!(
        captured
            .transient_inputs()
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        ["features"]
    );
}

#[test]
fn public_capture_accepts_ordered_duplicate_identity_outputs() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("features", [2], DType::F32);
    let output = Identity.forward(&mut graph, input).unwrap();
    let captured =
        CapturedInference::from_module_graph(&Identity, &graph, &[output, output]).unwrap();

    assert!(captured.resident_bindings().is_empty());
    assert_eq!(captured.transient_inputs().len(), 1);
    assert_eq!(captured.execution_plan().schedule_item_count, 0);
    assert_eq!(captured.execution_plan().requested_outputs.len(), 2);
    assert_eq!(captured.capture().requested, [input.index() as u64; 2]);
}
