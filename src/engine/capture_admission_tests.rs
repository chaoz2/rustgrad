//! Capture collection relies on the complete, borrowed schedule admission.
use super::*;
use crate::{DType, Shape};

#[test]
fn binding_collection_preserves_canonical_artifact_and_replay() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", Shape::from([2]), DType::F32);
    let output = graph.square(input).unwrap();
    let requested = [output, input, output];
    let schedule = crate::schedule_many(&graph, &requested).unwrap();
    let binding = schedule.items[0]
        .ordered_inputs()
        .iter()
        .find(|binding| binding.input_node == input)
        .unwrap();
    // Assemble the expected wire object independently of capture's collector.
    let reference = crate::schedule::artifact::seal_fresh_capture(CapturedSchedule {
        items: schedule.items.clone(),
        inputs: vec![ReplayInput {
            name: "input".into(),
            node: input,
            desc: binding.desc.clone(),
        }],
        constants: BTreeMap::new(),
        quantized_constants: BTreeMap::new(),
        requested_passthroughs: schedule.requested_passthroughs.clone(),
        requested: requested.iter().map(|node| node.index() as u64).collect(),
        identity: 0,
        symbolic: None,
        specialized_from: None,
    })
    .unwrap();
    let captured = CapturedSchedule::capture(&graph, &schedule, &requested).unwrap();
    let bytes = captured.to_bytes().unwrap();
    assert_eq!(captured.identity, reference.identity);
    assert_eq!(bytes, reference.to_bytes().unwrap());
    let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.to_bytes().unwrap(), bytes);
    let values = TensorData::from_scalars(
        [2],
        DType::F32,
        [crate::Scalar::F(-2.0), crate::Scalar::F(3.0)],
    )
    .unwrap();
    let actual = decoded
        .replay(&BTreeMap::from([("input".into(), values)]))
        .unwrap();
    for (value, expected) in actual.iter().zip([[4.0, 9.0], [-2.0, 3.0], [4.0, 9.0]]) {
        let expected =
            TensorData::from_scalars([2], DType::F32, expected.map(crate::Scalar::F)).unwrap();
        assert_eq!(
            value.to_le_bytes().unwrap(),
            expected.to_le_bytes().unwrap()
        );
    }
    assert_eq!(actual.len(), 3);
}

#[test]
fn malformed_bindings_fail_at_schedule_admission_before_requested_nodes() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", Shape::from([2]), DType::F32);
    let output = graph.square(input).unwrap();
    let schedule = crate::schedule(&graph, output).unwrap();
    let mut missing = schedule.clone();
    missing.items[0].input_bindings.clear();
    let mut wrong_abi = schedule.clone();
    wrong_abi.items[0].input_bindings[0].abi_index += 1;
    let mut wrong_descriptor = schedule.clone();
    wrong_descriptor.items[0].input_bindings[0].desc.dtype = DType::I32;
    let mut duplicate = schedule;
    let binding = duplicate.items[0].input_bindings[0].clone();
    duplicate.items[0].input_bindings.push(binding);
    for (name, malformed) in [
        ("missing", missing),
        ("ABI", wrong_abi),
        ("descriptor", wrong_descriptor),
        ("duplicate", duplicate),
    ] {
        let expected = ReplayError::Corrupt(malformed.validate().unwrap_err().to_string());
        for requested in [output, NodeId(usize::MAX)] {
            assert_eq!(
                CapturedSchedule::capture(&graph, &malformed, &[requested]).unwrap_err(),
                expected,
                "{name}"
            );
        }
    }
}
