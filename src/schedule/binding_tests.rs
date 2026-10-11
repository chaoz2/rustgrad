use super::*;

#[test]
fn retained_order_preserves_bindings_and_independent_capture() {
    for (name, rows, transpose) in [("dense", 2, false), ("view", 2, true), ("empty", 0, false)] {
        let mut graph = Graph::new();
        let input = graph.input("input", [rows, 3]);
        let lhs = if transpose {
            graph.permute(input, vec![1, 0]).unwrap()
        } else {
            input
        };
        let rhs = graph.input("rhs", []);
        let output = graph.add(lhs, rhs).unwrap();
        let scheduled = schedule_many(&graph, &[output]).unwrap();
        let item = scheduled.items.last().unwrap();
        let order = crate::uop::TopologicalOrder::new(&item.kernel).unwrap();
        let expected = input_bindings(&item.kernel, &item.inputs, item.primary_output()).unwrap();
        let before = BINDING_ORDER_REBUILDS.get();
        let actual = input_bindings_with_order(
            &item.kernel,
            &item.inputs,
            item.primary_output(),
            Some(&order),
        )
        .unwrap();
        item.validate_input_bindings_with_order(Some(&order))
            .unwrap();
        assert_eq!(actual, expected, "{name}");
        assert_eq!(actual, item.input_bindings, "{name}");
        assert_eq!(BINDING_ORDER_REBUILDS.get(), before, "{name}");

        // Caller-visible validation and capture still inspect the schedule.
        scheduled.validate().unwrap();
        let capture = crate::CapturedSchedule::capture(&graph, &scheduled, &[output]).unwrap();
        let bytes = capture.to_bytes().unwrap();
        let decoded = crate::CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes, "{name}");
    }
}

#[test]
fn retained_order_is_not_binding_admission() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2]);
    let output = graph.neg(input).unwrap();
    let scheduled = schedule_many(&graph, &[output]).unwrap();
    let item = scheduled.items.last().unwrap();
    let order = crate::uop::TopologicalOrder::new(&item.kernel).unwrap();
    let foreign = crate::uop::TopologicalOrder::new(&UOp::sink(vec![])).unwrap();
    let before = BINDING_ORDER_REBUILDS.get();
    item.validate_input_bindings_with_order(Some(&foreign))
        .unwrap();
    assert_eq!(BINDING_ORDER_REBUILDS.get(), before + 1);

    let mut malformed = item.clone();
    malformed.input_bindings[0].abi_index += 1;
    let expected = malformed.validate_input_bindings().unwrap_err().to_string();
    for retained in [&order, &foreign] {
        assert_eq!(
            malformed
                .validate_input_bindings_with_order(Some(retained))
                .unwrap_err()
                .to_string(),
            expected
        );
    }
}

#[test]
fn external_provenance_uses_the_final_ordered_bindings() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3]);
    let external = graph.neg(input).unwrap();
    let view = graph.permute(external, vec![1, 0]).unwrap();
    let rhs = graph.input("rhs", []);
    let output = graph.add(view, rhs).unwrap();
    let scheduled =
        schedule_with_external_materializations(&graph, &[output], &[external]).unwrap();
    let item = scheduled.items.last().unwrap();
    assert_eq!(item.external_materializations, vec![external]);
    let bindings = input_bindings(&item.kernel, &item.inputs, item.primary_output()).unwrap();
    assert_eq!(item.input_bindings, bindings);
    assert!(
        bindings
            .iter()
            .any(|binding| binding.input_node == external)
    );
    scheduled.validate().unwrap();
    let capture = crate::CapturedSchedule::capture(&graph, &scheduled, &[output]).unwrap();
    let bytes = capture.to_bytes().unwrap();
    assert_eq!(
        crate::CapturedSchedule::from_bytes(&bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
}
