use super::*;
use crate::uop::Binary;
use crate::{
    DType, Graph, LinearKernel, MemorySpacePlan, NodeId, Shape, UOp, UType, VectorProgram,
    schedule, schedule_many,
};

#[test]
fn normalized_model_is_order_independent_and_dot_escapes() {
    let a = VizNode::new("a", "source", "quoted \"source\"").field("line", "a\nb");
    let b = VizNode::new("b", "sink", "sink");
    let edge = VizEdge::new("a", "b", "data", "0");
    let first =
        VizGraph::try_new("inspection", vec![a.clone(), b.clone()], vec![edge.clone()]).unwrap();
    let second = VizGraph::try_new("inspection", vec![b, a], vec![edge]).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.to_dot(), second.to_dot());
    assert!(
        first
            .to_dot()
            .contains("quoted \\\"source\\\"\\nkind=source")
    );
    assert!(first.to_dot().contains("line=a\\nb"));
}

#[test]
fn malformed_models_and_unsupported_graph_ops_fail_closed() {
    assert!(matches!(
        VizGraph::try_new(
            "x",
            vec![VizNode::new("a", "k", "a")],
            vec![VizEdge::new("a", "b", "data", "")]
        ),
        Err(VizError::MissingEndpoint { .. })
    ));
    let mut graph = Graph::new();
    let input = graph.input("x", [2]);
    let padded = graph.pad(input, [(1, 1)], crate::Scalar::F(0.0)).unwrap();
    assert!(matches!(
        graph_viz(&graph, &[padded]),
        Err(VizError::UnsupportedGraphOp(_))
    ));
    assert!(matches!(
        graph_viz(&graph, &[NodeId::from_index(99)]),
        Err(VizError::InvalidGraphNode(99))
    ));
}

#[test]
fn fused_graph_snapshot_has_typed_edges_shape_and_dtype() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 3]);
    let y = graph.input("y", [2, 3]);
    let added = graph.add(x, y).unwrap();
    let output = graph.relu(added).unwrap();
    let dot = graph_viz(&graph, &[output]).unwrap().to_dot();
    assert_eq!(
        dot,
        "digraph \"rustgrad_graph\" {\n  graph [rankdir=\"LR\"];\n  node [shape=\"box\"];\n  \"g0\" [label=\"input\\nkind=graph_op\\ndtype=f32\\nname=x\\nnode=0\\nshape=[2,3]\"];\n  \"g1\" [label=\"input\\nkind=graph_op\\ndtype=f32\\nname=y\\nnode=1\\nshape=[2,3]\"];\n  \"g2\" [label=\"binary\\nkind=graph_op\\ndtype=f32\\nnode=2\\noperator=add\\nshape=[2,3]\"];\n  \"g3\" [label=\"unary\\nkind=graph_op\\ndtype=f32\\nnode=3\\noperator=relu\\nshape=[2,3]\"];\n  \"g0\" -> \"g2\" [label=\"data:0:lhs\"];\n  \"g1\" -> \"g2\" [label=\"data:1:rhs\"];\n  \"g2\" -> \"g3\" [label=\"data:0:input\"];\n}\n"
    );
}

#[test]
fn uop_snapshot_preserves_shared_subgraph_and_typed_metadata() {
    let shared = UOp::constant(7, UType::scalar(DType::I32));
    let root = UOp::binary(Binary::Add, shared.clone(), shared);
    let model = uop_viz(&root).unwrap();
    assert_eq!(model.nodes().len(), 2);
    assert_eq!(model.edges().len(), 2);
    assert_eq!(model.edges()[0].from(), "u0");
    assert_eq!(model.edges()[1].from(), "u0");
    assert_eq!(model.edges()[0].to(), "u1");
    assert_eq!(model.edges()[1].to(), "u1");
    assert_ne!(model.edges()[0].label(), model.edges()[1].label());
    assert!(model.to_dot().contains("binary.add"));
}

#[test]
fn schedule_and_capture_show_dependencies_materializations_and_identity() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 3], DType::F32);
    let rhs = graph.input_dtype("rhs", [3, 2], DType::F32);
    let bias = graph.input_dtype("bias", [2, 2], DType::F32);
    let lhs = graph.square(input).unwrap();
    let product = graph.matmul(lhs, rhs).unwrap();
    let output = graph.add(product, bias).unwrap();
    let scheduled = schedule(&graph, output).unwrap();
    let model = schedule_viz(&scheduled).unwrap();
    assert_eq!(scheduled.items.len(), 3);
    assert!(model.edges().iter().any(|edge| edge.kind() == "dependency"));
    assert!(
        model
            .edges()
            .iter()
            .any(|edge| edge.kind() == "materializes")
    );
    assert!(model.nodes().iter().any(|node| {
        node.fields()
            .get("strategy")
            .is_some_and(|value| value.contains("matmul"))
    }));
    let matmul_item = scheduled
        .items
        .iter()
        .find(|item| item.node == product)
        .unwrap();
    let matmul_model = uop_viz(&matmul_item.kernel).unwrap();
    assert!(
        matmul_model
            .nodes()
            .iter()
            .any(|node| node.fields().get("m_n_k").map(String::as_str) == Some("2x2x3"))
    );
    let capture = crate::CapturedSchedule::capture(&graph, &scheduled, &[output]).unwrap();
    let capture_model = captured_schedule_viz(&capture).unwrap();
    assert_eq!(
        capture_model
            .nodes()
            .iter()
            .find(|node| node.id() == "capture")
            .unwrap()
            .fields()["identity"],
        capture.identity.to_string()
    );
    assert!(
        capture_model
            .edges()
            .iter()
            .any(|edge| edge.kind() == "requested")
    );
    let roundtrip = crate::CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap();
    assert_eq!(
        captured_schedule_viz(&capture).unwrap().to_dot(),
        captured_schedule_viz(&roundtrip).unwrap().to_dot()
    );
}

#[test]
fn reduction_movement_and_late_kernel_plans_are_inspectable() {
    let mut graph = Graph::new();
    let left = graph.input("left", [2, 3]);
    let right = graph.input("right", [2, 3]);
    let joined = graph.concat([left, right], 1).unwrap();
    let reduced = graph.sum(joined, 1).unwrap();
    let scheduled = schedule_many(&graph, &[joined, reduced]).unwrap();
    let schedule_dot = schedule_viz(&scheduled).unwrap().to_dot();
    assert!(schedule_dot.contains("strategy=movement"));
    assert!(schedule_dot.contains("strategy=reduction"));

    let mut elementwise = Graph::new();
    let a = elementwise.input("a", Shape::from([17]));
    let b = elementwise.input("b", Shape::from([17]));
    let out = elementwise.add(a, b).unwrap();
    let item = schedule(&elementwise, out).unwrap().items.remove(0);
    let linear = LinearKernel::from_uop(&item.kernel).unwrap();
    let spaces = MemorySpacePlan::from_linear(&linear).unwrap();
    let vector = VectorProgram::from_linear(&linear, &spaces).unwrap();
    assert!(linear_viz(&linear).unwrap().to_dot().contains("cache_key="));
    assert!(
        memory_space_viz(&spaces)
            .unwrap()
            .to_dot()
            .contains("lifetime=")
    );
    assert!(
        vector_viz(&vector, &spaces)
            .unwrap()
            .to_dot()
            .contains("tail_elements=1")
    );
}

#[test]
fn affine_view_bindings_render_physical_and_logical_addressing() {
    let mut graph = Graph::new();
    let source = graph.input("source", [4, 1]);
    let other = graph.input("other", [3, 2]);
    let shrunk = graph.shrink(source, [(1, 4), (0, 1)]).unwrap();
    let expanded = graph.expand(shrunk, [3, 2]).unwrap();
    let output = graph.add(expanded, other).unwrap();
    let model = schedule_viz(&schedule(&graph, output).unwrap()).unwrap();
    let view = model
        .nodes()
        .iter()
        .find(|node| node.kind() == "buffer_view")
        .unwrap();
    assert_eq!(view.fields()["source_shape"], "[4,1]");
    assert_eq!(view.fields()["logical_shape"], "[3,2]");
    assert_eq!(view.fields()["strides"], "[1,0]");
    assert_eq!(view.fields()["offset"], "1");
    assert!(model.edges().iter().any(|edge| edge.kind() == "view"));
}

#[test]
fn malformed_schedule_rejects_before_rendering() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2]);
    let y = graph.neg(x).unwrap();
    let mut scheduled = schedule(&graph, y).unwrap();
    scheduled.items[0].dependencies.push(999);
    assert!(matches!(
        schedule_viz(&scheduled),
        Err(VizError::InvalidSchedule(_))
    ));

    let duplicate = VizGraph::try_new(
        "duplicate",
        vec![VizNode::new("x", "a", "a"), VizNode::new("x", "b", "b")],
        vec![],
    );
    assert!(matches!(duplicate, Err(VizError::DuplicateNode(id)) if id == "x"));
}
