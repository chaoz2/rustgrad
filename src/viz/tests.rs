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
    let input = graph.input("x", [1]);
    let unsupported = graph
        .static_index(
            input,
            &[crate::ir::indexing::StaticIndex::Slice {
                start: None,
                stop: None,
                step: 1,
            }],
        )
        .unwrap();
    assert!(matches!(
        graph_viz(&graph, &[unsupported]),
        Err(VizError::UnsupportedGraphOp(_))
    ));
    assert!(matches!(
        graph_viz(&graph, &[NodeId::from_index(99)]),
        Err(VizError::InvalidGraphNode(99))
    ));
}

#[test]
fn static_index_graph_visualization_preserves_normalized_output_geometry() {
    let mut graph = Graph::new();
    let input = graph.input("input", [3, 4]);
    let indexed = graph
        .static_index(
            input,
            &[
                crate::ir::indexing::StaticIndex::Slice {
                    start: Some(-3),
                    stop: None,
                    step: 2,
                },
                crate::ir::indexing::StaticIndex::Advanced {
                    shape: Shape::from([2]),
                    values: vec![1, 1],
                },
            ],
        )
        .unwrap();
    let first = graph_viz(&graph, &[indexed]).unwrap();
    assert_eq!(first, graph_viz(&graph, &[indexed]).unwrap());
    let dot = first.to_dot();
    assert!(dot.contains("static_index\\nkind=graph_op"));
    assert!(dot.contains("index_shape=[2,2]"));
    assert!(dot.contains("data:0:input"));
}

#[test]
fn reduction_derivative_graph_visualization_preserves_axes_and_sum_to_target() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 3, 4]);
    let upstream = graph.input("upstream", [2, 1, 1]);
    let gradient = graph
        .reduce_grad(
            input,
            upstream,
            crate::ReduceKind::Mean,
            vec![1, 2],
            true,
        )
        .unwrap();
    let cotangent = graph.input("cotangent", [2, 3, 4]);
    let vjp = graph
        .reduce_grad_vjp(
            cotangent,
            input,
            upstream,
            crate::ReduceKind::Mean,
            vec![1, 2],
            true,
            1,
        )
        .unwrap();
    let compact_upstream = graph.input("compact_upstream", [2]);
    let compact_gradient = graph
        .reduce_grad(
            input,
            compact_upstream,
            crate::ReduceKind::Sum,
            vec![1, 2],
            false,
        )
        .unwrap();
    let scalar = graph.input("scalar", []);
    let scalar_sum = graph.sum_to(scalar, Shape::from([])).unwrap();
    let broadcast = graph.input("broadcast", [2, 3, 4]);
    let summed = graph.sum_to(broadcast, Shape::from([1, 3, 1])).unwrap();

    let first = graph_viz(
        &graph,
        &[gradient, vjp, compact_gradient, scalar_sum, summed],
    )
    .unwrap();
    let second = graph_viz(
        &graph,
        &[gradient, vjp, compact_gradient, scalar_sum, summed],
    )
    .unwrap();
    assert_eq!(first, second);
    let dot = first.to_dot();
    assert!(dot.contains("reduce_grad\\nkind=graph_op"));
    assert!(dot.contains("reduce_grad_vjp\\nkind=graph_op"));
    assert!(dot.contains("reduction=mean"));
    assert!(dot.contains("axes=[1,2]"));
    assert!(dot.contains("keepdim=true"));
    assert!(dot.contains("keepdim=false"));
    assert!(dot.contains("wrt=1"));
    assert!(dot.contains("data:0:input"));
    assert!(dot.contains("data:1:upstream"));
    assert!(dot.contains("data:0:cotangent"));
    assert!(dot.contains("sum_to\\nkind=graph_op"));
    assert!(dot.contains("target_shape=[]"));
    assert!(dot.contains("target_shape=[1,3,1]"));
    assert!(dot.contains("shape=[2,3,4]"));
}

#[test]
fn matmul_grad_graph_visualization_preserves_batched_vector_and_vjp_roles() {
    let mut graph = Graph::new();
    let lhs = graph.input("lhs", [2, 1, 3, 4]);
    let rhs = graph.input("rhs", [1, 5, 4, 6]);
    let upstream = graph.input("upstream", [2, 5, 3, 6]);
    let gradient = graph.matmul_grad(upstream, lhs, rhs, true).unwrap();
    let cotangent = graph.input("cotangent", [2, 1, 3, 4]);
    let vjp = graph
        .matmul_grad_vjp(cotangent, upstream, lhs, rhs, true, 2)
        .unwrap();

    let vector = graph.input("vector", [4]);
    let matrix = graph.input("matrix", [2, 4, 3]);
    let vector_upstream = graph.input("vector_upstream", [2, 3]);
    let vector_gradient = graph
        .matmul_grad(vector_upstream, vector, matrix, false)
        .unwrap();

    let first = graph_viz(&graph, &[gradient, vjp, vector_gradient]).unwrap();
    let second = graph_viz(&graph, &[gradient, vjp, vector_gradient]).unwrap();
    assert_eq!(first, second);
    let dot = first.to_dot();
    assert!(dot.contains("matmul_grad\\nkind=graph_op"));
    assert!(dot.contains("matmul_grad_vjp\\nkind=graph_op"));
    assert!(dot.contains("target=lhs"));
    assert!(dot.contains("target=rhs"));
    assert!(dot.contains("wrt=2"));
    assert!(dot.contains("data:0:upstream"));
    assert!(dot.contains("data:1:lhs"));
    assert!(dot.contains("data:2:rhs"));
    assert!(dot.contains("data:0:cotangent"));
    assert!(dot.contains("shape=[2,1,3,4]"));
    assert!(dot.contains("shape=[1,5,4,6]"));
    assert!(dot.contains("shape=[2,4,3]"));
}

#[test]
fn convolution_graph_visualization_preserves_roles_geometry_and_derivatives() {
    let mut graph = Graph::new();
    let input = graph.input("input", [1, 2, 5, 6]);
    let weight = graph.input("weight", [2, 1, 2, 3]);
    let bias = graph.input("bias", [2]);
    let options = crate::Conv2dOptions {
        groups: 2,
        stride: [2, 1],
        dilation: [1, 2],
        padding: [1, 0, 2, 1],
    };
    let forward = graph.conv2d(input, weight, Some(bias), options).unwrap();
    let upstream = graph.input("upstream", [1, 2, 3, 5]);
    let gradient = graph
        .conv2d_grad(upstream, input, weight, Some(bias), options, 1)
        .unwrap();
    let cotangent = graph.input("cotangent", [2, 1, 2, 3]);
    let vjp = graph
        .conv2d_grad_vjp(cotangent, upstream, input, weight, Some(bias), options, 1, 0)
        .unwrap();

    let transpose_input = graph.input("transpose_input", [1, 2, 3, 4]);
    let transpose_weight = graph.input("transpose_weight", [2, 1, 2, 3]);
    let transpose_options = crate::ConvTranspose2dOptions {
        groups: 2,
        stride: [2, 2],
        dilation: [1, 2],
        padding: [1, 0, 2, 1],
        output_padding: [1, 1],
    };
    let transpose = graph
        .conv_transpose2d(transpose_input, transpose_weight, None, transpose_options)
        .unwrap();
    let transpose_upstream = graph.input("transpose_upstream", [1, 2, 6, 9]);
    let transpose_gradient = graph
        .conv_transpose2d_grad(
            transpose_upstream,
            transpose_input,
            transpose_weight,
            None,
            transpose_options,
            0,
        )
        .unwrap();
    let transpose_cotangent = graph.input("transpose_cotangent", [1, 2, 3, 4]);
    let transpose_vjp = graph
        .conv_transpose2d_grad_vjp(
            transpose_cotangent,
            transpose_upstream,
            transpose_input,
            transpose_weight,
            None,
            transpose_options,
            0,
            1,
        )
        .unwrap();

    let first = graph_viz(
        &graph,
        &[forward, gradient, vjp, transpose, transpose_gradient, transpose_vjp],
    )
    .unwrap();
    let second = graph_viz(
        &graph,
        &[forward, gradient, vjp, transpose, transpose_gradient, transpose_vjp],
    )
    .unwrap();
    assert_eq!(first, second);
    let dot = first.to_dot();
    assert!(dot.contains("conv2d\\nkind=graph_op"));
    assert!(dot.contains("conv_transpose2d\\nkind=graph_op"));
    assert!(dot.contains("groups=2"));
    assert!(dot.contains("stride=[2,1]"));
    assert!(dot.contains("dilation=[1,2]"));
    assert!(dot.contains("padding=[1,0,2,1]"));
    assert!(dot.contains("output_padding=[1,1]"));
    assert!(dot.contains("target=1"));
    assert!(dot.contains("wrt=0"));
    assert!(dot.contains("target=0"));
    assert!(dot.contains("wrt=1"));
    assert!(dot.contains("data:0:input"));
    assert!(dot.contains("data:1:weight"));
    assert!(dot.contains("data:2:bias"));
    assert!(dot.contains("data:0:upstream"));
    assert!(dot.contains("data:0:cotangent"));
    assert!(dot.contains("shape=[1,2,3,5]"));
    assert!(dot.contains("shape=[1,2,6,9]"));
}

#[test]
fn einsum_graph_visualization_preserves_normalized_plan_and_derivative_roles() {
    let mut graph = Graph::new();
    let lhs = graph.input("lhs", [2, 3]);
    let rhs = graph.input("rhs", [3, 4]);
    let forward = graph.einsum("ij,jk->ik", &[lhs, rhs]).unwrap();
    let upstream = graph.input("upstream", [2, 4]);
    let plan = crate::EinsumPlan::parse(
        "ij,jk->ik",
        &[Shape::from([2, 3]), Shape::from([3, 4])],
    )
    .unwrap();
    let gradient = graph
        .einsum_grad(upstream, &[lhs, rhs], plan.clone(), 0)
        .unwrap();
    let cotangent = graph.input("cotangent", [2, 3]);
    let vjp = graph
        .einsum_grad_vjp(cotangent, upstream, &[lhs, rhs], plan, 0, 1)
        .unwrap();

    let first = graph_viz(&graph, &[forward, gradient, vjp]).unwrap();
    let second = graph_viz(&graph, &[forward, gradient, vjp]).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.to_dot(),
        "digraph \"rustgrad_graph\" {\n  graph [rankdir=\"LR\"];\n  node [shape=\"box\"];\n  \"g0\" [label=\"input\\nkind=graph_op\\ndtype=f32\\nname=lhs\\nnode=0\\nshape=[2,3]\"];\n  \"g1\" [label=\"input\\nkind=graph_op\\ndtype=f32\\nname=rhs\\nnode=1\\nshape=[3,4]\"];\n  \"g2\" [label=\"einsum\\nkind=graph_op\\ncontracted_labels=[j]\\ndtype=f32\\nnode=2\\noperand_labels=[[i,j],[j,k]]\\noutput_labels=[i,k]\\nplan_key=operands=[[i,j],[j,k]];extents=[i:2,j:3,k:4];output=[i,k];contracted=[j]\\nshape=[2,4]\"];\n  \"g3\" [label=\"input\\nkind=graph_op\\ndtype=f32\\nname=upstream\\nnode=3\\nshape=[2,4]\"];\n  \"g4\" [label=\"einsum_grad\\nkind=graph_op\\ncontracted_labels=[j]\\ndtype=f32\\nnode=4\\noperand_labels=[[i,j],[j,k]]\\noutput_labels=[i,k]\\nplan_key=operands=[[i,j],[j,k]];extents=[i:2,j:3,k:4];output=[i,k];contracted=[j]\\nshape=[2,3]\\ntarget_operand=0\"];\n  \"g5\" [label=\"input\\nkind=graph_op\\ndtype=f32\\nname=cotangent\\nnode=5\\nshape=[2,3]\"];\n  \"g6\" [label=\"einsum_grad_vjp\\nkind=graph_op\\ncontracted_labels=[j]\\ndtype=f32\\nnode=6\\noperand_labels=[[i,j],[j,k]]\\noutput_labels=[i,k]\\nplan_key=operands=[[i,j],[j,k]];extents=[i:2,j:3,k:4];output=[i,k];contracted=[j]\\nshape=[3,4]\\ntarget_operand=0\\nwrt=1\"];\n  \"g0\" -> \"g2\" [label=\"data:0:operand_0\"];\n  \"g0\" -> \"g4\" [label=\"data:1:operand_0\"];\n  \"g0\" -> \"g6\" [label=\"data:2:operand_0\"];\n  \"g1\" -> \"g2\" [label=\"data:1:operand_1\"];\n  \"g1\" -> \"g4\" [label=\"data:2:operand_1\"];\n  \"g1\" -> \"g6\" [label=\"data:3:operand_1\"];\n  \"g3\" -> \"g4\" [label=\"data:0:upstream\"];\n  \"g3\" -> \"g6\" [label=\"data:1:upstream\"];\n  \"g5\" -> \"g6\" [label=\"data:0:cotangent\"];\n}\n"
    );

    let ellipsis_lhs = graph.input("ellipsis_lhs", [2, 3, 4]);
    let ellipsis_rhs = graph.input("ellipsis_rhs", [4]);
    let ellipsis = graph
        .einsum("...i,i->...", &[ellipsis_lhs, ellipsis_rhs])
        .unwrap();
    let diagonal = graph.input("diagonal", [3, 3]);
    let trace = graph.einsum("ii->", &[diagonal]).unwrap();
    let metadata = format!(
        "{}{}",
        graph_viz(&graph, &[ellipsis]).unwrap().to_dot(),
        graph_viz(&graph, &[trace]).unwrap().to_dot(),
    );
    assert!(metadata.contains("...0"));
    assert!(metadata.contains("operand_labels=[[i,i]]"));
}

#[test]
fn scatter_positions_graph_visualization_preserves_static_map_geometry() {
    let mut graph = Graph::new();
    let input = graph.input("x", [2]);
    // A zero step makes both input coordinates target the same destination.
    // The graph-level map preserves that duplicate geometry verbatim.
    let placed = graph
        .scatter_positions(input, Shape::from([1]), vec![0], vec![0])
        .unwrap();
    let cotangent = graph.input("cotangent", [1]);
    let read = graph
        .scatter_positions_vjp(cotangent, Shape::from([2]), vec![0], vec![0])
        .unwrap();
    let first = graph_viz(&graph, &[placed, read]).unwrap();
    let second = graph_viz(&graph, &[placed, read]).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.to_dot(),
        "digraph \"rustgrad_graph\" {\n  graph [rankdir=\"LR\"];\n  node [shape=\"box\"];\n  \"g0\" [label=\"input\\nkind=graph_op\\ndtype=f32\\nname=x\\nnode=0\\nshape=[2]\"];\n  \"g1\" [label=\"scatter_positions\\nkind=graph_op\\ndtype=f32\\nmode=place\\nnode=1\\nshape=[1]\\nstarts=[0]\\nsteps=[0]\\ntarget_shape=[1]\"];\n  \"g2\" [label=\"input\\nkind=graph_op\\ndtype=f32\\nname=cotangent\\nnode=2\\nshape=[1]\"];\n  \"g3\" [label=\"scatter_positions_vjp\\nkind=graph_op\\ndtype=f32\\ninput_shape=[2]\\nmode=read_static_map\\nnode=3\\nshape=[2]\\nstarts=[0]\\nsteps=[0]\"];\n  \"g0\" -> \"g1\" [label=\"data:0:input\"];\n  \"g2\" -> \"g3\" [label=\"data:0:cotangent\"];\n}\n"
    );

    let empty = graph.input("empty", [0, 2]);
    let empty_map = graph
        .scatter_positions(empty, Shape::from([0, 4]), vec![0, 3], vec![1, -1])
        .unwrap();
    let empty_dot = graph_viz(&graph, &[empty_map]).unwrap().to_dot();
    assert!(empty_dot.contains("target_shape=[0,4]"));
    assert!(empty_dot.contains("starts=[0,3]"));
    assert!(empty_dot.contains("steps=[1,-1]"));
}

#[test]
fn masked_select_graph_visualization_preserves_fixed_and_dynamic_contracts() {
    let mut graph = Graph::new();
    let input = graph.input("x", [2, 3]);
    let mask = graph.input_dtype("mask", [1, 3], DType::Bool);
    let selected = graph
        .masked_select(input, mask, 4, crate::Scalar::F(-0.0))
        .unwrap();
    let first = graph_viz(&graph, &[selected]).unwrap();
    let second = graph_viz(&graph, &[selected]).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.to_dot(),
        "digraph \"rustgrad_graph\" {\n  graph [rankdir=\"LR\"];\n  node [shape=\"box\"];\n  \"g0\" [label=\"input\\nkind=graph_op\\ndtype=f32\\nname=x\\nnode=0\\nshape=[2,3]\"];\n  \"g1\" [label=\"input\\nkind=graph_op\\ndtype=bool\\nname=mask\\nnode=1\\nshape=[1,3]\"];\n  \"g2\" [label=\"masked_select\\nkind=graph_op\\ndtype=f32\\ndynamic_counterpart=runtime_rank1\\nfill=f:0x8000000000000000\\nnode=2\\nresult_policy=fixed_size_pad_truncate\\nshape=[4]\\nsize=4\"];\n  \"g0\" -> \"g2\" [label=\"data:0:input\"];\n  \"g1\" -> \"g2\" [label=\"data:1:mask\"];\n}\n"
    );

    let empty = graph
        .masked_select(input, mask, 0, crate::Scalar::I(7))
        .unwrap();
    let empty_dot = graph_viz(&graph, &[empty]).unwrap().to_dot();
    assert!(empty_dot.contains("fill=i:7"));
    assert!(empty_dot.contains("result_policy=fixed_size_pad_truncate"));
    assert!(empty_dot.contains("shape=[0]"));
}

#[test]
fn arg_reduce_graph_visualization_preserves_normalized_axes_and_index_contract() {
    let mut graph = Graph::new();
    let input = graph.input("x", [2, 3]);
    let reduced = graph.argmin(input, Some(-1), true).unwrap();
    let first = graph_viz(&graph, &[reduced]).unwrap();
    let second = graph_viz(&graph, &[reduced]).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.to_dot(),
        "digraph \"rustgrad_graph\" {\n  graph [rankdir=\"LR\"];\n  node [shape=\"box\"];\n  \"g0\" [label=\"input\\nkind=graph_op\\ndtype=f32\\nname=x\\nnode=0\\nshape=[2,3]\"];\n  \"g1\" [label=\"arg_reduce\\nkind=graph_op\\naxes=[1]\\ndtype=i32\\nkeepdim=true\\nnode=1\\nreduction=argmin\\nshape=[2,1]\"];\n  \"g0\" -> \"g1\" [label=\"data:0:input\"];\n}\n"
    );

    let scalar = graph.input("scalar", []);
    let global = graph.argmax(scalar, None, false).unwrap();
    let global_dot = graph_viz(&graph, &[global]).unwrap().to_dot();
    assert!(global_dot.contains("axes=all"));
    assert!(global_dot.contains("dtype=i32"));
    assert!(global_dot.contains("shape=[]"));
}

#[test]
fn pad_graph_visualization_preserves_geometry_fill_and_dependency() {
    let mut graph = Graph::new();
    let input = graph.input("x", [2, 0]);
    let padded = graph
        .pad(input, [(1, 0), (0, 2)], crate::Scalar::F(-0.0))
        .unwrap();
    let first = graph_viz(&graph, &[padded]).unwrap();
    let second = graph_viz(&graph, &[padded]).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.to_dot(),
        "digraph \"rustgrad_graph\" {\n  graph [rankdir=\"LR\"];\n  node [shape=\"box\"];\n  \"g0\" [label=\"input\\nkind=graph_op\\ndtype=f32\\nname=x\\nnode=0\\nshape=[2,0]\"];\n  \"g1\" [label=\"pad\\nkind=graph_op\\ndtype=f32\\nfill=f:0x8000000000000000\\nnode=1\\npadding=[1:0,0:2]\\nshape=[3,2]\"];\n  \"g0\" -> \"g1\" [label=\"data:0:input\"];\n}\n"
    );

    // Signed public padding lowers crop first, then the raw Pad geometry;
    // both movements remain inspectable without extending Pad's unsigned IR.
    let signed = graph
        .pad_signed(input, [(-1, 2), (0, 0)], crate::Scalar::I(0))
        .unwrap();
    let signed_dot = graph_viz(&graph, &[signed]).unwrap().to_dot();
    assert!(signed_dot.contains("shrink"));
    assert!(signed_dot.contains("bounds=[1:2,0:0]"));
    assert!(signed_dot.contains("padding=[0:2,0:0]"));
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
        "digraph \"rustgrad_graph\" {\n  graph [rankdir=\"LR\"];\n  node [shape=\"box\"];\n  \"g0\" [label=\"input\\nkind=graph_op\\ndtype=f32\\nname=x\\nnode=0\\nshape=[2,3]\"];\n  \"g1\" [label=\"input\\nkind=graph_op\\ndtype=f32\\nname=y\\nnode=1\\nshape=[2,3]\"];\n  \"g2\" [label=\"binary\\nkind=graph_op\\ndtype=f32\\nnode=2\\noperator=add\\nshape=[2,3]\"];\n  \"g3\" [label=\"constant\\nkind=graph_op\\ndtype=f32\\nelements=1\\nnode=3\\nshape=[]\"];\n  \"g4\" [label=\"compare\\nkind=graph_op\\ndtype=bool\\nnode=4\\noperator=lt\\nshape=[2,3]\"];\n  \"g5\" [label=\"select\\nkind=graph_op\\ndtype=f32\\nnode=5\\nshape=[2,3]\"];\n  \"g0\" -> \"g2\" [label=\"data:0:lhs\"];\n  \"g1\" -> \"g2\" [label=\"data:1:rhs\"];\n  \"g2\" -> \"g4\" [label=\"data:1:rhs\"];\n  \"g2\" -> \"g5\" [label=\"data:1:true\"];\n  \"g3\" -> \"g4\" [label=\"data:0:lhs\"];\n  \"g3\" -> \"g5\" [label=\"data:2:false\"];\n  \"g4\" -> \"g5\" [label=\"data:0:condition\"];\n}\n"
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
