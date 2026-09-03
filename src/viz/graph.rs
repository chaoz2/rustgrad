use super::{VizEdge, VizError, VizGraph, VizNode};
use crate::{
    DType, EinsumLabel, EinsumPlan, Graph, NodeId, Op, RandomKind, ReduceKind, Scalar, Shape,
};
use std::collections::BTreeSet;

pub(super) fn dtype_name(dtype: DType) -> &'static str {
    dtype.stable_name()
}

pub(super) fn shape_name(shape: &Shape) -> String {
    format!(
        "[{}]",
        shape
            .dims()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(super) fn usize_list(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(super) fn i64_list(values: &[i64]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn isize_list(values: &[isize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(isize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn scalar_name(value: Scalar) -> String {
    match value {
        Scalar::Bool(value) => format!("bool:{value}"),
        Scalar::I(value) => format!("i:{value}"),
        Scalar::U(value) => format!("u:{value}"),
        // Preserve signed zero and NaN payload identity rather than relying
        // on the platform's floating-point display formatting.
        Scalar::F(value) => format!("f:0x{:016x}", value.to_bits()),
    }
}

fn reduce_name(kind: ReduceKind) -> &'static str {
    match kind {
        ReduceKind::Sum => "sum",
        ReduceKind::Mean => "mean",
        ReduceKind::Product => "product",
        ReduceKind::Max => "max",
        ReduceKind::Min => "min",
        ReduceKind::Any => "any",
        ReduceKind::All => "all",
    }
}

fn einsum_label_name(label: &EinsumLabel) -> String {
    match label {
        EinsumLabel::Named(label) => label.to_string(),
        EinsumLabel::Ellipsis(axis) => format!("...{axis}"),
    }
}

fn einsum_labels(labels: &[EinsumLabel]) -> String {
    format!(
        "[{}]",
        labels
            .iter()
            .map(einsum_label_name)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn einsum_operand_labels(operands: &[Vec<EinsumLabel>]) -> String {
    format!(
        "[{}]",
        operands
            .iter()
            .map(|labels| einsum_labels(labels))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn einsum_extents(plan: &EinsumPlan) -> String {
    format!(
        "[{}]",
        plan.label_extents
            .iter()
            .map(|(label, extent)| format!("{}:{extent}", einsum_label_name(label)))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn einsum_plan_key(plan: &EinsumPlan) -> String {
    format!(
        "operands={};extents={};output={};contracted={}",
        einsum_operand_labels(&plan.operand_labels),
        einsum_extents(plan),
        einsum_labels(&plan.output_labels),
        einsum_labels(&plan.contracted_labels),
    )
}

fn dependency(role: impl Into<String>, id: NodeId) -> (String, NodeId) {
    (role.into(), id)
}

fn operand_dependencies(inputs: &[NodeId]) -> Vec<(String, NodeId)> {
    inputs
        .iter()
        .enumerate()
        .map(|(index, id)| dependency(format!("operand_{index}"), *id))
        .collect()
}

fn convolution_dependencies(
    input: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
) -> Vec<(String, NodeId)> {
    let mut dependencies = vec![dependency("input", input), dependency("weight", weight)];
    if let Some(bias) = bias {
        dependencies.push(dependency("bias", bias));
    }
    dependencies
}

fn convolution_gradient_dependencies(
    upstream: NodeId,
    input: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
) -> Vec<(String, NodeId)> {
    let mut dependencies = vec![dependency("upstream", upstream)];
    dependencies.extend(convolution_dependencies(input, weight, bias));
    dependencies
}

fn convolution_gradient_vjp_dependencies(
    cotangent: NodeId,
    upstream: NodeId,
    input: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
) -> Vec<(String, NodeId)> {
    let mut dependencies = vec![dependency("cotangent", cotangent)];
    dependencies.extend(convolution_gradient_dependencies(
        upstream, input, weight, bias,
    ));
    dependencies
}

fn conv2d_geometry(node: VizNode, options: crate::Conv2dOptions) -> VizNode {
    node.field("groups", options.groups.to_string())
        .field("stride", usize_list(&options.stride))
        .field("dilation", usize_list(&options.dilation))
        .field("padding", usize_list(&options.padding))
}

fn conv_transpose2d_geometry(node: VizNode, options: crate::ConvTranspose2dOptions) -> VizNode {
    node.field("groups", options.groups.to_string())
        .field("stride", usize_list(&options.stride))
        .field("dilation", usize_list(&options.dilation))
        .field("padding", usize_list(&options.padding))
        .field("output_padding", usize_list(&options.output_padding))
}

fn inputs(op: &Op) -> Result<Vec<(String, NodeId)>, VizError> {
    Ok(match op {
        Op::Input { .. }
        | Op::Constant(_)
        | Op::Random { .. }
        | Op::RandomPermutation { .. }
        | Op::ShapeIota { .. } => {
            vec![]
        }
        Op::Cast { input, .. }
        | Op::Bitcast { input, .. }
        | Op::Contiguous { input }
        | Op::ContiguousBackward { input }
        | Op::Detach { input }
        | Op::Unary { input, .. }
        | Op::Reduce { input, .. }
        | Op::PrefixScan { input, .. }
        | Op::TensorGuard { input, .. }
        | Op::ArgReduce { input, .. }
        | Op::SumTo { input, .. }
        | Op::Sort { input, .. }
        | Op::Reshape { input, .. }
        | Op::Permute { input, .. }
        | Op::Expand { input, .. }
        | Op::Shrink { input, .. }
        | Op::Pad { input, .. }
        | Op::Stride { input, .. } => vec![dependency("input", *input)],
        Op::ScatterPositions { input, .. } => vec![dependency("input", *input)],
        Op::ScatterPositionsVjp { cotangent, .. } => vec![dependency("cotangent", *cotangent)],
        Op::Binary { lhs, rhs, .. } | Op::Compare { lhs, rhs, .. } | Op::Matmul { lhs, rhs } => {
            vec![dependency("lhs", *lhs), dependency("rhs", *rhs)]
        }
        Op::Threefry { counter, key } => {
            vec![dependency("counter", *counter), dependency("key", *key)]
        }
        Op::Logical { lhs, rhs, .. } => std::iter::once(dependency("lhs", *lhs))
            .chain(rhs.iter().copied().map(|id| dependency("rhs", id)))
            .collect(),
        Op::Select {
            condition,
            on_true,
            on_false,
        } => vec![
            dependency("condition", *condition),
            dependency("true", *on_true),
            dependency("false", *on_false),
        ],
        Op::Concat { inputs, .. } => inputs
            .iter()
            .copied()
            .map(|id| dependency("input", id))
            .collect(),
        Op::Gather { input, index, .. } => {
            vec![dependency("input", *input), dependency("index", *index)]
        }
        Op::Scatter {
            base,
            index,
            updates,
            ..
        } => vec![
            dependency("base", *base),
            dependency("index", *index),
            dependency("updates", *updates),
        ],
        Op::MaskedSelect { input, mask, .. } => {
            vec![dependency("input", *input), dependency("mask", *mask)]
        }
        Op::Einsum { inputs, .. } => operand_dependencies(inputs),
        Op::EinsumGrad {
            upstream, inputs, ..
        } => std::iter::once(dependency("upstream", *upstream))
            .chain(operand_dependencies(inputs))
            .collect(),
        Op::EinsumGradVjp {
            cotangent,
            upstream,
            inputs,
            ..
        } => std::iter::once(dependency("cotangent", *cotangent))
            .chain(std::iter::once(dependency("upstream", *upstream)))
            .chain(operand_dependencies(inputs))
            .collect(),
        Op::MatmulGrad {
            upstream, lhs, rhs, ..
        } => vec![
            dependency("upstream", *upstream),
            dependency("lhs", *lhs),
            dependency("rhs", *rhs),
        ],
        Op::MatmulGradVjp {
            cotangent,
            upstream,
            lhs,
            rhs,
            ..
        } => vec![
            dependency("cotangent", *cotangent),
            dependency("upstream", *upstream),
            dependency("lhs", *lhs),
            dependency("rhs", *rhs),
        ],
        Op::ReduceGrad {
            input, upstream, ..
        } => vec![
            dependency("input", *input),
            dependency("upstream", *upstream),
        ],
        Op::ReduceGradVjp {
            cotangent,
            input,
            upstream,
            ..
        } => vec![
            dependency("cotangent", *cotangent),
            dependency("input", *input),
            dependency("upstream", *upstream),
        ],
        Op::StaticIndex { input, .. } => vec![dependency("input", *input)],
        Op::StaticIndexGrad { cotangent, .. } => vec![dependency("cotangent", *cotangent)],
        Op::StaticIndexUpdate { base, value, .. } => {
            vec![dependency("base", *base), dependency("value", *value)]
        }
        Op::StaticIndexUpdateGrad { cotangent, .. } => vec![dependency("cotangent", *cotangent)],
        Op::Conv2d {
            input,
            weight,
            bias,
            ..
        }
        | Op::ConvTranspose2d {
            input,
            weight,
            bias,
            ..
        } => convolution_dependencies(*input, *weight, *bias),
        Op::Conv2dGrad {
            upstream,
            input,
            weight,
            bias,
            ..
        }
        | Op::ConvTranspose2dGrad {
            upstream,
            input,
            weight,
            bias,
            ..
        } => convolution_gradient_dependencies(*upstream, *input, *weight, *bias),
        Op::Conv2dGradVjp {
            cotangent,
            upstream,
            input,
            weight,
            bias,
            ..
        }
        | Op::ConvTranspose2dGradVjp {
            cotangent,
            upstream,
            input,
            weight,
            bias,
            ..
        } => convolution_gradient_vjp_dependencies(*cotangent, *upstream, *input, *weight, *bias),
    })
}

fn op_class(op: &Op) -> &'static str {
    match op {
        Op::Input { .. } => "input",
        Op::Constant(_) => "constant",
        Op::Random { .. } => "random",
        Op::RandomPermutation { .. } => "random_permutation",
        Op::Cast { .. } => "cast",
        Op::Bitcast { .. } => "bitcast",
        Op::Contiguous { .. } => "contiguous",
        Op::ContiguousBackward { .. } => "contiguous_backward",
        Op::Detach { .. } => "detach",
        Op::Unary { .. } => "unary",
        Op::Binary { .. } => "binary",
        Op::Threefry { .. } => "threefry",
        Op::Compare { .. } => "compare",
        Op::Logical { .. } => "logical",
        Op::Select { .. } => "select",
        Op::ShapeIota { .. } => "shape_iota",
        Op::Reduce { .. } => "reduce",
        Op::PrefixScan { .. } => "prefix_scan",
        Op::Sort { .. } => "sort",
        Op::TensorGuard { .. } => "tensor_guard",
        Op::ArgReduce { .. } => "arg_reduce",
        Op::ReduceGrad { .. } => "reduce_grad",
        Op::ReduceGradVjp { .. } => "reduce_grad_vjp",
        Op::SumTo { .. } => "sum_to",
        Op::Reshape { .. } => "reshape",
        Op::Permute { .. } => "permute",
        Op::Expand { .. } => "expand",
        Op::Shrink { .. } => "shrink",
        Op::Pad { .. } => "pad",
        Op::Stride { .. } => "stride",
        Op::Concat { .. } => "concat",
        Op::ScatterPositions { .. } => "scatter_positions",
        Op::ScatterPositionsVjp { .. } => "scatter_positions_vjp",
        Op::Gather { .. } => "gather",
        Op::StaticIndex { .. } => "static_index",
        Op::StaticIndexGrad { .. } => "static_index_grad",
        Op::StaticIndexUpdate { .. } => "static_index_update",
        Op::StaticIndexUpdateGrad { .. } => "static_index_update_grad",
        Op::Scatter { .. } => "scatter",
        Op::MaskedSelect { .. } => "masked_select",
        Op::Matmul { .. } => "matmul",
        Op::Einsum { .. } => "einsum",
        Op::EinsumGrad { .. } => "einsum_grad",
        Op::EinsumGradVjp { .. } => "einsum_grad_vjp",
        Op::MatmulGrad { .. } => "matmul_grad",
        Op::MatmulGradVjp { .. } => "matmul_grad_vjp",
        Op::Conv2d { .. } => "conv2d",
        Op::Conv2dGrad { .. } => "conv2d_grad",
        Op::Conv2dGradVjp { .. } => "conv2d_grad_vjp",
        Op::ConvTranspose2d { .. } => "conv_transpose2d",
        Op::ConvTranspose2dGrad { .. } => "conv_transpose2d_grad",
        Op::ConvTranspose2dGradVjp { .. } => "conv_transpose2d_grad_vjp",
    }
}

fn node_for(id: NodeId, op: &Op) -> Result<VizNode, VizError> {
    let node = VizNode::new(format!("g{}", id.index()), "graph_op", op_class(op));
    Ok(match op {
        Op::Input { name } => node.field("name", name),
        Op::Constant(data) => node.field("elements", data.len().to_string()),
        Op::Random { kind, stream } => {
            let distribution = match kind {
                RandomKind::Uniform { low, high } => {
                    format!("uniform:0x{:016x}:0x{:016x}", low.to_bits(), high.to_bits())
                }
                RandomKind::Normal { mean, std } => {
                    format!("normal:0x{:016x}:0x{:016x}", mean.to_bits(), std.to_bits())
                }
                RandomKind::RandInt { low, high } => format!("randint:{low}:{high}"),
            };
            node.field("distribution", distribution)
                .field("device", stream.device.to_string())
                .field("key", format!("{:?}", stream.key))
                .field("counter", format!("{:?}", stream.counter))
        }
        Op::RandomPermutation { stream } => node
            .field("device", stream.device.to_string())
            .field("key", format!("{:?}", stream.key))
            .field("counter", format!("{:?}", stream.counter)),
        Op::Cast { dtype, .. } => node.field("to", dtype_name(*dtype)),
        Op::Bitcast { dtype, .. } => node.field("to", dtype_name(*dtype)),
        Op::Unary { op, .. } => node.field("operator", op.name()),
        Op::Binary { op, .. } => node.field("operator", op.name()),
        Op::Threefry { .. } => node.field("algorithm", "threefry2x32-20"),
        Op::Compare { op, .. } => node.field("operator", op.name()),
        Op::Logical { op, .. } => node.field("operator", op.name()),
        Op::ShapeIota { source, axis } => node
            .field("source", source.to_string())
            .field("axis", axis.to_string()),
        Op::Reduce {
            kind,
            axes,
            keepdim,
            accumulator,
            ..
        } => node
            .field("reduction", reduce_name(*kind))
            .field("axes", usize_list(axes))
            .field("keepdim", keepdim.to_string())
            .field("accumulator", dtype_name(*accumulator)),
        Op::PrefixScan {
            axis, kind, output, ..
        } => node
            .field(
                "operation",
                match (kind, output) {
                    (crate::PrefixScanKind::Sum, crate::PrefixScanOutput::Values) => "cumsum",
                    (crate::PrefixScanKind::Product, crate::PrefixScanOutput::Values) => "cumprod",
                    (crate::PrefixScanKind::Max, crate::PrefixScanOutput::Values) => "cummax",
                    (crate::PrefixScanKind::Min, crate::PrefixScanOutput::Values) => "cummin",
                    (crate::PrefixScanKind::Max, crate::PrefixScanOutput::Indices) => {
                        "cummax_indices"
                    }
                    (crate::PrefixScanKind::Min, crate::PrefixScanOutput::Indices) => {
                        "cummin_indices"
                    }
                    (_, crate::PrefixScanOutput::Indices) => "prefix_scan_indices",
                },
            )
            .field("axis", axis.to_string()),
        Op::TensorGuard { axis, .. } => node
            .field("axis", axis.to_string())
            .field("contract", "finite_nonnegative_positive_row_sum"),
        Op::ArgReduce {
            max, axis, keepdim, ..
        } => node
            .field("reduction", if *max { "argmax" } else { "argmin" })
            .field(
                "axes",
                axis.map_or_else(|| "all".to_owned(), |axis| format!("[{axis}]")),
            )
            .field("keepdim", keepdim.to_string()),
        Op::Sort {
            axis, descending, ..
        } => node
            .field("axis", axis.to_string())
            .field("descending", descending.to_string()),
        Op::Reshape { shape, .. } | Op::Expand { shape, .. } => {
            node.field("target_shape", shape_name(shape))
        }
        Op::Permute { axes, .. } => node.field("axes", usize_list(axes)),
        Op::Shrink { bounds, .. } => node.field(
            "bounds",
            format!(
                "[{}]",
                bounds
                    .iter()
                    .map(|(a, b)| format!("{a}:{b}"))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        ),
        Op::Pad { padding, fill, .. } => node
            .field(
                "padding",
                format!(
                    "[{}]",
                    padding
                        .iter()
                        .map(|(before, after)| format!("{before}:{after}"))
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            )
            .field("fill", scalar_name(*fill)),
        Op::Stride { slices, .. } => node.field("rank", slices.len().to_string()),
        Op::Concat { axis, .. } | Op::Gather { axis, .. } => node.field("axis", axis.to_string()),
        Op::ScatterPositions {
            shape,
            starts,
            steps,
            ..
        } => node
            .field("mode", "place")
            .field("target_shape", shape_name(shape))
            .field("starts", isize_list(starts))
            .field("steps", isize_list(steps)),
        Op::ScatterPositionsVjp {
            input_shape,
            starts,
            steps,
            ..
        } => node
            .field("mode", "read_static_map")
            .field("input_shape", shape_name(input_shape))
            .field("starts", isize_list(starts))
            .field("steps", isize_list(steps)),
        Op::Scatter { axis, add, .. } => node
            .field("axis", axis.to_string())
            .field("mode", if *add { "add" } else { "replace" }),
        // The static graph Op is explicitly a fixed-size, pad/truncate
        // selection. The unbounded counterpart lives in Graph's separate
        // dynamic-result arena and is rank-one at realization; graph_viz does
        // not claim to execute or differentiate that dynamic result.
        Op::MaskedSelect { size, fill, .. } => node
            .field("result_policy", "fixed_size_pad_truncate")
            .field("size", size.to_string())
            .field("fill", scalar_name(*fill))
            .field("dynamic_counterpart", "runtime_rank1"),
        Op::Einsum { plan, .. } => node
            .field("plan_key", einsum_plan_key(plan))
            .field(
                "operand_labels",
                einsum_operand_labels(&plan.operand_labels),
            )
            .field("output_labels", einsum_labels(&plan.output_labels))
            .field("contracted_labels", einsum_labels(&plan.contracted_labels)),
        Op::EinsumGrad { plan, target, .. } => node
            .field("plan_key", einsum_plan_key(plan))
            .field(
                "operand_labels",
                einsum_operand_labels(&plan.operand_labels),
            )
            .field("output_labels", einsum_labels(&plan.output_labels))
            .field("contracted_labels", einsum_labels(&plan.contracted_labels))
            .field("target_operand", target.to_string()),
        Op::EinsumGradVjp {
            plan, target, wrt, ..
        } => node
            .field("plan_key", einsum_plan_key(plan))
            .field(
                "operand_labels",
                einsum_operand_labels(&plan.operand_labels),
            )
            .field("output_labels", einsum_labels(&plan.output_labels))
            .field("contracted_labels", einsum_labels(&plan.contracted_labels))
            .field("target_operand", target.to_string())
            .field("wrt", wrt.to_string()),
        Op::MatmulGrad { lhs_gradient, .. } => {
            node.field("target", if *lhs_gradient { "lhs" } else { "rhs" })
        }
        Op::MatmulGradVjp {
            lhs_gradient, wrt, ..
        } => node
            .field("target", if *lhs_gradient { "lhs" } else { "rhs" })
            .field("wrt", wrt.to_string()),
        Op::ReduceGrad {
            kind,
            axes,
            keepdim,
            ..
        } => node
            .field("reduction", reduce_name(*kind))
            .field("axes", usize_list(axes))
            .field("keepdim", keepdim.to_string()),
        Op::ReduceGradVjp {
            kind,
            axes,
            keepdim,
            wrt,
            ..
        } => node
            .field("reduction", reduce_name(*kind))
            .field("axes", usize_list(axes))
            .field("keepdim", keepdim.to_string())
            .field("wrt", wrt.to_string()),
        Op::SumTo { shape, .. } => node.field("target_shape", shape_name(shape)),
        Op::StaticIndex { plan, .. } => node.field("index_shape", shape_name(plan.output_shape())),
        Op::StaticIndexGrad {
            input_shape, plan, ..
        } => node
            .field("input_shape", shape_name(input_shape))
            .field("index_shape", shape_name(plan.output_shape())),
        Op::StaticIndexUpdate { plan, .. } => {
            node.field("index_shape", shape_name(plan.output_shape()))
        }
        Op::StaticIndexUpdateGrad {
            base_shape,
            value_shape,
            plan,
            wrt,
            ..
        } => node
            .field("base_shape", shape_name(base_shape))
            .field("value_shape", shape_name(value_shape))
            .field("index_shape", shape_name(plan.output_shape()))
            .field("wrt", format!("{wrt:?}")),
        Op::Conv2d { options, .. } => conv2d_geometry(node, *options),
        Op::Conv2dGrad {
            options, target, ..
        } => conv2d_geometry(node, *options).field("target", target.to_string()),
        Op::Conv2dGradVjp {
            options,
            target,
            wrt,
            ..
        } => conv2d_geometry(node, *options)
            .field("target", target.to_string())
            .field("wrt", wrt.to_string()),
        Op::ConvTranspose2d { options, .. } => conv_transpose2d_geometry(node, *options),
        Op::ConvTranspose2dGrad {
            options, target, ..
        } => conv_transpose2d_geometry(node, *options).field("target", target.to_string()),
        Op::ConvTranspose2dGradVjp {
            options,
            target,
            wrt,
            ..
        } => conv_transpose2d_geometry(node, *options)
            .field("target", target.to_string())
            .field("wrt", wrt.to_string()),
        Op::Contiguous { .. }
        | Op::ContiguousBackward { .. }
        | Op::Detach { .. }
        | Op::Select { .. }
        | Op::Matmul { .. } => node,
    })
}

/// Builds a deterministic model for nodes reachable from `roots`. Empty roots
/// inspect all currently allocated graph nodes.
pub fn graph_viz(graph: &Graph, roots: &[NodeId]) -> Result<VizGraph, VizError> {
    let mut selected = BTreeSet::new();
    let mut stack = if roots.is_empty() {
        (0..graph.node_count()).map(NodeId::from_index).collect()
    } else {
        roots.to_vec()
    };
    while let Some(id) = stack.pop() {
        let op = graph
            .op(id)
            .map_err(|_| VizError::InvalidGraphNode(id.index()))?;
        if selected.insert(id.index()) {
            stack.extend(inputs(op)?.into_iter().map(|(_, id)| id));
        }
    }
    let mut nodes = Vec::with_capacity(selected.len());
    let mut edges = Vec::new();
    for index in selected {
        let id = NodeId::from_index(index);
        let op = graph
            .op(id)
            .map_err(|_| VizError::InvalidGraphNode(index))?;
        nodes.push(
            node_for(id, op)?
                .field("node", index.to_string())
                .field(
                    "dtype",
                    dtype_name(
                        graph
                            .dtype(id)
                            .map_err(|_| VizError::InvalidGraphNode(index))?,
                    ),
                )
                .field(
                    "shape",
                    shape_name(
                        graph
                            .shape(id)
                            .map_err(|_| VizError::InvalidGraphNode(index))?,
                    ),
                ),
        );
        for (position, (role, source)) in inputs(op)?.into_iter().enumerate() {
            edges.push(VizEdge::new(
                format!("g{}", source.index()),
                format!("g{index}"),
                "data",
                format!("{position}:{role}"),
            ));
        }
    }
    VizGraph::try_new("rustgrad_graph", nodes, edges)
}
