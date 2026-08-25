use super::{VizEdge, VizError, VizGraph, VizNode};
use crate::{DType, Graph, NodeId, Op, RandomKind, ReduceKind, Shape};
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

fn unsupported(op: &Op) -> VizError {
    VizError::UnsupportedGraphOp(op_class(op).into())
}

fn inputs(op: &Op) -> Result<Vec<(&'static str, NodeId)>, VizError> {
    Ok(match op {
        Op::Input { .. } | Op::Constant(_) | Op::Random { .. } | Op::RandomPermutation { .. } => {
            vec![]
        }
        Op::Cast { input, .. }
        | Op::Detach { input }
        | Op::Unary { input, .. }
        | Op::Reduce { input, .. }
        | Op::PrefixScan { input, .. }
        | Op::Reshape { input, .. }
        | Op::Permute { input, .. }
        | Op::Expand { input, .. }
        | Op::Shrink { input, .. }
        | Op::Stride { input, .. } => vec![("input", *input)],
        Op::Binary { lhs, rhs, .. } | Op::Compare { lhs, rhs, .. } | Op::Matmul { lhs, rhs } => {
            vec![("lhs", *lhs), ("rhs", *rhs)]
        }
        Op::Logical { lhs, rhs, .. } => std::iter::once(("lhs", *lhs))
            .chain(rhs.iter().copied().map(|id| ("rhs", id)))
            .collect(),
        Op::Select {
            condition,
            on_true,
            on_false,
        } => vec![
            ("condition", *condition),
            ("true", *on_true),
            ("false", *on_false),
        ],
        Op::Concat { inputs, .. } => inputs.iter().copied().map(|id| ("input", id)).collect(),
        Op::Gather { input, index, .. } => vec![("input", *input), ("index", *index)],
        Op::Scatter {
            base,
            index,
            updates,
            ..
        } => vec![("base", *base), ("index", *index), ("updates", *updates)],
        Op::StaticIndexUpdateGrad { cotangent, .. } => vec![("cotangent", *cotangent)],
        _ => return Err(unsupported(op)),
    })
}

fn op_class(op: &Op) -> &'static str {
    match op {
        Op::Input { .. } => "input",
        Op::Constant(_) => "constant",
        Op::Random { .. } => "random",
        Op::RandomPermutation { .. } => "random_permutation",
        Op::Cast { .. } => "cast",
        Op::Detach { .. } => "detach",
        Op::Unary { .. } => "unary",
        Op::Binary { .. } => "binary",
        Op::Compare { .. } => "compare",
        Op::Logical { .. } => "logical",
        Op::Select { .. } => "select",
        Op::Reduce { .. } => "reduce",
        Op::PrefixScan { .. } => "prefix_scan",
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
        Op::Unary { op, .. } => node.field("operator", op.name()),
        Op::Binary { op, .. } => node.field("operator", op.name()),
        Op::Compare { op, .. } => node.field("operator", op.name()),
        Op::Logical { op, .. } => node.field("operator", op.name()),
        Op::Reduce {
            kind,
            axes,
            keepdim,
            ..
        } => node
            .field("reduction", reduce_name(*kind))
            .field("axes", usize_list(axes))
            .field("keepdim", keepdim.to_string()),
        Op::PrefixScan { axis, .. } => node
            .field("operation", "cumsum")
            .field("axis", axis.to_string()),
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
        Op::Stride { slices, .. } => node.field("rank", slices.len().to_string()),
        Op::Concat { axis, .. } | Op::Gather { axis, .. } => node.field("axis", axis.to_string()),
        Op::Scatter { axis, add, .. } => node
            .field("axis", axis.to_string())
            .field("mode", if *add { "add" } else { "replace" }),
        Op::Detach { .. } | Op::Select { .. } | Op::Matmul { .. } => node,
        _ => return Err(unsupported(op)),
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
