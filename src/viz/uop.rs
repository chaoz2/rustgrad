use super::graph::{dtype_name, i64_list, shape_name, usize_list};
use super::{VizEdge, VizError, VizGraph, VizNode};
use crate::uop::{Binary, Unary};
use crate::{AddressSpace, IndexValue, LiteralValue, MatmulValue, MovementValue, Operation, UOp};
use std::collections::BTreeMap;

fn unary_name(op: Unary) -> &'static str {
    match op {
        Unary::Neg => "neg",
        Unary::Not => "not",
        Unary::Abs => "abs",
    }
}

fn binary_name(op: Binary) -> &'static str {
    match op {
        Binary::Add => "add",
        Binary::Sub => "sub",
        Binary::Mul => "mul",
        Binary::FloorDiv => "floor_div",
        Binary::Mod => "mod",
        Binary::Min => "min",
        Binary::Max => "max",
        Binary::Eq => "eq",
        Binary::Lt => "lt",
        Binary::Le => "le",
        Binary::And => "and",
        Binary::Or => "or",
    }
}

pub(super) fn kind_name(kind: &Operation) -> String {
    match kind {
        Operation::Const(_) => "const".into(),
        Operation::VConst(_) => "vconst".into(),
        Operation::DefineVar(_) => "define_var".into(),
        Operation::DefineGlobal(_) => "define_global".into(),
        Operation::DefineLocal(_) => "define_local".into(),
        Operation::DefineRegister(_) => "define_register".into(),
        Operation::Special(_) => "special".into(),
        Operation::Range(_) => "range".into(),
        Operation::EndRange => "end_range".into(),
        Operation::If => "if".into(),
        Operation::EndIf => "end_if".into(),
        Operation::Unary(op) => format!("unary.{}", unary_name(*op)),
        Operation::Binary(op) => format!("binary.{}", binary_name(*op)),
        Operation::GraphUnary(op) => format!("graph_unary.{}", op.name()),
        Operation::GraphBinary(op) => format!("graph_binary.{}", op.name()),
        Operation::GraphCompare(op) => format!("graph_compare.{}", op.name()),
        Operation::GraphLogical(op) => format!("graph_logical.{}", op.name()),
        Operation::Matmul(_) => "matmul".into(),
        Operation::Conv2d(_) => "conv2d.static_1x1".into(),
        Operation::Movement(_) => "movement".into(),
        Operation::Random(_) => "random".into(),
        Operation::PrefixScan(_) => "prefix_scan".into(),
        Operation::Sort(_) => "sort.pair".into(),
        Operation::TensorGuard(_) => "tensor_guard".into(),
        Operation::ReduceInit(_) => "reduce_init".into(),
        Operation::ReduceAccumulate => "reduce_accumulate".into(),
        Operation::ReduceFinalize => "reduce_finalize".into(),
        Operation::Ternary(_) => "ternary.where".into(),
        Operation::Cast => "cast".into(),
        Operation::Bitcast => "bitcast".into(),
        Operation::Vectorize => "vectorize".into(),
        Operation::Gep(_) => "gep".into(),
        Operation::Index(_) => "index".into(),
        Operation::Load => "load".into(),
        Operation::Store => "store".into(),
        Operation::EffectStore(_) => "effect_store".into(),
        Operation::After(_) => "after".into(),
        Operation::Barrier => "barrier".into(),
        Operation::Sink => "sink".into(),
    }
}

fn space_name(space: AddressSpace) -> &'static str {
    match space {
        AddressSpace::Global => "global",
        AddressSpace::Local => "local",
        AddressSpace::Register => "register",
    }
}

fn operation_fields(operation: &Operation) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    match operation {
        Operation::Const(LiteralValue::Int(value))
        | Operation::VConst(LiteralValue::Int(value)) => {
            out.insert("arg".into(), value.to_string());
        }
        Operation::Const(LiteralValue::Scalar { dtype, bits })
        | Operation::VConst(LiteralValue::Scalar { dtype, bits }) => {
            out.insert("scalar_dtype".into(), dtype_name(*dtype).into());
            out.insert("scalar_bits".into(), format!("0x{bits:016x}"));
        }
        Operation::Special(name) => {
            out.insert("name".into(), name.to_owned());
        }
        Operation::DefineVar(value) => {
            out.insert("name".into(), value.name.clone());
            out.insert("bounds".into(), value.bounds.to_string());
        }
        Operation::DefineGlobal(value)
        | Operation::DefineLocal(value)
        | Operation::DefineRegister(value) => {
            out.insert("space".into(), space_name(value.space).into());
            out.insert("name".into(), value.name.clone());
            out.insert(
                "element".into(),
                format!(
                    "{}x{}",
                    dtype_name(value.element.scalar),
                    value.element.lanes
                ),
            );
        }
        Operation::Range(axis) => {
            out.insert("axis".into(), axis.to_string());
        }
        Operation::Gep(lane) => {
            out.insert("lane".into(), lane.to_string());
        }
        Operation::Index(IndexValue::Buffer {
            buffer,
            elements,
            input_shape,
            output_shape,
        }) => {
            out.insert("buffer".into(), buffer.to_string());
            out.insert("elements".into(), elements.to_string());
            out.insert("input_shape".into(), shape_name(input_shape));
            out.insert("output_shape".into(), shape_name(output_shape));
        }
        Operation::Index(IndexValue::View {
            buffer,
            elements,
            input_shape,
            output_shape,
            view,
        }) => {
            out.insert("buffer".into(), buffer.to_string());
            out.insert("elements".into(), elements.to_string());
            out.insert("input_shape".into(), shape_name(input_shape));
            out.insert("output_shape".into(), shape_name(output_shape));
            out.insert("view_source".into(), shape_name(&view.source_shape));
            out.insert("view_logical".into(), shape_name(&view.logical_shape));
            out.insert("view_strides".into(), i64_list(&view.strides));
            out.insert("view_offset".into(), view.offset.to_string());
        }
        Operation::ReduceInit(crate::ReductionValue {
            input_shape,
            output_shape,
            axes,
            keepdim,
            kind,
            mean,
        }) => {
            out.insert("input_shape".into(), shape_name(input_shape));
            out.insert("output_shape".into(), shape_name(output_shape));
            out.insert("axes".into(), usize_list(axes));
            out.insert("keepdim".into(), keepdim.to_string());
            out.insert(
                "reduction".into(),
                match kind {
                    crate::ReduceKind::Sum => "sum",
                    crate::ReduceKind::Mean => "mean",
                    crate::ReduceKind::Product => "product",
                    crate::ReduceKind::Max => "max",
                    crate::ReduceKind::Min => "min",
                    crate::ReduceKind::Any => "any",
                    crate::ReduceKind::All => "all",
                }
                .into(),
            );
            out.insert("mean".into(), mean.to_string());
        }
        Operation::Matmul(MatmulValue::Serial(plan)) => matmul_fields(&mut out, plan, "serial"),
        Operation::Conv2d(plan) => {
            out.insert("strategy".into(), "static_f32_1x1".into());
            out.insert("cache_key".into(), plan.cache_key.to_string());
            out.insert("input_shape".into(), shape_name(&plan.input_shape));
            out.insert("weight_shape".into(), shape_name(&plan.weight_shape));
            out.insert("output_shape".into(), shape_name(&plan.output_shape));
            out.insert(
                "n_cin_cout_hw".into(),
                format!(
                    "{}x{}x{}x{}x{}",
                    plan.batch, plan.input_channels, plan.output_channels, plan.height, plan.width
                ),
            );
            out.insert("bias".into(), plan.bias.is_some().to_string());
        }
        Operation::Matmul(MatmulValue::Tiled(payload)) => {
            matmul_fields(&mut out, &payload.matmul, "tiled");
            out.insert("plan_key".into(), payload.tile.cache_key.to_string());
            out.insert(
                "tile".into(),
                format!(
                    "{}x{}x{}",
                    payload.tile.block_m, payload.tile.block_n, payload.tile.block_k
                ),
            );
        }
        Operation::Matmul(MatmulValue::TensorCore(payload)) => {
            matmul_fields(&mut out, &payload.matmul, "tensor_core");
            out.insert("plan_key".into(), payload.tensor_core.cache_key.to_string());
            out.insert(
                "mma".into(),
                match payload.tensor_core.instruction {
                    crate::MmaInstruction::M16N8K16RowColF32 => "m16n8k16.row.col.f32",
                }
                .into(),
            );
        }
        Operation::Matmul(MatmulValue::Quantized(plan)) => {
            out.insert("strategy".into(), "quantized".into());
            out.insert("cache_key".into(), plan.cache_key.to_string());
            out.insert(
                "activation_shape".into(),
                shape_name(&plan.activation_shape),
            );
            out.insert("output_shape".into(), shape_name(&plan.output_shape));
            out.insert("m_n_k".into(), format!("{}x{}x{}", plan.m, plan.n, plan.k));
        }
        Operation::Movement(MovementValue::QuantizedRowGather(plan)) => {
            out.insert("strategy".into(), "quantized_row_gather".into());
            out.insert("cache_key".into(), plan.cache_key.to_string());
            out.insert("indices_shape".into(), shape_name(&plan.indices_shape));
            out.insert("output_shape".into(), shape_name(&plan.output_shape));
        }
        Operation::Movement(MovementValue::Plan(plan)) => {
            out.insert("strategy".into(), "movement".into());
            out.insert("cache_key".into(), plan.cache_key.to_string());
            out.insert("output_shape".into(), shape_name(&plan.output_shape));
            out.insert(
                "movement".into(),
                match &plan.kind {
                    crate::MovementKernelKind::AffineCopy { .. } => "affine_copy",
                    crate::MovementKernelKind::Pad { .. } => "pad",
                    crate::MovementKernelKind::Concat { .. } => "concat",
                    crate::MovementKernelKind::Gather { .. } => "gather",
                    crate::MovementKernelKind::Scatter { add, .. } => {
                        if *add {
                            "scatter_add"
                        } else {
                            "scatter"
                        }
                    }
                    crate::MovementKernelKind::Bitcast { .. } => "bitcast",
                    crate::MovementKernelKind::Contiguous { .. } => "contiguous",
                }
                .into(),
            );
        }
        Operation::Random(plan) => {
            out.insert("strategy".into(), "threefry".into());
            out.insert("output".into(), plan.output.to_string());
            out.insert("output_shape".into(), shape_name(&plan.shape));
            out.insert("word_count".into(), plan.word_count.to_string());
            out.insert("device".into(), plan.stream.device.to_string());
        }
        Operation::PrefixScan(value) => {
            out.insert("input".into(), value.input.to_string());
            out.insert("input_shape".into(), shape_name(&value.input_shape));
            out.insert("axis".into(), value.axis.to_string());
            out.insert(
                "operation".into(),
                format!("{:?}", value.kind).to_lowercase(),
            );
            out.insert(
                "output".into(),
                format!("{:?}", value.output).to_lowercase(),
            );
            out.insert("dtype".into(), dtype_name(value.dtype).to_string());
        }
        Operation::Sort(value) => {
            out.insert("input".into(), value.input.to_string());
            out.insert("input_shape".into(), shape_name(&value.input_shape));
            out.insert("axis".into(), value.axis.to_string());
            out.insert("descending".into(), value.descending.to_string());
            out.insert("values".into(), value.values.to_string());
            out.insert("indices".into(), value.indices.to_string());
            out.insert("dtype".into(), dtype_name(value.dtype).to_string());
        }
        Operation::TensorGuard(value) => {
            out.insert("input".into(), value.input.to_string());
            out.insert("input_shape".into(), shape_name(&value.input_shape));
            out.insert("axis".into(), value.axis.to_string());
            out.insert(
                "contract".into(),
                "finite_nonnegative_positive_row_sum".into(),
            );
            out.insert("dtype".into(), dtype_name(value.dtype).to_string());
        }
        Operation::EffectStore(payload) | Operation::After(payload) => {
            out.insert("effect_step".into(), payload.step.to_string());
            out.insert("target_buffer".into(), payload.target.buffer.to_string());
            out.insert("target_version".into(), payload.target.version.to_string());
            out.insert("source_buffer".into(), payload.source.buffer.to_string());
        }
        Operation::EndRange
        | Operation::If
        | Operation::EndIf
        | Operation::Unary(_)
        | Operation::Binary(_)
        | Operation::GraphUnary(_)
        | Operation::GraphBinary(_)
        | Operation::GraphCompare(_)
        | Operation::GraphLogical(_)
        | Operation::ReduceAccumulate
        | Operation::ReduceFinalize
        | Operation::Ternary(_)
        | Operation::Cast
        | Operation::Bitcast
        | Operation::Vectorize
        | Operation::Load
        | Operation::Store
        | Operation::Barrier
        | Operation::Sink => {}
    }
    out
}

fn matmul_fields(
    out: &mut BTreeMap<String, String>,
    plan: &crate::MatmulKernelPlan,
    strategy: &str,
) {
    out.insert("strategy".into(), strategy.into());
    out.insert("cache_key".into(), plan.cache_key.to_string());
    out.insert("lhs_shape".into(), shape_name(&plan.lhs_shape));
    out.insert("rhs_shape".into(), shape_name(&plan.rhs_shape));
    out.insert("output_shape".into(), shape_name(&plan.output_shape));
    out.insert("m_n_k".into(), format!("{}x{}x{}", plan.m, plan.n, plan.k));
    out.insert("batch".into(), usize_list(&plan.batch_shape));
}

/// Normalizes one validated UOp DAG using explicit topological IDs. Shared
/// sources are represented once and retain every numbered consumer edge.
pub fn uop_viz(root: &UOp) -> Result<VizGraph, VizError> {
    root.validate()
        .map_err(|error| VizError::InvalidUOp(error.to_string()))?;
    let nodes = root
        .topological()
        .map_err(|error| VizError::InvalidUOp(error.to_string()))?;
    let ids = nodes
        .iter()
        .cloned()
        .enumerate()
        .map(|(id, node)| (node, id))
        .collect::<BTreeMap<_, _>>();
    let mut viz_nodes = Vec::with_capacity(nodes.len());
    let mut edges = Vec::new();
    for (id, node) in nodes.iter().enumerate() {
        let mut viz = VizNode::new(format!("u{id}"), "uop", kind_name(node.operation()));
        if let Some(ty) = node.ty() {
            viz = viz.field("type", format!("{}x{}", dtype_name(ty.scalar), ty.lanes));
        }
        for (key, value) in operation_fields(node.operation()) {
            viz = viz.field(key, value);
        }
        for (slot, source) in node.sources().iter().enumerate() {
            let source_id = ids
                .get(source)
                .ok_or_else(|| VizError::InvalidUOp("topology omitted source".into()))?;
            edges.push(VizEdge::new(
                format!("u{source_id}"),
                format!("u{id}"),
                "source",
                slot.to_string(),
            ));
        }
        viz_nodes.push(viz);
    }
    VizGraph::try_new("rustgrad_uop", viz_nodes, edges)
}
