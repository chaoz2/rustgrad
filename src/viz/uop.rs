use super::graph::{dtype_name, i64_list, shape_name, usize_list};
use super::{VizEdge, VizError, VizGraph, VizNode};
use crate::uop::{Binary, Unary};
use crate::{AddressSpace, UArg, UOp, UOpKind};
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

pub(super) fn kind_name(kind: &UOpKind) -> String {
    match kind {
        UOpKind::Const => "const".into(),
        UOpKind::VConst => "vconst".into(),
        UOpKind::DefineVar => "define_var".into(),
        UOpKind::DefineGlobal => "define_global".into(),
        UOpKind::DefineLocal => "define_local".into(),
        UOpKind::DefineRegister => "define_register".into(),
        UOpKind::Special => "special".into(),
        UOpKind::Range => "range".into(),
        UOpKind::EndRange => "end_range".into(),
        UOpKind::If => "if".into(),
        UOpKind::EndIf => "end_if".into(),
        UOpKind::Unary(op) => format!("unary.{}", unary_name(*op)),
        UOpKind::Binary(op) => format!("binary.{}", binary_name(*op)),
        UOpKind::GraphUnary(op) => format!("graph_unary.{}", op.name()),
        UOpKind::GraphBinary(op) => format!("graph_binary.{}", op.name()),
        UOpKind::GraphCompare(op) => format!("graph_compare.{}", op.name()),
        UOpKind::GraphLogical(op) => format!("graph_logical.{}", op.name()),
        UOpKind::Matmul => "matmul".into(),
        UOpKind::Conv2d => "conv2d.static_1x1".into(),
        UOpKind::Movement => "movement".into(),
        UOpKind::Random => "random".into(),
        UOpKind::PrefixScan => "prefix_scan".into(),
        UOpKind::Sort => "sort.pair".into(),
        UOpKind::ReduceInit => "reduce_init".into(),
        UOpKind::ReduceAccumulate => "reduce_accumulate".into(),
        UOpKind::ReduceFinalize => "reduce_finalize".into(),
        UOpKind::Ternary(_) => "ternary.where".into(),
        UOpKind::Cast => "cast".into(),
        UOpKind::Bitcast => "bitcast".into(),
        UOpKind::Vectorize => "vectorize".into(),
        UOpKind::Gep => "gep".into(),
        UOpKind::Index => "index".into(),
        UOpKind::Load => "load".into(),
        UOpKind::Store => "store".into(),
        UOpKind::EffectStore => "effect_store".into(),
        UOpKind::After => "after".into(),
        UOpKind::Barrier => "barrier".into(),
        UOpKind::Sink => "sink".into(),
    }
}

fn space_name(space: AddressSpace) -> &'static str {
    match space {
        AddressSpace::Global => "global",
        AddressSpace::Local => "local",
        AddressSpace::Register => "register",
    }
}

fn arg_fields(arg: &UArg) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    match arg {
        UArg::None => {}
        UArg::Int(value) => {
            out.insert("arg".into(), value.to_string());
        }
        UArg::Scalar { dtype, bits } => {
            out.insert("scalar_dtype".into(), dtype_name(*dtype).into());
            out.insert("scalar_bits".into(), format!("0x{bits:016x}"));
        }
        UArg::Name(name) => {
            out.insert("name".into(), name.clone());
        }
        UArg::Variable { name, bounds } => {
            out.insert("name".into(), name.clone());
            out.insert("bounds".into(), bounds.to_string());
        }
        UArg::Address {
            space,
            name,
            element,
        } => {
            out.insert("space".into(), space_name(*space).into());
            out.insert("name".into(), name.clone());
            out.insert(
                "element".into(),
                format!("{}x{}", dtype_name(element.scalar), element.lanes),
            );
        }
        UArg::RangeAxis(axis) => {
            out.insert("axis".into(), axis.to_string());
        }
        UArg::GepLane(lane) => {
            out.insert("lane".into(), lane.to_string());
        }
        UArg::BufferIndex {
            buffer,
            elements,
            input_shape,
            output_shape,
        } => {
            out.insert("buffer".into(), buffer.to_string());
            out.insert("elements".into(), elements.to_string());
            out.insert("input_shape".into(), shape_name(input_shape));
            out.insert("output_shape".into(), shape_name(output_shape));
        }
        UArg::ViewBufferIndex {
            buffer,
            elements,
            input_shape,
            output_shape,
            view,
        } => {
            out.insert("buffer".into(), buffer.to_string());
            out.insert("elements".into(), elements.to_string());
            out.insert("input_shape".into(), shape_name(input_shape));
            out.insert("output_shape".into(), shape_name(output_shape));
            out.insert("view_source".into(), shape_name(&view.source_shape));
            out.insert("view_logical".into(), shape_name(&view.logical_shape));
            out.insert("view_strides".into(), i64_list(&view.strides));
            out.insert("view_offset".into(), view.offset.to_string());
        }
        UArg::Reduction {
            input_shape,
            output_shape,
            axes,
            keepdim,
            kind,
            mean,
        } => {
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
        UArg::Matmul(plan) => matmul_fields(&mut out, plan, "serial"),
        UArg::Conv2d(plan) => {
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
        UArg::TiledMatmul(payload) => {
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
        UArg::TensorCoreMatmul(payload) => {
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
        UArg::QuantizedMatmul(plan) => {
            out.insert("strategy".into(), "quantized".into());
            out.insert("cache_key".into(), plan.cache_key.to_string());
            out.insert(
                "activation_shape".into(),
                shape_name(&plan.activation_shape),
            );
            out.insert("output_shape".into(), shape_name(&plan.output_shape));
            out.insert("m_n_k".into(), format!("{}x{}x{}", plan.m, plan.n, plan.k));
        }
        UArg::QuantizedRowGather(plan) => {
            out.insert("strategy".into(), "quantized_row_gather".into());
            out.insert("cache_key".into(), plan.cache_key.to_string());
            out.insert("indices_shape".into(), shape_name(&plan.indices_shape));
            out.insert("output_shape".into(), shape_name(&plan.output_shape));
        }
        UArg::Movement(plan) => {
            out.insert("strategy".into(), "movement".into());
            out.insert("cache_key".into(), plan.cache_key.to_string());
            out.insert("output_shape".into(), shape_name(&plan.output_shape));
            out.insert(
                "movement".into(),
                match &plan.kind {
                    crate::MovementKernelKind::AffineCopy { .. } => "affine_copy",
                    crate::MovementKernelKind::Concat { .. } => "concat",
                    crate::MovementKernelKind::Gather { .. } => "gather",
                    crate::MovementKernelKind::Scatter { add, .. } => {
                        if *add {
                            "scatter_add"
                        } else {
                            "scatter"
                        }
                    }
                }
                .into(),
            );
        }
        UArg::Random(plan) => {
            out.insert("strategy".into(), "threefry".into());
            out.insert("output".into(), plan.output.to_string());
            out.insert("output_shape".into(), shape_name(&plan.shape));
            out.insert("word_count".into(), plan.word_count.to_string());
            out.insert("device".into(), plan.stream.device.to_string());
        }
        UArg::PrefixScan {
            input,
            input_shape,
            axis,
            kind,
            output,
            dtype,
            ..
        } => {
            out.insert("input".into(), input.to_string());
            out.insert("input_shape".into(), shape_name(input_shape));
            out.insert("axis".into(), axis.to_string());
            out.insert("operation".into(), format!("{kind:?}").to_lowercase());
            out.insert("output".into(), format!("{output:?}").to_lowercase());
            out.insert("dtype".into(), dtype_name(*dtype).to_string());
        }
        UArg::Effect(payload) => {
            out.insert("effect_step".into(), payload.step.to_string());
            out.insert("target_buffer".into(), payload.target.buffer.to_string());
            out.insert("target_version".into(), payload.target.version.to_string());
            out.insert("source_buffer".into(), payload.source.buffer.to_string());
        }
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
        let mut viz = VizNode::new(format!("u{id}"), "uop", kind_name(node.kind()));
        if let Some(ty) = node.ty() {
            viz = viz.field("type", format!("{}x{}", dtype_name(ty.scalar), ty.lanes));
        }
        for (key, value) in arg_fields(node.arg()) {
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
