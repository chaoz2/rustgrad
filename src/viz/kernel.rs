use super::graph::{dtype_name, shape_name};
use super::uop::kind_name;
use super::{VizEdge, VizError, VizGraph, VizNode};
use crate::{
    LinearInstKind, LinearKernel, MemorySpace, MemorySpacePlan, VectorInstKind, VectorOperand,
    VectorProgram,
};

fn linear_kind(kind: &LinearInstKind) -> &'static str {
    match kind {
        LinearInstKind::Constant => "constant",
        LinearInstKind::Address => "address",
        LinearInstKind::Index => "index",
        LinearInstKind::Load { .. } => "load",
        LinearInstKind::Cast => "cast",
        LinearInstKind::Unary => "unary",
        LinearInstKind::Binary => "binary",
        LinearInstKind::Compare => "compare",
        LinearInstKind::Select => "select",
        LinearInstKind::Store { .. } => "store",
        LinearInstKind::Other(_) => "other",
    }
}

/// Inspects validated linear instructions, explicit value dependencies,
/// register pressure, vector/tail geometry, buffer identities, and cache key.
pub fn linear_viz(linear: &LinearKernel) -> Result<VizGraph, VizError> {
    linear
        .validate()
        .map_err(|error| VizError::InvalidUOp(error.to_string()))?;
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    nodes.push(
        VizNode::new("linear", "linear_kernel", "linear kernel")
            .field("output_buffer", linear.output_buffer.to_string())
            .field("output_shape", shape_name(&linear.output_shape))
            .field("dtype", dtype_name(linear.dtype))
            .field("elements", linear.elements.to_string())
            .field("lanes", linear.lanes.to_string())
            .field("vector_main", linear.vector_main.to_string())
            .field("scalar_tail", linear.scalar_tail.to_string())
            .field("cache_key", linear.cache_key.to_string())
            .field("enabled", linear.enabled.to_string())
            .field("reason", linear.reason.clone())
            .field("peak_scalar", linear.program.peak_scalar.to_string())
            .field("peak_vector", linear.program.peak_vector.to_string()),
    );
    for inst in &linear.program.instructions {
        let mut node = VizNode::new(
            format!("l{}", inst.index),
            "linear_inst",
            linear_kind(&inst.kind),
        )
        .field("dtype", dtype_name(inst.dtype))
        .field("lanes", inst.lanes.to_string())
        .field("uop", kind_name(&inst.payload.kind()));
        if let Some(dst) = inst.dst {
            node = node.field("dst", dst.to_string());
        }
        if let LinearInstKind::Load { buffer } | LinearInstKind::Store { buffer } = inst.kind {
            node = node.field("buffer", buffer.to_string());
        }
        if let LinearInstKind::Other(reason) = &inst.kind {
            node = node.field("detail", reason.clone());
        }
        for (slot, input) in inst.inputs.iter().enumerate() {
            if let Some(producer) = linear
                .program
                .instructions
                .iter()
                .find(|candidate| candidate.dst == Some(*input))
            {
                edges.push(VizEdge::new(
                    format!("l{}", producer.index),
                    format!("l{}", inst.index),
                    "value",
                    slot.to_string(),
                ));
            }
        }
        edges.push(VizEdge::new(
            format!("l{}", inst.index),
            "linear",
            "instruction",
            inst.index.to_string(),
        ));
        nodes.push(node);
    }
    VizGraph::try_new("rustgrad_linear", nodes, edges)
}

fn space_name(space: MemorySpace) -> &'static str {
    match space {
        MemorySpace::Global => "global",
        MemorySpace::RegisterScalar => "register_scalar",
        MemorySpace::RegisterVector => "register_vector",
        MemorySpace::Private => "private",
        MemorySpace::Shared => "shared",
    }
}

/// Inspects one validated memory-space allocation/lifetime plan.
pub fn memory_space_viz(plan: &MemorySpacePlan) -> Result<VizGraph, VizError> {
    plan.validate()
        .map_err(|error| VizError::InvalidSchedule(error.to_string()))?;
    let mut nodes = vec![
        VizNode::new("memory", "memory_plan", "memory space")
            .field("cache_key", plan.cache_key.to_string())
            .field("registers", plan.registers.len().to_string())
            .field("globals", plan.globals.len().to_string())
            .field("private", plan.private.len().to_string())
            .field("shared", plan.shared.len().to_string()),
    ];
    let mut edges = Vec::new();
    for register in &plan.registers {
        let id = format!(
            "r{}:{}:{}",
            space_name(register.space),
            register.physical_reg,
            register.virtual_reg
        );
        nodes.push(
            VizNode::new(id.clone(), "register", "register")
                .field("space", space_name(register.space))
                .field("virtual", register.virtual_reg.to_string())
                .field("physical", register.physical_reg.to_string())
                .field("dtype", dtype_name(register.dtype))
                .field("lifetime", format!("{}:{}", register.start, register.end)),
        );
        edges.push(VizEdge::new(id, "memory", "allocation", "register"));
    }
    for access in &plan.globals {
        let id = format!("mb{}", access.buffer);
        nodes.push(
            VizNode::new(id.clone(), "global", "global buffer")
                .field("buffer", access.buffer.to_string())
                .field("bytes", access.bytes.to_string())
                .field("offset", access.byte_offset.to_string())
                .field("alignment", access.alignment.to_string())
                .field("mutable", access.mutable.to_string()),
        );
        edges.push(VizEdge::new(id, "memory", "allocation", "global"));
    }
    for allocation in plan.private.iter().chain(&plan.shared) {
        let id = format!("a{}:{}", space_name(allocation.space), allocation.id);
        nodes.push(
            VizNode::new(id.clone(), "allocation", "temporary")
                .field("space", space_name(allocation.space))
                .field("bytes", allocation.bytes.to_string())
                .field("alignment", allocation.alignment.to_string())
                .field(
                    "lifetime",
                    format!("{}:{}", allocation.start, allocation.end),
                ),
        );
        edges.push(VizEdge::new(id, "memory", "allocation", "temporary"));
    }
    for barrier in &plan.barriers {
        let id = format!("barrier{}", barrier.instruction);
        nodes.push(
            VizNode::new(id.clone(), "barrier", "barrier")
                .field("instruction", barrier.instruction.to_string())
                .field("uniform", barrier.uniform_control.to_string()),
        );
        edges.push(VizEdge::new(id, "memory", "synchronization", "workgroup"));
    }
    VizGraph::try_new("rustgrad_memory_space", nodes, edges)
}

fn vector_kind(kind: &VectorInstKind) -> &'static str {
    match kind {
        VectorInstKind::Splat => "splat",
        VectorInstKind::Address => "address",
        VectorInstKind::Index => "index",
        VectorInstKind::Load { .. } => "load",
        VectorInstKind::Cast => "cast",
        VectorInstKind::Unary => "unary",
        VectorInstKind::Binary => "binary",
        VectorInstKind::Compare => "compare",
        VectorInstKind::Select => "select",
        VectorInstKind::Store { .. } => "store",
        VectorInstKind::Control => "control",
    }
}

fn operand_id(operand: &VectorOperand) -> String {
    match operand {
        VectorOperand::Register {
            physical, vector, ..
        } => format!("vr{}:{physical}", u8::from(*vector)),
        VectorOperand::Global { buffer } => format!("vg{buffer}"),
    }
}

/// Inspects a validated vector program. Validation uses the supplied memory
/// plan so register and global operand identities cannot silently diverge.
pub fn vector_viz(program: &VectorProgram, spaces: &MemorySpacePlan) -> Result<VizGraph, VizError> {
    program
        .validate(spaces)
        .map_err(|error| VizError::InvalidSchedule(error.to_string()))?;
    let mut nodes = vec![
        VizNode::new("vector", "vector_program", "vector program")
            .field("cache_key", program.cache_key.to_string())
            .field("lanes", program.lanes.to_string())
            .field("main_elements", program.main_elements.to_string())
            .field("tail_elements", program.tail_elements.to_string())
            .field("enabled", program.enabled.to_string()),
    ];
    let mut edges = Vec::new();
    for inst in &program.instructions {
        let mut node = VizNode::new(
            format!("v{}", inst.index),
            "vector_inst",
            vector_kind(&inst.kind),
        )
        .field("lanes", inst.lanes.to_string())
        .field("uop", kind_name(&inst.payload.kind()))
        .field(
            "mask",
            inst.mask
                .iter()
                .map(|bit| if *bit { '1' } else { '0' })
                .collect::<String>(),
        );
        if let Some(dst) = &inst.dst {
            node = node.field("dst", operand_id(dst));
        }
        for (slot, input) in inst.inputs.iter().enumerate() {
            if let Some(producer) = program
                .instructions
                .iter()
                .find(|candidate| candidate.dst.as_ref() == Some(input))
            {
                edges.push(VizEdge::new(
                    format!("v{}", producer.index),
                    format!("v{}", inst.index),
                    "value",
                    slot.to_string(),
                ));
            }
        }
        edges.push(VizEdge::new(
            format!("v{}", inst.index),
            "vector",
            "instruction",
            inst.index.to_string(),
        ));
        nodes.push(node);
    }
    VizGraph::try_new("rustgrad_vector", nodes, edges)
}
