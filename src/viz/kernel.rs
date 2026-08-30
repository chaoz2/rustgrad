use super::graph::{dtype_name, shape_name};
use super::{VizEdge, VizError, VizGraph, VizNode};
use crate::{LinearKernel, MemorySpace, MemorySpacePlan, VectorOperand, VectorProgram};
use std::collections::BTreeMap;

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
            .field(
                "control_operations",
                linear
                    .program
                    .control_operations
                    .iter()
                    .map(|operation| format!("{}:{:?}", operation.index, operation.operation))
                    .collect::<Vec<_>>()
                    .join(" | "),
            )
            .field(
                "unsupported_operations",
                linear
                    .program
                    .unsupported_operations
                    .iter()
                    .map(|operation| format!("{}:{:?}", operation.index, operation.operation))
                    .collect::<Vec<_>>()
                    .join(" | "),
            )
            .field("peak_scalar", linear.program.peak_scalar.to_string())
            .field("peak_vector", linear.program.peak_vector.to_string()),
    );
    for inst in &linear.program.instructions {
        let view = inst.instruction.view();
        let mut node = VizNode::new(
            format!("l{}", inst.index),
            "linear_inst",
            view.semantic_name,
        )
        .field(
            "dtype",
            view.result_type()
                .map(|ty| dtype_name(ty.scalar))
                .unwrap_or("none"),
        )
        .field(
            "lanes",
            if linear.enabled { linear.lanes } else { 1 }.to_string(),
        )
        .field("uop", view.semantic_name);
        if let Some(dst) = view.output() {
            node = node.field("dst", dst.to_string());
        }
        if let Some(buffer) = view.buffer {
            node = node.field("buffer", buffer.to_string());
        }
        for (slot, input) in view.inputs().enumerate() {
            if let Some(producer) = linear
                .program
                .instructions
                .iter()
                .find(|candidate| candidate.instruction.view().output() == Some(input))
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

fn operand_id(operand: &VectorOperand) -> String {
    format!("vr{}:{}", u8::from(operand.vector), operand.physical)
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
    let mask = (0..usize::from(program.lanes))
        .map(|lane| {
            if lane < program.tail_elements {
                '1'
            } else {
                '0'
            }
        })
        .collect::<String>();
    let operand_key = |operand: &VectorOperand| (operand.physical, operand.vector, operand.dtype);
    let mut definitions = BTreeMap::<(u32, bool, crate::DType), u32>::new();
    for inst in &program.instructions {
        let view = inst.instruction.view();
        let mut node = VizNode::new(
            format!("v{}", inst.index),
            "vector_inst",
            view.semantic_name,
        )
        .field("lanes", program.lanes.to_string())
        .field("uop", view.semantic_name)
        .field("mask", mask.clone());
        if let Some(dst) = view.output() {
            node = node.field("dst", operand_id(dst));
        }
        for (slot, input) in view.inputs().enumerate() {
            if let Some(producer) = definitions.get(&operand_key(input)) {
                edges.push(VizEdge::new(
                    format!("v{producer}"),
                    format!("v{}", inst.index),
                    "value",
                    slot.to_string(),
                ));
            }
        }
        if let Some(output) = view.output() {
            definitions.insert(operand_key(output), inst.index);
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
