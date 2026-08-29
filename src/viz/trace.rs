use super::{VizEdge, VizError, VizGraph, VizNode};
use crate::{CompileTrace, ItemBackend, RealizationTrace};

fn dtype_name(dtype: crate::DType) -> &'static str {
    match dtype {
        crate::DType::Bool => "bool", crate::DType::I8 => "i8", crate::DType::I16 => "i16", crate::DType::I32 => "i32", crate::DType::I64 => "i64", crate::DType::U8 => "u8", crate::DType::U16 => "u16", crate::DType::U32 => "u32", crate::DType::U64 => "u64", crate::DType::F16 => "f16", crate::DType::BF16 => "bf16", crate::DType::F32 => "f32", crate::DType::F64 => "f64",
    }
}
fn shape_name(shape: &crate::Shape) -> String { format!("[{}]", shape.dims().iter().map(usize::to_string).collect::<Vec<_>>().join(",")) }
fn backend_name(backend: ItemBackend) -> &'static str { match backend { ItemBackend::Interpreter => "interpreter", ItemBackend::NativeJit => "native_jit", ItemBackend::JitFallback => "jit_fallback" } }
fn u64_list(values: &[u64]) -> String { format!("[{}]", values.iter().map(u64::to_string).collect::<Vec<_>>().join(",")) }

pub fn compile_trace_viz(trace: &CompileTrace) -> Result<VizGraph, VizError> {
    if !trace.steps.iter().any(|step| step.node == trace.output) { return Err(VizError::InvalidGraphNode(trace.output.index())); }
    let nodes = trace.steps.iter().enumerate().map(|(sequence, step)| VizNode::new(format!("c{}", step.node.index()), "compile_trace", "step")
        .field("sequence", sequence.to_string()).field("operation", step.operation.clone()).field("shape", shape_name(&step.shape)).field("dtype", dtype_name(step.dtype)).field("declared_output", (step.node == trace.output).to_string())).collect();
    let edges = trace.steps.windows(2).map(|pair| VizEdge::new(format!("c{}", pair[0].node.index()), format!("c{}", pair[1].node.index()), "order", "next")).collect();
    VizGraph::try_new("rustgrad_compile_trace", nodes, edges)
}

pub fn realization_trace_viz(trace: &RealizationTrace) -> Result<VizGraph, VizError> {
    let nodes = trace.items.iter().map(|item| VizNode::new(format!("r{}", item.item), "realization_trace", "item")
        .field("backend", backend_name(item.backend)).field("cache_key", item.cache_key.to_string()).field("buffer", item.materialized_buffer.to_string()).field("last_consumer", item.last_consumer.map_or_else(|| "none".into(), |x| x.to_string())).field("allocation", item.allocation_id.map_or_else(|| "none".into(), |x| x.to_string())).field("slot", item.physical_slot.map_or_else(|| "none".into(), |x| x.to_string())).field("generation", item.generation.map_or_else(|| "none".into(), |x| x.to_string())).field("reused_from", item.reused_from.map_or_else(|| "none".into(), |x| x.to_string())).field("released", u64_list(&item.released_buffers)).field("lanes", item.lanes.to_string()).field("vector_main", item.vector_main.to_string()).field("vector_tail", item.vector_tail.to_string()).field("vector_reason", item.vector_reason.clone())).collect();
    let edges = trace.items.iter().flat_map(|item| item.dependencies.iter().map(move |dependency| VizEdge::new(format!("r{dependency}"), format!("r{}", item.item), "dependency", "data"))).collect();
    VizGraph::try_new("rustgrad_realization_trace", nodes, edges)
}
