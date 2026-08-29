use super::{VizEdge, VizError, VizGraph, VizNode};
use crate::{CapturedReplayTrace, CapturedSpecializationTrace, CompileTrace, CudaCollectiveTrace, ItemBackend, NativeMixedBatchTrace, NativeMixedReplayTrace, RealizationTrace, ShardedCudaExecutionTrace};

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

pub fn cuda_collective_trace_viz(trace: &[CudaCollectiveTrace]) -> Result<VizGraph, VizError> {
    let nodes = trace.iter().enumerate().map(|(sequence, action)| VizNode::new(format!("cc{}", action.action_id), "cuda_collective_trace", "submission")
        .field("sequence", sequence.to_string()).field("operation", action.operation).field("device", action.device.as_str()).field("range", format!("{}:{}", action.range.start, action.range.len)).field("cache_key", action.cache_key.clone().unwrap_or_else(|| "none".into()))).collect();
    let edges = trace.windows(2).map(|pair| VizEdge::new(format!("cc{}", pair[0].action_id), format!("cc{}", pair[1].action_id), "order", "next")).collect();
    VizGraph::try_new("rustgrad_cuda_collective_trace", nodes, edges)
}

pub fn sharded_cuda_execution_trace_viz(trace: &[ShardedCudaExecutionTrace]) -> Result<VizGraph, VizError> {
    let nodes = trace.iter().enumerate().map(|(sequence, stage)| VizNode::new(format!("cs{}", stage.stage), "sharded_cuda_execution_trace", "stage")
        .field("sequence", sequence.to_string()).field("action", stage.action).field("skipped", stage.skipped.to_string())).collect();
    let edges = trace.windows(2).map(|pair| VizEdge::new(format!("cs{}", pair[0].stage), format!("cs{}", pair[1].stage), "order", "next")).collect();
    VizGraph::try_new("rustgrad_sharded_cuda_execution_trace", nodes, edges)
}

pub fn captured_replay_trace_viz(trace: &CapturedReplayTrace) -> Result<VizGraph, VizError> {
    let nodes = trace.items.iter().enumerate().map(|(sequence, item)| VizNode::new(format!("cr{}:{}", item.invocation, item.item), "captured_replay_trace", "item").field("sequence", sequence.to_string()).field("backend", backend_name(item.backend)).field("schedule_cache_key", item.schedule_cache_key.to_string()).field("native_cache_key", item.native_cache_key.clone().unwrap_or_else(|| "none".into())).field("cache_hit", item.cache_hit.to_string()).field("lanes", item.lanes.to_string()).field("vector_main", item.vector_main.to_string()).field("vector_tail", item.vector_tail.to_string()).field("packed_weight_bytes", item.packed_weight_bytes.to_string()).field("reason", item.reason.clone())).collect();
    let edges = trace.items.windows(2).map(|p| VizEdge::new(format!("cr{}:{}", p[0].invocation,p[0].item),format!("cr{}:{}",p[1].invocation,p[1].item),"order","next")).collect(); VizGraph::try_new("rustgrad_captured_replay_trace",nodes,edges)
}
pub fn captured_specialization_trace_viz(trace: &CapturedSpecializationTrace) -> Result<VizGraph, VizError> { VizGraph::try_new("rustgrad_captured_specialization_trace", vec![VizNode::new("specialization","captured_specialization_trace","specialization").field("source_identity",trace.source_identity.to_string()).field("concrete_identity",trace.concrete_identity.to_string()).field("bindings",format!("[{}]",trace.bindings.iter().map(|(a,b)|format!("{a}:{b}")).collect::<Vec<_>>().join(","))).field("cache_hit",trace.cache_hit.to_string())],vec![]) }
pub fn native_mixed_replay_trace_viz(trace: &NativeMixedReplayTrace) -> Result<VizGraph, VizError> { VizGraph::try_new("rustgrad_native_mixed_replay_trace",vec![VizNode::new("mixed_replay","native_mixed_replay_trace","summary").field("identity",trace.identity.to_string()).field("artifact_identity",trace.artifact_identity.to_string()).field("vectorized",trace.vectorized.to_string()).field("pure_item_cache_keys",u64_list(&trace.pure_item_cache_keys))],vec![]) }
pub fn native_mixed_batch_trace_viz(trace: &NativeMixedBatchTrace) -> Result<VizGraph, VizError> { if trace.binding_count != trace.binding_schema_keys.len() { return Err(VizError::InvalidSchedule("binding count does not match schema keys".into())); } VizGraph::try_new("rustgrad_native_mixed_batch_trace",vec![VizNode::new("mixed_batch","native_mixed_batch_trace","summary").field("identity",trace.identity.to_string()).field("batch_identity",trace.batch_identity.to_string()).field("vectorized",trace.vectorized.to_string()).field("binding_count",trace.binding_count.to_string()).field("binding_schema_keys",u64_list(&trace.binding_schema_keys)).field("pure_item_cache_keys",u64_list(&trace.pure_item_cache_keys))],vec![]) }
