use super::graph::{dtype_name, i64_list, shape_name};
use super::uop::kind_name;
use super::{VizEdge, VizError, VizGraph, VizNode};
use crate::{
    BufferDesc, CapturedSchedule, MatmulValue, MovementValue, Operation, Schedule,
    ScheduleBoundary, ScheduleItem,
};
use std::collections::{BTreeMap, BTreeSet};

fn buffer_node(desc: &BufferDesc) -> VizNode {
    let mut node = VizNode::new(format!("b{}", desc.id), "buffer", "buffer")
        .field("buffer", desc.id.to_string())
        .field("dtype", dtype_name(desc.dtype))
        .field("shape", shape_name(&desc.shape))
        .field("bytes", desc.bytes.to_string())
        .field("alignment", desc.alignment.to_string())
        .field("read_only", desc.read_only.to_string());
    if let Some(view) = &desc.view {
        node = node
            .field("view_source", shape_name(&view.source_shape))
            .field("view_logical", shape_name(&view.logical_shape))
            .field("view_strides", i64_list(&view.strides))
            .field("view_offset", view.offset.to_string());
    }
    node
}

fn boundary_name(boundary: &Option<ScheduleBoundary>) -> &'static str {
    match boundary {
        None => "lowered",
        Some(ScheduleBoundary::Unsupported(_)) => "unsupported",
        Some(ScheduleBoundary::NonScalarUOpBridge) => "non_scalar_uop_bridge",
        Some(ScheduleBoundary::Effect) => "effect",
    }
}

fn kernel_strategy(item: &ScheduleItem) -> &'static str {
    let root = match item.kernel.operation() {
        Operation::Matmul(MatmulValue::Serial(_)) => "serial_matmul",
        Operation::Matmul(MatmulValue::Tiled(_)) => "tiled_matmul",
        Operation::Matmul(MatmulValue::TensorCore(_)) => "tensor_core_matmul",
        Operation::Matmul(MatmulValue::Quantized(_)) => "quantized_matmul",
        Operation::Movement(_) => "movement",
        Operation::ReduceInit(_) => "reduction",
        _ => "uop",
    };
    if root != "uop" {
        return root;
    }
    match item.kernel.topological() {
        Ok(nodes)
            if nodes
                .iter()
                .any(|node| matches!(node.operation(), Operation::ReduceInit(_))) =>
        {
            "reduction"
        }
        Ok(nodes)
            if nodes.iter().any(|node| {
                matches!(
                    node.operation(),
                    Operation::Movement(MovementValue::Plan(_))
                )
            }) =>
        {
            "movement"
        }
        _ => "uop",
    }
}

fn collect_buffers(items: &[ScheduleItem]) -> Result<BTreeMap<u64, BufferDesc>, VizError> {
    let mut buffers: BTreeMap<u64, BufferDesc> = BTreeMap::new();
    for item in items {
        for desc in item.inputs.iter().chain(item.outputs.iter()) {
            if let Some(previous) = buffers.get_mut(&desc.id) {
                if previous.shape != desc.shape
                    || previous.dtype != desc.dtype
                    || previous.bytes != desc.bytes
                    || previous.alignment != desc.alignment
                {
                    return Err(VizError::InvalidSchedule(format!(
                        "buffer {} has conflicting physical descriptors",
                        desc.id
                    )));
                }
                previous.read_only &= desc.read_only;
            } else {
                let mut physical = desc.clone();
                physical.view = None;
                buffers.insert(desc.id, physical);
            }
        }
    }
    Ok(buffers)
}

fn base_model(items: &[ScheduleItem]) -> Result<(Vec<VizNode>, Vec<VizEdge>), VizError> {
    let ids = items.iter().map(|item| item.id).collect::<BTreeSet<_>>();
    if ids.len() != items.len() {
        return Err(VizError::InvalidSchedule("duplicate item identity".into()));
    }
    let buffers = collect_buffers(items)?;
    let mut nodes = buffers.values().map(buffer_node).collect::<Vec<_>>();
    let mut edges = Vec::new();
    let mut quantized_nodes = BTreeSet::new();
    for item in items {
        item.validate_input_bindings()
            .map_err(|error| VizError::InvalidSchedule(error.to_string()))?;
        let expected_key = crate::schedule::item_cache_key(item)
            .map_err(|error| VizError::InvalidSchedule(error.to_string()))?;
        if item.cache_key != expected_key {
            return Err(VizError::InvalidSchedule(format!(
                "item {} cache identity mismatch",
                item.id
            )));
        }
        let kernel_nodes = item
            .kernel
            .topological()
            .map_err(|error| VizError::InvalidSchedule(error.to_string()))?
            .len();
        let mut node = VizNode::new(format!("s{}", item.id), "schedule_item", "kernel")
            .field("item", item.id.to_string())
            .field("graph_node", item.node.index().to_string())
            .field("cache_key", item.cache_key.to_string())
            .field("kernel", kind_name(item.kernel.operation()))
            .field("strategy", kernel_strategy(item))
            .field("uops", kernel_nodes.to_string())
            .field("boundary", boundary_name(&item.boundary))
            .field(
                "dependencies",
                format!(
                    "[{}]",
                    item.dependencies
                        .iter()
                        .map(u64::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            )
            .field(
                "consumers",
                format!(
                    "[{}]",
                    item.consumers
                        .iter()
                        .map(u64::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            );
        if let Some(ScheduleBoundary::Unsupported(reason)) = &item.boundary {
            node = node.field("unsupported", *reason);
        }
        if !item.external_materializations.is_empty() {
            node = node.field(
                "external_nodes",
                format!(
                    "[{}]",
                    item.external_materializations
                        .iter()
                        .map(|id| id.index().to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            );
        }
        nodes.push(node);
        for dependency in &item.dependencies {
            if !ids.contains(dependency) {
                return Err(VizError::InvalidSchedule(format!(
                    "item {} has missing dependency {dependency}",
                    item.id
                )));
            }
            edges.push(VizEdge::new(
                format!("s{dependency}"),
                format!("s{}", item.id),
                "dependency",
                "",
            ));
        }
        if let Some(missing) = item
            .consumers
            .iter()
            .find(|consumer| !ids.contains(consumer))
        {
            return Err(VizError::InvalidSchedule(format!(
                "item {} has missing consumer {missing}",
                item.id
            )));
        }
        for binding in item.ordered_inputs() {
            let source = if let Some(view) = &binding.desc.view {
                let id = format!("view{}:{}", item.id, binding.abi_index);
                nodes.push(
                    VizNode::new(id.clone(), "buffer_view", "affine view")
                        .field("buffer", binding.desc.id.to_string())
                        .field("source_shape", shape_name(&view.source_shape))
                        .field("logical_shape", shape_name(&view.logical_shape))
                        .field("strides", i64_list(&view.strides))
                        .field("offset", view.offset.to_string()),
                );
                edges.push(VizEdge::new(
                    format!("b{}", binding.desc.id),
                    id.clone(),
                    "view",
                    "physical",
                ));
                id
            } else {
                format!("b{}", binding.desc.id)
            };
            edges.push(VizEdge::new(
                source,
                format!("s{}", item.id),
                "binding",
                binding.abi_index.to_string(),
            ));
        }
        for binding in item.ordered_quantized_inputs() {
            let id = format!("q{}", binding.input_node.index());
            if quantized_nodes.insert(id.clone()) {
                nodes.push(
                    VizNode::new(id.clone(), "quantized_buffer", "packed buffer")
                        .field("node", binding.input_node.index().to_string())
                        .field("bytes", binding.desc.bytes.to_string())
                        .field("identity", binding.desc.identity.to_string()),
                );
            }
            edges.push(VizEdge::new(
                id,
                format!("s{}", item.id),
                "quantized_binding",
                binding.abi_index.to_string(),
            ));
        }
        for (position, output) in item.outputs.iter().enumerate() {
            edges.push(VizEdge::new(
                format!("s{}", item.id),
                format!("b{}", output.id),
                "materializes",
                format!("output:{position}"),
            ));
        }
    }
    Ok((nodes, edges))
}

/// Builds a deterministic schedule DAG with explicit buffers, bindings,
/// dependencies, output materializations, boundaries, and kernel/cache IDs.
pub fn schedule_viz(schedule: &Schedule) -> Result<VizGraph, VizError> {
    let (nodes, edges) = base_model(&schedule.items)?;
    VizGraph::try_new("rustgrad_schedule", nodes, edges)
}

/// Builds a deterministic graph-independent capture view. The capture node
/// records portable identity and ordered requested outputs; no tensor contents
/// or process-specific compiled objects are retained.
pub fn captured_schedule_viz(capture: &CapturedSchedule) -> Result<VizGraph, VizError> {
    crate::schedule::artifact::validate_capture(capture)
        .map_err(|error| VizError::InvalidSchedule(error.to_string()))?;
    let (mut nodes, mut edges) = base_model(&capture.items)?;
    nodes.push(
        VizNode::new("capture", "capture", "captured schedule")
            .field("identity", capture.identity.to_string())
            .field(
                "requested",
                format!(
                    "[{}]",
                    capture
                        .requested
                        .iter()
                        .map(u64::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            )
            .field("inputs", capture.inputs.len().to_string())
            .field("constants", capture.constants.len().to_string())
            .field(
                "quantized_constants",
                capture.quantized_constants.len().to_string(),
            ),
    );
    for requested in &capture.requested {
        edges.push(VizEdge::new(
            format!("b{requested}"),
            "capture",
            "requested",
            requested.to_string(),
        ));
    }
    for input in &capture.inputs {
        if let Some(node) = nodes
            .iter_mut()
            .find(|node| node.id == format!("b{}", input.desc.id))
        {
            node.fields
                .insert("binding_name".into(), input.name.clone());
        }
    }
    for id in capture.constants.keys() {
        if let Some(node) = nodes.iter_mut().find(|node| node.id == format!("b{id}")) {
            node.fields
                .insert("materialization".into(), "constant".into());
        }
    }
    VizGraph::try_new("rustgrad_capture", nodes, edges)
}
