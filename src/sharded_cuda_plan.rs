//! Deterministic CUDA realization planning for graph-composed sharded tensors.
//!
//! This is deliberately a data-only Phase 3A plan. It neither enters a CUDA context
//! nor creates streams, allocations, modules, or Driver work.
use crate::collective::{
    CollectiveKind, CollectivePlan, CollectivePlanner, CollectiveRequest, DeviceGroup,
    DeviceId as SemanticDeviceId, Reduction,
};
use crate::sharded_cuda_execute::{BufferSubstitution, ShardedCudaPlanComposition};
use crate::{
    Capability, DType, Error, Graph, PrimaryContext, PtxRenderer, RenderedPtx, Shape,
    ShardedGraphTensor, schedule, schedule_with_external_materializations,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Caller-supplied owner/capability binding. Context resources stay outside the serializable plan.
#[derive(Clone)]
pub struct CudaPlanBinding {
    pub device: SemanticDeviceId,
    pub context: PrimaryContext,
    pub capability: Capability,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CudaPlanDiagnostic {
    Unsupported { node: usize, reason: String },
    CapabilityMismatch { reason: String },
    Trace { action: String, reason: String },
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CudaPlanStage {
    Local {
        id: usize,
        device: SemanticDeviceId,
        owner_identity: usize,
        node: usize,
        shape: Shape,
        dtype: DType,
        inputs: Vec<u64>,
        /// Typed computed input nodes explicitly supplied by a preceding stage.
        external_materializations: Vec<u64>,
        output: u64,
        dependencies: Vec<usize>,
        source_key: String,
        module_key: String,
        diagnostic: Option<CudaPlanDiagnostic>,
    },
    Collective {
        id: usize,
        action: String,
        plan: CollectivePlan,
        dependencies: Vec<usize>,
    },
    Transfer {
        id: usize,
        action: String,
        routes: Vec<CudaTransferRoute>,
        dependencies: Vec<usize>,
    },
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CudaTransferRoute {
    pub source_rank: usize,
    pub source_device: SemanticDeviceId,
    pub source_buffer: u64,
    pub source_element_offset: usize,
    pub destination_rank: usize,
    pub destination_device: SemanticDeviceId,
    pub destination_buffer: u64,
    pub destination_element_offset: usize,
    pub elements: usize,
    pub bytes: usize,
    pub dtype: DType,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShardedCudaPlan {
    pub graph_id: u64,
    pub layout_key: String,
    pub bindings: Vec<(SemanticDeviceId, usize, u32)>,
    pub stages: Vec<CudaPlanStage>,
    pub diagnostics: Vec<CudaPlanDiagnostic>,
    pub cache_key: String,
}
/// Non-serializable execution companion retaining exact PTX ABI artifacts and primary owners.
pub struct ExecutableShardedCudaPlan {
    pub logical: ShardedCudaPlan,
    pub owners: Vec<PrimaryContext>,
    pub kernels: Vec<Option<RenderedPtx>>,
    pub buffers: Vec<ExecutableBuffer>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutableBufferRole {
    External,
    Output,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutableBuffer {
    pub rank: usize,
    pub device: SemanticDeviceId,
    pub owner_identity: usize,
    pub buffer: u64,
    pub dtype: DType,
    pub shape: Shape,
    pub bytes: usize,
    pub producer: Option<usize>,
    pub consumers: Vec<usize>,
    pub first_stage: usize,
    pub last_stage: usize,
    pub role: ExecutableBufferRole,
}
impl ExecutableShardedCudaPlan {
    /// Pure preflight of the canonical map and exact transfer endpoints; it has no CUDA side effects.
    pub fn validate(&self) -> Result<(), Error> {
        for stage in &self.logical.stages {
            if let CudaPlanStage::Transfer { routes, .. } = stage {
                for route in routes {
                    let source = self
                        .buffers
                        .iter()
                        .find(|buffer| {
                            buffer.rank == route.source_rank && buffer.buffer == route.source_buffer
                        })
                        .ok_or_else(|| {
                            err("transfer source buffer is absent from canonical map")
                        })?;
                    let destination = self
                        .buffers
                        .iter()
                        .find(|buffer| {
                            buffer.rank == route.destination_rank
                                && buffer.buffer == route.destination_buffer
                        })
                        .ok_or_else(|| {
                            err("transfer destination buffer is absent from canonical map")
                        })?;
                    if source.device != route.source_device
                        || destination.device != route.destination_device
                        || source.dtype != route.dtype
                        || destination.dtype != route.dtype
                    {
                        return Err(err("transfer route owner/device/dtype mismatch"));
                    }
                    let source_end = route
                        .source_element_offset
                        .checked_mul(route.dtype.itemsize())
                        .and_then(|x| x.checked_add(route.bytes))
                        .ok_or_else(|| err("transfer source range overflow"))?;
                    let destination_end = route
                        .destination_element_offset
                        .checked_mul(route.dtype.itemsize())
                        .and_then(|x| x.checked_add(route.bytes))
                        .ok_or_else(|| err("transfer destination range overflow"))?;
                    if source_end > source.bytes
                        || destination_end > destination.bytes
                        || route.bytes
                            != route
                                .elements
                                .checked_mul(route.dtype.itemsize())
                                .ok_or_else(|| err("transfer byte overflow"))?
                    {
                        return Err(err("transfer range exceeds canonical buffer"));
                    }
                }
            }
        }
        Ok(())
    }
}
pub struct ShardedCudaPlanner;
impl ShardedCudaPlanner {
    pub fn build(
        graph: &Graph,
        value: &ShardedGraphTensor,
        bindings: &[CudaPlanBinding],
    ) -> Result<ShardedCudaPlan, Error> {
        if value.graph_id() != graph.id() {
            return Err(err("sharded tensor belongs to another graph"));
        }
        let group = value.layout().group();
        validate_bindings(group, bindings)?;
        let mut stages = Vec::new();
        let mut diagnostics = Vec::new();
        let mut previous = Vec::new();
        for (rank, node) in value.nodes().iter().enumerate() {
            let binding = &bindings[rank];
            let owner_identity = binding.context.identity();
            let scheduled = schedule(graph, *node).map_err(|e| err(e.to_string()))?;
            let mut diagnostic = scheduled
                .items
                .first()
                .and_then(|item| item.boundary.as_ref())
                .map(|x| CudaPlanDiagnostic::Unsupported {
                    node: node.index(),
                    reason: format!("schedule boundary: {x:?}"),
                });
            let source_key = scheduled
                .items
                .first()
                .map(|x| format!("schedule:{}", x.cache_key))
                .unwrap_or_else(|| "schedule:empty".into());
            if diagnostic.is_none()
                && let Some(item) = scheduled.items.first()
                && PtxRenderer::new(binding.capability.sm())
                    .and_then(|r| r.render(&item.kernel))
                    .is_err()
            {
                diagnostic=Some(CudaPlanDiagnostic::Unsupported{node:node.index(),reason:"current PTX renderer accepts only elementwise/select/cast; reductions are Phase 3B1 diagnostics".into()});
            }
            if let Some(d) = diagnostic.clone() {
                diagnostics.push(d);
            }
            let id = stages.len();
            let item = scheduled.items.first();
            stages.push(CudaPlanStage::Local {
                id,
                device: binding.device.clone(),
                owner_identity,
                node: node.index(),
                shape: graph.shape(*node)?.clone(),
                dtype: graph.dtype(*node)?,
                inputs: item
                    .map(|x| x.inputs.iter().map(|b| b.id).collect())
                    .unwrap_or_default(),
                external_materializations: vec![],
                output: item.map(|x| x.output.id).unwrap_or(node.index() as u64),
                dependencies: previous.clone(),
                source_key: source_key.clone(),
                module_key: format!(
                    "owner:{}:sm{}:{source_key}",
                    owner_identity,
                    binding.capability.sm()
                ),
                diagnostic,
            });
            previous = vec![id];
        }
        for trace in &value.trace().steps {
            if trace.action.contains("all-reduce") {
                let plan = collective_plan(group, value.dtype(), value.layout())?;
                let id = stages.len();
                stages.push(CudaPlanStage::Collective {
                    id,
                    action: trace.action.into(),
                    plan,
                    dependencies: previous.clone(),
                });
                previous = vec![id];
            } else if trace.action == "redistribute" || trace.action == "gather-movement" {
                let id = stages.len();
                if trace.routes.is_empty() {
                    return Err(err("redistribution trace has no concrete routes"));
                }
                let routes = trace
                    .routes
                    .iter()
                    .map(|route| {
                        let bytes = route
                            .elements
                            .checked_mul(value.dtype().itemsize())
                            .ok_or_else(|| err("redistribution byte overflow"))?;
                        Ok(CudaTransferRoute {
                            source_rank: route.source_rank,
                            source_device: route.source_device.clone(),
                            source_buffer: route.source_node.index() as u64,
                            source_element_offset: route.source_offset,
                            destination_rank: route.destination_rank,
                            destination_device: route.destination_device.clone(),
                            destination_buffer: route.destination_node.index() as u64,
                            destination_element_offset: route.destination_offset,
                            elements: route.elements,
                            bytes,
                            dtype: value.dtype(),
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                stages.push(CudaPlanStage::Transfer {
                    id,
                    action: trace.action.into(),
                    routes,
                    dependencies: previous.clone(),
                });
                previous = vec![id];
            }
        }
        let cache_key = format!(
            "sharded-cuda-plan:v1:{}:{}",
            value.layout().cache_key(),
            bindings
                .iter()
                .map(|b| format!("{}:{}", b.context.identity(), b.capability.sm()))
                .collect::<Vec<_>>()
                .join(",")
        );
        Ok(ShardedCudaPlan {
            graph_id: graph.id(),
            layout_key: value.layout().cache_key().into(),
            bindings: bindings
                .iter()
                .map(|b| (b.device.clone(), b.context.identity(), b.capability.sm()))
                .collect(),
            stages,
            diagnostics,
            cache_key,
        })
    }
    /// Builds the proven transfer-then-local executable composition directly
    /// from typed graph provenance.  The transfer destination is the only
    /// explicitly materialized computed input; no operation label or graph
    /// walk is used to discover substitutions.
    pub fn executable_fused(
        graph: &Graph,
        value: &ShardedGraphTensor,
        bindings: &[CudaPlanBinding],
    ) -> Result<ShardedCudaPlanComposition, Error> {
        if value.graph_id() != graph.id() {
            return Err(err("sharded tensor belongs to another graph"));
        }
        validate_bindings(value.layout().group(), bindings)?;
        // With one rank, redistribution is an identity layout transition.  The
        // normal retained local plan is the exact executable artifact; there
        // is no computed transfer output to substitute or materialize.
        if value.layout().group().len() == 1 {
            let local = Self::executable(graph, Self::build(graph, value, bindings)?, bindings)?;
            return Ok(ShardedCudaPlanComposition {
                plan: local,
                substitutions: vec![],
            });
        }
        let local_step = value
            .trace()
            .steps
            .last()
            .ok_or_else(|| err("fused plan requires a local provenance step"))?;
        if local_step.local_inputs.len() != value.nodes().len() {
            return Err(err("local provenance rank count mismatch"));
        }
        let mut substitutions = Vec::new();
        let mut local_stages = Vec::new();
        let mut diagnostics = Vec::new();
        for (rank, node) in value.nodes().iter().enumerate() {
            let provenance = local_step
                .local_inputs
                .get(rank)
                .ok_or_else(|| err("local provenance rank missing"))?;
            if provenance.rank != rank || provenance.consumer_local_node != *node {
                return Err(err("local provenance rank or consumer mismatch"));
            }
            let external = provenance
                .ordered_inputs
                .iter()
                .filter_map(|operand| operand.producer_redistribution_destination)
                .collect::<Vec<_>>();
            if external.len()
                != external
                    .iter()
                    .map(|node| node.index())
                    .collect::<BTreeSet<_>>()
                    .len()
            {
                return Err(err("duplicate redistribution destination provenance"));
            }
            let scheduled = schedule_with_external_materializations(graph, &[*node], &external)
                .map_err(|e| err(e.to_string()))?;
            let item = scheduled
                .items
                .first()
                .ok_or_else(|| err("local stage schedule missing"))?;
            item.validate_input_bindings()
                .map_err(|e| err(e.to_string()))?;
            if item.external_materializations != external {
                return Err(err("schedule external materialization provenance mismatch"));
            }
            if item.ordered_inputs().len() != provenance.ordered_inputs.len() {
                return Err(err("local provenance/ABI input count mismatch"));
            }
            for (operand, abi) in provenance.ordered_inputs.iter().zip(item.ordered_inputs()) {
                if abi.abi_index >= item.ordered_inputs().len() || !item.inputs.contains(&abi.desc)
                {
                    return Err(err("local provenance/ABI descriptor mismatch"));
                }
                if let Some(destination) = operand.producer_redistribution_destination {
                    if destination != abi.input_node || destination.index() as u64 != abi.desc.id {
                        return Err(err("redistribution destination ABI mismatch"));
                    }
                    substitutions.push(BufferSubstitution {
                        rank,
                        local_buffer: abi.desc.id,
                        transfer_buffer: destination.index() as u64,
                    });
                } else if operand.input_node != abi.input_node && abi.desc.view.is_none() {
                    // Static local shrink operands deliberately retain the
                    // original backing buffer in the ABI.  Any other node-id
                    // mismatch would lose the ordered provenance contract.
                    return Err(err("local provenance/ABI ordering or node mismatch"));
                }
            }
            let binding = &bindings[rank];
            let diagnostic =
                item.boundary
                    .as_ref()
                    .map(|boundary| CudaPlanDiagnostic::Unsupported {
                        node: node.index(),
                        reason: format!("schedule boundary: {boundary:?}"),
                    });
            let diagnostic = if diagnostic.is_none()
                && PtxRenderer::new(binding.capability.sm())
                    .and_then(|renderer| renderer.render(&item.kernel))
                    .is_err()
            {
                Some(CudaPlanDiagnostic::Unsupported {
                    node: node.index(),
                    reason: "current PTX renderer accepts only elementwise/select/cast; reductions are Phase 3B1 diagnostics".into(),
                })
            } else {
                diagnostic
            };
            if let Some(diagnostic) = diagnostic.clone() {
                diagnostics.push(diagnostic);
            }
            let source_key = format!("schedule:{}", item.cache_key);
            local_stages.push(CudaPlanStage::Local {
                id: rank,
                device: binding.device.clone(),
                owner_identity: binding.context.identity(),
                node: node.index(),
                shape: graph.shape(*node)?.clone(),
                dtype: graph.dtype(*node)?,
                inputs: item.inputs.iter().map(|desc| desc.id).collect(),
                external_materializations: external
                    .iter()
                    .map(|node| node.index() as u64)
                    .collect(),
                output: item.output.id,
                dependencies: vec![],
                source_key: source_key.clone(),
                module_key: format!(
                    "owner:{}:sm{}:{source_key}",
                    binding.context.identity(),
                    binding.capability.sm()
                ),
                diagnostic,
            });
        }
        let local_logical = ShardedCudaPlan {
            graph_id: graph.id(),
            layout_key: value.layout().cache_key().into(),
            bindings: bindings
                .iter()
                .map(|binding| {
                    (
                        binding.device.clone(),
                        binding.context.identity(),
                        binding.capability.sm(),
                    )
                })
                .collect(),
            stages: local_stages,
            diagnostics,
            cache_key: format!("sharded-cuda-local-fused:{}", value.layout().cache_key()),
        };
        let local = Self::executable(graph, local_logical, bindings)?;
        if substitutions.is_empty() {
            return Err(err("fused plan has no redistribution-produced local input"));
        }
        let transfer = transfer_from_provenance(graph, value, bindings, &substitutions)?;
        ShardedCudaPlanComposition::compose(&transfer, &local, substitutions)
    }
    /// Rehydrates only the exact local graph nodes named by the logical plan and verifies
    /// their schedule identity before retaining their rendered ABI. It never infers work
    /// from trace labels and performs no Driver operation.
    pub fn executable(
        graph: &Graph,
        logical: ShardedCudaPlan,
        bindings: &[CudaPlanBinding],
    ) -> Result<ExecutableShardedCudaPlan, Error> {
        if logical.graph_id != graph.id() || logical.bindings.len() != bindings.len() {
            return Err(err("logical plan graph or binding mismatch"));
        }
        let mut owners = Vec::with_capacity(bindings.len());
        for (record, binding) in logical.bindings.iter().zip(bindings) {
            if record.0 != binding.device
                || record.1 != binding.context.identity()
                || record.2 != binding.capability.sm()
            {
                return Err(err("logical plan owner/capability mismatch"));
            }
            owners.push(binding.context.clone());
        }
        let mut kernels = Vec::with_capacity(logical.stages.len());
        for stage in &logical.stages {
            match stage {
                CudaPlanStage::Local {
                    node,
                    owner_identity,
                    source_key,
                    diagnostic,
                    external_materializations,
                    ..
                } => {
                    let binding = bindings
                        .iter()
                        .find(|binding| binding.context.identity() == *owner_identity)
                        .ok_or_else(|| err("local stage owner missing"))?;
                    let materialized = external_materializations
                        .iter()
                        .map(|node| crate::NodeId::from_index(*node as usize))
                        .collect::<Vec<_>>();
                    let item = schedule_with_external_materializations(
                        graph,
                        &[crate::NodeId::from_index(*node)],
                        &materialized,
                    )
                    .map_err(|e| err(e.to_string()))?
                    .items
                    .into_iter()
                    .next()
                    .ok_or_else(|| err("local stage schedule missing"))?;
                    if source_key != &format!("schedule:{}", item.cache_key) {
                        return Err(err("local stage schedule identity mismatch"));
                    }
                    kernels.push(if diagnostic.is_none() {
                        Some(
                            PtxRenderer::new(binding.capability.sm())
                                .and_then(|renderer| renderer.render(&item.kernel))
                                .map_err(|e| err(e.to_string()))?,
                        )
                    } else {
                        None
                    });
                }
                _ => kernels.push(None),
            }
        }
        let mut buffers = Vec::new();
        for (stage_index, stage) in logical.stages.iter().enumerate() {
            if let CudaPlanStage::Local {
                device,
                owner_identity,
                node,
                external_materializations,
                ..
            } = stage
            {
                let rank = owners
                    .iter()
                    .position(|owner| owner.identity() == *owner_identity)
                    .ok_or_else(|| err("buffer owner missing"))?;
                let materialized = external_materializations
                    .iter()
                    .map(|node| crate::NodeId::from_index(*node as usize))
                    .collect::<Vec<_>>();
                let item = schedule_with_external_materializations(
                    graph,
                    &[crate::NodeId::from_index(*node)],
                    &materialized,
                )
                .map_err(|e| err(e.to_string()))?
                .items
                .into_iter()
                .next()
                .ok_or_else(|| err("local stage schedule missing"))?;
                for descriptor in item.inputs.iter().chain(std::iter::once(&item.output)) {
                    let buffer = descriptor.id;
                    let producer = (buffer == item.output.id).then_some(stage_index);
                    let bytes = descriptor.bytes;
                    if let Some(entry) = buffers.iter_mut().find(|entry: &&mut ExecutableBuffer| {
                        entry.rank == rank && entry.buffer == buffer
                    }) {
                        if entry.dtype != descriptor.dtype
                            || entry.shape != descriptor.shape
                            || entry.bytes != bytes
                        {
                            return Err(err("incompatible canonical buffer descriptor"));
                        }
                        entry.last_stage = stage_index;
                        entry.consumers.push(stage_index);
                        if producer.is_some() {
                            entry.producer = producer;
                            entry.role = ExecutableBufferRole::Output;
                        }
                    } else {
                        buffers.push(ExecutableBuffer {
                            rank,
                            device: device.clone(),
                            owner_identity: *owner_identity,
                            buffer,
                            dtype: descriptor.dtype,
                            shape: descriptor.shape.clone(),
                            bytes,
                            producer,
                            consumers: vec![stage_index],
                            first_stage: stage_index,
                            last_stage: stage_index,
                            role: if producer.is_some() {
                                ExecutableBufferRole::Output
                            } else {
                                ExecutableBufferRole::External
                            },
                        });
                    }
                }
            }
        }
        for stage in &logical.stages {
            if let CudaPlanStage::Transfer { routes, .. } = stage {
                for route in routes {
                    for (rank, device, buffer) in [
                        (route.source_rank, &route.source_device, route.source_buffer),
                        (
                            route.destination_rank,
                            &route.destination_device,
                            route.destination_buffer,
                        ),
                    ] {
                        if buffers
                            .iter()
                            .any(|entry| entry.rank == rank && entry.buffer == buffer)
                        {
                            continue;
                        }
                        let owner = logical
                            .bindings
                            .get(rank)
                            .ok_or_else(|| err("transfer rank outside bindings"))?;
                        let shape = graph
                            .shape(crate::NodeId::from_index(buffer as usize))?
                            .clone();
                        let dtype = graph.dtype(crate::NodeId::from_index(buffer as usize))?;
                        let bytes = shape
                            .numel()?
                            .checked_mul(dtype.itemsize())
                            .ok_or_else(|| err("transfer buffer byte overflow"))?;
                        buffers.push(ExecutableBuffer {
                            rank,
                            device: device.clone(),
                            owner_identity: owner.1,
                            buffer,
                            dtype,
                            shape,
                            bytes,
                            producer: None,
                            consumers: vec![],
                            first_stage: 0,
                            last_stage: logical.stages.len(),
                            role: ExecutableBufferRole::External,
                        });
                    }
                }
            }
        }
        Ok(ExecutableShardedCudaPlan {
            logical,
            owners,
            kernels,
            buffers,
        })
    }
}
fn transfer_from_provenance(
    graph: &Graph,
    value: &ShardedGraphTensor,
    bindings: &[CudaPlanBinding],
    substitutions: &[BufferSubstitution],
) -> Result<ExecutableShardedCudaPlan, Error> {
    let wanted = substitutions
        .iter()
        .map(|substitution| (substitution.rank, substitution.transfer_buffer))
        .collect::<BTreeSet<_>>();
    let trace = value
        .trace()
        .steps
        .iter()
        .find(|step| {
            let destinations = step
                .routes
                .iter()
                .map(|route| {
                    (
                        route.destination_rank,
                        route.destination_node.index() as u64,
                    )
                })
                .collect::<BTreeSet<_>>();
            !step.routes.is_empty() && wanted.is_subset(&destinations)
        })
        .ok_or_else(|| err("provenance redistribution routes are absent"))?;
    let mut routes = Vec::new();
    let mut buffers = BTreeMap::new();
    for route in &trace.routes {
        let bytes = route
            .elements
            .checked_mul(graph.dtype(route.source_node)?.itemsize())
            .ok_or_else(|| err("provenance route byte overflow"))?;
        let dtype = graph.dtype(route.source_node)?;
        if dtype != graph.dtype(route.destination_node)? {
            return Err(err("provenance route dtype mismatch"));
        }
        let source_shape = graph.shape(route.source_node)?.clone();
        let destination_shape = graph.shape(route.destination_node)?.clone();
        for (rank, device, node, shape, role, producer) in [
            (
                route.source_rank,
                route.source_device.clone(),
                route.source_node,
                source_shape,
                ExecutableBufferRole::External,
                None,
            ),
            (
                route.destination_rank,
                route.destination_device.clone(),
                route.destination_node,
                destination_shape,
                ExecutableBufferRole::Output,
                Some(0),
            ),
        ] {
            let binding = bindings
                .get(rank)
                .ok_or_else(|| err("route rank outside bindings"))?;
            if binding.device != device {
                return Err(err("provenance route device/rank mismatch"));
            }
            let bytes = shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| err("provenance buffer byte overflow"))?;
            let key = (rank, node.index() as u64);
            let entry = ExecutableBuffer {
                rank,
                device,
                owner_identity: binding.context.identity(),
                buffer: key.1,
                dtype,
                shape,
                bytes,
                producer,
                consumers: vec![0],
                first_stage: 0,
                last_stage: 0,
                role,
            };
            if let Some(existing) = buffers.get(&key) {
                if existing != &entry {
                    return Err(err("provenance transfer buffer descriptor mismatch"));
                }
            } else {
                buffers.insert(key, entry);
            }
        }
        routes.push(CudaTransferRoute {
            source_rank: route.source_rank,
            source_device: route.source_device.clone(),
            source_buffer: route.source_node.index() as u64,
            source_element_offset: route.source_offset,
            destination_rank: route.destination_rank,
            destination_device: route.destination_device.clone(),
            destination_buffer: route.destination_node.index() as u64,
            destination_element_offset: route.destination_offset,
            elements: route.elements,
            bytes,
            dtype,
        });
    }
    let logical = ShardedCudaPlan {
        graph_id: graph.id(),
        layout_key: value.layout().cache_key().into(),
        bindings: bindings
            .iter()
            .map(|binding| {
                (
                    binding.device.clone(),
                    binding.context.identity(),
                    binding.capability.sm(),
                )
            })
            .collect(),
        stages: vec![CudaPlanStage::Transfer {
            id: 0,
            action: "redistribute".into(),
            routes,
            dependencies: vec![],
        }],
        diagnostics: vec![],
        cache_key: format!(
            "sharded-cuda-provenance-transfer:{}",
            value.layout().cache_key()
        ),
    };
    Ok(ExecutableShardedCudaPlan {
        logical,
        owners: bindings
            .iter()
            .map(|binding| binding.context.clone())
            .collect(),
        kernels: vec![None],
        buffers: buffers.into_values().collect(),
    })
}
fn validate_bindings(group: &DeviceGroup, bindings: &[CudaPlanBinding]) -> Result<(), Error> {
    if bindings.len() != group.len() {
        return Err(err("CUDA bindings do not match device group length"));
    }
    if bindings
        .iter()
        .map(|b| &b.device)
        .collect::<BTreeSet<_>>()
        .len()
        != bindings.len()
        || bindings
            .iter()
            .map(|b| b.context.identity())
            .collect::<BTreeSet<_>>()
            .len()
            != bindings.len()
    {
        return Err(err(
            "CUDA plan bindings require distinct semantic devices and owners",
        ));
    }
    for (expected, actual) in group.devices().iter().zip(bindings) {
        if expected != &actual.device {
            return Err(err("CUDA plan binding device order does not match layout"));
        }
        if actual.context.device() != actual.capability.device {
            return Err(err("CUDA capability device does not match primary context"));
        }
    }
    Ok(())
}
fn collective_plan(
    group: &DeviceGroup,
    dtype: DType,
    layout: &crate::ShardLayout,
) -> Result<CollectivePlan, Error> {
    let n = layout.global_shape().numel()?;
    CollectivePlanner::plan(CollectiveRequest {
        group: group.clone(),
        kind: CollectiveKind::AllReduce {
            reduction: Reduction::Sum,
        },
        dtype,
        input_lengths: vec![n; group.len()],
    })
}
fn err(reason: impl Into<String>) -> Error {
    Error::Collective {
        reason: reason.into(),
    }
}
