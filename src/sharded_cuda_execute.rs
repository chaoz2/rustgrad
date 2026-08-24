//! Phase 3B1 local PTX realization for a validated executable sharded CUDA plan.
use crate::{
    ConcurrentPtxCache, CudaPlanStage, DType, Error, ExecutableBufferRole,
    ExecutableShardedCudaPlan, PrimaryBufferLease, PrimaryCudaAllocator, PtxBinding, Shape,
    ShardedCudaCompositionErrorKind as CompositionError,
    ShardedCudaCompositionField as CompositionField,
};
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::sync::Arc;

/// One explicit local-ABI input replacement by a transfer-produced buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferSubstitution {
    pub rank: usize,
    pub local_buffer: u64,
    pub transfer_buffer: u64,
}

/// An immutable, validated transfer-then-local executable composition.
pub struct ShardedCudaPlanComposition {
    pub plan: ExecutableShardedCudaPlan,
    pub substitutions: Vec<BufferSubstitution>,
}
impl ShardedCudaPlanComposition {
    pub fn compose(
        redistribution: &ExecutableShardedCudaPlan,
        local: &ExecutableShardedCudaPlan,
        substitutions: Vec<BufferSubstitution>,
    ) -> Result<Self, Error> {
        redistribution.validate()?;
        local.validate()?;
        if redistribution.logical.graph_id != local.logical.graph_id
            || redistribution.logical.bindings != local.logical.bindings
            || redistribution.owners.len() != local.owners.len()
        {
            return Err(err("composition graph or owner bindings mismatch"));
        }
        if redistribution
            .logical
            .stages
            .iter()
            .any(|stage| !matches!(stage, CudaPlanStage::Transfer { .. }))
            || local
                .logical
                .stages
                .iter()
                .any(|stage| !matches!(stage, CudaPlanStage::Local { .. }))
        {
            return Err(err(
                "composition requires transfer-only then local-only plans",
            ));
        }
        let composition_error = |kind| Error::ShardedCudaComposition { kind };
        let mut alias = BTreeMap::new();
        let mut destinations = BTreeSet::new();
        for substitution in &substitutions {
            let source = redistribution
                .buffers
                .iter()
                .find(|buffer| {
                    buffer.rank == substitution.rank
                        && buffer.buffer == substitution.transfer_buffer
                })
                .ok_or_else(|| {
                    composition_error(CompositionError::MissingTransferDestination {
                        rank: substitution.rank,
                        buffer: substitution.transfer_buffer,
                    })
                })?;
            let target = local
                .buffers
                .iter()
                .find(|buffer| {
                    buffer.rank == substitution.rank && buffer.buffer == substitution.local_buffer
                })
                .ok_or_else(|| {
                    composition_error(CompositionError::MissingLocalExternal {
                        rank: substitution.rank,
                        buffer: substitution.local_buffer,
                    })
                })?;
            if !matches!(target.role, ExecutableBufferRole::External) {
                return Err(composition_error(CompositionError::MissingLocalExternal {
                    rank: substitution.rank,
                    buffer: substitution.local_buffer,
                }));
            }
            let Some(producer) = source.producer else {
                return Err(composition_error(CompositionError::MissingProducer {
                    rank: substitution.rank,
                    buffer: substitution.transfer_buffer,
                }));
            };
            if !matches!(source.role, ExecutableBufferRole::Output)
                || !matches!(
                    redistribution.logical.stages.get(producer),
                    Some(CudaPlanStage::Transfer { .. })
                )
            {
                return Err(composition_error(
                    CompositionError::DestinationNotProducedByTransfer {
                        rank: substitution.rank,
                        buffer: substitution.transfer_buffer,
                    },
                ));
            }
            for (field, same) in [
                (CompositionField::Device, source.device == target.device),
                (
                    CompositionField::Owner,
                    source.owner_identity == target.owner_identity,
                ),
                (CompositionField::DType, source.dtype == target.dtype),
                (CompositionField::Shape, source.shape == target.shape),
                (CompositionField::Bytes, source.bytes == target.bytes),
            ] {
                if !same {
                    return Err(composition_error(CompositionError::DescriptorMismatch {
                        rank: substitution.rank,
                        local_buffer: substitution.local_buffer,
                        transfer_buffer: substitution.transfer_buffer,
                        field,
                    }));
                }
            }
            if alias
                .insert(
                    (substitution.rank, substitution.local_buffer),
                    substitution.transfer_buffer,
                )
                .is_some()
            {
                return Err(composition_error(
                    CompositionError::DuplicateLocalSubstitution {
                        rank: substitution.rank,
                        buffer: substitution.local_buffer,
                    },
                ));
            }
            if !destinations.insert((substitution.rank, substitution.transfer_buffer)) {
                return Err(composition_error(
                    CompositionError::DuplicateTransferDestination {
                        rank: substitution.rank,
                        buffer: substitution.transfer_buffer,
                    },
                ));
            }
        }
        if alias.is_empty() {
            return Err(err(
                "composition requires at least one explicit substitution",
            ));
        }
        let shift = redistribution.logical.stages.len();
        let mut stages = redistribution.logical.stages.clone();
        for stage in &local.logical.stages {
            let CudaPlanStage::Local {
                id,
                device,
                owner_identity,
                node,
                shape,
                dtype,
                inputs,
                external_materializations,
                output,
                source_key,
                module_key,
                diagnostic,
                dependencies,
                ..
            } = stage
            else {
                unreachable!()
            };
            stages.push(CudaPlanStage::Local {
                id: id + shift,
                device: device.clone(),
                owner_identity: *owner_identity,
                node: *node,
                shape: shape.clone(),
                dtype: *dtype,
                inputs: inputs.clone(),
                external_materializations: external_materializations.clone(),
                output: *output,
                source_key: source_key.clone(),
                module_key: module_key.clone(),
                diagnostic: diagnostic.clone(),
                dependencies: dependencies
                    .iter()
                    .map(|dependency| dependency + shift)
                    .chain(inputs.iter().filter_map(|buffer| {
                        let rank = local
                            .owners
                            .iter()
                            .position(|owner| owner.identity() == *owner_identity)?;
                        alias.get(&(rank, *buffer)).and_then(|transfer| {
                            redistribution
                                .buffers
                                .iter()
                                .find(|entry| entry.buffer == *transfer)
                                .and_then(|entry| entry.producer)
                        })
                    }))
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect(),
            });
        }
        validate_composition_dependencies(&stages)?;
        let mut buffers = redistribution.buffers.clone();
        for buffer in &local.buffers {
            if matches!(buffer.role, ExecutableBufferRole::External)
                && alias.contains_key(&(buffer.rank, buffer.buffer))
            {
                continue;
            }
            let mut buffer = buffer.clone();
            buffer.producer = buffer.producer.map(|stage| stage + shift);
            buffer.consumers = buffer
                .consumers
                .into_iter()
                .map(|stage| stage + shift)
                .collect();
            buffer.first_stage += shift;
            buffer.last_stage += shift;
            if buffers
                .iter()
                .any(|entry| entry.rank == buffer.rank && entry.buffer == buffer.buffer)
            {
                return Err(err("composition has duplicate canonical buffer identity"));
            }
            buffers.push(buffer);
        }
        Ok(Self {
            plan: ExecutableShardedCudaPlan {
                logical: crate::ShardedCudaPlan {
                    graph_id: redistribution.logical.graph_id,
                    layout_key: local.logical.layout_key.clone(),
                    bindings: local.logical.bindings.clone(),
                    stages,
                    diagnostics: [
                        redistribution.logical.diagnostics.clone(),
                        local.logical.diagnostics.clone(),
                    ]
                    .concat(),
                    cache_key: format!(
                        "compose:{}:{}",
                        redistribution.logical.cache_key, local.logical.cache_key
                    ),
                },
                owners: redistribution.owners.clone(),
                kernels: redistribution
                    .kernels
                    .iter()
                    .cloned()
                    .chain(local.kernels.iter().cloned())
                    .collect(),
                buffers,
            },
            substitutions,
        })
    }
}
fn validate_composition_dependencies(stages: &[CudaPlanStage]) -> Result<(), Error> {
    let ids: BTreeSet<_> = stages
        .iter()
        .map(|stage| match stage {
            CudaPlanStage::Local { id, .. }
            | CudaPlanStage::Transfer { id, .. }
            | CudaPlanStage::Collective { id, .. } => *id,
        })
        .collect();
    let mut incoming = BTreeMap::new();
    let mut outgoing: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for stage in stages {
        let (id, dependencies) = match stage {
            CudaPlanStage::Local {
                id, dependencies, ..
            }
            | CudaPlanStage::Transfer {
                id, dependencies, ..
            }
            | CudaPlanStage::Collective {
                id, dependencies, ..
            } => (*id, dependencies),
        };
        incoming.insert(id, dependencies.len());
        for dependency in dependencies {
            if !ids.contains(dependency) {
                return Err(Error::ShardedCudaComposition {
                    kind: CompositionError::UnknownDependency {
                        stage: id,
                        dependency: *dependency,
                    },
                });
            }
            outgoing.entry(*dependency).or_default().push(id);
        }
    }
    let mut ready: BTreeSet<_> = incoming
        .iter()
        .filter_map(|(&id, &count)| (count == 0).then_some(id))
        .collect();
    let mut visited = 0;
    while let Some(id) = ready.pop_first() {
        visited += 1;
        for dependent in outgoing.get(&id).into_iter().flatten() {
            let count = incoming.get_mut(dependent).unwrap();
            *count -= 1;
            if *count == 0 {
                ready.insert(*dependent);
            }
        }
    }
    if visited != stages.len() {
        return Err(Error::ShardedCudaComposition {
            kind: CompositionError::DependencyCycle {
                stages: incoming
                    .into_iter()
                    .filter_map(|(id, count)| (count > 0).then_some(id))
                    .collect(),
            },
        });
    }
    Ok(())
}

/// A zero-element logical binding. It deliberately has no device pointer or view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalZeroBuffer {
    pub owner_identity: usize,
    pub rank: usize,
    pub buffer: u64,
    pub dtype: DType,
    pub shape: Shape,
    pub generation: u64,
}
impl LogicalZeroBuffer {
    pub const fn new(
        owner_identity: usize,
        rank: usize,
        buffer: u64,
        dtype: DType,
        shape: Shape,
    ) -> Self {
        Self {
            owner_identity,
            rank,
            buffer,
            dtype,
            shape,
            generation: 0,
        }
    }
}
pub struct ShardedCudaExecutionEnvironment {
    pub external: BTreeMap<(usize, u64), PrimaryBufferLease>,
    pub zero_external: BTreeMap<(usize, u64), LogicalZeroBuffer>,
    caches: Vec<ConcurrentPtxCache>,
    allocators: Option<Vec<Arc<PrimaryCudaAllocator>>>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardedCudaExecutionTrace {
    pub stage: usize,
    pub action: &'static str,
    pub skipped: bool,
}
pub struct ShardedCudaExecutionResult {
    pub outputs: BTreeMap<(usize, u64), PrimaryBufferLease>,
    pub zero_outputs: BTreeMap<(usize, u64), LogicalZeroBuffer>,
    pub trace: Vec<ShardedCudaExecutionTrace>,
}
impl ShardedCudaExecutionEnvironment {
    pub fn new(external: BTreeMap<(usize, u64), PrimaryBufferLease>, owners: usize) -> Self {
        Self {
            external,
            zero_external: BTreeMap::new(),
            caches: (0..owners).map(|_| ConcurrentPtxCache::new()).collect(),
            allocators: None,
        }
    }
    pub fn with_logical_zeros(
        external: BTreeMap<(usize, u64), PrimaryBufferLease>,
        zero_external: BTreeMap<(usize, u64), LogicalZeroBuffer>,
        owners: usize,
    ) -> Self {
        Self {
            external,
            zero_external,
            caches: (0..owners).map(|_| ConcurrentPtxCache::new()).collect(),
            allocators: None,
        }
    }
    /// Uses exact owner-scoped pools for executor allocations and accounting.
    pub fn with_primary_allocators(
        external: BTreeMap<(usize, u64), PrimaryBufferLease>,
        zero_external: BTreeMap<(usize, u64), LogicalZeroBuffer>,
        allocators: Vec<Arc<PrimaryCudaAllocator>>,
    ) -> Self {
        let owners = allocators.len();
        Self {
            external,
            zero_external,
            caches: (0..owners).map(|_| ConcurrentPtxCache::new()).collect(),
            allocators: Some(allocators),
        }
    }
    pub fn execute(
        &mut self,
        plan: &ExecutableShardedCudaPlan,
    ) -> Result<ShardedCudaExecutionResult, Error> {
        self.execute_with_substitutions(plan, &BTreeMap::new())
    }
    pub fn execute_composed(
        &mut self,
        composition: &ShardedCudaPlanComposition,
    ) -> Result<ShardedCudaExecutionResult, Error> {
        let substitutions = composition
            .substitutions
            .iter()
            .map(|entry| ((entry.rank, entry.local_buffer), entry.transfer_buffer))
            .collect();
        self.execute_with_substitutions(&composition.plan, &substitutions)
    }
    fn execute_with_substitutions(
        &mut self,
        plan: &ExecutableShardedCudaPlan,
        substitutions: &BTreeMap<(usize, u64), u64>,
    ) -> Result<ShardedCudaExecutionResult, Error> {
        plan.validate()?;
        if let Some(allocators) = &self.allocators
            && (allocators.len() != plan.owners.len()
                || allocators
                    .iter()
                    .zip(&plan.owners)
                    .any(|(allocator, owner)| allocator.stats().owner_id != owner.identity()))
        {
            return Err(err("primary allocator bindings do not match plan owners"));
        }
        if plan.logical.stages.iter().any(|stage| {
            matches!(stage, CudaPlanStage::Collective { .. })
                || matches!(
                    stage,
                    CudaPlanStage::Local {
                        diagnostic: Some(_),
                        ..
                    }
                )
        }) {
            return Err(err(
                "Phase 3B1 rejects collective and diagnostic stages before execution",
            ));
        }
        let expected_external = plan
            .buffers
            .iter()
            .filter(|buffer| matches!(buffer.role, ExecutableBufferRole::External))
            .map(|buffer| (buffer.rank, buffer.buffer))
            .collect::<BTreeSet<_>>();
        let actual_external = self
            .external
            .keys()
            .chain(self.zero_external.keys())
            .copied()
            .collect::<BTreeSet<_>>();
        if actual_external != expected_external {
            return Err(err("external sharded CUDA bindings are missing or extra"));
        }
        let mut leases = std::mem::take(&mut self.external);
        let mut zeros = std::mem::take(&mut self.zero_external);
        let result = (|| -> Result<ShardedCudaExecutionResult, Error> {
            let mut trace = Vec::new();
            for buffer in &plan.buffers {
                let key = (buffer.rank, buffer.buffer);
                if matches!(buffer.role, ExecutableBufferRole::External) {
                    if buffer.bytes == 0 {
                        let zero = zeros
                            .get(&key)
                            .ok_or_else(|| err("missing logical zero binding"))?;
                        if zero.owner_identity != buffer.owner_identity
                            || zero.rank != buffer.rank
                            || zero.buffer != buffer.buffer
                            || zero.dtype != buffer.dtype
                            || zero.shape != buffer.shape
                        {
                            return Err(err("logical zero binding metadata mismatch"));
                        }
                    } else {
                        let lease = leases
                            .get(&key)
                            .ok_or_else(|| err("missing external sharded CUDA lease"))?;
                        let (owner, bytes, _, _) =
                            lease.execution_metadata().map_err(|e| err(e.to_string()))?;
                        if owner != buffer.owner_identity || bytes < buffer.bytes {
                            return Err(err("external lease owner or bytes mismatch"));
                        }
                    }
                } else if buffer.bytes > 0 {
                    let allocator = self
                        .allocators
                        .as_ref()
                        .map(|allocators| allocators[buffer.rank].clone())
                        .unwrap_or_else(|| plan.owners[buffer.rank].allocator());
                    leases.insert(
                        key,
                        allocator
                            .allocate(NonZeroUsize::new(buffer.bytes).unwrap())
                            .map_err(|e| err(e.to_string()))?,
                    );
                } else {
                    zeros.insert(
                        key,
                        LogicalZeroBuffer::new(
                            buffer.owner_identity,
                            buffer.rank,
                            buffer.buffer,
                            buffer.dtype,
                            buffer.shape.clone(),
                        ),
                    );
                }
            }
            for (index, stage) in plan.logical.stages.iter().enumerate() {
                let CudaPlanStage::Local {
                    id, owner_identity, ..
                } = stage
                else {
                    if let CudaPlanStage::Transfer { id, routes, .. } = stage {
                        for route in routes {
                            if route.bytes == 0 {
                                continue;
                            }
                            let source = leases
                                .get(&(route.source_rank, route.source_buffer))
                                .ok_or_else(|| err("missing peer source lease"))?;
                            let destination = leases
                                .get(&(route.destination_rank, route.destination_buffer))
                                .ok_or_else(|| err("missing peer destination lease"))?;
                            if route.source_rank == route.destination_rank {
                                let destination_view = destination
                                    .view()
                                    .map_err(|e| err(format!("transfer {id}: {e}")))?;
                                let source_view = source
                                    .view()
                                    .map_err(|e| err(format!("transfer {id}: {e}")))?;
                                let stream = plan.owners[route.destination_rank]
                                    .stream()
                                    .map_err(|e| err(e.to_string()))?;
                                let mut transfer = destination_view
                                    .copy_from_view_async(
                                        route
                                            .destination_element_offset
                                            .checked_mul(route.dtype.itemsize())
                                            .ok_or_else(|| err("destination offset overflow"))?,
                                        &source_view,
                                        route
                                            .source_element_offset
                                            .checked_mul(route.dtype.itemsize())
                                            .ok_or_else(|| err("source offset overflow"))?,
                                        route.bytes,
                                        &stream,
                                    )
                                    .map_err(|e| err(format!("transfer {id}: {e}")))?;
                                transfer
                                    .wait()
                                    .map_err(|e| err(format!("transfer {id}: {e}")))?;
                                continue;
                            }
                            let peer = plan.owners[route.source_rank]
                                .peer_access_to(&plan.owners[route.destination_rank])
                                .map_err(|e| err(format!("transfer {id}: {e}")))?;
                            let stream = plan.owners[route.destination_rank]
                                .stream()
                                .map_err(|e| err(e.to_string()))?;
                            let mut transfer = destination
                                .copy_from_peer_async(
                                    route
                                        .destination_element_offset
                                        .checked_mul(route.dtype.itemsize())
                                        .ok_or_else(|| err("destination offset overflow"))?,
                                    &peer,
                                    source,
                                    route
                                        .source_element_offset
                                        .checked_mul(route.dtype.itemsize())
                                        .ok_or_else(|| err("source offset overflow"))?,
                                    route.bytes,
                                    &stream,
                                )
                                .map_err(|e| err(format!("transfer {id}: {e}")))?;
                            transfer
                                .wait()
                                .map_err(|e| err(format!("transfer {id}: {e}")))?;
                        }
                        trace.push(ShardedCudaExecutionTrace {
                            stage: *id,
                            action: "transfer",
                            skipped: routes.iter().all(|route| route.bytes == 0),
                        });
                        continue;
                    }
                    unreachable!()
                };
                let rendered = plan.kernels[index]
                    .as_ref()
                    .ok_or_else(|| err("missing retained PTX artifact"))?;
                let rank = plan
                    .owners
                    .iter()
                    .position(|owner| owner.identity() == *owner_identity)
                    .ok_or_else(|| err("stage owner missing"))?;
                if rendered.extent == 0 {
                    trace.push(ShardedCudaExecutionTrace {
                        stage: *id,
                        action: "local",
                        skipped: true,
                    });
                    continue;
                }
                let stream = plan.owners[rank].stream().map_err(|e| err(e.to_string()))?;
                let kernel = self.caches[rank]
                    .get_or_load(&plan.owners[rank], rendered.clone(), 256)
                    .map_err(|e| err(e.to_string()))?;
                let views = rendered
                    .buffers
                    .iter()
                    .map(|abi| {
                        let lease = leases
                            .get(&(
                                rank,
                                substitutions
                                    .get(&(rank, abi.id))
                                    .copied()
                                    .unwrap_or(abi.id),
                            ))
                            .ok_or_else(|| err("missing ABI lease"))?;
                        Ok(PtxBinding {
                            buffer: lease.view().map_err(|e| err(e.to_string()))?,
                            dtype: abi.dtype,
                            mutable: abi.mutable,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                kernel
                    .launch(&stream, &views, true)
                    .map_err(|e| err(format!("stage {id}: {e}")))?;
                trace.push(ShardedCudaExecutionTrace {
                    stage: *id,
                    action: "local",
                    skipped: false,
                });
            }
            let mut outputs = BTreeMap::new();
            let mut zero_outputs = BTreeMap::new();
            for buffer in &plan.buffers {
                if matches!(buffer.role, ExecutableBufferRole::Output)
                    && let Some(lease) = leases.remove(&(buffer.rank, buffer.buffer))
                {
                    outputs.insert((buffer.rank, buffer.buffer), lease);
                }
                if matches!(buffer.role, ExecutableBufferRole::Output)
                    && let Some(zero) = zeros.remove(&(buffer.rank, buffer.buffer))
                {
                    zero_outputs.insert((buffer.rank, buffer.buffer), zero);
                }
            }
            Ok(ShardedCudaExecutionResult {
                outputs,
                zero_outputs,
                trace,
            })
        })();
        if result.is_err() {
            for buffer in &plan.buffers {
                if matches!(buffer.role, ExecutableBufferRole::Output) {
                    leases.remove(&(buffer.rank, buffer.buffer));
                    zeros.remove(&(buffer.rank, buffer.buffer));
                }
            }
        }
        self.external = leases;
        self.zero_external = zeros;
        result
    }
}
fn err(reason: impl Into<String>) -> Error {
    Error::Collective {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collective::DeviceGroup;
    use crate::sharding::executable_redistribution_plan;
    use crate::{
        Backend, CpuBackend, CudaPlanDiagnostic, CudaPlanStage, CudaTransferRoute, DType, DeviceId,
        Driver, ExecutableBuffer, Graph, PtxRenderer, Shape, ShardedCudaPlan, Storage, TensorData,
        lower_graph_elementwise,
    };
    use crate::{BinaryOp, CudaPlanBinding, ShardedCudaPlanner};
    use std::collections::HashMap;
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    #[test]
    fn executor_runs_retained_generic_ptx_against_owner_scoped_mock_bytes() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let mut graph = Graph::new();
        let left = graph.input("left", [2, 2]);
        let right = graph.input("right", [1, 2]);
        let output = graph.binary(crate::BinaryOp::Add, left, right).unwrap();
        let rendered = PtxRenderer::new(80)
            .unwrap()
            .render(&lower_graph_elementwise(&graph, output).unwrap())
            .unwrap();
        let device = crate::collective::DeviceId::new("CUDA:0").unwrap();
        let owner = primary.identity();
        let logical = ShardedCudaPlan {
            graph_id: graph.id(),
            layout_key: "test".into(),
            bindings: vec![(device.clone(), owner, 80)],
            stages: vec![CudaPlanStage::Local {
                id: 0,
                device: device.clone(),
                owner_identity: owner,
                node: output.index(),
                shape: Shape::new(vec![2, 2]),
                dtype: DType::F32,
                inputs: vec![left.index() as u64, right.index() as u64],
                external_materializations: vec![],
                output: output.index() as u64,
                dependencies: vec![],
                source_key: rendered.cache_key.clone(),
                module_key: rendered.cache_key.clone(),
                diagnostic: None,
            }],
            diagnostics: vec![],
            cache_key: "test".into(),
        };
        let buffer = |id: u64, shape: Shape, role: ExecutableBufferRole| ExecutableBuffer {
            rank: 0,
            device: device.clone(),
            owner_identity: owner,
            buffer: id,
            dtype: DType::F32,
            bytes: shape.numel().unwrap() * DType::F32.itemsize(),
            shape,
            producer: None,
            consumers: vec![0],
            first_stage: 0,
            last_stage: 0,
            role,
        };
        let mut plan = ExecutableShardedCudaPlan {
            logical,
            owners: vec![primary.clone()],
            kernels: vec![Some(rendered)],
            buffers: vec![
                buffer(
                    left.index() as u64,
                    Shape::new(vec![2, 2]),
                    ExecutableBufferRole::External,
                ),
                buffer(
                    right.index() as u64,
                    Shape::new(vec![1, 2]),
                    ExecutableBufferRole::External,
                ),
                buffer(
                    output.index() as u64,
                    Shape::new(vec![2, 2]),
                    ExecutableBufferRole::Output,
                ),
            ],
        };
        let pool = primary.allocator();
        let left_lease = pool.allocate(NonZeroUsize::new(16).unwrap()).unwrap();
        let right_lease = pool.allocate(NonZeroUsize::new(8).unwrap()).unwrap();
        left_lease
            .view()
            .unwrap()
            .copy_from(
                0,
                &[0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64],
            )
            .unwrap();
        right_lease
            .view()
            .unwrap()
            .copy_from(0, &[0, 0, 32, 65, 0, 0, 160, 65])
            .unwrap();
        let mut environment = ShardedCudaExecutionEnvironment::new(
            BTreeMap::from([
                ((0, left.index() as u64), left_lease),
                ((0, right.index() as u64), right_lease),
            ]),
            1,
        );
        {
            let CudaPlanStage::Local { diagnostic, .. } = &mut plan.logical.stages[0] else {
                unreachable!();
            };
            *diagnostic = Some(CudaPlanDiagnostic::Unsupported {
                node: output.index(),
                reason: "test diagnostic".into(),
            });
        }
        let before = mock.calls().len();
        assert!(environment.execute(&plan).is_err());
        assert_eq!(
            mock.calls().len(),
            before,
            "diagnostics reject before Driver work"
        );
        let CudaPlanStage::Local { diagnostic, .. } = &mut plan.logical.stages[0] else {
            unreachable!();
        };
        *diagnostic = None;
        mock.set_launch_result(2);
        let failed = environment.execute(&plan).err().unwrap();
        assert!(failed.to_string().contains("stage 0"));
        assert_eq!(
            environment.external.len(),
            2,
            "failed output allocation is not rebound"
        );
        mock.set_launch_result(0);
        let result = environment.execute(&plan).unwrap();
        assert_eq!(
            result.trace,
            vec![ShardedCudaExecutionTrace {
                stage: 0,
                action: "local",
                skipped: false
            }]
        );
        let output = result.outputs.get(&(0, output.index() as u64)).unwrap();
        let mut bytes = vec![0; 16];
        output.view().unwrap().copy_to(0, &mut bytes).unwrap();
        assert_eq!(
            bytes,
            &[0, 0, 48, 65, 0, 0, 176, 65, 0, 0, 80, 65, 0, 0, 192, 65]
        );
        let before = mock.calls().len();
        environment
            .external
            .insert((0, 99), result.outputs.into_values().next().unwrap());
        assert!(environment.execute(&plan).is_err());
        assert_eq!(
            mock.calls().len(),
            before,
            "extra binding is rejected before Driver work"
        );
        assert_eq!(mock.generic_kernel_count(), 1);
    }

    #[test]
    fn executor_routes_same_owner_and_peer_bytes_in_deterministic_order() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let retain = |ordinal| {
            driver
                .device(DeviceId(ordinal))
                .unwrap()
                .retain_primary_context()
                .unwrap()
        };
        let first = retain(0);
        let second = retain(1);
        assert_ne!(first.identity(), second.identity());
        let device0 = crate::collective::DeviceId::new("CUDA:0").unwrap();
        let device1 = crate::collective::DeviceId::new("CUDA:1").unwrap();
        let route = |rank, device, buffer| CudaTransferRoute {
            source_rank: 0,
            source_device: device0.clone(),
            source_buffer: 10,
            source_element_offset: 0,
            destination_rank: rank,
            destination_device: device,
            destination_buffer: buffer,
            destination_element_offset: 0,
            elements: 2,
            bytes: 8,
            dtype: DType::F32,
        };
        let logical = ShardedCudaPlan {
            graph_id: 7,
            layout_key: "route-test".into(),
            bindings: vec![
                (device0.clone(), first.identity(), 80),
                (device1.clone(), second.identity(), 80),
            ],
            stages: vec![CudaPlanStage::Transfer {
                id: 0,
                action: "redistribute".into(),
                routes: vec![
                    CudaTransferRoute {
                        source_rank: 0,
                        source_device: device0.clone(),
                        source_buffer: 10,
                        source_element_offset: 0,
                        destination_rank: 0,
                        destination_device: device0.clone(),
                        destination_buffer: 11,
                        destination_element_offset: 0,
                        elements: 0,
                        bytes: 0,
                        dtype: DType::F32,
                    },
                    route(0, device0.clone(), 11),
                    route(1, device1.clone(), 20),
                ],
                dependencies: vec![],
            }],
            diagnostics: vec![],
            cache_key: "route-test".into(),
        };
        let buffer = |rank, device, owner, id| ExecutableBuffer {
            rank,
            device,
            owner_identity: owner,
            buffer: id,
            dtype: DType::F32,
            shape: Shape::new(vec![2]),
            bytes: 8,
            producer: None,
            consumers: vec![0],
            first_stage: 0,
            last_stage: 0,
            role: ExecutableBufferRole::External,
        };
        let plan = ExecutableShardedCudaPlan {
            logical,
            owners: vec![first.clone(), second.clone()],
            kernels: vec![None],
            buffers: vec![
                buffer(0, device0.clone(), first.identity(), 10),
                buffer(0, device0, first.identity(), 11),
                buffer(1, device1, second.identity(), 20),
            ],
        };
        let source = first
            .allocator()
            .allocate(NonZeroUsize::new(8).unwrap())
            .unwrap();
        let local = first
            .allocator()
            .allocate(NonZeroUsize::new(8).unwrap())
            .unwrap();
        let peer = second
            .allocator()
            .allocate(NonZeroUsize::new(8).unwrap())
            .unwrap();
        source
            .view()
            .unwrap()
            .copy_from(0, &[0, 0, 128, 63, 0, 0, 32, 64])
            .unwrap();
        let local_desc = mock
            .allocation_descriptor(first.owner(), local.view().unwrap().device_ptr().unwrap())
            .unwrap();
        let peer_desc = mock
            .allocation_descriptor(second.owner(), peer.view().unwrap().device_ptr().unwrap())
            .unwrap();
        let mut environment = ShardedCudaExecutionEnvironment::new(
            BTreeMap::from([((0, 10), source), ((0, 11), local), ((1, 20), peer)]),
            2,
        );
        let result = environment.execute(&plan).unwrap();
        assert_eq!(
            result.trace,
            vec![ShardedCudaExecutionTrace {
                stage: 0,
                action: "transfer",
                skipped: false
            }]
        );
        let expected = vec![0, 0, 128, 63, 0, 0, 32, 64];
        assert_eq!(
            mock.allocation_snapshot(first.owner(), local_desc).unwrap()[..8],
            expected
        );
        assert_eq!(
            mock.allocation_snapshot(second.owner(), peer_desc).unwrap()[..8],
            expected
        );
        let calls = mock.calls();
        assert!(
            calls.iter().position(|call| *call == "dtod_async").unwrap()
                < calls.iter().position(|call| *call == "peer_copy").unwrap()
        );
    }

    #[test]
    fn executor_keeps_zero_bindings_logical_and_never_requests_a_pointer() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let device = crate::collective::DeviceId::new("CUDA:0").unwrap();
        let empty = Shape::new(vec![0]);
        let logical = ShardedCudaPlan {
            graph_id: 0,
            layout_key: "zero".into(),
            bindings: vec![(device.clone(), primary.identity(), 80)],
            stages: vec![CudaPlanStage::Transfer {
                id: 0,
                action: "redistribute".into(),
                routes: vec![CudaTransferRoute {
                    source_rank: 0,
                    source_device: device.clone(),
                    source_buffer: 1,
                    source_element_offset: 0,
                    destination_rank: 0,
                    destination_device: device.clone(),
                    destination_buffer: 2,
                    destination_element_offset: 0,
                    elements: 0,
                    bytes: 0,
                    dtype: DType::F32,
                }],
                dependencies: vec![],
            }],
            diagnostics: vec![],
            cache_key: "zero".into(),
        };
        let buffer = |id, role| ExecutableBuffer {
            rank: 0,
            device: device.clone(),
            owner_identity: primary.identity(),
            buffer: id,
            dtype: DType::F32,
            shape: empty.clone(),
            bytes: 0,
            producer: matches!(role, ExecutableBufferRole::Output).then_some(0),
            consumers: vec![],
            first_stage: 0,
            last_stage: 0,
            role,
        };
        let plan = ExecutableShardedCudaPlan {
            logical,
            owners: vec![primary.clone()],
            kernels: vec![None],
            buffers: vec![
                buffer(1, ExecutableBufferRole::External),
                buffer(2, ExecutableBufferRole::Output),
            ],
        };
        let zero = LogicalZeroBuffer::new(primary.identity(), 0, 1, DType::F32, empty.clone());
        let before = mock.calls().len();
        let result = ShardedCudaExecutionEnvironment::with_logical_zeros(
            BTreeMap::new(),
            BTreeMap::from([((0, 1), zero)]),
            1,
        )
        .execute(&plan)
        .unwrap();
        assert_eq!(
            mock.calls().len(),
            before,
            "zero work makes no Driver calls"
        );
        assert!(result.outputs.is_empty());
        assert_eq!(
            result.zero_outputs.get(&(0, 2)),
            Some(&LogicalZeroBuffer::new(
                primary.identity(),
                0,
                2,
                DType::F32,
                empty
            ))
        );
        assert!(result.trace[0].skipped);
    }

    #[test]
    fn graph_transfer_then_local_add_composes_without_rebinding_destination_bytes() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let owners = (0..2)
            .map(|ordinal| {
                let device = driver.device(DeviceId(ordinal)).unwrap();
                let capability = device.capability().unwrap();
                (device.retain_primary_context().unwrap(), capability)
            })
            .collect::<Vec<_>>();
        let group = DeviceGroup::new(
            (0..2).map(|rank| crate::collective::DeviceId::new(format!("CUDA:{rank}")).unwrap()),
        )
        .unwrap();
        let bindings = owners
            .iter()
            .enumerate()
            .map(|(rank, (owner, capability))| CudaPlanBinding {
                device: group.devices()[rank].clone(),
                capability: capability.clone(),
                context: owner.clone(),
            })
            .collect::<Vec<_>>();
        let mut graph = Graph::new();
        let source_input = graph.input_dtype("source", [4, 2], DType::F32);
        let source = graph
            .shard_node(source_input, group.clone(), Some(0))
            .unwrap();
        let replicated = graph
            .redistribute_sharded(&source, group.clone(), None)
            .unwrap();
        let mut transfer = executable_redistribution_plan(&source, &replicated, &bindings).unwrap();

        let local_input = graph.input_dtype("replicated_input", [4, 2], DType::F32);
        let addend_input = graph.input_dtype("addend", [4, 2], DType::F32);
        let local_lhs = graph.replicate_node(local_input, group.clone()).unwrap();
        let local_rhs = graph.replicate_node(addend_input, group.clone()).unwrap();
        let local_value = graph
            .sharded_binary(&local_lhs, &local_rhs, BinaryOp::Add)
            .unwrap();
        let local_logical = ShardedCudaPlanner::build(&graph, &local_value, &bindings).unwrap();
        assert!(
            local_logical.diagnostics.is_empty(),
            "{:#?}",
            local_logical.diagnostics
        );
        let mut local = ShardedCudaPlanner::executable(&graph, local_logical, &bindings).unwrap();
        let duplicate = BufferSubstitution {
            rank: 0,
            local_buffer: local_input.index() as u64,
            transfer_buffer: replicated.nodes()[0].index() as u64,
        };
        let pools = owners
            .iter()
            .map(|(owner, _)| owner.allocator())
            .collect::<Vec<_>>();
        let preflight_stats = pools.iter().map(|pool| pool.stats()).collect::<Vec<_>>();
        let before = mock.calls().len();
        assert!(matches!(
            ShardedCudaPlanComposition::compose(
                &transfer,
                &local,
                vec![duplicate.clone(), duplicate.clone()],
            ),
            Err(Error::ShardedCudaComposition {
                kind: CompositionError::DuplicateLocalSubstitution { rank: 0, .. }
            })
        ));
        assert_eq!(
            mock.calls().len(),
            before,
            "invalid substitutions have no Driver work"
        );
        let no_work = |result: Result<ShardedCudaPlanComposition, Error>, expected| {
            match result {
                Err(Error::ShardedCudaComposition { kind }) => assert_eq!(kind, expected),
                _ => panic!("expected composition error {expected:?}"),
            }
            assert_eq!(
                mock.calls().len(),
                before,
                "composition preflight has no Driver work"
            );
            assert_eq!(
                pools.iter().map(|pool| pool.stats()).collect::<Vec<_>>(),
                preflight_stats,
                "composition preflight does not mutate pool accounting"
            );
        };
        no_work(
            ShardedCudaPlanComposition::compose(
                &transfer,
                &local,
                vec![BufferSubstitution {
                    rank: 9,
                    local_buffer: duplicate.local_buffer,
                    transfer_buffer: duplicate.transfer_buffer,
                }],
            ),
            CompositionError::MissingTransferDestination {
                rank: 9,
                buffer: duplicate.transfer_buffer,
            },
        );
        no_work(
            ShardedCudaPlanComposition::compose(
                &transfer,
                &local,
                vec![BufferSubstitution {
                    rank: 0,
                    local_buffer: u64::MAX,
                    transfer_buffer: duplicate.transfer_buffer,
                }],
            ),
            CompositionError::MissingLocalExternal {
                rank: 0,
                buffer: u64::MAX,
            },
        );
        let target_index = local
            .buffers
            .iter()
            .position(|buffer| buffer.rank == 0 && buffer.buffer == duplicate.local_buffer)
            .unwrap();
        let original_dtype = local.buffers[target_index].dtype;
        local.buffers[target_index].dtype = DType::I32;
        no_work(
            ShardedCudaPlanComposition::compose(&transfer, &local, vec![duplicate.clone()]),
            CompositionError::DescriptorMismatch {
                rank: 0,
                local_buffer: duplicate.local_buffer,
                transfer_buffer: duplicate.transfer_buffer,
                field: CompositionField::DType,
            },
        );
        local.buffers[target_index].dtype = original_dtype;
        let original_device = local.buffers[target_index].device.clone();
        local.buffers[target_index].device =
            crate::collective::DeviceId::new("CUDA:mismatch").unwrap();
        no_work(
            ShardedCudaPlanComposition::compose(&transfer, &local, vec![duplicate.clone()]),
            CompositionError::DescriptorMismatch {
                rank: 0,
                local_buffer: duplicate.local_buffer,
                transfer_buffer: duplicate.transfer_buffer,
                field: CompositionField::Device,
            },
        );
        local.buffers[target_index].device = original_device;
        let original_owner = local.buffers[target_index].owner_identity;
        local.buffers[target_index].owner_identity = usize::MAX;
        no_work(
            ShardedCudaPlanComposition::compose(&transfer, &local, vec![duplicate.clone()]),
            CompositionError::DescriptorMismatch {
                rank: 0,
                local_buffer: duplicate.local_buffer,
                transfer_buffer: duplicate.transfer_buffer,
                field: CompositionField::Owner,
            },
        );
        local.buffers[target_index].owner_identity = original_owner;
        let original_shape = local.buffers[target_index].shape.clone();
        local.buffers[target_index].shape = Shape::new(vec![8]);
        no_work(
            ShardedCudaPlanComposition::compose(&transfer, &local, vec![duplicate.clone()]),
            CompositionError::DescriptorMismatch {
                rank: 0,
                local_buffer: duplicate.local_buffer,
                transfer_buffer: duplicate.transfer_buffer,
                field: CompositionField::Shape,
            },
        );
        local.buffers[target_index].shape = original_shape;
        let original_bytes = local.buffers[target_index].bytes;
        local.buffers[target_index].bytes += 1;
        no_work(
            ShardedCudaPlanComposition::compose(&transfer, &local, vec![duplicate.clone()]),
            CompositionError::DescriptorMismatch {
                rank: 0,
                local_buffer: duplicate.local_buffer,
                transfer_buffer: duplicate.transfer_buffer,
                field: CompositionField::Bytes,
            },
        );
        local.buffers[target_index].bytes = original_bytes;
        let source_index = transfer
            .buffers
            .iter()
            .position(|buffer| buffer.rank == 0 && buffer.buffer == duplicate.transfer_buffer)
            .unwrap();
        let producer = transfer.buffers[source_index].producer.take();
        no_work(
            ShardedCudaPlanComposition::compose(&transfer, &local, vec![duplicate.clone()]),
            CompositionError::MissingProducer {
                rank: 0,
                buffer: duplicate.transfer_buffer,
            },
        );
        transfer.buffers[source_index].producer = producer;
        if let CudaPlanStage::Transfer { dependencies, .. } = &mut transfer.logical.stages[0] {
            dependencies.push(0);
        }
        no_work(
            ShardedCudaPlanComposition::compose(&transfer, &local, vec![duplicate.clone()]),
            CompositionError::DependencyCycle {
                stages: vec![0, 1, 2],
            },
        );
        if let CudaPlanStage::Transfer { dependencies, .. } = &mut transfer.logical.stages[0] {
            dependencies.clear();
        }
        let composition = ShardedCudaPlanComposition::compose(
            &transfer,
            &local,
            (0..2)
                .map(|rank| BufferSubstitution {
                    rank,
                    local_buffer: local_input.index() as u64,
                    transfer_buffer: replicated.nodes()[rank].index() as u64,
                })
                .collect(),
        )
        .unwrap();
        assert_eq!(composition.plan.logical.stages.len(), 3);
        assert!(
            composition.plan.logical.stages[1..]
                .iter()
                .all(|stage| match stage {
                    CudaPlanStage::Local { dependencies, .. } => dependencies.contains(&0),
                    _ => false,
                })
        );

        let source_bytes = TensorData::new([4, 2], (0..8).map(|x| x as f32).collect())
            .unwrap()
            .to_le_bytes()
            .unwrap();
        let addend_bytes = TensorData::new([4, 2], vec![10.0; 8])
            .unwrap()
            .to_le_bytes()
            .unwrap();
        let baseline = pools.iter().map(|pool| pool.stats()).collect::<Vec<_>>();
        assert!(baseline.iter().all(|stats| stats.logical_leased_bytes == 0));
        let mut external = BTreeMap::new();
        for (rank, pool) in pools.iter().enumerate() {
            let shard = &source_bytes[rank * 16..(rank + 1) * 16];
            for (buffer, bytes) in [
                (source.nodes()[rank].index() as u64, shard),
                (addend_input.index() as u64, addend_bytes.as_slice()),
            ] {
                let lease = pool
                    .allocate(NonZeroUsize::new(bytes.len()).unwrap())
                    .unwrap();
                lease.view().unwrap().copy_from(0, bytes).unwrap();
                external.insert((rank, buffer), lease);
            }
        }
        let after_external = pools.iter().map(|pool| pool.stats()).collect::<Vec<_>>();
        assert!(
            after_external
                .iter()
                .all(|stats| stats.logical_leased_bytes == 48)
        );
        assert!(
            pools
                .iter()
                .zip(&after_external)
                .all(|(pool, stats)| pool.stats().pool_id == stats.pool_id)
        );
        let mut environment = ShardedCudaExecutionEnvironment::with_primary_allocators(
            external,
            BTreeMap::new(),
            pools.clone(),
        );
        let result = environment.execute_composed(&composition).unwrap();
        assert_eq!(
            result
                .trace
                .iter()
                .map(|entry| entry.action)
                .collect::<Vec<_>>(),
            vec!["transfer", "local", "local"]
        );
        let expected = TensorData::new([4, 2], (10..18).map(|x| x as f32).collect())
            .unwrap()
            .to_le_bytes()
            .unwrap();
        for rank in 0..2 {
            let output = result
                .outputs
                .get(&(rank, local_value.nodes()[rank].index() as u64))
                .unwrap();
            let mut actual = vec![0; expected.len()];
            output.view().unwrap().copy_to(0, &mut actual).unwrap();
            assert_eq!(actual, expected, "rank {rank}");
        }
        assert!(mock.calls().contains(&"dtod_async"));
        assert!(mock.calls().contains(&"peer_copy"));
        assert_eq!(mock.generic_kernel_count(), 2);
        assert!(
            pools
                .iter()
                .all(|pool| pool.stats().logical_leased_bytes == 112)
        );
        drop(result);
        assert!(
            pools
                .iter()
                .all(|pool| pool.stats().logical_leased_bytes == 48)
        );
        mock.fail_peer_after(0, 2);
        let Err(failed) = environment.execute_composed(&composition) else {
            panic!("injected transfer failure unexpectedly succeeded")
        };
        assert!(failed.to_string().contains("transfer 0"));
        assert_eq!(
            environment.external.len(),
            4,
            "transfer failure restores true externals"
        );
        assert!(pools.iter().all(|pool| {
            let stats = pool.stats();
            stats.logical_leased_bytes == 48 && stats.peak_in_use_bytes >= 112
        }));
        let retry = environment.execute_composed(&composition).unwrap();
        drop(retry);
        mock.set_launch_result(2);
        let Err(failed) = environment.execute_composed(&composition) else {
            panic!("injected local launch failure unexpectedly succeeded")
        };
        assert!(failed.to_string().contains("stage 1"));
        assert_eq!(
            environment.external.len(),
            4,
            "local failure restores true externals"
        );
        assert!(
            pools
                .iter()
                .all(|pool| pool.stats().logical_leased_bytes == 48)
        );
        mock.set_launch_result(0);
        let final_retry = environment.execute_composed(&composition).unwrap();
        assert_eq!(final_retry.trace.len(), 3);
    }

    #[test]
    fn planner_fuses_graph_redistribution_into_local_add_from_provenance() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let owners = (0..2)
            .map(|ordinal| {
                let device = driver.device(DeviceId(ordinal)).unwrap();
                let capability = device.capability().unwrap();
                (device.retain_primary_context().unwrap(), capability)
            })
            .collect::<Vec<_>>();
        let group = DeviceGroup::new(
            (0..2).map(|rank| crate::collective::DeviceId::new(format!("CUDA:{rank}")).unwrap()),
        )
        .unwrap();
        let bindings = owners
            .iter()
            .enumerate()
            .map(|(rank, (owner, capability))| CudaPlanBinding {
                device: group.devices()[rank].clone(),
                capability: capability.clone(),
                context: owner.clone(),
            })
            .collect::<Vec<_>>();
        let mut graph = Graph::new();
        let source_input = graph.input_dtype("source", [4, 2], DType::F32);
        let addend_input = graph.input_dtype("addend", [4, 2], DType::F32);
        let source = graph
            .shard_node(source_input, group.clone(), Some(0))
            .unwrap();
        let replicated = graph
            .redistribute_sharded(&source, group.clone(), None)
            .unwrap();
        let addend = graph.replicate_node(addend_input, group.clone()).unwrap();
        let value = graph
            .sharded_binary(&replicated, &addend, BinaryOp::Add)
            .unwrap();
        let gathered = graph.gather_sharded(&value).unwrap();
        let fused = ShardedCudaPlanner::executable_fused(&graph, &value, &bindings).unwrap();
        assert_eq!(fused.substitutions.len(), 2);
        assert_eq!(fused.plan.logical.stages.len(), 3);
        assert!(fused.plan.logical.stages[1..].iter().all(|stage| matches!(
            stage,
            CudaPlanStage::Local { dependencies, .. } if dependencies == &vec![0]
        )));

        let source_data = TensorData::new([4, 2], (0..8).map(|x| x as f32).collect()).unwrap();
        let addend_data = TensorData::new([4, 2], vec![10.; 8]).unwrap();
        let source_bytes = source_data.to_le_bytes().unwrap();
        let addend_bytes = addend_data.to_le_bytes().unwrap();
        let mut external = BTreeMap::new();
        for (rank, (owner, _)) in owners.iter().enumerate() {
            for (node, bytes) in [
                (
                    source.nodes()[rank],
                    &source_bytes[rank * 16..(rank + 1) * 16],
                ),
                (addend_input, addend_bytes.as_slice()),
            ] {
                let lease = owner
                    .allocator()
                    .allocate(NonZeroUsize::new(bytes.len()).unwrap())
                    .unwrap();
                lease.view().unwrap().copy_from(0, bytes).unwrap();
                external.insert((rank, node.index() as u64), lease);
            }
        }
        let mut environment = ShardedCudaExecutionEnvironment::new(external, 2);
        let result = environment.execute_composed(&fused).unwrap();
        assert_eq!(
            result
                .trace
                .iter()
                .map(|entry| entry.action)
                .collect::<Vec<_>>(),
            vec!["transfer", "local", "local"]
        );
        let expected = CpuBackend
            .execute(
                &graph,
                gathered,
                &HashMap::from([
                    ("source".into(), source_data),
                    ("addend".into(), addend_data),
                ]),
            )
            .unwrap()
            .to_le_bytes()
            .unwrap();
        for rank in 0..2 {
            let output = result
                .outputs
                .get(&(rank, value.nodes()[rank].index() as u64))
                .unwrap();
            let mut actual = vec![0; expected.len()];
            output.view().unwrap().copy_to(0, &mut actual).unwrap();
            assert_eq!(actual, expected);
        }
        assert!(mock.calls().contains(&"peer_copy"));
        drop(result);
        let repeat = environment.execute_composed(&fused).unwrap();
        assert_eq!(repeat.trace.len(), 3);
        assert_eq!(mock.generic_kernel_count(), 2);
        drop(repeat);
        mock.fail_peer_after(0, 2);
        let Err(failed) = environment.execute_composed(&fused) else {
            panic!("injected peer failure unexpectedly succeeded")
        };
        assert!(failed.to_string().contains("transfer 0"));
        assert_eq!(
            environment.external.len(),
            4,
            "transfer failure restores externals"
        );
        let retry = environment.execute_composed(&fused).unwrap();
        drop(retry);
        mock.set_launch_result(2);
        let Err(failed) = environment.execute_composed(&fused) else {
            panic!("injected launch failure unexpectedly succeeded")
        };
        assert!(failed.to_string().contains("stage 1"));
        assert_eq!(
            environment.external.len(),
            4,
            "launch failure restores externals"
        );
        mock.set_launch_result(0);
        assert_eq!(environment.execute_composed(&fused).unwrap().trace.len(), 3);
    }

    #[test]
    fn planner_fuses_axis_to_replica_add_for_one_and_four_owners() {
        for owners_count in [1usize, 4] {
            let mock = Arc::new(crate::cuda::tests::Mock::default());
            let driver = Driver::from_dispatch(mock.clone()).unwrap();
            let owners = (0..owners_count)
                .map(|ordinal| {
                    let device = driver.device(DeviceId(ordinal as u32)).unwrap();
                    let capability = device.capability().unwrap();
                    (device.retain_primary_context().unwrap(), capability)
                })
                .collect::<Vec<_>>();
            let group = DeviceGroup::new(
                (0..owners_count)
                    .map(|rank| crate::collective::DeviceId::new(format!("CUDA:{rank}")).unwrap()),
            )
            .unwrap();
            let bindings = owners
                .iter()
                .enumerate()
                .map(|(rank, (owner, capability))| CudaPlanBinding {
                    device: group.devices()[rank].clone(),
                    capability: capability.clone(),
                    context: owner.clone(),
                })
                .collect::<Vec<_>>();
            let mut graph = Graph::new();
            let source_input = graph.input_dtype("source", [4, 2], DType::F32);
            let addend_input = graph.input_dtype("addend", [4, 2], DType::F32);
            let source = graph
                .shard_node(source_input, group.clone(), Some(0))
                .unwrap();
            let replicated = graph
                .redistribute_sharded(&source, group.clone(), None)
                .unwrap();
            let addend = graph.replicate_node(addend_input, group.clone()).unwrap();
            let value = graph
                .sharded_binary(&replicated, &addend, BinaryOp::Add)
                .unwrap();
            let gathered = graph.gather_sharded(&value).unwrap();
            let fused = ShardedCudaPlanner::executable_fused(&graph, &value, &bindings).unwrap();
            let transport_stages = usize::from(owners_count != 1);
            assert_eq!(fused.substitutions.len(), owners_count * transport_stages);
            if owners_count == 1 {
                // A one-rank redistribution is represented by the existing
                // identity route after the local graph schedule; it requires
                // no substitution or cross-buffer transfer.
                assert!(matches!(
                    fused.plan.logical.stages.as_slice(),
                    [CudaPlanStage::Local { .. }, CudaPlanStage::Transfer { routes, .. }]
                    if routes[0].source_buffer == routes[0].destination_buffer
                ));
            } else {
                assert_eq!(fused.plan.logical.stages.len(), owners_count + 1);
                assert!(fused.plan.logical.stages[1..].iter().all(|stage| matches!(
                    stage,
                    CudaPlanStage::Local { dependencies, .. } if dependencies == &vec![0]
                )));
            }

            let source_data = TensorData::new([4, 2], (0..8).map(|x| x as f32).collect()).unwrap();
            let addend_data = TensorData::new([4, 2], vec![10.; 8]).unwrap();
            let source_bytes = source_data.to_le_bytes().unwrap();
            let addend_bytes = addend_data.to_le_bytes().unwrap();
            let shard_bytes = source_bytes.len() / owners_count;
            let mut external = BTreeMap::new();
            for (rank, (owner, _)) in owners.iter().enumerate() {
                let mut required = vec![
                    (
                        if owners_count == 1 {
                            source_input
                        } else {
                            source.nodes()[rank]
                        },
                        &source_bytes[rank * shard_bytes..(rank + 1) * shard_bytes],
                    ),
                    (addend_input, addend_bytes.as_slice()),
                ];
                if owners_count == 1 {
                    required.push((source.nodes()[rank], source_bytes.as_slice()));
                }
                for (node, bytes) in required {
                    let lease = owner
                        .allocator()
                        .allocate(NonZeroUsize::new(bytes.len()).unwrap())
                        .unwrap();
                    lease.view().unwrap().copy_from(0, bytes).unwrap();
                    external.insert((rank, node.index() as u64), lease);
                }
            }
            let mut environment = ShardedCudaExecutionEnvironment::new(external, owners_count);
            let expected = CpuBackend
                .execute(
                    &graph,
                    gathered,
                    &HashMap::from([
                        ("source".into(), source_data),
                        ("addend".into(), addend_data),
                    ]),
                )
                .unwrap()
                .to_le_bytes()
                .unwrap();
            let result = environment.execute_composed(&fused).unwrap();
            assert_eq!(
                result
                    .trace
                    .iter()
                    .map(|entry| entry.action)
                    .collect::<Vec<_>>(),
                if owners_count == 1 {
                    vec!["local", "transfer"]
                } else {
                    std::iter::once("transfer")
                        .chain(std::iter::repeat_n("local", owners_count))
                        .collect::<Vec<_>>()
                },
                "owners={owners_count}"
            );
            for rank in 0..owners_count {
                let output = result
                    .outputs
                    .get(&(rank, value.nodes()[rank].index() as u64))
                    .unwrap();
                let mut actual = vec![0; expected.len()];
                output.view().unwrap().copy_to(0, &mut actual).unwrap();
                assert_eq!(actual, expected, "owners={owners_count} rank={rank}");
            }
            drop(result);
            let repeat = environment.execute_composed(&fused).unwrap();
            assert_eq!(
                repeat.trace.len(),
                owners_count + transport_stages + (owners_count == 1) as usize
            );
            assert_eq!(
                mock.generic_kernel_count(),
                owners_count,
                "owners={owners_count}"
            );
        }
    }

    #[test]
    fn planner_fuses_axis_zero_to_axis_one_before_local_add() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let owners = (0..2)
            .map(|ordinal| {
                let device = driver.device(DeviceId(ordinal)).unwrap();
                let capability = device.capability().unwrap();
                (device.retain_primary_context().unwrap(), capability)
            })
            .collect::<Vec<_>>();
        let group = DeviceGroup::new(
            (0..2).map(|rank| crate::collective::DeviceId::new(format!("CUDA:{rank}")).unwrap()),
        )
        .unwrap();
        let bindings = owners
            .iter()
            .enumerate()
            .map(|(rank, (owner, capability))| CudaPlanBinding {
                device: group.devices()[rank].clone(),
                capability: capability.clone(),
                context: owner.clone(),
            })
            .collect::<Vec<_>>();
        let mut graph = Graph::new();
        let source_input = graph.input_dtype("source", [4, 4], DType::F32);
        let addend_input = graph.input_dtype("addend", [4, 4], DType::F32);
        let source = graph
            .shard_node(source_input, group.clone(), Some(0))
            .unwrap();
        let axis_one = graph
            .redistribute_sharded(&source, group.clone(), Some(1))
            .unwrap();
        let addend = graph
            .shard_node(addend_input, group.clone(), Some(1))
            .unwrap();
        let value = graph
            .sharded_binary(&axis_one, &addend, BinaryOp::Add)
            .unwrap();
        let gathered = graph.gather_sharded(&value).unwrap();
        let fused = ShardedCudaPlanner::executable_fused(&graph, &value, &bindings).unwrap();
        assert_eq!(fused.substitutions.len(), 2);
        assert_eq!(fused.plan.logical.stages.len(), 3);
        assert!(fused.plan.logical.stages[1..].iter().all(|stage| matches!(
            stage,
            CudaPlanStage::Local { dependencies, inputs, .. }
            if dependencies == &vec![0] && inputs.len() == 2
        )));
        let source_data = TensorData::new([4, 4], (0..16).map(|x| x as f32).collect()).unwrap();
        let addend_data = TensorData::new([4, 4], vec![10.; 16]).unwrap();
        let source_bytes = source_data.to_le_bytes().unwrap();
        let addend_bytes = addend_data.to_le_bytes().unwrap();
        let mut external = BTreeMap::new();
        for (rank, (owner, _)) in owners.iter().enumerate() {
            let source_shard = &source_bytes[rank * 32..(rank + 1) * 32];
            for (node, bytes) in [
                (source.nodes()[rank], source_shard),
                // Static axis-one shrink views retain the original global
                // addend source as their ABI backing allocation.
                (addend_input, addend_bytes.as_slice()),
            ] {
                let lease = owner
                    .allocator()
                    .allocate(NonZeroUsize::new(bytes.len()).unwrap())
                    .unwrap();
                lease.view().unwrap().copy_from(0, bytes).unwrap();
                external.insert((rank, node.index() as u64), lease);
            }
        }
        let expected = CpuBackend
            .execute(
                &graph,
                gathered,
                &HashMap::from([
                    ("source".into(), source_data),
                    ("addend".into(), addend_data),
                ]),
            )
            .unwrap()
            .to_le_bytes()
            .unwrap();
        let result = ShardedCudaExecutionEnvironment::new(external, 2)
            .execute_composed(&fused)
            .unwrap();
        assert_eq!(
            result
                .trace
                .iter()
                .map(|entry| entry.action)
                .collect::<Vec<_>>(),
            vec!["transfer", "local", "local"]
        );
        for rank in 0..2 {
            let output = result
                .outputs
                .get(&(rank, value.nodes()[rank].index() as u64))
                .unwrap();
            let mut actual = vec![0; source_bytes.len() / 2];
            output.view().unwrap().copy_to(0, &mut actual).unwrap();
            let expected_shard = match rank {
                0 => [
                    &expected[0..8],
                    &expected[16..24],
                    &expected[32..40],
                    &expected[48..56],
                ]
                .concat(),
                _ => [
                    &expected[8..16],
                    &expected[24..32],
                    &expected[40..48],
                    &expected[56..64],
                ]
                .concat(),
            };
            assert_eq!(actual, expected_shard, "rank {rank}");
        }
        assert!(mock.calls().contains(&"peer_copy"));
    }

    #[test]
    fn planner_fuses_zero_domain_without_device_work() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let owners = (0..2)
            .map(|ordinal| {
                let device = driver.device(DeviceId(ordinal)).unwrap();
                let capability = device.capability().unwrap();
                (device.retain_primary_context().unwrap(), capability)
            })
            .collect::<Vec<_>>();
        let group = DeviceGroup::new(
            (0..2).map(|rank| crate::collective::DeviceId::new(format!("CUDA:{rank}")).unwrap()),
        )
        .unwrap();
        let bindings = owners
            .iter()
            .enumerate()
            .map(|(rank, (owner, capability))| CudaPlanBinding {
                device: group.devices()[rank].clone(),
                capability: capability.clone(),
                context: owner.clone(),
            })
            .collect::<Vec<_>>();
        let mut graph = Graph::new();
        let source_input = graph.input_dtype("source", [0, 2], DType::F32);
        let addend_input = graph.input_dtype("addend", [0, 2], DType::F32);
        let source = graph
            .shard_node(source_input, group.clone(), Some(0))
            .unwrap();
        let replicated = graph
            .redistribute_sharded(&source, group.clone(), None)
            .unwrap();
        let addend = graph.replicate_node(addend_input, group.clone()).unwrap();
        let value = graph
            .sharded_binary(&replicated, &addend, BinaryOp::Add)
            .unwrap();
        let fused = ShardedCudaPlanner::executable_fused(&graph, &value, &bindings).unwrap();
        assert_eq!(fused.substitutions.len(), 2);
        let zeros = owners
            .iter()
            .enumerate()
            .flat_map(|(rank, (owner, _))| {
                [source.nodes()[rank], addend_input].map(move |node| {
                    (
                        (rank, node.index() as u64),
                        LogicalZeroBuffer::new(
                            owner.identity(),
                            rank,
                            node.index() as u64,
                            DType::F32,
                            Shape::new(vec![0, 2]),
                        ),
                    )
                })
            })
            .collect::<BTreeMap<_, _>>();
        let pools = owners
            .iter()
            .map(|(owner, _)| owner.allocator())
            .collect::<Vec<_>>();
        let baseline = pools.iter().map(|pool| pool.stats()).collect::<Vec<_>>();
        let calls = mock.calls().len();
        let result = ShardedCudaExecutionEnvironment::with_logical_zeros(BTreeMap::new(), zeros, 2)
            .execute_composed(&fused)
            .unwrap();
        assert_eq!(mock.calls().len(), calls, "zero fusion has no Driver work");
        assert_eq!(
            pools.iter().map(|pool| pool.stats()).collect::<Vec<_>>(),
            baseline,
            "logical zeros do not change allocator accounting"
        );
        assert!(result.outputs.is_empty());
        assert!(value.nodes().iter().enumerate().all(|(rank, node)| {
            result
                .zero_outputs
                .contains_key(&(rank, node.index() as u64))
        }));
        assert!(result.trace.iter().all(|entry| entry.skipped));
    }

    #[test]
    fn planner_retains_two_owner_sharded_graph_local_ptx_and_view_buffers() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let driver = Driver::from_dispatch(mock).unwrap();
        let first_device = driver.device(DeviceId(0)).unwrap();
        let first_capability = first_device.capability().unwrap();
        let first = first_device.retain_primary_context().unwrap();
        let second_device = driver.device(DeviceId(1)).unwrap();
        let second_capability = second_device.capability().unwrap();
        let second = second_device.retain_primary_context().unwrap();
        let group = DeviceGroup::new([
            crate::collective::DeviceId::new("CUDA:0").unwrap(),
            crate::collective::DeviceId::new("CUDA:1").unwrap(),
        ])
        .unwrap();
        let mut graph = Graph::new();
        let left = graph.input("left", [4, 2]);
        let right = graph.input("right", [4, 2]);
        let lhs = graph.shard_node(left, group.clone(), Some(0)).unwrap();
        let rhs = graph.shard_node(right, group.clone(), Some(0)).unwrap();
        let value = graph.sharded_binary(&lhs, &rhs, BinaryOp::Add).unwrap();
        let bindings = vec![
            CudaPlanBinding {
                device: group.devices()[0].clone(),
                capability: first_capability,
                context: first,
            },
            CudaPlanBinding {
                device: group.devices()[1].clone(),
                capability: second_capability,
                context: second,
            },
        ];
        let plan = ShardedCudaPlanner::build(&graph, &value, &bindings).unwrap();
        assert!(plan.diagnostics.is_empty());
        assert_eq!(plan.stages.len(), 2);
        for (rank, stage) in plan.stages.iter().enumerate() {
            let CudaPlanStage::Local {
                diagnostic,
                shape,
                dtype,
                inputs,
                output,
                ..
            } = stage
            else {
                panic!("rank {rank} did not plan a local stage");
            };
            assert!(diagnostic.is_none());
            assert_eq!(shape, &Shape::new(vec![2, 2]));
            assert_eq!(*dtype, DType::F32);
            assert_eq!(inputs, &vec![left.index() as u64, right.index() as u64]);
            assert_eq!(*output, value.nodes()[rank].index() as u64);
        }
        let executable = ShardedCudaPlanner::executable(&graph, plan, &bindings).unwrap();
        assert!(executable.kernels.iter().all(Option::is_some));
        for rank in 0..2 {
            let external = executable
                .buffers
                .iter()
                .filter(|buffer| {
                    buffer.rank == rank && matches!(buffer.role, ExecutableBufferRole::External)
                })
                .collect::<Vec<_>>();
            assert_eq!(external.len(), 2);
            for buffer in external {
                assert!(matches!(buffer.buffer, 0 | 1));
                assert_eq!(buffer.dtype, DType::F32);
                assert_eq!(buffer.shape, Shape::new(vec![4, 2]));
                assert_eq!(buffer.bytes, 32);
                assert_eq!(buffer.owner_identity, executable.owners[rank].identity());
            }
            let output = executable
                .buffers
                .iter()
                .find(|buffer| {
                    buffer.rank == rank && matches!(buffer.role, ExecutableBufferRole::Output)
                })
                .unwrap();
            assert_eq!(output.shape, Shape::new(vec![2, 2]));
            assert_eq!(output.bytes, 16);
        }
    }

    #[test]
    fn executor_runs_two_owner_graph_shrink_views_against_cpu_oracle() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let first_device = driver.device(DeviceId(0)).unwrap();
        let first_capability = first_device.capability().unwrap();
        let first = first_device.retain_primary_context().unwrap();
        let second_device = driver.device(DeviceId(1)).unwrap();
        let second_capability = second_device.capability().unwrap();
        let second = second_device.retain_primary_context().unwrap();
        assert_ne!(
            first.identity(),
            second.identity(),
            "stable owners isolate colliding mock handles"
        );
        let group = DeviceGroup::new([
            crate::collective::DeviceId::new("CUDA:0").unwrap(),
            crate::collective::DeviceId::new("CUDA:1").unwrap(),
        ])
        .unwrap();
        let mut graph = Graph::new();
        let left = graph.input("left", [4, 2]);
        let right = graph.input("right", [4, 2]);
        let lhs = graph.shard_node(left, group.clone(), Some(0)).unwrap();
        let rhs = graph.shard_node(right, group.clone(), Some(0)).unwrap();
        let value = graph.sharded_binary(&lhs, &rhs, BinaryOp::Add).unwrap();
        let gathered = graph.gather_sharded(&value).unwrap();
        let bindings = vec![
            CudaPlanBinding {
                device: group.devices()[0].clone(),
                capability: first_capability,
                context: first.clone(),
            },
            CudaPlanBinding {
                device: group.devices()[1].clone(),
                capability: second_capability,
                context: second.clone(),
            },
        ];
        let logical = ShardedCudaPlanner::build(&graph, &value, &bindings).unwrap();
        assert!(logical.diagnostics.is_empty());
        let plan = ShardedCudaPlanner::executable(&graph, logical, &bindings).unwrap();
        assert_eq!(plan.kernels.len(), 2);
        for kernel in plan.kernels.iter().flatten() {
            assert_eq!(kernel.extent, 4);
            assert!(
                kernel
                    .buffers
                    .iter()
                    .any(|abi| !abi.mutable && abi.source_shape == Shape::new(vec![4, 2]))
            );
        }
        let left_data = TensorData::new([4, 2], (0..8).map(|n| n as f32).collect()).unwrap();
        let right_data = TensorData::new([4, 2], (10..18).map(|n| n as f32).collect()).unwrap();
        let left_bytes = left_data.to_le_bytes().unwrap();
        let right_bytes = right_data.to_le_bytes().unwrap();
        let mut external = BTreeMap::new();
        for (rank, primary) in [first.clone(), second.clone()].into_iter().enumerate() {
            for (node, bytes) in [(left, &left_bytes), (right, &right_bytes)] {
                let lease = primary
                    .allocator()
                    .allocate(NonZeroUsize::new(bytes.len()).unwrap())
                    .unwrap();
                lease.view().unwrap().copy_from(0, bytes).unwrap();
                external.insert((rank, node.index() as u64), lease);
            }
        }
        let mut environment = ShardedCudaExecutionEnvironment::new(external, 2);
        let first_result = environment.execute(&plan).unwrap();
        assert_eq!(
            first_result.trace,
            vec![
                ShardedCudaExecutionTrace {
                    stage: 0,
                    action: "local",
                    skipped: false
                },
                ShardedCudaExecutionTrace {
                    stage: 1,
                    action: "local",
                    skipped: false
                },
            ]
        );
        let mut actual = Vec::new();
        for rank in 0..2 {
            let key = (rank, value.nodes()[rank].index() as u64);
            let output = first_result.outputs.get(&key).unwrap();
            let mut bytes = vec![0; 16];
            output.view().unwrap().copy_to(0, &mut bytes).unwrap();
            actual.extend(bytes);
        }
        let expected = CpuBackend
            .execute(
                &graph,
                gathered,
                &HashMap::from([("left".into(), left_data), ("right".into(), right_data)]),
            )
            .unwrap()
            .to_le_bytes()
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(mock.generic_kernel_count(), 2);
        drop(first_result);
        let repeated = environment.execute(&plan).unwrap();
        assert_eq!(repeated.trace.len(), 2);
        assert_eq!(
            mock.generic_kernel_count(),
            2,
            "same owner caches reuse semantic registrations"
        );
    }

    #[test]
    fn graph_sharded_cast_and_select_execute_with_static_views_across_owner_counts() {
        for (name, select) in [("cast-i32-f32", false), ("select-f32", true)] {
            for ranks in [1_usize, 2, 4] {
                let mock = Arc::new(crate::cuda::tests::Mock::default());
                let driver = Driver::from_dispatch(mock.clone()).unwrap();
                let owners = (0..ranks)
                    .map(|ordinal| {
                        let device = driver.device(DeviceId(ordinal as u32)).unwrap();
                        let capability = device.capability().unwrap();
                        (device.retain_primary_context().unwrap(), capability)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    owners
                        .iter()
                        .map(|(owner, _)| owner.identity())
                        .collect::<BTreeSet<_>>()
                        .len(),
                    ranks,
                    "{name}: stable owners isolate colliding raw handles"
                );
                let group =
                    DeviceGroup::new((0..ranks).map(|rank| {
                        crate::collective::DeviceId::new(format!("CUDA:{rank}")).unwrap()
                    }))
                    .unwrap();
                let mut graph = Graph::new();
                let (value, gathered, input_data) = if select {
                    let condition_input = graph.input_dtype("condition", [4, 1], DType::Bool);
                    let true_input = graph.input_dtype("on_true", [4, 2], DType::F32);
                    let false_input = graph.input_dtype("on_false", [1, 2], DType::F32);
                    let condition = graph
                        .shard_node(condition_input, group.clone(), Some(0))
                        .unwrap();
                    let on_true = graph
                        .shard_node(true_input, group.clone(), Some(0))
                        .unwrap();
                    let on_false = graph.replicate_node(false_input, group.clone()).unwrap();
                    let value = graph
                        .sharded_select(&condition, &on_true, &on_false)
                        .unwrap();
                    let gathered = if ranks == 1 {
                        value.nodes()[0]
                    } else {
                        graph.gather_sharded(&value).unwrap()
                    };
                    (
                        value,
                        gathered,
                        vec![
                            (
                                condition_input,
                                TensorData::from_storage(
                                    [4, 1],
                                    Storage::Bool(vec![true, false, false, true]),
                                )
                                .unwrap(),
                                String::from("condition"),
                            ),
                            (
                                true_input,
                                TensorData::new([4, 2], (0..8).map(|value| value as f32).collect())
                                    .unwrap(),
                                String::from("on_true"),
                            ),
                            (
                                false_input,
                                TensorData::new([1, 2], vec![-1., -2.]).unwrap(),
                                String::from("on_false"),
                            ),
                        ],
                    )
                } else {
                    let input_node = graph.input_dtype("input", [4, 2], DType::I32);
                    let input = graph
                        .shard_node(input_node, group.clone(), Some(0))
                        .unwrap();
                    let value = graph.sharded_cast(&input, DType::F32).unwrap();
                    let gathered = if ranks == 1 {
                        value.nodes()[0]
                    } else {
                        graph.gather_sharded(&value).unwrap()
                    };
                    (
                        value,
                        gathered,
                        vec![(
                            input_node,
                            TensorData::from_storage(
                                [4, 2],
                                Storage::I32(vec![i32::MIN, -1, 0, 1, 2, 3, 4, i32::MAX]),
                            )
                            .unwrap(),
                            String::from("input"),
                        )],
                    )
                };
                let bindings = owners
                    .iter()
                    .enumerate()
                    .map(|(rank, (owner, capability))| CudaPlanBinding {
                        device: group.devices()[rank].clone(),
                        capability: capability.clone(),
                        context: owner.clone(),
                    })
                    .collect::<Vec<_>>();
                let logical = ShardedCudaPlanner::build(&graph, &value, &bindings).unwrap();
                assert!(logical.diagnostics.is_empty(), "{name}: {ranks} ranks");
                let plan = ShardedCudaPlanner::executable(&graph, logical, &bindings).unwrap();
                assert!(plan.kernels.iter().all(Option::is_some));
                let inputs = input_data
                    .iter()
                    .map(|(_, data, name)| (name.clone(), data.clone()))
                    .collect::<HashMap<_, _>>();
                let expected = CpuBackend
                    .execute(&graph, gathered, &inputs)
                    .unwrap()
                    .to_le_bytes()
                    .unwrap();
                let mut external = BTreeMap::new();
                for (rank, (owner, _)) in owners.iter().enumerate() {
                    for (node, data, _) in &input_data {
                        let bytes = data.to_le_bytes().unwrap();
                        let lease = owner
                            .allocator()
                            .allocate(NonZeroUsize::new(bytes.len()).unwrap())
                            .unwrap();
                        lease.view().unwrap().copy_from(0, &bytes).unwrap();
                        external.insert((rank, node.index() as u64), lease);
                    }
                }
                let mut environment = ShardedCudaExecutionEnvironment::new(external, ranks);
                let result = environment.execute(&plan).unwrap();
                assert_eq!(result.trace.len(), ranks, "{name}: trace order");
                assert!(result.trace.iter().enumerate().all(|(rank, trace)| {
                    trace.stage == rank && trace.action == "local" && !trace.skipped
                }));
                let mut actual = Vec::new();
                for rank in 0..ranks {
                    let output = result
                        .outputs
                        .get(&(rank, value.nodes()[rank].index() as u64))
                        .unwrap();
                    let mut bytes = vec![
                        0;
                        value.layout().local_shape(rank).unwrap().numel().unwrap()
                            * value.dtype().itemsize()
                    ];
                    output.view().unwrap().copy_to(0, &mut bytes).unwrap();
                    actual.extend(bytes);
                }
                assert_eq!(actual, expected, "{name}: {ranks} rank output");
                drop(result);
                let repeat = environment.execute(&plan).unwrap();
                assert_eq!(repeat.trace.len(), ranks);
                assert_eq!(
                    mock.generic_kernel_count(),
                    ranks,
                    "{name}: owner-scoped cache reuse"
                );
            }
        }
    }

    #[test]
    fn graph_shrink_add_owner_count_matrix_reuses_owner_caches() {
        let cases = [
            (
                "f32",
                DType::F32,
                TensorData::new([4, 2], (0..8).map(|x| x as f32).collect()).unwrap(),
                TensorData::new([4, 2], (10..18).map(|x| x as f32).collect()).unwrap(),
            ),
            (
                "i32-wrap",
                DType::I32,
                TensorData::from_storage(
                    [4, 2],
                    Storage::I32(vec![i32::MAX, 1, -2, 3, 4, 5, 6, 7]),
                )
                .unwrap(),
                TensorData::from_storage(
                    [4, 2],
                    Storage::I32(vec![1, -1, 2, i32::MAX, 1, -5, -6, 9]),
                )
                .unwrap(),
            ),
            (
                "u64-wrap",
                DType::U64,
                TensorData::from_storage([4, 2], Storage::U64(vec![u64::MAX, 1, 2, 3, 4, 5, 6, 7]))
                    .unwrap(),
                TensorData::from_storage([4, 2], Storage::U64(vec![1, u64::MAX, 2, 3, 4, 5, 6, 9]))
                    .unwrap(),
            ),
        ];
        for (name, dtype, left_data, right_data) in cases {
            for ranks in [1_usize, 2, 4] {
                let mock = Arc::new(crate::cuda::tests::Mock::default());
                let driver = Driver::from_dispatch(mock.clone()).unwrap();
                let owners = (0..ranks)
                    .map(|ordinal| {
                        let device = driver.device(DeviceId(ordinal as u32)).unwrap();
                        let capability = device.capability().unwrap();
                        (device.retain_primary_context().unwrap(), capability)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    owners
                        .iter()
                        .map(|(owner, _)| owner.identity())
                        .collect::<BTreeSet<_>>()
                        .len(),
                    ranks,
                    "{ranks} stable owners isolate colliding mock handles"
                );
                let group =
                    DeviceGroup::new((0..ranks).map(|rank| {
                        crate::collective::DeviceId::new(format!("CUDA:{rank}")).unwrap()
                    }))
                    .unwrap();
                let mut graph = Graph::new();
                let left = graph.input_dtype("left", [4, 2], dtype);
                let right = graph.input_dtype("right", [4, 2], dtype);
                let lhs = graph.shard_node(left, group.clone(), Some(0)).unwrap();
                let rhs = graph.shard_node(right, group.clone(), Some(0)).unwrap();
                let value = graph.sharded_binary(&lhs, &rhs, BinaryOp::Add).unwrap();
                let gathered = if ranks == 1 {
                    value.nodes()[0]
                } else {
                    graph.gather_sharded(&value).unwrap()
                };
                let bindings = owners
                    .iter()
                    .enumerate()
                    .map(|(rank, (owner, capability))| CudaPlanBinding {
                        device: group.devices()[rank].clone(),
                        capability: capability.clone(),
                        context: owner.clone(),
                    })
                    .collect::<Vec<_>>();
                let logical = ShardedCudaPlanner::build(&graph, &value, &bindings).unwrap();
                assert!(logical.diagnostics.is_empty(), "{ranks} ranks");
                let plan = ShardedCudaPlanner::executable(&graph, logical, &bindings).unwrap();
                assert_eq!(plan.kernels.len(), ranks);
                let (left_bytes, right_bytes) = (
                    left_data.to_le_bytes().unwrap(),
                    right_data.to_le_bytes().unwrap(),
                );
                let mut external = BTreeMap::new();
                for (rank, (owner, _)) in owners.iter().enumerate() {
                    for (node, bytes) in [(left, &left_bytes), (right, &right_bytes)] {
                        let lease = owner
                            .allocator()
                            .allocate(NonZeroUsize::new(bytes.len()).unwrap())
                            .unwrap();
                        lease.view().unwrap().copy_from(0, bytes).unwrap();
                        external.insert((rank, node.index() as u64), lease);
                    }
                }
                let mut environment = ShardedCudaExecutionEnvironment::new(external, ranks);
                let result = environment.execute(&plan).unwrap();
                assert_eq!(result.trace.len(), ranks);
                let mut actual = Vec::new();
                for rank in 0..ranks {
                    let output = result
                        .outputs
                        .get(&(rank, value.nodes()[rank].index() as u64))
                        .unwrap();
                    let mut bytes = vec![0; left_bytes.len() / ranks];
                    output.view().unwrap().copy_to(0, &mut bytes).unwrap();
                    actual.extend(bytes);
                }
                let expected = CpuBackend
                    .execute(
                        &graph,
                        gathered,
                        &HashMap::from([
                            ("left".into(), left_data.clone()),
                            ("right".into(), right_data.clone()),
                        ]),
                    )
                    .unwrap()
                    .to_le_bytes()
                    .unwrap();
                assert_eq!(actual, expected, "{name}: {ranks} ranks");
                drop(result);
                let repeat = environment.execute(&plan).unwrap();
                assert_eq!(repeat.trace.len(), ranks);
                assert_eq!(
                    mock.generic_kernel_count(),
                    ranks,
                    "{name}: {ranks} ranks cache reuse"
                );
            }
        }
    }

    #[test]
    fn graph_sharded_neg_executes_with_views_across_owner_counts() {
        let cases = [
            (
                "i32 wrapping",
                DType::I32,
                TensorData::from_storage(
                    [4, 2],
                    Storage::I32(vec![i32::MIN, -7, -1, 0, 1, 2, 7, i32::MAX]),
                )
                .unwrap(),
            ),
            (
                "f32 exact bytes",
                DType::F32,
                TensorData::from_storage(
                    [4, 2],
                    Storage::F32(vec![-0.0, -1.25, 0.0, 2.5, f32::INFINITY, -3.0, 4.0, -8.0]),
                )
                .unwrap(),
            ),
        ];
        for (name, dtype, input_data) in cases {
            for ranks in [1_usize, 2, 4] {
                let mock = Arc::new(crate::cuda::tests::Mock::default());
                let driver = Driver::from_dispatch(mock.clone()).unwrap();
                let owners = (0..ranks)
                    .map(|ordinal| {
                        let device = driver.device(DeviceId(ordinal as u32)).unwrap();
                        let capability = device.capability().unwrap();
                        (device.retain_primary_context().unwrap(), capability)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    owners
                        .iter()
                        .map(|(owner, _)| owner.identity())
                        .collect::<BTreeSet<_>>()
                        .len(),
                    ranks,
                    "{name}: stable owners isolate colliding raw mock handles"
                );
                let group =
                    DeviceGroup::new((0..ranks).map(|rank| {
                        crate::collective::DeviceId::new(format!("CUDA:{rank}")).unwrap()
                    }))
                    .unwrap();
                let mut graph = Graph::new();
                let input = graph.input_dtype("input", [4, 2], dtype);
                let sharded = graph.shard_node(input, group.clone(), Some(0)).unwrap();
                let value = graph.sharded_unary(&sharded, crate::UnaryOp::Neg).unwrap();
                let gathered = if ranks == 1 {
                    value.nodes()[0]
                } else {
                    graph.gather_sharded(&value).unwrap()
                };
                let bindings = owners
                    .iter()
                    .enumerate()
                    .map(|(rank, (owner, capability))| CudaPlanBinding {
                        device: group.devices()[rank].clone(),
                        capability: capability.clone(),
                        context: owner.clone(),
                    })
                    .collect::<Vec<_>>();
                let logical = ShardedCudaPlanner::build(&graph, &value, &bindings).unwrap();
                assert!(logical.diagnostics.is_empty(), "{name}: {ranks} ranks");
                let plan = ShardedCudaPlanner::executable(&graph, logical, &bindings).unwrap();
                assert!(plan.kernels.iter().all(Option::is_some));
                let expected = CpuBackend
                    .execute(
                        &graph,
                        gathered,
                        &HashMap::from([(String::from("input"), input_data.clone())]),
                    )
                    .unwrap()
                    .to_le_bytes()
                    .unwrap();
                let source_bytes = input_data.to_le_bytes().unwrap();
                let mut external = BTreeMap::new();
                for (rank, (owner, _)) in owners.iter().enumerate() {
                    let lease = owner
                        .allocator()
                        .allocate(NonZeroUsize::new(source_bytes.len()).unwrap())
                        .unwrap();
                    lease.view().unwrap().copy_from(0, &source_bytes).unwrap();
                    // Static views retain the original input node in their ABI.
                    external.insert((rank, input.index() as u64), lease);
                }
                let mut environment = ShardedCudaExecutionEnvironment::new(external, ranks);
                let result = environment.execute(&plan).unwrap();
                assert!(result.trace.iter().enumerate().all(|(rank, trace)| {
                    trace.stage == rank && trace.action == "local" && !trace.skipped
                }));
                let mut actual = Vec::new();
                for rank in 0..ranks {
                    let output = result
                        .outputs
                        .get(&(rank, value.nodes()[rank].index() as u64))
                        .unwrap();
                    let mut bytes = vec![
                        0;
                        value.layout().local_shape(rank).unwrap().numel().unwrap()
                            * value.dtype().itemsize()
                    ];
                    output.view().unwrap().copy_to(0, &mut bytes).unwrap();
                    actual.extend(bytes);
                }
                assert_eq!(actual, expected, "{name}: {ranks} rank output");
                drop(result);
                let repeat = environment.execute(&plan).unwrap();
                assert_eq!(repeat.trace.len(), ranks);
                assert_eq!(
                    mock.generic_kernel_count(),
                    ranks,
                    "{name}: owner-scoped cache reuse"
                );
            }
        }
    }

    #[test]
    fn graph_sharded_neg_zero_domain_is_logical_and_unsupported_unary_is_diagnostic() {
        for ranks in [1_usize, 2, 4] {
            let mock = Arc::new(crate::cuda::tests::Mock::default());
            let driver = Driver::from_dispatch(mock.clone()).unwrap();
            let owners = (0..ranks)
                .map(|ordinal| {
                    let device = driver.device(DeviceId(ordinal as u32)).unwrap();
                    let capability = device.capability().unwrap();
                    (device.retain_primary_context().unwrap(), capability)
                })
                .collect::<Vec<_>>();
            let group = DeviceGroup::new(
                (0..ranks)
                    .map(|rank| crate::collective::DeviceId::new(format!("CUDA:{rank}")).unwrap()),
            )
            .unwrap();
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::new(vec![0, 2]), DType::F32);
            let sharded = graph.shard_node(input, group.clone(), Some(0)).unwrap();
            let value = graph.sharded_unary(&sharded, crate::UnaryOp::Neg).unwrap();
            let bindings = owners
                .iter()
                .enumerate()
                .map(|(rank, (owner, capability))| CudaPlanBinding {
                    device: group.devices()[rank].clone(),
                    capability: capability.clone(),
                    context: owner.clone(),
                })
                .collect::<Vec<_>>();
            let logical = ShardedCudaPlanner::build(&graph, &value, &bindings).unwrap();
            assert!(logical.diagnostics.is_empty(), "zero: {ranks} ranks");
            let plan = ShardedCudaPlanner::executable(&graph, logical, &bindings).unwrap();
            let zero_external = owners
                .iter()
                .enumerate()
                .map(|(rank, (owner, _))| {
                    (
                        (rank, input.index() as u64),
                        LogicalZeroBuffer::new(
                            owner.identity(),
                            rank,
                            input.index() as u64,
                            DType::F32,
                            Shape::new(vec![0, 2]),
                        ),
                    )
                })
                .collect();
            let result = ShardedCudaExecutionEnvironment::with_logical_zeros(
                BTreeMap::new(),
                zero_external,
                ranks,
            )
            .execute(&plan)
            .unwrap();
            assert_eq!(result.zero_outputs.len(), ranks);
            assert!(result.trace.iter().enumerate().all(|(rank, trace)| {
                trace.stage == rank && trace.action == "local" && trace.skipped
            }));
            assert_eq!(mock.generic_kernel_count(), 0, "zero: {ranks} ranks");
        }

        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let device = driver.device(DeviceId(0)).unwrap();
        let owner = device.retain_primary_context().unwrap();
        let group =
            DeviceGroup::new([crate::collective::DeviceId::new("CUDA:0").unwrap()]).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let sharded = graph.shard_node(input, group.clone(), Some(0)).unwrap();
        let value = graph.sharded_unary(&sharded, crate::UnaryOp::Exp).unwrap();
        let bindings = vec![CudaPlanBinding {
            device: group.devices()[0].clone(),
            capability: device.capability().unwrap(),
            context: owner,
        }];
        let logical = ShardedCudaPlanner::build(&graph, &value, &bindings).unwrap();
        assert!(
            matches!(
                logical.diagnostics.as_slice(),
                [CudaPlanDiagnostic::Unsupported { reason, .. }] if reason.contains("unary")
            ),
            "{:#?}",
            logical.diagnostics
        );
        let plan = ShardedCudaPlanner::executable(&graph, logical, &bindings).unwrap();
        let before = mock.calls().len();
        assert!(
            ShardedCudaExecutionEnvironment::new(BTreeMap::new(), 1)
                .execute(&plan)
                .is_err()
        );
        assert_eq!(
            mock.calls().len(),
            before,
            "diagnostic preflight has no Driver work"
        );
    }

    #[test]
    fn graph_redistribution_trace_executes_exact_d2d_and_peer_routes() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let owners = (0..2)
            .map(|ordinal| {
                let device = driver.device(DeviceId(ordinal)).unwrap();
                let capability = device.capability().unwrap();
                (device.retain_primary_context().unwrap(), capability)
            })
            .collect::<Vec<_>>();
        let group = DeviceGroup::new(
            (0..2).map(|rank| crate::collective::DeviceId::new(format!("CUDA:{rank}")).unwrap()),
        )
        .unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [4, 2], DType::I32);
        let source = graph.shard_node(input, group.clone(), Some(0)).unwrap();
        let destination = graph
            .redistribute_sharded(&source, group.clone(), None)
            .unwrap();
        let bindings = owners
            .iter()
            .enumerate()
            .map(|(rank, (owner, capability))| CudaPlanBinding {
                device: group.devices()[rank].clone(),
                capability: capability.clone(),
                context: owner.clone(),
            })
            .collect::<Vec<_>>();
        let plan = executable_redistribution_plan(&source, &destination, &bindings).unwrap();
        plan.validate().unwrap();
        let CudaPlanStage::Transfer { routes, .. } = &plan.logical.stages[0] else {
            panic!("expected typed transfer stage")
        };
        assert_eq!(routes.len(), 4, "two sources copied to each replica");
        assert!(
            routes
                .iter()
                .any(|route| route.source_rank == route.destination_rank)
        );
        assert!(
            routes
                .iter()
                .any(|route| route.source_rank != route.destination_rank)
        );
        assert!(
            routes
                .iter()
                .all(|route| route.bytes == 16 && route.elements == 4)
        );

        let source_bytes = [
            TensorData::from_storage([2, 2], Storage::I32(vec![1, 2, 3, 4]))
                .unwrap()
                .to_le_bytes()
                .unwrap(),
            TensorData::from_storage([2, 2], Storage::I32(vec![5, 6, 7, 8]))
                .unwrap()
                .to_le_bytes()
                .unwrap(),
        ];
        let mut external = BTreeMap::new();
        for (rank, (owner, _)) in owners.iter().enumerate() {
            let lease = owner
                .allocator()
                .allocate(NonZeroUsize::new(source_bytes[rank].len()).unwrap())
                .unwrap();
            lease
                .view()
                .unwrap()
                .copy_from(0, &source_bytes[rank])
                .unwrap();
            external.insert((rank, source.nodes()[rank].index() as u64), lease);
        }
        let mut environment = ShardedCudaExecutionEnvironment::new(external, 2);
        mock.fail_dtod_after(0, 2);
        let Err(failed) = environment.execute(&plan) else {
            panic!("injected DtoD failure unexpectedly succeeded")
        };
        assert!(failed.to_string().contains("transfer 0"));
        assert_eq!(environment.external.len(), 2);
        mock.fail_peer_after(0, 2);
        let Err(failed) = environment.execute(&plan) else {
            panic!("injected peer failure unexpectedly succeeded")
        };
        assert!(failed.to_string().contains("transfer 0"));
        assert_eq!(
            environment.external.len(),
            2,
            "failed routes restore all external source leases for retry"
        );
        let result = environment.execute(&plan).unwrap();
        assert_eq!(
            result.trace,
            vec![ShardedCudaExecutionTrace {
                stage: 0,
                action: "transfer",
                skipped: false,
            }]
        );
        let expected = [1_i32, 2, 3, 4, 5, 6, 7, 8]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        for rank in 0..2 {
            let output = result
                .outputs
                .get(&(rank, destination.nodes()[rank].index() as u64))
                .unwrap();
            let mut actual = vec![0; expected.len()];
            output.view().unwrap().copy_to(0, &mut actual).unwrap();
            assert_eq!(actual, expected, "replica {rank}");
        }
        assert!(mock.calls().contains(&"dtod_async"));
        assert!(mock.calls().contains(&"peer_copy"));
    }

    #[test]
    fn graph_redistribution_trace_four_owners_replicates_exact_bytes() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let owners = (0..4)
            .map(|ordinal| {
                let device = driver.device(DeviceId(ordinal)).unwrap();
                let capability = device.capability().unwrap();
                (device.retain_primary_context().unwrap(), capability)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            owners
                .iter()
                .map(|(owner, _)| owner.identity())
                .collect::<BTreeSet<_>>()
                .len(),
            4
        );
        let group = DeviceGroup::new(
            (0..4).map(|rank| crate::collective::DeviceId::new(format!("CUDA:{rank}")).unwrap()),
        )
        .unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [4, 2], DType::I32);
        let source = graph.shard_node(input, group.clone(), Some(0)).unwrap();
        let destination = graph
            .redistribute_sharded(&source, group.clone(), None)
            .unwrap();
        let bindings = owners
            .iter()
            .enumerate()
            .map(|(rank, (owner, capability))| CudaPlanBinding {
                device: group.devices()[rank].clone(),
                capability: capability.clone(),
                context: owner.clone(),
            })
            .collect::<Vec<_>>();
        let plan = executable_redistribution_plan(&source, &destination, &bindings).unwrap();
        let CudaPlanStage::Transfer { routes, .. } = &plan.logical.stages[0] else {
            panic!("expected typed transfer stage")
        };
        assert_eq!(routes.len(), 16);
        assert!(
            routes
                .iter()
                .all(|route| route.elements == 2 && route.bytes == 8)
        );
        let expected = [1_i32, 2, 3, 4, 5, 6, 7, 8]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut external = BTreeMap::new();
        for (rank, (owner, _)) in owners.iter().enumerate() {
            let bytes = &expected[rank * 8..(rank + 1) * 8];
            let lease = owner
                .allocator()
                .allocate(NonZeroUsize::new(bytes.len()).unwrap())
                .unwrap();
            lease.view().unwrap().copy_from(0, bytes).unwrap();
            external.insert((rank, source.nodes()[rank].index() as u64), lease);
        }
        let result = ShardedCudaExecutionEnvironment::new(external, 4)
            .execute(&plan)
            .unwrap();
        assert_eq!(result.trace.len(), 1);
        for rank in 0..4 {
            let output = result
                .outputs
                .get(&(rank, destination.nodes()[rank].index() as u64))
                .unwrap();
            let mut actual = vec![0; expected.len()];
            output.view().unwrap().copy_to(0, &mut actual).unwrap();
            assert_eq!(actual, expected, "replica {rank}");
        }
        assert!(mock.calls().contains(&"dtod_async"));
        assert!(mock.calls().contains(&"peer_copy"));
    }

    #[test]
    fn graph_redistribution_trace_replica_to_axis_uses_typed_destination_ranges() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let owners = (0..2)
            .map(|ordinal| {
                let device = driver.device(DeviceId(ordinal)).unwrap();
                let capability = device.capability().unwrap();
                (device.retain_primary_context().unwrap(), capability)
            })
            .collect::<Vec<_>>();
        let group = DeviceGroup::new(
            (0..2).map(|rank| crate::collective::DeviceId::new(format!("CUDA:{rank}")).unwrap()),
        )
        .unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [4, 2], DType::I32);
        let source = graph.replicate_node(input, group.clone()).unwrap();
        let destination = graph
            .redistribute_sharded(&source, group.clone(), Some(0))
            .unwrap();
        let bindings = owners
            .iter()
            .enumerate()
            .map(|(rank, (owner, capability))| CudaPlanBinding {
                device: group.devices()[rank].clone(),
                capability: capability.clone(),
                context: owner.clone(),
            })
            .collect::<Vec<_>>();
        let plan = executable_redistribution_plan(&source, &destination, &bindings).unwrap();
        let CudaPlanStage::Transfer { routes, .. } = &plan.logical.stages[0] else {
            panic!("expected typed transfer stage")
        };
        assert_eq!(routes.len(), 2);
        assert!(
            routes.iter().all(|route| route.source_rank == 1),
            "replicated trace selects its deterministic final source rank"
        );
        assert_eq!(routes[0].destination_element_offset, 0);
        assert_eq!(routes[1].destination_element_offset, 0);
        let expected = [1_i32, 2, 3, 4, 5, 6, 7, 8]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut external = BTreeMap::new();
        for (rank, (owner, _)) in owners.iter().enumerate() {
            let lease = owner
                .allocator()
                .allocate(NonZeroUsize::new(expected.len()).unwrap())
                .unwrap();
            lease.view().unwrap().copy_from(0, &expected).unwrap();
            external.insert((rank, source.nodes()[rank].index() as u64), lease);
        }
        let result = ShardedCudaExecutionEnvironment::new(external, 2)
            .execute(&plan)
            .unwrap();
        for rank in 0..2 {
            let output = result
                .outputs
                .get(&(rank, destination.nodes()[rank].index() as u64))
                .unwrap();
            let mut actual = vec![0; expected.len() / 2];
            output.view().unwrap().copy_to(0, &mut actual).unwrap();
            assert_eq!(actual, expected[rank * 16..(rank + 1) * 16]);
        }
    }

    #[test]
    fn graph_redistribution_trace_axis_to_axis_preserves_strided_global_ownership() {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let owners = (0..2)
            .map(|ordinal| {
                let device = driver.device(DeviceId(ordinal)).unwrap();
                let capability = device.capability().unwrap();
                (device.retain_primary_context().unwrap(), capability)
            })
            .collect::<Vec<_>>();
        let group = DeviceGroup::new(
            (0..2).map(|rank| crate::collective::DeviceId::new(format!("CUDA:{rank}")).unwrap()),
        )
        .unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [4, 4], DType::I32);
        let source = graph.shard_node(input, group.clone(), Some(0)).unwrap();
        let destination = graph
            .redistribute_sharded(&source, group.clone(), Some(1))
            .unwrap();
        let bindings = owners
            .iter()
            .enumerate()
            .map(|(rank, (owner, capability))| CudaPlanBinding {
                device: group.devices()[rank].clone(),
                capability: capability.clone(),
                context: owner.clone(),
            })
            .collect::<Vec<_>>();
        let plan = executable_redistribution_plan(&source, &destination, &bindings).unwrap();
        let CudaPlanStage::Transfer { routes, .. } = &plan.logical.stages[0] else {
            panic!("expected typed transfer stage")
        };
        assert_eq!(
            routes.len(),
            8,
            "each row has one route from each source rank"
        );
        assert!(
            routes
                .iter()
                .all(|route| route.bytes == 8 && route.elements == 2)
        );
        let full = (1_i32..=16).flat_map(i32::to_le_bytes).collect::<Vec<_>>();
        let mut external = BTreeMap::new();
        for (rank, (owner, _)) in owners.iter().enumerate() {
            let bytes = &full[rank * 32..(rank + 1) * 32];
            let lease = owner
                .allocator()
                .allocate(NonZeroUsize::new(bytes.len()).unwrap())
                .unwrap();
            lease.view().unwrap().copy_from(0, bytes).unwrap();
            external.insert((rank, source.nodes()[rank].index() as u64), lease);
        }
        let result = ShardedCudaExecutionEnvironment::new(external, 2)
            .execute(&plan)
            .unwrap();
        let expected = [
            [1_i32, 2, 5, 6, 9, 10, 13, 14],
            [3_i32, 4, 7, 8, 11, 12, 15, 16],
        ];
        for (rank, values) in expected.into_iter().enumerate() {
            let output = result
                .outputs
                .get(&(rank, destination.nodes()[rank].index() as u64))
                .unwrap();
            let mut actual = vec![0; 32];
            output.view().unwrap().copy_to(0, &mut actual).unwrap();
            assert_eq!(
                actual,
                values
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>()
            );
        }
    }
}
