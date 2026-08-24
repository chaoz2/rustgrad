//! Phase 3B1 local PTX realization for a validated executable sharded CUDA plan.
use crate::{
    ConcurrentPtxCache, CudaPlanStage, Error, ExecutableBufferRole, ExecutableShardedCudaPlan,
    PrimaryBufferLease, PtxBinding,
};
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;

pub struct ShardedCudaExecutionEnvironment {
    pub external: BTreeMap<(usize, u64), PrimaryBufferLease>,
    caches: Vec<ConcurrentPtxCache>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardedCudaExecutionTrace {
    pub stage: usize,
    pub action: &'static str,
    pub skipped: bool,
}
pub struct ShardedCudaExecutionResult {
    pub outputs: BTreeMap<(usize, u64), PrimaryBufferLease>,
    pub trace: Vec<ShardedCudaExecutionTrace>,
}
impl ShardedCudaExecutionEnvironment {
    pub fn new(external: BTreeMap<(usize, u64), PrimaryBufferLease>, owners: usize) -> Self {
        Self {
            external,
            caches: (0..owners).map(|_| ConcurrentPtxCache::new()).collect(),
        }
    }
    pub fn execute(
        &mut self,
        plan: &ExecutableShardedCudaPlan,
    ) -> Result<ShardedCudaExecutionResult, Error> {
        plan.validate()?;
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
        let actual_external = self.external.keys().copied().collect::<BTreeSet<_>>();
        if actual_external != expected_external {
            return Err(err("external sharded CUDA bindings are missing or extra"));
        }
        let mut leases = std::mem::take(&mut self.external);
        let mut trace = Vec::new();
        for buffer in &plan.buffers {
            let key = (buffer.rank, buffer.buffer);
            if matches!(buffer.role, ExecutableBufferRole::External) {
                let lease = leases
                    .get(&key)
                    .ok_or_else(|| err("missing external sharded CUDA lease"))?;
                let (owner, bytes, _, _) =
                    lease.execution_metadata().map_err(|e| err(e.to_string()))?;
                if owner != buffer.owner_identity || bytes < buffer.bytes {
                    return Err(err("external lease owner or bytes mismatch"));
                }
            } else if buffer.bytes > 0 {
                let allocator = plan.owners[buffer.rank].allocator();
                leases.insert(
                    key,
                    allocator
                        .allocate(NonZeroUsize::new(buffer.bytes).unwrap())
                        .map_err(|e| err(e.to_string()))?,
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
                        .get(&(rank, abi.id))
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
        for buffer in &plan.buffers {
            if matches!(buffer.role, ExecutableBufferRole::Output)
                && let Some(lease) = leases.remove(&(buffer.rank, buffer.buffer))
            {
                outputs.insert((buffer.rank, buffer.buffer), lease);
            }
        }
        self.external = leases;
        Ok(ShardedCudaExecutionResult { outputs, trace })
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
    use crate::{
        CudaPlanStage, CudaTransferRoute, DType, DeviceId, Driver, ExecutableBuffer, Graph,
        PtxRenderer, Shape, ShardedCudaPlan, lower_graph_elementwise,
    };
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
        let plan = ExecutableShardedCudaPlan {
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
                routes: vec![route(0, device0.clone(), 11), route(1, device1.clone(), 20)],
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
}
