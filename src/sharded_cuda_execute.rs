//! Phase 3B1 local PTX realization for a validated executable sharded CUDA plan.
use crate::{
    ConcurrentPtxCache, CudaPlanStage, Error, ExecutableBufferRole, ExecutableShardedCudaPlan,
    PrimaryBufferLease, PtxBinding,
};
use std::collections::BTreeMap;
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
            !matches!(
                stage,
                CudaPlanStage::Local {
                    diagnostic: None,
                    ..
                }
            )
        }) {
            return Err(err("Phase 3B1 only executes supported local PTX stages"));
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
