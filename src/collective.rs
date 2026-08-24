//! Deterministic, backend-neutral planning for the tinygrad multi-device
//! reduction boundary. CUDA/NCCL execution deliberately lives elsewhere.

use crate::ptx::PrimaryCollectiveAddCache;
use crate::{
    CudaError, PeerAccess, PrimaryBufferLease, PrimaryContext, PrimaryCudaAllocator, Stream,
};
use crate::{DType, Error, Result, Scalar, TensorData};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::num::NonZeroUsize;

/// Stable semantic device identity. It intentionally contains no runtime handle.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct DeviceId(String);
impl DeviceId {
    pub fn new(identity: impl Into<String>) -> Result<Self> {
        let identity = identity.into();
        if identity.is_empty() {
            return Err(err("device identity must not be empty"));
        }
        Ok(Self(identity))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Ordered collective membership. Caller order is semantic and never inferred from handles.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct DeviceGroup {
    devices: Vec<DeviceId>,
}
impl DeviceGroup {
    pub fn new(devices: impl IntoIterator<Item = DeviceId>) -> Result<Self> {
        let devices: Vec<_> = devices.into_iter().collect();
        if devices.is_empty() {
            return Err(err("device group must not be empty"));
        }
        if devices.iter().collect::<BTreeSet<_>>().len() != devices.len() {
            return Err(err("duplicate device identity"));
        }
        Ok(Self { devices })
    }
    pub fn devices(&self) -> &[DeviceId] {
        &self.devices
    }
    pub fn len(&self) -> usize {
        self.devices.len()
    }
    pub const fn is_empty(&self) -> bool {
        false
    }
    pub fn index_of(&self, device: &DeviceId) -> Option<usize> {
        self.devices.iter().position(|x| x == device)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum Reduction {
    Sum,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum CollectiveKind {
    Broadcast { root: DeviceId },
    AllGather,
    ReduceScatter { reduction: Reduction },
    AllReduce { reduction: Reduction },
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum StreamLane {
    Copy,
    Compute,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct LogicalRange {
    pub start: usize,
    pub len: usize,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum ActionOp {
    LocalCopy,
    Transfer,
    ReduceSum,
}
/// A serializable action. Dependencies reference preceding action ids only.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct CollectiveAction {
    pub id: usize,
    pub depends_on: Vec<usize>,
    pub op: ActionOp,
    pub source: DeviceId,
    pub destination: DeviceId,
    pub range: LogicalRange,
    pub stream: StreamLane,
    pub dtype: DType,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct CollectiveRequest {
    pub group: DeviceGroup,
    pub kind: CollectiveKind,
    pub dtype: DType,
    /// Per-device input lengths. Equal lengths are required except all-gather.
    pub input_lengths: Vec<usize>,
}
/// Immutable plan. Chunks use quotient/remainder partitioning: chunk `i` is
/// `[i*q + min(i,r), (i+1)*q + min(i+1,r))` for `count = q*n + r`.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct CollectivePlan {
    pub request: CollectiveRequest,
    pub output_lengths: Vec<usize>,
    pub chunks: Vec<LogicalRange>,
    pub actions: Vec<CollectiveAction>,
    pub cache_key: String,
}
impl CollectivePlan {
    pub fn cache_key(&self) -> &str {
        &self.cache_key
    }
}

pub struct CollectivePlanner;
impl CollectivePlanner {
    pub fn plan(request: CollectiveRequest) -> Result<CollectivePlan> {
        validate(&request)?;
        let n = request.group.len();
        let total = match request.kind {
            CollectiveKind::AllGather => {
                request.input_lengths.iter().try_fold(0usize, |a, b| {
                    a.checked_add(*b)
                        .ok_or_else(|| err("element count overflow"))
                })?
            }
            _ => request.input_lengths[0],
        };
        let chunks = chunks(total, n);
        let output_lengths = match request.kind {
            CollectiveKind::AllGather
            | CollectiveKind::AllReduce { .. }
            | CollectiveKind::Broadcast { .. } => vec![total; n],
            CollectiveKind::ReduceScatter { .. } => chunks.iter().map(|c| c.len).collect(),
        };
        let mut actions = Vec::new();
        let mut add =
            |op, source: usize, destination: usize, range, stream, depends_on: Vec<usize>| {
                let id = actions.len();
                actions.push(CollectiveAction {
                    id,
                    depends_on,
                    op,
                    source: request.group.devices[source].clone(),
                    destination: request.group.devices[destination].clone(),
                    range,
                    stream,
                    dtype: request.dtype,
                });
                id
            };
        match &request.kind {
            CollectiveKind::Broadcast { root } => {
                let root = request
                    .group
                    .index_of(root)
                    .ok_or_else(|| err("broadcast root is not in group"))?;
                add(
                    ActionOp::LocalCopy,
                    root,
                    root,
                    LogicalRange {
                        start: 0,
                        len: total,
                    },
                    StreamLane::Copy,
                    vec![],
                );
                for dst in 0..n {
                    if dst != root {
                        let t = add(
                            ActionOp::Transfer,
                            root,
                            dst,
                            LogicalRange {
                                start: 0,
                                len: total,
                            },
                            StreamLane::Copy,
                            vec![],
                        );
                        add(
                            ActionOp::LocalCopy,
                            root,
                            dst,
                            LogicalRange {
                                start: 0,
                                len: total,
                            },
                            StreamLane::Copy,
                            vec![t],
                        );
                    }
                }
            }
            CollectiveKind::AllGather => {
                let mut offset = 0;
                for src in 0..n {
                    let range = LogicalRange {
                        start: offset,
                        len: request.input_lengths[src],
                    };
                    offset += range.len;
                    for dst in 0..n {
                        if src == dst {
                            add(
                                ActionOp::LocalCopy,
                                src,
                                dst,
                                range,
                                StreamLane::Copy,
                                vec![],
                            );
                        } else {
                            let t = add(
                                ActionOp::Transfer,
                                src,
                                dst,
                                range,
                                StreamLane::Copy,
                                vec![],
                            );
                            add(
                                ActionOp::LocalCopy,
                                src,
                                dst,
                                range,
                                StreamLane::Copy,
                                vec![t],
                            );
                        }
                    }
                }
            }
            CollectiveKind::AllReduce { .. } | CollectiveKind::ReduceScatter { .. } => {
                let targets: Vec<usize> = match request.kind {
                    CollectiveKind::AllReduce { .. } => (0..n).collect(),
                    _ => (0..n).collect(),
                };
                for dst in targets {
                    let range = match request.kind {
                        CollectiveKind::AllReduce { .. } => LogicalRange {
                            start: 0,
                            len: total,
                        },
                        _ => chunks[dst],
                    };
                    let mut last = add(
                        ActionOp::LocalCopy,
                        dst,
                        dst,
                        range,
                        StreamLane::Copy,
                        vec![],
                    );
                    for src in 0..n {
                        if src != dst {
                            let t = add(
                                ActionOp::Transfer,
                                src,
                                dst,
                                range,
                                StreamLane::Copy,
                                vec![],
                            );
                            last = add(
                                ActionOp::ReduceSum,
                                src,
                                dst,
                                range,
                                StreamLane::Compute,
                                vec![last, t],
                            );
                        }
                    }
                }
            }
        }
        let mut plan = CollectivePlan {
            request,
            output_lengths,
            chunks,
            actions,
            cache_key: String::new(),
        };
        plan.cache_key = serde_json::to_string(&plan)
            .map_err(|e| err(format!("cache key serialization failed: {e}")))?;
        Ok(plan)
    }
}

/// Backend boundary for a validated plan. Phase 2 will implement this for CUDA.
pub trait CollectiveExecutor {
    fn execute(&self, plan: &CollectivePlan, inputs: &[TensorData]) -> Result<Vec<TensorData>>;
}

/// Sequential two-owner CUDA realization of an all-reduce plan.  It is kept
/// deliberately separate from the dense oracle: every nonempty plan action is
/// a Driver copy or a typed primary PTX add launch.
pub struct CudaCollectiveGroup {
    devices: [DeviceId; 2],
    contexts: [PrimaryContext; 2],
    allocators: [std::sync::Arc<PrimaryCudaAllocator>; 2],
    streams: [Stream; 2],
    peers: [PeerAccess; 2],
    adds: [PrimaryCollectiveAddCache; 2],
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaCollectiveTrace {
    pub action_id: usize,
    pub operation: &'static str,
    pub device: DeviceId,
    pub range: LogicalRange,
    pub cache_key: Option<String>,
}
impl CudaCollectiveGroup {
    pub fn new(bindings: [(DeviceId, PrimaryContext); 2]) -> Result<Self> {
        if bindings[0].0 == bindings[1].0 || bindings[0].1.identity() == bindings[1].1.identity() {
            return Err(err(
                "CUDA collective group requires two distinct identities and primary owners",
            ));
        }
        let a0 = bindings[0].1.allocator();
        let a1 = bindings[1].1.allocator();
        let s0 = bindings[0].1.stream().map_err(cuda_err)?;
        let s1 = bindings[1].1.stream().map_err(cuda_err)?;
        let p01 = bindings[0]
            .1
            .peer_access_to(&bindings[1].1)
            .map_err(cuda_err)?;
        let p10 = bindings[1]
            .1
            .peer_access_to(&bindings[0].1)
            .map_err(cuda_err)?;
        Ok(Self {
            devices: [bindings[0].0.clone(), bindings[1].0.clone()],
            contexts: [bindings[0].1.clone(), bindings[1].1.clone()],
            allocators: [a0, a1],
            streams: [s0, s1],
            peers: [p01, p10],
            adds: [
                PrimaryCollectiveAddCache::new(),
                PrimaryCollectiveAddCache::new(),
            ],
        })
    }
    /// Mutates the two input leases in place and returns an inspectable trace.
    pub fn all_reduce_sum(
        &self,
        plan: &CollectivePlan,
        inputs: [&PrimaryBufferLease; 2],
    ) -> Result<Vec<CudaCollectiveTrace>> {
        if !matches!(
            plan.request.kind,
            CollectiveKind::AllReduce {
                reduction: Reduction::Sum
            }
        ) || plan.request.group.devices() != self.devices
            || plan.request.input_lengths.len() != 2
        {
            return Err(err(
                "CUDA Phase 2B1 supports exactly this group's two-device sum all-reduce",
            ));
        }
        let dtype = plan.request.dtype;
        if !matches!(
            dtype,
            DType::I8
                | DType::U8
                | DType::I32
                | DType::U32
                | DType::I64
                | DType::U64
                | DType::F32
                | DType::F64
        ) {
            return Err(Error::UnsupportedDType { dtype });
        }
        let count = plan.request.input_lengths[0];
        if plan.request.input_lengths[1] != count {
            return Err(err("all-reduce input length mismatch"));
        }
        let bytes = count
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| err("collective byte count overflow"))?;
        if inputs
            .iter()
            .any(|x| x.view().map_or(true, |v| v.len() < bytes))
        {
            return Err(err("collective input lease is too small"));
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let scratch = [
            self.allocators[0]
                .allocate(NonZeroUsize::new(bytes).unwrap())
                .map_err(cuda_err)?,
            self.allocators[1]
                .allocate(NonZeroUsize::new(bytes).unwrap())
                .map_err(cuda_err)?,
        ];
        // In-place outputs must not become later transfer sources: preserve
        // each rank's original contribution before the first reduction.
        let originals = [
            self.allocators[0]
                .allocate(NonZeroUsize::new(bytes).unwrap())
                .map_err(cuda_err)?,
            self.allocators[1]
                .allocate(NonZeroUsize::new(bytes).unwrap())
                .map_err(cuda_err)?,
        ];
        for index in 0..2 {
            let original = originals[index].view().map_err(cuda_err)?;
            let input = inputs[index].view().map_err(cuda_err)?;
            original
                .copy_from_view(0, &input, 0, bytes)
                .map_err(cuda_err)?;
        }
        let mut trace = Vec::new();
        let mut completed = vec![false; plan.actions.len()];
        for action in &plan.actions {
            if action
                .depends_on
                .iter()
                .any(|d| *d >= action.id || !completed[*d])
            {
                return Err(err("invalid collective action dependency"));
            }
            let src = self
                .devices
                .iter()
                .position(|d| d == &action.source)
                .ok_or_else(|| err("unknown action source"))?;
            let dst = self
                .devices
                .iter()
                .position(|d| d == &action.destination)
                .ok_or_else(|| err("unknown action destination"))?;
            let off = action
                .range
                .start
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| err("range overflow"))?;
            let n = action
                .range
                .len
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| err("range overflow"))?;
            if n != 0 {
                let action_error =
                    |operation: &'static str, error: CudaError| Error::CollectiveAction {
                        action_id: action.id,
                        operation,
                        reason: error.to_string(),
                    };
                match action.op {
                    ActionOp::LocalCopy => {
                        let destination = inputs[dst]
                            .view()
                            .map_err(|e| action_error("local-copy", e))?;
                        if src == dst {
                            let original = originals[src]
                                .view()
                                .map_err(|e| action_error("local-copy", e))?;
                            destination
                                .copy_from_view(off, &original, off, n)
                                .map_err(|e| action_error("local-copy", e))?;
                        } else {
                            let staged = scratch[dst]
                                .view()
                                .map_err(|e| action_error("local-copy", e))?;
                            destination
                                .copy_from_view(off, &staged, off, n)
                                .map_err(|e| action_error("local-copy", e))?;
                        }
                        trace.push(CudaCollectiveTrace {
                            action_id: action.id,
                            operation: "local-copy",
                            device: self.devices[dst].clone(),
                            range: action.range,
                            cache_key: None,
                        });
                    }
                    ActionOp::Transfer => {
                        let peer = if src == 0 {
                            &self.peers[0]
                        } else {
                            &self.peers[1]
                        };
                        let mut t = scratch[dst]
                            .copy_from_peer_async(
                                off,
                                peer,
                                &originals[src],
                                off,
                                n,
                                &self.streams[dst],
                            )
                            .map_err(|e| action_error("peer-copy", e))?;
                        t.wait().map_err(|e| action_error("peer-copy", e))?;
                        trace.push(CudaCollectiveTrace {
                            action_id: action.id,
                            operation: "peer-copy",
                            device: self.devices[dst].clone(),
                            range: action.range,
                            cache_key: None,
                        });
                    }
                    ActionOp::ReduceSum => {
                        let k = self.adds[dst]
                            .get_or_load(&self.contexts[dst], dtype)
                            .map_err(|e| Error::CollectiveAction {
                                action_id: action.id,
                                operation: "add",
                                reason: e.to_string(),
                            })?;
                        k.launch(
                            inputs[dst],
                            action.range.start,
                            &scratch[dst],
                            action.range.start,
                            action.range.len,
                            &self.streams[dst],
                            true,
                        )
                        .map_err(|e| Error::CollectiveAction {
                            action_id: action.id,
                            operation: "add",
                            reason: e.to_string(),
                        })?;
                        trace.push(CudaCollectiveTrace {
                            action_id: action.id,
                            operation: "add",
                            device: self.devices[dst].clone(),
                            range: action.range,
                            cache_key: Some(k.rendered().cache_key.clone()),
                        });
                    }
                }
            }
            completed[action.id] = true;
        }
        Ok(trace)
    }
}
fn cuda_err(error: CudaError) -> Error {
    err(error.to_string())
}
#[derive(Clone, Copy, Debug, Default)]
pub struct InMemoryCollectiveExecutor;
impl CollectiveExecutor for InMemoryCollectiveExecutor {
    fn execute(&self, plan: &CollectivePlan, inputs: &[TensorData]) -> Result<Vec<TensorData>> {
        let r = &plan.request;
        validate(r)?;
        if inputs.len() != r.group.len() {
            return Err(err("input/group count mismatch"));
        }
        for (i, input) in inputs.iter().enumerate() {
            if input.dtype() != r.dtype {
                return Err(err("input dtype does not match plan"));
            }
            if input.len() != r.input_lengths[i] {
                return Err(err("input length does not match plan"));
            }
        }
        let n = inputs.len();
        // Reduce-scatter actions use global logical offsets before extracting
        // each local shard, so the execution workspace is the full input.
        let max = plan
            .output_lengths
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(r.input_lengths.iter().copied().max().unwrap_or(0));
        let mut output: Vec<Vec<Scalar>> = (0..n).map(|_| vec![Scalar::I(0); max]).collect();
        let mut inbox: Vec<Vec<Scalar>> = (0..n).map(|_| vec![Scalar::I(0); max]).collect();
        let mut done = vec![false; plan.actions.len()];
        for action in &plan.actions {
            if action.id >= done.len()
                || action
                    .depends_on
                    .iter()
                    .any(|id| *id >= action.id || !done[*id])
            {
                return Err(err("invalid action dependency trace"));
            }
            let src = r
                .group
                .index_of(&action.source)
                .ok_or_else(|| err("action source outside group"))?;
            let dst = r
                .group
                .index_of(&action.destination)
                .ok_or_else(|| err("action destination outside group"))?;
            let range = action.range;
            let source_base = source_base(r, src, range.start)?;
            match action.op {
                ActionOp::Transfer => {
                    for j in 0..range.len {
                        inbox[dst][range.start + j] = inputs[src].scalar_at(source_base + j);
                    }
                }
                ActionOp::LocalCopy => {
                    for j in 0..range.len {
                        output[dst][range.start + j] = if src == dst {
                            inputs[src].scalar_at(source_base + j)
                        } else {
                            inbox[dst][range.start + j]
                        };
                    }
                }
                ActionOp::ReduceSum => {
                    for j in 0..range.len {
                        output[dst][range.start + j] = add(
                            output[dst][range.start + j],
                            inbox[dst][range.start + j],
                            r.dtype,
                        );
                    }
                }
            }
            done[action.id] = true;
        }
        (0..n)
            .map(|device| {
                let values: Vec<_> = match r.kind {
                    CollectiveKind::ReduceScatter { .. } => {
                        let c = plan.chunks[device];
                        output[device][c.start..c.start + c.len].to_vec()
                    }
                    _ => output[device][..plan.output_lengths[device]].to_vec(),
                };
                TensorData::from_scalars([values.len()], r.dtype, values)
            })
            .collect()
    }
}
fn validate(r: &CollectiveRequest) -> Result<()> {
    if r.input_lengths.len() != r.group.len() {
        return Err(err("input lengths/group count mismatch"));
    }
    if !matches!(r.kind, CollectiveKind::AllGather)
        && r.input_lengths.windows(2).any(|x| x[0] != x[1])
    {
        return Err(err("collective requires equal input lengths"));
    }
    if matches!(
        r.kind,
        CollectiveKind::AllReduce { .. } | CollectiveKind::ReduceScatter { .. }
    ) && !r.dtype.is_float()
        && !r.dtype.is_integer()
        && r.dtype != DType::Bool
    {
        return Err(err("unsupported reduction dtype"));
    }
    if let CollectiveKind::Broadcast { ref root } = r.kind
        && r.group.index_of(root).is_none()
    {
        return Err(err("broadcast root is not in group"));
    }
    Ok(())
}
fn chunks(count: usize, n: usize) -> Vec<LogicalRange> {
    let q = count / n;
    let rem = count % n;
    (0..n)
        .map(|i| {
            let start = i * q + i.min(rem);
            LogicalRange {
                start,
                len: q + usize::from(i < rem),
            }
        })
        .collect()
}
fn source_base(r: &CollectiveRequest, source: usize, start: usize) -> Result<usize> {
    match r.kind {
        CollectiveKind::AllGather => {
            let base: usize = r.input_lengths[..source].iter().sum();
            start
                .checked_sub(base)
                .ok_or_else(|| err("invalid all-gather range"))
        }
        _ => Ok(start),
    }
}
fn add(a: Scalar, b: Scalar, dtype: DType) -> Scalar {
    let raw = if dtype.is_float() {
        Scalar::F(a.as_f64() + b.as_f64())
    } else if dtype == DType::Bool {
        Scalar::Bool(a.as_bool() || b.as_bool())
    } else if dtype.category() == crate::DTypeCategory::Unsigned {
        Scalar::U(a.as_u64().wrapping_add(b.as_u64()))
    } else {
        Scalar::I(a.as_i64().wrapping_add(b.as_i64()))
    };
    // Re-materialize after every action: F16/BF16 and narrow integers have
    // the same intermediate-storage semantics as the dense CPU oracle.
    TensorData::from_scalars([1], dtype, [raw])
        .expect("one scalar is a valid dense tensor")
        .scalar_at(0)
}
fn err(reason: impl AsRef<str>) -> Error {
    Error::Collective {
        reason: reason.as_ref().to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn group(n: usize) -> DeviceGroup {
        DeviceGroup::new((0..n).map(|i| DeviceId::new(format!("CPU:{i}")).unwrap())).unwrap()
    }
    fn input(n: usize, len: usize, dtype: DType) -> Vec<TensorData> {
        (0..n)
            .map(|d| {
                TensorData::from_scalars(
                    [len],
                    dtype,
                    (0..len).map(|i| Scalar::I((d * 10 + i) as i64)),
                )
                .unwrap()
            })
            .collect()
    }
    #[test]
    fn table_driven_collectives_and_traces() {
        for (name, kind, n, len, dtype) in [
            (
                "one",
                CollectiveKind::AllReduce {
                    reduction: Reduction::Sum,
                },
                1,
                0,
                DType::I8,
            ),
            (
                "two",
                CollectiveKind::AllReduce {
                    reduction: Reduction::Sum,
                },
                2,
                5,
                DType::I16,
            ),
            (
                "three",
                CollectiveKind::ReduceScatter {
                    reduction: Reduction::Sum,
                },
                3,
                5,
                DType::F16,
            ),
            (
                "four",
                CollectiveKind::Broadcast {
                    root: DeviceId::new("CPU:0").unwrap(),
                },
                4,
                3,
                DType::Bool,
            ),
        ] {
            let p = CollectivePlanner::plan(CollectiveRequest {
                group: group(n),
                kind,
                dtype,
                input_lengths: vec![len; n],
            })
            .unwrap();
            assert_eq!(
                p.cache_key,
                CollectivePlanner::plan(p.request.clone())
                    .unwrap()
                    .cache_key,
                "{name}"
            );
            assert!(
                p.actions
                    .iter()
                    .enumerate()
                    .all(|(i, a)| a.id == i && a.depends_on.iter().all(|d| *d < i))
            );
            let out = InMemoryCollectiveExecutor
                .execute(&p, &input(n, len, dtype))
                .unwrap();
            assert_eq!(out.len(), n, "{name}");
        }
    }
    #[test]
    fn all_gather_uneven_and_validation() {
        let p = CollectivePlanner::plan(CollectiveRequest {
            group: group(3),
            kind: CollectiveKind::AllGather,
            dtype: DType::I32,
            input_lengths: vec![1, 0, 2],
        })
        .unwrap();
        let v = vec![
            TensorData::from_scalars([1], DType::I32, [Scalar::I(1)]).unwrap(),
            TensorData::from_scalars([0], DType::I32, []).unwrap(),
            TensorData::from_scalars([2], DType::I32, [Scalar::I(3), Scalar::I(4)]).unwrap(),
        ];
        assert_eq!(
            InMemoryCollectiveExecutor.execute(&p, &v).unwrap()[2].to_vec_f64(),
            vec![1., 3., 4.]
        );
        assert!(
            DeviceGroup::new([DeviceId::new("x").unwrap(), DeviceId::new("x").unwrap()]).is_err()
        );
        assert!(
            CollectivePlanner::plan(CollectiveRequest {
                group: group(2),
                kind: CollectiveKind::AllReduce {
                    reduction: Reduction::Sum
                },
                dtype: DType::I32,
                input_lengths: vec![1, 2]
            })
            .is_err()
        );
    }

    fn native_sum(dtype: DType, a: &[u8], b: &[u8]) -> Vec<u8> {
        a.chunks_exact(dtype.itemsize())
            .zip(b.chunks_exact(dtype.itemsize()))
            .flat_map(|(a, b)| match dtype {
                DType::I8 => (i8::from_ne_bytes([a[0]]).wrapping_add(i8::from_ne_bytes([b[0]]))
                    as u8)
                    .to_ne_bytes()
                    .to_vec(),
                DType::U8 => a[0].wrapping_add(b[0]).to_ne_bytes().to_vec(),
                DType::I32 => i32::from_ne_bytes(a.try_into().unwrap())
                    .wrapping_add(i32::from_ne_bytes(b.try_into().unwrap()))
                    .to_ne_bytes()
                    .to_vec(),
                DType::U32 => u32::from_ne_bytes(a.try_into().unwrap())
                    .wrapping_add(u32::from_ne_bytes(b.try_into().unwrap()))
                    .to_ne_bytes()
                    .to_vec(),
                DType::I64 => i64::from_ne_bytes(a.try_into().unwrap())
                    .wrapping_add(i64::from_ne_bytes(b.try_into().unwrap()))
                    .to_ne_bytes()
                    .to_vec(),
                DType::U64 => u64::from_ne_bytes(a.try_into().unwrap())
                    .wrapping_add(u64::from_ne_bytes(b.try_into().unwrap()))
                    .to_ne_bytes()
                    .to_vec(),
                DType::F32 => (f32::from_ne_bytes(a.try_into().unwrap())
                    + f32::from_ne_bytes(b.try_into().unwrap()))
                .to_ne_bytes()
                .to_vec(),
                DType::F64 => (f64::from_ne_bytes(a.try_into().unwrap())
                    + f64::from_ne_bytes(b.try_into().unwrap()))
                .to_ne_bytes()
                .to_vec(),
                _ => unreachable!(),
            })
            .collect()
    }
    fn fixture() -> (
        std::sync::Arc<crate::cuda::tests::Mock>,
        CudaCollectiveGroup,
        [PrimaryBufferLease; 2],
        CollectivePlan,
        [PrimaryContext; 2],
    ) {
        use crate::Driver;
        use crate::cuda::tests::Mock;
        use std::{num::NonZeroUsize, sync::Arc};
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let first = driver
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let second = driver
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let group = group(2);
        let plan = CollectivePlanner::plan(CollectiveRequest {
            group: group.clone(),
            kind: CollectiveKind::AllReduce {
                reduction: Reduction::Sum,
            },
            dtype: DType::I32,
            input_lengths: vec![3, 3],
        })
        .unwrap();
        let pools = [first.allocator(), second.allocator()];
        let inputs = [
            pools[0].allocate(NonZeroUsize::new(24).unwrap()).unwrap(),
            pools[1].allocate(NonZeroUsize::new(24).unwrap()).unwrap(),
        ];
        let executor = CudaCollectiveGroup::new([
            (group.devices()[0].clone(), first.clone()),
            (group.devices()[1].clone(), second.clone()),
        ])
        .unwrap();
        (mock, executor, inputs, plan, [first, second])
    }
    fn write(
        mock: &crate::cuda::tests::Mock,
        primary: &PrimaryContext,
        lease: &PrimaryBufferLease,
        bytes: &[u8],
    ) {
        let view = lease.view().unwrap();
        let descriptor = mock
            .allocation_descriptor(primary.owner(), view.device_ptr().unwrap())
            .unwrap();
        mock.write_allocation(primary.owner(), descriptor, 0, bytes)
            .unwrap();
    }
    fn read(
        mock: &crate::cuda::tests::Mock,
        primary: &PrimaryContext,
        lease: &PrimaryBufferLease,
        bytes: usize,
    ) -> Vec<u8> {
        let view = lease.view().unwrap();
        let desc = mock
            .allocation_descriptor(primary.owner(), view.device_ptr().unwrap())
            .unwrap();
        mock.allocation_snapshot(primary.owner(), desc).unwrap()[..bytes].to_vec()
    }

    #[test]
    fn cuda_two_device_all_reduce_matrix_plan_trace_and_cache_reuse() {
        let cases = [
            (DType::I8, vec![127, 2, 3], vec![1, 4, 5]),
            (DType::U8, vec![255, 2, 3], vec![1, 4, 5]),
            (
                DType::I32,
                [i32::MAX, 2, 3]
                    .into_iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect(),
                [1_i32, 4, 5]
                    .into_iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect(),
            ),
            (
                DType::U32,
                [u32::MAX, 2, 3]
                    .into_iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect(),
                [1_u32, 4, 5]
                    .into_iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect(),
            ),
            (
                DType::I64,
                [i64::MAX, 2, 3]
                    .into_iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect(),
                [1_i64, 4, 5]
                    .into_iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect(),
            ),
            (
                DType::U64,
                [u64::MAX, 2, 3]
                    .into_iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect(),
                [1_u64, 4, 5]
                    .into_iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect(),
            ),
            (
                DType::F32,
                [1.5_f32, 2., 3.]
                    .into_iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect(),
                [2.25_f32, 4., 5.]
                    .into_iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect(),
            ),
            (
                DType::F64,
                [1.5_f64, 2., 3.]
                    .into_iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect(),
                [2.25_f64, 4., 5.]
                    .into_iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect(),
            ),
        ];
        for (dtype, left, right) in cases {
            let (mock, executor, inputs, mut plan, primaries) = fixture();
            plan.request.dtype = dtype;
            for action in &mut plan.actions {
                action.dtype = dtype;
            }
            let expected = native_sum(dtype, &left, &right);
            write(&mock, &primaries[0], &inputs[0], &left);
            write(&mock, &primaries[1], &inputs[1], &right);
            let trace = executor
                .all_reduce_sum(&plan, [&inputs[0], &inputs[1]])
                .unwrap();
            assert_eq!(
                read(&mock, &primaries[0], &inputs[0], expected.len()),
                expected,
                "{dtype:?}"
            );
            assert_eq!(
                read(&mock, &primaries[1], &inputs[1], expected.len()),
                expected,
                "{dtype:?}"
            );
            assert_eq!(trace.len(), plan.actions.len());
            for (action, observed) in plan.actions.iter().zip(&trace) {
                assert_eq!(observed.action_id, action.id);
                assert_eq!(observed.range, action.range);
                assert_eq!(
                    observed.operation,
                    match action.op {
                        ActionOp::LocalCopy => "local-copy",
                        ActionOp::Transfer => "peer-copy",
                        ActionOp::ReduceSum => "add",
                    }
                );
                assert!(action.depends_on.iter().all(|id| *id < action.id));
            }
            let trace_again = executor
                .all_reduce_sum(&plan, [&inputs[0], &inputs[1]])
                .unwrap();
            assert_eq!(trace_again.len(), trace.len());
            let calls = mock.calls();
            assert_eq!(
                calls.iter().filter(|x| **x == "module_load").count(),
                2,
                "{dtype:?}"
            );
            assert_eq!(
                calls.iter().filter(|x| **x == "peer_copy").count(),
                4,
                "{dtype:?}"
            );
            assert_eq!(
                calls.iter().filter(|x| **x == "launch").count(),
                4,
                "{dtype:?}"
            );
            let driver_order: Vec<_> = calls
                .into_iter()
                .filter(|call| matches!(*call, "dtod" | "peer_copy" | "launch"))
                .collect();
            assert_eq!(
                driver_order,
                [
                    "dtod",
                    "dtod", // immutable rank snapshots
                    "dtod",
                    "peer_copy",
                    "launch",
                    "dtod",
                    "peer_copy",
                    "launch",
                    "dtod",
                    "dtod", // second execution snapshots
                    "dtod",
                    "peer_copy",
                    "launch",
                    "dtod",
                    "peer_copy",
                    "launch",
                ],
                "{dtype:?}"
            );
        }
        let (mock, executor, inputs, mut plan, primaries) = fixture();
        plan.request.input_lengths = vec![0, 0];
        plan.actions.clear();
        assert!(
            executor
                .all_reduce_sum(&plan, [&inputs[0], &inputs[1]])
                .unwrap()
                .is_empty()
        );
        assert!(mock.calls().iter().all(|call| *call != "launch"));
        drop((executor, inputs, primaries));
        let calls = mock.calls();
        assert!(
            calls.iter().rposition(|x| *x == "peer_disable").unwrap()
                < calls.iter().position(|x| *x == "primary_release").unwrap()
        );
    }

    #[test]
    fn cuda_two_device_all_reduce_rejects_unsupported_dtypes_before_mutation() {
        for dtype in [DType::Bool, DType::I16, DType::U16, DType::F16, DType::BF16] {
            let (mock, executor, inputs, mut plan, primaries) = fixture();
            plan.request.dtype = dtype;
            for action in &mut plan.actions {
                action.dtype = dtype;
            }
            let before = [vec![0xa5; 24], vec![0x5a; 24]];
            write(&mock, &primaries[0], &inputs[0], &before[0]);
            write(&mock, &primaries[1], &inputs[1], &before[1]);
            let allocs = [
                mock.live_allocation_count(primaries[0].owner()),
                mock.live_allocation_count(primaries[1].owner()),
            ];
            assert!(
                matches!(executor.all_reduce_sum(&plan, [&inputs[0], &inputs[1]]), Err(Error::UnsupportedDType { dtype: actual }) if actual == dtype)
            );
            assert_eq!(mock.live_allocation_count(primaries[0].owner()), allocs[0]);
            assert_eq!(mock.live_allocation_count(primaries[1].owner()), allocs[1]);
            assert_eq!(read(&mock, &primaries[0], &inputs[0], 24), before[0]);
            assert_eq!(read(&mock, &primaries[1], &inputs[1], 24), before[1]);
        }
    }

    #[test]
    fn cuda_two_device_all_reduce_action_failures_are_precise_and_retryable() {
        for (is_peer, action_id, operation) in [(true, 4, "peer-copy"), (false, 5, "add")] {
            let (mock, executor, inputs, plan, primaries) = fixture();
            let left: Vec<u8> = [i32::MAX, 2, 3]
                .into_iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect();
            let right: Vec<u8> = [1_i32, 4, 5]
                .into_iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect();
            let expected = native_sum(DType::I32, &left, &right);
            write(&mock, &primaries[0], &inputs[0], &left);
            write(&mock, &primaries[1], &inputs[1], &right);
            if is_peer {
                mock.fail_peer_after(1, 2);
            } else {
                mock.fail_launch_after(1, 2);
            }
            assert!(
                matches!(executor.all_reduce_sum(&plan, [&inputs[0], &inputs[1]]), Err(Error::CollectiveAction { action_id: actual_id, operation: actual_op, .. }) if actual_id == action_id && actual_op == operation)
            );
            assert_eq!(read(&mock, &primaries[0], &inputs[0], 12), expected);
            assert_eq!(read(&mock, &primaries[1], &inputs[1], 12), right);
            assert_eq!(primaries[0].allocator().deferred_bytes(), 0);
            assert_eq!(primaries[1].allocator().deferred_bytes(), 0);
            write(&mock, &primaries[0], &inputs[0], &left);
            write(&mock, &primaries[1], &inputs[1], &right);
            assert!(
                executor
                    .all_reduce_sum(&plan, [&inputs[0], &inputs[1]])
                    .is_ok()
            );
            assert_eq!(read(&mock, &primaries[0], &inputs[0], 12), expected);
            assert_eq!(read(&mock, &primaries[1], &inputs[1], 12), expected);
        }
    }
}
