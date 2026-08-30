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

    /// Rebuilds the deterministic plan from its request before an executor
    /// allocates scratch storage or observes an action. Public fields are
    /// inspectable metadata, not an alternate executable authoring surface.
    pub fn validate(&self) -> Result<()> {
        let canonical = CollectivePlanner::plan(self.request.clone())?;
        if self.output_lengths != canonical.output_lengths
            || self.chunks != canonical.chunks
            || self.actions != canonical.actions
            || self.cache_key != canonical.cache_key
        {
            return Err(err("collective plan does not match its canonical request"));
        }
        Ok(())
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

/// Sequential CUDA realization of a small all-reduce plan.  It is kept
/// deliberately separate from the dense oracle: every nonempty plan action is
/// a Driver copy or a typed primary PTX add launch.
pub struct CudaCollectiveGroup {
    devices: Vec<DeviceId>,
    contexts: Vec<PrimaryContext>,
    allocators: Vec<std::sync::Arc<PrimaryCudaAllocator>>,
    streams: Vec<Stream>,
    peers: std::sync::Mutex<std::collections::BTreeMap<(usize, usize), PeerAccess>>,
    adds: Vec<PrimaryCollectiveAddCache>,
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
    pub fn new(bindings: impl IntoIterator<Item = (DeviceId, PrimaryContext)>) -> Result<Self> {
        let bindings: Vec<_> = bindings.into_iter().collect();
        if !(1..=4).contains(&bindings.len()) {
            return Err(err(
                "CUDA collective group supports one through four owners",
            ));
        }
        let devices: Vec<_> = bindings.iter().map(|(device, _)| device.clone()).collect();
        if devices.iter().collect::<BTreeSet<_>>().len() != devices.len()
            || bindings
                .iter()
                .map(|(_, context)| context.identity())
                .collect::<BTreeSet<_>>()
                .len()
                != bindings.len()
        {
            return Err(err(
                "CUDA collective group requires distinct identities and primary owners",
            ));
        }
        let contexts: Vec<_> = bindings
            .iter()
            .map(|(_, context)| context.clone())
            .collect();
        let allocators = contexts.iter().map(PrimaryContext::allocator).collect();
        let streams = Self::create_streams(&contexts)?;
        Ok(Self {
            devices,
            contexts,
            allocators,
            streams,
            peers: Default::default(),
            adds: bindings
                .iter()
                .map(|_| PrimaryCollectiveAddCache::new())
                .collect(),
        })
    }
    fn create_streams(contexts: &[PrimaryContext]) -> Result<Vec<Stream>> {
        let mut streams = Vec::with_capacity(contexts.len());
        for context in contexts {
            match context.stream() {
                Ok(stream) => streams.push(stream),
                Err(error) => {
                    // `Stream::close` preserves retryability when destruction
                    // fails. Explicitly close earlier streams before unwinding
                    // so Drop gets one final retry instead of silently losing
                    // a partially created group resource.
                    for stream in &streams {
                        let _ = stream.close();
                    }
                    return Err(cuda_err(error));
                }
            }
        }
        Ok(streams)
    }
    fn ensure_peers(&self, plan: &CollectivePlan) -> Result<()> {
        let mut peers = self.peers.lock().expect("collective peer mutex poisoned");
        // Publish newly enabled directional pairs only after the entire plan
        // has acquired them. On a later enable failure, this temporary map
        // drops its owned peers and leaves the retry cache unchanged.
        let mut acquired = std::collections::BTreeMap::new();
        for action in &plan.actions {
            if action.op != ActionOp::Transfer || action.range.len == 0 {
                continue;
            }
            let source = self
                .devices
                .iter()
                .position(|d| d == &action.source)
                .unwrap();
            let destination = self
                .devices
                .iter()
                .position(|d| d == &action.destination)
                .unwrap();
            let key = (source, destination);
            if !peers.contains_key(&key) && !acquired.contains_key(&key) {
                let peer = self.contexts[source]
                    .peer_access_to(&self.contexts[destination])
                    .map_err(|error| Error::CollectiveAction {
                        action_id: action.id,
                        operation: "peer-copy",
                        reason: error.to_string(),
                    })?;
                acquired.insert(key, peer);
            }
        }
        peers.append(&mut acquired);
        Ok(())
    }
    fn synchronize_before_rollback(&self) -> Result<()> {
        // A failed synchronous launch may still have submitted work. Reclaim
        // every rank stream before restoring caller-owned outputs, so the
        // snapshot copy cannot race an earlier collective action.
        for stream in &self.streams {
            stream.synchronize().map_err(cuda_err)?;
        }
        Ok(())
    }
    /// Mutates each input lease in place and returns an inspectable trace.
    pub fn all_reduce_sum<'a, I: AsRef<[&'a PrimaryBufferLease]>>(
        &self,
        plan: &CollectivePlan,
        inputs: I,
    ) -> Result<Vec<CudaCollectiveTrace>> {
        plan.validate()?;
        let inputs = inputs.as_ref();
        if !matches!(
            plan.request.kind,
            CollectiveKind::AllReduce {
                reduction: Reduction::Sum
            }
        ) || plan.request.group.devices() != self.devices
            || plan.request.input_lengths.len() != self.devices.len()
            || inputs.len() != self.devices.len()
        {
            return Err(err("CUDA sum all-reduce group or input ownership mismatch"));
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
        if plan
            .request
            .input_lengths
            .iter()
            .any(|&length| length != count)
        {
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
        if inputs.iter().zip(&self.contexts).any(|(lease, context)| {
            lease
                .primary()
                .map_or(true, |owner| owner.identity() != context.identity())
        }) {
            return Err(err("collective input lease owner mismatch"));
        }
        if count == 0 || self.devices.len() == 1 {
            return Ok(Vec::new());
        }
        self.ensure_peers(plan)?;
        let scratch = self
            .allocators
            .iter()
            .map(|allocator| {
                allocator
                    .allocate(NonZeroUsize::new(bytes).unwrap())
                    .map_err(cuda_err)
            })
            .collect::<Result<Vec<_>>>()?;
        // In-place outputs must not become later transfer sources: preserve
        // each rank's original contribution before the first reduction.
        let originals = self
            .allocators
            .iter()
            .map(|allocator| {
                allocator
                    .allocate(NonZeroUsize::new(bytes).unwrap())
                    .map_err(cuda_err)
            })
            .collect::<Result<Vec<_>>>()?;
        for index in 0..self.devices.len() {
            let original = originals[index].view().map_err(cuda_err)?;
            let input = inputs[index].view().map_err(cuda_err)?;
            original
                .copy_from_view(0, &input, 0, bytes)
                .map_err(cuda_err)?;
        }
        let action_result = (|| -> Result<Vec<CudaCollectiveTrace>> {
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
                            let peers = self.peers.lock().expect("collective peer mutex poisoned");
                            let peer = peers
                                .get(&(src, dst))
                                .expect("required collective peer exists");
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
                            drop(t);
                            drop(peers);
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
        })();
        match action_result {
            Ok(trace) => Ok(trace),
            Err(action_error) => {
                // Every action writes only a caller-owned rank output. Restore
                // all of them from the pre-action snapshots before exposing the
                // failed collective, so a retry sees the original contributions.
                self.synchronize_before_rollback()?;
                for index in 0..self.devices.len() {
                    let destination = inputs[index].view().map_err(cuda_err)?;
                    let original = originals[index].view().map_err(cuda_err)?;
                    destination
                        .copy_from_view(0, &original, 0, bytes)
                        .map_err(cuda_err)?;
                }
                Err(action_error)
            }
        }
    }
}
fn cuda_err(error: CudaError) -> Error {
    err(error.to_string())
}
#[derive(Clone, Copy, Debug, Default)]
pub struct InMemoryCollectiveExecutor;
impl CollectiveExecutor for InMemoryCollectiveExecutor {
    fn execute(&self, plan: &CollectivePlan, inputs: &[TensorData]) -> Result<Vec<TensorData>> {
        plan.validate()?;
        let r = &plan.request;
        plan.validate()?;
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

    #[test]
    fn public_plan_tampering_rejects_before_in_memory_execution() {
        let mut plan = CollectivePlanner::plan(CollectiveRequest {
            group: group(2),
            kind: CollectiveKind::AllReduce {
                reduction: Reduction::Sum,
            },
            dtype: DType::I32,
            input_lengths: vec![2, 2],
        })
        .unwrap();
        assert!(plan.validate().is_ok());
        let inputs = input(2, 2, DType::I32);
        let before = inputs
            .iter()
            .map(|input| input.storage().clone())
            .collect::<Vec<_>>();
        plan.actions[0].range.len = usize::MAX;

        assert!(plan.validate().is_err());
        assert!(InMemoryCollectiveExecutor.execute(&plan, &inputs).is_err());
        assert_eq!(
            inputs
                .iter()
                .map(|input| input.storage().clone())
                .collect::<Vec<_>>(),
            before
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
    fn fixture_n(
        n: usize,
        dtype: DType,
        count: usize,
    ) -> (
        std::sync::Arc<crate::cuda::tests::Mock>,
        CudaCollectiveGroup,
        Vec<PrimaryBufferLease>,
        CollectivePlan,
        Vec<PrimaryContext>,
    ) {
        use crate::Driver;
        use crate::cuda::tests::Mock;
        use std::{num::NonZeroUsize, sync::Arc};
        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let primaries: Vec<_> = (0..n)
            .map(|_| {
                driver
                    .device(crate::DeviceId(0))
                    .unwrap()
                    .retain_primary_context()
                    .unwrap()
            })
            .collect();
        let group = group(n);
        let plan = CollectivePlanner::plan(CollectiveRequest {
            group: group.clone(),
            kind: CollectiveKind::AllReduce {
                reduction: Reduction::Sum,
            },
            dtype,
            input_lengths: vec![count; n],
        })
        .unwrap();
        let bytes = count.max(1) * dtype.itemsize();
        let inputs = primaries
            .iter()
            .map(|primary| {
                primary
                    .allocator()
                    .allocate(NonZeroUsize::new(bytes).unwrap())
                    .unwrap()
            })
            .collect();
        let executor = CudaCollectiveGroup::new(
            group
                .devices()
                .iter()
                .cloned()
                .zip(primaries.iter().cloned()),
        )
        .unwrap();
        (mock, executor, inputs, plan, primaries)
    }

    #[test]
    fn cuda_collective_stream_setup_failure_cleans_and_retries() {
        use crate::Driver;
        use crate::cuda::tests::Mock;
        use std::{num::NonZeroUsize, sync::Arc};

        let mock = Arc::new(Mock::default());
        let driver = Driver::from_dispatch(mock.clone()).unwrap();
        let primaries: Vec<_> = (0..3)
            .map(|_| {
                driver
                    .device(crate::DeviceId(0))
                    .unwrap()
                    .retain_primary_context()
                    .unwrap()
            })
            .collect();
        let group = group(3);
        let plan = CollectivePlanner::plan(CollectiveRequest {
            group: group.clone(),
            kind: CollectiveKind::AllReduce {
                reduction: Reduction::Sum,
            },
            dtype: DType::I32,
            input_lengths: vec![2; 3],
        })
        .unwrap();
        let inputs: Vec<_> = primaries
            .iter()
            .map(|primary| {
                primary
                    .allocator()
                    .allocate(NonZeroUsize::new(8).unwrap())
                    .unwrap()
            })
            .collect();
        let values: Vec<Vec<u8>> = (0_i32..3)
            .map(|rank| {
                [rank + 1, rank + 10]
                    .into_iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect()
            })
            .collect();
        for (rank, bytes) in values.iter().enumerate() {
            write(&mock, &primaries[rank], &inputs[rank], bytes);
        }
        let allocations: Vec<_> = primaries
            .iter()
            .map(|primary| mock.live_allocation_count(primary.owner()))
            .collect();

        // The second stream cannot be created. The first stream's explicit
        // close also fails once, then its Drop retry releases it.
        mock.fail_stream_create_after(1, 2);
        mock.fail_stream_destroy_after(0, 2);
        assert!(
            CudaCollectiveGroup::new(
                group
                    .devices()
                    .iter()
                    .cloned()
                    .zip(primaries.iter().cloned()),
            )
            .is_err()
        );
        for (rank, input) in inputs.iter().enumerate() {
            assert_eq!(read(&mock, &primaries[rank], input, 8), values[rank]);
            assert_eq!(
                mock.live_allocation_count(primaries[rank].owner()),
                allocations[rank]
            );
        }
        let calls = mock.calls();
        assert_eq!(
            calls
                .iter()
                .filter(|&&call| call == "stream_create")
                .count(),
            2
        );
        assert_eq!(
            calls
                .iter()
                .filter(|&&call| call == "stream_destroy")
                .count(),
            2
        );
        assert!(
            calls
                .iter()
                .all(|&call| call != "peer_enable" && call != "launch")
        );

        let executor = CudaCollectiveGroup::new(
            group
                .devices()
                .iter()
                .cloned()
                .zip(primaries.iter().cloned()),
        )
        .unwrap();
        let expected = values[1..].iter().fold(values[0].clone(), |sum, next| {
            native_sum(DType::I32, &sum, next)
        });
        let refs: Vec<_> = inputs.iter().collect();
        executor.all_reduce_sum(&plan, refs).unwrap();
        for (rank, input) in inputs.iter().enumerate() {
            assert_eq!(read(&mock, &primaries[rank], input, 8), expected);
        }
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
            let (mock, executor, inputs, plan, primaries) = fixture_n(2, dtype, 3);
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
        let (mock, executor, inputs, plan, primaries) = fixture_n(2, DType::I32, 0);
        assert!(
            executor
                .all_reduce_sum(&plan, [&inputs[0], &inputs[1]])
                .unwrap()
                .is_empty()
        );
        assert!(mock.calls().iter().all(|call| *call != "launch"));
        drop((executor, inputs, primaries));
    }

    #[test]
    fn cuda_two_device_all_reduce_rejects_unsupported_dtypes_before_mutation() {
        for dtype in [DType::Bool, DType::I16, DType::U16, DType::F16, DType::BF16] {
            let (mock, executor, inputs, plan, primaries) = fixture_n(2, dtype, 3);
            let bytes = 3 * dtype.itemsize();
            let before = [vec![0xa5; bytes], vec![0x5a; bytes]];
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
            assert_eq!(read(&mock, &primaries[0], &inputs[0], bytes), before[0]);
            assert_eq!(read(&mock, &primaries[1], &inputs[1], bytes), before[1]);
        }
    }

    #[test]
    fn collective_artifact_preflight_rejects_tampering_before_driver_work() {
        let (mock, executor, inputs, plan, primaries) = fixture();
        let left: Vec<u8> = [1_i32, 2, 3]
            .into_iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let right: Vec<u8> = [4_i32, 5, 6]
            .into_iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        write(&mock, &primaries[0], &inputs[0], &left);
        write(&mock, &primaries[1], &inputs[1], &right);
        let calls = mock.calls().len();
        let allocations = [
            mock.live_allocation_count(primaries[0].owner()),
            mock.live_allocation_count(primaries[1].owner()),
        ];
        let mut cases = Vec::new();
        let mut bad_range = plan.clone();
        bad_range.actions[0].range.len += 1;
        cases.push(bad_range);
        let mut bad_dependency = plan.clone();
        bad_dependency.actions[1].depends_on.push(1);
        cases.push(bad_dependency);
        let mut bad_chunk = plan.clone();
        bad_chunk.chunks[0].start += 1;
        cases.push(bad_chunk);
        let mut bad_key = plan.clone();
        bad_key.cache_key.push('!');
        cases.push(bad_key);
        for invalid in cases {
            assert!(invalid.validate().is_err());
            assert!(
                executor
                    .all_reduce_sum(&invalid, [&inputs[0], &inputs[1]])
                    .is_err()
            );
            assert_eq!(mock.calls().len(), calls);
            assert_eq!(
                [
                    mock.live_allocation_count(primaries[0].owner()),
                    mock.live_allocation_count(primaries[1].owner()),
                ],
                allocations
            );
            assert_eq!(read(&mock, &primaries[0], &inputs[0], 12), left);
            assert_eq!(read(&mock, &primaries[1], &inputs[1], 12), right);
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
            let allocations = [
                mock.live_allocation_count(primaries[0].owner()),
                mock.live_allocation_count(primaries[1].owner()),
            ];
            if is_peer {
                mock.fail_peer_after(1, 2);
            } else {
                mock.fail_launch_after(1, 2);
            }
            assert!(
                matches!(executor.all_reduce_sum(&plan, [&inputs[0], &inputs[1]]), Err(Error::CollectiveAction { action_id: actual_id, operation: actual_op, .. }) if actual_id == action_id && actual_op == operation)
            );
            assert_eq!(read(&mock, &primaries[0], &inputs[0], 12), left);
            assert_eq!(read(&mock, &primaries[1], &inputs[1], 12), right);
            assert_eq!(
                mock.live_allocation_count(primaries[0].owner()),
                allocations[0]
            );
            assert_eq!(
                mock.live_allocation_count(primaries[1].owner()),
                allocations[1]
            );
            assert_eq!(primaries[0].allocator().deferred_bytes(), 0);
            assert_eq!(primaries[1].allocator().deferred_bytes(), 0);
            assert!(
                executor
                    .all_reduce_sum(&plan, [&inputs[0], &inputs[1]])
                    .is_ok()
            );
            assert_eq!(read(&mock, &primaries[0], &inputs[0], 12), expected);
            assert_eq!(read(&mock, &primaries[1], &inputs[1], 12), expected);
        }
    }

    #[test]
    fn cuda_collective_peer_setup_is_transactional_and_retryable() {
        let (mock, executor, inputs, plan, primaries) = fixture_n(3, DType::I32, 2);
        let allocations: Vec<_> = primaries
            .iter()
            .map(|primary| mock.live_allocation_count(primary.owner()))
            .collect();
        // The first directional pair is acquired, then the next enable fails.
        // The first one must be released rather than retained in the cache.
        mock.fail_peer_enable_after(1, 2);
        let refs: Vec<_> = inputs.iter().collect();
        assert!(matches!(
            executor.all_reduce_sum(&plan, refs),
            Err(Error::CollectiveAction {
                action_id: 3,
                operation: "peer-copy",
                ..
            })
        ));
        assert_eq!(
            primaries
                .iter()
                .map(|primary| mock.live_allocation_count(primary.owner()))
                .collect::<Vec<_>>(),
            allocations
        );
        let calls = mock.calls();
        assert_eq!(
            calls.iter().filter(|&&call| call == "peer_enable").count(),
            2
        );
        assert_eq!(
            calls.iter().filter(|&&call| call == "peer_disable").count(),
            1
        );

        let values: Vec<Vec<u8>> = (0_i32..3)
            .map(|rank| {
                [rank + 1, rank + 10]
                    .into_iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect()
            })
            .collect();
        let expected = values[1..].iter().fold(values[0].clone(), |sum, next| {
            native_sum(DType::I32, &sum, next)
        });
        for (rank, bytes) in values.iter().enumerate() {
            write(&mock, &primaries[rank], &inputs[rank], bytes);
        }
        let refs: Vec<_> = inputs.iter().collect();
        executor.all_reduce_sum(&plan, refs).unwrap();
        for (rank, input) in inputs.iter().enumerate() {
            assert_eq!(
                read(&mock, &primaries[rank], input, expected.len()),
                expected
            );
        }
        let calls = mock.calls();
        assert_eq!(
            calls.iter().filter(|&&call| call == "peer_enable").count(),
            8
        );
        assert_eq!(
            calls.iter().filter(|&&call| call == "peer_disable").count(),
            1
        );
        drop((executor, inputs, primaries));
        assert_eq!(
            mock.calls()
                .iter()
                .filter(|&&call| call == "peer_disable")
                .count(),
            7
        );
    }

    #[test]
    fn cuda_collective_later_rank_sync_failure_restores_and_retries() {
        let (mock, executor, inputs, plan, primaries) = fixture_n(3, DType::I32, 2);
        let values: Vec<Vec<u8>> = (0_i32..3)
            .map(|rank| {
                [rank + 1, rank + 10]
                    .into_iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect()
            })
            .collect();
        let expected = values[1..].iter().fold(values[0].clone(), |sum, next| {
            native_sum(DType::I32, &sum, next)
        });
        for (rank, bytes) in values.iter().enumerate() {
            write(&mock, &primaries[rank], &inputs[rank], bytes);
        }
        let allocations: Vec<_> = primaries
            .iter()
            .map(|primary| mock.live_allocation_count(primary.owner()))
            .collect();
        let stream_creates = mock
            .calls()
            .iter()
            .filter(|&&call| call == "stream_create")
            .count();
        // Rank zero completes two reductions; the third reduction belongs to
        // rank one and has already submitted its add when synchronization
        // reports the one-shot failure.
        mock.fail_stream_sync_after(2, 2);
        let refs: Vec<_> = inputs.iter().collect();
        assert!(matches!(
            executor.all_reduce_sum(&plan, refs),
            Err(Error::CollectiveAction {
                action_id: 7,
                operation: "add",
                ..
            })
        ));
        for (rank, input) in inputs.iter().enumerate() {
            assert_eq!(
                read(&mock, &primaries[rank], input, values[rank].len()),
                values[rank]
            );
            assert_eq!(
                mock.live_allocation_count(primaries[rank].owner()),
                allocations[rank]
            );
            assert_eq!(primaries[rank].allocator().deferred_bytes(), 0);
        }
        assert_eq!(
            mock.calls()
                .iter()
                .filter(|&&call| call == "stream_create")
                .count(),
            stream_creates
        );
        let refs: Vec<_> = inputs.iter().collect();
        executor.all_reduce_sum(&plan, refs).unwrap();
        for (rank, input) in inputs.iter().enumerate() {
            assert_eq!(
                read(&mock, &primaries[rank], input, expected.len()),
                expected
            );
        }
    }

    #[test]
    fn cuda_many_device_all_reduce_matches_dense_reference_and_reuses_edges() {
        for (n, count, dtype) in [
            (1, 0, DType::I32),
            (1, 3, DType::I32),
            (3, 1, DType::I32),
            (3, 2, DType::F32),
            (3, 5, DType::I64),
            (4, 3, DType::F64),
            (4, 7, DType::I32),
        ] {
            let (mock, executor, inputs, plan, primaries) = fixture_n(n, dtype, count);
            let mut values = Vec::new();
            for rank in 0..n {
                let bytes: Vec<u8> = (0..count)
                    .flat_map(|index| match dtype {
                        DType::I32 => (if rank == 0 && index == 0 {
                            i32::MAX
                        } else {
                            (rank * 10 + index) as i32
                        })
                        .to_ne_bytes()
                        .to_vec(),
                        DType::I64 => (if rank == 0 && index == 0 {
                            i64::MAX
                        } else {
                            (rank * 10 + index) as i64
                        })
                        .to_ne_bytes()
                        .to_vec(),
                        DType::F32 => (rank as f32 + index as f32 + 0.25).to_ne_bytes().to_vec(),
                        DType::F64 => (rank as f64 + index as f64 + 0.25).to_ne_bytes().to_vec(),
                        _ => unreachable!(),
                    })
                    .collect();
                write(&mock, &primaries[rank], &inputs[rank], &bytes);
                values.push(bytes);
            }
            let expected = values[1..]
                .iter()
                .fold(values[0].clone(), |sum, next| native_sum(dtype, &sum, next));
            let refs: Vec<_> = inputs.iter().collect();
            let trace = executor.all_reduce_sum(&plan, refs).unwrap();
            if n == 1 || count == 0 {
                assert!(trace.is_empty());
            } else {
                assert_eq!(trace.len(), plan.actions.len());
                assert!(trace.iter().zip(&plan.actions).all(|(seen, action)| {
                    seen.action_id == action.id && seen.range == action.range
                }));
                for (rank, input) in inputs.iter().enumerate() {
                    assert_eq!(
                        read(&mock, &primaries[rank], input, expected.len()),
                        expected
                    );
                }
                let peer_before = mock
                    .calls()
                    .iter()
                    .filter(|&&call| call == "peer_enable")
                    .count();
                let refs: Vec<_> = inputs.iter().collect();
                executor.all_reduce_sum(&plan, refs).unwrap();
                assert_eq!(
                    mock.calls()
                        .iter()
                        .filter(|&&call| call == "peer_enable")
                        .count(),
                    peer_before
                );
            }
        }
    }

    #[test]
    fn cuda_three_device_required_edge_failures_are_structured_and_retryable() {
        let (mock, executor, inputs, plan, primaries) = fixture_n(3, DType::I32, 2);
        let values: Vec<Vec<u8>> = (0_i32..3)
            .map(|rank| {
                [rank + 1, rank + 10]
                    .into_iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect()
            })
            .collect();
        for (rank, bytes) in values.iter().enumerate() {
            write(&mock, &primaries[rank], &inputs[rank], bytes);
        }
        let expected = values[1..].iter().fold(values[0].clone(), |sum, next| {
            native_sum(DType::I32, &sum, next)
        });
        mock.set_peer_capable(false);
        let refs: Vec<_> = inputs.iter().collect();
        assert!(matches!(
            executor.all_reduce_sum(&plan, refs),
            Err(Error::CollectiveAction {
                action_id: 1,
                operation: "peer-copy",
                ..
            })
        ));
        assert_eq!(mock.live_allocation_count(primaries[0].owner()), 1);
        mock.set_peer_capable(true);
        mock.fail_peer_after(2, 2);
        let refs: Vec<_> = inputs.iter().collect();
        assert!(matches!(
            executor.all_reduce_sum(&plan, refs),
            Err(Error::CollectiveAction {
                action_id: 6,
                operation: "peer-copy",
                ..
            })
        ));
        for (rank, input) in inputs.iter().enumerate() {
            assert_eq!(read(&mock, &primaries[rank], input, 8), values[rank]);
        }
        assert_eq!(primaries[0].allocator().deferred_bytes(), 0);
        mock.fail_launch_after(2, 2);
        for (rank, bytes) in values.iter().enumerate() {
            write(&mock, &primaries[rank], &inputs[rank], bytes);
        }
        let refs: Vec<_> = inputs.iter().collect();
        assert!(matches!(
            executor.all_reduce_sum(&plan, refs),
            Err(Error::CollectiveAction {
                action_id: 7,
                operation: "add",
                ..
            })
        ));
        for (rank, input) in inputs.iter().enumerate() {
            assert_eq!(read(&mock, &primaries[rank], input, 8), values[rank]);
        }
        let refs: Vec<_> = inputs.iter().collect();
        executor.all_reduce_sum(&plan, refs).unwrap();
        for (rank, input) in inputs.iter().enumerate() {
            assert_eq!(
                read(&mock, &primaries[rank], input, expected.len()),
                expected
            );
        }
        drop((executor, inputs, primaries));
        let calls = mock.calls();
        assert!(
            calls
                .iter()
                .rposition(|call| *call == "peer_disable")
                .unwrap()
                < calls
                    .iter()
                    .position(|call| *call == "primary_release")
                    .unwrap()
        );
    }
}
