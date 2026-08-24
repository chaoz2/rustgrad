//! Deterministic, backend-neutral planning for the tinygrad multi-device
//! reduction boundary. CUDA/NCCL execution deliberately lives elsewhere.

use crate::{DType, Error, Result, Scalar, TensorData};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

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
}
