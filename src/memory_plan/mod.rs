//! Alias-safe logical host allocation planning for schedule materializations.
mod dynamic;

pub(crate) use dynamic::RuntimeAllocationLifetime;

use crate::{BufferDesc, NodeId, Schedule, ScheduleItem, Shape};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

/// The only address space currently owned by lazy realization.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MemoryAddressSpace {
    HostDense,
}

/// A checked, non-external temporary lifetime derived from one schedule output.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct AllocationRequest {
    pub buffer_id: u64,
    pub elements: usize,
    pub bytes: usize,
    pub dtype: crate::DType,
    pub shape: Shape,
    pub alignment: usize,
    pub address_space: MemoryAddressSpace,
    pub producer_item: u64,
    pub last_consumer: u64,
}

/// One deterministic assignment. `None` denotes a private zero-byte sentinel;
/// sentinels are never put into an allocation reuse pool.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TemporaryAllocation {
    pub buffer_id: u64,
    pub allocation_id: Option<u64>,
    pub producer_item: u64,
    pub last_consumer: u64,
    pub bytes: usize,
    pub alignment: usize,
    pub reused_from: Option<u64>,
}

/// Immutable allocation decisions. Reuse is exact: capacity, dtype, shape,
/// alignment, and address space must all match. Larger-capacity reuse is
/// deliberately disallowed until a backend can prove its logical bounds ABI.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MemoryPlan {
    pub requests: Vec<AllocationRequest>,
    pub temporaries: Vec<TemporaryAllocation>,
    pub peak_allocations: usize,
    pub peak_bytes: usize,
}

/// Immutable liveness record for one versioned persistent base/view write.
/// Persistent effect leases intentionally remain owned by `EffectRuntime`; this
/// plan proves their logical aliases cannot be treated as reusable temporaries.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct AliasLifetime {
    pub base_buffer: u64,
    pub predecessor_version: u64,
    pub successor_version: u64,
    pub view: Option<crate::AffineView>,
    pub producer_step: u64,
    pub last_consumer_step: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AliasLivenessPlan {
    pub lifetimes: Vec<AliasLifetime>,
    pub persistent_bytes: usize,
}

impl AliasLivenessPlan {
    /// Derives canonical version lifetimes before mixed/effect realization.
    /// A physical base stays live through its final successor commit; aliases
    /// never receive a temporary allocation/reuse identity of their own.
    pub fn from_effects(plan: &crate::EffectPlan) -> Result<Self, MemoryPlanError> {
        plan.validate()
            .map_err(|_| MemoryPlanError::AliasEscape(u64::MAX))?;
        let mut lifetimes = Vec::with_capacity(plan.steps.len());
        let mut bytes = 0usize;
        for step in &plan.steps {
            bytes = bytes
                .checked_add(step.write.bytes)
                .ok_or(MemoryPlanError::Overflow)?;
            lifetimes.push(AliasLifetime {
                base_buffer: step.write.buffer,
                predecessor_version: step.reads[0].version,
                successor_version: step.write.version,
                view: step.target_view.clone(),
                producer_step: step.id,
                last_consumer_step: plan.steps.last().map_or(step.id, |last| last.id),
            });
        }
        Ok(Self {
            lifetimes,
            persistent_bytes: bytes,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryPlanError {
    InvalidSchedule(String),
    Overflow,
    DuplicateBuffer(u64),
    MissingProducer(u64),
    UseBeforeProduce {
        buffer: u64,
        producer: u64,
        consumer: u64,
    },
    InvalidConsumer {
        buffer: u64,
        producer: u64,
        consumer: u64,
    },
    ConsumerMismatch {
        producer: u64,
    },
    AliasEscape(u64),
}
impl fmt::Display for MemoryPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "memory plan error: {self:?}")
    }
}
impl std::error::Error for MemoryPlanError {}

impl MemoryPlan {
    pub fn from_schedule(
        schedule: &Schedule,
        requested: &[NodeId],
        reuse: bool,
    ) -> Result<Self, MemoryPlanError> {
        schedule
            .validate()
            .map_err(|error| MemoryPlanError::InvalidSchedule(error.to_string()))?;
        let temporaries = schedule.internal_temporaries(requested);
        Self::build(&schedule.items, &temporaries, reuse)
    }

    /// Compatibility entry point for callers that already own a conservative
    /// temporary list. It shares the exact same validation and allocation rules.
    pub fn from_temporaries(
        items: &[ScheduleItem],
        temporaries: &[BufferDesc],
        reuse: bool,
    ) -> Result<Self, MemoryPlanError> {
        Self::build(items, temporaries, reuse)
    }

    fn build(
        items: &[ScheduleItem],
        temporaries: &[BufferDesc],
        reuse: bool,
    ) -> Result<Self, MemoryPlanError> {
        let positions: BTreeMap<u64, usize> = items
            .iter()
            .enumerate()
            .map(|(position, item)| (item.id, position))
            .collect();
        if positions.len() != items.len() {
            return Err(MemoryPlanError::DuplicateBuffer(u64::MAX));
        }
        let mut producers = BTreeMap::new();
        for (position, item) in items.iter().enumerate() {
            if producers
                .insert(item.output.id, (item.id, position))
                .is_some()
            {
                return Err(MemoryPlanError::DuplicateBuffer(item.output.id));
            }
        }
        let temporary_ids = temporaries
            .iter()
            .map(|desc| desc.id)
            .collect::<BTreeSet<_>>();
        if temporary_ids.len() != temporaries.len() {
            return Err(MemoryPlanError::DuplicateBuffer(u64::MAX));
        }
        let mut requests = Vec::with_capacity(temporaries.len());
        for desc in temporaries {
            if desc.view.is_some() {
                return Err(MemoryPlanError::AliasEscape(desc.id));
            }
            let (producer_item, producer_position) = producers
                .get(&desc.id)
                .copied()
                .ok_or(MemoryPlanError::MissingProducer(desc.id))?;
            let users = items
                .iter()
                .enumerate()
                .filter(|(_, item)| item.inputs.iter().any(|input| input.id == desc.id))
                .map(|(position, item)| (position, item.id))
                .collect::<Vec<_>>();
            let declared = items[producer_position].consumers.clone();
            if declared != users.iter().map(|(_, id)| *id).collect::<Vec<_>>() {
                return Err(MemoryPlanError::ConsumerMismatch {
                    producer: producer_item,
                });
            }
            let last = users
                .last()
                .copied()
                .unwrap_or((producer_position, producer_item));
            if last.0 < producer_position {
                return Err(MemoryPlanError::UseBeforeProduce {
                    buffer: desc.id,
                    producer: producer_item,
                    consumer: last.1,
                });
            }
            let elements = desc.shape.numel().map_err(|_| MemoryPlanError::Overflow)?;
            let bytes = elements
                .checked_mul(desc.dtype.itemsize())
                .ok_or(MemoryPlanError::Overflow)?;
            if bytes != desc.bytes {
                return Err(MemoryPlanError::Overflow);
            }
            requests.push((
                producer_position,
                AllocationRequest {
                    buffer_id: desc.id,
                    elements,
                    bytes,
                    dtype: desc.dtype,
                    shape: desc.shape.clone(),
                    alignment: desc.alignment,
                    address_space: MemoryAddressSpace::HostDense,
                    producer_item,
                    last_consumer: last.1,
                },
            ));
        }
        for (position, item) in items.iter().enumerate() {
            for input in &item.inputs {
                if temporary_ids.contains(&input.id) {
                    let (_, producer_position) = producers
                        .get(&input.id)
                        .copied()
                        .ok_or(MemoryPlanError::MissingProducer(input.id))?;
                    if producer_position >= position {
                        return Err(MemoryPlanError::UseBeforeProduce {
                            buffer: input.id,
                            producer: items[producer_position].id,
                            consumer: item.id,
                        });
                    }
                }
            }
        }
        requests.sort_by_key(|(position, request)| (*position, request.buffer_id));
        let mut slots: Vec<(u64, AllocationRequest, usize)> = vec![];
        let mut temporaries_out = Vec::with_capacity(requests.len());
        for (producer_position, request) in &requests {
            if request.bytes == 0 {
                temporaries_out.push(TemporaryAllocation {
                    buffer_id: request.buffer_id,
                    allocation_id: None,
                    producer_item: request.producer_item,
                    last_consumer: request.last_consumer,
                    bytes: 0,
                    alignment: request.alignment,
                    reused_from: None,
                });
                continue;
            }
            let available = reuse
                .then(|| {
                    slots
                        .iter()
                        .enumerate()
                        .find_map(|(index, (_, prior, last_position))| {
                            (last_position < producer_position
                                && prior.bytes == request.bytes
                                && prior.dtype == request.dtype
                                && prior.shape == request.shape
                                && prior.alignment == request.alignment
                                && prior.address_space == request.address_space)
                                .then_some(index)
                        })
                })
                .flatten();
            let (allocation_id, reused_from) = if let Some(slot) = available {
                let (id, prior, last_position) = &mut slots[slot];
                let previous = prior.buffer_id;
                *prior = request.clone();
                *last_position = request_last_position(items, request)?;
                (*id, Some(previous))
            } else {
                let id = slots.len() as u64;
                slots.push((id, request.clone(), request_last_position(items, request)?));
                (id, None)
            };
            temporaries_out.push(TemporaryAllocation {
                buffer_id: request.buffer_id,
                allocation_id: Some(allocation_id),
                producer_item: request.producer_item,
                last_consumer: request.last_consumer,
                bytes: request.bytes,
                alignment: request.alignment,
                reused_from,
            });
        }
        let peak = peak(&temporaries_out)?;
        temporaries_out.sort_by_key(|entry| entry.buffer_id);
        Ok(Self {
            requests: requests.into_iter().map(|(_, request)| request).collect(),
            temporaries: temporaries_out,
            peak_allocations: peak.0,
            peak_bytes: peak.1,
        })
    }
}

fn request_last_position(
    items: &[ScheduleItem],
    request: &AllocationRequest,
) -> Result<usize, MemoryPlanError> {
    items
        .iter()
        .position(|item| item.id == request.last_consumer)
        .ok_or(MemoryPlanError::InvalidConsumer {
            buffer: request.buffer_id,
            producer: request.producer_item,
            consumer: request.last_consumer,
        })
}

fn peak(allocations: &[TemporaryAllocation]) -> Result<(usize, usize), MemoryPlanError> {
    let mut slots = BTreeMap::new();
    for entry in allocations {
        if let Some(id) = entry.allocation_id {
            slots.entry(id).or_insert(entry.bytes);
        }
    }
    let bytes = slots
        .values()
        .try_fold(0usize, |sum, bytes| sum.checked_add(*bytes))
        .ok_or(MemoryPlanError::Overflow)?;
    Ok((slots.len(), bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Shape, TensorData};

    fn shared_schedule() -> (Graph, Schedule, crate::NodeId, crate::NodeId) {
        let mut graph = Graph::new();
        let x = graph.input("x", Shape::from([2]));
        let producer = graph.square(x).unwrap();
        let one = graph.constant(TensorData::scalar(1.0));
        let left = graph.add(producer, one).unwrap();
        let right = graph.mul(producer, one).unwrap();
        let schedule = crate::schedule_many(&graph, &[left, right]).unwrap();
        (graph, schedule, left, right)
    }

    #[test]
    fn rejects_alias_escape_and_malformed_consumers() {
        let (_, mut schedule, left, right) = shared_schedule();
        schedule.items[0].output.view = Some(crate::ViewMap::identity(Shape::from([2])).into());
        assert!(matches!(
            MemoryPlan::from_schedule(&schedule, &[left, right], true),
            Err(MemoryPlanError::AliasEscape(_))
        ));

        let (_, mut schedule, left, right) = shared_schedule();
        schedule.items[0].consumers.clear();
        assert!(matches!(
            MemoryPlan::from_schedule(&schedule, &[left, right], true),
            Err(MemoryPlanError::InvalidSchedule(_))
        ));
    }

    #[test]
    fn zero_byte_temporaries_receive_private_sentinels() {
        let mut graph = Graph::new();
        let x = graph.input("x", Shape::from([0, 2]));
        let shared = graph.neg(x).unwrap();
        let left = graph.neg(shared).unwrap();
        let right = graph.square(shared).unwrap();
        let schedule = crate::schedule_many(&graph, &[left, right]).unwrap();
        let plan = MemoryPlan::from_schedule(&schedule, &[left, right], true).unwrap();
        assert_eq!(plan.temporaries.len(), 1);
        assert_eq!(plan.temporaries[0].allocation_id, None);
        assert_eq!(plan.peak_bytes, 0);
    }
}
