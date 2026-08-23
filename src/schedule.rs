//! A deterministic, non-mutating scheduling view.  The first phase recognizes
//! pure elementwise graph regions and describes their buffers/UOp boundary;
//! execution remains intentionally outside this planning layer.
use crate::{DType, Graph, NodeId, Op, Shape, UOp, UOpError};
use std::{
    collections::{BTreeSet, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BufferDesc {
    pub id: u64,
    pub shape: Shape,
    pub dtype: DType,
    pub bytes: usize,
    pub alignment: usize,
    pub read_only: bool,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ScheduleBoundary {
    Unsupported(&'static str),
    NonScalarUOpBridge,
}
#[derive(Clone, Debug)]
pub struct ScheduleItem {
    pub inputs: Vec<BufferDesc>,
    pub output: BufferDesc,
    pub kernel: UOp,
    pub boundary: Option<ScheduleBoundary>,
    pub cache_key: u64,
}
#[derive(Clone, Debug)]
pub struct Schedule {
    pub items: Vec<ScheduleItem>,
}
/// Deterministic, conservative allocation assignment for compiler-created
/// temporaries. Callers supply only internal buffers; graph inputs, constants,
/// requested outputs and aliases are therefore never candidates for reuse.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporaryAllocation {
    pub buffer_id: u64,
    pub allocation_id: u64,
    pub first_item: usize,
    pub last_item: usize,
    pub bytes: usize,
    pub alignment: usize,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MemoryPlan {
    pub temporaries: Vec<TemporaryAllocation>,
}
/// Assigns allocations in buffer-ID order, reusing only an earlier compatible
/// allocation whose last use precedes the candidate's first use.
pub fn plan_temporary_reuse(
    items: &[ScheduleItem],
    temporaries: &[BufferDesc],
) -> Result<MemoryPlan, ScheduleError> {
    let mut lifetimes = Vec::with_capacity(temporaries.len());
    for buffer in temporaries {
        let mut uses = items.iter().enumerate().filter_map(|(item, scheduled)| {
            (scheduled.output.id == buffer.id
                || scheduled.inputs.iter().any(|input| input.id == buffer.id))
            .then_some(item)
        });
        let first = uses.next().ok_or(ScheduleError::Overflow)?;
        let last = uses.next_back().unwrap_or(first);
        lifetimes.push((buffer.clone(), first, last));
    }
    lifetimes.sort_by_key(|(buffer, first, _)| (*first, buffer.id));
    let mut slots: Vec<(u64, usize, usize, usize)> = vec![]; // id, bytes, alignment, last use
    let mut result = Vec::with_capacity(lifetimes.len());
    for (buffer, first, last) in lifetimes {
        let compatible = slots
            .iter()
            .enumerate()
            .filter(|(_, (_, bytes, alignment, available))| {
                *available < first && *bytes >= buffer.bytes && *alignment >= buffer.alignment
            })
            .map(|(index, _)| index)
            .min();
        let allocation_id = if buffer.bytes == 0 {
            let id = slots.len() as u64;
            slots.push((id, 0, buffer.alignment, last));
            id
        } else if let Some(slot) = compatible {
            slots[slot].3 = last;
            slots[slot].0
        } else {
            let id = slots.len() as u64;
            slots.push((id, buffer.bytes, buffer.alignment, last));
            id
        };
        result.push(TemporaryAllocation {
            buffer_id: buffer.id,
            allocation_id,
            first_item: first,
            last_item: last,
            bytes: buffer.bytes,
            alignment: buffer.alignment,
        });
    }
    result.sort_by_key(|entry| entry.buffer_id);
    Ok(MemoryPlan {
        temporaries: result,
    })
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScheduleError {
    Graph(crate::Error),
    Overflow,
    UOp(UOpError),
}
impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "schedule error: {self:?}")
    }
}
impl std::error::Error for ScheduleError {}
fn buffer(graph: &Graph, id: NodeId, read_only: bool) -> Result<BufferDesc, ScheduleError> {
    let shape = graph.shape(id).map_err(ScheduleError::Graph)?.clone();
    let dtype = graph.dtype(id).map_err(ScheduleError::Graph)?;
    let bytes = shape
        .numel()
        .map_err(ScheduleError::Graph)?
        .checked_mul(dtype.itemsize())
        .ok_or(ScheduleError::Overflow)?;
    Ok(BufferDesc {
        id: id.index() as u64,
        shape,
        dtype,
        bytes,
        alignment: dtype.itemsize().max(1),
        read_only,
    })
}
fn supported(op: &Op) -> bool {
    matches!(
        op,
        Op::Input { .. }
            | Op::Constant(_)
            | Op::Cast { .. }
            | Op::Unary { .. }
            | Op::Binary { .. }
            | Op::Compare { .. }
            | Op::Logical { .. }
            | Op::Select { .. }
            | Op::Reduce {
                kind: crate::ReduceKind::Sum | crate::ReduceKind::Mean,
                ..
            }
    )
}
/// Creates one conservative fused item for a pure elementwise output. Anything
/// else is a visible schedule boundary, never an implicit mislowering.
pub fn schedule(graph: &Graph, output: NodeId) -> Result<Schedule, ScheduleError> {
    let mut leaves = BTreeSet::new();
    let mut boundary = None;
    fn walk(
        g: &Graph,
        id: NodeId,
        leaves: &mut BTreeSet<usize>,
        boundary: &mut Option<ScheduleBoundary>,
    ) -> Result<(), ScheduleError> {
        let op = g.op(id).map_err(ScheduleError::Graph)?;
        if !supported(op) {
            *boundary = Some(ScheduleBoundary::Unsupported(match op {
                Op::Reduce {
                    kind: crate::ReduceKind::Product,
                    ..
                } => "product reductions are outside sum/mean lowering",
                Op::Reduce {
                    kind: crate::ReduceKind::Min | crate::ReduceKind::Max,
                    ..
                } => "min/max reductions are outside sum/mean lowering",
                _ => "operation is outside phase-one elementwise lowering",
            }));
            leaves.insert(id.index());
            return Ok(());
        }
        match op {
            Op::Input { .. } | Op::Constant(_) => {
                leaves.insert(id.index());
            }
            Op::Cast { input, .. } | Op::Unary { input, .. } => walk(g, *input, leaves, boundary)?,
            Op::Binary { lhs, rhs, .. } | Op::Compare { lhs, rhs, .. } => {
                walk(g, *lhs, leaves, boundary)?;
                walk(g, *rhs, leaves, boundary)?
            }
            Op::Logical { lhs, rhs, .. } => {
                walk(g, *lhs, leaves, boundary)?;
                if let Some(rhs) = rhs {
                    walk(g, *rhs, leaves, boundary)?
                }
            }
            Op::Select {
                condition,
                on_true,
                on_false,
            } => {
                walk(g, *condition, leaves, boundary)?;
                walk(g, *on_true, leaves, boundary)?;
                walk(g, *on_false, leaves, boundary)?
            }
            Op::Reduce { input, .. } => walk(g, *input, leaves, boundary)?,
            _ => unreachable!(),
        }
        Ok(())
    }
    walk(graph, output, &mut leaves, &mut boundary)?;
    let inputs = leaves
        .into_iter()
        .map(|i| buffer(graph, NodeId::from_index(i), true))
        .collect::<Result<Vec<_>, _>>()?;
    let out = buffer(graph, output, false)?;
    let kernel =
        if boundary.is_none() {
            match graph.op(output).map_err(ScheduleError::Graph)? {
                Op::Reduce { .. } => crate::kernel::lower_graph_reduction(graph, output)
                    .map_err(ScheduleError::UOp)?,
                _ => crate::kernel::lower_graph_elementwise(graph, output)
                    .map_err(ScheduleError::UOp)?,
            }
        } else {
            UOp::sink(vec![])
        };
    let mut h = DefaultHasher::new();
    inputs.hash(&mut h);
    out.hash(&mut h);
    boundary.hash(&mut h);
    kernel.hash(&mut h);
    let cache_key = h.finish();
    Ok(Schedule {
        items: vec![ScheduleItem {
            inputs,
            output: out,
            kernel,
            boundary,
            cache_key,
        }],
    })
}
