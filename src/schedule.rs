//! A deterministic, non-mutating producer-aware schedule DAG. Pure
//! elementwise/view regions fuse into their consumers while materialization
//! roots retain stable buffer and UOp identities for realization.
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
    pub view: Option<crate::ViewMap>,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ScheduleBoundary {
    Unsupported(&'static str),
    NonScalarUOpBridge,
}
#[derive(Clone, Debug)]
pub struct ScheduleItem {
    pub id: u64,
    pub node: NodeId,
    pub dependencies: Vec<u64>,
    pub consumers: Vec<u64>,
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
impl Schedule {
    /// Returns only compiler-owned outputs that can become candidates for a
    /// future allocator. Requested outputs and external identities are kept
    /// out of this list, so a planner cannot accidentally reuse them.
    pub fn internal_temporaries(&self, requested: &[NodeId]) -> Vec<BufferDesc> {
        let requested = requested
            .iter()
            .map(|node| node.index() as u64)
            .collect::<BTreeSet<_>>();
        self.items
            .iter()
            .filter(|item| !requested.contains(&item.output.id))
            .map(|item| item.output.clone())
            .collect()
    }
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
        view: None,
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
            | Op::Shrink { .. }
            | Op::Reduce {
                kind: crate::ReduceKind::Sum | crate::ReduceKind::Mean,
                ..
            }
    )
}
/// Creates one conservative fused item for a pure elementwise output. Anything
/// else is a visible schedule boundary, never an implicit mislowering.
pub fn schedule(graph: &Graph, output: NodeId) -> Result<Schedule, ScheduleError> {
    schedule_many(graph, &[output])
}
/// Schedules requested graph outputs as a stable producer-aware DAG. Pure
/// elementwise/view chains are fused until an explicit materialization root.
pub fn schedule_many(graph: &Graph, outputs: &[NodeId]) -> Result<Schedule, ScheduleError> {
    if outputs.is_empty() {
        return Ok(Schedule { items: vec![] });
    }
    let mut needed = BTreeSet::new();
    let mut consumers = vec![0usize; graph.node_count()];
    fn mark(
        g: &Graph,
        id: NodeId,
        needed: &mut BTreeSet<usize>,
        consumers: &mut [usize],
    ) -> Result<(), ScheduleError> {
        if !needed.insert(id.index()) {
            return Ok(());
        }
        let mut child = |child: NodeId| -> Result<(), ScheduleError> {
            consumers[child.index()] += 1;
            mark(g, child, needed, consumers)
        };
        match g.op(id).map_err(ScheduleError::Graph)? {
            Op::Cast { input, .. }
            | Op::Unary { input, .. }
            | Op::Shrink { input, .. }
            | Op::Reduce { input, .. } => child(*input)?,
            Op::Binary { lhs, rhs, .. } | Op::Compare { lhs, rhs, .. } => {
                child(*lhs)?;
                child(*rhs)?;
            }
            Op::Logical { lhs, rhs, .. } => {
                child(*lhs)?;
                if let Some(rhs) = rhs {
                    child(*rhs)?;
                }
            }
            Op::Select {
                condition,
                on_true,
                on_false,
            } => {
                child(*condition)?;
                child(*on_true)?;
                child(*on_false)?;
            }
            _ => {}
        };
        Ok(())
    }
    for output in outputs {
        graph.op(*output).map_err(ScheduleError::Graph)?;
        mark(graph, *output, &mut needed, &mut consumers)?;
    }
    let requested: BTreeSet<usize> = outputs.iter().map(|id| id.index()).collect();
    let roots: BTreeSet<usize> = needed
        .iter()
        .copied()
        .filter(|index| {
            let id = NodeId::from_index(*index);
            requested.contains(index)
                || (consumers[*index] > 1
                    && !matches!(graph.op(id), Ok(Op::Input { .. } | Op::Constant(_))))
                || matches!(graph.op(id), Ok(Op::Reduce { .. }))
                || !matches!(graph.op(id), Ok(op) if supported(op))
        })
        .collect();
    let node_to_item: std::collections::BTreeMap<usize, u64> = roots
        .iter()
        .enumerate()
        .map(|(item, node)| (*node, item as u64))
        .collect();
    fn leaves(
        g: &Graph,
        id: NodeId,
        roots: &BTreeSet<usize>,
        here: usize,
        out: &mut BTreeSet<usize>,
        boundary: &mut Option<ScheduleBoundary>,
    ) -> Result<(), ScheduleError> {
        if id.index() != here && roots.contains(&id.index()) {
            out.insert(id.index());
            return Ok(());
        }
        let op = g.op(id).map_err(ScheduleError::Graph)?;
        if !supported(op) {
            *boundary = Some(ScheduleBoundary::Unsupported(
                "operation requires materialization",
            ));
            out.insert(id.index());
            return Ok(());
        }
        match op {
            Op::Input { .. } | Op::Constant(_) => {
                out.insert(id.index());
            }
            Op::Cast { input, .. } | Op::Unary { input, .. } | Op::Reduce { input, .. } => {
                leaves(g, *input, roots, here, out, boundary)?
            }
            Op::Shrink { input, .. } => match g.op(*input).map_err(ScheduleError::Graph)? {
                Op::Input { .. } | Op::Constant(_) | Op::Shrink { .. } => {
                    leaves(g, *input, roots, here, out, boundary)?
                }
                _ => {
                    *boundary = Some(ScheduleBoundary::Unsupported(
                        "shrink of a computed value requires materialization",
                    ));
                    out.insert(input.index());
                }
            },
            Op::Binary { lhs, rhs, .. } | Op::Compare { lhs, rhs, .. } => {
                leaves(g, *lhs, roots, here, out, boundary)?;
                leaves(g, *rhs, roots, here, out, boundary)?;
            }
            Op::Logical { lhs, rhs, .. } => {
                leaves(g, *lhs, roots, here, out, boundary)?;
                if let Some(rhs) = rhs {
                    leaves(g, *rhs, roots, here, out, boundary)?;
                }
            }
            Op::Select {
                condition,
                on_true,
                on_false,
            } => {
                leaves(g, *condition, roots, here, out, boundary)?;
                leaves(g, *on_true, roots, here, out, boundary)?;
                leaves(g, *on_false, roots, here, out, boundary)?;
            }
            _ => unreachable!(),
        };
        Ok(())
    }
    let mut items = Vec::with_capacity(roots.len());
    for &index in &roots {
        let node = NodeId::from_index(index);
        let mut leaf_ids = BTreeSet::new();
        let mut boundary = None;
        leaves(graph, node, &roots, index, &mut leaf_ids, &mut boundary)?;
        let materialized = leaf_ids
            .iter()
            .filter(|leaf| roots.contains(leaf))
            .copied()
            .collect::<BTreeSet<_>>();
        let mut inputs = leaf_ids
            .into_iter()
            .map(|leaf| buffer(graph, NodeId::from_index(leaf), true))
            .collect::<Result<Vec<_>, _>>()?;
        let output = buffer(graph, node, false)?;
        let kernel = if boundary.is_none() {
            match graph.op(node).map_err(ScheduleError::Graph)? {
                Op::Reduce { .. } => crate::kernel::lower_graph_reduction_with_materialized(
                    graph,
                    node,
                    &materialized,
                )
                .map_err(ScheduleError::UOp)?,
                _ => crate::kernel::lower_graph_elementwise_with_materialized(
                    graph,
                    node,
                    &materialized,
                )
                .map_err(ScheduleError::UOp)?,
            }
        } else {
            UOp::sink(vec![])
        };
        for value in kernel.topological().map_err(ScheduleError::UOp)? {
            if let crate::UArg::ViewBufferIndex { buffer, view, .. } = value.arg()
                && let Some(desc) = inputs.iter_mut().find(|desc| desc.id == *buffer)
            {
                desc.view = Some(view.clone());
            }
        }
        let id = *node_to_item.get(&index).ok_or(ScheduleError::Overflow)?;
        let dependencies: Vec<u64> = inputs
            .iter()
            .filter_map(|desc| node_to_item.get(&(desc.id as usize)).copied())
            .collect();
        let mut h = DefaultHasher::new();
        id.hash(&mut h);
        node.hash(&mut h);
        dependencies.hash(&mut h);
        inputs.hash(&mut h);
        output.hash(&mut h);
        boundary.hash(&mut h);
        kernel.hash(&mut h);
        items.push(ScheduleItem {
            id,
            node,
            dependencies,
            consumers: vec![],
            inputs,
            output,
            kernel,
            boundary,
            cache_key: h.finish(),
        });
    }
    let positions: std::collections::BTreeMap<u64, usize> = items
        .iter()
        .enumerate()
        .map(|(position, item)| (item.id, position))
        .collect();
    for item in items.clone() {
        for dependency in item.dependencies {
            items[*positions.get(&dependency).ok_or(ScheduleError::Overflow)?]
                .consumers
                .push(item.id);
        }
    }
    Ok(Schedule { items })
}
/* legacy single-root lowering retained below for reference during the DAG transition. */
#[allow(dead_code)]
fn schedule_single_legacy(graph: &Graph, output: NodeId) -> Result<Schedule, ScheduleError> {
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
            Op::Shrink { input, .. } => match g.op(*input).map_err(ScheduleError::Graph)? {
                Op::Input { .. } | Op::Constant(_) | Op::Shrink { .. } => {
                    walk(g, *input, leaves, boundary)?
                }
                _ => {
                    *boundary = Some(ScheduleBoundary::Unsupported(
                        "shrink of a computed value requires materialization",
                    ));
                    leaves.insert(input.index());
                }
            },
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
    let mut inputs = leaves
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
    for node in kernel.topological().map_err(ScheduleError::UOp)? {
        if let crate::UArg::ViewBufferIndex { buffer, view, .. } = node.arg()
            && let Some(desc) = inputs.iter_mut().find(|desc| desc.id == *buffer)
        {
            desc.view = Some(view.clone());
        }
    }
    let mut h = DefaultHasher::new();
    inputs.hash(&mut h);
    out.hash(&mut h);
    boundary.hash(&mut h);
    kernel.hash(&mut h);
    let cache_key = h.finish();
    Ok(Schedule {
        items: vec![ScheduleItem {
            id: 0,
            node: output,
            dependencies: vec![],
            consumers: vec![],
            inputs,
            output: out,
            kernel,
            boundary,
            cache_key,
        }],
    })
}
