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
            *boundary = Some(ScheduleBoundary::Unsupported(
                "operation is outside phase-one elementwise lowering",
            ));
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
    let kernel = if boundary.is_none() {
        crate::kernel::lower_graph_elementwise(graph, output).map_err(ScheduleError::UOp)?
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
