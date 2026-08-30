//! Typed late linearization of ranged UOps for portable lane renderers.
use crate::{DType, IndexValue, Operation, Shape, UOp};
use std::{
    collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum LinearAccess {
    ContiguousVector,
    ScalarSplat,
    ScalarOnly(String),
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LinearBuffer {
    pub buffer: u64,
    pub dtype: DType,
    pub elements: usize,
    pub input_shape: Shape,
    pub byte_offset: usize,
    pub byte_stride: usize,
    pub alignment: usize,
    pub mutable: bool,
    pub access: LinearAccess,
}
#[derive(Clone, Debug)]
pub struct LinearKernel {
    /// Retained immutable source DAG; scalar UOp meaning is unchanged.
    pub source: UOp,
    pub output_buffer: u64,
    pub output_shape: Shape,
    pub dtype: DType,
    pub elements: usize,
    pub lanes: usize,
    pub vector_main: usize,
    pub scalar_tail: usize,
    pub tail_mask: Vec<bool>,
    pub buffers: Vec<LinearBuffer>,
    pub enabled: bool,
    pub reason: String,
    pub cache_key: u64,
    pub program: LinearProgram,
}
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RegisterClass {
    Scalar,
    Vector,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum LinearInstKind {
    Constant,
    Address,
    Index,
    Load { buffer: u64 },
    Cast,
    Unary,
    Binary,
    Compare,
    Select,
    Store { buffer: u64 },
    Other(String),
}
/// Exact immutable semantic data retained for late renderers.  This prevents a
/// backend from silently consulting the original UOp DAG to recover an opcode,
/// literal, cast type, or checked index/view descriptor.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LinearPayload {
    pub operation: Operation,
    pub ty: Option<crate::UType>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LinearInst {
    pub index: u32,
    pub dst: Option<u32>,
    pub inputs: Vec<u32>,
    pub kind: LinearInstKind,
    pub dtype: DType,
    pub lanes: u16,
    pub tail_mask: Vec<bool>,
    pub payload: LinearPayload,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LiveInterval {
    pub virtual_reg: u32,
    pub class: RegisterClass,
    /// C expression type: F16/BF16 are decoded into F32 registers.
    pub dtype: DType,
    pub start: u32,
    pub end: u32,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RegisterAssignment {
    pub virtual_reg: u32,
    pub class: RegisterClass,
    pub dtype: DType,
    pub physical_reg: u32,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LinearProgram {
    pub instructions: Vec<LinearInst>,
    pub intervals: Vec<LiveInterval>,
    pub assignments: Vec<RegisterAssignment>,
    pub peak_scalar: usize,
    pub peak_vector: usize,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinearizeError {
    MissingStore,
    Untyped,
    Overflow,
    Invalid(String),
    RegisterPressure { class: RegisterClass, limit: usize },
}
impl fmt::Display for LinearizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "linearize error: {self:?}")
    }
}
impl std::error::Error for LinearizeError {}

impl LinearKernel {
    pub fn from_uop(source: &UOp) -> Result<Self, LinearizeError> {
        source
            .topological()
            .map_err(|e| LinearizeError::Invalid(e.to_string()))?;
        let mut nodes = Vec::new();
        producer_order(source, &mut BTreeSet::new(), &mut nodes);
        let store = source
            .sources()
            .iter()
            .find(|node| matches!(node.operation(), Operation::Store))
            .ok_or(LinearizeError::MissingStore)?;
        let output = store
            .sources()
            .first()
            .ok_or(LinearizeError::MissingStore)?;
        let (output_buffer, elements, output_shape) = match output.operation() {
            Operation::Index(IndexValue::Buffer {
                buffer,
                elements,
                output_shape,
                ..
            }) => (*buffer, *elements, output_shape.clone()),
            _ => return Err(LinearizeError::MissingStore),
        };
        let dtype = output.ty().ok_or(LinearizeError::Untyped)?.scalar;
        let lanes = (16 / dtype.itemsize()).max(1);
        let mut enabled = lanes > 1;
        let mut reason = if enabled {
            "contiguous portable lane plan".to_string()
        } else {
            "64-bit scalar policy".to_string()
        };
        if nodes.iter().any(|node| {
            matches!(
                node.operation(),
                Operation::ReduceInit(_)
                    | Operation::ReduceAccumulate
                    | Operation::ReduceFinalize
                    | Operation::Barrier
            )
        }) {
            enabled = false;
            reason = "reduction or effect requires scalar path".into();
        }
        let mut buffers = BTreeMap::new();
        for node in &nodes {
            let Some(ty) = node.ty() else { continue };
            let (
                buffer,
                logical_count,
                physical_count,
                input_shape,
                indexed_output,
                offset,
                contiguous,
            ) = match node.operation() {
                Operation::Index(IndexValue::Buffer {
                    buffer,
                    elements,
                    input_shape,
                    output_shape,
                }) => (
                    *buffer,
                    *elements,
                    *elements,
                    input_shape.clone(),
                    output_shape.clone(),
                    0usize,
                    true,
                ),
                Operation::Index(IndexValue::View {
                    buffer,
                    elements,
                    input_shape,
                    output_shape,
                    view,
                }) => {
                    let contiguous = view.strides
                        == view
                            .logical_shape
                            .contiguous_strides()
                            .into_iter()
                            .map(|stride| stride as i64)
                            .collect::<Vec<_>>();
                    (
                        *buffer,
                        *elements,
                        view.source_shape
                            .numel()
                            .map_err(|_| LinearizeError::Overflow)?,
                        input_shape.clone(),
                        output_shape.clone(),
                        usize::try_from(view.offset).map_err(|_| LinearizeError::Overflow)?,
                        contiguous,
                    )
                }
                _ => continue,
            };
            let byte_offset = offset
                .checked_mul(ty.scalar.itemsize())
                .ok_or(LinearizeError::Overflow)?;
            let access = if buffer == output_buffer {
                LinearAccess::ContiguousVector
            } else if physical_count == 1 {
                LinearAccess::ScalarSplat
            } else if logical_count == 1
                || indexed_output != output_shape
                || input_shape != output_shape
                || !contiguous
            {
                enabled = false;
                reason = "varying broadcast, view, or non-contiguous index".into();
                LinearAccess::ScalarOnly(reason.clone())
            } else if byte_offset % (lanes * ty.scalar.itemsize()) != 0 {
                enabled = false;
                reason = "misaligned view byte offset".into();
                LinearAccess::ScalarOnly(reason.clone())
            } else {
                LinearAccess::ContiguousVector
            };
            buffers.entry(buffer).or_insert(LinearBuffer {
                buffer,
                dtype: ty.scalar,
                elements: physical_count,
                input_shape,
                byte_offset,
                byte_stride: ty.scalar.itemsize(),
                alignment: ty.scalar.itemsize().max(1),
                mutable: buffer == output_buffer,
                access,
            });
        }
        let vector_main = if enabled { elements / lanes * lanes } else { 0 };
        let scalar_tail = elements
            .checked_sub(vector_main)
            .ok_or(LinearizeError::Overflow)?;
        let tail_mask = (0..lanes)
            .map(|lane| lane < scalar_tail)
            .collect::<Vec<_>>();
        let buffers = buffers.into_values().collect::<Vec<_>>();
        let program = linear_program(
            &nodes,
            dtype,
            if enabled { lanes as u16 } else { 1 },
            &tail_mask,
        )?;
        let mut h = DefaultHasher::new();
        output_buffer.hash(&mut h);
        output_shape.hash(&mut h);
        dtype.hash(&mut h);
        elements.hash(&mut h);
        lanes.hash(&mut h);
        vector_main.hash(&mut h);
        scalar_tail.hash(&mut h);
        tail_mask.hash(&mut h);
        buffers.hash(&mut h);
        enabled.hash(&mut h);
        reason.hash(&mut h);
        program.hash(&mut h);
        Ok(Self {
            source: source.clone(),
            output_buffer,
            output_shape,
            dtype,
            elements,
            lanes,
            vector_main,
            scalar_tail,
            tail_mask,
            buffers,
            enabled,
            reason,
            cache_key: h.finish(),
            program,
        })
    }
    pub fn validate(&self) -> Result<(), LinearizeError> {
        if self.lanes == 0 || self.tail_mask.len() != self.lanes {
            return Err(LinearizeError::Invalid("invalid lane mask".into()));
        }
        if self.vector_main.checked_add(self.scalar_tail) != Some(self.elements) {
            return Err(LinearizeError::Overflow);
        }
        if self.enabled && self.vector_main % self.lanes != 0 {
            return Err(LinearizeError::Invalid(
                "vector main is not lane aligned".into(),
            ));
        }
        if self.buffers.iter().filter(|buffer| buffer.mutable).count() != 1 {
            return Err(LinearizeError::Invalid(
                "requires exactly one mutable output".into(),
            ));
        }
        validate_program(&self.program)?;
        Ok(())
    }
}

fn producer_order(node: &UOp, seen: &mut BTreeSet<String>, output: &mut Vec<UOp>) {
    for source in node.sources() {
        producer_order(source, seen, output);
    }
    if seen.insert(format!("{node:?}")) {
        output.push(node.clone());
    }
}

fn linear_program(
    nodes: &[UOp],
    dtype: DType,
    lanes: u16,
    tail_mask: &[bool],
) -> Result<LinearProgram, LinearizeError> {
    let mut ids = BTreeMap::new();
    for (index, node) in nodes.iter().enumerate() {
        ids.entry(format!("{node:?}")).or_insert(index as u32);
    }
    let mut instructions = Vec::with_capacity(nodes.len());
    for (index, node) in nodes.iter().enumerate() {
        let inputs = node
            .sources()
            .iter()
            .filter(|source| !matches!(source.operation(), Operation::Store))
            .filter_map(|source| ids.get(&format!("{source:?}")).copied())
            .collect::<Vec<_>>();
        let kind = match node.operation() {
            Operation::Const(_) | Operation::VConst(_) => LinearInstKind::Constant,
            Operation::DefineGlobal(_)
            | Operation::DefineLocal(_)
            | Operation::DefineRegister(_) => LinearInstKind::Address,
            Operation::Index(_) => LinearInstKind::Index,
            Operation::Load => {
                match node
                    .sources()
                    .first()
                    .and_then(|source| match source.operation() {
                        Operation::Index(IndexValue::Buffer { buffer, .. })
                        | Operation::Index(IndexValue::View { buffer, .. }) => Some(*buffer),
                        _ => None,
                    }) {
                    Some(buffer) => LinearInstKind::Load { buffer },
                    None => LinearInstKind::Other("untyped load".into()),
                }
            }
            Operation::Cast | Operation::Bitcast => LinearInstKind::Cast,
            Operation::Unary(_) | Operation::GraphUnary(_) => LinearInstKind::Unary,
            Operation::Binary(_) | Operation::GraphBinary(_) | Operation::GraphLogical(_) => {
                LinearInstKind::Binary
            }
            Operation::GraphCompare(_) => LinearInstKind::Compare,
            Operation::Ternary(_) => LinearInstKind::Select,
            Operation::Store => LinearInstKind::Store {
                buffer: source_output_buffer(node).unwrap_or(u64::MAX),
            },
            other => LinearInstKind::Other(format!("{other:?}")),
        };
        let typed = node.ty().unwrap_or(crate::UType::scalar(dtype));
        let dst = (!matches!(kind, LinearInstKind::Store { .. })).then_some(index as u32);
        instructions.push(LinearInst {
            index: index as u32,
            dst,
            inputs,
            kind,
            dtype: typed.scalar,
            lanes: if typed.lanes == 1 { lanes } else { typed.lanes },
            tail_mask: tail_mask.to_vec(),
            payload: LinearPayload {
                operation: node.operation().clone(),
                ty: node.ty(),
            },
        });
    }
    let intervals = intervals(&instructions);
    let assignments = allocate(&intervals, 64)?;
    let peak_scalar = assignments
        .iter()
        .filter(|assignment| assignment.class == RegisterClass::Scalar)
        .map(|assignment| assignment.physical_reg as usize + 1)
        .max()
        .unwrap_or(0);
    let peak_vector = assignments
        .iter()
        .filter(|assignment| assignment.class == RegisterClass::Vector)
        .map(|assignment| assignment.physical_reg as usize + 1)
        .max()
        .unwrap_or(0);
    Ok(LinearProgram {
        instructions,
        intervals,
        assignments,
        peak_scalar,
        peak_vector,
    })
}
fn source_output_buffer(source: &UOp) -> Option<u64> {
    source
        .sources()
        .first()
        .and_then(|node| match node.operation() {
            Operation::Index(IndexValue::Buffer { buffer, .. }) => Some(*buffer),
            _ => None,
        })
}
fn intervals(instructions: &[LinearInst]) -> Vec<LiveInterval> {
    let mut result = Vec::new();
    for instruction in instructions {
        if let Some(reg) = instruction.dst {
            let end = instructions
                .iter()
                .filter(|other| other.inputs.contains(&reg))
                .map(|other| other.index)
                .max()
                .unwrap_or(instruction.index);
            result.push(LiveInterval {
                virtual_reg: reg,
                class: if instruction.lanes > 1 {
                    RegisterClass::Vector
                } else {
                    RegisterClass::Scalar
                },
                dtype: match instruction.dtype {
                    DType::F16 | DType::BF16 => DType::F32,
                    x => x,
                },
                start: instruction.index,
                end,
            });
        }
    }
    result
}
pub fn allocate(
    intervals: &[LiveInterval],
    limit: usize,
) -> Result<Vec<RegisterAssignment>, LinearizeError> {
    let mut sorted = intervals.to_vec();
    sorted.sort_by_key(|interval| {
        (
            interval.class,
            interval.dtype,
            interval.start,
            interval.virtual_reg,
        )
    });
    let mut active: BTreeMap<(RegisterClass, DType), Vec<(u32, u32)>> = BTreeMap::new();
    let mut result = Vec::new();
    for interval in sorted {
        let live = active.entry((interval.class, interval.dtype)).or_default();
        live.retain(|(end, _)| *end >= interval.start);
        let physical = (0..limit as u32)
            .find(|candidate| !live.iter().any(|(_, used)| used == candidate))
            .ok_or(LinearizeError::RegisterPressure {
                class: interval.class,
                limit,
            })?;
        live.push((interval.end, physical));
        live.sort();
        result.push(RegisterAssignment {
            virtual_reg: interval.virtual_reg,
            class: interval.class,
            dtype: interval.dtype,
            physical_reg: physical,
        });
    }
    result.sort_by_key(|assignment| assignment.virtual_reg);
    Ok(result)
}
fn validate_program(program: &LinearProgram) -> Result<(), LinearizeError> {
    let mut defined = BTreeMap::new();
    for instruction in &program.instructions {
        for input in &instruction.inputs {
            if !defined.contains_key(input) {
                return Err(LinearizeError::Invalid(format!(
                    "use before definition r{input}"
                )));
            }
        }
        if let Some(dst) = instruction.dst
            && defined.insert(dst, instruction.index).is_some()
        {
            return Err(LinearizeError::Invalid(format!(
                "duplicate definition r{dst}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Shape};
    #[test]
    fn snapshots_contiguous_and_varying_broadcast_plans() {
        let mut graph = Graph::new();
        let x = graph.input("x", Shape::from([5]));
        let out = graph.square(x).unwrap();
        let plan =
            LinearKernel::from_uop(&crate::lower_graph_elementwise(&graph, out).unwrap()).unwrap();
        plan.validate().unwrap();
        assert!(plan.enabled);
        assert_eq!((plan.lanes, plan.vector_main, plan.scalar_tail), (4, 4, 1));
        let mut broadcast = Graph::new();
        let a = broadcast.input("a", Shape::from([2, 3]));
        let b = broadcast.input("b", Shape::from([1, 3]));
        let out = broadcast.add(a, b).unwrap();
        let plan =
            LinearKernel::from_uop(&crate::lower_graph_elementwise(&broadcast, out).unwrap())
                .unwrap();
        assert!(!plan.enabled);
        assert!(plan.reason.contains("varying"));

        let mut views = Graph::new();
        let x = views.input("x", Shape::from([8]));
        let aligned = views.shrink(x, vec![(4, 8)]).unwrap();
        let out = views.neg(aligned).unwrap();
        assert!(
            LinearKernel::from_uop(&crate::lower_graph_elementwise(&views, out).unwrap())
                .unwrap()
                .enabled
        );
        let misaligned = views.shrink(x, vec![(1, 5)]).unwrap();
        let out = views.neg(misaligned).unwrap();
        let plan =
            LinearKernel::from_uop(&crate::lower_graph_elementwise(&views, out).unwrap()).unwrap();
        assert!(!plan.enabled);
        assert!(plan.reason.contains("misaligned"));
    }

    #[test]
    fn deterministic_linear_scan_reuses_and_reports_pressure() {
        let intervals = vec![
            LiveInterval {
                virtual_reg: 2,
                class: RegisterClass::Vector,
                dtype: DType::F32,
                start: 0,
                end: 1,
            },
            LiveInterval {
                virtual_reg: 7,
                class: RegisterClass::Vector,
                dtype: DType::F32,
                start: 2,
                end: 3,
            },
        ];
        let first = allocate(&intervals, 1).unwrap();
        assert_eq!(first[0].physical_reg, 0);
        assert_eq!(first[1].physical_reg, 0);
        assert_eq!(first, allocate(&intervals, 1).unwrap());
        let overlapping = vec![
            LiveInterval {
                virtual_reg: 1,
                class: RegisterClass::Scalar,
                dtype: DType::I32,
                start: 0,
                end: 2,
            },
            LiveInterval {
                virtual_reg: 2,
                class: RegisterClass::Scalar,
                dtype: DType::I32,
                start: 1,
                end: 3,
            },
        ];
        assert_eq!(
            allocate(&overlapping, 1),
            Err(LinearizeError::RegisterPressure {
                class: RegisterClass::Scalar,
                limit: 1
            })
        );
    }
}
