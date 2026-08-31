//! Typed late linearization of ranged UOps for portable lane renderers.
use crate::{
    AddressValue, BinaryOp, CompareOp, DType, IndexValue, LiteralValue, Operation, Shape, UOp,
    UType, UnaryOp,
};
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
pub struct TypedValue<R> {
    pub register: R,
    pub ty: UType,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct AddressRef<R> {
    pub register: R,
    pub value: AddressValue,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct IndexRef<R> {
    pub register: R,
    pub value: IndexValue,
    pub element: UType,
}

impl<R> IndexRef<R> {
    pub fn buffer(&self) -> u64 {
        match &self.value {
            IndexValue::Buffer { buffer, .. } | IndexValue::View { buffer, .. } => *buffer,
        }
    }
}

/// One exact lane-level instruction used with virtual or allocated operands.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum LaneInstruction<R> {
    Constant {
        output: TypedValue<R>,
        value: LiteralValue,
        vector: bool,
    },
    Address {
        output: AddressRef<R>,
    },
    Range {
        output: TypedValue<R>,
        bound: TypedValue<R>,
        axis: u32,
    },
    Index {
        output: IndexRef<R>,
        address: AddressRef<R>,
        offset: TypedValue<R>,
    },
    Load {
        output: TypedValue<R>,
        index: IndexRef<R>,
    },
    Cast {
        output: TypedValue<R>,
        input: TypedValue<R>,
    },
    Bitcast {
        output: TypedValue<R>,
        input: TypedValue<R>,
    },
    CoreUnary {
        output: TypedValue<R>,
        input: TypedValue<R>,
        op: crate::uop::Unary,
    },
    GraphUnary {
        output: TypedValue<R>,
        input: TypedValue<R>,
        op: UnaryOp,
    },
    CoreBinary {
        output: TypedValue<R>,
        lhs: TypedValue<R>,
        rhs: TypedValue<R>,
        op: crate::uop::Binary,
    },
    CoreEq {
        output: TypedValue<R>,
        lhs: TypedValue<R>,
        rhs: TypedValue<R>,
    },
    CoreLt {
        output: TypedValue<R>,
        lhs: TypedValue<R>,
        rhs: TypedValue<R>,
    },
    CoreLe {
        output: TypedValue<R>,
        lhs: TypedValue<R>,
        rhs: TypedValue<R>,
    },
    GraphBinary {
        output: TypedValue<R>,
        lhs: TypedValue<R>,
        rhs: TypedValue<R>,
        op: BinaryOp,
    },
    LogicalNot {
        output: TypedValue<R>,
        input: TypedValue<R>,
    },
    LogicalAnd {
        output: TypedValue<R>,
        lhs: TypedValue<R>,
        rhs: TypedValue<R>,
    },
    LogicalOr {
        output: TypedValue<R>,
        lhs: TypedValue<R>,
        rhs: TypedValue<R>,
    },
    Compare {
        output: TypedValue<R>,
        lhs: TypedValue<R>,
        rhs: TypedValue<R>,
        op: CompareOp,
    },
    Select {
        output: TypedValue<R>,
        condition: TypedValue<R>,
        on_true: TypedValue<R>,
        on_false: TypedValue<R>,
    },
    Store {
        index: IndexRef<R>,
        value: TypedValue<R>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LaneDescriptorRef<'a> {
    Value(UType),
    Address(&'a AddressValue),
    Index {
        value: &'a IndexValue,
        element: UType,
    },
}

impl LaneDescriptorRef<'_> {
    fn ty(self) -> UType {
        match self {
            Self::Value(ty) => ty,
            Self::Address(value) => value.element,
            Self::Index { element, .. } => element,
        }
    }

    fn into_owned(self) -> LaneDescriptor {
        match self {
            Self::Value(ty) => LaneDescriptor::Value(ty),
            Self::Address(value) => LaneDescriptor::Address(value.clone()),
            Self::Index { value, element } => LaneDescriptor::Index {
                value: value.clone(),
                element,
            },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum LaneDescriptor {
    Value(UType),
    Address(AddressValue),
    Index { value: IndexValue, element: UType },
}

impl LaneDescriptor {
    pub(crate) fn ty(&self) -> UType {
        match self {
            Self::Value(ty) => *ty,
            Self::Address(value) => value.element,
            Self::Index { element, .. } => *element,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LaneOperandView<'a, R> {
    register: &'a R,
    descriptor: LaneDescriptorRef<'a>,
}

pub struct LaneInstructionView<'a, R> {
    inputs: [Option<LaneOperandView<'a, R>>; 3],
    output: Option<LaneOperandView<'a, R>>,
    pub buffer: Option<u64>,
    pub semantic_name: &'static str,
}

impl<'a, R> LaneInstructionView<'a, R> {
    pub fn inputs(&self) -> impl Iterator<Item = &'a R> + '_ {
        self.inputs.iter().flatten().map(|operand| operand.register)
    }

    pub fn typed_inputs(&self) -> impl Iterator<Item = (&'a R, UType)> + '_ {
        self.inputs
            .iter()
            .flatten()
            .map(|operand| (operand.register, operand.descriptor.ty()))
    }

    pub fn output(&self) -> Option<&'a R> {
        self.output.as_ref().map(|operand| operand.register)
    }

    pub fn result_type(&self) -> Option<UType> {
        self.output.as_ref().map(|operand| operand.descriptor.ty())
    }
}

fn value_operand<'a, R>(value: &'a TypedValue<R>) -> LaneOperandView<'a, R> {
    LaneOperandView {
        register: &value.register,
        descriptor: LaneDescriptorRef::Value(value.ty),
    }
}

fn address_operand<'a, R>(address: &'a AddressRef<R>) -> LaneOperandView<'a, R> {
    LaneOperandView {
        register: &address.register,
        descriptor: LaneDescriptorRef::Address(&address.value),
    }
}

fn index_operand<'a, R>(index: &'a IndexRef<R>) -> LaneOperandView<'a, R> {
    LaneOperandView {
        register: &index.register,
        descriptor: LaneDescriptorRef::Index {
            value: &index.value,
            element: index.element,
        },
    }
}

impl<R> LaneInstruction<R> {
    pub fn view(&self) -> LaneInstructionView<'_, R> {
        match self {
            Self::Constant { output, vector, .. } => LaneInstructionView {
                inputs: [None, None, None],
                output: Some(value_operand(output)),
                buffer: None,
                semantic_name: if *vector { "vconst" } else { "const" },
            },
            Self::Address { output } => LaneInstructionView {
                inputs: [None, None, None],
                output: Some(address_operand(output)),
                buffer: None,
                semantic_name: match output.value.space {
                    crate::AddressSpace::Global => "define_global",
                    crate::AddressSpace::Local => "define_local",
                    crate::AddressSpace::Register => "define_register",
                },
            },
            Self::Range { output, bound, .. } => LaneInstructionView {
                inputs: [Some(value_operand(bound)), None, None],
                output: Some(value_operand(output)),
                buffer: None,
                semantic_name: "range",
            },
            Self::Index {
                output,
                address,
                offset,
            } => LaneInstructionView {
                inputs: [
                    Some(address_operand(address)),
                    Some(value_operand(offset)),
                    None,
                ],
                output: Some(index_operand(output)),
                buffer: Some(output.buffer()),
                semantic_name: "index",
            },
            Self::Load { output, index } => LaneInstructionView {
                inputs: [Some(index_operand(index)), None, None],
                output: Some(value_operand(output)),
                buffer: Some(index.buffer()),
                semantic_name: "load",
            },
            Self::Cast { output, input } | Self::Bitcast { output, input } => LaneInstructionView {
                inputs: [Some(value_operand(input)), None, None],
                output: Some(value_operand(output)),
                buffer: None,
                semantic_name: if matches!(self, Self::Cast { .. }) {
                    "cast"
                } else {
                    "bitcast"
                },
            },
            Self::CoreUnary { output, input, .. }
            | Self::GraphUnary { output, input, .. }
            | Self::LogicalNot { output, input } => LaneInstructionView {
                inputs: [Some(value_operand(input)), None, None],
                output: Some(value_operand(output)),
                buffer: None,
                semantic_name: match self {
                    Self::CoreUnary { .. } => "unary",
                    Self::GraphUnary { .. } => "graph_unary",
                    Self::LogicalNot { .. } => "graph_logical",
                    _ => unreachable!(),
                },
            },
            Self::CoreBinary {
                output, lhs, rhs, ..
            }
            | Self::CoreEq { output, lhs, rhs }
            | Self::CoreLt { output, lhs, rhs }
            | Self::CoreLe { output, lhs, rhs }
            | Self::GraphBinary {
                output, lhs, rhs, ..
            }
            | Self::LogicalAnd { output, lhs, rhs }
            | Self::LogicalOr { output, lhs, rhs }
            | Self::Compare {
                output, lhs, rhs, ..
            } => LaneInstructionView {
                inputs: [Some(value_operand(lhs)), Some(value_operand(rhs)), None],
                output: Some(value_operand(output)),
                buffer: None,
                semantic_name: match self {
                    Self::CoreBinary { .. } => "binary",
                    Self::CoreEq { .. } | Self::CoreLt { .. } | Self::CoreLe { .. } => "compare",
                    Self::GraphBinary { .. } => "graph_binary",
                    Self::LogicalAnd { .. } | Self::LogicalOr { .. } => "graph_logical",
                    Self::Compare { .. } => "graph_compare",
                    _ => unreachable!(),
                },
            },
            Self::Select {
                output,
                condition,
                on_true,
                on_false,
            } => LaneInstructionView {
                inputs: [
                    Some(value_operand(condition)),
                    Some(value_operand(on_true)),
                    Some(value_operand(on_false)),
                ],
                output: Some(value_operand(output)),
                buffer: None,
                semantic_name: "where",
            },
            Self::Store { index, value } => LaneInstructionView {
                inputs: [Some(index_operand(index)), Some(value_operand(value)), None],
                output: None,
                buffer: Some(index.buffer()),
                semantic_name: "store",
            },
        }
    }

    pub fn validate(&self) -> Result<(), LinearizeError> {
        let same = |a: UType, b: UType| (a == b).then_some(()).ok_or(LinearizeError::Untyped);
        match self {
            Self::Constant { output, value, .. } => match value {
                LiteralValue::Int(_) => Ok(()),
                LiteralValue::Scalar { dtype, .. } if *dtype == output.ty.scalar => Ok(()),
                LiteralValue::Scalar { .. } => Err(LinearizeError::Untyped),
            },
            Self::Address { .. } => Ok(()),
            Self::Range { output, bound, .. } if output.ty.scalar.is_integer() => {
                same(output.ty, bound.ty)
            }
            Self::Range { .. } => Err(LinearizeError::Untyped),
            Self::Index {
                output,
                address,
                offset,
            } if offset.ty.scalar.is_integer() && output.element == address.value.element => Ok(()),
            Self::Index { .. } => Err(LinearizeError::Untyped),
            Self::Load { output, index } => same(output.ty, index.element),
            Self::Store { index, value } => same(index.element, value.ty),
            Self::Cast { output, input } | Self::Bitcast { output, input }
                if output.ty.lanes == input.ty.lanes =>
            {
                Ok(())
            }
            Self::Cast { .. } | Self::Bitcast { .. } => Err(LinearizeError::Untyped),
            Self::CoreUnary { output, input, .. } => same(output.ty, input.ty),
            Self::GraphUnary { output, input, op } => {
                let expected = UType {
                    scalar: crate::ir::unary_dtype(*op, input.ty.scalar),
                    lanes: input.ty.lanes,
                };
                same(output.ty, expected)
            }
            Self::CoreBinary {
                output,
                lhs,
                rhs,
                op,
            } if !matches!(
                op,
                crate::uop::Binary::Eq | crate::uop::Binary::Lt | crate::uop::Binary::Le
            ) =>
            {
                same(output.ty, lhs.ty).and_then(|()| same(output.ty, rhs.ty))
            }
            Self::CoreBinary { .. } => Err(LinearizeError::Untyped),
            Self::GraphBinary {
                output, lhs, rhs, ..
            } if output.ty.lanes == lhs.ty.lanes && lhs.ty.lanes == rhs.ty.lanes => Ok(()),
            Self::GraphBinary { .. } => Err(LinearizeError::Untyped),
            Self::CoreEq { output, lhs, rhs }
            | Self::CoreLt { output, lhs, rhs }
            | Self::CoreLe { output, lhs, rhs }
                if output.ty.scalar == DType::Bool
                    && output.ty.lanes == lhs.ty.lanes
                    && lhs.ty == rhs.ty =>
            {
                Ok(())
            }
            Self::CoreEq { .. } | Self::CoreLt { .. } | Self::CoreLe { .. } => {
                Err(LinearizeError::Untyped)
            }
            Self::LogicalNot { output, input } if output.ty.is_bool() && input.ty == output.ty => {
                Ok(())
            }
            Self::LogicalAnd { output, lhs, rhs } | Self::LogicalOr { output, lhs, rhs }
                if output.ty.is_bool() && lhs.ty == output.ty && rhs.ty == output.ty =>
            {
                Ok(())
            }
            Self::LogicalNot { .. } | Self::LogicalAnd { .. } | Self::LogicalOr { .. } => {
                Err(LinearizeError::Untyped)
            }
            Self::Compare {
                output, lhs, rhs, ..
            } if output.ty.scalar == DType::Bool
                && output.ty.lanes == lhs.ty.lanes
                && lhs.ty.lanes == rhs.ty.lanes =>
            {
                Ok(())
            }
            Self::Compare { .. } => Err(LinearizeError::Untyped),
            Self::Select {
                output,
                condition,
                on_true,
                on_false,
            } if condition.ty.scalar == DType::Bool
                && condition.ty.lanes == output.ty.lanes
                && on_true.ty == output.ty
                && on_false.ty == output.ty =>
            {
                Ok(())
            }
            Self::Select { .. } => Err(LinearizeError::Untyped),
        }
    }

    pub fn map_operands<S, E>(
        &self,
        mut map: impl FnMut(&R) -> Result<S, E>,
    ) -> Result<LaneInstruction<S>, E> {
        let value = |value: &TypedValue<R>, map: &mut dyn FnMut(&R) -> Result<S, E>| {
            Ok(TypedValue {
                register: map(&value.register)?,
                ty: value.ty,
            })
        };
        let address = |address: &AddressRef<R>, map: &mut dyn FnMut(&R) -> Result<S, E>| {
            Ok(AddressRef {
                register: map(&address.register)?,
                value: address.value.clone(),
            })
        };
        let index = |index: &IndexRef<R>, map: &mut dyn FnMut(&R) -> Result<S, E>| {
            Ok(IndexRef {
                register: map(&index.register)?,
                value: index.value.clone(),
                element: index.element,
            })
        };
        Ok(match self {
            Self::Constant {
                output,
                value: literal,
                vector,
            } => LaneInstruction::Constant {
                output: value(output, &mut map)?,
                value: literal.clone(),
                vector: *vector,
            },
            Self::Address { output } => LaneInstruction::Address {
                output: address(output, &mut map)?,
            },
            Self::Range {
                output,
                bound,
                axis,
            } => LaneInstruction::Range {
                output: value(output, &mut map)?,
                bound: value(bound, &mut map)?,
                axis: *axis,
            },
            Self::Index {
                output,
                address: source,
                offset,
            } => LaneInstruction::Index {
                output: index(output, &mut map)?,
                address: address(source, &mut map)?,
                offset: value(offset, &mut map)?,
            },
            Self::Load {
                output,
                index: source,
            } => LaneInstruction::Load {
                output: value(output, &mut map)?,
                index: index(source, &mut map)?,
            },
            Self::Cast { output, input } => LaneInstruction::Cast {
                output: value(output, &mut map)?,
                input: value(input, &mut map)?,
            },
            Self::Bitcast { output, input } => LaneInstruction::Bitcast {
                output: value(output, &mut map)?,
                input: value(input, &mut map)?,
            },
            Self::CoreUnary { output, input, op } => LaneInstruction::CoreUnary {
                output: value(output, &mut map)?,
                input: value(input, &mut map)?,
                op: *op,
            },
            Self::GraphUnary { output, input, op } => LaneInstruction::GraphUnary {
                output: value(output, &mut map)?,
                input: value(input, &mut map)?,
                op: *op,
            },
            Self::CoreBinary {
                output,
                lhs,
                rhs,
                op,
            } => LaneInstruction::CoreBinary {
                output: value(output, &mut map)?,
                lhs: value(lhs, &mut map)?,
                rhs: value(rhs, &mut map)?,
                op: *op,
            },
            Self::CoreEq { output, lhs, rhs } => LaneInstruction::CoreEq {
                output: value(output, &mut map)?,
                lhs: value(lhs, &mut map)?,
                rhs: value(rhs, &mut map)?,
            },
            Self::CoreLt { output, lhs, rhs } => LaneInstruction::CoreLt {
                output: value(output, &mut map)?,
                lhs: value(lhs, &mut map)?,
                rhs: value(rhs, &mut map)?,
            },
            Self::CoreLe { output, lhs, rhs } => LaneInstruction::CoreLe {
                output: value(output, &mut map)?,
                lhs: value(lhs, &mut map)?,
                rhs: value(rhs, &mut map)?,
            },
            Self::GraphBinary {
                output,
                lhs,
                rhs,
                op,
            } => LaneInstruction::GraphBinary {
                output: value(output, &mut map)?,
                lhs: value(lhs, &mut map)?,
                rhs: value(rhs, &mut map)?,
                op: *op,
            },
            Self::LogicalNot { output, input } => LaneInstruction::LogicalNot {
                output: value(output, &mut map)?,
                input: value(input, &mut map)?,
            },
            Self::LogicalAnd { output, lhs, rhs } => LaneInstruction::LogicalAnd {
                output: value(output, &mut map)?,
                lhs: value(lhs, &mut map)?,
                rhs: value(rhs, &mut map)?,
            },
            Self::LogicalOr { output, lhs, rhs } => LaneInstruction::LogicalOr {
                output: value(output, &mut map)?,
                lhs: value(lhs, &mut map)?,
                rhs: value(rhs, &mut map)?,
            },
            Self::Compare {
                output,
                lhs,
                rhs,
                op,
            } => LaneInstruction::Compare {
                output: value(output, &mut map)?,
                lhs: value(lhs, &mut map)?,
                rhs: value(rhs, &mut map)?,
                op: *op,
            },
            Self::Select {
                output,
                condition,
                on_true,
                on_false,
            } => LaneInstruction::Select {
                output: value(output, &mut map)?,
                condition: value(condition, &mut map)?,
                on_true: value(on_true, &mut map)?,
                on_false: value(on_false, &mut map)?,
            },
            Self::Store {
                index: destination,
                value: source,
            } => LaneInstruction::Store {
                index: index(destination, &mut map)?,
                value: value(source, &mut map)?,
            },
        })
    }
}

pub(crate) fn validate_lane_sequence<R, K>(
    instructions: &[LaneProgramInstruction<R>],
    omitted_definitions: &BTreeSet<K>,
    mut key: impl FnMut(&R) -> K,
    mut binding_is_live: impl FnMut(&R, u32, &LaneDescriptor) -> bool,
    mut output_is_canonical: impl FnMut(u32, &R) -> bool,
) -> Result<(), String>
where
    K: Clone + fmt::Debug + Ord,
{
    let mut previous_index = None;
    let mut defined = BTreeMap::<K, LaneDescriptor>::new();
    for instruction in instructions {
        if previous_index.is_some_and(|previous| previous >= instruction.index) {
            return Err("lane instruction indices are not strictly ordered".into());
        }
        instruction
            .instruction
            .validate()
            .map_err(|error| error.to_string())?;
        let view = instruction.instruction.view();
        for operand in view.inputs.iter().flatten() {
            let register = key(operand.register);
            let expected = operand.descriptor.into_owned();
            match defined.get(&register) {
                Some(actual) if *actual == expected => {
                    if !binding_is_live(operand.register, instruction.index, &expected) {
                        return Err(format!(
                            "lane operand {register:?} has no live compatible binding at instruction {}",
                            instruction.index
                        ));
                    }
                }
                Some(actual) => {
                    return Err(format!(
                        "lane operand {register:?} descriptor mismatch: {actual:?} != {expected:?}"
                    ));
                }
                None if omitted_definitions.contains(&register) => {}
                None => {
                    return Err(format!(
                        "lane operand {register:?} is used before definition"
                    ));
                }
            }
        }
        if let Some(output) = view.output {
            if !output_is_canonical(instruction.index, output.register) {
                return Err(format!(
                    "lane instruction {} defines a noncanonical output",
                    instruction.index
                ));
            }
            let register = key(output.register);
            let descriptor = output.descriptor.into_owned();
            if !binding_is_live(output.register, instruction.index, &descriptor) {
                return Err(format!(
                    "lane output {register:?} has no live compatible binding at instruction {}",
                    instruction.index
                ));
            }
            defined.insert(register, descriptor);
        }
        previous_index = Some(instruction.index);
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LaneProgramInstruction<R> {
    pub index: u32,
    pub instruction: LaneInstruction<R>,
}
/// One source UOp that has no lane-level representation. Disabled plans keep
/// its exact source register and typed operation so validation
/// can distinguish an intentionally omitted producer from a forged reference.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LaneSourceRecord {
    pub index: u32,
    pub operation: Operation,
}
#[derive(Clone, Debug, Eq, PartialEq)]
struct LaneProjection {
    instructions: Vec<LaneProgramInstruction<u32>>,
    source_records: Vec<LaneSourceRecord>,
    unsupported_operations: Vec<LaneSourceRecord>,
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
    pub instructions: Vec<LaneProgramInstruction<u32>>,
    /// Source ordering delimiters retained outside the lane ALU.
    pub control_operations: Vec<LaneSourceRecord>,
    /// Operations retained by the source DAG but intentionally excluded from
    /// the lane ABI. A non-empty list forces the scalar fallback.
    pub unsupported_operations: Vec<LaneSourceRecord>,
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
            .validate()
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
        let projection = lane_instructions(&nodes)?;
        if let Some(operation) = projection.unsupported_operations.first() {
            enabled = false;
            reason = format!("unsupported lane operation {:?}", operation.operation);
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
            projection.instructions,
            projection.source_records,
            projection.unsupported_operations,
            if enabled { lanes as u16 } else { 1 },
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
        if !self.program.unsupported_operations.is_empty() && self.enabled {
            return Err(LinearizeError::Invalid(
                "a lane plan with unsupported operations cannot be enabled".into(),
            ));
        }
        let mut source_nodes = Vec::new();
        producer_order(&self.source, &mut BTreeSet::new(), &mut source_nodes);
        let expected = lane_instructions(&source_nodes)?;
        if self.program.instructions != expected.instructions
            || self.program.control_operations != expected.source_records
            || self.program.unsupported_operations != expected.unsupported_operations
        {
            return Err(LinearizeError::Invalid(
                "lane program does not match its retained source UOps".into(),
            ));
        }
        validate_program(
            &self.program,
            u16::try_from(if self.enabled { self.lanes } else { 1 })
                .map_err(|_| LinearizeError::Overflow)?,
        )?;
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

pub(crate) fn project_lane_instruction(
    node: &UOp,
    output_register: u32,
    mut source_register: impl FnMut(usize, &UOp) -> Result<u32, LinearizeError>,
) -> Result<Option<LaneInstruction<u32>>, LinearizeError> {
    let source =
        |slot: usize,
         source_register: &mut dyn FnMut(usize, &UOp) -> Result<u32, LinearizeError>| {
            let source = node.sources().get(slot).ok_or_else(|| {
                LinearizeError::Invalid(format!("{:?} missing source {slot}", node.operation()))
            })?;
            Ok((source, source_register(slot, source)?))
        };
    let typed_source =
        |slot: usize,
         source_register: &mut dyn FnMut(usize, &UOp) -> Result<u32, LinearizeError>| {
            let (source, register) = source(slot, source_register)?;
            Ok(TypedValue {
                register,
                ty: source.ty().ok_or(LinearizeError::Untyped)?,
            })
        };
    let address_source =
        |slot: usize,
         source_register: &mut dyn FnMut(usize, &UOp) -> Result<u32, LinearizeError>| {
            let (source, register) = source(slot, source_register)?;
            let value = match source.operation() {
                Operation::DefineGlobal(value)
                | Operation::DefineLocal(value)
                | Operation::DefineRegister(value) => value.clone(),
                _ => {
                    return Err(LinearizeError::Invalid(format!(
                        "{:?} source {slot} is not an address",
                        node.operation()
                    )));
                }
            };
            if source.ty() != Some(value.element) {
                return Err(LinearizeError::Untyped);
            }
            Ok(AddressRef { register, value })
        };
    let index_source =
        |slot: usize,
         source_register: &mut dyn FnMut(usize, &UOp) -> Result<u32, LinearizeError>| {
            let (source, register) = source(slot, source_register)?;
            let Operation::Index(value) = source.operation() else {
                return Err(LinearizeError::Invalid(format!(
                    "{:?} source {slot} is not an index",
                    node.operation()
                )));
            };
            Ok(IndexRef {
                register,
                value: value.clone(),
                element: source.ty().ok_or(LinearizeError::Untyped)?,
            })
        };
    let output = || {
        Ok(TypedValue {
            register: output_register,
            ty: node.ty().ok_or(LinearizeError::Untyped)?,
        })
    };
    let instruction = match node.operation() {
        Operation::Const(value) => LaneInstruction::Constant {
            output: output()?,
            value: value.clone(),
            vector: false,
        },
        Operation::VConst(value) => LaneInstruction::Constant {
            output: output()?,
            value: value.clone(),
            vector: true,
        },
        Operation::DefineGlobal(value)
        | Operation::DefineLocal(value)
        | Operation::DefineRegister(value) => {
            if node.ty() != Some(value.element) {
                return Err(LinearizeError::Untyped);
            }
            LaneInstruction::Address {
                output: AddressRef {
                    register: output_register,
                    value: value.clone(),
                },
            }
        }
        Operation::Range(axis) => LaneInstruction::Range {
            output: output()?,
            bound: typed_source(0, &mut source_register)?,
            axis: *axis,
        },
        Operation::Index(value) => LaneInstruction::Index {
            output: IndexRef {
                register: output_register,
                value: value.clone(),
                element: node.ty().ok_or(LinearizeError::Untyped)?,
            },
            address: address_source(0, &mut source_register)?,
            offset: typed_source(1, &mut source_register)?,
        },
        Operation::Load => LaneInstruction::Load {
            output: output()?,
            index: index_source(0, &mut source_register)?,
        },
        Operation::Cast => LaneInstruction::Cast {
            output: output()?,
            input: typed_source(0, &mut source_register)?,
        },
        Operation::Bitcast => LaneInstruction::Bitcast {
            output: output()?,
            input: typed_source(0, &mut source_register)?,
        },
        Operation::Unary(op) => LaneInstruction::CoreUnary {
            output: output()?,
            input: typed_source(0, &mut source_register)?,
            op: *op,
        },
        Operation::GraphUnary(op) => LaneInstruction::GraphUnary {
            output: output()?,
            input: typed_source(0, &mut source_register)?,
            op: *op,
        },
        Operation::Binary(crate::uop::Binary::Eq) => LaneInstruction::CoreEq {
            output: output()?,
            lhs: typed_source(0, &mut source_register)?,
            rhs: typed_source(1, &mut source_register)?,
        },
        Operation::Binary(crate::uop::Binary::Lt) => LaneInstruction::CoreLt {
            output: output()?,
            lhs: typed_source(0, &mut source_register)?,
            rhs: typed_source(1, &mut source_register)?,
        },
        Operation::Binary(crate::uop::Binary::Le) => LaneInstruction::CoreLe {
            output: output()?,
            lhs: typed_source(0, &mut source_register)?,
            rhs: typed_source(1, &mut source_register)?,
        },
        Operation::Binary(op) => LaneInstruction::CoreBinary {
            output: output()?,
            lhs: typed_source(0, &mut source_register)?,
            rhs: typed_source(1, &mut source_register)?,
            op: *op,
        },
        Operation::GraphBinary(op) => LaneInstruction::GraphBinary {
            output: output()?,
            lhs: typed_source(0, &mut source_register)?,
            rhs: typed_source(1, &mut source_register)?,
            op: *op,
        },
        Operation::GraphLogical(crate::LogicalOp::Not) => LaneInstruction::LogicalNot {
            output: output()?,
            input: typed_source(0, &mut source_register)?,
        },
        Operation::GraphLogical(crate::LogicalOp::And) => LaneInstruction::LogicalAnd {
            output: output()?,
            lhs: typed_source(0, &mut source_register)?,
            rhs: typed_source(1, &mut source_register)?,
        },
        Operation::GraphLogical(crate::LogicalOp::Or) => LaneInstruction::LogicalOr {
            output: output()?,
            lhs: typed_source(0, &mut source_register)?,
            rhs: typed_source(1, &mut source_register)?,
        },
        Operation::GraphCompare(op) => LaneInstruction::Compare {
            output: output()?,
            lhs: typed_source(0, &mut source_register)?,
            rhs: typed_source(1, &mut source_register)?,
            op: *op,
        },
        Operation::Ternary(crate::uop::Ternary::Where) => LaneInstruction::Select {
            output: output()?,
            condition: typed_source(0, &mut source_register)?,
            on_true: typed_source(1, &mut source_register)?,
            on_false: typed_source(2, &mut source_register)?,
        },
        Operation::Store => LaneInstruction::Store {
            index: index_source(0, &mut source_register)?,
            value: typed_source(1, &mut source_register)?,
        },
        _ => return Ok(None),
    };
    instruction.validate()?;
    Ok(Some(instruction))
}

fn lane_instructions(nodes: &[UOp]) -> Result<LaneProjection, LinearizeError> {
    let mut ids = BTreeMap::new();
    for (index, node) in nodes.iter().enumerate() {
        ids.entry(format!("{node:?}")).or_insert(index as u32);
    }
    let mut instructions = Vec::with_capacity(nodes.len());
    let mut control_operations = Vec::new();
    let mut unsupported_operations = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let instruction = project_lane_instruction(node, index as u32, |slot, source| {
            ids.get(&format!("{source:?}")).copied().ok_or_else(|| {
                LinearizeError::Invalid(format!(
                    "{:?} source {slot} is unordered",
                    node.operation()
                ))
            })
        })?;
        let Some(instruction) = instruction else {
            if matches!(node.operation(), Operation::EndRange | Operation::Sink) {
                control_operations.push(LaneSourceRecord {
                    index: index as u32,
                    operation: node.operation().clone(),
                });
                continue;
            }
            unsupported_operations.push(LaneSourceRecord {
                index: index as u32,
                operation: node.operation().clone(),
            });
            continue;
        };
        instructions.push(LaneProgramInstruction {
            index: index as u32,
            instruction,
        });
    }
    Ok(LaneProjection {
        instructions,
        source_records: control_operations,
        unsupported_operations,
    })
}

fn linear_program(
    instructions: Vec<LaneProgramInstruction<u32>>,
    control_operations: Vec<LaneSourceRecord>,
    unsupported_operations: Vec<LaneSourceRecord>,
    lanes: u16,
) -> Result<LinearProgram, LinearizeError> {
    let intervals = intervals(&instructions, lanes);
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
        control_operations,
        unsupported_operations,
        intervals,
        assignments,
        peak_scalar,
        peak_vector,
    })
}
fn intervals(instructions: &[LaneProgramInstruction<u32>], lanes: u16) -> Vec<LiveInterval> {
    let mut result = Vec::new();
    for instruction in instructions {
        let view = instruction.instruction.view();
        if let Some(reg) = view.output().copied() {
            let end = instructions
                .iter()
                .filter(|other| other.instruction.view().inputs().any(|input| *input == reg))
                .map(|other| other.index)
                .max()
                .unwrap_or(instruction.index);
            let dtype = view
                .result_type()
                .expect("value instruction output has a type")
                .scalar;
            result.push(LiveInterval {
                virtual_reg: reg,
                class: if lanes > 1 {
                    RegisterClass::Vector
                } else {
                    RegisterClass::Scalar
                },
                dtype: match dtype {
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
fn validate_program(program: &LinearProgram, lanes: u16) -> Result<(), LinearizeError> {
    for instruction in &program.instructions {
        let view = instruction.instruction.view();
        if view
            .typed_inputs()
            .map(|(_, ty)| ty)
            .chain(view.result_type())
            .any(|ty| {
                let effective = if ty.lanes == 1 { lanes } else { ty.lanes };
                effective != lanes
            })
        {
            return Err(LinearizeError::Invalid(format!(
                "linear instruction {} has inconsistent lane width",
                instruction.index
            )));
        }
    }
    let mut previous_control = None;
    for operation in &program.control_operations {
        if !matches!(operation.operation, Operation::EndRange | Operation::Sink)
            || previous_control.is_some_and(|previous| previous >= operation.index)
            || program
                .instructions
                .binary_search_by_key(&operation.index, |instruction| instruction.index)
                .is_ok()
        {
            return Err(LinearizeError::Invalid(
                "lane control operation records are not canonical".into(),
            ));
        }
        previous_control = Some(operation.index);
    }
    let mut previous_unsupported = None;
    for operation in &program.unsupported_operations {
        if previous_unsupported.is_some_and(|previous| previous >= operation.index)
            || program
                .control_operations
                .binary_search_by_key(&operation.index, |operation| operation.index)
                .is_ok()
            || program
                .instructions
                .binary_search_by_key(&operation.index, |instruction| instruction.index)
                .is_ok()
        {
            return Err(LinearizeError::Invalid(
                "unsupported lane operation indices are not canonical".into(),
            ));
        }
        previous_unsupported = Some(operation.index);
    }
    let omitted_definitions = program
        .unsupported_operations
        .iter()
        .map(|operation| operation.index)
        .collect::<BTreeSet<_>>();
    validate_lane_sequence(
        &program.instructions,
        &omitted_definitions,
        |register| *register,
        |_, _, _| true,
        |instruction, output| instruction == *output,
    )
    .map_err(LinearizeError::Invalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AddressSpace, AddressValue, Graph, Shape, UType};

    fn scalar_copy_kernel(space: AddressSpace) -> UOp {
        let f32_type = UType::scalar(DType::F32);
        let i64_type = UType::scalar(DType::I64);
        let address = UOp::from_operation(
            Operation::DefineGlobal(AddressValue {
                space,
                name: "b0".into(),
                element: f32_type,
            }),
            Some(f32_type),
            vec![],
        );
        let range = UOp::from_operation(
            Operation::Range(0),
            Some(i64_type),
            vec![UOp::constant(1, i64_type)],
        );
        let index = UOp::from_operation(
            Operation::Index(IndexValue::Buffer {
                buffer: 0,
                elements: 1,
                input_shape: Shape::from([1]),
                output_shape: Shape::from([1]),
            }),
            Some(f32_type),
            vec![address, range.clone()],
        );
        let store = UOp::from_operation(
            Operation::Store,
            None,
            vec![
                index,
                UOp::scalar_constant(DType::F32, 1.0_f32.to_bits() as u64, f32_type),
            ],
        );
        UOp::sink(vec![
            store,
            UOp::from_operation(Operation::EndRange, None, vec![range]),
        ])
    }

    #[test]
    fn rejects_invalid_source_uop_before_linearization() {
        let valid = scalar_copy_kernel(AddressSpace::Global);
        valid.validate().unwrap();
        LinearKernel::from_uop(&valid).unwrap();

        let invalid = scalar_copy_kernel(AddressSpace::Local);
        assert_eq!(invalid.topological().unwrap().last(), Some(&invalid));
        assert_eq!(invalid.validate(), Err(crate::UOpError::InvalidArgument));
        assert!(matches!(
            LinearKernel::from_uop(&invalid),
            Err(LinearizeError::Invalid(reason)) if reason.contains("InvalidArgument")
        ));
    }

    #[test]
    fn rejects_address_outer_type_that_disagrees_with_its_element_type() {
        let f32_type = UType::scalar(DType::F32);
        let i32_type = UType::scalar(DType::I32);
        let malformed = UOp::from_operation(
            Operation::DefineGlobal(AddressValue {
                space: AddressSpace::Global,
                name: "b0".into(),
                element: f32_type,
            }),
            Some(i32_type),
            vec![],
        );
        // The source validator currently establishes the address space but
        // does not duplicate the lane projection's outer/element type
        // invariant. Linearization must therefore fail closed before two
        // distinct typed addresses can collapse to the same lane program.
        malformed.validate().unwrap();
        assert_eq!(
            lane_instructions(&[malformed]),
            Err(LinearizeError::Untyped)
        );
    }

    #[test]
    fn admits_source_lifted_transcendentals_before_linearization() {
        for op in [
            crate::UnaryOp::Exp2,
            crate::UnaryOp::Log2,
            crate::UnaryOp::Sin,
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", Shape::from([4]), DType::I64);
            let output = graph.unary(op, input).unwrap();
            let source = crate::lower_graph_elementwise(&graph, output).unwrap();
            let unary = source
                .topological()
                .unwrap()
                .into_iter()
                .find(
                    |node| matches!(node.operation(), Operation::GraphUnary(found) if *found == op),
                )
                .unwrap();
            assert_eq!(unary.sources()[0].ty(), Some(UType::scalar(DType::I64)));
            assert_eq!(unary.ty(), Some(UType::scalar(DType::F32)));
            source.validate().unwrap();
            LinearKernel::from_uop(&source).unwrap();
        }
    }

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

    #[test]
    fn lane_instructions_encode_exact_logical_and_core_comparison_arity() {
        let bool_ty = UType::scalar(DType::Bool);
        let int_ty = UType::scalar(DType::I32);
        let lhs = UOp::scalar_constant(DType::Bool, 0, bool_ty);
        let rhs = UOp::scalar_constant(DType::Bool, 1, bool_ty);
        let not = UOp::from_operation(
            Operation::GraphLogical(crate::LogicalOp::Not),
            Some(bool_ty),
            vec![lhs.clone()],
        );
        let and = UOp::from_operation(
            Operation::GraphLogical(crate::LogicalOp::And),
            Some(bool_ty),
            vec![lhs.clone(), rhs.clone()],
        );
        let or = UOp::from_operation(
            Operation::GraphLogical(crate::LogicalOp::Or),
            Some(bool_ty),
            vec![not, and],
        );
        let mut logical_nodes = Vec::new();
        producer_order(&or, &mut BTreeSet::new(), &mut logical_nodes);
        let projection = lane_instructions(&logical_nodes).unwrap();
        assert!(projection.source_records.is_empty());
        assert!(projection.unsupported_operations.is_empty());
        let logical = projection
            .instructions
            .iter()
            .filter(|instruction| {
                matches!(
                    instruction.instruction,
                    LaneInstruction::LogicalNot { .. }
                        | LaneInstruction::LogicalAnd { .. }
                        | LaneInstruction::LogicalOr { .. }
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(logical.len(), 3);
        assert_eq!(logical[0].instruction.view().inputs().count(), 1);
        assert_eq!(logical[1].instruction.view().inputs().count(), 2);
        assert_eq!(logical[2].instruction.view().inputs().count(), 2);

        let a = UOp::constant(1, int_ty);
        let b = UOp::constant(2, int_ty);
        let comparisons = [
            (crate::uop::Binary::Eq, "CoreEq"),
            (crate::uop::Binary::Lt, "CoreLt"),
            (crate::uop::Binary::Le, "CoreLe"),
        ]
        .map(|(op, _)| {
            UOp::from_operation(
                Operation::Binary(op),
                Some(bool_ty),
                vec![a.clone(), b.clone()],
            )
        });
        let mut nodes = vec![a, b];
        nodes.extend(comparisons);
        let projection = lane_instructions(&nodes).unwrap();
        assert!(projection.source_records.is_empty());
        assert!(projection.unsupported_operations.is_empty());
        let instructions = projection.instructions;
        assert!(matches!(
            instructions[2].instruction,
            LaneInstruction::CoreEq { .. }
        ));
        assert!(matches!(
            instructions[3].instruction,
            LaneInstruction::CoreLt { .. }
        ));
        assert!(matches!(
            instructions[4].instruction,
            LaneInstruction::CoreLe { .. }
        ));
        let invalid = LaneInstruction::CoreBinary {
            output: TypedValue {
                register: 2,
                ty: bool_ty,
            },
            lhs: TypedValue {
                register: 0,
                ty: int_ty,
            },
            rhs: TypedValue {
                register: 1,
                ty: int_ty,
            },
            op: crate::uop::Binary::Eq,
        };
        assert_eq!(invalid.validate(), Err(LinearizeError::Untyped));
        let mapped = instructions[2]
            .instruction
            .map_operands(|register| Ok::<_, ()>(*register + 10))
            .unwrap();
        assert!(matches!(
            mapped,
            LaneInstruction::CoreEq {
                output: TypedValue { register: 12, .. },
                lhs: TypedValue { register: 10, .. },
                rhs: TypedValue { register: 11, .. }
            }
        ));

        // Graph ALU owns source promotion at the output; its input storage
        // descriptors may therefore differ even though their lane geometry
        // must agree. Preserve that established UOp/Graph contract rather
        // than imposing core-ALU homogeneous operands here.
        let mixed_binary = LaneInstruction::GraphBinary {
            output: TypedValue {
                register: 2,
                ty: UType::scalar(DType::F32),
            },
            lhs: TypedValue {
                register: 0,
                ty: UType::scalar(DType::I32),
            },
            rhs: TypedValue {
                register: 1,
                ty: UType::scalar(DType::F16),
            },
            op: BinaryOp::Add,
        };
        mixed_binary.validate().unwrap();
        let mixed_compare = LaneInstruction::Compare {
            output: TypedValue {
                register: 3,
                ty: bool_ty,
            },
            lhs: TypedValue {
                register: 0,
                ty: UType::scalar(DType::I32),
            },
            rhs: TypedValue {
                register: 1,
                ty: UType::scalar(DType::F16),
            },
            op: CompareOp::Eq,
        };
        mixed_compare.validate().unwrap();
    }

    #[test]
    fn retained_descriptor_provenance_is_checked_even_for_disabled_programs() {
        let mut graph = Graph::new();
        let input = graph.input("input", Shape::from([2, 3]));
        let output = graph
            .reduce(input, crate::ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        let scheduled = crate::schedule(&graph, output).unwrap();
        let mut linear = LinearKernel::from_uop(&scheduled.items.last().unwrap().kernel).unwrap();
        assert!(!linear.enabled);
        assert!(!linear.program.unsupported_operations.is_empty());
        linear.validate().unwrap();

        let mut forged = linear.clone();
        forged.program.unsupported_operations[0].operation = Operation::Sink;
        assert!(matches!(
            forged.validate(),
            Err(LinearizeError::Invalid(reason)) if reason.contains("retained source")
        ));

        let mut changed = false;
        for instruction in &mut linear.program.instructions {
            let LaneInstruction::Index { address, .. } = &mut instruction.instruction else {
                continue;
            };
            address.value.name.push_str("_mismatch");
            changed = true;
            break;
        }
        assert!(changed);
        assert!(matches!(
            validate_program(&linear.program, 1),
            Err(LinearizeError::Invalid(reason)) if reason.contains("descriptor mismatch")
        ));
        assert!(matches!(
            linear.validate(),
            Err(LinearizeError::Invalid(reason)) if reason.contains("retained source")
        ));
    }
}
