use super::{AddressSpace, AffineView, Binary, Ternary, UType, Unary as CoreUnary};
use crate::{DType, NodeId, Shape, SymbolicExpr};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LiteralValue {
    Int(i64),
    Scalar { dtype: DType, bits: u64 },
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VariableValue {
    pub name: String,
    pub bounds: SymbolicExpr,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AddressValue {
    pub space: AddressSpace,
    pub name: String,
    pub element: UType,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum IndexAddressing {
    Broadcast,
    Projected,
    /// An explicit projected address paired with a Bool validity expression.
    /// Loads return the dtype's canonical zero when validity is false and must
    /// not access the underlying buffer on that lane.
    Predicated,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum IndexValue {
    Buffer {
        buffer: u64,
        elements: usize,
        input_shape: Shape,
        output_shape: Shape,
        addressing: IndexAddressing,
    },
    View {
        buffer: u64,
        elements: usize,
        input_shape: Shape,
        output_shape: Shape,
        view: AffineView,
    },
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ReductionValue {
    pub input_shape: Shape,
    pub output_shape: Shape,
    pub axes: Vec<usize>,
    pub keepdim: bool,
    pub kind: crate::ReduceKind,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MatmulValue {
    Serial(Box<crate::MatmulKernelPlan>),
    Tiled(Box<crate::TiledMatmulPayload>),
    TensorCore(Box<crate::TensorCoreMatmulPayload>),
    Quantized(Box<crate::QuantizedMatmulPlan>),
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MovementValue {
    Plan(Box<crate::MovementKernelPlan>),
    QuantizedRowGather(Box<crate::QuantizedRowGatherPlan>),
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PrefixScanValue {
    pub input: NodeId,
    pub destination: NodeId,
    pub input_shape: Shape,
    pub output_shape: Shape,
    pub axis: usize,
    pub kind: crate::PrefixScanKind,
    pub output: crate::PrefixScanOutput,
    pub input_dtype: DType,
    pub dtype: DType,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SortValue {
    pub input: NodeId,
    pub input_shape: Shape,
    pub axis: usize,
    pub descending: bool,
    pub values: NodeId,
    pub indices: NodeId,
    pub dtype: DType,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TensorGuardValue {
    pub input: NodeId,
    pub input_shape: Shape,
    pub axis: usize,
    pub dtype: DType,
}

/// Complete dependency-bearing live Threefry2x32 semantic. The two graph
/// operands are packed U64 words; shapes are retained so capture and cache
/// identity do not depend on a live graph.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ThreefryValue {
    pub counter: NodeId,
    pub key: NodeId,
    pub counter_shape: Shape,
    pub key_shape: Shape,
    pub output: NodeId,
    pub output_shape: Shape,
}

impl ThreefryValue {
    pub fn validate(&self) -> Result<(), super::UOpError> {
        if self.output == self.counter
            || self.output == self.key
            || (self.counter == self.key && self.counter_shape != self.key_shape)
            || self
                .counter_shape
                .broadcast_with(&self.key_shape)
                .ok()
                .as_ref()
                != Some(&self.output_shape)
        {
            return Err(super::UOpError::InvalidArgument);
        }
        for (_, shape, _) in self.buffer_operands() {
            shape
                .numel()
                .ok()
                .and_then(|elements| elements.checked_mul(DType::U64.itemsize()))
                .ok_or(super::UOpError::InvalidArgument)?;
        }
        Ok(())
    }

    /// Canonical pointer ABI: first semantic use wins and an aliased key does
    /// not create a duplicate pointer. The output is always a distinct final
    /// buffer because graph nodes are append-only.
    pub(crate) fn input_operands(&self) -> impl Iterator<Item = (NodeId, &Shape)> {
        std::iter::once((self.counter, &self.counter_shape))
            .chain((self.key != self.counter).then_some((self.key, &self.key_shape)))
    }

    /// Complete canonical pointer ABI. Inputs retain first-use order and the
    /// distinct mutable output is always last.
    pub(crate) fn buffer_operands(&self) -> impl Iterator<Item = (NodeId, &Shape, bool)> {
        self.input_operands()
            .map(|(node, shape)| (node, shape, false))
            .chain(std::iter::once((self.output, &self.output_shape, true)))
    }
}

/// A closed, typed universal operation. The variant is the semantic identity;
/// payload-bearing operations cannot exist without their matching payload.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Operation {
    Const(LiteralValue),
    VConst(LiteralValue),
    DefineVar(VariableValue),
    DefineGlobal(AddressValue),
    DefineLocal(AddressValue),
    DefineRegister(AddressValue),
    Special(String),
    Range(u32),
    EndRange,
    If,
    EndIf,
    Unary(CoreUnary),
    Binary(Binary),
    /// High-level ALU semantic retained by the portable interpreter.
    GraphUnary(crate::UnaryOp),
    GraphBinary(crate::BinaryOp),
    GraphCompare(crate::CompareOp),
    GraphLogical(crate::LogicalOp),
    /// Live, two-input packed-U64 Threefry2x32 permutation.
    Threefry(ThreefryValue),
    /// Complete static generalized-matmul semantic.
    Matmul(MatmulValue),
    /// Narrow static F32 NCHW 1x1 convolution semantic.
    Conv2d(Box<crate::StaticConv2dPlan>),
    /// Complete materializing concat/gather/scatter semantic and ordered ABI.
    Movement(MovementValue),
    /// Captured random source semantic with an immutable stream reservation.
    Random(Box<crate::random::plan::RandomKernelPlan>),
    /// Static inclusive prefix scan.
    PrefixScan(PrefixScanValue),
    /// Stable CPU-static ordering with values and I32-index outputs.
    Sort(SortValue),
    /// Value-preserving CPU-static distribution validation boundary.
    TensorGuard(TensorGuardValue),
    ReduceInit(ReductionValue),
    ReduceAccumulate,
    ReduceFinalize,
    Ternary(Ternary),
    Cast,
    Bitcast,
    Vectorize,
    Gep(u16),
    Index(IndexValue),
    Load,
    Store,
    /// Immutable graph-adjacent assignment commit; never a pure kernel store.
    EffectStore(Box<crate::EffectPayload>),
    /// Orders an effect store after explicitly named predecessor effect IDs.
    After(Box<crate::EffectPayload>),
    Barrier,
    Sink,
}
