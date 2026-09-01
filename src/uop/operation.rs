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

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum IndexValue {
    Buffer {
        buffer: u64,
        elements: usize,
        input_shape: Shape,
        output_shape: Shape,
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
    pub mean: bool,
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
