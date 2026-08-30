use super::indexing;
use crate::{DType, EinsumPlan, Scalar, Shape, TensorData};
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId(pub(crate) usize);

impl NodeId {
    pub fn index(self) -> usize {
        self.0
    }
    pub(crate) fn from_index(index: usize) -> Self {
        Self(index)
    }
}

/// Selects the first-order derivative input of a functional static update.
/// The payload is deliberately separate from `StaticIndexGrad`: replacement
/// semantics require a final-writer map rather than a scatter accumulation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StaticIndexUpdateWrt {
    Base,
    Value,
}

/// Normalized NCHW convolution parameters. Padding order is top, bottom,
/// left, right; negative padding deliberately is not part of this static API.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Conv2dOptions {
    pub groups: usize,
    pub stride: [usize; 2],
    pub dilation: [usize; 2],
    pub padding: [usize; 4],
}
/// NCHW transpose-convolution geometry. Padding order is top, bottom, left, right.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ConvTranspose2dOptions {
    pub groups: usize,
    pub stride: [usize; 2],
    pub dilation: [usize; 2],
    pub padding: [usize; 4],
    pub output_padding: [usize; 2],
}
/// Normalized NCL transpose-convolution geometry. Padding is `(left, right)`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ConvTranspose1dOptions {
    pub groups: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: [usize; 2],
    pub output_padding: usize,
}
impl Default for ConvTranspose1dOptions {
    fn default() -> Self {
        Self {
            groups: 1,
            stride: 1,
            dilation: 1,
            padding: [0; 2],
            output_padding: 0,
        }
    }
}
impl Default for ConvTranspose2dOptions {
    fn default() -> Self {
        Self {
            groups: 1,
            stride: [1; 2],
            dilation: [1; 2],
            padding: [0; 4],
            output_padding: [0; 2],
        }
    }
}
/// Normalized 2D pooling geometry in top, bottom, left, right padding order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Pool2dOptions {
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub dilation: [usize; 2],
    pub padding: [usize; 4],
    pub ceil_mode: bool,
    pub count_include_pad: bool,
}
impl Default for Pool2dOptions {
    fn default() -> Self {
        Self {
            kernel: [2, 2],
            stride: [2, 2],
            dilation: [1, 1],
            padding: [0; 4],
            ceil_mode: false,
            count_include_pad: true,
        }
    }
}
/// General trailing-spatial pooling geometry. Padding is `(before, after)` per axis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PoolOptions {
    pub kernel: Vec<usize>,
    pub stride: Vec<usize>,
    pub dilation: Vec<usize>,
    pub padding: Vec<(usize, usize)>,
    pub ceil_mode: bool,
    pub count_include_pad: bool,
}
impl From<Pool2dOptions> for PoolOptions {
    fn from(x: Pool2dOptions) -> Self {
        Self {
            kernel: x.kernel.to_vec(),
            stride: x.stride.to_vec(),
            dilation: x.dilation.to_vec(),
            padding: vec![(x.padding[0], x.padding[1]), (x.padding[2], x.padding[3])],
            ceil_mode: x.ceil_mode,
            count_include_pad: x.count_include_pad,
        }
    }
}

/// Options for [`Graph::scaled_dot_product_attention`].
///
/// Training dropout uses `dropout_seed`; that explicit seed keeps graph replay
/// deterministic without tinygrad's process-global RNG state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionOptions {
    pub scale: Option<f64>,
    pub is_causal: bool,
    pub enable_gqa: bool,
    pub dropout_p: f64,
    pub training: bool,
    pub dropout_seed: Option<u64>,
}

/// Concrete Python-style shift argument accepted by public tinygrad
/// `Tensor.roll`: one integer or an ordered tuple of integers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RollShifts {
    Scalar(i64),
    Tuple(Vec<i64>),
}

impl RollShifts {
    pub(crate) fn into_vec(self) -> Vec<i64> {
        match self {
            Self::Scalar(shift) => vec![shift],
            Self::Tuple(shifts) => shifts,
        }
    }
}

/// Concrete Python-style dimension argument accepted by public tinygrad
/// `Tensor.roll`, including its flattening `dims=None` default.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RollDims {
    None,
    Scalar(isize),
    Tuple(Vec<isize>),
}

impl RollDims {
    pub(crate) fn into_option_vec(self) -> Option<Vec<isize>> {
        match self {
            Self::None => None,
            Self::Scalar(axis) => Some(vec![axis]),
            Self::Tuple(axes) => Some(axes),
        }
    }
}

impl Default for AttentionOptions {
    fn default() -> Self {
        Self {
            scale: None,
            is_causal: false,
            enable_gqa: false,
            dropout_p: 0.0,
            training: false,
            dropout_seed: None,
        }
    }
}
impl Default for Conv2dOptions {
    fn default() -> Self {
        Self {
            groups: 1,
            stride: [1, 1],
            dilation: [1, 1],
            padding: [0, 0, 0, 0],
        }
    }
}
impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug)]
pub enum Op {
    Input {
        name: String,
    },
    Constant(TensorData),
    Random {
        kind: RandomKind,
        stream: RandomStream,
    },
    RandomPermutation {
        stream: RandomStream,
    },
    Cast {
        input: NodeId,
        dtype: DType,
    },
    /// Raw storage reinterpretation. Differing item sizes rescale the final
    /// axis while preserving the tensor's total byte extent.
    Bitcast {
        input: NodeId,
        dtype: DType,
    },
    /// Value-preserving boundary which deliberately stops reverse-mode edges.
    Detach {
        input: NodeId,
    },
    /// CPU-static validation boundary that preserves `input` on success.
    TensorGuard {
        input: NodeId,
        axis: usize,
    },
    Unary {
        op: UnaryOp,
        input: NodeId,
    },
    Binary {
        op: BinaryOp,
        lhs: NodeId,
        rhs: NodeId,
    },
    Compare {
        op: CompareOp,
        lhs: NodeId,
        rhs: NodeId,
    },
    Logical {
        op: LogicalOp,
        lhs: NodeId,
        rhs: Option<NodeId>,
    },
    Select {
        condition: NodeId,
        on_true: NodeId,
        on_false: NodeId,
    },
    Reduce {
        input: NodeId,
        kind: ReduceKind,
        axes: Vec<usize>,
        keepdim: bool,
    },
    /// Inclusive, static cumulative operation along one normalized axis.
    PrefixScan {
        input: NodeId,
        axis: usize,
        kind: PrefixScanKind,
        output: PrefixScanOutput,
    },
    /// One selector from a stable static sort pair. Both selectors carry the
    /// same `pair` identity and are scheduled as one values+indices producer.
    Sort {
        input: NodeId,
        axis: usize,
        descending: bool,
        pair: u64,
        output: SortOutput,
    },
    ArgReduce {
        input: NodeId,
        max: bool,
        axis: Option<usize>,
        keepdim: bool,
    },
    ReduceGrad {
        input: NodeId,
        upstream: NodeId,
        kind: ReduceKind,
        axes: Vec<usize>,
        keepdim: bool,
    },
    /// Second-order VJP of the zero-aware/tie-aware reduction reverse node.
    /// `wrt` is 0 for the source input and 1 for the original upstream.
    ReduceGradVjp {
        cotangent: NodeId,
        input: NodeId,
        upstream: NodeId,
        kind: ReduceKind,
        axes: Vec<usize>,
        keepdim: bool,
        wrt: u8,
    },
    SumTo {
        input: NodeId,
        shape: Shape,
    },
    Reshape {
        input: NodeId,
        shape: Shape,
    },
    Permute {
        input: NodeId,
        axes: Vec<usize>,
    },
    Expand {
        input: NodeId,
        shape: Shape,
    },
    Shrink {
        input: NodeId,
        bounds: Vec<(usize, usize)>,
    },
    /// Constant padding. The fill scalar is cast to the input dtype at execution.
    Pad {
        input: NodeId,
        padding: Vec<(usize, usize)>,
        fill: Scalar,
    },
    Stride {
        input: NodeId,
        slices: Vec<Slice>,
    },
    Concat {
        inputs: Vec<NodeId>,
        axis: usize,
    },
    /// Internal reverse-mode primitive: place each input coordinate at
    /// `starts + coordinate * steps`, leaving all other output positions zero.
    ScatterPositions {
        input: NodeId,
        shape: Shape,
        starts: Vec<isize>,
        steps: Vec<isize>,
    },
    /// VJP of `ScatterPositions`: read the cotangent at the same static map.
    ScatterPositionsVjp {
        cotangent: NodeId,
        input_shape: Shape,
        starts: Vec<isize>,
        steps: Vec<isize>,
    },
    Gather {
        input: NodeId,
        index: NodeId,
        axis: usize,
    },
    /// Immutable static mixed indexing, normalized by `ir::indexing`.
    StaticIndex {
        input: NodeId,
        plan: indexing::StaticIndexPlan,
    },
    /// Reverse-mode scatter for [`Op::StaticIndex`].
    StaticIndexGrad {
        cotangent: NodeId,
        input_shape: Shape,
        plan: indexing::StaticIndexPlan,
    },
    /// Immutable snapshot replacement at a normalized static index map.
    StaticIndexUpdate {
        base: NodeId,
        value: NodeId,
        plan: indexing::StaticIndexPlan,
    },
    /// First-order F32 VJP of [`Op::StaticIndexUpdate`].
    StaticIndexUpdateGrad {
        cotangent: NodeId,
        base_shape: Shape,
        value_shape: Shape,
        plan: indexing::StaticIndexPlan,
        wrt: StaticIndexUpdateWrt,
    },
    Scatter {
        base: NodeId,
        index: NodeId,
        updates: NodeId,
        axis: usize,
        add: bool,
    },
    /// Fixed-length mask selection. `size` makes the output shape static;
    /// excess matches are truncated and missing matches receive `fill`.
    MaskedSelect {
        input: NodeId,
        mask: NodeId,
        size: usize,
        fill: Scalar,
    },
    Matmul {
        lhs: NodeId,
        rhs: NodeId,
    },
    /// Static, normalized Einstein summation.  The plan is retained in the IR
    /// so execution and later lowering inspect identical indexing semantics.
    Einsum {
        inputs: Vec<NodeId>,
        plan: EinsumPlan,
        /// Dtype used by tinygrad's elementwise `uprod` before any optional
        /// einsum reduction override is applied.
        product_dtype: DType,
        /// Explicit `Tensor.einsum(dtype=...)` accumulation/output dtype.
        /// `None` retains RustGrad's established default einsum behavior.
        accumulation_dtype: Option<DType>,
    },
    /// Internal reverse-mode scatter-add for a static normalized einsum plan.
    EinsumGrad {
        upstream: NodeId,
        inputs: Vec<NodeId>,
        plan: EinsumPlan,
        target: usize,
    },
    /// VJP of `EinsumGrad`, retaining its normalized plan and scatter map.
    EinsumGradVjp {
        cotangent: NodeId,
        upstream: NodeId,
        inputs: Vec<NodeId>,
        plan: EinsumPlan,
        target: usize,
        wrt: usize,
    },
    /// Internal reverse-mode primitive for generalized matmul.  Keeping the
    /// coordinate mapping in the CPU oracle avoids rank-dependent transpose
    /// graphs and makes broadcast accumulation explicit.
    MatmulGrad {
        upstream: NodeId,
        lhs: NodeId,
        rhs: NodeId,
        lhs_gradient: bool,
    },
    /// VJP of `MatmulGrad` over its generalized dense coordinate map.
    /// `wrt` is 0=upstream, 1=lhs, 2=rhs.
    MatmulGradVjp {
        cotangent: NodeId,
        upstream: NodeId,
        lhs: NodeId,
        rhs: NodeId,
        lhs_gradient: bool,
        wrt: u8,
    },
    Conv2d {
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: Conv2dOptions,
    },
    Conv2dGrad {
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: Conv2dOptions,
        target: u8,
    },
    Conv2dGradVjp {
        cotangent: NodeId,
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: Conv2dOptions,
        target: u8,
        wrt: u8,
    },
    ConvTranspose2d {
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose2dOptions,
    },
    ConvTranspose2dGrad {
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose2dOptions,
        target: u8,
    },
    ConvTranspose2dGradVjp {
        cotangent: NodeId,
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose2dOptions,
        target: u8,
        wrt: u8,
    },
}

/// Stateless seeded distributions used by replayable random graph nodes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RandomKind {
    Uniform { low: f64, high: f64 },
    Normal { mean: f64, std: f64 },
    RandInt { low: i64, high: i64 },
}

// Random distribution parameters are immutable IR data.  Their IEEE bits are
// the semantic identity (rather than a lossy formatted float), which lets
// captured kernels be ordered and keyed deterministically.
impl Eq for RandomKind {}
impl core::hash::Hash for RandomKind {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Uniform { low, high } => {
                low.to_bits().hash(state);
                high.to_bits().hash(state);
            }
            Self::Normal { mean, std } => {
                mean.to_bits().hash(state);
                std.to_bits().hash(state);
            }
            Self::RandInt { low, high } => {
                low.hash(state);
                high.hash(state);
            }
        }
    }
}
impl Ord for RandomKind {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use RandomKind::*;
        match (self, other) {
            (Uniform { low: a, high: b }, Uniform { low: c, high: d }) => {
                (a.to_bits(), b.to_bits()).cmp(&(c.to_bits(), d.to_bits()))
            }
            (Normal { mean: a, std: b }, Normal { mean: c, std: d }) => {
                (a.to_bits(), b.to_bits()).cmp(&(c.to_bits(), d.to_bits()))
            }
            (RandInt { low: a, high: b }, RandInt { low: c, high: d }) => (a, b).cmp(&(c, d)),
            (Uniform { .. }, _) => core::cmp::Ordering::Less,
            (Normal { .. }, Uniform { .. }) => core::cmp::Ordering::Greater,
            (Normal { .. }, RandInt { .. }) => core::cmp::Ordering::Less,
            (RandInt { .. }, _) => core::cmp::Ordering::Greater,
        }
    }
}
impl PartialOrd for RandomKind {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A fully captured Threefry stream reservation. Device identity and counter
/// are part of the typed IR, so CPU realization cannot depend on scheduling.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RandomStream {
    pub device: u32,
    pub key: [u32; 2],
    pub counter: [u32; 2],
}

/// A Python-style per-axis signed slice. Bounds are normalized against each
/// input dimension; `None` selects the direction-appropriate endpoint.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Slice {
    pub start: Option<isize>,
    pub stop: Option<isize>,
    pub step: isize,
}

/// Source-defined forms for [`Graph::split`](super::Graph::split).
///
/// `Uniform` uses a maximum section size and gives the final nonempty output
/// the remaining tail. `Explicit` preserves every ordered section, including
/// zero-sized sections when their checked total covers a zero-sized axis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SplitSections {
    Uniform(usize),
    Explicit(Vec<usize>),
}

impl From<usize> for SplitSections {
    fn from(size: usize) -> Self {
        Self::Uniform(size)
    }
}

impl From<Vec<usize>> for SplitSections {
    fn from(sections: Vec<usize>) -> Self {
        Self::Explicit(sections)
    }
}

impl From<&[usize]> for SplitSections {
    fn from(sections: &[usize]) -> Self {
        Self::Explicit(sections.to_vec())
    }
}

/// A concrete or inferred extent for [`Graph::unflatten`](super::Graph::unflatten).
///
/// `Infer` is the sole source-compatible negative reshape form: exactly one
/// extent may be inferred from the replaced input axis. Arbitrary negative
/// dimensions are intentionally not representable.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum UnflattenExtent {
    Exact(usize),
    Infer,
}

/// A concrete, copied, or inferred extent for [`Graph::reshape_with_extents`](super::Graph::reshape_with_extents).
///
/// This is the Rust representation of tinygrad's public reshape forms:
/// concrete extents, one `-1` inference, and `None` copying the source extent
/// at the same position.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ReshapeExtent {
    Exact(usize),
    Infer,
    Copy,
}

/// A concrete or copied extent for [`Graph::expand_with_extents`](super::Graph::expand_with_extents).
///
/// `Copy` represents tinygrad's public `-1`/`None` expand sentinel after
/// right-aligning the requested shape with the input shape.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExpandExtent {
    Exact(usize),
    Copy,
}

/// One axis of [`Graph::shrink_with_ranges`](super::Graph::shrink_with_ranges).
///
/// `Full` is tinygrad's public `None` axis marker; `Bounds` is its concrete
/// nonnegative half-open `(start, end)` form.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ShrinkRange {
    Full,
    Bounds { start: usize, end: usize },
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ReduceKind {
    Sum,
    Mean,
    Product,
    Max,
    Min,
    /// Boolean disjunction. Its identity is false, including empty domains.
    Any,
    /// Boolean conjunction. Its identity is true, including empty domains.
    All,
}

/// The deliberately small static prefix-scan vocabulary. Each kind retains
/// the input shape and operates along one normalized axis.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PrefixScanKind {
    Sum,
    Product,
    Max,
    Min,
}

/// Selects one static result from a prefix scan. Extrema expose their value
/// and I32 position as separate graph nodes with a shared typed operation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PrefixScanOutput {
    Values,
    Indices,
}

/// Selects one output from the coupled static sort producer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SortOutput {
    Values,
    Indices,
}

/// The explicit accumulator and final-storage dtypes for a reduction.
///
/// The default Sum pair mirrors tinygrad's checked-in `sum_acc_dtype` rule.
/// Narrow floating inputs accumulate in F32 and narrow only at the final
/// result; other supported dtypes retain their source-defined accumulation
/// width. Product defaults to its input dtype for both stages.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ReductionDType {
    pub accumulator: DType,
    pub output: DType,
}

/// Signed Bessel-style correction used by [`Graph::var`](super::Graph::var)
/// and [`Graph::std`](super::Graph::std).
///
/// This is the concrete host representation of tinygrad's `sint` correction
/// argument. The denominator is always formed as `max(n - correction, 0)`;
/// negative corrections are therefore intentionally valid.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VarianceCorrection(i64);

impl VarianceCorrection {
    /// The tinygrad default: sample variance / standard deviation.
    pub const UNBIASED: Self = Self(1);

    pub const fn new(value: i64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> i64 {
        self.0
    }
}

impl ReductionDType {
    pub const fn new(accumulator: DType, output: DType) -> Self {
        Self {
            accumulator,
            output,
        }
    }

    pub const fn sum_default(input: DType) -> Self {
        let accumulator = input.sum_accumulator_dtype();
        let output = match input {
            DType::F8E4M3
            | DType::F8E5M2
            | DType::F8E4M3FNUZ
            | DType::F8E5M2FNUZ
            | DType::F16
            | DType::BF16 => input,
            _ => accumulator,
        };
        Self::new(accumulator, output)
    }

    pub const fn product_default(input: DType) -> Self {
        Self::new(input, input)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum UnaryOp {
    Neg,
    Exp,
    Log,
    Relu,
    Step,
    Abs,
    Reciprocal,
    Square,
    Sqrt,
    Rsqrt,
    Exp2,
    Log2,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Erf,
    Erfc,
    Asin,
    Acos,
    Atan,
    Asinh,
    Acosh,
    Atanh,
    Floor,
    Ceil,
    Trunc,
    Round,
    Sign,
    IsNan,
    IsInf,
    IsFinite,
}

impl UnaryOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Relu => "relu",
            Self::Step => "step",
            Self::Abs => "abs",
            Self::Reciprocal => "reciprocal",
            Self::Square => "square",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::Exp2 => "exp2",
            Self::Log2 => "log2",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Erf => "erf",
            Self::Erfc => "erfc",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Asinh => "asinh",
            Self::Acosh => "acosh",
            Self::Atanh => "atanh",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Trunc => "trunc",
            Self::Round => "round",
            Self::Sign => "sign",
            Self::IsNan => "isnan",
            Self::IsInf => "isinf",
            Self::IsFinite => "isfinite",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum,
    Minimum,
    FloorDiv,
    TruncDiv,
    Mod,
    FMod,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Atan2,
    Copysign,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CompareOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LogicalOp {
    Not,
    And,
    Or,
}

impl LogicalOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Not => "logical_not",
            Self::And => "logical_and",
            Self::Or => "logical_or",
        }
    }
}

impl BinaryOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Pow => "pow",
            Self::Maximum => "maximum",
            Self::Minimum => "minimum",
            Self::FloorDiv => "floor_div",
            Self::TruncDiv => "trunc_div",
            Self::Mod => "mod",
            Self::FMod => "fmod",
            Self::BitAnd => "bitwise_and",
            Self::BitOr => "bitwise_or",
            Self::BitXor => "bitwise_xor",
            Self::Shl => "lshift",
            Self::Shr => "rshift",
            Self::Atan2 => "atan2",
            Self::Copysign => "copysign",
        }
    }
}

impl Op {
    /// Direct value dependencies. This is the authoritative pure-DAG edge
    /// inventory; effect safety analysis uses it without inspecting labels.
    pub(crate) fn value_inputs(&self) -> Vec<NodeId> {
        match self {
            Self::Input { .. }
            | Self::Constant(_)
            | Self::Random { .. }
            | Self::RandomPermutation { .. } => vec![],
            Self::Cast { input, .. }
            | Self::Bitcast { input, .. }
            | Self::Detach { input }
            | Self::TensorGuard { input, .. }
            | Self::Unary { input, .. }
            | Self::Reduce { input, .. }
            | Self::PrefixScan { input, .. }
            | Self::Sort { input, .. }
            | Self::ArgReduce { input, .. }
            | Self::SumTo { input, .. }
            | Self::Reshape { input, .. }
            | Self::Permute { input, .. }
            | Self::Expand { input, .. }
            | Self::Shrink { input, .. }
            | Self::Pad { input, .. }
            | Self::Stride { input, .. }
            | Self::ScatterPositions { input, .. }
            | Self::StaticIndex { input, .. } => vec![*input],
            Self::Binary { lhs, rhs, .. }
            | Self::Compare { lhs, rhs, .. }
            | Self::Matmul { lhs, rhs } => vec![*lhs, *rhs],
            Self::Logical { lhs, rhs, .. } => rhs.iter().copied().chain([*lhs]).collect(),
            Self::Select {
                condition,
                on_true,
                on_false,
            } => vec![*condition, *on_true, *on_false],
            Self::ReduceGrad {
                input, upstream, ..
            } => vec![*input, *upstream],
            Self::ReduceGradVjp {
                cotangent,
                input,
                upstream,
                ..
            } => vec![*cotangent, *input, *upstream],
            Self::Concat { inputs, .. } | Self::Einsum { inputs, .. } => inputs.clone(),
            Self::ScatterPositionsVjp { cotangent, .. }
            | Self::StaticIndexGrad { cotangent, .. }
            | Self::StaticIndexUpdateGrad { cotangent, .. } => vec![*cotangent],
            Self::Gather { input, index, .. } => vec![*input, *index],
            Self::StaticIndexUpdate { base, value, .. } => vec![*base, *value],
            Self::Scatter {
                base,
                index,
                updates,
                ..
            } => vec![*base, *index, *updates],
            Self::MaskedSelect { input, mask, .. } => vec![*input, *mask],
            Self::EinsumGrad {
                upstream, inputs, ..
            } => std::iter::once(*upstream)
                .chain(inputs.iter().copied())
                .collect(),
            Self::EinsumGradVjp {
                cotangent,
                upstream,
                inputs,
                ..
            } => std::iter::once(*cotangent)
                .chain([*upstream])
                .chain(inputs.iter().copied())
                .collect(),
            Self::MatmulGrad {
                upstream, lhs, rhs, ..
            } => vec![*upstream, *lhs, *rhs],
            Self::MatmulGradVjp {
                cotangent,
                upstream,
                lhs,
                rhs,
                ..
            } => vec![*cotangent, *upstream, *lhs, *rhs],
            Self::Conv2d {
                input,
                weight,
                bias,
                ..
            }
            | Self::ConvTranspose2d {
                input,
                weight,
                bias,
                ..
            } => std::iter::once(*input)
                .chain([*weight])
                .chain(bias.iter().copied())
                .collect(),
            Self::Conv2dGrad {
                upstream,
                input,
                weight,
                bias,
                ..
            }
            | Self::ConvTranspose2dGrad {
                upstream,
                input,
                weight,
                bias,
                ..
            } => std::iter::once(*upstream)
                .chain([*input, *weight])
                .chain(bias.iter().copied())
                .collect(),
            Self::Conv2dGradVjp {
                cotangent,
                upstream,
                input,
                weight,
                bias,
                ..
            }
            | Self::ConvTranspose2dGradVjp {
                cotangent,
                upstream,
                input,
                weight,
                bias,
                ..
            } => std::iter::once(*cotangent)
                .chain([*upstream, *input, *weight])
                .chain(bias.iter().copied())
                .collect(),
        }
    }

    /// Direct dependencies whose values may be read by this operation's
    /// reverse rule. Predicate/index/control edges and `Detach` are excluded.
    pub(crate) fn backward_inputs(&self) -> Vec<NodeId> {
        match self {
            Self::Detach { .. }
            | Self::Bitcast { .. }
            | Self::Input { .. }
            | Self::Constant(_)
            | Self::Random { .. }
            | Self::RandomPermutation { .. }
            | Self::ArgReduce { .. }
            | Self::Compare { .. }
            | Self::Logical { .. }
            // Boolean reductions are predicate operations. They never form
            // reverse-mode value edges, even when their source was float.
            | Self::Reduce {
                kind: ReduceKind::Any | ReduceKind::All,
                ..
            } => vec![],
            Self::Select {
                on_true, on_false, ..
            } => vec![*on_true, *on_false],
            Self::Gather { input, .. }
            | Self::StaticIndex { input, .. }
            | Self::MaskedSelect { input, .. } => vec![*input],
            Self::Scatter { base, updates, .. } => vec![*base, *updates],
            _ => self.value_inputs(),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Input { name } => format!("input({name:?})"),
            Self::Constant(_) => "constant".into(),
            Self::Random { kind, stream } => format!(
                "random_{kind:?}(device={}, key={:?}, counter={:?})",
                stream.device, stream.key, stream.counter
            ),
            Self::RandomPermutation { stream } => format!(
                "randperm(device={}, key={:?}, counter={:?})",
                stream.device, stream.key, stream.counter
            ),
            Self::Cast { input, dtype } => format!("cast(%{input}, {dtype:?})"),
            Self::Bitcast { input, dtype } => format!("bitcast(%{input}, {dtype:?})"),
            Self::Detach { input } => format!("detach(%{input})"),
            Self::TensorGuard { input, axis } => {
                format!("tensor_guard(%{input}, axis={axis})")
            }
            Self::Unary { op, input } => format!("{}(%{input})", op.name()),
            Self::Binary { op, lhs, rhs } => format!("{}(%{lhs}, %{rhs})", op.name()),
            Self::Compare { op, lhs, rhs } => format!("{}(%{lhs}, %{rhs})", op.name()),
            Self::Logical {
                op,
                lhs,
                rhs: Some(rhs),
            } => format!("{}(%{lhs}, %{rhs})", op.name()),
            Self::Logical { op, lhs, rhs: None } => format!("{}(%{lhs})", op.name()),
            Self::Select {
                condition,
                on_true,
                on_false,
            } => format!("where(%{condition}, %{on_true}, %{on_false})"),
            Self::Reduce {
                input,
                kind,
                axes,
                keepdim,
            } => format!("{kind:?}(%{input}, axes={axes:?}, keepdim={keepdim})"),
            Self::PrefixScan {
                input,
                axis,
                kind,
                output,
            } => format!(
                "{}(%{input}, axis={axis})",
                match (kind, output) {
                    (PrefixScanKind::Sum, _) => "cumsum",
                    (PrefixScanKind::Product, _) => "cumprod",
                    (PrefixScanKind::Max, PrefixScanOutput::Values) => "cummax",
                    (PrefixScanKind::Min, PrefixScanOutput::Values) => "cummin",
                    (PrefixScanKind::Max, PrefixScanOutput::Indices) => "cummax_indices",
                    (PrefixScanKind::Min, PrefixScanOutput::Indices) => "cummin_indices",
                }
            ),
            Self::ArgReduce {
                input,
                max,
                axis,
                keepdim,
            } => format!(
                "arg{}(%{input}, axis={axis:?}, keepdim={keepdim})",
                if *max { "max" } else { "min" }
            ),
            Self::Sort {
                input,
                axis,
                descending,
                output,
                ..
            } => format!(
                "{}(%{input}, axis={axis}, descending={descending})",
                match output {
                    SortOutput::Values => "sort",
                    SortOutput::Indices => "argsort",
                }
            ),
            Self::ReduceGrad {
                input,
                upstream,
                kind,
                ..
            } => format!("reduce_grad_{kind:?}(%{input}, %{upstream})"),
            Self::ReduceGradVjp { kind, wrt, .. } => format!("reduce_grad_vjp_{kind:?}(wrt={wrt})"),
            Self::SumTo { input, shape } => format!("sum_to(%{input}, {shape})"),
            Self::Reshape { input, shape } => format!("reshape(%{input}, {shape})"),
            Self::Permute { input, axes } => format!("permute(%{input}, {axes:?})"),
            Self::Expand { input, shape } => format!("expand(%{input}, {shape})"),
            Self::Shrink { input, bounds } => format!("shrink(%{input}, {bounds:?})"),
            Self::Pad {
                input,
                padding,
                fill,
            } => format!("pad(%{input}, {padding:?}, {fill:?})"),
            Self::Stride { input, slices } => format!("stride(%{input}, {slices:?})"),
            Self::Concat { inputs, axis } => format!("concat({inputs:?}, axis={axis})"),
            Self::ScatterPositions { input, shape, .. } => {
                format!("scatter_positions(%{input}, {shape})")
            }
            Self::ScatterPositionsVjp {
                cotangent,
                input_shape,
                ..
            } => {
                format!("scatter_positions_vjp(%{cotangent}, {input_shape})")
            }
            Self::Gather { input, index, axis } => {
                format!("gather(%{input}, %{index}, axis={axis})")
            }
            Self::StaticIndex { input, plan } => {
                format!("static_index(%{input}, {:?})", plan.output_shape())
            }
            Self::StaticIndexGrad { cotangent, .. } => {
                format!("static_index_grad(%{cotangent})")
            }
            Self::StaticIndexUpdate { base, value, plan } => {
                format!(
                    "static_index_update(%{base}, %{value}, {:?})",
                    plan.output_shape()
                )
            }
            Self::StaticIndexUpdateGrad { cotangent, wrt, .. } => {
                format!("static_index_update_grad_{wrt:?}(%{cotangent})")
            }
            Self::Scatter {
                base,
                index,
                updates,
                axis,
                add,
            } => format!(
                "scatter_{}(%{base}, %{index}, %{updates}, axis={axis})",
                if *add { "add" } else { "replace" }
            ),
            Self::MaskedSelect {
                input,
                mask,
                size,
                fill,
            } => format!("masked_select(%{input}, %{mask}, size={size}, {fill:?})"),
            Self::Matmul { lhs, rhs } => format!("matmul(%{lhs}, %{rhs})"),
            Self::Einsum {
                inputs,
                plan,
                product_dtype,
                accumulation_dtype,
            } => match accumulation_dtype {
                Some(accumulation_dtype) => format!(
                    "einsum({inputs:?}, output={:?}, contract={:?}, product={}, accumulate={})",
                    plan.output_labels,
                    plan.contracted_labels,
                    product_dtype.stable_name(),
                    accumulation_dtype.stable_name(),
                ),
                None => format!(
                    "einsum({inputs:?}, output={:?}, contract={:?})",
                    plan.output_labels, plan.contracted_labels
                ),
            },
            Self::EinsumGrad {
                upstream, target, ..
            } => format!("einsum_grad(%{upstream}, target={target})"),
            Self::EinsumGradVjp { target, wrt, .. } => {
                format!("einsum_grad_vjp(target={target}, wrt={wrt})")
            }
            Self::MatmulGrad {
                upstream,
                lhs,
                rhs,
                lhs_gradient,
            } => format!(
                "matmul_{}_grad(%{upstream}, %{lhs}, %{rhs})",
                if *lhs_gradient { "lhs" } else { "rhs" }
            ),
            Self::MatmulGradVjp { wrt, .. } => format!("matmul_grad_vjp(wrt={wrt})"),
            Self::Conv2d {
                input,
                weight,
                bias,
                options,
            } => format!(
                "conv2d(%{input}, %{weight}, {bias:?}, groups={}, stride={:?}, dilation={:?}, padding={:?})",
                options.groups, options.stride, options.dilation, options.padding
            ),
            Self::Conv2dGrad { target, .. } => format!("conv2d_grad(target={target})"),
            Self::Conv2dGradVjp { target, wrt, .. } => {
                format!("conv2d_grad_vjp(target={target}, wrt={wrt})")
            }
            Self::ConvTranspose2d {
                input,
                weight,
                bias,
                options,
            } => format!(
                "conv_transpose2d(%{input}, %{weight}, {bias:?}, groups={}, stride={:?}, dilation={:?}, padding={:?}, output_padding={:?})",
                options.groups,
                options.stride,
                options.dilation,
                options.padding,
                options.output_padding
            ),
            Self::ConvTranspose2dGrad { target, .. } => {
                format!("conv_transpose2d_grad(target={target})")
            }
            Self::ConvTranspose2dGradVjp { target, wrt, .. } => {
                format!("conv_transpose2d_grad_vjp(target={target}, wrt={wrt})")
            }
        }
    }
}
