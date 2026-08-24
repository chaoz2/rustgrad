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
        seed: u64,
    },
    Cast {
        input: NodeId,
        dtype: DType,
    },
    /// Value-preserving boundary which deliberately stops reverse-mode edges.
    Detach {
        input: NodeId,
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
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ReduceKind {
    Sum,
    Mean,
    Product,
    Max,
    Min,
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
    pub fn label(&self) -> String {
        match self {
            Self::Input { name } => format!("input({name:?})"),
            Self::Constant(_) => "constant".into(),
            Self::Random { kind, stream } => format!(
                "random_{kind:?}(device={}, key={:?}, counter={:?})",
                stream.device, stream.key, stream.counter
            ),
            Self::RandomPermutation { seed } => format!("randperm(seed={seed})"),
            Self::Cast { input, dtype } => format!("cast(%{input}, {dtype:?})"),
            Self::Detach { input } => format!("detach(%{input})"),
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
            Self::ArgReduce {
                input,
                max,
                axis,
                keepdim,
            } => format!(
                "arg{}(%{input}, axis={axis:?}, keepdim={keepdim})",
                if *max { "max" } else { "min" }
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
            Self::Einsum { inputs, plan } => format!(
                "einsum({inputs:?}, output={:?}, contract={:?})",
                plan.output_labels, plan.contracted_labels
            ),
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
