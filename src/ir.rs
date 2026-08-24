use crate::{
    CompileTrace, DType, EinsumPlan, Error, Result, Scalar, Shape, SymbolicShape, SymbolicVar,
    TensorData, TraceStep,
};
use std::collections::BTreeMap;
use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

mod attention;
mod creation;
pub mod pool;
pub mod rearrange;
mod reduce;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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
        seed: u64,
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
            Self::Random { kind, seed } => format!("random_{kind:?}(seed={seed})"),
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

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub op: Op,
    pub shape: Shape,
    pub dtype: DType,
    pub requires_grad: bool,
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    id: u64,
    pub(crate) grad_enabled: bool,
}

impl Default for Graph {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            grad_enabled: true,
        }
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stable identity used to reject parameters from another graph.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Number of graph nodes currently allocated.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn input(&mut self, name: impl Into<String>, shape: impl Into<Shape>) -> NodeId {
        self.input_dtype(name, shape, DType::F32)
    }

    /// Adds an input after explicitly specializing a symbolic shape.  This is
    /// intentionally a one-way boundary: graph nodes and CPU allocation retain
    /// the existing concrete `Shape` invariant.
    pub fn input_symbolic(
        &mut self,
        name: impl Into<String>,
        shape: &SymbolicShape,
        bindings: &BTreeMap<SymbolicVar, i64>,
    ) -> Result<NodeId> {
        Ok(self.input(name, shape.bind_for_graph(bindings)?))
    }

    pub fn input_dtype(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> NodeId {
        self.input_dtype_requires_grad(name, shape, dtype, dtype.is_float())
    }

    /// Adds an input leaf with an explicit gradient-tracking contract.
    pub fn input_dtype_requires_grad(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        dtype: DType,
        requires_grad: bool,
    ) -> NodeId {
        self.push_with_grad(
            Op::Input { name: name.into() },
            shape.into(),
            dtype,
            requires_grad && dtype.is_float(),
        )
    }

    pub fn constant(&mut self, data: TensorData) -> NodeId {
        let shape = data.shape().clone();
        let dtype = data.dtype();
        self.push_with_grad(Op::Constant(data), shape, dtype, false)
    }

    /// Returns whether future graph operations record reverse-mode edges.
    pub fn grad_enabled(&self) -> bool {
        self.grad_enabled
    }

    /// Runs a graph-building closure with reverse-mode recording disabled.
    /// The guard is stored on this graph only, so it is thread-safe and cannot
    /// leak to another graph instance.
    pub fn no_grad<T>(&mut self, build: impl FnOnce(&mut Self) -> T) -> T {
        let previous = self.grad_enabled;
        self.grad_enabled = false;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| build(self)));
        self.grad_enabled = previous;
        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Creates a value-sharing node that is a new gradient leaf.
    pub fn detach(&mut self, input: NodeId) -> Result<NodeId> {
        let node = self.node(input)?;
        Ok(self.push_with_grad(
            Op::Detach { input },
            node.shape.clone(),
            node.dtype,
            node.dtype.is_float(),
        ))
    }

    /// Returns the explicit gradient-tracking state of a graph node.
    pub fn requires_grad(&self, id: NodeId) -> Result<bool> {
        Ok(self.node(id)?.requires_grad)
    }

    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Add, lhs, rhs)
    }

    pub fn sub(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Sub, lhs, rhs)
    }

    pub fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Mul, lhs, rhs)
    }

    pub fn div(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Div, lhs, rhs)
    }
    pub fn pow(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Pow, lhs, rhs)
    }
    pub fn maximum(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Maximum, lhs, rhs)
    }
    pub fn minimum(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Minimum, lhs, rhs)
    }
    pub fn floor_div(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::FloorDiv, lhs, rhs)
    }
    pub fn trunc_div(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::TruncDiv, lhs, rhs)
    }
    pub fn modulo(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Mod, lhs, rhs)
    }
    pub fn fmod(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::FMod, lhs, rhs)
    }
    pub fn bit_and(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::BitAnd, lhs, rhs)
    }
    pub fn bit_or(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::BitOr, lhs, rhs)
    }
    pub fn bit_xor(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::BitXor, lhs, rhs)
    }
    pub fn shl(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Shl, lhs, rhs)
    }
    pub fn shr(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Shr, lhs, rhs)
    }

    pub fn eq(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.compare(CompareOp::Eq, lhs, rhs)
    }
    pub fn ne(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.compare(CompareOp::Ne, lhs, rhs)
    }
    pub fn lt(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.compare(CompareOp::Lt, lhs, rhs)
    }
    pub fn le(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.compare(CompareOp::Le, lhs, rhs)
    }
    pub fn gt(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.compare(CompareOp::Gt, lhs, rhs)
    }
    pub fn ge(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.compare(CompareOp::Ge, lhs, rhs)
    }

    pub fn logical_not(&mut self, input: NodeId) -> Result<NodeId> {
        self.require_bool(input, "logical_not")?;
        let shape = self.node(input)?.shape.clone();
        Ok(self.push(
            Op::Logical {
                op: LogicalOp::Not,
                lhs: input,
                rhs: None,
            },
            shape,
            DType::Bool,
        ))
    }

    pub fn logical_and(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.logical_binary(LogicalOp::And, lhs, rhs)
    }
    pub fn logical_or(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.logical_binary(LogicalOp::Or, lhs, rhs)
    }

    /// Selects `on_true` where `condition` is true and `on_false` otherwise.
    /// The condition must be boolean; both value branches are promoted.
    pub fn select(
        &mut self,
        condition: NodeId,
        on_true: NodeId,
        on_false: NodeId,
    ) -> Result<NodeId> {
        self.require_bool(condition, "select")?;
        let value_shape = self.broadcast_shape(on_true, on_false)?;
        let shape = self.node(condition)?.shape.broadcast_with(&value_shape)?;
        let dtype = self
            .node(on_true)?
            .dtype
            .promote(self.node(on_false)?.dtype);
        Ok(self.push(
            Op::Select {
                condition,
                on_true,
                on_false,
            },
            shape,
            dtype,
        ))
    }

    pub fn neg(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Neg, input)
    }
    pub fn exp(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Exp, input)
    }
    pub fn log(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Log, input)
    }
    pub fn abs(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Abs, input)
    }
    pub fn reciprocal(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Reciprocal, input)
    }
    pub fn square(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Square, input)
    }
    pub fn sqrt(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Sqrt, input)
    }
    pub fn rsqrt(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Rsqrt, input)
    }
    pub fn exp2(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Exp2, input)
    }
    pub fn log2(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Log2, input)
    }
    pub fn sin(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Sin, input)
    }
    pub fn cos(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Cos, input)
    }
    pub fn tan(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Tan, input)
    }
    pub fn sinh(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Sinh, input)
    }
    pub fn cosh(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Cosh, input)
    }
    pub fn tanh(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Tanh, input)
    }
    /// Applies the Gauss error function elementwise.
    pub fn erf(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Erf, input)
    }
    /// Applies the complementary Gauss error function elementwise.
    pub fn erfc(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Erfc, input)
    }
    pub fn asin(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Asin, input)
    }
    pub fn acos(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Acos, input)
    }
    pub fn atan(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Atan, input)
    }
    pub fn asinh(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Asinh, input)
    }
    pub fn acosh(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Acosh, input)
    }
    pub fn atanh(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Atanh, input)
    }
    /// Returns the quadrant-aware angle of `(y, x)` elementwise.
    pub fn atan2(&mut self, y: NodeId, x: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Atan2, y, x)
    }
    /// Returns the magnitude of `magnitude` with the sign selected by `sign`.
    pub fn copysign(&mut self, magnitude: NodeId, sign: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Copysign, magnitude, sign)
    }
    /// Compositional tinygrad-style sigmoid, retaining an inspectable graph.
    pub fn sigmoid(&mut self, input: NodeId) -> Result<NodeId> {
        let one = self.constant(TensorData::scalar(1.0f32));
        let neg = self.neg(input)?;
        let exp = self.exp(neg)?;
        let denominator = self.add(one, exp)?;
        let numerator = self.constant(TensorData::scalar(1.0f32));
        self.div(numerator, denominator)
    }
    pub fn clamp(
        &mut self,
        input: NodeId,
        min: Option<NodeId>,
        max: Option<NodeId>,
    ) -> Result<NodeId> {
        if min.is_none() && max.is_none() {
            return Err(Error::InvalidElementwiseDType {
                op: "clamp requires a bound",
                actual: self.node(input)?.dtype,
            });
        }
        let mut value = input;
        if let Some(min) = min {
            value = self.maximum(value, min)?;
        }
        if let Some(max) = max {
            value = self.minimum(value, max)?;
        }
        Ok(value)
    }
    pub fn relu6(&mut self, input: NodeId) -> Result<NodeId> {
        let zero = self.constant(TensorData::scalar(0.0f32));
        let six = self.constant(TensorData::scalar(6.0f32));
        self.clamp(input, Some(zero), Some(six))
    }
    pub fn leaky_relu(&mut self, input: NodeId, slope: NodeId) -> Result<NodeId> {
        let zero = self.constant(TensorData::scalar(0.0f32));
        let negative = self.lt(input, zero)?;
        let scaled = self.mul(input, slope)?;
        self.select(negative, scaled, input)
    }
    pub fn silu(&mut self, input: NodeId) -> Result<NodeId> {
        let sigmoid = self.sigmoid(input)?;
        self.mul(input, sigmoid)
    }
    pub fn hardsigmoid(&mut self, input: NodeId) -> Result<NodeId> {
        let three = self.constant(TensorData::scalar(3.0f32));
        let six = self.constant(TensorData::scalar(6.0f32));
        let shifted = self.add(input, three)?;
        let zero = self.constant(TensorData::scalar(0.0f32));
        let clipped = self.clamp(shifted, Some(zero), Some(six))?;
        let divisor = self.constant(TensorData::scalar(6.0f32));
        self.div(clipped, divisor)
    }
    pub fn hardtanh(&mut self, input: NodeId, min: NodeId, max: NodeId) -> Result<NodeId> {
        self.clamp(input, Some(min), Some(max))
    }
    pub fn swish(&mut self, input: NodeId) -> Result<NodeId> {
        self.silu(input)
    }
    pub fn hardswish(&mut self, input: NodeId) -> Result<NodeId> {
        let three = self.constant(TensorData::scalar(3.0f32));
        let six = self.constant(TensorData::scalar(6.0f32));
        let zero = self.constant(TensorData::scalar(0.0f32));
        let shifted = self.add(input, three)?;
        let clipped = self.clamp(shifted, Some(zero), Some(six))?;
        let scaled = self.mul(input, clipped)?;
        let divisor = self.constant(TensorData::scalar(6.0f32));
        self.div(scaled, divisor)
    }
    pub fn quick_gelu(&mut self, input: NodeId) -> Result<NodeId> {
        let scale = self.constant(TensorData::scalar(1.702f32));
        let scaled = self.mul(input, scale)?;
        let sigmoid = self.sigmoid(scaled)?;
        self.mul(input, sigmoid)
    }
    /// Applies GELU using tinygrad's `"tanh"` approximation or the exact
    /// error-function form selected by `"none"`.
    pub fn gelu(&mut self, input: NodeId, approximate: &str) -> Result<NodeId> {
        match approximate {
            "tanh" => {
                let half = self.constant(TensorData::scalar(0.5f32));
                let one = self.constant(TensorData::scalar(1.0f32));
                let scale =
                    self.constant(TensorData::scalar((2.0f32 / std::f32::consts::PI).sqrt()));
                let coefficient = self.constant(TensorData::scalar(0.044_715f32));
                let square = self.square(input)?;
                let cube = self.mul(square, input)?;
                let scaled_cube = self.mul(coefficient, cube)?;
                let inner = self.add(input, scaled_cube)?;
                let scaled = self.mul(scale, inner)?;
                let tanh = self.tanh(scaled)?;
                let left = self.mul(half, input)?;
                let right = self.add(one, tanh)?;
                self.mul(left, right)
            }
            "none" => {
                let half = self.constant(TensorData::scalar(0.5f32));
                let one = self.constant(TensorData::scalar(1.0f32));
                let root_two = self.constant(TensorData::scalar(std::f32::consts::SQRT_2));
                let scaled = self.div(input, root_two)?;
                let erf = self.erf(scaled)?;
                let left = self.mul(input, half)?;
                let right = self.add(one, erf)?;
                self.mul(left, right)
            }
            _ => Err(Error::InvalidElementwiseDType {
                op: "gelu approximate must be `tanh` or `none`",
                actual: self.node(input)?.dtype,
            }),
        }
    }
    pub fn elu(&mut self, input: NodeId, alpha: NodeId) -> Result<NodeId> {
        let zero = self.constant(TensorData::scalar(0.0f32));
        let positive = self.gt(input, zero)?;
        let exp = self.exp(input)?;
        let one = self.constant(TensorData::scalar(1.0f32));
        let delta = self.sub(exp, one)?;
        let negative = self.mul(alpha, delta)?;
        self.select(positive, input, negative)
    }
    pub fn celu(&mut self, input: NodeId, alpha: NodeId) -> Result<NodeId> {
        let zero = self.constant(TensorData::scalar(0.0f32));
        let positive = self.maximum(input, zero)?;
        let scaled = self.div(input, alpha)?;
        let exp = self.exp(scaled)?;
        let one = self.constant(TensorData::scalar(1.0f32));
        let delta = self.sub(exp, one)?;
        let scaled_negative = self.mul(alpha, delta)?;
        let negative = self.minimum(scaled_negative, zero)?;
        self.add(positive, negative)
    }
    pub fn selu(&mut self, input: NodeId, alpha: NodeId, gamma: NodeId) -> Result<NodeId> {
        let elu = self.elu(input, alpha)?;
        self.mul(gamma, elu)
    }
    pub fn softplus(&mut self, input: NodeId, beta: NodeId) -> Result<NodeId> {
        let scaled = self.mul(input, beta)?;
        let exp = self.exp(scaled)?;
        let one = self.constant(TensorData::scalar(1.0f32));
        let sum = self.add(one, exp)?;
        let logged = self.log(sum)?;
        self.div(logged, beta)
    }
    pub fn mish(&mut self, input: NodeId) -> Result<NodeId> {
        let one = self.constant(TensorData::scalar(1.0f32));
        let exp = self.exp(input)?;
        let sum = self.add(one, exp)?;
        let softplus = self.log(sum)?;
        let tanh = self.tanh(softplus)?;
        self.mul(input, tanh)
    }
    pub fn logsigmoid(&mut self, input: NodeId) -> Result<NodeId> {
        let neg = self.neg(input)?;
        let one = self.constant(TensorData::scalar(1.0f32));
        let exp = self.exp(neg)?;
        let sum = self.add(one, exp)?;
        let log = self.log(sum)?;
        self.neg(log)
    }
    pub fn softsign(&mut self, input: NodeId) -> Result<NodeId> {
        let one = self.constant(TensorData::scalar(1.0f32));
        let abs = self.abs(input)?;
        let denominator = self.add(one, abs)?;
        self.div(input, denominator)
    }
    pub fn log10(&mut self, input: NodeId) -> Result<NodeId> {
        let log = self.log2(input)?;
        let scale = self.constant(TensorData::scalar(std::f32::consts::LOG10_2));
        self.mul(log, scale)
    }
    pub fn logaddexp(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let maximum = self.maximum(lhs, rhs)?;
        let left = self.sub(lhs, maximum)?;
        let right = self.sub(rhs, maximum)?;
        let left_exp = self.exp(left)?;
        let right_exp = self.exp(right)?;
        let sum = self.add(left_exp, right_exp)?;
        let log = self.log(sum)?;
        self.add(log, maximum)
    }
    pub fn logaddexp2(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let maximum = self.maximum(lhs, rhs)?;
        let left = self.sub(lhs, maximum)?;
        let right = self.sub(rhs, maximum)?;
        let left_exp = self.exp2(left)?;
        let right_exp = self.exp2(right)?;
        let sum = self.add(left_exp, right_exp)?;
        let log = self.log2(sum)?;
        self.add(log, maximum)
    }
    pub fn lerp(&mut self, start: NodeId, end: NodeId, weight: NodeId) -> Result<NodeId> {
        let delta = self.sub(end, start)?;
        let weighted = self.mul(delta, weight)?;
        self.add(start, weighted)
    }
    pub fn isclose(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        rtol: NodeId,
        atol: NodeId,
        equal_nan: bool,
    ) -> Result<NodeId> {
        let raw_difference = self.sub(lhs, rhs)?;
        let difference = self.abs(raw_difference)?;
        let abs_rhs = self.abs(rhs)?;
        let relative = self.mul(rtol, abs_rhs)?;
        let tolerance = self.add(atol, relative)?;
        let lhs_finite = self.isfinite(lhs)?;
        let rhs_finite = self.isfinite(rhs)?;
        let finite = self.logical_and(lhs_finite, rhs_finite)?;
        let near = self.le(difference, tolerance)?;
        let finite_near = self.logical_and(finite, near)?;
        let lhs_inf = self.isinf(lhs)?;
        let rhs_inf = self.isinf(rhs)?;
        let infinities = self.logical_or(lhs_inf, rhs_inf)?;
        let equal = self.eq(lhs, rhs)?;
        let same_infinity = self.logical_and(infinities, equal)?;
        let result = self.logical_or(finite_near, same_infinity)?;
        if equal_nan {
            let lhs_nan = self.isnan(lhs)?;
            let rhs_nan = self.isnan(rhs)?;
            let both_nan = self.logical_and(lhs_nan, rhs_nan)?;
            self.logical_or(result, both_nan)
        } else {
            Ok(result)
        }
    }
    pub fn floor(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Floor, input)
    }
    pub fn ceil(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Ceil, input)
    }
    pub fn trunc(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Trunc, input)
    }
    pub fn round(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Round, input)
    }
    pub fn sign(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Sign, input)
    }
    pub fn isnan(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::IsNan, input)
    }
    pub fn isinf(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::IsInf, input)
    }
    pub fn isfinite(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::IsFinite, input)
    }
    pub fn relu(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Relu, input)
    }
    pub(crate) fn step(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Step, input)
    }

    pub fn unary(&mut self, op: UnaryOp, input: NodeId) -> Result<NodeId> {
        let source = self.node(input)?;
        let dtype = unary_dtype(op, source.dtype);
        Ok(self.push(Op::Unary { op, input }, source.shape.clone(), dtype))
    }

    pub fn binary(&mut self, op: BinaryOp, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let shape = self.broadcast_shape(lhs, rhs)?;
        let promoted = self.node(lhs)?.dtype.promote(self.node(rhs)?.dtype);
        // As with unary transcendental helpers, atan2 lifts exact storage to
        // the default floating dtype rather than performing integer math.
        let dtype = if op == BinaryOp::Atan2 && !promoted.is_float() {
            DType::F32
        } else {
            promoted
        };
        if matches!(op, BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor) && dtype.is_float() {
            return Err(Error::InvalidElementwiseDType {
                op: op.name(),
                actual: dtype,
            });
        }
        if matches!(op, BinaryOp::Shl | BinaryOp::Shr) && !dtype.is_integer() {
            return Err(Error::InvalidElementwiseDType {
                op: op.name(),
                actual: dtype,
            });
        }
        Ok(self.push(Op::Binary { op, lhs, rhs }, shape, dtype))
    }

    pub fn compare(&mut self, op: CompareOp, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let shape = self.broadcast_shape(lhs, rhs)?;
        Ok(self.push(Op::Compare { op, lhs, rhs }, shape, DType::Bool))
    }

    fn logical_binary(&mut self, op: LogicalOp, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.require_bool(lhs, op.name())?;
        self.require_bool(rhs, op.name())?;
        let shape = self.broadcast_shape(lhs, rhs)?;
        Ok(self.push(
            Op::Logical {
                op,
                lhs,
                rhs: Some(rhs),
            },
            shape,
            DType::Bool,
        ))
    }

    fn require_bool(&self, input: NodeId, op: &'static str) -> Result<()> {
        let actual = self.node(input)?.dtype;
        if actual == DType::Bool {
            Ok(())
        } else {
            Err(Error::InvalidLogicalDType { op, actual })
        }
    }

    pub fn cast(&mut self, input: NodeId, dtype: DType) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        Ok(self.push(Op::Cast { input, dtype }, shape, dtype))
    }

    pub fn sum(&mut self, input: NodeId, axis: usize) -> Result<NodeId> {
        self.reduce(input, ReduceKind::Sum, Some(vec![axis as isize]), false)
    }
    pub fn reduce(
        &mut self,
        input: NodeId,
        kind: ReduceKind,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let axes = normalize_axes(input, source.shape.rank(), axes)?;
        let shape = reduction_shape(&source.shape, &axes, keepdim);
        if matches!(kind, ReduceKind::Max | ReduceKind::Min)
            && has_empty_reduction_domain(&source.shape, &shape, &axes)
        {
            return Err(Error::EmptyReduction {
                op: match kind {
                    ReduceKind::Max => "max",
                    ReduceKind::Min => "min",
                    _ => unreachable!(),
                },
                shape: source.shape.clone(),
                axes,
            });
        }
        let dtype = match kind {
            ReduceKind::Mean if !source.dtype.is_float() => DType::F32,
            ReduceKind::Sum => sum_dtype(source.dtype),
            _ => source.dtype,
        };
        Ok(self.push(
            Op::Reduce {
                input,
                kind,
                axes,
                keepdim,
            },
            shape,
            dtype,
        ))
    }
    pub fn argmax(&mut self, input: NodeId, axis: Option<isize>, keepdim: bool) -> Result<NodeId> {
        self.arg_reduce(input, true, axis, keepdim)
    }
    pub fn argmin(&mut self, input: NodeId, axis: Option<isize>, keepdim: bool) -> Result<NodeId> {
        self.arg_reduce(input, false, axis, keepdim)
    }
    fn arg_reduce(
        &mut self,
        input: NodeId,
        max: bool,
        axis: Option<isize>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let axis = axis
            .map(|a| normalize_axes(input, source.shape.rank(), Some(vec![a])))
            .transpose()?
            .map(|v| v[0]);
        let axes = axis.map_or_else(|| (0..source.shape.rank()).collect(), |a| vec![a]);
        let shape = reduction_shape(&source.shape, &axes, keepdim);
        if has_empty_reduction_domain(&source.shape, &shape, &axes) {
            return Err(Error::EmptyReduction {
                op: if max { "argmax" } else { "argmin" },
                shape: source.shape.clone(),
                axes,
            });
        }
        Ok(self.push(
            Op::ArgReduce {
                input,
                max,
                axis,
                keepdim,
            },
            shape,
            DType::I32,
        ))
    }
    pub(crate) fn reduce_grad(
        &mut self,
        input: NodeId,
        upstream: NodeId,
        kind: ReduceKind,
        axes: Vec<usize>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let dtype = self.node(upstream)?.dtype;
        Ok(self.push(
            Op::ReduceGrad {
                input,
                upstream,
                kind,
                axes,
                keepdim,
            },
            shape,
            dtype,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduce_grad_vjp(
        &mut self,
        cotangent: NodeId,
        input: NodeId,
        upstream: NodeId,
        kind: ReduceKind,
        axes: Vec<usize>,
        keepdim: bool,
        wrt: u8,
    ) -> Result<NodeId> {
        let shape = match wrt {
            0 => self.node(input)?.shape.clone(),
            1 => self.node(upstream)?.shape.clone(),
            _ => return Err(Error::InvalidIndex),
        };
        Ok(self.push(
            Op::ReduceGradVjp {
                cotangent,
                input,
                upstream,
                kind,
                axes,
                keepdim,
                wrt,
            },
            shape,
            self.node(cotangent)?.dtype,
        ))
    }

    pub fn sum_to(&mut self, input: NodeId, shape: impl Into<Shape>) -> Result<NodeId> {
        let source = self.node(input)?;
        let shape = shape.into();
        if shape.broadcast_with(&source.shape).as_ref() != Ok(&source.shape) {
            return Err(Error::InvalidSumTo {
                from: source.shape.clone(),
                to: shape,
            });
        }
        Ok(self.push(
            Op::SumTo {
                input,
                shape: shape.clone(),
            },
            shape,
            source.dtype,
        ))
    }

    pub fn reshape(&mut self, input: NodeId, shape: impl Into<Shape>) -> Result<NodeId> {
        let source = self.node(input)?;
        let shape = shape.into();
        if source.shape.numel()? != shape.numel()? {
            return Err(Error::InvalidReshape {
                from: source.shape.clone(),
                to: shape,
            });
        }
        Ok(self.push(
            Op::Reshape {
                input,
                shape: shape.clone(),
            },
            shape,
            source.dtype,
        ))
    }

    pub fn permute(&mut self, input: NodeId, axes: impl Into<Vec<usize>>) -> Result<NodeId> {
        let source = self.node(input)?;
        let axes = axes.into();
        let mut sorted = axes.clone();
        sorted.sort_unstable();
        if sorted != (0..source.shape.rank()).collect::<Vec<_>>() {
            return Err(Error::InvalidPermutation {
                shape: source.shape.clone(),
                axes,
            });
        }
        let shape = Shape::new(
            axes.iter()
                .map(|axis| source.shape.dims()[*axis])
                .collect::<Vec<_>>(),
        );
        Ok(self.push(Op::Permute { input, axes }, shape, source.dtype))
    }

    pub fn expand(&mut self, input: NodeId, shape: impl Into<Shape>) -> Result<NodeId> {
        let source = self.node(input)?;
        let shape = shape.into();
        if source.shape.broadcast_with(&shape).as_ref() != Ok(&shape) {
            return Err(Error::InvalidExpand {
                from: source.shape.clone(),
                to: shape,
            });
        }
        Ok(self.push(
            Op::Expand {
                input,
                shape: shape.clone(),
            },
            shape,
            source.dtype,
        ))
    }

    /// Takes checked, half-open bounds for every input axis.
    pub fn shrink(
        &mut self,
        input: NodeId,
        bounds: impl Into<Vec<(usize, usize)>>,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let bounds = bounds.into();
        if bounds.len() != source.shape.rank() {
            return Err(Error::InvalidMovementRank {
                op: "shrink",
                expected: source.shape.rank(),
                actual: bounds.len(),
            });
        }
        let mut dims = Vec::with_capacity(bounds.len());
        for (axis, ((start, end), dim)) in bounds.iter().zip(source.shape.dims()).enumerate() {
            if start > end || *end > *dim {
                return Err(Error::InvalidBounds {
                    axis,
                    start: *start,
                    end: *end,
                    dim: *dim,
                });
            }
            dims.push(end - start);
        }
        Ok(self.push(Op::Shrink { input, bounds }, Shape::new(dims), source.dtype))
    }

    /// Pads every axis with `(before, after)`. `fill` is deterministically
    /// converted to the input dtype; padding never changes tensor dtype.
    pub fn pad(
        &mut self,
        input: NodeId,
        padding: impl Into<Vec<(usize, usize)>>,
        fill: Scalar,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let padding = padding.into();
        if padding.len() != source.shape.rank() {
            return Err(Error::InvalidMovementRank {
                op: "pad",
                expected: source.shape.rank(),
                actual: padding.len(),
            });
        }
        let dims = source
            .shape
            .dims()
            .iter()
            .zip(&padding)
            .map(|(dim, (before, after))| {
                dim.checked_add(*before)
                    .and_then(|x| x.checked_add(*after))
                    .ok_or_else(|| Error::ShapeOverflow(source.shape.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self.push(
            Op::Pad {
                input,
                padding,
                fill,
            },
            Shape::new(dims),
            source.dtype,
        ))
    }

    /// Applies Python-style signed slices, including negative steps and flips.
    pub fn stride(&mut self, input: NodeId, slices: impl Into<Vec<Slice>>) -> Result<NodeId> {
        let source = self.node(input)?;
        let slices = slices.into();
        if slices.len() != source.shape.rank() {
            return Err(Error::InvalidMovementRank {
                op: "stride",
                expected: source.shape.rank(),
                actual: slices.len(),
            });
        }
        let dims = slices
            .iter()
            .zip(source.shape.dims())
            .enumerate()
            .map(|(axis, (slice, dim))| {
                normalized_slice(*dim, *slice, axis).map(|(_, _, _, length)| length)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self.push(Op::Stride { input, slices }, Shape::new(dims), source.dtype))
    }

    /// Alias for [`Graph::stride`], emphasizing ordinary slicing semantics.
    pub fn slice(&mut self, input: NodeId, slices: impl Into<Vec<Slice>>) -> Result<NodeId> {
        self.stride(input, slices)
    }

    /// Concatenates at least two equally ranked tensors along `axis`.
    pub fn concat(&mut self, inputs: impl Into<Vec<NodeId>>, axis: usize) -> Result<NodeId> {
        let inputs = inputs.into();
        if inputs.len() < 2 {
            return Err(Error::InvalidConcat {
                axis,
                shapes: inputs
                    .iter()
                    .filter_map(|id| self.node(*id).ok().map(|n| n.shape.clone()))
                    .collect(),
            });
        }
        let first = self.node(inputs[0])?;
        if axis >= first.shape.rank() {
            return Err(Error::InvalidAxis {
                node: inputs[0],
                axis,
                rank: first.shape.rank(),
            });
        }
        let shape = first.shape.clone();
        let mut dtype = first.dtype;
        let mut total = 0usize;
        let shapes = inputs
            .iter()
            .map(|id| self.node(*id).map(|n| n.shape.clone()))
            .collect::<Result<Vec<_>>>()?;
        for (id, node_shape) in inputs.iter().zip(&shapes) {
            let node = self.node(*id)?;
            if node_shape.rank() != shape.rank()
                || node_shape
                    .dims()
                    .iter()
                    .enumerate()
                    .any(|(i, dim)| i != axis && *dim != shape.dims()[i])
            {
                return Err(Error::InvalidConcat { axis, shapes });
            }
            total = total
                .checked_add(node_shape.dims()[axis])
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            dtype = dtype.promote(node.dtype);
        }
        let mut dims = shape.dims().to_vec();
        dims[axis] = total;
        Ok(self.push(Op::Concat { inputs, axis }, Shape::new(dims), dtype))
    }

    pub(crate) fn scatter_positions(
        &mut self,
        input: NodeId,
        shape: Shape,
        starts: Vec<isize>,
        steps: Vec<isize>,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        if starts.len() != shape.rank()
            || steps.len() != shape.rank()
            || source.shape.rank() != shape.rank()
        {
            return Err(Error::InvalidMovementRank {
                op: "scatter",
                expected: shape.rank(),
                actual: starts.len().min(steps.len()).min(source.shape.rank()),
            });
        }
        Ok(self.push(
            Op::ScatterPositions {
                input,
                shape: shape.clone(),
                starts,
                steps,
            },
            shape,
            source.dtype,
        ))
    }

    pub(crate) fn scatter_positions_vjp(
        &mut self,
        cotangent: NodeId,
        input_shape: Shape,
        starts: Vec<isize>,
        steps: Vec<isize>,
    ) -> Result<NodeId> {
        let source = self.node(cotangent)?;
        if starts.len() != input_shape.rank() || steps.len() != input_shape.rank() {
            return Err(Error::InvalidMovementRank {
                op: "scatter vjp",
                expected: input_shape.rank(),
                actual: starts.len().min(steps.len()),
            });
        }
        Ok(self.push(
            Op::ScatterPositionsVjp {
                cotangent,
                input_shape: input_shape.clone(),
                starts,
                steps,
            },
            input_shape,
            source.dtype,
        ))
    }

    /// Takes values from `input` at integer coordinates supplied by `index`.
    /// Index rank matches input rank and every non-axis index dimension must
    /// not exceed the corresponding input dimension. Negative indices are not
    /// accepted, matching tinygrad's gather contract.
    pub fn gather(&mut self, input: NodeId, index: NodeId, axis: usize) -> Result<NodeId> {
        let source = self.node(input)?;
        let index_node = self.node(index)?;
        validate_indexed("gather", source, index_node, axis)?;
        Ok(self.push(
            Op::Gather { input, index, axis },
            index_node.shape.clone(),
            source.dtype,
        ))
    }

    /// Replaces indexed base positions. Duplicate indices are deterministic:
    /// row-major later update coordinates win. Replacement scatter is
    /// deliberately non-differentiable; use [`Graph::scatter_add`] for a
    /// differentiable accumulation operation.
    pub fn scatter(
        &mut self,
        base: NodeId,
        index: NodeId,
        updates: NodeId,
        axis: usize,
    ) -> Result<NodeId> {
        self.indexed_scatter(base, index, updates, axis, false)
    }

    /// Adds updates into indexed base positions. Duplicate coordinates are
    /// accumulated in row-major order and result dtype promotes base/updates.
    pub fn scatter_add(
        &mut self,
        base: NodeId,
        index: NodeId,
        updates: NodeId,
        axis: usize,
    ) -> Result<NodeId> {
        self.indexed_scatter(base, index, updates, axis, true)
    }

    fn indexed_scatter(
        &mut self,
        base: NodeId,
        index: NodeId,
        updates: NodeId,
        axis: usize,
        add: bool,
    ) -> Result<NodeId> {
        let base_node = self.node(base)?;
        let index_node = self.node(index)?;
        let update_node = self.node(updates)?;
        validate_indexed("scatter", base_node, index_node, axis)?;
        if update_node.shape.rank() != index_node.shape.rank()
            || update_node
                .shape
                .dims()
                .iter()
                .zip(index_node.shape.dims())
                .any(|(update, index)| update < index)
        {
            return Err(Error::InvalidUpdateShape {
                index: index_node.shape.clone(),
                updates: update_node.shape.clone(),
            });
        }
        Ok(self.push(
            Op::Scatter {
                base,
                index,
                updates,
                axis,
                add,
            },
            base_node.shape.clone(),
            base_node.dtype.promote(update_node.dtype),
        ))
    }

    /// Fixed-shape form of tinygrad's `masked_select(size=N)`. The mask must
    /// be bool and broadcastable to input; matches use row-major order.
    pub fn masked_select(
        &mut self,
        input: NodeId,
        mask: NodeId,
        size: usize,
        fill: Scalar,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let mask_node = self.node(mask)?;
        if mask_node.dtype != DType::Bool {
            return Err(Error::InvalidLogicalDType {
                op: "masked_select",
                actual: mask_node.dtype,
            });
        }
        if mask_node.shape.broadcast_with(&source.shape).as_ref() != Ok(&source.shape) {
            return Err(Error::InvalidIndexedShape {
                op: "masked_select",
                input: source.shape.clone(),
                index: mask_node.shape.clone(),
            });
        }
        Ok(self.push(
            Op::MaskedSelect {
                input,
                mask,
                size,
                fill,
            },
            Shape::from([size]),
            source.dtype,
        ))
    }

    pub fn matmul(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let lhs_shape = &self.node(lhs)?.shape;
        let rhs_shape = &self.node(rhs)?.shape;
        let Some(shape) = matmul_shape(lhs_shape, rhs_shape) else {
            return Err(Error::InvalidMatmul {
                lhs: lhs_shape.clone(),
                rhs: rhs_shape.clone(),
            });
        };
        let dtype = self.node(lhs)?.dtype.promote(self.node(rhs)?.dtype);
        Ok(self.push(Op::Matmul { lhs, rhs }, shape, dtype))
    }

    /// Adds a static dense Einstein summation node with NumPy/tinygrad-style
    /// subscript grammar, including ellipses and repeated-label diagonals.
    pub fn einsum(&mut self, equation: &str, inputs: &[NodeId]) -> Result<NodeId> {
        let shapes = inputs
            .iter()
            .map(|id| Ok(self.node(*id)?.shape.clone()))
            .collect::<Result<Vec<_>>>()?;
        let plan = EinsumPlan::parse(equation, &shapes)?;
        let dtype = inputs.iter().try_fold(DType::Bool, |dtype, id| {
            Ok::<_, Error>(dtype.promote(self.node(*id)?.dtype))
        })?;
        Ok(self.push(
            Op::Einsum {
                inputs: inputs.to_vec(),
                plan: plan.clone(),
            },
            plan.output_shape(),
            dtype,
        ))
    }

    pub(crate) fn einsum_grad(
        &mut self,
        upstream: NodeId,
        inputs: &[NodeId],
        plan: EinsumPlan,
        target: usize,
    ) -> Result<NodeId> {
        let target_id = *inputs.get(target).ok_or(Error::InvalidIndex)?;
        let target_node = self.node(target_id)?;
        let output_shape = plan.output_shape();
        if self.node(upstream)?.shape != output_shape {
            return Err(Error::ShapeMismatch {
                op: "einsum gradient",
                lhs: self.node(upstream)?.shape.clone(),
                rhs: output_shape,
            });
        }
        if !target_node.dtype.is_float() {
            return Err(Error::NonDifferentiableIndexing(
                "einsum gradients require floating point target tensors",
            ));
        }
        Ok(self.push(
            Op::EinsumGrad {
                upstream,
                inputs: inputs.to_vec(),
                plan,
                target,
            },
            target_node.shape.clone(),
            target_node.dtype,
        ))
    }

    pub(crate) fn einsum_grad_vjp(
        &mut self,
        cotangent: NodeId,
        upstream: NodeId,
        inputs: &[NodeId],
        plan: EinsumPlan,
        target: usize,
        wrt: usize,
    ) -> Result<NodeId> {
        let output = if wrt == inputs.len() {
            plan.output_shape()
        } else {
            self.node(*inputs.get(wrt).ok_or(Error::InvalidIndex)?)?
                .shape
                .clone()
        };
        Ok(self.push(
            Op::EinsumGradVjp {
                cotangent,
                upstream,
                inputs: inputs.to_vec(),
                plan,
                target,
                wrt,
            },
            output,
            self.node(cotangent)?.dtype,
        ))
    }

    /// Adds a first-class NCHW/OIHW 2D convolution node.
    pub fn conv2d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: Conv2dOptions,
    ) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let weight_node = self.node(weight)?;
        let shape = conv2d_shape(&input_node.shape, &weight_node.shape, options)?;
        if let Some(bias) = bias {
            let b = self.node(bias)?;
            if b.shape != Shape::from([weight_node.shape.dims()[0]]) {
                return Err(Error::InvalidConv2d {
                    input: input_node.shape.clone(),
                    weight: weight_node.shape.clone(),
                    reason: "bias must be [output_channels]",
                });
            }
        }
        let mut dtype = input_node.dtype.promote(weight_node.dtype);
        if let Some(bias) = bias {
            dtype = dtype.promote(self.node(bias)?.dtype);
        }
        Ok(self.push(
            Op::Conv2d {
                input,
                weight,
                bias,
                options,
            },
            shape,
            dtype,
        ))
    }
    pub fn conv_transpose2d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose2dOptions,
    ) -> Result<NodeId> {
        let x = self.node(input)?;
        let w = self.node(weight)?;
        let shape = conv_transpose2d_shape(&x.shape, &w.shape, options)?;
        if let Some(b) = bias
            && self.node(b)?.shape != Shape::from([w.shape.dims()[1] * options.groups])
        {
            return Err(Error::InvalidConv2d {
                input: x.shape.clone(),
                weight: w.shape.clone(),
                reason: "bias must be [output_channels]",
            });
        }
        let mut dtype = x.dtype.promote(w.dtype);
        if let Some(b) = bias {
            dtype = dtype.promote(self.node(b)?.dtype);
        }
        Ok(self.push(
            Op::ConvTranspose2d {
                input,
                weight,
                bias,
                options,
            },
            shape,
            dtype,
        ))
    }
    /// Lowers NCL/IOK transpose convolution through the singleton-height 2D core.
    pub fn conv_transpose1d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose1dOptions,
    ) -> Result<NodeId> {
        let x = self.node(input)?.shape.clone();
        let w = self.node(weight)?.shape.clone();
        if x.rank() != 3
            || w.rank() != 3
            || options.stride == 0
            || options.dilation == 0
            || options.output_padding >= options.stride
        {
            return Err(Error::InvalidConv2d {
                input: x.clone(),
                weight: w.clone(),
                reason: "invalid 1d transpose convolution geometry",
            });
        }
        let x4 = self.reshape(
            input,
            Shape::new([x.dims()[0], x.dims()[1], 1, x.dims()[2]]),
        )?;
        let w4 = self.reshape(
            weight,
            Shape::new([w.dims()[0], w.dims()[1], 1, w.dims()[2]]),
        )?;
        let y4 = self.conv_transpose2d(
            x4,
            w4,
            bias,
            ConvTranspose2dOptions {
                groups: options.groups,
                stride: [1, options.stride],
                dilation: [1, options.dilation],
                padding: [0, 0, options.padding[0], options.padding[1]],
                output_padding: [0, options.output_padding],
            },
        )?;
        let y = self.node(y4)?.shape.clone();
        self.reshape(y4, Shape::new([y.dims()[0], y.dims()[1], y.dims()[3]]))
    }
    pub(crate) fn conv_transpose2d_grad(
        &mut self,
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose2dOptions,
        target: u8,
    ) -> Result<NodeId> {
        let output =
            conv_transpose2d_shape(&self.node(input)?.shape, &self.node(weight)?.shape, options)?;
        if self.node(upstream)?.shape != output {
            return Err(Error::ShapeMismatch {
                op: "conv_transpose2d gradient",
                lhs: self.node(upstream)?.shape.clone(),
                rhs: output,
            });
        }
        let node = match target {
            0 => input,
            1 => weight,
            2 => bias.ok_or(Error::NonDifferentiableIndexing("missing transpose bias"))?,
            _ => return Err(Error::InvalidIndex),
        };
        let n = self.node(node)?;
        if !n.dtype.is_float() {
            return Err(Error::NonDifferentiableIndexing(
                "transpose convolution gradients require floating point tensors",
            ));
        }
        Ok(self.push(
            Op::ConvTranspose2dGrad {
                upstream,
                input,
                weight,
                bias,
                options,
                target,
            },
            n.shape.clone(),
            n.dtype,
        ))
    }

    pub(crate) fn conv2d_grad(
        &mut self,
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: Conv2dOptions,
        target: u8,
    ) -> Result<NodeId> {
        let output = conv2d_shape(&self.node(input)?.shape, &self.node(weight)?.shape, options)?;
        if self.node(upstream)?.shape != output {
            return Err(Error::ShapeMismatch {
                op: "conv2d gradient",
                lhs: self.node(upstream)?.shape.clone(),
                rhs: output,
            });
        }
        let target_node = match target {
            0 => input,
            1 => weight,
            2 => bias.ok_or(Error::NonDifferentiableIndexing("missing conv2d bias"))?,
            _ => return Err(Error::InvalidIndex),
        };
        let target_data = self.node(target_node)?;
        if !target_data.dtype.is_float() {
            return Err(Error::NonDifferentiableIndexing(
                "conv2d gradients require floating point tensors",
            ));
        }
        Ok(self.push(
            Op::Conv2dGrad {
                upstream,
                input,
                weight,
                bias,
                options,
                target,
            },
            target_data.shape.clone(),
            target_data.dtype,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv2d_grad_vjp(
        &mut self,
        cotangent: NodeId,
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: Conv2dOptions,
        target: u8,
        wrt: u8,
    ) -> Result<NodeId> {
        let shape = conv_vjp_shape(self, upstream, input, weight, bias, wrt)?;
        Ok(self.push(
            Op::Conv2dGradVjp {
                cotangent,
                upstream,
                input,
                weight,
                bias,
                options,
                target,
                wrt,
            },
            shape,
            self.node(cotangent)?.dtype,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv_transpose2d_grad_vjp(
        &mut self,
        cotangent: NodeId,
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose2dOptions,
        target: u8,
        wrt: u8,
    ) -> Result<NodeId> {
        let shape = conv_vjp_shape(self, upstream, input, weight, bias, wrt)?;
        Ok(self.push(
            Op::ConvTranspose2dGradVjp {
                cotangent,
                upstream,
                input,
                weight,
                bias,
                options,
                target,
                wrt,
            },
            shape,
            self.node(cotangent)?.dtype,
        ))
    }

    pub(crate) fn matmul_grad(
        &mut self,
        upstream: NodeId,
        lhs: NodeId,
        rhs: NodeId,
        lhs_gradient: bool,
    ) -> Result<NodeId> {
        let lhs_shape = self.node(lhs)?.shape.clone();
        let rhs_shape = self.node(rhs)?.shape.clone();
        let output = matmul_shape(&lhs_shape, &rhs_shape).ok_or(Error::InvalidMatmul {
            lhs: lhs_shape,
            rhs: rhs_shape,
        })?;
        if self.node(upstream)?.shape != output {
            return Err(Error::ShapeMismatch {
                op: "matmul gradient",
                lhs: self.node(upstream)?.shape.clone(),
                rhs: output,
            });
        }
        let target = if lhs_gradient { lhs } else { rhs };
        let shape = self.node(target)?.shape.clone();
        let dtype = self.node(target)?.dtype;
        Ok(self.push(
            Op::MatmulGrad {
                upstream,
                lhs,
                rhs,
                lhs_gradient,
            },
            shape,
            dtype,
        ))
    }

    pub(crate) fn matmul_grad_vjp(
        &mut self,
        cotangent: NodeId,
        upstream: NodeId,
        lhs: NodeId,
        rhs: NodeId,
        lhs_gradient: bool,
        wrt: u8,
    ) -> Result<NodeId> {
        let output = match wrt {
            0 => matmul_shape(&self.node(lhs)?.shape, &self.node(rhs)?.shape).ok_or_else(|| {
                Error::InvalidMatmul {
                    lhs: self.node(lhs).unwrap().shape.clone(),
                    rhs: self.node(rhs).unwrap().shape.clone(),
                }
            })?,
            1 => self.node(lhs)?.shape.clone(),
            2 => self.node(rhs)?.shape.clone(),
            _ => return Err(Error::InvalidIndex),
        };
        Ok(self.push(
            Op::MatmulGradVjp {
                cotangent,
                upstream,
                lhs,
                rhs,
                lhs_gradient,
                wrt,
            },
            output,
            self.node(cotangent)?.dtype,
        ))
    }

    pub fn shape(&self, id: NodeId) -> Result<&Shape> {
        Ok(&self.node(id)?.shape)
    }

    pub fn dtype(&self, id: NodeId) -> Result<DType> {
        Ok(self.node(id)?.dtype)
    }

    /// Returns the typed operation for inspection without exposing graph
    /// storage internals.
    pub fn op(&self, id: NodeId) -> Result<&Op> {
        Ok(&self.node(id)?.op)
    }

    pub fn trace(&self, output: NodeId) -> Result<CompileTrace> {
        self.node(output)?;
        let steps = self.nodes[..=output.index()]
            .iter()
            .enumerate()
            .map(|(id, node)| TraceStep {
                node: NodeId(id),
                operation: node.op.label(),
                shape: node.shape.clone(),
                dtype: node.dtype,
            })
            .collect();
        Ok(CompileTrace { output, steps })
    }

    pub(crate) fn push(&mut self, op: Op, shape: Shape, dtype: DType) -> NodeId {
        let requires_grad =
            self.grad_enabled && dtype.is_float() && self.op_inputs_require_grad(&op);
        self.push_with_grad(op, shape, dtype, requires_grad)
    }

    fn push_with_grad(
        &mut self,
        op: Op,
        shape: Shape,
        dtype: DType,
        requires_grad: bool,
    ) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            op,
            shape,
            dtype,
            requires_grad,
        });
        id
    }

    fn op_inputs_require_grad(&self, op: &Op) -> bool {
        let mut tracked = |id: NodeId| {
            self.nodes
                .get(id.index())
                .is_some_and(|node| node.requires_grad)
        };
        match op {
            Op::Input { .. }
            | Op::Constant(_)
            | Op::Random { .. }
            | Op::RandomPermutation { .. } => false,
            Op::Cast { input, .. }
            | Op::Unary { input, .. }
            | Op::Reduce { input, .. }
            | Op::ArgReduce { input, .. }
            | Op::SumTo { input, .. }
            | Op::Reshape { input, .. }
            | Op::Permute { input, .. }
            | Op::Expand { input, .. }
            | Op::Shrink { input, .. }
            | Op::Pad { input, .. }
            | Op::Stride { input, .. }
            | Op::ScatterPositions { input, .. }
            | Op::Gather { input, .. }
            | Op::MaskedSelect { input, .. } => tracked(*input),
            Op::ScatterPositionsVjp { cotangent, .. } => tracked(*cotangent),
            Op::Detach { .. } => false,
            Op::Binary { lhs, rhs, .. } | Op::Compare { lhs, rhs, .. } => {
                tracked(*lhs) || tracked(*rhs)
            }
            Op::Logical { lhs, rhs, .. } => tracked(*lhs) || rhs.is_some_and(tracked),
            Op::Select {
                on_true, on_false, ..
            } => tracked(*on_true) || tracked(*on_false),
            Op::ReduceGrad {
                input, upstream, ..
            } => tracked(*input) || tracked(*upstream),
            Op::ReduceGradVjp {
                cotangent,
                input,
                upstream,
                ..
            } => tracked(*cotangent) || tracked(*input) || tracked(*upstream),
            Op::Concat { inputs, .. } | Op::Einsum { inputs, .. } => {
                inputs.iter().copied().any(&mut tracked)
            }
            Op::Scatter { base, updates, .. } => tracked(*base) || tracked(*updates),
            Op::Matmul { lhs, rhs } => tracked(*lhs) || tracked(*rhs),
            Op::EinsumGrad {
                upstream, inputs, ..
            } => tracked(*upstream) || inputs.iter().copied().any(&mut tracked),
            Op::EinsumGradVjp {
                cotangent,
                upstream,
                inputs,
                ..
            } => {
                tracked(*cotangent)
                    || tracked(*upstream)
                    || inputs.iter().copied().any(&mut tracked)
            }
            Op::MatmulGrad {
                upstream, lhs, rhs, ..
            } => tracked(*upstream) || tracked(*lhs) || tracked(*rhs),
            Op::MatmulGradVjp {
                cotangent,
                upstream,
                lhs,
                rhs,
                ..
            } => tracked(*cotangent) || tracked(*upstream) || tracked(*lhs) || tracked(*rhs),
            Op::Conv2d {
                input,
                weight,
                bias,
                ..
            }
            | Op::ConvTranspose2d {
                input,
                weight,
                bias,
                ..
            } => tracked(*input) || tracked(*weight) || bias.is_some_and(tracked),
            Op::Conv2dGrad {
                upstream,
                input,
                weight,
                bias,
                ..
            }
            | Op::ConvTranspose2dGrad {
                upstream,
                input,
                weight,
                bias,
                ..
            } => {
                tracked(*upstream)
                    || tracked(*input)
                    || tracked(*weight)
                    || bias.is_some_and(tracked)
            }
            Op::Conv2dGradVjp {
                cotangent,
                upstream,
                input,
                weight,
                bias,
                ..
            }
            | Op::ConvTranspose2dGradVjp {
                cotangent,
                upstream,
                input,
                weight,
                bias,
                ..
            } => {
                tracked(*cotangent)
                    || tracked(*upstream)
                    || tracked(*input)
                    || tracked(*weight)
                    || bias.is_some_and(tracked)
            }
        }
    }

    pub(crate) fn node(&self, id: NodeId) -> Result<&Node> {
        self.nodes.get(id.index()).ok_or(Error::UnknownNode(id))
    }

    fn broadcast_shape(&self, lhs: NodeId, rhs: NodeId) -> Result<Shape> {
        let lhs = &self.node(lhs)?.shape;
        let rhs = &self.node(rhs)?.shape;
        lhs.broadcast_with(rhs)
    }
}

fn conv_vjp_shape(
    graph: &Graph,
    upstream: NodeId,
    input: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
    wrt: u8,
) -> Result<Shape> {
    match wrt {
        0 => Ok(graph.node(upstream)?.shape.clone()),
        1 => Ok(graph.node(input)?.shape.clone()),
        2 => Ok(graph.node(weight)?.shape.clone()),
        3 => Ok(graph.node(bias.ok_or(Error::InvalidIndex)?)?.shape.clone()),
        _ => Err(Error::InvalidIndex),
    }
}

/// The scalar dtype contract for unary ALU operations.
///
/// Tinygrad's public transcendental helpers lift non-floats to the default
/// float. RustGrad has no configurable default dtype, so that type is F32.
/// Narrow floats retain their storage dtype and are quantized at the CPU
/// result boundary. Predicates always produce bool; discrete operations retain
/// their input dtype so integer paths never travel through floating point.
fn unary_dtype(op: UnaryOp, input: DType) -> DType {
    if matches!(op, UnaryOp::IsNan | UnaryOp::IsInf | UnaryOp::IsFinite) {
        return DType::Bool;
    }
    if matches!(
        op,
        UnaryOp::Exp
            | UnaryOp::Log
            | UnaryOp::Reciprocal
            | UnaryOp::Sqrt
            | UnaryOp::Rsqrt
            | UnaryOp::Exp2
            | UnaryOp::Log2
            | UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Tan
            | UnaryOp::Sinh
            | UnaryOp::Cosh
            | UnaryOp::Tanh
            | UnaryOp::Erf
            | UnaryOp::Erfc
            | UnaryOp::Asin
            | UnaryOp::Acos
            | UnaryOp::Atan
            | UnaryOp::Asinh
            | UnaryOp::Acosh
            | UnaryOp::Atanh
    ) && !input.is_float()
    {
        DType::F32
    } else {
        input
    }
}

/// Infers NumPy-style matmul shape.  Vectors are temporarily treated as a
/// leading (lhs) or trailing (rhs) matrix axis, then that artificial axis is
/// removed from the result.  All preceding axes broadcast normally.
pub(crate) fn matmul_shape(lhs: &Shape, rhs: &Shape) -> Option<Shape> {
    if lhs.rank() == 0 || rhs.rank() == 0 {
        return None;
    }
    let lhs_dims = lhs.dims();
    let rhs_dims = rhs.dims();
    let lhs_vector = lhs.rank() == 1;
    let rhs_vector = rhs.rank() == 1;
    let k_lhs = *lhs_dims.last()?;
    let k_rhs = if rhs_vector {
        rhs_dims[0]
    } else {
        rhs_dims[rhs.rank() - 2]
    };
    if k_lhs != k_rhs {
        return None;
    }
    let lhs_batch = if lhs_vector {
        &[][..]
    } else {
        &lhs_dims[..lhs.rank() - 2]
    };
    let rhs_batch = if rhs_vector {
        &[][..]
    } else {
        &rhs_dims[..rhs.rank() - 2]
    };
    let rank = lhs_batch.len().max(rhs_batch.len());
    let mut result = Vec::with_capacity(rank + 2);
    for axis in 0..rank {
        let lhs_axis = axis
            .checked_sub(rank - lhs_batch.len())
            .and_then(|i| lhs_batch.get(i))
            .copied()
            .unwrap_or(1);
        let rhs_axis = axis
            .checked_sub(rank - rhs_batch.len())
            .and_then(|i| rhs_batch.get(i))
            .copied()
            .unwrap_or(1);
        if lhs_axis != rhs_axis && lhs_axis != 1 && rhs_axis != 1 {
            return None;
        }
        result.push(lhs_axis.max(rhs_axis));
    }
    if !lhs_vector {
        result.push(lhs_dims[lhs.rank() - 2]);
    }
    if !rhs_vector {
        result.push(rhs_dims[rhs.rank() - 1]);
    }
    Some(Shape::new(result))
}

pub(crate) fn conv_transpose2d_shape(
    input: &Shape,
    weight: &Shape,
    options: ConvTranspose2dOptions,
) -> Result<Shape> {
    if input.rank() != 4
        || weight.rank() != 4
        || options.groups == 0
        || options.stride.contains(&0)
        || options.dilation.contains(&0)
        || options.output_padding[0] >= options.stride[0]
        || options.output_padding[1] >= options.stride[1]
        || input.dims()[1] != weight.dims()[0]
        || weight.dims()[0] % options.groups != 0
    {
        return Err(Error::InvalidConv2d {
            input: input.clone(),
            weight: weight.clone(),
            reason: "invalid transpose convolution geometry",
        });
    }
    let oc = weight.dims()[1]
        .checked_mul(options.groups)
        .ok_or_else(|| Error::ShapeOverflow(weight.clone()))?;
    let dim = |n: usize, k: usize, s: usize, d: usize, b: usize, a: usize, op: usize| {
        n.checked_sub(1)
            .and_then(|x| x.checked_mul(s))
            .and_then(|x| x.checked_add(d.checked_mul(k.checked_sub(1)?)?))
            .and_then(|x| x.checked_add(op))
            .and_then(|x| x.checked_add(1))
            .and_then(|x| x.checked_sub(b))
            .and_then(|x| x.checked_sub(a))
    };
    let h = dim(
        input.dims()[2],
        weight.dims()[2],
        options.stride[0],
        options.dilation[0],
        options.padding[0],
        options.padding[1],
        options.output_padding[0],
    )
    .ok_or_else(|| Error::InvalidConv2d {
        input: input.clone(),
        weight: weight.clone(),
        reason: "invalid transpose output shape",
    })?;
    let w = dim(
        input.dims()[3],
        weight.dims()[3],
        options.stride[1],
        options.dilation[1],
        options.padding[2],
        options.padding[3],
        options.output_padding[1],
    )
    .ok_or_else(|| Error::InvalidConv2d {
        input: input.clone(),
        weight: weight.clone(),
        reason: "invalid transpose output shape",
    })?;
    Ok(Shape::new([input.dims()[0], oc, h, w]))
}
pub(crate) fn conv2d_shape(input: &Shape, weight: &Shape, options: Conv2dOptions) -> Result<Shape> {
    if input.rank() != 4 || weight.rank() != 4 {
        return Err(Error::InvalidConv2d {
            input: input.clone(),
            weight: weight.clone(),
            reason: "input and weight must be rank 4",
        });
    }
    if options.groups == 0 || options.stride.contains(&0) || options.dilation.contains(&0) {
        return Err(Error::InvalidConv2d {
            input: input.clone(),
            weight: weight.clone(),
            reason: "groups, stride, and dilation must be positive",
        });
    }
    let i = input.dims();
    let w = weight.dims();
    if w[0] % options.groups != 0
        || i[1]
            != w[1]
                .checked_mul(options.groups)
                .ok_or_else(|| Error::ShapeOverflow(input.clone()))?
    {
        return Err(Error::InvalidConv2d {
            input: input.clone(),
            weight: weight.clone(),
            reason: "channel/group geometry",
        });
    }
    let spatial = |size: usize,
                   kernel: usize,
                   before: usize,
                   after: usize,
                   stride: usize,
                   dilation: usize|
     -> Result<usize> {
        let extent = kernel
            .checked_sub(1)
            .and_then(|x| x.checked_mul(dilation))
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| Error::ShapeOverflow(input.clone()))?;
        let padded = size
            .checked_add(before)
            .and_then(|x| x.checked_add(after))
            .ok_or_else(|| Error::ShapeOverflow(input.clone()))?;
        if padded < extent {
            return Err(Error::InvalidConv2d {
                input: input.clone(),
                weight: weight.clone(),
                reason: "kernel exceeds padded input",
            });
        }
        Ok((padded - extent) / stride + 1)
    };
    Ok(Shape::from([
        i[0],
        w[0],
        spatial(
            i[2],
            w[2],
            options.padding[0],
            options.padding[1],
            options.stride[0],
            options.dilation[0],
        )?,
        spatial(
            i[3],
            w[3],
            options.padding[2],
            options.padding[3],
            options.stride[1],
            options.dilation[1],
        )?,
    ]))
}

/// Returns normalized `(start, stop, step, output_length)` with the same
/// endpoint clipping rules as Rust's/Python's signed slicing model.
pub(crate) fn normalized_slice(
    dim: usize,
    slice: Slice,
    axis: usize,
) -> Result<(isize, isize, isize, usize)> {
    if slice.step == 0 {
        return Err(Error::InvalidSliceStep { axis });
    }
    let dim =
        isize::try_from(dim).map_err(|_| Error::ShapeOverflow(Shape::new(vec![usize::MAX])))?;
    let step = slice.step;
    let clamp = |value: isize, lo: isize, hi: isize| value.clamp(lo, hi);
    let (start, stop) = if step > 0 {
        let start = match slice.start {
            Some(x) => clamp(if x < 0 { x.saturating_add(dim) } else { x }, 0, dim),
            None => 0,
        };
        let stop = match slice.stop {
            Some(x) => clamp(if x < 0 { x.saturating_add(dim) } else { x }, 0, dim),
            None => dim,
        };
        (start, stop)
    } else {
        let start = match slice.start {
            Some(x) => clamp(if x < 0 { x.saturating_add(dim) } else { x }, -1, dim - 1),
            None => dim - 1,
        };
        // An omitted negative-step stop is the sentinel -1, not an index.
        let stop = match slice.stop {
            Some(x) => clamp(if x < 0 { x.saturating_add(dim) } else { x }, -1, dim - 1),
            None => -1,
        };
        (start, stop)
    };
    let length = if step > 0 {
        if start >= stop {
            0
        } else {
            usize::try_from((stop - start - 1) / step + 1).unwrap_or(0)
        }
    } else if start <= stop {
        0
    } else {
        usize::try_from((start - stop - 1) / (-step) + 1).unwrap_or(0)
    };
    Ok((start, stop, step, length))
}

fn validate_indexed(op: &'static str, input: &Node, index: &Node, axis: usize) -> Result<()> {
    if !index.dtype.is_integer() {
        return Err(Error::InvalidIndexDType {
            op,
            actual: index.dtype,
        });
    }
    if axis >= input.shape.rank() {
        return Err(Error::InvalidAxis {
            node: NodeId(usize::MAX),
            axis,
            rank: input.shape.rank(),
        });
    }
    if input.shape.rank() != index.shape.rank()
        || input
            .shape
            .dims()
            .iter()
            .zip(index.shape.dims())
            .enumerate()
            .any(|(dim, (input, index))| dim != axis && index > input)
    {
        return Err(Error::InvalidIndexedShape {
            op,
            input: input.shape.clone(),
            index: index.shape.clone(),
        });
    }
    Ok(())
}
fn normalize_axes(node: NodeId, rank: usize, axes: Option<Vec<isize>>) -> Result<Vec<usize>> {
    let mut axes = axes.unwrap_or_else(|| (0..rank).map(|x| x as isize).collect());
    for axis in &mut axes {
        if *axis < 0 {
            *axis += rank as isize;
        }
    }
    if axes.iter().any(|axis| *axis < 0 || *axis >= rank as isize) {
        return Err(Error::InvalidReductionAxes {
            node,
            axes: axes
                .iter()
                .map(|x| usize::try_from(*x).unwrap_or(usize::MAX))
                .collect(),
            rank,
        });
    }
    let mut normalized = axes.into_iter().map(|x| x as usize).collect::<Vec<_>>();
    normalized.sort_unstable();
    if normalized.windows(2).any(|x| x[0] == x[1]) {
        return Err(Error::InvalidReductionAxes {
            node,
            axes: normalized,
            rank,
        });
    }
    Ok(normalized)
}
fn reduction_shape(shape: &Shape, axes: &[usize], keepdim: bool) -> Shape {
    Shape::new(
        shape
            .dims()
            .iter()
            .enumerate()
            .filter_map(|(i, dim)| {
                if axes.contains(&i) {
                    keepdim.then_some(1)
                } else {
                    Some(*dim)
                }
            })
            .collect::<Vec<_>>(),
    )
}
fn has_empty_reduction_domain(input: &Shape, output: &Shape, axes: &[usize]) -> bool {
    matches!(output.numel(), Ok(numel) if numel > 0)
        && axes.iter().any(|axis| input.dims()[*axis] == 0)
}
fn sum_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::BF16 => dtype,
        DType::Bool => DType::I32,
        DType::I8 | DType::I16 => DType::I32,
        DType::U8 | DType::U16 => DType::U32,
        _ => dtype,
    }
}
