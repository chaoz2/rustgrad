use crate::{CompileTrace, DType, Error, Result, Scalar, Shape, TensorData, TraceStep};
use std::fmt;

mod creation;
mod reduce;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NodeId(pub(crate) usize);

impl NodeId {
    pub fn index(self) -> usize {
        self.0
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
    Cast {
        input: NodeId,
        dtype: DType,
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
    Sum {
        input: NodeId,
        axis: usize,
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
    Scatter {
        input: NodeId,
        shape: Shape,
        starts: Vec<isize>,
        steps: Vec<isize>,
    },
    Matmul {
        lhs: NodeId,
        rhs: NodeId,
    },
}

/// A Python-style per-axis signed slice. Bounds are normalized against each
/// input dimension; `None` selects the direction-appropriate endpoint.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Slice {
    pub start: Option<isize>,
    pub stop: Option<isize>,
    pub step: isize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UnaryOp {
    Neg,
    Exp,
    Log,
    Relu,
    Step,
}

impl UnaryOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Relu => "relu",
            Self::Step => "step",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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
        }
    }
}

impl Op {
    pub fn label(&self) -> String {
        match self {
            Self::Input { name } => format!("input({name:?})"),
            Self::Constant(_) => "constant".into(),
            Self::Cast { input, dtype } => format!("cast(%{input}, {dtype:?})"),
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
            Self::Sum { input, axis } => format!("sum(%{input}, axis={axis})"),
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
            Self::Scatter { input, shape, .. } => format!("scatter(%{input}, {shape})"),
            Self::Matmul { lhs, rhs } => format!("matmul(%{lhs}, %{rhs})"),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub op: Op,
    pub shape: Shape,
    pub dtype: DType,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn input(&mut self, name: impl Into<String>, shape: impl Into<Shape>) -> NodeId {
        self.input_dtype(name, shape, DType::F32)
    }

    pub fn input_dtype(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> NodeId {
        self.push(Op::Input { name: name.into() }, shape.into(), dtype)
    }

    pub fn constant(&mut self, data: TensorData) -> NodeId {
        let shape = data.shape().clone();
        let dtype = data.dtype();
        self.push(Op::Constant(data), shape, dtype)
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
    pub fn relu(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Relu, input)
    }
    pub(crate) fn step(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Step, input)
    }

    pub fn unary(&mut self, op: UnaryOp, input: NodeId) -> Result<NodeId> {
        let source = self.node(input)?;
        Ok(self.push(Op::Unary { op, input }, source.shape.clone(), source.dtype))
    }

    pub fn binary(&mut self, op: BinaryOp, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let shape = self.broadcast_shape(lhs, rhs)?;
        let dtype = self.node(lhs)?.dtype.promote(self.node(rhs)?.dtype);
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
        let source = self.node(input)?;
        let shape = source.shape.without_axis(axis).ok_or(Error::InvalidAxis {
            node: input,
            axis,
            rank: source.shape.rank(),
        })?;
        Ok(self.push(Op::Sum { input, axis }, shape, source.dtype))
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

    pub(crate) fn scatter(
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
            Op::Scatter {
                input,
                shape: shape.clone(),
                starts,
                steps,
            },
            shape,
            source.dtype,
        ))
    }

    pub fn matmul(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let lhs_shape = &self.node(lhs)?.shape;
        let rhs_shape = &self.node(rhs)?.shape;
        if lhs_shape.rank() != 2
            || rhs_shape.rank() != 2
            || lhs_shape.dims()[1] != rhs_shape.dims()[0]
        {
            return Err(Error::InvalidMatmul {
                lhs: lhs_shape.clone(),
                rhs: rhs_shape.clone(),
            });
        }
        let shape = Shape::from([lhs_shape.dims()[0], rhs_shape.dims()[1]]);
        let dtype = self.node(lhs)?.dtype.promote(self.node(rhs)?.dtype);
        Ok(self.push(Op::Matmul { lhs, rhs }, shape, dtype))
    }

    pub fn shape(&self, id: NodeId) -> Result<&Shape> {
        Ok(&self.node(id)?.shape)
    }

    pub fn dtype(&self, id: NodeId) -> Result<DType> {
        Ok(self.node(id)?.dtype)
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
            })
            .collect();
        Ok(CompileTrace { output, steps })
    }

    pub(crate) fn push(&mut self, op: Op, shape: Shape, dtype: DType) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { op, shape, dtype });
        id
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
