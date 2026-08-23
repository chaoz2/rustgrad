use crate::{CompileTrace, Error, Result, Shape, TensorData, TraceStep};
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
    Unary {
        op: UnaryOp,
        input: NodeId,
    },
    Binary {
        op: BinaryOp,
        lhs: NodeId,
        rhs: NodeId,
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
    Matmul {
        lhs: NodeId,
        rhs: NodeId,
    },
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
            Self::Unary { op, input } => format!("{}(%{input})", op.name()),
            Self::Binary { op, lhs, rhs } => format!("{}(%{lhs}, %{rhs})", op.name()),
            Self::Sum { input, axis } => format!("sum(%{input}, axis={axis})"),
            Self::SumTo { input, shape } => format!("sum_to(%{input}, {shape})"),
            Self::Reshape { input, shape } => format!("reshape(%{input}, {shape})"),
            Self::Permute { input, axes } => format!("permute(%{input}, {axes:?})"),
            Self::Expand { input, shape } => format!("expand(%{input}, {shape})"),
            Self::Matmul { lhs, rhs } => format!("matmul(%{lhs}, %{rhs})"),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub op: Op,
    pub shape: Shape,
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
        self.push(Op::Input { name: name.into() }, shape.into())
    }

    pub fn constant(&mut self, data: TensorData) -> NodeId {
        let shape = data.shape().clone();
        self.push(Op::Constant(data), shape)
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
        let shape = self.node(input)?.shape.clone();
        Ok(self.push(Op::Unary { op, input }, shape))
    }

    pub fn binary(&mut self, op: BinaryOp, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let shape = self.broadcast_shape(lhs, rhs)?;
        Ok(self.push(Op::Binary { op, lhs, rhs }, shape))
    }

    pub fn sum(&mut self, input: NodeId, axis: usize) -> Result<NodeId> {
        let source = self.node(input)?;
        let shape = source.shape.without_axis(axis).ok_or(Error::InvalidAxis {
            node: input,
            axis,
            rank: source.shape.rank(),
        })?;
        Ok(self.push(Op::Sum { input, axis }, shape))
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
        Ok(self.push(Op::Permute { input, axes }, shape))
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
        Ok(self.push(Op::Matmul { lhs, rhs }, shape))
    }

    pub fn shape(&self, id: NodeId) -> Result<&Shape> {
        Ok(&self.node(id)?.shape)
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

    pub(crate) fn push(&mut self, op: Op, shape: Shape) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { op, shape });
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
