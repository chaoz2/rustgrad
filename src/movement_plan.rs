//! Immutable contracts for materializing concat and indexed movement kernels.
use crate::{DType, Graph, NodeId, Op, Shape};
use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MovementOperand {
    pub node: NodeId,
    pub shape: Shape,
    pub dtype: DType,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MovementKernelKind {
    Concat {
        inputs: Vec<MovementOperand>,
        axis: usize,
    },
    Gather {
        input: MovementOperand,
        index: MovementOperand,
        axis: usize,
    },
    Scatter {
        base: MovementOperand,
        index: MovementOperand,
        updates: MovementOperand,
        axis: usize,
        add: bool,
    },
}

/// Fully validated materializing movement geometry and ordered pointer ABI.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MovementKernelPlan {
    pub kind: MovementKernelKind,
    pub output: NodeId,
    pub output_shape: Shape,
    pub dtype: DType,
    pub cache_key: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MovementPlanError {
    NotMovement,
    InvalidGeometry,
    UnsupportedDType,
    Overflow,
}

impl fmt::Display for MovementPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "movement plan error: {self:?}")
    }
}
impl std::error::Error for MovementPlanError {}

impl MovementOperand {
    fn from_graph(graph: &Graph, node: NodeId) -> Result<Self, MovementPlanError> {
        Ok(Self {
            node,
            shape: graph
                .shape(node)
                .map_err(|_| MovementPlanError::InvalidGeometry)?
                .clone(),
            dtype: graph
                .dtype(node)
                .map_err(|_| MovementPlanError::UnsupportedDType)?,
        })
    }
}

impl MovementKernelPlan {
    pub fn from_graph(graph: &Graph, output: NodeId) -> Result<Self, MovementPlanError> {
        let kind = match graph
            .op(output)
            .map_err(|_| MovementPlanError::InvalidGeometry)?
        {
            Op::Concat { inputs, axis } => MovementKernelKind::Concat {
                inputs: inputs
                    .iter()
                    .map(|node| MovementOperand::from_graph(graph, *node))
                    .collect::<Result<Vec<_>, _>>()?,
                axis: *axis,
            },
            Op::Gather { input, index, axis } => MovementKernelKind::Gather {
                input: MovementOperand::from_graph(graph, *input)?,
                index: MovementOperand::from_graph(graph, *index)?,
                axis: *axis,
            },
            Op::Scatter {
                base,
                index,
                updates,
                axis,
                add,
            } => MovementKernelKind::Scatter {
                base: MovementOperand::from_graph(graph, *base)?,
                index: MovementOperand::from_graph(graph, *index)?,
                updates: MovementOperand::from_graph(graph, *updates)?,
                axis: *axis,
                add: *add,
            },
            _ => return Err(MovementPlanError::NotMovement),
        };
        let mut plan = Self {
            kind,
            output,
            output_shape: graph
                .shape(output)
                .map_err(|_| MovementPlanError::InvalidGeometry)?
                .clone(),
            dtype: graph
                .dtype(output)
                .map_err(|_| MovementPlanError::UnsupportedDType)?,
            cache_key: 0,
        };
        plan.cache_key = plan.expected_cache_key();
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), MovementPlanError> {
        self.output_shape
            .numel()
            .map_err(|_| MovementPlanError::Overflow)?;
        if self.cache_key != self.expected_cache_key() {
            return Err(MovementPlanError::InvalidGeometry);
        }
        match &self.kind {
            MovementKernelKind::Concat { inputs, axis } => {
                let first = inputs.first().ok_or(MovementPlanError::InvalidGeometry)?;
                if *axis >= first.shape.rank()
                    || self.dtype != first.dtype
                    || self.output_shape.rank() != first.shape.rank()
                {
                    return Err(MovementPlanError::InvalidGeometry);
                }
                let mut axis_total = 0usize;
                for input in inputs {
                    input
                        .shape
                        .numel()
                        .map_err(|_| MovementPlanError::Overflow)?;
                    if input.dtype != self.dtype
                        || input.shape.rank() != first.shape.rank()
                        || input
                            .shape
                            .dims()
                            .iter()
                            .zip(first.shape.dims())
                            .enumerate()
                            .any(|(dim, (actual, expected))| dim != *axis && actual != expected)
                    {
                        return Err(MovementPlanError::InvalidGeometry);
                    }
                    axis_total = axis_total
                        .checked_add(input.shape.dims()[*axis])
                        .ok_or(MovementPlanError::Overflow)?;
                }
                let mut expected = first.shape.dims().to_vec();
                expected[*axis] = axis_total;
                if self.output_shape.dims() != expected {
                    return Err(MovementPlanError::InvalidGeometry);
                }
            }
            MovementKernelKind::Gather { input, index, axis } => {
                validate_index_geometry(input, index, *axis)?;
                if self.dtype != input.dtype || self.output_shape != index.shape {
                    return Err(MovementPlanError::InvalidGeometry);
                }
            }
            MovementKernelKind::Scatter {
                base,
                index,
                updates,
                axis,
                add,
            } => {
                validate_index_geometry(base, index, *axis)?;
                if self.output_shape != base.shape
                    || self.dtype != base.dtype
                    || updates.dtype != base.dtype
                    || updates.shape.rank() != index.shape.rank()
                    || updates
                        .shape
                        .dims()
                        .iter()
                        .zip(index.shape.dims())
                        .any(|(update, index)| update < index)
                {
                    return Err(MovementPlanError::InvalidGeometry);
                }
                if *add && !matches!(self.dtype, DType::F32 | DType::F64) {
                    return Err(MovementPlanError::UnsupportedDType);
                }
            }
        }
        if self
            .input_operands()
            .iter()
            .any(|operand| operand.node == self.output)
        {
            return Err(MovementPlanError::InvalidGeometry);
        }
        Ok(())
    }

    pub fn input_operands(&self) -> Vec<&MovementOperand> {
        match &self.kind {
            MovementKernelKind::Concat { inputs, .. } => inputs.iter().collect(),
            MovementKernelKind::Gather { input, index, .. } => vec![input, index],
            MovementKernelKind::Scatter {
                base,
                index,
                updates,
                ..
            } => vec![base, index, updates],
        }
    }

    fn expected_cache_key(&self) -> u64 {
        let mut plan = self.clone();
        plan.cache_key = 0;
        let mut hasher = DefaultHasher::new();
        plan.hash(&mut hasher);
        hasher.finish()
    }
}

fn validate_index_geometry(
    input: &MovementOperand,
    index: &MovementOperand,
    axis: usize,
) -> Result<(), MovementPlanError> {
    input
        .shape
        .numel()
        .and_then(|_| index.shape.numel())
        .map_err(|_| MovementPlanError::Overflow)?;
    if !index.dtype.is_integer()
        || axis >= input.shape.rank()
        || input.shape.rank() != index.shape.rank()
        || input
            .shape
            .dims()
            .iter()
            .zip(index.shape.dims())
            .enumerate()
            .any(|(dim, (input, index))| dim != axis && index > input)
    {
        return Err(MovementPlanError::InvalidGeometry);
    }
    Ok(())
}
