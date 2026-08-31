//! Symbolic affine view metadata shared by capture, artifact validation, and
//! Graph-free concrete specialization.
use super::{capture::ReplayError, symbolic::SymbolicGuard};
use crate::{
    Graph, NodeId, Op, Shape, SymbolicDim, SymbolicExpr, SymbolicShape, SymbolicVar, ViewMap,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SymbolicViewMap {
    pub(crate) source_shape: SymbolicShape,
    pub(crate) logical_shape: SymbolicShape,
    pub(crate) strides: Vec<SymbolicExpr>,
    pub(crate) offset: SymbolicExpr,
}

impl SymbolicViewMap {
    fn identity(shape: SymbolicShape) -> Result<Self, ReplayError> {
        Ok(Self {
            strides: contiguous_strides(&shape)?,
            logical_shape: shape.clone(),
            source_shape: shape,
            offset: SymbolicExpr::constant(0),
        })
    }

    fn shrink(
        &self,
        bounds: &[(usize, usize)],
        environment: &BTreeMap<SymbolicVar, i64>,
    ) -> Result<Self, ReplayError> {
        if bounds.len() != self.logical_shape.rank() {
            return Err(ReplayError::Symbolic(
                "symbolic shrink rank mismatch".into(),
            ));
        }
        let mut offset = self.offset.clone();
        let mut logical = Vec::with_capacity(bounds.len());
        for (axis, ((start, end), (dim, stride))) in bounds
            .iter()
            .zip(self.logical_shape.dims().iter().zip(&self.strides))
            .enumerate()
        {
            let dim = dim.expression();
            let (min, template) = (
                dim.bounds().map_err(symbolic_error)?.min,
                evaluate(dim, environment)?,
            );
            let start = i64::try_from(*start)
                .map_err(|_| ReplayError::Symbolic("symbolic shrink start exceeds i64".into()))?;
            if start > min {
                return Err(ReplayError::Unsupported(format!(
                    "symbolic shrink axis {axis} start is not valid across the declared domain"
                )));
            }
            let end = i64::try_from(*end)
                .map_err(|_| ReplayError::Symbolic("symbolic shrink end exceeds i64".into()))?;
            let end = if end == template {
                dim.clone()
            } else if end <= min {
                SymbolicExpr::constant(end)
            } else {
                return Err(ReplayError::Unsupported(format!(
                    "symbolic shrink axis {axis} end is not stable across the declared domain"
                )));
            };
            let start = SymbolicExpr::constant(start);
            offset = offset + start.clone() * stride.clone();
            logical.push(SymbolicDim::new(end - start));
        }
        Ok(Self {
            source_shape: self.source_shape.clone(),
            logical_shape: SymbolicShape::new(logical),
            strides: self.strides.clone(),
            offset,
        })
    }

    fn reshape(&self, shape: SymbolicShape) -> Result<Self, ReplayError> {
        if self.strides != contiguous_strides(&self.logical_shape)? {
            return Err(ReplayError::Unsupported(
                "symbolic reshape requires a contiguous affine input view".into(),
            ));
        }
        Ok(Self {
            source_shape: self.source_shape.clone(),
            strides: contiguous_strides(&shape)?,
            logical_shape: shape,
            offset: self.offset.clone(),
        })
    }

    fn permute(&self, axes: &[usize]) -> Result<Self, ReplayError> {
        let mut sorted = axes.to_vec();
        sorted.sort_unstable();
        if sorted != (0..self.logical_shape.rank()).collect::<Vec<_>>() {
            return Err(ReplayError::Symbolic(
                "symbolic permutation is malformed".into(),
            ));
        }
        Ok(Self {
            source_shape: self.source_shape.clone(),
            logical_shape: SymbolicShape::new(
                axes.iter()
                    .map(|axis| self.logical_shape.dims()[*axis].clone())
                    .collect::<Vec<_>>(),
            ),
            strides: axes
                .iter()
                .map(|axis| self.strides[*axis].clone())
                .collect(),
            offset: self.offset.clone(),
        })
    }

    fn expand(&self, shape: SymbolicShape) -> Result<Self, ReplayError> {
        if self.logical_shape.rank() > shape.rank() {
            return Err(ReplayError::Symbolic(
                "symbolic expand rank mismatch".into(),
            ));
        }
        let pad = shape.rank() - self.logical_shape.rank();
        let mut strides = vec![SymbolicExpr::constant(0); pad];
        for ((input, output), stride) in self
            .logical_shape
            .dims()
            .iter()
            .zip(&shape.dims()[pad..])
            .zip(&self.strides)
        {
            let input_one = input
                .expression()
                .bounds()
                .map_err(symbolic_error)?
                .constant()
                == Some(1);
            if input == output {
                strides.push(stride.clone());
            } else if input_one {
                strides.push(SymbolicExpr::constant(0));
            } else {
                return Err(ReplayError::Symbolic(
                    "symbolic expand dimensions are incompatible".into(),
                ));
            }
        }
        Ok(Self {
            source_shape: self.source_shape.clone(),
            logical_shape: shape,
            strides,
            offset: self.offset.clone(),
        })
    }

    fn stride(
        &self,
        slices: &[crate::Slice],
        environment: &BTreeMap<SymbolicVar, i64>,
    ) -> Result<Self, ReplayError> {
        if slices.len() != self.logical_shape.rank() {
            return Err(ReplayError::Symbolic(
                "symbolic stride rank mismatch".into(),
            ));
        }
        let mut offset = self.offset.clone();
        let mut logical = Vec::with_capacity(slices.len());
        let mut strides = Vec::with_capacity(slices.len());
        for (axis, ((slice, dim), stride)) in slices
            .iter()
            .zip(self.logical_shape.dims())
            .zip(&self.strides)
            .enumerate()
        {
            if slice.step <= 0 {
                return Err(ReplayError::Unsupported(
                    "symbolic affine views require positive strides".into(),
                ));
            }
            let step = i64::try_from(slice.step)
                .map_err(|_| ReplayError::Symbolic("symbolic stride exceeds i64".into()))?;
            let dim = dim.expression();
            let dim_bounds = dim.bounds().map_err(symbolic_error)?;
            let template = evaluate(dim, environment)?;
            let start = match slice.start {
                None => 0,
                Some(start) if start >= 0 => i64::try_from(start).map_err(|_| {
                    ReplayError::Symbolic("symbolic slice start exceeds i64".into())
                })?,
                _ => {
                    return Err(ReplayError::Unsupported(
                        "symbolic affine views do not infer negative slice bounds".into(),
                    ));
                }
            };
            if start > dim_bounds.min {
                return Err(ReplayError::Unsupported(format!(
                    "symbolic slice axis {axis} start is not valid across the declared domain"
                )));
            }
            let stop = match slice.stop {
                None => dim.clone(),
                Some(stop) if stop >= 0 => {
                    let stop = i64::try_from(stop).map_err(|_| {
                        ReplayError::Symbolic("symbolic slice stop exceeds i64".into())
                    })?;
                    if stop == template {
                        dim.clone()
                    } else if stop <= dim_bounds.min {
                        SymbolicExpr::constant(stop)
                    } else {
                        return Err(ReplayError::Unsupported(format!(
                            "symbolic slice axis {axis} stop is not stable across the declared domain"
                        )));
                    }
                }
                _ => {
                    return Err(ReplayError::Unsupported(
                        "symbolic affine views do not infer negative slice bounds".into(),
                    ));
                }
            };
            let start_expr = SymbolicExpr::constant(start);
            let step_expr = SymbolicExpr::constant(step);
            let length = (stop - start_expr.clone() + SymbolicExpr::constant(step - 1))
                .try_floor_div(step_expr.clone())
                .map_err(symbolic_error)?
                .maximum(SymbolicExpr::constant(0));
            offset = offset + start_expr * stride.clone();
            logical.push(SymbolicDim::new(length));
            strides.push(stride.clone() * step_expr);
        }
        Ok(Self {
            source_shape: self.source_shape.clone(),
            logical_shape: SymbolicShape::new(logical),
            strides,
            offset,
        })
    }

    pub(crate) fn expressions(&self) -> Vec<&SymbolicExpr> {
        self.source_shape
            .dims()
            .iter()
            .chain(self.logical_shape.dims())
            .map(SymbolicDim::expression)
            .chain(self.strides.iter())
            .chain(std::iter::once(&self.offset))
            .collect()
    }

    pub(crate) fn specialize(
        &self,
        environment: &BTreeMap<SymbolicVar, i64>,
    ) -> Result<ViewMap, ReplayError> {
        let source_shape = bind_shape(&self.source_shape, environment)?;
        let logical_shape = bind_shape(&self.logical_shape, environment)?;
        let strides = self
            .strides
            .iter()
            .map(|stride| evaluate_usize(stride, environment))
            .collect::<Result<Vec<_>, _>>()?;
        let view = ViewMap {
            source_shape,
            logical_shape,
            strides,
            offset: evaluate_usize(&self.offset, environment)?,
        };
        crate::uop::artifact::validate_view(&view)
            .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
        Ok(view)
    }

    pub(crate) fn validate_bounds(&self) -> Result<(), ReplayError> {
        for expression in self.expressions() {
            expression.bounds().map_err(symbolic_error)?;
        }
        let offset_bounds = self.offset.bounds().map_err(symbolic_error)?;
        if self.strides.len() != self.logical_shape.rank()
            || offset_bounds.min < 0
            || self.strides.iter().try_fold(false, |invalid, stride| {
                Ok::<_, ReplayError>(invalid || stride.bounds().map_err(symbolic_error)?.min < 0)
            })?
        {
            return Err(ReplayError::Symbolic(
                "symbolic view rank, offset, or stride is malformed".into(),
            ));
        }
        validate_shape_bounds(&self.source_shape)?;
        validate_shape_bounds(&self.logical_shape)?;
        let source_max = self
            .source_shape
            .numel()
            .map_err(symbolic_error)?
            .bounds()
            .map_err(symbolic_error)?
            .max;
        let logical_max = self
            .logical_shape
            .numel()
            .map_err(symbolic_error)?
            .bounds()
            .map_err(symbolic_error)?
            .max;
        let mut address_max = offset_bounds.max;
        for (dimension, stride) in self.logical_shape.dims().iter().zip(&self.strides) {
            let dimension = dimension.expression().bounds().map_err(symbolic_error)?.max;
            let stride = stride.bounds().map_err(symbolic_error)?.max;
            address_max =
                address_max
                    .checked_add(dimension.saturating_sub(1).checked_mul(stride).ok_or_else(
                        || ReplayError::Symbolic("symbolic view address overflows i64".into()),
                    )?)
                    .ok_or_else(|| {
                        ReplayError::Symbolic("symbolic view address overflows i64".into())
                    })?;
        }
        if (source_max == 0 && (logical_max != 0 || address_max != 0))
            || logical_max != 0 && address_max >= source_max
        {
            return Err(ReplayError::Symbolic(
                "symbolic view address exceeds its source extent".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn movement_shape(
    op: &Op,
    input: &SymbolicShape,
    concrete_output: &Shape,
    candidates: &[SymbolicExpr],
    environment: &BTreeMap<SymbolicVar, i64>,
    guards: &mut Vec<SymbolicGuard>,
) -> Result<SymbolicShape, ReplayError> {
    match op {
        Op::Shrink { bounds, .. } => Ok(SymbolicViewMap::identity(input.clone())?
            .shrink(bounds, environment)?
            .logical_shape),
        Op::Reshape { .. } => {
            let output = lift_shape(concrete_output, candidates, environment)?;
            let input_elements = input.numel().map_err(symbolic_error)?;
            let output_elements = output.numel().map_err(symbolic_error)?;
            if input_elements != output_elements {
                guards.push(SymbolicGuard::equal(input_elements, output_elements));
            }
            Ok(output)
        }
        Op::Permute { axes, .. } => Ok(SymbolicShape::new(
            axes.iter()
                .map(|axis| input.dims()[*axis].clone())
                .collect::<Vec<_>>(),
        )),
        Op::Expand { .. } => {
            if input.rank() > concrete_output.rank() {
                return Err(ReplayError::Symbolic(
                    "symbolic expand rank mismatch".into(),
                ));
            }
            let pad = concrete_output.rank() - input.rank();
            let mut output = Vec::with_capacity(concrete_output.rank());
            for axis in 0..concrete_output.rank() {
                let concrete = concrete_output.dims()[axis];
                let input_dim = axis
                    .checked_sub(pad)
                    .and_then(|axis| input.dims().get(axis));
                if let Some(input_dim) = input_dim {
                    let is_one = input_dim
                        .expression()
                        .bounds()
                        .map_err(symbolic_error)?
                        .constant()
                        == Some(1);
                    if !is_one {
                        if evaluate(input_dim.expression(), environment)?
                            != i64::try_from(concrete).map_err(|_| {
                                ReplayError::Symbolic("expand dimension exceeds i64".into())
                            })?
                        {
                            return Err(ReplayError::Symbolic(
                                "symbolic expand template is inconsistent".into(),
                            ));
                        }
                        output.push(input_dim.clone());
                        continue;
                    }
                }
                output.push(SymbolicDim::new(lift_dimension(
                    concrete,
                    candidates,
                    environment,
                )?));
            }
            Ok(SymbolicShape::new(output))
        }
        Op::Stride { slices, .. } => Ok(SymbolicViewMap::identity(input.clone())?
            .stride(slices, environment)?
            .logical_shape),
        _ => Err(ReplayError::Unsupported(
            "operation is not an affine symbolic movement".into(),
        )),
    }
}

pub(crate) fn derive_view(
    graph: &Graph,
    node: NodeId,
    shapes: &BTreeMap<NodeId, SymbolicShape>,
    environment: &BTreeMap<SymbolicVar, i64>,
) -> Result<(NodeId, SymbolicViewMap), ReplayError> {
    let shape = || {
        shapes
            .get(&node)
            .cloned()
            .ok_or_else(|| ReplayError::Symbolic("symbolic view shape is absent".into()))
    };
    match graph
        .op(node)
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?
    {
        Op::Input { .. } | Op::Constant(_) => Ok((node, SymbolicViewMap::identity(shape()?)?)),
        Op::Shrink { input, bounds } => {
            let (source, view) = derive_view(graph, *input, shapes, environment)?;
            Ok((source, view.shrink(bounds, environment)?))
        }
        Op::Reshape { input, .. } => {
            let (source, view) = derive_view(graph, *input, shapes, environment)?;
            Ok((source, view.reshape(shape()?)?))
        }
        Op::Permute { input, axes } => {
            let (source, view) = derive_view(graph, *input, shapes, environment)?;
            Ok((source, view.permute(axes)?))
        }
        Op::Expand { input, .. } => {
            let (source, view) = derive_view(graph, *input, shapes, environment)?;
            Ok((source, view.expand(shape()?)?))
        }
        Op::Stride { input, slices } => {
            let (source, view) = derive_view(graph, *input, shapes, environment)?;
            Ok((source, view.stride(slices, environment)?))
        }
        _ => Err(ReplayError::Unsupported(
            "symbolic view source is computed or non-affine".into(),
        )),
    }
}

pub(crate) fn candidates(
    shapes: impl Iterator<Item = SymbolicShape>,
) -> Result<Vec<SymbolicExpr>, ReplayError> {
    let mut candidates = BTreeSet::new();
    for shape in shapes {
        for dim in shape.dims() {
            let expression = dim
                .expression()
                .simplify()
                .map_err(symbolic_error)?
                .expression;
            if !expression.variables().is_empty() {
                candidates.insert(expression.clone());
            }
            for variable in expression.variables() {
                candidates.insert(SymbolicExpr::Var(variable));
            }
        }
        let elements = shape
            .numel()
            .map_err(symbolic_error)?
            .simplify()
            .map_err(symbolic_error)?
            .expression;
        if !elements.variables().is_empty() {
            candidates.insert(elements);
        }
    }
    Ok(candidates.into_iter().collect())
}

fn lift_shape(
    concrete: &Shape,
    candidates: &[SymbolicExpr],
    environment: &BTreeMap<SymbolicVar, i64>,
) -> Result<SymbolicShape, ReplayError> {
    concrete
        .dims()
        .iter()
        .map(|dimension| lift_dimension(*dimension, candidates, environment).map(SymbolicDim::new))
        .collect::<Result<Vec<_>, _>>()
        .map(SymbolicShape::new)
}

fn lift_dimension(
    concrete: usize,
    candidates: &[SymbolicExpr],
    environment: &BTreeMap<SymbolicVar, i64>,
) -> Result<SymbolicExpr, ReplayError> {
    let concrete = i64::try_from(concrete)
        .map_err(|_| ReplayError::Symbolic("concrete dimension exceeds i64".into()))?;
    let matches = candidates
        .iter()
        .filter(|candidate| evaluate(candidate, environment).ok() == Some(concrete))
        .cloned()
        .collect::<BTreeSet<_>>();
    if matches.len() == 1 {
        Ok(matches.into_iter().next().unwrap())
    } else {
        Ok(SymbolicExpr::constant(concrete))
    }
}

fn contiguous_strides(shape: &SymbolicShape) -> Result<Vec<SymbolicExpr>, ReplayError> {
    let mut strides = vec![SymbolicExpr::constant(1); shape.rank()];
    let mut running = SymbolicExpr::constant(1);
    for axis in (0..shape.rank()).rev() {
        strides[axis] = running.clone();
        running = running * shape.dims()[axis].expression().clone();
        running.bounds().map_err(symbolic_error)?;
    }
    Ok(strides)
}

fn bind_shape(
    shape: &SymbolicShape,
    environment: &BTreeMap<SymbolicVar, i64>,
) -> Result<Shape, ReplayError> {
    shape
        .dims()
        .iter()
        .map(|dim| evaluate_usize(dim.expression(), environment))
        .collect::<Result<Vec<_>, _>>()
        .map(Shape::new)
}

fn evaluate(
    expression: &SymbolicExpr,
    environment: &BTreeMap<SymbolicVar, i64>,
) -> Result<i64, ReplayError> {
    let projected = expression
        .variables()
        .into_iter()
        .filter_map(|variable| environment.get(&variable).map(|value| (variable, *value)))
        .collect();
    expression
        .evaluate(&projected)
        .map_err(|error| ReplayError::Symbolic(error.to_string()))
}

fn evaluate_usize(
    expression: &SymbolicExpr,
    environment: &BTreeMap<SymbolicVar, i64>,
) -> Result<usize, ReplayError> {
    usize::try_from(evaluate(expression, environment)?)
        .map_err(|_| ReplayError::Symbolic("symbolic view value is not a usize".into()))
}

fn validate_shape_bounds(shape: &SymbolicShape) -> Result<(), ReplayError> {
    for dim in shape.dims() {
        let bounds = dim.expression().bounds().map_err(symbolic_error)?;
        if bounds.min < 0 {
            return Err(ReplayError::Symbolic(
                "symbolic view dimension may be negative".into(),
            ));
        }
        usize::try_from(bounds.max)
            .map_err(|_| ReplayError::Symbolic("symbolic view dimension exceeds usize".into()))?;
    }
    let elements = shape.numel().map_err(symbolic_error)?;
    usize::try_from(elements.bounds().map_err(symbolic_error)?.max)
        .map_err(|_| ReplayError::Symbolic("symbolic view extent exceeds usize".into()))?;
    Ok(())
}

fn symbolic_error(error: crate::SymbolicError) -> ReplayError {
    ReplayError::Symbolic(error.to_string())
}
