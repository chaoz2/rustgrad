//! Symbolic constant specialization for the shared projected-index algebra.

use super::capture::ReplayError;
use crate::{
    Graph, NodeId, Op, Shape, SymbolicDim, SymbolicExpr, SymbolicShape, SymbolicVar,
    projected_index::{ProjectedExpr, ProjectedIndexEmitter, ProjectedIndexPlan},
    uop::Binary,
};
use std::collections::BTreeMap;

/// One graph-derived projected address family. The map is keyed by its
/// schedule item and projected-Index ordinal; buffer identity alone is not
/// sufficient because one source may be read through multiple projections.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SymbolicProjectedIndexMap {
    pub(crate) source_shape: SymbolicShape,
    pub(crate) output_shape: SymbolicShape,
    pub(crate) expression: ProjectedExpr<SymbolicExpr>,
}

impl SymbolicProjectedIndexMap {
    pub(crate) fn derive(
        graph: &Graph,
        node: NodeId,
        source: NodeId,
        output_shape: SymbolicShape,
        shapes: &BTreeMap<NodeId, SymbolicShape>,
        template: &BTreeMap<SymbolicVar, i64>,
    ) -> Result<Self, ReplayError> {
        let logical = shape(shapes, node)?;
        let coordinates = logical_coordinates(logical, &output_shape)?;
        let (derived_source, expression) = derive_chain(
            graph,
            node,
            source,
            coordinates,
            &output_shape,
            shapes,
            template,
        )?;
        if derived_source != source {
            return Err(ReplayError::Unsupported(
                "symbolic projected source is inconsistent".into(),
            ));
        }
        let map = Self {
            source_shape: shape(shapes, source)?.clone(),
            output_shape,
            expression,
        };
        map.validate_bounds()?;
        Ok(map)
    }

    pub(crate) fn expressions(&self) -> Vec<&SymbolicExpr> {
        self.source_shape
            .dims()
            .iter()
            .chain(self.output_shape.dims())
            .map(SymbolicDim::expression)
            .chain(self.expression.constants())
            .collect()
    }

    /// Proves address validity without enumerating the declared domain. The
    /// linear upper bound remains a symbolic expression, preserving its
    /// correlation with shape variables through the final polynomial proof.
    pub(crate) fn validate_bounds(&self) -> Result<(), ReplayError> {
        validate_shape(&self.source_shape)?;
        validate_shape(&self.output_shape)?;
        let mut nodes = 0;
        self.expression
            .validate_size(0, &mut nodes)
            .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
        let source_elements = self
            .source_shape
            .numel()
            .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
        let output_elements = self
            .output_shape
            .numel()
            .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
        if output_elements
            .bounds()
            .map_err(|error| ReplayError::Symbolic(error.to_string()))?
            .max
            == 0
        {
            return Ok(());
        }
        if !nonempty_source_is_proven(
            &self.source_shape,
            &self.output_shape,
            &source_elements,
            &output_elements,
        )? {
            return Err(ReplayError::Unsupported(
                "symbolic projected nonempty output may read an empty source".into(),
            ));
        }
        // Addressing is conditional on a nonempty output. In that branch every
        // output dimension is positive. Shape validation makes every source
        // dimension nonnegative, so the positive source numel proved above
        // makes every source dimension positive as well. Remove only the
        // max(d, 1) guards justified by those facts; the stored/runtime
        // expression remains zero-safe for empty invocations.
        let positive_dimensions = self
            .source_shape
            .dims()
            .iter()
            .chain(self.output_shape.dims())
            .map(|dimension| dimension.expression().clone())
            .collect::<Vec<_>>();
        let linear_max = output_elements - SymbolicExpr::constant(1);
        let bounds = projected_bounds(&self.expression, &linear_max, &positive_dimensions)?;
        let minimum = super::symbolic_view::simplified(assume_nonempty(
            &bounds.minimum,
            &positive_dimensions,
        )?)?;
        let source_max = source_elements - SymbolicExpr::constant(1);
        let maximum = assume_nonempty(&bounds.maximum, &positive_dimensions)?;
        let upper_margin = super::symbolic_view::simplified(source_max - maximum)?;
        if !super::symbolic_view::proven_nonnegative(&minimum)?
            || !super::symbolic_view::proven_nonnegative(&upper_margin)?
        {
            return Err(ReplayError::Symbolic(
                "symbolic projected address exceeds its source extent".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn specialize_expression(
        &self,
        environment: &BTreeMap<SymbolicVar, i64>,
    ) -> Result<ProjectedExpr<i64>, ReplayError> {
        self.expression
            .try_map(&mut |constant| evaluate(constant, environment))
    }

    pub(crate) fn specialize_uop(
        &self,
        environment: &BTreeMap<SymbolicVar, i64>,
    ) -> Result<crate::UOp, ReplayError> {
        let output = bind_shape(&self.output_shape, environment)?;
        let output_elements = output.numel().map_err(|_| {
            ReplayError::Symbolic("symbolic projected output extent overflows".into())
        })?;
        self.specialize_expression(environment)?
            .canonicalized_for_output(output_elements)
            .to_uop(output_elements)
            .map_err(|error| ReplayError::Symbolic(error.to_string()))
    }

    pub(crate) fn matches_template(
        &self,
        index: &crate::UOp,
        environment: &BTreeMap<SymbolicVar, i64>,
    ) -> Result<bool, ReplayError> {
        let concrete = ProjectedIndexPlan::from_index(index)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        let crate::Operation::Index(crate::IndexValue::Buffer {
            input_shape,
            output_shape,
            ..
        }) = index.operation()
        else {
            return Ok(false);
        };
        if input_shape != &bind_shape(&self.source_shape, environment)?
            || output_shape != &bind_shape(&self.output_shape, environment)?
        {
            return Ok(false);
        }
        if concrete.output_elements == 0 {
            // A zero template cannot authenticate an address formula that may
            // become live under another binding. Permanently empty families
            // remain addressless and safe.
            return Ok(self
                .output_shape
                .numel()
                .and_then(|elements| elements.bounds())
                .map_err(|error| ReplayError::Symbolic(error.to_string()))?
                .max
                == 0);
        }
        Ok(crate::projected_index::projected_expr_eq(
            &concrete.canonical_expression(),
            &self
                .specialize_expression(environment)?
                .canonicalized_for_output(concrete.output_elements),
        ))
    }

    pub(crate) fn render<E: ProjectedIndexEmitter<SymbolicExpr>>(
        &self,
        emitter: &mut E,
    ) -> Result<E::Value, E::Error> {
        self.expression.emit(emitter)
    }
}

fn derive_chain(
    graph: &Graph,
    node: NodeId,
    source: NodeId,
    coordinates: Vec<ProjectedExpr<SymbolicExpr>>,
    iteration_shape: &SymbolicShape,
    shapes: &BTreeMap<NodeId, SymbolicShape>,
    template: &BTreeMap<SymbolicVar, i64>,
) -> Result<(NodeId, ProjectedExpr<SymbolicExpr>), ReplayError> {
    if node == source {
        return Ok((
            source,
            linearize(shape(shapes, node)?, &coordinates, iteration_shape)?,
        ));
    }
    match graph
        .op(node)
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?
    {
        Op::Shrink { input, bounds } => {
            if coordinates.len() != bounds.len() {
                return Err(ReplayError::Symbolic(
                    "projected shrink rank mismatch".into(),
                ));
            }
            let coordinates = coordinates
                .into_iter()
                .zip(bounds)
                .map(|(coordinate, (start, _))| {
                    if *start == 0 {
                        Ok(coordinate)
                    } else {
                        binary(
                            Binary::Add,
                            coordinate,
                            constant_usize(*start, "projected shrink start")?,
                        )
                    }
                })
                .collect::<Result<Vec<_>, ReplayError>>()?;
            derive_chain(
                graph,
                *input,
                source,
                coordinates,
                iteration_shape,
                shapes,
                template,
            )
        }
        Op::Reshape { input, .. } => {
            let linear = linearize(shape(shapes, node)?, &coordinates, iteration_shape)?;
            let coordinates = decompose(shape(shapes, *input)?, linear)?;
            derive_chain(
                graph,
                *input,
                source,
                coordinates,
                iteration_shape,
                shapes,
                template,
            )
        }
        Op::Permute { input, axes } => {
            let mut input_coordinates = vec![zero(); axes.len()];
            for (output_axis, input_axis) in axes.iter().copied().enumerate() {
                input_coordinates[input_axis] =
                    coordinates.get(output_axis).cloned().ok_or_else(|| {
                        ReplayError::Symbolic("projected permutation rank mismatch".into())
                    })?;
            }
            derive_chain(
                graph,
                *input,
                source,
                input_coordinates,
                iteration_shape,
                shapes,
                template,
            )
        }
        Op::Expand { input, .. } => {
            let input_shape = shape(shapes, *input)?;
            if input_shape.rank() > coordinates.len() {
                return Err(ReplayError::Symbolic(
                    "projected expand rank mismatch".into(),
                ));
            }
            let delta = coordinates.len() - input_shape.rank();
            let coordinates = input_shape
                .dims()
                .iter()
                .enumerate()
                .map(|(axis, dimension)| {
                    if is_one(dimension)? {
                        Ok(zero())
                    } else {
                        Ok(coordinates[axis + delta].clone())
                    }
                })
                .collect::<Result<Vec<_>, ReplayError>>()?;
            derive_chain(
                graph,
                *input,
                source,
                coordinates,
                iteration_shape,
                shapes,
                template,
            )
        }
        Op::Stride { input, slices } => {
            if slices.len() != coordinates.len() {
                return Err(ReplayError::Symbolic(
                    "projected stride rank mismatch".into(),
                ));
            }
            let input_shape = shape(shapes, *input)?;
            let concrete = bind_shape(input_shape, template)?;
            let coordinates = coordinates
                .into_iter()
                .zip(slices)
                .zip(input_shape.dims().iter().zip(concrete.dims()))
                .enumerate()
                .map(|(axis, ((coordinate, slice), (dimension, concrete)))| {
                    let (_, _, step, _) = crate::ir::normalized_slice(*concrete, *slice, axis)
                        .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
                    if slice.start.is_some() || slice.stop.is_some() || !matches!(step, 1 | -1) {
                        return Err(ReplayError::Unsupported(
                            "symbolic projected stride admits only full forward or reverse slices"
                                .into(),
                        ));
                    }
                    let scaled = if step == 1 {
                        coordinate
                    } else {
                        binary(
                            Binary::Mul,
                            coordinate,
                            constant(SymbolicExpr::constant(-1)),
                        )?
                    };
                    if step == 1 {
                        Ok(scaled)
                    } else {
                        binary(
                            Binary::Add,
                            scaled,
                            constant(dimension.expression().clone() - SymbolicExpr::constant(1)),
                        )
                    }
                })
                .collect::<Result<Vec<_>, ReplayError>>()?;
            derive_chain(
                graph,
                *input,
                source,
                coordinates,
                iteration_shape,
                shapes,
                template,
            )
        }
        _ => Err(ReplayError::Unsupported(
            "symbolic projected movement chain is not source-backed".into(),
        )),
    }
}

fn logical_coordinates(
    logical: &SymbolicShape,
    output: &SymbolicShape,
) -> Result<Vec<ProjectedExpr<SymbolicExpr>>, ReplayError> {
    if logical.rank() > output.rank() {
        return Err(ReplayError::Symbolic(
            "projected broadcast rank mismatch".into(),
        ));
    }
    let delta = output.rank() - logical.rank();
    let strides = contiguous_strides(output)?;
    logical
        .dims()
        .iter()
        .enumerate()
        .map(|(axis, dimension)| {
            let output_axis = axis + delta;
            let output_dimension = &output.dims()[output_axis];
            if is_one(dimension)? {
                return Ok(zero());
            }
            if dimension != output_dimension {
                return Err(ReplayError::Unsupported(
                    "symbolic projected broadcast relation is not structural".into(),
                ));
            }
            let divided = if is_projected_one(&strides[output_axis])? {
                ProjectedExpr::Linear
            } else {
                binary(
                    Binary::FloorDiv,
                    ProjectedExpr::Linear,
                    strides[output_axis].clone(),
                )?
            };
            binary(Binary::Mod, divided, positive_dimension(output_dimension))
        })
        .collect()
}

fn linearize(
    shape: &SymbolicShape,
    coordinates: &[ProjectedExpr<SymbolicExpr>],
    iteration_shape: &SymbolicShape,
) -> Result<ProjectedExpr<SymbolicExpr>, ReplayError> {
    if shape.rank() != coordinates.len() {
        return Err(ReplayError::Symbolic(
            "projected coordinate rank mismatch".into(),
        ));
    }
    if let Some(recomposed) = recomposed_linear(shape, coordinates, iteration_shape)? {
        return Ok(recomposed);
    }
    let strides = contiguous_strides(shape)?;
    coordinates
        .iter()
        .cloned()
        .zip(strides)
        .try_fold(zero(), |sum, (coordinate, stride)| {
            binary(Binary::Add, sum, binary(Binary::Mul, coordinate, stride)?)
        })
}

fn recomposed_linear(
    shape: &SymbolicShape,
    coordinates: &[ProjectedExpr<SymbolicExpr>],
    iteration_shape: &SymbolicShape,
) -> Result<Option<ProjectedExpr<SymbolicExpr>>, ReplayError> {
    let strides = contiguous_strides(shape)?;
    let mut decomposed = None;
    for ((coordinate, dimension), stride) in coordinates.iter().zip(shape.dims()).zip(strides) {
        if is_one(dimension)? {
            if coordinate != &zero() {
                return Ok(None);
            }
            continue;
        }
        let ProjectedExpr::Binary {
            operation: Binary::Mod,
            lhs: divided,
            rhs: modulus,
        } = coordinate
        else {
            return Ok(None);
        };
        if modulus.as_ref() != &positive_dimension(dimension) {
            return Ok(None);
        }
        let candidate = if is_projected_one(&stride)? {
            divided.as_ref()
        } else {
            let ProjectedExpr::Binary {
                operation: Binary::FloorDiv,
                lhs: candidate,
                rhs: divisor,
            } = divided.as_ref()
            else {
                return Ok(None);
            };
            if divisor.as_ref() != &stride {
                return Ok(None);
            }
            candidate.as_ref()
        };
        if decomposed.as_ref().is_some_and(|prior| prior != candidate) {
            return Ok(None);
        }
        decomposed = Some(candidate.clone());
    }
    let Some(decomposed) = decomposed else {
        return Ok(None);
    };
    // The same structural recovery is safe only when the complete symbolic
    // consumer domain proves that shared numerator is already a valid address
    // in this shape. In particular, an outer broadcast does not collapse to
    // Linear because its iteration extent can exceed this logical extent.
    let proof = SymbolicProjectedIndexMap {
        source_shape: shape.clone(),
        output_shape: iteration_shape.clone(),
        expression: decomposed.clone(),
    };
    Ok(proof.validate_bounds().is_ok().then_some(decomposed))
}

fn decompose(
    shape: &SymbolicShape,
    linear: ProjectedExpr<SymbolicExpr>,
) -> Result<Vec<ProjectedExpr<SymbolicExpr>>, ReplayError> {
    let strides = contiguous_strides(shape)?;
    shape
        .dims()
        .iter()
        .zip(strides)
        .map(|(dimension, stride)| {
            if is_one(dimension)? {
                Ok(zero())
            } else {
                let divided = if is_projected_one(&stride)? {
                    linear.clone()
                } else {
                    binary(Binary::FloorDiv, linear.clone(), stride)?
                };
                binary(Binary::Mod, divided, positive_dimension(dimension))
            }
        })
        .collect()
}

fn contiguous_strides(
    shape: &SymbolicShape,
) -> Result<Vec<ProjectedExpr<SymbolicExpr>>, ReplayError> {
    let mut running = constant(SymbolicExpr::constant(1));
    let mut strides = Vec::with_capacity(shape.rank());
    for dimension in shape.dims().iter().rev() {
        strides.push(running.clone());
        running = binary(Binary::Mul, running, positive_dimension(dimension))?;
    }
    strides.reverse();
    Ok(strides)
}

fn positive_dimension(dimension: &SymbolicDim) -> ProjectedExpr<SymbolicExpr> {
    constant(
        dimension
            .expression()
            .clone()
            .maximum(SymbolicExpr::constant(1)),
    )
}

fn zero() -> ProjectedExpr<SymbolicExpr> {
    ProjectedExpr::Binary {
        operation: Binary::Mul,
        lhs: std::sync::Arc::new(ProjectedExpr::Linear),
        rhs: std::sync::Arc::new(constant(SymbolicExpr::constant(0))),
    }
}

fn constant(value: SymbolicExpr) -> ProjectedExpr<SymbolicExpr> {
    ProjectedExpr::Constant(value)
}

fn constant_usize(
    value: usize,
    context: &'static str,
) -> Result<ProjectedExpr<SymbolicExpr>, ReplayError> {
    Ok(constant(SymbolicExpr::constant(
        i64::try_from(value)
            .map_err(|_| ReplayError::Symbolic(format!("{context} exceeds i64")))?,
    )))
}

fn binary(
    operation: Binary,
    lhs: ProjectedExpr<SymbolicExpr>,
    rhs: ProjectedExpr<SymbolicExpr>,
) -> Result<ProjectedExpr<SymbolicExpr>, ReplayError> {
    ProjectedExpr::binary(operation, lhs, rhs)
        .map_err(|error| ReplayError::Symbolic(error.to_string()))
}

fn is_one(dimension: &SymbolicDim) -> Result<bool, ReplayError> {
    Ok(dimension
        .expression()
        .bounds()
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?
        .constant()
        == Some(1))
}

fn is_projected_one(expression: &ProjectedExpr<SymbolicExpr>) -> Result<bool, ReplayError> {
    let ProjectedExpr::Constant(expression) = expression else {
        return Ok(false);
    };
    Ok(expression
        .bounds()
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?
        .constant()
        == Some(1))
}

struct ProjectedBounds {
    minimum: SymbolicExpr,
    maximum: SymbolicExpr,
}

fn projected_bounds(
    expression: &ProjectedExpr<SymbolicExpr>,
    linear_max: &SymbolicExpr,
    positive_dimensions: &[SymbolicExpr],
) -> Result<ProjectedBounds, ReplayError> {
    let bounds = match expression {
        ProjectedExpr::Linear => ProjectedBounds {
            minimum: SymbolicExpr::constant(0),
            maximum: linear_max.clone(),
        },
        ProjectedExpr::Constant(value) => ProjectedBounds {
            minimum: value.clone(),
            maximum: value.clone(),
        },
        ProjectedExpr::Binary {
            operation,
            lhs,
            rhs,
        } => {
            let rhs_contains_linear = rhs.contains_linear();
            let lhs = projected_bounds(lhs, linear_max, positive_dimensions)?;
            let rhs = projected_bounds(rhs, linear_max, positive_dimensions)?;
            match operation {
                Binary::Add => ProjectedBounds {
                    minimum: lhs.minimum + rhs.minimum,
                    maximum: lhs.maximum + rhs.maximum,
                },
                Binary::Sub => ProjectedBounds {
                    minimum: lhs.minimum - rhs.maximum,
                    maximum: lhs.maximum - rhs.minimum,
                },
                Binary::Mul => {
                    // Choose the monotone envelope in the active, nonempty
                    // domain, but retain the guarded endpoints themselves so
                    // FloorDiv/Mod remain defined over empty bindings.
                    let lhs_nonnegative =
                        proven_nonnegative_when_nonempty(&lhs.minimum, positive_dimensions)?;
                    let rhs_nonnegative =
                        proven_nonnegative_when_nonempty(&rhs.minimum, positive_dimensions)?;
                    let lhs_nonpositive = proven_nonnegative_when_nonempty(
                        &(-lhs.maximum.clone()),
                        positive_dimensions,
                    )?;
                    let rhs_nonpositive = proven_nonnegative_when_nonempty(
                        &(-rhs.maximum.clone()),
                        positive_dimensions,
                    )?;
                    match (
                        lhs_nonnegative,
                        lhs_nonpositive,
                        rhs_nonnegative,
                        rhs_nonpositive,
                    ) {
                        (true, _, true, _) => ProjectedBounds {
                            minimum: lhs.minimum * rhs.minimum,
                            maximum: lhs.maximum * rhs.maximum,
                        },
                        (true, _, _, true) => ProjectedBounds {
                            minimum: lhs.maximum * rhs.minimum,
                            maximum: lhs.minimum * rhs.maximum,
                        },
                        (_, true, true, _) => ProjectedBounds {
                            minimum: lhs.minimum * rhs.maximum,
                            maximum: lhs.maximum * rhs.minimum,
                        },
                        (_, true, _, true) => ProjectedBounds {
                            minimum: lhs.maximum * rhs.maximum,
                            maximum: lhs.minimum * rhs.minimum,
                        },
                        _ => {
                            let products = [
                                lhs.minimum.clone() * rhs.minimum.clone(),
                                lhs.minimum * rhs.maximum.clone(),
                                lhs.maximum.clone() * rhs.minimum,
                                lhs.maximum * rhs.maximum,
                            ];
                            let mut minimum = products[0].clone();
                            let mut maximum = products[0].clone();
                            for product in products.into_iter().skip(1) {
                                minimum = minimum.minimum(product.clone());
                                maximum = maximum.maximum(product);
                            }
                            ProjectedBounds { minimum, maximum }
                        }
                    }
                }
                Binary::FloorDiv => {
                    if rhs_contains_linear
                        || !super::symbolic_view::proven_nonnegative(
                            &(rhs.minimum.clone() - SymbolicExpr::constant(1)),
                        )?
                        || !super::symbolic_view::proven_nonnegative(&lhs.minimum)?
                    {
                        return Err(ReplayError::Unsupported(
                            "symbolic projected divisor or dividend is unproven".into(),
                        ));
                    }
                    ProjectedBounds {
                        minimum: lhs
                            .minimum
                            .try_floor_div(rhs.maximum)
                            .map_err(|error| ReplayError::Symbolic(error.to_string()))?,
                        maximum: lhs
                            .maximum
                            .try_floor_div(rhs.minimum)
                            .map_err(|error| ReplayError::Symbolic(error.to_string()))?,
                    }
                }
                Binary::Mod => {
                    if rhs_contains_linear
                        || !super::symbolic_view::proven_nonnegative(
                            &(rhs.minimum.clone() - SymbolicExpr::constant(1)),
                        )?
                        || !super::symbolic_view::proven_nonnegative(&lhs.minimum)?
                    {
                        return Err(ReplayError::Unsupported(
                            "symbolic projected modulo domain is unproven".into(),
                        ));
                    }
                    ProjectedBounds {
                        minimum: SymbolicExpr::constant(0),
                        maximum: rhs.maximum - SymbolicExpr::constant(1),
                    }
                }
                _ => unreachable!("ProjectedExpr construction admitted the operation"),
            }
        }
    };
    bounds
        .minimum
        .bounds()
        .and_then(|_| bounds.maximum.bounds())
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
    Ok(bounds)
}

fn proven_nonnegative_when_nonempty(
    expression: &SymbolicExpr,
    positive_dimensions: &[SymbolicExpr],
) -> Result<bool, ReplayError> {
    let expression =
        super::symbolic_view::simplified(assume_nonempty(expression, positive_dimensions)?)?;
    super::symbolic_view::proven_nonnegative(&expression)
}

fn assume_nonempty(
    value: &SymbolicExpr,
    positive_dimensions: &[SymbolicExpr],
) -> Result<SymbolicExpr, ReplayError> {
    use SymbolicExpr::*;
    let recurse = |value: &SymbolicExpr| assume_nonempty(value, positive_dimensions);
    Ok(match value {
        Const(_) | Var(_) => value.clone(),
        Add(values) => Add(values.iter().map(recurse).collect::<Result<_, _>>()?),
        Mul(values) => Mul(values.iter().map(recurse).collect::<Result<_, _>>()?),
        Neg(value) => Neg(Box::new(recurse(value)?)),
        FloorDiv(lhs, rhs) => FloorDiv(Box::new(recurse(lhs)?), Box::new(recurse(rhs)?)),
        Mod(lhs, rhs) => Mod(Box::new(recurse(lhs)?), Box::new(recurse(rhs)?)),
        Min(lhs, rhs) => Min(Box::new(recurse(lhs)?), Box::new(recurse(rhs)?)),
        Max(lhs, rhs) => {
            for (candidate, one) in [(lhs.as_ref(), rhs.as_ref()), (rhs.as_ref(), lhs.as_ref())] {
                if one
                    .bounds()
                    .map_err(|error| ReplayError::Symbolic(error.to_string()))?
                    .constant()
                    == Some(1)
                    && positive_when_nonempty(candidate, positive_dimensions)?
                {
                    return recurse(candidate);
                }
            }
            Max(Box::new(recurse(lhs)?), Box::new(recurse(rhs)?))
        }
        Eq(lhs, rhs) => Eq(Box::new(recurse(lhs)?), Box::new(recurse(rhs)?)),
        Lt(lhs, rhs) => Lt(Box::new(recurse(lhs)?), Box::new(recurse(rhs)?)),
        Le(lhs, rhs) => Le(Box::new(recurse(lhs)?), Box::new(recurse(rhs)?)),
        And(lhs, rhs) => And(Box::new(recurse(lhs)?), Box::new(recurse(rhs)?)),
        Or(lhs, rhs) => Or(Box::new(recurse(lhs)?), Box::new(recurse(rhs)?)),
        Not(value) => Not(Box::new(recurse(value)?)),
        Where(condition, yes, no) => Where(
            Box::new(recurse(condition)?),
            Box::new(recurse(yes)?),
            Box::new(recurse(no)?),
        ),
    })
}

fn positive_when_nonempty(
    expression: &SymbolicExpr,
    positive_dimensions: &[SymbolicExpr],
) -> Result<bool, ReplayError> {
    if expression
        .bounds()
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?
        .min
        > 0
    {
        return Ok(true);
    }
    for dimension in positive_dimensions {
        if super::symbolic_view::proven_equal(expression, dimension)? {
            return Ok(true);
        }
    }
    match expression {
        SymbolicExpr::Mul(factors) => {
            for factor in factors {
                if !positive_when_nonempty(factor, positive_dimensions)? {
                    return Ok(false);
                }
            }
            Ok(!factors.is_empty())
        }
        _ => Ok(false),
    }
}

fn validate_shape(shape: &SymbolicShape) -> Result<(), ReplayError> {
    for dimension in shape.dims() {
        if dimension
            .expression()
            .bounds()
            .map_err(|error| ReplayError::Symbolic(error.to_string()))?
            .min
            < 0
        {
            return Err(ReplayError::Symbolic(
                "symbolic projected dimension may be negative".into(),
            ));
        }
    }
    shape
        .numel()
        .and_then(|value| value.bounds())
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
    Ok(())
}

fn nonempty_source_is_proven(
    source: &SymbolicShape,
    output: &SymbolicShape,
    source_elements: &SymbolicExpr,
    output_elements: &SymbolicExpr,
) -> Result<bool, ReplayError> {
    let source_bounds = source_elements
        .bounds()
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
    let output_bounds = output_elements
        .bounds()
        .map_err(|error| ReplayError::Symbolic(error.to_string()))?;
    if source_bounds.min > 0 || output_bounds.max == 0 {
        return Ok(true);
    }
    if super::symbolic_view::proven_equal(source_elements, output_elements)? {
        return Ok(true);
    }
    for source_dim in source.dims() {
        if source_dim
            .expression()
            .bounds()
            .map_err(|error| ReplayError::Symbolic(error.to_string()))?
            .min
            > 0
        {
            continue;
        }
        let mut implied = false;
        for output_dim in output.dims() {
            if super::symbolic_view::proven_equal(source_dim.expression(), output_dim.expression())?
            {
                implied = true;
                break;
            }
        }
        if !implied {
            return Ok(false);
        }
    }
    Ok(true)
}

fn shape(
    shapes: &BTreeMap<NodeId, SymbolicShape>,
    node: NodeId,
) -> Result<&SymbolicShape, ReplayError> {
    shapes
        .get(&node)
        .ok_or_else(|| ReplayError::Symbolic("symbolic projected shape is absent".into()))
}

fn bind_shape(
    shape: &SymbolicShape,
    environment: &BTreeMap<SymbolicVar, i64>,
) -> Result<Shape, ReplayError> {
    shape
        .dims()
        .iter()
        .map(|dimension| {
            usize::try_from(evaluate(dimension.expression(), environment)?).map_err(|_| {
                ReplayError::Symbolic("symbolic projected dimension is not usize".into())
            })
        })
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
        .filter_map(|variable| {
            environment
                .get(&variable)
                .copied()
                .map(|value| (variable, value))
        })
        .collect();
    expression
        .evaluate(&projected)
        .map_err(|error| ReplayError::Symbolic(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbolic_recomposition_requires_the_complete_iteration_domain_to_fit() {
        let extent = SymbolicExpr::variable("extent", 0, 5).unwrap();
        let shape = SymbolicShape::new(vec![1usize.into(), extent.clone().into(), 4usize.into()]);
        let coordinates = decompose(&shape, ProjectedExpr::Linear).unwrap();
        assert!(matches!(
            coordinates.last(),
            Some(ProjectedExpr::Binary {
                operation: Binary::Mod,
                lhs,
                ..
            }) if lhs.as_ref() == &ProjectedExpr::Linear
        ));
        assert_eq!(
            linearize(&shape, &coordinates, &shape).unwrap(),
            ProjectedExpr::Linear
        );

        let broadcast = SymbolicShape::new(vec![
            3usize.into(),
            1usize.into(),
            extent.into(),
            4usize.into(),
        ]);
        assert_ne!(
            linearize(&shape, &coordinates, &broadcast).unwrap(),
            ProjectedExpr::Linear
        );
    }

    #[test]
    fn template_matching_uses_the_authenticated_output_extent_canonical_form() {
        let expression = binary(
            Binary::Mod,
            ProjectedExpr::Linear,
            constant(SymbolicExpr::constant(8)),
        )
        .unwrap();
        let map = SymbolicProjectedIndexMap {
            source_shape: SymbolicShape::new(vec![8usize.into()]),
            output_shape: SymbolicShape::new(vec![8usize.into()]),
            expression,
        };
        let ty = crate::UType::scalar(crate::DType::F32);
        let address = crate::UOp::from_operation(
            crate::Operation::DefineGlobal(crate::AddressValue {
                space: crate::AddressSpace::Global,
                name: "b7".into(),
                element: ty,
            }),
            Some(ty),
            vec![],
        );
        let index = crate::UOp::from_operation(
            crate::Operation::Index(crate::IndexValue::Buffer {
                buffer: 7,
                elements: 8,
                input_shape: Shape::from([8]),
                output_shape: Shape::from([8]),
                addressing: crate::IndexAddressing::Projected,
            }),
            Some(ty),
            vec![address, ProjectedExpr::Linear.to_uop(8).unwrap()],
        );

        let environment = BTreeMap::new();
        assert!(map.matches_template(&index, &environment).unwrap());
        let specialized = map.specialize_uop(&environment).unwrap();
        assert_eq!(
            ProjectedIndexPlan::from_index(&crate::UOp::from_operation(
                index.operation().clone(),
                index.ty(),
                vec![index.sources()[0].clone(), specialized],
            ))
            .unwrap()
            .expression,
            ProjectedExpr::Linear
        );
    }
}
