use crate::uop::{Binary, Operation, UOp};
use crate::{AffineView, DType, Graph, NodeId, Op, Shape, UType, ViewMap};
use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RangeifiedView {
    pub source: NodeId,
    pub view: AffineView,
    pub cache_key: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RangeifiedProjection {
    pub source: NodeId,
    pub expression: UOp,
    pub predicate: Option<UOp>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RangeifyError {
    Unsupported(NodeId),
    Invalid,
}
impl fmt::Display for RangeifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rangeify error: {self:?}")
    }
}
impl std::error::Error for RangeifyError {}

/// Resolves the statically provable shrink subset into one source storage map.
/// Computed producers deliberately remain an explicit materialization boundary.
pub(crate) fn static_view(graph: &Graph, node: NodeId) -> Result<RangeifiedView, RangeifyError> {
    fn go(g: &Graph, n: NodeId) -> Result<(NodeId, AffineView), RangeifyError> {
        let n = g
            .contiguous_backward_owner(n)
            .map_err(|_| RangeifyError::Invalid)?;
        match g.op(n).map_err(|_| RangeifyError::Invalid)? {
            Op::Input { .. } | Op::Constant(_) => {
                let s = g.shape(n).map_err(|_| RangeifyError::Invalid)?.clone();
                Ok((n, AffineView::from(ViewMap::identity(s))))
            }
            Op::Shrink { input, bounds } => {
                let (src, v) = go(g, *input)?;
                Ok((src, v.shrink(bounds).map_err(|_| RangeifyError::Invalid)?))
            }
            Op::Reshape { input, shape } => {
                let (src, view) = go(g, *input)?;
                Ok((
                    src,
                    view.reshape_read(shape.clone())
                        .map_err(|_| RangeifyError::Unsupported(n))?,
                ))
            }
            Op::Permute { input, axes } => {
                let (src, view) = go(g, *input)?;
                Ok((src, view.permute(axes).map_err(|_| RangeifyError::Invalid)?))
            }
            Op::Expand { input, shape } => {
                let (src, view) = go(g, *input)?;
                Ok((
                    src,
                    view.expand(shape.clone())
                        .map_err(|_| RangeifyError::Invalid)?,
                ))
            }
            Op::Stride { input, slices } => {
                let (src, view) = go(g, *input)?;
                let normalized = slices
                    .iter()
                    .zip(view.logical_shape.dims())
                    .enumerate()
                    .map(|(axis, (slice, dim))| {
                        let (start, _, step, length) =
                            crate::ir::normalized_slice(*dim, *slice, axis)
                                .map_err(|_| RangeifyError::Invalid)?;
                        Ok((start, step, length))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut out = view;
                let mut dims = out.logical_shape.dims().to_vec();
                for (axis, (start, step, length)) in normalized.into_iter().enumerate() {
                    let start = i64::try_from(start).map_err(|_| RangeifyError::Invalid)?;
                    let step = i64::try_from(step).map_err(|_| RangeifyError::Invalid)?;
                    out.offset = out
                        .offset
                        .checked_add(
                            start
                                .checked_mul(out.strides[axis])
                                .ok_or(RangeifyError::Invalid)?,
                        )
                        .ok_or(RangeifyError::Invalid)?;
                    out.strides[axis] = out.strides[axis]
                        .checked_mul(step)
                        .ok_or(RangeifyError::Invalid)?;
                    dims[axis] = length;
                }
                out.logical_shape = crate::Shape::new(dims);
                out.validate_read().map_err(|_| RangeifyError::Invalid)?;
                Ok((src, out))
            }
            _ => Err(RangeifyError::Unsupported(n)),
        }
    }
    let (source, view) = go(graph, node)?;
    let mut h = DefaultHasher::new();
    source.hash(&mut h);
    view.hash(&mut h);
    Ok(RangeifiedView {
        source,
        view,
        cache_key: h.finish(),
    })
}

fn computed_view_seeded(
    graph: &Graph,
    node: NodeId,
    seed: Option<(NodeId, &AffineView)>,
) -> Result<(NodeId, AffineView), RangeifyError> {
    fn go(
        g: &Graph,
        n: NodeId,
        seed: Option<(NodeId, &AffineView)>,
    ) -> Result<(NodeId, AffineView), RangeifyError> {
        let n = g
            .contiguous_backward_owner(n)
            .map_err(|_| RangeifyError::Invalid)?;
        if let Some((source, view)) = seed
            && n == source
        {
            return Ok((source, view.clone()));
        }
        match g.op(n).map_err(|_| RangeifyError::Invalid)? {
            Op::Shrink { input, bounds } => {
                let (source, view) = go(g, *input, seed)?;
                Ok((
                    source,
                    view.shrink(bounds).map_err(|_| RangeifyError::Invalid)?,
                ))
            }
            Op::Reshape { input, shape } => {
                let (source, view) = go(g, *input, seed)?;
                Ok((
                    source,
                    view.reshape_read(shape.clone())
                        .map_err(|_| RangeifyError::Unsupported(n))?,
                ))
            }
            Op::Permute { input, axes } => {
                let (source, view) = go(g, *input, seed)?;
                Ok((
                    source,
                    view.permute(axes).map_err(|_| RangeifyError::Invalid)?,
                ))
            }
            Op::Expand { input, shape } => {
                let (source, view) = go(g, *input, seed)?;
                Ok((
                    source,
                    view.expand(shape.clone())
                        .map_err(|_| RangeifyError::Invalid)?,
                ))
            }
            Op::Stride { input, slices } => {
                let (source, mut view) = go(g, *input, seed)?;
                let normalized = slices
                    .iter()
                    .zip(view.logical_shape.dims())
                    .enumerate()
                    .map(|(axis, (slice, dim))| {
                        let (start, _, step, length) =
                            crate::ir::normalized_slice(*dim, *slice, axis)
                                .map_err(|_| RangeifyError::Invalid)?;
                        Ok((start, step, length))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut dims = view.logical_shape.dims().to_vec();
                for (axis, (start, step, length)) in normalized.into_iter().enumerate() {
                    let start = i64::try_from(start).map_err(|_| RangeifyError::Invalid)?;
                    let step = i64::try_from(step).map_err(|_| RangeifyError::Invalid)?;
                    view.offset = view
                        .offset
                        .checked_add(
                            start
                                .checked_mul(view.strides[axis])
                                .ok_or(RangeifyError::Invalid)?,
                        )
                        .ok_or(RangeifyError::Invalid)?;
                    view.strides[axis] = view.strides[axis]
                        .checked_mul(step)
                        .ok_or(RangeifyError::Invalid)?;
                    dims[axis] = length;
                }
                view.logical_shape = crate::Shape::new(dims);
                view.validate_read().map_err(|_| RangeifyError::Invalid)?;
                Ok((source, view))
            }
            // Stop at the first non-view node. It must be a computed producer;
            // input and constant views are already handled by static_view.
            Op::Input { .. } | Op::Constant(_) => Err(RangeifyError::Unsupported(n)),
            _ if seed.is_none() => {
                let shape = g.shape(n).map_err(|_| RangeifyError::Invalid)?.clone();
                Ok((n, AffineView::from(ViewMap::identity(shape))))
            }
            _ => Err(RangeifyError::Unsupported(n)),
        }
    }
    go(graph, node, seed)
}

/// Resolves a static view whose storage source is a pure computed producer.
/// Source-backed views deliberately stay on the ordinary load-addressing path;
/// this helper exists only for the explicit owned materialization boundary.
pub(crate) fn computed_view(graph: &Graph, node: NodeId) -> Result<RangeifiedView, RangeifyError> {
    let (source, view) = computed_view_seeded(graph, node, None)?;
    let mut h = DefaultHasher::new();
    source.hash(&mut h);
    view.hash(&mut h);
    Ok(RangeifiedView {
        source,
        view,
        cache_key: h.finish(),
    })
}

/// Composes one right-aligned producer-leaf broadcast with an already checked
/// computed movement chain. The result is a direct logical-output-to-leaf
/// storage map. Reshapes that would require coordinate div/mod decomposition
/// remain non-affine and fail closed through `reshape_read`.
pub(crate) fn computed_broadcast_view(
    graph: &Graph,
    node: NodeId,
    producer: NodeId,
    leaf_shape: Shape,
) -> Result<AffineView, RangeifyError> {
    let producer_shape = graph
        .shape(producer)
        .map_err(|_| RangeifyError::Invalid)?
        .clone();
    let seed = AffineView::identity(leaf_shape)
        .expand(producer_shape)
        .map_err(|_| RangeifyError::Unsupported(producer))?;
    seed.validate_read().map_err(|_| RangeifyError::Invalid)?;
    let (source, view) = computed_view_seeded(graph, node, Some((producer, &seed)))?;
    if source != producer {
        return Err(RangeifyError::Invalid);
    }
    view.validate_read().map_err(|_| RangeifyError::Invalid)?;
    Ok(view)
}

fn iconstant(value: i64) -> UOp {
    UOp::constant(value, UType::scalar(DType::I64))
}

fn ibinary(operation: Binary, lhs: UOp, rhs: UOp) -> UOp {
    UOp::from_operation(
        Operation::Binary(operation),
        Some(UType::scalar(DType::I64)),
        vec![lhs, rhs],
    )
}

fn compare(operation: crate::CompareOp, lhs: UOp, rhs: UOp) -> UOp {
    UOp::from_operation(
        Operation::GraphCompare(operation),
        Some(UType::scalar(DType::Bool)),
        vec![lhs, rhs],
    )
}

fn logical_and(lhs: UOp, rhs: UOp) -> UOp {
    let constant = |value: &UOp| match value.operation() {
        Operation::Const(crate::uop::LiteralValue::Scalar {
            dtype: DType::Bool,
            bits,
        }) if value.sources().is_empty() && *bits <= 1 => Some(*bits != 0),
        _ => None,
    };
    match (constant(&lhs), constant(&rhs)) {
        (Some(false), _) | (_, Some(false)) => return bool_constant(false),
        (Some(true), _) => return rhs,
        (_, Some(true)) => return lhs,
        _ => {}
    }
    UOp::from_operation(
        Operation::GraphLogical(crate::LogicalOp::And),
        Some(UType::scalar(DType::Bool)),
        vec![lhs, rhs],
    )
}

fn bool_constant(value: bool) -> UOp {
    UOp::scalar_constant(DType::Bool, u64::from(value), UType::scalar(DType::Bool))
}

pub(crate) fn is_constant_zero_pad(graph: &Graph, node: NodeId) -> bool {
    matches!(
        graph.op(node),
        Ok(Op::Pad {
            fill: crate::Scalar::Bool(false) | crate::Scalar::I(0) | crate::Scalar::U(0),
            ..
        })
    ) || matches!(
        graph.op(node),
        Ok(Op::Pad {
            fill: crate::Scalar::F(value),
            ..
        }) if value.to_bits() == 0
    )
}

fn checked_i64(value: usize) -> Result<i64, RangeifyError> {
    i64::try_from(value).map_err(|_| RangeifyError::Invalid)
}

fn zero_from(range: &UOp) -> UOp {
    ibinary(Binary::Mul, range.clone(), iconstant(0))
}

fn range_extent(range: &UOp) -> Result<usize, RangeifyError> {
    let Operation::Range(0) = range.operation() else {
        return Err(RangeifyError::Invalid);
    };
    let [bound] = range.sources() else {
        return Err(RangeifyError::Invalid);
    };
    let Operation::Const(crate::uop::LiteralValue::Int(bound)) = bound.operation() else {
        return Err(RangeifyError::Invalid);
    };
    usize::try_from(*bound).map_err(|_| RangeifyError::Invalid)
}

fn canonicalize_projected_address(
    expression: &UOp,
    output_elements: usize,
    source_elements: usize,
) -> Result<UOp, RangeifyError> {
    crate::projected_index::canonicalize_graph_projected_address(
        expression,
        output_elements,
        source_elements,
    )
    .map_err(|_| RangeifyError::Invalid)
}

fn canonicalize_projected_predicate(
    predicate: &UOp,
    output_elements: usize,
) -> Result<UOp, RangeifyError> {
    crate::projected_index::canonicalize_graph_projected_predicate(predicate, output_elements)
        .map_err(|_| RangeifyError::Invalid)
}

fn logical_coordinates(
    logical: &Shape,
    output: &Shape,
    range: &UOp,
) -> Result<Vec<UOp>, RangeifyError> {
    if logical.rank() > output.rank() {
        return Err(RangeifyError::Invalid);
    }
    let rank_delta = output.rank() - logical.rank();
    let output_strides = output.contiguous_strides();
    let zero = zero_from(range);
    logical
        .dims()
        .iter()
        .enumerate()
        .map(|(axis, logical_dim)| {
            let output_axis = axis + rank_delta;
            let output_dim = output.dims()[output_axis];
            if *logical_dim != 1 && logical_dim != &output_dim {
                return Err(RangeifyError::Invalid);
            }
            if *logical_dim == 1 || output_dim == 0 {
                return Ok(zero.clone());
            }
            let divisor = output_strides[output_axis];
            let divided = if divisor == 1 {
                range.clone()
            } else {
                ibinary(
                    Binary::FloorDiv,
                    range.clone(),
                    iconstant(checked_i64(divisor)?),
                )
            };
            Ok(if output_dim == 1 {
                zero.clone()
            } else {
                ibinary(Binary::Mod, divided, iconstant(checked_i64(output_dim)?))
            })
        })
        .collect()
}

fn canonicalize_coordinates(
    shape: &Shape,
    coordinates: Vec<UOp>,
    range: &UOp,
) -> Result<Vec<UOp>, RangeifyError> {
    if shape.rank() != coordinates.len() {
        return Err(RangeifyError::Invalid);
    }
    let output_elements = range_extent(range)?;
    let coordinates = coordinates
        .into_iter()
        .zip(shape.dims())
        .map(|(coordinate, axis_extent)| {
            if *axis_extent == 0 {
                // No lane can address a zero axis. The final projected-plan
                // check still requires either a zero output domain or an
                // authenticated all-false predicate before admitting a load.
                return (zero_from(range), 1);
            }
            (coordinate, *axis_extent)
        })
        .collect::<Vec<_>>();
    crate::projected_index::canonicalize_graph_projected_coordinates(&coordinates, output_elements)
        .map_err(|_| RangeifyError::Invalid)
}

fn recomposed_linear(
    shape: &Shape,
    coordinates: &[UOp],
    range: &UOp,
) -> Result<Option<UOp>, RangeifyError> {
    let strides = shape.contiguous_strides();
    // Reshape decomposes one already-authenticated linear address into every
    // source coordinate. A later dense/computed boundary commonly linearizes
    // those same coordinates again. Recover that exact shared numerator before
    // constructing a much larger quotient/remainder tree, but only when the
    // complete consumer range proves the numerator is already in this shape.
    // This deliberately does not collapse a broadcast: its root Range exceeds
    // the logical source extent and therefore fails the checked proof below.
    let mut decomposed = None;
    for ((coordinate, dim), stride) in coordinates.iter().zip(shape.dims()).zip(&strides) {
        if *dim == 0 || *stride == 0 || *dim == 1 {
            let is_zero = matches!(
                (coordinate.operation(), coordinate.sources()),
                (Operation::Const(crate::uop::LiteralValue::Int(0)), [])
            ) || matches!(
                (coordinate.operation(), coordinate.sources()),
                (Operation::Binary(Binary::Mul), [_, rhs])
                    if matches!(rhs.operation(), Operation::Const(crate::uop::LiteralValue::Int(0)))
            );
            if !is_zero {
                return Ok(None);
            }
            continue;
        }
        let (Operation::Binary(Binary::Mod), [divided, modulus]) =
            (coordinate.operation(), coordinate.sources())
        else {
            return Ok(None);
        };
        if !matches!(
            modulus.operation(),
            Operation::Const(crate::uop::LiteralValue::Int(value))
                if usize::try_from(*value).ok() == Some(*dim)
        ) {
            return Ok(None);
        }
        let candidate = if *stride == 1 {
            divided
        } else {
            let (Operation::Binary(Binary::FloorDiv), [candidate, divisor]) =
                (divided.operation(), divided.sources())
            else {
                return Ok(None);
            };
            if !matches!(
                divisor.operation(),
                Operation::Const(crate::uop::LiteralValue::Int(value))
                    if usize::try_from(*value).ok() == Some(*stride)
            ) {
                return Ok(None);
            }
            candidate
        };
        if decomposed
            .as_ref()
            .is_some_and(|prior: &UOp| prior.node_identity() != candidate.node_identity())
        {
            return Ok(None);
        }
        decomposed = Some(candidate.clone());
    }
    let Some(decomposed) = decomposed else {
        return Ok(None);
    };
    Ok(canonicalize_projected_address(
        &decomposed,
        range_extent(range)?,
        shape.numel().map_err(|_| RangeifyError::Invalid)?,
    )
    .ok())
}

fn linearize_coordinates(
    shape: &Shape,
    coordinates: &[UOp],
    range: &UOp,
) -> Result<UOp, RangeifyError> {
    if coordinates.len() != shape.rank() {
        return Err(RangeifyError::Invalid);
    }
    if let Some(recomposed) = recomposed_linear(shape, coordinates, range)? {
        return Ok(recomposed);
    }
    let strides = shape.contiguous_strides();
    let mut expression = zero_from(range);
    for (coordinate, stride) in coordinates.iter().zip(strides) {
        let term = if stride == 1 {
            coordinate.clone()
        } else {
            ibinary(
                Binary::Mul,
                coordinate.clone(),
                iconstant(checked_i64(stride)?),
            )
        };
        expression = ibinary(Binary::Add, expression, term);
    }
    Ok(expression)
}

fn decompose_linear(shape: &Shape, linear: UOp, range: &UOp) -> Result<Vec<UOp>, RangeifyError> {
    let strides = shape.contiguous_strides();
    let zero = zero_from(range);
    shape
        .dims()
        .iter()
        .zip(strides)
        .map(|(dim, stride)| {
            if *dim == 0 || stride == 0 {
                return Ok(zero.clone());
            }
            let divided = if stride == 1 {
                linear.clone()
            } else {
                ibinary(
                    Binary::FloorDiv,
                    linear.clone(),
                    iconstant(checked_i64(stride)?),
                )
            };
            Ok(if *dim == 1 {
                zero.clone()
            } else {
                ibinary(Binary::Mod, divided, iconstant(checked_i64(*dim)?))
            })
        })
        .collect()
}

/// Recognizes one literal `Graph::repeat` reshape/expand/reshape stage and
/// maps its collapsed coordinate back to the repeated source axis.  The
/// shapes authenticate the periodic map; unrelated expand/reshape chains keep
/// using the general projected-address path below.
fn periodic_repeat_source(
    graph: &Graph,
    node: NodeId,
    coordinates: &[UOp],
    range: &UOp,
) -> Result<Option<(NodeId, Vec<UOp>)>, RangeifyError> {
    let Op::Reshape {
        input: expanded, ..
    } = graph.op(node).map_err(|_| RangeifyError::Invalid)?
    else {
        return Ok(None);
    };
    let Op::Expand {
        input: unsqueezed, ..
    } = graph.op(*expanded).map_err(|_| RangeifyError::Invalid)?
    else {
        return Ok(None);
    };
    let Op::Reshape { input: source, .. } =
        graph.op(*unsqueezed).map_err(|_| RangeifyError::Invalid)?
    else {
        return Ok(None);
    };

    let source_shape = graph.shape(*source).map_err(|_| RangeifyError::Invalid)?;
    let unsqueezed_shape = graph
        .shape(*unsqueezed)
        .map_err(|_| RangeifyError::Invalid)?;
    let expanded_shape = graph.shape(*expanded).map_err(|_| RangeifyError::Invalid)?;
    let collapsed_shape = graph.shape(node).map_err(|_| RangeifyError::Invalid)?;
    if coordinates.len() != collapsed_shape.rank()
        || collapsed_shape.rank() != source_shape.rank()
        || unsqueezed_shape.rank() != source_shape.rank() + 1
        || expanded_shape.rank() != unsqueezed_shape.rank()
    {
        return Ok(None);
    }

    let changed = unsqueezed_shape
        .dims()
        .iter()
        .zip(expanded_shape.dims())
        .enumerate()
        .filter_map(|(axis, (before, after))| (before != after).then_some(axis))
        .collect::<Vec<_>>();
    let [axis] = changed.as_slice() else {
        return Ok(None);
    };
    if *axis >= source_shape.rank() || unsqueezed_shape.dims()[*axis] != 1 {
        return Ok(None);
    }
    let repeat = expanded_shape.dims()[*axis];
    if repeat == 1 {
        return Ok(None);
    }

    let mut expected_unsqueezed = source_shape.dims().to_vec();
    expected_unsqueezed.insert(*axis, 1);
    let mut expected_expanded = expected_unsqueezed.clone();
    expected_expanded[*axis] = repeat;
    let mut expected_collapsed = source_shape.dims().to_vec();
    expected_collapsed[*axis] = expected_collapsed[*axis]
        .checked_mul(repeat)
        .ok_or(RangeifyError::Invalid)?;
    if unsqueezed_shape.dims() != expected_unsqueezed
        || expanded_shape.dims() != expected_expanded
        || collapsed_shape.dims() != expected_collapsed
    {
        return Ok(None);
    }

    let source_extent = source_shape.dims()[*axis];
    let mut source_coordinates = coordinates.to_vec();
    source_coordinates[*axis] = if source_extent <= 1 {
        zero_from(range)
    } else {
        let periodic = ibinary(
            Binary::Mod,
            source_coordinates[*axis].clone(),
            iconstant(checked_i64(source_extent)?),
        );
        canonicalize_projected_address(&periodic, range_extent(range)?, source_extent)?
    };
    Ok(Some((*source, source_coordinates)))
}

/// Builds an explicit concrete output-linear to source-linear address for a
/// movement chain that cannot collapse to one AffineView. The chain terminates
/// at the first dense source or computed producer; ownership remains a
/// scheduler decision and this function never invents a materialization.
pub(crate) fn projected_view(
    graph: &Graph,
    node: NodeId,
    output_shape: &Shape,
    range: &UOp,
) -> Result<RangeifiedProjection, RangeifyError> {
    fn go(
        graph: &Graph,
        node: NodeId,
        coordinates: Vec<UOp>,
        range: &UOp,
    ) -> Result<(NodeId, UOp, Option<UOp>), RangeifyError> {
        let node = graph
            .contiguous_backward_owner(node)
            .map_err(|_| RangeifyError::Invalid)?;
        let coordinates = canonicalize_coordinates(
            graph.shape(node).map_err(|_| RangeifyError::Invalid)?,
            coordinates,
            range,
        )?;
        match graph.op(node).map_err(|_| RangeifyError::Invalid)? {
            Op::Shrink { input, bounds } => {
                let coordinates = coordinates
                    .into_iter()
                    .zip(bounds)
                    .map(|(coordinate, (start, _))| {
                        if *start == 0 {
                            Ok(coordinate)
                        } else {
                            Ok(ibinary(
                                Binary::Add,
                                coordinate,
                                iconstant(checked_i64(*start)?),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, RangeifyError>>()?;
                go(graph, *input, coordinates, range)
            }
            Op::Pad { input, padding, .. } if is_constant_zero_pad(graph, node) => {
                let input_shape = graph.shape(*input).map_err(|_| RangeifyError::Invalid)?;
                if input_shape.rank() != coordinates.len() || padding.len() != coordinates.len() {
                    return Err(RangeifyError::Unsupported(node));
                }
                if input_shape.dims().contains(&0) {
                    // A source with no elements has no legal address, but a
                    // canonical-zero Pad can still have a populated output.
                    // Authenticate that case explicitly as an addressless
                    // all-false load while still resolving the real storage
                    // owner through any movement aliases below the Pad.
                    let input_coordinates = vec![zero_from(range); input_shape.rank()];
                    let (source, _, _) = go(graph, *input, input_coordinates, range)?;
                    return Ok((source, zero_from(range), Some(bool_constant(false))));
                }
                let mut valid = None;
                let coordinates = coordinates
                    .into_iter()
                    .zip(input_shape.dims())
                    .zip(padding)
                    .map(|((coordinate, dim), (before, _))| {
                        let before = checked_i64(*before)?;
                        let dim = checked_i64(*dim)?;
                        let end = before.checked_add(dim).ok_or(RangeifyError::Invalid)?;
                        let lower =
                            compare(crate::CompareOp::Ge, coordinate.clone(), iconstant(before));
                        let upper =
                            compare(crate::CompareOp::Lt, coordinate.clone(), iconstant(end));
                        let axis_valid = logical_and(lower, upper);
                        valid = Some(match valid.take() {
                            Some(previous) => logical_and(previous, axis_valid),
                            None => axis_valid,
                        });

                        // The physical address is validated independently of
                        // the lane predicate. Wrap every padded coordinate
                        // into the nonempty source axis so even a false lane
                        // carries an in-bounds, total integer expression.
                        let shift = before
                            .checked_add(dim - 1)
                            .and_then(|value| value.checked_div(dim))
                            .and_then(|value| value.checked_mul(dim))
                            .ok_or(RangeifyError::Invalid)?;
                        let shifted = ibinary(Binary::Add, coordinate, iconstant(shift - before));
                        Ok(ibinary(Binary::Mod, shifted, iconstant(dim)))
                    })
                    .collect::<Result<Vec<_>, RangeifyError>>()?;
                let (source, expression, child_valid) = go(graph, *input, coordinates, range)?;
                let predicate = match (valid, child_valid) {
                    (Some(lhs), Some(rhs)) => Some(logical_and(lhs, rhs)),
                    (Some(value), None) | (None, Some(value)) => Some(value),
                    (None, None) => None,
                };
                Ok((source, expression, predicate))
            }
            Op::Reshape { input, .. } => {
                if let Some((source, coordinates)) =
                    periodic_repeat_source(graph, node, &coordinates, range)?
                {
                    return go(graph, source, coordinates, range);
                }
                let output_shape = graph.shape(node).map_err(|_| RangeifyError::Invalid)?;
                let input_shape = graph.shape(*input).map_err(|_| RangeifyError::Invalid)?;
                let output_elements = range_extent(range)?;
                let linear = linearize_coordinates(output_shape, &coordinates, range)?;
                // A long reshape/decompose chain is a compact shared UOp DAG,
                // but expanding that DAG directly into the closed projected
                // expression tree can multiply identical quotient/remainder
                // subtrees. Authenticate and canonicalize this complete linear
                // map while it is still bounded by the current reshape input.
                let input_elements = input_shape.numel().map_err(|_| RangeifyError::Invalid)?;
                let linear =
                    canonicalize_projected_address(&linear, output_elements, input_elements)?;
                let coordinates = decompose_linear(input_shape, linear, range)?;
                go(graph, *input, coordinates, range)
            }
            Op::Permute { input, axes } => {
                let mut input_coordinates = vec![zero_from(range); axes.len()];
                for (output_axis, input_axis) in axes.iter().copied().enumerate() {
                    input_coordinates[input_axis] = coordinates
                        .get(output_axis)
                        .cloned()
                        .ok_or(RangeifyError::Invalid)?;
                }
                go(graph, *input, input_coordinates, range)
            }
            Op::Expand { input, .. } => {
                let input_shape = graph.shape(*input).map_err(|_| RangeifyError::Invalid)?;
                if input_shape.rank() > coordinates.len() {
                    return Err(RangeifyError::Invalid);
                }
                let delta = coordinates.len() - input_shape.rank();
                let zero = zero_from(range);
                let input_coordinates = input_shape
                    .dims()
                    .iter()
                    .enumerate()
                    .map(|(axis, dim)| {
                        if *dim == 1 {
                            zero.clone()
                        } else {
                            coordinates[axis + delta].clone()
                        }
                    })
                    .collect();
                go(graph, *input, input_coordinates, range)
            }
            Op::Stride { input, slices } => {
                let input_shape = graph.shape(*input).map_err(|_| RangeifyError::Invalid)?;
                let coordinates = coordinates
                    .into_iter()
                    .zip(slices)
                    .zip(input_shape.dims())
                    .enumerate()
                    .map(|(axis, ((coordinate, slice), dim))| {
                        let (start, _, step, _) = crate::ir::normalized_slice(*dim, *slice, axis)
                            .map_err(|_| RangeifyError::Invalid)?;
                        let scaled = if step == 1 {
                            coordinate
                        } else {
                            ibinary(
                                Binary::Mul,
                                coordinate,
                                iconstant(i64::try_from(step).map_err(|_| RangeifyError::Invalid)?),
                            )
                        };
                        Ok(if start == 0 {
                            scaled
                        } else {
                            ibinary(
                                Binary::Add,
                                scaled,
                                iconstant(
                                    i64::try_from(start).map_err(|_| RangeifyError::Invalid)?,
                                ),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, RangeifyError>>()?;
                go(graph, *input, coordinates, range)
            }
            Op::Input { .. } | Op::Constant(_) => {
                let shape = graph.shape(node).map_err(|_| RangeifyError::Invalid)?;
                Ok((
                    node,
                    linearize_coordinates(shape, &coordinates, range)?,
                    None,
                ))
            }
            _ => {
                let shape = graph.shape(node).map_err(|_| RangeifyError::Invalid)?;
                Ok((
                    node,
                    linearize_coordinates(shape, &coordinates, range)?,
                    None,
                ))
            }
        }
    }

    let logical_shape = graph.shape(node).map_err(|_| RangeifyError::Invalid)?;
    let coordinates = logical_coordinates(logical_shape, output_shape, range)?;
    let (source, expression, predicate) = go(graph, node, coordinates, range)?;
    let output_elements = range_extent(range)?;
    let source_elements = graph
        .shape(source)
        .map_err(|_| RangeifyError::Invalid)?
        .numel()
        .map_err(|_| RangeifyError::Invalid)?;
    let expression = canonicalize_projected_address(
        &expression,
        output_elements,
        // A populated Pad over an empty source carries canonical address zero
        // behind an authenticated all-false predicate. Prove the expression
        // itself against the singleton canonical-address domain here; the
        // ordinary Predicated Index validator below remains responsible for
        // admitting that addressless zero-source pair and rejects every
        // predicate other than canonical false.
        source_elements.max(1),
    )?;
    let predicate = predicate
        .map(|predicate| canonicalize_projected_predicate(&predicate, output_elements))
        .transpose()?;
    Ok(RangeifiedProjection {
        source,
        expression,
        predicate,
    })
}

fn projection_for_output(
    graph: &Graph,
    node: NodeId,
    output_shape: &Shape,
) -> Result<RangeifiedProjection, RangeifyError> {
    let extent = output_shape.numel().map_err(|_| RangeifyError::Invalid)?;
    let range = UOp::from_operation(
        Operation::Range(0),
        Some(UType::scalar(DType::I64)),
        vec![iconstant(checked_i64(extent)?)],
    );
    projected_view(graph, node, output_shape, &range)
}

pub(crate) fn projected_source(
    graph: &Graph,
    node: NodeId,
    output_shape: &Shape,
) -> Result<NodeId, RangeifyError> {
    projection_for_output(graph, node, output_shape).map(|projection| projection.source)
}

pub(crate) fn predicated_source(
    graph: &Graph,
    node: NodeId,
    output_shape: &Shape,
) -> Result<NodeId, RangeifyError> {
    let projection = projection_for_output(graph, node, output_shape)?;
    projection
        .predicate
        .is_some()
        .then_some(projection.source)
        .ok_or(RangeifyError::Unsupported(node))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Shape, Slice};

    #[test]
    fn reshape_linear_recomposition_requires_the_complete_domain_to_fit() {
        let shape = Shape::from([2, 4]);
        let bounded_range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![iconstant(8)],
        );
        let bounded = decompose_linear(&shape, bounded_range.clone(), &bounded_range).unwrap();
        assert!(
            recomposed_linear(&shape, &bounded, &bounded_range)
                .unwrap()
                .is_some()
        );

        let broadcast_range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![iconstant(24)],
        );
        let broadcast =
            decompose_linear(&shape, broadcast_range.clone(), &broadcast_range).unwrap();
        assert!(
            recomposed_linear(&shape, &broadcast, &broadcast_range)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn mixed_radix_linearization_recombines_independent_add_terms() {
        let shape = Shape::from([2, 3, 4]);
        let range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![iconstant(24)],
        );
        // Use separately constructed but equivalent roots so the direct
        // shared-numerator fast path cannot hide the mixed-radix fold.
        let equivalent_range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![iconstant(24)],
        );
        let coordinates = vec![
            ibinary(
                Binary::Mod,
                ibinary(Binary::FloorDiv, range.clone(), iconstant(12)),
                iconstant(2),
            ),
            ibinary(
                Binary::Mod,
                ibinary(Binary::FloorDiv, equivalent_range, iconstant(4)),
                iconstant(3),
            ),
            ibinary(Binary::Mod, range.clone(), iconstant(4)),
        ];
        let linear = linearize_coordinates(&shape, &coordinates, &range).unwrap();
        let canonical = canonicalize_projected_address(&linear, 24, 24).unwrap();
        let element = UType::scalar(DType::F32);
        let index = UOp::from_operation(
            Operation::Index(crate::IndexValue::Buffer {
                buffer: 0,
                elements: 24,
                input_shape: shape,
                output_shape: Shape::from([24]),
                addressing: crate::IndexAddressing::Projected,
            }),
            Some(element),
            vec![
                UOp::from_operation(
                    Operation::DefineGlobal(crate::uop::AddressValue {
                        space: crate::uop::AddressSpace::Global,
                        name: "b0".to_owned(),
                        element,
                    }),
                    Some(element),
                    vec![],
                ),
                canonical,
            ],
        );
        let plan = crate::projected_index::ProjectedIndexPlan::from_index(&index).unwrap();
        assert_eq!(
            (0..24)
                .map(|lane| plan.offset(lane).unwrap())
                .collect::<Vec<_>>(),
            (0..24).collect::<Vec<_>>()
        );
    }

    #[test]
    fn nested_shrink_has_stable_offsets() {
        let mut g = Graph::new();
        let x = g.input("x", Shape::from([4, 4]));
        let a = g.shrink(x, vec![(1, 4), (0, 3)]).unwrap();
        let b = g.shrink(a, vec![(1, 2), (1, 3)]).unwrap();
        let p = static_view(&g, b).unwrap();
        assert_eq!(p.source, x);
        assert_eq!(p.view.logical_shape, Shape::from([1, 2]));
        assert_eq!(p.view.element_offset(0).unwrap(), 9);
        assert_eq!(p.cache_key, static_view(&g, b).unwrap().cache_key);
    }

    #[test]
    fn affine_movement_chain_preserves_source_coordinates() {
        let mut graph = Graph::new();
        let input = graph.input("input", [4, 6]);
        let shrink = graph.shrink(input, [(1, 4), (0, 6)]).unwrap();
        let reshape = graph.reshape(shrink, [3, 2, 3]).unwrap();
        let permute = graph.permute(reshape, [1, 0, 2]).unwrap();
        let stride = graph
            .stride(
                permute,
                [
                    Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    Slice {
                        start: None,
                        stop: None,
                        step: 2,
                    },
                    Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                ],
            )
            .unwrap();
        let plan = static_view(&graph, stride).unwrap();
        assert_eq!(plan.source, input);
        assert_eq!(plan.view.source_shape, Shape::from([4, 6]));
        assert_eq!(plan.view.logical_shape, Shape::from([2, 2, 3]));
        assert_eq!(plan.view.strides, vec![3, 12, 1]);
        assert_eq!(plan.view.offset, 6);
        assert_eq!(plan.view.element_offset(0).unwrap(), 6);
        assert_eq!(plan.view.element_offset(9).unwrap(), 21);

        let scalar = graph.input("scalar", [1, 1]);
        let expanded = graph.expand(scalar, [3, 8]).unwrap();
        let splat = static_view(&graph, expanded).unwrap();
        assert_eq!(splat.view.strides, vec![0, 0]);
        assert_eq!(splat.view.element_offset(23).unwrap(), 0);
    }

    #[test]
    fn negative_stride_is_rangeified_as_signed_affine_view() {
        let mut graph = Graph::new();
        let input = graph.input("input", [4]);
        let reverse = graph
            .stride(
                input,
                [Slice {
                    start: None,
                    stop: None,
                    step: -1,
                }],
            )
            .unwrap();
        let view = static_view(&graph, reverse).unwrap().view;
        assert_eq!(view.offset, 3);
        assert_eq!(view.strides, vec![-1]);
        assert_eq!(view.element_offset(0).unwrap(), 3);
        assert_eq!(view.element_offset(3).unwrap(), 0);

        let unsqueezed = graph.reshape(reverse, [1, 4, 1]).unwrap();
        let view = static_view(&graph, unsqueezed).unwrap().view;
        assert_eq!(view.strides, vec![0, -1, 0]);
        assert_eq!(view.element_offset(0).unwrap(), 3);
        assert_eq!(view.element_offset(3).unwrap(), 0);
    }

    #[test]
    fn signed_view_composes_shrink_and_permute_without_unsigned_adapter() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 4]);
        let flipped = graph
            .stride(
                input,
                [
                    Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    Slice {
                        start: None,
                        stop: None,
                        step: -1,
                    },
                ],
            )
            .unwrap();
        let shrunk = graph.shrink(flipped, [(0, 2), (1, 3)]).unwrap();
        let permuted = graph.permute(shrunk, [1, 0]).unwrap();
        let view = static_view(&graph, permuted).unwrap().view;
        assert_eq!(view.logical_shape, Shape::from([2, 2]));
        assert_eq!(view.offset, 2);
        assert_eq!(view.strides, vec![-1, 4]);
        assert_eq!(view.element_offset(0).unwrap(), 2);
        assert_eq!(view.element_offset(3).unwrap(), 5);
    }

    #[test]
    fn singleton_reshape_preserves_strided_source_and_computed_views() {
        let mut graph = Graph::new();
        let input = graph.input("input", [1, 3]);
        let padded = graph
            .pad(input, [(0, 0), (1, 1)], crate::Scalar::F(0.0))
            .unwrap();
        let window = graph
            .stride(
                padded,
                [
                    Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    Slice {
                        start: Some(1),
                        stop: Some(4),
                        step: 1,
                    },
                ],
            )
            .unwrap();
        let unsqueezed = graph.reshape(window, [1, 3, 1]).unwrap();
        let planned = computed_view(&graph, unsqueezed).unwrap();
        assert_eq!(planned.source, padded);
        assert_eq!(planned.view.logical_shape, Shape::from([1, 3, 1]));
        assert_eq!(planned.view.strides, vec![0, 1, 0]);
        assert_eq!(planned.view.offset, 1);
        assert_eq!(planned.view.element_offset(0).unwrap(), 1);
        assert_eq!(planned.view.element_offset(2).unwrap(), 3);

        let source_window = graph
            .stride(
                input,
                [
                    Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    Slice {
                        start: Some(0),
                        stop: Some(3),
                        step: 2,
                    },
                ],
            )
            .unwrap();
        let source_unsqueezed = graph.reshape(source_window, [1, 2, 1]).unwrap();
        let source_plan = static_view(&graph, source_unsqueezed).unwrap();
        assert_eq!(source_plan.source, input);
        assert_eq!(source_plan.view.strides, vec![0, 2, 0]);
        assert_eq!(source_plan.view.element_offset(1).unwrap(), 2);
    }

    #[test]
    fn computed_broadcast_view_composes_affine_axes_and_rejects_divmod_reshape() {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 3]);
        let row = graph.input("row", [3]);
        let column = graph.input("column", [2, 1]);
        let producer = graph.add(input, row).unwrap();
        let producer = graph.add(producer, column).unwrap();
        let permuted = graph.permute(producer, [1, 0]).unwrap();

        let row_view = computed_broadcast_view(
            &graph,
            permuted,
            producer,
            graph.shape(row).unwrap().clone(),
        )
        .unwrap();
        assert_eq!(row_view.source_shape, Shape::from([3]));
        assert_eq!(row_view.logical_shape, Shape::from([3, 2]));
        assert_eq!(row_view.strides, vec![1, 0]);

        let column_view = computed_broadcast_view(
            &graph,
            permuted,
            producer,
            graph.shape(column).unwrap().clone(),
        )
        .unwrap();
        assert_eq!(column_view.source_shape, Shape::from([2, 1]));
        assert_eq!(column_view.logical_shape, Shape::from([3, 2]));
        assert_eq!(column_view.strides, vec![0, 1]);

        let reshaped = graph.reshape(producer, [3, 2]).unwrap();
        assert!(
            computed_broadcast_view(
                &graph,
                reshaped,
                producer,
                graph.shape(row).unwrap().clone(),
            )
            .is_err()
        );
    }

    #[test]
    fn projected_view_composes_permute_then_non_affine_reshape() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1, 2, 2, 2], DType::F32);
        let producer = graph.square(input).unwrap();
        let permuted = graph.permute(producer, [0, 2, 1, 3]).unwrap();
        let reshaped = graph.reshape(permuted, [1, 2, 4]).unwrap();
        assert!(computed_view(&graph, reshaped).is_err());

        let range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![iconstant(8)],
        );
        let projected = projected_view(&graph, reshaped, &Shape::from([1, 2, 4]), &range).unwrap();
        assert_eq!(projected.source, producer);
        let source_shape = graph.shape(producer).unwrap().clone();
        let ty = UType::scalar(DType::F32);
        let address = UOp::from_operation(
            Operation::DefineGlobal(crate::uop::AddressValue {
                space: crate::uop::AddressSpace::Global,
                name: format!("b{}", producer.index()),
                element: ty,
            }),
            Some(ty),
            vec![],
        );
        let index = UOp::from_operation(
            Operation::Index(crate::IndexValue::Buffer {
                buffer: producer.index() as u64,
                elements: 8,
                input_shape: source_shape,
                output_shape: Shape::from([1, 2, 4]),
                addressing: crate::IndexAddressing::Projected,
            }),
            Some(ty),
            vec![address.clone(), projected.expression],
        );
        let plan = crate::projected_index::ProjectedIndexPlan::from_index(&index).unwrap();
        assert_eq!(
            (0..8)
                .map(|linear| plan.offset(linear).unwrap())
                .collect::<Vec<_>>(),
            vec![0, 1, 4, 5, 2, 3, 6, 7]
        );

        let broadcast_shape = Shape::from([3, 1, 2, 4]);
        let broadcast_range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![iconstant(24)],
        );
        let broadcast =
            projected_view(&graph, reshaped, &broadcast_shape, &broadcast_range).unwrap();
        let broadcast_index = UOp::from_operation(
            Operation::Index(crate::IndexValue::Buffer {
                buffer: producer.index() as u64,
                elements: 8,
                input_shape: graph.shape(producer).unwrap().clone(),
                output_shape: broadcast_shape,
                addressing: crate::IndexAddressing::Projected,
            }),
            Some(ty),
            vec![address, broadcast.expression],
        );
        let broadcast_plan =
            crate::projected_index::ProjectedIndexPlan::from_index(&broadcast_index).unwrap();
        assert_eq!(
            (0..24)
                .map(|linear| broadcast_plan.offset(linear).unwrap())
                .collect::<Vec<_>>(),
            [0, 1, 4, 5, 2, 3, 6, 7].repeat(3)
        );

        let reversed = graph
            .stride(
                reshaped,
                [
                    Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    Slice {
                        start: None,
                        stop: None,
                        step: -1,
                    },
                ],
            )
            .unwrap();
        let projected = projected_view(&graph, reversed, &Shape::from([1, 2, 4]), &range).unwrap();
        let index = UOp::from_operation(
            Operation::Index(crate::IndexValue::Buffer {
                buffer: producer.index() as u64,
                elements: 8,
                input_shape: graph.shape(producer).unwrap().clone(),
                output_shape: Shape::from([1, 2, 4]),
                addressing: crate::IndexAddressing::Projected,
            }),
            Some(ty),
            vec![
                UOp::from_operation(
                    Operation::DefineGlobal(crate::uop::AddressValue {
                        space: crate::uop::AddressSpace::Global,
                        name: format!("b{}", producer.index()),
                        element: ty,
                    }),
                    Some(ty),
                    vec![],
                ),
                projected.expression,
            ],
        );
        let plan = crate::projected_index::ProjectedIndexPlan::from_index(&index).unwrap();
        assert_eq!(
            (0..8)
                .map(|linear| plan.offset(linear).unwrap())
                .collect::<Vec<_>>(),
            vec![5, 4, 1, 0, 7, 6, 3, 2]
        );
    }

    #[test]
    fn projected_view_authenticates_periodic_repeat_stages_and_empty_domains() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::F32);
        let repeated = graph.repeat(input, &[2, 3]).unwrap();
        let output_shape = graph.shape(repeated).unwrap().clone();
        assert_eq!(output_shape, Shape::from([4, 9]));
        let range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![iconstant(36)],
        );
        let projected = projected_view(&graph, repeated, &output_shape, &range).unwrap();
        assert_eq!(projected.source, input);
        assert!(projected.predicate.is_none());
        let ty = UType::scalar(DType::F32);
        let address = UOp::from_operation(
            Operation::DefineGlobal(crate::uop::AddressValue {
                space: crate::uop::AddressSpace::Global,
                name: format!("b{}", input.index()),
                element: ty,
            }),
            Some(ty),
            vec![],
        );
        let index = UOp::from_operation(
            Operation::Index(crate::IndexValue::Buffer {
                buffer: input.index() as u64,
                elements: 6,
                input_shape: Shape::from([2, 3]),
                output_shape,
                addressing: crate::IndexAddressing::Projected,
            }),
            Some(ty),
            vec![address.clone(), projected.expression],
        );
        let plan = crate::projected_index::ProjectedIndexPlan::from_index(&index).unwrap();
        let expected = (0..4)
            .flat_map(|row| (0..9).map(move |column| (row % 2) * 3 + column % 3))
            .collect::<Vec<_>>();
        assert_eq!(
            (0..36)
                .map(|linear| plan.offset(linear).unwrap())
                .collect::<Vec<_>>(),
            expected
        );

        let empty = graph.repeat(input, &[0, 2]).unwrap();
        let empty_shape = graph.shape(empty).unwrap().clone();
        let empty_range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![iconstant(0)],
        );
        let empty_projection = projected_view(&graph, empty, &empty_shape, &empty_range).unwrap();
        assert_eq!(empty_projection.source, input);
        let empty_index = UOp::from_operation(
            Operation::Index(crate::IndexValue::Buffer {
                buffer: input.index() as u64,
                elements: 6,
                input_shape: Shape::from([2, 3]),
                output_shape: empty_shape,
                addressing: crate::IndexAddressing::Projected,
            }),
            Some(ty),
            vec![address, empty_projection.expression],
        );
        assert!(crate::projected_index::ProjectedIndexPlan::from_index(&empty_index).is_ok());
    }

    #[test]
    fn periodic_repeat_normalizes_the_immediate_axis_before_padded_projection() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::F32);
        let padded = graph
            .pad(input, [(1, 2), (2, 1)], crate::Scalar::F(0.0))
            .unwrap();
        let repeated = graph.repeat(padded, &[2, 3]).unwrap();
        let output_shape = graph.shape(repeated).unwrap().clone();
        assert_eq!(output_shape, Shape::from([10, 18]));
        let range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![iconstant(180)],
        );
        let projected = projected_view(&graph, repeated, &output_shape, &range).unwrap();
        assert_eq!(projected.source, input);
        assert!(projected.predicate.is_some());
        let ty = UType::scalar(DType::F32);
        let index = UOp::from_operation(
            Operation::Index(crate::IndexValue::Buffer {
                buffer: input.index() as u64,
                elements: 6,
                input_shape: Shape::from([2, 3]),
                output_shape,
                addressing: crate::IndexAddressing::Predicated,
            }),
            Some(ty),
            vec![
                UOp::from_operation(
                    Operation::DefineGlobal(crate::uop::AddressValue {
                        space: crate::uop::AddressSpace::Global,
                        name: format!("b{}", input.index()),
                        element: ty,
                    }),
                    Some(ty),
                    vec![],
                ),
                projected.expression,
                projected.predicate.unwrap(),
            ],
        );
        let plan = crate::projected_index::ProjectedIndexPlan::from_index(&index).unwrap();
        assert!(!plan.valid(2).unwrap());
        assert!(plan.valid(20).unwrap());
        assert_eq!(plan.offset(20).unwrap(), 0);
        assert!(plan.valid(22).unwrap());
        assert_eq!(plan.offset(22).unwrap(), 2);
        assert!(!plan.valid(23).unwrap());
        assert!(!plan.valid(74).unwrap());
        // Row six wraps by the immediate padded extent five. Using the final
        // repeated extent ten would incorrectly classify this lane as pad.
        assert!(plan.valid(110).unwrap());
        assert_eq!(plan.offset(110).unwrap(), 0);

        let unit = graph.input_dtype("unit", [1], DType::F32);
        let repeated_unit = graph.repeat(unit, &[4]).unwrap();
        let unit_shape = graph.shape(repeated_unit).unwrap().clone();
        let unit_range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![iconstant(4)],
        );
        let unit_projection =
            projected_view(&graph, repeated_unit, &unit_shape, &unit_range).unwrap();
        let unit_index = UOp::from_operation(
            Operation::Index(crate::IndexValue::Buffer {
                buffer: unit.index() as u64,
                elements: 1,
                input_shape: Shape::from([1]),
                output_shape: unit_shape,
                addressing: crate::IndexAddressing::Projected,
            }),
            Some(ty),
            vec![
                UOp::from_operation(
                    Operation::DefineGlobal(crate::uop::AddressValue {
                        space: crate::uop::AddressSpace::Global,
                        name: format!("b{}", unit.index()),
                        element: ty,
                    }),
                    Some(ty),
                    vec![],
                ),
                unit_projection.expression,
            ],
        );
        let unit_plan =
            crate::projected_index::ProjectedIndexPlan::from_index(&unit_index).unwrap();
        assert_eq!(
            (0..4)
                .map(|linear| unit_plan.offset(linear).unwrap())
                .collect::<Vec<_>>(),
            vec![0; 4]
        );

        let empty_source = graph.input_dtype("empty", [0], DType::F32);
        let padded_empty = graph
            .pad(empty_source, [(1, 1)], crate::Scalar::F(0.0))
            .unwrap();
        let repeated_empty = graph.repeat(padded_empty, &[2]).unwrap();
        let empty_output_shape = graph.shape(repeated_empty).unwrap().clone();
        let empty_range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![iconstant(4)],
        );
        let empty_projection =
            projected_view(&graph, repeated_empty, &empty_output_shape, &empty_range).unwrap();
        let empty_index = UOp::from_operation(
            Operation::Index(crate::IndexValue::Buffer {
                buffer: empty_source.index() as u64,
                elements: 0,
                input_shape: Shape::from([0]),
                output_shape: empty_output_shape,
                addressing: crate::IndexAddressing::Predicated,
            }),
            Some(ty),
            vec![
                UOp::from_operation(
                    Operation::DefineGlobal(crate::uop::AddressValue {
                        space: crate::uop::AddressSpace::Global,
                        name: format!("b{}", empty_source.index()),
                        element: ty,
                    }),
                    Some(ty),
                    vec![],
                ),
                empty_projection.expression,
                empty_projection.predicate.unwrap(),
            ],
        );
        let empty_plan =
            crate::projected_index::ProjectedIndexPlan::from_index(&empty_index).unwrap();
        assert!((0..4).all(|linear| !empty_plan.valid(linear).unwrap()));
    }

    #[test]
    fn zero_pad_projects_total_addresses_with_exact_valid_lanes() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let padded = graph
            .pad(input, [(1, 0), (0, 1)], crate::Scalar::F(0.0))
            .unwrap();
        let reversed = graph
            .stride(
                padded,
                [
                    Slice {
                        start: None,
                        stop: None,
                        step: -1,
                    },
                    Slice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                ],
            )
            .unwrap();
        let range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![iconstant(9)],
        );
        let projection = projected_view(&graph, reversed, &Shape::from([3, 3]), &range).unwrap();
        assert_eq!(projection.source, input);
        let ty = UType::scalar(DType::F32);
        let index = UOp::from_operation(
            Operation::Index(crate::IndexValue::Buffer {
                buffer: input.index() as u64,
                elements: 4,
                input_shape: Shape::from([2, 2]),
                output_shape: Shape::from([3, 3]),
                addressing: crate::IndexAddressing::Predicated,
            }),
            Some(ty),
            vec![
                UOp::from_operation(
                    Operation::DefineGlobal(crate::uop::AddressValue {
                        space: crate::uop::AddressSpace::Global,
                        name: format!("b{}", input.index()),
                        element: ty,
                    }),
                    Some(ty),
                    vec![],
                ),
                projection.expression,
                projection.predicate.unwrap(),
            ],
        );
        let plan = crate::projected_index::ProjectedIndexPlan::from_index(&index).unwrap();
        assert_eq!(
            (0..9)
                .map(|lane| plan.valid(lane).unwrap())
                .collect::<Vec<_>>(),
            vec![true, true, false, true, true, false, false, false, false]
        );
        assert!((0..9).all(|lane| plan.offset(lane).unwrap() < 4));

        let empty = graph.input_dtype("empty", [0, 2], DType::F32);
        let empty_pad = graph
            .pad(empty, [(1, 0), (0, 0)], crate::Scalar::I(0))
            .unwrap();
        let empty_view = graph.permute(empty_pad, [1, 0]).unwrap();
        let empty_range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![iconstant(2)],
        );
        let empty_projection =
            projected_view(&graph, empty_view, &Shape::from([2, 1]), &empty_range).unwrap();
        assert_eq!(empty_projection.source, empty);
        let empty_index = UOp::from_operation(
            Operation::Index(crate::IndexValue::Buffer {
                buffer: empty.index() as u64,
                elements: 0,
                input_shape: Shape::from([0, 2]),
                output_shape: Shape::from([2, 1]),
                addressing: crate::IndexAddressing::Predicated,
            }),
            Some(ty),
            vec![
                UOp::from_operation(
                    Operation::DefineGlobal(crate::uop::AddressValue {
                        space: crate::uop::AddressSpace::Global,
                        name: format!("b{}", empty.index()),
                        element: ty,
                    }),
                    Some(ty),
                    vec![],
                ),
                empty_projection.expression,
                empty_projection.predicate.unwrap(),
            ],
        );
        let empty_plan =
            crate::projected_index::ProjectedIndexPlan::from_index(&empty_index).unwrap();
        assert!((0..2).all(|lane| !empty_plan.valid(lane).unwrap()));
        assert!((0..2).all(|lane| empty_plan.offset(lane).is_err()));
    }
}
