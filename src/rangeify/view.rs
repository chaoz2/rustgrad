use crate::{AffineView, Graph, NodeId, Op, Shape, ViewMap};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Shape, Slice};
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
}
