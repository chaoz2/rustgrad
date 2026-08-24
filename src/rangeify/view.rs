use crate::{Graph, NodeId, Op, ViewMap};
use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RangeifiedView {
    pub source: NodeId,
    pub view: ViewMap,
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
    fn go(g: &Graph, n: NodeId) -> Result<(NodeId, ViewMap), RangeifyError> {
        match g.op(n).map_err(|_| RangeifyError::Invalid)? {
            Op::Input { .. } | Op::Constant(_) => {
                let s = g.shape(n).map_err(|_| RangeifyError::Invalid)?.clone();
                Ok((n, ViewMap::identity(s)))
            }
            Op::Shrink { input, bounds } => {
                let (src, v) = go(g, *input)?;
                Ok((src, v.shrink(bounds).map_err(|_| RangeifyError::Invalid)?))
            }
            Op::Reshape { input, shape } => {
                let (src, view) = go(g, *input)?;
                Ok((
                    src,
                    view.reshape(shape.clone())
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
                        if start < 0 || step <= 0 {
                            return Err(RangeifyError::Unsupported(n));
                        }
                        Ok((start as usize, step as usize, length))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((
                    src,
                    view.stride_positive(&normalized)
                        .map_err(|_| RangeifyError::Invalid)?,
                ))
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
    fn negative_stride_stays_outside_unsigned_affine_views() {
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
        assert!(matches!(
            static_view(&graph, reverse),
            Err(RangeifyError::Unsupported(node)) if node == reverse
        ));
    }
}
