//! Pure planning for statically-shaped immutable tensor indexing.
//!
//! This module deliberately contains no graph nodes or storage access. A
//! [`StaticIndexPlan`] validates the public index specification, computes its
//! static output shape, and maps one output coordinate back to a source
//! coordinate. The graph and CPU layers consume that map without re-parsing
//! indexing syntax.

use crate::index::DenseIndex;
use crate::{Error, Graph, NodeId, Result, Shape};

/// A statically-known component of an immutable tensor index.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum StaticIndex {
    Integer(isize),
    Slice {
        start: Option<isize>,
        stop: Option<isize>,
        step: isize,
    },
    NewAxis,
    Ellipsis,
    /// An integer index tensor known when the graph is built.
    ///
    /// Multiple advanced indices broadcast by trailing axes. Values normalize
    /// eagerly, so an out-of-bounds index is a construction error.
    Advanced {
        shape: Shape,
        values: Vec<isize>,
    },
}

impl Graph {
    /// Returns overlapping static windows along `dim`.
    ///
    /// The output replaces `dim` with `(window_count, size)`, where
    /// `window_count = (input[dim] - size) / step + 1`.  It is represented as
    /// one immutable static advanced-index operation, so overlapping source
    /// lanes retain the existing static-index gather and reverse-scatter
    /// contracts.
    pub fn unfold(
        &mut self,
        input: NodeId,
        dim: isize,
        size: isize,
        step: isize,
    ) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let specs = unfold_specs(input, &shape, dim, size, step)?;
        self.static_index(input, &specs)
    }
}

fn unfold_specs(
    input: NodeId,
    shape: &Shape,
    dim: isize,
    size: isize,
    step: isize,
) -> Result<Vec<StaticIndex>> {
    if size < 0 {
        return Err(Error::InvalidUnfold {
            reason: "window size must be non-negative",
        });
    }
    if step <= 0 {
        return Err(Error::InvalidUnfold {
            reason: "window step must be positive",
        });
    }
    let axis = normalize_unfold_axis(input, dim, shape.rank())?;
    let size = usize::try_from(size).map_err(|_| Error::InvalidUnfold {
        reason: "window size does not fit usize",
    })?;
    let step = usize::try_from(step).map_err(|_| Error::InvalidUnfold {
        reason: "window step does not fit usize",
    })?;
    let extent = shape.dims()[axis];
    if size > extent {
        return Err(Error::InvalidUnfold {
            reason: "window size exceeds the selected axis",
        });
    }
    let windows = extent
        .checked_sub(size)
        .and_then(|remaining| remaining.checked_div(step))
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    let lanes = windows
        .checked_mul(size)
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    let mut values = Vec::with_capacity(lanes);
    for window in 0..windows {
        let start = window
            .checked_mul(step)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        for offset in 0..size {
            let value = start
                .checked_add(offset)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            values.push(isize::try_from(value).map_err(|_| Error::ShapeOverflow(shape.clone()))?);
        }
    }
    Ok((0..shape.rank())
        .map(|current| {
            if current == axis {
                StaticIndex::Advanced {
                    shape: Shape::new(vec![windows, size]),
                    values: values.clone(),
                }
            } else {
                StaticIndex::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                }
            }
        })
        .collect())
}

fn normalize_unfold_axis(input: NodeId, axis: isize, rank: usize) -> Result<usize> {
    let normalized = if axis < 0 {
        axis.checked_add(rank as isize)
    } else {
        Some(axis)
    };
    normalized
        .and_then(|axis| usize::try_from(axis).ok())
        .filter(|axis| *axis < rank)
        .ok_or(Error::InvalidAxis {
            node: input,
            axis: usize::try_from(axis).unwrap_or(usize::MAX),
            rank,
        })
}

/// A checked diagonal lowering into a permutation followed by static indexing.
/// Keeping this distinct from [`StaticIndexPlan`] preserves that plan's role
/// as the canonical coordinate map while avoiding a second indexing engine.
#[derive(Clone, Debug, Eq, PartialEq)]
struct StaticDiagonalPlan {
    permutation: Vec<usize>,
    specs: Vec<StaticIndex>,
}

impl StaticDiagonalPlan {
    fn new(input: NodeId, shape: &Shape, offset: isize, dim1: isize, dim2: isize) -> Result<Self> {
        let dim1 = normalize_diagonal_axis(input, dim1, shape.rank())?;
        let dim2 = normalize_diagonal_axis(input, dim2, shape.rank())?;
        if dim1 == dim2 {
            return Err(Error::InvalidDiagonal {
                reason: "diagonal axes must be distinct",
            });
        }
        let mut permutation = (0..shape.rank())
            .filter(|axis| *axis != dim1 && *axis != dim2)
            .collect::<Vec<_>>();
        permutation.extend([dim1, dim2]);
        let rows = shape.dims()[dim1];
        let columns = shape.dims()[dim2];
        let (row_start, column_start) = if offset >= 0 {
            (0, usize::try_from(offset).unwrap_or(usize::MAX))
        } else {
            (
                offset
                    .checked_abs()
                    .and_then(|offset| usize::try_from(offset).ok())
                    .unwrap_or(usize::MAX),
                0,
            )
        };
        let length = rows
            .saturating_sub(row_start)
            .min(columns.saturating_sub(column_start));
        let mut row_values = Vec::with_capacity(length);
        let mut column_values = Vec::with_capacity(length);
        for lane in 0..length {
            row_values.push(
                isize::try_from(
                    row_start
                        .checked_add(lane)
                        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?,
                )
                .map_err(|_| Error::ShapeOverflow(shape.clone()))?,
            );
            column_values.push(
                isize::try_from(
                    column_start
                        .checked_add(lane)
                        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?,
                )
                .map_err(|_| Error::ShapeOverflow(shape.clone()))?,
            );
        }
        let mut specs = (0..shape.rank() - 2)
            .map(|_| StaticIndex::Slice {
                start: None,
                stop: None,
                step: 1,
            })
            .collect::<Vec<_>>();
        specs.push(StaticIndex::Advanced {
            shape: Shape::new(vec![length]),
            values: row_values,
        });
        specs.push(StaticIndex::Advanced {
            shape: Shape::new(vec![length]),
            values: column_values,
        });
        Ok(Self { permutation, specs })
    }
}

impl Graph {
    /// Selects a diagonal across two static axes, retaining all other axes in
    /// source order and appending the diagonal axis. Positive offsets start
    /// above the main diagonal; negative offsets start below it.
    pub fn diagonal_static(
        &mut self,
        input: NodeId,
        offset: isize,
        dim1: isize,
        dim2: isize,
    ) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let plan = StaticDiagonalPlan::new(input, &shape, offset, dim1, dim2)?;
        let identity = plan
            .permutation
            .iter()
            .enumerate()
            .all(|(axis, source)| axis == *source);
        let input = if identity {
            input
        } else {
            self.permute(input, plan.permutation)?
        };
        self.static_index(input, &plan.specs)
    }
}

fn normalize_diagonal_axis(input: NodeId, axis: isize, rank: usize) -> Result<usize> {
    let normalized = if axis < 0 {
        axis.checked_add(rank as isize)
    } else {
        Some(axis)
    };
    normalized
        .and_then(|axis| usize::try_from(axis).ok())
        .filter(|axis| *axis < rank)
        .ok_or(Error::InvalidAxis {
            node: input,
            axis: usize::try_from(axis).unwrap_or(usize::MAX),
            rank,
        })
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum SourceAxis {
    Fixed(usize),
    Slice { values: Vec<usize>, basic: usize },
    Advanced { shape: Shape, values: Vec<usize> },
}

/// A normalized, checked immutable indexing map.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StaticIndexPlan {
    source: Shape,
    output: Shape,
    // The canonical normalized identity. Keeping the selected base offsets
    // makes replay independent of source syntax while preserving duplicate
    // lane order for replacement updates.
    offsets: Vec<usize>,
}

impl StaticIndexPlan {
    /// Normalizes Python/tinygrad-style static indexing.
    ///
    /// Integers collapse source axes, slices retain them, `NewAxis` inserts
    /// one, and one ellipsis expands to the required full slices. A
    /// non-consecutive group of two or more advanced indices is moved to the
    /// front when it does not already begin there, matching tinygrad's current
    /// `_getitem` placement rule.
    pub fn new(source: Shape, specs: &[StaticIndex]) -> Result<Self> {
        let consumed = specs
            .iter()
            .filter(|spec| !matches!(spec, StaticIndex::NewAxis | StaticIndex::Ellipsis))
            .count();
        if consumed > source.rank()
            || specs
                .iter()
                .filter(|spec| matches!(spec, StaticIndex::Ellipsis))
                .count()
                > 1
        {
            return Err(Error::InvalidIndex);
        }
        let full = || StaticIndex::Slice {
            start: None,
            stop: None,
            step: 1,
        };
        let missing = source.rank() - consumed;
        let mut normalized = Vec::new();
        let mut had_ellipsis = false;
        for spec in specs {
            if matches!(spec, StaticIndex::Ellipsis) {
                had_ellipsis = true;
                normalized.extend((0..missing).map(|_| full()));
            } else {
                normalized.push(spec.clone());
            }
        }
        if !had_ellipsis {
            normalized.extend((0..missing).map(|_| full()));
        }

        let mut source_axis = 0usize;
        let mut axes = Vec::with_capacity(source.rank());
        let mut advanced_positions = Vec::new();
        let mut advanced_shapes = Vec::new();
        let mut basic_lengths = Vec::new();
        let mut x_dim = 0usize;
        for spec in normalized {
            match spec {
                StaticIndex::NewAxis => {
                    basic_lengths.push(1);
                    x_dim += 1;
                }
                StaticIndex::Integer(value) => {
                    axes.push(SourceAxis::Fixed(normalize(
                        value,
                        source.dims()[source_axis],
                    )?));
                    source_axis += 1;
                }
                StaticIndex::Slice { start, stop, step } => {
                    let basic = basic_lengths.len();
                    basic_lengths
                        .push(slice_indices(source.dims()[source_axis], start, stop, step)?.len());
                    axes.push(SourceAxis::Slice {
                        values: slice_indices(source.dims()[source_axis], start, stop, step)?,
                        basic,
                    });
                    source_axis += 1;
                    x_dim += 1;
                }
                StaticIndex::Advanced { shape, values } => {
                    if values.len() != shape.numel()? {
                        return Err(Error::InvalidIndex);
                    }
                    let dim = source.dims()[source_axis];
                    axes.push(SourceAxis::Advanced {
                        shape: shape.clone(),
                        values: values
                            .into_iter()
                            .map(|value| normalize(value, dim))
                            .collect::<Result<_>>()?,
                    });
                    advanced_positions.push(x_dim);
                    advanced_shapes.push(shape);
                    source_axis += 1;
                    x_dim += 1;
                }
                StaticIndex::Ellipsis => unreachable!("ellipsis is expanded above"),
            }
        }
        debug_assert_eq!(source_axis, source.rank());

        let advanced_shape = broadcast_shapes(&advanced_shapes)?;
        let consecutive = advanced_positions
            .windows(2)
            .all(|pair| pair[1] == pair[0] + 1);
        let move_advanced_front = advanced_positions.len() > 1
            && !consecutive
            && advanced_positions
                .first()
                .is_some_and(|position| *position != 0);
        let pre_basics = advanced_positions
            .first()
            .map_or(basic_lengths.len(), |first| {
                // Integers are absent from x-dim; each previous non-advanced x-dim is basic.
                *first
                    - advanced_positions
                        .iter()
                        .filter(|position| **position < *first)
                        .count()
            });
        let advanced_start = if move_advanced_front { 0 } else { pre_basics };
        let advanced_rank = advanced_shape.rank();
        let basic_output = (0..basic_lengths.len())
            .map(|basic| {
                if move_advanced_front {
                    advanced_rank + basic
                } else if basic < pre_basics {
                    basic
                } else {
                    advanced_start + advanced_rank + (basic - pre_basics)
                }
            })
            .collect::<Vec<_>>();
        let mut output = Vec::with_capacity(basic_lengths.len() + advanced_rank);
        if move_advanced_front {
            output.extend_from_slice(advanced_shape.dims());
            output.extend_from_slice(&basic_lengths);
        } else {
            output.extend_from_slice(&basic_lengths[..pre_basics]);
            output.extend_from_slice(advanced_shape.dims());
            output.extend_from_slice(&basic_lengths[pre_basics..]);
        }
        let output = Shape::new(output);
        let source_index = DenseIndex::new(source.clone())?;
        let output_index = DenseIndex::new(output.clone())?;
        let offsets = (0..output_index.len())
            .map(|linear| {
                let coordinates = output_index.coords(linear)?;
                let advanced = &coordinates[advanced_start..advanced_start + advanced_shape.rank()];
                let source_coords = axes
                    .iter()
                    .map(|axis| match axis {
                        SourceAxis::Fixed(value) => Ok(*value),
                        SourceAxis::Slice { values, basic } => values
                            .get(coordinates[basic_output[*basic]])
                            .copied()
                            .ok_or(Error::InvalidIndex),
                        SourceAxis::Advanced { shape, values } => values
                            .get(broadcasted_offset(shape, &advanced_shape, advanced)?)
                            .copied()
                            .ok_or(Error::InvalidIndex),
                    })
                    .collect::<Result<Vec<_>>>()?;
                source_index.offset(&source_coords)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            source,
            output,
            offsets,
        })
    }

    pub fn output_shape(&self) -> &Shape {
        &self.output
    }

    /// Original dense source descriptor validated while normalizing the
    /// specifications.  Consumers use it to reject stale effect targets.
    pub fn source_shape(&self) -> &Shape {
        &self.source
    }

    /// Row-major base offsets selected by each output lane. This is the
    /// canonical ordering for functional update and effect STORE execution;
    /// assigning the offsets in sequence implements deterministic
    /// last-writer-wins duplicates.
    pub(crate) fn source_offsets(&self) -> Result<Vec<usize>> {
        Ok(self.offsets.clone())
    }

    /// Rebuilds an already-normalized plan from its deterministic row-major
    /// source offsets. This is the artifact/replay boundary: it validates all
    /// rank, count, and bounds contracts without re-parsing source syntax.
    pub(crate) fn from_offsets(source: Shape, output: Shape, offsets: Vec<usize>) -> Result<Self> {
        let source_len = source.numel()?;
        if offsets.len() != output.numel()? || offsets.iter().any(|offset| *offset >= source_len) {
            return Err(Error::InvalidIndex);
        }
        Ok(Self {
            source,
            output,
            offsets,
        })
    }

    /// Maps a checked output coordinate to its source coordinate.
    pub fn source_coords(&self, output: &[usize]) -> Result<Vec<usize>> {
        if output.len() != self.output.rank()
            || output
                .iter()
                .zip(self.output.dims())
                .any(|(coordinate, dim)| *coordinate >= *dim)
        {
            return Err(Error::InvalidIndex);
        }
        let linear = DenseIndex::new(self.output.clone())?.offset(output)?;
        DenseIndex::new(self.source.clone())?
            .coords(*self.offsets.get(linear).ok_or(Error::InvalidIndex)?)
    }
}

fn broadcast_shapes(shapes: &[Shape]) -> Result<Shape> {
    let rank = shapes.iter().map(Shape::rank).max().unwrap_or(0);
    let mut dims = vec![1; rank];
    for shape in shapes {
        for (destination, source) in dims.iter_mut().rev().zip(shape.dims().iter().rev()) {
            if *destination == *source || *source == 1 {
                continue;
            }
            if *destination == 1 {
                *destination = *source;
            } else {
                return Err(Error::InvalidIndex);
            }
        }
    }
    Ok(Shape::new(dims))
}

fn broadcasted_offset(shape: &Shape, output: &Shape, coords: &[usize]) -> Result<usize> {
    if shape.rank() > output.rank() || coords.len() != output.rank() {
        return Err(Error::InvalidIndex);
    }
    let leading = output.rank() - shape.rank();
    shape
        .dims()
        .iter()
        .enumerate()
        .try_fold(0usize, |offset, (axis, dim)| {
            let coordinate = if *dim == 1 { 0 } else { coords[leading + axis] };
            if coordinate >= *dim {
                return Err(Error::InvalidIndex);
            }
            offset
                .checked_mul(*dim)
                .and_then(|value| value.checked_add(coordinate))
                .ok_or(Error::InvalidIndex)
        })
}

fn normalize(value: isize, dim: usize) -> Result<usize> {
    let value = if value < 0 {
        value + dim as isize
    } else {
        value
    };
    let value = usize::try_from(value).map_err(|_| Error::InvalidIndex)?;
    if value >= dim {
        Err(Error::InvalidIndex)
    } else {
        Ok(value)
    }
}

fn slice_indices(
    dim: usize,
    start: Option<isize>,
    stop: Option<isize>,
    step: isize,
) -> Result<Vec<usize>> {
    if step == 0 {
        return Err(Error::InvalidIndex);
    }
    let dim = isize::try_from(dim).map_err(|_| Error::InvalidIndex)?;
    let clamp = |value: isize, low: isize, high: isize| value.clamp(low, high);
    let (mut start, stop) = if step > 0 {
        let start = start.map_or(0, |value| if value < 0 { value + dim } else { value });
        let stop = stop.map_or(dim, |value| if value < 0 { value + dim } else { value });
        (clamp(start, 0, dim), clamp(stop, 0, dim))
    } else {
        let start = start.map_or(dim - 1, |value| if value < 0 { value + dim } else { value });
        let stop = stop.map_or(-1, |value| if value < 0 { value + dim } else { value });
        (clamp(start, -1, dim - 1), clamp(stop, -1, dim - 1))
    };
    let mut values = Vec::new();
    while if step > 0 { start < stop } else { start > stop } {
        values.push(usize::try_from(start).map_err(|_| Error::InvalidIndex)?);
        start = start.checked_add(step).ok_or(Error::InvalidIndex)?;
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plans_mixed_slices_newaxes_and_negative_integers() {
        let plan = StaticIndexPlan::new(
            Shape::from([2, 3, 4]),
            &[
                StaticIndex::NewAxis,
                StaticIndex::Integer(-1),
                StaticIndex::Ellipsis,
                StaticIndex::Slice {
                    start: None,
                    stop: None,
                    step: -2,
                },
            ],
        )
        .unwrap();
        assert_eq!(plan.output_shape(), &Shape::from([1, 3, 2]));
        assert_eq!(plan.source_coords(&[0, 2, 1]).unwrap(), vec![1, 2, 1]);
        assert!(StaticIndexPlan::new(Shape::from([2]), &[StaticIndex::Integer(2)]).is_err());
    }
    #[test]
    fn broadcasts_multiple_advanced_indices_in_source_position() {
        let plan = StaticIndexPlan::new(
            Shape::from([3, 4]),
            &[
                StaticIndex::Advanced {
                    shape: Shape::from([2, 1]),
                    values: vec![0, -1],
                },
                StaticIndex::Advanced {
                    shape: Shape::from([1, 3]),
                    values: vec![1, 3, 0],
                },
            ],
        )
        .unwrap();
        assert_eq!(plan.output_shape(), &Shape::from([2, 3]));
        assert_eq!(plan.source_coords(&[1, 2]).unwrap(), vec![2, 0]);
    }
    #[test]
    fn moves_separated_advanced_indices_to_front_like_tinygrad() {
        let plan = StaticIndexPlan::new(
            Shape::from([2, 3, 4, 5]),
            &[
                StaticIndex::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
                StaticIndex::Advanced {
                    shape: Shape::from([2]),
                    values: vec![0, 2],
                },
                StaticIndex::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
                StaticIndex::Advanced {
                    shape: Shape::from([2]),
                    values: vec![1, 3],
                },
            ],
        )
        .unwrap();
        assert_eq!(plan.output_shape(), &Shape::from([2, 2, 4]));
        assert_eq!(plan.source_coords(&[1, 0, 3]).unwrap(), vec![0, 2, 3, 3]);
    }
    #[test]
    fn rejects_bad_broadcast_and_preserves_empty_slices() {
        let bad = [
            StaticIndex::Advanced {
                shape: Shape::from([2]),
                values: vec![0, 1],
            },
            StaticIndex::Advanced {
                shape: Shape::from([3]),
                values: vec![0, 1, 2],
            },
        ];
        assert!(StaticIndexPlan::new(Shape::from([2, 3]), &bad).is_err());
        let empty = StaticIndexPlan::new(
            Shape::from([0]),
            &[
                StaticIndex::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
                StaticIndex::NewAxis,
            ],
        )
        .unwrap();
        assert_eq!(empty.output_shape(), &Shape::from([0, 1]));
    }
}
