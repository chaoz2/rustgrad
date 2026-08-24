//! Pure planning for statically-shaped immutable tensor indexing.
//!
//! This module deliberately contains no graph nodes or storage access. A
//! [`StaticIndexPlan`] validates the public index specification, computes its
//! static output shape, and maps one output coordinate back to a source
//! coordinate. The graph and CPU layers consume that map without re-parsing
//! indexing syntax.

use crate::{Error, Result, Shape};

/// A statically-known component of an immutable tensor index.
#[derive(Clone, Debug, Eq, PartialEq)]
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

#[derive(Clone, Debug, Eq, PartialEq)]
enum SourceAxis {
    Fixed(usize),
    Slice { values: Vec<usize>, basic: usize },
    Advanced { shape: Shape, values: Vec<usize> },
}

/// A normalized, checked immutable indexing map.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticIndexPlan {
    output: Shape,
    axes: Vec<SourceAxis>,
    advanced_start: usize,
    advanced_shape: Shape,
    basic_output: Vec<usize>,
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
        Ok(Self {
            output: Shape::new(output),
            axes,
            advanced_start,
            advanced_shape,
            basic_output,
        })
    }

    pub fn output_shape(&self) -> &Shape {
        &self.output
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
        let advanced =
            &output[self.advanced_start..self.advanced_start + self.advanced_shape.rank()];
        self.axes
            .iter()
            .map(|axis| match axis {
                SourceAxis::Fixed(value) => Ok(*value),
                SourceAxis::Slice { values, basic } => values
                    .get(output[self.basic_output[*basic]])
                    .copied()
                    .ok_or(Error::InvalidIndex),
                SourceAxis::Advanced { shape, values } => values
                    .get(broadcasted_offset(shape, &self.advanced_shape, advanced)?)
                    .copied()
                    .ok_or(Error::InvalidIndex),
            })
            .collect()
    }
}

fn broadcast_shapes(shapes: &[Shape]) -> Result<Shape> {
    let rank = shapes.iter().map(Shape::rank).max().unwrap_or(0);
    let mut dims = vec![1; rank];
    for shape in shapes {
        for (destination, source) in dims.iter_mut().rev().zip(shape.dims().iter().rev()) {
            if *destination != *source && *destination != 1 && *source != 1 {
                return Err(Error::InvalidIndex);
            }
            *destination = (*destination).max(*source);
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
