//! Source-audited float8 reduction policy for the CPU semantic oracle.

use crate::index::DenseIndex;
use crate::{
    Error, Float8Format, Float8Storage, ReduceKind, Result, Scalar, Shape, Storage, TensorData,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Float8ReductionPolicy {
    F32Accumulate,
    QuantizedProduct,
    Extremum,
    Unsupported,
}

pub(crate) const fn policy(kind: ReduceKind) -> Float8ReductionPolicy {
    match kind {
        ReduceKind::Sum | ReduceKind::Mean => Float8ReductionPolicy::F32Accumulate,
        ReduceKind::Product => Float8ReductionPolicy::QuantizedProduct,
        ReduceKind::Max | ReduceKind::Min => Float8ReductionPolicy::Extremum,
        // Graph::any/all preflight casts numeric inputs to bool, and the
        // released float8 surface rejects that cast before execution.
        ReduceKind::Any | ReduceKind::All => Float8ReductionPolicy::Unsupported,
    }
}

pub(crate) fn reduce(
    input: &TensorData,
    kind: ReduceKind,
    axes: &[usize],
    keepdim: bool,
) -> Result<TensorData> {
    let format = input
        .dtype()
        .float8_format()
        .ok_or(Error::UnsupportedDType {
            dtype: input.dtype(),
        })?;
    let shape = reduction_shape(input.shape(), axes, keepdim);
    let input_index = DenseIndex::new(input.shape().clone())?;
    let output_index = DenseIndex::new(shape.clone())?;
    let mut groups = vec![Vec::new(); output_index.len()];
    for lane in 0..input_index.len() {
        let coordinates = input_index.coords(lane)?;
        let output_coordinates = coordinates
            .iter()
            .enumerate()
            .filter_map(|(axis, coordinate)| {
                if axes.contains(&axis) {
                    keepdim.then_some(0)
                } else {
                    Some(*coordinate)
                }
            })
            .collect::<Vec<_>>();
        groups[output_index.offset(&output_coordinates)?].push(lane);
    }
    match policy(kind) {
        Float8ReductionPolicy::F32Accumulate => f32_accumulate(input, format, kind, shape, &groups),
        Float8ReductionPolicy::QuantizedProduct => product(input, format, shape, &groups),
        Float8ReductionPolicy::Extremum => extremum(input, format, kind, shape, &groups),
        Float8ReductionPolicy::Unsupported => Err(Error::UnsupportedDType {
            dtype: input.dtype(),
        }),
    }
}

fn reduction_shape(input: &Shape, axes: &[usize], keepdim: bool) -> Shape {
    Shape::new(
        input
            .dims()
            .iter()
            .enumerate()
            .filter_map(|(axis, dimension)| {
                if axes.contains(&axis) {
                    keepdim.then_some(1)
                } else {
                    Some(*dimension)
                }
            })
            .collect::<Vec<_>>(),
    )
}

fn f32_accumulate(
    input: &TensorData,
    format: Float8Format,
    kind: ReduceKind,
    shape: Shape,
    groups: &[Vec<usize>],
) -> Result<TensorData> {
    TensorData::from_scalars(
        shape,
        format.dtype(),
        groups.iter().map(|group| {
            // IEEE does not prescribe the sign bit of the NaN produced by
            // `0.0f32 / 0.0f32`. Canonicalize empty means before encoding so
            // E4M3's raw NaN lane is deterministic across CPU targets.
            if kind == ReduceKind::Mean && group.is_empty() {
                return Scalar::F(f64::NAN);
            }
            let sum = group.iter().fold(0.0f32, |sum, lane| {
                sum + input.scalar_at(*lane).as_f64() as f32
            });
            Scalar::F(f64::from(if kind == ReduceKind::Mean {
                sum / group.len() as f32
            } else {
                sum
            }))
        }),
    )
}

fn product(
    input: &TensorData,
    format: Float8Format,
    shape: Shape,
    groups: &[Vec<usize>],
) -> Result<TensorData> {
    TensorData::from_scalars(
        shape,
        format.dtype(),
        groups.iter().map(|group| {
            Scalar::F(group.iter().fold(1.0f64, |accumulator, lane| {
                format.decode(format.encode(accumulator * input.scalar_at(*lane).as_f64()))
            }))
        }),
    )
}

fn extremum(
    input: &TensorData,
    format: Float8Format,
    kind: ReduceKind,
    shape: Shape,
    groups: &[Vec<usize>],
) -> Result<TensorData> {
    let Storage::Float8(source) = input.storage() else {
        return Err(Error::UnsupportedDType {
            dtype: input.dtype(),
        });
    };
    let output = groups
        .iter()
        .map(|group| {
            group
                .iter()
                .copied()
                .filter(|lane| !format.decode(source.as_raw()[*lane]).is_nan())
                .fold(None, |best, lane| match best {
                    None => Some(lane),
                    Some(previous)
                        if kind == ReduceKind::Max
                            && format.decode(source.as_raw()[lane])
                                > format.decode(source.as_raw()[previous]) =>
                    {
                        Some(lane)
                    }
                    Some(previous)
                        if kind == ReduceKind::Min
                            && format.decode(source.as_raw()[lane])
                                < format.decode(source.as_raw()[previous]) =>
                    {
                        Some(lane)
                    }
                    Some(previous) => Some(previous),
                })
                .map(|lane| source.as_raw()[lane])
                .unwrap_or_else(|| {
                    format.encode(if kind == ReduceKind::Max {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    })
                })
        })
        .collect();
    TensorData::from_storage(
        shape,
        Storage::Float8(Float8Storage::from_raw(format, output)),
    )
}
