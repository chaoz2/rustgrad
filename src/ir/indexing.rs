//! Pure normalization for static integer/slice/newaxis/ellipsis indexing.
use crate::{Error, Result, Shape};
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
    Advanced {
        shape: Shape,
        values: Vec<isize>,
    },
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticIndexPlan {
    output: Shape,
    source_axes: Vec<Option<Vec<usize>>>,
}
impl StaticIndexPlan {
    pub fn new(source: Shape, specs: &[StaticIndex]) -> Result<Self> {
        let used = specs
            .iter()
            .filter(|x| !matches!(x, StaticIndex::NewAxis | StaticIndex::Ellipsis))
            .count();
        if used > source.rank()
            || specs
                .iter()
                .filter(|x| matches!(x, StaticIndex::Ellipsis))
                .count()
                > 1
        {
            return Err(Error::InvalidIndex);
        }
        let fill = source.rank() - used;
        let mut xs = Vec::new();
        let mut ell = false;
        for x in specs {
            if matches!(x, StaticIndex::Ellipsis) {
                ell = true;
                xs.extend((0..fill).map(|_| StaticIndex::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                }))
            } else {
                xs.push(x.clone())
            }
        }
        if !ell {
            xs.extend((0..fill).map(|_| StaticIndex::Slice {
                start: None,
                stop: None,
                step: 1,
            }))
        }
        let (mut axis, mut out, mut maps) = (0, Vec::new(), Vec::new());
        for x in xs {
            match x {
                StaticIndex::NewAxis => {
                    out.push(1);
                    maps.push(None)
                }
                StaticIndex::Integer(v) => {
                    maps.push(Some(vec![norm(v, source.dims()[axis])?]));
                    axis += 1
                }
                StaticIndex::Slice { start, stop, step } => {
                    let v = slice(source.dims()[axis], start, stop, step)?;
                    out.push(v.len());
                    maps.push(Some(v));
                    axis += 1
                }
                StaticIndex::Advanced { shape, values } => {
                    if values.len() != shape.numel()? {
                        return Err(Error::InvalidIndex);
                    }
                    out.extend_from_slice(shape.dims());
                    let d = source.dims()[axis];
                    maps.push(Some(
                        values
                            .into_iter()
                            .map(|v| norm(v, d))
                            .collect::<Result<_>>()?,
                    ));
                    axis += 1
                }
                StaticIndex::Ellipsis => unreachable!(),
            }
        }
        Ok(Self {
            output: Shape::new(out),
            source_axes: maps,
        })
    }
    pub fn output_shape(&self) -> &Shape {
        &self.output
    }
}
fn norm(v: isize, d: usize) -> Result<usize> {
    let v = if v < 0 { v + d as isize } else { v };
    let v = usize::try_from(v).map_err(|_| Error::InvalidIndex)?;
    if v >= d {
        Err(Error::InvalidIndex)
    } else {
        Ok(v)
    }
}
fn slice(d: usize, start: Option<isize>, stop: Option<isize>, step: isize) -> Result<Vec<usize>> {
    if step == 0 {
        return Err(Error::InvalidIndex);
    }
    let mut v = Vec::new();
    let mut i = start.unwrap_or(if step > 0 { 0 } else { d as isize - 1 });
    let end = stop.unwrap_or(if step > 0 { d as isize } else { -1 });
    if i < 0 {
        i += d as isize
    }
    let end = if end < 0 && !(step < 0 && end == -1) {
        end + d as isize
    } else {
        end
    };
    while if step > 0 { i < end } else { i > end } {
        if let Ok(x) = norm(i, d) {
            v.push(x)
        }
        i += step
    }
    Ok(v)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plan_handles_mixed_static_forms() {
        let p = StaticIndexPlan::new(
            Shape::from([2, 3]),
            &[
                StaticIndex::NewAxis,
                StaticIndex::Integer(-1),
                StaticIndex::Ellipsis,
            ],
        )
        .unwrap();
        assert_eq!(p.output_shape(), &Shape::from([1, 3]));
        assert!(StaticIndexPlan::new(Shape::from([2]), &[StaticIndex::Integer(2)]).is_err());
    }
}
