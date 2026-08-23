//! Checked row-major coordinate mapping for dense tensor storage.
use crate::{Error, Result, Shape};
#[derive(Clone, Debug)]
pub(crate) struct DenseIndex {
    shape: Shape,
    strides: Vec<usize>,
    len: usize,
}
impl DenseIndex {
    pub(crate) fn new(shape: Shape) -> Result<Self> {
        let len = shape.numel()?;
        let mut st: usize = 1;
        let mut ss = vec![0; shape.rank()];
        for (i, d) in shape.dims().iter().enumerate().rev() {
            ss[i] = st;
            st = st
                .checked_mul(*d)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        }
        Ok(Self {
            shape,
            strides: ss,
            len,
        })
    }
    pub(crate) fn len(&self) -> usize {
        self.len
    }
    pub(crate) fn shape(&self) -> &Shape {
        &self.shape
    }
    pub(crate) fn coords(&self, n: usize) -> Result<Vec<usize>> {
        if n >= self.len {
            return Err(Error::InvalidIndex);
        }
        Ok(self
            .strides
            .iter()
            .enumerate()
            .map(|(i, s)| (n / s) % self.shape.dims()[i])
            .collect())
    }
    pub(crate) fn offset(&self, c: &[usize]) -> Result<usize> {
        if c.len() != self.shape.rank()
            || c.iter()
                .enumerate()
                .any(|(i, x)| *x >= self.shape.dims()[i])
        {
            return Err(Error::InvalidIndex);
        }
        c.iter().zip(&self.strides).try_fold(0usize, |o, (x, s)| {
            o.checked_add(x.checked_mul(*s).ok_or(Error::InvalidIndex)?)
                .ok_or(Error::InvalidIndex)
        })
    }
    pub(crate) fn broadcast_offset(&self, out: &Self, c: &[usize]) -> Result<usize> {
        if c.len() != out.shape.rank() || self.shape.rank() > out.shape.rank() {
            return Err(Error::InvalidIndex);
        }
        let p = out.shape.rank() - self.shape.rank();
        self.offset(
            &self
                .shape
                .dims()
                .iter()
                .enumerate()
                .map(|(i, d)| if *d == 1 { 0 } else { c[i + p] })
                .collect::<Vec<_>>(),
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maps_scalars_empty_and_broadcast() {
        let s = DenseIndex::new(Shape::from([])).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s.coords(0).unwrap(), Vec::<usize>::new());
        let e = DenseIndex::new(Shape::from([2, 0])).unwrap();
        assert_eq!(e.len(), 0);
        assert!(e.coords(0).is_err());
        let o = DenseIndex::new(Shape::from([2, 3])).unwrap();
        let r = DenseIndex::new(Shape::from([3])).unwrap();
        assert_eq!(r.broadcast_offset(&o, &[1, 2]).unwrap(), 2);
        assert_eq!(o.offset(&o.coords(5).unwrap()).unwrap(), 5);
    }
}
