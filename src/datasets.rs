//! Deterministic local dataset readers; network transport is deliberately absent.
use crate::{DType, Error, Result, Shape, TensorData};

fn bad(reason: impl Into<String>) -> Error {
    Error::Dataset {
        reason: reason.into(),
    }
}
fn be32(bytes: &[u8]) -> Result<usize> {
    usize::try_from(u32::from_be_bytes(
        bytes.try_into().map_err(|_| bad("truncated IDX header"))?,
    ))
    .map_err(|_| bad("IDX count overflow"))
}

#[derive(Clone, Debug, PartialEq)]
pub struct MnistIdx {
    pub images: TensorData,
    pub labels: TensorData,
    pub rows: usize,
    pub cols: usize,
}
impl MnistIdx {
    pub fn normalized_f32(&self) -> Result<TensorData> {
        TensorData::from_scalars(
            self.images.shape().clone(),
            DType::F32,
            (0..self.images.len())
                .map(|i| crate::Scalar::F(self.images.scalar_at(i).as_f64() / 255.)),
        )
    }
}
pub fn parse_mnist_idx(images: &[u8], labels: &[u8]) -> Result<MnistIdx> {
    if images.len() < 16 || labels.len() < 8 {
        return Err(bad("truncated IDX header"));
    }
    if be32(&images[..4])? != 2051 || be32(&labels[..4])? != 2049 {
        return Err(bad("invalid IDX magic"));
    }
    let n = be32(&images[4..8])?;
    let rows = be32(&images[8..12])?;
    let cols = be32(&images[12..16])?;
    let ln = be32(&labels[4..8])?;
    if n != ln {
        return Err(bad("IDX image/label counts differ"));
    }
    let pixels = n
        .checked_mul(rows)
        .and_then(|x| x.checked_mul(cols))
        .ok_or_else(|| bad("IDX shape overflow"))?;
    if images.len()
        != 16usize
            .checked_add(pixels)
            .ok_or_else(|| bad("IDX length overflow"))?
        || labels.len()
            != 8usize
                .checked_add(n)
                .ok_or_else(|| bad("IDX trailing or truncated data"))?
    {
        return Err(bad("IDX payload length mismatch"));
    }
    Ok(MnistIdx {
        images: TensorData::from_le_bytes(
            Shape::new([n, 1, rows, cols]),
            DType::U8,
            &images[16..],
        )?,
        labels: TensorData::from_le_bytes([n], DType::U8, &labels[8..])?,
        rows,
        cols,
    })
}

#[derive(Clone, Debug)]
pub struct BatchIter {
    order: Vec<usize>,
    at: usize,
    batch: usize,
    drop_last: bool,
}
impl BatchIter {
    pub fn new(
        len: usize,
        batch: usize,
        seed: u64,
        shuffle: bool,
        drop_last: bool,
    ) -> Result<Self> {
        if batch == 0 {
            return Err(bad("batch size must be nonzero"));
        };
        let mut order: Vec<usize> = (0..len).collect();
        if shuffle {
            for i in (1..len).rev() {
                let mut x = seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
                x ^= x >> 30;
                x = x.wrapping_mul(0xBF58476D1CE4E5B9);
                order.swap(i, (x as usize) % (i + 1));
            }
        }
        Ok(Self {
            order,
            at: 0,
            batch,
            drop_last,
        })
    }
}
impl Iterator for BatchIter {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.at >= self.order.len() {
            return None;
        }
        let end = (self.at + self.batch).min(self.order.len());
        if self.drop_last && end - self.at < self.batch {
            self.at = self.order.len();
            return None;
        }
        let out = self.order[self.at..end].to_vec();
        self.at = end;
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn idx_and_batches_are_exact_and_seeded() {
        let mut i = vec![];
        i.extend_from_slice(&2051u32.to_be_bytes());
        i.extend_from_slice(&2u32.to_be_bytes());
        i.extend_from_slice(&2u32.to_be_bytes());
        i.extend_from_slice(&2u32.to_be_bytes());
        i.extend_from_slice(&[0, 255, 1, 2, 3, 4, 5, 6]);
        let mut l = vec![];
        l.extend_from_slice(&2049u32.to_be_bytes());
        l.extend_from_slice(&2u32.to_be_bytes());
        l.extend_from_slice(&[1, 9]);
        let d = parse_mnist_idx(&i, &l).unwrap();
        assert_eq!(d.images.shape().dims(), &[2, 1, 2, 2]);
        assert_eq!(d.normalized_f32().unwrap().values()[1], 1.);
        assert_eq!(
            BatchIter::new(5, 2, 7, true, false)
                .unwrap()
                .collect::<Vec<_>>(),
            BatchIter::new(5, 2, 7, true, false)
                .unwrap()
                .collect::<Vec<_>>()
        );
        assert_eq!(BatchIter::new(5, 2, 0, false, true).unwrap().count(), 2);
        assert!(parse_mnist_idx(&i[..16], &l).is_err());
    }
}
