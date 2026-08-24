//! MNIST IDX decoding.

use super::bad;
use crate::{DType, Result, Scalar, Shape, TensorData};

fn be32(bytes: &[u8]) -> Result<usize> {
    usize::try_from(u32::from_be_bytes(
        bytes.try_into().map_err(|_| bad("truncated IDX header"))?,
    ))
    .map_err(|_| bad("IDX count overflow"))
}

/// An exact local MNIST IDX image/label pair.
#[derive(Clone, Debug, PartialEq)]
pub struct MnistIdx {
    pub images: TensorData,
    pub labels: TensorData,
    pub rows: usize,
    pub cols: usize,
}

impl MnistIdx {
    /// Converts byte pixels to F32 in the inclusive range `0..=1`.
    pub fn normalized_f32(&self) -> Result<TensorData> {
        TensorData::from_scalars(
            self.images.shape().clone(),
            DType::F32,
            (0..self.images.len())
                .map(|index| Scalar::F(self.images.scalar_at(index).as_f64() / 255.)),
        )
    }
}

/// Parses one uncompressed MNIST IDX image file and its label file.
pub fn parse_mnist_idx(images: &[u8], labels: &[u8]) -> Result<MnistIdx> {
    if images.len() < 16 || labels.len() < 8 {
        return Err(bad("truncated IDX header"));
    }
    if be32(&images[..4])? != 2051 || be32(&labels[..4])? != 2049 {
        return Err(bad("invalid IDX magic"));
    }
    let count = be32(&images[4..8])?;
    let rows = be32(&images[8..12])?;
    let cols = be32(&images[12..16])?;
    let label_count = be32(&labels[4..8])?;
    if count != label_count {
        return Err(bad("IDX image/label counts differ"));
    }
    let pixels = count
        .checked_mul(rows)
        .and_then(|value| value.checked_mul(cols))
        .ok_or_else(|| bad("IDX shape overflow"))?;
    if images.len()
        != 16usize
            .checked_add(pixels)
            .ok_or_else(|| bad("IDX length overflow"))?
        || labels.len()
            != 8usize
                .checked_add(count)
                .ok_or_else(|| bad("IDX trailing or truncated data"))?
    {
        return Err(bad("IDX payload length mismatch"));
    }
    Ok(MnistIdx {
        images: TensorData::from_le_bytes(
            Shape::new([count, 1, rows, cols]),
            DType::U8,
            &images[16..],
        )?,
        labels: TensorData::from_le_bytes([count], DType::U8, &labels[8..])?,
        rows,
        cols,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idx_layout_normalization_and_lengths_are_exact() {
        let mut images = Vec::new();
        images.extend_from_slice(&2051u32.to_be_bytes());
        images.extend_from_slice(&2u32.to_be_bytes());
        images.extend_from_slice(&2u32.to_be_bytes());
        images.extend_from_slice(&2u32.to_be_bytes());
        images.extend_from_slice(&[0, 255, 1, 2, 3, 4, 5, 6]);
        let mut labels = Vec::new();
        labels.extend_from_slice(&2049u32.to_be_bytes());
        labels.extend_from_slice(&2u32.to_be_bytes());
        labels.extend_from_slice(&[1, 9]);

        let dataset = parse_mnist_idx(&images, &labels).unwrap();
        assert_eq!(dataset.images.shape().dims(), &[2, 1, 2, 2]);
        assert_eq!(dataset.images.to_le_bytes().unwrap(), &images[16..]);
        assert_eq!(dataset.labels.to_le_bytes().unwrap(), &labels[8..]);
        assert_eq!(dataset.normalized_f32().unwrap().values()[1], 1.);

        assert!(parse_mnist_idx(&images[..16], &labels).is_err());
        images.push(0);
        assert!(parse_mnist_idx(&images, &labels).is_err());
    }
}
