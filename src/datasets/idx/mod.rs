//! MNIST IDX decoding and bounded local-file loading.

mod file;

pub use file::{
    MnistIdxFileError, MnistIdxReadLimits, load_mnist_idx_files, load_mnist_idx_files_with_limits,
};

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
    /// Verifies that this public feature/label pair retains the exact MNIST
    /// storage contract produced by [`parse_mnist_idx`].
    ///
    /// The parser permits any IDX row/column geometry, so
    /// this validates against the pair's declared `rows` and `cols` rather
    /// than imposing a separate 28×28 policy. Public tensors remain
    /// inspectable, but derived consumers must not normalize a mismatched
    /// sample axis or invalid label inventory.
    pub fn validate(&self) -> Result<()> {
        if self.images.dtype() != DType::U8 {
            return Err(bad("MNIST images must have dtype U8"));
        }
        if self.labels.dtype() != DType::U8 {
            return Err(bad("MNIST labels must have dtype U8"));
        }
        let image_shape = self.images.shape().dims();
        if image_shape.len() != 4 || image_shape[1..] != [1, self.rows, self.cols] {
            return Err(bad(format!(
                "MNIST images must have shape [N, 1, {}, {}], got {:?}",
                self.rows, self.cols, image_shape
            )));
        }
        let count = image_shape[0];
        if self.labels.shape().dims() != [count] {
            return Err(bad(format!(
                "MNIST label shape must be [{count}], got {:?}",
                self.labels.shape().dims()
            )));
        }
        if let Some((index, label)) = (0..self.labels.len())
            .map(|index| (index, self.labels.scalar_at(index).as_u64()))
            .find(|(_, label)| *label > 9)
        {
            return Err(bad(format!(
                "MNIST label {label} at record {index} is outside 0..=9"
            )));
        }
        Ok(())
    }

    /// Converts byte pixels to F32 in the inclusive range `0..=1`.
    pub fn normalized_f32(&self) -> Result<TensorData> {
        self.validate()?;
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
    let dataset = MnistIdx {
        images: TensorData::from_le_bytes(
            Shape::new([count, 1, rows, cols]),
            DType::U8,
            &images[16..],
        )?,
        labels: TensorData::from_le_bytes([count], DType::U8, &labels[8..])?,
        rows,
        cols,
    };
    dataset.validate()?;
    Ok(dataset)
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

    #[test]
    fn public_mnist_pair_validation_rejects_misaligned_or_invalid_tensors_without_derivation() {
        let mut images = Vec::new();
        images.extend_from_slice(&2051u32.to_be_bytes());
        images.extend_from_slice(&1u32.to_be_bytes());
        images.extend_from_slice(&2u32.to_be_bytes());
        images.extend_from_slice(&2u32.to_be_bytes());
        images.extend_from_slice(&[0, 1, 2, 3]);
        let mut labels = Vec::new();
        labels.extend_from_slice(&2049u32.to_be_bytes());
        labels.extend_from_slice(&1u32.to_be_bytes());
        labels.push(9);
        let valid = parse_mnist_idx(&images, &labels).unwrap();
        let before = valid.clone();
        assert!(valid.validate().is_ok());
        assert_eq!(valid, before);

        let malformed = [
            MnistIdx {
                images: valid.images.clone(),
                labels: TensorData::from_scalars([2], DType::U8, [Scalar::U(1), Scalar::U(2)])
                    .unwrap(),
                rows: valid.rows,
                cols: valid.cols,
            },
            MnistIdx {
                images: TensorData::from_scalars([1, 1, 1, 4], DType::U8, [Scalar::U(0); 4])
                    .unwrap(),
                labels: valid.labels.clone(),
                rows: valid.rows,
                cols: valid.cols,
            },
            MnistIdx {
                images: valid.images.clone().cast(DType::F32),
                labels: valid.labels.clone(),
                rows: valid.rows,
                cols: valid.cols,
            },
            MnistIdx {
                images: valid.images.clone(),
                labels: TensorData::from_scalars([1], DType::U8, [Scalar::U(10)]).unwrap(),
                rows: valid.rows,
                cols: valid.cols,
            },
        ];
        for dataset in malformed {
            let before = dataset.clone();
            assert!(dataset.validate().is_err());
            assert!(dataset.normalized_f32().is_err());
            assert_eq!(dataset, before);
        }

        labels.pop();
        labels.push(10);
        assert!(parse_mnist_idx(&images, &labels).is_err());
    }
}
