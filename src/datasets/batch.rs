//! Deterministic batch index generation.

use super::bad;
use crate::{Result, Shape, TensorData};

/// Output feature layout for a selected classification batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClassificationFeatureLayout {
    /// Retain every feature dimension after the sample axis.
    Preserve,
    /// Collapse every feature dimension after the sample axis into one axis.
    Flatten,
}

/// Owned, prevalidated features and integer class targets for one CPU step.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassificationBatch {
    pub features: TensorData,
    pub targets: TensorData,
}

/// Materializes caller-selected sample rows without scalar conversion.
///
/// `features` must have one leading sample dimension and `targets` must be an
/// integer rank-one tensor with the same count. Every index and output byte
/// size is checked before either result buffer is allocated.
pub fn materialize_classification_batch(
    features: &TensorData,
    targets: &TensorData,
    indices: &[usize],
    layout: ClassificationFeatureLayout,
) -> Result<ClassificationBatch> {
    if features.shape().rank() == 0 {
        return Err(bad("classification features need a sample dimension"));
    }
    if targets.shape().rank() != 1 || !targets.dtype().is_integer() {
        return Err(bad(
            "classification targets must be a rank-one integer tensor",
        ));
    }
    let sample_count = features.shape().dims()[0];
    if targets.shape().dims()[0] != sample_count {
        return Err(bad("classification feature and target counts differ"));
    }
    let feature_elements = features.shape().dims()[1..]
        .iter()
        .try_fold(1usize, |size, dim| size.checked_mul(*dim))
        .ok_or_else(|| bad("classification feature shape overflows"))?;
    let feature_bytes = feature_elements
        .checked_mul(features.dtype().itemsize())
        .ok_or_else(|| bad("classification feature row byte length overflows"))?;
    let target_bytes = targets.dtype().itemsize();
    for &index in indices {
        if index >= sample_count {
            return Err(bad(format!(
                "classification sample index {index} is outside 0..{sample_count}"
            )));
        }
    }
    let output_feature_bytes = indices
        .len()
        .checked_mul(feature_bytes)
        .ok_or_else(|| bad("classification output feature byte length overflows"))?;
    let output_target_bytes = indices
        .len()
        .checked_mul(target_bytes)
        .ok_or_else(|| bad("classification output target byte length overflows"))?;
    let source_features = features
        .to_le_bytes()
        .map_err(|_| bad("classification feature bytes are unavailable"))?;
    let source_targets = targets
        .to_le_bytes()
        .map_err(|_| bad("classification target bytes are unavailable"))?;
    let mut feature_bytes_out = Vec::new();
    let mut target_bytes_out = Vec::new();
    feature_bytes_out
        .try_reserve_exact(output_feature_bytes)
        .map_err(|_| bad("classification feature batch allocation failed"))?;
    target_bytes_out
        .try_reserve_exact(output_target_bytes)
        .map_err(|_| bad("classification target batch allocation failed"))?;
    for &index in indices {
        let feature_start = index
            .checked_mul(feature_bytes)
            .ok_or_else(|| bad("classification feature offset overflows"))?;
        feature_bytes_out
            .extend_from_slice(&source_features[feature_start..feature_start + feature_bytes]);
        let target_start = index
            .checked_mul(target_bytes)
            .ok_or_else(|| bad("classification target offset overflows"))?;
        target_bytes_out
            .extend_from_slice(&source_targets[target_start..target_start + target_bytes]);
    }
    let output_shape = match layout {
        ClassificationFeatureLayout::Preserve => Shape::new(
            std::iter::once(indices.len())
                .chain(features.shape().dims()[1..].iter().copied())
                .collect::<Vec<_>>(),
        ),
        ClassificationFeatureLayout::Flatten => Shape::new([indices.len(), feature_elements]),
    };
    Ok(ClassificationBatch {
        features: TensorData::from_le_bytes(output_shape, features.dtype(), &feature_bytes_out)?,
        targets: TensorData::from_le_bytes([indices.len()], targets.dtype(), &target_bytes_out)?,
    })
}

/// An iterator over deterministic index batches.
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
        }
        let mut order: Vec<usize> = (0..len).collect();
        if shuffle {
            for index in (1..len).rev() {
                let mut mixed = seed ^ (index as u64).wrapping_mul(0x9E3779B97F4A7C15);
                mixed ^= mixed >> 30;
                mixed = mixed.wrapping_mul(0xBF58476D1CE4E5B9);
                order.swap(index, (mixed as usize) % (index + 1));
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
        let output = self.order[self.at..end].to_vec();
        self.at = end;
        Some(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn batch_order_is_seeded_and_drop_last_is_explicit() {
        assert_eq!(
            BatchIter::new(5, 2, 7, true, false)
                .unwrap()
                .collect::<Vec<_>>(),
            BatchIter::new(5, 2, 7, true, false)
                .unwrap()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            BatchIter::new(5, 2, 0, false, false)
                .unwrap()
                .collect::<Vec<_>>(),
            vec![vec![0, 1], vec![2, 3], vec![4]]
        );
        assert_eq!(BatchIter::new(5, 2, 0, false, true).unwrap().count(), 2);
        assert!(BatchIter::new(1, 0, 0, false, false).is_err());
    }

    #[test]
    fn classification_batch_preserves_raw_rows_and_validates_before_output() {
        let features =
            TensorData::from_le_bytes([3, 1, 2], DType::U8, &[1, 2, 3, 4, 5, 6]).unwrap();
        let targets = TensorData::from_le_bytes([3], DType::U8, &[8, 7, 6]).unwrap();
        let batch = materialize_classification_batch(
            &features,
            &targets,
            &[2, 0],
            ClassificationFeatureLayout::Flatten,
        )
        .unwrap();
        assert_eq!(batch.features.shape().dims(), &[2, 2]);
        assert_eq!(batch.features.to_le_bytes().unwrap(), vec![5, 6, 1, 2]);
        assert_eq!(batch.targets.to_le_bytes().unwrap(), vec![6, 8]);
        let empty = materialize_classification_batch(
            &features,
            &targets,
            &[],
            ClassificationFeatureLayout::Preserve,
        )
        .unwrap();
        assert_eq!(empty.features.shape().dims(), &[0, 1, 2]);
        assert!(
            materialize_classification_batch(
                &features,
                &targets,
                &[3],
                ClassificationFeatureLayout::Preserve,
            )
            .is_err()
        );
        assert!(
            materialize_classification_batch(
                &features,
                &TensorData::new([3, 1], vec![0.; 3]).unwrap(),
                &[0],
                ClassificationFeatureLayout::Preserve,
            )
            .is_err()
        );
    }

    #[test]
    fn classification_batch_preserves_representative_dense_raw_payloads() {
        let cases = [
            (DType::U8, vec![1, 2, 3, 4]),
            (DType::I16, vec![1, 0, 2, 0, 3, 0, 4, 0]),
            (
                DType::F32,
                [0x7fc0_0001u32, 0x8000_0000, 0x3f80_0000, 0x7f80_0000]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect(),
            ),
            (
                DType::BF16,
                vec![0x01, 0x7f, 0x00, 0x80, 0x80, 0x3f, 0x80, 0x7f],
            ),
        ];
        let targets =
            TensorData::from_le_bytes([2], DType::I32, &[4, 0, 0, 0, 2, 0, 0, 0]).unwrap();
        for (dtype, bytes) in cases {
            let features = TensorData::from_le_bytes([2, 2], dtype, &bytes).unwrap();
            let batch = materialize_classification_batch(
                &features,
                &targets,
                &[1],
                ClassificationFeatureLayout::Preserve,
            )
            .unwrap();
            assert_eq!(batch.features.dtype(), dtype);
            assert_eq!(
                batch.features.to_le_bytes().unwrap(),
                bytes[bytes.len() / 2..]
            );
            assert_eq!(batch.targets.dtype(), DType::I32);
            assert_eq!(batch.targets.to_le_bytes().unwrap(), vec![2, 0, 0, 0]);
        }

        assert!(
            materialize_classification_batch(
                &TensorData::scalar(1.),
                &targets,
                &[],
                ClassificationFeatureLayout::Preserve,
            )
            .is_err()
        );
        assert!(
            materialize_classification_batch(
                &TensorData::new([2, 2], vec![0.; 4]).unwrap(),
                &TensorData::new([1], vec![0.]).unwrap(),
                &[],
                ClassificationFeatureLayout::Flatten,
            )
            .is_err()
        );
    }
}
