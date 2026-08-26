//! Pure static-CPU classification summaries for evaluated module logits.

use crate::{DType, Error, Result, TensorData};

/// Deterministic first-tie classification results for one evaluated batch.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassificationSummary {
    predictions: Vec<usize>,
    correct_count: usize,
    total_count: usize,
}

impl ClassificationSummary {
    pub fn predictions(&self) -> &[usize] {
        &self.predictions
    }
    pub const fn correct_count(&self) -> usize {
        self.correct_count
    }
    pub const fn total_count(&self) -> usize {
        self.total_count
    }
    /// `None` for a legal empty batch; otherwise `correct / total`.
    pub fn accuracy(&self) -> Option<f64> {
        (self.total_count != 0).then(|| self.correct_count as f64 / self.total_count as f64)
    }
}

/// Deterministic binary-logit results for one evaluated batch.
#[derive(Clone, Debug, PartialEq)]
pub struct BinaryClassificationSummary {
    predictions: Vec<u8>,
    correct_count: usize,
    total_count: usize,
}

impl BinaryClassificationSummary {
    pub fn predictions(&self) -> &[u8] {
        &self.predictions
    }
    pub const fn correct_count(&self) -> usize {
        self.correct_count
    }
    pub const fn total_count(&self) -> usize {
        self.total_count
    }
    /// `None` for a legal empty batch; otherwise `correct / total`.
    pub fn accuracy(&self) -> Option<f64> {
        (self.total_count != 0).then(|| self.correct_count as f64 / self.total_count as f64)
    }
}

/// Summarizes rank-two F32 binary logits `[batch, 1]` against F32 `{0, 1}` targets.
///
/// A logit greater than or equal to zero predicts the positive class, matching
/// the probability threshold `sigmoid(logit) >= 0.5`. Non-finite logits and
/// targets outside the exact binary target set are rejected rather than given
/// an implicit host policy.
pub fn summarize_binary_classification(
    logits: &TensorData,
    targets: &TensorData,
) -> Result<BinaryClassificationSummary> {
    if logits.dtype() != DType::F32 || logits.shape().rank() != 2 || logits.shape().dims()[1] != 1 {
        return Err(error("logits must be rank-two F32 with one class lane"));
    }
    if targets.dtype() != DType::F32 || targets.shape() != logits.shape() {
        return Err(error("targets must be F32 and exactly match binary logits"));
    }
    let batch = logits.shape().dims()[0];
    let mut predictions = Vec::with_capacity(batch);
    let mut correct_count = 0;
    for row in 0..batch {
        let logit = logits.scalar_at(row).as_f64();
        let target = targets.scalar_at(row).as_f64();
        if !logit.is_finite() {
            return Err(error("logits must be finite"));
        }
        if target != 0.0 && target != 1.0 {
            return Err(error("binary targets must be exactly zero or one"));
        }
        let prediction = u8::from(logit >= 0.0);
        predictions.push(prediction);
        correct_count += usize::from(prediction as f64 == target);
    }
    Ok(BinaryClassificationSummary {
        predictions,
        correct_count,
        total_count: batch,
    })
}

/// Summarizes rank-two F32 logits `[batch, classes]` against integer targets.
///
/// Equal maxima choose the lowest class index. Non-finite logits and targets
/// outside the class range are rejected rather than assigned a host policy.
pub fn summarize_classification(
    logits: &TensorData,
    targets: &TensorData,
) -> Result<ClassificationSummary> {
    if logits.dtype() != DType::F32 || logits.shape().rank() != 2 || logits.shape().dims()[1] == 0 {
        return Err(error(
            "logits must be rank-two F32 with a nonzero class dimension",
        ));
    }
    if targets.shape().rank() != 1
        || !targets.dtype().is_integer()
        || targets.shape().dims()[0] != logits.shape().dims()[0]
    {
        return Err(error(
            "targets must be a rank-one integer tensor matching the logits batch",
        ));
    }
    let batch = logits.shape().dims()[0];
    let classes = logits.shape().dims()[1];
    let mut predictions = Vec::with_capacity(batch);
    let mut correct_count = 0;
    for row in 0..batch {
        let mut best = 0usize;
        let mut best_value = logits.scalar_at(row * classes).as_f64();
        if !best_value.is_finite() {
            return Err(error("logits must be finite"));
        }
        for class in 1..classes {
            let value = logits.scalar_at(row * classes + class).as_f64();
            if !value.is_finite() {
                return Err(error("logits must be finite"));
            }
            if value > best_value {
                best = class;
                best_value = value;
            }
        }
        let target = targets.scalar_at(row).as_i64();
        if target < 0 || target as usize >= classes {
            return Err(error("target class is outside logits classes"));
        }
        predictions.push(best);
        correct_count += usize::from(best == target as usize);
    }
    Ok(ClassificationSummary {
        predictions,
        correct_count,
        total_count: batch,
    })
}

fn error(reason: impl Into<String>) -> Error {
    Error::SessionTraining {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Scalar, TensorData};

    #[test]
    fn summary_is_first_tie_typed_and_empty_explicit() {
        let logits = TensorData::new([2, 3], vec![2., 2., 1., -1., 3., 3.]).unwrap();
        let targets =
            TensorData::from_scalars([2], DType::I16, [Scalar::I(0), Scalar::I(2)]).unwrap();
        let summary = summarize_classification(&logits, &targets).unwrap();
        assert_eq!(summary.predictions(), &[0, 1]);
        assert_eq!(summary.correct_count(), 1);
        assert_eq!(summary.accuracy(), Some(0.5));
        let empty = summarize_classification(
            &TensorData::new([0, 2], vec![]).unwrap(),
            &TensorData::from_scalars([0], DType::U8, []).unwrap(),
        )
        .unwrap();
        assert_eq!(empty.accuracy(), None);
        assert!(
            summarize_classification(&TensorData::new([2], vec![1., 2.]).unwrap(), &targets)
                .is_err()
        );
        assert!(
            summarize_classification(&logits, &TensorData::new([2], vec![0., 1.]).unwrap())
                .is_err()
        );
        assert!(
            summarize_classification(
                &logits,
                &TensorData::from_scalars([2], DType::U8, [Scalar::U(0), Scalar::U(3)]).unwrap()
            )
            .is_err()
        );
        for dtype in [
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
        ] {
            let targets =
                TensorData::from_scalars([2], dtype, [Scalar::I(0), Scalar::I(1)]).unwrap();
            assert_eq!(
                summarize_classification(&logits, &targets)
                    .unwrap()
                    .predictions(),
                &[0, 1]
            );
        }
        assert!(
            summarize_classification(
                &TensorData::new([1, 2], vec![f32::NAN, 0.]).unwrap(),
                &TensorData::from_scalars([1], DType::U8, [Scalar::U(0)]).unwrap(),
            )
            .is_err()
        );
    }

    #[test]
    fn binary_summary_uses_logit_zero_threshold_and_checked_f32_targets() {
        let logits = TensorData::new([3, 1], vec![-2., 0., 3.]).unwrap();
        let targets = TensorData::new([3, 1], vec![0., 1., 0.]).unwrap();
        let summary = summarize_binary_classification(&logits, &targets).unwrap();
        assert_eq!(summary.predictions(), &[0, 1, 1]);
        assert_eq!(summary.correct_count(), 2);
        assert_eq!(summary.accuracy(), Some(2. / 3.));
        let empty = summarize_binary_classification(
            &TensorData::new([0, 1], vec![]).unwrap(),
            &TensorData::new([0, 1], vec![]).unwrap(),
        )
        .unwrap();
        assert_eq!(empty.accuracy(), None);
        assert!(
            summarize_binary_classification(
                &TensorData::new([3], vec![-2., 0., 3.]).unwrap(),
                &targets,
            )
            .is_err()
        );
        assert!(
            summarize_binary_classification(
                &logits,
                &TensorData::new([3], vec![0., 1., 0.]).unwrap(),
            )
            .is_err()
        );
        assert!(
            summarize_binary_classification(
                &logits,
                &TensorData::new([3, 1], vec![0., 0.5, 1.]).unwrap(),
            )
            .is_err()
        );
        assert!(
            summarize_binary_classification(
                &TensorData::new([1, 1], vec![f32::NAN]).unwrap(),
                &TensorData::new([1, 1], vec![0.]).unwrap(),
            )
            .is_err()
        );
    }
}
