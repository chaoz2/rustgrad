//! CIFAR-10 binary record decoding and pure normalization.

use super::{bad, checked_exact_len};
use crate::{DType, Result, Scalar, TensorData};

const CHANNELS: usize = 3;
const HEIGHT: usize = 32;
const WIDTH: usize = 32;
const PIXELS_PER_CHANNEL: usize = HEIGHT * WIDTH;
const IMAGE_BYTES: usize = CHANNELS * PIXELS_PER_CHANNEL;
const RECORD_BYTES: usize = 1 + IMAGE_BYTES;

/// Exact CIFAR-10 records stored as U8 NCHW images and U8 class labels.
#[derive(Clone, Debug, PartialEq)]
pub struct Cifar10 {
    pub images: TensorData,
    pub labels: TensorData,
}

impl Cifar10 {
    /// Converts U8 pixels to F32 and applies `(pixel / 255 - mean) / std`
    /// independently to each NCHW channel.
    pub fn normalized_f32(
        &self,
        mean: [f32; CHANNELS],
        std: [f32; CHANNELS],
    ) -> Result<TensorData> {
        for channel in 0..CHANNELS {
            if !mean[channel].is_finite() {
                return Err(bad(format!(
                    "CIFAR-10 mean for channel {channel} must be finite"
                )));
            }
            if !std[channel].is_finite() || std[channel] <= 0. {
                return Err(bad(format!(
                    "CIFAR-10 standard deviation for channel {channel} must be finite and positive"
                )));
            }
        }
        TensorData::from_scalars(
            self.images.shape().clone(),
            DType::F32,
            (0..self.images.len()).map(|index| {
                let channel = (index / PIXELS_PER_CHANNEL) % CHANNELS;
                let pixel = self.images.scalar_at(index).as_f64() / 255.;
                Scalar::F((pixel - f64::from(mean[channel])) / f64::from(std[channel]))
            }),
        )
    }
}

/// Parses exactly `record_count` local CIFAR-10 binary records.
///
/// Every record must contain one label in `0..=9`, followed by 3072 bytes in
/// channel-major red, green, blue order. Extra bytes are rejected.
pub fn parse_cifar10(bytes: &[u8], record_count: usize) -> Result<Cifar10> {
    checked_exact_len(bytes.len(), record_count, RECORD_BYTES, "CIFAR-10")?;
    let image_len = record_count
        .checked_mul(IMAGE_BYTES)
        .ok_or_else(|| bad("CIFAR-10 image length overflow"))?;
    let mut images = Vec::with_capacity(image_len);
    let mut labels = Vec::with_capacity(record_count);
    for (record_index, record) in bytes.chunks_exact(RECORD_BYTES).enumerate() {
        let label = record[0];
        if label > 9 {
            return Err(bad(format!(
                "CIFAR-10 label {label} at record {record_index} is outside 0..=9"
            )));
        }
        labels.push(label);
        images.extend_from_slice(&record[1..]);
    }
    Ok(Cifar10 {
        images: TensorData::from_le_bytes(
            [record_count, CHANNELS, HEIGHT, WIDTH],
            DType::U8,
            &images,
        )?,
        labels: TensorData::from_le_bytes([record_count], DType::U8, &labels)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(label: u8, red: u8, green: u8, blue: u8) -> Vec<u8> {
        let mut output = Vec::with_capacity(RECORD_BYTES);
        output.push(label);
        output.extend(std::iter::repeat_n(red, PIXELS_PER_CHANNEL));
        output.extend(std::iter::repeat_n(green, PIXELS_PER_CHANNEL));
        output.extend(std::iter::repeat_n(blue, PIXELS_PER_CHANNEL));
        output
    }

    #[test]
    fn cifar_channel_layout_and_normalization_are_exact() {
        let mut bytes = record(2, 0, 127, 255);
        bytes.extend(record(9, 1, 2, 3));
        let dataset = parse_cifar10(&bytes, 2).unwrap();
        assert_eq!(dataset.images.shape().dims(), &[2, 3, 32, 32]);
        assert_eq!(dataset.labels.to_le_bytes().unwrap(), vec![2, 9]);
        let raw = dataset.images.to_le_bytes().unwrap();
        assert_eq!(raw[0], 0);
        assert_eq!(raw[PIXELS_PER_CHANNEL], 127);
        assert_eq!(raw[2 * PIXELS_PER_CHANNEL], 255);
        assert_eq!(raw[IMAGE_BYTES], 1);

        let normalized = dataset
            .normalized_f32([0., 0.5, 1.], [1., 0.5, 2.])
            .unwrap();
        assert_eq!(normalized.values()[0], 0.);
        assert!((normalized.values()[PIXELS_PER_CHANNEL] + 0.003_921_569).abs() < 1e-7);
        assert_eq!(normalized.values()[2 * PIXELS_PER_CHANNEL], 0.);
    }

    #[test]
    fn cifar_rejects_count_length_label_and_normalization_errors() {
        let valid = record(0, 0, 0, 0);
        assert!(parse_cifar10(&valid[..valid.len() - 1], 1).is_err());
        let mut trailing = valid.clone();
        trailing.push(0);
        assert!(parse_cifar10(&trailing, 1).is_err());
        assert!(parse_cifar10(&valid, 2).is_err());

        let invalid_label = record(10, 0, 0, 0);
        assert!(parse_cifar10(&invalid_label, 1).is_err());
        let dataset = parse_cifar10(&valid, 1).unwrap();
        assert!(dataset.normalized_f32([f32::NAN; 3], [1.; 3]).is_err());
        assert!(dataset.normalized_f32([0.; 3], [1., 0., 1.]).is_err());
    }

    #[test]
    fn empty_cifar_dataset_is_typed_and_shaped() {
        let dataset = parse_cifar10(&[], 0).unwrap();
        assert_eq!(dataset.images.shape().dims(), &[0, 3, 32, 32]);
        assert_eq!(dataset.labels.shape().dims(), &[0]);
        assert!(dataset.images.to_le_bytes().unwrap().is_empty());
    }
}
