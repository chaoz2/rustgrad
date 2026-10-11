//! Deterministic byte windows and a corpus-bound resume envelope.

use super::{BATCH, TARGETS, TIME, TOKENS};
use rustgrad::nn::StateDict;
use rustgrad::{
    CompiledAdamWResumeBundle, CompiledInputBatch, CompiledInputSpec, DType, Result, TensorData,
};
use std::{collections::BTreeMap, error::Error, fs::File, io::Read, path::Path};

const MAX_CORPUS_BYTES: usize = 1 << 20;
const MAX_CHECKPOINT_BYTES: usize = 128 << 20;

pub(super) struct Corpus(Vec<u8>);

impl Corpus {
    pub(super) fn new(bytes: Vec<u8>) -> std::result::Result<Self, Box<dyn Error>> {
        if !(2..=MAX_CORPUS_BYTES).contains(&bytes.len()) {
            return Err("corpus must contain 2..=1048576 bytes".into());
        }
        Ok(Self(bytes))
    }

    pub(super) fn load(path: &Path) -> std::result::Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        if !file.metadata()?.is_file() {
            return Err("corpus must be a regular local file".into());
        }
        let mut bytes = Vec::new();
        file.take((MAX_CORPUS_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        Self::new(bytes)
    }

    pub(super) fn batch_count(&self) -> usize {
        (self.0.len() - 1).div_ceil(TIME).div_ceil(BATCH)
    }

    // `completed_replays` is taken from the authenticated optimizer checkpoint.
    // No independent mutable iterator cursor can drift from a committed step.
    pub(super) fn batch(&self, completed_replays: u64) -> Result<CorpusBatch> {
        let phase = (completed_replays % self.batch_count() as u64) as usize;
        let mut tokens = vec![0_i32; BATCH * TIME];
        let mut targets = vec![-100_i32; BATCH * TIME];
        for row in 0..BATCH {
            let start = (phase * BATCH + row) * TIME;
            let valid = (self.0.len() - 1).saturating_sub(start).min(TIME);
            for column in 0..valid {
                tokens[row * TIME + column] = i32::from(self.0[start + column]);
                targets[row * TIME + column] = i32::from(self.0[start + column + 1]);
            }
        }
        Ok(CorpusBatch {
            tokens: TensorData::from_scalars(
                [BATCH, TIME],
                DType::I32,
                tokens
                    .into_iter()
                    .map(|x| rustgrad::Scalar::I(i64::from(x))),
            )?,
            targets: TensorData::from_scalars(
                [BATCH, TIME],
                DType::I32,
                targets
                    .into_iter()
                    .map(|x| rustgrad::Scalar::I(i64::from(x))),
            )?,
        })
    }

    pub(super) fn save(
        &self,
        path: &Path,
        bundle: &CompiledAdamWResumeBundle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        if bundle.as_bytes().len() > MAX_CHECKPOINT_BYTES - MAX_CORPUS_BYTES - 4096 {
            return Err("training bundle exceeds the example checkpoint budget".into());
        }
        let state = StateDict::from(BTreeMap::from([
            (
                "corpus".into(),
                TensorData::from_le_bytes([self.0.len()], DType::U8, &self.0)?,
            ),
            (
                "training".into(),
                TensorData::from_le_bytes([bundle.as_bytes().len()], DType::U8, bundle.as_bytes())?,
            ),
        ]));
        rustgrad::save_safetensors_file(path, &state, &metadata())?;
        Ok(())
    }

    pub(super) fn restore(
        &self,
        path: &Path,
    ) -> std::result::Result<CompiledAdamWResumeBundle, Box<dyn Error>> {
        let (state, actual_metadata) = rustgrad::load_safetensors_file_with_limits(
            path,
            rustgrad::SafetensorsReadLimits {
                max_file_bytes: MAX_CHECKPOINT_BYTES,
            },
        )?;
        if actual_metadata != metadata() || state.tensors().len() != 2 {
            return Err("not a version-one corpus training checkpoint".into());
        }
        let bytes = |name: &str| -> std::result::Result<Vec<u8>, Box<dyn Error>> {
            let value = state
                .tensors()
                .get(name)
                .ok_or("missing corpus checkpoint field")?;
            if value.dtype() != DType::U8 || value.shape().rank() != 1 {
                return Err("corpus checkpoint fields must be rank-one U8".into());
            }
            Ok(value.to_le_bytes()?)
        };
        // Compare exact bytes, not a collision-prone content hint. The corpus
        // is intentionally part of this bounded, self-contained checkpoint.
        if bytes("corpus")? != self.0 {
            return Err("corpus differs from the saved training data".into());
        }
        Ok(CompiledAdamWResumeBundle::from_bytes(bytes("training")?)?)
    }
}

fn metadata() -> BTreeMap<String, String> {
    BTreeMap::from([("format".into(), "rustgrad-byte-corpus-training-v1".into())])
}

pub(super) struct CorpusBatch {
    tokens: TensorData,
    targets: TensorData,
}

impl CompiledInputBatch for CorpusBatch {
    fn schema() -> &'static [CompiledInputSpec] {
        const SCHEMA: [CompiledInputSpec; 2] = [
            CompiledInputSpec::host_token(TOKENS, &[BATCH, TIME]),
            CompiledInputSpec::new(TARGETS, &[BATCH, TIME], DType::I32),
        ];
        &SCHEMA
    }

    fn into_compiled_inputs(self) -> Result<BTreeMap<String, TensorData>> {
        Ok(BTreeMap::from([
            (TOKENS.into(), self.tokens),
            (TARGETS.into(), self.targets),
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_windows_shift_targets_pad_tails_and_resume_by_replay_count() {
        let bytes: Vec<u8> = (0..(BATCH * TIME + 4)).map(|n| n as u8).collect();
        let corpus = Corpus::new(bytes.clone()).unwrap();
        assert_eq!(corpus.batch_count(), 2);
        let first = corpus.batch(0).unwrap();
        assert_eq!(first.tokens.scalar_at(0).as_i64(), 0);
        assert_eq!(first.targets.scalar_at(0).as_i64(), 1);
        let tail = corpus.batch(1).unwrap();
        for index in 0..BATCH * TIME {
            assert_eq!(
                tail.tokens.scalar_at(index).as_i64(),
                if index < 3 {
                    (BATCH * TIME + index) as i64
                } else {
                    0
                }
            );
            assert_eq!(
                tail.targets.scalar_at(index).as_i64(),
                if index < 3 {
                    (BATCH * TIME + index + 1) as i64
                } else {
                    -100
                }
            );
        }
        assert_eq!(corpus.batch(2).unwrap().tokens, first.tokens);
        assert_eq!(corpus.batch(u64::MAX).unwrap().targets, tail.targets);
        assert_eq!(corpus.0, bytes);
        assert!(Corpus::new(vec![]).is_err());
        assert!(Corpus::new(vec![0]).is_err());
        assert!(Corpus::new(vec![0; MAX_CORPUS_BYTES + 1]).is_err());
        let minimum = Corpus::new(vec![255, 128]).unwrap().batch(0).unwrap();
        assert_eq!(minimum.tokens.scalar_at(0).as_i64(), 255);
        assert_eq!(minimum.targets.scalar_at(0).as_i64(), 128);
    }
}
