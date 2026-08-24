//! RGMB: bounded logical envelopes around existing RGSM artifacts.
//!
//! The batch format deliberately knows no mixed-schedule fields. Each entry is
//! an opaque, independently validated RGSM byte stream, which keeps one
//! canonical codec for schedule/state/effect semantics.
use super::CapturedMixedBatch;
use crate::ReplayError;
use crate::engine::mixed_capture::CapturedMixedSchedule;
use crate::uop::artifact::checksum;

const MAGIC: &[u8; 4] = b"RGMB";
const VERSION: u8 = 1;
const MAX_BYTES: usize = 64 << 20;
const MAX_ENTRIES: usize = 1 << 16;
const MAX_ENTRY_BYTES: usize = 64 << 20;

/// Structured RGMB envelope failure. Embedded schedules retain their typed
/// RGSM [`ReplayError`] instead of being reinterpreted by this wrapper.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MixedBatchArtifactError {
    Corrupt(&'static str),
    Embedded(ReplayError),
}

impl std::fmt::Display for MixedBatchArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RGMB artifact: {self:?}")
    }
}
impl std::error::Error for MixedBatchArtifactError {}

impl From<ReplayError> for MixedBatchArtifactError {
    fn from(value: ReplayError) -> Self {
        Self::Embedded(value)
    }
}

impl CapturedMixedBatch {
    /// Encodes this ordered logical batch as a bounded RGMB envelope. Runtime
    /// leases, slot/generation identities, bytes, compiled code, and traces do
    /// not participate.
    pub fn to_bytes(&self) -> Result<Vec<u8>, MixedBatchArtifactError> {
        let captures = self.captures();
        if captures.is_empty() || captures.len() > MAX_ENTRIES {
            return Err(MixedBatchArtifactError::Corrupt("entry count"));
        }
        let entries = captures
            .iter()
            .map(CapturedMixedSchedule::to_bytes)
            .collect::<Result<Vec<_>, _>>()?;
        let body_len = 4usize
            .checked_add(1)
            .and_then(|n| n.checked_add(8))
            .and_then(|n| n.checked_add(4))
            .and_then(|n| {
                entries.iter().try_fold(n, |size, entry| {
                    if entry.len() > MAX_ENTRY_BYTES {
                        None
                    } else {
                        size.checked_add(4)?.checked_add(entry.len())
                    }
                })
            })
            .ok_or(MixedBatchArtifactError::Corrupt("length"))?;
        if body_len.checked_add(4).is_none_or(|n| n > MAX_BYTES) {
            return Err(MixedBatchArtifactError::Corrupt("byte limit"));
        }
        let mut out = Vec::with_capacity(body_len + 4);
        out.extend_from_slice(MAGIC);
        out.push(VERSION);
        out.extend_from_slice(&self.identity().to_le_bytes());
        out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for entry in entries {
            out.extend_from_slice(&(entry.len() as u32).to_le_bytes());
            out.extend_from_slice(&entry);
        }
        out.extend_from_slice(&checksum(&out).to_le_bytes());
        Ok(out)
    }

    /// Decodes every embedded RGSM artifact and validates the resulting batch
    /// before returning it. No runtime object is accepted or observed here.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MixedBatchArtifactError> {
        if bytes.len() < 21 || bytes.len() > MAX_BYTES {
            return Err(MixedBatchArtifactError::Corrupt("length"));
        }
        let body = bytes.len() - 4;
        let stored_sum = u32::from_le_bytes(
            bytes[body..]
                .try_into()
                .map_err(|_| MixedBatchArtifactError::Corrupt("checksum"))?,
        );
        if checksum(&bytes[..body]) != stored_sum {
            return Err(MixedBatchArtifactError::Corrupt("checksum"));
        }
        let mut cursor = 0usize;
        let take = |cursor: &mut usize, count: usize| -> Result<&[u8], MixedBatchArtifactError> {
            let end = cursor
                .checked_add(count)
                .ok_or(MixedBatchArtifactError::Corrupt("overflow"))?;
            let slice = bytes[..body]
                .get(*cursor..end)
                .ok_or(MixedBatchArtifactError::Corrupt("truncated"))?;
            *cursor = end;
            Ok(slice)
        };
        if take(&mut cursor, 4)? != MAGIC {
            return Err(MixedBatchArtifactError::Corrupt("magic"));
        }
        if take(&mut cursor, 1)?[0] != VERSION {
            return Err(MixedBatchArtifactError::Corrupt("version"));
        }
        let stored_identity = u64::from_le_bytes(take(&mut cursor, 8)?.try_into().unwrap());
        let count = u32::from_le_bytes(take(&mut cursor, 4)?.try_into().unwrap()) as usize;
        if count == 0 || count > MAX_ENTRIES {
            return Err(MixedBatchArtifactError::Corrupt("entry count"));
        }
        let mut captures = Vec::with_capacity(count);
        for _ in 0..count {
            let length = u32::from_le_bytes(take(&mut cursor, 4)?.try_into().unwrap()) as usize;
            if length == 0 || length > MAX_ENTRY_BYTES {
                return Err(MixedBatchArtifactError::Corrupt("entry length"));
            }
            captures.push(CapturedMixedSchedule::from_bytes(take(
                &mut cursor,
                length,
            )?)?);
        }
        if cursor != body {
            return Err(MixedBatchArtifactError::Corrupt("trailing bytes"));
        }
        let batch = CapturedMixedBatch::new(captures)?;
        if batch.identity() != stored_identity {
            return Err(MixedBatchArtifactError::Corrupt("identity"));
        }
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CapturedSchedule, EffectGraph, Shape, Storage, TensorData, schedule_effects};
    use std::collections::BTreeMap;

    fn tensor(shape: impl Into<Shape>, storage: Storage) -> TensorData {
        TensorData::from_storage(shape, storage).unwrap()
    }

    fn batch() -> CapturedMixedBatch {
        let mut graph = EffectGraph::default();
        let base = graph
            .insert(10, tensor([2], Storage::F16(vec![1, 2])))
            .unwrap();
        let source = graph
            .insert(11, tensor([2], Storage::F16(vec![9, 8])))
            .unwrap();
        let next = graph.assign(&base, &source).unwrap();
        let schedule = schedule_effects(&graph).unwrap();
        let capture = CapturedMixedSchedule::from_parts(
            CapturedSchedule {
                items: schedule.items.clone(),
                inputs: vec![],
                constants: BTreeMap::new(),
                quantized_constants: BTreeMap::new(),
                requested: vec![],
                identity: 0,
                symbolic: None,
                specialized_from: None,
            },
            &schedule,
            vec![
                base.state().clone(),
                source.state().clone(),
                next.state().clone(),
            ],
        )
        .unwrap();
        CapturedMixedBatch::new(vec![capture]).unwrap()
    }

    #[test]
    fn rgmb_is_deterministic_and_embeds_only_rgsm() {
        let original = batch();
        let bytes = original.to_bytes().unwrap();
        assert_eq!(bytes, original.to_bytes().unwrap());
        let decoded = CapturedMixedBatch::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.identity(), original.identity());
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
    }

    #[test]
    fn rgmb_rejects_envelope_and_embedded_corruption() {
        let bytes = batch().to_bytes().unwrap();
        let mut cases = vec![
            bytes[..bytes.len() - 1].to_vec(),
            {
                let mut x = bytes.clone();
                x[0] ^= 1;
                x
            },
            {
                let mut x = bytes.clone();
                x[4] = 2;
                x
            },
            {
                let mut x = bytes.clone();
                x[17] = 0;
                x[18] = 0;
                x[19] = 0;
                x[20] = 0;
                x
            },
            {
                let mut x = bytes.clone();
                x[21] ^= 1;
                x
            },
        ];
        for mut bad in cases.drain(..) {
            // Every mutation must first pass checksum verification to reach
            // the intended structural decoder boundary.
            let sum = checksum(&bad[..bad.len() - 4]);
            let end = bad.len();
            bad[end - 4..].copy_from_slice(&sum.to_le_bytes());
            assert!(CapturedMixedBatch::from_bytes(&bad).is_err());
        }
    }
}
