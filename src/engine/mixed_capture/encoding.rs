//! Canonical RGSM payload and envelope encoding, without runtime resources.

use super::{
    AuthenticatedRecurrentFrontier, CapturedMixedSchedule, MAGIC, MAX_BYTES, ReplayError, VERSION,
    Writer, checksum, codec, validate, write_constants, write_inputs, write_len,
    write_state_bindings, write_states, write_u64s, write_value_bindings,
};
use crate::schedule::artifact::KernelEncodings;

pub(super) fn identity(value: &CapturedMixedSchedule) -> Result<u64, ReplayError> {
    identity_with_kernel_encodings(value, None)
}

pub(super) fn identity_with_kernel_encodings(
    value: &CapturedMixedSchedule,
    kernels: Option<&KernelEncodings>,
) -> Result<u64, ReplayError> {
    #[cfg(test)]
    super::record_prepared_replay_validation(|counts| counts.identity_serializations += 1);
    // Ordinary capture identity is not part of the mixed logical payload.
    let mut writer = Writer::new();
    write_payload(&mut writer, value, kernels)?;
    Ok(writer.out.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    }))
}

impl CapturedMixedSchedule {
    /// Bounded deterministic RGSM encoding. It intentionally excludes every
    /// runtime lease, slot, pointer, generation, and current buffer byte.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ReplayError> {
        validate(self, true)?;
        let mut writer = Writer::new();
        writer.bytes(MAGIC).map_err(codec)?;
        writer.u8(VERSION).map_err(codec)?;
        let identity = identity(self)?;
        writer.u64(identity).map_err(codec)?;
        write_payload(&mut writer, self, None)?;
        finish_envelope(writer)
    }

    pub(super) fn to_bytes_without_identity(&self) -> Result<Vec<u8>, ReplayError> {
        let mut writer = Writer::new();
        write_payload(&mut writer, self, None)?;
        Ok(writer.out)
    }

    #[cfg(test)]
    pub(super) fn payload_with_kernel_encodings(
        &self,
        kernels: Option<&KernelEncodings>,
    ) -> Result<Vec<u8>, ReplayError> {
        let mut writer = Writer::new();
        write_payload(&mut writer, self, kernels)?;
        Ok(writer.out)
    }
}

/// Only the immutable admitted owner can supply a retained mixed identity.
/// Field writers still validate/encode actual kernels and bounded metadata.
pub(super) fn encode_frontier(
    frontier: &AuthenticatedRecurrentFrontier,
) -> Result<Vec<u8>, ReplayError> {
    let mut writer = Writer::new();
    writer.bytes(MAGIC).map_err(codec)?;
    writer.u8(VERSION).map_err(codec)?;
    writer.u64(frontier.capture_identity()).map_err(codec)?;
    write_payload(&mut writer, frontier.capture(), None)?;
    finish_envelope(writer)
}

fn write_payload(
    writer: &mut Writer,
    value: &CapturedMixedSchedule,
    kernels: Option<&KernelEncodings>,
) -> Result<(), ReplayError> {
    write_len(writer, value.schedule.items.len())?;
    for item in &value.schedule.items {
        match kernels {
            Some(kernels) => kernels.write_effect_item(writer, item),
            None => crate::schedule::artifact::write_effect_item(writer, item),
        }
        .map_err(codec)?;
    }
    write_inputs(writer, &value.schedule.inputs)?;
    write_constants(writer, &value.schedule.constants)?;
    write_u64s(writer, &value.schedule.requested)?;
    write_value_bindings(writer, &value.value_bindings)?;
    write_state_bindings(writer, &value.state_bindings)?;
    write_states(writer, &value.states)
}

fn finish_envelope(mut writer: Writer) -> Result<Vec<u8>, ReplayError> {
    if writer
        .out
        .len()
        .checked_add(4)
        .is_none_or(|n| n > MAX_BYTES)
    {
        return Err(ReplayError::Corrupt("RGSM byte limit".into()));
    }
    let sum = checksum(&writer.out);
    writer.u32(sum).map_err(codec)?;
    Ok(writer.out)
}

#[cfg(test)]
#[path = "encoding_tests.rs"]
mod tests;
