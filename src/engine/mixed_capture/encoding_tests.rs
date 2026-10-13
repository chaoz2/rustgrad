use super::super::{HEADER_LEN, prepared_replay_validation_counts};
use super::*;
use crate::{DType, EffectGraph, Storage, TensorData, schedule_effects};
use std::sync::Arc;

fn capture(dtype: DType, shape: &[usize]) -> CapturedMixedSchedule {
    let mut effects = EffectGraph::default();
    let value = TensorData::zeros_with_dtype(shape.to_vec(), dtype).unwrap();
    let target = effects.insert(40, value.clone()).unwrap();
    let source = effects.insert(41, value).unwrap();
    let next = effects.assign(&target, &source).unwrap();
    let schedule = schedule_effects(&effects).unwrap();
    let ordinary = crate::CapturedSchedule {
        items: schedule.items.clone(),
        inputs: vec![],
        constants: Default::default(),
        quantized_constants: Default::default(),
        requested_passthroughs: vec![],
        requested: vec![],
        identity: 0,
        symbolic: None,
        specialized_from: None,
    };
    CapturedMixedSchedule::from_parts(
        ordinary,
        &schedule,
        vec![
            target.state().clone(),
            source.state().clone(),
            next.state().clone(),
        ],
    )
    .unwrap()
}

fn released_identity(value: &CapturedMixedSchedule) -> Result<u64, ReplayError> {
    super::super::record_prepared_replay_validation(|counts| counts.identity_serializations += 1);
    let mut writer = Writer::new();
    write_len(&mut writer, value.schedule.items.len())?;
    for item in &value.schedule.items {
        crate::schedule::artifact::write_effect_item(&mut writer, item).map_err(codec)?;
    }
    write_inputs(&mut writer, &value.schedule.inputs)?;
    write_constants(&mut writer, &value.schedule.constants)?;
    write_u64s(&mut writer, &value.schedule.requested)?;
    write_value_bindings(&mut writer, &value.value_bindings)?;
    write_state_bindings(&mut writer, &value.state_bindings)?;
    write_states(&mut writer, &value.states)?;
    Ok(writer.out.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    }))
}

// Frozen released public writer and identity calculation, independent of the
// shared payload/envelope helpers. Keep both size-failure points visible here.
fn released_bytes(value: &CapturedMixedSchedule) -> Result<Vec<u8>, ReplayError> {
    validate(value, true)?;
    let mut writer = Writer::new();
    writer.bytes(MAGIC).map_err(codec)?;
    writer.u8(VERSION).map_err(codec)?;
    writer.u64(released_identity(value)?).map_err(codec)?;
    write_len(&mut writer, value.schedule.items.len())?;
    for item in &value.schedule.items {
        crate::schedule::artifact::write_effect_item(&mut writer, item).map_err(codec)?;
    }
    write_inputs(&mut writer, &value.schedule.inputs)?;
    write_constants(&mut writer, &value.schedule.constants)?;
    write_u64s(&mut writer, &value.schedule.requested)?;
    write_value_bindings(&mut writer, &value.value_bindings)?;
    write_state_bindings(&mut writer, &value.state_bindings)?;
    write_states(&mut writer, &value.states)?;
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

#[test]
fn frontier_encoding_preserves_bytes_with_one_kernel_pass() {
    for dtype in [
        DType::Bool,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        for shape in [vec![], vec![0, 2], vec![2]] {
            let source = Arc::new(capture(dtype, &shape));
            let frontier = AuthenticatedRecurrentFrontier::authenticate(source.clone()).unwrap();
            assert_ne!(frontier.capture_identity(), source.schedule.identity);
            let counts = prepared_replay_validation_counts();
            let encodings = crate::uop::artifact::schedule_kernel_encoding_count();
            let actual = frontier.to_bytes().unwrap();
            assert_eq!(prepared_replay_validation_counts(), counts);
            assert_eq!(
                crate::uop::artifact::schedule_kernel_encoding_count() - encodings,
                source.schedule.items.len(),
            );
            let counts = prepared_replay_validation_counts();
            let encodings = crate::uop::artifact::schedule_kernel_encoding_count();
            let reference = released_bytes(&source).unwrap();
            let after = prepared_replay_validation_counts();
            assert_eq!(
                after.mixed_capture_validations,
                counts.mixed_capture_validations + 1
            );
            assert_eq!(after.schedule_rekeys, counts.schedule_rekeys + 1);
            assert_eq!(
                after.identity_serializations,
                counts.identity_serializations + 1
            );
            assert_eq!(
                crate::uop::artifact::schedule_kernel_encoding_count() - encodings,
                source.schedule.items.len() * 3,
            );
            assert_eq!(actual, reference, "{dtype:?} {shape:?}");
            assert_eq!(source.to_bytes().unwrap(), reference);
            let decoded = AuthenticatedRecurrentFrontier::from_bytes(&actual).unwrap();
            assert_eq!(decoded.capture_identity(), frontier.capture_identity());
            assert_eq!(decoded.to_bytes().unwrap(), actual);
        }
    }
}

#[test]
fn frontier_encoding_keeps_public_mutation_and_import_checks() {
    let mut raw = capture(DType::F32, &[2]);
    raw.schedule.constants.insert(
        99,
        TensorData::from_storage([3], Storage::BF16(vec![0x8000, 0x7fc1, 0x7f80])).unwrap(),
    );
    let frontier = AuthenticatedRecurrentFrontier::authenticate(Arc::new(raw)).unwrap();
    let bytes = frontier.to_bytes().unwrap();
    assert_eq!(bytes, released_bytes(frontier.capture()).unwrap());
    let decoded = CapturedMixedSchedule::from_bytes(&bytes).unwrap();
    assert_eq!(
        decoded.schedule.constants[&99].storage(),
        frontier.capture().schedule.constants[&99].storage()
    );

    let mut changed = frontier.capture().clone();
    changed.states.push(changed.states[0].clone());
    let expected = ReplayError::Corrupt("duplicate logical state version".into());
    assert_eq!(changed.to_bytes().unwrap_err(), expected);
    assert_eq!(released_bytes(&changed).unwrap_err(), expected);
    assert_eq!(
        AuthenticatedRecurrentFrontier::authenticate(Arc::new(changed))
            .err()
            .unwrap(),
        expected
    );
    assert_eq!(frontier.to_bytes().unwrap(), bytes);

    for (offset, expected) in [(4, "RGSM version"), (5, "RGSM identity")] {
        let mut corrupt = bytes.clone();
        corrupt[offset] ^= 0x80;
        let body = corrupt.len() - 4;
        let sum = checksum(&corrupt[..body]).to_le_bytes();
        corrupt[body..].copy_from_slice(&sum);
        assert_eq!(
            AuthenticatedRecurrentFrontier::from_bytes(&corrupt)
                .err()
                .unwrap(),
            ReplayError::Corrupt(expected.into()),
        );
    }
    for version in [1, 2] {
        let mut legacy = bytes.clone();
        legacy[4] = version;
        let body = legacy.len() - 4;
        let sum = checksum(&legacy[..body]).to_le_bytes();
        legacy[body..].copy_from_slice(&sum);
        let imported = AuthenticatedRecurrentFrontier::from_bytes(&legacy).unwrap();
        assert_eq!(imported.to_bytes().unwrap(), bytes);
    }
}

#[test]
fn admitted_payload_still_checks_header_and_checksum_capacity() {
    // Exercise the real released 64 MiB limit, not a smaller surrogate. U8
    // writes one bounded raw slice, avoiding per-element setup or execution.
    let mut calibration = capture(DType::F32, &[2]);
    calibration.schedule.constants.insert(
        99,
        TensorData::from_storage([1], Storage::U8(vec![0])).unwrap(),
    );
    let overhead = calibration.to_bytes_without_identity().unwrap().len() - 1;
    drop(calibration);
    for (payload_len, error) in [
        (MAX_BYTES - HEADER_LEN - 4, None),
        (MAX_BYTES - HEADER_LEN - 3, Some("RGSM byte limit")),
        (MAX_BYTES - HEADER_LEN, Some("RGSM byte limit")),
        (
            MAX_BYTES - HEADER_LEN + 1,
            Some("artifact: Format(\"byte limit\")"),
        ),
    ] {
        let mut raw = capture(DType::F32, &[2]);
        let elements = payload_len - overhead;
        raw.schedule.constants.insert(
            99,
            TensorData::from_storage([elements], Storage::U8(vec![0; elements])).unwrap(),
        );
        let frontier = AuthenticatedRecurrentFrontier::authenticate(Arc::new(raw)).unwrap();
        let actual = frontier.to_bytes();
        let reference = released_bytes(frontier.capture());
        match error {
            Some(error) => {
                let expected = ReplayError::Corrupt(error.into());
                assert_eq!(actual.unwrap_err(), expected);
                assert_eq!(reference.unwrap_err(), expected);
            }
            None => {
                let actual = actual.unwrap();
                let reference = reference.unwrap();
                assert_eq!(actual.len(), MAX_BYTES);
                assert!(actual == reference, "near-limit canonical bytes differ");
            }
        }
    }
}

#[test]
fn frontier_encoding_retains_no_additional_capture_owners() {
    let source = Arc::new(capture(DType::F32, &[2]));
    let weak = Arc::downgrade(&source);
    let frontier = AuthenticatedRecurrentFrontier::authenticate(source.clone()).unwrap();
    let kernel = frontier.capture().schedule.items[0]
        .kernel
        .allocation_liveness_probe();
    let bytes = frontier.to_bytes().unwrap();
    assert_eq!(Arc::strong_count(&source), 2);
    drop(source);
    assert!(weak.upgrade().is_some() && kernel());
    drop(frontier);
    assert!(weak.upgrade().is_none() && !kernel());
    assert!(!bytes.is_empty());
}
