//! Public local-file acceptance for the CPU session interchange route.

use rustgrad::interop::host::{
    NpyError, NpyFileError, NpyReadLimits, load_npy_file, load_npy_file_with_limits, save_npy_file,
};
use rustgrad::{
    CpuSession, DType, Metadata, StateDict, TensorData, load_safetensors_file,
    save_safetensors_file,
};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

fn directory() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "rustgrad-local-file-interchange-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&directory).unwrap();
    directory
}

#[test]
fn local_npy_to_cpu_session_to_npy_preserves_result_bits() {
    let directory = directory();
    let input_path = directory.join("input.npy");
    let output_path = directory.join("output.npy");
    let input =
        TensorData::from_le_bytes([2], DType::F32, &[0, 0, 0x80, 0x3f, 0, 0, 0, 0x40]).unwrap();
    save_npy_file(&input_path, &input).unwrap();

    let mut session = CpuSession::new();
    let input = session
        .variable_data(load_npy_file(&input_path).unwrap())
        .unwrap();
    let scale = session
        .constant(
            TensorData::from_le_bytes([2], DType::F32, &[0, 0, 0, 0x40, 0, 0, 0x40, 0x40]).unwrap(),
        )
        .unwrap();
    let result = session.mul(&input, &scale).unwrap();
    let realized = session.realize(&result).unwrap();
    save_npy_file(&output_path, &realized).unwrap();
    assert_eq!(
        load_npy_file(&output_path).unwrap().to_le_bytes().unwrap(),
        [0, 0, 0, 0x40, 0, 0, 0xc0, 0x40]
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn static_safetensors_state_feeds_the_same_cpu_session_route() {
    let directory = directory();
    let path = directory.join("weights.safetensors");
    let weight =
        TensorData::from_le_bytes([2], DType::F32, &[0, 0, 0, 0x40, 0, 0, 0x40, 0x40]).unwrap();
    let mut state = StateDict::new();
    state.insert("weight".into(), weight);
    save_safetensors_file(&path, &state, &Metadata::new()).unwrap();

    let (loaded, metadata) = load_safetensors_file(&path).unwrap();
    assert_eq!(metadata, BTreeMap::new());
    let mut session = CpuSession::new();
    let input = session.tensor([2], [3.0, 4.0]).unwrap();
    let weights = session.constant(loaded["weight"].clone()).unwrap();
    let result = session.mul(&input, &weights).unwrap();
    assert_eq!(session.realize(&result).unwrap().to_vec_f64(), [6.0, 12.0]);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn scalar_empty_and_limited_or_malformed_files_fail_or_round_trip_explicitly() {
    let directory = directory();
    for (name, tensor) in [
        (
            "scalar",
            TensorData::from_le_bytes([], DType::F32, &[0, 0, 0, 0x80]).unwrap(),
        ),
        (
            "empty",
            TensorData::from_le_bytes([0, 2], DType::U16, &[]).unwrap(),
        ),
    ] {
        let path = directory.join(format!("{name}.npy"));
        save_npy_file(&path, &tensor).unwrap();
        assert_eq!(
            load_npy_file(&path).unwrap().to_le_bytes().unwrap(),
            tensor.to_le_bytes().unwrap()
        );
    }
    let malformed = directory.join("malformed.npy");
    fs::write(&malformed, b"not-npy").unwrap();
    assert!(matches!(
        load_npy_file(&malformed),
        Err(NpyFileError::Format(NpyError::Magic))
    ));
    assert!(matches!(
        load_npy_file_with_limits(
            &malformed,
            NpyReadLimits {
                max_file_bytes: 1,
                ..NpyReadLimits::default()
            }
        ),
        Err(NpyFileError::Format(NpyError::Limit {
            limit: "file bytes",
            ..
        }))
    ));
    fs::remove_dir_all(directory).unwrap();
}
