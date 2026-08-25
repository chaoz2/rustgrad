//! Public restricted-Torch-file to strict-module CPU workflow acceptance.

use rustgrad::nn::Linear;
use rustgrad::{
    Backend, CpuBackend, DType, Graph, Module, ModuleStateDict, TensorData, TorchStateFileError,
    TorchStateReadLimits, load_torch_state_dict, load_torch_state_dict_with_limits,
    load_torch_state_file, load_torch_state_file_strict_with_limits,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

type StateEntry<'a> = (&'a str, &'a str, &'a str, &'a [u8], &'a [u8], &'a [u8]);

fn directory() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "rustgrad-torch-state-file-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).unwrap();
    path
}

// This is a compact independent protocol-2/stored-ZIP fixture encoder. It
// deliberately does not call the RustGrad importer or safetensors writer.
fn state_pickle(entries: &[StateEntry<'_>]) -> Vec<u8> {
    let mut pickle = vec![0x80, 2, b'c'];
    pickle.extend_from_slice(b"collections\nOrderedDict\n)R");
    for (key, storage, dtype, device, shape, strides) in entries {
        pickle.extend_from_slice(b"X");
        pickle.extend_from_slice(&(key.len() as u32).to_le_bytes());
        pickle.extend_from_slice(key.as_bytes());
        pickle.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n((X");
        pickle.extend_from_slice(&(7u32).to_le_bytes());
        pickle.extend_from_slice(b"storagectorch\n");
        pickle.extend_from_slice(dtype.as_bytes());
        pickle.extend_from_slice(b"\nX");
        pickle.extend_from_slice(&(storage.len() as u32).to_le_bytes());
        pickle.extend_from_slice(storage.as_bytes());
        pickle.extend_from_slice(b"X");
        pickle.extend_from_slice(&(device.len() as u32).to_le_bytes());
        pickle.extend_from_slice(device);
        let elements = match *storage {
            "0" => 2,
            "1" => 1,
            _ => 1,
        };
        pickle.extend_from_slice(&[b'K', elements, b't', b'Q', b'K', 0, b'(']);
        for &dimension in *shape {
            pickle.extend_from_slice(&[b'K', dimension]);
        }
        pickle.push(b't');
        pickle.push(b'(');
        for &stride in *strides {
            pickle.extend_from_slice(&[b'K', stride]);
        }
        pickle.extend_from_slice(&[b't', 0x89, b't', b'R', b's']);
    }
    pickle.push(b'.');
    pickle
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (!((crc & 1).wrapping_sub(1))));
        }
    }
    !crc
}

fn zip(files: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut output = Vec::new();
    let mut central = Vec::new();
    for (name, data) in files {
        let offset = output.len() as u32;
        let crc = crc32(data);
        output.extend_from_slice(b"PK\x03\x04\x14\0\0\0\0\0\0\0\0\0");
        output.extend_from_slice(&crc.to_le_bytes());
        output.extend_from_slice(&(data.len() as u32).to_le_bytes());
        output.extend_from_slice(&(data.len() as u32).to_le_bytes());
        output.extend_from_slice(&(name.len() as u16).to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes());
        output.extend_from_slice(name.as_bytes());
        output.extend_from_slice(data);
        central.extend_from_slice(b"PK\x01\x02\x14\0\x14\0\0\0\0\0\0\0\0\0");
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0; 6]);
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u32.to_le_bytes());
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let central_offset = output.len() as u32;
    output.extend_from_slice(&central);
    output.extend_from_slice(b"PK\x05\x06\0\0\0\0");
    output.extend_from_slice(&(files.len() as u16).to_le_bytes());
    output.extend_from_slice(&(files.len() as u16).to_le_bytes());
    output.extend_from_slice(&(central.len() as u32).to_le_bytes());
    output.extend_from_slice(&central_offset.to_le_bytes());
    output.extend_from_slice(&0u16.to_le_bytes());
    output
}

fn fixture(entries: &[StateEntry<'_>], storage: &[(&str, &[u8])]) -> Vec<u8> {
    let mut files = vec![("archive/data.pkl".into(), state_pickle(entries))];
    files.extend(
        storage
            .iter()
            .map(|(name, bytes)| (format!("archive/data/{name}"), bytes.to_vec())),
    );
    zip(&files)
}

fn valid_fixture() -> Vec<u8> {
    fixture(
        &[
            ("weight", "0", "FloatStorage", b"cpu", &[1, 2], &[2, 1]),
            ("bias", "1", "FloatStorage", b"cpu", &[1], &[1]),
        ],
        &[
            ("0", &[0, 0, 0, 0x40, 0, 0, 0x40, 0x40]),
            ("1", &[0, 0, 0x80, 0x3f]),
        ],
    )
}

fn linear() -> Linear {
    Linear::new(&mut Graph::new(), 2, 1, true, 7).unwrap()
}

fn f32(values: &[f32]) -> TensorData {
    TensorData::from_le_bytes(
        [2, 2],
        DType::F32,
        &values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn execute(module: &Linear, input: TensorData) -> TensorData {
    let mut graph = Graph::new();
    let input_node = graph.input("input", input.shape().clone());
    let output = module.forward(&mut graph, input_node).unwrap();
    let mut bindings = module.input_bindings(&graph).unwrap();
    bindings.insert("input".into(), input);
    CpuBackend.execute(&graph, output, &bindings).unwrap()
}

#[test]
fn local_torch_state_strictly_loads_fresh_linear_and_runs_on_cpu() {
    let directory = directory();
    let path = directory.join("linear.pt");
    let bytes = valid_fixture();
    fs::write(&path, &bytes).unwrap();
    let expected = load_torch_state_dict(&bytes).unwrap();
    assert_eq!(load_torch_state_file(&path).unwrap(), expected);

    let first = linear();
    let second = linear();
    let report =
        load_torch_state_file_strict_with_limits(&first, &path, TorchStateReadLimits::default())
            .unwrap();
    assert_eq!(report.loaded_keys, ["bias", "weight"]);
    assert_eq!(first.state_dict().unwrap().into_tensors(), expected);
    assert_eq!(
        execute(&first, f32(&[1., 2., 3., 4.])).to_vec_f64(),
        [9., 19.]
    );
    load_torch_state_file_strict_with_limits(&second, &path, TorchStateReadLimits::default())
        .unwrap();
    assert_eq!(first.state_dict().unwrap(), second.state_dict().unwrap());
    assert_ne!(first.weight.id(), second.weight.id());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn local_torch_file_and_strict_schema_fail_closed() {
    let directory = directory();
    let valid = directory.join("valid.pt");
    fs::write(&valid, valid_fixture()).unwrap();
    let target = linear();
    let before = target.state_dict().unwrap();

    let cases = [
        (
            "missing",
            fixture(
                &[("weight", "0", "FloatStorage", b"cpu", &[1, 2], &[2, 1])],
                &[("0", &[0; 8])],
            ),
        ),
        (
            "extra",
            fixture(
                &[
                    ("weight", "0", "FloatStorage", b"cpu", &[1, 2], &[2, 1]),
                    ("bias", "1", "FloatStorage", b"cpu", &[1], &[1]),
                    ("extra", "2", "FloatStorage", b"cpu", &[1], &[1]),
                ],
                &[("0", &[0; 8]), ("1", &[0; 4]), ("2", &[0; 4])],
            ),
        ),
        (
            "shape",
            fixture(
                &[
                    ("weight", "0", "FloatStorage", b"cpu", &[2], &[1]),
                    ("bias", "1", "FloatStorage", b"cpu", &[1], &[1]),
                ],
                &[("0", &[0; 8]), ("1", &[0; 4])],
            ),
        ),
        (
            "dtype",
            fixture(
                &[
                    ("weight", "0", "IntStorage", b"cpu", &[1, 2], &[2, 1]),
                    ("bias", "1", "FloatStorage", b"cpu", &[1], &[1]),
                ],
                &[("0", &[0; 8]), ("1", &[0; 4])],
            ),
        ),
    ];
    for (name, bytes) in cases {
        let path = directory.join(format!("{name}.pt"));
        fs::write(&path, bytes).unwrap();
        assert!(
            load_torch_state_file_strict_with_limits(
                &target,
                &path,
                TorchStateReadLimits::default()
            )
            .is_err()
        );
        assert_eq!(target.state_dict().unwrap(), before, "{name}");
    }

    let limits = TorchStateReadLimits {
        max_file_bytes: 1,
        ..TorchStateReadLimits::default()
    };
    assert!(matches!(
        load_torch_state_file_strict_with_limits(&target, &valid, limits),
        Err(TorchStateFileError::Limit { .. })
    ));
    assert_eq!(target.state_dict().unwrap(), before);
    fs::write(directory.join("truncated.pt"), b"PK").unwrap();
    assert!(matches!(
        load_torch_state_file(directory.join("truncated.pt")),
        Err(TorchStateFileError::Format(_))
    ));
    let hostile = directory.join("hostile.pt");
    fs::write(
        &hostile,
        zip(&[
            (
                "archive/data.pkl".into(),
                b"\x80\x02cos\nsystem\n.".to_vec(),
            ),
            ("archive/data/0".into(), vec![]),
        ]),
    )
    .unwrap();
    assert!(matches!(
        load_torch_state_file(&hostile),
        Err(TorchStateFileError::Format(_))
    ));
    for (name, bytes) in [
        (
            "unknown-storage",
            fixture(
                &[("weight", "0", "UnknownStorage", b"cpu", &[1, 2], &[2, 1])],
                &[("0", &[0; 8])],
            ),
        ),
        (
            "device",
            fixture(
                &[("weight", "0", "FloatStorage", b"cuda", &[1, 2], &[2, 1])],
                &[("0", &[0; 8])],
            ),
        ),
    ] {
        let path = directory.join(format!("{name}.pt"));
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            load_torch_state_file(&path),
            Err(TorchStateFileError::Format(_))
        ));
    }
    let entry_limit = TorchStateReadLimits {
        max_archive_entries: 1,
        ..TorchStateReadLimits::default()
    };
    assert!(load_torch_state_dict_with_limits(&valid_fixture(), entry_limit).is_err());
    let byte_limit = TorchStateReadLimits {
        max_tensor_bytes: 1,
        ..TorchStateReadLimits::default()
    };
    assert!(load_torch_state_dict_with_limits(&valid_fixture(), byte_limit).is_err());
    let element_limit = TorchStateReadLimits {
        max_tensor_elements: 1,
        ..TorchStateReadLimits::default()
    };
    assert!(load_torch_state_dict_with_limits(&valid_fixture(), element_limit).is_err());
    assert_eq!(target.state_dict().unwrap(), before);
    assert!(
        ModuleStateDict::from(load_torch_state_file(&valid).unwrap())
            .tensors()
            .contains_key("weight")
    );
    fs::remove_dir_all(directory).unwrap();
}
