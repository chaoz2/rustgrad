//! Portable dense bytes and the safetensors state-dictionary interchange format.
//!
//! This module deliberately copies bytes into typed `Storage`; it does not use
//! unaligned or lifetime-sensitive zero-copy casts.  The wire representation is
//! canonical little-endian regardless of the host architecture.

use crate::{DType, Error, Result, Shape, Storage, TensorData};
use serde::de::{self, Deserialize, Deserializer, MapAccess, Visitor};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fmt, fs,
    io::{self, Read, Write},
    path::Path,
};

pub type StateDict = BTreeMap<String, TensorData>;
/// Safetensors' reserved `__metadata__` object, whose values are strings.
pub type Metadata = BTreeMap<String, String>;

/// Resource limits for a local safetensors file read.
///
/// The byte cap is checked from filesystem metadata before allocating and is
/// checked again while reading, so a file that grows between those steps does
/// not reach the parser unbounded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SafetensorsReadLimits {
    pub max_file_bytes: usize,
}

impl Default for SafetensorsReadLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 1 << 30,
        }
    }
}

/// A local safetensors file failure, distinct from the validated format error.
#[derive(Debug)]
pub enum SafetensorsFileError {
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
    },
    Limit {
        actual: u64,
        maximum: usize,
    },
    Format(Error),
}

impl fmt::Display for SafetensorsFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, kind } => {
                write!(f, "safetensors file {operation} failed: {kind:?}")
            }
            Self::Limit { actual, maximum } => write!(
                f,
                "safetensors file has {actual} bytes, exceeding byte limit {maximum}"
            ),
            Self::Format(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for SafetensorsFileError {}

fn ser(reason: impl Into<String>) -> Error {
    Error::Serialization {
        reason: reason.into(),
    }
}

impl TensorData {
    /// Returns the canonical safetensors-compatible little-endian bytes.
    pub fn to_le_bytes(&self) -> Result<Vec<u8>> {
        let byte_len = self
            .len()
            .checked_mul(self.dtype().itemsize())
            .ok_or_else(|| ser("byte length overflows usize"))?;
        let mut out = Vec::new();
        out.try_reserve_exact(byte_len)
            .map_err(|_| ser("byte allocation failed"))?;
        macro_rules! push {
            ($values:expr) => {
                for value in $values {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            };
        }
        match self.storage() {
            Storage::Bool(v) => out.extend(v.iter().map(|&x| u8::from(x))),
            Storage::I8(v) => out.extend(v.iter().map(|&x| x as u8)),
            Storage::U8(v) => out.extend_from_slice(v),
            Storage::Float8(v) => out.extend_from_slice(v.as_raw()),
            Storage::I16(v) => push!(v),
            Storage::U16(v) => push!(v),
            Storage::I32(v) => push!(v),
            Storage::U32(v) => push!(v),
            Storage::I64(v) => push!(v),
            Storage::U64(v) => push!(v),
            Storage::F16(v) => push!(v),
            Storage::BF16(v) => push!(v),
            Storage::F32(v) => {
                for value in v {
                    out.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
            Storage::F64(v) => {
                for value in v {
                    out.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
        }
        Ok(out)
    }

    /// Decodes canonical little-endian dense bytes, preserving float bit patterns.
    pub fn from_le_bytes(shape: impl Into<Shape>, dtype: DType, bytes: &[u8]) -> Result<Self> {
        let shape = shape.into();
        let count = shape.numel()?;
        let expected = count
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| ser("byte length overflows usize"))?;
        if bytes.len() != expected {
            return Err(ser(format!(
                "{dtype:?} {shape} needs {expected} bytes, got {}",
                bytes.len()
            )));
        }
        macro_rules! chunks {
            ($type:ty, $from:ident) => {{
                let mut values = Vec::with_capacity(count);
                for chunk in bytes.chunks_exact(std::mem::size_of::<$type>()) {
                    values.push(<$type>::$from(chunk.try_into().expect("exact chunk")));
                }
                values
            }};
        }
        let storage = match dtype {
            DType::Bool => {
                let mut values = Vec::with_capacity(count);
                for &byte in bytes {
                    values.push(match byte {
                        0 => false,
                        1 => true,
                        _ => return Err(ser("bool bytes must be 0 or 1")),
                    });
                }
                Storage::Bool(values)
            }
            DType::I8 => Storage::I8(bytes.iter().map(|&x| x as i8).collect()),
            DType::U8 => Storage::U8(bytes.to_vec()),
            dtype @ (DType::F8E4M3 | DType::F8E5M2 | DType::F8E4M3FNUZ | DType::F8E5M2FNUZ) => {
                Storage::Float8(crate::Float8Storage::from_raw(
                    dtype.float8_format().expect("float8 dtype"),
                    bytes.to_vec(),
                ))
            }
            DType::I16 => Storage::I16(chunks!(i16, from_le_bytes)),
            DType::U16 => Storage::U16(chunks!(u16, from_le_bytes)),
            DType::I32 => Storage::I32(chunks!(i32, from_le_bytes)),
            DType::U32 => Storage::U32(chunks!(u32, from_le_bytes)),
            DType::I64 => Storage::I64(chunks!(i64, from_le_bytes)),
            DType::U64 => Storage::U64(chunks!(u64, from_le_bytes)),
            DType::F16 => Storage::F16(chunks!(u16, from_le_bytes)),
            DType::BF16 => Storage::BF16(chunks!(u16, from_le_bytes)),
            DType::F32 => Storage::F32(
                chunks!(u32, from_le_bytes)
                    .into_iter()
                    .map(f32::from_bits)
                    .collect(),
            ),
            DType::F64 => Storage::F64(
                chunks!(u64, from_le_bytes)
                    .into_iter()
                    .map(f64::from_bits)
                    .collect(),
            ),
        };
        Self::from_storage(shape, storage)
    }
}

#[derive(Debug)]
struct Entry {
    dtype: DType,
    shape: Vec<usize>,
    offsets: [usize; 2],
}

#[derive(Debug)]
struct Header {
    metadata: Metadata,
    tensors: BTreeMap<String, Entry>,
}

impl<'de> Deserialize<'de> for Header {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct HeaderVisitor;
        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = Header;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a safetensors header object")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Header, A::Error> {
                let mut metadata = Metadata::new();
                let mut tensors = BTreeMap::new();
                while let Some((name, value)) = map.next_entry::<String, Value>()? {
                    if name == "__metadata__" {
                        let object = value
                            .as_object()
                            .ok_or_else(|| de::Error::custom("__metadata__ must be an object"))?;
                        for (k, v) in object {
                            metadata.insert(
                                k.clone(),
                                v.as_str()
                                    .ok_or_else(|| {
                                        de::Error::custom("metadata values must be strings")
                                    })?
                                    .to_owned(),
                            );
                        }
                    } else {
                        if name.is_empty() {
                            return Err(de::Error::custom("tensor name must not be empty"));
                        }
                        if tensors.contains_key(&name) {
                            return Err(de::Error::custom("duplicate tensor name"));
                        }
                        let object = value
                            .as_object()
                            .ok_or_else(|| de::Error::custom("tensor entry must be an object"))?;
                        if object.len() != 3
                            || !object.contains_key("dtype")
                            || !object.contains_key("shape")
                            || !object.contains_key("data_offsets")
                        {
                            return Err(de::Error::custom(
                                "tensor entry must contain only dtype, shape, and data_offsets",
                            ));
                        }
                        let dtype = dtype_from_tag(
                            object["dtype"]
                                .as_str()
                                .ok_or_else(|| de::Error::custom("dtype must be a string"))?,
                        )
                        .map_err(de::Error::custom)?;
                        let shape = object["shape"]
                            .as_array()
                            .ok_or_else(|| de::Error::custom("shape must be an array"))?
                            .iter()
                            .map(|v| {
                                v.as_u64()
                                    .and_then(|x| usize::try_from(x).ok())
                                    .ok_or_else(|| {
                                        de::Error::custom("shape dimensions must be usize integers")
                                    })
                            })
                            .collect::<std::result::Result<Vec<_>, _>>()?;
                        let offsets = object["data_offsets"]
                            .as_array()
                            .ok_or_else(|| de::Error::custom("data_offsets must be an array"))?;
                        if offsets.len() != 2 {
                            return Err(de::Error::custom("data_offsets must have two values"));
                        }
                        let offset = |v: &Value| {
                            v.as_u64()
                                .and_then(|x| usize::try_from(x).ok())
                                .ok_or_else(|| de::Error::custom("offset must be a usize integer"))
                        };
                        tensors.insert(
                            name,
                            Entry {
                                dtype,
                                shape,
                                offsets: [offset(&offsets[0])?, offset(&offsets[1])?],
                            },
                        );
                    }
                }
                Ok(Header { metadata, tensors })
            }
        }
        deserializer.deserialize_map(HeaderVisitor)
    }
}

fn dtype_from_tag(tag: &str) -> std::result::Result<DType, String> {
    Ok(match tag {
        "BOOL" => DType::Bool,
        "I8" => DType::I8,
        "U8" => DType::U8,
        "I16" => DType::I16,
        "U16" => DType::U16,
        "I32" => DType::I32,
        "U32" => DType::U32,
        "I64" => DType::I64,
        "U64" => DType::U64,
        "F16" => DType::F16,
        "F8_E4M3FN" => DType::F8E4M3,
        "F8_E5M2" => DType::F8E5M2,
        "BF16" => DType::BF16,
        "F32" => DType::F32,
        "F64" => DType::F64,
        _ => return Err(format!("unsupported safetensors dtype {tag:?}")),
    })
}
fn dtype_tag(dtype: DType) -> std::result::Result<&'static str, String> {
    Ok(match dtype {
        DType::Bool => "BOOL",
        DType::I8 => "I8",
        DType::U8 => "U8",
        DType::I16 => "I16",
        DType::U16 => "U16",
        DType::I32 => "I32",
        DType::U32 => "U32",
        DType::I64 => "I64",
        DType::U64 => "U64",
        DType::F16 => "F16",
        DType::F8E4M3 => "F8_E4M3FN",
        DType::F8E5M2 => "F8_E5M2",
        DType::F8E4M3FNUZ | DType::F8E5M2FNUZ => {
            return Err("safetensors has no accepted portable FNUZ float8 tag".into());
        }
        DType::BF16 => "BF16",
        DType::F32 => "F32",
        DType::F64 => "F64",
    })
}

/// Loads an ordered state dictionary and string metadata from an in-memory file.
pub fn load_safetensors(bytes: &[u8]) -> Result<(StateDict, Metadata)> {
    if bytes.len() < 8 {
        return Err(ser("file is shorter than the 8-byte header length"));
    }
    let header_len = usize::try_from(u64::from_le_bytes(
        bytes[..8].try_into().expect("eight bytes"),
    ))
    .map_err(|_| ser("header length does not fit usize"))?;
    let data_start = 8usize
        .checked_add(header_len)
        .ok_or_else(|| ser("header length overflows usize"))?;
    if data_start > bytes.len() {
        return Err(ser("truncated header"));
    }
    let header: Header = serde_json::from_slice(&bytes[8..data_start])
        .map_err(|e| ser(format!("invalid header JSON: {e}")))?;
    let data = &bytes[data_start..];
    let mut entries: Vec<_> = header.tensors.into_iter().collect();
    entries.sort_by_key(|(_, entry)| entry.offsets[0]);
    let mut expected_offset = 0usize;
    let mut result = StateDict::new();
    for (name, entry) in entries {
        let [start, end] = entry.offsets;
        if start != expected_offset || end < start {
            return Err(ser(format!(
                "tensor {name:?} has non-contiguous or overlapping offsets"
            )));
        }
        let count = Shape::new(entry.shape.clone()).numel()?;
        let len = count
            .checked_mul(entry.dtype.itemsize())
            .ok_or_else(|| ser(format!("tensor {name:?} byte length overflows usize")))?;
        if end.checked_sub(start) != Some(len) || end > data.len() {
            return Err(ser(format!(
                "tensor {name:?} data offsets do not match its shape/dtype or are truncated"
            )));
        }
        result.insert(
            name,
            TensorData::from_le_bytes(entry.shape, entry.dtype, &data[start..end])?,
        );
        expected_offset = end;
    }
    if expected_offset != data.len() {
        return Err(ser("data section contains unreferenced bytes"));
    }
    Ok((result, header.metadata))
}

/// Serializes a deterministic, name-sorted state dictionary.
pub fn save_safetensors(tensors: &StateDict, metadata: &Metadata) -> Result<Vec<u8>> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        header.insert(
            "__metadata__".into(),
            serde_json::to_value(metadata).map_err(|e| ser(e.to_string()))?,
        );
    }
    let mut payload = Vec::new();
    for (name, tensor) in tensors {
        if name.is_empty() || name == "__metadata__" {
            return Err(ser("tensor name must not be empty or __metadata__"));
        }
        let start = payload.len();
        let raw = tensor.to_le_bytes()?;
        payload
            .try_reserve(raw.len())
            .map_err(|_| ser("payload allocation failed"))?;
        payload.extend_from_slice(&raw);
        let end = payload.len();
        header.insert(name.clone(), serde_json::json!({"dtype": dtype_tag(tensor.dtype()).map_err(ser)?, "shape": tensor.shape().dims(), "data_offsets": [start, end]}));
    }
    let mut header = serde_json::to_vec(&header).map_err(|e| ser(e.to_string()))?;
    let padding = (8 - header.len() % 8) % 8;
    header.extend(std::iter::repeat_n(b' ', padding));
    let header_len = u64::try_from(header.len()).map_err(|_| ser("header is too large"))?;
    let total = 8usize
        .checked_add(header.len())
        .and_then(|n| n.checked_add(payload.len()))
        .ok_or_else(|| ser("output size overflows usize"))?;
    let mut out = Vec::new();
    out.try_reserve(total)
        .map_err(|_| ser("output allocation failed"))?;
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Loads a local safetensors file under [`SafetensorsReadLimits::default`].
///
/// This compatibility convenience maps filesystem and limit failures into the
/// crate-wide serialization error. Prefer [`load_safetensors_file_with_limits`]
/// when callers need a typed local-file boundary.
pub fn load_safetensors_file(path: impl AsRef<Path>) -> Result<(StateDict, Metadata)> {
    match load_safetensors_file_with_limits(path, SafetensorsReadLimits::default()) {
        Ok(value) => Ok(value),
        Err(SafetensorsFileError::Format(error)) => Err(error),
        Err(error) => Err(ser(error.to_string())),
    }
}

/// Loads a local safetensors file under an explicit byte limit.
///
/// The entire validated file is copied into bounded owned bytes before the
/// canonical parser runs. This API performs no mapping, lazy tensor ownership,
/// device transfer, or code execution.
pub fn load_safetensors_file_with_limits(
    path: impl AsRef<Path>,
    limits: SafetensorsReadLimits,
) -> std::result::Result<(StateDict, Metadata), SafetensorsFileError> {
    let path = path.as_ref();
    let metadata = fs::metadata(path).map_err(|error| SafetensorsFileError::Io {
        operation: "inspect",
        kind: error.kind(),
    })?;
    if metadata.len() > u64::try_from(limits.max_file_bytes).unwrap_or(u64::MAX) {
        return Err(SafetensorsFileError::Limit {
            actual: metadata.len(),
            maximum: limits.max_file_bytes,
        });
    }
    let file = fs::File::open(path).map_err(|error| SafetensorsFileError::Io {
        operation: "open",
        kind: error.kind(),
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(limits.max_file_bytes.min(64 << 10))
        .map_err(|_| SafetensorsFileError::Format(ser("file buffer allocation failed")))?;
    file.take(
        u64::try_from(limits.max_file_bytes)
            .unwrap_or(u64::MAX)
            .saturating_add(1),
    )
    .read_to_end(&mut bytes)
    .map_err(|error| SafetensorsFileError::Io {
        operation: "read",
        kind: error.kind(),
    })?;
    if bytes.len() > limits.max_file_bytes {
        return Err(SafetensorsFileError::Limit {
            actual: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            maximum: limits.max_file_bytes,
        });
    }
    load_safetensors(&bytes).map_err(SafetensorsFileError::Format)
}
/// Atomically replaces `path` after constructing the whole file in memory.
///
/// The staged file is created exclusively beside the target, written and
/// synced before replacement, then cleaned after every failed write or rename.
/// An existing target is never opened for writing before the final rename.
pub fn save_safetensors_file(
    path: impl AsRef<Path>,
    tensors: &StateDict,
    metadata: &Metadata,
) -> Result<()> {
    let path = path.as_ref();
    let bytes = save_safetensors(tensors, metadata)?;
    let file_name = path
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| ser("path must have a UTF-8 filename"))?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut temp = None;
    for attempt in 0..128u16 {
        let candidate = parent.join(format!(
            ".{file_name}.rustgrad-{}-{attempt}.tmp",
            std::process::id()
        ));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(mut file) => {
                let result = (|| {
                    file.write_all(&bytes).map_err(|error| ser(error.to_string()))?;
                    file.sync_all().map_err(|error| ser(error.to_string()))
                })();
                if let Err(error) = result {
                    drop(file);
                    let _ = fs::remove_file(&candidate);
                    return Err(error);
                }
                temp = Some(candidate);
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(ser(error.to_string())),
        }
    }
    let temp = temp.ok_or_else(|| ser("could not create unique safetensors staging file"))?;
    fs::rename(&temp, path).map_err(|e| {
        let _ = fs::remove_file(&temp);
        ser(e.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn raw(shape: impl Into<Shape>, storage: Storage) -> TensorData {
        TensorData::from_storage(shape, storage).unwrap()
    }

    fn file_directory() -> std::path::PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "rustgrad-safetensors-file-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        directory
    }
    #[test]
    fn portable_bytes_round_trip_all_dtypes() {
        let values = vec![
            raw([2], Storage::Bool(vec![false, true])),
            raw([2], Storage::I8(vec![-128, 127])),
            raw([2], Storage::U8(vec![0, 255])),
            raw([2], Storage::I16(vec![i16::MIN, i16::MAX])),
            raw([2], Storage::U16(vec![0, u16::MAX])),
            raw([2], Storage::I32(vec![i32::MIN, i32::MAX])),
            raw([2], Storage::U32(vec![0, u32::MAX])),
            raw([2], Storage::I64(vec![i64::MIN, i64::MAX])),
            raw([2], Storage::U64(vec![0, u64::MAX])),
            raw([2], Storage::F16(vec![0x7e55, 0x8000])),
            raw([2], Storage::BF16(vec![0x7fc1, 0x8000])),
            raw([2], Storage::F32(vec![f32::from_bits(0x7fc0_1234), -0.0])),
            raw(
                [2],
                Storage::F64(vec![f64::from_bits(0x7ff8_0000_0000_1234), -0.0]),
            ),
        ];
        for tensor in values {
            let loaded = TensorData::from_le_bytes(
                tensor.shape().clone(),
                tensor.dtype(),
                &tensor.to_le_bytes().unwrap(),
            )
            .unwrap();
            assert_eq!(loaded.to_le_bytes().unwrap(), tensor.to_le_bytes().unwrap());
        }
        assert_eq!(
            raw([2], Storage::I16(vec![1, -2])).to_le_bytes().unwrap(),
            vec![1, 0, 254, 255]
        );
        assert_eq!(
            TensorData::from_le_bytes(
                [],
                DType::F32,
                &raw([], Storage::F32(vec![f32::from_bits(0x8000_0000)]))
                    .to_le_bytes()
                    .unwrap()
            )
            .unwrap()
            .storage(),
            &Storage::F32(vec![-0.0])
        );
        assert!(TensorData::from_le_bytes([1], DType::Bool, &[2]).is_err());
    }
    #[test]
    fn safetensors_round_trip_and_determinism() {
        let mut a = StateDict::new();
        a.insert(
            "z".into(),
            raw([], Storage::F32(vec![f32::from_bits(0x8000_0000)])),
        );
        a.insert("a".into(), raw([2, 0], Storage::U16(vec![])));
        let meta = Metadata::from([("source".into(), "test".into())]);
        let bytes = save_safetensors(&a, &meta).unwrap();
        let (loaded, got_meta) = load_safetensors(&bytes).unwrap();
        assert_eq!(loaded, a);
        assert_eq!(got_meta, meta);
        let mut reversed = StateDict::new();
        reversed.insert("a".into(), a["a"].clone());
        reversed.insert("z".into(), a["z"].clone());
        assert_eq!(bytes, save_safetensors(&reversed, &meta).unwrap());
    }
    #[test]
    fn float8_standard_tags_round_trip_and_fnuz_is_rejected() {
        for (dtype, tag) in [(DType::F8E4M3, "F8_E4M3FN"), (DType::F8E5M2, "F8_E5M2")] {
            let mut state = StateDict::new();
            state.insert(
                "x".into(),
                raw(
                    [2],
                    Storage::Float8(crate::Float8Storage::from_raw(
                        dtype.float8_format().unwrap(),
                        vec![0x80, 0xff],
                    )),
                ),
            );
            let bytes = save_safetensors(&state, &Metadata::new()).unwrap();
            assert!(
                bytes
                    .windows(tag.len())
                    .any(|window| window == tag.as_bytes())
            );
            assert_eq!(load_safetensors(&bytes).unwrap().0, state);
        }
        for dtype in [DType::F8E4M3FNUZ, DType::F8E5M2FNUZ] {
            let mut state = StateDict::new();
            state.insert(
                "x".into(),
                raw(
                    [1],
                    Storage::Float8(crate::Float8Storage::from_raw(
                        dtype.float8_format().unwrap(),
                        vec![0x80],
                    )),
                ),
            );
            assert!(save_safetensors(&state, &Metadata::new()).is_err());
        }
    }
    #[test]
    fn rejects_malformed_layouts() {
        assert!(load_safetensors(&[0; 7]).is_err());
        let bad = br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}"#;
        let mut file = (bad.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(bad);
        file.extend_from_slice(&[0; 4]);
        assert!(load_safetensors(&file).is_err());
        let bad_dtype = br#"{"x":{"dtype":"NOPE","shape":[1],"data_offsets":[0,4]}}"#;
        let mut file = (bad_dtype.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(bad_dtype);
        file.extend_from_slice(&[0; 4]);
        assert!(load_safetensors(&file).is_err());

        let overlapping = br#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#;
        let mut file = (overlapping.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(overlapping);
        file.extend_from_slice(&[0; 3]);
        assert!(load_safetensors(&file).is_err());
        let overflow =
            br#"{"x":{"dtype":"U64","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#;
        let mut file = (overflow.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(overflow);
        assert!(load_safetensors(&file).is_err());
        let duplicate = br#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut file = (duplicate.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(duplicate);
        file.push(0);
        assert!(load_safetensors(&file).is_err());
    }
    #[test]
    fn independently_constructed_minimal_fixture_and_file_api() {
        // Safetensors format: 8-byte LE header size, JSON header, then payload.
        let header = br#"{"x":{"dtype":"I16","shape":[2],"data_offsets":[0,4]}}"#;
        let mut fixture = (header.len() as u64).to_le_bytes().to_vec();
        fixture.extend_from_slice(header);
        fixture.extend_from_slice(&[1, 0, 254, 255]);
        let (loaded, _) = load_safetensors(&fixture).unwrap();
        assert_eq!(loaded["x"].to_le_bytes().unwrap(), vec![1, 0, 254, 255]);
        let directory = std::env::temp_dir().join(format!("rustgrad-safe-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.safetensors");
        save_safetensors_file(&path, &loaded, &Metadata::new()).unwrap();
        assert_eq!(load_safetensors_file(&path).unwrap().0, loaded);
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn bounded_file_api_preserves_raw_bits_and_rejects_before_parsing() {
        // Independently encoded: eight-byte LE header length, JSON, then F32
        // signed-zero and NaN payload bytes.
        let header = br#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut fixture = (header.len() as u64).to_le_bytes().to_vec();
        fixture.extend_from_slice(header);
        fixture.extend_from_slice(&[0, 0, 0, 0x80, 0x34, 0x12, 0xc0, 0x7f]);
        let directory =
            std::env::temp_dir().join(format!("rustgrad-safe-bounded-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.safetensors");
        fs::write(&path, &fixture).unwrap();
        let (loaded, metadata) = load_safetensors_file_with_limits(
            &path,
            SafetensorsReadLimits {
                max_file_bytes: fixture.len(),
            },
        )
        .unwrap();
        assert!(metadata.is_empty());
        assert_eq!(
            loaded["x"].to_le_bytes().unwrap(),
            fixture[8 + header.len()..]
        );

        assert!(matches!(
            load_safetensors_file_with_limits(
                &path,
                SafetensorsReadLimits {
                    max_file_bytes: fixture.len() - 1,
                }
            ),
            Err(SafetensorsFileError::Limit { .. })
        ));
        fs::write(&path, b"short").unwrap();
        assert!(matches!(
            load_safetensors_file_with_limits(&path, SafetensorsReadLimits::default()),
            Err(SafetensorsFileError::Format(_))
        ));
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn staged_file_save_preserves_targets_and_cleans_its_own_failed_staging() {
        let directory = file_directory();
        let target = directory.join("target.safetensors");
        let occupied = directory.join(format!(
            ".target.safetensors.rustgrad-{}-0.tmp",
            std::process::id()
        ));
        fs::write(&occupied, b"another writer").unwrap();
        fs::create_dir(&target).unwrap();
        let state = StateDict::from(BTreeMap::from([(
            "x".into(),
            TensorData::from_le_bytes([1], DType::U8, &[7]).unwrap(),
        )]));

        assert!(save_safetensors_file(&target, &state, &Metadata::new()).is_err());
        assert!(target.is_dir());
        assert_eq!(fs::read(&occupied).unwrap(), b"another writer");
        assert!(
            !fs::read_dir(&directory)
                .unwrap()
                .filter_map(Result::ok)
                .any(|entry| {
                    entry
                        .file_name()
                        .to_string_lossy()
                        .ends_with("-1.tmp")
                })
        );

        fs::remove_dir(&target).unwrap();
        save_safetensors_file(&target, &state, &Metadata::new()).unwrap();
        assert_eq!(load_safetensors_file(&target).unwrap().0, state);
        fs::remove_file(target).unwrap();
        fs::remove_file(occupied).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
