//! Portable dense bytes and the safetensors state-dictionary interchange format.
//!
//! This module deliberately copies bytes into typed `Storage`; it does not use
//! unaligned or lifetime-sensitive zero-copy casts.  The wire representation is
//! canonical little-endian regardless of the host architecture.

use crate::{DType, Error, Result, Shape, Storage, TensorData};
use serde::de::{self, Deserialize, Deserializer, MapAccess, Visitor};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

pub type StateDict = BTreeMap<String, TensorData>;
/// Safetensors' reserved `__metadata__` object, whose values are strings.
pub type Metadata = BTreeMap<String, String>;

/// Borrowed safetensors prefix information without tensor-descriptor validation.
///
/// `header` is the raw JSON value from the file prefix. It intentionally does
/// not require a safetensors tensor-map shape: use [`load_safetensors`] when
/// the tensor descriptors and payload must be validated.
#[derive(Debug, PartialEq)]
pub struct SafetensorsMetadata<'a> {
    /// The original complete safetensors byte slice.
    pub source: &'a [u8],
    /// Byte offset at which the data section begins.
    pub data_start: usize,
    /// The unvalidated JSON header value.
    pub header: Value,
}

fn ser(reason: impl Into<String>) -> Error {
    Error::Serialization {
        reason: reason.into(),
    }
}

/// Validates the shared 8-byte safetensors prefix and borrows its JSON bytes.
fn safetensors_header_prefix(bytes: &[u8]) -> Result<(&[u8], usize)> {
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
    Ok((&bytes[8..data_start], data_start))
}

/// Inspects only the borrowed safetensors prefix and raw JSON header.
///
/// This matches tinygrad's metadata inspection boundary: it checks the
/// length-prefixed header, parses JSON, and deliberately leaves tensor
/// descriptor, dtype, offset, and payload validation to [`load_safetensors`].
pub fn inspect_safetensors_metadata(bytes: &[u8]) -> Result<SafetensorsMetadata<'_>> {
    let (header_bytes, data_start) = safetensors_header_prefix(bytes)?;
    let header = serde_json::from_slice(header_bytes)
        .map_err(|e| ser(format!("invalid header JSON: {e}")))?;
    Ok(SafetensorsMetadata {
        source: bytes,
        data_start,
        header,
    })
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

struct StateHeader {
    tensors: BTreeMap<String, Entry>,
}

fn parse_header_map<'de, A: MapAccess<'de>>(
    mut map: A,
    strict_metadata: bool,
) -> std::result::Result<Header, A::Error> {
    let mut metadata = Metadata::new();
    let mut tensors = BTreeMap::new();
    while let Some((name, value)) = map.next_entry::<String, Value>()? {
        if name == "__metadata__" {
            if strict_metadata {
                let object = value
                    .as_object()
                    .ok_or_else(|| de::Error::custom("__metadata__ must be an object"))?;
                for (k, v) in object {
                    metadata.insert(
                        k.clone(),
                        v.as_str()
                            .ok_or_else(|| de::Error::custom("metadata values must be strings"))?
                            .to_owned(),
                    );
                }
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
                        .ok_or_else(|| de::Error::custom("shape dimensions must be usize integers"))
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
                map: A,
            ) -> std::result::Result<Header, A::Error> {
                parse_header_map(map, true)
            }
        }
        deserializer.deserialize_map(HeaderVisitor)
    }
}

impl<'de> Deserialize<'de> for StateHeader {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct StateHeaderVisitor;
        impl<'de> Visitor<'de> for StateHeaderVisitor {
            type Value = StateHeader;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a safetensors header object")
            }

            fn visit_map<A: MapAccess<'de>>(
                self,
                map: A,
            ) -> std::result::Result<StateHeader, A::Error> {
                Ok(StateHeader {
                    tensors: parse_header_map(map, false)?.tensors,
                })
            }
        }
        deserializer.deserialize_map(StateHeaderVisitor)
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
        "BF16" => DType::BF16,
        "F32" => DType::F32,
        "F64" => DType::F64,
        _ => return Err(format!("unsupported safetensors dtype {tag:?}")),
    })
}
fn dtype_tag(dtype: DType) -> &'static str {
    match dtype {
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
        DType::BF16 => "BF16",
        DType::F32 => "F32",
        DType::F64 => "F64",
    }
}

fn load_safetensors_state(
    bytes: &[u8],
    data_start: usize,
    tensors: BTreeMap<String, Entry>,
) -> Result<StateDict> {
    let data = &bytes[data_start..];
    let mut entries: Vec<_> = tensors.into_iter().collect();
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
    Ok(result)
}

/// Loads an ordered state dictionary and string metadata from an in-memory file.
pub fn load_safetensors(bytes: &[u8]) -> Result<(StateDict, Metadata)> {
    let (header_bytes, data_start) = safetensors_header_prefix(bytes)?;
    let header: Header = serde_json::from_slice(header_bytes)
        .map_err(|e| ser(format!("invalid header JSON: {e}")))?;
    let state = load_safetensors_state(bytes, data_start, header.tensors)?;
    Ok((state, header.metadata))
}

/// Loads safetensors tensor entries while deliberately ignoring `__metadata__`.
///
/// Tensor descriptors, offsets, payload layout, and dense bytes receive the
/// same validation as [`load_safetensors`]. Unlike that strict compatibility
/// loader, this state-only entry point accepts any JSON value for the reserved
/// metadata field.
pub fn load_safetensors_state_only(bytes: &[u8]) -> Result<StateDict> {
    let (header_bytes, data_start) = safetensors_header_prefix(bytes)?;
    let header: StateHeader = serde_json::from_slice(header_bytes)
        .map_err(|e| ser(format!("invalid header JSON: {e}")))?;
    load_safetensors_state(bytes, data_start, header.tensors)
}

fn checked_raw_metadata(metadata: Option<&Value>) -> Result<Option<Value>> {
    match metadata {
        None => Ok(None),
        Some(Value::Object(object)) if object.is_empty() => Ok(None),
        Some(Value::Object(_)) => Ok(metadata.cloned()),
        Some(_) => Err(ser("metadata must be a JSON object")),
    }
}

fn serialize_safetensors(tensors: &StateDict, metadata: Option<&Value>) -> Result<Vec<u8>> {
    let metadata = checked_raw_metadata(metadata)?;
    let mut header = serde_json::Map::new();
    if let Some(metadata) = metadata {
        header.insert("__metadata__".into(), metadata);
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
        header.insert(name.clone(), serde_json::json!({"dtype": dtype_tag(tensor.dtype()), "shape": tensor.shape().dims(), "data_offsets": [start, end]}));
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

/// Serializes a deterministic, name-sorted state dictionary and string metadata.
pub fn save_safetensors(tensors: &StateDict, metadata: &Metadata) -> Result<Vec<u8>> {
    let metadata = serde_json::to_value(metadata).map_err(|e| ser(e.to_string()))?;
    serialize_safetensors(tensors, Some(&metadata))
}

/// Serializes a state dictionary with optional raw JSON object metadata.
///
/// Empty metadata and `None` omit `__metadata__`. Non-object metadata is
/// rejected before tensor or payload serialization begins.
pub fn save_safetensors_with_json_metadata(
    tensors: &StateDict,
    metadata: Option<&Value>,
) -> Result<Vec<u8>> {
    serialize_safetensors(tensors, metadata)
}

pub fn load_safetensors_file(path: impl AsRef<Path>) -> Result<(StateDict, Metadata)> {
    load_safetensors(&fs::read(path).map_err(|e| ser(e.to_string()))?)
}

fn save_safetensors_file_bytes(path: impl AsRef<Path>, bytes: Vec<u8>) -> Result<()> {
    let path = path.as_ref();
    let file_name = path
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| ser("path must have a UTF-8 filename"))?;
    let temp = path.with_file_name(format!(".{file_name}.rustgrad-{}.tmp", std::process::id()));
    fs::write(&temp, bytes).map_err(|e| ser(e.to_string()))?;
    fs::rename(&temp, path).map_err(|e| {
        let _ = fs::remove_file(&temp);
        ser(e.to_string())
    })
}

/// Atomically replaces `path` after constructing the whole file in memory.
pub fn save_safetensors_file(
    path: impl AsRef<Path>,
    tensors: &StateDict,
    metadata: &Metadata,
) -> Result<()> {
    save_safetensors_file_bytes(path, save_safetensors(tensors, metadata)?)
}

/// Atomically saves a state dictionary with optional raw JSON object metadata.
pub fn save_safetensors_file_with_json_metadata(
    path: impl AsRef<Path>,
    tensors: &StateDict,
    metadata: Option<&Value>,
) -> Result<()> {
    save_safetensors_file_bytes(path, save_safetensors_with_json_metadata(tensors, metadata)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn raw(shape: impl Into<Shape>, storage: Storage) -> TensorData {
        TensorData::from_storage(shape, storage).unwrap()
    }

    #[test]
    fn metadata_inspection_borrows_raw_json_without_validating_tensors() {
        let header = br#"{"__metadata__":{"producer":"tinygrad"},"x":{"dtype":"NOT_A_DTYPE","shape":"unchecked","data_offsets":[9]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[7, 8, 9]);
        let original = bytes.clone();

        let metadata = inspect_safetensors_metadata(&bytes).unwrap();
        assert_eq!(metadata.source.as_ptr(), bytes.as_ptr());
        assert_eq!(metadata.source, original.as_slice());
        assert_eq!(metadata.data_start, 8 + header.len());
        assert_eq!(
            metadata.header,
            serde_json::json!({
                "__metadata__": {"producer": "tinygrad"},
                "x": {"dtype": "NOT_A_DTYPE", "shape": "unchecked", "data_offsets": [9]}
            })
        );
        assert_eq!(bytes, original);
        assert!(load_safetensors(&bytes).is_err());
    }

    #[test]
    fn metadata_inspection_accepts_arbitrary_json_header_shape() {
        let header = br#"[{"metadata":{"nested":true}},["tensor",0]]"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);

        let metadata = inspect_safetensors_metadata(&bytes).unwrap();
        assert_eq!(metadata.data_start, bytes.len());
        assert_eq!(
            metadata.header,
            serde_json::json!([{"metadata": {"nested": true}}, ["tensor", 0]])
        );
    }

    #[test]
    fn metadata_inspection_rejects_invalid_prefixes_and_json() {
        assert!(inspect_safetensors_metadata(&[0; 7]).is_err());

        let mut truncated = 4u64.to_le_bytes().to_vec();
        truncated.extend_from_slice(b"{}");
        assert!(inspect_safetensors_metadata(&truncated).is_err());

        assert!(inspect_safetensors_metadata(&u64::MAX.to_le_bytes()).is_err());

        let mut invalid_json = 1u64.to_le_bytes().to_vec();
        invalid_json.push(b'{');
        assert!(inspect_safetensors_metadata(&invalid_json).is_err());
    }

    #[test]
    fn state_only_load_ignores_raw_metadata_but_preserves_tensor_lanes() {
        let header = br#"{"__metadata__":{"nested":[1,true],"number":7},"x":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
        let payload = [0x00, 0x80, 0x55, 0x7e];
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&payload);
        let original = bytes.clone();

        let state = load_safetensors_state_only(&bytes).unwrap();
        assert_eq!(state["x"].to_le_bytes().unwrap(), payload);
        assert!(load_safetensors(&bytes).is_err());
        assert_eq!(bytes, original);
    }

    #[test]
    fn state_only_and_strict_loaders_agree_for_string_metadata() {
        let tensors = StateDict::from([("x".into(), raw([1], Storage::U8(vec![9])))]);
        let metadata = Metadata::from([("source".into(), "tinygrad".into())]);
        let bytes = save_safetensors(&tensors, &metadata).unwrap();

        assert_eq!(load_safetensors_state_only(&bytes).unwrap(), tensors);
        assert_eq!(load_safetensors(&bytes).unwrap(), (tensors, metadata));
    }

    #[test]
    fn raw_json_metadata_save_round_trips_state_and_preserves_bytes() {
        let tensors = StateDict::from([(
            "x".into(),
            raw([2], Storage::F16(vec![0x8000, 0x7e55])),
        )]);
        let metadata = serde_json::json!({
            "nested": {"flags": [true, null], "count": 7},
            "number": 1.5
        });
        let original_tensors = tensors.clone();
        let original_metadata = metadata.clone();

        let bytes = save_safetensors_with_json_metadata(&tensors, Some(&metadata)).unwrap();
        assert_eq!(
            inspect_safetensors_metadata(&bytes).unwrap().header["__metadata__"],
            metadata
        );
        assert_eq!(load_safetensors_state_only(&bytes).unwrap(), tensors);
        assert!(load_safetensors(&bytes).is_err());
        assert_eq!(bytes, save_safetensors_with_json_metadata(&tensors, Some(&metadata)).unwrap());
        assert_eq!(tensors, original_tensors);
        assert_eq!(metadata, original_metadata);

        let empty = serde_json::json!({});
        let absent = save_safetensors_with_json_metadata(&tensors, None).unwrap();
        assert_eq!(
            absent,
            save_safetensors_with_json_metadata(&tensors, Some(&empty)).unwrap()
        );
        assert!(inspect_safetensors_metadata(&absent)
            .unwrap()
            .header
            .get("__metadata__")
            .is_none());
    }

    #[test]
    fn raw_json_metadata_save_matches_legacy_strings_and_fails_before_tensors() {
        let tensors = StateDict::from([("x".into(), raw([1], Storage::U8(vec![9])))]);
        let metadata = Metadata::from([("source".into(), "tinygrad".into())]);
        let raw_metadata = serde_json::to_value(&metadata).unwrap();
        assert_eq!(
            save_safetensors(&tensors, &metadata).unwrap(),
            save_safetensors_with_json_metadata(&tensors, Some(&raw_metadata)).unwrap()
        );

        let invalid_tensors = StateDict::from([("".into(), raw([], Storage::U8(vec![1])))]);
        let non_object = serde_json::json!(false);
        assert!(matches!(
            save_safetensors_with_json_metadata(&invalid_tensors, Some(&non_object)),
            Err(Error::Serialization { reason }) if reason == "metadata must be a JSON object"
        ));
        assert!(save_safetensors_with_json_metadata(&invalid_tensors, None).is_err());
        assert_eq!(invalid_tensors[""].to_le_bytes().unwrap(), vec![1]);
    }

    #[test]
    fn raw_json_metadata_file_save_is_atomic_wrapper() {
        let tensors = StateDict::from([("x".into(), raw([1], Storage::U8(vec![4])))]);
        let metadata = serde_json::json!({"nested": [1, true]});
        let directory = std::env::temp_dir().join(format!("rustgrad-safe-json-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.safetensors");

        save_safetensors_file_with_json_metadata(&path, &tensors, Some(&metadata)).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(
            inspect_safetensors_metadata(&bytes).unwrap().header["__metadata__"],
            metadata
        );
        assert_eq!(load_safetensors_state_only(&bytes).unwrap(), tensors);
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn state_only_loader_rejects_non_object_headers_and_bad_tensor_layouts() {
        let top_level_array = br#"[]"#;
        let mut non_object = (top_level_array.len() as u64).to_le_bytes().to_vec();
        non_object.extend_from_slice(top_level_array);
        assert!(load_safetensors_state_only(&non_object).is_err());
        assert!(load_safetensors(&non_object).is_err());

        let bad_descriptor = br#"{"x":{"dtype":"NOPE","shape":[1],"data_offsets":[0,1]}}"#;
        let mut descriptor = (bad_descriptor.len() as u64).to_le_bytes().to_vec();
        descriptor.extend_from_slice(bad_descriptor);
        descriptor.push(0);
        assert!(load_safetensors_state_only(&descriptor).is_err());
        assert!(load_safetensors(&descriptor).is_err());

        let bad_offsets = br#"{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,1]}}"#;
        let mut offsets = (bad_offsets.len() as u64).to_le_bytes().to_vec();
        offsets.extend_from_slice(bad_offsets);
        offsets.push(0);
        assert!(load_safetensors_state_only(&offsets).is_err());
        assert!(load_safetensors(&offsets).is_err());

        let valid_header = br#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut truncated_payload = (valid_header.len() as u64).to_le_bytes().to_vec();
        truncated_payload.extend_from_slice(valid_header);
        assert!(load_safetensors_state_only(&truncated_payload).is_err());
        assert!(load_safetensors(&truncated_payload).is_err());
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
}
