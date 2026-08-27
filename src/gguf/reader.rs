use super::{
    GgmlType, GgufError, GgufErrorKind, GgufFile, GgufLimits, GgufMetadata, GgufMetadataType,
    GgufMetadataValue, GgufTensor, GgufVersion,
};
use crate::Shape;
use std::collections::BTreeSet;

pub(super) fn parse(bytes: &[u8], limits: GgufLimits) -> Result<GgufFile<'_>, GgufError> {
    let mut reader = Reader {
        bytes,
        pos: 0,
        limits,
        metadata_values: 0,
    };
    if reader.take(4)? != b"GGUF" {
        return Err(reader.error_at(GgufErrorKind::InvalidMagic, 0));
    }
    let version_offset = reader.pos;
    let version = match reader.u32()? {
        2 => GgufVersion::V2,
        3 => GgufVersion::V3,
        raw => {
            return Err(reader.error_at(GgufErrorKind::UnsupportedVersion(raw), version_offset));
        }
    };
    let tensor_count = reader.count("tensor count", limits.max_tensors)?;
    let metadata_count = reader.count("metadata count", limits.max_metadata_entries)?;

    let mut metadata = Vec::with_capacity(metadata_count);
    let mut metadata_keys = BTreeSet::new();
    for _ in 0..metadata_count {
        let key_offset = reader.pos;
        let key = reader.string(
            "metadata key",
            limits.max_string_bytes.min(u16::MAX as usize),
        )?;
        if key.is_empty() {
            return Err(reader.error_at(
                GgufErrorKind::EmptyName {
                    field: "metadata key",
                },
                key_offset,
            ));
        }
        if !valid_metadata_key(&key) {
            return Err(reader.error_at(GgufErrorKind::InvalidMetadataKey(key), key_offset));
        }
        if !metadata_keys.insert(key.clone()) {
            return Err(reader.error_at(GgufErrorKind::DuplicateMetadata(key), key_offset));
        }
        let value_type = reader.metadata_type()?;
        let value = reader.metadata_value(value_type, 0)?;
        metadata.push(GgufMetadata { key, value });
    }

    let alignment = metadata
        .iter()
        .find(|entry| entry.key == "general.alignment")
        .map_or(Ok(32usize), |entry| match entry.value {
            GgufMetadataValue::U32(value) => usize::try_from(value).map_err(|_| {
                reader.error_at(
                    GgufErrorKind::InvalidAlignment(u64::from(value)),
                    reader.pos,
                )
            }),
            _ => Err(reader.error_at(
                GgufErrorKind::InvalidAlignmentType(entry.value.value_type()),
                reader.pos,
            )),
        })?;
    if alignment == 0 || !alignment.is_multiple_of(8) {
        return Err(reader.error(GgufErrorKind::InvalidAlignment(alignment as u64)));
    }

    struct PendingTensor {
        name: String,
        dimensions: Vec<usize>,
        shape: Shape,
        elements: usize,
        kind: GgmlType,
        relative_offset: u64,
        info_offset: usize,
    }

    let mut pending = Vec::with_capacity(tensor_count);
    let mut tensor_names = BTreeSet::new();
    for _ in 0..tensor_count {
        let info_offset = reader.pos;
        let name = reader.string("tensor name", limits.max_string_bytes.min(64))?;
        if name.is_empty() {
            return Err(reader.error_at(
                GgufErrorKind::EmptyName {
                    field: "tensor name",
                },
                info_offset,
            ));
        }
        if !tensor_names.insert(name.clone()) {
            return Err(reader.error_at(GgufErrorKind::DuplicateTensor(name), info_offset));
        }
        let rank_offset = reader.pos;
        let rank = reader.u32()?;
        let rank_limit = limits.max_rank.min(4);
        if rank == 0 || usize::try_from(rank).map_or(true, |rank| rank > rank_limit) {
            return Err(reader.error_at(
                GgufErrorKind::InvalidRank { tensor: name, rank },
                rank_offset,
            ));
        }
        let mut dimensions = Vec::with_capacity(rank as usize);
        let mut elements = 1usize;
        for axis in 0..rank as usize {
            let dim_offset = reader.pos;
            let raw = reader.u64()?;
            let dimension = usize::try_from(raw)
                .ok()
                .filter(|&value| value != 0)
                .ok_or_else(|| {
                    reader.error_at(
                        GgufErrorKind::InvalidDimension {
                            tensor: name.clone(),
                            axis,
                            value: raw,
                        },
                        dim_offset,
                    )
                })?;
            elements = elements.checked_mul(dimension).ok_or_else(|| {
                reader.error_at(
                    GgufErrorKind::ShapeOverflow {
                        tensor: name.clone(),
                    },
                    dim_offset,
                )
            })?;
            dimensions.push(dimension);
        }
        let kind_offset = reader.pos;
        let raw_kind = reader.u32()?;
        let kind = GgmlType::from_raw(raw_kind).ok_or_else(|| {
            reader.error_at(GgufErrorKind::UnknownTensorType(raw_kind), kind_offset)
        })?;
        let relative_offset = reader.u64()?;
        let mut logical = dimensions.clone();
        logical.reverse();
        pending.push(PendingTensor {
            name,
            dimensions,
            shape: Shape::new(logical),
            elements,
            kind,
            relative_offset,
            info_offset,
        });
    }

    let data_offset = if tensor_count == 0 {
        Some(reader.pos)
    } else {
        align_up(reader.pos, alignment)
    }
    .ok_or_else(|| {
        reader.error(GgufErrorKind::LimitExceeded {
            field: "aligned header",
            value: reader.pos as u64,
            limit: usize::MAX,
        })
    })?;
    if data_offset > bytes.len() {
        return Err(reader.error(GgufErrorKind::Truncated));
    }
    if bytes[reader.pos..data_offset].iter().any(|&byte| byte != 0) {
        return Err(reader.error_at(
            GgufErrorKind::InvalidPadding { section: "header" },
            reader.pos,
        ));
    }

    let mut tensors = Vec::with_capacity(pending.len());
    for item in pending {
        let relative = usize::try_from(item.relative_offset).map_err(|_| {
            reader.error_at(
                GgufErrorKind::TensorRangeOutOfBounds {
                    tensor: item.name.clone(),
                },
                item.info_offset,
            )
        })?;
        if relative % alignment != 0 {
            return Err(reader.error_at(
                GgufErrorKind::MisalignedTensorOffset {
                    tensor: item.name,
                    offset: item.relative_offset,
                    alignment,
                },
                item.info_offset,
            ));
        }
        let byte_len =
            super::tensor::byte_len(&item.name, item.elements, item.kind, item.info_offset)?;
        let start = data_offset.checked_add(relative).ok_or_else(|| {
            reader.error_at(
                GgufErrorKind::TensorRangeOutOfBounds {
                    tensor: item.name.clone(),
                },
                item.info_offset,
            )
        })?;
        let end = start
            .checked_add(byte_len)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| {
                reader.error_at(
                    GgufErrorKind::TensorRangeOutOfBounds {
                        tensor: item.name.clone(),
                    },
                    item.info_offset,
                )
            })?;
        tensors.push(GgufTensor {
            name: item.name,
            dimensions: item.dimensions,
            shape: item.shape,
            elements: item.elements,
            kind: item.kind,
            relative_offset: item.relative_offset,
            raw_range: start..end,
        });
    }

    let mut by_range: Vec<_> = tensors.iter().collect();
    by_range.sort_by_key(|tensor| tensor.raw_range.start);
    if let Some(first) = by_range.first() {
        if bytes[data_offset..first.raw_range.start]
            .iter()
            .any(|&byte| byte != 0)
        {
            return Err(reader.error_at(
                GgufErrorKind::InvalidPadding { section: "tensor" },
                data_offset,
            ));
        }
    }
    for pair in by_range.windows(2) {
        if pair[1].raw_range.start < pair[0].raw_range.end {
            return Err(reader.error_at(
                GgufErrorKind::OverlappingTensors {
                    first: pair[0].name.clone(),
                    second: pair[1].name.clone(),
                },
                pair[1].raw_range.start,
            ));
        }
        if bytes[pair[0].raw_range.end..pair[1].raw_range.start]
            .iter()
            .any(|&byte| byte != 0)
        {
            return Err(reader.error_at(
                GgufErrorKind::InvalidPadding { section: "tensor" },
                pair[0].raw_range.end,
            ));
        }
    }
    let used_end = by_range
        .last()
        .map_or(data_offset, |tensor| tensor.raw_range.end);
    let padded_end = if by_range.is_empty() {
        used_end
    } else {
        data_offset
            .checked_add(
                align_up(used_end - data_offset, alignment)
                    .ok_or_else(|| reader.error(GgufErrorKind::Truncated))?,
            )
            .ok_or_else(|| reader.error(GgufErrorKind::Truncated))?
    };
    if bytes.len() > padded_end {
        return Err(reader.error_at(
            GgufErrorKind::TrailingData {
                bytes: bytes.len() - padded_end,
            },
            padded_end,
        ));
    }
    if bytes[used_end..].iter().any(|&byte| byte != 0) {
        return Err(reader.error_at(
            GgufErrorKind::InvalidPadding {
                section: "trailing",
            },
            used_end,
        ));
    }

    Ok(GgufFile {
        bytes,
        version,
        alignment,
        data_offset,
        metadata,
        tensors,
    })
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    value.checked_add((alignment - value % alignment) % alignment)
}

fn valid_metadata_key(key: &str) -> bool {
    key.is_ascii()
        && key.split('.').all(|segment| {
            !segment.is_empty()
                && !segment.starts_with('_')
                && !segment.ends_with('_')
                && segment
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        })
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
    limits: GgufLimits,
    metadata_values: usize,
}

impl<'a> Reader<'a> {
    fn error(&self, kind: GgufErrorKind) -> GgufError {
        self.error_at(kind, self.pos)
    }

    fn error_at(&self, kind: GgufErrorKind, offset: usize) -> GgufError {
        GgufError::new(kind, offset)
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| self.error(GgufErrorKind::Truncated))?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("two-byte slice"),
        ))
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four-byte slice"),
        ))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight-byte slice"),
        ))
    }

    fn count(&mut self, field: &'static str, limit: usize) -> Result<usize, GgufError> {
        let offset = self.pos;
        let value = self.u64()?;
        usize::try_from(value)
            .ok()
            .filter(|&count| count <= limit)
            .ok_or_else(|| {
                self.error_at(
                    GgufErrorKind::LimitExceeded {
                        field,
                        value,
                        limit,
                    },
                    offset,
                )
            })
    }

    fn string(&mut self, field: &'static str, limit: usize) -> Result<String, GgufError> {
        let offset = self.pos;
        let len = self.count(field, limit)?;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| self.error_at(GgufErrorKind::InvalidUtf8 { field }, offset))
    }

    fn metadata_type(&mut self) -> Result<GgufMetadataType, GgufError> {
        let offset = self.pos;
        let raw = self.u32()?;
        GgufMetadataType::from_raw(raw)
            .ok_or_else(|| self.error_at(GgufErrorKind::UnknownMetadataType(raw), offset))
    }

    fn metadata_value(
        &mut self,
        value_type: GgufMetadataType,
        depth: usize,
    ) -> Result<GgufMetadataValue, GgufError> {
        self.metadata_values = self.metadata_values.checked_add(1).ok_or_else(|| {
            self.error(GgufErrorKind::LimitExceeded {
                field: "metadata values",
                value: u64::MAX,
                limit: self.limits.max_metadata_values,
            })
        })?;
        if self.metadata_values > self.limits.max_metadata_values {
            return Err(self.error(GgufErrorKind::LimitExceeded {
                field: "metadata values",
                value: self.metadata_values as u64,
                limit: self.limits.max_metadata_values,
            }));
        }
        Ok(match value_type {
            GgufMetadataType::U8 => GgufMetadataValue::U8(self.u8()?),
            GgufMetadataType::I8 => GgufMetadataValue::I8(self.u8()? as i8),
            GgufMetadataType::U16 => GgufMetadataValue::U16(self.u16()?),
            GgufMetadataType::I16 => GgufMetadataValue::I16(self.u16()? as i16),
            GgufMetadataType::U32 => GgufMetadataValue::U32(self.u32()?),
            GgufMetadataType::I32 => GgufMetadataValue::I32(self.u32()? as i32),
            GgufMetadataType::F32 => GgufMetadataValue::F32(f32::from_bits(self.u32()?)),
            GgufMetadataType::Bool => {
                let offset = self.pos;
                match self.u8()? {
                    0 => GgufMetadataValue::Bool(false),
                    1 => GgufMetadataValue::Bool(true),
                    raw => return Err(self.error_at(GgufErrorKind::InvalidBoolean(raw), offset)),
                }
            }
            GgufMetadataType::String => GgufMetadataValue::String(
                self.string("metadata string", self.limits.max_string_bytes)?,
            ),
            GgufMetadataType::Array => {
                if depth >= self.limits.max_array_depth {
                    return Err(self.error(GgufErrorKind::ArrayNestingLimit));
                }
                let element_type = self.metadata_type()?;
                let count = self.count("array elements", self.limits.max_array_elements)?;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.metadata_value(element_type, depth + 1)?);
                }
                GgufMetadataValue::Array {
                    element_type,
                    values,
                }
            }
            GgufMetadataType::U64 => GgufMetadataValue::U64(self.u64()?),
            GgufMetadataType::I64 => GgufMetadataValue::I64(self.u64()? as i64),
            GgufMetadataType::F64 => GgufMetadataValue::F64(f64::from_bits(self.u64()?)),
        })
    }
}
