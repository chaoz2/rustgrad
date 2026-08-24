use std::{error, fmt};

/// Wire-level GGUF metadata value tags.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GgufMetadataType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl GgufMetadataType {
    pub(super) fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        })
    }
}

/// Lossless typed GGUF metadata.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: GgufMetadataType,
        values: Vec<GgufMetadataValue>,
    },
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufMetadataValue {
    pub const fn value_type(&self) -> GgufMetadataType {
        match self {
            Self::U8(_) => GgufMetadataType::U8,
            Self::I8(_) => GgufMetadataType::I8,
            Self::U16(_) => GgufMetadataType::U16,
            Self::I16(_) => GgufMetadataType::I16,
            Self::U32(_) => GgufMetadataType::U32,
            Self::I32(_) => GgufMetadataType::I32,
            Self::F32(_) => GgufMetadataType::F32,
            Self::Bool(_) => GgufMetadataType::Bool,
            Self::String(_) => GgufMetadataType::String,
            Self::Array { .. } => GgufMetadataType::Array,
            Self::U64(_) => GgufMetadataType::U64,
            Self::I64(_) => GgufMetadataType::I64,
            Self::F64(_) => GgufMetadataType::F64,
        }
    }

    /// Returns this value as a string when its wire type is `STRING`.
    pub const fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value.as_str()),
            _ => None,
        }
    }

    /// Returns this value as a boolean when its wire type is `BOOL`.
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// Converts any non-negative GGUF integer wire value to `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(value) => Some(u64::from(*value)),
            Self::I8(value) => u64::try_from(*value).ok(),
            Self::U16(value) => Some(u64::from(*value)),
            Self::I16(value) => u64::try_from(*value).ok(),
            Self::U32(value) => Some(u64::from(*value)),
            Self::I32(value) => u64::try_from(*value).ok(),
            Self::U64(value) => Some(*value),
            Self::I64(value) => u64::try_from(*value).ok(),
            _ => None,
        }
    }

    /// Returns the homogeneous array element type and values.
    pub fn as_array(&self) -> Option<(GgufMetadataType, &[Self])> {
        match self {
            Self::Array {
                element_type,
                values,
            } => Some((*element_type, values)),
            _ => None,
        }
    }
}

/// Expected logical type for a typed GGUF metadata lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GgufMetadataExpectation {
    String,
    Bool,
    UnsignedInteger,
    StringArray,
    IntegerArray,
}

/// A metadata entry exists but cannot satisfy a typed lookup.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GgufMetadataAccessError {
    TypeMismatch {
        key: String,
        expected: GgufMetadataExpectation,
        actual: GgufMetadataType,
    },
    NegativeInteger {
        key: String,
    },
    IntegerOutOfRange {
        key: String,
    },
    ArrayElementTypeMismatch {
        key: String,
        expected: GgufMetadataExpectation,
        actual: GgufMetadataType,
    },
}

impl fmt::Display for GgufMetadataAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid GGUF metadata: {self:?}")
    }
}

impl error::Error for GgufMetadataAccessError {}

pub(super) fn lookup_string<'a>(
    value: Option<&'a GgufMetadataValue>,
    key: &str,
) -> Result<Option<&'a str>, GgufMetadataAccessError> {
    value
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| GgufMetadataAccessError::TypeMismatch {
                    key: key.to_owned(),
                    expected: GgufMetadataExpectation::String,
                    actual: value.value_type(),
                })
        })
        .transpose()
}

pub(super) fn lookup_bool(
    value: Option<&GgufMetadataValue>,
    key: &str,
) -> Result<Option<bool>, GgufMetadataAccessError> {
    value
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| GgufMetadataAccessError::TypeMismatch {
                    key: key.to_owned(),
                    expected: GgufMetadataExpectation::Bool,
                    actual: value.value_type(),
                })
        })
        .transpose()
}

pub(super) fn lookup_u64(
    value: Option<&GgufMetadataValue>,
    key: &str,
) -> Result<Option<u64>, GgufMetadataAccessError> {
    value
        .map(|value| match value {
            GgufMetadataValue::I8(value) if *value < 0 => {
                Err(GgufMetadataAccessError::NegativeInteger {
                    key: key.to_owned(),
                })
            }
            GgufMetadataValue::I16(value) if *value < 0 => {
                Err(GgufMetadataAccessError::NegativeInteger {
                    key: key.to_owned(),
                })
            }
            GgufMetadataValue::I32(value) if *value < 0 => {
                Err(GgufMetadataAccessError::NegativeInteger {
                    key: key.to_owned(),
                })
            }
            GgufMetadataValue::I64(value) if *value < 0 => {
                Err(GgufMetadataAccessError::NegativeInteger {
                    key: key.to_owned(),
                })
            }
            _ => value
                .as_u64()
                .ok_or_else(|| GgufMetadataAccessError::TypeMismatch {
                    key: key.to_owned(),
                    expected: GgufMetadataExpectation::UnsignedInteger,
                    actual: value.value_type(),
                }),
        })
        .transpose()
}

pub(super) fn lookup_strings<'a>(
    value: Option<&'a GgufMetadataValue>,
    key: &str,
) -> Result<Option<Vec<&'a str>>, GgufMetadataAccessError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Some((element_type, values)) = value.as_array() else {
        return Err(GgufMetadataAccessError::TypeMismatch {
            key: key.to_owned(),
            expected: GgufMetadataExpectation::StringArray,
            actual: value.value_type(),
        });
    };
    if element_type != GgufMetadataType::String {
        return Err(GgufMetadataAccessError::ArrayElementTypeMismatch {
            key: key.to_owned(),
            expected: GgufMetadataExpectation::StringArray,
            actual: element_type,
        });
    }
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| GgufMetadataAccessError::ArrayElementTypeMismatch {
                    key: key.to_owned(),
                    expected: GgufMetadataExpectation::StringArray,
                    actual: value.value_type(),
                })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

pub(super) fn lookup_integers(
    value: Option<&GgufMetadataValue>,
    key: &str,
) -> Result<Option<Vec<i64>>, GgufMetadataAccessError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Some((element_type, values)) = value.as_array() else {
        return Err(GgufMetadataAccessError::TypeMismatch {
            key: key.to_owned(),
            expected: GgufMetadataExpectation::IntegerArray,
            actual: value.value_type(),
        });
    };
    if !matches!(
        element_type,
        GgufMetadataType::U8
            | GgufMetadataType::I8
            | GgufMetadataType::U16
            | GgufMetadataType::I16
            | GgufMetadataType::U32
            | GgufMetadataType::I32
            | GgufMetadataType::U64
            | GgufMetadataType::I64
    ) {
        return Err(GgufMetadataAccessError::ArrayElementTypeMismatch {
            key: key.to_owned(),
            expected: GgufMetadataExpectation::IntegerArray,
            actual: element_type,
        });
    }
    values
        .iter()
        .map(|value| match value {
            GgufMetadataValue::U8(value) => Ok(i64::from(*value)),
            GgufMetadataValue::I8(value) => Ok(i64::from(*value)),
            GgufMetadataValue::U16(value) => Ok(i64::from(*value)),
            GgufMetadataValue::I16(value) => Ok(i64::from(*value)),
            GgufMetadataValue::U32(value) => Ok(i64::from(*value)),
            GgufMetadataValue::I32(value) => Ok(i64::from(*value)),
            GgufMetadataValue::U64(integer) => {
                i64::try_from(*integer).map_err(|_| GgufMetadataAccessError::IntegerOutOfRange {
                    key: key.to_owned(),
                })
            }
            GgufMetadataValue::I64(value) => Ok(*value),
            _ => Err(GgufMetadataAccessError::ArrayElementTypeMismatch {
                key: key.to_owned(),
                expected: GgufMetadataExpectation::IntegerArray,
                actual: value.value_type(),
            }),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// One metadata key/value pair in original file order.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufMetadata {
    pub(super) key: String,
    pub(super) value: GgufMetadataValue,
}

impl GgufMetadata {
    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn value(&self) -> &GgufMetadataValue {
        &self.value
    }
}
