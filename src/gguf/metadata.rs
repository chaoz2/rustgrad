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
