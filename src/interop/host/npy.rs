//! Bounded copy-based NumPy `.npy` version 1/2 interoperability.
//!
//! Only dense, portable little-endian primitive arrays are accepted. This is
//! deliberately neither a Python binding nor a zero-copy array interface.

use super::{BorrowedHostTensor, HostInteropError, HostTensorLayout};
use crate::{DType, Shape, TensorData};
use std::fmt;

const MAGIC: &[u8; 6] = b"\x93NUMPY";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NpyError {
    Magic,
    Version { major: u8, minor: u8 },
    Truncated,
    HeaderLength,
    HeaderAlignment,
    HeaderSyntax,
    HeaderField(&'static str),
    DuplicateField(&'static str),
    UnsupportedDType(DType),
    UnsupportedDescriptor(String),
    Endianness(String),
    ShapeOverflow,
    PayloadLength { expected: usize, actual: usize },
    HostLayout(HostInteropError),
    Codec,
}

impl fmt::Display for NpyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "npy error: {self:?}")
    }
}
impl std::error::Error for NpyError {}

#[derive(Clone, Copy)]
struct Descriptor {
    dtype: DType,
    tag: &'static str,
}

const DESCRIPTORS: &[Descriptor] = &[
    Descriptor {
        dtype: DType::Bool,
        tag: "|b1",
    },
    Descriptor {
        dtype: DType::I8,
        tag: "|i1",
    },
    Descriptor {
        dtype: DType::U8,
        tag: "|u1",
    },
    Descriptor {
        dtype: DType::I16,
        tag: "<i2",
    },
    Descriptor {
        dtype: DType::U16,
        tag: "<u2",
    },
    Descriptor {
        dtype: DType::I32,
        tag: "<i4",
    },
    Descriptor {
        dtype: DType::U32,
        tag: "<u4",
    },
    Descriptor {
        dtype: DType::I64,
        tag: "<i8",
    },
    Descriptor {
        dtype: DType::U64,
        tag: "<u8",
    },
    Descriptor {
        dtype: DType::F16,
        tag: "<f2",
    },
    Descriptor {
        dtype: DType::F32,
        tag: "<f4",
    },
    Descriptor {
        dtype: DType::F64,
        tag: "<f8",
    },
];

fn descriptor_for_dtype(dtype: DType) -> Result<&'static str, NpyError> {
    DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.dtype == dtype)
        .map(|descriptor| descriptor.tag)
        .ok_or(NpyError::UnsupportedDType(dtype))
}

fn dtype_for_descriptor(tag: &str) -> Result<DType, NpyError> {
    if tag.starts_with('>') || tag.starts_with('=') {
        return Err(NpyError::Endianness(tag.into()));
    }
    DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.tag == tag)
        .map(|descriptor| descriptor.dtype)
        .ok_or_else(|| NpyError::UnsupportedDescriptor(tag.into()))
}

/// Deterministically encodes a dense `TensorData` as a version 1 or 2 NPY
/// byte stream. It always emits a materialized C-order, little-endian array.
pub fn encode_npy(tensor: &TensorData) -> Result<Vec<u8>, NpyError> {
    let descriptor = descriptor_for_dtype(tensor.dtype())?;
    let shape = shape_literal(tensor.shape());
    let dictionary =
        format!("{{'descr': '{descriptor}', 'fortran_order': False, 'shape': {shape}, }}");
    let (major, header) = encode_header(&dictionary)?;
    let payload = tensor.to_le_bytes().map_err(|_| NpyError::Codec)?;
    let prefix = MAGIC
        .len()
        .checked_add(2)
        .and_then(|value| value.checked_add(if major == 1 { 2 } else { 4 }))
        .ok_or(NpyError::HeaderLength)?;
    let total = prefix
        .checked_add(header.len())
        .and_then(|value| value.checked_add(payload.len()))
        .ok_or(NpyError::ShapeOverflow)?;
    let mut out = Vec::new();
    out.try_reserve_exact(total).map_err(|_| NpyError::Codec)?;
    out.extend_from_slice(MAGIC);
    out.push(major);
    out.push(0);
    if major == 1 {
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    } else {
        out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    }
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Decodes a bounded primitive NPY v1/v2 byte stream into independent,
/// canonical row-major `TensorData`. Fortran-order payloads are materialized.
pub fn decode_npy(bytes: &[u8]) -> Result<TensorData, NpyError> {
    if bytes.len() < 8 || &bytes[..6] != MAGIC {
        return Err(NpyError::Magic);
    }
    let major = bytes[6];
    let minor = bytes[7];
    let (length_width, prefix) = match (major, minor) {
        (1, 0) => (2, 10),
        (2, 0) => (4, 12),
        _ => return Err(NpyError::Version { major, minor }),
    };
    if bytes.len() < prefix {
        return Err(NpyError::Truncated);
    }
    let header_len = if length_width == 2 {
        usize::from(u16::from_le_bytes(
            bytes[8..10].try_into().map_err(|_| NpyError::Truncated)?,
        ))
    } else {
        usize::try_from(u32::from_le_bytes(
            bytes[8..12].try_into().map_err(|_| NpyError::Truncated)?,
        ))
        .map_err(|_| NpyError::HeaderLength)?
    };
    let header_end = prefix
        .checked_add(header_len)
        .ok_or(NpyError::HeaderLength)?;
    if header_end > bytes.len() {
        return Err(NpyError::Truncated);
    }
    if header_end % 16 != 0 {
        return Err(NpyError::HeaderAlignment);
    }
    let header =
        std::str::from_utf8(&bytes[prefix..header_end]).map_err(|_| NpyError::HeaderSyntax)?;
    let parsed = parse_header(header)?;
    let dtype = dtype_for_descriptor(&parsed.descriptor)?;
    let shape = Shape::new(parsed.shape);
    let count = shape.numel().map_err(|_| NpyError::ShapeOverflow)?;
    let payload_len = count
        .checked_mul(dtype.itemsize())
        .ok_or(NpyError::ShapeOverflow)?;
    let payload = &bytes[header_end..];
    if payload.len() != payload_len {
        return Err(NpyError::PayloadLength {
            expected: payload_len,
            actual: payload.len(),
        });
    }
    let layout = if parsed.fortran_order {
        fortran_layout(dtype, shape.clone())?
    } else {
        HostTensorLayout::contiguous(dtype, shape.clone()).map_err(NpyError::HostLayout)?
    };
    BorrowedHostTensor::new(payload, layout)
        .map_err(NpyError::HostLayout)?
        .to_tensor_data()
        .map_err(NpyError::HostLayout)
}

fn shape_literal(shape: &Shape) -> String {
    match shape.dims() {
        [] => "()".into(),
        [dimension] => format!("({dimension},)"),
        dimensions => format!(
            "({})",
            dimensions
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn encode_header(dictionary: &str) -> Result<(u8, String), NpyError> {
    for (major, prefix) in [(1, 10usize), (2, 12usize)] {
        let padding = (16 - ((prefix + dictionary.len() + 1) % 16)) % 16;
        let header = format!("{dictionary}{}\n", " ".repeat(padding));
        if (major == 1 && u16::try_from(header.len()).is_ok())
            || (major == 2 && u32::try_from(header.len()).is_ok())
        {
            return Ok((major, header));
        }
    }
    Err(NpyError::HeaderLength)
}

fn fortran_layout(dtype: DType, shape: Shape) -> Result<HostTensorLayout, NpyError> {
    let width = isize::try_from(dtype.itemsize()).map_err(|_| NpyError::ShapeOverflow)?;
    let mut stride = width;
    let mut strides = Vec::with_capacity(shape.rank());
    for dimension in shape.dims() {
        strides.push(stride);
        stride = stride
            .checked_mul(isize::try_from(*dimension).map_err(|_| NpyError::ShapeOverflow)?)
            .ok_or(NpyError::ShapeOverflow)?;
    }
    HostTensorLayout::new(dtype, shape, 0, strides).map_err(NpyError::HostLayout)
}

#[derive(Debug)]
struct ParsedHeader {
    descriptor: String,
    fortran_order: bool,
    shape: Vec<usize>,
}

fn parse_header(header: &str) -> Result<ParsedHeader, NpyError> {
    let content = header.strip_suffix('\n').ok_or(NpyError::HeaderSyntax)?;
    let content = content.trim_end_matches(' ');
    if content.bytes().any(|byte| !byte.is_ascii()) {
        return Err(NpyError::HeaderSyntax);
    }
    let mut parser = HeaderParser::new(content);
    parser.dictionary()
}

struct HeaderParser<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> HeaderParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            position: 0,
        }
    }
    fn dictionary(&mut self) -> Result<ParsedHeader, NpyError> {
        self.punctuation(b'{')?;
        let mut descriptor = None;
        let mut fortran_order = None;
        let mut shape = None;
        loop {
            self.space();
            if self.consume(b'}') {
                break;
            }
            let key = self.string()?;
            self.punctuation(b':')?;
            match key.as_str() {
                "descr" => set_once(&mut descriptor, self.string()?, "descr")?,
                "fortran_order" => set_once(&mut fortran_order, self.boolean()?, "fortran_order")?,
                "shape" => set_once(&mut shape, self.shape()?, "shape")?,
                _ => return Err(NpyError::HeaderField("unknown")),
            }
            self.space();
            if self.consume(b'}') {
                break;
            }
            self.punctuation(b',')?;
        }
        self.space();
        if self.position != self.input.len() {
            return Err(NpyError::HeaderSyntax);
        }
        Ok(ParsedHeader {
            descriptor: descriptor.ok_or(NpyError::HeaderField("descr"))?,
            fortran_order: fortran_order.ok_or(NpyError::HeaderField("fortran_order"))?,
            shape: shape.ok_or(NpyError::HeaderField("shape"))?,
        })
    }
    fn shape(&mut self) -> Result<Vec<usize>, NpyError> {
        self.punctuation(b'(')?;
        let mut dimensions = Vec::new();
        self.space();
        if self.consume(b')') {
            return Ok(dimensions);
        }
        loop {
            dimensions.push(self.usize()?);
            self.space();
            if self.consume(b')') {
                return if dimensions.len() == 1 {
                    Err(NpyError::HeaderSyntax)
                } else {
                    Ok(dimensions)
                };
            }
            self.punctuation(b',')?;
            self.space();
            if self.consume(b')') {
                return Ok(dimensions);
            }
        }
    }
    fn boolean(&mut self) -> Result<bool, NpyError> {
        self.space();
        if self.word(b"True") {
            Ok(true)
        } else if self.word(b"False") {
            Ok(false)
        } else {
            Err(NpyError::HeaderSyntax)
        }
    }
    fn usize(&mut self) -> Result<usize, NpyError> {
        self.space();
        let start = self.position;
        while self.position < self.input.len() && self.input[self.position].is_ascii_digit() {
            self.position += 1;
        }
        if self.position == start {
            return Err(NpyError::HeaderSyntax);
        }
        std::str::from_utf8(&self.input[start..self.position])
            .ok()
            .and_then(|value| value.parse().ok())
            .ok_or(NpyError::ShapeOverflow)
    }
    fn string(&mut self) -> Result<String, NpyError> {
        self.space();
        let quote = *self
            .input
            .get(self.position)
            .ok_or(NpyError::HeaderSyntax)?;
        if quote != b'\'' && quote != b'"' {
            return Err(NpyError::HeaderSyntax);
        }
        self.position += 1;
        let start = self.position;
        while self.position < self.input.len() && self.input[self.position] != quote {
            if self.input[self.position] == b'\\' {
                return Err(NpyError::HeaderSyntax);
            }
            self.position += 1;
        }
        if self.position == self.input.len() {
            return Err(NpyError::HeaderSyntax);
        }
        let value = std::str::from_utf8(&self.input[start..self.position])
            .map_err(|_| NpyError::HeaderSyntax)?
            .to_owned();
        self.position += 1;
        Ok(value)
    }
    fn punctuation(&mut self, expected: u8) -> Result<(), NpyError> {
        self.space();
        if self.consume(expected) {
            Ok(())
        } else {
            Err(NpyError::HeaderSyntax)
        }
    }
    fn consume(&mut self, expected: u8) -> bool {
        if self.input.get(self.position) == Some(&expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }
    fn word(&mut self, word: &[u8]) -> bool {
        if self.input.get(self.position..self.position + word.len()) == Some(word) {
            self.position += word.len();
            true
        } else {
            false
        }
    }
    fn space(&mut self) {
        while self
            .input
            .get(self.position)
            .is_some_and(|byte| *byte == b' ')
        {
            self.position += 1;
        }
    }
}

fn set_once<T>(slot: &mut Option<T>, value: T, field: &'static str) -> Result<(), NpyError> {
    if slot.replace(value).is_some() {
        Err(NpyError::DuplicateField(field))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v1_fixture(header: &str, payload: &[u8]) -> Vec<u8> {
        assert!(header.ends_with('\n'));
        assert_eq!((10 + header.len()) % 16, 0);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&[1, 0]);
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn known_good_scalar_c_fortran_and_empty_fixtures_decode() {
        let scalar = v1_fixture(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (), }              \n",
            &[0, 0, 0, 0x80],
        );
        let decoded = decode_npy(&scalar).unwrap();
        assert_eq!(decoded.shape(), &Shape::new([]));
        assert_eq!(decoded.to_le_bytes().unwrap(), [0, 0, 0, 0x80]);

        let c_order = v1_fixture(
            "{'descr': '<i2', 'fortran_order': False, 'shape': (2, 3), }          \n",
            &[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0],
        );
        assert_eq!(
            decode_npy(&c_order).unwrap().to_le_bytes().unwrap(),
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]
        );

        let fortran = v1_fixture(
            "{'descr': '<i2', 'fortran_order': True, 'shape': (2, 3), }           \n",
            &[1, 0, 4, 0, 2, 0, 5, 0, 3, 0, 6, 0],
        );
        assert_eq!(
            decode_npy(&fortran).unwrap().to_le_bytes().unwrap(),
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]
        );

        let empty = v1_fixture(
            "{'descr': '|u1', 'fortran_order': False, 'shape': (0, 3), }          \n",
            &[],
        );
        let empty = decode_npy(&empty).unwrap();
        assert_eq!(empty.shape(), &Shape::new([0, 3]));
        assert!(empty.is_empty());
    }

    #[test]
    fn raw_float_specials_and_all_supported_dtype_tags_round_trip() {
        let specials =
            TensorData::from_le_bytes([2], DType::F32, &[0, 0, 0, 0x80, 1, 0, 0xc0, 0x7f]).unwrap();
        let encoded = encode_npy(&specials).unwrap();
        assert_eq!(
            decode_npy(&encoded).unwrap().to_le_bytes().unwrap(),
            specials.to_le_bytes().unwrap()
        );
        for dtype in DESCRIPTORS.iter().map(|descriptor| descriptor.dtype) {
            let width = dtype.itemsize();
            let raw = if dtype == DType::Bool {
                vec![0, 1]
            } else {
                (0..2 * width).map(|value| value as u8).collect()
            };
            let tensor = TensorData::from_le_bytes([2], dtype, &raw).unwrap();
            let encoded = encode_npy(&tensor).unwrap();
            let descriptor = descriptor_for_dtype(dtype).unwrap().as_bytes();
            assert!(
                encoded
                    .windows(descriptor.len())
                    .any(|window| window == descriptor)
            );
            assert_eq!(decode_npy(&encoded).unwrap().to_le_bytes().unwrap(), raw);
        }
        let bf16 = TensorData::from_le_bytes([1], DType::BF16, &[1, 0x7e]).unwrap();
        assert_eq!(
            encode_npy(&bf16),
            Err(NpyError::UnsupportedDType(DType::BF16))
        );
    }

    #[test]
    fn float8_has_no_portable_npy_descriptor() {
        for dtype in [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ] {
            let tensor = TensorData::from_le_bytes([1], dtype, &[0x80]).unwrap();
            assert_eq!(encode_npy(&tensor), Err(NpyError::UnsupportedDType(dtype)));
        }
    }

    #[test]
    fn output_is_deterministic_and_uses_v2_when_needed() {
        let tensor =
            TensorData::from_le_bytes([2, 2], DType::U16, &[1, 0, 2, 0, 3, 0, 4, 0]).unwrap();
        let first = encode_npy(&tensor).unwrap();
        assert_eq!(first, encode_npy(&tensor).unwrap());
        assert_eq!(&first[..8], b"\x93NUMPY\x01\x00");
        let huge_rank = TensorData::from_le_bytes(vec![0; 30_000], DType::U8, &[]).unwrap();
        let v2 = encode_npy(&huge_rank).unwrap();
        assert_eq!(&v2[..8], b"\x93NUMPY\x02\x00");
        assert_eq!(decode_npy(&v2).unwrap(), huge_rank);
    }

    #[test]
    fn malformed_headers_payloads_and_descriptors_fail_closed() {
        let good = encode_npy(&TensorData::from_le_bytes([1], DType::U8, &[9]).unwrap()).unwrap();
        assert_eq!(decode_npy(&good[..5]), Err(NpyError::Magic));
        let mut version = good.clone();
        version[6] = 3;
        assert_eq!(
            decode_npy(&version),
            Err(NpyError::Version { major: 3, minor: 0 })
        );
        let mut alignment = good.clone();
        alignment[8] = alignment[8].wrapping_sub(1);
        assert_eq!(decode_npy(&alignment), Err(NpyError::HeaderAlignment));
        let mut header_length = good.clone();
        header_length[8..10].copy_from_slice(&u16::MAX.to_le_bytes());
        assert_eq!(decode_npy(&header_length), Err(NpyError::Truncated));
        let mut no_newline = good.clone();
        let header_end = 10 + usize::from(u16::from_le_bytes(good[8..10].try_into().unwrap()));
        no_newline[header_end - 1] = b' ';
        assert_eq!(decode_npy(&no_newline), Err(NpyError::HeaderSyntax));
        let mut trailing = good.clone();
        trailing.push(0);
        assert_eq!(
            decode_npy(&trailing),
            Err(NpyError::PayloadLength {
                expected: 1,
                actual: 2
            })
        );
        let mut truncated = good.clone();
        truncated.pop();
        assert_eq!(
            decode_npy(&truncated),
            Err(NpyError::PayloadLength {
                expected: 1,
                actual: 0
            })
        );
        for header in [
            "{'descr': '<i2', 'fortran_order': False, 'shape': (1,), 'descr': '<i2', }\n",
            "{'descr': '<i2', 'fortran_order': False, 'shape': (1,), 'extra': 1, }\n",
            "{'descr': '<i2', 'fortran_order': False, }\n",
            "{'descr': '>i2', 'fortran_order': False, 'shape': (1,), }\n",
            "{'descr': '<V4', 'fortran_order': False, 'shape': (1,), }\n",
            "{'descr': '<i2', 'fortran_order': False, 'shape': (1) }\n",
        ] {
            let padding = (16 - ((10 + header.len()) % 16)) % 16;
            let mut padded = header.trim_end_matches('\n').to_owned();
            padded.push_str(&" ".repeat(padding));
            padded.push('\n');
            assert!(
                decode_npy(&v1_fixture(&padded, &[0, 0])).is_err(),
                "{header}"
            );
        }
    }
}
