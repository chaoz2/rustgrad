//! Bounded portable node-table encoding for validated UOps.
use super::{UArg, UOp, UOpKind, UType};
use crate::DType;
use std::fmt;
const MAGIC: &[u8; 4] = b"RGUA";
const VERSION: u8 = 1;
const MAX_NODES: usize = 1 << 20;
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactError {
    Format(&'static str),
    Unsupported,
    Checksum,
}
impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "artifact: {self:?}")
    }
}
impl std::error::Error for ArtifactError {}
pub fn encode(root: &UOp) -> Result<Vec<u8>, ArtifactError> {
    let nodes = root
        .topological()
        .map_err(|_| ArtifactError::Format("dag"))?;
    if nodes.len() > MAX_NODES {
        return Err(ArtifactError::Format("limit"));
    }
    let mut out = Vec::new();
    out.extend(MAGIC);
    out.push(VERSION);
    out.extend((nodes.len() as u32).to_le_bytes());
    for n in nodes {
        match (n.kind(), n.ty(), n.arg(), n.sources().len()) {
            (UOpKind::Const, Some(UType { scalar, lanes: 1 }), UArg::Scalar { dtype, bits }, 0)
                if scalar == *dtype =>
            {
                out.push(1);
                out.push(dtype_tag(*dtype));
                out.extend(bits.to_le_bytes());
            }
            (UOpKind::Const, Some(UType { scalar, lanes: 1 }), UArg::Int(v), 0) => {
                out.push(2);
                out.push(dtype_tag(scalar));
                out.extend(v.to_le_bytes());
            }
            _ => return Err(ArtifactError::Unsupported),
        }
    }
    let sum = checksum(&out);
    out.extend(sum.to_le_bytes());
    Ok(out)
}
pub fn decode(bytes: &[u8]) -> Result<UOp, ArtifactError> {
    if bytes.len() < 13 || &bytes[..4] != MAGIC {
        return Err(ArtifactError::Format("magic"));
    }
    if bytes[4] != VERSION {
        return Err(ArtifactError::Format("version"));
    }
    let n = u32::from_le_bytes(bytes[5..9].try_into().unwrap()) as usize;
    if n == 0 || n > MAX_NODES {
        return Err(ArtifactError::Format("count"));
    }
    let body = 9 + n.checked_mul(10).ok_or(ArtifactError::Format("overflow"))?;
    if bytes.len() != body + 4 {
        return Err(ArtifactError::Format("length"));
    }
    let got = u32::from_le_bytes(bytes[body..].try_into().unwrap());
    if checksum(&bytes[..body]) != got {
        return Err(ArtifactError::Checksum);
    }
    let mut last = None;
    for i in 0..n {
        let p = 9 + i * 10;
        let tag = bytes[p];
        let dtype = dtype(bytes[p + 1])?;
        let bits = u64::from_le_bytes(bytes[p + 2..p + 10].try_into().unwrap());
        let ty = UType::scalar(dtype);
        last = Some(match tag {
            1 => UOp::scalar_constant(dtype, bits, ty),
            2 => UOp::constant(bits as i64, ty),
            _ => return Err(ArtifactError::Format("tag")),
        });
    }
    last.ok_or(ArtifactError::Format("empty"))
}
fn checksum(x: &[u8]) -> u32 {
    x.iter().fold(0x811c9dc5u32, |h, b| {
        (h ^ u32::from(*b)).wrapping_mul(0x01000193)
    })
}
fn dtype_tag(d: DType) -> u8 {
    match d {
        DType::Bool => 0,
        DType::I8 => 1,
        DType::U8 => 2,
        DType::I16 => 3,
        DType::U16 => 4,
        DType::I32 => 5,
        DType::U32 => 6,
        DType::I64 => 7,
        DType::U64 => 8,
        DType::F16 => 9,
        DType::BF16 => 10,
        DType::F32 => 11,
        DType::F64 => 12,
    }
}
fn dtype(t: u8) -> Result<DType, ArtifactError> {
    Ok(match t {
        0 => DType::Bool,
        1 => DType::I8,
        2 => DType::U8,
        3 => DType::I16,
        4 => DType::U16,
        5 => DType::I32,
        6 => DType::U32,
        7 => DType::I64,
        8 => DType::U64,
        9 => DType::F16,
        10 => DType::BF16,
        11 => DType::F32,
        12 => DType::F64,
        _ => return Err(ArtifactError::Format("dtype")),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_scalar_round_trip_is_deterministic() {
        for (d, b) in [
            (DType::U64, u64::MAX),
            (DType::F16, 0x8001),
            (DType::F32, 0x7fc01234),
            (DType::F64, 0x8000000000000000),
        ] {
            let x = UOp::scalar_constant(d, b, UType::scalar(d));
            let a = encode(&x).unwrap();
            assert_eq!(a, encode(&x).unwrap());
            let y = decode(&a).unwrap();
            assert_eq!(x, y);
        }
        assert!(matches!(decode(b"bad"), Err(ArtifactError::Format(_))));
    }
}
