use super::{DType, Scalar};
use core::hash::{Hash, Hasher};

/// A storage-less scalar literal resolved against a graph operand before
/// lowering. It is never a tensor storage or artifact dtype.
#[derive(Clone, Copy, Debug)]
pub enum LiteralScalar {
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
}

impl LiteralScalar {
    pub(crate) fn scalar(self) -> Scalar {
        match self {
            Self::Bool(x) => Scalar::Bool(x),
            Self::I64(x) => Scalar::I(x),
            Self::U64(x) => Scalar::U(x),
            Self::F64(x) => Scalar::F(x),
        }
    }
    pub(crate) fn dtype_against(self, peer: DType) -> DType {
        match self {
            Self::Bool(_) => peer.promote(DType::Bool),
            Self::I64(_) | Self::U64(_) => {
                if peer == DType::Bool {
                    DType::I32
                } else {
                    peer
                }
            }
            Self::F64(_) => {
                if peer.is_float() {
                    peer
                } else {
                    DType::F32
                }
            }
        }
    }
    pub(crate) fn default_dtype(self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::I64(_) => DType::I32,
            Self::U64(_) => DType::U32,
            Self::F64(_) => DType::F32,
        }
    }
    fn float_bits(x: f64) -> u64 {
        if x.is_nan() {
            f64::NAN.to_bits()
        } else {
            x.to_bits()
        }
    }
}
impl PartialEq for LiteralScalar {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::I64(a), Self::I64(b)) => a == b,
            (Self::U64(a), Self::U64(b)) => a == b,
            (Self::F64(a), Self::F64(b)) => Self::float_bits(a) == Self::float_bits(b),
            _ => false,
        }
    }
}
impl Eq for LiteralScalar {}
impl Hash for LiteralScalar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match *self {
            Self::Bool(x) => x.hash(state),
            Self::I64(x) => x.hash(state),
            Self::U64(x) => x.hash(state),
            Self::F64(x) => Self::float_bits(x).hash(state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_canonicalizes_nan_but_keeps_signed_zero() {
        assert_eq!(
            LiteralScalar::F64(f64::NAN),
            LiteralScalar::F64(f64::from_bits(0x7ff0_0000_0000_0001))
        );
        assert_ne!(LiteralScalar::F64(0.0), LiteralScalar::F64(-0.0));
    }
    #[test]
    fn resolution_never_returns_a_weak_storage_dtype() {
        assert_eq!(
            LiteralScalar::I64(-1000).dtype_against(DType::I8),
            DType::I8
        );
        assert_eq!(LiteralScalar::U64(1).dtype_against(DType::U16), DType::U16);
        assert_eq!(LiteralScalar::I64(1).dtype_against(DType::Bool), DType::I32);
        assert_eq!(
            LiteralScalar::F64(-0.0).dtype_against(DType::I32),
            DType::F32
        );
        assert_eq!(
            LiteralScalar::F64(1.0).dtype_against(DType::F64),
            DType::F64
        );
    }
}
