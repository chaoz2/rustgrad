use super::{AddressSpace, AffineView, UType};
use crate::{DType, NodeId, Shape, SymbolicExpr};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LiteralValue {
    Int(i64),
    Scalar { dtype: DType, bits: u64 },
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VariableValue {
    pub name: String,
    pub bounds: SymbolicExpr,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AddressValue {
    pub space: AddressSpace,
    pub name: String,
    pub element: UType,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum IndexAddressing {
    Broadcast,
    Projected,
    /// An explicit projected address paired with a Bool validity expression.
    /// Loads return the dtype's canonical zero when validity is false and must
    /// not access the underlying buffer on that lane.
    Predicated,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum IndexValue {
    Buffer {
        buffer: u64,
        elements: usize,
        input_shape: Shape,
        output_shape: Shape,
        addressing: IndexAddressing,
    },
    View {
        buffer: u64,
        elements: usize,
        input_shape: Shape,
        output_shape: Shape,
        view: AffineView,
    },
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ReductionValue {
    pub input_shape: Shape,
    pub output_shape: Shape,
    pub axes: Vec<usize>,
    pub keepdim: bool,
    pub kind: crate::ReduceKind,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MatmulValue {
    Serial(Box<crate::MatmulKernelPlan>),
    Tiled(Box<crate::TiledMatmulPayload>),
    TensorCore(Box<crate::TensorCoreMatmulPayload>),
    Quantized(Box<crate::QuantizedMatmulPlan>),
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MovementValue {
    Plan(Box<crate::MovementKernelPlan>),
    QuantizedRowGather(Box<crate::QuantizedRowGatherPlan>),
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PrefixScanValue {
    pub input: NodeId,
    pub destination: NodeId,
    pub input_shape: Shape,
    pub output_shape: Shape,
    pub axis: usize,
    pub kind: crate::PrefixScanKind,
    pub output: crate::PrefixScanOutput,
    pub input_dtype: DType,
    pub dtype: DType,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SortValue {
    pub input: NodeId,
    pub input_shape: Shape,
    pub axis: usize,
    pub descending: bool,
    pub values: NodeId,
    pub indices: NodeId,
    pub dtype: DType,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TensorGuardValue {
    pub input: NodeId,
    pub input_shape: Shape,
    pub axis: usize,
    pub dtype: DType,
}

/// Complete dependency-bearing live Threefry2x32 semantic. The two graph
/// operands are packed U64 words; shapes are retained so capture and cache
/// identity do not depend on a live graph.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ThreefryValue {
    pub counter: NodeId,
    pub key: NodeId,
    pub counter_shape: Shape,
    pub key_shape: Shape,
    pub output: NodeId,
    pub output_shape: Shape,
}

impl ThreefryValue {
    pub fn validate(&self) -> Result<(), super::UOpError> {
        if self.output == self.counter
            || self.output == self.key
            || (self.counter == self.key && self.counter_shape != self.key_shape)
            || self
                .counter_shape
                .broadcast_with(&self.key_shape)
                .ok()
                .as_ref()
                != Some(&self.output_shape)
        {
            return Err(super::UOpError::InvalidArgument);
        }
        for (_, shape, _) in self.buffer_operands() {
            shape
                .numel()
                .ok()
                .and_then(|elements| elements.checked_mul(DType::U64.itemsize()))
                .ok_or(super::UOpError::InvalidArgument)?;
        }
        Ok(())
    }

    /// Canonical pointer ABI: first semantic use wins and an aliased key does
    /// not create a duplicate pointer. The output is always a distinct final
    /// buffer because graph nodes are append-only.
    pub(crate) fn input_operands(&self) -> impl Iterator<Item = (NodeId, &Shape)> {
        std::iter::once((self.counter, &self.counter_shape))
            .chain((self.key != self.counter).then_some((self.key, &self.key_shape)))
    }

    /// Complete canonical pointer ABI. Inputs retain first-use order and the
    /// distinct mutable output is always last.
    pub(crate) fn buffer_operands(&self) -> impl Iterator<Item = (NodeId, &Shape, bool)> {
        self.input_operands()
            .map(|(node, shape)| (node, shape, false))
            .chain(std::iter::once((self.output, &self.output_shape, true)))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SourceArity {
    Exact(usize),
    NonEmpty,
    Any,
}

impl SourceArity {
    pub(crate) fn accepts(self, actual: usize) -> bool {
        match self {
            Self::Exact(expected) => actual == expected,
            Self::NonEmpty => actual != 0,
            Self::Any => true,
        }
    }

    pub(crate) fn expected(self) -> &'static str {
        match self {
            Self::Exact(0) => "no sources",
            Self::Exact(1) => "one source",
            Self::Exact(2) => "two sources",
            Self::Exact(3) => "three sources",
            Self::Exact(_) => "the declared source count",
            Self::NonEmpty => "one or more sources",
            Self::Any => "any source count",
        }
    }
}

macro_rules! operation_source_arity {
    (zero) => {
        SourceArity::Exact(0)
    };
    (zero, $value:ident) => {{
        let _ = $value;
        SourceArity::Exact(0)
    }};
    (one) => {
        SourceArity::Exact(1)
    };
    (one, $value:ident) => {{
        let _ = $value;
        SourceArity::Exact(1)
    }};
    (two) => {
        SourceArity::Exact(2)
    };
    (two, $value:ident) => {{
        let _ = $value;
        SourceArity::Exact(2)
    }};
    (three) => {
        SourceArity::Exact(3)
    };
    (three, $value:ident) => {{
        let _ = $value;
        SourceArity::Exact(3)
    }};
    (nonempty) => {
        SourceArity::NonEmpty
    };
    (nonempty, $value:ident) => {{
        let _ = $value;
        SourceArity::NonEmpty
    }};
    (any) => {
        SourceArity::Any
    };
    (any, $value:ident) => {{
        let _ = $value;
        SourceArity::Any
    }};
    (graph_logical, $value:ident) => {
        match $value {
            crate::LogicalOp::Not => SourceArity::Exact(1),
            crate::LogicalOp::And | crate::LogicalOp::Or => SourceArity::Exact(2),
        }
    };
    (index, $value:ident) => {
        match $value {
            IndexValue::Buffer {
                addressing: IndexAddressing::Predicated,
                ..
            } => SourceArity::Exact(3),
            _ => SourceArity::Exact(2),
        }
    };
}

macro_rules! operation_is_pure {
    (pure) => {
        true
    };
    (pure, $value:ident) => {{
        let _ = $value;
        true
    }};
    (impure) => {
        false
    };
    (impure, $value:ident) => {{
        let _ = $value;
        false
    }};
}

macro_rules! define_operations {
    (
        $(
            $(#[$docs:meta])*
            $variant:ident $(($binding:ident : $payload:ty))?
                => [$arity:ident, $purity:ident, $wire:ident $(($wire_binding:ident : $wire_payload:ty))?, $tag:literal, $gate:ident, $sub:ident];
        )*
    ) => {
        /// A closed, typed universal operation. The variant is the semantic identity;
        /// payload-bearing operations cannot exist without their matching payload.
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub enum Operation {
            $(
                $(#[$docs])*
                $variant $(($payload))?,
            )*
        }

        impl Operation {
            pub(crate) fn source_arity(&self) -> SourceArity {
                match self {
                    $(
                        Self::$variant $(($binding))? =>
                            operation_source_arity!($arity $(, $binding)?),
                    )*
                }
            }

            pub(crate) fn is_pure(&self) -> bool {
                match self {
                    $(
                        Self::$variant $(($binding))? =>
                            operation_is_pure!($purity $(, $binding)?),
                    )*
                }
            }
        }
    };
}

super::schema::uop_operation_schema!(define_operations);

#[cfg(test)]
mod tests {
    use super::*;

    fn index(addressing: IndexAddressing) -> Operation {
        Operation::Index(IndexValue::Buffer {
            buffer: 0,
            elements: 1,
            input_shape: Shape::from([1]),
            output_shape: Shape::from([1]),
            addressing,
        })
    }

    #[test]
    fn structural_metadata_handles_payload_arity_and_effects() {
        assert_eq!(
            Operation::GraphLogical(crate::LogicalOp::Not).source_arity(),
            SourceArity::Exact(1)
        );
        assert_eq!(
            Operation::GraphLogical(crate::LogicalOp::And).source_arity(),
            SourceArity::Exact(2)
        );
        assert_eq!(
            index(IndexAddressing::Broadcast).source_arity(),
            SourceArity::Exact(2)
        );
        assert_eq!(
            index(IndexAddressing::Projected).source_arity(),
            SourceArity::Exact(2)
        );
        assert_eq!(
            index(IndexAddressing::Predicated).source_arity(),
            SourceArity::Exact(3)
        );
        assert_eq!(Operation::Vectorize.source_arity(), SourceArity::NonEmpty);
        assert_eq!(Operation::Sink.source_arity(), SourceArity::Any);

        assert!(Operation::Load.is_pure());
        assert!(!Operation::Store.is_pure());
        assert!(!Operation::Barrier.is_pure());
        assert!(!Operation::Sink.is_pure());
    }
}
