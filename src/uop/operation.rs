use super::{AddressSpace, AffineView, Binary, Ternary, UArg, UOpError, UType, spec};
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

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum IndexValue {
    Buffer {
        buffer: u64,
        elements: usize,
        input_shape: Shape,
        output_shape: Shape,
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
    pub mean: bool,
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
    pub input_shape: Shape,
    pub output_shape: Shape,
    pub axis: usize,
    pub kind: crate::PrefixScanKind,
    pub output: crate::PrefixScanOutput,
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

#[derive(Clone, Copy, Debug)]
pub enum UArgRef<'a> {
    None,
    Int(&'a i64),
    Scalar {
        dtype: &'a DType,
        bits: &'a u64,
    },
    Name(&'a str),
    Variable {
        name: &'a str,
        bounds: &'a SymbolicExpr,
    },
    Address {
        space: &'a AddressSpace,
        name: &'a str,
        element: &'a UType,
    },
    RangeAxis(&'a u32),
    GepLane(&'a u16),
    BufferIndex {
        buffer: &'a u64,
        elements: &'a usize,
        input_shape: &'a Shape,
        output_shape: &'a Shape,
    },
    ViewBufferIndex {
        buffer: &'a u64,
        elements: &'a usize,
        input_shape: &'a Shape,
        output_shape: &'a Shape,
        view: &'a AffineView,
    },
    Reduction {
        input_shape: &'a Shape,
        output_shape: &'a Shape,
        axes: &'a [usize],
        keepdim: &'a bool,
        kind: &'a crate::ReduceKind,
        mean: &'a bool,
    },
    Matmul(&'a crate::MatmulKernelPlan),
    Conv2d(&'a crate::StaticConv2dPlan),
    TiledMatmul(&'a crate::TiledMatmulPayload),
    TensorCoreMatmul(&'a crate::TensorCoreMatmulPayload),
    QuantizedMatmul(&'a crate::QuantizedMatmulPlan),
    QuantizedRowGather(&'a crate::QuantizedRowGatherPlan),
    Movement(&'a crate::MovementKernelPlan),
    Random(&'a crate::random::plan::RandomKernelPlan),
    PrefixScan {
        input: &'a NodeId,
        input_shape: &'a Shape,
        output_shape: &'a Shape,
        axis: &'a usize,
        kind: &'a crate::PrefixScanKind,
        output: &'a crate::PrefixScanOutput,
        dtype: &'a DType,
    },
    Sort {
        input: &'a NodeId,
        input_shape: &'a Shape,
        axis: &'a usize,
        descending: &'a bool,
        values: &'a NodeId,
        indices: &'a NodeId,
        dtype: &'a DType,
    },
    TensorGuard {
        input: &'a NodeId,
        input_shape: &'a Shape,
        axis: &'a usize,
        dtype: &'a DType,
    },
    Effect(&'a crate::EffectPayload),
}

impl<'a> UArgRef<'a> {
    /// Creates the owned compatibility payload used by artifact and rewrite
    /// boundaries. Interpreters and renderers should consume this borrowed
    /// view directly instead of cloning operation payloads.
    pub fn to_owned(self) -> UArg {
        match self {
            Self::None => UArg::None,
            Self::Int(value) => UArg::Int(*value),
            Self::Scalar { dtype, bits } => UArg::Scalar {
                dtype: *dtype,
                bits: *bits,
            },
            Self::Name(name) => UArg::Name(name.to_owned()),
            Self::Variable { name, bounds } => UArg::Variable {
                name: name.to_owned(),
                bounds: bounds.clone(),
            },
            Self::Address {
                space,
                name,
                element,
            } => UArg::Address {
                space: *space,
                name: name.to_owned(),
                element: *element,
            },
            Self::RangeAxis(axis) => UArg::RangeAxis(*axis),
            Self::GepLane(lane) => UArg::GepLane(*lane),
            Self::BufferIndex {
                buffer,
                elements,
                input_shape,
                output_shape,
            } => UArg::BufferIndex {
                buffer: *buffer,
                elements: *elements,
                input_shape: input_shape.clone(),
                output_shape: output_shape.clone(),
            },
            Self::ViewBufferIndex {
                buffer,
                elements,
                input_shape,
                output_shape,
                view,
            } => UArg::ViewBufferIndex {
                buffer: *buffer,
                elements: *elements,
                input_shape: input_shape.clone(),
                output_shape: output_shape.clone(),
                view: view.clone(),
            },
            Self::Reduction {
                input_shape,
                output_shape,
                axes,
                keepdim,
                kind,
                mean,
            } => UArg::Reduction {
                input_shape: input_shape.clone(),
                output_shape: output_shape.clone(),
                axes: axes.to_vec(),
                keepdim: *keepdim,
                kind: *kind,
                mean: *mean,
            },
            Self::Matmul(plan) => UArg::Matmul(Box::new((*plan).clone())),
            Self::Conv2d(plan) => UArg::Conv2d(Box::new((*plan).clone())),
            Self::TiledMatmul(payload) => UArg::TiledMatmul(Box::new((*payload).clone())),
            Self::TensorCoreMatmul(payload) => UArg::TensorCoreMatmul(Box::new((*payload).clone())),
            Self::QuantizedMatmul(plan) => UArg::QuantizedMatmul(Box::new((*plan).clone())),
            Self::QuantizedRowGather(plan) => UArg::QuantizedRowGather(Box::new((*plan).clone())),
            Self::Movement(plan) => UArg::Movement(Box::new((*plan).clone())),
            Self::Random(plan) => UArg::Random(Box::new((*plan).clone())),
            Self::PrefixScan {
                input,
                input_shape,
                output_shape,
                axis,
                kind,
                output,
                dtype,
            } => UArg::PrefixScan {
                input: *input,
                input_shape: input_shape.clone(),
                output_shape: output_shape.clone(),
                axis: *axis,
                kind: *kind,
                output: *output,
                dtype: *dtype,
            },
            Self::Sort {
                input,
                input_shape,
                axis,
                descending,
                values,
                indices,
                dtype,
            } => UArg::Sort {
                input: *input,
                input_shape: input_shape.clone(),
                axis: *axis,
                descending: *descending,
                values: *values,
                indices: *indices,
                dtype: *dtype,
            },
            Self::TensorGuard {
                input,
                input_shape,
                axis,
                dtype,
            } => UArg::TensorGuard {
                input: *input,
                input_shape: input_shape.clone(),
                axis: *axis,
                dtype: *dtype,
            },
            Self::Effect(payload) => UArg::Effect(Box::new((*payload).clone())),
        }
    }

    pub(crate) fn equals_owned(self, other: &UArg) -> bool {
        match (self, other) {
            (Self::None, UArg::None) => true,
            (Self::Int(left), UArg::Int(right)) => left == right,
            (
                Self::Scalar {
                    dtype: left_dtype,
                    bits: left_bits,
                },
                UArg::Scalar {
                    dtype: right_dtype,
                    bits: right_bits,
                },
            ) => left_dtype == right_dtype && left_bits == right_bits,
            (Self::Name(left), UArg::Name(right)) => left == right,
            (
                Self::Variable {
                    name: left_name,
                    bounds: left_bounds,
                },
                UArg::Variable {
                    name: right_name,
                    bounds: right_bounds,
                },
            ) => left_name == right_name && left_bounds == right_bounds,
            (
                Self::Address {
                    space: left_space,
                    name: left_name,
                    element: left_element,
                },
                UArg::Address {
                    space: right_space,
                    name: right_name,
                    element: right_element,
                },
            ) => {
                left_space == right_space
                    && left_name == right_name
                    && left_element == right_element
            }
            (Self::RangeAxis(left), UArg::RangeAxis(right)) => left == right,
            (Self::GepLane(left), UArg::GepLane(right)) => left == right,
            (
                Self::BufferIndex {
                    buffer: left_buffer,
                    elements: left_elements,
                    input_shape: left_input,
                    output_shape: left_output,
                },
                UArg::BufferIndex {
                    buffer: right_buffer,
                    elements: right_elements,
                    input_shape: right_input,
                    output_shape: right_output,
                },
            ) => {
                left_buffer == right_buffer
                    && left_elements == right_elements
                    && left_input == right_input
                    && left_output == right_output
            }
            (
                Self::ViewBufferIndex {
                    buffer: left_buffer,
                    elements: left_elements,
                    input_shape: left_input,
                    output_shape: left_output,
                    view: left_view,
                },
                UArg::ViewBufferIndex {
                    buffer: right_buffer,
                    elements: right_elements,
                    input_shape: right_input,
                    output_shape: right_output,
                    view: right_view,
                },
            ) => {
                left_buffer == right_buffer
                    && left_elements == right_elements
                    && left_input == right_input
                    && left_output == right_output
                    && left_view == right_view
            }
            (
                Self::Reduction {
                    input_shape: left_input,
                    output_shape: left_output,
                    axes: left_axes,
                    keepdim: left_keepdim,
                    kind: left_kind,
                    mean: left_mean,
                },
                UArg::Reduction {
                    input_shape: right_input,
                    output_shape: right_output,
                    axes: right_axes,
                    keepdim: right_keepdim,
                    kind: right_kind,
                    mean: right_mean,
                },
            ) => {
                left_input == right_input
                    && left_output == right_output
                    && left_axes == right_axes
                    && left_keepdim == right_keepdim
                    && left_kind == right_kind
                    && left_mean == right_mean
            }
            (Self::Matmul(left), UArg::Matmul(right)) => left == right.as_ref(),
            (Self::Conv2d(left), UArg::Conv2d(right)) => left == right.as_ref(),
            (Self::TiledMatmul(left), UArg::TiledMatmul(right)) => left == right.as_ref(),
            (Self::TensorCoreMatmul(left), UArg::TensorCoreMatmul(right)) => left == right.as_ref(),
            (Self::QuantizedMatmul(left), UArg::QuantizedMatmul(right)) => left == right.as_ref(),
            (Self::QuantizedRowGather(left), UArg::QuantizedRowGather(right)) => {
                left == right.as_ref()
            }
            (Self::Movement(left), UArg::Movement(right)) => left == right.as_ref(),
            (Self::Random(left), UArg::Random(right)) => left == right.as_ref(),
            (
                Self::PrefixScan {
                    input: left_input,
                    input_shape: left_input_shape,
                    output_shape: left_output_shape,
                    axis: left_axis,
                    kind: left_kind,
                    output: left_output,
                    dtype: left_dtype,
                },
                UArg::PrefixScan {
                    input: right_input,
                    input_shape: right_input_shape,
                    output_shape: right_output_shape,
                    axis: right_axis,
                    kind: right_kind,
                    output: right_output,
                    dtype: right_dtype,
                },
            ) => {
                left_input == right_input
                    && left_input_shape == right_input_shape
                    && left_output_shape == right_output_shape
                    && left_axis == right_axis
                    && left_kind == right_kind
                    && left_output == right_output
                    && left_dtype == right_dtype
            }
            (
                Self::Sort {
                    input: left_input,
                    input_shape: left_shape,
                    axis: left_axis,
                    descending: left_descending,
                    values: left_values,
                    indices: left_indices,
                    dtype: left_dtype,
                },
                UArg::Sort {
                    input: right_input,
                    input_shape: right_shape,
                    axis: right_axis,
                    descending: right_descending,
                    values: right_values,
                    indices: right_indices,
                    dtype: right_dtype,
                },
            ) => {
                left_input == right_input
                    && left_shape == right_shape
                    && left_axis == right_axis
                    && left_descending == right_descending
                    && left_values == right_values
                    && left_indices == right_indices
                    && left_dtype == right_dtype
            }
            (
                Self::TensorGuard {
                    input: left_input,
                    input_shape: left_shape,
                    axis: left_axis,
                    dtype: left_dtype,
                },
                UArg::TensorGuard {
                    input: right_input,
                    input_shape: right_shape,
                    axis: right_axis,
                    dtype: right_dtype,
                },
            ) => {
                left_input == right_input
                    && left_shape == right_shape
                    && left_axis == right_axis
                    && left_dtype == right_dtype
            }
            (Self::Effect(left), UArg::Effect(right)) => left == right.as_ref(),
            _ => false,
        }
    }

    pub(crate) fn matmul_plan(self) -> Option<&'a crate::MatmulKernelPlan> {
        match self {
            Self::Matmul(plan) => Some(plan),
            Self::TiledMatmul(payload) => Some(&payload.matmul),
            Self::TensorCoreMatmul(payload) => Some(&payload.matmul),
            _ => None,
        }
    }
    pub(crate) fn quantized_matmul_plan(self) -> Option<&'a crate::QuantizedMatmulPlan> {
        match self {
            Self::QuantizedMatmul(plan) => Some(plan),
            _ => None,
        }
    }
    pub(crate) fn static_conv2d_plan(self) -> Option<&'a crate::StaticConv2dPlan> {
        match self {
            Self::Conv2d(plan) => Some(plan),
            _ => None,
        }
    }
    pub(crate) fn quantized_row_gather_plan(self) -> Option<&'a crate::QuantizedRowGatherPlan> {
        match self {
            Self::QuantizedRowGather(plan) => Some(plan),
            _ => None,
        }
    }
}

fn literal_ref(value: &LiteralValue) -> UArgRef<'_> {
    match value {
        LiteralValue::Int(value) => UArgRef::Int(value),
        LiteralValue::Scalar { dtype, bits } => UArgRef::Scalar { dtype, bits },
    }
}

fn address_ref(value: &AddressValue) -> UArgRef<'_> {
    UArgRef::Address {
        space: &value.space,
        name: &value.name,
        element: &value.element,
    }
}

fn matmul_ref(value: &MatmulValue) -> UArgRef<'_> {
    match value {
        MatmulValue::Serial(plan) => UArgRef::Matmul(plan),
        MatmulValue::Tiled(plan) => UArgRef::TiledMatmul(plan),
        MatmulValue::TensorCore(plan) => UArgRef::TensorCoreMatmul(plan),
        MatmulValue::Quantized(plan) => UArgRef::QuantizedMatmul(plan),
    }
}

fn movement_ref(value: &MovementValue) -> UArgRef<'_> {
    match value {
        MovementValue::Plan(plan) => UArgRef::Movement(plan),
        MovementValue::QuantizedRowGather(plan) => UArgRef::QuantizedRowGather(plan),
    }
}

fn index_ref(value: &IndexValue) -> UArgRef<'_> {
    match value {
        IndexValue::Buffer {
            buffer,
            elements,
            input_shape,
            output_shape,
        } => UArgRef::BufferIndex {
            buffer,
            elements,
            input_shape,
            output_shape,
        },
        IndexValue::View {
            buffer,
            elements,
            input_shape,
            output_shape,
            view,
        } => UArgRef::ViewBufferIndex {
            buffer,
            elements,
            input_shape,
            output_shape,
            view,
        },
    }
}

macro_rules! define_uops {
    (
        $(
            $(#[$meta:meta])*
            $variant:ident $(($payload:ty as $binding:ident))?
                => $kind_variant:ident $(($kind_ty:ty) { $kind_expr:expr })?
                => $spec:expr
                => $argument:expr
                => legacy { $($legacy_pattern:pat => $legacy_value:expr;)+ };
        )+
    ) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub enum Operation {
            $($(#[$meta])* $variant $(($payload))?,)+
        }

        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub enum UOpKind {
            $($(#[$meta])* $kind_variant $(($kind_ty))?,)+
        }

        impl Operation {
            pub fn kind(&self) -> UOpKind {
                match self {
                    $(
                        Self::$variant $(($binding))? => {
                            $(let _ = $binding;)?
                            UOpKind::$kind_variant $(($kind_expr))?
                        }
                    )+
                }
            }

            pub(crate) fn signature(&self) -> spec::OpSignature {
                use spec::OpSpec;
                match self {
                    $(
                        Self::$variant $(($binding))? => {
                            $(let _ = $binding;)?
                            ($spec).signature()
                        }
                    )+
                }
            }

            pub(crate) fn argument(&self) -> UArgRef<'_> {
                match self {
                    $(
                        Self::$variant $(($binding))? => {
                            $(let _ = $binding;)?
                            $argument
                        }
                    )+
                }
            }

            pub(crate) fn from_legacy(kind: UOpKind, arg: UArg) -> Result<Self, UOpError> {
                Ok(match (kind, arg) {
                    $($($legacy_pattern => $legacy_value,)+)+
                    _ => return Err(UOpError::InvalidArgument),
                })
            }
        }
    };
}

define_uops! {
    Const(LiteralValue as value) => Const => spec::Literal => literal_ref(value) => legacy {
        (UOpKind::Const, UArg::Int(value)) => Self::Const(LiteralValue::Int(value));
        (UOpKind::Const, UArg::Scalar { dtype, bits }) => Self::Const(LiteralValue::Scalar { dtype, bits });
    };
    VConst(LiteralValue as value) => VConst => spec::Literal => literal_ref(value) => legacy {
        (UOpKind::VConst, UArg::Int(value)) => Self::VConst(LiteralValue::Int(value));
        (UOpKind::VConst, UArg::Scalar { dtype, bits }) => Self::VConst(LiteralValue::Scalar { dtype, bits });
    };
    DefineVar(VariableValue as value) => DefineVar => spec::Definition::Variable => UArgRef::Variable { name: &value.name, bounds: &value.bounds } => legacy {
        (UOpKind::DefineVar, UArg::Variable { name, bounds }) => Self::DefineVar(VariableValue { name, bounds });
    };
    DefineGlobal(AddressValue as value) => DefineGlobal => spec::Definition::Address => address_ref(value) => legacy {
        (UOpKind::DefineGlobal, UArg::Address { space, name, element }) => Self::DefineGlobal(AddressValue { space, name, element });
    };
    DefineLocal(AddressValue as value) => DefineLocal => spec::Definition::Address => address_ref(value) => legacy {
        (UOpKind::DefineLocal, UArg::Address { space, name, element }) => Self::DefineLocal(AddressValue { space, name, element });
    };
    DefineRegister(AddressValue as value) => DefineRegister => spec::Definition::Address => address_ref(value) => legacy {
        (UOpKind::DefineRegister, UArg::Address { space, name, element }) => Self::DefineRegister(AddressValue { space, name, element });
    };
    Special(String as value) => Special => spec::Definition::Name => UArgRef::Name(value) => legacy {
        (UOpKind::Special, UArg::Name(name)) => Self::Special(name);
    };
    Range(u32 as axis) => Range => spec::Control::Range => UArgRef::RangeAxis(axis) => legacy {
        (UOpKind::Range, UArg::RangeAxis(axis)) => Self::Range(axis);
    };
    EndRange => EndRange => spec::Control::Boundary => UArgRef::None => legacy {
        (UOpKind::EndRange, UArg::None) => Self::EndRange;
    };
    If => If => spec::Control::Boundary => UArgRef::None => legacy {
        (UOpKind::If, UArg::None) => Self::If;
    };
    EndIf => EndIf => spec::Control::Boundary => UArgRef::None => legacy {
        (UOpKind::EndIf, UArg::None) => Self::EndIf;
    };
    Unary(super::Unary as value) => Unary(super::Unary) { *value } => spec::CoreAlu::Unary => UArgRef::None => legacy {
        (UOpKind::Unary(op), UArg::None) => Self::Unary(op);
    };
    Binary(Binary as value) => Binary(Binary) { *value } => spec::CoreAlu::Binary => UArgRef::None => legacy {
        (UOpKind::Binary(op), UArg::None) => Self::Binary(op);
    };
    /// High-level ALU tags retained by the portable interpreter. Renderers may
    /// lower these to the smaller core ALU vocabulary later.
    GraphUnary(crate::UnaryOp as op) => GraphUnary(crate::UnaryOp) { *op } => spec::GraphAlu::Unary => UArgRef::None => legacy {
        (UOpKind::GraphUnary(op), UArg::None) => Self::GraphUnary(op);
    };
    GraphBinary(crate::BinaryOp as op) => GraphBinary(crate::BinaryOp) { *op } => spec::GraphAlu::Binary => UArgRef::None => legacy {
        (UOpKind::GraphBinary(op), UArg::None) => Self::GraphBinary(op);
    };
    GraphCompare(crate::CompareOp as op) => GraphCompare(crate::CompareOp) { *op } => spec::GraphAlu::Binary => UArgRef::None => legacy {
        (UOpKind::GraphCompare(op), UArg::None) => Self::GraphCompare(op);
    };
    GraphLogical(crate::LogicalOp as op) => GraphLogical(crate::LogicalOp) { *op } => spec::GraphAlu::Logical(*op) => UArgRef::None => legacy {
        (UOpKind::GraphLogical(op), UArg::None) => Self::GraphLogical(op);
    };
    /// Complete static generalized-matmul semantic. Its typed payload owns the
    /// lhs/rhs/output ABI and normalized contraction geometry.
    Matmul(MatmulValue as value) => Matmul => spec::Materialized::Matmul => matmul_ref(value) => legacy {
        (UOpKind::Matmul, UArg::Matmul(plan)) => Self::Matmul(MatmulValue::Serial(plan));
        (UOpKind::Matmul, UArg::TiledMatmul(plan)) => Self::Matmul(MatmulValue::Tiled(plan));
        (UOpKind::Matmul, UArg::TensorCoreMatmul(plan)) => Self::Matmul(MatmulValue::TensorCore(plan));
        (UOpKind::Matmul, UArg::QuantizedMatmul(plan)) => Self::Matmul(MatmulValue::Quantized(plan));
    };
    /// Narrow static F32 NCHW 1x1 convolution semantic. The payload owns the
    /// exact ordered input/weight/bias/output ABI and rejects all other Conv2d
    /// geometries before a renderer sees them.
    Conv2d(Box<crate::StaticConv2dPlan> as value) => Conv2d => spec::Materialized::Conv2d => UArgRef::Conv2d(value) => legacy {
        (UOpKind::Conv2d, UArg::Conv2d(plan)) => Self::Conv2d(plan);
    };
    /// Complete materializing concat/gather/scatter semantic and ordered ABI.
    Movement(MovementValue as value) => Movement => spec::Materialized::Movement => movement_ref(value) => legacy {
        (UOpKind::Movement, UArg::Movement(plan)) => Self::Movement(MovementValue::Plan(plan));
        (UOpKind::Movement, UArg::QuantizedRowGather(plan)) => Self::Movement(MovementValue::QuantizedRowGather(plan));
    };
    /// Captured random source semantic. The payload owns its stream
    /// reservation, so execution is independent of mutable graph RNG state.
    Random(Box<crate::random::plan::RandomKernelPlan> as value) => Random => spec::Materialized::Random => UArgRef::Random(value) => legacy {
        (UOpKind::Random, UArg::Random(plan)) => Self::Random(plan);
    };
    /// Static inclusive prefix scan. The payload owns normalized axis and the
    /// exact input/output ABI; optimized renderers reject it until lowered.
    PrefixScan(PrefixScanValue as value) => PrefixScan => spec::Materialized::PrefixScan => UArgRef::PrefixScan { input: &value.input, input_shape: &value.input_shape, output_shape: &value.output_shape, axis: &value.axis, kind: &value.kind, output: &value.output, dtype: &value.dtype } => legacy {
        (UOpKind::PrefixScan, UArg::PrefixScan { input, input_shape, output_shape, axis, kind, output, dtype }) => Self::PrefixScan(PrefixScanValue { input, input_shape, output_shape, axis, kind, output, dtype });
    };
    /// Stable CPU-static ordering with one values and one I32-index output.
    /// The paired buffer IDs live in the typed payload; renderers reject it
    /// until they implement the coupled output ABI.
    Sort(SortValue as value) => Sort => spec::Materialized::Sort => UArgRef::Sort { input: &value.input, input_shape: &value.input_shape, axis: &value.axis, descending: &value.descending, values: &value.values, indices: &value.indices, dtype: &value.dtype } => legacy {
        (UOpKind::Sort, UArg::Sort { input, input_shape, axis, descending, values, indices, dtype }) => Self::Sort(SortValue { input, input_shape, axis, descending, values, indices, dtype });
    };
    /// Value-preserving CPU-static distribution validation boundary.
    TensorGuard(TensorGuardValue as value) => TensorGuard => spec::Materialized::TensorGuard => UArgRef::TensorGuard { input: &value.input, input_shape: &value.input_shape, axis: &value.axis, dtype: &value.dtype } => legacy {
        (UOpKind::TensorGuard, UArg::TensorGuard { input, input_shape, axis, dtype }) => Self::TensorGuard(TensorGuardValue { input, input_shape, axis, dtype });
    };
    ReduceInit(ReductionValue as value) => ReduceInit => spec::Reduction::Init => UArgRef::Reduction { input_shape: &value.input_shape, output_shape: &value.output_shape, axes: &value.axes, keepdim: &value.keepdim, kind: &value.kind, mean: &value.mean } => legacy {
        (UOpKind::ReduceInit, UArg::Reduction { input_shape, output_shape, axes, keepdim, kind, mean }) => Self::ReduceInit(ReductionValue { input_shape, output_shape, axes, keepdim, kind, mean });
    };
    ReduceAccumulate => ReduceAccumulate => spec::Reduction::Accumulate => UArgRef::None => legacy {
        (UOpKind::ReduceAccumulate, UArg::None) => Self::ReduceAccumulate;
    };
    ReduceFinalize => ReduceFinalize => spec::Reduction::Finalize => UArgRef::None => legacy {
        (UOpKind::ReduceFinalize, UArg::None) => Self::ReduceFinalize;
    };
    Ternary(Ternary as op) => Ternary(Ternary) { *op } => spec::CoreAlu::Ternary => UArgRef::None => legacy {
        (UOpKind::Ternary(op), UArg::None) => Self::Ternary(op);
    };
    Cast => Cast => spec::Conversion::Cast => UArgRef::None => legacy {
        (UOpKind::Cast, UArg::None) => Self::Cast;
    };
    Bitcast => Bitcast => spec::Conversion::Bitcast => UArgRef::None => legacy {
        (UOpKind::Bitcast, UArg::None) => Self::Bitcast;
    };
    Vectorize => Vectorize => spec::Conversion::Vectorize => UArgRef::None => legacy {
        (UOpKind::Vectorize, UArg::None) => Self::Vectorize;
    };
    Gep(u16 as lane) => Gep => spec::Conversion::Gep => UArgRef::GepLane(lane) => legacy {
        (UOpKind::Gep, UArg::GepLane(lane)) => Self::Gep(lane);
    };
    Index(IndexValue as value) => Index => spec::Memory::Index => index_ref(value) => legacy {
        (UOpKind::Index, UArg::BufferIndex { buffer, elements, input_shape, output_shape }) => Self::Index(IndexValue::Buffer { buffer, elements, input_shape, output_shape });
        (UOpKind::Index, UArg::ViewBufferIndex { buffer, elements, input_shape, output_shape, view }) => Self::Index(IndexValue::View { buffer, elements, input_shape, output_shape, view });
    };
    Load => Load => spec::Memory::Load => UArgRef::None => legacy {
        (UOpKind::Load, UArg::None) => Self::Load;
    };
    Store => Store => spec::Memory::Store => UArgRef::None => legacy {
        (UOpKind::Store, UArg::None) => Self::Store;
    };
    /// Immutable graph-adjacent assignment commit; never a pure kernel store.
    EffectStore(Box<crate::EffectPayload> as value) => EffectStore => spec::Effect::Store => UArgRef::Effect(value) => legacy {
        (UOpKind::EffectStore, UArg::Effect(value)) => Self::EffectStore(value);
    };
    /// Orders an effect store after explicitly named predecessor effect IDs.
    After(Box<crate::EffectPayload> as value) => After => spec::Effect::After => UArgRef::Effect(value) => legacy {
        (UOpKind::After, UArg::Effect(value)) => Self::After(value);
    };
    Barrier => Barrier => spec::Effect::Barrier => UArgRef::None => legacy {
        (UOpKind::Barrier, UArg::None) => Self::Barrier;
    };
    Sink => Sink => spec::Effect::Sink => UArgRef::None => legacy {
        (UOpKind::Sink, UArg::None) => Self::Sink;
    };
}
