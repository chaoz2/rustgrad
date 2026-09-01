//! Bounded portable node-table encoding for validated UOps.
use super::{
    AddressSpace, AddressValue, AffineView, Binary, IndexValue, LiteralValue, MatmulValue,
    MovementValue, Operation, PrefixScanValue, ReductionValue, SortValue, TensorGuardValue,
    ThreefryValue, UOp, UType, Unary, VariableValue, ViewMap, graph_unary_type_is_valid,
};
use crate::{
    BinaryOp, CompareOp, DType, GgmlType, LogicalOp, MatmulBarrierKind, MatmulBarrierPhase,
    MatmulKernelPlan, MatmulResourceEstimate, MatmulTargetCaps, MmaFragmentLayout, MmaInstruction,
    MovementKernelKind, MovementKernelPlan, MovementOperand, NodeId, QuantizedBufferDesc,
    QuantizedMatmulOrientation, QuantizedMatmulPlan, QuantizedRowGatherPlan, RandomKind,
    ReduceKind, Shape, SharedTileLayout, StaticConv2dPlan, SymbolicExpr, TensorCoreMatmulPayload,
    TensorCoreMatmulPlan, TensorCoreOutputPolicy, TensorCoreTailPolicy, TiledMatmulPayload,
    TiledMatmulPlan, TiledMatmulTails, UnaryOp,
};
use std::{collections::BTreeMap, fmt};

const MAGIC: &[u8; 4] = b"RGUA";
/// v14 adds the prefix-scan output selector; v16 adds the coupled Sort pair;
/// v17 adds the CPU-static TensorGuard validation boundary; v18 makes the
/// prefix-scan source/result buffer ABI and source dtype self-contained.
/// v19 adds the dependency-bearing live packed-U64 Threefry plan; v20 adds
/// the checked static-position movement payload.
/// v15 is retained as the first internal mixed-schedule envelope.
const VERSION: u8 = 20;
const PREVIOUS_EFFECT_VERSION: u8 = 15;
const LEGACY_EFFECT_VERSION: u8 = 13;
const MAX_BYTES: usize = 64 << 20;
const MAX_NODES: usize = 1 << 20;
const MAX_SOURCES: usize = 1 << 20;
const MAX_COLLECTION: usize = 1 << 20;
const MAX_STRING: usize = 1 << 20;
const MAX_SYMBOLIC_DEPTH: usize = 256;

/// Private RGUA opcode vocabulary. It exists only while encoding or decoding
/// stable wire tags and is never stored in a UOp node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WireOpcode {
    Const,
    VConst,
    DefineVar,
    DefineGlobal,
    DefineLocal,
    DefineRegister,
    Special,
    Range,
    EndRange,
    If,
    EndIf,
    Unary(Unary),
    Binary(Binary),
    GraphUnary(UnaryOp),
    GraphBinary(BinaryOp),
    GraphCompare(CompareOp),
    GraphLogical(LogicalOp),
    Threefry,
    Matmul,
    Conv2d,
    Movement,
    Random,
    PrefixScan,
    Sort,
    TensorGuard,
    ReduceInit,
    ReduceAccumulate,
    ReduceFinalize,
    Ternary(super::Ternary),
    Cast,
    Bitcast,
    Vectorize,
    Gep,
    Index,
    Load,
    Store,
    EffectStore,
    After,
    Barrier,
    Sink,
}

/// Private owned payload used only by the stable RGUA codec. Runtime UOps
/// store the corresponding typed [`Operation`] variant directly.
#[derive(Clone, Debug, Eq, PartialEq)]
enum WireArg {
    None,
    Int(i64),
    Scalar {
        dtype: DType,
        bits: u64,
    },
    Name(String),
    Variable {
        name: String,
        bounds: SymbolicExpr,
    },
    Address {
        space: AddressSpace,
        name: String,
        element: UType,
    },
    RangeAxis(u32),
    GepLane(u16),
    BufferIndex {
        buffer: u64,
        elements: usize,
        input_shape: Shape,
        output_shape: Shape,
    },
    ViewBufferIndex {
        buffer: u64,
        elements: usize,
        input_shape: Shape,
        output_shape: Shape,
        view: AffineView,
    },
    Reduction {
        input_shape: Shape,
        output_shape: Shape,
        axes: Vec<usize>,
        keepdim: bool,
        kind: ReduceKind,
        mean: bool,
    },
    Matmul(Box<MatmulKernelPlan>),
    Conv2d(Box<StaticConv2dPlan>),
    TiledMatmul(Box<TiledMatmulPayload>),
    TensorCoreMatmul(Box<TensorCoreMatmulPayload>),
    QuantizedMatmul(Box<QuantizedMatmulPlan>),
    QuantizedRowGather(Box<QuantizedRowGatherPlan>),
    Movement(Box<MovementKernelPlan>),
    Random(Box<crate::random::plan::RandomKernelPlan>),
    PrefixScan {
        input: NodeId,
        destination: NodeId,
        input_shape: Shape,
        output_shape: Shape,
        axis: usize,
        kind: crate::PrefixScanKind,
        output: crate::PrefixScanOutput,
        input_dtype: DType,
        dtype: DType,
    },
    Sort {
        input: NodeId,
        input_shape: Shape,
        axis: usize,
        descending: bool,
        values: NodeId,
        indices: NodeId,
        dtype: DType,
    },
    TensorGuard {
        input: NodeId,
        input_shape: Shape,
        axis: usize,
        dtype: DType,
    },
    Threefry {
        counter: NodeId,
        key: NodeId,
        counter_shape: Shape,
        key_shape: Shape,
        output: NodeId,
        output_shape: Shape,
    },
    Effect(Box<crate::EffectPayload>),
}

fn operation_to_wire(operation: &Operation) -> (WireOpcode, WireArg) {
    match operation {
        Operation::Const(LiteralValue::Int(value)) => (WireOpcode::Const, WireArg::Int(*value)),
        Operation::Const(LiteralValue::Scalar { dtype, bits }) => (
            WireOpcode::Const,
            WireArg::Scalar {
                dtype: *dtype,
                bits: *bits,
            },
        ),
        Operation::VConst(LiteralValue::Int(value)) => (WireOpcode::VConst, WireArg::Int(*value)),
        Operation::VConst(LiteralValue::Scalar { dtype, bits }) => (
            WireOpcode::VConst,
            WireArg::Scalar {
                dtype: *dtype,
                bits: *bits,
            },
        ),
        Operation::DefineVar(value) => (
            WireOpcode::DefineVar,
            WireArg::Variable {
                name: value.name.clone(),
                bounds: value.bounds.clone(),
            },
        ),
        Operation::DefineGlobal(value) => address_to_wire(WireOpcode::DefineGlobal, value),
        Operation::DefineLocal(value) => address_to_wire(WireOpcode::DefineLocal, value),
        Operation::DefineRegister(value) => address_to_wire(WireOpcode::DefineRegister, value),
        Operation::Special(name) => (WireOpcode::Special, WireArg::Name(name.clone())),
        Operation::Range(axis) => (WireOpcode::Range, WireArg::RangeAxis(*axis)),
        Operation::EndRange => (WireOpcode::EndRange, WireArg::None),
        Operation::If => (WireOpcode::If, WireArg::None),
        Operation::EndIf => (WireOpcode::EndIf, WireArg::None),
        Operation::Unary(op) => (WireOpcode::Unary(*op), WireArg::None),
        Operation::Binary(op) => (WireOpcode::Binary(*op), WireArg::None),
        Operation::GraphUnary(op) => (WireOpcode::GraphUnary(*op), WireArg::None),
        Operation::GraphBinary(op) => (WireOpcode::GraphBinary(*op), WireArg::None),
        Operation::GraphCompare(op) => (WireOpcode::GraphCompare(*op), WireArg::None),
        Operation::GraphLogical(op) => (WireOpcode::GraphLogical(*op), WireArg::None),
        Operation::Threefry(value) => (
            WireOpcode::Threefry,
            WireArg::Threefry {
                counter: value.counter,
                key: value.key,
                counter_shape: value.counter_shape.clone(),
                key_shape: value.key_shape.clone(),
                output: value.output,
                output_shape: value.output_shape.clone(),
            },
        ),
        Operation::Matmul(MatmulValue::Serial(plan)) => {
            (WireOpcode::Matmul, WireArg::Matmul(plan.clone()))
        }
        Operation::Matmul(MatmulValue::Tiled(plan)) => {
            (WireOpcode::Matmul, WireArg::TiledMatmul(plan.clone()))
        }
        Operation::Matmul(MatmulValue::TensorCore(plan)) => {
            (WireOpcode::Matmul, WireArg::TensorCoreMatmul(plan.clone()))
        }
        Operation::Matmul(MatmulValue::Quantized(plan)) => {
            (WireOpcode::Matmul, WireArg::QuantizedMatmul(plan.clone()))
        }
        Operation::Conv2d(plan) => (WireOpcode::Conv2d, WireArg::Conv2d(plan.clone())),
        Operation::Movement(MovementValue::Plan(plan)) => {
            (WireOpcode::Movement, WireArg::Movement(plan.clone()))
        }
        Operation::Movement(MovementValue::QuantizedRowGather(plan)) => (
            WireOpcode::Movement,
            WireArg::QuantizedRowGather(plan.clone()),
        ),
        Operation::Random(plan) => (WireOpcode::Random, WireArg::Random(plan.clone())),
        Operation::PrefixScan(value) => (
            WireOpcode::PrefixScan,
            WireArg::PrefixScan {
                input: value.input,
                destination: value.destination,
                input_shape: value.input_shape.clone(),
                output_shape: value.output_shape.clone(),
                axis: value.axis,
                kind: value.kind,
                output: value.output,
                input_dtype: value.input_dtype,
                dtype: value.dtype,
            },
        ),
        Operation::Sort(value) => (
            WireOpcode::Sort,
            WireArg::Sort {
                input: value.input,
                input_shape: value.input_shape.clone(),
                axis: value.axis,
                descending: value.descending,
                values: value.values,
                indices: value.indices,
                dtype: value.dtype,
            },
        ),
        Operation::TensorGuard(value) => (
            WireOpcode::TensorGuard,
            WireArg::TensorGuard {
                input: value.input,
                input_shape: value.input_shape.clone(),
                axis: value.axis,
                dtype: value.dtype,
            },
        ),
        Operation::ReduceInit(value) => (
            WireOpcode::ReduceInit,
            WireArg::Reduction {
                input_shape: value.input_shape.clone(),
                output_shape: value.output_shape.clone(),
                axes: value.axes.clone(),
                keepdim: value.keepdim,
                kind: value.kind,
                mean: matches!(value.kind, crate::ReduceKind::Mean),
            },
        ),
        Operation::ReduceAccumulate => (WireOpcode::ReduceAccumulate, WireArg::None),
        Operation::ReduceFinalize => (WireOpcode::ReduceFinalize, WireArg::None),
        Operation::Ternary(op) => (WireOpcode::Ternary(*op), WireArg::None),
        Operation::Cast => (WireOpcode::Cast, WireArg::None),
        Operation::Bitcast => (WireOpcode::Bitcast, WireArg::None),
        Operation::Vectorize => (WireOpcode::Vectorize, WireArg::None),
        Operation::Gep(lane) => (WireOpcode::Gep, WireArg::GepLane(*lane)),
        Operation::Index(IndexValue::Buffer {
            buffer,
            elements,
            input_shape,
            output_shape,
        }) => (
            WireOpcode::Index,
            WireArg::BufferIndex {
                buffer: *buffer,
                elements: *elements,
                input_shape: input_shape.clone(),
                output_shape: output_shape.clone(),
            },
        ),
        Operation::Index(IndexValue::View {
            buffer,
            elements,
            input_shape,
            output_shape,
            view,
        }) => (
            WireOpcode::Index,
            WireArg::ViewBufferIndex {
                buffer: *buffer,
                elements: *elements,
                input_shape: input_shape.clone(),
                output_shape: output_shape.clone(),
                view: view.clone(),
            },
        ),
        Operation::Load => (WireOpcode::Load, WireArg::None),
        Operation::Store => (WireOpcode::Store, WireArg::None),
        Operation::EffectStore(payload) => {
            (WireOpcode::EffectStore, WireArg::Effect(payload.clone()))
        }
        Operation::After(payload) => (WireOpcode::After, WireArg::Effect(payload.clone())),
        Operation::Barrier => (WireOpcode::Barrier, WireArg::None),
        Operation::Sink => (WireOpcode::Sink, WireArg::None),
    }
}

fn address_to_wire(opcode: WireOpcode, value: &AddressValue) -> (WireOpcode, WireArg) {
    (
        opcode,
        WireArg::Address {
            space: value.space,
            name: value.name.clone(),
            element: value.element,
        },
    )
}

fn operation_from_wire(opcode: WireOpcode, arg: WireArg) -> Result<Operation, ArtifactError> {
    Ok(match (opcode, arg) {
        (WireOpcode::Const, WireArg::Int(value)) => Operation::Const(LiteralValue::Int(value)),
        (WireOpcode::Const, WireArg::Scalar { dtype, bits }) => {
            Operation::Const(LiteralValue::Scalar { dtype, bits })
        }
        (WireOpcode::VConst, WireArg::Int(value)) => Operation::VConst(LiteralValue::Int(value)),
        (WireOpcode::VConst, WireArg::Scalar { dtype, bits }) => {
            Operation::VConst(LiteralValue::Scalar { dtype, bits })
        }
        (WireOpcode::DefineVar, WireArg::Variable { name, bounds }) => {
            Operation::DefineVar(VariableValue { name, bounds })
        }
        (
            WireOpcode::DefineGlobal,
            WireArg::Address {
                space,
                name,
                element,
            },
        ) => Operation::DefineGlobal(AddressValue {
            space,
            name,
            element,
        }),
        (
            WireOpcode::DefineLocal,
            WireArg::Address {
                space,
                name,
                element,
            },
        ) => Operation::DefineLocal(AddressValue {
            space,
            name,
            element,
        }),
        (
            WireOpcode::DefineRegister,
            WireArg::Address {
                space,
                name,
                element,
            },
        ) => Operation::DefineRegister(AddressValue {
            space,
            name,
            element,
        }),
        (WireOpcode::Special, WireArg::Name(name)) => Operation::Special(name),
        (WireOpcode::Range, WireArg::RangeAxis(axis)) => Operation::Range(axis),
        (WireOpcode::EndRange, WireArg::None) => Operation::EndRange,
        (WireOpcode::If, WireArg::None) => Operation::If,
        (WireOpcode::EndIf, WireArg::None) => Operation::EndIf,
        (WireOpcode::Unary(op), WireArg::None) => Operation::Unary(op),
        (WireOpcode::Binary(op), WireArg::None) => Operation::Binary(op),
        (WireOpcode::GraphUnary(op), WireArg::None) => Operation::GraphUnary(op),
        (WireOpcode::GraphBinary(op), WireArg::None) => Operation::GraphBinary(op),
        (WireOpcode::GraphCompare(op), WireArg::None) => Operation::GraphCompare(op),
        (WireOpcode::GraphLogical(op), WireArg::None) => Operation::GraphLogical(op),
        (
            WireOpcode::Threefry,
            WireArg::Threefry {
                counter,
                key,
                counter_shape,
                key_shape,
                output,
                output_shape,
            },
        ) => Operation::Threefry(ThreefryValue {
            counter,
            key,
            counter_shape,
            key_shape,
            output,
            output_shape,
        }),
        (WireOpcode::Matmul, WireArg::Matmul(plan)) => Operation::Matmul(MatmulValue::Serial(plan)),
        (WireOpcode::Matmul, WireArg::TiledMatmul(plan)) => {
            Operation::Matmul(MatmulValue::Tiled(plan))
        }
        (WireOpcode::Matmul, WireArg::TensorCoreMatmul(plan)) => {
            Operation::Matmul(MatmulValue::TensorCore(plan))
        }
        (WireOpcode::Matmul, WireArg::QuantizedMatmul(plan)) => {
            Operation::Matmul(MatmulValue::Quantized(plan))
        }
        (WireOpcode::Conv2d, WireArg::Conv2d(plan)) => Operation::Conv2d(plan),
        (WireOpcode::Movement, WireArg::Movement(plan)) => {
            Operation::Movement(MovementValue::Plan(plan))
        }
        (WireOpcode::Movement, WireArg::QuantizedRowGather(plan)) => {
            Operation::Movement(MovementValue::QuantizedRowGather(plan))
        }
        (WireOpcode::Random, WireArg::Random(plan)) => Operation::Random(plan),
        (
            WireOpcode::PrefixScan,
            WireArg::PrefixScan {
                input,
                destination,
                input_shape,
                output_shape,
                axis,
                kind,
                output,
                input_dtype,
                dtype,
            },
        ) => Operation::PrefixScan(PrefixScanValue {
            input,
            destination,
            input_shape,
            output_shape,
            axis,
            kind,
            output,
            input_dtype,
            dtype,
        }),
        (
            WireOpcode::Sort,
            WireArg::Sort {
                input,
                input_shape,
                axis,
                descending,
                values,
                indices,
                dtype,
            },
        ) => Operation::Sort(SortValue {
            input,
            input_shape,
            axis,
            descending,
            values,
            indices,
            dtype,
        }),
        (
            WireOpcode::TensorGuard,
            WireArg::TensorGuard {
                input,
                input_shape,
                axis,
                dtype,
            },
        ) => Operation::TensorGuard(TensorGuardValue {
            input,
            input_shape,
            axis,
            dtype,
        }),
        (
            WireOpcode::ReduceInit,
            WireArg::Reduction {
                input_shape,
                output_shape,
                axes,
                keepdim,
                kind,
                mean,
            },
        ) => Operation::ReduceInit(ReductionValue {
            // `mean` is a historical redundant wire bit. Preserve decoding
            // while refusing artifacts whose projection disagrees with the
            // canonical kind before a UOp can enter scheduling or caching.
            kind: if mean == matches!(kind, crate::ReduceKind::Mean) {
                kind
            } else {
                return Err(ArtifactError::Format("reduction mean projection"));
            },
            input_shape,
            output_shape,
            axes,
            keepdim,
        }),
        (WireOpcode::ReduceAccumulate, WireArg::None) => Operation::ReduceAccumulate,
        (WireOpcode::ReduceFinalize, WireArg::None) => Operation::ReduceFinalize,
        (WireOpcode::Ternary(op), WireArg::None) => Operation::Ternary(op),
        (WireOpcode::Cast, WireArg::None) => Operation::Cast,
        (WireOpcode::Bitcast, WireArg::None) => Operation::Bitcast,
        (WireOpcode::Vectorize, WireArg::None) => Operation::Vectorize,
        (WireOpcode::Gep, WireArg::GepLane(lane)) => Operation::Gep(lane),
        (
            WireOpcode::Index,
            WireArg::BufferIndex {
                buffer,
                elements,
                input_shape,
                output_shape,
            },
        ) => Operation::Index(IndexValue::Buffer {
            buffer,
            elements,
            input_shape,
            output_shape,
        }),
        (
            WireOpcode::Index,
            WireArg::ViewBufferIndex {
                buffer,
                elements,
                input_shape,
                output_shape,
                view,
            },
        ) => Operation::Index(IndexValue::View {
            buffer,
            elements,
            input_shape,
            output_shape,
            view,
        }),
        (WireOpcode::Load, WireArg::None) => Operation::Load,
        (WireOpcode::Store, WireArg::None) => Operation::Store,
        (WireOpcode::EffectStore, WireArg::Effect(payload)) => Operation::EffectStore(payload),
        (WireOpcode::After, WireArg::Effect(payload)) => Operation::After(payload),
        (WireOpcode::Barrier, WireArg::None) => Operation::Barrier,
        (WireOpcode::Sink, WireArg::None) => Operation::Sink,
        _ => return Err(ArtifactError::Format("kind argument")),
    })
}

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

/// Encodes one immutable UOp DAG. Node IDs are dense topological indices and
/// repeated source IDs preserve shared subgraphs. Historical operation sets
/// retain the released v19 standalone envelope; only static-position movement
/// requires v20.
pub fn encode(root: &UOp) -> Result<Vec<u8>, ArtifactError> {
    let nodes = root
        .topological()
        .map_err(|_| ArtifactError::Format("dag"))?;
    let version = if nodes.iter().any(|node| {
        matches!(
            node.operation(),
            Operation::Movement(MovementValue::Plan(plan))
                if matches!(&plan.kind, MovementKernelKind::ScatterPositions { .. })
        )
    }) {
        VERSION
    } else {
        // v20 adds no representation for historical operations. Keep their
        // standalone canonical bytes on the released v19 writer envelope.
        19
    };
    encode_inner(root, false, version)
}

/// Encodes the semantic kernel blob used by durable schedule identities.
/// Existing operations remain pinned to the released v18 envelope so adding
/// a later RGUA operation cannot silently invalidate every RGSA/RGSO/RGSM
/// cache key. Threefry requires the v19 envelope and static-position movement
/// requires v20; corrected reduction code generation is separated by
/// renderer-specific source/cache versions without changing its released
/// logical UOp identity.
pub(crate) fn encode_schedule_identity(root: &UOp) -> Result<Vec<u8>, ArtifactError> {
    let nodes = root
        .topological()
        .map_err(|_| ArtifactError::Format("dag"))?;
    let effects = nodes.iter().any(|node| {
        matches!(
            node.operation(),
            Operation::EffectStore(_) | Operation::After(_)
        )
    });
    let version = if nodes.iter().any(|node| {
        matches!(
            node.operation(),
            Operation::Movement(MovementValue::Plan(plan))
                if matches!(&plan.kind, MovementKernelKind::ScatterPositions { .. })
        )
    }) {
        20
    } else if nodes
        .iter()
        .any(|node| matches!(node.operation(), Operation::Threefry(_)))
    {
        19
    } else {
        18
    };
    encode_inner(root, effects, version)
}

fn encode_inner(root: &UOp, effects: bool, version: u8) -> Result<Vec<u8>, ArtifactError> {
    root.validate().map_err(|_| ArtifactError::Format("uop"))?;
    let nodes = root
        .topological()
        .map_err(|_| ArtifactError::Format("dag"))?;
    if !effects
        && nodes.iter().any(|node| {
            matches!(
                node.operation(),
                Operation::EffectStore(_) | Operation::After(_)
            )
        })
    {
        return Err(ArtifactError::Unsupported);
    }
    if nodes.is_empty() || nodes.len() > MAX_NODES {
        return Err(ArtifactError::Format("count"));
    }
    let ids = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i as u32))
        .collect::<BTreeMap<_, _>>();
    let mut w = Writer::new();
    w.bytes(MAGIC)?;
    w.u8(version)?;
    w.u32(nodes.len() as u32)?;
    w.u32((nodes.len() - 1) as u32)?;
    for (id, node) in nodes.iter().enumerate() {
        let (kind, arg) = operation_to_wire(node.operation());
        validate_fields(&kind, node.ty(), &arg, node.sources(), effects)?;
        w.u32(id as u32)?;
        write_kind(&mut w, &kind, effects)?;
        write_type(&mut w, node.ty())?;
        write_arg(&mut w, &arg, effects)?;
        if node.sources().len() > MAX_SOURCES {
            return Err(ArtifactError::Format("source limit"));
        }
        w.u32(node.sources().len() as u32)?;
        for source in node.sources() {
            let source_id = *ids.get(source).ok_or(ArtifactError::Format("source"))?;
            if source_id >= id as u32 {
                return Err(ArtifactError::Format("source order"));
            }
            w.u32(source_id)?;
        }
    }
    let sum = checksum(&w.out);
    w.u32(sum)?;
    Ok(w.out)
}

/// Decodes and completely validates an artifact before returning its root.
pub fn decode(bytes: &[u8]) -> Result<UOp, ArtifactError> {
    if bytes.len() > MAX_BYTES {
        return Err(ArtifactError::Format("byte limit"));
    }
    if bytes.len() < 17 {
        return Err(ArtifactError::Format("length"));
    }
    let body_len = bytes.len() - 4;
    let got = u32::from_le_bytes(
        bytes[body_len..]
            .try_into()
            .map_err(|_| ArtifactError::Format("checksum"))?,
    );
    if checksum(&bytes[..body_len]) != got {
        return Err(ArtifactError::Checksum);
    }
    let mut r = Reader::new(&bytes[..body_len]);
    if r.take(4)? != MAGIC {
        return Err(ArtifactError::Format("magic"));
    }
    let version = r.u8()?;
    if !matches!(
        version,
        2 | 3
            | 4
            | 5
            | 6
            | 7
            | 8
            | 9
            | 10
            | 11
            | 12
            | LEGACY_EFFECT_VERSION
            | PREVIOUS_EFFECT_VERSION
            | 16
            | 17
            | 18
            | 19
            | VERSION
    ) {
        return Err(ArtifactError::Format("version"));
    }
    let count = r.count(MAX_NODES)?;
    if count == 0 {
        return Err(ArtifactError::Format("count"));
    }
    let root = r.u32()? as usize;
    if root != count - 1 {
        return Err(ArtifactError::Format("root"));
    }
    let mut nodes: Vec<UOp> = Vec::with_capacity(count);
    for expected in 0..count {
        if r.u32()? as usize != expected {
            return Err(ArtifactError::Format("node id"));
        }
        let kind = read_kind(&mut r, version)?;
        let ty = read_type(&mut r)?;
        let arg = read_arg(&mut r, version)?;
        let source_count = r.count(MAX_SOURCES)?;
        let mut sources = Vec::with_capacity(source_count);
        for _ in 0..source_count {
            let source = r.u32()? as usize;
            if source >= expected {
                return Err(ArtifactError::Format("forward source"));
            }
            sources.push(nodes[source].clone());
        }
        validate_fields(
            &kind,
            ty,
            &arg,
            &sources,
            version == LEGACY_EFFECT_VERSION || version >= PREVIOUS_EFFECT_VERSION,
        )?;
        let operation = operation_from_wire(kind, arg)?;
        nodes.push(UOp::from_operation(operation, ty, sources));
    }
    if !r.done() {
        return Err(ArtifactError::Format("trailing bytes"));
    }
    let root = nodes[root].clone();
    root.validate().map_err(|_| ArtifactError::Format("uop"))?;
    if root
        .topological()
        .map_err(|_| ArtifactError::Format("dag"))?
        .len()
        != count
    {
        return Err(ArtifactError::Format("unreachable node"));
    }
    Ok(root)
}

fn validate_fields(
    kind: &WireOpcode,
    ty: Option<UType>,
    arg: &WireArg,
    sources: &[UOp],
    effects: bool,
) -> Result<(), ArtifactError> {
    if ty.is_some_and(|x| x.lanes == 0) {
        return Err(ArtifactError::Format("lane width"));
    }
    match arg {
        WireArg::Scalar { dtype, bits } if !super::scalar_literal_is_valid(ty, *dtype, *bits) => {
            return Err(ArtifactError::Format("scalar literal"));
        }
        WireArg::Variable { name, bounds } => {
            if name.is_empty() || bounds.bounds().is_err() {
                return Err(ArtifactError::Format("variable"));
            }
            let mut variables = BTreeMap::new();
            for variable in bounds.variables() {
                let metadata = (variable.name().to_owned(), variable.bounds());
                if variables
                    .insert(variable.id(), metadata.clone())
                    .is_some_and(|old| old != metadata)
                {
                    return Err(ArtifactError::Format("symbolic identity"));
                }
            }
        }
        WireArg::Address { name, element, .. } => {
            if name.is_empty() || element.lanes == 0 {
                return Err(ArtifactError::Format("address"));
            }
        }
        WireArg::BufferIndex {
            elements,
            input_shape,
            output_shape,
            ..
        } => validate_index(*elements, input_shape, output_shape, None)?,
        WireArg::ViewBufferIndex {
            elements,
            input_shape,
            output_shape,
            view,
            ..
        } => validate_index(*elements, input_shape, output_shape, Some(view))?,
        WireArg::Reduction {
            input_shape,
            output_shape,
            axes,
            keepdim,
            ..
        } => {
            checked_shape(input_shape)?;
            checked_shape(output_shape)?;
            if axes.windows(2).any(|x| x[0] >= x[1])
                || axes.iter().any(|x| *x >= input_shape.rank())
            {
                return Err(ArtifactError::Format("reduction axes"));
            }
            let mut want = input_shape.dims().to_vec();
            if *keepdim {
                for axis in axes {
                    want[*axis] = 1;
                }
            } else {
                for axis in axes.iter().rev() {
                    want.remove(*axis);
                }
            }
            if output_shape.dims() != want {
                return Err(ArtifactError::Format("reduction shape"));
            }
        }
        WireArg::Matmul(plan) => plan
            .validate()
            .map_err(|_| ArtifactError::Format("matmul plan"))?,
        WireArg::Conv2d(plan) => plan
            .validate()
            .map_err(|_| ArtifactError::Format("static conv2d plan"))?,
        WireArg::TiledMatmul(payload) => payload
            .validate()
            .map_err(|_| ArtifactError::Format("tiled matmul plan"))?,
        WireArg::TensorCoreMatmul(payload) => payload
            .validate()
            .map_err(|_| ArtifactError::Format("tensor-core matmul plan"))?,
        WireArg::QuantizedMatmul(plan) => plan
            .validate()
            .map_err(|_| ArtifactError::Format("quantized matmul plan"))?,
        WireArg::QuantizedRowGather(plan) => plan
            .validate()
            .map_err(|_| ArtifactError::Format("quantized row gather plan"))?,
        WireArg::Movement(plan) => plan
            .validate()
            .map_err(|_| ArtifactError::Format("movement plan"))?,
        WireArg::Random(plan) => plan
            .validate()
            .map_err(|_| ArtifactError::Format("random plan"))?,
        WireArg::PrefixScan {
            input,
            destination,
            input_shape,
            output_shape,
            axis,
            kind,
            output,
            input_dtype,
            dtype,
            ..
        } => {
            checked_shape(input_shape)?;
            if input == destination
                || input_shape != output_shape
                || (input_shape.rank() != 0 && *axis >= input_shape.rank())
                || (input_shape.rank() == 0 && *axis != 0)
                || (*kind == crate::PrefixScanKind::Sum && *dtype == DType::Bool)
                || crate::ir::prefix_scan_output_dtype(*input_dtype, *kind, *output) != Some(*dtype)
                || (matches!(
                    kind,
                    crate::PrefixScanKind::Sum | crate::PrefixScanKind::Product
                ) && *output != crate::PrefixScanOutput::Values)
                || (matches!(
                    kind,
                    crate::PrefixScanKind::Max | crate::PrefixScanKind::Min
                ) && *output == crate::PrefixScanOutput::Indices
                    && *dtype != DType::I32)
            {
                return Err(ArtifactError::Format("prefix scan"));
            }
        }
        WireArg::Sort {
            input_shape,
            axis,
            values,
            indices,
            dtype: _,
            ..
        } => {
            checked_shape(input_shape)?;
            if (input_shape.rank() != 0 && *axis >= input_shape.rank())
                || (input_shape.rank() == 0 && *axis != 0)
                || values == indices
            {
                return Err(ArtifactError::Format("sort"));
            }
        }
        WireArg::TensorGuard {
            input_shape,
            axis,
            dtype,
            ..
        } => {
            checked_shape(input_shape)?;
            if !(1..=2).contains(&input_shape.rank())
                || *axis >= input_shape.rank()
                || !dtype.is_float()
            {
                return Err(ArtifactError::Format("tensor guard"));
            }
        }
        WireArg::Threefry {
            counter,
            key,
            counter_shape,
            key_shape,
            output,
            output_shape,
        } => ThreefryValue {
            counter: *counter,
            key: *key,
            counter_shape: counter_shape.clone(),
            key_shape: key_shape.clone(),
            output: *output,
            output_shape: output_shape.clone(),
        }
        .validate()
        .map_err(|_| ArtifactError::Format("threefry plan"))?,
        _ => {}
    }
    let operation = operation_from_wire(*kind, arg.clone())?;
    // RGUA's effects envelope is a wire-format distinction, not a general
    // impurity classification. Ordinary Store, Barrier, and Sink operations
    // have always been valid in the standard envelope; only stateful effect
    // transactions carry EffectStore/After and require the effects envelope.
    if matches!(operation, Operation::EffectStore(_) | Operation::After(_)) && !effects {
        return Err(ArtifactError::Format("kind argument"));
    }
    let type_ok = match kind {
        WireOpcode::Const | WireOpcode::VConst => ty.is_some_and(|node_ty| match arg {
            WireArg::Scalar { dtype, .. } => node_ty.scalar == *dtype,
            WireArg::Int(_) => true,
            _ => false,
        }),
        WireOpcode::DefineGlobal => {
            matches!(arg, WireArg::Address { space: AddressSpace::Global, element, .. } if ty == Some(*element))
        }
        WireOpcode::DefineLocal => {
            matches!(arg, WireArg::Address { space: AddressSpace::Local, element, .. } if ty == Some(*element))
        }
        WireOpcode::DefineRegister => {
            matches!(arg, WireArg::Address { space: AddressSpace::Register, element, .. } if ty == Some(*element))
        }
        WireOpcode::Range => {
            ty.is_some_and(|x| x.scalar.is_integer())
                && sources.first().is_some_and(|x| x.ty() == ty)
        }
        WireOpcode::Index => sources.first().is_some_and(|x| x.ty() == ty),
        WireOpcode::Load => sources
            .first()
            .is_some_and(|x| x.ty() == ty && matches!(x.operation(), Operation::Index(_))),
        WireOpcode::Store => {
            ty.is_none()
                && sources
                    .first()
                    .zip(sources.get(1))
                    .is_some_and(|(index, value)| {
                        matches!(index.operation(), Operation::Index(_)) && index.ty() == value.ty()
                    })
        }
        WireOpcode::GraphUnary(op) => {
            graph_unary_type_is_valid(*op, sources.first().and_then(|source| source.ty()), ty)
        }
        WireOpcode::Unary(_) => sources.first().is_some_and(|x| x.ty() == ty),
        WireOpcode::GraphBinary(_) => ty.is_some() && sources.iter().all(|x| x.ty().is_some()),
        WireOpcode::GraphCompare(_) => {
            ty == Some(UType::scalar(DType::Bool)) && sources.iter().all(|x| x.ty().is_some())
        }
        WireOpcode::GraphLogical(_) => {
            ty == Some(UType::scalar(DType::Bool)) && sources.iter().all(|x| x.ty() == ty)
        }
        WireOpcode::Threefry => {
            matches!(arg, WireArg::Threefry { .. })
                && ty == Some(UType::scalar(DType::U64))
                && sources.is_empty()
        }
        WireOpcode::Matmul => match &operation {
            Operation::Matmul(MatmulValue::Serial(plan)) => ty == Some(UType::scalar(plan.dtype)),
            Operation::Matmul(MatmulValue::Tiled(plan)) => {
                ty == Some(UType::scalar(plan.matmul.dtype))
            }
            Operation::Matmul(MatmulValue::TensorCore(plan)) => {
                ty == Some(UType::scalar(plan.matmul.dtype))
            }
            Operation::Matmul(MatmulValue::Quantized(plan)) => {
                ty == Some(UType::scalar(plan.output_dtype))
            }
            _ => false,
        },
        WireOpcode::Movement => {
            matches!(arg, WireArg::Movement(plan) if ty == Some(UType::scalar(plan.dtype)))
                || matches!(arg, WireArg::QuantizedRowGather(plan) if ty == Some(UType::scalar(plan.output_dtype)))
        }
        WireOpcode::Random => {
            matches!(arg, WireArg::Random(plan) if ty == Some(UType::scalar(plan.dtype)))
        }
        WireOpcode::PrefixScan => {
            matches!(arg, WireArg::PrefixScan { dtype, .. } if ty == Some(UType::scalar(*dtype)))
        }
        WireOpcode::Sort => {
            matches!(arg, WireArg::Sort { dtype, .. } if ty == Some(UType::scalar(*dtype)))
        }
        WireOpcode::TensorGuard => {
            matches!(arg, WireArg::TensorGuard { dtype, .. } if ty == Some(UType::scalar(*dtype)))
        }
        WireOpcode::ReduceAccumulate => {
            ty.zip(sources.first().and_then(|source| source.ty()))
                .is_some_and(|(accumulator, init)| accumulator == init)
                && ty
                    .zip(sources.get(1).and_then(|source| source.ty()))
                    .is_some_and(|(accumulator, value)| accumulator.lanes == value.lanes)
        }
        // ReduceFinalize is the explicit accumulator-to-output storage
        // boundary. Default narrow reductions therefore legitimately consume
        // an F32 accumulator while producing F16/BF16 output storage.
        WireOpcode::ReduceFinalize => ty
            .zip(sources.first().and_then(|source| source.ty()))
            .is_some_and(|(output, accumulator)| output.lanes == accumulator.lanes),
        WireOpcode::Ternary(super::Ternary::Where) => {
            sources
                .first()
                .is_some_and(|x| x.ty() == Some(UType::scalar(DType::Bool)))
                && sources
                    .get(1)
                    .zip(sources.get(2))
                    .is_some_and(|(a, b)| a.ty() == ty && b.ty() == ty)
        }
        WireOpcode::Sink => ty.is_none(),
        _ => true,
    };
    if !type_ok {
        return Err(ArtifactError::Format("kind type"));
    }
    Ok(())
}

fn validate_index(
    elements: usize,
    input: &Shape,
    output: &Shape,
    view: Option<&AffineView>,
) -> Result<(), ArtifactError> {
    checked_shape(input)?;
    checked_shape(output)?;
    if input.numel().ok() != Some(elements)
        || input.rank() > output.rank()
        || !input
            .dims()
            .iter()
            .rev()
            .zip(output.dims().iter().rev())
            .all(|(a, b)| *a == 1 || a == b)
    {
        return Err(ArtifactError::Format("index shape"));
    }
    if let Some(view) = view {
        view.validate_read()
            .map_err(|_| ArtifactError::Format("view"))?;
        if &view.logical_shape != input {
            return Err(ArtifactError::Format("view logical shape"));
        }
    }
    Ok(())
}

pub(crate) fn validate_view(view: &ViewMap) -> Result<(), ArtifactError> {
    let source = checked_shape(&view.source_shape)?;
    checked_shape(&view.logical_shape)?;
    if view.strides.len() != view.logical_shape.rank() {
        return Err(ArtifactError::Format("view strides"));
    }
    if source == 0 {
        if view.logical_shape.numel().ok() != Some(0) || view.offset != 0 {
            return Err(ArtifactError::Format("empty view"));
        }
        return Ok(());
    }
    let mut max = view.offset;
    for (&dim, &stride) in view.logical_shape.dims().iter().zip(&view.strides) {
        if dim != 0 {
            max = max
                .checked_add(
                    (dim - 1)
                        .checked_mul(stride)
                        .ok_or(ArtifactError::Format("view overflow"))?,
                )
                .ok_or(ArtifactError::Format("view overflow"))?;
        }
    }
    if view.logical_shape.numel().ok() == Some(0) {
        if view.offset > source {
            return Err(ArtifactError::Format("empty view bounds"));
        }
    } else if max >= source {
        return Err(ArtifactError::Format("view bounds"));
    }
    Ok(())
}

fn checked_shape(shape: &Shape) -> Result<usize, ArtifactError> {
    if shape.rank() > MAX_COLLECTION {
        return Err(ArtifactError::Format("rank limit"));
    }
    shape
        .numel()
        .map_err(|_| ArtifactError::Format("shape overflow"))
}

fn write_type(w: &mut Writer, ty: Option<UType>) -> Result<(), ArtifactError> {
    match ty {
        None => w.u8(0),
        Some(ty) if ty.lanes != 0 => {
            w.u8(1)?;
            w.u8(dtype_tag(ty.scalar))?;
            w.u16(ty.lanes)
        }
        Some(_) => Err(ArtifactError::Format("lane width")),
    }
}
fn read_type(r: &mut Reader<'_>) -> Result<Option<UType>, ArtifactError> {
    match r.u8()? {
        0 => Ok(None),
        1 => {
            let scalar = dtype(r.u8()?)?;
            let lanes = r.u16()?;
            if lanes == 0 {
                Err(ArtifactError::Format("lane width"))
            } else {
                Ok(Some(UType { scalar, lanes }))
            }
        }
        _ => Err(ArtifactError::Format("type tag")),
    }
}

fn write_kind(w: &mut Writer, kind: &WireOpcode, effects: bool) -> Result<(), ArtifactError> {
    use WireOpcode::*;
    let (tag, sub) = match kind {
        Const => (0, None),
        VConst => (1, None),
        DefineVar => (2, None),
        DefineGlobal => (3, None),
        DefineLocal => (4, None),
        DefineRegister => (5, None),
        Special => (6, None),
        Range => (7, None),
        EndRange => (8, None),
        If => (9, None),
        EndIf => (10, None),
        Unary(x) => (11, Some(tag_unary(*x))),
        Binary(x) => (12, Some(tag_binary(*x))),
        GraphUnary(x) => (13, Some(tag_graph_unary(*x))),
        GraphBinary(x) => (14, Some(tag_graph_binary(*x))),
        GraphCompare(x) => (15, Some(tag_compare(*x))),
        GraphLogical(x) => (16, Some(tag_logical(*x))),
        ReduceInit => (17, None),
        ReduceAccumulate => (18, None),
        ReduceFinalize => (19, None),
        Ternary(super::Ternary::Where) => (20, Some(0)),
        Cast => (21, None),
        Bitcast => (22, None),
        Vectorize => (23, None),
        Gep => (24, None),
        Index => (25, None),
        Load => (26, None),
        Store => (27, None),
        Barrier => (28, None),
        Sink => (29, None),
        Matmul => (30, None),
        Conv2d => (35, None),
        Movement => (31, None),
        Random => (32, None),
        PrefixScan => (36, None),
        Sort => (37, None),
        TensorGuard => (38, None),
        Threefry => (39, None),
        EffectStore if effects => (33, None),
        After if effects => (34, None),
        EffectStore | After => return Err(ArtifactError::Unsupported),
    };
    w.u8(tag)?;
    if let Some(x) = sub {
        w.u8(x)?;
    }
    Ok(())
}
fn read_kind(r: &mut Reader<'_>, version: u8) -> Result<WireOpcode, ArtifactError> {
    use WireOpcode::*;
    Ok(match r.u8()? {
        0 => Const,
        1 => VConst,
        2 => DefineVar,
        3 => DefineGlobal,
        4 => DefineLocal,
        5 => DefineRegister,
        6 => Special,
        7 => Range,
        8 => EndRange,
        9 => If,
        10 => EndIf,
        11 => Unary(enum_unary(r.u8()?)?),
        12 => Binary(enum_binary(r.u8()?)?),
        13 => GraphUnary(enum_graph_unary(r.u8()?)?),
        14 => GraphBinary(enum_graph_binary(r.u8()?)?),
        15 => GraphCompare(enum_compare(r.u8()?)?),
        16 => GraphLogical(enum_logical(r.u8()?)?),
        17 => ReduceInit,
        18 => ReduceAccumulate,
        19 => ReduceFinalize,
        20 => Ternary(match r.u8()? {
            0 => super::Ternary::Where,
            _ => return Err(ArtifactError::Format("ternary")),
        }),
        21 => Cast,
        22 => Bitcast,
        23 => Vectorize,
        24 => Gep,
        25 => Index,
        26 => Load,
        27 => Store,
        28 => Barrier,
        29 => Sink,
        30 if version >= 3 => Matmul,
        31 if version >= 4 => Movement,
        32 if version >= 9 => Random,
        33 if version == LEGACY_EFFECT_VERSION || version >= PREVIOUS_EFFECT_VERSION => EffectStore,
        34 if version == LEGACY_EFFECT_VERSION || version >= PREVIOUS_EFFECT_VERSION => After,
        35 if version >= 10 => Conv2d,
        36 if version >= 11 => PrefixScan,
        37 if version >= 16 => Sort,
        38 if version >= 17 => TensorGuard,
        39 if version >= 19 => Threefry,
        _ => return Err(ArtifactError::Format("kind tag")),
    })
}

fn write_arg(w: &mut Writer, arg: &WireArg, effects: bool) -> Result<(), ArtifactError> {
    match arg {
        WireArg::None => w.u8(0),
        WireArg::Int(x) => {
            w.u8(1)?;
            w.i64(*x)
        }
        WireArg::Scalar { dtype, bits } => {
            w.u8(2)?;
            w.u8(dtype_tag(*dtype))?;
            w.u64(*bits)
        }
        WireArg::Name(x) => {
            w.u8(3)?;
            w.string(x)
        }
        WireArg::Variable { name, bounds } => {
            w.u8(4)?;
            w.string(name)?;
            write_symbolic(w, bounds, 0)
        }
        WireArg::Address {
            space,
            name,
            element,
        } => {
            w.u8(5)?;
            w.u8(match space {
                AddressSpace::Global => 0,
                AddressSpace::Local => 1,
                AddressSpace::Register => 2,
            })?;
            w.string(name)?;
            write_type(w, Some(*element))
        }
        WireArg::RangeAxis(x) => {
            w.u8(6)?;
            w.u32(*x)
        }
        WireArg::GepLane(x) => {
            w.u8(7)?;
            w.u16(*x)
        }
        WireArg::BufferIndex {
            buffer,
            elements,
            input_shape,
            output_shape,
        } => {
            w.u8(8)?;
            w.u64(*buffer)?;
            w.usize(*elements)?;
            write_shape(w, input_shape)?;
            write_shape(w, output_shape)
        }
        WireArg::ViewBufferIndex {
            buffer,
            elements,
            input_shape,
            output_shape,
            view,
        } => {
            w.u8(9)?;
            w.u64(*buffer)?;
            w.usize(*elements)?;
            write_shape(w, input_shape)?;
            write_shape(w, output_shape)?;
            write_affine_view(w, view)
        }
        WireArg::Reduction {
            input_shape,
            output_shape,
            axes,
            keepdim,
            kind,
            mean,
        } => {
            w.u8(10)?;
            write_shape(w, input_shape)?;
            write_shape(w, output_shape)?;
            w.usizes(axes)?;
            w.bool(*keepdim)?;
            w.u8(tag_reduce(*kind))?;
            w.bool(*mean)
        }
        WireArg::Matmul(plan) => {
            w.u8(11)?;
            write_matmul(w, plan)
        }
        WireArg::Conv2d(plan) => {
            w.u8(19)?;
            write_static_conv2d(w, plan)
        }
        WireArg::TiledMatmul(payload) => {
            w.u8(13)?;
            write_tiled_matmul(w, payload)
        }
        WireArg::QuantizedMatmul(plan) => {
            w.u8(14)?;
            write_quantized_matmul(w, plan)
        }
        WireArg::TensorCoreMatmul(payload) => {
            w.u8(15)?;
            write_tensor_core_matmul(w, payload)
        }
        WireArg::QuantizedRowGather(plan) => {
            w.u8(16)?;
            write_quantized_row_gather(w, plan)
        }
        WireArg::Movement(plan) => {
            w.u8(12)?;
            write_movement(w, plan)
        }
        WireArg::Random(plan) => {
            plan.validate()
                .map_err(|_| ArtifactError::Format("random plan"))?;
            w.u8(17)?;
            w.u64(plan.output.index() as u64)?;
            write_shape(w, &plan.shape)?;
            w.u8(dtype_tag(plan.dtype))?;
            match plan.kind {
                RandomKind::Uniform { low, high } => {
                    w.u8(0)?;
                    w.u64(low.to_bits())?;
                    w.u64(high.to_bits())?;
                }
                RandomKind::Normal { mean, std } => {
                    w.u8(1)?;
                    w.u64(mean.to_bits())?;
                    w.u64(std.to_bits())?;
                }
                RandomKind::RandInt { low, high } => {
                    w.u8(2)?;
                    w.i64(low)?;
                    w.i64(high)?;
                }
            }
            for value in plan.stream.key {
                w.u32(value)?;
            }
            for value in plan.stream.counter {
                w.u32(value)?;
            }
            w.u32(plan.stream.device)?;
            w.usize(plan.word_count)
        }
        WireArg::PrefixScan {
            input,
            destination,
            input_shape,
            output_shape,
            axis,
            kind,
            output,
            input_dtype,
            dtype,
        } => {
            w.u8(20)?;
            w.u64(input.index() as u64)?;
            w.u64(destination.index() as u64)?;
            write_shape(w, input_shape)?;
            write_shape(w, output_shape)?;
            w.usize(*axis)?;
            w.u8(tag_prefix_scan(*kind))?;
            w.u8(tag_prefix_scan_output(*output))?;
            w.u8(dtype_tag(*input_dtype))?;
            w.u8(dtype_tag(*dtype))
        }
        WireArg::Sort {
            input,
            input_shape,
            axis,
            descending,
            values,
            indices,
            dtype,
        } => {
            w.u8(21)?;
            w.u64(input.index() as u64)?;
            write_shape(w, input_shape)?;
            w.usize(*axis)?;
            w.bool(*descending)?;
            w.u64(values.index() as u64)?;
            w.u64(indices.index() as u64)?;
            w.u8(dtype_tag(*dtype))
        }
        WireArg::TensorGuard {
            input,
            input_shape,
            axis,
            dtype,
        } => {
            w.u8(22)?;
            w.u64(input.index() as u64)?;
            write_shape(w, input_shape)?;
            w.usize(*axis)?;
            w.u8(dtype_tag(*dtype))
        }
        WireArg::Threefry {
            counter,
            key,
            counter_shape,
            key_shape,
            output,
            output_shape,
        } => {
            w.u8(23)?;
            w.u64(counter.index() as u64)?;
            w.u64(key.index() as u64)?;
            write_shape(w, counter_shape)?;
            write_shape(w, key_shape)?;
            w.u64(output.index() as u64)?;
            write_shape(w, output_shape)
        }
        WireArg::Effect(payload) if effects => {
            w.u8(18)?;
            write_effect_payload(w, payload)
        }
        WireArg::Effect(_) => Err(ArtifactError::Unsupported),
    }
}
fn read_arg(r: &mut Reader<'_>, version: u8) -> Result<WireArg, ArtifactError> {
    Ok(match r.u8()? {
        0 => WireArg::None,
        1 => WireArg::Int(r.i64()?),
        2 => WireArg::Scalar {
            dtype: dtype(r.u8()?)?,
            bits: r.u64()?,
        },
        3 => WireArg::Name(r.string()?),
        4 => WireArg::Variable {
            name: r.string()?,
            bounds: read_symbolic(r, 0)?,
        },
        5 => WireArg::Address {
            space: match r.u8()? {
                0 => AddressSpace::Global,
                1 => AddressSpace::Local,
                2 => AddressSpace::Register,
                _ => return Err(ArtifactError::Format("address space")),
            },
            name: r.string()?,
            element: read_type(r)?.ok_or(ArtifactError::Format("address type"))?,
        },
        6 => WireArg::RangeAxis(r.u32()?),
        7 => WireArg::GepLane(r.u16()?),
        8 => WireArg::BufferIndex {
            buffer: r.u64()?,
            elements: r.usize()?,
            input_shape: read_shape(r)?,
            output_shape: read_shape(r)?,
        },
        9 => WireArg::ViewBufferIndex {
            buffer: r.u64()?,
            elements: r.usize()?,
            input_shape: read_shape(r)?,
            output_shape: read_shape(r)?,
            view: if version >= 10 {
                read_affine_view(r)?
            } else {
                read_view(r)?.into()
            },
        },
        10 => WireArg::Reduction {
            input_shape: read_shape(r)?,
            output_shape: read_shape(r)?,
            axes: r.usizes()?,
            keepdim: r.bool()?,
            kind: enum_reduce(r.u8()?)?,
            mean: r.bool()?,
        },
        11 if version >= 3 => WireArg::Matmul(Box::new(read_matmul(r)?)),
        12 if version >= 4 => WireArg::Movement(Box::new(read_movement(r, version)?)),
        13 if version >= 5 => WireArg::TiledMatmul(Box::new(read_tiled_matmul(r)?)),
        14 if version >= 6 => WireArg::QuantizedMatmul(Box::new(read_quantized_matmul(r)?)),
        15 if version >= 7 => WireArg::TensorCoreMatmul(Box::new(read_tensor_core_matmul(r)?)),
        16 if version >= 8 => WireArg::QuantizedRowGather(Box::new(read_quantized_row_gather(r)?)),
        17 if version >= 9 => {
            let output = crate::NodeId::from_index(
                usize::try_from(r.u64()?).map_err(|_| ArtifactError::Format("random node"))?,
            );
            let shape = read_shape(r)?;
            let dtype = dtype(r.u8()?)?;
            let kind = match r.u8()? {
                0 => RandomKind::Uniform {
                    low: f64::from_bits(r.u64()?),
                    high: f64::from_bits(r.u64()?),
                },
                1 => RandomKind::Normal {
                    mean: f64::from_bits(r.u64()?),
                    std: f64::from_bits(r.u64()?),
                },
                2 => RandomKind::RandInt {
                    low: r.i64()?,
                    high: r.i64()?,
                },
                _ => return Err(ArtifactError::Format("random kind")),
            };
            let key = [r.u32()?, r.u32()?];
            let counter = [r.u32()?, r.u32()?];
            let stream = crate::RandomStream {
                device: r.u32()?,
                key,
                counter,
            };
            let stored_words = r.usize()?;
            let plan =
                crate::random::plan::RandomKernelPlan::new(output, shape, dtype, kind, stream)
                    .map_err(|_| ArtifactError::Format("random plan"))?;
            if plan.word_count != stored_words {
                return Err(ArtifactError::Format("random words"));
            }
            WireArg::Random(Box::new(plan))
        }
        18 if version >= 11 => WireArg::Effect(Box::new(read_effect_payload(
            r,
            version == LEGACY_EFFECT_VERSION || version >= PREVIOUS_EFFECT_VERSION,
        )?)),
        19 if version >= 10 => WireArg::Conv2d(Box::new(read_static_conv2d(r)?)),
        20 if version >= 18 => WireArg::PrefixScan {
            input: crate::NodeId::from_index(
                usize::try_from(r.u64()?)
                    .map_err(|_| ArtifactError::Format("prefix scan input"))?,
            ),
            destination: crate::NodeId::from_index(
                usize::try_from(r.u64()?)
                    .map_err(|_| ArtifactError::Format("prefix scan destination"))?,
            ),
            input_shape: read_shape(r)?,
            output_shape: read_shape(r)?,
            axis: r.usize()?,
            kind: enum_prefix_scan(r.u8()?)?,
            output: enum_prefix_scan_output(r.u8()?)?,
            input_dtype: dtype(r.u8()?)?,
            dtype: dtype(r.u8()?)?,
        },
        20 if version >= 11 => {
            return Err(ArtifactError::Format(
                "legacy prefix scan lacks source/result ABI",
            ));
        }
        21 if version >= 16 => WireArg::Sort {
            input: crate::NodeId::from_index(
                usize::try_from(r.u64()?).map_err(|_| ArtifactError::Format("sort input"))?,
            ),
            input_shape: read_shape(r)?,
            axis: r.usize()?,
            descending: r.bool()?,
            values: crate::NodeId::from_index(
                usize::try_from(r.u64()?).map_err(|_| ArtifactError::Format("sort values"))?,
            ),
            indices: crate::NodeId::from_index(
                usize::try_from(r.u64()?).map_err(|_| ArtifactError::Format("sort indices"))?,
            ),
            dtype: dtype(r.u8()?)?,
        },
        22 if version >= 17 => WireArg::TensorGuard {
            input: crate::NodeId::from_index(
                usize::try_from(r.u64()?)
                    .map_err(|_| ArtifactError::Format("tensor guard input"))?,
            ),
            input_shape: read_shape(r)?,
            axis: r.usize()?,
            dtype: dtype(r.u8()?)?,
        },
        23 if version >= 19 => WireArg::Threefry {
            counter: crate::NodeId::from_index(
                usize::try_from(r.u64()?).map_err(|_| ArtifactError::Format("threefry counter"))?,
            ),
            key: crate::NodeId::from_index(
                usize::try_from(r.u64()?).map_err(|_| ArtifactError::Format("threefry key"))?,
            ),
            counter_shape: read_shape(r)?,
            key_shape: read_shape(r)?,
            output: crate::NodeId::from_index(
                usize::try_from(r.u64()?).map_err(|_| ArtifactError::Format("threefry output"))?,
            ),
            output_shape: read_shape(r)?,
        },
        _ => return Err(ArtifactError::Format("argument tag")),
    })
}

fn write_tensor_core_matmul(
    w: &mut Writer,
    payload: &TensorCoreMatmulPayload,
) -> Result<(), ArtifactError> {
    payload
        .validate()
        .map_err(|_| ArtifactError::Format("tensor-core matmul plan"))?;
    write_matmul(w, &payload.matmul)?;
    let plan = &payload.tensor_core;
    write_target(w, &plan.target)?;
    w.u8(match plan.instruction {
        MmaInstruction::M16N8K16RowColF32 => 0,
    })?;
    w.u8(dtype_tag(plan.input_dtype))?;
    w.u8(dtype_tag(plan.accumulator_dtype))?;
    w.u8(dtype_tag(plan.output_dtype))?;
    w.u8(match plan.output_policy {
        TensorCoreOutputPolicy::RequantizeToGraphDType => 0,
    })?;
    w.u8(match plan.tail_policy {
        TensorCoreTailPolicy::ExactTilesOnly => 0,
    })?;
    w.u32(plan.block_m)?;
    w.u32(plan.block_n)?;
    w.u32(plan.block_k)?;
    for dimension in plan.workgroup {
        w.u32(dimension)?;
    }
    write_shared_layout(w, &plan.lhs_shared)?;
    write_shared_layout(w, &plan.rhs_shared)?;
    w.u32(plan.fragments.lanes)?;
    w.u32(plan.fragments.lhs_elements_per_lane)?;
    w.u32(plan.fragments.rhs_elements_per_lane)?;
    w.u32(plan.fragments.accumulator_elements_per_lane)?;
    w.u32(plan.fragments.lhs_registers_per_lane)?;
    w.u32(plan.fragments.rhs_registers_per_lane)?;
    w.u32(plan.fragments.accumulator_registers_per_lane)?;
    if plan.barriers.len() > MAX_COLLECTION {
        return Err(ArtifactError::Format("barrier count"));
    }
    w.u32(plan.barriers.len() as u32)?;
    for barrier in &plan.barriers {
        w.u32(barrier.sequence)?;
        w.u8(match barrier.kind {
            MatmulBarrierKind::LoadsVisible => 0,
            MatmulBarrierKind::TileConsumed => 1,
        })?;
        w.bool(barrier.uniform)?;
        write_u32s(w, &barrier.initializes)?;
        write_u32s(w, &barrier.consumes)?;
    }
    let resources = &plan.resources;
    w.u32(resources.threads_per_block)?;
    w.u32(resources.warps_per_block)?;
    w.u32(resources.registers_per_thread)?;
    w.u32(resources.registers_per_block)?;
    w.usize(resources.shared_bytes_per_block)?;
    w.u32(resources.resident_blocks_per_sm)?;
    w.u32(resources.resident_warps_per_sm)?;
    w.u64(plan.estimated_cost)?;
    w.u64(plan.cache_key)
}

fn read_tensor_core_matmul(r: &mut Reader<'_>) -> Result<TensorCoreMatmulPayload, ArtifactError> {
    let matmul = read_matmul(r)?;
    let target = read_target(r)?;
    let instruction = match r.u8()? {
        0 => MmaInstruction::M16N8K16RowColF32,
        _ => return Err(ArtifactError::Format("mma instruction")),
    };
    let input_dtype = dtype(r.u8()?)?;
    let accumulator_dtype = dtype(r.u8()?)?;
    let output_dtype = dtype(r.u8()?)?;
    let output_policy = match r.u8()? {
        0 => TensorCoreOutputPolicy::RequantizeToGraphDType,
        _ => return Err(ArtifactError::Format("tensor-core output policy")),
    };
    let tail_policy = match r.u8()? {
        0 => TensorCoreTailPolicy::ExactTilesOnly,
        _ => return Err(ArtifactError::Format("tensor-core tail policy")),
    };
    let block_m = r.u32()?;
    let block_n = r.u32()?;
    let block_k = r.u32()?;
    let workgroup = [r.u32()?, r.u32()?, r.u32()?];
    let lhs_shared = read_shared_layout(r)?;
    let rhs_shared = read_shared_layout(r)?;
    let fragments = MmaFragmentLayout {
        lanes: r.u32()?,
        lhs_elements_per_lane: r.u32()?,
        rhs_elements_per_lane: r.u32()?,
        accumulator_elements_per_lane: r.u32()?,
        lhs_registers_per_lane: r.u32()?,
        rhs_registers_per_lane: r.u32()?,
        accumulator_registers_per_lane: r.u32()?,
    };
    let count = r.count(MAX_COLLECTION)?;
    let mut barriers = Vec::with_capacity(count);
    for _ in 0..count {
        barriers.push(MatmulBarrierPhase {
            sequence: r.u32()?,
            kind: match r.u8()? {
                0 => MatmulBarrierKind::LoadsVisible,
                1 => MatmulBarrierKind::TileConsumed,
                _ => return Err(ArtifactError::Format("matmul barrier kind")),
            },
            uniform: r.bool()?,
            initializes: read_u32s(r)?,
            consumes: read_u32s(r)?,
        });
    }
    let resources = MatmulResourceEstimate {
        threads_per_block: r.u32()?,
        warps_per_block: r.u32()?,
        registers_per_thread: r.u32()?,
        registers_per_block: r.u32()?,
        shared_bytes_per_block: r.usize()?,
        resident_blocks_per_sm: r.u32()?,
        resident_warps_per_sm: r.u32()?,
    };
    let payload = TensorCoreMatmulPayload {
        matmul,
        tensor_core: TensorCoreMatmulPlan {
            target,
            instruction,
            input_dtype,
            accumulator_dtype,
            output_dtype,
            output_policy,
            tail_policy,
            block_m,
            block_n,
            block_k,
            workgroup,
            lhs_shared,
            rhs_shared,
            fragments,
            barriers,
            resources,
            estimated_cost: r.u64()?,
            cache_key: r.u64()?,
        },
    };
    payload
        .validate()
        .map_err(|_| ArtifactError::Format("tensor-core matmul plan"))?;
    Ok(payload)
}

fn write_tiled_matmul(w: &mut Writer, payload: &TiledMatmulPayload) -> Result<(), ArtifactError> {
    payload
        .validate()
        .map_err(|_| ArtifactError::Format("tiled matmul plan"))?;
    write_matmul(w, &payload.matmul)?;
    let tile = &payload.tile;
    write_target(w, &tile.target)?;
    w.u32(tile.block_m)?;
    w.u32(tile.block_n)?;
    w.u32(tile.block_k)?;
    for dimension in tile.workgroup {
        w.u32(dimension)?;
    }
    for dimension in tile.register_tile {
        w.u32(dimension)?;
    }
    w.u32(tile.vector_width)?;
    write_shared_layout(w, &tile.lhs_shared)?;
    write_shared_layout(w, &tile.rhs_shared)?;
    w.bool(tile.tails.m)?;
    w.bool(tile.tails.n)?;
    w.bool(tile.tails.k)?;
    w.bool(tile.tails.broadcast_batch)?;
    if tile.barriers.len() > MAX_COLLECTION {
        return Err(ArtifactError::Format("barrier count"));
    }
    w.u32(tile.barriers.len() as u32)?;
    for barrier in &tile.barriers {
        w.u32(barrier.sequence)?;
        w.u8(match barrier.kind {
            MatmulBarrierKind::LoadsVisible => 0,
            MatmulBarrierKind::TileConsumed => 1,
        })?;
        w.bool(barrier.uniform)?;
        write_u32s(w, &barrier.initializes)?;
        write_u32s(w, &barrier.consumes)?;
    }
    let resources = &tile.resources;
    w.u32(resources.threads_per_block)?;
    w.u32(resources.warps_per_block)?;
    w.u32(resources.registers_per_thread)?;
    w.u32(resources.registers_per_block)?;
    w.usize(resources.shared_bytes_per_block)?;
    w.u32(resources.resident_blocks_per_sm)?;
    w.u32(resources.resident_warps_per_sm)?;
    w.u64(tile.estimated_cost)?;
    w.u64(tile.cache_key)
}

fn read_tiled_matmul(r: &mut Reader<'_>) -> Result<TiledMatmulPayload, ArtifactError> {
    let matmul = read_matmul(r)?;
    let target = read_target(r)?;
    let block_m = r.u32()?;
    let block_n = r.u32()?;
    let block_k = r.u32()?;
    let workgroup = [r.u32()?, r.u32()?, r.u32()?];
    let register_tile = [r.u32()?, r.u32()?];
    let vector_width = r.u32()?;
    let lhs_shared = read_shared_layout(r)?;
    let rhs_shared = read_shared_layout(r)?;
    let tails = TiledMatmulTails {
        m: r.bool()?,
        n: r.bool()?,
        k: r.bool()?,
        broadcast_batch: r.bool()?,
    };
    let count = r.count(MAX_COLLECTION)?;
    let mut barriers = Vec::with_capacity(count);
    for _ in 0..count {
        barriers.push(MatmulBarrierPhase {
            sequence: r.u32()?,
            kind: match r.u8()? {
                0 => MatmulBarrierKind::LoadsVisible,
                1 => MatmulBarrierKind::TileConsumed,
                _ => return Err(ArtifactError::Format("matmul barrier kind")),
            },
            uniform: r.bool()?,
            initializes: read_u32s(r)?,
            consumes: read_u32s(r)?,
        });
    }
    let resources = MatmulResourceEstimate {
        threads_per_block: r.u32()?,
        warps_per_block: r.u32()?,
        registers_per_thread: r.u32()?,
        registers_per_block: r.u32()?,
        shared_bytes_per_block: r.usize()?,
        resident_blocks_per_sm: r.u32()?,
        resident_warps_per_sm: r.u32()?,
    };
    let payload = TiledMatmulPayload {
        matmul,
        tile: TiledMatmulPlan {
            target,
            block_m,
            block_n,
            block_k,
            workgroup,
            register_tile,
            vector_width,
            lhs_shared,
            rhs_shared,
            tails,
            barriers,
            resources,
            estimated_cost: r.u64()?,
            cache_key: r.u64()?,
        },
    };
    payload
        .validate()
        .map_err(|_| ArtifactError::Format("tiled matmul plan"))?;
    Ok(payload)
}

fn write_target(w: &mut Writer, target: &MatmulTargetCaps) -> Result<(), ArtifactError> {
    w.u32(target.sm)?;
    w.u32(target.warp_size)?;
    w.u32(target.max_threads_per_block)?;
    w.u32(target.max_threads_per_sm)?;
    w.usize(target.max_shared_bytes_per_block)?;
    w.usize(target.max_shared_bytes_per_sm)?;
    w.u32(target.max_registers_per_thread)?;
    w.u32(target.max_registers_per_sm)?;
    w.u32(target.max_blocks_per_sm)
}

fn read_target(r: &mut Reader<'_>) -> Result<MatmulTargetCaps, ArtifactError> {
    Ok(MatmulTargetCaps {
        sm: r.u32()?,
        warp_size: r.u32()?,
        max_threads_per_block: r.u32()?,
        max_threads_per_sm: r.u32()?,
        max_shared_bytes_per_block: r.usize()?,
        max_shared_bytes_per_sm: r.usize()?,
        max_registers_per_thread: r.u32()?,
        max_registers_per_sm: r.u32()?,
        max_blocks_per_sm: r.u32()?,
    })
}

fn write_shared_layout(w: &mut Writer, layout: &SharedTileLayout) -> Result<(), ArtifactError> {
    w.u32(layout.allocation_id)?;
    w.u32(layout.rows)?;
    w.u32(layout.columns)?;
    w.u32(layout.row_stride)?;
    w.usize(layout.bytes)?;
    w.usize(layout.alignment)
}

fn read_shared_layout(r: &mut Reader<'_>) -> Result<SharedTileLayout, ArtifactError> {
    Ok(SharedTileLayout {
        allocation_id: r.u32()?,
        rows: r.u32()?,
        columns: r.u32()?,
        row_stride: r.u32()?,
        bytes: r.usize()?,
        alignment: r.usize()?,
    })
}

fn write_u32s(w: &mut Writer, values: &[u32]) -> Result<(), ArtifactError> {
    if values.len() > MAX_COLLECTION {
        return Err(ArtifactError::Format("u32 collection"));
    }
    w.u32(values.len() as u32)?;
    for value in values {
        w.u32(*value)?;
    }
    Ok(())
}

fn read_u32s(r: &mut Reader<'_>) -> Result<Vec<u32>, ArtifactError> {
    let count = r.count(MAX_COLLECTION)?;
    (0..count).map(|_| r.u32()).collect()
}

fn write_matmul(w: &mut Writer, plan: &MatmulKernelPlan) -> Result<(), ArtifactError> {
    plan.validate()
        .map_err(|_| ArtifactError::Format("matmul plan"))?;
    w.u64(plan.lhs.index() as u64)?;
    w.u64(plan.rhs.index() as u64)?;
    w.u64(plan.output.index() as u64)?;
    write_shape(w, &plan.lhs_shape)?;
    write_shape(w, &plan.rhs_shape)?;
    write_shape(w, &plan.output_shape)?;
    w.u8(dtype_tag(plan.lhs_dtype))?;
    w.u8(dtype_tag(plan.rhs_dtype))?;
    w.u8(dtype_tag(plan.dtype))?;
    w.usizes(&plan.batch_shape)?;
    w.usize(plan.m)?;
    w.usize(plan.n)?;
    w.usize(plan.k)?;
    w.bool(plan.lhs_vector)?;
    w.bool(plan.rhs_vector)?;
    w.u64(plan.cache_key)
}

fn read_matmul(r: &mut Reader<'_>) -> Result<MatmulKernelPlan, ArtifactError> {
    let node = |id| {
        usize::try_from(id)
            .map(NodeId::from_index)
            .map_err(|_| ArtifactError::Format("matmul node"))
    };
    let plan = MatmulKernelPlan {
        lhs: node(r.u64()?)?,
        rhs: node(r.u64()?)?,
        output: node(r.u64()?)?,
        lhs_shape: read_shape(r)?,
        rhs_shape: read_shape(r)?,
        output_shape: read_shape(r)?,
        lhs_dtype: dtype(r.u8()?)?,
        rhs_dtype: dtype(r.u8()?)?,
        dtype: dtype(r.u8()?)?,
        batch_shape: r.usizes()?,
        m: r.usize()?,
        n: r.usize()?,
        k: r.usize()?,
        lhs_vector: r.bool()?,
        rhs_vector: r.bool()?,
        cache_key: r.u64()?,
    };
    plan.validate()
        .map_err(|_| ArtifactError::Format("matmul plan"))?;
    Ok(plan)
}

fn write_static_conv2d(w: &mut Writer, plan: &StaticConv2dPlan) -> Result<(), ArtifactError> {
    plan.validate()
        .map_err(|_| ArtifactError::Format("static conv2d plan"))?;
    w.u64(plan.input.index() as u64)?;
    w.u64(plan.weight.index() as u64)?;
    match plan.bias {
        Some(bias) => {
            w.bool(true)?;
            w.u64(bias.index() as u64)?;
        }
        None => w.bool(false)?,
    }
    w.u64(plan.output.index() as u64)?;
    write_shape(w, &plan.input_shape)?;
    write_shape(w, &plan.weight_shape)?;
    match &plan.bias_shape {
        Some(shape) => {
            w.bool(true)?;
            write_shape(w, shape)?;
        }
        None => w.bool(false)?,
    }
    write_shape(w, &plan.output_shape)?;
    w.usize(plan.batch)?;
    w.usize(plan.input_channels)?;
    w.usize(plan.output_channels)?;
    w.usize(plan.height)?;
    w.usize(plan.width)?;
    w.u64(plan.cache_key)
}

fn read_static_conv2d(r: &mut Reader<'_>) -> Result<StaticConv2dPlan, ArtifactError> {
    let node = |id| {
        usize::try_from(id)
            .map(NodeId::from_index)
            .map_err(|_| ArtifactError::Format("static conv node"))
    };
    let input = node(r.u64()?)?;
    let weight = node(r.u64()?)?;
    let bias = if r.bool()? {
        Some(node(r.u64()?)?)
    } else {
        None
    };
    let output = node(r.u64()?)?;
    let input_shape = read_shape(r)?;
    let weight_shape = read_shape(r)?;
    let bias_shape = if r.bool()? {
        Some(read_shape(r)?)
    } else {
        None
    };
    let plan = StaticConv2dPlan {
        input,
        weight,
        bias,
        output,
        input_shape,
        weight_shape,
        bias_shape,
        output_shape: read_shape(r)?,
        batch: r.usize()?,
        input_channels: r.usize()?,
        output_channels: r.usize()?,
        height: r.usize()?,
        width: r.usize()?,
        cache_key: r.u64()?,
    };
    plan.validate()
        .map_err(|_| ArtifactError::Format("static conv2d plan"))?;
    Ok(plan)
}

fn write_quantized_desc(w: &mut Writer, desc: &QuantizedBufferDesc) -> Result<(), ArtifactError> {
    desc.validate_metadata()
        .map_err(|_| ArtifactError::Format("quantized descriptor"))?;
    w.u32(desc.ggml_type.raw())?;
    write_shape(w, &desc.logical_shape)?;
    w.usize(desc.block_elements)?;
    w.usize(desc.block_bytes)?;
    w.usize(desc.bytes)?;
    w.usize(desc.alignment)?;
    w.u64(desc.identity)
}

fn read_quantized_desc(r: &mut Reader<'_>) -> Result<QuantizedBufferDesc, ArtifactError> {
    let ggml_type = match r.u32()? {
        2 => GgmlType::Q4_0,
        8 => GgmlType::Q8_0,
        12 => GgmlType::Q4K,
        14 => GgmlType::Q6K,
        _ => return Err(ArtifactError::Format("quantized type")),
    };
    let desc = QuantizedBufferDesc {
        ggml_type,
        logical_shape: read_shape(r)?,
        block_elements: r.usize()?,
        block_bytes: r.usize()?,
        bytes: r.usize()?,
        alignment: r.usize()?,
        identity: r.u64()?,
    };
    desc.validate_metadata()
        .map_err(|_| ArtifactError::Format("quantized descriptor"))?;
    Ok(desc)
}

fn write_quantized_matmul(w: &mut Writer, plan: &QuantizedMatmulPlan) -> Result<(), ArtifactError> {
    plan.validate()
        .map_err(|_| ArtifactError::Format("quantized matmul plan"))?;
    w.u64(plan.activation.index() as u64)?;
    w.u64(plan.weight.index() as u64)?;
    w.u64(plan.output.index() as u64)?;
    write_shape(w, &plan.activation_shape)?;
    write_quantized_desc(w, &plan.weight_desc)?;
    write_shape(w, &plan.output_shape)?;
    w.u8(dtype_tag(plan.activation_dtype))?;
    w.u8(dtype_tag(plan.output_dtype))?;
    w.u8(match plan.orientation {
        QuantizedMatmulOrientation::OutputByInput => 0,
    })?;
    w.usizes(&plan.batch_shape)?;
    w.usize(plan.m)?;
    w.usize(plan.n)?;
    w.usize(plan.k)?;
    w.bool(plan.activation_vector)?;
    w.u64(plan.cache_key)
}

fn read_quantized_matmul(r: &mut Reader<'_>) -> Result<QuantizedMatmulPlan, ArtifactError> {
    let node = |id| {
        usize::try_from(id)
            .map(NodeId::from_index)
            .map_err(|_| ArtifactError::Format("quantized matmul node"))
    };
    let plan = QuantizedMatmulPlan {
        activation: node(r.u64()?)?,
        weight: node(r.u64()?)?,
        output: node(r.u64()?)?,
        activation_shape: read_shape(r)?,
        weight_desc: read_quantized_desc(r)?,
        output_shape: read_shape(r)?,
        activation_dtype: dtype(r.u8()?)?,
        output_dtype: dtype(r.u8()?)?,
        orientation: match r.u8()? {
            0 => QuantizedMatmulOrientation::OutputByInput,
            _ => return Err(ArtifactError::Format("quantized orientation")),
        },
        batch_shape: r.usizes()?,
        m: r.usize()?,
        n: r.usize()?,
        k: r.usize()?,
        activation_vector: r.bool()?,
        cache_key: r.u64()?,
    };
    plan.validate()
        .map_err(|_| ArtifactError::Format("quantized matmul plan"))?;
    Ok(plan)
}

fn write_quantized_row_gather(
    w: &mut Writer,
    plan: &QuantizedRowGatherPlan,
) -> Result<(), ArtifactError> {
    plan.validate()
        .map_err(|_| ArtifactError::Format("quantized row gather plan"))?;
    w.u64(plan.indices.index() as u64)?;
    w.u64(plan.weight.index() as u64)?;
    w.u64(plan.output.index() as u64)?;
    write_shape(w, &plan.indices_shape)?;
    w.u8(dtype_tag(plan.indices_dtype))?;
    write_quantized_desc(w, &plan.weight_desc)?;
    write_shape(w, &plan.output_shape)?;
    w.u8(dtype_tag(plan.output_dtype))?;
    w.u64(plan.cache_key)
}

fn read_quantized_row_gather(r: &mut Reader<'_>) -> Result<QuantizedRowGatherPlan, ArtifactError> {
    let node = |id| {
        usize::try_from(id)
            .map(NodeId::from_index)
            .map_err(|_| ArtifactError::Format("quantized row gather node"))
    };
    let plan = QuantizedRowGatherPlan {
        indices: node(r.u64()?)?,
        weight: node(r.u64()?)?,
        output: node(r.u64()?)?,
        indices_shape: read_shape(r)?,
        indices_dtype: dtype(r.u8()?)?,
        weight_desc: read_quantized_desc(r)?,
        output_shape: read_shape(r)?,
        output_dtype: dtype(r.u8()?)?,
        cache_key: r.u64()?,
    };
    plan.validate()
        .map_err(|_| ArtifactError::Format("quantized row gather plan"))?;
    Ok(plan)
}

fn write_operand(w: &mut Writer, operand: &MovementOperand) -> Result<(), ArtifactError> {
    w.u64(operand.node.index() as u64)?;
    write_shape(w, &operand.shape)?;
    w.u8(dtype_tag(operand.dtype))
}

fn read_operand(r: &mut Reader<'_>) -> Result<MovementOperand, ArtifactError> {
    Ok(MovementOperand {
        node: NodeId::from_index(
            usize::try_from(r.u64()?).map_err(|_| ArtifactError::Format("movement node"))?,
        ),
        shape: read_shape(r)?,
        dtype: dtype(r.u8()?)?,
    })
}

fn write_movement(w: &mut Writer, plan: &MovementKernelPlan) -> Result<(), ArtifactError> {
    plan.validate()
        .map_err(|_| ArtifactError::Format("movement plan"))?;
    match &plan.kind {
        MovementKernelKind::AffineCopy { input, view } => {
            w.u8(3)?;
            write_operand(w, input)?;
            write_affine_view(w, view)?;
        }
        MovementKernelKind::Pad {
            input,
            padding,
            fill_bits,
        } => {
            w.u8(4)?;
            write_operand(w, input)?;
            w.u32(
                u32::try_from(padding.len())
                    .map_err(|_| ArtifactError::Format("movement padding"))?,
            )?;
            for (before, after) in padding {
                w.usize(*before)?;
                w.usize(*after)?;
            }
            w.u64(*fill_bits)?;
        }
        MovementKernelKind::Concat { inputs, axis } => {
            w.u8(0)?;
            w.u32(
                u32::try_from(inputs.len())
                    .map_err(|_| ArtifactError::Format("movement inputs"))?,
            )?;
            for input in inputs {
                write_operand(w, input)?;
            }
            w.usize(*axis)?;
        }
        MovementKernelKind::Gather { input, index, axis } => {
            w.u8(1)?;
            write_operand(w, input)?;
            write_operand(w, index)?;
            w.usize(*axis)?;
        }
        MovementKernelKind::Scatter {
            base,
            index,
            updates,
            axis,
            add,
        } => {
            w.u8(2)?;
            write_operand(w, base)?;
            write_operand(w, index)?;
            write_operand(w, updates)?;
            w.usize(*axis)?;
            w.bool(*add)?;
        }
        MovementKernelKind::ScatterPositions {
            input,
            starts,
            steps,
        } => {
            w.u8(7)?;
            write_operand(w, input)?;
            w.u32(
                u32::try_from(starts.len())
                    .map_err(|_| ArtifactError::Format("movement positions"))?,
            )?;
            for start in starts {
                w.i64(*start)?;
            }
            w.u32(
                u32::try_from(steps.len())
                    .map_err(|_| ArtifactError::Format("movement positions"))?,
            )?;
            for step in steps {
                w.i64(*step)?;
            }
        }
        MovementKernelKind::Bitcast { input } => {
            w.u8(5)?;
            write_operand(w, input)?;
        }
        MovementKernelKind::Contiguous { input } => {
            w.u8(6)?;
            write_operand(w, input)?;
        }
    }
    w.u64(plan.output.index() as u64)?;
    write_shape(w, &plan.output_shape)?;
    w.u8(dtype_tag(plan.dtype))?;
    w.u64(plan.cache_key)
}

fn read_movement(r: &mut Reader<'_>, version: u8) -> Result<MovementKernelPlan, ArtifactError> {
    let kind = match r.u8()? {
        3 => MovementKernelKind::AffineCopy {
            input: read_operand(r)?,
            view: read_affine_view(r)?,
        },
        4 => {
            let input = read_operand(r)?;
            let count = r.count(MAX_COLLECTION)?;
            let mut padding = Vec::with_capacity(count);
            for _ in 0..count {
                padding.push((r.usize()?, r.usize()?));
            }
            MovementKernelKind::Pad {
                input,
                padding,
                fill_bits: r.u64()?,
            }
        }
        0 => {
            let count = r.count(MAX_COLLECTION)?;
            let mut inputs = Vec::with_capacity(count);
            for _ in 0..count {
                inputs.push(read_operand(r)?);
            }
            MovementKernelKind::Concat {
                inputs,
                axis: r.usize()?,
            }
        }
        1 => MovementKernelKind::Gather {
            input: read_operand(r)?,
            index: read_operand(r)?,
            axis: r.usize()?,
        },
        2 => MovementKernelKind::Scatter {
            base: read_operand(r)?,
            index: read_operand(r)?,
            updates: read_operand(r)?,
            axis: r.usize()?,
            add: r.bool()?,
        },
        5 => MovementKernelKind::Bitcast {
            input: read_operand(r)?,
        },
        6 => MovementKernelKind::Contiguous {
            input: read_operand(r)?,
        },
        7 if version >= 20 => {
            let input = read_operand(r)?;
            let starts = (0..r.count(MAX_COLLECTION)?)
                .map(|_| r.i64())
                .collect::<Result<Vec<_>, _>>()?;
            let steps = (0..r.count(MAX_COLLECTION)?)
                .map(|_| r.i64())
                .collect::<Result<Vec<_>, _>>()?;
            MovementKernelKind::ScatterPositions {
                input,
                starts,
                steps,
            }
        }
        _ => return Err(ArtifactError::Format("movement kind")),
    };
    let plan = MovementKernelPlan {
        kind,
        output: NodeId::from_index(
            usize::try_from(r.u64()?).map_err(|_| ArtifactError::Format("movement output"))?,
        ),
        output_shape: read_shape(r)?,
        dtype: dtype(r.u8()?)?,
        cache_key: r.u64()?,
    };
    plan.validate()
        .map_err(|_| ArtifactError::Format("movement plan"))?;
    Ok(plan)
}

pub(crate) fn write_shape(w: &mut Writer, x: &Shape) -> Result<(), ArtifactError> {
    w.usizes(x.dims())
}
pub(crate) fn read_shape(r: &mut Reader<'_>) -> Result<Shape, ArtifactError> {
    let x = Shape::new(r.usizes()?);
    checked_shape(&x)?;
    Ok(x)
}
pub(crate) fn write_view(w: &mut Writer, x: &ViewMap) -> Result<(), ArtifactError> {
    validate_view(x)?;
    write_shape(w, &x.source_shape)?;
    write_shape(w, &x.logical_shape)?;
    w.usizes(&x.strides)?;
    w.usize(x.offset)
}
pub(crate) fn read_view(r: &mut Reader<'_>) -> Result<ViewMap, ArtifactError> {
    let x = ViewMap {
        source_shape: read_shape(r)?,
        logical_shape: read_shape(r)?,
        strides: r.usizes()?,
        offset: r.usize()?,
    };
    validate_view(&x)?;
    Ok(x)
}
pub(crate) fn write_affine_view(w: &mut Writer, x: &AffineView) -> Result<(), ArtifactError> {
    x.validate_read()
        .map_err(|_| ArtifactError::Format("affine view"))?;
    write_shape(w, &x.source_shape)?;
    write_shape(w, &x.logical_shape)?;
    w.usize(x.strides.len())?;
    for stride in &x.strides {
        w.i64(*stride)?;
    }
    w.i64(x.offset)
}
pub(crate) fn read_affine_view(r: &mut Reader<'_>) -> Result<AffineView, ArtifactError> {
    let source_shape = read_shape(r)?;
    let logical_shape = read_shape(r)?;
    let count = r.usize()?;
    if count > MAX_COLLECTION {
        return Err(ArtifactError::Format("affine stride count"));
    }
    let mut strides = Vec::with_capacity(count);
    for _ in 0..count {
        strides.push(r.i64()?);
    }
    let x = AffineView {
        source_shape,
        logical_shape,
        strides,
        offset: r.i64()?,
    };
    x.validate_read()
        .map_err(|_| ArtifactError::Format("affine view"))?;
    Ok(x)
}

pub(crate) fn write_buffer_state(
    w: &mut Writer,
    state: &crate::BufferState,
) -> Result<(), ArtifactError> {
    crate::effects::validate_buffer_state(state)
        .map_err(|_| ArtifactError::Format("effect state"))?;
    w.u64(state.buffer)?;
    w.u64(state.version)?;
    write_shape(w, &state.shape)?;
    w.u8(dtype_tag(state.dtype))?;
    w.usize(state.bytes)
}

pub(crate) fn read_buffer_state(r: &mut Reader<'_>) -> Result<crate::BufferState, ArtifactError> {
    let state = crate::BufferState {
        buffer: r.u64()?,
        version: r.u64()?,
        shape: read_shape(r)?,
        dtype: dtype(r.u8()?)?,
        bytes: r.usize()?,
    };
    crate::effects::validate_buffer_state(&state)
        .map_err(|_| ArtifactError::Format("effect state"))?;
    Ok(state)
}

pub(crate) fn write_effect_payload(
    w: &mut Writer,
    payload: &crate::EffectPayload,
) -> Result<(), ArtifactError> {
    w.u64(payload.step)?;
    write_buffer_state(w, &payload.target)?;
    write_buffer_state(w, &payload.source)?;
    write_buffer_state(w, &payload.snapshot)?;
    w.bool(payload.target_view.is_some())?;
    if let Some(view) = &payload.target_view {
        view.validate_write()
            .map_err(|_| ArtifactError::Format("effect target view"))?;
        write_affine_view(w, view)?;
    }
    w.bool(payload.index_plan.is_some())?;
    if let Some(plan) = &payload.index_plan {
        write_static_index_plan(w, plan)?;
    }
    Ok(())
}

pub(crate) fn read_effect_payload(
    r: &mut Reader<'_>,
    has_index_plan: bool,
) -> Result<crate::EffectPayload, ArtifactError> {
    let payload = crate::EffectPayload {
        step: r.u64()?,
        target: read_buffer_state(r)?,
        source: read_buffer_state(r)?,
        snapshot: read_buffer_state(r)?,
        target_view: if r.bool()? {
            Some(read_affine_view(r)?)
        } else {
            None
        },
        index_plan: if has_index_plan && r.bool()? {
            Some(read_static_index_plan(r)?)
        } else {
            None
        },
    };
    crate::effects::validate_effect_payload(&payload)
        .map_err(|_| ArtifactError::Format("effect payload"))?;
    Ok(payload)
}

fn write_static_index_plan(
    w: &mut Writer,
    plan: &crate::ir::indexing::StaticIndexPlan,
) -> Result<(), ArtifactError> {
    write_shape(w, plan.source_shape())?;
    write_shape(w, plan.output_shape())?;
    let offsets = plan
        .source_offsets()
        .map_err(|_| ArtifactError::Format("static index plan"))?;
    if offsets.len() > MAX_COLLECTION {
        return Err(ArtifactError::Format("static index count"));
    }
    w.u32(offsets.len() as u32)?;
    for offset in offsets {
        w.usize(offset)?;
    }
    Ok(())
}

fn read_static_index_plan(
    r: &mut Reader<'_>,
) -> Result<crate::ir::indexing::StaticIndexPlan, ArtifactError> {
    let source = read_shape(r)?;
    let output = read_shape(r)?;
    let count = r.count(MAX_COLLECTION)?;
    let offsets = (0..count)
        .map(|_| r.usize())
        .collect::<Result<Vec<_>, _>>()?;
    crate::ir::indexing::StaticIndexPlan::from_offsets(source, output, offsets)
        .map_err(|_| ArtifactError::Format("static index plan"))
}

pub(crate) fn write_symbolic(
    w: &mut Writer,
    x: &SymbolicExpr,
    depth: usize,
) -> Result<(), ArtifactError> {
    if depth >= MAX_SYMBOLIC_DEPTH {
        return Err(ArtifactError::Format("symbolic depth"));
    }
    use SymbolicExpr::*;
    match x {
        Const(v) => {
            w.u8(0)?;
            w.i64(*v)
        }
        Var(v) => {
            let (min, max) = v.bounds();
            w.u8(1)?;
            w.u64(v.id())?;
            w.string(v.name())?;
            w.i64(min)?;
            w.i64(max)
        }
        Add(xs) | Mul(xs) => {
            w.u8(if matches!(x, Add(_)) { 2 } else { 3 })?;
            w.u32_len(xs.len())?;
            for y in xs {
                write_symbolic(w, y, depth + 1)?;
            }
            Ok(())
        }
        Neg(a) | Not(a) => {
            w.u8(if matches!(x, Neg(_)) { 4 } else { 14 })?;
            write_symbolic(w, a, depth + 1)
        }
        FloorDiv(a, b)
        | Mod(a, b)
        | Min(a, b)
        | Max(a, b)
        | Eq(a, b)
        | Lt(a, b)
        | Le(a, b)
        | And(a, b)
        | Or(a, b) => {
            let tag = match x {
                FloorDiv(..) => 5,
                Mod(..) => 6,
                Min(..) => 7,
                Max(..) => 8,
                Eq(..) => 9,
                Lt(..) => 10,
                Le(..) => 11,
                And(..) => 12,
                Or(..) => 13,
                _ => unreachable!(),
            };
            w.u8(tag)?;
            write_symbolic(w, a, depth + 1)?;
            write_symbolic(w, b, depth + 1)
        }
        Where(a, b, c) => {
            w.u8(15)?;
            write_symbolic(w, a, depth + 1)?;
            write_symbolic(w, b, depth + 1)?;
            write_symbolic(w, c, depth + 1)
        }
    }
}
pub(crate) fn read_symbolic(
    r: &mut Reader<'_>,
    depth: usize,
) -> Result<SymbolicExpr, ArtifactError> {
    if depth >= MAX_SYMBOLIC_DEPTH {
        return Err(ArtifactError::Format("symbolic depth"));
    }
    use SymbolicExpr::*;
    let tag = r.u8()?;
    Ok(match tag {
        0 => Const(r.i64()?),
        1 => {
            let id = r.u64()?;
            let name = r.string()?;
            let min = r.i64()?;
            let max = r.i64()?;
            Var(crate::SymbolicVar::from_artifact(id, name, min, max)
                .map_err(|_| ArtifactError::Format("symbolic variable"))?)
        }
        2 | 3 => {
            let n = r.count(MAX_COLLECTION)?;
            let mut xs = Vec::with_capacity(n);
            for _ in 0..n {
                xs.push(read_symbolic(r, depth + 1)?);
            }
            if tag == 2 { Add(xs) } else { Mul(xs) }
        }
        4 => Neg(Box::new(read_symbolic(r, depth + 1)?)),
        tag @ 5..=13 => {
            let a = Box::new(read_symbolic(r, depth + 1)?);
            let b = Box::new(read_symbolic(r, depth + 1)?);
            match tag {
                5 => FloorDiv(a, b),
                6 => Mod(a, b),
                7 => Min(a, b),
                8 => Max(a, b),
                9 => Eq(a, b),
                10 => Lt(a, b),
                11 => Le(a, b),
                12 => And(a, b),
                13 => Or(a, b),
                _ => unreachable!(),
            }
        }
        14 => Not(Box::new(read_symbolic(r, depth + 1)?)),
        15 => Where(
            Box::new(read_symbolic(r, depth + 1)?),
            Box::new(read_symbolic(r, depth + 1)?),
            Box::new(read_symbolic(r, depth + 1)?),
        ),
        _ => return Err(ArtifactError::Format("symbolic tag")),
    })
}

pub(crate) fn checksum(x: &[u8]) -> u32 {
    x.iter().fold(0x811c9dc5u32, |h, b| {
        (h ^ u32::from(*b)).wrapping_mul(0x01000193)
    })
}
pub(crate) fn dtype_tag(d: DType) -> u8 {
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
        DType::F8E4M3 => 13,
        DType::F8E5M2 => 14,
        DType::F8E4M3FNUZ => 15,
        DType::F8E5M2FNUZ => 16,
    }
}
pub(crate) fn dtype(t: u8) -> Result<DType, ArtifactError> {
    Ok(match t {
        0 => DType::Bool,
        1 => DType::I8,
        2 => DType::U8,
        3 => DType::I16,
        4 => DType::U16,
        13 => DType::F8E4M3,
        14 => DType::F8E5M2,
        15 => DType::F8E4M3FNUZ,
        16 => DType::F8E5M2FNUZ,
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
macro_rules! enum_codec{($encode:ident,$decode:ident,$ty:ty,[$($v:path),+$(,)?])=>{
    fn $encode(value:$ty)->u8{let all=[$($v),+];all.iter().position(|x|*x==value).expect("exhaustive enum codec") as u8}
    fn $decode(t:u8)->Result<$ty,ArtifactError>{let all=[$($v),+];all.get(t as usize).copied().ok_or(ArtifactError::Format("enum tag"))}
};}
enum_codec!(
    tag_unary,
    enum_unary,
    Unary,
    [Unary::Neg, Unary::Not, Unary::Abs]
);
enum_codec!(
    tag_binary,
    enum_binary,
    Binary,
    [
        Binary::Add,
        Binary::Sub,
        Binary::Mul,
        Binary::FloorDiv,
        Binary::Mod,
        Binary::Min,
        Binary::Max,
        Binary::Eq,
        Binary::Lt,
        Binary::Le,
        Binary::And,
        Binary::Or
    ]
);
enum_codec!(
    tag_graph_unary,
    enum_graph_unary,
    UnaryOp,
    [
        UnaryOp::Neg,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Relu,
        UnaryOp::Step,
        UnaryOp::Abs,
        UnaryOp::Reciprocal,
        UnaryOp::Square,
        UnaryOp::Sqrt,
        UnaryOp::Rsqrt,
        UnaryOp::Exp2,
        UnaryOp::Log2,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Tan,
        UnaryOp::Sinh,
        UnaryOp::Cosh,
        UnaryOp::Tanh,
        UnaryOp::Erf,
        UnaryOp::Erfc,
        UnaryOp::Asin,
        UnaryOp::Acos,
        UnaryOp::Atan,
        UnaryOp::Asinh,
        UnaryOp::Acosh,
        UnaryOp::Atanh,
        UnaryOp::Floor,
        UnaryOp::Ceil,
        UnaryOp::Trunc,
        UnaryOp::Round,
        UnaryOp::Sign,
        UnaryOp::IsNan,
        UnaryOp::IsInf,
        UnaryOp::IsFinite
    ]
);
enum_codec!(
    tag_graph_binary,
    enum_graph_binary,
    BinaryOp,
    [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Pow,
        BinaryOp::Maximum,
        BinaryOp::Minimum,
        BinaryOp::FloorDiv,
        BinaryOp::TruncDiv,
        BinaryOp::Mod,
        BinaryOp::FMod,
        BinaryOp::BitAnd,
        BinaryOp::BitOr,
        BinaryOp::BitXor,
        BinaryOp::Shl,
        BinaryOp::Shr,
        BinaryOp::Atan2,
        BinaryOp::Copysign
    ]
);
enum_codec!(
    tag_compare,
    enum_compare,
    CompareOp,
    [
        CompareOp::Eq,
        CompareOp::Ne,
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Gt,
        CompareOp::Ge
    ]
);
enum_codec!(
    tag_logical,
    enum_logical,
    LogicalOp,
    [LogicalOp::Not, LogicalOp::And, LogicalOp::Or]
);
enum_codec!(
    tag_reduce,
    enum_reduce,
    ReduceKind,
    [
        ReduceKind::Sum,
        ReduceKind::Mean,
        ReduceKind::Product,
        ReduceKind::Max,
        ReduceKind::Min,
        ReduceKind::Any,
        ReduceKind::All
    ]
);
enum_codec!(
    tag_prefix_scan,
    enum_prefix_scan,
    crate::PrefixScanKind,
    [
        crate::PrefixScanKind::Sum,
        crate::PrefixScanKind::Product,
        crate::PrefixScanKind::Max,
        crate::PrefixScanKind::Min
    ]
);
enum_codec!(
    tag_prefix_scan_output,
    enum_prefix_scan_output,
    crate::PrefixScanOutput,
    [
        crate::PrefixScanOutput::Values,
        crate::PrefixScanOutput::Indices
    ]
);

pub(crate) struct Writer {
    pub(crate) out: Vec<u8>,
}
impl Writer {
    pub(crate) fn new() -> Self {
        Self { out: Vec::new() }
    }
    fn reserve(&self, n: usize) -> Result<(), ArtifactError> {
        if self.out.len().checked_add(n).is_none_or(|x| x > MAX_BYTES) {
            Err(ArtifactError::Format("byte limit"))
        } else {
            Ok(())
        }
    }
    pub(crate) fn bytes(&mut self, x: &[u8]) -> Result<(), ArtifactError> {
        self.reserve(x.len())?;
        self.out.extend(x);
        Ok(())
    }
    pub(crate) fn u8(&mut self, x: u8) -> Result<(), ArtifactError> {
        self.bytes(&[x])
    }
    pub(crate) fn bool(&mut self, x: bool) -> Result<(), ArtifactError> {
        self.u8(u8::from(x))
    }
    pub(crate) fn u16(&mut self, x: u16) -> Result<(), ArtifactError> {
        self.bytes(&x.to_le_bytes())
    }
    pub(crate) fn u32(&mut self, x: u32) -> Result<(), ArtifactError> {
        self.bytes(&x.to_le_bytes())
    }
    pub(crate) fn u64(&mut self, x: u64) -> Result<(), ArtifactError> {
        self.bytes(&x.to_le_bytes())
    }
    pub(crate) fn i64(&mut self, x: i64) -> Result<(), ArtifactError> {
        self.bytes(&x.to_le_bytes())
    }
    pub(crate) fn usize(&mut self, x: usize) -> Result<(), ArtifactError> {
        self.u64(x as u64)
    }
    fn u32_len(&mut self, n: usize) -> Result<(), ArtifactError> {
        if n > MAX_COLLECTION || n > u32::MAX as usize {
            Err(ArtifactError::Format("collection limit"))
        } else {
            self.u32(n as u32)
        }
    }
    pub(crate) fn string(&mut self, x: &str) -> Result<(), ArtifactError> {
        if x.len() > MAX_STRING {
            return Err(ArtifactError::Format("string limit"));
        }
        self.u32(x.len() as u32)?;
        self.bytes(x.as_bytes())
    }
    pub(crate) fn usizes(&mut self, x: &[usize]) -> Result<(), ArtifactError> {
        self.u32_len(x.len())?;
        for v in x {
            self.usize(*v)?;
        }
        Ok(())
    }
}
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
    last: u8,
}
impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            last: 0,
        }
    }
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], ArtifactError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(ArtifactError::Format("overflow"))?;
        let x = self
            .bytes
            .get(self.pos..end)
            .ok_or(ArtifactError::Format("truncated"))?;
        self.pos = end;
        Ok(x)
    }
    pub(crate) fn u8(&mut self) -> Result<u8, ArtifactError> {
        let x = self.take(1)?[0];
        self.last = x;
        Ok(x)
    }
    pub(crate) fn bool(&mut self) -> Result<bool, ArtifactError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(ArtifactError::Format("bool")),
        }
    }
    pub(crate) fn u16(&mut self) -> Result<u16, ArtifactError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub(crate) fn u32(&mut self) -> Result<u32, ArtifactError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(crate) fn u64(&mut self) -> Result<u64, ArtifactError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub(crate) fn i64(&mut self) -> Result<i64, ArtifactError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub(crate) fn usize(&mut self) -> Result<usize, ArtifactError> {
        usize::try_from(self.u64()?).map_err(|_| ArtifactError::Format("usize"))
    }
    pub(crate) fn count(&mut self, max: usize) -> Result<usize, ArtifactError> {
        let x = self.u32()? as usize;
        if x > max {
            Err(ArtifactError::Format("count limit"))
        } else {
            Ok(x)
        }
    }
    pub(crate) fn string(&mut self) -> Result<String, ArtifactError> {
        let n = self.count(MAX_STRING)?;
        let x = self.take(n)?;
        let s = std::str::from_utf8(x).map_err(|_| ArtifactError::Format("utf8"))?;
        Ok(s.to_owned())
    }
    pub(crate) fn usizes(&mut self) -> Result<Vec<usize>, ArtifactError> {
        let n = self.count(MAX_COLLECTION)?;
        let mut x = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(self.usize()?);
        }
        Ok(x)
    }
    pub(crate) fn done(&self) -> bool {
        self.pos == self.bytes.len()
    }
    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finish(mut w: Writer) -> Vec<u8> {
        let sum = checksum(&w.out);
        w.u32(sum).unwrap();
        w.out
    }

    fn header(count: u32, root: u32) -> Writer {
        let mut w = Writer::new();
        w.bytes(MAGIC).unwrap();
        w.u8(VERSION).unwrap();
        w.u32(count).unwrap();
        w.u32(root).unwrap();
        w
    }

    #[test]
    fn reduction_mean_wire_projection_is_private_and_fail_closed() {
        let reduction_arg = |kind, mean| WireArg::Reduction {
            input_shape: Shape::from([2]),
            output_shape: Shape::new([]),
            axes: vec![0],
            keepdim: false,
            kind,
            mean,
        };
        for (kind, mean) in [
            (crate::ReduceKind::Sum, false),
            (crate::ReduceKind::Mean, true),
        ] {
            let operation =
                operation_from_wire(WireOpcode::ReduceInit, reduction_arg(kind, mean)).unwrap();
            let (_, wire) = operation_to_wire(&operation);
            assert_eq!(wire, reduction_arg(kind, mean));
        }
        for (kind, mean) in [
            (crate::ReduceKind::Sum, true),
            (crate::ReduceKind::Mean, false),
        ] {
            assert!(matches!(
                operation_from_wire(WireOpcode::ReduceInit, reduction_arg(kind, mean)),
                Err(ArtifactError::Format("reduction mean projection"))
            ));
        }
    }

    #[test]
    fn typed_reduction_schedule_identity_preserves_the_released_v18_envelope() {
        let ty = UType::scalar(DType::F32);
        let source = UOp::from_operation(Operation::Const(LiteralValue::Int(1)), Some(ty), vec![]);
        let init = UOp::from_operation(
            Operation::ReduceInit(ReductionValue {
                input_shape: Shape::from([2]),
                output_shape: Shape::new([]),
                axes: vec![0],
                keepdim: false,
                kind: crate::ReduceKind::Sum,
            }),
            Some(ty),
            vec![],
        );
        let update = UOp::from_operation(Operation::ReduceAccumulate, Some(ty), vec![init, source]);
        let finalize = UOp::from_operation(Operation::ReduceFinalize, Some(ty), vec![update]);
        assert_eq!(encode_schedule_identity(&finalize).unwrap()[4], 18);
    }

    #[test]
    fn v18_reduction_preserves_distinct_accumulator_and_output_types() {
        for output in [DType::F16, DType::BF16] {
            let source_ty = UType::scalar(output);
            let accumulator_ty = UType::scalar(DType::F32);
            let source = UOp::from_operation(
                Operation::Const(LiteralValue::Scalar {
                    dtype: output,
                    bits: 0,
                }),
                Some(source_ty),
                vec![],
            );
            let init = UOp::from_operation(
                Operation::ReduceInit(ReductionValue {
                    input_shape: Shape::from([2]),
                    output_shape: Shape::new([]),
                    axes: vec![0],
                    keepdim: false,
                    kind: crate::ReduceKind::Sum,
                }),
                Some(accumulator_ty),
                vec![],
            );
            let update = UOp::from_operation(
                Operation::ReduceAccumulate,
                Some(accumulator_ty),
                vec![init, source],
            );
            let finalize =
                UOp::from_operation(Operation::ReduceFinalize, Some(source_ty), vec![update]);
            let bytes = encode_schedule_identity(&finalize).unwrap();
            assert_eq!(bytes[4], 18);
            let decoded = decode(&bytes).unwrap();
            assert_eq!(decoded, finalize);
            let (plan, _) =
                crate::reduction_native::NativeReductionPlan::from_finalize(&decoded).unwrap();
            assert_eq!(plan.accumulator_dtype, DType::F32);
            assert_eq!(plan.output_dtype, output);

            let decoded_update = decoded.sources().first().unwrap();
            let tampered_update = UOp::from_operation(
                Operation::ReduceAccumulate,
                Some(UType::scalar(output)),
                decoded_update.sources().to_vec(),
            );
            let tampered = UOp::from_operation(
                Operation::ReduceFinalize,
                Some(source_ty),
                vec![tampered_update],
            );
            assert!(encode_schedule_identity(&tampered).is_err());
        }
    }

    #[test]
    fn v18_legacy_float8_mean_same_storage_round_trips_and_replays() {
        let dtype = DType::F8E5M2;
        let ty = UType::scalar(dtype);
        let source = UOp::from_operation(
            Operation::Const(LiteralValue::Scalar { dtype, bits: 0 }),
            Some(ty),
            vec![],
        );
        let init = UOp::from_operation(
            Operation::ReduceInit(ReductionValue {
                input_shape: Shape::from([3]),
                output_shape: Shape::new([]),
                axes: vec![0],
                keepdim: false,
                kind: crate::ReduceKind::Mean,
            }),
            Some(ty),
            vec![],
        );
        let update = UOp::from_operation(Operation::ReduceAccumulate, Some(ty), vec![init, source]);
        let finalize = UOp::from_operation(Operation::ReduceFinalize, Some(ty), vec![update]);
        let bytes = encode_schedule_identity(&finalize).unwrap();
        assert_eq!(bytes[4], 18);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded, finalize);
        let (plan, _) =
            crate::reduction_native::NativeReductionPlan::from_finalize(&decoded).unwrap();
        let accumulator = [8.0, 1.0, -8.0]
            .into_iter()
            .fold(plan.identity(), |acc, value| {
                plan.update(acc, crate::Scalar::F(value))
            });
        let expected = dtype.commit_scalar(crate::Scalar::F(f64::from(
            dtype.commit_scalar(accumulator).as_f64() as f32
                / dtype.commit_scalar(crate::Scalar::F(3.0)).as_f64() as f32,
        )));
        assert_eq!(plan.finalize(accumulator), expected);
    }

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
            assert_eq!(x, decode(&a).unwrap());
        }
    }
    #[test]
    fn tensor_guard_v17_round_trip_and_legacy_exclusion_are_exact() {
        let guard = UOp::from_operation(
            Operation::TensorGuard(TensorGuardValue {
                input: crate::NodeId::from_index(3),
                input_shape: Shape::from([2, 3]),
                axis: 1,
                dtype: DType::F32,
            }),
            Some(UType::scalar(DType::F32)),
            vec![],
        );
        let bytes = encode(&guard).unwrap();
        assert_eq!(bytes, encode(&guard).unwrap());
        assert_eq!(decode(&bytes).unwrap(), guard);
        let mut legacy = bytes.clone();
        legacy[4] = 16;
        let sum = checksum(&legacy[..legacy.len() - 4]);
        let end = legacy.len();
        legacy[end - 4..].copy_from_slice(&sum.to_le_bytes());
        assert!(decode(&legacy).is_err());
    }

    #[test]
    fn prefix_scan_v18_owns_source_result_abi_and_legacy_payloads_fail_closed() {
        let mut graph = crate::Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::I16);
        let output = graph.cumsum(input, 1).unwrap();
        let scan = crate::lower_graph_prefix_scan(&graph, output).unwrap();
        let bytes = encode(&scan).unwrap();
        assert_eq!(bytes[4], 19);
        assert_eq!(decode(&bytes).unwrap(), scan);
        assert_eq!(encode_schedule_identity(&scan).unwrap()[4], 18);

        let mut v18 = bytes.clone();
        v18[4] = 18;
        let body_len = v18.len() - 4;
        let sum = checksum(&v18[..body_len]);
        v18[body_len..].copy_from_slice(&sum.to_le_bytes());
        assert_eq!(decode(&v18).unwrap(), scan);

        let mut legacy = bytes;
        legacy[4] = 17;
        let body_len = legacy.len() - 4;
        let sum = checksum(&legacy[..body_len]);
        legacy[body_len..].copy_from_slice(&sum.to_le_bytes());
        assert!(matches!(
            decode(&legacy),
            Err(ArtifactError::Format(
                "legacy prefix scan lacks source/result ABI"
            ))
        ));
    }

    #[test]
    fn live_threefry_v19_round_trip_and_tamper_rejection_are_exact() {
        let value = ThreefryValue {
            counter: NodeId::from_index(1),
            key: NodeId::from_index(2),
            counter_shape: Shape::from([2, 1]),
            key_shape: Shape::from([1, 3]),
            output: NodeId::from_index(3),
            output_shape: Shape::from([2, 3]),
        };
        let root = UOp::from_operation(
            Operation::Threefry(value.clone()),
            Some(UType::scalar(DType::U64)),
            vec![],
        );
        let bytes = encode(&root).unwrap();
        assert_eq!(bytes[4], 19);
        assert_eq!(encode_schedule_identity(&root).unwrap()[4], 19);
        assert_eq!(decode(&bytes).unwrap(), root);
        assert_eq!(encode(&decode(&bytes).unwrap()).unwrap(), bytes);

        assert!(read_kind(&mut Reader::new(&[39]), 18).is_err());
        assert_eq!(
            read_kind(&mut Reader::new(&[39]), 19).unwrap(),
            WireOpcode::Threefry
        );
        let (_, wire_arg) = operation_to_wire(root.operation());
        let mut writer = Writer::new();
        write_arg(&mut writer, &wire_arg, false).unwrap();
        assert!(read_arg(&mut Reader::new(&writer.out), 18).is_err());
        assert!(matches!(
            read_arg(&mut Reader::new(&writer.out), 19).unwrap(),
            WireArg::Threefry { .. }
        ));

        let malformed = UOp::from_operation(
            Operation::Threefry(ThreefryValue {
                output_shape: Shape::from([3, 2]),
                ..value
            }),
            Some(UType::scalar(DType::U64)),
            vec![],
        );
        assert!(encode(&malformed).is_err());

        let mut legacy = bytes;
        legacy[4] = 18;
        let sum = checksum(&legacy[..legacy.len() - 4]);
        let end = legacy.len();
        legacy[end - 4..].copy_from_slice(&sum.to_le_bytes());
        assert!(decode(&legacy).is_err());
    }
    #[test]
    fn shared_sources_remain_shared() {
        let x = UOp::constant(7, UType::scalar(DType::I64));
        let root = UOp::binary(Binary::Add, x.clone(), x);
        let decoded = decode(&encode(&root).unwrap()).unwrap();
        assert!(decoded.sources()[0].shares_node_with(&decoded.sources()[1]));
    }

    #[test]
    fn standard_envelope_retains_ordinary_impure_kernel_nodes() {
        let mut graph = crate::Graph::new();
        let input = graph.input_dtype("input", [4], DType::F32);
        let output = graph.relu(input).unwrap();
        let root = crate::lower_graph_elementwise(&graph, output).unwrap();
        let kinds = root
            .topological()
            .unwrap()
            .into_iter()
            .map(|node| operation_to_wire(node.operation()).0)
            .collect::<Vec<_>>();
        assert!(kinds.contains(&WireOpcode::Store));
        assert!(kinds.contains(&WireOpcode::Sink));
        let bytes = encode(&root).unwrap();
        assert_eq!(decode(&bytes).unwrap(), root);
    }
    #[test]
    fn matmul_payload_round_trip_and_validation_are_exact() {
        let mut graph = crate::Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 1, 2, 3], DType::F64);
        let rhs = graph.input_dtype("rhs", [1, 4, 3, 2], DType::F64);
        let output = graph.matmul(lhs, rhs).unwrap();
        let root = crate::lower_graph_matmul(&graph, output).unwrap();
        let bytes = encode(&root).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(bytes, encode(&decoded).unwrap());
        assert_eq!(root, decoded);
        let mut legacy_v4 = bytes.clone();
        legacy_v4[4] = 4;
        let body_len = legacy_v4.len() - 4;
        let sum = checksum(&legacy_v4[..body_len]);
        legacy_v4[body_len..].copy_from_slice(&sum.to_le_bytes());
        assert_eq!(decode(&legacy_v4).unwrap(), root);

        let mut legacy_version = bytes;
        legacy_version[4] = 2;
        let body_len = legacy_version.len() - 4;
        let sum = checksum(&legacy_version[..body_len]);
        legacy_version[body_len..].copy_from_slice(&sum.to_le_bytes());
        assert!(decode(&legacy_version).is_err());

        let Operation::Matmul(MatmulValue::Serial(plan)) = root.operation() else {
            panic!("matmul payload missing");
        };
        let mut malformed = plan.clone();
        malformed.k += 1;
        let malformed = UOp::from_operation(
            Operation::Matmul(MatmulValue::Serial(malformed)),
            Some(UType::scalar(DType::F64)),
            vec![],
        );
        assert!(malformed.validate().is_err());
        assert!(encode(&malformed).is_err());

        let mut tiled_graph = crate::Graph::new();
        let lhs = tiled_graph.input_dtype("lhs", [17, 9], DType::F32);
        let rhs = tiled_graph.input_dtype("rhs", [9, 13], DType::F32);
        let output = tiled_graph.matmul(lhs, rhs).unwrap();
        let tiled = crate::lower_graph_matmul(&tiled_graph, output).unwrap();
        let tiled_bytes = encode(&tiled).unwrap();
        assert_eq!(decode(&tiled_bytes).unwrap(), tiled);
        assert_eq!(encode(&decode(&tiled_bytes).unwrap()).unwrap(), tiled_bytes);
        let Operation::Matmul(MatmulValue::Tiled(payload)) = tiled.operation() else {
            panic!("tiled matmul payload missing");
        };
        let mut malformed = payload.clone();
        malformed.tile.barriers[0].uniform = false;
        let malformed = UOp::from_operation(
            Operation::Matmul(MatmulValue::Tiled(malformed)),
            Some(UType::scalar(DType::F32)),
            vec![],
        );
        assert!(malformed.validate().is_err());
        assert!(encode(&malformed).is_err());

        let mut misaligned = payload.clone();
        misaligned.tile.lhs_shared.alignment = 3;
        let misaligned = UOp::from_operation(
            Operation::Matmul(MatmulValue::Tiled(misaligned)),
            Some(UType::scalar(DType::F32)),
            vec![],
        );
        assert!(misaligned.validate().is_err());
        assert!(encode(&misaligned).is_err());

        let mut legacy_version = tiled_bytes;
        legacy_version[4] = 4;
        let body_len = legacy_version.len() - 4;
        let sum = checksum(&legacy_version[..body_len]);
        legacy_version[body_len..].copy_from_slice(&sum.to_le_bytes());
        assert!(decode(&legacy_version).is_err());
    }

    #[test]
    fn movement_payload_round_trip_and_validation_are_exact() {
        let mut graph = crate::Graph::new();
        let base = graph.input_dtype("base", [2, 3], DType::F32);
        let index = graph.input_dtype("index", [2, 2], DType::I64);
        let updates = graph.input_dtype("updates", [2, 2], DType::F32);
        let output = graph.scatter_add(base, index, updates, 1).unwrap();
        let root = crate::lower_graph_movement(&graph, output).unwrap();
        let bytes = encode(&root).unwrap();
        assert_eq!(bytes[4], 19);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(bytes, encode(&decoded).unwrap());
        assert_eq!(root, decoded);
        assert_eq!(encode_schedule_identity(&root).unwrap()[4], 18);

        let Operation::Movement(MovementValue::Plan(plan)) = root.operation() else {
            panic!("movement payload missing");
        };
        let mut malformed = plan.clone();
        malformed.output_shape = Shape::from([3, 2]);
        let malformed = UOp::from_operation(
            Operation::Movement(MovementValue::Plan(malformed)),
            Some(UType::scalar(DType::F32)),
            vec![],
        );
        assert!(malformed.validate().is_err());
        assert!(encode(&malformed).is_err());

        let mut pad_graph = crate::Graph::new();
        let input = pad_graph.input_dtype("input", [1, 2], DType::F16);
        let output = pad_graph
            .pad(input, [(1, 0), (0, 2)], crate::Scalar::F(-0.0))
            .unwrap();
        let root = crate::lower_graph_movement(&pad_graph, output).unwrap();
        let bytes = encode(&root).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(encode(&decoded).unwrap(), bytes);
        assert_eq!(decoded, root);
        let Operation::Movement(MovementValue::Plan(plan)) = root.operation() else {
            panic!("movement payload missing");
        };
        assert!(matches!(
            &plan.kind,
            MovementKernelKind::Pad {
                padding,
                fill_bits: 0x8000,
                ..
            } if padding.as_slice() == [(1, 0), (0, 2)]
        ));

        let mut bitcast_graph = crate::Graph::new();
        let input = bitcast_graph.input_dtype("input", [2, 8], DType::U8);
        let output = bitcast_graph.bitcast(input, DType::U32).unwrap();
        let root = crate::lower_graph_movement(&bitcast_graph, output).unwrap();
        let bytes = encode(&root).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(encode(&decoded).unwrap(), bytes);
        assert_eq!(decoded, root);
        let Operation::Movement(MovementValue::Plan(plan)) = root.operation() else {
            panic!("movement payload missing");
        };
        assert!(matches!(
            &plan.kind,
            MovementKernelKind::Bitcast { input: operand }
                if operand.node == input && operand.dtype == DType::U8
        ));
        assert_eq!(plan.output_shape, Shape::from([2, 2]));
        assert_eq!(plan.dtype, DType::U32);

        let mut positions_graph = crate::Graph::new();
        let input = positions_graph.input_dtype("input", [2], DType::F32);
        let output = positions_graph
            .scatter_positions(input, Shape::from([5]), vec![4], vec![-2])
            .unwrap();
        let root = crate::lower_graph_movement(&positions_graph, output).unwrap();
        let bytes = encode(&root).unwrap();
        assert_eq!(bytes[4], 20);
        assert_eq!(decode(&bytes).unwrap(), root);
        assert_eq!(encode_schedule_identity(&root).unwrap()[4], 20);

        let mut forged_v19 = bytes.clone();
        forged_v19[4] = 19;
        let body_len = forged_v19.len() - 4;
        let sum = checksum(&forged_v19[..body_len]);
        forged_v19[body_len..].copy_from_slice(&sum.to_le_bytes());
        assert!(decode(&forged_v19).is_err());

        let Operation::Movement(MovementValue::Plan(plan)) = root.operation() else {
            panic!("movement payload missing")
        };
        assert_eq!(plan.cache_key, 0xa069_ceb4_6f40_d7b4);
        let mut malformed = plan.clone();
        let MovementKernelKind::ScatterPositions { starts, .. } = &mut malformed.kind else {
            panic!("static-position movement")
        };
        starts[0] = 5;
        let malformed = UOp::from_operation(
            Operation::Movement(MovementValue::Plan(malformed)),
            Some(UType::scalar(DType::F32)),
            vec![],
        );
        assert!(malformed.validate().is_err());
        assert!(encode(&malformed).is_err());

        let mut malformed = plan.clone();
        let MovementKernelKind::ScatterPositions { steps, .. } = &mut malformed.kind else {
            panic!("static-position movement")
        };
        steps[0] = 0;
        malformed.cache_key = malformed.expected_cache_key();
        let malformed = UOp::from_operation(
            Operation::Movement(MovementValue::Plan(malformed)),
            Some(UType::scalar(DType::F32)),
            vec![],
        );
        assert!(malformed.validate().is_err());
        assert!(encode(&malformed).is_err());
    }

    #[test]
    fn established_and_signed_affine_copy_payloads_round_trip_deterministically() {
        let mut graph = crate::Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let producer = graph.relu(input).unwrap();
        let output = graph.reshape(producer, [1, 4]).unwrap();
        let root = crate::kernel::lower_graph_computed_affine_view(&graph, output).unwrap();
        let bytes = encode(&root).unwrap();
        assert_eq!(encode(&decode(&bytes).unwrap()).unwrap(), bytes);
        let Operation::Movement(MovementValue::Plan(plan)) = root.operation() else {
            panic!("movement payload missing");
        };
        assert!(matches!(plan.kind, MovementKernelKind::AffineCopy { .. }));

        let mut signed = crate::Graph::new();
        let input = signed.input_dtype("input", [4], DType::F32);
        let producer = signed.square(input).unwrap();
        let output = signed
            .stride(
                producer,
                [crate::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                }],
            )
            .unwrap();
        let root = crate::kernel::lower_graph_computed_affine_view(&signed, output).unwrap();
        let bytes = encode(&root).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(encode(&decoded).unwrap(), bytes);
        let Operation::Movement(MovementValue::Plan(plan)) = decoded.operation() else {
            panic!("movement payload missing")
        };
        let MovementKernelKind::AffineCopy { view, .. } = &plan.kind else {
            panic!("affine-copy payload missing")
        };
        assert_eq!(view.offset, 3);
        assert_eq!(view.strides, vec![-1]);
    }
    #[test]
    fn corruption_and_truncation_fail_closed() {
        let x = encode(&UOp::constant(1, UType::scalar(DType::I64))).unwrap();
        for n in 0..x.len() {
            assert!(decode(&x[..n]).is_err());
        }
        let mut bad = x;
        bad[4] = 99;
        let end = bad.len() - 4;
        let sum = checksum(&bad[..end]);
        bad[end..].copy_from_slice(&sum.to_le_bytes());
        assert!(decode(&bad).is_err());
    }

    #[test]
    fn malformed_node_tables_fail_closed() {
        let ty = Some(UType::scalar(DType::I64));

        let mut out_of_order = header(1, 0);
        out_of_order.u32(1).unwrap();
        write_kind(&mut out_of_order, &WireOpcode::Const, false).unwrap();
        write_type(&mut out_of_order, ty).unwrap();
        write_arg(&mut out_of_order, &WireArg::Int(1), false).unwrap();
        out_of_order.u32(0).unwrap();
        assert!(decode(&finish(out_of_order)).is_err());

        let mut forward = header(2, 1);
        forward.u32(0).unwrap();
        write_kind(&mut forward, &WireOpcode::Cast, false).unwrap();
        write_type(&mut forward, ty).unwrap();
        write_arg(&mut forward, &WireArg::None, false).unwrap();
        forward.u32(1).unwrap();
        forward.u32(0).unwrap();
        forward.u32(1).unwrap();
        write_kind(&mut forward, &WireOpcode::Const, false).unwrap();
        write_type(&mut forward, ty).unwrap();
        write_arg(&mut forward, &WireArg::Int(1), false).unwrap();
        forward.u32(0).unwrap();
        assert!(decode(&finish(forward)).is_err());

        let mut wrong_arg = header(1, 0);
        wrong_arg.u32(0).unwrap();
        write_kind(&mut wrong_arg, &WireOpcode::Const, false).unwrap();
        write_type(&mut wrong_arg, ty).unwrap();
        write_arg(&mut wrong_arg, &WireArg::None, false).unwrap();
        wrong_arg.u32(0).unwrap();
        assert!(decode(&finish(wrong_arg)).is_err());

        let mut count_limit = header(MAX_NODES as u32 + 1, 0);
        assert!(decode(&finish(count_limit)).is_err());
        count_limit = header(1, 0);
        count_limit.u32(0).unwrap();
        count_limit.u8(u8::MAX).unwrap();
        assert!(decode(&finish(count_limit)).is_err());

        let mut trailing = encode(&UOp::constant(1, UType::scalar(DType::I64))).unwrap();
        trailing.truncate(trailing.len() - 4);
        trailing.push(0);
        let sum = checksum(&trailing);
        trailing.extend(sum.to_le_bytes());
        assert!(decode(&trailing).is_err());
    }
}
