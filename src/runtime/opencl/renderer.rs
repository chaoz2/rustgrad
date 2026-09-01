//! Pure deterministic OpenCL C lowering for a deliberately small scalar/reduction UOp subset.
use super::{
    OpenClCapabilities, OpenClError,
    guard::{emit_transactional, emit_transactional_reduction},
    narrow,
    reduction::{
        input_offset_expression, required_capabilities as reduction_capabilities, validate_dtype,
    },
    transaction::{GuardedIntegerOp, OpenClGuardDomain, OpenClTransactionAbi},
    view::OpenClViewAccess,
};
use crate::{
    AffineView, DType, IndexValue, LiteralValue, Operation, ScheduleInputBinding, Shape, UOp,
    runtime::scalar_lane::{
        ScalarLaneDialect, dialect_seal, emit_scalar_lane, project_scalar_lane,
    },
};
use std::{
    collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::Arc,
};

pub const OPENCL_RENDERER_VERSION: &str = "rustgrad-opencl-static-v8";
pub const OPENCL_ABI_VERSION: u32 = 4;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct OpenClBufferAbi {
    pub id: u64,
    pub dtype: DType,
    pub source_shape: Shape,
    pub elements: usize,
    pub mutable: bool,
    pub view: Option<AffineView>,
}

#[derive(Clone, Debug)]
pub struct RenderedOpenCl {
    pub source: String,
    pub source_map: BTreeMap<usize, usize>,
    pub buffers: Vec<OpenClBufferAbi>,
    pub extent: usize,
    pub entry: String,
    pub cache_key: String,
    pub required_capabilities: OpenClCapabilities,
    pub transaction: Option<OpenClTransactionAbi>,
    pub(crate) schedule_inputs: Vec<OpenClBufferAbi>,
    pub(crate) semantic_program: Arc<super::dispatch::KernelSemanticProgram>,
}

impl RenderedOpenCl {
    /// Validates the schedule-owned first-use order against the pointer ABI.
    pub fn validate_schedule_bindings(
        &self,
        bindings: &[ScheduleInputBinding],
    ) -> Result<(), OpenClError> {
        if bindings.len() != self.schedule_inputs.len() {
            return Err(OpenClError::InvalidBinding(
                "schedule/OpenCL input count mismatch".into(),
            ));
        }
        for (index, (binding, expected)) in bindings.iter().zip(&self.schedule_inputs).enumerate() {
            if binding.abi_index != index
                || binding.desc.id != expected.id
                || binding.desc.dtype != expected.dtype
                || binding.desc.shape != expected.source_shape
                || binding.desc.view != expected.view
                || binding.desc.bytes
                    != expected
                        .elements
                        .checked_mul(expected.dtype.itemsize())
                        .ok_or(OpenClError::Overflow)?
            {
                return Err(OpenClError::InvalidBinding(format!(
                    "schedule binding {index} mismatches OpenCL ABI"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpenClRenderer {
    pub local_size: usize,
    pub capabilities: OpenClCapabilities,
}

impl Default for OpenClRenderer {
    fn default() -> Self {
        Self {
            local_size: 64,
            capabilities: OpenClCapabilities::CORE_32,
        }
    }
}

impl OpenClRenderer {
    pub fn new(local_size: usize) -> Result<Self, OpenClError> {
        if local_size == 0 {
            return Err(OpenClError::InvalidArgument("zero local size"));
        }
        Ok(Self {
            local_size,
            capabilities: OpenClCapabilities::CORE_32,
        })
    }

    pub fn with_capabilities(
        local_size: usize,
        capabilities: OpenClCapabilities,
    ) -> Result<Self, OpenClError> {
        if local_size == 0 {
            return Err(OpenClError::InvalidArgument("zero local size"));
        }
        Ok(Self {
            local_size,
            capabilities,
        })
    }

    pub fn render(&self, root: &UOp) -> Result<RenderedOpenCl, OpenClError> {
        if let Operation::Random(plan) = root.operation() {
            return super::random::render(self, plan);
        }
        if matches!(
            root.operation(),
            Operation::PrefixScan(_)
                | Operation::Sort(_)
                | Operation::TensorGuard(_)
                | Operation::Threefry(_)
        ) {
            return Err(OpenClError::Unsupported(
                "prefix scans, sort pairs, guards, and live Threefry are outside OpenCL lowering"
                    .into(),
            ));
        }
        root.validate()
            .map_err(|error| OpenClError::Unsupported(error.to_string()))?;
        let nodes = root
            .topological()
            .map_err(|error| OpenClError::Unsupported(error.to_string()))?;
        let uses_f16 = nodes
            .iter()
            .any(|node| node.ty().is_some_and(|ty| ty.scalar == DType::F16));
        let uses_bf16 = nodes
            .iter()
            .any(|node| node.ty().is_some_and(|ty| ty.scalar == DType::BF16));
        if nodes.iter().any(|node| {
            matches!(
                node.operation(),
                Operation::Barrier | Operation::If | Operation::EndIf
            )
        }) {
            return Err(OpenClError::Unsupported(
                "effects and barriers are outside the OpenCL static subset".into(),
            ));
        }
        let store = root
            .sources()
            .iter()
            .find(|node| matches!(node.operation(), Operation::Store))
            .ok_or_else(|| OpenClError::Unsupported("sink has no store".into()))?;
        let output_index = store
            .sources()
            .first()
            .ok_or_else(|| OpenClError::Unsupported("store has no index".into()))?;
        let Operation::Index(IndexValue::Buffer {
            buffer: output_id,
            elements: extent,
            input_shape: output_shape,
            output_shape: store_shape,
        }) = output_index.operation()
        else {
            return Err(OpenClError::Unsupported(
                "output requires a contiguous BufferIndex".into(),
            ));
        };
        if output_shape != store_shape {
            return Err(OpenClError::Unsupported(
                "non-contiguous output addressing".into(),
            ));
        }
        let output_dtype = output_index
            .ty()
            .ok_or_else(|| OpenClError::Unsupported("untyped output index".into()))?
            .scalar;
        supported_storage(output_dtype, self.capabilities)?;

        let mut inventory = BTreeMap::<u64, OpenClBufferAbi>::new();
        for node in &nodes {
            let (buffer, source_shape, elements, view) = match node.operation() {
                Operation::Index(IndexValue::Buffer {
                    buffer,
                    elements,
                    input_shape,
                    ..
                }) => (*buffer, input_shape.clone(), *elements, None),
                Operation::Index(IndexValue::View { buffer, view, .. }) => {
                    let access = OpenClViewAccess::new(
                        view,
                        node.ty()
                            .ok_or_else(|| OpenClError::Unsupported("untyped view index".into()))?
                            .scalar,
                    )?;
                    let elements = access
                        .source_shape
                        .numel()
                        .map_err(|_| OpenClError::Overflow)?;
                    (*buffer, access.source_shape, elements, Some(view.clone()))
                }
                _ => continue,
            };
            let dtype = node
                .ty()
                .ok_or_else(|| OpenClError::Unsupported("untyped buffer index".into()))?
                .scalar;
            supported_storage(dtype, self.capabilities)?;
            let abi = OpenClBufferAbi {
                id: buffer,
                dtype,
                source_shape,
                elements,
                mutable: buffer == *output_id,
                view,
            };
            if let Some(old) = inventory.insert(buffer, abi.clone())
                && old != abi
            {
                return Err(OpenClError::InvalidBinding(format!(
                    "buffer {buffer} has conflicting ABI metadata"
                )));
            }
        }

        let store_value = store
            .sources()
            .get(1)
            .ok_or_else(|| OpenClError::Unsupported("store has no value".into()))?;
        let reduction = crate::reduction_native::NativeReductionKernel::from_store(store)
            .map_err(|reason| OpenClError::Unsupported(reason.into()))?;
        let mut schedule_inputs = Vec::new();
        let mut seen = BTreeSet::new();
        for node in &nodes {
            if !matches!(node.operation(), Operation::Load) {
                continue;
            }
            let index = node
                .sources()
                .first()
                .ok_or_else(|| OpenClError::InvalidBinding("load lacks index".into()))?;
            let buffer = match index.operation() {
                Operation::Index(IndexValue::Buffer { buffer, .. })
                | Operation::Index(IndexValue::View { buffer, .. }) => *buffer,
                _ => {
                    return Err(OpenClError::Unsupported(
                        "load requires a checked static buffer index".into(),
                    ));
                }
            };
            if seen.insert(buffer) {
                schedule_inputs.push(
                    inventory
                        .get(&buffer)
                        .ok_or_else(|| OpenClError::InvalidBinding("load ABI missing".into()))?
                        .clone(),
                );
            }
        }
        let zero_epilogue_buffers = reduction
            .as_ref()
            .filter(|reduction| reduction.plan.reduction_len() == 0)
            .map(|reduction| {
                fn collect(node: &UOp, finalize: &UOp, buffers: &mut BTreeSet<u64>) {
                    if node.shares_node_with(finalize) {
                        return;
                    }
                    if let Operation::Index(
                        IndexValue::Buffer { buffer, .. } | IndexValue::View { buffer, .. },
                    ) = node.operation()
                    {
                        buffers.insert(*buffer);
                    }
                    for source in node.sources() {
                        collect(source, finalize, buffers);
                    }
                }
                let mut buffers = BTreeSet::new();
                collect(reduction.epilogue_root, reduction.finalize, &mut buffers);
                buffers
            });
        let mut buffers = schedule_inputs
            .iter()
            .filter(|buffer| {
                zero_epilogue_buffers
                    .as_ref()
                    .is_none_or(|epilogue| epilogue.contains(&buffer.id))
            })
            .cloned()
            .collect::<Vec<_>>();
        if seen.insert(*output_id) {
            buffers.push(
                inventory
                    .get(output_id)
                    .ok_or_else(|| OpenClError::InvalidBinding("output ABI missing".into()))?
                    .clone(),
            );
        }
        let output_position = buffers
            .iter()
            .position(|buffer| buffer.id == *output_id)
            .ok_or_else(|| OpenClError::InvalidBinding("output absent from ABI".into()))?;
        if output_position + 1 != buffers.len() {
            return Err(OpenClError::InvalidBinding(
                "output aliases an input buffer".into(),
            ));
        }
        let transaction = if reduction
            .as_ref()
            .is_some_and(|reduction| reduction.plan.reduction_len() == 0)
        {
            None
        } else if let Some(reduction) = &reduction {
            OpenClTransactionAbi::analyze(
                reduction.producer,
                output_position,
                OpenClGuardDomain::ReductionSource {
                    shape: reduction.plan.geometry.input.clone(),
                },
            )?
        } else {
            OpenClTransactionAbi::analyze(
                store_value,
                output_position,
                OpenClGuardDomain::Elementwise {
                    shape: buffers[output_position].source_shape.clone(),
                },
            )?
        };

        let entry = format!("rg_opencl_e{}_b{}", extent, buffers.len());
        let mut required_capabilities = required_capabilities(&buffers, uses_f16 || uses_bf16);
        if let Some(reduction) = &reduction {
            validate_dtype(
                &reduction.plan,
                reduction.plan.output_dtype,
                self.capabilities,
            )?;
            supported_storage(output_dtype, self.capabilities)?;
            let needed = reduction_capabilities(&reduction.plan, output_dtype);
            required_capabilities.int64 |= needed.int64;
            required_capabilities.fp64 |= needed.fp64;
        }
        let mut lines = Vec::new();
        if required_capabilities.fp64 {
            lines.push("#pragma OPENCL EXTENSION cl_khr_fp64 : enable".into());
        }
        if uses_f16 {
            lines.push(narrow::F16_SOURCE.into());
        }
        if uses_bf16 {
            lines.push(narrow::BF16_SOURCE.into());
        }
        lines.extend([
            format!("// {OPENCL_RENDERER_VERSION} ABI {OPENCL_ABI_VERSION}"),
            format!("__kernel void {entry}("),
        ]);
        for (index, buffer) in buffers.iter().enumerate() {
            let qualifier = if buffer.mutable { "" } else { "const " };
            lines.push(format!(
                "    __global {qualifier}{}* b{index},",
                cl_type(buffer.dtype)
            ));
        }
        lines.push("    ulong extent) {".into());
        if transaction.is_some() {
            let last = lines.pop().expect("signature terminator");
            debug_assert_eq!(last, "    ulong extent) {");
            lines.push("    ulong extent,".into());
            lines.push("    volatile __global uint* rg_status) {".into());
        }
        lines.push("  const ulong gid = (ulong)get_global_id(0);".into());
        lines.push("  if (gid >= extent) return;".into());
        let ids = buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| (buffer.id, index))
            .collect::<BTreeMap<_, _>>();
        let mut source_map = BTreeMap::new();
        if let Some(reduction) = &reduction {
            emit_reduction(
                reduction,
                output_dtype,
                output_position,
                &ids,
                &mut source_map,
                &mut lines,
                self.capabilities,
                transaction.as_ref(),
            )?;
        } else {
            let value = if let Some(transaction) = &transaction {
                emit_transactional(
                    store_value,
                    transaction,
                    &ids,
                    &mut source_map,
                    &mut lines,
                    self.capabilities,
                )?
            } else {
                emit_expr(
                    store_value,
                    &ids,
                    &mut source_map,
                    &mut lines,
                    "gid",
                    self.capabilities,
                )?
            };
            let store = format!(
                "b{output_position}[gid] = {};",
                encode_store(output_dtype, value)
            );
            lines.push(if transaction.is_some() {
                format!("  if (rg_ok) {store}")
            } else {
                format!("  {store}")
            });
        }
        lines.push("}".into());
        let source = lines.join("\n") + "\n";
        let cache_key = stable_key(&(
            OPENCL_RENDERER_VERSION,
            OPENCL_ABI_VERSION,
            self.local_size,
            self.capabilities,
            &source,
            &buffers,
            &schedule_inputs,
            &reduction
                .as_ref()
                .map(|kernel| (&kernel.plan, kernel.has_epilogue())),
            &transaction,
        ));
        Ok(RenderedOpenCl {
            source,
            source_map,
            buffers,
            extent: *extent,
            entry,
            cache_key,
            required_capabilities,
            transaction,
            schedule_inputs,
            semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::UOp(Arc::new(
                root.clone(),
            ))),
        })
    }
}

fn supported_storage(dtype: DType, capabilities: OpenClCapabilities) -> Result<(), OpenClError> {
    match dtype {
        DType::Bool | DType::I32 | DType::U32 | DType::F32 => Ok(()),
        DType::F16 | DType::BF16 if capabilities.fp64 => Ok(()),
        DType::I64 | DType::U64 if capabilities.int64 => Ok(()),
        DType::F64 if capabilities.fp64 => Ok(()),
        DType::F16 | DType::BF16 => Err(OpenClError::Unsupported(
            "exact narrow-float execution requires fp64 capability".into(),
        )),
        DType::I64 | DType::U64 => Err(OpenClError::Unsupported(
            "64-bit integer storage requires device capability".into(),
        )),
        DType::F64 => Err(OpenClError::Unsupported(
            "F64 storage requires fp64 device capability".into(),
        )),
        _ => Err(OpenClError::Unsupported(format!(
            "dtype {dtype:?} is outside the exact OpenCL static subset"
        ))),
    }
}

pub(super) fn cl_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "uchar",
        DType::I32 => "int",
        DType::U32 => "uint",
        DType::I64 => "long",
        DType::U64 => "ulong",
        DType::F32 => "float",
        DType::F64 => "double",
        DType::F16 | DType::BF16 => "ushort",
        _ => unreachable!("validated by supported_storage"),
    }
}

fn expression_type(dtype: DType) -> &'static str {
    if narrow::is_narrow(dtype) {
        "double"
    } else {
        cl_type(dtype)
    }
}

pub(super) struct OpenClScalarDialect {
    pub(super) capabilities: OpenClCapabilities,
}

impl dialect_seal::Sealed for OpenClScalarDialect {}

impl ScalarLaneDialect for OpenClScalarDialect {
    fn name(&self) -> &'static str {
        "OpenCL"
    }

    fn supports_value(&self, dtype: DType) -> bool {
        supported_storage(dtype, self.capabilities).is_ok()
    }

    fn cast(&self, source: DType, target: DType, value: &str) -> Result<String, String> {
        let result = match (source, target) {
            (source, target) if source == target => value.into(),
            (DType::Bool, target) if self.supports_value(target) => {
                format!("(({})({value}))", cl_type(target))
            }
            (source, DType::Bool) if self.supports_value(source) => {
                format!("((uchar)(({value}) != ({})0))", expression_type(source))
            }
            (DType::I32, DType::U32) => format!("as_uint({value})"),
            (DType::U32, DType::I32) => format!("as_int({value})"),
            (DType::I64, DType::U64) => format!("as_ulong({value})"),
            (DType::U64, DType::I64) => format!("as_long({value})"),
            (DType::I32 | DType::U32 | DType::I64 | DType::U64, DType::F32) => {
                format!("((float)({value}))")
            }
            (DType::F32, DType::F64) => format!("((double)({value}))"),
            (DType::F64, DType::F32) => format!("((float)({value}))"),
            (source, target)
                if narrow::is_narrow(target)
                    && matches!(source, DType::F16 | DType::BF16 | DType::F32 | DType::F64) =>
            {
                narrow::quantize(target, value).expect("target is a narrow float")
            }
            (source, DType::F32) if narrow::is_narrow(source) => {
                format!("((float)({value}))")
            }
            (source, DType::F64) if narrow::is_narrow(source) => value.into(),
            _ => return Err("cast is outside the exact OpenCL subset".into()),
        };
        Ok(result)
    }

    fn finish_float(&self, dtype: DType, value: String) -> Result<String, String> {
        Ok(narrow::quantize(dtype, &value).unwrap_or(value))
    }

    fn signed_infix(
        &self,
        dtype: DType,
        operator: &'static str,
        lhs: &str,
        rhs: &str,
    ) -> Result<String, String> {
        match dtype {
            DType::I32 => Ok(format!("as_int(as_uint({lhs}) {operator} as_uint({rhs}))")),
            DType::I64 => Ok(format!(
                "as_long(as_ulong({lhs}) {operator} as_ulong({rhs}))"
            )),
            _ => Err("OpenCL signed wrapping requires I32 or I64".into()),
        }
    }

    fn signed_neg(&self, dtype: DType, value: &str) -> Result<String, String> {
        match dtype {
            DType::I32 => Ok(format!("as_int((uint)0u - as_uint({value}))")),
            DType::I64 => Ok(format!("as_long((ulong)0ul - as_ulong({value}))")),
            _ => Err("OpenCL signed negation requires I32 or I64".into()),
        }
    }

    fn unsigned_neg(&self, dtype: DType, value: &str) -> Result<String, String> {
        match dtype {
            DType::U32 => Ok(format!("((uint)0u - ({value}))")),
            DType::U64 => Ok(format!("((ulong)0ul - ({value}))")),
            _ => Err("OpenCL unsigned negation requires U32 or U64".into()),
        }
    }

    fn signed_abs(&self, dtype: DType, value: &str) -> Result<String, String> {
        match dtype {
            DType::I32 => Ok(format!(
                "as_int(({value}) < 0 ? ((uint)0u - as_uint({value})) : as_uint({value}))"
            )),
            DType::I64 => Ok(format!(
                "as_long(({value}) < 0 ? ((ulong)0ul - as_ulong({value})) : as_ulong({value}))"
            )),
            _ => Err("OpenCL signed absolute value requires I32 or I64".into()),
        }
    }

    fn float_abs(&self, value: &str) -> String {
        format!("fabs({value})")
    }

    fn bool_value(&self, expression: String) -> String {
        format!("((uchar)({expression}))")
    }

    fn select(&self, condition: &str, on_true: &str, on_false: &str) -> String {
        format!("(({condition}) ? ({on_true}) : ({on_false}))")
    }

    fn call_intrinsic(&self, canonical_name: &'static str, value: &str) -> String {
        format!("{canonical_name}({value})")
    }

    fn float_one(&self, dtype: DType) -> Result<&'static str, String> {
        match dtype {
            DType::F32 => Ok("1.0f"),
            DType::F16 | DType::BF16 | DType::F64 => Ok("1.0"),
            _ => Err("OpenCL reciprocal requires floating dtype".into()),
        }
    }
}

fn encode_store(dtype: DType, value: impl AsRef<str>) -> String {
    narrow::encode(dtype, value.as_ref()).unwrap_or_else(|| value.as_ref().into())
}

fn required_capabilities(buffers: &[OpenClBufferAbi], uses_narrow: bool) -> OpenClCapabilities {
    OpenClCapabilities {
        int64: buffers
            .iter()
            .any(|buffer| matches!(buffer.dtype, DType::I64 | DType::U64)),
        fp64: uses_narrow || buffers.iter().any(|buffer| buffer.dtype == DType::F64),
    }
}

pub(super) fn guarded_value(
    op: GuardedIntegerOp,
    dtype: DType,
    lhs: &str,
    rhs: &str,
) -> Result<String, OpenClError> {
    let signed = matches!(dtype, DType::I32 | DType::I64);
    let min = match dtype {
        DType::I32 => "as_int((uint)0x80000000u)",
        DType::I64 => "as_long((ulong)0x8000000000000000ul)",
        _ => "0",
    };
    let overflow = if signed {
        format!("(({lhs}) == {min} && ({rhs}) == ({})-1)", cl_type(dtype))
    } else {
        "0".into()
    };
    let div = format!("(({overflow}) ? ({min}) : (({lhs}) / ({rhs})))");
    let rem = format!(
        "(({overflow}) ? ({})0 : (({lhs}) % ({rhs})))",
        cl_type(dtype)
    );
    Ok(match op {
        GuardedIntegerOp::Div | GuardedIntegerOp::TruncDiv => div,
        GuardedIntegerOp::FMod => rem,
        GuardedIntegerOp::FloorDiv if !signed => format!("(({lhs}) / ({rhs}))"),
        GuardedIntegerOp::Mod if !signed => format!("(({lhs}) % ({rhs}))"),
        GuardedIntegerOp::FloorDiv => {
            // RustGrad's signed integer oracle uses Euclidean division: the remainder is
            // nonnegative even when the divisor is negative. OpenCL integer division
            // truncates toward zero, so correct only a negative remainder; its direction
            // depends on the divisor sign.
            let rem = format!("(({lhs}) % ({rhs}))");
            let correction = format!("((({rhs}) > 0) ? 1 : -1)");
            format!(
                "(({overflow}) ? ({min}) : (({rem} < 0) ? (({lhs}) / ({rhs}) - {correction}) : (({lhs}) / ({rhs}))))"
            )
        }
        GuardedIntegerOp::Mod => {
            let signed_name = if dtype == DType::I32 { "int" } else { "long" };
            let unsigned_name = if dtype == DType::I32 { "uint" } else { "ulong" };
            let rem = format!("(({lhs}) % ({rhs}))");
            let magnitude = format!(
                "((({rhs}) < 0) ? (({unsigned_name})0 - as_{unsigned_name}({rhs})) : as_{unsigned_name}({rhs}))"
            );
            format!(
                "(({overflow}) ? ({})0 : (({rem} < 0) ? as_{signed_name}(as_{unsigned_name}({rem}) + {magnitude}) : {rem}))",
                cl_type(dtype),
            )
        }
        GuardedIntegerOp::Shl => match dtype {
            DType::I32 => format!("as_int(as_uint({lhs}) << (uint)({rhs}))"),
            DType::I64 => format!("as_long(as_ulong({lhs}) << (ulong)({rhs}))"),
            _ => format!("(({lhs}) << ({rhs}))"),
        },
        GuardedIntegerOp::Shr => format!("(({lhs}) >> ({rhs}))"),
    })
}

fn emit_expr(
    node: &UOp,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    linear: &str,
    capabilities: OpenClCapabilities,
) -> Result<String, OpenClError> {
    emit_expr_with_substitution(node, ids, source_map, lines, linear, capabilities, None)
}

fn emit_expr_with_substitution(
    node: &UOp,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    linear: &str,
    capabilities: OpenClCapabilities,
    substitution: Option<(&UOp, &str)>,
) -> Result<String, OpenClError> {
    if let Some((target, value)) = substitution
        && node.shares_node_with(target)
    {
        return Ok(value.into());
    }
    let map_id = source_map.len();
    source_map.insert(map_id, lines.len() + 1);
    let dtype = node
        .ty()
        .ok_or_else(|| OpenClError::Unsupported(format!("untyped {:?}", node.operation())))?
        .scalar;
    supported_storage(dtype, capabilities)?;
    let child = |index: usize, source_map: &mut BTreeMap<usize, usize>, lines: &mut Vec<String>| {
        node.sources()
            .get(index)
            .ok_or_else(|| OpenClError::Unsupported("missing expression operand".into()))
            .and_then(|source| {
                emit_expr_with_substitution(
                    source,
                    ids,
                    source_map,
                    lines,
                    linear,
                    capabilities,
                    substitution,
                )
            })
    };
    match node.operation() {
        Operation::Const(value) => match value {
            LiteralValue::Scalar {
                dtype: DType::F32,
                bits,
            } => Ok(format!("as_float((uint)0x{:08x}u)", *bits as u32)),
            LiteralValue::Scalar {
                dtype: DType::Bool,
                bits,
            } if *bits <= 1 => Ok(format!("((uchar){bits}u)")),
            LiteralValue::Scalar {
                dtype: DType::I32,
                bits,
            } => Ok(format!("as_int((uint)0x{:08x}u)", *bits as u32)),
            LiteralValue::Scalar {
                dtype: DType::U32,
                bits,
            } => Ok(format!("((uint)0x{:08x}u)", *bits as u32)),
            LiteralValue::Scalar {
                dtype: DType::I64,
                bits,
            } => Ok(format!("as_long((ulong)0x{bits:016x}ul)")),
            LiteralValue::Scalar {
                dtype: DType::U64,
                bits,
            } => Ok(format!("((ulong)0x{bits:016x}ul)")),
            LiteralValue::Scalar {
                dtype: DType::F64,
                bits,
            } => Ok(format!("as_double((ulong)0x{bits:016x}ul)")),
            LiteralValue::Scalar {
                dtype: DType::F16,
                bits,
            } if dtype == DType::F16 => Ok(narrow::decode(
                DType::F16,
                format!("((ushort)0x{:04x}u)", *bits as u16),
            )
            .expect("F16 is a narrow float")),
            LiteralValue::Scalar {
                dtype: DType::BF16,
                bits,
            } if dtype == DType::BF16 => Ok(narrow::decode(
                DType::BF16,
                format!("((ushort)0x{:04x}u)", *bits as u16),
            )
            .expect("BF16 is a narrow float")),
            LiteralValue::Scalar { .. } => Err(OpenClError::Unsupported(
                "scalar literal/type mismatch".into(),
            )),
            _ => Err(OpenClError::Unsupported("invalid scalar literal".into())),
        },
        Operation::Load => {
            let index = node
                .sources()
                .first()
                .ok_or_else(|| OpenClError::Unsupported("load has no index".into()))?;
            let (buffer, input_shape, output_shape, view) = match index.operation() {
                Operation::Index(IndexValue::Buffer {
                    buffer,
                    input_shape,
                    output_shape,
                    ..
                }) => (*buffer, input_shape, output_shape, None),
                Operation::Index(IndexValue::View {
                    buffer,
                    input_shape,
                    output_shape,
                    view,
                    ..
                }) => (*buffer, input_shape, output_shape, Some(view)),
                _ => {
                    return Err(OpenClError::Unsupported(
                        "load requires a checked static buffer index".into(),
                    ));
                }
            };
            let position = ids
                .get(&buffer)
                .ok_or_else(|| OpenClError::InvalidBinding("load buffer absent from ABI".into()))?;
            let logical = broadcast_offset(input_shape, output_shape, linear)?;
            let offset = match view {
                Some(view) => OpenClViewAccess::new(view, dtype)?.expression(logical),
                None => logical,
            };
            let raw = format!("b{position}[{offset}]");
            Ok(narrow::decode(dtype, &raw).unwrap_or(raw))
        }
        other => {
            let mut sources = Vec::with_capacity(node.sources().len());
            for slot in 0..node.sources().len() {
                sources.push(child(slot, source_map, lines)?);
            }
            let instruction = project_scalar_lane(node, &sources)
                .map_err(OpenClError::Unsupported)?
                .ok_or_else(|| OpenClError::Unsupported(format!("{other:?}")))?;
            emit_scalar_lane(&OpenClScalarDialect { capabilities }, &instruction)
                .map_err(OpenClError::Unsupported)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_reduction(
    reduction: &crate::reduction_native::NativeReductionKernel<'_>,
    dtype: DType,
    output_position: usize,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    capabilities: OpenClCapabilities,
    transaction: Option<&OpenClTransactionAbi>,
) -> Result<(), OpenClError> {
    let value = reduction.producer;
    let plan = &reduction.plan;
    let reduction_len = plan.reduction_len();
    if reduction_len == 0 {
        let final_value =
            reduction_identity_expr(plan.output_dtype, plan.finalize(plan.identity()))?;
        let committed = reduction_finalize_commit_expr(
            plan.output_dtype,
            &final_value,
            reduction.has_epilogue(),
        );
        let final_value = if reduction.has_epilogue() {
            emit_expr_with_substitution(
                reduction.epilogue_root,
                ids,
                source_map,
                lines,
                "gid",
                capabilities,
                Some((reduction.finalize, committed.as_str())),
            )?
        } else {
            committed
        };
        lines.push(format!(
            "  b{output_position}[gid] = {};",
            encode_store(dtype, final_value)
        ));
        return Ok(());
    }
    if transaction.is_some() {
        lines.push("  uchar rg_ok = (uchar)1u;".into());
    }
    let accumulator_type = reduction_accumulator_type(plan.accumulator_dtype, plan.kind);
    let identity = reduction_identity_expr(plan.accumulator_dtype, plan.identity())?;
    lines.push(format!("  {accumulator_type} acc = {identity};"));
    lines.push(format!(
        "  for (ulong r = 0ul; r < {}ul; ++r) {{",
        reduction_len
    ));
    lines.push(format!(
        "    const ulong src_gid = {};",
        input_offset_expression(plan)?
    ));
    let expression = if let Some(transaction) = transaction {
        emit_transactional_reduction(value, transaction, ids, source_map, lines, capabilities)?
    } else {
        emit_expr(value, ids, source_map, lines, "src_gid", capabilities)?
    };
    let prefix = if transaction.is_some() {
        "    if (rg_ok) "
    } else {
        "    "
    };
    let expression = reduction_cast_expr(plan.source_dtype, plan.accumulator_dtype, &expression);
    if plan.is_singleton_identity() {
        lines.push(format!("{prefix}acc = ({accumulator_type})({expression});"));
    } else {
        match plan.kind {
            crate::ReduceKind::Any => {
                lines.push(format!("{prefix}acc = (uchar)(acc || ({expression}));"));
            }
            crate::ReduceKind::All => {
                lines.push(format!("{prefix}acc = (uchar)(acc && ({expression}));"));
            }
            crate::ReduceKind::Sum | crate::ReduceKind::Mean
                if plan.accumulator_dtype == DType::Bool =>
            {
                lines.push(format!("{prefix}acc = (uchar)(acc || ({expression}));"));
            }
            crate::ReduceKind::Sum | crate::ReduceKind::Mean | crate::ReduceKind::Product => {
                let product = plan.kind == crate::ReduceKind::Product;
                let update =
                    reduction_arithmetic_expr(plan.accumulator_dtype, "acc", &expression, product);
                lines.push(format!("{prefix}acc = {update};"));
            }
            crate::ReduceKind::Min | crate::ReduceKind::Max => {
                let comparison = if plan.kind == crate::ReduceKind::Max {
                    ">"
                } else {
                    "<"
                };
                lines.push(format!(
                    "    const {accumulator_type} next = ({accumulator_type})({expression});"
                ));
                lines.push(format!("{prefix}if (next {comparison} acc) acc = next;"));
            }
        }
    }
    lines.push("  }".into());
    if matches!(plan.kind, crate::ReduceKind::Mean) {
        let divisor = reduction_identity_expr(
            plan.accumulator_dtype,
            plan.mean_divisor()
                .expect("nonempty validated Mean divisor"),
        )?;
        let update = reduction_commit_expr(plan.accumulator_dtype, &format!("(acc / {divisor})"));
        lines.push(if transaction.is_some() {
            format!("  if (rg_ok) acc = {update};")
        } else {
            format!("  acc = {update};")
        });
    }
    let final_value = reduction_cast_expr(plan.accumulator_dtype, plan.output_dtype, "acc");
    let committed =
        reduction_finalize_commit_expr(plan.output_dtype, &final_value, reduction.has_epilogue());
    let final_value = if reduction.has_epilogue() {
        emit_expr_with_substitution(
            reduction.epilogue_root,
            ids,
            source_map,
            lines,
            "gid",
            capabilities,
            Some((reduction.finalize, committed.as_str())),
        )?
    } else {
        committed
    };
    let store = format!(
        "b{output_position}[gid] = {};",
        encode_store(dtype, final_value)
    );
    lines.push(if transaction.is_some() {
        format!("  if (rg_ok) {store}")
    } else {
        format!("  {store}")
    });
    Ok(())
}

fn reduction_accumulator_type(dtype: DType, kind: crate::ReduceKind) -> &'static str {
    match (dtype, kind) {
        (DType::I32, crate::ReduceKind::Sum | crate::ReduceKind::Product) => "uint",
        (DType::I64, crate::ReduceKind::Sum | crate::ReduceKind::Product) => "ulong",
        (DType::F16 | DType::BF16 | DType::F32, _) => "float",
        (DType::F64, _) => "double",
        _ => cl_type(dtype),
    }
}

fn reduction_identity_expr(dtype: DType, value: crate::Scalar) -> Result<String, OpenClError> {
    Ok(match value {
        crate::Scalar::Bool(value) => format!("(uchar){}u", u8::from(value)),
        crate::Scalar::I(value) if dtype == DType::I32 => {
            format!("as_int((uint)0x{:08x}u)", value as u32)
        }
        crate::Scalar::I(value) if dtype == DType::I64 => {
            format!("as_long((ulong)0x{:016x}ul)", value as u64)
        }
        crate::Scalar::I(value) => value.to_string(),
        crate::Scalar::U(value) if dtype == DType::U64 => format!("(ulong){value}ul"),
        crate::Scalar::U(value) => format!("(uint){value}u"),
        crate::Scalar::F(value) if value.is_nan() => "NAN".into(),
        crate::Scalar::F(value) if value == f64::NEG_INFINITY => "-INFINITY".into(),
        crate::Scalar::F(value) if value == f64::INFINITY => "INFINITY".into(),
        crate::Scalar::F(0.0) => "0.0".into(),
        crate::Scalar::F(1.0) => "1.0".into(),
        crate::Scalar::F(value) if value.is_finite() => format!("{value:.17e}"),
        crate::Scalar::F(_) => {
            return Err(OpenClError::Unsupported(
                "OpenCL reduction identity is not representable".into(),
            ));
        }
    })
}

fn reduction_cast_expr(source: DType, target: DType, value: &str) -> String {
    match (source, target) {
        (DType::I32, DType::I32) | (DType::I64, DType::I64) => value.into(),
        (_, DType::I32) => format!("(int)({value})"),
        (_, DType::U32) => format!("(uint)({value})"),
        (_, DType::I64) => format!("(long)({value})"),
        (_, DType::U64) => format!("(ulong)({value})"),
        (_, DType::F32) => format!("(float)({value})"),
        (_, DType::F64) => format!("(double)({value})"),
        (_, DType::F16 | DType::BF16) => format!("(float)({value})"),
        (_, DType::Bool) => format!("(uchar)(({value}) != 0)"),
        _ => value.into(),
    }
}

fn reduction_arithmetic_expr(dtype: DType, lhs: &str, rhs: &str, product: bool) -> String {
    let operator = if product { "*" } else { "+" };
    reduction_commit_expr(dtype, &format!("(({lhs}) {operator} ({rhs}))"))
}

fn reduction_commit_expr(dtype: DType, value: &str) -> String {
    match dtype {
        DType::I32 => format!("(uint)({value})"),
        DType::I64 => format!("(ulong)({value})"),
        DType::F32 => format!("(float)({value})"),
        DType::F16 | DType::BF16 => {
            let encoded = narrow::encode(dtype, value).expect("validated narrow reduction dtype");
            narrow::decode(dtype, &encoded).expect("validated narrow reduction dtype")
        }
        _ => format!("({})({value})", cl_type(dtype)),
    }
}

fn reduction_storage_expr(dtype: DType, value: &str) -> String {
    match dtype {
        DType::I32 => format!("as_int({value})"),
        DType::I64 => format!("as_long({value})"),
        _ => value.into(),
    }
}

fn reduction_finalize_commit_expr(dtype: DType, value: &str, has_epilogue: bool) -> String {
    if has_epilogue && matches!(dtype, DType::F16 | DType::BF16) {
        narrow::quantize(dtype, value).expect("validated narrow reduction output dtype")
    } else {
        reduction_storage_expr(dtype, value)
    }
}

pub(super) fn broadcast_offset(
    input: &Shape,
    output: &Shape,
    linear: &str,
) -> Result<String, OpenClError> {
    if input.rank() > output.rank() {
        return Err(OpenClError::Unsupported(
            "input rank exceeds output rank".into(),
        ));
    }
    if input.rank() == 0 {
        return Ok("0ul".into());
    }
    let input_strides = input.contiguous_strides();
    let output_strides = output.contiguous_strides();
    let pad = output.rank() - input.rank();
    let mut terms = Vec::new();
    for axis in 0..input.rank() {
        let dim = input.dims()[axis];
        let out_dim = output.dims()[pad + axis];
        if dim != 1 && dim != out_dim {
            return Err(OpenClError::Unsupported(
                "invalid broadcast metadata".into(),
            ));
        }
        if dim != 1 {
            terms.push(format!(
                "(({linear} / {}ul) % {}ul) * {}ul",
                output_strides[pad + axis],
                dim,
                input_strides[axis]
            ));
        }
    }
    Ok(if terms.is_empty() {
        "0ul".into()
    } else {
        terms.join(" + ")
    })
}

fn stable_key(value: &impl Hash) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
