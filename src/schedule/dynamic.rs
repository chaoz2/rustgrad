//! Runtime-sized schedule ABI for exact CPU dynamic-cardinality results.
//!
//! Dynamic graph operations lower into one data-bearing instruction DAG. Each
//! item is one runtime instruction; fixed-size values produced by reductions
//! use [`RuntimeValueDesc::Fixed`] without importing a second schedule ABI.

use crate::ir::{DynamicInput, DynamicOperation, dynamic_reduction_dtypes, source_lub};
use crate::{
    BinaryOp, BufferDesc, DType, DynamicAllocation, DynamicAllocationError, DynamicAllocationPlan,
    DynamicAllocationTarget, DynamicBinding, DynamicNodeId, Graph, NodeId, ReduceKind,
    ReductionDType, Shape, UnaryOp,
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RuntimeCountId(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RuntimeCount {
    pub id: RuntimeCountId,
    pub value: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RuntimeBufferId(pub u64);

/// A runtime-sized dense buffer whose shape is derived from one exact count.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RuntimeBufferDesc {
    pub id: RuntimeBufferId,
    pub dtype: DType,
    pub shape: crate::DynamicOutputShape,
    pub count: RuntimeCountId,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum RuntimeValueDesc {
    Dynamic(RuntimeBufferDesc),
    Fixed(BufferDesc),
}

impl RuntimeValueDesc {
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::Dynamic(value) => value.dtype,
            Self::Fixed(value) => value.dtype,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct StaticScalarBinding {
    pub node: NodeId,
    pub descriptor: BufferDesc,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum RuntimeValueSource {
    Produced(RuntimeValueDesc),
    StaticScalar(StaticScalarBinding),
}

/// One executable runtime action. Operands and outputs live in the variant that
/// owns them, making kind/payload/output cross-products unrepresentable.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum RuntimeInstruction {
    Count {
        plan: DynamicAllocationPlan,
        output: RuntimeCountId,
    },
    Allocate {
        output: RuntimeBufferDesc,
    },
    MaterializeNonzero {
        input: DynamicBinding,
        output: RuntimeBufferDesc,
    },
    MaterializeMaskedSelect {
        input: DynamicBinding,
        mask: DynamicBinding,
        output: RuntimeBufferDesc,
    },
    Unary {
        origin: usize,
        op: UnaryOp,
        input: RuntimeValueDesc,
        output: RuntimeValueDesc,
    },
    Binary {
        origin: usize,
        op: BinaryOp,
        lhs: RuntimeValueSource,
        rhs: RuntimeValueSource,
        output: RuntimeValueDesc,
    },
    Reduce {
        origin: usize,
        op: ReduceKind,
        dtypes: ReductionDType,
        input: RuntimeValueDesc,
        output: BufferDesc,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct RuntimeScheduleItem {
    pub id: u64,
    pub dependencies: Vec<u64>,
    pub instruction: RuntimeInstruction,
    pub cache_key: u64,
}

/// Canonical runtime-cardinality DAG. Runtime operands are stored only in
/// instruction variants and produced values only in their descriptors.
#[derive(Clone, Debug)]
pub(crate) struct RuntimeSchedule {
    pub items: Vec<RuntimeScheduleItem>,
    pub buffers: Vec<RuntimeBufferDesc>,
    pub output: RuntimeValueDesc,
    pub lifetimes: Vec<crate::memory_plan::RuntimeAllocationLifetime>,
    pub identity: u64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RuntimeBufferTable {
    descriptors: Vec<RuntimeBufferDesc>,
    allocations: BTreeMap<RuntimeBufferId, DynamicAllocation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeScheduleError {
    Plan(DynamicAllocationError),
    InvalidOrdering(&'static str),
    DuplicateBuffer(RuntimeBufferId),
    UnknownBuffer(RuntimeBufferId),
    UnknownCount(RuntimeCountId),
    LiveLookupBeforeAllocation(RuntimeBufferId),
    DuplicateAllocation(RuntimeBufferId),
}

impl fmt::Display for RuntimeScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "runtime schedule error: {self:?}")
    }
}
impl std::error::Error for RuntimeScheduleError {}

impl RuntimeInstruction {
    pub(crate) fn output(&self) -> Option<RuntimeValueDesc> {
        match self {
            Self::Count { .. } | Self::Allocate { .. } => None,
            Self::MaterializeNonzero { output, .. }
            | Self::MaterializeMaskedSelect { output, .. } => {
                Some(RuntimeValueDesc::Dynamic(output.clone()))
            }
            Self::Unary { output, .. } | Self::Binary { output, .. } => Some(output.clone()),
            Self::Reduce { output, .. } => Some(RuntimeValueDesc::Fixed(output.clone())),
        }
    }

    pub(crate) fn runtime_inputs(&self) -> impl Iterator<Item = &RuntimeBufferDesc> {
        let mut inputs = Vec::with_capacity(2);
        match self {
            Self::Unary { input, .. } | Self::Reduce { input, .. } => {
                if let RuntimeValueDesc::Dynamic(input) = input {
                    inputs.push(input);
                }
            }
            Self::Binary { lhs, rhs, .. } => {
                if let RuntimeValueSource::Produced(RuntimeValueDesc::Dynamic(input)) = lhs {
                    inputs.push(input);
                }
                if let RuntimeValueSource::Produced(RuntimeValueDesc::Dynamic(input)) = rhs {
                    inputs.push(input);
                }
            }
            _ => {}
        }
        inputs.into_iter()
    }
}

impl RuntimeScheduleItem {
    pub(crate) fn output(&self) -> Option<RuntimeValueDesc> {
        self.instruction.output()
    }
}

pub(crate) fn schedule_dynamic(
    graph: &Graph,
    output: DynamicNodeId,
) -> Result<RuntimeSchedule, RuntimeScheduleError> {
    let mut builder = DynamicScheduleBuilder::new(graph);
    let output = builder.lower(output)?;
    builder.finish(output)
}

#[derive(Clone)]
struct PlannedValue {
    descriptor: RuntimeValueDesc,
    producer: u64,
}

struct BinaryValueRequest<'a> {
    origin: usize,
    runtime: Option<&'a RuntimeBufferDesc>,
    op: BinaryOp,
    lhs: &'a RuntimeValueSource,
    rhs: &'a RuntimeValueSource,
    shape: crate::DynamicOutputShape,
    dtype: DType,
}

struct DynamicScheduleBuilder<'a> {
    graph: &'a Graph,
    items: Vec<RuntimeScheduleItem>,
    buffers: Vec<RuntimeBufferDesc>,
    memo: HashMap<DynamicNodeId, PlannedValue>,
}

impl<'a> DynamicScheduleBuilder<'a> {
    fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            items: Vec::new(),
            buffers: Vec::new(),
            memo: HashMap::new(),
        }
    }

    fn lower(&mut self, id: DynamicNodeId) -> Result<PlannedValue, RuntimeScheduleError> {
        if let Some(value) = self.memo.get(&id) {
            return Ok(value.clone());
        }
        let node = self
            .graph
            .dynamic_node(id)
            .map_err(|_| RuntimeScheduleError::InvalidOrdering("dynamic node is absent"))?
            .clone();
        let value = match node.operation.clone() {
            DynamicOperation::Nonzero { .. } => self.lower_root(id, true)?,
            DynamicOperation::MaskedSelect { .. } => self.lower_root(id, false)?,
            DynamicOperation::Unary { op, input } => {
                let source = self.lower(input)?;
                let output = self.derived_unary_value(
                    id.index,
                    &source.descriptor,
                    op,
                    node.output,
                    node.dtype,
                )?;
                let mut dependencies = vec![source.producer];
                if let RuntimeValueDesc::Dynamic(output) = &output {
                    let count = self.count_producer(output.count)?;
                    dependencies.push(self.push_runtime(
                        vec![count],
                        RuntimeInstruction::Allocate {
                            output: output.clone(),
                        },
                    )?);
                }
                let producer = self.push_runtime(
                    dependencies,
                    RuntimeInstruction::Unary {
                        origin: id.index,
                        op,
                        input: source.descriptor,
                        output: output.clone(),
                    },
                )?;
                PlannedValue {
                    descriptor: output,
                    producer,
                }
            }
            DynamicOperation::Binary { op, lhs, rhs } => {
                let lhs = self.lower_operand(lhs)?;
                let rhs = self.lower_operand(rhs)?;
                let runtime = [&lhs.0, &rhs.0]
                    .into_iter()
                    .find_map(|source| match source {
                        RuntimeValueSource::Produced(RuntimeValueDesc::Dynamic(value)) => {
                            Some(value)
                        }
                        _ => None,
                    });
                for operand in [&lhs.0, &rhs.0] {
                    if let (
                        Some(runtime),
                        RuntimeValueSource::Produced(RuntimeValueDesc::Dynamic(other)),
                    ) = (runtime, operand)
                        && (other.count != runtime.count || other.shape != runtime.shape)
                    {
                        return Err(RuntimeScheduleError::InvalidOrdering(
                            "dynamic binary count provenance differs",
                        ));
                    }
                }
                let output = self.derived_binary_value(BinaryValueRequest {
                    origin: id.index,
                    runtime,
                    op,
                    lhs: &lhs.0,
                    rhs: &rhs.0,
                    shape: node.output,
                    dtype: node.dtype,
                })?;
                let mut inputs = lhs.1;
                inputs.extend(rhs.1);
                inputs.sort_unstable();
                inputs.dedup();
                if let RuntimeValueDesc::Dynamic(output) = &output {
                    let count = self.count_producer(output.count)?;
                    inputs.push(self.push_runtime(
                        vec![count],
                        RuntimeInstruction::Allocate {
                            output: output.clone(),
                        },
                    )?);
                }
                let producer = self.push_runtime(
                    inputs,
                    RuntimeInstruction::Binary {
                        origin: id.index,
                        op,
                        lhs: lhs.0,
                        rhs: rhs.0,
                        output: output.clone(),
                    },
                )?;
                PlannedValue {
                    descriptor: output,
                    producer,
                }
            }
            DynamicOperation::Sum { input } | DynamicOperation::Mean { input } => {
                let source = self.lower(input)?;
                let op = if matches!(node.operation, DynamicOperation::Sum { .. }) {
                    ReduceKind::Sum
                } else {
                    ReduceKind::Mean
                };
                let dtypes = dynamic_reduction_dtypes(source.descriptor.dtype(), op).ok_or(
                    RuntimeScheduleError::InvalidOrdering("dynamic reduction is unsupported"),
                )?;
                if node.dtype != dtypes.output {
                    return Err(RuntimeScheduleError::InvalidOrdering(
                        "dynamic reduction dtype is not canonical",
                    ));
                }
                let output =
                    fixed_reduction_desc(id.index, value_desc_id(&source.descriptor), op, dtypes);
                let producer = self.push_runtime(
                    vec![source.producer],
                    RuntimeInstruction::Reduce {
                        origin: id.index,
                        op,
                        dtypes,
                        input: source.descriptor,
                        output: output.clone(),
                    },
                )?;
                PlannedValue {
                    descriptor: RuntimeValueDesc::Fixed(output),
                    producer,
                }
            }
        };
        self.memo.insert(id, value.clone());
        Ok(value)
    }

    fn lower_root(
        &mut self,
        id: DynamicNodeId,
        nonzero: bool,
    ) -> Result<PlannedValue, RuntimeScheduleError> {
        let plan = DynamicAllocationPlan::for_output(self.graph, id)
            .map_err(RuntimeScheduleError::Plan)?;
        plan.validate_target(DynamicAllocationTarget::RuntimeSchedule)
            .map_err(RuntimeScheduleError::Plan)?;
        let count = RuntimeCountId(plan.identity());
        let count_item = self.push_runtime(
            vec![],
            RuntimeInstruction::Count {
                plan: plan.clone(),
                output: count,
            },
        )?;
        let output = RuntimeBufferDesc {
            id: RuntimeBufferId(plan.identity()),
            dtype: plan.output_dtype(),
            shape: plan.output_shape(),
            count,
        };
        self.insert_buffer(output.clone())?;
        let allocation = self.push_runtime(
            vec![count_item],
            RuntimeInstruction::Allocate {
                output: output.clone(),
            },
        )?;
        let bindings = plan.bindings();
        let instruction = if nonzero {
            RuntimeInstruction::MaterializeNonzero {
                input: bindings[0].clone(),
                output: output.clone(),
            }
        } else {
            RuntimeInstruction::MaterializeMaskedSelect {
                input: bindings[0].clone(),
                mask: bindings[1].clone(),
                output: output.clone(),
            }
        };
        let producer = self.push_runtime(vec![allocation], instruction)?;
        Ok(PlannedValue {
            descriptor: RuntimeValueDesc::Dynamic(output),
            producer,
        })
    }

    fn lower_operand(
        &mut self,
        operand: DynamicInput,
    ) -> Result<(RuntimeValueSource, Vec<u64>), RuntimeScheduleError> {
        match operand {
            DynamicInput::Dynamic(id) => {
                let value = self.lower(id)?;
                Ok((
                    RuntimeValueSource::Produced(value.descriptor),
                    vec![value.producer],
                ))
            }
            DynamicInput::StaticScalar(node) => Ok((
                RuntimeValueSource::StaticScalar(static_scalar_binding(self.graph, node)?),
                vec![],
            )),
        }
    }

    fn derived_unary_value(
        &mut self,
        origin: usize,
        source: &RuntimeValueDesc,
        op: UnaryOp,
        shape: crate::DynamicOutputShape,
        dtype: DType,
    ) -> Result<RuntimeValueDesc, RuntimeScheduleError> {
        match source {
            RuntimeValueDesc::Dynamic(source) => {
                let descriptor = RuntimeBufferDesc {
                    id: unary_buffer_id(origin, source.id, op, dtype),
                    dtype,
                    shape,
                    count: source.count,
                };
                self.insert_buffer(descriptor.clone())?;
                Ok(RuntimeValueDesc::Dynamic(descriptor))
            }
            RuntimeValueDesc::Fixed(source) => Ok(RuntimeValueDesc::Fixed(fixed_pointwise_desc(
                "runtime-fixed-unary-v1",
                derived_branch_id(origin, source.id),
                source.shape.clone(),
                dtype,
            )?)),
        }
    }

    fn derived_binary_value(
        &mut self,
        request: BinaryValueRequest<'_>,
    ) -> Result<RuntimeValueDesc, RuntimeScheduleError> {
        let BinaryValueRequest {
            origin,
            runtime,
            op,
            lhs,
            rhs,
            shape,
            dtype,
        } = request;
        let id = binary_buffer_id(origin, op, lhs, rhs, dtype);
        if let Some(runtime) = runtime {
            let descriptor = RuntimeBufferDesc {
                id,
                dtype,
                shape,
                count: runtime.count,
            };
            self.insert_buffer(descriptor.clone())?;
            Ok(RuntimeValueDesc::Dynamic(descriptor))
        } else {
            Ok(RuntimeValueDesc::Fixed(fixed_pointwise_desc(
                "runtime-fixed-binary-v1",
                id.0,
                Shape::from([]),
                dtype,
            )?))
        }
    }

    fn insert_buffer(&mut self, descriptor: RuntimeBufferDesc) -> Result<(), RuntimeScheduleError> {
        if self.buffers.iter().any(|old| old.id == descriptor.id) {
            return Err(RuntimeScheduleError::DuplicateBuffer(descriptor.id));
        }
        self.buffers.push(descriptor);
        Ok(())
    }

    fn count_producer(&self, count: RuntimeCountId) -> Result<u64, RuntimeScheduleError> {
        self.items
            .iter()
            .find_map(|item| match &item.instruction {
                RuntimeInstruction::Count { output, .. } if *output == count => Some(item.id),
                _ => None,
            })
            .ok_or(RuntimeScheduleError::UnknownCount(count))
    }

    fn push_runtime(
        &mut self,
        dependencies: Vec<u64>,
        instruction: RuntimeInstruction,
    ) -> Result<u64, RuntimeScheduleError> {
        let id = u64::try_from(self.items.len()).map_err(|_| {
            RuntimeScheduleError::InvalidOrdering("runtime schedule item ID overflows")
        })?;
        let cache_key = runtime_item_key(id, &dependencies, &instruction);
        let item = RuntimeScheduleItem {
            id,
            dependencies,
            instruction,
            cache_key,
        };
        self.items.push(item);
        Ok(id)
    }

    fn finish(self, output: PlannedValue) -> Result<RuntimeSchedule, RuntimeScheduleError> {
        let output = output.descriptor;
        let lifetimes = runtime_lifetimes(&self.items, &self.buffers)?;
        let mut schedule = RuntimeSchedule {
            items: self.items,
            buffers: self.buffers,
            output,
            lifetimes,
            identity: 0,
        };
        schedule.identity = runtime_schedule_identity(&schedule);
        schedule.validate()?;
        Ok(schedule)
    }
}

fn static_scalar_binding(
    graph: &Graph,
    node: NodeId,
) -> Result<StaticScalarBinding, RuntimeScheduleError> {
    let value = graph
        .node(node)
        .map_err(|_| RuntimeScheduleError::InvalidOrdering("dynamic scalar is absent"))?;
    if value.shape.numel().ok() != Some(1) {
        return Err(RuntimeScheduleError::InvalidOrdering(
            "dynamic static operand is not scalar",
        ));
    }
    Ok(StaticScalarBinding {
        node,
        descriptor: BufferDesc {
            id: derived_static_scalar_id(node, value.dtype),
            shape: value.shape.clone(),
            dtype: value.dtype,
            bytes: value.dtype.itemsize(),
            alignment: value.dtype.itemsize().max(1),
            read_only: true,
            view: None,
        },
    })
}

fn fixed_reduction_desc(
    origin: usize,
    source: RuntimeBufferId,
    op: ReduceKind,
    dtypes: ReductionDType,
) -> BufferDesc {
    let mut hasher = DefaultHasher::new();
    match op {
        ReduceKind::Sum => "runtime-reduce-sum-buffer-v1",
        ReduceKind::Mean => "runtime-reduce-mean-buffer-v1",
        _ => "runtime-reduce-buffer-v2",
    }
    .hash(&mut hasher);
    source.hash(&mut hasher);
    origin.hash(&mut hasher);
    dtypes.hash(&mut hasher);
    BufferDesc {
        id: hasher.finish(),
        shape: Shape::from([]),
        dtype: dtypes.output,
        bytes: dtypes.output.itemsize(),
        alignment: dtypes.output.itemsize().max(1),
        read_only: false,
        view: None,
    }
}

fn fixed_pointwise_desc(
    tag: &str,
    source: u64,
    shape: Shape,
    dtype: DType,
) -> Result<BufferDesc, RuntimeScheduleError> {
    let mut hasher = DefaultHasher::new();
    tag.hash(&mut hasher);
    source.hash(&mut hasher);
    dtype.hash(&mut hasher);
    let elements = shape
        .numel()
        .map_err(|_| RuntimeScheduleError::InvalidOrdering("dynamic fixed shape overflows"))?;
    let bytes =
        elements
            .checked_mul(dtype.itemsize())
            .ok_or(RuntimeScheduleError::InvalidOrdering(
                "dynamic fixed bytes overflow",
            ))?;
    Ok(BufferDesc {
        id: hasher.finish(),
        shape,
        dtype,
        bytes,
        alignment: dtype.itemsize().max(1),
        read_only: false,
        view: None,
    })
}

fn value_desc_id(value: &RuntimeValueDesc) -> RuntimeBufferId {
    match value {
        RuntimeValueDesc::Dynamic(value) => value.id,
        RuntimeValueDesc::Fixed(value) => RuntimeBufferId(value.id),
    }
}

fn derived_static_scalar_id(node: NodeId, dtype: DType) -> u64 {
    let mut hasher = DefaultHasher::new();
    "runtime-static-scalar-v1".hash(&mut hasher);
    node.hash(&mut hasher);
    dtype.hash(&mut hasher);
    hasher.finish()
}

fn derived_branch_id(origin: usize, source: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    "runtime-branch-origin-v1".hash(&mut hasher);
    origin.hash(&mut hasher);
    source.hash(&mut hasher);
    hasher.finish()
}

fn unary_buffer_id(
    origin: usize,
    source: RuntimeBufferId,
    op: UnaryOp,
    dtype: DType,
) -> RuntimeBufferId {
    let mut hasher = DefaultHasher::new();
    "runtime-unary-buffer-v1".hash(&mut hasher);
    source.hash(&mut hasher);
    origin.hash(&mut hasher);
    op.hash(&mut hasher);
    dtype.hash(&mut hasher);
    RuntimeBufferId(hasher.finish())
}

fn binary_buffer_id(
    origin: usize,
    op: BinaryOp,
    lhs: &RuntimeValueSource,
    rhs: &RuntimeValueSource,
    dtype: DType,
) -> RuntimeBufferId {
    let mut hasher = DefaultHasher::new();
    match (lhs, rhs) {
        (
            RuntimeValueSource::Produced(RuntimeValueDesc::Dynamic(source)),
            RuntimeValueSource::StaticScalar(scalar),
        ) => {
            "runtime-binary-buffer-v1".hash(&mut hasher);
            source.id.hash(&mut hasher);
            op.hash(&mut hasher);
            scalar.hash(&mut hasher);
        }
        _ => {
            "runtime-binary-buffer-v2".hash(&mut hasher);
            op.hash(&mut hasher);
            hash_runtime_source(lhs, &mut hasher);
            hash_runtime_source(rhs, &mut hasher);
        }
    }
    dtype.hash(&mut hasher);
    origin.hash(&mut hasher);
    RuntimeBufferId(hasher.finish())
}

fn runtime_lifetimes(
    items: &[RuntimeScheduleItem],
    buffers: &[RuntimeBufferDesc],
) -> Result<Vec<crate::memory_plan::RuntimeAllocationLifetime>, RuntimeScheduleError> {
    buffers
        .iter()
        .map(|buffer| {
            let allocation = items
                .iter()
                .find(|item| {
                    matches!(
                        &item.instruction,
                        RuntimeInstruction::Allocate { output } if output == buffer
                    )
                })
                .ok_or(RuntimeScheduleError::InvalidOrdering(
                    "runtime buffer allocation is absent",
                ))?;
            let final_consumer = items
                .iter()
                .filter(|item| {
                    item.instruction
                        .runtime_inputs()
                        .any(|input| input == buffer)
                })
                .map(|item| item.id)
                .max()
                .unwrap_or_else(|| {
                    items
                        .iter()
                        .filter(|item| {
                            item.output() == Some(RuntimeValueDesc::Dynamic(buffer.clone()))
                        })
                        .map(|item| item.id)
                        .max()
                        .unwrap_or(allocation.id)
                });
            Ok(crate::memory_plan::RuntimeAllocationLifetime::new(
                buffer.id.0,
                allocation.id,
                final_consumer,
            ))
        })
        .collect()
}

impl RuntimeSchedule {
    pub(crate) fn count_plan(
        &self,
        count: RuntimeCountId,
    ) -> Result<&DynamicAllocationPlan, RuntimeScheduleError> {
        self.items
            .iter()
            .find_map(|item| match &item.instruction {
                RuntimeInstruction::Count { plan, output } if *output == count => Some(plan),
                _ => None,
            })
            .ok_or(RuntimeScheduleError::UnknownCount(count))
    }

    pub(crate) fn validate(&self) -> Result<(), RuntimeScheduleError> {
        let mut ids = BTreeSet::new();
        for (position, item) in self.items.iter().enumerate() {
            if item.id != position as u64 || !ids.insert(item.id) {
                return Err(RuntimeScheduleError::InvalidOrdering(
                    "runtime item IDs are not canonical",
                ));
            }
            if item
                .dependencies
                .iter()
                .any(|dependency| *dependency >= item.id)
                || item.dependencies.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(RuntimeScheduleError::InvalidOrdering(
                    "runtime dependency is not produced earlier",
                ));
            }
            for dependency in &item.dependencies {
                self.items.get(*dependency as usize).ok_or(
                    RuntimeScheduleError::InvalidOrdering("runtime dependency is absent"),
                )?;
            }
        }
        let mut counts = BTreeMap::new();
        let mut allocations = BTreeMap::new();
        let mut producers = BTreeMap::new();
        let mut produced_values = HashSet::new();
        let known_buffers = self
            .buffers
            .iter()
            .map(|buffer| (buffer.id, buffer))
            .collect::<BTreeMap<_, _>>();
        if known_buffers.len() != self.buffers.len() {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime buffers are not unique",
            ));
        }
        let expected_buffers = self
            .items
            .iter()
            .filter_map(|item| match &item.instruction {
                RuntimeInstruction::Allocate { output } => Some(output.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if self.buffers != expected_buffers {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime buffer inventory is not canonical",
            ));
        }
        for item in &self.items {
            let instruction = &item.instruction;
            match instruction {
                RuntimeInstruction::Count { plan, output } => {
                    plan.validate_target(DynamicAllocationTarget::RuntimeSchedule)
                        .map_err(RuntimeScheduleError::Plan)?;
                    if !item.dependencies.is_empty()
                        || *output != RuntimeCountId(plan.identity())
                        || counts.insert(*output, item.id).is_some()
                    {
                        return Err(RuntimeScheduleError::InvalidOrdering(
                            "runtime count identity mismatch",
                        ));
                    }
                }
                RuntimeInstruction::Allocate { output } => {
                    let count_item = counts
                        .get(&output.count)
                        .ok_or(RuntimeScheduleError::UnknownCount(output.count))?;
                    if item.dependencies.as_slice() != [*count_item]
                        || known_buffers.get(&output.id) != Some(&output)
                        || allocations.insert(output.id, item.id).is_some()
                    {
                        return Err(RuntimeScheduleError::InvalidOrdering(
                            "runtime allocation contract mismatch",
                        ));
                    }
                }
                RuntimeInstruction::MaterializeNonzero { input, output } => {
                    validate_root_materialization(self, item, output, &[input])?;
                    register_value_producer(&mut producers, output, item.id)?;
                }
                RuntimeInstruction::MaterializeMaskedSelect {
                    input,
                    mask,
                    output,
                } => {
                    validate_root_materialization(self, item, output, &[input, mask])?;
                    register_value_producer(&mut producers, output, item.id)?;
                }
                RuntimeInstruction::Unary {
                    origin,
                    op,
                    input,
                    output,
                } => {
                    if !matches!(op, UnaryOp::Neg | UnaryOp::Square)
                        || input.dtype() != output.dtype()
                        || !output.dtype().is_float()
                        || !valid_unary_output(*origin, input, output, *op)
                    {
                        return Err(RuntimeScheduleError::InvalidOrdering(
                            "runtime unary contract mismatch",
                        ));
                    }
                    validate_compute(self, item, &allocations, output, &[input])?;
                    if let RuntimeValueDesc::Dynamic(output) = output {
                        register_value_producer(&mut producers, output, item.id)?;
                    }
                }
                RuntimeInstruction::Binary {
                    origin,
                    op,
                    lhs,
                    rhs,
                    output,
                } => {
                    if !matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul)
                        || !output.dtype().is_float()
                    {
                        return Err(RuntimeScheduleError::InvalidOrdering(
                            "runtime binary operation is unsupported",
                        ));
                    }
                    let mut invalid_static = false;
                    let promoted = match (lhs, rhs) {
                        (RuntimeValueSource::Produced(lhs), RuntimeValueSource::Produced(rhs)) => {
                            source_lub(lhs.dtype(), rhs.dtype())
                        }
                        (
                            RuntimeValueSource::Produced(lhs),
                            RuntimeValueSource::StaticScalar(rhs),
                        )
                        | (
                            RuntimeValueSource::StaticScalar(rhs),
                            RuntimeValueSource::Produced(lhs),
                        ) => source_lub(lhs.dtype(), rhs.descriptor.dtype),
                        _ => output.dtype(),
                    };
                    let inputs = [lhs, rhs]
                        .into_iter()
                        .filter_map(|source| match source {
                            RuntimeValueSource::Produced(input) => Some(input),
                            RuntimeValueSource::StaticScalar(binding) => {
                                invalid_static |= !valid_static_scalar(binding);
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    if invalid_static
                        || inputs.is_empty()
                        || output.dtype() != promoted
                        || !valid_binary_output(*origin, lhs, rhs, output, *op)
                        || inputs
                            .iter()
                            .filter_map(|input| match input {
                                RuntimeValueDesc::Dynamic(input) => Some(input),
                                RuntimeValueDesc::Fixed(_) => None,
                            })
                            .any(|input| match output {
                                RuntimeValueDesc::Dynamic(output) => {
                                    input.count != output.count || input.shape != output.shape
                                }
                                RuntimeValueDesc::Fixed(_) => true,
                            })
                    {
                        return Err(RuntimeScheduleError::InvalidOrdering(
                            "runtime binary cardinality contract mismatch",
                        ));
                    }
                    validate_compute(self, item, &allocations, output, &inputs)?;
                    if let RuntimeValueDesc::Dynamic(output) = output {
                        register_value_producer(&mut producers, output, item.id)?;
                    }
                }
                RuntimeInstruction::Reduce {
                    origin,
                    op,
                    dtypes,
                    input,
                    output,
                } => {
                    let Some(canonical_dtypes) = dynamic_reduction_dtypes(input.dtype(), *op)
                    else {
                        return Err(RuntimeScheduleError::InvalidOrdering(
                            "runtime reduction contract mismatch",
                        ));
                    };
                    if *dtypes != canonical_dtypes
                        || output.shape != Shape::from([])
                        || output.dtype != canonical_dtypes.output
                        || output
                            != &fixed_reduction_desc(
                                *origin,
                                value_desc_id(input),
                                *op,
                                canonical_dtypes,
                            )
                        || item.dependencies.len() != 1
                        || self.items[item.dependencies[0] as usize].output() != Some(input.clone())
                    {
                        return Err(RuntimeScheduleError::InvalidOrdering(
                            "runtime reduction contract mismatch",
                        ));
                    }
                }
            }
            if let Some(output) = instruction.output()
                && !produced_values.insert(output)
            {
                return Err(RuntimeScheduleError::InvalidOrdering(
                    "runtime value has multiple producers",
                ));
            }
            if item.cache_key != runtime_item_key(item.id, &item.dependencies, &item.instruction) {
                return Err(RuntimeScheduleError::InvalidOrdering(
                    "runtime cache identity mismatch",
                ));
            }
        }
        if allocations.len() != self.buffers.len() {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime allocation inventory mismatch",
            ));
        }
        if producers.len() != self.buffers.len() {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime initialized-value producer inventory mismatch",
            ));
        }
        if self.items.last().and_then(RuntimeScheduleItem::output) != Some(self.output.clone()) {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime output is not the terminal producer",
            ));
        }
        let expected_lifetimes = runtime_lifetimes(&self.items, &self.buffers)?;
        if self.lifetimes != expected_lifetimes {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime allocation lifetime mismatch",
            ));
        }
        for lifetime in &self.lifetimes {
            lifetime
                .validate()
                .map_err(RuntimeScheduleError::InvalidOrdering)?;
        }
        if self.identity != runtime_schedule_identity(self) {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime schedule identity mismatch",
            ));
        }
        Ok(())
    }
}

fn register_value_producer(
    producers: &mut BTreeMap<RuntimeBufferId, u64>,
    output: &RuntimeBufferDesc,
    item: u64,
) -> Result<(), RuntimeScheduleError> {
    if producers.insert(output.id, item).is_some() {
        return Err(RuntimeScheduleError::InvalidOrdering(
            "runtime buffer has multiple initialized-value producers",
        ));
    }
    Ok(())
}

fn validate_root_materialization(
    schedule: &RuntimeSchedule,
    item: &RuntimeScheduleItem,
    output: &RuntimeBufferDesc,
    bindings: &[&DynamicBinding],
) -> Result<(), RuntimeScheduleError> {
    let allocation = item
        .dependencies
        .iter()
        .find_map(
            |dependency| match &schedule.items[*dependency as usize].instruction {
                RuntimeInstruction::Allocate { output: allocated } if allocated == output => {
                    Some(*dependency)
                }
                _ => None,
            },
        )
        .ok_or(RuntimeScheduleError::InvalidOrdering(
            "runtime materialization allocation is absent",
        ))?;
    let plan = schedule.count_plan(output.count)?;
    let expected = plan.bindings();
    if item.dependencies.as_slice() != [allocation]
        || bindings != expected.as_slice()
        || schedule
            .buffers
            .iter()
            .find(|buffer| buffer.id == output.id)
            != Some(output)
        || output.dtype != plan.output_dtype()
        || output.shape != plan.output_shape()
    {
        return Err(RuntimeScheduleError::InvalidOrdering(
            "runtime root materialization contract mismatch",
        ));
    }
    Ok(())
}
fn validate_compute(
    schedule: &RuntimeSchedule,
    item: &RuntimeScheduleItem,
    allocations: &BTreeMap<RuntimeBufferId, u64>,
    output: &RuntimeValueDesc,
    inputs: &[&RuntimeValueDesc],
) -> Result<(), RuntimeScheduleError> {
    let mut expected_dependencies = Vec::with_capacity(inputs.len() + 1);
    if let RuntimeValueDesc::Dynamic(output) = output {
        if schedule
            .buffers
            .iter()
            .find(|buffer| buffer.id == output.id)
            != Some(output)
        {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime compute descriptor is not canonical",
            ));
        }
        let allocation = allocations
            .get(&output.id)
            .ok_or(RuntimeScheduleError::UnknownBuffer(output.id))?;
        expected_dependencies.push(*allocation);
    }
    for input in inputs {
        if let RuntimeValueDesc::Dynamic(input) = input
            && schedule.buffers.iter().find(|buffer| buffer.id == input.id) != Some(input)
        {
            return Err(RuntimeScheduleError::InvalidOrdering(
                "runtime input descriptor is not canonical",
            ));
        }
        let producer = schedule
            .items
            .iter()
            .find(|producer| producer.output() == Some((*input).clone()))
            .ok_or(RuntimeScheduleError::InvalidOrdering(
                "runtime compute producer is absent",
            ))?;
        expected_dependencies.push(producer.id);
    }
    expected_dependencies.sort_unstable();
    expected_dependencies.dedup();
    if item.dependencies != expected_dependencies {
        return Err(RuntimeScheduleError::InvalidOrdering(
            "runtime compute dependencies are not canonical",
        ));
    }
    Ok(())
}

fn valid_static_scalar(binding: &StaticScalarBinding) -> bool {
    binding.descriptor.shape.numel().ok() == Some(1)
        && binding.descriptor.bytes == binding.descriptor.dtype.itemsize()
        && binding.descriptor.alignment == binding.descriptor.dtype.itemsize().max(1)
        && binding.descriptor.read_only
        && binding.descriptor.view.is_none()
        && binding.descriptor.id == derived_static_scalar_id(binding.node, binding.descriptor.dtype)
}

fn valid_unary_output(
    origin: usize,
    input: &RuntimeValueDesc,
    output: &RuntimeValueDesc,
    op: UnaryOp,
) -> bool {
    match (input, output) {
        (RuntimeValueDesc::Dynamic(input), RuntimeValueDesc::Dynamic(output)) => {
            input.count == output.count
                && input.shape == output.shape
                && output.id == unary_buffer_id(origin, input.id, op, output.dtype)
        }
        (RuntimeValueDesc::Fixed(input), RuntimeValueDesc::Fixed(output)) => {
            fixed_pointwise_desc(
                "runtime-fixed-unary-v1",
                derived_branch_id(origin, input.id),
                input.shape.clone(),
                output.dtype,
            )
            .ok()
            .as_ref()
                == Some(output)
        }
        _ => false,
    }
}

fn valid_binary_output(
    origin: usize,
    lhs: &RuntimeValueSource,
    rhs: &RuntimeValueSource,
    output: &RuntimeValueDesc,
    op: BinaryOp,
) -> bool {
    let id = binary_buffer_id(origin, op, lhs, rhs, output.dtype());
    match output {
        RuntimeValueDesc::Dynamic(output) => output.id == id,
        RuntimeValueDesc::Fixed(output) => {
            fixed_pointwise_desc(
                "runtime-fixed-binary-v1",
                id.0,
                Shape::from([]),
                output.dtype,
            )
            .ok()
            .as_ref()
                == Some(output)
        }
    }
}

impl RuntimeBufferTable {
    pub(crate) fn new(schedule: &RuntimeSchedule) -> Result<Self, RuntimeScheduleError> {
        schedule.validate()?;
        Ok(Self {
            descriptors: schedule.buffers.clone(),
            allocations: BTreeMap::new(),
        })
    }

    pub(crate) fn allocate_buffer_after_count(
        &mut self,
        schedule: &RuntimeSchedule,
        id: RuntimeBufferId,
        count: RuntimeCount,
    ) -> Result<&DynamicAllocation, RuntimeScheduleError> {
        schedule.validate()?;
        let descriptor = self.descriptor(id)?.clone();
        if descriptor.count != count.id {
            return Err(RuntimeScheduleError::UnknownCount(count.id));
        }
        if self.allocations.contains_key(&id) {
            return Err(RuntimeScheduleError::DuplicateAllocation(id));
        }
        let shape = descriptor.shape.resolve(count.value).map_err(|_| {
            RuntimeScheduleError::Plan(DynamicAllocationError::AllocationOverflow {
                elements: count.value,
                dtype: descriptor.dtype,
            })
        })?;
        let elements = shape.numel().map_err(|_| {
            RuntimeScheduleError::Plan(DynamicAllocationError::AllocationOverflow {
                elements: count.value,
                dtype: descriptor.dtype,
            })
        })?;
        let bytes =
            elements
                .checked_mul(descriptor.dtype.itemsize())
                .ok_or(RuntimeScheduleError::Plan(
                    DynamicAllocationError::AllocationOverflow {
                        elements,
                        dtype: descriptor.dtype,
                    },
                ))?;
        self.allocations.insert(
            id,
            DynamicAllocation {
                shape,
                dtype: descriptor.dtype,
                elements,
                bytes,
            },
        );
        self.allocation(id)
    }

    pub(crate) fn allocation(
        &self,
        id: RuntimeBufferId,
    ) -> Result<&DynamicAllocation, RuntimeScheduleError> {
        self.descriptor(id)?;
        self.allocations
            .get(&id)
            .ok_or(RuntimeScheduleError::LiveLookupBeforeAllocation(id))
    }

    pub(crate) fn descriptor(
        &self,
        id: RuntimeBufferId,
    ) -> Result<&RuntimeBufferDesc, RuntimeScheduleError> {
        self.descriptors
            .iter()
            .find(|descriptor| descriptor.id == id)
            .ok_or(RuntimeScheduleError::UnknownBuffer(id))
    }
}

fn runtime_item_key(id: u64, dependencies: &[u64], instruction: &RuntimeInstruction) -> u64 {
    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    dependencies.hash(&mut hasher);
    match instruction {
        RuntimeInstruction::Count { plan, output } => {
            0_u8.hash(&mut hasher);
            plan.identity().hash(&mut hasher);
            output.hash(&mut hasher);
        }
        RuntimeInstruction::Allocate { output } => {
            1_u8.hash(&mut hasher);
            hash_runtime_buffer(output, &mut hasher);
        }
        RuntimeInstruction::MaterializeNonzero { input, output } => {
            2_u8.hash(&mut hasher);
            input.hash(&mut hasher);
            hash_runtime_buffer(output, &mut hasher);
        }
        RuntimeInstruction::MaterializeMaskedSelect {
            input,
            mask,
            output,
        } => {
            3_u8.hash(&mut hasher);
            input.hash(&mut hasher);
            mask.hash(&mut hasher);
            hash_runtime_buffer(output, &mut hasher);
        }
        RuntimeInstruction::Unary {
            origin,
            op,
            input,
            output,
        } => {
            4_u8.hash(&mut hasher);
            origin.hash(&mut hasher);
            op.hash(&mut hasher);
            hash_runtime_value(input, &mut hasher);
            hash_runtime_value(output, &mut hasher);
        }
        RuntimeInstruction::Binary {
            origin,
            op,
            lhs,
            rhs,
            output,
        } => {
            5_u8.hash(&mut hasher);
            origin.hash(&mut hasher);
            op.hash(&mut hasher);
            hash_runtime_source(lhs, &mut hasher);
            hash_runtime_source(rhs, &mut hasher);
            hash_runtime_value(output, &mut hasher);
        }
        RuntimeInstruction::Reduce {
            origin,
            op,
            dtypes,
            input,
            output,
        } => {
            6_u8.hash(&mut hasher);
            origin.hash(&mut hasher);
            op.hash(&mut hasher);
            dtypes.hash(&mut hasher);
            hash_runtime_value(input, &mut hasher);
            output.hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn runtime_schedule_identity(schedule: &RuntimeSchedule) -> u64 {
    let mut hasher = DefaultHasher::new();
    for item in &schedule.items {
        item.id.hash(&mut hasher);
        item.dependencies.hash(&mut hasher);
        item.cache_key.hash(&mut hasher);
    }
    schedule.buffers.len().hash(&mut hasher);
    for buffer in &schedule.buffers {
        hash_runtime_buffer(buffer, &mut hasher);
    }
    hash_runtime_value(&schedule.output, &mut hasher);
    schedule.lifetimes.hash(&mut hasher);
    hasher.finish()
}

fn hash_runtime_buffer(buffer: &RuntimeBufferDesc, hasher: &mut DefaultHasher) {
    buffer.id.hash(hasher);
    buffer.dtype.hash(hasher);
    match buffer.shape {
        crate::DynamicOutputShape::Scalar => 0_u8.hash(hasher),
        crate::DynamicOutputShape::Count1d { .. } => 1_u8.hash(hasher),
        crate::DynamicOutputShape::CountRows { width, .. } => {
            2_u8.hash(hasher);
            width.hash(hasher);
        }
    }
    buffer.count.hash(hasher);
}

fn hash_runtime_value(value: &RuntimeValueDesc, hasher: &mut DefaultHasher) {
    match value {
        RuntimeValueDesc::Dynamic(value) => {
            0_u8.hash(hasher);
            hash_runtime_buffer(value, hasher);
        }
        RuntimeValueDesc::Fixed(value) => {
            1_u8.hash(hasher);
            value.hash(hasher);
        }
    }
}

fn hash_runtime_source(source: &RuntimeValueSource, hasher: &mut DefaultHasher) {
    match source {
        RuntimeValueSource::Produced(value) => {
            0_u8.hash(hasher);
            hash_runtime_value(value, hasher);
        }
        RuntimeValueSource::StaticScalar(value) => {
            1_u8.hash(hasher);
            value.hash(hasher);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, DynamicInput, Graph, Scalar, TensorData};

    #[test]
    fn nonzero_schedule_retains_count_rows_shape() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 3], DType::F32);
        let output = graph.nonzero(input).unwrap();
        let schedule = schedule_dynamic(&graph, output).unwrap();
        let RuntimeValueDesc::Dynamic(output) = &schedule.output else {
            panic!("nonzero output must remain runtime-sized")
        };
        let plan = schedule.count_plan(output.count).unwrap();
        assert_eq!(
            plan.allocation_for_count(0).unwrap().shape,
            Shape::from([0, 2])
        );
        assert_eq!(
            plan.allocation_for_count(3).unwrap().shape,
            Shape::from([3, 2])
        );
        assert!(schedule.items.iter().any(|item| matches!(
            &item.instruction,
            RuntimeInstruction::MaterializeNonzero { .. }
        )));

        let mut equivalent = Graph::new();
        let input = equivalent.input_dtype("input", [2, 3], DType::F32);
        let output = equivalent.nonzero(input).unwrap();
        let equivalent = schedule_dynamic(&equivalent, output).unwrap();
        assert_eq!(schedule.identity, equivalent.identity);
        assert_eq!(
            schedule
                .items
                .iter()
                .map(|item| item.cache_key)
                .collect::<Vec<_>>(),
            equivalent
                .items
                .iter()
                .map(|item| item.cache_key)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn arbitrary_chain_and_same_count_dynamic_binary_form_one_dag() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let mask = graph.input_dtype("mask", [1, 2], DType::Bool);
        let selected = graph.masked_select_dynamic(input, mask).unwrap();
        let left = graph.dynamic_unary(selected, UnaryOp::Neg).unwrap();
        let right = graph.dynamic_unary(selected, UnaryOp::Square).unwrap();
        let added = graph
            .dynamic_binary(left, DynamicInput::Dynamic(right), BinaryOp::Add)
            .unwrap();
        let squared = graph.dynamic_unary(added, UnaryOp::Square).unwrap();
        let output = graph.dynamic_sum(squared).unwrap();
        let schedule = schedule_dynamic(&graph, output).unwrap();
        assert_eq!(
            schedule
                .items
                .iter()
                .filter(|item| matches!(&item.instruction, RuntimeInstruction::Count { .. }))
                .count(),
            1
        );
        assert_eq!(
            schedule
                .items
                .iter()
                .filter(|item| matches!(&item.instruction, RuntimeInstruction::Unary { .. }))
                .count(),
            3
        );
        assert!(schedule.items.iter().any(|item| matches!(
            &item.instruction,
            RuntimeInstruction::Binary {
                lhs: RuntimeValueSource::Produced(RuntimeValueDesc::Dynamic(_)),
                rhs: RuntimeValueSource::Produced(RuntimeValueDesc::Dynamic(_)),
                ..
            }
        )));
        assert!(matches!(schedule.output, RuntimeValueDesc::Fixed(_)));
    }

    #[test]
    fn runtime_shape_overflow_rejects_before_allocation_publication() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1, 2], DType::F32);
        let output = graph.nonzero(input).unwrap();
        let schedule = schedule_dynamic(&graph, output).unwrap();
        let RuntimeValueDesc::Dynamic(output) = schedule.output.clone() else {
            panic!("expected runtime output")
        };
        let mut table = RuntimeBufferTable::new(&schedule).unwrap();
        assert!(matches!(
            table.allocate_buffer_after_count(
                &schedule,
                output.id,
                RuntimeCount {
                    id: output.count,
                    value: usize::MAX,
                },
            ),
            Err(RuntimeScheduleError::Plan(
                DynamicAllocationError::AllocationOverflow { .. }
            ))
        ));
        assert!(matches!(
            table.allocation(output.id),
            Err(RuntimeScheduleError::LiveLookupBeforeAllocation(_))
        ));
    }

    #[test]
    fn scalar_binding_is_embedded_in_binary_instruction() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let mask = graph.input_dtype("mask", [2], DType::Bool);
        let scalar = graph.constant(TensorData::scalar_with_dtype(Scalar::F(2.0), DType::F32));
        let selected = graph.masked_select_dynamic(input, mask).unwrap();
        let output = graph
            .dynamic_binary(selected, DynamicInput::StaticScalar(scalar), BinaryOp::Mul)
            .unwrap();
        let schedule = schedule_dynamic(&graph, output).unwrap();
        assert!(schedule.items.iter().any(|item| matches!(
            &item.instruction,
            RuntimeInstruction::Binary {
                rhs: RuntimeValueSource::StaticScalar(binding),
                ..
            } if binding.node == scalar
        )));
    }

    #[test]
    fn fixed_reductions_remain_composable_runtime_values() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [3], DType::F32);
        let mask = graph.input_dtype("mask", [3], DType::Bool);
        let selected = graph.masked_select_dynamic(input, mask).unwrap();
        let sum = graph.dynamic_sum(selected).unwrap();
        let mean = graph.dynamic_mean(selected).unwrap();
        let output = graph
            .dynamic_binary(sum, DynamicInput::Dynamic(mean), BinaryOp::Add)
            .unwrap();
        let schedule = schedule_dynamic(&graph, output).unwrap();
        assert!(matches!(schedule.output, RuntimeValueDesc::Fixed(_)));
        assert_eq!(
            schedule
                .items
                .iter()
                .filter(|item| matches!(&item.instruction, RuntimeInstruction::Reduce { .. }))
                .count(),
            2
        );
        assert!(schedule.items.iter().any(|item| matches!(
            &item.instruction,
            RuntimeInstruction::Binary {
                lhs: RuntimeValueSource::Produced(RuntimeValueDesc::Fixed(_)),
                rhs: RuntimeValueSource::Produced(RuntimeValueDesc::Fixed(_)),
                output: RuntimeValueDesc::Fixed(_),
                ..
            }
        )));
    }

    #[test]
    fn distinct_equivalent_branches_retain_graph_independent_node_identity() {
        fn build() -> RuntimeSchedule {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [3], DType::F32);
            let mask = graph.input_dtype("mask", [3], DType::Bool);
            let selected = graph.masked_select_dynamic(input, mask).unwrap();
            let left = graph.dynamic_unary(selected, UnaryOp::Neg).unwrap();
            let right = graph.dynamic_unary(selected, UnaryOp::Neg).unwrap();
            let sum_left = graph.dynamic_sum(selected).unwrap();
            let sum_right = graph.dynamic_sum(selected).unwrap();
            let duplicate_reductions = graph
                .dynamic_binary(sum_left, DynamicInput::Dynamic(sum_right), BinaryOp::Add)
                .unwrap();
            let duplicate_unaries = graph
                .dynamic_binary(left, DynamicInput::Dynamic(right), BinaryOp::Add)
                .unwrap();
            let duplicate_unary_sum = graph.dynamic_sum(duplicate_unaries).unwrap();
            let output = graph
                .dynamic_binary(
                    duplicate_reductions,
                    DynamicInput::Dynamic(duplicate_unary_sum),
                    BinaryOp::Add,
                )
                .unwrap();
            schedule_dynamic(&graph, output).unwrap()
        }

        let schedule = build();
        let unary_outputs = schedule
            .items
            .iter()
            .filter_map(|item| match &item.instruction {
                RuntimeInstruction::Unary { output, .. } => Some(output.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(unary_outputs.len(), 2);
        assert_ne!(unary_outputs[0], unary_outputs[1]);
        let reduction_outputs = schedule
            .items
            .iter()
            .filter_map(|item| match &item.instruction {
                RuntimeInstruction::Reduce { output, .. } => Some(output.id),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(reduction_outputs.len(), 3);
        assert_ne!(reduction_outputs[0], reduction_outputs[1]);
        let equivalent = build();
        assert_eq!(schedule.identity, equivalent.identity);
        assert_eq!(
            schedule
                .items
                .iter()
                .map(|item| item.cache_key)
                .collect::<Vec<_>>(),
            equivalent
                .items
                .iter()
                .map(|item| item.cache_key)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn validation_rejects_spurious_edges_and_descriptor_drift() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let mask = graph.input_dtype("mask", [2], DType::Bool);
        let selected = graph.masked_select_dynamic(input, mask).unwrap();
        let output = graph.dynamic_unary(selected, UnaryOp::Neg).unwrap();
        let schedule = schedule_dynamic(&graph, output).unwrap();

        let mut spurious = schedule.clone();
        let unary = spurious
            .items
            .iter_mut()
            .find(|item| matches!(&item.instruction, RuntimeInstruction::Unary { .. }))
            .unwrap();
        unary.dependencies.push(0);
        unary.dependencies.sort_unstable();
        unary.dependencies.dedup();
        unary.cache_key = runtime_item_key(unary.id, &unary.dependencies, &unary.instruction);
        assert!(matches!(
            spurious.validate(),
            Err(RuntimeScheduleError::InvalidOrdering(
                "runtime compute dependencies are not canonical"
            ))
        ));

        let mut drifted = schedule.clone();
        let unary = drifted
            .items
            .iter_mut()
            .find(|item| matches!(&item.instruction, RuntimeInstruction::Unary { .. }))
            .unwrap();
        let RuntimeInstruction::Unary {
            output: RuntimeValueDesc::Dynamic(output),
            ..
        } = &mut unary.instruction
        else {
            panic!("expected a runtime unary output")
        };
        output.dtype = DType::F64;
        unary.cache_key = runtime_item_key(unary.id, &unary.dependencies, &unary.instruction);
        assert!(matches!(
            drifted.validate(),
            Err(RuntimeScheduleError::InvalidOrdering(_))
        ));

        let mut duplicate = schedule;
        duplicate.buffers.push(duplicate.buffers[0].clone());
        assert!(matches!(
            duplicate.validate(),
            Err(RuntimeScheduleError::InvalidOrdering(
                "runtime buffers are not unique"
            ))
        ));

        let mut reordered = duplicate;
        reordered.buffers.pop();
        reordered.buffers.reverse();
        reordered.identity = runtime_schedule_identity(&reordered);
        assert!(matches!(
            reordered.validate(),
            Err(RuntimeScheduleError::InvalidOrdering(
                "runtime buffer inventory is not canonical"
            ))
        ));

        let mut identity_only = reordered;
        identity_only.buffers.reverse();
        identity_only.identity = runtime_schedule_identity(&identity_only) ^ 1;
        assert!(matches!(
            identity_only.validate(),
            Err(RuntimeScheduleError::InvalidOrdering(
                "runtime schedule identity mismatch"
            ))
        ));
    }

    #[test]
    fn reduction_dtype_is_derived_instead_of_trusted_from_the_output() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let mask = graph.input_dtype("mask", [2], DType::Bool);
        let selected = graph.masked_select_dynamic(input, mask).unwrap();
        let output = graph.dynamic_sum(selected).unwrap();
        let mut schedule = schedule_dynamic(&graph, output).unwrap();
        let reduction = schedule
            .items
            .iter_mut()
            .find(|item| matches!(&item.instruction, RuntimeInstruction::Reduce { .. }))
            .unwrap();
        let RuntimeInstruction::Reduce {
            origin,
            op,
            dtypes,
            input,
            output,
        } = &mut reduction.instruction
        else {
            unreachable!()
        };
        let forged_dtypes = ReductionDType::new(DType::F64, DType::F64);
        *dtypes = forged_dtypes;
        let forged = fixed_reduction_desc(*origin, value_desc_id(input), *op, forged_dtypes);
        *output = forged.clone();
        reduction.cache_key = runtime_item_key(
            reduction.id,
            &reduction.dependencies,
            &reduction.instruction,
        );
        schedule.output = RuntimeValueDesc::Fixed(forged);
        schedule.identity = runtime_schedule_identity(&schedule);
        assert!(matches!(
            schedule.validate(),
            Err(RuntimeScheduleError::InvalidOrdering(
                "runtime reduction contract mismatch"
            ))
        ));

        for (dtype, sum_dtypes, mean_dtypes) in [
            (
                DType::Bool,
                ReductionDType::new(DType::I32, DType::I32),
                ReductionDType::new(DType::I32, DType::F32),
            ),
            (
                DType::I8,
                ReductionDType::new(DType::I32, DType::I32),
                ReductionDType::new(DType::I32, DType::F32),
            ),
            (
                DType::F16,
                ReductionDType::new(DType::F32, DType::F16),
                ReductionDType::new(DType::F32, DType::F16),
            ),
            (
                DType::BF16,
                ReductionDType::new(DType::F32, DType::BF16),
                ReductionDType::new(DType::F32, DType::BF16),
            ),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [2], dtype);
            let mask = graph.input_dtype("mask", [2], DType::Bool);
            let selected = graph.masked_select_dynamic(input, mask).unwrap();
            for (node, expected) in [
                (graph.dynamic_sum(selected).unwrap(), sum_dtypes),
                (graph.dynamic_mean(selected).unwrap(), mean_dtypes),
            ] {
                let schedule = schedule_dynamic(&graph, node).unwrap();
                assert_eq!(schedule.output.dtype(), expected.output, "{dtype:?}");
                let actual = schedule
                    .items
                    .iter()
                    .find_map(|item| match &item.instruction {
                        RuntimeInstruction::Reduce { dtypes, .. } => Some(*dtypes),
                        _ => None,
                    })
                    .unwrap();
                assert_eq!(actual, expected, "{dtype:?}");
            }
        }
    }

    #[test]
    fn validation_rejects_a_produced_intermediate_as_the_requested_output() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
        let mask = graph.input_dtype("mask", [2], DType::Bool);
        let selected = graph.masked_select_dynamic(input, mask).unwrap();
        let negated = graph.dynamic_unary(selected, UnaryOp::Neg).unwrap();
        let sum = graph.dynamic_sum(negated).unwrap();
        let mut schedule = schedule_dynamic(&graph, sum).unwrap();
        let intermediate = schedule
            .items
            .iter()
            .find_map(|item| match &item.instruction {
                RuntimeInstruction::MaterializeMaskedSelect { output, .. } => {
                    Some(RuntimeValueDesc::Dynamic(output.clone()))
                }
                _ => None,
            })
            .unwrap();
        schedule.output = intermediate;
        schedule.identity = runtime_schedule_identity(&schedule);
        assert!(matches!(
            schedule.validate(),
            Err(RuntimeScheduleError::InvalidOrdering(
                "runtime output is not the terminal producer"
            ))
        ));
    }
}
