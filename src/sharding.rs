//! Static, backend-neutral tensor sharding metadata and exact host reference data movement.
//!
//! This module deliberately does not attach layouts to [`crate::Graph`] or execute a device
//! kernel.  It is the semantic boundary used by a later scheduler/runtime implementation.

use crate::collective::{DeviceGroup, DeviceId};
use crate::{
    CudaPlanBinding, CudaPlanStage, CudaTransferRoute, DType, Error, ExecutableBuffer,
    ExecutableBufferRole, ExecutableShardedCudaPlan, Result, Shape, ShardedCudaPlan,
    ShardedGraphTensor, Storage, TensorData,
};
use std::collections::{BTreeMap, BTreeSet};

/// An exact half-open interval of a global sharded dimension.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ShardRange {
    pub start: usize,
    pub end: usize,
}
impl ShardRange {
    pub fn len(self) -> usize {
        self.end - self.start
    }
    pub fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// The ownership form of a tensor.  Axis ranges follow `DeviceGroup` caller order.
#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum ShardDistribution {
    Replicated,
    Axis {
        axis: usize,
        ranges: Vec<ShardRange>,
    },
}

/// Immutable global tensor metadata plus a device ownership layout.
#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ShardLayout {
    group: DeviceGroup,
    global_shape: Shape,
    dtype: DType,
    distribution: ShardDistribution,
    cache_key: String,
}
impl ShardLayout {
    pub fn replicated(
        group: DeviceGroup,
        global_shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<Self> {
        Self::build(
            group,
            global_shape.into(),
            dtype,
            ShardDistribution::Replicated,
        )
    }

    /// Creates tinygrad-compatible equal axis shards.  Negative axes are normalized.
    /// tinygrad rejects dimensions that are not exactly divisible by the device count.
    pub fn axis_sharded(
        group: DeviceGroup,
        global_shape: impl Into<Shape>,
        dtype: DType,
        axis: isize,
    ) -> Result<Self> {
        let shape = global_shape.into();
        if shape.rank() == 0 {
            return Self::replicated(group, shape, dtype);
        }
        let axis = normalize_axis(axis, shape.rank())?;
        let dim = shape.dims()[axis];
        if dim % group.len() != 0 {
            return Err(shard_error(format!(
                "multi axis uneven: shape[{axis}]={dim} is not divisible by {} devices",
                group.len()
            )));
        }
        let width = dim / group.len();
        let ranges = (0..group.len())
            .map(|i| ShardRange {
                start: i * width,
                end: (i + 1) * width,
            })
            .collect();
        Self::build(
            group,
            shape,
            dtype,
            ShardDistribution::Axis { axis, ranges },
        )
    }

    fn build(
        group: DeviceGroup,
        global_shape: Shape,
        dtype: DType,
        distribution: ShardDistribution,
    ) -> Result<Self> {
        let cache_key = canonical_layout_key(&group, &global_shape, dtype, &distribution);
        let layout = Self {
            group,
            global_shape,
            dtype,
            distribution,
            cache_key,
        };
        layout.validate()?;
        Ok(layout)
    }

    /// Revalidates a layout that may have crossed a serde or public artifact boundary.
    /// This is pure: callers can reject malformed ownership/range metadata before creating
    /// local views, consuming caller shards, or entering a device executor.
    pub fn validate(&self) -> Result<()> {
        if self.group.devices().is_empty()
            || self.group.devices().iter().collect::<BTreeSet<_>>().len() != self.group.len()
        {
            return Err(shard_error(
                "shard layout requires distinct nonempty device owners",
            ));
        }
        self.global_shape.numel()?;
        if let ShardDistribution::Axis { axis, ranges } = &self.distribution {
            if *axis >= self.global_shape.rank() || ranges.len() != self.group.len() {
                return Err(shard_error("invalid axis-sharded layout"));
            }
            let mut cursor = 0;
            for range in ranges {
                if range.start != cursor
                    || range.end < range.start
                    || range.end > self.global_shape.dims()[*axis]
                {
                    return Err(shard_error(
                        "shard ranges must exactly cover the global axis in device order",
                    ));
                }
                cursor = range.end;
            }
            if cursor != self.global_shape.dims()[*axis] {
                return Err(shard_error("shard ranges do not cover global axis"));
            }
        }
        if self.cache_key
            != canonical_layout_key(
                &self.group,
                &self.global_shape,
                self.dtype,
                &self.distribution,
            )
        {
            return Err(shard_error("shard layout cache identity is noncanonical"));
        }
        Ok(())
    }
    pub fn group(&self) -> &DeviceGroup {
        &self.group
    }
    pub fn global_shape(&self) -> &Shape {
        &self.global_shape
    }
    pub const fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn distribution(&self) -> &ShardDistribution {
        &self.distribution
    }
    pub fn cache_key(&self) -> &str {
        &self.cache_key
    }
    pub fn local_shape(&self, index: usize) -> Result<Shape> {
        self.validate()?;
        if index >= self.group.len() {
            return Err(shard_error("device index is outside layout group"));
        }
        let mut dims = self.global_shape.dims().to_vec();
        if let ShardDistribution::Axis { axis, ranges } = &self.distribution {
            dims[*axis] = ranges[index].len();
        }
        Ok(Shape::from(dims))
    }
}

/// Builds the transfer-only executable companion for one graph-composed static
/// redistribution. Routes are copied from the typed graph trace while source
/// and destination local node identities are still available; execution never
/// has to infer a route from a label or a rendered kernel.
pub fn executable_redistribution_plan(
    source: &ShardedGraphTensor,
    destination: &ShardedGraphTensor,
    bindings: &[CudaPlanBinding],
) -> Result<ExecutableShardedCudaPlan> {
    if source.graph_id() != destination.graph_id()
        || source.layout().group() != destination.layout().group()
        || source.dtype() != destination.dtype()
        || source.global_shape() != destination.global_shape()
    {
        return Err(shard_error("redistribution graph/layout contract mismatch"));
    }
    let group = source.layout().group();
    if bindings.len() != group.len()
        || bindings
            .iter()
            .enumerate()
            .any(|(rank, binding)| binding.device != group.devices()[rank])
        || bindings
            .iter()
            .map(|binding| binding.context.identity())
            .collect::<BTreeSet<_>>()
            .len()
            != bindings.len()
    {
        return Err(shard_error(
            "redistribution CUDA bindings do not match ordered device owners",
        ));
    }
    let trace = destination
        .trace()
        .steps
        .iter()
        .rev()
        .find(|step| step.action == "redistribute" || step.action == "gather-movement")
        .ok_or_else(|| shard_error("redistribution destination has no typed route trace"))?;
    if trace.routes.is_empty() {
        return Err(shard_error("redistribution trace has no concrete routes"));
    }
    let mut routes = Vec::with_capacity(trace.routes.len());
    for route in &trace.routes {
        if route.source_rank >= group.len()
            || route.destination_rank >= group.len()
            || route.source_device != group.devices()[route.source_rank]
            || route.destination_device != group.devices()[route.destination_rank]
            || source.nodes()[route.source_rank] != route.source_node
            || destination.nodes()[route.destination_rank] != route.destination_node
        {
            return Err(shard_error(
                "redistribution route does not match source/destination identities",
            ));
        }
        let bytes = route
            .elements
            .checked_mul(source.dtype().itemsize())
            .ok_or_else(|| shard_error("redistribution byte overflow"))?;
        let source_elements = source.layout().local_shape(route.source_rank)?.numel()?;
        let destination_elements = destination
            .layout()
            .local_shape(route.destination_rank)?
            .numel()?;
        if route
            .source_offset
            .checked_add(route.elements)
            .is_none_or(|end| end > source_elements)
            || route
                .destination_offset
                .checked_add(route.elements)
                .is_none_or(|end| end > destination_elements)
        {
            return Err(shard_error(
                "redistribution route range exceeds local layout",
            ));
        }
        routes.push(CudaTransferRoute {
            source_rank: route.source_rank,
            source_device: route.source_device.clone(),
            source_buffer: route.source_node.index() as u64,
            source_element_offset: route.source_offset,
            destination_rank: route.destination_rank,
            destination_device: route.destination_device.clone(),
            destination_buffer: route.destination_node.index() as u64,
            destination_element_offset: route.destination_offset,
            elements: route.elements,
            bytes,
            dtype: source.dtype(),
        });
    }
    let stage = CudaPlanStage::Transfer {
        id: 0,
        action: trace.action.into(),
        routes,
        dependencies: vec![],
    };
    let mut buffers = BTreeMap::new();
    for (rank, node) in source.nodes().iter().enumerate() {
        let entry = redistribution_buffer(
            rank,
            group.devices()[rank].clone(),
            bindings[rank].context.identity(),
            node.index() as u64,
            source.dtype(),
            source.layout().local_shape(rank)?,
            ExecutableBufferRole::External,
        )?;
        if buffers.insert((rank, node.index() as u64), entry).is_some() {
            return Err(shard_error(
                "redistribution source and destination buffer identity alias",
            ));
        }
    }
    for (rank, node) in destination.nodes().iter().enumerate() {
        let entry = redistribution_buffer(
            rank,
            group.devices()[rank].clone(),
            bindings[rank].context.identity(),
            node.index() as u64,
            destination.dtype(),
            destination.layout().local_shape(rank)?,
            ExecutableBufferRole::Output,
        )?;
        if buffers.insert((rank, node.index() as u64), entry).is_some() {
            return Err(shard_error(
                "redistribution source and destination buffer identity alias",
            ));
        }
    }
    let bindings_key = bindings
        .iter()
        .map(|binding| format!("{}:{}", binding.context.identity(), binding.capability.sm()))
        .collect::<Vec<_>>();
    let logical = ShardedCudaPlan {
        graph_id: source.graph_id(),
        layout_key: destination.layout().cache_key().into(),
        bindings: bindings
            .iter()
            .map(|binding| {
                (
                    binding.device.clone(),
                    binding.context.identity(),
                    binding.capability.sm(),
                )
            })
            .collect(),
        stages: vec![stage],
        diagnostics: vec![],
        cache_key: format!(
            "sharded-cuda-redistribute:v1:{}:{}:{}",
            source.layout().cache_key(),
            destination.layout().cache_key(),
            bindings_key.join(",")
        ),
        materializations: vec![],
    };
    Ok(ExecutableShardedCudaPlan {
        logical,
        owners: bindings
            .iter()
            .map(|binding| binding.context.clone())
            .collect(),
        kernels: vec![None],
        buffers: buffers.into_values().collect(),
    })
}

fn redistribution_buffer(
    rank: usize,
    device: DeviceId,
    owner_identity: usize,
    buffer: u64,
    dtype: DType,
    shape: Shape,
    role: ExecutableBufferRole,
) -> Result<ExecutableBuffer> {
    let bytes = shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| shard_error("redistribution buffer byte overflow"))?;
    Ok(ExecutableBuffer {
        rank,
        device,
        owner_identity,
        buffer,
        dtype,
        shape,
        bytes,
        producer: matches!(role, ExecutableBufferRole::Output).then_some(0),
        consumers: matches!(role, ExecutableBufferRole::External)
            .then_some(0)
            .into_iter()
            .collect(),
        first_stage: 0,
        last_stage: 0,
        role,
    })
}

/// One exact dense shard associated with its semantic device identity.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceShard {
    pub device: DeviceId,
    pub data: TensorData,
}

/// Reference-only sharded tensor data.  Copies preserve `Storage` variants and raw float bits.
#[derive(Clone, Debug, PartialEq)]
pub struct ShardedTensorData {
    layout: ShardLayout,
    shards: Vec<DeviceShard>,
}
impl ShardedTensorData {
    pub fn new(layout: ShardLayout, shards: Vec<DeviceShard>) -> Result<Self> {
        layout.validate()?;
        if shards.len() != layout.group.len() {
            return Err(shard_error("shard count does not match device group"));
        }
        for (i, shard) in shards.iter().enumerate() {
            if shard.device != layout.group.devices()[i] {
                return Err(shard_error("shards must appear once in DeviceGroup order"));
            }
            if shard.data.dtype() != layout.dtype {
                return Err(shard_error("shard dtype does not match layout"));
            }
            if shard.data.shape() != &layout.local_shape(i)? {
                return Err(shard_error("shard shape does not match layout"));
            }
        }
        if matches!(layout.distribution, ShardDistribution::Replicated)
            && shards
                .windows(2)
                .any(|p| !same_storage(p[0].data.storage(), p[1].data.storage()))
        {
            return Err(shard_error(
                "replicated shards must contain identical raw storage",
            ));
        }
        Ok(Self { layout, shards })
    }
    pub fn shard(data: &TensorData, group: DeviceGroup, axis: Option<isize>) -> Result<Self> {
        let layout = match axis {
            Some(axis) => {
                ShardLayout::axis_sharded(group, data.shape().clone(), data.dtype(), axis)?
            }
            None => ShardLayout::replicated(group, data.shape().clone(), data.dtype())?,
        };
        let shards = (0..layout.group.len())
            .map(|i| {
                let local_shape = layout.local_shape(i)?;
                let indices = local_global_indices(&layout, i)?;
                Ok(DeviceShard {
                    device: layout.group.devices()[i].clone(),
                    data: TensorData::from_storage(local_shape, select(data.storage(), &indices))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new(layout, shards)
    }
    pub fn replicate(data: &TensorData, group: DeviceGroup) -> Result<Self> {
        Self::shard(data, group, None)
    }
    pub fn layout(&self) -> &ShardLayout {
        &self.layout
    }
    pub fn shards(&self) -> &[DeviceShard] {
        &self.shards
    }
    pub fn gather(&self) -> Result<TensorData> {
        self.layout.validate()?;
        let global_len = self.layout.global_shape.numel()?;
        let mut per_global = vec![None; global_len];
        for (i, _shard) in self.shards.iter().enumerate() {
            for (local, global) in local_global_indices(&self.layout, i)?
                .into_iter()
                .enumerate()
            {
                per_global[global] = Some((i, local));
            }
            if matches!(self.layout.distribution, ShardDistribution::Replicated) {
                break;
            }
        }
        let indexes = per_global
            .into_iter()
            .enumerate()
            .map(|(global, location)| {
                location.ok_or_else(|| shard_error(format!("missing global element {global}")))
            })
            .collect::<Result<Vec<_>>>()?;
        TensorData::from_storage(
            self.layout.global_shape.clone(),
            gather_storage(&self.shards, &indexes),
        )
    }
    pub fn redistribute(&self, group: DeviceGroup, axis: Option<isize>) -> Result<Self> {
        Self::shard(&self.gather()?, group, axis)
    }
}

/// A movement operation that can be reasoned about without a graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LayoutTransform {
    Reshape(Shape),
    Permute(Vec<usize>),
    Expand(Shape),
    Shrink(Vec<(usize, usize)>),
    Stride(Vec<usize>),
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MovementDecision {
    Local(ShardLayout),
    NeedsRedistribution { reason: String },
}
impl ShardLayout {
    pub fn movement(&self, transform: LayoutTransform) -> Result<MovementDecision> {
        self.validate()?;
        let ShardDistribution::Axis { axis, .. } = &self.distribution else {
            return Ok(MovementDecision::Local(self.clone()));
        };
        match transform {
            LayoutTransform::Permute(axes)
                if valid_permutation(&axes, self.global_shape.rank()) =>
            {
                let next = axes.iter().position(|a| a == axis).unwrap();
                Ok(MovementDecision::Local(ShardLayout::axis_sharded(
                    self.group.clone(),
                    permuted_shape(&self.global_shape, &axes),
                    self.dtype,
                    next as isize,
                )?))
            }
            LayoutTransform::Reshape(shape) => {
                let before: usize = self.global_shape.dims()[..*axis].iter().product();
                let mut running = 1usize;
                let mut candidate = None;
                for (i, dim) in shape.dims().iter().enumerate() {
                    if running == before {
                        candidate = Some(i);
                    }
                    running = running
                        .checked_mul(*dim)
                        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                }
                if running == before {
                    candidate = Some(shape.rank());
                }
                if let Some(next) = candidate.filter(|next| {
                    *next < shape.rank() && shape.dims()[*next] % self.group.len() == 0
                }) {
                    Ok(MovementDecision::Local(ShardLayout::axis_sharded(
                        self.group.clone(),
                        shape,
                        self.dtype,
                        next as isize,
                    )?))
                } else {
                    Ok(MovementDecision::NeedsRedistribution {
                        reason: "reshape moves elements across shard boundaries".into(),
                    })
                }
            }
            LayoutTransform::Expand(shape) => {
                if shape.rank() < self.global_shape.rank() {
                    return Ok(MovementDecision::NeedsRedistribution {
                        reason: "expand lowers rank".into(),
                    });
                }
                let shift = shape.rank() - self.global_shape.rank();
                if shape.dims()[shift + *axis] != self.global_shape.dims()[*axis] {
                    return Ok(MovementDecision::NeedsRedistribution {
                        reason: "expand broadcasts shard axis".into(),
                    });
                }
                Ok(MovementDecision::Local(ShardLayout::axis_sharded(
                    self.group.clone(),
                    shape,
                    self.dtype,
                    (*axis + shift) as isize,
                )?))
            }
            LayoutTransform::Shrink(bounds)
                if bounds.len() == self.global_shape.rank()
                    && bounds[*axis].0 == 0
                    && bounds[*axis].1 == self.global_shape.dims()[*axis] =>
            {
                Ok(MovementDecision::Local(self.clone()))
            }
            LayoutTransform::Stride(steps)
                if steps.len() == self.global_shape.rank() && steps[*axis] == 1 =>
            {
                Ok(MovementDecision::Local(self.clone()))
            }
            _ => Ok(MovementDecision::NeedsRedistribution {
                reason: "transform changes provable shard ownership".into(),
            }),
        }
    }
}

/// Backend-neutral lowering decisions; actions are intentionally not executed here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShardExecutionPlan {
    Local {
        layout: ShardLayout,
        op: &'static str,
    },
    PeerRedistribute {
        from: ShardLayout,
        to: ShardLayout,
    },
    Gather {
        layout: ShardLayout,
    },
    SumAllReduce {
        layout: ShardLayout,
        axis: usize,
    },
    NeedsRedistribution {
        reason: String,
    },
}
impl ShardExecutionPlan {
    pub fn elementwise(layout: &ShardLayout) -> Self {
        Self::Local {
            layout: layout.clone(),
            op: "elementwise",
        }
    }
    pub fn binary(lhs: &ShardLayout, rhs: &ShardLayout) -> Self {
        if lhs == rhs {
            Self::elementwise(lhs)
        } else {
            Self::PeerRedistribute {
                from: rhs.clone(),
                to: lhs.clone(),
            }
        }
    }
    pub fn reduce_sum(layout: &ShardLayout, axis: usize) -> Self {
        match layout.distribution() {
            ShardDistribution::Axis {
                axis: shard_axis, ..
            } if axis == *shard_axis => Self::SumAllReduce {
                layout: layout.clone(),
                axis,
            },
            _ => Self::Local {
                layout: layout.clone(),
                op: "reduce",
            },
        }
    }
    pub fn matmul(
        lhs: &ShardLayout,
        rhs: &ShardLayout,
        lhs_contracting_axis: usize,
        rhs_contracting_axis: usize,
    ) -> Self {
        match (lhs.distribution(), rhs.distribution()) {
            (ShardDistribution::Axis { axis, .. }, _) if *axis == lhs_contracting_axis => {
                Self::SumAllReduce {
                    layout: lhs.clone(),
                    axis: *axis,
                }
            }
            (_, ShardDistribution::Axis { axis, .. }) if *axis == rhs_contracting_axis => {
                Self::SumAllReduce {
                    layout: rhs.clone(),
                    axis: *axis,
                }
            }
            _ => Self::Local {
                layout: lhs.clone(),
                op: "matmul",
            },
        }
    }
}

fn normalize_axis(axis: isize, rank: usize) -> Result<usize> {
    let axis = if axis < 0 {
        rank.checked_add_signed(axis)
            .ok_or_else(|| shard_error("axis is outside tensor rank"))?
    } else {
        axis as usize
    };
    if axis >= rank {
        Err(shard_error("axis is outside tensor rank"))
    } else {
        Ok(axis)
    }
}
fn shard_error(reason: impl Into<String>) -> Error {
    Error::Collective {
        reason: reason.into(),
    }
}
fn valid_permutation(axes: &[usize], rank: usize) -> bool {
    axes.len() == rank && {
        let mut seen = vec![false; rank];
        axes.iter()
            .all(|&a| a < rank && !std::mem::replace(&mut seen[a], true))
    }
}
fn permuted_shape(shape: &Shape, axes: &[usize]) -> Shape {
    Shape::from(axes.iter().map(|&a| shape.dims()[a]).collect::<Vec<_>>())
}
pub(crate) fn local_global_indices(layout: &ShardLayout, device: usize) -> Result<Vec<usize>> {
    layout.validate()?;
    let local = layout.local_shape(device)?;
    let len = local.numel()?;
    match layout.distribution() {
        ShardDistribution::Replicated => Ok((0..len).collect()),
        ShardDistribution::Axis { axis, ranges } => {
            let global_strides = strides(layout.global_shape())?;
            let local_strides = strides(&local)?;
            Ok((0..len)
                .map(|linear| {
                    (0..local.rank()).fold(0, |global, dimension| {
                        let mut coordinate =
                            (linear / local_strides[dimension]) % local.dims()[dimension];
                        if dimension == *axis {
                            coordinate += ranges[device].start;
                        }
                        global + coordinate * global_strides[dimension]
                    })
                })
                .collect())
        }
    }
}
fn canonical_layout_key(
    group: &DeviceGroup,
    global_shape: &Shape,
    dtype: DType,
    distribution: &ShardDistribution,
) -> String {
    format!(
        "shard:v1:{}:{dtype:?}:{:?}:{distribution:?}",
        group
            .devices()
            .iter()
            .map(DeviceId::as_str)
            .collect::<Vec<_>>()
            .join(","),
        global_shape.dims()
    )
}
fn strides(shape: &Shape) -> Result<Vec<usize>> {
    let mut out = vec![1usize; shape.rank()];
    for i in (1..shape.rank()).rev() {
        out[i - 1] = out[i]
            .checked_mul(shape.dims()[i])
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    }
    Ok(out)
}
fn select(storage: &Storage, idx: &[usize]) -> Storage {
    macro_rules! s {
        ($v:expr,$n:ident) => {
            Storage::$n(idx.iter().map(|&i| $v[i]).collect())
        };
    }
    match storage {
        Storage::Bool(v) => s!(v, Bool),
        Storage::I8(v) => s!(v, I8),
        Storage::U8(v) => s!(v, U8),
        Storage::Float8(v) => Storage::Float8(crate::Float8Storage::from_raw(
            v.format(),
            idx.iter().map(|&i| v.as_raw()[i]).collect(),
        )),
        Storage::I16(v) => s!(v, I16),
        Storage::U16(v) => s!(v, U16),
        Storage::I32(v) => s!(v, I32),
        Storage::U32(v) => s!(v, U32),
        Storage::I64(v) => s!(v, I64),
        Storage::U64(v) => s!(v, U64),
        Storage::F16(v) => s!(v, F16),
        Storage::BF16(v) => s!(v, BF16),
        Storage::F32(v) => s!(v, F32),
        Storage::F64(v) => s!(v, F64),
    }
}
fn gather_storage(shards: &[DeviceShard], idx: &[(usize, usize)]) -> Storage {
    macro_rules! g {
        ($n:ident) => {
            Storage::$n(
                idx.iter()
                    .map(|&(s, i)| match shards[s].data.storage() {
                        Storage::$n(v) => v[i],
                        _ => unreachable!(),
                    })
                    .collect(),
            )
        };
    }
    match shards[0].data.storage() {
        Storage::Float8(first) => Storage::Float8(crate::Float8Storage::from_raw(
            first.format(),
            idx.iter()
                .map(|&(s, i)| match shards[s].data.storage() {
                    Storage::Float8(values) if values.format() == first.format() => {
                        values.as_raw()[i]
                    }
                    _ => unreachable!(),
                })
                .collect(),
        )),
        Storage::Bool(_) => g!(Bool),
        Storage::I8(_) => g!(I8),
        Storage::U8(_) => g!(U8),
        Storage::I16(_) => g!(I16),
        Storage::U16(_) => g!(U16),
        Storage::I32(_) => g!(I32),
        Storage::U32(_) => g!(U32),
        Storage::I64(_) => g!(I64),
        Storage::U64(_) => g!(U64),
        Storage::F16(_) => g!(F16),
        Storage::BF16(_) => g!(BF16),
        Storage::F32(_) => g!(F32),
        Storage::F64(_) => g!(F64),
    }
}
fn same_storage(a: &Storage, b: &Storage) -> bool {
    macro_rules! e { ($n:ident) => { matches!((a,b), (Storage::$n(x), Storage::$n(y)) if x == y) } }
    e!(Bool)
        || e!(I8)
        || e!(U8)
        || e!(I16)
        || e!(U16)
        || e!(I32)
        || e!(U32)
        || e!(I64)
        || e!(U64)
        || e!(F16)
        || e!(BF16)
        || matches!((a, b), (Storage::F32(x), Storage::F32(y)) if x.iter().map(|v| v.to_bits()).eq(y.iter().map(|v| v.to_bits())))
        || matches!((a, b), (Storage::F64(x), Storage::F64(y)) if x.iter().map(|v| v.to_bits()).eq(y.iter().map(|v| v.to_bits())))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn group(n: usize) -> DeviceGroup {
        DeviceGroup::new((0..n).map(|i| DeviceId::new(format!("CPU:{i}")).unwrap())).unwrap()
    }
    fn ints(shape: impl Into<Shape>) -> TensorData {
        let shape = shape.into();
        TensorData::from_storage(
            shape.clone(),
            Storage::I32((0..shape.numel().unwrap() as i32).collect()),
        )
        .unwrap()
    }
    #[test]
    fn exact_round_trips_and_tinygrad_even_policy() {
        for n in 1..=4 {
            let dense = ints([8, 4]);
            for axis in [0, 1] {
                let t =
                    ShardedTensorData::shard(&dense, group(n), Some(axis)).unwrap_or_else(|_| {
                        assert_ne!(8usize % n, 0);
                        ShardedTensorData::replicate(&dense, group(n)).unwrap()
                    });
                if 8 % n == 0 || axis == 1 && 4 % n == 0 {
                    assert_eq!(t.gather().unwrap(), dense);
                }
            }
        }
        assert!(ShardedTensorData::shard(&ints([5]), group(2), Some(0)).is_err());
    }
    #[test]
    fn raw_bits_redistribute_and_edge_shapes() {
        let raw = TensorData::from_storage(
            [4],
            Storage::F32(vec![f32::from_bits(0x7fc01234), -0.0, 1.0, 2.0]),
        )
        .unwrap();
        let t = ShardedTensorData::shard(&raw, group(2), Some(-1))
            .unwrap()
            .redistribute(group(2), None)
            .unwrap();
        assert!(same_storage(raw.storage(), t.gather().unwrap().storage()));
        assert_eq!(
            ShardedTensorData::shard(&TensorData::scalar(1.0), group(3), Some(0))
                .unwrap()
                .layout()
                .distribution(),
            &ShardDistribution::Replicated
        );
        assert!(ShardedTensorData::shard(&ints([1]), group(2), Some(0)).is_err());
        assert_eq!(
            ShardedTensorData::shard(&ints([0, 4]), group(4), Some(0))
                .unwrap()
                .gather()
                .unwrap()
                .len(),
            0
        );
        let matrix = ints([4, 4]);
        assert_eq!(
            ShardedTensorData::shard(&matrix, group(2), Some(0))
                .unwrap()
                .redistribute(group(2), Some(1))
                .unwrap()
                .gather()
                .unwrap(),
            matrix
        );
    }
    #[test]
    fn shard_assembly_revalidates_deserialized_layout_identity_before_consuming_shards() {
        let layout = ShardLayout::axis_sharded(group(2), [4], DType::I32, 0).unwrap();
        let layout: ShardLayout =
            serde_json::from_value(serde_json::to_value(layout).unwrap()).unwrap();
        assert!(layout.validate().is_ok());

        let mut reversed_range = layout.clone();
        let ShardDistribution::Axis { ranges, .. } = &mut reversed_range.distribution else {
            unreachable!();
        };
        ranges[0] = ShardRange { start: 2, end: 0 };
        assert!(reversed_range.validate().is_err());
        assert!(
            ShardedTensorData::new(reversed_range, vec![]).is_err(),
            "malformed ranges fail before shard inventory indexing"
        );

        let mut stale_identity = layout;
        stale_identity.cache_key.push_str(":tampered");
        assert!(stale_identity.validate().is_err());
        assert!(
            ShardedTensorData::new(stale_identity, vec![]).is_err(),
            "noncanonical layouts fail before shard inventory consumption"
        );
    }
    #[test]
    fn movement_and_collectives_are_inspectable() {
        let l = ShardLayout::axis_sharded(group(2), [4, 6], DType::I32, 0).unwrap();
        assert!(matches!(
            l.movement(LayoutTransform::Permute(vec![1, 0])).unwrap(),
            MovementDecision::Local(_)
        ));
        assert!(matches!(
            l.movement(LayoutTransform::Reshape(Shape::from([3, 8])))
                .unwrap(),
            MovementDecision::NeedsRedistribution { .. }
        ));
        assert!(matches!(
            ShardExecutionPlan::reduce_sum(&l, 0),
            ShardExecutionPlan::SumAllReduce { .. }
        ));
        assert!(matches!(
            ShardExecutionPlan::binary(&l, &l),
            ShardExecutionPlan::Local { .. }
        ));
        assert_eq!(
            l.cache_key(),
            ShardLayout::axis_sharded(group(2), [4, 6], DType::I32, 0)
                .unwrap()
                .cache_key()
        );
    }
}
