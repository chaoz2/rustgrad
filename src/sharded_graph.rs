//! Graph composition for the static multi-device sharding boundary.
//!
//! There is no device scheduler here: every local node is an ordinary node in one
//! [`Graph`].  This makes the lowering inspectable and keeps CPU/autograd semantics exact.

use crate::collective::{DeviceGroup, DeviceId};
use crate::sharding::{LayoutTransform, MovementDecision, ShardDistribution, ShardLayout};
use crate::{BinaryOp, DType, Error, Graph, NodeId, ReduceKind, Result, Shape, Slice, UnaryOp};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardGraphTraceStep {
    pub action: &'static str,
    pub nodes: Vec<NodeId>,
    pub layout_key: String,
    pub collective_key: Option<String>,
    /// Immutable boundary record for a graph-visible collective. The legacy
    /// duplicated `nodes` vector remains source-compatible, but this record
    /// retains the single replicated result and its ordered rank producers.
    pub collective: Option<CollectiveBoundaryProvenance>,
    /// Ordered local partials consumed by a terminal collective. The ordinary
    /// graph result remains the exact CPU reference value; this provenance is
    /// the separately typed CUDA execution ABI and never changes CPU/autograd
    /// composition.
    pub collective_inputs: Vec<NodeId>,
    pub routes: Vec<RedistributionRoute>,
    /// Ordered rank-local operand identities for a local graph operation.
    pub local_inputs: Vec<LocalInputProvenance>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectiveBoundaryProvenance {
    /// Stable semantic key for this collective boundary. It is not inferred
    /// from an action label by CUDA planning.
    pub boundary_key: String,
    /// Rank-ordered local partial producers in immutable `DeviceGroup` order.
    pub ordered_inputs: Vec<NodeId>,
    /// The one graph node holding the replicated result; compatibility output
    /// nodes may repeat it once per rank.
    pub replicated_result: NodeId,
    /// Explicit lifetime mode. The trace can preserve one checked local
    /// consumer, but the CUDA planner remains fail-closed until a later
    /// execution vertical materializes it.
    pub lifecycle: CollectiveBoundaryLifecycle,
}
/// Canonical trace-level collective result lifetime. `Downstream` is an
/// immutable declaration, not permission to execute a local CUDA stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectiveBoundaryLifecycle {
    Terminal,
    Downstream {
        first_consumer_step: usize,
        lifetime_end_step: usize,
        ordered_consumers: Vec<NodeId>,
    },
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalInputProvenance {
    pub rank: usize,
    pub consumer_local_node: NodeId,
    pub ordered_inputs: Vec<LocalOperandProvenance>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalOperandProvenance {
    pub input_node: NodeId,
    /// Filled by CUDA schedule attachment; graph composition never invents it.
    pub canonical_schedule_buffer: Option<u64>,
    pub producer_redistribution_destination: Option<NodeId>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RedistributionRoute {
    pub source_rank: usize,
    pub source_device: DeviceId,
    pub source_node: NodeId,
    pub source_offset: usize,
    pub destination_rank: usize,
    pub destination_device: DeviceId,
    pub destination_node: NodeId,
    pub destination_offset: usize,
    pub elements: usize,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShardGraphTrace {
    pub steps: Vec<ShardGraphTraceStep>,
}
impl ShardGraphTrace {
    /// Deterministic typed-boundary identity. Legacy traces retain `None`, so
    /// their released candidate-free planner/cache identity remains unchanged.
    pub fn collective_identity(&self) -> Option<String> {
        let boundaries = self
            .steps
            .iter()
            .filter_map(|step| step.collective.as_ref().map(|boundary| (step, boundary)))
            .map(|(step, boundary)| {
                format!(
                    "{}:{}:{}:{}:{}",
                    boundary.boundary_key,
                    step.layout_key,
                    boundary.replicated_result.index(),
                    boundary
                        .ordered_inputs
                        .iter()
                        .map(|node| node.index().to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                    match &boundary.lifecycle {
                        CollectiveBoundaryLifecycle::Terminal => "terminal".into(),
                        CollectiveBoundaryLifecycle::Downstream {
                            first_consumer_step,
                            lifetime_end_step,
                            ordered_consumers,
                        } => format!(
                            "downstream:{first_consumer_step}:{lifetime_end_step}:{}",
                            ordered_consumers
                                .iter()
                                .map(|node| node.index().to_string())
                                .collect::<Vec<_>>()
                                .join(",")
                        ),
                    }
                )
            })
            .collect::<Vec<_>>();
        (!boundaries.is_empty()).then(|| boundaries.join("|"))
    }

    /// Native-only validation. CPU/autograd graph composition is unchanged;
    /// unsupported downstream collective consumers fail before CUDA planning.
    pub fn validate_collective_provenance(
        &self,
        group: &DeviceGroup,
        output_nodes: &[NodeId],
    ) -> Result<()> {
        let mut keys = Vec::<&str>::new();
        let mut typed_boundaries = 0usize;
        for (index, step) in self.steps.iter().enumerate() {
            let collective_action = step.action.contains("all-reduce");
            match (&step.collective, &step.collective_key) {
                (Some(boundary), Some(key)) => {
                    typed_boundaries += 1;
                    if key != &boundary.boundary_key || boundary.boundary_key.is_empty() {
                        return Err(shard_error("collective boundary key is invalid"));
                    }
                    if keys.contains(&boundary.boundary_key.as_str()) {
                        return Err(shard_error("collective boundary key is duplicated"));
                    }
                    keys.push(boundary.boundary_key.as_str());
                    if !collective_action {
                        return Err(shard_error("typed collective boundary action is invalid"));
                    }
                    if boundary.ordered_inputs.is_empty()
                        || boundary.ordered_inputs.len() != group.len()
                        || boundary.ordered_inputs != step.collective_inputs
                    {
                        return Err(shard_error(
                            "collective producer rank provenance is invalid",
                        ));
                    }
                    if boundary
                        .ordered_inputs
                        .iter()
                        .enumerate()
                        .any(|(rank, node)| {
                            boundary.ordered_inputs[..rank].contains(node)
                                || *node == boundary.replicated_result
                        })
                    {
                        return Err(shard_error(
                            "collective producer provenance is cyclic or duplicated",
                        ));
                    }
                    if step.nodes.len() != group.len()
                        || step
                            .nodes
                            .iter()
                            .any(|node| *node != boundary.replicated_result)
                    {
                        return Err(shard_error(
                            "collective replicated result ownership is invalid",
                        ));
                    }
                    if !self.steps[..index].iter().any(|prior| {
                        boundary
                            .ordered_inputs
                            .iter()
                            .all(|node| prior.nodes.contains(node))
                    }) {
                        return Err(shard_error("collective producer provenance is unreachable"));
                    }
                    match &boundary.lifecycle {
                        CollectiveBoundaryLifecycle::Terminal => {
                            if index + 1 != self.steps.len()
                                || output_nodes.len() != group.len()
                                || output_nodes
                                    .iter()
                                    .any(|node| *node != boundary.replicated_result)
                            {
                                return Err(shard_error(
                                    "typed collective boundary is not terminal",
                                ));
                            }
                        }
                        CollectiveBoundaryLifecycle::Downstream {
                            first_consumer_step,
                            lifetime_end_step,
                            ordered_consumers,
                        } => {
                            if *first_consumer_step <= index
                                || *first_consumer_step >= self.steps.len()
                                || *lifetime_end_step < *first_consumer_step
                                || *lifetime_end_step >= self.steps.len()
                                || ordered_consumers.len() != group.len()
                                || ordered_consumers
                                    .windows(2)
                                    .any(|nodes| nodes[0] == nodes[1])
                            {
                                return Err(shard_error(
                                    "collective downstream lifetime is invalid",
                                ));
                            }
                            let consumer =
                                self.steps.get(*first_consumer_step).ok_or_else(|| {
                                    shard_error("collective downstream consumer is absent")
                                })?;
                            if !consumer.action.starts_with("local-")
                                || consumer.nodes != *ordered_consumers
                                || output_nodes
                                    != self
                                        .steps
                                        .last()
                                        .map(|step| step.nodes.as_slice())
                                        .unwrap_or_default()
                                || consumer.local_inputs.len() != group.len()
                                || consumer
                                    .local_inputs
                                    .iter()
                                    .enumerate()
                                    .any(|(rank, input)| {
                                        input.rank != rank
                                            || input.consumer_local_node != ordered_consumers[rank]
                                            || !input.ordered_inputs.iter().any(|operand| {
                                                operand.input_node == boundary.replicated_result
                                            })
                                    })
                            {
                                return Err(shard_error(
                                    "collective downstream consumer provenance is invalid",
                                ));
                            }
                        }
                    }
                }
                (Some(_), None) => {
                    return Err(shard_error("typed collective boundary lacks canonical key"));
                }
                (None, Some(_)) if !collective_action || !step.collective_inputs.is_empty() => {
                    return Err(shard_error("legacy collective metadata is ambiguous"));
                }
                (None, Some(_)) | (None, None) if step.collective_inputs.is_empty() => {}
                (None, Some(_)) => {
                    return Err(shard_error("legacy collective metadata is ambiguous"));
                }
                (None, None) => {
                    return Err(shard_error("collective inputs lack boundary metadata"));
                }
            }
        }
        if typed_boundaries > 1 {
            return Err(shard_error(
                "multiple collective boundaries are not yet supported",
            ));
        }
        Ok(())
    }
}

/// A single-graph collection of local nodes with one immutable global layout.
#[derive(Clone, Debug)]
pub struct ShardedGraphTensor {
    graph_id: u64,
    layout: ShardLayout,
    nodes: Vec<NodeId>,
    requires_grad: bool,
    trace: ShardGraphTrace,
}
impl ShardedGraphTensor {
    pub const fn graph_id(&self) -> u64 {
        self.graph_id
    }
    pub fn layout(&self) -> &ShardLayout {
        &self.layout
    }
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }
    pub fn global_shape(&self) -> &Shape {
        self.layout.global_shape()
    }
    pub const fn dtype(&self) -> DType {
        self.layout.dtype()
    }
    pub const fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    pub fn trace(&self) -> &ShardGraphTrace {
        &self.trace
    }

    fn checked(&self, graph: &Graph) -> Result<()> {
        if self.graph_id != graph.id() {
            return Err(shard_error("sharded tensor belongs to another graph"));
        }
        if self.nodes.len() != self.layout.group().len() {
            return Err(shard_error("local node count does not match layout"));
        }
        for (i, node) in self.nodes.iter().enumerate() {
            if graph.shape(*node)? != &self.layout.local_shape(i)?
                || graph.dtype(*node)? != self.dtype()
            {
                return Err(shard_error(
                    "local node shape or dtype does not match layout",
                ));
            }
        }
        Ok(())
    }
    fn make(
        graph: &Graph,
        layout: ShardLayout,
        nodes: Vec<NodeId>,
        mut trace: ShardGraphTrace,
        action: &'static str,
        collective_key: Option<String>,
    ) -> Result<Self> {
        let requires_grad = nodes.iter().try_fold(false, |tracked, node| {
            Ok::<_, Error>(tracked || graph.requires_grad(*node)?)
        })?;
        trace.steps.push(ShardGraphTraceStep {
            action,
            nodes: nodes.clone(),
            layout_key: layout.cache_key().to_owned(),
            collective_key,
            collective: None,
            collective_inputs: vec![],
            routes: vec![],
            local_inputs: vec![],
        });
        let value = Self {
            graph_id: graph.id(),
            layout,
            nodes,
            requires_grad,
            trace,
        };
        value.checked(graph)?;
        Ok(value)
    }
}

impl Graph {
    /// Lowers one dense node to local checked shrink views. `None` replicates it.
    pub fn shard_node(
        &mut self,
        node: NodeId,
        group: DeviceGroup,
        axis: Option<isize>,
    ) -> Result<ShardedGraphTensor> {
        let shape = self.shape(node)?.clone();
        let dtype = self.dtype(node)?;
        let layout = match axis {
            Some(axis) => ShardLayout::axis_sharded(group, shape.clone(), dtype, axis)?,
            None => ShardLayout::replicated(group, shape.clone(), dtype)?,
        };
        let nodes = match layout.distribution() {
            ShardDistribution::Replicated => vec![node; layout.group().len()],
            ShardDistribution::Axis { axis, ranges } => ranges
                .iter()
                .map(|range| {
                    let bounds = shape
                        .dims()
                        .iter()
                        .enumerate()
                        .map(|(i, &dim)| {
                            if i == *axis {
                                (range.start, range.end)
                            } else {
                                (0, dim)
                            }
                        })
                        .collect::<Vec<_>>();
                    self.shrink(node, bounds)
                })
                .collect::<Result<Vec<_>>>()?,
        };
        ShardedGraphTensor::make(
            self,
            layout,
            nodes,
            ShardGraphTrace::default(),
            "shard",
            None,
        )
    }
    pub fn replicate_node(
        &mut self,
        node: NodeId,
        group: DeviceGroup,
    ) -> Result<ShardedGraphTensor> {
        self.shard_node(node, group, None)
    }
    pub fn gather_sharded(&mut self, value: &ShardedGraphTensor) -> Result<NodeId> {
        value.checked(self)?;
        match value.layout.distribution() {
            ShardDistribution::Replicated => Ok(value.nodes[0]),
            // A one-member axis layout still has a sharding layout (and thus
            // meaningful provenance), but there is no concat operation to lower.
            ShardDistribution::Axis { .. } if value.nodes.len() == 1 => Ok(value.nodes[0]),
            ShardDistribution::Axis { axis, .. } => self.concat(value.nodes.clone(), *axis),
        }
    }
    pub fn redistribute_sharded(
        &mut self,
        value: &ShardedGraphTensor,
        group: DeviceGroup,
        axis: Option<isize>,
    ) -> Result<ShardedGraphTensor> {
        value.checked(self)?;
        let dense = self.gather_sharded(value)?;
        let mut next = self.shard_node(dense, group, axis)?;
        next.trace = value.trace.clone();
        next.trace.steps.push(ShardGraphTraceStep {
            action: "redistribute",
            nodes: next.nodes.clone(),
            layout_key: next.layout.cache_key().to_owned(),
            collective_key: None,
            collective: None,
            collective_inputs: vec![],
            routes: redistribution_routes(value, &next)?,
            local_inputs: vec![],
        });
        Ok(next)
    }
    pub fn sharded_unary(
        &mut self,
        value: &ShardedGraphTensor,
        op: UnaryOp,
    ) -> Result<ShardedGraphTensor> {
        value.checked(self)?;
        let nodes = value
            .nodes
            .iter()
            .map(|n| self.unary(op, *n))
            .collect::<Result<Vec<_>>>()?;
        let mut output = ShardedGraphTensor::make(
            self,
            value.layout.clone(),
            nodes,
            value.trace.clone(),
            "local-unary",
            None,
        )?;
        attach_local_inputs(&mut output, std::slice::from_ref(value));
        Ok(output)
    }
    /// Applies one checked graph cast independently to every rank while preserving
    /// the static ownership layout. The resulting layout carries the new dtype.
    pub fn sharded_cast(
        &mut self,
        value: &ShardedGraphTensor,
        dtype: DType,
    ) -> Result<ShardedGraphTensor> {
        value.checked(self)?;
        let nodes = value
            .nodes
            .iter()
            .map(|node| self.cast(*node, dtype))
            .collect::<Result<Vec<_>>>()?;
        let mut output = ShardedGraphTensor::make(
            self,
            layout_with_dtype(&value.layout, dtype)?,
            nodes,
            value.trace.clone(),
            "local-cast",
            None,
        )?;
        attach_local_inputs(&mut output, std::slice::from_ref(value));
        Ok(output)
    }
    pub fn sharded_binary(
        &mut self,
        lhs: &ShardedGraphTensor,
        rhs: &ShardedGraphTensor,
        op: BinaryOp,
    ) -> Result<ShardedGraphTensor> {
        lhs.checked(self)?;
        rhs.checked(self)?;
        if lhs.layout.group() != rhs.layout.group() {
            return Err(shard_error(
                "binary operands have different ordered device groups",
            ));
        }
        let target = if matches!(lhs.layout.distribution(), ShardDistribution::Axis { .. }) {
            lhs.layout.clone()
        } else if matches!(rhs.layout.distribution(), ShardDistribution::Axis { .. }) {
            rhs.layout.clone()
        } else {
            lhs.layout.clone()
        };
        let target_axis = match target.distribution() {
            ShardDistribution::Replicated => None,
            ShardDistribution::Axis { axis, .. } => Some(*axis as isize),
        };
        let both_replicated = matches!(lhs.layout.distribution(), ShardDistribution::Replicated)
            && matches!(rhs.layout.distribution(), ShardDistribution::Replicated);
        let left_redistributed = lhs.layout != target && !both_replicated;
        let right_redistributed = rhs.layout != target && !both_replicated;
        let left = if !left_redistributed {
            lhs.clone()
        } else {
            self.redistribute_sharded(lhs, target.group().clone(), target_axis)?
        };
        let right = if !right_redistributed {
            rhs.clone()
        } else {
            self.redistribute_sharded(rhs, target.group().clone(), target_axis)?
        };
        let nodes = left
            .nodes
            .iter()
            .zip(&right.nodes)
            .map(|(a, b)| self.binary(op, *a, *b))
            .collect::<Result<Vec<_>>>()?;
        let mut output = ShardedGraphTensor::make(
            self,
            target,
            nodes,
            merged_trace(&left.trace, &right.trace),
            "local-binary",
            None,
        )?;
        attach_local_inputs(&mut output, &[left, right]);
        Ok(output)
    }
    pub fn sharded_select(
        &mut self,
        condition: &ShardedGraphTensor,
        on_true: &ShardedGraphTensor,
        on_false: &ShardedGraphTensor,
    ) -> Result<ShardedGraphTensor> {
        condition.checked(self)?;
        on_true.checked(self)?;
        on_false.checked(self)?;
        if condition.layout.group() != on_true.layout.group()
            || condition.layout.group() != on_false.layout.group()
        {
            return Err(shard_error("select condition device group differs"));
        }
        let value_shape = on_true
            .global_shape()
            .broadcast_with(on_false.global_shape())?;
        let output_shape = condition.global_shape().broadcast_with(&value_shape)?;
        let output_dtype = on_true.dtype().promote(on_false.dtype());
        let target = select_layout(condition, on_true, on_false, output_shape, output_dtype)?;
        let condition = select_operand(self, condition, &target)?;
        let on_true = select_operand(self, on_true, &target)?;
        let on_false = select_operand(self, on_false, &target)?;
        let nodes = (0..target.group().len())
            .map(|i| self.select(condition.nodes[i], on_true.nodes[i], on_false.nodes[i]))
            .collect::<Result<Vec<_>>>()?;
        let mut output = ShardedGraphTensor::make(
            self,
            target,
            nodes,
            merged_trace(
                &merged_trace(&condition.trace, &on_true.trace),
                &on_false.trace,
            ),
            "local-select",
            None,
        )?;
        attach_local_inputs(&mut output, &[condition, on_true, on_false]);
        Ok(output)
    }
    pub fn sharded_reduce(
        &mut self,
        value: &ShardedGraphTensor,
        kind: ReduceKind,
        axis: usize,
    ) -> Result<ShardedGraphTensor> {
        value.checked(self)?;
        if axis >= value.global_shape().rank() {
            return Err(shard_error("reduction axis is outside sharded tensor rank"));
        }
        if matches!(kind, ReduceKind::Max | ReduceKind::Min)
            && matches!(value.layout.distribution(), ShardDistribution::Axis { axis: shard_axis, .. } if *shard_axis == axis)
        {
            return Err(shard_error(
                "max/min reduction over a shard axis is not yet a composable collective",
            ));
        }
        let local = value
            .nodes
            .iter()
            .map(|n| self.reduce(*n, kind, Some(vec![axis as isize]), false))
            .collect::<Result<Vec<_>>>()?;
        if matches!(value.layout.distribution(), ShardDistribution::Axis { axis: shard_axis, .. } if *shard_axis == axis)
        {
            let sum = local
                .iter()
                .copied()
                .reduce(|a, b| self.add(a, b).expect("validated local sum"))
                .ok_or_else(|| shard_error("empty device group"))?;
            let output_shape = self.shape(sum)?.clone();
            let layout = ShardLayout::replicated(
                value.layout.group().clone(),
                output_shape,
                self.dtype(sum)?,
            )?;
            let mut output = ShardedGraphTensor::make(
                self,
                layout,
                vec![sum; value.layout.group().len()],
                value.trace.clone(),
                "sum-all-reduce",
                Some(format!("sum-all-reduce:{}", value.layout.cache_key())),
            )?;
            attach_collective_inputs(&mut output, local);
            Ok(output)
        } else {
            let dims = self.shape(local[0])?.dims().to_vec();
            let layout = match value.layout.distribution() {
                ShardDistribution::Replicated => ShardLayout::replicated(
                    value.layout.group().clone(),
                    Shape::from(dims),
                    self.dtype(local[0])?,
                )?,
                ShardDistribution::Axis {
                    axis: shard_axis, ..
                } => {
                    let next = if axis < *shard_axis {
                        *shard_axis - 1
                    } else {
                        *shard_axis
                    };
                    ShardLayout::axis_sharded(
                        value.layout.group().clone(),
                        Shape::from(dims),
                        self.dtype(local[0])?,
                        next as isize,
                    )?
                }
            };
            ShardedGraphTensor::make(
                self,
                layout,
                local,
                value.trace.clone(),
                "local-reduce",
                None,
            )
        }
    }
    /// Mean uses one global divisor after the sum-all-reduce path.
    pub fn sharded_mean(
        &mut self,
        value: &ShardedGraphTensor,
        axis: usize,
    ) -> Result<ShardedGraphTensor> {
        let sum = self.sharded_reduce(value, ReduceKind::Sum, axis)?;
        let divisor = self.constant(crate::TensorData::scalar(
            value.global_shape().dims()[axis] as f32,
        ));
        let replicated = self.replicate_node(divisor, sum.layout.group().clone())?;
        self.sharded_binary(&sum, &replicated, BinaryOp::Div)
    }
    /// Rank-two matmul lowering. Contracting-axis shards form local partial products followed
    /// by a graph-visible sum all-reduce; row/column shards with a replicated peer stay local.
    /// Other layouts deliberately gather and re-replicate rather than guessing ownership.
    pub fn sharded_matmul(
        &mut self,
        lhs: &ShardedGraphTensor,
        rhs: &ShardedGraphTensor,
    ) -> Result<ShardedGraphTensor> {
        lhs.checked(self)?;
        rhs.checked(self)?;
        if lhs.layout.group() != rhs.layout.group() {
            return Err(shard_error(
                "matmul operands have different ordered device groups",
            ));
        }
        let rank_two = lhs.global_shape().rank() == 2 && rhs.global_shape().rank() == 2;
        let lhs_axis = match lhs.layout.distribution() {
            ShardDistribution::Axis { axis, .. } => Some(*axis),
            _ => None,
        };
        let rhs_axis = match rhs.layout.distribution() {
            ShardDistribution::Axis { axis, .. } => Some(*axis),
            _ => None,
        };
        let contracting = lhs_axis == Some(1) || rhs_axis == Some(0);
        if rank_two
            && contracting
            && (lhs_axis.is_none() || lhs_axis == Some(1))
            && (rhs_axis.is_none() || rhs_axis == Some(0))
        {
            let left = if lhs_axis.is_none() {
                self.redistribute_sharded(lhs, lhs.layout.group().clone(), Some(1))?
            } else {
                lhs.clone()
            };
            let right = if rhs_axis.is_none() {
                self.redistribute_sharded(rhs, rhs.layout.group().clone(), Some(0))?
            } else {
                rhs.clone()
            };
            let partials = left
                .nodes
                .iter()
                .zip(&right.nodes)
                .map(|(a, b)| self.matmul(*a, *b))
                .collect::<Result<Vec<_>>>()?;
            let total = partials
                .iter()
                .copied()
                .reduce(|a, b| self.add(a, b).expect("validated matmul partials"))
                .ok_or_else(|| shard_error("empty device group"))?;
            let layout = ShardLayout::replicated(
                lhs.layout.group().clone(),
                self.shape(total)?.clone(),
                self.dtype(total)?,
            )?;
            let mut output = ShardedGraphTensor::make(
                self,
                layout,
                vec![total; lhs.layout.group().len()],
                lhs.trace.clone(),
                "matmul-sum-all-reduce",
                Some(format!("sum-all-reduce:{}", lhs.layout.cache_key())),
            )?;
            attach_collective_inputs(&mut output, partials);
            return Ok(output);
        }
        if rank_two && lhs_axis == Some(0) && rhs_axis.is_none() {
            let nodes = lhs
                .nodes
                .iter()
                .zip(&rhs.nodes)
                .map(|(a, b)| self.matmul(*a, *b))
                .collect::<Result<Vec<_>>>()?;
            let layout = ShardLayout::axis_sharded(
                lhs.layout.group().clone(),
                Shape::from([lhs.global_shape().dims()[0], rhs.global_shape().dims()[1]]),
                self.dtype(nodes[0])?,
                0,
            )?;
            return ShardedGraphTensor::make(
                self,
                layout,
                nodes,
                lhs.trace.clone(),
                "local-matmul",
                None,
            );
        }
        if rank_two && lhs_axis.is_none() && rhs_axis == Some(1) {
            let nodes = lhs
                .nodes
                .iter()
                .zip(&rhs.nodes)
                .map(|(a, b)| self.matmul(*a, *b))
                .collect::<Result<Vec<_>>>()?;
            let layout = ShardLayout::axis_sharded(
                rhs.layout.group().clone(),
                Shape::from([lhs.global_shape().dims()[0], rhs.global_shape().dims()[1]]),
                self.dtype(nodes[0])?,
                1,
            )?;
            return ShardedGraphTensor::make(
                self,
                layout,
                nodes,
                rhs.trace.clone(),
                "local-matmul",
                None,
            );
        }
        let left = self.gather_sharded(lhs)?;
        let right = self.gather_sharded(rhs)?;
        let dense = self.matmul(left, right)?;
        let mut output = self.replicate_node(dense, lhs.layout.group().clone())?;
        output.trace = lhs.trace.clone();
        output.trace.steps.push(ShardGraphTraceStep {
            action: "gather-matmul",
            nodes: output.nodes.clone(),
            layout_key: output.layout.cache_key().to_owned(),
            collective_key: None,
            collective: None,
            collective_inputs: vec![],
            routes: vec![],
            local_inputs: vec![],
        });
        Ok(output)
    }
    pub fn sharded_movement(
        &mut self,
        value: &ShardedGraphTensor,
        transform: LayoutTransform,
    ) -> Result<ShardedGraphTensor> {
        value.checked(self)?;
        let decision = value.layout.movement(transform.clone())?;
        let (source, layout, action) = match decision {
            MovementDecision::Local(layout) => (value.clone(), layout, "local-movement"),
            MovementDecision::NeedsRedistribution { .. } => {
                let dense = self.gather_sharded(value)?;
                let replicated = self.replicate_node(dense, value.layout.group().clone())?;
                (replicated, value.layout.clone(), "gather-movement")
            }
        };
        let nodes = source
            .nodes
            .iter()
            .map(|n| match &transform {
                LayoutTransform::Reshape(s) => self.reshape(*n, s.clone()),
                LayoutTransform::Permute(a) => self.permute(*n, a.clone()),
                LayoutTransform::Expand(s) => self.expand(*n, s.clone()),
                LayoutTransform::Shrink(b) => self.shrink(*n, b.clone()),
                LayoutTransform::Stride(steps) => self.stride(
                    *n,
                    steps
                        .iter()
                        .map(|&step| Slice {
                            start: None,
                            stop: None,
                            step: step as isize,
                        })
                        .collect::<Vec<_>>(),
                ),
            })
            .collect::<Result<Vec<_>>>()?;
        let final_layout = if action == "local-movement" {
            layout
        } else {
            ShardLayout::replicated(
                value.layout.group().clone(),
                self.shape(nodes[0])?.clone(),
                self.dtype(nodes[0])?,
            )?
        };
        ShardedGraphTensor::make(self, final_layout, nodes, value.trace.clone(), action, None)
    }
}
fn redistribution_routes(
    source: &ShardedGraphTensor,
    destination: &ShardedGraphTensor,
) -> Result<Vec<RedistributionRoute>> {
    use std::collections::BTreeMap;
    let mut locations = BTreeMap::new();
    for rank in 0..source.nodes.len() {
        for (offset, global) in crate::sharding::local_global_indices(&source.layout, rank)?
            .into_iter()
            .enumerate()
        {
            locations.insert(global, (rank, offset));
        }
    }
    let mut out = Vec::new();
    // Empty layouts still need typed producer/destination identities for a
    // later executor to carry logical-zero bindings without inventing a route.
    if source.layout.global_shape().numel()? == 0 {
        for dst in 0..destination.nodes.len() {
            let src = dst % source.nodes.len();
            out.push(RedistributionRoute {
                source_rank: src,
                source_device: source.layout.group().devices()[src].clone(),
                source_node: source.nodes[src],
                source_offset: 0,
                destination_rank: dst,
                destination_device: destination.layout.group().devices()[dst].clone(),
                destination_node: destination.nodes[dst],
                destination_offset: 0,
                elements: 0,
            });
        }
        return Ok(out);
    }
    for dst in 0..destination.nodes.len() {
        let globals = crate::sharding::local_global_indices(&destination.layout, dst)?;
        let mut run: Option<(usize, usize, usize, usize)> = None;
        for (dst_offset, global) in globals.into_iter().enumerate() {
            let (src, src_offset) = *locations
                .get(&global)
                .ok_or_else(|| shard_error("redistribution source does not cover destination"))?;
            match run {
                Some((r, so, doff, len))
                    if r == src && so + len == src_offset && doff + len == dst_offset =>
                {
                    run = Some((r, so, doff, len + 1))
                }
                _ => {
                    if let Some((r, so, doff, len)) = run.take() {
                        out.push(RedistributionRoute {
                            source_rank: r,
                            source_device: source.layout.group().devices()[r].clone(),
                            source_node: source.nodes[r],
                            source_offset: so,
                            destination_rank: dst,
                            destination_device: destination.layout.group().devices()[dst].clone(),
                            destination_node: destination.nodes[dst],
                            destination_offset: doff,
                            elements: len,
                        });
                    }
                    run = Some((src, src_offset, dst_offset, 1));
                }
            }
        }
        if let Some((r, so, doff, len)) = run {
            out.push(RedistributionRoute {
                source_rank: r,
                source_device: source.layout.group().devices()[r].clone(),
                source_node: source.nodes[r],
                source_offset: so,
                destination_rank: dst,
                destination_device: destination.layout.group().devices()[dst].clone(),
                destination_node: destination.nodes[dst],
                destination_offset: doff,
                elements: len,
            });
        }
    }
    Ok(out)
}
/// A binary operation consumes both typed local graph branches. Keep their exact
/// route steps in deterministic left-then-right order so a later planner never
/// has to rediscover a redistribution from an action label or graph walk.
fn merged_trace(left: &ShardGraphTrace, right: &ShardGraphTrace) -> ShardGraphTrace {
    let mut steps = left.steps.clone();
    for step in &right.steps {
        if !steps.contains(step) {
            steps.push(step.clone());
        }
    }
    ShardGraphTrace { steps }
}
fn layout_with_dtype(layout: &ShardLayout, dtype: DType) -> Result<ShardLayout> {
    match layout.distribution() {
        ShardDistribution::Replicated => {
            ShardLayout::replicated(layout.group().clone(), layout.global_shape().clone(), dtype)
        }
        ShardDistribution::Axis { axis, .. } => ShardLayout::axis_sharded(
            layout.group().clone(),
            layout.global_shape().clone(),
            dtype,
            *axis as isize,
        ),
    }
}
fn select_layout(
    condition: &ShardedGraphTensor,
    on_true: &ShardedGraphTensor,
    on_false: &ShardedGraphTensor,
    shape: Shape,
    dtype: DType,
) -> Result<ShardLayout> {
    let mut axis = None;
    for value in [condition, on_true, on_false] {
        if let Some(candidate) = compatible_output_axis(value, &shape)
            && axis
                .replace(candidate)
                .is_some_and(|prior| prior != candidate)
        {
            return Err(shard_error(
                "select has incompatible broadcasted sharded axes",
            ));
        }
    }
    match axis {
        Some(axis) => ShardLayout::axis_sharded(
            condition.layout.group().clone(),
            shape,
            dtype,
            axis as isize,
        ),
        None => ShardLayout::replicated(condition.layout.group().clone(), shape, dtype),
    }
}
fn compatible_output_axis(value: &ShardedGraphTensor, output: &Shape) -> Option<usize> {
    let ShardDistribution::Axis { axis, .. } = value.layout.distribution() else {
        return None;
    };
    let rank_offset = output.rank().checked_sub(value.global_shape().rank())?;
    let output_axis = rank_offset.checked_add(*axis)?;
    (value.global_shape().dims()[*axis] == output.dims()[output_axis]).then_some(output_axis)
}
fn select_operand(
    graph: &mut Graph,
    value: &ShardedGraphTensor,
    target: &ShardLayout,
) -> Result<ShardedGraphTensor> {
    let desired = match target.distribution() {
        ShardDistribution::Replicated => None,
        ShardDistribution::Axis { axis, .. } => {
            let rank_offset = target
                .global_shape()
                .rank()
                .checked_sub(value.global_shape().rank())
                .ok_or_else(|| shard_error("select operand rank exceeds result rank"))?;
            let input_axis = axis.checked_sub(rank_offset).ok_or_else(|| {
                shard_error("select shard axis is absent from broadcasted operand")
            })?;
            let input_dim = value.global_shape().dims()[input_axis];
            let output_dim = target.global_shape().dims()[*axis];
            (input_dim == output_dim).then_some(input_axis)
        }
    };
    let desired_layout = match desired {
        Some(axis) => ShardLayout::axis_sharded(
            target.group().clone(),
            value.global_shape().clone(),
            value.dtype(),
            axis as isize,
        )?,
        None => ShardLayout::replicated(
            target.group().clone(),
            value.global_shape().clone(),
            value.dtype(),
        )?,
    };
    if value.layout == desired_layout {
        Ok(value.clone())
    } else {
        graph.redistribute_sharded(
            value,
            target.group().clone(),
            desired.map(|axis| axis as isize),
        )
    }
}
fn attach_local_inputs(output: &mut ShardedGraphTensor, inputs: &[ShardedGraphTensor]) {
    let step = output
        .trace
        .steps
        .last_mut()
        .expect("local operation trace step");
    step.local_inputs = output
        .nodes
        .iter()
        .enumerate()
        .map(|(rank, &consumer_local_node)| LocalInputProvenance {
            rank,
            consumer_local_node,
            ordered_inputs: inputs
                .iter()
                .map(|input| {
                    let input_node = input.nodes[rank];
                    LocalOperandProvenance {
                        input_node,
                        canonical_schedule_buffer: None,
                        producer_redistribution_destination: redistribution_destination(
                            input, rank, input_node,
                        ),
                    }
                })
                .collect(),
        })
        .collect();
    let consumer_step = output.trace.steps.len().saturating_sub(1);
    for input in inputs {
        if let Some((boundary_step, boundary)) = output
            .trace
            .steps
            .iter_mut()
            .enumerate()
            .rev()
            .find_map(|(index, step)| step.collective.as_mut().map(|boundary| (index, boundary)))
            && input.trace.steps.len() == consumer_step
        {
            match &mut boundary.lifecycle {
                CollectiveBoundaryLifecycle::Terminal if boundary_step + 1 == consumer_step => {
                    boundary.lifecycle = CollectiveBoundaryLifecycle::Downstream {
                        first_consumer_step: consumer_step,
                        lifetime_end_step: consumer_step,
                        ordered_consumers: output.nodes.clone(),
                    };
                    break;
                }
                CollectiveBoundaryLifecycle::Downstream {
                    lifetime_end_step, ..
                } if *lifetime_end_step < consumer_step => {
                    // A permitted local composition retains the typed
                    // collective result until its last observed consumer.
                    // Keep the first consumer immutable and only extend the
                    // lifetime; native execution remains fail-closed.
                    *lifetime_end_step = consumer_step;
                    break;
                }
                _ => {}
            }
        }
    }
}

fn attach_collective_inputs(output: &mut ShardedGraphTensor, inputs: Vec<NodeId>) {
    debug_assert_eq!(inputs.len(), output.layout.group().len());
    // Preserve the otherwise graph-internal rank-local producer boundary in the
    // immutable trace. This is metadata only: CPU/autograd still consume the
    // ordinary graph result, while native preflight can prove that the typed
    // collective producers are reachable without rediscovering the graph walk.
    let collective_index = output.trace.steps.len().saturating_sub(1);
    output.trace.steps.insert(
        collective_index,
        ShardGraphTraceStep {
            action: "collective-local-partials",
            nodes: inputs.clone(),
            layout_key: output.layout.cache_key().to_owned(),
            collective_key: None,
            collective: None,
            collective_inputs: vec![],
            routes: vec![],
            local_inputs: vec![],
        },
    );
    output
        .trace
        .steps
        .last_mut()
        .expect("collective trace step")
        .collective_inputs = inputs.clone();
    let step = output
        .trace
        .steps
        .last_mut()
        .expect("collective trace step");
    let boundary_key = step.collective_key.clone().expect("collective trace key");
    let replicated_result = *step.nodes.first().expect("collective result node");
    debug_assert!(step.nodes.iter().all(|node| *node == replicated_result));
    step.collective = Some(CollectiveBoundaryProvenance {
        boundary_key,
        ordered_inputs: inputs,
        replicated_result,
        lifecycle: CollectiveBoundaryLifecycle::Terminal,
    });
}
fn redistribution_destination(
    value: &ShardedGraphTensor,
    rank: usize,
    input_node: NodeId,
) -> Option<NodeId> {
    value.trace.steps.iter().rev().find_map(|step| {
        (step.action == "redistribute" && step.nodes.get(rank) == Some(&input_node))
            .then_some(input_node)
    })
}
fn shard_error(reason: impl Into<String>) -> Error {
    Error::Collective {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collective::DeviceId;
    use crate::{Backend, CpuBackend, Storage, TensorData};
    use std::collections::HashMap;
    fn group(n: usize) -> DeviceGroup {
        DeviceGroup::new((0..n).map(|i| DeviceId::new(format!("CPU:{i}")).unwrap())).unwrap()
    }
    fn data(shape: impl Into<Shape>, x: &[f32]) -> TensorData {
        TensorData::new(shape, x.to_vec()).unwrap()
    }
    #[test]
    fn graph_shard_roundtrip_and_axis_sum_grad() {
        let mut g = Graph::new();
        let x = g.input("x", [4, 2]);
        let s = g.shard_node(x, group(2), Some(0)).unwrap();
        let y = g.sharded_mean(&s, 0).unwrap();
        let dense = g.gather_sharded(&y).unwrap();
        let loss = g.sum_all(dense).unwrap();
        let dx = g.grad(loss, x).unwrap();
        let input = HashMap::from([(
            String::from("x"),
            data([4, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]),
        )]);
        let cpu = CpuBackend;
        assert_eq!(cpu.execute(&g, dense, &input).unwrap().values(), &[4., 5.]);
        assert_eq!(cpu.execute(&g, dx, &input).unwrap().values(), &[0.25; 8]);
    }
    #[test]
    fn terminal_sum_all_reduce_retains_ordered_cuda_partial_provenance() {
        let mut graph = Graph::new();
        let input = graph.input("x", [4]);
        let sharded = graph.shard_node(input, group(2), Some(0)).unwrap();
        let reduced = graph.sharded_reduce(&sharded, ReduceKind::Sum, 0).unwrap();
        let collective = reduced.trace().steps.last().unwrap();
        assert_eq!(collective.action, "sum-all-reduce");
        assert_eq!(collective.collective_inputs.len(), 2);
        assert_ne!(
            collective.collective_inputs[0],
            collective.collective_inputs[1]
        );
        assert!(reduced.nodes().windows(2).all(|pair| pair[0] == pair[1]));
        assert_eq!(
            collective.collective.as_ref().unwrap().ordered_inputs,
            collective.collective_inputs
        );
        assert!(reduced.trace().collective_identity().is_some());
        reduced
            .trace()
            .validate_collective_provenance(reduced.layout().group(), reduced.nodes())
            .unwrap();
        let output = graph.gather_sharded(&reduced).unwrap();
        let values = HashMap::from([("x".into(), data([4], &[1., 2., 3., 4.]))]);
        assert_eq!(
            CpuBackend
                .execute(&graph, output, &values)
                .unwrap()
                .values(),
            &[10.]
        );
    }
    #[test]
    fn typed_collective_provenance_preserves_one_checked_downstream_consumer() {
        let mut graph = Graph::new();
        let input = graph.input("x", [4]);
        let sharded = graph.shard_node(input, group(2), Some(0)).unwrap();
        let reduced = graph.sharded_reduce(&sharded, ReduceKind::Sum, 0).unwrap();
        let downstream = graph.sharded_unary(&reduced, UnaryOp::Neg).unwrap();
        downstream
            .trace()
            .validate_collective_provenance(downstream.layout().group(), downstream.nodes())
            .unwrap();
        let boundary = downstream
            .trace()
            .steps
            .iter()
            .find_map(|step| step.collective.as_ref())
            .unwrap();
        assert!(matches!(
            &boundary.lifecycle,
            CollectiveBoundaryLifecycle::Downstream {
                first_consumer_step: _,
                lifetime_end_step: _,
                ..
            }
        ));

        let mut malformed = reduced.clone();
        malformed.trace.steps.last_mut().unwrap().collective_key = None;
        assert!(
            malformed
                .trace()
                .validate_collective_provenance(malformed.layout().group(), malformed.nodes())
                .is_err()
        );
    }
    #[test]
    fn typed_collective_provenance_has_stable_identity_and_rejects_malformed_boundaries() {
        let mut graph = Graph::new();
        let input = graph.input("x", [4]);
        let sharded = graph.shard_node(input, group(2), Some(0)).unwrap();
        let reduced = graph.sharded_reduce(&sharded, ReduceKind::Sum, 0).unwrap();
        assert_eq!(
            reduced.trace().collective_identity(),
            reduced.trace().clone().collective_identity()
        );

        let collective_index = reduced.trace().steps.len() - 1;
        let mut duplicate_producer = reduced.clone();
        let boundary = duplicate_producer.trace.steps[collective_index]
            .collective
            .as_mut()
            .unwrap();
        boundary.ordered_inputs[1] = boundary.ordered_inputs[0];
        assert!(
            duplicate_producer
                .trace()
                .validate_collective_provenance(
                    duplicate_producer.layout().group(),
                    duplicate_producer.nodes()
                )
                .is_err()
        );

        let mut wrong_result = reduced.clone();
        wrong_result.trace.steps[collective_index]
            .collective
            .as_mut()
            .unwrap()
            .replicated_result = sharded.nodes()[0];
        assert!(
            wrong_result
                .trace()
                .validate_collective_provenance(wrong_result.layout().group(), wrong_result.nodes())
                .is_err()
        );

        let mut missing_provenance = reduced.clone();
        missing_provenance.trace.steps[collective_index]
            .collective
            .as_mut()
            .unwrap()
            .ordered_inputs
            .clear();
        assert!(
            missing_provenance
                .trace()
                .validate_collective_provenance(
                    missing_provenance.layout().group(),
                    missing_provenance.nodes()
                )
                .is_err()
        );
    }
    #[test]
    fn local_sharded_composition_preserves_collective_record_for_native_preflight_gate() {
        let mut graph = Graph::new();
        let input = graph.input("x", [4, 1]);
        let selector = graph.input_dtype("selector", [1, 1], DType::Bool);
        let sharded = graph.shard_node(input, group(2), Some(0)).unwrap();
        let replicated_selector = graph.replicate_node(selector, group(2)).unwrap();
        let reduced = graph.sharded_reduce(&sharded, ReduceKind::Sum, 0).unwrap();
        let cast = graph.sharded_cast(&reduced, DType::F64).unwrap();
        let unary = graph.sharded_unary(&cast, UnaryOp::Neg).unwrap();
        let binary = graph.sharded_binary(&unary, &unary, BinaryOp::Add).unwrap();
        let selected = graph
            .sharded_select(&replicated_selector, &binary, &binary)
            .unwrap();
        let moved = graph
            .sharded_movement(&selected, LayoutTransform::Permute(vec![1, 0]))
            .unwrap();
        for value in [&cast, &unary, &binary, &selected, &moved] {
            assert!(value.trace().collective_identity().is_some());
            value
                .trace()
                .validate_collective_provenance(value.layout().group(), value.nodes())
                .unwrap();
        }
    }
    #[test]
    fn local_binary_movement_and_trace() {
        let mut g = Graph::new();
        let x = g.input("x", [4, 4]);
        let a = g.shard_node(x, group(2), Some(0)).unwrap();
        let b = g.replicate_node(x, group(2)).unwrap();
        let c = g.sharded_binary(&a, &b, BinaryOp::Add).unwrap();
        let d = g
            .sharded_movement(&c, LayoutTransform::Permute(vec![1, 0]))
            .unwrap();
        let out = g.gather_sharded(&d).unwrap();
        let inp = HashMap::from([(
            String::from("x"),
            data([4, 4], &(0..16).map(|x| x as f32).collect::<Vec<_>>()),
        )]);
        assert_eq!(
            CpuBackend.execute(&g, out, &inp).unwrap().values(),
            &[
                0., 8., 16., 24., 2., 10., 18., 26., 4., 12., 20., 28., 6., 14., 22., 30.
            ]
        );
        assert!(d.trace().steps.len() >= 2);
    }
    #[test]
    fn local_binary_trace_preserves_ordered_redistribution_provenance() {
        let mut g = Graph::new();
        let left_input = g.input("left", [4, 2]);
        let right_input = g.input("right", [4, 2]);
        let left = g.shard_node(left_input, group(2), Some(0)).unwrap();
        let right = g.replicate_node(right_input, group(2)).unwrap();

        let output = g.sharded_binary(&left, &right, BinaryOp::Add).unwrap();
        let redistribution = output
            .trace()
            .steps
            .iter()
            .find(|step| step.action == "redistribute")
            .expect("replicated rhs is redistributed to axis shards");
        let local = output.trace().steps.last().unwrap();
        assert_eq!(local.action, "local-binary");
        assert_eq!(local.local_inputs.len(), 2);
        for (rank, provenance) in local.local_inputs.iter().enumerate() {
            assert_eq!(provenance.rank, rank);
            assert_eq!(provenance.consumer_local_node, output.nodes()[rank]);
            assert_eq!(provenance.ordered_inputs.len(), 2);
            assert_eq!(provenance.ordered_inputs[0].input_node, left.nodes()[rank]);
            assert_eq!(
                provenance.ordered_inputs[0].producer_redistribution_destination,
                None
            );
            assert_eq!(
                provenance.ordered_inputs[1].input_node,
                redistribution.nodes[rank]
            );
            assert_eq!(
                provenance.ordered_inputs[1].producer_redistribution_destination,
                Some(redistribution.nodes[rank])
            );
            assert_eq!(provenance.ordered_inputs[0].canonical_schedule_buffer, None);
            assert_eq!(provenance.ordered_inputs[1].canonical_schedule_buffer, None);
        }
    }
    #[test]
    fn local_unary_cast_and_broadcast_select_preserve_static_layouts() {
        let mut g = Graph::new();
        let x = g.input_dtype("x", [4, 2], DType::I32);
        let condition = g.input_dtype("condition", [4, 1], DType::Bool);
        let on_true = g.input_dtype("on_true", [4, 2], DType::I32);
        let on_false = g.input_dtype("on_false", [1, 2], DType::I32);
        let x_shards = g.shard_node(x, group(2), Some(0)).unwrap();
        let condition_shards = g.shard_node(condition, group(2), Some(0)).unwrap();
        let true_replicas = g.replicate_node(on_true, group(2)).unwrap();
        let false_replicas = g.replicate_node(on_false, group(2)).unwrap();

        let negated = g.sharded_unary(&x_shards, UnaryOp::Neg).unwrap();
        let cast = g.sharded_cast(&negated, DType::F32).unwrap();
        let selected = g
            .sharded_select(&condition_shards, &true_replicas, &false_replicas)
            .unwrap();
        assert_eq!(cast.dtype(), DType::F32);
        assert_eq!(
            cast.layout().distribution(),
            x_shards.layout().distribution()
        );
        assert_eq!(selected.global_shape(), &Shape::from([4, 2]));
        assert!(matches!(
            selected.layout().distribution(),
            ShardDistribution::Axis { axis: 0, .. }
        ));
        for rank in 0..2 {
            assert_eq!(
                g.shape(selected.nodes()[rank]).unwrap(),
                &Shape::from([2, 2])
            );
        }
        let gathered_cast = g.gather_sharded(&cast).unwrap();
        let gathered_select = g.gather_sharded(&selected).unwrap();
        let inputs = HashMap::from([
            (
                String::from("x"),
                TensorData::from_storage([4, 2], Storage::I32(vec![-2, -1, 0, 1, 2, 3, 4, 5]))
                    .unwrap(),
            ),
            (
                String::from("condition"),
                TensorData::from_storage([4, 1], Storage::Bool(vec![true, false, true, false]))
                    .unwrap(),
            ),
            (
                String::from("on_true"),
                TensorData::from_storage([4, 2], Storage::I32(vec![1, 2, 3, 4, 5, 6, 7, 8]))
                    .unwrap(),
            ),
            (
                String::from("on_false"),
                TensorData::from_storage([1, 2], Storage::I32(vec![-1, -2])).unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&g, gathered_cast, &inputs)
                .unwrap()
                .values(),
            &[2., 1., 0., -1., -2., -3., -4., -5.]
        );
        assert_eq!(
            CpuBackend
                .execute(&g, gathered_select, &inputs)
                .unwrap()
                .to_le_bytes()
                .unwrap(),
            TensorData::from_storage([4, 2], Storage::I32(vec![1, 2, -1, -2, 5, 6, -1, -2]))
                .unwrap()
                .to_le_bytes()
                .unwrap()
        );
        assert_eq!(selected.trace().steps.last().unwrap().local_inputs.len(), 2);
        assert_eq!(
            selected.trace().steps.last().unwrap().local_inputs[0]
                .ordered_inputs
                .len(),
            3
        );
    }
    #[test]
    fn local_neg_keeps_reverse_mode_parity_across_equal_shards() {
        for ranks in [1_usize, 2, 4] {
            let mut g = Graph::new();
            let x = g.input_dtype_requires_grad("x", [4, 2], DType::F32, true);
            let sharded = g.shard_node(x, group(ranks), Some(0)).unwrap();
            let negated = g.sharded_unary(&sharded, UnaryOp::Neg).unwrap();
            let dense = if ranks == 1 {
                negated.nodes()[0]
            } else {
                g.gather_sharded(&negated).unwrap()
            };
            let loss = g.sum_all(dense).unwrap();
            let gradient = g.grad(loss, x).unwrap();
            let inputs = HashMap::from([(
                String::from("x"),
                TensorData::new([4, 2], (0..8).map(|value| value as f32).collect()).unwrap(),
            )]);
            assert_eq!(
                CpuBackend.execute(&g, gradient, &inputs).unwrap().values(),
                &[-1.; 8],
                "{ranks} ranks"
            );
        }
    }
    #[test]
    fn contracting_and_noncontracting_matmul_match_dense() {
        let mut g = Graph::new();
        let a = g.input("a", [4, 4]);
        let b = g.input("b", [4, 2]);
        let a_contract = g.shard_node(a, group(2), Some(1)).unwrap();
        let b_replicated = g.replicate_node(b, group(2)).unwrap();
        let contract = g.sharded_matmul(&a_contract, &b_replicated).unwrap();
        let a_rows = g.shard_node(a, group(2), Some(0)).unwrap();
        let rows = g.sharded_matmul(&a_rows, &b_replicated).unwrap();
        let input = HashMap::from([
            (
                String::from("a"),
                data([4, 4], &(0..16).map(|x| x as f32).collect::<Vec<_>>()),
            ),
            (
                String::from("b"),
                data([4, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]),
            ),
        ]);
        let dense = g.matmul(a, b).unwrap();
        let contract_dense = g.gather_sharded(&contract).unwrap();
        let rows_dense = g.gather_sharded(&rows).unwrap();
        let expected = CpuBackend.execute(&g, dense, &input).unwrap();
        assert_eq!(
            CpuBackend.execute(&g, contract_dense, &input).unwrap(),
            expected
        );
        assert_eq!(
            CpuBackend.execute(&g, rows_dense, &input).unwrap(),
            expected
        );
        assert!(
            contract
                .trace()
                .steps
                .iter()
                .any(|s| s.action == "matmul-sum-all-reduce")
        );
    }
}
