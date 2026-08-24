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
    pub routes: Vec<RedistributionRoute>,
    /// Ordered rank-local operand identities for a local graph operation.
    pub local_inputs: Vec<LocalInputProvenance>,
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
        ShardedGraphTensor::make(
            self,
            value.layout.clone(),
            nodes,
            value.trace.clone(),
            "local-unary",
            None,
        )
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
        let step = output
            .trace
            .steps
            .last_mut()
            .expect("local binary trace step");
        step.local_inputs = output
            .nodes
            .iter()
            .enumerate()
            .map(|(rank, &consumer_local_node)| LocalInputProvenance {
                rank,
                consumer_local_node,
                ordered_inputs: vec![
                    LocalOperandProvenance {
                        input_node: left.nodes[rank],
                        canonical_schedule_buffer: None,
                        producer_redistribution_destination: left_redistributed
                            .then_some(left.nodes[rank]),
                    },
                    LocalOperandProvenance {
                        input_node: right.nodes[rank],
                        canonical_schedule_buffer: None,
                        producer_redistribution_destination: right_redistributed
                            .then_some(right.nodes[rank]),
                    },
                ],
            })
            .collect();
        Ok(output)
    }
    pub fn sharded_select(
        &mut self,
        condition: &ShardedGraphTensor,
        on_true: &ShardedGraphTensor,
        on_false: &ShardedGraphTensor,
    ) -> Result<ShardedGraphTensor> {
        condition.checked(self)?;
        let values = self.sharded_binary(on_true, on_false, BinaryOp::Add)?; // validates/unifies layout; substitute exact branches below
        if condition.layout.group() != values.layout.group() {
            return Err(shard_error("select condition device group differs"));
        }
        let condition = if condition.layout == values.layout
            || matches!(
                condition.layout.distribution(),
                ShardDistribution::Replicated
            ) {
            condition.clone()
        } else {
            self.redistribute_sharded(condition, values.layout.group().clone(), None)?
        };
        let on_true = if on_true.layout == values.layout
            || matches!(on_true.layout.distribution(), ShardDistribution::Replicated)
        {
            on_true.clone()
        } else {
            self.redistribute_sharded(on_true, values.layout.group().clone(), None)?
        };
        let on_false = if on_false.layout == values.layout
            || matches!(
                on_false.layout.distribution(),
                ShardDistribution::Replicated
            ) {
            on_false.clone()
        } else {
            self.redistribute_sharded(on_false, values.layout.group().clone(), None)?
        };
        let nodes = (0..values.nodes.len())
            .map(|i| self.select(condition.nodes[i], on_true.nodes[i], on_false.nodes[i]))
            .collect::<Result<Vec<_>>>()?;
        ShardedGraphTensor::make(
            self,
            values.layout,
            nodes,
            condition.trace.clone(),
            "local-select",
            None,
        )
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
                .into_iter()
                .reduce(|a, b| self.add(a, b).expect("validated local sum"))
                .ok_or_else(|| shard_error("empty device group"))?;
            let output_shape = self.shape(sum)?.clone();
            let layout = ShardLayout::replicated(
                value.layout.group().clone(),
                output_shape,
                self.dtype(sum)?,
            )?;
            ShardedGraphTensor::make(
                self,
                layout,
                vec![sum; value.layout.group().len()],
                value.trace.clone(),
                "sum-all-reduce",
                Some(format!("sum-all-reduce:{}", value.layout.cache_key())),
            )
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
                .into_iter()
                .reduce(|a, b| self.add(a, b).expect("validated matmul partials"))
                .ok_or_else(|| shard_error("empty device group"))?;
            let layout = ShardLayout::replicated(
                lhs.layout.group().clone(),
                self.shape(total)?.clone(),
                self.dtype(total)?,
            )?;
            return ShardedGraphTensor::make(
                self,
                layout,
                vec![total; lhs.layout.group().len()],
                lhs.trace.clone(),
                "matmul-sum-all-reduce",
                Some(format!("sum-all-reduce:{}", lhs.layout.cache_key())),
            );
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
fn shard_error(reason: impl Into<String>) -> Error {
    Error::Collective {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collective::DeviceId;
    use crate::{Backend, CpuBackend, TensorData};
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
