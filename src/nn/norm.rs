//! Stateful and stateless normalization modules.

use super::{
    Mode, ModeForwardOutput, ModeModuleForward, Module, ModuleForward, Parameter, ParameterRestore,
    ParameterSnapshot, PendingModeEffects, StateKind, parameter::next_version, restore_parameters,
    state::join,
};
use crate::{DType, Error, Graph, NodeId, Result, Scalar, Shape, TensorData};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// Result of a BatchNorm graph build. In training mode with running statistics,
/// `pending` must be realized and committed after executing the graph.
pub struct BatchNormOutput {
    pub output: NodeId,
    pub pending: Option<PendingBatchNormStats>,
}

/// A one-shot capability for updating BatchNorm running buffers after execution.
/// It contains only snapshots and node IDs; no parameter lock survives graph work.
pub struct PendingBatchNormStats {
    module_identity: usize,
    running_mean: Parameter,
    running_var: Parameter,
    batches: Parameter,
    mean_version: u64,
    var_version: u64,
    batch_version: u64,
    pub mean: NodeId,
    pub variance: NodeId,
    momentum: f32,
    sample_count: usize,
    used: Arc<AtomicBool>,
}
impl PendingBatchNormStats {
    /// Commits realized batch statistics. A token is single-use and is bound to
    /// the originating module's running-buffer identities and versions.
    pub fn commit_stats(
        &self,
        module: &BatchNorm,
        mean: TensorData,
        variance: TensorData,
    ) -> Result<()> {
        self.reserve()?;
        let result = (|| restore_parameters(self.prepare(module, mean, variance)?))();
        if result.is_err() {
            self.release();
        }
        result
    }

    pub(crate) fn reserve(&self) -> Result<()> {
        if self
            .used
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(Error::BatchNormToken {
                reason: "token already committed",
            });
        }
        Ok(())
    }

    pub(crate) fn release(&self) {
        self.used.store(false, Ordering::Release);
    }

    pub(crate) fn prepare(
        &self,
        module: &BatchNorm,
        mean: TensorData,
        variance: TensorData,
    ) -> Result<Vec<ParameterRestore>> {
        if self.module_identity != module.identity() {
            return Err(Error::BatchNormToken {
                reason: "wrong module",
            });
        }
        let mean_snapshot = self.running_mean.snapshot()?;
        let var_snapshot = self.running_var.snapshot()?;
        let batch_snapshot = self.batches.snapshot()?;
        if Some(mean_snapshot.identity) != module.running_mean.as_ref().map(Parameter::identity)
            || Some(var_snapshot.identity) != module.running_var.as_ref().map(Parameter::identity)
            || batch_snapshot.identity != module.num_batches_tracked.identity()
        {
            return Err(Error::BatchNormToken {
                reason: "wrong running buffers",
            });
        }
        if mean_snapshot.version != self.mean_version
            || var_snapshot.version != self.var_version
            || batch_snapshot.version != self.batch_version
        {
            return Err(Error::BatchNormToken {
                reason: "stale running statistics",
            });
        }
        if mean.shape() != &mean_snapshot.shape
            || variance.shape() != &var_snapshot.shape
            || !mean.dtype().is_float()
            || !variance.dtype().is_float()
        {
            return Err(Error::BatchNormToken {
                reason: "statistics shape or dtype mismatch",
            });
        }
        let batches = batch_snapshot.data.scalar_at(0).as_u64();
        let factor = if self.momentum.is_nan() {
            1.0 / (batches + 1) as f64
        } else {
            self.momentum as f64
        };
        let unbiased = if self.sample_count > 1 {
            self.sample_count as f64 / (self.sample_count - 1) as f64
        } else {
            1.0
        };
        let blend = |old: &TensorData, fresh: &TensorData, correction: f64| -> Result<TensorData> {
            TensorData::from_scalars(
                old.shape().clone(),
                old.dtype(),
                (0..old.len()).map(|i| {
                    Scalar::F(
                        (1.0 - factor) * old.scalar_at(i).as_f64()
                            + factor * fresh.scalar_at(i).as_f64() * correction,
                    )
                }),
            )
        };
        Ok(vec![
            ParameterRestore {
                parameter: self.running_mean.clone(),
                data: blend(&mean_snapshot.data, &mean, 1.0)?,
                expected_version: self.mean_version,
                restored_version: self.mean_version.wrapping_add(1),
            },
            ParameterRestore {
                parameter: self.running_var.clone(),
                data: blend(&var_snapshot.data, &variance, unbiased)?,
                expected_version: self.var_version,
                restored_version: self.var_version.wrapping_add(1),
            },
            ParameterRestore {
                parameter: self.batches.clone(),
                data: TensorData::scalar_with_dtype(Scalar::U(batches.wrapping_add(1)), DType::U64),
                expected_version: self.batch_version,
                restored_version: self.batch_version.wrapping_add(1),
            },
        ])
    }
}

/// Tinygrad-compatible channel BatchNorm for rank-two-or-greater inputs.
pub struct BatchNorm {
    pub weight: Option<Parameter>,
    pub bias: Option<Parameter>,
    pub running_mean: Option<Parameter>,
    pub running_var: Option<Parameter>,
    pub num_batches_tracked: Parameter,
    channels: usize,
    pub eps: f32,
    /// `NaN` selects tinygrad's cumulative-update extension; finite values are momentum.
    pub momentum: f32,
    pub track_running_stats: bool,
    identity: Arc<()>,
}
pub type BatchNorm2d = BatchNorm;
/// Tinygrad aliases `BatchNorm3d` to the rank-two-or-greater `BatchNorm` module.
pub type BatchNorm3d = BatchNorm;
impl BatchNorm {
    pub fn new(
        _graph: &mut Graph,
        channels: usize,
        eps: f32,
        affine: bool,
        track_running_stats: bool,
        momentum: f32,
    ) -> Result<Self> {
        if channels == 0
            || !eps.is_finite()
            || eps < 0.0
            || (!momentum.is_nan() && (!momentum.is_finite() || !(0.0..=1.0).contains(&momentum)))
        {
            return Err(Error::InvalidRandom {
                reason: "invalid BatchNorm configuration",
            });
        }
        let shape = Shape::new([channels]);
        Ok(Self {
            weight: affine.then(|| {
                Parameter::new(
                    TensorData::ones(shape.clone()).expect("valid BatchNorm shape"),
                    true,
                )
            }),
            bias: affine.then(|| {
                Parameter::new(
                    TensorData::zeros(shape.clone()).expect("valid BatchNorm shape"),
                    true,
                )
            }),
            running_mean: track_running_stats.then(|| {
                Parameter::new(
                    TensorData::zeros(shape.clone()).expect("valid BatchNorm shape"),
                    false,
                )
            }),
            running_var: track_running_stats.then(|| {
                Parameter::new(
                    TensorData::ones(shape).expect("valid BatchNorm shape"),
                    false,
                )
            }),
            num_batches_tracked: Parameter::new(
                TensorData::scalar_with_dtype(Scalar::U(0), DType::U64),
                false,
            ),
            channels,
            eps,
            momentum,
            track_running_stats,
            identity: Arc::new(()),
        })
    }
    fn identity(&self) -> usize {
        Arc::as_ptr(&self.identity) as usize
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId, mode: Mode) -> Result<BatchNormOutput> {
        let shape = graph.shape(input)?.clone();
        if shape.rank() < 2 || shape.dims()[1] != self.channels {
            return Err(Error::InvalidReshape {
                from: shape,
                to: Shape::new([0, self.channels]),
            });
        }
        let channels = shape.dims()[1];
        let axes = (0..shape.rank())
            .filter(|&axis| axis != 1)
            .map(|axis| axis as isize)
            .collect::<Vec<_>>();
        let count = axes
            .iter()
            .try_fold(1usize, |n, axis| {
                n.checked_mul(shape.dims()[*axis as usize])
            })
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let stat_shape = Shape::new([channels]);
        let broadcast_shape = Shape::new(
            std::iter::once(1)
                .chain(std::iter::once(channels))
                .chain(std::iter::repeat_n(1, shape.rank() - 2))
                .collect::<Vec<_>>(),
        );
        let training_stats = mode == Mode::Training || !self.track_running_stats;
        let (mean, variance, pending) = if training_stats {
            let mean = graph.reduce(input, crate::ReduceKind::Mean, Some(axes.clone()), false)?;
            let mean_broadcast = graph.reshape(mean, broadcast_shape.clone())?;
            let centered = graph.sub(input, mean_broadcast)?;
            let squared = graph.square(centered)?;
            let variance = graph.reduce(squared, crate::ReduceKind::Mean, Some(axes), false)?;
            let pending = if self.track_running_stats && mode == Mode::Training {
                let mean_snapshot = self
                    .running_mean
                    .as_ref()
                    .ok_or(Error::BatchNormToken {
                        reason: "missing running mean",
                    })?
                    .snapshot()?;
                let var_snapshot = self
                    .running_var
                    .as_ref()
                    .ok_or(Error::BatchNormToken {
                        reason: "missing running variance",
                    })?
                    .snapshot()?;
                let batch_snapshot = self.num_batches_tracked.snapshot()?;
                if mean_snapshot.shape != stat_shape || var_snapshot.shape != stat_shape {
                    return Err(Error::BatchNormToken {
                        reason: "running buffer shape mismatch",
                    });
                }
                Some(PendingBatchNormStats {
                    module_identity: self.identity(),
                    running_mean: self.running_mean.as_ref().unwrap().clone(),
                    running_var: self.running_var.as_ref().unwrap().clone(),
                    batches: self.num_batches_tracked.clone(),
                    mean_version: mean_snapshot.version,
                    var_version: var_snapshot.version,
                    batch_version: batch_snapshot.version,
                    mean,
                    variance,
                    momentum: self.momentum,
                    sample_count: count,
                    used: Arc::new(AtomicBool::new(false)),
                })
            } else {
                None
            };
            (mean, variance, pending)
        } else {
            (
                self.running_mean
                    .as_ref()
                    .ok_or(Error::BatchNormToken {
                        reason: "missing running mean",
                    })?
                    .bind(graph)?,
                self.running_var
                    .as_ref()
                    .ok_or(Error::BatchNormToken {
                        reason: "missing running variance",
                    })?
                    .bind(graph)?,
                None,
            )
        };
        let mean = graph.reshape(mean, broadcast_shape.clone())?;
        let variance = graph.reshape(variance, broadcast_shape.clone())?;
        let centered = graph.sub(input, mean)?;
        let eps = graph.constant(TensorData::scalar(self.eps));
        let variance = graph.add(variance, eps)?;
        let denom = graph.rsqrt(variance)?;
        let mut output = graph.mul(centered, denom)?;
        if let Some(weight) = &self.weight {
            let weight = weight.bind(graph)?;
            let weight = graph.reshape(weight, broadcast_shape.clone())?;
            output = graph.mul(output, weight)?;
        }
        if let Some(bias) = &self.bias {
            let bias = bias.bind(graph)?;
            let bias = graph.reshape(bias, broadcast_shape)?;
            output = graph.add(output, bias)?;
        }
        Ok(BatchNormOutput { output, pending })
    }
}
impl Module for BatchNorm {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        if let Some(x) = &self.weight {
            v(join(p, "weight"), x, StateKind::Parameter);
        }
        if let Some(x) = &self.bias {
            v(join(p, "bias"), x, StateKind::Parameter);
        }
        if let Some(x) = &self.running_mean {
            v(join(p, "running_mean"), x, StateKind::Buffer);
        }
        if let Some(x) = &self.running_var {
            v(join(p, "running_var"), x, StateKind::Buffer);
        }
        v(
            join(p, "num_batches_tracked"),
            &self.num_batches_tracked,
            StateKind::Buffer,
        );
    }
}

impl ModeModuleForward for BatchNorm {
    fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<ModeForwardOutput<'a>> {
        let BatchNormOutput { output, pending } = self.forward(graph, input, mode)?;
        Ok(ModeForwardOutput {
            output,
            pending: pending
                .map(|stats| PendingModeEffects::batchnorm(self, stats))
                .unwrap_or_else(PendingModeEffects::empty),
        })
    }
}

/// Tinygrad GroupNorm over channel groups and all remaining per-sample axes.
pub struct GroupNorm {
    pub weight: Option<Parameter>,
    pub bias: Option<Parameter>,
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f32,
}
impl GroupNorm {
    /// Creates graph-independent host parameters for static module workflows.
    pub fn new_static(
        num_groups: usize,
        num_channels: usize,
        eps: f32,
        affine: bool,
    ) -> Result<Self> {
        Self::new_impl(num_groups, num_channels, eps, affine)
    }

    /// Legacy construction spelling retained for source compatibility.
    pub fn new(
        _graph: &mut Graph,
        num_groups: usize,
        num_channels: usize,
        eps: f32,
        affine: bool,
    ) -> Result<Self> {
        Self::new_static(num_groups, num_channels, eps, affine)
    }

    fn new_impl(num_groups: usize, num_channels: usize, eps: f32, affine: bool) -> Result<Self> {
        if num_groups == 0
            || num_channels == 0
            || num_channels % num_groups != 0
            || !eps.is_finite()
            || eps < 0.0
        {
            return Err(Error::InvalidRandom {
                reason: "invalid GroupNorm configuration",
            });
        }
        let shape = Shape::new([num_channels]);
        Ok(Self {
            weight: affine.then(|| {
                Parameter::new(
                    TensorData::ones(shape.clone()).expect("valid GroupNorm shape"),
                    true,
                )
            }),
            bias: affine.then(|| {
                Parameter::new(
                    TensorData::zeros(shape).expect("valid GroupNorm shape"),
                    true,
                )
            }),
            num_groups,
            num_channels,
            eps,
        })
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        let shape = graph.shape(input)?.clone();
        if shape.rank() < 2 || shape.dims()[1] != self.num_channels {
            return Err(Error::InvalidReshape {
                from: shape,
                to: Shape::new([0, self.num_channels]),
            });
        }
        let n = shape.dims()[0];
        let rest = shape.dims()[2..]
            .iter()
            .try_fold(1usize, |a, &x| a.checked_mul(x))
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let per_group = (self.num_channels / self.num_groups)
            .checked_mul(rest)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let grouped_shape = Shape::new([n, self.num_groups, per_group]);
        grouped_shape.numel()?;
        let grouped = graph.reshape(input, grouped_shape)?;
        let mean = graph.reduce(grouped, crate::ReduceKind::Mean, Some(vec![-1]), true)?;
        let centered = graph.sub(grouped, mean)?;
        let squared = graph.square(centered)?;
        let variance = graph.reduce(squared, crate::ReduceKind::Mean, Some(vec![-1]), true)?;
        let eps = graph.constant(TensorData::scalar(self.eps));
        let variance = graph.add(variance, eps)?;
        let scale = graph.rsqrt(variance)?;
        let normalized = graph.mul(centered, scale)?;
        let mut output = graph.reshape(normalized, shape.clone())?;
        let broadcast = Shape::new(
            std::iter::once(1)
                .chain(std::iter::once(self.num_channels))
                .chain(std::iter::repeat_n(1, shape.rank() - 2))
                .collect::<Vec<_>>(),
        );
        if let Some(w) = &self.weight {
            let w = w.bind(graph)?;
            let w = graph.reshape(w, broadcast.clone())?;
            output = graph.mul(output, w)?;
        }
        if let Some(b) = &self.bias {
            let b = b.bind(graph)?;
            let b = graph.reshape(b, broadcast)?;
            output = graph.add(output, b)?;
        }
        Ok(output)
    }
}
impl Module for GroupNorm {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        if let Some(x) = &self.weight {
            v(join(p, "weight"), x, StateKind::Parameter)
        }
        if let Some(x) = &self.bias {
            v(join(p, "bias"), x, StateKind::Parameter)
        }
    }
}
impl ModuleForward for GroupNorm {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

/// InstanceNorm is GroupNorm with one group per channel, matching tinygrad.
pub struct InstanceNorm {
    inner: GroupNorm,
}
impl InstanceNorm {
    /// Creates graph-independent host parameters for static module workflows.
    pub fn new_static(features: usize, eps: f32, affine: bool) -> Result<Self> {
        Ok(Self {
            inner: GroupNorm::new_static(features, features, eps, affine)?,
        })
    }

    /// Legacy construction spelling retained for source compatibility.
    pub fn new(graph: &mut Graph, features: usize, eps: f32, affine: bool) -> Result<Self> {
        let _ = graph;
        Self::new_static(features, eps, affine)
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        self.inner.forward(graph, input)
    }
}
impl Module for InstanceNorm {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.inner.visit(p, v)
    }
}
impl ModuleForward for InstanceNorm {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

pub struct LayerNorm {
    pub weight: Option<Parameter>,
    pub bias: Option<Parameter>,
    normalized_shape: Shape,
    eps: f32,
}
impl LayerNorm {
    /// Creates graph-independent host parameters for static module workflows.
    pub fn new_static(normalized_shape: impl Into<Shape>, eps: f32, affine: bool) -> Result<Self> {
        Self::new_impl(normalized_shape.into(), eps, affine)
    }

    /// Legacy construction spelling retained for source compatibility.
    pub fn new(
        _graph: &mut Graph,
        normalized_shape: impl Into<Shape>,
        eps: f32,
        affine: bool,
    ) -> Result<Self> {
        Self::new_static(normalized_shape, eps, affine)
    }

    fn new_impl(shape: Shape, eps: f32, affine: bool) -> Result<Self> {
        let shape = shape;
        if shape.rank() == 0 || shape.dims().contains(&0) || !eps.is_finite() || eps < 0.0 {
            return Err(Error::InvalidRandom {
                reason: "invalid LayerNorm shape or epsilon",
            });
        };
        shape.numel()?;
        Ok(Self {
            weight: affine.then(|| {
                Parameter::new(TensorData::ones(shape.clone()).expect("valid shape"), true)
            }),
            bias: affine.then(|| {
                Parameter::new(TensorData::zeros(shape.clone()).expect("valid shape"), true)
            }),
            normalized_shape: shape,
            eps,
        })
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        let shape = graph.shape(input)?.clone();
        if !shape.dims().ends_with(self.normalized_shape.dims()) {
            return Err(Error::InvalidReshape {
                from: shape,
                to: self.normalized_shape.clone(),
            });
        };
        let axes = (0..self.normalized_shape.rank())
            .map(|i| -1 - i as isize)
            .collect();
        let mean = graph.reduce(input, crate::ReduceKind::Mean, Some(axes), true)?;
        let centered = graph.sub(input, mean)?;
        let squared = graph.square(centered)?;
        let variance = graph.reduce(
            squared,
            crate::ReduceKind::Mean,
            Some(
                (0..self.normalized_shape.rank())
                    .map(|i| -1 - i as isize)
                    .collect(),
            ),
            true,
        )?;
        let eps = graph.constant(TensorData::scalar(self.eps));
        let variance = graph.add(variance, eps)?;
        let denominator = graph.sqrt(variance)?;
        let out = graph.div(centered, denominator)?;
        let out = if let Some(weight) = &self.weight {
            let weight = weight.bind(graph)?;
            graph.mul(out, weight)?
        } else {
            out
        };
        if let Some(bias) = &self.bias {
            let bias = bias.bind(graph)?;
            graph.add(out, bias)
        } else {
            Ok(out)
        }
    }
}
impl Module for LayerNorm {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        if let Some(w) = &self.weight {
            v(join(p, "weight"), w, StateKind::Parameter)
        }
        if let Some(b) = &self.bias {
            v(join(p, "bias"), b, StateKind::Parameter)
        }
    }
}
impl ModuleForward for LayerNorm {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

/// Channel-wise layer normalization for NCHW tensors, matching tinygrad's
/// `LayerNorm2d` permutation-to-NHWC contract.
pub struct LayerNorm2d {
    pub inner: LayerNorm,
}
impl LayerNorm2d {
    /// Creates graph-independent channel parameters for static NCHW module workflows.
    pub fn new_static(channels: usize, eps: f32, affine: bool) -> Result<Self> {
        if channels == 0 {
            return Err(Error::InvalidRandom {
                reason: "LayerNorm2d channels must be nonzero",
            });
        }
        Ok(Self {
            inner: LayerNorm::new_static(Shape::new([channels]), eps, affine)?,
        })
    }

    /// Legacy construction spelling retained for source compatibility.
    pub fn new(_graph: &mut Graph, channels: usize, eps: f32, affine: bool) -> Result<Self> {
        Self::new_static(channels, eps, affine)
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        let s = graph.shape(input)?.clone();
        if s.rank() != 4 {
            return Err(Error::InvalidReshape {
                from: s,
                to: Shape::new([0; 4]),
            });
        }
        if s.dims()[1] != self.inner.normalized_shape.dims()[0] {
            return Err(Error::InvalidReshape {
                from: s,
                to: self.inner.normalized_shape.clone(),
            });
        }
        let nhwc = graph.permute(input, vec![0, 2, 3, 1])?;
        let out = self.inner.forward(graph, nhwc)?;
        graph.permute(out, vec![0, 3, 1, 2])
    }
}
impl Module for LayerNorm2d {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.inner.visit(p, v)
    }
}
impl ModuleForward for LayerNorm2d {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

pub struct RMSNorm {
    pub weight: Option<Parameter>,
    dim: usize,
    eps: f32,
}

/// Fully validates tinygrad's RMSNorm descriptor before a graph node or
/// parameter binding is published.  The public layer always computes its
/// statistic in F32, then casts the normalized value back to the input dtype
/// before applying its optional F32 weight.
struct RMSNormPlan {
    input: NodeId,
    input_dtype: DType,
    input_is_empty: bool,
    weight: Option<ParameterSnapshot>,
}

impl RMSNorm {
    /// Creates graph-independent host parameters for static module workflows.
    pub fn new_static(dim: usize, eps: f32, affine: bool) -> Result<Self> {
        Self::new_impl(dim, eps, affine)
    }

    /// Legacy construction spelling retained for source compatibility.
    pub fn new(_graph: &mut Graph, dim: usize, eps: f32, affine: bool) -> Result<Self> {
        Self::new_static(dim, eps, affine)
    }

    fn new_impl(dim: usize, eps: f32, affine: bool) -> Result<Self> {
        if dim == 0 || !eps.is_finite() || eps < 0.0 {
            return Err(Error::InvalidRandom {
                reason: "invalid RMSNorm dimension or epsilon",
            });
        }
        Ok(Self {
            weight: affine
                .then(|| Parameter::new(TensorData::ones(Shape::new([dim])).expect("valid"), true)),
            dim,
            eps,
        })
    }

    fn plan(&self, graph: &Graph, input: NodeId) -> Result<RMSNormPlan> {
        let input_shape = graph.shape(input)?.clone();
        let input_dtype = graph.dtype(input)?;
        if input_shape.dims().last().copied() != Some(self.dim) {
            return Err(Error::InvalidReshape {
                from: input_shape,
                to: Shape::new([self.dim]),
            });
        }
        let input_is_empty = input_shape.numel()? == 0;
        input_shape
            .numel()?
            .checked_mul(input_dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
        input_shape
            .numel()?
            .checked_mul(DType::F32.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
        let mut statistic_dims = input_shape.dims().to_vec();
        *statistic_dims
            .last_mut()
            .expect("RMSNorm rank was checked above") = 1;
        let statistic_shape = Shape::new(statistic_dims);
        statistic_shape
            .numel()?
            .checked_mul(DType::F32.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(statistic_shape.clone()))?;
        let weight = self.weight.as_ref().map(Parameter::snapshot).transpose()?;
        if let Some(weight) = &weight {
            let expected = Shape::new([self.dim]);
            if weight.shape != expected || weight.dtype != DType::F32 {
                return Err(Error::ParameterValueMismatch {
                    expected_shape: expected,
                    actual_shape: weight.shape.clone(),
                    expected_dtype: DType::F32,
                    actual_dtype: weight.dtype,
                });
            }
            if input_shape.broadcast_with(&weight.shape)? != input_shape {
                return Err(Error::BroadcastMismatch {
                    lhs: input_shape.clone(),
                    rhs: weight.shape.clone(),
                });
            }
            input_shape
                .numel()?
                .checked_mul(input_dtype.promote(weight.dtype).itemsize())
                .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
        }
        Ok(RMSNormPlan {
            input,
            input_dtype,
            input_is_empty,
            weight,
        })
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        let plan = self.plan(graph, input)?;
        if plan.input_is_empty {
            return match plan.weight {
                Some(weight) => {
                    let weight = graph.bind_parameter(weight)?;
                    graph.mul(plan.input, weight)
                }
                None => Ok(plan.input),
            };
        }
        let x = graph.cast(plan.input, DType::F32)?;
        let squared = graph.square(x)?;
        let mean = graph.reduce(squared, crate::ReduceKind::Mean, Some(vec![-1]), true)?;
        let eps = graph.constant(TensorData::scalar(self.eps));
        let mean = graph.add(mean, eps)?;
        let scale = graph.rsqrt(mean)?;
        let out = graph.mul(x, scale)?;
        let out = graph.cast(out, plan.input_dtype)?;
        if let Some(weight) = plan.weight {
            let weight = graph.bind_parameter(weight)?;
            graph.mul(out, weight)
        } else {
            Ok(out)
        }
    }
}
impl Module for RMSNorm {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        if let Some(w) = &self.weight {
            v(join(p, "weight"), w, StateKind::Parameter)
        }
    }
}
impl ModuleForward for RMSNorm {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}
