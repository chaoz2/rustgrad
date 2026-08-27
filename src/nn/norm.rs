//! Stateful and stateless normalization modules.

use super::{
    Mode, Module, Parameter, StateKind,
    parameter::{ParameterRestore, restore_parameters},
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
        if self.module_identity != module.identity() {
            return Err(Error::BatchNormToken {
                reason: "wrong module",
            });
        }
        if self.used.swap(true, Ordering::AcqRel) {
            return Err(Error::BatchNormToken {
                reason: "token already committed",
            });
        }
        let result = (|| {
            let mean_snapshot = self.running_mean.snapshot()?;
            let var_snapshot = self.running_var.snapshot()?;
            let batch_snapshot = self.batches.snapshot()?;
            if Some(mean_snapshot.identity) != module.running_mean.as_ref().map(Parameter::identity)
                || Some(var_snapshot.identity)
                    != module.running_var.as_ref().map(Parameter::identity)
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
            let blend =
                |old: &TensorData, fresh: &TensorData, correction: f64| -> Result<TensorData> {
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
            let new_mean = blend(&mean_snapshot.data, &mean, 1.0)?;
            let new_var = blend(&var_snapshot.data, &variance, unbiased)?;
            // Running statistics are one logical state transition. Acquire all
            // locks in the shared parameter-restore order and validate every
            // version before publishing any replacement.
            restore_parameters(vec![
                ParameterRestore {
                    parameter: self.running_mean.clone(),
                    data: new_mean,
                    expected_version: self.mean_version,
                    restored_version: self.mean_version.wrapping_add(1),
                },
                ParameterRestore {
                    parameter: self.running_var.clone(),
                    data: new_var,
                    expected_version: self.var_version,
                    restored_version: self.var_version.wrapping_add(1),
                },
                ParameterRestore {
                    parameter: self.batches.clone(),
                    data: TensorData::scalar_with_dtype(
                        Scalar::U(batches.wrapping_add(1)),
                        DType::U64,
                    ),
                    expected_version: self.batch_version,
                    restored_version: self.batch_version.wrapping_add(1),
                },
            ])?;
            Ok(())
        })();
        if result.is_err() {
            self.used.store(false, Ordering::Release);
        }
        result
    }
}

/// Tinygrad-compatible channel BatchNorm for rank-two-or-greater inputs.
pub struct BatchNorm {
    pub weight: Option<Parameter>,
    pub bias: Option<Parameter>,
    pub running_mean: Option<Parameter>,
    pub running_var: Option<Parameter>,
    pub num_batches_tracked: Parameter,
    pub eps: f32,
    /// `NaN` selects tinygrad's cumulative-update extension; finite values are momentum.
    pub momentum: f32,
    pub track_running_stats: bool,
    identity: Arc<()>,
}
pub type BatchNorm2d = BatchNorm;
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
        if shape.rank() < 2 {
            return Err(Error::InvalidReshape {
                from: shape,
                to: Shape::new([0, 0]),
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

/// Tinygrad GroupNorm over channel groups and all remaining per-sample axes.
pub struct GroupNorm {
    pub weight: Option<Parameter>,
    pub bias: Option<Parameter>,
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f32,
}
impl GroupNorm {
    pub fn new(
        _graph: &mut Graph,
        num_groups: usize,
        num_channels: usize,
        eps: f32,
        affine: bool,
    ) -> Result<Self> {
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
        let grouped = graph.reshape(
            input,
            Shape::new([
                n,
                self.num_groups,
                self.num_channels / self.num_groups * rest,
            ]),
        )?;
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

/// InstanceNorm is GroupNorm with one group per channel, matching tinygrad.
pub struct InstanceNorm {
    inner: GroupNorm,
}
impl InstanceNorm {
    pub fn new(graph: &mut Graph, features: usize, eps: f32, affine: bool) -> Result<Self> {
        Ok(Self {
            inner: GroupNorm::new(graph, features, features, eps, affine)?,
        })
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

pub struct LayerNorm {
    pub weight: Option<Parameter>,
    pub bias: Option<Parameter>,
    normalized_shape: Shape,
    eps: f32,
}
impl LayerNorm {
    pub fn new(
        _graph: &mut Graph,
        normalized_shape: impl Into<Shape>,
        eps: f32,
        affine: bool,
    ) -> Result<Self> {
        let shape = normalized_shape.into();
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

/// Channel-wise layer normalization for NCHW tensors, matching tinygrad's
/// `LayerNorm2d` permutation-to-NHWC contract.
pub struct LayerNorm2d {
    pub inner: LayerNorm,
}
impl LayerNorm2d {
    pub fn new(graph: &mut Graph, channels: usize, eps: f32, affine: bool) -> Result<Self> {
        Ok(Self {
            inner: LayerNorm::new(graph, Shape::new([channels]), eps, affine)?,
        })
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        let s = graph.shape(input)?.clone();
        if s.rank() != 4 {
            return Err(Error::InvalidReshape {
                from: s,
                to: Shape::new([0; 4]),
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

pub struct RMSNorm {
    pub weight: Option<Parameter>,
    dim: usize,
    eps: f32,
}
impl RMSNorm {
    pub fn new(_graph: &mut Graph, dim: usize, eps: f32, affine: bool) -> Result<Self> {
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
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        if graph.shape(input)?.dims().last().copied() != Some(self.dim) {
            return Err(Error::InvalidReshape {
                from: graph.shape(input)?.clone(),
                to: Shape::new([self.dim]),
            });
        }
        let original = graph.dtype(input)?;
        let x = if original == DType::F16 || original == DType::BF16 {
            graph.cast(input, DType::F32)?
        } else {
            input
        };
        let squared = graph.square(x)?;
        let mean = graph.reduce(squared, crate::ReduceKind::Mean, Some(vec![-1]), true)?;
        let eps = graph.constant(TensorData::scalar(self.eps));
        let mean = graph.add(mean, eps)?;
        let scale = graph.rsqrt(mean)?;
        let out = graph.mul(x, scale)?;
        let out = if x != input {
            graph.cast(out, original)?
        } else {
            out
        };
        if let Some(weight) = &self.weight {
            let weight = weight.bind(graph)?;
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
