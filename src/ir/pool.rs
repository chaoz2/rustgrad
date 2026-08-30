use super::{Graph, NodeId, Pool2dOptions, PoolOptions, ReduceKind, Slice};
use crate::{Error, Result, Scalar};

/// Values plus flattened original-spatial argmax indices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxPool2dOutput {
    pub values: NodeId,
    pub indices: NodeId,
}

fn checked_pool_window_count(shape: &crate::Shape, kernel: &[usize]) -> Result<usize> {
    kernel.iter().try_fold(1usize, |count, extent| {
        count
            .checked_mul(*extent)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    })
}

fn checked_pool_divisor(shape: &crate::Shape, kernel: &[usize]) -> Result<i64> {
    i64::try_from(checked_pool_window_count(shape, kernel)?)
        .map_err(|_| Error::ShapeOverflow(shape.clone()))
}

impl Graph {
    /// Static adaptive average pooling over trailing axes. `None` preserves an axis.
    pub fn adaptive_avg_pool(
        &mut self,
        input: NodeId,
        output_size: Vec<Option<usize>>,
    ) -> Result<NodeId> {
        self.adaptive_pool(input, output_size, false)
    }
    /// Static adaptive max pooling over trailing axes. `None` preserves an axis.
    pub fn adaptive_max_pool(
        &mut self,
        input: NodeId,
        output_size: Vec<Option<usize>>,
    ) -> Result<NodeId> {
        self.adaptive_pool(input, output_size, true)
    }
    pub fn adaptive_avg_pool2d(
        &mut self,
        input: NodeId,
        output: [Option<usize>; 2],
    ) -> Result<NodeId> {
        self.adaptive_avg_pool(input, output.into())
    }
    pub fn adaptive_max_pool2d(
        &mut self,
        input: NodeId,
        output: [Option<usize>; 2],
    ) -> Result<NodeId> {
        self.adaptive_max_pool(input, output.into())
    }
    fn adaptive_pool(
        &mut self,
        input: NodeId,
        output_size: Vec<Option<usize>>,
        max: bool,
    ) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let n = output_size.len();
        if n == 0 || n > shape.rank() {
            return Err(Error::InvalidAttention {
                reason: "adaptive output rank must match trailing spatial axes",
            });
        }
        let input_spatial = &shape.dims()[shape.rank() - n..];
        let output = output_size
            .into_iter()
            .zip(input_spatial)
            .map(|(o, &i)| o.unwrap_or(i))
            .collect::<Vec<_>>();
        if output.contains(&0) {
            return Err(Error::InvalidAttention {
                reason: "adaptive output dimensions must be nonzero",
            });
        }
        let mut bins = vec![Vec::new()];
        for (&input_dim, &out_dim) in input_spatial.iter().zip(&output) {
            let mut next = Vec::new();
            for prior in bins {
                for i in 0..out_dim {
                    let start = i * input_dim / out_dim;
                    let end = ((i + 1) * input_dim).div_ceil(out_dim);
                    let mut p = prior.clone();
                    p.push((start, end));
                    next.push(p);
                }
            }
            bins = next;
        }
        let axes = (shape.rank() - n..shape.rank())
            .map(|x| x as isize)
            .collect::<Vec<_>>();
        let mut values = Vec::new();
        for bin in bins {
            let mut slices = vec![
                Slice {
                    start: None,
                    stop: None,
                    step: 1
                };
                shape.rank()
            ];
            for (a, (start, end)) in bin.into_iter().enumerate() {
                slices[shape.rank() - n + a] = Slice {
                    start: Some(start as isize),
                    stop: Some(end as isize),
                    step: 1,
                };
            }
            let window = self.stride(input, slices)?;
            values.push(self.reduce(
                window,
                if max {
                    ReduceKind::Max
                } else {
                    ReduceKind::Mean
                },
                Some(axes.clone()),
                false,
            )?);
        }
        // `stack` intentionally requires multiple inputs because it lowers to
        // `concat`. A single adaptive output bin still needs its trailing
        // output-axis before the final normalized reshape.
        let stacked = if values.len() == 1 {
            self.unsqueeze(values[0], -1)?
        } else {
            self.stack(values, -1)?
        };
        let mut result_shape = shape.dims()[..shape.rank() - n].to_vec();
        result_shape.extend(output);
        self.reshape(stacked, crate::Shape::new(result_shape))
    }
    /// General static trailing-spatial max pooling. This is the Rust mapping of
    /// tinygrad's generalized `max_pool2d` tuple API.
    pub fn max_pool(&mut self, input: NodeId, options: PoolOptions) -> Result<NodeId> {
        self.pool_nd(input, options, true)
    }
    /// General max pooling with tinygrad-compatible flattened spatial indices.
    pub fn max_pool_with_indices(
        &mut self,
        input: NodeId,
        o: PoolOptions,
    ) -> Result<MaxPool2dOutput> {
        let shape = self.shape(input)?.clone();
        let (output, o) = self.normalized_pool_geometry(&shape, o)?;
        let n = o.kernel.len();
        let spatial = shape.dims()[shape.rank() - n..]
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        if spatial > i32::MAX as usize {
            return Err(Error::InvalidAttention {
                reason: "pool indices exceed I32 spatial range",
            });
        }
        let values = self.max_pool(input, o.clone())?;
        let base = self.arange(0, spatial as i64, 1)?;
        let base = self.cast(base, crate::DType::I32)?;
        let mut base_shape = vec![1; shape.rank() - n];
        base_shape.extend_from_slice(&shape.dims()[shape.rank() - n..]);
        let base = self.reshape(base, crate::Shape::new(base_shape))?;
        let base = self.expand(base, shape.clone())?;
        let mut pad = vec![(0, 0); shape.rank()];
        for a in 0..n {
            pad[shape.rank() - n + a] = o.padding[a]
        }
        let indices_padded = self.pad(base, pad.clone(), Scalar::I(i32::MIN as i64))?;
        let values_padded = self.pad(input, pad, Scalar::F(f64::NEG_INFINITY))?;
        let mut offsets = vec![Vec::new()];
        for &k in &o.kernel {
            let mut next = Vec::new();
            for prior in offsets {
                for i in 0..k {
                    let mut x = prior.clone();
                    x.push(i);
                    next.push(x)
                }
            }
            offsets = next
        }
        let mut iv = Vec::new();
        let mut vv = Vec::new();
        for offset in offsets {
            let mut slices = vec![
                Slice {
                    start: None,
                    stop: None,
                    step: 1
                };
                shape.rank()
            ];
            for a in 0..n {
                let axis = shape.rank() - n + a;
                slices[axis] = Slice {
                    start: Some((offset[a] * o.dilation[a]) as isize),
                    stop: Some((offset[a] * o.dilation[a] + output[a] * o.stride[a]) as isize),
                    step: o.stride[a] as isize,
                };
            }
            iv.push(self.stride(indices_padded, slices.clone())?);
            vv.push(self.stride(values_padded, slices)?)
        }
        let indices = self.stack(iv, -1)?;
        let windows = self.stack(vv, -1)?;
        let nan = self.isnan(windows)?;
        let neginf = self.full_like(windows, Scalar::F(f64::NEG_INFINITY), None)?;
        let windows = self.select(nan, neginf, windows)?;
        let local = self.argmax(windows, Some(-1), false)?;
        let local = self.unsqueeze(local, -1)?;
        let local = self.gather(indices, local, shape.rank())?;
        let indices = self.squeeze(local, Some(-1))?;
        Ok(MaxPool2dOutput { values, indices })
    }
    /// General static trailing-spatial average pooling.
    pub fn avg_pool(&mut self, input: NodeId, options: PoolOptions) -> Result<NodeId> {
        self.pool_nd(input, options, false)
    }
    /// Static trailing-spatial max pooling. The reduction composition retains
    /// normal max-reduction tie gradients and is visible in graph traces.
    pub fn max_pool2d(&mut self, input: NodeId, options: Pool2dOptions) -> Result<NodeId> {
        self.pool2d(input, options, true)
    }
    /// 2D convenience wrapper for [`Graph::max_pool_with_indices`].
    pub fn max_pool2d_with_indices(
        &mut self,
        input: NodeId,
        o: Pool2dOptions,
    ) -> Result<MaxPool2dOutput> {
        self.max_pool_with_indices(input, o.into())
    }
    /// Static trailing-spatial average pooling, including border divisor policy.
    pub fn avg_pool2d(&mut self, input: NodeId, options: Pool2dOptions) -> Result<NodeId> {
        self.pool2d(input, options, false)
    }
    /// Computes output extents and the trailing padding needed for ceil-mode
    /// windows. Every N-D pooling consumer uses this normalization so its
    /// stacked windows have identical shapes.
    fn normalized_pool_geometry(
        &self,
        shape: &crate::Shape,
        mut o: PoolOptions,
    ) -> Result<(Vec<usize>, PoolOptions)> {
        let n = o.kernel.len();
        if n == 0
            || shape.rank() < n
            || o.stride.len() != n
            || o.dilation.len() != n
            || o.padding.len() != n
        {
            return Err(Error::InvalidAttention {
                reason: "pool option lengths must match spatial rank",
            });
        }
        if o.kernel
            .iter()
            .chain(&o.stride)
            .chain(&o.dilation)
            .any(|x| *x == 0)
        {
            return Err(Error::InvalidAttention {
                reason: "pool kernel, stride, and dilation must be positive",
            });
        }
        checked_pool_window_count(shape, &o.kernel)?;
        let mut out = Vec::with_capacity(n);
        for a in 0..n {
            let extent = (o.kernel[a] - 1)
                .checked_mul(o.dilation[a])
                .and_then(|x| x.checked_add(1))
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            let total = shape.dims()[shape.rank() - n + a]
                .checked_add(o.padding[a].0)
                .and_then(|x| x.checked_add(o.padding[a].1))
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            if total < extent {
                return Err(Error::InvalidAttention {
                    reason: "pool kernel exceeds padded input",
                });
            }
            let output = if o.ceil_mode {
                (total - extent).div_ceil(o.stride[a]) + 1
            } else {
                (total - extent) / o.stride[a] + 1
            };
            if o.ceil_mode {
                let needed = (output - 1)
                    .checked_mul(o.stride[a])
                    .and_then(|x| x.checked_add(extent))
                    .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                o.padding[a].1 = o.padding[a]
                    .1
                    .checked_add(needed.saturating_sub(total))
                    .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            }
            out.push(output);
        }
        Ok((out, o))
    }
    fn pool_nd(&mut self, input: NodeId, o: PoolOptions, max: bool) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let (out, o) = self.normalized_pool_geometry(&shape, o)?;
        if !max && o.count_include_pad {
            checked_pool_divisor(&shape, &o.kernel)?;
        }
        let n = o.kernel.len();
        let mut pad = vec![(0, 0); shape.rank()];
        for a in 0..n {
            let axis = shape.rank() - n + a;
            pad[axis] = o.padding[a];
        }
        let padded = self.pad(
            input,
            pad.clone(),
            if max {
                Scalar::F(f64::NEG_INFINITY)
            } else {
                Scalar::I(0)
            },
        )?;
        let mut offsets = vec![Vec::new()];
        for &k in &o.kernel {
            let mut next = Vec::new();
            for x in offsets {
                for i in 0..k {
                    let mut y = x.clone();
                    y.push(i);
                    next.push(y)
                }
            }
            offsets = next;
        }
        let mut windows = Vec::new();
        for offset in offsets {
            let mut slices = vec![
                Slice {
                    start: None,
                    stop: None,
                    step: 1
                };
                shape.rank()
            ];
            for a in 0..n {
                let axis = shape.rank() - n + a;
                slices[axis] = Slice {
                    start: Some((offset[a] * o.dilation[a]) as isize),
                    stop: Some((offset[a] * o.dilation[a] + out[a] * o.stride[a]) as isize),
                    step: o.stride[a] as isize,
                };
            }
            windows.push(self.stride(padded, slices)?)
        }
        let stacked = self.stack(windows, -1)?;
        let value = self.reduce(
            stacked,
            if max {
                ReduceKind::Max
            } else {
                ReduceKind::Sum
            },
            Some(vec![-1]),
            false,
        )?;
        if max {
            return Ok(value);
        }
        if o.count_include_pad {
            let d = checked_pool_divisor(&shape, &o.kernel)?;
            let divisor = self.full_like(value, Scalar::I(d), None)?;
            self.div(value, divisor)
        } else {
            let ones = self.full_like(input, Scalar::I(1), None)?;
            let valid = self.pad(ones, pad, Scalar::I(0))?;
            let mut offsets = vec![Vec::new()];
            for &k in &o.kernel {
                let mut next = Vec::new();
                for x in offsets {
                    for i in 0..k {
                        let mut y = x.clone();
                        y.push(i);
                        next.push(y)
                    }
                }
                offsets = next;
            }
            let mut windows = Vec::new();
            for offset in offsets {
                let mut slices = vec![
                    Slice {
                        start: None,
                        stop: None,
                        step: 1
                    };
                    shape.rank()
                ];
                for a in 0..n {
                    let axis = shape.rank() - n + a;
                    slices[axis] = Slice {
                        start: Some((offset[a] * o.dilation[a]) as isize),
                        stop: Some((offset[a] * o.dilation[a] + out[a] * o.stride[a]) as isize),
                        step: o.stride[a] as isize,
                    };
                }
                windows.push(self.stride(valid, slices)?)
            }
            let stacked = self.stack(windows, -1)?;
            let d = self.reduce(stacked, ReduceKind::Sum, Some(vec![-1]), false)?;
            self.div(value, d)
        }
    }
    fn pool2d(&mut self, input: NodeId, mut o: Pool2dOptions, max: bool) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        if shape.rank() < 2 {
            return Err(Error::InvalidAttention {
                reason: "pooling needs at least two spatial dimensions",
            });
        }
        if o.kernel.contains(&0) || o.stride.contains(&0) || o.dilation.contains(&0) {
            return Err(Error::InvalidAttention {
                reason: "pool kernel, stride, and dilation must be positive",
            });
        }
        checked_pool_window_count(&shape, &o.kernel)?;
        if !max && o.count_include_pad {
            checked_pool_divisor(&shape, &o.kernel)?;
        }
        let h = shape.dims()[shape.rank() - 2];
        let w = shape.dims()[shape.rank() - 1];
        let output =
            |size: usize, b: usize, a: usize, k: usize, s: usize, d: usize| -> Result<usize> {
                let extent = (k - 1)
                    .checked_mul(d)
                    .and_then(|x| x.checked_add(1))
                    .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                let padded = size
                    .checked_add(b)
                    .and_then(|x| x.checked_add(a))
                    .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                if padded < extent {
                    return Err(Error::InvalidAttention {
                        reason: "pool kernel exceeds padded input",
                    });
                }
                let n = if o.ceil_mode {
                    (padded - extent).div_ceil(s) + 1
                } else {
                    (padded - extent) / s + 1
                };
                Ok(n)
            };
        let oh = output(
            h,
            o.padding[0],
            o.padding[1],
            o.kernel[0],
            o.stride[0],
            o.dilation[0],
        )?;
        let ow = output(
            w,
            o.padding[2],
            o.padding[3],
            o.kernel[1],
            o.stride[1],
            o.dilation[1],
        )?;
        if o.ceil_mode {
            let needed_extent = |output: usize, stride: usize, kernel: usize, dilation: usize| {
                (output - 1)
                    .checked_mul(stride)
                    .and_then(|x| {
                        (kernel - 1)
                            .checked_mul(dilation)
                            .and_then(|extent| x.checked_add(extent))
                    })
                    .and_then(|x| x.checked_add(1))
                    .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            };
            let total_extent = |size: usize, before: usize, after: usize| {
                size.checked_add(before)
                    .and_then(|x| x.checked_add(after))
                    .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            };
            let need_h = needed_extent(oh, o.stride[0], o.kernel[0], o.dilation[0])?;
            let need_w = needed_extent(ow, o.stride[1], o.kernel[1], o.dilation[1])?;
            let total_h = total_extent(h, o.padding[0], o.padding[1])?;
            let total_w = total_extent(w, o.padding[2], o.padding[3])?;
            o.padding[1] = o.padding[1]
                .checked_add(need_h.saturating_sub(total_h))
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            o.padding[3] = o.padding[3]
                .checked_add(need_w.saturating_sub(total_w))
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        }
        let fill = if max {
            Scalar::F(f64::NEG_INFINITY)
        } else {
            Scalar::I(0)
        };
        let mut pad = vec![(0, 0); shape.rank()];
        pad[shape.rank() - 2] = (o.padding[0], o.padding[1]);
        pad[shape.rank() - 1] = (o.padding[2], o.padding[3]);
        let padded = self.pad(input, pad.clone(), fill)?;
        let mut windows = Vec::new();
        for kh in 0..o.kernel[0] {
            for kw in 0..o.kernel[1] {
                let slices = vec![
                    Slice {
                        start: None,
                        stop: None,
                        step: 1
                    };
                    shape.rank()
                ];
                let mut slices = slices;
                slices[shape.rank() - 2] = Slice {
                    start: Some((kh * o.dilation[0]) as isize),
                    stop: Some((kh * o.dilation[0] + oh * o.stride[0]) as isize),
                    step: o.stride[0] as isize,
                };
                slices[shape.rank() - 1] = Slice {
                    start: Some((kw * o.dilation[1]) as isize),
                    stop: Some((kw * o.dilation[1] + ow * o.stride[1]) as isize),
                    step: o.stride[1] as isize,
                };
                windows.push(self.stride(padded, slices)?);
            }
        }
        let stacked = self.stack(windows, -1)?;
        let result = self.reduce(
            stacked,
            if max {
                ReduceKind::Max
            } else {
                ReduceKind::Sum
            },
            Some(vec![-1]),
            false,
        )?;
        if max {
            Ok(result)
        } else {
            let divisor = if o.count_include_pad {
                self.full_like(
                    result,
                    Scalar::I(checked_pool_divisor(&shape, &o.kernel)?),
                    None,
                )?
            } else {
                let ones = self.full_like(input, Scalar::I(1), None)?;
                let valid = self.pad(ones, pad, Scalar::I(0))?;
                let mut terms = Vec::new();
                for kh in 0..o.kernel[0] {
                    for kw in 0..o.kernel[1] {
                        let mut slices = vec![
                            Slice {
                                start: None,
                                stop: None,
                                step: 1
                            };
                            shape.rank()
                        ];
                        slices[shape.rank() - 2] = Slice {
                            start: Some((kh * o.dilation[0]) as isize),
                            stop: Some((kh * o.dilation[0] + oh * o.stride[0]) as isize),
                            step: o.stride[0] as isize,
                        };
                        slices[shape.rank() - 1] = Slice {
                            start: Some((kw * o.dilation[1]) as isize),
                            stop: Some((kw * o.dilation[1] + ow * o.stride[1]) as isize),
                            step: o.stride[1] as isize,
                        };
                        terms.push(self.stride(valid, slices)?);
                    }
                }
                let terms = self.stack(terms, -1)?;
                self.reduce(terms, ReduceKind::Sum, Some(vec![-1]), false)?
            };
            self.div(result, divisor)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Backend, CpuBackend, Graph, Pool2dOptions, Storage, TensorData};
    use std::collections::HashMap;
    fn values(x: TensorData) -> Vec<f32> {
        match x.storage() {
            Storage::F32(v) => v.clone(),
            _ => panic!(),
        }
    }
    #[test]
    fn pool_forward_and_max_gradient_are_compositional() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 3, 3]);
        let opt = Pool2dOptions {
            kernel: [2, 2],
            stride: [1, 1],
            ..Pool2dOptions::default()
        };
        let y = g.max_pool2d(x, opt).unwrap();
        let loss = g.reduce(y, crate::ReduceKind::Sum, None, false).unwrap();
        let grad = g.grad(loss, x).unwrap();
        let data = TensorData::new([1, 1, 3, 3], vec![1., 2., 3., 4., 9., 6., 7., 8., 5.]).unwrap();
        let out = CpuBackend
            .execute(&g, y, &HashMap::from([("x".into(), data.clone())]))
            .unwrap();
        assert_eq!(values(out), vec![9., 9., 9., 9.]);
        let dx = values(
            CpuBackend
                .execute(&g, grad, &HashMap::from([("x".into(), data)]))
                .unwrap(),
        );
        assert_eq!(dx[4], 4.);
    }
    #[test]
    fn avg_pool_uses_kernel_divisor() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 2, 2]);
        let y = g.avg_pool2d(x, Pool2dOptions::default()).unwrap();
        let out = CpuBackend
            .execute(
                &g,
                y,
                &HashMap::from([(
                    "x".into(),
                    TensorData::new([1, 1, 2, 2], vec![1., 2., 3., 4.]).unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(values(out), vec![2.5]);
    }
    #[test]
    fn avg_pool_excludes_padding_from_border_divisor() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 1, 1]);
        let y = g
            .avg_pool2d(
                x,
                Pool2dOptions {
                    kernel: [2, 2],
                    stride: [1, 1],
                    padding: [1, 1, 1, 1],
                    count_include_pad: false,
                    ..Pool2dOptions::default()
                },
            )
            .unwrap();
        let out = CpuBackend
            .execute(
                &g,
                y,
                &HashMap::from([("x".into(), TensorData::new([1, 1, 1, 1], vec![4.]).unwrap())]),
            )
            .unwrap();
        assert_eq!(values(out), vec![4., 4., 4., 4.]);
    }
    #[test]
    fn max_pool_indices_are_flattened_and_choose_earliest_tie() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 2, 3]);
        let output = g
            .max_pool2d_with_indices(
                x,
                Pool2dOptions {
                    kernel: [2, 2],
                    stride: [1, 1],
                    ..Pool2dOptions::default()
                },
            )
            .unwrap();
        let input = TensorData::new([1, 1, 2, 3], vec![1., 9., 9., 2., 3., 4.]).unwrap();
        let pooled = CpuBackend
            .execute(
                &g,
                output.values,
                &HashMap::from([("x".into(), input.clone())]),
            )
            .unwrap();
        assert_eq!(values(pooled), vec![9., 9.]);
        let indices = CpuBackend
            .execute(&g, output.indices, &HashMap::from([("x".into(), input)]))
            .unwrap();
        assert_eq!(indices.storage(), &Storage::I32(vec![1, 1]));
    }
    #[test]
    fn pooling_edge_matrix_preserves_dense_dtypes_and_border_contracts() {
        struct Case {
            name: &'static str,
            dtype: crate::DType,
            input: TensorData,
            expected: Vec<crate::Scalar>,
        }
        let cases = vec![
            Case {
                name: "bool",
                dtype: crate::DType::Bool,
                input: TensorData::from_scalars(
                    [1, 1, 2, 2],
                    crate::DType::Bool,
                    [
                        crate::Scalar::Bool(false),
                        crate::Scalar::Bool(true),
                        crate::Scalar::Bool(true),
                        crate::Scalar::Bool(false),
                    ],
                )
                .unwrap(),
                expected: vec![crate::Scalar::Bool(true)],
            },
            Case {
                name: "i8-min",
                dtype: crate::DType::I8,
                input: TensorData::from_scalars(
                    [1, 1, 2, 2],
                    crate::DType::I8,
                    [
                        crate::Scalar::I(i8::MIN as i64),
                        crate::Scalar::I(-2),
                        crate::Scalar::I(-3),
                        crate::Scalar::I(-4),
                    ],
                )
                .unwrap(),
                expected: vec![crate::Scalar::I(-2)],
            },
            Case {
                name: "u8",
                dtype: crate::DType::U8,
                input: TensorData::from_scalars(
                    [1, 1, 2, 2],
                    crate::DType::U8,
                    [
                        crate::Scalar::U(0),
                        crate::Scalar::U(2),
                        crate::Scalar::U(3),
                        crate::Scalar::U(1),
                    ],
                )
                .unwrap(),
                expected: vec![crate::Scalar::U(3)],
            },
            Case {
                name: "f16",
                dtype: crate::DType::F16,
                input: TensorData::from_scalars(
                    [1, 1, 2, 2],
                    crate::DType::F16,
                    [
                        crate::Scalar::F(1.),
                        crate::Scalar::F(2.),
                        crate::Scalar::F(3.),
                        crate::Scalar::F(4.),
                    ],
                )
                .unwrap(),
                expected: vec![crate::Scalar::F(4.)],
            },
            Case {
                name: "bf16",
                dtype: crate::DType::BF16,
                input: TensorData::from_scalars(
                    [1, 1, 2, 2],
                    crate::DType::BF16,
                    [
                        crate::Scalar::F(1.),
                        crate::Scalar::F(2.),
                        crate::Scalar::F(3.),
                        crate::Scalar::F(4.),
                    ],
                )
                .unwrap(),
                expected: vec![crate::Scalar::F(4.)],
            },
            Case {
                name: "f64",
                dtype: crate::DType::F64,
                input: TensorData::from_scalars(
                    [1, 1, 2, 2],
                    crate::DType::F64,
                    [
                        crate::Scalar::F(1.),
                        crate::Scalar::F(2.),
                        crate::Scalar::F(3.),
                        crate::Scalar::F(4.),
                    ],
                )
                .unwrap(),
                expected: vec![crate::Scalar::F(4.)],
            },
        ];
        for case in cases {
            let mut g = Graph::new();
            let x = g.input_dtype("x", [1, 1, 2, 2], case.dtype);
            let out = g
                .max_pool2d_with_indices(x, Pool2dOptions::default())
                .unwrap();
            let values = CpuBackend
                .execute(&g, out.values, &HashMap::from([("x".into(), case.input)]))
                .unwrap();
            assert_eq!(values.dtype(), case.dtype, "{}", case.name);
            assert_eq!(
                values.scalar_at(0).as_f64(),
                case.expected[0].as_f64(),
                "{}",
                case.name
            );
            let indices = CpuBackend
                .execute(
                    &g,
                    out.indices,
                    &HashMap::from([(
                        "x".into(),
                        TensorData::from_scalars(
                            [1, 1, 2, 2],
                            case.dtype,
                            [
                                crate::Scalar::F(1.),
                                crate::Scalar::F(2.),
                                crate::Scalar::F(3.),
                                crate::Scalar::F(4.),
                            ],
                        )
                        .unwrap(),
                    )]),
                )
                .unwrap();
            assert_eq!(indices.dtype(), crate::DType::I32, "{}", case.name);
        }
    }
    #[test]
    fn max_pool_nan_inf_padding_and_average_gradients_are_explicit() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 2, 2]);
        let out = g
            .max_pool2d_with_indices(x, Pool2dOptions::default())
            .unwrap();
        let data = TensorData::new(
            [1, 1, 2, 2],
            vec![f32::NAN, 1., f32::INFINITY, -f32::INFINITY],
        )
        .unwrap();
        let value = CpuBackend
            .execute(&g, out.values, &HashMap::from([("x".into(), data.clone())]))
            .unwrap();
        assert!(
            value.scalar_at(0).as_f64().is_infinite()
                && value.scalar_at(0).as_f64().is_sign_positive()
        );
        let index = CpuBackend
            .execute(&g, out.indices, &HashMap::from([("x".into(), data)]))
            .unwrap();
        assert_eq!(index.scalar_at(0).as_i64(), 2);
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 1, 1]);
        let opt = Pool2dOptions {
            kernel: [2, 2],
            stride: [1, 1],
            padding: [1; 4],
            count_include_pad: false,
            ..Pool2dOptions::default()
        };
        let y = g.avg_pool2d(x, opt).unwrap();
        let loss = g.reduce(y, crate::ReduceKind::Sum, None, false).unwrap();
        let grad = g.grad(loss, x).unwrap();
        let dx = CpuBackend
            .execute(
                &g,
                grad,
                &HashMap::from([("x".into(), TensorData::new([1, 1, 1, 1], vec![2.]).unwrap())]),
            )
            .unwrap();
        assert!((dx.scalar_at(0).as_f64() - 4.).abs() < 1e-6);
    }
    #[test]
    fn pooling_zero_prefix_ceil_dilation_and_invalid_geometry_matrix() {
        for (name, shape) in [("zero batch", [0, 1, 2, 2]), ("zero channel", [1, 0, 2, 2])] {
            let mut g = Graph::new();
            let x = g.input("x", shape);
            let max = g
                .max_pool2d_with_indices(x, Pool2dOptions::default())
                .unwrap();
            let avg = g.avg_pool2d(x, Pool2dOptions::default()).unwrap();
            let input = TensorData::new(shape, vec![]).unwrap();
            for node in [max.values, max.indices, avg] {
                let out = CpuBackend
                    .execute(&g, node, &HashMap::from([("x".into(), input.clone())]))
                    .unwrap();
                assert_eq!(out.len(), 0, "{name}");
            }
        }
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 0, 2]);
        assert!(g.max_pool2d(x, Pool2dOptions::default()).is_err());
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 3, 3]);
        let opt = Pool2dOptions {
            kernel: [2, 2],
            stride: [2, 2],
            dilation: [1, 2],
            padding: [1, 0, 0, 1],
            ceil_mode: true,
            count_include_pad: false,
        };
        let out = g.max_pool2d_with_indices(x, opt).unwrap();
        let data = TensorData::new([1, 1, 3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap();
        let value = CpuBackend
            .execute(&g, out.values, &HashMap::from([("x".into(), data.clone())]))
            .unwrap();
        let index = CpuBackend
            .execute(&g, out.indices, &HashMap::from([("x".into(), data)]))
            .unwrap();
        assert_eq!(value.shape().dims(), index.shape().dims());
        assert!(index.storage().dtype() == crate::DType::I32);
    }
    #[test]
    fn ceil_pool_preflights_trailing_extent_overflow_before_lowering() {
        let mut g = Graph::new();
        let x = g.input("oversized", [1, 1, usize::MAX, 1]);
        let original_nodes = g.node_count();
        assert!(matches!(
            g.max_pool2d(
                x,
                Pool2dOptions {
                    kernel: [1, 1],
                    stride: [3, 1],
                    ceil_mode: true,
                    ..Pool2dOptions::default()
                }
            ),
            Err(crate::Error::ShapeOverflow(_))
        ));
        assert_eq!(g.node_count(), original_nodes);

        let mut valid = Graph::new();
        let x = valid.input("input", [1, 1, 4, 1]);
        let y = valid
            .avg_pool2d(
                x,
                Pool2dOptions {
                    kernel: [1, 1],
                    stride: [3, 1],
                    ceil_mode: true,
                    ..Pool2dOptions::default()
                },
            )
            .unwrap();
        let out = CpuBackend
            .execute(
                &valid,
                y,
                &HashMap::from([(
                    "input".into(),
                    TensorData::new([1, 1, 4, 1], vec![1., 2., 3., 4.]).unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(values(out), vec![1., 4.]);
    }
    #[test]
    fn signed_zero_ties_keep_earliest_index_and_split_gradient() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 2, 2]);
        let out = g
            .max_pool2d_with_indices(x, Pool2dOptions::default())
            .unwrap();
        let loss = g
            .reduce(out.values, crate::ReduceKind::Sum, None, false)
            .unwrap();
        let grad = g.grad(loss, x).unwrap();
        let input = TensorData::new([1, 1, 2, 2], vec![-0., 0., -1., -2.]).unwrap();
        let index = CpuBackend
            .execute(
                &g,
                out.indices,
                &HashMap::from([("x".into(), input.clone())]),
            )
            .unwrap();
        assert_eq!(index.scalar_at(0).as_i64(), 0);
        let dx = CpuBackend
            .execute(&g, grad, &HashMap::from([("x".into(), input)]))
            .unwrap();
        assert!(
            (dx.scalar_at(0).as_f64() - 0.5).abs() < 1e-6
                && (dx.scalar_at(1).as_f64() - 0.5).abs() < 1e-6
        );
    }
    fn pooled_loss(data: &[f64], options: Pool2dOptions, max: bool) -> f64 {
        let mut g = Graph::new();
        let x = g.input_dtype("x", [1, 1, 3, 3], crate::DType::F64);
        let y = if max {
            g.max_pool2d(x, options).unwrap()
        } else {
            g.avg_pool2d(x, options).unwrap()
        };
        let loss = g.reduce(y, crate::ReduceKind::Sum, None, false).unwrap();
        CpuBackend
            .execute(
                &g,
                loss,
                &HashMap::from([(
                    "x".into(),
                    TensorData::from_scalars(
                        [1, 1, 3, 3],
                        crate::DType::F64,
                        data.iter().copied().map(crate::Scalar::F),
                    )
                    .unwrap(),
                )]),
            )
            .unwrap()
            .scalar_at(0)
            .as_f64()
    }
    fn finite_difference_case(name: &str, options: Pool2dOptions, max: bool) {
        let base = [1., 2., 3., 4., 9., 6., 7., 8., 5.];
        let mut g = Graph::new();
        let x = g.input_dtype("x", [1, 1, 3, 3], crate::DType::F64);
        let y = if max {
            g.max_pool2d(x, options).unwrap()
        } else {
            g.avg_pool2d(x, options).unwrap()
        };
        let loss = g.reduce(y, crate::ReduceKind::Sum, None, false).unwrap();
        let grad = g.grad(loss, x).unwrap();
        let input = TensorData::from_scalars(
            [1, 1, 3, 3],
            crate::DType::F64,
            base.into_iter().map(crate::Scalar::F),
        )
        .unwrap();
        let analytic = CpuBackend
            .execute(&g, grad, &HashMap::from([("x".into(), input)]))
            .unwrap();
        let eps = 1e-5;
        for i in 0..base.len() {
            let mut plus = base;
            plus[i] += eps;
            let mut minus = base;
            minus[i] -= eps;
            let numeric =
                (pooled_loss(&plus, options, max) - pooled_loss(&minus, options, max)) / (2. * eps);
            assert!(
                (analytic.scalar_at(i).as_f64() - numeric).abs() < 1e-6,
                "{name} coordinate {i}"
            );
        }
    }
    #[test]
    fn pooling_input_gradients_match_finite_differences() {
        finite_difference_case(
            "max",
            Pool2dOptions {
                kernel: [2, 2],
                stride: [1, 1],
                ..Pool2dOptions::default()
            },
            true,
        );
        finite_difference_case(
            "avg include",
            Pool2dOptions {
                kernel: [2, 2],
                stride: [2, 1],
                padding: [1, 0, 0, 1],
                ceil_mode: true,
                count_include_pad: true,
                ..Pool2dOptions::default()
            },
            false,
        );
        finite_difference_case(
            "avg exclude",
            Pool2dOptions {
                kernel: [2, 2],
                stride: [2, 1],
                padding: [1, 0, 0, 1],
                ceil_mode: true,
                count_include_pad: false,
                ..Pool2dOptions::default()
            },
            false,
        );
    }
    #[test]
    fn all_nan_and_overflow_geometry_are_explicit() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 2, 2]);
        let out = g
            .max_pool2d_with_indices(x, Pool2dOptions::default())
            .unwrap();
        let data = TensorData::new([1, 1, 2, 2], vec![f32::NAN; 4]).unwrap();
        let value = CpuBackend
            .execute(&g, out.values, &HashMap::from([("x".into(), data.clone())]))
            .unwrap();
        assert!(
            value.scalar_at(0).as_f64().is_infinite()
                && value.scalar_at(0).as_f64().is_sign_negative()
        );
        let index = CpuBackend
            .execute(&g, out.indices, &HashMap::from([("x".into(), data)]))
            .unwrap();
        assert_eq!(index.scalar_at(0).as_i64(), 0);
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 1, 1]);
        assert!(matches!(
            g.avg_pool2d(
                x,
                Pool2dOptions {
                    padding: [usize::MAX, 0, 0, 0],
                    ..Pool2dOptions::default()
                }
            ),
            Err(crate::Error::ShapeOverflow(_))
        ));
    }
    #[test]
    fn max_pool_indices_preflight_spatial_i32_extent_before_lowering() {
        let mut oversized = Graph::new();
        let input = oversized.input("oversized", [1, 46_341, 46_341]);
        let original_nodes = oversized.node_count();
        assert!(matches!(
            oversized.max_pool_with_indices(
                input,
                crate::PoolOptions {
                    kernel: vec![1, 1],
                    stride: vec![1, 1],
                    dilation: vec![1, 1],
                    padding: vec![(0, 0), (0, 0)],
                    ceil_mode: false,
                    count_include_pad: true,
                },
            ),
            Err(crate::Error::InvalidAttention {
                reason: "pool indices exceed I32 spatial range"
            })
        ));
        assert_eq!(oversized.node_count(), original_nodes);

        let mut valid = Graph::new();
        let input = valid.input("input", [1, 2, 2]);
        let output = valid
            .max_pool_with_indices(
                input,
                crate::PoolOptions {
                    kernel: vec![2, 2],
                    stride: vec![1, 1],
                    dilation: vec![1, 1],
                    padding: vec![(0, 0), (0, 0)],
                    ceil_mode: false,
                    count_include_pad: true,
                },
            )
            .unwrap();
        let input = TensorData::new([1, 2, 2], vec![1., 2., 3., 4.]).unwrap();
        assert_eq!(
            values(
                CpuBackend
                    .execute(
                        &valid,
                        output.values,
                        &HashMap::from([("input".into(), input)])
                    )
                    .unwrap()
            ),
            vec![4.]
        );
    }
    #[test]
    fn generalized_pool_preflights_window_count_before_lowering() {
        let mut oversized = Graph::new();
        let input = oversized.input("oversized", [usize::MAX, 2]);
        let original_nodes = oversized.node_count();
        assert!(matches!(
            oversized.avg_pool(
                input,
                crate::PoolOptions {
                    kernel: vec![usize::MAX, 2],
                    stride: vec![1, 1],
                    dilation: vec![1, 1],
                    padding: vec![(0, 0), (0, 0)],
                    ceil_mode: false,
                    count_include_pad: true,
                },
            ),
            Err(crate::Error::ShapeOverflow(_))
        ));
        assert_eq!(oversized.node_count(), original_nodes);

        let mut valid = Graph::new();
        let input = valid.input("input", [2]);
        let output = valid
            .avg_pool(
                input,
                crate::PoolOptions {
                    kernel: vec![2],
                    stride: vec![1],
                    dilation: vec![1],
                    padding: vec![(0, 0)],
                    ceil_mode: false,
                    count_include_pad: true,
                },
            )
            .unwrap();
        let input = TensorData::new([2], vec![2., 6.]).unwrap();
        assert_eq!(
            values(
                CpuBackend
                    .execute(&valid, output, &HashMap::from([("input".into(), input)]))
                    .unwrap()
            ),
            vec![4.]
        );
    }
    #[test]
    fn generalized_three_dimensional_average_pool_matches_fixture() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 2, 2, 2]);
        let y = g
            .avg_pool(
                x,
                crate::PoolOptions {
                    kernel: vec![2, 2, 2],
                    stride: vec![2, 2, 2],
                    dilation: vec![1, 1, 1],
                    padding: vec![(0, 0); 3],
                    ceil_mode: false,
                    count_include_pad: true,
                },
            )
            .unwrap();
        let out = CpuBackend
            .execute(
                &g,
                y,
                &HashMap::from([(
                    "x".into(),
                    TensorData::new([1, 1, 2, 2, 2], (1..=8).map(|x| x as f32).collect()).unwrap(),
                )]),
            )
            .unwrap();
        assert_eq!(values(out), vec![4.5]);
    }
    #[test]
    fn generalized_three_dimensional_max_indices_are_row_major() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 2, 2, 2]);
        let out = g
            .max_pool_with_indices(
                x,
                crate::PoolOptions {
                    kernel: vec![2, 2, 2],
                    stride: vec![2, 2, 2],
                    dilation: vec![1, 1, 1],
                    padding: vec![(0, 0); 3],
                    ceil_mode: false,
                    count_include_pad: true,
                },
            )
            .unwrap();
        let input = TensorData::new([1, 1, 2, 2, 2], vec![1., 2., 3., 4., 5., 6., 7., 9.]).unwrap();
        let pooled = CpuBackend
            .execute(
                &g,
                out.values,
                &HashMap::from([("x".into(), input.clone())]),
            )
            .unwrap();
        assert_eq!(values(pooled), vec![9.]);
        let indices = CpuBackend
            .execute(&g, out.indices, &HashMap::from([("x".into(), input)]))
            .unwrap();
        assert_eq!(indices.scalar_at(0).as_i64(), 7);
    }
    #[test]
    fn generalized_pool_wrapper_and_gradient_matrix() {
        let opt2 = Pool2dOptions {
            kernel: [2, 2],
            stride: [1, 1],
            dilation: [1, 1],
            padding: [1, 0, 0, 1],
            ceil_mode: false,
            count_include_pad: false,
        };
        let mut a = Graph::new();
        let xa = a.input("x", [1, 1, 3, 3]);
        let left = a.max_pool2d_with_indices(xa, opt2).unwrap();
        let mut b = Graph::new();
        let xb = b.input("x", [1, 1, 3, 3]);
        let right = b.max_pool_with_indices(xb, opt2.into()).unwrap();
        let data = TensorData::new([1, 1, 3, 3], vec![1., 2., 3., 4., 9., 6., 7., 8., 5.]).unwrap();
        for (l, r) in [(left.values, right.values), (left.indices, right.indices)] {
            assert_eq!(
                CpuBackend
                    .execute(&a, l, &HashMap::from([("x".into(), data.clone())]))
                    .unwrap(),
                CpuBackend
                    .execute(&b, r, &HashMap::from([("x".into(), data.clone())]))
                    .unwrap()
            );
        }
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 2, 2, 2]);
        let o = crate::PoolOptions {
            kernel: vec![2, 2, 2],
            stride: vec![1, 1, 1],
            dilation: vec![1, 1, 1],
            padding: vec![(0, 0); 3],
            ceil_mode: false,
            count_include_pad: true,
        };
        let out = g.max_pool_with_indices(x, o.clone()).unwrap();
        let loss = g
            .reduce(out.values, crate::ReduceKind::Sum, None, false)
            .unwrap();
        let grad = g.grad(loss, x).unwrap();
        let input = TensorData::new([1, 1, 2, 2, 2], vec![1., 2., 3., 4., 5., 6., 7., 9.]).unwrap();
        let dx = CpuBackend
            .execute(&g, grad, &HashMap::from([("x".into(), input.clone())]))
            .unwrap();
        assert_eq!(dx.scalar_at(7).as_f64(), 1.);
        let index = CpuBackend
            .execute(&g, out.indices, &HashMap::from([("x".into(), input)]))
            .unwrap();
        assert_eq!(index.scalar_at(0).as_i64(), 7);
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 2, 2, 2]);
        let out = g.max_pool_with_indices(x, o).unwrap();
        let loss = g
            .reduce(out.values, crate::ReduceKind::Sum, None, false)
            .unwrap();
        let grad = g.grad(loss, x).unwrap();
        let tied = TensorData::new([1, 1, 2, 2, 2], vec![9., 9., 0., 0., 0., 0., 0., 0.]).unwrap();
        let index = CpuBackend
            .execute(
                &g,
                out.indices,
                &HashMap::from([("x".into(), tied.clone())]),
            )
            .unwrap();
        assert_eq!(index.scalar_at(0).as_i64(), 0);
        let dx = CpuBackend
            .execute(&g, grad, &HashMap::from([("x".into(), tied)]))
            .unwrap();
        assert_eq!(dx.scalar_at(0).as_f64(), 0.5);
        assert_eq!(dx.scalar_at(1).as_f64(), 0.5);
    }
    #[test]
    fn generalized_indices_match_every_2d_geometry() {
        struct Case {
            name: &'static str,
            options: Pool2dOptions,
            input: Vec<f32>,
        }
        let cases = [
            Case {
                name: "plain tie",
                options: Pool2dOptions::default(),
                input: vec![5., 5., 1., 0., 4., 3., 2., 1., 0.],
            },
            Case {
                name: "asymmetric padding",
                options: Pool2dOptions {
                    kernel: [2, 2],
                    stride: [1, 1],
                    dilation: [1, 1],
                    padding: [1, 0, 0, 1],
                    ceil_mode: false,
                    count_include_pad: true,
                },
                input: vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
            },
            Case {
                name: "ceil NaN",
                options: Pool2dOptions {
                    kernel: [2, 2],
                    stride: [2, 2],
                    dilation: [1, 1],
                    padding: [0; 4],
                    ceil_mode: true,
                    count_include_pad: true,
                },
                input: vec![f32::NAN, 2., 3., 4., 5., 6., 7., 8., 9.],
            },
            Case {
                name: "dilation",
                options: Pool2dOptions {
                    kernel: [2, 2],
                    stride: [1, 1],
                    dilation: [2, 1],
                    padding: [0; 4],
                    ceil_mode: false,
                    count_include_pad: true,
                },
                input: vec![1., 2., 3., 4., 9., 6., 7., 8., 5.],
            },
            Case {
                name: "ceil dilation asymmetric",
                options: Pool2dOptions {
                    kernel: [2, 2],
                    stride: [2, 2],
                    dilation: [1, 2],
                    padding: [1, 0, 0, 1],
                    ceil_mode: true,
                    count_include_pad: true,
                },
                input: vec![f32::NAN, 2., 3., 4., 5., 6., 7., 8., 9.],
            },
        ];
        for case in cases {
            let data = TensorData::new([1, 1, 3, 3], case.input).unwrap();
            let mut wrapper = Graph::new();
            let wx = wrapper.input("x", [1, 1, 3, 3]);
            let wrapped = wrapper.max_pool2d_with_indices(wx, case.options).unwrap();
            let mut core = Graph::new();
            let cx = core.input("x", [1, 1, 3, 3]);
            let direct = core.max_pool_with_indices(cx, case.options.into()).unwrap();
            let legacy_values = wrapper.max_pool2d(wx, case.options).unwrap();
            let wrapped_values = CpuBackend
                .execute(
                    &wrapper,
                    wrapped.values,
                    &HashMap::from([("x".into(), data.clone())]),
                )
                .unwrap();
            let direct_values = CpuBackend
                .execute(
                    &core,
                    direct.values,
                    &HashMap::from([("x".into(), data.clone())]),
                )
                .unwrap();
            let old_values = CpuBackend
                .execute(
                    &wrapper,
                    legacy_values,
                    &HashMap::from([("x".into(), data.clone())]),
                )
                .unwrap();
            assert_eq!(
                wrapped_values.shape(),
                direct_values.shape(),
                "{}",
                case.name
            );
            assert_eq!(wrapped_values.shape(), old_values.shape(), "{}", case.name);
            for i in 0..wrapped_values.len() {
                let a = wrapped_values.scalar_at(i).as_f64();
                let b = direct_values.scalar_at(i).as_f64();
                let c = old_values.scalar_at(i).as_f64();
                assert!(
                    (a.is_nan() && b.is_nan()) || a == b,
                    "{} core value {i}",
                    case.name
                );
                assert!(
                    (a.is_nan() && c.is_nan()) || a == c,
                    "{} 2d value {i}",
                    case.name
                );
            }
            let wrapped_indices = CpuBackend
                .execute(
                    &wrapper,
                    wrapped.indices,
                    &HashMap::from([("x".into(), data.clone())]),
                )
                .unwrap();
            let direct_indices = CpuBackend
                .execute(&core, direct.indices, &HashMap::from([("x".into(), data)]))
                .unwrap();
            assert_eq!(wrapped_indices, direct_indices, "{} indices", case.name);
            let wrapper_trace = wrapper.trace(wrapped.indices).unwrap();
            let core_trace = core.trace(direct.indices).unwrap();
            assert!(wrapper_trace.to_string().contains("argmax"));
            assert!(core_trace.to_string().contains("argmax"));
            assert_eq!(
                wrapper_trace.steps.last().unwrap().shape,
                core_trace.steps.last().unwrap().shape,
                "{} trace shape",
                case.name
            );
            assert_eq!(
                wrapper_trace.steps.last().unwrap().dtype,
                core_trace.steps.last().unwrap().dtype,
                "{} trace dtype",
                case.name
            );
        }
    }
    fn avg3_loss(values: &[f64], include: bool) -> f64 {
        let mut g = Graph::new();
        let x = g.input_dtype("x", [1, 1, 2, 2, 2], crate::DType::F64);
        let y = g
            .avg_pool(
                x,
                crate::PoolOptions {
                    kernel: vec![2, 2, 2],
                    stride: vec![1, 1, 1],
                    dilation: vec![1, 1, 1],
                    padding: vec![(1, 0), (0, 1), (1, 0)],
                    ceil_mode: false,
                    count_include_pad: include,
                },
            )
            .unwrap();
        let loss = g.reduce(y, crate::ReduceKind::Sum, None, false).unwrap();
        CpuBackend
            .execute(
                &g,
                loss,
                &HashMap::from([(
                    "x".into(),
                    TensorData::from_scalars(
                        [1, 1, 2, 2, 2],
                        crate::DType::F64,
                        values.iter().copied().map(crate::Scalar::F),
                    )
                    .unwrap(),
                )]),
            )
            .unwrap()
            .scalar_at(0)
            .as_f64()
    }
    #[test]
    fn generalized_three_dimensional_border_average_finite_differences() {
        for include in [true, false] {
            let base = [1., 2., 3., 4., 5., 6., 7., 8.];
            let mut g = Graph::new();
            let x = g.input_dtype("x", [1, 1, 2, 2, 2], crate::DType::F64);
            let y = g
                .avg_pool(
                    x,
                    crate::PoolOptions {
                        kernel: vec![2, 2, 2],
                        stride: vec![1, 1, 1],
                        dilation: vec![1, 1, 1],
                        padding: vec![(1, 0), (0, 1), (1, 0)],
                        ceil_mode: false,
                        count_include_pad: include,
                    },
                )
                .unwrap();
            let loss = g.reduce(y, crate::ReduceKind::Sum, None, false).unwrap();
            let grad = g.grad(loss, x).unwrap();
            let analytic = CpuBackend
                .execute(
                    &g,
                    grad,
                    &HashMap::from([(
                        "x".into(),
                        TensorData::from_scalars(
                            [1, 1, 2, 2, 2],
                            crate::DType::F64,
                            base.into_iter().map(crate::Scalar::F),
                        )
                        .unwrap(),
                    )]),
                )
                .unwrap();
            for i in 0..8 {
                let mut plus = base;
                plus[i] += 1e-5;
                let mut minus = base;
                minus[i] -= 1e-5;
                let numeric = (avg3_loss(&plus, include) - avg3_loss(&minus, include)) / 2e-5;
                assert!(
                    (analytic.scalar_at(i).as_f64() - numeric).abs() < 1e-6,
                    "include={include} coordinate={i}"
                );
            }
        }
    }

    #[test]
    fn adaptive_pool_uses_overlapping_uneven_bins_and_backpropagates() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 5]);
        let avg = g.adaptive_avg_pool(x, vec![Some(3)]).unwrap();
        let max = g.adaptive_max_pool(x, vec![Some(3)]).unwrap();
        let inputs = HashMap::from([(
            "x".into(),
            TensorData::new([1, 1, 5], vec![1., 2., 3., 4., 5.]).unwrap(),
        )]);
        assert_eq!(
            values(CpuBackend.execute(&g, avg, &inputs).unwrap()),
            vec![1.5, 3., 4.5]
        );
        assert_eq!(
            values(CpuBackend.execute(&g, max, &inputs).unwrap()),
            vec![2., 4., 5.]
        );
        let loss = g.reduce(avg, crate::ReduceKind::Sum, None, false).unwrap();
        let grad = g.grad(loss, x).unwrap();
        for (actual, expected) in values(CpuBackend.execute(&g, grad, &inputs).unwrap())
            .iter()
            .zip([0.5, 5. / 6., 1. / 3., 5. / 6., 0.5])
        {
            assert!((actual - expected).abs() < 1e-5);
        }
    }
    #[test]
    fn generalized_three_dimensional_ceil_dilation_nan_indices() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 3, 3, 3]);
        let out = g
            .max_pool_with_indices(
                x,
                crate::PoolOptions {
                    kernel: vec![2, 2, 2],
                    stride: vec![2, 2, 2],
                    dilation: vec![1, 1, 1],
                    padding: vec![(1, 0), (0, 1), (1, 0)],
                    ceil_mode: true,
                    count_include_pad: true,
                },
            )
            .unwrap();
        let mut values = vec![0.; 27];
        values[26] = 9.;
        values[0] = f32::NAN;
        let input = TensorData::new([1, 1, 3, 3, 3], values).unwrap();
        let value = CpuBackend
            .execute(
                &g,
                out.values,
                &HashMap::from([("x".into(), input.clone())]),
            )
            .unwrap();
        let index = CpuBackend
            .execute(&g, out.indices, &HashMap::from([("x".into(), input)]))
            .unwrap();
        assert_eq!(value.dtype(), crate::DType::F32);
        assert_eq!(index.dtype(), crate::DType::I32);
        assert!(index.scalar_at(index.len() - 1).as_i64() >= 0);
    }
}
