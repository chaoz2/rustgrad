use super::{Graph, NodeId, Pool2dOptions, ReduceKind, Slice};
use crate::{Error, Result, Scalar};

impl Graph {
    /// Static trailing-spatial max pooling. The reduction composition retains
    /// normal max-reduction tie gradients and is visible in graph traces.
    pub fn max_pool2d(&mut self, input: NodeId, options: Pool2dOptions) -> Result<NodeId> {
        self.pool2d(input, options, true)
    }
    /// Static trailing-spatial average pooling, including border divisor policy.
    pub fn avg_pool2d(&mut self, input: NodeId, options: Pool2dOptions) -> Result<NodeId> {
        self.pool2d(input, options, false)
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
            let need_h = (oh - 1) * o.stride[0] + (o.kernel[0] - 1) * o.dilation[0] + 1;
            let need_w = (ow - 1) * o.stride[1] + (o.kernel[1] - 1) * o.dilation[1] + 1;
            o.padding[1] += need_h.saturating_sub(h + o.padding[0] + o.padding[1]);
            o.padding[3] += need_w.saturating_sub(w + o.padding[2] + o.padding[3]);
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
                self.full_like(result, Scalar::I((o.kernel[0] * o.kernel[1]) as i64), None)?
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
}
