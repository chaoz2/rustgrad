use super::{Graph, NodeId, Pool2dOptions, ReduceKind, Slice};
use crate::{Error, Result, Scalar};

/// Values plus flattened original-spatial argmax indices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxPool2dOutput {
    pub values: NodeId,
    pub indices: NodeId,
}

impl Graph {
    /// Static trailing-spatial max pooling. The reduction composition retains
    /// normal max-reduction tie gradients and is visible in graph traces.
    pub fn max_pool2d(&mut self, input: NodeId, options: Pool2dOptions) -> Result<NodeId> {
        self.pool2d(input, options, true)
    }
    /// Max pooling with tinygrad-compatible earliest-tie flattened spatial indices.
    pub fn max_pool2d_with_indices(
        &mut self,
        input: NodeId,
        mut o: Pool2dOptions,
    ) -> Result<MaxPool2dOutput> {
        let values = self.max_pool2d(input, o)?;
        let shape = self.shape(input)?.clone();
        let rank = shape.rank();
        if rank < 2 {
            return Err(Error::InvalidAttention {
                reason: "pooling needs at least two spatial dimensions",
            });
        }
        let h = shape.dims()[rank - 2];
        let w = shape.dims()[rank - 1];
        let out = self.shape(values)?.clone();
        let oh = out.dims()[rank - 2];
        let ow = out.dims()[rank - 1];
        if o.ceil_mode {
            let need_h = (oh - 1) * o.stride[0] + (o.kernel[0] - 1) * o.dilation[0] + 1;
            let need_w = (ow - 1) * o.stride[1] + (o.kernel[1] - 1) * o.dilation[1] + 1;
            o.padding[1] += need_h.saturating_sub(h + o.padding[0] + o.padding[1]);
            o.padding[3] += need_w.saturating_sub(w + o.padding[2] + o.padding[3]);
        }
        let spatial = h
            .checked_mul(w)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let base = self.arange(0, spatial as i64, 1)?;
        let base = self.cast(base, crate::DType::I32)?;
        let base = self.reshape(
            base,
            crate::Shape::new([1; 0].into_iter().chain([h, w]).collect::<Vec<_>>()),
        )?;
        let base = self.expand(base, shape.clone())?;
        let mut pad = vec![(0, 0); rank];
        pad[rank - 2] = (o.padding[0], o.padding[1]);
        pad[rank - 1] = (o.padding[2], o.padding[3]);
        let padded = self.pad(base, pad, Scalar::I(i32::MIN as i64))?;
        let mut windows = Vec::new();
        for kh in 0..o.kernel[0] {
            for kw in 0..o.kernel[1] {
                let mut slices = vec![
                    Slice {
                        start: None,
                        stop: None,
                        step: 1
                    };
                    rank
                ];
                slices[rank - 2] = Slice {
                    start: Some((kh * o.dilation[0]) as isize),
                    stop: Some((kh * o.dilation[0] + oh * o.stride[0]) as isize),
                    step: o.stride[0] as isize,
                };
                slices[rank - 1] = Slice {
                    start: Some((kw * o.dilation[1]) as isize),
                    stop: Some((kw * o.dilation[1] + ow * o.stride[1]) as isize),
                    step: o.stride[1] as isize,
                };
                windows.push(self.stride(padded, slices)?);
            }
        }
        let indices = self.stack(windows, -1)?;
        let val_windows = self.max_pool_index_windows(input, o, oh, ow)?;
        // Match the CPU max reduction's NaN-ignoring comparison when choosing
        // an index: a NaN is never equal to the selected finite/infinite max.
        let nan = self.isnan(val_windows)?;
        let negative_infinity = self.full_like(val_windows, Scalar::F(f64::NEG_INFINITY), None)?;
        let val_windows = self.select(nan, negative_infinity, val_windows)?;
        let local = self.argmax(val_windows, Some(-1), false)?;
        let local = self.unsqueeze(local, -1)?;
        let local = self.gather(indices, local, rank)?;
        let indices = self.squeeze(local, Some(-1))?;
        Ok(MaxPool2dOutput { values, indices })
    }
    fn max_pool_index_windows(
        &mut self,
        input: NodeId,
        o: Pool2dOptions,
        oh: usize,
        ow: usize,
    ) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let rank = shape.rank();
        let mut pad = vec![(0, 0); rank];
        pad[rank - 2] = (o.padding[0], o.padding[1]);
        pad[rank - 1] = (o.padding[2], o.padding[3]);
        let padded = self.pad(input, pad, Scalar::F(f64::NEG_INFINITY))?;
        let mut windows = Vec::new();
        for kh in 0..o.kernel[0] {
            for kw in 0..o.kernel[1] {
                let mut s = vec![
                    Slice {
                        start: None,
                        stop: None,
                        step: 1
                    };
                    rank
                ];
                s[rank - 2] = Slice {
                    start: Some((kh * o.dilation[0]) as isize),
                    stop: Some((kh * o.dilation[0] + oh * o.stride[0]) as isize),
                    step: o.stride[0] as isize,
                };
                s[rank - 1] = Slice {
                    start: Some((kw * o.dilation[1]) as isize),
                    stop: Some((kw * o.dilation[1] + ow * o.stride[1]) as isize),
                    step: o.stride[1] as isize,
                };
                windows.push(self.stride(padded, s)?);
            }
        }
        self.stack(windows, -1)
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
}
