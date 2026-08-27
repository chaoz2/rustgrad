use super::{shape::normalize_axes, AttentionOptions, Graph, NodeId, ReduceKind, matmul_shape};
use crate::{DType, Error, Result, Scalar, Shape, TensorData};

impl Graph {
    /// Numerically stable log-sum-exp across signed axes.
    pub fn logsumexp(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let axes = normalize_axes(input, self.shape(input)?.rank(), axes)?;
        let reduction_axes = Some(axes.iter().map(|&axis| axis as isize).collect());
        let maximum = self.reduce(input, ReduceKind::Max, reduction_axes.clone(), true)?;
        let shifted = self.sub(input, maximum)?;
        let exponentials = self.exp(shifted)?;
        let sum = self.reduce(exponentials, ReduceKind::Sum, reduction_axes, keepdim)?;
        let logged = self.log(sum)?;
        let maximum = if keepdim {
            maximum
        } else {
            let mut dims = self.shape(maximum)?.dims().to_vec();
            for axis in axes.into_iter().rev() {
                dims.remove(axis);
            }
            self.reshape(maximum, Shape::new(dims))?
        };
        self.add(logged, maximum)
    }

    /// Numerically stable softmax over one signed axis. `dtype`, when set,
    /// controls the exp/sum calculation and output dtype like tinygrad.
    pub fn softmax(&mut self, input: NodeId, axis: isize, dtype: Option<DType>) -> Result<NodeId> {
        let (shifted, exponentials, sum) = self.softmax_parts(input, axis, dtype)?;
        let _ = shifted;
        self.div(exponentials, sum)
    }

    /// Numerically stable log-softmax over one signed axis.
    pub fn log_softmax(
        &mut self,
        input: NodeId,
        axis: isize,
        dtype: Option<DType>,
    ) -> Result<NodeId> {
        let (shifted, _, sum) = self.softmax_parts(input, axis, dtype)?;
        let logged = self.log(sum)?;
        self.sub(shifted, logged)
    }

    fn softmax_parts(
        &mut self,
        input: NodeId,
        axis: isize,
        dtype: Option<DType>,
    ) -> Result<(NodeId, NodeId, NodeId)> {
        if matches!(dtype, Some(dtype) if !dtype.is_float()) {
            return Err(Error::InvalidAttention {
                reason: "softmax dtype must be floating point",
            });
        }
        let maximum = self.reduce(input, ReduceKind::Max, Some(vec![axis]), true)?;
        let mut shifted = self.sub(input, maximum)?;
        if let Some(dtype) = dtype {
            shifted = self.cast(shifted, dtype)?;
        }
        let exponentials = self.exp(shifted)?;
        let sum = self.reduce(exponentials, ReduceKind::Sum, Some(vec![axis]), true)?;
        Ok((shifted, exponentials, sum))
    }

    /// Compositional scaled dot-product attention for tensors shaped
    /// `[..., heads, sequence, embedding]`.
    pub fn scaled_dot_product_attention(
        &mut self,
        query: NodeId,
        mut key: NodeId,
        mut value: NodeId,
        attn_mask: Option<NodeId>,
        options: AttentionOptions,
    ) -> Result<NodeId> {
        if !options.dropout_p.is_finite() || !(0.0..=1.0).contains(&options.dropout_p) {
            return Err(Error::InvalidAttention {
                reason: "dropout_p must be in [0, 1]",
            });
        }
        let query_shape = self.shape(query)?.clone();
        let key_shape = self.shape(key)?.clone();
        let value_shape = self.shape(value)?.clone();
        for (shape, name) in [
            (&query_shape, "query"),
            (&key_shape, "key"),
            (&value_shape, "value"),
        ] {
            if shape.rank() < 3 {
                return Err(Error::InvalidAttention {
                    reason: "query, key, and value need rank at least three",
                });
            }
            let _ = name;
        }
        for id in [query, key, value] {
            if !self.dtype(id)?.is_float() {
                return Err(Error::InvalidAttention {
                    reason: "query, key, and value must have floating point dtype",
                });
            }
        }
        if key_shape.dims()[key_shape.rank() - 2] != value_shape.dims()[value_shape.rank() - 2] {
            return Err(Error::InvalidAttention {
                reason: "key and value sequence lengths must match",
            });
        }
        if query_shape.dims()[query_shape.rank() - 1] != key_shape.dims()[key_shape.rank() - 1] {
            return Err(Error::InvalidAttention {
                reason: "query and key embedding sizes must match",
            });
        }
        if options.is_causal && attn_mask.is_some() {
            return Err(Error::InvalidAttention {
                reason: "attn_mask cannot be combined with is_causal",
            });
        }
        let (expected_key_shape, expected_value_shape) = if options.enable_gqa {
            (
                gqa_repeated_shape(&query_shape, &key_shape)?,
                gqa_repeated_shape(&query_shape, &value_shape)?,
            )
        } else {
            (key_shape.clone(), value_shape.clone())
        };
        let mut transposed_key_shape = expected_key_shape.dims().to_vec();
        let key_rank = transposed_key_shape.len();
        transposed_key_shape.swap(key_rank - 1, key_rank - 2);
        let score_shape = matmul_shape(&query_shape, &Shape::new(transposed_key_shape)).ok_or(
            Error::InvalidAttention {
                reason: "query and key batch dimensions must broadcast",
            },
        )?;
        score_shape.numel()?;
        matmul_shape(&score_shape, &expected_value_shape)
            .ok_or(Error::InvalidAttention {
                reason: "attention scores and value dimensions must match",
            })?
            .numel()?;
        if let Some(mask) = attn_mask {
            let mask_shape = self.shape(mask)?;
            if mask_shape.broadcast_with(&score_shape).as_ref() != Ok(&score_shape) {
                return Err(Error::InvalidAttention {
                    reason: "attn_mask must broadcast to attention scores",
                });
            }
        }
        let scale = options
            .scale
            .unwrap_or_else(|| 1.0 / (query_shape.dims()[query_shape.rank() - 1] as f64).sqrt());
        if !scale.is_finite() || scale == 0.0 {
            return Err(Error::InvalidAttention {
                reason: "attention scale must be finite and nonzero",
            });
        }
        if options.enable_gqa {
            key = self.repeat_heads_for_gqa(query, key)?;
            value = self.repeat_heads_for_gqa(query, value)?;
        }
        let compute_dtype = self
            .dtype(query)?
            .promote(self.dtype(key)?)
            .promote(DType::F32);
        let query_compute = self.cast(query, compute_dtype)?;
        let key_compute = self.cast(key, compute_dtype)?;
        let rank = self.shape(key_compute)?.rank();
        let mut axes: Vec<_> = (0..rank).collect();
        axes.swap(rank - 1, rank - 2);
        let transposed_key = self.permute(key_compute, axes)?;
        let mut scores = self.matmul(query_compute, transposed_key)?;
        let inverse_scale = self.constant(TensorData::scalar_with_dtype(
            Scalar::F(1.0 / scale),
            compute_dtype,
        ));
        scores = self.div(scores, inverse_scale)?;
        if options.is_causal {
            let l = query_shape.dims()[query_shape.rank() - 2];
            let s = key_shape.dims()[key_shape.rank() - 2];
            let causal = self.constant(TensorData::from_scalars(
                [l, s],
                DType::Bool,
                (0..l).flat_map(|row| (0..s).map(move |column| Scalar::Bool(column <= row))),
            )?);
            scores = self.apply_attention_mask(scores, causal)?;
        } else if let Some(mask) = attn_mask {
            scores = self.apply_attention_mask(scores, mask)?;
        }
        let query_dtype = self.dtype(query)?;
        let scores = self.cast(scores, query_dtype)?;
        let probabilities = self.softmax(scores, -1, None)?;
        let probabilities = if !options.training || options.dropout_p == 0.0 {
            probabilities
        } else if options.dropout_p == 1.0 {
            self.zeros_with_dtype(
                self.shape(probabilities)?.clone(),
                self.dtype(probabilities)?,
            )?
        } else {
            let seed = options.dropout_seed.ok_or(Error::InvalidAttention {
                reason: "training dropout requires an explicit dropout_seed",
            })?;
            let dtype = self.dtype(probabilities)?;
            let random = self.rand(self.shape(probabilities)?.clone(), dtype, seed)?;
            let threshold = self.constant(TensorData::scalar_with_dtype(
                Scalar::F(options.dropout_p),
                dtype,
            ));
            let keep = self.ge(random, threshold)?;
            let zero = self.constant(TensorData::scalar_with_dtype(Scalar::F(0.0), dtype));
            let masked = self.select(keep, probabilities, zero)?;
            let scale = self.constant(TensorData::scalar_with_dtype(
                Scalar::F(1.0 / (1.0 - options.dropout_p)),
                dtype,
            ));
            self.mul(masked, scale)?
        };
        self.matmul(probabilities, value)
    }

    fn apply_attention_mask(&mut self, scores: NodeId, mask: NodeId) -> Result<NodeId> {
        if self.dtype(mask)? == DType::Bool {
            let zero = self.constant(TensorData::scalar_with_dtype(
                Scalar::F(0.0),
                self.dtype(scores)?,
            ));
            let negative_infinity = self.constant(TensorData::scalar_with_dtype(
                Scalar::F(f64::NEG_INFINITY),
                self.dtype(scores)?,
            ));
            self.select(mask, zero, negative_infinity)
                .and_then(|bias| self.add(scores, bias))
        } else {
            self.add(scores, mask)
        }
    }

    fn repeat_heads_for_gqa(&mut self, query: NodeId, input: NodeId) -> Result<NodeId> {
        let query_shape = self.shape(query)?.clone();
        let input_shape = self.shape(input)?.clone();
        let final_shape = gqa_repeated_shape(&query_shape, &input_shape)?;
        let axis = input_shape.rank() - 3;
        let query_heads = query_shape.dims()[axis];
        let input_heads = input_shape.dims()[axis];
        let repeats = query_heads / input_heads;
        let mut reshaped = input_shape.dims().to_vec();
        reshaped.insert(axis + 1, 1);
        let reshaped_input = self.reshape(input, Shape::new(reshaped.clone()))?;
        reshaped[axis + 1] = repeats;
        let expanded = self.expand(reshaped_input, Shape::new(reshaped))?;
        self.reshape(expanded, final_shape)
    }
}

fn gqa_repeated_shape(query: &Shape, input: &Shape) -> Result<Shape> {
    let axis = input.rank() - 3;
    if query.rank() != input.rank() || query.dims()[..axis] != input.dims()[..axis] {
        return Err(Error::InvalidAttention {
            reason: "GQA batch dimensions must match",
        });
    }
    let query_heads = query.dims()[axis];
    let input_heads = input.dims()[axis];
    if input_heads == 0 || query_heads % input_heads != 0 {
        return Err(Error::InvalidAttention {
            reason: "GQA query heads must be a positive multiple of key/value heads",
        });
    }
    let mut output = input.dims().to_vec();
    output[axis] = query_heads;
    let output = Shape::new(output);
    output.numel()?;
    Ok(output)
}
