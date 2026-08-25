use super::{AttentionOptions, Graph, NodeId, ReduceKind};
use crate::{DType, Error, Result, Scalar, Shape, TensorData};

impl Graph {
    /// Numerically stable log-sum-exp across signed axes.
    pub fn logsumexp(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let maximum = self.reduce(input, ReduceKind::Max, axes.clone(), true)?;
        let shifted = self.sub(input, maximum)?;
        let exponentials = self.exp(shifted)?;
        let sum = self.reduce(exponentials, ReduceKind::Sum, axes.clone(), keepdim)?;
        let logged = self.log(sum)?;
        let maximum = if keepdim {
            maximum
        } else {
            let axes = normalized_axes(self, input, axes)?;
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
        let maximum = self.reduce(input, ReduceKind::Max, Some(vec![axis]), true)?;
        let mut shifted = self.sub(input, maximum)?;
        if let Some(dtype) = dtype {
            if !dtype.is_float() {
                return Err(Error::InvalidAttention {
                    reason: "softmax dtype must be floating point",
                });
            }
            shifted = self.cast(shifted, dtype)?;
        }
        let exponentials = self.exp(shifted)?;
        let sum = self.reduce(exponentials, ReduceKind::Sum, Some(vec![axis]), true)?;
        Ok((shifted, exponentials, sum))
    }

    /// Applies deterministic inverted dropout to a floating tensor.
    ///
    /// Evaluation and `dropout_p = 0` return `input` unchanged. Training
    /// calls require an explicit seed so the constructed graph captures its
    /// Threefry stream rather than reading process-global state.
    pub fn dropout(
        &mut self,
        input: NodeId,
        dropout_p: f64,
        training: bool,
        seed: Option<u64>,
    ) -> Result<NodeId> {
        if !(0.0..=1.0).contains(&dropout_p) {
            return Err(Error::InvalidAttention {
                reason: "dropout_p must be in [0, 1]",
            });
        }
        if !training || dropout_p == 0.0 {
            return Ok(input);
        }
        let dtype = self.dtype(input)?;
        if dropout_p == 1.0 {
            return self.zeros_with_dtype(self.shape(input)?.clone(), dtype);
        }
        if !dtype.is_float() {
            return Err(Error::InvalidAttention {
                reason: "dropout requires a floating point dtype",
            });
        }
        let seed = seed.ok_or(Error::InvalidAttention {
            reason: "training dropout requires an explicit dropout_seed",
        })?;
        let random = self.rand(self.shape(input)?.clone(), dtype, seed)?;
        let threshold = self.constant(TensorData::scalar_with_dtype(
            Scalar::F(dropout_p),
            dtype,
        ));
        let keep = self.ge(random, threshold)?;
        let zero = self.constant(TensorData::scalar_with_dtype(Scalar::F(0.0), dtype));
        let masked = self.select(keep, input, zero)?;
        let scale = self.constant(TensorData::scalar_with_dtype(
            Scalar::F(1.0 / (1.0 - dropout_p)),
            dtype,
        ));
        self.mul(masked, scale)
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
        if !(0.0..=1.0).contains(&options.dropout_p) {
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
        let scale = options
            .scale
            .unwrap_or_else(|| 1.0 / (query_shape.dims()[query_shape.rank() - 1] as f64).sqrt());
        let inverse_scale = self.constant(TensorData::scalar_with_dtype(
            Scalar::F(1.0 / scale),
            compute_dtype,
        ));
        scores = self.div(scores, inverse_scale)?;
        if options.is_causal {
            if attn_mask.is_some() {
                return Err(Error::InvalidAttention {
                    reason: "attn_mask cannot be combined with is_causal",
                });
            }
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
        let probabilities = self.dropout(
            probabilities,
            options.dropout_p,
            options.training,
            options.dropout_seed,
        )?;
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
        let axis = input_shape.rank() - 3;
        if query_shape.rank() != input_shape.rank()
            || query_shape.dims()[..axis] != input_shape.dims()[..axis]
        {
            return Err(Error::InvalidAttention {
                reason: "GQA batch dimensions must match",
            });
        }
        let query_heads = query_shape.dims()[axis];
        let input_heads = input_shape.dims()[axis];
        if input_heads == 0 || query_heads % input_heads != 0 {
            return Err(Error::InvalidAttention {
                reason: "GQA query heads must be a positive multiple of key/value heads",
            });
        }
        let repeats = query_heads / input_heads;
        let mut reshaped = input_shape.dims().to_vec();
        reshaped.insert(axis + 1, 1);
        let reshaped_input = self.reshape(input, Shape::new(reshaped.clone()))?;
        reshaped[axis + 1] = repeats;
        let expanded = self.expand(reshaped_input, Shape::new(reshaped))?;
        let mut final_shape = input_shape.dims().to_vec();
        final_shape[axis] = query_heads;
        self.reshape(expanded, Shape::new(final_shape))
    }
}

fn normalized_axes(graph: &Graph, input: NodeId, axes: Option<Vec<isize>>) -> Result<Vec<usize>> {
    let rank = graph.shape(input)?.rank();
    let mut axes = axes.unwrap_or_else(|| (0..rank).map(|axis| axis as isize).collect());
    for axis in &mut axes {
        if *axis < 0 {
            *axis += rank as isize;
        }
    }
    if axes.iter().any(|axis| *axis < 0 || *axis >= rank as isize) {
        return Err(Error::InvalidReductionAxes {
            node: input,
            axes: axes
                .into_iter()
                .map(|axis| usize::try_from(axis).unwrap_or(usize::MAX))
                .collect(),
            rank,
        });
    }
    let mut axes: Vec<usize> = axes.into_iter().map(|axis| axis as usize).collect();
    axes.sort_unstable();
    if axes.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(Error::InvalidReductionAxes {
            node: input,
            axes,
            rank,
        });
    }
    Ok(axes)
}
