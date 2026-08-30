use super::{
    AttentionOptions, Graph, NodeId, ReduceKind, matmul_shape,
    shape::{normalize_axes, reduction_shape},
};
use crate::{DType, Error, ReductionDType, Result, Scalar, Shape, TensorData};

/// Fully preflighted public Tensor.softmax contract. tinygrad subtracts only
/// a detached Max, then computes Exp through its typed Exp2 composition, a
/// typed Sum, Reciprocal, and a final Mul.
struct SoftmaxPlan {
    shape: Shape,
    source_dtype: DType,
    requested_dtype: DType,
    output_dtype: DType,
    axis: Option<isize>,
    max_shape: Shape,
    exp_work_dtype: DType,
    sum_dtypes: ReductionDType,
    inv_ln2: TensorData,
    empty: bool,
}

struct LogSoftmaxPlan {
    softmax: SoftmaxPlan,
    ln2: TensorData,
}

struct LogsumexpPlan {
    axes: Vec<isize>,
    max_shape: Shape,
    output_shape: Shape,
    source_dtype: DType,
    exp_work_dtype: DType,
    exp_dtype: DType,
    sum_dtypes: ReductionDType,
    output_dtype: DType,
    inv_ln2: TensorData,
    ln2: TensorData,
    max_identity: Option<Scalar>,
}

fn max_identity(dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(false),
        DType::I8 => Scalar::I(i8::MIN.into()),
        DType::U8 => Scalar::U(0),
        DType::I16 => Scalar::I(i16::MIN.into()),
        DType::U16 => Scalar::U(0),
        DType::I32 => Scalar::I(i32::MIN.into()),
        DType::U32 => Scalar::U(0),
        DType::I64 => Scalar::I(i64::MIN),
        DType::U64 => Scalar::U(0),
        DType::F16 | DType::BF16 | DType::F32 | DType::F64 => Scalar::F(f64::NEG_INFINITY),
    }
}

fn logsumexp_plan(
    input: NodeId,
    shape: &Shape,
    source_dtype: DType,
    axes: Option<Vec<isize>>,
    keepdim: bool,
) -> Result<LogsumexpPlan> {
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    let axes = if shape.rank() == 0 {
        if axes
            .as_ref()
            .is_some_and(|axes| axes.iter().any(|axis| !matches!(axis, -1 | 0)))
        {
            return Err(Error::InvalidAttention {
                reason: "logsumexp scalar axes must be -1 or 0",
            });
        }
        Vec::new()
    } else {
        normalize_axes(input, shape.rank(), axes)?
    };
    let max_shape = reduction_shape(shape, &axes, true);
    let output_shape = reduction_shape(shape, &axes, keepdim);
    let exp_dtype = if source_dtype.is_float() {
        source_dtype
    } else {
        DType::F32
    };
    let exp_work_dtype = if exp_dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let sum_dtypes = ReductionDType::sum_default(exp_dtype);
    let output_dtype = source_dtype.promote(sum_dtypes.output);
    extent(shape, source_dtype)?;
    extent(&max_shape, source_dtype)?;
    extent(shape, source_dtype)?; // centered
    extent(shape, exp_work_dtype)?;
    extent(shape, exp_dtype)?;
    extent(&output_shape, sum_dtypes.accumulator)?;
    extent(&output_shape, sum_dtypes.output)?;
    extent(&output_shape, sum_dtypes.output)?; // Log2 * ln(2)
    extent(&output_shape, source_dtype)?; // squeezed detached Max
    extent(&output_shape, output_dtype)?;
    if shape.broadcast_with(&max_shape)? != *shape
        || output_shape.broadcast_with(&output_shape)? != output_shape
        || source_dtype.promote(sum_dtypes.output) != output_dtype
    {
        return Err(Error::InvalidAttention {
            reason: "logsumexp intermediate cannot broadcast",
        });
    }
    let inv_ln2 =
        TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::LOG2_E), exp_work_dtype);
    let ln2 = TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::LN_2), sum_dtypes.output);
    if shape.broadcast_with(inv_ln2.shape())? != *shape
        || exp_work_dtype.promote(inv_ln2.dtype()) != exp_work_dtype
        || output_shape.broadcast_with(ln2.shape())? != output_shape
        || sum_dtypes.output.promote(ln2.dtype()) != sum_dtypes.output
    {
        return Err(Error::InvalidAttention {
            reason: "logsumexp scalar promotion mismatch",
        });
    }
    let max_identity = (max_shape.numel()? > 0
        && axes.iter().any(|axis| shape.dims()[*axis as usize] == 0))
    .then(|| max_identity(source_dtype));
    Ok(LogsumexpPlan {
        axes: axes.into_iter().map(|axis| axis as isize).collect(),
        max_shape,
        output_shape,
        source_dtype,
        exp_work_dtype,
        exp_dtype,
        sum_dtypes,
        output_dtype,
        inv_ln2,
        ln2,
        max_identity,
    })
}

fn softmax_plan(
    input: NodeId,
    shape: &Shape,
    source_dtype: DType,
    axis: isize,
    dtype: Option<DType>,
) -> Result<SoftmaxPlan> {
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    let axis = if shape.rank() == 0 {
        if !matches!(axis, -1 | 0) {
            return Err(Error::InvalidAttention {
                reason: "softmax scalar axis must be -1 or 0",
            });
        }
        None
    } else {
        Some(normalize_axes(input, shape.rank(), Some(vec![axis]))?[0] as isize)
    };
    let max_shape = match axis {
        None => shape.clone(),
        Some(axis) => Shape::new(
            shape
                .dims()
                .iter()
                .enumerate()
                .map(
                    |(index, &dimension)| {
                        if index == axis as usize { 1 } else { dimension }
                    },
                )
                .collect::<Vec<_>>(),
        ),
    };
    let requested_dtype = dtype.unwrap_or(source_dtype);
    // Tensor.exp first lifts exact/narrow storage to its float work width,
    // then restores a floating requested width. An integral requested dtype
    // therefore produces F32 after exp rather than retaining that cast.
    let output_dtype = if requested_dtype.is_float() {
        requested_dtype
    } else {
        DType::F32
    };
    let exp_work_dtype = if output_dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let sum_dtypes = ReductionDType::sum_default(output_dtype);
    extent(shape, source_dtype)?;
    extent(&max_shape, source_dtype)?;
    if shape.broadcast_with(&max_shape)? != *shape {
        return Err(Error::InvalidAttention {
            reason: "softmax Max cannot broadcast to input",
        });
    }
    for dtype in [source_dtype, requested_dtype, exp_work_dtype, output_dtype] {
        extent(shape, dtype)?;
    }
    extent(&max_shape, sum_dtypes.accumulator)?;
    extent(&max_shape, sum_dtypes.output)?;
    extent(&max_shape, sum_dtypes.output)?; // reciprocal
    if shape.broadcast_with(&max_shape)? != *shape
        || output_dtype.promote(sum_dtypes.output) != output_dtype
    {
        return Err(Error::InvalidAttention {
            reason: "softmax reciprocal cannot broadcast to exponentials",
        });
    }
    let inv_ln2 =
        TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::LOG2_E), exp_work_dtype);
    if inv_ln2.dtype() != exp_work_dtype
        || shape.broadcast_with(inv_ln2.shape())? != *shape
        || exp_work_dtype.promote(inv_ln2.dtype()) != exp_work_dtype
    {
        return Err(Error::InvalidAttention {
            reason: "softmax Exp2 scalar promotion mismatch",
        });
    }
    Ok(SoftmaxPlan {
        shape: shape.clone(),
        source_dtype,
        requested_dtype,
        output_dtype,
        axis,
        max_shape,
        exp_work_dtype,
        sum_dtypes,
        inv_ln2,
        empty: shape.numel()? == 0,
    })
}

fn log_softmax_plan(
    input: NodeId,
    shape: &Shape,
    source_dtype: DType,
    axis: isize,
    dtype: Option<DType>,
) -> Result<LogSoftmaxPlan> {
    let softmax = softmax_plan(input, shape, source_dtype, axis, dtype)?;
    let log_dtype = softmax.sum_dtypes.output;
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    // `Tensor.log()` is `log2() * ln(2)` at the concrete Sum storage width.
    extent(&softmax.max_shape, log_dtype)?; // Log2
    extent(&softmax.max_shape, log_dtype)?; // multiplication by ln(2)
    extent(&softmax.shape, softmax.output_dtype)?; // final subtraction
    if softmax.shape.broadcast_with(&softmax.max_shape)? != softmax.shape
        || softmax.output_dtype.promote(log_dtype) != softmax.output_dtype
    {
        return Err(Error::InvalidAttention {
            reason: "log_softmax log cannot broadcast to centered values",
        });
    }
    let ln2 = TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::LN_2), log_dtype);
    if ln2.dtype() != log_dtype
        || softmax.max_shape.broadcast_with(ln2.shape())? != softmax.max_shape
        || log_dtype.promote(ln2.dtype()) != log_dtype
    {
        return Err(Error::InvalidAttention {
            reason: "log_softmax Log2 scalar promotion mismatch",
        });
    }
    Ok(LogSoftmaxPlan { softmax, ln2 })
}

/// Validates the complete private LogSoftmax descriptor without exposing its
/// lowering details. Composite public helpers use this to retain atomic
/// whole-operation preflight before invoking `Graph::log_softmax`.
pub(crate) fn validate_log_softmax_plan(
    input: NodeId,
    shape: &Shape,
    source_dtype: DType,
    axis: isize,
    dtype: Option<DType>,
) -> Result<()> {
    log_softmax_plan(input, shape, source_dtype, axis, dtype).map(|_| ())
}

impl Graph {
    /// Numerically stable log-sum-exp across signed axes.
    pub fn logsumexp(
        &mut self,
        input: NodeId,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let plan = logsumexp_plan(input, &input_node.shape, input_node.dtype, axes, keepdim)?;
        let maximum = if let Some(identity) = plan.max_identity {
            self.full_with_dtype(plan.max_shape.clone(), identity, plan.source_dtype)?
        } else if plan.axes.is_empty() {
            input
        } else {
            self.reduce(input, ReduceKind::Max, Some(plan.axes.clone()), true)?
        };
        let centered = self.sub(input, self.detach(maximum)?)?;
        let exp_work = if plan.exp_work_dtype == plan.source_dtype {
            centered
        } else {
            self.cast(centered, plan.exp_work_dtype)?
        };
        let inv_ln2 = self.constant(plan.inv_ln2);
        let exponentials = self.exp2(self.mul(exp_work, inv_ln2)?)?;
        let exponentials = if plan.exp_dtype == plan.exp_work_dtype {
            exponentials
        } else {
            self.cast(exponentials, plan.exp_dtype)?
        };
        let sum = if plan.axes.is_empty() {
            exponentials
        } else {
            self.reduce_with_dtypes(
                exponentials,
                ReduceKind::Sum,
                Some(plan.axes.clone()),
                keepdim,
                plan.sum_dtypes,
            )?
        };
        let log2 = self.log2(sum)?;
        let ln2 = self.constant(plan.ln2);
        let logged = self.mul(log2, ln2)?;
        let maximum = if keepdim || plan.axes.is_empty() {
            maximum
        } else {
            self.reshape(maximum, plan.output_shape.clone())?
        };
        let output = self.add(logged, maximum)?;
        debug_assert_eq!(
            self.shape(output).expect("LogSumExp preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("LogSumExp preflighted"),
            plan.output_dtype
        );
        Ok(output)
    }

    /// Checked-in tinygrad's `Tensor.logsumexp()` defaults: reduce every axis
    /// and drop those reduced dimensions.
    pub fn logsumexp_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.logsumexp(input, None, false)
    }

    /// Numerically stable softmax over one signed axis. `dtype`, when set,
    /// controls the exp/sum calculation and output dtype like tinygrad.
    pub fn softmax(&mut self, input: NodeId, axis: isize, dtype: Option<DType>) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let plan = softmax_plan(input, &input_node.shape, input_node.dtype, axis, dtype)?;
        // An empty source never enters tinygrad's populated Max/Sum paths.
        // Its observable value is the corresponding typed empty tensor.
        if plan.empty {
            return if plan.output_dtype == plan.source_dtype {
                Ok(input)
            } else {
                self.cast(input, plan.output_dtype)
            };
        }
        let maximum = if let Some(axis) = plan.axis {
            self.reduce(input, ReduceKind::Max, Some(vec![axis]), true)?
        } else {
            input
        };
        debug_assert_eq!(
            self.shape(maximum).expect("Softmax max preflighted"),
            &plan.max_shape
        );
        let centered = self.sub(input, self.detach(maximum)?)?;
        let requested = if plan.requested_dtype == plan.source_dtype {
            centered
        } else {
            self.cast(centered, plan.requested_dtype)?
        };
        let exp_work = if plan.exp_work_dtype == plan.requested_dtype {
            requested
        } else {
            self.cast(requested, plan.exp_work_dtype)?
        };
        let inv_ln2 = self.constant(plan.inv_ln2);
        let exponentials = self.exp2(self.mul(exp_work, inv_ln2)?)?;
        let exponentials = if plan.output_dtype == plan.exp_work_dtype {
            exponentials
        } else {
            self.cast(exponentials, plan.output_dtype)?
        };
        let sum = if let Some(axis) = plan.axis {
            self.reduce_with_dtypes(
                exponentials,
                ReduceKind::Sum,
                Some(vec![axis]),
                true,
                plan.sum_dtypes,
            )?
        } else {
            exponentials
        };
        let reciprocal = self.reciprocal(sum)?;
        let output = self.mul(exponentials, reciprocal)?;
        debug_assert_eq!(
            self.shape(output).expect("Softmax preflighted"),
            &plan.shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("Softmax preflighted"),
            plan.output_dtype
        );
        Ok(output)
    }

    /// Checked-in tinygrad's `Tensor.softmax()` defaults: last axis and no
    /// requested output dtype override.
    pub fn softmax_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.softmax(input, -1, None)
    }

    /// Numerically stable softmin over one signed axis.
    ///
    /// tinygrad spells this public helper literally as
    /// `(-self).softmax(axis, dtype)`. Validate the complete Softmax
    /// descriptor before publishing the Neg node, so an invalid axis or
    /// requested dtype leaves the graph unchanged.
    pub fn softmin(&mut self, input: NodeId, axis: isize, dtype: Option<DType>) -> Result<NodeId> {
        let input_node = self.node(input)?;
        // `neg` preserves both the shape and concrete storage dtype for every
        // source-admitted dtype, including Bool's logical-not lowering.
        // Proving the downstream plan first makes the literal composition
        // atomic with respect to malformed Softmax controls.
        let _plan = softmax_plan(input, &input_node.shape, input_node.dtype, axis, dtype)?;
        let negated = self.neg(input)?;
        self.softmax(negated, axis, dtype)
    }

    /// Checked-in tinygrad's `Tensor.softmin()` defaults.
    pub fn softmin_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.softmin(input, -1, None)
    }

    /// Numerically stable log-softmax over one signed axis.
    pub fn log_softmax(
        &mut self,
        input: NodeId,
        axis: isize,
        dtype: Option<DType>,
    ) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let plan = log_softmax_plan(input, &input_node.shape, input_node.dtype, axis, dtype)?;
        if plan.softmax.empty {
            return if plan.softmax.output_dtype == plan.softmax.source_dtype {
                Ok(input)
            } else {
                self.cast(input, plan.softmax.output_dtype)
            };
        }
        let maximum = if let Some(axis) = plan.softmax.axis {
            self.reduce(input, ReduceKind::Max, Some(vec![axis]), true)?
        } else {
            input
        };
        let centered = self.sub(input, self.detach(maximum)?)?;
        let requested = if plan.softmax.requested_dtype == plan.softmax.source_dtype {
            centered
        } else {
            self.cast(centered, plan.softmax.requested_dtype)?
        };
        let exp_work = if plan.softmax.exp_work_dtype == plan.softmax.requested_dtype {
            requested
        } else {
            self.cast(requested, plan.softmax.exp_work_dtype)?
        };
        let inv_ln2 = self.constant(plan.softmax.inv_ln2);
        let exponentials = self.exp2(self.mul(exp_work, inv_ln2)?)?;
        let exponentials = if plan.softmax.output_dtype == plan.softmax.exp_work_dtype {
            exponentials
        } else {
            self.cast(exponentials, plan.softmax.output_dtype)?
        };
        let sum = if let Some(axis) = plan.softmax.axis {
            self.reduce_with_dtypes(
                exponentials,
                ReduceKind::Sum,
                Some(vec![axis]),
                true,
                plan.softmax.sum_dtypes,
            )?
        } else {
            exponentials
        };
        let log2 = self.log2(sum)?;
        let ln2 = self.constant(plan.ln2);
        let logged = self.mul(log2, ln2)?;
        let output = self.sub(requested, logged)?;
        debug_assert_eq!(
            self.shape(output).expect("LogSoftmax preflighted"),
            &plan.softmax.shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("LogSoftmax preflighted"),
            plan.softmax.output_dtype
        );
        Ok(output)
    }

    /// Applies tinygrad-style Lp normalization along one signed axis.
    ///
    /// `p == 0.0` divides by the count of nonzero elements; all other values
    /// use `sum(abs(input).pow(p)).pow(1 / p)`. The denominator is clamped
    /// below by `eps`, so zero vectors remain finite when `eps` is positive.
    /// Narrow, float8, and exact storage inputs are promoted to a supported
    /// floating compute dtype before the composition; ordinary floating inputs
    /// retain their dtype until the existing reduction-promotion rules apply.
    pub fn normalize_basic(
        &mut self,
        input: NodeId,
        p: f64,
        axis: isize,
        eps: f64,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let normalized_axis = normalize_axes(input, source.shape.rank(), Some(vec![axis]))?[0];
        let compute_dtype = if source.dtype.is_float() && !source.dtype.is_float8() {
            source.dtype
        } else {
            DType::F32
        };
        let input = if source.dtype == compute_dtype {
            input
        } else {
            self.cast(input, compute_dtype)?
        };

        let norm = if p == 0.0 {
            let zero = self.constant(TensorData::scalar_with_dtype(Scalar::I(0), compute_dtype));
            let nonzero = self.ne(input, zero)?;
            self.reduce(
                nonzero,
                ReduceKind::Sum,
                Some(vec![normalized_axis as isize]),
                true,
            )?
        } else {
            let power = self.constant(TensorData::scalar_with_dtype(Scalar::F(p), compute_dtype));
            let absolute = self.abs(input)?;
            let powered = self.pow(absolute, power)?;
            let summed = self.reduce(
                powered,
                ReduceKind::Sum,
                Some(vec![normalized_axis as isize]),
                true,
            )?;
            let exponent = self.constant(TensorData::scalar_with_dtype(
                Scalar::F(p.recip()),
                self.dtype(summed)?,
            ));
            self.pow(summed, exponent)?
        };
        let norm_dtype = self.dtype(norm)?;
        let bound_dtype = if norm_dtype.is_float() {
            norm_dtype
        } else {
            DType::F32
        };
        let epsilon = self.constant(TensorData::scalar_with_dtype(Scalar::F(eps), bound_dtype));
        let denominator = self.maximum(norm, epsilon)?;
        self.div(input, denominator)
    }

    /// Checked-in tinygrad's `Tensor.log_softmax()` defaults.
    pub fn log_softmax_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.log_softmax(input, -1, None)
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
        let threshold = self.constant(TensorData::scalar_with_dtype(Scalar::F(dropout_p), dtype));
        let keep = self.ge(random, threshold)?;
        let zero = self.constant(TensorData::scalar_with_dtype(Scalar::F(0.0), dtype));
        let masked = self.select(keep, input, zero)?;
        let scale = self.constant(TensorData::scalar_with_dtype(
            Scalar::F(1.0 / (1.0 - dropout_p)),
            dtype,
        ));
        self.mul(masked, scale)
    }

    /// Returns the lower triangular part of `input` over its final two axes.
    ///
    /// Positive `diagonal` includes diagonals above the main diagonal and
    /// negative values exclude diagonals below it, matching tinygrad's
    /// `Tensor.tril`. Leading dimensions are broadcast through the generated
    /// boolean mask.
    pub fn tril_static(&mut self, input: NodeId, diagonal: isize) -> Result<NodeId> {
        self.triangular_static(input, diagonal, true, "tril")
    }

    /// Returns the upper triangular part of `input` over its final two axes.
    ///
    /// Positive `diagonal` excludes diagonals below the requested upper
    /// boundary and negative values include lower diagonals, matching
    /// tinygrad's `Tensor.triu`.
    pub fn triu_static(&mut self, input: NodeId, diagonal: isize) -> Result<NodeId> {
        self.triangular_static(input, diagonal, false, "triu")
    }

    fn triangular_static(
        &mut self,
        input: NodeId,
        diagonal: isize,
        lower: bool,
        op: &'static str,
    ) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let rank = shape.rank();
        if rank < 2 {
            return Err(Error::InvalidMovementRank {
                op,
                expected: 2,
                actual: rank,
            });
        }
        let rows = shape.dims()[rank - 2];
        let columns = shape.dims()[rank - 1];
        let rows_i64 = i64::try_from(rows).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        let columns_i64 =
            i64::try_from(columns).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        let diagonal = i64::try_from(diagonal).map_err(|_| Error::ShapeOverflow(shape.clone()))?;

        if rows == 0 || columns == 0 {
            return Ok(input);
        }
        let all_keep = if lower {
            diagonal >= columns_i64 - 1
        } else {
            diagonal <= -(rows_i64 - 1)
        };
        if all_keep {
            return Ok(input);
        }
        let all_zero = if lower {
            diagonal <= -rows_i64
        } else {
            diagonal >= columns_i64
        };
        if all_zero {
            let condition = self.constant(TensorData::scalar_with_dtype(
                Scalar::Bool(false),
                DType::Bool,
            ));
            let zero = self.constant(TensorData::scalar_with_dtype(
                Scalar::I(0),
                self.dtype(input)?,
            ));
            return self.select(condition, input, zero);
        }
        (rows_i64 - 1)
            .checked_add(diagonal)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;

        let row_indices = self.arange(0, rows_i64, 1)?;
        let column_indices = self.arange(0, columns_i64, 1)?;
        let mut row_shape = vec![1; rank];
        row_shape[rank - 2] = rows;
        let mut column_shape = vec![1; rank];
        column_shape[rank - 1] = columns;
        let row_indices = self.reshape(row_indices, Shape::new(row_shape))?;
        let column_indices = self.reshape(column_indices, Shape::new(column_shape))?;
        let boundary = self.constant(TensorData::scalar_with_dtype(
            Scalar::I(diagonal),
            DType::I64,
        ));
        let boundary = self.add(row_indices, boundary)?;
        let keep = if lower {
            self.ge(boundary, column_indices)?
        } else {
            self.le(boundary, column_indices)?
        };
        let zero = self.constant(TensorData::scalar_with_dtype(
            Scalar::I(0),
            self.dtype(input)?,
        ));
        self.select(keep, input, zero)
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
            let causal = self.ones_with_dtype([l, s], DType::Bool)?;
            let causal = self.tril(causal, 0)?;
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
