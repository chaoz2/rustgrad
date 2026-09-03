use crate::{
    DType, IndexValue, Operation, ReduceKind, ReductionDType, ReductionPlan, ReductionValue,
    Scalar, Shape, UOp,
};

/// One validated, backend-neutral reduction recurrence.
///
/// The operation remains [`ReduceKind`]; this plan only binds the canonical
/// geometry to the producer, accumulator, and committed output storage types
/// already present in the UOp chain.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct NativeReductionPlan {
    pub(crate) geometry: ReductionPlan,
    pub(crate) kind: ReduceKind,
    pub(crate) source_dtype: DType,
    pub(crate) accumulator_dtype: DType,
    pub(crate) output_dtype: DType,
}

/// One checked reduction recurrence and its optional scalar epilogue under a
/// Store. The epilogue remains ordinary UOps; this view only identifies the
/// exact committed ReduceFinalize value that renderers substitute.
pub(crate) struct NativeReductionKernel<'a> {
    pub(crate) plan: NativeReductionPlan,
    pub(crate) producer: &'a UOp,
    pub(crate) finalize: &'a UOp,
    pub(crate) epilogue_root: &'a UOp,
    pub(crate) output_dtype: DType,
}

impl<'a> NativeReductionKernel<'a> {
    pub(crate) fn from_store(store: &'a UOp) -> Result<Option<Self>, &'static str> {
        if !matches!(store.operation(), Operation::Store) || store.sources().len() != 2 {
            return Err("native reduction kernel lacks Store");
        }
        let index = &store.sources()[0];
        let epilogue_root = &store.sources()[1];
        let mut finalizes = Vec::new();
        fn collect<'a>(node: &'a UOp, finalizes: &mut Vec<&'a UOp>) {
            if matches!(node.operation(), Operation::ReduceFinalize) {
                if !finalizes
                    .iter()
                    .any(|finalize| node.shares_node_with(finalize))
                {
                    finalizes.push(node);
                }
                return;
            }
            for source in node.sources() {
                collect(source, finalizes);
            }
        }
        collect(epilogue_root, &mut finalizes);
        let Some(finalize) = finalizes.first().copied() else {
            return Ok(None);
        };
        if finalizes.len() != 1 {
            return Err("native reduction kernel must contain exactly one ReduceFinalize");
        }
        let (plan, producer) = NativeReductionPlan::from_finalize(finalize)?;
        let (output_shape, output_dtype) = match index.operation() {
            Operation::Index(IndexValue::Buffer { output_shape, .. }) => (
                output_shape,
                index
                    .ty()
                    .ok_or("native reduction output index is untyped")?
                    .scalar,
            ),
            _ => return Err("native reduction output requires a dense Buffer index"),
        };
        if &plan.geometry.output != output_shape
            || epilogue_root
                .ty()
                .ok_or("native reduction epilogue is untyped")?
                .scalar
                != output_dtype
        {
            return Err("native reduction epilogue output descriptor is inconsistent");
        }
        Ok(Some(Self {
            plan,
            producer,
            finalize,
            epilogue_root,
            output_dtype,
        }))
    }

    pub(crate) fn has_epilogue(&self) -> bool {
        !self.epilogue_root.shares_node_with(self.finalize)
    }
}

impl NativeReductionPlan {
    pub(crate) fn new(
        input: Shape,
        output: Shape,
        axes: Vec<usize>,
        keepdim: bool,
        kind: ReduceKind,
        source_dtype: DType,
        dtypes: ReductionDType,
    ) -> Result<Self, &'static str> {
        let accumulator_dtype = dtypes.accumulator;
        let output_dtype = dtypes.output;
        let geometry = ReductionPlan::new(input, output.clone(), axes, keepdim)
            .map_err(|_| "native reduction geometry is invalid")?;
        let expected_output = Shape::new(
            geometry
                .input
                .dims()
                .iter()
                .enumerate()
                .filter_map(|(axis, dimension)| {
                    if geometry.axes.contains(&axis) {
                        geometry.keepdim.then_some(1)
                    } else {
                        Some(*dimension)
                    }
                })
                .collect::<Vec<_>>(),
        );
        if expected_output != output {
            return Err("native reduction output shape is inconsistent");
        }
        let expected_accumulator = match kind {
            ReduceKind::Sum => source_dtype.sum_accumulator_dtype(),
            ReduceKind::Mean if source_dtype.is_float8() => DType::F32,
            ReduceKind::Mean if source_dtype.is_float() => source_dtype,
            ReduceKind::Mean => DType::F32,
            ReduceKind::Product | ReduceKind::Max | ReduceKind::Min => source_dtype,
            ReduceKind::Any | ReduceKind::All => DType::Bool,
        };
        // Explicit reductions may deliberately commit Sum in source storage.
        // Tinygrad-facing default Sum instead supplies
        // `sum_accumulator_dtype()` through its typed plan.
        let explicit_same_storage_sum =
            kind == ReduceKind::Sum && accumulator_dtype == source_dtype;
        // Released RGUA v18 allowed raw Float8 Mean to commit every step in
        // source storage. New Graph lowering uses F32 work and one final
        // Float8 encoding, but the exact historical all-equal tuple remains
        // decodable and executable.
        let legacy_float8_mean = kind == ReduceKind::Mean
            && source_dtype.is_float8()
            && accumulator_dtype == source_dtype
            && output_dtype == source_dtype;
        if accumulator_dtype != expected_accumulator
            && !explicit_same_storage_sum
            && !legacy_float8_mean
        {
            return Err("native reduction dtype contract is inconsistent");
        }
        if matches!(kind, ReduceKind::Any | ReduceKind::All) && source_dtype != DType::Bool {
            return Err("native boolean reduction requires Bool storage");
        }
        let input_elements = geometry
            .input
            .numel()
            .map_err(|_| "native reduction input element count overflows")?;
        input_elements
            .checked_mul(source_dtype.itemsize())
            .ok_or("native reduction input byte count overflows")?;
        let output_elements = geometry
            .output_len()
            .map_err(|_| "native reduction output element count overflows")?;
        output_elements
            .checked_mul(accumulator_dtype.itemsize())
            .ok_or("native reduction accumulator byte count overflows")?;
        output_elements
            .checked_mul(output_dtype.itemsize())
            .ok_or("native reduction output byte count overflows")?;
        geometry
            .reduction_len()
            .map_err(|_| "native reduction domain overflows")?;
        Ok(Self {
            geometry,
            kind,
            source_dtype,
            accumulator_dtype,
            output_dtype,
        })
    }

    pub(crate) fn from_finalize(finalize: &UOp) -> Result<(Self, &UOp), &'static str> {
        if !matches!(finalize.operation(), Operation::ReduceFinalize)
            || finalize.sources().len() != 1
        {
            return Err("native reduction lacks ReduceFinalize");
        }
        let update = &finalize.sources()[0];
        if !matches!(update.operation(), Operation::ReduceAccumulate) || update.sources().len() != 2
        {
            return Err("native reduction lacks ReduceAccumulate");
        }
        let init = &update.sources()[0];
        let value = &update.sources()[1];
        let Operation::ReduceInit(ReductionValue {
            input_shape,
            output_shape,
            axes,
            keepdim,
            kind,
        }) = init.operation()
        else {
            return Err("native reduction lacks ReduceInit");
        };
        let source_ty = value.ty().ok_or("native reduction producer is untyped")?;
        let accumulator_ty = update
            .ty()
            .ok_or("native reduction accumulator is untyped")?;
        let output_ty = finalize.ty().ok_or("native reduction result is untyped")?;
        if source_ty.lanes != 1 || accumulator_ty.lanes != 1 || output_ty.lanes != 1 {
            return Err("native reduction requires scalar lanes");
        }
        let source_dtype = source_ty.scalar;
        let accumulator_dtype = accumulator_ty.scalar;
        let output_dtype = output_ty.scalar;
        if init.ty() != Some(accumulator_ty) || output_ty.lanes != accumulator_ty.lanes {
            return Err("native reduction UOp types are inconsistent");
        }
        Ok((
            Self::new(
                input_shape.clone(),
                output_shape.clone(),
                axes.clone(),
                *keepdim,
                *kind,
                source_dtype,
                ReductionDType::new(accumulator_dtype, output_dtype),
            )?,
            value,
        ))
    }

    pub(crate) fn reduction_len(&self) -> usize {
        self.geometry
            .reduction_len()
            .expect("validated native reduction length")
    }

    pub(crate) fn identity(&self) -> Scalar {
        reduction_identity(self.accumulator_dtype, self.kind)
    }

    pub(crate) fn update(&self, accumulator: Scalar, candidate: Scalar) -> Scalar {
        let dtype = self.accumulator_dtype;
        let candidate = dtype.commit_scalar(candidate);
        if self.is_singleton_identity() {
            return candidate;
        }
        let value = match self.kind {
            ReduceKind::Sum | ReduceKind::Mean => arithmetic(accumulator, candidate, dtype, false),
            ReduceKind::Product => arithmetic(accumulator, candidate, dtype, true),
            ReduceKind::Max => {
                if reduction_extrema_is_better(dtype, true, candidate, accumulator) {
                    candidate
                } else {
                    accumulator
                }
            }
            ReduceKind::Min => {
                if reduction_extrema_is_better(dtype, false, candidate, accumulator) {
                    candidate
                } else {
                    accumulator
                }
            }
            ReduceKind::Any => Scalar::Bool(accumulator.as_bool() || candidate.as_bool()),
            ReduceKind::All => Scalar::Bool(accumulator.as_bool() && candidate.as_bool()),
        };
        dtype.commit_scalar(value)
    }

    pub(crate) fn is_singleton_identity(&self) -> bool {
        self.reduction_len() == 1
    }

    pub(crate) fn finalize(&self, accumulator: Scalar) -> Scalar {
        if self.kind != ReduceKind::Mean {
            return self.output_dtype.commit_scalar(accumulator);
        }
        let value = mean_quotient(accumulator, self.reduction_len(), self.accumulator_dtype)
            .expect("validated Mean accumulator is floating");
        self.output_dtype.commit_scalar(value)
    }

    pub(crate) fn mean_divisor(&self) -> Option<Scalar> {
        (self.kind == ReduceKind::Mean && self.reduction_len() != 0).then(|| {
            self.accumulator_dtype
                .commit_scalar(Scalar::F(self.reduction_len() as f64))
        })
    }
}

/// Divides one Mean numerator by its concrete cardinality at the committed
/// work width. Static native reductions and runtime-cardinality reductions use
/// this same scalar boundary so an F32 divisor/count never inherits host F64
/// precision. Empty Mean is the canonical typed NaN.
pub(crate) fn mean_quotient(numerator: Scalar, count: usize, work_dtype: DType) -> Option<Scalar> {
    if !work_dtype.is_float() {
        return None;
    }
    let numerator = work_dtype.commit_scalar(numerator);
    if count == 0 {
        return Some(work_dtype.commit_scalar(Scalar::F(f64::NAN)));
    }
    let divisor = work_dtype.commit_scalar(Scalar::F(count as f64));
    let quotient = if work_dtype == DType::F64 {
        numerator.as_f64() / divisor.as_f64()
    } else {
        ((numerator.as_f64() as f32) / (divisor.as_f64() as f32)) as f64
    };
    Some(work_dtype.commit_scalar(Scalar::F(quotient)))
}

/// Emits the backend-neutral row-major integer address formula. Callers own
/// the concrete integer type and literal suffix; the reduction geometry owns
/// the coordinate decomposition exactly once.
pub(crate) fn index_expression(
    plan: &ReductionPlan,
    output: &str,
    reduction: &str,
    suffix: &str,
) -> String {
    let output_strides = plan.output.contiguous_strides();
    let reduction_shape = Shape::new(
        plan.axes
            .iter()
            .map(|axis| plan.input.dims()[*axis])
            .collect::<Vec<_>>(),
    );
    let reduction_strides = reduction_shape.contiguous_strides();
    let input_strides = plan.input.contiguous_strides();
    let mut output_axis = 0usize;
    let mut reduction_axis = 0usize;
    let mut terms = Vec::new();
    for (axis, input_stride) in input_strides.iter().copied().enumerate() {
        let (linear, stride, dimension) = if plan.axes.binary_search(&axis).is_ok() {
            let stride = reduction_strides[reduction_axis];
            reduction_axis += 1;
            (reduction, stride, plan.input.dims()[axis])
        } else {
            let stride = output_strides[output_axis];
            let dimension = plan.output.dims()[output_axis];
            output_axis += 1;
            (output, stride, dimension)
        };
        if dimension > 1 {
            terms.push(format!(
                "((({linear} / {stride}{suffix}) % {dimension}{suffix}) * {input_stride}{suffix})"
            ));
        }
        if plan.keepdim && plan.axes.binary_search(&axis).is_ok() {
            output_axis += 1;
        }
    }
    if terms.is_empty() {
        format!("0{suffix}")
    } else {
        terms.join(" + ")
    }
}

pub(crate) fn reduction_identity(dtype: DType, kind: ReduceKind) -> Scalar {
    dtype.commit_scalar(match kind {
        ReduceKind::Sum | ReduceKind::Mean => Scalar::I(0),
        ReduceKind::Product => Scalar::I(1),
        ReduceKind::Max => dtype.min(),
        ReduceKind::Min => dtype.max(),
        ReduceKind::Any => Scalar::Bool(false),
        ReduceKind::All => Scalar::Bool(true),
    })
}

pub(crate) fn reduction_extrema_is_better(
    dtype: DType,
    maximum: bool,
    candidate: Scalar,
    accumulator: Scalar,
) -> bool {
    use core::cmp::Ordering;
    let ordering = if dtype.is_float() {
        candidate.as_f64().partial_cmp(&accumulator.as_f64())
    } else if dtype.is_unsigned() {
        Some(candidate.as_u64().cmp(&accumulator.as_u64()))
    } else if dtype == DType::Bool {
        Some(candidate.as_bool().cmp(&accumulator.as_bool()))
    } else {
        Some(candidate.as_i64().cmp(&accumulator.as_i64()))
    };
    if maximum {
        ordering == Some(Ordering::Greater)
    } else {
        ordering == Some(Ordering::Less)
    }
}

fn arithmetic(lhs: Scalar, rhs: Scalar, dtype: DType, product: bool) -> Scalar {
    if dtype.is_float() {
        return Scalar::F(if product {
            lhs.as_f64() * rhs.as_f64()
        } else {
            lhs.as_f64() + rhs.as_f64()
        });
    }
    if dtype == DType::Bool {
        return Scalar::Bool(if product {
            lhs.as_bool() && rhs.as_bool()
        } else {
            lhs.as_bool() || rhs.as_bool()
        });
    }
    if dtype.is_unsigned() {
        return Scalar::U(if product {
            lhs.as_u64().wrapping_mul(rhs.as_u64())
        } else {
            lhs.as_u64().wrapping_add(rhs.as_u64())
        });
    }
    Scalar::I(if product {
        lhs.as_i64().wrapping_mul(rhs.as_i64())
    } else {
        lhs.as_i64().wrapping_add(rhs.as_i64())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(
        kind: ReduceKind,
        source: DType,
        accumulator: DType,
        len: usize,
    ) -> NativeReductionPlan {
        NativeReductionPlan::new(
            Shape::new([len]),
            Shape::new([]),
            vec![0],
            false,
            kind,
            source,
            ReductionDType::new(accumulator, accumulator),
        )
        .unwrap()
    }

    #[test]
    fn recurrence_commits_each_step_at_the_accumulator_dtype() {
        let sum = plan(ReduceKind::Sum, DType::F32, DType::F32, 3);
        let result = [16_777_216.0, 1.0, -16_777_216.0]
            .into_iter()
            .fold(sum.identity(), |acc, value| {
                sum.update(acc, Scalar::F(value))
            });
        assert_eq!(sum.finalize(result).as_f64(), 0.0);

        for (dtype, large) in [
            (DType::F16, 2_048.0),
            (DType::BF16, 256.0),
            (DType::F8E5M2, 8.0),
        ] {
            let narrow = plan(ReduceKind::Sum, dtype, dtype, 3);
            let result = [large, 1.0, -large]
                .into_iter()
                .fold(narrow.identity(), |acc, value| {
                    narrow.update(acc, Scalar::F(value))
                });
            assert_eq!(narrow.finalize(result).as_f64(), 0.0, "{dtype:?}");
        }

        for dtype in [DType::F16, DType::BF16, DType::F8E5M2] {
            let product = plan(ReduceKind::Product, dtype, dtype, 3);
            let result = [1.1, 1.1, 1.1]
                .into_iter()
                .fold(product.identity(), |acc, value| {
                    product.update(acc, Scalar::F(value))
                });
            let committed = dtype.commit_scalar(result);
            assert_eq!(product.finalize(result), committed, "{dtype:?}");
        }
    }

    #[test]
    fn bool_sum_is_or_and_output_storage_is_independent_of_work_storage() {
        let boolean = NativeReductionPlan::new(
            Shape::new([3]),
            Shape::new([]),
            vec![0],
            false,
            ReduceKind::Sum,
            DType::Bool,
            ReductionDType::new(DType::Bool, DType::Bool),
        )
        .unwrap();
        let result = [false, true, false]
            .into_iter()
            .fold(boolean.identity(), |acc, value| {
                boolean.update(acc, Scalar::Bool(value))
            });
        assert_eq!(boolean.finalize(result), Scalar::Bool(true));

        for output in [DType::F16, DType::BF16] {
            let narrow = NativeReductionPlan::new(
                Shape::new([3]),
                Shape::new([]),
                vec![0],
                false,
                ReduceKind::Sum,
                DType::F32,
                ReductionDType::new(DType::F32, output),
            )
            .unwrap();
            let result = [8.0, 1.0, -8.0]
                .into_iter()
                .fold(narrow.identity(), |acc, value| {
                    narrow.update(acc, Scalar::F(value))
                });
            assert_eq!(
                narrow.finalize(result),
                output.commit_scalar(Scalar::F(1.0))
            );
        }
    }

    #[test]
    fn raw_float8_sum_and_mean_use_f32_work_and_one_final_encoding() {
        for output in [
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E4M3FNUZ,
            DType::F8E5M2FNUZ,
        ] {
            for kind in [ReduceKind::Sum, ReduceKind::Mean] {
                let plan = NativeReductionPlan::new(
                    Shape::new([3]),
                    Shape::new([]),
                    vec![0],
                    false,
                    kind,
                    output,
                    ReductionDType::new(DType::F32, output),
                )
                .unwrap();
                let result = [8.0, 1.0, -8.0]
                    .into_iter()
                    .fold(plan.identity(), |acc, value| {
                        plan.update(acc, Scalar::F(value))
                    });
                let expected = if kind == ReduceKind::Mean {
                    1.0 / 3.0
                } else {
                    1.0
                };
                assert_eq!(
                    plan.finalize(result),
                    output.commit_scalar(Scalar::F(expected)),
                    "{output:?} {kind:?}"
                );
            }
        }
    }

    #[test]
    fn extrema_use_committed_identity_strict_order_and_singleton_bypass() {
        for (kind, identity) in [
            (ReduceKind::Max, f64::NEG_INFINITY),
            (ReduceKind::Min, f64::INFINITY),
        ] {
            let extrema = plan(kind, DType::F32, DType::F32, 2);
            let after_nan = extrema.update(extrema.identity(), Scalar::F(f64::NAN));
            assert_eq!(after_nan.as_f64(), identity);
            assert_eq!(extrema.update(after_nan, Scalar::F(3.0)).as_f64(), 3.0);

            let zero = extrema.update(extrema.identity(), Scalar::F(-0.0));
            let tied = extrema.update(zero, Scalar::F(0.0));
            assert_eq!(tied.as_f64().to_bits(), (-0.0f64).to_bits());

            let singleton = plan(kind, DType::F32, DType::F32, 1);
            assert!(
                singleton
                    .update(singleton.identity(), Scalar::F(f64::NAN))
                    .as_f64()
                    .is_nan()
            );
        }
    }

    #[test]
    fn extrema_identities_are_committed_for_bool_wide_integer_and_float8() {
        let bool_max = plan(ReduceKind::Max, DType::Bool, DType::Bool, 2);
        assert!(!bool_max.identity().as_bool());
        assert!(
            bool_max
                .update(bool_max.identity(), Scalar::Bool(true))
                .as_bool()
        );

        let wide = plan(ReduceKind::Max, DType::U64, DType::U64, 2);
        let above_f64 = (1_u64 << 53) + 1;
        assert_eq!(
            wide.update(Scalar::U(above_f64 - 1), Scalar::U(above_f64))
                .as_u64(),
            above_f64
        );

        for dtype in [DType::F8E4M3, DType::F8E4M3FNUZ, DType::F8E5M2FNUZ] {
            for kind in [ReduceKind::Max, ReduceKind::Min] {
                let extrema = plan(kind, dtype, dtype, 2);
                assert!(extrema.identity().as_f64().is_nan(), "{dtype:?} {kind:?}");
                assert!(
                    extrema
                        .update(extrema.identity(), Scalar::F(1.0))
                        .as_f64()
                        .is_nan()
                );
            }
        }
        for (kind, identity) in [
            (ReduceKind::Max, f64::NEG_INFINITY),
            (ReduceKind::Min, f64::INFINITY),
        ] {
            let e5m2 = plan(kind, DType::F8E5M2, DType::F8E5M2, 2);
            assert_eq!(e5m2.identity().as_f64(), identity);
            assert_eq!(e5m2.update(e5m2.identity(), Scalar::F(1.0)).as_f64(), 1.0);
        }
    }

    #[test]
    fn mean_commits_the_divisor_and_divides_at_float_work_width() {
        let mean = plan(ReduceKind::Mean, DType::F32, DType::F32, 3);
        let accumulator = [1.0, 2.0, 4.0]
            .into_iter()
            .fold(mean.identity(), |acc, value| {
                mean.update(acc, Scalar::F(value))
            });
        assert_eq!(
            mean.finalize(accumulator).as_f64(),
            f64::from(7.0f32 / 3.0f32)
        );

        let rounded_count = plan(ReduceKind::Mean, DType::F32, DType::F32, (1 << 24) + 1);
        assert_eq!(rounded_count.mean_divisor().unwrap().as_f64(), 16_777_216.0);
        assert_eq!(
            mean_quotient(Scalar::F(16_777_216.0), (1 << 24) + 1, DType::F32),
            Some(Scalar::F(1.0))
        );
    }

    #[test]
    fn malformed_geometry_and_dtype_fail_closed_while_empty_extrema_use_identity() {
        assert!(
            NativeReductionPlan::new(
                Shape::from([2, 3]),
                Shape::from([3]),
                vec![1],
                false,
                ReduceKind::Sum,
                DType::F32,
                ReductionDType::new(DType::F32, DType::F32),
            )
            .is_err()
        );
        assert!(
            NativeReductionPlan::new(
                Shape::from([2]),
                Shape::new([]),
                vec![0],
                false,
                ReduceKind::Sum,
                DType::F32,
                ReductionDType::new(DType::F64, DType::F64),
            )
            .is_err()
        );
        assert!(
            NativeReductionPlan::new(
                Shape::from([2]),
                Shape::new([]),
                vec![0],
                false,
                ReduceKind::Sum,
                DType::Bool,
                ReductionDType::new(DType::Bool, DType::Bool),
            )
            .is_ok()
        );
        assert!(
            NativeReductionPlan::new(
                Shape::from([2]),
                Shape::new([]),
                vec![0],
                false,
                ReduceKind::Mean,
                DType::F8E5M2,
                ReductionDType::new(DType::F8E5M2, DType::F32),
            )
            .is_err()
        );
        let empty_max = NativeReductionPlan::new(
            Shape::from([0]),
            Shape::new([]),
            vec![0],
            false,
            ReduceKind::Max,
            DType::I32,
            ReductionDType::new(DType::I32, DType::I32),
        )
        .unwrap();
        assert_eq!(empty_max.reduction_len(), 0);
        assert_eq!(
            empty_max.finalize(empty_max.identity()),
            Scalar::I(i32::MIN.into())
        );
        assert!(
            NativeReductionPlan::new(
                Shape::from([0, 0]),
                Shape::from([0]),
                vec![1],
                false,
                ReduceKind::Max,
                DType::I32,
                ReductionDType::new(DType::I32, DType::I32),
            )
            .is_ok()
        );
        assert!(
            NativeReductionPlan::new(
                Shape::from([0, usize::MAX, 2]),
                Shape::from([0]),
                vec![1, 2],
                false,
                ReduceKind::Max,
                DType::I32,
                ReductionDType::new(DType::I32, DType::I32),
            )
            .is_err()
        );
    }
}
