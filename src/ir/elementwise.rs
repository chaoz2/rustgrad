use super::*;
use crate::{DType, Error, Result, TensorData};

impl Graph {
    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Add, lhs, rhs)
    }

    pub fn sub(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Sub, lhs, rhs)
    }

    pub fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Mul, lhs, rhs)
    }

    pub fn div(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Div, lhs, rhs)
    }
    pub fn pow(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Pow, lhs, rhs)
    }
    pub fn maximum(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Maximum, lhs, rhs)
    }
    pub fn minimum(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Minimum, lhs, rhs)
    }
    pub fn floor_div(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::FloorDiv, lhs, rhs)
    }
    pub fn trunc_div(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::TruncDiv, lhs, rhs)
    }
    pub fn modulo(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Mod, lhs, rhs)
    }
    pub fn fmod(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::FMod, lhs, rhs)
    }
    pub fn bit_and(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::BitAnd, lhs, rhs)
    }
    pub fn bit_or(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::BitOr, lhs, rhs)
    }
    pub fn bit_xor(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::BitXor, lhs, rhs)
    }
    pub fn shl(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Shl, lhs, rhs)
    }
    pub fn shr(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Shr, lhs, rhs)
    }

    pub fn eq(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.compare(CompareOp::Eq, lhs, rhs)
    }
    pub fn ne(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.compare(CompareOp::Ne, lhs, rhs)
    }
    pub fn lt(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.compare(CompareOp::Lt, lhs, rhs)
    }
    pub fn le(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.compare(CompareOp::Le, lhs, rhs)
    }
    pub fn gt(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.compare(CompareOp::Gt, lhs, rhs)
    }
    pub fn ge(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.compare(CompareOp::Ge, lhs, rhs)
    }

    pub fn logical_not(&mut self, input: NodeId) -> Result<NodeId> {
        self.require_bool(input, "logical_not")?;
        let shape = self.node(input)?.shape.clone();
        Ok(self.push(
            Op::Logical {
                op: LogicalOp::Not,
                lhs: input,
                rhs: None,
            },
            shape,
            DType::Bool,
        ))
    }

    pub fn logical_and(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.logical_binary(LogicalOp::And, lhs, rhs)
    }
    pub fn logical_or(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.logical_binary(LogicalOp::Or, lhs, rhs)
    }

    /// Selects `on_true` where `condition` is true and `on_false` otherwise.
    /// The condition must be boolean; both value branches are promoted.
    pub fn select(
        &mut self,
        condition: NodeId,
        on_true: NodeId,
        on_false: NodeId,
    ) -> Result<NodeId> {
        self.require_bool(condition, "select")?;
        let value_shape = self.broadcast_shape(on_true, on_false)?;
        let shape = self.node(condition)?.shape.broadcast_with(&value_shape)?;
        let dtype = self
            .node(on_true)?
            .dtype
            .promote(self.node(on_false)?.dtype);
        Ok(self.push(
            Op::Select {
                condition,
                on_true,
                on_false,
            },
            shape,
            dtype,
        ))
    }

    /// Replaces `input` with `value` wherever the boolean `mask` is true.
    ///
    /// This is the named counterpart to tinygrad's `masked_fill`; it retains
    /// `select`'s checked broadcasting and value-dtype promotion contract.
    pub fn masked_fill(
        &mut self,
        input: NodeId,
        mask: NodeId,
        value: NodeId,
    ) -> Result<NodeId> {
        self.select(mask, value, input)
    }

    pub fn neg(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Neg, input)
    }
    pub fn exp(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Exp, input)
    }
    pub fn log(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Log, input)
    }
    pub fn abs(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Abs, input)
    }
    pub fn reciprocal(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Reciprocal, input)
    }
    pub fn square(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Square, input)
    }
    pub fn sqrt(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Sqrt, input)
    }
    pub fn rsqrt(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Rsqrt, input)
    }
    pub fn exp2(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Exp2, input)
    }
    pub fn log2(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Log2, input)
    }
    pub fn sin(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Sin, input)
    }
    pub fn cos(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Cos, input)
    }
    pub fn tan(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Tan, input)
    }
    pub fn sinh(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Sinh, input)
    }
    pub fn cosh(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Cosh, input)
    }
    pub fn tanh(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Tanh, input)
    }
    /// Applies the Gauss error function elementwise.
    pub fn erf(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Erf, input)
    }
    /// Applies the complementary Gauss error function elementwise.
    pub fn erfc(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Erfc, input)
    }
    pub fn asin(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Asin, input)
    }
    pub fn acos(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Acos, input)
    }
    pub fn atan(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Atan, input)
    }
    pub fn asinh(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Asinh, input)
    }
    pub fn acosh(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Acosh, input)
    }
    pub fn atanh(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Atanh, input)
    }
    /// Returns the quadrant-aware angle of `(y, x)` elementwise.
    pub fn atan2(&mut self, y: NodeId, x: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Atan2, y, x)
    }
    /// Returns the magnitude of `magnitude` with the sign selected by `sign`.
    pub fn copysign(&mut self, magnitude: NodeId, sign: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Copysign, magnitude, sign)
    }
    /// Compositional tinygrad-style sigmoid, retaining an inspectable graph.
    pub fn sigmoid(&mut self, input: NodeId) -> Result<NodeId> {
        let one = self.constant(TensorData::scalar(1.0f32));
        let neg = self.neg(input)?;
        let exp = self.exp(neg)?;
        let denominator = self.add(one, exp)?;
        let numerator = self.constant(TensorData::scalar(1.0f32));
        self.div(numerator, denominator)
    }
    pub fn clamp(
        &mut self,
        input: NodeId,
        min: Option<NodeId>,
        max: Option<NodeId>,
    ) -> Result<NodeId> {
        if min.is_none() && max.is_none() {
            return Err(Error::InvalidElementwiseDType {
                op: "clamp requires a bound",
                actual: self.node(input)?.dtype,
            });
        }
        // Validate every composition edge before constructing either half of
        // the clamp. In particular, an invalid upper bound must not leave a
        // valid lower-bound node in the graph.
        if let Some(min) = min {
            self.broadcast_shape(input, min)?;
        }
        if let Some(max) = max {
            self.broadcast_shape(input, max)?;
        }
        if let (Some(min), Some(max)) = (min, max) {
            self.broadcast_shape(min, max)?;
        }
        let mut value = input;
        if let Some(min) = min {
            value = self.maximum(value, min)?;
        }
        if let Some(max) = max {
            value = self.minimum(value, max)?;
        }
        Ok(value)
    }

    /// Alias for [`Self::clamp`], matching tinygrad's public `clip` helper.
    pub fn clip(
        &mut self,
        input: NodeId,
        min: Option<NodeId>,
        max: Option<NodeId>,
    ) -> Result<NodeId> {
        self.clamp(input, min, max)
    }
    pub fn relu6(&mut self, input: NodeId) -> Result<NodeId> {
        let zero = self.constant(TensorData::scalar(0.0f32));
        let six = self.constant(TensorData::scalar(6.0f32));
        self.clamp(input, Some(zero), Some(six))
    }
    pub fn leaky_relu(&mut self, input: NodeId, slope: NodeId) -> Result<NodeId> {
        // Validate the parameter broadcast before the constants and predicate
        // below make this composite visible in the graph.
        self.broadcast_shape(input, slope)?;
        let zero = self.constant(TensorData::scalar(0.0f32));
        let negative = self.lt(input, zero)?;
        let scaled = self.mul(input, slope)?;
        self.select(negative, scaled, input)
    }
    pub fn silu(&mut self, input: NodeId) -> Result<NodeId> {
        let sigmoid = self.sigmoid(input)?;
        self.mul(input, sigmoid)
    }
    pub fn hardsigmoid(&mut self, input: NodeId) -> Result<NodeId> {
        let three = self.constant(TensorData::scalar(3.0f32));
        let six = self.constant(TensorData::scalar(6.0f32));
        let shifted = self.add(input, three)?;
        let zero = self.constant(TensorData::scalar(0.0f32));
        let clipped = self.clamp(shifted, Some(zero), Some(six))?;
        let divisor = self.constant(TensorData::scalar(6.0f32));
        self.div(clipped, divisor)
    }
    pub fn hardtanh(&mut self, input: NodeId, min: NodeId, max: NodeId) -> Result<NodeId> {
        self.clamp(input, Some(min), Some(max))
    }
    pub fn swish(&mut self, input: NodeId) -> Result<NodeId> {
        self.silu(input)
    }
    pub fn hardswish(&mut self, input: NodeId) -> Result<NodeId> {
        let three = self.constant(TensorData::scalar(3.0f32));
        let six = self.constant(TensorData::scalar(6.0f32));
        let zero = self.constant(TensorData::scalar(0.0f32));
        let shifted = self.add(input, three)?;
        let clipped = self.clamp(shifted, Some(zero), Some(six))?;
        let scaled = self.mul(input, clipped)?;
        let divisor = self.constant(TensorData::scalar(6.0f32));
        self.div(scaled, divisor)
    }
    pub fn quick_gelu(&mut self, input: NodeId) -> Result<NodeId> {
        let scale = self.constant(TensorData::scalar(1.702f32));
        let scaled = self.mul(input, scale)?;
        let sigmoid = self.sigmoid(scaled)?;
        self.mul(input, sigmoid)
    }
    /// Applies GELU using tinygrad's `"tanh"` approximation or the exact
    /// error-function form selected by `"none"`.
    pub fn gelu(&mut self, input: NodeId, approximate: &str) -> Result<NodeId> {
        match approximate {
            "tanh" => {
                let half = self.constant(TensorData::scalar(0.5f32));
                let one = self.constant(TensorData::scalar(1.0f32));
                let scale =
                    self.constant(TensorData::scalar((2.0f32 / std::f32::consts::PI).sqrt()));
                let coefficient = self.constant(TensorData::scalar(0.044_715f32));
                let square = self.square(input)?;
                let cube = self.mul(square, input)?;
                let scaled_cube = self.mul(coefficient, cube)?;
                let inner = self.add(input, scaled_cube)?;
                let scaled = self.mul(scale, inner)?;
                let tanh = self.tanh(scaled)?;
                let left = self.mul(half, input)?;
                let right = self.add(one, tanh)?;
                self.mul(left, right)
            }
            "none" => {
                let half = self.constant(TensorData::scalar(0.5f32));
                let one = self.constant(TensorData::scalar(1.0f32));
                let root_two = self.constant(TensorData::scalar(std::f32::consts::SQRT_2));
                let scaled = self.div(input, root_two)?;
                let erf = self.erf(scaled)?;
                let left = self.mul(input, half)?;
                let right = self.add(one, erf)?;
                self.mul(left, right)
            }
            _ => Err(Error::InvalidElementwiseDType {
                op: "gelu approximate must be `tanh` or `none`",
                actual: self.node(input)?.dtype,
            }),
        }
    }
    pub fn elu(&mut self, input: NodeId, alpha: NodeId) -> Result<NodeId> {
        // `alpha` participates in the negative branch only, but its shape is
        // part of the result ABI. Preflight it before constructing that branch.
        self.broadcast_shape(input, alpha)?;
        let zero = self.constant(TensorData::scalar(0.0f32));
        let positive = self.gt(input, zero)?;
        let exp = self.exp(input)?;
        let one = self.constant(TensorData::scalar(1.0f32));
        let delta = self.sub(exp, one)?;
        let negative = self.mul(alpha, delta)?;
        self.select(positive, input, negative)
    }
    pub fn celu(&mut self, input: NodeId, alpha: NodeId) -> Result<NodeId> {
        self.broadcast_shape(input, alpha)?;
        let zero = self.constant(TensorData::scalar(0.0f32));
        let positive = self.maximum(input, zero)?;
        let scaled = self.div(input, alpha)?;
        let exp = self.exp(scaled)?;
        let one = self.constant(TensorData::scalar(1.0f32));
        let delta = self.sub(exp, one)?;
        let scaled_negative = self.mul(alpha, delta)?;
        let negative = self.minimum(scaled_negative, zero)?;
        self.add(positive, negative)
    }
    pub fn selu(&mut self, input: NodeId, alpha: NodeId, gamma: NodeId) -> Result<NodeId> {
        let elu_shape = self.broadcast_shape(input, alpha)?;
        elu_shape.broadcast_with(&self.node(gamma)?.shape)?;
        let elu = self.elu(input, alpha)?;
        self.mul(gamma, elu)
    }
    pub fn softplus(&mut self, input: NodeId, beta: NodeId) -> Result<NodeId> {
        let scaled = self.mul(input, beta)?;
        let exp = self.exp(scaled)?;
        let one = self.constant(TensorData::scalar(1.0f32));
        let sum = self.add(one, exp)?;
        let logged = self.log(sum)?;
        self.div(logged, beta)
    }
    pub fn mish(&mut self, input: NodeId) -> Result<NodeId> {
        let one = self.constant(TensorData::scalar(1.0f32));
        let exp = self.exp(input)?;
        let sum = self.add(one, exp)?;
        let softplus = self.log(sum)?;
        let tanh = self.tanh(softplus)?;
        self.mul(input, tanh)
    }
    pub fn logsigmoid(&mut self, input: NodeId) -> Result<NodeId> {
        let neg = self.neg(input)?;
        let one = self.constant(TensorData::scalar(1.0f32));
        let exp = self.exp(neg)?;
        let sum = self.add(one, exp)?;
        let log = self.log(sum)?;
        self.neg(log)
    }
    pub fn softsign(&mut self, input: NodeId) -> Result<NodeId> {
        let one = self.constant(TensorData::scalar(1.0f32));
        let abs = self.abs(input)?;
        let denominator = self.add(one, abs)?;
        self.div(input, denominator)
    }
    pub fn log10(&mut self, input: NodeId) -> Result<NodeId> {
        let log = self.log2(input)?;
        let scale = self.constant(TensorData::scalar(std::f32::consts::LOG10_2));
        self.mul(log, scale)
    }
    pub fn logaddexp(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let maximum = self.maximum(lhs, rhs)?;
        let left = self.sub(lhs, maximum)?;
        let right = self.sub(rhs, maximum)?;
        let left_exp = self.exp(left)?;
        let right_exp = self.exp(right)?;
        let sum = self.add(left_exp, right_exp)?;
        let log = self.log(sum)?;
        self.add(log, maximum)
    }
    pub fn logaddexp2(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let maximum = self.maximum(lhs, rhs)?;
        let left = self.sub(lhs, maximum)?;
        let right = self.sub(rhs, maximum)?;
        let left_exp = self.exp2(left)?;
        let right_exp = self.exp2(right)?;
        let sum = self.add(left_exp, right_exp)?;
        let log = self.log2(sum)?;
        self.add(log, maximum)
    }
    pub fn lerp(&mut self, start: NodeId, end: NodeId, weight: NodeId) -> Result<NodeId> {
        let start_end_shape = self.broadcast_shape(start, end)?;
        start_end_shape.broadcast_with(&self.node(weight)?.shape)?;
        let delta = self.sub(end, start)?;
        let weighted = self.mul(delta, weight)?;
        self.add(start, weighted)
    }
    pub fn isclose(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        rtol: NodeId,
        atol: NodeId,
        equal_nan: bool,
    ) -> Result<NodeId> {
        let raw_difference = self.sub(lhs, rhs)?;
        let difference = self.abs(raw_difference)?;
        let abs_rhs = self.abs(rhs)?;
        let relative = self.mul(rtol, abs_rhs)?;
        let tolerance = self.add(atol, relative)?;
        let lhs_finite = self.isfinite(lhs)?;
        let rhs_finite = self.isfinite(rhs)?;
        let finite = self.logical_and(lhs_finite, rhs_finite)?;
        let near = self.le(difference, tolerance)?;
        let finite_near = self.logical_and(finite, near)?;
        let lhs_inf = self.isinf(lhs)?;
        let rhs_inf = self.isinf(rhs)?;
        let infinities = self.logical_or(lhs_inf, rhs_inf)?;
        let equal = self.eq(lhs, rhs)?;
        let same_infinity = self.logical_and(infinities, equal)?;
        let result = self.logical_or(finite_near, same_infinity)?;
        if equal_nan {
            let lhs_nan = self.isnan(lhs)?;
            let rhs_nan = self.isnan(rhs)?;
            let both_nan = self.logical_and(lhs_nan, rhs_nan)?;
            self.logical_or(result, both_nan)
        } else {
            Ok(result)
        }
    }
    pub fn floor(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Floor, input)
    }
    pub fn ceil(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Ceil, input)
    }
    pub fn trunc(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Trunc, input)
    }
    pub fn round(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Round, input)
    }
    pub fn sign(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Sign, input)
    }
    pub fn isnan(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::IsNan, input)
    }
    pub fn isinf(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::IsInf, input)
    }
    /// Returns whether `input` is positive and/or negative infinity.
    ///
    /// This preserves tinygrad's `isinf(detect_positive, detect_negative)`
    /// selection contract while retaining [`Self::isinf`] as the default
    /// both-sign path. The selected-sign comparisons remain ordinary boolean
    /// graph operations, so NaNs and both signed zeroes are false.
    pub fn isinf_with_signs(
        &mut self,
        input: NodeId,
        detect_positive: bool,
        detect_negative: bool,
    ) -> Result<NodeId> {
        let dtype = self.node(input)?.dtype;
        if detect_positive && detect_negative {
            return self.isinf(input);
        }
        if !detect_positive && !detect_negative {
            let none = self.isinf(input)?;
            return self.logical_and(none, none);
        }
        if !dtype.is_float() {
            return self.isinf(input);
        }
        let infinity = if detect_positive {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };
        let bound = self.constant(TensorData::scalar(infinity));
        self.eq(input, bound)
    }
    pub fn isfinite(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::IsFinite, input)
    }
    pub fn relu(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Relu, input)
    }
    pub(crate) fn step(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Step, input)
    }

    pub fn unary(&mut self, op: UnaryOp, input: NodeId) -> Result<NodeId> {
        let source = self.node(input)?;
        let dtype = unary_dtype(op, source.dtype);
        Ok(self.push(Op::Unary { op, input }, source.shape.clone(), dtype))
    }

    pub fn binary(&mut self, op: BinaryOp, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let shape = self.broadcast_shape(lhs, rhs)?;
        let promoted = self.node(lhs)?.dtype.promote(self.node(rhs)?.dtype);
        // As with unary transcendental helpers, atan2 lifts exact storage to
        // the default floating dtype rather than performing integer math.
        let dtype = if op == BinaryOp::Atan2 && !promoted.is_float() {
            DType::F32
        } else {
            promoted
        };
        if matches!(op, BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor) && dtype.is_float() {
            return Err(Error::InvalidElementwiseDType {
                op: op.name(),
                actual: dtype,
            });
        }
        if matches!(op, BinaryOp::Shl | BinaryOp::Shr) && !dtype.is_integer() {
            return Err(Error::InvalidElementwiseDType {
                op: op.name(),
                actual: dtype,
            });
        }
        Ok(self.push(Op::Binary { op, lhs, rhs }, shape, dtype))
    }

    pub fn compare(&mut self, op: CompareOp, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let shape = self.broadcast_shape(lhs, rhs)?;
        Ok(self.push(Op::Compare { op, lhs, rhs }, shape, DType::Bool))
    }

    fn logical_binary(&mut self, op: LogicalOp, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.require_bool(lhs, op.name())?;
        self.require_bool(rhs, op.name())?;
        let shape = self.broadcast_shape(lhs, rhs)?;
        Ok(self.push(
            Op::Logical {
                op,
                lhs,
                rhs: Some(rhs),
            },
            shape,
            DType::Bool,
        ))
    }

    fn require_bool(&self, input: NodeId, op: &'static str) -> Result<()> {
        let actual = self.node(input)?.dtype;
        if actual == DType::Bool {
            Ok(())
        } else {
            Err(Error::InvalidLogicalDType { op, actual })
        }
    }

    pub fn cast(&mut self, input: NodeId, dtype: DType) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        Ok(self.push(Op::Cast { input, dtype }, shape, dtype))
    }
}
