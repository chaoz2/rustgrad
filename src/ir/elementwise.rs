use super::*;
use crate::{DType, Error, Result, TensorData};

struct HardsigmoidPlan {
    product_shape: Shape,
    product_dtype: DType,
    output_shape: Shape,
    output_dtype: DType,
    zero: TensorData,
    one: TensorData,
}

struct LeakyReluPlan {
    shape: Shape,
    dtype: DType,
    zero: TensorData,
}

struct CeluPlan {
    division_dtype: DType,
    dividend_dtype: DType,
    reciprocal_dtype: DType,
    scaled_shape: Shape,
    scaled_dtype: DType,
    exp_dtype: DType,
    negative_shape: Shape,
    negative_dtype: DType,
    output_shape: Shape,
    output_dtype: DType,
    input_zero: TensorData,
    one: TensorData,
    negative_zero: TensorData,
}

struct SwishPlan {
    shape: Shape,
    dtype: DType,
}

struct QuickGeluPlan {
    shape: Shape,
    dtype: DType,
    scale: TensorData,
    one: TensorData,
    neg_inv_ln2: TensorData,
}

struct MishPlan {
    shape: Shape,
    dtype: DType,
    beta: TensorData,
}

struct HardswishPlan {
    shape: Shape,
    dtype: DType,
    zero: TensorData,
    three: TensorData,
    six: TensorData,
    sixth: TensorData,
}

struct Relu6Plan {
    shape: Shape,
    dtype: DType,
    zero: TensorData,
    six: TensorData,
}

fn relu6_plan(input_shape: &Shape, input_dtype: DType) -> Result<Relu6Plan> {
    // Tensor.relu6 is exactly `relu(x) - relu(x - 6)`, and both ReLUs are
    // strict ordered selects at the source storage width.
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(input_shape, input_dtype)?;
    for _ in ["first relu", "shifted", "second relu", "output"] {
        extent(input_shape, input_dtype)?;
    }
    extent(input_shape, DType::Bool)?;
    extent(input_shape, DType::Bool)?;
    let zero = TensorData::scalar_with_dtype(Scalar::I(0), input_dtype);
    let six = TensorData::scalar_with_dtype(Scalar::I(6), input_dtype);
    if zero.dtype() != input_dtype
        || six.dtype() != input_dtype
        || input_shape.broadcast_with(zero.shape())? != *input_shape
        || input_shape.broadcast_with(six.shape())? != *input_shape
        || input_dtype.promote(zero.dtype()) != input_dtype
        || input_dtype.promote(six.dtype()) != input_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "relu6 scalar promotion",
            actual: input_dtype,
        });
    }
    Ok(Relu6Plan {
        shape: input_shape.clone(),
        dtype: input_dtype,
        zero,
        six,
    })
}

fn hardswish_plan(input_shape: &Shape, input_dtype: DType) -> Result<HardswishPlan> {
    // tinygrad spells Hardswish as `x * (x + 3).relu6() * (1/6)`, where
    // relu6 is itself `relu(y) - relu(y - 6)` with strict ReLU selects.
    let dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(input_shape, input_dtype)?;
    for _ in [
        "cast",
        "shifted",
        "positive condition",
        "positive relu",
        "shifted minus six",
        "upper condition",
        "upper relu",
        "relu6",
        "outer product",
        "output",
    ] {
        extent(input_shape, dtype)?;
    }
    // Conditions are Bool rather than source storage width.
    extent(input_shape, DType::Bool)?;
    let zero = TensorData::scalar_with_dtype(Scalar::I(0), dtype);
    let three = TensorData::scalar_with_dtype(Scalar::I(3), dtype);
    let six = TensorData::scalar_with_dtype(Scalar::I(6), dtype);
    let sixth = TensorData::scalar_with_dtype(Scalar::F(1.0 / 6.0), dtype);
    if zero.dtype() != dtype
        || three.dtype() != dtype
        || six.dtype() != dtype
        || sixth.dtype() != dtype
        || input_shape.broadcast_with(zero.shape())? != *input_shape
        || input_shape.broadcast_with(three.shape())? != *input_shape
        || input_shape.broadcast_with(six.shape())? != *input_shape
        || input_shape.broadcast_with(sixth.shape())? != *input_shape
        || input_dtype.promote(dtype) != dtype
        || dtype.promote(zero.dtype()) != dtype
        || dtype.promote(three.dtype()) != dtype
        || dtype.promote(six.dtype()) != dtype
        || dtype.promote(sixth.dtype()) != dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "hardswish scalar promotion",
            actual: dtype,
        });
    }
    Ok(HardswishPlan {
        shape: input_shape.clone(),
        dtype,
        zero,
        three,
        six,
        sixth,
    })
}

fn mish_plan(input_shape: &Shape, input_dtype: DType) -> Result<MishPlan> {
    // Tensor.mish is `x * x.softplus().tanh()`, where softplus's default
    // Python `1.0` is weak at the input floating storage width (or F32 for
    // exact input storage).  Plan all three composites before publishing that
    // otherwise-visible default constant.
    let dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(input_shape, input_dtype)?;
    // Softplus's cast/scale/stable-logaddexp/inverse/output stages, Tanh's
    // typed sigmoid expansion, and the final outer multiply all retain this
    // one output storage width.
    for _ in [
        "softplus cast",
        "softplus scale",
        "softplus maximum",
        "softplus exponentials",
        "softplus logarithm",
        "softplus inverse",
        "softplus output",
        "tanh inner multiply",
        "tanh exponent",
        "tanh exp2",
        "tanh denominator",
        "tanh reciprocal",
        "tanh output",
        "mish output",
    ] {
        extent(input_shape, dtype)?;
    }
    let beta = TensorData::scalar_with_dtype(Scalar::F(1.0), dtype);
    let zero = TensorData::scalar_with_dtype(Scalar::F(0.0), dtype);
    let one = TensorData::scalar_with_dtype(Scalar::F(1.0), dtype);
    let two = TensorData::scalar_with_dtype(Scalar::F(2.0), dtype);
    let neg_inv_ln2 = TensorData::scalar_with_dtype(
        Scalar::F(-1.0 / std::f64::consts::LN_2),
        dtype,
    );
    if beta.dtype() != dtype
        || zero.dtype() != dtype
        || one.dtype() != dtype
        || two.dtype() != dtype
        || neg_inv_ln2.dtype() != dtype
        || input_shape.broadcast_with(beta.shape())? != *input_shape
        || input_shape.broadcast_with(zero.shape())? != *input_shape
        || input_shape.broadcast_with(one.shape())? != *input_shape
        || input_shape.broadcast_with(two.shape())? != *input_shape
        || input_shape.broadcast_with(neg_inv_ln2.shape())? != *input_shape
        || input_dtype.promote(dtype) != dtype
        || dtype.promote(beta.dtype()) != dtype
        || dtype.promote(zero.dtype()) != dtype
        || dtype.promote(one.dtype()) != dtype
        || dtype.promote(two.dtype()) != dtype
        || dtype.promote(neg_inv_ln2.dtype()) != dtype
        || unary_dtype(UnaryOp::Exp, dtype) != dtype
        || unary_dtype(UnaryOp::Log, dtype) != dtype
        || unary_dtype(UnaryOp::Reciprocal, dtype) != dtype
        || unary_dtype(UnaryOp::Exp2, dtype) != dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "mish scalar promotion",
            actual: dtype,
        });
    }
    Ok(MishPlan {
        shape: input_shape.clone(),
        dtype,
        beta,
    })
}

fn swish_plan(input_shape: &Shape, input_dtype: DType) -> Result<SwishPlan> {
    // `swish` is literally `x * sigmoid(x)`.  Mirror the source-width
    // sigmoid descriptor here so a late outer multiply can never publish a
    // partial sigmoid graph.
    let dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(input_shape, input_dtype)?;
    for _ in ["cast", "scaled", "exp2", "denominator", "reciprocal", "output"] {
        extent(input_shape, dtype)?;
    }
    let one = TensorData::scalar_with_dtype(Scalar::F(1.0), dtype);
    let neg_inv_ln2 = TensorData::scalar_with_dtype(
        Scalar::F(-1.0 / std::f64::consts::LN_2),
        dtype,
    );
    if one.dtype() != dtype
        || neg_inv_ln2.dtype() != dtype
        || input_shape.broadcast_with(one.shape())? != *input_shape
        || input_shape.broadcast_with(neg_inv_ln2.shape())? != *input_shape
        || input_dtype.promote(dtype) != dtype
        || dtype.promote(one.dtype()) != dtype
        || dtype.promote(neg_inv_ln2.dtype()) != dtype
        || unary_dtype(UnaryOp::Exp2, dtype) != dtype
        || unary_dtype(UnaryOp::Reciprocal, dtype) != dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "swish scalar promotion",
            actual: dtype,
        });
    }
    Ok(SwishPlan {
        shape: input_shape.clone(),
        dtype,
    })
}

fn quick_gelu_plan(input_shape: &Shape, input_dtype: DType) -> Result<QuickGeluPlan> {
    // Tensor.quick_gelu is `x * (x * 1.702).sigmoid()`. The Python literal is
    // weak: it has the input floating storage width, or F32 for non-floats.
    // Plan the expanded typed sigmoid too, so no constant or half-built graph
    // can escape if a descriptor/extent fact is invalid.
    let dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    let extent = |shape: &Shape, dtype: DType| {
        shape.numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(input_shape, input_dtype)?;
    for _ in [
        "cast",
        "scaled",
        "exponent",
        "exp2",
        "denominator",
        "reciprocal",
        "output",
    ] {
        extent(input_shape, dtype)?;
    }
    let scale = TensorData::scalar_with_dtype(Scalar::F(1.702), dtype);
    let one = TensorData::scalar_with_dtype(Scalar::F(1.0), dtype);
    let neg_inv_ln2 = TensorData::scalar_with_dtype(
        Scalar::F(-1.0 / std::f64::consts::LN_2),
        dtype,
    );
    if scale.dtype() != dtype
        || one.dtype() != dtype
        || neg_inv_ln2.dtype() != dtype
        || input_shape.broadcast_with(scale.shape())? != *input_shape
        || input_shape.broadcast_with(one.shape())? != *input_shape
        || input_shape.broadcast_with(neg_inv_ln2.shape())? != *input_shape
        || input_dtype.promote(dtype) != dtype
        || dtype.promote(scale.dtype()) != dtype
        || dtype.promote(one.dtype()) != dtype
        || dtype.promote(neg_inv_ln2.dtype()) != dtype
        || unary_dtype(UnaryOp::Exp2, dtype) != dtype
        || unary_dtype(UnaryOp::Reciprocal, dtype) != dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "quick_gelu scalar promotion",
            actual: dtype,
        });
    }
    Ok(QuickGeluPlan {
        shape: input_shape.clone(),
        dtype,
        scale,
        one,
        neg_inv_ln2,
    })
}

fn celu_plan(
    input_shape: &Shape,
    input_dtype: DType,
    alpha_shape: &Shape,
    alpha_dtype: DType,
) -> Result<CeluPlan> {
    // The only locally represented difference between RustGrad's ordinary
    // lattice and tinygrad's weak source lattice is its I64/U64 default-F32
    // bridge.  CELU's reciprocal normally moves later arithmetic to float,
    // but retaining the source rule in the plan closes every stage.
    let source_promote = |lhs: DType, rhs: DType| {
        if matches!((lhs, rhs), (DType::I64, DType::U64) | (DType::U64, DType::I64)) {
            DType::F32
        } else {
            lhs.promote(rhs)
        }
    };
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(input_shape, input_dtype)?;
    extent(alpha_shape, alpha_dtype)?;
    let scaled_shape = input_shape.broadcast_with(alpha_shape)?;
    // Tensor.div first commits both operands to their common source dtype.
    // It then lifts only an integer/bool dividend to F32 before multiplying
    // by the reciprocal.  Taking the reciprocal of the original alpha would
    // incorrectly widen e.g. F16 x / I32 alpha to F32.
    let division_dtype = source_promote(input_dtype, alpha_dtype);
    let dividend_dtype = if division_dtype.is_float() {
        division_dtype
    } else {
        DType::F32
    };
    let reciprocal_dtype = unary_dtype(UnaryOp::Reciprocal, division_dtype);
    let scaled_dtype = source_promote(dividend_dtype, reciprocal_dtype);
    let exp_dtype = unary_dtype(UnaryOp::Exp, scaled_dtype);
    let negative_shape = scaled_shape.broadcast_with(alpha_shape)?;
    let negative_dtype = source_promote(alpha_dtype, exp_dtype);
    let output_shape = input_shape.broadcast_with(&negative_shape)?;
    let output_dtype = source_promote(input_dtype, negative_dtype);
    for (shape, dtype) in [
        (input_shape, division_dtype),
        (alpha_shape, division_dtype),
        (input_shape, dividend_dtype),
        (alpha_shape, reciprocal_dtype),
        (&scaled_shape, scaled_dtype),
        (&scaled_shape, exp_dtype),
        (&negative_shape, negative_dtype),
        (input_shape, input_dtype),
        (&output_shape, output_dtype),
    ] {
        extent(shape, dtype)?;
    }
    let input_zero = TensorData::scalar_with_dtype(Scalar::I(0), input_dtype);
    let one = TensorData::scalar_with_dtype(Scalar::I(1), exp_dtype);
    let negative_zero = TensorData::scalar_with_dtype(Scalar::I(0), negative_dtype);
    if input_zero.dtype() != input_dtype
        || one.dtype() != exp_dtype
        || negative_zero.dtype() != negative_dtype
        || input_shape.broadcast_with(input_zero.shape())? != *input_shape
        || scaled_shape.broadcast_with(one.shape())? != scaled_shape
        || negative_shape.broadcast_with(negative_zero.shape())? != negative_shape
        || source_promote(input_dtype, alpha_dtype) != division_dtype
        || source_promote(dividend_dtype, reciprocal_dtype) != scaled_dtype
        || source_promote(alpha_dtype, exp_dtype) != negative_dtype
        || source_promote(input_dtype, negative_dtype) != output_dtype
        || dividend_dtype.promote(reciprocal_dtype) != scaled_dtype
        || alpha_dtype.promote(exp_dtype) != negative_dtype
        || input_dtype.promote(negative_dtype) != output_dtype
        || unary_dtype(UnaryOp::Reciprocal, division_dtype) != reciprocal_dtype
        || unary_dtype(UnaryOp::Exp, scaled_dtype) != exp_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "celu scalar promotion",
            actual: output_dtype,
        });
    }
    Ok(CeluPlan {
        division_dtype,
        dividend_dtype,
        reciprocal_dtype,
        scaled_shape,
        scaled_dtype,
        exp_dtype,
        negative_shape,
        negative_dtype,
        output_shape,
        output_dtype,
        input_zero,
        one,
        negative_zero,
    })
}

fn leaky_relu_plan(
    input_shape: &Shape,
    input_dtype: DType,
    slope_shape: &Shape,
    slope_dtype: DType,
) -> Result<LeakyReluPlan> {
    let source_promote = |lhs: DType, rhs: DType| {
        if matches!((lhs, rhs), (DType::I64, DType::U64) | (DType::U64, DType::I64)) {
            DType::F32
        } else {
            lhs.promote(rhs)
        }
    };
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(input_shape, input_dtype)?;
    extent(slope_shape, slope_dtype)?;
    let shape = input_shape.broadcast_with(slope_shape)?;
    let dtype = source_promote(input_dtype, slope_dtype);
    for (shape, dtype) in [
        (input_shape, dtype),
        (slope_shape, dtype),
        (input_shape, DType::Bool),
        (&shape, dtype),
        (&shape, dtype),
    ] {
        extent(shape, dtype)?;
    }
    let zero = TensorData::scalar_with_dtype(Scalar::I(0), input_dtype);
    if zero.dtype() != input_dtype
        || input_shape.broadcast_with(zero.shape())? != *input_shape
        || input_shape.broadcast_with(&shape)? != shape
        || slope_shape.broadcast_with(&shape)? != shape
        || source_promote(input_dtype, slope_dtype) != dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "leaky_relu scalar promotion",
            actual: dtype,
        });
    }
    Ok(LeakyReluPlan { shape, dtype, zero })
}

fn hardsigmoid_plan(
    input_shape: &Shape,
    input_dtype: DType,
    alpha_shape: &Shape,
    alpha_dtype: DType,
    beta_shape: &Shape,
    beta_dtype: DType,
) -> Result<HardsigmoidPlan> {
    // tinygrad's weak promotion has one local disagreement with RustGrad's
    // generic lattice: the I64/U64 pair resolves through default F32.
    let source_promote = |lhs: DType, rhs: DType| {
        if matches!((lhs, rhs), (DType::I64, DType::U64) | (DType::U64, DType::I64)) {
            DType::F32
        } else {
            lhs.promote(rhs)
        }
    };
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(input_shape, input_dtype)?;
    extent(alpha_shape, alpha_dtype)?;
    extent(beta_shape, beta_dtype)?;
    let product_shape = input_shape.broadcast_with(alpha_shape)?;
    let product_dtype = source_promote(input_dtype, alpha_dtype);
    let output_shape = product_shape.broadcast_with(beta_shape)?;
    let output_dtype = source_promote(product_dtype, beta_dtype);
    for (shape, dtype) in [
        (input_shape, product_dtype),
        (alpha_shape, product_dtype),
        (&product_shape, product_dtype),
        (&product_shape, output_dtype),
        (beta_shape, output_dtype),
        (&output_shape, output_dtype),
        (&output_shape, DType::Bool),
        (&output_shape, output_dtype),
        (&output_shape, output_dtype),
        (&output_shape, DType::Bool),
        (&output_shape, output_dtype),
        (&output_shape, output_dtype),
    ] {
        extent(shape, dtype)?;
    }
    let zero = TensorData::scalar_with_dtype(Scalar::I(0), output_dtype);
    let one = TensorData::scalar_with_dtype(Scalar::I(1), output_dtype);
    if zero.dtype() != output_dtype
        || one.dtype() != output_dtype
        || output_shape.broadcast_with(zero.shape())? != output_shape
        || output_shape.broadcast_with(one.shape())? != output_shape
        || source_promote(product_dtype, beta_dtype) != output_dtype
        || source_promote(output_dtype, zero.dtype()) != output_dtype
        || source_promote(output_dtype, one.dtype()) != output_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "hardsigmoid scalar promotion",
            actual: output_dtype,
        });
    }
    Ok(HardsigmoidPlan {
        product_shape,
        product_dtype,
        output_shape,
        output_dtype,
        zero,
        one,
    })
}

impl Graph {
    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        // Tensor.add is `_broadcasted` ADD: both operands are explicitly
        // converted to tinygrad's least-upper dtype before storage-width
        // arithmetic.  The only local lattice difference is I64/U64, whose
        // source meet is F32 rather than the legacy F64 bridge.  Plan every
        // input/cast/broadcast/output extent before publishing a Cast or Add.
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let output_dtype = if matches!(
            (lhs_dtype, rhs_dtype),
            (DType::I64, DType::U64) | (DType::U64, DType::I64)
        ) {
            DType::F32
        } else {
            lhs_dtype.promote(rhs_dtype)
        };
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&lhs_shape, lhs_dtype)?;
        extent(&rhs_shape, rhs_dtype)?;
        extent(&lhs_shape, output_dtype)?;
        extent(&rhs_shape, output_dtype)?;
        extent(&output_shape, output_dtype)?;
        let lhs = if lhs_dtype == output_dtype {
            lhs
        } else {
            self.cast(lhs, output_dtype)?
        };
        let rhs = if rhs_dtype == output_dtype {
            rhs
        } else {
            self.cast(rhs, output_dtype)?
        };
        self.binary(BinaryOp::Add, lhs, rhs)
    }

    pub fn sub(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        // Tensor.sub first applies `_broadcasted`, then evaluates `a + -b`.
        // Its source LUB therefore applies before storage-width subtraction;
        // I64/U64 meets at F32 rather than RustGrad's legacy F64 bridge.  For
        // Bool/Bool specifically, tinygrad's negation is logical-not and ADD
        // is Bool OR, not the raw XOR used by BinaryOp::Sub.  Preflight every
        // input/cast/broadcast/intermediate/output descriptor before nodes.
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let output_dtype = if matches!(
            (lhs_dtype, rhs_dtype),
            (DType::I64, DType::U64) | (DType::U64, DType::I64)
        ) {
            DType::F32
        } else {
            lhs_dtype.promote(rhs_dtype)
        };
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&lhs_shape, lhs_dtype)?;
        extent(&rhs_shape, rhs_dtype)?;
        extent(&lhs_shape, output_dtype)?;
        extent(&rhs_shape, output_dtype)?;
        extent(&output_shape, output_dtype)?;
        if output_dtype == DType::Bool {
            // Plan `-b` as tinygrad's Bool logical-not before the final ADD.
            extent(&rhs_shape, DType::Bool)?;
            extent(&output_shape, DType::Bool)?;
        }
        let lhs = if lhs_dtype == output_dtype {
            lhs
        } else {
            self.cast(lhs, output_dtype)?
        };
        let rhs = if rhs_dtype == output_dtype {
            rhs
        } else {
            self.cast(rhs, output_dtype)?
        };
        if output_dtype == DType::Bool {
            let negated_rhs = self.logical_not(rhs)?;
            self.binary(BinaryOp::Add, lhs, negated_rhs)
        } else {
            self.binary(BinaryOp::Sub, lhs, rhs)
        }
    }

    pub fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        // Tensor.mul is promoted `_broadcasted` multiplication: both values
        // are cast to tinygrad's least-upper dtype before storage-width MUL.
        // Its I64/U64 meet is F32 rather than the legacy F64 bridge.  Fully
        // validate inputs, planned casts, broadcast output, and byte extents
        // before publishing any Cast or Binary node.
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let output_dtype = if matches!(
            (lhs_dtype, rhs_dtype),
            (DType::I64, DType::U64) | (DType::U64, DType::I64)
        ) {
            DType::F32
        } else {
            lhs_dtype.promote(rhs_dtype)
        };
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&lhs_shape, lhs_dtype)?;
        extent(&rhs_shape, rhs_dtype)?;
        extent(&lhs_shape, output_dtype)?;
        extent(&rhs_shape, output_dtype)?;
        extent(&output_shape, output_dtype)?;
        let lhs = if lhs_dtype == output_dtype {
            lhs
        } else {
            self.cast(lhs, output_dtype)?
        };
        let rhs = if rhs_dtype == output_dtype {
            rhs
        } else {
            self.cast(rhs, output_dtype)?
        };
        self.binary(BinaryOp::Mul, lhs, rhs)
    }

    pub fn div(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        // Tensor.div(rounding_mode=None) is true division, not the raw
        // integer DIV op: `_broadcasted` first commits both branches to the
        // source LUB (I64/U64 meets at F32), then only an integral/Bool
        // dividend lifts to F32 and the literal result is
        // `dividend * reciprocal(divisor)`. Plan every cast, unary, and Mul
        // extent before a node is published. Floor/trunc helpers intentionally
        // retain their raw BinaryOp contracts.
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let source_promote = |left: DType, right: DType| {
            if matches!(
                (left, right),
                (DType::I64, DType::U64) | (DType::U64, DType::I64)
            ) {
                DType::F32
            } else {
                left.promote(right)
            }
        };
        let division_dtype = source_promote(lhs_dtype, rhs_dtype);
        let dividend_dtype = if division_dtype.is_float() {
            division_dtype
        } else {
            DType::F32
        };
        let reciprocal_dtype = unary_dtype(UnaryOp::Reciprocal, division_dtype);
        let output_dtype = source_promote(dividend_dtype, reciprocal_dtype);
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        for (shape, dtype) in [
            (&lhs_shape, lhs_dtype),
            (&rhs_shape, rhs_dtype),
            (&lhs_shape, division_dtype),
            (&rhs_shape, division_dtype),
            (&lhs_shape, dividend_dtype),
            (&rhs_shape, reciprocal_dtype),
            (&output_shape, output_dtype),
        ] {
            extent(shape, dtype)?;
        }
        if unary_dtype(UnaryOp::Reciprocal, division_dtype) != reciprocal_dtype
            || source_promote(dividend_dtype, reciprocal_dtype) != output_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "div scalar promotion",
                actual: output_dtype,
            });
        }
        let lhs = if lhs_dtype == division_dtype {
            lhs
        } else {
            self.cast(lhs, division_dtype)?
        };
        let rhs = if rhs_dtype == division_dtype {
            rhs
        } else {
            self.cast(rhs, division_dtype)?
        };
        let dividend = if division_dtype == dividend_dtype {
            lhs
        } else {
            self.cast(lhs, dividend_dtype)?
        };
        let reciprocal = self.reciprocal(rhs)?;
        self.mul(dividend, reciprocal)
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
        // Tensor.div(rounding_mode="floor") first applies `_broadcasted`
        // source-LUB casts. Integer pairs use Python floor division with a
        // typed-zero divisor sentinel; floating and Bool pairs are literally
        // `floor(dividend * reciprocal(divisor))`. RustGrad's raw FloorDiv is
        // Euclidean for negative divisors, so construct the source integer
        // quotient from truncating division, truncating remainder, and the
        // signed nonzero-remainder correction. Preflight every stage first.
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let source_promote = |left: DType, right: DType| {
            if matches!(
                (left, right),
                (DType::I64, DType::U64) | (DType::U64, DType::I64)
            ) {
                DType::F32
            } else {
                left.promote(right)
            }
        };
        let division_dtype = source_promote(lhs_dtype, rhs_dtype);
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
        let dividend_dtype = if division_dtype.is_float() || division_dtype.is_integer() {
            division_dtype
        } else {
            DType::F32
        };
        let reciprocal_dtype = unary_dtype(UnaryOp::Reciprocal, division_dtype);
        let float_output_dtype = source_promote(dividend_dtype, reciprocal_dtype);
        let output_dtype = if division_dtype.is_integer() {
            division_dtype
        } else {
            float_output_dtype
        };
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        for (shape, dtype) in [
            (&lhs_shape, lhs_dtype),
            (&rhs_shape, rhs_dtype),
            (&lhs_shape, division_dtype),
            (&rhs_shape, division_dtype),
            (&output_shape, output_dtype),
        ] {
            extent(shape, dtype)?;
        }
        if division_dtype.is_integer() {
            // CDIV, CMOD, decrement, corrected quotient, and final sentinel
            // selection share the integer output descriptor; the predicates
            // all share its Bool broadcast extent.
            for _ in 0..5 {
                extent(&output_shape, division_dtype)?;
            }
            for _ in 0..5 {
                extent(&output_shape, DType::Bool)?;
            }
        } else {
            extent(&lhs_shape, dividend_dtype)?;
            extent(&rhs_shape, reciprocal_dtype)?;
            extent(&output_shape, float_output_dtype)?;
        }
        if !division_dtype.is_integer()
            && (unary_dtype(UnaryOp::Reciprocal, division_dtype) != reciprocal_dtype
                || source_promote(dividend_dtype, reciprocal_dtype) != float_output_dtype)
        {
            return Err(Error::InvalidElementwiseDType {
                op: "floor_div scalar promotion",
                actual: output_dtype,
            });
        }
        let integer_scalars = if division_dtype.is_integer() {
            let zero_data = TensorData::scalar_with_dtype(Scalar::I(0), division_dtype);
            let one_data = TensorData::scalar_with_dtype(Scalar::I(1), division_dtype);
            extent(zero_data.shape(), division_dtype)?;
            extent(one_data.shape(), division_dtype)?;
            if zero_data.dtype() != division_dtype
                || one_data.dtype() != division_dtype
                || output_shape.broadcast_with(zero_data.shape())? != output_shape
                || output_shape.broadcast_with(one_data.shape())? != output_shape
            {
                return Err(Error::InvalidElementwiseDType {
                    op: "floor_div integer scalar promotion",
                    actual: division_dtype,
                });
            }
            Some((zero_data, one_data))
        } else {
            None
        };
        let lhs = if lhs_dtype == division_dtype {
            lhs
        } else {
            self.cast(lhs, division_dtype)?
        };
        let rhs = if rhs_dtype == division_dtype {
            rhs
        } else {
            self.cast(rhs, division_dtype)?
        };
        if division_dtype.is_integer() {
            let (zero_data, one_data) = integer_scalars
                .expect("integer floor_div scalar plan was preflighted");
            let zero = self.constant(zero_data);
            let one = self.constant(one_data);
            let is_zero = self.eq(rhs, zero)?;
            let safe_rhs = self.select(is_zero, one, rhs)?;
            let quotient = self.binary(BinaryOp::TruncDiv, lhs, safe_rhs)?;
            let remainder = self.binary(BinaryOp::FMod, lhs, safe_rhs)?;
            let nonzero_remainder = self.ne(remainder, zero)?;
            let lhs_negative = self.lt(lhs, zero)?;
            let rhs_negative = self.lt(rhs, zero)?;
            let signs_differ = self.ne(lhs_negative, rhs_negative)?;
            let needs_floor = self.logical_and(nonzero_remainder, signs_differ)?;
            let decremented = self.binary(BinaryOp::Sub, quotient, one)?;
            let corrected = self.select(needs_floor, decremented, quotient)?;
            self.select(is_zero, zero, corrected)
        } else {
            let dividend = if division_dtype == dividend_dtype {
                lhs
            } else {
                self.cast(lhs, dividend_dtype)?
            };
            let reciprocal = self.reciprocal(rhs)?;
            let quotient = self.mul(dividend, reciprocal)?;
            self.floor(quotient)
        }
    }
    pub fn trunc_div(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        // Tensor.div(rounding_mode="trunc") starts with `_broadcasted` LUB
        // casts. Promoted integer pairs use CDIV (whose source zero-divisor
        // value is typed zero); float and Bool paths instead spell
        // `trunc(dividend * reciprocal(divisor))`, lifting only an integral
        // or Bool dividend to F32. Plan all casts, constants, predicates,
        // selections, and intermediate extents before publishing a node.
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let source_promote = |left: DType, right: DType| {
            if matches!(
                (left, right),
                (DType::I64, DType::U64) | (DType::U64, DType::I64)
            ) {
                DType::F32
            } else {
                left.promote(right)
            }
        };
        let division_dtype = source_promote(lhs_dtype, rhs_dtype);
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
        let dividend_dtype = if division_dtype.is_float() || division_dtype.is_integer() {
            division_dtype
        } else {
            DType::F32
        };
        let reciprocal_dtype = unary_dtype(UnaryOp::Reciprocal, division_dtype);
        let float_output_dtype = source_promote(dividend_dtype, reciprocal_dtype);
        let output_dtype = if division_dtype.is_integer() {
            division_dtype
        } else {
            float_output_dtype
        };
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        for (shape, dtype) in [
            (&lhs_shape, lhs_dtype),
            (&rhs_shape, rhs_dtype),
            (&lhs_shape, division_dtype),
            (&rhs_shape, division_dtype),
            (&output_shape, output_dtype),
        ] {
            extent(shape, dtype)?;
        }
        if division_dtype.is_integer() {
            // Integer CDIV has a Bool zero predicate, a guarded quotient,
            // and the final typed-zero selection; it never materializes the
            // float reciprocal descriptors below.
            extent(&output_shape, DType::Bool)?;
            extent(&output_shape, division_dtype)?;
        } else {
            extent(&lhs_shape, dividend_dtype)?;
            extent(&rhs_shape, reciprocal_dtype)?;
            extent(&output_shape, float_output_dtype)?;
        }
        if !division_dtype.is_integer()
            && (unary_dtype(UnaryOp::Reciprocal, division_dtype) != reciprocal_dtype
                || source_promote(dividend_dtype, reciprocal_dtype) != float_output_dtype)
        {
            return Err(Error::InvalidElementwiseDType {
                op: "trunc_div scalar promotion",
                actual: output_dtype,
            });
        }
        let integer_scalars = if division_dtype.is_integer() {
            let zero_data = TensorData::scalar_with_dtype(Scalar::I(0), division_dtype);
            let one_data = TensorData::scalar_with_dtype(Scalar::I(1), division_dtype);
            extent(zero_data.shape(), division_dtype)?;
            extent(one_data.shape(), division_dtype)?;
            if zero_data.dtype() != division_dtype
                || one_data.dtype() != division_dtype
                || output_shape.broadcast_with(zero_data.shape())? != output_shape
                || output_shape.broadcast_with(one_data.shape())? != output_shape
            {
                return Err(Error::InvalidElementwiseDType {
                    op: "trunc_div integer scalar promotion",
                    actual: division_dtype,
                });
            }
            Some((zero_data, one_data))
        } else {
            None
        };
        let lhs = if lhs_dtype == division_dtype {
            lhs
        } else {
            self.cast(lhs, division_dtype)?
        };
        let rhs = if rhs_dtype == division_dtype {
            rhs
        } else {
            self.cast(rhs, division_dtype)?
        };
        if division_dtype.is_integer() {
            // tinygrad's checked-in CDIV helper returns zero for a zero
            // divisor. Select a nonzero placeholder before RustGrad's raw
            // integer op, then restore that typed source sentinel.
            let (zero_data, one_data) = integer_scalars
                .expect("integer trunc_div scalar plan was preflighted");
            let zero = self.constant(zero_data);
            let one = self.constant(one_data);
            let is_zero = self.eq(rhs, zero)?;
            let safe_rhs = self.select(is_zero, one, rhs)?;
            let quotient = self.binary(BinaryOp::TruncDiv, lhs, safe_rhs)?;
            self.select(is_zero, zero, quotient)
        } else {
            let dividend = if division_dtype == dividend_dtype {
                lhs
            } else {
                self.cast(lhs, dividend_dtype)?
            };
            let reciprocal = self.reciprocal(rhs)?;
            let quotient = self.mul(dividend, reciprocal)?;
            self.trunc(quotient)
        }
    }
    pub fn modulo(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        // Tensor.mod first commits both operands to its source LUB, then is
        // literally `a - floor(a / b) * b` outside the integer-only
        // FLOORMOD fast path. Keeping that composition for every dtype also
        // preserves Python divisor-sign remainders, the source zero-divisor
        // value (`a`), and the float VJP instead of raw Mod's zero gradient.
        // Validate every cast, floor-division stage, product, and final
        // subtraction descriptor before publishing a node.
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let source_promote = |left: DType, right: DType| {
            if matches!(
                (left, right),
                (DType::I64, DType::U64) | (DType::U64, DType::I64)
            ) {
                DType::F32
            } else {
                left.promote(right)
            }
        };
        let operand_dtype = source_promote(lhs_dtype, rhs_dtype);
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
        let floor_dividend_dtype = if operand_dtype.is_float() || operand_dtype.is_integer() {
            operand_dtype
        } else {
            DType::F32
        };
        let reciprocal_dtype = unary_dtype(UnaryOp::Reciprocal, operand_dtype);
        let quotient_dtype = if operand_dtype.is_integer() {
            operand_dtype
        } else {
            source_promote(floor_dividend_dtype, reciprocal_dtype)
        };
        let product_dtype = source_promote(quotient_dtype, operand_dtype);
        let output_dtype = source_promote(operand_dtype, product_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        for (shape, dtype) in [
            (&lhs_shape, lhs_dtype),
            (&rhs_shape, rhs_dtype),
            (&lhs_shape, operand_dtype),
            (&rhs_shape, operand_dtype),
            (&output_shape, quotient_dtype),
            (&output_shape, product_dtype),
            (&output_shape, output_dtype),
        ] {
            extent(shape, dtype)?;
        }
        if operand_dtype.is_integer() {
            // floor_div's guarded CDIV/CMOD/sign correction and final
            // sentinel use typed output and Bool masks only.
            for _ in 0..5 {
                extent(&output_shape, operand_dtype)?;
            }
            for _ in 0..5 {
                extent(&output_shape, DType::Bool)?;
            }
        } else {
            extent(&lhs_shape, floor_dividend_dtype)?;
            extent(&rhs_shape, reciprocal_dtype)?;
        }
        if !operand_dtype.is_integer()
            && (unary_dtype(UnaryOp::Reciprocal, operand_dtype) != reciprocal_dtype
                || source_promote(floor_dividend_dtype, reciprocal_dtype) != quotient_dtype)
        {
            return Err(Error::InvalidElementwiseDType {
                op: "mod scalar promotion",
                actual: output_dtype,
            });
        }
        if operand_dtype.is_integer() {
            // Mirror the delegated floor_div scalar checks here so its
            // typed zero/one guard cannot be the first fallible work after
            // this method has published operand casts.
            let zero = TensorData::scalar_with_dtype(Scalar::I(0), operand_dtype);
            let one = TensorData::scalar_with_dtype(Scalar::I(1), operand_dtype);
            extent(zero.shape(), operand_dtype)?;
            extent(one.shape(), operand_dtype)?;
            if zero.dtype() != operand_dtype
                || one.dtype() != operand_dtype
                || output_shape.broadcast_with(zero.shape())? != output_shape
                || output_shape.broadcast_with(one.shape())? != output_shape
            {
                return Err(Error::InvalidElementwiseDType {
                    op: "mod floor_div scalar promotion",
                    actual: operand_dtype,
                });
            }
        }
        let lhs = if lhs_dtype == operand_dtype {
            lhs
        } else {
            self.cast(lhs, operand_dtype)?
        };
        let rhs = if rhs_dtype == operand_dtype {
            rhs
        } else {
            self.cast(rhs, operand_dtype)?
        };
        let quotient = self.floor_div(lhs, rhs)?;
        let product = self.mul(quotient, rhs)?;
        self.sub(lhs, product)
    }
    pub fn fmod(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        // Tensor.fmod first commits both operands to its source LUB, then is
        // literally `a - trunc(a / b) * b` outside the integer-only CMOD
        // path. Keeping that composition for every dtype preserves
        // dividend-sign remainders, the source zero-divisor value (`a`), and
        // the float VJP instead of raw FMod's zero gradient. Validate every
        // cast, truncating-division stage, product, and final subtraction
        // descriptor before publishing a node.
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let source_promote = |left: DType, right: DType| {
            if matches!(
                (left, right),
                (DType::I64, DType::U64) | (DType::U64, DType::I64)
            ) {
                DType::F32
            } else {
                left.promote(right)
            }
        };
        let operand_dtype = source_promote(lhs_dtype, rhs_dtype);
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
        let trunc_dividend_dtype = if operand_dtype.is_float() || operand_dtype.is_integer() {
            operand_dtype
        } else {
            DType::F32
        };
        let reciprocal_dtype = unary_dtype(UnaryOp::Reciprocal, operand_dtype);
        let quotient_dtype = if operand_dtype.is_integer() {
            operand_dtype
        } else {
            source_promote(trunc_dividend_dtype, reciprocal_dtype)
        };
        let product_dtype = source_promote(quotient_dtype, operand_dtype);
        let output_dtype = source_promote(operand_dtype, product_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        for (shape, dtype) in [
            (&lhs_shape, lhs_dtype),
            (&rhs_shape, rhs_dtype),
            (&lhs_shape, operand_dtype),
            (&rhs_shape, operand_dtype),
            (&output_shape, quotient_dtype),
            (&output_shape, product_dtype),
            (&output_shape, output_dtype),
        ] {
            extent(shape, dtype)?;
        }
        if operand_dtype.is_integer() {
            // trunc_div's guarded CDIV/CMOD and final sentinel use typed
            // output and Bool masks only.
            for _ in 0..5 {
                extent(&output_shape, operand_dtype)?;
            }
            for _ in 0..5 {
                extent(&output_shape, DType::Bool)?;
            }
        } else {
            extent(&lhs_shape, trunc_dividend_dtype)?;
            extent(&rhs_shape, reciprocal_dtype)?;
        }
        if !operand_dtype.is_integer()
            && (unary_dtype(UnaryOp::Reciprocal, operand_dtype) != reciprocal_dtype
                || source_promote(trunc_dividend_dtype, reciprocal_dtype) != quotient_dtype)
        {
            return Err(Error::InvalidElementwiseDType {
                op: "fmod scalar promotion",
                actual: output_dtype,
            });
        }
        if operand_dtype.is_integer() {
            // Mirror the delegated trunc_div scalar checks here so its typed
            // zero/one guard cannot be the first fallible work after this
            // method has published operand casts.
            let zero = TensorData::scalar_with_dtype(Scalar::I(0), operand_dtype);
            let one = TensorData::scalar_with_dtype(Scalar::I(1), operand_dtype);
            extent(zero.shape(), operand_dtype)?;
            extent(one.shape(), operand_dtype)?;
            if zero.dtype() != operand_dtype
                || one.dtype() != operand_dtype
                || output_shape.broadcast_with(zero.shape())? != output_shape
                || output_shape.broadcast_with(one.shape())? != output_shape
            {
                return Err(Error::InvalidElementwiseDType {
                    op: "fmod trunc_div scalar promotion",
                    actual: operand_dtype,
                });
            }
        }
        let lhs = if lhs_dtype == operand_dtype {
            lhs
        } else {
            self.cast(lhs, operand_dtype)?
        };
        let rhs = if rhs_dtype == operand_dtype {
            rhs
        } else {
            self.cast(rhs, operand_dtype)?
        };
        let quotient = self.trunc_div(lhs, rhs)?;
        let product = self.mul(quotient, rhs)?;
        self.sub(lhs, product)
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
        // Tensor.eq is `ne(...).logical_not()`, and ne first promotes both
        // value operands through tinygrad's least-upper lattice.  Keep the
        // direct Eq node (its Bool predicate is observably identical) while
        // making its operand contract source-compatible.  In particular,
        // I64/U64 meets at tinygrad's default float (F32), not the legacy F64
        // bridge.  Complete every descriptor/cast/output extent check before
        // either Cast or Compare can mutate the graph.
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let comparison_dtype = if matches!(
            (lhs_dtype, rhs_dtype),
            (DType::I64, DType::U64) | (DType::U64, DType::I64)
        ) {
            DType::F32
        } else {
            lhs_dtype.promote(rhs_dtype)
        };
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&lhs_shape, lhs_dtype)?;
        extent(&rhs_shape, rhs_dtype)?;
        extent(&lhs_shape, comparison_dtype)?;
        extent(&rhs_shape, comparison_dtype)?;
        extent(&output_shape, comparison_dtype)?;
        extent(&output_shape, DType::Bool)?;
        let lhs = if lhs_dtype == comparison_dtype {
            lhs
        } else {
            self.cast(lhs, comparison_dtype)?
        };
        let rhs = if rhs_dtype == comparison_dtype {
            rhs
        } else {
            self.cast(rhs, comparison_dtype)?
        };
        self.compare(CompareOp::Eq, lhs, rhs)
    }
    pub fn ne(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        // Tensor.ne lowers directly to promoted CMPNE.  Match that operand
        // contract while retaining the existing direct predicate (including
        // NaN != anything).  The tinygrad I64/U64 meet is F32 rather than the
        // legacy F64 bridge; preflight all descriptors and cast/output byte
        // extents before a Cast or Compare is appended.
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let comparison_dtype = if matches!(
            (lhs_dtype, rhs_dtype),
            (DType::I64, DType::U64) | (DType::U64, DType::I64)
        ) {
            DType::F32
        } else {
            lhs_dtype.promote(rhs_dtype)
        };
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&lhs_shape, lhs_dtype)?;
        extent(&rhs_shape, rhs_dtype)?;
        extent(&lhs_shape, comparison_dtype)?;
        extent(&rhs_shape, comparison_dtype)?;
        extent(&output_shape, comparison_dtype)?;
        extent(&output_shape, DType::Bool)?;
        let lhs = if lhs_dtype == comparison_dtype {
            lhs
        } else {
            self.cast(lhs, comparison_dtype)?
        };
        let rhs = if rhs_dtype == comparison_dtype {
            rhs
        } else {
            self.cast(rhs, comparison_dtype)?
        };
        self.compare(CompareOp::Ne, lhs, rhs)
    }
    pub fn lt(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        // Tensor.__lt__ lowers directly to promoted CMPLT.  Preserve the
        // typed ordered predicate while matching tinygrad's operand lattice:
        // the I64/U64 meet is F32, rather than RustGrad's legacy F64 bridge.
        // All descriptor, cast, broadcast, and Bool-output extents are
        // checked before either Cast or Compare can mutate the graph.
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let comparison_dtype = if matches!(
            (lhs_dtype, rhs_dtype),
            (DType::I64, DType::U64) | (DType::U64, DType::I64)
        ) {
            DType::F32
        } else {
            lhs_dtype.promote(rhs_dtype)
        };
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&lhs_shape, lhs_dtype)?;
        extent(&rhs_shape, rhs_dtype)?;
        extent(&lhs_shape, comparison_dtype)?;
        extent(&rhs_shape, comparison_dtype)?;
        extent(&output_shape, comparison_dtype)?;
        extent(&output_shape, DType::Bool)?;
        let lhs = if lhs_dtype == comparison_dtype {
            lhs
        } else {
            self.cast(lhs, comparison_dtype)?
        };
        let rhs = if rhs_dtype == comparison_dtype {
            rhs
        } else {
            self.cast(rhs, comparison_dtype)?
        };
        self.compare(CompareOp::Lt, lhs, rhs)
    }
    pub fn le(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        // Tensor.__le__ is literally `(self > rhs).logical_not()`.  Greater
        // itself is reverse CMPLT, so retain the source graph `!(rhs < lhs)`
        // instead of direct Le; in particular, unordered NaN maps to true.
        // Validate source/cast/broadcast/Bool extents before the first node.
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let comparison_dtype = if matches!(
            (lhs_dtype, rhs_dtype),
            (DType::I64, DType::U64) | (DType::U64, DType::I64)
        ) {
            DType::F32
        } else {
            lhs_dtype.promote(rhs_dtype)
        };
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&lhs_shape, lhs_dtype)?;
        extent(&rhs_shape, rhs_dtype)?;
        extent(&lhs_shape, comparison_dtype)?;
        extent(&rhs_shape, comparison_dtype)?;
        extent(&output_shape, comparison_dtype)?;
        extent(&output_shape, DType::Bool)?;
        // The reverse Less result and its logical-not output share this Bool
        // descriptor; validate both planned values explicitly.
        extent(&output_shape, DType::Bool)?;
        let lhs = if lhs_dtype == comparison_dtype {
            lhs
        } else {
            self.cast(lhs, comparison_dtype)?
        };
        let rhs = if rhs_dtype == comparison_dtype {
            rhs
        } else {
            self.cast(rhs, comparison_dtype)?
        };
        let greater = self.compare(CompareOp::Lt, rhs, lhs)?;
        self.logical_not(greater)
    }
    pub fn gt(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        // Tensor.__gt__ is the reverse form of CMPLT: after promoting both
        // operands, it literally evaluates `rhs < lhs`.  Preserve that graph
        // structure rather than a direct Gt predicate.  Its I64/U64 meet is
        // tinygrad's default F32, and all descriptor/cast/broadcast/output
        // extents are validated before any Cast or Compare is appended.
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let comparison_dtype = if matches!(
            (lhs_dtype, rhs_dtype),
            (DType::I64, DType::U64) | (DType::U64, DType::I64)
        ) {
            DType::F32
        } else {
            lhs_dtype.promote(rhs_dtype)
        };
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&lhs_shape, lhs_dtype)?;
        extent(&rhs_shape, rhs_dtype)?;
        extent(&lhs_shape, comparison_dtype)?;
        extent(&rhs_shape, comparison_dtype)?;
        extent(&output_shape, comparison_dtype)?;
        extent(&output_shape, DType::Bool)?;
        let lhs = if lhs_dtype == comparison_dtype {
            lhs
        } else {
            self.cast(lhs, comparison_dtype)?
        };
        let rhs = if rhs_dtype == comparison_dtype {
            rhs
        } else {
            self.cast(rhs, comparison_dtype)?
        };
        self.compare(CompareOp::Lt, rhs, lhs)
    }
    pub fn ge(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        // Tensor.__ge__ is literally `(self < rhs).logical_not()`.  Retain
        // that CMPLT-plus-Not structure instead of direct Ge, so unordered
        // NaN follows tinygrad and maps to true.  Validate every input/cast,
        // broadcast, and Bool intermediate/output extent before mutation.
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let comparison_dtype = if matches!(
            (lhs_dtype, rhs_dtype),
            (DType::I64, DType::U64) | (DType::U64, DType::I64)
        ) {
            DType::F32
        } else {
            lhs_dtype.promote(rhs_dtype)
        };
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)?;
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&lhs_shape, lhs_dtype)?;
        extent(&rhs_shape, rhs_dtype)?;
        extent(&lhs_shape, comparison_dtype)?;
        extent(&rhs_shape, comparison_dtype)?;
        extent(&output_shape, comparison_dtype)?;
        // CMPLT and logical-not both produce this exact Bool descriptor.
        extent(&output_shape, DType::Bool)?;
        extent(&output_shape, DType::Bool)?;
        let lhs = if lhs_dtype == comparison_dtype {
            lhs
        } else {
            self.cast(lhs, comparison_dtype)?
        };
        let rhs = if rhs_dtype == comparison_dtype {
            rhs
        } else {
            self.cast(rhs, comparison_dtype)?
        };
        let less = self.compare(CompareOp::Lt, lhs, rhs)?;
        self.logical_not(less)
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
        // Tensor.where first promotes its two value branches with tinygrad's
        // least-upper lattice, then applies WHERE with a Bool condition.  In
        // particular, the I64/U64 join is default-float (F32), not RustGrad's
        // legacy F64 integer bridge.  Plan every cast and broadcast before
        // appending either Cast or Select so an invalid late operand leaves
        // the graph unchanged.
        let condition_node = self.node(condition)?;
        let condition_shape = condition_node.shape.clone();
        let condition_dtype = condition_node.dtype;
        if condition_dtype != DType::Bool {
            return Err(Error::InvalidLogicalDType {
                op: "select",
                actual: condition_dtype,
            });
        }
        let true_node = self.node(on_true)?;
        let true_shape = true_node.shape.clone();
        let true_dtype = true_node.dtype;
        let false_node = self.node(on_false)?;
        let false_shape = false_node.shape.clone();
        let false_dtype = false_node.dtype;
        let dtype = if matches!(
            (true_dtype, false_dtype),
            (DType::I64, DType::U64) | (DType::U64, DType::I64)
        ) {
            DType::F32
        } else {
            true_dtype.promote(false_dtype)
        };
        let value_shape = true_shape.broadcast_with(&false_shape)?;
        let shape = condition_shape.broadcast_with(&value_shape)?;
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        // Input descriptors, the two source-equivalent cast results, and
        // the final broadcasted WHERE result must all fit before mutation.
        extent(&condition_shape, condition_dtype)?;
        extent(&true_shape, true_dtype)?;
        extent(&false_shape, false_dtype)?;
        extent(&true_shape, dtype)?;
        extent(&false_shape, dtype)?;
        extent(&value_shape, dtype)?;
        extent(&shape, dtype)?;
        let on_true = if true_dtype == dtype {
            on_true
        } else {
            self.cast(on_true, dtype)?
        };
        let on_false = if false_dtype == dtype {
            on_false
        } else {
            self.cast(on_false, dtype)?
        };
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
        // Tensor.neg is logical-not for Bool and a direct negation otherwise.
        // The numeric unary has the same source storage-width, wrapping, and
        // IEEE sign-bit behavior; only Bool needs its source predicate node
        // so it remains nondifferentiable. Validate the complete unary or
        // logical descriptor before either form is published.
        let input_node = self.node(input)?;
        let shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let output_dtype = unary_dtype(UnaryOp::Neg, input_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&shape, input_dtype)?;
        extent(&shape, output_dtype)?;
        if output_dtype != input_dtype {
            return Err(Error::InvalidElementwiseDType {
                op: "neg output dtype",
                actual: output_dtype,
            });
        }
        if input_dtype == DType::Bool {
            self.logical_not(input)
        } else {
            self.unary(UnaryOp::Neg, input)
        }
    }
    pub fn exp(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.exp first casts to its source float storage dtype, promotes
        // the multiply to at least F32, then spells exp as
        // `exp2(x * (1 / ln(2)))` before narrowing back. This differs from a
        // host Exp at narrow and F64 rounding boundaries. Prove every cast,
        // typed scalar, multiply, Exp2, and final output descriptor before
        // the constant or any node is published.
        let input_node = self.node(input)?;
        let shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let source_dtype = unary_dtype(UnaryOp::Exp, input_dtype);
        let work_dtype = source_dtype.promote(DType::F32);
        let output_dtype = source_dtype;
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        for dtype in [input_dtype, source_dtype, work_dtype, work_dtype, output_dtype] {
            extent(&shape, dtype)?;
        }
        let scale = TensorData::scalar_with_dtype(
            Scalar::F(1.0 / std::f64::consts::LN_2),
            work_dtype,
        );
        extent(scale.shape(), work_dtype)?;
        if (!input_dtype.is_float() && source_dtype != DType::F32)
            || (input_dtype.is_float() && source_dtype != input_dtype)
            || source_dtype.promote(DType::F32) != work_dtype
            || scale.dtype() != work_dtype
            || work_dtype.promote(scale.dtype()) != work_dtype
            || shape.broadcast_with(scale.shape())? != shape
        {
            return Err(Error::InvalidElementwiseDType {
                op: "exp source promotion",
                actual: work_dtype,
            });
        }
        let source = if input_dtype == source_dtype {
            input
        } else {
            self.cast(input, source_dtype)?
        };
        let work = if source_dtype == work_dtype {
            source
        } else {
            self.cast(source, work_dtype)?
        };
        let scale = self.constant(scale);
        let exponent = self.mul(work, scale)?;
        let output = self.exp2(exponent)?;
        if work_dtype == output_dtype {
            Ok(output)
        } else {
            self.cast(output, output_dtype)
        }
    }
    pub fn log(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Log, input)
    }
    pub fn abs(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.abs is literally `x * x.sign()`, not the raw unary absolute
        // value. This preserves a negative zero, tinygrad's NaN sign path,
        // and wrapping signed minima. Prove the Sign and Mul descriptors
        // before either node is published; UnaryOp::Abs remains available to
        // lower-level callers that explicitly request its host semantics.
        let input_node = self.node(input)?;
        let shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let sign_dtype = unary_dtype(UnaryOp::Sign, input_dtype);
        let output_dtype = input_dtype.promote(sign_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&shape, input_dtype)?;
        extent(&shape, sign_dtype)?;
        extent(&shape, output_dtype)?;
        if sign_dtype != input_dtype
            || output_dtype != input_dtype
            || shape.broadcast_with(&shape)? != shape
        {
            return Err(Error::InvalidElementwiseDType {
                op: "abs sign/mul promotion",
                actual: output_dtype,
            });
        }
        let sign = self.sign(input)?;
        self.mul(input, sign)
    }
    pub fn reciprocal(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.reciprocal is a direct RECIPROCAL ALU op, rather than a
        // source-level `1 / x` composition. Preserve that operation and its
        // nonfloat-to-F32 rule, but prove the input and output descriptors
        // before publishing the unary node.
        let input_node = self.node(input)?;
        let shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let output_dtype = unary_dtype(UnaryOp::Reciprocal, input_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&shape, input_dtype)?;
        extent(&shape, output_dtype)?;
        if (!input_dtype.is_float() && output_dtype != DType::F32)
            || (input_dtype.is_float() && output_dtype != input_dtype)
        {
            return Err(Error::InvalidElementwiseDType {
                op: "reciprocal output dtype",
                actual: output_dtype,
            });
        }
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
        // Tensor.exp2 is the direct EXP2 ALU primitive: non-floats lift to
        // F32 while every floating storage width is preserved. Unlike the
        // raw unary constructor, this public entry point proves both source
        // and result allocation extents before it can append a node.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let output_dtype = unary_dtype(UnaryOp::Exp2, input_dtype);
        let extent = |dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(input_dtype)?;
        extent(output_dtype)?;
        if (!input_dtype.is_float() && output_dtype != DType::F32)
            || (input_dtype.is_float() && output_dtype != input_dtype)
        {
            return Err(Error::InvalidElementwiseDType {
                op: "exp2 source promotion",
                actual: output_dtype,
            });
        }
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
        // tinygrad spells tanh as `2 * sigmoid(2 * x) - 1`, with sigmoid
        // itself expanded through source-width Exp2/Reciprocal arithmetic.
        // Plan the complete expansion here rather than delegating after a
        // partial graph mutation.
        let input_node = self.node(input)?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let output_dtype = if input_dtype.is_float() {
            input_dtype
        } else {
            DType::F32
        };
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&input_shape, input_dtype)?;
        for _ in [
            "cast",
            "inner multiply",
            "exponent",
            "exp2",
            "denominator",
            "reciprocal",
            "outer multiply",
            "output",
        ] {
            extent(&input_shape, output_dtype)?;
        }
        let one = TensorData::scalar_with_dtype(Scalar::F(1.0), output_dtype);
        let two = TensorData::scalar_with_dtype(Scalar::F(2.0), output_dtype);
        let neg_inv_ln2 = TensorData::scalar_with_dtype(
            Scalar::F(-1.0 / std::f64::consts::LN_2),
            output_dtype,
        );
        if one.dtype() != output_dtype
            || two.dtype() != output_dtype
            || neg_inv_ln2.dtype() != output_dtype
            || input_shape.broadcast_with(one.shape())? != input_shape
            || input_shape.broadcast_with(two.shape())? != input_shape
            || input_shape.broadcast_with(neg_inv_ln2.shape())? != input_shape
            || output_dtype.promote(one.dtype()) != output_dtype
            || output_dtype.promote(two.dtype()) != output_dtype
            || output_dtype.promote(neg_inv_ln2.dtype()) != output_dtype
            || unary_dtype(UnaryOp::Exp2, output_dtype) != output_dtype
            || unary_dtype(UnaryOp::Reciprocal, output_dtype) != output_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "tanh scalar promotion",
                actual: output_dtype,
            });
        }

        let work = if input_dtype == output_dtype {
            input
        } else {
            self.cast(input, output_dtype)?
        };
        let one = self.constant(one);
        let two = self.constant(two);
        let neg_inv_ln2 = self.constant(neg_inv_ln2);
        let doubled = self.mul(two, work)?;
        let exponent = self.mul(doubled, neg_inv_ln2)?;
        let sigmoid = self.reciprocal(self.add(one, self.exp2(exponent)?)?)?;
        self.sub(self.mul(two, sigmoid)?, one)
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
        // tinygrad spells sigmoid as
        // `(1 + (x * (-1 / ln(2))).exp2()).reciprocal()`.  Its weak
        // constants are at x's floating storage width, while non-floats
        // first promote to F32. Validate every cast, scalar, and intermediate
        // descriptor before adding a constant or operation to the graph.
        let input_node = self.node(input)?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let output_dtype = if input_dtype.is_float() {
            input_dtype
        } else {
            DType::F32
        };
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&input_shape, input_dtype)?;
        // Include the optional cast and every literal source operation.
        for _ in ["cast", "scaled", "exp2", "denominator", "reciprocal"] {
            extent(&input_shape, output_dtype)?;
        }
        let one = TensorData::scalar_with_dtype(Scalar::F(1.0), output_dtype);
        let neg_inv_ln2 = TensorData::scalar_with_dtype(
            Scalar::F(-1.0 / std::f64::consts::LN_2),
            output_dtype,
        );
        if one.dtype() != output_dtype
            || neg_inv_ln2.dtype() != output_dtype
            || input_shape.broadcast_with(one.shape())? != input_shape
            || input_shape.broadcast_with(neg_inv_ln2.shape())? != input_shape
            || output_dtype.promote(one.dtype()) != output_dtype
            || output_dtype.promote(neg_inv_ln2.dtype()) != output_dtype
            || unary_dtype(UnaryOp::Exp2, output_dtype) != output_dtype
            || unary_dtype(UnaryOp::Reciprocal, output_dtype) != output_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "sigmoid scalar promotion",
                actual: output_dtype,
            });
        }

        let work = if input_dtype == output_dtype {
            input
        } else {
            self.cast(input, output_dtype)?
        };
        let neg_inv_ln2 = self.constant(neg_inv_ln2);
        let one = self.constant(one);
        let exponent = self.mul(work, neg_inv_ln2)?;
        let denominator = self.add(one, self.exp2(exponent)?)?;
        self.reciprocal(denominator)
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
        // tinygrad implements clamp as two strict ordered Select stages:
        // `(value < min).where(min, value)`, then
        // `(value > max).where(max, value)`.  Plan both stages before any
        // cast, comparison, or Select can grow the graph.  The stage-local
        // dtype also covers tinygrad's I64/U64 default-F32 bridge.
        let input_node = self.node(input)?;
        let mut planned_shape = input_node.shape.clone();
        let mut planned_dtype = input_node.dtype;
        let stage_dtype = |lhs: DType, rhs: DType| {
            if matches!((lhs, rhs), (DType::I64, DType::U64) | (DType::U64, DType::I64)) {
                DType::F32
            } else {
                lhs.promote(rhs)
            }
        };
        let min_stage = if let Some(bound) = min {
            let node = self.node(bound)?;
            let shape = planned_shape.broadcast_with(&node.shape)?;
            shape.numel()?;
            let dtype = stage_dtype(planned_dtype, node.dtype);
            planned_shape = shape.clone();
            planned_dtype = dtype;
            Some((bound, shape, dtype))
        } else {
            None
        };
        let max_stage = if let Some(bound) = max {
            let node = self.node(bound)?;
            let shape = planned_shape.broadcast_with(&node.shape)?;
            shape.numel()?;
            let dtype = stage_dtype(planned_dtype, node.dtype);
            Some((bound, shape, dtype))
        } else {
            None
        };
        let mut value = input;
        if let Some((bound, _shape, dtype)) = min_stage {
            if self.dtype(value)? != dtype {
                value = self.cast(value, dtype)?;
            }
            let bound = if self.dtype(bound)? != dtype {
                self.cast(bound, dtype)?
            } else {
                bound
            };
            let below = self.lt(value, bound)?;
            value = self.select(below, bound, value)?;
        }
        if let Some((bound, _shape, dtype)) = max_stage {
            if self.dtype(value)? != dtype {
                value = self.cast(value, dtype)?;
            }
            let bound = if self.dtype(bound)? != dtype {
                self.cast(bound, dtype)?
            } else {
                bound
            };
            let above = self.gt(value, bound)?;
            value = self.select(above, bound, value)?;
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
        let input_node = self.node(input)?;
        let plan = relu6_plan(&input_node.shape, input_node.dtype)?;
        let zero = self.constant(plan.zero);
        // Both source ReLUs are strict: equality and unordered NaNs select
        // typed zero, unlike a clamp/maximum shortcut.
        let positive = self.select(self.gt(input, zero)?, input, zero)?;
        let six = self.constant(plan.six);
        let shifted = self.sub(input, six)?;
        let upper = self.select(self.gt(shifted, zero)?, shifted, zero)?;
        let output = self.sub(positive, upper)?;
        debug_assert_eq!(self.shape(output).expect("Relu6 preflighted"), &plan.shape);
        debug_assert_eq!(self.dtype(output).expect("Relu6 preflighted"), plan.dtype);
        Ok(output)
    }
    pub fn leaky_relu(&mut self, input: NodeId, slope: NodeId) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let slope_node = self.node(slope)?;
        let slope_shape = slope_node.shape.clone();
        let slope_dtype = slope_node.dtype;
        let plan = leaky_relu_plan(&input_shape, input_dtype, &slope_shape, slope_dtype)?;

        // tinygrad spells this as `(x < 0).where(slope * x, x)`.  Keep the
        // source operand order: it determines the selected NaN payload and
        // its weak common dtype for the I64/U64 bridge.
        let zero = self.constant(plan.zero);
        let negative = self.lt(input, zero)?;
        let input_value = if input_dtype == plan.dtype {
            input
        } else {
            self.cast(input, plan.dtype)?
        };
        let slope_value = if slope_dtype == plan.dtype {
            slope
        } else {
            self.cast(slope, plan.dtype)?
        };
        let scaled = self.mul(slope_value, input_value)?;
        let output = self.select(negative, scaled, input_value)?;
        debug_assert_eq!(self.shape(output).expect("LeakyRelu preflighted"), &plan.shape);
        debug_assert_eq!(self.dtype(output).expect("LeakyRelu preflighted"), plan.dtype);
        Ok(output)
    }
    pub fn silu(&mut self, input: NodeId) -> Result<NodeId> {
        // tinygrad publishes SiLU as the Swish alias, so retain one planned
        // implementation and identical graph/VJP structure for both names.
        self.swish(input)
    }
    pub fn hardsigmoid(&mut self, input: NodeId) -> Result<NodeId> {
        // Preserve the original convenience API using tinygrad's public
        // defaults, while planning the defaults before publishing constants.
        let input_node = self.node(input)?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let parameter_dtype = if input_dtype.is_float() {
            input_dtype
        } else {
            DType::F32
        };
        let alpha = TensorData::scalar_with_dtype(Scalar::F(1.0 / 6.0), parameter_dtype);
        let beta = TensorData::scalar_with_dtype(Scalar::F(0.5), parameter_dtype);
        let plan = hardsigmoid_plan(
            &input_shape,
            input_dtype,
            alpha.shape(),
            alpha.dtype(),
            beta.shape(),
            beta.dtype(),
        )?;
        let alpha = self.constant(alpha);
        let beta = self.constant(beta);
        self.lower_hardsigmoid(input, alpha, beta, plan)
    }

    /// Applies tinygrad's source Hardsigmoid formula with live parameters:
    /// `relu(alpha * x + beta) - relu(alpha * x + beta - 1)`.
    pub fn hardsigmoid_with(
        &mut self,
        input: NodeId,
        alpha: NodeId,
        beta: NodeId,
    ) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let alpha_node = self.node(alpha)?;
        let beta_node = self.node(beta)?;
        let plan = hardsigmoid_plan(
            &input_node.shape,
            input_node.dtype,
            &alpha_node.shape,
            alpha_node.dtype,
            &beta_node.shape,
            beta_node.dtype,
        )?;
        self.lower_hardsigmoid(input, alpha, beta, plan)
    }

    fn lower_hardsigmoid(
        &mut self,
        input: NodeId,
        alpha: NodeId,
        beta: NodeId,
        plan: HardsigmoidPlan,
    ) -> Result<NodeId> {
        let input = if self.node(input)?.dtype == plan.product_dtype {
            input
        } else {
            self.cast(input, plan.product_dtype)?
        };
        let alpha = if self.node(alpha)?.dtype == plan.product_dtype {
            alpha
        } else {
            self.cast(alpha, plan.product_dtype)?
        };
        let product = self.mul(alpha, input)?;
        debug_assert_eq!(
            self.shape(product).expect("Hardsigmoid preflighted"),
            &plan.product_shape
        );
        let product = if plan.product_dtype == plan.output_dtype {
            product
        } else {
            self.cast(product, plan.output_dtype)?
        };
        let beta = if self.node(beta)?.dtype == plan.output_dtype {
            beta
        } else {
            self.cast(beta, plan.output_dtype)?
        };
        let scaled = self.add(product, beta)?;
        let zero = self.constant(plan.zero);
        let one = self.constant(plan.one);
        // ReLU is source-strict: equality and NaN take the typed-zero branch.
        let positive = self.select(self.gt(scaled, zero)?, scaled, zero)?;
        let shifted = self.sub(scaled, one)?;
        let negative = self.select(self.gt(shifted, zero)?, shifted, zero)?;
        let output = self.sub(positive, negative)?;
        debug_assert_eq!(self.shape(output).expect("Hardsigmoid preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("Hardsigmoid preflighted"), plan.output_dtype);
        Ok(output)
    }
    pub fn hardtanh(&mut self, input: NodeId, min: NodeId, max: NodeId) -> Result<NodeId> {
        self.clamp(input, Some(min), Some(max))
    }
    pub fn swish(&mut self, input: NodeId) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let plan = swish_plan(&input_node.shape, input_node.dtype)?;
        // The source is `x * x.sigmoid()`: sigmoid itself is the typed
        // Exp2/Reciprocal composition, not a unary host shortcut.
        let sigmoid = self.sigmoid(input)?;
        let output = self.mul(input, sigmoid)?;
        debug_assert_eq!(self.shape(output).expect("Swish preflighted"), &plan.shape);
        debug_assert_eq!(self.dtype(output).expect("Swish preflighted"), plan.dtype);
        Ok(output)
    }
    pub fn hardswish(&mut self, input: NodeId) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let input_dtype = input_node.dtype;
        let plan = hardswish_plan(&input_node.shape, input_dtype)?;
        let work = if input_dtype == plan.dtype {
            input
        } else {
            self.cast(input, plan.dtype)?
        };
        let three = self.constant(plan.three);
        let shifted = self.add(work, three)?;
        let zero = self.constant(plan.zero);
        // `relu6` is source arithmetic, not a clamp: strict comparisons send
        // equality and NaN to typed zero before the subtraction.
        let positive = self.select(self.gt(shifted, zero)?, shifted, zero)?;
        let six = self.constant(plan.six);
        let shifted_minus_six = self.sub(shifted, six)?;
        let upper = self.select(
            self.gt(shifted_minus_six, zero)?,
            shifted_minus_six,
            zero,
        )?;
        let relu6 = self.sub(positive, upper)?;
        let scaled = self.mul(work, relu6)?;
        let sixth = self.constant(plan.sixth);
        let output = self.mul(scaled, sixth)?;
        debug_assert_eq!(self.shape(output).expect("Hardswish preflighted"), &plan.shape);
        debug_assert_eq!(self.dtype(output).expect("Hardswish preflighted"), plan.dtype);
        Ok(output)
    }
    pub fn quick_gelu(&mut self, input: NodeId) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let plan = quick_gelu_plan(&input_node.shape, input_node.dtype)?;
        // Keep the public source spelling rather than routing through an
        // older fixed-F32 helper: `x * sigmoid(x * 1.702)`.
        let work = if input_node.dtype == plan.dtype {
            input
        } else {
            self.cast(input, plan.dtype)?
        };
        let scale = self.constant(plan.scale);
        let scaled = self.mul(work, scale)?;
        let neg_inv_ln2 = self.constant(plan.neg_inv_ln2);
        let exponent = self.mul(scaled, neg_inv_ln2)?;
        let one = self.constant(plan.one);
        let sigmoid = self.reciprocal(self.add(one, self.exp2(exponent)?)?)?;
        let output = self.mul(work, sigmoid)?;
        debug_assert_eq!(self.shape(output).expect("QuickGELU preflighted"), &plan.shape);
        debug_assert_eq!(self.dtype(output).expect("QuickGELU preflighted"), plan.dtype);
        Ok(output)
    }
    /// Applies GELU using tinygrad's `"tanh"` approximation or the exact
    /// error-function form selected by `"none"`.
    pub fn gelu(&mut self, input: NodeId, approximate: &str) -> Result<NodeId> {
        // tinygrad implements exact GELU as `x * .5 * (1 + erf(x / sqrt(2)))`
        // and its approximation as `.5 * x * (1 + tanh(sqrt(2/pi) *
        // (x + .044715 * x**3)))`.  Both paths use weak source-width
        // constants; the approximate path retains Pow and the compositional
        // Exp2/Reciprocal tanh rather than raw unary shortcuts.
        let input_node = self.node(input)?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        if approximate != "none" && approximate != "tanh" {
            return Err(Error::InvalidElementwiseDType {
                op: "gelu approximate must be `tanh` or `none`",
                actual: input_dtype,
            });
        }
        let output_dtype = if input_dtype.is_float() {
            input_dtype
        } else {
            DType::F32
        };
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&input_shape, input_dtype)?;
        let operation_count = match approximate {
            "none" => 6,
            "tanh" => 14,
            _ => unreachable!("GELU mode validated"),
        };
        for _ in 0..operation_count {
            extent(&input_shape, output_dtype)?;
        }
        let scalar = |value| TensorData::scalar_with_dtype(Scalar::F(value), output_dtype);
        let half = scalar(0.5);
        let one = scalar(1.0);
        let two = scalar(2.0);
        let root_two = scalar(std::f64::consts::SQRT_2);
        let root_two_over_pi = scalar((2.0 / std::f64::consts::PI).sqrt());
        let coefficient = scalar(0.044_715);
        let three = scalar(3.0);
        let neg_inv_ln2 = scalar(-1.0 / std::f64::consts::LN_2);
        let scalars: &[&TensorData] = match approximate {
            "none" => &[&half, &one, &root_two],
            "tanh" => &[
                &half,
                &one,
                &two,
                &root_two_over_pi,
                &coefficient,
                &three,
                &neg_inv_ln2,
            ],
            _ => unreachable!("GELU mode validated"),
        };
        for scalar in scalars {
            if scalar.dtype() != output_dtype
                || input_shape.broadcast_with(scalar.shape())? != input_shape
                || output_dtype.promote(scalar.dtype()) != output_dtype
            {
                return Err(Error::InvalidElementwiseDType {
                    op: "gelu scalar promotion",
                    actual: output_dtype,
                });
            }
        }
        if unary_dtype(UnaryOp::Erf, output_dtype) != output_dtype
            || unary_dtype(UnaryOp::Exp2, output_dtype) != output_dtype
            || unary_dtype(UnaryOp::Reciprocal, output_dtype) != output_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "gelu scalar promotion",
                actual: output_dtype,
            });
        }

        let work = if input_dtype == output_dtype {
            input
        } else {
            self.cast(input, output_dtype)?
        };
        match approximate {
            "none" => {
                let half = self.constant(half);
                let one = self.constant(one);
                let root_two = self.constant(root_two);
                let scaled = self.div(work, root_two)?;
                let erf = self.erf(scaled)?;
                let left = self.mul(work, half)?;
                self.mul(left, self.add(one, erf)?)
            }
            "tanh" => {
                let half = self.constant(half);
                let one = self.constant(one);
                let two = self.constant(two);
                let root_two_over_pi = self.constant(root_two_over_pi);
                let coefficient = self.constant(coefficient);
                let three = self.constant(three);
                let neg_inv_ln2 = self.constant(neg_inv_ln2);
                let cube = self.pow(work, three)?;
                let inner = self.add(work, self.mul(coefficient, cube)?)?;
                let z = self.mul(root_two_over_pi, inner)?;
                let doubled = self.mul(two, z)?;
                let exponent = self.mul(doubled, neg_inv_ln2)?;
                let sigmoid = self.reciprocal(self.add(one, self.exp2(exponent)?)?)?;
                let tanh = self.sub(self.mul(two, sigmoid)?, one)?;
                self.mul(self.mul(half, work)?, self.add(one, tanh)?)
            }
            _ => unreachable!("GELU mode validated"),
        }
    }
    pub fn elu(&mut self, input: NodeId, alpha: NodeId) -> Result<NodeId> {
        // tinygrad ELU is `relu(x) - alpha * relu(1 - exp(x))`, where each
        // ReLU is a strict ordered Select.  Validate every source-width
        // scalar, branch, broadcast, and result descriptor before adding a
        // constant or operation to the graph.
        let input_node = self.node(input)?;
        let alpha_node = self.node(alpha)?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let alpha_shape = alpha_node.shape.clone();
        let alpha_dtype = alpha_node.dtype;
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&input_shape, input_dtype)?;
        extent(&alpha_shape, alpha_dtype)?;
        let exp_dtype = unary_dtype(UnaryOp::Exp, input_dtype);
        let positive_shape = input_shape.clone();
        let negative_shape = input_shape.clone();
        let scaled_shape = negative_shape.broadcast_with(&alpha_shape)?;
        let scaled_dtype = exp_dtype.promote(alpha_dtype);
        let output_shape = positive_shape.broadcast_with(&scaled_shape)?;
        let output_dtype = input_dtype.promote(scaled_dtype);
        for (shape, dtype) in [
            (&positive_shape, input_dtype),
            (&input_shape, exp_dtype),
            (&negative_shape, exp_dtype),
            (&scaled_shape, scaled_dtype),
            (&output_shape, output_dtype),
        ] {
            extent(shape, dtype)?;
        }
        let zero_input = TensorData::scalar_with_dtype(Scalar::I(0), input_dtype);
        let one_exp = TensorData::scalar_with_dtype(Scalar::I(1), exp_dtype);
        let zero_exp = TensorData::scalar_with_dtype(Scalar::I(0), exp_dtype);
        if zero_input.dtype() != input_dtype
            || one_exp.dtype() != exp_dtype
            || zero_exp.dtype() != exp_dtype
            || positive_shape.broadcast_with(zero_input.shape())? != positive_shape
            || input_shape.broadcast_with(one_exp.shape())? != input_shape
            || negative_shape.broadcast_with(zero_exp.shape())? != negative_shape
            || input_dtype.promote(zero_input.dtype()) != input_dtype
            || exp_dtype.promote(one_exp.dtype()) != exp_dtype
            || exp_dtype.promote(zero_exp.dtype()) != exp_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "elu scalar promotion",
                actual: output_dtype,
            });
        }

        let zero_input = self.constant(zero_input);
        let positive_condition = self.gt(input, zero_input)?;
        let positive = self.select(positive_condition, input, zero_input)?;
        let exp = self.exp(input)?;
        let one_exp = self.constant(one_exp);
        let negative_raw = self.sub(one_exp, exp)?;
        let zero_exp = self.constant(zero_exp);
        let negative_condition = self.gt(negative_raw, zero_exp)?;
        let negative_relu = self.select(negative_condition, negative_raw, zero_exp)?;
        let negative = self.mul(alpha, negative_relu)?;
        self.sub(positive, negative)
    }
    pub fn celu(&mut self, input: NodeId, alpha: NodeId) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let alpha_node = self.node(alpha)?;
        let alpha_shape = alpha_node.shape.clone();
        let alpha_dtype = alpha_node.dtype;
        let plan = celu_plan(&input_shape, input_dtype, &alpha_shape, alpha_dtype)?;

        // tinygrad literally evaluates
        // `x.maximum(0) + (alpha * ((x / alpha).exp() - 1)).minimum(0)`.
        // Its division is reciprocal then multiply, and the shared extrema
        // nodes retain tinygrad's ordered left payload on ties and NaNs.
        let input_zero = self.constant(plan.input_zero);
        let positive = self.maximum(input, input_zero)?;
        let division_input = if input_dtype == plan.division_dtype {
            input
        } else {
            self.cast(input, plan.division_dtype)?
        };
        let division_alpha = if alpha_dtype == plan.division_dtype {
            alpha
        } else {
            self.cast(alpha, plan.division_dtype)?
        };
        let dividend = if plan.division_dtype == plan.dividend_dtype {
            division_input
        } else {
            self.cast(division_input, plan.dividend_dtype)?
        };
        let reciprocal_alpha = self.reciprocal(division_alpha)?;
        debug_assert_eq!(
            self.dtype(reciprocal_alpha).expect("CELU preflighted"),
            plan.reciprocal_dtype
        );
        let scaled = self.mul(dividend, reciprocal_alpha)?;
        debug_assert_eq!(self.shape(scaled).expect("CELU preflighted"), &plan.scaled_shape);
        debug_assert_eq!(self.dtype(scaled).expect("CELU preflighted"), plan.scaled_dtype);
        let exp = self.exp(scaled)?;
        debug_assert_eq!(self.dtype(exp).expect("CELU preflighted"), plan.exp_dtype);
        let one = self.constant(plan.one);
        let delta = self.sub(exp, one)?;
        let scaled_negative = self.mul(alpha, delta)?;
        debug_assert_eq!(
            self.shape(scaled_negative).expect("CELU preflighted"),
            &plan.negative_shape
        );
        debug_assert_eq!(
            self.dtype(scaled_negative).expect("CELU preflighted"),
            plan.negative_dtype
        );
        let negative_zero = self.constant(plan.negative_zero);
        let negative = self.minimum(scaled_negative, negative_zero)?;
        let output = self.add(positive, negative)?;
        debug_assert_eq!(self.shape(output).expect("CELU preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("CELU preflighted"), plan.output_dtype);
        Ok(output)
    }
    pub fn selu(&mut self, input: NodeId, alpha: NodeId, gamma: NodeId) -> Result<NodeId> {
        // tinygrad SELU is `gamma * where(x >= 0, x, alpha * (exp(x) - 1))`.
        // In particular it is not gamma times ELU: the >= branch preserves
        // both zero signs and sends NaNs through the exponential branch.
        // Validate every source-width scalar, branch, broadcast, and result
        // descriptor before adding a constant or operation to the graph.
        let input_node = self.node(input)?;
        let alpha_node = self.node(alpha)?;
        let gamma_node = self.node(gamma)?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let alpha_shape = alpha_node.shape.clone();
        let alpha_dtype = alpha_node.dtype;
        let gamma_shape = gamma_node.shape.clone();
        let gamma_dtype = gamma_node.dtype;
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&input_shape, input_dtype)?;
        extent(&alpha_shape, alpha_dtype)?;
        extent(&gamma_shape, gamma_dtype)?;

        let exp_dtype = unary_dtype(UnaryOp::Exp, input_dtype);
        let condition_shape = input_shape.clone();
        let negative_raw_shape = input_shape.clone();
        let negative_shape = negative_raw_shape.broadcast_with(&alpha_shape)?;
        let negative_dtype = exp_dtype.promote(alpha_dtype);
        let branch_shape = input_shape.broadcast_with(&negative_shape)?;
        let branch_dtype = input_dtype.promote(negative_dtype);
        let output_shape = branch_shape.broadcast_with(&gamma_shape)?;
        let output_dtype = branch_dtype.promote(gamma_dtype);
        for (shape, dtype) in [
            (&condition_shape, DType::Bool),
            (&input_shape, exp_dtype),
            (&negative_raw_shape, exp_dtype),
            (&negative_shape, negative_dtype),
            (&branch_shape, branch_dtype),
            (&output_shape, output_dtype),
        ] {
            extent(shape, dtype)?;
        }

        let zero_input = TensorData::scalar_with_dtype(Scalar::I(0), input_dtype);
        let one_exp = TensorData::scalar_with_dtype(Scalar::I(1), exp_dtype);
        if zero_input.dtype() != input_dtype
            || one_exp.dtype() != exp_dtype
            || input_shape.broadcast_with(zero_input.shape())? != input_shape
            || input_shape.broadcast_with(one_exp.shape())? != input_shape
            || input_dtype.promote(zero_input.dtype()) != input_dtype
            || exp_dtype.promote(one_exp.dtype()) != exp_dtype
            || condition_shape.broadcast_with(&branch_shape)? != branch_shape
        {
            return Err(Error::InvalidElementwiseDType {
                op: "selu scalar promotion",
                actual: output_dtype,
            });
        }

        let zero_input = self.constant(zero_input);
        let condition = self.ge(input, zero_input)?;
        let exp = self.exp(input)?;
        let one_exp = self.constant(one_exp);
        let negative_raw = self.sub(exp, one_exp)?;
        let negative = self.mul(alpha, negative_raw)?;
        let branch = self.select(condition, input, negative)?;
        self.mul(gamma, branch)
    }
    pub fn softplus(&mut self, input: NodeId, beta: NodeId) -> Result<NodeId> {
        // tinygrad spells softplus as `(1 / beta) * (x * beta).logaddexp(0)`.
        // Its zero and one are weak constants at their receiving operation's
        // storage width; true division is reciprocal followed by multiply.
        // Validate the entire stable composition before appending a cast,
        // constant, or node.
        let input_node = self.node(input)?;
        let beta_node = self.node(beta)?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let beta_shape = beta_node.shape.clone();
        let beta_dtype = beta_node.dtype;
        let source_promote = |lhs: DType, rhs: DType| {
            if matches!((lhs, rhs), (DType::I64, DType::U64) | (DType::U64, DType::I64)) {
                DType::F32
            } else {
                lhs.promote(rhs)
            }
        };
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&input_shape, input_dtype)?;
        extent(&beta_shape, beta_dtype)?;
        let scaled_shape = input_shape.broadcast_with(&beta_shape)?;
        let scaled_dtype = source_promote(input_dtype, beta_dtype);
        // A weak floating zero retains a floating scaled dtype, but lifts an
        // exact scaled value to tinygrad's default F32 width.
        let log_dtype = if scaled_dtype.is_float() {
            scaled_dtype
        } else {
            DType::F32
        };
        // `1 / beta` uses the same source policy: reciprocal arithmetic is
        // float for exact beta storage and otherwise retains beta's width.
        let inverse_dtype = if beta_dtype.is_float() {
            beta_dtype
        } else {
            DType::F32
        };
        let output_shape = scaled_shape.broadcast_with(&beta_shape)?;
        let output_dtype = source_promote(log_dtype, inverse_dtype);
        for (shape, dtype) in [
            (&scaled_shape, scaled_dtype),
            (&scaled_shape, log_dtype),
            (&beta_shape, inverse_dtype),
            (&output_shape, output_dtype),
        ] {
            extent(shape, dtype)?;
        }
        let zero = TensorData::scalar_with_dtype(Scalar::F(0.0), log_dtype);
        let one = TensorData::scalar_with_dtype(Scalar::F(1.0), inverse_dtype);
        if zero.dtype() != log_dtype
            || one.dtype() != inverse_dtype
            || scaled_shape.broadcast_with(zero.shape())? != scaled_shape
            || beta_shape.broadcast_with(one.shape())? != beta_shape
            || source_promote(log_dtype, zero.dtype()) != log_dtype
            || source_promote(inverse_dtype, one.dtype()) != inverse_dtype
            || unary_dtype(UnaryOp::Reciprocal, inverse_dtype) != inverse_dtype
            || output_shape != scaled_shape
        {
            return Err(Error::InvalidElementwiseDType {
                op: "softplus scalar promotion",
                actual: output_dtype,
            });
        }

        let scaled_input = if input_dtype == scaled_dtype {
            input
        } else {
            self.cast(input, scaled_dtype)?
        };
        let scaled_beta = if beta_dtype == scaled_dtype {
            beta
        } else {
            self.cast(beta, scaled_dtype)?
        };
        let scaled = self.mul(scaled_input, scaled_beta)?;
        let logged_input = if scaled_dtype == log_dtype {
            scaled
        } else {
            self.cast(scaled, log_dtype)?
        };
        let zero = self.constant(zero);
        let logged = self.logaddexp(logged_input, zero)?;
        let inverse_beta_input = if beta_dtype == inverse_dtype {
            beta
        } else {
            self.cast(beta, inverse_dtype)?
        };
        let one = self.constant(one);
        let inverse_beta = self.mul(one, self.reciprocal(inverse_beta_input)?)?;
        let logged = if log_dtype == output_dtype {
            logged
        } else {
            self.cast(logged, output_dtype)?
        };
        let inverse_beta = if inverse_dtype == output_dtype {
            inverse_beta
        } else {
            self.cast(inverse_beta, output_dtype)?
        };
        self.mul(inverse_beta, logged)
    }
    pub fn mish(&mut self, input: NodeId) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let plan = mish_plan(&input_node.shape, input_node.dtype)?;
        // tinygrad spells Mish as `x * x.softplus().tanh()`.  The delegated
        // helpers are already source-aligned; `MishPlan` proves their full
        // descriptor chain before this default beta constant is published.
        let beta = self.constant(plan.beta);
        let softplus = self.softplus(input, beta)?;
        let tanh = self.tanh(softplus)?;
        let output = self.mul(input, tanh)?;
        debug_assert_eq!(self.shape(output).expect("Mish preflighted"), &plan.shape);
        debug_assert_eq!(self.dtype(output).expect("Mish preflighted"), plan.dtype);
        Ok(output)
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
        // tinygrad softsign is `x / (1 + x.abs())`, where abs is literally
        // `x * x.sign()` and true division is `x * reciprocal(denominator)`.
        // This preserves source signed-zero and signed-integer wrapping.
        let input_node = self.node(input)?;
        let shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let reciprocal_dtype = unary_dtype(UnaryOp::Reciprocal, input_dtype);
        let output_dtype = input_dtype.promote(reciprocal_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        for dtype in [
            input_dtype,
            input_dtype,
            input_dtype,
            input_dtype,
            reciprocal_dtype,
            output_dtype,
        ] {
            extent(&shape, dtype)?;
        }
        let one = TensorData::scalar_with_dtype(Scalar::I(1), input_dtype);
        if one.dtype() != input_dtype
            || shape.broadcast_with(one.shape())? != shape
            || input_dtype.promote(one.dtype()) != input_dtype
            || input_dtype.promote(reciprocal_dtype) != output_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "softsign scalar promotion",
                actual: output_dtype,
            });
        }

        let one = self.constant(one);
        let absolute = self.mul(input, self.sign(input)?)?;
        let denominator = self.add(one, absolute)?;
        self.mul(input, self.reciprocal(denominator)?)
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
        // The raw Sign evaluator already matches tinygrad's literal
        // `ne(0).where(lt(0).where(-1, 1), 0)` contract: NaN takes +1 and
        // both signed zeroes take canonical +0. Preflight its preserved
        // descriptor before publishing the unary node.
        let input_node = self.node(input)?;
        let shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let output_dtype = unary_dtype(UnaryOp::Sign, input_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&shape, input_dtype)?;
        extent(&shape, output_dtype)?;
        if output_dtype != input_dtype {
            return Err(Error::InvalidElementwiseDType {
                op: "sign output dtype",
                actual: output_dtype,
            });
        }
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
        let lhs_dtype = self.node(lhs)?.dtype;
        let rhs_dtype = self.node(rhs)?.dtype;
        let promoted = lhs_dtype.promote(rhs_dtype);
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
        // tinygrad's extrema path resolves the otherwise-unrepresentable
        // I64/U64 pair through its default F32 dtype *before* ordered
        // comparison.  Keep the public BinaryOp node, but make that concrete
        // source cast explicit so every evaluator sees the same operands.
        let extrema_i64_u64_bridge = matches!(op, BinaryOp::Maximum | BinaryOp::Minimum)
            && matches!(
                (lhs_dtype, rhs_dtype),
                (DType::I64, DType::U64) | (DType::U64, DType::I64)
            );
        let dtype = if extrema_i64_u64_bridge { DType::F32 } else { dtype };
        let lhs = if extrema_i64_u64_bridge {
            self.cast(lhs, dtype)?
        } else {
            lhs
        };
        let rhs = if extrema_i64_u64_bridge {
            self.cast(rhs, dtype)?
        } else {
            rhs
        };
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
