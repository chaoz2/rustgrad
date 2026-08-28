use super::*;
use crate::{DType, Error, Result, TensorData};

/// Descriptor-only plan for tinygrad's `Tensor.bitwise_not` spelling. Bool
/// delegates to `logical_not`; integer values XOR a scalar mask committed at
/// their storage width.
struct BitwiseNotPlan {
    shape: Shape,
    dtype: DType,
    mask: TensorData,
}

/// Complete source-LUB descriptor for tinygrad's binary bitwise methods.
/// `Tensor._broadcasted` casts both operands before the Bool/integer ALU.
struct BitwiseBinaryPlan {
    output_shape: Shape,
    lhs_dtype: DType,
    rhs_dtype: DType,
    output_dtype: DType,
}

fn bitwise_binary_plan(
    lhs_shape: &Shape,
    lhs_dtype: DType,
    rhs_shape: &Shape,
    rhs_dtype: DType,
    op: BinaryOp,
) -> Result<BitwiseBinaryPlan> {
    debug_assert!(matches!(op, BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor));
    // tinygrad's least-upper lattice bridges I64/U64 through its default
    // float (F32). Bitwise ALU then rejects that result before any cast.
    let output_dtype = if matches!(
        (lhs_dtype, rhs_dtype),
        (DType::I64, DType::U64) | (DType::U64, DType::I64)
    ) {
        DType::F32
    } else {
        lhs_dtype.promote(rhs_dtype)
    };
    let output_shape = lhs_shape.broadcast_with(rhs_shape)?;
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    // Validate both storage inputs, their source-LUB cast descriptors, and
    // the broadcast result before emitting Cast or Binary nodes.
    extent(lhs_shape, lhs_dtype)?;
    extent(rhs_shape, rhs_dtype)?;
    extent(lhs_shape, output_dtype)?;
    extent(rhs_shape, output_dtype)?;
    extent(&output_shape, output_dtype)?;
    if output_dtype.is_float() {
        return Err(Error::InvalidElementwiseDType {
            op: op.name(),
            actual: output_dtype,
        });
    }
    Ok(BitwiseBinaryPlan {
        output_shape,
        lhs_dtype,
        rhs_dtype,
        output_dtype,
    })
}

fn bitwise_scalar_dtype(input_dtype: DType, value: Scalar, op: BinaryOp) -> Result<DType> {
    if input_dtype.is_float() || matches!(value, Scalar::F(_)) {
        // Python float constants are weakfloat. Their least-upper result is
        // float and therefore rejected by the bitwise UOp just like a live
        // floating operand.
        return Err(Error::InvalidElementwiseDType {
            op: op.name(),
            actual: if input_dtype.is_float() { input_dtype } else { DType::F32 },
        });
    }
    if input_dtype.is_integer() {
        return Ok(input_dtype);
    }
    debug_assert_eq!(input_dtype, DType::Bool);
    Ok(match value {
        Scalar::Bool(_) => DType::Bool,
        // A Python int is weakint. With only Bool on the other side it
        // reaches tinygrad's configured default integer: I32 unless its
        // mathematical value overflows I32, otherwise I64.
        Scalar::I(value) if value >= i32::MIN as i64 && value <= i32::MAX as i64 => DType::I32,
        Scalar::U(value) if value <= i32::MAX as u64 => DType::I32,
        Scalar::I(_) | Scalar::U(_) => DType::I64,
        Scalar::F(_) => unreachable!("floating scalars returned above"),
    })
}

/// Complete descriptor and weak-scalar commitment for tinygrad's
/// `Tensor.maximum(const)` and `Tensor.minimum(const)` forms. The live
/// extrema root already carries tinygrad's ordered `Compare -> Select`
/// behavior (including retaining the left payload for ties and NaNs); this
/// plan only commits the Python scalar before that exact root is published.
struct ExtremaScalarPlan {
    output_shape: Shape,
    input_dtype: DType,
    output_dtype: DType,
    scalar: TensorData,
}

/// Scalar-value descriptor for tinygrad's `Tensor.masked_fill`. Source
/// spells this as `mask.where(value, input)`: the fill scalar commits at the
/// input branch's LUB before the Bool mask broadcasts the final Select.
struct MaskedFillScalarPlan {
    output_shape: Shape,
    output_dtype: DType,
    scalar: TensorData,
}

#[derive(Clone, Copy)]
enum WhereBranch {
    Live(NodeId),
    Scalar(Scalar),
}

/// Complete descriptor plan for public tinygrad `Tensor.where` forms with at
/// least one Python scalar branch.  Tinygrad picks the first live payload as
/// the weak-scalar reference, or the Bool condition when both are scalars;
/// then it promotes the two payloads before broadcasting the condition.
struct WhereScalarPlan {
    output_shape: Shape,
    output_dtype: DType,
    on_true_scalar: Option<TensorData>,
    on_false_scalar: Option<TensorData>,
}

/// Descriptor and weak-scalar commitment for tinygrad `Tensor.add` and its
/// reflected Python form. The final Add operand order stays observable for
/// raw floating payloads even though arithmetic is otherwise commutative.
struct AddScalarPlan {
    output_shape: Shape,
    input_dtype: DType,
    output_dtype: DType,
    scalar: TensorData,
}

/// Complete descriptor and weak-scalar commitment for tinygrad's public
/// comparison forms. Every predicate shares `_broadcasted` source-LUB
/// operands and produces Bool; inclusive/equality forms additionally use a
/// literal Bool inversion shell.
struct ComparisonScalarPlan {
    output_shape: Shape,
    comparison_dtype: DType,
    scalar: TensorData,
}

/// Descriptor and weak-scalar commitment for tinygrad `Tensor.sub` and its
/// reflected Python form. Its source graph is exactly `a + (-b)` after
/// `_broadcasted`, including Bool's logical-not right branch.
struct SubScalarPlan {
    output_shape: Shape,
    input_dtype: DType,
    output_dtype: DType,
    scalar: TensorData,
}

/// Descriptor and weak-scalar commitment for tinygrad `Tensor.mul` and its
/// reflected Python form. The final MUL order is retained for raw payload and
/// autodiff structure even though multiplication is mathematically symmetric.
struct MulScalarPlan {
    output_shape: Shape,
    input_dtype: DType,
    output_dtype: DType,
    scalar: TensorData,
}

/// Descriptor and weak-scalar commitment for tinygrad true division. The
/// source composition is `promoted_lhs * reciprocal(promoted_rhs)`; reflected
/// division changes which branch serves as the reciprocal denominator.
struct DivScalarPlan {
    output_shape: Shape,
    output_dtype: DType,
    scalar: TensorData,
}

/// Descriptor and weak-scalar commitment for tinygrad floor division. It
/// retains the existing integer sentinel/correction path or float
/// reciprocal-Mul-Floor path according to the committed source dtype.
struct FloorDivScalarPlan {
    output_shape: Shape,
    output_dtype: DType,
    scalar: TensorData,
}

/// Descriptor and weak-scalar commitment for tinygrad truncating division.
/// It retains the existing integer CDIV zero-sentinel path or the literal
/// float reciprocal-Mul-Trunc path according to the committed source dtype.
struct TruncDivScalarPlan {
    output_shape: Shape,
    output_dtype: DType,
    scalar: TensorData,
}

/// Descriptor and weak-scalar commitment for tinygrad `Tensor.mod`. This is
/// intentionally separate from fmod: it plans `a - floor_div(a, b) * b`.
struct ModuloScalarPlan {
    output_shape: Shape,
    output_dtype: DType,
    scalar: TensorData,
}

/// Descriptor and weak-scalar commitment for tinygrad `Tensor.fmod`. Unlike
/// modulo, this retains its truncation-based `a - trunc(a / b) * b` path.
struct FmodScalarPlan {
    output_shape: Shape,
    output_dtype: DType,
    scalar: TensorData,
}

/// Resolves a Python-style scalar at the width tinygrad's `_broadcasted`
/// commits after its weak scalar promotion. This is shared by scalar-right
/// public elementwise forms; it intentionally does not model a live U64
/// operand, so the I64/U64 bridge remains a live-binary-only boundary.
fn source_weak_scalar_dtype(input_dtype: DType, value: Scalar) -> DType {
    match value {
        // Python bool is a strong Bool. It lifts to the live tensor's dtype
        // when one exists, while Bool/Bool stays Bool.
        Scalar::Bool(_) => input_dtype,
        // Python integers are weakint. Against Bool alone, tinygrad resolves
        // them to its default integer width, selecting I64 only when the
        // mathematical constant exceeds I32. Against a concrete tensor they
        // commit directly at that tensor's storage width.
        Scalar::I(value) if input_dtype == DType::Bool => {
            if value < i32::MIN as i64 || value > i32::MAX as i64 {
                DType::I64
            } else {
                DType::I32
            }
        }
        Scalar::U(value) if input_dtype == DType::Bool => {
            if value > i32::MAX as u64 { DType::I64 } else { DType::I32 }
        }
        Scalar::I(_) | Scalar::U(_) => input_dtype,
        // Python floats are weakfloat. A concrete floating tensor commits the
        // constant at its own width; Bool and integer tensors instead meet
        // weakfloat at tinygrad's configured default float, F32.
        Scalar::F(_) if input_dtype.is_float() => input_dtype,
        Scalar::F(_) => DType::F32,
    }
}

fn extrema_scalar_plan(graph: &Graph, input: NodeId, value: Scalar) -> Result<ExtremaScalarPlan> {
    let source = graph.node(input)?;
    let output_shape = source.shape.clone();
    let input_dtype = source.dtype;
    let output_dtype = source_weak_scalar_dtype(input_dtype, value);
    let scalar = TensorData::scalar_with_dtype(value, output_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };

    // Validate the original input, weak scalar commitment, source-LUB cast,
    // ordered comparison predicate, and selected result before any Constant,
    // Cast, or Binary node is appended.
    extent(&output_shape, input_dtype)?;
    extent(scalar.shape(), scalar.dtype())?;
    extent(&output_shape, output_dtype)?;
    extent(&output_shape, DType::Bool)?;
    if scalar.dtype() != output_dtype
        || output_shape.broadcast_with(scalar.shape())? != output_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "extrema scalar promotion",
            actual: output_dtype,
        });
    }

    Ok(ExtremaScalarPlan {
        output_shape,
        input_dtype,
        output_dtype,
        scalar,
    })
}

fn masked_fill_scalar_plan(
    graph: &Graph,
    input: NodeId,
    mask: NodeId,
    value: Scalar,
) -> Result<MaskedFillScalarPlan> {
    let input_node = graph.node(input)?;
    let input_shape = input_node.shape.clone();
    let input_dtype = input_node.dtype;
    let mask_node = graph.node(mask)?;
    let mask_shape = mask_node.shape.clone();
    if mask_node.dtype != DType::Bool {
        return Err(Error::InvalidLogicalDType {
            op: "select",
            actual: mask_node.dtype,
        });
    }
    let value_dtype = source_weak_scalar_dtype(input_dtype, value);
    let scalar = TensorData::scalar_with_dtype(value, value_dtype);
    let value_shape = Shape::new([]);
    let branch_shape = value_shape.broadcast_with(&input_shape)?;
    let output_shape = mask_shape.broadcast_with(&branch_shape)?;
    let output_dtype = source_lub(value_dtype, input_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };

    // Validate the original mask/input, weak scalar storage, both promoted
    // WHERE branches, broadcast condition, and selected result before any
    // constant, cast, or Select node is made visible.
    for (shape, dtype) in [
        (&input_shape, input_dtype),
        (&mask_shape, DType::Bool),
        (scalar.shape(), scalar.dtype()),
        (&input_shape, output_dtype),
        (scalar.shape(), output_dtype),
        (&branch_shape, output_dtype),
        (&output_shape, DType::Bool),
        (&output_shape, output_dtype),
    ] {
        extent(shape, dtype)?;
    }
    if scalar.shape() != &value_shape
        || scalar.dtype() != value_dtype
        || output_dtype != source_lub(value_dtype, input_dtype)
        || value_shape.broadcast_with(&input_shape)? != branch_shape
        || mask_shape.broadcast_with(&branch_shape)? != output_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "masked_fill scalar promotion",
            actual: output_dtype,
        });
    }
    Ok(MaskedFillScalarPlan {
        output_shape,
        output_dtype,
        scalar,
    })
}

fn where_scalar_plan(
    graph: &Graph,
    condition: NodeId,
    on_true: WhereBranch,
    on_false: WhereBranch,
) -> Result<WhereScalarPlan> {
    let condition_node = graph.node(condition)?;
    let condition_shape = condition_node.shape.clone();
    if condition_node.dtype != DType::Bool {
        return Err(Error::InvalidLogicalDType {
            op: "select",
            actual: condition_node.dtype,
        });
    }
    let live = |branch: WhereBranch| -> Result<Option<(Shape, DType)>> {
        match branch {
            WhereBranch::Live(node) => {
                let node = graph.node(node)?;
                Ok(Some((node.shape.clone(), node.dtype)))
            }
            WhereBranch::Scalar(_) => Ok(None),
        }
    };
    let true_live = live(on_true)?;
    let false_live = live(on_false)?;

    // `ref` is true, then false, then condition in the checked-in source.
    // This fixes the first weak scalar before the second is committed.
    let true_reference_dtype = true_live
        .as_ref()
        .map(|(_, dtype)| *dtype)
        .or_else(|| false_live.as_ref().map(|(_, dtype)| *dtype))
        .unwrap_or(DType::Bool);
    let (true_shape, true_dtype, on_true_scalar) = match on_true {
        WhereBranch::Live(_) => {
            let (shape, dtype) = true_live.expect("live true branch was resolved");
            (shape, dtype, None)
        }
        WhereBranch::Scalar(value) => {
            let dtype = source_weak_scalar_dtype(true_reference_dtype, value);
            (Shape::new([]), dtype, Some(TensorData::scalar_with_dtype(value, dtype)))
        }
    };
    let (false_shape, false_dtype, on_false_scalar) = match on_false {
        WhereBranch::Live(_) => {
            let (shape, dtype) = false_live.expect("live false branch was resolved");
            (shape, dtype, None)
        }
        WhereBranch::Scalar(value) => {
            let dtype = source_weak_scalar_dtype(true_dtype, value);
            (Shape::new([]), dtype, Some(TensorData::scalar_with_dtype(value, dtype)))
        }
    };
    let value_shape = true_shape.broadcast_with(&false_shape)?;
    let output_shape = condition_shape.broadcast_with(&value_shape)?;
    let output_dtype = source_lub(true_dtype, false_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };

    // Validate all original/live/scalar descriptors, source-LUB cast results,
    // payload and condition broadcasts, and selected output before constants
    // or nodes are published.
    extent(&condition_shape, DType::Bool)?;
    extent(&true_shape, true_dtype)?;
    extent(&false_shape, false_dtype)?;
    if let Some(scalar) = &on_true_scalar {
        extent(scalar.shape(), scalar.dtype())?;
    }
    if let Some(scalar) = &on_false_scalar {
        extent(scalar.shape(), scalar.dtype())?;
    }
    for (shape, dtype) in [
        (&true_shape, output_dtype),
        (&false_shape, output_dtype),
        (&value_shape, output_dtype),
        (&output_shape, DType::Bool),
        (&output_shape, output_dtype),
    ] {
        extent(shape, dtype)?;
    }
    if on_true_scalar.as_ref().is_some_and(|scalar| scalar.shape() != &Shape::new([]) || scalar.dtype() != true_dtype)
        || on_false_scalar.as_ref().is_some_and(|scalar| scalar.shape() != &Shape::new([]) || scalar.dtype() != false_dtype)
        || true_shape.broadcast_with(&false_shape)? != value_shape
        || condition_shape.broadcast_with(&value_shape)? != output_shape
        || source_lub(true_dtype, false_dtype) != output_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "where scalar promotion",
            actual: output_dtype,
        });
    }
    Ok(WhereScalarPlan {
        output_shape,
        output_dtype,
        on_true_scalar,
        on_false_scalar,
    })
}

fn add_scalar_plan(graph: &Graph, input: NodeId, value: Scalar) -> Result<AddScalarPlan> {
    let input_node = graph.node(input)?;
    let output_shape = input_node.shape.clone();
    let input_dtype = input_node.dtype;
    let scalar_dtype = source_weak_scalar_dtype(input_dtype, value);
    let output_dtype = source_lub(input_dtype, scalar_dtype);
    let scalar = TensorData::scalar_with_dtype(value, scalar_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };

    // Preflight source input/scalar storage, both `_broadcasted` cast
    // results, scalar broadcast, and storage-width ADD result before a
    // constant or Cast becomes visible.
    for (shape, dtype) in [
        (&output_shape, input_dtype),
        (scalar.shape(), scalar.dtype()),
        (&output_shape, output_dtype),
        (scalar.shape(), output_dtype),
        (&output_shape, output_dtype),
    ] {
        extent(shape, dtype)?;
    }
    if scalar.shape() != &Shape::new([])
        || scalar.dtype() != scalar_dtype
        || scalar_dtype != output_dtype
        || source_lub(input_dtype, scalar_dtype) != output_dtype
        || output_shape.broadcast_with(scalar.shape())? != output_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "add scalar promotion",
            actual: output_dtype,
        });
    }
    Ok(AddScalarPlan {
        output_shape,
        input_dtype,
        output_dtype,
        scalar,
    })
}

fn comparison_scalar_plan(
    graph: &Graph,
    input: NodeId,
    value: Scalar,
) -> Result<ComparisonScalarPlan> {
    let input_node = graph.node(input)?;
    let output_shape = input_node.shape.clone();
    let input_dtype = input_node.dtype;
    let scalar_dtype = source_weak_scalar_dtype(input_dtype, value);
    let comparison_dtype = source_lub(input_dtype, scalar_dtype);
    let scalar = TensorData::scalar_with_dtype(value, scalar_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };

    // Validate source storage, weak scalar commitment, both promoted values,
    // the broadcast comparison, and Bool result before a Constant, Cast, or
    // Compare becomes visible. Inclusive and equality forms additionally
    // consume this same Bool descriptor through literal logical_not stages.
    for (shape, dtype) in [
        (&output_shape, input_dtype),
        (scalar.shape(), scalar.dtype()),
        (&output_shape, comparison_dtype),
        (scalar.shape(), comparison_dtype),
        (&output_shape, comparison_dtype),
        (&output_shape, DType::Bool),
        (&output_shape, DType::Bool),
    ] {
        extent(shape, dtype)?;
    }
    let truth = TensorData::scalar_with_dtype(Scalar::Bool(true), DType::Bool);
    extent(truth.shape(), truth.dtype())?;
    if scalar.shape() != &Shape::new([])
        || scalar.dtype() != scalar_dtype
        || scalar_dtype != comparison_dtype
        || source_lub(input_dtype, scalar_dtype) != comparison_dtype
        || output_shape.broadcast_with(scalar.shape())? != output_shape
        || truth.shape() != &Shape::new([])
        || truth.dtype() != DType::Bool
        || output_shape.broadcast_with(truth.shape())? != output_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "comparison scalar promotion",
            actual: comparison_dtype,
        });
    }
    Ok(ComparisonScalarPlan { output_shape, comparison_dtype, scalar })
}

fn sub_scalar_plan(graph: &Graph, input: NodeId, value: Scalar) -> Result<SubScalarPlan> {
    let input_node = graph.node(input)?;
    let output_shape = input_node.shape.clone();
    let input_dtype = input_node.dtype;
    let scalar_dtype = source_weak_scalar_dtype(input_dtype, value);
    let output_dtype = source_lub(input_dtype, scalar_dtype);
    let scalar = TensorData::scalar_with_dtype(value, scalar_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };

    // Preflight source input/scalar storage, `_broadcasted` casts, right
    // branch negation, and final Add. Bool negation also creates a typed true
    // scalar inside logical_not, so include it before publication.
    for (shape, dtype) in [
        (&output_shape, input_dtype),
        (scalar.shape(), scalar.dtype()),
        (&output_shape, output_dtype),
        (scalar.shape(), output_dtype),
        (&output_shape, output_dtype),
        (&output_shape, output_dtype),
    ] {
        extent(shape, dtype)?;
    }
    if output_dtype == DType::Bool {
        let truth = TensorData::scalar_with_dtype(Scalar::Bool(true), DType::Bool);
        extent(truth.shape(), truth.dtype())?;
        extent(&output_shape, DType::Bool)?;
        if truth.shape() != &Shape::new([]) || truth.dtype() != DType::Bool {
            return Err(Error::InvalidLogicalDType {
                op: "logical_not",
                actual: output_dtype,
            });
        }
    }
    if scalar.shape() != &Shape::new([])
        || scalar.dtype() != scalar_dtype
        || scalar_dtype != output_dtype
        || source_lub(input_dtype, scalar_dtype) != output_dtype
        || output_shape.broadcast_with(scalar.shape())? != output_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "sub scalar promotion",
            actual: output_dtype,
        });
    }
    Ok(SubScalarPlan {
        output_shape,
        input_dtype,
        output_dtype,
        scalar,
    })
}

fn mul_scalar_plan(graph: &Graph, input: NodeId, value: Scalar) -> Result<MulScalarPlan> {
    let input_node = graph.node(input)?;
    let output_shape = input_node.shape.clone();
    let input_dtype = input_node.dtype;
    let scalar_dtype = source_weak_scalar_dtype(input_dtype, value);
    let output_dtype = source_lub(input_dtype, scalar_dtype);
    let scalar = TensorData::scalar_with_dtype(value, scalar_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    // Validate original storage, `_broadcasted` cast results, scalar
    // broadcast, and final storage-width multiplication before publication.
    for (shape, dtype) in [
        (&output_shape, input_dtype),
        (scalar.shape(), scalar.dtype()),
        (&output_shape, output_dtype),
        (scalar.shape(), output_dtype),
        (&output_shape, output_dtype),
    ] {
        extent(shape, dtype)?;
    }
    if scalar.shape() != &Shape::new([])
        || scalar.dtype() != scalar_dtype
        || scalar_dtype != output_dtype
        || source_lub(input_dtype, scalar_dtype) != output_dtype
        || output_shape.broadcast_with(scalar.shape())? != output_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "mul scalar promotion",
            actual: output_dtype,
        });
    }
    Ok(MulScalarPlan {
        output_shape,
        input_dtype,
        output_dtype,
        scalar,
    })
}

fn div_scalar_plan(graph: &Graph, input: NodeId, value: Scalar) -> Result<DivScalarPlan> {
    let input_node = graph.node(input)?;
    let input_shape = input_node.shape.clone();
    let input_dtype = input_node.dtype;
    let scalar_dtype = source_weak_scalar_dtype(input_dtype, value);
    let division_dtype = source_lub(input_dtype, scalar_dtype);
    let dividend_dtype = if division_dtype.is_float() {
        division_dtype
    } else {
        DType::F32
    };
    let reciprocal_dtype = unary_dtype(UnaryOp::Reciprocal, division_dtype);
    let output_dtype = source_lub(dividend_dtype, reciprocal_dtype);
    let scalar = TensorData::scalar_with_dtype(value, scalar_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    // Validate source input/scalar, source-LUB casts, integral dividend lift,
    // denominator reciprocal (including its cast when nonfloat), and final
    // broadcasted Mul before the scalar constant is published.
    for (shape, dtype) in [
        (&input_shape, input_dtype),
        (scalar.shape(), scalar.dtype()),
        (&input_shape, division_dtype),
        (scalar.shape(), division_dtype),
        (&input_shape, dividend_dtype),
        (scalar.shape(), reciprocal_dtype),
        (&input_shape, output_dtype),
    ] {
        extent(shape, dtype)?;
    }
    if scalar.shape() != &Shape::new([])
        || scalar.dtype() != scalar_dtype
        || scalar_dtype != division_dtype
        || source_lub(input_dtype, scalar_dtype) != division_dtype
        || unary_dtype(UnaryOp::Reciprocal, division_dtype) != reciprocal_dtype
        || source_lub(dividend_dtype, reciprocal_dtype) != output_dtype
        || input_shape.broadcast_with(scalar.shape())? != input_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "div scalar promotion",
            actual: output_dtype,
        });
    }
    Ok(DivScalarPlan {
        output_shape: input_shape,
        output_dtype,
        scalar,
    })
}

fn floor_div_scalar_plan(
    graph: &Graph,
    input: NodeId,
    value: Scalar,
) -> Result<FloorDivScalarPlan> {
    let input_node = graph.node(input)?;
    let input_shape = input_node.shape.clone();
    let input_dtype = input_node.dtype;
    let scalar_dtype = source_weak_scalar_dtype(input_dtype, value);
    let division_dtype = source_lub(input_dtype, scalar_dtype);
    let dividend_dtype = if division_dtype.is_float() || division_dtype.is_integer() {
        division_dtype
    } else {
        DType::F32
    };
    let reciprocal_dtype = unary_dtype(UnaryOp::Reciprocal, division_dtype);
    let float_output_dtype = source_lub(dividend_dtype, reciprocal_dtype);
    let output_dtype = if division_dtype.is_integer() {
        division_dtype
    } else {
        float_output_dtype
    };
    let scalar = TensorData::scalar_with_dtype(value, scalar_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    // Validate source/LUB storage plus every existing floor_div branch
    // intermediate before allowing the scalar, sentinel constants, or nodes.
    for (shape, dtype) in [
        (&input_shape, input_dtype),
        (scalar.shape(), scalar.dtype()),
        (&input_shape, division_dtype),
        (scalar.shape(), division_dtype),
        (&input_shape, output_dtype),
    ] {
        extent(shape, dtype)?;
    }
    if division_dtype.is_integer() {
        for _ in 0..5 {
            extent(&input_shape, division_dtype)?;
        }
        for _ in 0..5 {
            extent(&input_shape, DType::Bool)?;
        }
        let zero = TensorData::scalar_with_dtype(Scalar::I(0), division_dtype);
        let one = TensorData::scalar_with_dtype(Scalar::I(1), division_dtype);
        extent(zero.shape(), zero.dtype())?;
        extent(one.shape(), one.dtype())?;
        if zero.shape() != &Shape::new([])
            || one.shape() != &Shape::new([])
            || zero.dtype() != division_dtype
            || one.dtype() != division_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "floor_div integer scalar promotion",
                actual: division_dtype,
            });
        }
    } else {
        extent(&input_shape, dividend_dtype)?;
        extent(scalar.shape(), reciprocal_dtype)?;
        extent(&input_shape, float_output_dtype)?;
        extent(&input_shape, output_dtype)?;
        if unary_dtype(UnaryOp::Reciprocal, division_dtype) != reciprocal_dtype
            || source_lub(dividend_dtype, reciprocal_dtype) != float_output_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "floor_div scalar promotion",
                actual: output_dtype,
            });
        }
    }
    if scalar.shape() != &Shape::new([])
        || scalar.dtype() != scalar_dtype
        || scalar_dtype != division_dtype
        || source_lub(input_dtype, scalar_dtype) != division_dtype
        || input_shape.broadcast_with(scalar.shape())? != input_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "floor_div scalar promotion",
            actual: output_dtype,
        });
    }
    Ok(FloorDivScalarPlan {
        output_shape: input_shape,
        output_dtype,
        scalar,
    })
}

fn trunc_div_scalar_plan(
    graph: &Graph,
    input: NodeId,
    value: Scalar,
) -> Result<TruncDivScalarPlan> {
    let input_node = graph.node(input)?;
    let input_shape = input_node.shape.clone();
    let input_dtype = input_node.dtype;
    let scalar_dtype = source_weak_scalar_dtype(input_dtype, value);
    let division_dtype = source_lub(input_dtype, scalar_dtype);
    let dividend_dtype = if division_dtype.is_float() || division_dtype.is_integer() {
        division_dtype
    } else {
        DType::F32
    };
    let reciprocal_dtype = unary_dtype(UnaryOp::Reciprocal, division_dtype);
    let float_output_dtype = source_lub(dividend_dtype, reciprocal_dtype);
    let output_dtype = if division_dtype.is_integer() {
        division_dtype
    } else {
        float_output_dtype
    };
    let scalar = TensorData::scalar_with_dtype(value, scalar_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    // Fully describe the existing trunc_div branch, including the typed
    // integer zero sentinel or float reciprocal/Mul/Trunc work, before the
    // weak scalar is allowed into the graph.
    for (shape, dtype) in [
        (&input_shape, input_dtype),
        (scalar.shape(), scalar.dtype()),
        (&input_shape, division_dtype),
        (scalar.shape(), division_dtype),
        (&input_shape, output_dtype),
    ] {
        extent(shape, dtype)?;
    }
    if division_dtype.is_integer() {
        extent(&input_shape, DType::Bool)?;
        extent(&input_shape, division_dtype)?;
        let zero = TensorData::scalar_with_dtype(Scalar::I(0), division_dtype);
        let one = TensorData::scalar_with_dtype(Scalar::I(1), division_dtype);
        extent(zero.shape(), zero.dtype())?;
        extent(one.shape(), one.dtype())?;
        if zero.shape() != &Shape::new([])
            || one.shape() != &Shape::new([])
            || zero.dtype() != division_dtype
            || one.dtype() != division_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "trunc_div integer scalar promotion",
                actual: division_dtype,
            });
        }
    } else {
        extent(&input_shape, dividend_dtype)?;
        extent(scalar.shape(), reciprocal_dtype)?;
        extent(&input_shape, float_output_dtype)?;
        extent(&input_shape, output_dtype)?;
        if unary_dtype(UnaryOp::Reciprocal, division_dtype) != reciprocal_dtype
            || source_lub(dividend_dtype, reciprocal_dtype) != float_output_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "trunc_div scalar promotion",
                actual: output_dtype,
            });
        }
    }
    if scalar.shape() != &Shape::new([])
        || scalar.dtype() != scalar_dtype
        || scalar_dtype != division_dtype
        || source_lub(input_dtype, scalar_dtype) != division_dtype
        || input_shape.broadcast_with(scalar.shape())? != input_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "trunc_div scalar promotion",
            actual: output_dtype,
        });
    }
    Ok(TruncDivScalarPlan {
        output_shape: input_shape,
        output_dtype,
        scalar,
    })
}

fn modulo_scalar_plan(
    graph: &Graph,
    input: NodeId,
    value: Scalar,
) -> Result<ModuloScalarPlan> {
    let input_node = graph.node(input)?;
    let input_shape = input_node.shape.clone();
    let input_dtype = input_node.dtype;
    let scalar_dtype = source_weak_scalar_dtype(input_dtype, value);
    let operand_dtype = source_lub(input_dtype, scalar_dtype);
    let floor_dividend_dtype = if operand_dtype.is_float() || operand_dtype.is_integer() {
        operand_dtype
    } else {
        DType::F32
    };
    let reciprocal_dtype = unary_dtype(UnaryOp::Reciprocal, operand_dtype);
    let quotient_dtype = if operand_dtype.is_integer() {
        operand_dtype
    } else {
        source_lub(floor_dividend_dtype, reciprocal_dtype)
    };
    let product_dtype = source_lub(quotient_dtype, operand_dtype);
    let output_dtype = source_lub(operand_dtype, product_dtype);
    let scalar = TensorData::scalar_with_dtype(value, scalar_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    // Validate source/LUB branches, all delegated floor_div work, product,
    // and source-literal subtraction before constants or nodes are visible.
    for (shape, dtype) in [
        (&input_shape, input_dtype),
        (scalar.shape(), scalar.dtype()),
        (&input_shape, operand_dtype),
        (scalar.shape(), operand_dtype),
        (&input_shape, quotient_dtype),
        (&input_shape, product_dtype),
        (&input_shape, output_dtype),
    ] {
        extent(shape, dtype)?;
    }
    if operand_dtype.is_integer() {
        for _ in 0..5 { extent(&input_shape, operand_dtype)?; }
        for _ in 0..5 { extent(&input_shape, DType::Bool)?; }
        let zero = TensorData::scalar_with_dtype(Scalar::I(0), operand_dtype);
        let one = TensorData::scalar_with_dtype(Scalar::I(1), operand_dtype);
        extent(zero.shape(), zero.dtype())?;
        extent(one.shape(), one.dtype())?;
        if zero.shape() != &Shape::new([]) || one.shape() != &Shape::new([])
            || zero.dtype() != operand_dtype || one.dtype() != operand_dtype {
            return Err(Error::InvalidElementwiseDType { op: "mod floor_div scalar promotion", actual: operand_dtype });
        }
    } else {
        extent(&input_shape, floor_dividend_dtype)?;
        extent(scalar.shape(), reciprocal_dtype)?;
        if unary_dtype(UnaryOp::Reciprocal, operand_dtype) != reciprocal_dtype
            || source_lub(floor_dividend_dtype, reciprocal_dtype) != quotient_dtype {
            return Err(Error::InvalidElementwiseDType { op: "mod scalar promotion", actual: output_dtype });
        }
    }
    if scalar.shape() != &Shape::new([]) || scalar.dtype() != scalar_dtype
        || scalar_dtype != operand_dtype || source_lub(input_dtype, scalar_dtype) != operand_dtype
        || source_lub(quotient_dtype, operand_dtype) != product_dtype
        || source_lub(operand_dtype, product_dtype) != output_dtype
        || input_shape.broadcast_with(scalar.shape())? != input_shape {
        return Err(Error::InvalidElementwiseDType { op: "mod scalar promotion", actual: output_dtype });
    }
    Ok(ModuloScalarPlan { output_shape: input_shape, output_dtype, scalar })
}

fn fmod_scalar_plan(graph: &Graph, input: NodeId, value: Scalar) -> Result<FmodScalarPlan> {
    let input_node = graph.node(input)?;
    let input_shape = input_node.shape.clone();
    let input_dtype = input_node.dtype;
    let scalar_dtype = source_weak_scalar_dtype(input_dtype, value);
    let operand_dtype = source_lub(input_dtype, scalar_dtype);
    let trunc_dividend_dtype = if operand_dtype.is_float() || operand_dtype.is_integer() {
        operand_dtype
    } else {
        DType::F32
    };
    let reciprocal_dtype = unary_dtype(UnaryOp::Reciprocal, operand_dtype);
    let quotient_dtype = if operand_dtype.is_integer() {
        operand_dtype
    } else {
        source_lub(trunc_dividend_dtype, reciprocal_dtype)
    };
    let product_dtype = source_lub(quotient_dtype, operand_dtype);
    let output_dtype = source_lub(operand_dtype, product_dtype);
    let scalar = TensorData::scalar_with_dtype(value, scalar_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    // Preflight the committed operands, all delegated trunc_div branches,
    // product, and source-literal subtraction before publishing the scalar.
    for (shape, dtype) in [
        (&input_shape, input_dtype),
        (scalar.shape(), scalar.dtype()),
        (&input_shape, operand_dtype),
        (scalar.shape(), operand_dtype),
        (&input_shape, quotient_dtype),
        (&input_shape, product_dtype),
        (&input_shape, output_dtype),
    ] {
        extent(shape, dtype)?;
    }
    if operand_dtype.is_integer() {
        for _ in 0..5 {
            extent(&input_shape, operand_dtype)?;
        }
        for _ in 0..5 {
            extent(&input_shape, DType::Bool)?;
        }
        let zero = TensorData::scalar_with_dtype(Scalar::I(0), operand_dtype);
        let one = TensorData::scalar_with_dtype(Scalar::I(1), operand_dtype);
        extent(zero.shape(), zero.dtype())?;
        extent(one.shape(), one.dtype())?;
        if zero.shape() != &Shape::new([])
            || one.shape() != &Shape::new([])
            || zero.dtype() != operand_dtype
            || one.dtype() != operand_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "fmod trunc_div scalar promotion",
                actual: operand_dtype,
            });
        }
    } else {
        extent(&input_shape, trunc_dividend_dtype)?;
        extent(scalar.shape(), reciprocal_dtype)?;
        if unary_dtype(UnaryOp::Reciprocal, operand_dtype) != reciprocal_dtype
            || source_lub(trunc_dividend_dtype, reciprocal_dtype) != quotient_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "fmod scalar promotion",
                actual: output_dtype,
            });
        }
    }
    if scalar.shape() != &Shape::new([])
        || scalar.dtype() != scalar_dtype
        || scalar_dtype != operand_dtype
        || source_lub(input_dtype, scalar_dtype) != operand_dtype
        || source_lub(quotient_dtype, operand_dtype) != product_dtype
        || source_lub(operand_dtype, product_dtype) != output_dtype
        || input_shape.broadcast_with(scalar.shape())? != input_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "fmod scalar promotion",
            actual: output_dtype,
        });
    }
    Ok(FmodScalarPlan { output_shape: input_shape, output_dtype, scalar })
}

fn bitwise_not_plan(graph: &Graph, input: NodeId) -> Result<BitwiseNotPlan> {
    let source = graph.node(input)?;
    let shape = source.shape.clone();
    let dtype = source.dtype;
    let value = match dtype {
        DType::Bool => Scalar::Bool(true),
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => Scalar::I(-1),
        DType::U8 => Scalar::U(u8::MAX.into()),
        DType::U16 => Scalar::U(u16::MAX.into()),
        DType::U32 => Scalar::U(u32::MAX.into()),
        DType::U64 => Scalar::U(u64::MAX),
        _ => {
            return Err(Error::InvalidElementwiseDType {
                op: "bitwise_not",
                actual: dtype,
            });
        }
    };
    let mask = TensorData::scalar_with_dtype(value, dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };

    // Validate the input, scalar, literal broadcast, and output before a
    // constant, cast, comparison, or XOR can be appended to the graph.
    extent(&shape, dtype)?;
    extent(mask.shape(), mask.dtype())?;
    if mask.dtype() != dtype || shape.broadcast_with(mask.shape())? != shape {
        return Err(Error::InvalidElementwiseDType {
            op: "bitwise_not scalar promotion",
            actual: dtype,
        });
    }
    extent(&shape, dtype)?;

    Ok(BitwiseNotPlan { shape, dtype, mask })
}

struct HardsigmoidPlan {
    product_shape: Shape,
    product_dtype: DType,
    output_shape: Shape,
    output_dtype: DType,
    zero: TensorData,
    one: TensorData,
}

/// Scalar/default wrapper around the live Hardsigmoid descriptor. tinygrad's
/// public alpha and beta are Python floats: alpha first commits against x for
/// the left-multiply, then beta commits against that stored product.
struct HardsigmoidScalarPlan {
    core: HardsigmoidPlan,
    alpha: TensorData,
    beta: TensorData,
}

struct LeakyReluPlan {
    shape: Shape,
    dtype: DType,
    zero: TensorData,
}

/// Complete descriptor and weak-scalar commitment for tinygrad's
/// `Tensor.leaky_relu(neg_slope=...)` convenience form. The live-slope
/// surface retains [`LeakyReluPlan`] directly; this wrapper owns only the
/// Python concrete scalar that source commits at the multiplication's input
/// width.
struct LeakyReluScalarPlan {
    core: LeakyReluPlan,
    slope: TensorData,
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

/// Descriptor-only wrapper for tinygrad's `Tensor.celu(alpha=...)` scalar
/// convenience form. The embedded plan remains the live-alpha contract; this
/// layer proves the Python float's source-width commitment before publishing
/// its constant.
struct CeluScalarPlan {
    core: CeluPlan,
    alpha: TensorData,
}

/// Fully resolved literal tinygrad ELU graph with live alpha.
struct EluPlan {
    exp_dtype: DType,
    positive_shape: Shape,
    negative_shape: Shape,
    scaled_shape: Shape,
    scaled_dtype: DType,
    output_shape: Shape,
    output_dtype: DType,
    zero_input: TensorData,
    one_exp: TensorData,
    zero_exp: TensorData,
}

/// Scalar/default wrapper around [`EluPlan`].
struct EluScalarPlan {
    core: EluPlan,
    alpha: TensorData,
}

struct SeluPlan {
    exp_dtype: DType,
    condition_shape: Shape,
    negative_shape: Shape,
    negative_dtype: DType,
    branch_shape: Shape,
    branch_dtype: DType,
    output_shape: Shape,
    output_dtype: DType,
    zero_input: TensorData,
    one_exp: TensorData,
}

struct SeluScalarPlan {
    core: SeluPlan,
    alpha: TensorData,
    gamma: TensorData,
}

struct ClampStagePlan { bound: NodeId, shape: Shape, dtype: DType }
struct ClampPlan { lower: Option<ClampStagePlan>, upper: Option<ClampStagePlan>, output_shape: Shape, output_dtype: DType }

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

/// Fully resolved literal tinygrad ReLU descriptor.  The public helper is a
/// strict ordered comparison followed by WHERE, rather than the raw unary
/// primitive that remains available to lower-level callers.
struct ReluPlan {
    shape: Shape,
    dtype: DType,
    zero: TensorData,
}

/// The complete descriptor and weak-scalar commitment plan for tinygrad's
/// public `Tensor.allclose`. The tolerance literals remain local to this
/// helper: they are weak Python floats in source and commit at the dtype of
/// `other.abs()`, not necessarily at the dtype of `self - other`.
struct AllclosePlan {
    output_shape: Shape,
}

/// Descriptor-only contract for the scalar/default form of checked-in
/// `Tensor.isclose`. The two Python float literals are weak independently:
/// both commit at the `other.abs()` tolerance branch, not at the subtraction
/// branch's potentially wider dtype.
struct IscloseScalarPlan {
    output_shape: Shape,
    difference_dtype: DType,
    tolerance_dtype: DType,
    comparison_dtype: DType,
    rtol: TensorData,
    atol: TensorData,
    equal_nan: TensorData,
}

/// The concrete public `Tensor.logaddexp` graph is not a raw binary ALU: the
/// two source operands first commit to one LUB storage dtype, then that shared
/// pair feeds Max and both centered Exp paths. Retaining the casted values
/// matters at narrow rounding boundaries as well as for graph structure.
struct LogaddexpPlan {
    shape: Shape,
    operand_dtype: DType,
    exp_dtype: DType,
    output_dtype: DType,
}

/// Scalar-right wrapper for [`LogaddexpPlan`]. The weak Python scalar stays
/// as payload data until the complete stable composite plan is known to fit.
struct LogaddexpScalarPlan {
    core: LogaddexpPlan,
    lhs_dtype: DType,
    scalar: TensorData,
}

/// Public tinygrad `log10` is `log2() * math.log10(2)`. The Python float is
/// weak and therefore commits at the Log2 result storage width, rather than
/// being an unconditional F32 literal.
struct Log10Plan {
    shape: Shape,
    log_dtype: DType,
    scale: TensorData,
}

/// Tinygrad defines LogSigmoid as `-(-x).softplus()`, not as an eager
/// `-log(1 + exp(-x))`. This plan proves the nested default-beta Softplus
/// construction and final Neg before it can publish the first inner node.
struct LogsigmoidPlan {
    shape: Shape,
    negated_dtype: DType,
    output_dtype: DType,
    beta: TensorData,
    softplus_zero: TensorData,
    softplus_one: TensorData,
}

/// Complete literal descriptor for tinygrad's public `Tensor.copysign`.
///
/// Source does not use a host copysign primitive: it first commits both
/// operands through `_broadcasted`, then detects a negative sign with the
/// ordered pair `(b < 0) | (b.reciprocal() < 0)`, and finally selects between
/// `-abs(a)` and `abs(a)`.  In particular, that second predicate distinguishes
/// negative zero, while unordered NaNs select the positive magnitude.
struct CopysignPlan {
    magnitude_shape: Shape,
    sign_shape: Shape,
    output_shape: Shape,
    operand_dtype: DType,
    reciprocal_dtype: DType,
    operand_zero: TensorData,
    reciprocal_zero: TensorData,
}

/// Scalar-right wrapper for [`CopysignPlan`]. The scalar is held as payload
/// data until the complete literal copysign plan has passed, preserving
/// tinygrad's weak-constant storage boundary (including negative zero).
struct CopysignScalarPlan {
    core: CopysignPlan,
    magnitude_dtype: DType,
    scalar: TensorData,
}

/// Fully resolved public `Tensor.lerp` descriptor. Tinygrad ordinarily uses
/// `start + (end - start) * weight`, but has a separate fixed-point path when
/// the start value is a live U8 tensor. The latter is intentionally planned
/// here rather than approximated by the ordinary float/integer expression.
struct LerpPlan {
    special_u8: bool,
    output_shape: Shape,
    output_dtype: DType,
    difference_dtype: DType,
    weighted_dtype: DType,
    weight_scale_dtype: DType,
    weight_fraction_dtype: DType,
    scale: Option<TensorData>,
    half: Option<TensorData>,
    rounding: Option<TensorData>,
    shift: Option<TensorData>,
}

/// Scalar-right `Tensor.lerp` plan. tinygrad's U8 fixed-point branch applies
/// only to a live Tensor weight; a Python scalar always follows the ordinary
/// `start + (end - start) * weight` composition.
struct LerpScalarPlan {
    difference_shape: Shape,
    difference_dtype: DType,
    weighted_shape: Shape,
    weighted_dtype: DType,
    output_shape: Shape,
    output_dtype: DType,
    scalar: TensorData,
}

/// tinygrad's source LUB, including its default-F32 bridge for the concrete
/// I64/U64 pair that RustGrad's raw storage promotion represents as F64.
fn source_lub(lhs: DType, rhs: DType) -> DType {
    if matches!((lhs, rhs), (DType::I64, DType::U64) | (DType::U64, DType::I64)) {
        DType::F32
    } else {
        lhs.promote(rhs)
    }
}

fn lerp_plan(
    start_shape: &Shape,
    start_dtype: DType,
    end_shape: &Shape,
    end_dtype: DType,
    weight_shape: &Shape,
    weight_dtype: DType,
) -> Result<LerpPlan> {
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    let difference_shape = end_shape.broadcast_with(start_shape)?;
    let difference_dtype = source_lub(end_dtype, start_dtype);
    let weighted_shape = difference_shape.broadcast_with(weight_shape)?;
    let weighted_dtype = source_lub(difference_dtype, weight_dtype);
    let output_shape = start_shape.broadcast_with(&weighted_shape)?;

    for (shape, dtype) in [
        (start_shape, start_dtype),
        (end_shape, end_dtype),
        (weight_shape, weight_dtype),
        (&difference_shape, difference_dtype),
        (&weighted_shape, weighted_dtype),
    ] {
        extent(shape, dtype)?;
    }

    if start_dtype != DType::U8 {
        let output_dtype = source_lub(start_dtype, weighted_dtype);
        extent(&output_shape, output_dtype)?;
        return Ok(LerpPlan {
            special_u8: false,
            output_shape,
            output_dtype,
            difference_dtype,
            weighted_dtype,
            weight_scale_dtype: weight_dtype,
            weight_fraction_dtype: weight_dtype,
            scale: None,
            half: None,
            rounding: None,
            shift: None,
        });
    }

    // tinygrad's U8/tensor-weight branch is:
    // `(start + (((end-start).cast(I8) * w_i + 64).cast(U16) >> 7)).cast(U8)`
    // where `w_i = (weight * 128 + .5).cast(I16)`. Weak 128 commits at the
    // live weight width (Bool first resolves to default I32); weak .5 then
    // commits at that float width or the default F32 after integer arithmetic.
    let weight_scale_dtype = if weight_dtype == DType::Bool {
        DType::I32
    } else {
        weight_dtype
    };
    let weight_fraction_dtype = if weight_dtype.is_float() {
        weight_dtype
    } else {
        DType::F32
    };
    let fixed_difference_dtype = DType::I8;
    let fixed_weight_dtype = DType::I16;
    let fixed_accumulator_dtype = DType::U16;
    let fixed_shape = difference_shape.broadcast_with(weight_shape)?;
    let special_output_shape = start_shape.broadcast_with(&fixed_shape)?;
    for (shape, dtype) in [
        (&difference_shape, fixed_difference_dtype),
        (weight_shape, weight_scale_dtype),
        (weight_shape, weight_scale_dtype),
        (weight_shape, weight_fraction_dtype),
        (weight_shape, fixed_weight_dtype),
        (&fixed_shape, fixed_weight_dtype),
        (&fixed_shape, fixed_accumulator_dtype),
        (&fixed_shape, fixed_accumulator_dtype),
        (&special_output_shape, fixed_accumulator_dtype),
        (&special_output_shape, DType::U8),
    ] {
        extent(shape, dtype)?;
    }
    let scale = TensorData::scalar_with_dtype(Scalar::I(128), weight_scale_dtype);
    let half = TensorData::scalar_with_dtype(Scalar::F(0.5), weight_fraction_dtype);
    let rounding = TensorData::scalar_with_dtype(Scalar::I(64), fixed_weight_dtype);
    let shift = TensorData::scalar_with_dtype(Scalar::I(7), fixed_accumulator_dtype);
    for scalar in [&scale, &half, &rounding, &shift] {
        extent(scalar.shape(), scalar.dtype())?;
    }
    if difference_dtype != source_lub(end_dtype, start_dtype)
        || weight_scale_dtype.promote(scale.dtype()) != weight_scale_dtype
        || weight_fraction_dtype.promote(half.dtype()) != weight_fraction_dtype
        || fixed_weight_dtype.promote(rounding.dtype()) != fixed_weight_dtype
        || fixed_accumulator_dtype.promote(shift.dtype()) != fixed_accumulator_dtype
        || weight_shape.broadcast_with(scale.shape())? != *weight_shape
        || weight_shape.broadcast_with(half.shape())? != *weight_shape
        || fixed_shape.broadcast_with(rounding.shape())? != fixed_shape
        || fixed_shape.broadcast_with(shift.shape())? != fixed_shape
        || source_lub(fixed_difference_dtype, fixed_weight_dtype) != fixed_weight_dtype
        || source_lub(start_dtype, fixed_accumulator_dtype) != fixed_accumulator_dtype
        || special_output_shape != output_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "lerp U8 fixed-point source promotion",
            actual: weight_fraction_dtype,
        });
    }
    Ok(LerpPlan {
        special_u8: true,
        output_shape,
        output_dtype: DType::U8,
        difference_dtype,
        weighted_dtype: fixed_weight_dtype,
        weight_scale_dtype,
        weight_fraction_dtype,
        scale: Some(scale),
        half: Some(half),
        rounding: Some(rounding),
        shift: Some(shift),
    })
}

fn lerp_scalar_plan(graph: &Graph, start: NodeId, end: NodeId, weight: Scalar) -> Result<LerpScalarPlan> {
    let start_node = graph.node(start)?;
    let start_shape = start_node.shape.clone();
    let start_dtype = start_node.dtype;
    let end_node = graph.node(end)?;
    let end_shape = end_node.shape.clone();
    let end_dtype = end_node.dtype;
    let difference_shape = end_shape.broadcast_with(&start_shape)?;
    let difference_dtype = source_lub(end_dtype, start_dtype);
    // The Python scalar is weak at the multiplication consumer, not at the
    // original start tensor. This is what deliberately keeps U8/scalar out
    // of tinygrad's live-Tensor fixed-point branch.
    let weight_dtype = source_weak_scalar_dtype(difference_dtype, weight);
    let weight_shape = Shape::new([]);
    let weighted_shape = difference_shape.broadcast_with(&weight_shape)?;
    let weighted_dtype = source_lub(difference_dtype, weight_dtype);
    let output_shape = start_shape.broadcast_with(&weighted_shape)?;
    let output_dtype = source_lub(start_dtype, weighted_dtype);
    let scalar = TensorData::scalar_with_dtype(weight, weight_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };

    // Validate original storage, both `_broadcasted` stages, the Bool
    // subtraction-negation special case, and the final add before publishing
    // the scalar or any ordinary lerp node.
    for (shape, dtype) in [
        (&start_shape, start_dtype),
        (&end_shape, end_dtype),
        (scalar.shape(), scalar.dtype()),
        (&start_shape, difference_dtype),
        (&end_shape, difference_dtype),
        (&difference_shape, difference_dtype),
        (&difference_shape, weighted_dtype),
        (scalar.shape(), weighted_dtype),
        (&weighted_shape, weighted_dtype),
        (&start_shape, output_dtype),
        (&weighted_shape, output_dtype),
        (&output_shape, output_dtype),
    ] {
        extent(shape, dtype)?;
    }
    if difference_dtype == DType::Bool {
        extent(&end_shape, DType::Bool)?;
        extent(&difference_shape, DType::Bool)?;
    }
    if scalar.shape() != &weight_shape
        || scalar.dtype() != weight_dtype
        || difference_dtype != source_lub(end_dtype, start_dtype)
        || weighted_dtype != source_lub(difference_dtype, weight_dtype)
        || output_dtype != source_lub(start_dtype, weighted_dtype)
        || difference_shape.broadcast_with(scalar.shape())? != weighted_shape
        || start_shape.broadcast_with(&weighted_shape)? != output_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "lerp scalar source promotion",
            actual: output_dtype,
        });
    }
    Ok(LerpScalarPlan {
        difference_shape,
        difference_dtype,
        weighted_shape,
        weighted_dtype,
        output_shape,
        output_dtype,
        scalar,
    })
}

fn copysign_plan(
    magnitude_shape: &Shape,
    magnitude_dtype: DType,
    sign_shape: &Shape,
    sign_dtype: DType,
) -> Result<CopysignPlan> {
    // `_broadcasted` uses tinygrad's default-float join for the otherwise
    // unrepresentable I64/U64 pair. Keep that source boundary local to this
    // public composition rather than changing raw BinaryOp::Copysign.
    let operand_dtype = if matches!(
        (magnitude_dtype, sign_dtype),
        (DType::I64, DType::U64) | (DType::U64, DType::I64)
    ) {
        DType::F32
    } else {
        magnitude_dtype.promote(sign_dtype)
    };
    let reciprocal_dtype = unary_dtype(UnaryOp::Reciprocal, operand_dtype);
    let output_shape = magnitude_shape.broadcast_with(sign_shape)?;
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };

    // Original inputs, `_broadcasted` cast values, abs/neg magnitude branch,
    // both sign predicates, their Bool OR, the nonfloat reciprocal cast, and
    // the final three-way WHERE all need valid concrete descriptors before
    // this helper publishes a constant or a node.
    extent(magnitude_shape, magnitude_dtype)?;
    extent(sign_shape, sign_dtype)?;
    extent(magnitude_shape, operand_dtype)?;
    extent(sign_shape, operand_dtype)?;
    extent(magnitude_shape, operand_dtype)?; // abs
    extent(magnitude_shape, operand_dtype)?; // neg(abs)
    extent(sign_shape, DType::Bool)?; // b < 0
    if !operand_dtype.is_float() {
        extent(sign_shape, DType::F32)?; // reciprocal's explicit source cast
    }
    extent(sign_shape, reciprocal_dtype)?;
    extent(sign_shape, DType::Bool)?; // reciprocal < 0
    extent(sign_shape, DType::Bool)?; // OR
    extent(&output_shape, DType::Bool)?; // broadcast condition
    extent(&output_shape, operand_dtype)?; // final select

    let operand_zero = TensorData::scalar_with_dtype(Scalar::I(0), operand_dtype);
    let reciprocal_zero = TensorData::scalar_with_dtype(Scalar::I(0), reciprocal_dtype);
    for scalar in [&operand_zero, &reciprocal_zero] {
        extent(scalar.shape(), scalar.dtype())?;
    }
    if operand_zero.dtype() != operand_dtype
        || reciprocal_zero.dtype() != reciprocal_dtype
        || magnitude_dtype.promote(operand_dtype) != operand_dtype
        || sign_dtype.promote(operand_dtype) != operand_dtype
        || operand_dtype.promote(operand_zero.dtype()) != operand_dtype
        || reciprocal_dtype.promote(reciprocal_zero.dtype()) != reciprocal_dtype
        || unary_dtype(UnaryOp::Sign, operand_dtype) != operand_dtype
        || unary_dtype(UnaryOp::Neg, operand_dtype) != operand_dtype
        || (!operand_dtype.is_float() && reciprocal_dtype != DType::F32)
        || (operand_dtype.is_float() && reciprocal_dtype != operand_dtype)
        || sign_shape.broadcast_with(operand_zero.shape())? != *sign_shape
        || sign_shape.broadcast_with(reciprocal_zero.shape())? != *sign_shape
        || sign_shape.broadcast_with(sign_shape)? != *sign_shape
        || sign_shape.broadcast_with(magnitude_shape)? != output_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "copysign source promotion",
            actual: operand_dtype,
        });
    }
    Ok(CopysignPlan {
        magnitude_shape: magnitude_shape.clone(),
        sign_shape: sign_shape.clone(),
        output_shape,
        operand_dtype,
        reciprocal_dtype,
        operand_zero,
        reciprocal_zero,
    })
}

fn copysign_scalar_plan(graph: &Graph, magnitude: NodeId, sign: Scalar) -> Result<CopysignScalarPlan> {
    let magnitude_node = graph.node(magnitude)?;
    let magnitude_shape = magnitude_node.shape.clone();
    let magnitude_dtype = magnitude_node.dtype;
    let sign_dtype = source_weak_scalar_dtype(magnitude_dtype, sign);
    let sign_shape = Shape::new([]);
    // The scalar payload is deliberately constructed but not published until
    // the live copysign descriptor has validated every cast, predicate,
    // reciprocal, branch, and selected output extent.
    let scalar = TensorData::scalar_with_dtype(sign, sign_dtype);
    let core = copysign_plan(&magnitude_shape, magnitude_dtype, &sign_shape, sign_dtype)?;
    if scalar.shape() != &sign_shape || scalar.dtype() != sign_dtype {
        return Err(Error::InvalidElementwiseDType {
            op: "copysign scalar promotion",
            actual: sign_dtype,
        });
    }
    Ok(CopysignScalarPlan {
        core,
        magnitude_dtype,
        scalar,
    })
}

fn logsigmoid_plan(input_shape: &Shape, input_dtype: DType) -> Result<LogsigmoidPlan> {
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
            .map(|_| ())
    };
    let negated_dtype = unary_dtype(UnaryOp::Neg, input_dtype);
    let beta_dtype = if negated_dtype.is_float() {
        negated_dtype
    } else {
        DType::F32
    };
    let scaled_dtype = source_promote(negated_dtype, beta_dtype);
    let log_dtype = if scaled_dtype.is_float() { scaled_dtype } else { DType::F32 };
    let inverse_dtype = if beta_dtype.is_float() { beta_dtype } else { DType::F32 };
    let output_dtype = source_promote(log_dtype, inverse_dtype);

    extent(input_shape, input_dtype)?;
    // Outer Neg, default-beta Softplus's scale/logaddexp/reciprocal/product,
    // and the final Neg all preserve one concrete elementwise shape.
    for dtype in [
        negated_dtype,
        scaled_dtype,
        log_dtype,
        log_dtype,
        log_dtype,
        inverse_dtype,
        output_dtype,
        output_dtype,
    ] {
        extent(input_shape, dtype)?;
    }
    let beta = TensorData::scalar_with_dtype(Scalar::F(1.0), beta_dtype);
    let softplus_zero = TensorData::scalar_with_dtype(Scalar::F(0.0), log_dtype);
    let softplus_one = TensorData::scalar_with_dtype(Scalar::F(1.0), inverse_dtype);
    for scalar in [&beta, &softplus_zero, &softplus_one] {
        extent(scalar.shape(), scalar.dtype())?;
    }
    if negated_dtype != input_dtype
        || beta.dtype() != beta_dtype
        || softplus_zero.dtype() != log_dtype
        || softplus_one.dtype() != inverse_dtype
        || input_shape.broadcast_with(beta.shape())? != *input_shape
        || input_shape.broadcast_with(softplus_zero.shape())? != *input_shape
        || input_shape.broadcast_with(softplus_one.shape())? != *input_shape
        || output_dtype != if input_dtype.is_float() { input_dtype } else { DType::F32 }
        || source_promote(negated_dtype, beta_dtype) != scaled_dtype
        || source_promote(log_dtype, inverse_dtype) != output_dtype
        || unary_dtype(UnaryOp::Reciprocal, inverse_dtype) != inverse_dtype
        || unary_dtype(UnaryOp::Neg, output_dtype) != output_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "logsigmoid source promotion",
            actual: output_dtype,
        });
    }
    Ok(LogsigmoidPlan {
        shape: input_shape.clone(),
        negated_dtype,
        output_dtype,
        beta,
        softplus_zero,
        softplus_one,
    })
}

fn log10_plan(input_shape: &Shape, input_dtype: DType) -> Result<Log10Plan> {
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    let log_dtype = unary_dtype(UnaryOp::Log2, input_dtype);
    extent(input_shape, input_dtype)?;
    extent(input_shape, log_dtype)?; // Log2
    let scale = TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::LOG10_2), log_dtype);
    extent(scale.shape(), scale.dtype())?;
    let output_shape = input_shape.broadcast_with(scale.shape())?;
    extent(&output_shape, log_dtype)?;
    if (!input_dtype.is_float() && log_dtype != DType::F32)
        || (input_dtype.is_float() && log_dtype != input_dtype)
        || scale.dtype() != log_dtype
        || output_shape != *input_shape
        || log_dtype.promote(scale.dtype()) != log_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "log10 source scalar promotion",
            actual: log_dtype,
        });
    }
    Ok(Log10Plan {
        shape: input_shape.clone(),
        log_dtype,
        scale,
    })
}

fn logaddexp_plan(
    lhs_shape: &Shape,
    lhs_dtype: DType,
    rhs_shape: &Shape,
    rhs_dtype: DType,
) -> Result<LogaddexpPlan> {
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    extent(lhs_shape, lhs_dtype)?;
    extent(rhs_shape, rhs_dtype)?;
    let shape = lhs_shape.broadcast_with(rhs_shape)?;
    let operand_dtype = source_lub(lhs_dtype, rhs_dtype);
    let exp_dtype = unary_dtype(UnaryOp::Exp, operand_dtype);
    let output_dtype = source_lub(exp_dtype, operand_dtype);

    // Source `_broadcasted` casts each operand once, then reuses those typed
    // values for Max and the two ordered subtracts. Exp and Log retain the
    // source float storage dtype, including the nonfloat-to-F32 lift.
    extent(lhs_shape, operand_dtype)?;
    extent(rhs_shape, operand_dtype)?;
    for dtype in [operand_dtype, operand_dtype, operand_dtype, exp_dtype, exp_dtype, exp_dtype, exp_dtype, output_dtype] {
        extent(&shape, dtype)?;
    }
    if (!operand_dtype.is_float() && exp_dtype != DType::F32)
        || (operand_dtype.is_float() && exp_dtype != operand_dtype)
        || output_dtype != exp_dtype
        || source_lub(lhs_dtype, rhs_dtype) != operand_dtype
        || source_lub(exp_dtype, operand_dtype) != output_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "logaddexp source promotion",
            actual: output_dtype,
        });
    }
    Ok(LogaddexpPlan {
        shape,
        operand_dtype,
        exp_dtype,
        output_dtype,
    })
}

fn logaddexp_scalar_plan(graph: &Graph, lhs: NodeId, rhs: Scalar) -> Result<LogaddexpScalarPlan> {
    let lhs_node = graph.node(lhs)?;
    let lhs_shape = lhs_node.shape.clone();
    let lhs_dtype = lhs_node.dtype;
    let rhs_dtype = source_weak_scalar_dtype(lhs_dtype, rhs);
    let rhs_shape = Shape::new([]);
    // Constructing TensorData is non-publishing; the live stable composite
    // plan below verifies the scalar bytes alongside every source cast and
    // Max/Sub/Exp/Add/Log/final-Add descriptor before a graph node exists.
    let scalar = TensorData::scalar_with_dtype(rhs, rhs_dtype);
    let core = logaddexp_plan(&lhs_shape, lhs_dtype, &rhs_shape, rhs_dtype)?;
    if scalar.shape() != &rhs_shape || scalar.dtype() != rhs_dtype {
        return Err(Error::InvalidElementwiseDType {
            op: "logaddexp scalar promotion",
            actual: rhs_dtype,
        });
    }
    Ok(LogaddexpScalarPlan { core, lhs_dtype, scalar })
}

fn allclose_plan(
    lhs_shape: &Shape,
    lhs_dtype: DType,
    rhs_shape: &Shape,
    rhs_dtype: DType,
    rtol: f64,
    atol: f64,
) -> Result<AllclosePlan> {
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    extent(lhs_shape, lhs_dtype)?;
    extent(rhs_shape, rhs_dtype)?;

    // `self - other` promotes both operands, whereas tinygrad independently
    // computes `other.abs()` before multiplying it by the weak `rtol` float.
    let output_shape = lhs_shape.broadcast_with(rhs_shape)?;
    let difference_dtype = source_lub(lhs_dtype, rhs_dtype);
    extent(&output_shape, difference_dtype)?; // subtraction
    extent(&output_shape, difference_dtype)?; // absolute difference
    extent(rhs_shape, rhs_dtype)?; // other.abs()

    let tolerance_dtype = if rhs_dtype.is_float() { rhs_dtype } else { DType::F32 };
    let rtol = TensorData::scalar_with_dtype(Scalar::F(rtol), tolerance_dtype);
    let atol = TensorData::scalar_with_dtype(Scalar::F(atol), tolerance_dtype);
    extent(rtol.shape(), rtol.dtype())?;
    extent(atol.shape(), atol.dtype())?;
    let relative_shape = rhs_shape.broadcast_with(rtol.shape())?;
    let tolerance_shape = relative_shape.broadcast_with(atol.shape())?;
    extent(&relative_shape, tolerance_dtype)?;
    extent(&tolerance_shape, tolerance_dtype)?;

    let near_shape = output_shape.broadcast_with(&tolerance_shape)?;
    let comparison_dtype = source_lub(difference_dtype, tolerance_dtype);
    extent(&near_shape, comparison_dtype)?;
    extent(&near_shape, DType::Bool)?;
    // Source emits isfinite, isinf, and isnan for both operands. Verify each
    // predicate descriptor independently, then every Bool combination.
    for _ in 0..3 {
        extent(lhs_shape, DType::Bool)?;
        extent(rhs_shape, DType::Bool)?;
    }
    for _ in 0..9 {
        extent(&output_shape, DType::Bool)?;
    }
    let reduced_shape = Shape::new([]);
    extent(&reduced_shape, DType::Bool)?;

    if rtol.dtype() != tolerance_dtype
        || atol.dtype() != tolerance_dtype
        || relative_shape != *rhs_shape
        || tolerance_shape != *rhs_shape
        || near_shape != output_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "allclose scalar promotion",
            actual: tolerance_dtype,
        });
    }

    Ok(AllclosePlan {
        output_shape,
    })
}

fn isclose_scalar_plan(
    lhs_shape: &Shape,
    lhs_dtype: DType,
    rhs_shape: &Shape,
    rhs_dtype: DType,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> Result<IscloseScalarPlan> {
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    let source_lub = |lhs: DType, rhs: DType| {
        if matches!((lhs, rhs), (DType::I64, DType::U64) | (DType::U64, DType::I64)) {
            DType::F32
        } else {
            lhs.promote(rhs)
        }
    };
    let output_shape = lhs_shape.broadcast_with(rhs_shape)?;
    let difference_dtype = source_lub(lhs_dtype, rhs_dtype);
    // `other.abs()` retains other storage; the first weak float commits here.
    let tolerance_dtype = if rhs_dtype.is_float() { rhs_dtype } else { DType::F32 };
    let comparison_dtype = source_lub(difference_dtype, tolerance_dtype);
    extent(lhs_shape, lhs_dtype)?;
    extent(rhs_shape, rhs_dtype)?;
    for dtype in [difference_dtype, difference_dtype] {
        extent(&output_shape, dtype)?; // subtraction and abs difference
    }
    extent(rhs_shape, rhs_dtype)?; // other.abs()
    extent(rhs_shape, tolerance_dtype)?; // rtol * other.abs()
    extent(rhs_shape, tolerance_dtype)?; // atol + relative
    extent(&output_shape, comparison_dtype)?; // comparison's typed operands
    extent(&output_shape, DType::Bool)?; // near
    // isfinite/isinf/isnan for both inputs plus their source Boolean tree.
    for _ in 0..3 {
        extent(lhs_shape, DType::Bool)?;
        extent(rhs_shape, DType::Bool)?;
    }
    for _ in 0..10 {
        extent(&output_shape, DType::Bool)?;
    }

    let rtol = TensorData::scalar_with_dtype(Scalar::F(rtol), tolerance_dtype);
    let atol = TensorData::scalar_with_dtype(Scalar::F(atol), tolerance_dtype);
    let equal_nan = TensorData::scalar_with_dtype(Scalar::Bool(equal_nan), DType::Bool);
    for scalar in [&rtol, &atol, &equal_nan] {
        extent(scalar.shape(), scalar.dtype())?;
    }
    if rtol.dtype() != tolerance_dtype
        || atol.dtype() != tolerance_dtype
        || equal_nan.dtype() != DType::Bool
        || rhs_shape.broadcast_with(rtol.shape())? != *rhs_shape
        || rhs_shape.broadcast_with(atol.shape())? != *rhs_shape
        || output_shape.broadcast_with(equal_nan.shape())? != output_shape
        || source_lub(lhs_dtype, rhs_dtype) != difference_dtype
        || source_lub(difference_dtype, tolerance_dtype) != comparison_dtype
        || source_lub(rhs_dtype, tolerance_dtype) != tolerance_dtype
        || source_lub(tolerance_dtype, tolerance_dtype) != tolerance_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "isclose scalar tolerance promotion",
            actual: tolerance_dtype,
        });
    }
    Ok(IscloseScalarPlan {
        output_shape,
        difference_dtype,
        tolerance_dtype,
        comparison_dtype,
        rtol,
        atol,
        equal_nan,
    })
}

fn relu_plan(input_shape: &Shape, input_dtype: DType) -> Result<ReluPlan> {
    // tinygrad spells ReLU as `(x > 0).where(x, 0)`.  The scalar zero is weak
    // at x's storage dtype, and every intermediate is concrete: x and zero
    // feed the ordered predicate, then the Bool predicate and both value
    // branches feed WHERE.  Prove all descriptors before publishing either
    // the scalar constant or an operation.
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(input_shape, input_dtype)?;

    let zero = TensorData::scalar_with_dtype(Scalar::I(0), input_dtype);
    extent(zero.shape(), zero.dtype())?;

    let predicate_shape = input_shape.broadcast_with(zero.shape())?;
    let comparison_dtype = input_dtype.promote(zero.dtype());
    let value_shape = input_shape.broadcast_with(zero.shape())?;
    let value_dtype = input_dtype.promote(zero.dtype());
    let output_shape = predicate_shape.broadcast_with(&value_shape)?;
    let output_dtype = value_dtype;
    extent(&predicate_shape, comparison_dtype)?;
    extent(&predicate_shape, DType::Bool)?;
    extent(&value_shape, value_dtype)?;
    extent(&output_shape, output_dtype)?;

    if zero.dtype() != input_dtype
        || predicate_shape != *input_shape
        || comparison_dtype != input_dtype
        || value_shape != *input_shape
        || value_dtype != input_dtype
        || output_shape != *input_shape
        || output_dtype != input_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "relu scalar promotion",
            actual: input_dtype,
        });
    }

    Ok(ReluPlan {
        shape: input_shape.clone(),
        dtype: input_dtype,
        zero,
    })
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

fn celu_scalar_plan(
    input_shape: &Shape,
    input_dtype: DType,
    alpha: f64,
) -> Result<CeluScalarPlan> {
    // `alpha` is a weak Python float in both of CELU's source branches. It
    // commits to float input storage, while exact input storage first joins
    // weakfloat at tinygrad's default F32 width.
    let alpha_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    let alpha = TensorData::scalar_with_dtype(Scalar::F(alpha), alpha_dtype);
    let core = celu_plan(input_shape, input_dtype, alpha.shape(), alpha.dtype())?;
    let extent = alpha
        .shape()
        .numel()?
        .checked_mul(alpha.dtype().itemsize())
        .ok_or_else(|| Error::ShapeOverflow(alpha.shape().clone()))?;
    if extent != alpha.dtype().itemsize()
        || alpha.dtype() != alpha_dtype
        || input_shape.broadcast_with(alpha.shape())? != *input_shape
        || core.output_shape != *input_shape
        || core.output_dtype != alpha_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "celu scalar alpha promotion",
            actual: alpha_dtype,
        });
    }
    Ok(CeluScalarPlan { core, alpha })
}

fn elu_plan(
    input_shape: &Shape,
    input_dtype: DType,
    alpha_shape: &Shape,
    alpha_dtype: DType,
) -> Result<EluPlan> {
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    extent(input_shape, input_dtype)?;
    extent(alpha_shape, alpha_dtype)?;
    let exp_dtype = unary_dtype(UnaryOp::Exp, input_dtype);
    let positive_shape = input_shape.clone();
    let negative_shape = input_shape.clone();
    let scaled_shape = negative_shape.broadcast_with(alpha_shape)?;
    let scaled_dtype = exp_dtype.promote(alpha_dtype);
    let output_shape = positive_shape.broadcast_with(&scaled_shape)?;
    let output_dtype = input_dtype.promote(scaled_dtype);
    for (shape, dtype) in [
        (&positive_shape, input_dtype),
        (input_shape, exp_dtype),
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
        || input_shape.broadcast_with(one_exp.shape())? != *input_shape
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
    Ok(EluPlan {
        exp_dtype,
        positive_shape,
        negative_shape,
        scaled_shape,
        scaled_dtype,
        output_shape,
        output_dtype,
        zero_input,
        one_exp,
        zero_exp,
    })
}

fn elu_scalar_plan(
    input_shape: &Shape,
    input_dtype: DType,
    alpha: Scalar,
) -> Result<EluScalarPlan> {
    // The unchecked Python parameter is first consumed by `alpha *
    // relu(1-exp(x))`, so it commits against Exp's dtype, not directly
    // against x. This is observably different for nonfloat input tensors.
    let alpha_dtype = source_weak_scalar_dtype(unary_dtype(UnaryOp::Exp, input_dtype), alpha);
    let alpha = TensorData::scalar_with_dtype(alpha, alpha_dtype);
    let core = elu_plan(input_shape, input_dtype, alpha.shape(), alpha.dtype())?;
    let extent = alpha
        .shape()
        .numel()?
        .checked_mul(alpha.dtype().itemsize())
        .ok_or_else(|| Error::ShapeOverflow(alpha.shape().clone()))?;
    if extent != alpha.dtype().itemsize()
        || alpha.dtype() != alpha_dtype
        || input_shape.broadcast_with(alpha.shape())? != *input_shape
        || core.output_shape != *input_shape
        || core.output_dtype != alpha_dtype
    {
        return Err(Error::InvalidElementwiseDType {
            op: "elu scalar alpha promotion",
            actual: alpha_dtype,
        });
    }
    Ok(EluScalarPlan { core, alpha })
}

fn selu_plan(input_shape: &Shape, input_dtype: DType, alpha_shape: &Shape, alpha_dtype: DType, gamma_shape: &Shape, gamma_dtype: DType) -> Result<SeluPlan> {
    let extent = |shape: &Shape, dtype: DType| shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()));
    extent(input_shape, input_dtype)?;
    extent(alpha_shape, alpha_dtype)?;
    extent(gamma_shape, gamma_dtype)?;
    let exp_dtype = unary_dtype(UnaryOp::Exp, input_dtype);
    let condition_shape = input_shape.clone();
    let negative_raw_shape = input_shape.clone();
    let negative_shape = negative_raw_shape.broadcast_with(alpha_shape)?;
    let negative_dtype = exp_dtype.promote(alpha_dtype);
    let branch_shape = input_shape.broadcast_with(&negative_shape)?;
    let branch_dtype = input_dtype.promote(negative_dtype);
    let output_shape = branch_shape.broadcast_with(gamma_shape)?;
    let output_dtype = branch_dtype.promote(gamma_dtype);
    for (shape, dtype) in [(&condition_shape,DType::Bool),(input_shape,exp_dtype),(&negative_raw_shape,exp_dtype),(&negative_shape,negative_dtype),(&branch_shape,branch_dtype),(&output_shape,output_dtype)] { extent(shape,dtype)?; }
    let zero_input = TensorData::scalar_with_dtype(Scalar::I(0), input_dtype);
    let one_exp = TensorData::scalar_with_dtype(Scalar::I(1), exp_dtype);
    if zero_input.dtype()!=input_dtype || one_exp.dtype()!=exp_dtype || input_shape.broadcast_with(zero_input.shape())? != *input_shape || input_shape.broadcast_with(one_exp.shape())? != *input_shape || input_dtype.promote(zero_input.dtype())!=input_dtype || exp_dtype.promote(one_exp.dtype())!=exp_dtype || condition_shape.broadcast_with(&branch_shape)?!=branch_shape {
        return Err(Error::InvalidElementwiseDType { op:"selu scalar promotion", actual:output_dtype });
    }
    Ok(SeluPlan { exp_dtype, condition_shape, negative_shape, negative_dtype, branch_shape, branch_dtype, output_shape, output_dtype, zero_input, one_exp })
}

fn selu_scalar_plan(input_shape: &Shape, input_dtype: DType, alpha: f64, gamma: f64) -> Result<SeluScalarPlan> {
    let dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
    let alpha = TensorData::scalar_with_dtype(Scalar::F(alpha), dtype);
    let gamma = TensorData::scalar_with_dtype(Scalar::F(gamma), dtype);
    let core = selu_plan(input_shape,input_dtype,alpha.shape(),alpha.dtype(),gamma.shape(),gamma.dtype())?;
    for scalar in [&alpha,&gamma] { let bytes=scalar.shape().numel()?.checked_mul(scalar.dtype().itemsize()).ok_or_else(|| Error::ShapeOverflow(scalar.shape().clone()))?; if bytes != scalar.dtype().itemsize() || scalar.dtype()!=dtype || input_shape.broadcast_with(scalar.shape())? != *input_shape { return Err(Error::InvalidElementwiseDType { op:"selu scalar parameter promotion", actual:dtype }); } }
    if core.output_shape != *input_shape || core.output_dtype != dtype { return Err(Error::InvalidElementwiseDType { op:"selu scalar parameter promotion", actual:dtype }); }
    Ok(SeluScalarPlan { core, alpha, gamma })
}

fn clamp_plan(graph: &Graph, input: NodeId, min: Option<NodeId>, max: Option<NodeId>) -> Result<ClampPlan> {
    if min.is_none() && max.is_none() { return Err(Error::InvalidElementwiseDType { op:"clamp requires a bound", actual:graph.node(input)?.dtype }); }
    let source_lub = |a:DType,b:DType| if matches!((a,b),(DType::I64,DType::U64)|(DType::U64,DType::I64)) { DType::F32 } else { a.promote(b) };
    let extent = |shape:&Shape,dtype:DType| shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()));
    let input_node=graph.node(input)?; let mut shape=input_node.shape.clone(); let mut dtype=input_node.dtype; extent(&shape,dtype)?;
    let stage = |bound:NodeId, shape:&Shape, dtype:DType| -> Result<ClampStagePlan> { let node=graph.node(bound)?; extent(&node.shape,node.dtype)?; let shape=shape.broadcast_with(&node.shape)?; let dtype=source_lub(dtype,node.dtype); extent(&shape,dtype)?; extent(&shape,DType::Bool)?; Ok(ClampStagePlan { bound, shape, dtype }) };
    let lower=match min { Some(bound)=>{let s=stage(bound,&shape,dtype)?; shape=s.shape.clone(); dtype=s.dtype; Some(s)},None=>None};
    let upper=match max { Some(bound)=>{let s=stage(bound,&shape,dtype)?; shape=s.shape.clone(); dtype=s.dtype; Some(s)},None=>None};
    extent(&shape,dtype)?; Ok(ClampPlan { lower, upper, output_shape:shape, output_dtype:dtype })
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

fn leaky_relu_scalar_plan(
    input_shape: &Shape,
    input_dtype: DType,
    neg_slope: Scalar,
) -> Result<LeakyReluScalarPlan> {
    // The source parameter is intentionally untyped. Its literal
    // `neg_slope * self` therefore takes the same weak concrete-scalar path
    // as Tensor.__rmul__, including Bool/int constants and float widening.
    let slope_dtype = source_weak_scalar_dtype(input_dtype, neg_slope);
    let slope = TensorData::scalar_with_dtype(neg_slope, slope_dtype);
    let core = leaky_relu_plan(input_shape, input_dtype, slope.shape(), slope.dtype())?;
    let extent = slope
        .shape()
        .numel()?
        .checked_mul(slope.dtype().itemsize())
        .ok_or_else(|| Error::ShapeOverflow(slope.shape().clone()))?;
    if extent != slope.dtype().itemsize()
        || slope.dtype() != slope_dtype
        || input_shape.broadcast_with(slope.shape())? != *input_shape
        || core.dtype != slope_dtype
        || core.shape != *input_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "leaky_relu scalar slope promotion",
            actual: slope_dtype,
        });
    }
    Ok(LeakyReluScalarPlan { core, slope })
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

fn hardsigmoid_scalar_plan(
    graph: &Graph,
    input: NodeId,
    alpha: f64,
    beta: f64,
) -> Result<HardsigmoidScalarPlan> {
    let input_node = graph.node(input)?;
    let input_shape = input_node.shape.clone();
    let input_dtype = input_node.dtype;
    // alpha is the left Python float in `alpha * self`; for non-float x it
    // weak-promotes to tinygrad's default F32. Its product is then the live
    // reference which commits the subsequent `+ beta` Python float.
    let alpha_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
    let alpha = TensorData::scalar_with_dtype(Scalar::F(alpha), alpha_dtype);
    let product_dtype = source_lub(input_dtype, alpha_dtype);
    let beta_dtype = if product_dtype.is_float() { product_dtype } else { DType::F32 };
    let beta = TensorData::scalar_with_dtype(Scalar::F(beta), beta_dtype);
    let core = hardsigmoid_plan(
        &input_shape,
        input_dtype,
        alpha.shape(),
        alpha.dtype(),
        beta.shape(),
        beta.dtype(),
    )?;
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    // The live plan has already covered every Mul/Add/ReLU/select/sub stage;
    // prove both source floats and their source-order commitments before a
    // constant can be published.
    extent(alpha.shape(), alpha.dtype())?;
    extent(beta.shape(), beta.dtype())?;
    if alpha.shape() != &Shape::new([])
        || beta.shape() != &Shape::new([])
        || alpha.dtype() != alpha_dtype
        || beta.dtype() != beta_dtype
        || alpha_dtype != core.product_dtype
        || beta_dtype != core.output_dtype
        || product_dtype != core.product_dtype
        || source_lub(core.product_dtype, beta_dtype) != core.output_dtype
        || input_shape.broadcast_with(alpha.shape())? != input_shape
        || core.product_shape.broadcast_with(beta.shape())? != core.output_shape
    {
        return Err(Error::InvalidElementwiseDType {
            op: "hardsigmoid scalar promotion",
            actual: core.output_dtype,
        });
    }
    Ok(HardsigmoidScalarPlan { core, alpha, beta })
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

    fn add_scalar_with_order(
        &mut self,
        input: NodeId,
        value: Scalar,
        reverse: bool,
    ) -> Result<NodeId> {
        let plan = add_scalar_plan(self, input, value)?;
        // All fallible descriptor work has completed. The scalar is committed
        // once at tinygrad's weak width, and only `reverse` changes the raw
        // Add operand order used by `Tensor.__radd__`.
        let scalar = self.constant(plan.scalar);
        let input = if plan.input_dtype == plan.output_dtype {
            input
        } else {
            self.cast(input, plan.output_dtype)?
        };
        let output = if reverse {
            self.add(scalar, input)?
        } else {
            self.add(input, scalar)?
        };
        debug_assert_eq!(self.shape(output).expect("add scalar preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("add scalar preflighted"), plan.output_dtype);
        Ok(output)
    }

    /// Source-compatible `Tensor.add(Python_scalar)` form. The tensor stays
    /// on the left of the final storage-width Add.
    pub fn add_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.add_scalar_with_order(input, value, false)
    }

    /// Source-compatible reflected `Python_scalar + Tensor` form. It has the
    /// same weak scalar commitment as `add_scalar`, with only Add's operand
    /// order reversed.
    pub fn scalar_add(&mut self, value: Scalar, input: NodeId) -> Result<NodeId> {
        self.add_scalar_with_order(input, value, true)
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
            // `_broadcasted` has already committed both branches. Preserve
            // tinygrad's literal `a + (-b)` root instead of exposing raw Sub
            // semantics (notably for floating payload/VJP structure).
            let negated_rhs = self.neg(rhs)?;
            self.binary(BinaryOp::Add, lhs, negated_rhs)
        }
    }

    fn sub_scalar_with_order(
        &mut self,
        input: NodeId,
        value: Scalar,
        reverse: bool,
    ) -> Result<NodeId> {
        let plan = sub_scalar_plan(self, input, value)?;
        // The scalar is only published after the complete source `a + (-b)`
        // descriptor has passed. Reverse changes which promoted branch is
        // negated, matching tinygrad's `__rsub__` exactly.
        let scalar = self.constant(plan.scalar);
        let input = if plan.input_dtype == plan.output_dtype {
            input
        } else {
            self.cast(input, plan.output_dtype)?
        };
        let output = if reverse {
            self.sub(scalar, input)?
        } else {
            self.sub(input, scalar)?
        };
        debug_assert_eq!(self.shape(output).expect("sub scalar preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("sub scalar preflighted"), plan.output_dtype);
        Ok(output)
    }

    /// Source-compatible `Tensor.sub(Python_scalar)` form. The scalar is the
    /// right branch and is therefore negated before the final Add.
    pub fn sub_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.sub_scalar_with_order(input, value, false)
    }

    /// Source-compatible reflected `Python_scalar - Tensor` form. The live
    /// tensor is the right branch and is negated before the final Add.
    pub fn scalar_sub(&mut self, value: Scalar, input: NodeId) -> Result<NodeId> {
        self.sub_scalar_with_order(input, value, true)
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

    fn mul_scalar_with_order(
        &mut self,
        input: NodeId,
        value: Scalar,
        reverse: bool,
    ) -> Result<NodeId> {
        let plan = mul_scalar_plan(self, input, value)?;
        // Constants and any source-LUB cast are published only after the
        // complete descriptor passes; reverse changes only the MUL inputs.
        let scalar = self.constant(plan.scalar);
        let input = if plan.input_dtype == plan.output_dtype {
            input
        } else {
            self.cast(input, plan.output_dtype)?
        };
        let output = if reverse {
            self.mul(scalar, input)?
        } else {
            self.mul(input, scalar)?
        };
        debug_assert_eq!(self.shape(output).expect("mul scalar preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("mul scalar preflighted"), plan.output_dtype);
        Ok(output)
    }

    /// Source-compatible `Tensor.mul(Python_scalar)` form.
    pub fn mul_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.mul_scalar_with_order(input, value, false)
    }

    /// Source-compatible reflected `Python_scalar * Tensor` form.
    pub fn scalar_mul(&mut self, value: Scalar, input: NodeId) -> Result<NodeId> {
        self.mul_scalar_with_order(input, value, true)
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

    fn div_scalar_with_order(
        &mut self,
        input: NodeId,
        value: Scalar,
        reverse: bool,
    ) -> Result<NodeId> {
        let plan = div_scalar_plan(self, input, value)?;
        // The complete true-division descriptor is known before this weak
        // scalar becomes visible. Reuse Graph::div only for rounding_mode=None
        // so it retains its literal reciprocal-then-Mul graph.
        let scalar = self.constant(plan.scalar);
        let output = if reverse {
            self.div(scalar, input)?
        } else {
            self.div(input, scalar)?
        };
        debug_assert_eq!(self.shape(output).expect("div scalar preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("div scalar preflighted"), plan.output_dtype);
        Ok(output)
    }

    /// Source-compatible true-division `Tensor.div(Python_scalar)` form.
    /// This intentionally exposes no floor or truncating scalar variant.
    pub fn div_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.div_scalar_with_order(input, value, false)
    }

    /// Source-compatible reflected true-division `Python_scalar / Tensor`.
    pub fn scalar_div(&mut self, value: Scalar, input: NodeId) -> Result<NodeId> {
        self.div_scalar_with_order(input, value, true)
    }

    pub fn pow(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Pow, lhs, rhs)
    }
    pub fn maximum(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Maximum, lhs, rhs)
    }
    fn extrema_scalar(&mut self, op: BinaryOp, input: NodeId, value: Scalar) -> Result<NodeId> {
        debug_assert!(matches!(op, BinaryOp::Maximum | BinaryOp::Minimum));
        let plan = extrema_scalar_plan(self, input, value)?;
        // The pure plan above is the last fallible work. The scalar is only
        // published once all source-LUB, predicate, and result descriptors
        // are known to fit, then the existing extrema root preserves its
        // ordered lhs-payload and split-tie VJP semantics.
        let scalar = self.constant(plan.scalar);
        let input = if plan.input_dtype == plan.output_dtype {
            input
        } else {
            self.cast(input, plan.output_dtype)?
        };
        let output = match op {
            BinaryOp::Maximum => self.maximum(input, scalar)?,
            BinaryOp::Minimum => self.minimum(input, scalar)?,
            _ => unreachable!("extrema scalar plan only admits extrema"),
        };
        debug_assert_eq!(self.shape(output).expect("extrema scalar shape preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("extrema scalar dtype preflighted"), plan.output_dtype);
        Ok(output)
    }
    /// Source-compatible scalar-right form of tinygrad's
    /// `Tensor.maximum(x)`. The live tensor remains the ordered left operand.
    pub fn maximum_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.extrema_scalar(BinaryOp::Maximum, input, value)
    }
    pub fn minimum(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Minimum, lhs, rhs)
    }
    /// Source-compatible scalar-right form of tinygrad's
    /// `Tensor.minimum(x)`. There is intentionally no reflected scalar API:
    /// tinygrad exposes this method only with the tensor as lhs.
    pub fn minimum_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.extrema_scalar(BinaryOp::Minimum, input, value)
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

    fn floor_div_scalar_with_order(
        &mut self,
        input: NodeId,
        value: Scalar,
        reverse: bool,
    ) -> Result<NodeId> {
        let plan = floor_div_scalar_plan(self, input, value)?;
        // Publish the weak scalar only after both source floor-division
        // branches have been described. Existing floor_div retains its
        // integer sentinel correction and float reciprocal-Mul-Floor graph.
        let scalar = self.constant(plan.scalar);
        let output = if reverse {
            self.floor_div(scalar, input)?
        } else {
            self.floor_div(input, scalar)?
        };
        debug_assert_eq!(self.shape(output).expect("floor_div scalar preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("floor_div scalar preflighted"), plan.output_dtype);
        Ok(output)
    }

    /// Source-compatible `Tensor.div(Python_scalar, rounding_mode="floor")`.
    pub fn floor_div_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.floor_div_scalar_with_order(input, value, false)
    }

    /// Source-compatible reflected `Python_scalar // Tensor` form.
    pub fn scalar_floor_div(&mut self, value: Scalar, input: NodeId) -> Result<NodeId> {
        self.floor_div_scalar_with_order(input, value, true)
    }

    fn trunc_div_scalar_with_order(
        &mut self,
        input: NodeId,
        value: Scalar,
        reverse: bool,
    ) -> Result<NodeId> {
        let plan = trunc_div_scalar_plan(self, input, value)?;
        // Publish only after the complete source CDIV or reciprocal-Mul-Trunc
        // branch has been preflighted. Reverse preserves tinygrad's operand
        // roles through its real `reverse` flag.
        let scalar = self.constant(plan.scalar);
        let output = if reverse {
            self.trunc_div(scalar, input)?
        } else {
            self.trunc_div(input, scalar)?
        };
        debug_assert_eq!(self.shape(output).expect("trunc_div scalar preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("trunc_div scalar preflighted"), plan.output_dtype);
        Ok(output)
    }

    /// Source-compatible `Tensor.div(Python_scalar, rounding_mode="trunc")`.
    pub fn trunc_div_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.trunc_div_scalar_with_order(input, value, false)
    }

    /// Source-compatible reflected `Python_scalar / Tensor` truncating form.
    pub fn scalar_trunc_div(&mut self, value: Scalar, input: NodeId) -> Result<NodeId> {
        self.trunc_div_scalar_with_order(input, value, true)
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

    fn modulo_scalar_with_order(
        &mut self,
        input: NodeId,
        value: Scalar,
        reverse: bool,
    ) -> Result<NodeId> {
        let plan = modulo_scalar_plan(self, input, value)?;
        // Reuse only the source-aligned modulo composition after the whole
        // floor-div/product/subtract descriptor has passed.
        let scalar = self.constant(plan.scalar);
        let output = if reverse {
            self.modulo(scalar, input)?
        } else {
            self.modulo(input, scalar)?
        };
        debug_assert_eq!(self.shape(output).expect("mod scalar preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("mod scalar preflighted"), plan.output_dtype);
        Ok(output)
    }

    /// Source-compatible `Tensor.mod(Python_scalar)` form, distinct from fmod.
    pub fn modulo_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.modulo_scalar_with_order(input, value, false)
    }

    /// Source-compatible reflected `Python_scalar % Tensor` form.
    pub fn scalar_modulo(&mut self, value: Scalar, input: NodeId) -> Result<NodeId> {
        self.modulo_scalar_with_order(input, value, true)
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

    /// Source-compatible non-reflected `Tensor.fmod(Python_scalar)` form.
    /// The scalar is committed once at tinygrad's weak `_broadcasted` width,
    /// then the existing truncation-based live lowering owns the literal
    /// `a - trunc(a / b) * b` graph and its integer zero-divisor sentinel.
    pub fn fmod_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        let plan = fmod_scalar_plan(self, input, value)?;
        let scalar = self.constant(plan.scalar);
        let output = self.fmod(input, scalar)?;
        debug_assert_eq!(self.shape(output).expect("fmod scalar preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("fmod scalar preflighted"), plan.output_dtype);
        Ok(output)
    }

    fn bitwise_binary(&mut self, op: BinaryOp, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let plan = bitwise_binary_plan(
            &lhs_shape,
            lhs_dtype,
            &rhs_shape,
            rhs_dtype,
            op,
        )?;
        let lhs = if plan.lhs_dtype == plan.output_dtype {
            lhs
        } else {
            self.cast(lhs, plan.output_dtype)?
        };
        let rhs = if plan.rhs_dtype == plan.output_dtype {
            rhs
        } else {
            self.cast(rhs, plan.output_dtype)?
        };
        let output = self.binary(op, lhs, rhs)?;
        debug_assert_eq!(self.shape(output).expect("bitwise binary shape preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("bitwise binary dtype preflighted"), plan.output_dtype);
        Ok(output)
    }

    fn bitwise_scalar(
        &mut self,
        op: BinaryOp,
        input: NodeId,
        value: Scalar,
        reverse: bool,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let input_shape = source.shape.clone();
        let input_dtype = source.dtype;
        input_shape
            .numel()?
            .checked_mul(input_dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
        let scalar_dtype = bitwise_scalar_dtype(input_dtype, value, op)?;
        let scalar_shape = Shape::new([]);
        let (lhs_shape, lhs_dtype, rhs_shape, rhs_dtype) = if reverse {
            (&scalar_shape, scalar_dtype, &input_shape, input_dtype)
        } else {
            (&input_shape, input_dtype, &scalar_shape, scalar_dtype)
        };
        let plan = bitwise_binary_plan(lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, op)?;
        // A weak Python scalar commits directly at the source LUB width.
        // Construct it only after the full scalar/cast/broadcast plan passes.
        let scalar = TensorData::scalar_with_dtype(value, plan.output_dtype);
        debug_assert_eq!(scalar.shape(), &scalar_shape);
        debug_assert_eq!(scalar.dtype(), plan.output_dtype);
        let scalar = self.constant(scalar);
        let (lhs, rhs, lhs_dtype, rhs_dtype) = if reverse {
            (scalar, input, scalar_dtype, input_dtype)
        } else {
            (input, scalar, input_dtype, scalar_dtype)
        };
        let lhs = if lhs_dtype == plan.output_dtype {
            lhs
        } else {
            self.cast(lhs, plan.output_dtype)?
        };
        let rhs = if rhs_dtype == plan.output_dtype {
            rhs
        } else {
            self.cast(rhs, plan.output_dtype)?
        };
        let output = self.binary(op, lhs, rhs)?;
        debug_assert_eq!(self.shape(output).expect("bitwise scalar shape preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("bitwise scalar dtype preflighted"), plan.output_dtype);
        Ok(output)
    }

    /// Source-compatible short alias for tinygrad's `Tensor.bitwise_and`.
    pub fn bit_and(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.bitwise_binary(BinaryOp::BitAnd, lhs, rhs)
    }

    /// Source-compatible public name for tinygrad's `Tensor.bitwise_and`.
    pub fn bitwise_and(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.bit_and(lhs, rhs)
    }

    pub fn bitwise_and_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.bitwise_scalar(BinaryOp::BitAnd, input, value, false)
    }

    /// Reflected Python-style scalar form: `value & input`.
    pub fn scalar_bitwise_and(&mut self, value: Scalar, input: NodeId) -> Result<NodeId> {
        self.bitwise_scalar(BinaryOp::BitAnd, input, value, true)
    }

    /// Source-compatible short alias for tinygrad's `Tensor.bitwise_or`.
    pub fn bit_or(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.bitwise_binary(BinaryOp::BitOr, lhs, rhs)
    }

    /// Source-compatible public name for tinygrad's `Tensor.bitwise_or`.
    pub fn bitwise_or(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.bit_or(lhs, rhs)
    }

    pub fn bitwise_or_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.bitwise_scalar(BinaryOp::BitOr, input, value, false)
    }

    /// Reflected Python-style scalar form: `value | input`.
    pub fn scalar_bitwise_or(&mut self, value: Scalar, input: NodeId) -> Result<NodeId> {
        self.bitwise_scalar(BinaryOp::BitOr, input, value, true)
    }

    /// Source-compatible short alias for tinygrad's `Tensor.bitwise_xor`.
    pub fn bit_xor(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.bitwise_binary(BinaryOp::BitXor, lhs, rhs)
    }

    /// Source-compatible public name for tinygrad's `Tensor.bitwise_xor`.
    pub fn bitwise_xor(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.bit_xor(lhs, rhs)
    }

    pub fn bitwise_xor_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.bitwise_scalar(BinaryOp::BitXor, input, value, false)
    }

    /// Reflected Python-style scalar form: `value ^ input`.
    pub fn scalar_bitwise_xor(&mut self, value: Scalar, input: NodeId) -> Result<NodeId> {
        self.bitwise_scalar(BinaryOp::BitXor, input, value, true)
    }

    /// Mirrors tinygrad's `Tensor.bitwise_not` / `~x` without introducing a
    /// dedicated unary IR operation. Bool keeps its literal `logical_not`
    /// graph; signed integers XOR a typed `-1`, and unsigned integers XOR
    /// their storage-width maximum.
    pub fn bitwise_not(&mut self, input: NodeId) -> Result<NodeId> {
        let plan = bitwise_not_plan(self, input)?;
        let output = if plan.dtype == DType::Bool {
            self.logical_not(input)?
        } else {
            self.bit_xor(input, self.constant(plan.mask))?
        };
        debug_assert_eq!(self.shape(output).expect("bitwise_not shape preflighted"), &plan.shape);
        debug_assert_eq!(self.dtype(output).expect("bitwise_not dtype preflighted"), plan.dtype);
        Ok(output)
    }

    pub fn shl(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Shl, lhs, rhs)
    }
    pub fn shr(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Shr, lhs, rhs)
    }

    pub fn eq(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        // Tensor.eq is literally `ne(...).logical_not()`, after ne promotes
        // both value operands through tinygrad's least-upper lattice. In
        // particular, I64/U64 meets at source F32 rather than RustGrad's
        // legacy F64 bridge. Complete the comparison and logical-not stages
        // before either Cast, Compare, or Bool constant can mutate the graph.
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
        extent(&output_shape, DType::Bool)?;
        let truth = TensorData::scalar_with_dtype(Scalar::Bool(true), DType::Bool);
        extent(truth.shape(), truth.dtype())?;
        if truth.dtype() != DType::Bool || output_shape.broadcast_with(truth.shape())? != output_shape {
            return Err(Error::InvalidElementwiseDType {
                op: "eq logical_not promotion",
                actual: comparison_dtype,
            });
        }
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
        let unequal = self.compare(CompareOp::Ne, lhs, rhs)?;
        self.logical_not(unequal)
    }

    fn comparison_scalar(
        &mut self,
        input: NodeId,
        value: Scalar,
        op: CompareOp,
    ) -> Result<NodeId> {
        debug_assert!(matches!(op, CompareOp::Eq | CompareOp::Ne));
        let plan = comparison_scalar_plan(self, input, value)?;
        let scalar = self.constant(plan.scalar);
        let output = match op {
            CompareOp::Eq => self.eq(input, scalar)?,
            CompareOp::Ne => self.ne(input, scalar)?,
            _ => unreachable!("only equality predicates use comparison_scalar"),
        };
        debug_assert_eq!(self.shape(output).expect("comparison scalar preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("comparison scalar preflighted"), DType::Bool);
        debug_assert_eq!(self.dtype(scalar).expect("comparison scalar preflighted"), plan.comparison_dtype);
        Ok(output)
    }

    /// Source-compatible `Tensor.eq(Python_scalar)` form. tinygrad does not
    /// expose a reflected equality overload, so the tensor remains lhs.
    pub fn eq_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.comparison_scalar(input, value, CompareOp::Eq)
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

    /// Source-compatible `Tensor.ne(Python_scalar)` form. The tensor stays
    /// lhs, matching tinygrad's explicit non-reflected method.
    pub fn ne_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.comparison_scalar(input, value, CompareOp::Ne)
    }

    fn ordered_comparison_scalar(
        &mut self,
        input: NodeId,
        value: Scalar,
        op: CompareOp,
        reverse: bool,
    ) -> Result<NodeId> {
        debug_assert!(matches!(op, CompareOp::Lt | CompareOp::Gt | CompareOp::Le | CompareOp::Ge));
        let plan = comparison_scalar_plan(self, input, value)?;
        let scalar = self.constant(plan.scalar);
        // Python's reflected comparison dispatch invokes the complementary
        // Tensor dunder. Keep those calls explicit so the ordered LT and
        // inclusive Not orientations remain source-visible.
        let output = match (op, reverse) {
            (CompareOp::Lt, false) => self.lt(input, scalar)?,
            (CompareOp::Gt, false) => self.gt(input, scalar)?,
            (CompareOp::Le, false) => self.le(input, scalar)?,
            (CompareOp::Ge, false) => self.ge(input, scalar)?,
            (CompareOp::Lt, true) => self.gt(input, scalar)?,
            (CompareOp::Gt, true) => self.lt(input, scalar)?,
            (CompareOp::Le, true) => self.ge(input, scalar)?,
            (CompareOp::Ge, true) => self.le(input, scalar)?,
            _ => unreachable!("only ordered predicates use ordered_comparison_scalar"),
        };
        debug_assert_eq!(self.shape(output).expect("ordered scalar preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("ordered scalar preflighted"), DType::Bool);
        debug_assert_eq!(self.dtype(scalar).expect("ordered scalar preflighted"), plan.comparison_dtype);
        Ok(output)
    }

    /// Source-compatible `Tensor < Python_scalar` form.
    pub fn lt_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.ordered_comparison_scalar(input, value, CompareOp::Lt, false)
    }

    /// Source-compatible reflected `Python_scalar < Tensor` form, dispatched
    /// by Python to Tensor's reversed `__gt__` comparison.
    pub fn scalar_lt(&mut self, value: Scalar, input: NodeId) -> Result<NodeId> {
        self.ordered_comparison_scalar(input, value, CompareOp::Lt, true)
    }

    /// Source-compatible `Tensor > Python_scalar` form.
    pub fn gt_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.ordered_comparison_scalar(input, value, CompareOp::Gt, false)
    }

    /// Source-compatible reflected `Python_scalar > Tensor` form.
    pub fn scalar_gt(&mut self, value: Scalar, input: NodeId) -> Result<NodeId> {
        self.ordered_comparison_scalar(input, value, CompareOp::Gt, true)
    }

    /// Source-compatible `Tensor <= Python_scalar` form.
    pub fn le_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.ordered_comparison_scalar(input, value, CompareOp::Le, false)
    }

    /// Source-compatible reflected `Python_scalar <= Tensor` form.
    pub fn scalar_le(&mut self, value: Scalar, input: NodeId) -> Result<NodeId> {
        self.ordered_comparison_scalar(input, value, CompareOp::Le, true)
    }

    /// Source-compatible `Tensor >= Python_scalar` form.
    pub fn ge_scalar(&mut self, input: NodeId, value: Scalar) -> Result<NodeId> {
        self.ordered_comparison_scalar(input, value, CompareOp::Ge, false)
    }

    /// Source-compatible reflected `Python_scalar >= Tensor` form.
    pub fn scalar_ge(&mut self, value: Scalar, input: NodeId) -> Result<NodeId> {
        self.ordered_comparison_scalar(input, value, CompareOp::Ge, true)
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
        // Tensor.logical_not is `cast(bool).ne(True)`, admitting every
        // source dtype through tensor cast truthiness rather than requiring
        // a Bool input or using bitwise/numeric negation.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let truth = TensorData::scalar_with_dtype(Scalar::Bool(true), DType::Bool);
        let extent = |shape: &Shape, dtype: DType| shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()));
        extent(&shape, input_dtype)?;
        extent(&shape, DType::Bool)?;
        extent(truth.shape(), truth.dtype())?;
        if truth.dtype() != DType::Bool || shape.broadcast_with(truth.shape())? != shape {
            return Err(Error::InvalidLogicalDType { op: "logical_not", actual: input_dtype });
        }
        let boolean = self.cast(input, DType::Bool)?;
        self.ne(boolean, self.constant(truth))
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

    /// Public spelling of tinygrad's `Tensor.where(x, y)`. Rust requires the
    /// raw identifier; its condition is `self`, with `x` as the true branch.
    pub fn r#where(
        &mut self,
        condition: NodeId,
        on_true: NodeId,
        on_false: NodeId,
    ) -> Result<NodeId> {
        self.select(condition, on_true, on_false)
    }

    fn where_with_scalar(
        &mut self,
        condition: NodeId,
        on_true: WhereBranch,
        on_false: WhereBranch,
    ) -> Result<NodeId> {
        let WhereScalarPlan {
            output_shape,
            output_dtype,
            on_true_scalar,
            on_false_scalar,
        } = where_scalar_plan(self, condition, on_true, on_false)?;
        // The pure plan covers every fallible descriptor and byte calculation
        // performed by Select. Publish scalar constants only after it passes,
        // retaining the source true/false branch ordering.
        let on_true = match (on_true, on_true_scalar) {
            (WhereBranch::Live(node), None) => node,
            (WhereBranch::Scalar(_), Some(scalar)) => self.constant(scalar),
            _ => unreachable!("where scalar plan must match its true branch"),
        };
        let on_false = match (on_false, on_false_scalar) {
            (WhereBranch::Live(node), None) => node,
            (WhereBranch::Scalar(_), Some(scalar)) => self.constant(scalar),
            _ => unreachable!("where scalar plan must match its false branch"),
        };
        let output = self.select(condition, on_true, on_false)?;
        debug_assert_eq!(self.shape(output).expect("where scalar preflighted"), &output_shape);
        debug_assert_eq!(self.dtype(output).expect("where scalar preflighted"), output_dtype);
        Ok(output)
    }

    /// Tinygrad `condition.where(scalar, false_tensor)` with a weak scalar
    /// true branch committed against the live false branch.
    pub fn where_true_scalar(
        &mut self,
        condition: NodeId,
        on_true: Scalar,
        on_false: NodeId,
    ) -> Result<NodeId> {
        self.where_with_scalar(
            condition,
            WhereBranch::Scalar(on_true),
            WhereBranch::Live(on_false),
        )
    }

    /// Tinygrad `condition.where(true_tensor, scalar)` with a weak scalar
    /// false branch committed against the live true branch.
    pub fn where_false_scalar(
        &mut self,
        condition: NodeId,
        on_true: NodeId,
        on_false: Scalar,
    ) -> Result<NodeId> {
        self.where_with_scalar(
            condition,
            WhereBranch::Live(on_true),
            WhereBranch::Scalar(on_false),
        )
    }

    /// Tinygrad `condition.where(true_scalar, false_scalar)`. With no live
    /// payload, the source uses the Bool condition to materialize the first
    /// scalar before weak-promoting the second.
    pub fn where_scalars(
        &mut self,
        condition: NodeId,
        on_true: Scalar,
        on_false: Scalar,
    ) -> Result<NodeId> {
        self.where_with_scalar(
            condition,
            WhereBranch::Scalar(on_true),
            WhereBranch::Scalar(on_false),
        )
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

    /// Source-compatible scalar-value form of tinygrad's
    /// `Tensor.masked_fill(mask, value)`. This preserves literal
    /// `mask.where(value, input)` branch order.
    pub fn masked_fill_scalar(
        &mut self,
        input: NodeId,
        mask: NodeId,
        value: Scalar,
    ) -> Result<NodeId> {
        let plan = masked_fill_scalar_plan(self, input, mask, value)?;
        // The whole Select descriptor has passed before the weak scalar is
        // published. Reuse the live branch order rather than adding a
        // separate scalar WHERE surface.
        let value = self.constant(plan.scalar);
        let output = self.masked_fill(input, mask, value)?;
        debug_assert_eq!(self.shape(output).expect("masked_fill scalar preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("masked_fill scalar preflighted"), plan.output_dtype);
        Ok(output)
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
        // Tensor.log is literally `log2(x) * ln(2)`. Its weak scalar adopts
        // the Log2 result storage dtype, which preserves narrow/F64 rounding
        // and differs from a host natural-log unary. Establish every extent
        // and scalar promotion before either the constant or a node exists.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let output_dtype = unary_dtype(UnaryOp::Log2, input_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&shape, input_dtype)?;
        extent(&shape, output_dtype)?;
        let ln2 = TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::LN_2), output_dtype);
        extent(ln2.shape(), ln2.dtype())?;
        if (!input_dtype.is_float() && output_dtype != DType::F32)
            || (input_dtype.is_float() && output_dtype != input_dtype)
            || ln2.dtype() != output_dtype
            || output_dtype.promote(ln2.dtype()) != output_dtype
            || shape.broadcast_with(ln2.shape())? != shape
        {
            return Err(Error::InvalidElementwiseDType {
                op: "log source promotion",
                actual: output_dtype,
            });
        }
        let logged = self.log2(input)?;
        let ln2 = self.constant(ln2);
        self.mul(logged, ln2)
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
        // source-level `1 / x` composition. Its ALU promotion first casts a
        // nonfloat operand to the default float, then applies RECIPROCAL to
        // a same-dtype operand.  Keep the raw Unary ABI homogeneous: a
        // heterogeneous `GraphUnary(Reciprocal)` would be rejected by UOp
        // validation before any backend can execute it. Prove both the cast
        // and reciprocal descriptors before either node is published.
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
        // The source's nonfloat ALU promotion has the same concrete shape and
        // F32 storage descriptor as the reciprocal result.
        if !input_dtype.is_float() {
            extent(&shape, DType::F32)?;
        }
        extent(&shape, output_dtype)?;
        if (!input_dtype.is_float() && output_dtype != DType::F32)
            || (input_dtype.is_float() && output_dtype != input_dtype)
        {
            return Err(Error::InvalidElementwiseDType {
                op: "reciprocal output dtype",
                actual: output_dtype,
            });
        }
        let reciprocal_input = if input_dtype.is_float() {
            input
        } else {
            self.cast(input, DType::F32)?
        };
        self.unary(UnaryOp::Reciprocal, reciprocal_input)
    }
    pub fn square(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.square is literally `self * self`. Keep raw SQUARE available
        // to lower-level callers, but use the source multiplication so its
        // storage-width arithmetic and binary VJP structure are preserved.
        // Prove the full self-broadcast/result descriptor before publication.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let output_shape = shape.broadcast_with(&shape)?;
        let output_dtype = input_dtype.promote(input_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&shape, input_dtype)?;
        extent(&output_shape, output_dtype)?;
        if output_shape != shape || output_dtype != input_dtype {
            return Err(Error::InvalidElementwiseDType {
                op: "square self multiplication promotion",
                actual: output_dtype,
            });
        }
        self.mul(input, input)
    }
    pub fn sqrt(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.sqrt is the direct SQRT ALU primitive. Its source unary
        // lattice first casts every nonfloat input to the default F32, then
        // applies SQRT to homogeneous storage. Keep raw UnaryOp::Sqrt
        // available, but make the public helper's cast boundary explicit so
        // UOp validation and every fused backend see the same typed ALU.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let output_dtype = unary_dtype(UnaryOp::Sqrt, input_dtype);
        let extent = |dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(input_dtype)?;
        if !input_dtype.is_float() {
            extent(DType::F32)?;
        }
        extent(output_dtype)?;
        if (!input_dtype.is_float() && output_dtype != DType::F32)
            || (input_dtype.is_float() && output_dtype != input_dtype)
        {
            return Err(Error::InvalidElementwiseDType {
                op: "sqrt source promotion",
                actual: output_dtype,
            });
        }
        let sqrt_input = if input_dtype.is_float() {
            input
        } else {
            self.cast(input, DType::F32)?
        };
        self.unary(UnaryOp::Sqrt, sqrt_input)
    }
    pub fn rsqrt(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.rsqrt is intentionally not a raw RSQRT ALU op: tinygrad
        // spells it as `sqrt().reciprocal()`. Preserve the storage rounding
        // boundary and compositional VJP by proving both unary descriptors
        // before either node can be appended.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let sqrt_dtype = unary_dtype(UnaryOp::Sqrt, input_dtype);
        let output_dtype = unary_dtype(UnaryOp::Reciprocal, sqrt_dtype);
        let extent = |dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(input_dtype)?;
        extent(sqrt_dtype)?;
        extent(output_dtype)?;
        if (!input_dtype.is_float() && (sqrt_dtype != DType::F32 || output_dtype != DType::F32))
            || (input_dtype.is_float() && (sqrt_dtype != input_dtype || output_dtype != input_dtype))
        {
            return Err(Error::InvalidElementwiseDType {
                op: "rsqrt sqrt/reciprocal source promotion",
                actual: output_dtype,
            });
        }
        let root = self.sqrt(input)?;
        self.reciprocal(root)
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
        // Tensor.log2 is direct LOG2 ALU: it preserves every floating storage
        // width and lifts non-floats to F32. Prove both allocation extents
        // before constructing the public result node.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let output_dtype = unary_dtype(UnaryOp::Log2, input_dtype);
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
                op: "log2 source promotion",
                actual: output_dtype,
            });
        }
        self.unary(UnaryOp::Log2, input)
    }
    pub fn sin(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.sin is direct SIN ALU. Its unary source lattice preserves
        // floating storage widths and lifts every nonfloat to F32; validate
        // those concrete source/result extents before the node is published.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let output_dtype = unary_dtype(UnaryOp::Sin, input_dtype);
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
                op: "sin source promotion",
                actual: output_dtype,
            });
        }
        self.unary(UnaryOp::Sin, input)
    }
    pub fn cos(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.cos is `sin(pi/2 - x)`, with narrow floats widened to F32
        // for the phase arithmetic and narrowed only after SIN. Keep raw COS
        // available to low-level callers, but preflight this literal public
        // composition before publishing a cast, constant, or unary node.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
        let work_dtype = output_dtype.promote(DType::F32);
        let half_pi = TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::FRAC_PI_2), work_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&shape, input_dtype)?;
        extent(&shape, output_dtype)?;
        extent(&shape, work_dtype)?;
        extent(half_pi.shape(), half_pi.dtype())?;
        let phase_shape = half_pi.shape().broadcast_with(&shape)?;
        let phase_dtype = half_pi.dtype().promote(work_dtype);
        let sine_dtype = unary_dtype(UnaryOp::Sin, phase_dtype);
        if ((!input_dtype.is_float() && output_dtype != DType::F32)
            || (input_dtype.is_float() && output_dtype != input_dtype))
            || (work_dtype != output_dtype.promote(DType::F32))
            || half_pi.dtype() != work_dtype
            || phase_shape != shape
            || phase_dtype != work_dtype
            || sine_dtype != work_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "cos phase-shift source promotion",
                actual: sine_dtype,
            });
        }
        let lifted = if input_dtype == output_dtype { input } else { self.cast(input, output_dtype)? };
        let widened = if output_dtype == work_dtype { lifted } else { self.cast(lifted, work_dtype)? };
        let phase = self.sub(self.constant(half_pi), widened)?;
        let sine = self.sin(phase)?;
        if work_dtype == output_dtype { Ok(sine) } else { self.cast(sine, output_dtype) }
    }
    pub fn tan(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.tan is literally `self.sin() / self.cos()`. Plan both
        // source-aligned branches and their true-division result before any
        // node can be published; raw TAN remains a lower-level primitive.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let sin_dtype = unary_dtype(UnaryOp::Sin, input_dtype);
        let cos_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
        let output_dtype = sin_dtype.promote(cos_dtype);
        let extent = |dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(input_dtype)?;
        extent(sin_dtype)?;
        extent(cos_dtype)?;
        extent(output_dtype)?;
        if ((!input_dtype.is_float() && (sin_dtype != DType::F32 || cos_dtype != DType::F32))
            || (input_dtype.is_float() && (sin_dtype != input_dtype || cos_dtype != input_dtype)))
            || shape.broadcast_with(&shape)? != shape
            || output_dtype != sin_dtype
        {
            return Err(Error::InvalidElementwiseDType {
                op: "tan sin/cos division source promotion",
                actual: output_dtype,
            });
        }
        let numerator = self.sin(input)?;
        let denominator = self.cos(input)?;
        self.div(numerator, denominator)
    }
    pub fn sinh(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.sinh is `(exp(x) - exp(-x)) / 2`, not raw SINH. In
        // particular negation occurs at input storage before Exp promotes.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
        let neg_dtype = unary_dtype(UnaryOp::Neg, input_dtype);
        let exp_dtype = unary_dtype(UnaryOp::Exp, input_dtype);
        let two = TensorData::scalar_with_dtype(Scalar::I(2), output_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&shape, input_dtype)?;
        for dtype in [neg_dtype, exp_dtype, exp_dtype, output_dtype, output_dtype] {
            extent(&shape, dtype)?;
        }
        extent(two.shape(), two.dtype())?;
        if ((!input_dtype.is_float() && output_dtype != DType::F32)
            || (input_dtype.is_float() && output_dtype != input_dtype))
            || neg_dtype != input_dtype
            || exp_dtype != output_dtype
            || two.dtype() != output_dtype
            || shape.broadcast_with(two.shape())? != shape
            || output_dtype.promote(output_dtype) != output_dtype
        {
            return Err(Error::InvalidElementwiseDType { op: "sinh exp/sub/div source promotion", actual: output_dtype });
        }
        let positive = self.exp(input)?;
        let negative = self.exp(self.neg(input)?)?;
        self.div(self.sub(positive, negative)?, self.constant(two))
    }
    pub fn cosh(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.cosh is `(exp(x) + exp(-x)) / 2`, not raw COSH. As with
        // Sinh, negation deliberately occurs at input storage before Exp's
        // source floating promotion.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
        let neg_dtype = unary_dtype(UnaryOp::Neg, input_dtype);
        let exp_dtype = unary_dtype(UnaryOp::Exp, input_dtype);
        let two = TensorData::scalar_with_dtype(Scalar::I(2), output_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&shape, input_dtype)?;
        for dtype in [neg_dtype, exp_dtype, exp_dtype, output_dtype, output_dtype] {
            extent(&shape, dtype)?;
        }
        extent(two.shape(), two.dtype())?;
        if ((!input_dtype.is_float() && output_dtype != DType::F32)
            || (input_dtype.is_float() && output_dtype != input_dtype))
            || neg_dtype != input_dtype
            || exp_dtype != output_dtype
            || two.dtype() != output_dtype
            || shape.broadcast_with(two.shape())? != shape
            || output_dtype.promote(output_dtype) != output_dtype
        {
            return Err(Error::InvalidElementwiseDType { op: "cosh exp/add/div source promotion", actual: output_dtype });
        }
        let positive = self.exp(input)?;
        let negative = self.exp(self.neg(input)?)?;
        self.div(self.add(positive, negative)?, self.constant(two))
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
        // Tensor.erf is A&S 7.1.26, evaluated through weak source-width
        // scalars: sign(x) * (1 - t * polyN(t) * exp(-square(x))), where
        // t = 1 / (1 + 0.3275911 * abs(x)). Raw ERF has different narrow
        // rounding and a different reverse-mode graph.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
        let coefficients = [1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592];
        let coefficient = TensorData::scalar_with_dtype(Scalar::F(0.3275911), input_dtype);
        let input_one = TensorData::scalar_with_dtype(Scalar::F(1.0), input_dtype);
        let output_one = TensorData::scalar_with_dtype(Scalar::F(1.0), output_dtype);
        let zero = TensorData::scalar_with_dtype(Scalar::F(0.0), output_dtype);
        let polynomial = coefficients.map(|value| TensorData::scalar_with_dtype(Scalar::F(value), output_dtype));
        let extent = |shape: &Shape, dtype: DType| {
            shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        for dtype in [input_dtype, input_dtype, input_dtype, input_dtype, output_dtype, output_dtype, output_dtype, output_dtype, output_dtype] {
            extent(&shape, dtype)?;
        }
        for scalar in [&coefficient, &input_one] {
            extent(scalar.shape(), scalar.dtype())?;
            if scalar.dtype() != input_dtype || shape.broadcast_with(scalar.shape())? != shape {
                return Err(Error::InvalidElementwiseDType { op: "erf source scalar promotion", actual: scalar.dtype() });
            }
        }
        for scalar in [&output_one, &zero].into_iter().chain(polynomial.iter()) {
            extent(scalar.shape(), scalar.dtype())?;
            if scalar.dtype() != output_dtype || shape.broadcast_with(scalar.shape())? != shape {
                return Err(Error::InvalidElementwiseDType { op: "erf source scalar promotion", actual: scalar.dtype() });
            }
        }
        if unary_dtype(UnaryOp::Sign, input_dtype) != input_dtype
            || unary_dtype(UnaryOp::Neg, input_dtype) != input_dtype
            || unary_dtype(UnaryOp::Exp, input_dtype) != output_dtype
            || (!input_dtype.is_float() && output_dtype != DType::F32)
            || (input_dtype.is_float() && output_dtype != input_dtype)
            || input_dtype.promote(output_dtype) != output_dtype
        {
            return Err(Error::InvalidElementwiseDType { op: "erf source promotion", actual: output_dtype });
        }
        let absolute = self.abs(input)?;
        let denominator = self.add(self.constant(input_one.clone()), self.mul(self.constant(coefficient), absolute)?)?;
        let t = self.div(self.constant(input_one), denominator)?;
        let mut poly = self.constant(zero);
        for coefficient in polynomial {
            poly = self.add(self.mul(poly, t)?, self.constant(coefficient))?;
        }
        let exponent = self.exp(self.neg(self.square(input)?)?)?;
        let tail = self.mul(self.mul(t, poly)?, exponent)?;
        let body = self.sub(self.constant(output_one), tail)?;
        self.mul(self.sign(input)?, body)
    }
    /// Applies the complementary Gauss error function elementwise.
    pub fn erfc(&mut self, input: NodeId) -> Result<NodeId> {
        self.unary(UnaryOp::Erfc, input)
    }
    pub fn asin(&mut self, input: NodeId) -> Result<NodeId> {
        // tinygrad uses the 4.4.46 polynomial approximation, not raw ASIN:
        // sign(x) * (pi/2 - sqrt(1-abs(x)) * polyN(abs(x), coefficients)).
        // Preflight every source-width scalar and same-shape intermediate
        // before publishing the first part of the Horner expansion.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
        let coefficients = [
            -0.0012624911, 0.0066700901, -0.0170881256, 0.0308918810,
            -0.0501743046, 0.0889789874, -0.2145988016, 1.5707963050,
        ];
        let scalars = coefficients.map(|value| TensorData::scalar_with_dtype(Scalar::F(value), output_dtype));
        let zero = TensorData::scalar_with_dtype(Scalar::F(0.0), output_dtype);
        let one = TensorData::scalar_with_dtype(Scalar::F(1.0), output_dtype);
        let half_pi = TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::FRAC_PI_2), output_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&shape, input_dtype)?;
        extent(&shape, output_dtype)?;
        for scalar in scalars.iter().chain([&zero, &one, &half_pi]) {
            extent(scalar.shape(), scalar.dtype())?;
            if scalar.dtype() != output_dtype || shape.broadcast_with(scalar.shape())? != shape {
                return Err(Error::InvalidElementwiseDType { op: "asin polynomial scalar promotion", actual: scalar.dtype() });
            }
        }
        if (!input_dtype.is_float() && output_dtype != DType::F32)
            || (input_dtype.is_float() && output_dtype != input_dtype)
            || input_dtype.promote(output_dtype) != output_dtype
            || unary_dtype(UnaryOp::Sqrt, output_dtype) != output_dtype
        {
            return Err(Error::InvalidElementwiseDType { op: "asin source promotion", actual: output_dtype });
        }
        let absolute = self.abs(input)?;
        let absolute_work = if input_dtype == output_dtype { absolute } else { self.cast(absolute, output_dtype)? };
        let one = self.constant(one);
        let radius = self.sqrt(self.sub(one, absolute_work)?)?;
        let mut polynomial = self.constant(zero);
        for coefficient in scalars {
            polynomial = self.add(self.mul(polynomial, absolute_work)?, self.constant(coefficient))?;
        }
        let magnitude = self.sub(self.constant(half_pi), self.mul(radius, polynomial)?)?;
        self.mul(self.sign(input)?, magnitude)
    }
    pub fn acos(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.acos is `pi/2 - self.asin()`. Retain the public Asin
        // approximation and its storage rounding, with a weak pi/2 constant
        // at the fully resolved result dtype.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
        let half_pi = TensorData::scalar_with_dtype(Scalar::F(std::f64::consts::FRAC_PI_2), output_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&shape, input_dtype)?;
        extent(&shape, output_dtype)?;
        extent(half_pi.shape(), half_pi.dtype())?;
        if ((!input_dtype.is_float() && output_dtype != DType::F32)
            || (input_dtype.is_float() && output_dtype != input_dtype))
            || half_pi.dtype() != output_dtype
            || shape.broadcast_with(half_pi.shape())? != shape
            || output_dtype.promote(output_dtype) != output_dtype
        {
            return Err(Error::InvalidElementwiseDType { op: "acos asin source promotion", actual: output_dtype });
        }
        let asin = self.asin(input)?;
        self.sub(self.constant(half_pi), asin)
    }
    pub fn atan(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.atan is `(x / sqrt(1 + x*x)).asin()`, preserving the
        // multiplication/addition storage boundary before sqrt promotes.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let sqrt_dtype = unary_dtype(UnaryOp::Sqrt, input_dtype);
        let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
        let one = TensorData::scalar_with_dtype(Scalar::I(1), input_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&shape, input_dtype)?;
        extent(&shape, sqrt_dtype)?;
        extent(&shape, output_dtype)?;
        extent(one.shape(), one.dtype())?;
        if sqrt_dtype != output_dtype
            || one.dtype() != input_dtype
            || shape.broadcast_with(one.shape())? != shape
            || input_dtype.promote(input_dtype) != input_dtype
            || input_dtype.promote(sqrt_dtype) != output_dtype
        {
            return Err(Error::InvalidElementwiseDType { op: "atan source promotion", actual: output_dtype });
        }
        let square = self.mul(input, input)?;
        let denominator = self.sqrt(self.add(self.constant(one), square)?)?;
        self.asin(self.div(input, denominator)?)
    }
    pub fn asinh(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.asinh is literally `log(x + sqrt(square(x) + 1))`. Keep
        // square at input storage; only Sqrt lifts nonfloating values. This
        // ordering is observable for narrow arithmetic and -infinity.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let square_dtype = input_dtype.promote(input_dtype);
        let root_dtype = unary_dtype(UnaryOp::Sqrt, square_dtype);
        let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
        let one = TensorData::scalar_with_dtype(Scalar::I(1), square_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        for dtype in [input_dtype, square_dtype, square_dtype, root_dtype, output_dtype, output_dtype] {
            extent(&shape, dtype)?;
        }
        extent(one.shape(), one.dtype())?;
        if square_dtype != input_dtype
            || (!square_dtype.is_float() && root_dtype != DType::F32)
            || (square_dtype.is_float() && root_dtype != square_dtype)
            || (!input_dtype.is_float() && output_dtype != DType::F32)
            || (input_dtype.is_float() && output_dtype != input_dtype)
            || input_dtype.promote(root_dtype) != output_dtype
            || unary_dtype(UnaryOp::Log2, output_dtype) != output_dtype
            || one.dtype() != square_dtype
            || shape.broadcast_with(one.shape())? != shape
        {
            return Err(Error::InvalidElementwiseDType { op: "asinh square/sqrt/add/log source promotion", actual: output_dtype });
        }
        let square = self.square(input)?;
        let radicand = self.add(square, self.constant(one))?;
        let root = self.sqrt(radicand)?;
        self.log(self.add(input, root)?)
    }
    pub fn acosh(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.acosh is literally `log(x + sqrt(square(x) - 1))`. Keep
        // square and its weak-one subtraction at input storage; Sqrt is the
        // first operation that promotes nonfloating inputs.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let square_dtype = input_dtype.promote(input_dtype);
        let root_dtype = unary_dtype(UnaryOp::Sqrt, square_dtype);
        let output_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
        let one = TensorData::scalar_with_dtype(Scalar::I(1), square_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        for dtype in [input_dtype, square_dtype, square_dtype, root_dtype, output_dtype, output_dtype] {
            extent(&shape, dtype)?;
        }
        extent(one.shape(), one.dtype())?;
        if square_dtype != input_dtype
            || (!square_dtype.is_float() && root_dtype != DType::F32)
            || (square_dtype.is_float() && root_dtype != square_dtype)
            || (!input_dtype.is_float() && output_dtype != DType::F32)
            || (input_dtype.is_float() && output_dtype != input_dtype)
            || input_dtype.promote(root_dtype) != output_dtype
            || unary_dtype(UnaryOp::Log2, output_dtype) != output_dtype
            || one.dtype() != square_dtype
            || shape.broadcast_with(one.shape())? != shape
        {
            return Err(Error::InvalidElementwiseDType { op: "acosh square/sub/sqrt/add/log source promotion", actual: output_dtype });
        }
        let square = self.square(input)?;
        let radicand = self.sub(square, self.constant(one))?;
        let root = self.sqrt(radicand)?;
        self.log(self.add(input, root)?)
    }
    pub fn atanh(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.atanh is `log((1 + x) / (1 - x)) / 2`. The numerator and
        // denominator use weak ones at input storage, then true division is
        // the first nonfloat-to-F32 boundary.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let reciprocal_dtype = unary_dtype(UnaryOp::Reciprocal, input_dtype);
        let dividend_dtype = if input_dtype.is_float() { input_dtype } else { DType::F32 };
        let ratio_dtype = dividend_dtype.promote(reciprocal_dtype);
        let log_dtype = unary_dtype(UnaryOp::Log2, ratio_dtype);
        let one = TensorData::scalar_with_dtype(Scalar::I(1), input_dtype);
        let two = TensorData::scalar_with_dtype(Scalar::I(2), log_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        for dtype in [input_dtype, input_dtype, input_dtype, dividend_dtype, reciprocal_dtype, ratio_dtype, log_dtype, log_dtype] {
            extent(&shape, dtype)?;
        }
        extent(one.shape(), one.dtype())?;
        extent(two.shape(), two.dtype())?;
        if (!input_dtype.is_float() && (dividend_dtype != DType::F32 || reciprocal_dtype != DType::F32))
            || (input_dtype.is_float() && (dividend_dtype != input_dtype || reciprocal_dtype != input_dtype))
            || dividend_dtype.promote(reciprocal_dtype) != ratio_dtype
            || log_dtype != ratio_dtype
            || one.dtype() != input_dtype
            || two.dtype() != log_dtype
            || shape.broadcast_with(one.shape())? != shape
            || shape.broadcast_with(two.shape())? != shape
        {
            return Err(Error::InvalidElementwiseDType { op: "atanh add/sub/div/log source promotion", actual: log_dtype });
        }
        let numerator = self.add(self.constant(one.clone()), input)?;
        let denominator = self.sub(self.constant(one), input)?;
        let ratio = self.div(numerator, denominator)?;
        let logarithm = self.log(ratio)?;
        self.div(logarithm, self.constant(two))
    }
    /// Returns the quadrant-aware angle of `(y, x)` elementwise.
    pub fn atan2(&mut self, y: NodeId, x: NodeId) -> Result<NodeId> {
        self.binary(BinaryOp::Atan2, y, x)
    }
    /// Returns the magnitude of `magnitude` with the sign selected by `sign`.
    pub fn copysign(&mut self, magnitude: NodeId, sign: NodeId) -> Result<NodeId> {
        // Tensor.copysign is a literal comparison/reciprocal/WHERE graph,
        // rather than a raw host copysign ALU.  The distinction is observable
        // for -0 (whose reciprocal is negative) and NaN sign operands (both
        // ordered comparisons are false). Build a complete descriptor plan
        // before the first cast, scalar constant, or graph node is added.
        let magnitude_node = self.node(magnitude)?;
        let magnitude_shape = magnitude_node.shape.clone();
        let magnitude_dtype = magnitude_node.dtype;
        let sign_node = self.node(sign)?;
        let sign_shape = sign_node.shape.clone();
        let sign_dtype = sign_node.dtype;
        let plan = copysign_plan(
            &magnitude_shape,
            magnitude_dtype,
            &sign_shape,
            sign_dtype,
        )?;

        let magnitude = if magnitude_dtype == plan.operand_dtype {
            magnitude
        } else {
            self.cast(magnitude, plan.operand_dtype)?
        };
        let sign = if sign_dtype == plan.operand_dtype {
            sign
        } else {
            self.cast(sign, plan.operand_dtype)?
        };
        let operand_zero = self.constant(plan.operand_zero.clone());
        let reciprocal_zero = self.constant(plan.reciprocal_zero.clone());
        let negative = self.lt(sign, operand_zero)?;
        let reciprocal = self.reciprocal(sign)?;
        let reciprocal_negative = self.lt(reciprocal, reciprocal_zero)?;
        let negative = self.logical_or(negative, reciprocal_negative)?;
        let magnitude = self.abs(magnitude)?;
        let output = self.select(negative, self.neg(magnitude)?, magnitude)?;
        debug_assert_eq!(self.shape(output).expect("copysign preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("copysign preflighted"), plan.operand_dtype);
        debug_assert_eq!(self.shape(sign).expect("copysign preflighted"), &plan.sign_shape);
        debug_assert_eq!(self.shape(magnitude).expect("copysign preflighted"), &plan.magnitude_shape);
        debug_assert_eq!(self.dtype(reciprocal).expect("copysign preflighted"), plan.reciprocal_dtype);
        Ok(output)
    }
    /// Source-compatible scalar-right form of tinygrad's
    /// `Tensor.copysign(other)`. The live magnitude tensor remains lhs; no
    /// reflected scalar-magnitude surface is exposed by this method.
    pub fn copysign_scalar(&mut self, magnitude: NodeId, sign: Scalar) -> Result<NodeId> {
        let plan = copysign_scalar_plan(self, magnitude, sign)?;
        // `copysign_scalar_plan` completed all fallible descriptor and byte
        // validation before this scalar is published. Reuse the live public
        // literal lowerer so scalar and tensor signs share exact predicates,
        // source casts, branch ordering, and VJP behavior.
        let sign = self.constant(plan.scalar);
        let output = self.copysign(magnitude, sign)?;
        debug_assert_eq!(self.dtype(magnitude).expect("copysign scalar preflighted"), plan.magnitude_dtype);
        debug_assert_eq!(self.shape(output).expect("copysign scalar preflighted"), &plan.core.output_shape);
        debug_assert_eq!(self.dtype(output).expect("copysign scalar preflighted"), plan.core.operand_dtype);
        Ok(output)
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
        let validated = clamp_plan(self, input, min, max)?;
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
        debug_assert_eq!(self.shape(value).expect("clamp preflighted"), &validated.output_shape);
        debug_assert_eq!(self.dtype(value).expect("clamp preflighted"), validated.output_dtype);
        debug_assert_eq!(validated.lower.as_ref().map(|stage| stage.bound), min);
        debug_assert_eq!(validated.upper.as_ref().map(|stage| stage.bound), max);
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

        self.lower_leaky_relu(input, input_dtype, slope, slope_dtype, plan)
    }

    /// Source-compatible scalar-slope form of tinygrad
    /// `Tensor.leaky_relu(neg_slope=...)`. This is deliberately separate from
    /// [`Self::leaky_relu`], whose slope remains a live graph value for
    /// existing RustGrad callers.
    pub fn leaky_relu_scalar(&mut self, input: NodeId, neg_slope: f64) -> Result<NodeId> {
        self.leaky_relu_with_scalar(input, Scalar::F(neg_slope))
    }

    /// Source-compatible untyped concrete-scalar slope form. This preserves
    /// tinygrad's runtime `ConstType` surface without changing the established
    /// f64-only [`Self::leaky_relu_scalar`] API.
    pub fn leaky_relu_with_scalar(
        &mut self,
        input: NodeId,
        neg_slope: Scalar,
    ) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let plan = leaky_relu_scalar_plan(&input_shape, input_dtype, neg_slope)?;
        let slope_dtype = plan.slope.dtype();
        let slope = self.constant(plan.slope);
        self.lower_leaky_relu(input, input_dtype, slope, slope_dtype, plan.core)
    }

    /// Checked-in tinygrad's `Tensor.leaky_relu()` default slope.
    pub fn leaky_relu_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.leaky_relu_scalar(input, 0.01)
    }

    fn lower_leaky_relu(
        &mut self,
        input: NodeId,
        input_dtype: DType,
        slope: NodeId,
        slope_dtype: DType,
        plan: LeakyReluPlan,
    ) -> Result<NodeId> {

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
        self.hardsigmoid_scalar(input, 1.0 / 6.0, 0.5)
    }

    /// Source-compatible `Tensor.hardsigmoid(alpha, beta)` public float
    /// form. The checked-in source is the literal
    /// `(alpha * x + beta).relu() - (alpha * x + beta - 1).relu()`.
    pub fn hardsigmoid_scalar(
        &mut self,
        input: NodeId,
        alpha: f64,
        beta: f64,
    ) -> Result<NodeId> {
        let plan = hardsigmoid_scalar_plan(self, input, alpha, beta)?;
        let output_shape = plan.core.output_shape.clone();
        let output_dtype = plan.core.output_dtype;
        let alpha = self.constant(plan.alpha);
        let beta = self.constant(plan.beta);
        let output = self.lower_hardsigmoid(input, alpha, beta, plan.core)?;
        debug_assert_eq!(self.shape(output).expect("Hardsigmoid scalar preflighted"), &output_shape);
        debug_assert_eq!(self.dtype(output).expect("Hardsigmoid scalar preflighted"), output_dtype);
        Ok(output)
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
    pub fn gelu_default(&mut self, input: NodeId) -> Result<NodeId> {
        // Public Tensor.gelu() defaults to the tanh approximation. This is
        // intentionally distinct from the ONNX Gelu handler, whose omitted
        // attribute selects exact `none` in checked-in tinygrad.
        self.gelu(input, "tanh")
    }

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
        let plan = elu_plan(&input_shape, input_dtype, &alpha_shape, alpha_dtype)?;
        self.lower_elu(input, alpha, plan)
    }

    /// Source-compatible scalar-alpha form of checked-in tinygrad
    /// `Tensor.elu(alpha=...)`, preserving the live-alpha [`Self::elu`] API.
    pub fn elu_scalar(&mut self, input: NodeId, alpha: f64) -> Result<NodeId> {
        self.elu_with_scalar(input, Scalar::F(alpha))
    }

    /// Source-compatible untyped concrete-scalar alpha form. This preserves
    /// tinygrad's runtime `ConstType` surface without altering the established
    /// f64-only [`Self::elu_scalar`] callers.
    pub fn elu_with_scalar(&mut self, input: NodeId, alpha: Scalar) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let plan = elu_scalar_plan(&input_shape, input_dtype, alpha)?;
        let alpha = self.constant(plan.alpha);
        self.lower_elu(input, alpha, plan.core)
    }

    /// Checked-in tinygrad's `Tensor.elu()` default alpha.
    pub fn elu_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.elu_scalar(input, 1.0)
    }

    fn lower_elu(&mut self, input: NodeId, alpha: NodeId, plan: EluPlan) -> Result<NodeId> {
        let zero_input = self.constant(plan.zero_input);
        let positive_condition = self.gt(input, zero_input)?;
        let positive = self.select(positive_condition, input, zero_input)?;
        let exp = self.exp(input)?;
        debug_assert_eq!(self.dtype(exp).expect("ELU preflighted"), plan.exp_dtype);
        let one_exp = self.constant(plan.one_exp);
        let negative_raw = self.sub(one_exp, exp)?;
        let zero_exp = self.constant(plan.zero_exp);
        let negative_condition = self.gt(negative_raw, zero_exp)?;
        let negative_relu = self.select(negative_condition, negative_raw, zero_exp)?;
        let negative = self.mul(alpha, negative_relu)?;
        debug_assert_eq!(self.shape(positive).expect("ELU preflighted"), &plan.positive_shape);
        debug_assert_eq!(self.shape(negative_raw).expect("ELU preflighted"), &plan.negative_shape);
        debug_assert_eq!(self.shape(negative).expect("ELU preflighted"), &plan.scaled_shape);
        debug_assert_eq!(self.dtype(negative).expect("ELU preflighted"), plan.scaled_dtype);
        let output = self.sub(positive, negative)?;
        debug_assert_eq!(self.shape(output).expect("ELU preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("ELU preflighted"), plan.output_dtype);
        Ok(output)
    }
    pub fn celu(&mut self, input: NodeId, alpha: NodeId) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let alpha_node = self.node(alpha)?;
        let alpha_shape = alpha_node.shape.clone();
        let alpha_dtype = alpha_node.dtype;
        let plan = celu_plan(&input_shape, input_dtype, &alpha_shape, alpha_dtype)?;

        self.lower_celu(input, input_dtype, alpha, alpha_dtype, plan)
    }

    /// Source-compatible scalar-alpha form of checked-in tinygrad
    /// `Tensor.celu(alpha=...)`. The established [`Self::celu`] API continues
    /// to accept a live alpha tensor unchanged.
    pub fn celu_scalar(&mut self, input: NodeId, alpha: f64) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let plan = celu_scalar_plan(&input_shape, input_dtype, alpha)?;
        let alpha_dtype = plan.alpha.dtype();
        let alpha = self.constant(plan.alpha);
        self.lower_celu(input, input_dtype, alpha, alpha_dtype, plan.core)
    }

    /// Checked-in tinygrad's `Tensor.celu()` default alpha.
    pub fn celu_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.celu_scalar(input, 1.0)
    }

    fn lower_celu(
        &mut self,
        input: NodeId,
        input_dtype: DType,
        alpha: NodeId,
        alpha_dtype: DType,
        plan: CeluPlan,
    ) -> Result<NodeId> {

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
        let plan = selu_plan(&input_shape,input_dtype,&alpha_shape,alpha_dtype,&gamma_shape,gamma_dtype)?;
        self.lower_selu(input, alpha, gamma, plan)
    }

    /// Scalar form of checked-in `Tensor.selu(alpha=..., gamma=...)`.
    pub fn selu_scalar(&mut self, input: NodeId, alpha: f64, gamma: f64) -> Result<NodeId> {
        let node = self.node(input)?;
        let plan = selu_scalar_plan(&node.shape, node.dtype, alpha, gamma)?;
        let alpha = self.constant(plan.alpha);
        let gamma = self.constant(plan.gamma);
        self.lower_selu(input, alpha, gamma, plan.core)
    }

    /// Checked-in tinygrad's `Tensor.selu()` defaults.
    pub fn selu_default(&mut self, input: NodeId) -> Result<NodeId> { self.selu_scalar(input, 1.67326, 1.0507) }

    fn lower_selu(&mut self, input: NodeId, alpha: NodeId, gamma: NodeId, plan: SeluPlan) -> Result<NodeId> {
        let zero_input = self.constant(plan.zero_input);
        let condition = self.ge(input, zero_input)?;
        let exp = self.exp(input)?;
        let one_exp = self.constant(plan.one_exp);
        let negative_raw = self.sub(exp, one_exp)?;
        let negative = self.mul(alpha, negative_raw)?;
        let branch = self.select(condition, input, negative)?;
        debug_assert_eq!(self.dtype(exp).expect("SELU preflighted"), plan.exp_dtype);
        debug_assert_eq!(self.shape(condition).expect("SELU preflighted"), &plan.condition_shape);
        debug_assert_eq!(self.shape(negative).expect("SELU preflighted"), &plan.negative_shape);
        debug_assert_eq!(self.dtype(negative).expect("SELU preflighted"), plan.negative_dtype);
        debug_assert_eq!(self.shape(branch).expect("SELU preflighted"), &plan.branch_shape);
        debug_assert_eq!(self.dtype(branch).expect("SELU preflighted"), plan.branch_dtype);
        let output = self.mul(gamma, branch)?;
        debug_assert_eq!(self.shape(output).expect("SELU preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("SELU preflighted"), plan.output_dtype);
        Ok(output)
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
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        let plan = logsigmoid_plan(&shape, dtype)?;
        let negated = self.neg(input)?;
        let beta = self.constant(plan.beta);
        let softplus = self.softplus(negated, beta)?;
        let output = self.neg(softplus)?;
        debug_assert_eq!(self.shape(softplus).expect("logsigmoid preflighted"), &plan.shape);
        debug_assert_eq!(self.shape(output).expect("logsigmoid preflighted"), &plan.shape);
        debug_assert_eq!(self.dtype(negated).expect("logsigmoid preflighted"), plan.negated_dtype);
        debug_assert_eq!(self.dtype(output).expect("logsigmoid preflighted"), plan.output_dtype);
        // These nested weak constants were validated above; Softplus creates
        // its own typed instances while lowering the literal source graph.
        debug_assert_eq!(plan.softplus_zero.dtype(), if plan.output_dtype.is_float() { plan.output_dtype } else { DType::F32 });
        debug_assert_eq!(plan.softplus_one.dtype(), plan.output_dtype);
        Ok(output)
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
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        let plan = log10_plan(&shape, dtype)?;
        let log = self.log2(input)?;
        let scale = self.constant(plan.scale);
        let output = self.mul(log, scale)?;
        debug_assert_eq!(self.shape(output).expect("log10 preflighted"), &plan.shape);
        debug_assert_eq!(self.dtype(log).expect("log10 preflighted"), plan.log_dtype);
        debug_assert_eq!(self.dtype(output).expect("log10 preflighted"), plan.log_dtype);
        Ok(output)
    }
    pub fn logaddexp(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let (lhs_shape, lhs_dtype) = {
            let source = self.node(lhs)?;
            (source.shape.clone(), source.dtype)
        };
        let (rhs_shape, rhs_dtype) = {
            let source = self.node(rhs)?;
            (source.shape.clone(), source.dtype)
        };
        let plan = logaddexp_plan(&lhs_shape, lhs_dtype, &rhs_shape, rhs_dtype)?;
        let lhs = if lhs_dtype == plan.operand_dtype {
            lhs
        } else {
            self.cast(lhs, plan.operand_dtype)?
        };
        let rhs = if rhs_dtype == plan.operand_dtype {
            rhs
        } else {
            self.cast(rhs, plan.operand_dtype)?
        };
        let maximum = self.maximum(lhs, rhs)?;
        let left = self.sub(lhs, maximum)?;
        let right = self.sub(rhs, maximum)?;
        let left_exp = self.exp(left)?;
        let right_exp = self.exp(right)?;
        let sum = self.add(left_exp, right_exp)?;
        let log = self.log(sum)?;
        let output = self.add(log, maximum)?;
        debug_assert_eq!(self.shape(output).expect("logaddexp preflighted"), &plan.shape);
        debug_assert_eq!(self.dtype(left_exp).expect("logaddexp preflighted"), plan.exp_dtype);
        debug_assert_eq!(self.dtype(right_exp).expect("logaddexp preflighted"), plan.exp_dtype);
        debug_assert_eq!(self.dtype(output).expect("logaddexp preflighted"), plan.output_dtype);
        Ok(output)
    }
    /// Source-compatible scalar-right form of tinygrad's
    /// `Tensor.logaddexp(other)`. The live tensor remains the left operand;
    /// tinygrad does not expose a reflected scalar-left method here.
    pub fn logaddexp_scalar(&mut self, lhs: NodeId, rhs: Scalar) -> Result<NodeId> {
        let plan = logaddexp_scalar_plan(self, lhs, rhs)?;
        // All fallible source-LUB and stable-composite checks have completed
        // before this weak scalar is made visible to the graph. Delegating to
        // the live helper preserves one shared Max/Sub/Exp/Add/Log formula.
        let rhs = self.constant(plan.scalar);
        let output = self.logaddexp(lhs, rhs)?;
        debug_assert_eq!(self.dtype(lhs).expect("logaddexp scalar preflighted"), plan.lhs_dtype);
        debug_assert_eq!(self.shape(output).expect("logaddexp scalar preflighted"), &plan.core.shape);
        debug_assert_eq!(self.dtype(output).expect("logaddexp scalar preflighted"), plan.core.output_dtype);
        Ok(output)
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
        let start_node = self.node(start)?;
        let start_shape = start_node.shape.clone();
        let start_dtype = start_node.dtype;
        let end_node = self.node(end)?;
        let end_shape = end_node.shape.clone();
        let end_dtype = end_node.dtype;
        let weight_node = self.node(weight)?;
        let weight_shape = weight_node.shape.clone();
        let weight_dtype = weight_node.dtype;
        let plan = lerp_plan(
            &start_shape,
            start_dtype,
            &end_shape,
            end_dtype,
            &weight_shape,
            weight_dtype,
        )?;

        if !plan.special_u8 {
            let delta = self.sub(end, start)?;
            let weighted = self.mul(delta, weight)?;
            let output = self.add(start, weighted)?;
            debug_assert_eq!(self.shape(delta).expect("lerp preflighted"), &end_shape.broadcast_with(&start_shape).expect("lerp preflighted"));
            debug_assert_eq!(self.dtype(delta).expect("lerp preflighted"), plan.difference_dtype);
            debug_assert_eq!(self.dtype(weighted).expect("lerp preflighted"), plan.weighted_dtype);
            debug_assert_eq!(self.shape(output).expect("lerp preflighted"), &plan.output_shape);
            debug_assert_eq!(self.dtype(output).expect("lerp preflighted"), plan.output_dtype);
            return Ok(output);
        }

        // Checked-in tinygrad has a tensor-weight U8 interpolation path that
        // is deliberately fixed-point, not the generic `start + delta*w`
        // expression. Keep every visible cast boundary: in particular, the
        // width-local `weight * 128` happens before integer weights widen for
        // the weak `.5` addition.
        let difference = self.cast(self.sub(end, start)?, DType::I8)?;
        let weight = if weight_dtype == plan.weight_scale_dtype {
            weight
        } else {
            self.cast(weight, plan.weight_scale_dtype)?
        };
        let scaled = self.mul(weight, self.constant(plan.scale.clone().expect("lerp U8 plan")))?;
        let scaled = if plan.weight_scale_dtype == plan.weight_fraction_dtype {
            scaled
        } else {
            self.cast(scaled, plan.weight_fraction_dtype)?
        };
        let weight = self.cast(
            self.add(scaled, self.constant(plan.half.clone().expect("lerp U8 plan")))?,
            DType::I16,
        )?;
        let product = self.mul(difference, weight)?;
        let rounded = self.add(
            product,
            self.constant(plan.rounding.clone().expect("lerp U8 plan")),
        )?;
        let shifted = self.shr(
            self.cast(rounded, DType::U16)?,
            self.constant(plan.shift.clone().expect("lerp U8 plan")),
        )?;
        let output = self.cast(self.add(start, shifted)?, DType::U8)?;
        debug_assert_eq!(self.shape(output).expect("lerp preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("lerp preflighted"), plan.output_dtype);
        Ok(output)
    }
    /// Source-compatible scalar-weight form of tinygrad's `Tensor.lerp`.
    /// Unlike [`Self::lerp`], this intentionally never takes the U8
    /// live-Tensor fixed-point branch.
    pub fn lerp_scalar(&mut self, start: NodeId, end: NodeId, weight: Scalar) -> Result<NodeId> {
        let plan = lerp_scalar_plan(self, start, end, weight)?;
        // The scalar is published only after the ordinary source composition
        // is completely planned. Each public operation then retains its own
        // literal `_broadcasted` lowering and compositional VJP.
        let weight = self.constant(plan.scalar);
        let difference = self.sub(end, start)?;
        let weighted = self.mul(difference, weight)?;
        let output = self.add(start, weighted)?;
        debug_assert_eq!(self.shape(difference).expect("lerp scalar preflighted"), &plan.difference_shape);
        debug_assert_eq!(self.dtype(difference).expect("lerp scalar preflighted"), plan.difference_dtype);
        debug_assert_eq!(self.shape(weighted).expect("lerp scalar preflighted"), &plan.weighted_shape);
        debug_assert_eq!(self.dtype(weighted).expect("lerp scalar preflighted"), plan.weighted_dtype);
        debug_assert_eq!(self.shape(output).expect("lerp scalar preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("lerp scalar preflighted"), plan.output_dtype);
        Ok(output)
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

    /// Source-compatible scalar/default entry point for tinygrad
    /// `Tensor.isclose`. Unlike [`Self::isclose`], this owns the Python-float
    /// weak constants and therefore has no live tolerance tensors in its
    /// public contract.
    pub fn isclose_scalar(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        rtol: f64,
        atol: f64,
        equal_nan: bool,
    ) -> Result<NodeId> {
        let lhs_node = self.node(lhs)?;
        let lhs_shape = lhs_node.shape.clone();
        let lhs_dtype = lhs_node.dtype;
        let rhs_node = self.node(rhs)?;
        let rhs_shape = rhs_node.shape.clone();
        let rhs_dtype = rhs_node.dtype;
        let plan = isclose_scalar_plan(
            &lhs_shape,
            lhs_dtype,
            &rhs_shape,
            rhs_dtype,
            rtol,
            atol,
            equal_nan,
        )?;

        // Keep the checked-in literal ordering. `other.abs()` owns the weak
        // tolerance width independently of `self - other`, and equal_nan is
        // a Boolean scalar operand rather than a host-side graph shortcut.
        let lhs_finite = self.isfinite(lhs)?;
        let rhs_finite = self.isfinite(rhs)?;
        let raw_difference = self.sub(lhs, rhs)?;
        let difference = self.abs(raw_difference)?;
        let abs_rhs = self.abs(rhs)?;
        let rtol = self.constant(plan.rtol.clone());
        let atol = self.constant(plan.atol.clone());
        let relative = self.mul(rtol, abs_rhs)?;
        let tolerance = self.add(atol, relative)?;
        let near = self.le(difference, tolerance)?;
        let finite = self.logical_and(lhs_finite, rhs_finite)?;
        let finite_near = self.logical_and(finite, near)?;
        let lhs_inf = self.isinf(lhs)?;
        let rhs_inf = self.isinf(rhs)?;
        let infinities = self.logical_or(lhs_inf, rhs_inf)?;
        let equal = self.eq(lhs, rhs)?;
        let same_infinity = self.logical_and(infinities, equal)?;
        let result = self.logical_or(finite_near, same_infinity)?;
        let lhs_nan = self.isnan(lhs)?;
        let rhs_nan = self.isnan(rhs)?;
        let both_nan = self.logical_and(lhs_nan, rhs_nan)?;
        let nan_close = self.logical_and(both_nan, self.constant(plan.equal_nan.clone()))?;
        let output = self.logical_or(result, nan_close)?;
        debug_assert_eq!(self.shape(output).expect("isclose scalar preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("isclose scalar preflighted"), DType::Bool);
        debug_assert_eq!(self.dtype(raw_difference).expect("isclose scalar preflighted"), plan.difference_dtype);
        debug_assert_eq!(self.dtype(relative).expect("isclose scalar preflighted"), plan.tolerance_dtype);
        debug_assert_eq!(self.dtype(tolerance).expect("isclose scalar preflighted"), plan.tolerance_dtype);
        debug_assert_eq!(self.dtype(difference).expect("isclose scalar preflighted").promote(self.dtype(tolerance).expect("isclose scalar preflighted")), plan.comparison_dtype);
        Ok(output)
    }

    /// Checked-in tinygrad's parameterless `Tensor.isclose(other)` defaults.
    pub fn isclose_default(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.isclose_scalar(lhs, rhs, 1e-5, 1e-8, false)
    }

    /// Reduces tinygrad's public elementwise `isclose` predicate to one Bool
    /// scalar. Python float tolerances are committed at `rhs.abs()`'s source
    /// width before any constant or graph node is published.
    pub fn allclose(
        &mut self,
        lhs: NodeId,
        rhs: NodeId,
        rtol: f64,
        atol: f64,
        equal_nan: bool,
    ) -> Result<NodeId> {
        let (lhs_shape, lhs_dtype) = {
            let source = self.node(lhs)?;
            (source.shape.clone(), source.dtype)
        };
        let (rhs_shape, rhs_dtype) = {
            let source = self.node(rhs)?;
            (source.shape.clone(), source.dtype)
        };
        let plan = allclose_plan(&lhs_shape, lhs_dtype, &rhs_shape, rhs_dtype, rtol, atol)?;
        // Source is exactly `self.isclose(other, rtol, atol, equal_nan).all()`;
        // use the scalar isclose plan so its weak Python tolerances and full
        // Bool special-value tree are validated before either is published.
        let close = self.isclose_scalar(lhs, rhs, rtol, atol, equal_nan)?;
        let output = self.all(close, None, false)?;
        debug_assert_eq!(self.shape(close).expect("allclose preflighted"), &plan.output_shape);
        debug_assert_eq!(self.shape(output).expect("allclose preflighted"), &Shape::new([]));
        debug_assert_eq!(self.dtype(output).expect("allclose preflighted"), DType::Bool);
        debug_assert_eq!(self.dtype(close).expect("allclose preflighted"), DType::Bool);
        Ok(output)
    }

    /// Checked-in tinygrad's parameterless `Tensor.allclose(other)` defaults.
    pub fn allclose_default(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        self.allclose(lhs, rhs, 1e-5, 1e-8, false)
    }
    pub fn floor(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.floor is `where(x < (b := trunc(x)), b - 1, b)`, rather
        // than raw FLOOR. Its branch composition preserves source tracing and
        // nondifferentiability while retaining the exact storage dtype.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let dtype = source.dtype;
        let trunc_dtype = unary_dtype(UnaryOp::Trunc, dtype);
        let one = TensorData::scalar_with_dtype(Scalar::I(1), dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        for dtype in [dtype, trunc_dtype, DType::Bool, dtype, dtype] {
            extent(&shape, dtype)?;
        }
        extent(one.shape(), one.dtype())?;
        if trunc_dtype != dtype
            || one.dtype() != dtype
            || dtype.promote(dtype) != dtype
            || shape.broadcast_with(one.shape())? != shape
        {
            return Err(Error::InvalidElementwiseDType { op: "floor trunc/compare/select source promotion", actual: dtype });
        }
        let truncated = self.trunc(input)?;
        let condition = self.lt(input, truncated)?;
        let decremented = self.sub(truncated, self.constant(one))?;
        self.select(condition, decremented, truncated)
    }
    pub fn ceil(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.ceil is `where(x > (b := trunc(x)), b + 1, b)`, rather
        // than raw CEIL. Retain the source branch graph and its zero VJP.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let dtype = source.dtype;
        let trunc_dtype = unary_dtype(UnaryOp::Trunc, dtype);
        let one = TensorData::scalar_with_dtype(Scalar::I(1), dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        for dtype in [dtype, trunc_dtype, DType::Bool, dtype, dtype] {
            extent(&shape, dtype)?;
        }
        extent(one.shape(), one.dtype())?;
        if trunc_dtype != dtype
            || one.dtype() != dtype
            || dtype.promote(dtype) != dtype
            || shape.broadcast_with(one.shape())? != shape
        {
            return Err(Error::InvalidElementwiseDType { op: "ceil trunc/compare/select source promotion", actual: dtype });
        }
        let truncated = self.trunc(input)?;
        let condition = self.gt(input, truncated)?;
        let incremented = self.add(truncated, self.constant(one))?;
        self.select(condition, incremented, truncated)
    }
    pub fn trunc(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.trunc is the direct TRUNC ALU. Preserve raw semantics and
        // its explicit zero VJP, but validate both descriptors before a node
        // can be published for Floor/Ceil and division callers.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let output_dtype = unary_dtype(UnaryOp::Trunc, input_dtype);
        let extent = |shape: &Shape, dtype: DType| {
            shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        };
        extent(&shape, input_dtype)?;
        extent(&shape, output_dtype)?;
        if output_dtype != input_dtype {
            return Err(Error::InvalidElementwiseDType { op: "trunc output dtype", actual: output_dtype });
        }
        self.unary(UnaryOp::Trunc, input)
    }
    pub fn round(&mut self, input: NodeId) -> Result<NodeId> {
        // Tensor.round is the source ties-to-even composition, not raw ROUND:
        // `where((x>0).eq(trunc(trunc(x)/2).eq(trunc(x))), ceil(x-.5), floor(x+.5))`.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let dtype = source.dtype;
        let division_dtype = if dtype.is_float() { dtype } else { DType::F32 };
        let zero = TensorData::scalar_with_dtype(Scalar::I(0), dtype);
        let half = TensorData::scalar_with_dtype(Scalar::F(0.5), dtype);
        let two = TensorData::scalar_with_dtype(Scalar::F(2.0), dtype);
        let extent = |shape: &Shape, dtype: DType| shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()));
        for dtype in [dtype, DType::Bool, division_dtype, DType::Bool, DType::Bool, dtype, dtype, dtype, dtype] { extent(&shape, dtype)?; }
        for scalar in [&zero, &half, &two] {
            extent(scalar.shape(), scalar.dtype())?;
            if scalar.dtype() != dtype || shape.broadcast_with(scalar.shape())? != shape {
                return Err(Error::InvalidElementwiseDType { op: "round source scalar promotion", actual: scalar.dtype() });
            }
        }
        if unary_dtype(UnaryOp::Trunc, dtype) != dtype || (!dtype.is_float() && division_dtype != DType::F32) || (dtype.is_float() && division_dtype != dtype) {
            return Err(Error::InvalidElementwiseDType { op: "round trunc/div/select source promotion", actual: division_dtype });
        }
        let truncated = self.trunc(input)?;
        let positive = self.gt(input, self.constant(zero))?;
        let half_truncated = self.trunc(self.div(truncated, self.constant(two))?)?;
        let even = self.eq(half_truncated, truncated)?;
        let condition = self.eq(positive, even)?;
        let lower = self.ceil(self.sub(input, self.constant(half.clone()))?)?;
        let upper = self.floor(self.add(input, self.constant(half))?)?;
        self.select(condition, lower, upper)
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
        // Tensor.isnan is literal self-inequality, preserving the public
        // typed-comparison graph rather than exposing raw ISNAN.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let dtype = source.dtype;
        let extent = |shape: &Shape, dtype: DType| shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()));
        extent(&shape, dtype)?;
        extent(&shape, DType::Bool)?;
        if dtype.promote(dtype) != dtype {
            return Err(Error::InvalidElementwiseDType { op: "isnan self comparison promotion", actual: dtype });
        }
        self.ne(input, input)
    }
    pub fn isinf(&mut self, input: NodeId) -> Result<NodeId> {
        // Default Tensor.isinf enables both source sign predicates. Raw
        // ISINF is value-equivalent; retain it while validating descriptors
        // before any node is published.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let extent = |shape: &Shape, dtype: DType| shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()));
        extent(&shape, input_dtype)?;
        extent(&shape, DType::Bool)?;
        if unary_dtype(UnaryOp::IsInf, input_dtype) != DType::Bool {
            return Err(Error::InvalidElementwiseDType { op: "isinf output dtype", actual: input_dtype });
        }
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
        // Tensor.isfinite is `(isinf() | isnan()).logical_not()`, not raw
        // ISFINITE. Validate every Bool intermediate before publication.
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let input_dtype = source.dtype;
        let extent = |shape: &Shape, dtype: DType| shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()));
        extent(&shape, input_dtype)?;
        for _ in 0..4 { extent(&shape, DType::Bool)?; }
        let infinite = self.isinf(input)?;
        let nan = self.isnan(input)?;
        self.logical_not(self.logical_or(infinite, nan)?)
    }
    pub fn relu(&mut self, input: NodeId) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let plan = relu_plan(&input_node.shape, input_node.dtype)?;
        let zero = self.constant(plan.zero);
        // Keep tinygrad's literal strict predicate. Equality, either signed
        // zero, and unordered NaN all select the canonical typed scalar zero;
        // the true branch preserves the input payload unchanged.
        let positive = self.gt(input, zero)?;
        let output = self.select(positive, input, zero)?;
        debug_assert_eq!(self.shape(output).expect("ReLU preflighted"), &plan.shape);
        debug_assert_eq!(self.dtype(output).expect("ReLU preflighted"), plan.dtype);
        Ok(output)
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
