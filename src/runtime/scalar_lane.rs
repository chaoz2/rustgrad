use crate::{BinaryOp, CompareOp, DType, LaneInstruction, TypedValue, UOp, UnaryOp};

pub(crate) mod dialect_seal {
    pub trait Sealed {}
}

/// Backend syntax boundary for one validated scalar lane instruction.
///
/// Operation semantics remain centralized in [`emit_scalar_lane`]. Dialects
/// provide only language spelling, storage commitment, and exact signed-wrap
/// primitives; they do not own a second operation taxonomy.
pub(crate) trait ScalarLaneDialect: dialect_seal::Sealed {
    fn name(&self) -> &'static str;
    fn supports_value(&self, dtype: DType) -> bool;
    fn cast(&self, source: DType, target: DType, value: &str) -> Result<String, String>;
    fn finish_float(&self, dtype: DType, value: String) -> Result<String, String>;
    fn signed_infix(
        &self,
        dtype: DType,
        operator: &'static str,
        lhs: &str,
        rhs: &str,
    ) -> Result<String, String>;
    fn signed_neg(&self, dtype: DType, value: &str) -> Result<String, String>;
    fn unsigned_neg(&self, dtype: DType, value: &str) -> Result<String, String>;
    fn signed_abs(&self, dtype: DType, value: &str) -> Result<String, String>;
    fn float_abs(&self, value: &str) -> String;
    fn bool_value(&self, expression: String) -> String;
    fn select(&self, condition: &str, on_true: &str, on_false: &str) -> String;
    fn compare_operand(&self, dtype: DType, value: &str) -> String {
        let _ = dtype;
        value.into()
    }
    fn call_intrinsic(&self, canonical_name: &'static str, value: &str) -> String;
    fn float_one(&self, dtype: DType) -> Result<&'static str, String>;
}

fn common_intrinsic_name(op: UnaryOp) -> Option<&'static str> {
    Some(match op {
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Exp2 => "exp2",
        UnaryOp::Log2 => "log2",
        UnaryOp::Sin => "sin",
        UnaryOp::Trunc => "trunc",
        _ => return None,
    })
}

pub(crate) fn project_scalar_lane(
    node: &UOp,
    sources: &[String],
) -> Result<Option<LaneInstruction<String>>, String> {
    const OUTPUT: u32 = u32::MAX;
    let instruction = crate::linearize::project_lane_instruction(node, OUTPUT, |slot, _| {
        u32::try_from(slot).map_err(|_| crate::LinearizeError::Overflow)
    })
    .map_err(|error| error.to_string())?;
    instruction
        .map(|instruction| {
            instruction.map_operands(|register| {
                if *register == OUTPUT {
                    Ok(String::new())
                } else {
                    sources
                        .get(*register as usize)
                        .cloned()
                        .ok_or_else(|| "scalar-lane source slot is absent".to_string())
                }
            })
        })
        .transpose()
}

fn unsupported(
    dialect: &impl ScalarLaneDialect,
    family: &str,
    detail: impl std::fmt::Debug,
) -> String {
    format!(
        "{family} {detail:?} is outside the exact {} scalar-lane subset",
        dialect.name()
    )
}

fn ensure_scalar_lanes<R>(instruction: &LaneInstruction<R>) -> Result<(), String> {
    let view = instruction.view();
    if view.result_type().is_some_and(|ty| ty.lanes != 1)
        || view.typed_inputs().any(|(_, ty)| ty.lanes != 1)
    {
        return Err("device scalar-lane lowering rejects vector UTypes".into());
    }
    Ok(())
}

fn graph_unary(
    dialect: &impl ScalarLaneDialect,
    op: UnaryOp,
    input_dtype: DType,
    output_dtype: DType,
    value: &str,
) -> Result<String, String> {
    if !dialect.supports_value(input_dtype) || !dialect.supports_value(output_dtype) {
        return Err(unsupported(
            dialect,
            "unary dtype",
            (op, input_dtype, output_dtype),
        ));
    }
    let converted;
    let value = if matches!(
        op,
        UnaryOp::Reciprocal
            | UnaryOp::Sqrt
            | UnaryOp::Exp2
            | UnaryOp::Log2
            | UnaryOp::Sin
            | UnaryOp::Trunc
    ) && input_dtype != output_dtype
    {
        converted = dialect.cast(input_dtype, output_dtype, value)?;
        converted.as_str()
    } else {
        value
    };
    match op {
        UnaryOp::Neg => match output_dtype {
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 => {
                dialect.finish_float(output_dtype, format!("(-({value}))"))
            }
            DType::I32 | DType::I64 => dialect.signed_neg(output_dtype, value),
            DType::U32 | DType::U64 => dialect.unsigned_neg(output_dtype, value),
            DType::Bool => Ok(dialect.bool_value(format!("!({value})"))),
            _ => Err(unsupported(dialect, "unary", (op, output_dtype))),
        },
        UnaryOp::Abs => match output_dtype {
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 => {
                dialect.finish_float(output_dtype, dialect.float_abs(value))
            }
            DType::I32 | DType::I64 => dialect.signed_abs(output_dtype, value),
            DType::U32 | DType::U64 | DType::Bool => Ok(value.into()),
            _ => Err(unsupported(dialect, "unary", (op, output_dtype))),
        },
        UnaryOp::Reciprocal => match output_dtype {
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 => {
                let one = dialect.float_one(output_dtype)?;
                dialect.finish_float(output_dtype, format!("({one} / ({value}))"))
            }
            _ => Err(unsupported(dialect, "unary", (op, output_dtype))),
        },
        UnaryOp::Sqrt | UnaryOp::Exp2 | UnaryOp::Log2 | UnaryOp::Sin | UnaryOp::Trunc => {
            if !matches!(
                output_dtype,
                DType::F16 | DType::BF16 | DType::F32 | DType::F64
            ) {
                return Err(unsupported(dialect, "unary", (op, output_dtype)));
            }
            let name = common_intrinsic_name(op)
                .ok_or_else(|| unsupported(dialect, "unary", (op, output_dtype)))?;
            dialect.finish_float(output_dtype, dialect.call_intrinsic(name, value))
        }
        _ => Err(unsupported(dialect, "unary", (op, output_dtype))),
    }
}

fn bool_binary(
    dialect: &impl ScalarLaneDialect,
    op: BinaryOp,
    lhs: &str,
    rhs: &str,
) -> Result<String, String> {
    let expression = match op {
        BinaryOp::Add | BinaryOp::BitOr => format!("({lhs}) || ({rhs})"),
        BinaryOp::Sub | BinaryOp::BitXor => format!("({lhs}) != ({rhs})"),
        BinaryOp::Mul | BinaryOp::BitAnd => format!("({lhs}) && ({rhs})"),
        _ => return Err(unsupported(dialect, "binary", (op, DType::Bool))),
    };
    Ok(dialect.bool_value(expression))
}

fn promoted_operands(
    dialect: &impl ScalarLaneDialect,
    family: &str,
    lhs: &TypedValue<String>,
    rhs: &TypedValue<String>,
    expected_dtype: Option<DType>,
) -> Result<(DType, String, String), String> {
    let lhs_dtype = lhs.ty.scalar;
    let rhs_dtype = rhs.ty.scalar;
    let promoted = lhs_dtype.promote(rhs_dtype);
    if expected_dtype.is_some_and(|expected| expected != promoted)
        || !dialect.supports_value(lhs_dtype)
        || !dialect.supports_value(rhs_dtype)
        || !dialect.supports_value(promoted)
    {
        return Err(unsupported(
            dialect,
            family,
            (lhs_dtype, rhs_dtype, promoted, expected_dtype),
        ));
    }
    let lhs = if lhs_dtype == promoted {
        lhs.register.clone()
    } else {
        dialect.cast(lhs_dtype, promoted, &lhs.register)?
    };
    let rhs = if rhs_dtype == promoted {
        rhs.register.clone()
    } else {
        dialect.cast(rhs_dtype, promoted, &rhs.register)?
    };
    Ok((promoted, lhs, rhs))
}

fn graph_binary(
    dialect: &impl ScalarLaneDialect,
    op: BinaryOp,
    output_dtype: DType,
    lhs: &TypedValue<String>,
    rhs: &TypedValue<String>,
) -> Result<String, String> {
    let (dtype, lhs, rhs) =
        promoted_operands(dialect, "binary dtype", lhs, rhs, Some(output_dtype))?;
    if dtype == DType::Bool {
        return bool_binary(dialect, op, &lhs, &rhs);
    }
    let operator = match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div if matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) => "/",
        BinaryOp::BitAnd => "&",
        BinaryOp::BitOr => "|",
        BinaryOp::BitXor => "^",
        _ => return Err(unsupported(dialect, "binary", (op, dtype))),
    };
    let expression = match dtype {
        DType::I32 | DType::I64 if !matches!(op, BinaryOp::Div) => {
            dialect.signed_infix(dtype, operator, &lhs, &rhs)?
        }
        DType::U32 | DType::U64
            if matches!(
                op,
                BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::BitAnd
                    | BinaryOp::BitOr
                    | BinaryOp::BitXor
            ) =>
        {
            format!("(({lhs}) {operator} ({rhs}))")
        }
        DType::F16 | DType::BF16 | DType::F32 | DType::F64
            if matches!(
                op,
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
            ) =>
        {
            format!("(({lhs}) {operator} ({rhs}))")
        }
        _ => return Err(unsupported(dialect, "binary", (op, dtype))),
    };
    if matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
        dialect.finish_float(dtype, expression)
    } else {
        Ok(expression)
    }
}

fn compare(
    dialect: &impl ScalarLaneDialect,
    op: CompareOp,
    lhs: &TypedValue<String>,
    rhs: &TypedValue<String>,
) -> Result<String, String> {
    let lhs_dtype = lhs.ty.scalar;
    let rhs_dtype = rhs.ty.scalar;
    if lhs_dtype != rhs_dtype
        || !dialect.supports_value(lhs_dtype)
        || !dialect.supports_value(rhs_dtype)
    {
        return Err(unsupported(
            dialect,
            "compare dtype",
            (lhs_dtype, rhs_dtype),
        ));
    }
    // GraphCompare intentionally retains each source dtype. Its CPU oracle
    // has exact mixed signed/unsigned and float/integer ordering that cannot
    // be represented uniformly by OpenCL C, MSL, and WGSL promotion rules.
    // Keep heterogeneous comparisons fail-closed until that semantic is a
    // first-class shared instruction rather than silently narrowing a lane.
    let dtype = lhs_dtype;
    let operator = match op {
        CompareOp::Eq => "==",
        CompareOp::Ne => "!=",
        CompareOp::Lt => "<",
        CompareOp::Le => "<=",
        CompareOp::Gt => ">",
        CompareOp::Ge => ">=",
    };
    let lhs = dialect.compare_operand(dtype, &lhs.register);
    let rhs = dialect.compare_operand(dtype, &rhs.register);
    Ok(dialect.bool_value(format!("({lhs}) {operator} ({rhs})")))
}

/// Emits one exact pure scalar-lane instruction. Memory, control, and Bitcast
/// remain explicit fail-closed boundaries for the accelerator renderers.
pub(crate) fn emit_scalar_lane(
    dialect: &impl ScalarLaneDialect,
    instruction: &LaneInstruction<String>,
) -> Result<String, String> {
    ensure_scalar_lanes(instruction)?;
    match instruction {
        LaneInstruction::Cast { output, input } => {
            dialect.cast(input.ty.scalar, output.ty.scalar, &input.register)
        }
        LaneInstruction::GraphUnary { output, input, op } => graph_unary(
            dialect,
            *op,
            input.ty.scalar,
            output.ty.scalar,
            &input.register,
        ),
        LaneInstruction::GraphBinary {
            output,
            lhs,
            rhs,
            op,
        } => graph_binary(dialect, *op, output.ty.scalar, lhs, rhs),
        LaneInstruction::CoreBinary {
            output,
            lhs,
            rhs,
            op,
        } => {
            let op = match op {
                crate::uop::Binary::Add => BinaryOp::Add,
                crate::uop::Binary::Sub => BinaryOp::Sub,
                crate::uop::Binary::Mul => BinaryOp::Mul,
                crate::uop::Binary::And => BinaryOp::BitAnd,
                crate::uop::Binary::Or => BinaryOp::BitOr,
                _ => return Err(unsupported(dialect, "core binary", op)),
            };
            graph_binary(dialect, op, output.ty.scalar, lhs, rhs)
        }
        LaneInstruction::CoreEq { lhs, rhs, .. } => compare(dialect, CompareOp::Eq, lhs, rhs),
        LaneInstruction::CoreLt { lhs, rhs, .. } => compare(dialect, CompareOp::Lt, lhs, rhs),
        LaneInstruction::CoreLe { lhs, rhs, .. } => compare(dialect, CompareOp::Le, lhs, rhs),
        LaneInstruction::LogicalNot { input, .. } => {
            Ok(dialect.bool_value(format!("!({})", input.register)))
        }
        LaneInstruction::LogicalAnd { lhs, rhs, .. } => {
            Ok(dialect.bool_value(format!("({}) && ({})", lhs.register, rhs.register)))
        }
        LaneInstruction::LogicalOr { lhs, rhs, .. } => {
            Ok(dialect.bool_value(format!("({}) || ({})", lhs.register, rhs.register)))
        }
        LaneInstruction::Compare { lhs, rhs, op, .. } => compare(dialect, *op, lhs, rhs),
        LaneInstruction::Select {
            condition,
            on_true,
            on_false,
            ..
        } => Ok(dialect.select(&condition.register, &on_true.register, &on_false.register)),
        LaneInstruction::Bitcast { .. } => Err(format!(
            "Bitcast is outside the exact {} scalar-lane subset",
            dialect.name()
        )),
        LaneInstruction::Constant { .. }
        | LaneInstruction::Address { .. }
        | LaneInstruction::Range { .. }
        | LaneInstruction::Index { .. }
        | LaneInstruction::Load { .. }
        | LaneInstruction::CoreUnary { .. }
        | LaneInstruction::Store { .. } => Err(format!(
            "{} is not a pure scalar-lane expression for {}",
            instruction.view().semantic_name,
            dialect.name()
        )),
    }
}
