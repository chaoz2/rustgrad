use super::FuzzCase;

fn zero_values(case: &FuzzCase) -> FuzzCase {
    match case {
        FuzzCase::Binary { op, lhs, rhs } => FuzzCase::Binary {
            op: *op,
            lhs: lhs.zeroed(),
            rhs: rhs.zeroed(),
        },
        FuzzCase::Select {
            condition,
            on_true,
            on_false,
        } => FuzzCase::Select {
            condition: condition.zeroed(),
            on_true: on_true.zeroed(),
            on_false: on_false.zeroed(),
        },
        FuzzCase::Cast { input, to } => FuzzCase::Cast {
            input: input.zeroed(),
            to: *to,
        },
        FuzzCase::AffineView {
            input,
            start,
            end,
            expand,
        } => FuzzCase::AffineView {
            input: input.zeroed(),
            start: *start,
            end: *end,
            expand: *expand,
        },
        FuzzCase::Reduction {
            input,
            reduction,
            axis,
            keepdim,
        } => FuzzCase::Reduction {
            input: input.zeroed(),
            reduction: *reduction,
            axis: *axis,
            keepdim: *keepdim,
        },
        FuzzCase::Concat { lhs, rhs, axis } => FuzzCase::Concat {
            lhs: lhs.zeroed(),
            rhs: rhs.zeroed(),
            axis: *axis,
        },
        FuzzCase::Matmul { lhs, rhs } => FuzzCase::Matmul {
            lhs: lhs.zeroed(),
            rhs: rhs.zeroed(),
        },
    }
}

fn scalarize(case: &FuzzCase) -> Option<FuzzCase> {
    match case {
        FuzzCase::Binary { op, lhs, rhs } => Some(FuzzCase::Binary {
            op: *op,
            lhs: lhs.scalar_prefix()?,
            rhs: rhs.scalar_prefix()?,
        }),
        FuzzCase::Select {
            condition,
            on_true,
            on_false,
        } => Some(FuzzCase::Select {
            condition: condition.scalar_prefix()?,
            on_true: on_true.scalar_prefix()?,
            on_false: on_false.scalar_prefix()?,
        }),
        FuzzCase::Cast { input, to } => Some(FuzzCase::Cast {
            input: input.scalar_prefix()?,
            to: *to,
        }),
        _ => None,
    }
}

/// Deterministically shrinks values before shapes and accepts a candidate only
/// when `reproduces` confirms the same mismatch. The returned case therefore
/// always preserves the caller's reproduction predicate.
pub fn minimize_case(case: &FuzzCase, mut reproduces: impl FnMut(&FuzzCase) -> bool) -> FuzzCase {
    let mut current = case.clone();
    let zero = zero_values(&current);
    if zero != current && zero.validate().is_ok() && reproduces(&zero) {
        current = zero;
    }
    if let Some(scalar) = scalarize(&current)
        && scalar != current
        && scalar.validate().is_ok()
        && reproduces(&scalar)
    {
        current = scalar;
    }
    current
}
