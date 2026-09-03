use crate::uop::{
    AddressSpace, AddressValue, Binary, IndexAddressing, IndexValue, LiteralValue, Operation, UOp,
    UOpError,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub(crate) const MAX_PROJECTED_INDEX_DEPTH: usize = 256;
pub(crate) const MAX_PROJECTED_INDEX_NODES: usize = 4096;
// Compiler construction gets a separately enforced unique-DAG budget before
// canonicalization. It intentionally matches, rather than expands, the
// durable artifact occurrence budget above.
const MAX_GRAPH_PROJECTED_DAG_NODES: usize = MAX_PROJECTED_INDEX_NODES;

/// One completely validated explicit physical-address expression attached to
/// an ordinary Buffer index. Historical Buffer indices carry a Range as their
/// second source and retain descriptor-derived broadcast addressing; a
/// projected index carries this deliberately small core-integer UOp algebra.
///
/// Keeping the expression in the UOp DAG makes its identity and sharing
/// canonical. This plan is the sole semantic parser used by interpreters and
/// renderers, so backend code never rediscovers movement-operation policy.
pub(crate) struct ProjectedIndexPlan {
    pub(crate) buffer: u64,
    pub(crate) elements: usize,
    pub(crate) output_elements: usize,
    pub(crate) expression: ProjectedExpr<i64>,
    predicate: Option<ProjectedPredicate>,
    fits_i32: bool,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum ProjectedPredicate {
    Constant(bool),
    Compare {
        operation: crate::CompareOp,
        lhs: ProjectedExpr<i64>,
        rhs: ProjectedExpr<i64>,
    },
    Logical {
        operation: crate::LogicalOp,
        lhs: Box<Self>,
        rhs: Option<Box<Self>>,
    },
}

impl ProjectedPredicate {
    fn emit<E: ProjectedPredicateEmitter>(
        &self,
        emitter: &mut E,
    ) -> Result<E::Predicate, E::Error> {
        match self {
            Self::Constant(value) => emitter.boolean(*value),
            Self::Compare {
                operation,
                lhs,
                rhs,
            } => {
                let lhs = lhs.emit(emitter)?;
                let rhs = rhs.emit(emitter)?;
                emitter.compare(*operation, lhs, rhs)
            }
            Self::Logical {
                operation,
                lhs,
                rhs,
            } => {
                let lhs = lhs.emit(emitter)?;
                let rhs = rhs.as_ref().map(|rhs| rhs.emit(emitter)).transpose()?;
                emitter.logical(*operation, lhs, rhs)
            }
        }
    }

    fn is_always_false(&self) -> bool {
        matches!(self, Self::Constant(false))
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum ProjectedExpr<C> {
    Linear,
    Constant(C),
    Binary {
        operation: Binary,
        lhs: Arc<Self>,
        rhs: Arc<Self>,
    },
}

fn projected_expr_eq_with_visits(
    lhs: &ProjectedExpr<i64>,
    rhs: &ProjectedExpr<i64>,
) -> (bool, usize) {
    let mut pending = vec![(lhs, rhs)];
    let mut visited = HashSet::new();
    while let Some((lhs, rhs)) = pending.pop() {
        let pair = (
            lhs as *const ProjectedExpr<i64> as usize,
            rhs as *const ProjectedExpr<i64> as usize,
        );
        if !visited.insert(pair) {
            continue;
        }
        match (lhs, rhs) {
            (ProjectedExpr::Linear, ProjectedExpr::Linear) => {}
            (ProjectedExpr::Constant(lhs), ProjectedExpr::Constant(rhs)) if lhs == rhs => {}
            (
                ProjectedExpr::Binary {
                    operation: lhs_operation,
                    lhs: lhs_lhs,
                    rhs: lhs_rhs,
                },
                ProjectedExpr::Binary {
                    operation: rhs_operation,
                    lhs: rhs_lhs,
                    rhs: rhs_rhs,
                },
            ) if lhs_operation == rhs_operation => {
                pending.push((lhs_lhs, rhs_lhs));
                pending.push((lhs_rhs, rhs_rhs));
            }
            _ => return (false, visited.len()),
        }
    }
    (true, visited.len())
}

pub(crate) fn projected_expr_eq(lhs: &ProjectedExpr<i64>, rhs: &ProjectedExpr<i64>) -> bool {
    projected_expr_eq_with_visits(lhs, rhs).0
}

fn uop_structure_eq_with_visits(lhs: &UOp, rhs: &UOp) -> (bool, usize) {
    let mut pending = vec![(lhs, rhs)];
    let mut visited = HashSet::new();
    while let Some((lhs, rhs)) = pending.pop() {
        if lhs.shares_node_with(rhs) {
            continue;
        }
        let pair = (lhs.node_identity(), rhs.node_identity());
        if !visited.insert(pair) {
            continue;
        }
        if lhs.operation() != rhs.operation()
            || lhs.ty() != rhs.ty()
            || lhs.tag() != rhs.tag()
            || lhs.sources().len() != rhs.sources().len()
        {
            return (false, visited.len());
        }
        pending.extend(lhs.sources().iter().zip(rhs.sources()));
    }
    (true, visited.len())
}

fn uop_structure_eq(lhs: &UOp, rhs: &UOp) -> bool {
    uop_structure_eq_with_visits(lhs, rhs).0
}

impl<C> ProjectedExpr<C> {
    pub(crate) fn binary(operation: Binary, lhs: Self, rhs: Self) -> Result<Self, UOpError> {
        if !matches!(
            operation,
            Binary::Add | Binary::Sub | Binary::Mul | Binary::FloorDiv | Binary::Mod
        ) {
            return Err(UOpError::InvalidIndex);
        }
        Ok(Self::Binary {
            operation,
            lhs: Arc::new(lhs),
            rhs: Arc::new(rhs),
        })
    }

    pub(crate) fn constants(&self) -> Vec<&C> {
        let mut constants = Vec::new();
        self.collect_constants(&mut constants);
        constants
    }

    pub(crate) fn contains_linear(&self) -> bool {
        match self {
            Self::Linear => true,
            Self::Constant(_) => false,
            Self::Binary { lhs, rhs, .. } => lhs.contains_linear() || rhs.contains_linear(),
        }
    }

    pub(crate) fn validate_size(&self, depth: usize, nodes: &mut usize) -> Result<(), UOpError> {
        if depth > MAX_PROJECTED_INDEX_DEPTH {
            return Err(UOpError::InvalidIndex);
        }
        *nodes = nodes.checked_add(1).ok_or(UOpError::InvalidIndex)?;
        if *nodes > MAX_PROJECTED_INDEX_NODES {
            return Err(UOpError::InvalidIndex);
        }
        if let Self::Binary { lhs, rhs, .. } = self {
            lhs.validate_size(depth + 1, nodes)?;
            rhs.validate_size(depth + 1, nodes)?;
        }
        Ok(())
    }

    fn collect_constants<'b>(&'b self, out: &mut Vec<&'b C>) {
        match self {
            Self::Linear => {}
            Self::Constant(value) => out.push(value),
            Self::Binary { lhs, rhs, .. } => {
                lhs.collect_constants(out);
                rhs.collect_constants(out);
            }
        }
    }

    pub(crate) fn try_map<D, E>(
        &self,
        map: &mut impl FnMut(&C) -> Result<D, E>,
    ) -> Result<ProjectedExpr<D>, E> {
        Ok(match self {
            Self::Linear => ProjectedExpr::Linear,
            Self::Constant(value) => ProjectedExpr::Constant(map(value)?),
            Self::Binary {
                operation,
                lhs,
                rhs,
            } => ProjectedExpr::Binary {
                operation: *operation,
                lhs: Arc::new(lhs.try_map(map)?),
                rhs: Arc::new(rhs.try_map(map)?),
            },
        })
    }

    pub(crate) fn emit<E: ProjectedIndexEmitter<C>>(
        &self,
        emitter: &mut E,
    ) -> Result<E::Value, E::Error> {
        match self {
            Self::Linear => emitter.linear(),
            Self::Constant(value) => emitter.constant(value),
            Self::Binary {
                operation,
                lhs,
                rhs,
            } => {
                let lhs = lhs.emit(emitter)?;
                let rhs = rhs.emit(emitter)?;
                emitter.binary(*operation, lhs, rhs)
            }
        }
    }
}

impl ProjectedExpr<i64> {
    /// Canonicalizes one authenticated projected address with its exact
    /// iteration extent. This is deliberately narrower than general UOp
    /// algebra: projected indices admit only total core integer operations,
    /// positive divisors, and a proved in-bounds result.
    pub(crate) fn canonicalized_for_output(&self, output_elements: usize) -> Self {
        if output_elements == 0 {
            self.clone()
        } else {
            self.canonicalized_inner(Some(output_elements))
        }
    }

    fn canonicalized_inner(&self, output_elements: Option<usize>) -> Self {
        match self {
            Self::Linear | Self::Constant(_) => self.clone(),
            Self::Binary {
                operation,
                lhs,
                rhs,
            } => {
                let lhs = lhs.canonicalized_inner(output_elements);
                let rhs = rhs.canonicalized_inner(output_elements);
                Self::canonicalized_binary(*operation, lhs, rhs, output_elements)
            }
        }
    }

    fn canonicalized_binary(
        operation: Binary,
        mut lhs: Self,
        mut rhs: Self,
        output_elements: Option<usize>,
    ) -> Self {
        if let (Self::Constant(lhs), Self::Constant(rhs)) = (&lhs, &rhs) {
            let folded = match operation {
                Binary::Add => lhs.checked_add(*rhs),
                Binary::Sub => lhs.checked_sub(*rhs),
                Binary::Mul => lhs.checked_mul(*rhs),
                Binary::FloorDiv if *rhs > 0 => lhs.checked_div_euclid(*rhs),
                Binary::Mod if *rhs > 0 => lhs.checked_rem_euclid(*rhs),
                _ => None,
            };
            if let Some(value) = folded {
                return Self::Constant(value);
            }
        }
        if let Some(extent) = output_elements.and_then(|extent| i64::try_from(extent).ok())
            && extent > 0
            && matches!(operation, Binary::FloorDiv | Binary::Mod)
            && matches!(&lhs, Self::Linear)
            && matches!(&rhs, Self::Constant(value) if *value == extent)
        {
            return if matches!(operation, Binary::FloorDiv) {
                Self::Constant(0)
            } else {
                Self::Linear
            };
        }
        match (&operation, &lhs, &rhs) {
            (Binary::Add | Binary::Sub, _, Self::Constant(0)) => return lhs,
            (Binary::Add, Self::Constant(0), _) => return rhs,
            (Binary::Mul, _, Self::Constant(1)) | (Binary::FloorDiv, _, Self::Constant(1)) => {
                return lhs;
            }
            (Binary::Mul, Self::Constant(1), _) => return rhs,
            (Binary::Mul, _, Self::Constant(0))
            | (Binary::Mul, Self::Constant(0), _)
            | (Binary::Mod, _, Self::Constant(1)) => return Self::Constant(0),
            _ => {}
        }

        // In the authenticated linear domain `0 <= i < extent`, the
        // remainder around a quotient is redundant when the complete
        // quotient range already fits below the modulus. This is the
        // unit-axis form emitted by a symbolic permutation; concrete
        // lowering omits that axis before rebuilding the same address.
        if matches!(operation, Binary::Mod)
            && let Self::Binary {
                operation: Binary::FloorDiv,
                lhs: numerator,
                rhs: divisor,
            } = &lhs
            && matches!(numerator.as_ref(), Self::Linear)
            && let (Self::Constant(divisor), Self::Constant(modulus)) = (divisor.as_ref(), &rhs)
            && *divisor > 0
            && *modulus > 0
            && let Some(limit) = divisor.checked_mul(*modulus)
            && output_elements
                .and_then(|extent| i64::try_from(extent).ok())
                .is_some_and(|extent| extent <= limit)
        {
            return lhs;
        }

        // tinygrad's active symbolic rules keep repeated division and
        // remainder chains compact. Positive constant divisors are an
        // authenticated property of this closed address dialect.
        if matches!(operation, Binary::Mod)
            && let Self::Binary {
                operation: Binary::Mod,
                lhs: _,
                rhs: inner_divisor,
            } = &lhs
            && let (Self::Constant(inner), Self::Constant(outer)) = (inner_divisor.as_ref(), &rhs)
            && *inner > 0
            && *outer > 0
            && inner % outer == 0
        {
            let (inner, outer) = (*inner, *outer);
            return if inner == outer {
                lhs
            } else {
                Self::Binary {
                    operation: Binary::Mod,
                    lhs: match lhs {
                        Self::Binary { lhs, .. } => lhs,
                        _ => unreachable!("matched nested modulo"),
                    },
                    rhs: Arc::new(Self::Constant(outer)),
                }
            };
        }
        if matches!(operation, Binary::FloorDiv)
            && let Self::Binary {
                operation: Binary::Mod,
                lhs: numerator,
                rhs: inner_divisor,
            } = &lhs
            && let (Self::Constant(inner), Self::Constant(outer)) = (inner_divisor.as_ref(), &rhs)
            && *inner > 0
            && *outer > 0
            && inner % outer == 0
        {
            let (inner, outer) = (*inner, *outer);
            let modulus = inner / outer;
            let divided = Self::canonicalized_binary(
                Binary::FloorDiv,
                numerator.as_ref().clone(),
                Self::Constant(outer),
                output_elements,
            );
            return if modulus == 1 {
                divided
            } else {
                Self::canonicalized_binary(
                    Binary::Mod,
                    divided,
                    Self::Constant(modulus),
                    output_elements,
                )
            };
        }
        if matches!(operation, Binary::FloorDiv)
            && let Self::Binary {
                operation: Binary::FloorDiv,
                lhs: numerator,
                rhs: inner_divisor,
            } = &lhs
            && let (Self::Constant(inner), Self::Constant(outer)) = (inner_divisor.as_ref(), &rhs)
            && *inner > 0
            && *outer > 0
            && let Some(divisor) = inner.checked_mul(*outer)
        {
            return if divisor == 1 {
                numerator.as_ref().clone()
            } else {
                Self::Binary {
                    operation: Binary::FloorDiv,
                    lhs: numerator.clone(),
                    rhs: Arc::new(Self::Constant(divisor)),
                }
            };
        }

        // Split an additive numerator into exact multiples of one positive
        // divisor plus a bounded residual. Euclidean division then permits
        // `(q*d + r)//d == q` and `(q*d + r)%d == r` exactly when the
        // authenticated iteration domain proves `0 <= r < d`. This is the
        // mixed-radix ladder produced by reshape/permute/Pad convolution
        // windows; no general distributive rewrite is admitted.
        if matches!(operation, Binary::FloorDiv | Binary::Mod)
            && let Self::Constant(divisor) = &rhs
            && *divisor > 0
            && let Some(value) =
                Self::canonicalized_divmod_sum(operation, lhs.clone(), *divisor, output_elements)
        {
            return value;
        }

        if matches!(operation, Binary::Add)
            && let Some(value) = Self::canonicalized_sum(lhs.clone(), rhs.clone(), output_elements)
        {
            return value;
        }

        // Constants have one deterministic position, which exposes
        // adjacent associative literals to one checked fold without
        // reordering nonconstant address terms.
        if matches!(operation, Binary::Add | Binary::Mul)
            && matches!(&lhs, Self::Constant(_))
            && !matches!(&rhs, Self::Constant(_))
        {
            std::mem::swap(&mut lhs, &mut rhs);
        }
        if matches!(operation, Binary::Add | Binary::Mul)
            && let Self::Constant(outer) = &rhs
            && let Self::Binary {
                operation: inner_operation,
                lhs: inner_lhs,
                rhs: inner_rhs,
            } = &lhs
            && *inner_operation == operation
            && let Self::Constant(inner) = inner_rhs.as_ref()
        {
            let combined = match operation {
                Binary::Add => inner.checked_add(*outer),
                Binary::Mul => inner.checked_mul(*outer),
                _ => None,
            };
            if let Some(combined) = combined {
                return match (operation, combined) {
                    (Binary::Add, 0) | (Binary::Mul, 1) => inner_lhs.as_ref().clone(),
                    (Binary::Mul, 0) => Self::Constant(0),
                    _ => Self::Binary {
                        operation,
                        lhs: inner_lhs.clone(),
                        rhs: Arc::new(Self::Constant(combined)),
                    },
                };
            }
        }

        Self::Binary {
            operation,
            lhs: Arc::new(lhs),
            rhs: Arc::new(rhs),
        }
    }

    fn canonicalized_sum(lhs: Self, rhs: Self, output_elements: Option<usize>) -> Option<Self> {
        const MAX_RECOMPOSITION_TERMS: usize = MAX_PROJECTED_INDEX_DEPTH;

        fn append_terms(
            expression: ProjectedExpr<i64>,
            terms: &mut Vec<ProjectedExpr<i64>>,
        ) -> bool {
            let mut pending = vec![expression];
            while let Some(expression) = pending.pop() {
                if terms
                    .len()
                    .checked_add(pending.len())
                    .is_none_or(|nodes| nodes >= MAX_RECOMPOSITION_TERMS)
                {
                    return false;
                }
                match expression {
                    ProjectedExpr::Binary {
                        operation: Binary::Add,
                        lhs,
                        rhs,
                    } => {
                        pending.push(rhs.as_ref().clone());
                        pending.push(lhs.as_ref().clone());
                    }
                    expression => terms.push(expression),
                }
            }
            true
        }

        let mut terms = Vec::new();
        if !append_terms(lhs, &mut terms) || !append_terms(rhs, &mut terms) {
            return None;
        }
        let mut changed = false;
        'recombine: loop {
            for remainder in 0..terms.len() {
                for product in 0..terms.len() {
                    if remainder == product {
                        continue;
                    }
                    let Some(recombined) =
                        Self::recombined_divmod(&terms[product], &terms[remainder])
                    else {
                        continue;
                    };
                    let recombined = recombined.canonicalized_inner(output_elements);
                    let retained = remainder.min(product);
                    let removed = remainder.max(product);
                    terms[retained] = recombined;
                    terms.remove(removed);
                    changed = true;
                    continue 'recombine;
                }
            }
            break;
        }
        if !changed {
            return None;
        }
        let mut terms = terms.into_iter();
        let mut sum = terms.next().unwrap_or(Self::Constant(0));
        for term in terms {
            sum = Self::Binary {
                operation: Binary::Add,
                lhs: Arc::new(sum),
                rhs: Arc::new(term),
            };
        }
        Some(sum)
    }

    fn canonicalized_divmod_sum(
        operation: Binary,
        numerator: Self,
        divisor: i64,
        output_elements: Option<usize>,
    ) -> Option<Self> {
        const MAX_DIVMOD_TERMS: usize = MAX_PROJECTED_INDEX_DEPTH;

        fn append_terms(
            expression: ProjectedExpr<i64>,
            terms: &mut Vec<ProjectedExpr<i64>>,
        ) -> bool {
            let mut pending = vec![expression];
            while let Some(expression) = pending.pop() {
                if terms
                    .len()
                    .checked_add(pending.len())
                    .is_none_or(|nodes| nodes >= MAX_DIVMOD_TERMS)
                {
                    return false;
                }
                match expression {
                    ProjectedExpr::Binary {
                        operation: Binary::Add,
                        lhs,
                        rhs,
                    } => {
                        pending.push(rhs.as_ref().clone());
                        pending.push(lhs.as_ref().clone());
                    }
                    expression => terms.push(expression),
                }
            }
            true
        }

        fn exact_quotient(
            expression: &ProjectedExpr<i64>,
            divisor: i64,
            output_elements: Option<usize>,
        ) -> Option<ProjectedExpr<i64>> {
            match expression {
                ProjectedExpr::Constant(value) if value % divisor == 0 => {
                    Some(ProjectedExpr::Constant(value / divisor))
                }
                ProjectedExpr::Binary {
                    operation: Binary::Mul,
                    lhs,
                    rhs,
                } => {
                    let (value, factor) = match (lhs.as_ref(), rhs.as_ref()) {
                        (value, ProjectedExpr::Constant(factor)) => (value, *factor),
                        (ProjectedExpr::Constant(factor), value) => (value, *factor),
                        _ => return None,
                    };
                    (factor % divisor == 0).then(|| {
                        ProjectedExpr::canonicalized_binary(
                            Binary::Mul,
                            (*value).clone(),
                            ProjectedExpr::Constant(factor / divisor),
                            output_elements,
                        )
                    })
                }
                _ => None,
            }
        }

        fn sum(
            mut terms: impl Iterator<Item = ProjectedExpr<i64>>,
            output_elements: Option<usize>,
        ) -> ProjectedExpr<i64> {
            let mut value = terms.next().unwrap_or(ProjectedExpr::Constant(0));
            for term in terms {
                value =
                    ProjectedExpr::canonicalized_binary(Binary::Add, value, term, output_elements);
            }
            value
        }

        let mut terms = Vec::new();
        if !append_terms(numerator, &mut terms) || terms.len() < 2 {
            return None;
        }
        let mut quotients = Vec::new();
        let mut residuals = Vec::new();
        for term in terms {
            if let Some(quotient) = exact_quotient(&term, divisor, output_elements) {
                quotients.push(quotient);
            } else {
                residuals.push(term);
            }
        }
        if quotients.is_empty() {
            return None;
        }
        let residual = sum(residuals.into_iter(), output_elements);
        let output_elements = output_elements?;
        let mut state = ValidationState {
            output_elements,
            nodes: 0,
            fits_i32: true,
        };
        let (minimum, maximum) = validate_expression(&residual, 0, &mut state).ok()??;
        if minimum < 0 || maximum >= i128::from(divisor) {
            return None;
        }
        match operation {
            Binary::Mod => Some(residual),
            Binary::FloorDiv => Some(sum(quotients.into_iter(), Some(output_elements))),
            _ => None,
        }
    }

    fn recombined_divmod(product: &Self, remainder: &Self) -> Option<Self> {
        fn scaled_mod(expression: &ProjectedExpr<i64>) -> Option<(&ProjectedExpr<i64>, i64, i64)> {
            let (modulo, scale) = match expression {
                ProjectedExpr::Binary {
                    operation: Binary::Mul,
                    lhs,
                    rhs,
                } => match (lhs.as_ref(), rhs.as_ref()) {
                    (modulo, ProjectedExpr::Constant(scale)) if *scale > 0 => (modulo, *scale),
                    (ProjectedExpr::Constant(scale), modulo) if *scale > 0 => (modulo, *scale),
                    _ => return None,
                },
                modulo => (modulo, 1),
            };
            let ProjectedExpr::Binary {
                operation: Binary::Mod,
                lhs: base,
                rhs,
            } = modulo
            else {
                return None;
            };
            let ProjectedExpr::Constant(divisor) = rhs.as_ref() else {
                return None;
            };
            (*divisor > 0).then_some((base, *divisor, scale))
        }

        fn scaled_quotient(expression: &ProjectedExpr<i64>) -> Option<(&ProjectedExpr<i64>, i64)> {
            let ProjectedExpr::Binary {
                operation: Binary::Mul,
                lhs,
                rhs,
            } = expression
            else {
                return None;
            };
            match (lhs.as_ref(), rhs.as_ref()) {
                (quotient, ProjectedExpr::Constant(scale)) if *scale > 0 => {
                    Some((quotient, *scale))
                }
                (ProjectedExpr::Constant(scale), quotient) if *scale > 0 => {
                    Some((quotient, *scale))
                }
                _ => None,
            }
        }

        fn scaled(expression: ProjectedExpr<i64>, scale: i64) -> ProjectedExpr<i64> {
            if scale == 1 {
                expression
            } else {
                ProjectedExpr::Binary {
                    operation: Binary::Mul,
                    lhs: Arc::new(expression),
                    rhs: Arc::new(ProjectedExpr::Constant(scale)),
                }
            }
        }

        let (remainder_base, divisor, remainder_scale) = scaled_mod(remainder)?;
        let (quotient, quotient_scale) = scaled_quotient(product)?;
        if divisor.checked_mul(remainder_scale)? != quotient_scale {
            return None;
        }
        let (quotient, partial_modulus) = match quotient {
            ProjectedExpr::Binary {
                operation: Binary::Mod,
                lhs,
                rhs,
            } => {
                let ProjectedExpr::Constant(modulus) = rhs.as_ref() else {
                    return None;
                };
                if *modulus <= 0 {
                    return None;
                }
                (lhs.as_ref(), Some(*modulus))
            }
            quotient => (quotient, None),
        };
        let ProjectedExpr::Binary {
            operation: Binary::FloorDiv,
            lhs: quotient_base,
            rhs: quotient_divisor,
        } = quotient
        else {
            return None;
        };
        if !matches!(quotient_divisor.as_ref(), ProjectedExpr::Constant(value) if *value == divisor)
            || !projected_expr_eq(quotient_base, remainder_base)
        {
            return None;
        }
        if let Some(modulus) = partial_modulus {
            let widened = divisor.checked_mul(modulus)?;
            Some(scaled(
                ProjectedExpr::Binary {
                    operation: Binary::Mod,
                    lhs: quotient_base.clone(),
                    rhs: Arc::new(ProjectedExpr::Constant(widened)),
                },
                remainder_scale,
            ))
        } else {
            Some(scaled(quotient_base.as_ref().clone(), remainder_scale))
        }
    }

    pub(crate) fn to_uop(&self, output_elements: usize) -> Result<UOp, UOpError> {
        let output_elements = i64::try_from(output_elements).map_err(|_| UOpError::InvalidIndex)?;
        let ty = crate::UType::scalar(crate::DType::I64);
        let range = UOp::from_operation(
            Operation::Range(0),
            Some(ty),
            vec![UOp::constant(output_elements, ty)],
        );
        if output_elements == 0 {
            return Ok(UOp::from_operation(
                Operation::Binary(Binary::Mul),
                Some(ty),
                vec![range, UOp::constant(0, ty)],
            ));
        }
        Ok(self.to_uop_with_context(
            &range,
            &mut HashMap::new(),
            &mut HashMap::new(),
            &mut HashMap::new(),
        ))
    }

    fn to_uop_with_context(
        &self,
        range: &UOp,
        pointers: &mut HashMap<usize, UOp>,
        constants: &mut HashMap<i64, UOp>,
        binaries: &mut HashMap<(Binary, usize, usize), UOp>,
    ) -> UOp {
        fn build(
            expression: &ProjectedExpr<i64>,
            range: &UOp,
            pointers: &mut HashMap<usize, UOp>,
            constants: &mut HashMap<i64, UOp>,
            binaries: &mut HashMap<(Binary, usize, usize), UOp>,
        ) -> UOp {
            let ty = crate::UType::scalar(crate::DType::I64);
            match expression {
                ProjectedExpr::Linear => range.clone(),
                ProjectedExpr::Constant(value) => constants
                    .entry(*value)
                    .or_insert_with(|| UOp::constant(*value, ty))
                    .clone(),
                ProjectedExpr::Binary {
                    operation,
                    lhs,
                    rhs,
                } => {
                    let lhs = build_shared(lhs, range, pointers, constants, binaries);
                    let rhs = build_shared(rhs, range, pointers, constants, binaries);
                    let key = (*operation, lhs.node_identity(), rhs.node_identity());
                    binaries
                        .entry(key)
                        .or_insert_with(|| {
                            UOp::from_operation(
                                Operation::Binary(*operation),
                                Some(ty),
                                vec![lhs, rhs],
                            )
                        })
                        .clone()
                }
            }
        }
        fn build_shared(
            expression: &Arc<ProjectedExpr<i64>>,
            range: &UOp,
            pointers: &mut HashMap<usize, UOp>,
            constants: &mut HashMap<i64, UOp>,
            binaries: &mut HashMap<(Binary, usize, usize), UOp>,
        ) -> UOp {
            let identity = Arc::as_ptr(expression) as usize;
            if let Some(value) = pointers.get(&identity) {
                return value.clone();
            }
            let value = build(expression, range, pointers, constants, binaries);
            pointers.insert(identity, value.clone());
            value
        }
        build(self, range, pointers, constants, binaries)
    }
}

pub(crate) trait ProjectedIndexEmitter<C = i64> {
    type Value;
    type Error;

    fn linear(&mut self) -> Result<Self::Value, Self::Error>;
    fn constant(&mut self, value: &C) -> Result<Self::Value, Self::Error>;
    fn binary(
        &mut self,
        operation: Binary,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Value, Self::Error>;
}

pub(crate) trait ProjectedPredicateEmitter<C = i64>: ProjectedIndexEmitter<C> {
    type Predicate;

    fn boolean(&mut self, value: bool) -> Result<Self::Predicate, Self::Error>;
    fn compare(
        &mut self,
        operation: crate::CompareOp,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Predicate, Self::Error>;
    fn logical(
        &mut self,
        operation: crate::LogicalOp,
        lhs: Self::Predicate,
        rhs: Option<Self::Predicate>,
    ) -> Result<Self::Predicate, Self::Error>;
}

struct ProjectedEvaluator {
    linear: i128,
}

impl ProjectedIndexEmitter for ProjectedEvaluator {
    type Value = i128;
    type Error = UOpError;

    fn linear(&mut self) -> Result<Self::Value, Self::Error> {
        Ok(self.linear)
    }

    fn constant(&mut self, value: &i64) -> Result<Self::Value, Self::Error> {
        Ok(i128::from(*value))
    }

    fn binary(
        &mut self,
        operation: Binary,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        match operation {
            Binary::Add => lhs.checked_add(rhs),
            Binary::Sub => lhs.checked_sub(rhs),
            Binary::Mul => lhs.checked_mul(rhs),
            Binary::FloorDiv if rhs > 0 => Some(lhs.div_euclid(rhs)),
            Binary::Mod if rhs > 0 => Some(lhs.rem_euclid(rhs)),
            _ => None,
        }
        .ok_or(UOpError::InvalidIndex)
    }
}

impl ProjectedPredicateEmitter for ProjectedEvaluator {
    type Predicate = bool;

    fn boolean(&mut self, value: bool) -> Result<Self::Predicate, Self::Error> {
        Ok(value)
    }

    fn compare(
        &mut self,
        operation: crate::CompareOp,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Predicate, Self::Error> {
        Ok(match operation {
            crate::CompareOp::Eq => lhs == rhs,
            crate::CompareOp::Ne => lhs != rhs,
            crate::CompareOp::Lt => lhs < rhs,
            crate::CompareOp::Le => lhs <= rhs,
            crate::CompareOp::Gt => lhs > rhs,
            crate::CompareOp::Ge => lhs >= rhs,
        })
    }

    fn logical(
        &mut self,
        operation: crate::LogicalOp,
        lhs: Self::Predicate,
        rhs: Option<Self::Predicate>,
    ) -> Result<Self::Predicate, Self::Error> {
        match (operation, rhs) {
            (crate::LogicalOp::Not, None) => Ok(!lhs),
            (crate::LogicalOp::And, Some(rhs)) => Ok(lhs && rhs),
            (crate::LogicalOp::Or, Some(rhs)) => Ok(lhs || rhs),
            _ => Err(UOpError::InvalidIndex),
        }
    }
}

impl ProjectedIndexPlan {
    pub(crate) fn is_projected(index: &UOp) -> bool {
        matches!(
            index.operation(),
            Operation::Index(IndexValue::Buffer {
                addressing: IndexAddressing::Projected | IndexAddressing::Predicated,
                ..
            })
        )
    }

    pub(crate) fn is_predicated(index: &UOp) -> bool {
        matches!(
            index.operation(),
            Operation::Index(IndexValue::Buffer {
                addressing: IndexAddressing::Predicated,
                ..
            })
        )
    }

    pub(crate) fn from_index(index: &UOp) -> Result<Self, UOpError> {
        let Operation::Index(IndexValue::Buffer {
            buffer,
            elements,
            input_shape,
            output_shape,
            addressing,
        }) = index.operation()
        else {
            return Err(UOpError::InvalidIndex);
        };
        let (address, expression, predicate) = match (addressing, index.sources()) {
            (IndexAddressing::Projected, [address, expression]) => (address, expression, None),
            (IndexAddressing::Predicated, [address, expression, predicate]) => {
                (address, expression, Some(predicate))
            }
            _ => return Err(UOpError::InvalidIndex),
        };
        let Some(index_type) = index.ty() else {
            return Err(UOpError::InvalidIndex);
        };
        let Operation::DefineGlobal(AddressValue {
            space: AddressSpace::Global,
            name,
            element,
        }) = address.operation()
        else {
            return Err(UOpError::InvalidIndex);
        };
        if !Self::is_projected(index)
            || input_shape.numel().ok() != Some(*elements)
            || expression.ty() != Some(crate::UType::scalar(crate::DType::I64))
            || *element != index_type
            || name != &format!("b{buffer}")
        {
            return Err(UOpError::InvalidIndex);
        }
        let output_elements = output_shape.numel().map_err(|_| UOpError::InvalidIndex)?;
        elements
            .checked_mul(index_type.scalar.itemsize())
            .and_then(|_| output_elements.checked_mul(index_type.scalar.itemsize()))
            .ok_or(UOpError::InvalidIndex)?;
        let mut parsed_nodes = 0;
        let expression = parse_expression(expression, output_elements, 0, &mut parsed_nodes)?;
        let predicate = predicate
            .map(|predicate| parse_predicate(predicate, output_elements, 0, &mut parsed_nodes))
            .transpose()?;
        let mut state = ValidationState {
            output_elements,
            nodes: 0,
            fits_i32: true,
        };
        let bounds = validate_expression(&expression, 0, &mut state)?;
        if let Some(predicate) = &predicate {
            validate_predicate(predicate, 0, &mut state)?;
        }
        if state.nodes > MAX_PROJECTED_INDEX_NODES {
            return Err(UOpError::InvalidIndex);
        }
        match (output_elements, bounds) {
            (0, None) => {}
            (0, Some(_)) | (_, None) => return Err(UOpError::InvalidIndex),
            (_, Some((minimum, maximum))) => {
                let elements = i128::try_from(*elements).map_err(|_| UOpError::InvalidIndex)?;
                let addressless = elements == 0
                    && minimum == 0
                    && maximum == 0
                    && predicate
                        .as_ref()
                        .is_some_and(ProjectedPredicate::is_always_false);
                if !addressless && (minimum < 0 || maximum >= elements) {
                    return Err(UOpError::InvalidIndex);
                }
            }
        }
        Ok(Self {
            buffer: *buffer,
            elements: *elements,
            output_elements,
            expression,
            predicate,
            fits_i32: state.fits_i32,
        })
    }

    /// Recognizes the only projected path whose storage payload can bypass
    /// scalar decoding: a predicated Load stored as the value itself. The
    /// returned plan is still fully authenticated by `from_index`; renderers
    /// may preserve raw narrow lanes while spelling their own guarded load.
    pub(crate) fn from_direct_predicated_load(value: &UOp) -> Result<Option<Self>, UOpError> {
        let Operation::Load = value.operation() else {
            return Ok(None);
        };
        let [index] = value.sources() else {
            return Err(UOpError::InvalidIndex);
        };
        if !Self::is_predicated(index) {
            return Ok(None);
        }
        Self::from_index(index).map(Some)
    }

    pub(crate) fn fits_i32(&self) -> bool {
        self.fits_i32
    }

    /// Returns the canonical address under this plan's authenticated output
    /// extent. Schema authentication and schedule normalization must use the
    /// same extent-aware form so specialization cannot disagree with the
    /// canonical template while preserving identical lane addresses.
    pub(crate) fn canonical_expression(&self) -> ProjectedExpr<i64> {
        self.expression
            .canonicalized_for_output(self.output_elements)
    }

    pub(crate) fn emit<E: ProjectedIndexEmitter>(
        &self,
        emitter: &mut E,
    ) -> Result<E::Value, E::Error> {
        self.expression.emit(emitter)
    }

    pub(crate) fn emit_predicate<E: ProjectedPredicateEmitter>(
        &self,
        emitter: &mut E,
    ) -> Result<Option<E::Predicate>, E::Error> {
        self.predicate
            .as_ref()
            .map(|predicate| predicate.emit(emitter))
            .transpose()
    }

    pub(crate) fn is_guarded(&self) -> bool {
        self.predicate.is_some()
    }

    pub(crate) fn valid(&self, linear: usize) -> Result<bool, UOpError> {
        if linear >= self.output_elements {
            return Err(UOpError::InvalidIndex);
        }
        let mut evaluator = ProjectedEvaluator {
            linear: i128::try_from(linear).map_err(|_| UOpError::InvalidIndex)?,
        };
        self.predicate
            .as_ref()
            .map(|predicate| predicate.emit(&mut evaluator))
            .transpose()
            .map(|predicate| predicate.unwrap_or(true))
    }

    pub(crate) fn offset(&self, linear: usize) -> Result<usize, UOpError> {
        if linear >= self.output_elements {
            return Err(UOpError::InvalidIndex);
        }
        let mut evaluator = ProjectedEvaluator {
            linear: i128::try_from(linear).map_err(|_| UOpError::InvalidIndex)?,
        };
        let offset = self.emit(&mut evaluator)?;
        let offset = usize::try_from(offset).map_err(|_| UOpError::InvalidIndex)?;
        (offset < self.elements)
            .then_some(offset)
            .ok_or(UOpError::InvalidIndex)
    }
}

struct GraphProjectedCanonicalizer {
    output_elements: usize,
    nodes: usize,
    active: HashSet<usize>,
    expressions: HashMap<usize, (ProjectedExpr<i64>, usize)>,
    predicates: HashMap<usize, (CanonicalGraphPredicate, usize)>,
}

#[derive(Clone)]
enum CanonicalGraphPredicate {
    Constant(bool),
    Term(UOp),
    And(Vec<UOp>),
}

impl CanonicalGraphPredicate {
    fn into_uop(self) -> UOp {
        fn boolean(value: bool) -> UOp {
            UOp::scalar_constant(
                crate::DType::Bool,
                u64::from(value),
                crate::UType::scalar(crate::DType::Bool),
            )
        }
        fn balanced_and(mut terms: Vec<UOp>) -> UOp {
            while terms.len() > 1 {
                terms = terms
                    .chunks(2)
                    .map(|pair| match pair {
                        [lhs, rhs] => UOp::from_operation(
                            Operation::GraphLogical(crate::LogicalOp::And),
                            Some(crate::UType::scalar(crate::DType::Bool)),
                            vec![lhs.clone(), rhs.clone()],
                        ),
                        [value] => value.clone(),
                        _ => unreachable!("chunks of two are nonempty"),
                    })
                    .collect();
            }
            terms.pop().unwrap_or_else(|| boolean(true))
        }
        match self {
            Self::Constant(value) => boolean(value),
            Self::Term(term) => term,
            Self::And(terms) => balanced_and(terms),
        }
    }

    fn and(lhs: Self, rhs: Self) -> Self {
        match (lhs, rhs) {
            (Self::Constant(false), _) | (_, Self::Constant(false)) => Self::Constant(false),
            (Self::Constant(true), rhs) => rhs,
            (lhs, Self::Constant(true)) => lhs,
            (lhs, rhs) => {
                let mut terms = Vec::new();
                let mut extend = |predicate| match predicate {
                    Self::Term(term) => {
                        if !terms
                            .iter()
                            .any(|existing| uop_structure_eq(existing, &term))
                        {
                            terms.push(term);
                        }
                    }
                    Self::And(values) => {
                        for term in values {
                            if !terms
                                .iter()
                                .any(|existing| uop_structure_eq(existing, &term))
                            {
                                terms.push(term);
                            }
                        }
                    }
                    Self::Constant(_) => unreachable!("constants were folded above"),
                };
                extend(lhs);
                extend(rhs);
                Self::And(terms)
            }
        }
    }
}

impl GraphProjectedCanonicalizer {
    fn new(output_elements: usize) -> Self {
        Self {
            output_elements,
            nodes: 0,
            active: HashSet::new(),
            expressions: HashMap::new(),
            predicates: HashMap::new(),
        }
    }

    fn enter(&mut self, node: &UOp, depth: usize) -> Result<(), UOpError> {
        let identity = node.node_identity();
        if depth > MAX_PROJECTED_INDEX_DEPTH {
            return Err(UOpError::InvalidIndex);
        }
        self.nodes = self.nodes.checked_add(1).ok_or(UOpError::InvalidIndex)?;
        if self.nodes > MAX_GRAPH_PROJECTED_DAG_NODES || !self.active.insert(identity) {
            return Err(UOpError::InvalidIndex);
        }
        Ok(())
    }

    fn expression(
        &mut self,
        expression: &UOp,
        depth: usize,
    ) -> Result<(ProjectedExpr<i64>, usize), UOpError> {
        let identity = expression.node_identity();
        if let Some((expression, height)) = self.expressions.get(&identity) {
            depth
                .checked_add(*height)
                .filter(|deepest| *deepest <= MAX_PROJECTED_INDEX_DEPTH)
                .ok_or(UOpError::InvalidIndex)?;
            return Ok((expression.clone(), *height));
        }
        self.enter(expression, depth)?;
        if expression.ty() != Some(crate::UType::scalar(crate::DType::I64)) {
            return Err(UOpError::InvalidIndex);
        }
        let (parsed, height) = match expression.operation() {
            Operation::Range(0) => {
                let [bound] = expression.sources() else {
                    return Err(UOpError::InvalidIndex);
                };
                let Operation::Const(LiteralValue::Int(bound)) = bound.operation() else {
                    return Err(UOpError::InvalidIndex);
                };
                if usize::try_from(*bound).ok() != Some(self.output_elements) {
                    return Err(UOpError::InvalidIndex);
                }
                (ProjectedExpr::Linear, 0)
            }
            Operation::Const(LiteralValue::Int(value)) if expression.sources().is_empty() => {
                (ProjectedExpr::Constant(*value), 0)
            }
            Operation::Binary(operation) if expression.sources().len() == 2 => {
                let (lhs, lhs_height) = self.expression(&expression.sources()[0], depth + 1)?;
                let (rhs, rhs_height) = self.expression(&expression.sources()[1], depth + 1)?;
                let parsed = if self.output_elements == 0 {
                    ProjectedExpr::binary(*operation, lhs, rhs)?
                } else {
                    ProjectedExpr::canonicalized_binary(
                        *operation,
                        lhs,
                        rhs,
                        Some(self.output_elements),
                    )
                };
                (
                    parsed,
                    lhs_height
                        .max(rhs_height)
                        .checked_add(1)
                        .ok_or(UOpError::InvalidIndex)?,
                )
            }
            _ => return Err(UOpError::InvalidIndex),
        };
        self.active.remove(&identity);
        self.expressions.insert(identity, (parsed.clone(), height));
        Ok((parsed, height))
    }

    fn checked_expression_uop(&self, expression: &ProjectedExpr<i64>) -> Result<UOp, UOpError> {
        let mut state = ValidationState {
            output_elements: self.output_elements,
            nodes: 0,
            fits_i32: true,
        };
        validate_expression(expression, 0, &mut state)?;
        expression.to_uop(self.output_elements)
    }

    fn predicate(
        &mut self,
        predicate: &UOp,
        depth: usize,
    ) -> Result<(CanonicalGraphPredicate, usize), UOpError> {
        let identity = predicate.node_identity();
        if let Some((predicate, height)) = self.predicates.get(&identity) {
            depth
                .checked_add(*height)
                .filter(|deepest| *deepest <= MAX_PROJECTED_INDEX_DEPTH)
                .ok_or(UOpError::InvalidIndex)?;
            return Ok((predicate.clone(), *height));
        }
        self.enter(predicate, depth)?;
        if predicate.ty() != Some(crate::UType::scalar(crate::DType::Bool)) {
            return Err(UOpError::InvalidIndex);
        }
        let (parsed, height) = match predicate.operation() {
            Operation::Const(LiteralValue::Scalar {
                dtype: crate::DType::Bool,
                bits,
            }) if predicate.sources().is_empty() && *bits <= 1 => {
                (CanonicalGraphPredicate::Constant(*bits != 0), 0)
            }
            Operation::GraphCompare(operation) if predicate.sources().len() == 2 => {
                let (lhs, lhs_height) = self.expression(&predicate.sources()[0], depth + 1)?;
                let (rhs, rhs_height) = self.expression(&predicate.sources()[1], depth + 1)?;
                let folded = match (&lhs, &rhs) {
                    (ProjectedExpr::Constant(lhs), ProjectedExpr::Constant(rhs)) => {
                        Some(match operation {
                            crate::CompareOp::Eq => lhs == rhs,
                            crate::CompareOp::Ne => lhs != rhs,
                            crate::CompareOp::Lt => lhs < rhs,
                            crate::CompareOp::Le => lhs <= rhs,
                            crate::CompareOp::Gt => lhs > rhs,
                            crate::CompareOp::Ge => lhs >= rhs,
                        })
                    }
                    _ if projected_expr_eq(&lhs, &rhs) => Some(matches!(
                        operation,
                        crate::CompareOp::Eq | crate::CompareOp::Le | crate::CompareOp::Ge
                    )),
                    _ => None,
                };
                let parsed = if let Some(value) = folded {
                    CanonicalGraphPredicate::Constant(value)
                } else {
                    CanonicalGraphPredicate::Term(UOp::from_operation(
                        Operation::GraphCompare(*operation),
                        Some(crate::UType::scalar(crate::DType::Bool)),
                        vec![
                            self.checked_expression_uop(&lhs)?,
                            self.checked_expression_uop(&rhs)?,
                        ],
                    ))
                };
                (
                    parsed,
                    lhs_height
                        .max(rhs_height)
                        .checked_add(1)
                        .ok_or(UOpError::InvalidIndex)?,
                )
            }
            Operation::GraphLogical(operation) => {
                let ((lhs, lhs_height), rhs) = match (operation, predicate.sources()) {
                    (crate::LogicalOp::Not, [lhs]) => (self.predicate(lhs, depth + 1)?, None),
                    (crate::LogicalOp::And | crate::LogicalOp::Or, [lhs, rhs]) => (
                        self.predicate(lhs, depth + 1)?,
                        Some(self.predicate(rhs, depth + 1)?),
                    ),
                    _ => return Err(UOpError::InvalidIndex),
                };
                let rhs_height = rhs.as_ref().map_or(0, |(_, height)| *height);
                let parsed = match (*operation, lhs, rhs.map(|(predicate, _)| predicate)) {
                    (crate::LogicalOp::And, lhs, Some(rhs)) => {
                        CanonicalGraphPredicate::and(lhs, rhs)
                    }
                    (crate::LogicalOp::Not, CanonicalGraphPredicate::Constant(value), None) => {
                        CanonicalGraphPredicate::Constant(!value)
                    }
                    (crate::LogicalOp::Or, CanonicalGraphPredicate::Constant(true), Some(_))
                    | (crate::LogicalOp::Or, _, Some(CanonicalGraphPredicate::Constant(true))) => {
                        CanonicalGraphPredicate::Constant(true)
                    }
                    (crate::LogicalOp::Or, CanonicalGraphPredicate::Constant(false), Some(rhs)) => {
                        rhs
                    }
                    (crate::LogicalOp::Or, lhs, Some(CanonicalGraphPredicate::Constant(false))) => {
                        lhs
                    }
                    (operation, lhs, rhs) => CanonicalGraphPredicate::Term(UOp::from_operation(
                        Operation::GraphLogical(operation),
                        Some(crate::UType::scalar(crate::DType::Bool)),
                        std::iter::once(lhs.into_uop())
                            .chain(rhs.map(CanonicalGraphPredicate::into_uop))
                            .collect(),
                    )),
                };
                (
                    parsed,
                    lhs_height
                        .max(rhs_height)
                        .checked_add(1)
                        .ok_or(UOpError::InvalidIndex)?,
                )
            }
            _ => return Err(UOpError::InvalidIndex),
        };
        self.active.remove(&identity);
        self.predicates.insert(identity, (parsed.clone(), height));
        Ok((parsed, height))
    }
}

/// Canonicalizes one compiler-derived projected address before another
/// movement step composes it. The trusted construction DAG is bounded by
/// unique nodes first, then the canonical expanded expression must still pass
/// the ordinary hostile artifact occurrence/depth limits below.
pub(crate) fn canonicalize_graph_projected_address(
    expression: &UOp,
    output_elements: usize,
    source_elements: usize,
) -> Result<UOp, UOpError> {
    let (expression, _) =
        GraphProjectedCanonicalizer::new(output_elements).expression(expression, 0)?;
    let mut state = ValidationState {
        output_elements,
        nodes: 0,
        fits_i32: true,
    };
    let bounds = validate_expression(&expression, 0, &mut state)?;
    match (output_elements, bounds) {
        (0, None) => {}
        (0, Some(_)) | (_, None) => return Err(UOpError::InvalidIndex),
        (_, Some((minimum, maximum))) => {
            let source_elements =
                i128::try_from(source_elements).map_err(|_| UOpError::InvalidIndex)?;
            if minimum < 0 || maximum >= source_elements {
                return Err(UOpError::InvalidIndex);
            }
        }
    }
    expression.to_uop(output_elements)
}

/// Canonicalizes coordinates from one authenticated logical iteration as one
/// bounded DAG. Keeping one parser and UOp conversion memo across roots
/// preserves their exact common reshape numerator, allowing the rangeifier to
/// recompose it before later movement stages. Every coordinate independently
/// retains the ordinary expanded occurrence, depth, arithmetic, and bounds
/// checks used by durable projected indices.
pub(crate) fn canonicalize_graph_projected_coordinates(
    coordinates: &[(UOp, usize)],
    output_elements: usize,
) -> Result<Vec<UOp>, UOpError> {
    let output_elements_i64 = i64::try_from(output_elements).map_err(|_| UOpError::InvalidIndex)?;
    let ty = crate::UType::scalar(crate::DType::I64);
    let range = UOp::from_operation(
        Operation::Range(0),
        Some(ty),
        vec![UOp::constant(output_elements_i64, ty)],
    );
    let mut canonicalizer = GraphProjectedCanonicalizer::new(output_elements);
    let mut pointer_memo = HashMap::new();
    let mut constant_memo = HashMap::new();
    let mut binary_memo = HashMap::new();
    coordinates
        .iter()
        .map(|(coordinate, source_elements)| {
            // The complete multi-root construction shares one unique-node
            // budget. A later root therefore cannot assemble several
            // individually valid cached subgraphs into an oversized DAG.
            let (expression, _) = canonicalizer.expression(coordinate, 0)?;
            let mut state = ValidationState {
                output_elements,
                nodes: 0,
                fits_i32: true,
            };
            let bounds = validate_expression(&expression, 0, &mut state)?;
            match (output_elements, bounds) {
                (0, None) => {}
                (0, Some(_)) | (_, None) => return Err(UOpError::InvalidIndex),
                (_, Some((minimum, maximum))) => {
                    let source_elements =
                        i128::try_from(*source_elements).map_err(|_| UOpError::InvalidIndex)?;
                    if minimum < 0 || maximum >= source_elements {
                        return Err(UOpError::InvalidIndex);
                    }
                }
            }
            if output_elements == 0 {
                expression.to_uop(0)
            } else {
                Ok(expression.to_uop_with_context(
                    &range,
                    &mut pointer_memo,
                    &mut constant_memo,
                    &mut binary_memo,
                ))
            }
        })
        .collect()
}

/// Canonicalizes one compiler-derived projected predicate without changing
/// the public/wire dialect. Compared address expressions share the same
/// bounded DAG parser; conjunctions are constant-folded, flattened, and
/// deterministically deduplicated before the ordinary Index parser remains
/// the final admission authority.
pub(crate) fn canonicalize_graph_projected_predicate(
    predicate: &UOp,
    output_elements: usize,
) -> Result<UOp, UOpError> {
    Ok(GraphProjectedCanonicalizer::new(output_elements)
        .predicate(predicate, 0)?
        .0
        .into_uop())
}

struct InfixProjectedEmitter<'a, F, B> {
    linear: String,
    literal: &'a mut F,
    boolean: &'a mut B,
}

impl<F, B> ProjectedIndexEmitter for InfixProjectedEmitter<'_, F, B>
where
    F: FnMut(i64) -> Result<String, UOpError>,
    B: FnMut(bool) -> String,
{
    type Value = String;
    type Error = UOpError;

    fn linear(&mut self) -> Result<Self::Value, Self::Error> {
        Ok(self.linear.clone())
    }

    fn constant(&mut self, value: &i64) -> Result<Self::Value, Self::Error> {
        (self.literal)(*value)
    }

    fn binary(
        &mut self,
        operation: Binary,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        let operator = match operation {
            Binary::Add => "+",
            Binary::Sub => "-",
            Binary::Mul => "*",
            Binary::FloorDiv => "/",
            Binary::Mod => "%",
            _ => return Err(UOpError::InvalidIndex),
        };
        Ok(format!("(({lhs}) {operator} ({rhs}))"))
    }
}

impl<F, B> ProjectedPredicateEmitter for InfixProjectedEmitter<'_, F, B>
where
    F: FnMut(i64) -> Result<String, UOpError>,
    B: FnMut(bool) -> String,
{
    type Predicate = String;

    fn boolean(&mut self, value: bool) -> Result<Self::Predicate, Self::Error> {
        Ok((self.boolean)(value))
    }

    fn compare(
        &mut self,
        operation: crate::CompareOp,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Predicate, Self::Error> {
        let operator = match operation {
            crate::CompareOp::Eq => "==",
            crate::CompareOp::Ne => "!=",
            crate::CompareOp::Lt => "<",
            crate::CompareOp::Le => "<=",
            crate::CompareOp::Gt => ">",
            crate::CompareOp::Ge => ">=",
        };
        Ok(format!("(({lhs}) {operator} ({rhs}))"))
    }

    fn logical(
        &mut self,
        operation: crate::LogicalOp,
        lhs: Self::Predicate,
        rhs: Option<Self::Predicate>,
    ) -> Result<Self::Predicate, Self::Error> {
        match (operation, rhs) {
            (crate::LogicalOp::Not, None) => Ok(format!("(!({lhs}))")),
            (crate::LogicalOp::And, Some(rhs)) => Ok(format!("(({lhs}) && ({rhs}))")),
            (crate::LogicalOp::Or, Some(rhs)) => Ok(format!("(({lhs}) || ({rhs}))")),
            _ => Err(UOpError::InvalidIndex),
        }
    }
}

pub(crate) fn render_infix_projected_predicate(
    plan: &ProjectedIndexPlan,
    linear: impl Into<String>,
    mut literal: impl FnMut(i64) -> Result<String, UOpError>,
) -> Result<Option<String>, UOpError> {
    let mut boolean = |value| String::from(if value { "1" } else { "0" });
    let mut emitter = InfixProjectedEmitter {
        linear: linear.into(),
        literal: &mut literal,
        boolean: &mut boolean,
    };
    plan.emit_predicate(&mut emitter)
}

pub(crate) struct InfixProjectedAccess {
    pub(crate) offset: String,
    pub(crate) predicate: Option<String>,
}

/// Renders the address and optional validity predicate through one checked
/// projection. Backends remain responsible for spelling a genuinely guarded
/// load: an eager value selection is not sufficient for addressless sources.
pub(crate) fn render_infix_projected_access(
    plan: &ProjectedIndexPlan,
    linear: impl Into<String>,
    mut literal: impl FnMut(i64) -> Result<String, UOpError>,
    mut boolean: impl FnMut(bool) -> String,
) -> Result<InfixProjectedAccess, UOpError> {
    let mut emitter = InfixProjectedEmitter {
        linear: linear.into(),
        literal: &mut literal,
        boolean: &mut boolean,
    };
    Ok(InfixProjectedAccess {
        offset: plan.emit(&mut emitter)?,
        predicate: plan.emit_predicate(&mut emitter)?,
    })
}

pub(crate) fn render_infix_projected_index(
    plan: &ProjectedIndexPlan,
    linear: impl Into<String>,
    mut literal: impl FnMut(i64) -> Result<String, UOpError>,
) -> Result<String, UOpError> {
    let mut boolean = |value| String::from(if value { "1" } else { "0" });
    plan.emit(&mut InfixProjectedEmitter {
        linear: linear.into(),
        literal: &mut literal,
        boolean: &mut boolean,
    })
}

fn parse_expression(
    expression: &UOp,
    output_elements: usize,
    depth: usize,
    nodes: &mut usize,
) -> Result<ProjectedExpr<i64>, UOpError> {
    *nodes = nodes.checked_add(1).ok_or(UOpError::InvalidIndex)?;
    if depth > MAX_PROJECTED_INDEX_DEPTH
        || *nodes > MAX_PROJECTED_INDEX_NODES
        || expression.ty() != Some(crate::UType::scalar(crate::DType::I64))
    {
        return Err(UOpError::InvalidIndex);
    }
    match expression.operation() {
        Operation::Range(0) => {
            let [bound] = expression.sources() else {
                return Err(UOpError::InvalidIndex);
            };
            let Operation::Const(LiteralValue::Int(bound)) = bound.operation() else {
                return Err(UOpError::InvalidIndex);
            };
            if usize::try_from(*bound).ok() != Some(output_elements) {
                return Err(UOpError::InvalidIndex);
            }
            Ok(ProjectedExpr::Linear)
        }
        Operation::Const(LiteralValue::Int(value)) if expression.sources().is_empty() => {
            Ok(ProjectedExpr::Constant(*value))
        }
        Operation::Binary(operation) if expression.sources().len() == 2 => ProjectedExpr::binary(
            *operation,
            parse_expression(&expression.sources()[0], output_elements, depth + 1, nodes)?,
            parse_expression(&expression.sources()[1], output_elements, depth + 1, nodes)?,
        ),
        _ => Err(UOpError::InvalidIndex),
    }
}

fn parse_predicate(
    predicate: &UOp,
    output_elements: usize,
    depth: usize,
    nodes: &mut usize,
) -> Result<ProjectedPredicate, UOpError> {
    *nodes = nodes.checked_add(1).ok_or(UOpError::InvalidIndex)?;
    if depth > MAX_PROJECTED_INDEX_DEPTH
        || *nodes > MAX_PROJECTED_INDEX_NODES
        || predicate.ty() != Some(crate::UType::scalar(crate::DType::Bool))
    {
        return Err(UOpError::InvalidIndex);
    }
    match predicate.operation() {
        Operation::Const(LiteralValue::Scalar {
            dtype: crate::DType::Bool,
            bits,
        }) if predicate.sources().is_empty() && *bits <= 1 => {
            Ok(ProjectedPredicate::Constant(*bits != 0))
        }
        Operation::GraphCompare(operation) if predicate.sources().len() == 2 => {
            Ok(ProjectedPredicate::Compare {
                operation: *operation,
                lhs: parse_expression(&predicate.sources()[0], output_elements, depth + 1, nodes)?,
                rhs: parse_expression(&predicate.sources()[1], output_elements, depth + 1, nodes)?,
            })
        }
        Operation::GraphLogical(operation) => {
            let (lhs, rhs) = match (operation, predicate.sources()) {
                (crate::LogicalOp::Not, [lhs]) => (lhs, None),
                (crate::LogicalOp::And | crate::LogicalOp::Or, [lhs, rhs]) => (lhs, Some(rhs)),
                _ => return Err(UOpError::InvalidIndex),
            };
            Ok(ProjectedPredicate::Logical {
                operation: *operation,
                lhs: Box::new(parse_predicate(lhs, output_elements, depth + 1, nodes)?),
                rhs: rhs
                    .map(|rhs| {
                        parse_predicate(rhs, output_elements, depth + 1, nodes).map(Box::new)
                    })
                    .transpose()?,
            })
        }
        _ => Err(UOpError::InvalidIndex),
    }
}

struct ValidationState {
    output_elements: usize,
    nodes: usize,
    fits_i32: bool,
}

fn validate_predicate(
    predicate: &ProjectedPredicate,
    depth: usize,
    state: &mut ValidationState,
) -> Result<(), UOpError> {
    if depth > MAX_PROJECTED_INDEX_DEPTH {
        return Err(UOpError::InvalidIndex);
    }
    state.nodes = state.nodes.checked_add(1).ok_or(UOpError::InvalidIndex)?;
    if state.nodes > MAX_PROJECTED_INDEX_NODES {
        return Err(UOpError::InvalidIndex);
    }
    match predicate {
        ProjectedPredicate::Constant(_) => Ok(()),
        ProjectedPredicate::Compare { lhs, rhs, .. } => {
            validate_expression(lhs, depth + 1, state)?;
            validate_expression(rhs, depth + 1, state)?;
            Ok(())
        }
        ProjectedPredicate::Logical { lhs, rhs, .. } => {
            validate_predicate(lhs, depth + 1, state)?;
            if let Some(rhs) = rhs {
                validate_predicate(rhs, depth + 1, state)?;
            }
            Ok(())
        }
    }
}

fn validate_expression(
    expression: &ProjectedExpr<i64>,
    depth: usize,
    state: &mut ValidationState,
) -> Result<Option<(i128, i128)>, UOpError> {
    if depth > MAX_PROJECTED_INDEX_DEPTH {
        return Err(UOpError::InvalidIndex);
    }
    state.nodes = state.nodes.checked_add(1).ok_or(UOpError::InvalidIndex)?;
    if state.nodes > MAX_PROJECTED_INDEX_NODES {
        return Err(UOpError::InvalidIndex);
    }
    match expression {
        ProjectedExpr::Linear => {
            if state.output_elements > i32::MAX as usize {
                state.fits_i32 = false;
            }
            Ok((state.output_elements != 0).then(|| {
                (
                    0,
                    i128::try_from(state.output_elements - 1)
                        .expect("usize fits i128 on supported hosts"),
                )
            }))
        }
        ProjectedExpr::Constant(value) => {
            state.fits_i32 &= i32::try_from(*value).is_ok();
            let value = i128::from(*value);
            Ok(Some((value, value)))
        }
        ProjectedExpr::Binary {
            operation,
            lhs,
            rhs,
        } => {
            let lhs = validate_expression(lhs, depth + 1, state)?;
            let rhs = validate_expression(rhs, depth + 1, state)?;
            if matches!(operation, Binary::FloorDiv | Binary::Mod)
                && !matches!(rhs, Some((minimum, maximum)) if minimum == maximum && minimum > 0)
            {
                return Err(UOpError::InvalidIndex);
            }
            let (Some((lmin, lmax)), Some((rmin, rmax))) = (lhs, rhs) else {
                return Ok(None);
            };
            let bounds = match operation {
                Binary::Add => (lmin.checked_add(rmin), lmax.checked_add(rmax)),
                Binary::Sub => (lmin.checked_sub(rmax), lmax.checked_sub(rmin)),
                Binary::Mul => {
                    let values = [
                        lmin.checked_mul(rmin),
                        lmin.checked_mul(rmax),
                        lmax.checked_mul(rmin),
                        lmax.checked_mul(rmax),
                    ];
                    if values.iter().any(Option::is_none) {
                        return Err(UOpError::InvalidIndex);
                    }
                    let values = values.map(Option::unwrap);
                    (values.iter().min().copied(), values.iter().max().copied())
                }
                Binary::FloorDiv | Binary::Mod => {
                    if lmin < 0 {
                        return Err(UOpError::InvalidIndex);
                    }
                    if matches!(operation, Binary::FloorDiv) {
                        (Some(lmin.div_euclid(rmin)), Some(lmax.div_euclid(rmin)))
                    } else if lmin >= 0 {
                        (Some(0), Some(lmax.min(rmin - 1)))
                    } else {
                        return Err(UOpError::InvalidIndex);
                    }
                }
                _ => return Err(UOpError::InvalidIndex),
            };
            let minimum = bounds.0.ok_or(UOpError::InvalidIndex)?;
            let maximum = bounds.1.ok_or(UOpError::InvalidIndex)?;
            if minimum < i128::from(i64::MIN) || maximum > i128::from(i64::MAX) {
                return Err(UOpError::InvalidIndex);
            }
            state.fits_i32 &= minimum >= i128::from(i32::MIN) && maximum <= i128::from(i32::MAX);
            Ok(Some((minimum, maximum)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Shape, UType};

    fn integer(value: i64) -> UOp {
        UOp::constant(value, UType::scalar(DType::I64))
    }

    fn binary(operation: Binary, lhs: UOp, rhs: UOp) -> UOp {
        UOp::from_operation(
            Operation::Binary(operation),
            Some(UType::scalar(DType::I64)),
            vec![lhs, rhs],
        )
    }

    fn index(input: impl Into<Shape>, output: impl Into<Shape>, expression: UOp) -> UOp {
        let input = input.into();
        let output = output.into();
        let elements = input.numel().unwrap();
        let ty = UType::scalar(DType::F32);
        let address = UOp::from_operation(
            Operation::DefineGlobal(AddressValue {
                space: AddressSpace::Global,
                name: "b7".into(),
                element: ty,
            }),
            Some(ty),
            vec![],
        );
        UOp::from_operation(
            Operation::Index(IndexValue::Buffer {
                buffer: 7,
                elements,
                input_shape: input,
                output_shape: output,
                addressing: IndexAddressing::Projected,
            }),
            Some(ty),
            vec![address, expression],
        )
    }

    fn predicated_index(
        input: impl Into<Shape>,
        output: impl Into<Shape>,
        expression: UOp,
        predicate: UOp,
    ) -> UOp {
        let input = input.into();
        let output = output.into();
        let elements = input.numel().unwrap();
        let ty = UType::scalar(DType::F32);
        let address = UOp::from_operation(
            Operation::DefineGlobal(AddressValue {
                space: AddressSpace::Global,
                name: "b7".into(),
                element: ty,
            }),
            Some(ty),
            vec![],
        );
        UOp::from_operation(
            Operation::Index(IndexValue::Buffer {
                buffer: 7,
                elements,
                input_shape: input,
                output_shape: output,
                addressing: IndexAddressing::Predicated,
            }),
            Some(ty),
            vec![address, expression, predicate],
        )
    }

    fn range(extent: i64) -> UOp {
        UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![integer(extent)],
        )
    }

    #[test]
    fn validates_scalar_empty_reverse_and_rejects_oob_or_excess_depth() {
        let scalar = index(Shape::from([1, 1]), Shape::new([]), integer(0));
        assert_eq!(
            ProjectedIndexPlan::from_index(&scalar)
                .unwrap()
                .offset(0)
                .unwrap(),
            0
        );

        let empty = index(
            Shape::from([0, 2]),
            Shape::from([0, 1]),
            binary(Binary::Mul, range(0), integer(-1)),
        );
        assert_eq!(
            ProjectedIndexPlan::from_index(&empty)
                .unwrap()
                .output_elements,
            0
        );
        let empty_divide_by_zero = index(
            Shape::from([0, 2]),
            Shape::from([0, 1]),
            binary(Binary::FloorDiv, range(0), integer(0)),
        );
        assert!(ProjectedIndexPlan::from_index(&empty_divide_by_zero).is_err());

        let reverse = index(
            Shape::from([2, 2]),
            Shape::from([4]),
            binary(
                Binary::Sub,
                integer(3),
                binary(Binary::Mul, range(4), integer(1)),
            ),
        );
        let reverse = ProjectedIndexPlan::from_index(&reverse).unwrap();
        assert!(reverse.offset(4).is_err());
        assert_eq!(
            (0..4)
                .map(|i| reverse.offset(i).unwrap())
                .collect::<Vec<_>>(),
            vec![3, 2, 1, 0]
        );

        let oob = index(
            Shape::from([2, 2]),
            Shape::from([4]),
            binary(Binary::Add, range(4), integer(1)),
        );
        assert!(ProjectedIndexPlan::from_index(&oob).is_err());
        let invalid_divisor = index(
            Shape::from([2, 2]),
            Shape::from([4]),
            binary(Binary::FloorDiv, range(4), integer(0)),
        );
        assert!(ProjectedIndexPlan::from_index(&invalid_divisor).is_err());

        let mut too_deep = range(1);
        for _ in 0..=MAX_PROJECTED_INDEX_DEPTH {
            too_deep = binary(Binary::Add, too_deep, integer(0));
        }
        assert!(
            ProjectedIndexPlan::from_index(
                &index(Shape::from([1, 1]), Shape::from([1]), too_deep,)
            )
            .is_err()
        );

        let mut shared_diamond = range(1);
        for _ in 0..13 {
            shared_diamond = binary(Binary::Add, shared_diamond.clone(), shared_diamond);
        }
        assert!(
            ProjectedIndexPlan::from_index(&index(
                Shape::from([1]),
                Shape::from([1]),
                shared_diamond.clone(),
            ))
            .is_err()
        );
        assert!(canonicalize_graph_projected_address(&shared_diamond, 1, 1).is_err());
    }

    #[test]
    fn compiler_dag_canonicalizes_before_durable_occurrence_limits() {
        let mut reconstructed = range(8);
        for _ in 0..13 {
            let quotient = binary(Binary::FloorDiv, reconstructed.clone(), integer(4));
            reconstructed = binary(
                Binary::Add,
                binary(Binary::Mul, quotient, integer(4)),
                binary(Binary::Mod, reconstructed, integer(4)),
            );
        }
        assert!(
            ProjectedIndexPlan::from_index(&index(
                Shape::from([8]),
                Shape::from([8]),
                reconstructed.clone(),
            ))
            .is_err()
        );
        let compact = canonicalize_graph_projected_address(&reconstructed, 8, 8).unwrap();
        let compact =
            ProjectedIndexPlan::from_index(&index(Shape::from([8]), Shape::from([8]), compact))
                .unwrap();
        assert_eq!(
            (0..8)
                .map(|lane| compact.offset(lane).unwrap())
                .collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );

        let comparison = UOp::from_operation(
            Operation::GraphCompare(crate::CompareOp::Lt),
            Some(UType::scalar(DType::Bool)),
            vec![range(8), integer(8)],
        );
        let mut duplicate_conjunction = comparison;
        for _ in 0..13 {
            duplicate_conjunction = UOp::from_operation(
                Operation::GraphLogical(crate::LogicalOp::And),
                Some(UType::scalar(DType::Bool)),
                vec![duplicate_conjunction.clone(), duplicate_conjunction],
            );
        }
        let compact = canonicalize_graph_projected_predicate(&duplicate_conjunction, 8).unwrap();
        let plan =
            ProjectedIndexPlan::from_index(&predicated_index([8], [8], range(8), compact)).unwrap();
        assert!((0..8).all(|lane| plan.valid(lane).unwrap()));

        let shared = range(1);
        let mut deeply_reused = shared.clone();
        for _ in 0..MAX_PROJECTED_INDEX_DEPTH {
            deeply_reused = binary(Binary::Add, deeply_reused, integer(0));
        }
        let shallow_then_deep = binary(Binary::Add, shared, deeply_reused);
        assert!(canonicalize_graph_projected_address(&shallow_then_deep, 1, 1).is_err());

        let shared_predicate = UOp::from_operation(
            Operation::GraphCompare(crate::CompareOp::Eq),
            Some(UType::scalar(DType::Bool)),
            vec![range(1), integer(0)],
        );
        let mut deeply_reused_predicate = shared_predicate.clone();
        for _ in 0..MAX_PROJECTED_INDEX_DEPTH {
            deeply_reused_predicate = UOp::from_operation(
                Operation::GraphLogical(crate::LogicalOp::Not),
                Some(UType::scalar(DType::Bool)),
                vec![deeply_reused_predicate],
            );
        }
        let shallow_then_deep_predicate = UOp::from_operation(
            Operation::GraphLogical(crate::LogicalOp::And),
            Some(UType::scalar(DType::Bool)),
            vec![shared_predicate, deeply_reused_predicate],
        );
        assert!(canonicalize_graph_projected_predicate(&shallow_then_deep_predicate, 1).is_err());
    }

    #[test]
    fn canonical_structural_equality_visits_shared_diamonds_once() {
        fn projected_diamond(depth: usize) -> ProjectedExpr<i64> {
            let mut expression = Arc::new(ProjectedExpr::Linear);
            for _ in 0..depth {
                expression = Arc::new(ProjectedExpr::Binary {
                    operation: Binary::Add,
                    lhs: expression.clone(),
                    rhs: expression,
                });
            }
            expression.as_ref().clone()
        }

        let depth = 64;
        let lhs = projected_diamond(depth);
        let rhs = projected_diamond(depth);
        let (equal, visits) = projected_expr_eq_with_visits(&lhs, &rhs);
        assert!(equal);
        assert_eq!(visits, depth + 1);

        fn uop_diamond(depth: usize) -> UOp {
            let mut expression = range(1);
            for _ in 0..depth {
                expression = binary(Binary::Add, expression.clone(), expression);
            }
            expression
        }

        let lhs = uop_diamond(depth);
        let rhs = uop_diamond(depth);
        let (equal, visits) = uop_structure_eq_with_visits(&lhs, &rhs);
        assert!(equal);
        assert_eq!(visits, depth + 2);
    }

    #[test]
    fn projected_expression_to_uop_preserves_shared_children() {
        let shared = Arc::new(ProjectedExpr::Binary {
            operation: Binary::Mod,
            lhs: Arc::new(ProjectedExpr::Linear),
            rhs: Arc::new(ProjectedExpr::Constant(4)),
        });
        let expression = ProjectedExpr::Binary {
            operation: Binary::Add,
            lhs: shared.clone(),
            rhs: shared,
        };
        let uop = expression.to_uop(8).unwrap();
        assert!(uop.sources()[0].shares_node_with(&uop.sources()[1]));
    }

    #[test]
    fn coordinate_canonicalization_preserves_one_reshape_numerator() {
        let linear = range(8);
        let numerator = binary(Binary::Sub, integer(7), linear);
        let coordinates = vec![
            (
                binary(
                    Binary::Mod,
                    binary(Binary::FloorDiv, numerator.clone(), integer(4)),
                    integer(2),
                ),
                2,
            ),
            (binary(Binary::Mod, numerator, integer(4)), 4),
        ];
        let canonical = canonicalize_graph_projected_coordinates(&coordinates, 8).unwrap();
        assert!(
            canonical[0].sources()[0].sources()[0].shares_node_with(&canonical[1].sources()[0])
        );
    }

    #[test]
    fn predicated_addresses_are_total_and_false_lanes_never_widen_bounds() {
        let linear = range(5);
        let address = binary(Binary::Mod, linear.clone(), integer(2));
        let predicate = UOp::from_operation(
            Operation::GraphCompare(crate::CompareOp::Ge),
            Some(UType::scalar(DType::Bool)),
            vec![linear, integer(2)],
        );
        let index = predicated_index([2], [5], address, predicate);
        let plan = ProjectedIndexPlan::from_index(&index).unwrap();
        assert!(plan.is_guarded());
        assert_eq!(
            (0..5)
                .map(|lane| (plan.offset(lane).unwrap(), plan.valid(lane).unwrap()))
                .collect::<Vec<_>>(),
            vec![(0, false), (1, false), (0, true), (1, true), (0, true)]
        );

        let malformed_predicate = predicated_index(
            [2],
            [5],
            binary(Binary::Mod, range(5), integer(2)),
            integer(1),
        );
        assert!(ProjectedIndexPlan::from_index(&malformed_predicate).is_err());
        let unsafe_address = predicated_index(
            [2],
            [5],
            range(5),
            UOp::scalar_constant(DType::Bool, 0, UType::scalar(DType::Bool)),
        );
        assert!(ProjectedIndexPlan::from_index(&unsafe_address).is_err());

        let empty = predicated_index(
            [1],
            [0],
            binary(Binary::Mul, range(0), integer(0)),
            UOp::scalar_constant(DType::Bool, 0, UType::scalar(DType::Bool)),
        );
        let empty = ProjectedIndexPlan::from_index(&empty).unwrap();
        assert_eq!(empty.output_elements, 0);
        assert!(empty.valid(0).is_err());

        let addressless = predicated_index(
            [0],
            [3],
            integer(0),
            UOp::scalar_constant(DType::Bool, 0, UType::scalar(DType::Bool)),
        );
        let addressless = ProjectedIndexPlan::from_index(&addressless).unwrap();
        assert_eq!(addressless.elements, 0);
        assert!((0..3).all(|lane| !addressless.valid(lane).unwrap()));
        assert!((0..3).all(|lane| addressless.offset(lane).is_err()));
        let claimed_valid = predicated_index(
            [0],
            [3],
            integer(0),
            UOp::scalar_constant(DType::Bool, 1, UType::scalar(DType::Bool)),
        );
        assert!(ProjectedIndexPlan::from_index(&claimed_valid).is_err());
    }

    #[test]
    fn canonicalizes_authenticated_divmod_and_associative_index_algebra() {
        let linear = ProjectedExpr::Linear;
        let four = ProjectedExpr::Constant(4);
        let quotient =
            ProjectedExpr::binary(Binary::FloorDiv, linear.clone(), four.clone()).unwrap();
        let product = ProjectedExpr::binary(Binary::Mul, four.clone(), quotient).unwrap();
        let remainder = ProjectedExpr::binary(Binary::Mod, linear.clone(), four.clone()).unwrap();
        let reconstructed = ProjectedExpr::binary(Binary::Add, remainder, product).unwrap();
        let canonical_reconstruction = reconstructed.canonicalized_for_output(8);
        assert_eq!(canonical_reconstruction, linear);
        let raw_plan = ProjectedIndexPlan::from_index(&index(
            Shape::from([8]),
            Shape::from([8]),
            reconstructed.to_uop(8).unwrap(),
        ))
        .unwrap();
        let canonical_plan = ProjectedIndexPlan::from_index(&index(
            Shape::from([8]),
            Shape::from([8]),
            canonical_reconstruction.to_uop(8).unwrap(),
        ))
        .unwrap();
        assert_eq!(
            (0..8)
                .map(|lane| raw_plan.offset(lane).unwrap())
                .collect::<Vec<_>>(),
            (0..8)
                .map(|lane| canonical_plan.offset(lane).unwrap())
                .collect::<Vec<_>>()
        );

        let reverse = ProjectedExpr::binary(
            Binary::Sub,
            ProjectedExpr::Constant(7),
            ProjectedExpr::binary(
                Binary::Mul,
                ProjectedExpr::Linear,
                ProjectedExpr::Constant(1),
            )
            .unwrap(),
        )
        .unwrap();
        let reverse_plan = ProjectedIndexPlan::from_index(&index(
            Shape::from([8]),
            Shape::from([8]),
            reverse.canonicalized_for_output(8).to_uop(8).unwrap(),
        ))
        .unwrap();
        assert_eq!(
            (0..8)
                .map(|lane| reverse_plan.offset(lane).unwrap())
                .collect::<Vec<_>>(),
            (0..8_usize).rev().collect::<Vec<_>>()
        );

        let nested_division = ProjectedExpr::binary(
            Binary::FloorDiv,
            ProjectedExpr::binary(Binary::FloorDiv, ProjectedExpr::Linear, four.clone()).unwrap(),
            ProjectedExpr::Constant(2),
        )
        .unwrap();
        assert_eq!(
            nested_division.canonicalized_for_output(16),
            ProjectedExpr::binary(
                Binary::FloorDiv,
                ProjectedExpr::Linear,
                ProjectedExpr::Constant(8),
            )
            .unwrap()
        );

        let nested_remainder = ProjectedExpr::binary(
            Binary::Mod,
            ProjectedExpr::binary(Binary::Mod, ProjectedExpr::Linear, four.clone()).unwrap(),
            four,
        )
        .unwrap();
        assert_eq!(
            nested_remainder.canonicalized_for_output(8),
            ProjectedExpr::binary(
                Binary::Mod,
                ProjectedExpr::Linear,
                ProjectedExpr::Constant(4),
            )
            .unwrap()
        );

        let nested_wide_remainder = ProjectedExpr::binary(
            Binary::Mod,
            ProjectedExpr::binary(
                Binary::Mod,
                ProjectedExpr::Linear,
                ProjectedExpr::Constant(49),
            )
            .unwrap(),
            ProjectedExpr::Constant(7),
        )
        .unwrap();
        assert_eq!(
            nested_wide_remainder.canonicalized_for_output(128),
            ProjectedExpr::binary(
                Binary::Mod,
                ProjectedExpr::Linear,
                ProjectedExpr::Constant(7),
            )
            .unwrap()
        );
        let nested_wide_quotient = ProjectedExpr::binary(
            Binary::FloorDiv,
            ProjectedExpr::binary(
                Binary::Mod,
                ProjectedExpr::Linear,
                ProjectedExpr::Constant(49),
            )
            .unwrap(),
            ProjectedExpr::Constant(7),
        )
        .unwrap();
        assert_eq!(
            nested_wide_quotient.canonicalized_for_output(128),
            ProjectedExpr::binary(
                Binary::Mod,
                ProjectedExpr::binary(
                    Binary::FloorDiv,
                    ProjectedExpr::Linear,
                    ProjectedExpr::Constant(7),
                )
                .unwrap(),
                ProjectedExpr::Constant(7),
            )
            .unwrap()
        );

        let low = ProjectedExpr::binary(
            Binary::Mod,
            ProjectedExpr::Linear,
            ProjectedExpr::Constant(49),
        )
        .unwrap();
        let middle = ProjectedExpr::binary(
            Binary::Mul,
            ProjectedExpr::binary(
                Binary::Mod,
                ProjectedExpr::binary(
                    Binary::FloorDiv,
                    ProjectedExpr::Linear,
                    ProjectedExpr::Constant(147),
                )
                .unwrap(),
                ProjectedExpr::Constant(112),
            )
            .unwrap(),
            ProjectedExpr::Constant(49),
        )
        .unwrap();
        let high = ProjectedExpr::binary(
            Binary::Mul,
            ProjectedExpr::binary(
                Binary::Mod,
                ProjectedExpr::binary(
                    Binary::FloorDiv,
                    ProjectedExpr::Linear,
                    ProjectedExpr::Constant(49),
                )
                .unwrap(),
                ProjectedExpr::Constant(3),
            )
            .unwrap(),
            ProjectedExpr::Constant(614656),
        )
        .unwrap();
        let ladder = ProjectedExpr::binary(
            Binary::Add,
            ProjectedExpr::binary(Binary::Add, high, middle).unwrap(),
            low,
        )
        .unwrap();
        let selected_axis = ProjectedExpr::binary(
            Binary::Mod,
            ProjectedExpr::binary(Binary::FloorDiv, ladder, ProjectedExpr::Constant(49)).unwrap(),
            ProjectedExpr::Constant(112),
        )
        .unwrap()
        .canonicalized_for_output(118013952);
        assert_eq!(
            selected_axis,
            ProjectedExpr::binary(
                Binary::Mod,
                ProjectedExpr::binary(
                    Binary::FloorDiv,
                    ProjectedExpr::Linear,
                    ProjectedExpr::Constant(147),
                )
                .unwrap(),
                ProjectedExpr::Constant(112),
            )
            .unwrap()
        );
        let unbounded_residual = ProjectedExpr::binary(
            Binary::Add,
            ProjectedExpr::binary(
                Binary::Mul,
                ProjectedExpr::Linear,
                ProjectedExpr::Constant(49),
            )
            .unwrap(),
            ProjectedExpr::Linear,
        )
        .unwrap();
        assert!(
            ProjectedExpr::canonicalized_divmod_sum(
                Binary::FloorDiv,
                unbounded_residual,
                49,
                Some(50),
            )
            .is_none()
        );

        let base = ProjectedExpr::Linear;
        let low =
            ProjectedExpr::binary(Binary::Mod, base.clone(), ProjectedExpr::Constant(4)).unwrap();
        let middle = ProjectedExpr::binary(
            Binary::Mul,
            ProjectedExpr::binary(
                Binary::Mod,
                ProjectedExpr::binary(Binary::FloorDiv, base.clone(), ProjectedExpr::Constant(4))
                    .unwrap(),
                ProjectedExpr::Constant(3),
            )
            .unwrap(),
            ProjectedExpr::Constant(4),
        )
        .unwrap();
        let high = ProjectedExpr::binary(
            Binary::Mul,
            ProjectedExpr::binary(
                Binary::Mod,
                ProjectedExpr::binary(Binary::FloorDiv, base.clone(), ProjectedExpr::Constant(12))
                    .unwrap(),
                ProjectedExpr::Constant(2),
            )
            .unwrap(),
            ProjectedExpr::Constant(12),
        )
        .unwrap();
        let reconstructed = ProjectedExpr::binary(
            Binary::Add,
            ProjectedExpr::binary(Binary::Add, high, middle).unwrap(),
            low,
        )
        .unwrap()
        .canonicalized_for_output(24);
        assert!(projected_expr_eq(&reconstructed, &base));

        let constants = ProjectedExpr::binary(
            Binary::Add,
            ProjectedExpr::binary(
                Binary::Add,
                ProjectedExpr::Linear,
                ProjectedExpr::Constant(1),
            )
            .unwrap(),
            ProjectedExpr::Constant(2),
        )
        .unwrap();
        assert_eq!(
            constants.canonicalized_for_output(4),
            ProjectedExpr::binary(
                Binary::Add,
                ProjectedExpr::Linear,
                ProjectedExpr::Constant(3),
            )
            .unwrap()
        );
        assert_eq!(
            ProjectedExpr::binary(
                Binary::Mod,
                ProjectedExpr::Linear,
                ProjectedExpr::Constant(8),
            )
            .unwrap()
            .canonicalized_for_output(8),
            ProjectedExpr::Linear
        );

        let two = ProjectedExpr::Constant(2);
        let quotient =
            ProjectedExpr::binary(Binary::FloorDiv, ProjectedExpr::Linear, two.clone()).unwrap();
        let bounded_quotient =
            ProjectedExpr::binary(Binary::Mod, quotient.clone(), two.clone()).unwrap();
        assert_eq!(bounded_quotient.canonicalized_for_output(4), quotient);
        assert_eq!(
            bounded_quotient.canonicalized_for_output(12),
            bounded_quotient
        );
        let reconstructed = ProjectedExpr::binary(
            Binary::Add,
            ProjectedExpr::binary(Binary::Mul, bounded_quotient, two.clone()).unwrap(),
            ProjectedExpr::binary(Binary::Mod, ProjectedExpr::Linear, two).unwrap(),
        )
        .unwrap();
        assert_eq!(
            reconstructed.canonicalized_for_output(4),
            ProjectedExpr::Linear
        );

        let empty_marker = ProjectedExpr::binary(
            Binary::Mul,
            ProjectedExpr::Linear,
            ProjectedExpr::Constant(0),
        )
        .unwrap();
        assert_eq!(empty_marker.canonicalized_for_output(0), empty_marker);
    }
}
