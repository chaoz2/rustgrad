use crate::uop::{
    AddressSpace, AddressValue, Binary, IndexAddressing, IndexValue, LiteralValue, Operation, UOp,
    UOpError,
};

pub(crate) const MAX_PROJECTED_INDEX_DEPTH: usize = 256;
pub(crate) const MAX_PROJECTED_INDEX_NODES: usize = 4096;

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
    fits_i32: bool,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum ProjectedExpr<C> {
    Linear,
    Constant(C),
    Binary {
        operation: Binary,
        lhs: Box<Self>,
        rhs: Box<Self>,
    },
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
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
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
                lhs: Box::new(lhs.try_map(map)?),
                rhs: Box::new(rhs.try_map(map)?),
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
                let mut lhs = lhs.canonicalized_inner(output_elements);
                let mut rhs = rhs.canonicalized_inner(output_elements);
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
                match (operation, &lhs, &rhs) {
                    (Binary::Add | Binary::Sub, _, Self::Constant(0)) => return lhs,
                    (Binary::Add, Self::Constant(0), _) => return rhs,
                    (Binary::Mul, _, Self::Constant(1))
                    | (Binary::FloorDiv, _, Self::Constant(1)) => return lhs,
                    (Binary::Mul, Self::Constant(1), _) => return rhs,
                    (Binary::Mul, _, Self::Constant(0))
                    | (Binary::Mul, Self::Constant(0), _)
                    | (Binary::Mod, _, Self::Constant(1)) => return Self::Constant(0),
                    _ => {}
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
                    && inner_divisor.as_ref() == &rhs
                {
                    return lhs;
                }
                if matches!(operation, Binary::FloorDiv)
                    && let Self::Binary {
                        operation: Binary::FloorDiv,
                        lhs: numerator,
                        rhs: inner_divisor,
                    } = &lhs
                    && let (Self::Constant(inner), Self::Constant(outer)) =
                        (inner_divisor.as_ref(), &rhs)
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
                            rhs: Box::new(Self::Constant(divisor)),
                        }
                    };
                }

                if matches!(operation, Binary::Add)
                    && let Some(value) = Self::recombined_divmod(&lhs, &rhs)
                        .or_else(|| Self::recombined_divmod(&rhs, &lhs))
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
                    && *inner_operation == *operation
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
                                operation: *operation,
                                lhs: inner_lhs.clone(),
                                rhs: Box::new(Self::Constant(combined)),
                            },
                        };
                    }
                }

                Self::Binary {
                    operation: *operation,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }
            }
        }
    }

    fn recombined_divmod(product: &Self, remainder: &Self) -> Option<Self> {
        let Self::Binary {
            operation: Binary::Mod,
            lhs: remainder_numerator,
            rhs: remainder_divisor,
        } = remainder
        else {
            return None;
        };
        let Self::Binary {
            operation: Binary::Mul,
            lhs,
            rhs,
        } = product
        else {
            return None;
        };
        let quotient = [(lhs.as_ref(), rhs.as_ref()), (rhs.as_ref(), lhs.as_ref())]
            .into_iter()
            .find_map(|(quotient, multiplier)| {
                let Self::Binary {
                    operation: Binary::FloorDiv,
                    lhs: numerator,
                    rhs: divisor,
                } = quotient
                else {
                    return None;
                };
                (multiplier == divisor.as_ref()).then_some((numerator.as_ref(), divisor.as_ref()))
            })?;
        if matches!(quotient.1, Self::Constant(divisor) if *divisor > 0)
            && quotient.0 == remainder_numerator.as_ref()
            && quotient.1 == remainder_divisor.as_ref()
        {
            Some(quotient.0.clone())
        } else {
            None
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
        fn build(expression: &ProjectedExpr<i64>, range: &UOp) -> UOp {
            let ty = crate::UType::scalar(crate::DType::I64);
            match expression {
                ProjectedExpr::Linear => range.clone(),
                ProjectedExpr::Constant(value) => UOp::constant(*value, ty),
                ProjectedExpr::Binary {
                    operation,
                    lhs,
                    rhs,
                } => UOp::from_operation(
                    Operation::Binary(*operation),
                    Some(ty),
                    vec![build(lhs, range), build(rhs, range)],
                ),
            }
        }
        Ok(build(self, &range))
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

impl ProjectedIndexPlan {
    pub(crate) fn is_projected(index: &UOp) -> bool {
        matches!(
            index.operation(),
            Operation::Index(IndexValue::Buffer {
                addressing: IndexAddressing::Projected,
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
            addressing: IndexAddressing::Projected,
        }) = index.operation()
        else {
            return Err(UOpError::InvalidIndex);
        };
        let [address, expression] = index.sources() else {
            return Err(UOpError::InvalidIndex);
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
        let mut state = ValidationState {
            output_elements,
            nodes: 0,
            fits_i32: true,
        };
        let bounds = validate_expression(&expression, 0, &mut state)?;
        if state.nodes > MAX_PROJECTED_INDEX_NODES {
            return Err(UOpError::InvalidIndex);
        }
        match (output_elements, bounds) {
            (0, None) => {}
            (0, Some(_)) | (_, None) => return Err(UOpError::InvalidIndex),
            (_, Some((minimum, maximum))) => {
                let elements = i128::try_from(*elements).map_err(|_| UOpError::InvalidIndex)?;
                if minimum < 0 || maximum >= elements {
                    return Err(UOpError::InvalidIndex);
                }
            }
        }
        Ok(Self {
            buffer: *buffer,
            elements: *elements,
            output_elements,
            expression,
            fits_i32: state.fits_i32,
        })
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

    pub(crate) fn offset(&self, linear: usize) -> Result<usize, UOpError> {
        if linear >= self.output_elements {
            return Err(UOpError::InvalidIndex);
        }
        struct Evaluator {
            linear: i128,
        }
        impl ProjectedIndexEmitter for Evaluator {
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
        let mut evaluator = Evaluator {
            linear: i128::try_from(linear).map_err(|_| UOpError::InvalidIndex)?,
        };
        let offset = self.emit(&mut evaluator)?;
        let offset = usize::try_from(offset).map_err(|_| UOpError::InvalidIndex)?;
        (offset < self.elements)
            .then_some(offset)
            .ok_or(UOpError::InvalidIndex)
    }
}

pub(crate) fn render_infix_projected_index(
    plan: &ProjectedIndexPlan,
    linear: impl Into<String>,
    mut literal: impl FnMut(i64) -> Result<String, UOpError>,
) -> Result<String, UOpError> {
    struct Infix<'a, F> {
        linear: String,
        literal: &'a mut F,
    }
    impl<F: FnMut(i64) -> Result<String, UOpError>> ProjectedIndexEmitter for Infix<'_, F> {
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
    plan.emit(&mut Infix {
        linear: linear.into(),
        literal: &mut literal,
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

struct ValidationState {
    output_elements: usize,
    nodes: usize,
    fits_i32: bool,
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
                shared_diamond,
            ))
            .is_err()
        );
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

        let empty_marker = ProjectedExpr::binary(
            Binary::Mul,
            ProjectedExpr::Linear,
            ProjectedExpr::Constant(0),
        )
        .unwrap();
        assert_eq!(empty_marker.canonicalized_for_output(0), empty_marker);
    }
}
