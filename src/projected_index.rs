use crate::uop::{
    AddressSpace, AddressValue, Binary, IndexAddressing, IndexValue, LiteralValue, Operation, UOp,
    UOpError,
};

const MAX_PROJECTED_INDEX_DEPTH: usize = 256;
const MAX_PROJECTED_INDEX_NODES: usize = 4096;

/// One completely validated explicit physical-address expression attached to
/// an ordinary Buffer index. Historical Buffer indices carry a Range as their
/// second source and retain descriptor-derived broadcast addressing; a
/// projected index carries this deliberately small core-integer UOp algebra.
///
/// Keeping the expression in the UOp DAG makes its identity and sharing
/// canonical. This plan is the sole semantic parser used by interpreters and
/// renderers, so backend code never rediscovers movement-operation policy.
pub(crate) struct ProjectedIndexPlan<'a> {
    pub(crate) buffer: u64,
    pub(crate) elements: usize,
    pub(crate) output_elements: usize,
    pub(crate) expression: &'a UOp,
    fits_i32: bool,
}

pub(crate) trait ProjectedIndexEmitter {
    type Value;
    type Error;

    fn linear(&mut self) -> Result<Self::Value, Self::Error>;
    fn constant(&mut self, value: i64) -> Result<Self::Value, Self::Error>;
    fn binary(
        &mut self,
        operation: Binary,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Value, Self::Error>;
}

impl<'a> ProjectedIndexPlan<'a> {
    pub(crate) fn is_projected(index: &UOp) -> bool {
        matches!(
            index.operation(),
            Operation::Index(IndexValue::Buffer {
                addressing: IndexAddressing::Projected,
                ..
            })
        )
    }

    pub(crate) fn from_index(index: &'a UOp) -> Result<Self, UOpError> {
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
        let mut state = ValidationState {
            output_elements,
            nodes: 0,
            fits_i32: true,
        };
        let bounds = validate_expression(expression, 0, &mut state)?;
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

    pub(crate) fn emit<E: ProjectedIndexEmitter>(
        &self,
        emitter: &mut E,
    ) -> Result<E::Value, E::Error> {
        emit_expression(self.expression, emitter)
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
            fn constant(&mut self, value: i64) -> Result<Self::Value, Self::Error> {
                Ok(i128::from(value))
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
    plan: &ProjectedIndexPlan<'_>,
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
        fn constant(&mut self, value: i64) -> Result<Self::Value, Self::Error> {
            (self.literal)(value)
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

fn emit_expression<E: ProjectedIndexEmitter>(
    expression: &UOp,
    emitter: &mut E,
) -> Result<E::Value, E::Error> {
    match expression.operation() {
        Operation::Range(0) => emitter.linear(),
        Operation::Const(LiteralValue::Int(value)) => emitter.constant(*value),
        Operation::Binary(operation @ (Binary::Add | Binary::Sub | Binary::Mul)) => {
            let lhs = emit_expression(&expression.sources()[0], emitter)?;
            let rhs = emit_expression(&expression.sources()[1], emitter)?;
            emitter.binary(*operation, lhs, rhs)
        }
        Operation::Binary(operation @ (Binary::FloorDiv | Binary::Mod)) => {
            let lhs = emit_expression(&expression.sources()[0], emitter)?;
            let rhs = emit_expression(&expression.sources()[1], emitter)?;
            emitter.binary(*operation, lhs, rhs)
        }
        _ => unreachable!("ProjectedIndexPlan validated its expression"),
    }
}

struct ValidationState {
    output_elements: usize,
    nodes: usize,
    fits_i32: bool,
}

fn validate_expression(
    expression: &UOp,
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
    if expression.ty() != Some(crate::UType::scalar(crate::DType::I64)) {
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
            if usize::try_from(*bound).ok() != Some(state.output_elements) {
                return Err(UOpError::InvalidIndex);
            }
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
        Operation::Const(LiteralValue::Int(value)) if expression.sources().is_empty() => {
            state.fits_i32 &= i32::try_from(*value).is_ok();
            let value = i128::from(*value);
            Ok(Some((value, value)))
        }
        Operation::Binary(operation) if expression.sources().len() == 2 => {
            let lhs = validate_expression(&expression.sources()[0], depth + 1, state)?;
            let rhs = validate_expression(&expression.sources()[1], depth + 1, state)?;
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
        _ => Err(UOpError::InvalidIndex),
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
    }
}
