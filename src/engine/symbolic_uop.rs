//! Authenticated lowering from bounded symbolic integers into typed scalar UOps.
//!
//! The resulting DAG is an ephemeral compiler value. It is never inserted into
//! a captured schedule or serialized as RGUA, so historical capture and UOp
//! identities remain unchanged. In particular, the exact `DefineVar` payload
//! convention below is private to this authenticated path; existing generic
//! scalar UOps retain their broader legacy contract.

use super::symbolic::SymbolicParameter;
use crate::uop::{Binary, Ternary, UType, Unary, VariableValue};
use crate::{DType, Operation, SymbolicError, SymbolicExpr, SymbolicVar, UOp, UOpError};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SymbolicUOpError {
    MalformedParameters(&'static str),
    UnknownParameter(SymbolicVar),
    Symbolic(SymbolicError),
    InvalidUOp(UOpError),
}

impl fmt::Display for SymbolicUOpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedParameters(reason) => {
                write!(f, "malformed symbolic parameter table: {reason}")
            }
            Self::UnknownParameter(variable) => write!(
                f,
                "symbolic expression references unknown parameter {}#{}",
                variable.name(),
                variable.id()
            ),
            Self::Symbolic(error) => error.fmt(f),
            Self::InvalidUOp(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for SymbolicUOpError {}

impl From<SymbolicError> for SymbolicUOpError {
    fn from(value: SymbolicError) -> Self {
        Self::Symbolic(value)
    }
}

impl From<UOpError> for SymbolicUOpError {
    fn from(value: UOpError) -> Self {
        Self::InvalidUOp(value)
    }
}

#[derive(Clone)]
struct ParameterNode {
    variable: SymbolicVar,
    slot: usize,
    node: UOp,
}

/// One scalar UOp whose `DefineVar` leaves are proven to be exact members of a
/// canonical `SymbolicSchema` parameter table.
#[derive(Clone)]
pub(crate) struct AuthenticatedSymbolicUOp {
    root: UOp,
    slots: BTreeMap<VariableValue, usize>,
}

impl AuthenticatedSymbolicUOp {
    pub(crate) fn root(&self) -> &UOp {
        &self.root
    }

    pub(crate) fn variable_slot(&self, value: &VariableValue) -> Option<usize> {
        self.slots.get(value).copied()
    }
}

/// Lowers one expression with exact schema identity. The root has scalar I64
/// type even when the source expression denotes a predicate, matching
/// `SymbolicExpr`'s public 0/1 integer contract. Predicate subexpressions remain
/// Bool UOps until their value crosses back into integer algebra.
pub(crate) fn lower_symbolic_expression(
    expression: &SymbolicExpr,
    parameters: &[SymbolicParameter],
) -> Result<AuthenticatedSymbolicUOp, SymbolicUOpError> {
    let mut lowerer = SymbolicUOpLowerer::new(parameters)?;
    let root = lowerer.value(expression)?;
    root.validate()?;
    Ok(AuthenticatedSymbolicUOp {
        root,
        slots: lowerer
            .parameters
            .values()
            .map(|parameter| {
                let Operation::DefineVar(value) = parameter.node.operation() else {
                    unreachable!("authenticated parameter nodes are DefineVar")
                };
                (value.clone(), parameter.slot)
            })
            .collect(),
    })
}

fn validate_expression(expression: &SymbolicExpr) -> Result<(), SymbolicError> {
    expression.bounds()?;
    if let SymbolicExpr::Mod(lhs, rhs) = expression {
        let lhs = lhs.bounds()?;
        let rhs = rhs.bounds()?;
        // C signed remainder and SymbolicExpr::evaluate both reject the unique
        // non-representable quotient. `SymbolicExpr::mod_bounds` needs only the
        // divisor to bound the final value, so authenticate this intermediate
        // explicitly before native source exists.
        if lhs.min == i64::MIN && (rhs.min..=rhs.max).contains(&-1) {
            return Err(SymbolicError::Overflow { op: "remainder" });
        }
    }
    Ok(())
}

struct SymbolicUOpLowerer {
    parameters: BTreeMap<u64, ParameterNode>,
    values: BTreeMap<SymbolicExpr, UOp>,
    predicates: BTreeMap<SymbolicExpr, UOp>,
}

impl SymbolicUOpLowerer {
    fn new(parameters: &[SymbolicParameter]) -> Result<Self, SymbolicUOpError> {
        if parameters
            .windows(2)
            .any(|pair| pair[0].variable().id() >= pair[1].variable().id())
        {
            return Err(SymbolicUOpError::MalformedParameters(
                "identities are not strictly increasing",
            ));
        }
        let mut names = BTreeSet::new();
        let mut nodes = BTreeMap::new();
        for (slot, parameter) in parameters.iter().enumerate() {
            let variable = parameter.variable();
            let (min, max) = variable.bounds();
            if parameter.dtype() != DType::I64
                || variable.id() == 0
                || variable.name().is_empty()
                || min > max
                || !names.insert(variable.name().to_owned())
            {
                return Err(SymbolicUOpError::MalformedParameters(
                    "one parameter has invalid identity, bounds, name, or dtype",
                ));
            }
            let bounds = SymbolicExpr::Var(variable.clone());
            bounds.bounds()?;
            let node = UOp::from_operation(
                Operation::DefineVar(VariableValue {
                    name: variable.name().to_owned(),
                    bounds,
                }),
                Some(i64_type()),
                Vec::new(),
            );
            nodes.insert(
                variable.id(),
                ParameterNode {
                    variable: variable.clone(),
                    slot,
                    node,
                },
            );
        }
        Ok(Self {
            parameters: nodes,
            values: BTreeMap::new(),
            predicates: BTreeMap::new(),
        })
    }

    fn value(&mut self, expression: &SymbolicExpr) -> Result<UOp, SymbolicUOpError> {
        if let Some(node) = self.values.get(expression) {
            return Ok(node.clone());
        }
        validate_expression(expression)?;
        let node = match expression {
            SymbolicExpr::Const(value) => UOp::constant(*value, i64_type()),
            SymbolicExpr::Var(variable) => {
                let parameter = self
                    .parameters
                    .get(&variable.id())
                    .filter(|parameter| &parameter.variable == variable)
                    .ok_or_else(|| SymbolicUOpError::UnknownParameter(variable.clone()))?;
                parameter.node.clone()
            }
            SymbolicExpr::Add(values) => {
                let mut values = values.iter();
                let mut output = values
                    .next()
                    .map(|value| self.value(value))
                    .transpose()?
                    .unwrap_or_else(|| UOp::constant(0, i64_type()));
                for value in values {
                    output = UOp::binary(Binary::Add, output, self.value(value)?);
                }
                output
            }
            SymbolicExpr::Mul(values) => {
                let mut values = values.iter();
                let mut output = values
                    .next()
                    .map(|value| self.value(value))
                    .transpose()?
                    .unwrap_or_else(|| UOp::constant(1, i64_type()));
                for value in values {
                    output = UOp::binary(Binary::Mul, output, self.value(value)?);
                }
                output
            }
            SymbolicExpr::Neg(value) => UOp::unary(Unary::Neg, self.value(value)?),
            SymbolicExpr::FloorDiv(lhs, rhs) => {
                let lhs = self.value(lhs)?;
                let rhs = self.value(rhs)?;
                UOp::binary(Binary::FloorDiv, lhs, rhs)
            }
            SymbolicExpr::Mod(lhs, rhs) => {
                let lhs = self.value(lhs)?;
                let rhs = self.value(rhs)?;
                UOp::binary(Binary::Mod, lhs, rhs)
            }
            SymbolicExpr::Min(lhs, rhs) => {
                let lhs = self.value(lhs)?;
                let rhs = self.value(rhs)?;
                UOp::binary(Binary::Min, lhs, rhs)
            }
            SymbolicExpr::Max(lhs, rhs) => {
                let lhs = self.value(lhs)?;
                let rhs = self.value(rhs)?;
                UOp::binary(Binary::Max, lhs, rhs)
            }
            SymbolicExpr::Eq(..)
            | SymbolicExpr::Lt(..)
            | SymbolicExpr::Le(..)
            | SymbolicExpr::And(..)
            | SymbolicExpr::Or(..)
            | SymbolicExpr::Not(..) => UOp::cast(self.predicate(expression)?, i64_type()),
            SymbolicExpr::Where(condition, on_true, on_false) => UOp::from_operation(
                Operation::Ternary(Ternary::Where),
                Some(i64_type()),
                vec![
                    self.predicate(condition)?,
                    self.value(on_true)?,
                    self.value(on_false)?,
                ],
            ),
        };
        self.values.insert(expression.clone(), node.clone());
        Ok(node)
    }

    fn predicate(&mut self, expression: &SymbolicExpr) -> Result<UOp, SymbolicUOpError> {
        if let Some(node) = self.predicates.get(expression) {
            return Ok(node.clone());
        }
        validate_expression(expression)?;
        let node = match expression {
            SymbolicExpr::Eq(lhs, rhs) => {
                let lhs = self.value(lhs)?;
                let rhs = self.value(rhs)?;
                UOp::binary(Binary::Eq, lhs, rhs)
            }
            SymbolicExpr::Lt(lhs, rhs) => {
                let lhs = self.value(lhs)?;
                let rhs = self.value(rhs)?;
                UOp::binary(Binary::Lt, lhs, rhs)
            }
            SymbolicExpr::Le(lhs, rhs) => {
                let lhs = self.value(lhs)?;
                let rhs = self.value(rhs)?;
                UOp::binary(Binary::Le, lhs, rhs)
            }
            SymbolicExpr::And(lhs, rhs) => {
                let lhs = self.predicate(lhs)?;
                let rhs = self.predicate(rhs)?;
                UOp::binary(Binary::And, lhs, rhs)
            }
            SymbolicExpr::Or(lhs, rhs) => {
                let lhs = self.predicate(lhs)?;
                let rhs = self.predicate(rhs)?;
                UOp::binary(Binary::Or, lhs, rhs)
            }
            SymbolicExpr::Not(value) => UOp::unary(Unary::Not, self.predicate(value)?),
            _ => {
                let value = self.value(expression)?;
                let zero = UOp::constant(0, i64_type());
                UOp::unary(Unary::Not, UOp::binary(Binary::Eq, value, zero))
            }
        };
        self.predicates.insert(expression.clone(), node.clone());
        Ok(node)
    }
}

fn i64_type() -> UType {
    UType::scalar(DType::I64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variable(id: u64, name: &str, min: i64, max: i64) -> SymbolicVar {
        SymbolicVar::from_artifact(id, name.into(), min, max).unwrap()
    }

    fn parameter(variable: SymbolicVar) -> SymbolicParameter {
        SymbolicParameter {
            variable,
            dtype: DType::I64,
        }
    }

    #[test]
    fn lowering_is_typed_complete_and_deduplicates_exact_parameters() {
        let x = variable(10_001, "x", -8, 8);
        let y = variable(10_002, "y", 1, 8);
        let x_expr = SymbolicExpr::Var(x.clone());
        let y_expr = SymbolicExpr::Var(y.clone());
        let arithmetic = SymbolicExpr::Add(vec![
            x_expr.clone(),
            SymbolicExpr::Mul(vec![SymbolicExpr::Const(2), y_expr.clone()]),
            SymbolicExpr::Neg(Box::new(x_expr.clone())),
            SymbolicExpr::FloorDiv(Box::new(y_expr.clone()), Box::new(SymbolicExpr::Const(2))),
            SymbolicExpr::Mod(Box::new(x_expr.clone()), Box::new(SymbolicExpr::Const(3))),
        ]);
        let lower = SymbolicExpr::Min(
            Box::new(arithmetic.clone()),
            Box::new(SymbolicExpr::Const(7)),
        );
        let upper = SymbolicExpr::Max(Box::new(arithmetic), Box::new(SymbolicExpr::Const(-7)));
        let condition = SymbolicExpr::Or(
            Box::new(SymbolicExpr::And(
                Box::new(SymbolicExpr::Lt(
                    Box::new(x_expr.clone()),
                    Box::new(y_expr.clone()),
                )),
                Box::new(SymbolicExpr::Not(Box::new(SymbolicExpr::Eq(
                    Box::new(x_expr.clone()),
                    Box::new(SymbolicExpr::Const(0)),
                )))),
            )),
            Box::new(SymbolicExpr::Le(
                Box::new(y_expr),
                Box::new(SymbolicExpr::Const(1)),
            )),
        );
        let expression = SymbolicExpr::Where(
            Box::new(condition.clone()),
            Box::new(lower),
            Box::new(upper),
        );
        let lowered =
            lower_symbolic_expression(&expression, &[parameter(x.clone()), parameter(y.clone())])
                .unwrap();
        assert_eq!(lowered.root().ty(), Some(i64_type()));
        let nodes = lowered.root().topological().unwrap();
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.operation(), Operation::DefineVar(_)))
                .count(),
            2
        );
        for operation in [
            Binary::Add,
            Binary::Mul,
            Binary::FloorDiv,
            Binary::Mod,
            Binary::Min,
            Binary::Max,
            Binary::Eq,
            Binary::Lt,
            Binary::Le,
            Binary::And,
            Binary::Or,
        ] {
            assert!(nodes.iter().any(
                |node| matches!(node.operation(), Operation::Binary(found) if *found == operation)
            ));
        }
        assert!(
            nodes
                .iter()
                .any(|node| matches!(node.operation(), Operation::Unary(Unary::Neg)))
        );
        assert!(
            nodes
                .iter()
                .any(|node| matches!(node.operation(), Operation::Unary(Unary::Not)))
        );
        assert!(
            nodes
                .iter()
                .any(|node| matches!(node.operation(), Operation::Ternary(Ternary::Where)))
        );
        assert_eq!(
            lowered.root().sources()[0].ty(),
            Some(UType::scalar(DType::Bool))
        );
        assert!(
            nodes
                .iter()
                .all(|node| !matches!(node.operation(), Operation::Cast))
        );
        let predicate_value =
            lower_symbolic_expression(&condition, &[parameter(x.clone()), parameter(y.clone())])
                .unwrap();
        assert!(matches!(
            predicate_value.root().operation(),
            Operation::Cast
        ));
        assert_eq!(predicate_value.root().ty(), Some(i64_type()));
        assert_eq!(
            predicate_value.root().sources()[0].ty(),
            Some(UType::scalar(DType::Bool))
        );
        let x_payload = VariableValue {
            name: x.name().into(),
            bounds: SymbolicExpr::Var(x),
        };
        let y_payload = VariableValue {
            name: y.name().into(),
            bounds: SymbolicExpr::Var(y),
        };
        assert_eq!(lowered.variable_slot(&x_payload), Some(0));
        assert_eq!(lowered.variable_slot(&y_payload), Some(1));
    }

    #[test]
    fn integer_truthiness_and_empty_folds_have_exact_typed_shapes() {
        let variable = variable(10_010, "truth", -2, 2);
        let parameter = parameter(variable.clone());
        let truth = SymbolicExpr::Where(
            Box::new(SymbolicExpr::Var(variable)),
            Box::new(SymbolicExpr::Add(Vec::new())),
            Box::new(SymbolicExpr::Mul(Vec::new())),
        );
        let lowered = lower_symbolic_expression(&truth, &[parameter]).unwrap();
        let root = lowered.root();
        assert!(matches!(
            root.operation(),
            Operation::Ternary(Ternary::Where)
        ));
        let condition = &root.sources()[0];
        assert!(matches!(
            condition.operation(),
            Operation::Unary(Unary::Not)
        ));
        assert!(matches!(
            condition.sources()[0].operation(),
            Operation::Binary(Binary::Eq)
        ));
        assert_eq!(root.sources()[1].ty(), Some(i64_type()));
        assert_eq!(root.sources()[2].ty(), Some(i64_type()));
        assert!(matches!(
            root.sources()[1].operation(),
            Operation::Const(crate::uop::LiteralValue::Int(0))
        ));
        assert!(matches!(
            root.sources()[2].operation(),
            Operation::Const(crate::uop::LiteralValue::Int(1))
        ));
    }

    #[test]
    fn closed_expressions_need_no_synthetic_parameter() {
        let lowered = lower_symbolic_expression(
            &SymbolicExpr::Add(vec![SymbolicExpr::Const(3), SymbolicExpr::Const(4)]),
            &[],
        )
        .unwrap();
        assert!(
            lowered
                .variable_slot(&VariableValue {
                    name: "absent".into(),
                    bounds: SymbolicExpr::Const(0),
                })
                .is_none()
        );
        assert!(
            lowered
                .root()
                .topological()
                .unwrap()
                .iter()
                .all(|node| !matches!(node.operation(), Operation::DefineVar(_)))
        );
    }

    #[test]
    fn malformed_parameter_tables_and_unknown_variables_fail_closed() {
        let first = variable(10_020, "first", 0, 4);
        let second = variable(10_021, "second", 0, 4);
        let expression = SymbolicExpr::Var(first.clone());
        assert!(matches!(
            lower_symbolic_expression(
                &expression,
                &[parameter(second.clone()), parameter(first.clone())]
            ),
            Err(SymbolicUOpError::MalformedParameters(_))
        ));
        let same_name = variable(10_022, "first", 0, 4);
        assert!(matches!(
            lower_symbolic_expression(
                &expression,
                &[parameter(first.clone()), parameter(same_name)]
            ),
            Err(SymbolicUOpError::MalformedParameters(_))
        ));
        let mut wrong_dtype = parameter(first.clone());
        wrong_dtype.dtype = DType::I32;
        assert!(matches!(
            lower_symbolic_expression(&expression, &[wrong_dtype]),
            Err(SymbolicUOpError::MalformedParameters(_))
        ));
        assert!(matches!(
            lower_symbolic_expression(&expression, &[parameter(second)]),
            Err(SymbolicUOpError::UnknownParameter(variable)) if variable == first
        ));
    }

    #[test]
    fn checked_symbolic_failures_precede_uop_publication() {
        let divisor = variable(10_030, "divisor", -1, 1);
        let cases = [
            SymbolicExpr::Add(vec![SymbolicExpr::Const(i64::MAX), SymbolicExpr::Const(1)]),
            SymbolicExpr::Mul(vec![SymbolicExpr::Const(i64::MAX), SymbolicExpr::Const(2)]),
            SymbolicExpr::Neg(Box::new(SymbolicExpr::Const(i64::MIN))),
            SymbolicExpr::FloorDiv(
                Box::new(SymbolicExpr::Const(i64::MIN)),
                Box::new(SymbolicExpr::Const(-1)),
            ),
            SymbolicExpr::Mod(
                Box::new(SymbolicExpr::Const(i64::MIN)),
                Box::new(SymbolicExpr::Const(-1)),
            ),
        ];
        let unused = parameter(variable(10_029, "unused", 0, 1));
        for expression in cases {
            assert!(matches!(
                lower_symbolic_expression(&expression, std::slice::from_ref(&unused)),
                Err(SymbolicUOpError::Symbolic(SymbolicError::Overflow { .. }))
            ));
        }
        for expression in [
            SymbolicExpr::FloorDiv(
                Box::new(SymbolicExpr::Const(7)),
                Box::new(SymbolicExpr::Var(divisor.clone())),
            ),
            SymbolicExpr::Mod(
                Box::new(SymbolicExpr::Const(7)),
                Box::new(SymbolicExpr::Var(divisor.clone())),
            ),
        ] {
            assert!(matches!(
                lower_symbolic_expression(&expression, &[parameter(divisor.clone())]),
                Err(SymbolicUOpError::Symbolic(SymbolicError::DivisionByZero))
            ));
        }
    }
}
