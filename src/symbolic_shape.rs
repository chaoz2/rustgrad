//! Symbolic shapes are planning values; CPU tensors only receive their bound,
//! concrete [`Shape`](crate::Shape) specialization.
use crate::symbolic::{SymbolicError, SymbolicExpr, SymbolicVar};
use crate::{Error, Result, Shape};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SymbolicDim(SymbolicExpr);
impl SymbolicDim {
    pub fn new(expression: SymbolicExpr) -> Self {
        Self(expression)
    }
    pub fn expression(&self) -> &SymbolicExpr {
        &self.0
    }
    pub fn bind(
        &self,
        bindings: &BTreeMap<SymbolicVar, i64>,
    ) -> std::result::Result<usize, SymbolicError> {
        self.0.as_usize(bindings)
    }
}
impl From<usize> for SymbolicDim {
    fn from(x: usize) -> Self {
        Self(SymbolicExpr::constant(x as i64))
    }
}
impl From<SymbolicExpr> for SymbolicDim {
    fn from(x: SymbolicExpr) -> Self {
        Self(x)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SymbolicShape(Vec<SymbolicDim>);
impl SymbolicShape {
    pub fn new(dims: impl Into<Vec<SymbolicDim>>) -> Self {
        Self(dims.into())
    }
    pub fn dims(&self) -> &[SymbolicDim] {
        &self.0
    }
    pub fn rank(&self) -> usize {
        self.0.len()
    }
    pub fn numel(&self) -> std::result::Result<SymbolicExpr, SymbolicError> {
        self.0
            .iter()
            .try_fold(SymbolicExpr::constant(1), |n, d| Ok(n * d.0.clone()))
    }
    /// Replaces shape variables through the checked expression boundary.
    /// This remains a planning-only value; callers must still bind the result
    /// before passing it to a concrete graph or allocation API.
    pub fn substitute(
        &self,
        replacements: &BTreeMap<SymbolicVar, SymbolicExpr>,
    ) -> std::result::Result<Self, SymbolicError> {
        let variables = self
            .0
            .iter()
            .flat_map(|dimension| dimension.expression().variables())
            .collect::<std::collections::BTreeSet<_>>();
        if let Some(extra) = replacements
            .keys()
            .find(|variable| !variables.contains(*variable))
        {
            return Err(SymbolicError::ExtraBinding(extra.clone()));
        }
        self.0
            .iter()
            .map(|dimension| {
                let local = dimension
                    .expression()
                    .variables()
                    .into_iter()
                    .filter_map(|variable| {
                        replacements
                            .get(&variable)
                            .cloned()
                            .map(|replacement| (variable, replacement))
                    })
                    .collect();
                dimension
                    .expression()
                    .substitute(&local)
                    .map(|simplified| SymbolicDim::new(simplified.expression))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map(Self::new)
    }
    /// This is the explicit allocation boundary: it validates every binding and
    /// never permits an unbound shape to enter the concrete tensor API.
    pub fn bind(
        &self,
        bindings: &BTreeMap<SymbolicVar, i64>,
    ) -> std::result::Result<Shape, SymbolicError> {
        let used = self
            .0
            .iter()
            .flat_map(|d| d.expression().variables())
            .collect::<std::collections::BTreeSet<_>>();
        if let Some(extra) = bindings.keys().find(|v| !used.contains(*v)) {
            return Err(SymbolicError::ExtraBinding(extra.clone()));
        }
        let shape = self
            .0
            .iter()
            .map(|d| {
                // An axis need not mention every shape variable.  Preserve the
                // expression evaluator's strict extra-binding contract by
                // projecting the full shape environment to that axis.
                let axis_bindings = d
                    .expression()
                    .variables()
                    .into_iter()
                    .filter_map(|v| bindings.get(&v).map(|value| (v, *value)))
                    .collect();
                d.bind(&axis_bindings)
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map(Shape::new)?;
        // Binding is the last symbolic boundary before graph construction.
        // A concrete Shape can hold dimensions whose product is invalid, so
        // reject that extent here instead of letting input_symbolic append an
        // unusable graph leaf.
        shape
            .numel()
            .map_err(|_| SymbolicError::Overflow { op: "shape extent" })?;
        Ok(shape)
    }
    pub fn broadcast_with(&self, other: &Self) -> std::result::Result<Self, SymbolicError> {
        let rank = self.rank().max(other.rank());
        let mut out = Vec::with_capacity(rank);
        for offset in (0..rank).rev() {
            let a = self
                .0
                .get(self.rank().wrapping_sub(1 + offset))
                .cloned()
                .unwrap_or_else(|| 1usize.into());
            let b = other
                .0
                .get(other.rank().wrapping_sub(1 + offset))
                .cloned()
                .unwrap_or_else(|| 1usize.into());
            let ae = a.0.clone();
            let be = b.0.clone();
            let ab = ae.bounds()?;
            let bb = be.bounds()?;
            if ab.constant() == Some(1) {
                out.push(b)
            } else if bb.constant() == Some(1) || ae == be {
                out.push(a)
            } else {
                return Err(SymbolicError::InvalidBounds {
                    min: ab.min.max(bb.min),
                    max: ab.max.min(bb.max),
                });
            }
        }
        out.reverse();
        Ok(Self(out))
    }
    /// Product equality is accepted only when structural simplification proves it.
    pub fn reshape_compatible(&self, target: &Self) -> std::result::Result<bool, SymbolicError> {
        Ok(self.numel()?.simplify()?.expression == target.numel()?.simplify()?.expression)
    }
    pub fn bind_for_graph(&self, bindings: &BTreeMap<SymbolicVar, i64>) -> Result<Shape> {
        self.bind(bindings).map_err(symbolic_error)
    }
}
fn symbolic_error(error: SymbolicError) -> Error {
    Error::Serialization {
        reason: format!("symbolic shape: {error}"),
    }
}
