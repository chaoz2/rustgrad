//! Deterministic symbolic integers used by shape specialization.
//!
//! Expressions use mathematical, unbounded integer semantics at construction
//! time.  Evaluation and bound propagation are checked `i64` operations; an
//! expression which can overflow or divide by zero is rejected instead of
//! acquiring a target-dependent meaning.
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    ops::{Add, Mul, Neg, Not, Sub},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_VARIABLE_ID: AtomicU64 = AtomicU64::new(1);

/// A variable's identity is deliberately independent from its display name.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SymbolicVar {
    id: u64,
    name: String,
    min: i64,
    max: i64,
}
impl SymbolicVar {
    pub fn new(name: impl Into<String>, min: i64, max: i64) -> Result<Self, SymbolicError> {
        if min > max {
            return Err(SymbolicError::InvalidBounds { min, max });
        }
        Ok(Self {
            id: NEXT_VARIABLE_ID.fetch_add(1, Ordering::Relaxed),
            name: name.into(),
            min,
            max,
        })
    }
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn bounds(&self) -> (i64, i64) {
        (self.min, self.max)
    }
    pub(crate) fn from_artifact(
        id: u64,
        name: String,
        min: i64,
        max: i64,
    ) -> Result<Self, SymbolicError> {
        if id == 0 || id == u64::MAX || min > max || name.is_empty() {
            return Err(SymbolicError::InvalidBounds { min, max });
        }
        NEXT_VARIABLE_ID.fetch_max(id + 1, Ordering::Relaxed);
        Ok(Self { id, name, min, max })
    }
}

/// Inclusive bounds.  `min == max` means the expression is provably constant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bounds {
    pub min: i64,
    pub max: i64,
}
impl Bounds {
    pub fn constant(self) -> Option<i64> {
        (self.min == self.max).then_some(self.min)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SymbolicError {
    InvalidBounds { min: i64, max: i64 },
    DivisionByZero,
    Overflow { op: &'static str },
    MissingBinding(SymbolicVar),
    ExtraBinding(SymbolicVar),
    OutOfBounds { variable: SymbolicVar, value: i64 },
    NotUsize(i64),
    RewriteLimit,
}
impl fmt::Display for SymbolicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBounds { min, max } => write!(f, "invalid symbolic bounds [{min}, {max}]"),
            Self::DivisionByZero => write!(f, "symbolic floor division or modulo by zero"),
            Self::Overflow { op } => write!(f, "symbolic {op} overflows i64"),
            Self::MissingBinding(v) => write!(f, "missing binding for {}#{}", v.name, v.id),
            Self::ExtraBinding(v) => write!(f, "binding for unused {}#{}", v.name, v.id),
            Self::OutOfBounds { variable, value } => write!(
                f,
                "binding {value} is outside {}#{} bounds [{}, {}]",
                variable.name, variable.id, variable.min, variable.max
            ),
            Self::NotUsize(x) => write!(f, "{x} is not a usize"),
            Self::RewriteLimit => write!(f, "symbolic rewrite limit reached"),
        }
    }
}
impl std::error::Error for SymbolicError {}

/// Serialization-ready, structurally ordered expression tree. Booleans are
/// represented as `0` and `1`, so they can participate in shape predicates.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum SymbolicExpr {
    Const(i64),
    Var(SymbolicVar),
    Add(Vec<Self>),
    Mul(Vec<Self>),
    Neg(Box<Self>),
    FloorDiv(Box<Self>, Box<Self>),
    Mod(Box<Self>, Box<Self>),
    Min(Box<Self>, Box<Self>),
    Max(Box<Self>, Box<Self>),
    Eq(Box<Self>, Box<Self>),
    Lt(Box<Self>, Box<Self>),
    Le(Box<Self>, Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Not(Box<Self>),
    Where(Box<Self>, Box<Self>, Box<Self>),
}
impl SymbolicExpr {
    pub const fn constant(value: i64) -> Self {
        Self::Const(value)
    }
    pub fn variable(name: impl Into<String>, min: i64, max: i64) -> Result<Self, SymbolicError> {
        Ok(Self::Var(SymbolicVar::new(name, min, max)?))
    }
    pub fn variables(&self) -> BTreeSet<SymbolicVar> {
        let mut out = BTreeSet::new();
        self.collect_variables(&mut out);
        out
    }
    fn collect_variables(&self, out: &mut BTreeSet<SymbolicVar>) {
        match self {
            Self::Var(v) => {
                out.insert(v.clone());
            }
            Self::Const(_) => {}
            Self::Add(xs) | Self::Mul(xs) => {
                for x in xs {
                    x.collect_variables(out);
                }
            }
            Self::Neg(x) | Self::Not(x) => x.collect_variables(out),
            Self::FloorDiv(a, b)
            | Self::Mod(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b)
            | Self::Eq(a, b)
            | Self::Lt(a, b)
            | Self::Le(a, b)
            | Self::And(a, b)
            | Self::Or(a, b) => {
                a.collect_variables(out);
                b.collect_variables(out);
            }
            Self::Where(c, a, b) => {
                c.collect_variables(out);
                a.collect_variables(out);
                b.collect_variables(out);
            }
        }
    }
    pub fn bounds(&self) -> Result<Bounds, SymbolicError> {
        self.bounds_inner()
    }
    fn bounds_inner(&self) -> Result<Bounds, SymbolicError> {
        use SymbolicExpr::*;
        match self {
            Const(x) => Ok(Bounds { min: *x, max: *x }),
            Var(v) => Ok(Bounds {
                min: v.min,
                max: v.max,
            }),
            Add(xs) => xs.iter().try_fold(Bounds { min: 0, max: 0 }, |a, x| {
                let b = x.bounds()?;
                Ok(Bounds {
                    min: add(a.min, b.min)?,
                    max: add(a.max, b.max)?,
                })
            }),
            Mul(xs) => xs
                .iter()
                .try_fold(Bounds { min: 1, max: 1 }, |a, x| mul_bounds(a, x.bounds()?)),
            Neg(x) => {
                let b = x.bounds()?;
                Ok(Bounds {
                    min: neg(b.max)?,
                    max: neg(b.min)?,
                })
            }
            Min(a, b) => {
                let a = a.bounds()?;
                let b = b.bounds()?;
                Ok(Bounds {
                    min: a.min.min(b.min),
                    max: a.max.min(b.max),
                })
            }
            Max(a, b) => {
                let a = a.bounds()?;
                let b = b.bounds()?;
                Ok(Bounds {
                    min: a.min.max(b.min),
                    max: a.max.max(b.max),
                })
            }
            Eq(a, b) => predicate_bounds(a.bounds()?, b.bounds()?, |a, b| a == b),
            Lt(a, b) => predicate_bounds(a.bounds()?, b.bounds()?, |a, b| a < b),
            Le(a, b) => predicate_bounds(a.bounds()?, b.bounds()?, |a, b| a <= b),
            And(a, b) | Or(a, b) => {
                let a = bool_possibilities(a.bounds()?);
                let b = bool_possibilities(b.bounds()?);
                let values = a.into_iter().flat_map(|x| {
                    b.iter().copied().map(move |y| {
                        if matches!(self, And(..)) {
                            x && y
                        } else {
                            x || y
                        }
                    })
                });
                let values = values.map(i64::from).collect::<Vec<_>>();
                Ok(Bounds {
                    min: *values.iter().min().unwrap(),
                    max: *values.iter().max().unwrap(),
                })
            }
            Not(x) => {
                let values = bool_possibilities(x.bounds()?)
                    .into_iter()
                    .map(|v| (!v) as i64)
                    .collect::<Vec<_>>();
                Ok(Bounds {
                    min: *values.iter().min().unwrap(),
                    max: *values.iter().max().unwrap(),
                })
            }
            Where(c, a, b) => {
                let c = c.bounds()?;
                let a = a.bounds()?;
                let b = b.bounds()?;
                if c.min != 0 && c.max != 0 {
                    Ok(a)
                } else if c.min == 0 && c.max == 0 {
                    Ok(b)
                } else {
                    Ok(Bounds {
                        min: a.min.min(b.min),
                        max: a.max.max(b.max),
                    })
                }
            }
            FloorDiv(a, b) => div_bounds(a.bounds()?, b.bounds()?),
            Mod(a, b) => mod_bounds(a.bounds()?, b.bounds()?),
        }
    }
    pub fn evaluate(&self, bindings: &BTreeMap<SymbolicVar, i64>) -> Result<i64, SymbolicError> {
        let vars = self.variables();
        for v in &vars {
            let x = *bindings
                .get(v)
                .ok_or_else(|| SymbolicError::MissingBinding(v.clone()))?;
            if x < v.min || x > v.max {
                return Err(SymbolicError::OutOfBounds {
                    variable: v.clone(),
                    value: x,
                });
            }
        }
        for v in bindings.keys() {
            if !vars.contains(v) {
                return Err(SymbolicError::ExtraBinding(v.clone()));
            }
        }
        self.eval_inner(bindings)
    }
    fn eval_inner(&self, b: &BTreeMap<SymbolicVar, i64>) -> Result<i64, SymbolicError> {
        use SymbolicExpr::*;
        match self {
            Const(x) => Ok(*x),
            Var(v) => Ok(b[v]),
            Add(xs) => xs.iter().try_fold(0, |a, x| add(a, x.eval_inner(b)?)),
            Mul(xs) => xs.iter().try_fold(1, |a, x| mul(a, x.eval_inner(b)?)),
            Neg(x) => neg(x.eval_inner(b)?),
            FloorDiv(a, c) => floor_div(a.eval_inner(b)?, c.eval_inner(b)?),
            Mod(a, c) => floor_mod(a.eval_inner(b)?, c.eval_inner(b)?),
            Min(a, c) => Ok(a.eval_inner(b)?.min(c.eval_inner(b)?)),
            Max(a, c) => Ok(a.eval_inner(b)?.max(c.eval_inner(b)?)),
            Eq(a, c) => Ok((a.eval_inner(b)? == c.eval_inner(b)?) as i64),
            Lt(a, c) => Ok((a.eval_inner(b)? < c.eval_inner(b)?) as i64),
            Le(a, c) => Ok((a.eval_inner(b)? <= c.eval_inner(b)?) as i64),
            And(a, c) => Ok((boolv(a.eval_inner(b)?) && boolv(c.eval_inner(b)?)) as i64),
            Or(a, c) => Ok((boolv(a.eval_inner(b)?) || boolv(c.eval_inner(b)?)) as i64),
            Not(x) => Ok((!boolv(x.eval_inner(b)?)) as i64),
            Where(c, a, d) => {
                if boolv(c.eval_inner(b)?) {
                    a.eval_inner(b)
                } else {
                    d.eval_inner(b)
                }
            }
        }
    }
    pub fn try_floor_div(self, rhs: Self) -> Result<Self, SymbolicError> {
        if rhs.bounds()?.min <= 0 && rhs.bounds()?.max >= 0 {
            return Err(SymbolicError::DivisionByZero);
        }
        Ok(Self::FloorDiv(Box::new(self), Box::new(rhs)))
    }
    pub fn try_modulo(self, rhs: Self) -> Result<Self, SymbolicError> {
        if rhs.bounds()?.min <= 0 && rhs.bounds()?.max >= 0 {
            return Err(SymbolicError::DivisionByZero);
        }
        Ok(Self::Mod(Box::new(self), Box::new(rhs)))
    }
    pub fn minimum(self, rhs: Self) -> Self {
        Self::Min(Box::new(self), Box::new(rhs))
    }
    pub fn maximum(self, rhs: Self) -> Self {
        Self::Max(Box::new(self), Box::new(rhs))
    }
    pub fn eq_expr(self, rhs: Self) -> Self {
        Self::Eq(Box::new(self), Box::new(rhs))
    }
    pub fn lt(self, rhs: Self) -> Self {
        Self::Lt(Box::new(self), Box::new(rhs))
    }
    pub fn le(self, rhs: Self) -> Self {
        Self::Le(Box::new(self), Box::new(rhs))
    }
    pub fn and(self, rhs: Self) -> Self {
        Self::And(Box::new(self), Box::new(rhs))
    }
    pub fn or(self, rhs: Self) -> Self {
        Self::Or(Box::new(self), Box::new(rhs))
    }
    pub fn where_(self, yes: Self, no: Self) -> Self {
        Self::Where(Box::new(self), Box::new(yes), Box::new(no))
    }
    pub fn as_i64(&self, b: &BTreeMap<SymbolicVar, i64>) -> Result<i64, SymbolicError> {
        self.evaluate(b)
    }
    pub fn as_usize(&self, b: &BTreeMap<SymbolicVar, i64>) -> Result<usize, SymbolicError> {
        let x = self.evaluate(b)?;
        usize::try_from(x).map_err(|_| SymbolicError::NotUsize(x))
    }
    /// A deterministic, terminating rewrite pass. Trace entries name each accepted rewrite.
    pub fn simplify(&self) -> Result<Simplified, SymbolicError> {
        let mut current = self.clone();
        let mut trace = Vec::new();
        for _ in 0..128 {
            let next = simplify_once(&current, &mut trace)?;
            if next == current {
                return Ok(Simplified {
                    expression: next,
                    trace,
                });
            }
            current = next;
        }
        Err(SymbolicError::RewriteLimit)
    }
}
impl Add for SymbolicExpr {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        Self::Add(vec![self, r])
    }
}
impl Sub for SymbolicExpr {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        self + (-r)
    }
}
impl Mul for SymbolicExpr {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        Self::Mul(vec![self, r])
    }
}
impl Neg for SymbolicExpr {
    type Output = Self;
    fn neg(self) -> Self {
        Self::Neg(Box::new(self))
    }
}
impl Not for SymbolicExpr {
    type Output = Self;
    fn not(self) -> Self {
        Self::Not(Box::new(self))
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Simplified {
    pub expression: SymbolicExpr,
    pub trace: Vec<&'static str>,
}
fn simplify_once(
    x: &SymbolicExpr,
    trace: &mut Vec<&'static str>,
) -> Result<SymbolicExpr, SymbolicError> {
    use SymbolicExpr::*;
    let out = match x {
        Const(_) | Var(_) => x.clone(),
        Neg(a) => match simplify_once(a, trace)? {
            Const(v) => {
                trace.push("fold-neg");
                Const(neg(v)?)
            }
            Neg(v) => {
                trace.push("double-neg");
                *v
            }
            a => Neg(Box::new(a)),
        },
        Add(xs) => simplify_product_or_sum(xs, false, trace)?,
        Mul(xs) => simplify_product_or_sum(xs, true, trace)?,
        FloorDiv(a, b) => simplify_divmod(a, b, true, trace)?,
        Mod(a, b) => simplify_divmod(a, b, false, trace)?,
        Min(a, b) => simplify_minmax(a, b, true, trace)?,
        Max(a, b) => simplify_minmax(a, b, false, trace)?,
        Eq(a, b) | Lt(a, b) | Le(a, b) => simplify_compare(a, b, x, trace)?,
        And(a, b) | Or(a, b) => simplify_logic(a, b, matches!(x, And(..)), trace)?,
        Not(a) => match simplify_once(a, trace)? {
            Const(v) => Const((!boolv(v)) as i64),
            Not(v) => *v,
            a => Not(Box::new(a)),
        },
        Where(c, a, b) => {
            let c = simplify_once(c, trace)?;
            let a = simplify_once(a, trace)?;
            let b = simplify_once(b, trace)?;
            if a == b {
                a
            } else if let Some(v) = c.bounds()?.constant() {
                if boolv(v) { a } else { b }
            } else {
                Where(Box::new(c), Box::new(a), Box::new(b))
            }
        }
    };
    Ok(out)
}
fn simplify_product_or_sum(
    xs: &[SymbolicExpr],
    product: bool,
    t: &mut Vec<&'static str>,
) -> Result<SymbolicExpr, SymbolicError> {
    use SymbolicExpr::*;
    if product {
        // Do not reassociate multiplication: checked `i64` multiplication can
        // overflow in one grouping but not another.
        let mut factors = xs
            .iter()
            .map(|x| simplify_once(x, t))
            .collect::<Result<Vec<_>, _>>()?;
        factors.retain(|x| *x != Const(1));
        return Ok(match factors.len() {
            0 => Const(1),
            1 => factors.pop().unwrap(),
            _ => Mul(factors),
        });
    }
    let mut flat = Vec::new();
    let mut direct = Vec::new();
    for x in xs {
        let x = simplify_once(x, t)?;
        direct.push(x.clone());
        match x {
            Add(ys) => flat.extend(ys),
            Const(v) => flat.push(Const(v)),
            x => flat.push(x),
        }
    }
    if !safe_to_reassociate_addition(&flat) {
        return Ok(Add(direct));
    }
    let mut constant = 0;
    let mut non_constants = Vec::new();
    for term in flat {
        match term {
            Const(value) => constant = add(constant, value)?,
            term => non_constants.push(term),
        }
    }
    flat = non_constants;
    if constant != 0 {
        flat.push(Const(constant));
    }
    flat.sort();
    if !product {
        let mut combined = Vec::new();
        let mut index = 0;
        while index < flat.len() {
            let mut end = index + 1;
            while end < flat.len() && flat[end] == flat[index] {
                end += 1;
            }
            let count = end - index;
            if count == 1 {
                combined.push(flat[index].clone());
            } else {
                t.push("combine-like-terms");
                combined.push(Mul(vec![
                    Const(
                        i64::try_from(count)
                            .map_err(|_| SymbolicError::Overflow { op: "term count" })?,
                    ),
                    flat[index].clone(),
                ]));
            }
            index = end;
        }
        flat = combined;
    }
    Ok(if flat.is_empty() {
        Const(0)
    } else if flat.len() == 1 {
        t.push("identity");
        flat.pop().unwrap()
    } else {
        Add(flat)
    })
}
fn safe_to_reassociate_addition(terms: &[SymbolicExpr]) -> bool {
    // Any order is safe when the sum of all possible positive contributions
    // and all possible negative contributions both fit. This is stronger than
    // necessary, intentionally conservative, and preserves checked semantics.
    let mut positive = 0i64;
    let mut negative = 0i64;
    for term in terms {
        let Ok(bounds) = term.bounds() else {
            return false;
        };
        if bounds.max > 0 && positive.checked_add(bounds.max).is_none() {
            return false;
        }
        if bounds.max > 0 {
            positive += bounds.max;
        }
        if bounds.min < 0 && negative.checked_add(bounds.min).is_none() {
            return false;
        }
        if bounds.min < 0 {
            negative += bounds.min;
        }
    }
    true
}
fn simplify_divmod(
    a: &SymbolicExpr,
    b: &SymbolicExpr,
    div: bool,
    t: &mut Vec<&'static str>,
) -> Result<SymbolicExpr, SymbolicError> {
    use SymbolicExpr::*;
    let a = simplify_once(a, t)?;
    let b = simplify_once(b, t)?;
    if let (Const(x), Const(y)) = (&a, &b) {
        t.push("fold-divmod");
        return Ok(Const(if div {
            floor_div(*x, *y)?
        } else {
            floor_mod(*x, *y)?
        }));
    }
    if !div && (b == Const(1) || b == Const(-1) || a == b) {
        t.push("mod-zero");
        return Ok(Const(0));
    }
    if div && b == Const(1) {
        return Ok(a);
    }
    if div && b == Const(-1) {
        return Ok(-a);
    }
    if a.bounds()?.constant() == Some(0) {
        return Ok(Const(0));
    }
    Ok(if div {
        FloorDiv(Box::new(a), Box::new(b))
    } else {
        Mod(Box::new(a), Box::new(b))
    })
}
fn simplify_minmax(
    a: &SymbolicExpr,
    b: &SymbolicExpr,
    is_min: bool,
    t: &mut Vec<&'static str>,
) -> Result<SymbolicExpr, SymbolicError> {
    let a = simplify_once(a, t)?;
    let b = simplify_once(b, t)?;
    if a == b {
        return Ok(a);
    }
    if let (Some(x), Some(y)) = (a.bounds()?.constant(), b.bounds()?.constant()) {
        t.push("fold-minmax");
        return Ok(SymbolicExpr::Const(if is_min {
            x.min(y)
        } else {
            x.max(y)
        }));
    }
    Ok(if is_min {
        SymbolicExpr::Min(Box::new(a), Box::new(b))
    } else {
        SymbolicExpr::Max(Box::new(a), Box::new(b))
    })
}
fn simplify_compare(
    a: &SymbolicExpr,
    b: &SymbolicExpr,
    original: &SymbolicExpr,
    t: &mut Vec<&'static str>,
) -> Result<SymbolicExpr, SymbolicError> {
    use SymbolicExpr::*;
    let a = simplify_once(a, t)?;
    let b = simplify_once(b, t)?;
    let x = a.bounds()?;
    let y = b.bounds()?;
    let v = match original {
        Eq(..) => {
            if a == b {
                Some(1)
            } else if x.max < y.min || y.max < x.min {
                Some(0)
            } else {
                None
            }
        }
        Lt(..) => {
            if x.max < y.min {
                Some(1)
            } else if x.min >= y.max {
                Some(0)
            } else {
                None
            }
        }
        Le(..) => {
            if x.max <= y.min {
                Some(1)
            } else if x.min > y.max {
                Some(0)
            } else {
                None
            }
        }
        _ => None,
    };
    Ok(v.map(Const).unwrap_or_else(|| match original {
        Eq(..) => Eq(Box::new(a), Box::new(b)),
        Lt(..) => Lt(Box::new(a), Box::new(b)),
        _ => Le(Box::new(a), Box::new(b)),
    }))
}
fn simplify_logic(
    a: &SymbolicExpr,
    b: &SymbolicExpr,
    is_and: bool,
    t: &mut Vec<&'static str>,
) -> Result<SymbolicExpr, SymbolicError> {
    use SymbolicExpr::*;
    let a = simplify_once(a, t)?;
    let b = simplify_once(b, t)?;
    if a == b {
        return Ok(a);
    }
    for (constant, other) in [(&a, &b), (&b, &a)] {
        if let Some(v) = constant.bounds()?.constant() {
            if boolv(v) == is_and {
                return Ok(other.clone());
            }
            return Ok(Const(v));
        }
    }
    Ok(if a <= b {
        if is_and {
            And(Box::new(a), Box::new(b))
        } else {
            Or(Box::new(a), Box::new(b))
        }
    } else if is_and {
        And(Box::new(b), Box::new(a))
    } else {
        Or(Box::new(b), Box::new(a))
    })
}
fn boolv(x: i64) -> bool {
    x != 0
}
fn bool_possibilities(bounds: Bounds) -> Vec<bool> {
    let mut values = Vec::new();
    if bounds.min <= 0 && bounds.max >= 0 {
        values.push(false);
    }
    if bounds.min != 0 || bounds.max != 0 {
        values.push(true);
    }
    values
}
fn add(a: i64, b: i64) -> Result<i64, SymbolicError> {
    a.checked_add(b)
        .ok_or(SymbolicError::Overflow { op: "addition" })
}
fn mul(a: i64, b: i64) -> Result<i64, SymbolicError> {
    a.checked_mul(b).ok_or(SymbolicError::Overflow {
        op: "multiplication",
    })
}
fn neg(a: i64) -> Result<i64, SymbolicError> {
    a.checked_neg()
        .ok_or(SymbolicError::Overflow { op: "negation" })
}
fn floor_div(a: i64, b: i64) -> Result<i64, SymbolicError> {
    if b == 0 {
        return Err(SymbolicError::DivisionByZero);
    }
    let q = a
        .checked_div(b)
        .ok_or(SymbolicError::Overflow { op: "division" })?;
    let r = a
        .checked_rem(b)
        .ok_or(SymbolicError::Overflow { op: "remainder" })?;
    Ok(if r != 0 && (r > 0) != (b > 0) {
        q - 1
    } else {
        q
    })
}
fn floor_mod(a: i64, b: i64) -> Result<i64, SymbolicError> {
    let q = floor_div(a, b)?;
    a.checked_sub(
        q.checked_mul(b)
            .ok_or(SymbolicError::Overflow { op: "modulo" })?,
    )
    .ok_or(SymbolicError::Overflow { op: "modulo" })
}
fn mul_bounds(a: Bounds, b: Bounds) -> Result<Bounds, SymbolicError> {
    let xs = [
        mul(a.min, b.min)?,
        mul(a.min, b.max)?,
        mul(a.max, b.min)?,
        mul(a.max, b.max)?,
    ];
    Ok(Bounds {
        min: *xs.iter().min().unwrap(),
        max: *xs.iter().max().unwrap(),
    })
}
fn div_bounds(a: Bounds, b: Bounds) -> Result<Bounds, SymbolicError> {
    if b.min <= 0 && b.max >= 0 {
        return Err(SymbolicError::DivisionByZero);
    }
    let mut ds = vec![b.min, b.max];
    if b.min < 0 && b.max > 0 {
        ds.retain(|x| *x != 0)
    }
    let mut vs = Vec::new();
    for n in [a.min, a.max] {
        for d in &ds {
            vs.push(floor_div(n, *d)?)
        }
    }
    Ok(Bounds {
        min: *vs.iter().min().unwrap(),
        max: *vs.iter().max().unwrap(),
    })
}
fn mod_bounds(_: Bounds, b: Bounds) -> Result<Bounds, SymbolicError> {
    if b.min <= 0 && b.max >= 0 {
        return Err(SymbolicError::DivisionByZero);
    }
    if b.min > 0 {
        Ok(Bounds {
            min: 0,
            max: b.max - 1,
        })
    } else if b.max < 0 {
        Ok(Bounds {
            min: b.min + 1,
            max: 0,
        })
    } else {
        unreachable!()
    }
}
fn predicate_bounds(
    a: Bounds,
    b: Bounds,
    p: impl Fn(i64, i64) -> bool,
) -> Result<Bounds, SymbolicError> {
    if a.min == a.max && b.min == b.max {
        let x = p(a.min, b.min) as i64;
        Ok(Bounds { min: x, max: x })
    } else {
        Ok(Bounds { min: 0, max: 1 })
    }
}
impl fmt::Display for SymbolicExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use SymbolicExpr::*;
        match self {
            Const(x) => write!(f, "{x}"),
            Var(v) => write!(f, "{}#{}", v.name, v.id),
            Add(xs) => fmt_join(f, "+", xs),
            Mul(xs) => fmt_join(f, "*", xs),
            Neg(x) => write!(f, "(-{x})"),
            FloorDiv(a, b) => write!(f, "({a}//{b})"),
            Mod(a, b) => write!(f, "({a}%{b})"),
            Min(a, b) => write!(f, "min({a},{b})"),
            Max(a, b) => write!(f, "max({a},{b})"),
            Eq(a, b) => write!(f, "({a}=={b})"),
            Lt(a, b) => write!(f, "({a}<{b})"),
            Le(a, b) => write!(f, "({a}<={b})"),
            And(a, b) => write!(f, "({a}&{b})"),
            Or(a, b) => write!(f, "({a}|{b})"),
            Not(x) => write!(f, "!{x}"),
            Where(c, a, b) => write!(f, "where({c},{a},{b})"),
        }
    }
}
fn fmt_join(f: &mut fmt::Formatter<'_>, op: &str, xs: &[SymbolicExpr]) -> fmt::Result {
    write!(f, "(")?;
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            write!(f, "{op}")?
        }
        write!(f, "{x}")?
    }
    write!(f, ")")
}
