//! Backend-neutral universal operations. This layer is below the tensor graph
//! and above future scheduling/rendering; it deliberately does not execute.
use crate::{DType, SymbolicExpr};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    sync::Arc,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AddressSpace {
    Global,
    Local,
    Register,
}
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct UType {
    pub scalar: DType,
    pub lanes: u16,
}
impl UType {
    pub fn scalar(scalar: DType) -> Self {
        Self { scalar, lanes: 1 }
    }
    pub fn vector(scalar: DType, lanes: u16) -> Result<Self, UOpError> {
        if lanes == 0 {
            Err(UOpError::InvalidLaneWidth)
        } else {
            Ok(Self { scalar, lanes })
        }
    }
    pub fn is_bool(self) -> bool {
        self.scalar == DType::Bool
    }
}
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Unary {
    Neg,
    Not,
    Abs,
}
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Binary {
    Add,
    Sub,
    Mul,
    FloorDiv,
    Mod,
    Min,
    Max,
    Eq,
    Lt,
    Le,
    And,
    Or,
}
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Ternary {
    Where,
}
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum UOpKind {
    Const,
    VConst,
    DefineVar,
    DefineGlobal,
    DefineLocal,
    DefineRegister,
    Special,
    Range,
    EndRange,
    If,
    EndIf,
    Unary(Unary),
    Binary(Binary),
    Ternary(Ternary),
    Cast,
    Bitcast,
    Vectorize,
    Gep,
    Index,
    Load,
    Store,
    Barrier,
    Sink,
}
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum UArg {
    None,
    Int(i64),
    Name(String),
    Variable {
        name: String,
        bounds: SymbolicExpr,
    },
    Address {
        space: AddressSpace,
        name: String,
        element: UType,
    },
    RangeAxis(u32),
    GepLane(u16),
}
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct UOpNode {
    kind: UOpKind,
    ty: Option<UType>,
    sources: Vec<UOp>,
    arg: UArg,
}
/// Immutable and structurally hashable. Cloning preserves DAG sharing.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct UOp(Arc<UOpNode>);
impl UOp {
    pub fn new(kind: UOpKind, ty: Option<UType>, sources: Vec<UOp>, arg: UArg) -> Self {
        Self(Arc::new(UOpNode {
            kind,
            ty,
            sources,
            arg,
        }))
    }
    pub fn kind(&self) -> &UOpKind {
        &self.0.kind
    }
    pub fn ty(&self) -> Option<UType> {
        self.0.ty
    }
    pub fn sources(&self) -> &[UOp] {
        &self.0.sources
    }
    pub fn arg(&self) -> &UArg {
        &self.0.arg
    }
    pub fn constant(value: i64, ty: UType) -> Self {
        Self::new(UOpKind::Const, Some(ty), vec![], UArg::Int(value))
    }
    pub fn unary(op: Unary, x: UOp) -> Self {
        Self::new(UOpKind::Unary(op), x.ty(), vec![x], UArg::None)
    }
    pub fn binary(op: Binary, a: UOp, b: UOp) -> Self {
        let ty = if matches!(op, Binary::Eq | Binary::Lt | Binary::Le) {
            Some(UType::scalar(DType::Bool))
        } else {
            a.ty()
        };
        Self::new(UOpKind::Binary(op), ty, vec![a, b], UArg::None)
    }
    pub fn cast(x: UOp, to: UType) -> Self {
        Self::new(UOpKind::Cast, Some(to), vec![x], UArg::None)
    }
    pub fn sink(sources: Vec<UOp>) -> Self {
        Self::new(UOpKind::Sink, None, sources, UArg::None)
    }
    pub fn topological(&self) -> Result<Vec<UOp>, UOpError> {
        fn visit(
            n: &UOp,
            seen: &mut BTreeSet<UOp>,
            active: &mut BTreeSet<UOp>,
            out: &mut Vec<UOp>,
        ) -> Result<(), UOpError> {
            if seen.contains(n) {
                return Ok(());
            }
            if !active.insert(n.clone()) {
                return Err(UOpError::Cycle);
            }
            for s in n.sources() {
                visit(s, seen, active, out)?
            }
            active.remove(n);
            seen.insert(n.clone());
            out.push(n.clone());
            Ok(())
        }
        let mut out = vec![];
        visit(self, &mut BTreeSet::new(), &mut BTreeSet::new(), &mut out)?;
        Ok(out)
    }
    pub fn is_pure(&self) -> bool {
        !matches!(
            self.kind(),
            UOpKind::Store
                | UOpKind::Barrier
                | UOpKind::Sink
                | UOpKind::EndRange
                | UOpKind::If
                | UOpKind::EndIf
        )
    }
    pub fn validate(&self) -> Result<(), UOpError> {
        let nodes = self.topological()?;
        let mut ranges = BTreeSet::new();
        let mut ifs = Vec::new();
        for n in nodes {
            validate_one(&n, &mut ranges, &mut ifs)?
        }
        if !ifs.is_empty() || !ranges.is_empty() {
            return Err(UOpError::UnclosedControl);
        }
        Ok(())
    }
}
impl fmt::Display for UOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.kind())?;
        if let Some(t) = self.ty() {
            write!(f, ":{:?}x{}", t.scalar, t.lanes)?
        }
        if !matches!(self.arg(), UArg::None) {
            write!(f, "({:?})", self.arg())?
        }
        if !self.sources().is_empty() {
            write!(f, "[")?;
            for (i, s) in self.sources().iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?
                }
                write!(f, "{s}")?
            }
            write!(f, "]")?
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UOpError {
    InvalidArity {
        kind: UOpKind,
        expected: &'static str,
        actual: usize,
    },
    InvalidDType,
    InvalidLaneWidth,
    InvalidArgument,
    InvalidIndex,
    Cycle,
    UseBeforeDefinition,
    ControlMismatch,
    UnclosedControl,
    EffectRewrite,
}
impl fmt::Display for UOpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UOp validation error: {self:?}")
    }
}
impl std::error::Error for UOpError {}
fn exact(n: &UOp, count: usize) -> Result<(), UOpError> {
    if n.sources().len() == count {
        Ok(())
    } else {
        Err(UOpError::InvalidArity {
            kind: n.kind().clone(),
            expected: "exact source count",
            actual: n.sources().len(),
        })
    }
}
fn same(n: &UOp) -> bool {
    n.sources().iter().all(|s| s.ty() == n.ty())
}
fn validate_one(n: &UOp, ranges: &mut BTreeSet<u32>, ifs: &mut Vec<UOp>) -> Result<(), UOpError> {
    use UOpKind::*;
    match n.kind() {
        Const => {
            exact(n, 0)?;
            if !matches!(n.arg(), UArg::Int(_)) {
                return Err(UOpError::InvalidArgument);
            }
        }
        VConst => {
            if n.ty().is_none() {
                return Err(UOpError::InvalidDType);
            }
        }
        DefineVar => {
            exact(n, 0)?;
            if !matches!(n.arg(), UArg::Variable { .. }) {
                return Err(UOpError::InvalidArgument);
            }
        }
        DefineGlobal | DefineLocal | DefineRegister => {
            exact(n, 0)?;
            if !matches!(n.arg(), UArg::Address { .. }) {
                return Err(UOpError::InvalidArgument);
            }
        }
        Special => {
            exact(n, 0)?;
            if !matches!(n.arg(), UArg::Name(_)) {
                return Err(UOpError::InvalidArgument);
            }
        }
        Range => {
            exact(n, 1)?;
            if !n.sources()[0].ty().is_some_and(|t| t.scalar.is_integer()) {
                return Err(UOpError::InvalidDType);
            }
            let UArg::RangeAxis(axis) = n.arg() else {
                return Err(UOpError::InvalidArgument);
            };
            ranges.insert(*axis);
        }
        EndRange => {
            exact(n, 1)?;
            let UArg::RangeAxis(axis) = n.sources()[0].arg() else {
                return Err(UOpError::ControlMismatch);
            };
            if !ranges.remove(axis) {
                return Err(UOpError::ControlMismatch);
            }
        }
        If => {
            exact(n, 1)?;
            if !n.sources()[0].ty().is_some_and(UType::is_bool) {
                return Err(UOpError::InvalidDType);
            }
            ifs.push(n.clone())
        }
        EndIf => {
            exact(n, 1)?;
            if !matches!(n.sources()[0].kind(), If) || ifs.pop().as_ref() != Some(&n.sources()[0]) {
                return Err(UOpError::ControlMismatch);
            }
        }
        Unary(_) => {
            exact(n, 1)?;
            if !same(n) {
                return Err(UOpError::InvalidDType);
            }
        }
        Binary(op) => {
            exact(n, 2)?;
            if !matches!(
                op,
                crate::uop::Binary::Eq | crate::uop::Binary::Lt | crate::uop::Binary::Le
            ) && !same(n)
            {
                return Err(UOpError::InvalidDType);
            }
        }
        Ternary(crate::uop::Ternary::Where) => {
            exact(n, 3)?;
            if !n.sources()[0].ty().is_some_and(UType::is_bool)
                || n.sources()[1].ty() != n.sources()[2].ty()
                || n.ty() != n.sources()[1].ty()
            {
                return Err(UOpError::InvalidDType);
            }
        }
        Cast | Bitcast => {
            exact(n, 1)?;
            if n.ty().is_none() {
                return Err(UOpError::InvalidDType);
            }
        }
        Vectorize => {
            if n.sources().is_empty() || !same(n) {
                return Err(UOpError::InvalidDType);
            }
        }
        Gep => {
            exact(n, 1)?;
            if !matches!(n.arg(), UArg::GepLane(_)) {
                return Err(UOpError::InvalidArgument);
            }
        }
        Index => {
            exact(n, 2)?;
            if !matches!(
                n.sources()[0].kind(),
                DefineGlobal | DefineLocal | DefineRegister
            ) || !n.sources()[1].ty().is_some_and(|t| t.scalar.is_integer())
            {
                return Err(UOpError::InvalidIndex);
            }
        }
        Load => {
            exact(n, 1)?;
            if !matches!(n.sources()[0].kind(), Index) {
                return Err(UOpError::InvalidIndex);
            }
        }
        Store => {
            exact(n, 2)?;
            if !matches!(n.sources()[0].kind(), Index) {
                return Err(UOpError::InvalidIndex);
            }
        }
        Barrier => exact(n, 0)?,
        Sink => {
            if n.ty().is_some() {
                return Err(UOpError::InvalidDType);
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct Captures(BTreeMap<String, UOp>);
impl Captures {
    pub fn get(&self, name: &str) -> Option<&UOp> {
        self.0.get(name)
    }
}
#[derive(Clone, Debug)]
pub struct UPat {
    kinds: Option<BTreeSet<UOpKind>>,
    ty: Option<UType>,
    arg: Option<UArg>,
    sources: Option<Vec<UPat>>,
    name: Option<String>,
    any: bool,
}
impl UPat {
    pub fn any() -> Self {
        Self {
            kinds: None,
            ty: None,
            arg: None,
            sources: None,
            name: None,
            any: true,
        }
    }
    pub fn op(kind: UOpKind) -> Self {
        let mut x = Self::any();
        x.kinds = Some([kind].into());
        x.any = false;
        x
    }
    pub fn ops(kinds: impl IntoIterator<Item = UOpKind>) -> Self {
        let mut x = Self::any();
        x.kinds = Some(kinds.into_iter().collect());
        x.any = false;
        x
    }
    pub fn dtype(mut self, ty: UType) -> Self {
        self.ty = Some(ty);
        self
    }
    pub fn arg(mut self, arg: UArg) -> Self {
        self.arg = Some(arg);
        self
    }
    pub fn sources(mut self, s: Vec<UPat>) -> Self {
        self.sources = Some(s);
        self
    }
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    pub fn matches(&self, node: &UOp) -> Option<Captures> {
        let mut c = Captures::default();
        self.match_into(node, &mut c).then_some(c)
    }
    fn match_into(&self, n: &UOp, c: &mut Captures) -> bool {
        if !self.any && self.kinds.as_ref().is_some_and(|x| !x.contains(n.kind())) {
            return false;
        }
        if self.ty.is_some_and(|x| n.ty() != Some(x))
            || self.arg.as_ref().is_some_and(|x| x != n.arg())
        {
            return false;
        }
        if let Some(ps) = &self.sources {
            if ps.len() != n.sources().len() {
                return false;
            }
            for (p, s) in ps.iter().zip(n.sources()) {
                if !p.match_into(s, c) {
                    return false;
                }
            }
        }
        if let Some(name) = &self.name {
            if let Some(old) = c.0.get(name) {
                if old != n {
                    return false;
                }
            } else {
                c.0.insert(name.clone(), n.clone());
            }
        }
        true
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Walk {
    BottomUp,
    TopDown,
}
pub type RewriteFn = fn(&Captures, &UOp) -> Option<UOp>;
#[derive(Clone)]
pub struct RewriteRule {
    pub name: &'static str,
    pub priority: i32,
    pub pattern: UPat,
    pub apply: RewriteFn,
}
#[derive(Clone, Debug)]
pub struct RewriteTrace {
    pub rules: Vec<&'static str>,
}
pub fn rewrite(
    root: &UOp,
    rules: &mut [RewriteRule],
    walk: Walk,
) -> Result<(UOp, RewriteTrace), UOpError> {
    rules.sort_by_key(|r| r.priority);
    let mut trace = RewriteTrace { rules: vec![] };
    let mut memo = BTreeMap::new();
    fn go(
        n: &UOp,
        r: &[RewriteRule],
        w: Walk,
        m: &mut BTreeMap<UOp, UOp>,
        t: &mut RewriteTrace,
    ) -> Result<UOp, UOpError> {
        if let Some(x) = m.get(n) {
            return Ok(x.clone());
        }
        let mut x = n.clone();
        if w == Walk::BottomUp {
            x = UOp::new(
                x.kind().clone(),
                x.ty(),
                x.sources()
                    .iter()
                    .map(|s| go(s, r, w, m, t))
                    .collect::<Result<_, _>>()?,
                x.arg().clone(),
            )
        }
        for rule in r {
            if let Some(c) = rule.pattern.matches(&x)
                && let Some(next) = (rule.apply)(&c, &x)
            {
                if !x.is_pure() || !next.is_pure() {
                    return Err(UOpError::EffectRewrite);
                }
                t.rules.push(rule.name);
                x = next;
                break;
            }
        }
        if w == Walk::TopDown {
            x = UOp::new(
                x.kind().clone(),
                x.ty(),
                x.sources()
                    .iter()
                    .map(|s| go(s, r, w, m, t))
                    .collect::<Result<_, _>>()?,
                x.arg().clone(),
            )
        }
        m.insert(n.clone(), x.clone());
        Ok(x)
    }
    let x = go(root, rules, walk, &mut memo, &mut trace)?;
    Ok((x, trace))
}
pub fn builtin_rules() -> Vec<RewriteRule> {
    vec![
        RewriteRule {
            name: "add-zero",
            priority: 0,
            pattern: UPat::op(UOpKind::Binary(Binary::Add)).sources(vec![
                UPat::any().named("x"),
                UPat::op(UOpKind::Const).arg(UArg::Int(0)),
            ]),
            apply: |c, _| c.get("x").cloned(),
        },
        RewriteRule {
            name: "cast-same",
            priority: 1,
            pattern: UPat::op(UOpKind::Cast).sources(vec![UPat::any().named("x")]),
            apply: |c, n| c.get("x").filter(|x| x.ty() == n.ty()).cloned(),
        },
        RewriteRule {
            name: "where-same",
            priority: 2,
            pattern: UPat::op(UOpKind::Ternary(Ternary::Where)).sources(vec![
                UPat::any(),
                UPat::any().named("x"),
                UPat::any().named("x"),
            ]),
            apply: |c, _| c.get("x").cloned(),
        },
    ]
}

/// Lowers a scalar-expression pilot from the high-level graph. It is
/// inspectable metadata only; execution remains with the CPU backend.
pub fn lower_graph_scalar(graph: &crate::Graph, output: crate::NodeId) -> Result<UOp, UOpError> {
    fn lower(
        graph: &crate::Graph,
        id: crate::NodeId,
        memo: &mut HashMap<crate::NodeId, UOp>,
    ) -> Result<UOp, UOpError> {
        if let Some(x) = memo.get(&id) {
            return Ok(x.clone());
        }
        if graph
            .shape(id)
            .map_err(|_| UOpError::UseBeforeDefinition)?
            .numel()
            .map_err(|_| UOpError::InvalidArgument)?
            != 1
        {
            return Err(UOpError::InvalidArgument);
        }
        let ty = UType::scalar(graph.dtype(id).map_err(|_| UOpError::UseBeforeDefinition)?);
        let x = match graph.op(id).map_err(|_| UOpError::UseBeforeDefinition)? {
            crate::Op::Input { name } => UOp::new(
                UOpKind::DefineVar,
                Some(ty),
                vec![],
                UArg::Variable {
                    name: name.clone(),
                    bounds: SymbolicExpr::constant(0),
                },
            ),
            crate::Op::Constant(data) => UOp::constant(data.scalar_at(0).as_i64(), ty),
            crate::Op::Cast { input, .. } => UOp::cast(lower(graph, *input, memo)?, ty),
            crate::Op::Unary { op, input } => {
                let u = match op {
                    crate::UnaryOp::Neg => Unary::Neg,
                    crate::UnaryOp::Abs => Unary::Abs,
                    _ => return Err(UOpError::InvalidArgument),
                };
                UOp::unary(u, lower(graph, *input, memo)?)
            }
            crate::Op::Binary { op, lhs, rhs } => {
                let b = match op {
                    crate::BinaryOp::Add => Binary::Add,
                    crate::BinaryOp::Sub => Binary::Sub,
                    crate::BinaryOp::Mul => Binary::Mul,
                    crate::BinaryOp::FloorDiv => Binary::FloorDiv,
                    crate::BinaryOp::Mod => Binary::Mod,
                    crate::BinaryOp::Maximum => Binary::Max,
                    crate::BinaryOp::Minimum => Binary::Min,
                    _ => return Err(UOpError::InvalidArgument),
                };
                UOp::binary(b, lower(graph, *lhs, memo)?, lower(graph, *rhs, memo)?)
            }
            crate::Op::Compare { op, lhs, rhs } => {
                let b = match op {
                    crate::CompareOp::Eq => Binary::Eq,
                    crate::CompareOp::Lt => Binary::Lt,
                    crate::CompareOp::Le => Binary::Le,
                    _ => return Err(UOpError::InvalidArgument),
                };
                UOp::binary(b, lower(graph, *lhs, memo)?, lower(graph, *rhs, memo)?)
            }
            crate::Op::Select {
                condition,
                on_true,
                on_false,
            } => UOp::new(
                UOpKind::Ternary(Ternary::Where),
                Some(ty),
                vec![
                    lower(graph, *condition, memo)?,
                    lower(graph, *on_true, memo)?,
                    lower(graph, *on_false, memo)?,
                ],
                UArg::None,
            ),
            _ => return Err(UOpError::InvalidArgument),
        };
        memo.insert(id, x.clone());
        Ok(x)
    }
    lower(graph, output, &mut HashMap::new())
}
