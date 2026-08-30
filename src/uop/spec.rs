#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OpFamily {
    Literal,
    Definition,
    Control,
    CoreAlu,
    GraphAlu,
    Materialized,
    Reduction,
    Conversion,
    Memory,
    Effect,
}

impl OpFamily {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Literal => "literal",
            Self::Definition => "definition",
            Self::Control => "control",
            Self::CoreAlu => "core_alu",
            Self::GraphAlu => "graph_alu",
            Self::Materialized => "materialized",
            Self::Reduction => "reduction",
            Self::Conversion => "conversion",
            Self::Memory => "memory",
            Self::Effect => "effect",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OpArity {
    Exact(u8),
    AtLeastOne,
    Any,
}

impl OpArity {
    pub(crate) fn accepts(self, actual: usize) -> bool {
        match self {
            Self::Exact(expected) => actual == usize::from(expected),
            Self::AtLeastOne => actual != 0,
            Self::Any => true,
        }
    }

    pub(crate) fn expectation(self) -> &'static str {
        match self {
            Self::Exact(_) => "exact source count",
            Self::AtLeastOne => "one or more sources",
            Self::Any => "any source count",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OpEffect {
    Pure,
    ControlBoundary,
    Effectful,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OpSignature {
    pub(crate) family: OpFamily,
    pub(crate) arity: OpArity,
    pub(crate) effect: OpEffect,
}

/// Static interface implemented by operation families. The `define_uops!`
/// registry selects one of these family values without trait objects.
pub(crate) trait OpSpec: Copy {
    fn family(self) -> OpFamily;
    fn arity(self) -> OpArity;
    fn effect(self) -> OpEffect {
        OpEffect::Pure
    }
    fn signature(self) -> OpSignature {
        OpSignature {
            family: self.family(),
            arity: self.arity(),
            effect: self.effect(),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Literal;

impl OpSpec for Literal {
    fn family(self) -> OpFamily {
        OpFamily::Literal
    }
    fn arity(self) -> OpArity {
        OpArity::Exact(0)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Definition {
    Variable,
    Address,
    Name,
}

impl OpSpec for Definition {
    fn family(self) -> OpFamily {
        OpFamily::Definition
    }
    fn arity(self) -> OpArity {
        OpArity::Exact(0)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Control {
    Range,
    Boundary,
}

impl OpSpec for Control {
    fn family(self) -> OpFamily {
        OpFamily::Control
    }
    fn arity(self) -> OpArity {
        OpArity::Exact(1)
    }
    fn effect(self) -> OpEffect {
        match self {
            Self::Range => OpEffect::Pure,
            Self::Boundary => OpEffect::ControlBoundary,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum CoreAlu {
    Unary,
    Binary,
    Ternary,
}

impl OpSpec for CoreAlu {
    fn family(self) -> OpFamily {
        OpFamily::CoreAlu
    }
    fn arity(self) -> OpArity {
        match self {
            Self::Unary => OpArity::Exact(1),
            Self::Binary => OpArity::Exact(2),
            Self::Ternary => OpArity::Exact(3),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum GraphAlu {
    Unary,
    Binary,
    Logical(crate::LogicalOp),
}

impl OpSpec for GraphAlu {
    fn family(self) -> OpFamily {
        OpFamily::GraphAlu
    }
    fn arity(self) -> OpArity {
        match self {
            Self::Unary => OpArity::Exact(1),
            Self::Binary => OpArity::Exact(2),
            Self::Logical(crate::LogicalOp::Not) => OpArity::Exact(1),
            Self::Logical(crate::LogicalOp::And | crate::LogicalOp::Or) => OpArity::Exact(2),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Materialized {
    Matmul,
    Conv2d,
    Movement,
    Random,
    PrefixScan,
    Sort,
    TensorGuard,
}

impl OpSpec for Materialized {
    fn family(self) -> OpFamily {
        OpFamily::Materialized
    }
    fn arity(self) -> OpArity {
        OpArity::Exact(0)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Reduction {
    Init,
    Accumulate,
    Finalize,
}

impl OpSpec for Reduction {
    fn family(self) -> OpFamily {
        OpFamily::Reduction
    }
    fn arity(self) -> OpArity {
        match self {
            Self::Init => OpArity::Exact(0),
            Self::Accumulate => OpArity::Exact(2),
            Self::Finalize => OpArity::Exact(1),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Conversion {
    Cast,
    Bitcast,
    Vectorize,
    Gep,
}

impl OpSpec for Conversion {
    fn family(self) -> OpFamily {
        OpFamily::Conversion
    }
    fn arity(self) -> OpArity {
        match self {
            Self::Cast | Self::Bitcast | Self::Gep => OpArity::Exact(1),
            Self::Vectorize => OpArity::AtLeastOne,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Memory {
    Index,
    Load,
    Store,
}

impl OpSpec for Memory {
    fn family(self) -> OpFamily {
        OpFamily::Memory
    }
    fn arity(self) -> OpArity {
        match self {
            Self::Index | Self::Store => OpArity::Exact(2),
            Self::Load => OpArity::Exact(1),
        }
    }
    fn effect(self) -> OpEffect {
        match self {
            Self::Store => OpEffect::Effectful,
            Self::Index | Self::Load => OpEffect::Pure,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Effect {
    Store,
    After,
    Barrier,
    Sink,
}

impl OpSpec for Effect {
    fn family(self) -> OpFamily {
        OpFamily::Effect
    }
    fn arity(self) -> OpArity {
        match self {
            Self::Store | Self::Barrier => OpArity::Exact(0),
            Self::After => OpArity::Exact(1),
            Self::Sink => OpArity::Any,
        }
    }
    fn effect(self) -> OpEffect {
        OpEffect::Effectful
    }
}
