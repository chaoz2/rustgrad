use crate::{NodeId, Shape};
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TraceStep {
    pub node: NodeId,
    pub operation: String,
    pub shape: Shape,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompileTrace {
    pub output: NodeId,
    pub steps: Vec<TraceStep>,
}

impl fmt::Display for CompileTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for step in &self.steps {
            writeln!(f, "%{} = {} : {}", step.node, step.operation, step.shape)?;
        }
        write!(f, "return %{}", self.output)
    }
}
