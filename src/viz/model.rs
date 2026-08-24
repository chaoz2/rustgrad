use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// One normalized visualization node. Fields render in lexical key order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VizNode {
    pub(super) id: String,
    pub(super) kind: String,
    pub(super) title: String,
    pub(super) fields: BTreeMap<String, String>,
}

impl VizNode {
    pub fn new(id: impl Into<String>, kind: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            title: title.into(),
            fields: BTreeMap::new(),
        }
    }

    pub fn field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn kind(&self) -> &str {
        &self.kind
    }
    pub fn title(&self) -> &str {
        &self.title
    }
    pub fn fields(&self) -> &BTreeMap<String, String> {
        &self.fields
    }
}

/// One directed dependency edge. Parallel edges are allowed when labels differ.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct VizEdge {
    pub(super) from: String,
    pub(super) to: String,
    pub(super) kind: String,
    pub(super) label: String,
}

impl VizEdge {
    pub fn new(
        from: impl Into<String>,
        to: impl Into<String>,
        kind: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            kind: kind.into(),
            label: label.into(),
        }
    }

    pub fn from(&self) -> &str {
        &self.from
    }
    pub fn to(&self) -> &str {
        &self.to
    }
    pub fn kind(&self) -> &str {
        &self.kind
    }
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// A validated, canonically ordered visualization model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VizGraph {
    pub(super) name: String,
    pub(super) nodes: Vec<VizNode>,
    pub(super) edges: Vec<VizEdge>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VizError {
    EmptyName,
    EmptyNodeId,
    DuplicateNode(String),
    MissingEndpoint { edge: String, node: String },
    DuplicateEdge(String),
    InvalidGraphNode(usize),
    UnsupportedGraphOp(String),
    InvalidUOp(String),
    InvalidSchedule(String),
}

impl fmt::Display for VizError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => write!(f, "visualization name is empty"),
            Self::EmptyNodeId => write!(f, "visualization node id is empty"),
            Self::DuplicateNode(id) => write!(f, "duplicate visualization node {id}"),
            Self::MissingEndpoint { edge, node } => {
                write!(
                    f,
                    "visualization edge {edge} references missing node {node}"
                )
            }
            Self::DuplicateEdge(edge) => write!(f, "duplicate visualization edge {edge}"),
            Self::InvalidGraphNode(node) => write!(f, "invalid graph node {node}"),
            Self::UnsupportedGraphOp(op) => write!(f, "unsupported graph visualization op {op}"),
            Self::InvalidUOp(reason) => write!(f, "invalid UOp visualization input: {reason}"),
            Self::InvalidSchedule(reason) => {
                write!(f, "invalid schedule visualization input: {reason}")
            }
        }
    }
}

impl std::error::Error for VizError {}

impl VizGraph {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn nodes(&self) -> &[VizNode] {
        &self.nodes
    }
    pub fn edges(&self) -> &[VizEdge] {
        &self.edges
    }

    /// Validates and canonicalizes an inspection model. Input order does not
    /// affect the stored model or rendered DOT.
    pub fn try_new(
        name: impl Into<String>,
        mut nodes: Vec<VizNode>,
        mut edges: Vec<VizEdge>,
    ) -> Result<Self, VizError> {
        let name = name.into();
        if name.is_empty() {
            return Err(VizError::EmptyName);
        }
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        let mut ids = BTreeSet::new();
        for node in &nodes {
            if node.id.is_empty() {
                return Err(VizError::EmptyNodeId);
            }
            if !ids.insert(node.id.clone()) {
                return Err(VizError::DuplicateNode(node.id.clone()));
            }
        }
        edges.sort();
        let mut unique = BTreeSet::new();
        for edge in &edges {
            let signature = format!("{}->{}:{}:{}", edge.from, edge.to, edge.kind, edge.label);
            if !ids.contains(&edge.from) {
                return Err(VizError::MissingEndpoint {
                    edge: signature,
                    node: edge.from.clone(),
                });
            }
            if !ids.contains(&edge.to) {
                return Err(VizError::MissingEndpoint {
                    edge: signature,
                    node: edge.to.clone(),
                });
            }
            if !unique.insert(edge.clone()) {
                return Err(VizError::DuplicateEdge(signature));
            }
        }
        Ok(Self { name, nodes, edges })
    }
}
