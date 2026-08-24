//! Deterministic, dependency-free inspection models for compiler artifacts.
//!
//! Builders consume typed RustGrad IR and compiler metadata. They never parse
//! `Debug` output, retain runtime objects, or expose process-specific pointers.
//! The normalized [`VizGraph`] is useful directly and renders stable Graphviz
//! DOT through [`VizGraph::to_dot`] without invoking Graphviz.
//!
//! ```
//! use rustgrad::{Graph, graph_viz};
//!
//! let mut graph = Graph::new();
//! let x = graph.input("x", [4]);
//! let y = graph.relu(x)?;
//! let dot = graph_viz(&graph, &[y])?.to_dot();
//! assert!(dot.contains("operator=relu"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod dot;
mod graph;
mod kernel;
mod model;
mod schedule;
mod uop;

pub use graph::graph_viz;
pub use kernel::{linear_viz, memory_space_viz, vector_viz};
pub use model::{VizEdge, VizError, VizGraph, VizNode};
pub use schedule::{captured_schedule_viz, schedule_viz};
pub use uop::uop_viz;

#[cfg(test)]
mod tests;
