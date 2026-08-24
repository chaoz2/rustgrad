use super::VizGraph;
use std::fmt::Write;

fn quoted(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                write!(out, "\\u{:04x}", c as u32).expect("writing a string cannot fail");
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl VizGraph {
    /// Renders stable Graphviz DOT text without executing an external tool.
    pub fn to_dot(&self) -> String {
        let mut out = String::new();
        writeln!(out, "digraph {} {{", quoted(&self.name)).expect("writing a string cannot fail");
        out.push_str("  graph [rankdir=\"LR\"];\n  node [shape=\"box\"];\n");
        for node in &self.nodes {
            let mut label = node.title.clone();
            label.push_str("\nkind=");
            label.push_str(&node.kind);
            for (key, value) in &node.fields {
                label.push('\n');
                label.push_str(key);
                label.push('=');
                label.push_str(value);
            }
            writeln!(out, "  {} [label={}];", quoted(&node.id), quoted(&label))
                .expect("writing a string cannot fail");
        }
        for edge in &self.edges {
            let label = if edge.label.is_empty() {
                edge.kind.clone()
            } else {
                format!("{}:{}", edge.kind, edge.label)
            };
            writeln!(
                out,
                "  {} -> {} [label={}];",
                quoted(&edge.from),
                quoted(&edge.to),
                quoted(&label)
            )
            .expect("writing a string cannot fail");
        }
        out.push_str("}\n");
        out
    }
}
