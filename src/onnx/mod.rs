//! Bounded static ONNX protobuf import. This intentionally supports a small,
//! audited inference subset (opset 13, default domain) and never executes code.

use crate::{Backend, CpuBackend, Error, Graph, NodeId, Result, TensorData};
use std::collections::{BTreeMap, HashMap};

const MAX_BYTES: usize = 32 * 1024 * 1024;
fn bad(s: impl Into<String>) -> Error {
    Error::ModelIo { reason: s.into() }
}

/// A static ONNX graph lowered into RustGrad's existing CPU graph boundary.
pub struct OnnxModel {
    graph: Graph,
    inputs: BTreeMap<String, NodeId>,
    outputs: BTreeMap<String, NodeId>,
}
impl OnnxModel {
    pub fn inputs(&self) -> impl Iterator<Item = &str> {
        self.inputs.keys().map(String::as_str)
    }
    pub fn outputs(&self) -> impl Iterator<Item = &str> {
        self.outputs.keys().map(String::as_str)
    }
    pub fn run(&self, inputs: HashMap<String, TensorData>) -> Result<BTreeMap<String, TensorData>> {
        let cpu = CpuBackend;
        self.outputs
            .iter()
            .map(|(name, &node)| Ok((name.clone(), cpu.execute(&self.graph, node, &inputs)?)))
            .collect()
    }
}

/// Parses and lowers a static opset-13 ONNX MLP subset: Identity, Add,
/// MatMul, and Relu, with constant initializers and named concrete inputs.
pub fn import_onnx(bytes: &[u8]) -> Result<OnnxModel> {
    if bytes.len() > MAX_BYTES {
        return Err(bad("ONNX model exceeds byte limit"));
    }
    let m = Msg::new(bytes);
    let graph_bytes = one_bytes(&m, 7, "graph")?;
    let opsets = m.bytes(8)?;
    if opsets.len() != 1 {
        return Err(bad("ONNX requires exactly one opset import"));
    }
    let op = Msg::new(opsets[0]);
    if !op.string(1)?.unwrap_or("").is_empty() || one_varint(&op, 2, "opset version")? != 13 {
        return Err(bad("only default-domain ONNX opset 13 is supported"));
    }
    let g = Msg::new(graph_bytes);
    let mut graph = Graph::new();
    let mut constants = BTreeMap::new();
    let mut nodes = BTreeMap::new();
    let mut initializers = BTreeMap::new();
    for t in g.bytes(5)? {
        let (name, data) = tensor(Msg::new(t))?;
        constants.insert(name.clone(), data.clone());
        if initializers
            .insert(name.clone(), graph.constant(data))
            .is_some()
        {
            return Err(bad("duplicate ONNX initializer"));
        }
    }
    let mut inputs = BTreeMap::new();
    for v in g.bytes(11)? {
        let (name, shape, dtype) = value_info(Msg::new(v))?;
        if initializers.contains_key(&name) {
            continue;
        }
        if inputs
            .insert(name.clone(), graph.input_dtype(name.clone(), shape, dtype))
            .is_some()
        {
            return Err(bad("duplicate ONNX input"));
        }
    }
    nodes.extend(initializers.iter().map(|(k, &v)| (k.clone(), v)));
    nodes.extend(inputs.iter().map(|(k, &v)| (k.clone(), v)));
    for n in g.bytes(1)? {
        lower(&mut graph, Msg::new(n), &mut nodes, &mut constants)?;
    }
    let mut outputs = BTreeMap::new();
    for v in g.bytes(12)? {
        let (name, _, _) = value_info(Msg::new(v))?;
        let node = *nodes
            .get(&name)
            .ok_or_else(|| bad(format!("unknown ONNX output {name:?}")))?;
        if outputs.insert(name, node).is_some() {
            return Err(bad("duplicate ONNX output"));
        }
    }
    if outputs.is_empty() {
        return Err(bad("ONNX graph has no outputs"));
    }
    Ok(OnnxModel {
        graph,
        inputs,
        outputs,
    })
}

mod lower;
use lower::lower;

mod wire;
use wire::{Msg, one_bytes, one_varint};
mod tensor;
use tensor::tensor;
mod schema;
use schema::value_info;

#[cfg(test)]
mod tests;
