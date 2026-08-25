//! Bounded static ONNX protobuf import and CPU execution (opset 13, default
//! domain). Supported static dense operators lower into the existing graph;
//! parsing never executes code or loads external data.

use crate::{Backend, CpuBackend, DType, Error, Graph, NodeId, Result, Shape, TensorData};
use std::collections::{BTreeMap, HashMap};

const MAX_BYTES: usize = 32 * 1024 * 1024;
fn bad(s: impl Into<String>) -> Error {
    Error::ModelIo { reason: s.into() }
}

/// A concrete named ONNX graph value contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OnnxValueInfo {
    name: String,
    shape: Shape,
    dtype: DType,
}
impl OnnxValueInfo {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// A static ONNX graph lowered into RustGrad's existing CPU graph boundary.
pub struct OnnxModel {
    graph: Graph,
    inputs: BTreeMap<String, NodeId>,
    input_info: BTreeMap<String, OnnxValueInfo>,
    outputs: BTreeMap<String, NodeId>,
}
impl OnnxModel {
    pub fn inputs(&self) -> impl Iterator<Item = &str> {
        self.inputs.keys().map(String::as_str)
    }
    pub fn outputs(&self) -> impl Iterator<Item = &str> {
        self.outputs.keys().map(String::as_str)
    }
    /// Concrete static input schemas in deterministic name order.
    pub fn input_info(&self) -> impl Iterator<Item = &OnnxValueInfo> {
        self.input_info.values()
    }
    pub fn run(&self, inputs: HashMap<String, TensorData>) -> Result<BTreeMap<String, TensorData>> {
        self.run_named(&inputs.into_iter().collect())
    }
    /// Preflights exact names, shapes, and dtypes, then executes through the
    /// existing CPU graph boundary.
    pub fn run_named(
        &self,
        inputs: &BTreeMap<String, TensorData>,
    ) -> Result<BTreeMap<String, TensorData>> {
        self.validate_named_inputs(inputs)?;
        let execution_inputs = inputs.clone().into_iter().collect::<HashMap<_, _>>();
        let cpu = CpuBackend;
        self.outputs
            .iter()
            .map(|(name, &node)| {
                Ok((
                    name.clone(),
                    cpu.execute(&self.graph, node, &execution_inputs)?,
                ))
            })
            .collect()
    }

    pub(super) fn validate_named_inputs(
        &self,
        inputs: &BTreeMap<String, TensorData>,
    ) -> Result<()> {
        for name in self.inputs.keys() {
            if !inputs.contains_key(name) {
                return Err(bad(format!("missing ONNX input {name:?}")));
            }
        }
        for name in inputs.keys() {
            if !self.inputs.contains_key(name) {
                return Err(bad(format!("unexpected ONNX input {name:?}")));
            }
        }
        for (name, info) in &self.input_info {
            let input = &inputs[name];
            if input.shape() != &info.shape {
                return Err(bad(format!("ONNX input {name:?} shape mismatch")));
            }
            if input.dtype() != info.dtype {
                return Err(bad(format!("ONNX input {name:?} dtype mismatch")));
            }
        }
        Ok(())
    }
}

/// Parses and lowers the documented static dense opset-13 default-domain
/// subset, with concrete named inputs and no external data.
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
    let mut input_info = BTreeMap::new();
    for v in g.bytes(11)? {
        let (name, shape, dtype) = value_info(Msg::new(v))?;
        if initializers.contains_key(&name) {
            continue;
        }
        let info = OnnxValueInfo {
            name: name.clone(),
            shape: shape.clone(),
            dtype,
        };
        if inputs
            .insert(name.clone(), graph.input_dtype(name.clone(), shape, dtype))
            .is_some()
        {
            return Err(bad("duplicate ONNX input"));
        }
        input_info.insert(name, info);
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
        input_info,
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
mod file;
pub use file::{
    NamedPaths, NamedPathsError, OnnxFileError, OnnxReadLimits, OnnxWorkflowError,
    OnnxWorkflowLimits, load_onnx_file, load_onnx_file_with_limits, run_onnx_files,
    run_onnx_files_native,
};
mod native;
pub use native::{NativeOnnxInferenceResult, NativeOnnxInferenceTrace};

#[cfg(test)]
mod tests;
