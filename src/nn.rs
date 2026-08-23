//! Explicit module traversal and versioned graph-input parameters.
//!
//! A [`Parameter`] is an input leaf belonging to one [`Graph`].  Its host value
//! is shared between handles and is supplied to execution through
//! [`Module::input_bindings`]. Replacing it never mutates graph nodes: graphs
//! already built retain their topology and a subsequent execution observes the
//! new value only through that explicit binding.

use crate::{DType, Error, Graph, NodeId, Result, Scalar, Shape, TensorData};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashMap},
    rc::Rc,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateKind {
    Parameter,
    Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CastPolicy {
    Exact,
    Allow,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    pub missing_keys: Vec<String>,
    pub unexpected_keys: Vec<String>,
    pub shape_mismatches: Vec<String>,
    pub dtype_mismatches: Vec<String>,
    pub loaded_keys: Vec<String>,
}
impl LoadReport {
    pub fn is_clean(&self) -> bool {
        self.missing_keys.is_empty()
            && self.unexpected_keys.is_empty()
            && self.shape_mismatches.is_empty()
            && self.dtype_mismatches.is_empty()
    }
}

/// A deterministic state map that converts directly to RustGrad safetensors maps.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StateDict {
    tensors: BTreeMap<String, TensorData>,
}
impl StateDict {
    pub fn tensors(&self) -> &BTreeMap<String, TensorData> {
        &self.tensors
    }
    pub fn into_tensors(self) -> BTreeMap<String, TensorData> {
        self.tensors
    }
}
impl From<BTreeMap<String, TensorData>> for StateDict {
    fn from(tensors: BTreeMap<String, TensorData>) -> Self {
        Self { tensors }
    }
}
impl From<StateDict> for BTreeMap<String, TensorData> {
    fn from(value: StateDict) -> Self {
        value.tensors
    }
}

#[derive(Clone, Debug)]
pub struct Parameter {
    graph_id: u64,
    node: NodeId,
    input_name: String,
    trainable: bool,
    value: Rc<RefCell<ParameterValue>>,
}
#[derive(Clone, Debug)]
struct ParameterValue {
    data: TensorData,
    version: u64,
}

impl Parameter {
    pub fn new(graph: &mut Graph, data: TensorData, trainable: bool) -> Self {
        let input_name = format!("__rustgrad_parameter_{}", graph.id());
        // The node index makes names unique without relying on caller-provided names.
        let input_name = format!("{input_name}_{}", graph.node_count());
        let node = graph.input_dtype(input_name.clone(), data.shape().clone(), data.dtype());
        Self {
            graph_id: graph.id(),
            node,
            input_name,
            trainable,
            value: Rc::new(RefCell::new(ParameterValue { data, version: 0 })),
        }
    }
    pub fn node(&self, graph: &Graph) -> Result<NodeId> {
        if graph.id() == self.graph_id {
            Ok(self.node)
        } else {
            Err(Error::ParameterGraphMismatch)
        }
    }
    pub fn is_trainable(&self) -> bool {
        self.trainable
    }
    pub fn shape(&self) -> Shape {
        self.value.borrow().data.shape().clone()
    }
    pub fn dtype(&self) -> DType {
        self.value.borrow().data.dtype()
    }
    pub fn value(&self) -> TensorData {
        self.value.borrow().data.clone()
    }
    pub fn version(&self) -> u64 {
        self.value.borrow().version
    }
    pub fn replace(&self, data: TensorData) -> Result<()> {
        let mut value = self.value.borrow_mut();
        if data.shape() != value.data.shape() || data.dtype() != value.data.dtype() {
            return Err(Error::ParameterValueMismatch {
                expected_shape: value.data.shape().clone(),
                actual_shape: data.shape().clone(),
                expected_dtype: value.data.dtype(),
                actual_dtype: data.dtype(),
            });
        }
        value.data = data;
        value.version = value.version.wrapping_add(1);
        Ok(())
    }
    fn identity(&self) -> usize {
        Rc::as_ptr(&self.value) as usize
    }
    fn binding(&self) -> (String, TensorData) {
        (self.input_name.clone(), self.value())
    }
}

/// Rust-native explicit state traversal. Implementors call `visit` for fields,
/// nested modules, vectors, and options in their declared deterministic order.
pub trait Module {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind));
    fn state_dict(&self) -> StateDict {
        let mut tensors = BTreeMap::new();
        let mut seen = BTreeSet::new();
        self.visit("", &mut |name, parameter, _| {
            if seen.insert(parameter.identity()) {
                tensors.insert(name, parameter.value());
            }
        });
        StateDict { tensors }
    }
    fn input_bindings(&self) -> HashMap<String, TensorData> {
        let mut inputs = HashMap::new();
        let mut seen = BTreeSet::new();
        self.visit("", &mut |_, parameter, _| {
            if seen.insert(parameter.identity()) {
                let (name, value) = parameter.binding();
                inputs.insert(name, value);
            }
        });
        inputs
    }
    fn load_state_dict(
        &self,
        state: &StateDict,
        strict: bool,
        cast: CastPolicy,
    ) -> Result<LoadReport> {
        let mut entries = BTreeMap::<String, Parameter>::new();
        let mut seen = BTreeSet::new();
        self.visit("", &mut |name, parameter, _| {
            if seen.insert(parameter.identity()) {
                entries.insert(name, parameter.clone());
            }
        });
        let mut report = LoadReport::default();
        for (name, parameter) in &entries {
            let Some(value) = state.tensors.get(name) else {
                report.missing_keys.push(name.clone());
                continue;
            };
            if value.shape() != &parameter.shape() {
                report.shape_mismatches.push(name.clone());
                continue;
            }
            let value = if value.dtype() != parameter.dtype() {
                if cast == CastPolicy::Allow {
                    value.cast(parameter.dtype())
                } else {
                    report.dtype_mismatches.push(name.clone());
                    continue;
                }
            } else {
                value.clone()
            };
            parameter.replace(value)?;
            report.loaded_keys.push(name.clone());
        }
        report.unexpected_keys = state
            .tensors
            .keys()
            .filter(|name| !entries.contains_key(*name))
            .cloned()
            .collect();
        if strict && !report.is_clean() {
            return Err(Error::Serialization {
                reason: format!(
                    "state_dict mismatch: missing={:?}, unexpected={:?}, shape={:?}, dtype={:?}",
                    report.missing_keys,
                    report.unexpected_keys,
                    report.shape_mismatches,
                    report.dtype_mismatches
                ),
            });
        }
        Ok(report)
    }
}

fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.into()
    } else {
        format!("{prefix}.{name}")
    }
}
fn uniform(shape: Shape, low: f32, high: f32, seed: u64) -> Result<TensorData> {
    fn mix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    }
    TensorData::from_scalars(
        shape.clone(),
        DType::F32,
        (0..shape.numel()?).map(|i| {
            Scalar::F(
                (low + (high - low)
                    * ((mix(seed.wrapping_add(i as u64)) >> 40) as f32 / (1u32 << 24) as f32))
                    as f64,
            )
        }),
    )
}

pub struct Linear {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub in_features: usize,
    pub out_features: usize,
}
impl Linear {
    pub fn new(
        graph: &mut Graph,
        in_features: usize,
        out_features: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        if in_features == 0 {
            return Err(Error::InvalidRandom {
                reason: "Linear in_features must be nonzero",
            });
        }
        let bound = 1.0 / (in_features as f32).sqrt();
        Ok(Self {
            weight: Parameter::new(
                graph,
                uniform(Shape::new([out_features, in_features]), -bound, bound, seed)?,
                true,
            ),
            bias: bias.then(|| {
                Parameter::new(
                    graph,
                    uniform(
                        Shape::new([out_features]),
                        -bound,
                        bound,
                        seed.wrapping_add(1),
                    )
                    .expect("validated shape"),
                    true,
                )
            }),
            in_features,
            out_features,
        })
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        if graph.shape(input)?.dims().last().copied() != Some(self.in_features) {
            return Err(Error::InvalidMatmul {
                lhs: graph.shape(input)?.clone(),
                rhs: Shape::new([self.out_features, self.in_features]),
            });
        }
        let weight = self.weight.node(graph)?;
        let weight = graph.permute(weight, vec![1, 0])?;
        let output = graph.matmul(input, weight)?;
        self.bias
            .as_ref()
            .map_or(Ok(output), |bias| graph.add(output, bias.node(graph)?))
    }
}
impl Module for Linear {
    fn visit(&self, prefix: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        v(join(prefix, "weight"), &self.weight, StateKind::Parameter);
        if let Some(b) = &self.bias {
            v(join(prefix, "bias"), b, StateKind::Parameter)
        }
    }
}

pub struct Embedding {
    pub weight: Parameter,
    pub padding_idx: Option<usize>,
    embedding_dim: usize,
}
impl Embedding {
    pub fn new(
        graph: &mut Graph,
        vocab: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        seed: u64,
    ) -> Result<Self> {
        if padding_idx.is_some_and(|i| i >= vocab) {
            return Err(Error::InvalidIndex);
        }
        let bound = (6.0f32 / (vocab + embedding_dim) as f32).sqrt();
        Ok(Self {
            weight: Parameter::new(
                graph,
                uniform(Shape::new([vocab, embedding_dim]), -bound, bound, seed)?,
                true,
            ),
            padding_idx,
            embedding_dim,
        })
    }
    pub fn forward(&self, graph: &mut Graph, index: NodeId) -> Result<NodeId> {
        if !graph.dtype(index)?.is_integer() {
            return Err(Error::InvalidIndexDType {
                op: "embedding",
                actual: graph.dtype(index)?,
            });
        }
        let mut dims = graph.shape(index)?.dims().to_vec();
        dims.push(1);
        let expanded = graph.reshape(index, Shape::new(dims.clone()))?;
        *dims.last_mut().expect("added dimension") = self.embedding_dim;
        let expanded = graph.expand(expanded, Shape::new(dims))?;
        let output = graph.gather(self.weight.node(graph)?, expanded, 0)?;
        if let Some(padding) = self.padding_idx {
            let pad = graph.constant(TensorData::scalar_with_dtype(
                Scalar::I(padding as i64),
                graph.dtype(index)?,
            ));
            let mask = graph.eq(index, pad)?;
            let mask = graph.reshape(
                mask,
                Shape::new({
                    let mut d = graph.shape(index)?.dims().to_vec();
                    d.push(1);
                    d
                }),
            )?;
            let mask = graph.expand(mask, graph.shape(output)?.clone())?;
            let zero = graph.zeros_like(output, None)?;
            graph.select(mask, zero, output)
        } else {
            Ok(output)
        }
    }
}
impl Module for Embedding {
    fn visit(&self, prefix: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        v(join(prefix, "weight"), &self.weight, StateKind::Parameter)
    }
}

pub struct LayerNorm {
    pub weight: Option<Parameter>,
    pub bias: Option<Parameter>,
    normalized_shape: Shape,
    eps: f32,
}
impl LayerNorm {
    pub fn new(
        graph: &mut Graph,
        normalized_shape: impl Into<Shape>,
        eps: f32,
        affine: bool,
    ) -> Result<Self> {
        let shape = normalized_shape.into();
        if shape.rank() == 0 || !eps.is_finite() || eps < 0.0 {
            return Err(Error::InvalidRandom {
                reason: "invalid LayerNorm shape or epsilon",
            });
        };
        Ok(Self {
            weight: affine.then(|| {
                Parameter::new(
                    graph,
                    TensorData::ones(shape.clone()).expect("valid shape"),
                    true,
                )
            }),
            bias: affine.then(|| {
                Parameter::new(
                    graph,
                    TensorData::zeros(shape.clone()).expect("valid shape"),
                    true,
                )
            }),
            normalized_shape: shape,
            eps,
        })
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        let shape = graph.shape(input)?.clone();
        if !shape.dims().ends_with(self.normalized_shape.dims()) {
            return Err(Error::InvalidReshape {
                from: shape,
                to: self.normalized_shape.clone(),
            });
        };
        let axes = (0..self.normalized_shape.rank())
            .map(|i| -1 - i as isize)
            .collect();
        let mean = graph.reduce(input, crate::ReduceKind::Mean, Some(axes), true)?;
        let centered = graph.sub(input, mean)?;
        let squared = graph.square(centered)?;
        let variance = graph.reduce(
            squared,
            crate::ReduceKind::Mean,
            Some(
                (0..self.normalized_shape.rank())
                    .map(|i| -1 - i as isize)
                    .collect(),
            ),
            true,
        )?;
        let eps = graph.constant(TensorData::scalar(self.eps));
        let variance = graph.add(variance, eps)?;
        let denominator = graph.sqrt(variance)?;
        let out = graph.div(centered, denominator)?;
        let out = if let Some(weight) = &self.weight {
            graph.mul(out, weight.node(graph)?)?
        } else {
            out
        };
        self.bias
            .as_ref()
            .map_or(Ok(out), |bias| graph.add(out, bias.node(graph)?))
    }
}
impl Module for LayerNorm {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        if let Some(w) = &self.weight {
            v(join(p, "weight"), w, StateKind::Parameter)
        }
        if let Some(b) = &self.bias {
            v(join(p, "bias"), b, StateKind::Parameter)
        }
    }
}

pub struct RMSNorm {
    pub weight: Option<Parameter>,
    dim: usize,
    eps: f32,
}
impl RMSNorm {
    pub fn new(graph: &mut Graph, dim: usize, eps: f32, affine: bool) -> Result<Self> {
        if dim == 0 || !eps.is_finite() || eps < 0.0 {
            return Err(Error::InvalidRandom {
                reason: "invalid RMSNorm dimension or epsilon",
            });
        }
        Ok(Self {
            weight: affine.then(|| {
                Parameter::new(
                    graph,
                    TensorData::ones(Shape::new([dim])).expect("valid"),
                    true,
                )
            }),
            dim,
            eps,
        })
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        if graph.shape(input)?.dims().last().copied() != Some(self.dim) {
            return Err(Error::InvalidReshape {
                from: graph.shape(input)?.clone(),
                to: Shape::new([self.dim]),
            });
        }
        let original = graph.dtype(input)?;
        let x = if original == DType::F16 || original == DType::BF16 {
            graph.cast(input, DType::F32)?
        } else {
            input
        };
        let squared = graph.square(x)?;
        let mean = graph.reduce(squared, crate::ReduceKind::Mean, Some(vec![-1]), true)?;
        let eps = graph.constant(TensorData::scalar(self.eps));
        let mean = graph.add(mean, eps)?;
        let scale = graph.rsqrt(mean)?;
        let out = graph.mul(x, scale)?;
        let out = if x != input {
            graph.cast(out, original)?
        } else {
            out
        };
        self.weight
            .as_ref()
            .map_or(Ok(out), |w| graph.mul(out, w.node(graph)?))
    }
}
impl Module for RMSNorm {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        if let Some(w) = &self.weight {
            v(join(p, "weight"), w, StateKind::Parameter)
        }
    }
}

pub struct Dropout {
    pub probability: f64,
    pub training: bool,
    pub seed: u64,
}
impl Dropout {
    pub fn new(probability: f64, training: bool, seed: u64) -> Result<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(Error::UnsupportedDropout {
                probability_bits: probability.to_bits(),
            });
        }
        Ok(Self {
            probability,
            training,
            seed,
        })
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        if !self.training || self.probability == 0.0 {
            return Ok(input);
        }
        if self.probability == 1.0 {
            return graph.zeros_like(input, None);
        }
        let random = graph.rand_like(input, None, self.seed)?;
        let threshold = graph.constant(TensorData::scalar(self.probability as f32));
        let mask = graph.ge(random, threshold)?;
        let zero = graph.zeros_like(input, None)?;
        let kept = graph.select(mask, input, zero)?;
        let scale = graph.constant(TensorData::scalar((1.0 / (1.0 - self.probability)) as f32));
        graph.mul(kept, scale)
    }
}
impl Module for Dropout {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}

/// A deterministic traversal-only heterogeneous container. Forward composition
/// remains explicit because Rust cannot erase differing module call signatures.
#[derive(Default)]
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}
impl Sequential {
    pub fn push(&mut self, module: impl Module + 'static) {
        self.modules.push(Box::new(module));
    }
}
impl Module for Sequential {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        for (i, module) in self.modules.iter().enumerate() {
            module.visit(&join(p, &i.to_string()), v)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, Storage, save_safetensors};

    fn f32s(data: &TensorData) -> Vec<f32> {
        match data.storage() {
            Storage::F32(v) => v.clone(),
            _ => panic!("expected f32"),
        }
    }
    fn execute(
        graph: &Graph,
        output: NodeId,
        module: &impl Module,
        input: (&str, TensorData),
    ) -> TensorData {
        let mut bindings = module.input_bindings();
        bindings.insert(input.0.into(), input.1);
        CpuBackend.execute(graph, output, &bindings).unwrap()
    }

    #[test]
    fn linear_is_a_graph_leaf_and_replacement_is_versioned() {
        let mut graph = Graph::new();
        let linear = Linear::new(&mut graph, 2, 1, true, 7).unwrap();
        linear
            .weight
            .replace(TensorData::new([1, 2], vec![2., 3.]).unwrap())
            .unwrap();
        linear
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([1], vec![1.]).unwrap())
            .unwrap();
        let input = graph.input("x", [2, 2]);
        let output = linear.forward(&mut graph, input).unwrap();
        assert_eq!(
            f32s(&execute(
                &graph,
                output,
                &linear,
                ("x", TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap())
            )),
            vec![9., 19.]
        );
        assert!(
            linear
                .weight
                .replace(TensorData::new([2], vec![1., 2.]).unwrap())
                .is_err()
        );
        assert_eq!(linear.weight.version(), 1);
        let loss = graph
            .reduce(output, crate::ReduceKind::Sum, None, false)
            .unwrap();
        let gradient = graph
            .grad(loss, linear.weight.node(&graph).unwrap())
            .unwrap();
        assert_eq!(
            f32s(&execute(
                &graph,
                gradient,
                &linear,
                ("x", TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap())
            )),
            vec![4., 6.]
        );
    }

    struct Tied {
        left: Linear,
        right: Parameter,
        running: Parameter,
    }
    impl Module for Tied {
        fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
            self.left.visit(&join(p, "layers.0"), v);
            v(
                join(p, "layers.1.weight"),
                &self.right,
                StateKind::Parameter,
            );
            v(join(p, "running"), &self.running, StateKind::Buffer)
        }
    }
    #[test]
    fn state_is_deterministic_shared_and_safetensors_portable() {
        let mut graph = Graph::new();
        let left = Linear::new(&mut graph, 2, 2, false, 1).unwrap();
        let running = Parameter::new(&mut graph, TensorData::new([1], vec![0.]).unwrap(), false);
        let tied = Tied {
            right: left.weight.clone(),
            left,
            running,
        };
        let state = tied.state_dict();
        assert_eq!(
            state.tensors().keys().cloned().collect::<Vec<_>>(),
            vec!["layers.0.weight", "running"]
        );
        let bytes = save_safetensors(&state.clone().into_tensors(), &BTreeMap::new()).unwrap();
        let (raw, _) = crate::load_safetensors(&bytes).unwrap();
        let report = tied
            .load_state_dict(&StateDict::from(raw), true, CastPolicy::Exact)
            .unwrap();
        assert_eq!(report.loaded_keys, vec!["layers.0.weight", "running"]);
        let mut changed = state.clone().into_tensors();
        changed.insert("unexpected".into(), TensorData::scalar(1.));
        let report = tied
            .load_state_dict(&StateDict::from(changed), false, CastPolicy::Exact)
            .unwrap();
        assert_eq!(report.unexpected_keys, vec!["unexpected"]);
    }

    #[test]
    fn embedding_norm_and_dropout_have_expected_semantics() {
        let mut graph = Graph::new();
        let embedding = Embedding::new(&mut graph, 3, 2, Some(0), 1).unwrap();
        embedding
            .weight
            .replace(TensorData::new([3, 2], vec![9., 9., 1., 2., 3., 4.]).unwrap())
            .unwrap();
        let indices = graph.input_dtype("i", [2], DType::I32);
        let out = embedding.forward(&mut graph, indices).unwrap();
        assert_eq!(
            f32s(&execute(
                &graph,
                out,
                &embedding,
                (
                    "i",
                    TensorData::from_scalars([2], DType::I32, [Scalar::I(0), Scalar::I(2)])
                        .unwrap()
                )
            )),
            vec![0., 0., 3., 4.]
        );
        let mut dropout_graph = Graph::new();
        let dropout = Dropout::new(0.5, true, 42).unwrap();
        let x = dropout_graph.input("x", [4]);
        let a = dropout.forward(&mut dropout_graph, x).unwrap();
        let b = dropout.forward(&mut dropout_graph, x).unwrap();
        let data = TensorData::new([4], vec![1.; 4]).unwrap();
        assert_eq!(
            execute(&dropout_graph, a, &dropout, ("x", data.clone())),
            execute(&dropout_graph, b, &dropout, ("x", data))
        );
        let mut norm_graph = Graph::new();
        let norm = RMSNorm::new(&mut norm_graph, 2, 1e-6, false).unwrap();
        let nx = norm_graph.input("nx", [1, 2]);
        let no = norm.forward(&mut norm_graph, nx).unwrap();
        let values = f32s(&execute(
            &norm_graph,
            no,
            &norm,
            ("nx", TensorData::new([1, 2], vec![3., 4.]).unwrap()),
        ));
        assert!((values[0] - 0.848_528_1).abs() < 1e-5 && (values[1] - 1.131_370_9).abs() < 1e-5);
    }
}
