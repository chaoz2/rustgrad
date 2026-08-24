//! Immutable schedule capture and backend-neutral interpreter replay.
use crate::{Graph, NodeId, Op, Schedule, ScheduleItem, TensorData};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ReplayInput {
    pub name: String,
    pub node: NodeId,
    pub desc: crate::BufferDesc,
}
#[derive(Clone, Debug)]
pub struct CapturedSchedule {
    pub items: Vec<ScheduleItem>,
    pub inputs: Vec<ReplayInput>,
    pub constants: BTreeMap<u64, TensorData>,
    pub requested: Vec<u64>,
    pub identity: u64,
    pub(crate) symbolic: Option<super::symbolic::SymbolicSchema>,
    pub(crate) specialized_from: Option<super::symbolic::SpecializedFrom>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayError {
    Missing(String),
    Extra(String),
    Descriptor(String),
    Corrupt(String),
    Execute(String),
    Unsupported(String),
    Backend(String),
    Symbolic(String),
    Batch { invocation: usize, reason: String },
}
impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "replay error: {self:?}")
    }
}
impl std::error::Error for ReplayError {}
impl CapturedSchedule {
    pub fn capture(
        graph: &Graph,
        schedule: &Schedule,
        requested: &[NodeId],
    ) -> Result<Self, ReplayError> {
        let mut inputs = BTreeMap::new();
        let mut constants = BTreeMap::new();
        let mut produced = BTreeSet::new();
        for item in &schedule.items {
            item.validate_input_bindings()
                .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
            for b in item.ordered_inputs() {
                match graph
                    .op(b.input_node)
                    .map_err(|e| ReplayError::Corrupt(e.to_string()))?
                {
                    Op::Input { name } => {
                        inputs.entry(name.clone()).or_insert(ReplayInput {
                            name: name.clone(),
                            node: b.input_node,
                            desc: b.desc.clone(),
                        });
                    }
                    Op::Constant(v) => {
                        constants.insert(b.desc.id, v.clone());
                    }
                    _ if produced.contains(&b.desc.id) => {}
                    _ if item.external_materializations.contains(&b.input_node) => {
                        let name = format!("@materialized/{}", b.desc.id);
                        inputs.entry(name.clone()).or_insert(ReplayInput {
                            name,
                            node: b.input_node,
                            desc: b.desc.clone(),
                        });
                    }
                    _ => {
                        return Err(ReplayError::Corrupt(format!(
                            "unproduced binding {}",
                            b.desc.id
                        )));
                    }
                }
            }
            produced.insert(item.output.id);
        }
        let inputs = inputs.into_values().collect::<Vec<_>>();
        let mut capture = Self {
            items: schedule.items.clone(),
            inputs,
            constants,
            requested: requested.iter().map(|n| n.index() as u64).collect(),
            identity: 0,
            symbolic: None,
            specialized_from: None,
        };
        capture.identity = crate::schedule::artifact::identity(&capture)
            .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
        crate::schedule::artifact::validate_capture(&capture)
            .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
        Ok(capture)
    }
    /// Captures a symbolic shape family from one validated concrete template.
    /// The original graph is used only to derive expressions and is never
    /// retained or reconstructed by replay.
    pub fn capture_symbolic(
        graph: &Graph,
        schedule: &Schedule,
        requested: &[NodeId],
        spec: &crate::SymbolicCaptureSpec,
        template_bindings: &BTreeMap<String, i64>,
    ) -> Result<Self, ReplayError> {
        let mut capture = Self::capture(graph, schedule, requested)?;
        capture.symbolic = Some(super::symbolic::build_schema(
            graph,
            schedule,
            &capture,
            spec,
            template_bindings,
        )?);
        capture.identity = crate::schedule::artifact::identity(&capture)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        crate::schedule::artifact::validate_capture(&capture)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        Ok(capture)
    }
    pub fn is_symbolic(&self) -> bool {
        self.symbolic.is_some()
    }
    pub fn symbolic_parameters(&self) -> &[crate::SymbolicParameter] {
        self.symbolic
            .as_ref()
            .map_or(&[], super::symbolic::SymbolicSchema::parameters)
    }
    pub fn symbolic_guards(&self) -> &[crate::SymbolicGuard] {
        self.symbolic
            .as_ref()
            .map_or(&[], super::symbolic::SymbolicSchema::guards)
    }
    /// Serializes this graph-independent executable schedule with bounded,
    /// checksummed typed descriptors and exact constant storage.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ReplayError> {
        crate::schedule::artifact::encode(self).map_err(|e| ReplayError::Corrupt(e.to_string()))
    }

    /// Validates and reconstructs a graph-independent executable schedule.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReplayError> {
        crate::schedule::artifact::decode(bytes).map_err(|e| ReplayError::Corrupt(e.to_string()))
    }

    pub fn replay(
        &self,
        provided: &BTreeMap<String, TensorData>,
    ) -> Result<Vec<TensorData>, ReplayError> {
        Ok(crate::CapturedReplayExecutor::default()
            .replay(self, provided, crate::CapturedReplayOptions::default())?
            .outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, DType, Graph, Scalar, Shape};
    use std::collections::HashMap;
    #[test]
    fn capture_replays_without_graph_traversal() {
        let mut g = Graph::new();
        let x = g.input_dtype("x", Shape::from([3]), DType::F32);
        let y = g.square(x).unwrap();
        let s = crate::schedule(&g, y).unwrap();
        let c = CapturedSchedule::capture(&g, &s, &[y]).unwrap();
        let a = BTreeMap::from([(
            "x".into(),
            TensorData::from_scalars(
                [3],
                DType::F32,
                [Scalar::F(1.), Scalar::F(2.), Scalar::F(3.)],
            )
            .unwrap(),
        )]);
        let out = c.replay(&a).unwrap();
        let oracle = CpuBackend
            .execute(
                &g,
                y,
                &a.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            )
            .unwrap();
        assert_eq!(out[0].storage(), oracle.storage());
        assert!(matches!(
            c.replay(&BTreeMap::new()),
            Err(ReplayError::Missing(_))
        ));
        let mut extra = a;
        extra.insert("unexpected".into(), TensorData::scalar(0.0));
        assert!(matches!(c.replay(&extra), Err(ReplayError::Extra(_))));
    }

    fn replay_bytes_against_cpu(
        graph: &Graph,
        output: NodeId,
        provided: BTreeMap<String, TensorData>,
    ) {
        let schedule = crate::schedule(graph, output).unwrap();
        let capture = CapturedSchedule::capture(graph, &schedule, &[output]).unwrap();
        let bytes = capture.to_bytes().unwrap();
        assert_eq!(bytes, capture.to_bytes().unwrap());
        let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(bytes, decoded.to_bytes().unwrap());
        let first = decoded.replay(&provided).unwrap();
        let second = decoded.replay(&provided).unwrap();
        let oracle = CpuBackend
            .execute(
                graph,
                output,
                &provided.clone().into_iter().collect::<HashMap<_, _>>(),
            )
            .unwrap();
        assert_eq!(first[0].storage(), oracle.storage());
        assert_eq!(second[0].storage(), oracle.storage());
    }

    #[test]
    fn serialized_view_and_reduction_replay_match_cpu() {
        let mut view_graph = Graph::new();
        let x = view_graph.input_dtype("x", Shape::from([4]), DType::I32);
        let view = view_graph.shrink(x, [(1, 4)]).unwrap();
        let doubled = view_graph.add(view, view).unwrap();
        replay_bytes_against_cpu(
            &view_graph,
            doubled,
            BTreeMap::from([(
                "x".into(),
                TensorData::from_scalars(
                    [4],
                    DType::I32,
                    [Scalar::I(2), Scalar::I(3), Scalar::I(5), Scalar::I(7)],
                )
                .unwrap(),
            )]),
        );

        let mut reduction_graph = Graph::new();
        let x = reduction_graph.input_dtype("x", Shape::from([2, 3]), DType::F32);
        let reduced = reduction_graph
            .reduce(x, crate::ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        replay_bytes_against_cpu(
            &reduction_graph,
            reduced,
            BTreeMap::from([(
                "x".into(),
                TensorData::from_scalars(
                    [2, 3],
                    DType::F32,
                    [1., 2., 3., 4., 5., 6.].map(Scalar::F),
                )
                .unwrap(),
            )]),
        );
    }

    #[test]
    fn malformed_artifacts_fail_before_execution_and_matmul_replays() {
        let mut graph = Graph::new();
        let x = graph.input("x", Shape::from([2]));
        let y = graph.square(x).unwrap();
        let schedule = crate::schedule(&graph, y).unwrap();
        let capture = CapturedSchedule::capture(&graph, &schedule, &[y]).unwrap();
        let bytes = capture.to_bytes().unwrap();
        for n in [0, 1, bytes.len() - 1] {
            assert!(matches!(
                CapturedSchedule::from_bytes(&bytes[..n]),
                Err(ReplayError::Corrupt(_))
            ));
        }
        let mut corrupt = bytes;
        corrupt[8] ^= 1;
        assert!(matches!(
            CapturedSchedule::from_bytes(&corrupt),
            Err(ReplayError::Corrupt(_))
        ));
        let mut stale = capture.clone();
        stale.items[0].dependencies.push(999);
        assert!(matches!(
            stale.replay(&BTreeMap::new()),
            Err(ReplayError::Corrupt(_))
        ));

        let mut matmul_graph = Graph::new();
        let a = matmul_graph.input("a", Shape::from([1, 2]));
        let b = matmul_graph.input("b", Shape::from([2, 1]));
        let product = matmul_graph.matmul(a, b).unwrap();
        let schedule = crate::schedule(&matmul_graph, product).unwrap();
        let capture = CapturedSchedule::capture(&matmul_graph, &schedule, &[product]).unwrap();
        let decoded = CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap();
        let provided = BTreeMap::from([
            ("a".into(), TensorData::new([1, 2], vec![2., 3.]).unwrap()),
            ("b".into(), TensorData::new([2, 1], vec![4., 5.]).unwrap()),
        ]);
        assert_eq!(decoded.replay(&provided).unwrap()[0].values(), &[23.]);
    }

    #[test]
    fn external_materialization_is_an_explicit_replay_input() {
        let mut graph = Graph::new();
        let left = graph.input("left", Shape::from([1, 2]));
        let right = graph.input("right", Shape::from([1, 2]));
        let addend = graph.input("addend", Shape::from([1, 4]));
        let joined = graph.concat([left, right], 1).unwrap();
        let output = graph.add(joined, addend).unwrap();
        let schedule =
            crate::schedule_with_external_materializations(&graph, &[output], &[joined]).unwrap();
        let capture = CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        let decoded = CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap();
        let external_name = format!("@materialized/{}", joined.index());
        assert!(decoded.inputs.iter().any(|x| x.name == external_name));
        let values = BTreeMap::from([
            (
                "addend".into(),
                TensorData::new([1, 4], vec![10., 20., 30., 40.]).unwrap(),
            ),
            (
                external_name,
                TensorData::new([1, 4], vec![1., 2., 3., 4.]).unwrap(),
            ),
        ]);
        assert_eq!(
            decoded.replay(&values).unwrap()[0].values(),
            &[11., 22., 33., 44.]
        );
    }
}
