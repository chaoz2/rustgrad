//! Deterministic, CPU-only central-difference checks for scalar Graph losses.
//!
//! This is an inspection helper: it clones the graph before building VJPs and
//! clones bindings for every perturbation, so checking cannot mutate caller
//! graph structure or input values. Its default central-difference and
//! tolerance policy matches the checked-in tinygrad `extra.gradcheck` helper.

use crate::{Backend, CpuBackend, DType, Error, Graph, NodeId, Op, Shape, TensorData};
use std::collections::{BTreeMap, HashMap};
use std::fmt;

/// Central-difference configuration for [`gradcheck_cpu`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradcheckConfig {
    pub epsilon: f64,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
}

impl Default for GradcheckConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-3,
            absolute_tolerance: 1e-3,
            relative_tolerance: 1e-3,
        }
    }
}

/// One deterministic analytic-versus-numeric mismatch.
#[derive(Clone, Debug, PartialEq)]
pub struct GradcheckMismatch {
    pub target: NodeId,
    pub input_name: String,
    pub coordinate: usize,
    pub analytic: f64,
    pub numerical: f64,
    pub absolute_error: f64,
    pub tolerance: f64,
}

/// Stable result of checking one or more graph input leaves.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GradcheckReport {
    pub coordinates_checked: usize,
    pub mismatches: Vec<GradcheckMismatch>,
}

impl GradcheckReport {
    pub fn passed(&self) -> bool {
        self.mismatches.is_empty()
    }
}

/// A preflight or finite-difference failure distinct from an analytic mismatch.
#[derive(Clone, Debug, PartialEq)]
pub enum GradcheckError {
    InvalidConfig(&'static str),
    NoTargets,
    DuplicateTarget(NodeId),
    TargetNotInput(NodeId),
    UnsupportedTargetDType { target: NodeId, dtype: DType },
    MissingBinding(String),
    BindingShape {
        name: String,
        expected: Shape,
        actual: Shape,
    },
    BindingDType {
        name: String,
        expected: DType,
        actual: DType,
    },
    EmptyTarget(NodeId),
    NonScalarLoss(Shape),
    NonFloatLoss(DType),
    NonFiniteBaseLoss,
    NonFiniteAnalytic { target: NodeId, coordinate: usize },
    NonFiniteLoss { target: NodeId, coordinate: usize },
    Graph(Error),
}

impl fmt::Display for GradcheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(reason) => write!(f, "invalid gradcheck configuration: {reason}"),
            Self::NoTargets => write!(f, "gradcheck requires at least one target"),
            Self::DuplicateTarget(target) => write!(f, "duplicate gradcheck target %{target}"),
            Self::TargetNotInput(target) => write!(f, "gradcheck target %{target} is not an input"),
            Self::UnsupportedTargetDType { target, dtype } => {
                write!(f, "gradcheck target %{target} has unsupported dtype {dtype:?}")
            }
            Self::MissingBinding(name) => write!(f, "gradcheck is missing binding {name:?}"),
            Self::BindingShape {
                name,
                expected,
                actual,
            } => write!(f, "gradcheck binding {name:?} expected {expected}, got {actual}"),
            Self::BindingDType {
                name,
                expected,
                actual,
            } => write!(
                f,
                "gradcheck binding {name:?} expected {expected:?}, got {actual:?}"
            ),
            Self::EmptyTarget(target) => write!(f, "gradcheck target %{target} is empty"),
            Self::NonScalarLoss(shape) => write!(f, "gradcheck requires a scalar loss, got {shape}"),
            Self::NonFloatLoss(dtype) => write!(f, "gradcheck loss has non-float dtype {dtype:?}"),
            Self::NonFiniteBaseLoss => write!(f, "gradcheck base loss is non-finite"),
            Self::NonFiniteAnalytic { target, coordinate } => {
                write!(f, "gradcheck analytic gradient for %{target}[{coordinate}] is non-finite")
            }
            Self::NonFiniteLoss { target, coordinate } => {
                write!(f, "gradcheck perturbation for %{target}[{coordinate}] has non-finite loss")
            }
            Self::Graph(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for GradcheckError {}

impl From<Error> for GradcheckError {
    fn from(error: Error) -> Self {
        Self::Graph(error)
    }
}

fn validate_config(config: GradcheckConfig) -> Result<(), GradcheckError> {
    if !config.epsilon.is_finite() || config.epsilon <= 0.0 {
        return Err(GradcheckError::InvalidConfig(
            "epsilon must be finite and positive",
        ));
    }
    if !config.absolute_tolerance.is_finite() || config.absolute_tolerance < 0.0 {
        return Err(GradcheckError::InvalidConfig(
            "absolute tolerance must be finite and nonnegative",
        ));
    }
    if !config.relative_tolerance.is_finite() || config.relative_tolerance < 0.0 {
        return Err(GradcheckError::InvalidConfig(
            "relative tolerance must be finite and nonnegative",
        ));
    }
    Ok(())
}

fn bindings_for_cpu(bindings: &BTreeMap<String, TensorData>) -> HashMap<String, TensorData> {
    bindings
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn perturbed_bindings(
    bindings: &BTreeMap<String, TensorData>,
    name: &str,
    value: &TensorData,
    coordinate: usize,
    delta: f64,
) -> Result<HashMap<String, TensorData>, GradcheckError> {
    let mut perturbed = bindings_for_cpu(bindings);
    let replacement = TensorData::from_scalars(
        value.shape().clone(),
        value.dtype(),
        (0..value.len()).map(|index| {
            if index == coordinate {
                crate::Scalar::F(value.scalar_at(index).as_f64() + delta)
            } else {
                value.scalar_at(index)
            }
        }),
    )?;
    perturbed.insert(name.to_owned(), replacement);
    Ok(perturbed)
}

/// Compares reverse-mode gradients of a scalar loss with central differences.
///
/// Targets must be distinct F32/F64 input leaves. Target and coordinate order
/// is canonicalized by graph node id and row-major dense index. The caller's
/// graph and bindings are never mutated: VJPs are built in a graph clone and
/// every numeric evaluation uses freshly cloned bindings.
pub fn gradcheck_cpu(
    graph: &Graph,
    loss: NodeId,
    targets: &[NodeId],
    bindings: &BTreeMap<String, TensorData>,
    config: GradcheckConfig,
) -> Result<GradcheckReport, GradcheckError> {
    validate_config(config)?;
    if targets.is_empty() {
        return Err(GradcheckError::NoTargets);
    }

    let mut checked_graph = graph.clone();
    let loss_shape = checked_graph.shape(loss)?.clone();
    if loss_shape.numel()? != 1 {
        return Err(GradcheckError::NonScalarLoss(loss_shape));
    }
    let loss_dtype = checked_graph.dtype(loss)?;
    if !loss_dtype.is_float() {
        return Err(GradcheckError::NonFloatLoss(loss_dtype));
    }

    let mut ordered_targets = targets.to_vec();
    ordered_targets.sort_by_key(|target| target.index());
    for pair in ordered_targets.windows(2) {
        if pair[0] == pair[1] {
            return Err(GradcheckError::DuplicateTarget(pair[0]));
        }
    }

    let mut report = GradcheckReport::default();
    for target in ordered_targets {
        let name = match checked_graph.op(target)? {
            Op::Input { name } => name.clone(),
            _ => return Err(GradcheckError::TargetNotInput(target)),
        };
        let dtype = checked_graph.dtype(target)?;
        if !matches!(dtype, DType::F32 | DType::F64) {
            return Err(GradcheckError::UnsupportedTargetDType { target, dtype });
        }
        let shape = checked_graph.shape(target)?.clone();
        let value = bindings
            .get(&name)
            .ok_or_else(|| GradcheckError::MissingBinding(name.clone()))?;
        if value.shape() != &shape {
            return Err(GradcheckError::BindingShape {
                name,
                expected: shape,
                actual: value.shape().clone(),
            });
        }
        if value.dtype() != dtype {
            return Err(GradcheckError::BindingDType {
                name,
                expected: dtype,
                actual: value.dtype(),
            });
        }
        if value.is_empty() {
            return Err(GradcheckError::EmptyTarget(target));
        }

        let base_loss = CpuBackend
            .execute(&checked_graph, loss, &bindings_for_cpu(bindings))?
            .scalar_at(0)
            .as_f64();
        if !base_loss.is_finite() {
            return Err(GradcheckError::NonFiniteBaseLoss);
        }

        let analytic_node = checked_graph.grad(loss, target)?;
        let analytic = CpuBackend.execute(
            &checked_graph,
            analytic_node,
            &bindings_for_cpu(bindings),
        )?;
        for coordinate in 0..value.len() {
            let analytic_value = analytic.scalar_at(coordinate).as_f64();
            if !analytic_value.is_finite() {
                return Err(GradcheckError::NonFiniteAnalytic { target, coordinate });
            }
            let plus = CpuBackend.execute(
                &checked_graph,
                loss,
                &perturbed_bindings(bindings, &name, value, coordinate, config.epsilon)?,
            )?
            .scalar_at(0)
            .as_f64();
            let minus = CpuBackend.execute(
                &checked_graph,
                loss,
                &perturbed_bindings(bindings, &name, value, coordinate, -config.epsilon)?,
            )?
            .scalar_at(0)
            .as_f64();
            if !plus.is_finite() || !minus.is_finite() {
                return Err(GradcheckError::NonFiniteLoss { target, coordinate });
            }
            let numerical = (plus - minus) / (2.0 * config.epsilon);
            let absolute_error = (analytic_value - numerical).abs();
            let tolerance = config.absolute_tolerance
                + config.relative_tolerance * analytic_value.abs().max(numerical.abs());
            report.coordinates_checked += 1;
            if absolute_error > tolerance {
                report.mismatches.push(GradcheckMismatch {
                    target,
                    input_name: name.clone(),
                    coordinate,
                    analytic: analytic_value,
                    numerical,
                    absolute_error,
                    tolerance,
                });
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
        TensorData::new(shape, values.to_vec()).unwrap()
    }

    #[test]
    fn cpu_gradcheck_is_deterministic_and_preserves_callers() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 1]);
        let y = graph.input("y", [2]);
        let product = graph.mul(x, y).unwrap();
        let loss = graph.sum_all(product).unwrap();
        let bindings = BTreeMap::from([
            ("x".into(), data([2, 1], &[1.5, -2.0])),
            ("y".into(), data([2], &[3.0, 4.0])),
        ]);
        let original_nodes = graph.node_count();
        let original_bindings = bindings.clone();

        let report = gradcheck_cpu(&graph, loss, &[y, x], &bindings, GradcheckConfig::default())
            .unwrap();
        assert!(report.passed());
        assert_eq!(report.coordinates_checked, 4);
        assert_eq!(graph.node_count(), original_nodes);
        assert_eq!(bindings, original_bindings);
    }

    #[test]
    fn cpu_gradcheck_reports_mismatches_in_coordinate_order() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let square = graph.square(x).unwrap();
        let cube = graph.mul(square, x).unwrap();
        let loss = graph.sum_all(cube).unwrap();
        let bindings = BTreeMap::from([("x".into(), data([2], &[1.0, 2.0]))]);

        let report = gradcheck_cpu(
            &graph,
            loss,
            &[x],
            &bindings,
            GradcheckConfig {
                epsilon: 1.0,
                absolute_tolerance: 0.0,
                relative_tolerance: 0.0,
            },
        )
        .unwrap();
        assert_eq!(report.coordinates_checked, 2);
        assert_eq!(
            report
                .mismatches
                .iter()
                .map(|mismatch| mismatch.coordinate)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(report
            .mismatches
            .iter()
            .all(|mismatch| mismatch.analytic < mismatch.numerical));
    }

    #[test]
    fn cpu_gradcheck_rejects_empty_targets_without_mutation() {
        let mut graph = Graph::new();
        let x = graph.input("x", [0]);
        let loss = graph.sum_all(x).unwrap();
        let bindings = BTreeMap::from([("x".into(), data([0], &[]))]);
        let original_nodes = graph.node_count();
        let original_bindings = bindings.clone();

        assert_eq!(
            gradcheck_cpu(&graph, loss, &[x], &bindings, GradcheckConfig::default()),
            Err(GradcheckError::EmptyTarget(x))
        );
        assert_eq!(graph.node_count(), original_nodes);
        assert_eq!(bindings, original_bindings);
    }

    #[test]
    fn cpu_gradcheck_rejects_nonfinite_losses_without_mutation() {
        let mut graph = Graph::new();
        let x = graph.input("x", []);
        let loss = graph.log(x).unwrap();
        let bindings = BTreeMap::from([("x".into(), data([], &[0.0]))]);
        let original_nodes = graph.node_count();
        let original_bindings = bindings.clone();

        assert_eq!(
            gradcheck_cpu(&graph, loss, &[x], &bindings, GradcheckConfig::default()),
            Err(GradcheckError::NonFiniteBaseLoss)
        );
        assert_eq!(graph.node_count(), original_nodes);
        assert_eq!(bindings, original_bindings);
    }
}
