use super::artifact::outcomes_match;
use super::{
    FuzzCase, FuzzComparisonPolicy, FuzzFailureArtifact, FuzzOutcome, FuzzPath, generate_case,
    minimize_case,
};
use crate::{
    Backend, CapturedBackendPolicy, CapturedReplayExecutor, CapturedReplayOptions,
    CapturedSchedule, CpuBackend, DType, ReplayError, schedule,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum PathError {
    Unsupported(String),
    Failed { class: &'static str, detail: String },
}
impl PathError {
    fn detail(self) -> String {
        match self {
            Self::Unsupported(value) | Self::Failed { detail: value, .. } => value,
        }
    }
    fn failure_outcome(self) -> Result<FuzzOutcome, String> {
        match self {
            Self::Unsupported(detail) => Err(format!(
                "unsupported execution cannot become a failure outcome: {detail}"
            )),
            Self::Failed { class, detail } => Ok(FuzzOutcome::Error {
                class: class.into(),
                detail,
            }),
        }
    }
}

fn replay_error(error: ReplayError) -> PathError {
    match error {
        ReplayError::Missing(detail) => PathError::Failed {
            class: "missing",
            detail,
        },
        ReplayError::Extra(detail) => PathError::Failed {
            class: "extra",
            detail,
        },
        ReplayError::Descriptor(detail) => PathError::Failed {
            class: "descriptor",
            detail,
        },
        ReplayError::Corrupt(detail) => PathError::Failed {
            class: "corrupt",
            detail,
        },
        ReplayError::Execute(detail) => PathError::Failed {
            class: "execute",
            detail,
        },
        ReplayError::Unsupported(detail) => PathError::Unsupported(detail),
        ReplayError::Backend(detail) => PathError::Failed {
            class: "backend",
            detail,
        },
        ReplayError::Symbolic(detail) => PathError::Failed {
            class: "symbolic",
            detail,
        },
        ReplayError::Batch { invocation, reason } => PathError::Failed {
            class: "batch",
            detail: format!("invocation {invocation}: {reason}"),
        },
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FuzzConfig {
    pub seed: u64,
    pub cases: u64,
    /// Enables strict scalar CPU-JIT comparison. Unsupported kernels are
    /// reported explicitly and are not included in match counts.
    pub native: bool,
}
const MAX_CAMPAIGN_CASES: u64 = 4096;

impl Default for FuzzConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            cases: 64,
            native: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FuzzComparison {
    Match {
        path: FuzzPath,
        policy: FuzzComparisonPolicy,
    },
    Unsupported {
        path: FuzzPath,
        reason: String,
    },
    Failure(Box<FuzzFailureArtifact>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FuzzCampaign {
    pub seed: u64,
    pub generated: u64,
    pub interpreter_matches: u64,
    pub native_matches: u64,
    pub native_unsupported: u64,
    pub failures: Vec<FuzzFailureArtifact>,
}

pub(super) fn record_comparison(
    report: &mut FuzzCampaign,
    case_index: u64,
    comparison: FuzzComparison,
) -> Result<(), String> {
    match comparison {
        FuzzComparison::Match {
            path: FuzzPath::CapturedInterpreter,
            ..
        } => report.interpreter_matches += 1,
        FuzzComparison::Match {
            path: FuzzPath::NativeScalar | FuzzPath::NativeVector,
            ..
        } => report.native_matches += 1,
        FuzzComparison::Match {
            path: FuzzPath::CpuOracle,
            ..
        } => return Err("CPU oracle cannot be a target comparison".into()),
        FuzzComparison::Unsupported {
            path: FuzzPath::CapturedInterpreter,
            reason,
        } => {
            return Err(format!(
                "captured interpreter coverage failure at case {case_index}: {reason}"
            ));
        }
        FuzzComparison::Unsupported {
            path: FuzzPath::NativeScalar | FuzzPath::NativeVector,
            ..
        } => report.native_unsupported += 1,
        FuzzComparison::Unsupported {
            path: FuzzPath::CpuOracle,
            ..
        } => return Err("CPU oracle cannot be unsupported comparison target".into()),
        FuzzComparison::Failure(failure) => report.failures.push(*failure),
    }
    Ok(())
}

/// Typed lifecycle result for a persisted mismatch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FuzzReplayStatus {
    /// Both current outcomes still match the recorded outcomes under the
    /// artifact's stored policy.
    Reproduced,
    /// The current oracle and target now agree.
    Resolved,
    /// A mismatch remains, but at least one current outcome differs from the
    /// recorded outcome under the stored policy.
    Changed,
    /// The target path cannot execute the case under its current contract.
    Unsupported { path: FuzzPath, reason: String },
}

fn policy_for(path: FuzzPath, expected: &FuzzOutcome) -> FuzzComparisonPolicy {
    if path == FuzzPath::CapturedInterpreter {
        return FuzzComparisonPolicy::ExactBytes;
    }
    match expected {
        FuzzOutcome::Value { tensor } if tensor.dtype == DType::F32 => {
            FuzzComparisonPolicy::FloatTolerance {
                absolute_bits: 1e-6f64.to_bits(),
                relative_bits: 1e-6f64.to_bits(),
            }
        }
        FuzzOutcome::Value { tensor } if tensor.dtype == DType::F64 => {
            FuzzComparisonPolicy::FloatTolerance {
                absolute_bits: 1e-12f64.to_bits(),
                relative_bits: 1e-12f64.to_bits(),
            }
        }
        _ => FuzzComparisonPolicy::ExactBytes,
    }
}

fn prepare(case: &FuzzCase) -> Result<(super::case::BuiltCase, CapturedSchedule), String> {
    let built = case.build()?;
    let scheduled = schedule(&built.graph, built.output).map_err(|error| error.to_string())?;
    let capture = CapturedSchedule::capture(&built.graph, &scheduled, &[built.output])
        .map_err(|error| error.to_string())?;
    let capture =
        CapturedSchedule::from_bytes(&capture.to_bytes().map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    Ok((built, capture))
}

pub(super) fn exact_single_output(
    mut outputs: Vec<crate::TensorData>,
) -> Result<FuzzOutcome, PathError> {
    if outputs.len() != 1 {
        return Err(PathError::Failed {
            class: "output_count",
            detail: format!("expected exactly one replay output, got {}", outputs.len()),
        });
    }
    Ok(FuzzOutcome::value(
        &outputs.pop().expect("length checked as exactly one"),
    ))
}

fn execute_path(case: &FuzzCase, path: FuzzPath) -> Result<FuzzOutcome, PathError> {
    let (built, capture) = prepare(case).map_err(|detail| PathError::Failed {
        class: "setup",
        detail,
    })?;
    match path {
        FuzzPath::CpuOracle => CpuBackend
            .execute(&built.graph, built.output, &built.oracle)
            .map(|value| FuzzOutcome::value(&value))
            .map_err(|error| PathError::Failed {
                class: "cpu",
                detail: error.to_string(),
            }),
        FuzzPath::CapturedInterpreter | FuzzPath::NativeScalar | FuzzPath::NativeVector => {
            let backend = match path {
                FuzzPath::CapturedInterpreter => CapturedBackendPolicy::Interpreter,
                FuzzPath::NativeScalar => CapturedBackendPolicy::NativeJit { vectorized: false },
                FuzzPath::NativeVector => CapturedBackendPolicy::NativeJit { vectorized: true },
                FuzzPath::CpuOracle => unreachable!(),
            };
            match CapturedReplayExecutor::default().replay(
                &capture,
                &built.ordered,
                CapturedReplayOptions { backend },
            ) {
                Ok(result) => exact_single_output(result.outputs),
                Err(error) => Err(replay_error(error)),
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum MismatchSignature {
    Value,
    TargetError(String),
    OracleError(String),
    ErrorPair(String, String),
}

fn mismatch_signature(expected: &FuzzOutcome, actual: &FuzzOutcome) -> MismatchSignature {
    match (expected, actual) {
        (FuzzOutcome::Value { .. }, FuzzOutcome::Value { .. }) => MismatchSignature::Value,
        (FuzzOutcome::Value { .. }, FuzzOutcome::Error { class, .. }) => {
            MismatchSignature::TargetError(class.clone())
        }
        (FuzzOutcome::Error { class, .. }, FuzzOutcome::Value { .. }) => {
            MismatchSignature::OracleError(class.clone())
        }
        (FuzzOutcome::Error { class: lhs, .. }, FuzzOutcome::Error { class: rhs, .. }) => {
            MismatchSignature::ErrorPair(lhs.clone(), rhs.clone())
        }
    }
}

pub(super) fn compare_path_with(
    seed: u64,
    case_index: u64,
    case: &FuzzCase,
    path: FuzzPath,
    execute: &mut impl FnMut(&FuzzCase, FuzzPath) -> Result<FuzzOutcome, PathError>,
) -> Result<FuzzComparison, String> {
    let expected = execute(case, FuzzPath::CpuOracle).map_err(PathError::detail)?;
    let actual = match execute(case, path) {
        Ok(actual) => actual,
        Err(PathError::Unsupported(reason)) => {
            return Ok(FuzzComparison::Unsupported { path, reason });
        }
        Err(error) => error.failure_outcome()?,
    };
    let policy = policy_for(path, &expected);
    if outcomes_match(&expected, &actual, policy) {
        return Ok(FuzzComparison::Match { path, policy });
    }
    let signature = mismatch_signature(&expected, &actual);
    let minimized = minimize_case(case, |candidate| {
        let Ok(expected) = execute(candidate, FuzzPath::CpuOracle) else {
            return false;
        };
        let actual = match execute(candidate, path) {
            Ok(actual) => actual,
            Err(PathError::Failed { class, detail }) => FuzzOutcome::Error {
                class: class.into(),
                detail,
            },
            Err(PathError::Unsupported(_)) => return false,
        };
        !outcomes_match(&expected, &actual, policy_for(path, &expected))
            && mismatch_signature(&expected, &actual) == signature
    });
    let expected = execute(&minimized, FuzzPath::CpuOracle).map_err(PathError::detail)?;
    let actual = match execute(&minimized, path) {
        Ok(actual) => actual,
        Err(PathError::Unsupported(reason)) => {
            return Err(format!(
                "minimized case unexpectedly became unsupported on {path:?}: {reason}"
            ));
        }
        Err(error) => error.failure_outcome()?,
    };
    let policy = policy_for(path, &expected);
    if outcomes_match(&expected, &actual, policy)
        || mismatch_signature(&expected, &actual) != signature
    {
        return Err("minimizer did not preserve the mismatch signature".into());
    }
    Ok(FuzzComparison::Failure(Box::new(
        FuzzFailureArtifact::new(seed, case_index, minimized, path, policy, expected, actual)
            .map_err(|error| error.to_string())?,
    )))
}

fn compare_path(
    seed: u64,
    case_index: u64,
    case: &FuzzCase,
    path: FuzzPath,
) -> Result<FuzzComparison, String> {
    compare_path_with(seed, case_index, case, path, &mut execute_path)
}

/// Executes one case against captured interpreter and optionally strict native.
pub fn run_case(
    seed: u64,
    case_index: u64,
    case: &FuzzCase,
    native: bool,
) -> Result<Vec<FuzzComparison>, String> {
    case.validate()?;
    let mut comparisons = vec![compare_path(
        seed,
        case_index,
        case,
        FuzzPath::CapturedInterpreter,
    )?];
    if native {
        comparisons.push(compare_path(
            seed,
            case_index,
            case,
            FuzzPath::NativeScalar,
        )?);
    }
    Ok(comparisons)
}

/// Runs a fixed-count deterministic campaign in case-index order.
pub fn run_campaign(config: FuzzConfig) -> Result<FuzzCampaign, String> {
    if config.cases > MAX_CAMPAIGN_CASES {
        return Err(format!("campaign case count exceeds {MAX_CAMPAIGN_CASES}"));
    }
    let mut report = FuzzCampaign {
        seed: config.seed,
        generated: 0,
        interpreter_matches: 0,
        native_matches: 0,
        native_unsupported: 0,
        failures: vec![],
    };
    for index in 0..config.cases {
        let case = generate_case(config.seed, index);
        report.generated += 1;
        match run_case(config.seed, index, &case, config.native) {
            Ok(comparisons) => {
                for comparison in comparisons {
                    record_comparison(&mut report, index, comparison)?;
                }
            }
            Err(detail) => return Err(format!("generated case {index} was invalid: {detail}")),
        }
    }
    let interpreter_failures = report
        .failures
        .iter()
        .filter(|failure| failure.actual_path == FuzzPath::CapturedInterpreter)
        .count() as u64;
    if report.interpreter_matches + interpreter_failures != report.generated {
        return Err("captured interpreter campaign accounting is incomplete".into());
    }
    if config.native {
        let native_failures = report
            .failures
            .iter()
            .filter(|failure| {
                matches!(
                    failure.actual_path,
                    FuzzPath::NativeScalar | FuzzPath::NativeVector
                )
            })
            .count() as u64;
        if report.native_matches + report.native_unsupported + native_failures != report.generated {
            return Err("native campaign accounting is incomplete".into());
        }
    }
    Ok(report)
}

pub(super) fn replay_failure_with(
    artifact: &FuzzFailureArtifact,
    execute: &mut impl FnMut(&FuzzCase, FuzzPath) -> Result<FuzzOutcome, PathError>,
) -> Result<FuzzReplayStatus, String> {
    artifact.validate().map_err(|error| error.to_string())?;
    let expected = execute(&artifact.case, artifact.expected_path).map_err(PathError::detail)?;
    let actual = match execute(&artifact.case, artifact.actual_path) {
        Ok(actual) => actual,
        Err(PathError::Unsupported(reason)) => {
            return Ok(FuzzReplayStatus::Unsupported {
                path: artifact.actual_path,
                reason,
            });
        }
        Err(error) => error.failure_outcome()?,
    };
    if outcomes_match(&expected, &actual, artifact.policy) {
        return Ok(FuzzReplayStatus::Resolved);
    }
    if outcomes_match(&artifact.expected, &expected, artifact.policy)
        && outcomes_match(&artifact.actual, &actual, artifact.policy)
    {
        Ok(FuzzReplayStatus::Reproduced)
    } else {
        Ok(FuzzReplayStatus::Changed)
    }
}

/// Re-executes the artifact and classifies its current lifecycle state against
/// both recorded outcomes under the stored comparison contract.
pub fn replay_failure(artifact: &FuzzFailureArtifact) -> Result<FuzzReplayStatus, String> {
    replay_failure_with(artifact, &mut execute_path)
}
