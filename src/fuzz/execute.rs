use super::{
    FuzzCase, FuzzComparisonPolicy, FuzzFailureArtifact, FuzzOutcome, FuzzPath, generate_case,
    minimize_case,
};
use crate::{
    Backend, CapturedBackendPolicy, CapturedReplayExecutor, CapturedReplayOptions,
    CapturedSchedule, CpuBackend, DType, ReplayError, schedule,
};

enum PathError {
    Unsupported(String),
    Failed { class: &'static str, detail: String },
}
impl PathError {
    fn detail(self) -> String {
        match self {
            Self::Unsupported(value) | Self::Failed { detail: value, .. } => value,
        }
    }
    fn outcome(self) -> FuzzOutcome {
        match self {
            Self::Unsupported(detail) => FuzzOutcome::Error {
                class: "unsupported".into(),
                detail,
            },
            Self::Failed { class, detail } => FuzzOutcome::Error {
                class: class.into(),
                detail,
            },
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

fn outcomes_match(
    expected: &FuzzOutcome,
    actual: &FuzzOutcome,
    policy: FuzzComparisonPolicy,
) -> bool {
    match (expected, actual, policy) {
        (
            FuzzOutcome::Value { tensor: lhs },
            FuzzOutcome::Value { tensor: rhs },
            FuzzComparisonPolicy::ExactBytes,
        ) => lhs == rhs,
        (
            FuzzOutcome::Value { tensor: lhs },
            FuzzOutcome::Value { tensor: rhs },
            FuzzComparisonPolicy::FloatTolerance {
                absolute_bits,
                relative_bits,
            },
        ) => {
            if lhs.shape != rhs.shape || lhs.dtype != rhs.dtype {
                return false;
            }
            let Ok(lhs) = lhs.to_tensor() else {
                return false;
            };
            let Ok(rhs) = rhs.to_tensor() else {
                return false;
            };
            let absolute = f64::from_bits(absolute_bits);
            let relative = f64::from_bits(relative_bits);
            lhs.to_vec_f64()
                .into_iter()
                .zip(rhs.to_vec_f64())
                .all(|(a, b)| {
                    a.to_bits() == b.to_bits()
                        || (a.is_nan() && b.is_nan())
                        || (a - b).abs() <= absolute + relative * a.abs().max(b.abs())
                })
        }
        (
            FuzzOutcome::Error {
                class: ac,
                detail: ad,
            },
            FuzzOutcome::Error {
                class: bc,
                detail: bd,
            },
            _,
        ) => ac == bc && ad == bd,
        _ => false,
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
                Ok(result) => Ok(FuzzOutcome::value(&result.outputs[0])),
                Err(error) => Err(replay_error(error)),
            }
        }
    }
}

fn path_outcome(case: &FuzzCase, path: FuzzPath) -> FuzzOutcome {
    execute_path(case, path).unwrap_or_else(PathError::outcome)
}

fn compare_path(
    seed: u64,
    case_index: u64,
    case: &FuzzCase,
    path: FuzzPath,
) -> Result<FuzzComparison, String> {
    let expected = execute_path(case, FuzzPath::CpuOracle).map_err(PathError::detail)?;
    let actual = match execute_path(case, path) {
        Ok(actual) => actual,
        Err(PathError::Unsupported(reason)) => {
            return Ok(FuzzComparison::Unsupported { path, reason });
        }
        Err(error) => error.outcome(),
    };
    let policy = policy_for(path, &expected);
    if outcomes_match(&expected, &actual, policy) {
        return Ok(FuzzComparison::Match { path, policy });
    }
    let minimized = minimize_case(case, |candidate| {
        let expected = path_outcome(candidate, FuzzPath::CpuOracle);
        let actual = path_outcome(candidate, path);
        !outcomes_match(&expected, &actual, policy_for(path, &expected))
    });
    let expected = path_outcome(&minimized, FuzzPath::CpuOracle);
    let actual = path_outcome(&minimized, path);
    let policy = policy_for(path, &expected);
    Ok(FuzzComparison::Failure(Box::new(
        FuzzFailureArtifact::new(seed, case_index, minimized, path, policy, expected, actual)
            .map_err(|error| error.to_string())?,
    )))
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
                    match comparison {
                        FuzzComparison::Match {
                            path: FuzzPath::CapturedInterpreter,
                            ..
                        } => report.interpreter_matches += 1,
                        FuzzComparison::Match { .. } => report.native_matches += 1,
                        FuzzComparison::Unsupported { .. } => report.native_unsupported += 1,
                        FuzzComparison::Failure(failure) => report.failures.push(*failure),
                    }
                }
            }
            Err(detail) => return Err(format!("generated case {index} was invalid: {detail}")),
        }
    }
    Ok(report)
}

/// Re-executes the artifact's minimized case and returns whether its recorded
/// mismatch still reproduces under the stored comparison contract.
pub fn replay_failure(artifact: &FuzzFailureArtifact) -> Result<bool, String> {
    artifact.validate().map_err(|error| error.to_string())?;
    let expected =
        execute_path(&artifact.case, artifact.expected_path).map_err(PathError::detail)?;
    let actual = path_outcome(&artifact.case, artifact.actual_path);
    Ok(!outcomes_match(&expected, &actual, artifact.policy))
}
