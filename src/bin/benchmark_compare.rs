//! Combine normalized benchmark observations without executing a workload.

use rustgrad::{BenchmarkComparison, BenchmarkError, BenchmarkFramework, BenchmarkObservation};
use std::collections::BTreeSet;
use std::env;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str;

const USAGE: &str = "usage: benchmark_compare --baseline <rustgrad|tinygrad|candle|llama.cpp> [--output <new-file>] <observation.json> <observation.json> [...]";
const MAX_INPUT_COUNT: usize = 64;
const MAX_INPUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_TOTAL_INPUT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Eq, PartialEq)]
struct Args {
    baseline: BenchmarkFramework,
    output: Option<PathBuf>,
    inputs: Vec<PathBuf>,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, CliError> {
        let mut arguments = arguments.into_iter();
        let mut baseline = None;
        let mut output = None;
        let mut inputs = Vec::new();
        let mut saw_input = false;

        while let Some(argument) = arguments.next() {
            if saw_input && argument.starts_with('-') {
                return Err(CliError::OptionAfterInput(argument));
            }
            match argument.as_str() {
                "--baseline" => {
                    if baseline.is_some() {
                        return Err(CliError::DuplicateOption("--baseline"));
                    }
                    let value = option_value(&mut arguments, "--baseline")?;
                    baseline = Some(parse_framework(&value)?);
                }
                "--output" => {
                    if output.is_some() {
                        return Err(CliError::DuplicateOption("--output"));
                    }
                    let value = option_value(&mut arguments, "--output")?;
                    output = Some(PathBuf::from(value));
                }
                value if value.starts_with('-') => {
                    return Err(CliError::UnknownOption(value.to_owned()));
                }
                value => {
                    saw_input = true;
                    let path = PathBuf::from(value);
                    if inputs.contains(&path) {
                        return Err(CliError::DuplicateInput(path));
                    }
                    inputs.push(path);
                    if inputs.len() > MAX_INPUT_COUNT {
                        return Err(CliError::TooManyInputs {
                            maximum: MAX_INPUT_COUNT,
                        });
                    }
                }
            }
        }

        let baseline = baseline.ok_or(CliError::MissingBaseline)?;
        if inputs.len() < 2 {
            return Err(CliError::TooFewInputs);
        }
        if let Some(path) = &output {
            if inputs.contains(path) {
                return Err(CliError::OutputIsInput(path.clone()));
            }
        }
        Ok(Self {
            baseline,
            output,
            inputs,
        })
    }
}

fn option_value(
    arguments: &mut impl Iterator<Item = String>,
    option: &'static str,
) -> Result<String, CliError> {
    let value = arguments
        .next()
        .ok_or(CliError::MissingOptionValue(option))?;
    if value.is_empty() || value.starts_with('-') {
        Err(CliError::MissingOptionValue(option))
    } else {
        Ok(value)
    }
}

fn parse_framework(value: &str) -> Result<BenchmarkFramework, CliError> {
    match value {
        "rustgrad" => Ok(BenchmarkFramework::RustGrad),
        "tinygrad" => Ok(BenchmarkFramework::Tinygrad),
        "candle" => Ok(BenchmarkFramework::Candle),
        "llama.cpp" => Ok(BenchmarkFramework::LlamaCpp),
        _ => Err(CliError::InvalidBaseline(value.to_owned())),
    }
}

#[cfg(test)]
fn framework_name(value: BenchmarkFramework) -> &'static str {
    match value {
        BenchmarkFramework::RustGrad => "rustgrad",
        BenchmarkFramework::Tinygrad => "tinygrad",
        BenchmarkFramework::Candle => "candle",
        BenchmarkFramework::LlamaCpp => "llama.cpp",
    }
}

fn build_comparison(args: &Args) -> Result<Vec<u8>, CliError> {
    let observations = load_observations(&args.inputs)?;
    let comparison =
        BenchmarkComparison::new(args.baseline, observations).map_err(CliError::Comparison)?;
    comparison.to_json_bytes().map_err(CliError::Comparison)
}

fn load_observations(paths: &[PathBuf]) -> Result<Vec<BenchmarkObservation>, CliError> {
    let mut total_bytes = 0usize;
    let mut resolved_paths = BTreeSet::new();
    let mut observations = Vec::with_capacity(paths.len());
    for path in paths {
        let resolved = fs::canonicalize(path).map_err(|error| CliError::FileSystem {
            operation: "resolve input",
            path: path.clone(),
            kind: error.kind(),
        })?;
        if !resolved_paths.insert(resolved) {
            return Err(CliError::DuplicateInput(path.clone()));
        }
        let remaining = MAX_TOTAL_INPUT_BYTES.saturating_sub(total_bytes);
        let bytes = read_bounded(path, MAX_INPUT_BYTES, remaining)?;
        total_bytes = total_bytes
            .checked_add(bytes.len())
            .ok_or(CliError::TotalInputTooLarge {
                maximum: MAX_TOTAL_INPUT_BYTES,
            })?;
        str::from_utf8(&bytes).map_err(|_| CliError::InvalidUtf8(path.clone()))?;
        observations.push(
            BenchmarkObservation::from_json_bytes(&bytes).map_err(|error| {
                CliError::InvalidObservation {
                    path: path.clone(),
                    error,
                }
            })?,
        );
    }
    Ok(observations)
}

fn read_bounded(path: &Path, file_limit: usize, remaining: usize) -> Result<Vec<u8>, CliError> {
    let file = File::open(path).map_err(|error| CliError::FileSystem {
        operation: "open input",
        path: path.to_owned(),
        kind: error.kind(),
    })?;
    let read_limit = file_limit.min(remaining);
    let mut bytes = Vec::new();
    file.take(read_limit.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| CliError::FileSystem {
            operation: "read input",
            path: path.to_owned(),
            kind: error.kind(),
        })?;
    if bytes.len() > remaining {
        return Err(CliError::TotalInputTooLarge {
            maximum: MAX_TOTAL_INPUT_BYTES,
        });
    }
    if bytes.len() > file_limit {
        return Err(CliError::InputTooLarge {
            path: path.to_owned(),
            maximum: file_limit,
        });
    }
    Ok(bytes)
}

fn emit(args: &Args, bytes: &[u8]) -> Result<(), CliError> {
    if let Some(path) = &args.output {
        write_create_new(path, bytes)
    } else {
        let mut stdout = io::stdout().lock();
        stdout
            .write_all(bytes)
            .and_then(|()| stdout.flush())
            .map_err(|error| CliError::StandardOutput(error.kind()))
    }
}

fn write_create_new(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| CliError::FileSystem {
            operation: "create output",
            path: path.to_owned(),
            kind: error.kind(),
        })?;
    file.write_all(bytes)
        .and_then(|()| file.flush())
        .map_err(|error| CliError::FileSystem {
            operation: "write output",
            path: path.to_owned(),
            kind: error.kind(),
        })
}

#[derive(Debug)]
enum CliError {
    MissingBaseline,
    MissingOptionValue(&'static str),
    DuplicateOption(&'static str),
    UnknownOption(String),
    OptionAfterInput(String),
    InvalidBaseline(String),
    TooFewInputs,
    TooManyInputs {
        maximum: usize,
    },
    DuplicateInput(PathBuf),
    OutputIsInput(PathBuf),
    InputTooLarge {
        path: PathBuf,
        maximum: usize,
    },
    TotalInputTooLarge {
        maximum: usize,
    },
    InvalidUtf8(PathBuf),
    InvalidObservation {
        path: PathBuf,
        error: BenchmarkError,
    },
    Comparison(BenchmarkError),
    FileSystem {
        operation: &'static str,
        path: PathBuf,
        kind: io::ErrorKind,
    },
    StandardOutput(io::ErrorKind),
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBaseline => write!(formatter, "missing required --baseline; {USAGE}"),
            Self::MissingOptionValue(option) => {
                write!(formatter, "missing value for {option}; {USAGE}")
            }
            Self::DuplicateOption(option) => write!(formatter, "duplicate option {option}"),
            Self::UnknownOption(option) => write!(formatter, "unknown option {option}; {USAGE}"),
            Self::OptionAfterInput(option) => write!(
                formatter,
                "option {option} appears after an input path; options must precede inputs"
            ),
            Self::InvalidBaseline(value) => write!(
                formatter,
                "invalid baseline {value}; expected rustgrad, tinygrad, candle, or llama.cpp"
            ),
            Self::TooFewInputs => write!(formatter, "at least two observation files are required"),
            Self::TooManyInputs { maximum } => {
                write!(
                    formatter,
                    "too many observation files; maximum is {maximum}"
                )
            }
            Self::DuplicateInput(path) => {
                write!(formatter, "duplicate observation path {}", path.display())
            }
            Self::OutputIsInput(path) => write!(
                formatter,
                "output path is also an observation input: {}",
                path.display()
            ),
            Self::InputTooLarge { path, maximum } => write!(
                formatter,
                "observation file {} exceeds {maximum} bytes",
                path.display()
            ),
            Self::TotalInputTooLarge { maximum } => {
                write!(formatter, "observation inputs exceed {maximum} total bytes")
            }
            Self::InvalidUtf8(path) => {
                write!(
                    formatter,
                    "observation file {} is not UTF-8",
                    path.display()
                )
            }
            Self::InvalidObservation { path, error } => write!(
                formatter,
                "invalid observation file {}: {error}",
                path.display()
            ),
            Self::Comparison(error) => write!(formatter, "invalid comparison: {error}"),
            Self::FileSystem {
                operation,
                path,
                kind,
            } => write!(
                formatter,
                "failed to {operation} {}: {}",
                path.display(),
                io_kind_name(*kind)
            ),
            Self::StandardOutput(kind) => {
                write!(
                    formatter,
                    "failed to write standard output: {}",
                    io_kind_name(*kind)
                )
            }
        }
    }
}

fn io_kind_name(kind: io::ErrorKind) -> &'static str {
    match kind {
        io::ErrorKind::NotFound => "not found",
        io::ErrorKind::PermissionDenied => "permission denied",
        io::ErrorKind::AlreadyExists => "already exists",
        io::ErrorKind::InvalidData => "invalid data",
        io::ErrorKind::UnexpectedEof => "unexpected end of file",
        io::ErrorKind::WriteZero => "zero-byte write",
        io::ErrorKind::BrokenPipe => "broken pipe",
        _ => "I/O error",
    }
}

fn run(arguments: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let args = Args::parse(arguments)?;
    let bytes = build_comparison(&args)?;
    emit(&args, &bytes)
}

fn main() -> ExitCode {
    match run(env::args().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("benchmark_compare: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustgrad::{
        BenchmarkDevice, BenchmarkDuration, BenchmarkImplementation, BenchmarkMetrics,
        BenchmarkWorkload,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_PATH: AtomicUsize = AtomicUsize::new(0);

    fn arguments(values: &[&str]) -> impl Iterator<Item = String> {
        values.iter().map(|value| (*value).to_owned())
    }

    fn temporary_path(label: &str) -> PathBuf {
        env::temp_dir().join(format!(
            "rustgrad-benchmark-compare-{label}-{}-{}.json",
            std::process::id(),
            NEXT_PATH.fetch_add(1, Ordering::Relaxed)
        ))
    }

    struct TemporaryFile(PathBuf);

    impl TemporaryFile {
        fn write(label: &str, bytes: &[u8]) -> Self {
            let path = temporary_path(label);
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            file.write_all(bytes).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TemporaryFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn sha(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn observation(
        framework: BenchmarkFramework,
        workload: BenchmarkWorkload,
    ) -> BenchmarkObservation {
        BenchmarkObservation::new(
            BenchmarkImplementation {
                framework,
                version: "1.0".into(),
                revision: "revision".into(),
                configuration: "release; batch=1".into(),
                command: "offline fixture".into(),
            },
            workload,
            BenchmarkDevice {
                backend: "metal".into(),
                name: "Apple GPU".into(),
                hardware_identity: "registry:42".into(),
                operating_system: "macOS fixture".into(),
            },
            BenchmarkMetrics {
                planning_time: Some(BenchmarkDuration::new(0, 1).unwrap()),
                pipeline_compile_time: None,
                native_prepare_time: None,
                first_run_latency: None,
                steady_run_latency: None,
                prompt_prefill: None,
                steady_decode: None,
                planned_device_memory_bytes: None,
                measured_peak_device_memory_bytes: None,
                planned_kernel_count: None,
                executed_kernel_count: None,
                host_to_device: None,
                device_to_host: None,
                fallback_count: None,
            },
        )
        .unwrap()
    }

    fn workload(input_hash: char) -> BenchmarkWorkload {
        BenchmarkWorkload::ResNet18 {
            model_identity: "resnet18-seed-19".into(),
            input_shape: [1, 3, 224, 224],
            input_dtype: "f32".into(),
            input_sha256: sha(input_hash),
            correctness_contract: "full logits within declared tolerance".into(),
        }
    }

    fn observation_file(
        label: &str,
        framework: BenchmarkFramework,
        workload: BenchmarkWorkload,
    ) -> TemporaryFile {
        TemporaryFile::write(
            label,
            &observation(framework, workload).to_json_bytes().unwrap(),
        )
    }

    #[test]
    fn parser_requires_unambiguous_unique_options_and_inputs() {
        assert!(matches!(
            Args::parse(arguments(&["a", "b"])),
            Err(CliError::MissingBaseline)
        ));
        assert!(matches!(
            Args::parse(arguments(&[
                "--baseline",
                "rustgrad",
                "--baseline",
                "candle",
                "a",
                "b"
            ])),
            Err(CliError::DuplicateOption("--baseline"))
        ));
        assert!(matches!(
            Args::parse(arguments(&[
                "--baseline",
                "rustgrad",
                "--output",
                "first",
                "--output",
                "second",
                "a",
                "b"
            ])),
            Err(CliError::DuplicateOption("--output"))
        ));
        assert!(matches!(
            Args::parse(arguments(&[
                "--baseline",
                "rustgrad",
                "a",
                "--output",
                "out",
                "b"
            ])),
            Err(CliError::OptionAfterInput(_))
        ));
        assert!(matches!(
            Args::parse(arguments(&["--baseline", "rustgrad", "a", "a"])),
            Err(CliError::DuplicateInput(_))
        ));
        assert!(matches!(
            Args::parse(arguments(&["--baseline", "rustgrad", "a"])),
            Err(CliError::TooFewInputs)
        ));

        let mut too_many = vec!["--baseline".to_owned(), "rustgrad".to_owned()];
        too_many.extend((0..=MAX_INPUT_COUNT).map(|index| format!("input-{index}")));
        assert!(matches!(
            Args::parse(too_many),
            Err(CliError::TooManyInputs {
                maximum: MAX_INPUT_COUNT
            })
        ));
    }

    #[test]
    fn framework_names_are_exact_and_round_trip() {
        for (name, framework) in [
            ("rustgrad", BenchmarkFramework::RustGrad),
            ("tinygrad", BenchmarkFramework::Tinygrad),
            ("candle", BenchmarkFramework::Candle),
            ("llama.cpp", BenchmarkFramework::LlamaCpp),
        ] {
            assert_eq!(parse_framework(name).unwrap(), framework);
            assert_eq!(framework_name(framework), name);
        }
        assert!(matches!(
            parse_framework("llamacpp"),
            Err(CliError::InvalidBaseline(_))
        ));
    }

    #[test]
    fn comparison_output_is_deterministic_and_canonically_ordered() {
        let candle = observation_file("candle", BenchmarkFramework::Candle, workload('a'));
        let rustgrad = observation_file("rustgrad", BenchmarkFramework::RustGrad, workload('a'));
        let args = Args {
            baseline: BenchmarkFramework::RustGrad,
            output: None,
            inputs: vec![candle.0.clone(), rustgrad.0.clone()],
        };
        let first = build_comparison(&args).unwrap();
        let second = build_comparison(&args).unwrap();
        let comparison = BenchmarkComparison::from_json_bytes(&first).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.last(), Some(&b'\n'));
        assert_eq!(
            comparison.observations[0].implementation.framework,
            BenchmarkFramework::RustGrad
        );
        assert_eq!(
            comparison.observations[1].implementation.framework,
            BenchmarkFramework::Candle
        );
    }

    #[test]
    fn malformed_and_oversized_inputs_are_rejected() {
        let malformed = TemporaryFile::write("malformed", b"{");
        let valid = observation_file("valid", BenchmarkFramework::Candle, workload('a'));
        assert!(matches!(
            read_bounded(malformed.path(), 8, 8),
            Ok(bytes) if bytes == b"{"
        ));
        let invalid = Args {
            baseline: BenchmarkFramework::RustGrad,
            output: None,
            inputs: vec![malformed.0.clone(), valid.0.clone()],
        };
        assert!(matches!(
            build_comparison(&invalid),
            Err(CliError::InvalidObservation { .. })
        ));

        let invalid_utf8 = TemporaryFile::write("invalid-utf8", &[0xff]);
        let invalid = Args {
            baseline: BenchmarkFramework::Candle,
            output: None,
            inputs: vec![invalid_utf8.0.clone(), valid.0.clone()],
        };
        assert!(matches!(
            build_comparison(&invalid),
            Err(CliError::InvalidUtf8(_))
        ));

        let oversized = TemporaryFile::write("oversized", b"123456789");
        assert!(matches!(
            read_bounded(oversized.path(), 8, 16),
            Err(CliError::InputTooLarge { maximum: 8, .. })
        ));
        assert!(matches!(
            read_bounded(oversized.path(), 16, 8),
            Err(CliError::TotalInputTooLarge { .. })
        ));
    }

    #[test]
    fn workload_mismatch_is_propagated() {
        let rustgrad = observation_file("mismatch-a", BenchmarkFramework::RustGrad, workload('a'));
        let candle = observation_file("mismatch-b", BenchmarkFramework::Candle, workload('b'));
        let args = Args {
            baseline: BenchmarkFramework::RustGrad,
            output: None,
            inputs: vec![rustgrad.0.clone(), candle.0.clone()],
        };
        assert!(matches!(
            build_comparison(&args),
            Err(CliError::Comparison(BenchmarkError::WorkloadMismatch))
        ));
    }

    #[test]
    fn output_is_create_new_and_never_overwrites() {
        let output = temporary_path("output");
        write_create_new(&output, b"first").unwrap();
        assert!(matches!(
            write_create_new(&output, b"second"),
            Err(CliError::FileSystem {
                operation: "create output",
                kind: io::ErrorKind::AlreadyExists,
                ..
            })
        ));
        assert_eq!(fs::read(&output).unwrap(), b"first");
        fs::remove_file(output).unwrap();
    }
}
