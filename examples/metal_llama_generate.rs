//! Generate one bounded response from a supported local GGUF on Apple Metal.

use rustgrad::models::transformer::{LlamaChatMessage, LlamaChatRole};
use rustgrad::runtime::metal::{
    MetalDeviceInfo, MetalDeviceSessionSummary, MetalDiscovery, MetalPlanOptions, MetalRuntime,
    MetalScoreboardContext,
};
use rustgrad::{
    LlamaMetalGeneration, LlamaMetalGenerationError, LlamaMetalGenerationStage, LlamaMetalPlan,
    LlamaPromptWorkflow, LlamaSampling, ReplayInput,
};
use std::{
    env,
    error::Error,
    fmt,
    fs::OpenOptions,
    io::{self, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

const USAGE: &str = "usage: metal_llama_generate [--device INDEX] [--expected-registry-id ID] [--max-new-tokens N] [--chat] [--scoreboard PATH --revision LABEL] [--expected-ids ID,ID,...] [--] <model.gguf> <prompt>";
const DEFAULT_MAX_NEW_TOKENS: usize = 16;
const MAX_NEW_TOKENS: usize = 4_096;
const SCOREBOARD_WORKLOAD: &str = "gguf-llama-metal-generate";
const SCOREBOARD_EVIDENCE: &str = "live self-hosted Apple GPU prompt-to-tokens harness";

#[derive(Debug, Eq, PartialEq)]
struct Args {
    device_index: usize,
    expected_registry_id: Option<u64>,
    max_new_tokens: usize,
    chat: bool,
    scoreboard_path: Option<PathBuf>,
    revision: Option<String>,
    expected_ids: Option<Vec<u32>>,
    model_path: PathBuf,
    prompt: String,
}

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CliError {}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StablePlanFacts {
    device_info: MetalDeviceInfo,
    device_owner_id: u64,
    step_deployment_identity: u64,
    capture_identity: u64,
    summary: MetalDeviceSessionSummary,
    resident_inputs: Vec<ReplayInput>,
    state_inputs: Vec<ReplayInput>,
    transient_inputs: Vec<ReplayInput>,
    runtime_control_inputs: Vec<ReplayInput>,
    cache_keys: Vec<String>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("metal Llama generation failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args(env::args().skip(1))?;
    let scoreboard_context = match (&args.scoreboard_path, &args.revision) {
        (Some(_), Some(revision)) => Some(MetalScoreboardContext::new(
            SCOREBOARD_WORKLOAD,
            revision,
            SCOREBOARD_EVIDENCE,
        )?),
        (None, None) => None,
        _ => unreachable!("paired scoreboard arguments are validated by parse_args"),
    };
    if let Some(path) = &args.scoreboard_path
        && path.try_exists()?
    {
        return Err(io::Error::other("scoreboard path already exists").into());
    }
    let runtime = MetalRuntime::load()?;
    let device = match runtime.discover()? {
        MetalDiscovery::Devices(devices) => devices
            .into_iter()
            .nth(args.device_index)
            .ok_or_else(|| io::Error::other("selected Metal device index is out of range"))?,
        MetalDiscovery::NoDevices => {
            return Err(
                io::Error::other("generation requires a process-visible Metal device").into(),
            );
        }
    };
    if args
        .expected_registry_id
        .is_some_and(|expected| device.info().registry_id != expected)
    {
        return Err(io::Error::other("selected Metal device registry ID does not match").into());
    }
    let cache = device.cache();
    let cache_entries_before_prepare = cache.len();
    let workflow = LlamaPromptWorkflow::from_path(&args.model_path)?;
    let plan = LlamaMetalPlan::from_workflow(workflow, &device, MetalPlanOptions::default())?;
    let stable = StablePlanFacts {
        device_info: plan.selected_device_info().clone(),
        device_owner_id: plan.selected_device_owner_id(),
        step_deployment_identity: plan.step_deployment_identity(),
        capture_identity: plan.capture().identity,
        summary: plan.summary().clone(),
        resident_inputs: plan.resident_inputs().to_vec(),
        state_inputs: plan.state_inputs().to_vec(),
        transient_inputs: plan.transient_inputs().to_vec(),
        runtime_control_inputs: plan.runtime_control_inputs().to_vec(),
        cache_keys: plan
            .rendered_items()
            .filter(|rendered| rendered.extent != 0)
            .map(|rendered| rendered.cache_key.clone())
            .collect(),
    };
    if stable.summary.fallback_count != 0 {
        return Err(io::Error::other("Metal Llama plan admitted a fallback path").into());
    }
    eprintln!(
        "device={} registry_id={} deployment={:016x} capture={:016x} kernels={} planned_device_bytes={}",
        stable.device_info.name,
        stable.device_info.registry_id,
        stable.step_deployment_identity,
        stable.capture_identity,
        stable.summary.nonzero_item_count,
        stable.summary.planned_device_bytes,
    );

    let mut session = match scoreboard_context {
        Some(context) => plan.prepare_with_scoreboard(context)?,
        None => plan.prepare()?,
    };
    validate_stable_session(&session, &stable)?;
    let cache_entries_after_prepare = cache.len();
    let preparation = session.preparation_report().clone();
    if cache_entries_after_prepare
        != cache_entries_before_prepare
            .checked_add(preparation.pipeline_cache_miss_count)
            .ok_or_else(|| io::Error::other("Metal pipeline cache count overflow"))?
    {
        return Err(
            io::Error::other("Metal pipeline cache accounting changed during prepare").into(),
        );
    }
    eprintln!(
        "prepared cache_requests={} cache_hits={} cache_misses={} resident_h2d_calls={} resident_h2d_bytes={}",
        preparation.pipeline_cache_request_count,
        preparation.pipeline_cache_hit_count,
        preparation.pipeline_cache_miss_count,
        preparation.resident_h2d_calls,
        preparation.resident_h2d_bytes,
    );

    let output = if args.chat {
        let messages = [LlamaChatMessage::new(
            LlamaChatRole::User,
            args.prompt.as_str(),
        )?];
        session.generate_chat(&messages, args.max_new_tokens, LlamaSampling::Greedy)
    } else {
        session.generate_text(&args.prompt, args.max_new_tokens, LlamaSampling::Greedy)
    }
    .map_err(generation_error)?;
    let generation = output.generation();
    validate_generation(
        generation,
        args.max_new_tokens,
        args.expected_ids.as_deref(),
        session.position(),
        session.vocab_size(),
    )?;
    validate_stable_session(&session, &stable)?;
    if cache.len() != cache_entries_after_prepare {
        return Err(io::Error::other("Metal pipeline cache changed during generation").into());
    }

    println!("{}", generation.decoded());
    if let Some(error) = session.scoreboard_recording_error() {
        return Err(io::Error::other(format!(
            "generation succeeded but scoreboard recording failed: {error}"
        ))
        .into());
    }
    if let Some(path) = args.scoreboard_path {
        let report = session
            .execution_scoreboard()
            .ok_or_else(|| io::Error::other("scoreboard was requested but is not bound"))?
            .report()?;
        if report.fallback_count != 0
            || report.deployment_identity != stable.step_deployment_identity
            || report.capture_identity != stable.capture_identity
            || report.rendered_cache_keys != stable.cache_keys
            || report.successful_run_count != u64::try_from(generation.reports().len())?
            || report.successful_runs.len() != generation.reports().len()
            || report.committed_state_position != Some(generation.reports().len())
            || report.successful_runs.iter().zip(generation.reports()).any(
                |(recorded, executed)| {
                    recorded.successful_invocation != executed.successful_invocation
                        || recorded.committed_state_position != executed.committed_state_position
                        || recorded.transient_host_api_h2d_calls != executed.transient_h2d_calls
                        || recorded.runtime_control_host_api_h2d_calls
                            != executed.runtime_control_h2d_calls
                        || recorded.retained_host_api_d2h_calls != executed.retained_d2h_calls
                        || recorded.kernel_launch_count != executed.kernel_launch_count
                },
            )
        {
            return Err(io::Error::other("Metal Llama scoreboard evidence is inconsistent").into());
        }
        write_new_evidence(&path, &report.to_json_bytes()?)?;
    }
    eprintln!(
        "completed prompt_tokens={} generated_tokens={} committed_position={} invocations={}",
        generation.prompt_ids().len(),
        generation.generated_ids().len(),
        session.position(),
        generation.reports().len(),
    );
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args, CliError> {
    let mut device_index = None;
    let mut expected_registry_id = None;
    let mut max_new_tokens = None;
    let mut chat = false;
    let mut scoreboard_path = None;
    let mut revision = None;
    let mut expected_ids = None;
    let mut positional = Vec::new();
    let mut args = args.into_iter();
    let mut positional_only = false;
    while let Some(argument) = args.next() {
        if positional_only {
            positional.push(argument);
            continue;
        }
        match argument.as_str() {
            "--" => positional_only = true,
            "--device" => set_once(
                &mut device_index,
                parse_usize(args.next(), "--device")?,
                "--device",
            )?,
            "--expected-registry-id" => set_once(
                &mut expected_registry_id,
                parse_u64(args.next(), "--expected-registry-id")?,
                "--expected-registry-id",
            )?,
            "--max-new-tokens" => set_once(
                &mut max_new_tokens,
                parse_usize(args.next(), "--max-new-tokens")?,
                "--max-new-tokens",
            )?,
            "--chat" => {
                if chat {
                    return Err(cli("duplicate --chat"));
                }
                chat = true;
            }
            "--scoreboard" => set_once(
                &mut scoreboard_path,
                PathBuf::from(required_value(args.next(), "--scoreboard")?),
                "--scoreboard",
            )?,
            "--revision" => set_once(
                &mut revision,
                required_value(args.next(), "--revision")?,
                "--revision",
            )?,
            "--expected-ids" => set_once(
                &mut expected_ids,
                parse_expected_ids(&required_value(args.next(), "--expected-ids")?)?,
                "--expected-ids",
            )?,
            value if value.starts_with('-') => {
                return Err(cli(format!("unknown flag {value}")));
            }
            value => positional.push(value.to_owned()),
        }
    }
    if positional.len() != 2 {
        return Err(cli(USAGE));
    }
    if scoreboard_path.is_some() != revision.is_some() {
        return Err(cli("--scoreboard and --revision must be supplied together"));
    }
    let max_new_tokens = max_new_tokens.unwrap_or(DEFAULT_MAX_NEW_TOKENS);
    if max_new_tokens > MAX_NEW_TOKENS {
        return Err(cli(format!(
            "--max-new-tokens must not exceed {MAX_NEW_TOKENS}"
        )));
    }
    if expected_ids
        .as_ref()
        .is_some_and(|ids| ids.len() > max_new_tokens)
    {
        return Err(cli("--expected-ids is longer than --max-new-tokens"));
    }
    let prompt = positional.pop().expect("length checked");
    let model_path = PathBuf::from(positional.pop().expect("length checked"));
    Ok(Args {
        device_index: device_index.unwrap_or(0),
        expected_registry_id,
        max_new_tokens,
        chat,
        scoreboard_path,
        revision,
        expected_ids,
        model_path,
        prompt,
    })
}

fn required_value(value: Option<String>, flag: &'static str) -> Result<String, CliError> {
    match value {
        Some(value) if !value.is_empty() && !value.starts_with('-') => Ok(value),
        _ => Err(cli(format!("{flag} requires a value"))),
    }
}

fn parse_usize(value: Option<String>, flag: &'static str) -> Result<usize, CliError> {
    required_value(value, flag)?
        .parse::<usize>()
        .map_err(|_| cli(format!("{flag} requires a nonnegative integer")))
}

fn parse_u64(value: Option<String>, flag: &'static str) -> Result<u64, CliError> {
    required_value(value, flag)?
        .parse::<u64>()
        .map_err(|_| cli(format!("{flag} requires an unsigned integer")))
}

fn parse_expected_ids(value: &str) -> Result<Vec<u32>, CliError> {
    value
        .split(',')
        .map(|part| {
            if part.is_empty() {
                return Err(cli("--expected-ids requires comma-separated u32 values"));
            }
            part.parse::<u32>()
                .map_err(|_| cli("--expected-ids requires comma-separated u32 values"))
        })
        .collect()
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &'static str) -> Result<(), CliError> {
    if slot.replace(value).is_some() {
        Err(cli(format!("duplicate {flag}")))
    } else {
        Ok(())
    }
}

fn cli(message: impl Into<String>) -> CliError {
    CliError(format!("{}\n{USAGE}", message.into()))
}

fn validate_stable_session(
    session: &rustgrad::LlamaMetalSession,
    expected: &StablePlanFacts,
) -> Result<(), io::Error> {
    let cache_keys = session
        .compiled_kernels()
        .map(|kernel| kernel.cache_key.as_str())
        .collect::<Vec<_>>();
    if session.device_info() != &expected.device_info
        || session.device_owner_id() != expected.device_owner_id
        || session.capture().identity != expected.capture_identity
        || session.summary() != &expected.summary
        || session.resident_inputs() != expected.resident_inputs
        || session.state_inputs() != expected.state_inputs
        || session.transient_inputs() != expected.transient_inputs
        || session.runtime_control_inputs() != expected.runtime_control_inputs
        || cache_keys
            .into_iter()
            .ne(expected.cache_keys.iter().map(String::as_str))
    {
        return Err(io::Error::other(
            "persistent Metal deployment, cache, or resident ownership changed",
        ));
    }
    Ok(())
}

fn write_new_evidence(path: &Path, bytes: &[u8]) -> Result<(), io::Error> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    Ok(())
}

fn validate_generation(
    generation: &LlamaMetalGeneration,
    max_new_tokens: usize,
    expected_ids: Option<&[u32]>,
    committed_position: usize,
    vocab_size: usize,
) -> Result<(), io::Error> {
    if let Some(expected) = expected_ids
        && generation.generated_ids() != expected
    {
        return Err(io::Error::other(format!(
            "generated token IDs differ: expected {expected:?}, got {:?}",
            generation.generated_ids()
        )));
    }
    if generation.generated_ids().len() > max_new_tokens {
        return Err(io::Error::other("generation exceeded its token bound"));
    }
    let expected_reports = if max_new_tokens == 0 {
        0
    } else {
        generation
            .prompt_ids()
            .len()
            .checked_add(generation.generated_ids().len())
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| io::Error::other("generation report count overflow"))?
    };
    if generation.reports().len() != expected_reports || committed_position != expected_reports {
        return Err(io::Error::other(
            "generation omitted or invented a token invocation report",
        ));
    }
    let retained_bytes = vocab_size
        .checked_mul(4)
        .ok_or_else(|| io::Error::other("retained logits byte count overflow"))?;
    for (index, report) in generation.reports().iter().enumerate() {
        let ordinal = u64::try_from(index + 1)
            .map_err(|_| io::Error::other("generation invocation ordinal overflow"))?;
        let retained = if index + 1 >= generation.prompt_ids().len() {
            1
        } else {
            0
        };
        if report.successful_invocation != ordinal
            || report.committed_state_position != Some(index + 1)
            || report.transient_h2d_calls != 1
            || report.transient_h2d_bytes != 4
            || report.runtime_control_h2d_calls != 1
            || report.runtime_control_h2d_bytes != 4
            || report.retained_d2h_calls != retained
            || (retained == 0 && report.retained_d2h_bytes != 0)
            || report.retained_d2h_bytes != retained * retained_bytes
            || report.output_count != retained
        {
            return Err(io::Error::other(format!(
                "token invocation {} violates the persistent Metal report contract",
                index + 1
            )));
        }
    }
    Ok(())
}

fn generation_error(error: LlamaMetalGenerationError) -> io::Error {
    let message = match &error {
        LlamaMetalGenerationError::Execution {
            progress,
            stage,
            token_offset,
            token,
            source,
        } => partial_message(*stage, *token_offset, Some(*token), progress, source),
        LlamaMetalGenerationError::Decode {
            progress,
            stage,
            token_offset,
            token,
            source,
        } => partial_message(*stage, *token_offset, Some(*token), progress, source),
        LlamaMetalGenerationError::PostExecution {
            progress,
            stage,
            token_offset,
            source,
        } => partial_message(*stage, *token_offset, None, progress, source),
        _ => error.to_string(),
    };
    io::Error::other(message)
}

fn partial_message(
    stage: LlamaMetalGenerationStage,
    token_offset: usize,
    token: Option<u32>,
    progress: &rustgrad::LlamaMetalProgress,
    source: &dyn fmt::Debug,
) -> String {
    format!(
        "partial failure stage={stage:?} token_offset={token_offset} token={} committed_position={} generated_ids={:?} successful_reports={} source={source:?}; no automatic resume attempted",
        token.map_or_else(|| "unselected".to_owned(), |token| token.to_string()),
        progress.committed_position(),
        progress.generated_ids(),
        progress.reports().len(),
    )
}

#[cfg(test)]
mod tests {
    use super::{Args, parse_args, write_new_evidence};
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

    fn parse(values: &[&str]) -> Result<Args, super::CliError> {
        parse_args(values.iter().map(|value| (*value).to_owned()))
    }

    #[test]
    fn parser_accepts_plain_chat_and_complete_evidence_modes() {
        assert_eq!(
            parse(&["model.gguf", "hello"]).unwrap(),
            Args {
                device_index: 0,
                expected_registry_id: None,
                max_new_tokens: 16,
                chat: false,
                scoreboard_path: None,
                revision: None,
                expected_ids: None,
                model_path: PathBuf::from("model.gguf"),
                prompt: "hello".into(),
            }
        );
        let evidence = parse(&[
            "--device",
            "2",
            "--expected-registry-id",
            "1234",
            "--max-new-tokens",
            "3",
            "--chat",
            "--scoreboard",
            "scoreboard.json",
            "--revision",
            "reviewed-sha",
            "--expected-ids",
            "4,5,6",
            "model.gguf",
            "hello",
        ])
        .unwrap();
        assert_eq!(evidence.device_index, 2);
        assert_eq!(evidence.expected_registry_id, Some(1234));
        assert_eq!(evidence.max_new_tokens, 3);
        assert!(evidence.chat);
        assert_eq!(evidence.expected_ids, Some(vec![4, 5, 6]));
    }

    #[test]
    fn parser_rejects_duplicates_malformed_values_and_unpaired_evidence() {
        for values in [
            vec!["--chat", "--chat", "m", "p"],
            vec!["--device", "x", "m", "p"],
            vec!["--expected-registry-id", "x", "m", "p"],
            vec!["--max-new-tokens", "4097", "m", "p"],
            vec!["--expected-ids", "1,,2", "m", "p"],
            vec!["--max-new-tokens", "1", "--expected-ids", "1,2", "m", "p"],
            vec!["--scoreboard", "out.json", "m", "p"],
            vec!["--revision", "sha", "m", "p"],
            vec!["--unknown", "m", "p"],
            vec!["m"],
            vec!["m", "p", "extra"],
        ] {
            assert!(parse(&values).is_err(), "accepted {values:?}");
        }
        let terminated = parse(&["--", "-model.gguf", "-prompt"]).unwrap();
        assert_eq!(terminated.model_path, PathBuf::from("-model.gguf"));
        assert_eq!(terminated.prompt, "-prompt");
    }

    #[test]
    fn evidence_creation_never_overwrites_a_previous_run() {
        let path = std::env::temp_dir().join(format!(
            "rustgrad-metal-llama-evidence-{}-{}.json",
            std::process::id(),
            NEXT_PATH.fetch_add(1, Ordering::Relaxed)
        ));
        let first = br#"{"format_version":4,"run":"first"}\n"#;
        write_new_evidence(&path, first).unwrap();
        assert_eq!(fs::read(&path).unwrap(), first);
        assert!(write_new_evidence(&path, b"replacement").is_err());
        assert_eq!(fs::read(&path).unwrap(), first);
        fs::remove_file(path).unwrap();
    }
}
