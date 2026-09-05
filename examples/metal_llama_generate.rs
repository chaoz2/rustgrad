//! Generate one bounded response from a supported local GGUF on Apple Metal.

use rustgrad::models::transformer::{LlamaChatMessage, LlamaChatRole};
use rustgrad::runtime::metal::{
    MetalDeviceInfo, MetalDeviceSessionSummary, MetalDiscovery, MetalPlanOptions, MetalRuntime,
    MetalScoreboardContext, MetalSessionScoreboardReport,
};
use rustgrad::{
    LlamaMetalGeneration, LlamaMetalGenerationError, LlamaMetalGenerationStage, LlamaMetalPlan,
    LlamaPromptWorkflow, LlamaSampling, ReplayInput,
};
use serde::Serialize;
use std::{
    env,
    error::Error,
    fmt,
    fs::{self, OpenOptions},
    io::{self, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    process::ExitCode,
};

const USAGE: &str = "usage: metal_llama_generate [--device INDEX] [--expected-registry-id ID] [--max-new-tokens N] [--prefill-span N] [--chat] [--scoreboard PATH --revision LABEL] [--expected-ids ID,ID,...] [--attestation PATH --model-sha256 HEX --model-source LOCATOR --model-license LICENSE --model-conversion PROVENANCE --oracle-name NAME --oracle-revision REVISION --oracle-command COMMAND --workflow-run-url URL --workflow-run-id ID] [--] <model.gguf> <prompt>";
const DEFAULT_MAX_NEW_TOKENS: usize = 16;
const MAX_NEW_TOKENS: usize = 4_096;
const SCOREBOARD_WORKLOAD: &str = "gguf-llama-metal-generate";
const SCOREBOARD_EVIDENCE: &str = "live self-hosted Apple GPU prompt-to-tokens harness";

#[derive(Debug, Eq, PartialEq)]
struct Args {
    device_index: usize,
    expected_registry_id: Option<u64>,
    max_new_tokens: usize,
    prefill_span: Option<NonZeroUsize>,
    chat: bool,
    scoreboard_path: Option<PathBuf>,
    revision: Option<String>,
    expected_ids: Option<Vec<u32>>,
    attestation: Option<AttestationArgs>,
    model_path: PathBuf,
    prompt: String,
}

#[derive(Debug, Eq, PartialEq)]
struct AttestationArgs {
    path: PathBuf,
    model_sha256: String,
    model_source: String,
    model_license: String,
    model_conversion: String,
    oracle_name: String,
    oracle_revision: String,
    oracle_command: String,
    workflow_run_url: String,
    workflow_run_id: String,
}

#[derive(Debug)]
struct AttestedModelFile {
    filename: String,
    size_bytes: u64,
}

struct AttestationRecordContext<'a> {
    args: &'a Args,
    provenance: &'a AttestationArgs,
    model: &'a AttestedModelFile,
    stable: &'a StablePlanFacts,
    generation: &'a LlamaMetalGeneration,
    report: &'a MetalSessionScoreboardReport,
    final_position: usize,
    scoreboard_path: &'a Path,
}

#[derive(Serialize)]
struct MetalLlamaAttestation<'a> {
    format_version: u32,
    code_revision_sha: &'a str,
    model: AttestedModel<'a>,
    device: AttestedDevice<'a>,
    invocation: AttestedInvocation<'a>,
    oracle: AttestedOracle<'a>,
    evidence: AttestedEvidence<'a>,
}

#[derive(Serialize)]
struct AttestedModel<'a> {
    filename: &'a str,
    size_bytes: u64,
    supplied_sha256: &'a str,
    supplied_source_or_revision: &'a str,
    license: &'a str,
    supplied_conversion_or_quantization: &'a str,
}

#[derive(Serialize)]
struct AttestedDevice<'a> {
    name: &'a str,
    registry_id: u64,
}

#[derive(Serialize)]
struct AttestedInvocation<'a> {
    prompt: &'a str,
    mode: &'static str,
    max_new_tokens: usize,
    expected_token_ids: &'a [u32],
    actual_token_ids: &'a [u32],
}

#[derive(Serialize)]
struct AttestedOracle<'a> {
    name: &'a str,
    revision: &'a str,
    command: &'a str,
}

#[derive(Serialize)]
struct AttestedEvidence<'a> {
    workflow_run_url: &'a str,
    workflow_run_id: &'a str,
    scoreboard_filename: &'a str,
    deployment_identity: u64,
    capture_identity: u64,
    execution_plan_identity: u64,
    successful_run_count: u64,
    final_committed_position: usize,
    fallback_count: usize,
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
    let Some(args) = parse_args(env::args().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };
    validate_evidence_paths(&args)?;
    let attested_model = args
        .attestation
        .as_ref()
        .map(|_| inspect_attested_model(&args.model_path))
        .transpose()?;
    let scoreboard_context = match (&args.scoreboard_path, &args.revision) {
        (Some(_), Some(revision)) => Some(MetalScoreboardContext::new(
            SCOREBOARD_WORKLOAD,
            revision,
            SCOREBOARD_EVIDENCE,
        )?),
        (None, None) => None,
        _ => unreachable!("paired scoreboard arguments are validated by parse_args"),
    };
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
    let plan = match args.prefill_span {
        Some(span_rows) => LlamaMetalPlan::from_workflow_with_prefill_span(
            workflow,
            &device,
            MetalPlanOptions::default(),
            span_rows,
        )?,
        None => LlamaMetalPlan::from_workflow(workflow, &device, MetalPlanOptions::default())?,
    };
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
    if let Some(path) = &args.scoreboard_path {
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
        let scoreboard_bytes = report.to_json_bytes()?;
        let attestation_bytes = match (&args.attestation, &attested_model) {
            (Some(attestation), Some(model)) => Some(attestation_json(AttestationRecordContext {
                args: &args,
                provenance: attestation,
                model,
                stable: &stable,
                generation,
                report: &report,
                final_position: session.position(),
                scoreboard_path: path,
            })?),
            (None, None) => None,
            _ => unreachable!("attested model metadata is paired with attestation arguments"),
        };
        write_new_evidence(path, &scoreboard_bytes)?;
        if let (Some(attestation), Some(bytes)) = (&args.attestation, attestation_bytes) {
            write_new_evidence(&attestation.path, &bytes)?;
        }
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

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Option<Args>, CliError> {
    let mut device_index = None;
    let mut expected_registry_id = None;
    let mut max_new_tokens = None;
    let mut prefill_span = None;
    let mut chat = false;
    let mut scoreboard_path = None;
    let mut revision = None;
    let mut expected_ids = None;
    let mut attestation_path = None;
    let mut model_sha256 = None;
    let mut model_source = None;
    let mut model_license = None;
    let mut model_conversion = None;
    let mut oracle_name = None;
    let mut oracle_revision = None;
    let mut oracle_command = None;
    let mut workflow_run_url = None;
    let mut workflow_run_id = None;
    let mut positional = Vec::new();
    let mut args = args.into_iter();
    let mut positional_only = false;
    while let Some(argument) = args.next() {
        if positional_only {
            positional.push(argument);
            continue;
        }
        match argument.as_str() {
            "--help" | "-h" => return Ok(None),
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
            "--prefill-span" => set_once(
                &mut prefill_span,
                parse_prefill_span(args.next())?,
                "--prefill-span",
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
            "--attestation" => set_once(
                &mut attestation_path,
                PathBuf::from(required_value(args.next(), "--attestation")?),
                "--attestation",
            )?,
            "--model-sha256" => set_once(
                &mut model_sha256,
                required_value(args.next(), "--model-sha256")?,
                "--model-sha256",
            )?,
            "--model-source" => set_once(
                &mut model_source,
                required_value(args.next(), "--model-source")?,
                "--model-source",
            )?,
            "--model-license" => set_once(
                &mut model_license,
                required_value(args.next(), "--model-license")?,
                "--model-license",
            )?,
            "--model-conversion" => set_once(
                &mut model_conversion,
                required_value(args.next(), "--model-conversion")?,
                "--model-conversion",
            )?,
            "--oracle-name" => set_once(
                &mut oracle_name,
                required_value(args.next(), "--oracle-name")?,
                "--oracle-name",
            )?,
            "--oracle-revision" => set_once(
                &mut oracle_revision,
                required_value(args.next(), "--oracle-revision")?,
                "--oracle-revision",
            )?,
            "--oracle-command" => set_once(
                &mut oracle_command,
                required_value(args.next(), "--oracle-command")?,
                "--oracle-command",
            )?,
            "--workflow-run-url" => set_once(
                &mut workflow_run_url,
                required_value(args.next(), "--workflow-run-url")?,
                "--workflow-run-url",
            )?,
            "--workflow-run-id" => set_once(
                &mut workflow_run_id,
                required_value(args.next(), "--workflow-run-id")?,
                "--workflow-run-id",
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
    if prefill_span.is_some() && scoreboard_path.is_some() {
        return Err(cli(
            "--prefill-span is not supported with scoreboard evidence; use the T1 plan for scored runs",
        ));
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
    let attestation_requested = attestation_path.is_some()
        || model_sha256.is_some()
        || model_source.is_some()
        || model_license.is_some()
        || model_conversion.is_some()
        || oracle_name.is_some()
        || oracle_revision.is_some()
        || oracle_command.is_some()
        || workflow_run_url.is_some()
        || workflow_run_id.is_some();
    let attestation = if attestation_requested {
        if scoreboard_path.is_none() || revision.is_none() || expected_ids.is_none() {
            return Err(cli(
                "attestation requires --scoreboard, --revision, and --expected-ids",
            ));
        }
        let model_sha256 = attestation_value(model_sha256, "--model-sha256")?;
        if !is_lower_hex(&model_sha256, 64) {
            return Err(cli(
                "--model-sha256 requires 64 lowercase hexadecimal characters",
            ));
        }
        let revision_value = revision.as_deref().expect("checked above");
        if !is_lower_hex(revision_value, 40) {
            return Err(cli("attested --revision requires a lowercase full Git SHA"));
        }
        let attestation = AttestationArgs {
            path: attestation_value(attestation_path, "--attestation")?,
            model_sha256,
            model_source: attestation_value(model_source, "--model-source")?,
            model_license: attestation_value(model_license, "--model-license")?,
            model_conversion: attestation_value(model_conversion, "--model-conversion")?,
            oracle_name: attestation_value(oracle_name, "--oracle-name")?,
            oracle_revision: attestation_value(oracle_revision, "--oracle-revision")?,
            oracle_command: attestation_value(oracle_command, "--oracle-command")?,
            workflow_run_url: attestation_value(workflow_run_url, "--workflow-run-url")?,
            workflow_run_id: attestation_value(workflow_run_id, "--workflow-run-id")?,
        };
        validate_attestation_strings(&attestation)?;
        Some(attestation)
    } else {
        None
    };
    let prompt = positional.pop().expect("length checked");
    let model_path = PathBuf::from(positional.pop().expect("length checked"));
    Ok(Some(Args {
        device_index: device_index.unwrap_or(0),
        expected_registry_id,
        max_new_tokens,
        prefill_span,
        chat,
        scoreboard_path,
        revision,
        expected_ids,
        attestation,
        model_path,
        prompt,
    }))
}

fn attestation_value<T>(value: Option<T>, flag: &'static str) -> Result<T, CliError> {
    value.ok_or_else(|| cli(format!("attestation requires {flag}")))
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn validate_attestation_strings(attestation: &AttestationArgs) -> Result<(), CliError> {
    for (flag, value) in [
        ("--model-source", attestation.model_source.as_str()),
        ("--model-license", attestation.model_license.as_str()),
        ("--model-conversion", attestation.model_conversion.as_str()),
        ("--oracle-name", attestation.oracle_name.as_str()),
        ("--oracle-revision", attestation.oracle_revision.as_str()),
        ("--oracle-command", attestation.oracle_command.as_str()),
        ("--workflow-run-url", attestation.workflow_run_url.as_str()),
        ("--workflow-run-id", attestation.workflow_run_id.as_str()),
    ] {
        if value.contains('\r') || value.contains('\n') {
            return Err(cli(format!("{flag} must be a single line")));
        }
    }
    if !attestation
        .workflow_run_id
        .bytes()
        .all(|byte| byte.is_ascii_digit())
    {
        return Err(cli("--workflow-run-id requires an unsigned integer"));
    }
    Ok(())
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

fn parse_prefill_span(value: Option<String>) -> Result<NonZeroUsize, CliError> {
    let span = parse_usize(value, "--prefill-span")?;
    NonZeroUsize::new(span)
        .filter(|span| span.get() > 1)
        .ok_or_else(|| cli("--prefill-span must be at least 2"))
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

fn validate_evidence_paths(args: &Args) -> Result<(), io::Error> {
    if let Some(scoreboard) = &args.scoreboard_path {
        let scoreboard_destination = resolved_new_destination(scoreboard, "scoreboard")?;
        if scoreboard.try_exists()? {
            return Err(io::Error::other("scoreboard path already exists"));
        }
        if let Some(attestation) = &args.attestation {
            let attestation_destination =
                resolved_new_destination(&attestation.path, "attestation")?;
            if attestation.path.try_exists()? {
                return Err(io::Error::other("attestation path already exists"));
            }
            if scoreboard_destination == attestation_destination {
                return Err(io::Error::other(
                    "scoreboard and attestation paths must be distinct",
                ));
            }
        }
    }
    Ok(())
}

fn resolved_new_destination(path: &Path, label: &str) -> Result<PathBuf, io::Error> {
    let filename = evidence_filename(path, label)?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("{label} path requires an existing parent: {error}"),
        )
    })?;
    Ok(parent.join(filename))
}

fn evidence_filename<'a>(path: &'a Path, label: &str) -> Result<&'a str, io::Error> {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| io::Error::other(format!("{label} path requires a UTF-8 filename")))
}

fn inspect_attested_model(path: &Path) -> Result<AttestedModelFile, io::Error> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(io::Error::other(
            "attested model path is not a regular file",
        ));
    }
    Ok(AttestedModelFile {
        filename: evidence_filename(path, "model")?.to_owned(),
        size_bytes: metadata.len(),
    })
}

fn attestation_json(context: AttestationRecordContext<'_>) -> Result<Vec<u8>, io::Error> {
    let AttestationRecordContext {
        args,
        provenance,
        model,
        stable,
        generation,
        report,
        final_position,
        scoreboard_path,
    } = context;
    let revision = args
        .revision
        .as_deref()
        .ok_or_else(|| io::Error::other("attestation has no code revision"))?;
    let expected_ids = args
        .expected_ids
        .as_deref()
        .ok_or_else(|| io::Error::other("attestation has no expected token IDs"))?;
    let attestation = MetalLlamaAttestation {
        format_version: 1,
        code_revision_sha: revision,
        model: AttestedModel {
            filename: model.filename.as_str(),
            size_bytes: model.size_bytes,
            supplied_sha256: provenance.model_sha256.as_str(),
            supplied_source_or_revision: provenance.model_source.as_str(),
            license: provenance.model_license.as_str(),
            supplied_conversion_or_quantization: provenance.model_conversion.as_str(),
        },
        device: AttestedDevice {
            name: stable.device_info.name.as_str(),
            registry_id: stable.device_info.registry_id,
        },
        invocation: AttestedInvocation {
            prompt: args.prompt.as_str(),
            mode: if args.chat {
                "single_user_chat"
            } else {
                "plain"
            },
            max_new_tokens: args.max_new_tokens,
            expected_token_ids: expected_ids,
            actual_token_ids: generation.generated_ids(),
        },
        oracle: AttestedOracle {
            name: provenance.oracle_name.as_str(),
            revision: provenance.oracle_revision.as_str(),
            command: provenance.oracle_command.as_str(),
        },
        evidence: AttestedEvidence {
            workflow_run_url: provenance.workflow_run_url.as_str(),
            workflow_run_id: provenance.workflow_run_id.as_str(),
            scoreboard_filename: evidence_filename(scoreboard_path, "scoreboard")?,
            deployment_identity: report.deployment_identity,
            capture_identity: report.capture_identity,
            execution_plan_identity: report.execution_plan_identity,
            successful_run_count: report.successful_run_count,
            final_committed_position: final_position,
            fallback_count: report.fallback_count,
        },
    };
    serialize_attestation(&attestation)
}

fn serialize_attestation(attestation: &MetalLlamaAttestation<'_>) -> Result<Vec<u8>, io::Error> {
    let mut json = serde_json::to_vec_pretty(attestation)
        .map_err(|error| io::Error::other(format!("attestation JSON encoding failed: {error}")))?;
    json.push(b'\n');
    Ok(json)
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
    use super::{
        parse_args, serialize_attestation, validate_evidence_paths, write_new_evidence, Args,
        AttestationArgs, AttestedDevice, AttestedEvidence, AttestedInvocation, AttestedModel,
        AttestedOracle, MetalLlamaAttestation,
    };
    use std::{
        fs,
        num::NonZeroUsize,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

    fn parse(values: &[&str]) -> Result<Args, super::CliError> {
        parse_args(values.iter().map(|value| (*value).to_owned()))?
            .ok_or_else(|| super::cli("unexpected help action"))
    }

    #[test]
    fn parser_accepts_plain_chat_and_complete_evidence_modes() {
        assert_eq!(
            parse(&["model.gguf", "hello"]).unwrap(),
            Args {
                device_index: 0,
                expected_registry_id: None,
                max_new_tokens: 16,
                prefill_span: None,
                chat: false,
                scoreboard_path: None,
                revision: None,
                expected_ids: None,
                attestation: None,
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
        assert_eq!(
            parse(&["--prefill-span", "3", "model.gguf", "hello"])
                .unwrap()
                .prefill_span,
            NonZeroUsize::new(3)
        );
        assert!(evidence.chat);
        assert_eq!(evidence.expected_ids, Some(vec![4, 5, 6]));
        assert!(evidence.attestation.is_none());

        let attested = parse(&[
            "--scoreboard",
            "scoreboard.json",
            "--revision",
            "0123456789abcdef0123456789abcdef01234567",
            "--expected-ids",
            "4,5,6",
            "--attestation",
            "attestation.json",
            "--model-sha256",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "--model-source",
            "https://example.invalid/model@revision",
            "--model-license",
            "License-Id",
            "--model-conversion",
            "converter@revision --quantize q4_k",
            "--oracle-name",
            "reference-runtime",
            "--oracle-revision",
            "oracle-revision",
            "--oracle-command",
            "reference --seed 0",
            "--workflow-run-url",
            "https://example.invalid/actions/runs/7",
            "--workflow-run-id",
            "7",
            "model.gguf",
            "hello",
        ])
        .unwrap();
        assert_eq!(attested.attestation.unwrap().model_license, "License-Id");
        assert!(parse_args(["--help".to_owned()]).unwrap().is_none());
        assert!(parse_args(["-h".to_owned()]).unwrap().is_none());
    }

    #[test]
    fn parser_rejects_duplicates_malformed_values_and_unpaired_evidence() {
        for values in [
            vec!["--chat", "--chat", "m", "p"],
            vec!["--device", "x", "m", "p"],
            vec!["--expected-registry-id", "x", "m", "p"],
            vec!["--max-new-tokens", "4097", "m", "p"],
            vec!["--prefill-span", "0", "m", "p"],
            vec!["--prefill-span", "1", "m", "p"],
            vec![
                "--prefill-span",
                "2",
                "--scoreboard",
                "out.json",
                "--revision",
                "sha",
                "m",
                "p",
            ],
            vec!["--expected-ids", "1,,2", "m", "p"],
            vec!["--max-new-tokens", "1", "--expected-ids", "1,2", "m", "p"],
            vec!["--scoreboard", "out.json", "m", "p"],
            vec!["--revision", "sha", "m", "p"],
            vec!["--attestation", "attestation.json", "m", "p"],
            vec![
                "--scoreboard",
                "scoreboard.json",
                "--revision",
                "not-a-sha",
                "--expected-ids",
                "1",
                "--attestation",
                "attestation.json",
                "m",
                "p",
            ],
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

    #[test]
    fn paired_evidence_paths_are_distinct_and_create_new() {
        let suffix = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rustgrad-metal-llama-evidence-paths-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::create_dir(root.join("nested")).unwrap();
        let scoreboard = root.join("scoreboard.json");
        let attestation = root.join("attestation.json");
        let args = Args {
            device_index: 0,
            expected_registry_id: Some(7),
            max_new_tokens: 1,
            prefill_span: None,
            chat: false,
            scoreboard_path: Some(scoreboard.clone()),
            revision: Some("0123456789abcdef0123456789abcdef01234567".into()),
            expected_ids: Some(vec![1]),
            attestation: Some(AttestationArgs {
                path: attestation.clone(),
                model_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                    .into(),
                model_source: "source@revision".into(),
                model_license: "license".into(),
                model_conversion: "conversion".into(),
                oracle_name: "oracle".into(),
                oracle_revision: "oracle-revision".into(),
                oracle_command: "oracle command".into(),
                workflow_run_url: "https://example.invalid/actions/runs/7".into(),
                workflow_run_id: "7".into(),
            }),
            model_path: PathBuf::from("model.gguf"),
            prompt: "hello".into(),
        };
        validate_evidence_paths(&args).unwrap();
        write_new_evidence(&scoreboard, b"scoreboard").unwrap();
        assert!(validate_evidence_paths(&args).is_err());
        write_new_evidence(&attestation, b"attestation").unwrap();
        assert!(write_new_evidence(&attestation, b"replacement").is_err());
        fs::remove_file(scoreboard).unwrap();
        fs::remove_file(attestation).unwrap();

        let mut same = args;
        same.scoreboard_path = Some(root.join("same.json"));
        same.attestation.as_mut().unwrap().path = root.join("nested/../same.json");
        assert!(validate_evidence_paths(&same).is_err());

        #[cfg(unix)]
        {
            let alias = root.join("alias");
            std::os::unix::fs::symlink(&root, &alias).unwrap();
            same.scoreboard_path = Some(root.join("symlinked.json"));
            same.attestation.as_mut().unwrap().path = alias.join("symlinked.json");
            assert!(validate_evidence_paths(&same).is_err());
            fs::remove_file(alias).unwrap();
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn attestation_json_is_typed_deterministic_and_path_free() {
        let attestation = MetalLlamaAttestation {
            format_version: 1,
            code_revision_sha: "0123456789abcdef0123456789abcdef01234567",
            model: AttestedModel {
                filename: "model.gguf",
                size_bytes: 42,
                supplied_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                supplied_source_or_revision: "source@revision",
                license: "license",
                supplied_conversion_or_quantization: "conversion",
            },
            device: AttestedDevice {
                name: "Mock Metal",
                registry_id: 7,
            },
            invocation: AttestedInvocation {
                prompt: "hello",
                mode: "plain",
                max_new_tokens: 2,
                expected_token_ids: &[3, 4],
                actual_token_ids: &[3, 4],
            },
            oracle: AttestedOracle {
                name: "oracle",
                revision: "oracle-revision",
                command: "oracle command",
            },
            evidence: AttestedEvidence {
                workflow_run_url: "https://example.invalid/actions/runs/9",
                workflow_run_id: "9",
                scoreboard_filename: "scoreboard.json",
                deployment_identity: 10,
                capture_identity: 11,
                execution_plan_identity: 12,
                successful_run_count: 3,
                final_committed_position: 3,
                fallback_count: 0,
            },
        };
        let bytes = serialize_attestation(&attestation).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["format_version"], 1);
        assert_eq!(value["model"]["filename"], "model.gguf");
        assert_eq!(
            value["invocation"]["actual_token_ids"],
            serde_json::json!([3, 4])
        );
        assert_eq!(value["evidence"]["fallback_count"], 0);
        assert!(!std::str::from_utf8(&bytes).unwrap().contains("/runner/"));
        assert_eq!(bytes.last(), Some(&b'\n'));
        assert_eq!(bytes, serialize_attestation(&attestation).unwrap());
    }

    #[test]
    fn live_workflow_pairs_attestation_and_scoreboard_checksums() {
        let workflow = include_str!("../.github/workflows/metal-live.yml");
        for required in [
            "RUSTGRAD_METAL_LLAMA_MODEL_SOURCE",
            "RUSTGRAD_METAL_LLAMA_MODEL_LICENSE",
            "RUSTGRAD_METAL_LLAMA_MODEL_CONVERSION",
            "RUSTGRAD_METAL_LLAMA_ORACLE_NAME",
            "RUSTGRAD_METAL_LLAMA_ORACLE_REVISION",
            "RUSTGRAD_METAL_LLAMA_ORACLE_COMMAND",
            "--attestation",
            "--workflow-run-url",
            "SHA256SUMS",
        ] {
            assert!(workflow.contains(required), "workflow omits {required}");
        }
        assert!(workflow.contains("shasum -a 256"));
        assert!(workflow.contains("set -o noclobber"));
        assert!(workflow.contains("-e \"$output_path\" || -L \"$output_path\""));
        assert!(!workflow.contains("> SHA256SUMS"));
        let llama_job = workflow
            .split_once("  live-metal-llama:")
            .expect("live Llama job")
            .1;
        let checkout = llama_job.find("Check out exact reviewed revision").unwrap();
        let model_hash = llama_job.find("actual_model_sha=").unwrap();
        let output_gate = llama_job.find("for output_path in").unwrap();
        let toolchain = llama_job.find("Install pinned Rust toolchain").unwrap();
        let execution = llama_job
            .find("cargo run --release --example metal_llama_generate")
            .unwrap();
        assert!(checkout < model_hash);
        assert!(model_hash < output_gate);
        assert!(output_gate < toolchain);
        assert!(toolchain < execution);
        assert_eq!(
            llama_job
                .matches("cargo run --release --example metal_llama_generate")
                .count(),
            1
        );
        assert!(
            workflow.contains("actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02")
        );
        assert!(!workflow.contains("curl "));
        assert!(!workflow.contains("wget "));
    }
}
