#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_SHA:?GITHUB_SHA must identify the measured revision}"
: "${RUNNER_TEMP:?RUNNER_TEMP must provide isolated runner storage}"
if [[ ! "$GITHUB_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "GITHUB_SHA must be a lowercase full Git SHA" >&2
  exit 1
fi

actual_sha="$(git rev-parse HEAD)"
if [[ "$actual_sha" != "$GITHUB_SHA" ]]; then
  echo "checked-out revision does not match GITHUB_SHA" >&2
  exit 1
fi

output_dir="native-cpu-larger-transformer-evidence"
scoreboard_path="${output_dir}/native-cpu-training-scoreboard.json"
objective_path="${output_dir}/objective-evidence.json"
provenance_path="${output_dir}/provenance.txt"
resume_bundle_path="${output_dir}/replay-3-resume.rgab"
module_checkpoint_path="${output_dir}/replay-3-module-checkpoint.safetensors"
checksums_path="${output_dir}/sha256.txt"
if [[ -e "$output_dir" || -L "$output_dir" ]]; then
  echo "larger Transformer evidence output already exists: $output_dir" >&2
  exit 1
fi
mkdir -- "$output_dir"

measurement_tmpdir="$(mktemp -d "${RUNNER_TEMP%/}/rustgrad-larger-transformer-${GITHUB_SHA}.XXXXXX")"
cleanup() {
  rm -rf -- "$measurement_tmpdir"
}
trap cleanup EXIT

if ! lscpu_output="$(LC_ALL=C lscpu)"; then
  echo "lscpu must describe the pinned Ubuntu runner" >&2
  exit 1
fi

write_lscpu_field() {
  local key="$1"
  local label="$2"
  local value
  if ! value="$(
    printf '%s\n' "$lscpu_output" | awk -v label="$label" '
      index($0, label ":") == 1 {
        if (found) exit 2
        value = substr($0, length(label) + 2)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        gsub(/[[:space:]]+/, " ", value)
        print value
        found = 1
      }
      END { if (found != 1) exit 1 }
    '
  )" || [[ -z "$value" ]]; then
    echo "lscpu field is unavailable or ambiguous: $label" >&2
    exit 1
  fi
  if [[ "$value" == *$'\r'* || "$value" == *$'\n'* ]]; then
    echo "lscpu field must normalize to one line: $label" >&2
    exit 1
  fi
  printf '%s=%s\n' "$key" "$value"
}

{
  printf 'schema_version=1\n'
  printf 'git_sha=%s\n' "$actual_sha"
  printf 'cargo_profile=release\n'
  printf 'cargo_build_jobs=2\n'
  printf 'workflow_timeout_minutes=45\n'
  printf 'workload=batch4_time8_vocab16_embedding8_heads2_ff32_blocks2_replays6\n'
  printf 'warm_resume=portable_rgab_replay3_to6_fresh_executor_same_temporary_cache\n'
  printf 'timing_policy=observational_no_threshold\n'
  printf 'temporary_cache=fresh_sha_scoped\n'
  printf 'runner_os=%s\n' "${RUNNER_OS:-unknown}"
  printf 'runner_arch=%s\n' "${RUNNER_ARCH:-unknown}"
  printf 'runner_image_os=%s\n' "${ImageOS:-unknown}"
  printf 'runner_image_version=%s\n' "${ImageVersion:-unknown}"
  printf 'cpu_hardware_policy=required_normalized_lscpu_v1\n'
  printf '\n[cpu_hardware]\n'
  write_lscpu_field cpu_model_name 'Model name'
  write_lscpu_field cpu_socket_count 'Socket(s)'
  write_lscpu_field cpu_cores_per_socket 'Core(s) per socket'
  write_lscpu_field cpu_threads_per_core 'Thread(s) per core'
  write_lscpu_field cpu_logical_count 'CPU(s)'
  write_lscpu_field cpu_vendor_id 'Vendor ID'
  write_lscpu_field cpu_family 'CPU family'
  write_lscpu_field cpu_model 'Model'
  write_lscpu_field cpu_stepping 'Stepping'
  write_lscpu_field cpu_numa_node_count 'NUMA node(s)'
  printf '\n[rustc]\n'
  rustc -Vv
  printf '\n[cargo]\n'
  cargo -Vv
  printf '\n[c_compiler]\n'
  cc --version
} > "$provenance_path"

TMPDIR="$measurement_tmpdir" CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 RUSTFLAGS="-D warnings" \
  RUSTGRAD_EVIDENCE_GIT_SHA="$actual_sha" \
  RUSTGRAD_LARGER_SCOREBOARD_PATH="$scoreboard_path" \
  RUSTGRAD_LARGER_OBJECTIVE_PATH="$objective_path" \
  RUSTGRAD_LARGER_RESUME_BUNDLE_PATH="$resume_bundle_path" \
  RUSTGRAD_LARGER_MODULE_CHECKPOINT_PATH="$module_checkpoint_path" \
  timeout 40m cargo run --locked --release --quiet \
    --example compiled_transformer_scale_evidence

for evidence_path in \
  "$scoreboard_path" \
  "$objective_path" \
  "$provenance_path" \
  "$resume_bundle_path" \
  "$module_checkpoint_path"
do
  if [[ ! -s "$evidence_path" ]]; then
    echo "larger Transformer evidence file is absent or empty: $evidence_path" >&2
    exit 1
  fi
  if [[ "$(wc -c < "$evidence_path")" -gt 8388608 ]]; then
    echo "larger Transformer evidence file exceeds the 8 MiB artifact budget: $evidence_path" >&2
    exit 1
  fi
done

python3 - \
  "$objective_path" \
  "$scoreboard_path" \
  "$actual_sha" \
  "$resume_bundle_path" \
  "$module_checkpoint_path" <<'PY'
import json
import math
import pathlib
import sys

objective = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
scoreboard = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
scoreboard_main = scoreboard.get("main")
scoreboard_evaluation = scoreboard.get("evaluation")
def same_float(left, right):
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)

def phase_sample_count(report):
    if not isinstance(report, dict):
        return None
    wall_time = report.get("wall_time")
    return wall_time.get("sample_count") if isinstance(wall_time, dict) else None

def duration_nanos(value):
    if not isinstance(value, dict) or set(value) != {"secs", "nanos"}:
        return None
    secs = value.get("secs")
    nanos = value.get("nanos")
    if (
        type(secs) is not int
        or secs < 0
        or type(nanos) is not int
        or not 0 <= nanos < 1_000_000_000
    ):
        return None
    return secs * 1_000_000_000 + nanos

if objective.get("schema_version") != 6 or objective.get("git_sha") != sys.argv[3]:
    raise SystemExit("larger Transformer objective provenance is invalid")
expected_workload = {
    "batch": 4,
    "time": 8,
    "vocabulary": 16,
    "embedding": 8,
    "heads": 2,
    "feed_forward": 32,
    "blocks": 2,
    "compile_count": 1,
    "gradient_accumulation_steps": 2,
    "replays": 6,
}
if objective.get("workload") != expected_workload:
    raise SystemExit("larger Transformer workload identity is invalid")

objective_facts = objective.get("objective")
if not isinstance(objective_facts, dict):
    raise SystemExit("larger Transformer objective facts are absent")
trajectory = objective_facts.get("evaluation_trajectory")
expected_frontiers = [(0, 0, 0), (2, 1, 0), (4, 2, 0), (6, 3, 0)]
if not isinstance(trajectory, list) or len(trajectory) != len(expected_frontiers):
    raise SystemExit("larger Transformer evaluation trajectory is invalid")
trajectory_losses = []
for point, expected in zip(trajectory, expected_frontiers):
    if not isinstance(point, dict):
        raise SystemExit("larger Transformer evaluation point is invalid")
    if (
        (point.get("replay"), point.get("optimizer_step"), point.get("accumulation_index"))
        != expected
        or point.get("checkpoint_state_neutral") is not True
    ):
        raise SystemExit("larger Transformer evaluation frontier is invalid")
    loss = point.get("token_mean_loss")
    if type(loss) not in (int, float) or not math.isfinite(loss):
        raise SystemExit("larger Transformer evaluation loss is non-finite")
    samples = point.get("samples")
    if not isinstance(samples, list) or len(samples) != 2:
        raise SystemExit("larger Transformer evaluation samples are invalid")
    weighted_loss = 0.0
    total_weight = 0
    for sample, batch_replay, valid_tokens in zip(samples, [1, 2], [22, 16]):
        if not isinstance(sample, dict):
            raise SystemExit("larger Transformer evaluation sample is invalid")
        sample_loss = sample.get("token_mean_loss")
        if (
            sample.get("batch_replay") != batch_replay
            or not isinstance(scoreboard_evaluation, dict)
            or sample.get("capture_identity") != scoreboard_evaluation.get("capture_identity")
            or sample.get("valid_token_count") != valid_tokens
            or type(sample_loss) not in (int, float)
            or not math.isfinite(sample_loss)
            or sample.get("native_fallback_count") != 0
            or type(sample.get("executed_native_item_count")) is not int
            or sample["executed_native_item_count"] <= 0
            or type(sample.get("module_dispatch_count")) is not int
            or sample["module_dispatch_count"] <= 0
        ):
            raise SystemExit("larger Transformer evaluation sample contract is invalid")
        weighted_loss += sample_loss * valid_tokens
        total_weight += valid_tokens
    if total_weight != 38 or not same_float(loss, weighted_loss / total_weight):
        raise SystemExit("larger Transformer weighted evaluation mean is invalid")
    trajectory_losses.append(loss)
initial_loss = objective_facts.get("initial_token_mean_loss")
final_loss = objective_facts.get("final_token_mean_loss")
if (
    type(initial_loss) not in (int, float)
    or type(final_loss) not in (int, float)
    or not math.isfinite(initial_loss)
    or not math.isfinite(final_loss)
    or not same_float(initial_loss, trajectory_losses[0])
    or not same_float(final_loss, trajectory_losses[-1])
    or objective_facts.get("decreased") is not True
    or not final_loss < initial_loss
):
    raise SystemExit("larger Transformer objective trajectory does not decrease")

progress = objective.get("progress")
if not isinstance(progress, dict):
    raise SystemExit("larger Transformer progress evidence is absent")
expected_replays = [
    {
        "replay": replay,
        "optimizer_step": replay // 2,
        "accumulation_index": replay % 2,
        "did_update": replay % 2 == 0,
    }
    for replay in range(1, 7)
]
if progress.get("replays") != expected_replays:
    raise SystemExit("larger Transformer replay progression is invalid")
pending = progress.get("pending_resume_checkpoint")
terminal = progress.get("terminal_scoreboard_checkpoint")
if (
    not isinstance(pending, dict)
    or pending.get("replay") != 3
    or pending.get("optimizer_step") != 1
    or pending.get("accumulation_index") != 1
    or type(pending.get("bytes")) is not int
    or pending["bytes"] <= 0
    or type(pending.get("module_checkpoint_bytes")) is not int
    or pending["module_checkpoint_bytes"] <= 0
    or pending["module_checkpoint_bytes"] != pathlib.Path(sys.argv[5]).stat().st_size
):
    raise SystemExit("larger Transformer pending checkpoint evidence is invalid")
if (
    not isinstance(terminal, dict)
    or terminal.get("replay") != 6
    or terminal.get("optimizer_step") != 3
    or terminal.get("accumulation_index") != 0
    or type(terminal.get("bytes")) is not int
    or terminal["bytes"] <= 0
    or progress.get("final_replay") != 6
    or progress.get("final_optimizer_step") != 3
    or progress.get("final_accumulation_index") != 0
    or progress.get("exact_resume") is not True
):
    raise SystemExit("larger Transformer terminal checkpoint evidence is invalid")

native = objective.get("native")
if not isinstance(native, dict):
    raise SystemExit("larger Transformer native evidence is absent")
if scoreboard.get("format_version") != 23 or scoreboard.get("initial_replay_step") != 0:
    raise SystemExit("larger Transformer scoreboard identity is invalid")
compile_phases = scoreboard.get("compile_phases")
compile_phase_names = {
    "compile_count",
    "objective_forward",
    "autograd",
    "optimizer_lowering",
    "main_capture",
    "accumulation_capture",
    "partial_flush",
    "zero_grad",
    "evaluation",
    "residual_wall_time",
}
if (
    not isinstance(compile_phases, dict)
    or set(compile_phases) != compile_phase_names
    or compile_phases.get("compile_count") != 1
):
    raise SystemExit("larger Transformer compile-phase evidence is invalid")

compile_phase_total_ns = 0
graph_node_counts = []
for phase_name in ["objective_forward", "autograd", "optimizer_lowering"]:
    phase = compile_phases.get(phase_name)
    if (
        not isinstance(phase, dict)
        or set(phase) != {"wall_time", "graph_node_count"}
        or type(phase.get("graph_node_count")) is not int
        or phase["graph_node_count"] <= 0
        or duration_nanos(phase.get("wall_time")) is None
    ):
        raise SystemExit(f"larger Transformer {phase_name} compile phase is invalid")
    graph_node_counts.append(phase["graph_node_count"])
    compile_phase_total_ns += duration_nanos(phase["wall_time"])
if graph_node_counts != sorted(graph_node_counts):
    raise SystemExit("larger Transformer compiled graph inventories are not monotonic")

capture_programs = {
    "main_capture": scoreboard_main,
    "accumulation_capture": scoreboard.get("accumulation"),
    "partial_flush": scoreboard.get("partial_flush"),
    "zero_grad": scoreboard.get("zero_grad"),
    "evaluation": scoreboard_evaluation,
}
for phase_name, program in capture_programs.items():
    phase = compile_phases.get(phase_name)
    if (
        not isinstance(phase, dict)
        or set(phase) != {"wall_time", "logical_schedule_item_count"}
        or not isinstance(program, dict)
        or type(phase.get("logical_schedule_item_count")) is not int
        or phase["logical_schedule_item_count"] <= 0
        or phase["logical_schedule_item_count"]
        != program.get("logical_schedule_item_count")
        or duration_nanos(phase.get("wall_time")) is None
    ):
        raise SystemExit(f"larger Transformer {phase_name} compile phase is invalid")
    compile_phase_total_ns += duration_nanos(phase["wall_time"])

residual_compile_ns = duration_nanos(compile_phases.get("residual_wall_time"))
compile_wall_ns = duration_nanos(scoreboard.get("compile_wall_time"))
if (
    residual_compile_ns is None
    or compile_wall_ns is None
    or compile_phase_total_ns + residual_compile_ns != compile_wall_ns
):
    raise SystemExit("larger Transformer compile-phase timing does not balance")
scoreboard_checkpoint = scoreboard.get("checkpoint")
if (
    not isinstance(scoreboard_checkpoint, dict)
    or not isinstance(scoreboard_main, dict)
    or scoreboard_checkpoint.get("replay_step") != 6
    or scoreboard_checkpoint.get("byte_count") != terminal["bytes"]
    or scoreboard_checkpoint.get("capture_identity") != scoreboard_main.get("capture_identity")
):
    raise SystemExit("larger Transformer scoreboard checkpoint is invalid")
expected_native = {
    "fallback_count": scoreboard.get("fallback_count"),
    "successful_replay_count": scoreboard.get("successful_replay_count"),
    "recurrent_state_count": scoreboard.get("recurrent_logical_state_count"),
    "recurrent_state_bytes": scoreboard.get("recurrent_logical_state_bytes"),
}
if native != expected_native:
    raise SystemExit("larger Transformer native evidence differs from scoreboard")
if (
    native["fallback_count"] != 0
    or native["successful_replay_count"] != 6
    or type(native["recurrent_state_count"]) is not int
    or native["recurrent_state_count"] <= 0
    or type(native["recurrent_state_bytes"]) is not int
    or native["recurrent_state_bytes"] <= 0
):
    raise SystemExit("larger Transformer native replay inventory is invalid")
for program_name in ["main", "accumulation", "partial_flush", "zero_grad", "evaluation"]:
    program = scoreboard.get(program_name)
    if (
        not isinstance(program, dict)
        or program.get("vectorized") is not True
        or type(program.get("native_item_count")) is not int
        or program["native_item_count"] <= 0
    ):
        raise SystemExit(f"larger Transformer {program_name} program inventory is invalid")
step_phases = scoreboard.get("step_phases")
if (
    not isinstance(step_phases, dict)
    or not isinstance(step_phases.get("first"), dict)
    or step_phases["first"].get("phase") != "accumulation_only"
    or not isinstance(step_phases.get("warm_accumulation_only"), dict)
    or phase_sample_count(step_phases["warm_accumulation_only"]) != 2
    or not isinstance(step_phases.get("warm_optimizer_commit"), dict)
    or phase_sample_count(step_phases["warm_optimizer_commit"]) != 3
):
    raise SystemExit("larger Transformer scoreboard replay census is invalid")

warm = objective.get("warm_resume")
if not isinstance(warm, dict):
    raise SystemExit("larger Transformer warm resume evidence is absent")
if (
    warm.get("replay_from") != 3
    or warm.get("replay_to") != 6
    or warm.get("program_count") != 5
    or warm.get("loaded_module_count") != 5
    or warm.get("durable_artifact_cache_hit_count") != 5
    or warm.get("durable_artifact_cache_miss_count") != 0
    or warm.get("render_capsule_hit_count") != 5
    or warm.get("render_capsule_miss_count") != 0
    or warm.get("local_render_job_count") != 0
    or warm.get("compiler_invocation_count") != 0
    or warm.get("linker_invocation_count") != 0
    or warm.get("fallback_count") != 0
    or warm.get("module_visit_count") != 37
    or warm.get("canonical_state_count") != 36
):
    raise SystemExit("larger Transformer warm resume inventory is invalid")
preparation = warm.get("preparation")
preparation_roles = [
    "main",
    "accumulation",
    "partial_flush",
    "zero_grad",
    "evaluation",
]
preparation_totals = {
    "summed_program_wall_time_ns",
    "runtime_overhead_wall_time_ns",
    "module_overlap_wall_time_ns",
    "render_overlap_wall_time_ns",
    "effective_render_wall_time_ns",
    "effective_render_fraction",
}
if (
    not isinstance(preparation, dict)
    or set(preparation) != set(preparation_roles) | preparation_totals
):
    raise SystemExit("larger Transformer warm preparation evidence is invalid")
phase_fields = {
    "total_wall_time_ns",
    "layout_wall_time_ns",
    "render_wall_time_ns",
    "compiler_wall_time_ns",
    "load_wall_time_ns",
    "residual_wall_time_ns",
}
program_total = 0
render_total = 0
for role in preparation_roles:
    phases = preparation.get(role)
    if not isinstance(phases, dict) or set(phases) != phase_fields:
        raise SystemExit(f"larger Transformer warm {role} phases are invalid")
    if any(type(phases.get(field)) is not int or phases[field] < 0 for field in phase_fields):
        raise SystemExit(f"larger Transformer warm {role} timing is invalid")
    accounted = sum(
        phases[field]
        for field in [
            "layout_wall_time_ns",
            "render_wall_time_ns",
            "compiler_wall_time_ns",
            "load_wall_time_ns",
            "residual_wall_time_ns",
        ]
    )
    if phases["total_wall_time_ns"] != accounted:
        raise SystemExit(f"larger Transformer warm {role} timing does not balance")
    if phases["compiler_wall_time_ns"] != 0:
        raise SystemExit(f"larger Transformer warm {role} unexpectedly compiled")
    program_total += phases["total_wall_time_ns"]
    render_total += phases["render_wall_time_ns"]
for field in preparation_totals - {"effective_render_fraction"}:
    if type(preparation.get(field)) is not int or preparation[field] < 0:
        raise SystemExit(f"larger Transformer warm preparation {field} is invalid")
if preparation["summed_program_wall_time_ns"] != program_total:
    raise SystemExit("larger Transformer warm program timing sum is invalid")
if (
    preparation["module_overlap_wall_time_ns"]
    + preparation["render_overlap_wall_time_ns"]
    > program_total
):
    raise SystemExit("larger Transformer warm overlap exceeds program timing")
whole_preparation = warm.get("preparation_wall_time_ns")
if type(whole_preparation) is not int or whole_preparation <= 0:
    raise SystemExit("larger Transformer warm whole preparation timing is invalid")
reconstructed_preparation = (
    preparation["runtime_overhead_wall_time_ns"]
    + program_total
    - preparation["module_overlap_wall_time_ns"]
    - preparation["render_overlap_wall_time_ns"]
)
if reconstructed_preparation != whole_preparation:
    raise SystemExit("larger Transformer warm whole preparation timing does not balance")
effective_render = render_total - preparation["render_overlap_wall_time_ns"]
if (
    effective_render < 0
    or preparation["effective_render_wall_time_ns"] != effective_render
):
    raise SystemExit("larger Transformer warm effective render timing is invalid")
render_fraction = preparation.get("effective_render_fraction")
if (
    not isinstance(render_fraction, (int, float))
    or isinstance(render_fraction, bool)
    or not math.isfinite(render_fraction)
    or not same_float(render_fraction, effective_render / whole_preparation)
    or not 0.0 <= render_fraction <= 1.0
):
    raise SystemExit("larger Transformer warm effective render fraction is invalid")
for field in [
    "resume_bundle_bytes",
    "module_checkpoint_bytes",
]:
    if type(warm.get(field)) is not int or warm[field] <= 0:
        raise SystemExit(f"larger Transformer warm resume {field} is invalid")
if warm["resume_bundle_bytes"] != pathlib.Path(sys.argv[4]).stat().st_size:
    raise SystemExit("larger Transformer warm resume bundle byte count is invalid")
if warm["module_checkpoint_bytes"] != pathlib.Path(sys.argv[5]).stat().st_size:
    raise SystemExit("larger Transformer module checkpoint byte count is invalid")
for field in [
    "artifact_decode_wall_time_ns",
    "owner_restore_wall_time_ns",
    "preparation_wall_time_ns",
]:
    if type(warm.get(field)) is not int or warm[field] < 0:
        raise SystemExit(f"larger Transformer warm resume {field} is invalid")
for field in [
    "artifact_checkpoint_authenticated",
    "topology_authenticated",
    "different_initialization",
    "fresh_executor",
    "exact_continuation",
    "evaluation_state_neutral",
    "target_owned_module_published",
]:
    if warm.get(field) is not True:
        raise SystemExit(f"larger Transformer warm resume {field} is invalid")
for field in ["capture_identity", "evaluation_capture_identity"]:
    if type(warm.get(field)) is not int or warm[field] <= 0:
        raise SystemExit(f"larger Transformer warm resume {field} is invalid")
if (
    warm["capture_identity"] != scoreboard_main.get("capture_identity")
    or warm["evaluation_capture_identity"]
    != scoreboard["evaluation"].get("capture_identity")
):
    raise SystemExit("larger Transformer warm resume identities differ from scoreboard")
probe = objective.get("gradient_probe")
if not isinstance(probe, dict):
    raise SystemExit("larger Transformer gradient probe is absent")
if (
    probe.get("replay") != 1
    or probe.get("accumulation_index") != 1
    or probe.get("did_update") is not False
    or not same_float(probe.get("dropout_probability", math.nan), 0.0)
    or probe.get("valid_token_count") != 22
):
    raise SystemExit("larger Transformer gradient replay contract is invalid")
if probe.get("active_parameter_count") != 35 or probe.get("active_coordinate_count") != 1888:
    raise SystemExit("larger Transformer gradient inventory is invalid")
if probe.get("tied_output_head_is_canonical_alias") is not True:
    raise SystemExit("larger Transformer tied output ownership is invalid")
if probe.get("frozen_position_is_absent") is not True:
    raise SystemExit("larger Transformer frozen-position ownership is invalid")
if probe.get("native_preparation_fallback_count") != 0:
    raise SystemExit("larger Transformer gradient probe prepared fallback")
if probe.get("native_replay_fallback_count") != 0:
    raise SystemExit("larger Transformer gradient probe executed fallback")
projections = probe.get("projections")
if not isinstance(projections, list) or [item.get("direction_id") for item in projections] != [
    "alternating_dense_v1",
    "seven_phase_dense_v1",
]:
    raise SystemExit("larger Transformer gradient directions are invalid")
for projection in projections:
    numbers = [
        projection.get("epsilon"),
        projection.get("expected_projection"),
        projection.get("actual_projection"),
        projection.get("absolute_error"),
        projection.get("tolerance"),
    ]
    if not all(type(value) in (int, float) and math.isfinite(value) for value in numbers):
        raise SystemExit("larger Transformer gradient projection is non-finite")
    if not same_float(projection["epsilon"], 0.04):
        raise SystemExit("larger Transformer gradient epsilon policy is invalid")
    expected_error = abs(projection["actual_projection"] - projection["expected_projection"])
    expected_tolerance = 0.03 * max(
        1.0,
        abs(projection["actual_projection"]),
        abs(projection["expected_projection"]),
    )
    if not same_float(projection["absolute_error"], expected_error):
        raise SystemExit("larger Transformer gradient error is not authenticated")
    if not same_float(projection["tolerance"], expected_tolerance):
        raise SystemExit("larger Transformer gradient tolerance is not authenticated")
    if projection.get("mutation_sensitive") is not True:
        raise SystemExit("larger Transformer gradient projection is not mutation-sensitive")
    if abs(projection["expected_projection"]) <= projection["tolerance"]:
        raise SystemExit("larger Transformer expected projection is not informative")
    if abs(projection["actual_projection"]) <= projection["tolerance"]:
        raise SystemExit("larger Transformer actual projection is not informative")
    if expected_error > expected_tolerance and not same_float(expected_error, expected_tolerance):
        raise SystemExit("larger Transformer gradient projection exceeds tolerance")
PY

(
  cd "$output_dir"
  sha256sum \
    native-cpu-training-scoreboard.json \
    objective-evidence.json \
    provenance.txt \
    replay-3-resume.rgab \
    replay-3-module-checkpoint.safetensors
) > "$checksums_path"
