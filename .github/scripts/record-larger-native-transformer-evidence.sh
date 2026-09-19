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
  timeout 40m cargo run --locked --release --quiet \
    --example compiled_transformer_scale_evidence

for evidence_path in "$scoreboard_path" "$objective_path" "$provenance_path"; do
  if [[ ! -s "$evidence_path" ]]; then
    echo "larger Transformer evidence file is absent or empty: $evidence_path" >&2
    exit 1
  fi
  if [[ "$(wc -c < "$evidence_path")" -gt 8388608 ]]; then
    echo "larger Transformer evidence file exceeds the 8 MiB artifact budget: $evidence_path" >&2
    exit 1
  fi
done

python3 - "$objective_path" "$actual_sha" <<'PY'
import json
import math
import pathlib
import sys

objective = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
def same_float(left, right):
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)

if objective.get("schema_version") != 2 or objective.get("git_sha") != sys.argv[2]:
    raise SystemExit("larger Transformer objective provenance is invalid")
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
  sha256sum native-cpu-training-scoreboard.json objective-evidence.json provenance.txt
) > "$checksums_path"
