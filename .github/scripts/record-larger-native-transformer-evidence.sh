#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

: "${GITHUB_SHA:?GITHUB_SHA must identify the measured revision}"
: "${RUNNER_TEMP:?RUNNER_TEMP must provide isolated runner storage}"
# Manual same-runner comparisons set all three overrides. Their absence keeps
# the protected scale-evidence invocation and output names unchanged.
measured_sha="${RUSTGRAD_SCALE_EVIDENCE_MEASURED_SHA:-$GITHUB_SHA}"
output_dir="${RUSTGRAD_SCALE_EVIDENCE_OUTPUT_DIR:-native-cpu-larger-transformer-evidence}"
prebuilt_binary="${RUSTGRAD_SCALE_EVIDENCE_BINARY:-}"
workflow_timeout_minutes="${RUSTGRAD_SCALE_EVIDENCE_WORKFLOW_TIMEOUT_MINUTES:-45}"
if [[ ! "$GITHUB_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "GITHUB_SHA must be a lowercase full Git SHA" >&2
  exit 1
fi
if [[ ! "$measured_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "measured revision must be a lowercase full Git SHA" >&2
  exit 1
fi
if [[ -z "$output_dir" || "$output_dir" == "." || "$output_dir" == ".." ]]; then
  echo "larger Transformer output directory must be explicit" >&2
  exit 1
fi
if [[ "$workflow_timeout_minutes" != 45 && "$workflow_timeout_minutes" != 75 ]]; then
  echo "larger Transformer workflow timeout must be 45 or 75 minutes" >&2
  exit 1
fi

actual_sha="$(git rev-parse HEAD)"
if [[ "$actual_sha" != "$measured_sha" ]]; then
  echo "checked-out revision does not match measured revision" >&2
  exit 1
fi

scoreboard_path="${output_dir}/native-cpu-training-scoreboard.json"
objective_path="${output_dir}/objective-evidence.json"
provenance_path="${output_dir}/provenance.txt"
resume_bundle_path="${output_dir}/replay-3-resume.rgab"
module_checkpoint_path="${output_dir}/replay-3-module-checkpoint.safetensors"
checksums_path="${output_dir}/sha256.txt"
prebuilt_binary_sha256=""
if [[ -n "$prebuilt_binary" ]]; then
  if [[ "$prebuilt_binary" != /* || ! -f "$prebuilt_binary" || -L "$prebuilt_binary" || ! -x "$prebuilt_binary" ]]; then
    echo "prebuilt scale-evidence binary must be an absolute executable regular file" >&2
    exit 1
  fi
  prebuilt_binary_sha256="$(sha256sum -- "$prebuilt_binary" | awk '{ print $1 }')"
  if [[ ! "$prebuilt_binary_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "prebuilt scale-evidence binary digest is invalid" >&2
    exit 1
  fi
fi
if [[ -e "$output_dir" || -L "$output_dir" ]]; then
  echo "larger Transformer evidence output already exists: $output_dir" >&2
  exit 1
fi
mkdir -- "$output_dir"

measurement_tmpdir="$(mktemp -d "${RUNNER_TEMP%/}/rustgrad-larger-transformer-${measured_sha}.XXXXXX")"
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
  if [[ -n "$prebuilt_binary" ]]; then
    printf 'prebuilt_binary_source_sha=%s\n' "$actual_sha"
    printf 'prebuilt_binary_sha256=%s\n' "$prebuilt_binary_sha256"
  fi
  printf 'cargo_profile=release\n'
  printf 'cargo_build_jobs=2\n'
  printf 'workflow_timeout_minutes=%s\n' "$workflow_timeout_minutes"
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

if [[ -n "$prebuilt_binary" ]]; then
  TMPDIR="$measurement_tmpdir" \
    RUSTGRAD_EVIDENCE_GIT_SHA="$actual_sha" \
    RUSTGRAD_LARGER_SCOREBOARD_PATH="$scoreboard_path" \
    RUSTGRAD_LARGER_OBJECTIVE_PATH="$objective_path" \
    RUSTGRAD_LARGER_RESUME_BUNDLE_PATH="$resume_bundle_path" \
    RUSTGRAD_LARGER_MODULE_CHECKPOINT_PATH="$module_checkpoint_path" \
    timeout 40m "$prebuilt_binary"
else
  TMPDIR="$measurement_tmpdir" CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 RUSTFLAGS="-D warnings" \
    RUSTGRAD_EVIDENCE_GIT_SHA="$actual_sha" \
    RUSTGRAD_LARGER_SCOREBOARD_PATH="$scoreboard_path" \
    RUSTGRAD_LARGER_OBJECTIVE_PATH="$objective_path" \
    RUSTGRAD_LARGER_RESUME_BUNDLE_PATH="$resume_bundle_path" \
    RUSTGRAD_LARGER_MODULE_CHECKPOINT_PATH="$module_checkpoint_path" \
    timeout 40m cargo run --locked --release --quiet \
      --example compiled_transformer_scale_evidence
fi

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

python3 "$script_dir/check-larger-native-transformer-evidence.py" \
  --objective "$objective_path" \
  --scoreboard "$scoreboard_path" \
  --expected-sha "$actual_sha" \
  --resume-bundle "$resume_bundle_path" \
  --module-checkpoint "$module_checkpoint_path"

(
  cd "$output_dir"
  sha256sum \
    native-cpu-training-scoreboard.json \
    objective-evidence.json \
    provenance.txt \
    replay-3-resume.rgab \
    replay-3-module-checkpoint.safetensors
) > "$checksums_path"
