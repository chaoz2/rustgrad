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

output_dir="native-cpu-training-scoreboard-release"
scoreboard_path="${output_dir}/native-cpu-training-scoreboard.json"
provenance_path="${output_dir}/provenance.txt"

if [[ -e "$output_dir" || -L "$output_dir" ]]; then
  echo "release scoreboard output already exists: $output_dir" >&2
  exit 1
fi
mkdir -- "$output_dir"

measurement_tmpdir="$(mktemp -d "${RUNNER_TEMP%/}/rustgrad-native-scoreboard-${GITHUB_SHA}.XXXXXX")"
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

TMPDIR="$measurement_tmpdir" CARGO_INCREMENTAL=0 \
  cargo run --locked --release --quiet \
    --example compiled_transformer_train_resume -- native-cpu-scoreboard \
  | tee "$scoreboard_path"

if [[ ! -s "$scoreboard_path" || ! -s "$provenance_path" ]]; then
  echo "release scoreboard evidence must contain both nonempty files" >&2
  exit 1
fi
