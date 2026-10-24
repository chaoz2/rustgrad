#!/usr/bin/env bash
# The manual and unprivileged PR adapters share one fixed measurement recipe.
set -euo pipefail

: "${BASELINE_SHA:?exact baseline revision required}"
: "${CANDIDATE_SHA:?exact tested candidate revision required}"
: "${MEASUREMENT_MODE:?fixed measurement mode required}"
: "${GITHUB_WORKSPACE:?runner workspace required}"
: "${RUNNER_TEMP:?isolated runner storage required}"
if [[ ! "$BASELINE_SHA" =~ ^[0-9a-f]{40}$ || ! "$CANDIDATE_SHA" =~ ^[0-9a-f]{40}$ || "$BASELINE_SHA" == "$CANDIDATE_SHA" ]]; then
  echo 'comparison revisions must be distinct lowercase full Git SHAs' >&2
  exit 1
fi
case "$MEASUREMENT_MODE" in
  steady-replay)
    example=compiled_transformer_train_resume
    recorder=record-native-transformer-release-scoreboard.sh
    ;;
  warm-resume)
    example=compiled_transformer_scale_evidence
    recorder=record-larger-native-transformer-evidence.sh
    ;;
  *) echo 'unsupported measurement mode' >&2; exit 1 ;;
esac

baseline_source="$GITHUB_WORKSPACE/baseline-source"
candidate_source="$GITHUB_WORKSPACE/candidate-source"
scripts="$candidate_source/.github/scripts"
build_root="$RUNNER_TEMP/native-transformer-comparison-build"
evidence_root="$GITHUB_WORKSPACE/native-transformer-training-comparison"
baseline_binary="$build_root/baseline/release/examples/$example"
candidate_binary="$build_root/candidate/release/examples/$example"

case "${1:-}" in
  build)
    test "$(git -C "$baseline_source" rev-parse HEAD)" = "$BASELINE_SHA"
    test "$(git -C "$candidate_source" rev-parse HEAD)" = "$CANDIDATE_SHA"
    git -C "$candidate_source" merge-base --is-ancestor "$BASELINE_SHA" "$CANDIDATE_SHA"
    test -z "$(git -C "$baseline_source" status --short)"
    test -z "$(git -C "$candidate_source" status --short)"
    mkdir -- "$build_root"
    cargo build --locked --release --manifest-path "$baseline_source/Cargo.toml" \
      --target-dir "$build_root/baseline" --example "$example"
    cargo build --locked --release --manifest-path "$candidate_source/Cargo.toml" \
      --target-dir "$build_root/candidate" --example "$example"
    # The PR adapter creates this directory for its source binding first.
    mkdir -p -- "$evidence_root"
    python3 "$scripts/write-native-transformer-training-build-manifest.py" \
      --baseline-sha "$BASELINE_SHA" --candidate-sha "$CANDIDATE_SHA" \
      --baseline-binary "$baseline_binary" --candidate-binary "$candidate_binary" \
      --measurement-mode "$MEASUREMENT_MODE" --output "$evidence_root/build-manifest.json"
    ;;
  record)
    measurement_failed=0
    record_trial() {
      local ordinal="$1" role="$2" source_sha source_dir binary output
      if [[ "$role" == baseline ]]; then
        source_sha="$BASELINE_SHA"; source_dir="$baseline_source"; binary="$baseline_binary"
      else
        source_sha="$CANDIDATE_SHA"; source_dir="$candidate_source"; binary="$candidate_binary"
      fi
      output="$evidence_root/trial-${ordinal}-${role}"
      (
        cd "$source_dir"
        if [[ "$MEASUREMENT_MODE" == steady-replay ]]; then
          RUSTGRAD_NATIVE_SCOREBOARD_MEASURED_SHA="$source_sha" \
          RUSTGRAD_NATIVE_SCOREBOARD_OUTPUT_DIR="$output" \
          RUSTGRAD_NATIVE_SCOREBOARD_BINARY="$binary" bash "$scripts/$recorder"
        else
          RUSTGRAD_SCALE_EVIDENCE_MEASURED_SHA="$source_sha" \
          RUSTGRAD_SCALE_EVIDENCE_OUTPUT_DIR="$output" \
          RUSTGRAD_SCALE_EVIDENCE_BINARY="$binary" \
          RUSTGRAD_SCALE_EVIDENCE_WORKFLOW_TIMEOUT_MINUTES=75 bash "$scripts/$recorder"
        fi
      ) || measurement_failed=1
    }
    record_trial 01 baseline
    record_trial 02 candidate
    record_trial 03 candidate
    record_trial 04 baseline
    printf '%s\n' "$measurement_failed" > "$evidence_root/measurement-failed.txt"
    exit "$measurement_failed"
    ;;
  validate)
    test ! -e "$evidence_root/comparison.json"
    test ! -L "$evidence_root/comparison.json"
    python3 "$scripts/check-native-transformer-training-comparison.py" \
      --root "$evidence_root" --baseline-sha "$BASELINE_SHA" --candidate-sha "$CANDIDATE_SHA" \
      --baseline-binary "$baseline_binary" --candidate-binary "$candidate_binary" \
      --measurement-mode "$MEASUREMENT_MODE" --build-manifest "$evidence_root/build-manifest.json" \
      --output "$evidence_root/comparison.json"
    ;;
  *) echo 'expected comparison stage: build, record, or validate' >&2; exit 1 ;;
esac
