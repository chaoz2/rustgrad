#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_TEMP:?RUNNER_TEMP must provide isolated runner storage}"

resume_root="$(mktemp -d "${RUNNER_TEMP%/}/rustgrad-cross-process-resume.XXXXXX")"
cleanup() {
  rm -rf -- "$resume_root"
}
trap cleanup EXIT

for backend in cpu native-cpu; do
  resume_directory="${resume_root}/${backend}"
  mkdir -- "$resume_directory"
  cargo run --quiet --example compiled_transformer_train_resume -- \
    cross-process-produce "$backend" "$resume_directory"
  cargo run --quiet --example compiled_transformer_train_resume -- \
    cross-process-consume "$backend" "$resume_directory"
done
