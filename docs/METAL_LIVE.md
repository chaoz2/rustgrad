# Manual live Metal lane

RustGrad's live Metal workflow is a manual conformance lane, not part of normal
CI. The workflow definition, ignored acceptance test, and examples are not
live-hardware evidence by themselves. Evidence exists only after an exact-SHA
workflow run succeeds. The default run publishes the Linear/ResNet scoreboards,
normalized ResNet observation v1, and compiled-Transformer training evidence.
An opt-in GGUF run additionally publishes the device-greedy Llama execution
scoreboard v2, normalized Llama observation v1, attestation, and checksum
manifest.

## Provisioning contract

Provision these external resources before dispatching
`.github/workflows/metal-live.yml`:

1. Attach an Apple Silicon macOS self-hosted runner to this repository with all
   four labels: `self-hosted`, `macOS`, `ARM64`, and `rustgrad-metal`. The runner
   must be online, able to discover a process-visible Metal device, and able to
   fetch the pinned actions, Rust toolchain, and crate dependencies used by the
   workflow.
2. Configure the existing `live-metal` GitHub environment (ID `21345725438`).
   Restrict its deployment branches to the reviewed release branch and
   configure required reviewers. Do not allow unreviewed branches to use the
   environment.
3. Only when dispatching with `run_gguf=true`, configure these environment
   variables:

   - `RUSTGRAD_METAL_LLAMA_GGUF_PATH`: absolute runner-local path to a regular
     GGUF file outside the checkout;
   - `RUSTGRAD_METAL_LLAMA_GGUF_SHA256`: lowercase 64-character SHA-256 of that
     exact file;
   - `RUSTGRAD_METAL_LLAMA_REGISTRY_ID`: decimal registry ID of the intended
     Metal device;
   - `RUSTGRAD_METAL_LLAMA_PROMPT`: the exact nonempty conformance prompt; after
     the model's optional BOS insertion it must tokenize to at least
     `RUSTGRAD_METAL_LLAMA_PREFILL_SPAN + 1` IDs so the fixed-span program runs;
   - `RUSTGRAD_METAL_LLAMA_MAX_NEW_TOKENS`: an integer from 2 through 4096;
   - `RUSTGRAD_METAL_LLAMA_PREFILL_SPAN`: an integer from 2 through 4096 for
     the fixed-span device-resident prompt program;
   - `RUSTGRAD_METAL_LLAMA_EXPECTED_IDS`: at least two independently established
     greedy token IDs as a comma-separated decimal list, so the first selected
     token is fed back through one steady-decode invocation;
   - `RUSTGRAD_METAL_LLAMA_MODEL_SOURCE`,
     `RUSTGRAD_METAL_LLAMA_MODEL_LICENSE`, and
     `RUSTGRAD_METAL_LLAMA_MODEL_CONVERSION`: immutable, single-line model and
     conversion provenance;
   - `RUSTGRAD_METAL_LLAMA_ORACLE_NAME`,
     `RUSTGRAD_METAL_LLAMA_ORACLE_REVISION`, and
     `RUSTGRAD_METAL_LLAMA_ORACLE_COMMAND`: single-line provenance for the
     independent oracle that produced the expected IDs.

The repository never downloads or uploads the GGUF. Do not place credentials,
private model locations, or model bytes in commits or workflow inputs.

The current external audit found zero compatible runners. The `live-metal`
environment exists as ID `21345725438`, but its `protection_rules` are empty and
its `deployment_branch_policy` is unset, so it is currently unprotected and
unrestricted. Its environment-variable inventory is empty. Missing GGUF
variables block only the opt-in GGUF job and do not block the default
Linear/Transformer/ResNet job. These are external provisioning conditions, not
evidence produced by the repository.

## Remote preflight

From an authenticated `gh` session with repository administration access, set
the repository slug and inspect the external controls:

```sh
repo_slug=chaoz2/rustgrad
gh api "repos/$repo_slug/actions/runners" \
  --jq '.runners[] | {name, status, busy, labels: [.labels[].name]}'
gh api "repos/$repo_slug/environments/live-metal"
gh variable list --repo "$repo_slug" --env live-metal
```

Stop if the runner list has no online runner carrying every required label, if
the environment request fails, or if the environment has no required-reviewer
protection or reviewed-branch deployment restriction. For an opt-in GGUF run,
also stop if any GGUF variable is absent. The runner-local model path and Metal
registry identity are intentionally validated again inside the environment-
scoped GGUF job; a remote preflight cannot attest them. Missing GGUF variables
are irrelevant to the default run.

## Exact-SHA dispatch

Dispatch only a reviewed commit currently reachable as `origin/main`:

```sh
git fetch origin main
reviewed_sha=$(git rev-parse origin/main)
test "${#reviewed_sha}" -eq 40
case "$reviewed_sha" in
  ""|*[!0-9a-f]*)
    echo "origin/main did not resolve to a lowercase full Git SHA" >&2
    exit 1
    ;;
esac
gh workflow run metal-live.yml --repo "$repo_slug" --ref main \
  -f expected_sha="$reviewed_sha"
gh run list --repo "$repo_slug" --workflow metal-live.yml \
  --event workflow_dispatch --limit 5 \
  --json databaseId,headSha,status,conclusion,url
```

To additionally run the separately provisioned GGUF evidence job, dispatch the
same reviewed SHA with `-f run_gguf=true`. Omitting that typed boolean, or
leaving its default `false`, runs Linear, compiled-Transformer training, and
ResNet without reading any GGUF path, hash, prompt, expected-ID, or oracle
configuration.

Select the run whose `headSha` is exactly `reviewed_sha`, then follow it without
rerunning a different revision:

```sh
run_id=REPLACE_WITH_MATCHING_DATABASE_ID
gh run watch "$run_id" --repo "$repo_slug" --exit-status
gh run view "$run_id" --repo "$repo_slug" --json headSha,status,conclusion,url
```

Every running job rejects a malformed SHA, a dispatch revision mismatch, or a
checkout mismatch. The default job also rejects `MetalDiscovery::NoDevices`,
numerical disagreement, fallback, incomplete evidence, missing evidence files,
or an evidence-path collision. When `run_gguf=true`, the GGUF job additionally
rejects missing or malformed protected GGUF configuration, a model hash
mismatch, a wrong Metal registry ID, a prompt too short to execute fixed-span
prefill, or an absent steady-decode invocation. If `main`
advances between preflight and dispatch, the expected-SHA check fails instead
of silently testing the newer revision.

## Evidence boundary

The compiled-Transformer artifact uses schema v7 and executes the same
dropout-bearing capture seven times: four invocations before checkpoint
restoration and three afterward. Two observed invocations download only the
scalar loss; five device-only invocations
commit the complete fixed-state successor with zero outputs and zero retained
D2H calls or bytes. Training progress is measured separately as the
deterministic eval-mode mean sparse loss over the same three fixed microbatches
before module ownership and after final parameter publication; replay-loss
endpoints are not compared across different microbatches or dropout masks. A
three-microbatch accumulation window adds one gradient sum per effective
trainable parameter plus its cursor. The policy-frozen tied token
embedding/output head is absent from the optimizer frontier, so every
invocation must commit 75 state pairs, 952 logical state bytes, and 235
descriptor-derived work items. A
finite 0.25 global norm limit clips the complete averaged gradient set
immediately before each AdamW update. Its
declared fixed `[2,3]` I32 input and target
token matrices are capture-authenticated through separate autograd-recorded
proofs. The trainable target-selection Gather remains bound to its exact
F32-zero-base first-order ScatterAdd, shared flattened index/axis/domain, and
update cotangent. The frozen embedding has no reverse owner and instead uses
the disjoint authenticated forward-only fixed-host Gather proof. Every token
and target lane is range-checked on the host before driver work; only those
three proven owners use distinct status-free Metal kernels, while ordinary
indexed movement remains guarded. With no
transactional/indexed owners left, each replay submits and waits for exactly
one command buffer. The artifact records the paired target owners, one frozen
forward Gather, zero guarded indexed owners, planned kernel count, and exactly
seven replay submissions/waits. The prepared input descriptors prove three
transient writes totaling 52 bytes per replay, so the artifact separately
requires 364 transient H2D bytes and exactly two retained-output D2H calls
totaling eight bytes. The first two distinct
microbatches populate a nonempty window; `zero_grad` discards both without a
training replay, compute-command report, successful scoreboard run,
parameter/moment change, or dropout draw. The inactive-bank reset still submits
synchronous copy commands before the atomic epoch flip. Repeating `zero_grad`
on that empty window is an exact epoch/checkpoint no-op. Midpoint checkpointing
and final parameter publication remain explicit host-observation boundaries
outside these per-step transfer totals. After fresh restoration, a nonempty
two-microbatch flush uploads only the four-byte F32 learning rate, retains no
output, downloads no bytes, submits and waits once, commits the full shared
frontier, and leaves replay, dropout, and training-scoreboard progress
unchanged. A following empty flush is an exact command/epoch/checkpoint no-op.

Training evidence format v7 also records the compiled evaluation aggregate and
a same-seed CPU reference for every batch, cancellation, checkpoint, flush,
replay, evaluation, and finish boundary.
The owned plan prepares a read-only evaluation capture against both
physical parameter banks. Three final fixed-dataset evaluations select the
currently active bank, upload only tokens/targets, and retain loss/logits; they
perform no trainable-parameter H2D/D2H and do not change the training epoch,
checkpoint, dropout counter, successful-run count, or complete scoreboard
report. Each evaluation is a separate stateless one-submit/one-wait invocation,
not a training replay or throughput claim.

A successful compiled-Transformer job trains through step seven, resumes the
same Metal capture exactly from a partial step-four frontier (optimizer step
zero, accumulation index two), commits optimizer step one through the partial
flush and step two at replay seven, and consumes the owned resumed
session to atomically publish the final trainable frontier and return the fresh
reconstruction module. It checks 18 effective-trainable canonical tensors
totaling 232 logical bytes, bounded CPU/Metal agreement for parameters, both
moment sets, accumulators, evaluation, and finished module state, plus exact
uninterrupted-versus-restored CPU checkpoint/state equality. The tied
`tokens.weight`/`lm_head.weight` remains byte-, version-, and trainable-flag
identical and absent from recurrent, checkpoint, and publication state, while
at least one unfrozen parameter changes and fixed-dataset loss decreases. The
current public Metal scoreboard observes
training invocations but does not meter standalone state-snapshot reads, so the
live artifact records the 18-tensor/232-byte logical payload and leaves native
read count null rather than claiming 18 measured driver reads. The semantic
mock separately proves one read per nonempty requested parameter, no read for a
zero-byte parameter, no non-parameter state reads, and retry after a partial
read failure. These are the fail-closed requirements for the next protected
Apple run; this document does not claim schema-v7 live-hardware validation
until that exact-revision job is green.

A successful Linear/ResNet job uploads two v8 scoreboards plus the normalized
ResNet `BenchmarkObservation` v1. A successful Llama job uploads its
device-greedy execution scoreboard v2, whose token-step and
fixed-span components are authenticated v8 reports. The protected harness
requires at least one state-only fixed-span prompt invocation and at least one
steady-decode token-step invocation; the former downloads no output, while the
prompt selector and each decode selector download exactly one four-byte I32.
Both phases must record nonzero kernel, command-submission, and wait counts.
The job also publishes a normalized
`BenchmarkObservation` v1, typed provenance attestation, and `SHA256SUMS`. The
CLI accepts that attestation only together with the complete normalized
benchmark observation; ordinary scoreboard and workload-evidence outputs remain
available without attestation. The observation binds the workflow-verified
model hash, exact plain-prompt byte hash,
actual prompt token count, executed generation bound, canonical expected-ID hash,
selected device, runner OS, validated scoreboard metrics, and a required
`measured_peak_device_memory_bytes`. Both normalized observations carry one
`MetalDeviceBufferMeasurement` token across the workload on the freshly
discovered, exclusively used selected device. Finishing it authenticates the
same `MetalDevice`, checked-converts its
`lifetime_high_water_physical_buffer_bytes` to `u64`, and yields the attached
`RustGradDeviceBufferPeak`. It is the high-water sum of requested native
`MTLBuffer` lengths simultaneously owned by RustGrad—not allocator RSS, physical
residency, driver overhead, or unified-memory pressure. Planned memory remains a
separate metric: ResNet retains it and Llama leaves it unavailable. Raw reports
are unchanged. Preserve the workflow run URL and ID with any release record.

Host-wall durations and optional completed-compute-command GPU execution time
are reported separately, and copy counts are host API calls. Host-run and
compute-command token rates are derived only from their explicitly scoped
scoreboard phase durations. Command time does not establish end-to-end GPU
latency or throughput, copy time, energy use, allocator RSS, physical bus
traffic, or a speedup. Pinned token IDs demonstrate only the configured model,
prompt, and oracle contract; they are not broad cross-runtime parity.
