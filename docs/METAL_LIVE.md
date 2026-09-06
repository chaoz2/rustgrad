# Protected live Metal lane

RustGrad's live Metal workflow is a manual conformance lane, not part of normal
CI. The workflow definition, ignored acceptance test, and examples are not
live-hardware evidence by themselves. Evidence exists only after an exact-SHA
workflow run succeeds and publishes the Linear/ResNet scoreboards, normalized
ResNet observation v1, compiled-Transformer training evidence, device-greedy
Llama execution scoreboard v2, normalized Llama observation v1, attestation,
and checksum manifest.

## Provisioning contract

Provision these external resources before dispatching
`.github/workflows/metal-live.yml`:

1. Attach an Apple Silicon macOS self-hosted runner to this repository with all
   four labels: `self-hosted`, `macOS`, `ARM64`, and `rustgrad-metal`. The runner
   must be online, able to discover a process-visible Metal device, and able to
   fetch the pinned actions, Rust toolchain, and crate dependencies used by the
   workflow.
2. Create the `live-metal` GitHub environment. Restrict its deployment branches
   to the reviewed release branch and configure required reviewers. Do not
   allow unreviewed branches to use the environment.
3. Configure these environment variables:

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

The current external audit found zero runners, no `live-metal`
environment, and none of the required protected variables. These are external
provisioning blockers, not evidence produced by the repository.

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
the environment request fails, or if any required variable is absent. The
runner-local model path and Metal registry identity are intentionally validated
again inside the protected job; a remote preflight cannot attest them.

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

Select the run whose `headSha` is exactly `reviewed_sha`, then follow it without
rerunning a different revision:

```sh
run_id=REPLACE_WITH_MATCHING_DATABASE_ID
gh run watch "$run_id" --repo "$repo_slug" --exit-status
gh run view "$run_id" --repo "$repo_slug" --json headSha,status,conclusion,url
```

The workflow rejects a malformed SHA, a dispatch revision mismatch, a checkout
mismatch, missing or malformed protected configuration, a model hash mismatch,
a wrong Metal registry ID, `MetalDiscovery::NoDevices`, numerical disagreement,
fallback, a prompt too short to execute fixed-span prefill, an absent steady-
decode invocation, incomplete phase transfer/command evidence, missing evidence
files, or an evidence-path collision. If `main`
advances between preflight and dispatch, the expected-SHA check fails instead
of silently testing the newer revision.

## Evidence boundary

The compiled-Transformer artifact uses schema v4 and executes the same
dropout-bearing capture twelve times: eight invocations on the uninterrupted
session and four after checkpoint restoration. Four observed invocations
download only the scalar loss; eight device-only invocations
commit the complete fixed-state successor with zero outputs and zero retained
D2H calls or bytes. Every invocation must commit 59 state pairs, 784 logical
state bytes, and 194 work items. Its declared batch-one I32 token input is
capture-authenticated through an autograd-recorded proof binding the exact
embedding Gather data target to its F32-zero-base first-order ScatterAdd,
shared flatten/expand index/axis/domain, and update cotangent. Every token
lane is range-checked on the host before driver work; only that proven pair uses
distinct status-free Metal kernels, while ordinary indexed movement remains
guarded. With no transactional/indexed owners left, each replay submits and
waits for exactly one command buffer. The artifact records the two authenticated
owners, zero guarded indexed owners, planned kernel count, and exactly twelve
aggregate submissions/waits. It separately requires 336 transient H2D bytes and
exactly 4 retained-output D2H calls totaling 16 bytes. Midpoint checkpointing
and final parameter publication remain explicit host-observation boundaries
outside these per-step transfer totals.

A successful compiled-Transformer job trains through step eight, resumes the
same Metal capture exactly from step four, and explicitly publishes the final
resumed trainable frontier into the fresh reconstruction module. It checks 19
canonical tensors totaling 256 logical bytes, tied-head canonicalization,
one host-version advance per unique parameter, and unchanged runtime
progress/checkpoint state. The current public Metal scoreboard observes
training invocations but does not meter standalone state-snapshot reads, so the
live artifact records the 19-tensor/256-byte logical payload and leaves native
read count null rather than claiming 19 measured driver reads. The semantic
mock separately proves one read per nonempty requested parameter, no read for a
zero-byte parameter, no non-parameter state reads, and retry after a partial
read failure.

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
