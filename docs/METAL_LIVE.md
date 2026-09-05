# Protected live Metal lane

RustGrad's live Metal workflow is a manual conformance lane, not part of normal
CI. The workflow definition, ignored acceptance test, and examples are not
live-hardware evidence by themselves. Evidence exists only after an exact-SHA
workflow run succeeds and publishes the Linear/ResNet scoreboards, normalized
ResNet observation v1, device-greedy Llama execution scoreboard v2, normalized
Llama observation v1, attestation, and checksum manifest.

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
   - `RUSTGRAD_METAL_LLAMA_PROMPT`: the exact nonempty conformance prompt;
   - `RUSTGRAD_METAL_LLAMA_MAX_NEW_TOKENS`: an integer from 1 through 4096;
   - `RUSTGRAD_METAL_LLAMA_PREFILL_SPAN`: an integer from 2 through 4096 for
     the fixed-span device-resident prompt program;
   - `RUSTGRAD_METAL_LLAMA_EXPECTED_IDS`: independently established greedy
     token IDs as a nonempty comma-separated decimal list;
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
fallback, missing evidence files, or an evidence-path collision. If `main`
advances between preflight and dispatch, the expected-SHA check fails instead
of silently testing the newer revision.

## Evidence boundary

A successful Linear/ResNet job uploads two v7 scoreboards plus the normalized
ResNet `BenchmarkObservation` v1. A successful Llama job uploads its
device-greedy execution scoreboard v2, whose token-step and
fixed-span components are authenticated v7 reports, plus a normalized
`BenchmarkObservation` v1, typed provenance attestation, and `SHA256SUMS`. The
observation binds the workflow-verified model hash, exact plain-prompt byte hash,
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
