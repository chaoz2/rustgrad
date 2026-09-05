# Product priorities

This is RustGrad's maintained, usability-first roadmap. It orders released
evidence into the next user outcomes; it is not a second compatibility ledger.
The [compatibility map](COMPATIBILITY.md) remains the source for implemented
surface and acceptance evidence, while [architecture](ARCHITECTURE.md) defines
the constraints for changes that make these workflows possible.

## Priority definitions

- **P0 — usable workflow:** a person can complete a useful bounded task from
  documented public APIs, with a bounded acceptance test and useful failure
  diagnostics. A missing P0 blocks adoption even if its lower-level pieces are
  tested.
- **P1 — workflow enabler:** a shared public or architectural capability that
  unlocks more than one P0 or a mainstream interchange path. It is sequenced
  behind the P0 it serves unless it is its proven blocker.
- **P2 — breadth and depth:** backend coverage, rare dtype/device corners,
  optimization, or isolated parity work that does not currently unblock a P0.
  It remains valuable, but does not displace the active queue.

## Usability layering

Prioritize from basic adoption toward more advanced use. The layers describe
user value and dependency order; they do not demote already released shared
infrastructure.

1. **Basic adoption:** getting started, ergonomic tensor/session APIs, local
   data and state I/O, and actionable errors.
2. **Complete practical workflows:** train/evaluate/resume, local
   datasets/models, inference, and chat.
3. **Composability needed by those workflows:** common static operations,
   modules, and strict importers.
4. **Performance and deployment:** JIT, memory work, accelerator paths, and
   live-hardware evidence.
5. **Specialized parity:** uncommon dtypes, collectives, dynamic cardinality,
   and backend breadth unless a lower-layer workflow proves one is a blocker.

A higher layer moves forward only when evidence makes it a demonstrated
dependency for a lower-layer user story. Prefer a bounded end-to-end vertical
slice over a broad subsystem checklist.

## Active queue

The training-first queue is deliberately small and ordered. Broad tinygrad
parity, inference/serving expansion, generalized dynamic shapes, and additional
backend breadth are frozen unless the compiled training workload demonstrates
that they are the next concrete blocker.

### 1. P0 — one compiled persistent-state training runtime

Compile `forward → loss → backward → optimizer update → recurrent state` once,
then replay it with new batches without reconstructing a Graph or materializing
gradients through host `TensorData`. Generalize the existing graph-free
momentum-SGD vertical behind optimizer-program and persistent-state contracts;
AdamW is the first implementation, including first/second moments and a step
counter in the same atomic state frontier as parameters. Deterministic
capture-authenticated checkpoint bytes restore values and their exact logical
versions into that same runtime rather than creating a host optimizer path.
The module-bound AdamW entry point maps canonical trainable identities directly
onto that frontier, retains tied weights once, and captures frozen state as
immutable program data. This CPU runtime and its deterministic checkpoint
boundary are delivered; new optimizer work now requires the Transformer
workload to demonstrate the need.

### 2. P0 — tiny Transformer training and exact resume

The protected tiny causal Transformer now trains from deterministic random
initialization through embedding, causal attention, LayerNorm, tied output
weights, cross-entropy, one batched reverse traversal, and captured AdamW.
Eight CPU steps decrease loss, and recompilation from the midpoint checkpoint
continues with exact outputs and recurrent parameter/moment state. Compiled
AdamW now optionally retains F32 gradient sums and an accumulation cursor in
that same frontier, commits one averaged update at each fixed microbatch
window, resets the sums in-capture, and checkpoints partial windows exactly.
The identical recurrent program remains strict-Metal renderable with zero
fallback. Explicit cancellation, clipping, loss scaling, and broader freezing
semantics remain workload-driven follow-ups.

### 3. P1 — lower the identical training capture to Metal

The same captured loss/backward/AdamW program now passes strict Metal admission
with zero fallback and initializes parameter, first/second-moment, and U64-step
state in failure-atomic epoch-swapped device banks. The checked-in protected
Apple-GPU acceptance executes eight tiny-Transformer steps, proves decreasing
loss, checkpoints at step four, prepares a second Metal session from those
bytes, and requires exact continued outputs and final checkpoint equality. Its
create-new evidence records the selected device, capture/deployment identities,
kernel/command/transfer counts, loss endpoints, and resume result. Running that
manual exact-SHA lane on provisioned Apple hardware is the remaining proof;
generic epoch-state scoreboard integration follows rather than a parallel
training API.

## Deferred hardware inference queue

The existing Metal inference work remains useful foundation, but it no longer
drives near-term architecture unless it directly unblocks compiled training.

### Strict persistent Metal device session

**Status:** persistent runtime, a typed ResNet deployment facade, and full
default ResNet-18 structural conformance delivered; exact ignored live
numerical acceptance checked in, hardware evidence pending.

`MetalDeviceSessionPlan` authenticates one concrete pure capture and an
explicit resident/transient input partition before resources. Preparation
uploads immutable weights once and owns persistent slots, pipelines, and queue;
repeated synchronous runs stage only transients, download requested outputs,
expose truthful host-API/planned metrics, and have no CPU fallback. Protected
acceptance now takes the complete default Eval/F32 ResNet-18 `[1,3,224,224]`
graph through boundary-free scheduling and capture, renders every scheduled
item to MSL, then performs persistent resident preparation and repeated
ABI-validating virtual-resource runs. Complete unguarded persistent invocations,
including that protected semantic ResNet-18 path, prevalidate their launches,
encode ordered kernels into one compute command buffer, commit once, wait once,
and retain resources through completion; zero-work invocations submit nothing.
Any guarded or indexed item keeps the existing per-item transactional path. The
opt-in Metal scoreboard v7 records
one exact stateless or append-only session's successful prefix with ordered
per-run host-wall/copy/kernel/compute-command, optional GPU command execution
time, and append-position/row-commit records, checked aggregates, successful cache-miss pipeline-build time, logical
schedule/peak-live facts, and distinct physical Metal slot/state-bank facts.
Failed attempts cannot enter that prefix or advance recorder-owned counters.
The host-logits and device-greedy Llama facades bind one recorder to each real
token-step or fixed-prefill physical session before preparation. One shared
token-step observer owns bind, record, fail-soft freeze, error, and test
instrumentation for both paths. Their Llama execution scoreboard v2 keeps
their local identities and first-run attribution intact while linking exact
spans, positions, bytes, and work items in one global success order. Closed
standalone, prompt-prefill, and steady-decode labels join back to those physical
runs and produce checked row, duration, launch/command, and host API transfer
totals plus explicitly scoped host-run and compute-command token-rate helpers.
Failed attempts and the first latched recorder error cannot extend that prefix.
The records deliberately exclude tokenization, chat, and sampling work. It
remains host-observed execution
measurement plumbing, not live workload evidence, allocator peak memory, bus
traffic, copy timing, end-to-end GPU latency, or cross-runtime speedup evidence.

`ResNetMetalPlan::eval_f32` is the concise public entry point for this vertical:
it freezes the model, binds capability admission and preparation to one explicit
device, exposes its capture/plan/schemas/MSL, and returns a persistent typed
session whose only transient is the exact F32 NCHW image.

An exact ignored Linear acceptance, deterministic typed ResNet-18 benchmark,
and public GGUF Llama prompt-to-tokens harness plus a manual-only exact-SHA
workflow are checked in for
`[self-hosted, macOS, ARM64, rustgrad-metal]` behind the `live-metal`
environment. The benchmark executes the complete initialized body with one
deterministic image, computes one complete CPU oracle, compares finite logits
under a documented F32 native-compilation tolerance, and records ten persistent
session runs by default without duplicating the full execution elsewhere in the
workflow. It publishes both the authentic scoreboard v7 and a create-new
normalized `BenchmarkObservation` v1 bound to the exact revision, deterministic
model identity and checked-in raw little-endian F32 input-payload SHA-256,
selected Metal device, runner OS, command/configuration, planned static-slot
memory, and a separately attached measured RustGrad-owned physical-buffer
high-water. A separate job authenticates the selected
Metal registry ID and
runner-local GGUF bytes against protected values, uses an independently pinned
prompt and greedy token IDs, checks the persistent token-step
ownership/transfer contract, and emits its own device-greedy Llama execution
scoreboard v2 plus a typed provenance attestation and checksum manifest without
downloading or uploading a model. The attestation preserves the protected
model/oracle provenance and exact prompt/ID contract but records, rather than
independently re-proves, the workflow's prior GGUF hash check. Those IDs are
bounded conformance evidence, not a broad cross-runtime numerical oracle. The
release-profile lane is dormant: the current external audit found zero
runners, no `live-metal` environment, and none of the required protected
variables. The repository workflow only references those external controls.
Ordinary macOS CI remains mock-only and no live-device result is
claimed by the workflow definition itself. Its host-wall/API-copy and compute-
command observations remain distinct from the optional completed-command GPU
execution intervals. Derived phase rates are narrowly scoped host-run or
compute-command observations, not physical transfer measurement, end-to-end
latency, or evidence of a live-device speedup.

The normalized comparison plumbing is delivered: `BenchmarkObservation` and
`BenchmarkComparison` version 1, the two RustGrad Metal scoreboard adapters, and
the offline `benchmark_compare` CLI require exact workload/device identity and an
explicit included baseline while preserving unavailable fields as null. They do
not run workloads or derive speedups. The ResNet live harness now emits its
normalized RustGrad observation directly from the validated in-memory scoreboard.
The Llama harness now has the same create-new normalization boundary for its
attested plain-prompt workload, with prompt and expected-ID hashes derived inside
the maintained macOS command. Both harnesses require a fresh, exclusively used
selected RustGrad device and attach its requested-`MTLBuffer`-length lifetime
high-water as measured peak device memory through the typed attachment API,
without changing raw reports or conflating it with planned memory, allocator RSS,
physical residency, driver overhead, or unified-memory pressure. Because the
protected lane remains dormant, actual live Apple-GPU comparison measurements
are still absent.

### ResNet-18 Metal conformance

Provision the protected Apple-GPU lane and run the checked-in exact-SHA
acceptance, closing only demonstrated live renderer/runtime gaps. Required
evidence is full CPU-oracle output agreement, zero fallback, stable resident
weights and intermediates, inspectable kernels/memory/transfers, and the
checked-in v7 compile/prepare/first-run/ordered-steady reporting with host-wall
versus device evidence labeled exactly. Benchmark comparisons target the
equivalent tinygrad and Candle workload.

### Device-resident GGUF Llama prefill/KV/decode

The dense F32 fixed-batch-one token body now binds exact immutable GGUF weights
to the released append-state foundation: one exclusively owned physical KV
bank per tensor, one host-validated monotonic I32 position, sparse complete-row
updates, and same-capture downstream reads. Its scalar model position derives
from the committed position, while one shared row-shaped append index is
expanded and materialized from that scalar on device. Capture-authenticated
scalar embedding and RoPE lookup now removes
the checked-Gather status/candidate path. At the lower runtime boundary,
capture-owned Q4_0/Q8_0/Q4_K/Q6_K row-gather and matmul plans have a direct
persistent Metal packed-buffer ABI, one-time immutable upload,
host-preflighted I32 lookup indices, and versioned F32-accumulating MSL. This
is now wired through the model token body for packed embedding, q/k/v/o,
gated-FFN, and explicit or tied output projections, including mixed
dense/packed models. Sequential T=1 calls reuse one compiled/resident session.
The typed model facade now consumes the same GGUF-bound model, tokenizer, and
chat template, prepares one persistent Metal session, suppresses intermediate
prefill logits, and provides sequential T=1 ID/text/chat generation with host
sampling. The generic Metal append runtime now authenticates fixed nonzero row
spans and commits their checked position atomically. Scoreboard v7 records the
exact authenticated span, expected committed position, bytes, and work items
for each physical append session. Private Metal
admission now also accepts exact dense batch-one `[1,T]` I32 token or position
vectors only through canonical reshape/expand Gather lineage, checks all lanes
before driver work, and uses a distinct status-free direct-render cache domain
while preserving scalar T=1 identity. Integrated packed embedding capture
accepts the same fixed geometry for Q4_0/Q8_0/Q4_K/Q6_K without host
dequantization or repeated packed upload. The chunk-prefill Llama graph and
facade bind separate authenticated recorders to the shared fixed-prefill and
T=1 programs, then publish one ordered Llama execution scoreboard v2 without
inventing a shared physical-session identity. The public greedy CLI now records
that same envelope while retaining only one checked I32 token per selecting
invocation; the host-logits/Gumbel API remains separate. Its checked
prompt/decode phase accounting is ready for live evidence capture, but remaining
work still proves live-device output agreement and publishes measured
prompt/decode performance; the current scoreboard is host-observed execution
accounting, not a speedup claim.
Benchmark comparisons target tinygrad and Candle, plus llama.cpp for the GGUF
serving path.

### Release hygiene

Keep protected Linux/macOS/ASan CI, compatibility fingerprints, live-device
evidence labels, examples, and unsupported-boundary diagnostics synchronized
with each hardware slice.

## Delivered foundation

The CPU adoption, training, GGUF, local interchange, dataset, module, and
compiler/runtime foundations below are delivered evidence. They stay here as
dependency context rather than competing active priorities.

### 1. P0 — ergonomic CPU tensor session and getting started

**Status:** complete (CPU Phase B). **Owner:** `RustGrad — Tensor Semantics`.

**User outcome.** A new Rust user can construct tensors, run ordinary CPU
operations, inspect a result/trace, and get typed errors without assembling a
raw `Graph` plus `HashMap` bindings for every small program.

**Evidence.** `CpuSession` owns one inspectable `Graph` plus explicit owned
CPU bindings. Its session-identified tensor handles construct F32 or exact
typed constants/variables, compose broadcast arithmetic (`add/sub/mul/div`),
ReLU, matmul, stable signed-axis softmax, argmax, reshape/permute/transpose,
shrink/signed slice, concat, and integer gather. They realize through the CPU
oracle, inspect deterministic traces, and build the existing first-order pure
gradient nodes. Cross-session handles,
invalid rebinding shape/dtype, and unsupported device selection are structured
errors; there is no hidden device fallback or global graph. The README and a
public integration tests cover the representative broadcast/reduction and
small-classifier/static-movement workflows, traces, gradients, and repeated
realization. Broader eager aliases, device sessions, dynamic indexing, and
general session-side movement/indexing convenience methods remain future
ergonomic work rather than a second IR.

**Acceptance delivered.** The facade reuses the CPU oracle, graph validation,
and trace contracts; the checked public workflow covers construction,
broadcasting, model arithmetic/activation/softmax/argmax, static movement and
gather, reshape, reduction, realization, trace inspection, repeatable variable
rebinding, first-order gradients, and deterministic invalid-input or device
errors. No accelerator fallback is claimed.

### 2. P0 — minimal train, resume, and evaluate workflow

**Status:** complete (CPU Phase A). **Owner:** `RustGrad — NN Modules &
Optimizers`.

**User outcome.** A user can train a small local classifier, save it, restore
it into fresh module/optimizer/scheduler identities, and continue evaluation
or training through documented APIs.

**Evidence.** `examples/cpu_train_resume.rs` is a documented, dependency-free
CPU workflow: it creates a small classifier, uses deterministic `BatchIter`
ordering including a final partial batch, builds a fresh graph for each real
forward/loss/reverse/update step, captures a `PortableTrainingCheckpoint`,
restores it into fresh module/optimizer/scheduler identities, resumes, and
evaluates without mutation. The public acceptance test proves exact resumed
state/output equality and empty/invalid-batch diagnostics.

**Boundary.** This is local CPU evidence, not downloaded-MNIST accuracy,
distributed/device training, or a hidden eager trainer. The workflow composes
the released graph, optimizer, scheduler, dataset, and portable-checkpoint
contracts directly.

### 3. P0 — bounded GGUF Llama prompt-to-output workflow

**Status:** complete (CPU Phase A; explicit strict-native single-request P2).
**Owner:** `RustGrad — NN Modules & Optimizers`, with `RustGrad — Serialization
& Interop` owning GGUF/file-boundary changes.

**User outcome.** Given a supported local GGUF, a user can validate it, format
a supported Llama chat prompt, generate/decode tokens on CPU, and understand
why an unsupported model or template is rejected.

**Evidence.** `LlamaPromptWorkflow` and `examples/llama_prompt.rs` provide a
documented local-file route from checked GGUF through fixed-schema Llama
binding, tokenizer, exact supported chat rendering, CPU graph generation, and
decoded greedy text. Its fixture acceptance covers deterministic prompt/token/
text output, context rejection without a later-output leak, and malformed GGUF
rejection. The explicit `--native` route uses
`LlamaPromptWorkflow::generate_chat_native` and strict native replay with no
CPU fallback; its fixture acceptance compares deterministic tokens/text with
the CPU route and checks resource-free native stage evidence.

**Boundary.** This is fixed Llama, local GGUF, CPU/static, deterministic greedy
evidence. The native route is only an explicit stateless single request, not
native conversation, serving, sampling, dynamic batching/shapes, device
execution, or a general Llama compatibility claim. Generic Jinja, other model
families, arbitrary external-model parity, implicit RNG, live accelerator
inference, and unsupported layouts remain out of scope.

### 4. P1 — practical interchange path for the P0 workflows

**Status:** complete (CPU Phase A). **Owner:** `RustGrad — Serialization & Interop`.

**User outcome.** Users can move static dense inputs/weights into a supported
CPU workflow without depending on undocumented internal byte handling.

**Evidence.** The public `interop::host` NPY filesystem adapters reuse the
canonical little-endian codec with explicit file/header/rank/element limits
and staged same-directory replacement writes. The README and public acceptance
test demonstrate NPY input through `CpuSession` to NPY output, including exact
special-bit, scalar, and empty-array preservation; the same test loads a
named safetensors state tensor into the session. This is a coherent local
selection path, not a general NumPy, DLPack, or dynamic ONNX runtime.

**Acceptance delivered.** The workflow retains little-endian/raw-bit and
bounded-parser contracts, uses the existing session and safetensors APIs, and
keeps unsupported layouts/formats typed. It does not widen formats or claim
zero-copy compute/device interchange.

### 5. P1 — bounded local MNIST IDX-pair workflow

**Status:** complete (CPU Phase A). **Owner:** `RustGrad — NN Modules & Optimizers`.

**Evidence.** The bounded `load_mnist_idx_files` adapter preflights local file
sizes and declared IDX count/dimensions, then delegates exact parsing to the
existing byte decoder. `tests/mnist_idx_files_workflow.rs` writes a deterministic
local 28×28 pair and proves batching, CPU Graph/autograd updates, portable
fresh-identity resume, and evaluation non-mutation.

**Follow-on evidence.** `ClassificationBatch` now materializes normalized
dataset rows with validated preserve/flatten layout and integer targets, so the
runnable `mnist_idx_local` example feeds deterministic partial batches directly
to graph-free `Linear`, module-derived SGD, and `CpuModuleTrainer` without user
graph/binding/gradient/name-map plumbing.

**Boundary.** Local uncompressed IDX only; no downloader, cache, augmentation,
device training, or corpus-accuracy claim.

### 6. P1 — strict local module-state inference

**Status:** complete (CPU Phase A). **Owner:** `RustGrad — NN Modules &
Optimizers`, with `RustGrad — Serialization & Interop` owning the safetensors
boundary.

**User outcome.** A user can load a local named state file into an already
configured RustGrad module and run known CPU inference without undocumented
parameter replacement or partial mutation.

**Evidence.** `Module::load_state_dict_strict` validates deterministic keys,
exact shapes, and exact dtypes before delegating every candidate to the
existing all-lock parameter restore transaction. Its bounded safetensors byte
and file helpers reuse the existing fail-closed decoder. The public `Linear`
workflow writes a deterministic local safetensors fixture, restores it into a
fresh identity, and proves exact CPU output and mismatch atomicity.

**Boundary.** Module configuration is constructed explicitly before loading;
there is no key remapping, architecture inference, cast, Python/pickle
execution, network, or device fallback. Non-strict state reporting remains the
existing lower-level module API.

### 7. P1 — bounded stateful local Llama conversation

**Status:** complete (CPU Phase A). **Owner:** `RustGrad — NN Modules & Optimizers`.

**Evidence.** `LlamaConversation` composes the released validated local GGUF
workflow, exact supported chat template, and transactional CPU generator into
committed two-turn history. `examples/llama_chat.rs` is the runnable local-file
route; fixture acceptance proves deterministic turns, cache isolation/reset,
and history/cache preservation across rejected requests.

**Boundary.** Fixed supported static CPU Llama schema and greedy generation
only. There is no network/download, implicit sampling RNG, arbitrary Jinja,
device cache, dynamic shape support, or other model family claim.

### 8. P1 — bounded local static ONNX inference

**Status:** complete (CPU Phase A; narrow strict-native P2). **Owner:**
`RustGrad — Serialization & Interop`.

**Evidence.** The local ONNX facade bounds model-file reads, exposes concrete
input schemas, preflights exact names/shapes/dtypes, reuses named NPY files and
the existing CPU model execution, and stages selected named NPY outputs. A
public independently encoded MatMul→Add→Relu fixtures prove deterministic
model-file-to-file execution and preflight failures. The `onnx_npy_infer`
command exposes the caller-owned strict-native path for explicit fixed-F32
named input and selected output maps, with deterministic reports and no fallback.

**Boundary.** Native is only fixed-schema F32 selected-output
MatMul→Add→ReLU replay, with same-directory rollback-backed staging rather
than simultaneous filesystem atomicity; dynamic/empty schemas, broader ONNX ops, timing,
and device execution remain outside it. Default-domain opset-13 static dense
CPU inference remains broader, but has no dynamic shapes/control flow/external
data/quantization/custom domains/training or fetch.

### 9. P1 — bounded local CIFAR-10 binary workflow

**Status:** complete (CPU Phase A). **Owner:** `RustGrad — NN Modules & Optimizers`.

**Evidence.** `load_cifar10_files[_with_limits]` preserves explicit local batch
order and delegates each bounded canonical record stream to the existing decoder.
`tests/cifar_files_workflow.rs` proves the configured graph-free
`Conv2d → ReLU → AdaptiveAvgPool2d → Flatten → Linear` `ModuleForward` chain
trains with CPU autograd/SGD/scheduler, resumes through a portable
fresh-identity checkpoint exactly, and evaluates without mutation.

**Boundary.** Local uncompressed CIFAR-10 records only. No download, archive,
cache, augmentation, concurrent data loading, device training, dynamic shapes,
or corpus-accuracy claim is made.

### 10. P1 — restricted local PyTorch state inference

**Status:** complete (CPU Phase A). **Owner:** `RustGrad — Serialization &
Interop`.

**Evidence.** Typed bounded Torch ZIP file adapters compose the existing
protocol-2 CPU-dense parser with exact transactional `Module` state loading. A
public independent stored-ZIP fixture restores a fresh `Linear` and proves CPU
inference, deterministic identities, strict mismatch atomicity, and parser
limit/error rejection.

**Boundary.** Explicit preconfigured modules and the restricted CPU dense
state-dictionary subset only; no Python/pickle execution, general objects,
optimizer loading, model guessing, remapping/casts, device storage, network,
or fallback.

### 11. P1 — CpuSession static module train/evaluate bridge

**Status:** complete (CPU Phase C). **Owner:** `RustGrad — NN Modules & Optimizers`.

**Evidence.** `CpuModuleTrainer` bridges an existing `ModuleForward` module,
optimizer, and metric-free scheduler without raw `Graph`, `NodeId`, binding, or
gradient-map plumbing. Every request builds a fresh static graph from current
parameter versions. Public Linear acceptance proves loss decrease, one version
advance per successful step, deterministic evaluation/non-mutation, and exact
portable fresh-identity checkpoint resume.

**Boundary.** Static one-input F32-feature sparse cross-entropy classification
and first-order CPU only, except that a leading `Embedding` explicitly accepts
integer token indices for the same interpreted inference/evaluate/train seam.
Later modules remain unchanged. No generic trainer/data loader, dynamic/device/
JIT, mixed precision/Float8, metric-driven scheduler step, or second checkpoint
format is introduced.

**Evaluation evidence.** Static CPU evaluation also exposes a pure bounded
classification summary: first-tie predictions, correct/total counts, and
optional accuracy for legal empty batches, without a metrics framework.

### 12. P1 — typed Sequential CPU module composition

**Status:** complete. **Owner:** `RustGrad — NN Modules & Optimizers`.

**Evidence.** The existing `Sequential` is now a typed `ModuleForward`
container, so configured `Linear`, state-free `ReLU`, `Embedding`, `Dropout`, `LayerNorm`, `LayerNorm2d`, `GroupNorm`, `InstanceNorm`, `RMSNorm`,
`Conv1d`, `Conv2d`, `ConvTranspose1d`, `ConvTranspose2d`, `AvgPool2d`, `AdaptiveAvgPool2d`, and checked `Flatten` entries compose in declared
order through the released
`CpuModuleTrainer`, without runtime type-name dispatch or a second container.
External inputs are F32 features except that `Sequential` delegates the input
dtype policy to a leading `Embedding`, which accepts integer token indices.
Public acceptance strictly loads a fresh `Linear → ReLU → Linear` MLP with
deterministic `0.*`/`2.*` state names, proves CPU inference/trace parity,
train-step loss decrease, checkpoint fresh-identity resume, current parameter
snapshots, and evaluation non-mutation.

**Boundary.** `Linear`, `ReLU`, `Embedding`, `Dropout`, `Conv1d`, `Conv2d`, `ConvTranspose1d`, `ConvTranspose2d`,
`AvgPool2d`, `AdaptiveAvgPool2d`, `LayerNorm`, `LayerNorm2d`, `GroupNorm`, `InstanceNorm`, `RMSNorm`, checked `Flatten`, and nested `Sequential` currently
implement the one-input/one-output static forward seam. Other Conv/pool/
reshape, BatchNorm lifecycle, other normalization, and multi-input/explicit-mode/stateful modules stay
explicit; no generic model reflection, dynamic shapes, device or
mixed-precision training is claimed.

### Explicit BatchNorm CPU mode workflow

**Status:** complete (bounded CPU vertical). **Owner:** `RustGrad — NN Modules & Optimizers`.

**Evidence.** `ModeSequential` composes stateless leaves with BatchNorm through
one caller-selected mode and retains canonical state traversal. Its CPU trainer
realizes output/loss/gradients and pending statistics before preparing detached
optimizer/scheduler candidates; one existing all-lock transaction then updates
both trainable parameters and BatchNorm running buffers.

**Boundary.** Only static one-input F32 sparse-classification CPU chains such
as `Conv2d → BatchNorm → ReLU → AdaptiveAvgPool2d → Flatten → Linear`. No
implicit global mode, ordinary `Sequential` behavior change, generic trainer,
checkpoint format, native/device/dynamic/distributed, or mixed-precision
training claim is made.

### 13. P1 — graph-free static module setup and optimizer binding

**Status:** complete. **Owner:** `RustGrad — NN Modules & Optimizers`.

**Evidence.** `Linear::new_static` constructs deterministic graph-independent
host parameters, while the legacy constructor delegates unchanged for source
compatibility. `Module::trainable_parameters` is the canonical sorted,
tied-aware, trainable-only traversal, and `Optimizer::sgd_for_module` consumes
it directly. Public CPU module examples no longer create a construction Graph
or handwrite parameter names; acceptance covers legacy equivalence, nested
names, strict fresh loading, session training/evaluation, and traversal errors.

**Boundary.** This is setup ergonomics for the proven static CPU F32 module
route only. It does not add model reflection/configuration, a generic trainer,
new optimizer/state format, dynamic signatures, or device/mixed-precision
training.

### 14. P1 — graph-free static module inference

**Status:** complete. **Owner:** `RustGrad — NN Modules & Optimizers`.

**Evidence.** `infer_module_cpu` composes the existing `ModuleForward`, strict
local safetensors/restricted-Torch loading, and `CpuBackend` ownership paths
without exposing `Graph`, `NodeId`, bindings, or direct backend execution to
the user. Each call makes a fresh graph from current parameter snapshots and
returns detached output, deterministic trace, and canonical parameter-version
metadata. Public fixtures cover a fresh strict `Linear`, restricted Torch
state, and a nested two-Linear `Sequential`, including repeat determinism,
empty batches, poisoned locks, and duplicate traversal rejection before
execution.

**Boundary.** Single-input/single-output static CPU modules accept F32 feature
inputs, plus integer token indices only for a leading `Embedding`. There is no
inference cache, generic model reflection, multi-input/output signature,
dynamic/device/JIT fallback, mixed precision, or state-format change.

### 15. P1 — graph-free static ReLU MLP composition

**Status:** complete. **Owner:** `RustGrad — NN Modules & Optimizers`.

**Evidence.** `nn::ReLU` is a stateless `ModuleForward` leaf delegating to the
existing graph ReLU. The public MLP route uses graph-free Linear construction,
module-derived SGD, strict safetensors, fresh-graph CPU inference/training, and
portable checkpoint restore without user graph/binding plumbing.

**Boundary.** This is the smallest ordinary MLP composition slice only;
Broader Conv/pool/reshape and activation adapters remain separate.

### 16. P1 — graph-free static CIFAR classifier composition

**Status:** complete. **Owner:** `RustGrad — NN Modules & Optimizers`.

**Evidence.** `Conv2d::new_static`, state-free `AdaptiveAvgPool2d`, and checked
`Flatten` implement the existing `ModuleForward` seam. The public local CIFAR
route composes `Conv2d → ReLU → AdaptiveAvgPool2d → Flatten → Linear` with
module-derived SGD and `CpuModuleTrainer`; acceptance covers strict fresh
state, deterministic nested names/trace, partial and empty batches, training
loss, version updates, and evaluation non-mutation.

**Boundary.** One static CPU F32 NCHW classifier chain only. Other convolution,
pooling, normalization, reshape, multi-input/stateful, device, and dynamic
adapters remain separate.

### 17. P1 — dynamic cardinality only when a P0 proves the blocker

**Status:** bounded exact-cardinality CPU runtime DAG released; broader routes deferred.
**Owner:** `RustGrad — Symbolic Shapes`.

**User outcome.** A representative imported model or user workflow with a
runtime-sized selection can run without bespoke host glue.

**Evidence and gap.** CPU-only source-policy `nonzero`, dynamic masked select,
shared-count `Neg`/`Square` and `Add`/`Sub`/`Mul` DAGs, checked scalar operands,
composable `Sum`/`Mean`, and first-order Sum/Mean VJPs exist. Mean uses the
immediate input's exact realized all-element count at the canonical work dtype,
including zero-cardinality and scalar domains. The public
`CpuSession` path exposes the same exact shape expressions and rejects foreign
session provenance. General dynamic broadcasting, reverse rules beyond this
first-order CPU/session slice, graph-on-graph higher order,
artifact/capture/replay/native JIT, and device lowering remain deliberately
unavailable.
No current P0 acceptance has yet demonstrated that the remaining generality is
its concrete blocker.

**Dependencies and acceptance.** First attach the work to a failing P0 or
bounded importer fixture. Then add only the required typed shape/value edge,
CPU realization and regression set; fail closed elsewhere. Do not turn this
into an unbounded dynamic-IR rewrite pre-emptively.

## P2 holding area

Keep these behind the queue unless new evidence makes one a P0 blocker:

- Static ordering and guarded sampling are a staged CPU-only dependency chain:
  one ordered multi-output Sort pair unlocks `argsort` and slice-based `top_k`,
  while TensorGuard plus a commit-on-success implicit reservation unlocks
  bounded multinomial. The chain remains explicitly nondifferentiable and
  rejects capture, native/JIT, and device paths; it is not a generic scheduler,
  random-runtime, or sampling/training claim.

- Released static core-parity maintenance covers F16/BF16 F32 sum
  accumulation, shrink-backed split/chunk and schedulable static unfold, variance/std
  composition, typed like-creation and captured randperm helpers, normalized
  static index/update higher-order derivatives, Unicode rearrange identifiers,
  and presentation-whitespace einsum parsing. These are bounded CPU Graph
  surfaces, not a new P0/P1 workflow or a reason to promote dynamic/device
  breadth.

- Further released static maintenance keeps the same boundary: exact
  integral/bool UOp identity rewrites, left-biased float extrema,
  parameterized hardsigmoid and stable softplus/mish/logsigmoid,
  compositional seeded Graph dropout,
  boolean any/all, equal-width raw TensorData bitcast, bounded safetensors
  reads, MaxPool2d module composition, and weighted NLL mean semantics. None
  creates a trainer, runtime, device path, or demonstrated workflow blocker.

- The current static maintenance batch adds only checked Adam/LAMB checkpoint
  counter exhaustion, normalized-static-map `diagonal_static`, F32/F64 CPU-JIT `exp2`,
  and bounded copying-only raw tensor-file reads/writes. It does not promote dynamic
  indexing, broad native transcendental coverage, mmap/lazy storage, device
  execution, or optimizer redesign.

- Schedule artifacts now fail closed before planning, realization, or capture
  unless their contiguous IDs, ordered prior dependencies, and derived
  consumer mirrors agree exactly. This is integrity hardening, not a new
  scheduler, runtime, or device capability.

- Static `cumsum` and `cumprod` add only checked one-axis CPU prefix scans with
  typed Sum/Product UOps and artifact identity. Sums retain their checked
  integer/bool promotion while products preserve source dtype; narrow floats
  retain source storage. Floating `cumsum` reverse mode composes reverse-scan-
  reverse over the stored normalized axis and supports graph-on-graph seeds.
  Floating `cumprod` uses a compositional zero-aware reverse rule; non-floating
  and Float8 scan gradients reject before derivative graph mutation. Fixed-size masked select may use only sum's
  boolean prefix ranks to route an explicit upstream source-value cotangent;
  scan values, mask/size, dynamic/parallel scans, CPU-JIT, replay, and device
  lowering remain excluded except for that bounded floating-sum reverse edge.

- Signed-zero `Abs` and host scheduler-group atomicity are bounded maintenance:
  F16/BF16/F32/F64 CPU `Abs` retains a negative-zero lane, and a rejected
  scheduler child leaves all candidate epochs and learning rates unchanged.
  These changes add no Float8, optimizer-family, trainer, runtime, or device
  surface.

- Storage-less `LiteralScalar` resolution is likewise bounded maintenance:
  public Bool/I64/U64/F64 literals become ordinary scalar `TensorData` against
  a strong Graph operand before lowering. It adds no weak dtype/storage/UOp,
  artifact/cache identity, runtime, or device capability.

- Float8 autograd beyond typed broadcast unbroadcast, random, broader CPU-JIT/native replay beyond F32/F64 log2,
  and device execution;
  strict opt-in native static-module inference covers static F32 Linear,
  Linear→ReLU→Linear, and the released two-class configured 1×1-Conv CIFAR
  composition; general JIT coverage remains a separate deployment task. The released CPU
  transport/cast/elementwise/reduction/movement/contraction work is useful
  evidence, but it does not block the CPU workflows above.
- Additional accelerator lowering, live hardware validation, autotuning,
  tensor-core breadth, and device-resident mixed-effect state.
- Broader dynamic/control-flow semantics, rare import layouts/formats, and
  isolated tinygrad parity features without a demonstrated user workflow.

## Updating this list

- Reorder only when released evidence changes a user outcome or proves a
  dependency; move newly discovered low-impact work to P2 rather than growing
  the active queue.
- Mark an item complete only after its bounded acceptance is released and the
  corresponding CI run is green. Link that evidence from the compatibility
  ledger rather than copying it here.
- Keep product outcomes separate from shared prerequisites. The owner of a
  shared schema, public facade, or roadmap/architecture document is the one
  active writer; other subsystem owners hand off dependency changes rather
  than racing overlapping edits.
- Revisit this document with the compatibility ledger after a completed P0 or
  P1 release. It intentionally does not rank every unfinished parity row.
