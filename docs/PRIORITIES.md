# Product priorities

This is RustGrad's maintained, usability-first roadmap. It orders released
evidence into the next user outcomes; it is not a second compatibility ledger.
The [compatibility map](COMPATIBILITY.md) remains the source for implemented
surface and acceptance evidence, while [architecture](ARCHITECTURE.md) defines
the constraints for changes that make these workflows possible.

## Priority definitions

- **P0 — usable workflow:** a person can complete a common CPU-first task from
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

The queue is deliberately small and ordered. “Evidence” names what exists
today, not a promise that the user workflow is already complete.

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

**Strict Metal realization.** The separate opt-in
`CpuSession::realize_metal` route reuses this static schedule on a
caller-owned Metal device for the verified elementwise/view subset. It
preflights every item before resource work, returns detached output plus
handle-free cache/capability trace evidence, and has no fallback. Empty domains
are exact typed no-resource skips. This is not yet a device session,
model/ONNX/Linear route, or accelerator-training claim.

The release-host framework probe currently reaches typed discovery but can
return `MetalDiscovery::NoDevices` despite system hardware inventory. Its
semantic mock evidence remains valid, but this is not portable live-device
evidence until a process-visible device is available.

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

**Boundary.** Static one-input F32 sparse cross-entropy classification and
first-order CPU only. No generic trainer/data loader, dynamic/device/JIT,
mixed precision/Float8, metric-driven scheduler step, or second checkpoint
format is introduced.

**Evaluation evidence.** Static CPU evaluation also exposes a pure bounded
classification summary: first-tie predictions, correct/total counts, and
optional accuracy for legal empty batches, without a metrics framework.

### 12. P1 — typed Sequential CPU module composition

**Status:** complete. **Owner:** `RustGrad — NN Modules & Optimizers`.

**Evidence.** The existing `Sequential` is now a typed `ModuleForward`
container, so configured `Linear`, state-free `ReLU`, `Embedding`, `Dropout`, `LayerNorm`, `LayerNorm2d`, `RMSNorm`,
`Conv1d`, `Conv2d`, `ConvTranspose1d`, `ConvTranspose2d`, `AvgPool2d`, `AdaptiveAvgPool2d`, and checked `Flatten` entries compose in declared
order through the released
`CpuModuleTrainer`, without runtime type-name dispatch or a second container.
Public acceptance strictly loads a fresh `Linear → ReLU → Linear` MLP with
deterministic `0.*`/`2.*` state names, proves CPU inference/trace parity,
train-step loss decrease, checkpoint fresh-identity resume, current parameter
snapshots, and evaluation non-mutation.

**Boundary.** `Linear`, `ReLU`, `Embedding`, `Dropout`, `Conv1d`, `Conv2d`, `ConvTranspose1d`, `ConvTranspose2d`,
`AvgPool2d`, `AdaptiveAvgPool2d`, `LayerNorm`, `LayerNorm2d`, `RMSNorm`, checked `Flatten`, and nested `Sequential` currently
implement the one-input/one-output static forward seam. Other Conv/pool/
reshape, BatchNorm lifecycle, other normalization, and multi-input/explicit-mode/stateful modules stay
explicit; no generic model reflection, dynamic shapes, device or
mixed-precision training is claimed.

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

**Boundary.** Single-input/single-output static CPU F32 modules only. There is
no inference cache, generic model reflection, multi-input/output signature,
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

**Status:** deferred as a workflow blocker; bounded CPU P2 substrate released.
**Owner:** `RustGrad — Symbolic Shapes`.

**User outcome.** A representative imported model or user workflow with a
runtime-sized selection can run without bespoke host glue.

**Evidence and gap.** CPU-only `nonzero`, dynamic masked select, a narrow F32
elementwise composition set, `DynamicSum`, and first-order VJPs exist. A
bounded P2 path now gives `masked_select_dynamic`, optional `Neg`/`Square`,
and one F32 `Add`/`Sub`/`Mul` with a checked static scalar, plus scalar
`Sum`/forward-only `Mean`, an exact allocation plan and typed mixed CPU
schedule. There is deliberately no dynamic-to-dynamic binary composition,
general dynamic broadcasting, artifact/capture/replay/native JIT/device
lowering, or dynamic-mean autograd.
No current P0 acceptance has yet demonstrated that the remaining generality is
its concrete blocker.

**Dependencies and acceptance.** First attach the work to a failing P0 or
bounded importer fixture. Then add only the required typed shape/value edge,
CPU realization and regression set; fail closed elsewhere. Do not turn this
into an unbounded dynamic-IR rewrite pre-emptively.

## P2 holding area

Keep these behind the queue unless new evidence makes one a P0 blocker:

- Released static core-parity maintenance covers F16/BF16 F32 sum
  accumulation, shrink-backed split/chunk and static unfold, variance/std
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
  counter exhaustion, `StaticIndex`-lowered diagonal, F32/F64 CPU-JIT `exp2`,
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

- Float8 autograd, random, broader CPU-JIT/native replay beyond F32/F64 log2,
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
