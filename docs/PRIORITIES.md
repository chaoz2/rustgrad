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

## Active queue

The queue is deliberately small and ordered. “Evidence” names what exists
today, not a promise that the user workflow is already complete.

### 1. P0 — ergonomic CPU tensor session and getting started

**Status:** complete (CPU Phase A). **Owner:** `RustGrad — Tensor Semantics`.

**User outcome.** A new Rust user can construct tensors, run ordinary CPU
operations, inspect a result/trace, and get typed errors without assembling a
raw `Graph` plus `HashMap` bindings for every small program.

**Evidence.** `CpuSession` owns one inspectable `Graph` plus explicit owned
CPU bindings. Its session-identified tensor handles construct F32 or exact
typed constants/variables, compose broadcast arithmetic, reshape, matmul and
all-axis sum, realize through the CPU oracle, inspect deterministic traces, and
build the existing first-order pure gradient nodes. Cross-session handles,
invalid rebinding shape/dtype, and unsupported device selection are structured
errors; there is no hidden device fallback or global graph. The README and a
public integration test cover the representative broadcast/reduction workflow,
trace, gradient, and repeated realization. Broader eager aliases, device
sessions, and general session-side movement/indexing convenience methods remain
future ergonomic work rather than a second IR.

**Acceptance delivered.** The facade reuses the CPU oracle, graph validation,
and trace contracts; the checked public workflow covers construction,
broadcasting, reshape, reduction, realization, trace inspection, repeatable
variable rebinding, first-order gradients, and deterministic invalid-input or
device errors. No accelerator fallback is claimed.

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

**Status:** complete (CPU Phase A). **Owner:** `RustGrad — NN Modules &
Optimizers`, with `RustGrad — Serialization & Interop` owning GGUF/file-boundary
changes.

**User outcome.** Given a supported local GGUF, a user can validate it, format
a supported Llama chat prompt, generate/decode tokens on CPU, and understand
why an unsupported model or template is rejected.

**Evidence.** `LlamaPromptWorkflow` and `examples/llama_prompt.rs` provide a
documented local-file route from checked GGUF through fixed-schema Llama
binding, tokenizer, exact supported chat rendering, CPU graph generation, and
decoded greedy text. Its fixture acceptance covers deterministic prompt/token/
text output, context rejection without a later-output leak, and malformed GGUF
rejection.

**Boundary.** This is fixed Llama, local GGUF, CPU/static, deterministic greedy
evidence. Generic Jinja, other model families, arbitrary external-model parity,
implicit RNG, live accelerator inference, and unsupported layouts remain out
of scope.

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

### 5. P1 — dynamic cardinality only when a P0 proves the blocker

**Status:** deferred discovery. **Owner:** `RustGrad — Symbolic Shapes`.

**User outcome.** A representative imported model or user workflow with a
runtime-sized selection can run without bespoke host glue.

**Evidence and gap.** CPU-only `nonzero`, dynamic masked select, a narrow F32
elementwise composition set, `DynamicSum`, and first-order VJPs exist. There is
deliberately no general dynamic broadcasting, graph scheduling/artifact, JIT,
or device lowering. No current P0 acceptance has yet demonstrated that this
missing generality is its concrete blocker.

**Dependencies and acceptance.** First attach the work to a failing P0 or
bounded importer fixture. Then add only the required typed shape/value edge,
CPU realization and regression set; fail closed elsewhere. Do not turn this
into an unbounded dynamic-IR rewrite pre-emptively.

## P2 holding area

Keep these behind the queue unless new evidence makes one a P0 blocker:

- Float8 autograd, random, CPU-JIT/native replay, and device execution; the
  released CPU transport/cast/elementwise/reduction/movement/contraction work
  is useful evidence, but it does not block the CPU workflows above.
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
