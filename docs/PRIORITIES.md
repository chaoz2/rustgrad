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

**Status:** next. **Owner:** `RustGrad — Tensor Semantics`, coordinated with
`RustGrad — Compiler UOps & Scheduling` for the public execution boundary.

**User outcome.** A new Rust user can construct tensors, run ordinary CPU
operations, inspect a result/trace, and get typed errors without assembling a
raw `Graph` plus `HashMap` bindings for every small program.

**Evidence and gap.** `TensorData`, `Graph`, and `CpuBackend` are extensively
tested, but the README's only quickstart is intentionally low-level graph
construction and no ergonomic public tensor-session facade exists. This is the
clearest difference from tinygrad's documented eager `Tensor` workflow.

**Dependencies and acceptance.** Reuse the exact CPU oracle, dtype/shape
validation, and trace contracts; define one stable, cohesive public facade
rather than a parallel IR. Ship a copy-paste CPU quickstart and integration
evidence for construction, arithmetic/broadcasting, a reduction,
movement/indexing, realization, trace inspection, and deterministic invalid
input errors. Device selection and implicit accelerator fallback are not
required for this P0.

### 2. P0 — minimal train, resume, and evaluate workflow

**Status:** queued after the CPU session boundary. **Owner:** `RustGrad — NN
Modules & Optimizers`.

**User outcome.** A user can train a small local classifier, save it, restore
it into fresh module/optimizer/scheduler identities, and continue evaluation
or training through documented APIs.

**Evidence and gap.** The repository already has deterministic local IDX and
CIFAR parsers, NN modules, optimizers/schedulers, and public synthetic MLP
loss-decrease and portable-checkpoint acceptance. Those pieces are primarily
test assembly rather than one supported user workflow.

**Dependencies and acceptance.** Build on item 1's public CPU boundary, the
released local data parsers, and portable checkpoint contract. Add one
documented, hardware-independent end-to-end example with deterministic local
fixtures: batch, forward, loss, backward/update, metric, checkpoint, fresh
rehydration, and resumed step. It must state that this is local CPU evidence,
not downloaded-MNIST accuracy or distributed/device training.

### 3. P0 — bounded GGUF Llama prompt-to-output workflow

**Status:** queued. **Owner:** `RustGrad — NN Modules & Optimizers`, with
`RustGrad — Serialization & Interop` owning GGUF/file-boundary changes.

**User outcome.** Given a supported local GGUF, a user can validate it, format
a supported Llama chat prompt, generate/decode tokens on CPU, and understand
why an unsupported model or template is rejected.

**Evidence and gap.** `models::transformer` has checked Llama tokenizer, GGUF
schema binding, dense/packed projections, cache, generation, and chat
acceptance; it has no concise supported public walkthrough or runnable
prompt-to-output entry point. Tinygrad exposes model-oriented public workflows,
so this is more useful than another isolated backend parity row.

**Dependencies and acceptance.** Keep the fixed Llama architecture, supported
quantization/layout, and CPU/static boundaries explicit. Add a bounded
fixture-backed example or binary that covers GGUF open, `LlamaModel` binding,
prompt/tokenization, deterministic greedy and explicit-tape generation, cache
continuation, decode, and typed rejection. It must not claim generic Jinja,
other model families, live accelerator inference, or arbitrary external-model
parity.

### 4. P1 — practical interchange path for the P0 workflows

**Status:** queued. **Owner:** `RustGrad — Serialization & Interop`.

**User outcome.** Users can move static dense inputs/weights into a supported
CPU workflow without depending on undocumented internal byte handling.

**Evidence and gap.** Exact host views/copies, `.npy`, safetensors/Torch
state-dict import, bounded GGUF, and a fail-closed static ONNX importer are
implemented. They need one coherent public selection/usage guide and
workflow-level integration evidence; they are not a general NumPy, DLPack, or
dynamic ONNX runtime.

**Dependencies and acceptance.** Follow the CPU session facade, retaining
little-endian/raw-bit and bounded-parser contracts. Demonstrate one typed
host/NPY input path and one static model/state import path feeding a P0
workflow, with clear unsupported-layout/op errors. Do not widen formats or
claim zero-copy compute/device interchange.

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
