# RustGrad contributor guide

## Product contract

RustGrad is an idiomatic Rust implementation targeting feature-complete parity
with the checked-in `tinygrad` reference. `docs/COMPATIBILITY.md` is the
audited parity ledger: mark work partial until its executable contract and
tests justify more. Never infer broad compatibility from a narrow test set.

## Sources of truth and architecture

- `src/tensor.rs`: public tensor data, shape, dtype, and owned dense storage.
- `src/ir.rs`: typed graph/UOp-like operations and shape/dtype propagation.
- `src/backend/cpu.rs`: semantic oracle used to validate implementations.
- `src/autograd.rs`: reverse-mode differentiation over graph operations.
- `src/trace.rs`: inspectable compile trace.
- `docs/ARCHITECTURE.md`: current architecture plus the planned module map.
- `docs/COMPATIBILITY.md`: current, testable scope and remaining work; it
  overrides aspirational wording elsewhere.

Read `docs/ARCHITECTURE.md` before moving or adding major modules. A path shown
there is not necessarily implemented; check the working tree and compatibility
ledger before depending on it. Preserve the documented tinygrad-inspired
structure, but do not do a line-by-line Python port. Keep accelerator-specific
capabilities out of a flattened common API.

## Invariants

- The CPU backend is the semantic oracle. Optimized paths must be tested
  differentially against it.
- Shapes and dtypes propagate through every graph operation. Do not reintroduce
  f32-only shortcuts where exact bool/integer storage is required.
- Use checked shape/index arithmetic and explicit `Error` variants for invalid
  public inputs. Document every `unsafe` boundary and its invariant.
- Ownership and RAII govern buffers, device handles, queues, and lifetimes.
- Comments explain invariants or why, not a restatement of code. Public APIs
  need docs. Avoid speculative dependencies and abstractions.

## Workflow

During iteration, run the narrowest test target or name filter that observes
the changed contract. Before every commit, run these commands from this crate
root:

```sh
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
```

`cargo test` also runs Rust doc tests; no separate documentation build or
Markdown lint command is configured. Keep documentation examples truthful and
add a doc test when an example becomes executable. Do not weaken tests or lint
policy to make a change pass.

### Task ownership and reuse

Long-lived Codex tasks own a stable area and use the name
`RustGrad — <Subsystem>`, never a milestone, phase, commit, or prompt title.
Before dispatching work, inspect existing RustGrad tasks and reuse the matching
owner when its checkout is compatible and it has no conflicting active work.
Send follow-ups back to that owner; rename a task when its durable area becomes
clear. Completion reports return to the same owner and state the achieved
boundary plus exact remaining gaps.
Create a task only when no compatible owner exists, isolation/worktree semantics
require one, or parallel work would conflict. A temporary task uses its durable
area plus a concise qualifier, then is consolidated or archived after handoff.
Task ownership is context coordination, not exclusive files: shared-worktree
agents inspect HEAD, preserve concurrent changes, avoid overlapping edits, and
verify before committing. The user's explicit task/worktree request wins.

Before parallel dispatch, identify dependencies, architectural boundaries,
likely file overlap, source-of-truth docs, and integration/commit order. Reuse
area owners for independent read-only audits, reference research, test
inventories, platform probes, or implementation in genuinely disjoint files.
Give every parallel task a deliverable, base/prerequisite state, allowed
subsystem or files, validation gate, and handoff; the coordinator tracks
dependencies and integrates findings into the owning task before code changes.

Phase shared IR/public APIs/error types, Cargo dependencies, architecture or
compatibility ledgers, generated schemas, shared mocks/fixtures, same files, or
the same commit chain. In a shared worktree, assign non-overlapping write
scopes; inspect HEAD/status before editing and committing. If overlap emerges,
stop one writer and hand off rather than racing or overwriting. Use isolated
worktrees only when repository topology supports them and isolation is needed;
do not invent them for this nested setup.

Run narrow tests concurrently only when they do not mutate shared global state.
Run full gates on integrated HEAD before completion or commit claims.
Parallelism must reduce critical-path time without weakening correctness,
reviewability, or compatibility claims; do not use it merely to keep agents busy.

Suggested areas, created only when needed: Project Coordinator; Repository &
Release; Tensor Semantics; Movement & Indexing; Reductions; Linear Algebra &
Convolution; Autograd; Compiler UOps & Scheduling; Symbolic Shapes; CPU JIT;
CUDA Driver Runtime; CUDA Memory & Peer; CUDA Mock Runtime; Collective Planning
& Runtime; Sharding & Distributed Tensors; NN Modules & Optimizers; and
Serialization & Interop.
## Test design

Design tests from observable contracts, not from implementation branches or a
target test count. Before editing production code, identify the smallest test
that would fail for the missing behavior. For a bug fix, observe that failure,
apply the fix, and keep the minimized case as a regression.

Build a small orthogonal case matrix for each semantic family:

- the representative success path;
- boundaries that change behavior, such as scalar, empty, zero, one, maximum,
  NaN, infinity, signed zero, or integer overflow;
- shape rules such as broadcasting, non-contiguous movement, and invalid ranks;
- representative dtype classes: bool, signed, unsigned, narrow float, and wide
  float when their paths differ;
- each distinct public error contract.

Use equivalence classes and pairwise cases instead of a Cartesian product. Put
cases with the same setup, operation, and assertion shape in one table-driven
test, and include a case name in every failure message. Split cases when they
exercise different invariants, require different fixtures, or would make a
failure ambiguous. Consolidated tests should remove duplicated plumbing, not
become long end-to-end scenarios with many unrelated failure causes.

Keep expected values visible beside their inputs. Helpers may construct graphs,
run a backend, or compare tensors, but must not hide the behavior being proved.
Assert the complete contract that matters: value, shape, dtype, exact error
variant, and one representative trace when lowering is part of the contract.
Use exact comparisons for bool/integer/raw-bit behavior and documented
dtype-aware tolerances for floating results.

Choose the lowest test layer that observes the contract without duplicating it:

- module unit tests cover parsers, normalization, checked index math, and other
  private invariants;
- public cross-module behavior belongs in integration-style test modules; add a
  `tests/` target when the consumer-visible crate boundary itself matters;
- optimized backends compare with the CPU semantic oracle using shared cases;
- autograd uses analytic fixtures plus central finite differences away from
  discontinuities, including broadcast accumulation;
- property tests cover broad shape, stride, dtype, and round-trip invariants and
  must preserve a reproducible seed or minimized failing case;
- model tests use the smallest end-to-end workload that crosses the intended
  boundary; they do not replace focused operator tests.

Tests must be deterministic, order-independent, and safe under parallel
`cargo test`. Use explicit seeds and non-flaky statistical bounds. If a test
touches global state, reset it and serialize only that test scope. Do not use
wall-clock sleeps, network access, external devices, or mutable user files in
the default unit suite.

## Coverage by change type

- Tensor semantics: values, shapes, broadcasting, movement, creation, and exact
  invalid-input errors.
- Dtype/storage: raw bits where relevant, casts, promotion, and mixed-dtype CPU
  oracle behavior.
- Autograd: analytic and finite-difference cases, broadcasts, accumulation, and
  explicit nondifferentiable contracts.
- Compiler rewrites/schedules: deterministic traces plus differential results;
  do not bless a changed trace without proving the semantic change is intended.
- Runtimes/backends: resource lifetime and failure cleanup plus oracle
  comparisons; hardware-only coverage stays outside the default unit suite.
- Serialization: independent known-good fixtures, malformed input cases, exact
  round trips, and deterministic bytes.
- Models: small end-to-end correctness cases and an honest supported-scope claim.

## Change and review discipline

Update compatibility and architecture documents whenever a claim or boundary
changes. Preserve user changes. This repository is only `rustgrad`; never add
the sibling reference repositories or parent workspace to commits. Do not
force-push.

Review semantic correctness first: dtype/shape validity, autodiff validity,
deterministic compiler behavior, memory/resource safety, and honest
compatibility claims. Prefer focused, reviewable changes over unrelated cleanup.
