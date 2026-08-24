# RustGrad contributor guide

## Product contract

RustGrad is an idiomatic Rust implementation targeting feature-complete parity
with the checked-in `tinygrad` reference. `docs/COMPATIBILITY.md` is the
audited parity ledger: mark work partial until its executable contract and
tests justify more. Never infer broad compatibility from a narrow test set.

## Sources of truth

- `src/tensor/mod.rs`: public tensor data, shape, dtype, and owned dense storage facade.
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

Use the Rust Book's module and test-organization guidance and the Rust API
Guidelines as the default idiom. Apply external guidance with judgment: the
checked-in product contract and executable invariants win over generic style
rules.

## Engineering design

- The CPU backend is the semantic oracle. Optimized paths must be tested
  differentially against it.
- Shapes and dtypes propagate through every graph operation. Do not reintroduce
  f32-only shortcuts where exact bool/integer storage is required.
- Use checked shape/index arithmetic and explicit `Error` variants for invalid
  public inputs. Document every `unsafe` boundary and its invariant.
- Ownership and RAII govern buffers, device handles, queues, and lifetimes.
- Comments explain invariants or why, not a restatement of code. Public APIs
  need docs.

Design for cohesion and explicit dependency direction:

- A module owns one coherent responsibility and exposes a small typed boundary.
  Split by behavior and ownership, not an arbitrary line count. Do not add a
  new subsystem to a file that already mixes unrelated responsibilities; first
  extract the relevant parser, plan, renderer, runtime resource, or test support
  behind a named module boundary.
- Group related modules into one subsystem directory with its facade and
  re-exports in `foo/mod.rs` and implementation siblings in `foo/*.rs`. Do not
  scatter one subsystem across unrelated root-level files or use a simultaneous
  `foo.rs` and `foo/` layout. Keep a cohesive leaf module as a single `foo.rs`.
  When changing an existing hybrid layout, migrate the owning module when that
  move is local and behavior-preserving; do not create unrelated file churn
  solely for uniformity.
- Dependencies flow from public composition to typed plans and then execution:
  tensor/NN APIs may depend on graph contracts; scheduling and renderers consume
  IR; runtimes execute rendered artifacts. Shared semantic layers must not
  import a concrete backend, and a backend must not reconstruct frontend intent
  from labels, debug text, or source strings.
- Separate parsing, validation/normalization, planning, pure execution, and
  side effects. Validate complete inputs before allocation, mutation, Driver
  calls, or partial state updates. Keep coordinate maps and normalized plans
  pure so the oracle, optimized path, and tests can reuse or compare them.
- Encode semantic distinctions and invalid states in enums, newtypes, and
  validated constructors. Avoid boolean mode arguments, loosely related option
  bags, stringly typed dispatch, and public structs whose fields permit invalid
  combinations.
- Prefer private or `pub(crate)` implementation details. Add a public item or
  root re-export only for a demonstrated consumer contract; document its errors,
  ownership, lifecycle, and compatibility boundary. Do not expose a concrete
  dependency or backend type through a common API without a deliberate reason.
- Introduce traits only at real substitution boundaries such as a backend,
  external I/O, clock, entropy source, or Driver dispatch. Prefer concrete types
  and pure functions within one implementation. Avoid both speculative
  abstraction and hard-coded side effects that make failures untestable.
- Keep one authoritative implementation of each semantic rule. A second
  implementation is acceptable only as an explicitly independent oracle or
  target lowering. Share typed cases and normalized metadata, not copied
  branching logic.
- Optimize for change locality: a feature should touch its owning module, its
  boundary adapter, focused tests, and the relevant ledger. If unrelated layers
  need coordinated edits, identify the missing contract and land that boundary
  first instead of spreading knowledge across the tree.

Before implementing a cross-layer feature, write down its owner, inputs,
outputs, invariants, dependency direction, side-effect boundary, and test seam.
Update `docs/ARCHITECTURE.md` when adding a public module, reversing a dependency,
introducing a shared IR/schema, or moving resource ownership. Avoid drive-by
cleanup, but leave newly touched code more cohesive than it was.

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
  private invariants and stay beside the owning module;
- tests that only use the public crate API belong under `tests/`; cross-module
  white-box tests are justified only when an internal typed boundary is itself
  the contract;
- optimized backends compare with the CPU semantic oracle using shared cases;
- autograd uses analytic fixtures plus central finite differences away from
  discontinuities, including broadcast accumulation;
- property tests cover broad shape, stride, dtype, and round-trip invariants and
  must preserve a reproducible seed or minimized failing case;
- model tests use the smallest end-to-end workload that crosses the intended
  boundary; they do not replace focused operator tests.

When an inline test module or fixture set obscures its production module, move
it to a sibling test module while retaining private access, or to `tests/` when
only public behavior is needed. Keep reusable fixtures narrow and owned by the
semantic family they serve; do not create a global test-helper dependency that
couples unrelated subsystems.

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
compatibility claims. Then review architecture: responsibility cohesion,
dependency direction, public-surface growth, duplicated semantics, test seams,
and whether the change remains local. A passing test suite does not justify a
new cross-layer dependency or another responsibility in an already mixed
module. Prefer focused, reviewable changes over unrelated cleanup.

Handoffs for architectural changes state the boundary added or moved, why the
dependency direction is valid, which behavior is independently testable, and
any remaining coupling. Completion reports must distinguish implemented
semantics from structural debt; do not call a feature complete when its only
working path depends on an acknowledged temporary coupling.
