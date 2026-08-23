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

During iteration, run the narrowest relevant test. Before every commit, run
these commands from this crate root:

```sh
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
```

`cargo test` also runs Rust doc tests; no separate documentation build or
Markdown lint command is configured. Keep documentation examples truthful and
add a doc test when an example becomes executable. Do not weaken tests or lint
policy to make a change pass.

## Tests by change type

- Tensor semantics: shapes, broadcasting, movement, creation, and error cases.
- Dtype/storage: exact values, casts, promotion, and CPU mixed-dtype oracle
  behavior; add property/differential tests as their harnesses land.
- Autograd: analytic and finite-difference regressions, including broadcasts.
- Compiler rewrites/schedules: deterministic traces and oracle differential
  tests.
- Runtimes/backends: resource lifetime/error tests plus oracle comparisons.
- Models: small end-to-end correctness cases and documented supported scope.

Every bug fix needs a regression test.

## Change and review discipline

Update compatibility and architecture documents whenever a claim or boundary
changes. Preserve user changes. This repository is only `rustgrad`; never add
the sibling reference repositories or parent workspace to commits. Do not
force-push.

Review semantic correctness first: dtype/shape validity, autodiff validity,
deterministic compiler behavior, memory/resource safety, and honest
compatibility claims. Prefer focused, reviewable changes over unrelated cleanup.
