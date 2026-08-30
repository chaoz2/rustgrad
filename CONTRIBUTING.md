# Contributing to RustGrad

RustGrad uses small, complete pull requests to keep `main` reviewable and
releasable. The project values end-to-end user outcomes over isolated surface
area.

## Choose work by priority

Before writing code, place the change in the priority ladder described in
[`docs/PRIORITIES.md`](docs/PRIORITIES.md):

1. adoption and complete CPU-first workflows;
2. shared capabilities that unblock those workflows;
3. performance and deployment work with a concrete consumer;
4. specialized parity, dtype, and backend breadth.

Tinygrad is the semantic reference for tensor and compiler behavior. Relevant
Rust projects are design references for ownership, public API shape, backend
composition, and deployment. Record source evidence when behavior is subtle;
do not copy an API merely because another framework exposes it.

## Pull request scope

One pull request should deliver one coherent outcome. For behavior or capability
changes, include the implementation, focused tests, public documentation, and
compatibility-ledger update together when they describe the same capability.
Documentation, CI, and repository-tooling PRs should instead include the
appropriate structural validation and mark implementation-only evidence as not
applicable.

Prefer:

- a complete vertical slice over several prerequisite-only PRs;
- one canonical PR based on `main` over duplicate rebases;
- explicit typed rejection over partial or silent fallback behavior;
- focused commits during review and a concise squash commit on `main`;
- follow-up issues for findings outside the declared scope.

Avoid:

- separate code, test, and documentation PRs for one feature;
- speculative backend or dtype breadth without a priority-level consumer;
- long stacks of PRs whose intermediate branches are not independently useful;
- direct pushes or force pushes to `main`;
- mixing refactors, generated churn, and behavior changes without a clear need.

### Universal-operation taxonomy

Each UOp stores one typed `Operation` whose enum variant owns its payload; do
not add a parallel kind field or an untyped payload field. Declare new variants
once in the `define_uops!` registry so it generates the enum, compatibility
kind projection, borrowed payload view, and generic family/arity/effect
metadata together. Generic taxonomy policy must not be restated as variant
lists in artifact validation, visualization, scheduling utilities, or
rewrites.

Exhaustive matches remain required where the variants have genuinely different
semantics: detailed validation, interpreter evaluation, artifact numeric tags
and version gates, schedule lowering, and each backend's renderer/capability
boundary. Those switches must fail closed for unsupported variants rather than
infer support from the generic family. Legacy `UOpKind`/`UArg` values are owned
projections only at explicit wire or backend compatibility boundaries, never
independent node state. This distinction keeps one canonical taxonomy without
hiding backend-specific contracts behind trait objects.

If stacking is necessary, keep the stack shallow, state the dependency in every
PR, and retarget the child to `main` immediately after its parent merges.

## Required evidence

Every PR must explain the applicable parts of:

- the user or architecture outcome;
- why the change belongs at its claimed priority;
- the tinygrad behavior or Rust design reference used for semantic or
  architectural changes;
- supported and deliberately unsupported boundaries;
- tests or equivalent static evidence added;
- compatibility and documentation impact.

All required GitHub Actions checks must pass on the current base revision before
merge. Local development may use focused checks, but CI is the authoritative
release gate.

## Review and merge

The integrator reviews the complete diff, public API, failure behavior,
compatibility claim, and CI result. Review findings outside scope become a new
issue or PR rather than expanding the current change without bound.

RustGrad uses squash merging so `main` receives one meaningful commit per PR.
Use the repository's established imperative commit style, for example:

- `graph: add source literal dot`
- `jit: preserve numeric bool cast truthiness`
- `docs: simplify project onboarding`

Merged branches are deleted automatically. Superseded or duplicated PRs are
closed with a link to the canonical replacement.

## Work in progress

Keep at most three active implementation PRs. Each worker owns one branch and
one worktree. The integrator owns review, merge order, conflict resolution, and
the current `main` checkout. New implementation work pauses when the review or
merge backlog exceeds that limit.
