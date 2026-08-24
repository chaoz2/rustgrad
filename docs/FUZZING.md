# Deterministic semantic fuzzing

`rustgrad::fuzz` is a bounded property and differential toolkit, not a byte-to-IR
fuzzer. Every case is constructed through typed `Graph` APIs and has an explicit
seed and case index. The current generator covers scalar, zero-domain, broadcast,
and vector-tail shapes across elementwise arithmetic, select, casts, affine
shrink/expand views, sum/mean/product reductions, and eligible static matmul.
The minimized zero-width concat case is retained as an ordinary regression and
must match the captured interpreter byte-for-byte.

The CPU backend is the oracle. Captured interpreter replay is byte-exact. Strict
native replay uses exact bytes for Bool, integers, F16, and BF16; F32 uses absolute
and relative tolerance `1e-6`, while F64 uses `1e-12`. Those policies are stored
in every failure artifact. Native `Unsupported` is a typed campaign result and is
never counted as a match or converted to fallback. Captured-interpreter
`Unsupported` is instead a terminal coverage error: a clean interpreter campaign
accounts for every generated case as either a match or a persisted mismatch.

Run a deterministic campaign:

```text
cargo run --bin semantic_fuzz -- run 7 64
cargo run --bin semantic_fuzz -- run 7 64 interpreter-only
```

Replay one newly persisted mismatch or a corpus:

```text
cargo run --bin semantic_fuzz -- replay path/to/failure.rgfz
cargo run --bin semantic_fuzz -- corpus path/to/failure-1.rgfz path/to/failure-2.rgfz
```

Check the fixed regression cases against an existing corpus without changing it:

```text
cargo run --bin semantic_fuzz -- regressions tests/fuzz_corpus
```

Writing new failures and pruning artifacts proven resolved are separate explicit
operations. Pruning requires write mode:

```text
cargo run --bin semantic_fuzz -- regressions tests/fuzz_corpus --write
cargo run --bin semantic_fuzz -- regressions tests/fuzz_corpus --write --prune-resolved
```

The command inventories direct `.rgfz` children by deterministic identity and
reports reproduced, new, changed, resolved, unsupported, written, and pruned
states. New/current failures, changed outcomes, unsupported replay, or unpruned
resolved artifacts produce an unsuccessful exit. Writes use a temporary file in
the selected corpus directory followed by an atomic rename; only an explicit
prune removes a direct corpus child.

`RGFZ` version 1 wraps a canonical JSON payload with a size, CRC-32 checksum, and
deterministic FNV-1a identity. The payload retains the seed, minimized typed case,
comparison path and policy, and expected/actual value bytes or error contract.
File replay enforces the envelope cap before allocating file contents. Decoding
rejects oversized input, unknown fields at every nested case/outcome/policy
boundary, invalid shapes/storage, corrupt checksums, identity mismatches,
truncation, equal expected/actual outcomes, unsupported-as-failure outcomes, and
trailing bytes before replay. Replay compares both current outcomes to the
recorded pair under the stored policy and returns one typed state:
`Reproduced`, `Resolved`, `Changed`, or `Unsupported`. Minimization accepts only
candidates with the same stable mismatch category (including target error
class), so an unsupported simplification can never be blessed as a failure. The
fixed regression set, including the repaired concat case, is expected to produce
zero failures.
