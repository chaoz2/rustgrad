# Deterministic semantic fuzzing

`rustgrad::fuzz` is a bounded property and differential toolkit, not a byte-to-IR
fuzzer. Every case is constructed through typed `Graph` APIs and has an explicit
seed and case index. The current generator covers scalar, zero-domain, broadcast,
and vector-tail shapes across elementwise arithmetic, select, casts, affine
shrink/expand views, sum/mean/product reductions, and eligible static matmul.
Typed concat cases are retained in the regression corpus even though the current
captured interpreter mismatch is not included in clean random campaigns.

The CPU backend is the oracle. Captured interpreter replay is byte-exact. Strict
native replay uses exact bytes for Bool, integers, F16, and BF16; F32 uses absolute
and relative tolerance `1e-6`, while F64 uses `1e-12`. Those policies are stored
in every failure artifact. Native `Unsupported` is a typed campaign result and is
never counted as a match or converted to fallback.

Run a deterministic campaign:

```text
cargo run --bin semantic_fuzz -- run 7 64
cargo run --bin semantic_fuzz -- run 7 64 interpreter-only
```

Replay one persisted mismatch or a corpus:

```text
cargo run --bin semantic_fuzz -- replay tests/fuzz_corpus/failure-a30335b03b77b166.rgfz
cargo run --bin semantic_fuzz -- corpus tests/fuzz_corpus/*.rgfz
```

Refresh regression failure artifacts only when the underlying semantics change:

```text
cargo run --bin semantic_fuzz -- regressions tests/fuzz_corpus
```

`RGFZ` version 1 wraps a canonical JSON payload with a size, CRC-32 checksum, and
deterministic FNV-1a identity. The payload retains the seed, minimized typed case,
comparison path and policy, and expected/actual value bytes or error contract.
Decoding rejects oversized input, unknown fields, invalid shapes/storage, corrupt
checksums, identity mismatches, truncation, and trailing bytes before replay.

The checked-in failure is genuine: dense I32 concat succeeds in the CPU oracle
but captured interpreter replay currently reports an invalid dense coordinate.
The toolkit deliberately does not modify that shared compiler semantic. The
fixture makes the gap reproducible and will stop reproducing when the compiler is
fixed, at which point the corpus and compatibility ledger can be reconciled.
