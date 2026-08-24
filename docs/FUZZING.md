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
never counted as a match or converted to fallback.

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

Check the fixed regression cases and write any newly discovered failures:

```text
cargo run --bin semantic_fuzz -- regressions tests/fuzz_corpus
```

`RGFZ` version 1 wraps a canonical JSON payload with a size, CRC-32 checksum, and
deterministic FNV-1a identity. The payload retains the seed, minimized typed case,
comparison path and policy, and expected/actual value bytes or error contract.
Decoding rejects oversized input, unknown fields, invalid shapes/storage, corrupt
checksums, identity mismatches, truncation, and trailing bytes before replay.
Replay exits unsuccessfully when a recorded mismatch has been resolved, so a
stale failure cannot be mistaken for a current reproducer. The regression
command likewise exits unsuccessfully whenever a fixed case produces a failure.
