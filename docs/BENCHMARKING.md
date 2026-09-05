# Offline benchmark comparison

RustGrad's benchmark comparison surface normalizes already measured evidence. It
does not execute workloads, calculate speedups, or infer missing measurements.
`BenchmarkObservation` records one implementation, workload, device, and metric
set; `BenchmarkComparison` validates and bundles observations for inspection.

## Workflow

1. Measure each implementation independently on the same physical device and
   exact workload.
2. Encode each result as a version-1 `BenchmarkObservation` JSON document. Record
   the implementation version, revision, configuration, and command that produced
   it.
3. Choose the baseline explicitly and combine at least two observations offline:

   ```text
   benchmark_compare --baseline rustgrad --output comparison.json \
     rustgrad.json tinygrad.json
   ```

The baseline must be present. All observations must have distinct frameworks and
exactly equal `BenchmarkWorkload` and `BenchmarkDevice` values. Output is ordered
canonically by framework and is deterministic; an output path is create-new and
is never overwritten.

For ResNet-18, workload identity includes the model identity, `[N,C,H,W]` input
shape, input dtype and SHA-256, and correctness contract. For GGUF Llama, it
includes the model and prompt SHA-256 values, prompt token count, generation
bound, and expected-token-ID SHA-256. Device identity includes the backend,
device name, hardware identity, and operating system. Every compared producer
must agree on those values exactly.

## Missing values and Metal mappings

An optional JSON field set to `null` means that the producer did not provide that
measurement. A numeric zero means a measured zero. Producers must not substitute
zero, an estimate, or a value from another scope for unavailable evidence.

`BenchmarkObservation::from_metal_session_scoreboard` maps the maintained
ResNet-18 Metal scoreboard into planning, pipeline compilation, native
preparation, first-run and steady-run host latency, planned kernel and static-slot
memory facts, executed kernels, host-API transfers, and fallback count. Its
`planned_device_memory_bytes` value is planned physical static tensor-slot storage,
not measured peak device memory. Transfer counts and bytes are logical host-API
payload, not physical transfer traffic.

`BenchmarkObservation::from_llama_metal_scoreboard` combines the token-step and
optional fixed-prefill components with checked sums. It records component
planning/preparation, the first physical invocation, prompt-prefill and
steady-decode phases, executed kernels, host-API transfers, and fallback count.
Shared component ownership and the absence of a comparable global sample series
mean that the raw adapter leaves `planned_device_memory_bytes`,
`measured_peak_device_memory_bytes`, `planned_kernel_count`, and
`steady_run_latency` `null`.

The maintained RustGrad adapters accept only the exact scoreboard workload labels
`RUSTGRAD_METAL_RESNET18_WORKLOAD` and
`RUSTGRAD_METAL_GGUF_LLAMA_WORKLOAD`. Callers still supply immutable workload,
implementation, and operating-system provenance.

The maintained live Llama command constructs that provenance only in its plain
prompt, fully attested scoreboard mode. It hashes the exact UTF-8 prompt bytes and
the compact UTF-8 JSON encoding of the parsed expected `u32` IDs (for example,
`[3,4]`) through the macOS `shasum -a 256` implementation. The model hash is the
workflow-verified attestation value, the prompt token count comes from the actual
generation, and the generation bound comes from the executed arguments. The
normalized observation is created directly from the validated in-memory v2
scoreboard before any evidence file is published. Both maintained live Metal
harnesses begin one `MetalDeviceBufferMeasurement` on a freshly discovered,
exclusively used RustGrad `MetalDevice` and carry that token across the workload.
Finishing the token authenticates the same device, samples its
`lifetime_high_water_physical_buffer_bytes`, checked-converts `usize` to `u64`,
and returns the `RustGradDeviceBufferPeak` to attach. Consequently their
normalized hardware observations
require `measured_peak_device_memory_bytes` to be present. This is the high-water
sum of requested native `MTLBuffer` lengths simultaneously owned by RustGrad on
that device lifetime—not allocator RSS, physical residency, driver overhead, or
unified-memory pressure. It remains separate from planned memory: ResNet retains
its adapter-derived planned static-slot bytes, while Llama planning remains
unavailable. Raw scoreboard reports are unchanged.

## External observations

tinygrad, Candle, and llama.cpp measurements enter through their own validated
`BenchmarkObservation` JSON documents using the exact framework names
`tinygrad`, `candle`, and `llama.cpp`. Each producer records only values actually
measured under the shared workload and device identity and leaves unsupported or
uncollected optional fields `null`. The comparison CLI validates and bundles
those documents; it never fabricates parity, memory, traffic, latency, or speedup
values.

The checked-in Apple-GPU lane remains dormant. Its Llama path is prepared to emit
the create-new normalized observation beside the raw scoreboard and attestation,
but the repository currently publishes no live Apple-GPU comparison measurements.
