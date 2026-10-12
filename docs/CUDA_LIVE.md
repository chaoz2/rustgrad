# Native CUDA acceptance

`tests/cuda_live.rs` is an opt-in single-device acceptance suite for the existing
CUDA Driver API, PTX renderer, and prepared CUDA graph execution path. It loads
the native driver and compares device results with `CpuBackend`; it never
substitutes a mock or falls back to CPU execution. All tests are ignored in the
default suite, and an explicitly requested live run fails if its device,
capability, driver, PTX compilation, or execution is unavailable.

## Run locally

Use Rust 1.89 or newer and a working native linker. On Windows, the native CPU
JIT tests additionally require MinGW `gcc` on PATH. Install an NVIDIA driver
supporting the runtime's minimum Driver API version. The suite uses driver PTX
JIT and does not require `nvcc`, model downloads, Python, or the CUDA Toolkit.
The complete suite requires an sm_80+ device for F16/BF16 tensor-core tests.

```sh
cargo test --test cuda_live -- --ignored --nocapture --test-threads=1
```

The default device ordinal is zero. Set `RUSTGRAD_CUDA_DEVICE` to another integer
ordinal to select a device explicitly. Its name and compute capability are
printed during preparation. To run only the F32 cases on another admitted
device, select their individual test names rather than the complete suite.
Set `RUSTGRAD_CUDA_DUMP_PTX` to print each generated kernel for diagnostics.

## Checked boundaries

| Test | Contract |
| --- | --- |
| `live_cuda_elementwise_reduction_replay_and_rejected_input_retry` | F32 square, broadcast bias, and axis sum; three replays including changed inputs and return to the original input; typed malformed-shape rejection followed by a valid retry. |
| `live_cuda_matmul_serial_tails_and_empty_domains` | F32 vector/matrix, matrix tails, broadcast batches, K=0, and zero-output shapes; one prepared graph per case, replayed with two distinct input sets and the original set again. |
| `live_cuda_tensor_core_f16_bf16_exact_fixtures` | Compiler selection of tensor-core matmul for exact-tile F16/BF16 broadcast batches, followed by three native replays and exact output bytes for small integral fixtures. |

Every numerical check compares shape, dtype, and all output lanes. F32 uses
`abs(actual - expected) <= 1e-5 * max(1, abs(expected))` and rejects non-finite
outputs. The tensor-core fixtures keep products and sums exactly representable;
their byte equality does not claim general narrow-precision bitwise parity.

These tests establish bounded numerical and replay behavior. They do not
measure allocation/transfer counts, bus traffic, memory high-water, latency,
throughput, or speedup. They do not validate multi-device collectives, arbitrary
operations/dtypes, symbolic graph updates, or compiled recurrent training on
CUDA. Record the exact revision, command, OS/toolchain, GPU, driver version, and
complete test output when reporting a hardware result.
