# tinygrad compatibility ledger

This ledger is RustGrad's release gate. "Feature complete" means observable capability and semantics compatible with the checked-in tinygrad revision, not identical Python APIs or files.

Status: ✅ verified · 🚧 partial · ⬜ not started · 🚫 user-approved exclusion. No item becomes ✅ without an automated acceptance test; exclusions require an explicit user decision.

## Tensor surface

| Capability | Status | Acceptance requirement |
|---|---:|---|
| Dense data, scalars and shape validation | ✅ | Valid/invalid storage tests |
| NumPy-style broadcasting | ✅ | Matrix/row/scalar cases |
| Creation and seeded random generation | ⬜ | Shape/value/dtype/error and determinism parity |
| Bool, integer, fp16/bf16/fp32/fp64 and special dtypes | 🚧 | Dense tagged storage and a bool/i8-u64/f16/bf16/f32/f64 taxonomy exist. CPU supports explicit casts and mixed binary promotion; fp8, weak/pointer/image dtypes, complete tinygrad promotion/accumulation rules, and device ABI lowering remain. |
| Unary, binary, comparison and transcendental ALU | 🚧 | Generic ALU with add/sub/mul/div/neg/exp/log/relu exists; comparisons and full `test_ops` mapping remain |
| Reductions and arg reductions | 🚧 | Single-axis sum exists; full axes/keepdim/min/max/mean remain |
| Matmul, batched matmul and linalg | 🚧 | Rank-2 matmul exists; batching/linalg remain |
| Convolution, pooling and Winograd | ⬜ | Conv/Winograd and model parity |
| Indexing, slicing, gather/scatter and setitem | ⬜ | Indexing/assignment suites |
| Mask, pad, concat, stack and split | ⬜ | Unit/backend behavior parity |
| Reshape, permute, expand, shrink, stride and contiguous views | 🚧 | Materialized reshape/permute exist; true views and remaining movement ops remain |
| Einsum/rearrange and attention helpers | ⬜ | Rearrange/attention suites |
| Tensor I/O, NumPy, zero-copy, disk/shm/tinyfs | ⬜ | I/O, interop and lifetime suites |

## Compiler and symbolic system

| Capability | Status | Acceptance requirement |
|---|---:|---|
| Typed backend-neutral graph | 🚧 | Initial ops and trace verified |
| Universal IR for ALU, memory, ranges and control | ⬜ | UOp spec/validation mapping |
| Symbolic integers, variables, bounds and shapes | ⬜ | Symbolic suites and failure parity |
| Pattern matching and deterministic rewriting | ⬜ | Rewrite/UPat suites |
| Constant folding and algebraic simplification | ⬜ | Constant/transcendental suites |
| Lazy realization and scheduling | ⬜ | Schedule/realize semantics |
| Fusion, range lowering and indexing | ⬜ | Rangeify/linearizer suites |
| Memory planning, reuse, subbuffers and alias safety | ⬜ | Memory/subbuffer/assign suites |
| Vectorization and shared/local memory | ⬜ | Kernel opts/float4 suites |
| Tensor cores and optimization search | ⬜ | Tensor-core/GEMM tests |
| Inspectable, serialized process replay | 🚧 | Text trace exists; replay remains |

## Autograd

| Capability | Status | Acceptance requirement |
|---|---:|---|
| Reverse-mode graph transform | 🚧 | Initial IR-to-IR transform covers current ALU, movement, reduction and matmul ops |
| Broadcast/reduction/view gradients | 🚧 | Broadcast/reduction and finite-difference checks exist; full view coverage remains |
| Matmul/conv/attention gradients | ⬜ | Operator/model differential tests |
| Accumulation, detach/no-grad and assignment rules | ⬜ | Behavioral parity |
| Supported higher-order composition | ⬜ | Match tinygrad-supported cases |

## Runtime, codegen and JIT

| Capability | Status | Acceptance requirement |
|---|---:|---|
| CPU semantic oracle | 🚧 | Verified for implemented ops |
| Optimized CPU renderer/compiler/JIT | ⬜ | Generated-code differential tests |
| CUDA Driver API runtime, allocator and copies | ⬜ | Device/stream/event/failure tests |
| CUDA kernel/PTX rendering, profiling and graph replay | ⬜ | Renderer/JIT/graph suites |
| NV, AMD/HIP, Metal, OpenCL, WebGPU and QCOM | ⬜ | Platform-gated backend suites |
| Disk and null/mock devices | ⬜ | Disk/null/mockgpu/replay suites |
| JIT capture, specialization, cache and symbolic replay | ⬜ | JIT/symbolic/footgun suites |
| Interop and zero-copy | ⬜ | Interop and lifetime tests |

## NN, models and distributed

| Capability | Status | Acceptance requirement |
|---|---:|---|
| Layers, activations, losses and normalization | ⬜ | NN/operator tests |
| Optimizers and learning-rate schedules | ⬜ | Optimizer/training parity |
| State traversal, safetensors, GGUF and checkpoints | ⬜ | Round-trip tests |
| ONNX execution/import and quantization | ⬜ | ONNX suites |
| Datasets and preprocessing | ⬜ | Dataset tests |
| MNIST and end-to-end training | ⬜ | Loss/accuracy parity |
| BERT, EfficientNet, RNNT, Whisper and conv models | ⬜ | Checked-in model tests |
| Transformer/LLM, tokenizer, MLA and MoE | ⬜ | LLM and representative external tests |
| Multi-device, sharding and all-reduce | ⬜ | Multitensor/allreduce tests |

## Tooling and release quality

| Capability | Status | Acceptance requirement |
|---|---:|---|
| Graph/kernel visualization and profiling | ⬜ | Snapshot/smoke tests |
| Differential/property tests | 🚧 | Initial CPU tests exist; generated corpus remains |
| Fuzzing and minimized persistent failures | ⬜ | Replay fixtures |
| Process replay and mock accelerators | ⬜ | Replay/mockgpu parity |
| Rustfmt, Clippy warnings denied, unit/doc tests | ✅ | Clean on every change |
| Platform CI and sanitizer coverage | ⬜ | Required matrix green |
| Public API docs and examples | 🚧 | README exists; full docs remain |

## Test-family closure

Every checked-in tinygrad family must map to Rust tests or an explicit exclusion:

- `test/backend`: ops, dtype, JIT/graph, scheduler, renderer, optimizer, randomness, profiler and interop.
- `test/null`: IR, symbolic algebra, rewriting, indexing, memory planning, visualization and compile failures.
- `test/unit`: tensors, gradients, JIT cases, assignment, distributed, I/O, LLM and platform graphs.
- `test/opt`: vectorization, kernel optimization and tensor cores.
- `test/device`, `test/amd`, `test/mockgpu`: runtime, driver, renderer and emulation.
- `test/models`: representative inference and training.
- `test/external` and `test/testextra`: selected interop, fuzz, benchmark and real-world acceptance.

Completion also requires a machine-readable manifest derived from this ledger so CI rejects undocumented gaps.
