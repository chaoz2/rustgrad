# RustGrad architecture

RustGrad is a Rust-native reimplementation of tinygrad's *capability*, not a line-by-line port.

## Ideas retained from tinygrad

- one inspectable path from tensor expression to device execution;
- a small universal IR used for algebra, shape logic, lowering, and codegen;
- lazy scheduling and aggressive graph rewriting;
- generated kernels rather than a catalog of one kernel per operation;
- differential tests, fuzzing, process replay, and mock devices.

## Ideas adopted from Rust projects

- **Luminal:** a small graph vocabulary and explicit compiler pipeline;
- **Burn:** composable backend boundaries, without forcing accelerator-specific capabilities into the common trait;
- **dfdx:** shape and resource invariants encoded in types where that improves errors without preventing dynamic shapes;
- **RustTensor:** direct CPU/CUDA differential tests and an easy-to-follow reference path;
- **cuda-oxide:** a future optional Rust-to-PTX kernel path, isolated because its toolchain is currently experimental.

## Source layout

```text
src/
  tensor.rs              public TensorData, Shape and dtype-facing API
  tensor/                creation, random and serialization implementations
  ir.rs                  typed frontend graph while the UOp layer is built
  ir/                    operation-family extensions: creation/reduce/...
  autograd.rs            reverse-mode graph transform
  uop/                   tinygrad-style universal operation IR
    ops.rs               typed operations and values
    symbolic.rs          symbolic integers and validity rules
    pattern.rs           deterministic pattern matcher and rewrites
  schedule/              realization, fusion, indexing and memory planning
  renderer/              C/LLVM/PTX/WGSL and platform renderers
  engine/                lazy realization, JIT capture and replay
  device.rs              discovery, capabilities and allocator contracts
  runtime/               CPU/CUDA/Metal/HIP/OpenCL/WebGPU/... implementations
  nn/                    layers, optimizers, datasets and state/ONNX/Torch I/O
  llm/                   the bundled language-model path
  viz/                   graph, schedule and kernel inspection
  trace.rs               inspectable and replayable compiler decisions
tests/
  differential/          RustGrad vs tinygrad/NumPy/PyTorch
  property/              generated dtype/shape/stride/view tests
  compiler/              UOp, rewrite, schedule, renderer and replay tests
  runtime/               per-device conformance suites
  models/                training and inference acceptance workloads
```

This mirrors tinygrad's responsibility flow without copying its Python mixin
mechanics. Rust extension `impl` blocks split the public API by operation family;
the compiler and runtime remain explicit typed layers.

## tinygrad-to-RustGrad mapping

| tinygrad | RustGrad | Responsibility |
| --- | --- | --- |
| `tensor.py`, `mixin/*` | `tensor.rs`, `tensor/*`, `ir/*` | public tensor semantics grouped by operation family |
| `dtype.py` | `dtype` types exposed by `tensor` | scalar/vector/image/pointer dtype rules |
| `uop/*` | `uop/*` | universal IR, symbolic values, validation and rewrites |
| `schedule/*` | `schedule/*` | rangeification, indexing, memory and multi-device scheduling |
| `renderer/*` | `renderer/*` | target source and ISA generation |
| `engine/*` | `engine/*` | realization, JIT, graph batching and workers |
| `device.py`, `runtime/*` | `device.rs`, `runtime/*` | allocation, transfers, launches and synchronization |
| `nn/*`, `llm/*` | `nn/*`, `llm/*` | ecosystem and representative workloads |
| `viz/*` | `viz/*` | compiler introspection |

The current `backend::CpuBackend` is deliberately the semantic oracle. It will
move behind the runtime/device contracts once those contracts are executable;
optimized CPU and GPU paths must match it through differential tests.
