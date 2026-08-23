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

The layout below is the target responsibility map, not an assertion that every
listed module already exists. Consult `docs/COMPATIBILITY.md` and the working
tree for the currently executable surface.

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

## Symbolic integer and shape boundary

`symbolic.rs` owns immutable, structurally ordered `SymbolicExpr` trees and
identity-bearing `SymbolicVar`s. A variable name is presentation only: its
monotonic identity prevents two independently introduced `N` variables from
aliasing. Values use checked `i64` floor-division/modulo semantics, including
negative operands; an interval that could divide by zero, or a checked overflow,
is an explicit error rather than a backend-dependent expression.

Simplification is a small typed rewrite boundary, not a tensor-op dispatcher.
It recursively folds constants, normalizes associative arithmetic/boolean
operands, and uses bounds only for proofs. Its trace records accepted rewrites,
and the bounded fixed-point driver makes rewrite inspection reproducible. More
specialized UOp patterns remain future universal-IR work.

`SymbolicShape` is a planning value beside concrete `Shape`. Binding validates
the complete variable environment and converts every non-negative dimension to
`usize`; `Graph::input_symbolic` is the intentional specialization point. No
unbound symbolic expression can reach CPU allocation or an existing graph node.

## Universal UOp boundary

`uop.rs` owns the backend-neutral immutable DAG used after the typed tensor
`Graph` has chosen an expression. It has typed payloads, address-space
metadata, structural ordering, validation and deterministic rewrites. The
portable `kernel.rs` layer adds owned typed bindings, logical element versus
byte addressing, normalized row-major/broadcast index plans, and a range/load/
store interpreter for pure elementwise graphs. Bindings clone `TensorData` at
the execution boundary, so the UOp runtime cannot borrow or alias caller
storage. The CPU backend remains the differential semantic oracle.

Future scheduling will turn validated effect/control UOps into kernel bodies;
renderers will consume that scheduled form. Rewrites only touch pure nodes and
memoize by structural identity, so they cannot reorder stores, barriers, or
control delimiters.

## Scheduling boundary

`schedule.rs` is a non-mutating deterministic planning view over a requested
Graph output. It classifies pure elementwise regions, records typed buffer
descriptors and cache keys, and lowers scalar or rank-N elementwise chains to
a single ranged UOp sink. Static sum/mean reductions fuse a pure producer and
expose accumulator initialization/update/finalization UOps; the portable
interpreter traverses separate output and reduction domains. A deterministic
temporary-plan utility only reuses caller-designated internal buffers with
non-overlapping lifetimes and compatible size/alignment. Product/min/max,
vectorization and device rendering remain explicit boundaries.

## Static-graph autograd lifecycle

Gradient recording is graph-local state. `Graph::no_grad` temporarily disables
recording only for its closure and restores the prior state even while unwinding;
there is no process-global gradient switch. Float inputs default to tracked,
while constants and explicitly frozen inputs do not. Every resulting node
carries an inspectable `requires_grad` bit derived from its value inputs.

`Graph::detach` is a value-preserving `Detach` node: it is a new tracked float
leaf, but reverse traversal deliberately does not cross its input edge. This
matches the useful tinygrad distinction between sharing a value and sharing a
gradient history.

`Graph::grad` retains a differentiable derivative graph. `Graph::grad_with`
accepts an explicit same-shaped upstream node and its `create_graph` flag
controls whether newly built derivative nodes themselves record reverse edges.
The static graph does not retain or free a tape: the graph is immutable in
meaning, and each transform appends nodes. Parameters retain their separate
versioned host-value snapshots; optimizer writes already reject stale versions.

Generalized contractions retain their normalized index descriptions in the
graph. `MatmulGradVjp` walks the same dense generalized-matmul map as the
first reverse node, while `EinsumGradVjp` retains the original `EinsumPlan`.
Both are inspectable trace operations and accumulate broadcast/diagonal
coordinates exactly; they avoid eager host-side derivative tensors. They are
second-order closures, not a claim of arbitrary-order indexed contraction:
their VJPs remain a deliberate future primitive.

`ReduceGradVjp` similarly retains normalized axes and `keepdim`. Its product
rule uses the same zero-count branch contract as the first reverse node; max
and min retain their equality/tie/NaN routing for the upstream VJP while
treating those predicate masks as nondifferentiable for input second
derivatives.

Indexed linear maps use the same approach: gather and additive scatter form
each other's adjoints, while `ScatterPositionsVjp` reads the cotangent through
the checked static start/step coordinate map used by shrink/pad/stride
backwards. Integer indices and boolean masks remain control values, never
gradient targets; replacement scatter remains explicitly nondifferentiable.
# CPU C JIT

The optional scalar CPU path is: Graph -> schedule -> UOp -> deterministic C11
source -> system shared library -> validated pointer-array ABI. `CpuJit` is
kept separate from `CpuBackend` and the portable UOp interpreter, so it is an
optimization rather than a correctness dependency.

Its stable entry point is `void rustgrad_kernel(void **buffers, const int64_t
*symbols)`. Buffer order, dtype, element count, byte length, output mutability,
alignment, ABI version, and symbol count are checked by Rust before the call.
`JitKernel` owns the dynamic library (and therefore outlives its function
pointer); calls borrow all buffers for their whole duration. The sole unsafe
boundary is `dlopen`/`dlsym` plus that C ABI call. No writable executable memory
is allocated by Rust.

Source and libraries are content-addressed below the OS temporary directory.
The key includes renderer/ABI version, host target, fixed compiler flags, and
the rendered UOp source. A process-local mutex and atomic rename prevent
duplicate publication; compiler diagnostics are bounded.
