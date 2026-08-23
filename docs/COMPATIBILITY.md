# tinygrad compatibility ledger

This ledger is RustGrad's release gate. "Feature complete" means observable capability and semantics compatible with the checked-in tinygrad revision, not identical Python APIs or files.

Status: ✅ verified · 🚧 partial · ⬜ not started · 🚫 user-approved exclusion. No item becomes ✅ without an automated acceptance test; exclusions require an explicit user decision.

Validation note: reduction semantics are covered by RustGrad regressions against the checked-in tinygrad implementation and tests. A live tinygrad differential run is still unavailable in this workspace because tinygrad reports no usable device; this is an environment-validation gap, not an unimplemented reduction semantic.

## Tensor surface

| Capability | Status | Acceptance requirement |
|---|---:|---|
| Dense data, scalars and shape validation | ✅ | Valid/invalid storage tests |
| NumPy-style broadcasting | ✅ | Matrix/row/scalar cases |
| Creation and seeded random generation | 🚧 | Static dense `full/zeros/ones/empty`, integer `arange`, `linspace`, `eye`, all static `*_like` forms, `randperm`, scaled/Glorot/Kaiming uniform/normal initializers, `one_hot`, and `meshgrid` are CPU-tested. Replayable graph random nodes use explicit per-node SplitMix64 seeds; atomic `manual_seed` plus `*_implicit` calls provides a safe serialized compatibility façade. This deliberately differs from tinygrad's stateful per-device Threefry bytes. Remaining names are `const_like`, implicit aliases for every initializer, Threefry byte compatibility, and initializer families requiring QR/linalg or sparse/convolution layouts. RustGrad `empty` deterministically zero-initializes instead of exposing allocator contents. |
| Bool, integer, fp16/bf16/fp32/fp64 and special dtypes | 🚧 | Dense tagged storage and a bool/i8-u64/f16/bf16/f32/f64 taxonomy exist. CPU supports explicit casts and mixed binary promotion; fp8, weak/pointer/image dtypes, complete tinygrad promotion/accumulation rules, and device ABI lowering remain. |
| Unary, binary, comparison and transcendental ALU | 🚧 | CPU primitives inventory includes add/sub/mul/div, neg, exp/log/exp2/log2, abs/reciprocal/square/sqrt/rsqrt, sin/cos/tan/sinh/cosh/tanh, erf/erfc, asin/acos/atan/atan2, asinh/acosh/atanh, copysign, relu, floor/ceil/trunc/round/sign, isnan/isinf/isfinite, pow, maximum/minimum, floor/trunc division, floor/C remainder, bitwise and/or/xor and shifts; comparisons, bool logic and broadcasted select are verified. Dtype matrix: predicates return bool for every stored dtype; float-only unary math promotes bool and all integer types to F32, while F16/BF16/F32/F64 retain and re-quantize their dtype; neg/relu/step/abs/square/rounding/sign retain exact bool/integer storage; binary operations use the checked-in tinygrad least-upper lattice (including signed/unsigned widening and F16+BF16→F32). Bool bitwise is AND/OR/XOR; shifts require a non-bool integer result type. Integer division/remainder by zero return `DivisionByZero`; signed MIN/-1 uses wrapping arithmetic. Shift counts must be `0 <= count < result bit width`; this deliberate CPU-oracle divergence narrows tinygrad's renderer-dependent out-of-range shift behavior. Float special values, inverse-function domains, copysign signed-zero/NaN semantics, and F16/BF16 construction/cast/unary/binary/reduction quantization are regression-tested. Float reverse mode covers smooth primitives (including special functions and atan2) plus pow broadcasting/zero guards and 50/50 extrema ties; copysign differentiates its magnitude only, matching tinygrad's predicate-based sign contract. |
| Reductions and arg reductions | 🚧 | Unified static-axis sum/mean/product/min/max and argmin/argmax support all/signed multi-axis selection and keepdim, with CPU exact storage. Focused CPU regressions verify NaN-ignoring extrema, even tie shares, zero-aware product gradients, finite differences away from nondifferentiable points, typed empty sum/mean/product behavior, explicit empty extrema/arg errors, and `ReduceGrad` trace shape/dtype. Full tinygrad dtype-accumulation and dynamic/symbolic reduction coverage remain. |
| Matmul, batched matmul and linalg | 🚧 | CPU-oracle generalized matmul supports rank-1 dot, matrix/vector forms, and broadcasted batch dimensions with promoted dense dtypes. Its inspectable `MatmulGradVjp` reuses the generalized coordinate map for second-order derivatives of upstream and the non-target operand, including vector squeeze and broadcast accumulation. Linalg beyond matmul and exhaustive tinygrad accumulation-dtype parity remain. |
| Convolution, pooling and Winograd | 🚧 | CPU-oracle NCHW/OIHW Conv2d is acceptance-tested. **Static pooling is ✅ complete:** general trailing-spatial `max_pool`/`avg_pool` support arbitrary-rank options, including checked-in 3D tuple usage, and 2D wrappers delegate to the same normalized core. Generalized max indices return checked I32 flattened original-spatial indices with earliest-index ties; NaN, padding, dilation, ceil windows, divisors, and split-tie value gradients are acceptance-tested. Pooling behavior is validated from checked-in source and RustGrad's CPU oracle; only a live tinygrad runtime differential remains unavailable in this workspace. Winograd and broader convolution-family work remain. |
| Indexing, slicing, gather/scatter and setitem | 🚧 | Checked shrink/signed slicing, integer gather, deterministic replacement scatter, and scatter-add exist. Gather/scatter-add compose through second order for floating values and upstreams, including duplicate-index accumulation; indices remain nondifferentiable. General/fancy indexing and assignment remain; replacement scatter is deliberately nondifferentiable. |
| Mask, pad, concat, stack and split | 🚧 | Constant typed padding, multi-input concat with promotion, and fixed-size `masked_select(size, fill)` exist. Unbounded boolean-mask selection has data-dependent output shape and requires the future symbolic/dynamic-shape layer; stack and split remain. |
| Reshape, permute, expand, shrink, stride and contiguous views | 🚧 | Materialized reshape/permute/expand, signed `unsqueeze`/`squeeze`/`flatten`, compositional `stack`, checked shrink, signed stride, and reverse-mode movement mappings exist. `ScatterPositionsVjp` closes second-order static shrink/pad/stride movement adjoints; true views and remaining movement ops remain. |
| Einsum/rearrange and attention helpers | 🚧 | Static dense `einsum` has a normalized inspected plan plus CPU forward and reverse oracle for arbitrary operand count, explicit/implicit output, ASCII labels, ellipses, repeated-label diagonals, broadcast-compatible labeled dimensions, scalar and zero-sized domains, and promoted storage. `EinsumGradVjp` reuses that normalized plan for second-order upstream and non-target operand derivatives, preserving diagonal and broadcast scatter accumulation. Third-order indexed-contraction derivatives remain an explicit boundary. Static `rearrange` supports tinygrad's one-arrow identifier grammar with whitespace, `() / 1` singleton axes, non-nested parenthesized split/merge groups, Unicode/ASCII ellipsis, and named static factors (one inferred group factor); it lowers to reshape/permute and inherits their reverse mode. `repeat`/`tile` and scalar-count `repeat_interleave` lower through reshape/expand, including zero repetitions and gradient accumulation. Per-element repeat tensors and symbolic/dynamic output lengths, nested groups, and remaining rearrange features are unimplemented. Stable `logsumexp`, `softmax`, `log_softmax`, and compositional scaled dot-product attention support signed/static axes, narrow-float calculation control, bool/additive/causal masks, and static GQA head replication with first-order reverse mode. Training dropout supports inverted scaling through an explicit `dropout_seed`; eval mode bypasses it. |
| Tensor I/O, NumPy, zero-copy, disk/shm/tinyfs | 🚧 | Portable checked little-endian dense bytes and safetensors state dictionaries (in-memory and atomic filesystem APIs) are implemented. Explicit module state traversal/loading is available for the foundational NN layer; Torch pickle, ONNX, NumPy interop, and device-backed/file-backed lazy tensors remain separate milestones. |

## Compiler and symbolic system

| Capability | Status | Acceptance requirement |
|---|---:|---|
| Typed backend-neutral graph | 🚧 | Initial ops and trace verified |
| Universal IR for ALU, memory, ranges and control | 🚧 | Phase-one immutable typed UOp DAG covers constants/vconsts, definitions, special/range/if delimiters, selected ALU/cast/vector/index/load/store/barrier/sink families, typed payloads/address spaces, deterministic traversal and control/effect validation. Full tinygrad op vocabulary and kernel lowering remain. |
| Symbolic integers, variables, bounds and shapes | 🚧 | Checked `i64` constants/identity-bearing variables, add/sub/neg/mul, floor div/mod, min/max, predicates, boolean/select, conservative bounds, strict bindings, symbolic broadcast/reshape products, and concrete graph specialization are tested. Tinygrad range/index values and full UOp symbolic coverage remain. |
| Pattern matching and deterministic rewriting | 🚧 | Symbolic and UOp typed deterministic match/rewrite drivers record traces; phase-one UPat supports op sets, dtypes, arguments, source patterns and named captures. Full tinygrad UPat varargs, predicates and compiler-pattern IR remain. |
| Constant folding and algebraic simplification | 🚧 | Constant/identity folding, associative canonicalization, bound-proved comparisons and boolean normalization are tested; tinygrad's modular-congruence and transcendental rewrite suites remain. |
| Lazy realization and scheduling | 🚧 | Deterministic non-mutating schedule view has typed buffer descriptors, checked byte sizes, stable cache keys, scalar/rank-N pure-elementwise lowering, and static sum/mean accumulator lowering. A deterministic conservative temporary-allocation plan supports compatible non-overlapping internal buffers only. Product/min/max, aliases/subbuffers and device realization remain. |
| Fusion, range lowering and indexing | 🚧 | Owned typed runtime bindings, checked element/byte addressing, row-major scalar/zero/rank-N domains, broadcast offset maps, symbolic shape specialization, independent elementwise execution, and fused static sum/mean output/reduction domains are tested against the CPU oracle. Renderer linearization remains. |
| Memory planning, reuse, subbuffers and alias safety | ⬜ | Memory/subbuffer/assign suites |
| Vectorization and shared/local memory | ⬜ | Kernel opts/float4 suites |
| Tensor cores and optimization search | ⬜ | Tensor-core/GEMM tests |
| Inspectable, serialized process replay | 🚧 | Text trace exists; replay remains |

## Autograd

| Capability | Status | Acceptance requirement |
|---|---:|---|
| Reverse-mode graph transform | 🚧 | IR-to-IR transform covers current ALU, movement, reductions and first-order dedicated primitives. `grad_with` supplies explicit upstream validation and `create_graph` control. |
| Broadcast/reduction/view gradients | 🚧 | Broadcast/reduction and finite-difference checks exist, including selection gradients through value branches plus product/extrema gradients across multi-axis and keepdim cases; predicates are intentionally nondifferentiable. Sum/mean, zero-aware product, max/min tie routing, reshape/permute/expand/shrink/pad/stride/concat, gather/scatter-add and select value branches retain second-order graph edges. `ReduceGradVjp` preserves product zero-count branches and extrema upstream routing; `ScatterPositionsVjp` preserves static movement adjoints. Fixed-size masked selection currently has no first-order value gradient. |
| Matmul/conv/attention gradients | 🚧 | Generalized `MatmulGradVjp`, normalized-plan `EinsumGradVjp`, `Conv2dGradVjp`, and `ConvTranspose2dGradVjp` provide inspectable second-order VJPs for their corresponding value contractions. Convolution VJPs reuse grouped stride/dilation/padding coordinate maps; transpose convolution includes output padding and 1D inherits the singleton-height lowering. Pooling is compositional: average pooling differentiates normally and max pooling inherits the explicit tie/NaN `ReduceGradVjp` contract. Compositional attention paths backpropagate through query/key/value and are covered by finite fixtures. |
| Accumulation, detach/no-grad and assignment rules | 🚧 | Gradient paths accumulate by graph addition. Inputs carry explicit tracking state; `detach` shares values while stopping its edge; `no_grad` is scoped per graph and panic-safe. Parameter writes remain version-checked. Persistent leaf `.grad` buffers and mutation/assignment autograd are not part of the static graph API. |
| Supported higher-order composition | 🚧 | Smooth unary/binary expressions, broadcasting, ordinary movement, sum/mean/product/max/min, gather/scatter-add, select value branches, generalized matmul/einsum, convolution, transpose convolution, and compositional pooling support graph-on-graph second derivatives. Third-order dedicated contractions/reductions/movement/convolution, fixed-size masked selection, and replacement scatter remain explicit boundaries. |

## Runtime, codegen and JIT

| Capability | Status | Acceptance requirement |
|---|---:|---|
| CPU semantic oracle | 🚧 | Verified for implemented ops |
| Optimized CPU renderer/compiler/JIT | ⬜ | Generated-code differential tests |
| CUDA Driver API runtime, allocator and copies | ◐ | Runtime-loaded Driver API; owned and retained primary contexts, buffers, synchronous copies, streams/events and module/launch handles are mock-tested. Owner-scoped pinned host staging and async HtoD/DtoH/DtoD transfers with explicit event tokens are mock-tested; missing native symbols fail explicitly. Primary resources use per-thread push/pop guards. Live-CUDA validation remains open. |
| CUDA kernel/PTX rendering, profiling and graph replay | ◐ | Deterministic phase-one PTX renderer and mock module/function/cache/launch path for scalar f32 elementwise stores. Primary-context `ConcurrentPtxCache` deduplicates same-owner concurrent loads, partitions owner identities, and retries failed entries; this is mock-validated only. Static primary-stream graph capture/instantiate/replay has optional native Driver symbol resolution and mock lifecycle coverage; graph parameter updates, capture invalidation diagnostics, profiling, and live CUDA validation remain open. ModuleLoadDataEx option/log layout and bounded success/failure logs are mock-verified. |
| NV, AMD/HIP, Metal, OpenCL, WebGPU and QCOM | ⬜ | Platform-gated backend suites |
| Disk and null/mock devices | ⬜ | Disk/null/mockgpu/replay suites |
| JIT capture, specialization, cache and symbolic replay | ⬜ | JIT/symbolic/footgun suites |
| Interop and zero-copy | ⬜ | Interop and lifetime tests |

## NN, models and distributed

| Capability | Status | Acceptance requirement |
|---|---:|---|
| Layers, activations, losses and normalization | 🚧 | Graph-composed `Linear`, `Embedding` (including `padding_idx`), `LayerNorm`, `RMSNorm`, replayable `Dropout`, `Conv1d`, `Conv2d`, `ConvTranspose1d`, `ConvTranspose2d`, fixed pooling, and static adaptive avg/max pooling are tested. Adaptive pooling accepts static trailing-axis output sizes with `None` pass-through axes and lowers to inspected slice/reduce/stack bins using floor-start/ceiling-end geometry; uneven overlapping-bin gradients therefore use the ordinary reduction reverse path. The checked-in tinygrad tree has no public adaptive-pooling symbol (including no `adaptive_max_pool2d` and no `return_indices` form), so RustGrad deliberately exposes no adaptive index result rather than inventing one. Transpose convolution uses tinygrad IOHW/IOK layouts and typed first-order reverse nodes; 1D lowers through the singleton-height 2D core. CNN module convolution state uses OIHW/OIK parameter layouts, grouped geometry, fan-in uniform initialization, optional bias, and deterministic state traversal; `MaxPool2d::forward_with_indices` exposes the typed fixed-pool index result. BatchNorm/GroupNorm/InstanceNorm have explicit mode/lifecycle support. Losses support typed none/sum/mean reduction, label smoothing, ignore masks, class axes, class weights (NLL), and BCE logits positive weights. Dynamic/data-dependent masking and remaining tinygrad layers remain. |
| Optimizers and learning-rate schedules | 🚧 | Explicit-gradient dense CPU `SGD` (momentum, dampening, Nesterov, L2 decay), `Adam`, and decoupled-decay `AdamW` are version-checked and state-serializable. Optimizer slots use f64 accumulation and deterministic requantization for F16/BF16. LARS/LAMB trust-ratio variants, Muon/Newton-Schulz, fused/device scheduling, gradient scaling, distributed optimizer state, and learning-rate schedules remain. |
| State traversal, safetensors, GGUF and checkpoints | 🚧 | Explicit deterministic `Module` traversal covers nested `Sequential`, buffers, tied parameters, strict/non-strict reports, fixed shape/dtype replacement, and safetensors round trips. GGUF, Torch checkpoints and full ecosystem state traversal remain. |
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

## Public elementwise helper audit

This maintained inventory maps every public helper in the checked-in
`tinygrad/mixin/elementwise.py` to RustGrad's graph surface. `xlogy`, `erfc`,
and `atan2` are not public helpers in that checked-in file; RustGrad supplies
the latter two as explicitly tested primitives, while **xlogy remains absent
from both surfaces**.

| tinygrad helper(s) | RustGrad Graph method(s) | Status |
|---|---|---:|
| `neg/add/sub/mul/div/pow`, `mod/fmod`, floor division | `neg/add/sub/mul/div/pow/modulo/fmod/floor_div/trunc_div` | ✅ |
| bitwise ops, shifts, comparisons, `logical_not` | `bit_and/bit_or/bit_xor/shl/shr`, comparisons, logical methods | ✅ |
| `maximum/minimum`, `where`, `masked_fill`, `clamp/clip` | `maximum/minimum`, `select` (and `select` composition), `clamp` | 🚧 (`masked_fill` has no named wrapper) |
| `reciprocal/trunc/sqrt/sin/cos/exp/log2/exp2/square` | same-named methods | ✅ |
| `log/log10/tan/sinh/cosh`, `asin/acos/atan`, `asinh/acosh/atanh` | same-named methods | ✅ |
| `erf`, `copysign`, `logaddexp`, `isclose`, `lerp` | same-named methods | ✅ |
| `isnan/isinf/isfinite`, `ceil/floor/round/sign/abs/relu` | same-named methods | ✅ |
| activations: `sigmoid/relu6/hardswish/hardsigmoid/hardtanh/leaky_relu/tanh/quick_gelu/gelu/swish/silu/elu/celu/selu/softplus/mish/logsigmoid/softsign/rsqrt` | same-named methods; `gelu("tanh"|"none")` | ✅ |
| `alu/ufix/usum/uprod/detach/contiguous/contiguous_backward/threefry` | no public Graph equivalent | ⬜ (IR/runtime/internal, not scalar helper parity) |

The only scalar-style checked-in helper without a direct named Graph wrapper is
`masked_fill`; it is representable as `select(condition, fill, input)`. No
other public numeric helper is silently omitted from this inventory.

Every checked-in tinygrad family must map to Rust tests or an explicit exclusion:

- `test/backend`: ops, dtype, JIT/graph, scheduler, renderer, optimizer, randomness, profiler and interop.
- `test/null`: IR, symbolic algebra, rewriting, indexing, memory planning, visualization and compile failures.
- `test/unit`: tensors, gradients, JIT cases, assignment, distributed, I/O, LLM and platform graphs.
- `test/opt`: vectorization, kernel optimization and tensor cores.
- `test/device`, `test/amd`, `test/mockgpu`: runtime, driver, renderer and emulation.
- `test/models`: representative inference and training.
- `test/external` and `test/testextra`: selected interop, fuzz, benchmark and real-world acceptance.

Completion also requires a machine-readable manifest derived from this ledger so CI rejects undocumented gaps.
# CPU scalar C JIT

The first native JIT supports static scalar-domain fused elementwise kernels:
range/index/load/store, broadcasting, integer constants, casts, select,
comparisons, guarded exact integer division/modulo/shifts, bitwise basic ALU,
and add/sub/mul/div/min/max plus neg/abs/square/relu/sqrt. F16 and BF16 are
loaded from raw `u16` storage into scalar `f32` and encoded back with deterministic
half/bfloat rounding helpers. Native failures return a status plus linear index;
they never invoke C divide-by-zero or invalid-shift behaviour.

Static sum and mean reductions are also native: normalized multi-axis and
keepdim geometry is carried in the UOp reduction marker, then rendered as
separate output and reduction domains with a defined zero accumulator. Product,
minimum, maximum, symbolic extents, and vector lanes remain rejected.
Native differential regression covers bool; I8/I32/I64; U8/U32/U64;
F16/BF16/F32/F64 for sum and mean, including raw narrow-float storage and empty
reduction domains. Mean follows the CPU oracle's f64-finalization conversion;
zero-count float means produce NaN without issuing a C divide-by-zero.

`compile_specialized` accepts the existing symbolic planning shapes and a full
binding map, validates variable identities/bounds/no extras, binds checked
concrete extents, and requires those domains to match the lowered ABI. Its cache
identity includes symbolic structure, variable identities, and binding values;
this is compile-time specialization, not a runtime-polymorphic C kernel.

The CPU JIT also has a deterministic 16-byte portable lane policy for contiguous
elementwise kernels: 16 lanes for byte storage, 8 for `u16`, 4 for `u32`/F32,
and 2 for 64-bit storage. It emits a lane main range and isolated scalar tail;
this is deliberately an unrolled portable C representation, not an alignment-
sensitive compiler vector intrinsic. Reductions and varying broadcast offsets
remain on the scalar renderer path.

There is still no SIMD, CUDA, symbolic extents, vector lanes,
or advanced ALU operations. Unsupported UOps are rejected before C compilation
rather than rendered with altered semantics.
