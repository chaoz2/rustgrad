# tinygrad compatibility ledger

This ledger is RustGrad's release gate. "Feature complete" means observable capability and semantics compatible with the checked-in tinygrad revision, not identical Python APIs or files.

Status: ✅ verified · 🚧 partial · ⬜ not started · 🚫 user-approved exclusion. No item becomes ✅ without an automated acceptance test; exclusions require an explicit user decision.

Validation note: reduction semantics are covered by RustGrad regressions against the checked-in tinygrad implementation and tests. A live tinygrad differential run is still unavailable in this workspace because tinygrad reports no usable device; this is an environment-validation gap, not an unimplemented reduction semantic.

## Multi-device collectives

| Capability | Status | Acceptance requirement |
|---|---:|---|
| Inspectable collective planning and dense reference execution | 🚧 | `collective.rs` provides deterministic semantic `DeviceId`/ordered `DeviceGroup`, broadcast, all-gather, reduce-scatter, and sum all-reduce plans plus a stepwise in-memory executor. It is tested across 1–4 devices, empty/uneven partitions, narrow storage, trace dependencies, and validation errors. |
| CUDA collective execution | 🚧 | Sequential one-through-four-device sum all-reduce executes existing plan actions through primary pooled peer copies and typed add kernels against the deterministic mock. Other collectives, overlap, narrow dtypes, NCCL/process transport, and live CUDA validation remain pending. |
| NCCL/distributed process collectives/live multi-GPU validation | ⬜ | Not claimed. Checked-in tinygrad evidence is reduction-driven `ALLREDUCE` in `tinygrad/schedule/multi.py` and naive/ring/all-to-all schedule selection in `tinygrad/schedule/allreduce.py`; it is not a distributed-process API inventory. |

| Static tensor sharding layouts and exact reference data movement | 🚧 | `sharding.rs` implements immutable replicated/equal-axis layouts over the stable collective device group; validates ordered exact dense shards; and has exact `shard`/`gather`/replicate/redistribute reference movement. Evidence: checked-in `tinygrad/tensor.py::Tensor.shard` and `tinygrad/uop/ops.py::_shard`; uneven axis dimensions are rejected exactly as tinygrad does. Static movement and collective-lowering decisions are inspectable independently of Graph/runtime execution. |
| Graph/CPU/autograd sharding composition | 🚧 | `sharded_graph.rs` validates graph-bound local node wrappers and composes shrink/concat gather, explicit redistribution, local elementwise/select/movement, sharded-axis sum/mean, and rank-two matmul through the existing CPU/reverse-mode graph semantics. Local binary traces retain ordered per-rank operand/provenance records for future transfer-to-local fusion; canonical schedule-buffer attachment, CUDA scheduling, lazy multi-device realization, and runtime collectives remain Phase 3 boundaries. |
| Deterministic sharded CUDA planning | 🚧 | `sharded_cuda_plan.rs` validates semantic-device/primary-owner/capability bindings and serializes local schedule/PTX cache stages plus trace-derived transfer/all-reduce stages without entering CUDA or materializing data. It retains renderable static elementwise/select/cast, exact Neg/Abs, and static reduction stages; graph-specific reduction/matmul and every unsupported renderer contract remain explicit diagnostics carrying the renderer reason. Driver allocation, streams, module load, and launch remain Phase 3B. |
| Executable CUDA plan companion | 🚧 | Graph-derived local Add has cache/oracle evidence across 1/2/4 owners for F32/I32/U64, including canonical zero routes. Static-view cast, broadcasted boolean select, and exact I32/F32 `GraphUnary` Neg execute through the generic semantic Mock across 1/2/4 owners; Neg retains original view-source bindings, owner-scoped cache isolation, and logical zero-domain no-work behavior. Unsupported unary/dtype pairs remain typed diagnostics. Typed transfer→local composition substitutes exact descriptor-matched outputs into local inputs and preserves a validated dependency DAG; direct provenance fusion validates explicit external-materialized ABI order and has CPU-byte evidence for 1/2/4-owner axis→replicated Add, 2-owner axis0→axis1 Add, and zero-domain no-work execution. Computed-shrink broadcast, collectives, and live CUDA remain pending. |
| Concrete redistribution routes | 🚧 | Typed graph-derived two/four-owner axis-to-replicated, replicated-to-axis, and axis-to-axis routes validate ordered owners, layouts, graph-buffer identities, dtype, and checked element/byte ranges before exact mock execution. Same-owner DtoD and cross-owner peer transfers retain deterministic route order; injected DtoD/peer failures restore external sources for retry, including composed local execution. Retained-allocator stat assertions, collectives, and live CUDA remain pending. |

## Tensor surface

| Capability | Status | Acceptance requirement |
|---|---:|---|
| Dense data, scalars and shape validation | ✅ | Valid/invalid storage tests |
| NumPy-style broadcasting | ✅ | Matrix/row/scalar cases |
| Creation and seeded random generation | 🚧 | Static dense `full/zeros/ones/empty`, integer `arange`, `linspace`, `eye`, all static `*_like` forms, `randperm`, scaled/Glorot/Kaiming uniform/normal initializers, `one_hot`, and `meshgrid` are CPU-tested. Replayable graph random nodes use explicit per-node SplitMix64 seeds; atomic `manual_seed` plus `*_implicit` calls provides a safe serialized compatibility façade. This deliberately differs from tinygrad's stateful per-device Threefry bytes. Remaining names are `const_like`, implicit aliases for every initializer, Threefry byte compatibility, and initializer families requiring QR/linalg or sparse/convolution layouts. RustGrad `empty` deterministically zero-initializes instead of exposing allocator contents. |
| Bool, integer, fp16/bf16/fp32/fp64 and special dtypes | 🚧 | Dense tagged storage and a bool/i8-u64/f16/bf16/f32/f64 taxonomy exist. CPU supports explicit casts and mixed binary promotion; fp8, weak/pointer/image dtypes, complete tinygrad promotion/accumulation rules, and device ABI lowering remain. |
| Unary, binary, comparison and transcendental ALU | 🚧 | CPU primitives inventory includes add/sub/mul/div, neg, exp/log/exp2/log2, abs/reciprocal/square/sqrt/rsqrt, sin/cos/tan/sinh/cosh/tanh, erf/erfc, asin/acos/atan/atan2, asinh/acosh/atanh, copysign, relu, floor/ceil/trunc/round/sign, isnan/isinf/isfinite, pow, maximum/minimum, floor/trunc division, floor/C remainder, bitwise and/or/xor and shifts; comparisons, bool logic and broadcasted select are verified. Dtype matrix: predicates return bool for every stored dtype; float-only unary math promotes bool and all integer types to F32, while F16/BF16/F32/F64 retain and re-quantize their dtype; neg/relu/step/abs/square/rounding/sign retain exact bool/integer storage; binary operations use the checked-in tinygrad least-upper lattice (including signed/unsigned widening and F16+BF16→F32). Bool bitwise is AND/OR/XOR; shifts require a non-bool integer result type. Integer division/remainder by zero return `DivisionByZero`; signed MIN/-1 uses wrapping arithmetic. Shift counts must be `0 <= count < result bit width`; this deliberate CPU-oracle divergence narrows tinygrad's renderer-dependent out-of-range shift behavior. Float special values, inverse-function domains, copysign signed-zero/NaN semantics, and F16/BF16 construction/cast/unary/binary/reduction quantization are regression-tested. Float reverse mode covers smooth primitives (including special functions and atan2) plus pow broadcasting/zero guards and 50/50 extrema ties; copysign differentiates its magnitude only, matching tinygrad's predicate-based sign contract. |
| Reductions and arg reductions | 🚧 | Unified static-axis sum/mean/product/min/max and argmin/argmax support all/signed multi-axis selection and keepdim, with CPU exact storage. Focused CPU regressions verify NaN-ignoring extrema, even tie shares, zero-aware product gradients, finite differences away from nondifferentiable points, typed empty sum/mean/product behavior, explicit empty extrema/arg errors, and `ReduceGrad` trace shape/dtype. Full tinygrad dtype-accumulation and dynamic/symbolic reduction coverage remain. |
| Matmul, batched matmul and linalg | 🚧 | CPU-oracle generalized matmul supports rank-1 dot, matrix/vector forms, and broadcasted batch dimensions with promoted dense dtypes. Static PTX has a correctness-first serial-K renderer for homogeneous F32/F64 plans with lhs/rhs/output ABI and mock semantic execution; narrow floats, bool, exact integers, tiling/shared memory/tensor cores, and live-CUDA validation remain explicit gaps. Its inspectable `MatmulGradVjp` reuses the generalized coordinate map for second-order derivatives of upstream and the non-target operand, including vector squeeze and broadcast accumulation. Linalg beyond matmul and exhaustive tinygrad accumulation-dtype parity remain. |
| Convolution, pooling and Winograd | 🚧 | CPU-oracle NCHW/OIHW Conv2d is acceptance-tested. **Static pooling is ✅ complete:** general trailing-spatial `max_pool`/`avg_pool` support arbitrary-rank options, including checked-in 3D tuple usage, and 2D wrappers delegate to the same normalized core. Generalized max indices return checked I32 flattened original-spatial indices with earliest-index ties; NaN, padding, dilation, ceil windows, divisors, and split-tie value gradients are acceptance-tested. Pooling behavior is validated from checked-in source and RustGrad's CPU oracle; only a live tinygrad runtime differential remains unavailable in this workspace. Winograd and broader convolution-family work remain. |
| Indexing, slicing, gather/scatter and setitem | 🚧 | Checked shrink/signed slicing, integer gather, deterministic replacement scatter, and scatter-add exist. `ir::indexing` provides typed immutable static mixed indexing (integer/slice/newaxis/ellipsis and constant broadcasted integer tensors), a narrow plan-carrying Graph op, CPU execution, and duplicate-accumulating F32 reverse scatter. `ir::dynamic` adds CPU-oracle `nonzero` with a typed rank/dtype contract and realization-time `[count, rank]` shape. Static index values currently validate out-of-bounds eagerly, deliberately differing from tinygrad's zero-filled advanced-index read behavior. Functional assignment, dynamic boolean masking, schedule/JIT lowering, and mutable alias semantics remain separate boundaries. |
| Mask, pad, concat, stack and split | 🚧 | Constant typed padding, multi-input concat with promotion, and fixed-size `masked_select(size, fill)` remain compatible. The typed dynamic-result CPU oracle now realizes both `nonzero` and unbounded `masked_select_dynamic`, including checked bool-mask broadcasting and exact runtime output shapes; its F32 first-order VJP executor scatters a validated dynamic upstream into source positions. Wiring that VJP into static Graph reverse mode, higher order gradients, and schedule/JIT/PTX/device lowering remain explicit boundaries. Stack and split remain. |
| Reshape, permute, expand, shrink, stride and contiguous views | 🚧 | Materialized reshape/permute/expand, signed `unsqueeze`/`squeeze`/`flatten`, compositional `stack`, checked shrink, signed stride, and reverse-mode movement mappings exist. `ScatterPositionsVjp` closes second-order static shrink/pad/stride movement adjoints; true views and remaining movement ops remain. |
| Einsum/rearrange and attention helpers | 🚧 | Static dense `einsum` has a normalized inspected plan plus CPU forward and reverse oracle for arbitrary operand count, explicit/implicit output, ASCII labels, ellipses, repeated-label diagonals, broadcast-compatible labeled dimensions, scalar and zero-sized domains, and promoted storage. `EinsumGradVjp` reuses that normalized plan for second-order upstream and non-target operand derivatives, preserving diagonal and broadcast scatter accumulation. Third-order indexed-contraction derivatives remain an explicit boundary. Static `rearrange` supports tinygrad's one-arrow identifier grammar with whitespace, `() / 1` singleton axes, non-nested parenthesized split/merge groups, Unicode/ASCII ellipsis, and named static factors (one inferred group factor); it lowers to reshape/permute and inherits their reverse mode. `repeat`/`tile` and scalar-count `repeat_interleave` lower through reshape/expand, including zero repetitions and gradient accumulation. Per-element repeat tensors and symbolic/dynamic output lengths, nested groups, and remaining rearrange features are unimplemented. Stable `logsumexp`, `softmax`, `log_softmax`, and compositional scaled dot-product attention support signed/static axes, narrow-float calculation control, bool/additive/causal masks, and static GQA head replication with first-order reverse mode. Training dropout supports inverted scaling through an explicit `dropout_seed`; eval mode bypasses it. |
| Tensor I/O, NumPy, zero-copy, disk/shm/tinyfs | 🚧 | Portable checked little-endian dense bytes and safetensors state dictionaries (in-memory and atomic filesystem APIs) are implemented. Fail-closed Torch ZIP and legacy ustar TAR state-dict importers accept bounded protocol-2 CPU dense tensors through a no-code-execution pickle whitelist, retaining raw storage bits and validating non-overlapping stride reconstruction before materialization. A bounded default-domain opset-13 ONNX CPU inference slice lowers concrete initializers and embedded tensor attributes; common arithmetic, predicates, selection and unary math; static Clip and inference-only no-op Dropout; Shape, Expand, Tile, same-rank constant-index Gather, constant-control Slice, positive constant-mode Pad, ConstantOfShape, static ReduceSum/Mean/Prod/Min/Max, and first-tie ArgMax/ArgMin; static NCHW 2D Conv; inference-only BatchNormalization; GlobalAveragePool; and static 2D float MaxPool/AveragePool. Dynamic/control-flow/custom-domain, external data, dynamic indexing/pads/slices/reduction axes, Pad crop/reflect/edge, general-rank Gather, last-tie ArgMin/ArgMax, MaxPool index outputs/storage-order variants, BatchNormalization training/stat outputs, nonzero/training Dropout, CUDA/sparse/quantized/custom objects, NumPy interop, and device-backed/file-backed lazy tensors remain separate boundaries. |

## Compiler and symbolic system

| Capability | Status | Acceptance requirement |
|---|---:|---|
| Typed backend-neutral graph | 🚧 | Initial ops and trace verified |
| Universal IR for ALU, memory, ranges and control | 🚧 | Phase-one immutable typed UOp DAG covers constants/vconsts, definitions, special/range/if delimiters, selected ALU/cast/vector/index/load/store/barrier/sink families, typed payloads/address spaces, deterministic traversal and control/effect validation. Full tinygrad op vocabulary and kernel lowering remain. |
| Symbolic integers, variables, bounds and shapes | 🚧 | Checked `i64` constants/identity-bearing variables, add/sub/neg/mul, floor div/mod, min/max, predicates, boolean/select, conservative bounds, strict bindings, symbolic broadcast/reshape products, and concrete graph specialization are tested. Tinygrad range/index values and full UOp symbolic coverage remain. |
| Pattern matching and deterministic rewriting | 🚧 | Symbolic and UOp typed deterministic match/rewrite drivers record traces; phase-one UPat supports op sets, dtypes, arguments, source patterns and named captures. Full tinygrad UPat varargs, predicates and compiler-pattern IR remain. |
| Constant folding and algebraic simplification | 🚧 | Constant/identity folding, associative canonicalization, bound-proved comparisons and boolean normalization are tested; tinygrad's modular-congruence and transcendental rewrite suites remain. |
| Lazy realization and scheduling | 🚧 | Producer-aware schedule DAG/lazy realization records interpreter/native-JIT/fallback traces. Generation-checked HostSlotPool leases/views/detached outputs and exact-compatible MemoryPlan reuse are alias-safe. Late `LinearKernel` builds a typed portable contiguous lane plan plus deterministic instruction/register allocation. Backend-neutral `MemorySpacePlan` validates global/register/private/shared identities, liveness aliases, and uniform barriers; `VectorProgram` maps legal lanes to typed physical-register instructions with masks/tail and deterministic cache identity, validated by CPU JIT before portable C rendering. Current elementwise plans retain an explicit no-shared-promotion decision. Target ISA SIMD lowering, actual workgroup allocation/tiled reductions, and tensor cores remain boundaries. |
| Fusion, range lowering and indexing | 🚧 | Static direct/nested shrink lowers through `ViewMap`/`ViewBufferIndex` across schedule, interpreter, and PTX. Computed-value shrink and other movement families remain explicit lowering boundaries. |
| Memory planning, reuse, subbuffers and alias safety | ⬜ | Memory/subbuffer/assign suites |
| Vectorization and shared/local memory | ⬜ | Kernel opts/float4 suites |
| Tensor cores and optimization search | ⬜ | Tensor-core/GEMM tests |
| Inspectable, serialized process replay | 🚧 | Text trace and retained concrete in-memory schedule replay exist; serialized process replay remains. |

## Autograd

| Capability | Status | Acceptance requirement |
|---|---:|---|
| Reverse-mode graph transform | 🚧 | IR-to-IR transform covers current ALU, movement, reductions and first-order dedicated primitives. `grad_with` supplies explicit upstream validation and `create_graph` control. |
| Broadcast/reduction/view gradients | 🚧 | Broadcast/reduction and finite-difference checks exist, including selection gradients through value branches plus product/extrema gradients across multi-axis and keepdim cases; predicates are intentionally nondifferentiable. Sum/mean, zero-aware product, max/min tie routing, reshape/permute/expand/shrink/pad/stride/concat, gather/scatter-add and select value branches retain second-order graph edges. `ReduceGradVjp` preserves product zero-count branches and extrema upstream routing; `ScatterPositionsVjp` preserves static movement adjoints. Fixed-size masked selection currently has no first-order value gradient. |
| Matmul/conv/attention gradients | 🚧 | Generalized `MatmulGradVjp`, normalized-plan `EinsumGradVjp`, `Conv2dGradVjp`, and `ConvTranspose2dGradVjp` provide inspectable second-order VJPs for their corresponding value contractions. Convolution VJPs reuse grouped stride/dilation/padding coordinate maps; transpose convolution includes output padding and 1D inherits the singleton-height lowering. Pooling is compositional: average pooling differentiates normally and max pooling inherits the explicit tie/NaN `ReduceGradVjp` contract. Compositional attention paths backpropagate through query/key/value and are covered by finite fixtures. |
| Accumulation, detach/no-grad and assignment rules | 🚧 | Gradient paths accumulate by graph addition. Inputs carry explicit tracking state; `detach` shares values while stopping its edge; `no_grad` is scoped per graph and panic-safe. Host-only parameters bind by stable identity/current version into graph-local captured leaves; old graphs retain old values, tied handles share one leaf per version, and optimizer gradients carry identity/version stamps. Persistent leaf `.grad` buffers and mutation/assignment autograd are not part of the static graph API. |
| Supported higher-order composition | 🚧 | Smooth unary/binary expressions, broadcasting, ordinary movement, sum/mean/product/max/min, gather/scatter-add, select value branches, generalized matmul/einsum, convolution, transpose convolution, and compositional pooling support graph-on-graph second derivatives. Third-order dedicated contractions/reductions/movement/convolution, fixed-size masked selection, and replacement scatter remain explicit boundaries. |

## Runtime, codegen and JIT

| Capability | Status | Acceptance requirement |
|---|---:|---|
| CPU semantic oracle | 🚧 | Verified for implemented ops |
| Optimized CPU renderer/compiler/JIT | 🚧 | `CpuJitBackend` provides an internal cached native execution boundary for eligible static kernels; schedule-DAG integration and broader differential coverage remain. Pure `rangeify` static shrink planning now owns nested storage-offset normalization before kernel lowering; computed movement and validity predicates remain explicit boundaries. |
| CUDA Driver API runtime, allocator and copies | ◐ | Runtime-loaded Driver API; owned and retained primary contexts, buffers, synchronous copies, streams/events and module/launch handles are mock-tested. `PrimaryPoolStats` reports one exact allocator handle, with `pool_id` distinguishing separate pools on one owner and clones sharing state. The deterministic mock models owner-scoped primary device bytes, including colliding raw handles, and applies async HtoD/DtoH/DtoD and peer-copy bytes at submission while preserving event readiness behavior. Owner-scoped pinned host staging and async transfers with explicit event tokens are mock-tested; missing native symbols fail explicitly. Sharded accounting must still query retained allocator handles; PTX semantic kernels and collective CUDA execution remain pending; live-CUDA validation remains open. |
| CUDA kernel/PTX rendering, profiling and graph replay | ◐ | Deterministic phase-one PTX renderer and mock module/function/cache/launch path. Generic retained PTX carries its lowered UOp only for test dispatch; owner-scoped semantic mock coverage includes exact i32/i64/f32/f64 `neg`/`abs` over scalar, broadcast, and static views, plus correctness-first static sum/mean/product/min/max for every current stored dtype. Product uses typed wrapping storage semantics (Bool AND); extrema use the CPU `f64` ordering projection with NaN-ignore/first-tie selection while retaining raw words, including F16/BF16 conversion/requantization. Optimized reductions, other unary operations, broader generic coverage, and live CUDA validation remain open; `NativeDispatch` is unchanged. |
| OpenCL runtime and OpenCL C rendering | 🚧 | Runtime-loaded OpenCL 1.2 ICD with exact symbol errors; thread-confined RAII resources; checked copies/build logs/owner and capability-aware cache/launch preflight. Static OpenCL C and owner-scoped semantic mock cover contiguous/broadcast plus contiguous shrink views, Bool/I32/U32/F32 and capability-gated I64/U64/F64 elementwise wrapping arithmetic/compare/select/narrow casts, zero domains, and exact fp64-gated serial F32/F64 Sum/Mean including multi-axis/keepdim/empty domains. Ordered schedule bindings and view/reduction/capability metadata participate in ABI/source identity; native execution has no host fallback. F16/BF16, non-contiguous views, guarded integer div/mod/shift status, Product/Min/Max, broader unary ops, polymorphic shapes, and broad live-device validation remain. |
| NV, AMD/HIP, Metal, WebGPU and QCOM | ⬜ | Platform-gated backend suites |
| Disk and null/mock devices | 🚧 | Checked owned file byte windows and deterministic null planning traces are tested; file-backed lazy tensors, mmap, and mockgpu/replay parity remain. |
| JIT capture, specialization, cache and symbolic replay | 🚧 | `CapturedSchedule` replays retained concrete schedule UOps through the checked interpreter with ordered ABI/input preflight, without Graph traversal or CUDA capture. Native batching, symbolic runtime replay, and CUDA graph integration remain absent. |
| Interop and zero-copy | ⬜ | Interop and lifetime tests |

## NN, models and distributed

| Capability | Status | Acceptance requirement |
|---|---:|---|
| Layers, activations, losses and normalization | 🚧 | Graph-composed modules include `LayerNorm2d` and checked-in `LSTMCell` constructor/forward/state/autograd coverage alongside the existing normalization, convolution, pooling, and loss surface. Tinygrad layout/init gates are tested; dynamic/data-dependent masking and remaining layers remain. |
| Optimizers and learning-rate schedules | 🚧 | Static host `Muon` joins SGD/Adam/AdamW/LARS/LAMB with checked rank/float/config validation, momentum/Nesterov, and tinygrad-style Newton--Schulz matrix orthogonalization. `OptimizerGroup` and host MultiStep, ReduceLROnPlateau, CosineAnnealing, and OneCycle schedulers have checked state/config contracts. Existing optimizer checkpoints retain config fingerprints, legacy rejection, and strict expected-key loads; LAMB uses corrected bias correction. Fused/device paths, distributed state, and loss/gradient scaling remain. |
| State traversal, safetensors, GGUF and checkpoints | 🚧 | Explicit deterministic `Module` traversal covers nested `Sequential`, buffers, tied parameters, strict/non-strict reports, fixed shape/dtype replacement, and safetensors round trips. Graph-independent stable host `Parameter` identities bind versioned leaves per Graph. `TrainingCheckpoint` remains the strict same-identity in-process path. The separate bounded `PortableTrainingCheckpoint` restores module values/versions, tie topology, one `Optimizer`'s parameter-group ownership/state/LRs, and one scheduler into fresh process-local identities with staged validation and deterministic bytes. The bounded in-memory GGUF v2/v3 reader preserves typed metadata, validates dense and source-evidenced quantized tensor descriptors/ranges, materializes exact dense F32/F16/I8/I16/I32/I64/F64/BF16 storage, and dequantizes Q4_0/Q8_0/Q4_K/Q6_K blocks to F32. Atomic whole-file F32 state mapping fails on any unsupported quantized tensor. All other quantized layouts remain opaque. Optimizer-group orchestration, Graph/device state, GGUF model loading/split files/LLM execution, full Torch checkpoint compatibility, and full ecosystem state traversal remain. |
| ONNX execution/import and quantization | 🚧 | Bounded fail-closed default-domain opset-13 static inference import lowers typed/raw initializers; elementwise/activations; movement/indexing/shape; Gemm/MatMul and softmax; Conv/pooling/BatchNorm/GlobalAveragePool; predicate/Where/Clip/inference Dropout; ConstantOfShape; and reductions/args to the CPU oracle. The checked acceptance set is limited to the imported static fixtures. Dynamic controls/shapes, broader Gather/index semantics, control flow, sequences, sparse/quantized/external data, custom domains/opsets, training, and live external-model differential validation remain unsupported. |
| Datasets and preprocessing | 🚧 | The `datasets` facade and private format modules validate local uncompressed MNIST IDX pairs and exact-count CIFAR-10 binary records. CIFAR label/channel layout, checked lengths, zero records, and explicit per-channel F32 normalization are acceptance-tested; seeded batching is deterministic. No downloads, caching, random crop/flip, or global RNG are provided. |
| MNIST, CIFAR and end-to-end training | 🚧 | A synthetic IDX two-layer MLP proves deterministic loss decrease and uninterrupted/in-process-resumed agreement. A tiny public CIFAR fixture composes parsing and normalization through Conv2d, pooling, Linear, and the CPU oracle. These are not downloaded-corpus accuracy or loss-parity claims; real corpus training, evaluation, augmentation, and checkpoint portability remain. |
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

There is still no target SIMD/vector-instruction lowering, CUDA, symbolic
extents, or broad advanced ALU coverage. Unsupported UOps are rejected before C
compilation rather than rendered with altered semantics.

# CUDA allocation cache phase 3B0 representation foundation

CUDA pooled leases now preserve logical capacity independently from physical
allocation classes and support primary-context shared pool state. Checked views
are used at the public lease boundary and PTX bindings can carry a `BufferView`.
Primary pooled physical allocations are now primary-only `PrimaryBlock` values
with retained primary identity, generation and explicit cleanup; logical leases
and views never round-trip through mixed `DeviceBuffer`. `PrimaryEventFence` is
shareable and validates its primary owner. The primary allocator now keeps
completion fences in a deferred registry and promotes completed generations
without synchronizing ordinary pooled primary PTX launches. Direct and owned-context
Primary pooled leases also support directional async peer copies through
`PeerAccess`, retaining both leases and a shared completion fence. Owned and
direct buffer peer copies remain intentionally unsupported; live multi-GPU
validation remains open.
resources retain their existing thread-affine mixed-owner design. Async pooled views are retained by transfer/capture/profile tokens. The
unprofiled PTX API has no completion token, so it synchronizes before a pooled
view can return to its cache; this is safe but deliberately conservative.
