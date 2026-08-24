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
  datasets.rs            local dataset facade
  datasets/              IDX/CIFAR parsing and deterministic batching
  onnx.rs                bounded public ONNX import facade
  onnx/                  private protobuf wire, tensor, schema, lowering and tests
  ir.rs                  typed frontend graph while the UOp layer is built
  ir/                    operation-family extensions: creation/reduce/indexing/...
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
  nn/                    layers, graph-independent parameter state, and modules
    parameter.rs         stable host Parameter identity, versions, graph bindings
  optim.rs               host optimizers and learning-rate schedulers
  training_checkpoint.rs in-process module/optimizer/scheduler checkpoint boundary
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

`ir::indexing` is the pure static-indexing boundary: it normalizes immutable
integer/slice/newaxis/ellipsis and constant advanced-index specifications into
checked shapes and coordinate maps. The narrow `Op::StaticIndex` and its
reverse scatter consume the plan without re-parsing it; dynamic
boolean/nonzero cardinality and mutable aliasing remain outside it.

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
| ONNX model interchange | `onnx.rs`, `onnx/{wire,tensor,schema,lower}.rs` | bounded parse, normalize, and CPU-graph lowering |
| `viz/*` | `viz/*` | compiler introspection |

The current `backend::CpuBackend` is deliberately the semantic oracle. It will
move behind the runtime/device contracts once those contracts are executable;
optimized CPU and GPU paths must match it through differential tests.

`onnx.rs` is a bounded fail-closed default-domain opset-13 facade. Private wire
parsing, typed/raw tensor decoding, schema normalization, and graph lowering
keep untrusted bytes separate from the CPU-graph boundary. The checked surface
is static inference only: elementwise/activations; movement/indexing/shape;
Gemm/MatMul and softmax; Conv/pooling/BatchNorm/GlobalAveragePool;
predicate/Where/Clip/inference Dropout; ConstantOfShape; and reductions/args.
Dynamic controls or shapes, general Gather/indexing, control flow, sequences,
sparse/quantized/external data, custom domains/opsets, training, and live
external-model differential validation remain outside this boundary.

Static direct and nested shrink views lower as `ViewMap`/`ViewBufferIndex`
through scheduling, interpretation, and PTX. Computed-value shrink and other
movement families stay explicit lowering boundaries. `CpuJitBackend` is an
internal cached native-execution boundary; a future schedule-DAG hook will own
its broader compiler integration.

Late `LinearKernel` construction validates a typed portable contiguous
elementwise lane plan before C rendering. Its immutable `LinearProgram` records
producer-first instructions, virtual definitions/uses, lane/tail metadata, live
intervals, and deterministic scalar/vector register assignment. A backend-neutral
`MemorySpacePlan` consumes those assignments, validates global/register/private/
shared identities, byte/alignment/lifetime aliases, and uniform workgroup
barriers. `VectorProgram` is the backend-neutral physical-register instruction
view (splat/address/index/load/cast/ALU/compare/select/store/control) with
explicit lane mask and scalar-tail identity; CPU JIT validates and keys this
form before portable rendering. Current elementwise kernels explicitly choose no shared promotion.
B1/B2 CPU JIT consumes eligible VectorProgram instructions directly in physical-register order.
Alongside F32/F64/bool constants, loads, neg/abs, add/sub/mul, compare/select, casts, and stores,
B2 has defined unsigned-intermediate wrapping for stored integer widths, guarded integer division,
modulo, and shifts with the ABI failure index, and raw F16/BF16-to-F32 register conversion with
raw-bit stores. Unsupported transcendental/logical families, reductions, and non-contiguous views
remain structured scalar fallbacks. Portable C lane loops retain explicit main/tail bounds rather
than target SIMD, workgroup memory, or tensor-core instructions.

Each scheduled kernel retains immutable `ScheduleInputBinding` entries ordered
by first lowered `Load` use (with repeated reads canonicalized), never by graph
node or buffer ID. The set-like input inventory remains for dependency planning;
bindings carry the input node, descriptor, and contiguous pointer-ABI index and
validate uniqueness, view consistency, output exclusion, and completeness. CPU
interpreter/JIT and PTX can validate the same map without changing their ordered
pointer-slice ABI.

`schedule_with_external_materializations(graph, outputs, materialized)` is the
explicit opt-in for a caller-owned computed buffer such as a redistribution
destination. It validates reachability and rejects outputs, inputs, constants,
duplicates, and unnamed operations; the named producer is replaced by exactly
one ordered Load binding and recorded in the item cache metadata. It does not
execute itself; the sharded CUDA planner consumes these typed bindings for
direct transfer-to-local fusion.

Sharded two-owner shrink→binary composition retains PTX and its static
`ViewBufferIndex` ABI binds the original global source lease; the owner-scoped
mock executes that view once and matches the CPU oracle.

Scheduling records a deterministic producer-aware DAG and lazy-realization
trace, selecting interpreter, native JIT, or an explicit fallback. HostDense
temporary slots reuse only exact-compatible non-aliasing buffers; backend-owned
slots and vectorization remain outside this planner. Sharded CUDA mock execution
has graph-derived local Add evidence across one, two, and four owners for F32,
I32, and U64, including canonical zero-byte routes. Typed graph-derived
redistribution now validates layouts, ranks, owners, node-buffer identities,
element/byte ranges, and dtype before deterministic same-owner DtoD and
cross-owner peer execution. The mock covers two/four-owner axis-to-replicated,
replicated-to-axis, and axis-to-axis routes plus injected DtoD/peer failure and
retry. Logical zero buffers and checked transfer-to-local composition preserve
typed descriptor substitutions and a dependency DAG. Static-layout local Neg
and cast compose through Graph/CPU/autograd, while static-view cast and
broadcasted boolean select have owner-scoped mock-CUDA byte-oracle evidence
across one, two, and four owners. Exact `GraphUnary` Neg now has the same
one/two/four-owner static-view mock-CUDA evidence for I32 and F32, including
logical zero domains; unsupported unary/dtype pairs remain typed diagnostics.
Computed-shrink broadcast, allocator-stat assertions, collectives, and live
CUDA remain explicit boundaries. Typed local provenance now drives direct
redistribution-to-local CUDA fusion: only named transfer destinations become
external schedule materializations, then exact ordered ABI bindings validate
and substitute the canonical transfer buffers before local launch. CPU-byte
mock evidence covers one-, two-, and four-owner axis-to-replica Add,
two-owner axis0-to-axis1 Add, and a zero-domain logical-buffer path with no
allocation, copy, or launch.

`HostSlotPool` leases are generation-checked and views/detached outputs retain
their runtime ownership. Exact-compatible `MemoryPlan` reuse is alias-safe; the
remaining boundary is backend-owned slot placement and vector/lane byte-window
planning. `PrimaryPoolStats` snapshots one exact allocator handle: its `pool_id`
distinguishes independently constructed pools on one primary context, while
clones share accounting; sharded execution still needs to query its retained
allocator handles for accounting assertions. Optimizer checkpoints use a config
fingerprint with legacy rejection and strict atomic expected-key loading;
LARS/LAMB reference updates include corrected LAMB bias correction and
independent resume evidence, while host Muon implements its checked
Newton--Schulz update surface.

`datasets.rs` is intentionally a small local, deterministic facade. Private
`datasets/idx.rs`, `datasets/cifar.rs`, and `datasets/batch.rs` own uncompressed
MNIST IDX decoding, exact CIFAR-10 binary records, and seeded batch ordering.
CIFAR records retain their channel-major bytes as U8 NCHW tensors; pure F32
normalization accepts explicit per-channel means and positive standard
deviations. Parser unit tests own format and malformed-input contracts, while
public training/composition workloads live under `tests/`. The boundary does
not download, cache, randomly augment, or claim corpus parity. `nn::Parameter`
is graph-independent versioned host state, while each `Graph` owns its binding
leaves. `training_checkpoint.rs` depends one way on `nn`, `optim`, and
`safetensors`; its exact in-process resume retains the same host parameter
identities but permits fresh graphs, optimizers, and schedulers. Cross-process
identity rehydration remains outside this boundary.

## Bounded Torch state import boundary

`torch::load_torch_state_dict` is a read-only, fail-closed interchange boundary,
not a Python compatibility layer. It accepts a single-root, stored or raw-deflate
ZIP archive containing protocol-2 `data.pkl` and CPU dense storage members. Its
small pickle VM recognizes only string dictionaries/`OrderedDict`, persistent
CPU storages, and Torch's tensor-rebuild symbols; it never invokes a Python
class, imports a module, or extracts an archive entry to the filesystem. ZIP
paths, duplicate names, symlink attributes, count/byte limits, bounds, shape,
stride, storage offset, and overlapping views are validated before construction.
The importer materializes a fresh contiguous `TensorData`, preserving exact
little-endian raw element bits. Deflate output is bounded by declared size and
ratio and CRC-checked. `extract_tar_files` separately provides a regular-file-
only, checksum-validated in-memory ustar boundary; legacy Torch's `storages` /
`tensors` / `pickle` streams use a separate record-oriented safe protocol-2 VM
which retains exact raw CPU storage bytes and only unwraps inert `Parameter`
`BUILD` state, not general pickle execution. CUDA, sparse/quantized tensors, custom classes,
and unsupported pickle opcodes are explicitly rejected. ZIP64 single-disk
central-directory and per-entry metadata is accepted only with one exact extra
field and checked u64-to-usize conversion; multi-disk and ambiguous ZIP64
metadata fail closed. The returned
`BTreeMap<String, TensorData>` converts
directly to `nn::StateDict`; callers retain the module loader's existing
validate-then-versioned-replace lifecycle.

## Collective planning boundary

## Static tensor sharding boundary (Phase 1)

`sharding.rs` adds an immutable, backend-neutral `ShardLayout` over the stable
semantic `collective::DeviceId` and ordered `DeviceGroup`.  Its forms are
replicated or one axis sharded, with normalized axes, exact global ranges,
shape/dtype, and a deterministic cache key.  `ShardedTensorData` is a typed
host reference container: it validates one exact dense `TensorData` per device
in caller order and can shard, gather, replicate, and redistribute without
numeric conversion, including raw narrow-float, NaN, and signed-zero bits.

The checked-in tinygrad source is the policy evidence: `Tensor.shard` delegates
to `UOp._shard`, and `tinygrad/uop/ops.py` rejects `shape[axis] % device_count
!= 0`; scalar `_shard` is a no-op.  Therefore Phase 1 rejects non-empty uneven
axis partitions rather than inventing quotient/remainder tensor shards (the
existing collective buffer planner independently supports those chunks).

Movement and operator lowering return inspectable static decisions only:
provably ownership-preserving reshape/permute/expand/shrink/stride stays local;
otherwise a typed redistribution requirement is returned.  Elementwise layout
matches are local, mismatched binaries request peer redistribution, reductions
over a sharded axis and matmul contracting shards request sum all-reduce.

## Graph sharding composition (Phase 2)

`sharded_graph.rs` connects static layouts to `Graph` without putting device
state on ordinary nodes. `ShardedGraphTensor` is bound to one graph identity,
holds ordered local `NodeId`s and layout metadata, and records inspectable
layout/collective transitions. Local binary trace steps retain their ordered
per-rank operand `NodeId`s and identify operands produced by a typed
redistribution destination; ordered schedule-buffer attachment now supplies
direct planner fusion without labels or graph reinspection. Dense nodes lower through checked `Shrink`
views; gather is `Concat` (or replica identity); redistribution is explicit
graph composition. CPU execution and reverse mode therefore use the existing
dense graph oracle, not eager host calculations.

Local elementwise/select/movement, sharded-axis sum/mean with graph-visible
partial sums and replicated sum-all-reduce, and rank-two local/contracting
matmul are supported. Mean divides only after the global sum. CUDA scheduling,
lazy multi-device realization, and runtime collective execution remain Phase 3.

## Sharded CUDA plan (Phase 3A)

`sharded_cuda_plan.rs` is a deterministic, serializable planning boundary over
`ShardedGraphTensor`. It validates ordered semantic-device to primary-context
and capability bindings without entering a context or issuing Driver work. It
uses the existing `schedule` and PTX renderer to produce owner-scoped local
stage cache identities, buffer descriptors, dependencies, and explicit
unsupported diagnostics. Graph trace all-reduce and redistribution transitions
become explicit collective/transfer stages; no collective is inferred from a
node label. Phase 3B will consume these stages to create streams, allocations,
modules, and launches. The currently renderable subset is elementwise/select/
cast; reductions and matmul remain diagnosed rather than executed.

Phase 3A.1 splits that portable logical record from `ExecutableShardedCudaPlan`:
the latter is intentionally non-serializable and validates the exact graph-node
schedule key before retaining rendered PTX ABI artifacts and primary owners.
It still performs no Driver operation. Graph composition now records typed
redistribution routes, so its transfer companion validates source/destination
layouts, node-buffer identities, ordered owners, dtype, and checked element/
byte ranges before constructing exact external/output buffer records.

Phase 3A.2 emits those routes at graph composition time. Redistributions become
deterministic contiguous local-storage runs with source/destination semantic
devices, graph buffer identities, element offsets, exact bytes and dtype. CUDA
plan transfer stages consume them verbatim.

Generic PTX semantic-mock S1 registers immutable renderer ABI/extent metadata
against a stable primary owner/function identity. Native dispatch ignores the
hook; the owner-scoped test mock retains it for inspection only. Evaluating a
generic kernel over mock bytes remains the explicit S2 boundary.

S1.1 attaches the exact immutable ranged `UOp` used by generic PTX rendering
to that rendered artifact. Manual and collective PTX artifacts explicitly carry
no generic semantics. S2 now uses the existing independent UOp interpreter with
checked `TensorData` snapshots of the owner-scoped mock allocations, then commits
the output bytes atomically. This is test-mock simulation only: native dispatch
and sharded CUDA execution still submit retained PTX and never materialize host
values. The generic path includes direct PTX `neg`/`abs` for i32/i64/f32/f64,
including wrapping signed-min integers and floating signed-zero behavior, over
scalar, broadcast, and static-view bindings. It deliberately does not claim a
libdevice contract for reciprocal, roots, exponentials, logarithms, or
trigonometry; broader acceptance remains pending.

Phase 3B1 now has a first executor-level proof: a retained broadcast-add PTX
artifact runs through `ShardedCudaExecutionEnvironment`, which validates the
external primary leases, allocates its output from the owner pool, loads through
the owner cache, and exposes the mock device bytes and deterministic local trace.
It now also executes a graph-composed two-owner axis-sharded shrink→binary
workload: each rank binds the original global input leases, retains static view
source shapes in PTX ABI metadata, and produces local bytes which gather exactly
to the CPU `ShardedGraphTensor` oracle. A second execution reuses the
owner-scoped semantic registrations.

Executor preflight now requires the external binding set to match the canonical
map exactly. On a local-stage failure it restores caller-owned external leases,
drops only executor-created outputs, and permits deterministic retry; the mock
fixture covers injected launch failure and proves extra bindings make no Driver
calls.

The owner-scoped mock executes typed graph-derived redistribution across two
and four owners. It covers axis-to-replicated, replicated-to-axis, and
axis-to-axis layouts; validates exact route/layout/buffer identities and byte
ranges before allocation; performs same-owner DtoD before directional peer
copies in trace order; and restores external source leases after injected DtoD
or peer failures so the plan can be retried. Zero-element bindings are logical
metadata with no device pointer. A checked composition substitutes a
transfer-produced output for an exactly matching local external input, carries
the transfer producer into the local dependency DAG, rejects duplicates and
descriptor/dependency/cycle violations with structured errors, and retains the
same retry boundary. Direct graph-derived fusion now uses the same composition
artifact after validating explicit provenance against ordered external-materialized
ABI bindings; retained-allocator stat assertions, CUDA collectives, and live-CUDA
validation remain pending.

`collective.rs` is a backend-neutral Phase 1 boundary for the multi-device
reduction pattern checked into tinygrad. tinygrad's `schedule/multi.py` lowers a
reduction across a sharded axis to `ALLREDUCE`, while
`schedule/allreduce.py` selects naive, ring, or all-to-all schedules; its ring
path partitions a flat buffer into quotient/remainder chunks and performs an
ordered add. RustGrad exposes that schedule boundary explicitly as immutable,
serde-serializable `CollectivePlan` actions rather than treating a raw CUDA
handle as a device identity.

`DeviceId` is a semantic string and `DeviceGroup` retains caller order after
rejecting duplicates. The planner's chunks for `count = q*n + r` give device
`i` `[i*q + min(i,r), (i+1)*q + min(i+1,r))`; therefore empty and uneven tails
are represented, never discarded. Plans contain local copies, directed
transfers, and ordered reductions with dependency ids and lanes. The dense
in-memory executor follows those actions and re-materializes after each add so
narrow storage has CPU-oracle behavior.

Phase 2 is only an implementation of `CollectiveExecutor` for validated plan
actions using the existing CUDA primary peer-transfer/stream ownership layer.
It may choose a ring transport but must not alter plan ordering, stable IDs,
or dtype semantics. NCCL, cross-process rendezvous, discovery, and live
multi-GPU support are deliberately outside this boundary.

## CUDA PTX cache ownership and concurrency

`PtxCache` remains deliberately local to an owned, thread-affine `Context`.
`ConcurrentPtxCache` is the primary-context counterpart: its key is
`(primary-owner identity, rendered PTX key, block size)`, never a raw CUDA
handle. The map mutex is held only while creating/removing a per-key entry;
the entry mutex/condition variable coordinates waiters, and module loading and
function lookup occur with neither cache lock held. A failed load wakes current
waiters with the same structured `PtxError`, removes its entry, and permits a
later retry. Only primary-owned cached kernels are `Send + Sync`; resource sum
types that can contain an owned context are intentionally not marked sendable.

## CUDA asynchronous staging

Async Driver copies use the optional `_v2` memcpy symbols plus `cuMemHostAlloc`
and `cuMemFreeHost`; unavailable symbols return `MissingSymbol` and do not
silently fall back to synchronous copies. `PinnedHostBuffer` is owner-scoped
page-locked memory with checked ranges. Async HtoD, DtoH, and DtoD calls require
the exact same sealed owner and a stream, validate every range before calling
the driver, and return a non-cloneable `Transfer` token. The token borrows all
involved resources and owns a completion event; `query`/`wait` are explicit.
Dropping an unfinished token performs a best-effort event wait, so callers that
need error visibility must call `wait`; no live-CUDA validation is claimed.

## CUDA graph foundation

`Stream::begin_capture` is a primary-context-only, non-cloneable capture
session. Callers explicitly retain buffers and pinned allocations used by the
captured work; the resulting graph-exec borrows them for its full lifetime and
validates the replay stream owner. Capture abandonment ends and destroys any
returned graph best-effort. This is a static capture foundation only: parameter
updates, capture invalidation diagnostics, and live-driver validation remain
open.

## CUDA profiling foundation

`cuda_profile` is a crate-private, Driver-free recorder core. Enabled sessions
are stable-primary-owner scoped and assign monotonic submission sequence numbers;
disabled sessions allocate no trace state. Pending samples own abstract `Arc`
retention sentinels and transition deterministically through ready, collected,
failed, or abandoned. Its isolated CUDA adapter owns default-flag (timing
enabled) start/end events bound to one primary stream; it records without
synchronizing submission, supports nonblocking query and explicit wait/collect,
and validates elapsed durations. Driver/event calls occur outside recorder locks.
Primary PTX launches and supported primary async HtoD, DtoH, and DtoD copies
now use explicit profiled submission surfaces. Transfer timing composes the
existing completion token with the timing pair so the latter is authoritative;
live-CUDA validation remains pending.

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
a single ranged UOp sink. Static sum/mean/product/min/max reductions fuse a pure producer and
expose accumulator initialization/update/finalization UOps; the portable
interpreter traverses separate output and reduction domains. A deterministic
temporary-plan utility only reuses caller-designated internal buffers with
non-overlapping lifetimes and compatible size/alignment. Vectorization and
device rendering retain their own capability boundaries.

`engine::capture` retains an immutable schedule, ordered input ABI, constants,
and requested buffer identities for backend-neutral interpreter replay. It does
not retain a Graph, rebuild scheduling, provide runtime-polymorphic shapes, or
participate in CUDA graph capture.

`rangeify/` owns pure movement-to-index metadata. Its first consumer extracts
direct and nested static shrink chains into a deterministic `ViewMap` source
plan before kernel lowering. Computed producers, pad validity, symbolic runtime
extents, and broader reshape/permute/expand/stride composition remain explicit
boundaries rather than hidden host materializations.

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
meaning, and each transform appends nodes. Parameters retain graph-independent
versioned host state. A graph-local registry captures each parameter identity
and version into one immutable input leaf; optimizer writes reject stale or
wrong-identity gradients, and subsequent forwards bind the new host version.
In-process `TrainingCheckpoint` resume retains those host objects and validates
their exact identity/version/value stamps before restoring fresh optimizer and
scheduler state, so versions never roll back into a graph-cache collision.

`training_checkpoint/portable.rs` owns the distinct cross-process artifact.
`PortableTrainingCheckpoint` identifies state by deterministic module paths,
typed descriptors, and explicit tied-parameter equivalence classes rather than
`ParameterId`. Its versioned, bounded container combines a checked manifest
with canonical safetensors sections for module, optimizer, and scheduler state;
section lengths and checksums reject truncation and corruption before restore.
Restore validates the complete module schema, optimizer parameter-group/path
ownership, and scheduler/config state into candidates before taking a stable
parameter lock order and committing all module values and versions together.
Targets must be freshly constructed at version zero and restored before graph
binding, preventing process-local graph-cache identities from being reused.
The artifact serializes no Graph, executable code, device state, or backend
resources.

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

Its stable entry point is `int rustgrad_kernel(void **buffers, const int64_t
*symbols, uint64_t *failure)`. A nonzero result reports a guarded per-element
division/modulo or shift error together with its first linear index. Buffer order, dtype, element count, byte length, output mutability,
alignment, ABI version, and symbol count are checked by Rust before the call.
`JitKernel` owns the dynamic library (and therefore outlives its function
pointer); calls borrow all buffers for their whole duration. The sole unsafe
boundary is `dlopen`/`dlsym` plus that C ABI call. No writable executable memory
is allocated by Rust.

Source and libraries are content-addressed below the OS temporary directory.
The key includes renderer/ABI version, host target, fixed compiler flags, and
the rendered UOp source. A process-local mutex and atomic rename prevent
duplicate publication; compiler diagnostics are bounded.

## File and null runtime boundary

`runtime::file::FileBuffer` is an owned, checked file-I/O byte resource. It is
explicitly copying I/O rather than mmap or zero-copy: every window validates
logical bounds before seeking, writes require read-write access, and typed
`TensorData` adapters use the existing portable little-endian representation.
`runtime::null::NullRuntime` validates logical allocation/copy requests and
records deterministic planning traces, but intentionally has no values or
semantic execution. These concrete modules establish runtime-resource evidence
without introducing a speculative common backend trait.

## CUDA Driver runtime boundary

`cuda.rs` is a deliberately toolkit-free, dynamically loaded CUDA Driver API
foundation. Loading `Driver` never creates a CUDA context, and fails with a
distinguishable missing-library, missing-symbol, API-version, or Driver-error
result. The tiny native loader is the sole function-pointer conversion boundary;
all operational calls instead go through a typed `Dispatch` trait, which keeps
the default test suite deterministic and CUDA-free.

Retained primary contexts additionally carry a stable Rust owner identity.
`Dispatch` has default no-op registration and per-thread enter/exit/current
observation hooks so deterministic mocks can distinguish owners even when the
Driver returns colliding raw context or device handles. This is a validation
seam only, not a CUDA identity mechanism: it never supplies, validates, or
changes CUDA currentness. Observation follows a successful push and successful
pop; a failed pop leaves the observed owner in place because real currentness
is unknown, while RAII cleanup remains best effort.

The test-only mock dispatch now models primary-owner-scoped device allocation
bytes exactly, including colliding raw pointers across distinct owners. Mock
copies resolve storage through that stable owner metadata (and record peer-copy
pairs explicitly), while asynchronous mock copies mutate bytes at submission
time for deterministic testing; events still retain their normal readiness
contract. This is a mock-memory foundation only: PTX semantic execution and
CUDA collective execution remain pending.

The next mock-only semantic foundation is a typed primary-context local-add
PTX kernel with an inspectable five-word pointer-array ABI. Its mock launch
handler executes the exact supported scalar addition at submission, while the
native Driver receives ordinary PTX launch calls. This is deliberately only a
local add building block: collective scheduling/execution, live CUDA proof,
and bool/half/narrow-16-bit dtype support remain pending.

`Device` creates an owned, thread-affine `Context`, matching tinygrad's current
explicit-context policy. `ContextGuard` snapshots the thread's preceding Driver
context and restores it in `Drop`, including panic unwinding. Buffers, streams,
events, modules and functions privately retain their context. They reject
closed resources, checked out-of-bounds copies and cross-device use before a
Driver call; their destructors make a best-effort current-context cleanup.
This is the foundation for the next PTX renderer/loader milestone, not yet a
CUDA backend or a claim of hardware execution parity.

## Phase-one PTX boundary

`ptx.rs` renders a deterministic, content-keyed PTX 7.0 source form for the
existing scalar fused elementwise UOp sink. Its ABI orders pointer parameters by
UOp buffer id, followed by a specialized `u64` extent. The generated entry
uses a global linear CUDA thread index with an extent guard; zero domains skip
the launch on the host and scalar domains still launch exactly one thread.

The accepted type subset is bool, i32/u32/i64/u64, f32 and f64, with typed
loads/stores, casts, comparison/select and ordinary add/sub/mul/min/max (plus
floating division). Unary `neg` and `abs` are accepted only for i32/i64/f32/f64
and lower to PTX scalar instructions; bool/unsigned and all other unary pairs
are structured renderer errors. Guarded integer division, modulo and shifts are
rejected until a device-status ABI exists. f16/bf16 are likewise intentionally
rejected until their capability-specific conversion and requantization path is
proven. Transcendental unary functions remain rejected because this renderer
does not yet carry a versioned libdevice contract.

Static sum, mean, product, min, and max UOp programs have a separate correctness-first PTX path.
One CUDA thread owns one logical output and serially walks the normalized
row-major reduction domain, including multi-axis and keepdim layouts; fused
eligible producers reuse the ordinary emitter with that computed input index.

Static matmul rendering is deliberately separate in `ptx_matmul.rs`;
`PtxRenderer::render_matmul_plan` is only the public adapter in `ptx.rs`.
Its immutable `MatmulKernelPlan` fixes the ordered lhs/rhs/output ABI and the
dot, vector, matrix, and broadcast-batch coordinate map.  The current path is
one output thread with a serial K loop for homogeneous F32 or F64 storage only.
It retains `KernelSemanticProgram::Matmul` for owner-scoped mock execution;
other dtypes are rejected before driver work.  This is a correctness boundary,
not a claim of tiling, shared memory, tensor cores, or live-CUDA coverage.
F32/F64 retain CPU-equivalent floating accumulation/finalization; I32/I64 and
U32/U64 sums use defined wrapping PTX arithmetic; bool sum is the I32 count of
true inputs; and bool/wide-integer mean promotes through F64 before the F32
store, matching the CPU scalar contract. Empty sum domains store zero and empty
float mean domains store a canonical quiet NaN without emitting a divide by
zero. I8/I16 and U8/U16 loads explicitly sign/zero extend into their promoted
I32/U32 sum accumulators or F32 mean finalization. F16 (on sm_53+) and BF16
reduction buffers are decoded from raw 16-bit storage,
accumulated through F64, and deterministically requantized at the final store;
the BF16 store uses the same raw ties-to-even bit arithmetic as `TensorData`.
The same serial path also carries Product/Min/Max for every static stored
scalar dtype. Product uses typed wrapping ALU (Bool is AND), its multiplicative
identity, and the existing raw narrow-float finalization. Extremum selection
projects each candidate through the CPU oracle's `f64` comparison contract but
retains the selected raw storage word: NaNs are ignored and equal values retain
their first row-major occurrence, including signed-zero and high-bit integer
ties. F16 requires sm_53 conversion support; BF16 uses the explicit raw decode
and ties-to-even requantization path. Empty extrema remain graph validation
errors, while empty Product stores its typed identity. This is not a
shared-memory or symbolic reduction claim; optimized reductions remain an
explicit boundary.
`PtxCache` owns modules and functions by content key within its thread-affine
context, and `PtxKernel::launch` owns all parameter words until the synchronous
Driver call returns while validating buffer ABI, bytes, device and geometry.
The default mock tests validate this wiring; live CUDA smoke validation remains
optional future work.

Module loading uses bounded JIT option/log storage (`ModuleLoadOptions`) and
captures Driver compile diagnostics as a distinct `CudaError::JitCompile`.
The dispatch trait exposes the exact LoadDataEx option/value-array shape, while
the compatibility default intentionally falls back to LoadData only for
dispatches that do not implement Ex; native Ex symbol hardening and live smoke
remain follow-up work.

Successful module metadata exposes the negotiated Driver load path and bounded
info/error logs. The PTX cache retains that module object, so cache hits expose
the same immutable metadata without changing launch semantics. The deterministic
mock verifies the full six-option LoadDataEx layout, distinct writable log
buffers, length cells and both success/failure log capture.

## CUDA allocation cache, phase 3B0 representation foundation

Pooled allocations are represented by `BufferLease` (owned contexts) or
`PrimaryBufferLease` (primary contexts).  They expose only a borrow-tied
`BufferView`, whose length is the requested logical length, not the physical
size-class capacity.  The view performs checked copy/range/address operations;
returning a lease is therefore impossible while a safe view exists, and a later
attempt to obtain one reports `CudaError::StaleLease`.

Classes are deterministic powers of two with a 256-byte minimum. Allocation
uses ordered best-fit reuse, bounding internal class waste below one class.
Zero allocations are rejected and overflow is checked. Owned-context state
remains `Rc`/thread-affine. Primary-cache state is an `Arc` with a mutex and
contains `Arc<PrimaryBlock>` physical allocations, never the thread-affine
`Owner`/`DeviceBuffer` sum. A block retains only primary-owner state, its stable
owner/device identity, physical capacity, generation and explicit close state.
`PrimaryBufferLease` is a logical checked view of that block; checkout advances
the generation and views reject a stale generation. The internal checked view
descriptor is shared with direct and owned buffers without exposing a raw owner
or pointer. Driver allocation/free calls occur outside that mutex.
Cached blocks are detached on trim/close and freed afterwards; only a CUDA OOM
causes this pressure trim and one retry. Accounting distinguishes requested
in-use bytes, cached physical bytes, reserved physical bytes, and peak in-use
bytes. Phase 3 currently closes the borrow-based asynchronous paths: async
copies are submitted through `BufferView` and return a transfer borrowing that
view (whose drop waits), captured graphs retain `BufferView` rather than a raw
buffer, and non-profiled PTX launches synchronize before a pooled view can be
released. Profiled launches retain their views through their timing sample.
Owned-context pooled launches remain conservative and synchronize because their
mixed owner is thread-affine. Primary launches use allocator-owned event
deferral without weakening the reuse invariant. Phase 3B
now records one shared primary completion fence after each successful pooled
primary PTX submission. A returned lease with pending fences moves to deferred
state; `collect_deferred` queries outside the pool lock and atomically promotes
matching completed generations, while `wait_deferred` blocks outside locks.
Trim/OOM paths exclude deferred blocks. `PrimaryEventFence` is a shareable,
primary-only query/wait resource whose cleanup retains the primary context until
the event is destroyed. It is registered in primary allocator deferred state;
owned and mixed resources remain deliberately thread-affine. Live CUDA
validation remains a hardware-dependent caveat.

## CUDA primary peer transfers

Directional `PeerAccess` sessions retain both primary contexts and resolve the
Driver peer capability/copy symbols optionally at runtime. Primary pooled
leases can submit checked asynchronous peer copies and return a borrow-tied
`PeerTransfer`; its completion fence is attached to both allocator generations,
so neither block is reusable while the copy is in flight. Live multi-GPU
validation, direct/owned buffers, and profiled peer copies remain later boundaries.

## CUDA collective Phase 2B2

`CudaCollectiveGroup` is a sequential, deterministic one-through-four-primary-
owner sum all-reduce executor. It walks the existing immutable plan actions,
uses immutable per-rank snapshots and pooled per-destination staging leases,
creates only the directed peer sessions required by transfers, waits each
completion token, and invokes the typed primary collective-add kernel for
reductions. This is mock-driver validation only: other collectives, overlap,
narrow dtypes, NCCL, processes, and live CUDA remain outside the implemented
boundary.
