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
- **Candle:** small deployable Rust model workflows and practical safetensors/tokenizer integration, without adopting an eager-first runtime;
- **tract:** a strict translation-versus-runtime boundary for model import and a compact, fail-closed deployment surface;
- **ndarray:** ownership-aware dense values and views, with explicit copying and interoperability boundaries;
- **tch-rs:** a PyTorch interoperability and workflow baseline, treated as a comparison rather than a compiler/runtime dependency because it binds LibTorch;
- **RustTensor:** direct CPU/CUDA differential tests and an easy-to-follow reference path;
- **cuda-oxide:** a future optional Rust-to-PTX kernel path, isolated because its toolchain is currently experimental.

## Source layout

The layout below is the target responsibility map, not an assertion that every
listed module already exists. Consult `docs/COMPATIBILITY.md` and the working
tree for the currently executable surface.

```text
src/
  tensor/                dense tensor value subsystem
    mod.rs               public TensorData, Shape, Storage and dtype facade
    dtype.rs             dtype taxonomy and promotion policy; float8 transport/cast boundary
    scalar.rs            scalar and exact F16/BF16 conversion semantics
    storage.rs           owned typed dense storage
    shape.rs             checked shape arithmetic
    data.rs              TensorData construction, casts and dense access
    creation.rs          dense creation helpers
  session/               public CPU-first Graph/binding ownership facade
    mod.rs               narrow public session and tensor-handle exports
    cpu.rs               explicit CPU realization, handle validation, bindings,
                         and thin static model/movement Graph delegates
    train.rs             fresh-graph static-module train/evaluate bridge
    inference.rs         fresh-graph static-module inference and native opt-in
  datasets/              local facade, IDX/CIFAR parsing, and deterministic batching
  gguf/                  bounded GGUF reader, metadata and tensor descriptors
  tokenizer/             GGUF SimpleTokenizer metadata binding and byte-level coding
  models/
    transformer/         validated dense-Llama GGUF model and graph execution
      decoder.rs         typed graph planning and CPU semantic-oracle execution
      cache.rs           transactional single-sequence KV cache ownership
      layer.rs           authoritative dense block Graph composition
      model.rs           GGUF config/state binding and N-layer graph/cache path
      generation.rs      greedy and explicit-uniform Gumbel-max generation
      batch.rs           padded batched Graph planning and transactional KV caches
      batch_generation.rs deterministic per-row stopping and sampling
      chat.rs            checked Llama fallback/chat-template formatting
      native.rs          staged CapturedSchedule/native CPU JIT execution
      native_generation.rs transactional single/fixed-batch native generation
      serving/           continuous native batches and immutable prefix-cache reuse
  onnx/                  bounded facade; private wire, tensor, schema, lowering, native, file, tests
  ir/                    typed frontend graph facade, vocabulary, shape planning,
                         storage/lifecycle, and operation-family extensions
    mod.rs               concise module wiring and public re-exports
    types.rs             public IR vocabulary and operation options
    convolution.rs       validated rank-generic windows/spec and compositional lowering
    dynamic.rs           typed dynamic-result nodes with rank/dtype contracts
    graph.rs             graph storage, bindings, lifecycle, and composition
    shape.rs             pure checked shape/dtype validation helpers
    elementwise.rs        elementwise graph construction and validation
  autograd.rs            reverse-mode graph transform
  uop/                   tinygrad-style universal operation IR
    mod.rs               typed operations, values, validation and rewrites
    artifact.rs          bounded typed UOp DAG node-table codec
  schedule/              realization, fusion, indexing and memory planning
    artifact.rs          portable schedule descriptors and bindings
    execution_summary.rs immutable static schedule and logical-memory summary
  matmul/                normalized and tiled matmul compiler contracts
    mod.rs               generalized serial semantic plan
    tile.rs              target caps, candidates, cost and tiled simulator
    tensor_core.rs       capability-gated MMA plan and fragment simulator
  memory_space/          register/global/shared promotion planning
    mod.rs               allocation, alias and barrier validation
    promotion.rs         tiled matmul shared-memory derivation
  movement_plan.rs       typed materializing concat/gather/scatter kernel contract
  renderer/              C/LLVM/PTX/WGSL and platform renderers
  engine/                lazy realization, JIT capture and replay
    capture.rs           immutable concrete schedule capture
    captured_replay.rs   Graph-free backend policy, cache and batching
    symbolic.rs          captured symbolic schema and specialization
    symbolic_view.rs     affine symbolic view expressions and bounds
  device.rs              discovery, capabilities and allocator contracts
  runtime/               CPU/CUDA/Metal/HIP/OpenCL/WebGPU/... implementations
    metal/               SDK-free Objective-C Metal resources and static MSL rendering
    opencl/              dynamically loaded ICD, resources, and OpenCL C rendering
    webgpu/              typed WebGPU ownership, WGSL lowering, and semantic mock
  nn/                    module facade and graph-composed neural-network layers
    parameter.rs         stable host Parameter identity, versions, graph bindings
    state.rs             deterministic module traversal and state loading
    init.rs              deterministic parameter initialization
    linear.rs            fully connected layers
    embedding.rs         embedding lookup
    conv.rs              convolution and transpose-convolution layers
    pool.rs              pooling adapters
    norm.rs              normalization layers and BatchNorm state commits
    recurrent.rs         recurrent cells
    regularization.rs    training-time regularization
    sequential.rs        heterogeneous traversal composition
  optim.rs               host optimizers and learning-rate schedulers
  training_checkpoint/  in-process and portable checkpoint boundaries
    portable.rs         fresh-identity module/optimizer/scheduler checkpoint
  llm/                   the bundled language-model path
  fuzz/                  deterministic semantic generation, replay and failure artifacts
  viz/                   graph, schedule and kernel inspection
  compatibility_manifest/ deterministic compatibility-ledger projection
  bin/compatibility_manifest.rs manifest generation and drift-check command
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

Static core-parity additions stay within existing operation families:
`ir::rearrange` lowers checked `split`/`chunk` only to `Shrink`; `ir::reduce`
composes variance/std from existing mean, square, cast, and sum nodes;
`ir::creation` reuses typed constants and captured Threefry; and the
`StaticIndexGrad` reverse edge reuses the normalized static-index map. Einsum
normalizes presentation whitespace before its existing parser. The same static
CPU boundary holds explicit boolean `Any`/`All` empty identities, left-biased
unordered/tied float extrema, stable finite-tail softplus/mish/logsigmoid
graphs, and raw-payload `TensorData::bitcast` for equal-width canonical
little-endian dtypes. `tril`/`triu` build checked final-two-axis boolean masks
through existing `arange`/comparison/select nodes, and causal attention reuses
`tril` rather than owning another mask path. Rearrange rejects an empty side
before Graph mutation. UOp scalar literals must exactly match their node type
and storage width through construction and artifact decoding; pure Add folds
only a type-matched canonical positive raw zero, never a negative-zero literal.
Public storage-less `LiteralScalar` values resolve immediately into ordinary
concrete scalar `TensorData` before Graph lowering: Bool/I64/U64/F64 choose the
strong peer dtype (or their documented concrete default) with the existing
checked integer conversion semantics. They therefore never add a weak `DType`,
storage, UOp, artifact, or cache-identity variant.
None adds a runtime, IR, backend, dynamic-shape, or device path.

`reduction_native.rs` is the single checked reduction recurrence boundary for
CPU, capture, and native renderers. It derives exact source/accumulator/output
dtypes from one scalar Init→Accumulate→Finalize chain, commits every step at
accumulator width, owns typed identities and Mean finalization, and rejects
empty extrema. The finalize result may have a different storage dtype from
the accumulator. Newly emitted raw Float8 Sum/Mean use F32 recurrence followed
by one final narrow encoding when a real reduction axis remains; a singleton or
no-effective-axis reduction is an exact raw-lane identity. Explicit
same-storage reductions commit each step in that storage. The decoder also
admits only the released v18 all-Float8 Mean tuple, preserving its historical
per-step commitment without broadening new Graph lowering. Float8 therefore
follows the same checked recurrence instead of a CPU-only policy module.
`backend/float8_contract.rs` separately owns the
source-audited MatMul, Conv2d, and contraction-form Einsum policies: F32
contraction accumulation followed by one result narrowing. Diagonal-free
single-input Einsum reorders remain raw-lane movement operations.
The private historical RGUA reduction codec, opcode tags, and durable v18
schedule identity are unchanged. Corrected compiled source is separated by
the bumped renderer/cache versions rather than invalidating released schedule
artifacts.

`ReductionValue::mean` was removed from the public in-memory UOp payload:
`kind: ReduceKind::Mean` is now its only source of truth. This is an explicit
pre-1.0 Rust source migration for struct-literal callers. `Op::Reduce` literals
now also state their accumulator dtype independently from the node's final
storage dtype. The historical wire bit remains private, is checked against
`kind` while decoding, and does not reintroduce the invalid in-memory
cross-product.

Float8 C4 transport is deliberately separate from numeric execution:
`TensorData::reorder_raw` and `replace_raw_offsets` preserve tagged storage
lanes for movement and replacement plans, while the CPU capability table admits
only static byte-transport operations. Same-format select/concat, static
indexing/update, gather, replacement scatter, and fixed-size masked selection
retain raw bytes. Accumulating scatter-add and every compiler/device path remain
outside this boundary.

The compatibility ledger has one machine-readable projection at
`docs/compatibility.json`. The `compatibility_manifest` binary parses only
Markdown tables with an explicit `Status` column, accepts the four documented
status markers, and emits a deterministic versioned JSON document. Its test
compares the checked-in bytes with the ledger on every `cargo test`, so changing
a capability claim requires regenerating the projection with
`cargo run --bin compatibility_manifest -- --write`. This keeps the Markdown
ledger authoritative while giving CI and external tooling a stable input.

`.github/workflows/ci.yml` is the repository release-gate entry point. The
package and workflow pin Rust 1.89 so formatting, all-target checks, and strict
Clippy do not change underneath a release; compatibility-manifest drift is also
checked on Linux. The default suite runs independently on Linux and Apple
Silicon macOS, and a date-pinned nightly Linux job instruments library tests
with AddressSanitizer and LeakSanitizer. The one CUDA regression that
deliberately leaks quarantined mock blocks when asynchronous completion cannot
be proven remains in the normal Linux/macOS suites but is explicitly skipped by
the sanitizer job; unexpected leaks everywhere else still fail it.
The first checked-in remote run passed this complete Linux, Apple Silicon
macOS, quality, and sanitizer matrix.
Hardware-only tests remain explicitly ignored in the default suite. This is a
portable baseline rather than a complete device, architecture, Miri, coverage,
or cross-compilation matrix.

`viz` is the pure inspection boundary. Typed normalizers consume graph,
schedule/capture, UOp, linear, memory-space, and vector metadata into a small
validated model. Model construction sorts node IDs, fields, and edges before a
dependency-free DOT renderer escapes labels, so construction order cannot leak
into snapshots. Graph-local node IDs and portable buffer, item, artifact, and
cache identities remain explicit; pointer identities, compiled modules, runtime
handles, profiling samples, and `Debug` text are not inputs. Every current
typed Graph Op family is normalized; future unsupported operation families fail
with a typed visualization error instead of being silently flattened.

`ir::indexing` is the pure static-indexing boundary: it normalizes immutable
integer/slice/newaxis/ellipsis and constant advanced-index specifications into
checked shapes and coordinate maps. The narrow `Op::StaticIndex`, functional
`Op::StaticIndexUpdate`, and their first-order CPU VJPs consume the same plan
without re-parsing it; the update VJP uses a final-writer map, so duplicate
coordinates preserve replacement rather than scatter-add semantics. Dynamic
boolean/nonzero cardinality and mutable aliasing remain outside it.
`Graph::diagonal` is a checked static convenience lowering that permutes its
two selected axes last and delegates rectangular, batched, signed-axis, signed-
offset, zero-domain, and Bool cases to that same `StaticIndex` substrate; it is
not a dynamic indexing or aliasing path.
`Graph::diag` is distinct rank-one construction: checked `n + 1` and `n²`
arithmetic lowers only through existing unsqueeze, typed-zero pad, flatten,
shrink, and reshape nodes, so it preserves storage and inherits their static
movement reverse edges without another indexing substrate.

`ir::dynamic` owns a separate typed dynamic-cardinality arena. Dynamic inputs
are either graph-owned dynamic values or validated scalar static nodes; they
never acquire a sentinel static `Shape`. CPU realization memoizes this arena
within one request. `DynamicOutputShape` retains the exact count-producing node
and its scalar, one-dimensional, or row-coordinate expression. Pointwise
`Neg`/`Square` and `Add`/`Sub`/`Mul` compose arbitrary topological branches
when dynamic operands share that count provenance; scalar reductions remain
ordinary composable values in the runtime DAG. Their first-order CPU/session
VJP evaluates Mean's divisor from the immediate input's realized element count
through the same committed work-width quotient as static native reductions;
an empty input produces an empty local cotangent without division. General
dynamic broadcasting, graph-on-graph higher order, and every device boundary
remain explicit follow-up work.

The 0.1 runtime-cardinality API intentionally replaced the former rank-only
shape constructor and side binding projection. `DynamicOutputShape` is now the
direct `Scalar`/`Count1d`/`CountRows` expression enum, `DynamicCountStage` owns
its bindings, and allocation-plan validation accepts the canonical ordered
binding slice. This keeps unresolved shapes and stage/binding drift
unrepresentable rather than preserving a parallel legacy projection.

`EffectGraph::static_index_assign` is the explicit pure-plan-to-effect bridge.
It embeds the normalized `StaticIndexPlan` in the typed STORE/AFTER payload;
both detached execution and `EffectRuntime` stage an immutable target snapshot
through the same raw-storage update helper before a pool-wide commit. Effects
remain graph-free: they cannot carry `NodeId` uses or gradient state, and their
public `grad` boundary rejects before mutation. `EffectMutationPermit` is the
explicit, host/interpreter-only safety bridge: a pure graph computes an
immutable backward-slice analysis for one requested loss/target pair, then the
permit binds that result to one exact pre-write `BufferState`. Guarded whole,
affine, and static-index STORE construction rejects a backward-required old
value before a step exists; detached, non-differentiable-only, and unrelated
uses receive distinct safe classifications. This is not a VJP or alias
registry: higher-order and device mutation autograd still need an owned use
registry and tape.
RGSM v3 serializes the normalized static-index offset map in its typed effect
payload and replays it graph-free through the same raw-storage transaction;
v1--v2 envelopes remain decodable, authenticate their stored opaque keys, and
upgrade to the canonical current schedule identity. Native and device
indexed-effect replay remain deliberately unsupported.

`effects::EffectSourceBridge` is an immutable host-interpreter sidecar: it
binds exact persistent snapshots to pure Graph inputs and one pure output to an
existing frozen STORE source, then delegates the only mutation to
`EffectRuntime::execute_with_sources`. It carries typed provenance but embeds
no NodeIds in effects and is not a VJP, capture, native, or device path.

`ir::dynamic` keeps data-dependent extents separate from static graph nodes.
`DynamicAllocationPlan` is the graph-owned exact count/allocation contract for
CPU `nonzero` and `masked_select_dynamic`: the count-stage enum owns its typed
bindings and has no side projection, sentinel capacity, or placeholder.
`RuntimeSchedule` is one validated topological instruction DAG. Count,
allocation-resource, materialization, unary, binary, and reduction variants
own their operands and results; `RuntimeValueDesc` distinguishes dynamic
buffers from fixed scalar results without colliding raw IDs. Allocation does
not produce a tensor value, and validation requires exactly one allocation and
one later value producer per runtime buffer, exact descriptor/count identity,
and canonical dependency edges. `nonzero` materializes row-major
`[count, rank]` coordinates in I32 unless a source dimension's maximum
coordinate exceeds I32, then I64, including scalar and zero-domain shapes.
The fixed-size form validates maximum coordinates through the final I64 value
before publication and plans coordinate ranges by length; zero-domain inputs
therefore use a descriptor-only scalar expansion and need neither dense fill
storage nor a narrowed or wrapped exclusive range endpoint.
Shared-count dynamic branches,
checked static scalars, and fixed `Sum`/`Mean` results compose through the same
DAG. Both reductions carry the canonical `ReductionDType`: narrow floats and
small integers accumulate at the source-policy width, then cast once to the
public output dtype. CPU realization validates a concrete ranked
`engine::RuntimeShape` before
exposing owned storage.

Runtime allocation lifetimes are checked sizing/liveness metadata. The CPU
executor currently owns independent `TensorData` values and neither releases
nor reuses storage from those records. Capture, artifacts, replay, native JIT,
PTX, and device lowering reject rather than falling back; the fixed
`ScheduleItem` and RGSA/RGSM artifact ABIs are unchanged.

`DynamicVjpPlan` is the read-only graph-owned reverse descriptor: it binds one
dynamic output and its exact count-bearing shape/dtype to one fixed static
target descriptor, validates the supported dynamic route, rehearses every
participating static reverse boundary on a private graph clone, and
excludes Bool mask/cardinality inputs before CPU work. The shared CPU executor
requires an upstream with the exact realized output shape, then scatters its
row-major lanes through broadcast masks into the static source shape. Once
that exact count is known, `DynamicCompactionVjpRule` clone-rehearses the
inverse as ordinary lazy arange, fixed masked selection, additive scatter, and
reshape nodes. The generated fixed graph retains higher-order edges through
the compacted cotangent while the Bool mask/index route remains non-value
data. Dynamic
`Sum` uses that same plan for its implicit one seed. Dynamic `Mean` derives its
denominator from the immediate input's realized element count, divides at the
canonical F32/F64 work width, narrows to the input storage, and broadcasts the
local cotangent over that exact shape; an empty input produces no lanes. Local
cotangents cross admitted derived static graph boundaries through an explicit cloned-graph
`grad_with` seed, including the masked value expression and derived scalar
operands. The public result remains an owned first-order host value: dynamic
handles still do not participate directly in `Graph::grad`, capture, or a
higher-order dynamic-output arena.

`CpuSession` exposes this same exact ABI without mutable or raw arena-index
access: `DynamicTensor::shape_expression` carries an opaque graph-local count
provenance token, and every operation validates session ownership.
`nonzero_dynamic` and `masked_select_dynamic` create session-owned handles;
pointwise branches,
checked static scalars, fixed scalar `Sum`/`Mean`, their exact first-order VJP,
and CPU realization accept them. `dynamic_vjp` takes exact realized upstream storage
and returns the fixed target-shaped first-order cotangent. Handles are
session-identified like static session tensors. This is a public workflow
facade over the existing plans, not a capture surface or dynamic device
generalization.

## tinygrad-to-RustGrad mapping

| tinygrad | RustGrad | Responsibility |
| --- | --- | --- |
| `tensor.py`, `mixin/*` | `tensor/*`, `ir/*` | public tensor semantics grouped by operation family |
| `dtype.py` | `dtype` types exposed by `tensor` | scalar/vector/image/pointer dtype rules |
| `uop/*` | `uop/*` | universal IR, symbolic values, validation and rewrites |
| `schedule/*` | `schedule/*` | rangeification, indexing, memory and multi-device scheduling |
| `renderer/*` | `renderer/*` | target source and ISA generation |
| `engine/*` | `engine/*` | realization, JIT, graph batching and workers |
| `device.py`, `runtime/*` | `device.rs`, `runtime/*` | allocation, transfers, launches and synchronization |
| `nn/*`, `llm/*` | `nn/*`, `llm/*` | ecosystem and representative workloads |
| ONNX model interchange | `onnx/mod.rs`, `onnx/{wire,tensor,schema,lower,native,file}.rs` | bounded parse, normalize, CPU-graph lowering, and narrow strict-native replay |
| `viz/*` | `viz/*` | compiler introspection |

The current `backend::CpuBackend` is deliberately the semantic oracle. It will
move behind the runtime/device contracts once those contracts are executable;
optimized CPU and GPU paths must match it through differential tests.

`onnx/mod.rs` is a bounded fail-closed default-domain opset-13 facade. Its
`file` sibling owns bounded local reads, deterministic named NPY orchestration,
and output-path validation while delegating byte parsing/lowering and NPY codec
semantics to their existing owners. Private wire
parsing, typed/raw tensor decoding, schema normalization, and graph lowering
keep untrusted bytes separate from the CPU-graph boundary. The checked surface
is static inference only: elementwise/activations; movement/indexing/shape;
Gemm/MatMul and softmax; Conv/pooling/BatchNorm/GlobalAveragePool;
predicate/Where/Clip/inference Dropout; ConstantOfShape; and reductions/args.
The `native` sibling owns a separate strict-native deployment adapter for fixed
concrete F32 named input and selected output sets: it reuses the model-owned immutable
Graph, canonical schedule/capture, and caller-owned `CapturedReplayExecutor`.
The verified `MatMul → Add → ReLU` fixtures return detached ordered outputs plus
a handle-free logical trace; the `file` path stages same-directory NPY output
replacements and rolls back earlier targets after a later replacement failure.
That is fail-atomic rollback, not simultaneous filesystem multi-path atomicity.
`onnx_npy_infer --native` is the thin command boundary over that same path: it
parses only explicit `NAME=PATH.npy` maps and owns one executor per invocation.
It is not a second ONNX runtime/cache/IR; dynamic/empty schemas, general ONNX
operation coverage, and fallback remain explicit rejections.
Dynamic controls or shapes, general Gather/indexing, control flow, sequences,
sparse/quantized/external data, custom domains/opsets, training, and live
external-model differential validation remain outside this boundary.

Source-backed affine shrink, contiguous reshape, permutation, expansion, and
signed-stride chains lower as canonical `AffineView`/`ViewBufferIndex` through
scheduling, interpretation, native CPU execution, and PTX rendering. Computed
value and non-affine chains stay explicit lowering boundaries; OpenCL, Metal,
and WebGPU consume validated signed affine maps with target-native signed
address arithmetic. `CpuJitBackend` is an
internal cached native-execution boundary with validated `ScheduleItem`
preparation and invocation; replay never reconstructs a Graph.

Late `LinearKernel` construction validates a typed portable contiguous
elementwise lane plan before C rendering. Its immutable `LinearProgram` records
producer-first generic `LaneInstruction<R>` values, exact virtual
definitions/uses, program-wide lane/tail metadata, live intervals, and
deterministic scalar/vector register assignment. Fixed-arity variants own typed
value/address/index operands; stores have no result, and unsupported UOps plus
range/sink delimiters remain exact indexed source records rather than fake lane
instructions. A backend-neutral
`MemorySpacePlan` consumes those assignments, validates global/register/private/
shared identities, byte/alignment/lifetime aliases, and uniform workgroup
barriers. Eligible homogeneous F32 matrix matmul additionally derives two
shared tile promotions and its accumulator/register/barrier lifetimes from the
selected `TiledMatmulPlan`; elementwise kernels still choose no shared
promotion. `VectorProgram` maps the same `LaneInstruction` enum onto validated
physical registers. It owns one lane width plus scalar-tail identity instead of
cloning masks or semantic payloads into each instruction; CPU JIT validates and
keys this form before portable rendering.

The static OpenCL, Metal, and WebGPU renderers project each validated pure UOp
node through that same `LaneInstruction<R>` semantic boundary. One shared,
exhaustive scalar-lane emitter consumes the enum's structurally fixed arity,
owns `GraphBinary` operand promotion, and keeps heterogeneous `GraphCompare`
fail-closed until its exact mixed signed/unsigned and float/integer ordering has
a portable representation. Bitcast, memory, and control remain fail-closed;
sealed backend dialects own only target syntax, storage commitment, capability
checks, and signed-wrap spelling. Guarded integer renderers reuse the same
projection for pure subexpressions while retaining backend-local lazy branch
and first-fault status ordering. This avoids a second renderer operation taxonomy without
changing artifact or storage ABIs.

Dense homogeneous F32 matmul is a separate operation-scoped projection rather
than three renderer-local semantic trees. `PortableF32Matmul` authenticates a
Serial or selected Tiled payload, proves the deduplicated dense pointer ABI and
right-aligned broadcast address terms, and exposes only immutable geometry for
OpenCL C, MSL, and WGSL syntax emission. Each output lane performs a serial K
walk with an F32 product followed by an F32 running-sum update. A nonempty K=0
launch retains its logical zero-byte operands while the shared prepared-static
allocator supplies private four-byte native handles; the kernel performs no
read and writes positive zero. Whole-output zero domains still submit no
launch. TensorCore, Quantized, and affine-view forms plus unspecialized symbolic
capture/device replay fail closed.
Renderer-specific source/cache versions distinguish the new compiled kernels;
RGUA, schedule, capture, and artifact encodings are unchanged. PTX retains its
existing independent serial/tiled/tensor-core lowering and correction boundary.
B1/B2 CPU JIT consumes eligible VectorProgram instructions directly in physical-register order.
Enabled vector mains must be lane-aligned, permit at most one partial tail, and
derive one deterministic program tail mask; disabled plans retain zero vector
main elements and no executable vector instructions. Malformed lane control,
reaching-definition metadata, or physical register lifetime rejects before native source
generation or cache work.
Alongside F32/F64/bool constants, loads, neg/abs, add/sub/mul, F32/F64 `log2`, compare/select, casts, and stores,
B2 has defined unsigned-intermediate wrapping for stored integer widths, exact Bool logical-not and
signed-integer negation through modulo subtraction plus bit reinterpretation, guarded integer division,
modulo, and shifts with the ABI failure index, and raw F16/BF16-to-F32 register conversion with
raw-bit stores. Unsupported transcendental/logical families, reductions, and non-contiguous views
remain structured scalar fallbacks. Portable C lane loops retain explicit main/tail bounds rather
than target SIMD, workgroup memory, or tensor-core instructions.
F32/F64 `exp2` follows the same strict native-renderer path and cache identity;
F16/BF16 and Float8 `exp2` remain outside the native contract.

`Graph::cumsum` and `Graph::cumprod` share a typed static `PrefixScan` schedule
materialization with an explicit normalized axis, Sum/Product kind, source and
destination identities, and exact source/result dtypes in UOp/RGUA v18. One
checked `NativePrefixScanPlan` derives the row/axis/inner domain and the work
dtype for the CPU oracle and every native renderer instead of duplicating a
scan operation taxonomy. Scalar and zero-extent shapes remain exact. F32 Sum
commits each recurrence in F32; F16/BF16/Float8 Sum works in F32 and casts each prefix
result to source storage, while Product/extrema commit source-width arithmetic.
Integer Sum retains its public promotion contract and Product retains source
dtype, including Bool. Floating `cumsum` reverse mode composes existing
signed-axis reverse views around another sum scan, and floating `cumprod` uses
the existing zero-aware scan composition. Floating cumulative-extrema values
move the normalized axis last and build one prefix/equality winner matrix from
ordinary compare, logical, cast, reduction, multiply, and divide nodes. Each
prefix cotangent is divided among every equal winner, including signed-zero
ties; NaN follows the same equality/count route. Both the winner-count and
final contribution reductions retain the upstream storage dtype. These paths
retain graph-on-graph seed edges, while non-floating and Float8 scans reject
before derivative graph mutation.
`Graph::cummax` and `Graph::cummin` use the same typed path to return values plus
I32 first-matching-prefix indices. Index state begins at the axis-length
sentinel, moves on a strict winner, otherwise records only the first source lane
equal to the current cumulative value, and therefore preserves zero's first
sign. Extrema values begin at the dtype identity and retain that identity for an
unordered lane, so a leading NaN leaves the index sentinel until a later strict
winner; a source lane equal to the retained identity still records its index.
The identity is committed through the source dtype before recurrence, including
Float8 infinity-to-NaN formats. Rank-zero scans bypass recurrence and preserve
the source value's raw bits when storage is unchanged; widened Sum casts the
single source value once, while the index result is I32 zero.
Their index outputs remain explicitly
nondifferentiable. Scalar CPU-JIT covers all concrete storage dtypes and both
value/index results through captured replay. PTX v31 and operation-specific
OpenCL C, MSL, and WGSL renderer identities use the same checked portable
Bool/I32/U32/F32 projection and two-buffer ABI, assigning one work item to each
independent row/inner lane and scanning the selected axis serially. WGSL Bool
outputs use packed atomic byte-lane writes. Static prepared prefixes validate
the logical output extent separately from this launch domain, keep producer and
consumer intermediates device-resident, and create no buffers, queues, or
launches for a zero work domain. Other scan dtypes, dynamic domains, parallel
algorithms, and live-device numeric validation remain fail-closed or unclaimed. Existing RGUA
v18 schedule/capture identities are unchanged; only renderer source/cache keys
distinguish the new accelerator programs. Legacy PrefixScan RGUA v11--v17
payloads also fail closed because they cannot prove source dtype or destination
identity. The fixed-size
`MaskedSelect` reverse edge alone
reuses its boolean prefix ranks as nondifferentiable control/index values to
gather explicit upstream cotangents into retained row-major source lanes;
padding, truncation, and false lanes are zeroed. This does not add a dynamic
cardinality gradient path.

Each scheduled kernel retains immutable `ScheduleInputBinding` entries ordered
by first lowered `Load` use (with repeated reads canonicalized), never by graph
node or buffer ID. The set-like input inventory remains for dependency planning;
bindings carry the input node, descriptor, and contiguous pointer-ABI index and
validate uniqueness, view consistency, output exclusion, and completeness. CPU
interpreter/JIT and PTX can validate the same map without changing their ordered
pointer-slice ABI.

A single-use, unrequested Reduce followed by a pure same-shape scalar epilogue
may remain one scheduled item. One checked `NativeReductionKernel` view locates
the exact ReduceFinalize inside the existing Store UOp, while renderers bind its
storage-committed value into their existing scalar emitters. The omitted
reduction intermediate therefore has no buffer, dependency, or temporary
identity. Requested/shared reductions, another reduction, movement or broadcast
epilogues, external materializations, and faulting scalar operations retain the
ordinary materialization boundary. RGUA tags and the durable encoding version
are unchanged; the fused graph intentionally has one schedule item and
therefore a different item/cache identity. Only the affected renderer/source
cache versions advance.

Prepared OpenCL, Metal, and WebGPU pure prefixes, plus the fixed-schema CUDA
graph path, then project those validated bindings into one crate-private static
residency plan. Rendered writable pointers carry ordered output ordinals; the
plan proves that they bijectively match every `ScheduledOutputs` descriptor
before deriving producers, dependencies, lifetimes, or resources. The plan
renders and validates the complete prefix plus its canonical physical buffer
inventory
before queue, cache, compilation, or allocation work. Execution uploads each
external logical buffer once, retains producer outputs on device across ordered
consumers (whose affine views remain renderer-local addressing metadata), and
downloads only the exact caller-retained outputs; mixed execution supplies its
value-binding outputs while compatibility prepared-prefix APIs retain every
item output. Host outputs are decoded completely before publication, so
launch/read failures expose no partial result and a retry starts from fresh
external uploads. Backend resource types, zero-domain cache policy, renderer
source/cache keys, and RGSA/RGSM bytes remain outside and unchanged by this
derived runtime plan.

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
trace, selecting interpreter, native JIT, or an explicit fallback. Before
memory planning, backend selection, or capture, `Schedule::validate` requires
contiguous position-matching item IDs, strictly prior ordered dependencies,
consumer lists that exactly mirror the derived reverse edges, and valid shared
buffer descriptors for every input/output. The same descriptor seam validates
direct temporary planning and artifact decode before allocation, cache, or
backend work. It is artifact integrity validation, not a new scheduler, compiler backend, or device
contract. HostDense temporary slots reuse only exact-compatible non-aliasing
buffers. Validated pure static prefixes additionally project the authenticated
rendered buffer graph onto runtime-private device slots: only nonexternal,
unretained, single-producer temporaries with strictly disjoint inclusive
lifetimes and exactly equal dtype, source shape, byte length, alignment, and
backend domain may share a slot. External inputs and retained outputs stay
private because uploads precede all launches and downloads follow them; zero-byte
native sentinels also stay private. OpenCL, Metal, and WebGPU prepared prefixes
own each physical slot once behind a logical-ID map, while CUDA graph capture
retains one stable lease per physical slot and fences or quarantines it once.
This projection changes no logical binding order, BufferDesc, ScheduleItem,
artifact, or cache identity; capacity-class reuse, suballocation, aliases,
effects, dynamic schedules, and cross-backend slots remain outside it. Sharded CUDA mock execution
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
planning. Ordinary CPU batch realization derives one checked ordered output
projection before allocation or compilation, then executes the canonical
`schedule_many` DAG once. Scheduled outputs retain producer-owned buffers;
requested Inputs and Constants retain exact owned source storage; duplicate
requests are projected back into their original positions without duplicate
execution. The entire output vector is published only after every item and
host-slot invariant succeeds. This changes neither the recursive single-output
CPU oracle nor capture/artifact identity. The backend-neutral `effects`
subsystem separately owns immutable
logical buffer versions and explicit read/write-after dependencies; it is the
only planned bridge from pure graph dataflow to STORE/AFTER scheduling. Dense
`TensorData::assign_from` is its CPU reference predecessor, with same-dtype
broadcast and source-snapshot semantics. `EffectGraph` is graph-adjacent so
effect handles cannot be mistaken for pure nodes; it validates and stages
whole-buffer assignment commits before exposing a new state map. `EffectRuntime`
owns one generation-checked `HostSlotPool` lease per logical buffer and stages
every successor from immutable detached snapshots; it validates every target
then atomically commits the final per-buffer values, so an injected or borrow
failure leaves bytes, versions, slot identities, and pool liveness unchanged.
Each public `EffectScheduleNode` owns one validated assignment payload and its
predecessor IDs. It synthesizes the matching STORE source and AFTER root on
demand, so the schedule cannot retain mismatched kinds, payload copies, or UOp
graphs. Those private-wire STORE/AFTER UOps lower to ordinary effect-boundary
`ScheduleItem`s with stable dependencies; `realize_effects_persistent` uses
that same canonical schedule rather than a parallel runtime IR.
`schedule::mixed` adds a typed,
immutable pure-output-to-STORE binding and transactional realization: pure
values are owned until the pool-wide effect commit. Typed
`ScheduleStateBinding` injects one immutable, version-checked persistent
snapshot (or checked signed `AffineView` read) into its exact Graph input before
interpreter realization. `engine::mixed_capture::CapturedMixedSchedule` is a
separate RGSM envelope, not an extension of ordinary RGSA: it serializes typed
STORE/AFTER UOps through the canonical UOp table plus logical state/version and
value/state ABI sidecars, named pure inputs, detached constants, and affine
maps. Decode completes topology/descriptor validation before graph-free
interpreter replay injects caller-owned snapshots and performs the same single
pool-wide `EffectRuntime` commit. Strict native CPU replay preflights the whole
artifact, runs its supported pure prefix through the existing native JIT cache,
and commits only detached outputs through that same transaction. Its stable
trace identity binds RGSM contents, ABI sidecars, pure cache keys, renderer
target, and vector policy—never leases, slots, generations, pointers, or
current bytes. `MixedReplayCursor` adds an in-memory interpreter-only recurrent
frontier for one exact RGSM identity. It contains only the canonical logical
buffer/version descriptors required by persistent reads and writes. Each step
uses those versions to snapshot detached candidates, preserves the requested
pure outputs, stages all effect successors through the existing mixed-batch
seam, and advances the cursor only after the one `EffectRuntime` commit
succeeds. Wrong-capture, incomplete, stale-runtime, descriptor, input-shadowing,
execution, and injected failures leave both runtime and cursor unchanged. The
cursor is not serialized, adds no RGSM/RGMB wire fields, and has no native or
device replay claim. Unsupported native pure items remain fail-closed; the
separately bounded mixed-batch adapters below are the only device-prefix path.
Read-only runtime statistics expose
lease/view/sentinel liveness but never backing capacity or pointers.
Capture/artifact and autograd entry points reject effects explicitly. Affine
aliases, HostSlotPool alias-version liveness integration, device effects, effect
replay, and mutation autograd are not yet lowered through this contract. Effect
targets carry the canonical immutable signed `AffineView`; writable targets are
checked injective regions, while staging preserves untouched base raw lanes and
commits the full base candidate atomically. `AliasLivenessPlan`
derives base/view/predecessor/successor lifetimes before mixed realization, so
an affine alias never receives a temporary reuse identity while its persistent
base lease remains live. `uop::AffineView` is the canonical signed late-IR
descriptor for ordinary views and effect targets, converting losslessly from
unsigned `ViewMap`; checked flips use signed strides with immutable source
snapshots. UOp artifact v10 encodes signed maps and upgrades v2–v9 unsigned
maps deterministically. CPU interpreter/JIT and replay execute signed maps;
PTX, OpenCL, and Metal emit checked signed 64-bit affine read arithmetic for
those maps; WebGPU emits checked signed i32 arithmetic and rejects maps that
cannot be represented without intermediate overflow. State-to-pure
reads remain an explicit boundary.
`engine::mixed_batch` owns a backend-neutral prepared-prefix coordinator:
adapters bind every capture against detached candidates, prepare every retained
prefix before any submission, execute deterministically into detached typed
values, then use one host `EffectRuntime` commit. `replay_opencl` and
`replay_metal` keep their thread-confined resource/cache ownership and typed
errors at their adapters; neither has a CPU/native fallback or makes persistent
effect state device-resident.
`replay_webgpu` uses the same coordinator with retained WGSL semantic plans:
it validates and allocates every supported prefix before submission, produces
detached values, then performs that same one host commit. Its SDK-free native
probe remains fail-closed for the unpinned callback/future ABI.
`replay_ptx` is the analogous primary-context CUDA adapter. It retains the
owner-scoped concurrent PTX cache, module/function semantics, stream, and
transient primary leases only for one replay attempt; it validates every batch
prefix and allocates all of those leases before the first launch. Each retained
semantic-mock launch writes a detached `TensorData` candidate, after which the
same single host `EffectRuntime` transaction publishes effects. CUDA persistent
state is never device-resident here, and there is no CPU/native fallback or
serialized CUDA resource. The default evidence is mock-only: no live toolkit
or device mixed-batch replay evidence is claimed.
`effects::EffectBatch` is the runtime-owned ordered transaction seam for
several independently constructed local `EffectPlan`s: it rebases explicit
persistent start states, stages private intermediate versions, and publishes
only final candidates in one `HostSlotPool` commit. `CapturedMixedBatch` is an
ordered coordinator: interpreter, strict-native, and prepared-backend paths
stage every RGSM capture against detached rebased candidates and the runtime
commits once after all pure prefixes succeed. A caller may supply
`MixedStateRebinding` to substitute a
complete bijective logical persistent namespace before any snapshot, pure
execution, prepared backend work, or commit; rebinding is replay-local and
does not alter RGSM/RGMB bytes, identity, descriptors, versions, or state
bytes. Strict-native in-memory batches bind every capture, compile
every pure prefix before executing any, then retain detached results for that
same one commit; their logical trace binds batch identity, vector policy,
input schema, and planned cache keys only. `RGMB` is the portable logical batch
envelope: it carries only bounded, checksummed ordered RGSM byte entries and
their recomputed batch identity, then decodes every entry through the canonical
RGSM validator. `RGBS` is a separate bounded/checksummed host bootstrap: it
embeds those canonical RGMB bytes unchanged plus only each referenced logical
buffer's exact version-zero raw `TensorData` frontier. A caller supplies a
validated rebinding to atomically register that frontier into a fresh host
runtime before interpreter, strict-native, or prepared-prefix replay; it never
serializes destination IDs, slots, generations, pointers, caches, device
handles, or later mutable state. Device-resident state, incompatible rebinding,
compiler-failure injection, and mutation autograd remain fail-closed.

Metal pure prefixes retain deterministic rendered cache keys and reuse the
device-scoped pipeline cache across equivalent logical batches. Preparation or
launch failures leave the host transaction uncommitted and may retry through
the same retained semantic path. This remains retained-semantic-mock evidence;
no ignored live test currently exercises `replay_metal`.
RGSM validator. Device execution, runtime rebinding, compiler-failure
injection, and mutation autograd remain fail-closed.
`PrimaryPoolStats` snapshots one exact allocator handle: its `pool_id`
distinguishes independently constructed pools on one primary context, while
clones share accounting; sharded execution still needs to query its retained
allocator handles for accounting assertions. Optimizer checkpoints use a config
fingerprint with legacy rejection and strict atomic expected-key loading;
LARS/LAMB reference updates include corrected LAMB bias correction and
independent resume evidence. Adam and LAMB checkpoint steps use checked `u64`
advancement and full-width bias-correction exponents, so an exhausted counter
rejects before any parameter or optimizer-state mutation. Host Muon implements its checked
Newton--Schulz update surface.
Every host learning-rate scheduler similarly checks its next `u64` epoch before
mutation; `LrSchedulerGroup` advances cloned scheduler/optimizer candidates and
commits them together only after every child succeeds. This is scheduler/LR
state atomicity only: it neither updates parameters nor adds a trainer, device,
or distributed optimizer path.

`datasets/mod.rs` is intentionally a small local, deterministic facade. Private
`datasets/idx/`, `datasets/cifar/`, and `datasets/batch.rs` own uncompressed
MNIST IDX decoding, exact CIFAR-10 records and bounded file loading, and seeded
batch ordering. CIFAR records retain their channel-major bytes as U8 NCHW
tensors; pure F32 normalization accepts explicit per-channel means and positive
standard deviations. The CIFAR file adapter preflights caller-ordered paths,
file count, total bytes, and record count before deterministic concatenation.
`datasets/batch.rs` also owns the prevalidated `ClassificationBatch` row
materializer: it copies canonical dense little-endian rows without scalar
conversion, validates all index/count/rank/dtype/overflow contracts before
allocation, and explicitly preserves or flattens trailing feature dimensions.
Parser unit tests own format and malformed-input contracts, while public
training/composition workloads live under `tests/`. The boundary does
not download, cache, randomly augment, or claim corpus parity. `nn::Parameter`
is graph-independent versioned host state, while each `Graph` owns its binding
leaves. `training_checkpoint/` depends one way on `nn`, `optim`, and
`safetensors`; its exact in-process resume retains the same host parameter
identities but permits fresh graphs, optimizers, and schedulers. Cross-process
identity rehydration remains outside this boundary.

`session/cpu.rs` projects the source-facing batch transform into the ordinary
CPU tensor workflow without adding another reverse engine. `CpuSession::gradient`
authenticates the loss, ordered targets, and optional seed before cloning the
complete session and invoking `Graph::gradient` exactly once. It publishes the
candidate only after every lazy output handle is constructed, so duplicates
retain one graph-node identity, failures publish no derivative nodes, and the
returned tensors remain available for higher-order graph composition.

The same module owns the detached persistent-gradient lifecycle.
`CpuGradientStore` is external to graph nodes but authenticated to one session
and keyed by stable `NodeId`; tensor-handle aliases therefore project one
logical slot. `CpuSession::backward` clones the complete session, invokes the
shared batch `Graph::gradient` transform once, adds each prior stored value to
its unique target once at the target's exact shape and dtype, and realizes all
staged results through one `realize_many` transaction. Only after every graph,
realization, and old/new descriptor check succeeds is the detached store
swapped into place; the candidate derivative graph is discarded and the live
session graph and bindings remain unchanged.
Disconnected floating targets retain `Graph::gradient`'s typed-zero contract,
while connected untracked/frozen targets receive their real derivative because
the source-facing transform deliberately ignores leaf tracking. Detach remains
an edge barrier, and `zero_grad` validates the complete store before atomically
removing every entry, matching tinygrad's `grad = None` reset. Targets are
always caller-supplied; no ambient live-tensor registry or automatic discovery
is introduced. This is deliberately not a
mutable field on `Graph` nodes or `Tensor`, and it adds no parameter/optimizer,
serialization, higher-order realized-gradient, capture, native, or device
gradient-store contract. Lazy higher-order composition remains on the shared
`Graph::gradient` transform, directly or through `CpuSession::gradient`.

`session/train.rs` is the bounded handoff from CPU-session tensor ergonomics to
versioned static module training. `CpuModuleTrainer` borrows an existing
`ModuleForward`, optimizer, and scheduler but owns no graph: each request builds
a fresh graph, so host parameter replacements cannot be observed by an old
graph. It validates the one-input F32/sparse-target/scalar-loss and canonical
module/optimizer-name contract, realizes output/loss/gradients before the
existing optimizer update, then advances a metric-free scheduler. Checkpoint
ownership remains solely with `PortableTrainingCheckpoint`; this bridge adds no
trainer, optimizer, state format, device fallback, or persistent gradient map.
`session/compiled_momentum.rs` is a separate bounded static-training seam.
`CpuCompiledMomentumSgd` consumes detached named F32 parameter values, builds
one private Graph with one batched reverse traversal, and captures the pure
loss/output/update prefix together with ordered momentum and parameter stores.
The Graph is discarded after compilation. `EffectRuntime` then solely owns the
parameter and momentum bytes, while `MixedReplayCursor` proves and advances the
exact recurrent state frontier only after the complete effect batch commits.
Every step accepts exact declared inputs plus one rank-zero F32 learning rate;
all user input order is canonicalized by name before graph-free interpreter
replay. Owned snapshots are diagnostic copies, not mutable aliases or live
`nn::Parameter` synchronization. This narrow CPU surface deliberately adds no
generic optimizer, module binding, checkpoint, native/device execution, mixed
precision, or dynamic-shape training ABI.
`session/classification.rs` is a pure post-evaluation helper for rank-two F32
logits and integer targets; it owns deterministic first-tie predictions and
optional empty-batch accuracy without retaining a graph or mutating training state.
`session/inference.rs` owns the corresponding graph-free single-input static
module route: it snapshots canonical trainable state, builds and discards one
fresh CPU graph, and returns detached output, deterministic trace, and
name-to-version metadata. It shares `ModuleForward` rather than introducing a
second module runtime; shape composition, strict state loading, and parameter
ownership remain in their owning subsystems.

`nn::Sequential` is the canonical heterogeneous composition for this same
single-input/single-output seam. It stores typed `ModuleForward` entries and
delegates graph composition to each entry, preserving deterministic numeric
state-path traversal without runtime type-name dispatch. Modules with distinct
multi-input, multi-output, or explicit-mode lifecycles remain outside this
container rather than being coerced into a hidden calling convention.
`nn::ModeSequential` is the separate explicit-mode companion: it stores
`ModeModuleForward` entries, admits ordinary stateless leaves through their
state-free forwarding implementation, and returns output plus the ordered
pending-effect collection. It deliberately does not make BatchNorm eligible
for ordinary `Sequential` or invent a global training flag.
`nn::LSTM` is a separate typed stateful composition rather than another
sequential trait adapter. It owns graph-independent `LSTMCell`s traversed as
`cells.{layer}.*`, accepts one static F32 `[time,batch,input]` sequence plus
optional separate `[layers,batch,hidden]` hidden/cell state, and returns the
last-layer sequence with its final typed state. Descriptor, state, layer, and
explicit-mode seeded-dropout work is rehearsed on a cloned graph before the
live graph is replaced. The composition is validated through the CPU oracle
and graph-independent captured interpreter; it adds no Op/UOp/artifact or
native/device ABI and deliberately does not implement the one-output
`ModuleForward` seam. Bidirectionality, projections, packed/dynamic sequences,
fused recurrent kernels, and a full RNNT model remain outside this boundary.
`nn/activation.rs` owns the state-free `ReLU` leaf, which delegates only to
`Graph::relu`; it contributes no traversal state and lets ordinary
`Linear → ReLU → Linear` static MLPs use the same Sequential/session path.
`ir/convolution.rs` owns one validated `SpatialWindow`, `ConvolutionSpec`, and
`TransposedConvolutionSpec`. Ordinary convolution lowers through movement,
promotion, multiplication, typed reduction, and bias operations. Transposed
convolution first reshapes/transposes/flips grouped weights, inserts source-typed
stride zeros, transforms signed asymmetric padding plus signed output padding,
and then calls that same ordinary core. `Graph::conv2d`, `conv_transpose1d`, and
`conv_transpose2d` are syntax adapters rather than additional semantic nodes.
No new public forward graph emits the old first-class Conv2d or ConvTranspose2d
operation, but their graph encode, CPU oracle, autograd, scheduler, and
visualization paths remain internal compatibility seams; StaticConv2d and RGUA
v10 remain decodable and replayable.
`nn/conv.rs` owns graph-free `Conv2d` construction plus its static one-input
forward adapter; `nn/pool.rs` owns the matching `AvgPool2d`, `AdaptiveAvgPool2d`,
and `MaxPool2d` adapters. `ConvTranspose2d` and `ConvTranspose1d` likewise own
graph-independent constructors and delegate their configured NCHW/NCL forwards
to the rank-generic compositional transpose-convolution contract; and
`nn/shape.rs` owns checked static `Flatten`. Together they cover the one
verified CIFAR classifier chain. `nn/norm.rs` additionally gives `LayerNorm`, `LayerNorm2d`,
`GroupNorm`, `InstanceNorm`, and `RMSNorm` graph-independent construction and their existing checked one-input
Graph forwards; `LayerNorm2d` preserves the exact NCHW-to-NHWC-to-LayerNorm-to-NCHW composition,
while `GroupNorm` and `InstanceNorm` preserve their existing static NCHW grouping composition. BatchNorm lifecycle and
other normalization adapters remain separate composition work.

Graph-independent parameter construction is owned by `nn` rather than the
session bridge. `Linear::new_static` constructs only versioned host state, and
the legacy graph-taking constructor delegates to it. `Module::trainable_parameters`
is the one canonical trainable traversal for optimizer setup: it snapshots
locks, filters non-trainable values, sorts names, and collapses tied identities.
`Optimizer::sgd_for_module` consumes that output without introducing a second
optimizer configuration or parameter naming convention.

`interop/host/` is the local dense-byte boundary. Its layout and view modules
validate signed host strides without pointer escape; its copy module remains
the sole bridge to independent `TensorData`. The NPY codec owns only portable
v1/v2 syntax and dtype policy, while the sibling file adapter owns bounded
filesystem reads and staged same-directory replacement writes. Thus file I/O
does not reimplement NPY parsing or create a second session/tensor abstraction:
the public workflow feeds its returned owned `TensorData` directly into
`CpuSession`, and static named weights continue through safetensors. Mmap,
device backing, Python/NumPy objects, and compute-time aliasing remain outside
this one-way host-to-owned-data boundary.

`nn/state.rs` owns deterministic module traversal and strict host-state
application. Its strict helpers consume an already ordered `StateDict` or
bounded owned safetensors bytes, validate the complete traversal schema before
calling the existing identity-sorted all-lock parameter restore transaction,
then leave graph construction to the module's ordinary `forward` method. This
keeps local file parsing, state validation, host mutation, and CPU execution
one-way and separate: it does not infer an architecture, remap keys, construct
a device module, or mutate an already captured graph.

`safetensors.rs` remains the sole canonical dense state codec. Its local-file
adapter adds `SafetensorsReadLimits` and typed file failures around a bounded
owned read before that parser. Saving constructs all bytes first, exclusively
stages and syncs a same-directory temporary file, then renames it into place;
failed staging or replacement cleans only its own temporary file and never
opens an existing target for writing. It does not introduce lazy mapping,
device ownership, key remapping, a multi-file transaction, or a second state
protocol.

## Bounded GGUF container boundary

`gguf/mod.rs` is the in-memory GGUF facade; private `reader`, `metadata`, and
`tensor` and `quantization` modules keep wire parsing, typed metadata, tensor-range validation,
and dense materialization separate. The parser accepts source-evidenced GGUF
versions 2 and 3, preserves metadata and tensor inventory order, bounds every
untrusted count/string/array/rank, and validates alignment, shapes, block
geometry, non-overlapping ranges, truncation, duplicates, and trailing bytes
before exposing payloads. The dense GGML F32/F16/I8/I16/I32/I64/F64/BF16
layouts materialize exact little-endian storage into `TensorData`.

Q4_0, Q8_0, Q4_K, and Q6_K additionally materialize source-evidenced
little-endian blocks to F32. They can also become an owned
`QuantizedTensorData`: exact bytes plus a no-`DType` descriptor containing the
GGML type, logical `[out_features, in_features]` shape, checked block geometry,
portable byte alignment, and stable content identity. GGUF source alignment is
validated before the owned copy and file offsets/pointers are not retained.
The block decoders are pure checked bit-layout
functions: Q4_K retains its packed six-bit scale/min fields and Q6_K retains
its low/high planes and signed subgroup scales. Whole-file F32 materialization
walks the validated tensor inventory in file order and returns a deterministic
name map only when every dense or supported quantized tensor converts. Other
quantized layouts remain opaque validated payloads. This is not model-key
interpretation, split-file merging, mmap/zero-copy, Graph construction, or LLM
execution.

## Checked-in SimpleTokenizer boundary

`tokenizer/mod.rs` owns the tokenizer used by tinygrad's checked-in GGUF LLM
CLI. It consumes typed GGUF token strings, token types, pre-tokenizer preset,
and BOS/EOS/EOT metadata only after complete type, length, and ID validation.
The pure coding boundary reproduces the checked-in GPT-2 byte alphabet,
bounded Unicode general-category splitter, early whole-word lookup,
rank-ordered greedy pair merging, ordered special-token recognition, UTF-8
replacement decoding, and incremental decoding. It accepts only the explicit
llama3/llama-v3/llama-bpe/qwen2/olmo/kimi-k2/tekken/glm4 preset family and its
two checked-in qwen aliases. This is not a generic SentencePiece/tokenizer.json
runtime or generic chat-template renderer.

`models/transformer/mod.rs` retains the explicit one-layer dense Llama state
schema and also exposes a supported multi-layer GGUF model boundary. Typed GGUF
metadata fixes the `llama` architecture, block/embedding/feed-forward/context,
head/GQA/key/value/rotary widths, RMS epsilon, rotary base, vocabulary, and
BOS/EOS/EOT IDs. The binder atomically validates every `blk.N` tensor against
fixed source-evidenced names and shapes without materializing the whole state.
Norms, biases, and optional RoPE auxiliaries remain exact F32; the embedding
table, rank-two q/k/v/output-attention and feed-forward projections, and an
explicit output projection retain dense F32 or exact Q4_0/Q8_0/Q4_K/Q6_K
bytes. A typed packed row-gather validates all token indices before decoding
only the selected rows. A missing output tensor reuses that exact embedding
owner, including packed bytes, for the tied output projection. Even partial
rotary widths are supported. The exact
checked-in q/k RMSNorm convention is supported either per head after reshape or
at the complete projection width before reshape, always before RoPE. The
source-evidenced all-or-none q/k/v projection bias family is also supported.
Optional `rope_freqs.weight` is named and shaped explicitly but remains unused,
matching the checked-in loader. RoPE frequency scaling metadata, unequal value
width, every other bias family, experts, LoRA/MLA, SSM, and non-Llama
architectures fail as typed unsupported variants; tensor names are never
discovered heuristically.

Private decoder, cache, model, generation, batch, batch-generation, and chat
modules compose that state
into inspectable fixed-shape Graphs and execute them through the CPU semantic
oracle. The supported single-sequence layer path is dense-or-packed Llama: I64
token embedding, RMSNorm, optional q/k/v projection bias, optional
source-positioned q/k RMSNorm, source-exact q/k interleaved-to-half-split output permutation,
positioned split-half full or partial RoPE, causal scaled attention with GQA,
attention projection/residual, SiLU-gated feed-forward/residual, final RMSNorm,
and explicit or tied output projection. The N-layer plan loops that exact graph
composition and commits every layer's graph-produced F32 keys and values only
after all logits and caches execute. The CPU oracle dequantizes ordinary bound
projections one at a time for graph evaluation rather than owning a whole dense
model state; packed embedding rows and a packed final projection instead decode
blocks directly without materializing their full tensors. Two-layer GQA fixtures, including partial
RoPE, q/k normalization, and projection bias at nonzero positions, match an
independent dense oracle and both token-by-token and chunked cached execution.

Generation stages a fresh full-model cache and commits it only after the whole
call succeeds. Greedy selection has deterministic lowest-ID tie behavior,
EOS/EOT stopping, explicit context errors, and tokenizer prompt/ID/decode
composition. The checked-in Gumbel-max score transform is also supported with
an explicit row-major uniform tape, making replay and tape consumption exact;
this does not claim parity with tinygrad's implicit Threefry RNG state.

The padded batch plan extends the same Graph composition to independent row
lengths and absolute RoPE positions. Fixed `[batch, kv_heads, context,
head_dim]` caches scatter each active right-padded chunk into its row, mask
future and padding positions, and commit all rows and layers only after every
output succeeds. Batch generation has independent EOS/EOT state and an
explicit `[step, batch, vocabulary]` row-major tape. Serialized dense and mixed
packed GGUF fixtures prove reader, fixed-schema binding, tokenizer, chat
formatting, model execution, and generation together.

The checked-in tinygrad CLI delegates GGUF `tokenizer.chat_template` to the
external Jinja runtime. RustGrad therefore accepts only one exact simple Llama
template string whose semantics match the checked-in Llama fallback formatter;
absent metadata selects that fallback, while every other Jinja/control template
is rejected structurally. String-only system/user/assistant messages are
bounded. Tool, multimodal, generic Jinja, other tokenizer-family templates,
symbolic/asynchronous/distributed batching, automatic family-specific tensor
rewriting, RoPE scaling, non-qkv bias, MLA/MoE/SSM variants, accelerated-device
embedding gather, accelerated-device decoding, and native quantized cache arithmetic
remain unsupported.

`transformer/native.rs` stages the same concrete Graph into one typed operation
per boundary. Arithmetic, comparisons/selects, reductions, static shrinks, and
matmuls are captured, serialized, decoded, and replayed under strict scalar
`NativeJit`; fallback is never selected. Packed Llama embedding lookup becomes
a `QuantizedRowGatherPlan` artifact whose ordered ABI is integer indices,
read-only packed bytes, then F32 output. It preflights every index before
compilation or output mutation and decodes only selected rows. Packed Llama projections become
`QuantizedMatmulPlan` artifacts with exact typed bytes in a separate ABI slot;
their F32 placeholder and transpose nodes are never executed, and traces expose
the tensor name, GGML format, and packed-byte count. A shared `MovementKernelPlan` adds
graph-independent, preflighted interpreter execution plus artifact-backed native
concat, checked integer gather, replacement scatter, and homogeneous F32/F64
scatter-add with an ordered pointer ABI. Reshape,
permutation, and expansion remain explicit movement-only CPU-oracle stages
pending shared affine-view lowering. The complete trace exposes which path
produced every node. Native single-sequence and fixed-batch caches commit all
layer outputs only after the whole staged execution succeeds. Full/token/chunk
parity, compile-cache reuse, artifact round trips, and one fixed right-padded
batch are differentially tested. A two-layer mixed-format partial-RoPE/GQA
fixture matches its independently dequantized dense control for direct/native,
full/token/chunk, and fixed-batch execution while asserting that every packed
embedding lookup and projection uses an explicit quantized stage and no dense full-weight binding. Different sequence and padded-batch extents
produce honest separate artifacts; symbolic/dynamic batch artifacts remain
absent.

`transformer/native_generation.rs` drives those native caches for tokenizer
text and the checked Llama chat formatter. Greedy and explicit row-major
Gumbel tapes preserve direct-generation token/text results, independent
EOS/EOT state, and fixed-batch row ordering. Every step records input/cache
lengths, native stage traces, and compile-cache growth. Cache and generated
tokens commit only after every native step and final decode succeeds; injected
stage failures prove rollback. This is fixed concrete-shape generation, not
continuous or symbolic batching.

`transformer/serving/` adds continuous request admission and removal between
decode steps without introducing a second numerical path. It deterministically
maps arrival-ordered active requests into concrete fixed-shape strict-native
batches, preserves one explicit Gumbel tape per request, and commits token,
request, and per-layer cache state only after every selected native stage and
decode succeeds. Unrelated queued requests are not part of that transaction.
Immutable prefix entries contain cloned per-layer K/V rows and their verified
last logits. Keys combine deterministic configuration and dense plus exact
packed-byte/type state identities
with the exact token prefix; longest-prefix lookup is deterministic and row
snapshots are copied into a fresh batch so diverging requests cannot alias.
Bounded byte/entry accounting, unreferenced-only LRU eviction, cache generations,
stale rejection, explicit invalidation, and model rebinding prevent cross-model
reuse. The checked-in tinygrad source evidences common-prefix reuse for a single
model stream, not a continuous-batching API, so RustGrad does not claim serving
API parity. Concrete batch and padded-token shapes still compile separately;
symbolic batching, asynchronous execution, distributed serving, implicit RNG,
accelerators, and native quantized cache execution remain unsupported.

## Bounded Torch state import boundary

`torch::load_torch_state_dict` is a read-only, fail-closed interchange boundary,
not a Python compatibility layer. Its `TorchStateReadLimits` and typed local
file adapters cap filesystem, archive, entry, tensor-byte, and tensor-element
budgets before composing the deterministic decoded map with the existing strict
module restore transaction. It accepts a single-root, stored or raw-deflate
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
validate-then-versioned-replace lifecycle. The file helpers do not extract ZIP
members, follow paths, create model configurations, or expose a general pickle
or Python runtime.

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

Every executor rederives and compares the deterministic plan artifact before
it allocates scratch storage or submits a copy/compute action. A serialized or
in-memory plan with altered chunks, actions, ranges, dependencies, or cache
identity therefore fails before either the dense reference executor or CUDA
mock path can mutate data; this is validation of the existing sum-plan
contract, not an additional collective form or transport.

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
returned graph best-effort. The fixed-schema prepared-prefix path instead
retains shared leases and kernels in an owned, non-self-referential graph exec.
It first applies the shared pure static-schedule plan, allocates one stable
lease per nonzero graph-ABI logical buffer, captures every nonzero PTX kernel without
per-kernel fences, uploads external values before one graph launch, records one
reusable completion fence, and downloads only exact requested outputs after
that fence completes. Decoding precedes host-map publication. A launch whose
completion cannot be established poisons the prepared executor and quarantines
its leases; settled submission and read failures leave host outputs untouched
and remain retryable. The all-zero path creates no CUDA graph or resources.

The mixed-batch coordinator plans every prefix before any backend starts
resource preparation, so a malformed later prefix cannot leave an earlier CUDA
capture or accelerator program behind. CUDA graph replay remains limited to a
fixed pointer/shape/dtype schema, one primary context, kernel-only pure static
prefixes, and explicit retained outputs. Graph updates, effects, guarded or
quantized execution, dynamic schemas, multi-device capture, capture
invalidation diagnostics, and live-driver validation remain open. The public
borrowed graph-exec foundation retains its low-level best-effort destruction
contract; the owned prepared path is the lifetime-safe surface.

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
`usize`; `Graph::input_symbolic` is the ordinary Graph specialization point. No
unbound symbolic expression can reach CPU allocation or an existing graph node.

Captured symbolic families use a separate immutable artifact schema. Capture
records stable variable identities and names, I64 domains and template values,
equality/divisibility guards, symbolic buffer shapes, the symbolic output,
reduction, or matmul domain for every schedule item, and authenticated signed
affine-view source/logical shapes, strides, and offsets. Symbolic constants are opt-in
and resize only when their nonempty template storage is one exact repeated raw
scalar pattern. Artifact decoding validates all expression references,
conservative shape/view bounds, storage policy, schedule coverage, and template
UOp geometry before exposing the capture. Specialization accepts a
complete name-to-value map, applies checked arithmetic and every guard, and
rebuilds a concrete schedule directly from the retained UOp DAG; it never
reconstructs the source Graph. Canonically ID-ordered binding values participate
in the concrete artifact identity and process-local specialization/JIT cache keys.
Materializing Pad, Concat, Gather, Scatter, Contiguous, and equal-itemsize
Bitcast kernels reuse the same artifact output domain: specialization replaces
only their authenticated operand/output shapes, then derives a fresh plan cache
key and invokes `MovementKernelPlan::validate`. Schema construction and decoded
artifact validation call the same plan-driven symbolic geometry function, so a
template-valid shape cannot conceal an invalid in-range specialization. Gather
and Scatter variable extents require a conservative all-domain inequality proof;
the source reshape/expand-to-Gather embedding composition consequently replays
variable token counts through dense materialized Gather operands without
rebuilding a Graph. A source-backed input view or direct/Contiguous-backed
computed `AffineCopy` reuses that authenticated symbolic view metadata while
keeping its input descriptor physical. Construction stops only at the exact
scheduled source; signed endpoint expressions are proved across the declared
domain before specialization. Shape-changing bitcasts still need
byte-divisibility proofs and remain fail-closed.
Specialization changes descriptor geometry, not scalar algebra: eligible newly
captured elementwise kernels were already normalized before publication, while
a decoded historical artifact keeps its original UOp structure for
byte-compatible schema validation.

## Universal UOp boundary

`uop.rs` owns the backend-neutral immutable DAG used after the typed tensor
`Graph` has chosen an expression. It has typed payloads, address-space
metadata, structural ordering, validation and deterministic rewrites. The
portable `kernel.rs` layer adds owned typed bindings, logical element versus
byte addressing, normalized row-major/broadcast index plans, and a range/load/
store interpreter for pure elementwise graphs. Bindings clone `TensorData` at
the execution boundary, so the UOp runtime cannot borrow or alias caller
storage. The CPU backend remains the differential semantic oracle.

`UOpNode` stores one typed `Operation`, its result type, and its sources. Each
`Operation` variant owns its payload, removing the former invalid cross-product
of an independently stored kind and untyped argument. The enum is declared
directly and is the sole in-memory operation taxonomy. DAG validation, rewrite
purity, interpretation, schedules, and renderers match it explicitly where
their semantic policies differ. Artifact encoding alone projects operations to
a private wire opcode and payload; the existing numeric tags and version gates
are unchanged and cannot leak back into the DAG.

Eligible pure elementwise schedules run typed UPat rules bottom-up immediately
after kernel lowering. The memoized walk preserves shared DAG nodes, revisits a
replacement until stable, and rejects a cycle or bounded-step exhaustion before
view descriptors, ordered input bindings, dependencies, cache identity, or
capture state can be published. Its constant rules delegate numeric semantics
to the portable interpreter and admit only exact Bool/integer operations,
homogeneous Bool/integral comparisons, Bool logic, same-type casts, and constant
Bool selection. Floating identities and reassociation remain deliberately
unfolded because signed zero, NaN ordering, and payload behavior are observable.
Conditional control, guards, reductions, effects, and artifact decoding stay
outside this normalization boundary.

Validation also binds address semantics to the defining operation:
`DefineGlobal`, `DefineLocal`, and `DefineRegister` require the matching embedded
`AddressValue` memory space. `LinearKernel::from_uop` validates the complete
source DAG before deriving instruction order, so a structurally sortable but
semantically invalid address graph cannot enter late linearization.

This is an intentional pre-1.0 Rust source migration. `UOp::from_operation`
is the typed constructor and `UOp::operation()` exposes the borrowed enum.
There is no legacy opcode/argument constructor or projection that could
recreate an invalid combination. Late scalar/vector planning similarly uses one
generic `LaneInstruction<R>` whose variants own their exact fixed-arity typed
operands and optional result. Value, address and index roles are explicit;
stores have no result; casts and bitcasts, unary and binary logical operations,
and core comparisons remain distinct variants. Virtual and allocated programs
share that semantic enum through fallible operand mapping instead of parallel
kind/payload taxonomies. A single descriptor-sequence validator checks ordered
definitions, exact reaching metadata, lane widths and live physical bindings.
Unsupported source UOps are retained as exact indexed `Operation` records on a
disabled program, where scalar fallback can inspect them without manufacturing
a lane instruction. Only the private artifact codec translates `Operation` to
stable wire tags.

The direct enum does not replace semantic boundaries. Detailed type/control
validation, portable interpretation, schedule lowering, artifact tag codecs,
and CPU/PTX/OpenCL/Metal/WebGPU rendering remain local exhaustive matches because
their accepted subsets and failure behavior intentionally differ. The same
rule applies to future refactors: operations and their payloads stay in the one
typed enum, while evaluation, wire ABI, and backend capability decisions stay
visible and fail closed. Remaining high-value follow-ups are ordered as:

1. reconcile repeated scalar/linear/vector presentation mapping where a
   single typed mapping is demonstrably shared;
2. centralize presentation names without coupling them to artifact tags;
3. leave backend capability lists local unless two backends first share an
   exact typed lowering contract.

The current exhaustive-switch inventory is intentional and reviewable:

| Hotspot | Responsibility | Disposition |
| --- | --- | --- |
| `uop::validate_one` | dtype, indexing, control pairing, and payload semantics | canonical semantic boundary; keep exhaustive |
| `uop::artifact` structural validation | typed operation, source-shape, and wire admission | validate `Operation` payloads and sources before explicit wire encoding |
| `uop::artifact` private opcode codec | stable numeric tags and version gates | canonical wire boundary; keep exhaustive and private |
| `kernel` evaluators | portable operation semantics | canonical interpreter boundary; keep exhaustive |
| `schedule` lowering | materialization, dependencies, and fusion roots | canonical planning boundary; keep exhaustive |
| `viz` | operation-specific names and retained payload fields | match `Operation` directly; keep presentation metadata local |
| CPU/PTX/device renderers | backend capability and source emission | canonical backend boundaries; keep exhaustive and fail closed |
| linear/vector instruction mapping | exact lane semantics, virtual-to-physical mapping, and portable-vector admission | one generic payload-bearing instruction enum plus shared descriptor validation; backend capability remains local and fail closed |

Future scheduling will turn validated effect/control UOps into kernel bodies;
renderers will consume that scheduled form. Rewrites only touch pure nodes and
memoize by structural identity, so they cannot reorder stores, barriers, or
control delimiters.

## Scheduling boundary

`schedule/mod.rs` is a non-mutating deterministic planning view over a requested
Graph output. It classifies pure elementwise regions, records typed buffer
descriptors and cache keys, and lowers scalar or rank-N elementwise chains to
a single ranged UOp sink. Static sum/mean/product/min/max reductions fuse a pure
producer and expose one exact scalar Init→Accumulate→Finalize UOp chain. The
backend-neutral `NativeReductionPlan` validates that topology, geometry, and
source/accumulator/output dtype contract before the portable interpreter or a
renderer can traverse separate output and reduction domains. Historical RGUA
Reduce opcode/kind tags and bytes remain unchanged; its redundant `mean` bit is
a private wire projection and a mismatch rejects before scheduling or caching.
Generalized static
matmul is a materialization root whose immutable Matmul UOp payload reuses
`MatmulKernelPlan` for normalized batch/vector/M/N/K geometry, original and
promoted dtypes, ordered operands, and cache identity. Eligible nonempty F32
matrix forms carry a distinct `TiledMatmulPayload`; it records conservative
target limits, selected block M/N/K, exact workgroup/shared layouts, register
tile/vector width, tail predicates, uniform barrier phases, occupancy/resource
estimates, transparent estimated cost, and deterministic identity. Candidate
enumeration is a fixed heuristic, not hardware profiling. Exact-tile homogeneous
F16/BF16 matrices on retained sm_80-or-newer capabilities instead carry a
`TensorCoreMatmulPayload`. It fixes the m16n8k16 row/col instruction, one-warp
geometry, raw narrow shared staging, per-lane A/B/F32-accumulator fragments,
uniform barriers, checked resources, an exact-tile tail policy, and deterministic
identity. Vector/dot, zero/K=0, zero-batch, M/N/K tails, and unsupported
capabilities retain an explicit serial payload. Computed operands become
ordinary producer items, so matmul participates in dependencies and temporary
lifetimes. A deterministic
temporary-plan utility only reuses caller-designated internal buffers with
non-overlapping lifetimes and compatible size/alignment. Vectorization and
device rendering retain their own capability boundaries. A separate
`ScheduledOutputs` collection is the canonical nonempty ordered output ABI:
`ScheduleItem` stores no second primary projection, and one-output consumers
call `primary_output()` explicitly. Historical artifact codecs privately
project the first descriptor to preserve and validate released RGSA/RGSM/RGSO
bytes, while a coupled static `Sort` owns one value descriptor and one I32 index
descriptor. The CPU interpreter remains its oracle. One checked
`PortableSortPair` projects dense Bool/I32/U32/F32 Sort onto a serial tinygrad
bitonic network plus occurrence-count index reconstruction for PTX, OpenCL,
Metal, and WebGPU prepared execution. Equal and unordered comparisons retain
the left lane, so padding, NaN/signed-zero payload selection, and stable
duplicate indices stay source-aligned. Unsupported storage/views/symbolic
plans and serialized executable capture remain closed; renderer-private keys
separate generated source without an Op/UOp/schedule/artifact identity change.
`argsort` chooses the index descriptor and `top_k` composes only checked
slices over the same stable ordered pair. `TensorGuard` is a typed value
passthrough schedule root with finite/nonnegative/positive-row-total metadata.
It authorizes a session-owned pending Threefry reservation only after CPU
validation, so a failed guard cannot advance the implicit stream or append a
downstream random node. Capture, prepared, native, and device paths reject
these guarded/order-specific routes explicitly.
A separate
`QuantizedMatmulPlan` owns the Llama linear orientation: dense F32 activation
`[..., K]` times a read-only packed GGML `[N, K]` weight produces F32
`[..., N]`. Its packed binding has its own descriptor and ABI slot rather than
a fake dense buffer dtype. The exact block size must divide K (including the
defined K=0 case), and packed constants never enter temporary reuse.

`engine::capture` retains an immutable schedule, ordered input ABI, constants,
and requested buffer identities for backend-neutral interpreter replay. It does
not retain a Graph, rebuild scheduling, provide one runtime-polymorphic kernel,
or participate in CUDA graph capture. `CapturedSchedule::to_bytes` writes a
versioned, bounded, checksummed artifact containing typed schedule descriptors,
explicit dependencies and ordered dense/packed bindings, topological UOp node
tables, exact raw `TensorData` storage, and exact quantized constant bytes.
RGSA v7 derives durable item, symbolic-specialization, mixed-state-binding, and
sharded source identities from explicit versioned canonical bytes and stable
FNV-1a, never Rust's implementation-selected `DefaultHasher`. Legacy v1--v6
decode first authenticates the historical envelope over its stored opaque
keys, validates structure and bindings without comparing them to current keys,
then rekeys in dependency order and performs the complete current validation.
The related inspection-only RGSO v2 and mixed/effect RGSM v3 envelopes apply
the same legacy authentication and current-key upgrade rule.
RGSM deliberately rejects symbolic schemas and specialization provenance until
its wire format can retain them. Sharded CUDA local stages likewise admit one
schedule item exactly; empty or multi-item DAGs fail before a partial source
identity or kernel can be published.
Dynamic-control buffer identities and execution-summary hashes remain
process-local implementation details and never enter persistent artifacts.
`from_bytes` validates the complete artifact,
including view bounds, scalar-tiled/tensor-core resource, barrier and fragment
metadata, and resource identities,
before rebuilding UOps. Static
elementwise, shrink-view, reduction, generalized dense matmul, quantized
linear, and quantized row-gather schedules replay
without a Graph. Malformed matmul geometry, dtypes, identities, and ordered
descriptors are rejected during artifact validation.
Captured Threefry sources use a zero-input `RandomKernelPlan` UOp whose shape,
distribution, device/key/counter reservation, and planned word count are
immutable artifact data. Interpreter replay executes that plan through the
pure Threefry core without consulting the stream registry. The native C
renderer is isolated in `cpu_jit_random.rs` and renders static uniform,
tinygrad's F32-source Box--Muller normal, and F32-uniform-scale-then-cast
randint plans for every currently public CPU dtype contract. The CUDA PTX path
uses the same immutable payload for static uniform/normal F16 (sm_53+), BF16, F32, and F64,
plus F32-uniform-scale randint for every integer storage type: it
inlines Random123 Threefry2x32, including carry-safe chunk counters and the
low-lane-then-high-lane word packing, and retains that payload only as
owner-scoped deterministic mock metadata. Neither loading nor launching reads
or reserves a process-global stream. Normal follows the paired F32 Box--Muller
control flow with target approximate transcendental instructions, so live CUDA
is a documented tolerance contract while mock execution remains plan-exact.
State reservation and live accelerator validation remain explicit boundaries.

The public live `Graph::threefry(counter, key)` route is deliberately a
different operation from that zero-input random source. One typed
`ThreefryValue` retains both packed-U64 dependency identities, their exact
broadcast shapes, the output identity and shape, and the first-use pointer
order (deduplicating aliased operands). The graph-free CPU interpreter, RGUA
v19 codec, strict C11 renderer, and PTX renderer all validate and consume that
same value. One checked `PortableThreefry` projection now also owns the dense
U64 ABI and right-aligned broadcast address terms for prepared OpenCL, Metal,
and WebGPU prefixes. A shared dialect coordinator emits the source's exact
20-round wrapping rotation/injection program; OpenCL and Metal bind native
U64 words, while WGSL binds each word as two adjacent U32 lanes. Renderer-only
versions isolate generated source without changing Op, UOp, schedule, capture,
or artifact identity. OpenCL capability-gates 64-bit integer storage, zero
domains submit no work, and unsupported or malformed bindings reject before
resource work. Semantic mocks are exact; live-device validation, ambient stream
mutation, and device-random reservation remain unclaimed.

Durable schedule keys encode each kernel with its minimum admitted semantic
RGUA envelope rather than the current standalone writer version. Existing
operations, including reductions, retain their released v18 key bytes when the
v19 Threefry tag is added; a kernel containing Threefry uses v19. The checked
static-position movement payload is the sole v20 addition. Its plan key uses an
explicit canonical FNV encoding rather than extending the historical
`DefaultHasher` movement-key seam; every earlier movement plan remains on its
released v18 durable schedule bytes and key, while ordinary standalone RGUA
encoding stays on the prior v19 writer envelope. Corrected reduction and
movement code generation is separated by renderer-specific source/cache
versions. RGSA, RGSO, and RGSM identities consequently remain stable without
pretending that an older decoder understands a newer operation.

`CapturedReplayExecutor` owns process-local scalar and vector CPU-JIT caches;
compiled libraries and pointers never enter the artifact. A typed replay policy
selects interpreter, strict native JIT, or explicit interpreter fallback. The
executor validates the whole artifact, all named binding descriptors, and every
native schedule ABI before compiling any item, then compiles all eligible items
before executing one. Symbolic artifacts must first specialize through a complete
guarded binding set; repeated canonical values reuse the concrete specialization,
while distinct values receive distinct concrete and native cache identities.
Native invocation maps the schedule's operand-order ABI
onto the renderer's buffer-ID ABI without reconstructing Graph nodes.
Strict-native module inference additionally applies a private conservative
reverse-demand pass. Requested empty pure outputs become exact owned typed zero
`TensorData` values, while only their dead pure ancestors are retained as
non-escaping `ReplayValue::PrunedZeroDomain` placeholders. Boundaries, effects,
and externally observable values remain roots; an attempted live read of a
placeholder is a typed invariant error. This leaves ordinary captured replay,
mixed/effect execution, artifact identity, and positive-domain JIT caching
unchanged while avoiding native preparation for proven-dead empty-module work.
The public graph-free adapter differentially covers static F32 `Linear` and
`Sequential[Linear, ReLU, Linear]` under scalar and vector policies; ReLU uses
the existing typed `GraphUnary(Relu)` C lowering rather than a module-specific
native path. Cache/trace identity includes the canonical `0.*`/`2.*` parameter
versions and input descriptor, while unsupported later graph items fail during
complete planning before native execution.
The same adapter also covers the released two-class configured CIFAR chain:
static F32 NCHW/OIHW `Conv2d(3→2, 1×1, groups=1, unit stride/dilation, zero
padding, optional bias) → ReLU → AdaptiveAvgPool2d(1,1) → Flatten → Linear(2→2)`.
`MovementKernelPlan::AffineCopy` is the narrow pure static affine-view boundary:
any exactly bounded affine read map from a source or computed producer is copied
into fresh owned dense storage. A Contiguous boundary selects this plan directly
instead of publishing an intermediate view. For a concrete static schedule, one
checked sole-use, unrequested ordinary scalar producer may instead be lowered
directly into the Contiguous node's fresh dense output, eliminating only the
intermediate producer allocation and raw copy. The same checked route admits a
rangeifiable reshape, permutation, expansion, shrink, or signed stride between
the producer and Contiguous when every materialized producer leaf has either
the producer's exact dense shape or one element. Such loads reuse the existing
`IndexValue::View`; no second index or movement taxonomy is introduced. The
Contiguous node remains the schedule, capture, dependency, and output identity;
requested, shared, external, stateful, guarded/faulting, specialized, nested
source-view, coordinate-div/mod reshape, dynamic, and otherwise uncertain producers
retain the explicit copy. Eligibility is rehearsed through the existing PTX,
OpenCL, Metal, and WGSL ordinary-kernel renderers, so a dtype or scalar operation
outside any established backend route keeps the raw copy.
Ordinary same-geometry scalar consumers also use this checked affine read
composition without inserting a Contiguous boundary. Candidate collection is
branch-local: each removable computed producer must have one canonical affine
map across all of its exclusively owned occurrences, while scalar leaves splat
and exact or right-broadcast-compatible materialized leaves receive their own
`IndexValue::View`. Lowering memoizes by graph node plus affine map, so a diamond
under one map shares its UOp while two maps of the same producer remain split.
The final normalized Sink—not the graph walk—owns the input/dependency ABI and
is rehearsed by every portable renderer before roots are removed. Requested,
shared, external, faulting, specialized, nested-map, symbolic, or otherwise
uncertain branches retain their materialization boundaries. Schedule item and
cache identities can consequently change from a split producer/view/consumer
chain to one existing ordinary Sink, but durable UOp and artifact formats do not.
Signed reverse and zero-stride broadcast reads are valid
because source aliases never alias the output or each other as write targets.
The interpreter and established movement-v2 CPU renderer share that immutable
addressing normalizes the proof to nonnegative indices and copies raw storage
widths. One checked `RawCopyView` projects only `AffineCopy` and dense
`Contiguous` plans into their exact two-buffer ABI. Its renderer-neutral
`RawCopyAddress` owns the row-major divisor, reverse, and stride terms, leaving
PTX, OpenCL C, MSL, and WGSL to spell only backend syntax and packing. Affine
copies use the source descriptor's full element count and map each output lane
through the normalized address; dense copies use the identity lane. PTX,
OpenCL, and Metal load/store raw 8/16/32/64-bit storage words (OpenCL gates
64-bit words on integer capability), while WGSL uses disjoint packed-word
atomics for 8/16-bit lanes and raw words for 32/64-bit lanes. Operation-specific
renderer versions isolate these new source/cache identities without changing
RGUA/RGSA movement bytes or generic elementwise identities. The fixed-schema
static executors can therefore retain computed producers and materialized
outputs on device. Guarded static-rank symbolic capture retains the explicit
producer and movement boundary, authenticates its exact source, and specializes
contiguous/singleton reshape, permute, zero-stride expand, stable shrink, and
full reverse maps through the same signed `AffineView`. Effects,
coordinate-div/mod reshape, ambiguous stride signs, dynamic rank/cardinality,
broader producer-output redirection, other movement kinds, and live-device
validation remain fail-closed or unclaimed.

`StaticPositionMap` is the single crate-private geometry proof for the existing
`ScatterPositions` graph adjoint and its `ScatterPositionsVjp` reverse read. It
validates rank, checked byte geometry, nonzero steps and both endpoints in
O(rank), while admitting a normalized `-1` start on an empty reverse domain.
Forward CPU execution allocates a fresh dense output, zeroes its raw bytes, and
places each source lane exactly once; the VJP lowers to the existing
`AffineCopy` plan. One checked `StaticPositionWrite` projects the same proof
into an unsigned output-to-input inverse map. PTX, OpenCL C, MSL, and WGSL use
that projection for one output-driven kernel in which every lane writes either
the exact raw source payload or raw zero, avoiding a second memset kernel and
write races. Their operation-specific static-position-v1 renderer identities
preserve RGUA v20 and historical movement/cache identities. Scheduling,
captured interpreter replay, strict C11 native execution, and fixed-schema
prepared accelerator execution keep computed and external inputs as read-only
nonaliasing operands across every concrete storage width. Symbolic
specialization and unproven live-device configurations still reject before
backend allocation or cache publication. Adding `ScatterPositions` to the
otherwise established public `MovementKernelKind` is an intentional 0.1
exhaustive-match source API change; no second operation taxonomy or backend
trait is introduced.
The opt-in `infer_module_native_cpu_with_report` facade reuses that exact
preflight/plan/execute path rather than adding a profiler or executor. Its
immutable report pairs the canonical no-reuse static `ExecutionPlanSummary`
with existing native/cache/zero-domain facts and three current-call local
wall-clock phases: graph/schedule/capture construction, complete native
preparation, and detached native execution. Durations are deliberately excluded
from deterministic identity and are not hardware performance, per-kernel, RSS,
allocator-capacity, device-memory, or cross-thread observations.
Immutable
`CapturedBatch` values bind several same-identity invocations; batch preflight
specializes and validates every invocation and compiles every concrete schedule
before any invocation executes; invocation and item traces are ordered, and each
invocation receives fresh owned outputs. Scalar and contiguous-vector native
elementwise, homogeneous F32/F64 matmul, static reductions, and exact
Q4_0/Q8_0/Q4_K/Q6_K linear and row-gather replay are covered,
including vector tails, zero-sized domains, broadcast batches, materialized
dependencies, aligned contiguous views, legal strided scalar views, and vector
scalar splats. Symbolic specialization covers static-rank dense elementwise
broadcasting, static-axis reductions, generalized matmul, source-backed affine
movement chains, direct or Contiguous-backed computed affine copies, and
exact-splat constant resizing. Non-affine or misaligned
vector views require the explicit fallback policy or return an
error. Rank or output-cardinality changes, arbitrary constant resizing, mutation
aliases, control flow, device launch expressions, and native cache serialization
remain outside the artifact contract.

`rangeify/` owns pure movement-to-index metadata. It extracts source-backed
static shrink, contiguous reshape, permutation, expansion, and signed-stride
chains into a deterministic canonical `AffineView` before kernel lowering.
For a computed base, the scheduler materializes that producer exactly once and
uses the same affine descriptor for a fresh dense read-copy output; source-backed
views remain direct addressing or passthrough values. Pad validity, unsupported
coordinate-div/mod or multi-map compositions, dynamic rank, and non-affine
composition remain explicit boundaries rather than hidden host materializations.

## Static-graph autograd lifecycle

Gradient recording is graph-local state. `Graph::no_grad` temporarily disables
recording only for its closure and restores the prior state even while unwinding;
there is no process-global gradient switch. Float inputs default to tracked,
while constants and explicitly frozen inputs do not. Every resulting node
carries an inspectable `requires_grad` bit derived from the shared structural
reverse-edge projection, with the coupled Sort producer retaining its
historical lifecycle-bit policy.

`Graph::detach` is a value-preserving `Detach` node: it is a new tracked float
leaf, but reverse traversal deliberately does not cross its input edge. This
matches the useful tinygrad distinction between sharing a value and sharing a
gradient history.

`Graph::grad` retains a differentiable derivative graph. `Graph::grad_with`
accepts an explicit same-shaped upstream node and its `create_graph` flag
controls whether newly built derivative nodes themselves record reverse edges.
Both it and the ordered multi-target transform derive one graph-checked,
edge-aware root-to-target frontier: predicate/index edges, Detach, nonfloating
Cast boundaries, and unrequested operand derivatives are not traversed or
built. The complete seed and derivative candidate commits only after every
local rule succeeds. The static graph does not retain or free a tape: the graph
is immutable in meaning, and each successful transform appends nodes.
Ordinary broadcast-shaped reverse edges, including `Expand`, share one private
typed unbroadcast projection: equal descriptors are identities; otherwise the
incoming cotangent casts to its `sum_accumulator_dtype`, reduces the checked
leading and singleton broadcast axes, reshapes to the operand descriptor, and
casts once back to the incoming cotangent storage. Thus Bool/integer cotangents
follow their established accumulator widths, Float8/F16/BF16 accumulate in F32,
and F32/F64 retain their width. The public raw
`Graph::sum_to` operation and its same-storage CPU contract remain available
but are no longer emitted by ordinary VJPs. Float8 broadcast unbroadcast is
covered by the CPU oracle for all four formats; broader Float8 derivative rules
and native/device execution retain their existing local capability gates. No
new Op, UOp, schedule, capture, or artifact tag is introduced.
Parameters retain graph-independent versioned host state. A graph-local
registry captures each parameter identity and version into one immutable input
leaf; optimizer writes reject stale or
wrong-identity gradients, and subsequent forwards bind the new host version.
In-process `TrainingCheckpoint` resume retains those host objects and validates
their exact identity/version/value stamps before restoring fresh optimizer and
scheduler state, so versions never roll back into a graph-cache collision.

`effects::EffectSourceBridge` is the only host-interpreter seam from a pure
graph output to a frozen effect STORE source. Its immutable sidecar binds one
`NodeId`, exact persistent input snapshots, and the existing AFTER position;
it does not place graph IDs in `EffectGraph` or add an effect IR. A
`MutationTapeRecord` can derive a first-order F32 local VJP from that exact
binding for whole, injective signed-affine, and normalized static-index
replacement writes. It returns the old-state adjoint and an RHS adjoint in the
actual pure-output descriptor, reducing assignment broadcasts and preserving
last-writer semantics. `graph_vjp` hands that explicit RHS seed to
`Graph::grad_with` for pure leaves within the same clone-then-commit
transaction. Effect graph gradients, higher-order
mutation AD, capture/native/device mutation AD, global mutable aliases, and
device-resident effect state remain intentionally unsupported.

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

`examples/cpu_train_resume.rs` is the deliberately thin user-level composition
of the CPU training boundary. It owns no model state, graph IR, optimizer, or
checkpoint format: every step creates a fresh `Graph`, modules bind their
current versioned host parameters, the CPU backend realizes loss/gradient
nodes, and the existing optimizer/scheduler own mutation. `BatchIter` owns
only deterministic index order. Portable restore is performed before new graph
binding into freshly constructed module, optimizer, and scheduler objects;
evaluation builds only a read graph and performs no state transition.

`datasets/idx/mod.rs` owns exact IDX bytes and `datasets/idx/file.rs` owns the
bounded local-file adapter. The adapter validates file size and declared count/
dimensions before delegating complete magic/count/payload validation to the
single byte parser; it has no network/cache/augmentation or graph dependency.

`models/transformer/workflow.rs` is the narrow local-file Llama user facade.
It owns only composition: `read_gguf` validates bytes, `LlamaModel` binds the
fixed schema, `SimpleTokenizer` and `LlamaChatTemplate` validate prompt
formatting, and `LlamaGenerator` executes the existing CPU graph/cache path.
`generate_chat_native` is a separate explicit single-request route through
`LlamaNativeGenerator`: it returns detached native tokens/text plus the strict
native stage trace and never falls back to CPU. Both routes use deterministic
greedy selection. Each call owns a staged cache that commits only on success
and is then discarded with the request, so a rejected request cannot leak
cache state into a later workflow call. The facade has no model-runtime/device
fallback, global RNG, alternative tokenizer, native conversation cache, or
serving ownership.

`models/transformer/conversation.rs` owns the next stateful composition layer:
one borrowed immutable workflow, one released `LlamaGenerator`, and committed
chat messages. It stages the candidate transcript and relies on generator
preflight/staged-cache semantics before appending the user/assistant pair.
`examples/llama_chat.rs` is the bounded public two-turn local-GGUF entrypoint;
it adds no second model, tokenizer, cache, or generation runtime.

Generalized contractions retain their normalized geometry in the graph.
Homogeneous F32/F64 raw Matmul reverse edges reuse `SourceDotPlan`: vector,
matrix, and right-broadcast batch adjoints are expressed as checked
reshape/transpose/expand/Mul/typed-unbroadcast compositions, so ordinary
static scheduling, capture, memory planning, and native lowering own the
complete derivative. Only the requested operand role is constructed, and a
private clone commits atomically after the final descriptor check. Narrow and
mixed raw Matmul retain the dedicated `MatmulGrad`/`MatmulGradVjp` coordinate
map because replacing their storage-width recurrence would change rounding.
`EinsumGradVjp` likewise retains the original `EinsumPlan`. Those dedicated
maps are inspectable second-order closures, not a claim of arbitrary-order
indexed contraction; their VJPs remain a deliberate future primitive.

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
Its strict F32/F64 unary subset includes `Sin` and `Trunc` with deterministic
scalar/vector renderer identities; narrow storage rejects before rendering, and
`Tan` and `Cos` remain outside the native contract.

Its stable entry point is `int rustgrad_kernel(void **buffers, const int64_t
*symbols, uint64_t *failure)`. A nonzero result reports a guarded per-element
division/modulo or shift error together with its first linear index. Buffer order, dtype, element count, byte length, output mutability,
alignment, ABI version, and symbol count are checked by Rust before the call.
`JitKernel` owns the dynamic library (and therefore outlives its function
pointer); calls borrow all buffers for their whole duration. The sole unsafe
boundary is `dlopen`/`dlsym` plus that C ABI call. No writable executable memory
is allocated by Rust.

Quantized linear C kernels receive dense activation/output buffers and one
separate read-only packed byte resource. Format-specific code decodes one GGML
block on demand inside the K loop, accumulates through f64, and requantizes only
the final F32 output; it never allocates or accepts a full dense weight buffer.
Q4_0/Q8_0 nibble/signed-byte ordering and Q4_K/Q6_K scale/min/high-plane
ordering come from the same checked decoder contracts used by GGUF
materialization. Portable artifact identity includes exact packed bytes, and
process-local native cache identity includes the validated content identity.
The sibling row-gather kernel receives an integer tensor, that same typed
packed resource, and an F32 output. It checks all signedness and bounds first,
then decodes only the addressed row blocks; repeated and empty static index
domains are defined without a dense embedding allocation.
This CPU path is serial and inference-only; quantized backward, additional
formats, tiled decoding, and device execution remain separate work.

Source and libraries are content-addressed below the OS temporary directory.
The key includes renderer/ABI version, host target, the literal `cc` command,
every fixed compiler flag, a renderer-path discriminator, and the rendered UOp
source. Unique same-directory temporary files, a process-local mutex, and
atomic rename prevent duplicate publication; loader rejection evicts one corrupt
regular-file entry and rebuilds it once, while compiler diagnostics remain
bounded.

## File and null runtime boundary

`runtime::file::FileBuffer` is an owned, checked file-I/O byte resource. It is
explicitly copying I/O rather than mmap or zero-copy: every window validates
logical bounds before seeking, writes require read-write access, and typed
`TensorData` adapters use the existing portable little-endian representation.
`read_tensor_file` additionally bounds file metadata before copying one aligned
flat dense raw file into a rank-one `TensorData`, preserving canonical raw
storage bits (including an empty file); it does not infer richer shape metadata
or provide mapped backing. `save_tensor_file` first encodes complete canonical
little-endian `TensorData` bytes, then exclusively creates and syncs only its
own same-directory staging file before final rename; failed staging is cleaned
without opening an existing target for writing.
or introduce mmap, lazy, device, or native-endian storage.
The owned descriptor deliberately preserves OS handle identity rather than
canonicalizing paths, while each nonempty read/write rechecks that an
externally truncated backing file still covers its fixed logical extent before
it can extend or partially read that file.
`runtime::null::NullRuntime` validates logical allocation/copy requests and
records deterministic planning traces, but intentionally has no values or
semantic execution. These concrete modules establish runtime-resource evidence
without introducing a speculative common backend trait.

## OpenCL runtime boundary

`runtime/opencl/mod.rs` is the facade for a dynamically loaded OpenCL 1.2
foundation. `ffi.rs` confines exact C ABI declarations, symbol casts, and raw
ICD calls; `dispatch.rs` is the one real substitution seam used by native and
deterministic mock ICDs; `buffer.rs` seals logical identity and physical
generations; `renderer.rs`, `random.rs`, `guard.rs`, `view.rs`, `narrow.rs`, `reduction.rs`,
and `transaction.rs` own pure source/ABI, dependency-ordered guarded emission,
checked view, exact
narrow-float conversion, serial-reduction planning, and guarded-integer staged
metadata; and `resource.rs` owns side effects and RAII
lifetimes. Context children retain
their owner through cleanup, stable Rust owner identities prevent colliding raw
handles from crossing contexts, and complete bounds/owner/geometry checks run
before ICD mutation. Resources are deliberately thread-confined (`!Send` and
`!Sync`); the injected dispatch is `Send + Sync`. This avoids claiming a
concurrency contract for mutable kernel arguments or queue ordering that the
current safe wrapper does not provide.

`random.rs` is a separate graph-free OpenCL C lowering for captured
`RandomKernelPlan` sources. Its ABI is exactly one mutable output pointer plus
the checked `ulong` extent; key, counter, planned word count, distribution, and
output dtype are immutable source/cache identity. It inlines tinygrad's
Threefry2x32 chunk counter carry and low-lanes-then-high-lanes packing, and it
never touches the mutable stream registry. Uniform F16/BF16/F32/F64,
paired-F32-source Box--Muller normal for those float storages, and F32-uniform
affine `randint` for I8/I16/I32/I64/U8/U16/U32/U64 are represented. F16/BF16
and F64 reject before ICD calls without fp64; I64/U64 randint similarly needs
int64. Empty domains perform no submission. The injected ICD mock receives a
retained `RandomKernelPlan` semantic payload and realizes it through the pure
Threefry plan rather than through `CpuBackend`; native execution uses only the
generated source. Normal's OpenCL transcendental result is a live-device
tolerance contract, not a promise of bitwise equality. No device-side random
state reservation is introduced.

The correctness-first OpenCL C renderer accepts static contiguous/broadcast
elementwise UOps and checked static shrink views, including scalar splats and
non-contiguous row-major slices. View ABI metadata retains the source/logical
shape, source strides, and element offset; generated address expressions and
checked source bounds use that same descriptor. Exact storage covers Bool,
I32/U32, F32, capability-gated I64/U64 and F64, plus raw F16/BF16. Narrow floats
use portable `ushort` storage, software IEEE decode and ties-to-even encode, and
fp64 expressions so fused arithmetic follows the CPU f64 oracle before its
required f32-to-storage requantization. This deliberately does not depend on
`cl_khr_fp16`; devices without fp64 reject narrow-float kernels before ICD
calls. Raw literals, signed zero, subnormals, infinity, NaNs, strided loads,
stores, and float-family casts retain the same conversion boundary. Add/Sub/Mul
preserve integer wrapping through unsigned intermediates; comparisons, select,
floating division, and Neg/Abs are supported. Guarded I32/U32 and
capability-gated I64/U64 Div/FloorDiv/TruncDiv/Mod/FMod/Shl/Shr use versioned
transactional metadata and source identity. Every guarded UOp in an elementwise
DAG receives a deterministic producer-first guard ID. Generated C evaluates
typed temporaries in dependency order, checks operands before unsafe arithmetic,
short-circuits dependent work after a fault, and lazily evaluates only the
selected branch. Guard operands may therefore be computed from retained
casts/arithmetic and static broadcast/view loads rather than only direct loads.

Static Sum/Mean/Product/Min/Max reductions consume that shared typed recurrence
through a serial row-major OpenCL plan, including multi-axis, keepdim, scalar,
zero-output, and empty-domain geometry. Sum and Product use exact wrapping or
floating accumulator-width arithmetic and commit every update; Bool uses OR/AND.
Mean commits its divisor and divides at F32 or F64 work width. Extrema start at
the committed dtype identity and use strict typed comparisons, so equal or
unordered candidates retain the accumulator and I64/U64 never project through
floating point. Effective singleton domains bypass recurrence. F16/BF16 use the
software codecs and require the existing fp64 helper capability; Float8 remains
outside OpenCL. Empty sum writes zero, empty mean writes a canonical quiet NaN,
empty product writes its typed identity, and empty extrema remain a graph error.
Pointer arguments
follow `ScheduleInputBinding` first-use order, the output follows inputs, and a
checked `ulong` extent is the final scalar ABI. Compile-time empty reductions
omit their unused input pointer. The semantic mock executes the retained typed
UOp independently of rendered C and compares bytes with the CPU oracle; native
execution never falls back to the host. The ignored live smoke exercises a
strided view, extrema, fp64-gated Product, and raw F16/BF16 special values when
explicitly invoked. Every `OpenClBuffer` is now a stable thread-confined logical
identity with checked byte length, optional storage dtype, and a private visible
physical generation. Ordinary copy and launch submission snapshot that
generation, and their event retains the physical allocation until command
cleanup even if logical visibility later changes. Typed allocation rejects an
ABI dtype mismatch before ICD calls.

Guarded integer kernels bind an invisible candidate generation and a status
word initialized to `u32::MAX`; invalid work-items atomically minimize a packed
`(logical index, guard ID)` key. Elementwise kernels use output index, while a
fused reduction uses the original row-major source index. Each reduction source
belongs to exactly one output, so this bounded key defines a global order without
depending on output work-item order. The serial reduction evaluates its producer
guards before accumulator arithmetic and writes only its candidate output. Lazy
logical And/Or and select evaluate only the active operand or branch in generated
C, semantic mock execution, and bounded diagnostic reconstruction. The
non-cloneable launch token
retains kernel, queue, borrowed bindings/destination, submitted input/output
generation snapshots, candidate/status allocations, and compute event. `query`
observes compute readiness. Consuming `wait` completes
compute, reads bounded status, and reconstructs an invalid computed shift count
with a retained typed scalar-expression evaluator. That diagnostic path reads
only the exact captured-generation scalar loads needed at the selected lane; it
does not participate in successful device computation. The token then
revalidates every submitted generation and swaps
the candidate into logical visibility only when clean. No device copy mutates
the old visible allocation. Thus allocation/submission/compute/status/detail
and terminal wait failures leave both visible generation and bytes unchanged;
overlapping transactions deterministically reject the stale collector. Old
physical generations are released only after all retained events drop.
Reverse-order mock visitation proves deterministic index-then-guard selection.
Dynamic control/reduction axes, other unary/cast families,
runtime-polymorphic views/shapes, cross-thread
resources, and broad live ICD validation remain explicit boundaries.

## Metal runtime boundary

`CpuSession::realize_metal` is the intentionally narrow public deployment
adapter for this runtime. The caller owns the loaded `MetalDevice` and therefore
its thread-confined cache; the session keeps Graph nodes and binding maps
private. It resolves only canonical static input bindings and graph constants,
builds the ordinary schedule, and runs `MetalPrefixPlan::plan` across every
item before resource preparation. `MetalSessionTrace` retains ordered logical
item/cache identities and capabilities but no handles, pointers, or current
tensor bytes. Legal zero-domain pure items are retained as typed plan sentinels:
they materialize exact empty `TensorData` values without queue, pipeline,
buffer, or command work. This adapter is strict—unsupported renderer/ABI/view/
dtype/capability items return before resource work and never select CPU
fallback. It currently covers static elementwise/view and proven typed-reduction
session graphs, not model/Linear/ONNX inference, unsupported unary activation,
effects, dynamic
shapes, persistent device state, graph capture, or profiling.

`MetalRuntime::discover` is the narrow diagnostic seam for deployment setup:
framework/symbol errors remain structured `MetalError`s, while a successfully
loaded runtime with no process-visible GPU yields `MetalDiscovery::NoDevices`.
It creates no queue or executable resource. This matters on managed macOS
processes where hardware inventory can report Metal support without granting a
usable device to the current process; live smokes are evidence only when this
discovery step returns a device.

`runtime/metal/mod.rs` is the facade for the first Apple Metal execution
boundary. `ffi.rs` dynamically loads the Objective-C runtime, CoreGraphics, and
Metal frameworks on macOS; RustGrad therefore needs neither Apple SDK headers
nor link-time framework flags. Opaque Objective-C calls and function-pointer
casts remain confined to that file. `dispatch.rs` is the private native/mock
substitution seam, `buffer.rs` seals logical buffer metadata and retained
physical generations, `renderer.rs` owns ordinary pure MSL and pointer-ABI lowering,
`random.rs` owns graph-free captured Threefry MSL lowering, and
`guard.rs`/`transaction.rs` own guarded source emission and typed fault
metadata/reconstruction. `resource.rs` owns device, queue, library, pipeline,
command, transaction, cache, and completion lifetimes. No raw handle is
reachable from a safe public API.

Devices are returned in deterministic registry-ID/name order with capability
metadata in renderer and pipeline cache identities. Resources are deliberately
thread-confined; command tokens retain every submitted physical allocation,
offer a nonblocking readiness query, and consume themselves when collecting
completion. Shared-storage H2D/D2H copies and encoded D2D blits validate owner,
generation, byte range, and optional storage dtype before native calls. Source
compilation disables fast math and returns bounded native diagnostics. Launch
preflight validates ordered buffers, exact byte/dtype contracts, static extent,
and pipeline threadgroup limits before constructing an encoder. Logical buffers
have a stable identity around an interior visible physical generation; ordinary
commands retain snapshots, while guarded commands retain inputs, candidate,
status, and their original generation through terminal collection.

The correctness-first MSL renderer consumes scheduled UOps and their
first-lowered-load `ScheduleInputBinding` order. It supports exact stored
F32/Bool/I32/U32 constants and loads; wrapping integer and floating Add/Sub/Mul;
comparisons, logical operations, select, and exact Bool/I32/U32 plus F32/Bool
casts; contiguous/broadcast addressing; and source-backed affine
shrink/reshape/permute/expand/positive-stride maps. Its ordinary ABI binds input
pointers first, the output pointer last, then a checked `ulong` extent.

Captured `RandomKernelPlan` sources have a one-output, zero-input ABI and
dedicated `random.rs` lowering. The immutable plan’s key, counter, word count,
distribution, dtype, device capability identity, and emitted source enter the
cache key; neither source rendering nor launch reads the stream registry. The
MSL helper reproduces tinygrad’s Threefry2x32 carry-safe `2^32-1` chunking and
low-lane-then-high-lane packing. Current Metal storage safely represents F32
uniform and paired F32-source Box--Muller normal, plus F32-uniform affine
randint for I32/U32. Empty plans do not submit. The injected mock retains the
typed plan and uses the pure Threefry semantic directly, independent of
`CpuBackend`; MSL normal transcendental agreement is an ignored-live-smoke
tolerance contract. The ordinary command path retains submitted generations and
validates them on collection; all random capability/owner/binding failures are
preflighted before command submission. F16/BF16/F64 and 64-bit integer random
plans are structurally rejected because current Metal capability/storage and
atomic-status plumbing cannot make their raw-storage contract.

Guarded I32/U32 Div/FloorDiv/TruncDiv/Mod/FMod/Shl/Shr use a provisional output
and a bounded `atomic_uint` status after the extent slot. Dependency-order guard
IDs combine with logical output indices under atomic minimum, so shuffled GPU
work-item completion cannot change the reported fault. Generated MSL checks
zero divisors and signed/unsigned shift ranges before issuing arithmetic, uses
defined unsigned intermediates for signed overflow and left shift, preserves
RustGrad's Euclidean versus truncating division/remainder distinction, and
emits lazy select/And/Or branches. Completion waits before reading status,
reconstructs computed/broadcast/affine-view shift counts from retained typed
input generations, validates every submitted snapshot, and swaps the candidate
generation only when all checks succeed. Failed or competing stale transactions
leave the old visible bytes and generation unchanged; query never reads status
or commits.

Source identity includes renderer/ABI/transaction versions, local size,
complete device capabilities, ordered buffer/view/guard metadata, and emitted
source. The injectable semantic mock interprets typed UOps independently of
`CpuBackend`; CPU is used only as an external expected-value oracle. It covers
exact integer differentials, reverse/shuffled fault visitation, nested and
computed guards, broadcast/affine RHS detail, lazy branches, zero domains,
retry/stale/retention/cleanup, and allocation/status-initialization/build/
pipeline/encode-submit/compute/query/status/detail failures. Ignored live Apple
smokes remain available for a process-visible device, but the current release
host returns `MetalDiscovery::NoDevices` before queue creation; consequently
this document claims semantic/mock evidence and structured unavailable-device
handling, not successful live deployment evidence.

MSL has no F64 type. Correctness-first serial reductions consume the shared
typed recurrence for F32/Bool/I32/U32, including F32 accumulator-width
arithmetic, exact Bool/integer ordering, singleton bypass, and zero-domain
identities. F16/BF16/F64 and I64/U64 reject before submission: this milestone
does not claim their storage or 64-bit atomic status/detail contracts. Other
unary/transcendental operations, bitwise integer operations, dynamic/symbolic shapes,
runtime-polymorphic views, shared/local memory, tensor cores, graph capture,
profiling, multi-device synchronization, and broad model workloads remain
explicit Metal boundaries.

## WebGPU runtime boundary

`runtime/webgpu/mod.rs` is the facade for the first WebGPU/WGSL execution
boundary. `dispatch.rs` defines the private typed native/mock seam;
`buffer.rs` seals logical byte/dtype identity and command-retained physical
generations; `transaction.rs` owns producer-ordered guard/status metadata and
bounded diagnostic reconstruction; `guard.rs` owns dependency-ordered guarded
WGSL emission; `narrow.rs` owns the versioned software F16/BF16 conversion and
packed-word contract; `renderer.rs` owns pure WGSL plus the ordered bind-group
ABI; `random.rs` owns graph-free captured Threefry WGSL lowering; and `resource.rs` owns the thread-confined instance, adapter, device,
queue, buffer, shader, pipeline, cache, command, transaction, and completion
lifetimes. Safe APIs expose no native handles. Instance→adapter→device and shader→pipeline
retention makes the release order structural, while pending commands and
transactions retain every submitted physical generation through consuming
collection. Discovery ordering uses backend/vendor/device/name/driver metadata;
renderer and cache identity include complete adapter capabilities, backend
identity, WGSL/ABI/status/transaction/narrow versions, local size, source,
ordered buffers, affine views, and guard metadata.

The checked transfer boundary preserves RustGrad's logical byte lengths while
rounding private WebGPU allocations to four bytes. H2D/D2H validate complete
logical ranges and owner identity; native D2D additionally requires the
four-byte offset/size alignment mandated by WebGPU. Typed copies reject dtype
mismatches. Static launch preflight validates exact ordered buffer count,
dtype, logical bytes, owner/generation, `u32` extent, workgroup size, and adapter
workgroup-count limits before dispatch. Readiness query is nonblocking;
collection consumes the command, waits, revalidates submitted generations, and
returns handle-free completion metadata. WGSL build diagnostics are bounded.

The deterministic WGSL renderer consumes scheduled UOps and preserves
`ScheduleInputBinding` first-load order, followed by the unique output storage
buffer and a final uniform extent binding. It supports stored F16, BF16, F32,
I32, U32, and byte Bool constants/loads/stores; Add/Sub/Mul; comparisons,
logical operations, select; all casts among F32/Bool/I32/U32; and exact
F16/BF16↔F32 plus cross-narrow casts. Contiguous and broadcast indexing and
source-backed affine shrink/reshape/permute/expand/positive-stride views retain
the same ordered ABI. I32 arithmetic uses explicit unsigned intermediates so
overflow wraps exactly. Bool storage retains RustGrad's byte ABI: input bytes
are packed four per `u32`, while disjoint output lanes use atomic byte-field
clear/set operations so adjacent results cannot race. F16 and BF16 retain their
two-byte RustGrad ABI while physical storage packs two raw lanes per `u32`;
software bit conversion handles signed zero, subnormal, infinity, and NaN
payload/classification behavior and ties-to-even stores. Disjoint output lanes
use atomic half-word clear/set, so support never depends on WGSL `shader-f16`.
View and dispatch math is bounded to WGSL's `u32` index domain.

Captured `RandomKernelPlan` WGSL uses one mutable output storage binding and
the final extent uniform, with immutable key/counter/word count/distribution
in source and cache identity. `random.rs` reproduces tinygrad's Threefry2x32
carry-safe chunking and low-then-high word packing for F32 uniform and paired
F32-source Box--Muller normal, plus F32-uniform affine randint for I32/U32.
It is zero-input and never consults the mutable stream registry. The retained
plan semantic mock executes the pure plan independently of `CpuBackend`; zero
domains do not submit. Although ordinary WebGPU supports packed narrow storage,
random F16/BF16 stores are rejected until a dedicated disjoint-lane random
write protocol is validated; F64/I64/U64 likewise remain unsupported.

Guarded I32/U32 Div/FloorDiv/TruncDiv/Mod/FMod/Shl/Shr use a versioned staged
ABI. A private candidate allocation replaces the ordinary output binding and a
final storage `atomic<u32>` status follows the uniform extent. Each active lane
records `logical_index * guard_count + producer_guard_id` with `atomicMin`, so
execution order cannot change the earliest visible fault. Signed MIN/-1 cases
avoid WGSL's invalid division path and preserve Rust wrapping semantics. Lazy
select and logical And/Or do not evaluate inactive guards. Submission retains
the base input/output generations, candidate, and status; a fault, query/wait/
read failure, dropped token, or stale overlapping collector releases scratch
without changing the logical output. Only a clean consuming collection swaps
the candidate generation. Shift diagnostics reconstruct computed, broadcast,
and affine-view counts from retained inputs after status collection.

The injected semantic mock validates resources, copies, build/pipeline state,
geometry, bindings, owners, generations, and cleanup, then interprets the
retained typed lowered UOp with `kernel::execute_lowered_elementwise` or the
WebGPU-local typed narrow evaluator that models per-node storage rounding; it
never routes successful mock execution through `CpuBackend`. CPU is used only
for external expected values. Deterministic tests cover cache reuse, source/ABI
identity, affine+broadcast execution, bool and odd-lane narrow packing, integer
wrapping, the supported cast matrix, raw F16/BF16 special-value and cross-cast
byte differentials, guarded-operation byte differentials, shuffled fault
visitation, lazy branches, zero domains, retry/stale/cleanup behavior, retained
lifetimes, malformed artifacts/capabilities, and injected discovery,
allocation/status, transfer, build, pipeline, launch, query, wait, and
read/detail failures.

`ffi.rs` dynamically probes the usual `wgpu-native` and Dawn library names and
required symbols without a compile-time SDK. Their public C symbols do not pin
one callback/future descriptor layout across released header revisions. No
versioned WebGPU C header is checked into this repository, so the native path
currently returns a structured library or ABI error before creating an instance
or registering a callback. This deliberately prevents a callback from
outliving stack or resource ownership and means the ignored live discovery/
compile/H2D/D2D/launch/query/collect/D2H smoke cannot execute on this host.

WGSL has no F64 storage/arithmetic contract. Static serial reductions instead
consume the backend-neutral checked recurrence for the proven
F16/BF16/F32/Bool/I32/U32 intersection; narrow arithmetic commits through the
same software codec at every update, and F32 recurrence remains F32. F64,
I64/U64, dynamic axes, and parallel reduction remain fail-closed before
submission. Guarded F32/F16/BF16 and 8/16/64-bit integer
division/modulo/shift remain outside the exact status ABI and reject before
submission. F64 and narrow/wide integer storage, narrow division and remainder,
broader narrow casts, unary/transcendental/bitwise ops, dynamic or
runtime-polymorphic shapes/views, arbitrary-byte D2D, timestamps, shared/local
memory, graph capture, multi-adapter execution, native C ABI calls, live
hardware validation, and representative accelerated models remain explicit
WebGPU boundaries.

## CUDA Driver runtime boundary

`cuda.rs` is a deliberately toolkit-free, dynamically loaded CUDA Driver API
foundation. Loading `Driver` never creates a CUDA context, and fails with a
distinguishable missing-library, missing-symbol, API-version, or Driver-error
result. The tiny native loader is the sole function-pointer conversion boundary;
all operational calls instead go through a typed `Dispatch` trait, which keeps
the default test suite deterministic and CUDA-free.

The Driver boundary treats a successful status plus a null graph, graph-exec,
module, function, or link-state output as an invalid argument before
constructing an RAII owner or issuing later work. Peer-copy submission independently requires its
own optional Driver symbol, rather than inferring it from ordinary async-copy
support. Mock stream-wait Driver failures are surfaced without consuming the
owned stream/event dependency, so the unchanged pair can be retried. These
are mock/source safety contracts only; they neither invent an ABI nor claim
live CUDA or multi-GPU validation.

The linked-module path is an explicit opt-in beside legacy PTX module loading.
`LinkInput` owns ordered immutable PTX, CUDA-library, or caller-attested
pre-CUDA-12 NVVM bytes; a versioned content identity feeds owner-scoped
module/function caches. A private RAII link state destroys every created
`CUlinkState` after success or failure, and missing `cuLink*` symbols fail
closed. No toolkit discovery, filesystem lookup, payload serialization, or
claim about CUDA 12+ NVVM input is implicit in this path.

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
proven. Transcendental unary functions remain rejected by the ordinary
renderer. The sole exception is `LinkedF32ExpRequest`: after an exact F32 Exp
UOp proof with distinct canonical `DefineGlobal` buffers, one shared axis-zero
`Range`/`EndRange`, and a range bound equal to the dense element count, plus one
matching NVVM `__nv_expf: f32 -> f32` producer attestation, it emits a versioned
external call and loads only through linked caches keyed by launch block size.

`linked_exp/` owns the corresponding resource boundary. Its v1 descriptor and
sidecar are canonical, payload-free records tied to one fully revalidated
captured schedule and one request. Preparation revalidates the complete capture,
captured UOp, and F32 input/output ABI; rebind then requires the original
immutable link bytes and exact caller-owned primary-context leases. The
dedicated launcher stages into a private candidate and publishes through
backup/copy with best-effort rollback. A separate v2 artifact admits exactly
two independent requested outputs and publishes them in order through the same
mechanism. Persistent CUDA failure can also prevent restoration, in which case
the composite error explicitly reports that caller-owned outputs may be
partially modified; neither route claims atomic commit. Generic captured replay
cannot consume either sidecar. The mock uses link/request/cache,
generic-launch/completion, and commit/rollback-phase hooks to prove successful
rollback and the documented partial-mutation boundary; live CUDA/ptxas, broader
operations/dtypes, payload embedding, resource discovery, and general
captured-device replay are still outside this boundary.

Static sum, mean, product, min, and max UOp programs have a separate correctness-first PTX path.
One CUDA thread owns one logical output and serially walks the normalized
row-major reduction domain, including multi-axis and keepdim layouts; fused
eligible producers reuse the ordinary emitter with that computed input index.

Static matmul rendering stays cohesive in `ptx_matmul.rs`. Vector/dot, zero/K=0,
zero-batch and F64 forms retain the explicit one-output-thread serial-K adapter.
Eligible nonempty homogeneous F32 matrix and broadcast-batch schedules instead
carry the selected tiled UOp payload. Its PTX uses one two-dimensional
workgroup per output tile and batch, cooperative predicated lhs/rhs global
loads into dynamic shared memory, a uniform barrier after loads and after
consumption, predicated M/N/K tails, and F64 multiply/accumulation before the
F32 store to preserve the CPU scalar contract. Exact grid/block/shared launch
geometry is validated against the retained payload before module loading and
participates in rendered/cache identity. Owner-scoped mock dispatch runs the
independent tiled simulator, never the serial matmul executor, while retaining
the same ordered lhs/rhs/output ABI and broadcast projection. The deterministic
candidate cost is a heuristic only. Exact-tile homogeneous F16/BF16 schedules on
sm_80+ have a separate real tensor-core source path: one warp stages raw 16-bit
A/B tiles into aligned shared memory, manually loads the checked-in tinygrad
fragment mapping, emits
`mma.sync.aligned.m16n8k16.row.col.f32.{f16|bf16}.{f16|bf16}.f32`, and
requantizes each F32 accumulator once to the graph's narrow output. Exact
launch/shared geometry and fragment phases are source-mapped and cache-keyed;
the owner-scoped mock runs an independent lane/fragment simulator. The CPU
oracle accumulates floating matmul in F64, so broad bitwise equivalence to F32
MMA is not claimed. Current differentials use exact-representable fixtures and
raw special-value classification. Live CUDA/ptxas, tail padding, multi-warp
tiles, asynchronous copy, double buffering, and empirical autotuning remain
unclaimed.
F32/F64 retain CPU-equivalent typed accumulation/finalization; I32/I64 and
U32/U64 sums use defined wrapping PTX arithmetic; bool sum is the I32 count of
true inputs; and bool/integer mean promotes through F32 before the F32 store,
matching the checked recurrence. Empty sum domains store zero and empty
float mean domains store a canonical quiet NaN without emitting a divide by
zero. I8/I16 and U8/U16 loads explicitly sign/zero extend into their promoted
I32/U32 sum accumulators or F32 mean finalization. F16 (on sm_53+) and BF16
reduction buffers are decoded from raw 16-bit storage,
accumulated at the plan-selected F32 or source width, and deterministically
requantized after each source-width update and at the final store;
the BF16 store uses the same raw ties-to-even bit arithmetic as `TensorData`.
The same serial path also carries Product/Min/Max for every static stored
scalar dtype. Product uses typed wrapping ALU (Bool is AND), its multiplicative
identity, and the existing raw narrow-float finalization. Extrema begin at the
committed dtype identity and use strict typed comparisons, so unordered and
equal candidates retain the accumulator while I64/U64 never project through
floating point. Effective singleton reductions bypass recurrence. F16 requires
sm_53 conversion support; BF16 uses the explicit raw decode
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
