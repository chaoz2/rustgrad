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
  tensor/                dense tensor value subsystem
    mod.rs               public TensorData, Shape, Storage and dtype facade
    dtype.rs             dtype taxonomy and promotion policy
    scalar.rs            scalar and exact F16/BF16 conversion semantics
    storage.rs           owned typed dense storage
    shape.rs             checked shape arithmetic
    data.rs              TensorData construction, casts and dense access
    creation.rs          dense creation helpers
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
  onnx/                  bounded facade; private wire, tensor, schema, lowering, tests
  ir/                    typed frontend graph facade, vocabulary, shape planning,
                         storage/lifecycle, and operation-family extensions
    mod.rs               concise module wiring and public re-exports
    types.rs             public IR vocabulary and operation options
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
Hardware-only tests remain explicitly ignored in the default suite. This is a
portable baseline rather than a complete device, architecture, Miri, coverage,
or cross-compilation matrix.

`viz` is the pure inspection boundary. Typed normalizers consume graph,
schedule/capture, UOp, linear, memory-space, and vector metadata into a small
validated model. Model construction sorts node IDs, fields, and edges before a
dependency-free DOT renderer escapes labels, so construction order cannot leak
into snapshots. Graph-local node IDs and portable buffer, item, artifact, and
cache identities remain explicit; pointer identities, compiled modules, runtime
handles, profiling samples, and `Debug` text are not inputs. Unsupported Graph
operation families fail with a typed visualization error instead of being
silently flattened.

`ir::indexing` is the pure static-indexing boundary: it normalizes immutable
integer/slice/newaxis/ellipsis and constant advanced-index specifications into
checked shapes and coordinate maps. The narrow `Op::StaticIndex` and its
reverse scatter consume the plan without re-parsing it; dynamic
boolean/nonzero cardinality and mutable aliasing remain outside it.

`ir::dynamic` keeps data-dependent extents separate from static graph nodes.
Its CPU-oracle consumers are `nonzero` and unbounded boolean `masked_select`:
realization validates a concrete ranked `engine::RuntimeShape` before exposing
output storage. Schedules and optimized backends do not lower dynamic nodes yet.

Dynamic nodes can compose through a typed scalar sum and CPU first-order loss
executor. That executor carries validated runtime upstream shapes and returns a
gradient in the requested static source shape; it is intentionally separate
from `Graph::grad` until dynamic results participate in the general graph tape.

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
| ONNX model interchange | `onnx/mod.rs`, `onnx/{wire,tensor,schema,lower}.rs` | bounded parse, normalize, and CPU-graph lowering |
| `viz/*` | `viz/*` | compiler introspection |

The current `backend::CpuBackend` is deliberately the semantic oracle. It will
move behind the runtime/device contracts once those contracts are executable;
optimized CPU and GPU paths must match it through differential tests.

`onnx/mod.rs` is a bounded fail-closed default-domain opset-13 facade. Private wire
parsing, typed/raw tensor decoding, schema normalization, and graph lowering
keep untrusted bytes separate from the CPU-graph boundary. The checked surface
is static inference only: elementwise/activations; movement/indexing/shape;
Gemm/MatMul and softmax; Conv/pooling/BatchNorm/GlobalAveragePool;
predicate/Where/Clip/inference Dropout; ConstantOfShape; and reductions/args.
Dynamic controls or shapes, general Gather/indexing, control flow, sequences,
sparse/quantized/external data, custom domains/opsets, training, and live
external-model differential validation remain outside this boundary.

Source-backed affine shrink, contiguous reshape, permutation, expansion, and
positive-stride chains lower as `ViewMap`/`ViewBufferIndex` through scheduling,
interpretation, and native CPU execution. Computed-value and non-affine or
negative-stride chains stay explicit lowering boundaries. `CpuJitBackend` is an
internal cached native-execution boundary with validated `ScheduleItem`
preparation and invocation; replay never reconstructs a Graph.

Late `LinearKernel` construction validates a typed portable contiguous
elementwise lane plan before C rendering. Its immutable `LinearProgram` records
producer-first instructions, virtual definitions/uses, lane/tail metadata, live
intervals, and deterministic scalar/vector register assignment. A backend-neutral
`MemorySpacePlan` consumes those assignments, validates global/register/private/
shared identities, byte/alignment/lifetime aliases, and uniform workgroup
barriers. Eligible homogeneous F32 matrix matmul additionally derives two
shared tile promotions and its accumulator/register/barrier lifetimes from the
selected `TiledMatmulPlan`; elementwise kernels still choose no shared
promotion. `VectorProgram` is the backend-neutral physical-register instruction
view (splat/address/index/load/cast/ALU/compare/select/store/control) with
explicit lane mask and scalar-tail identity; CPU JIT validates and keys this
form before portable rendering.
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

`datasets/mod.rs` is intentionally a small local, deterministic facade. Private
`datasets/idx.rs`, `datasets/cifar.rs`, and `datasets/batch.rs` own uncompressed
MNIST IDX decoding, exact CIFAR-10 binary records, and seeded batch ordering.
CIFAR records retain their channel-major bytes as U8 NCHW tensors; pure F32
normalization accepts explicit per-channel means and positive standard
deviations. Parser unit tests own format and malformed-input contracts, while
public training/composition workloads live under `tests/`. The boundary does
not download, cache, randomly augment, or claim corpus parity. `nn::Parameter`
is graph-independent versioned host state, while each `Graph` owns its binding
leaves. `training_checkpoint/` depends one way on `nn`, `optim`, and
`safetensors`; its exact in-process resume retains the same host parameter
identities but permits fresh graphs, optimizers, and schedulers. Cross-process
identity rehydration remains outside this boundary.

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
`usize`; `Graph::input_symbolic` is the ordinary Graph specialization point. No
unbound symbolic expression can reach CPU allocation or an existing graph node.

Captured symbolic families use a separate immutable artifact schema. Capture
records stable variable identities and names, I64 domains and template values,
equality/divisibility guards, symbolic buffer shapes, the symbolic output,
reduction, or matmul domain for every schedule item, and source-backed affine
view source/logical shapes, strides, and offsets. Symbolic constants are opt-in
and resize only when their nonempty template storage is one exact repeated raw
scalar pattern. Artifact decoding validates all expression references,
conservative shape/view bounds, storage policy, schedule coverage, and template
UOp geometry before exposing the capture. Specialization accepts a
complete name-to-value map, applies checked arithmetic and every guard, and
rebuilds a concrete schedule directly from the retained UOp DAG; it never
reconstructs the source Graph. Canonically ID-ordered binding values participate
in the concrete artifact identity and process-local specialization/JIT cache keys.

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

`schedule/mod.rs` is a non-mutating deterministic planning view over a requested
Graph output. It classifies pure elementwise regions, records typed buffer
descriptors and cache keys, and lowers scalar or rank-N elementwise chains to
a single ranged UOp sink. Static sum/mean/product/min/max reductions fuse a pure producer and
expose accumulator initialization/update/finalization UOps; the portable
interpreter traverses separate output and reduction domains. Generalized static
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
`from_bytes` validates the complete artifact,
including view bounds, scalar-tiled/tensor-core resource, barrier and fragment
metadata, and resource identities,
before rebuilding UOps. Static
elementwise, shrink-view, reduction, generalized dense matmul, quantized
linear, and quantized row-gather schedules replay
without a Graph. Malformed matmul geometry, dtypes, identities, and ordered
descriptors are rejected during artifact validation.

`CapturedReplayExecutor` owns process-local scalar and vector CPU-JIT caches;
compiled libraries and pointers never enter the artifact. A typed replay policy
selects interpreter, strict native JIT, or explicit interpreter fallback. The
executor validates the whole artifact, all named binding descriptors, and every
native schedule ABI before compiling any item, then compiles all eligible items
before executing one. Symbolic artifacts must first specialize through a complete
guarded binding set; repeated canonical values reuse the concrete specialization,
while distinct values receive distinct concrete and native cache identities.
Native invocation maps the schedule's operand-order ABI
onto the renderer's buffer-ID ABI without reconstructing Graph nodes. Immutable
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
movement chains, and exact-splat constant resizing. Non-affine, negative-stride,
or misaligned vector views require the explicit fallback policy or return an
error. Rank or output-cardinality changes, arbitrary constant resizing, mutation
aliases, control flow, device launch expressions, and native cache serialization
remain outside the artifact contract.

`rangeify/` owns pure movement-to-index metadata. It extracts source-backed
static shrink, contiguous reshape, permutation, expansion, and positive-stride
chains into a deterministic affine `ViewMap` before kernel lowering. Computed
producers, pad validity, negative strides, and non-affine composition remain
explicit boundaries rather than hidden host materializations.

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

## OpenCL runtime boundary

`runtime/opencl/mod.rs` is the facade for a dynamically loaded OpenCL 1.2
foundation. `ffi.rs` confines exact C ABI declarations, symbol casts, and raw
ICD calls; `dispatch.rs` is the one real substitution seam used by native and
deterministic mock ICDs; `buffer.rs` seals logical identity and physical
generations; `renderer.rs`, `guard.rs`, `view.rs`, `narrow.rs`, `reduction.rs`,
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

Static Sum/Mean/Product/Min/Max reductions use a separate serial row-major plan,
including multi-axis, keepdim, scalar, zero-output, and empty-domain geometry.
Sum covers every supported stored dtype with wrapping integer/Bool-OR storage
semantics; floating Sum and Mean require fp64 because the CPU oracle accumulates
through f64. Integer Mean is promoted by the graph to F32 before finalization.
Product covers every supported stored dtype with
typed wrapping/Bool-AND identities; floating Product also follows the CPU f64
intermediate and requires fp64. Extrema cover the same stored dtypes, ignore
NaNs, retain the first equal value (including signed zero), and preserve raw
selected words. I64/U64 extrema require fp64 because their CPU ordering is an
f64 projection and projection ties must retain the first raw word. Empty sum
writes zero, empty mean writes the canonical quiet NaN, empty product writes its
typed identity, and empty extrema remain a graph error. Pointer arguments
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

`runtime/metal/mod.rs` is the facade for the first Apple Metal execution
boundary. `ffi.rs` dynamically loads the Objective-C runtime, CoreGraphics, and
Metal frameworks on macOS; RustGrad therefore needs neither Apple SDK headers
nor link-time framework flags. Opaque Objective-C calls and function-pointer
casts remain confined to that file. `dispatch.rs` is the private native/mock
substitution seam, `buffer.rs` seals logical buffer metadata and retained
physical generations, `renderer.rs` owns pure MSL and pointer-ABI lowering, and
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
smokes validate the original F32 transfer path and native I32 success plus
division/shift status, rollback, and exact output.

MSL has no F64 type, while RustGrad's CPU oracle accumulates floating Sum/Mean
through F64 before storage conversion. This milestone therefore rejects all
reductions rather than claiming inexact F32 accumulation. I64/U64 are also
rejected before submission: this milestone does not claim an exact 64-bit
atomic status/detail capability contract. F16/BF16/F64 and other storage,
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
ABI; and `resource.rs` owns the thread-confined instance, adapter, device,
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

WGSL has no F64 storage/arithmetic contract, while RustGrad reductions require
F64 intermediate accumulation for floating parity. All reductions therefore
reject before submission. Guarded F32/F16/BF16 and 8/16/64-bit integer
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
