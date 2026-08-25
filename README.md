# RustGrad

RustGrad is an inspectable, differentially validated tensor compiler written in Rust. The long-term target is feature parity with tinygrad while using an architecture natural to Rust.

Start with an explicit CPU session. It owns one inspectable graph and its input
bindings, so ordinary values do not require manual backend binding assembly.
Unsupported devices are rejected rather than silently using CPU.

```rust
use rustgrad::{CpuSession, SessionDevice};

let mut session = CpuSession::on(SessionDevice::Cpu)?;
let input = session.variable([2, 1], [1.0, 2.0])?;
let scale = session.tensor([3], [10.0, 20.0, 30.0])?;
let bias = session.tensor([3], [1.0, 1.0, 1.0])?;

let product = session.mul(&input, &scale)?;
let output = session.add(&product, &bias)?;
let loss = session.sum_all(&output)?;
let gradient = session.grad(&loss, &input)?;

let result = session.realize(&output)?;
assert_eq!(result.shape().dims(), &[2, 3]);
assert_eq!(result.to_vec_f64(), vec![11.0, 21.0, 31.0, 21.0, 41.0, 61.0]);
assert_eq!(session.realize(&gradient)?.to_vec_f64(), vec![60.0, 60.0]);
println!("{}", session.trace(&output)?);
let summary = session.execution_summary(&output, true)?;
assert_eq!(summary.requested_outputs[0].shape.dims(), &[2, 3]);
println!("{} logical peak bytes", summary.peak_logical_bytes);
# Ok::<(), rustgrad::Error>(())
```

See the usability-first [product priorities](docs/PRIORITIES.md),
[architecture](docs/ARCHITECTURE.md), and the [tinygrad compatibility
map](docs/COMPATIBILITY.md).

## Strict static Metal elementwise realization

On macOS, a caller can explicitly load and retain a `MetalRuntime`/device and
run the narrow static `CpuSession::realize_metal` route. It has no CPU or
interpreter fallback: the complete schedule is rendered and validated before
the route creates a queue, compiles a pipeline, allocates buffers, or submits
a command. The result is detached and includes a handle-free `MetalSessionTrace`
with ordered compiled cache keys, device capabilities, and a logical identity.
Reusing the same caller-owned device reuses its pipeline cache.

Run `cargo run --example metal_session_infer` on a Metal-capable macOS host.
The supported public subset is static F32/Bool/I32/U32 elementwise/select/cast
and checked affine reads only. Reductions, matmul/Linear/ONNX/model inference,
ReLU and other graph unary ops, dynamic or symbolic shapes, F16/BF16/F64/I64/U64,
effects, aliases, graph capture, profiling, and device-resident state fail
closed. A zero requested output still fully preflights the schedule, then
returns an exact empty detached tensor without pipeline-cache growth or command
submission.

`MetalRuntime::discover` separates framework/symbol loading errors from a
typed `MetalDiscovery::NoDevices` result. The maintained live smoke requires a
process-visible device; the current release-host probe can report `NoDevices`
even when `system_profiler` reports Metal-capable hardware, so the mock suite
is the only stable evidence in that environment.

## Common CPU session operations

The same session delegates ordinary static model expressions to its underlying
Graph. Subtraction, division, ReLU, matmul, stable signed-axis softmax,
argmax, checked permutation/transpose, shrink/signed slice, concat, and
integer gather are available without weakening the session ownership boundary.

```rust
use rustgrad::CpuSession;

let mut session = CpuSession::new();
let input = session.variable([2, 2], [1.0, 2.0, -1.0, 1.0])?;
let weights = session.tensor([2, 3], [1.0, 0.0, -1.0, 0.0, 1.0, 1.0])?;
let zero = session.tensor([1], [0.0])?;
let one = session.tensor([1], [1.0])?;
let logits = session.matmul(&input, &weights)?;
let shifted = session.sub(&logits, &zero)?;
let scaled = session.div(&shifted, &one)?;
let activated = session.relu(&scaled)?;
let probabilities = session.softmax(&activated, -1)?;
let class = session.argmax(&probabilities, -1)?;
assert_eq!(session.realize(&class)?.to_vec_f64(), vec![1.0, 2.0]);
# Ok::<(), rustgrad::Error>(())
```

These are static CPU Graph operations, not a general eager/device API. Dynamic
cardinality, accelerator session execution, and additional convenience wrappers
remain separate boundaries.

At the lower-level static `Graph` boundary, `split`/`chunk` create checked
`Shrink` views; `var`, `var_mean`, `std`, and `std_mean` compose existing
reductions; and `ones_with_dtype`, `const_like`, `rand_like_implicit`, and
`randn_like_implicit` reuse the existing typed creation and Threefry contracts.
F16/BF16 sums accumulate and return F32. Static einsum accepts presentation
whitespace, while higher-order static indexing retains its normalized index
map. CPU `maximum`/`minimum` retain tinygrad's left operand on unordered or
tied float lanes; `softplus`, `mish`, and `logsigmoid` use stable finite-tail
forms; and boolean `any`/`all` retain their distinct empty identities. These
are not dynamic-shape, device, or generic eager conveniences.

## Move local arrays and weights through a CPU session

The bounded copy-based NPY file API is the practical route for local dense
arrays. It accepts only the documented little-endian primitive NPY v1/v2
descriptors and preserves raw float bits; it does not expose pointers, map a
file, or silently cast a dtype. Safetensors files provide the matching static
named-weight route.

```rust
use rustgrad::{CpuSession, load_safetensors_file};
use rustgrad::interop::host::{load_npy_file, save_npy_file};

let input = load_npy_file("input.npy")?;
let (weights, _) = load_safetensors_file("weights.safetensors")?;
let mut session = CpuSession::new();
let input = session.variable_data(input)?;
let weight = session.constant(weights["weight"].clone())?;
let output = session.mul(&input, &weight)?;
save_npy_file("result.npy", &session.realize(&output)?)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`load_npy_file_with_limits` accepts explicit file, header, rank, and element
limits for tighter local-input budgets. `load_safetensors_file_with_limits`
likewise bounds a local safetensors read before canonical parsing.
`save_npy_file` stages and syncs a
same-directory temporary before replacing the target. Unsupported NPY object,
string, structured, BF16, float8, and non-little-endian descriptors fail with
typed errors; device storage, NumPy/DLPack bindings, mmap, and zero-copy
compute are not part of this route.

## Train, checkpoint, resume, and evaluate on CPU

Run `cargo run --example cpu_train_resume` for a dependency-free, deterministic
classification workflow. It uses fresh `Graph` instances for every step,
`Graph::grad`, `Optimizer`, `LearningRateScheduler`, deterministic `BatchIter`
ordering (including a final partial batch), and `PortableTrainingCheckpoint`.
The example captures after a training prefix, restores into freshly constructed
module/optimizer/scheduler identities, then evaluates without mutating any
training state. It is a small local CPU contract, not downloaded-MNIST or
accelerator training support. See `tests/cpu_train_resume.rs` for the exact
resume and non-mutation assertions.

## Train a static module without raw Graph plumbing

Run `cargo run --example cpu_module_train` for the next step after
`CpuSession` inference. `CpuModuleTrainer` accepts a static `ModuleForward`
module, including a configured `Sequential` of `Linear`, state-free `ReLU`,
`Embedding`, `Dropout`, `Conv2d`, `AdaptiveAvgPool2d`, `MaxPool2d`, or
`Flatten` entries,
an existing `Optimizer` and scheduler, plus typed F32 inputs and integer class
targets. Every `train_step` or `evaluate` builds
and discards a fresh CPU graph: parameter leaves capture current versions,
loss/logits/gradients are inspected through the CPU oracle, and only a
successful step advances the existing optimizer and scheduler. Results expose
loss, logits, trace, versions, optimizer step, and scheduler epoch.

Static setup also needs no construction graph or handwritten parameter map:
compose `Linear::new_static(...)`, `ReLU::new()`, and another `Linear` in a
`Sequential`, then use `Optimizer::sgd_for_module(&model, config)`.
The optimizer consumes the module's deterministic trainable traversal, so
nested names and tied parameters remain aligned with strict state loading.
The legacy `Linear::new(&mut Graph, ...)` remains available for existing code
and produces the same seeded host state.

`Sequential` composes its typed entries in insertion order and retains
deterministic nested state names such as `0.weight` and `2.bias`; `ReLU` is
state-free and therefore owns no `1.*` state keys. `Conv2d`,
`AdaptiveAvgPool2d`, and `Flatten::new(start_dim)` also compose for the
verified static CIFAR chain. Other pooling, normalization, stateful, and
multi-input modules remain
explicit rather than being guessed or dispatched by module name.

The bridge is deliberately not a generic trainer or data loader. It supports
one-input static F32 sparse cross-entropy classification, first-order CPU
gradients, and metric-free schedulers. Device/JIT/dynamic/mixed-precision
training and metric-driven scheduler steps remain explicit boundaries. Use the
existing `PortableTrainingCheckpoint` directly for fresh-identity resume.

## Strictly load local module weights for CPU inference

`Module::load_safetensors_file_strict` is the single exact state-loading
boundary for an existing module: it accepts only the module traversal's exact
keys, shapes, and dtypes, validates the full map before the existing all-lock
restore transaction, and leaves the module unchanged on mismatch. It reads
bounded owned bytes from a local safetensors file; it does not guess key names,
cast values, execute pickle/Python, or select a device.

```rust,no_run
use rustgrad::{CapturedReplayExecutor, Module, TensorData, infer_module_cpu, infer_module_native_cpu_with_report};
use rustgrad::nn::Linear;
use std::path::Path;

let model = Linear::new_static(2, 1, true, 7)?;
model.load_safetensors_file_strict(Path::new("linear.safetensors"))?;
let result = infer_module_cpu(&model, TensorData::new([2, 2], vec![1., 2., 3., 4.])?)?;
let executor = CapturedReplayExecutor::default();
let native = infer_module_native_cpu_with_report(&model, TensorData::new([2, 2], vec![1., 2., 3., 4.])?, &executor, false)?;
assert_eq!(result.output(), native.inference().output());
println!("native items={}, logical peak={} bytes", native.report().native_item_count, native.report().execution_plan.peak_logical_bytes);
# let _ = result.output();
# Ok::<(), Box<dyn std::error::Error>>(())
```

Run `cargo run --example strict_state_inference` for a self-contained
deterministic local safetensors fixture and known `Linear` output. The narrow
workflow is CPU/static only; non-strict casts, heuristic key remapping,
architecture inference, device loading, and Python/Torch execution remain
separate boundaries. `infer_module_cpu` builds and discards one fresh graph on
each call, returning a detached output with a deterministic trace and canonical
parameter-version map; it never mutates the module or caller input.
`infer_module_native_cpu` is an explicit, strict no-fallback CPU-JIT opt-in
for the verified static F32 single-input/single-output `ModuleForward` subset.
The caller owns `CapturedReplayExecutor` and its cache; the detached result
exposes a logical native trace and cache keys without runtime resource IDs.
Adapter-only empty-domain pruning avoids native compilation for dead pure work.
Verified public compositions include `Linear` and
`Sequential[Linear, ReLU, Linear]`: the latter retains canonical `0.*`/`2.*`
parameter names and exact CPU parity under scalar and vector native policies.
The released two-class configured CIFAR composition
`Sequential[Conv2d(3→2, 1×1, groups=1), ReLU, AdaptiveAvgPool2d(1,1),
Flatten(1), Linear(2→2)]` is also verified under the same opt-in route. Its
native Conv boundary is static F32 NCHW/OIHW only (unit stride/dilation, zero
padding, optional F32 bias); positive injective computed affine views are
materialized into owned dense buffers before existing reduction/matmul plans.
Broader Conv geometry, signed/broadcast/overlapping views, devices, dynamic
shapes, mixed precision, training, and general replay pruning remain outside
this route.

`infer_module_native_cpu_with_report` is a separate opt-in observation route
used by `examples/strict_state_inference.rs`. It returns the same detached
strict-native result plus immutable static-plan/cache facts and current-call
wall-clock durations for graph/schedule/capture construction, native
preparation, and detached native execution. Those durations are nondeterministic
local observations, not benchmark, throughput, RSS, allocator, device-memory,
or per-kernel measurements; they never enter the report identity or cache key.

## Run a supported local GGUF Llama prompt

```text
cargo run --example llama_prompt -- path/to/model.gguf "hello" 16
cargo run --example llama_prompt -- --native path/to/model.gguf "hello" 16
```

This local CPU-only route validates the GGUF, fixed Llama schema, tokenizer,
and exact supported chat template before deterministic greedy generation. The
final argument bounds new tokens; prompt-plus-generation context is checked
before graph execution and EOS/EOT stops early. There is no network download,
device/model fallback, arbitrary Jinja template, or implicit sampling. Dense
and audited packed CPU projections follow the existing Llama model contract;
unsupported files, schemas, layouts, and templates return typed errors.

Pass `--native` to opt into the separate strict-native replay path. It uses the
same validated model, tokenizer, exact chat template, and deterministic greedy
policy, but never falls back to CPU if native compilation or execution rejects.
The returned native generation carries only detached token/text data and a
resource-free strict-native stage trace. This is one stateless request at a
time: it does not claim a native conversation cache, serving API, device
execution, sampling, dynamic shapes, or general Llama compatibility.

## Stateful local Llama chat

Run two bounded local turns with:

```text
cargo run --example llama_chat -- path/to/model.gguf "hello" "tell me more" 16
```

Or create `let mut chat = workflow.conversation();` from a validated
`LlamaPromptWorkflow`, then call `chat.send("hello", 16)?` for each user turn.
The conversation owns only committed user/assistant history and the released
transactional CPU generator cache; `history()`, `cache_len()`, and `reset()`
are inspectable. Greedy decoding is deliberate: no implicit RNG or sampling
state is introduced. Failed empty/template/context/model requests leave history
and cache reusable and unchanged; conversations never share a cache.

## Load local MNIST IDX files

Use `cargo run --example mnist_idx_local -- train-images.idx3-ubyte
train-labels.idx1-ubyte` to train then evaluate a small CPU classifier from a
local uncompressed IDX pair. It normalizes the owned images once, materializes
seeded `BatchIter` rows (including the final partial batch), then uses
graph-free `Linear::new_static`, `Optimizer::sgd_for_module`, and
`CpuModuleTrainer`; no user `Graph`, `NodeId`, bindings, or parameter-name map
is required. `materialize_classification_batch` is also public for normalized
MNIST or NCHW CIFAR tensors, preserving raw dense rows in caller order. The
public IDX workflow tests cover portable fresh-identity resume and non-mutating
evaluation. This does not download, cache, augment, or claim benchmark MNIST
accuracy.

`summarize_classification(result.logits(), &targets)` provides deterministic
first-tie predictions, correct/total counts, and `Some(accuracy)` for static
rank-two F32 logits; an empty batch has zero counts and `None` accuracy.

## Load local CIFAR-10 binary batches

Use `cargo run --example cifar10_local -- data_batch_1.bin data_batch_2.bin`
to validate one or more local uncompressed CIFAR-10 binary batches in the
provided order. Each record is one class label plus 3072 channel-major bytes;
the loader returns U8 NCHW `[N, 3, 32, 32]` images and U8 labels, with explicit
file, total-byte, file-count, and record-count limits. The public
`examples/cifar10_local.rs` builds the graph-free configured
Conv2d → ReLU → AdaptiveAvgPool2d → Flatten → Linear route, then trains and
evaluates it through `CpuModuleTrainer`. Public acceptance covers deterministic
nested state/inference, partial and empty batches, non-mutating evaluation, and
the existing local checkpoint coverage. It has no downloader, archive/cache
handling, augmentation, device training, concurrency, or CIFAR accuracy claim.

## Run a bounded local ONNX model with NPY files

```text
cargo run --example onnx_npy_infer -- model.onnx x=input.npy --output y=output.npy
cargo run --example onnx_npy_infer -- model.onnx x=input.npy z=offset.npy --native --output a=linear.npy y=relu.npy
```

The route imports only the documented static default-domain opset-13 subset,
requires exact named input shapes and dtypes before CPU execution, and writes
selected named outputs through the canonical staged NPY writer. It never fetches
models, loads external data, guesses names, or converts dtypes. `--native` is
an explicit strict no-fallback CPU-JIT route for fixed static-F32
`MatMul → Add → ReLU` models; it uses one caller-owned example executor and
prints deterministic cache keys. The bounded library native-many API accepts fixed F32
named inputs and deterministic selected outputs through one `schedule_many`
capture and caller-owned strict NativeJit replay. Its file adapter stages
same-directory NPY replacements and rolls earlier targets back on a later
replacement failure; this is fail-atomic rollback, not simultaneous multi-path
filesystem atomicity. It rejects unsupported operations before
compilation or output staging. It does not claim dynamic/empty input schemas,
general ONNX native execution, devices, or timing results.

## Load a restricted local PyTorch state dictionary

```text
cargo run --example torch_linear_infer -- linear.pt 1.0 2.0
```

This constructs the documented `Linear(2, 1)` configuration, strictly loads a
local protocol-2 CPU-dense `torch.save(state_dict)` ZIP subset, then runs the
existing CPU graph. Keys, shapes, dtypes, and aliases must match exactly before
any parameter changes. It does not run Python/pickle code, import modules,
guess a model, convert weights, load device/sparse/quantized storage, or fetch
files.
