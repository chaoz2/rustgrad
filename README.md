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
# Ok::<(), rustgrad::Error>(())
```

See the usability-first [product priorities](docs/PRIORITIES.md),
[architecture](docs/ARCHITECTURE.md), and the [tinygrad compatibility
map](docs/COMPATIBILITY.md).

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
limits for tighter local-input budgets. `save_npy_file` stages and syncs a
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
module, including a configured `Sequential` of supported single-input modules,
an existing `Optimizer` and scheduler, plus typed F32 inputs and integer class
targets. Every `train_step` or `evaluate` builds
and discards a fresh CPU graph: parameter leaves capture current versions,
loss/logits/gradients are inspected through the CPU oracle, and only a
successful step advances the existing optimizer and scheduler. Results expose
loss, logits, trace, versions, optimizer step, and scheduler epoch.

Static setup also needs no construction graph or handwritten parameter map:
use `Linear::new_static(...)` and `Optimizer::sgd_for_module(&model, config)`.
The optimizer consumes the module's deterministic trainable traversal, so
nested names and tied parameters remain aligned with strict state loading.
The legacy `Linear::new(&mut Graph, ...)` remains available for existing code
and produces the same seeded host state.

`Sequential` composes its typed entries in insertion order and retains
deterministic nested state names such as `0.weight`. State-only, multi-input,
and training-mode-dependent modules remain explicit rather than being guessed
or dispatched by module name.

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
use rustgrad::{Backend, CpuBackend, Graph, Module, TensorData};
use rustgrad::nn::Linear;
use std::path::Path;

let model = Linear::new(&mut Graph::new(), 2, 1, true, 7)?;
model.load_safetensors_file_strict(Path::new("linear.safetensors"))?;
let mut graph = Graph::new();
let input = graph.input("input", [2, 2]);
let output = model.forward(&mut graph, input)?;
let mut bindings = model.input_bindings(&graph)?;
bindings.insert("input".into(), TensorData::new([2, 2], vec![1., 2., 3., 4.])?);
let result = CpuBackend.execute(&graph, output, &bindings)?;
# let _ = result;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Run `cargo run --example strict_state_inference` for a self-contained
deterministic local safetensors fixture and known `Linear` output. The narrow
workflow is CPU/static only; non-strict casts, heuristic key remapping,
architecture inference, device loading, and Python/Torch execution remain
separate boundaries.

## Run a supported local GGUF Llama prompt

```text
cargo run --example llama_prompt -- path/to/model.gguf "hello" 16
```

This local CPU-only route validates the GGUF, fixed Llama schema, tokenizer,
and exact supported chat template before deterministic greedy generation. The
final argument bounds new tokens; prompt-plus-generation context is checked
before graph execution and EOS/EOT stops early. There is no network download,
device/model fallback, arbitrary Jinja template, or implicit sampling. Dense
and audited packed CPU projections follow the existing Llama model contract;
unsupported files, schemas, layouts, and templates return typed errors.

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
`tests/cifar_files_workflow.rs` demonstrates the bounded CPU Conv2d → adaptive
pool → Linear train/checkpoint/fresh-identity-resume/evaluate path over generated
local batch files. It has no downloader, archive/cache handling, augmentation,
device training, concurrency, or CIFAR accuracy claim.

## Run a bounded local ONNX model with NPY files

```text
cargo run --example onnx_npy_infer -- model.onnx x=input.npy --output y=output.npy
```

The route imports only the documented static default-domain opset-13 subset,
requires exact named input shapes and dtypes before CPU execution, and writes
selected named outputs through the canonical staged NPY writer. It never fetches
models, loads external data, guesses names, converts dtypes, or falls back to
JIT/device execution.

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
