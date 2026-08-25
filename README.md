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

## Load local MNIST IDX files

Use `cargo run --example mnist_idx_local -- train-images.idx3-ubyte
train-labels.idx1-ubyte` to validate a local uncompressed IDX pair with
explicit file/count/dimension limits and deterministic batching. The public
`tests/mnist_idx_files_workflow.rs` shows the complete small CPU workflow from
generated local files through Graph/autograd, SGD/scheduler, portable
fresh-identity resume, and non-mutating evaluation. This does not download,
cache, augment, or claim benchmark MNIST accuracy.
