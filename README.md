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
