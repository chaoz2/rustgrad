# RustGrad

RustGrad is an inspectable tensor compiler and machine-learning runtime written
in Rust. It follows tinygrad's small, explicit compiler model while using Rust
ownership, typed errors, and fail-closed backend boundaries to keep execution
predictable.

The project is built around a few principles:

- tensor programs remain inspectable from graph construction through execution;
- the CPU implementation is the semantic oracle for optimized backends;
- unsupported operations and devices return typed errors instead of silently
  falling back;
- scheduling, capture, replay, memory planning, and device ownership are explicit;
- correctness claims are backed by differential tests, fuzzing, and CI.

RustGrad is under active development. The supported surface is intentionally
bounded and documented rather than implied.

## A small tensor program

`CpuSession` is the simplest public entry point. It owns one graph and its input
bindings, so ordinary tensor programs do not need manual backend plumbing.

```rust
use rustgrad::CpuSession;

let mut session = CpuSession::new();
let input = session.variable([2, 1], [1.0, 2.0])?;
let scale = session.tensor([3], [10.0, 20.0, 30.0])?;
let bias = session.tensor([3], [1.0, 1.0, 1.0])?;

let product = session.mul(&input, &scale)?;
let output = session.add(&product, &bias)?;
let loss = session.sum_all(&output)?;
let gradient = session.grad(&loss, &input)?;

assert_eq!(
    session.realize(&output)?.to_vec_f64(),
    vec![11.0, 21.0, 31.0, 21.0, 41.0, 61.0],
);
assert_eq!(session.realize(&gradient)?.to_vec_f64(), vec![60.0, 60.0]);
# Ok::<(), rustgrad::Error>(())
```

The same session exposes common static CPU operations for model arithmetic,
activations, reductions, movement, indexing, and first-order gradients.

## Build and run a module

RustGrad modules own deterministic parameter state independently of any one
graph. Static modules can be initialized, loaded from supported local state,
and executed through a fresh CPU graph.

```rust,no_run
use rustgrad::{Module, TensorData, infer_module_cpu};
use rustgrad::nn::Linear;
use std::path::Path;

let model = Linear::new_static(2, 1, true, 7)?;
model.load_safetensors_file_strict(Path::new("linear.safetensors"))?;

let input = TensorData::new([2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
let result = infer_module_cpu(&model, input)?;
println!("{:?}", result.output());
# Ok::<(), Box<dyn std::error::Error>>(())
```

The maintained examples cover local training and resume, module inference,
MNIST and CIFAR data, static ONNX with NPY files, and bounded GGUF Llama prompt
and chat workflows. They are source examples rather than separate framework
layers:

- [`examples/cpu_train_resume.rs`](examples/cpu_train_resume.rs)
- [`examples/cpu_module_train.rs`](examples/cpu_module_train.rs)
- [`examples/strict_state_inference.rs`](examples/strict_state_inference.rs)
- [`examples/mnist_idx_local.rs`](examples/mnist_idx_local.rs)
- [`examples/cifar10_local.rs`](examples/cifar10_local.rs)
- [`examples/onnx_npy_infer.rs`](examples/onnx_npy_infer.rs)
- [`examples/llama_prompt.rs`](examples/llama_prompt.rs)
- [`examples/llama_chat.rs`](examples/llama_chat.rs)
- [`examples/metal_scoreboard.rs`](examples/metal_scoreboard.rs)

## Run ResNet on a persistent Metal session

The typed ResNet facade builds and captures the complete Eval/F32 graph, freezes
its parameters, and binds the plan to one explicitly selected Metal device.
Preparation uploads residents once; repeated runs stage only the image and
download logits. The graph, capture, memory plan, input schemas, rendered MSL,
and reports remain inspectable, and unsupported work returns an error instead
of using CPU fallback.

```rust,no_run
use rustgrad::nn::{ResNet, ResNetConfig, ResNetMetalPlan};
use rustgrad::runtime::metal::{MetalPlanOptions, MetalRuntime};
use rustgrad::TensorData;

let device = MetalRuntime::load()?.device(0)?;
let model = ResNet::new_static(ResNetConfig::default(), 7)?;
let plan = ResNetMetalPlan::eval_f32(
    &model,
    &device,
    [1, 3, 224, 224],
    MetalPlanOptions::default(),
)?;
assert_eq!(plan.summary().fallback_count, 0);
println!("kernels: {}", plan.rendered_items().count());

let mut session = plan.prepare()?;
let image = TensorData::zeros([1, 3, 224, 224])?;
let first = session.run(image.clone())?;
let second = session.run(image)?;
assert_eq!(first.logits().shape().dims(), &[1, 1000]);
println!("steady run: {:?}", second.report());
# Ok::<(), Box<dyn std::error::Error>>(())
```

The lower-level session and opt-in scoreboard remain available for detailed
deployment evidence. Their host wall-clock and host API copy counts do not
claim GPU time, physical bus traffic, allocator RSS, energy, or throughput.

## How the system fits together

1. `tensor` owns concrete dtypes, shapes, scalars, and dense values.
2. `ir` builds validated lazy graphs and `autograd` transforms them.
3. `schedule`, `uop`, and the memory planners lower graphs into executable work.
4. the CPU oracle establishes semantics; native and device backends must match it.
5. capture and replay retain typed, resource-free execution descriptions.
6. `nn`, importers, datasets, and model workflows compose those foundations.

This retains tinygrad's inspectable graph-to-kernel path without copying its
Python API mechanically. Rust ecosystem projects inform different design
choices: Luminal's small compiler vocabulary, Burn's explicit backend
composition, dfdx's useful type-level invariants, Candle's deployable Rust model
workflows, tract's translate-versus-runtime boundary, ndarray's ownership-aware
data model, tch-rs's LibTorch interoperability baseline, RustTensor's direct
differential reference path, and cuda-oxide's isolated experimental Rust-to-PTX
direction.

## Scope and priorities

Work is ordered by user value:

1. a strict persistent Metal device session with no fallback;
2. ResNet-18 Metal conformance on the Apple M5;
3. device-resident GGUF Llama prefill, KV state, and decode;
4. evidence-labeled performance and release hygiene.

The CPU adoption, training, state, interchange, and module layers are delivered
foundation. Hardware comparisons target tinygrad and Candle, plus llama.cpp for
GGUF, and distinguish compile, first-run, steady-state, planned device memory,
kernel count, host API transfer count/bytes, and fallback count. GPU timing,
allocator RSS, or physical bus traffic is reported only when measured directly.

Tinygrad is the primary semantic reference for tensor and compiler behavior.
Rust projects are design references for API ergonomics, ownership, backend
boundaries, and deployment. A parity item does not outrank an incomplete user
workflow unless it is the demonstrated blocker.

See:

- [Product priorities](docs/PRIORITIES.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Compatibility and evidence](docs/COMPATIBILITY.md)
- [Fuzzing and replay](docs/FUZZING.md)
- [Contributing](CONTRIBUTING.md)

GitHub Actions is the release gate for formatting, compilation, Clippy, the
compatibility manifest, Linux and macOS tests, and sanitizer coverage.

## Project status

RustGrad is not yet a general replacement for tinygrad, PyTorch, Burn, Candle,
or tract. Dynamic shapes, backend breadth, model coverage, and live accelerator
evidence remain deliberately incomplete. The compatibility ledger is the
authoritative record of what is implemented and how each claim is validated.
