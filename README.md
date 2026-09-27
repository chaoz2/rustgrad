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

## Compile once, train, and resume

RustGrad's central near-term workflow is a fixed-shape
`forward → loss → backward → AdamW update` program compiled once and
replayed against persistent state. The maintained
[`compiled_transformer_train_resume`](examples/compiled_transformer_train_resume.rs)
example exercises masked causal Transformer training end to end:

- parameters, optimizer moments, gradient windows, and dropout progress survive
  between replays;
- tied and frozen parameters, clipping, accumulation, partial-window commit, and
  reset remain part of the captured lifecycle;
- portable checkpoints restore into an owned module and continue exactly; and
- the interpreter and strict-native CPU paths execute the same program without
  silent fallback.

Strict Metal shares the capture and failure-atomic recurrent-state foundation
for its admitted policy subset. CPU-only policies still fail closed, and live
Apple-hardware training evidence remains an explicit follow-up rather than an
implied claim.

See [Compiled recurrent training](docs/ARCHITECTURE.md#compiled-recurrent-training)
for ownership, resume, and backend boundaries. The
[compatibility ledger](docs/COMPATIBILITY.md) records the executable evidence.

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
- [`examples/compiled_transformer_train_resume.rs`](examples/compiled_transformer_train_resume.rs)
- [`examples/strict_state_inference.rs`](examples/strict_state_inference.rs)
- [`examples/mnist_idx_local.rs`](examples/mnist_idx_local.rs)
- [`examples/cifar10_local.rs`](examples/cifar10_local.rs)
- [`examples/onnx_npy_infer.rs`](examples/onnx_npy_infer.rs)
- [`examples/llama_prompt.rs`](examples/llama_prompt.rs)
- [`examples/llama_chat.rs`](examples/llama_chat.rs)
- [`examples/metal_scoreboard.rs`](examples/metal_scoreboard.rs)

## Bounded accelerator inference

The typed ResNet facade builds and captures the complete Eval/F32 graph, freezes
its parameters, and binds the plan to one explicitly selected Metal device.
Preparation uploads residents once; repeated runs stage only the image and
download logits. Unsupported work returns an error instead of selecting CPU.

```rust,no_run
use rustgrad::nn::{ResNet, ResNetConfig, ResNetMetalPlan};
use rustgrad::runtime::metal::MetalRuntime;
use rustgrad::{MetalSessionTarget, TensorData};

let device = MetalRuntime::load()?.device(0)?;
let target = MetalSessionTarget::new(device, 64)?;
let model = ResNet::new_static(ResNetConfig::default(), 7)?;
let plan = ResNetMetalPlan::eval_f32_on(&model, &target, [1, 3, 224, 224])?;
assert_eq!(plan.summary().fallback_count, 0);
println!("kernels: {}", plan.rendered_items().count());

let mut session = target.prepare(plan)?;
let image = TensorData::zeros([1, 3, 224, 224])?;
let first = session.run(image.clone())?;
let second = session.run(image)?;
assert_eq!(first.logits().shape().dims(), &[1, 1000]);
println!("steady run: {:?}", second.report());
# Ok::<(), Box<dyn std::error::Error>>(())
```

GGUF Llama has a separate typed prompt-to-tokens Metal facade for supported
dense and packed models. It owns the validated model, tokenizer, chat template,
resident weights, and fixed-capacity K/V state for one selected device. The
maintained [`llama_prompt`](examples/llama_prompt.rs),
[`llama_chat`](examples/llama_chat.rs), and
[`metal_llama_generate`](examples/metal_llama_generate.rs) examples show the
supported local workflows.

Both Metal facades keep capture, selected-device ownership, fallback count, and
execution evidence inspectable. Timing fields are observations, not claims
about end-to-end throughput, physical transfers, memory use, energy, or
speedup. Protected coverage is semantic unless a provisioned live lane says
otherwise; no live Apple-GPU comparison is currently published. See the
[Metal runtime boundary](docs/ARCHITECTURE.md#metal-runtime-boundary) and the
[protected live Metal lane guide](docs/METAL_LIVE.md) for the precise contracts.

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

1. one compiled persistent-state training runtime;
2. exact tiny-Transformer train/resume across CPU and strict Metal;
3. live-hardware evidence for the existing ResNet and GGUF Llama Metal paths;
4. evidence-labeled performance and release hygiene.

The CPU adoption, training, state, interchange, and module layers are delivered
foundation. Hardware comparisons target tinygrad and Candle, plus llama.cpp for
GGUF, and distinguish compile, first-run, steady-state, planned device memory,
kernel count, host API transfer count/bytes, and fallback count. GPU timing,
allocator RSS, or physical bus traffic is reported only when measured directly.
Typed observations can now be bundled into deterministic offline comparisons
under one exact workload, device, and explicit baseline; see
[Benchmarking](docs/BENCHMARKING.md). No live Apple-GPU comparison measurements
are currently published.

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
