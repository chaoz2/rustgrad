# RustGrad

RustGrad is an inspectable, differentially validated tensor compiler written in Rust. The long-term target is feature parity with tinygrad while using an architecture natural to Rust.

The first milestone contains a shape-checked graph IR, a thin backend trait, a deliberately simple CPU reference evaluator, and a human-readable compile trace.

```rust
use rustgrad::{Backend, CpuBackend, Graph, TensorData};
use std::collections::HashMap;

let mut graph = Graph::new();
let x = graph.input("x", [2]);
let two = graph.constant(TensorData::new([2], vec![2.0, 2.0])?);
let output = graph.mul(x, two)?;
let inputs = HashMap::from([("x".into(), TensorData::new([2], vec![3.0, 4.0])?)]);

assert_eq!(CpuBackend.execute(&graph, output, &inputs)?.values(), &[6.0, 8.0]);
println!("{}", graph.trace(output)?);
# Ok::<(), rustgrad::Error>(())
```

See [architecture](docs/ARCHITECTURE.md) and the [tinygrad compatibility map](docs/COMPATIBILITY.md).

