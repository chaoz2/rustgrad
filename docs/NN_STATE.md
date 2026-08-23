# NN state ownership

`nn::Parameter` is a graph-specific input leaf plus an `Arc<RwLock<...>>`
host value. It is intentionally not mutable graph storage. A module supplies
its current values with fallible `Module::input_bindings()` when executing a graph.

This keeps the graph immutable and traceable: replacing a parameter checks its
shape and dtype, optionally checks an expected version, increments a version,
and changes only a later execution's
input binding. Previously built graph topology and `NodeId`s stay valid.
Parameters from another `Graph` are rejected.

`Parameter::snapshot()` takes one read lock and returns a cloned immutable value,
shape, dtype, version, stable allocation identity, and input metadata. All reads
used for traversal, serialization, graph execution bindings, and optimizer math
must be snapshots obtained before any write. Tied parameters are sorted and
canonicalized by stable identity; updates acquire at most one write lock at a
time. Lock poisoning is never recovered: every affected public API returns
`Error::ParameterLockPoisoned` with operation context.

Modules implement explicit deterministic `visit` traversal; nested containers
call it with dot-separated paths. Repeated parameter handles are identified by
their shared allocation, emitted once at the first path, and loaded once, so
tied parameters cannot diverge. Buffers use the same storage mechanism with
`trainable = false`. `nn::StateDict` converts to/from the safetensors map.

## Optimizer lifecycle

`optim::Optimizer` accepts explicit evaluated `Gradient` values, each stamped
with the parameter version at evaluation time. `step` rejects a stale gradient
or an externally replaced parameter, then updates through `Parameter::replace`.
Consequently a training loop is: build graph, evaluate scalar-loss gradients
with current bindings, wrap them with fallible `Gradient::for_parameter`, step, then
build/evaluate the next graph cycle. Optimizer slots accumulate in f64 and are
saved under deterministic `optimizer.<parameter-name>.*` keys in `StateDict`.

## Stateful normalization lifecycle

`Mode::Training` and `Mode::Eval` are explicit inputs to BatchNorm forwards;
there is no global mutable training flag. Training BatchNorm returns a
`BatchNormOutput` with graph output plus a non-cloneable `PendingBatchNormStats`
token when running statistics are enabled. The caller must follow this order:

1. snapshot/build the graph and collect input bindings;
2. execute the output and the token's `mean` and `variance` nodes;
3. call `token.commit_stats(&module, mean, variance)`;
4. rebuild subsequent executions because buffer versions have changed.

The token is one-shot and binds the module, running-buffer identities and their
snapshot versions. Commits reject stale, duplicate, wrong-module, or malformed
statistics. No lock is held while executing graph nodes. Running variance uses
the batch's unbiased correction, while the forward normalization uses biased
variance, matching tinygrad. `track_running_stats=false` uses batch statistics
in both modes. GroupNorm and InstanceNorm are stateless and use the same
explicit parameter/binding traversal as other modules.
