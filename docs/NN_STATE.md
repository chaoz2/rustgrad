# NN state ownership

`nn::Parameter` is graph-independent host state: a stable `ParameterId`, input
metadata, trainable flag, and an `Arc<RwLock<...>>` value/version pair. It owns
no `Graph` identity or `NodeId`. `Parameter::new(data, trainable)` therefore does
not allocate a graph node.

`Parameter::bind(&mut graph)` takes one coherent snapshot and asks the target
graph for a versioned input leaf. The graph-local registry is keyed by stable
parameter identity and captured version. Repeated binds of one version in one
graph return one `NodeId`; a different graph owns a distinct leaf; and a later
host version creates a new leaf rather than reusing the old one. Module forwards
bind every parameter and buffer they consume. `Parameter::node(&graph)` is a
read-only lookup for the current version and rejects an unbound target graph.

`Graph::parameter_bindings()` returns every captured parameter input;
`Module::input_bindings(&graph)` filters that map to the module's identities.
Both include multiple versions when those versions remain in the topology. Replacing
a parameter checks shape and dtype, optionally checks an expected version, and
increments the host version without changing any previously built graph.
Consequently old graph outputs remain replayable with their old values while a
new graph sees the current value. Ordinary input validation remains unchanged.

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
with parameter identity and version. `step` rejects a gradient for another
parameter, a stale gradient, or an externally replaced parameter, then updates
through `Parameter::replace`.
Consequently a training loop is: build graph, evaluate scalar-loss gradients
with current bindings, wrap them with fallible `Gradient::for_parameter`, step, then
build/evaluate the next graph cycle. Optimizer slots accumulate in f64 and are
saved under deterministic `optimizer.<parameter-name>.*` keys in `StateDict`.

## Exact in-process training resume

`TrainingCheckpoint::capture(module, optimizer, scheduler)` packages three
typed parts: module tensors encoded as safetensors, optimizer `StateDict`, and
scheduler `StateDict` (including its `LrSchedulerState` epoch). It also records
the module's stable parameter identities and versions plus the optimizer's
explicit name-to-identity ownership map.

`TrainingCheckpoint::resume` is intentionally an in-process identity-preserving
operation. The original host `Parameter` objects must still hold the captured
versions and exact values; only graphs, optimizer objects, and scheduler objects
are recreated. Resume parses and compares the safetensors payload, identities,
versions, optimizer ownership/config/slots, and scheduler config/state before
applying either mutable state dictionary. A same-name, same-shape module with
new `ParameterId`s is rejected rather than silently attaching old slots. This
also keeps parameter versions monotonic and prevents a restored version from
colliding with an existing graph-local binding. Invalid module, optimizer, or
scheduler parts leave module, optimizer, and scheduler state unchanged.

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
