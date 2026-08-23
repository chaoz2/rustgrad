# NN state ownership

`nn::Parameter` is a graph-specific input leaf plus an `Rc<RefCell<...>>`
host value. It is intentionally not mutable graph storage. A module supplies
its current values with `Module::input_bindings()` when executing a graph.

This keeps the graph immutable and traceable: replacing a parameter checks its
shape and dtype, increments a version, and changes only a later execution's
input binding. Previously built graph topology and `NodeId`s stay valid.
Parameters from another `Graph` are rejected.

Modules implement explicit deterministic `visit` traversal; nested containers
call it with dot-separated paths. Repeated parameter handles are identified by
their shared allocation, emitted once at the first path, and loaded once, so
tied parameters cannot diverge. Buffers use the same storage mechanism with
`trainable = false`. `nn::StateDict` converts to/from the safetensors map.
