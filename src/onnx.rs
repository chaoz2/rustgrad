//! Bounded static ONNX protobuf import. This intentionally supports a small,
//! audited inference subset (opset 13, default domain) and never executes code.

use crate::{
    Backend, Conv2dOptions, CpuBackend, DType, Error, Graph, NodeId, PoolOptions, ReduceKind,
    Result, Scalar, Shape, TensorData,
};
use std::collections::{BTreeMap, HashMap};

const MAX_BYTES: usize = 32 * 1024 * 1024;
fn bad(s: impl Into<String>) -> Error {
    Error::ModelIo { reason: s.into() }
}

/// A static ONNX graph lowered into RustGrad's existing CPU graph boundary.
pub struct OnnxModel {
    graph: Graph,
    inputs: BTreeMap<String, NodeId>,
    outputs: BTreeMap<String, NodeId>,
}
impl OnnxModel {
    pub fn inputs(&self) -> impl Iterator<Item = &str> {
        self.inputs.keys().map(String::as_str)
    }
    pub fn outputs(&self) -> impl Iterator<Item = &str> {
        self.outputs.keys().map(String::as_str)
    }
    pub fn run(&self, inputs: HashMap<String, TensorData>) -> Result<BTreeMap<String, TensorData>> {
        let cpu = CpuBackend;
        self.outputs
            .iter()
            .map(|(name, &node)| Ok((name.clone(), cpu.execute(&self.graph, node, &inputs)?)))
            .collect()
    }
}

/// Parses and lowers a static opset-13 ONNX MLP subset: Identity, Add,
/// MatMul, and Relu, with constant initializers and named concrete inputs.
pub fn import_onnx(bytes: &[u8]) -> Result<OnnxModel> {
    if bytes.len() > MAX_BYTES {
        return Err(bad("ONNX model exceeds byte limit"));
    }
    let m = Msg::new(bytes);
    let graph_bytes = one_bytes(&m, 7, "graph")?;
    let opsets = m.bytes(8)?;
    if opsets.len() != 1 {
        return Err(bad("ONNX requires exactly one opset import"));
    }
    let op = Msg::new(opsets[0]);
    if !op.string(1)?.unwrap_or("").is_empty() || one_varint(&op, 2, "opset version")? != 13 {
        return Err(bad("only default-domain ONNX opset 13 is supported"));
    }
    let g = Msg::new(graph_bytes);
    let mut graph = Graph::new();
    let mut constants = BTreeMap::new();
    let mut nodes = BTreeMap::new();
    let mut initializers = BTreeMap::new();
    for t in g.bytes(5)? {
        let (name, data) = tensor(Msg::new(t))?;
        constants.insert(name.clone(), data.clone());
        if initializers
            .insert(name.clone(), graph.constant(data))
            .is_some()
        {
            return Err(bad("duplicate ONNX initializer"));
        }
    }
    let mut inputs = BTreeMap::new();
    for v in g.bytes(11)? {
        let (name, shape, dtype) = value_info(Msg::new(v))?;
        if initializers.contains_key(&name) {
            continue;
        }
        if inputs
            .insert(name.clone(), graph.input_dtype(name.clone(), shape, dtype))
            .is_some()
        {
            return Err(bad("duplicate ONNX input"));
        }
    }
    nodes.extend(initializers.iter().map(|(k, &v)| (k.clone(), v)));
    nodes.extend(inputs.iter().map(|(k, &v)| (k.clone(), v)));
    for n in g.bytes(1)? {
        lower(&mut graph, Msg::new(n), &mut nodes, &mut constants)?;
    }
    let mut outputs = BTreeMap::new();
    for v in g.bytes(12)? {
        let (name, _, _) = value_info(Msg::new(v))?;
        let node = *nodes
            .get(&name)
            .ok_or_else(|| bad(format!("unknown ONNX output {name:?}")))?;
        if outputs.insert(name, node).is_some() {
            return Err(bad("duplicate ONNX output"));
        }
    }
    if outputs.is_empty() {
        return Err(bad("ONNX graph has no outputs"));
    }
    Ok(OnnxModel {
        graph,
        inputs,
        outputs,
    })
}

fn lower(
    g: &mut Graph,
    n: Msg<'_>,
    values: &mut BTreeMap<String, NodeId>,
    constants: &mut BTreeMap<String, TensorData>,
) -> Result<()> {
    if !n.string(7)?.unwrap_or("").is_empty() {
        return Err(bad("ONNX custom domains and attributes are unsupported"));
    }
    let op = n.string(4)?.ok_or_else(|| bad("ONNX node lacks op_type"))?;
    let ins = n.strings(1)?;
    let outs = n.strings(2)?;
    if op == "MaxPool" && outs.len() == 2 {
        return Err(bad("MaxPool indices output is unsupported"));
    }
    if outs.len() != 1 || outs[0].is_empty() || values.contains_key(outs[0]) {
        return Err(bad("invalid or duplicate ONNX node output"));
    }
    let get = |i: usize| -> Result<NodeId> {
        ins.get(i)
            .and_then(|x| values.get(*x))
            .copied()
            .ok_or_else(|| bad("missing ONNX node input"))
    };
    let attrs = attrs(&n)?;
    let out = match op {
        "Identity" if ins.len() == 1 => get(0)?,
        "Relu" if ins.len() == 1 => g.relu(get(0)?)?,
        "Sigmoid" if ins.len() == 1 => g.sigmoid(get(0)?)?,
        "Tanh" if ins.len() == 1 => g.tanh(get(0)?)?,
        "Add" if ins.len() == 2 => g.add(get(0)?, get(1)?)?,
        "Sub" if ins.len() == 2 => g.sub(get(0)?, get(1)?)?,
        "Mul" if ins.len() == 2 => g.mul(get(0)?, get(1)?)?,
        "Div" if ins.len() == 2 => g.div(get(0)?, get(1)?)?,
        "MatMul" if ins.len() == 2 => g.matmul(get(0)?, get(1)?)?,
        "Cast" if ins.len() == 1 && attrs.len() == 1 => {
            let x = attrs.get("to").ok_or_else(|| bad("Cast needs to"))?;
            let mut at = 0;
            g.cast(get(0)?, onnx_dtype(var(x, &mut at)?)?)?
        }
        "Constant" if ins.is_empty() && attrs.len() == 1 => {
            let data = tensor_data(Msg::new(
                attrs
                    .get("value")
                    .ok_or_else(|| bad("Constant needs value"))?,
            ))?;
            constants.insert(outs[0].to_owned(), data.clone());
            g.constant(data)
        }
        "Reshape" if ins.len() == 2 => {
            let shape = const_i64(constants, ins[1])?;
            let source = g.shape(get(0)?)?.dims().to_vec();
            g.reshape(get(0)?, reshape_dims(&source, &shape)?)?
        }
        "Transpose" if ins.len() == 1 => {
            let rank = g.shape(get(0)?)?.rank();
            let axes = attrs
                .get("perm")
                .map(|x| packed_i64(x))
                .transpose()?
                .unwrap_or_else(|| (0..rank).rev().map(|x| x as i64).collect());
            g.permute(get(0)?, axes_usize(&axes, rank)?)?
        }
        "Flatten" if ins.len() == 1 => {
            let axis = attrs
                .get("axis")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(1);
            let rank = g.shape(get(0)?)?.rank() as i64;
            g.flatten(
                get(0)?,
                isize::try_from(axis).map_err(|_| bad("Flatten axis overflow"))?,
                isize::try_from(rank - 1).map_err(|_| bad("Flatten rank overflow"))?,
            )?
        }
        "Squeeze" if ins.len() == 2 => {
            let axes = const_i64(constants, ins[1])?;
            let mut out = get(0)?;
            for a in axes_usize(&axes, g.shape(out)?.rank())?.into_iter().rev() {
                out = g.squeeze(out, Some(a as isize))?
            }
            out
        }
        "Unsqueeze" if ins.len() == 2 => {
            let axes = const_i64(constants, ins[1])?;
            let mut out = get(0)?;
            let mut ax = axes
                .into_iter()
                .map(|x| usize::try_from(x).map_err(|_| bad("negative axis")))
                .collect::<Result<Vec<_>>>()?;
            ax.sort_unstable();
            for a in ax {
                out = g.unsqueeze(out, a as isize)?
            }
            out
        }
        "Concat" if ins.len() >= 2 => {
            let axis = scalar_i64(attrs.get("axis").ok_or_else(|| bad("Concat needs axis"))?)?;
            let rank = g.shape(get(0)?)?.rank();
            g.concat(
                ins.iter()
                    .map(|x| {
                        values
                            .get(*x)
                            .copied()
                            .ok_or_else(|| bad("missing ONNX input"))
                    })
                    .collect::<Result<Vec<_>>>()?,
                axes_usize(&[axis], rank)?[0],
            )?
        }
        "Softmax" if ins.len() == 1 => {
            let axis = attrs
                .get("axis")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(-1);
            g.softmax(
                get(0)?,
                isize::try_from(axis).map_err(|_| bad("Softmax axis overflow"))?,
                None,
            )?
        }
        "LogSoftmax" if ins.len() == 1 => {
            let axis = attrs
                .get("axis")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(-1);
            g.log_softmax(
                get(0)?,
                isize::try_from(axis).map_err(|_| bad("LogSoftmax axis overflow"))?,
                None,
            )?
        }
        "Gemm" if ins.len() == 2 || ins.len() == 3 => {
            let alpha = attrs
                .get("alpha")
                .map(|x| scalar_f32(x))
                .transpose()?
                .unwrap_or(1.);
            let beta = attrs
                .get("beta")
                .map(|x| scalar_f32(x))
                .transpose()?
                .unwrap_or(1.);
            if !alpha.is_finite() || !beta.is_finite() {
                return Err(bad("Gemm alpha/beta must be finite"));
            }
            let transpose = |g: &mut Graph, n: NodeId, on: bool| -> Result<NodeId> {
                if !on {
                    return Ok(n);
                }
                let rank = g.shape(n)?.rank();
                if rank < 2 {
                    return Err(bad("Gemm transpose needs rank >= 2"));
                }
                let mut p: Vec<usize> = (0..rank).collect();
                p.swap(rank - 1, rank - 2);
                g.permute(n, p)
            };
            let a = transpose(
                g,
                get(0)?,
                attrs
                    .get("transA")
                    .map(|x| scalar_i64(x))
                    .transpose()?
                    .unwrap_or(0)
                    != 0,
            )?;
            let b = transpose(
                g,
                get(1)?,
                attrs
                    .get("transB")
                    .map(|x| scalar_i64(x))
                    .transpose()?
                    .unwrap_or(0)
                    != 0,
            )?;
            let y = g.matmul(a, b)?;
            let y = if alpha == 1. {
                y
            } else {
                let scale = g.constant(TensorData::scalar(alpha));
                g.mul(y, scale)?
            };
            if ins.len() == 3 {
                let c = get(2)?;
                let c = if beta == 1. {
                    c
                } else {
                    let scale = g.constant(TensorData::scalar(beta));
                    g.mul(c, scale)?
                };
                g.add(y, c)?
            } else {
                y
            }
        }
        "Equal" if ins.len() == 2 && attrs.is_empty() => g.eq(get(0)?, get(1)?)?,
        "Less" if ins.len() == 2 && attrs.is_empty() => g.lt(get(0)?, get(1)?)?,
        "LessOrEqual" if ins.len() == 2 && attrs.is_empty() => g.le(get(0)?, get(1)?)?,
        "Greater" if ins.len() == 2 && attrs.is_empty() => g.gt(get(0)?, get(1)?)?,
        "GreaterOrEqual" if ins.len() == 2 && attrs.is_empty() => g.ge(get(0)?, get(1)?)?,
        "Where" if ins.len() == 3 && attrs.is_empty() => g.select(get(0)?, get(1)?, get(2)?)?,
        "Pow" if ins.len() == 2 && attrs.is_empty() => g.pow(get(0)?, get(1)?)?,
        "Sqrt" if ins.len() == 1 && attrs.is_empty() => g.sqrt(get(0)?)?,
        "Exp" if ins.len() == 1 && attrs.is_empty() => g.exp(get(0)?)?,
        "Log" if ins.len() == 1 && attrs.is_empty() => g.log(get(0)?)?,
        "Abs" if ins.len() == 1 && attrs.is_empty() => g.abs(get(0)?)?,
        "Neg" if ins.len() == 1 && attrs.is_empty() => g.neg(get(0)?)?,
        "LeakyRelu" if ins.len() == 1 => {
            if attrs.keys().any(|x| x != "alpha") {
                return Err(bad("unsupported LeakyRelu attribute"));
            }
            let alpha = attrs
                .get("alpha")
                .map(|x| scalar_f32(x))
                .transpose()?
                .unwrap_or(0.01);
            if !alpha.is_finite() {
                return Err(bad("LeakyRelu alpha must be finite"));
            }
            let x = get(0)?;
            let dtype = g.dtype(x)?;
            let slope = g.constant(TensorData::scalar(alpha));
            let slope = g.cast(slope, dtype)?;
            g.leaky_relu(x, slope)?
        }
        "Clip" if (1..=3).contains(&ins.len()) && attrs.is_empty() => {
            let x = get(0)?;
            let bound = |i: usize| -> Result<Option<NodeId>> {
                let Some(name) = ins.get(i).filter(|x| !x.is_empty()) else {
                    return Ok(None);
                };
                let data = constants
                    .get(*name)
                    .ok_or_else(|| bad("Clip bounds must be constant initializers"))?;
                if data.len() != 1 || data.dtype() != g.dtype(x)? {
                    return Err(bad("Clip bounds must be same-dtype scalar tensors"));
                }
                Ok(Some(get(i)?))
            };
            g.clamp(x, bound(1)?, bound(2)?)?
        }
        "Dropout" if (1..=3).contains(&ins.len()) && attrs.is_empty() => {
            let x = get(0)?;
            if let Some(name) = ins.get(1).filter(|x| !x.is_empty()) {
                let value = constants
                    .get(*name)
                    .ok_or_else(|| bad("Dropout ratio must be constant"))?;
                if value.len() != 1
                    || !value.dtype().is_float()
                    || value.scalar_at(0).as_f64() != 0.0
                {
                    return Err(bad("only inference Dropout with zero ratio is supported"));
                }
            }
            if let Some(name) = ins.get(2).filter(|x| !x.is_empty()) {
                let value = constants
                    .get(*name)
                    .ok_or_else(|| bad("Dropout training_mode must be constant"))?;
                if value.len() != 1 || value.dtype() != DType::Bool || value.scalar_at(0).as_bool()
                {
                    return Err(bad(
                        "only inference Dropout with training_mode=false is supported",
                    ));
                }
            }
            x
        }
        "Shape" if ins.len() == 1 => {
            if attrs.keys().any(|x| x != "start" && x != "end") {
                return Err(bad("unsupported Shape attribute"));
            }
            let dims = g.shape(get(0)?)?.dims();
            let start = attrs
                .get("start")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(0);
            let end = attrs
                .get("end")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(dims.len() as i64);
            let normalize = |x: i64| -> Result<usize> {
                usize::try_from(if x < 0 { x + dims.len() as i64 } else { x })
                    .ok()
                    .filter(|&x| x <= dims.len())
                    .ok_or_else(|| bad("invalid Shape start/end"))
            };
            let (start, end) = (normalize(start)?, normalize(end)?);
            if start > end {
                return Err(bad("Shape start exceeds end"));
            }
            let data = TensorData::from_scalars(
                [end - start],
                DType::I64,
                dims[start..end].iter().map(|&x| Scalar::I(x as i64)),
            )?;
            constants.insert(outs[0].to_owned(), data.clone());
            g.constant(data)
        }
        "Expand" if ins.len() == 2 && attrs.is_empty() => {
            let x = get(0)?;
            let shape = const_i64(constants, ins[1])?
                .into_iter()
                .map(|x| usize::try_from(x).map_err(|_| bad("Expand shape must be nonnegative")))
                .collect::<Result<Vec<_>>>()?;
            g.expand(x, Shape::new(shape))?
        }
        "Tile" if ins.len() == 2 && attrs.is_empty() => {
            let x = get(0)?;
            let repeats = const_i64(constants, ins[1])?;
            if repeats.len() != g.shape(x)?.rank() || repeats.iter().any(|&x| x < 0) {
                return Err(bad("Tile repeats must be nonnegative and match rank"));
            }
            g.tile(
                x,
                &repeats.into_iter().map(|x| x as isize).collect::<Vec<_>>(),
            )?
        }
        "Gather" if ins.len() == 2 => {
            if attrs.keys().any(|x| x != "axis") {
                return Err(bad("unsupported Gather attribute"));
            }
            let x = get(0)?;
            let axis = attrs
                .get("axis")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(0);
            let rank = g.shape(x)?.rank();
            let axis = axes_usize(&[axis], rank)?[0];
            let name = ins[1];
            let data = constants
                .get(name)
                .ok_or_else(|| bad("Gather indices must be constant"))?;
            if !matches!(data.dtype(), DType::I32 | DType::I64) || data.shape() != g.shape(x)? {
                return Err(bad("Gather requires same-rank constant I32/I64 indices"));
            }
            if (0..data.len()).any(|i| data.scalar_at(i).as_i64() < 0) {
                return Err(bad("Gather negative indices are unsupported"));
            }
            g.gather(x, get(1)?, axis)?
        }
        "Slice" if (3..=5).contains(&ins.len()) && attrs.is_empty() => {
            let x = get(0)?;
            let starts = const_i64(constants, ins[1])?;
            let ends = const_i64(constants, ins[2])?;
            if starts.len() != ends.len() {
                return Err(bad("Slice starts/ends length mismatch"));
            }
            let axes = if ins.len() >= 4 && !ins[3].is_empty() {
                const_i64(constants, ins[3])?
            } else {
                (0..starts.len()).map(|x| x as i64).collect()
            };
            let steps = if ins.len() == 5 && !ins[4].is_empty() {
                const_i64(constants, ins[4])?
            } else {
                vec![1; starts.len()]
            };
            if axes.len() != starts.len() || steps.len() != starts.len() {
                return Err(bad("Slice control lengths mismatch"));
            }
            let rank = g.shape(x)?.rank();
            let axes = axes_usize(&axes, rank)?;
            let mut slices = vec![
                crate::Slice {
                    start: None,
                    stop: None,
                    step: 1
                };
                rank
            ];
            for ((axis, start), (end, step)) in axes
                .into_iter()
                .zip(starts)
                .zip(ends.into_iter().zip(steps))
            {
                if step == 0 {
                    return Err(bad("Slice step must be nonzero"));
                }
                let step = isize::try_from(step).map_err(|_| bad("Slice step overflow"))?;
                let start = isize::try_from(start).map_err(|_| bad("Slice start overflow"))?;
                let end = isize::try_from(end).map_err(|_| bad("Slice end overflow"))?;
                if slices[axis].step != 1 {
                    return Err(bad("duplicate Slice axis"));
                }
                slices[axis] = crate::Slice {
                    start: Some(start),
                    stop: Some(end),
                    step,
                };
            }
            g.stride(x, slices)?
        }
        "Pad" if (2..=3).contains(&ins.len()) => {
            if attrs.keys().any(|x| x != "mode") {
                return Err(bad("unsupported Pad attribute"));
            }
            if attrs.get("mode").map(Vec::as_slice).unwrap_or(b"constant") != b"constant" {
                return Err(bad("only constant Pad mode is supported"));
            }
            let x = get(0)?;
            let rank = g.shape(x)?.rank();
            let pads = const_i64(constants, ins[1])?;
            if pads.len() != 2 * rank {
                return Err(bad("Pad pads must contain begin/end values for every axis"));
            }
            if pads.iter().any(|&x| x < 0) {
                return Err(bad("negative ONNX Pad cropping is unsupported"));
            }
            let fill = if ins.len() == 3 && !ins[2].is_empty() {
                let value = constants
                    .get(ins[2])
                    .ok_or_else(|| bad("Pad constant_value must be constant"))?;
                if value.len() != 1 || value.dtype() != g.dtype(x)? {
                    return Err(bad("Pad constant_value must be a same-dtype scalar"));
                }
                value.scalar_at(0)
            } else {
                Scalar::I(0)
            };
            let padding = (0..rank)
                .map(|i| {
                    Ok((
                        usize::try_from(pads[i]).map_err(|_| bad("Pad overflow"))?,
                        usize::try_from(pads[rank + i]).map_err(|_| bad("Pad overflow"))?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            g.pad(x, padding, fill)?
        }
        "ConstantOfShape" if ins.len() == 1 => {
            if attrs.keys().any(|x| x != "value") {
                return Err(bad("unsupported ConstantOfShape attribute"));
            }
            let dims = const_i64(constants, ins[0])?
                .into_iter()
                .map(|x| {
                    usize::try_from(x)
                        .map_err(|_| bad("ConstantOfShape dimensions must be nonnegative"))
                })
                .collect::<Result<Vec<_>>>()?;
            let (value, dtype) = match attrs.get("value") {
                Some(bytes) => {
                    let value = tensor_data(Msg::new(bytes))?;
                    if value.len() != 1 {
                        return Err(bad("ConstantOfShape value must contain one element"));
                    }
                    (value.scalar_at(0), value.dtype())
                }
                None => (Scalar::F(0.0), DType::F32),
            };
            g.full_with_dtype(Shape::new(dims), value, dtype)?
        }
        op @ ("ReduceSum" | "ReduceMean" | "ReduceProd" | "ReduceMin" | "ReduceMax")
            if (1..=2).contains(&ins.len()) =>
        {
            if attrs
                .keys()
                .any(|x| x != "keepdims" && x != "noop_with_empty_axes")
            {
                return Err(bad("unsupported Reduce attribute"));
            }
            let x = get(0)?;
            let keepdims = attrs
                .get("keepdims")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(1);
            let noop = attrs
                .get("noop_with_empty_axes")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(0);
            if !matches!(keepdims, 0 | 1) || !matches!(noop, 0 | 1) {
                return Err(bad("Reduce boolean attributes must be 0 or 1"));
            }
            let axes = if ins.len() == 2 && !ins[1].is_empty() {
                const_i64(constants, ins[1])?
            } else {
                Vec::new()
            };
            if axes.is_empty() && noop == 1 {
                x
            } else {
                let rank = g.shape(x)?.rank();
                let axes = if axes.is_empty() {
                    (0..rank).map(|x| x as isize).collect()
                } else {
                    let axes = axes_usize(&axes, rank)?;
                    if axes
                        .iter()
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                        != axes.len()
                    {
                        return Err(bad("duplicate Reduce axis"));
                    }
                    axes.into_iter().map(|x| x as isize).collect()
                };
                let kind = match op {
                    "ReduceSum" => ReduceKind::Sum,
                    "ReduceMean" => ReduceKind::Mean,
                    "ReduceProd" => ReduceKind::Product,
                    "ReduceMin" => ReduceKind::Min,
                    "ReduceMax" => ReduceKind::Max,
                    _ => unreachable!(),
                };
                g.reduce(x, kind, Some(axes), keepdims == 1)?
            }
        }
        op @ ("ArgMax" | "ArgMin") if ins.len() == 1 => {
            if attrs
                .keys()
                .any(|x| !matches!(x.as_str(), "axis" | "keepdims" | "select_last_index"))
            {
                return Err(bad("unsupported Arg attribute"));
            }
            if attrs
                .get("select_last_index")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(0)
                != 0
            {
                return Err(bad("Arg select_last_index is unsupported"));
            }
            let x = get(0)?;
            let axis = attrs
                .get("axis")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(0);
            let axis = axes_usize(&[axis], g.shape(x)?.rank())?[0] as isize;
            let keepdims = attrs
                .get("keepdims")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(1);
            if !matches!(keepdims, 0 | 1) {
                return Err(bad("Arg keepdims must be 0 or 1"));
            }
            let value = if op == "ArgMax" {
                g.argmax(x, Some(axis), keepdims == 1)?
            } else {
                g.argmin(x, Some(axis), keepdims == 1)?
            };
            g.cast(value, DType::I64)?
        }
        "BatchNormalization" if ins.len() == 5 => {
            if attrs.keys().any(|x| {
                !matches!(
                    x.as_str(),
                    "epsilon" | "training_mode" | "momentum" | "spatial"
                )
            }) {
                return Err(bad("unsupported BatchNormalization attribute"));
            }
            if attrs.contains_key("momentum") || attrs.contains_key("spatial") {
                return Err(bad(
                    "BatchNormalization momentum/spatial attributes are unsupported",
                ));
            }
            if attrs
                .get("training_mode")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(0)
                != 0
            {
                return Err(bad("BatchNormalization training mode is unsupported"));
            }
            let epsilon = attrs
                .get("epsilon")
                .map(|x| scalar_f32(x))
                .transpose()?
                .unwrap_or(1e-5);
            if !epsilon.is_finite() || epsilon < 0. {
                return Err(bad(
                    "BatchNormalization epsilon must be finite and nonnegative",
                ));
            }
            let x = get(0)?;
            let shape = g.shape(x)?.clone();
            let dtype = g.dtype(x)?;
            if shape.rank() < 2 || !dtype.is_float() {
                return Err(bad("BatchNormalization X must be a rank >= 2 float tensor"));
            }
            let channels = shape.dims()[1];
            let param_shape = Shape::new([channels]);
            let mut broadcast = vec![1; shape.rank()];
            broadcast[1] = channels;
            let params = [get(1)?, get(2)?, get(3)?, get(4)?];
            for param in params {
                if g.dtype(param)? != dtype || g.shape(param)? != &param_shape {
                    return Err(bad(
                        "BatchNormalization parameters must be same-dtype [C] tensors",
                    ));
                }
            }
            let scale = g.reshape(params[0], broadcast.clone())?;
            let bias = g.reshape(params[1], broadcast.clone())?;
            let mean = g.reshape(params[2], broadcast.clone())?;
            let variance = g.reshape(params[3], broadcast)?;
            let epsilon = g.constant(TensorData::scalar(epsilon));
            let epsilon = g.cast(epsilon, dtype)?;
            let centered = g.sub(x, mean)?;
            let variance = g.add(variance, epsilon)?;
            let inv_std = g.sqrt(variance)?;
            let normalized = g.div(centered, inv_std)?;
            let scaled = g.mul(normalized, scale)?;
            g.add(scaled, bias)?
        }
        "GlobalAveragePool" if ins.len() == 1 && attrs.is_empty() => {
            let x = get(0)?;
            let rank = g.shape(x)?.rank();
            if rank < 3 || !g.dtype(x)?.is_float() {
                return Err(bad("GlobalAveragePool requires a rank >= 3 float tensor"));
            }
            g.reduce(
                x,
                ReduceKind::Mean,
                Some((2..rank).map(|x| x as isize).collect()),
                true,
            )?
        }
        "MaxPool" if ins.len() == 1 => {
            let x = get(0)?;
            if g.shape(x)?.rank() != 4 || !g.dtype(x)?.is_float() {
                return Err(bad("MaxPool requires a rank-4 float NCHW tensor"));
            }
            let options = onnx_pool_options(&attrs, true, g.shape(x)?.dims())?;
            g.max_pool(x, options)?
        }
        "AveragePool" if ins.len() == 1 => {
            let x = get(0)?;
            if g.shape(x)?.rank() != 4 || !g.dtype(x)?.is_float() {
                return Err(bad("AveragePool requires a rank-4 float NCHW tensor"));
            }
            let options = onnx_pool_options(&attrs, false, g.shape(x)?.dims())?;
            g.avg_pool(x, options)?
        }
        "Conv" if ins.len() == 2 || ins.len() == 3 => {
            let x = get(0)?;
            let w = get(1)?;
            let strides = conv_pair(&attrs, "strides", [1, 1], false)?;
            let dilations = conv_pair(&attrs, "dilations", [1, 1], false)?;
            let groups = attrs
                .get("group")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(1);
            let groups = usize::try_from(groups)
                .ok()
                .filter(|&x| x != 0)
                .ok_or_else(|| bad("Conv group must be positive"))?;
            let explicit_pads = attrs.contains_key("pads");
            let pads = conv_pads(&attrs)?;
            let auto_pad = attrs
                .get("auto_pad")
                .map(Vec::as_slice)
                .unwrap_or(b"NOTSET");
            if auto_pad != b"NOTSET" && explicit_pads {
                return Err(bad("Conv pads conflicts with auto_pad"));
            }
            let padding = match auto_pad {
                b"NOTSET" => pads,
                b"VALID" => [0; 4],
                b"SAME_UPPER" => conv_same_padding(
                    g.shape(x)?.dims(),
                    g.shape(w)?.dims(),
                    strides,
                    dilations,
                    false,
                )?,
                b"SAME_LOWER" => conv_same_padding(
                    g.shape(x)?.dims(),
                    g.shape(w)?.dims(),
                    strides,
                    dilations,
                    true,
                )?,
                _ => return Err(bad("unsupported Conv auto_pad")),
            };
            g.conv2d(
                x,
                w,
                if ins.len() == 3 { Some(get(2)?) } else { None },
                Conv2dOptions {
                    groups,
                    stride: strides,
                    dilation: dilations,
                    padding,
                },
            )?
        }
        _ => return Err(bad(format!("unsupported ONNX opset-13 operator {op}"))),
    };
    values.insert(outs[0].to_owned(), out);
    Ok(())
}
fn onnx_dtype(x: u64) -> Result<DType> {
    match x {
        1 => Ok(DType::F32),
        11 => Ok(DType::F64),
        6 => Ok(DType::I32),
        7 => Ok(DType::I64),
        9 => Ok(DType::Bool),
        10 => Ok(DType::F16),
        16 => Ok(DType::BF16),
        2 => Ok(DType::U8),
        3 => Ok(DType::I8),
        5 => Ok(DType::I16),
        4 => Ok(DType::U16),
        12 => Ok(DType::U32),
        13 => Ok(DType::U64),
        _ => Err(bad("unsupported ONNX dtype")),
    }
}
fn attrs(n: &Msg<'_>) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut out = BTreeMap::new();
    for b in n.bytes(5)? {
        let a = Msg::new(b);
        let name = a
            .string(1)?
            .ok_or_else(|| bad("attribute lacks name"))?
            .to_owned();
        let fields = a.fields()?;
        let value = if let Some((_, 5, x)) = fields.iter().find(|(i, w, _)| *i == 2 && *w == 5) {
            x.to_vec()
        } else if let Some((_, 0, x)) = fields.iter().find(|(i, w, _)| *i == 3 && *w == 0) {
            x.to_vec()
        } else if let Some((_, 2, x)) = fields
            .iter()
            .find(|(i, w, _)| (*i == 4 || *i == 5 || *i == 8) && *w == 2)
        {
            x.to_vec()
        } else {
            return Err(bad("unsupported ONNX attribute form"));
        };
        if out.insert(name, value).is_some() {
            return Err(bad("duplicate ONNX attribute"));
        }
    }
    Ok(out)
}
fn scalar_i64(b: &[u8]) -> Result<i64> {
    let mut at = 0;
    Ok(var(b, &mut at)? as i64)
}
fn scalar_f32(b: &[u8]) -> Result<f32> {
    let a: [u8; 4] = b
        .try_into()
        .map_err(|_| bad("ONNX float attribute must be f32"))?;
    Ok(f32::from_le_bytes(a))
}
fn packed_i64(b: &[u8]) -> Result<Vec<i64>> {
    let mut at = 0;
    let mut x = vec![];
    while at < b.len() {
        x.push(var(b, &mut at)? as i64)
    }
    Ok(x)
}
fn conv_pair(
    attrs: &BTreeMap<String, Vec<u8>>,
    name: &str,
    default: [usize; 2],
    allow_zero: bool,
) -> Result<[usize; 2]> {
    let x = match attrs.get(name) {
        Some(x) => packed_i64(x)?,
        None => return Ok(default),
    };
    if x.len() != 2 {
        return Err(bad(format!("Conv {name} must have two values")));
    }
    let mut out = [0; 2];
    for (dst, src) in out.iter_mut().zip(x) {
        *dst = usize::try_from(src)
            .ok()
            .filter(|&v| allow_zero || v != 0)
            .ok_or_else(|| bad(format!("Conv {name} must be positive")))?;
    }
    Ok(out)
}
fn conv_pads(attrs: &BTreeMap<String, Vec<u8>>) -> Result<[usize; 4]> {
    let x = match attrs.get("pads") {
        Some(x) => packed_i64(x)?,
        None => return Ok([0; 4]),
    };
    if x.len() != 4 {
        return Err(bad("Conv pads must have four values"));
    }
    let x: Vec<usize> = x
        .into_iter()
        .map(|v| usize::try_from(v).map_err(|_| bad("Conv pads must be nonnegative")))
        .collect::<Result<_>>()?;
    Ok([x[0], x[2], x[1], x[3]])
}
fn conv_same_padding(
    input: &[usize],
    weight: &[usize],
    stride: [usize; 2],
    dilation: [usize; 2],
    lower: bool,
) -> Result<[usize; 4]> {
    if input.len() != 4 || weight.len() != 4 {
        return Err(bad("Conv SAME padding requires rank-4 NCHW tensors"));
    }
    let mut out = [0; 4];
    for i in 0..2 {
        let spatial = input[2 + i];
        let kernel = weight[2 + i];
        if spatial == 0 || kernel == 0 {
            return Err(bad("Conv SAME padding requires nonzero spatial dimensions"));
        }
        let output = spatial
            .checked_add(stride[i] - 1)
            .ok_or_else(|| bad("Conv SAME padding overflow"))?
            / stride[i];
        let effective = dilation[i]
            .checked_mul(kernel - 1)
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| bad("Conv SAME padding overflow"))?;
        let needed = output
            .checked_sub(1)
            .and_then(|x| x.checked_mul(stride[i]))
            .and_then(|x| x.checked_add(effective))
            .ok_or_else(|| bad("Conv SAME padding overflow"))?
            .saturating_sub(spatial);
        let before = if lower {
            needed.div_ceil(2)
        } else {
            needed / 2
        };
        let after = needed - before;
        out[i * 2] = before;
        out[i * 2 + 1] = after;
    }
    Ok(out)
}
fn onnx_pool_options(
    attrs: &BTreeMap<String, Vec<u8>>,
    max: bool,
    input: &[usize],
) -> Result<PoolOptions> {
    if attrs.keys().any(|name| {
        !matches!(
            name.as_str(),
            "kernel_shape"
                | "strides"
                | "dilations"
                | "pads"
                | "auto_pad"
                | "ceil_mode"
                | "count_include_pad"
                | "storage_order"
        )
    }) {
        return Err(bad("unsupported ONNX pool attribute"));
    }
    let kernel = conv_pair(attrs, "kernel_shape", [0, 0], false)?;
    if !attrs.contains_key("kernel_shape") {
        return Err(bad("ONNX pool requires kernel_shape"));
    }
    let stride = conv_pair(attrs, "strides", [1, 1], false)?;
    let dilation = conv_pair(attrs, "dilations", [1, 1], false)?;
    if max
        && attrs
            .get("storage_order")
            .map(|x| scalar_i64(x))
            .transpose()?
            .unwrap_or(0)
            != 0
    {
        return Err(bad(
            "MaxPool storage_order other than row-major is unsupported",
        ));
    }
    if !max && (attrs.contains_key("storage_order") || attrs.contains_key("dilations")) {
        return Err(bad("unsupported AveragePool attribute"));
    }
    let bool_attr = |name: &str, default: bool| -> Result<bool> {
        match attrs.get(name) {
            None => Ok(default),
            Some(value) => match scalar_i64(value)? {
                0 => Ok(false),
                1 => Ok(true),
                _ => Err(bad(format!("{name} must be 0 or 1"))),
            },
        }
    };
    let explicit = attrs.contains_key("pads");
    let pads = conv_pads(attrs)?;
    let auto = attrs
        .get("auto_pad")
        .map(Vec::as_slice)
        .unwrap_or(b"NOTSET");
    if auto != b"NOTSET" && explicit {
        return Err(bad("pool pads conflicts with auto_pad"));
    }
    let padding = match auto {
        b"NOTSET" => pads,
        b"VALID" => [0; 4],
        b"SAME_UPPER" => conv_same_padding(
            input,
            &[1, 1, kernel[0], kernel[1]],
            stride,
            dilation,
            false,
        )?,
        b"SAME_LOWER" => {
            conv_same_padding(input, &[1, 1, kernel[0], kernel[1]], stride, dilation, true)?
        }
        _ => return Err(bad("unsupported pool auto_pad")),
    };
    Ok(PoolOptions {
        kernel: kernel.to_vec(),
        stride: stride.to_vec(),
        dilation: dilation.to_vec(),
        padding: vec![(padding[0], padding[1]), (padding[2], padding[3])],
        ceil_mode: bool_attr("ceil_mode", false)?,
        count_include_pad: if max {
            false
        } else {
            bool_attr("count_include_pad", false)?
        },
    })
}
fn axes_usize(x: &[i64], rank: usize) -> Result<Vec<usize>> {
    x.iter()
        .map(|&a| {
            let a = if a < 0 { a + rank as i64 } else { a };
            usize::try_from(a)
                .ok()
                .filter(|&a| a < rank)
                .ok_or_else(|| bad("invalid ONNX axis"))
        })
        .collect()
}
fn const_i64(c: &BTreeMap<String, TensorData>, name: &str) -> Result<Vec<i64>> {
    let x = c
        .get(name)
        .ok_or_else(|| bad("ONNX shape/axes input must be a constant initializer"))?;
    if x.dtype() != DType::I64 {
        return Err(bad("ONNX shape/axes constant must be I64"));
    }
    Ok((0..x.len()).map(|i| x.scalar_at(i).as_i64()).collect())
}
fn reshape_dims(old: &[usize], shape: &[i64]) -> Result<Shape> {
    let mut out = Vec::new();
    let mut infer = None;
    let mut known = 1usize;
    for (i, &x) in shape.iter().enumerate() {
        if x == 0 {
            let d = *old
                .get(i)
                .ok_or_else(|| bad("Reshape zero axis out of range"))?;
            out.push(d);
            known = known
                .checked_mul(d)
                .ok_or_else(|| bad("Reshape overflow"))?
        } else if x == -1 {
            if infer.replace(i).is_some() {
                return Err(bad("multiple Reshape -1 dimensions"));
            }
            out.push(1)
        } else {
            let d = usize::try_from(x).map_err(|_| bad("negative Reshape dimension"))?;
            out.push(d);
            known = known
                .checked_mul(d)
                .ok_or_else(|| bad("Reshape overflow"))?
        }
    }
    let total = old.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d).ok_or_else(|| bad("Reshape overflow"))
    })?;
    if let Some(i) = infer {
        if known == 0 || total % known != 0 {
            return Err(bad("invalid Reshape inferred dimension"));
        }
        out[i] = total / known
    } else if total != known {
        return Err(bad("Reshape element count mismatch"));
    }
    Ok(Shape::new(out))
}

fn tensor(m: Msg<'_>) -> Result<(String, TensorData)> {
    let name = m
        .string(8)?
        .ok_or_else(|| bad("ONNX initializer lacks name"))?
        .to_owned();
    if name.is_empty() {
        return Err(bad("empty ONNX initializer name"));
    }
    Ok((name, tensor_data(m)?))
}
fn tensor_data(m: Msg<'_>) -> Result<TensorData> {
    if !m.bytes(13)?.is_empty() {
        return Err(bad("ONNX external tensor data is unsupported"));
    }
    let dtype = onnx_dtype(one_varint(&m, 2, "tensor dtype")?)?;
    let dims = m.packed(1)?;
    let shape = Shape::new(
        dims.into_iter()
            .map(|x| {
                usize::try_from(i64::try_from(x).map_err(|_| bad("negative ONNX dimension"))?)
                    .map_err(|_| bad("ONNX dimension overflow"))
            })
            .collect::<Result<Vec<_>>>()?,
    );
    let raw = m.bytes(9)?;
    let typed = typed_tensor_bytes(&m, dtype, shape.numel()?)?;
    if !raw.is_empty() && !typed.is_empty() {
        return Err(bad("ONNX tensor raw_data conflicts with typed data"));
    }
    let data = match (raw.as_slice(), typed.as_slice()) {
        ([x], []) => x.to_vec(),
        ([], [x]) => x.clone(),
        ([], []) => return Err(bad("ONNX tensor lacks data")),
        _ => return Err(bad("duplicate ONNX tensor data field")),
    };
    TensorData::from_le_bytes(shape, dtype, &data)
        .map_err(|error| bad(format!("invalid ONNX tensor data: {error}")))
}
fn typed_tensor_bytes(m: &Msg<'_>, dtype: DType, count: usize) -> Result<Vec<Vec<u8>>> {
    let f = m.fields()?;
    let fields: Vec<_> = f
        .iter()
        .filter(|(i, _, _)| matches!(*i, 4 | 5 | 7 | 10 | 11))
        .collect();
    if fields.is_empty() {
        return Ok(vec![]);
    }
    let (mut out, field) = (
        Vec::new(),
        match dtype {
            DType::F32 => 4,
            DType::F64 => 10,
            DType::I64 => 7,
            DType::U64 => 11,
            DType::I32
            | DType::U8
            | DType::I8
            | DType::I16
            | DType::U16
            | DType::U32
            | DType::Bool
            | DType::F16
            | DType::BF16 => 5,
        },
    );
    for (i, w, b) in fields {
        if *i != field {
            return Err(bad("typed field incompatible with dtype"));
        }
        if matches!(dtype, DType::F32 | DType::F64) {
            if *w != if dtype == DType::F32 { 5 } else { 1 } {
                return Err(bad("typed float wire"));
            }
            out.extend_from_slice(b);
            continue;
        }
        let mut vals = Vec::new();
        if *w == 0 {
            let mut at = 0;
            vals.push(var(b, &mut at)?)
        } else if *w == 2 {
            let mut at = 0;
            while at < b.len() {
                vals.push(var(b, &mut at)?)
            }
        } else {
            return Err(bad("typed integer wire"));
        }
        for v in vals {
            match dtype {
                DType::I32 => out.extend_from_slice(&(v as u32 as i32).to_le_bytes()),
                DType::I64 => out.extend_from_slice(&(v as i64).to_le_bytes()),
                DType::U8 => out.push(u8::try_from(v).map_err(|_| bad("u8 range"))?),
                DType::I8 => out.push(i8::try_from(v as i64).map_err(|_| bad("i8 range"))? as u8),
                DType::I16 => out.extend_from_slice(
                    &(i16::try_from(v as i64).map_err(|_| bad("i16 range"))?).to_le_bytes(),
                ),
                DType::U16 => out.extend_from_slice(
                    &u16::try_from(v)
                        .map_err(|_| bad("u16 range"))?
                        .to_le_bytes(),
                ),
                DType::U32 => out.extend_from_slice(
                    &u32::try_from(v)
                        .map_err(|_| bad("u32 range"))?
                        .to_le_bytes(),
                ),
                DType::U64 => out.extend_from_slice(&v.to_le_bytes()),
                DType::Bool => out.push(if v == 0 {
                    0
                } else if v == 1 {
                    1
                } else {
                    return Err(bad("bool range"));
                }),
                DType::F16 | DType::BF16 => out.extend_from_slice(
                    &(u16::try_from(v).map_err(|_| bad("half range"))?).to_le_bytes(),
                ),
                _ => {}
            }
        }
    }
    if out.len()
        != count
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| bad("typed overflow"))?
    {
        return Err(bad("typed count mismatch"));
    }
    Ok(vec![out])
}
fn value_info(m: Msg<'_>) -> Result<(String, Shape, DType)> {
    let name = m
        .string(1)?
        .ok_or_else(|| bad("value info lacks name"))?
        .to_owned();
    let ty = Msg::new(one_bytes(&m, 2, "value type")?);
    let ten = Msg::new(one_bytes(&ty, 1, "tensor type")?);
    let dtype = match one_varint(&ten, 1, "value dtype")? {
        1 => DType::F32,
        11 => DType::F64,
        6 => DType::I32,
        7 => DType::I64,
        9 => DType::Bool,
        x => return Err(bad(format!("unsupported ONNX value dtype {x}"))),
    };
    let sh = Msg::new(one_bytes(&ten, 2, "tensor shape")?);
    let dims = sh
        .bytes(1)?
        .into_iter()
        .map(|d| {
            usize::try_from(one_varint(&Msg::new(d), 1, "dimension")?)
                .map_err(|_| bad("ONNX dimension overflow"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((name, Shape::new(dims), dtype))
}

mod wire;
use wire::{Msg, one_bytes, one_varint, var};

#[cfg(test)]
mod tests;
