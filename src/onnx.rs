//! Bounded static ONNX protobuf import. This intentionally supports a small,
//! audited inference subset (opset 13, default domain) and never executes code.

use crate::{
    Backend, Conv2dOptions, CpuBackend, DType, Error, Graph, NodeId, PoolOptions, ReduceKind,
    Result, Scalar, Shape, TensorData,
};
use std::collections::{BTreeMap, HashMap};

const MAX_BYTES: usize = 32 * 1024 * 1024;
const MAX_ITEMS: usize = 4096;
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
            let (name, data) = tensor(Msg::new(
                attrs
                    .get("value")
                    .ok_or_else(|| bad("Constant needs value"))?,
            ))?;
            constants.insert(outs[0].to_owned(), data.clone());
            let _ = name;
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
    if !matches!(x.dtype(), DType::I64 | DType::I32) {
        return Err(bad("ONNX shape/axes constant must be integer"));
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
    if !m.bytes(13)?.is_empty() {
        return Err(bad("ONNX external tensor data is unsupported"));
    }
    let name = m
        .string(8)?
        .ok_or_else(|| bad("ONNX initializer lacks name"))?
        .to_owned();
    if name.is_empty() {
        return Err(bad("empty ONNX initializer name"));
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
    TensorData::from_le_bytes(shape, dtype, &data).map(|x| (name, x))
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

struct Msg<'a> {
    b: &'a [u8],
}
impl<'a> Msg<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b }
    }
    fn fields(&self) -> Result<Vec<(u32, u8, &'a [u8])>> {
        let (mut at, mut v) = (0, Vec::new());
        while at < self.b.len() {
            if v.len() >= MAX_ITEMS {
                return Err(bad("ONNX field count exceeds limit"));
            }
            let key = var(self.b, &mut at)?;
            let wire = (key & 7) as u8;
            let n = match wire {
                0 => {
                    let s = at;
                    var(self.b, &mut at)?;
                    &self.b[s..at]
                }
                2 => {
                    let n = usize::try_from(var(self.b, &mut at)?)
                        .map_err(|_| bad("ONNX length overflow"))?;
                    let s = at;
                    at = at
                        .checked_add(n)
                        .ok_or_else(|| bad("ONNX length overflow"))?;
                    self.b
                        .get(s..at)
                        .ok_or_else(|| bad("truncated ONNX field"))?
                }
                5 => {
                    let s = at;
                    at = at
                        .checked_add(4)
                        .ok_or_else(|| bad("ONNX fixed32 overflow"))?;
                    self.b
                        .get(s..at)
                        .ok_or_else(|| bad("truncated ONNX fixed32"))?
                }
                1 => {
                    let s = at;
                    at = at
                        .checked_add(8)
                        .ok_or_else(|| bad("ONNX fixed64 overflow"))?;
                    self.b
                        .get(s..at)
                        .ok_or_else(|| bad("truncated ONNX fixed64"))?
                }
                _ => return Err(bad("unsupported ONNX protobuf wire type")),
            };
            v.push(((key >> 3) as u32, wire, n));
        }
        Ok(v)
    }
    fn bytes(&self, id: u32) -> Result<Vec<&'a [u8]>> {
        Ok(self
            .fields()?
            .into_iter()
            .filter_map(|(i, w, x)| (i == id && w == 2).then_some(x))
            .collect())
    }
    fn string(&self, id: u32) -> Result<Option<&'a str>> {
        match self.bytes(id)?.as_slice() {
            [] => Ok(None),
            [x] => std::str::from_utf8(x)
                .map(Some)
                .map_err(|_| bad("ONNX string is not UTF-8")),
            _ => Err(bad("duplicate ONNX string field")),
        }
    }
    fn strings(&self, id: u32) -> Result<Vec<&'a str>> {
        self.bytes(id)?
            .into_iter()
            .map(|x| std::str::from_utf8(x).map_err(|_| bad("ONNX string is not UTF-8")))
            .collect()
    }
    fn packed(&self, id: u32) -> Result<Vec<u64>> {
        let mut out = Vec::new();
        for x in self.bytes(id)? {
            let mut at = 0;
            while at < x.len() {
                out.push(var(x, &mut at)?);
            }
        }
        Ok(out)
    }
}
fn var(b: &[u8], at: &mut usize) -> Result<u64> {
    let (mut x, mut s) = (0u64, 0);
    loop {
        let z = *b.get(*at).ok_or_else(|| bad("truncated ONNX varint"))?;
        *at += 1;
        x |= u64::from(z & 127) << s;
        if z < 128 {
            return Ok(x);
        }
        s += 7;
        if s >= 64 {
            return Err(bad("invalid ONNX varint"));
        }
    }
}
fn one_bytes<'a>(m: &Msg<'a>, id: u32, what: &str) -> Result<&'a [u8]> {
    match m.bytes(id)?.as_slice() {
        [x] => Ok(*x),
        _ => Err(bad(format!("ONNX {what} must occur once"))),
    }
}
fn one_varint(m: &Msg<'_>, id: u32, what: &str) -> Result<u64> {
    match m
        .fields()?
        .into_iter()
        .filter(|(i, w, _)| *i == id && *w == 0)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [(_, _, x)] => {
            let mut at = 0;
            var(x, &mut at)
        }
        _ => Err(bad(format!("ONNX {what} must occur once"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn vi(mut id: u32, out: &mut Vec<u8>) {
        loop {
            let b = (id & 127) as u8;
            id >>= 7;
            out.push(if id == 0 { b } else { b | 128 });
            if id == 0 {
                return;
            }
        }
    }
    fn field(out: &mut Vec<u8>, id: u32, data: &[u8]) {
        vi(id << 3 | 2, out);
        vi(data.len() as u32, out);
        out.extend_from_slice(data)
    }
    fn var(out: &mut Vec<u8>, id: u32, n: u32) {
        vi(id << 3, out);
        vi(n, out)
    }
    fn text(out: &mut Vec<u8>, id: u32, s: &str) {
        field(out, id, s.as_bytes())
    }
    fn ints_attr(name: &str, values: &[u32]) -> Vec<u8> {
        let mut a = vec![];
        text(&mut a, 1, name);
        let mut packed = vec![];
        for &value in values {
            vi(value, &mut packed);
        }
        field(&mut a, 8, &packed);
        a
    }
    fn int_attr(name: &str, value: u32) -> Vec<u8> {
        let mut a = vec![];
        text(&mut a, 1, name);
        var(&mut a, 3, value);
        a
    }
    fn string_attr(name: &str, value: &str) -> Vec<u8> {
        let mut a = vec![];
        text(&mut a, 1, name);
        text(&mut a, 4, value);
        a
    }
    fn value(name: &str, dims: &[u32]) -> Vec<u8> {
        let mut shape = vec![];
        for &d in dims {
            let mut dm = vec![];
            var(&mut dm, 1, d);
            field(&mut shape, 1, &dm)
        }
        let mut ten = vec![];
        var(&mut ten, 1, 1);
        field(&mut ten, 2, &shape);
        let mut ty = vec![];
        field(&mut ty, 1, &ten);
        let mut x = vec![];
        text(&mut x, 1, name);
        field(&mut x, 2, &ty);
        x
    }
    fn tensor(name: &str, dims: &[u32], data: &[f32]) -> Vec<u8> {
        let mut x = vec![];
        let mut packed = vec![];
        for &d in dims {
            vi(d, &mut packed)
        }
        field(&mut x, 1, &packed);
        var(&mut x, 2, 1);
        text(&mut x, 8, name);
        let raw: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        field(&mut x, 9, &raw);
        x
    }
    fn node(op: &str, ins: &[&str], out: &str) -> Vec<u8> {
        let mut x = vec![];
        for i in ins {
            text(&mut x, 1, i)
        }
        text(&mut x, 2, out);
        text(&mut x, 4, op);
        x
    }
    fn fattr(name: &str, value: f32) -> Vec<u8> {
        let mut a = vec![];
        text(&mut a, 1, name);
        vi(2 << 3 | 5, &mut a);
        a.extend_from_slice(&value.to_le_bytes());
        a
    }
    fn mlp() -> Vec<u8> {
        let mut g = vec![];
        field(&mut g, 11, &value("x", &[1, 2]));
        field(&mut g, 12, &value("y", &[1, 2]));
        field(&mut g, 5, &tensor("w", &[2, 2], &[1., 2., 3., 4.]));
        field(&mut g, 5, &tensor("b", &[1, 2], &[1., -10.]));
        field(&mut g, 1, &node("MatMul", &["x", "w"], "m"));
        field(&mut g, 1, &node("Add", &["m", "b"], "a"));
        field(&mut g, 1, &node("Relu", &["a"], "y"));
        let mut op = vec![];
        var(&mut op, 2, 13);
        let mut m = vec![];
        field(&mut m, 7, &g);
        field(&mut m, 8, &op);
        m
    }
    #[test]
    fn imports_static_mlp_and_rejects_schema() {
        let model = import_onnx(&mlp()).unwrap();
        let out = model
            .run(HashMap::from([(
                "x".into(),
                TensorData::new([1, 2], vec![1., 2.]).unwrap(),
            )]))
            .unwrap();
        assert_eq!(out["y"].values(), &[8., 0.]);
        let mut bad = mlp();
        bad[0] = 0xff;
        assert!(import_onnx(&bad).is_err());
    }
    #[test]
    fn imports_additional_static_activations() {
        let mut bytes = mlp();
        let at = bytes.windows(4).position(|x| x == b"Relu").unwrap();
        bytes[at..at + 4].copy_from_slice(b"Tanh");
        let out = import_onnx(&bytes)
            .unwrap()
            .run(HashMap::from([(
                "x".into(),
                TensorData::new([1, 2], vec![1., 2.]).unwrap(),
            )]))
            .unwrap();
        assert!(out["y"].values()[0] > 0.999 && out["y"].values()[1].abs() < 1e-6);
    }
    #[test]
    fn static_movement_shape_and_axis_contracts_are_checked() {
        assert_eq!(reshape_dims(&[2, 3], &[3, -1]).unwrap().dims(), &[3, 2]);
        assert!(reshape_dims(&[2, 3], &[0, 0, 0]).is_err());
        assert!(reshape_dims(&[2, 3], &[-1, -1]).is_err());
        assert_eq!(axes_usize(&[-1, 0], 2).unwrap(), vec![1, 0]);
        assert!(axes_usize(&[2], 2).is_err());
        let constants = BTreeMap::from([(
            "shape".into(),
            TensorData::from_scalars([2], DType::I64, [crate::Scalar::I(3), crate::Scalar::I(2)])
                .unwrap(),
        )]);
        assert_eq!(const_i64(&constants, "shape").unwrap(), vec![3, 2]);
    }
    #[test]
    fn gemm_and_softmax_lower_through_cpu_graph() {
        let mut g = Graph::new();
        let a = g.input("a", [1, 2]);
        let b = g.input("b", [2, 2]);
        let c = g.input("c", [1, 2]);
        let mut values = BTreeMap::from([("a".into(), a), ("b".into(), b), ("c".into(), c)]);
        let mut constants = BTreeMap::new();
        lower(
            &mut g,
            Msg::new(&node("Gemm", &["a", "b", "c"], "m")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        lower(
            &mut g,
            Msg::new(&node("Softmax", &["m"], "y")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        let out = CpuBackend
            .execute(
                &g,
                values["y"],
                &HashMap::from([
                    ("a".into(), TensorData::new([1, 2], vec![1., 2.]).unwrap()),
                    (
                        "b".into(),
                        TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
                    ),
                    ("c".into(), TensorData::new([1, 2], vec![1., 0.]).unwrap()),
                ]),
            )
            .unwrap();
        assert!(out.values()[1] > out.values()[0]);
        assert!((out.values()[0] + out.values()[1] - 1.).abs() < 1e-6);
    }
    #[test]
    fn gemm_finite_scales_are_compositional() {
        let mut g = Graph::new();
        let a = g.input("a", [1, 1]);
        let b = g.input("b", [1, 1]);
        let c = g.input("c", [1, 1]);
        let mut values = BTreeMap::from([("a".into(), a), ("b".into(), b), ("c".into(), c)]);
        let mut n = node("Gemm", &["a", "b", "c"], "y");
        field(&mut n, 5, &fattr("alpha", 2.));
        field(&mut n, 5, &fattr("beta", 3.));
        lower(&mut g, Msg::new(&n), &mut values, &mut BTreeMap::new()).unwrap();
        let out = CpuBackend
            .execute(
                &g,
                values["y"],
                &HashMap::from([
                    ("a".into(), TensorData::new([1, 1], vec![2.]).unwrap()),
                    ("b".into(), TensorData::new([1, 1], vec![4.]).unwrap()),
                    ("c".into(), TensorData::new([1, 1], vec![5.]).unwrap()),
                ]),
            )
            .unwrap();
        assert_eq!(out.values(), &[31.]);
    }
    #[test]
    fn typed_payloads_match_raw_bits_including_u64() {
        let raw = tensor(
            "f",
            &[2],
            &[f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234)],
        );
        let mut typed = vec![];
        let mut dims = vec![];
        vi(2, &mut dims);
        field(&mut typed, 1, &dims);
        var(&mut typed, 2, 1);
        text(&mut typed, 8, "f");
        vi(4 << 3 | 5, &mut typed);
        typed.extend_from_slice(&0x8000_0000u32.to_le_bytes());
        vi(4 << 3 | 5, &mut typed);
        typed.extend_from_slice(&0x7fc0_1234u32.to_le_bytes());
        assert_eq!(
            super::tensor(Msg::new(&raw))
                .unwrap()
                .1
                .to_le_bytes()
                .unwrap(),
            super::tensor(Msg::new(&typed))
                .unwrap()
                .1
                .to_le_bytes()
                .unwrap()
        );
        let mut u = vec![];
        field(&mut u, 1, &[1]);
        var(&mut u, 2, 13);
        text(&mut u, 8, "u");
        let mut packed = vec![0xff; 9];
        packed.push(1);
        field(&mut u, 11, &packed);
        assert_eq!(
            super::tensor(Msg::new(&u))
                .unwrap()
                .1
                .to_le_bytes()
                .unwrap(),
            u64::MAX.to_le_bytes()
        );
    }
    #[test]
    fn typed_payload_acceptance_and_rejection_matrix() {
        fn msg(dtype: u32, fid: u32, payload: Vec<u8>) -> Vec<u8> {
            let mut x = vec![];
            field(&mut x, 1, &[1]);
            var(&mut x, 2, dtype);
            text(&mut x, 8, "x");
            field(&mut x, fid, &payload);
            x
        }
        let cases = [
            (9, 5, vec![1], vec![1]),
            (3, 5, vec![127], vec![127]),
            (2, 5, vec![0xff, 1], vec![0xff]),
            (5, 5, vec![123], vec![123, 0]),
            (4, 5, vec![0xff, 0xff, 3], vec![0xff, 0xff]),
            (
                6,
                5,
                vec![0xff, 0xff, 0xff, 0xff, 0x0f],
                (-1i32).to_le_bytes().to_vec(),
            ),
            (
                12,
                5,
                vec![0xff, 0xff, 0xff, 0xff, 0x0f],
                u32::MAX.to_le_bytes().to_vec(),
            ),
            (7, 7, vec![0x7f], 127i64.to_le_bytes().to_vec()),
            (10, 5, vec![0xff, 0xff, 3], vec![0xff, 0xff]),
            (16, 5, vec![0x81, 0xfc, 3], vec![0x01, 0xfe]),
        ];
        for (dtype, field, payload, expect) in cases {
            assert_eq!(
                super::tensor(Msg::new(&msg(dtype, field, payload)))
                    .unwrap()
                    .1
                    .to_le_bytes()
                    .unwrap(),
                expect,
                "dtype {dtype}"
            )
        }
        let mut bad = msg(9, 5, vec![2]);
        assert!(super::tensor(Msg::new(&bad)).is_err());
        bad = msg(2, 5, vec![0x80, 0x02]);
        assert!(super::tensor(Msg::new(&bad)).is_err());
        bad = msg(1, 5, vec![1]);
        assert!(super::tensor(Msg::new(&bad)).is_err());
        let mut conflict = msg(1, 4, 0u32.to_le_bytes().to_vec());
        field(&mut conflict, 9, &0f32.to_le_bytes());
        assert!(super::tensor(Msg::new(&conflict)).is_err());
    }
    #[test]
    fn default_nchw_conv_lowers_through_cpu_graph() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 2, 2]);
        let w = g.input("w", [1, 1, 1, 1]);
        let mut values = BTreeMap::from([("x".into(), x), ("w".into(), w)]);
        lower(
            &mut g,
            Msg::new(&node("Conv", &["x", "w"], "y")),
            &mut values,
            &mut BTreeMap::new(),
        )
        .unwrap();
        let out = CpuBackend
            .execute(
                &g,
                values["y"],
                &HashMap::from([
                    (
                        "x".into(),
                        TensorData::new([1, 1, 2, 2], vec![1., 2., 3., 4.]).unwrap(),
                    ),
                    ("w".into(), TensorData::new([1, 1, 1, 1], vec![2.]).unwrap()),
                ]),
            )
            .unwrap();
        assert_eq!(out.values(), &[2., 4., 6., 8.]);
    }
    #[test]
    fn conv_attributes_cover_grouped_asymmetric_and_same_padding() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 2, 3, 4]);
        let w = g.input("w", [2, 1, 2, 2]);
        let mut values = BTreeMap::from([("x".into(), x), ("w".into(), w)]);
        let mut n = node("Conv", &["x", "w"], "y");
        for a in [
            int_attr("group", 2),
            ints_attr("strides", &[2, 1]),
            ints_attr("dilations", &[1, 2]),
            ints_attr("pads", &[1, 2, 0, 1]),
        ] {
            field(&mut n, 5, &a);
        }
        lower(&mut g, Msg::new(&n), &mut values, &mut BTreeMap::new()).unwrap();
        let out = CpuBackend
            .execute(
                &g,
                values["y"],
                &HashMap::from([
                    (
                        "x".into(),
                        TensorData::new([1, 2, 3, 4], vec![1.; 24]).unwrap(),
                    ),
                    (
                        "w".into(),
                        TensorData::new([2, 1, 2, 2], vec![1.; 8]).unwrap(),
                    ),
                ]),
            )
            .unwrap();
        assert_eq!(out.shape().dims(), &[1, 2, 2, 5]);

        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 3, 3]);
        let w = g.input("w", [1, 1, 2, 2]);
        let mut values = BTreeMap::from([("x".into(), x), ("w".into(), w)]);
        let mut n = node("Conv", &["x", "w"], "y");
        field(&mut n, 5, &string_attr("auto_pad", "SAME_LOWER"));
        field(&mut n, 5, &ints_attr("strides", &[2, 2]));
        lower(&mut g, Msg::new(&n), &mut values, &mut BTreeMap::new()).unwrap();
        let out = CpuBackend
            .execute(
                &g,
                values["y"],
                &HashMap::from([
                    (
                        "x".into(),
                        TensorData::new([1, 1, 3, 3], vec![1.; 9]).unwrap(),
                    ),
                    (
                        "w".into(),
                        TensorData::new([1, 1, 2, 2], vec![1.; 4]).unwrap(),
                    ),
                ]),
            )
            .unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 2, 2]);
    }
    #[test]
    fn conv_attributes_reject_bad_lengths_and_pad_conflicts() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 2, 2]);
        let w = g.input("w", [1, 1, 1, 1]);
        let mut values = BTreeMap::from([("x".into(), x), ("w".into(), w)]);
        let mut n = node("Conv", &["x", "w"], "y");
        field(&mut n, 5, &ints_attr("strides", &[1]));
        assert!(lower(&mut g, Msg::new(&n), &mut values, &mut BTreeMap::new()).is_err());
        let mut n = node("Conv", &["x", "w"], "z");
        field(&mut n, 5, &string_attr("auto_pad", "VALID"));
        field(&mut n, 5, &ints_attr("pads", &[0, 0, 0, 0]));
        assert!(lower(&mut g, Msg::new(&n), &mut values, &mut BTreeMap::new()).is_err());
    }
    #[test]
    fn batch_norm_and_global_average_pool_lower_through_cpu_graph() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 2, 2, 2]);
        let scale = g.input("scale", [2]);
        let bias = g.input("bias", [2]);
        let mean = g.input("mean", [2]);
        let variance = g.input("variance", [2]);
        let mut values = BTreeMap::from([
            ("x".into(), x),
            ("scale".into(), scale),
            ("bias".into(), bias),
            ("mean".into(), mean),
            ("variance".into(), variance),
        ]);
        let mut bn = node(
            "BatchNormalization",
            &["x", "scale", "bias", "mean", "variance"],
            "bn",
        );
        field(&mut bn, 5, &fattr("epsilon", 0.));
        lower(&mut g, Msg::new(&bn), &mut values, &mut BTreeMap::new()).unwrap();
        lower(
            &mut g,
            Msg::new(&node("Relu", &["bn"], "relu")),
            &mut values,
            &mut BTreeMap::new(),
        )
        .unwrap();
        lower(
            &mut g,
            Msg::new(&node("GlobalAveragePool", &["relu"], "y")),
            &mut values,
            &mut BTreeMap::new(),
        )
        .unwrap();
        let out = CpuBackend
            .execute(
                &g,
                values["y"],
                &HashMap::from([
                    (
                        "x".into(),
                        TensorData::new([1, 2, 2, 2], vec![1., 2., 3., 4., 5., 6., 7., 8.])
                            .unwrap(),
                    ),
                    ("scale".into(), TensorData::new([2], vec![2., 1.]).unwrap()),
                    ("bias".into(), TensorData::new([2], vec![0., -1.]).unwrap()),
                    ("mean".into(), TensorData::new([2], vec![1., 5.]).unwrap()),
                    (
                        "variance".into(),
                        TensorData::new([2], vec![1., 1.]).unwrap(),
                    ),
                ]),
            )
            .unwrap();
        assert_eq!(out.shape().dims(), &[1, 2, 1, 1]);
        assert_eq!(out.values(), &[3., 0.75]);
    }
    #[test]
    fn batch_norm_rejects_training_outputs_and_bad_parameter_contracts() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 2, 1, 1]);
        let p = g.input("p", [2]);
        let mut values = BTreeMap::from([("x".into(), x), ("p".into(), p)]);
        let mut n = node("BatchNormalization", &["x", "p", "p", "p", "p"], "y");
        field(&mut n, 5, &int_attr("training_mode", 1));
        assert!(lower(&mut g, Msg::new(&n), &mut values, &mut BTreeMap::new()).is_err());
        let mut g = Graph::new();
        let x = g.input("x", [1, 2]);
        let p = g.input("p", [1]);
        let mut values = BTreeMap::from([("x".into(), x), ("p".into(), p)]);
        assert!(
            lower(
                &mut g,
                Msg::new(&node("BatchNormalization", &["x", "p", "p", "p", "p"], "y")),
                &mut values,
                &mut BTreeMap::new()
            )
            .is_err()
        );
        assert!(
            lower(
                &mut g,
                Msg::new(&node("GlobalAveragePool", &["x"], "z")),
                &mut values,
                &mut BTreeMap::new()
            )
            .is_err()
        );
    }
    #[test]
    fn static_pools_lower_with_border_and_same_geometry() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 2, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        let mut max = node("MaxPool", &["x"], "max");
        field(&mut max, 5, &ints_attr("kernel_shape", &[2, 2]));
        lower(&mut g, Msg::new(&max), &mut values, &mut BTreeMap::new()).unwrap();
        let mut avg = node("AveragePool", &["x"], "avg");
        field(&mut avg, 5, &ints_attr("kernel_shape", &[2, 2]));
        field(&mut avg, 5, &ints_attr("pads", &[1, 1, 1, 1]));
        lower(&mut g, Msg::new(&avg), &mut values, &mut BTreeMap::new()).unwrap();
        let mut same = node("MaxPool", &["x"], "same");
        field(&mut same, 5, &ints_attr("kernel_shape", &[2, 2]));
        field(&mut same, 5, &string_attr("auto_pad", "SAME_UPPER"));
        lower(&mut g, Msg::new(&same), &mut values, &mut BTreeMap::new()).unwrap();
        let inputs = HashMap::from([(
            "x".into(),
            TensorData::new([1, 1, 2, 2], vec![1., 2., 3., 4.]).unwrap(),
        )]);
        assert_eq!(
            CpuBackend
                .execute(&g, values["max"], &inputs)
                .unwrap()
                .values(),
            &[4.]
        );
        assert_eq!(
            CpuBackend
                .execute(&g, values["avg"], &inputs)
                .unwrap()
                .values(),
            &[1., 1.5, 2., 2., 2.5, 3., 3., 3.5, 4.]
        );
        assert_eq!(
            CpuBackend
                .execute(&g, values["same"], &inputs)
                .unwrap()
                .shape()
                .dims(),
            &[1, 1, 2, 2]
        );
    }
    #[test]
    fn pools_reject_missing_bad_and_indices_contracts() {
        let mut g = Graph::new();
        let x = g.input("x", [1, 1, 2, 2]);
        let mut values = BTreeMap::from([("x".into(), x)]);
        assert!(
            lower(
                &mut g,
                Msg::new(&node("MaxPool", &["x"], "a")),
                &mut values,
                &mut BTreeMap::new()
            )
            .is_err()
        );
        let mut bad = node("AveragePool", &["x"], "b");
        field(&mut bad, 5, &ints_attr("kernel_shape", &[2, 2]));
        field(&mut bad, 5, &ints_attr("dilations", &[1, 1]));
        assert!(lower(&mut g, Msg::new(&bad), &mut values, &mut BTreeMap::new()).is_err());
        let mut indexed = node("MaxPool", &["x"], "c");
        text(&mut indexed, 2, "indices");
        field(&mut indexed, 5, &ints_attr("kernel_shape", &[2, 2]));
        assert!(
            lower(
                &mut g,
                Msg::new(&indexed),
                &mut values,
                &mut BTreeMap::new()
            )
            .is_err()
        );
    }
    #[test]
    fn static_predicates_math_clip_and_inference_dropout_lower() {
        let mut g = Graph::new();
        let x = g.input("x", [2]);
        let y = g.input("y", [2]);
        let lo = TensorData::scalar(-1.0f32);
        let hi = TensorData::scalar(1.0f32);
        let ratio = TensorData::scalar(0.0f32);
        let training = TensorData::scalar_with_dtype(crate::Scalar::Bool(false), DType::Bool);
        let mut constants = BTreeMap::from([
            ("lo".into(), lo.clone()),
            ("hi".into(), hi.clone()),
            ("ratio".into(), ratio.clone()),
            ("training".into(), training.clone()),
        ]);
        let mut values = BTreeMap::from([("x".into(), x), ("y".into(), y)]);
        for (name, value) in [
            ("lo", lo),
            ("hi", hi),
            ("ratio", ratio),
            ("training", training),
        ] {
            values.insert(name.into(), g.constant(value));
        }
        lower(
            &mut g,
            Msg::new(&node("Greater", &["x", "y"], "p")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        lower(
            &mut g,
            Msg::new(&node("Where", &["p", "x", "y"], "w")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        let mut leaky = node("LeakyRelu", &["w"], "l");
        field(&mut leaky, 5, &fattr("alpha", 0.5));
        lower(&mut g, Msg::new(&leaky), &mut values, &mut constants).unwrap();
        lower(
            &mut g,
            Msg::new(&node("Clip", &["l", "lo", "hi"], "c")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        lower(
            &mut g,
            Msg::new(&node("Dropout", &["c", "ratio", "training"], "d")),
            &mut values,
            &mut constants,
        )
        .unwrap();
        let out = CpuBackend
            .execute(
                &g,
                values["d"],
                &HashMap::from([
                    ("x".into(), TensorData::new([2], vec![-4., 2.]).unwrap()),
                    ("y".into(), TensorData::new([2], vec![3., 1.]).unwrap()),
                ]),
            )
            .unwrap();
        assert_eq!(out.values(), &[1., 1.]);
    }
    #[test]
    fn static_phase_four_rejects_dynamic_clip_and_dropout_training() {
        let mut g = Graph::new();
        let x = g.input("x", [1]);
        let b = g.input("b", []);
        let mut values = BTreeMap::from([("x".into(), x), ("b".into(), b)]);
        assert!(
            lower(
                &mut g,
                Msg::new(&node("Clip", &["x", "b"], "c")),
                &mut values,
                &mut BTreeMap::new()
            )
            .is_err()
        );
        assert!(
            lower(
                &mut g,
                Msg::new(&node("Dropout", &["x", "b"], "d")),
                &mut values,
                &mut BTreeMap::new()
            )
            .is_err()
        );
    }
}
