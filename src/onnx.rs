//! Bounded static ONNX protobuf import. This intentionally supports a small,
//! audited inference subset (opset 13, default domain) and never executes code.

use crate::{Backend, CpuBackend, DType, Error, Graph, NodeId, Result, Shape, TensorData};
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
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(1);
            let beta = attrs
                .get("beta")
                .map(|x| scalar_i64(x))
                .transpose()?
                .unwrap_or(1);
            if alpha != 1 || beta != 1 {
                return Err(bad("Gemm alpha/beta other than 1 are unsupported"));
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
            if ins.len() == 3 {
                g.add(y, get(2)?)?
            } else {
                y
            }
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
        let value = if let Some((_, 0, x)) = fields.iter().find(|(i, w, _)| *i == 3 && *w == 0) {
            x.to_vec()
        } else if let Some((_, 2, x)) = fields
            .iter()
            .find(|(i, w, _)| (*i == 5 || *i == 8) && *w == 2)
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
fn packed_i64(b: &[u8]) -> Result<Vec<i64>> {
    let mut at = 0;
    let mut x = vec![];
    while at < b.len() {
        x.push(var(b, &mut at)? as i64)
    }
    Ok(x)
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
    let dtype = match one_varint(&m, 2, "tensor dtype")? {
        1 => DType::F32,
        11 => DType::F64,
        6 => DType::I32,
        7 => DType::I64,
        9 => DType::Bool,
        x => return Err(bad(format!("unsupported ONNX tensor dtype {x}"))),
    };
    let dims = m.packed(1)?;
    let shape = Shape::new(
        dims.into_iter()
            .map(|x| {
                usize::try_from(i64::try_from(x).map_err(|_| bad("negative ONNX dimension"))?)
                    .map_err(|_| bad("ONNX dimension overflow"))
            })
            .collect::<Result<Vec<_>>>()?,
    );
    let raw = one_bytes(&m, 9, "tensor raw_data")?;
    TensorData::from_le_bytes(shape, dtype, raw).map(|x| (name, x))
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
}
