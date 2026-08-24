//! Deliberately bounded Torch state-dictionary import.
//!
//! This is **not** a Python pickle implementation.  It accepts only an
//! uncompressed ZIP archive containing `data.pkl` and CPU dense storages, and
//! interprets a small, documented pickle opcode/object whitelist needed for a
//! plain `torch.save(state_dict)` style mapping.  No GLOBAL target is ever
//! invoked: `_rebuild_tensor[_v2]` is represented as data and all other class
//! references fail closed.

use crate::{DType, Error, Result, Shape, TensorData};
use std::collections::{BTreeMap, BTreeSet};

const MAX_ARCHIVE_ENTRIES: usize = 4096;
const MAX_ARCHIVE_BYTES: usize = 256 * 1024 * 1024;
const MAX_TENSOR_BYTES: usize = 128 * 1024 * 1024;

fn err(reason: impl Into<String>) -> Error {
    Error::ModelIo {
        reason: reason.into(),
    }
}

/// Imports the safe, CPU-dense Torch ZIP subset into a deterministic state map.
///
/// Supported archives have one top-level directory, an uncompressed
/// `data.pkl`, and uncompressed `data/<storage-id>` entries.  Pickle protocol
/// 2/3/4 opcodes are accepted only when they build a string-keyed dictionary of
/// `_rebuild_tensor`/`_rebuild_tensor_v2` values with persistent CPU storages.
/// CUDA, sparse, quantized, custom objects, compressed ZIP members, TAR and
/// legacy pre-ZIP serialization are rejected before any module mutation.
pub fn load_torch_state_dict(bytes: &[u8]) -> Result<BTreeMap<String, TensorData>> {
    let files = zip_stored_files(bytes)?;
    let roots: BTreeSet<_> = files
        .keys()
        .filter_map(|name| name.split_once('/').map(|(root, _)| root))
        .collect();
    if roots.len() != 1 {
        return Err(err(
            "Torch ZIP must contain exactly one top-level directory",
        ));
    }
    let root = roots
        .into_iter()
        .next()
        .ok_or_else(|| err("Torch ZIP has no top-level directory"))?;
    let pkl_name = format!("{root}/data.pkl");
    let pickle = files
        .get(&pkl_name)
        .ok_or_else(|| err("Torch ZIP has no data.pkl"))?;
    let prefix = format!("{root}/data/");
    let mut storages = BTreeMap::new();
    for (name, data) in &files {
        if let Some(key) = name.strip_prefix(&prefix) {
            if key.is_empty() || key.contains('/') {
                return Err(err("invalid Torch storage path"));
            }
            storages.insert(key.to_owned(), data.as_slice());
        }
    }
    let root_value = Pickle::new(pickle, &storages).parse()?;
    let Value::Dict(entries) = root_value else {
        return Err(err("Torch pickle root must be a dictionary"));
    };
    let mut state = BTreeMap::new();
    for (name, value) in entries {
        let Value::Tensor(spec) = value else {
            return Err(err(format!("state entry {name:?} is not a dense tensor")));
        };
        if state
            .insert(name.clone(), tensor_from_spec(spec, &storages)?)
            .is_some()
        {
            return Err(err(format!("duplicate state key {name:?}")));
        }
    }
    Ok(state)
}

fn zip_stored_files(bytes: &[u8]) -> Result<BTreeMap<String, Vec<u8>>> {
    if bytes.len() > MAX_ARCHIVE_BYTES {
        return Err(err("archive exceeds configured byte limit"));
    }
    // EOCD has a fixed 22-byte tail plus at most 65535 bytes of comment.
    let start = bytes.len().saturating_sub(65_557);
    let eocd = bytes[start..]
        .windows(4)
        .rposition(|w| w == b"PK\x05\x06")
        .map(|i| start + i)
        .ok_or_else(|| err("not a ZIP archive with a terminal central directory"))?;
    let tail = take(bytes, eocd, 22, "truncated ZIP end record")?;
    let entries = u16le(&tail[10..12]) as usize;
    let central_size = u32le(&tail[12..16]) as usize;
    let central_start = u32le(&tail[16..20]) as usize;
    if entries > MAX_ARCHIVE_ENTRIES || u16le(&tail[8..10]) as usize != entries {
        return Err(err("unsupported ZIP multi-disk or excessive entry count"));
    }
    let central_end = central_start
        .checked_add(central_size)
        .ok_or_else(|| err("ZIP central directory overflow"))?;
    if central_end > eocd {
        return Err(err("invalid ZIP central directory bounds"));
    }
    let mut cursor = central_start;
    let mut files = BTreeMap::new();
    let mut total = 0usize;
    for _ in 0..entries {
        let fixed = take(bytes, cursor, 46, "truncated ZIP central entry")?;
        if &fixed[..4] != b"PK\x01\x02" {
            return Err(err("invalid ZIP central entry signature"));
        }
        let flags = u16le(&fixed[8..10]);
        let method = u16le(&fixed[10..12]);
        let compressed = u32le(&fixed[20..24]) as usize;
        let uncompressed = u32le(&fixed[24..28]) as usize;
        let name_len = u16le(&fixed[28..30]) as usize;
        let extra_len = u16le(&fixed[30..32]) as usize;
        let comment_len = u16le(&fixed[32..34]) as usize;
        let external = u32le(&fixed[38..42]);
        let local = u32le(&fixed[42..46]) as usize;
        if flags & 1 != 0 || method != 0 || compressed != uncompressed {
            return Err(err("only unencrypted, stored ZIP members are supported"));
        }
        // Unix symlink file-type bits are hostile even if a caller later writes files.
        if external >> 16 & 0o170000 == 0o120000 {
            return Err(err("ZIP symlink entry rejected"));
        }
        cursor = cursor
            .checked_add(46)
            .ok_or_else(|| err("ZIP cursor overflow"))?;
        let name_bytes = take(bytes, cursor, name_len, "truncated ZIP member name")?;
        let name =
            std::str::from_utf8(name_bytes).map_err(|_| err("ZIP member name is not UTF-8"))?;
        validate_path(name)?;
        cursor = cursor
            .checked_add(name_len)
            .and_then(|x| x.checked_add(extra_len))
            .and_then(|x| x.checked_add(comment_len))
            .ok_or_else(|| err("ZIP entry length overflow"))?;
        if cursor > central_end {
            return Err(err("ZIP central entry escapes directory"));
        }
        let local_fixed = take(bytes, local, 30, "truncated ZIP local entry")?;
        if &local_fixed[..4] != b"PK\x03\x04" {
            return Err(err("invalid ZIP local entry signature"));
        }
        if u16le(&local_fixed[6..8]) != flags || u16le(&local_fixed[8..10]) != method {
            return Err(err("ZIP local/central method metadata mismatch"));
        }
        let local_name_len = u16le(&local_fixed[26..28]) as usize;
        let local_extra_len = u16le(&local_fixed[28..30]) as usize;
        let local_name = take(
            bytes,
            local
                .checked_add(30)
                .ok_or_else(|| err("ZIP local name offset overflow"))?,
            local_name_len,
            "truncated ZIP local member name",
        )?;
        if local_name != name_bytes {
            return Err(err("ZIP local/central member-name mismatch"));
        }
        let data_offset = local
            .checked_add(30)
            .and_then(|x| x.checked_add(local_name_len))
            .and_then(|x| x.checked_add(local_extra_len))
            .ok_or_else(|| err("ZIP data offset overflow"))?;
        let data = take(
            bytes,
            data_offset,
            uncompressed,
            "truncated ZIP member data",
        )?;
        total = total
            .checked_add(data.len())
            .ok_or_else(|| err("ZIP size overflow"))?;
        if total > MAX_ARCHIVE_BYTES {
            return Err(err("archive member bytes exceed configured limit"));
        }
        if files.insert(name.to_owned(), data.to_vec()).is_some() {
            return Err(err("duplicate ZIP member name"));
        }
    }
    if cursor != central_end {
        return Err(err("ambiguous trailing central-directory data"));
    }
    Ok(files)
}

fn validate_path(name: &str) -> Result<()> {
    if name.is_empty()
        || name.starts_with('/')
        || name.contains('\\')
        || name
            .split('/')
            .any(|p| p.is_empty() || p == "." || p == "..")
    {
        return Err(err("ZIP path traversal or non-canonical member path"));
    }
    Ok(())
}
fn take<'a>(b: &'a [u8], off: usize, len: usize, message: &'static str) -> Result<&'a [u8]> {
    b.get(off..off.checked_add(len).ok_or_else(|| err(message))?)
        .ok_or_else(|| err(message))
}
fn u16le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}
fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[derive(Clone, Debug)]
enum Value {
    None,
    Bool,
    Int(i64),
    Str(String),
    Symbol(String, String),
    Tuple(Vec<Value>),
    Dict(BTreeMap<String, Value>),
    Storage(StorageRef),
    Tensor(TensorSpec),
}
#[derive(Clone, Debug)]
struct StorageRef {
    key: String,
    dtype: DType,
    elements: usize,
}
#[derive(Clone, Debug)]
struct TensorSpec {
    storage: StorageRef,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

struct Pickle<'a> {
    bytes: &'a [u8],
    at: usize,
    stack: Vec<Value>,
    marks: Vec<usize>,
    memo: BTreeMap<usize, Value>,
    storages: &'a BTreeMap<String, &'a [u8]>,
}
impl<'a> Pickle<'a> {
    fn new(bytes: &'a [u8], storages: &'a BTreeMap<String, &'a [u8]>) -> Self {
        Self {
            bytes,
            at: 0,
            stack: vec![],
            marks: vec![],
            memo: BTreeMap::new(),
            storages,
        }
    }
    fn parse(mut self) -> Result<Value> {
        if self.byte()? != 0x80 {
            return Err(err("Torch pickle must start with PROTO"));
        }
        let protocol = self.byte()?;
        if protocol != 2 {
            return Err(err("unsupported Torch pickle protocol"));
        }
        loop {
            match self.byte()? {
                b'.' => {
                    if self.at != self.bytes.len() || self.stack.len() != 1 {
                        return Err(err("ambiguous trailing pickle data"));
                    }
                    return self.pop();
                }
                b'(' => self.marks.push(self.stack.len()),
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b'}' => self.stack.push(Value::Dict(BTreeMap::new())),
                b'N' => self.stack.push(Value::None),
                0x88 | 0x89 => self.stack.push(Value::Bool),
                b'K' => {
                    let value = self.byte()? as i64;
                    self.stack.push(Value::Int(value));
                }
                b'M' => {
                    let value = u16le(self.read(2)?) as i64;
                    self.stack.push(Value::Int(value));
                }
                b'J' => {
                    let value = i32::from_le_bytes(
                        self.read(4)?.try_into().map_err(|_| err("bad BININT"))?,
                    ) as i64;
                    self.stack.push(Value::Int(value));
                }
                b'X' => {
                    let n = u32le(self.read(4)?) as usize;
                    self.string(n)?;
                }
                0x8c => {
                    let n = self.byte()? as usize;
                    self.string(n)?;
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(Value::Symbol(module, name));
                }
                b'q' => {
                    let i = self.byte()? as usize;
                    self.memoize(i)?;
                }
                b'r' => {
                    let i = u32le(self.read(4)?) as usize;
                    self.memoize(i)?;
                }
                b'h' => {
                    let i = self.byte()? as usize;
                    self.getmemo(i)?;
                }
                b'j' => {
                    let i = u32le(self.read(4)?) as usize;
                    self.getmemo(i)?;
                }
                b't' => {
                    let values = self.pop_mark()?;
                    self.stack.push(Value::Tuple(values));
                }
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a]));
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b]));
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b, c]));
                }
                b'Q' => {
                    let pid = self.pop()?;
                    self.stack.push(self.persistent(pid)?);
                }
                b'R' => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    self.stack.push(self.reduce(callable, args)?);
                }
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop_string()?;
                    match self.stack.last_mut() {
                        Some(Value::Dict(map)) => {
                            if map.insert(key, value).is_some() {
                                return Err(err("duplicate pickle dictionary key"));
                            }
                        }
                        _ => return Err(err("SETITEM without dictionary")),
                    }
                }
                b'u' => {
                    let values = self.pop_mark()?;
                    let dict = match self.stack.last_mut() {
                        Some(Value::Dict(x)) => x,
                        _ => return Err(err("SETITEMS without dictionary")),
                    };
                    if values.len() % 2 != 0 {
                        return Err(err("odd SETITEMS payload"));
                    }
                    for pair in values.chunks_exact(2) {
                        let key = value_string(&pair[0])?;
                        if dict.insert(key, pair[1].clone()).is_some() {
                            return Err(err("duplicate pickle dictionary key"));
                        }
                    }
                }
                b'0' => {
                    self.pop()?;
                }
                b'1' => {
                    let mark = *self
                        .marks
                        .last()
                        .ok_or_else(|| err("POP_MARK without MARK"))?;
                    self.stack.truncate(mark);
                    self.marks.pop();
                }
                op => {
                    return Err(err(format!(
                        "pickle opcode 0x{op:02x} is not in the Torch state-dict whitelist"
                    )));
                }
            }
        }
    }
    fn byte(&mut self) -> Result<u8> {
        let x = *self
            .bytes
            .get(self.at)
            .ok_or_else(|| err("truncated pickle"))?;
        self.at += 1;
        Ok(x)
    }
    fn read(&mut self, n: usize) -> Result<&'a [u8]> {
        let x = take(self.bytes, self.at, n, "truncated pickle")?;
        self.at += n;
        Ok(x)
    }
    fn line(&mut self) -> Result<String> {
        let rest = &self.bytes[self.at..];
        let n = rest
            .iter()
            .position(|&x| x == b'\n')
            .ok_or_else(|| err("unterminated pickle GLOBAL"))?;
        self.at += n + 1;
        std::str::from_utf8(&rest[..n])
            .map(|x| x.to_owned())
            .map_err(|_| err("pickle string is not UTF-8"))
    }
    fn string(&mut self, n: usize) -> Result<()> {
        let b = self.read(n)?;
        let x = std::str::from_utf8(b)
            .map_err(|_| err("pickle string is not UTF-8"))?
            .to_owned();
        self.stack.push(Value::Str(x));
        Ok(())
    }
    fn pop(&mut self) -> Result<Value> {
        self.stack
            .pop()
            .ok_or_else(|| err("pickle stack underflow"))
    }
    fn pop_string(&mut self) -> Result<String> {
        value_string(&self.pop()?)
    }
    fn pop_mark(&mut self) -> Result<Vec<Value>> {
        let mark = self
            .marks
            .pop()
            .ok_or_else(|| err("pickle MARK underflow"))?;
        Ok(self.stack.split_off(mark))
    }
    fn memoize(&mut self, key: usize) -> Result<()> {
        let v = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| err("pickle memo stack underflow"))?;
        if self.memo.insert(key, v).is_some() {
            return Err(err("duplicate pickle memo index"));
        }
        Ok(())
    }
    fn getmemo(&mut self, key: usize) -> Result<()> {
        self.stack.push(
            self.memo
                .get(&key)
                .cloned()
                .ok_or_else(|| err("unknown pickle memo index"))?,
        );
        Ok(())
    }
    fn persistent(&self, pid: Value) -> Result<Value> {
        let Value::Tuple(v) = pid else {
            return Err(err("persistent id is not a tuple"));
        };
        if v.len() != 5 || value_string(&v[0])? != "storage" || value_string(&v[3])? != "cpu" {
            return Err(err("only CPU Torch storage persistent ids are supported"));
        }
        let Value::Symbol(module, name) = &v[1] else {
            return Err(err("invalid Torch storage type"));
        };
        let dtype = storage_dtype(module, name)?;
        let key = value_string(&v[2])?;
        let elements = value_usize(&v[4], "storage size")?;
        let raw = self
            .storages
            .get(&key)
            .ok_or_else(|| err(format!("missing Torch storage {key:?}")))?;
        let bytes = elements
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| err("storage byte size overflow"))?;
        if raw.len() != bytes {
            return Err(err(format!("storage {key:?} has wrong byte length")));
        }
        Ok(Value::Storage(StorageRef {
            key,
            dtype,
            elements,
        }))
    }
    fn reduce(&self, callable: Value, args: Value) -> Result<Value> {
        let Value::Symbol(module, name) = callable else {
            return Err(err("pickle REDUCE callable is not a whitelisted symbol"));
        };
        if module == "collections" && name == "OrderedDict" {
            if !matches!(args, Value::Tuple(ref values) if values.is_empty()) {
                return Err(err("OrderedDict constructor arguments are unsupported"));
            }
            return Ok(Value::Dict(BTreeMap::new()));
        }
        if module != "torch._utils" || !(name == "_rebuild_tensor" || name == "_rebuild_tensor_v2")
        {
            return Err(err(format!("pickle GLOBAL {module}.{name} is not allowed")));
        }
        let Value::Tuple(v) = args else {
            return Err(err("tensor rebuild arguments are not a tuple"));
        };
        if v.len() < 4 {
            return Err(err("tensor rebuild has too few arguments"));
        }
        let Value::Storage(storage) = &v[0] else {
            return Err(err("tensor rebuild does not reference CPU storage"));
        };
        let offset = value_usize(&v[1], "storage offset")?;
        let shape = tuple_usizes(&v[2], "tensor shape")?;
        let strides = tuple_usizes(&v[3], "tensor stride")?;
        if shape.len() != strides.len() {
            return Err(err("tensor shape/stride ranks differ"));
        }
        Ok(Value::Tensor(TensorSpec {
            storage: storage.clone(),
            offset,
            shape,
            strides,
        }))
    }
}
fn value_string(v: &Value) -> Result<String> {
    match v {
        Value::Str(x) => Ok(x.clone()),
        _ => Err(err("pickle value must be a string")),
    }
}
fn value_usize(v: &Value, what: &'static str) -> Result<usize> {
    match v {
        Value::Int(x) if *x >= 0 => {
            usize::try_from(*x).map_err(|_| err(format!("{what} does not fit usize")))
        }
        _ => Err(err(format!("{what} must be a non-negative integer"))),
    }
}
fn tuple_usizes(v: &Value, what: &'static str) -> Result<Vec<usize>> {
    let Value::Tuple(xs) = v else {
        return Err(err(format!("{what} must be a tuple")));
    };
    xs.iter().map(|x| value_usize(x, what)).collect()
}
fn storage_dtype(module: &str, name: &str) -> Result<DType> {
    if module != "torch" {
        return Err(err("unsupported Torch storage module"));
    }
    Ok(match name {
        "BoolStorage" => DType::Bool,
        "CharStorage" => DType::I8,
        "ByteStorage" => DType::U8,
        "ShortStorage" => DType::I16,
        "IntStorage" => DType::I32,
        "LongStorage" => DType::I64,
        "HalfStorage" => DType::F16,
        "BFloat16Storage" => DType::BF16,
        "FloatStorage" => DType::F32,
        "DoubleStorage" => DType::F64,
        _ => return Err(err(format!("unsupported Torch storage type {name}"))),
    })
}

fn tensor_from_spec(spec: TensorSpec, storages: &BTreeMap<String, &[u8]>) -> Result<TensorData> {
    let shape = Shape::new(spec.shape.clone());
    let count = shape.numel()?;
    let out_bytes = count
        .checked_mul(spec.storage.dtype.itemsize())
        .ok_or_else(|| err("tensor byte length overflow"))?;
    if out_bytes > MAX_TENSOR_BYTES {
        return Err(err("tensor exceeds configured byte limit"));
    }
    let raw = storages
        .get(&spec.storage.key)
        .ok_or_else(|| err("storage disappeared during parse"))?;
    if count == 0 {
        return TensorData::from_le_bytes(shape, spec.storage.dtype, &[]);
    }
    let mut max = spec.offset;
    for (&dim, &stride) in spec.shape.iter().zip(&spec.strides) {
        if dim > 0 {
            max = max
                .checked_add(
                    (dim - 1)
                        .checked_mul(stride)
                        .ok_or_else(|| err("tensor stride offset overflow"))?,
                )
                .ok_or_else(|| err("tensor stride offset overflow"))?;
        }
    }
    if max >= spec.storage.elements {
        return Err(err("tensor view exceeds storage"));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(out_bytes)
        .map_err(|_| err("tensor allocation failed"))?;
    let mut seen = BTreeSet::new();
    for linear in 0..count {
        let mut rem = linear;
        let mut off = spec.offset;
        for axis in (0..spec.shape.len()).rev() {
            let dim = spec.shape[axis];
            let coord = rem % dim;
            rem /= dim;
            off = off
                .checked_add(
                    coord
                        .checked_mul(spec.strides[axis])
                        .ok_or_else(|| err("tensor offset overflow"))?,
                )
                .ok_or_else(|| err("tensor offset overflow"))?;
        }
        if !seen.insert(off) {
            return Err(err("overlapping Torch tensor strides are unsupported"));
        }
        let b = off
            .checked_mul(spec.storage.dtype.itemsize())
            .ok_or_else(|| err("tensor byte offset overflow"))?;
        out.extend_from_slice(take(
            raw,
            b,
            spec.storage.dtype.itemsize(),
            "truncated Torch storage",
        )?);
    }
    TensorData::from_le_bytes(shape, spec.storage.dtype, &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CastPolicy, Graph, Metadata, Module, ModuleStateDict, load_safetensors, nn::Linear,
        save_safetensors,
    };

    // This fixture writer is intentionally a tiny external-format encoder, not
    // a call to the importer. It emits protocol-2 pickle and stored ZIP bytes.
    fn fixture(data: &[u8], dtype: &str, shape: &[u8], strides: &[u8], offset: u8) -> Vec<u8> {
        let mut p = vec![0x80, 2, b'c'];
        p.extend_from_slice(b"collections\nOrderedDict\n");
        p.extend_from_slice(b")RX");
        p.extend_from_slice(&(6u32).to_le_bytes());
        p.extend_from_slice(b"weight");
        p.push(b'c');
        p.extend_from_slice(b"torch._utils\n_rebuild_tensor_v2\n");
        p.push(b'(');
        p.push(b'(');
        p.extend_from_slice(b"X");
        p.extend_from_slice(&(7u32).to_le_bytes());
        p.extend_from_slice(b"storage");
        p.push(b'c');
        p.extend_from_slice(b"torch\n");
        p.extend_from_slice(dtype.as_bytes());
        p.push(b'\n');
        p.extend_from_slice(b"X");
        p.extend_from_slice(&(1u32).to_le_bytes());
        p.extend_from_slice(b"0");
        p.extend_from_slice(b"X");
        p.extend_from_slice(&(3u32).to_le_bytes());
        p.extend_from_slice(b"cpu");
        let itemsize = match dtype {
            "BoolStorage" | "CharStorage" | "ByteStorage" => 1,
            "ShortStorage" | "HalfStorage" | "BFloat16Storage" => 2,
            "IntStorage" | "FloatStorage" => 4,
            "LongStorage" | "DoubleStorage" => 8,
            _ => 1,
        };
        p.extend_from_slice(&[
            b'K',
            (data.len() / itemsize) as u8,
            b't',
            b'Q',
            b'K',
            offset,
            b'(',
        ]);
        for &d in shape {
            p.extend_from_slice(&[b'K', d]);
        }
        p.push(b't');
        p.push(b'(');
        for &s in strides {
            p.extend_from_slice(&[b'K', s]);
        }
        p.extend_from_slice(&[b't', 0x89, b't', b'R', b's', b'.']);
        zip(&[("archive/data.pkl", p), ("archive/data/0", data.to_vec())])
    }
    fn zip(files: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = vec![];
        let mut central = vec![];
        for (name, data) in files {
            let off = out.len() as u32;
            out.extend_from_slice(b"PK\x03\x04");
            out.extend_from_slice(&20u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&[0; 4]);
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(data);
            central.extend_from_slice(b"PK\x01\x02");
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&[0; 4]);
            central.extend_from_slice(&0u32.to_le_bytes());
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&[0; 6]);
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u32.to_le_bytes());
            central.extend_from_slice(&off.to_le_bytes());
            central.extend_from_slice(name.as_bytes());
        }
        let begin = out.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(b"PK\x05\x06");
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&begin.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    #[test]
    fn torch_zip_state_dict_reconstructs_bits_and_strides() {
        let raw = [
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00,
            0x80, 0x40,
        ];
        let state =
            load_torch_state_dict(&fixture(&raw, "FloatStorage", &[2, 2], &[1, 2], 0)).unwrap();
        let weight = state.get("weight").unwrap();
        assert_eq!(weight.shape().dims(), &[2, 2]);
        assert_eq!(weight.dtype(), DType::F32);
        assert_eq!(
            weight.to_le_bytes().unwrap(),
            [
                raw[0..4].to_vec(),
                raw[8..12].to_vec(),
                raw[4..8].to_vec(),
                raw[12..16].to_vec()
            ]
            .concat()
        );
        let safe = save_safetensors(&state, &Metadata::new()).unwrap();
        assert_eq!(load_safetensors(&safe).unwrap().0, state);

        let mut graph = Graph::new();
        let linear = Linear::new(&mut graph, 2, 2, false, 7).unwrap();
        let report = linear
            .load_state_dict(
                &ModuleStateDict::from(state.clone()),
                true,
                CastPolicy::Exact,
            )
            .unwrap();
        assert_eq!(report.loaded_keys, ["weight"]);
        assert_eq!(linear.state_dict().unwrap().into_tensors(), state);

        let narrow = load_torch_state_dict(&fixture(
            &[0x01, 0x7e, 0x00, 0x80],
            "HalfStorage",
            &[2],
            &[1],
            0,
        ))
        .unwrap();
        assert_eq!(narrow["weight"].dtype(), DType::F16);
        assert_eq!(
            narrow["weight"].to_le_bytes().unwrap(),
            [0x01, 0x7e, 0x00, 0x80]
        );
    }
    #[test]
    fn torch_import_rejects_hostile_archive_and_tensor_cases() {
        let raw = vec![0; 16];
        let good = fixture(&raw, "FloatStorage", &[2, 2], &[2, 1], 0);
        let cases = [
            (
                "traversal",
                zip(&[("../data.pkl", vec![0x80, 2, b'}', b'.'])]),
            ),
            ("compressed flag", {
                let mut x = good.clone();
                x[6] = 8;
                x
            }),
            (
                "unsupported pickle",
                zip(&[
                    (
                        "archive/data.pkl",
                        vec![
                            0x80, 2, b'c', b'o', b's', b'\n', b's', b'y', b's', b't', b'e', b'm',
                            b'\n', b'.',
                        ],
                    ),
                    ("archive/data/0", raw.clone()),
                ]),
            ),
            (
                "overlap",
                fixture(&raw, "FloatStorage", &[2, 2], &[0, 1], 0),
            ),
        ];
        for (name, bytes) in cases {
            assert!(
                matches!(load_torch_state_dict(&bytes), Err(Error::ModelIo { .. })),
                "{name}"
            );
        }
    }
}
