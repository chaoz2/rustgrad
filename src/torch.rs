//! Deliberately bounded Torch state-dictionary import.
//!
//! This is **not** a Python pickle implementation.  It accepts only an
//! uncompressed ZIP archive containing `data.pkl` and CPU dense storages, and
//! interprets a small, documented pickle opcode/object whitelist needed for a
//! plain `torch.save(state_dict)` style mapping.  No GLOBAL target is ever
//! invoked: `_rebuild_tensor[_v2]` is represented as data and all other class
//! references fail closed.

use crate::{DType, Error, Result, Shape, TensorData};
use flate2::read::DeflateDecoder;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

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
/// Supported archives have one top-level directory, stored or raw-deflate
/// `data.pkl`, and CPU `data/<storage-id>` entries. Pickle protocol 2 opcodes
/// are accepted only when they build a string-keyed dictionary of
/// `_rebuild_tensor`/`_rebuild_tensor_v2` values with persistent CPU storages.
/// CUDA, sparse, quantized, custom objects, ZIP64, TAR and legacy pre-ZIP
/// serialization are rejected before any module mutation. See
/// `load_legacy_torch_state_dict` for the separate bounded TAR subset.
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

/// Imports the bounded legacy (pre-ZIP) Torch TAR state-dictionary subset.
///
/// This accepts only a regular-file ustar archive with exact `storages`,
/// `tensors`, and `pickle` streams. All pickle evaluation is data-only: the
/// stateful protocol-2 VM permits persistent references to the typed tensor
/// registry and inert `torch.nn.parameter.Parameter` wrappers, never code.
pub fn load_legacy_torch_state_dict(bytes: &[u8]) -> Result<BTreeMap<String, TensorData>> {
    let files = extract_tar_files(bytes)?;
    if files.len() != 3
        || !files.contains_key("storages")
        || !files.contains_key("tensors")
        || !files.contains_key("pickle")
    {
        return Err(err(
            "legacy Torch TAR must contain only storages, tensors, and pickle",
        ));
    }
    let storages = legacy_storages(&files["storages"])?;
    let tensors = legacy_tensors(&files["tensors"], &storages)?;
    let registry = tensors
        .into_iter()
        .map(|(key, data)| (key, Value::Data(data)))
        .collect();
    let mut pickle = LegacyPickle::new(&files["pickle"], &registry);
    let Value::Dict(entries) = pickle.next()? else {
        return Err(err("legacy Torch pickle root must be a dictionary"));
    };
    if pickle.at != files["pickle"].len() {
        return Err(err("trailing legacy Torch pickle records"));
    }
    let mut state = BTreeMap::new();
    for (key, value) in entries {
        let data = match value {
            Value::Data(data) => data,
            Value::Parameter(Some(inner)) => match *inner {
                Value::Data(data) => data,
                _ => return Err(err("legacy Parameter does not wrap a tensor")),
            },
            _ => return Err(err(format!("legacy state entry {key:?} is not a tensor"))),
        };
        if state.insert(key.clone(), data).is_some() {
            return Err(err(format!("duplicate legacy state key {key:?}")));
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
    let (entries, central_start, central_size) = zip_directory(bytes, eocd, tail)?;
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
        let crc = u32le(&fixed[16..20]);
        let raw_compressed = u32le(&fixed[20..24]);
        let raw_uncompressed = u32le(&fixed[24..28]);
        let name_len = u16le(&fixed[28..30]) as usize;
        let extra_len = u16le(&fixed[30..32]) as usize;
        let comment_len = u16le(&fixed[32..34]) as usize;
        let external = u32le(&fixed[38..42]);
        let raw_local = u32le(&fixed[42..46]);
        if flags & 0b1001 != 0 || !matches!(method, 0 | 8) {
            return Err(err(
                "only stored or raw-deflate ZIP members without encryption or data descriptors are supported",
            ));
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
        let extra_start = cursor
            .checked_add(name_len)
            .ok_or_else(|| err("ZIP extra offset overflow"))?;
        let extra = take(bytes, extra_start, extra_len, "truncated ZIP extra field")?;
        let (compressed, uncompressed, local) =
            zip64_entry_values(raw_compressed, raw_uncompressed, raw_local, extra)?;
        if uncompressed > MAX_ARCHIVE_BYTES
            || (compressed != 0 && uncompressed / compressed > 1000)
            || (compressed == 0 && uncompressed != 0)
        {
            return Err(err(
                "ZIP member exceeds configured decompression-ratio limit",
            ));
        }
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
        let compressed_data = take(bytes, data_offset, compressed, "truncated ZIP member data")?;
        total = total
            .checked_add(uncompressed)
            .ok_or_else(|| err("ZIP size overflow"))?;
        if total > MAX_ARCHIVE_BYTES {
            return Err(err("archive member bytes exceed configured limit"));
        }
        let data = decode_zip_member(method, compressed_data, uncompressed)?;
        if crc32(&data) != crc {
            return Err(err("ZIP member CRC mismatch"));
        }
        if files.insert(name.to_owned(), data).is_some() {
            return Err(err("duplicate ZIP member name"));
        }
    }
    if cursor != central_end {
        return Err(err("ambiguous trailing central-directory data"));
    }
    Ok(files)
}

fn zip_directory(bytes: &[u8], eocd: usize, tail: &[u8]) -> Result<(usize, usize, usize)> {
    let entries16 = u16le(&tail[10..12]);
    let size32 = u32le(&tail[12..16]);
    let offset32 = u32le(&tail[16..20]);
    if u16le(&tail[4..6]) != 0 || u16le(&tail[6..8]) != 0 {
        return Err(err("ZIP multi-disk archives are unsupported"));
    }
    let needs_zip64 = entries16 == u16::MAX || size32 == u32::MAX || offset32 == u32::MAX;
    if !needs_zip64 {
        if u16le(&tail[8..10]) != entries16 {
            return Err(err("ZIP central-directory entry counts disagree"));
        }
        let entries = usize::from(entries16);
        if entries > MAX_ARCHIVE_ENTRIES {
            return Err(err("ZIP entry count exceeds configured limit"));
        }
        return Ok((entries, offset32 as usize, size32 as usize));
    }
    let locator_at = eocd
        .checked_sub(20)
        .ok_or_else(|| err("missing ZIP64 locator"))?;
    let locator = take(bytes, locator_at, 20, "truncated ZIP64 locator")?;
    if &locator[..4] != b"PK\x06\x07" || u32le(&locator[4..8]) != 0 || u32le(&locator[16..20]) != 1
    {
        return Err(err("invalid or multi-disk ZIP64 locator"));
    }
    let record_at = usize::try_from(u64le(&locator[8..16]))
        .map_err(|_| err("ZIP64 record offset overflows usize"))?;
    let record = take(bytes, record_at, 56, "truncated ZIP64 end record")?;
    if &record[..4] != b"PK\x06\x06"
        || u64le(&record[4..12]) < 44
        || u32le(&record[16..20]) != 0
        || u32le(&record[20..24]) != 0
    {
        return Err(err("invalid ZIP64 end record"));
    }
    let disk_entries = u64le(&record[24..32]);
    let entries = u64le(&record[32..40]);
    if disk_entries != entries {
        return Err(err("ZIP64 multi-disk entry counts are unsupported"));
    }
    let entries = usize::try_from(entries).map_err(|_| err("ZIP64 entry count overflows usize"))?;
    let size = usize::try_from(u64le(&record[40..48]))
        .map_err(|_| err("ZIP64 central size overflows usize"))?;
    let offset = usize::try_from(u64le(&record[48..56]))
        .map_err(|_| err("ZIP64 central offset overflows usize"))?;
    if entries > MAX_ARCHIVE_ENTRIES {
        return Err(err("ZIP64 entry count exceeds configured limit"));
    }
    Ok((entries, offset, size))
}

fn zip64_entry_values(
    raw_compressed: u32,
    raw_uncompressed: u32,
    raw_local: u32,
    extra: &[u8],
) -> Result<(usize, usize, usize)> {
    let needed = [
        raw_uncompressed == u32::MAX,
        raw_compressed == u32::MAX,
        raw_local == u32::MAX,
    ];
    if !needed.iter().any(|x| *x) {
        if extra.windows(2).any(|x| x == [1, 0]) {
            return Err(err("unexpected ZIP64 extra field"));
        }
        return Ok((
            raw_compressed as usize,
            raw_uncompressed as usize,
            raw_local as usize,
        ));
    }
    let mut at = 0usize;
    let mut field = None;
    while at < extra.len() {
        let head = take(extra, at, 4, "truncated ZIP extra header")?;
        let id = u16le(&head[..2]);
        let len = u16le(&head[2..4]) as usize;
        at = at
            .checked_add(4)
            .ok_or_else(|| err("ZIP extra offset overflow"))?;
        let data = take(extra, at, len, "truncated ZIP extra data")?;
        at = at
            .checked_add(len)
            .ok_or_else(|| err("ZIP extra offset overflow"))?;
        if id == 1 && field.replace(data).is_some() {
            return Err(err("duplicate ZIP64 extra field"));
        }
    }
    let data = field.ok_or_else(|| err("ZIP64 fields require one ZIP64 extra field"))?;
    let mut at = 0usize;
    let mut next = || -> Result<usize> {
        let value = usize::try_from(u64le(take(data, at, 8, "truncated ZIP64 extra value")?))
            .map_err(|_| err("ZIP64 value overflows usize"))?;
        at += 8;
        Ok(value)
    };
    let uncompressed = if needed[0] {
        next()?
    } else {
        raw_uncompressed as usize
    };
    let compressed = if needed[1] {
        next()?
    } else {
        raw_compressed as usize
    };
    let local = if needed[2] {
        next()?
    } else {
        raw_local as usize
    };
    if at != data.len() {
        return Err(err("ambiguous trailing ZIP64 extra data"));
    }
    Ok((compressed, uncompressed, local))
}

fn decode_zip_member(method: u16, input: &[u8], expected: usize) -> Result<Vec<u8>> {
    if method == 0 {
        if input.len() != expected {
            return Err(err("stored ZIP member size mismatch"));
        }
        return Ok(input.to_vec());
    }
    let mut decoder = DeflateDecoder::new(input);
    let mut output = Vec::new();
    output
        .try_reserve_exact(expected)
        .map_err(|_| err("ZIP decompression allocation failed"))?;
    let mut chunk = [0u8; 8192];
    loop {
        let count = decoder
            .read(&mut chunk)
            .map_err(|_| err("invalid raw-deflate ZIP member"))?;
        if count == 0 {
            break;
        }
        if output
            .len()
            .checked_add(count)
            .ok_or_else(|| err("ZIP decompression size overflow"))?
            > expected
        {
            return Err(err("ZIP deflate output exceeds advertised size"));
        }
        output.extend_from_slice(&chunk[..count]);
    }
    if output.len() != expected {
        return Err(err("ZIP deflate output does not match advertised size"));
    }
    Ok(output)
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (!((crc & 1).wrapping_sub(1))));
        }
    }
    !crc
}

/// Safely reads the regular-file subset of a POSIX ustar archive in memory.
///
/// No member is written to disk. Directories, links, devices, PAX/GNU extension
/// records, duplicate/path-traversal names, malformed checksums, and truncated
/// padding are rejected rather than interpreted permissively.
pub fn extract_tar_files(bytes: &[u8]) -> Result<BTreeMap<String, Vec<u8>>> {
    if bytes.len() > MAX_ARCHIVE_BYTES {
        return Err(err("TAR archive exceeds configured byte limit"));
    }
    let mut files = BTreeMap::new();
    let mut at = 0usize;
    let mut total = 0usize;
    let mut zero_blocks = 0usize;
    while at < bytes.len() {
        let header = take(bytes, at, 512, "truncated TAR header")?;
        if header.iter().all(|&b| b == 0) {
            zero_blocks += 1;
            at = at
                .checked_add(512)
                .ok_or_else(|| err("TAR offset overflow"))?;
            if zero_blocks == 2 {
                if bytes[at..].iter().any(|&b| b != 0) {
                    return Err(err("ambiguous trailing TAR bytes"));
                }
                return Ok(files);
            }
            continue;
        }
        zero_blocks = 0;
        validate_tar_checksum(header)?;
        let name = tar_text(&header[..100], "TAR member name")?;
        let prefix = tar_text(&header[345..500], "TAR member prefix")?;
        let name = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        validate_path(&name)?;
        let kind = header[156];
        if !matches!(kind, 0 | b'0') {
            return Err(err("TAR contains a non-regular member"));
        }
        let size = tar_octal(&header[124..136], "TAR member size")?;
        let data_at = at
            .checked_add(512)
            .ok_or_else(|| err("TAR data offset overflow"))?;
        let data = take(bytes, data_at, size, "truncated TAR member data")?;
        total = total
            .checked_add(size)
            .ok_or_else(|| err("TAR size overflow"))?;
        if files.len() >= MAX_ARCHIVE_ENTRIES || total > MAX_ARCHIVE_BYTES {
            return Err(err("TAR exceeds configured member count or size limit"));
        }
        if files.insert(name, data.to_vec()).is_some() {
            return Err(err("duplicate TAR member name"));
        }
        let padded = size
            .checked_add(511)
            .ok_or_else(|| err("TAR padding overflow"))?
            / 512
            * 512;
        at = data_at
            .checked_add(padded)
            .ok_or_else(|| err("TAR offset overflow"))?;
        if at > bytes.len() {
            return Err(err("truncated TAR member padding"));
        }
    }
    Err(err("TAR lacks the required two zero end blocks"))
}

fn validate_tar_checksum(header: &[u8]) -> Result<()> {
    let expected = tar_octal(&header[148..156], "TAR checksum")?;
    let actual: usize = header
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            if (148..156).contains(&i) {
                usize::from(b' ')
            } else {
                usize::from(b)
            }
        })
        .sum();
    if expected != actual {
        return Err(err("TAR header checksum mismatch"));
    }
    Ok(())
}
fn tar_text(field: &[u8], what: &'static str) -> Result<String> {
    let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
    std::str::from_utf8(&field[..end])
        .map(|x| x.to_owned())
        .map_err(|_| err(format!("{what} is not UTF-8")))
}
fn tar_octal(field: &[u8], what: &'static str) -> Result<usize> {
    let text = std::str::from_utf8(field).map_err(|_| err(format!("{what} is not ASCII octal")))?;
    let text = text.trim_matches(['\0', ' ']);
    if text.is_empty() {
        return Ok(0);
    }
    usize::from_str_radix(text, 8).map_err(|_| err(format!("invalid {what}")))
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
fn u64le(b: &[u8]) -> u64 {
    u64::from_le_bytes(b.try_into().expect("eight-byte slice"))
}

#[allow(dead_code)]
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
    Data(TensorData),
    Parameter(Option<Box<Value>>),
}
#[derive(Clone, Debug)]
struct StorageRef {
    key: String,
    dtype: DType,
    elements: usize,
    /// Canonical little-endian storage bytes retained for legacy TAR tensor views.
    raw: Vec<u8>,
}
#[derive(Clone, Debug)]
struct TensorSpec {
    storage: StorageRef,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

/// Stateful protocol-2 record reader for the legacy Torch TAR streams.
///
/// This deliberately remains private until the three stream framings are wired.
/// Each `next` starts a fresh pickle object at the shared byte cursor, retaining
/// only the caller-owned typed registry between objects. The sole object build
/// operation is the inert Torch `Parameter` wrapper with a one-element tuple.
#[allow(dead_code)]
struct LegacyPickle<'a> {
    bytes: &'a [u8],
    at: usize,
    registry: &'a BTreeMap<String, Value>,
}
#[allow(dead_code)]
impl<'a> LegacyPickle<'a> {
    fn new(bytes: &'a [u8], registry: &'a BTreeMap<String, Value>) -> Self {
        Self {
            bytes,
            at: 0,
            registry,
        }
    }
    fn next(&mut self) -> Result<Value> {
        let mut stack = Vec::new();
        let mut marks = Vec::new();
        let mut memo = BTreeMap::new();
        if self.byte()? != 0x80 || self.byte()? != 2 {
            return Err(err("legacy pickle must use protocol 2"));
        }
        loop {
            match self.byte()? {
                b'.' => {
                    if stack.len() != 1 || !marks.is_empty() {
                        return Err(err("malformed legacy pickle stack"));
                    }
                    return stack
                        .pop()
                        .ok_or_else(|| err("legacy pickle stack underflow"));
                }
                b'(' => marks.push(stack.len()),
                b')' => stack.push(Value::Tuple(vec![])),
                b'}' => stack.push(Value::Dict(BTreeMap::new())),
                b'N' => stack.push(Value::None),
                0x88 | 0x89 => stack.push(Value::Bool),
                b'K' => stack.push(Value::Int(self.byte()? as i64)),
                b'M' => stack.push(Value::Int(u16le(self.read(2)?) as i64)),
                b'J' => {
                    let x = i32::from_le_bytes(
                        self.read(4)?
                            .try_into()
                            .map_err(|_| err("bad legacy BININT"))?,
                    ) as i64;
                    stack.push(Value::Int(x));
                }
                b'X' => {
                    let n = u32le(self.read(4)?) as usize;
                    stack.push(Value::Str(self.utf8(n)?));
                }
                b'c' => {
                    let m = self.line()?;
                    let n = self.line()?;
                    stack.push(Value::Symbol(m, n));
                }
                b'q' => {
                    let i = self.byte()? as usize;
                    legacy_memo(&mut memo, i, &stack)?;
                }
                b'r' => {
                    let i = u32le(self.read(4)?) as usize;
                    legacy_memo(&mut memo, i, &stack)?;
                }
                b'h' => {
                    let i = self.byte()? as usize;
                    stack.push(
                        memo.get(&i)
                            .cloned()
                            .ok_or_else(|| err("unknown legacy memo"))?,
                    );
                }
                b'j' => {
                    let i = u32le(self.read(4)?) as usize;
                    stack.push(
                        memo.get(&i)
                            .cloned()
                            .ok_or_else(|| err("unknown legacy memo"))?,
                    );
                }
                b't' => {
                    let mark = marks.pop().ok_or_else(|| err("legacy MARK underflow"))?;
                    let values = stack.split_off(mark);
                    stack.push(Value::Tuple(values));
                }
                0x85 => {
                    let a = stack.pop().ok_or_else(|| err("legacy stack underflow"))?;
                    stack.push(Value::Tuple(vec![a]));
                }
                b'Q' => {
                    let id = stack
                        .pop()
                        .ok_or_else(|| err("legacy persistent stack underflow"))?;
                    let key = value_string(&id)?;
                    stack.push(
                        self.registry
                            .get(&key)
                            .cloned()
                            .ok_or_else(|| err("unknown legacy persistent tensor id"))?,
                    );
                }
                0x81 => {
                    let args = stack
                        .pop()
                        .ok_or_else(|| err("legacy NEWOBJ stack underflow"))?;
                    let class = stack
                        .pop()
                        .ok_or_else(|| err("legacy NEWOBJ stack underflow"))?;
                    if !matches!(class,Value::Symbol(ref m,ref n) if m=="torch.nn.parameter" && n=="Parameter")
                        || !matches!(args,Value::Tuple(ref x) if x.is_empty())
                    {
                        return Err(err("legacy NEWOBJ target is not inert Parameter"));
                    }
                    stack.push(Value::Parameter(None));
                }
                b'b' => {
                    let state = stack
                        .pop()
                        .ok_or_else(|| err("legacy BUILD stack underflow"))?;
                    let target = stack
                        .last_mut()
                        .ok_or_else(|| err("legacy BUILD target missing"))?;
                    let Value::Parameter(inner) = target else {
                        return Err(err("legacy BUILD target is not inert Parameter"));
                    };
                    let Value::Tuple(mut x) = state else {
                        return Err(err("legacy Parameter BUILD state must be a tuple"));
                    };
                    if x.len() != 1 || inner.is_some() {
                        return Err(err("legacy Parameter BUILD state is invalid"));
                    };
                    *inner = Some(Box::new(x.remove(0)));
                }
                b's' => {
                    let value = stack
                        .pop()
                        .ok_or_else(|| err("legacy SETITEM stack underflow"))?;
                    let key = value_string(
                        &stack
                            .pop()
                            .ok_or_else(|| err("legacy SETITEM stack underflow"))?,
                    )?;
                    let Some(Value::Dict(map)) = stack.last_mut() else {
                        return Err(err("legacy SETITEM target is not dict"));
                    };
                    if map.insert(key, value).is_some() {
                        return Err(err("duplicate legacy dict key"));
                    };
                }
                op => {
                    return Err(err(format!(
                        "legacy pickle opcode 0x{op:02x} is not whitelisted"
                    )));
                }
            }
        }
    }
    fn byte(&mut self) -> Result<u8> {
        let b = *self
            .bytes
            .get(self.at)
            .ok_or_else(|| err("truncated legacy pickle"))?;
        self.at += 1;
        Ok(b)
    }
    fn read(&mut self, n: usize) -> Result<&'a [u8]> {
        let b = take(self.bytes, self.at, n, "truncated legacy pickle")?;
        self.at += n;
        Ok(b)
    }
    fn utf8(&mut self, n: usize) -> Result<String> {
        std::str::from_utf8(self.read(n)?)
            .map(|x| x.to_owned())
            .map_err(|_| err("legacy pickle string is not UTF-8"))
    }
    fn line(&mut self) -> Result<String> {
        let rest = &self.bytes[self.at..];
        let n = rest
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| err("unterminated legacy GLOBAL"))?;
        self.at += n + 1;
        std::str::from_utf8(&rest[..n])
            .map(|x| x.to_owned())
            .map_err(|_| err("legacy GLOBAL is not UTF-8"))
    }
}
#[allow(dead_code)]
fn legacy_memo(memo: &mut BTreeMap<usize, Value>, index: usize, stack: &[Value]) -> Result<()> {
    if index > MAX_ARCHIVE_ENTRIES * 16 {
        return Err(err("legacy pickle memo limit exceeded"));
    };
    let value = stack
        .last()
        .cloned()
        .ok_or_else(|| err("legacy memo stack underflow"))?;
    if memo.insert(index, value).is_some() {
        return Err(err("duplicate legacy memo index"));
    };
    Ok(())
}

#[allow(dead_code)]
fn legacy_storages(bytes: &[u8]) -> Result<BTreeMap<String, StorageRef>> {
    let empty = BTreeMap::new();
    let mut vm = LegacyPickle::new(bytes, &empty);
    let count = value_usize(&vm.next()?, "legacy storage count")?;
    if count > MAX_ARCHIVE_ENTRIES {
        return Err(err("legacy storage count exceeds limit"));
    }
    let mut out = BTreeMap::new();
    for _ in 0..count {
        let Value::Tuple(record) = vm.next()? else {
            return Err(err("legacy storage record is not a tuple"));
        };
        if record.len() != 3 {
            return Err(err("legacy storage record has wrong arity"));
        }
        let key = match &record[0] {
            Value::Str(x) => x.clone(),
            Value::Int(x) if *x >= 0 => x.to_string(),
            _ => return Err(err("invalid legacy storage id")),
        };
        let Value::Symbol(module, name) = &record[2] else {
            return Err(err("invalid legacy storage type"));
        };
        let dtype = storage_dtype(module, name)?;
        let size = usize::try_from(i64::from_le_bytes(
            vm.read(8)?
                .try_into()
                .map_err(|_| err("bad legacy storage size"))?,
        ))
        .map_err(|_| err("negative legacy storage size"))?;
        let raw = vm.read(
            size.checked_mul(dtype.itemsize())
                .ok_or_else(|| err("legacy storage byte overflow"))?,
        )?;
        if raw.len() > MAX_ARCHIVE_BYTES {
            return Err(err("legacy storage exceeds limit"));
        }
        if out
            .insert(
                key.clone(),
                StorageRef {
                    key,
                    dtype,
                    elements: size,
                    raw: raw.to_vec(),
                },
            )
            .is_some()
        {
            return Err(err("duplicate legacy storage id"));
        }
    }
    if vm.at != bytes.len() {
        return Err(err("trailing legacy storage records"));
    }
    Ok(out)
}

/// Parses the legacy `tensors` stream after `legacy_storages` has retained the
/// exact CPU storage bytes.  Each record is a protocol-2 `(tensor_id,
/// storage_id, tensor_type)` tuple followed by Torch's fixed binary view
/// framing: LE i32 rank, four zero marker bytes, then LE i64 sizes, strides,
/// and storage offset.  Views are materialized rather than borrowed.
#[allow(dead_code)]
fn legacy_tensors(
    bytes: &[u8],
    storages: &BTreeMap<String, StorageRef>,
) -> Result<BTreeMap<String, TensorData>> {
    let empty = BTreeMap::new();
    let mut vm = LegacyPickle::new(bytes, &empty);
    let count = value_usize(&vm.next()?, "legacy tensor count")?;
    if count > MAX_ARCHIVE_ENTRIES {
        return Err(err("legacy tensor count exceeds limit"));
    }
    let mut tensors = BTreeMap::new();
    for _ in 0..count {
        let Value::Tuple(record) = vm.next()? else {
            return Err(err("legacy tensor record is not a tuple"));
        };
        if record.len() != 3 {
            return Err(err("legacy tensor record has wrong arity"));
        }
        let key = legacy_record_id(&record[0], "tensor")?;
        let storage_key = legacy_record_id(&record[1], "storage")?;
        let Value::Symbol(module, name) = &record[2] else {
            return Err(err("legacy tensor type is not a symbol"));
        };
        let dtype = tensor_dtype(module, name)?;
        let storage = storages
            .get(&storage_key)
            .ok_or_else(|| err(format!("missing legacy storage id {storage_key:?}")))?;
        if storage.dtype != dtype {
            return Err(err("legacy tensor/storage dtype mismatch"));
        }
        let rank = usize::try_from(i32::from_le_bytes(
            vm.read(4)?
                .try_into()
                .map_err(|_| err("bad legacy tensor rank"))?,
        ))
        .map_err(|_| err("negative legacy tensor rank"))?;
        if rank > 64 {
            return Err(err("legacy tensor rank exceeds limit"));
        }
        if vm.read(4)? != [0, 0, 0, 0] {
            return Err(err("legacy tensor framing marker is invalid"));
        }
        let read_i64s = |vm: &mut LegacyPickle<'_>| -> Result<Vec<usize>> {
            (0..rank)
                .map(|_| {
                    usize::try_from(i64::from_le_bytes(
                        vm.read(8)?
                            .try_into()
                            .map_err(|_| err("bad legacy tensor dimension"))?,
                    ))
                    .map_err(|_| err("negative legacy tensor dimension"))
                })
                .collect()
        };
        let shape = read_i64s(&mut vm)?;
        let strides = read_i64s(&mut vm)?;
        let offset = usize::try_from(i64::from_le_bytes(
            vm.read(8)?
                .try_into()
                .map_err(|_| err("bad legacy tensor offset"))?,
        ))
        .map_err(|_| err("negative legacy tensor offset"))?;
        let spec = TensorSpec {
            storage: storage.clone(),
            offset,
            shape,
            strides,
        };
        let data = tensor_from_raw_spec(&spec, &storage.raw)?;
        if tensors.insert(key, data).is_some() {
            return Err(err("duplicate legacy tensor id"));
        }
    }
    if vm.at != bytes.len() {
        return Err(err("trailing legacy tensor records"));
    }
    Ok(tensors)
}

fn legacy_record_id(value: &Value, kind: &'static str) -> Result<String> {
    match value {
        Value::Str(x) if !x.is_empty() => Ok(x.clone()),
        Value::Int(x) if *x >= 0 => Ok(x.to_string()),
        _ => Err(err(format!("invalid legacy {kind} id"))),
    }
}

fn tensor_dtype(module: &str, name: &str) -> Result<DType> {
    if module != "torch" {
        return Err(err("unsupported legacy tensor module"));
    }
    Ok(match name {
        "BoolTensor" => DType::Bool,
        "CharTensor" => DType::I8,
        "ByteTensor" => DType::U8,
        "ShortTensor" => DType::I16,
        "IntTensor" => DType::I32,
        "LongTensor" => DType::I64,
        "HalfTensor" => DType::F16,
        "BFloat16Tensor" => DType::BF16,
        "FloatTensor" => DType::F32,
        "DoubleTensor" => DType::F64,
        _ => return Err(err(format!("unsupported legacy tensor type {name}"))),
    })
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
            raw: raw.to_vec(),
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
        if module != "torch._utils" || !(name == "_rebuild_tensor" || name == "_rebuild_tensor_v2") {
            return Err(err(format!("pickle GLOBAL {module}.{name} is not allowed")));
        }
        let Value::Tuple(v) = args else {
            return Err(err("tensor rebuild arguments are not a tuple"));
        };
        match name.as_str() {
            "_rebuild_tensor" if v.len() != 4 => {
                return Err(err("tensor rebuild has unsupported arguments"));
            }
            "_rebuild_tensor_v2" if !matches!(v.len(), 5 | 6) => {
                return Err(err("tensor rebuild v2 has unsupported arguments"));
            }
            "_rebuild_tensor_v2" if !matches!(&v[4], Value::Bool) => {
                return Err(err("tensor rebuild v2 requires a Boolean gradient flag"));
            }
            "_rebuild_tensor_v2"
                if v.len() == 6 && !matches!(&v[5], Value::Dict(_) | Value::None) =>
            {
                return Err(err("tensor rebuild v2 has unsupported backward hooks"));
            }
            _ => {}
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
    let raw = storages
        .get(&spec.storage.key)
        .ok_or_else(|| err("storage disappeared during parse"))?;
    tensor_from_raw_spec(&spec, raw)
}

fn tensor_from_raw_spec(spec: &TensorSpec, raw: &[u8]) -> Result<TensorData> {
    let shape = Shape::new(spec.shape.clone());
    let count = shape.numel()?;
    let out_bytes = count
        .checked_mul(spec.storage.dtype.itemsize())
        .ok_or_else(|| err("tensor byte length overflow"))?;
    if out_bytes > MAX_TENSOR_BYTES {
        return Err(err("tensor exceeds configured byte limit"));
    }
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
            let crc = crc32(data);
            out.extend_from_slice(b"PK\x03\x04");
            out.extend_from_slice(&20u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&[0; 4]);
            out.extend_from_slice(&crc.to_le_bytes());
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
            central.extend_from_slice(&crc.to_le_bytes());
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
    fn tar(files: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, data) in files {
            let mut header = [0u8; 512];
            header[..name.len()].copy_from_slice(name.as_bytes());
            header[100..108].copy_from_slice(b"0000644\0");
            header[108..116].copy_from_slice(b"0000000\0");
            header[116..124].copy_from_slice(b"0000000\0");
            let size = format!("{:011o}\0", data.len());
            header[124..136].copy_from_slice(size.as_bytes());
            header[136..148].copy_from_slice(b"00000000000\0");
            header[148..156].fill(b' ');
            header[156] = b'0';
            header[257..263].copy_from_slice(b"ustar\0");
            header[263..265].copy_from_slice(b"00");
            let checksum: usize = header.iter().map(|&b| usize::from(b)).sum();
            let checksum = format!("{:06o}\0 ", checksum);
            header[148..156].copy_from_slice(checksum.as_bytes());
            out.extend_from_slice(&header);
            out.extend_from_slice(data);
            out.resize(out.len().div_ceil(512) * 512, 0);
        }
        out.extend_from_slice(&[0; 1024]);
        out
    }
    fn legacy_tensor_stream(
        tensor: &str,
        storage: &str,
        dtype: &str,
        shape: &[i64],
        strides: &[i64],
        offset: i64,
    ) -> Vec<u8> {
        let mut out = vec![0x80, 2, b'K', 1, b'.'];
        out.extend_from_slice(&[0x80, 2, b'(']);
        for id in [tensor, storage] {
            out.push(b'X');
            out.extend_from_slice(&(id.len() as u32).to_le_bytes());
            out.extend_from_slice(id.as_bytes());
        }
        out.extend_from_slice(b"ctorch\n");
        out.extend_from_slice(dtype.as_bytes());
        out.extend_from_slice(b"\nt.");
        out.extend_from_slice(&(shape.len() as i32).to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        for values in [shape, strides] {
            for &value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        out.extend_from_slice(&offset.to_le_bytes());
        out
    }
    fn legacy_storage(key: &str, dtype: DType, raw: Vec<u8>) -> BTreeMap<String, StorageRef> {
        BTreeMap::from([(
            key.into(),
            StorageRef {
                key: key.into(),
                dtype,
                elements: raw.len() / dtype.itemsize(),
                raw,
            },
        )])
    }
    fn legacy_storage_stream(key: &str, dtype: &str, raw: &[u8]) -> Vec<u8> {
        let itemsize = match dtype {
            "FloatStorage" => 4,
            _ => unreachable!("test only builds FloatStorage"),
        };
        let mut out = vec![0x80, 2, b'K', 1, b'.', 0x80, 2, b'('];
        out.push(b'X');
        out.extend_from_slice(&(key.len() as u32).to_le_bytes());
        out.extend_from_slice(key.as_bytes());
        out.push(b'X');
        out.extend_from_slice(&(3u32).to_le_bytes());
        out.extend_from_slice(b"cpu");
        out.extend_from_slice(b"ctorch\n");
        out.extend_from_slice(dtype.as_bytes());
        out.extend_from_slice(b"\nt.");
        out.extend_from_slice(&((raw.len() / itemsize) as i64).to_le_bytes());
        out.extend_from_slice(raw);
        out
    }
    fn legacy_parameter_state_pickle(key: &str, tensor_id: &str) -> Vec<u8> {
        let mut out = vec![0x80, 2, b'}', b'X'];
        out.extend_from_slice(&(key.len() as u32).to_le_bytes());
        out.extend_from_slice(key.as_bytes());
        out.extend_from_slice(b"ctorch.nn.parameter\nParameter\n)");
        out.push(0x81);
        out.push(b'(');
        out.push(b'X');
        out.extend_from_slice(&(tensor_id.len() as u32).to_le_bytes());
        out.extend_from_slice(tensor_id.as_bytes());
        out.extend_from_slice(b"Qtbs.");
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

        let mut pickle = zip_stored_files(&good)
            .unwrap()
            .remove("archive/data.pkl")
            .unwrap();
        let marker = [b't', 0x89, b't', b'R'];
        let at = pickle
            .windows(marker.len())
            .rposition(|window| window == marker)
            .unwrap();
        // A sixth v2 argument is present but is not an inert hook dictionary.
        // The importer must reject it rather than silently ignoring it.
        pickle.splice(at + 2..at + 2, [b'K', 0]);
        let malformed = zip(&[("archive/data.pkl", pickle), ("archive/data/0", raw)]);
        assert!(matches!(
            load_torch_state_dict(&malformed),
            Err(Error::ModelIo { .. })
        ));
    }

    #[test]
    fn deflate_and_tar_containers_are_bounded_and_validated() {
        use flate2::{Compression, write::DeflateEncoder};
        use std::io::Write;
        let source = b"raw deflate fixture with exact bytes";
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(source).unwrap();
        let encoded = encoder.finish().unwrap();
        assert_eq!(
            decode_zip_member(8, &encoded, source.len()).unwrap(),
            source
        );
        assert!(decode_zip_member(8, &encoded, source.len() - 1).is_err());

        let archive = tar(&[("weights/a", vec![1, 2, 3]), ("weights/empty", vec![])]);
        assert_eq!(extract_tar_files(&archive).unwrap()["weights/a"], [1, 2, 3]);
        let mut bad_checksum = archive.clone();
        bad_checksum[0] ^= 1;
        assert!(matches!(
            extract_tar_files(&bad_checksum),
            Err(Error::ModelIo { .. })
        ));
        let mut bad_kind = tar(&[("weights/a", vec![1])]);
        bad_kind[156] = b'2';
        assert!(matches!(
            extract_tar_files(&bad_kind),
            Err(Error::ModelIo { .. })
        ));
    }

    #[test]
    fn zip64_metadata_requires_one_exact_checked_extra_field() {
        let mut fields = Vec::new();
        fields.extend_from_slice(&1u16.to_le_bytes());
        fields.extend_from_slice(&24u16.to_le_bytes());
        fields.extend_from_slice(&4u64.to_le_bytes());
        fields.extend_from_slice(&3u64.to_le_bytes());
        fields.extend_from_slice(&17u64.to_le_bytes());
        assert_eq!(
            zip64_entry_values(u32::MAX, u32::MAX, u32::MAX, &fields).unwrap(),
            (3, 4, 17)
        );
        assert!(zip64_entry_values(u32::MAX, 1, 2, &[]).is_err());
        let mut duplicate = fields.clone();
        duplicate.extend_from_slice(&fields);
        assert!(zip64_entry_values(u32::MAX, u32::MAX, u32::MAX, &duplicate).is_err());
        let mut tail = [0u8; 22];
        tail[..4].copy_from_slice(b"PK\x05\x06");
        tail[8..10].copy_from_slice(&u16::MAX.to_le_bytes());
        tail[10..12].copy_from_slice(&u16::MAX.to_le_bytes());
        tail[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        tail[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(zip_directory(&tail, 0, &tail).is_err());
    }

    #[test]
    fn legacy_record_vm_frames_objects_and_restricts_parameter_build() {
        let registry = BTreeMap::new();
        let mut vm =
            LegacyPickle::new(&[0x80, 2, b'K', 7, b'.', 0x80, 2, b'K', 9, b'.'], &registry);
        assert!(matches!(vm.next().unwrap(), Value::Int(7)));
        assert!(matches!(vm.next().unwrap(), Value::Int(9)));
        let parameter = [
            0x80, 2, b'c', b't', b'o', b'r', b'c', b'h', b'.', b'n', b'n', b'.', b'p', b'a', b'r',
            b'a', b'm', b'e', b't', b'e', b'r', b'\n', b'P', b'a', b'r', b'a', b'm', b'e', b't',
            b'e', b'r', b'\n', b')', 0x81, b'(', b'K', 3, b't', b'b', b'.',
        ];
        assert!(matches!(
            LegacyPickle::new(&parameter, &registry).next().unwrap(),
            Value::Parameter(Some(_))
        ));
        let bad = [0x80, 2, b'N', b'N', b'b', b'.'];
        assert!(LegacyPickle::new(&bad, &registry).next().is_err());
    }

    #[test]
    fn legacy_tensor_records_materialize_exact_strided_views() {
        let raw = [
            0x34, 0x12, 0xc0, 0x7f, // NaN payload
            0x00, 0x00, 0x00, 0x80, // negative zero
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40,
        ];
        let storages = legacy_storage("s", DType::F32, raw.to_vec());
        let tensors = legacy_tensors(
            &legacy_tensor_stream("weight", "s", "FloatTensor", &[2, 2], &[1, 2], 0),
            &storages,
        )
        .unwrap();
        assert_eq!(
            tensors["weight"].to_le_bytes().unwrap(),
            [
                raw[0..4].to_vec(),
                raw[8..12].to_vec(),
                raw[4..8].to_vec(),
                raw[12..16].to_vec()
            ]
            .concat()
        );
    }

    #[test]
    fn legacy_storage_registry_retains_raw_bytes_and_metadata() {
        let raw = [0x34, 0x12, 0xc0, 0x7f, 0, 0, 0, 0x80];
        let registry = legacy_storages(&legacy_storage_stream("s", "FloatStorage", &raw)).unwrap();
        let storage = &registry["s"];
        assert_eq!(storage.dtype, DType::F32);
        assert_eq!(storage.elements, 2);
        assert_eq!(storage.raw, raw);
    }

    #[test]
    fn legacy_tensor_records_reject_bad_framing_ids_and_views() {
        let storages = legacy_storage("s", DType::F32, vec![0; 16]);
        let good = legacy_tensor_stream("x", "s", "FloatTensor", &[2, 2], &[2, 1], 0);
        let mut marker = good.clone();
        let marker_at = marker.len() - (4 + 4 + 2 * 16 + 8) + 4;
        marker[marker_at] = 1;
        assert!(legacy_tensors(&marker, &storages).is_err());
        assert!(
            legacy_tensors(
                &legacy_tensor_stream("x", "missing", "FloatTensor", &[1], &[1], 0),
                &storages
            )
            .is_err()
        );
        assert!(
            legacy_tensors(
                &legacy_tensor_stream("x", "s", "FloatTensor", &[2, 2], &[0, 1], 0),
                &storages
            )
            .is_err()
        );
        assert!(
            legacy_tensors(
                &legacy_tensor_stream("x", "s", "FloatTensor", &[3], &[1], 2),
                &storages
            )
            .is_err()
        );
    }

    #[test]
    fn legacy_tar_import_is_strict_and_safetensors_portable() {
        let raw = [0x34, 0x12, 0xc0, 0x7f, 0, 0, 0, 0x80];
        let archive = tar(&[
            ("storages", legacy_storage_stream("s", "FloatStorage", &raw)),
            (
                "tensors",
                legacy_tensor_stream("weight-id", "s", "FloatTensor", &[1, 2], &[2, 1], 0),
            ),
            (
                "pickle",
                legacy_parameter_state_pickle("weight", "weight-id"),
            ),
        ]);
        let state = load_legacy_torch_state_dict(&archive).unwrap();
        assert_eq!(state["weight"].to_le_bytes().unwrap(), raw);
        let safe = save_safetensors(&state, &Metadata::new()).unwrap();
        assert_eq!(
            load_safetensors(&safe).unwrap().0["weight"]
                .to_le_bytes()
                .unwrap(),
            raw
        );
        let mut graph = Graph::new();
        let linear = Linear::new(&mut graph, 2, 1, false, 99).unwrap();
        let report = linear
            .load_state_dict(
                &ModuleStateDict::from(state.clone()),
                true,
                CastPolicy::Exact,
            )
            .unwrap();
        assert_eq!(report.loaded_keys, ["weight"]);
        assert_eq!(
            linear.state_dict().unwrap().into_tensors()["weight"]
                .to_le_bytes()
                .unwrap(),
            raw
        );
        let mut bad = archive.clone();
        bad[156] = b'2';
        assert!(load_legacy_torch_state_dict(&bad).is_err());
    }
}
