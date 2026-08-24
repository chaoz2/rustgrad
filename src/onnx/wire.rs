//! Bounded protobuf wire parsing for the private ONNX importer.

use crate::{Error, Result};

const MAX_ITEMS: usize = 4096;

fn bad(s: impl Into<String>) -> Error {
    Error::ModelIo { reason: s.into() }
}

pub(super) struct Msg<'a> {
    b: &'a [u8],
}
impl<'a> Msg<'a> {
    pub(super) fn new(b: &'a [u8]) -> Self {
        Self { b }
    }
    pub(super) fn fields(&self) -> Result<Vec<(u32, u8, &'a [u8])>> {
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
    pub(super) fn bytes(&self, id: u32) -> Result<Vec<&'a [u8]>> {
        Ok(self
            .fields()?
            .into_iter()
            .filter_map(|(i, w, x)| (i == id && w == 2).then_some(x))
            .collect())
    }
    pub(super) fn string(&self, id: u32) -> Result<Option<&'a str>> {
        match self.bytes(id)?.as_slice() {
            [] => Ok(None),
            [x] => std::str::from_utf8(x)
                .map(Some)
                .map_err(|_| bad("ONNX string is not UTF-8")),
            _ => Err(bad("duplicate ONNX string field")),
        }
    }
    pub(super) fn strings(&self, id: u32) -> Result<Vec<&'a str>> {
        self.bytes(id)?
            .into_iter()
            .map(|x| std::str::from_utf8(x).map_err(|_| bad("ONNX string is not UTF-8")))
            .collect()
    }
    pub(super) fn packed(&self, id: u32) -> Result<Vec<u64>> {
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
pub(super) fn var(b: &[u8], at: &mut usize) -> Result<u64> {
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
pub(super) fn one_bytes<'a>(m: &Msg<'a>, id: u32, what: &str) -> Result<&'a [u8]> {
    match m.bytes(id)?.as_slice() {
        [x] => Ok(*x),
        _ => Err(bad(format!("ONNX {what} must occur once"))),
    }
}
pub(super) fn one_varint(m: &Msg<'_>, id: u32, what: &str) -> Result<u64> {
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
