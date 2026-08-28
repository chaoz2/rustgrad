//! Validated static ONNX schema records and attribute normalization.

use super::{
    bad,
    wire::{Msg, one_bytes, one_varint, var},
};
use crate::{DType, PoolOptions, Result, Shape, TensorData};
use std::collections::BTreeMap;

pub(super) fn attrs(n: &Msg<'_>) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut out = BTreeMap::new();
    for b in n.bytes(5)? {
        let a = Msg::new(b);
        let name = a
            .string(1)?
            .ok_or_else(|| bad("attribute lacks name"))?
            .to_owned();
        let fields = a.fields()?;
        let mut values = fields.iter().filter(|(id, wire, _)| {
            (*id == 2 && *wire == 5)
                || (*id == 3 && *wire == 0)
                || ((*id == 4 || *id == 5 || *id == 8) && *wire == 2)
        });
        let Some((_, _, value)) = values.next() else {
            return Err(bad("unsupported ONNX attribute form"));
        };
        if values.next().is_some() {
            return Err(bad("duplicate ONNX attribute value"));
        }
        let value = value.to_vec();
        if out.insert(name, value).is_some() {
            return Err(bad("duplicate ONNX attribute"));
        }
    }
    Ok(out)
}
pub(super) fn scalar_i64(b: &[u8]) -> Result<i64> {
    let mut at = 0;
    Ok(var(b, &mut at)? as i64)
}
pub(super) fn scalar_f32(b: &[u8]) -> Result<f32> {
    let a: [u8; 4] = b
        .try_into()
        .map_err(|_| bad("ONNX float attribute must be f32"))?;
    Ok(f32::from_le_bytes(a))
}

/// Reads one named ONNX FLOAT attribute without allowing another AttributeProto
/// value field to masquerade as its fixed-width payload.  Most legacy lowering
/// sites intentionally operate on the normalized raw attribute bytes; the
/// parameterized HardSigmoid adapter needs this narrower source-level check.
pub(super) fn typed_scalar_f32_attr(n: &Msg<'_>, wanted: &str) -> Result<Option<f32>> {
    let mut out = None;
    for raw in n.bytes(5)? {
        let attribute = Msg::new(raw);
        if attribute.string(1)? != Some(wanted) {
            continue;
        }
        if out.is_some() {
            return Err(bad("duplicate ONNX attribute"));
        }
        let fields = attribute.fields()?;
        let types: Vec<_> = fields
            .iter()
            .filter(|(id, wire, _)| *id == 20 && *wire == 0)
            .collect();
        let values: Vec<_> = fields
            .iter()
            .filter(|(id, wire, _)| {
                (*id == 2 && *wire == 5)
                    || (*id == 3 && *wire == 0)
                    || ((*id == 4 || *id == 5 || *id == 8) && *wire == 2)
            })
            .collect();
        let [(_, _, ty)] = types.as_slice() else {
            return Err(bad("ONNX float attribute must declare FLOAT type"));
        };
        let mut at = 0;
        if var(ty, &mut at)? != 1 || at != ty.len() {
            return Err(bad("ONNX attribute is not FLOAT"));
        }
        let [(id, wire, value)] = values.as_slice() else {
            return Err(bad("ONNX float attribute must have one FLOAT value"));
        };
        if *id != 2 || *wire != 5 {
            return Err(bad("ONNX attribute is not FLOAT"));
        }
        out = Some(scalar_f32(value)?);
    }
    Ok(out)
}

/// Reads one named ONNX INT attribute without allowing a STRING, FLOAT, or
/// another AttributeProto payload field to be interpreted as a varint.  The
/// older importer adapters intentionally keep their raw normalized bytes;
/// CumSum needs this narrow check because its flags use Python truthiness.
pub(super) fn typed_scalar_i64_attr(n: &Msg<'_>, wanted: &str) -> Result<Option<i64>> {
    let mut out = None;
    for raw in n.bytes(5)? {
        let attribute = Msg::new(raw);
        if attribute.string(1)? != Some(wanted) {
            continue;
        }
        if out.is_some() {
            return Err(bad("duplicate ONNX attribute"));
        }
        let fields = attribute.fields()?;
        let types: Vec<_> = fields
            .iter()
            .filter(|(id, wire, _)| *id == 20 && *wire == 0)
            .collect();
        if !types.is_empty() {
            let [(_, _, ty)] = types.as_slice() else {
                return Err(bad("ONNX integer attribute must declare one INT type"));
            };
            let mut at = 0;
            if var(ty, &mut at)? != 2 || at != ty.len() {
                return Err(bad("ONNX attribute is not INT"));
            }
        }
        let values: Vec<_> = fields
            .iter()
            .filter(|(id, wire, _)| {
                (*id == 2 && *wire == 5)
                    || (*id == 3 && *wire == 0)
                    || ((*id == 4 || *id == 5 || *id == 8) && *wire == 2)
            })
            .collect();
        let [(id, wire, raw_value)] = values.as_slice() else {
            return Err(bad("ONNX integer attribute must have one INT value"));
        };
        if *id != 3 || *wire != 0 {
            return Err(bad("ONNX attribute is not INT"));
        }
        let mut at = 0;
        let value = var(raw_value, &mut at)?;
        if at != raw_value.len() {
            return Err(bad("invalid ONNX integer attribute"));
        }
        out = Some(value as i64);
    }
    Ok(out)
}

/// Reads one named ONNX INT attribute whose declared AttributeProto type must
/// be INT.  Some legacy adapters predate declared-type enforcement; creation
/// operators such as EyeLike use this narrower form because their dtype and
/// diagonal controls must not accept an untyped wire alias.
pub(super) fn strict_typed_scalar_i64_attr(n: &Msg<'_>, wanted: &str) -> Result<Option<i64>> {
    let mut out = None;
    for raw in n.bytes(5)? {
        let attribute = Msg::new(raw);
        if attribute.string(1)? != Some(wanted) {
            continue;
        }
        if out.is_some() {
            return Err(bad("duplicate ONNX attribute"));
        }
        let fields = attribute.fields()?;
        let types: Vec<_> = fields
            .iter()
            .filter(|(id, wire, _)| *id == 20 && *wire == 0)
            .collect();
        let [(_, _, ty)] = types.as_slice() else {
            return Err(bad("ONNX integer attribute must declare INT type"));
        };
        let mut at = 0;
        if var(ty, &mut at)? != 2 || at != ty.len() {
            return Err(bad("ONNX attribute is not INT"));
        }
        let values: Vec<_> = fields
            .iter()
            .filter(|(id, wire, _)| {
                (*id == 2 && *wire == 5)
                    || (*id == 3 && *wire == 0)
                    || ((*id == 4 || *id == 5 || *id == 8) && *wire == 2)
            })
            .collect();
        let [(id, wire, raw_value)] = values.as_slice() else {
            return Err(bad("ONNX integer attribute must have one INT value"));
        };
        if *id != 3 || *wire != 0 {
            return Err(bad("ONNX attribute is not INT"));
        }
        let mut at = 0;
        let value = var(raw_value, &mut at)?;
        if at != raw_value.len() {
            return Err(bad("invalid ONNX integer attribute"));
        }
        out = Some(value as i64);
    }
    Ok(out)
}

/// Reads one named ONNX STRING attribute whose declared AttributeProto type
/// is STRING. This is deliberately narrower than the legacy raw attribute
/// map: movement adapters must not treat an INT/FLOAT/TENSOR wire payload as
/// a UTF-8 mode string.
pub(super) fn strict_typed_string_attr(n: &Msg<'_>, wanted: &str) -> Result<Option<String>> {
    let mut out = None;
    for raw in n.bytes(5)? {
        let attribute = Msg::new(raw);
        if attribute.string(1)? != Some(wanted) {
            continue;
        }
        if out.is_some() {
            return Err(bad("duplicate ONNX attribute"));
        }
        let fields = attribute.fields()?;
        let types: Vec<_> = fields
            .iter()
            .filter(|(id, wire, _)| *id == 20 && *wire == 0)
            .collect();
        let [(_, _, ty)] = types.as_slice() else {
            return Err(bad("ONNX string attribute must declare STRING type"));
        };
        let mut at = 0;
        if var(ty, &mut at)? != 3 || at != ty.len() {
            return Err(bad("ONNX attribute is not STRING"));
        }
        let values: Vec<_> = fields
            .iter()
            .filter(|(id, wire, _)| {
                (*id == 2 && *wire == 5)
                    || (*id == 3 && *wire == 0)
                    || ((*id == 4 || *id == 5 || *id == 8) && *wire == 2)
            })
            .collect();
        let [(id, wire, value)] = values.as_slice() else {
            return Err(bad("ONNX string attribute must have one STRING value"));
        };
        if *id != 4 || *wire != 2 {
            return Err(bad("ONNX attribute is not STRING"));
        }
        out = Some(
            std::str::from_utf8(value)
                .map_err(|_| bad("ONNX string attribute is not UTF-8"))?
                .to_owned(),
        );
    }
    Ok(out)
}
pub(super) fn packed_i64(b: &[u8]) -> Result<Vec<i64>> {
    let mut at = 0;
    let mut x = vec![];
    while at < b.len() {
        x.push(var(b, &mut at)? as i64)
    }
    Ok(x)
}
pub(super) fn conv_pair(
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
pub(super) fn conv_pads(attrs: &BTreeMap<String, Vec<u8>>) -> Result<[usize; 4]> {
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
pub(super) fn conv_same_padding(
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
pub(super) fn onnx_pool_options(
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
    if max && attrs.contains_key("count_include_pad") {
        return Err(bad("unsupported MaxPool count_include_pad attribute"));
    }
    if !max && attrs.contains_key("storage_order") {
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
pub(super) fn axes_usize(x: &[i64], rank: usize) -> Result<Vec<usize>> {
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
pub(super) fn const_i64(c: &BTreeMap<String, TensorData>, name: &str) -> Result<Vec<i64>> {
    let x = c
        .get(name)
        .ok_or_else(|| bad("ONNX shape/axes input must be a constant initializer"))?;
    if x.dtype() != DType::I64 {
        return Err(bad("ONNX shape/axes constant must be I64"));
    }
    Ok((0..x.len()).map(|i| x.scalar_at(i).as_i64()).collect())
}
pub(super) fn reshape_dims(old: &[usize], shape: &[i64]) -> Result<Shape> {
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

pub(super) fn value_info(m: Msg<'_>) -> Result<(String, Shape, DType)> {
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
