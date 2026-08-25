//! OpenCL C lowering for immutable captured Threefry random plans.
//!
//! The generated kernel has one mutable output pointer and an extent scalar.
//! Key, counter, chunk layout, and distribution parameters are literal plan
//! data, so neither rendering nor launch consults the mutable stream registry.

use super::{
    OpenClBufferAbi, OpenClCapabilities, OpenClError, OpenClRenderer, RenderedOpenCl, narrow,
    renderer::{OPENCL_ABI_VERSION, OPENCL_RENDERER_VERSION},
};
use crate::{DType, RandomKind, random::plan::RandomKernelPlan};
use std::{
    collections::BTreeMap,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::Arc,
};

pub(super) fn render(
    renderer: &OpenClRenderer,
    plan: &RandomKernelPlan,
) -> Result<RenderedOpenCl, OpenClError> {
    plan.validate()
        .map_err(|error| OpenClError::Unsupported(error.to_string()))?;
    let extent = plan.shape.numel().map_err(|_| OpenClError::Overflow)?;
    let required_capabilities = required(plan)?;
    if !renderer.capabilities.supports(required_capabilities) {
        return Err(OpenClError::Unsupported(
            "captured random plan requires unavailable OpenCL device capability".into(),
        ));
    }
    let entry = format!("rg_opencl_random_e{extent}");
    let mut lines = Vec::new();
    if required_capabilities.fp64 {
        lines.push("#pragma OPENCL EXTENSION cl_khr_fp64 : enable".into());
    }
    if matches!(plan.dtype, DType::F16) {
        lines.push(narrow::F16_SOURCE.into());
    }
    if matches!(plan.dtype, DType::BF16) {
        lines.push(narrow::BF16_SOURCE.into());
    }
    lines.push(format!(
        "// {OPENCL_RENDERER_VERSION} captured-threefry ABI {OPENCL_ABI_VERSION}"
    ));
    lines.push(helpers(plan, required_capabilities.fp64));
    lines.push(format!(
        "__kernel void {entry}(__global {}* out, ulong extent) {{",
        cl_type(plan.dtype)?
    ));
    lines.push("  const ulong gid = (ulong)get_global_id(0);".into());
    lines.push("  if (gid >= extent) return;".into());
    let value = match plan.kind {
        RandomKind::Uniform { low, high } => uniform_value(plan.dtype, low, high),
        RandomKind::Normal { mean, std } => normal_value(plan.dtype, mean, std),
        RandomKind::RandInt { low, high } => randint_value(plan.dtype, low, high)?,
    };
    let source_line = lines.len() + 1;
    lines.push(format!("  out[gid] = {value};"));
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let buffer = OpenClBufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        source_shape: plan.shape.clone(),
        elements: extent,
        mutable: true,
        view: None,
    };
    Ok(RenderedOpenCl {
        source: source.clone(),
        source_map: BTreeMap::from([(plan.output.index(), source_line)]),
        buffers: vec![buffer.clone()],
        extent,
        entry,
        cache_key: stable_key(&(
            OPENCL_RENDERER_VERSION,
            OPENCL_ABI_VERSION,
            renderer.local_size,
            renderer.capabilities,
            &source,
            plan,
        )),
        required_capabilities,
        transaction: None,
        schedule_inputs: Vec::new(),
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::Random(Arc::new(
            plan.clone(),
        ))),
    })
}

fn required(plan: &RandomKernelPlan) -> Result<OpenClCapabilities, OpenClError> {
    let supported = match plan.kind {
        RandomKind::Uniform { .. } | RandomKind::Normal { .. } => matches!(
            plan.dtype,
            DType::F16 | DType::BF16 | DType::F32 | DType::F64
        ),
        RandomKind::RandInt { .. } => matches!(
            plan.dtype,
            DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::U8
                | DType::U16
                | DType::U32
                | DType::U64
        ),
    };
    if !supported {
        return Err(OpenClError::Unsupported(format!(
            "OpenCL captured Threefry {:?} dtype {:?}",
            plan.kind, plan.dtype
        )));
    }
    Ok(OpenClCapabilities {
        int64: matches!(plan.dtype, DType::I64 | DType::U64),
        // Existing raw narrow conversions deliberately use the exact fp64
        // helper contract.  F64 affine/normal arithmetic also needs it.
        fp64: matches!(plan.dtype, DType::F16 | DType::BF16 | DType::F64),
    })
}

fn cl_type(dtype: DType) -> Result<&'static str, OpenClError> {
    Ok(match dtype {
        DType::F16 | DType::BF16 | DType::U16 => "ushort",
        DType::F32 => "float",
        DType::F64 => "double",
        DType::I8 => "char",
        DType::U8 => "uchar",
        DType::I16 => "short",
        DType::I32 => "int",
        DType::U32 => "uint",
        DType::I64 => "long",
        DType::U64 => "ulong",
        DType::Bool => {
            return Err(OpenClError::Unsupported(
                "captured random does not define bool storage".into(),
            ));
        }
        DType::F8E4M3 | DType::F8E5M2 | DType::F8E4M3FNUZ | DType::F8E5M2FNUZ => {
            return Err(OpenClError::Unsupported(
                "captured random does not support float8 storage".into(),
            ));
        }
    })
}

fn uniform_value(dtype: DType, low: f64, high: f64) -> String {
    let unit = match dtype {
        DType::F16 => "rg_u16_f16(gid)",
        DType::BF16 => "rg_u16_bf16(gid)",
        DType::F64 => "rg_u64(gid)",
        _ => "rg_u32(gid)",
    };
    match dtype {
        DType::F16 => format!("rg_f32_to_f16((float)({low:?}f + ({high:?}f - {low:?}f) * {unit}))"),
        DType::BF16 => {
            format!("rg_f32_to_bf16((float)({low:?}f + ({high:?}f - {low:?}f) * {unit}))")
        }
        DType::F32 => format!("(float)({low:?}f + ({high:?}f - {low:?}f) * {unit})"),
        DType::F64 => format!("(double)({low:?} + ({high:?} - {low:?}) * {unit})"),
        _ => unreachable!(),
    }
}

fn normal_value(dtype: DType, mean: f64, std: f64) -> String {
    let base = "rg_normal(gid)";
    match dtype {
        DType::F16 => format!("rg_f32_to_f16((float)({mean:?}f + {std:?}f * {base}))"),
        DType::BF16 => format!("rg_f32_to_bf16((float)({mean:?}f + {std:?}f * {base}))"),
        DType::F32 => format!("(float)({mean:?}f + {std:?}f * {base})"),
        DType::F64 => format!("(double)({mean:?} + {std:?} * (double){base})"),
        _ => unreachable!(),
    }
}

fn randint_value(dtype: DType, low: i64, high: i64) -> Result<String, OpenClError> {
    let value = format!("((float){low:?} + (float)({high:?} - {low:?}) * rg_u32(gid))");
    let cast = match dtype {
        DType::I8 => "char",
        DType::U8 => "uchar",
        DType::I16 => "short",
        DType::U16 => "ushort",
        DType::I32 => "int",
        DType::U32 => "uint",
        DType::I64 => "long",
        DType::U64 => "ulong",
        _ => {
            return Err(OpenClError::Unsupported(
                "randint output must be integer storage".into(),
            ));
        }
    };
    Ok(format!("({cast})({value})"))
}

fn helpers(plan: &RandomKernelPlan, fp64: bool) -> String {
    // Match tinygrad's random_bits layout: chunks of 2^32-1 words, a derived
    // key per chunk, then all low lanes followed by high lanes. `gid` never
    // exceeds the captured plan word count because each caller derives its
    // source-word index from the checked output extent.
    format!(
        r#"static uint rg_rot(uint x, uint n) {{ return (x << n) | (x >> (32u-n)); }}
static void rg_tf(uint k0,uint k1,uint c0,uint c1,uint*o0,uint*o1) {{
  const uint r[8]={{13u,15u,26u,6u,17u,29u,16u,24u}}; uint k[3]={{k0,k1,k0^k1^0x1bd11bdau}}; uint a=c0+k0,b=c1+k1;
  for(uint q=0u;q<20u;q++) {{ a+=b; b=rg_rot(b,r[q&7u])^a; if((q&3u)==3u) {{ uint z=q/4u+1u; a+=k[z%3u]; b+=k[(z+1u)%3u]+z; }} }} *o0=a; *o1=b;
}}
static uint rg_word(ulong i) {{
  const ulong maxw=4294967295ul, words={words}ul; ulong chunk=i/maxw, offset=chunk*maxw, size=words-offset; if(size>maxw) size=maxw; ulong pairs=(size+1ul)/2ul, local=i-offset, lane=local<pairs?local:local-pairs;
  uint lo=(uint)offset, hi=(uint)(offset>>32), c0=lo+{c0}u, carry=(c0<lo), c1=hi+{c1}u+carry, dk0,dk1,a,b;
  rg_tf({k0}u,{k1}u,c0,c1,&dk0,&dk1); rg_tf(dk0,dk1,(uint)lane,(uint)(lane+pairs),&a,&b); return local<pairs?a:b;
}}
static float rg_u32(ulong i) {{ return as_float((rg_word(i)>>9)|0x3f800000u)-1.0f; }}
{f64_helper}static float rg_u16_f16(ulong i) {{ ushort h=(ushort)(((i&1ul)?rg_word(i/2ul)>>16:rg_word(i/2ul))>>6)|0x3c00u; uint e=((uint)h>>10)&31u,m=(uint)h&1023u,b=e?((e==31u?255u:e+112u)<<23)|(m<<13):m<<13; return as_float(b)-1.0f; }}
static float rg_u16_bf16(ulong i) {{ ushort h=(ushort)(((i&1ul)?rg_word(i/2ul)>>16:rg_word(i/2ul))>>9)|0x3f80u; return as_float(((uint)h)<<16)-1.0f; }}
static float rg_normal(ulong i) {{ float u0=rg_u32(i*2ul),u1=rg_u32(i*2ul+1ul); return cos(6.2831853071795864769f*u0)*sqrt(-2.0f*log(1.0f-u1)); }}"#,
        words = plan.word_count,
        c0 = plan.stream.counter[0],
        c1 = plan.stream.counter[1],
        k0 = plan.stream.key[0],
        k1 = plan.stream.key[1],
        f64_helper = if fp64 {
            "static double rg_u64(ulong i) { return as_double((((ulong)rg_word(i*2ul+1ul)<<32)|(ulong)rg_word(i*2ul))>>12|0x3ff0000000000000ul)-1.0; }\n"
        } else {
            ""
        },
    )
}

fn stable_key(value: &impl Hash) -> String {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    format!("{:#016x}", h.finish())
}
