//! Graph-free MSL lowering for immutable captured Threefry random plans.
//!
//! Metal's current exact storage contract is deliberately limited to F32,
//! I32, and U32.  The source has one output pointer and one extent scalar;
//! stream registry state is never read at render or launch time.

use super::{
    MetalBufferAbi, MetalError, MetalRenderer, RenderedMetal,
    renderer::{METAL_ABI_VERSION, METAL_RENDERER_VERSION},
};
use crate::{DType, RandomKind, random::plan::RandomKernelPlan};
use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::Arc,
};

pub(super) fn render(
    renderer: &MetalRenderer,
    plan: &RandomKernelPlan,
) -> Result<RenderedMetal, MetalError> {
    plan.validate()
        .map_err(|error| MetalError::Unsupported(error.to_string()))?;
    supported(plan)?;
    let extent = plan.shape.numel().map_err(|_| MetalError::Overflow)?;
    let entry = format!("rg_metal_random_e{extent}");
    let mut lines = vec![
        "#include <metal_stdlib>".into(),
        "using namespace metal;".into(),
        format!("// {METAL_RENDERER_VERSION} captured-threefry ABI {METAL_ABI_VERSION}"),
        helpers(plan),
        format!(
            "kernel void {entry}(device {}* out [[buffer(0)]], constant ulong& extent [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{",
            storage_type(plan.dtype)
        ),
        "  if ((ulong)gid >= extent) return;".into(),
    ];
    let value = match plan.kind {
        RandomKind::Uniform { low, high } => {
            format!("(float)({low:?}f + ({high:?}f - {low:?}f) * rg_u32((ulong)gid))")
        }
        RandomKind::Normal { mean, std } => {
            format!("(float)({mean:?}f + {std:?}f * rg_normal((ulong)gid))")
        }
        RandomKind::RandInt { low, high } => {
            let unit =
                format!("((float){low:?} + (float)({high:?} - {low:?}) * rg_u32((ulong)gid))");
            match plan.dtype {
                DType::I32 => format!("rg_i32({unit})"),
                DType::U32 => format!("rg_u32_cast({unit})"),
                _ => unreachable!("validated random integer storage"),
            }
        }
    };
    let source_line = lines.len() + 1;
    lines.push(format!("  out[gid] = {value};"));
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let buffer = MetalBufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        source_shape: plan.shape.clone(),
        elements: extent,
        mutable: true,
        view: None,
    };
    Ok(RenderedMetal {
        source: source.clone(),
        source_map: BTreeMap::from([(plan.output.index(), source_line)]),
        buffers: vec![buffer],
        extent,
        entry,
        cache_key: stable_key(&(
            METAL_RENDERER_VERSION,
            METAL_ABI_VERSION,
            renderer.local_size,
            &renderer.capabilities,
            &source,
            plan,
        )),
        capabilities: renderer.capabilities.clone(),
        transaction: None,
        indexed_movement: None,
        append_state: None,
        schedule_inputs: Vec::new(),
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::Random(Arc::new(
            plan.clone(),
        ))),
    })
}

fn supported(plan: &RandomKernelPlan) -> Result<(), MetalError> {
    let valid = match plan.kind {
        RandomKind::Uniform { .. } | RandomKind::Normal { .. } => plan.dtype == DType::F32,
        RandomKind::RandInt { .. } => matches!(plan.dtype, DType::I32 | DType::U32),
    };
    if valid {
        Ok(())
    } else {
        Err(MetalError::Unsupported(format!(
            "Metal captured Threefry {:?} dtype {:?} exceeds the F32/I32/U32 storage contract",
            plan.kind, plan.dtype
        )))
    }
}

fn storage_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::I32 => "int",
        DType::U32 => "uint",
        _ => unreachable!("validated Metal random storage"),
    }
}

fn helpers(plan: &RandomKernelPlan) -> String {
    // This is the exact tinygrad random_bits shape: a carry-safe 2^32-1-word
    // chunk counter, derived Threefry key, then low lanes followed by highs.
    format!(
        r#"inline uint rg_rot(uint x, uint n) {{ return (x << n) | (x >> (32u-n)); }}
inline void rg_tf(uint k0,uint k1,uint c0,uint c1,thread uint&o0,thread uint&o1) {{
  const uint r[8]={{13u,15u,26u,6u,17u,29u,16u,24u}}; uint k[3]={{k0,k1,k0^k1^0x1bd11bdau}}; uint a=c0+k0,b=c1+k1;
  for(uint q=0u;q<20u;q++) {{ a+=b; b=rg_rot(b,r[q&7u])^a; if((q&3u)==3u) {{ uint z=q/4u+1u; a+=k[z%3u]; b+=k[(z+1u)%3u]+z; }} }} o0=a; o1=b;
}}
inline uint rg_word(ulong i) {{
  const ulong maxw=4294967295ul, words={words}ul; ulong chunk=i/maxw, offset=chunk*maxw, size=words-offset; if(size>maxw) size=maxw; ulong pairs=(size+1ul)/2ul, local=i-offset, lane=local<pairs?local:local-pairs;
  uint lo=(uint)offset, hi=(uint)(offset>>32), c0=lo+{c0}u, carry=(c0<lo), c1=hi+{c1}u+carry, dk0,dk1,a,b;
  rg_tf({k0}u,{k1}u,c0,c1,dk0,dk1); rg_tf(dk0,dk1,(uint)lane,(uint)(lane+pairs),a,b); return local<pairs?a:b;
}}
inline float rg_u32(ulong i) {{ return as_type<float>((rg_word(i)>>9)|0x3f800000u)-1.0f; }}
inline float rg_normal(ulong i) {{ float u0=rg_u32(i*2ul),u1=rg_u32(i*2ul+1ul); return cos(6.2831853071795864769f*u0)*sqrt(-2.0f*log(1.0f-u1)); }}
inline int rg_i32(float x) {{ return x >= 2147483520.0f ? 2147483647 : (x <= -2147483648.0f ? (-2147483647-1) : (int)x); }}
inline uint rg_u32_cast(float x) {{ return x >= 4294967040.0f ? 0xffffffffu : (x <= 0.0f ? 0u : (uint)x); }}"#,
        words = plan.word_count,
        c0 = plan.stream.counter[0],
        c1 = plan.stream.counter[1],
        k0 = plan.stream.key[0],
        k1 = plan.stream.key[1],
    )
}

fn stable_key(value: &impl Hash) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
