//! Graph-free WGSL lowering for immutable captured Threefry plans.
//!
//! The current safe WebGPU random surface uses only the renderer's unpacked
//! F32/I32/U32 storage ABI. Narrow packed storage remains an explicit future
//! extension because a random write needs a separate disjoint-lane protocol.

use super::{
    RenderedWgsl, WebGpuError, WgslBufferAbi, WgslRenderer,
    renderer::{WEBGPU_ABI_VERSION, WEBGPU_STATUS_VERSION, WGSL_RENDERER_VERSION},
};
use crate::{DType, RandomKind, random::plan::RandomKernelPlan};
use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::Arc,
};

pub(super) fn render(
    renderer: &WgslRenderer,
    plan: &RandomKernelPlan,
) -> Result<RenderedWgsl, WebGpuError> {
    plan.validate()
        .map_err(|error| WebGpuError::Unsupported(error.to_string()))?;
    supported(plan)?;
    let extent = plan.shape.numel().map_err(|_| WebGpuError::Overflow)?;
    if plan.word_count > u32::MAX as usize {
        return Err(WebGpuError::Unsupported(
            "random word count exceeds WGSL u32 source indexing".into(),
        ));
    }
    if extent > u32::MAX as usize {
        return Err(WebGpuError::Unsupported(
            "random extent exceeds WGSL u32 indexing".into(),
        ));
    }
    let buffer = WgslBufferAbi {
        id: plan.output.index() as u64,
        dtype: plan.dtype,
        source_shape: plan.shape.clone(),
        elements: extent,
        mutable: true,
        view: None,
    };
    if buffer.physical_bytes()? > renderer.capabilities.max_buffer_size {
        return Err(WebGpuError::Unsupported(
            "random output exceeds adapter buffer limit".into(),
        ));
    }
    if renderer.capabilities.max_storage_buffers_per_shader_stage < 1 {
        return Err(WebGpuError::Unsupported(
            "adapter has no storage-buffer binding capacity".into(),
        ));
    }
    let entry = format!("rg_webgpu_random_e{extent}");
    let value = match plan.kind {
        RandomKind::Uniform { low, high } => {
            format!("{low:?}f + ({high:?}f - {low:?}f) * rg_u32(gid.x)")
        }
        RandomKind::Normal { mean, std } => format!("{mean:?}f + {std:?}f * rg_normal(gid.x)"),
        RandomKind::RandInt { low, high } => {
            let x = format!("(f32({low:?}) + f32({high:?} - {low:?}) * rg_u32(gid.x))");
            if plan.dtype == DType::I32 {
                format!("rg_i32({x})")
            } else {
                format!("rg_u32_cast({x})")
            }
        }
    };
    let mut lines = vec![
        format!(
            "// {WGSL_RENDERER_VERSION} captured-threefry ABI {WEBGPU_ABI_VERSION} STATUS {WEBGPU_STATUS_VERSION}"
        ),
        "struct RustGradExtent { value: u32, };".into(),
        helpers(plan),
        format!(
            "@group(0) @binding(0) var<storage, read_write> out: array<{}>;",
            ty(plan.dtype)
        ),
        "@group(0) @binding(1) var<uniform> rg_extent: RustGradExtent;".into(),
        format!(
            "@compute @workgroup_size({}) fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{",
            renderer.local_size
        ),
        "  if (gid.x >= rg_extent.value) { return; }".into(),
    ];
    let line = lines.len() + 1;
    lines.push(format!("  out[gid.x] = {value};"));
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    Ok(RenderedWgsl {
        source: source.clone(),
        source_map: BTreeMap::from([(plan.output.index(), line)]),
        buffers: vec![buffer],
        extent,
        entry,
        cache_key: key(&(
            WGSL_RENDERER_VERSION,
            WEBGPU_ABI_VERSION,
            WEBGPU_STATUS_VERSION,
            renderer.local_size,
            &renderer.capabilities,
            &source,
            plan,
        )),
        capabilities: renderer.capabilities.clone(),
        local_size: renderer.local_size,
        transaction: None,
        schedule_inputs: Vec::new(),
        semantic_program: Arc::new(super::dispatch::KernelSemanticProgram::Random(Arc::new(
            plan.clone(),
        ))),
    })
}
fn supported(plan: &RandomKernelPlan) -> Result<(), WebGpuError> {
    let ok = match plan.kind {
        RandomKind::Uniform { .. } | RandomKind::Normal { .. } => plan.dtype == DType::F32,
        RandomKind::RandInt { .. } => matches!(plan.dtype, DType::I32 | DType::U32),
    };
    if ok {
        Ok(())
    } else {
        Err(WebGpuError::Unsupported(format!(
            "WebGPU captured Threefry {:?} dtype {:?} exceeds unpacked F32/I32/U32 storage",
            plan.kind, plan.dtype
        )))
    }
}
fn ty(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::U32 => "u32",
        _ => unreachable!(),
    }
}
fn helpers(plan: &RandomKernelPlan) -> String {
    format!(
        r#"fn rg_rot(x:u32,n:u32)->u32{{return (x<<n)|(x>>(32u-n));}}
fn rg_tf(k0:u32,k1:u32,c0:u32,c1:u32)->vec2<u32>{{var k: array<u32, 3> = array<u32, 3>(k0,k1,k0^k1^0x1bd11bdau);var a=c0+k0;var b=c1+k1;let r: array<u32, 8> = array<u32, 8>(13u,15u,26u,6u,17u,29u,16u,24u);for(var q=0u;q<20u;q=q+1u){{a=a+b;b=rg_rot(b,r[q&7u])^a;if((q&3u)==3u){{let z=q/4u+1u;a=a+k[z%3u];b=b+k[(z+1u)%3u]+z;}}}}return vec2<u32>(a,b);}}
fn rg_word(i:u32)->u32{{let maxw:u32=0xffffffffu;let words:u32={words}u;let chunk=i/maxw;let offset=chunk*maxw;let size=min(words-offset,maxw);let pairs=(size+1u)/2u;let local=i-offset;let lane=select(local-pairs,local,local<pairs);let lo=offset;let c0=lo+{c0}u;let c1={c1}u+select(0u,1u,c0<lo);let dk=rg_tf({k0}u,{k1}u,c0,c1);let pair=rg_tf(dk.x,dk.y,lane,lane+pairs);return select(pair.y,pair.x,local<pairs);}}
fn rg_u32(i:u32)->f32{{return bitcast<f32>((rg_word(i)>>9u)|0x3f800000u)-1.0;}}
fn rg_normal(i:u32)->f32{{let u0=rg_u32(i*2u);let u1=rg_u32(i*2u+1u);return cos(6.283185307179586f*u0)*sqrt(-2.0*log(1.0-u1));}}
fn rg_i32(x:f32)->i32{{if(x>=2147483648.0){{return bitcast<i32>(0x7fffffffu);}}if(x<=-2147483648.0){{return bitcast<i32>(0x80000000u);}}return i32(x);}}
fn rg_u32_cast(x:f32)->u32{{if(x>=4294967296.0){{return 0xffffffffu;}}if(x<=0.0){{return 0u;}}return u32(x);}}"#,
        words = plan.word_count,
        c0 = plan.stream.counter[0],
        c1 = plan.stream.counter[1],
        k0 = plan.stream.key[0],
        k1 = plan.stream.key[1]
    )
}
fn key(x: &impl Hash) -> String {
    let mut h = DefaultHasher::new();
    x.hash(&mut h);
    format!("{:016x}", h.finish())
}
