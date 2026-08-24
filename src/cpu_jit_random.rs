//! Native C11 lowering for immutable captured Threefry random plans.
//!
//! This is deliberately a child of the CPU JIT renderer: it consumes only the
//! typed plan and produces a graph-free one-output ABI.  It never asks the
//! mutable implicit-stream registry for state.

use super::{
    ABI_VERSION, BufferAbi, DType, JitError, KernelAbi, KernelPointerAbi, RENDERER_VERSION,
    RenderedC, key,
};
use crate::{RandomKind, random::plan::RandomKernelPlan};
use std::collections::BTreeMap;

pub(super) fn render(plan: &RandomKernelPlan) -> Result<RenderedC, JitError> {
    plan.validate()
        .map_err(|error| JitError::Unsupported(error.to_string()))?;
    let count = plan
        .shape
        .numel()
        .map_err(|error| JitError::Unsupported(error.to_string()))?;
    let abi = KernelAbi {
        version: ABI_VERSION,
        buffers: vec![BufferAbi {
            id: plan.output.index() as u64,
            dtype: plan.dtype,
            elements: count,
            mutable: true,
        }],
        quantized_buffers: vec![],
        pointer_order: vec![KernelPointerAbi::Dense(0)],
        symbol_count: 0,
    };
    let (kind, unit, value) = match plan.kind {
        RandomKind::Uniform { low, high } => (
            "uniform",
            uniform_unit(plan.dtype)?,
            format!("({low:?}) + (({high:?}) - ({low:?})) * rg_u"),
        ),
        RandomKind::Normal { mean, std } => (
            "normal",
            "rg_normal((uint64_t)i)",
            format!("({mean:?}) + ({std:?}) * rg_u"),
        ),
        RandomKind::RandInt { low, high } => (
            "randint",
            "rg_u32((uint64_t)i)",
            format!("({low:?}) + (({high:?}) - ({low:?})) * rg_u"),
        ),
    };
    let store = store_expression(plan.dtype, &value)?;
    let source = format!(
        "#include <stdint.h>\n#include <stddef.h>\n#include <math.h>\n#include <limits.h>\n/* rustgrad captured Threefry C11 {kind}; immutable key/counter */\n{helpers}\nint rustgrad_kernel(void **buffers,const int64_t*symbols,uint64_t*failure){{(void)symbols;failure[0]=UINT64_MAX;failure[1]=0;{ctype}*out=({ctype}*)buffers[0];for(size_t i=0;i<{count}u;i++){{double rg_u={unit};out[i]={store};}}return 0;}}\n",
        kind = kind,
        helpers = helpers(
            plan.stream.key[0],
            plan.stream.key[1],
            plan.stream.counter[0],
            plan.stream.counter[1],
            plan.word_count.div_ceil(2),
        ),
        ctype = ctype(plan.dtype),
        count = count,
        unit = unit,
        store = store,
    );
    let cache_key = key(&(RENDERER_VERSION.to_owned()
        + std::env::consts::ARCH
        + std::env::consts::OS
        + &source));
    Ok(RenderedC {
        source,
        source_map: BTreeMap::new(),
        abi,
        cache_key,
    })
}

fn uniform_unit(dtype: DType) -> Result<&'static str, JitError> {
    match dtype {
        DType::F16 => Ok("rg_u16_f16((uint64_t)i)"),
        DType::BF16 => Ok("rg_u16_bf16((uint64_t)i)"),
        DType::F32 => Ok("rg_u32((uint64_t)i)"),
        DType::F64 => Ok("rg_u64((uint64_t)i)"),
        _ => Err(JitError::Unsupported(
            "uniform requires floating output storage".into(),
        )),
    }
}

fn store_expression(dtype: DType, value: &str) -> Result<String, JitError> {
    Ok(match dtype {
        DType::F16 => format!("rg_f32_to_f16((float)({value}))"),
        DType::BF16 => format!("rg_f32_to_bf16((float)({value}))"),
        DType::F32 => format!("(float)({value})"),
        DType::F64 => value.into(),
        DType::I8 => format!("rg_i8({value})"),
        DType::U8 => format!("rg_u8({value})"),
        DType::I16 => format!("rg_i16({value})"),
        DType::U16 => format!("rg_u16({value})"),
        DType::I32 => format!("rg_i32({value})"),
        DType::U32 => format!("rg_u32_cast({value})"),
        DType::I64 => format!("rg_i64({value})"),
        DType::U64 => format!("rg_u64_cast({value})"),
        DType::Bool => {
            return Err(JitError::Unsupported(
                "captured random does not define bool output storage".into(),
            ));
        }
    })
}

fn ctype(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 | DType::BF16 | DType::U16 => "uint16_t",
        DType::F32 => "float",
        DType::F64 => "double",
        DType::I8 => "int8_t",
        DType::U8 | DType::Bool => "uint8_t",
        DType::I16 => "int16_t",
        DType::I32 => "int32_t",
        DType::U32 => "uint32_t",
        DType::I64 => "int64_t",
        DType::U64 => "uint64_t",
    }
}

fn helpers(k0: u32, k1: u32, c0: u32, c1: u32, pairs: usize) -> String {
    format!(
        "static uint32_t rg_rot(uint32_t x,unsigned n){{return(x<<n)|(x>>(32-n));}}\nstatic void rg_tf(uint32_t k0,uint32_t k1,uint32_t c0,uint32_t c1,uint32_t*o0,uint32_t*o1){{static const unsigned r[8]={{13,15,26,6,17,29,16,24}};uint32_t k[3]={{k0,k1,k0^k1^0x1bd11bdau}},a=c0+k0,b=c1+k1;for(unsigned q=0;q<20;q++){{a+=b;b=rg_rot(b,r[q&7])^a;if((q&3)==3){{unsigned z=q/4+1;a+=k[z%3];b+=k[(z+1)%3]+z;}}}}*o0=a;*o1=b;}}\nstatic uint32_t rg_word(uint64_t i){{uint32_t dk0,dk1,a,b;rg_tf({k0}u,{k1}u,{c0}u,{c1}u,&dk0,&dk1);uint64_t pairs={pairs}ull;uint64_t lane=i<pairs?i:i-pairs;rg_tf(dk0,dk1,(uint32_t)lane,(uint32_t)(lane+pairs),&a,&b);return i<pairs?a:b;}}\nstatic double rg_u32(uint64_t i){{union{{uint32_t u;float f;}}v={{(rg_word(i)>>9)|0x3f800000u}};return(double)v.f-1.0;}}\nstatic double rg_u64(uint64_t i){{uint64_t lo=rg_word(i*2u),hi=rg_word(i*2u+1u);union{{uint64_t u;double f;}}v={{((hi<<32)|lo)>>12|0x3ff0000000000000ull}};return v.f-1.0;}}\nstatic double rg_u16_f16(uint64_t i){{uint16_t w=(uint16_t)(rg_word(i/2u)>>(i&1u?16:0));uint16_t h=(uint16_t)((w>>6)|0x3c00);uint32_t e=(uint32_t)(h>>10)&31,m=(uint32_t)h&1023,b=e?((e==31?255:e+112)<<23)|(m<<13):m<<13;union{{uint32_t u;float f;}}v={{b}};return(double)v.f-1.0;}}\nstatic double rg_u16_bf16(uint64_t i){{uint16_t w=(uint16_t)(rg_word(i/2u)>>(i&1u?16:0));union{{uint32_t u;float f;}}v={{(uint32_t)((w>>9)|0x3f80)<<16}};return(double)v.f-1.0;}}\nstatic double rg_normal(uint64_t i){{double u0=rg_u32(i*2u),u1=rg_u32(i*2u+1u);return cos(6.283185307179586476925286766559*u0)*sqrt(-2.0*log(1.0-u1));}}\nstatic uint16_t rg_f32_to_f16(float x){{union{{float f;uint32_t u;}}v={{x}};uint32_t b=v.u,s=(b>>16)&0x8000,e=(b>>23)&255,m=b&0x7fffff;if(e==255)return(uint16_t)(s|0x7c00|(m?((m>>13)|1):0));int q=(int)e-112;if(q<=0){{if(q<-10)return(uint16_t)s;uint32_t z=m|0x800000,sh=(uint32_t)(14-q),r=z>>sh,rem=z&((1u<<sh)-1),half=1u<<(sh-1);return(uint16_t)(s+r+(rem>half||(rem==half&&(r&1))));}}if(q>=31)return(uint16_t)(s|0x7c00);uint32_t r=m>>13,rem=m&0x1fff;r+=rem>0x1000||(rem==0x1000&&(r&1));if(r==0x400){{if(q==30)return(uint16_t)(s|0x7c00);q++;r=0;}}return(uint16_t)(s|((uint32_t)q<<10)|r);}}\nstatic uint16_t rg_f32_to_bf16(float x){{union{{float f;uint32_t u;}}v={{x}};uint32_t b=v.u,hi=b>>16;if((b&0x7f800000)==0x7f800000&&(b&0x007fffff))return(uint16_t)((hi&0x7f)?hi:(hi|1));return(uint16_t)((b+0x7fff+((b>>16)&1))>>16);}}\nstatic int8_t rg_i8(double x){{return x>=INT8_MAX?INT8_MAX:x<=INT8_MIN?INT8_MIN:(int8_t)x;}} static uint8_t rg_u8(double x){{return x>=UINT8_MAX?UINT8_MAX:x<=0?0:(uint8_t)x;}} static int16_t rg_i16(double x){{return x>=INT16_MAX?INT16_MAX:x<=INT16_MIN?INT16_MIN:(int16_t)x;}} static uint16_t rg_u16(double x){{return x>=UINT16_MAX?UINT16_MAX:x<=0?0:(uint16_t)x;}} static int32_t rg_i32(double x){{return x>=INT32_MAX?INT32_MAX:x<=INT32_MIN?INT32_MIN:(int32_t)x;}} static uint32_t rg_u32_cast(double x){{return x>=UINT32_MAX?UINT32_MAX:x<=0?0:(uint32_t)x;}} static int64_t rg_i64(double x){{return x>=(double)INT64_MAX?INT64_MAX:x<=(double)INT64_MIN?INT64_MIN:(int64_t)x;}} static uint64_t rg_u64_cast(double x){{return x>=(double)UINT64_MAX?UINT64_MAX:x<=0?0:(uint64_t)x;}}\n",
        k0 = k0,
        k1 = k1,
        c0 = c0,
        c1 = c1,
        pairs = pairs,
    )
}
