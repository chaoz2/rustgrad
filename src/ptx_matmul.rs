//! Static serial-K PTX lowering from the immutable matmul plan.
use super::{
    KernelSemanticProgram, PTX_ABI_VERSION, PTX_RENDERER_VERSION, PtxBufferAbi, PtxError,
    PtxRenderer, RenderedPtx, stable_key,
};
use crate::{DType, MatmulKernelPlan};
use std::{collections::BTreeMap, sync::Arc};

pub(super) fn render(
    renderer: &PtxRenderer,
    plan: &MatmulKernelPlan,
) -> Result<RenderedPtx, PtxError> {
    if !matches!(
        (plan.lhs_dtype, plan.rhs_dtype, plan.dtype),
        (DType::F32, DType::F32, DType::F32) | (DType::F64, DType::F64, DType::F64)
    ) {
        return Err(PtxError::Unsupported(
            "static matmul PTX supports only homogeneous F32 or F64".into(),
        ));
    }
    let extent = plan.output_shape.numel().map_err(|_| PtxError::Overflow)?;
    let elems = |shape: &crate::Shape| shape.numel().map_err(|_| PtxError::Overflow);
    let buffers = vec![
        PtxBufferAbi {
            id: plan.lhs.index() as u64,
            dtype: plan.lhs_dtype,
            source_shape: plan.lhs_shape.clone(),
            elements: elems(&plan.lhs_shape)?,
            mutable: false,
        },
        PtxBufferAbi {
            id: plan.rhs.index() as u64,
            dtype: plan.rhs_dtype,
            source_shape: plan.rhs_shape.clone(),
            elements: elems(&plan.rhs_shape)?,
            mutable: false,
        },
        PtxBufferAbi {
            id: plan.output.index() as u64,
            dtype: plan.dtype,
            source_shape: plan.output_shape.clone(),
            elements: extent,
            mutable: true,
        },
    ];
    let ty = if plan.dtype == DType::F32 {
        "f32"
    } else {
        "f64"
    };
    let item = plan.dtype.itemsize();
    let entry = format!("rg_matmul_{}_{}", plan.cache_key, renderer.sm);
    let mut lines=vec![format!("// {PTX_RENDERER_VERSION} matmul ABI {PTX_ABI_VERSION}"),".version 7.0".into(),format!(".target sm_{}",renderer.sm),".address_size 64".into(),format!(".visible .entry {entry}(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 extent){{"),".reg .pred %p<4>;".into(),".reg .b32 %r<32>;".into(),".reg .b64 %rd<32>;".into(),".reg .f32 %f<8>;".into(),".reg .f64 %fd<8>;".into(),"ld.param.u64 %rd10,[p0]; ld.param.u64 %rd11,[p1]; ld.param.u64 %rd12,[p2]; ld.param.u64 %rd0,[extent];".into(),"mov.u32 %r0,%ctaid.x; mov.u32 %r1,%ntid.x; mov.u32 %r2,%tid.x; mad.lo.u32 %r3,%r0,%r1,%r2; cvt.u64.u32 %rd1,%r3; setp.ge.u64 %p0,%rd1,%rd0; @%p0 bra DONE;".into()];
    // Decompose the linear output into N, M and broadcast batch. Static dimensions
    // make this deterministic and avoid any host-side coordinate reconstruction.
    lines.push("mov.u64 %rd2,%rd1; mov.u64 %rd3,0; mov.u64 %rd4,0;".into());
    if !plan.rhs_vector {
        lines.push(format!(
            "rem.u64 %rd4,%rd2,{}; div.u64 %rd2,%rd2,{};",
            plan.n, plan.n
        ));
    }
    if !plan.lhs_vector {
        lines.push(format!(
            "rem.u64 %rd3,%rd2,{}; div.u64 %rd2,%rd2,{};",
            plan.m, plan.m
        ));
    }
    lines.push("mov.u64 %rd5,%rd2; mov.u64 %rd6,%rd2;".into());
    // Convert broadcast batch linear coordinate to each input's packed batch offset.
    let batch_offset = |shape: &crate::Shape, vector: bool, reg: &str| -> String {
        let dims = shape.dims();
        let batch = if vector {
            &[][..]
        } else {
            &dims[..dims.len() - 2]
        };
        let mut s = String::new();
        for (i, d) in batch.iter().enumerate().rev() {
            s.push_str(&format!(
                " rem.u64 %rd20,{reg},{d}; div.u64 {reg},{reg},{d};"
            ));
            let stride = batch[i + 1..].iter().product::<usize>();
            if *d != 1 {
                s.push_str(&format!(" mad.lo.u64 %rd7,%rd20,{stride},%rd7;"));
            }
        }
        s
    };
    lines.push("mov.u64 %rd7,0;".into());
    lines.push(batch_offset(&plan.lhs_shape, plan.lhs_vector, "%rd5"));
    lines.push("mov.u64 %rd8,0;".into());
    lines.push(batch_offset(&plan.rhs_shape, plan.rhs_vector, "%rd6"));
    lines.push(if ty == "f32" {
        "mov.f32 %f0,0f00000000;".into()
    } else {
        "mov.f64 %fd0,0d0000000000000000;".into()
    });
    lines.push(
        "mov.u32 %r4,0; LOOP: setp.ge.u32 %p1,%r4,".to_string()
            + &plan.k.to_string()
            + "; @%p1 bra STORE;",
    );
    // lhs: batch*M*K + row*K + k; rhs: batch*K*N + k*N + col.
    lines.push(format!("mad.lo.u64 %rd21,%rd7,{},%rd3; mad.lo.u64 %rd21,%rd21,{},%r4; mad.lo.u64 %rd22,%rd8,{},%r4; mad.lo.u64 %rd22,%rd22,{},%rd4;",plan.m,plan.k,plan.k,plan.n));
    let reg = if ty == "f32" { "f" } else { "fd" };
    lines.push(format!("mul.lo.u64 %rd21,%rd21,{item}; add.u64 %rd21,%rd10,%rd21; mul.lo.u64 %rd22,%rd22,{item}; add.u64 %rd22,%rd11,%rd22; ld.global.{ty} %{reg}1,[%rd21]; ld.global.{ty} %{reg}2,[%rd22]; fma.rn.{ty} %{reg}0,%{reg}0,%{reg}1,%{reg}2; add.u32 %r4,%r4,1; bra LOOP;"));
    lines.push(format!("STORE: mul.lo.u64 %rd24,%rd1,{item}; add.u64 %rd24,%rd12,%rd24; st.global.{ty} [%rd24],%{}0; DONE: ret; }}",if ty=="f32"{"f"}else{"fd"}));
    let source = lines.join("\n") + "\n";
    Ok(RenderedPtx {
        source_map: BTreeMap::from([(0, 1)]),
        cache_key: stable_key(&(PTX_RENDERER_VERSION, renderer.sm, &plan.cache_key, &source)),
        source,
        buffers,
        extent,
        entry,
        semantic_program: Some(KernelSemanticProgram::Matmul(Arc::new(plan.clone()))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Shape};
    #[test]
    fn serial_k_source_has_ordered_abi_and_rejects_narrow_storage() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", Shape::from([2, 3]), DType::F32);
        let rhs = graph.input_dtype("rhs", Shape::from([3, 2]), DType::F32);
        let out = graph.matmul(lhs, rhs).unwrap();
        let plan = MatmulKernelPlan::from_graph(&graph, out).unwrap();
        let first = PtxRenderer::new(80)
            .unwrap()
            .render_matmul_plan(&plan)
            .unwrap();
        let second = PtxRenderer::new(80)
            .unwrap()
            .render_matmul_plan(&plan)
            .unwrap();
        assert_eq!(first.cache_key, second.cache_key);
        assert_eq!(
            first.buffers.iter().map(|b| b.id).collect::<Vec<_>>(),
            vec![lhs.index() as u64, rhs.index() as u64, out.index() as u64]
        );
        assert!(first.source.contains("LOOP:"));
        assert!(matches!(
            first.semantic_program,
            Some(KernelSemanticProgram::Matmul(_))
        ));
        let mut narrow = Graph::new();
        let a = narrow.input_dtype("a", [2, 2], DType::F16);
        let b = narrow.input_dtype("b", [2, 2], DType::F16);
        let z = narrow.matmul(a, b).unwrap();
        let p = MatmulKernelPlan::from_graph(&narrow, z).unwrap();
        assert!(matches!(
            PtxRenderer::new(80).unwrap().render_matmul_plan(&p),
            Err(PtxError::Unsupported(_))
        ));
    }
}
