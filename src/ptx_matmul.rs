//! Static serial-K PTX lowering from the immutable matmul plan.
use super::{
    KernelSemanticProgram, PTX_ABI_VERSION, PTX_RENDERER_VERSION, PtxBufferAbi, PtxError,
    PtxLaunchGeometry, PtxRenderer, RenderedPtx, stable_key,
};
use crate::{DType, MatmulKernelPlan, TiledMatmulPayload};
use std::{collections::BTreeMap, sync::Arc};

pub(super) fn render_serial(
    renderer: &PtxRenderer,
    plan: &MatmulKernelPlan,
) -> Result<RenderedPtx, PtxError> {
    plan.validate()
        .map_err(|error| PtxError::Unsupported(error.to_string()))?;
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
    if !plan.rhs_vector && plan.n != 0 {
        lines.push(format!(
            "rem.u64 %rd4,%rd2,{}; div.u64 %rd2,%rd2,{};",
            plan.n, plan.n
        ));
    }
    if !plan.lhs_vector && plan.m != 0 {
        lines.push(format!(
            "rem.u64 %rd3,%rd2,{}; div.u64 %rd2,%rd2,{};",
            plan.m, plan.m
        ));
    }
    lines.push("mov.u64 %rd5,%rd2; mov.u64 %rd6,%rd2;".into());
    // Convert broadcast batch linear coordinate to each input's packed batch offset.
    let batch_offset =
        |shape: &crate::Shape, vector: bool, reg: &str, accumulator: &str| -> String {
            let dims = shape.dims();
            let batch = if vector {
                &[][..]
            } else {
                &dims[..dims.len() - 2]
            };
            if plan.batch_shape.contains(&0) {
                return String::new();
            }
            let pad = plan.batch_shape.len() - batch.len();
            let mut s = String::new();
            for axis in (0..plan.batch_shape.len()).rev() {
                let d = plan.batch_shape[axis];
                s.push_str(&format!(
                    " rem.u64 %rd20,{reg},{d}; div.u64 {reg},{reg},{d};"
                ));
                if axis >= pad && batch[axis - pad] != 1 {
                    let stride = batch[axis - pad + 1..].iter().product::<usize>();
                    s.push_str(&format!(
                        " mad.lo.u64 {accumulator},%rd20,{stride},{accumulator};"
                    ));
                }
            }
            s
        };
    lines.push("mov.u64 %rd7,0;".into());
    lines.push(batch_offset(
        &plan.lhs_shape,
        plan.lhs_vector,
        "%rd5",
        "%rd7",
    ));
    lines.push("mov.u64 %rd8,0;".into());
    lines.push(batch_offset(
        &plan.rhs_shape,
        plan.rhs_vector,
        "%rd6",
        "%rd8",
    ));
    lines.push("mov.f64 %fd0,0d0000000000000000;".into());
    lines.push(
        "mov.u32 %r4,0; LOOP: setp.ge.u32 %p1,%r4,".to_string()
            + &plan.k.to_string()
            + "; @%p1 bra STORE;",
    );
    // lhs: batch*M*K + row*K + k; rhs: batch*K*N + k*N + col.
    lines.push(format!("mad.lo.u64 %rd21,%rd7,{},%rd3; mad.lo.u64 %rd21,%rd21,{},%r4; mad.lo.u64 %rd22,%rd8,{},%r4; mad.lo.u64 %rd22,%rd22,{},%rd4;",plan.m,plan.k,plan.k,plan.n));
    lines.push(if ty == "f32" {
        format!("mul.lo.u64 %rd21,%rd21,{item}; add.u64 %rd21,%rd10,%rd21; mul.lo.u64 %rd22,%rd22,{item}; add.u64 %rd22,%rd11,%rd22; ld.global.f32 %f1,[%rd21]; ld.global.f32 %f2,[%rd22]; cvt.f64.f32 %fd1,%f1; cvt.f64.f32 %fd2,%f2; mul.rn.f64 %fd3,%fd1,%fd2; add.rn.f64 %fd0,%fd0,%fd3; add.u32 %r4,%r4,1; bra LOOP;")
    } else {
        format!("mul.lo.u64 %rd21,%rd21,{item}; add.u64 %rd21,%rd10,%rd21; mul.lo.u64 %rd22,%rd22,{item}; add.u64 %rd22,%rd11,%rd22; ld.global.f64 %fd1,[%rd21]; ld.global.f64 %fd2,[%rd22]; mul.rn.f64 %fd3,%fd1,%fd2; add.rn.f64 %fd0,%fd0,%fd3; add.u32 %r4,%r4,1; bra LOOP;")
    });
    lines.push(if ty == "f32" {
        format!("STORE: mul.lo.u64 %rd24,%rd1,{item}; add.u64 %rd24,%rd12,%rd24; cvt.rn.f32.f64 %f0,%fd0; st.global.f32 [%rd24],%f0; DONE: ret; }}")
    } else {
        format!("STORE: mul.lo.u64 %rd24,%rd1,{item}; add.u64 %rd24,%rd12,%rd24; st.global.f64 [%rd24],%fd0; DONE: ret; }}")
    });
    let source = lines.join("\n") + "\n";
    Ok(RenderedPtx {
        source_map: BTreeMap::from([(0, 1)]),
        cache_key: stable_key(&(PTX_RENDERER_VERSION, renderer.sm, &plan.cache_key, &source)),
        source,
        buffers,
        extent,
        entry,
        launch: PtxLaunchGeometry::Linear,
        semantic_program: Some(KernelSemanticProgram::Matmul(Arc::new(plan.clone()))),
    })
}

pub(super) fn render_tiled(
    renderer: &PtxRenderer,
    payload: &TiledMatmulPayload,
) -> Result<RenderedPtx, PtxError> {
    payload
        .validate()
        .map_err(|error| PtxError::Unsupported(error.to_string()))?;
    if renderer.sm < payload.tile.target.sm {
        return Err(PtxError::Unsupported(format!(
            "tiled matmul targets sm_{} but renderer is sm_{}",
            payload.tile.target.sm, renderer.sm
        )));
    }
    let plan = &payload.matmul;
    if (plan.lhs_dtype, plan.rhs_dtype, plan.dtype) != (DType::F32, DType::F32, DType::F32) {
        return Err(PtxError::Unsupported(
            "tiled matmul PTX supports homogeneous F32 only".into(),
        ));
    }
    let extent = plan.output_shape.numel().map_err(|_| PtxError::Overflow)?;
    let elements = |shape: &crate::Shape| shape.numel().map_err(|_| PtxError::Overflow);
    let buffers = vec![
        PtxBufferAbi {
            id: plan.lhs.index() as u64,
            dtype: DType::F32,
            source_shape: plan.lhs_shape.clone(),
            elements: elements(&plan.lhs_shape)?,
            mutable: false,
        },
        PtxBufferAbi {
            id: plan.rhs.index() as u64,
            dtype: DType::F32,
            source_shape: plan.rhs_shape.clone(),
            elements: elements(&plan.rhs_shape)?,
            mutable: false,
        },
        PtxBufferAbi {
            id: plan.output.index() as u64,
            dtype: DType::F32,
            source_shape: plan.output_shape.clone(),
            elements: extent,
            mutable: true,
        },
    ];
    let tile = &payload.tile;
    let memory = crate::plan_tiled_matmul_promotion(payload)
        .map_err(|error| PtxError::Unsupported(error.to_string()))?;
    let launch = tile
        .launch_geometry(plan)
        .map_err(|error| PtxError::Unsupported(error.to_string()))?;
    let entry = format!("rg_tiled_matmul_{}_{}", tile.cache_key, renderer.sm);
    let mut lines = vec![
        format!("// {PTX_RENDERER_VERSION} tiled-matmul ABI {PTX_ABI_VERSION}"),
        ".version 7.0".into(),
        format!(".target sm_{}", renderer.sm),
        ".address_size 64".into(),
        format!(
            ".visible .entry {entry}(.param .u64 p0,.param .u64 p1,.param .u64 p2,.param .u64 extent){{"
        ),
        format!(
            ".reqntid {},{},1;",
            tile.workgroup[0], tile.workgroup[1]
        ),
        ".extern .shared .align 16 .b8 smem[];".into(),
        ".reg .pred %p<16>;".into(),
        ".reg .b32 %r<96>;".into(),
        ".reg .b64 %rd<96>;".into(),
        ".reg .f32 %f<16>;".into(),
        ".reg .f64 %fd<16>;".into(),
        "ld.param.u64 %rd40,[p0]; ld.param.u64 %rd41,[p1]; ld.param.u64 %rd42,[p2]; ld.param.u64 %rd0,[extent];".into(),
        "mov.u32 %r0,%ctaid.x; mov.u32 %r1,%ctaid.y; mov.u32 %r2,%ctaid.z; mov.u32 %r3,%tid.x; mov.u32 %r4,%tid.y;".into(),
        format!(
            "mad.lo.u32 %r5,%r0,{},%r3; mad.lo.u32 %r6,%r1,{},%r4;",
            tile.block_n, tile.block_m
        ),
        format!(
            "mad.lo.u32 %r7,%r4,{},%r3; mov.u32 %r8,{};",
            tile.block_n, tile.resources.threads_per_block
        ),
        "cvt.u64.u32 %rd3,%r2; mov.u64 %rd4,%rd3; mov.u64 %rd5,0; mov.u64 %rd6,0;".into(),
    ];
    lines.push(batch_projection(plan, &plan.lhs_shape, "%rd4", "%rd5"));
    lines.push("mov.u64 %rd4,%rd3;".into());
    lines.push(batch_projection(plan, &plan.rhs_shape, "%rd4", "%rd6"));
    lines.extend([
        "cvta.to.shared.u64 %rd30,smem;".into(),
        "mov.f64 %fd0,0d0000000000000000;".into(),
        "mov.u32 %r20,0;".into(),
        "K_TILE:".into(),
        format!("setp.ge.u32 %p0,%r20,{}; @%p0 bra STORE;", plan.k),
        "mov.u32 %r10,%r7;".into(),
        "LOAD_LHS:".into(),
        format!(
            "setp.ge.u32 %p1,%r10,{}; @%p1 bra LOAD_RHS_INIT;",
            tile.block_m * tile.block_k
        ),
        format!(
            "div.u32 %r11,%r10,{}; rem.u32 %r12,%r10,{}; mad.lo.u32 %r13,%r1,{},%r11; add.u32 %r14,%r20,%r12;",
            tile.block_k, tile.block_k, tile.block_m
        ),
        format!(
            "setp.lt.u32 %p2,%r13,{}; setp.lt.u32 %p3,%r14,{}; and.pred %p4,%p2,%p3; mov.f32 %f1,0f00000000;",
            plan.m, plan.k
        ),
        "cvt.u64.u32 %rd10,%r13; cvt.u64.u32 %rd11,%r14;".into(),
        format!(
            "mad.lo.u64 %rd12,%rd5,{},%rd10; mad.lo.u64 %rd12,%rd12,{},%rd11; mul.lo.u64 %rd12,%rd12,4; add.u64 %rd12,%rd40,%rd12; @%p4 ld.global.f32 %f1,[%rd12];",
            plan.m, plan.k
        ),
        "mul.wide.u32 %rd13,%r10,4; add.u64 %rd13,%rd30,%rd13; st.shared.f32 [%rd13],%f1; add.u32 %r10,%r10,%r8; bra LOAD_LHS;".into(),
        "LOAD_RHS_INIT: mov.u32 %r10,%r7;".into(),
        "LOAD_RHS:".into(),
        format!(
            "setp.ge.u32 %p1,%r10,{}; @%p1 bra LOADS_DONE;",
            tile.block_k * tile.block_n
        ),
        format!(
            "div.u32 %r11,%r10,{}; rem.u32 %r12,%r10,{}; add.u32 %r13,%r20,%r11; mad.lo.u32 %r14,%r0,{},%r12;",
            tile.block_n, tile.block_n, tile.block_n
        ),
        format!(
            "setp.lt.u32 %p2,%r13,{}; setp.lt.u32 %p3,%r14,{}; and.pred %p4,%p2,%p3; mov.f32 %f1,0f00000000;",
            plan.k, plan.n
        ),
        "cvt.u64.u32 %rd10,%r13; cvt.u64.u32 %rd11,%r14;".into(),
        format!(
            "mad.lo.u64 %rd12,%rd6,{},%rd10; mad.lo.u64 %rd12,%rd12,{},%rd11; mul.lo.u64 %rd12,%rd12,4; add.u64 %rd12,%rd41,%rd12; @%p4 ld.global.f32 %f1,[%rd12];",
            plan.k, plan.n
        ),
        format!(
            "mul.wide.u32 %rd13,%r10,4; add.u64 %rd13,%rd13,{}; add.u64 %rd13,%rd30,%rd13; st.shared.f32 [%rd13],%f1; add.u32 %r10,%r10,%r8; bra LOAD_RHS;",
            tile.lhs_shared.bytes
        ),
        "LOADS_DONE: bar.sync 0; mov.u32 %r30,0;".into(),
        "ACCUMULATE:".into(),
        format!(
            "setp.ge.u32 %p5,%r30,{}; @%p5 bra ACCUMULATE_DONE;",
            tile.block_k
        ),
        format!(
            "mad.lo.u32 %r31,%r4,{},%r30; mul.wide.u32 %rd20,%r31,4; add.u64 %rd20,%rd30,%rd20; ld.shared.f32 %f2,[%rd20];",
            tile.block_k
        ),
        format!(
            "mad.lo.u32 %r32,%r30,{},%r3; mul.wide.u32 %rd21,%r32,4; add.u64 %rd21,%rd21,{}; add.u64 %rd21,%rd30,%rd21; ld.shared.f32 %f3,[%rd21];",
            tile.block_n, tile.lhs_shared.bytes
        ),
        "cvt.f64.f32 %fd1,%f2; cvt.f64.f32 %fd2,%f3; mul.rn.f64 %fd3,%fd1,%fd2; add.rn.f64 %fd0,%fd0,%fd3; add.u32 %r30,%r30,1; bra ACCUMULATE;".into(),
        "ACCUMULATE_DONE: bar.sync 0;".into(),
        format!("add.u32 %r20,%r20,{}; bra K_TILE;", tile.block_k),
        "STORE:".into(),
        format!(
            "setp.lt.u32 %p6,%r6,{}; setp.lt.u32 %p7,%r5,{}; and.pred %p8,%p6,%p7; @!%p8 bra DONE;",
            plan.m, plan.n
        ),
        "cvt.u64.u32 %rd10,%r6; cvt.u64.u32 %rd11,%r5;".into(),
        format!(
            "mad.lo.u64 %rd12,%rd3,{},%rd10; mad.lo.u64 %rd12,%rd12,{},%rd11; mul.lo.u64 %rd12,%rd12,4; add.u64 %rd12,%rd42,%rd12; cvt.rn.f32.f64 %f0,%fd0; st.global.f32 [%rd12],%f0;",
            plan.m, plan.n
        ),
        "DONE: ret; }".into(),
    ]);
    let source_map = BTreeMap::from([
        (
            0,
            lines
                .iter()
                .position(|line| line.contains("LOADS_DONE"))
                .unwrap_or(0)
                + 1,
        ),
        (
            1,
            lines
                .iter()
                .position(|line| line.contains("ACCUMULATE_DONE"))
                .unwrap_or(0)
                + 1,
        ),
    ]);
    let source = lines.join("\n") + "\n";
    Ok(RenderedPtx {
        cache_key: stable_key(&(
            PTX_RENDERER_VERSION,
            renderer.sm,
            plan.cache_key,
            tile.cache_key,
            memory.cache_key,
            launch.grid,
            launch.block,
            launch.shared_bytes,
            &source,
        )),
        source,
        source_map,
        buffers,
        extent,
        entry,
        launch: PtxLaunchGeometry::Exact(launch),
        semantic_program: Some(KernelSemanticProgram::TiledMatmul(Arc::new(
            payload.clone(),
        ))),
    })
}

fn batch_projection(
    plan: &MatmulKernelPlan,
    shape: &crate::Shape,
    coordinate: &str,
    accumulator: &str,
) -> String {
    let batch = &shape.dims()[..shape.rank() - 2];
    let pad = plan.batch_shape.len() - batch.len();
    let mut source = format!("mov.u64 {accumulator},0;");
    for axis in (0..plan.batch_shape.len()).rev() {
        let dimension = plan.batch_shape[axis];
        source.push_str(&format!(
            " rem.u64 %rd20,{coordinate},{dimension}; div.u64 {coordinate},{coordinate},{dimension};"
        ));
        if axis >= pad && batch[axis - pad] != 1 {
            let stride = batch[axis - pad + 1..].iter().product::<usize>();
            source.push_str(&format!(
                " mad.lo.u64 {accumulator},%rd20,{stride},{accumulator};"
            ));
        }
    }
    source
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Shape};
    #[test]
    fn serial_and_tiled_sources_are_deterministic_and_narrow_storage_is_rejected() {
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
        let kernel = crate::lower_graph_matmul(&graph, out).unwrap();
        let tiled = PtxRenderer::new(80).unwrap().render(&kernel).unwrap();
        let crate::UArg::TiledMatmul(payload) = kernel.arg() else {
            panic!("eligible F32 matrix matmul was not tiled");
        };
        let direct_tiled = PtxRenderer::new(80)
            .unwrap()
            .render_tiled_matmul_plan(payload)
            .unwrap();
        assert_eq!(first.cache_key, second.cache_key);
        assert_eq!(tiled.cache_key, direct_tiled.cache_key);
        assert_eq!(tiled.source, direct_tiled.source);
        assert_eq!(
            first.buffers.iter().map(|b| b.id).collect::<Vec<_>>(),
            vec![lhs.index() as u64, rhs.index() as u64, out.index() as u64]
        );
        assert!(first.source.contains("LOOP:"));
        assert!(matches!(
            first.semantic_program,
            Some(KernelSemanticProgram::Matmul(_))
        ));
        assert!(tiled.source.contains(".extern .shared"));
        assert!(tiled.source.matches("bar.sync 0").count() >= 2);
        assert_eq!(tiled.source_map.keys().copied().collect::<Vec<_>>(), [0, 1]);
        assert!(matches!(
            tiled.semantic_program,
            Some(KernelSemanticProgram::TiledMatmul(_))
        ));
        assert!(matches!(tiled.launch, PtxLaunchGeometry::Exact(_)));
        assert!(matches!(
            PtxRenderer::new(75)
                .unwrap()
                .render_tiled_matmul_plan(payload),
            Err(PtxError::Unsupported(_))
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
