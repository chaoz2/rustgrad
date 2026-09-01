use super::{
    KernelSemanticProgram, PTX_ABI_VERSION, PTX_PORTABLE_SORT_RENDERER_VERSION, PtxBufferAbi,
    PtxError, PtxLaunchGeometry, PtxRenderer, RenderedPtx, stable_key,
};
use crate::{DType, SortValue, UOp};
use std::{collections::BTreeMap, sync::Arc};

pub(super) fn render(
    renderer: &PtxRenderer,
    root: &UOp,
    value: &SortValue,
) -> Result<RenderedPtx, PtxError> {
    let portable =
        crate::portable_sort::PortableSortPair::new(value).map_err(|error| match error {
            crate::portable_sort::PortableSortError::Unsupported(reason) => {
                PtxError::Unsupported(reason.into())
            }
            crate::portable_sort::PortableSortError::Overflow => PtxError::Overflow,
            other => PtxError::InvalidBinding(other.to_string()),
        })?;
    let elements = portable.elements();
    let buffers = vec![
        PtxBufferAbi {
            id: value.input.index() as u64,
            dtype: value.dtype,
            source_shape: value.input_shape.clone(),
            elements,
            mutable: false,
        },
        PtxBufferAbi {
            id: value.values.index() as u64,
            dtype: value.dtype,
            source_shape: value.input_shape.clone(),
            elements,
            mutable: true,
        },
        PtxBufferAbi {
            id: value.indices.index() as u64,
            dtype: DType::I32,
            source_shape: value.input_shape.clone(),
            elements,
            mutable: true,
        },
    ];
    let entry = format!(
        "rg_ptx_sort_{:?}_a{}_n{}",
        value.dtype, value.axis, elements
    )
    .to_ascii_lowercase();
    let value_bytes = portable.padded_len().max(1) * 4;
    let count_bytes = portable.axis_len().max(1) * 4;
    let mut lines = vec![
        format!("// {PTX_PORTABLE_SORT_RENDERER_VERSION} ABI {PTX_ABI_VERSION}"),
        ".version 7.0".into(),
        format!(".target sm_{}", renderer.sm),
        ".address_size 64".into(),
        String::new(),
        format!(".visible .entry {entry}("),
        "  .param .u64 p0,".into(),
        "  .param .u64 p1,".into(),
        "  .param .u64 p2,".into(),
        "  .param .u64 extent".into(),
        ")".into(),
        "{".into(),
        format!("  .local .align 4 .b8 rg_original[{value_bytes}];"),
        format!("  .local .align 4 .b8 rg_work[{value_bytes}];"),
        format!("  .local .align 4 .b8 rg_original_count[{count_bytes}];"),
        format!("  .local .align 4 .b8 rg_sorted_count[{count_bytes}];"),
        "  .reg .pred %p<16>;".into(),
        "  .reg .b32 %r<64>;".into(),
        "  .reg .b64 %rd<32>;".into(),
        "  .reg .f32 %f<8>;".into(),
        "  ld.param.u64 %rd0, [p0];".into(),
        "  ld.param.u64 %rd1, [p1];".into(),
        "  ld.param.u64 %rd2, [p2];".into(),
        "  ld.param.u64 %rd3, [extent];".into(),
        "  mov.u32 %r0, %ctaid.x;".into(),
        "  mov.u32 %r1, %ntid.x;".into(),
        "  mov.u32 %r2, %tid.x;".into(),
        "  mad.lo.u32 %r3, %r0, %r1, %r2;".into(),
        "  cvt.u64.u32 %rd4, %r3;".into(),
        "  setp.ge.u64 %p0, %rd4, %rd3;".into(),
        "  @%p0 bra DONE;".into(),
        format!("  div.u32 %r4, %r3, {};", portable.inner().max(1)),
        format!("  rem.u32 %r5, %r3, {};", portable.inner().max(1)),
        "  cvta.local.u64 %rd10, rg_original;".into(),
        "  cvta.local.u64 %rd11, rg_work;".into(),
        "  cvta.local.u64 %rd12, rg_original_count;".into(),
        "  cvta.local.u64 %rd13, rg_sorted_count;".into(),
    ];
    for lane in 0..portable.axis_len() {
        lines.extend(load_lane(&portable, value.dtype, lane));
    }
    for lane in portable.axis_len()..portable.padded_len() {
        lines.extend(pad_lane(value.dtype, value.descending, lane));
    }
    for &step in portable.steps() {
        lines.extend(match step {
            crate::portable_sort::PortableSortStep::Swap { left, right } => swap_step(left, right),
            crate::portable_sort::PortableSortStep::Compare(compare) => {
                compare_step(value.dtype, compare)
            }
        });
    }
    lines.extend(counts(value.dtype, portable.axis_len(), false));
    lines.extend(counts(value.dtype, portable.axis_len(), true));
    lines.extend(reconstruct(&portable, value.dtype));
    lines.extend(["DONE:".into(), "  ret;".into(), "}".into()]);
    let source = lines.join("\n") + "\n";
    Ok(RenderedPtx {
        cache_key: stable_key(&(
            PTX_PORTABLE_SORT_RENDERER_VERSION,
            renderer.sm,
            &source,
            &buffers,
            portable.value(),
        )),
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: portable.launch_extent(),
        entry,
        launch: PtxLaunchGeometry::Linear,
        semantic_program: Some(KernelSemanticProgram::UOp(Arc::new(root.clone()))),
    })
}

fn load_lane(
    plan: &crate::portable_sort::PortableSortPair<'_>,
    dtype: DType,
    lane: usize,
) -> Vec<String> {
    let global_width = dtype.itemsize();
    let local = lane * 4;
    let mut lines = vec![
        format!("  mad.lo.u32 %r6, %r4, {}, {};", plan.axis_len(), lane),
        format!("  mad.lo.u32 %r6, %r6, {}, %r5;", plan.inner()),
        format!("  mul.wide.u32 %rd20, %r6, {global_width};"),
        "  add.u64 %rd21, %rd0, %rd20;".into(),
    ];
    lines.push(match dtype {
        DType::Bool => "  ld.global.u8 %r20, [%rd21];".into(),
        DType::I32 => "  ld.global.s32 %r20, [%rd21];".into(),
        DType::U32 => "  ld.global.u32 %r20, [%rd21];".into(),
        DType::F32 => "  ld.global.b32 %r20, [%rd21];".into(),
        _ => unreachable!("portable sort validated storage"),
    });
    lines.extend([
        format!("  st.local.b32 [%rd10+{local}], %r20;"),
        format!("  st.local.b32 [%rd11+{local}], %r20;"),
    ]);
    lines
}

fn pad_lane(dtype: DType, descending: bool, lane: usize) -> Vec<String> {
    let bits: u32 = match (dtype, descending) {
        (DType::Bool, true) | (DType::U32, true) => 0,
        (DType::Bool, false) => 1,
        (DType::I32, true) => 0x8000_0000,
        (DType::I32, false) => 0x7fff_ffff,
        (DType::U32, false) => 0xffff_ffff,
        (DType::F32, true) => 0xff80_0000,
        (DType::F32, false) => 0x7f80_0000,
        _ => unreachable!("portable sort validated storage"),
    };
    vec![
        format!("  mov.u32 %r20, 0x{bits:08x};"),
        format!("  st.local.b32 [%rd11+{}], %r20;", lane * 4),
    ]
}

fn compare_step(dtype: DType, step: crate::portable_sort::PortableSortCompare) -> Vec<String> {
    let left = step.left * 4;
    let right = step.right * 4;
    let mut lines = vec![
        format!("  ld.local.b32 %r20, [%rd11+{left}];"),
        format!("  ld.local.b32 %r21, [%rd11+{right}];"),
    ];
    if dtype == DType::F32 {
        lines.extend([
            "  mov.b32 %f0, %r20;".into(),
            "  mov.b32 %f1, %r21;".into(),
            "  setp.gt.f32 %p1, %f1, %f0;".into(),
            "  selp.b32 %r22, %r21, %r20, %p1;".into(),
            "  setp.lt.f32 %p2, %f1, %f0;".into(),
            "  selp.b32 %r23, %r21, %r20, %p2;".into(),
        ]);
    } else if dtype == DType::Bool {
        lines.extend([
            "  or.b32 %r22, %r20, %r21;".into(),
            "  and.b32 %r23, %r20, %r21;".into(),
        ]);
    } else {
        let suffix = if dtype == DType::I32 { "s32" } else { "u32" };
        lines.extend([
            format!("  setp.gt.{suffix} %p1, %r21, %r20;"),
            "  selp.b32 %r22, %r21, %r20, %p1;".into(),
            format!("  setp.lt.{suffix} %p2, %r21, %r20;"),
            "  selp.b32 %r23, %r21, %r20, %p2;".into(),
        ]);
    }
    let (left_value, right_value) = if step.left_takes_larger {
        ("%r22", "%r23")
    } else {
        ("%r23", "%r22")
    };
    lines.extend([
        format!("  st.local.b32 [%rd11+{left}], {left_value};"),
        format!("  st.local.b32 [%rd11+{right}], {right_value};"),
    ]);
    lines
}

fn swap_step(left: usize, right: usize) -> Vec<String> {
    let left = left * 4;
    let right = right * 4;
    vec![
        format!("  ld.local.b32 %r20, [%rd11+{left}];"),
        format!("  ld.local.b32 %r21, [%rd11+{right}];"),
        format!("  st.local.b32 [%rd11+{left}], %r21;"),
        format!("  st.local.b32 [%rd11+{right}], %r20;"),
    ]
}

fn counts(dtype: DType, axis_len: usize, sorted: bool) -> Vec<String> {
    let (values, counts, prefix) = if sorted {
        ("%rd11", "%rd13", "SORTED_COUNT")
    } else {
        ("%rd10", "%rd12", "ORIGINAL_COUNT")
    };
    let mut lines = vec![
        "  mov.u32 %r30, 0;".into(),
        format!("{prefix}_OUTER:"),
        format!("  setp.ge.u32 %p3, %r30, {axis_len};"),
        format!("  @%p3 bra {prefix}_DONE;"),
        "  mov.u32 %r31, 0;".into(),
        "  mov.u32 %r32, 0;".into(),
        "  mul.wide.u32 %rd20, %r30, 4;".into(),
        format!("  add.u64 %rd21, {values}, %rd20;"),
        "  ld.local.b32 %r33, [%rd21];".into(),
        format!("{prefix}_INNER:"),
        "  setp.gt.u32 %p4, %r31, %r30;".into(),
        format!("  @%p4 bra {prefix}_STORE;"),
        "  mul.wide.u32 %rd22, %r31, 4;".into(),
        format!("  add.u64 %rd23, {values}, %rd22;"),
        "  ld.local.b32 %r34, [%rd23];".into(),
    ];
    lines.extend(equal(dtype, "%r34", "%r33", "%p5"));
    lines.extend([
        "  selp.u32 %r35, 1, 0, %p5;".into(),
        "  add.u32 %r32, %r32, %r35;".into(),
        "  add.u32 %r31, %r31, 1;".into(),
        format!("  bra {prefix}_INNER;"),
        format!("{prefix}_STORE:"),
        format!("  add.u64 %rd21, {counts}, %rd20;"),
        "  st.local.u32 [%rd21], %r32;".into(),
        "  add.u32 %r30, %r30, 1;".into(),
        format!("  bra {prefix}_OUTER;"),
        format!("{prefix}_DONE:"),
    ]);
    lines
}

fn reconstruct(plan: &crate::portable_sort::PortableSortPair<'_>, dtype: DType) -> Vec<String> {
    let axis = plan.axis_len();
    let inner = plan.inner();
    let width = dtype.itemsize();
    let mut lines = vec![
        "  mov.u32 %r30, 0;".into(),
        "RECONSTRUCT_OUTER:".into(),
        format!("  setp.ge.u32 %p3, %r30, {axis};"),
        "  @%p3 bra RECONSTRUCT_DONE;".into(),
        "  mov.u32 %r31, 0;".into(),
        "  mov.u32 %r32, 0;".into(),
        "  mul.wide.u32 %rd20, %r30, 4;".into(),
        "  add.u64 %rd21, %rd11, %rd20;".into(),
        "  ld.local.b32 %r33, [%rd21];".into(),
        "  add.u64 %rd21, %rd13, %rd20;".into(),
        "  ld.local.u32 %r36, [%rd21];".into(),
        "RECONSTRUCT_INNER:".into(),
        format!("  setp.ge.u32 %p4, %r31, {axis};"),
        "  @%p4 bra RECONSTRUCT_STORE;".into(),
        "  mul.wide.u32 %rd22, %r31, 4;".into(),
        "  add.u64 %rd23, %rd10, %rd22;".into(),
        "  ld.local.b32 %r34, [%rd23];".into(),
    ];
    lines.extend(equal(dtype, "%r34", "%r33", "%p5"));
    lines.extend([
        "  add.u64 %rd23, %rd12, %rd22;".into(),
        "  ld.local.u32 %r37, [%rd23];".into(),
        "  setp.eq.u32 %p6, %r37, %r36;".into(),
        "  and.pred %p7, %p5, %p6;".into(),
        "  selp.u32 %r38, %r31, 0, %p7;".into(),
        "  add.u32 %r32, %r32, %r38;".into(),
        "  add.u32 %r31, %r31, 1;".into(),
        "  bra RECONSTRUCT_INNER;".into(),
        "RECONSTRUCT_STORE:".into(),
        format!("  mad.lo.u32 %r39, %r4, {axis}, %r30;"),
        format!("  mad.lo.u32 %r39, %r39, {inner}, %r5;"),
        format!("  mul.wide.u32 %rd24, %r39, {width};"),
        "  add.u64 %rd25, %rd1, %rd24;".into(),
    ]);
    lines.push(match dtype {
        DType::Bool => "  st.global.u8 [%rd25], %r33;".into(),
        DType::I32 => "  st.global.s32 [%rd25], %r33;".into(),
        DType::U32 => "  st.global.u32 [%rd25], %r33;".into(),
        DType::F32 => "  st.global.b32 [%rd25], %r33;".into(),
        _ => unreachable!("portable sort validated storage"),
    });
    lines.extend([
        "  mul.wide.u32 %rd26, %r39, 4;".into(),
        "  add.u64 %rd27, %rd2, %rd26;".into(),
        "  st.global.s32 [%rd27], %r32;".into(),
        "  add.u32 %r30, %r30, 1;".into(),
        "  bra RECONSTRUCT_OUTER;".into(),
        "RECONSTRUCT_DONE:".into(),
    ]);
    lines
}

fn equal(dtype: DType, lhs: &str, rhs: &str, predicate: &str) -> Vec<String> {
    if dtype == DType::F32 {
        vec![
            format!("  mov.b32 %f0, {lhs};"),
            format!("  mov.b32 %f1, {rhs};"),
            format!("  setp.eq.f32 {predicate}, %f0, %f1;"),
        ]
    } else {
        vec![format!("  setp.eq.b32 {predicate}, {lhs}, {rhs};")]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CudaGraphPrefixPlan, Graph, Shape, schedule_many};

    #[test]
    fn ptx_sort_pair_has_ordered_outputs_and_static_cuda_graph_launch_extent() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3], DType::F32);
        let (values, indices) = graph.sort(input, 1, false).unwrap();
        let schedule = schedule_many(&graph, &[values, indices]).unwrap();
        let renderer = PtxRenderer::new(80).unwrap();
        let rendered = renderer.render(&schedule.items[0].kernel).unwrap();
        rendered
            .validate_schedule_bindings(schedule.items[0].ordered_inputs())
            .unwrap();
        assert_eq!((rendered.extent, rendered.buffers.len()), (2, 3));
        assert_eq!(
            rendered
                .buffers
                .iter()
                .filter(|buffer| buffer.mutable)
                .map(|buffer| buffer.id)
                .collect::<Vec<_>>(),
            vec![values.index() as u64, indices.index() as u64]
        );
        assert!(rendered.source.contains(PTX_PORTABLE_SORT_RENDERER_VERSION));
        assert!(rendered.source.contains("ORIGINAL_COUNT_OUTER"));
        assert_eq!(
            CudaGraphPrefixPlan::plan(&schedule.items, renderer)
                .unwrap()
                .kernel_cache_keys(),
            vec![rendered.cache_key]
        );
    }

    #[test]
    fn ptx_sort_zero_domain_is_resource_free_and_f64_stays_closed() {
        let mut empty = Graph::new();
        let input = empty.input_dtype("x", [2, 0, 3], DType::U32);
        let (values, indices) = empty.sort(input, 1, false).unwrap();
        let schedule = schedule_many(&empty, &[values, indices]).unwrap();
        let plan =
            CudaGraphPrefixPlan::plan(&schedule.items, PtxRenderer::new(80).unwrap()).unwrap();
        assert!(plan.kernel_cache_keys().is_empty());

        let mut unsupported = Graph::new();
        let input = unsupported.input_dtype("x", Shape::new([3]), DType::F64);
        let (values, indices) = unsupported.sort(input, 0, false).unwrap();
        let schedule = schedule_many(&unsupported, &[values, indices]).unwrap();
        assert!(matches!(
            PtxRenderer::new(80).unwrap().render(&schedule.items[0].kernel),
            Err(PtxError::Unsupported(reason)) if reason.contains("Bool/I32/U32/F32")
        ));
    }
}
