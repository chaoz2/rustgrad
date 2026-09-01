use super::{
    KernelSemanticProgram, PTX_ABI_VERSION, PTX_RENDERER_VERSION, PtxBufferAbi, PtxError,
    PtxLaunchGeometry, PtxRenderer, RenderedPtx, stable_key,
};
use crate::{DType, PrefixScanKind, PrefixScanOutput, PrefixScanValue, UOp};
use std::{collections::BTreeMap, sync::Arc};

pub(super) fn render(
    renderer: &PtxRenderer,
    root: &UOp,
    value: &PrefixScanValue,
) -> Result<RenderedPtx, PtxError> {
    let plan = crate::prefix_scan_native::NativePrefixScanPlan::new(value)
        .map_err(|reason| PtxError::Unsupported(reason.into()))?;
    if !matches!(
        plan.input_dtype,
        DType::Bool | DType::I32 | DType::U32 | DType::F32
    ) || !matches!(
        plan.output_dtype,
        DType::Bool | DType::I32 | DType::U32 | DType::F32
    ) || [
        plan.elements,
        plan.rows,
        plan.inner,
        plan.axis_len,
        plan.work_items(),
    ]
    .into_iter()
    .any(|extent| extent > u32::MAX as usize)
    {
        return Err(PtxError::Unsupported(
            "PTX prefix scan requires a 32-bit Bool/I32/U32/F32 domain".into(),
        ));
    }
    let buffers = vec![
        PtxBufferAbi {
            id: plan.input,
            dtype: plan.input_dtype,
            source_shape: value.input_shape.clone(),
            elements: plan.elements,
            mutable: false,
        },
        PtxBufferAbi {
            id: plan.output,
            dtype: plan.output_dtype,
            source_shape: value.output_shape.clone(),
            elements: plan.elements,
            mutable: true,
        },
    ];
    let entry = format!(
        "rg_prefix_{:?}_{:?}_a{}_n{}",
        plan.kind, plan.result, plan.axis, plan.elements
    )
    .to_ascii_lowercase();
    let mut lines = vec![
        format!("// {PTX_RENDERER_VERSION} prefix-scan ABI {PTX_ABI_VERSION}"),
        ".version 7.0".into(),
        format!(".target sm_{}", renderer.sm),
        ".address_size 64".into(),
        String::new(),
        format!(".visible .entry {entry}("),
        "  .param .u64 p0,".into(),
        "  .param .u64 p1,".into(),
        "  .param .u64 extent".into(),
        ")".into(),
        "{".into(),
        "  .reg .pred %p<8>;".into(),
        "  .reg .b32 %r<32>;".into(),
        "  .reg .b64 %rd<16>;".into(),
        "  .reg .f32 %f<8>;".into(),
        "  ld.param.u64 %rd0, [p0];".into(),
        "  ld.param.u64 %rd1, [p1];".into(),
        "  ld.param.u64 %rd2, [extent];".into(),
        "  mov.u32 %r0, %ctaid.x;".into(),
        "  mov.u32 %r1, %ntid.x;".into(),
        "  mov.u32 %r2, %tid.x;".into(),
        "  mad.lo.u32 %r3, %r0, %r1, %r2;".into(),
        "  cvt.u64.u32 %rd3, %r3;".into(),
        "  setp.ge.u64 %p0, %rd3, %rd2;".into(),
        "  @%p0 bra DONE;".into(),
    ];
    if plan.scalar_identity {
        lines.extend(scalar_identity(&plan)?);
    } else {
        lines.extend([
            format!("  div.u32 %r4, %r3, {}u;", plan.inner.max(1)),
            format!("  rem.u32 %r5, %r3, {}u;", plan.inner.max(1)),
            "  mov.u32 %r6, 0;".into(),
            format!("  mov.u32 %r7, {};", plan.index_sentinel),
            identity(plan.kind, plan.work_dtype)?,
            "SCAN_LOOP:".into(),
            format!("  setp.ge.u32 %p1, %r6, {}u;", plan.axis_len),
            "  @%p1 bra DONE_SCAN;".into(),
            format!("  mad.lo.u32 %r9, %r4, {}u, %r6;", plan.axis_len),
            format!("  mad.lo.u32 %r9, %r9, {}u, %r5;", plan.inner),
            format!("  mul.wide.u32 %rd4, %r9, {};", plan.input_dtype.itemsize()),
            "  add.u64 %rd5, %rd0, %rd4;".into(),
            load(plan.input_dtype),
        ]);
        lines.push(update(plan.kind, plan.work_dtype, plan.input_dtype)?);
        if plan.result == PrefixScanOutput::Indices {
            lines.push(compare("eq", plan.work_dtype, "%p2"));
            lines.push(format!("  setp.eq.u32 %p6, %r7, {};", plan.index_sentinel));
            lines.push("  and.pred %p7, %p2, %p6;".into());
            lines.push("  or.pred %p7, %p7, %p3;".into());
            lines.push("  selp.u32 %r7, %r6, %r7, %p7;".into());
        }
        lines.extend([
            format!(
                "  mul.wide.u32 %rd6, %r9, {};",
                plan.output_dtype.itemsize()
            ),
            "  add.u64 %rd7, %rd1, %rd6;".into(),
            store(plan.result, plan.output_dtype),
            "  add.u32 %r6, %r6, 1;".into(),
            "  bra SCAN_LOOP;".into(),
            "DONE_SCAN:".into(),
        ]);
    }
    lines.extend(["DONE:".into(), "  ret;".into(), "}".into()]);
    let source = lines.join("\n") + "\n";
    Ok(RenderedPtx {
        cache_key: stable_key(&(PTX_RENDERER_VERSION, renderer.sm, &source, &buffers, &plan)),
        source,
        source_map: BTreeMap::new(),
        buffers,
        extent: plan.work_items(),
        entry,
        launch: PtxLaunchGeometry::Linear,
        semantic_program: Some(KernelSemanticProgram::UOp(Arc::new(root.clone()))),
    })
}

fn scalar_identity(
    plan: &crate::prefix_scan_native::NativePrefixScanPlan,
) -> Result<Vec<String>, PtxError> {
    if plan.result == PrefixScanOutput::Indices {
        return Ok(vec![
            "  mov.u32 %r7, 0;".into(),
            "  st.global.s32 [%rd1], %r7;".into(),
        ]);
    }
    Ok(match (plan.input_dtype, plan.output_dtype) {
        (DType::F32, DType::F32) => vec![
            "  ld.global.b32 %r10, [%rd0];".into(),
            "  st.global.b32 [%rd1], %r10;".into(),
        ],
        (DType::Bool, DType::Bool) => vec![
            "  ld.global.u8 %r10, [%rd0];".into(),
            "  st.global.u8 [%rd1], %r10;".into(),
        ],
        (DType::Bool, DType::I32) => vec![
            "  ld.global.u8 %r10, [%rd0];".into(),
            "  st.global.s32 [%rd1], %r10;".into(),
        ],
        (DType::I32, DType::I32) => vec![
            "  ld.global.b32 %r10, [%rd0];".into(),
            "  st.global.b32 [%rd1], %r10;".into(),
        ],
        (DType::U32, DType::U32) => vec![
            "  ld.global.b32 %r10, [%rd0];".into(),
            "  st.global.b32 [%rd1], %r10;".into(),
        ],
        _ => {
            return Err(PtxError::Unsupported(
                "PTX scalar prefix identity dtype".into(),
            ));
        }
    })
}

fn identity(kind: PrefixScanKind, dtype: DType) -> Result<String, PtxError> {
    Ok(match (kind, dtype) {
        (PrefixScanKind::Sum, DType::F32) => "  mov.f32 %f0, 0f00000000;".into(),
        (PrefixScanKind::Product, DType::F32) => "  mov.f32 %f0, 0f3f800000;".into(),
        (PrefixScanKind::Max, DType::F32) => "  mov.b32 %f0, 0xff800000;".into(),
        (PrefixScanKind::Min, DType::F32) => "  mov.b32 %f0, 0x7f800000;".into(),
        (PrefixScanKind::Product | PrefixScanKind::Min, DType::Bool) => "  mov.u32 %r8, 1;".into(),
        (PrefixScanKind::Max, DType::I32) => "  mov.u32 %r8, 0x80000000;".into(),
        (PrefixScanKind::Min, DType::I32) => "  mov.u32 %r8, 0x7fffffff;".into(),
        (PrefixScanKind::Min, DType::U32) => "  mov.u32 %r8, 0xffffffff;".into(),
        (_, DType::Bool | DType::I32 | DType::U32) => "  mov.u32 %r8, 0;".into(),
        _ => return Err(PtxError::Unsupported("PTX prefix identity dtype".into())),
    })
}

fn load(dtype: DType) -> String {
    match dtype {
        DType::F32 => "  ld.global.f32 %f1, [%rd5];".into(),
        DType::Bool => "  ld.global.u8 %r10, [%rd5];".into(),
        DType::I32 => "  ld.global.s32 %r10, [%rd5];".into(),
        DType::U32 => "  ld.global.u32 %r10, [%rd5];".into(),
        _ => unreachable!("validated PTX scan dtype"),
    }
}

fn compare(operator: &str, dtype: DType, predicate: &str) -> String {
    match dtype {
        DType::F32 => format!("  setp.{operator}.f32 {predicate}, %f1, %f0;"),
        DType::I32 => format!("  setp.{operator}.s32 {predicate}, %r10, %r8;"),
        DType::Bool | DType::U32 => {
            format!("  setp.{operator}.u32 {predicate}, %r10, %r8;")
        }
        _ => unreachable!("validated PTX scan work dtype"),
    }
}

fn update(kind: PrefixScanKind, work: DType, input: DType) -> Result<String, PtxError> {
    Ok(match (kind, work, input) {
        (PrefixScanKind::Sum, DType::F32, DType::F32) => "  add.rn.f32 %f0, %f0, %f1;".into(),
        (PrefixScanKind::Product, DType::F32, DType::F32) => "  mul.rn.f32 %f0, %f0, %f1;".into(),
        (PrefixScanKind::Sum, DType::I32, DType::Bool) => "  add.u32 %r8, %r8, %r10;".into(),
        (PrefixScanKind::Sum, DType::I32 | DType::U32, _) => "  add.u32 %r8, %r8, %r10;".into(),
        (PrefixScanKind::Product, DType::Bool, DType::Bool) => "  and.b32 %r8, %r8, %r10;".into(),
        (PrefixScanKind::Product, DType::I32 | DType::U32, _) => {
            "  mul.lo.u32 %r8, %r8, %r10;".into()
        }
        (PrefixScanKind::Max | PrefixScanKind::Min, dtype, _) => {
            let operator = if kind == PrefixScanKind::Max {
                "gt"
            } else {
                "lt"
            };
            let set = compare(operator, dtype, "%p3");
            let select = if dtype == DType::F32 {
                "selp.f32 %f0, %f1, %f0, %p3;"
            } else {
                "selp.u32 %r8, %r10, %r8, %p3;"
            };
            format!("{set} {select}")
        }
        _ => return Err(PtxError::Unsupported("PTX prefix update dtype".into())),
    })
}

fn store(result: PrefixScanOutput, dtype: DType) -> String {
    if result == PrefixScanOutput::Indices {
        return "  st.global.s32 [%rd7], %r7;".into();
    }
    match dtype {
        DType::F32 => "  st.global.f32 [%rd7], %f0;".into(),
        DType::Bool => "  st.global.u8 [%rd7], %r8;".into(),
        DType::I32 => "  st.global.s32 [%rd7], %r8;".into(),
        DType::U32 => "  st.global.u32 [%rd7], %r8;".into(),
        _ => unreachable!("validated PTX scan output"),
    }
}
