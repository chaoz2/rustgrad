//! Binding-independent C11 rendering for authenticated bounded-symbolic items.
//!
//! The ordinary renderer remains the semantic admission oracle. This module
//! changes only iteration and address geometry: every scalar operation is
//! emitted by the shared scalar UOp emitter, while shape expressions arrive
//! through the already-versioned symbolic capture schema.

use super::{
    ABI_VERSION, BufferAbi, JitError, KernelAbi, KernelPointerAbi, RenderedC, SymbolicLoadOffsets,
    ctype, emit_with_substitution, native_cache_key, reduction_accumulator_type,
    reduction_arithmetic_expr, runtime_mean_divisor_expr, scalar_kernel_prologue,
    scalar_store_expr, scan_commit_expr, scan_store_expr,
};
use crate::engine::symbolic::{SymbolicItemDomain, SymbolicSchema};
use crate::engine::symbolic_uop::{AuthenticatedSymbolicUOp, lower_symbolic_expression};
use crate::engine::symbolic_view::SymbolicViewMap;
use crate::projected_index::ProjectedIndexEmitter;
use crate::uop::Binary;
use crate::{
    DType, IndexAddressing, IndexValue, MatmulValue, MovementValue, Operation, ScheduleItem,
    SymbolicExpr, SymbolicShape, UOp,
};
use std::collections::BTreeMap;

pub(crate) fn render(
    capture_identity: u64,
    item: &ScheduleItem,
    schema: &SymbolicSchema,
) -> Result<RenderedC, JitError> {
    if !item.outputs.is_single() || item.boundary.is_some() || item.is_effect() {
        return Err(JitError::Unsupported(
            "runtime-symbolic programs require one pure output per item".into(),
        ));
    }
    // Preserve the complete existing operation/dtype admission policy. The
    // template is concrete structural evidence and is never executed here.
    let admitted = super::render_with_policy(&item.kernel, false)?;
    if !admitted.abi.quantized_buffers.is_empty() {
        return Err(JitError::Unsupported(
            "runtime-symbolic packed resources are unsupported".into(),
        ));
    }
    let domain = schema
        .item_domains
        .get(&item.id)
        .ok_or_else(|| JitError::Symbolic("symbolic item domain is absent".into()))?;
    match item.kernel.operation() {
        Operation::Movement(MovementValue::Plan(plan)) => {
            render_raw_copy(capture_identity, item, schema, &admitted, plan)
        }
        Operation::Matmul(MatmulValue::Serial(plan)) => {
            render_matmul(capture_identity, item, schema, &admitted, plan, domain)
        }
        Operation::Matmul(MatmulValue::Tiled(payload)) => render_matmul(
            capture_identity,
            item,
            schema,
            &admitted,
            &payload.matmul,
            domain,
        ),
        Operation::Matmul(MatmulValue::TensorCore(payload)) => render_matmul(
            capture_identity,
            item,
            schema,
            &admitted,
            &payload.matmul,
            domain,
        ),
        Operation::Matmul(_) | Operation::Movement(_) => Err(JitError::Unsupported(
            "runtime-symbolic item family is outside the pure dense subset".into(),
        )),
        _ => render_uop(capture_identity, item, schema, &admitted, domain),
    }
}

fn runtime_abi(admitted: &RenderedC, schema: &SymbolicSchema) -> Result<KernelAbi, JitError> {
    let buffers = admitted
        .abi
        .buffers
        .iter()
        .map(|buffer| {
            let shape = schema.buffer_shapes.get(&buffer.id).ok_or_else(|| {
                JitError::Symbolic(format!("symbolic buffer {} is absent", buffer.id))
            })?;
            Ok(BufferAbi {
                id: buffer.id,
                dtype: buffer.dtype,
                elements: maximum_elements(shape)?,
                mutable: buffer.mutable,
            })
        })
        .collect::<Result<Vec<_>, JitError>>()?;
    Ok(KernelAbi {
        version: ABI_VERSION,
        pointer_order: (0..buffers.len()).map(KernelPointerAbi::Dense).collect(),
        buffers,
        quantized_buffers: Vec::new(),
        symbol_count: schema.parameters.len(),
    })
}

fn render_uop(
    capture_identity: u64,
    item: &ScheduleItem,
    schema: &SymbolicSchema,
    admitted: &RenderedC,
    domain: &SymbolicItemDomain,
) -> Result<RenderedC, JitError> {
    let abi = runtime_abi(admitted, schema)?;
    let ids = abi
        .buffers
        .iter()
        .enumerate()
        .map(|(slot, buffer)| (buffer.id, slot))
        .collect::<BTreeMap<_, _>>();
    let store = item
        .kernel
        .sources()
        .iter()
        .find(|node| matches!(node.operation(), Operation::Store))
        .ok_or_else(|| JitError::Unsupported("symbolic Sink lacks Store".into()))?;
    let out = item.primary_output().id;
    let output = match domain {
        SymbolicItemDomain::Elementwise { output }
        | SymbolicItemDomain::Reduction { output, .. } => output,
        SymbolicItemDomain::Matmul { .. } => {
            return Err(JitError::Unsupported(
                "symbolic Matmul reached elementwise renderer".into(),
            ));
        }
    };
    let mut lines = prologue(capture_identity, item.id);
    match domain {
        SymbolicItemDomain::Elementwise { .. } => {
            let projected_ordinals = projected_ordinals(&item.kernel)?;
            let extent = shape_elements_c(output, schema)?;
            lines.push(format!(
                "  for (size_t rg_i=0; rg_i<(size_t)({extent}); ++rg_i) {{"
            ));
            let offsets = Offsets {
                item: item.id,
                projected_ordinals: &projected_ordinals,
                schema,
                domain: output,
            };
            let mut map = BTreeMap::new();
            let value = emit_with_substitution(
                store
                    .sources()
                    .get(1)
                    .ok_or_else(|| JitError::Unsupported("symbolic Store lacks value".into()))?,
                &ids,
                &mut map,
                &mut lines,
                None,
                Some(&offsets),
            )?;
            let dtype = item.primary_output().dtype;
            lines.push(format!(
                "    (({}*)buffers[{}])[rg_i]={};",
                ctype(dtype),
                ids[&out],
                scalar_store_expr(dtype, &value)
            ));
            lines.push("  }".into());
        }
        SymbolicItemDomain::Reduction {
            input,
            output,
            reduction,
            ..
        } => render_reduction(
            item,
            schema,
            store,
            &ids,
            ReductionDomain {
                input,
                output,
                reduction,
            },
            &mut lines,
        )?,
        SymbolicItemDomain::Matmul { .. } => unreachable!(),
    }
    lines.push("  return failure[1] ? (int)failure[1] : 0;".into());
    lines.push("}".into());
    finish(capture_identity, item, abi, lines)
}

#[derive(Clone, Copy)]
struct ReductionDomain<'a> {
    input: &'a SymbolicShape,
    output: &'a SymbolicShape,
    reduction: &'a SymbolicShape,
}

fn render_reduction(
    item: &ScheduleItem,
    schema: &SymbolicSchema,
    store: &UOp,
    ids: &BTreeMap<u64, usize>,
    domain: ReductionDomain<'_>,
    lines: &mut Vec<String>,
) -> Result<(), JitError> {
    let kernel = crate::reduction_native::NativeReductionKernel::from_store(store)
        .map_err(|reason| JitError::Unsupported(reason.into()))?
        .ok_or_else(|| JitError::Unsupported("symbolic reduction plan is absent".into()))?;
    if !matches!(
        kernel.plan.kind,
        crate::ReduceKind::Sum
            | crate::ReduceKind::Mean
            | crate::ReduceKind::Product
            | crate::ReduceKind::Max
            | crate::ReduceKind::Min
    ) {
        return Err(JitError::Unsupported(
            "runtime-symbolic reduction kind is unsupported".into(),
        ));
    }
    let projected_ordinals = projected_ordinals(&item.kernel)?;
    if kernel.plan.output_dtype.is_float8() && kernel.has_epilogue() {
        return Err(JitError::Unsupported(
            "runtime-symbolic Float8 reduction epilogues are unsupported".into(),
        ));
    }
    let out_len = shape_elements_c(domain.output, schema)?;
    let reduce_len = shape_elements_c(domain.reduction, schema)?;
    lines.push(format!("  size_t rg_reduce_len=(size_t)({reduce_len});"));
    lines.push(format!(
        "  for(size_t rg_out=0; rg_out<(size_t)({out_len}); ++rg_out) {{"
    ));
    if kernel.plan.source_dtype == kernel.plan.output_dtype
        && !kernel.has_epilogue()
        && let Some((buffer, index)) = direct_load_index(kernel.producer)
    {
        let offsets = Offsets {
            item: item.id,
            projected_ordinals: &projected_ordinals,
            schema,
            domain: domain.input,
        };
        let input_offset = offsets.offset(index, buffer)?;
        let input_index = reduction_index_c(
            domain.input,
            domain.output,
            &kernel.plan.geometry.axes,
            kernel.plan.geometry.keepdim,
            "0u",
            schema,
        )?;
        let storage = ctype(kernel.plan.output_dtype);
        lines.push("    if(rg_reduce_len==1u) {".into());
        lines.push(format!("      size_t rg_i={input_index};"));
        lines.push(format!(
            "      (({storage}*)buffers[{}])[rg_out]=((const {storage}*)buffers[{}])[{input_offset}];",
            ids[&item.primary_output().id], ids[&buffer]
        ));
        lines.push("      continue;".into());
        lines.push("    }".into());
    }
    let accumulator = reduction_accumulator_type(kernel.plan.accumulator_dtype);
    let identity = super::native_scalar_literal(kernel.plan.identity())?;
    lines.push(format!("    {accumulator} rg_acc={identity};"));
    lines.push("    for(size_t rg_r=0; rg_r<rg_reduce_len; ++rg_r) {".into());
    lines.push(format!(
        "      size_t rg_i={};",
        reduction_index_c(
            domain.input,
            domain.output,
            &kernel.plan.geometry.axes,
            kernel.plan.geometry.keepdim,
            "rg_r",
            schema,
        )?
    ));
    let producer_offsets = Offsets {
        item: item.id,
        projected_ordinals: &projected_ordinals,
        schema,
        domain: domain.input,
    };
    let mut map = BTreeMap::new();
    let value = emit_with_substitution(
        kernel.producer,
        ids,
        &mut map,
        lines,
        None,
        Some(&producer_offsets),
    )?;
    let value = if kernel.plan.source_dtype.is_float8() {
        super::float8_decode_expr(kernel.plan.source_dtype, &value)
            .expect("guarded Float8 reduction source")
    } else {
        super::cast_expression(
            kernel.plan.source_dtype,
            kernel.plan.accumulator_dtype,
            value,
        )
    };
    lines.push(format!(
        "      if(rg_reduce_len==1u) rg_acc=({accumulator})({value}); else {{"
    ));
    if matches!(
        kernel.plan.kind,
        crate::ReduceKind::Max | crate::ReduceKind::Min
    ) {
        let comparison = if kernel.plan.kind == crate::ReduceKind::Max {
            ">"
        } else {
            "<"
        };
        lines.push(format!(
            "        if(({value}) {comparison} rg_acc) rg_acc=({value});"
        ));
    } else if kernel.plan.accumulator_dtype == DType::Bool {
        let operator = if kernel.plan.kind == crate::ReduceKind::Product {
            "&&"
        } else {
            "||"
        };
        lines.push(format!(
            "        rg_acc=(uint8_t)(rg_acc {operator} ({value}));"
        ));
    } else {
        lines.push(format!(
            "        rg_acc={};",
            reduction_arithmetic_expr(
                kernel.plan.accumulator_dtype,
                "rg_acc",
                &value,
                kernel.plan.kind == crate::ReduceKind::Product,
            )?
        ));
    }
    lines.push("      }".into());
    lines.push("    }".into());
    let finalized = if kernel.plan.kind == crate::ReduceKind::Mean {
        // Mean commits both operands at the accumulator width. This is the
        // runtime-cardinality equivalent of NativeReductionPlan::mean_divisor
        // and preserves the authenticated legacy same-storage Float8 tuple.
        let divisor =
            runtime_mean_divisor_expr(kernel.plan.accumulator_dtype, "((double)rg_reduce_len)");
        format!(
            "(rg_reduce_len==0u ? NAN : {})",
            scan_commit_expr(
                kernel.plan.accumulator_dtype,
                &format!("(rg_acc/({accumulator})({divisor}))"),
            )
        )
    } else {
        "rg_acc".into()
    };
    let committed = scan_commit_expr(kernel.plan.output_dtype, &finalized);
    let store_value = if kernel.has_epilogue() {
        lines.push("    size_t rg_i=rg_out;".into());
        let epilogue_offsets = Offsets {
            item: item.id,
            projected_ordinals: &projected_ordinals,
            schema,
            domain: domain.output,
        };
        let mut map = BTreeMap::new();
        emit_with_substitution(
            kernel.epilogue_root,
            ids,
            &mut map,
            lines,
            Some((kernel.finalize, committed.as_str())),
            Some(&epilogue_offsets),
        )?
    } else {
        committed
    };
    lines.push(format!(
        "    (({}*)buffers[{}])[rg_out]={};",
        ctype(item.primary_output().dtype),
        ids[&item.primary_output().id],
        scan_store_expr(kernel.output_dtype, &store_value)
    ));
    lines.push("  }".into());
    Ok(())
}

fn direct_load_index(node: &UOp) -> Option<(u64, &UOp)> {
    let Operation::Load = node.operation() else {
        return None;
    };
    let index = node.sources().first()?;
    match index.operation() {
        Operation::Index(crate::IndexValue::Buffer { buffer, .. })
        | Operation::Index(crate::IndexValue::View { buffer, .. }) => Some((*buffer, index)),
        _ => None,
    }
}

fn render_raw_copy(
    capture_identity: u64,
    item: &ScheduleItem,
    schema: &SymbolicSchema,
    admitted: &RenderedC,
    plan: &crate::MovementKernelPlan,
) -> Result<RenderedC, JitError> {
    let copy = plan
        .raw_copy()
        .map_err(|error| JitError::Unsupported(error.to_string()))?
        .ok_or_else(|| {
            JitError::Unsupported("runtime-symbolic movement is not a raw copy".into())
        })?;
    let input = copy.input();
    let has_affine_address = copy
        .address()
        .map_err(|error| JitError::Unsupported(error.to_string()))?
        .is_some();
    if input.dtype != plan.dtype {
        return Err(JitError::Unsupported(
            "runtime-symbolic raw copy must preserve storage dtype".into(),
        ));
    }
    let abi = runtime_abi(admitted, schema)?;
    let ids = abi
        .buffers
        .iter()
        .enumerate()
        .map(|(slot, buffer)| (buffer.id, slot))
        .collect::<BTreeMap<_, _>>();
    let input_id = input.node.index() as u64;
    let output_id = item.primary_output().id;
    let output = schema
        .buffer_shapes
        .get(&output_id)
        .ok_or_else(|| JitError::Symbolic("symbolic raw-copy output shape is absent".into()))?;
    let offset = if has_affine_address {
        let view = schema
            .views
            .get(&(item.id, input_id))
            .ok_or_else(|| JitError::Symbolic("symbolic AffineCopy view is absent".into()))?;
        affine_offset_c(view, "rg_i", schema)?
    } else {
        "rg_i".into()
    };
    let extent = shape_elements_c(output, schema)?;
    let bytes = input.dtype.itemsize();
    let mut lines = prologue(capture_identity, item.id);
    lines.push(format!(
        "  for(size_t rg_i=0; rg_i<(size_t)({extent}); ++rg_i) memcpy(((uint8_t*)buffers[{}])+rg_i*{bytes}u,((const uint8_t*)buffers[{}])+(size_t)({offset})*{bytes}u,{bytes}u);",
        ids[&output_id], ids[&input_id]
    ));
    lines.push("  return 0;".into());
    lines.push("}".into());
    finish(capture_identity, item, abi, lines)
}

fn render_matmul(
    capture_identity: u64,
    item: &ScheduleItem,
    schema: &SymbolicSchema,
    admitted: &RenderedC,
    plan: &crate::MatmulKernelPlan,
    domain: &SymbolicItemDomain,
) -> Result<RenderedC, JitError> {
    if !matches!(plan.dtype, DType::F32 | DType::F64) {
        return Err(JitError::Unsupported(
            "runtime-symbolic Matmul supports dense F32/F64 only".into(),
        ));
    }
    if item
        .ordered_inputs()
        .iter()
        .any(|binding| binding.desc.view.is_some())
    {
        return Err(JitError::Unsupported(
            "runtime-symbolic Matmul requires dense inputs".into(),
        ));
    }
    let SymbolicItemDomain::Matmul {
        lhs_buffer,
        rhs_buffer,
        output,
        batch,
        m,
        n,
        k,
    } = domain
    else {
        return Err(JitError::Symbolic(
            "symbolic Matmul domain is malformed".into(),
        ));
    };
    let abi = runtime_abi(admitted, schema)?;
    let ids = abi
        .buffers
        .iter()
        .enumerate()
        .map(|(slot, buffer)| (buffer.id, slot))
        .collect::<BTreeMap<_, _>>();
    let lhs_shape = schema
        .buffer_shapes
        .get(lhs_buffer)
        .ok_or_else(|| JitError::Symbolic("symbolic Matmul lhs shape is absent".into()))?;
    let rhs_shape = schema
        .buffer_shapes
        .get(rhs_buffer)
        .ok_or_else(|| JitError::Symbolic("symbolic Matmul rhs shape is absent".into()))?;
    let lhs_batch = SymbolicShape::new(
        lhs_shape.dims()[..lhs_shape
            .rank()
            .saturating_sub(if plan.lhs_vector { 1 } else { 2 })]
            .to_vec(),
    );
    let rhs_batch = SymbolicShape::new(
        rhs_shape.dims()[..rhs_shape
            .rank()
            .saturating_sub(if plan.rhs_vector { 1 } else { 2 })]
            .to_vec(),
    );
    let out_len = shape_elements_c(output, schema)?;
    let m = expression_c(m, schema)?;
    let n = expression_c(n, schema)?;
    let k = expression_c(k, schema)?;
    let mut lines = prologue(capture_identity, item.id);
    lines.push(format!(
        "  for(size_t rg_i=0; rg_i<(size_t)({out_len}); ++rg_i) {{"
    ));
    lines.push("    size_t rg_q=rg_i,rg_col=0,rg_row=0;".into());
    if !plan.rhs_vector {
        lines.push(format!(
            "    rg_col=rg_q%(size_t)({n}); rg_q/=(size_t)({n});"
        ));
    }
    if !plan.lhs_vector {
        lines.push(format!(
            "    rg_row=rg_q%(size_t)({m}); rg_q/=(size_t)({m});"
        ));
    }
    let lhs_batch_offset = broadcast_offset_c(&lhs_batch, batch, "rg_q", schema)?;
    let rhs_batch_offset = broadcast_offset_c(&rhs_batch, batch, "rg_q", schema)?;
    let lhs_offset = if plan.lhs_vector {
        format!("((size_t)({lhs_batch_offset})*(size_t)({k})+rg_k)")
    } else {
        format!("(((size_t)({lhs_batch_offset})*(size_t)({m})+rg_row)*(size_t)({k})+rg_k)")
    };
    let rhs_offset = if plan.rhs_vector {
        format!("((size_t)({rhs_batch_offset})*(size_t)({k})+rg_k)")
    } else {
        format!("(((size_t)({rhs_batch_offset})*(size_t)({k})+rg_k)*(size_t)({n})+rg_col)")
    };
    let storage = ctype(plan.dtype);
    if plan.dtype == DType::F32 {
        lines.push("    float rg_acc=0.0f;".into());
        lines.push(format!(
            "    for(size_t rg_k=0; rg_k<(size_t)({k}); ++rg_k) {{ float rg_product=(float)(((const {storage}*)buffers[{}])[{lhs_offset}]*((const {storage}*)buffers[{}])[{rhs_offset}]); rg_acc=(float)(rg_acc+rg_product); }}",
            ids[lhs_buffer], ids[rhs_buffer]
        ));
    } else {
        lines.push("    double rg_acc=0.0;".into());
        lines.push(format!(
            "    for(size_t rg_k=0; rg_k<(size_t)({k}); ++rg_k) rg_acc+=((const {storage}*)buffers[{}])[{lhs_offset}]*((const {storage}*)buffers[{}])[{rhs_offset}];",
            ids[lhs_buffer], ids[rhs_buffer]
        ));
    }
    lines.push(format!(
        "    (({storage}*)buffers[{}])[rg_i]=rg_acc;",
        ids[&item.primary_output().id]
    ));
    lines.push("  }".into());
    lines.push("  return 0;".into());
    lines.push("}".into());
    finish(capture_identity, item, abi, lines)
}

struct Offsets<'a> {
    item: u64,
    projected_ordinals: &'a BTreeMap<UOp, u32>,
    schema: &'a SymbolicSchema,
    domain: &'a SymbolicShape,
}

impl SymbolicLoadOffsets for Offsets<'_> {
    fn offset(&self, index: &UOp, buffer: u64) -> Result<String, JitError> {
        if crate::projected_index::ProjectedIndexPlan::is_projected(index) {
            let ordinal =
                self.projected_ordinals.get(index).copied().ok_or_else(|| {
                    JitError::Symbolic("projected Index ordinal is absent".into())
                })?;
            let projected = self
                .schema
                .projected
                .get(&(self.item, ordinal))
                .ok_or_else(|| {
                    JitError::Symbolic("symbolic projected expression is absent".into())
                })?;
            return projected.render(&mut SymbolicProjectedC {
                linear: "((int64_t)rg_i)",
                schema: self.schema,
            });
        }
        if let Some(view) = self.schema.views.get(&(self.item, buffer)) {
            let logical =
                broadcast_offset_c(&view.logical_shape, self.domain, "rg_i", self.schema)?;
            affine_offset_c(view, &logical, self.schema)
        } else {
            let input = self.schema.buffer_shapes.get(&buffer).ok_or_else(|| {
                JitError::Symbolic(format!("symbolic load buffer {buffer} is absent"))
            })?;
            broadcast_offset_c(input, self.domain, "rg_i", self.schema)
        }
    }
}

fn projected_ordinals(kernel: &UOp) -> Result<BTreeMap<UOp, u32>, JitError> {
    kernel
        .topological()
        .map_err(|error| JitError::Symbolic(error.to_string()))?
        .into_iter()
        .filter(|node| {
            matches!(
                node.operation(),
                Operation::Index(IndexValue::Buffer {
                    addressing: IndexAddressing::Projected,
                    ..
                })
            )
        })
        .enumerate()
        .map(|(ordinal, node)| {
            Ok((
                node,
                u32::try_from(ordinal).map_err(|_| {
                    JitError::Symbolic("too many projected indices in one kernel".into())
                })?,
            ))
        })
        .collect()
}

struct SymbolicProjectedC<'a> {
    linear: &'a str,
    schema: &'a SymbolicSchema,
}

impl ProjectedIndexEmitter<SymbolicExpr> for SymbolicProjectedC<'_> {
    type Value = String;
    type Error = JitError;

    fn linear(&mut self) -> Result<Self::Value, Self::Error> {
        Ok(self.linear.into())
    }

    fn constant(&mut self, value: &SymbolicExpr) -> Result<Self::Value, Self::Error> {
        expression_c(value, self.schema)
    }

    fn binary(
        &mut self,
        operation: Binary,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        match operation {
            Binary::Add => Ok(format!("(({lhs})+({rhs}))")),
            Binary::Sub => Ok(format!("(({lhs})-({rhs}))")),
            Binary::Mul => Ok(format!("(({lhs})*({rhs}))")),
            Binary::FloorDiv => Ok(format!("rg_floor_div(({lhs}),({rhs}))")),
            Binary::Mod => Ok(format!("rg_floor_mod(({lhs}),({rhs}))")),
            _ => Err(JitError::Symbolic(
                "symbolic projected operation is unsupported".into(),
            )),
        }
    }
}

fn broadcast_offset_c(
    input: &SymbolicShape,
    output: &SymbolicShape,
    linear: &str,
    schema: &SymbolicSchema,
) -> Result<String, JitError> {
    if input.rank() > output.rank() {
        return Err(JitError::Symbolic(
            "symbolic broadcast rank is inconsistent".into(),
        ));
    }
    if input == output {
        return Ok(linear.into());
    }
    let pad = output.rank() - input.rank();
    let mut coordinates = Vec::new();
    let mut dimensions = Vec::new();
    for (axis, dimension) in input.dims().iter().enumerate() {
        if dimension
            .expression()
            .bounds()
            .map_err(|error| JitError::Symbolic(error.to_string()))?
            .constant()
            == Some(1)
        {
            continue;
        }
        let divisor = shape_elements_c(
            &SymbolicShape::new(output.dims()[pad + axis + 1..].to_vec()),
            schema,
        )?;
        let dimension_c = expression_c(dimension.expression(), schema)?;
        coordinates.push(format!(
            "(((size_t)({divisor})==0u||(size_t)({dimension_c})==0u)?0u:((({linear})/(size_t)({divisor}))%(size_t)({dimension_c})))"
        ));
        dimensions.push(dimension_c);
    }
    let Some(mut offset) = coordinates.first().cloned() else {
        return Ok("0u".into());
    };
    for (coordinate, dimension) in coordinates
        .into_iter()
        .skip(1)
        .zip(dimensions.iter().skip(1))
    {
        offset = format!("(({offset})*(size_t)({dimension})+({coordinate}))");
    }
    Ok(offset)
}

fn affine_offset_c(
    view: &SymbolicViewMap,
    logical: &str,
    schema: &SymbolicSchema,
) -> Result<String, JitError> {
    let mut offset = format!("(int64_t)({})", expression_c(&view.offset, schema)?);
    for (axis, (dimension, stride)) in view
        .logical_shape
        .dims()
        .iter()
        .zip(&view.strides)
        .enumerate()
    {
        let divisor = shape_elements_c(
            &SymbolicShape::new(view.logical_shape.dims()[axis + 1..].to_vec()),
            schema,
        )?;
        let dimension = expression_c(dimension.expression(), schema)?;
        let stride = expression_c(stride, schema)?;
        offset = format!(
            "({offset}+(((size_t)({divisor})==0u||(size_t)({dimension})==0u)?0:(int64_t)((({logical})/(size_t)({divisor}))%(size_t)({dimension}))*(int64_t)({stride})))"
        );
    }
    Ok(offset)
}

fn reduction_index_c(
    input: &SymbolicShape,
    output: &SymbolicShape,
    axes: &[usize],
    keepdim: bool,
    reduction_linear: &str,
    schema: &SymbolicSchema,
) -> Result<String, JitError> {
    let mut coordinates = Vec::with_capacity(input.rank());
    let mut out_axis = 0usize;
    let mut reduction_axis = 0usize;
    for axis in 0..input.rank() {
        let dimension = expression_c(input.dims()[axis].expression(), schema)?;
        let coordinate = if axes.contains(&axis) {
            let trailing = axes[reduction_axis + 1..]
                .iter()
                .map(|axis| input.dims()[*axis].clone())
                .collect::<Vec<_>>();
            reduction_axis += 1;
            let divisor = shape_elements_c(&SymbolicShape::new(trailing), schema)?;
            format!(
                "(((size_t)({divisor})==0u||(size_t)({dimension})==0u)?0u:(({reduction_linear}/(size_t)({divisor}))%(size_t)({dimension})))"
            )
        } else {
            let output_axis = if keepdim {
                axis
            } else {
                let current = out_axis;
                out_axis += 1;
                current
            };
            let divisor = shape_elements_c(
                &SymbolicShape::new(output.dims()[output_axis + 1..].to_vec()),
                schema,
            )?;
            if keepdim {
                out_axis += 1;
            }
            format!(
                "(((size_t)({divisor})==0u||(size_t)({dimension})==0u)?0u:((rg_out/(size_t)({divisor}))%(size_t)({dimension})))"
            )
        };
        coordinates.push((coordinate, dimension));
    }
    let Some((mut index, _)) = coordinates.first().cloned() else {
        return Ok("0u".into());
    };
    for (coordinate, dimension) in coordinates.into_iter().skip(1) {
        index = format!("(({index})*(size_t)({dimension})+({coordinate}))");
    }
    Ok(index)
}

fn shape_elements_c(shape: &SymbolicShape, schema: &SymbolicSchema) -> Result<String, JitError> {
    expression_c(
        &shape
            .numel()
            .map_err(|error| JitError::Symbolic(error.to_string()))?,
        schema,
    )
}

fn maximum_elements(shape: &SymbolicShape) -> Result<usize, JitError> {
    let maximum = shape
        .numel()
        .map_err(|error| JitError::Symbolic(error.to_string()))?
        .bounds()
        .map_err(|error| JitError::Symbolic(error.to_string()))?
        .max;
    usize::try_from(maximum)
        .map_err(|_| JitError::Symbolic("symbolic maximum extent is not usize".into()))
}

fn expression_c(expression: &SymbolicExpr, schema: &SymbolicSchema) -> Result<String, JitError> {
    let lowered = lower_symbolic_expression(expression, schema.parameters())
        .map_err(|error| JitError::Symbolic(error.to_string()))?;
    render_symbolic_expression(&lowered)
}

/// Renders only the authenticated scalar subset produced by
/// `lower_symbolic_expression`. Keeping this consumer typed prevents an
/// arbitrary legacy `DefineVar` UOp from acquiring schema-slot semantics.
fn render_symbolic_expression(program: &AuthenticatedSymbolicUOp) -> Result<String, JitError> {
    fn render(
        node: &UOp,
        program: &AuthenticatedSymbolicUOp,
        memo: &mut BTreeMap<UOp, String>,
    ) -> Result<String, JitError> {
        if let Some(value) = memo.get(node) {
            return Ok(value.clone());
        }
        let mut source = |index: usize| render(&node.sources()[index], program, memo);
        let value = match node.operation() {
            Operation::Const(crate::uop::LiteralValue::Int(i64::MIN)) => "INT64_MIN".into(),
            Operation::Const(crate::uop::LiteralValue::Int(value)) => {
                format!("INT64_C({value})")
            }
            Operation::DefineVar(value) => {
                let slot = program.variable_slot(value).ok_or_else(|| {
                    JitError::Symbolic("symbolic variable is not authenticated".into())
                })?;
                format!("symbols[{slot}]")
            }
            Operation::Unary(crate::uop::Unary::Neg) => format!("(-({}))", source(0)?),
            Operation::Unary(crate::uop::Unary::Not) => format!("(!({}))", source(0)?),
            Operation::Binary(operation) => {
                let (left, right) = (source(0)?, source(1)?);
                match operation {
                    Binary::Add => format!("(({left})+({right}))"),
                    Binary::Mul => format!("(({left})*({right}))"),
                    Binary::FloorDiv => format!("rg_floor_div(({left}),({right}))"),
                    Binary::Mod => format!("rg_floor_mod(({left}),({right}))"),
                    Binary::Min => format!("(({left})<({right})?({left}):({right}))"),
                    Binary::Max => format!("(({left})>({right})?({left}):({right}))"),
                    Binary::Eq => format!("(({left})==({right}))"),
                    Binary::Lt => format!("(({left})<({right}))"),
                    Binary::Le => format!("(({left})<=({right}))"),
                    Binary::And => format!("(({left})&&({right}))"),
                    Binary::Or => format!("(({left})||({right}))"),
                    Binary::Sub => {
                        return Err(JitError::Symbolic(
                            "unauthenticated symbolic binary operation".into(),
                        ));
                    }
                }
            }
            Operation::Cast => format!("((int64_t)({}))", source(0)?),
            Operation::Ternary(crate::uop::Ternary::Where) => {
                format!("(({})?({}):({}))", source(0)?, source(1)?, source(2)?)
            }
            _ => {
                return Err(JitError::Symbolic(
                    "operation is outside authenticated symbolic scalar lowering".into(),
                ));
            }
        };
        memo.insert(node.clone(), value.clone());
        Ok(value)
    }

    render(program.root(), program, &mut BTreeMap::new())
}

fn prologue(capture_identity: u64, item: u64) -> Vec<String> {
    scalar_kernel_prologue(
        format!("/* runtime-symbolic capture={capture_identity:016x} item={item:016x} */"),
        true,
        true,
        vec!["static int64_t rg_floor_div(int64_t a,int64_t b){int64_t q=a/b,r=a%b;return(r&&((r<0)!=(b<0)))?q-1:q;} static int64_t rg_floor_mod(int64_t a,int64_t b){int64_t r=a%b;return(r&&((r<0)!=(b<0)))?r+b:r;}".into()],
        "int rustgrad_kernel(void **buffers,const int64_t *symbols,uint64_t *failure){failure[0]=UINT64_MAX;failure[1]=0;".into(),
    )
}

fn finish(
    capture_identity: u64,
    item: &ScheduleItem,
    abi: KernelAbi,
    lines: Vec<String>,
) -> Result<RenderedC, JitError> {
    let source = lines.join("\n") + "\n";
    let cache_key = native_cache_key(
        &format!(
            "runtime-symbolic-{capture_identity:016x}-{:016x}",
            item.cache_key
        ),
        &source,
    );
    Ok(RenderedC {
        source,
        source_map: BTreeMap::new(),
        abi,
        cache_key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SymbolicVar;
    use crate::engine::SymbolicParameter;

    #[test]
    fn symbolic_c_uses_authenticated_uop_slots_and_checked_helpers() {
        let lhs = SymbolicVar::from_artifact(20_001, "lhs".into(), -8, 8).unwrap();
        let rhs = SymbolicVar::from_artifact(20_002, "rhs".into(), 1, 8).unwrap();
        let condition = SymbolicExpr::And(
            Box::new(SymbolicExpr::Var(lhs.clone())),
            Box::new(SymbolicExpr::Lt(
                Box::new(SymbolicExpr::Var(lhs.clone())),
                Box::new(SymbolicExpr::Var(rhs.clone())),
            )),
        );
        let expression = SymbolicExpr::Where(
            Box::new(condition.clone()),
            Box::new(SymbolicExpr::FloorDiv(
                Box::new(SymbolicExpr::Var(lhs)),
                Box::new(SymbolicExpr::Var(rhs.clone())),
            )),
            Box::new(SymbolicExpr::Mod(
                Box::new(SymbolicExpr::Const(-7)),
                Box::new(SymbolicExpr::Var(rhs)),
            )),
        );
        let parameters = [
            SymbolicParameter {
                variable: SymbolicVar::from_artifact(20_001, "lhs".into(), -8, 8).unwrap(),
                dtype: DType::I64,
            },
            SymbolicParameter {
                variable: SymbolicVar::from_artifact(20_002, "rhs".into(), 1, 8).unwrap(),
                dtype: DType::I64,
            },
        ];
        let lowered = lower_symbolic_expression(&expression, &parameters).unwrap();
        let rendered = render_symbolic_expression(&lowered).unwrap();
        assert!(rendered.contains("symbols[0]"));
        assert!(rendered.contains("symbols[1]"));
        assert!(rendered.contains("rg_floor_div"));
        assert!(rendered.contains("rg_floor_mod"));
        assert_eq!(
            lowered.root().sources()[0].ty(),
            Some(crate::uop::UType::scalar(DType::Bool))
        );
        assert!(!rendered.contains("((int64_t)"));
        assert!(!rendered.contains("lhs"));
        assert!(!rendered.contains("rhs"));

        let predicate_value = lower_symbolic_expression(&condition, &parameters).unwrap();
        assert!(matches!(
            predicate_value.root().operation(),
            Operation::Cast
        ));
        assert!(
            render_symbolic_expression(&predicate_value)
                .unwrap()
                .contains("((int64_t)")
        );
    }

    #[test]
    fn symbolic_c_preserves_i64_min_without_an_invalid_literal() {
        let parameter = SymbolicParameter {
            variable: SymbolicVar::from_artifact(20_010, "unused".into(), 0, 1).unwrap(),
            dtype: DType::I64,
        };
        let lowered = lower_symbolic_expression(
            &SymbolicExpr::Const(i64::MIN),
            std::slice::from_ref(&parameter),
        )
        .unwrap();
        assert_eq!(render_symbolic_expression(&lowered).unwrap(), "INT64_MIN");
    }
}
