//! Dependency-ordered WGSL emission for guarded integer DAGs.
use super::{
    WebGpuError,
    renderer::{WgslScalarDialect, WgslViewAccess, broadcast_offset},
    transaction::{GuardedIntegerOp, WebGpuTransactionAbi},
};
use crate::{
    DType, IndexValue, LiteralValue, Operation, UOp,
    runtime::scalar_lane::{emit_scalar_lane, project_scalar_lane},
};
use std::collections::BTreeMap;

pub(super) fn emit_transactional(
    root: &UOp,
    transaction: &WebGpuTransactionAbi,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
) -> Result<String, WebGpuError> {
    lines.push("  var rg_ok: bool = true;".into());
    Emitter {
        transaction,
        guard_ids: transaction
            .guards
            .iter()
            .map(|guard| (guard.expression.clone(), guard.id))
            .collect(),
        ids,
        source_map,
        lines,
        next_value: 0,
    }
    .node(root, "  ")
}

struct Emitter<'a> {
    transaction: &'a WebGpuTransactionAbi,
    guard_ids: BTreeMap<UOp, u32>,
    ids: &'a BTreeMap<u64, usize>,
    source_map: &'a mut BTreeMap<usize, usize>,
    lines: &'a mut Vec<String>,
    next_value: usize,
}

impl Emitter<'_> {
    fn node(&mut self, node: &UOp, indent: &str) -> Result<String, WebGpuError> {
        self.source_map
            .insert(self.source_map.len(), self.lines.len() + 1);
        let dtype = node
            .ty()
            .ok_or_else(|| WebGpuError::Unsupported("untyped transactional expression".into()))?
            .scalar;
        supported_transaction_dtype(dtype)?;
        let name = self.value_name();
        match node.operation() {
            Operation::Const(_) => {
                let value = scalar_literal(node, dtype)?;
                self.lines.push(format!(
                    "{indent}let {name}: {} = {value};",
                    wgsl_value_type(dtype)
                ));
            }
            Operation::Load => {
                let value = self.load(node, dtype)?;
                self.lines.push(format!(
                    "{indent}let {name}: {} = {value};",
                    wgsl_value_type(dtype)
                ));
            }
            Operation::Cast => {
                let source = self.node(&node.sources()[0], indent)?;
                let value = self.pure_expression(node, vec![source])?;
                self.assign_if_ok(indent, dtype, &name, &value);
            }
            Operation::GraphUnary(_) => {
                let source = self.node(&node.sources()[0], indent)?;
                let value = self.pure_expression(node, vec![source])?;
                self.assign_if_ok(indent, dtype, &name, &value);
            }
            Operation::GraphBinary(op) => {
                let lhs = self.node(&node.sources()[0], indent)?;
                let rhs = self.node(&node.sources()[1], indent)?;
                self.declare_zero(indent, dtype, &name);
                if let Some(id) = self.guard_ids.get(node).copied() {
                    let operation = GuardedIntegerOp::from_binary(*op).ok_or_else(|| {
                        WebGpuError::InvalidBinding("guard opcode mismatch".into())
                    })?;
                    let invalid = invalid_expression(operation, dtype, &rhs)?;
                    let value = guarded_value(operation, dtype, &lhs, &rhs)?;
                    let count = self.transaction.guard_count();
                    self.lines.push(format!("{indent}if (rg_ok) {{"));
                    self.lines.push(format!("{indent}  if ({invalid}) {{"));
                    self.lines.push(format!(
                        "{indent}    atomicMin(&rg_status.value, gid * {count}u + {id}u);"
                    ));
                    self.lines.push(format!("{indent}    rg_ok = false;"));
                    self.lines.push(format!("{indent}  }} else {{"));
                    self.lines.push(format!("{indent}    {name} = {value};"));
                    self.lines.push(format!("{indent}  }}"));
                    self.lines.push(format!("{indent}}}"));
                } else {
                    let value = self.pure_expression(node, vec![lhs, rhs])?;
                    self.lines
                        .push(format!("{indent}if (rg_ok) {{ {name} = {value}; }}"));
                }
            }
            Operation::Binary(_) => {
                let lhs = self.node(&node.sources()[0], indent)?;
                let rhs = self.node(&node.sources()[1], indent)?;
                self.declare_zero(indent, dtype, &name);
                let value = self.pure_expression(node, vec![lhs, rhs])?;
                self.lines
                    .push(format!("{indent}if (rg_ok) {{ {name} = {value}; }}"));
            }
            Operation::GraphCompare(_) => {
                let lhs = self.node(&node.sources()[0], indent)?;
                let rhs = self.node(&node.sources()[1], indent)?;
                let value = self.pure_expression(node, vec![lhs, rhs])?;
                self.assign_if_ok(indent, DType::Bool, &name, &value);
            }
            Operation::GraphLogical(op) => {
                let lhs = self.node(&node.sources()[0], indent)?;
                self.declare_zero(indent, DType::Bool, &name);
                match op {
                    crate::LogicalOp::Not => self
                        .lines
                        .push(format!("{indent}if (rg_ok) {{ {name} = !({lhs}); }}")),
                    crate::LogicalOp::And => {
                        self.lines.push(format!("{indent}if (rg_ok && ({lhs})) {{"));
                        let rhs = self.node(&node.sources()[1], &format!("{indent}  "))?;
                        self.lines
                            .push(format!("{indent}  if (rg_ok) {{ {name} = {rhs}; }}"));
                        self.lines.push(format!("{indent}}}"));
                    }
                    crate::LogicalOp::Or => {
                        self.lines.push(format!(
                            "{indent}if (rg_ok && ({lhs})) {{ {name} = true; }}"
                        ));
                        self.lines.push(format!("{indent}else if (rg_ok) {{"));
                        let rhs = self.node(&node.sources()[1], &format!("{indent}  "))?;
                        self.lines
                            .push(format!("{indent}  if (rg_ok) {{ {name} = {rhs}; }}"));
                        self.lines.push(format!("{indent}}}"));
                    }
                }
            }
            Operation::Ternary(crate::uop::Ternary::Where) => {
                let condition = self.node(&node.sources()[0], indent)?;
                self.declare_zero(indent, dtype, &name);
                self.lines
                    .push(format!("{indent}if (rg_ok && ({condition})) {{"));
                let yes = self.node(&node.sources()[1], &format!("{indent}  "))?;
                self.lines
                    .push(format!("{indent}  if (rg_ok) {{ {name} = {yes}; }}"));
                self.lines.push(format!("{indent}}} else if (rg_ok) {{"));
                let no = self.node(&node.sources()[2], &format!("{indent}  "))?;
                self.lines
                    .push(format!("{indent}  if (rg_ok) {{ {name} = {no}; }}"));
                self.lines.push(format!("{indent}}}"));
            }
            other => {
                return Err(WebGpuError::Unsupported(format!(
                    "transactional expression {other:?}"
                )));
            }
        }
        Ok(name)
    }

    fn pure_expression(&self, node: &UOp, sources: Vec<String>) -> Result<String, WebGpuError> {
        let instruction = project_scalar_lane(node, &sources)
            .map_err(WebGpuError::Unsupported)?
            .ok_or_else(|| {
                WebGpuError::Unsupported(format!("transactional expression {:?}", node.operation()))
            })?;
        emit_scalar_lane(&WgslScalarDialect, &instruction).map_err(WebGpuError::Unsupported)
    }

    fn assign_if_ok(&mut self, indent: &str, dtype: DType, name: &str, value: &str) {
        self.declare_zero(indent, dtype, name);
        self.lines
            .push(format!("{indent}if (rg_ok) {{ {name} = {value}; }}"));
    }

    fn declare_zero(&mut self, indent: &str, dtype: DType, name: &str) {
        self.lines.push(format!(
            "{indent}var {name}: {} = {};",
            wgsl_value_type(dtype),
            zero(dtype)
        ));
    }

    fn value_name(&mut self) -> String {
        let name = format!("rg_v{}", self.next_value);
        self.next_value += 1;
        name
    }

    fn load(&self, node: &UOp, dtype: DType) -> Result<String, WebGpuError> {
        let index = node
            .sources()
            .first()
            .ok_or_else(|| WebGpuError::Unsupported("load has no index".into()))?;
        let (buffer, input_shape, output_shape, view) = match index.operation() {
            Operation::Index(IndexValue::Buffer {
                buffer,
                input_shape,
                output_shape,
                ..
            }) => (*buffer, input_shape, output_shape, None),
            Operation::Index(IndexValue::View {
                buffer,
                input_shape,
                output_shape,
                view,
                ..
            }) => (*buffer, input_shape, output_shape, Some(view)),
            _ => {
                return Err(WebGpuError::Unsupported(
                    "load requires checked static indexing".into(),
                ));
            }
        };
        let position = self
            .ids
            .get(&buffer)
            .ok_or_else(|| WebGpuError::InvalidBinding("load absent from ABI".into()))?;
        let logical = broadcast_offset(input_shape, output_shape, "gid")?;
        let offset = match view {
            Some(view) => WgslViewAccess::new(view)?.expression(&logical),
            None => logical,
        };
        Ok(if dtype == DType::Bool {
            format!("(((b{position}[({offset}) >> 2u] >> ((({offset}) & 3u) * 8u)) & 0xffu) != 0u)")
        } else {
            format!("b{position}[{offset}]")
        })
    }
}

fn supported_transaction_dtype(dtype: DType) -> Result<(), WebGpuError> {
    if matches!(dtype, DType::Bool | DType::I32 | DType::U32) {
        Ok(())
    } else {
        Err(WebGpuError::Unsupported(format!(
            "dtype {dtype:?} is outside the guarded WGSL subset"
        )))
    }
}

fn wgsl_value_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::I32 => "i32",
        DType::U32 => "u32",
        _ => unreachable!("validated transaction dtype"),
    }
}

fn zero(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "false",
        DType::I32 => "0i",
        DType::U32 => "0u",
        _ => unreachable!("validated transaction dtype"),
    }
}

fn scalar_literal(node: &UOp, dtype: DType) -> Result<String, WebGpuError> {
    match node.operation() {
        Operation::Const(LiteralValue::Scalar {
            dtype: actual,
            bits,
        }) if *actual == DType::Bool && dtype == DType::Bool && *bits <= 1 => Ok(if *bits == 0 {
            "false".into()
        } else {
            "true".into()
        }),
        Operation::Const(LiteralValue::Scalar {
            dtype: actual,
            bits,
        }) if *actual == DType::I32 && dtype == DType::I32 => {
            Ok(format!("bitcast<i32>(0x{:08x}u)", *bits as u32))
        }
        Operation::Const(LiteralValue::Scalar {
            dtype: actual,
            bits,
        }) if *actual == DType::U32 && dtype == DType::U32 => {
            Ok(format!("0x{:08x}u", *bits as u32))
        }
        _ => Err(WebGpuError::Unsupported(
            "transactional scalar literal/type mismatch".into(),
        )),
    }
}

fn invalid_expression(
    op: GuardedIntegerOp,
    dtype: DType,
    rhs: &str,
) -> Result<String, WebGpuError> {
    if !matches!(dtype, DType::I32 | DType::U32) {
        return Err(WebGpuError::Unsupported(
            "guard dtype is not 32-bit integer".into(),
        ));
    }
    Ok(if op.is_shift() {
        if dtype == DType::I32 {
            format!("(({rhs}) < 0i || bitcast<u32>({rhs}) >= 32u)")
        } else {
            format!("(({rhs}) >= 32u)")
        }
    } else if dtype == DType::I32 {
        format!("(({rhs}) == 0i)")
    } else {
        format!("(({rhs}) == 0u)")
    })
}

fn guarded_value(
    op: GuardedIntegerOp,
    dtype: DType,
    lhs: &str,
    rhs: &str,
) -> Result<String, WebGpuError> {
    Ok(match (op, dtype) {
        (GuardedIntegerOp::Div | GuardedIntegerOp::TruncDiv, DType::I32) => {
            format!("rg_i32_trunc_div({lhs}, {rhs})")
        }
        (GuardedIntegerOp::FloorDiv, DType::I32) => {
            format!("rg_i32_floor_div({lhs}, {rhs})")
        }
        (GuardedIntegerOp::Mod, DType::I32) => format!("rg_i32_mod({lhs}, {rhs})"),
        (GuardedIntegerOp::FMod, DType::I32) => format!("rg_i32_fmod({lhs}, {rhs})"),
        (GuardedIntegerOp::Shl, DType::I32) => {
            format!("bitcast<i32>(bitcast<u32>({lhs}) << bitcast<u32>({rhs}))")
        }
        (GuardedIntegerOp::Shr, DType::I32) => {
            format!("(({lhs}) >> bitcast<u32>({rhs}))")
        }
        (
            GuardedIntegerOp::Div | GuardedIntegerOp::FloorDiv | GuardedIntegerOp::TruncDiv,
            DType::U32,
        ) => format!("(({lhs}) / ({rhs}))"),
        (GuardedIntegerOp::Mod | GuardedIntegerOp::FMod, DType::U32) => {
            format!("(({lhs}) % ({rhs}))")
        }
        (GuardedIntegerOp::Shl, DType::U32) => format!("(({lhs}) << ({rhs}))"),
        (GuardedIntegerOp::Shr, DType::U32) => format!("(({lhs}) >> ({rhs}))"),
        _ => {
            return Err(WebGpuError::Unsupported(
                "guard dtype is not 32-bit integer".into(),
            ));
        }
    })
}
