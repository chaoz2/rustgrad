//! Dependency-ordered WGSL emission for guarded integer DAGs.
use super::{
    WebGpuError,
    renderer::{WgslViewAccess, broadcast_offset, ordered_compare_operand},
    transaction::{GuardedIntegerOp, WebGpuTransactionAbi},
};
use crate::{DType, UArg, UOp, UOpKind};
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
        match node.kind() {
            UOpKind::Const => {
                let value = scalar_literal(node, dtype)?;
                self.lines.push(format!(
                    "{indent}let {name}: {} = {value};",
                    wgsl_value_type(dtype)
                ));
            }
            UOpKind::Load => {
                let value = self.load(node, dtype)?;
                self.lines.push(format!(
                    "{indent}let {name}: {} = {value};",
                    wgsl_value_type(dtype)
                ));
            }
            UOpKind::Cast => {
                let source = self.node(&node.sources()[0], indent)?;
                let source_dtype = node.sources()[0]
                    .ty()
                    .ok_or_else(|| WebGpuError::Unsupported("untyped cast source".into()))?
                    .scalar;
                let value = cast_expression(source_dtype, dtype, &source)?;
                self.assign_if_ok(indent, dtype, &name, &value);
            }
            UOpKind::GraphBinary(op) => {
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
                    let value = plain_binary(*op, dtype, &lhs, &rhs)?;
                    self.lines
                        .push(format!("{indent}if (rg_ok) {{ {name} = {value}; }}"));
                }
            }
            UOpKind::Binary(op) => {
                let lhs = self.node(&node.sources()[0], indent)?;
                let rhs = self.node(&node.sources()[1], indent)?;
                self.declare_zero(indent, dtype, &name);
                use crate::uop::Binary::{Add, Eq, Le, Lt, Mul, Sub};
                let value = match op {
                    Add => plain_binary(crate::BinaryOp::Add, dtype, &lhs, &rhs)?,
                    Sub => plain_binary(crate::BinaryOp::Sub, dtype, &lhs, &rhs)?,
                    Mul => plain_binary(crate::BinaryOp::Mul, dtype, &lhs, &rhs)?,
                    Eq => format!("(({lhs}) == ({rhs}))"),
                    Lt | Le => {
                        let operand_dtype = node.sources()[0]
                            .ty()
                            .ok_or_else(|| {
                                WebGpuError::Unsupported("untyped compare source".into())
                            })?
                            .scalar;
                        let lhs = ordered_compare_operand(operand_dtype, &lhs);
                        let rhs = ordered_compare_operand(operand_dtype, &rhs);
                        let operator = if matches!(op, Lt) { "<" } else { "<=" };
                        format!("(({lhs}) {operator} ({rhs}))")
                    }
                    _ => {
                        return Err(WebGpuError::Unsupported(format!(
                            "transactional core binary {op:?}"
                        )));
                    }
                };
                self.lines
                    .push(format!("{indent}if (rg_ok) {{ {name} = {value}; }}"));
            }
            UOpKind::GraphCompare(op) => {
                let lhs = self.node(&node.sources()[0], indent)?;
                let rhs = self.node(&node.sources()[1], indent)?;
                let operand_dtype = node.sources()[0]
                    .ty()
                    .ok_or_else(|| WebGpuError::Unsupported("untyped compare source".into()))?
                    .scalar;
                let operator = match op {
                    crate::CompareOp::Eq => "==",
                    crate::CompareOp::Ne => "!=",
                    crate::CompareOp::Lt => "<",
                    crate::CompareOp::Le => "<=",
                    crate::CompareOp::Gt => ">",
                    crate::CompareOp::Ge => ">=",
                };
                let lhs = if matches!(op, crate::CompareOp::Eq | crate::CompareOp::Ne) {
                    lhs
                } else {
                    ordered_compare_operand(operand_dtype, &lhs)
                };
                let rhs = if matches!(op, crate::CompareOp::Eq | crate::CompareOp::Ne) {
                    rhs
                } else {
                    ordered_compare_operand(operand_dtype, &rhs)
                };
                self.assign_if_ok(
                    indent,
                    DType::Bool,
                    &name,
                    &format!("(({lhs}) {operator} ({rhs}))"),
                );
            }
            UOpKind::GraphLogical(op) => {
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
            UOpKind::Ternary(crate::uop::Ternary::Where) => {
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
        let (buffer, input_shape, output_shape, view) = match index.arg() {
            UArg::BufferIndex {
                buffer,
                input_shape,
                output_shape,
                ..
            } => (*buffer, input_shape, output_shape, None),
            UArg::ViewBufferIndex {
                buffer,
                input_shape,
                output_shape,
                view,
                ..
            } => (*buffer, input_shape, output_shape, Some(view)),
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
    match node.arg() {
        UArg::Scalar {
            dtype: DType::Bool,
            bits,
        } if dtype == DType::Bool && *bits <= 1 => Ok(if *bits == 0 {
            "false".into()
        } else {
            "true".into()
        }),
        UArg::Scalar {
            dtype: DType::I32,
            bits,
        } if dtype == DType::I32 => Ok(format!("bitcast<i32>(0x{:08x}u)", *bits as u32)),
        UArg::Scalar {
            dtype: DType::U32,
            bits,
        } if dtype == DType::U32 => Ok(format!("0x{:08x}u", *bits as u32)),
        _ => Err(WebGpuError::Unsupported(
            "transactional scalar literal/type mismatch".into(),
        )),
    }
}

fn cast_expression(source: DType, target: DType, value: &str) -> Result<String, WebGpuError> {
    match (source, target) {
        (source, target) if source == target => Ok(value.into()),
        (DType::Bool, DType::I32) => Ok(format!("select(0i, 1i, {value})")),
        (DType::Bool, DType::U32) => Ok(format!("select(0u, 1u, {value})")),
        (DType::I32, DType::Bool) => Ok(format!("(({value}) != 0i)")),
        (DType::U32, DType::Bool) => Ok(format!("(({value}) != 0u)")),
        (DType::I32, DType::U32) => Ok(format!("bitcast<u32>({value})")),
        (DType::U32, DType::I32) => Ok(format!("bitcast<i32>({value})")),
        _ => Err(WebGpuError::Unsupported(
            "transactional cast is outside the exact subset".into(),
        )),
    }
}

fn plain_binary(
    op: crate::BinaryOp,
    dtype: DType,
    lhs: &str,
    rhs: &str,
) -> Result<String, WebGpuError> {
    match (op, dtype) {
        (crate::BinaryOp::Add, DType::I32) => Ok(format!(
            "bitcast<i32>(bitcast<u32>({lhs}) + bitcast<u32>({rhs}))"
        )),
        (crate::BinaryOp::Sub, DType::I32) => Ok(format!(
            "bitcast<i32>(bitcast<u32>({lhs}) - bitcast<u32>({rhs}))"
        )),
        (crate::BinaryOp::Mul, DType::I32) => Ok(format!(
            "bitcast<i32>(bitcast<u32>({lhs}) * bitcast<u32>({rhs}))"
        )),
        (crate::BinaryOp::Add, DType::U32) => Ok(format!("(({lhs}) + ({rhs}))")),
        (crate::BinaryOp::Sub, DType::U32) => Ok(format!("(({lhs}) - ({rhs}))")),
        (crate::BinaryOp::Mul, DType::U32) => Ok(format!("(({lhs}) * ({rhs}))")),
        (crate::BinaryOp::Add, DType::Bool) => Ok(format!("(({lhs}) || ({rhs}))")),
        (crate::BinaryOp::Sub, DType::Bool) => Ok(format!("(({lhs}) != ({rhs}))")),
        (crate::BinaryOp::Mul, DType::Bool) => Ok(format!("(({lhs}) && ({rhs}))")),
        _ => Err(WebGpuError::Unsupported(format!(
            "binary {op:?} for {dtype:?} is outside the guarded WGSL subset"
        ))),
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
