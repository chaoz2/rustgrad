//! Dependency-ordered MSL emission for guarded integer DAGs.
use super::{
    MetalError,
    renderer::{MetalViewAccess, broadcast_offset, metal_storage_type, unsigned_view},
    transaction::{GuardedIntegerOp, MetalTransactionAbi},
};
use crate::{DType, UArg, UOp, UOpKind};
use std::collections::BTreeMap;

pub(super) fn emit_transactional(
    root: &UOp,
    transaction: &MetalTransactionAbi,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
) -> Result<String, MetalError> {
    lines.push("  bool rg_ok = true;".into());
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
    transaction: &'a MetalTransactionAbi,
    guard_ids: BTreeMap<UOp, u32>,
    ids: &'a BTreeMap<u64, usize>,
    source_map: &'a mut BTreeMap<usize, usize>,
    lines: &'a mut Vec<String>,
    next_value: usize,
}

impl Emitter<'_> {
    fn node(&mut self, node: &UOp, indent: &str) -> Result<String, MetalError> {
        self.source_map
            .insert(self.source_map.len(), self.lines.len() + 1);
        let dtype = node
            .ty()
            .ok_or_else(|| MetalError::Unsupported("untyped transactional expression".into()))?
            .scalar;
        let name = self.value_name();
        match node.kind() {
            UOpKind::Const => {
                let value = scalar_literal(node, dtype)?;
                self.lines.push(format!(
                    "{indent}const {} {name} = {value};",
                    metal_storage_type(dtype)
                ));
            }
            UOpKind::Load => {
                let value = self.load(node, dtype)?;
                self.lines.push(format!(
                    "{indent}const {} {name} = {value};",
                    metal_storage_type(dtype)
                ));
            }
            UOpKind::Cast => {
                let source = self.node(&node.sources()[0], indent)?;
                let source_dtype = node.sources()[0].ty().unwrap().scalar;
                let value = cast_expression(source_dtype, dtype, &source)?;
                self.assign_if_ok(indent, dtype, &name, &value);
            }
            UOpKind::GraphBinary(op) => {
                let lhs = self.node(&node.sources()[0], indent)?;
                let rhs = self.node(&node.sources()[1], indent)?;
                self.lines.push(format!(
                    "{indent}{} {name} = ({})0;",
                    metal_storage_type(dtype),
                    metal_storage_type(dtype)
                ));
                if let Some(id) = self.guard_ids.get(node).copied() {
                    let operation = GuardedIntegerOp::from_binary(*op).ok_or_else(|| {
                        MetalError::InvalidBinding("guard opcode mismatch".into())
                    })?;
                    let invalid = invalid_expression(operation, dtype, &rhs);
                    let safe = guarded_value(operation, dtype, &lhs, &rhs)?;
                    let count = self.transaction.guard_count();
                    self.lines.push(format!("{indent}if (rg_ok) {{"));
                    self.lines.push(format!("{indent}  if ({invalid}) {{"));
                    self.lines.push(format!(
                        "{indent}    atomic_fetch_min_explicit(rg_status, (uint)((uint)gid * {count}u + {id}u), memory_order_relaxed);"
                    ));
                    self.lines.push(format!("{indent}    rg_ok = false;"));
                    self.lines.push(format!("{indent}  }} else {{"));
                    self.lines.push(format!("{indent}    {name} = {safe};"));
                    self.lines.push(format!("{indent}  }}"));
                    self.lines.push(format!("{indent}}}"));
                } else {
                    let value = plain_binary(*op, dtype, &lhs, &rhs)?;
                    self.lines
                        .push(format!("{indent}if (rg_ok) {name} = {value};"));
                }
            }
            UOpKind::GraphCompare(op) => {
                let lhs = self.node(&node.sources()[0], indent)?;
                let rhs = self.node(&node.sources()[1], indent)?;
                let operator = match op {
                    crate::CompareOp::Eq => "==",
                    crate::CompareOp::Ne => "!=",
                    crate::CompareOp::Lt => "<",
                    crate::CompareOp::Le => "<=",
                    crate::CompareOp::Gt => ">",
                    crate::CompareOp::Ge => ">=",
                };
                self.assign_if_ok(
                    indent,
                    DType::Bool,
                    &name,
                    &format!("(uchar)(({lhs}) {operator} ({rhs}))"),
                );
            }
            UOpKind::GraphLogical(op) => {
                let lhs = self.node(&node.sources()[0], indent)?;
                self.lines
                    .push(format!("{indent}uchar {name} = (uchar)0u;"));
                match op {
                    crate::LogicalOp::Not => self
                        .lines
                        .push(format!("{indent}if (rg_ok) {name} = (uchar)!({lhs});")),
                    crate::LogicalOp::And => {
                        self.lines.push(format!("{indent}if (rg_ok && ({lhs})) {{"));
                        let rhs = self.node(&node.sources()[1], &format!("{indent}  "))?;
                        self.lines
                            .push(format!("{indent}  if (rg_ok) {name} = (uchar)!!({rhs});"));
                        self.lines.push(format!("{indent}}}"));
                    }
                    crate::LogicalOp::Or => {
                        self.lines
                            .push(format!("{indent}if (rg_ok && ({lhs})) {name} = (uchar)1u;"));
                        self.lines.push(format!("{indent}else if (rg_ok) {{"));
                        let rhs = self.node(&node.sources()[1], &format!("{indent}  "))?;
                        self.lines
                            .push(format!("{indent}  if (rg_ok) {name} = (uchar)!!({rhs});"));
                        self.lines.push(format!("{indent}}}"));
                    }
                }
            }
            UOpKind::Ternary(crate::uop::Ternary::Where) => {
                let condition = self.node(&node.sources()[0], indent)?;
                self.lines.push(format!(
                    "{indent}{} {name} = ({})0;",
                    metal_storage_type(dtype),
                    metal_storage_type(dtype)
                ));
                self.lines
                    .push(format!("{indent}if (rg_ok && ({condition})) {{"));
                let yes = self.node(&node.sources()[1], &format!("{indent}  "))?;
                self.lines
                    .push(format!("{indent}  if (rg_ok) {name} = {yes};"));
                self.lines.push(format!("{indent}}} else if (rg_ok) {{"));
                let no = self.node(&node.sources()[2], &format!("{indent}  "))?;
                self.lines
                    .push(format!("{indent}  if (rg_ok) {name} = {no};"));
                self.lines.push(format!("{indent}}}"));
            }
            other => {
                return Err(MetalError::Unsupported(format!(
                    "transactional expression {other:?}"
                )));
            }
        }
        Ok(name)
    }

    fn assign_if_ok(&mut self, indent: &str, dtype: DType, name: &str, value: &str) {
        self.lines.push(format!(
            "{indent}{} {name} = ({})0;",
            metal_storage_type(dtype),
            metal_storage_type(dtype)
        ));
        self.lines
            .push(format!("{indent}if (rg_ok) {name} = {value};"));
    }

    fn value_name(&mut self) -> String {
        let name = format!("rg_v{}", self.next_value);
        self.next_value += 1;
        name
    }

    fn load(&self, node: &UOp, _dtype: DType) -> Result<String, MetalError> {
        let index = node
            .sources()
            .first()
            .ok_or_else(|| MetalError::Unsupported("load has no index".into()))?;
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
                return Err(MetalError::Unsupported(
                    "load requires checked static indexing".into(),
                ));
            }
        };
        let position = self
            .ids
            .get(&buffer)
            .ok_or_else(|| MetalError::InvalidBinding("load absent from ABI".into()))?;
        let logical = broadcast_offset(input_shape, output_shape, "(ulong)gid")?;
        let offset = match view {
            Some(view) => MetalViewAccess::new(&unsigned_view(view)?)?.expression(&logical),
            None => logical,
        };
        Ok(format!("b{position}[{offset}]"))
    }
}

fn scalar_literal(node: &UOp, dtype: DType) -> Result<String, MetalError> {
    match node.arg() {
        UArg::Scalar {
            dtype: DType::Bool,
            bits,
        } if dtype == DType::Bool && *bits <= 1 => Ok(format!("(uchar){bits}u")),
        UArg::Scalar {
            dtype: DType::I32,
            bits,
        } if dtype == DType::I32 => Ok(format!("as_type<int>((uint)0x{:08x}u)", *bits as u32)),
        UArg::Scalar {
            dtype: DType::U32,
            bits,
        } if dtype == DType::U32 => Ok(format!("(uint)0x{:08x}u", *bits as u32)),
        _ => Err(MetalError::Unsupported(
            "transactional scalar literal/type mismatch".into(),
        )),
    }
}

fn cast_expression(source: DType, target: DType, value: &str) -> Result<String, MetalError> {
    match (source, target) {
        (source, target) if source == target => Ok(value.into()),
        (DType::Bool, DType::I32) => Ok(format!("(int)({value})")),
        (DType::Bool, DType::U32) => Ok(format!("(uint)({value})")),
        (DType::I32 | DType::U32, DType::Bool) => Ok(format!("(uchar)(({value}) != 0)")),
        (DType::I32, DType::U32) => Ok(format!("as_type<uint>({value})")),
        (DType::U32, DType::I32) => Ok(format!("as_type<int>({value})")),
        _ => Err(MetalError::Unsupported(
            "transactional cast is outside the exact subset".into(),
        )),
    }
}

fn plain_binary(
    op: crate::BinaryOp,
    dtype: DType,
    lhs: &str,
    rhs: &str,
) -> Result<String, MetalError> {
    match (op, dtype) {
        (crate::BinaryOp::Add, DType::I32) => Ok(format!(
            "as_type<int>(as_type<uint>({lhs}) + as_type<uint>({rhs}))"
        )),
        (crate::BinaryOp::Sub, DType::I32) => Ok(format!(
            "as_type<int>(as_type<uint>({lhs}) - as_type<uint>({rhs}))"
        )),
        (crate::BinaryOp::Mul, DType::I32) => Ok(format!(
            "as_type<int>(as_type<uint>({lhs}) * as_type<uint>({rhs}))"
        )),
        (crate::BinaryOp::Add, DType::U32) => Ok(format!("(({lhs}) + ({rhs}))")),
        (crate::BinaryOp::Sub, DType::U32) => Ok(format!("(({lhs}) - ({rhs}))")),
        (crate::BinaryOp::Mul, DType::U32) => Ok(format!("(({lhs}) * ({rhs}))")),
        _ => Err(MetalError::Unsupported(format!(
            "binary {op:?} for {dtype:?} is outside the transactional subset"
        ))),
    }
}

fn invalid_expression(op: GuardedIntegerOp, dtype: DType, rhs: &str) -> String {
    if op.is_shift() {
        if dtype == DType::I32 {
            format!("(({rhs}) < 0 || (uint)({rhs}) >= 32u)")
        } else {
            format!("((uint)({rhs}) >= 32u)")
        }
    } else {
        format!("(({rhs}) == ({})0)", metal_storage_type(dtype))
    }
}

fn guarded_value(
    op: GuardedIntegerOp,
    dtype: DType,
    lhs: &str,
    rhs: &str,
) -> Result<String, MetalError> {
    let signed = dtype == DType::I32;
    if !matches!(dtype, DType::I32 | DType::U32) {
        return Err(MetalError::Unsupported(
            "guard dtype is not 32-bit integer".into(),
        ));
    }
    let min = "as_type<int>((uint)0x80000000u)";
    let overflow = if signed {
        format!("(({lhs}) == {min} && ({rhs}) == (int)-1)")
    } else {
        "false".into()
    };
    let div = format!("(({overflow}) ? ({min}) : (({lhs}) / ({rhs})))");
    let rem = format!(
        "(({overflow}) ? ({})0 : (({lhs}) % ({rhs})))",
        metal_storage_type(dtype)
    );
    Ok(match op {
        GuardedIntegerOp::Div | GuardedIntegerOp::TruncDiv => div,
        GuardedIntegerOp::FMod => rem,
        GuardedIntegerOp::FloorDiv if !signed => format!("(({lhs}) / ({rhs}))"),
        GuardedIntegerOp::Mod if !signed => format!("(({lhs}) % ({rhs}))"),
        GuardedIntegerOp::FloorDiv => {
            let rem = format!("(({lhs}) % ({rhs}))");
            format!(
                "(({overflow}) ? ({min}) : (({rem} < 0) ? (({lhs}) / ({rhs}) - ((({rhs}) > 0) ? 1 : -1)) : (({lhs}) / ({rhs}))))"
            )
        }
        GuardedIntegerOp::Mod => {
            let rem = format!("(({lhs}) % ({rhs}))");
            let magnitude = format!(
                "((({rhs}) < 0) ? ((uint)0u - as_type<uint>({rhs})) : as_type<uint>({rhs}))"
            );
            format!(
                "(({overflow}) ? (int)0 : (({rem} < 0) ? as_type<int>(as_type<uint>({rem}) + {magnitude}) : {rem}))"
            )
        }
        GuardedIntegerOp::Shl if signed => {
            format!("as_type<int>(as_type<uint>({lhs}) << (uint)({rhs}))")
        }
        GuardedIntegerOp::Shl => format!("(({lhs}) << ({rhs}))"),
        GuardedIntegerOp::Shr if signed => format!(
            "as_type<int>(((uint)({rhs}) == 0u) ? as_type<uint>({lhs}) : ((as_type<uint>({lhs}) >> (uint)({rhs})) | ((({lhs}) < 0) ? (~(uint)0u << (32u - (uint)({rhs}))) : (uint)0u)))"
        ),
        GuardedIntegerOp::Shr => format!("(({lhs}) >> ({rhs}))"),
    })
}
