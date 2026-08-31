//! Dependency-ordered OpenCL C emission for transactional elementwise DAGs.
use super::{
    OpenClCapabilities, OpenClError, narrow,
    renderer::{OpenClScalarDialect, broadcast_offset, cl_type, guarded_value},
    transaction::OpenClTransactionAbi,
    view::OpenClViewAccess,
};
use crate::{
    DType, IndexValue, LiteralValue, Operation, UOp,
    runtime::scalar_lane::{emit_scalar_lane, project_scalar_lane},
};
use std::collections::BTreeMap;

pub(super) fn emit_transactional(
    root: &UOp,
    transaction: &OpenClTransactionAbi,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    capabilities: OpenClCapabilities,
) -> Result<String, OpenClError> {
    lines.push("  uchar rg_ok = (uchar)1u;".into());
    emit_at(
        root,
        EmitContext {
            transaction,
            ids,
            source_map,
            lines,
            capabilities,
        },
        "gid",
        "  ",
    )
}

pub(super) fn emit_transactional_reduction(
    root: &UOp,
    transaction: &OpenClTransactionAbi,
    ids: &BTreeMap<u64, usize>,
    source_map: &mut BTreeMap<usize, usize>,
    lines: &mut Vec<String>,
    capabilities: OpenClCapabilities,
) -> Result<String, OpenClError> {
    emit_at(
        root,
        EmitContext {
            transaction,
            ids,
            source_map,
            lines,
            capabilities,
        },
        "src_gid",
        "    ",
    )
}

struct EmitContext<'a> {
    transaction: &'a OpenClTransactionAbi,
    ids: &'a BTreeMap<u64, usize>,
    source_map: &'a mut BTreeMap<usize, usize>,
    lines: &'a mut Vec<String>,
    capabilities: OpenClCapabilities,
}

fn emit_at(
    root: &UOp,
    context: EmitContext<'_>,
    linear: &str,
    indent: &str,
) -> Result<String, OpenClError> {
    let EmitContext {
        transaction,
        ids,
        source_map,
        lines,
        capabilities,
    } = context;
    let mut emitter = Emitter {
        transaction,
        guard_ids: transaction.guard_ids(),
        ids,
        source_map,
        lines,
        next_value: 0,
        linear,
        capabilities,
    };
    emitter.node(root, indent)
}

struct Emitter<'a> {
    transaction: &'a OpenClTransactionAbi,
    guard_ids: BTreeMap<UOp, u32>,
    ids: &'a BTreeMap<u64, usize>,
    source_map: &'a mut BTreeMap<usize, usize>,
    lines: &'a mut Vec<String>,
    next_value: usize,
    linear: &'a str,
    capabilities: OpenClCapabilities,
}

impl Emitter<'_> {
    fn node(&mut self, node: &UOp, indent: &str) -> Result<String, OpenClError> {
        let map_id = self.source_map.len();
        self.source_map.insert(map_id, self.lines.len() + 1);
        let dtype = node
            .ty()
            .ok_or_else(|| OpenClError::Unsupported("untyped transactional expression".into()))?
            .scalar;
        let name = self.value_name();
        match node.operation() {
            Operation::Const(_) => {
                let value = scalar_literal(node, dtype)?;
                self.lines.push(format!(
                    "{indent}const {} {name} = {value};",
                    expression_type(dtype)
                ));
            }
            Operation::Load => {
                let value = self.load(node, dtype, self.linear)?;
                self.lines.push(format!(
                    "{indent}const {} {name} = {value};",
                    expression_type(dtype)
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
                self.lines.push(format!(
                    "{indent}{} {name} = ({})0;",
                    expression_type(dtype),
                    expression_type(dtype)
                ));
                if let Some(id) = self.guard_ids.get(node).copied() {
                    let operation = super::GuardedIntegerOp::from_binary(*op).ok_or_else(|| {
                        OpenClError::InvalidBinding("guard opcode mismatch".into())
                    })?;
                    let invalid = invalid_expression(operation, dtype, &rhs);
                    let safe = guarded_value(operation, dtype, &lhs, &rhs)?;
                    let count = self.transaction.guard_count();
                    self.lines.push(format!("{indent}if (rg_ok) {{"));
                    self.lines.push(format!("{indent}  if ({invalid}) {{"));
                    self.lines.push(format!(
                        "{indent}    atomic_min(rg_status, (uint)((uint){} * {count}u + {id}u));",
                        self.linear
                    ));
                    self.lines.push(format!("{indent}    rg_ok = (uchar)0u;"));
                    self.lines.push(format!("{indent}  }} else {{"));
                    self.lines.push(format!("{indent}    {name} = {safe};"));
                    self.lines.push(format!("{indent}  }}"));
                    self.lines.push(format!("{indent}}}"));
                } else {
                    let value = self.pure_expression(node, vec![lhs, rhs])?;
                    self.lines
                        .push(format!("{indent}if (rg_ok) {name} = {value};"));
                }
            }
            Operation::GraphCompare(_) => {
                let lhs = self.node(&node.sources()[0], indent)?;
                let rhs = self.node(&node.sources()[1], indent)?;
                let value = self.pure_expression(node, vec![lhs, rhs])?;
                self.assign_if_ok(indent, DType::Bool, &name, &value);
            }
            Operation::GraphLogical(op) => {
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
            Operation::Ternary(crate::uop::Ternary::Where) => {
                let condition = self.node(&node.sources()[0], indent)?;
                self.lines.push(format!(
                    "{indent}{} {name} = ({})0;",
                    expression_type(dtype),
                    expression_type(dtype)
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
                return Err(OpenClError::Unsupported(format!(
                    "transactional expression {other:?}"
                )));
            }
        }
        Ok(name)
    }

    fn pure_expression(&self, node: &UOp, sources: Vec<String>) -> Result<String, OpenClError> {
        let instruction = project_scalar_lane(node, &sources)
            .map_err(OpenClError::Unsupported)?
            .ok_or_else(|| {
                OpenClError::Unsupported(format!("transactional expression {:?}", node.operation()))
            })?;
        emit_scalar_lane(
            &OpenClScalarDialect {
                capabilities: self.capabilities,
            },
            &instruction,
        )
        .map_err(OpenClError::Unsupported)
    }

    fn assign_if_ok(&mut self, indent: &str, dtype: DType, name: &str, value: &str) {
        self.lines.push(format!(
            "{indent}{} {name} = ({})0;",
            expression_type(dtype),
            expression_type(dtype)
        ));
        self.lines
            .push(format!("{indent}if (rg_ok) {name} = {value};"));
    }

    fn value_name(&mut self) -> String {
        let name = format!("rg_v{}", self.next_value);
        self.next_value += 1;
        name
    }

    fn load(&self, node: &UOp, dtype: DType, linear: &str) -> Result<String, OpenClError> {
        let index = node
            .sources()
            .first()
            .ok_or_else(|| OpenClError::Unsupported("load has no index".into()))?;
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
                return Err(OpenClError::Unsupported(
                    "load requires checked static indexing".into(),
                ));
            }
        };
        let position = self
            .ids
            .get(&buffer)
            .ok_or_else(|| OpenClError::InvalidBinding("load absent from ABI".into()))?;
        let logical = broadcast_offset(input_shape, output_shape, linear)?;
        let offset = match view {
            Some(view) => OpenClViewAccess::new(view, dtype)?.expression(logical),
            None => logical,
        };
        let raw = format!("b{position}[{offset}]");
        Ok(narrow::decode(dtype, &raw).unwrap_or(raw))
    }
}

fn expression_type(dtype: DType) -> &'static str {
    if narrow::is_narrow(dtype) {
        "double"
    } else {
        cl_type(dtype)
    }
}

fn scalar_literal(node: &UOp, dtype: DType) -> Result<String, OpenClError> {
    match node.operation() {
        Operation::Const(LiteralValue::Scalar {
            dtype: actual,
            bits,
        }) if *actual == DType::Bool && dtype == DType::Bool && *bits <= 1 => {
            Ok(format!("((uchar){bits}u)"))
        }
        Operation::Const(LiteralValue::Scalar {
            dtype: actual,
            bits,
        }) if *actual == DType::I32 && dtype == DType::I32 => {
            Ok(format!("as_int((uint)0x{:08x}u)", *bits as u32))
        }
        Operation::Const(LiteralValue::Scalar {
            dtype: actual,
            bits,
        }) if *actual == DType::U32 && dtype == DType::U32 => {
            Ok(format!("((uint)0x{:08x}u)", *bits as u32))
        }
        Operation::Const(LiteralValue::Scalar {
            dtype: actual,
            bits,
        }) if *actual == DType::I64 && dtype == DType::I64 => {
            Ok(format!("as_long((ulong)0x{bits:016x}ul)"))
        }
        Operation::Const(LiteralValue::Scalar {
            dtype: actual,
            bits,
        }) if *actual == DType::U64 && dtype == DType::U64 => {
            Ok(format!("((ulong)0x{bits:016x}ul)"))
        }
        _ => Err(OpenClError::Unsupported(
            "transactional scalar literal/type mismatch".into(),
        )),
    }
}

fn invalid_expression(op: super::GuardedIntegerOp, dtype: DType, rhs: &str) -> String {
    if op.is_shift() {
        if matches!(dtype, DType::I32 | DType::I64) {
            format!("(({rhs}) < 0 || (ulong)({rhs}) >= {}ul)", dtype.bits())
        } else {
            format!("((ulong)({rhs}) >= {}ul)", dtype.bits())
        }
    } else {
        format!("(({rhs}) == ({})0)", cl_type(dtype))
    }
}
