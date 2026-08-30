//! Explicit backend-neutral vector instructions over assigned register spaces.
use crate::{LinearInstKind, LinearKernel, LinearPayload, MemorySpacePlan, RegisterBinding};
use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum VectorOperand {
    Register {
        physical: u32,
        vector: bool,
        dtype: crate::DType,
    },
    Global {
        buffer: u64,
    },
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum VectorInstKind {
    Splat,
    Address,
    Index,
    Load { buffer: u64 },
    Cast,
    Unary,
    Binary,
    Compare,
    Select,
    Store { buffer: u64 },
    Control,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct VectorInst {
    pub index: u32,
    pub dst: Option<VectorOperand>,
    pub inputs: Vec<VectorOperand>,
    pub kind: VectorInstKind,
    pub lanes: u16,
    pub mask: Vec<bool>,
    pub payload: LinearPayload,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VectorProgram {
    pub instructions: Vec<VectorInst>,
    pub lanes: u16,
    pub main_elements: usize,
    pub tail_elements: usize,
    pub enabled: bool,
    pub fallback_reason: Option<String>,
    pub cache_key: u64,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VectorIrError {
    MissingRegister(u32),
    InvalidMask { instruction: u32 },
    InvalidRegisterClass(u32),
    InvalidAddress(u64),
    Unsupported(String),
}
impl fmt::Display for VectorIrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vector IR error: {self:?}")
    }
}
impl std::error::Error for VectorIrError {}

impl VectorProgram {
    /// Validates the lane control shared by every instruction before a
    /// portable renderer can derive its main and tail loops. A tail has at
    /// most one partial vector; accepting a larger or inconsistent mask could
    /// make the generated tail loop address elements beyond the output domain.
    fn validate_lane_control(&self) -> Result<(), VectorIrError> {
        let invalid = || VectorIrError::InvalidMask {
            instruction: self
                .instructions
                .first()
                .map_or(0, |instruction| instruction.index),
        };
        if self.lanes == 0 {
            return Err(invalid());
        }
        if !self.enabled {
            return (self.main_elements == 0).then_some(()).ok_or_else(invalid);
        }
        let lanes = usize::from(self.lanes);
        if self.main_elements % lanes != 0 || self.tail_elements >= lanes {
            return Err(invalid());
        }
        let expected_mask = (0..lanes)
            .map(|lane| lane < self.tail_elements)
            .collect::<Vec<_>>();
        for instruction in &self.instructions {
            if instruction.lanes != self.lanes || instruction.mask != expected_mask {
                return Err(VectorIrError::InvalidMask {
                    instruction: instruction.index,
                });
            }
        }
        Ok(())
    }

    /// The portable VectorProgram emitter deliberately has a small, auditable
    /// semantic surface. Other programs retain the scalar renderer path.
    pub fn b1_eligibility(&self) -> Result<(), VectorIrError> {
        self.validate_lane_control()?;
        if !self.enabled {
            return Err(VectorIrError::Unsupported(
                self.fallback_reason
                    .clone()
                    .unwrap_or_else(|| "scalar vector policy".into()),
            ));
        }
        for inst in &self.instructions {
            let ty = inst.payload.ty.map(|ty| ty.scalar);
            // B2 physically represents narrow lanes as raw u16 values, while
            // the source-correct scalar renderer must decode to float for
            // every arithmetic/select operation and encode again at storage
            // boundaries. Reject every instruction that consumes or produces
            // a narrow register, including Load/Store whose payload type can
            // be absent, until B2 has a tagged half-vector ABI.
            let narrow_register = |operand: &VectorOperand| {
                matches!(
                    operand,
                    VectorOperand::Register {
                        dtype: crate::DType::F16 | crate::DType::BF16,
                        ..
                    }
                )
            };
            if matches!(ty, Some(crate::DType::F16 | crate::DType::BF16))
                || inst.dst.as_ref().is_some_and(narrow_register)
                || inst.inputs.iter().any(narrow_register)
            {
                return Err(VectorIrError::Unsupported(
                    "portable narrow vector ABI needs tagged float lanes".into(),
                ));
            }
            if !matches!(
                inst.kind,
                VectorInstKind::Splat
                    | VectorInstKind::Address
                    | VectorInstKind::Index
                    | VectorInstKind::Control
            ) && ty.is_some_and(|ty| {
                !matches!(
                    ty,
                    crate::DType::Bool
                        | crate::DType::I8
                        | crate::DType::I16
                        | crate::DType::I32
                        | crate::DType::I64
                        | crate::DType::U8
                        | crate::DType::U16
                        | crate::DType::U32
                        | crate::DType::U64
                        | crate::DType::F16
                        | crate::DType::BF16
                        | crate::DType::F32
                        | crate::DType::F64
                )
            }) {
                return Err(VectorIrError::Unsupported(format!("portable dtype {ty:?}")));
            }
            match inst.kind {
                VectorInstKind::Splat
                | VectorInstKind::Address
                | VectorInstKind::Index
                | VectorInstKind::Load { .. }
                | VectorInstKind::Cast
                | VectorInstKind::Unary
                | VectorInstKind::Binary
                | VectorInstKind::Compare
                | VectorInstKind::Select
                | VectorInstKind::Store { .. }
                | VectorInstKind::Control => {}
            }
            if matches!(
                inst.payload.uop_kind,
                crate::UOpKind::ReduceInit
                    | crate::UOpKind::ReduceAccumulate
                    | crate::UOpKind::ReduceFinalize
                    | crate::UOpKind::Barrier
            ) {
                return Err(VectorIrError::Unsupported(
                    "portable effects/reductions".into(),
                ));
            }
            if matches!(inst.payload.arg, crate::UArg::ViewBufferIndex { .. }) {
                return Err(VectorIrError::Unsupported(
                    "portable vector instruction ABI does not encode affine view offsets".into(),
                ));
            }
            // The physical B1 emitter intentionally supports a narrower
            // opcode set than the generic VectorInstKind tags. Keep an
            // otherwise valid logical/core ALU program on the fallback path
            // instead of letting it fail after vector planning.
            if matches!(inst.kind, VectorInstKind::Unary)
                && !matches!(
                    inst.payload.uop_kind,
                    crate::UOpKind::GraphUnary(crate::UnaryOp::Neg | crate::UnaryOp::Abs)
                )
            {
                return Err(VectorIrError::Unsupported("portable unary opcode".into()));
            }
            if matches!(inst.kind, VectorInstKind::Binary)
                && !matches!(
                    inst.payload.uop_kind,
                    crate::UOpKind::GraphBinary(
                        crate::BinaryOp::Add
                            | crate::BinaryOp::Sub
                            | crate::BinaryOp::Mul
                            | crate::BinaryOp::Div
                            | crate::BinaryOp::FloorDiv
                            | crate::BinaryOp::TruncDiv
                            | crate::BinaryOp::Mod
                            | crate::BinaryOp::FMod
                            | crate::BinaryOp::Shl
                            | crate::BinaryOp::Shr
                    )
                )
            {
                return Err(VectorIrError::Unsupported("portable binary opcode".into()));
            }
        }
        Ok(())
    }
    pub fn from_linear(
        linear: &LinearKernel,
        spaces: &MemorySpacePlan,
    ) -> Result<Self, VectorIrError> {
        spaces
            .validate()
            .map_err(|e| VectorIrError::Unsupported(e.to_string()))?;
        let regs = spaces
            .registers
            .iter()
            .map(|r| (r.virtual_reg, r))
            .collect::<BTreeMap<_, _>>();
        let enabled = linear.enabled;
        let fallback_reason = (!enabled).then(|| linear.reason.clone());
        let lanes = if enabled { linear.lanes as u16 } else { 1 };
        let mask = if enabled {
            linear.tail_mask.clone()
        } else {
            vec![true]
        };
        let mut instructions = Vec::with_capacity(linear.program.instructions.len());
        for inst in &linear.program.instructions {
            let operand = |reg: u32| physical(regs.get(&reg).copied(), reg);
            let inputs = inst
                .inputs
                .iter()
                .map(|r| operand(*r))
                .collect::<Result<Vec<_>, _>>()?;
            let kind = match &inst.kind {
                LinearInstKind::Constant => VectorInstKind::Splat,
                LinearInstKind::Address => VectorInstKind::Address,
                LinearInstKind::Index => VectorInstKind::Index,
                LinearInstKind::Load { buffer } => VectorInstKind::Load { buffer: *buffer },
                LinearInstKind::Cast => VectorInstKind::Cast,
                LinearInstKind::Unary => VectorInstKind::Unary,
                LinearInstKind::Binary => VectorInstKind::Binary,
                LinearInstKind::Compare => VectorInstKind::Compare,
                LinearInstKind::Select => VectorInstKind::Select,
                LinearInstKind::Store { buffer } => VectorInstKind::Store { buffer: *buffer },
                LinearInstKind::Other(_) => VectorInstKind::Control,
            };
            let dst = inst.dst.map(operand).transpose()?;
            instructions.push(VectorInst {
                index: inst.index,
                dst,
                inputs,
                kind,
                lanes,
                mask: mask.clone(),
                payload: inst.payload.clone(),
            });
        }
        let mut out = Self {
            instructions,
            lanes,
            main_elements: if enabled { linear.vector_main } else { 0 },
            tail_elements: linear.scalar_tail,
            enabled,
            fallback_reason,
            cache_key: 0,
        };
        out.rekey();
        out.validate(spaces)?;
        Ok(out)
    }
    pub fn validate(&self, spaces: &MemorySpacePlan) -> Result<(), VectorIrError> {
        self.validate_lane_control()?;
        let register_set = spaces
            .registers
            .iter()
            .map(|r| {
                (
                    r.physical_reg,
                    matches!(r.space, crate::MemorySpace::RegisterVector),
                    r.dtype,
                )
            })
            .collect::<Vec<_>>();
        for inst in &self.instructions {
            if !payload_matches(&inst.kind, &inst.payload.uop_kind) {
                return Err(VectorIrError::Unsupported(
                    "instruction/payload kind mismatch".into(),
                ));
            }
            for op in inst.inputs.iter().chain(inst.dst.iter()) {
                if let VectorOperand::Register {
                    physical,
                    vector,
                    dtype,
                } = op
                    && !register_set.contains(&(*physical, *vector, *dtype))
                {
                    return Err(VectorIrError::InvalidRegisterClass(*physical));
                }
            }
            if let VectorInstKind::Load { buffer } | VectorInstKind::Store { buffer } = inst.kind {
                let access = spaces
                    .globals
                    .iter()
                    .find(|g| g.buffer == buffer)
                    .ok_or(VectorIrError::InvalidAddress(buffer))?;
                if access.alignment == 0 || access.byte_offset % access.alignment != 0 {
                    return Err(VectorIrError::InvalidAddress(buffer));
                }
            }
        }
        Ok(())
    }
    fn rekey(&mut self) {
        let mut h = DefaultHasher::new();
        self.instructions.hash(&mut h);
        self.lanes.hash(&mut h);
        self.main_elements.hash(&mut h);
        self.tail_elements.hash(&mut h);
        self.enabled.hash(&mut h);
        self.fallback_reason.hash(&mut h);
        self.cache_key = h.finish();
    }
}
fn payload_matches(kind: &VectorInstKind, payload: &crate::UOpKind) -> bool {
    matches!(
        (kind, payload),
        (
            VectorInstKind::Splat,
            crate::UOpKind::Const | crate::UOpKind::VConst
        ) | (
            VectorInstKind::Address,
            crate::UOpKind::DefineGlobal
                | crate::UOpKind::DefineLocal
                | crate::UOpKind::DefineRegister
        ) | (VectorInstKind::Index, crate::UOpKind::Index)
            | (VectorInstKind::Load { .. }, crate::UOpKind::Load)
            | (
                VectorInstKind::Cast,
                crate::UOpKind::Cast | crate::UOpKind::Bitcast
            )
            | (
                VectorInstKind::Unary,
                crate::UOpKind::Unary(_) | crate::UOpKind::GraphUnary(_)
            )
            | (
                VectorInstKind::Binary,
                crate::UOpKind::Binary(_)
                    | crate::UOpKind::GraphBinary(_)
                    | crate::UOpKind::GraphLogical(_)
            )
            | (VectorInstKind::Compare, crate::UOpKind::GraphCompare(_))
            | (VectorInstKind::Select, crate::UOpKind::Ternary(_))
            | (VectorInstKind::Store { .. }, crate::UOpKind::Store)
            | (VectorInstKind::Control, _)
    )
}
fn physical(
    binding: Option<&RegisterBinding>,
    virtual_reg: u32,
) -> Result<VectorOperand, VectorIrError> {
    let b = binding.ok_or(VectorIrError::MissingRegister(virtual_reg))?;
    Ok(VectorOperand::Register {
        physical: b.physical_reg,
        vector: matches!(b.space, crate::MemorySpace::RegisterVector),
        dtype: b.dtype,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Shape, lower_graph_elementwise};
    #[test]
    fn vector_program_is_stable_and_tailed() {
        let mut g = Graph::new();
        let x = g.input("x", Shape::from([5]));
        let y = g.square(x).unwrap();
        let l = crate::LinearKernel::from_uop(&lower_graph_elementwise(&g, y).unwrap()).unwrap();
        let s = MemorySpacePlan::from_linear(&l).unwrap();
        let p = VectorProgram::from_linear(&l, &s).unwrap();
        assert!(p.enabled);
        assert_eq!((p.lanes, p.main_elements, p.tail_elements), (4, 4, 1));
        assert_eq!(
            p.cache_key,
            VectorProgram::from_linear(&l, &s).unwrap().cache_key
        );
        let mut malformed = p.clone();
        malformed.instructions[0].mask.pop();
        assert!(matches!(
            malformed.validate(&s),
            Err(VectorIrError::InvalidMask { .. })
        ));
        let mut malformed_tail = VectorProgram::from_linear(&l, &s).unwrap();
        malformed_tail.tail_elements = usize::from(malformed_tail.lanes);
        assert!(matches!(
            malformed_tail.b1_eligibility(),
            Err(VectorIrError::InvalidMask { .. })
        ));
        let mut malformed_tail_mask = VectorProgram::from_linear(&l, &s).unwrap();
        malformed_tail_mask.instructions[0].mask.fill(false);
        assert!(matches!(
            malformed_tail_mask.b1_eligibility(),
            Err(VectorIrError::InvalidMask { .. })
        ));
        let mut bad_register = p;
        for instruction in &mut bad_register.instructions {
            if let Some(VectorOperand::Register { physical, .. }) = instruction.dst.as_mut() {
                *physical = u32::MAX;
                break;
            }
        }
        assert!(matches!(
            bad_register.validate(&s),
            Err(VectorIrError::InvalidRegisterClass(u32::MAX))
        ));
    }

    #[test]
    fn b1_eligibility_rejects_logical_programs_before_late_rendering() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", Shape::from([5]), crate::DType::Bool);
        let rhs = graph.input_dtype("rhs", Shape::from([5]), crate::DType::Bool);
        let output = graph.logical_and(lhs, rhs).unwrap();
        let linear =
            crate::LinearKernel::from_uop(&lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
        let spaces = MemorySpacePlan::from_linear(&linear).unwrap();
        let program = VectorProgram::from_linear(&linear, &spaces).unwrap();
        assert!(program.enabled);
        assert!(matches!(
            program.b1_eligibility(),
            Err(VectorIrError::Unsupported(reason)) if reason == "portable binary opcode"
        ));
    }
}
