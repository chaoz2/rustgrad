//! Explicit backend-neutral vector instructions over assigned register spaces.
use crate::{
    LaneInstruction, LaneProgramInstruction, LinearKernel, MemorySpacePlan, RegisterBinding,
};
use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct VectorOperand {
    pub physical: u32,
    pub vector: bool,
    pub dtype: crate::DType,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VectorProgram {
    pub instructions: Vec<LaneProgramInstruction<VectorOperand>>,
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
            let view = inst.instruction.view();
            let ty = view.result_type().map(|ty| ty.scalar);
            // B2 physically represents narrow lanes as raw u16 values, while
            // the source-correct scalar renderer must decode to float for
            // every arithmetic/select operation and encode again at storage
            // boundaries. Reject every instruction that consumes or produces
            // a narrow register, including Load/Store whose payload type can
            // be absent, until B2 has a tagged half-vector ABI.
            let narrow_register = |operand: &VectorOperand| {
                matches!(operand.dtype, crate::DType::F16 | crate::DType::BF16)
            };
            let float8_register = |operand: &VectorOperand| operand.dtype.is_float8();
            if matches!(ty, Some(crate::DType::F16 | crate::DType::BF16))
                || view.output().is_some_and(narrow_register)
                || view.inputs().any(narrow_register)
            {
                return Err(VectorIrError::Unsupported(
                    "portable narrow vector ABI needs tagged float lanes".into(),
                ));
            }
            // Float8 registers have the same raw-storage problem, but their
            // four format tags also change numeric decoding. Keep decoded
            // comparisons and raw-byte Select on the legacy scalar-per-lane
            // renderer even when Load/Store payload types are absent.
            if ty.is_some_and(crate::DType::is_float8)
                || view.output().is_some_and(float8_register)
                || view.inputs().any(float8_register)
            {
                return Err(VectorIrError::Unsupported(
                    "portable Float8 vector ABI needs tagged decoded lanes".into(),
                ));
            }
            let structural = matches!(
                &inst.instruction,
                LaneInstruction::Constant { .. }
                    | LaneInstruction::Address { .. }
                    | LaneInstruction::Range { .. }
                    | LaneInstruction::Index { .. }
            );
            if !structural
                && ty.is_some_and(|ty| {
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
                })
            {
                return Err(VectorIrError::Unsupported(format!("portable dtype {ty:?}")));
            }
            if matches!(
                &inst.instruction,
                LaneInstruction::Index {
                    output: crate::IndexRef {
                        value: crate::IndexValue::View { .. }
                            | crate::IndexValue::Buffer {
                                addressing: crate::IndexAddressing::Projected
                                    | crate::IndexAddressing::Predicated,
                                ..
                            },
                        ..
                    },
                    ..
                }
            ) {
                return Err(VectorIrError::Unsupported(
                    "portable vector instruction ABI does not encode view or projected offsets"
                        .into(),
                ));
            }
            match &inst.instruction {
                LaneInstruction::GraphUnary {
                    op: crate::UnaryOp::Neg | crate::UnaryOp::Abs,
                    ..
                } => {}
                LaneInstruction::CoreUnary { .. }
                | LaneInstruction::GraphUnary { .. }
                | LaneInstruction::LogicalNot { .. } => {
                    return Err(VectorIrError::Unsupported("portable unary opcode".into()));
                }
                LaneInstruction::GraphBinary {
                    op:
                        crate::BinaryOp::Add
                        | crate::BinaryOp::Sub
                        | crate::BinaryOp::Mul
                        | crate::BinaryOp::Div
                        | crate::BinaryOp::FloorDiv
                        | crate::BinaryOp::TruncDiv
                        | crate::BinaryOp::Mod
                        | crate::BinaryOp::FMod
                        | crate::BinaryOp::Shl
                        | crate::BinaryOp::Shr,
                    ..
                } => {}
                LaneInstruction::CoreBinary { .. }
                | LaneInstruction::CoreEq { .. }
                | LaneInstruction::CoreLt { .. }
                | LaneInstruction::CoreLe { .. }
                | LaneInstruction::GraphBinary { .. }
                | LaneInstruction::LogicalAnd { .. }
                | LaneInstruction::LogicalOr { .. } => {
                    return Err(VectorIrError::Unsupported("portable binary opcode".into()));
                }
                _ => {}
            }
            if matches!(&inst.instruction, LaneInstruction::Bitcast { .. }) {
                return Err(VectorIrError::Unsupported("portable bitcast opcode".into()));
            }
            if matches!(
                &inst.instruction,
                LaneInstruction::Address { output }
                    if output.value.space != crate::AddressSpace::Global
            ) {
                return Err(VectorIrError::Unsupported(
                    "portable lane address requires global memory".into(),
                ));
            }
            if let LaneInstruction::GraphBinary { op, .. } = &inst.instruction {
                if ty.is_some_and(crate::DType::is_float)
                    && !matches!(
                        op,
                        crate::BinaryOp::Add | crate::BinaryOp::Sub | crate::BinaryOp::Mul
                    )
                {
                    return Err(VectorIrError::Unsupported(
                        "portable floating binary opcode".into(),
                    ));
                }
                if ty == Some(crate::DType::Bool)
                    && !matches!(
                        op,
                        crate::BinaryOp::Add
                            | crate::BinaryOp::Sub
                            | crate::BinaryOp::Mul
                            | crate::BinaryOp::Div
                    )
                {
                    return Err(VectorIrError::Unsupported(
                        "portable boolean binary opcode".into(),
                    ));
                }
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
        let instructions = if enabled {
            linear
                .program
                .instructions
                .iter()
                .map(|inst| {
                    let instruction = inst
                        .instruction
                        .map_operands(|reg| physical(regs.get(reg).copied(), *reg))?;
                    instruction
                        .validate()
                        .map_err(|error| VectorIrError::Unsupported(error.to_string()))?;
                    Ok(LaneProgramInstruction {
                        index: inst.index,
                        instruction,
                    })
                })
                .collect::<Result<Vec<_>, VectorIrError>>()?
        } else {
            Vec::new()
        };
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
        let key = |operand: &VectorOperand| (operand.physical, operand.vector, operand.dtype);
        for inst in &self.instructions {
            let view = inst.instruction.view();
            let output = view.output().zip(view.result_type());
            for (_, expected) in view.typed_inputs().chain(output) {
                let effective_lanes = if expected.lanes == 1 {
                    self.lanes
                } else {
                    expected.lanes
                };
                if effective_lanes != self.lanes {
                    return Err(VectorIrError::Unsupported(format!(
                        "instruction {} has inconsistent lane width",
                        inst.index
                    )));
                }
            }
            if let Some(buffer) = view.buffer {
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
        crate::linearize::validate_lane_sequence(
            &self.instructions,
            &std::collections::BTreeSet::new(),
            key,
            |operand, instruction, descriptor| {
                let expected_dtype = match descriptor.ty().scalar {
                    crate::DType::F16 | crate::DType::BF16 => crate::DType::F32,
                    dtype => dtype,
                };
                operand.dtype == expected_dtype
                    && spaces.registers.iter().any(|binding| {
                        binding.physical_reg == operand.physical
                            && matches!(binding.space, crate::MemorySpace::RegisterVector)
                                == operand.vector
                            && binding.dtype == operand.dtype
                            && binding.start <= instruction
                            && instruction <= binding.end
                    })
            },
            |_, _| true,
        )
        .map_err(VectorIrError::Unsupported)
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
fn physical(
    binding: Option<&RegisterBinding>,
    virtual_reg: u32,
) -> Result<VectorOperand, VectorIrError> {
    let b = binding.ok_or(VectorIrError::MissingRegister(virtual_reg))?;
    Ok(VectorOperand {
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
        let mut malformed_tail = VectorProgram::from_linear(&l, &s).unwrap();
        malformed_tail.tail_elements = usize::from(malformed_tail.lanes);
        assert!(matches!(
            malformed_tail.b1_eligibility(),
            Err(VectorIrError::InvalidMask { .. })
        ));
        let mut bad_register = p;
        for instruction in &mut bad_register.instructions {
            let LaneInstruction::Address { output } = &mut instruction.instruction else {
                continue;
            };
            output.register.physical = u32::MAX;
            break;
        }
        assert!(matches!(
            bad_register.validate(&s),
            Err(VectorIrError::Unsupported(reason)) if reason.contains("no live compatible binding")
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

    #[test]
    fn vector_validation_tracks_reaching_descriptor_and_lane_width() {
        let mut graph = Graph::new();
        let input = graph.input("input", Shape::from([4]));
        let output = graph.square(input).unwrap();
        let linear =
            crate::LinearKernel::from_uop(&lower_graph_elementwise(&graph, output).unwrap())
                .unwrap();
        let spaces = MemorySpacePlan::from_linear(&linear).unwrap();
        let program = VectorProgram::from_linear(&linear, &spaces).unwrap();

        let mut wrong_address = program.clone();
        let mut changed = false;
        for instruction in &mut wrong_address.instructions {
            let LaneInstruction::Index { address, .. } = &mut instruction.instruction else {
                continue;
            };
            address.value.name.push_str("_mismatch");
            changed = true;
            break;
        }
        assert!(changed);
        assert!(matches!(
            wrong_address.validate(&spaces),
            Err(VectorIrError::Unsupported(reason)) if reason.contains("descriptor mismatch")
        ));

        let mut wrong_lanes = program.clone();
        let mut changed = false;
        for instruction in &mut wrong_lanes.instructions {
            let LaneInstruction::Constant { output, .. } = &mut instruction.instruction else {
                continue;
            };
            output.ty.lanes = 2;
            changed = true;
            break;
        }
        assert!(changed);
        assert!(matches!(
            wrong_lanes.validate(&spaces),
            Err(VectorIrError::Unsupported(reason)) if reason.contains("lane width")
        ));

        let mut expired_spaces = spaces.clone();
        let (instruction_index, operand) = program
            .instructions
            .iter()
            .find_map(|instruction| {
                (instruction.index > 0)
                    .then(|| {
                        instruction
                            .instruction
                            .view()
                            .inputs()
                            .next()
                            .map(|operand| (instruction.index, operand.clone()))
                    })
                    .flatten()
            })
            .unwrap();
        for binding in &mut expired_spaces.registers {
            if binding.physical_reg == operand.physical
                && matches!(binding.space, crate::MemorySpace::RegisterVector) == operand.vector
                && binding.dtype == operand.dtype
                && binding.start <= instruction_index
                && instruction_index <= binding.end
            {
                binding.end = instruction_index - 1;
            }
        }
        assert!(matches!(
            program.validate(&expired_spaces),
            Err(VectorIrError::Unsupported(reason)) if reason.contains("no live compatible binding")
        ));
    }

    #[test]
    fn bitcast_and_reduction_programs_fail_closed_before_vector_emission() {
        let mut cast_graph = Graph::new();
        let input = cast_graph.input("input", Shape::from([4]));
        let output = cast_graph.cast(input, crate::DType::I32).unwrap();
        let linear =
            crate::LinearKernel::from_uop(&lower_graph_elementwise(&cast_graph, output).unwrap())
                .unwrap();
        let spaces = MemorySpacePlan::from_linear(&linear).unwrap();
        let original = VectorProgram::from_linear(&linear, &spaces).unwrap();
        let mut local_address = original.clone();
        let mut changed = false;
        for instruction in &mut local_address.instructions {
            let LaneInstruction::Address { output } = &mut instruction.instruction else {
                continue;
            };
            output.value.space = crate::AddressSpace::Local;
            changed = true;
            break;
        }
        assert!(changed);
        assert!(matches!(
            local_address.b1_eligibility(),
            Err(VectorIrError::Unsupported(reason)) if reason.contains("global memory")
        ));

        let mut bitcast = original;
        let mut changed = false;
        for instruction in &mut bitcast.instructions {
            let LaneInstruction::Cast { output, input } = &instruction.instruction else {
                continue;
            };
            instruction.instruction = LaneInstruction::Bitcast {
                output: output.clone(),
                input: input.clone(),
            };
            changed = true;
            break;
        }
        assert!(changed);
        assert!(matches!(
            bitcast.b1_eligibility(),
            Err(VectorIrError::Unsupported(reason)) if reason == "portable bitcast opcode"
        ));

        let mut reduction_graph = Graph::new();
        let input = reduction_graph.input("input", Shape::from([2, 3]));
        let output = reduction_graph
            .reduce(input, crate::ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        let scheduled = crate::schedule(&reduction_graph, output).unwrap();
        let kernel = &scheduled.items.last().unwrap().kernel;
        let linear = crate::LinearKernel::from_uop(kernel).unwrap();
        assert!(!linear.enabled);
        assert!(
            linear
                .program
                .unsupported_operations
                .iter()
                .any(|operation| matches!(
                    operation.operation,
                    crate::Operation::ReduceInit(_)
                        | crate::Operation::ReduceAccumulate
                        | crate::Operation::ReduceFinalize
                ))
        );
        let spaces = MemorySpacePlan::from_linear(&linear).unwrap();
        let vector = VectorProgram::from_linear(&linear, &spaces).unwrap();
        assert!(vector.instructions.is_empty());
        assert!(matches!(
            vector.b1_eligibility(),
            Err(VectorIrError::Unsupported(_))
        ));
    }
}
