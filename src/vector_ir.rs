//! Explicit backend-neutral vector instructions over assigned register spaces.
use crate::{LinearInstKind, LinearKernel, MemorySpacePlan, RegisterBinding};
use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum VectorOperand {
    Register { physical: u32, vector: bool },
    Global { buffer: u64 },
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
        if self.lanes == 0
            || self
                .instructions
                .iter()
                .any(|i| i.lanes != self.lanes || i.mask.len() != self.lanes as usize)
        {
            return Err(VectorIrError::InvalidMask {
                instruction: self.instructions.first().map_or(0, |i| i.index),
            });
        }
        let register_set = spaces
            .registers
            .iter()
            .map(|r| {
                (
                    r.physical_reg,
                    matches!(r.space, crate::MemorySpace::RegisterVector),
                )
            })
            .collect::<Vec<_>>();
        for inst in &self.instructions {
            for op in inst.inputs.iter().chain(inst.dst.iter()) {
                if let VectorOperand::Register { physical, vector } = op
                    && !register_set.contains(&(*physical, *vector))
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
fn physical(
    binding: Option<&RegisterBinding>,
    virtual_reg: u32,
) -> Result<VectorOperand, VectorIrError> {
    let b = binding.ok_or(VectorIrError::MissingRegister(virtual_reg))?;
    Ok(VectorOperand::Register {
        physical: b.physical_reg,
        vector: matches!(b.space, crate::MemorySpace::RegisterVector),
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
    }
}
