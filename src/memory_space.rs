//! Backend-neutral late memory-space planning.
//!
//! This is deliberately a validation and identity boundary.  It makes register
//! assignments and any future private/shared allocations inspectable without
//! claiming that portable CPU elementwise kernels benefit from workgroup memory.
use crate::{LinearKernel, RegisterClass};
use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MemorySpace {
    Global,
    RegisterScalar,
    RegisterVector,
    Private,
    Shared,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RegisterBinding {
    pub virtual_reg: u32,
    pub physical_reg: u32,
    pub space: MemorySpace,
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GlobalAccess {
    pub buffer: u64,
    pub bytes: usize,
    pub byte_offset: usize,
    pub alignment: usize,
    pub mutable: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SpaceAllocation {
    pub id: u32,
    pub space: MemorySpace,
    pub bytes: usize,
    pub alignment: usize,
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum PromotionDecision {
    NotPromoted { reason: String },
    Shared { allocation: SpaceAllocation },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BarrierScope {
    Workgroup,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BarrierPoint {
    pub instruction: u32,
    pub scope: BarrierScope,
    /// A future GPU linearizer must prove this independently of per-lane masks.
    pub uniform_control: bool,
    pub initializes: Vec<u32>,
    pub consumes: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemorySpacePlan {
    pub registers: Vec<RegisterBinding>,
    pub globals: Vec<GlobalAccess>,
    pub private: Vec<SpaceAllocation>,
    pub shared: Vec<SpaceAllocation>,
    pub barriers: Vec<BarrierPoint>,
    pub promotions: Vec<PromotionDecision>,
    pub cache_key: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemorySpaceError {
    Overflow,
    MissingInterval(u32),
    DuplicateRegister {
        space: MemorySpace,
        physical: u32,
    },
    OverlappingAlias {
        space: MemorySpace,
        first: u32,
        second: u32,
    },
    InvalidAllocation {
        id: u32,
        space: MemorySpace,
    },
    InvalidAlignment {
        id: u32,
        alignment: usize,
    },
    DivergentBarrier(u32),
    BarrierUseBeforeInitialize {
        barrier: u32,
        allocation: u32,
    },
}
impl fmt::Display for MemorySpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "memory space plan error: {self:?}")
    }
}
impl std::error::Error for MemorySpaceError {}

impl MemorySpacePlan {
    pub fn from_linear(linear: &LinearKernel) -> Result<Self, MemorySpaceError> {
        let intervals = linear
            .program
            .intervals
            .iter()
            .map(|i| (i.virtual_reg, i))
            .collect::<BTreeMap<_, _>>();
        let mut registers = Vec::with_capacity(linear.program.assignments.len());
        for assignment in &linear.program.assignments {
            let interval = intervals
                .get(&assignment.virtual_reg)
                .ok_or(MemorySpaceError::MissingInterval(assignment.virtual_reg))?;
            registers.push(RegisterBinding {
                virtual_reg: assignment.virtual_reg,
                physical_reg: assignment.physical_reg,
                space: match assignment.class {
                    RegisterClass::Scalar => MemorySpace::RegisterScalar,
                    RegisterClass::Vector => MemorySpace::RegisterVector,
                },
                start: interval.start,
                end: interval.end,
            });
        }
        registers.sort_by_key(|r| (r.space, r.physical_reg, r.virtual_reg));
        let mut globals = Vec::with_capacity(linear.buffers.len());
        for buffer in &linear.buffers {
            globals.push(GlobalAccess {
                buffer: buffer.buffer,
                bytes: buffer
                    .elements
                    .checked_mul(buffer.dtype.itemsize())
                    .ok_or(MemorySpaceError::Overflow)?,
                byte_offset: buffer.byte_offset,
                alignment: buffer.alignment,
                mutable: buffer.mutable,
            });
        }
        globals.sort_by_key(|g| g.buffer);
        let mut plan = Self {
            registers,
            globals,
            private: Vec::new(),
            shared: Vec::new(),
            barriers: Vec::new(),
            promotions: vec![PromotionDecision::NotPromoted {
                reason: "pure elementwise program has no statically proven cross-lane reuse".into(),
            }],
            cache_key: 0,
        };
        plan.rekey();
        plan.validate()?;
        Ok(plan)
    }

    /// Adds a shared allocation only after the caller proves uniform control.
    /// This is intentionally opt-in: the current elementwise linearizer never
    /// invokes it, so CPU execution remains global/register only.
    pub fn with_shared(
        mut self,
        allocation: SpaceAllocation,
        barrier: BarrierPoint,
    ) -> Result<Self, MemorySpaceError> {
        self.shared.push(allocation.clone());
        self.barriers.push(barrier);
        self.promotions
            .push(PromotionDecision::Shared { allocation });
        self.rekey();
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), MemorySpaceError> {
        for global in &self.globals {
            if global.alignment == 0 || global.byte_offset % global.alignment != 0 {
                return Err(MemorySpaceError::InvalidAlignment {
                    id: global.buffer as u32,
                    alignment: global.alignment,
                });
            }
        }
        validate_nonoverlap(
            &self
                .registers
                .iter()
                .map(|r| SpaceAllocation {
                    id: r.physical_reg,
                    space: r.space,
                    bytes: 1,
                    alignment: 1,
                    start: r.start,
                    end: r.end,
                })
                .collect::<Vec<_>>(),
        )?;
        validate_nonoverlap(&self.private)?;
        validate_nonoverlap(&self.shared)?;
        for allocation in self.private.iter().chain(&self.shared) {
            if !matches!(allocation.space, MemorySpace::Private | MemorySpace::Shared)
                || allocation.alignment == 0
                || !allocation.alignment.is_power_of_two()
                || (allocation.bytes != 0 && allocation.bytes % allocation.alignment != 0)
            {
                return Err(MemorySpaceError::InvalidAllocation {
                    id: allocation.id,
                    space: allocation.space,
                });
            }
        }
        for barrier in &self.barriers {
            if !barrier.uniform_control {
                return Err(MemorySpaceError::DivergentBarrier(barrier.instruction));
            }
            for used in &barrier.consumes {
                let initialized = barrier.initializes.contains(used)
                    || self
                        .shared
                        .iter()
                        .any(|a| a.id == *used && a.start < barrier.instruction);
                if !initialized {
                    return Err(MemorySpaceError::BarrierUseBeforeInitialize {
                        barrier: barrier.instruction,
                        allocation: *used,
                    });
                }
            }
        }
        Ok(())
    }

    fn rekey(&mut self) {
        let mut hasher = DefaultHasher::new();
        self.registers.hash(&mut hasher);
        self.globals.hash(&mut hasher);
        self.private.hash(&mut hasher);
        self.shared.hash(&mut hasher);
        self.barriers.hash(&mut hasher);
        self.promotions.hash(&mut hasher);
        self.cache_key = hasher.finish();
    }
}

fn validate_nonoverlap(allocations: &[SpaceAllocation]) -> Result<(), MemorySpaceError> {
    for (index, first) in allocations.iter().enumerate() {
        for second in &allocations[index + 1..] {
            if first.space == second.space
                && first.id == second.id
                && first.start <= second.end
                && second.start <= first.end
            {
                return Err(MemorySpaceError::OverlappingAlias {
                    space: first.space,
                    first: first.id,
                    second: second.id,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Shape, lower_graph_elementwise};

    #[test]
    fn plans_registers_deterministically_and_rejects_illegal_shared_use() {
        let mut graph = Graph::new();
        let x = graph.input("x", Shape::from([5]));
        let y = graph.square(x).unwrap();
        let linear =
            crate::LinearKernel::from_uop(&lower_graph_elementwise(&graph, y).unwrap()).unwrap();
        let plan = MemorySpacePlan::from_linear(&linear).unwrap();
        assert!(!plan.registers.is_empty());
        assert_eq!(
            plan.cache_key,
            MemorySpacePlan::from_linear(&linear).unwrap().cache_key
        );
        let allocation = SpaceAllocation {
            id: 7,
            space: MemorySpace::Shared,
            bytes: 16,
            alignment: 16,
            start: 0,
            end: 4,
        };
        assert!(matches!(
            plan.clone().with_shared(
                allocation.clone(),
                BarrierPoint {
                    instruction: 2,
                    scope: BarrierScope::Workgroup,
                    uniform_control: false,
                    initializes: vec![7],
                    consumes: vec![7]
                }
            ),
            Err(MemorySpaceError::DivergentBarrier(2))
        ));
        assert!(matches!(
            plan.with_shared(
                allocation,
                BarrierPoint {
                    instruction: 2,
                    scope: BarrierScope::Workgroup,
                    uniform_control: true,
                    initializes: vec![],
                    consumes: vec![9]
                }
            ),
            Err(MemorySpaceError::BarrierUseBeforeInitialize { .. })
        ));
    }

    #[test]
    fn allocation_validation_covers_alias_alignment_and_zero_bytes() {
        let base = MemorySpacePlan {
            registers: vec![],
            globals: vec![],
            private: vec![],
            shared: vec![],
            barriers: vec![],
            promotions: vec![],
            cache_key: 0,
        };
        let zero = SpaceAllocation {
            id: 1,
            space: MemorySpace::Private,
            bytes: 0,
            alignment: 8,
            start: 0,
            end: 0,
        };
        assert!(
            base.clone()
                .with_shared(
                    zero,
                    BarrierPoint {
                        instruction: 1,
                        scope: BarrierScope::Workgroup,
                        uniform_control: true,
                        initializes: vec![],
                        consumes: vec![]
                    }
                )
                .is_ok()
        );
        let bad = SpaceAllocation {
            id: 2,
            space: MemorySpace::Shared,
            bytes: 8,
            alignment: 3,
            start: 0,
            end: 1,
        };
        assert!(matches!(
            base.clone().with_shared(
                bad,
                BarrierPoint {
                    instruction: 1,
                    scope: BarrierScope::Workgroup,
                    uniform_control: true,
                    initializes: vec![],
                    consumes: vec![]
                }
            ),
            Err(MemorySpaceError::InvalidAllocation { .. })
        ));
        let alias = SpaceAllocation {
            id: 3,
            space: MemorySpace::Shared,
            bytes: 8,
            alignment: 8,
            start: 0,
            end: 2,
        };
        let plan = base
            .with_shared(
                alias.clone(),
                BarrierPoint {
                    instruction: 1,
                    scope: BarrierScope::Workgroup,
                    uniform_control: true,
                    initializes: vec![3],
                    consumes: vec![],
                },
            )
            .unwrap();
        assert!(matches!(
            plan.with_shared(
                alias,
                BarrierPoint {
                    instruction: 2,
                    scope: BarrierScope::Workgroup,
                    uniform_control: true,
                    initializes: vec![3],
                    consumes: vec![]
                }
            ),
            Err(MemorySpaceError::OverlappingAlias { .. })
        ));
    }
}
