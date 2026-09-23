//! Ordered planning inventory for strict-native CPU training programs.

use super::*;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum NativeCpuTrainingProgramRole {
    Main,
    Accumulation,
    PartialFlush,
    ZeroGrad,
    Evaluation,
}

impl NativeCpuTrainingProgramRole {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Accumulation => "accumulation",
            Self::PartialFlush => "partial-flush",
            Self::ZeroGrad => "zero-grad",
            Self::Evaluation => "evaluation",
        }
    }

    pub(super) const fn render_capsule_role(self) -> NativeCpuRenderCapsuleProgramRole {
        match self {
            Self::Main => NativeCpuRenderCapsuleProgramRole::Main,
            Self::Accumulation => NativeCpuRenderCapsuleProgramRole::Accumulation,
            Self::PartialFlush => NativeCpuRenderCapsuleProgramRole::PartialFlush,
            Self::ZeroGrad => NativeCpuRenderCapsuleProgramRole::ZeroGrad,
            Self::Evaluation => NativeCpuRenderCapsuleProgramRole::Evaluation,
        }
    }
}

pub(super) struct NativeCpuTrainingProgramDrafts<D> {
    by_role: BTreeMap<NativeCpuTrainingProgramRole, D>,
}

impl<D> NativeCpuTrainingProgramDrafts<D> {
    pub(super) fn new() -> Self {
        Self {
            by_role: BTreeMap::new(),
        }
    }

    pub(super) fn insert(&mut self, role: NativeCpuTrainingProgramRole, draft: D) -> Result<()> {
        if self.by_role.insert(role, draft).is_some() {
            return Err(training(format!(
                "compiled native CPU {} planning draft is duplicated",
                role.name()
            )));
        }
        Ok(())
    }

    pub(super) fn take(&mut self, role: NativeCpuTrainingProgramRole) -> Result<D> {
        self.by_role.remove(&role).ok_or_else(|| {
            training(format!(
                "compiled native CPU {} planning draft is absent",
                role.name()
            ))
        })
    }

    pub(super) fn is_empty(&self) -> bool {
        self.by_role.is_empty()
    }
}

pub(super) struct NativeCpuTrainingProgramBatch<C, D> {
    programs: Vec<(NativeCpuTrainingProgramRole, C, D)>,
}

impl<C, D> NativeCpuTrainingProgramBatch<C, D> {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            programs: Vec::with_capacity(capacity),
        }
    }

    pub(super) fn push(
        &mut self,
        role: NativeCpuTrainingProgramRole,
        capture: C,
        draft: D,
    ) -> Result<()> {
        match self.programs.last() {
            None if role != NativeCpuTrainingProgramRole::Main => {
                return Err(training("compiled native CPU main program is absent"));
            }
            Some((previous, _, _)) if *previous >= role => {
                return Err(training("compiled native CPU program role order differs"));
            }
            _ => {}
        }
        self.programs.push((role, capture, draft));
        Ok(())
    }

    pub(super) fn into_planning_inputs(self) -> (Vec<NativeCpuTrainingProgramRole>, Vec<(C, D)>) {
        let mut roles = Vec::with_capacity(self.programs.len());
        let mut programs = Vec::with_capacity(self.programs.len());
        for (role, capture, draft) in self.programs {
            roles.push(role);
            programs.push((capture, draft));
        }
        (roles, programs)
    }
}

pub(super) struct NativeCpuTrainingPrograms<P> {
    pub(super) main: P,
    pub(super) accumulation: Option<P>,
    pub(super) partial_flush: Option<P>,
    pub(super) zero_grad: Option<P>,
    pub(super) evaluation: Option<P>,
}

impl<P> NativeCpuTrainingPrograms<P> {
    pub(super) fn from_ordered(
        roles: Vec<NativeCpuTrainingProgramRole>,
        plans: Vec<P>,
    ) -> Result<Self> {
        if roles.len() != plans.len() {
            return Err(training("compiled native CPU plan inventory differs"));
        }
        let mut main = None;
        let mut accumulation = None;
        let mut partial_flush = None;
        let mut zero_grad = None;
        let mut evaluation = None;
        for (role, plan) in roles.into_iter().zip(plans) {
            let slot = match role {
                NativeCpuTrainingProgramRole::Main => &mut main,
                NativeCpuTrainingProgramRole::Accumulation => &mut accumulation,
                NativeCpuTrainingProgramRole::PartialFlush => &mut partial_flush,
                NativeCpuTrainingProgramRole::ZeroGrad => &mut zero_grad,
                NativeCpuTrainingProgramRole::Evaluation => &mut evaluation,
            };
            if slot.replace(plan).is_some() {
                return Err(training("compiled native CPU program role is duplicated"));
            }
        }
        Ok(Self {
            main: main.ok_or_else(|| training("compiled native CPU main plan is absent"))?,
            accumulation,
            partial_flush,
            zero_grad,
            evaluation,
        })
    }
}
