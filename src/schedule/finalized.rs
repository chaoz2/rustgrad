//! Call-scoped ownership of a compiled schedule and its exact-root encodings.

use super::{
    Schedule, ScheduleError, ScheduleStateBinding, ScheduleValueBinding, artifact::KernelEncodings,
};

pub(crate) struct FinalizedSchedule {
    schedule: Schedule,
    kernels: KernelEncodings,
}

impl FinalizedSchedule {
    pub(super) fn new(schedule: Schedule, kernels: KernelEncodings) -> Self {
        Self { schedule, kernels }
    }

    pub(crate) fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    pub(crate) fn kernels(&self) -> &KernelEncodings {
        &self.kernels
    }

    pub(crate) fn bind_states(
        mut self,
        bindings: Vec<ScheduleStateBinding>,
    ) -> Result<Self, ScheduleError> {
        self.schedule =
            super::mixed::bind_states_with_encodings(self.schedule, bindings, Some(&self.kernels))?;
        Ok(self)
    }

    pub(crate) fn combine_effects(
        mut self,
        effects: Schedule,
        bindings: Vec<ScheduleValueBinding>,
    ) -> Result<Self, ScheduleError> {
        self.kernels.retain_new_roots(&effects.items);
        self.schedule = super::mixed::combine_with_encodings(
            self.schedule,
            effects,
            bindings,
            Some(&self.kernels),
        )?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{CapturedSchedule, DType, Graph, ReduceKind};

    #[test]
    fn final_schedule_keeps_codec_bytes_through_capture_and_state_binding() {
        for shape in [vec![], vec![0, 3], vec![2, 3]] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", shape, DType::F32);
            let output = graph.square(input).unwrap();
            let ordinary = super::super::schedule_many(&graph, &[output]).unwrap();
            let before = crate::uop::artifact::schedule_kernel_encoding_count();
            let finalized = super::super::schedule_many_finalized(&graph, &[output]).unwrap();
            let after_schedule = crate::uop::artifact::schedule_kernel_encoding_count();
            assert_eq!(after_schedule - before, finalized.schedule().items.len());
            let capture =
                CapturedSchedule::capture_finalized(&graph, &finalized, &[output]).unwrap();
            assert_eq!(
                crate::uop::artifact::schedule_kernel_encoding_count(),
                after_schedule
            );
            let reference = CapturedSchedule::capture(&graph, &ordinary, &[output]).unwrap();
            assert_eq!(capture.to_bytes().unwrap(), reference.to_bytes().unwrap());
            let before_binding = crate::uop::artifact::schedule_kernel_encoding_count();
            let finalized = finalized.bind_states(Vec::new()).unwrap();
            assert_eq!(
                crate::uop::artifact::schedule_kernel_encoding_count(),
                before_binding
            );
            assert_eq!(
                finalized.schedule().items[0].cache_key,
                ordinary.items[0].cache_key
            );
        }
    }

    #[test]
    fn finalized_roots_live_only_until_assembly_and_capture_owners_release() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2], DType::F32);
        let output = graph.square(input).unwrap();
        let finalized = super::super::schedule_many_finalized(&graph, &[output]).unwrap();
        let probes = finalized
            .schedule()
            .items
            .iter()
            .map(|item| item.kernel.allocation_liveness_probe())
            .collect::<Vec<_>>();
        let captured = CapturedSchedule::capture_finalized(&graph, &finalized, &[output]).unwrap();
        drop(graph);
        drop(finalized);
        assert!(probes.iter().all(|probe| probe()));
        drop(captured);
        assert!(probes.iter().all(|probe| !probe()));
    }

    #[test]
    fn real_reduction_is_validated_during_retained_capture() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3], DType::F32);
        let output = graph
            .reduce(input, ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        let finalized = super::super::schedule_many_finalized(&graph, &[output]).unwrap();
        let before = crate::schedule::kernel_validation_count();
        let encoded = crate::uop::artifact::schedule_kernel_encoding_count();
        let actual = CapturedSchedule::capture_finalized(&graph, &finalized, &[output]).unwrap();
        assert!(crate::schedule::kernel_validation_count() > before);
        assert_eq!(
            crate::uop::artifact::schedule_kernel_encoding_count(),
            encoded
        );
        let ordinary = super::super::schedule_many(&graph, &[output]).unwrap();
        let expected = CapturedSchedule::capture(&graph, &ordinary, &[output]).unwrap();
        assert_eq!(actual.to_bytes().unwrap(), expected.to_bytes().unwrap());
    }
}
