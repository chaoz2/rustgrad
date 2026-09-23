//! Typed output and observation protocols for compiled training phases.

use super::{
    CompiledAdamWClipReport, CompiledAdamWWindowLossValue, CompiledStepOutputSelection,
    CpuNonFinitePolicy, has_non_finite_f32, training,
};
use crate::{DType, Graph, NodeId, Result, Shape, TensorData};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy)]
pub(super) struct CompiledAdamWClipNodes {
    pub(super) pre_clip_global_norm: NodeId,
    pub(super) applied_scale: NodeId,
}

#[derive(Clone, Copy)]
pub(super) struct CompiledAdamWWindowLossNodes {
    pub(super) mean_loss: NodeId,
    pub(super) loss_weight: NodeId,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct CompiledTrainingObservationKey(u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CompiledTrainingObservationConstraint {
    ScalarF32,
    ScalarU64Positive,
}

#[derive(Clone, Copy)]
pub(super) struct CompiledTrainingObservationNode {
    pub(super) spec: CompiledTrainingObservationSpec,
    pub(super) node: NodeId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CompiledTrainingObservationSpec {
    pub(super) key: CompiledTrainingObservationKey,
    pub(super) constraint: CompiledTrainingObservationConstraint,
    pub(super) descriptor_error: &'static str,
    pub(super) invalid_value_error: &'static str,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct CompiledTrainingObservationSchema {
    pub(super) entries: Vec<CompiledTrainingObservationSpec>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CompiledTrainingLossOutput {
    ScalarF32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CompiledTrainingPhaseOutputSchema {
    pub(super) loss: CompiledTrainingLossOutput,
    pub(super) named_outputs: Vec<String>,
    pub(super) observations: CompiledTrainingObservationSchema,
}

pub(super) struct CompiledTrainingPhaseOutputs {
    pub(super) loss: TensorData,
    pub(super) named_outputs: BTreeMap<String, TensorData>,
    pub(super) observations: Vec<CompiledTrainingObservationValue>,
}

impl CompiledTrainingPhaseOutputSchema {
    pub(super) fn selected_len(
        &self,
        include_named_outputs: bool,
        include_observations: bool,
    ) -> Result<usize> {
        let loss_count: usize = match self.loss {
            CompiledTrainingLossOutput::ScalarF32 => 1,
        };
        loss_count
            .checked_add(usize::from(include_named_outputs) * self.named_outputs.len())
            .and_then(|count| {
                count.checked_add(usize::from(include_observations) * self.observations.len())
            })
            .ok_or_else(|| training("compiled requested output count overflows"))
    }

    pub(super) fn take(
        &self,
        values: Vec<TensorData>,
        selection: CompiledStepOutputSelection,
        include_observations: bool,
    ) -> CompiledTrainingPhaseOutputs {
        let expected = self
            .selected_len(selection.includes_named_outputs(), include_observations)
            .expect("compiled output schema was authenticated before replay");
        debug_assert_eq!(values.len(), expected);
        let mut values = values.into_iter();
        let loss = values
            .next()
            .expect("compiled loss cardinality was authenticated");
        let named_outputs = if selection.includes_named_outputs() {
            self.named_outputs
                .iter()
                .cloned()
                .zip(&mut values)
                .collect()
        } else {
            BTreeMap::new()
        };
        let observations = if include_observations {
            self.observations
                .entries
                .iter()
                .map(|spec| CompiledTrainingObservationValue {
                    key: spec.key,
                    value: values
                        .next()
                        .expect("compiled observation cardinality was authenticated"),
                })
                .collect()
        } else {
            Vec::new()
        };
        debug_assert!(values.next().is_none());
        CompiledTrainingPhaseOutputs {
            loss,
            named_outputs,
            observations,
        }
    }
}

impl CompiledTrainingObservationSchema {
    pub(super) fn from_nodes(
        graph: &Graph,
        nodes: &[CompiledTrainingObservationNode],
    ) -> Result<Self> {
        let mut keys = BTreeSet::new();
        for observation in nodes {
            if !keys.insert(observation.spec.key) {
                return Err(training("compiled training observation key repeats"));
            }
            validate_observation_descriptor(graph, observation.node, observation.spec.constraint)?;
        }
        Ok(Self {
            entries: nodes.iter().map(|observation| observation.spec).collect(),
        })
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Clone, Debug)]
pub(super) struct CompiledTrainingObservationValue {
    pub(super) key: CompiledTrainingObservationKey,
    pub(super) value: TensorData,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AdamWObservation {
    ClipNorm,
    ClipScale,
    WindowMean,
    WindowWeight,
}

impl AdamWObservation {
    pub(super) const fn spec(self) -> CompiledTrainingObservationSpec {
        let (key, constraint, descriptor_error, invalid_value_error) = match self {
            Self::ClipNorm => (
                0,
                CompiledTrainingObservationConstraint::ScalarF32,
                "compiled CPU clip report must contain rank-zero F32 values",
                "compiled CPU transition has a non-finite clip report",
            ),
            Self::ClipScale => (
                1,
                CompiledTrainingObservationConstraint::ScalarF32,
                "compiled CPU clip report must contain rank-zero F32 values",
                "compiled CPU transition has a non-finite clip report",
            ),
            Self::WindowMean => (
                2,
                CompiledTrainingObservationConstraint::ScalarF32,
                "compiled CPU window loss must be rank-zero F32",
                "compiled CPU transition has a non-finite window loss",
            ),
            Self::WindowWeight => (
                3,
                CompiledTrainingObservationConstraint::ScalarU64Positive,
                "compiled CPU window-loss weight must be rank-zero U64",
                "compiled CPU window-loss weight must be positive",
            ),
        };
        CompiledTrainingObservationSpec {
            key: CompiledTrainingObservationKey(key),
            constraint,
            descriptor_error,
            invalid_value_error,
        }
    }
}

fn adamw_observation_order(clip_report: bool, window_loss_report: bool) -> Vec<AdamWObservation> {
    let mut observations =
        Vec::with_capacity(usize::from(clip_report) * 2 + usize::from(window_loss_report) * 2);
    if clip_report {
        observations.extend([AdamWObservation::ClipNorm, AdamWObservation::ClipScale]);
    }
    if window_loss_report {
        observations.extend([AdamWObservation::WindowMean, AdamWObservation::WindowWeight]);
    }
    observations
}

pub(super) fn adamw_observation_nodes(
    clip: Option<CompiledAdamWClipNodes>,
    window_loss: Option<CompiledAdamWWindowLossNodes>,
) -> Vec<CompiledTrainingObservationNode> {
    adamw_observation_order(clip.is_some(), window_loss.is_some())
        .into_iter()
        .map(|observation| {
            let node = match observation {
                AdamWObservation::ClipNorm => {
                    clip.expect("clip observations require clip nodes")
                        .pre_clip_global_norm
                }
                AdamWObservation::ClipScale => {
                    clip.expect("clip observations require clip nodes")
                        .applied_scale
                }
                AdamWObservation::WindowMean => {
                    window_loss
                        .expect("window observations require window nodes")
                        .mean_loss
                }
                AdamWObservation::WindowWeight => {
                    window_loss
                        .expect("window observations require window nodes")
                        .loss_weight
                }
            };
            CompiledTrainingObservationNode {
                spec: observation.spec(),
                node,
            }
        })
        .collect()
}

pub(super) fn adamw_observation_schema(
    clip_report: bool,
    window_loss_report: bool,
) -> CompiledTrainingObservationSchema {
    CompiledTrainingObservationSchema {
        entries: adamw_observation_order(clip_report, window_loss_report)
            .into_iter()
            .map(AdamWObservation::spec)
            .collect(),
    }
}

pub(super) fn validate_adamw_observation_schema(
    actual: &CompiledTrainingObservationSchema,
    clip_report: bool,
    window_loss_report: bool,
) -> Result<()> {
    if actual != &adamw_observation_schema(clip_report, window_loss_report) {
        return Err(training("compiled AdamW observation schema differs"));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CompiledAdamWAuxiliaryOutputSchema {
    pub(super) observations: CompiledTrainingObservationSchema,
}

impl CompiledAdamWAuxiliaryOutputSchema {
    pub(super) fn from_nodes(
        graph: &Graph,
        nodes: &[CompiledTrainingObservationNode],
    ) -> Result<Self> {
        let schema = Self {
            observations: CompiledTrainingObservationSchema::from_nodes(graph, nodes)?,
        };
        schema.report_flags().ok_or_else(|| {
            training("compiled AdamW auxiliary observation schema is not canonical")
        })?;
        Ok(schema)
    }

    pub(super) fn from_report_flags(clip_report: bool, window_loss_report: bool) -> Self {
        Self {
            observations: adamw_observation_schema(clip_report, window_loss_report),
        }
    }

    pub(super) fn validate_report_flags(
        &self,
        clip_report: bool,
        window_loss_report: bool,
    ) -> Result<()> {
        if self != &Self::from_report_flags(clip_report, window_loss_report) {
            return Err(training(
                "compiled AdamW auxiliary observation flags differ from its schema",
            ));
        }
        Ok(())
    }

    pub(super) fn report_flags(&self) -> Option<(bool, bool)> {
        [(false, false), (true, false), (false, true), (true, true)]
            .into_iter()
            .find(|(clip_report, window_loss_report)| {
                self.observations == adamw_observation_schema(*clip_report, *window_loss_report)
            })
    }

    #[cfg(test)]
    pub(super) fn clip_report_enabled(&self) -> bool {
        self.report_flags()
            .expect("compiled auxiliary output schema was authenticated")
            .0
    }

    #[cfg(test)]
    pub(super) fn window_loss_report_enabled(&self) -> bool {
        self.report_flags()
            .expect("compiled auxiliary output schema was authenticated")
            .1
    }

    pub(super) fn node_ids(
        &self,
        nodes: &[CompiledTrainingObservationNode],
    ) -> Result<Vec<NodeId>> {
        if nodes.iter().map(|observation| observation.spec).ne(self
            .observations
            .entries
            .iter()
            .copied())
        {
            return Err(training(
                "compiled AdamW auxiliary observation nodes differ from its schema",
            ));
        }
        Ok(nodes.iter().map(|observation| observation.node).collect())
    }

    pub(super) fn validate_and_decode(
        &self,
        outputs: &[TensorData],
        policy: CpuNonFinitePolicy,
    ) -> std::result::Result<CompiledAdamWAuxiliaryReports, String> {
        let (clip_report, window_loss_report) = self.report_flags().ok_or_else(|| {
            "compiled AdamW auxiliary observation schema is not canonical".to_owned()
        })?;
        if outputs.len() != self.observations.len() {
            return Err("compiled CPU auxiliary output inventory differs".to_owned());
        }
        validate_staged_observations(outputs, 0, &self.observations, true, policy)?;
        let mut values = outputs.iter();
        let clip_report = clip_report.then(|| {
            let norm = values
                .next()
                .expect("compiled clip-report norm cardinality was authenticated");
            let scale = values
                .next()
                .expect("compiled clip-report scale cardinality was authenticated");
            CompiledAdamWClipReport::new(norm.values()[0], scale.values()[0])
        });
        let window_loss = window_loss_report.then(|| {
            let mean_loss = values
                .next()
                .expect("compiled window-loss mean cardinality was authenticated");
            let loss_weight = values
                .next()
                .expect("compiled window-loss weight cardinality was authenticated");
            CompiledAdamWWindowLossValue {
                mean_loss_bits: mean_loss.values()[0].to_bits(),
                loss_weight: loss_weight.scalar_at(0).as_u64(),
            }
        });
        debug_assert!(values.next().is_none());
        Ok(CompiledAdamWAuxiliaryReports {
            clip_report,
            window_loss,
        })
    }
}

pub(super) struct CompiledAdamWAuxiliaryReports {
    pub(super) clip_report: Option<CompiledAdamWClipReport>,
    pub(super) window_loss: Option<CompiledAdamWWindowLossValue>,
}

fn validate_observation_descriptor(
    graph: &Graph,
    node: NodeId,
    constraint: CompiledTrainingObservationConstraint,
) -> Result<()> {
    let expected_dtype = match constraint {
        CompiledTrainingObservationConstraint::ScalarF32 => DType::F32,
        CompiledTrainingObservationConstraint::ScalarU64Positive => DType::U64,
    };
    if graph.dtype(node)? != expected_dtype || graph.shape(node)? != &Shape::from([]) {
        return Err(training(
            "compiled training observation descriptor mismatch",
        ));
    }
    Ok(())
}

pub(super) fn validate_observation_value_descriptor(
    value: &TensorData,
    constraint: CompiledTrainingObservationConstraint,
) -> Result<()> {
    let expected_dtype = match constraint {
        CompiledTrainingObservationConstraint::ScalarF32 => DType::F32,
        CompiledTrainingObservationConstraint::ScalarU64Positive => DType::U64,
    };
    if value.dtype() != expected_dtype || value.shape() != &Shape::from([]) {
        return Err(training(
            "compiled training observation descriptor mismatch",
        ));
    }
    Ok(())
}

pub(super) fn validate_staged_observations(
    outputs: &[TensorData],
    start: usize,
    schema: &CompiledTrainingObservationSchema,
    enabled: bool,
    policy: CpuNonFinitePolicy,
) -> std::result::Result<(), String> {
    if !enabled {
        return Ok(());
    }
    let end = start
        .checked_add(schema.len())
        .ok_or_else(|| "compiled CPU observation output inventory overflows".to_owned())?;
    let observations = outputs
        .get(start..end)
        .ok_or_else(|| "compiled CPU observation output inventory differs".to_owned())?;
    for (value, spec) in observations.iter().zip(&schema.entries) {
        match spec.constraint {
            CompiledTrainingObservationConstraint::ScalarF32 => {
                if value.shape() != &Shape::from([]) || value.dtype() != DType::F32 {
                    return Err(spec.descriptor_error.to_owned());
                }
                if policy == CpuNonFinitePolicy::RejectTransition
                    && has_non_finite_f32(std::iter::once(value))
                {
                    return Err(spec.invalid_value_error.to_owned());
                }
            }
            CompiledTrainingObservationConstraint::ScalarU64Positive => {
                if value.shape() != &Shape::from([]) || value.dtype() != DType::U64 {
                    return Err(spec.descriptor_error.to_owned());
                }
                if value.scalar_at(0).as_u64() == 0 {
                    return Err(spec.invalid_value_error.to_owned());
                }
            }
        }
    }
    Ok(())
}
