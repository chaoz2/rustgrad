//! Shared public capabilities for interpreter and strict-native CPU AdamW.

use super::*;

/// Backend-specific execution hooks behind the common compiled-training API.
///
/// Interpreter and strict-native sessions own the same authenticated AdamW
/// frontier and policy. They differ only in how replay, evaluation, reset, and
/// partial-window transitions execute and in the evidence returned by those
/// transitions. Keeping those differences here lets the public capability
/// traits remain one coherent implementation instead of two parallel lists of
/// forwarding impls.
trait CpuAdamWExecution {
    type Step: CompiledAdamWStep;
    type Evaluation: CompiledEvaluation;
    type Flush: CompiledAdamWFlush;

    fn adamw(&self) -> &CpuCompiledAdamW;
    fn adamw_mut(&mut self) -> &mut CpuCompiledAdamW;

    fn run_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step>;

    fn run_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step>;

    fn run_scheduled_step(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step>;

    fn run_scheduled_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step>;

    fn run_evaluation(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation>;

    fn run_zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult>;

    fn run_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush>;

    fn run_scheduled_partial_window(&mut self) -> Result<Self::Flush>;
}

impl CpuAdamWExecution for CpuCompiledAdamW {
    type Step = CompiledAdamWStepResult;
    type Evaluation = CompiledEvaluationResult;
    type Flush = CompiledAdamWFlushResult;

    fn adamw(&self) -> &CpuCompiledAdamW {
        self
    }

    fn adamw_mut(&mut self) -> &mut CpuCompiledAdamW {
        self
    }

    fn run_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        CpuCompiledAdamW::step(self, inputs, learning_rate)
    }

    fn run_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        CpuCompiledAdamW::step_commit_only(self, inputs, learning_rate)
    }

    fn run_scheduled_step(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step> {
        CpuCompiledAdamW::step_scheduled(self, inputs)
    }

    fn run_scheduled_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        CpuCompiledAdamW::step_commit_only_scheduled(self, inputs)
    }

    fn run_evaluation(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        CpuCompiledAdamW::evaluate(self, inputs)
    }

    fn run_zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        CpuCompiledAdamW::zero_grad(self)
    }

    fn run_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush> {
        CpuCompiledAdamW::flush_partial_window(self, learning_rate)
    }

    fn run_scheduled_partial_window(&mut self) -> Result<Self::Flush> {
        CpuCompiledAdamW::flush_partial_window_scheduled(self)
    }
}

impl CpuAdamWExecution for NativeCpuCompiledAdamW<'_> {
    type Step = NativeCpuCompiledAdamWStepResult;
    type Evaluation = NativeCpuCompiledEvaluationResult;
    type Flush = NativeCpuCompiledAdamWFlushResult;

    fn adamw(&self) -> &CpuCompiledAdamW {
        &self.inner
    }

    fn adamw_mut(&mut self) -> &mut CpuCompiledAdamW {
        &mut self.inner
    }

    fn run_step(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step(self, inputs, learning_rate)
    }

    fn run_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
        learning_rate: TensorData,
    ) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step_commit_only(self, inputs, learning_rate)
    }

    fn run_scheduled_step(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step_scheduled(self, inputs)
    }

    fn run_scheduled_commit_only(
        &mut self,
        inputs: BTreeMap<String, TensorData>,
    ) -> Result<Self::Step> {
        NativeCpuCompiledAdamW::step_commit_only_scheduled(self, inputs)
    }

    fn run_evaluation(&mut self, inputs: BTreeMap<String, TensorData>) -> Result<Self::Evaluation> {
        NativeCpuCompiledAdamW::evaluate(self, inputs)
    }

    fn run_zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
        NativeCpuCompiledAdamW::zero_grad(self)
    }

    fn run_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush> {
        NativeCpuCompiledAdamW::flush_partial_window(self, learning_rate)
    }

    fn run_scheduled_partial_window(&mut self) -> Result<Self::Flush> {
        NativeCpuCompiledAdamW::flush_partial_window_scheduled(self)
    }
}

macro_rules! impl_cpu_adamw_capabilities {
    (
        [$($impl_prefix:tt)*] => $runtime:ty,
        step = $step:ty,
        evaluation = $evaluation:ty,
        flush = $flush:ty
    ) => {
        $($impl_prefix)* CompiledTrainingRuntime for $runtime {
            type Step = $step;

            fn step(
                &mut self,
                inputs: BTreeMap<String, TensorData>,
                learning_rate: TensorData,
            ) -> Result<Self::Step> {
                self.run_step(inputs, learning_rate)
            }

            fn step_count(&self) -> u64 {
                self.adamw().step_count()
            }

            fn capture_identity(&self) -> u64 {
                self.adamw().capture_identity()
            }

            fn parameter_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
                self.adamw().parameter_snapshots()
            }

            fn publish_parameters(&self, module: &dyn Module) -> Result<LoadReport> {
                publish_parameters_with_freeze_policy(
                    module,
                    self.adamw().parameter_snapshots()?,
                    &self.adamw().contract.frozen_parameters,
                )
            }
        }

        $($impl_prefix)* CompiledEvaluationRuntime for $runtime {
            type Evaluation = $evaluation;

            fn evaluate(
                &mut self,
                inputs: BTreeMap<String, TensorData>,
            ) -> Result<Self::Evaluation> {
                self.run_evaluation(inputs)
            }

            fn evaluation_capture_identity(&self) -> Option<u64> {
                self.adamw().evaluation_capture_identity()
            }
        }

        $($impl_prefix)* CompiledCheckpointRuntime for $runtime {
            type Checkpoint = CompiledAdamWCheckpoint;

            fn checkpoint(&self) -> Result<Self::Checkpoint> {
                self.adamw().checkpoint()
            }
        }

        $($impl_prefix)* CompiledCheckpointRestoreRuntime for $runtime {
            fn restore_checkpoint_in_place(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
                let restored = self.adamw().restored_candidate(checkpoint)?;
                *self.adamw_mut() = restored;
                Ok(())
            }
        }

        $($impl_prefix)* CompiledAdamWRuntime for $runtime {
            fn gradient_accumulation_steps(&self) -> u64 {
                self.adamw().gradient_accumulation_steps()
            }

            fn max_gradient_norm(&self) -> Option<f32> {
                self.adamw().max_gradient_norm()
            }

            fn loss_scale(&self) -> f32 {
                self.adamw().loss_scale()
            }

            fn window_loss_report_enabled(&self) -> bool {
                self.adamw().window_loss_report_enabled()
            }

            fn optimizer_step(&self) -> Result<u64> {
                self.adamw().optimizer_step()
            }

            fn accumulation_index(&self) -> Result<u64> {
                self.adamw().accumulation_index()
            }

            fn zero_grad(&mut self) -> Result<CompiledAdamWZeroGradResult> {
                self.run_zero_grad()
            }

            fn zero_grad_capture_identity(&self) -> Option<u64> {
                self.adamw().zero_grad_capture_identity()
            }

            fn first_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
                self.adamw().first_moment_snapshots()
            }

            fn second_moment_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
                self.adamw().second_moment_snapshots()
            }

            fn gradient_accumulator_snapshots(&self) -> Result<BTreeMap<String, TensorData>> {
                self.adamw().gradient_accumulator_snapshots()
            }
        }

        $($impl_prefix)* CompiledAdamWCommitOnlyRuntime for $runtime {
            fn step_commit_only(
                &mut self,
                inputs: BTreeMap<String, TensorData>,
                learning_rate: TensorData,
            ) -> Result<Self::Step> {
                self.run_commit_only(inputs, learning_rate)
            }
        }

        $($impl_prefix)* CompiledTrainingCommitOnlyRuntime for $runtime {
            fn commit_step(
                &mut self,
                inputs: BTreeMap<String, TensorData>,
                learning_rate: TensorData,
            ) -> Result<Self::Step> {
                self.run_commit_only(inputs, learning_rate)
            }
        }

        $($impl_prefix)* CompiledTrainingWindowCommitRuntime for $runtime {
            type WindowCommit = $flush;

            fn commit_partial_window(
                &mut self,
                learning_rate: TensorData,
            ) -> Result<Self::WindowCommit> {
                self.run_partial_window(learning_rate)
            }

            fn partial_window_commit_capture_identity(&self) -> Option<u64> {
                self.adamw().flush_capture_identity()
            }
        }

        $($impl_prefix)* CompiledAdamWFlushRuntime for $runtime {
            type Flush = $flush;

            fn flush_partial_window(&mut self, learning_rate: TensorData) -> Result<Self::Flush> {
                self.run_partial_window(learning_rate)
            }

            fn flush_capture_identity(&self) -> Option<u64> {
                self.adamw().flush_capture_identity()
            }
        }

        $($impl_prefix)* CompiledTrainingRatePolicyWindowCommitRuntime for $runtime {
            fn commit_partial_window_with_rate_policy(&mut self) -> Result<Self::WindowCommit> {
                self.run_scheduled_partial_window()
            }
        }

        $($impl_prefix)* CompiledScheduledAdamWRuntime for $runtime {
            type ScheduledFlush = $flush;

            fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
                self.adamw().captured_multi_step_lr()
            }

            fn step_scheduled(
                &mut self,
                inputs: BTreeMap<String, TensorData>,
            ) -> Result<Self::Step> {
                self.run_scheduled_step(inputs)
            }

            fn flush_partial_window_scheduled(&mut self) -> Result<Self::ScheduledFlush> {
                self.run_scheduled_partial_window()
            }
        }

        $($impl_prefix)* CompiledScheduledAdamWCommitOnlyRuntime for $runtime {
            fn step_commit_only_scheduled(
                &mut self,
                inputs: BTreeMap<String, TensorData>,
            ) -> Result<Self::Step> {
                self.run_scheduled_commit_only(inputs)
            }
        }
    };
}

impl_cpu_adamw_capabilities!(
    [impl] => CpuCompiledAdamW,
    step = CompiledAdamWStepResult,
    evaluation = CompiledEvaluationResult,
    flush = CompiledAdamWFlushResult
);
impl_cpu_adamw_capabilities!(
    [impl<'a>] => NativeCpuCompiledAdamW<'a>,
    step = NativeCpuCompiledAdamWStepResult,
    evaluation = NativeCpuCompiledEvaluationResult,
    flush = NativeCpuCompiledAdamWFlushResult
);

impl<'a> SessionTarget<&'a CompiledAdamWPlan> for CpuSessionTarget {
    type Session = CpuCompiledAdamW;
    type Error = Error;

    fn prepare(&self, plan: &'a CompiledAdamWPlan) -> Result<Self::Session> {
        plan.prepare_cpu()
    }
}

impl<'a> SessionTarget<&'a CompiledAdamWPlan> for ConfiguredCpuSessionTarget {
    type Session = CpuCompiledAdamW;
    type Error = Error;

    fn prepare(&self, plan: &'a CompiledAdamWPlan) -> Result<Self::Session> {
        plan.prepare_cpu_with_non_finite_policy(self.non_finite_policy())
    }
}

impl<'executor> SessionTarget<&CompiledAdamWPlan> for NativeCpuSessionTarget<'executor> {
    type Session = NativeCpuCompiledAdamW<'executor>;
    type Error = Error;

    fn prepare(&self, plan: &CompiledAdamWPlan) -> Result<Self::Session> {
        plan.prepare_native_cpu(self)
    }
}
