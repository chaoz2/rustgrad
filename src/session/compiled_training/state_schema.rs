use crate::{DType, Result, Scalar, Shape, TensorData};
use std::{cmp::Ordering, ops::Range};

pub(super) const INTERNAL_PREFIX: &str = "__rustgrad_compiled_training_";
const DROPOUT_COUNTER_INPUT: &str = "__rustgrad_compiled_training_dropout_block_counter";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AdamWParameterState {
    FirstMoment,
    SecondMoment,
    GradientAccumulator,
}

impl AdamWParameterState {
    const fn canonical_suffix(self) -> &'static str {
        match self {
            Self::FirstMoment => "first_moment",
            Self::SecondMoment => "second_moment",
            Self::GradientAccumulator => "gradient_accumulator",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AdamWGlobalState {
    Step,
    AccumulationIndex,
    AccumulatedTokenCount,
    AccumulatedLossNumerator,
}

impl AdamWGlobalState {
    const fn canonical_suffix(self) -> &'static str {
        match self {
            Self::Step => "step",
            Self::AccumulationIndex => "accumulation_index",
            Self::AccumulatedTokenCount => "accumulated_token_count",
            Self::AccumulatedLossNumerator => "accumulated_loss_numerator",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum RecurrentStateSemantic {
    Parameter(Range<usize>),
    Momentum(Range<usize>),
    AdamWParameter {
        parameter: Range<usize>,
        state: AdamWParameterState,
    },
    AdamWGlobal(AdamWGlobalState),
    DropoutCounter,
}

/// Private semantic identity for every compiled-optimizer recurrent value.
///
/// Each constructor materializes the exact legacy spelling once. Equality and
/// ordering use those stored bytes, so arbitrary parameter punctuation cannot
/// change capture, checkpoint, or state-bank order. Parameter-bearing kinds
/// retain only a byte range into that same allocation.
#[derive(Clone, Debug)]
pub(super) struct RecurrentStateKey {
    canonical: Box<str>,
    semantic: RecurrentStateSemantic,
}

impl RecurrentStateKey {
    pub(super) fn parameter(name: impl AsRef<str>) -> Self {
        let (canonical, parameter) = Self::parameterized("parameter:", name.as_ref(), "", "");
        Self {
            canonical,
            semantic: RecurrentStateSemantic::Parameter(parameter),
        }
    }

    pub(super) fn momentum(name: impl AsRef<str>) -> Self {
        let (canonical, parameter) = Self::parameterized("slot:", name.as_ref(), ":", "momentum");
        Self {
            canonical,
            semantic: RecurrentStateSemantic::Momentum(parameter),
        }
    }

    pub(super) fn adamw_parameter(name: impl AsRef<str>, state: AdamWParameterState) -> Self {
        let (canonical, parameter) =
            Self::parameterized("slot:", name.as_ref(), ":", state.canonical_suffix());
        Self {
            canonical,
            semantic: RecurrentStateSemantic::AdamWParameter { parameter, state },
        }
    }

    pub(super) fn adamw_global(state: AdamWGlobalState) -> Self {
        Self {
            canonical: format!("global:{}", state.canonical_suffix()).into_boxed_str(),
            semantic: RecurrentStateSemantic::AdamWGlobal(state),
        }
    }

    pub(super) fn dropout_counter() -> Self {
        Self {
            canonical: Box::from("workload:dropout_block_counter"),
            semantic: RecurrentStateSemantic::DropoutCounter,
        }
    }

    fn parameterized(
        prefix: &str,
        parameter: &str,
        separator: &str,
        suffix: &str,
    ) -> (Box<str>, Range<usize>) {
        let start = prefix.len();
        let end = start + parameter.len();
        let mut canonical = String::with_capacity(end + separator.len() + suffix.len());
        canonical.push_str(prefix);
        canonical.push_str(parameter);
        canonical.push_str(separator);
        canonical.push_str(suffix);
        (canonical.into_boxed_str(), start..end)
    }

    pub(super) fn canonical_name(&self) -> &str {
        &self.canonical
    }

    fn stored_parameter_name(&self, range: &Range<usize>) -> &str {
        &self.canonical[range.clone()]
    }

    pub(super) fn parameter_name(&self) -> Option<&str> {
        match &self.semantic {
            RecurrentStateSemantic::Parameter(parameter) => {
                Some(self.stored_parameter_name(parameter))
            }
            _ => None,
        }
    }

    pub(super) fn momentum_parameter_name(&self) -> Option<&str> {
        match &self.semantic {
            RecurrentStateSemantic::Momentum(parameter) => {
                Some(self.stored_parameter_name(parameter))
            }
            _ => None,
        }
    }

    pub(super) fn parameter_for_adamw_state(&self, expected: AdamWParameterState) -> Option<&str> {
        match &self.semantic {
            RecurrentStateSemantic::AdamWParameter { parameter, state } if *state == expected => {
                Some(self.stored_parameter_name(parameter))
            }
            _ => None,
        }
    }

    pub(super) fn is_accumulation_reset_state(&self) -> bool {
        matches!(
            &self.semantic,
            RecurrentStateSemantic::AdamWGlobal(AdamWGlobalState::AccumulationIndex)
                | RecurrentStateSemantic::AdamWGlobal(AdamWGlobalState::AccumulatedTokenCount)
                | RecurrentStateSemantic::AdamWGlobal(AdamWGlobalState::AccumulatedLossNumerator,)
                | RecurrentStateSemantic::AdamWParameter {
                    state: AdamWParameterState::GradientAccumulator,
                    ..
                }
        )
    }
}

impl PartialEq for RecurrentStateKey {
    fn eq(&self, other: &Self) -> bool {
        self.canonical_name() == other.canonical_name()
    }
}

impl Eq for RecurrentStateKey {}

impl Ord for RecurrentStateKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.canonical_name().cmp(other.canonical_name())
    }
}

impl PartialOrd for RecurrentStateKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
pub(super) struct StateSpec {
    pub(super) key: RecurrentStateKey,
    pub(super) input_name: String,
    pub(super) value: TensorData,
    pub(super) requires_grad: bool,
}

impl StateSpec {
    pub(super) fn parameter(ordinal: usize, name: &str, value: TensorData) -> Self {
        Self {
            key: RecurrentStateKey::parameter(name),
            input_name: format!("{INTERNAL_PREFIX}parameter_{ordinal}"),
            value,
            requires_grad: true,
        }
    }

    pub(super) fn momentum(ordinal: usize, name: &str, value: &TensorData) -> Result<Self> {
        Ok(Self {
            key: RecurrentStateKey::momentum(name),
            input_name: format!("{INTERNAL_PREFIX}momentum_{ordinal}"),
            value: TensorData::zeros_with_dtype(value.shape().clone(), DType::F32)?,
            requires_grad: false,
        })
    }

    pub(super) fn adamw_parameter(
        ordinal: usize,
        name: &str,
        value: &TensorData,
        state: AdamWParameterState,
    ) -> Result<Self> {
        Ok(Self {
            key: RecurrentStateKey::adamw_parameter(name, state),
            input_name: format!("{INTERNAL_PREFIX}{}_{ordinal}", state.canonical_suffix()),
            value: TensorData::zeros_with_dtype(value.shape().clone(), DType::F32)?,
            requires_grad: false,
        })
    }

    pub(super) fn adamw_global(state: AdamWGlobalState) -> Result<Self> {
        let dtype = match state {
            AdamWGlobalState::AccumulatedLossNumerator => DType::F32,
            _ => DType::U64,
        };
        Ok(Self {
            key: RecurrentStateKey::adamw_global(state),
            input_name: format!("{INTERNAL_PREFIX}adamw_{}", state.canonical_suffix()),
            value: TensorData::zeros_with_dtype(Shape::from([]), dtype)?,
            requires_grad: false,
        })
    }

    pub(super) fn dropout_counter() -> Result<Self> {
        Ok(Self {
            key: RecurrentStateKey::dropout_counter(),
            input_name: DROPOUT_COUNTER_INPUT.into(),
            value: TensorData::from_scalars(Shape::from([]), DType::U64, [Scalar::U(0)])?,
            requires_grad: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn recurrent_state_keys_preserve_canonical_names_and_lexical_order() {
        let parameter_names = [
            "a",
            "a0",
            "block:weight",
            "parameter:weight:momentum",
            "slot:block:first_moment",
            "global:step:second_moment",
            "workload:dropout_block_counter:gradient_accumulator",
        ];
        let mut keys = vec![
            RecurrentStateKey::adamw_global(AdamWGlobalState::Step),
            RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulationIndex),
            RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedTokenCount),
            RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedLossNumerator),
            RecurrentStateKey::dropout_counter(),
        ];
        for parameter in parameter_names {
            let parameter_key = RecurrentStateKey::parameter(parameter);
            assert_eq!(
                parameter_key.canonical_name(),
                format!("parameter:{parameter}")
            );
            assert_eq!(parameter_key.parameter_name(), Some(parameter));
            assert_eq!(parameter_key.momentum_parameter_name(), None);
            assert_eq!(
                parameter_key.parameter_for_adamw_state(AdamWParameterState::FirstMoment),
                None
            );
            keys.push(parameter_key);

            let momentum = RecurrentStateKey::momentum(parameter);
            assert_eq!(
                momentum.canonical_name(),
                format!("slot:{parameter}:momentum")
            );
            assert_eq!(momentum.parameter_name(), None);
            assert_eq!(momentum.momentum_parameter_name(), Some(parameter));
            assert_eq!(
                momentum.parameter_for_adamw_state(AdamWParameterState::FirstMoment),
                None
            );
            keys.push(momentum);

            for state in [
                AdamWParameterState::FirstMoment,
                AdamWParameterState::SecondMoment,
                AdamWParameterState::GradientAccumulator,
            ] {
                let key = RecurrentStateKey::adamw_parameter(parameter, state);
                assert_eq!(
                    key.canonical_name(),
                    format!("slot:{parameter}:{}", state.canonical_suffix())
                );
                assert_eq!(key.parameter_name(), None);
                assert_eq!(key.momentum_parameter_name(), None);
                assert_eq!(key.parameter_for_adamw_state(state), Some(parameter));
                for other in [
                    AdamWParameterState::FirstMoment,
                    AdamWParameterState::SecondMoment,
                    AdamWParameterState::GradientAccumulator,
                ] {
                    assert_eq!(
                        key.parameter_for_adamw_state(other).is_some(),
                        other == state
                    );
                }
                keys.push(key);
            }
        }
        assert_eq!(
            keys[0].canonical_name(),
            "global:step",
            "global spelling changed"
        );
        assert_eq!(
            keys[1].canonical_name(),
            "global:accumulation_index",
            "global spelling changed"
        );
        assert_eq!(
            keys[2].canonical_name(),
            "global:accumulated_token_count",
            "global spelling changed"
        );
        assert_eq!(
            keys[3].canonical_name(),
            "global:accumulated_loss_numerator",
            "global spelling changed"
        );
        assert_eq!(
            keys[4].canonical_name(),
            "workload:dropout_block_counter",
            "workload spelling changed"
        );

        let canonical = keys
            .iter()
            .map(|key| key.canonical_name().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            canonical.iter().collect::<BTreeSet<_>>().len(),
            canonical.len(),
            "typed key constructors collided"
        );

        let typed_order = keys
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|key| key.canonical_name().to_owned())
            .collect::<Vec<_>>();
        let mut lexical_order = canonical;
        lexical_order.sort();
        assert_eq!(typed_order, lexical_order);
    }

    #[test]
    fn state_specs_preserve_internal_binding_schema() {
        let parameter = TensorData::zeros([2, 3]).unwrap();
        let cases = [
            StateSpec::parameter(7, "weight", parameter.clone()),
            StateSpec::momentum(7, "weight", &parameter).unwrap(),
            StateSpec::adamw_parameter(7, "weight", &parameter, AdamWParameterState::FirstMoment)
                .unwrap(),
            StateSpec::adamw_parameter(7, "weight", &parameter, AdamWParameterState::SecondMoment)
                .unwrap(),
            StateSpec::adamw_parameter(
                7,
                "weight",
                &parameter,
                AdamWParameterState::GradientAccumulator,
            )
            .unwrap(),
            StateSpec::adamw_global(AdamWGlobalState::Step).unwrap(),
            StateSpec::adamw_global(AdamWGlobalState::AccumulationIndex).unwrap(),
            StateSpec::adamw_global(AdamWGlobalState::AccumulatedTokenCount).unwrap(),
            StateSpec::adamw_global(AdamWGlobalState::AccumulatedLossNumerator).unwrap(),
            StateSpec::dropout_counter().unwrap(),
        ];
        assert!(cases[0].requires_grad);
        assert!(cases[1..].iter().all(|spec| !spec.requires_grad));
        assert!(cases[..5].iter().all(|spec| {
            spec.value.shape() == &Shape::from([2, 3]) && spec.value.dtype() == DType::F32
        }));
        assert!(cases[5..8].iter().all(|spec| {
            spec.value.shape() == &Shape::from([]) && spec.value.dtype() == DType::U64
        }));
        assert_eq!(cases[8].value.dtype(), DType::F32);
        assert_eq!(cases[8].value.shape(), &Shape::from([]));
        assert_eq!(cases[9].value.dtype(), DType::U64);
        assert_eq!(cases[9].value.shape(), &Shape::from([]));
        assert_eq!(
            cases.each_ref().map(|spec| spec.key.canonical_name()),
            [
                "parameter:weight",
                "slot:weight:momentum",
                "slot:weight:first_moment",
                "slot:weight:second_moment",
                "slot:weight:gradient_accumulator",
                "global:step",
                "global:accumulation_index",
                "global:accumulated_token_count",
                "global:accumulated_loss_numerator",
                "workload:dropout_block_counter",
            ]
        );
        assert_eq!(
            cases.map(|spec| spec.input_name),
            [
                "__rustgrad_compiled_training_parameter_7",
                "__rustgrad_compiled_training_momentum_7",
                "__rustgrad_compiled_training_first_moment_7",
                "__rustgrad_compiled_training_second_moment_7",
                "__rustgrad_compiled_training_gradient_accumulator_7",
                "__rustgrad_compiled_training_adamw_step",
                "__rustgrad_compiled_training_adamw_accumulation_index",
                "__rustgrad_compiled_training_adamw_accumulated_token_count",
                "__rustgrad_compiled_training_adamw_accumulated_loss_numerator",
                "__rustgrad_compiled_training_dropout_block_counter",
            ]
        );
    }
}
