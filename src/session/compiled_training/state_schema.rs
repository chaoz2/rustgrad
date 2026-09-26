use crate::{DType, Result, Scalar, Shape, TensorData};
use std::{cmp::Ordering, ops::Range};

pub(super) const INTERNAL_PREFIX: &str = "__rustgrad_compiled_training_";
const DROPOUT_COUNTER_INPUT: &str = "__rustgrad_compiled_training_dropout_block_counter";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct OptimizerStateRole(&'static str);

impl OptimizerStateRole {
    pub(super) const fn new(canonical: &'static str) -> Self {
        Self(canonical)
    }

    pub(super) const fn canonical(self) -> &'static str {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct OptimizerStateSchema {
    parameter_roles: &'static [OptimizerStateRole],
    global_roles: &'static [OptimizerStateRole],
}

impl OptimizerStateSchema {
    pub(super) const fn new(
        parameter_roles: &'static [OptimizerStateRole],
        global_roles: &'static [OptimizerStateRole],
    ) -> Self {
        Self {
            parameter_roles,
            global_roles,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum RecurrentStateSemantic {
    Parameter(Range<usize>),
    OptimizerParameter {
        parameter: Range<usize>,
        role: Range<usize>,
    },
    OptimizerGlobal(Range<usize>),
    Workload,
}

/// Private semantic identity for every compiled-training recurrent value.
///
/// Each constructor materializes the exact legacy spelling once. Equality and
/// ordering use those stored bytes, so arbitrary parameter punctuation cannot
/// change capture, checkpoint, or state-bank order. Optimizer meanings remain
/// in their adapters; this common key authenticates only their opaque roles.
#[derive(Clone, Debug)]
pub(super) struct RecurrentStateKey {
    canonical: Box<str>,
    semantic: RecurrentStateSemantic,
}

impl RecurrentStateKey {
    pub(super) fn from_canonical(name: &str, schema: OptimizerStateSchema) -> Result<Self> {
        if let Some(parameter) = name.strip_prefix("parameter:") {
            return Ok(Self::parameter(parameter));
        }
        if let Some(parameter_and_role) = name.strip_prefix("slot:") {
            for role in schema.parameter_roles {
                let suffix = format!(":{}", role.canonical());
                if let Some(parameter) = parameter_and_role.strip_suffix(&suffix) {
                    return Ok(Self::optimizer_parameter(parameter, *role));
                }
            }
        }
        if let Some(role_name) = name.strip_prefix("global:")
            && let Some(role) = schema
                .global_roles
                .iter()
                .find(|role| role.canonical() == role_name)
        {
            return Ok(Self::optimizer_global(*role));
        }
        if name == "workload:dropout_block_counter" {
            return Ok(Self::dropout_counter());
        }
        Err(crate::Error::SessionTraining {
            reason: "compiled recurrent state key is invalid".into(),
        })
    }

    pub(super) fn parameter(name: impl AsRef<str>) -> Self {
        let (canonical, parameter, _) = Self::parameterized("parameter:", name.as_ref(), "", "");
        Self {
            canonical,
            semantic: RecurrentStateSemantic::Parameter(parameter),
        }
    }

    pub(super) fn optimizer_parameter(name: impl AsRef<str>, role: OptimizerStateRole) -> Self {
        let (canonical, parameter, role_range) =
            Self::parameterized("slot:", name.as_ref(), ":", role.canonical());
        Self {
            canonical,
            semantic: RecurrentStateSemantic::OptimizerParameter {
                parameter,
                role: role_range.expect("optimizer role is present"),
            },
        }
    }

    pub(super) fn optimizer_global(role: OptimizerStateRole) -> Self {
        let prefix = "global:";
        let canonical = format!("{prefix}{}", role.canonical()).into_boxed_str();
        Self {
            semantic: RecurrentStateSemantic::OptimizerGlobal(prefix.len()..canonical.len()),
            canonical,
        }
    }

    pub(super) fn dropout_counter() -> Self {
        Self {
            canonical: Box::from("workload:dropout_block_counter"),
            semantic: RecurrentStateSemantic::Workload,
        }
    }

    fn parameterized(
        prefix: &str,
        parameter: &str,
        separator: &str,
        suffix: &str,
    ) -> (Box<str>, Range<usize>, Option<Range<usize>>) {
        let parameter_start = prefix.len();
        let parameter_end = parameter_start + parameter.len();
        let role_start = parameter_end + separator.len();
        let mut canonical = String::with_capacity(role_start + suffix.len());
        canonical.push_str(prefix);
        canonical.push_str(parameter);
        canonical.push_str(separator);
        canonical.push_str(suffix);
        let role = (!suffix.is_empty()).then_some(role_start..role_start + suffix.len());
        (
            canonical.into_boxed_str(),
            parameter_start..parameter_end,
            role,
        )
    }

    pub(super) fn canonical_name(&self) -> &str {
        &self.canonical
    }

    fn stored(&self, range: &Range<usize>) -> &str {
        &self.canonical[range.clone()]
    }

    pub(super) fn parameter_name(&self) -> Option<&str> {
        match &self.semantic {
            RecurrentStateSemantic::Parameter(parameter) => Some(self.stored(parameter)),
            _ => None,
        }
    }

    pub(super) fn parameter_for_optimizer_role(
        &self,
        expected: OptimizerStateRole,
    ) -> Option<&str> {
        match &self.semantic {
            RecurrentStateSemantic::OptimizerParameter { parameter, role }
                if self.stored(role) == expected.canonical() =>
            {
                Some(self.stored(parameter))
            }
            _ => None,
        }
    }

    pub(super) fn is_optimizer_global(&self, expected: OptimizerStateRole) -> bool {
        matches!(
            &self.semantic,
            RecurrentStateSemantic::OptimizerGlobal(role)
                if self.stored(role) == expected.canonical()
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

    const PARAMETER_ROLE: OptimizerStateRole = OptimizerStateRole::new("opaque_parameter");
    const GLOBAL_ROLE: OptimizerStateRole = OptimizerStateRole::new("opaque_global");
    const SCHEMA: OptimizerStateSchema =
        OptimizerStateSchema::new(&[PARAMETER_ROLE], &[GLOBAL_ROLE]);

    #[test]
    fn recurrent_state_keys_preserve_opaque_roles_and_parameter_punctuation() {
        let mut keys = vec![
            RecurrentStateKey::optimizer_global(GLOBAL_ROLE),
            RecurrentStateKey::dropout_counter(),
        ];
        for parameter in [
            "a",
            "slot:block:first_moment",
            "global:step:opaque_parameter",
            "workload:dropout_block_counter:opaque_parameter",
        ] {
            let parameter_key = RecurrentStateKey::parameter(parameter);
            assert_eq!(parameter_key.parameter_name(), Some(parameter));
            let optimizer_key = RecurrentStateKey::optimizer_parameter(parameter, PARAMETER_ROLE);
            assert_eq!(
                optimizer_key.canonical_name(),
                format!("slot:{parameter}:opaque_parameter")
            );
            assert_eq!(
                optimizer_key.parameter_for_optimizer_role(PARAMETER_ROLE),
                Some(parameter)
            );
            assert_eq!(
                RecurrentStateKey::from_canonical(optimizer_key.canonical_name(), SCHEMA).unwrap(),
                optimizer_key
            );
            keys.extend([parameter_key, optimizer_key]);
        }
        let global = RecurrentStateKey::optimizer_global(GLOBAL_ROLE);
        assert!(global.is_optimizer_global(GLOBAL_ROLE));
        assert_eq!(
            RecurrentStateKey::from_canonical(global.canonical_name(), SCHEMA).unwrap(),
            global
        );
        let canonical = keys
            .iter()
            .map(|key| key.canonical_name().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            canonical.iter().collect::<BTreeSet<_>>().len(),
            canonical.len()
        );
        let typed_order = keys
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|key| key.canonical_name().to_owned())
            .collect::<Vec<_>>();
        let mut lexical_order = canonical;
        lexical_order.sort();
        assert_eq!(typed_order, lexical_order);
    }

    #[test]
    fn recurrent_state_keys_reject_roles_outside_the_authenticated_schema() {
        let error = RecurrentStateKey::from_canonical(
            "slot:weight:foreign",
            OptimizerStateSchema::new(&[], &[]),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            crate::Error::SessionTraining { reason }
                if reason == "compiled recurrent state key is invalid"
        ));
    }
}
