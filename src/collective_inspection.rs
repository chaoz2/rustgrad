//! Canonical inspection-only persistence for a logical collective plan.
//!
//! The envelope retains no contexts, leases, cache observations, timestamps,
//! or execution API. Decoding validates topology without returning a runnable plan.

use crate::CollectivePlan;
use serde::{Deserialize, Serialize};
use std::fmt;

const KIND: &str = "rustgrad_collective_plan_inspection";
const VERSION: u16 = 1;
const MAX_BYTES: usize = 1 << 20;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    kind: String,
    version: u16,
    plan: CollectivePlan,
    identity: u64,
}

/// A versioned, canonical, non-replayable projection of a collective plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectivePlanInspection {
    wire: Wire,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectivePlanInspectionError {
    TooLarge,
    Json(String),
    Kind,
    Version(u16),
    Invalid(String),
    Identity,
    Noncanonical,
}

impl fmt::Display for CollectivePlanInspectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "collective plan inspection error: {self:?}")
    }
}
impl std::error::Error for CollectivePlanInspectionError {}

impl CollectivePlanInspection {
    pub fn new(plan: &CollectivePlan) -> Result<Self, CollectivePlanInspectionError> {
        plan.validate()
            .map_err(|error| CollectivePlanInspectionError::Invalid(error.to_string()))?;
        let mut wire = Wire {
            kind: KIND.into(),
            version: VERSION,
            plan: plan.clone(),
            identity: 0,
        };
        wire.identity = expected_identity(&wire)?;
        Ok(Self { wire })
    }
    pub const fn version(&self) -> u16 { self.wire.version }
    pub const fn identity(&self) -> u64 { self.wire.identity }
    pub fn plan_cache_key(&self) -> &str { &self.wire.plan.cache_key }
    pub fn action_count(&self) -> usize { self.wire.plan.actions.len() }
    pub fn encode(&self) -> Result<Vec<u8>, CollectivePlanInspectionError> {
        validate(&self.wire)?;
        encode_wire(&self.wire)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, CollectivePlanInspectionError> {
        if bytes.len() > MAX_BYTES {
            return Err(CollectivePlanInspectionError::TooLarge);
        }
        let wire: Wire = serde_json::from_slice(bytes)
            .map_err(|error| CollectivePlanInspectionError::Json(error.to_string()))?;
        validate(&wire)?;
        if encode_wire(&wire)? != bytes {
            return Err(CollectivePlanInspectionError::Noncanonical);
        }
        Ok(Self { wire })
    }
}

fn validate(wire: &Wire) -> Result<(), CollectivePlanInspectionError> {
    if wire.kind != KIND { return Err(CollectivePlanInspectionError::Kind); }
    if wire.version != VERSION { return Err(CollectivePlanInspectionError::Version(wire.version)); }
    wire.plan.validate().map_err(|error| CollectivePlanInspectionError::Invalid(error.to_string()))?;
    if wire.identity != expected_identity(wire)? { return Err(CollectivePlanInspectionError::Identity); }
    Ok(())
}
fn expected_identity(wire: &Wire) -> Result<u64, CollectivePlanInspectionError> {
    let mut identity_free = wire.clone();
    identity_free.identity = 0;
    Ok(fnv1a64(&encode_wire(&identity_free)?))
}
fn encode_wire(wire: &Wire) -> Result<Vec<u8>, CollectivePlanInspectionError> {
    serde_json::to_vec(wire).map_err(|error| CollectivePlanInspectionError::Json(error.to_string()))
}
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CollectiveKind, CollectivePlanner, CollectiveRequest, DType, DeviceGroup, Reduction};
    use crate::collective::DeviceId;
    fn plan() -> CollectivePlan {
        CollectivePlanner::plan(CollectiveRequest {
            group: DeviceGroup::new([DeviceId::new("CPU:0").unwrap(), DeviceId::new("CPU:1").unwrap()]).unwrap(),
            kind: CollectiveKind::AllReduce { reduction: Reduction::Sum },
            dtype: DType::F32,
            input_lengths: vec![5, 5],
        }).unwrap()
    }
    #[test]
    fn collective_inspection_is_canonical_and_non_replayable() {
        let artifact = CollectivePlanInspection::new(&plan()).unwrap();
        let bytes = artifact.encode().unwrap();
        let decoded = CollectivePlanInspection::decode(&bytes).unwrap();
        assert_eq!(decoded.encode().unwrap(), bytes);
        assert_eq!(decoded.action_count(), plan().actions.len());
        assert_eq!(decoded.plan_cache_key(), plan().cache_key());
        let mut tampered: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        tampered["identity"] = serde_json::json!(0);
        assert!(CollectivePlanInspection::decode(&serde_json::to_vec(&tampered).unwrap()).is_err());
        let mut bad_topology: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        bad_topology["plan"]["actions"][0]["id"] = serde_json::json!(99);
        assert!(CollectivePlanInspection::decode(&serde_json::to_vec(&bad_topology).unwrap()).is_err());
        let mut unknown: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        unknown["unexpected"] = serde_json::json!(true);
        assert!(CollectivePlanInspection::decode(&serde_json::to_vec(&unknown).unwrap()).is_err());
    }
}
