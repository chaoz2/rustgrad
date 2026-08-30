//! Sidecar resource artifacts for the opt-in linked F32 Exp request.
//!
//! This schema intentionally does not alter `CapturedSchedule` bytes.  A
//! captured schedule therefore remains a legacy, resource-free artifact, and
//! generic capture/replay has no entrypoint that can consume these slots.
//! Linked payloads stay caller-owned: this sidecar records only their stable
//! descriptor identity and requires them again for a data-only rebind.

use crate::{
    CapturedSchedule, PrimaryContext,
    cuda::LinkInput,
    linked_exp::{LinkedF32ExpResourceBinding, LinkedF32ExpResourceDescriptor},
    ptx::LinkedF32ExpRequest,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const LINKED_F32_EXP_RESOURCE_ARTIFACT_SCHEMA_VERSION: u32 = 1;

/// One unambiguous external resource slot for an opt-in typed Exp request.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LinkedF32ExpResourceSlot {
    pub key: String,
    pub consumer_request_identity: String,
    pub resource_identity: String,
    pub owner_device: u32,
    pub sm: u32,
}

/// Canonical, payload-free sidecar bound to one legacy captured schedule.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LinkedF32ExpResourceArtifact {
    pub schema_version: u32,
    pub artifact_identity: String,
    pub capture_identity: u64,
    pub slots: Vec<LinkedF32ExpResourceSlot>,
}

/// Process-local table produced only by a caller-owned payload rebind.
/// It intentionally has no replay or session execution method.
#[derive(Clone, Debug)]
pub struct BoundLinkedF32ExpResources {
    artifact: LinkedF32ExpResourceArtifact,
    bindings: BTreeMap<String, LinkedF32ExpResourceBinding>,
}

impl LinkedF32ExpResourceArtifact {
    /// Creates the sole currently supported slot. The fixed one-slot shape
    /// prevents unused or ambiguous linked-resource metadata from being
    /// attached to a generic captured schedule.
    pub fn from_capture_request(
        capture: &CapturedSchedule,
        descriptor: &LinkedF32ExpResourceDescriptor,
        request: &LinkedF32ExpRequest,
    ) -> Result<Self, crate::ptx::PtxError> {
        crate::schedule::artifact::validate_capture(capture)
            .map_err(|_| invalid("linked resource captured schedule"))?;
        if descriptor.request_identity != request.identity() {
            return Err(invalid("linked resource request identity"));
        }
        let slot = LinkedF32ExpResourceSlot {
            key: format!("linked-f32-exp/{}", request.identity()),
            consumer_request_identity: request.identity().into(),
            resource_identity: descriptor.resource_identity.clone(),
            owner_device: descriptor.device,
            sm: descriptor.sm,
        };
        let mut artifact = Self {
            schema_version: LINKED_F32_EXP_RESOURCE_ARTIFACT_SCHEMA_VERSION,
            artifact_identity: String::new(),
            capture_identity: capture.identity,
            slots: vec![slot],
        };
        artifact.validate()?;
        artifact.artifact_identity = artifact.canonical_identity()?;
        Ok(artifact)
    }

    pub fn encode(&self) -> Result<Vec<u8>, crate::ptx::PtxError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|_| invalid("linked resource artifact encoding"))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, crate::ptx::PtxError> {
        let artifact = serde_json::from_slice::<Self>(bytes)
            .map_err(|_| invalid("linked resource artifact encoding"))?;
        if serde_json::to_vec(&artifact)
            .map_err(|_| invalid("linked resource artifact encoding"))?
            != bytes
        {
            return Err(invalid(
                "linked resource artifact unknown or noncanonical fields",
            ));
        }
        artifact.validate()?;
        Ok(artifact)
    }

    /// Rebinds only caller-supplied immutable payload witnesses. This performs
    /// no cache, allocation, driver, capture, or replay work.
    pub fn rebind(
        &self,
        capture: &CapturedSchedule,
        primary: &PrimaryContext,
        sm: u32,
        request: &LinkedF32ExpRequest,
        descriptors: &BTreeMap<String, LinkedF32ExpResourceDescriptor>,
        payloads: &BTreeMap<String, Vec<LinkInput>>,
    ) -> Result<BoundLinkedF32ExpResources, crate::ptx::PtxError> {
        crate::schedule::artifact::validate_capture(capture)
            .map_err(|_| invalid("linked resource captured schedule"))?;
        self.validate()?;
        if self.capture_identity != capture.identity {
            return Err(invalid("linked resource capture identity"));
        }
        let expected = self.slots[0].key.as_str();
        if descriptors.len() != 1
            || payloads.len() != 1
            || descriptors.keys().next().map(String::as_str) != Some(expected)
            || payloads.keys().next().map(String::as_str) != Some(expected)
        {
            return Err(invalid("linked resource slot table"));
        }
        let slot = &self.slots[0];
        let descriptor = descriptors
            .get(expected)
            .ok_or_else(|| invalid("linked resource slot"))?;
        if descriptor.resource_identity != slot.resource_identity
            || descriptor.request_identity != slot.consumer_request_identity
            || descriptor.device != slot.owner_device
            || descriptor.sm != slot.sm
            || request.identity() != slot.consumer_request_identity
        {
            return Err(invalid("linked resource slot linkage"));
        }
        let payload = payloads
            .get(expected)
            .ok_or_else(|| invalid("linked resource payload"))?;
        let binding = descriptor.rebind(primary, sm, payload)?;
        let bindings = BTreeMap::from([(slot.key.clone(), binding)]);
        Ok(BoundLinkedF32ExpResources {
            artifact: self.clone(),
            bindings,
        })
    }

    fn validate(&self) -> Result<(), crate::ptx::PtxError> {
        if self.schema_version != LINKED_F32_EXP_RESOURCE_ARTIFACT_SCHEMA_VERSION
            || self.capture_identity == 0
            || self.slots.len() != 1
        {
            return Err(invalid("linked resource artifact schema"));
        }
        let slot = &self.slots[0];
        if slot.key.is_empty()
            || slot.consumer_request_identity.is_empty()
            || slot.resource_identity.is_empty()
            || slot.owner_device > i32::MAX as u32
            || slot.sm == 0
            || slot.key != format!("linked-f32-exp/{}", slot.consumer_request_identity)
        {
            return Err(invalid("linked resource artifact slot"));
        }
        if self.artifact_identity.is_empty() {
            return Ok(());
        }
        if self.artifact_identity != self.canonical_identity()? {
            return Err(invalid("linked resource artifact identity"));
        }
        Ok(())
    }

    fn canonical_identity(&self) -> Result<String, crate::ptx::PtxError> {
        let mut canonical = self.clone();
        canonical.artifact_identity.clear();
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|_| invalid("linked resource artifact encoding"))?;
        let hash = bytes.iter().fold(0xcbf29ce484222325_u64, |value, byte| {
            (value ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
        Ok(format!(
            "linked-f32-exp-resource-artifact-v{}:{hash:016x}",
            self.schema_version
        ))
    }
}

impl BoundLinkedF32ExpResources {
    pub fn artifact(&self) -> &LinkedF32ExpResourceArtifact {
        &self.artifact
    }
    pub fn binding(&self, key: &str) -> Option<&LinkedF32ExpResourceBinding> {
        self.bindings.get(key)
    }
    pub fn len(&self) -> usize {
        self.bindings.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

fn invalid(message: &str) -> crate::ptx::PtxError {
    crate::ptx::PtxError::InvalidBinding(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact() -> LinkedF32ExpResourceArtifact {
        let mut artifact = LinkedF32ExpResourceArtifact {
            schema_version: LINKED_F32_EXP_RESOURCE_ARTIFACT_SCHEMA_VERSION,
            artifact_identity: String::new(),
            capture_identity: 41,
            slots: vec![LinkedF32ExpResourceSlot {
                key: "linked-f32-exp/linked-f32-exp-v1:request".into(),
                consumer_request_identity: "linked-f32-exp-v1:request".into(),
                resource_identity: "linked-f32-exp-resource-v1:0000000000000001".into(),
                owner_device: 0,
                sm: 80,
            }],
        };
        artifact.artifact_identity = artifact.canonical_identity().unwrap();
        artifact
    }

    #[test]
    fn linked_resource_sidecar_is_canonical_payload_free_and_tamper_checked() {
        let artifact = artifact();
        let bytes = artifact.encode().unwrap();
        assert_eq!(bytes, artifact.encode().unwrap());
        assert_eq!(
            LinkedF32ExpResourceArtifact::decode(&bytes).unwrap(),
            artifact
        );
        assert!(
            !String::from_utf8(bytes.clone())
                .unwrap()
                .contains("nvvm-payload")
        );

        let mut tampered = artifact.clone();
        tampered.slots[0].sm = 81;
        assert!(tampered.encode().is_err());
        assert!(LinkedF32ExpResourceArtifact::decode(b"{}").is_err());
    }

    #[test]
    fn linked_resource_sidecar_rejects_legacy_ambiguous_and_unknown_slots() {
        let mut unknown = artifact();
        unknown.schema_version += 1;
        assert!(unknown.encode().is_err());

        let mut duplicate = artifact();
        duplicate.slots.push(duplicate.slots[0].clone());
        assert!(duplicate.encode().is_err());

        let mut missing = artifact();
        missing.slots.clear();
        assert!(missing.encode().is_err());

        let mut wrong_key = artifact();
        wrong_key.slots[0].key = "legacy-slot".into();
        assert!(wrong_key.encode().is_err());
    }
}
