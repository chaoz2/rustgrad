//! Payload-free immutable resource records for the opt-in linked F32 Exp path.
//!
//! Captured schedules and sharded CUDA artifacts intentionally remain resource-free.
//! This descriptor records only an attested payload fingerprint; rebind requires
//! the caller to supply the immutable bytes again before any cache or driver work.

use crate::{
    DeviceId, PrimaryContext,
    cuda::{LinkInput, LinkInputResourceDescriptor},
    ptx::{LINKED_F32_EXP_RENDERER_CONTRACT_VERSION, LinkedF32ExpRequest},
};
use serde::{Deserialize, Serialize};

pub const LINKED_F32_EXP_RESOURCE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LinkedF32ExpResourceDescriptor {
    pub schema_version: u32,
    pub resource_identity: String,
    pub renderer_contract_version: u32,
    pub request_identity: String,
    pub device: u32,
    pub sm: u32,
    pub inputs: Vec<LinkInputResourceDescriptor>,
}

/// Process-local owner proof produced only after a descriptor has rebound.
/// It is deliberately not serialized and cannot authorize capture/replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedF32ExpResourceBinding {
    descriptor: LinkedF32ExpResourceDescriptor,
    owner_identity: usize,
}

impl LinkedF32ExpResourceDescriptor {
    pub fn from_request(
        request: &LinkedF32ExpRequest,
        device: DeviceId,
        sm: u32,
    ) -> Result<Self, crate::ptx::PtxError> {
        let mut descriptor = Self {
            schema_version: LINKED_F32_EXP_RESOURCE_SCHEMA_VERSION,
            resource_identity: String::new(),
            renderer_contract_version: LINKED_F32_EXP_RENDERER_CONTRACT_VERSION,
            request_identity: request.identity().into(),
            device: device.0,
            sm,
            inputs: request
                .link_inputs()
                .iter()
                .map(LinkInput::resource_descriptor)
                .collect(),
        };
        descriptor.validate()?;
        descriptor.resource_identity = descriptor.canonical_identity()?;
        Ok(descriptor)
    }

    pub fn encode(&self) -> Result<Vec<u8>, crate::ptx::PtxError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|_| invalid("linked resource encoding"))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, crate::ptx::PtxError> {
        let decoded = serde_json::from_slice::<Self>(bytes)
            .map_err(|_| invalid("linked resource encoding"))?;
        if serde_json::to_vec(&decoded).map_err(|_| invalid("linked resource encoding"))? != bytes {
            return Err(invalid("linked resource unknown or noncanonical fields"));
        }
        decoded.validate()?;
        Ok(decoded)
    }

    pub fn rebind(
        &self,
        primary: &PrimaryContext,
        sm: u32,
        inputs: &[LinkInput],
    ) -> Result<LinkedF32ExpResourceBinding, crate::ptx::PtxError> {
        self.validate()?;
        if self.device != primary.device().0 || self.sm != sm {
            return Err(invalid("linked resource owner/device/SM"));
        }
        let supplied = inputs
            .iter()
            .map(LinkInput::resource_descriptor)
            .collect::<Vec<_>>();
        if supplied != self.inputs {
            return Err(invalid("linked resource payload binding"));
        }
        Ok(LinkedF32ExpResourceBinding {
            descriptor: self.clone(),
            owner_identity: primary.identity(),
        })
    }

    fn validate(&self) -> Result<(), crate::ptx::PtxError> {
        if self.schema_version != LINKED_F32_EXP_RESOURCE_SCHEMA_VERSION
            || self.renderer_contract_version != LINKED_F32_EXP_RENDERER_CONTRACT_VERSION
            || self.request_identity.is_empty()
            || self.device > i32::MAX as u32
            || self.sm == 0
            || self.inputs.len() != 1
            || self.inputs[0].name.is_empty()
            || !self.inputs[0].supports_f32_expf(self.sm)
            || self.inputs.iter().enumerate().any(|(index, input)| {
                self.inputs[..index]
                    .iter()
                    .any(|prior| prior.name == input.name)
            })
        {
            return Err(invalid("linked resource descriptor"));
        }
        if self.resource_identity.is_empty() {
            return Ok(());
        }
        if self.resource_identity != self.canonical_identity()? {
            return Err(invalid("linked resource identity"));
        }
        Ok(())
    }

    fn canonical_identity(&self) -> Result<String, crate::ptx::PtxError> {
        let mut canonical = self.clone();
        canonical.resource_identity.clear();
        let bytes =
            serde_json::to_vec(&canonical).map_err(|_| invalid("linked resource encoding"))?;
        let hash = bytes.iter().fold(0xcbf29ce484222325_u64, |value, byte| {
            (value ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
        Ok(format!(
            "linked-f32-exp-resource-v{}:{hash:016x}",
            self.schema_version
        ))
    }
}

impl LinkedF32ExpResourceBinding {
    pub fn descriptor(&self) -> &LinkedF32ExpResourceDescriptor {
        &self.descriptor
    }
    pub fn validate_owner(&self, primary: &PrimaryContext) -> Result<(), crate::ptx::PtxError> {
        if self.owner_identity != primary.identity() {
            return Err(invalid("linked resource owner"));
        }
        Ok(())
    }
}

fn invalid(message: &str) -> crate::ptx::PtxError {
    crate::ptx::PtxError::InvalidBinding(message.into())
}
