//! Private, resource-free reconstruction recipe for admitted inference wrappers.

use super::{
    CapturedHostGather, CapturedHostIndexedMovement, CapturedHostIndexedMovementKind,
    CapturedInference, CapturedInferenceError, CapturedInferenceState, CapturedStatefulInference,
    InferenceStateLink, ReplayInput, captured_inference_identity, captured_owned_output,
    captured_stateful_identity, validate_captured_inference_binding, validate_state_descriptors,
};
use crate::{CapturedSchedule, ExecutionPlanSummary, NodeId, Shape, TensorData};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PortableInferenceHostPolicy {
    None,
    FixedGathers,
    Training,
}

#[derive(Clone, Debug)]
pub(crate) struct PortableCapturedInferenceRecipe {
    wire: PortableCapturedInferenceWire,
}

const PORTABLE_INFERENCE_RECIPE_MAGIC: &[u8; 4] = b"RGMI";
const PORTABLE_INFERENCE_RECIPE_VERSION: u8 = 1;
const MAX_PORTABLE_INFERENCE_RECIPE_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PortableCapturedInferenceWire {
    capture: Vec<u8>,
    public_output_count: Option<u64>,
    state_links: Vec<[u64; 2]>,
    resident_names: Vec<String>,
    host_policy: u8,
    host_gathers: Vec<PortableHostGatherWire>,
    host_indexed_movements: Vec<PortableHostIndexedMovementWire>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PortableHostGatherWire {
    input: String,
    index: u64,
    output: u64,
    axis: u64,
    axis_extent: u64,
    index_elements: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PortableHostIndexedMovementWire {
    input: String,
    index: u64,
    output: u64,
    axis: u64,
    axis_extent: u64,
    index_elements: u64,
    kind: u8,
    provenance: PortableGatherVjpWire,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PortableGatherVjpWire {
    gather: u64,
    data: u64,
    index: u64,
    axis: u64,
    zero_base: u64,
    update: u64,
    scatter_add: u64,
    data_shape: Shape,
    index_shape: Shape,
    gather_shape: Shape,
}

struct PortableStateStructure {
    public_output_count: usize,
    states: Vec<CapturedInferenceState>,
    names: BTreeSet<String>,
    nodes: BTreeSet<NodeId>,
}

fn portable_u64(value: usize, label: &str) -> std::result::Result<u64, CapturedInferenceError> {
    u64::try_from(value).map_err(|_| {
        CapturedInferenceError::Binding(format!("portable inference {label} overflows"))
    })
}

fn portable_usize(value: u64, label: &str) -> std::result::Result<usize, CapturedInferenceError> {
    usize::try_from(value).map_err(|_| {
        CapturedInferenceError::Binding(format!("portable inference {label} overflows"))
    })
}

fn portable_node(value: u64, label: &str) -> std::result::Result<NodeId, CapturedInferenceError> {
    Ok(NodeId::from_index(portable_usize(value, label)?))
}

fn portable_host_gather_wire(
    link: &CapturedHostGather,
) -> std::result::Result<PortableHostGatherWire, CapturedInferenceError> {
    Ok(PortableHostGatherWire {
        input: link.input.name.clone(),
        index: link.index,
        output: link.output,
        axis: portable_u64(link.axis, "host Gather axis")?,
        axis_extent: portable_u64(link.axis_extent, "host Gather extent")?,
        index_elements: portable_u64(link.index_elements, "host Gather index count")?,
    })
}

fn portable_gather_vjp_wire(
    provenance: &crate::ir::GatherVjpProvenance,
) -> std::result::Result<PortableGatherVjpWire, CapturedInferenceError> {
    Ok(PortableGatherVjpWire {
        gather: portable_u64(provenance.gather.index(), "Gather VJP node")?,
        data: portable_u64(provenance.data.index(), "Gather VJP data node")?,
        index: portable_u64(provenance.index.index(), "Gather VJP index node")?,
        axis: portable_u64(provenance.axis, "Gather VJP axis")?,
        zero_base: portable_u64(provenance.zero_base.index(), "Gather VJP zero node")?,
        update: portable_u64(provenance.update.index(), "Gather VJP update node")?,
        scatter_add: portable_u64(provenance.scatter_add.index(), "Gather VJP scatter node")?,
        data_shape: provenance.data_shape.clone(),
        index_shape: provenance.index_shape.clone(),
        gather_shape: provenance.gather_shape.clone(),
    })
}

fn portable_host_indexed_movement_wire(
    link: &CapturedHostIndexedMovement,
) -> std::result::Result<PortableHostIndexedMovementWire, CapturedInferenceError> {
    Ok(PortableHostIndexedMovementWire {
        input: link.input.name.clone(),
        index: link.index,
        output: link.output,
        axis: portable_u64(link.axis, "host indexed movement axis")?,
        axis_extent: portable_u64(link.axis_extent, "host indexed movement extent")?,
        index_elements: portable_u64(link.index_elements, "host indexed movement index count")?,
        kind: match link.kind {
            CapturedHostIndexedMovementKind::Gather => 0,
            CapturedHostIndexedMovementKind::ScatterAdd => 1,
        },
        provenance: portable_gather_vjp_wire(&link.provenance)?,
    })
}

fn portable_recipe_input(
    capture: &CapturedSchedule,
    name: &str,
) -> std::result::Result<ReplayInput, CapturedInferenceError> {
    let inputs = capture
        .inputs
        .iter()
        .filter(|input| input.name.as_str() == name)
        .collect::<Vec<_>>();
    let [input] = inputs.as_slice() else {
        return Err(CapturedInferenceError::Binding(format!(
            "portable inference input {name:?} is not unique"
        )));
    };
    Ok((*input).clone())
}

fn portable_host_gather_for_capture(
    capture: &CapturedSchedule,
    wire: &PortableHostGatherWire,
) -> std::result::Result<CapturedHostGather, CapturedInferenceError> {
    Ok(CapturedHostGather {
        input: portable_recipe_input(capture, &wire.input)?,
        index: wire.index,
        output: wire.output,
        axis: portable_usize(wire.axis, "host Gather axis")?,
        axis_extent: portable_usize(wire.axis_extent, "host Gather extent")?,
        index_elements: portable_usize(wire.index_elements, "host Gather index count")?,
    })
}

fn portable_gather_vjp(
    wire: &PortableGatherVjpWire,
) -> std::result::Result<crate::ir::GatherVjpProvenance, CapturedInferenceError> {
    Ok(crate::ir::GatherVjpProvenance {
        gather: portable_node(wire.gather, "Gather VJP node")?,
        data: portable_node(wire.data, "Gather VJP data node")?,
        index: portable_node(wire.index, "Gather VJP index node")?,
        axis: portable_usize(wire.axis, "Gather VJP axis")?,
        zero_base: portable_node(wire.zero_base, "Gather VJP zero node")?,
        update: portable_node(wire.update, "Gather VJP update node")?,
        scatter_add: portable_node(wire.scatter_add, "Gather VJP scatter node")?,
        data_shape: wire.data_shape.clone(),
        index_shape: wire.index_shape.clone(),
        gather_shape: wire.gather_shape.clone(),
    })
}

fn portable_host_indexed_movement_for_capture(
    capture: &CapturedSchedule,
    wire: &PortableHostIndexedMovementWire,
) -> std::result::Result<CapturedHostIndexedMovement, CapturedInferenceError> {
    Ok(CapturedHostIndexedMovement {
        input: portable_recipe_input(capture, &wire.input)?,
        index: wire.index,
        output: wire.output,
        axis: portable_usize(wire.axis, "host indexed movement axis")?,
        axis_extent: portable_usize(wire.axis_extent, "host indexed movement extent")?,
        index_elements: portable_usize(wire.index_elements, "host indexed movement index count")?,
        kind: match wire.kind {
            0 => CapturedHostIndexedMovementKind::Gather,
            1 => CapturedHostIndexedMovementKind::ScatterAdd,
            _ => {
                return Err(CapturedInferenceError::Binding(
                    "portable host indexed movement kind is invalid".into(),
                ));
            }
        },
        provenance: portable_gather_vjp(&wire.provenance)?,
    })
}

fn authenticate_portable_host_policy(
    inference: &mut CapturedInference,
    host_policy: u8,
) -> std::result::Result<(), CapturedInferenceError> {
    if !inference
        .host_gathers
        .windows(2)
        .all(|pair| pair[0].output < pair[1].output)
        || !inference
            .host_indexed_movements
            .windows(2)
            .all(|pair| pair[0].output < pair[1].output)
    {
        return Err(CapturedInferenceError::Binding(
            "portable host proof order is not canonical".into(),
        ));
    }
    let mut proof_names = BTreeSet::new();
    for link in &inference.host_gathers {
        let static_link = crate::runtime::static_schedule::StaticHostGather {
            input: link.input.desc.id,
            input_desc: link.input.desc.clone(),
            index: link.index,
            output: link.output,
            axis: link.axis,
            axis_extent: link.axis_extent,
            index_elements: link.index_elements,
        };
        crate::runtime::static_schedule::authenticate_fixed_host_gather_lineage(
            &inference.capture.items,
            &static_link,
        )
        .map_err(|error| {
            CapturedInferenceError::Binding(format!(
                "portable host Gather proof is invalid: {error}"
            ))
        })?;
        if !proof_names.insert(link.input.name.clone()) {
            return Err(CapturedInferenceError::Binding(
                "portable host proof input repeats".into(),
            ));
        }
    }
    let mut movements = BTreeMap::<String, Vec<&CapturedHostIndexedMovement>>::new();
    for link in &inference.host_indexed_movements {
        movements
            .entry(link.input.name.clone())
            .or_default()
            .push(link);
    }
    for (name, links) in movements {
        if !proof_names.insert(name) || links.len() != 2 {
            return Err(CapturedInferenceError::Binding(
                "portable host indexed proof inventory differs".into(),
            ));
        }
        let mut static_links = Vec::with_capacity(links.len());
        for link in links {
            static_links.push(crate::runtime::static_schedule::StaticHostIndexedMovement {
                input: link.input.desc.id,
                input_desc: link.input.desc.clone(),
                index: link.index,
                output: link.output,
                axis: link.axis,
                axis_extent: link.axis_extent,
                index_elements: link.index_elements,
                kind: match link.kind {
                    CapturedHostIndexedMovementKind::Gather => {
                        crate::runtime::static_schedule::StaticHostIndexedMovementKind::Gather
                    }
                    CapturedHostIndexedMovementKind::ScatterAdd => {
                        crate::runtime::static_schedule::StaticHostIndexedMovementKind::ScatterAdd
                    }
                },
                provenance: link.provenance.clone(),
            });
        }
        crate::runtime::static_schedule::authenticate_host_indexed_movement_lineage(
            &inference.capture.items,
            &static_links,
        )
        .map_err(|error| {
            CapturedInferenceError::Binding(format!(
                "portable host indexed proof is invalid: {error}"
            ))
        })?;
    }
    let mut hasher = DefaultHasher::new();
    match host_policy {
        0 => return Ok(()),
        1 => "rustgrad-captured-host-gather-fixed-v1".hash(&mut hasher),
        2 if inference.host_gathers.is_empty() => {
            "rustgrad-captured-host-indexed-movement-v1".hash(&mut hasher)
        }
        2 => "rustgrad-captured-training-host-index-proof-v1".hash(&mut hasher),
        _ => {
            return Err(CapturedInferenceError::Binding(
                "portable inference host policy is invalid".into(),
            ));
        }
    }
    inference.identity.hash(&mut hasher);
    if host_policy == 1 {
        inference.host_gathers.hash(&mut hasher);
    } else if inference.host_gathers.is_empty() {
        inference.host_indexed_movements.hash(&mut hasher);
    } else {
        inference.host_gathers.hash(&mut hasher);
        inference.host_indexed_movements.hash(&mut hasher);
    }
    inference.identity = hasher.finish();
    Ok(())
}

impl PortableCapturedInferenceRecipe {
    pub(crate) fn capture_bytes(&self) -> &[u8] {
        &self.wire.capture
    }

    pub(crate) fn host_policy(&self) -> PortableInferenceHostPolicy {
        match self.wire.host_policy {
            0 => PortableInferenceHostPolicy::None,
            1 => PortableInferenceHostPolicy::FixedGathers,
            2 => PortableInferenceHostPolicy::Training,
            _ => unreachable!("validated portable host policy"),
        }
    }

    pub(crate) fn host_input_names(&self) -> BTreeSet<&str> {
        self.wire
            .host_gathers
            .iter()
            .map(|link| link.input.as_str())
            .chain(
                self.wire
                    .host_indexed_movements
                    .iter()
                    .map(|link| link.input.as_str()),
            )
            .collect()
    }

    pub(super) fn from_inference(
        inference: &CapturedInference,
        public_output_count: Option<usize>,
        state_links: &[InferenceStateLink],
        host_policy: PortableInferenceHostPolicy,
    ) -> std::result::Result<Self, CapturedInferenceError> {
        let host_policy = match (host_policy, inference.host_gathers.is_empty()) {
            (PortableInferenceHostPolicy::FixedGathers, true) => 0,
            (PortableInferenceHostPolicy::None, _) => 0,
            (PortableInferenceHostPolicy::FixedGathers, false) => 1,
            (PortableInferenceHostPolicy::Training, _) => 2,
        };
        let public_output_count = public_output_count
            .map(|count| {
                u64::try_from(count).map_err(|_| {
                    CapturedInferenceError::Binding(
                        "portable inference public output count overflow".into(),
                    )
                })
            })
            .transpose()?;
        let state_links = state_links
            .iter()
            .map(|link| {
                Ok([
                    u64::try_from(link.input.index()).map_err(|_| {
                        CapturedInferenceError::Binding(
                            "portable inference state input identity overflow".into(),
                        )
                    })?,
                    u64::try_from(link.output.index()).map_err(|_| {
                        CapturedInferenceError::Binding(
                            "portable inference state output identity overflow".into(),
                        )
                    })?,
                ])
            })
            .collect::<std::result::Result<Vec<_>, CapturedInferenceError>>()?;
        let host_gathers = inference
            .host_gathers
            .iter()
            .map(portable_host_gather_wire)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let host_indexed_movements = inference
            .host_indexed_movements
            .iter()
            .map(portable_host_indexed_movement_wire)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let recipe = Self {
            wire: PortableCapturedInferenceWire {
                capture: inference
                    .capture
                    .to_bytes()
                    .map_err(CapturedInferenceError::Capture)?,
                public_output_count,
                state_links,
                resident_names: inference.resident_bindings.keys().cloned().collect(),
                host_policy,
                host_gathers,
                host_indexed_movements,
            },
        };
        recipe.validate_shape()?;
        recipe.authenticate_structure()?;
        Ok(recipe)
    }

    pub(crate) fn to_bytes(&self) -> std::result::Result<Vec<u8>, CapturedInferenceError> {
        let payload = serde_json::to_vec(&self.wire).map_err(|error| {
            CapturedInferenceError::Binding(format!(
                "portable inference recipe encode failed: {error}"
            ))
        })?;
        let payload_len = u64::try_from(payload.len()).map_err(|_| {
            CapturedInferenceError::Binding("portable inference recipe length overflows".into())
        })?;
        let total = payload.len().checked_add(21).ok_or_else(|| {
            CapturedInferenceError::Binding("portable inference recipe length overflows".into())
        })?;
        if total > MAX_PORTABLE_INFERENCE_RECIPE_BYTES {
            return Err(CapturedInferenceError::Binding(
                "portable inference recipe exceeds byte limit".into(),
            ));
        }
        let mut bytes = Vec::with_capacity(total);
        bytes.extend_from_slice(PORTABLE_INFERENCE_RECIPE_MAGIC);
        bytes.push(PORTABLE_INFERENCE_RECIPE_VERSION);
        bytes.extend_from_slice(&payload_len.to_le_bytes());
        bytes.extend_from_slice(&payload);
        let checksum = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
        bytes.extend_from_slice(&checksum.to_le_bytes());
        Ok(bytes)
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> std::result::Result<Self, CapturedInferenceError> {
        if bytes.len() < 21
            || bytes.len() > MAX_PORTABLE_INFERENCE_RECIPE_BYTES
            || &bytes[..4] != PORTABLE_INFERENCE_RECIPE_MAGIC
            || bytes[4] != PORTABLE_INFERENCE_RECIPE_VERSION
        {
            return Err(CapturedInferenceError::Binding(
                "portable inference recipe header is invalid".into(),
            ));
        }
        let payload_len = u64::from_le_bytes(bytes[5..13].try_into().map_err(|_| {
            CapturedInferenceError::Binding("portable inference recipe length is invalid".into())
        })?);
        let payload_len = usize::try_from(payload_len).map_err(|_| {
            CapturedInferenceError::Binding("portable inference recipe length overflows".into())
        })?;
        let payload_end = 13usize.checked_add(payload_len).ok_or_else(|| {
            CapturedInferenceError::Binding("portable inference recipe length overflows".into())
        })?;
        if payload_end.checked_add(8) != Some(bytes.len()) {
            return Err(CapturedInferenceError::Binding(
                "portable inference recipe length is invalid".into(),
            ));
        }
        let expected = u64::from_le_bytes(bytes[payload_end..].try_into().map_err(|_| {
            CapturedInferenceError::Binding("portable inference recipe checksum is invalid".into())
        })?);
        let actual = bytes[..payload_end]
            .iter()
            .fold(0xcbf29ce484222325u64, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
            });
        if actual != expected {
            return Err(CapturedInferenceError::Binding(
                "portable inference recipe checksum mismatch".into(),
            ));
        }
        let wire = serde_json::from_slice(&bytes[13..payload_end]).map_err(|error| {
            CapturedInferenceError::Binding(format!(
                "portable inference recipe decode failed: {error}"
            ))
        })?;
        let recipe = Self { wire };
        recipe.validate_shape()?;
        recipe.authenticate_structure()?;
        Ok(recipe)
    }

    fn authenticate_structure(&self) -> std::result::Result<(), CapturedInferenceError> {
        let inference = self.inference_structure()?;
        if self.wire.public_output_count.is_some() {
            self.state_structure(&inference)?;
        }
        Ok(())
    }

    fn validate_shape(&self) -> std::result::Result<(), CapturedInferenceError> {
        match self.wire.host_policy {
            0 if self.wire.host_gathers.is_empty()
                && self.wire.host_indexed_movements.is_empty() => {}
            1 if !self.wire.host_gathers.is_empty()
                && self.wire.host_indexed_movements.is_empty() => {}
            2 if !self.wire.host_gathers.is_empty()
                || !self.wire.host_indexed_movements.is_empty() => {}
            _ => {
                return Err(CapturedInferenceError::Binding(
                    "portable inference host policy is inconsistent".into(),
                ));
            }
        }
        if self.wire.public_output_count.is_some() == self.wire.state_links.is_empty() {
            return Err(CapturedInferenceError::Binding(
                "portable inference state inventory is inconsistent".into(),
            ));
        }
        if self.wire.public_output_count.is_some() && !self.wire.resident_names.is_empty() {
            return Err(CapturedInferenceError::Binding(
                "portable stateful inference cannot retain resident values".into(),
            ));
        }
        if !self
            .wire
            .resident_names
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        {
            return Err(CapturedInferenceError::Binding(
                "portable inference resident names are not canonical".into(),
            ));
        }
        Ok(())
    }

    fn inference_structure(
        &self,
    ) -> std::result::Result<CapturedInference, CapturedInferenceError> {
        let capture = CapturedSchedule::from_bytes(&self.wire.capture)
            .map_err(CapturedInferenceError::Capture)?;
        if !capture.quantized_constants.is_empty() {
            return Err(CapturedInferenceError::Binding(
                "portable training inference cannot retain packed constants".into(),
            ));
        }
        let execution_plan = ExecutionPlanSummary::from_capture(&capture, true)
            .map_err(CapturedInferenceError::Summary)?;
        let resident_names = self
            .wire
            .resident_names
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        for name in &resident_names {
            portable_recipe_input(&capture, name)?;
        }
        let transient_inputs = capture
            .inputs
            .iter()
            .filter(|input| !resident_names.contains(input.name.as_str()))
            .cloned()
            .collect();
        let identity = capture.identity;
        let mut inference = CapturedInference {
            capture,
            execution_plan,
            resident_bindings: BTreeMap::new(),
            quantized_input_names: BTreeMap::new(),
            transient_inputs,
            host_gathers: Vec::new(),
            host_indexed_movements: Vec::new(),
            gather_vjp_provenance: Vec::new(),
            identity,
        };
        inference.host_gathers = self
            .wire
            .host_gathers
            .iter()
            .map(|link| portable_host_gather_for_capture(&inference.capture, link))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        inference.host_indexed_movements = self
            .wire
            .host_indexed_movements
            .iter()
            .map(|link| portable_host_indexed_movement_for_capture(&inference.capture, link))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        authenticate_portable_host_policy(&mut inference, self.wire.host_policy)?;
        Ok(inference)
    }

    fn state_structure(
        &self,
        inference: &CapturedInference,
    ) -> std::result::Result<PortableStateStructure, CapturedInferenceError> {
        let public_output_count =
            usize::try_from(self.wire.public_output_count.ok_or_else(|| {
                CapturedInferenceError::Binding(
                    "portable stateful inference output count is absent".into(),
                )
            })?)
            .map_err(|_| {
                CapturedInferenceError::Binding(
                    "portable stateful inference output count overflows".into(),
                )
            })?;
        let mut states = Vec::with_capacity(self.wire.state_links.len());
        let mut names = BTreeSet::new();
        let mut state_nodes = BTreeSet::new();
        if public_output_count > inference.capture.requested.len()
            || public_output_count.checked_add(self.wire.state_links.len())
                != Some(inference.capture.requested.len())
            || inference.capture.requested[public_output_count..]
                .iter()
                .copied()
                .ne(self.wire.state_links.iter().map(|link| link[1]))
        {
            return Err(CapturedInferenceError::Binding(
                "portable stateful inference requested inventory differs".into(),
            ));
        }
        let public_nodes = inference.capture.requested[..public_output_count]
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        for [input, output] in &self.wire.state_links {
            if public_nodes.contains(input) || public_nodes.contains(output) {
                return Err(CapturedInferenceError::Binding(
                    "portable state links and public outputs must own distinct nodes".into(),
                ));
            }
            let input = portable_node(*input, "state input")?;
            let output = portable_node(*output, "state output")?;
            let captured_input = inference
                .capture
                .inputs
                .iter()
                .find(|candidate| candidate.node == input)
                .cloned()
                .ok_or_else(|| {
                    CapturedInferenceError::Binding(
                        "portable state input is absent from captured ownership".into(),
                    )
                })?;
            let captured_output = captured_owned_output(&inference.capture, output, "state")?;
            validate_state_descriptors(&captured_input, &captured_output, "state")?;
            if !names.insert(captured_input.name.clone())
                || !state_nodes.insert(input)
                || !state_nodes.insert(output)
            {
                return Err(CapturedInferenceError::Binding(
                    "portable inference state ownership repeats".into(),
                ));
            }
            states.push(CapturedInferenceState {
                link: InferenceStateLink::new(input, output),
                input: captured_input,
                output: captured_output,
            });
        }
        Ok(PortableStateStructure {
            public_output_count,
            states,
            names,
            nodes: state_nodes,
        })
    }

    fn inference(
        &self,
        resident_bindings: BTreeMap<String, TensorData>,
    ) -> std::result::Result<CapturedInference, CapturedInferenceError> {
        if resident_bindings.keys().ne(self.wire.resident_names.iter()) {
            return Err(CapturedInferenceError::Binding(
                "portable inference resident inventory differs".into(),
            ));
        }
        let mut inference = self.inference_structure()?;
        for (name, value) in &resident_bindings {
            let input = portable_recipe_input(&inference.capture, name)?;
            validate_captured_inference_binding(&input, input.node, value)?;
        }
        inference.identity =
            captured_inference_identity(&inference.capture, &resident_bindings, &BTreeMap::new())?;
        inference.resident_bindings = resident_bindings;
        authenticate_portable_host_policy(&mut inference, self.wire.host_policy)?;
        Ok(inference)
    }

    pub(crate) fn instantiate(
        &self,
        resident_bindings: BTreeMap<String, TensorData>,
    ) -> std::result::Result<CapturedInference, CapturedInferenceError> {
        if self.wire.public_output_count.is_some() {
            return Err(CapturedInferenceError::Binding(
                "portable stateful inference used as stateless inference".into(),
            ));
        }
        self.inference(resident_bindings)
    }

    pub(crate) fn instantiate_stateful(
        &self,
        initial_state: BTreeMap<String, TensorData>,
    ) -> std::result::Result<CapturedStatefulInference, CapturedInferenceError> {
        let mut inference = self.inference(BTreeMap::new())?;
        let structure = self.state_structure(&inference)?;
        for state in &structure.states {
            let input = state.link.input();
            let initial = initial_state.get(&state.input.name).ok_or_else(|| {
                CapturedInferenceError::Binding(format!(
                    "missing portable initial state {}",
                    state.input.name
                ))
            })?;
            validate_captured_inference_binding(&state.input, input, initial)?;
        }
        if initial_state.keys().collect::<BTreeSet<_>>()
            != structure.names.iter().collect::<BTreeSet<_>>()
        {
            return Err(CapturedInferenceError::Binding(
                "portable initial state names mismatch".into(),
            ));
        }
        inference
            .transient_inputs
            .retain(|input| !structure.nodes.contains(&input.node));
        let identity = captured_stateful_identity(
            inference.identity,
            structure.public_output_count,
            &structure.states,
            &initial_state,
        )?;
        Ok(CapturedStatefulInference {
            inference,
            public_output_count: structure.public_output_count,
            states: structure.states,
            initial_state,
            identity,
        })
    }

    #[cfg(test)]
    pub(super) fn with_state_output_for_test(mut self, index: usize, output: NodeId) -> Self {
        self.wire.state_links[index][1] = output.index() as u64;
        self
    }

    #[cfg(test)]
    pub(super) fn swap_state_links_for_test(mut self, left: usize, right: usize) -> Self {
        self.wire.state_links.swap(left, right);
        self
    }
}
