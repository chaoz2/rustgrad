//! Deterministic, process-portable training checkpoint container.

use crate::nn::{ParameterRestore, StateDict, StateKind, restore_parameters};
use crate::optim::{LearningRateScheduler, Optimizer};
use crate::{
    DType, Error, Module, Parameter, ParameterId, Result, TensorData, load_safetensors,
    save_safetensors,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const MAGIC: &[u8; 8] = b"RGPTCKP\0";
const FORMAT_VERSION: u32 = 1;
const SECTION_COUNT: usize = 4;
const HEADER_LEN: usize = 8 + 4 + SECTION_COUNT * 8 + SECTION_COUNT * 8;
const MAX_CHECKPOINT_BYTES: usize = 256 * 1024 * 1024;
const MAX_MANIFEST_BYTES: usize = 4 * 1024 * 1024;
const MAX_STATE_PATHS: usize = 1_000_000;

fn invalid(reason: impl Into<String>) -> Error {
    Error::Serialization {
        reason: format!("portable checkpoint: {}", reason.into()),
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ManifestStateKind {
    Parameter,
    Buffer,
}

impl From<StateKind> for ManifestStateKind {
    fn from(value: StateKind) -> Self {
        match value {
            StateKind::Parameter => Self::Parameter,
            StateKind::Buffer => Self::Buffer,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ManifestStatePath {
    path: String,
    canonical_path: String,
    kind: ManifestStateKind,
    dtype: DType,
    shape: Vec<usize>,
    version: u64,
    trainable: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ManifestOptimizerBinding {
    name: String,
    group: usize,
    canonical_module_path: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: u32,
    module: Vec<ManifestStatePath>,
    optimizer: Vec<ManifestOptimizerBinding>,
}

#[derive(Clone)]
struct ObservedState {
    path: String,
    parameter: Parameter,
    identity: ParameterId,
    kind: ManifestStateKind,
    data: TensorData,
    version: u64,
    trainable: bool,
}

struct Decoded {
    manifest: Manifest,
    module: StateDict,
    optimizer: StateDict,
    scheduler: StateDict,
}

/// A deterministic checkpoint whose identities are module paths and structural
/// tie classes rather than process-local [`crate::ParameterId`] values.
///
/// The byte format has a fixed magic/version header, checked section lengths
/// and checksums, a bounded typed JSON manifest, and three canonical
/// safetensors payloads for module, optimizer, and scheduler state. It never
/// deserializes executable code, graphs, devices, or backend resources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PortableTrainingCheckpoint {
    bytes: Vec<u8>,
}

impl PortableTrainingCheckpoint {
    /// Captures module values and versions, tie topology, optimizer ownership,
    /// and optimizer/scheduler state into deterministic portable bytes.
    pub fn capture(
        module: &(impl Module + ?Sized),
        optimizer: &Optimizer,
        scheduler: &LearningRateScheduler,
    ) -> Result<Self> {
        let observed = observe_module(module)?;
        let (manifest_module, module_state, canonical_by_id) = manifest_module(&observed)?;
        let mut optimizer_bindings = Vec::new();
        for (name, group, identity) in optimizer.portable_checkpoint_bindings() {
            let canonical_module_path = canonical_by_id.get(&identity).ok_or_else(|| {
                invalid(format!(
                    "optimizer parameter {name} is absent from module state"
                ))
            })?;
            let source = observed
                .iter()
                .find(|entry| entry.identity == identity)
                .expect("canonical identity came from observed state");
            if !source.trainable || source.kind != ManifestStateKind::Parameter {
                return Err(invalid(format!(
                    "optimizer parameter {name} does not own a trainable module parameter"
                )));
            }
            optimizer_bindings.push(ManifestOptimizerBinding {
                name,
                group,
                canonical_module_path: canonical_module_path.clone(),
            });
        }
        optimizer_bindings.sort_by(|a, b| (&a.name, a.group).cmp(&(&b.name, b.group)));
        let manifest = Manifest {
            schema: FORMAT_VERSION,
            module: manifest_module,
            optimizer: optimizer_bindings,
        };
        let manifest = serde_json::to_vec(&manifest)
            .map_err(|error| invalid(format!("manifest encoding failed: {error}")))?;
        let module = save_safetensors(module_state.tensors(), &BTreeMap::new())?;
        let optimizer = save_safetensors(optimizer.state_dict()?.tensors(), &BTreeMap::new())?;
        let scheduler = save_safetensors(scheduler.state_dict()?.tensors(), &BTreeMap::new())?;
        Self::from_sections([manifest, module, optimizer, scheduler])
    }

    /// Validates a complete checkpoint container before retaining its bytes.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self> {
        let bytes = bytes.into();
        decode(&bytes)?;
        Ok(Self { bytes })
    }

    /// Returns the exact deterministic container bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Restores into fresh, structurally equivalent, version-zero host objects
    /// before they are bound into a [`crate::Graph`]. All paths,
    /// descriptors, ties, ownership/group bindings, and serialized states are
    /// validated before the module batch is locked and mutated. Optimizer and
    /// scheduler candidates are assigned only after that infallible commit.
    pub fn restore(
        &self,
        module: &(impl Module + ?Sized),
        optimizer: &mut Optimizer,
        scheduler: &mut LearningRateScheduler,
    ) -> Result<()> {
        let decoded = decode(&self.bytes)?;
        let observed = observe_module(module)?;
        if observed.iter().any(|entry| entry.version != 0) {
            return Err(invalid(
                "restore target must be freshly constructed at version zero",
            ));
        }
        let target = target_schema(&observed)?;
        validate_manifest(&decoded.manifest, &decoded.module)?;
        if !same_module_schema(&target.manifest, &decoded.manifest.module) {
            return Err(invalid(
                "module paths, descriptors, or tie topology mismatch",
            ));
        }

        let target_by_canonical: BTreeMap<_, _> = target
            .canonical_parameters
            .iter()
            .map(|(path, entry)| (path.clone(), entry.clone()))
            .collect();
        let target_canonical_by_id: BTreeMap<_, _> = target_by_canonical
            .iter()
            .map(|(path, entry)| (entry.parameter.id(), path.clone()))
            .collect();
        let actual_bindings = optimizer.portable_checkpoint_bindings();
        let actual_names = actual_bindings
            .iter()
            .map(|(name, _, _)| name.clone())
            .collect::<BTreeSet<_>>();
        let expected_names = decoded
            .manifest
            .optimizer
            .iter()
            .map(|binding| binding.name.clone())
            .collect::<BTreeSet<_>>();
        if actual_names != expected_names || actual_bindings.len() != expected_names.len() {
            return Err(invalid("optimizer parameter names mismatch"));
        }
        let expected_bindings: BTreeMap<_, _> = decoded
            .manifest
            .optimizer
            .iter()
            .map(|binding| (binding.name.clone(), binding))
            .collect();
        let mut restored_versions = BTreeMap::new();
        for (name, group, identity) in actual_bindings {
            let expected = expected_bindings[&name];
            let canonical = target_canonical_by_id
                .get(&identity)
                .ok_or_else(|| invalid("optimizer owns a parameter outside the module"))?;
            if expected.group != group || expected.canonical_module_path != *canonical {
                return Err(invalid(format!(
                    "optimizer ownership or group mismatch for {name}"
                )));
            }
            let version = decoded
                .manifest
                .module
                .iter()
                .find(|entry| entry.path == *canonical)
                .expect("validated canonical path")
                .version;
            restored_versions.insert(name, version);
        }

        let next_optimizer =
            optimizer.portable_restore_candidate(&decoded.optimizer, &restored_versions)?;
        let next_scheduler = scheduler.restore_candidate(&decoded.scheduler)?;
        let mut restores = Vec::with_capacity(target_by_canonical.len());
        for (canonical, observed) in target_by_canonical {
            let manifest = decoded
                .manifest
                .module
                .iter()
                .find(|entry| entry.path == canonical)
                .expect("validated canonical entry");
            restores.push(ParameterRestore {
                parameter: observed.parameter,
                data: decoded.module.tensors()[&canonical].clone(),
                expected_version: observed.version,
                restored_version: manifest.version,
            });
        }
        restore_parameters(restores)?;
        *optimizer = next_optimizer;
        *scheduler = next_scheduler;
        Ok(())
    }

    fn from_sections(sections: [Vec<u8>; SECTION_COUNT]) -> Result<Self> {
        if sections[0].len() > MAX_MANIFEST_BYTES {
            return Err(invalid("manifest exceeds size limit"));
        }
        let body_len = sections.iter().try_fold(0usize, |total, section| {
            total
                .checked_add(section.len())
                .ok_or_else(|| invalid("container length overflow"))
        })?;
        let total = HEADER_LEN
            .checked_add(body_len)
            .ok_or_else(|| invalid("container length overflow"))?;
        if total > MAX_CHECKPOINT_BYTES {
            return Err(invalid("container exceeds size limit"));
        }
        let mut bytes = Vec::with_capacity(total);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        for section in &sections {
            bytes.extend_from_slice(&(section.len() as u64).to_le_bytes());
        }
        for section in &sections {
            bytes.extend_from_slice(&checksum(section).to_le_bytes());
        }
        for section in sections {
            bytes.extend_from_slice(&section);
        }
        Self::from_bytes(bytes)
    }
}

struct TargetSchema {
    manifest: Vec<ManifestStatePath>,
    canonical_parameters: BTreeMap<String, ObservedState>,
}

fn observe_module(module: &(impl Module + ?Sized)) -> Result<Vec<ObservedState>> {
    let mut observed = Vec::new();
    let mut failure = None;
    module.visit("", &mut |path, parameter, kind| {
        if failure.is_some() {
            return;
        }
        if path.is_empty() {
            failure = Some(invalid("module state path must not be empty"));
            return;
        }
        match parameter.snapshot() {
            Ok(snapshot) => observed.push(ObservedState {
                path,
                parameter: parameter.clone(),
                identity: snapshot.identity,
                kind: kind.into(),
                data: snapshot.data,
                version: snapshot.version,
                trainable: snapshot.trainable,
            }),
            Err(error) => failure = Some(error),
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    observed.sort_by(|a, b| a.path.cmp(&b.path));
    if observed.windows(2).any(|pair| pair[0].path == pair[1].path) {
        return Err(invalid("module state paths must be unique"));
    }
    if observed.len() > MAX_STATE_PATHS {
        return Err(invalid("module state path count exceeds limit"));
    }
    Ok(observed)
}

fn manifest_module(
    observed: &[ObservedState],
) -> Result<(
    Vec<ManifestStatePath>,
    StateDict,
    BTreeMap<ParameterId, String>,
)> {
    let mut canonical_by_id = BTreeMap::<ParameterId, String>::new();
    for entry in observed {
        canonical_by_id
            .entry(entry.identity)
            .or_insert_with(|| entry.path.clone());
    }
    let mut tensors = BTreeMap::new();
    let mut manifest = Vec::with_capacity(observed.len());
    for entry in observed {
        let canonical_path = canonical_by_id[&entry.identity].clone();
        let canonical = observed
            .iter()
            .find(|candidate| candidate.path == canonical_path)
            .expect("canonical path came from observed state");
        if entry.kind != canonical.kind
            || entry.data.dtype() != canonical.data.dtype()
            || entry.data.shape() != canonical.data.shape()
            || entry.version != canonical.version
            || entry.trainable != canonical.trainable
        {
            return Err(invalid("tied module paths have incompatible descriptors"));
        }
        if entry.path == canonical_path {
            tensors.insert(entry.path.clone(), entry.data.clone());
        }
        manifest.push(ManifestStatePath {
            path: entry.path.clone(),
            canonical_path,
            kind: entry.kind,
            dtype: entry.data.dtype(),
            shape: entry.data.shape().dims().to_vec(),
            version: entry.version,
            trainable: entry.trainable,
        });
    }
    Ok((manifest, StateDict::from(tensors), canonical_by_id))
}

fn target_schema(observed: &[ObservedState]) -> Result<TargetSchema> {
    let (manifest, _, _) = manifest_module(observed)?;
    let mut canonical_parameters = BTreeMap::new();
    for entry in observed {
        let schema = manifest
            .iter()
            .find(|schema| schema.path == entry.path)
            .expect("schema mirrors observed state");
        if schema.path == schema.canonical_path {
            canonical_parameters.insert(schema.path.clone(), entry.clone());
        }
    }
    Ok(TargetSchema {
        manifest,
        canonical_parameters,
    })
}

fn same_module_schema(lhs: &[ManifestStatePath], rhs: &[ManifestStatePath]) -> bool {
    lhs.len() == rhs.len()
        && lhs.iter().zip(rhs).all(|(left, right)| {
            left.path == right.path
                && left.canonical_path == right.canonical_path
                && left.kind == right.kind
                && left.dtype == right.dtype
                && left.shape == right.shape
                && left.trainable == right.trainable
        })
}

fn validate_manifest(manifest: &Manifest, module: &StateDict) -> Result<()> {
    if manifest.schema != FORMAT_VERSION {
        return Err(invalid("manifest schema mismatch"));
    }
    if manifest.module.len() > MAX_STATE_PATHS {
        return Err(invalid("module state path count exceeds limit"));
    }
    let mut paths = BTreeSet::new();
    let mut canonical = BTreeSet::new();
    for entry in &manifest.module {
        if entry.path.is_empty() || !paths.insert(entry.path.clone()) {
            return Err(invalid("module state paths must be nonempty and unique"));
        }
        if entry.path == entry.canonical_path {
            canonical.insert(entry.path.clone());
        }
    }
    for entry in &manifest.module {
        let source = manifest
            .module
            .iter()
            .find(|candidate| candidate.path == entry.canonical_path)
            .ok_or_else(|| invalid("tie canonical path is missing"))?;
        if source.path != source.canonical_path
            || entry.kind != source.kind
            || entry.dtype != source.dtype
            || entry.shape != source.shape
            || entry.version != source.version
            || entry.trainable != source.trainable
        {
            return Err(invalid("tie class descriptors are inconsistent"));
        }
    }
    if module.tensors().keys().cloned().collect::<BTreeSet<_>>() != canonical {
        return Err(invalid("module tensor payload keys mismatch manifest"));
    }
    for entry in manifest
        .module
        .iter()
        .filter(|entry| entry.path == entry.canonical_path)
    {
        let value = &module.tensors()[&entry.path];
        if value.dtype() != entry.dtype || value.shape().dims() != entry.shape {
            return Err(invalid("module tensor payload descriptor mismatch"));
        }
    }
    let mut optimizer_names = BTreeSet::new();
    for binding in &manifest.optimizer {
        if binding.name.is_empty() || !optimizer_names.insert(binding.name.clone()) {
            return Err(invalid(
                "optimizer binding names must be nonempty and unique",
            ));
        }
        let state = manifest
            .module
            .iter()
            .find(|entry| entry.path == binding.canonical_module_path)
            .ok_or_else(|| invalid("optimizer binding module path is missing"))?;
        if state.path != state.canonical_path
            || state.kind != ManifestStateKind::Parameter
            || !state.trainable
        {
            return Err(invalid(
                "optimizer binding must own a canonical trainable parameter",
            ));
        }
    }
    Ok(())
}

fn decode(bytes: &[u8]) -> Result<Decoded> {
    if bytes.len() > MAX_CHECKPOINT_BYTES {
        return Err(invalid("container exceeds size limit"));
    }
    if bytes.len() < HEADER_LEN || &bytes[..8] != MAGIC {
        return Err(invalid("invalid or truncated header"));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("header checked"));
    if version != FORMAT_VERSION {
        return Err(invalid(format!("unsupported format version {version}")));
    }
    let mut lengths = [0usize; SECTION_COUNT];
    let mut checksums = [0u64; SECTION_COUNT];
    let mut cursor = 12;
    for length in &mut lengths {
        let raw = u64::from_le_bytes(
            bytes[cursor..cursor + 8]
                .try_into()
                .expect("header checked"),
        );
        *length = usize::try_from(raw).map_err(|_| invalid("section length overflows usize"))?;
        cursor += 8;
    }
    for expected in &mut checksums {
        *expected = u64::from_le_bytes(
            bytes[cursor..cursor + 8]
                .try_into()
                .expect("header checked"),
        );
        cursor += 8;
    }
    if lengths[0] > MAX_MANIFEST_BYTES {
        return Err(invalid("manifest exceeds size limit"));
    }
    let expected = lengths.iter().try_fold(HEADER_LEN, |total, length| {
        total
            .checked_add(*length)
            .ok_or_else(|| invalid("container length overflow"))
    })?;
    if expected != bytes.len() {
        return Err(invalid("section lengths do not match container length"));
    }
    let mut sections: [&[u8]; SECTION_COUNT] = [&[]; SECTION_COUNT];
    cursor = HEADER_LEN;
    for index in 0..SECTION_COUNT {
        sections[index] = &bytes[cursor..cursor + lengths[index]];
        if checksum(sections[index]) != checksums[index] {
            return Err(invalid(format!("section {index} checksum mismatch")));
        }
        cursor += lengths[index];
    }
    let manifest: Manifest = serde_json::from_slice(sections[0])
        .map_err(|error| invalid(format!("invalid manifest: {error}")))?;
    let load = |section: &[u8], name: &str| -> Result<StateDict> {
        let (state, metadata) = load_safetensors(section)?;
        if !metadata.is_empty() {
            return Err(invalid(format!(
                "{name} safetensors metadata must be empty"
            )));
        }
        Ok(StateDict::from(state))
    };
    let decoded = Decoded {
        manifest,
        module: load(sections[1], "module")?,
        optimizer: load(sections[2], "optimizer")?,
        scheduler: load(sections[3], "scheduler")?,
    };
    validate_manifest(&decoded.manifest, &decoded.module)?;
    Ok(decoded)
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;
    use crate::nn::Linear;
    use crate::optim::{LearningRateScheduler, Optimizer, SgdConfig};

    fn sections(bytes: &[u8]) -> [Vec<u8>; SECTION_COUNT] {
        let mut lengths = [0usize; SECTION_COUNT];
        let mut cursor = 12;
        for length in &mut lengths {
            *length = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap()) as usize;
            cursor += 8;
        }
        cursor = HEADER_LEN;
        std::array::from_fn(|index| {
            let section = bytes[cursor..cursor + lengths[index]].to_vec();
            cursor += lengths[index];
            section
        })
    }

    #[test]
    fn portable_container_is_deterministic_and_rejects_corruption() {
        let mut graph = Graph::new();
        let linear = Linear::new(&mut graph, 2, 1, true, 7).unwrap();
        let optimizer = Optimizer::sgd(
            vec![
                ("bias".into(), linear.bias.clone().unwrap()),
                ("weight".into(), linear.weight.clone()),
            ],
            SgdConfig::default(),
        )
        .unwrap();
        let scheduler = LearningRateScheduler::multi_step(vec![2], 0.5).unwrap();
        let first = PortableTrainingCheckpoint::capture(&linear, &optimizer, &scheduler).unwrap();
        let second = PortableTrainingCheckpoint::capture(&linear, &optimizer, &scheduler).unwrap();
        assert_eq!(first, second);
        assert_eq!(&first.as_bytes()[..8], MAGIC);
        assert_eq!(
            u32::from_le_bytes(first.as_bytes()[8..12].try_into().unwrap()),
            FORMAT_VERSION
        );

        let mut corrupt = first.clone().into_bytes();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 1;
        assert!(PortableTrainingCheckpoint::from_bytes(corrupt).is_err());
        assert!(PortableTrainingCheckpoint::from_bytes(first.as_bytes()[..20].to_vec()).is_err());
    }

    #[test]
    fn portable_restore_rejects_each_serialized_state_before_module_mutation() {
        let mut graph = Graph::new();
        let source = Linear::new(&mut graph, 2, 1, true, 7).unwrap();
        let source_optimizer = Optimizer::sgd(
            vec![
                ("bias".into(), source.bias.clone().unwrap()),
                ("weight".into(), source.weight.clone()),
            ],
            SgdConfig::default(),
        )
        .unwrap();
        let source_scheduler = LearningRateScheduler::multi_step(vec![2], 0.5).unwrap();
        let good =
            PortableTrainingCheckpoint::capture(&source, &source_optimizer, &source_scheduler)
                .unwrap();

        let mut malformed = Vec::new();
        let mut optimizer_missing = sections(good.as_bytes());
        let (optimizer, _) = load_safetensors(&optimizer_missing[2]).unwrap();
        let mut optimizer = optimizer;
        optimizer.remove("optimizer.step");
        optimizer_missing[2] = save_safetensors(&optimizer, &BTreeMap::new()).unwrap();
        malformed.push(PortableTrainingCheckpoint::from_sections(optimizer_missing).unwrap());

        let mut scheduler_missing = sections(good.as_bytes());
        let (scheduler, _) = load_safetensors(&scheduler_missing[3]).unwrap();
        let mut scheduler = scheduler;
        let key = scheduler.keys().next().unwrap().clone();
        scheduler.remove(&key);
        scheduler_missing[3] = save_safetensors(&scheduler, &BTreeMap::new()).unwrap();
        malformed.push(PortableTrainingCheckpoint::from_sections(scheduler_missing).unwrap());

        let mut extra_path = sections(good.as_bytes());
        let mut manifest: Manifest = serde_json::from_slice(&extra_path[0]).unwrap();
        let mut alias = manifest.module[0].clone();
        alias.path = "unexpected_alias".into();
        alias.canonical_path = manifest.module[0].path.clone();
        manifest.module.push(alias);
        manifest.module.sort_by(|a, b| a.path.cmp(&b.path));
        extra_path[0] = serde_json::to_vec(&manifest).unwrap();
        malformed.push(PortableTrainingCheckpoint::from_sections(extra_path).unwrap());

        let mut missing_path = sections(good.as_bytes());
        let mut manifest: Manifest = serde_json::from_slice(&missing_path[0]).unwrap();
        let removed = manifest.module.remove(0);
        manifest
            .optimizer
            .retain(|binding| binding.canonical_module_path != removed.path);
        let (module, _) = load_safetensors(&missing_path[1]).unwrap();
        let mut module = module;
        module.remove(&removed.path);
        missing_path[0] = serde_json::to_vec(&manifest).unwrap();
        missing_path[1] = save_safetensors(&module, &BTreeMap::new()).unwrap();
        malformed.push(PortableTrainingCheckpoint::from_sections(missing_path).unwrap());

        for checkpoint in malformed {
            let mut target_graph = Graph::new();
            let target = Linear::new(&mut target_graph, 2, 1, true, 9).unwrap();
            let mut target_optimizer = Optimizer::sgd(
                vec![
                    ("bias".into(), target.bias.clone().unwrap()),
                    ("weight".into(), target.weight.clone()),
                ],
                SgdConfig::default(),
            )
            .unwrap();
            let mut target_scheduler = LearningRateScheduler::multi_step(vec![2], 0.5).unwrap();
            let before_module = target.state_dict().unwrap();
            let before_optimizer = target_optimizer.state_dict().unwrap();
            let before_scheduler = target_scheduler.state_dict().unwrap();
            assert!(
                checkpoint
                    .restore(&target, &mut target_optimizer, &mut target_scheduler)
                    .is_err()
            );
            assert_eq!(target.state_dict().unwrap(), before_module);
            assert_eq!(target_optimizer.state_dict().unwrap(), before_optimizer);
            assert_eq!(target_scheduler.state_dict().unwrap(), before_scheduler);
        }

        let mut bad_manifest = sections(good.as_bytes());
        let mut manifest: Manifest = serde_json::from_slice(&bad_manifest[0]).unwrap();
        manifest.module[0].shape.push(99);
        bad_manifest[0] = serde_json::to_vec(&manifest).unwrap();
        assert!(PortableTrainingCheckpoint::from_sections(bad_manifest).is_err());
    }
}
