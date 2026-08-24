//! Graph-independent, versioned host parameter storage.

use crate::{DType, Error, Graph, NodeId, Result, Shape, TensorData};
use std::sync::{
    Arc, RwLock, RwLockReadGuard, RwLockWriteGuard,
    atomic::{AtomicU64, Ordering},
};

static NEXT_PARAMETER_ID: AtomicU64 = AtomicU64::new(1);

/// Stable identity shared by cloned handles to one host parameter.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ParameterId(u64);

#[derive(Clone, Debug)]
pub struct Parameter {
    id: ParameterId,
    input_name: String,
    trainable: bool,
    value: Arc<RwLock<ParameterValue>>,
}

#[derive(Clone, Debug)]
struct ParameterValue {
    data: TensorData,
    version: u64,
}

pub(crate) struct ParameterRestore {
    pub parameter: Parameter,
    pub data: TensorData,
    pub expected_version: u64,
    pub restored_version: u64,
}

/// A coherent, immutable parameter value captured under a single read lock.
///
/// The `identity` is stable across `Parameter::clone` and is used to collapse
/// tied parameters. Reads are snapshotted before graph construction or writes;
/// writers acquire only one parameter lock at a time.
#[derive(Clone, Debug)]
pub struct ParameterSnapshot {
    pub data: TensorData,
    pub shape: Shape,
    pub dtype: DType,
    pub version: u64,
    pub identity: ParameterId,
    pub trainable: bool,
    pub input_name: String,
}

impl Parameter {
    pub fn new(data: TensorData, trainable: bool) -> Self {
        let id = ParameterId(NEXT_PARAMETER_ID.fetch_add(1, Ordering::Relaxed));
        Self {
            id,
            input_name: format!("__rustgrad_parameter_{}", id.0),
            trainable,
            value: Arc::new(RwLock::new(ParameterValue { data, version: 0 })),
        }
    }

    /// Snapshots the current host version into `graph`, reusing an existing
    /// leaf only when both the parameter identity and version match.
    pub fn bind(&self, graph: &mut Graph) -> Result<NodeId> {
        graph.bind_parameter(self.snapshot()?)
    }

    /// Returns the current version's already-bound node without mutating the graph.
    /// Call [`Parameter::bind`] first when constructing a forward graph.
    pub fn node(&self, graph: &Graph) -> Result<NodeId> {
        let snapshot = self.snapshot()?;
        graph
            .bound_parameter_node(snapshot.identity, snapshot.version)
            .ok_or(Error::ParameterGraphMismatch)
    }

    pub fn is_trainable(&self) -> bool {
        self.trainable
    }

    fn read(&self, context: &'static str) -> Result<RwLockReadGuard<'_, ParameterValue>> {
        self.value
            .read()
            .map_err(|_| Error::ParameterLockPoisoned { context })
    }

    fn write(&self, context: &'static str) -> Result<RwLockWriteGuard<'_, ParameterValue>> {
        self.value
            .write()
            .map_err(|_| Error::ParameterLockPoisoned { context })
    }

    pub fn snapshot(&self) -> Result<ParameterSnapshot> {
        let value = self.read("snapshotting parameter")?;
        Ok(ParameterSnapshot {
            data: value.data.clone(),
            shape: value.data.shape().clone(),
            dtype: value.data.dtype(),
            version: value.version,
            identity: self.identity(),
            trainable: self.trainable,
            input_name: self.input_name.clone(),
        })
    }

    pub fn shape(&self) -> Result<Shape> {
        Ok(self.snapshot()?.shape)
    }

    pub fn dtype(&self) -> Result<DType> {
        Ok(self.snapshot()?.dtype)
    }

    pub fn value(&self) -> Result<TensorData> {
        Ok(self.snapshot()?.data)
    }

    pub fn version(&self) -> Result<u64> {
        Ok(self.snapshot()?.version)
    }

    pub fn replace(&self, data: TensorData) -> Result<u64> {
        self.replace_expected(data, None)
    }

    pub fn replace_expected(&self, data: TensorData, expected_version: Option<u64>) -> Result<u64> {
        let mut value = self.write("replacing parameter")?;
        if let Some(expected) = expected_version
            && expected != value.version
        {
            return Err(Error::ParameterVersionConflict {
                expected,
                actual: value.version,
            });
        }
        if data.shape() != value.data.shape() || data.dtype() != value.data.dtype() {
            return Err(Error::ParameterValueMismatch {
                expected_shape: value.data.shape().clone(),
                actual_shape: data.shape().clone(),
                expected_dtype: value.data.dtype(),
                actual_dtype: data.dtype(),
            });
        }
        value.data = data;
        value.version = value.version.wrapping_add(1);
        Ok(value.version)
    }

    pub fn id(&self) -> ParameterId {
        self.id
    }

    pub(crate) fn identity(&self) -> ParameterId {
        self.id
    }

    #[cfg(test)]
    pub(crate) fn poison_for_test(&self) {
        let parameter = self.clone();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _guard = parameter.value.write().unwrap();
            panic!("intentional parameter lock poison");
        }));
    }
}

/// Validates and commits a portable module restore while holding every target
/// parameter write lock. Sorting by stable process-local identity gives all
/// callers one lock order; no value is changed until the complete batch has
/// passed version, shape, dtype, and duplicate checks.
pub(crate) fn restore_parameters(mut restores: Vec<ParameterRestore>) -> Result<()> {
    restores.sort_by_key(|restore| restore.parameter.id);
    if restores
        .windows(2)
        .any(|pair| pair[0].parameter.id == pair[1].parameter.id)
    {
        return Err(Error::Serialization {
            reason: "portable checkpoint contains duplicate canonical parameters".into(),
        });
    }
    let mut values = restores
        .iter()
        .map(|restore| restore.parameter.write("restoring portable checkpoint"))
        .collect::<Result<Vec<_>>>()?;
    for (restore, value) in restores.iter().zip(&values) {
        if value.version != restore.expected_version {
            return Err(Error::ParameterVersionConflict {
                expected: restore.expected_version,
                actual: value.version,
            });
        }
        if restore.data.shape() != value.data.shape() || restore.data.dtype() != value.data.dtype()
        {
            return Err(Error::ParameterValueMismatch {
                expected_shape: value.data.shape().clone(),
                actual_shape: restore.data.shape().clone(),
                expected_dtype: value.data.dtype(),
                actual_dtype: restore.data.dtype(),
            });
        }
    }
    for (restore, value) in restores.iter().zip(&mut values) {
        value.data = restore.data.clone();
        value.version = restore.restored_version;
    }
    Ok(())
}
