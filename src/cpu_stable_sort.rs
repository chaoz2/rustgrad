//! Read-only descriptor and binding preflight for Graph's CPU-only stable
//! Sort pair. It has one dedicated CPU executor but deliberately does not use
//! `ScheduleItem`, serialize a replay artifact, or expose generic execution.

use crate::{DType, Error, Graph, NodeId, Shape, TensorData};
use std::fmt;

const IDENTITY_DOMAIN: &[u8] = b"rustgrad.cpu-stable-sort-plan.v1\0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CpuStableSortPlanError {
    InvalidPair,
    DescriptorOverflow,
    InputDescriptor,
    ValuesDescriptor,
    IndicesDescriptor,
    AliasedBindings,
}

impl fmt::Display for CpuStableSortPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPair => write!(f, "invalid or ambiguous stable sort selector pair"),
            Self::DescriptorOverflow => write!(f, "stable sort descriptor byte extent overflows"),
            Self::InputDescriptor => write!(f, "stable sort input descriptor does not match"),
            Self::ValuesDescriptor => write!(f, "stable sort values descriptor does not match"),
            Self::IndicesDescriptor => write!(f, "stable sort indices descriptor does not match"),
            Self::AliasedBindings => write!(f, "stable sort bindings must not alias"),
        }
    }
}

impl std::error::Error for CpuStableSortPlanError {}

#[derive(Debug)]
pub enum CpuStableSortExecutionError {
    Oracle(Error),
    OutputDescriptor,
}

impl fmt::Display for CpuStableSortExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Oracle(error) => write!(f, "stable sort CPU oracle failed: {error}"),
            Self::OutputDescriptor => {
                write!(f, "stable sort CPU oracle returned an invalid descriptor")
            }
        }
    }
}

impl std::error::Error for CpuStableSortExecutionError {}

/// A payload-free logical tensor descriptor. `bytes` is checked from shape and
/// dtype, never borrowed from a live allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuStableSortDescriptor {
    pub node: NodeId,
    pub shape: Shape,
    pub dtype: DType,
    pub bytes: usize,
}

/// An in-memory-only canonical description of one Graph stable Sort pair.
/// It intentionally carries no executable state, storage, or cache handle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuStableSortPlan {
    graph_id: u64,
    source: CpuStableSortDescriptor,
    values: CpuStableSortDescriptor,
    indices: CpuStableSortDescriptor,
    axis: usize,
    descending: bool,
    pair: u64,
    identity: u64,
}

/// A uniquely borrowed, preflighted CPU Sort binding. Constructing this value
/// performs no execution; consuming it is the only execution entrypoint.
pub struct BoundCpuStableSortPlan<'a> {
    plan: &'a CpuStableSortPlan,
    input: &'a TensorData,
    values: &'a mut TensorData,
    indices: &'a mut TensorData,
}

impl CpuStableSortPlan {
    /// Constructs an exact, unique Graph Sort pair inventory. This is not a
    /// scheduler entrypoint and does not realize, allocate, or cache anything.
    pub fn from_graph(
        graph: &Graph,
        source: NodeId,
        values: NodeId,
        indices: NodeId,
    ) -> Result<Self, CpuStableSortPlanError> {
        let (shape, dtype, axis, descending, pair) = graph
            .stable_sort_pair_for_cpu_plan(source, values, indices)
            .ok_or(CpuStableSortPlanError::InvalidPair)?;
        let source = make_descriptor(source, shape.clone(), dtype)?;
        let values = make_descriptor(values, shape.clone(), dtype)?;
        let indices = make_descriptor(indices, shape, DType::I32)?;
        let mut plan = Self {
            graph_id: graph.id(),
            source,
            values,
            indices,
            axis,
            descending,
            pair,
            identity: 0,
        };
        plan.identity = plan.compute_identity();
        Ok(plan)
    }

    /// A closed, deterministic identity over logical Graph and descriptor
    /// metadata only. It intentionally excludes data bytes and live objects.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    pub fn source(&self) -> &CpuStableSortDescriptor {
        &self.source
    }

    pub fn values(&self) -> &CpuStableSortDescriptor {
        &self.values
    }

    pub fn indices(&self) -> &CpuStableSortDescriptor {
        &self.indices
    }

    pub fn axis(&self) -> usize {
        self.axis
    }

    pub fn descending(&self) -> bool {
        self.descending
    }

    pub fn pair(&self) -> u64 {
        self.pair
    }

    /// Checks concrete CPU values against this logical inventory without
    /// writing or retaining any of them. `TensorData` is owned dense storage,
    /// so pointer equality is the only aliasing relation it represents.
    pub fn preflight_bindings(
        &self,
        input: &TensorData,
        values: &TensorData,
        indices: &TensorData,
    ) -> Result<(), CpuStableSortPlanError> {
        self.validate_canonical()?;
        if std::ptr::eq(input, values)
            || std::ptr::eq(input, indices)
            || std::ptr::eq(values, indices)
        {
            return Err(CpuStableSortPlanError::AliasedBindings);
        }
        validate_data(input, &self.source)
            .then_some(())
            .ok_or(CpuStableSortPlanError::InputDescriptor)?;
        validate_data(values, &self.values)
            .then_some(())
            .ok_or(CpuStableSortPlanError::ValuesDescriptor)?;
        validate_data(indices, &self.indices)
            .then_some(())
            .ok_or(CpuStableSortPlanError::IndicesDescriptor)?;
        Ok(())
    }

    /// Preflights exact caller-owned output tensors and returns the typed
    /// binding required for CPU execution. Rust's exclusive output borrows
    /// make the executable form strictly stronger than logical alias checks.
    pub fn bind<'a>(
        &'a self,
        input: &'a TensorData,
        values: &'a mut TensorData,
        indices: &'a mut TensorData,
    ) -> Result<BoundCpuStableSortPlan<'a>, CpuStableSortPlanError> {
        self.preflight_bindings(input, values, indices)?;
        Ok(BoundCpuStableSortPlan {
            plan: self,
            input,
            values,
            indices,
        })
    }

    fn validate_canonical(&self) -> Result<(), CpuStableSortPlanError> {
        if self.identity != self.compute_identity()
            || self.source.node == self.values.node
            || self.source.node == self.indices.node
            || self.values.node == self.indices.node
            || self.source.shape != self.values.shape
            || self.source.shape != self.indices.shape
            || self.source.dtype != self.values.dtype
            || self.indices.dtype != DType::I32
            || (self.source.shape.rank() == 0 && self.axis != 0)
            || (self.source.shape.rank() != 0
                && (self.axis >= self.source.shape.rank()
                    || self.source.shape.dims()[self.axis] > i32::MAX as usize))
        {
            return Err(CpuStableSortPlanError::InvalidPair);
        }
        for item in [&self.source, &self.values, &self.indices] {
            if make_descriptor(item.node, item.shape.clone(), item.dtype)
                .map(|expected| expected.bytes != item.bytes)
                .unwrap_or(true)
            {
                return Err(CpuStableSortPlanError::DescriptorOverflow);
            }
        }
        Ok(())
    }

    fn compute_identity(&self) -> u64 {
        let mut state = 0xcbf2_9ce4_8422_2325u64;
        fn write(state: &mut u64, bytes: &[u8]) {
            for byte in bytes {
                *state ^= u64::from(*byte);
                *state = state.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        fn write_descriptor(state: &mut u64, descriptor: &CpuStableSortDescriptor) {
            write(state, &(descriptor.node.index() as u64).to_le_bytes());
            write(state, &[dtype_tag(descriptor.dtype)]);
            write(state, &(descriptor.shape.rank() as u64).to_le_bytes());
            for dim in descriptor.shape.dims() {
                write(state, &(*dim as u64).to_le_bytes());
            }
            write(state, &(descriptor.bytes as u64).to_le_bytes());
        }
        write(&mut state, IDENTITY_DOMAIN);
        write(&mut state, &self.graph_id.to_le_bytes());
        write_descriptor(&mut state, &self.source);
        write_descriptor(&mut state, &self.values);
        write_descriptor(&mut state, &self.indices);
        write(&mut state, &(self.axis as u64).to_le_bytes());
        write(&mut state, &[u8::from(self.descending)]);
        write(&mut state, &self.pair.to_le_bytes());
        state
    }
}

impl BoundCpuStableSortPlan<'_> {
    /// Executes the already-validated pair through the shared CPU stable-sort
    /// oracle. Both private outputs are descriptor-checked before two
    /// infallible swaps publish them to their caller-owned destinations.
    pub fn execute(self) -> Result<(), CpuStableSortExecutionError> {
        let (mut sorted_values, mut sorted_indices) =
            crate::backend::stable_sort_pair(self.input, self.plan.axis, self.plan.descending)
                .map_err(CpuStableSortExecutionError::Oracle)?;
        if !validate_data(&sorted_values, &self.plan.values)
            || !validate_data(&sorted_indices, &self.plan.indices)
        {
            return Err(CpuStableSortExecutionError::OutputDescriptor);
        }
        // `TensorData` owns dense storage. These replacements cannot fail and
        // no callback or validation remains between the two publications.
        std::mem::swap(self.values, &mut sorted_values);
        std::mem::swap(self.indices, &mut sorted_indices);
        Ok(())
    }
}

fn make_descriptor(
    node: NodeId,
    shape: Shape,
    dtype: DType,
) -> Result<CpuStableSortDescriptor, CpuStableSortPlanError> {
    let bytes = shape
        .numel()
        .map_err(|_| CpuStableSortPlanError::DescriptorOverflow)?
        .checked_mul(dtype.itemsize())
        .ok_or(CpuStableSortPlanError::DescriptorOverflow)?;
    Ok(CpuStableSortDescriptor {
        node,
        shape,
        dtype,
        bytes,
    })
}

fn validate_data(data: &TensorData, descriptor: &CpuStableSortDescriptor) -> bool {
    data.shape() == &descriptor.shape
        && data.dtype() == descriptor.dtype
        && data.len().checked_mul(data.dtype().itemsize()) == Some(descriptor.bytes)
}

const fn dtype_tag(dtype: DType) -> u8 {
    match dtype {
        DType::Bool => 0,
        DType::I8 => 1,
        DType::U8 => 2,
        DType::I16 => 3,
        DType::U16 => 4,
        DType::I32 => 5,
        DType::U32 => 6,
        DType::I64 => 7,
        DType::U64 => 8,
        DType::F16 => 9,
        DType::BF16 => 10,
        DType::F32 => 11,
        DType::F64 => 12,
        DType::F8E4M3 => 13,
        DType::F8E5M2 => 14,
        DType::F8E4M3FNUZ => 15,
        DType::F8E5M2FNUZ => 16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
        TensorData::new(shape, values.to_vec()).unwrap()
    }

    #[test]
    fn canonical_plan_is_deterministic_and_preflights_without_work() {
        let mut graph = Graph::new();
        let source = graph.input("source", [2, 3]);
        let (values, indices) = graph.sort(source, -1, true).unwrap();
        let first = CpuStableSortPlan::from_graph(&graph, source, values, indices).unwrap();
        let second = CpuStableSortPlan::from_graph(&graph, source, values, indices).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.identity(), second.identity());
        assert_eq!(first.axis(), 1);
        assert_eq!(first.values().dtype, DType::F32);
        assert_eq!(first.indices().dtype, DType::I32);
        assert_eq!(first.values().bytes, 24);
        assert_eq!(first.indices().bytes, 24);
        let input = data([2, 3], &[1., 2., 3., 4., 5., 6.]);
        let output_values = data([2, 3], &[0.; 6]);
        let output_indices =
            TensorData::from_scalars([2, 3], DType::I32, [crate::Scalar::I(0); 6]).unwrap();
        first
            .preflight_bindings(&input, &output_values, &output_indices)
            .unwrap();
        assert_eq!(output_values, data([2, 3], &[0.; 6]));
    }

    #[test]
    fn plan_rejects_swapped_missing_ambiguous_and_bad_bindings_without_work() {
        let mut graph = Graph::new();
        let source = graph.input("source", [2]);
        let (values, indices) = graph.sort(source, 0, false).unwrap();
        assert!(matches!(
            CpuStableSortPlan::from_graph(&graph, source, indices, values),
            Err(CpuStableSortPlanError::InvalidPair)
        ));
        assert!(matches!(
            CpuStableSortPlan::from_graph(&graph, source, values, values),
            Err(CpuStableSortPlanError::InvalidPair)
        ));
        graph.nodes.push(graph.nodes[indices.index()].clone());
        assert!(matches!(
            CpuStableSortPlan::from_graph(&graph, source, values, indices),
            Err(CpuStableSortPlanError::InvalidPair)
        ));

        let mut clean = Graph::new();
        let source = clean.input("source", [2]);
        let (values, indices) = clean.sort(source, 0, false).unwrap();
        let plan = CpuStableSortPlan::from_graph(&clean, source, values, indices).unwrap();
        let input = data([2], &[1., 2.]);
        let correct_values = data([2], &[0., 0.]);
        let wrong_values = data([1], &[0.]);
        let correct_indices =
            TensorData::from_scalars([2], DType::I32, [crate::Scalar::I(0); 2]).unwrap();
        assert!(matches!(
            plan.preflight_bindings(&input, &wrong_values, &correct_indices),
            Err(CpuStableSortPlanError::ValuesDescriptor)
        ));
        assert!(matches!(
            plan.preflight_bindings(&input, &input, &correct_indices),
            Err(CpuStableSortPlanError::AliasedBindings)
        ));
        let wrong_input =
            TensorData::from_scalars([2], DType::I32, [crate::Scalar::I(1), crate::Scalar::I(2)])
                .unwrap();
        assert!(matches!(
            plan.preflight_bindings(&wrong_input, &correct_values, &correct_indices),
            Err(CpuStableSortPlanError::InputDescriptor)
        ));
        let wrong_indices = data([2], &[0., 0.]);
        assert!(matches!(
            plan.preflight_bindings(&input, &correct_values, &wrong_indices),
            Err(CpuStableSortPlanError::IndicesDescriptor)
        ));
        assert!(matches!(
            plan.preflight_bindings(&input, &correct_values, &input),
            Err(CpuStableSortPlanError::AliasedBindings)
        ));
        let mut tampered = plan.clone();
        tampered.axis = 1;
        assert!(matches!(
            tampered.preflight_bindings(&input, &correct_values, &correct_indices),
            Err(CpuStableSortPlanError::InvalidPair)
        ));
        let mut tampered = plan.clone();
        tampered.pair = tampered.pair.wrapping_add(1);
        assert!(matches!(
            tampered.preflight_bindings(&input, &correct_values, &correct_indices),
            Err(CpuStableSortPlanError::InvalidPair)
        ));
        let mut tampered = plan.clone();
        tampered.values.bytes = 1;
        assert!(matches!(
            tampered.preflight_bindings(&input, &correct_values, &correct_indices),
            Err(CpuStableSortPlanError::InvalidPair)
        ));
        assert_eq!(input, data([2], &[1., 2.]));
    }

    #[test]
    fn bound_plan_executes_the_shared_stable_oracle_and_retries_deterministically() {
        let mut graph = Graph::new();
        let source = graph.input("source", [2, 3]);
        let (values, indices) = graph.sort(source, -1, true).unwrap();
        let plan = CpuStableSortPlan::from_graph(&graph, source, values, indices).unwrap();
        let input = data([2, 3], &[1., 1., f32::NAN, -0.0, 0.0, -0.0]);
        let mut output_values = data([2, 3], &[-9.; 6]);
        let mut output_indices =
            TensorData::from_scalars([2, 3], DType::I32, [crate::Scalar::I(-1); 6]).unwrap();
        plan.bind(&input, &mut output_values, &mut output_indices)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(
            (0..output_values.len())
                .map(|index| output_values.scalar_at(index).as_f64())
                .collect::<Vec<_>>(),
            vec![1., 1., 1., -0., -0., -0.]
        );
        assert_eq!(
            output_values.scalar_at(3).as_f64().to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(
            output_values.scalar_at(4).as_f64().to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(
            output_values.scalar_at(5).as_f64().to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(
            (0..output_indices.len())
                .map(|index| output_indices.scalar_at(index).as_i64())
                .collect::<Vec<_>>(),
            vec![0, 1, 0, 0, 1, 2]
        );
        let first_indices = output_indices.clone();
        plan.bind(&input, &mut output_values, &mut output_indices)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(
            (0..output_values.len())
                .map(|index| output_values.scalar_at(index).as_f64())
                .collect::<Vec<_>>(),
            vec![1., 1., 1., -0., -0., -0.]
        );
        assert_eq!(output_indices, first_indices);
        assert_eq!(input.scalar_at(0).as_f64(), 1.0);
        assert_eq!(input.scalar_at(1).as_f64(), 1.0);
        assert!(input.scalar_at(2).as_f64().is_nan());
        assert_eq!(input.scalar_at(3).as_f64().to_bits(), (-0.0f64).to_bits());
        assert_eq!(input.scalar_at(4).as_f64().to_bits(), 0.0f64.to_bits());
        assert_eq!(input.scalar_at(5).as_f64().to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn bound_plan_rejects_scalar_and_sort_short_circuits_empty_before_publication() {
        let mut scalar_graph = Graph::new();
        let scalar = scalar_graph.input("scalar", []);
        let scalar_nodes = scalar_graph.node_count();
        assert!(scalar_graph.sort(scalar, -1, false).is_err());
        assert_eq!(scalar_graph.node_count(), scalar_nodes);

        let mut empty_graph = Graph::new();
        let empty = empty_graph.input("empty", [0]);
        let (values, indices) = empty_graph.sort(empty, 0, false).unwrap();
        assert_eq!(values, empty);
        assert!(CpuStableSortPlan::from_graph(&empty_graph, empty, values, indices).is_err());
        assert_eq!(empty_graph.shape(indices).unwrap(), &Shape::new([0]));
        assert_eq!(empty_graph.dtype(indices).unwrap(), DType::I32);
    }
}
