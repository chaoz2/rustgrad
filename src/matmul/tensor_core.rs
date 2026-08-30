//! Capability-gated CUDA tensor-core matmul planning and semantic simulation.
use super::{
    MatmulBarrierKind, MatmulBarrierPhase, MatmulKernelPlan, MatmulPlanError,
    MatmulResourceEstimate, MatmulTargetCaps, SharedTileLayout,
};
use crate::{DType, Scalar, Shape, TensorData};
use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
};

const LHS_SHARED_ID: u32 = 11;
const RHS_SHARED_ID: u32 = 12;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MmaInstruction {
    M16N8K16RowColF32,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MmaFragmentLayout {
    pub lanes: u32,
    pub lhs_elements_per_lane: u32,
    pub rhs_elements_per_lane: u32,
    pub accumulator_elements_per_lane: u32,
    pub lhs_registers_per_lane: u32,
    pub rhs_registers_per_lane: u32,
    pub accumulator_registers_per_lane: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorCoreTailPolicy {
    ExactTilesOnly,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TensorCoreOutputPolicy {
    /// Accumulate in F32 fragments, then requantize once to the graph dtype.
    RequantizeToGraphDType,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TensorCoreMatmulPlan {
    pub target: MatmulTargetCaps,
    pub instruction: MmaInstruction,
    pub input_dtype: DType,
    pub accumulator_dtype: DType,
    pub output_dtype: DType,
    pub output_policy: TensorCoreOutputPolicy,
    pub tail_policy: TensorCoreTailPolicy,
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub workgroup: [u32; 3],
    pub lhs_shared: SharedTileLayout,
    pub rhs_shared: SharedTileLayout,
    pub fragments: MmaFragmentLayout,
    pub barriers: Vec<MatmulBarrierPhase>,
    pub resources: MatmulResourceEstimate,
    /// Static fragment invocations plus staged scalar loads. Not a timing claim.
    pub estimated_cost: u64,
    pub cache_key: u64,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TensorCoreMatmulPayload {
    pub matmul: MatmulKernelPlan,
    pub tensor_core: TensorCoreMatmulPlan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorCoreMatmulError {
    Base(MatmulPlanError),
    Unsupported,
    Capability,
    ResourceLimit,
    InvalidPlan,
    Overflow,
    DType,
}

impl fmt::Display for TensorCoreMatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor-core matmul error: {self:?}")
    }
}
impl std::error::Error for TensorCoreMatmulError {}
impl From<MatmulPlanError> for TensorCoreMatmulError {
    fn from(value: MatmulPlanError) -> Self {
        Self::Base(value)
    }
}

impl TensorCoreMatmulPayload {
    pub fn select(
        matmul: MatmulKernelPlan,
        target: MatmulTargetCaps,
    ) -> Result<Option<Self>, TensorCoreMatmulError> {
        let Some(tensor_core) = TensorCoreMatmulPlan::select(&matmul, &target)? else {
            return Ok(None);
        };
        let payload = Self {
            matmul,
            tensor_core,
        };
        payload.validate()?;
        Ok(Some(payload))
    }

    pub fn validate(&self) -> Result<(), TensorCoreMatmulError> {
        self.matmul.validate()?;
        self.tensor_core.validate(&self.matmul)
    }

    /// Independent lane/fragment semantic model. It does not call serial or
    /// tiled matmul execution. Each logical MMA accumulator follows the CUDA
    /// m16n8k16 lane mapping and uses fused F32 accumulation before one output
    /// requantization.
    pub fn simulate(
        &self,
        lhs: &TensorData,
        rhs: &TensorData,
    ) -> Result<TensorData, TensorCoreMatmulError> {
        self.validate()?;
        if lhs.shape() != &self.matmul.lhs_shape || rhs.shape() != &self.matmul.rhs_shape {
            return Err(TensorCoreMatmulError::InvalidPlan);
        }
        if lhs.dtype() != self.tensor_core.input_dtype
            || rhs.dtype() != self.tensor_core.input_dtype
        {
            return Err(TensorCoreMatmulError::DType);
        }
        let batch_count = checked_product(&self.matmul.batch_shape)?;
        let output_len = self
            .matmul
            .output_shape
            .numel()
            .map_err(|_| TensorCoreMatmulError::Overflow)?;
        let mut output = vec![Scalar::F(0.0); output_len];
        for batch in 0..batch_count {
            let batch_coords = coords(&self.matmul.batch_shape, batch);
            for tile_row in (0..self.matmul.m).step_by(16) {
                for tile_col in (0..self.matmul.n).step_by(8) {
                    for lane in 0..32usize {
                        for accumulator_element in 0..4usize {
                            let (local_row, local_col) =
                                accumulator_coordinate(lane, accumulator_element);
                            let row = tile_row + local_row;
                            let col = tile_col + local_col;
                            let mut accumulator = 0.0f32;
                            for tile_k in (0..self.matmul.k).step_by(16) {
                                for inner in 0..16usize {
                                    let lhs_index = matrix_offset(
                                        &self.matmul.lhs_shape,
                                        &batch_coords,
                                        row,
                                        tile_k + inner,
                                    )?;
                                    let rhs_index = matrix_offset(
                                        &self.matmul.rhs_shape,
                                        &batch_coords,
                                        tile_k + inner,
                                        col,
                                    )?;
                                    let a = lhs.scalar_at(lhs_index).as_f64() as f32;
                                    let b = rhs.scalar_at(rhs_index).as_f64() as f32;
                                    accumulator = a.mul_add(b, accumulator);
                                }
                            }
                            let index = batch
                                .checked_mul(self.matmul.m)
                                .and_then(|value| value.checked_add(row))
                                .and_then(|value| value.checked_mul(self.matmul.n))
                                .and_then(|value| value.checked_add(col))
                                .ok_or(TensorCoreMatmulError::Overflow)?;
                            output[index] = Scalar::F(f64::from(accumulator));
                        }
                    }
                }
            }
        }
        TensorData::from_scalars(
            self.matmul.output_shape.clone(),
            self.tensor_core.output_dtype,
            output,
        )
        .map_err(|_| TensorCoreMatmulError::DType)
    }
}

impl TensorCoreMatmulPlan {
    pub fn enumerate(
        matmul: &MatmulKernelPlan,
        target: &MatmulTargetCaps,
    ) -> Result<Vec<Self>, TensorCoreMatmulError> {
        matmul.validate()?;
        validate_target(target)?;
        if !eligible(matmul, target) {
            return Ok(Vec::new());
        }
        Ok(vec![Self::candidate(matmul, target)?])
    }

    pub fn select(
        matmul: &MatmulKernelPlan,
        target: &MatmulTargetCaps,
    ) -> Result<Option<Self>, TensorCoreMatmulError> {
        Ok(Self::enumerate(matmul, target)?.into_iter().next())
    }

    pub fn validate(&self, matmul: &MatmulKernelPlan) -> Result<(), TensorCoreMatmulError> {
        matmul.validate()?;
        validate_target(&self.target)?;
        if !eligible(matmul, &self.target)
            || self.instruction != MmaInstruction::M16N8K16RowColF32
            || self.input_dtype != matmul.lhs_dtype
            || self.accumulator_dtype != DType::F32
            || self.output_dtype != matmul.dtype
            || self.output_policy != TensorCoreOutputPolicy::RequantizeToGraphDType
            || self.tail_policy != TensorCoreTailPolicy::ExactTilesOnly
            || self.block_m != 16
            || self.block_n != 8
            || self.block_k != 16
            || self.workgroup != [32, 1, 1]
        {
            return Err(TensorCoreMatmulError::InvalidPlan);
        }
        let expected = Self::candidate(matmul, &self.target)?;
        if self.lhs_shared != expected.lhs_shared
            || self.rhs_shared != expected.rhs_shared
            || self.fragments != expected.fragments
            || self.barriers != expected.barriers
            || self.resources != expected.resources
            || self.estimated_cost != expected.estimated_cost
            || self.cache_key != expected.cache_key
        {
            return Err(TensorCoreMatmulError::InvalidPlan);
        }
        Ok(())
    }

    pub fn launch_geometry(
        &self,
        matmul: &MatmulKernelPlan,
    ) -> Result<crate::LaunchConfig, TensorCoreMatmulError> {
        self.validate(matmul)?;
        let batch = checked_product(&matmul.batch_shape)?;
        Ok(crate::LaunchConfig {
            grid: [
                u32::try_from(matmul.n / 8).map_err(|_| TensorCoreMatmulError::Overflow)?,
                u32::try_from(matmul.m / 16).map_err(|_| TensorCoreMatmulError::Overflow)?,
                u32::try_from(batch).map_err(|_| TensorCoreMatmulError::Overflow)?,
            ],
            block: [32, 1, 1],
            shared_bytes: u32::try_from(self.resources.shared_bytes_per_block)
                .map_err(|_| TensorCoreMatmulError::Overflow)?,
        })
    }

    fn candidate(
        matmul: &MatmulKernelPlan,
        target: &MatmulTargetCaps,
    ) -> Result<Self, TensorCoreMatmulError> {
        if !eligible(matmul, target) {
            return Err(TensorCoreMatmulError::Unsupported);
        }
        let lhs_shared = shared_layout(LHS_SHARED_ID, 16, 16)?;
        let rhs_shared = shared_layout(RHS_SHARED_ID, 16, 8)?;
        let shared_bytes = lhs_shared
            .bytes
            .checked_add(rhs_shared.bytes)
            .ok_or(TensorCoreMatmulError::Overflow)?;
        let registers_per_thread = 24u32;
        if shared_bytes > target.max_shared_bytes_per_block
            || registers_per_thread > target.max_registers_per_thread
            || target.max_threads_per_block < 32
        {
            return Err(TensorCoreMatmulError::ResourceLimit);
        }
        let registers_per_block = registers_per_thread
            .checked_mul(32)
            .ok_or(TensorCoreMatmulError::Overflow)?;
        let resident_blocks = target
            .max_blocks_per_sm
            .min(target.max_threads_per_sm / 32)
            .min(u32::try_from(target.max_shared_bytes_per_sm / shared_bytes).unwrap_or(u32::MAX))
            .min(target.max_registers_per_sm / registers_per_block);
        if resident_blocks == 0 {
            return Err(TensorCoreMatmulError::ResourceLimit);
        }
        let resources = MatmulResourceEstimate {
            threads_per_block: 32,
            warps_per_block: 1,
            registers_per_thread,
            registers_per_block,
            shared_bytes_per_block: shared_bytes,
            resident_blocks_per_sm: resident_blocks,
            resident_warps_per_sm: resident_blocks,
        };
        let fragments = MmaFragmentLayout {
            lanes: 32,
            lhs_elements_per_lane: 8,
            rhs_elements_per_lane: 4,
            accumulator_elements_per_lane: 4,
            lhs_registers_per_lane: 4,
            rhs_registers_per_lane: 2,
            accumulator_registers_per_lane: 4,
        };
        let barriers = vec![
            MatmulBarrierPhase {
                sequence: 0,
                kind: MatmulBarrierKind::LoadsVisible,
                uniform: true,
                initializes: vec![LHS_SHARED_ID, RHS_SHARED_ID],
                consumes: vec![LHS_SHARED_ID, RHS_SHARED_ID],
            },
            MatmulBarrierPhase {
                sequence: 1,
                kind: MatmulBarrierKind::TileConsumed,
                uniform: true,
                initializes: Vec::new(),
                consumes: vec![LHS_SHARED_ID, RHS_SHARED_ID],
            },
        ];
        let batch = u64::try_from(checked_product(&matmul.batch_shape)?)
            .map_err(|_| TensorCoreMatmulError::Overflow)?;
        let invocations = batch
            .checked_mul(u64::try_from(matmul.m / 16).map_err(|_| TensorCoreMatmulError::Overflow)?)
            .and_then(|value| value.checked_mul(u64::try_from(matmul.n / 8).ok()?))
            .and_then(|value| value.checked_mul(u64::try_from(matmul.k / 16).ok()?))
            .ok_or(TensorCoreMatmulError::Overflow)?;
        let estimated_cost = invocations
            .checked_mul(1 + 16 * 16 + 16 * 8)
            .ok_or(TensorCoreMatmulError::Overflow)?;
        let mut plan = Self {
            target: target.clone(),
            instruction: MmaInstruction::M16N8K16RowColF32,
            input_dtype: matmul.lhs_dtype,
            accumulator_dtype: DType::F32,
            output_dtype: matmul.dtype,
            output_policy: TensorCoreOutputPolicy::RequantizeToGraphDType,
            tail_policy: TensorCoreTailPolicy::ExactTilesOnly,
            block_m: 16,
            block_n: 8,
            block_k: 16,
            workgroup: [32, 1, 1],
            lhs_shared,
            rhs_shared,
            fragments,
            barriers,
            resources,
            estimated_cost,
            cache_key: 0,
        };
        plan.cache_key = plan.expected_cache_key(matmul.cache_key);
        Ok(plan)
    }

    fn expected_cache_key(&self, matmul_key: u64) -> u64 {
        let mut copy = self.clone();
        copy.cache_key = 0;
        let mut hasher = DefaultHasher::new();
        matmul_key.hash(&mut hasher);
        copy.hash(&mut hasher);
        hasher.finish()
    }
}

fn validate_target(target: &MatmulTargetCaps) -> Result<(), TensorCoreMatmulError> {
    if !(80..=90).contains(&target.sm)
        || target.warp_size != 32
        || target.max_threads_per_block < 32
        || target.max_threads_per_sm < 32
        || target.max_shared_bytes_per_block == 0
        || target.max_shared_bytes_per_sm < target.max_shared_bytes_per_block
        || target.max_registers_per_thread == 0
        || target.max_registers_per_sm == 0
        || target.max_blocks_per_sm == 0
    {
        return Err(TensorCoreMatmulError::Capability);
    }
    Ok(())
}

fn eligible(matmul: &MatmulKernelPlan, target: &MatmulTargetCaps) -> bool {
    let dtype_supported = match matmul.lhs_dtype {
        DType::F16 => target.sm >= 80,
        DType::BF16 => target.sm >= 80,
        _ => false,
    };
    dtype_supported
        && matmul.rhs_dtype == matmul.lhs_dtype
        && matmul.dtype == matmul.lhs_dtype
        && !matmul.lhs_vector
        && !matmul.rhs_vector
        && matmul.m != 0
        && matmul.n != 0
        && matmul.k != 0
        && matmul.m.is_multiple_of(16)
        && matmul.n.is_multiple_of(8)
        && matmul.k.is_multiple_of(16)
        && matmul.batch_shape.iter().all(|dimension| *dimension != 0)
}

fn shared_layout(
    allocation_id: u32,
    rows: u32,
    columns: u32,
) -> Result<SharedTileLayout, TensorCoreMatmulError> {
    let elements = usize::try_from(rows)
        .ok()
        .and_then(|rows| {
            usize::try_from(columns)
                .ok()
                .and_then(|columns| rows.checked_mul(columns))
        })
        .ok_or(TensorCoreMatmulError::Overflow)?;
    let bytes = elements
        .checked_mul(2)
        .ok_or(TensorCoreMatmulError::Overflow)?;
    Ok(SharedTileLayout {
        allocation_id,
        rows,
        columns,
        row_stride: columns,
        bytes,
        alignment: 16,
    })
}

pub(crate) fn accumulator_coordinate(lane: usize, element: usize) -> (usize, usize) {
    let column = element % 2 + (lane % 4) * 2;
    let row = lane / 4 + (element / 2) * 8;
    (row, column)
}

fn checked_product(values: &[usize]) -> Result<usize, TensorCoreMatmulError> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(TensorCoreMatmulError::Overflow)
    })
}

fn coords(shape: &[usize], mut linear: usize) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    for (axis, dimension) in shape.iter().enumerate().rev() {
        coordinates[axis] = linear % dimension;
        linear /= dimension;
    }
    coordinates
}

fn matrix_offset(
    shape: &Shape,
    output_batch: &[usize],
    row: usize,
    column: usize,
) -> Result<usize, TensorCoreMatmulError> {
    let dimensions = shape.dims();
    let batch_rank = dimensions
        .len()
        .checked_sub(2)
        .ok_or(TensorCoreMatmulError::InvalidPlan)?;
    let pad = output_batch
        .len()
        .checked_sub(batch_rank)
        .ok_or(TensorCoreMatmulError::InvalidPlan)?;
    let mut offset = 0usize;
    for axis in 0..batch_rank {
        offset = offset
            .checked_mul(dimensions[axis])
            .and_then(|value| {
                value.checked_add(if dimensions[axis] == 1 {
                    0
                } else {
                    output_batch[axis + pad]
                })
            })
            .ok_or(TensorCoreMatmulError::Overflow)?;
    }
    offset
        .checked_mul(dimensions[batch_rank])
        .and_then(|value| value.checked_add(row))
        .and_then(|value| value.checked_mul(dimensions[batch_rank + 1]))
        .and_then(|value| value.checked_add(column))
        .ok_or(TensorCoreMatmulError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, Graph};
    use std::collections::HashMap;

    fn fixture(dtype: DType, batch: bool) -> (Graph, MatmulKernelPlan, TensorData, TensorData) {
        let mut graph = Graph::new();
        let lhs_shape = if batch { vec![2, 16, 32] } else { vec![16, 32] };
        let rhs_shape = if batch { vec![1, 32, 16] } else { vec![32, 16] };
        let lhs_node = graph.input_dtype("lhs", lhs_shape.clone(), dtype);
        let rhs_node = graph.input_dtype("rhs", rhs_shape.clone(), dtype);
        let output = graph.matmul(lhs_node, rhs_node).unwrap();
        let plan = MatmulKernelPlan::from_graph(&graph, output).unwrap();
        let lhs = TensorData::from_scalars(
            lhs_shape,
            dtype,
            (0..plan.lhs_shape.numel().unwrap()).map(|index| Scalar::F((index % 7) as f64 - 3.0)),
        )
        .unwrap();
        let rhs = TensorData::from_scalars(
            rhs_shape,
            dtype,
            (0..plan.rhs_shape.numel().unwrap()).map(|index| Scalar::F((index % 5) as f64 - 2.0)),
        )
        .unwrap();
        (graph, plan, lhs, rhs)
    }

    #[test]
    fn candidate_and_fragment_mapping_are_deterministic_and_capability_gated() {
        for dtype in [DType::F16, DType::BF16] {
            let (_, plan, _, _) = fixture(dtype, false);
            let target = MatmulTargetCaps::conservative_ptx(80).unwrap();
            let first = TensorCoreMatmulPlan::enumerate(&plan, &target).unwrap();
            let second = TensorCoreMatmulPlan::enumerate(&plan, &target).unwrap();
            assert_eq!(first, second);
            assert_eq!(first.len(), 1);
            first[0].validate(&plan).unwrap();
            assert_eq!(first[0].fragments.lhs_elements_per_lane, 8);
            assert_eq!(first[0].fragments.rhs_elements_per_lane, 4);
            let coordinates = (0..32)
                .flat_map(|lane| (0..4).map(move |element| accumulator_coordinate(lane, element)))
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(coordinates.len(), 16 * 8);
        }
        let (_, plan, _, _) = fixture(DType::F16, false);
        let sm75 = MatmulTargetCaps::conservative_ptx(75).unwrap();
        assert!(matches!(
            TensorCoreMatmulPlan::enumerate(&plan, &sm75),
            Err(TensorCoreMatmulError::Capability)
        ));
    }

    #[test]
    fn fragment_simulator_matches_cpu_on_exact_fixtures_and_batches() {
        for dtype in [DType::F16, DType::BF16] {
            for batch in [false, true] {
                let (graph, plan, lhs, rhs) = fixture(dtype, batch);
                let payload = TensorCoreMatmulPayload::select(
                    plan.clone(),
                    MatmulTargetCaps::conservative_ptx(80).unwrap(),
                )
                .unwrap()
                .unwrap();
                let simulated = payload.simulate(&lhs, &rhs).unwrap();
                let serial = plan.execute(&lhs, &rhs).unwrap();
                let cpu = CpuBackend
                    .execute(
                        &graph,
                        plan.output,
                        &HashMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]),
                    )
                    .unwrap();
                assert_eq!(
                    simulated.to_le_bytes().unwrap(),
                    serial.to_le_bytes().unwrap()
                );
                assert_eq!(simulated.to_le_bytes().unwrap(), cpu.to_le_bytes().unwrap());
            }
        }
    }

    #[test]
    fn raw_narrow_specials_are_decoded_by_the_fragment_simulator() {
        for (dtype, infinity, nan, negative_zero, subnormal) in [
            (DType::F16, 0x7c00u16, 0x7e01u16, 0x8000u16, 0x0001u16),
            (DType::BF16, 0x7f80u16, 0x7fc1u16, 0x8000u16, 0x0001u16),
        ] {
            let (_, plan, _, _) = fixture(dtype, false);
            let payload = TensorCoreMatmulPayload::select(
                plan.clone(),
                MatmulTargetCaps::conservative_ptx(80).unwrap(),
            )
            .unwrap()
            .unwrap();
            let mut lhs_bytes = vec![0; plan.lhs_shape.numel().unwrap() * 2];
            lhs_bytes[..2].copy_from_slice(&infinity.to_le_bytes());
            lhs_bytes[2..4].copy_from_slice(&negative_zero.to_le_bytes());
            lhs_bytes[4..6].copy_from_slice(&subnormal.to_le_bytes());
            let lhs = TensorData::from_le_bytes(plan.lhs_shape.clone(), dtype, &lhs_bytes).unwrap();
            assert!(lhs.scalar_at(0).as_f64().is_infinite());
            assert!(lhs.scalar_at(1).as_f64().is_sign_negative());
            assert_ne!(lhs.scalar_at(2).as_f64(), 0.0);
            let mut rhs_bytes = vec![0; plan.rhs_shape.numel().unwrap() * 2];
            let one = match dtype {
                DType::F16 => 0x3c00u16,
                DType::BF16 => 0x3f80u16,
                _ => unreachable!(),
            };
            rhs_bytes[..2].copy_from_slice(&one.to_le_bytes());
            let rhs = TensorData::from_le_bytes(plan.rhs_shape.clone(), dtype, &rhs_bytes).unwrap();
            let infinity_output = payload.simulate(&lhs, &rhs).unwrap();
            assert!(infinity_output.scalar_at(0).as_f64().is_infinite());

            lhs_bytes[..2].copy_from_slice(&nan.to_le_bytes());
            let lhs = TensorData::from_le_bytes(plan.lhs_shape.clone(), dtype, &lhs_bytes).unwrap();
            assert!(
                payload
                    .simulate(&lhs, &rhs)
                    .unwrap()
                    .scalar_at(0)
                    .as_f64()
                    .is_nan()
            );
        }
    }

    #[test]
    fn tensor_core_uop_artifact_round_trip_and_malformed_metadata_fail_closed() {
        let (graph, plan, _, _) = fixture(DType::F16, false);
        let output = plan.output;
        let kernel = crate::lower_graph_matmul(&graph, output).unwrap();
        let crate::Operation::Matmul(crate::MatmulValue::TensorCore(payload)) = kernel.operation()
        else {
            panic!("eligible narrow matmul was not tensor-core lowered");
        };
        let bytes = crate::uop::artifact::encode(&kernel).unwrap();
        let decoded = crate::uop::artifact::decode(&bytes).unwrap();
        assert_eq!(decoded, kernel);
        assert_eq!(crate::uop::artifact::encode(&decoded).unwrap(), bytes);

        let mut bad_fragment = payload.clone();
        bad_fragment.tensor_core.fragments.lhs_registers_per_lane = 3;
        let bad_fragment = crate::UOp::from_operation(
            crate::Operation::Matmul(crate::MatmulValue::TensorCore(bad_fragment)),
            Some(crate::UType::scalar(DType::F16)),
            vec![],
        );
        assert!(bad_fragment.validate().is_err());
        assert!(crate::uop::artifact::encode(&bad_fragment).is_err());

        let mut bad_barrier = payload.clone();
        bad_barrier.tensor_core.barriers[0].uniform = false;
        let bad_barrier = crate::UOp::from_operation(
            crate::Operation::Matmul(crate::MatmulValue::TensorCore(bad_barrier)),
            Some(crate::UType::scalar(DType::F16)),
            vec![],
        );
        assert!(bad_barrier.validate().is_err());
        assert!(crate::uop::artifact::encode(&bad_barrier).is_err());

        let mut bad_layout = payload.clone();
        bad_layout.tensor_core.lhs_shared.alignment = 2;
        let bad_layout = crate::UOp::from_operation(
            crate::Operation::Matmul(crate::MatmulValue::TensorCore(bad_layout)),
            Some(crate::UType::scalar(DType::F16)),
            vec![],
        );
        assert!(bad_layout.validate().is_err());
        assert!(crate::uop::artifact::encode(&bad_layout).is_err());
    }

    #[test]
    fn symbolic_specialization_reselects_tensor_core_and_separates_identities() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [16, 16], DType::F16);
        let rhs = graph.input_dtype("rhs", [16, 8], DType::F16);
        let output = graph.matmul(lhs, rhs).unwrap();
        let m = crate::SymbolicExpr::variable("m", 16, 32).unwrap();
        let n = crate::SymbolicExpr::variable("n", 8, 16).unwrap();
        let spec = crate::SymbolicCaptureSpec::new(std::collections::BTreeMap::from([
            (
                lhs,
                crate::SymbolicShape::new(vec![m.clone().into(), 16usize.into()]),
            ),
            (
                rhs,
                crate::SymbolicShape::new(vec![16usize.into(), n.clone().into()]),
            ),
        ]))
        .with_guard(crate::SymbolicGuard::divisible(m, 16).unwrap())
        .with_guard(crate::SymbolicGuard::divisible(n, 8).unwrap());
        let schedule = crate::schedule(&graph, output).unwrap();
        let capture = crate::CapturedSchedule::capture_symbolic(
            &graph,
            &schedule,
            &[output],
            &spec,
            &std::collections::BTreeMap::from([("m".into(), 16), ("n".into(), 8)]),
        )
        .unwrap();
        let bytes = capture.to_bytes().unwrap();
        let decoded = crate::CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
        let executor = crate::CapturedReplayExecutor::default();
        let first = executor
            .specialize(
                &decoded,
                &std::collections::BTreeMap::from([("m".into(), 16), ("n".into(), 8)]),
            )
            .unwrap();
        let repeated = executor
            .specialize(
                &decoded,
                &std::collections::BTreeMap::from([("m".into(), 16), ("n".into(), 8)]),
            )
            .unwrap();
        let larger = executor
            .specialize(
                &decoded,
                &std::collections::BTreeMap::from([("m".into(), 32), ("n".into(), 16)]),
            )
            .unwrap();
        assert!(!first.trace().cache_hit);
        assert!(repeated.trace().cache_hit);
        assert_ne!(
            first.trace().concrete_identity,
            larger.trace().concrete_identity
        );
        for specialized in [first.capture(), larger.capture()] {
            assert!(matches!(
                specialized.items[0].kernel.operation(),
                crate::Operation::Matmul(crate::MatmulValue::TensorCore(_))
            ));
        }
    }

    #[test]
    fn captured_tensor_core_schedule_round_trips_and_interpreter_replays() {
        let (graph, plan, lhs, rhs) = fixture(DType::BF16, true);
        let schedule = crate::schedule(&graph, plan.output).unwrap();
        let captured = crate::CapturedSchedule::capture(&graph, &schedule, &[plan.output]).unwrap();
        let bytes = captured.to_bytes().unwrap();
        let decoded = crate::CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
        assert!(matches!(
            decoded.items[0].kernel.operation(),
            crate::Operation::Matmul(crate::MatmulValue::TensorCore(_))
        ));
        let replayed = crate::CapturedReplayExecutor::default()
            .replay(
                &decoded,
                &std::collections::BTreeMap::from([
                    ("lhs".into(), lhs.clone()),
                    ("rhs".into(), rhs.clone()),
                ]),
                crate::CapturedReplayOptions::default(),
            )
            .unwrap();
        let expected = plan.execute(&lhs, &rhs).unwrap();
        assert_eq!(
            replayed.outputs[0].to_le_bytes().unwrap(),
            expected.to_le_bytes().unwrap()
        );
    }

    #[test]
    fn tails_vectors_zero_and_resource_limits_are_explicit_fallbacks() {
        for (lhs_shape, rhs_shape) in [
            (vec![15, 16], vec![16, 8]),
            (vec![16, 15], vec![15, 8]),
            (vec![16, 16], vec![16, 7]),
            (vec![16], vec![16, 8]),
            (vec![16, 0], vec![0, 8]),
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", lhs_shape, DType::F16);
            let rhs = graph.input_dtype("rhs", rhs_shape, DType::F16);
            let output = graph.matmul(lhs, rhs).unwrap();
            let plan = MatmulKernelPlan::from_graph(&graph, output).unwrap();
            assert!(matches!(
                crate::lower_graph_matmul(&graph, output)
                    .unwrap()
                    .operation(),
                crate::Operation::Matmul(crate::MatmulValue::Serial(_))
            ));
            assert!(
                TensorCoreMatmulPayload::select(
                    plan,
                    MatmulTargetCaps::conservative_ptx(80).unwrap()
                )
                .unwrap()
                .is_none()
            );
        }
        let (_, plan, _, _) = fixture(DType::F16, false);
        let mut target = MatmulTargetCaps::conservative_ptx(80).unwrap();
        target.max_shared_bytes_per_block = 128;
        assert!(matches!(
            TensorCoreMatmulPlan::enumerate(&plan, &target),
            Err(TensorCoreMatmulError::ResourceLimit)
        ));
    }
}
