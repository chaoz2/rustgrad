//! Deterministic backend-neutral tiling for homogeneous F32 matrix matmul.
use super::{MatmulKernelPlan, MatmulPlanError};
use crate::{DType, Scalar, Shape, TensorData};
use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
};

const LHS_SHARED_ID: u32 = 1;
const RHS_SHARED_ID: u32 = 2;
const CANONICAL_TILES: &[(u32, u32, u32)] = &[
    (8, 8, 8),
    (8, 16, 8),
    (16, 8, 8),
    (16, 16, 8),
    (8, 8, 16),
    (8, 16, 16),
    (16, 8, 16),
];

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MatmulTargetCaps {
    pub sm: u32,
    pub warp_size: u32,
    pub max_threads_per_block: u32,
    pub max_threads_per_sm: u32,
    pub max_shared_bytes_per_block: usize,
    pub max_shared_bytes_per_sm: usize,
    pub max_registers_per_thread: u32,
    pub max_registers_per_sm: u32,
    pub max_blocks_per_sm: u32,
}

impl MatmulTargetCaps {
    /// Conservative CUDA capabilities used by portable schedule selection.
    pub fn conservative_ptx(sm: u32) -> Result<Self, TiledMatmulError> {
        if !(20..=90).contains(&sm) {
            return Err(TiledMatmulError::Capability);
        }
        Ok(Self {
            sm,
            warp_size: 32,
            max_threads_per_block: 256,
            max_threads_per_sm: 2_048,
            max_shared_bytes_per_block: 48 * 1_024,
            max_shared_bytes_per_sm: 64 * 1_024,
            max_registers_per_thread: 64,
            max_registers_per_sm: 65_536,
            max_blocks_per_sm: 16,
        })
    }

    fn validate(&self) -> Result<(), TiledMatmulError> {
        if !(20..=90).contains(&self.sm)
            || self.warp_size == 0
            || !self.warp_size.is_power_of_two()
            || self.max_threads_per_block == 0
            || self.max_threads_per_sm < self.max_threads_per_block
            || self.max_shared_bytes_per_block == 0
            || self.max_shared_bytes_per_sm < self.max_shared_bytes_per_block
            || self.max_registers_per_thread == 0
            || self.max_registers_per_sm == 0
            || self.max_blocks_per_sm == 0
        {
            return Err(TiledMatmulError::Capability);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SharedTileLayout {
    pub allocation_id: u32,
    pub rows: u32,
    pub columns: u32,
    pub row_stride: u32,
    pub bytes: usize,
    pub alignment: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MatmulBarrierKind {
    LoadsVisible,
    TileConsumed,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MatmulBarrierPhase {
    pub sequence: u32,
    pub kind: MatmulBarrierKind,
    pub uniform: bool,
    pub initializes: Vec<u32>,
    pub consumes: Vec<u32>,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TiledMatmulTails {
    pub m: bool,
    pub n: bool,
    pub k: bool,
    pub broadcast_batch: bool,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MatmulResourceEstimate {
    pub threads_per_block: u32,
    pub warps_per_block: u32,
    pub registers_per_thread: u32,
    pub registers_per_block: u32,
    pub shared_bytes_per_block: usize,
    pub resident_blocks_per_sm: u32,
    pub resident_warps_per_sm: u32,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TiledMatmulPlan {
    pub target: MatmulTargetCaps,
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub workgroup: [u32; 3],
    pub register_tile: [u32; 2],
    pub vector_width: u32,
    pub lhs_shared: SharedTileLayout,
    pub rhs_shared: SharedTileLayout,
    pub tails: TiledMatmulTails,
    pub barriers: Vec<MatmulBarrierPhase>,
    pub resources: MatmulResourceEstimate,
    /// Static work units: shared scalar loads plus padded scalar FMAs plus two
    /// barrier arrivals per thread and K tile. This is not a timing estimate.
    pub estimated_cost: u64,
    pub cache_key: u64,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TiledMatmulPayload {
    pub matmul: MatmulKernelPlan,
    pub tile: TiledMatmulPlan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TiledMatmulError {
    Base(MatmulPlanError),
    Unsupported,
    Capability,
    ResourceLimit,
    InvalidPlan,
    Overflow,
    DType,
}

impl fmt::Display for TiledMatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tiled matmul error: {self:?}")
    }
}
impl std::error::Error for TiledMatmulError {}
impl From<MatmulPlanError> for TiledMatmulError {
    fn from(value: MatmulPlanError) -> Self {
        Self::Base(value)
    }
}

impl TiledMatmulPayload {
    pub fn select(
        matmul: MatmulKernelPlan,
        target: MatmulTargetCaps,
    ) -> Result<Option<Self>, TiledMatmulError> {
        let Some(tile) = TiledMatmulPlan::select(&matmul, &target)? else {
            return Ok(None);
        };
        let payload = Self { matmul, tile };
        payload.validate()?;
        Ok(Some(payload))
    }

    pub fn validate(&self) -> Result<(), TiledMatmulError> {
        self.matmul.validate()?;
        self.tile.validate(&self.matmul)
    }

    /// Executes the retained tiled order without calling the serial matmul
    /// implementation. This is the semantic model used by mock PTX dispatch.
    pub fn simulate(
        &self,
        lhs: &TensorData,
        rhs: &TensorData,
    ) -> Result<TensorData, TiledMatmulError> {
        self.validate()?;
        if lhs.shape() != &self.matmul.lhs_shape || rhs.shape() != &self.matmul.rhs_shape {
            return Err(TiledMatmulError::InvalidPlan);
        }
        if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 {
            return Err(TiledMatmulError::DType);
        }
        let batch_count = checked_product(&self.matmul.batch_shape)?;
        let output_len = self
            .matmul
            .output_shape
            .numel()
            .map_err(|_| TiledMatmulError::Overflow)?;
        let mut output = vec![Scalar::F(0.0); output_len];
        let bm = self.tile.block_m as usize;
        let bn = self.tile.block_n as usize;
        let bk = self.tile.block_k as usize;
        for batch in 0..batch_count {
            let batch_coords = coords(&self.matmul.batch_shape, batch);
            for tile_row in (0..self.matmul.m).step_by(bm) {
                for tile_col in (0..self.matmul.n).step_by(bn) {
                    for local_row in 0..bm {
                        let row = tile_row + local_row;
                        if row >= self.matmul.m {
                            continue;
                        }
                        for local_col in 0..bn {
                            let col = tile_col + local_col;
                            if col >= self.matmul.n {
                                continue;
                            }
                            let mut accumulator = 0.0f64;
                            for tile_k in (0..self.matmul.k).step_by(bk) {
                                for local_k in 0..bk {
                                    let inner = tile_k + local_k;
                                    if inner >= self.matmul.k {
                                        continue;
                                    }
                                    let lhs_index = matrix_offset(
                                        &self.matmul.lhs_shape,
                                        &batch_coords,
                                        row,
                                        inner,
                                    )?;
                                    let rhs_index = matrix_offset(
                                        &self.matmul.rhs_shape,
                                        &batch_coords,
                                        inner,
                                        col,
                                    )?;
                                    accumulator += lhs.scalar_at(lhs_index).as_f64()
                                        * rhs.scalar_at(rhs_index).as_f64();
                                }
                            }
                            let index = batch
                                .checked_mul(self.matmul.m)
                                .and_then(|value| value.checked_add(row))
                                .and_then(|value| value.checked_mul(self.matmul.n))
                                .and_then(|value| value.checked_add(col))
                                .ok_or(TiledMatmulError::Overflow)?;
                            output[index] = Scalar::F(accumulator);
                        }
                    }
                }
            }
        }
        TensorData::from_scalars(self.matmul.output_shape.clone(), DType::F32, output)
            .map_err(|_| TiledMatmulError::DType)
    }
}

impl TiledMatmulPlan {
    /// Enumerates the fixed portable candidate set and orders it by the static
    /// cost, then occupancy, then tile dimensions for deterministic ties.
    pub fn enumerate(
        matmul: &MatmulKernelPlan,
        target: &MatmulTargetCaps,
    ) -> Result<Vec<Self>, TiledMatmulError> {
        matmul.validate()?;
        target.validate()?;
        if !eligible(matmul) {
            return Ok(Vec::new());
        }
        let mut candidates = Vec::new();
        for &(block_m, block_n, block_k) in CANONICAL_TILES {
            match Self::candidate(matmul, target, block_m, block_n, block_k) {
                Ok(candidate) => candidates.push(candidate),
                // Candidate-local hardware pressure is an expected scalar
                // fallback. Arithmetic or contract failures must not silently
                // produce a partial ordered candidate set/cache identity.
                Err(TiledMatmulError::ResourceLimit) => {}
                Err(error) => return Err(error),
            }
        }
        candidates.sort_by_key(|candidate| {
            (
                candidate.estimated_cost,
                std::cmp::Reverse(candidate.resources.resident_warps_per_sm),
                candidate.block_m,
                candidate.block_n,
                candidate.block_k,
            )
        });
        Ok(candidates)
    }

    pub fn select(
        matmul: &MatmulKernelPlan,
        target: &MatmulTargetCaps,
    ) -> Result<Option<Self>, TiledMatmulError> {
        Ok(Self::enumerate(matmul, target)?.into_iter().next())
    }

    pub fn validate(&self, matmul: &MatmulKernelPlan) -> Result<(), TiledMatmulError> {
        matmul.validate()?;
        self.target.validate()?;
        if !eligible(matmul)
            || self.block_m == 0
            || self.block_n == 0
            || self.block_k == 0
            || !CANONICAL_TILES.contains(&(self.block_m, self.block_n, self.block_k))
            || self.register_tile != [1, 1]
            || self.vector_width != 1
            || self.workgroup != [self.block_n, self.block_m, 1]
        {
            return Err(TiledMatmulError::InvalidPlan);
        }
        let expected = Self::candidate(
            matmul,
            &self.target,
            self.block_m,
            self.block_n,
            self.block_k,
        )?;
        if self.lhs_shared != expected.lhs_shared
            || self.rhs_shared != expected.rhs_shared
            || self.tails != expected.tails
            || self.barriers != expected.barriers
            || self.resources != expected.resources
            || self.estimated_cost != expected.estimated_cost
            || self.cache_key != expected.cache_key
        {
            return Err(TiledMatmulError::InvalidPlan);
        }
        Ok(())
    }

    pub fn launch_geometry(
        &self,
        matmul: &MatmulKernelPlan,
    ) -> Result<crate::LaunchConfig, TiledMatmulError> {
        self.validate(matmul)?;
        let grid_x = ceil_div(matmul.n, self.block_n as usize)?;
        let grid_y = ceil_div(matmul.m, self.block_m as usize)?;
        let grid_z = checked_product(&matmul.batch_shape)?;
        Ok(crate::LaunchConfig {
            grid: [
                u32::try_from(grid_x).map_err(|_| TiledMatmulError::Overflow)?,
                u32::try_from(grid_y).map_err(|_| TiledMatmulError::Overflow)?,
                u32::try_from(grid_z).map_err(|_| TiledMatmulError::Overflow)?,
            ],
            block: self.workgroup,
            shared_bytes: u32::try_from(self.resources.shared_bytes_per_block)
                .map_err(|_| TiledMatmulError::Overflow)?,
        })
    }

    fn candidate(
        matmul: &MatmulKernelPlan,
        target: &MatmulTargetCaps,
        block_m: u32,
        block_n: u32,
        block_k: u32,
    ) -> Result<Self, TiledMatmulError> {
        let threads = block_m
            .checked_mul(block_n)
            .ok_or(TiledMatmulError::Overflow)?;
        if threads > target.max_threads_per_block {
            return Err(TiledMatmulError::ResourceLimit);
        }
        let lhs_shared = shared_layout(LHS_SHARED_ID, block_m, block_k)?;
        let rhs_shared = shared_layout(RHS_SHARED_ID, block_k, block_n)?;
        let shared_bytes = lhs_shared
            .bytes
            .checked_add(rhs_shared.bytes)
            .ok_or(TiledMatmulError::Overflow)?;
        let registers_per_thread = 16u32;
        if shared_bytes > target.max_shared_bytes_per_block
            || registers_per_thread > target.max_registers_per_thread
        {
            return Err(TiledMatmulError::ResourceLimit);
        }
        let registers_per_block = registers_per_thread
            .checked_mul(threads)
            .ok_or(TiledMatmulError::Overflow)?;
        let by_threads = target.max_threads_per_sm / threads;
        let by_shared = target.max_shared_bytes_per_sm / shared_bytes;
        let by_registers = target.max_registers_per_sm / registers_per_block;
        let resident_blocks = target
            .max_blocks_per_sm
            .min(by_threads)
            .min(u32::try_from(by_shared).unwrap_or(u32::MAX))
            .min(by_registers);
        if resident_blocks == 0 {
            return Err(TiledMatmulError::ResourceLimit);
        }
        let warps = threads
            .checked_add(target.warp_size - 1)
            .ok_or(TiledMatmulError::Overflow)?
            / target.warp_size;
        let resources = MatmulResourceEstimate {
            threads_per_block: threads,
            warps_per_block: warps,
            registers_per_thread,
            registers_per_block,
            shared_bytes_per_block: shared_bytes,
            resident_blocks_per_sm: resident_blocks,
            resident_warps_per_sm: resident_blocks
                .checked_mul(warps)
                .ok_or(TiledMatmulError::Overflow)?,
        };
        let tails = TiledMatmulTails {
            m: matmul.m % block_m as usize != 0,
            n: matmul.n % block_n as usize != 0,
            k: matmul.k % block_k as usize != 0,
            broadcast_batch: has_broadcast_batch(matmul),
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
        let estimated_cost = estimate_cost(matmul, block_m, block_n, block_k, threads)?;
        let mut plan = Self {
            target: target.clone(),
            block_m,
            block_n,
            block_k,
            workgroup: [block_n, block_m, 1],
            register_tile: [1, 1],
            vector_width: 1,
            lhs_shared,
            rhs_shared,
            tails,
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

fn eligible(matmul: &MatmulKernelPlan) -> bool {
    matmul.lhs_dtype == DType::F32
        && matmul.rhs_dtype == DType::F32
        && matmul.dtype == DType::F32
        && !matmul.lhs_vector
        && !matmul.rhs_vector
        && matmul.m != 0
        && matmul.n != 0
        && matmul.k != 0
        && matmul.batch_shape.iter().all(|dimension| *dimension != 0)
}

fn shared_layout(
    allocation_id: u32,
    rows: u32,
    columns: u32,
) -> Result<SharedTileLayout, TiledMatmulError> {
    let elements = usize::try_from(rows)
        .ok()
        .and_then(|rows| {
            usize::try_from(columns)
                .ok()
                .and_then(|columns| rows.checked_mul(columns))
        })
        .ok_or(TiledMatmulError::Overflow)?;
    let bytes = elements
        .checked_mul(DType::F32.itemsize())
        .ok_or(TiledMatmulError::Overflow)?;
    Ok(SharedTileLayout {
        allocation_id,
        rows,
        columns,
        row_stride: columns,
        bytes,
        alignment: 16,
    })
}

fn estimate_cost(
    matmul: &MatmulKernelPlan,
    block_m: u32,
    block_n: u32,
    block_k: u32,
    threads: u32,
) -> Result<u64, TiledMatmulError> {
    let batch = u64::try_from(checked_product(&matmul.batch_shape)?)
        .map_err(|_| TiledMatmulError::Overflow)?;
    let tiles_m = u64::try_from(ceil_div(matmul.m, block_m as usize)?)
        .map_err(|_| TiledMatmulError::Overflow)?;
    let tiles_n = u64::try_from(ceil_div(matmul.n, block_n as usize)?)
        .map_err(|_| TiledMatmulError::Overflow)?;
    let tiles_k = u64::try_from(ceil_div(matmul.k, block_k as usize)?)
        .map_err(|_| TiledMatmulError::Overflow)?;
    let blocks = batch
        .checked_mul(tiles_m)
        .and_then(|value| value.checked_mul(tiles_n))
        .ok_or(TiledMatmulError::Overflow)?;
    let loads_per_k = u64::from(block_m)
        .checked_mul(u64::from(block_k))
        .and_then(|lhs| {
            u64::from(block_k)
                .checked_mul(u64::from(block_n))
                .and_then(|rhs| lhs.checked_add(rhs))
        })
        .ok_or(TiledMatmulError::Overflow)?;
    let padded_fma = u64::from(block_m)
        .checked_mul(u64::from(block_n))
        .and_then(|value| value.checked_mul(u64::from(block_k)))
        .ok_or(TiledMatmulError::Overflow)?;
    blocks
        .checked_mul(tiles_k)
        .and_then(|value| value.checked_mul(loads_per_k.checked_add(padded_fma)?))
        .and_then(|value| {
            blocks
                .checked_mul(tiles_k)?
                .checked_mul(u64::from(threads))?
                .checked_mul(2)?
                .checked_add(value)
        })
        .ok_or(TiledMatmulError::Overflow)
}

fn has_broadcast_batch(matmul: &MatmulKernelPlan) -> bool {
    [&matmul.lhs_shape, &matmul.rhs_shape]
        .into_iter()
        .any(|shape| {
            let batch = &shape.dims()[..shape.rank() - 2];
            let pad = matmul.batch_shape.len() - batch.len();
            batch
                .iter()
                .enumerate()
                .any(|(axis, dimension)| *dimension == 1 && matmul.batch_shape[axis + pad] != 1)
        })
}

fn checked_product(values: &[usize]) -> Result<usize, TiledMatmulError> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(TiledMatmulError::Overflow)
    })
}

fn ceil_div(value: usize, divisor: usize) -> Result<usize, TiledMatmulError> {
    value
        .checked_add(
            divisor
                .checked_sub(1)
                .ok_or(TiledMatmulError::InvalidPlan)?,
        )
        .ok_or(TiledMatmulError::Overflow)
        .map(|value| value / divisor)
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
) -> Result<usize, TiledMatmulError> {
    let dimensions = shape.dims();
    let batch_rank = dimensions
        .len()
        .checked_sub(2)
        .ok_or(TiledMatmulError::InvalidPlan)?;
    let pad = output_batch
        .len()
        .checked_sub(batch_rank)
        .ok_or(TiledMatmulError::InvalidPlan)?;
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
            .ok_or(TiledMatmulError::Overflow)?;
    }
    offset
        .checked_mul(dimensions[batch_rank])
        .and_then(|value| value.checked_add(row))
        .and_then(|value| value.checked_mul(dimensions[batch_rank + 1]))
        .and_then(|value| value.checked_add(column))
        .ok_or(TiledMatmulError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, Graph};
    use std::collections::HashMap;

    #[test]
    fn candidates_are_deterministic_and_resource_checked() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [19, 13], DType::F32);
        let rhs = graph.input_dtype("rhs", [13, 21], DType::F32);
        let output = graph.matmul(lhs, rhs).unwrap();
        let base = MatmulKernelPlan::from_graph(&graph, output).unwrap();
        let target = MatmulTargetCaps::conservative_ptx(80).unwrap();
        let first = TiledMatmulPlan::enumerate(&base, &target).unwrap();
        let second = TiledMatmulPlan::enumerate(&base, &target).unwrap();
        assert_eq!(first, second);
        assert!(!first.is_empty());
        assert!(
            first
                .windows(2)
                .all(|pair| pair[0].estimated_cost <= pair[1].estimated_cost)
        );
        first[0].validate(&base).unwrap();
        let other_target = MatmulTargetCaps::conservative_ptx(90).unwrap();
        let other = TiledMatmulPlan::select(&base, &other_target)
            .unwrap()
            .unwrap();
        assert_ne!(first[0].cache_key, other.cache_key);

        let mut tiny = target;
        tiny.max_threads_per_block = 32;
        tiny.max_threads_per_sm = 32;
        assert!(TiledMatmulPlan::enumerate(&base, &tiny).unwrap().is_empty());

        let mut malformed = first[0].clone();
        malformed.target.max_shared_bytes_per_block = 16;
        assert!(matches!(
            malformed.validate(&base),
            Err(TiledMatmulError::ResourceLimit | TiledMatmulError::InvalidPlan)
        ));
    }

    #[test]
    fn tiled_simulator_matches_serial_and_cpu_for_tails_and_broadcast() {
        let mut graph = Graph::new();
        let lhs_node = graph.input_dtype("lhs", [2, 1, 9, 7], DType::F32);
        let rhs_node = graph.input_dtype("rhs", [1, 3, 7, 11], DType::F32);
        let output = graph.matmul(lhs_node, rhs_node).unwrap();
        let base = MatmulKernelPlan::from_graph(&graph, output).unwrap();
        let payload = TiledMatmulPayload::select(
            base.clone(),
            MatmulTargetCaps::conservative_ptx(80).unwrap(),
        )
        .unwrap()
        .unwrap();
        assert!(payload.tile.tails.m && payload.tile.tails.n && payload.tile.tails.k);
        assert!(payload.tile.tails.broadcast_batch);
        let lhs = TensorData::from_scalars(
            base.lhs_shape.clone(),
            DType::F32,
            (0..base.lhs_shape.numel().unwrap()).map(|index| Scalar::F(index as f64 * 0.125 - 3.0)),
        )
        .unwrap();
        let rhs = TensorData::from_scalars(
            base.rhs_shape.clone(),
            DType::F32,
            (0..base.rhs_shape.numel().unwrap())
                .map(|index| Scalar::F(index as f64 * -0.0625 + 1.0)),
        )
        .unwrap();
        let tiled = payload.simulate(&lhs, &rhs).unwrap();
        let serial = base.execute(&lhs, &rhs).unwrap();
        let cpu = CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([("lhs".into(), lhs), ("rhs".into(), rhs)]),
            )
            .unwrap();
        assert_eq!(tiled.storage(), serial.storage());
        assert_eq!(tiled.storage(), cpu.storage());
    }

    #[test]
    fn vectors_zero_and_non_f32_route_to_serial() {
        for (lhs_shape, rhs_shape, dtype) in [
            (vec![4], vec![4, 3], DType::F32),
            (vec![2, 0], vec![0, 3], DType::F32),
            (vec![0, 2, 4], vec![0, 4, 3], DType::F32),
            (vec![2, 4], vec![4, 3], DType::F64),
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", lhs_shape, dtype);
            let rhs = graph.input_dtype("rhs", rhs_shape, dtype);
            let output = graph.matmul(lhs, rhs).unwrap();
            let base = MatmulKernelPlan::from_graph(&graph, output).unwrap();
            assert!(
                TiledMatmulPayload::select(base, MatmulTargetCaps::conservative_ptx(80).unwrap())
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn candidate_enumeration_propagates_cost_overflow_without_partial_plan() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [4_000_000, 4_000_000], DType::F32);
        let rhs = graph.input_dtype("rhs", [4_000_000, 4_000_000], DType::F32);
        let output = graph.matmul(lhs, rhs).unwrap();
        let base = MatmulKernelPlan::from_graph(&graph, output).unwrap();
        assert_eq!(
            TiledMatmulPlan::enumerate(&base, &MatmulTargetCaps::conservative_ptx(80).unwrap()),
            Err(TiledMatmulError::Overflow)
        );
    }

    #[test]
    fn validation_rejects_off_policy_candidate_before_artifact_identity() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [8, 8], DType::F32);
        let rhs = graph.input_dtype("rhs", [8, 8], DType::F32);
        let output = graph.matmul(lhs, rhs).unwrap();
        let base = MatmulKernelPlan::from_graph(&graph, output).unwrap();
        let target = MatmulTargetCaps::conservative_ptx(80).unwrap();
        let off_policy = TiledMatmulPlan::candidate(&base, &target, 1, 1, 1).unwrap();
        assert!(matches!(
            off_policy.validate(&base),
            Err(TiledMatmulError::InvalidPlan)
        ));
        let kernel = crate::UOp::try_new(
            crate::UOpKind::Matmul,
            Some(crate::UType::scalar(DType::F32)),
            vec![],
            crate::UArg::TiledMatmul(Box::new(TiledMatmulPayload {
                matmul: base,
                tile: off_policy,
            })),
        )
        .unwrap();
        assert!(kernel.validate().is_err());
        assert!(crate::uop::artifact::encode(&kernel).is_err());
    }
}
