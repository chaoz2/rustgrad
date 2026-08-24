//! Shared-memory promotion derived solely from a validated tiled payload.
use super::{
    BarrierPoint, BarrierScope, GlobalAccess, MemorySpace, MemorySpaceError, MemorySpacePlan,
    PromotionDecision, RegisterBinding, SpaceAllocation,
};
use crate::{DType, TiledMatmulPayload};

pub fn plan_tiled_matmul_promotion(
    payload: &TiledMatmulPayload,
) -> Result<MemorySpacePlan, MemorySpaceError> {
    payload
        .validate()
        .map_err(|_| MemorySpaceError::InvalidTiledMatmul)?;
    let matmul = &payload.matmul;
    let tile = &payload.tile;
    let bytes = |shape: &crate::Shape, dtype: DType| {
        shape
            .numel()
            .ok()
            .and_then(|elements| elements.checked_mul(dtype.itemsize()))
            .ok_or(MemorySpaceError::Overflow)
    };
    let lhs = SpaceAllocation {
        id: tile.lhs_shared.allocation_id,
        space: MemorySpace::Shared,
        bytes: tile.lhs_shared.bytes,
        alignment: tile.lhs_shared.alignment,
        start: 0,
        end: 2,
    };
    let rhs = SpaceAllocation {
        id: tile.rhs_shared.allocation_id,
        space: MemorySpace::Shared,
        bytes: tile.rhs_shared.bytes,
        alignment: tile.rhs_shared.alignment,
        start: 0,
        end: 2,
    };
    let barriers = tile
        .barriers
        .iter()
        .map(|phase| BarrierPoint {
            instruction: phase.sequence + 1,
            scope: BarrierScope::Workgroup,
            uniform_control: phase.uniform,
            initializes: phase.initializes.clone(),
            consumes: phase.consumes.clone(),
        })
        .collect::<Vec<_>>();
    let mut plan = MemorySpacePlan {
        registers: vec![RegisterBinding {
            virtual_reg: 0,
            physical_reg: 0,
            space: MemorySpace::RegisterScalar,
            dtype: DType::F64,
            start: 0,
            end: 2,
        }],
        globals: vec![
            GlobalAccess {
                buffer: matmul.lhs.index() as u64,
                bytes: bytes(&matmul.lhs_shape, matmul.lhs_dtype)?,
                byte_offset: 0,
                alignment: matmul.lhs_dtype.itemsize(),
                mutable: false,
            },
            GlobalAccess {
                buffer: matmul.rhs.index() as u64,
                bytes: bytes(&matmul.rhs_shape, matmul.rhs_dtype)?,
                byte_offset: 0,
                alignment: matmul.rhs_dtype.itemsize(),
                mutable: false,
            },
            GlobalAccess {
                buffer: matmul.output.index() as u64,
                bytes: bytes(&matmul.output_shape, matmul.dtype)?,
                byte_offset: 0,
                alignment: matmul.dtype.itemsize(),
                mutable: true,
            },
        ],
        private: Vec::new(),
        shared: vec![lhs.clone(), rhs.clone()],
        barriers,
        promotions: vec![
            PromotionDecision::Shared { allocation: lhs },
            PromotionDecision::Shared { allocation: rhs },
        ],
        cache_key: 0,
    };
    plan.globals.sort_by_key(|global| global.buffer);
    plan.rekey();
    plan.validate()?;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, MatmulKernelPlan, MatmulTargetCaps, TiledMatmulPayload};

    #[test]
    fn promotion_matches_tiled_shared_and_barrier_contract() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [17, 9], DType::F32);
        let rhs = graph.input_dtype("rhs", [9, 13], DType::F32);
        let output = graph.matmul(lhs, rhs).unwrap();
        let payload = TiledMatmulPayload::select(
            MatmulKernelPlan::from_graph(&graph, output).unwrap(),
            MatmulTargetCaps::conservative_ptx(80).unwrap(),
        )
        .unwrap()
        .unwrap();
        let first = plan_tiled_matmul_promotion(&payload).unwrap();
        let second = plan_tiled_matmul_promotion(&payload).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.shared.len(), 2);
        assert_eq!(first.barriers.len(), 2);
        assert_eq!(
            first.shared.iter().map(|value| value.bytes).sum::<usize>(),
            payload.tile.resources.shared_bytes_per_block
        );
        assert!(first.barriers.iter().all(|barrier| barrier.uniform_control));
    }
}
