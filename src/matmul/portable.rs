//! Checked backend-neutral projection for correctness-first portable F32 matmul.
use super::MatmulKernelPlan;
use crate::{DType, MatmulValue, NodeId, ScheduleInputBinding, Shape};
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PortableF32MatmulError {
    InvalidPlan(String),
    InvalidBinding(String),
    Unsupported(&'static str),
    Overflow,
}

impl fmt::Display for PortableF32MatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(reason) => write!(f, "invalid matmul payload: {reason}"),
            Self::InvalidBinding(reason) => write!(f, "invalid matmul binding: {reason}"),
            Self::Unsupported(reason) => write!(f, "unsupported matmul: {reason}"),
            Self::Overflow => f.write_str("matmul geometry overflow"),
        }
    }
}

impl std::error::Error for PortableF32MatmulError {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PortableBatchAxis {
    pub(crate) output_axis: usize,
    pub(crate) divisor: usize,
    pub(crate) dimension: usize,
    pub(crate) input_stride: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PortableMatmulInput<'a> {
    pub(crate) node: NodeId,
    pub(crate) shape: &'a Shape,
}

/// One fully checked logical matmul projected from an eligible Serial or Tiled
/// payload.
/// Tiled payloads are authenticated before their base plan is exposed;
/// portable renderers intentionally retain only the base serial F32
/// recurrence and dense pointer ABI. Tensor-core and quantized payloads are
/// authenticated and then fail closed because their selected execution
/// contracts are not interchangeable with portable scalar arithmetic.
pub(crate) struct PortableF32Matmul<'a> {
    value: &'a MatmulValue,
    plan: &'a MatmulKernelPlan,
    extent: usize,
    lhs_elements: usize,
    rhs_elements: usize,
    lhs_batch_axes: Vec<PortableBatchAxis>,
    rhs_batch_axes: Vec<PortableBatchAxis>,
}

impl<'a> PortableF32Matmul<'a> {
    pub(crate) fn new(value: &'a MatmulValue) -> Result<Self, PortableF32MatmulError> {
        let plan = match value {
            MatmulValue::Serial(plan) => {
                plan.validate()
                    .map_err(|error| PortableF32MatmulError::InvalidPlan(error.to_string()))?;
                plan.as_ref()
            }
            MatmulValue::Tiled(payload) => {
                payload
                    .validate()
                    .map_err(|error| PortableF32MatmulError::InvalidPlan(error.to_string()))?;
                &payload.matmul
            }
            MatmulValue::TensorCore(payload) => {
                payload
                    .validate()
                    .map_err(|error| PortableF32MatmulError::InvalidPlan(error.to_string()))?;
                return Err(PortableF32MatmulError::Unsupported(
                    "tensor-core matmul retains its selected device contract",
                ));
            }
            MatmulValue::Quantized(plan) => {
                plan.validate()
                    .map_err(|error| PortableF32MatmulError::InvalidPlan(error.to_string()))?;
                return Err(PortableF32MatmulError::Unsupported(
                    "quantized matmul has a separate backend contract",
                ));
            }
        };
        if (plan.lhs_dtype, plan.rhs_dtype, plan.dtype) != (DType::F32, DType::F32, DType::F32) {
            return Err(PortableF32MatmulError::Unsupported(
                "portable matmul requires homogeneous F32 storage",
            ));
        }
        if plan.lhs == plan.rhs
            && (plan.lhs_shape != plan.rhs_shape || plan.lhs_dtype != plan.rhs_dtype)
        {
            return Err(PortableF32MatmulError::InvalidPlan(
                "aliased operands disagree on their descriptor".into(),
            ));
        }
        let extent = plan
            .output_shape
            .numel()
            .map_err(|_| PortableF32MatmulError::Overflow)?;
        let lhs_elements = plan
            .lhs_shape
            .numel()
            .map_err(|_| PortableF32MatmulError::Overflow)?;
        let rhs_elements = plan
            .rhs_shape
            .numel()
            .map_err(|_| PortableF32MatmulError::Overflow)?;
        let lhs_batch_axes = batch_axes(plan, &plan.lhs_shape, plan.lhs_vector)?;
        let rhs_batch_axes = batch_axes(plan, &plan.rhs_shape, plan.rhs_vector)?;
        let projection = Self {
            value,
            plan,
            extent,
            lhs_elements,
            rhs_elements,
            lhs_batch_axes,
            rhs_batch_axes,
        };
        projection.validate_bounds()?;
        Ok(projection)
    }

    pub(crate) fn value(&self) -> &'a MatmulValue {
        self.value
    }

    pub(crate) fn plan(&self) -> &'a MatmulKernelPlan {
        self.plan
    }

    pub(crate) fn extent(&self) -> usize {
        self.extent
    }

    pub(crate) fn lhs_elements(&self) -> usize {
        self.lhs_elements
    }

    pub(crate) fn rhs_elements(&self) -> usize {
        self.rhs_elements
    }

    pub(crate) fn lhs_batch_axes(&self) -> &[PortableBatchAxis] {
        &self.lhs_batch_axes
    }

    pub(crate) fn rhs_batch_axes(&self) -> &[PortableBatchAxis] {
        &self.rhs_batch_axes
    }

    /// Logical first-use pointer order. Equal lhs/rhs IDs share one pointer.
    pub(crate) fn inputs(&self) -> Vec<PortableMatmulInput<'a>> {
        let mut inputs = vec![PortableMatmulInput {
            node: self.plan.lhs,
            shape: &self.plan.lhs_shape,
        }];
        if self.plan.rhs != self.plan.lhs {
            inputs.push(PortableMatmulInput {
                node: self.plan.rhs,
                shape: &self.plan.rhs_shape,
            });
        }
        inputs
    }

    /// Validates schedule-owned dense logical inputs before cache lookup or
    /// native resource work. Consumer-local views remain an explicit fallback.
    pub(crate) fn validate_schedule_bindings(
        &self,
        bindings: &[ScheduleInputBinding],
    ) -> Result<(), PortableF32MatmulError> {
        let inputs = self.inputs();
        if bindings.len() != inputs.len() {
            return Err(PortableF32MatmulError::InvalidBinding(
                "pointer count disagrees with the canonical operand ABI".into(),
            ));
        }
        for (abi_index, (binding, input)) in bindings.iter().zip(inputs).enumerate() {
            let elements = input
                .shape
                .numel()
                .map_err(|_| PortableF32MatmulError::Overflow)?;
            let bytes = elements
                .checked_mul(DType::F32.itemsize())
                .ok_or(PortableF32MatmulError::Overflow)?;
            if binding.abi_index != abi_index
                || binding.input_node != input.node
                || binding.desc.id != input.node.index() as u64
                || binding.desc.shape != *input.shape
                || binding.desc.dtype != DType::F32
                || binding.desc.bytes != bytes
                || !binding.desc.read_only
                || binding.desc.view.is_some()
            {
                return Err(PortableF32MatmulError::InvalidBinding(format!(
                    "operand {abi_index} is not its exact dense F32 descriptor"
                )));
            }
        }
        Ok(())
    }

    fn validate_bounds(&self) -> Result<(), PortableF32MatmulError> {
        if self.extent == 0 || self.plan.k == 0 {
            return Ok(());
        }
        let lhs_batch = max_batch_offset(&self.lhs_batch_axes)?;
        let rhs_batch = max_batch_offset(&self.rhs_batch_axes)?;
        let lhs_matrix = if self.plan.lhs_vector {
            self.plan.k - 1
        } else {
            self.plan
                .m
                .checked_mul(self.plan.k)
                .and_then(|elements| elements.checked_sub(1))
                .ok_or(PortableF32MatmulError::Overflow)?
        };
        let rhs_matrix = if self.plan.rhs_vector {
            self.plan.k - 1
        } else {
            self.plan
                .k
                .checked_mul(self.plan.n)
                .and_then(|elements| elements.checked_sub(1))
                .ok_or(PortableF32MatmulError::Overflow)?
        };
        let lhs_offset = lhs_batch
            .checked_mul(if self.plan.lhs_vector {
                self.plan.k
            } else {
                self.plan
                    .m
                    .checked_mul(self.plan.k)
                    .ok_or(PortableF32MatmulError::Overflow)?
            })
            .and_then(|offset| offset.checked_add(lhs_matrix))
            .ok_or(PortableF32MatmulError::Overflow)?;
        let rhs_offset = rhs_batch
            .checked_mul(if self.plan.rhs_vector {
                self.plan.k
            } else {
                self.plan
                    .k
                    .checked_mul(self.plan.n)
                    .ok_or(PortableF32MatmulError::Overflow)?
            })
            .and_then(|offset| offset.checked_add(rhs_matrix))
            .ok_or(PortableF32MatmulError::Overflow)?;
        if lhs_offset >= self.lhs_elements || rhs_offset >= self.rhs_elements {
            return Err(PortableF32MatmulError::InvalidPlan(
                "projected address exceeds an operand".into(),
            ));
        }
        Ok(())
    }
}

fn batch_axes(
    plan: &MatmulKernelPlan,
    shape: &Shape,
    vector: bool,
) -> Result<Vec<PortableBatchAxis>, PortableF32MatmulError> {
    if vector {
        return Ok(Vec::new());
    }
    let input_batch = &shape.dims()[..shape.rank() - 2];
    let pad = plan
        .batch_shape
        .len()
        .checked_sub(input_batch.len())
        .ok_or_else(|| PortableF32MatmulError::InvalidPlan("batch rank mismatch".into()))?;
    let mut axes = Vec::new();
    for (input_axis, dimension) in input_batch.iter().copied().enumerate() {
        if dimension == 1 {
            continue;
        }
        let output_axis = pad + input_axis;
        if plan.batch_shape.get(output_axis).copied() != Some(dimension) {
            return Err(PortableF32MatmulError::InvalidPlan(
                "batch broadcast projection mismatch".into(),
            ));
        }
        axes.push(PortableBatchAxis {
            output_axis,
            divisor: checked_product(&plan.batch_shape[output_axis + 1..])?,
            dimension,
            input_stride: checked_product(&input_batch[input_axis + 1..])?,
        });
    }
    Ok(axes)
}

fn checked_product(values: &[usize]) -> Result<usize, PortableF32MatmulError> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(PortableF32MatmulError::Overflow)
    })
}

fn max_batch_offset(axes: &[PortableBatchAxis]) -> Result<usize, PortableF32MatmulError> {
    axes.iter().try_fold(0usize, |offset, axis| {
        (axis.dimension - 1)
            .checked_mul(axis.input_stride)
            .and_then(|term| offset.checked_add(term))
            .ok_or(PortableF32MatmulError::Overflow)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GgmlType, Graph, MatmulTargetCaps, Operation, QuantizedMatmulPlan, QuantizedTensorData,
        Scalar, Storage, TensorData, TiledMatmulPayload, ViewMap, kernel::lower_graph_matmul,
        schedule,
    };

    #[test]
    fn portable_projection_covers_vector_matrix_and_right_broadcast_geometry() {
        let mut saw_serial = false;
        let mut saw_tiled = false;
        for (lhs_shape, rhs_shape, flags, output_shape) in [
            (vec![3], vec![3], (true, true), vec![]),
            (vec![2, 3], vec![3], (false, true), vec![2]),
            (vec![3], vec![3, 2], (true, false), vec![2]),
            (vec![2, 3], vec![3, 4], (false, false), vec![2, 4]),
            (vec![2, 3, 4], vec![1, 4, 5], (false, false), vec![2, 3, 5]),
            (
                vec![2, 1, 3, 4],
                vec![1, 5, 4, 6],
                (false, false),
                vec![2, 5, 3, 6],
            ),
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", lhs_shape, DType::F32);
            let rhs = graph.input_dtype("rhs", rhs_shape, DType::F32);
            let output = graph.matmul(lhs, rhs).unwrap();
            let kernel = lower_graph_matmul(&graph, output).unwrap();
            let Operation::Matmul(value) = kernel.operation() else {
                unreachable!()
            };
            saw_serial |= matches!(value, crate::MatmulValue::Serial(_));
            saw_tiled |= matches!(value, crate::MatmulValue::Tiled(_));
            let portable = PortableF32Matmul::new(value).unwrap();
            assert_eq!(
                (portable.plan().lhs_vector, portable.plan().rhs_vector),
                flags
            );
            assert_eq!(portable.plan().output_shape, Shape::new(output_shape));
            portable
                .validate_schedule_bindings(
                    schedule(&graph, output).unwrap().items[0].ordered_inputs(),
                )
                .unwrap();
        }
        assert!(saw_serial && saw_tiled);
    }

    #[test]
    fn portable_projection_deduplicates_alias_and_rejects_views_and_selected_payloads() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 2], DType::F32);
        let output = graph.matmul(input, input).unwrap();
        let kernel = lower_graph_matmul(&graph, output).unwrap();
        let Operation::Matmul(value) = kernel.operation() else {
            unreachable!()
        };
        assert_eq!(PortableF32Matmul::new(value).unwrap().inputs().len(), 1);

        let mut viewed = Graph::new();
        let lhs = viewed.input_dtype("lhs", [2, 3], DType::F32);
        let rhs = viewed.input_dtype("rhs", [3, 2], DType::F32);
        let output = viewed.matmul(lhs, rhs).unwrap();
        let mut item = schedule(&viewed, output).unwrap().items.pop().unwrap();
        item.input_bindings[0].desc.view = Some(ViewMap::identity(Shape::from([2, 3])).into());
        let Operation::Matmul(value) = item.kernel.operation() else {
            unreachable!()
        };
        assert!(matches!(
            PortableF32Matmul::new(value)
                .unwrap()
                .validate_schedule_bindings(item.ordered_inputs()),
            Err(PortableF32MatmulError::InvalidBinding(_))
        ));

        let mut narrow = Graph::new();
        let lhs = narrow.input_dtype("lhs", [16, 16], DType::F16);
        let rhs = narrow.input_dtype("rhs", [16, 8], DType::F16);
        let output = narrow.matmul(lhs, rhs).unwrap();
        let kernel = lower_graph_matmul(&narrow, output).unwrap();
        let Operation::Matmul(value) = kernel.operation() else {
            unreachable!()
        };
        assert!(matches!(value, crate::MatmulValue::TensorCore(_)));
        assert!(matches!(
            PortableF32Matmul::new(value),
            Err(PortableF32MatmulError::Unsupported(_))
        ));

        let mut wide = Graph::new();
        let lhs = wide.input_dtype("lhs", [2, 3], DType::F64);
        let rhs = wide.input_dtype("rhs", [3, 2], DType::F64);
        let output = wide.matmul(lhs, rhs).unwrap();
        let kernel = lower_graph_matmul(&wide, output).unwrap();
        let Operation::Matmul(value) = kernel.operation() else {
            unreachable!()
        };
        assert!(matches!(
            PortableF32Matmul::new(value),
            Err(PortableF32MatmulError::Unsupported(_))
        ));

        let packed = QuantizedTensorData::new(GgmlType::Q4_0, [1, 32].into(), vec![0; 18]).unwrap();
        let quantized = QuantizedMatmulPlan::new(
            NodeId::from_index(10),
            NodeId::from_index(11),
            NodeId::from_index(12),
            [32].into(),
            packed.descriptor().clone(),
        )
        .unwrap();
        assert!(matches!(
            PortableF32Matmul::new(&crate::MatmulValue::Quantized(Box::new(quantized))),
            Err(PortableF32MatmulError::Unsupported(_))
        ));
    }

    #[test]
    fn portable_projection_authenticates_tiled_payload_and_preserves_f32_step_order() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [1, 3], DType::F32);
        let rhs = graph.input_dtype("rhs", [3, 1], DType::F32);
        let output = graph.matmul(lhs, rhs).unwrap();
        let base = MatmulKernelPlan::from_graph(&graph, output).unwrap();
        let tiled = TiledMatmulPayload::select(
            base.clone(),
            MatmulTargetCaps::conservative_ptx(80).unwrap(),
        )
        .unwrap()
        .unwrap();
        PortableF32Matmul::new(&crate::MatmulValue::Tiled(Box::new(tiled.clone()))).unwrap();
        let mut malformed = tiled;
        malformed.tile.cache_key ^= 1;
        assert!(matches!(
            PortableF32Matmul::new(&crate::MatmulValue::Tiled(Box::new(malformed))),
            Err(PortableF32MatmulError::InvalidPlan(_))
        ));

        let lhs =
            TensorData::from_storage([1, 3], Storage::F32(vec![16_777_216.0, 1.0, -16_777_216.0]))
                .unwrap();
        let rhs = TensorData::from_storage([3, 1], Storage::F32(vec![1.0; 3])).unwrap();
        assert_eq!(
            base.execute(&lhs, &rhs).unwrap().storage(),
            &Storage::F32(vec![0.0])
        );

        let mut special = Graph::new();
        let lhs_id = special.input_dtype("lhs", [1, 1], DType::F32);
        let rhs_id = special.input_dtype("rhs", [1, 1], DType::F32);
        let output = special.matmul(lhs_id, rhs_id).unwrap();
        let plan = MatmulKernelPlan::from_graph(&special, output).unwrap();
        let negative_zero =
            TensorData::from_storage([1, 1], Storage::F32(vec![f32::from_bits(0x8000_0000)]))
                .unwrap();
        let one = TensorData::from_storage([1, 1], Storage::F32(vec![1.0])).unwrap();
        let Storage::F32(values) = plan
            .execute(&negative_zero, &one)
            .unwrap()
            .storage()
            .clone()
        else {
            unreachable!()
        };
        assert_eq!(values[0].to_bits(), 0);
        let infinity = TensorData::from_storage([1, 1], Storage::F32(vec![f32::INFINITY])).unwrap();
        let zero = TensorData::from_storage([1, 1], Storage::F32(vec![0.0])).unwrap();
        assert!(matches!(
            plan.execute(&infinity, &zero).unwrap().scalar_at(0),
            Scalar::F(value) if value.is_nan()
        ));
    }

    #[test]
    fn portable_projection_retains_populated_zero_contraction_and_empty_output() {
        for (lhs_shape, rhs_shape, extent, k) in [([2, 0], [0, 3], 6, 0), ([0, 4], [4, 3], 0, 4)] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", lhs_shape, DType::F32);
            let rhs = graph.input_dtype("rhs", rhs_shape, DType::F32);
            let output = graph.matmul(lhs, rhs).unwrap();
            let kernel = lower_graph_matmul(&graph, output).unwrap();
            let Operation::Matmul(value) = kernel.operation() else {
                unreachable!()
            };
            let portable = PortableF32Matmul::new(value).unwrap();
            assert_eq!(portable.extent(), extent);
            assert_eq!(portable.plan().k, k);
        }
    }
}
