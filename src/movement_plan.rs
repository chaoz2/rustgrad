//! Immutable contracts and pure execution for materializing movement kernels.
use crate::{DType, Graph, NodeId, Op, Scalar, Shape, Storage, TensorData, index::DenseIndex};
use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MovementOperand {
    pub node: NodeId,
    pub shape: Shape,
    pub dtype: DType,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MovementKernelKind {
    /// Concrete constant padding. `fill_bits` is already converted to the
    /// output storage dtype, so all executors share one raw-payload contract.
    Pad {
        input: MovementOperand,
        padding: Vec<(usize, usize)>,
        fill_bits: u64,
    },
    Concat {
        inputs: Vec<MovementOperand>,
        axis: usize,
    },
    Gather {
        input: MovementOperand,
        index: MovementOperand,
        axis: usize,
    },
    Scatter {
        base: MovementOperand,
        index: MovementOperand,
        updates: MovementOperand,
        axis: usize,
        add: bool,
    },
}

/// Fully validated materializing movement geometry and ordered pointer ABI.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MovementKernelPlan {
    pub kind: MovementKernelKind,
    pub output: NodeId,
    pub output_shape: Shape,
    pub dtype: DType,
    pub cache_key: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MovementPlanError {
    NotMovement,
    InvalidGeometry,
    UnsupportedDType,
    Overflow,
}

impl fmt::Display for MovementPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "movement plan error: {self:?}")
    }
}
impl std::error::Error for MovementPlanError {}

/// A fail-closed error from graph-independent movement execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MovementExecutionError {
    InvalidPlan(MovementPlanError),
    OperandCount {
        expected: usize,
        actual: usize,
    },
    OperandDescriptor {
        position: usize,
        expected_shape: Shape,
        actual_shape: Shape,
        expected_dtype: DType,
        actual_dtype: DType,
    },
    IndexOutOfBounds {
        axis: usize,
        index: i64,
        dim: usize,
    },
    Overflow,
    InvalidGeometry,
}

impl fmt::Display for MovementExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "movement execution error: {self:?}")
    }
}
impl std::error::Error for MovementExecutionError {}

impl From<MovementPlanError> for MovementExecutionError {
    fn from(value: MovementPlanError) -> Self {
        Self::InvalidPlan(value)
    }
}

impl MovementOperand {
    fn from_graph(graph: &Graph, node: NodeId) -> Result<Self, MovementPlanError> {
        Ok(Self {
            node,
            shape: graph
                .shape(node)
                .map_err(|_| MovementPlanError::InvalidGeometry)?
                .clone(),
            dtype: graph
                .dtype(node)
                .map_err(|_| MovementPlanError::UnsupportedDType)?,
        })
    }
}

impl MovementKernelPlan {
    pub fn from_graph(graph: &Graph, output: NodeId) -> Result<Self, MovementPlanError> {
        let kind = match graph
            .op(output)
            .map_err(|_| MovementPlanError::InvalidGeometry)?
        {
            Op::Pad { input, padding, fill } => {
                let input = MovementOperand::from_graph(graph, *input)?;
                MovementKernelKind::Pad {
                    fill_bits: scalar_bits(input.dtype, *fill),
                    input,
                    padding: padding.clone(),
                }
            }
            Op::Concat { inputs, axis } => MovementKernelKind::Concat {
                inputs: inputs
                    .iter()
                    .map(|node| MovementOperand::from_graph(graph, *node))
                    .collect::<Result<Vec<_>, _>>()?,
                axis: *axis,
            },
            Op::Gather { input, index, axis } => MovementKernelKind::Gather {
                input: MovementOperand::from_graph(graph, *input)?,
                index: MovementOperand::from_graph(graph, *index)?,
                axis: *axis,
            },
            Op::Scatter {
                base,
                index,
                updates,
                axis,
                add,
            } => MovementKernelKind::Scatter {
                base: MovementOperand::from_graph(graph, *base)?,
                index: MovementOperand::from_graph(graph, *index)?,
                updates: MovementOperand::from_graph(graph, *updates)?,
                axis: *axis,
                add: *add,
            },
            _ => return Err(MovementPlanError::NotMovement),
        };
        let mut plan = Self {
            kind,
            output,
            output_shape: graph
                .shape(output)
                .map_err(|_| MovementPlanError::InvalidGeometry)?
                .clone(),
            dtype: graph
                .dtype(output)
                .map_err(|_| MovementPlanError::UnsupportedDType)?,
            cache_key: 0,
        };
        plan.cache_key = plan.expected_cache_key();
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), MovementPlanError> {
        self.output_shape
            .numel()
            .map_err(|_| MovementPlanError::Overflow)?;
        if self.cache_key != self.expected_cache_key() {
            return Err(MovementPlanError::InvalidGeometry);
        }
        match &self.kind {
            MovementKernelKind::Pad { input, padding, fill_bits } => {
                input.shape.numel().map_err(|_| MovementPlanError::Overflow)?;
                if padding.len() != input.shape.rank() || self.dtype != input.dtype {
                    return Err(MovementPlanError::InvalidGeometry);
                }
                let expected = input.shape.dims().iter().zip(padding).map(|(dim, (before, after))| {
                    dim.checked_add(*before).and_then(|x| x.checked_add(*after))
                        .ok_or(MovementPlanError::Overflow)
                }).collect::<Result<Vec<_>, _>>()?;
                if self.output_shape.dims() != expected || !valid_bits(self.dtype, *fill_bits) {
                    return Err(MovementPlanError::InvalidGeometry);
                }
            }
            MovementKernelKind::Concat { inputs, axis } => {
                let first = inputs.first().ok_or(MovementPlanError::InvalidGeometry)?;
                if *axis >= first.shape.rank() || self.output_shape.rank() != first.shape.rank() {
                    return Err(MovementPlanError::InvalidGeometry);
                }
                let mut axis_total = 0usize;
                let mut dtype = first.dtype;
                for input in inputs {
                    input
                        .shape
                        .numel()
                        .map_err(|_| MovementPlanError::Overflow)?;
                    if input.shape.rank() != first.shape.rank()
                        || input
                            .shape
                            .dims()
                            .iter()
                            .zip(first.shape.dims())
                            .enumerate()
                            .any(|(dim, (actual, expected))| dim != *axis && actual != expected)
                    {
                        return Err(MovementPlanError::InvalidGeometry);
                    }
                    axis_total = axis_total
                        .checked_add(input.shape.dims()[*axis])
                        .ok_or(MovementPlanError::Overflow)?;
                    dtype = dtype.promote(input.dtype);
                }
                let mut expected = first.shape.dims().to_vec();
                expected[*axis] = axis_total;
                if self.output_shape.dims() != expected || self.dtype != dtype {
                    return Err(MovementPlanError::InvalidGeometry);
                }
            }
            MovementKernelKind::Gather { input, index, axis } => {
                validate_index_geometry(input, index, *axis)?;
                if self.dtype != input.dtype || self.output_shape != index.shape {
                    return Err(MovementPlanError::InvalidGeometry);
                }
            }
            MovementKernelKind::Scatter {
                base,
                index,
                updates,
                axis,
                add,
            } => {
                validate_index_geometry(base, index, *axis)?;
                if self.output_shape != base.shape
                    || self.dtype != base.dtype.promote(updates.dtype)
                    || updates.shape.rank() != index.shape.rank()
                    || updates
                        .shape
                        .dims()
                        .iter()
                        .zip(index.shape.dims())
                        .any(|(update, index)| update < index)
                {
                    return Err(MovementPlanError::InvalidGeometry);
                }
                if *add && !matches!(self.dtype, DType::F32 | DType::F64) {
                    return Err(MovementPlanError::UnsupportedDType);
                }
            }
        }
        if self
            .input_operands()
            .iter()
            .any(|operand| operand.node == self.output)
        {
            return Err(MovementPlanError::InvalidGeometry);
        }
        Ok(())
    }

    pub fn input_operands(&self) -> Vec<&MovementOperand> {
        match &self.kind {
            MovementKernelKind::Pad { input, .. } => vec![input],
            MovementKernelKind::Concat { inputs, .. } => inputs.iter().collect(),
            MovementKernelKind::Gather { input, index, .. } => vec![input, index],
            MovementKernelKind::Scatter {
                base,
                index,
                updates,
                ..
            } => vec![base, index, updates],
        }
    }

    /// Executes this validated plan against operands in [`Self::input_operands`]
    /// order. All descriptors and every data-dependent index are checked before
    /// output storage is allocated or cloned.
    pub fn execute(&self, operands: &[TensorData]) -> Result<TensorData, MovementExecutionError> {
        self.validate()?;
        let expected = self.input_operands();
        if operands.len() != expected.len() {
            return Err(MovementExecutionError::OperandCount {
                expected: expected.len(),
                actual: operands.len(),
            });
        }
        for (position, (value, desc)) in operands.iter().zip(expected).enumerate() {
            if value.shape() != &desc.shape || value.dtype() != desc.dtype {
                return Err(MovementExecutionError::OperandDescriptor {
                    position,
                    expected_shape: desc.shape.clone(),
                    actual_shape: value.shape().clone(),
                    expected_dtype: desc.dtype,
                    actual_dtype: value.dtype(),
                });
            }
        }
        match &self.kind {
            MovementKernelKind::Pad { input, padding, fill_bits } => {
                self.execute_pad(&operands[0], input, padding, *fill_bits)
            }
            MovementKernelKind::Concat { inputs, axis } => {
                self.execute_concat(operands, inputs, *axis)
            }
            MovementKernelKind::Gather { input, index, axis } => {
                self.execute_gather(&operands[0], &operands[1], input, index, *axis)
            }
            MovementKernelKind::Scatter {
                base,
                index,
                updates,
                axis,
                add,
            } => self.execute_scatter(
                &operands[0],
                &operands[1],
                &operands[2],
                base,
                index,
                updates,
                *axis,
                *add,
            ),
        }
    }

    fn execute_pad(&self, input: &TensorData, desc: &MovementOperand, padding: &[(usize, usize)], fill_bits: u64) -> Result<TensorData, MovementExecutionError> {
        let source = dense(&desc.shape)?;
        let output = dense(&self.output_shape)?;
        let fill = scalar_from_bits(self.dtype, fill_bits);
        let mut values = Vec::with_capacity(output.len());
        for linear in 0..output.len() {
            let coords = output.coords(linear).map_err(|_| MovementExecutionError::InvalidGeometry)?;
            let inside = coords.iter().zip(padding).zip(desc.shape.dims()).all(|((coord, (before, _)), dim)| *coord >= *before && *coord - *before < *dim);
            values.push(if inside {
                let input_coords = coords.iter().zip(padding).map(|(coord, (before, _))| coord - before).collect::<Vec<_>>();
                input.scalar_at(source.offset(&input_coords).map_err(|_| MovementExecutionError::InvalidGeometry)?)
            } else { fill });
        }
        TensorData::from_scalars(self.output_shape.clone(), self.dtype, values).map_err(|_| MovementExecutionError::InvalidGeometry)
    }

    fn execute_concat(
        &self,
        operands: &[TensorData],
        inputs: &[MovementOperand],
        axis: usize,
    ) -> Result<TensorData, MovementExecutionError> {
        let output = dense(&self.output_shape)?;
        let source_indices = inputs
            .iter()
            .map(|input| dense(&input.shape))
            .collect::<Result<Vec<_>, _>>()?;
        let mut ends = Vec::with_capacity(inputs.len());
        let mut total = 0usize;
        for input in inputs {
            total = total
                .checked_add(input.shape.dims()[axis])
                .ok_or(MovementExecutionError::Overflow)?;
            ends.push(total);
        }
        let mut map = Vec::with_capacity(output.len());
        for linear in 0..output.len() {
            let mut coords = output
                .coords(linear)
                .map_err(|_| MovementExecutionError::InvalidGeometry)?;
            let operand = ends
                .iter()
                .position(|end| coords[axis] < *end)
                .ok_or(MovementExecutionError::InvalidGeometry)?;
            let prior = operand.checked_sub(1).map_or(0, |position| ends[position]);
            coords[axis] -= prior;
            let offset = source_indices[operand]
                .offset(&coords)
                .map_err(|_| MovementExecutionError::InvalidGeometry)?;
            map.push((operand, offset));
        }
        let storage = if operands.iter().all(|value| value.dtype() == self.dtype) {
            select_many_raw(self.dtype, operands, &map)?
        } else {
            Storage::from_scalars(
                self.dtype,
                map.iter()
                    .map(|(operand, offset)| operands[*operand].scalar_at(*offset)),
            )
        };
        TensorData::from_storage(self.output_shape.clone(), storage)
            .map_err(|_| MovementExecutionError::InvalidGeometry)
    }

    fn execute_gather(
        &self,
        input: &TensorData,
        index: &TensorData,
        input_desc: &MovementOperand,
        index_desc: &MovementOperand,
        axis: usize,
    ) -> Result<TensorData, MovementExecutionError> {
        let map = indexed_map(input_desc, index_desc, index, axis)?;
        let offsets = map
            .iter()
            .map(|(destination, _)| *destination)
            .collect::<Vec<_>>();
        let storage = select_raw(input.storage(), &offsets);
        TensorData::from_storage(self.output_shape.clone(), storage)
            .map_err(|_| MovementExecutionError::InvalidGeometry)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_scatter(
        &self,
        base_value: &TensorData,
        index_value: &TensorData,
        updates_value: &TensorData,
        base: &MovementOperand,
        index: &MovementOperand,
        updates: &MovementOperand,
        axis: usize,
        add: bool,
    ) -> Result<TensorData, MovementExecutionError> {
        let destinations = indexed_map(base, index, index_value, axis)?;
        let update_index = dense(&updates.shape)?;
        let index_index = dense(&index.shape)?;
        let mut map = Vec::with_capacity(destinations.len());
        for (destination, linear) in destinations {
            let coords = index_index
                .coords(linear)
                .map_err(|_| MovementExecutionError::InvalidGeometry)?;
            let source = update_index
                .offset(&coords)
                .map_err(|_| MovementExecutionError::InvalidGeometry)?;
            map.push((destination, source));
        }
        let storage =
            if !add && base_value.dtype() == self.dtype && updates_value.dtype() == self.dtype {
                scatter_raw(base_value.storage(), updates_value.storage(), &map)?
            } else {
                let mut values = (0..base_value.len())
                    .map(|position| base_value.scalar_at(position))
                    .collect::<Vec<_>>();
                for (destination, source) in map {
                    let update = updates_value.scalar_at(source);
                    values[destination] = if add {
                        Scalar::F(values[destination].as_f64() + update.as_f64())
                    } else {
                        update
                    };
                }
                Storage::from_scalars(self.dtype, values)
            };
        TensorData::from_storage(self.output_shape.clone(), storage)
            .map_err(|_| MovementExecutionError::InvalidGeometry)
    }

    fn expected_cache_key(&self) -> u64 {
        let mut plan = self.clone();
        plan.cache_key = 0;
        let mut hasher = DefaultHasher::new();
        plan.hash(&mut hasher);
        hasher.finish()
    }
}

fn scalar_bits(dtype: DType, value: Scalar) -> u64 {
    let data = TensorData::scalar_with_dtype(value, dtype);
    match data.storage() {
        Storage::Bool(v) => u64::from(v[0]), Storage::I8(v) => v[0] as u8 as u64, Storage::U8(v) => v[0] as u64,
        Storage::I16(v) => v[0] as u16 as u64, Storage::U16(v) | Storage::F16(v) | Storage::BF16(v) => v[0] as u64,
        Storage::I32(v) => v[0] as u32 as u64, Storage::U32(v) => v[0] as u64, Storage::I64(v) => v[0] as u64,
        Storage::U64(v) => v[0], Storage::F32(v) => v[0].to_bits() as u64, Storage::F64(v) => v[0].to_bits(),
    }
}
fn valid_bits(dtype: DType, bits: u64) -> bool { dtype.bits() == 64 || bits >> dtype.bits() == 0 }
fn scalar_from_bits(dtype: DType, bits: u64) -> Scalar {
    match dtype { DType::Bool => Scalar::Bool(bits != 0), DType::I8 => Scalar::I(bits as u8 as i8 as i64), DType::U8 => Scalar::U(bits as u8 as u64), DType::I16 => Scalar::I(bits as u16 as i16 as i64), DType::U16 => Scalar::U(bits as u16 as u64), DType::I32 => Scalar::I(bits as u32 as i32 as i64), DType::U32 => Scalar::U(bits as u32 as u64), DType::I64 => Scalar::I(bits as i64), DType::U64 => Scalar::U(bits), DType::F16 => Scalar::F(crate::f16_to_f32(bits as u16) as f64), DType::BF16 => Scalar::F(crate::bf16_to_f32(bits as u16) as f64), DType::F32 => Scalar::F(f32::from_bits(bits as u32) as f64), DType::F64 => Scalar::F(f64::from_bits(bits)) }
}

fn dense(shape: &Shape) -> Result<DenseIndex, MovementExecutionError> {
    DenseIndex::new(shape.clone()).map_err(|error| match error {
        crate::Error::ShapeOverflow(_) => MovementExecutionError::Overflow,
        _ => MovementExecutionError::InvalidGeometry,
    })
}

fn indexed_map(
    input: &MovementOperand,
    index: &MovementOperand,
    index_value: &TensorData,
    axis: usize,
) -> Result<Vec<(usize, usize)>, MovementExecutionError> {
    let input_index = dense(&input.shape)?;
    let index_index = dense(&index.shape)?;
    let mut mapped = Vec::with_capacity(index_index.len());
    for linear in 0..index_index.len() {
        let mut coords = index_index
            .coords(linear)
            .map_err(|_| MovementExecutionError::InvalidGeometry)?;
        coords[axis] = checked_index(
            index_value.scalar_at(linear),
            axis,
            input.shape.dims()[axis],
        )?;
        let destination = input_index
            .offset(&coords)
            .map_err(|_| MovementExecutionError::InvalidGeometry)?;
        mapped.push((destination, linear));
    }
    Ok(mapped)
}

fn checked_index(value: Scalar, axis: usize, dim: usize) -> Result<usize, MovementExecutionError> {
    let signed = match value {
        Scalar::I(value) => value,
        Scalar::U(value) => {
            i64::try_from(value).map_err(|_| MovementExecutionError::IndexOutOfBounds {
                axis,
                index: i64::MAX,
                dim,
            })?
        }
        _ => return Err(MovementExecutionError::InvalidGeometry),
    };
    let index = usize::try_from(signed).map_err(|_| MovementExecutionError::IndexOutOfBounds {
        axis,
        index: signed,
        dim,
    })?;
    if index >= dim {
        return Err(MovementExecutionError::IndexOutOfBounds {
            axis,
            index: signed,
            dim,
        });
    }
    Ok(index)
}

fn select_raw(storage: &Storage, offsets: &[usize]) -> Storage {
    macro_rules! selected {
        ($values:expr, $variant:ident) => {
            Storage::$variant(offsets.iter().map(|offset| $values[*offset]).collect())
        };
    }
    match storage {
        Storage::Bool(values) => selected!(values, Bool),
        Storage::I8(values) => selected!(values, I8),
        Storage::U8(values) => selected!(values, U8),
        Storage::I16(values) => selected!(values, I16),
        Storage::U16(values) => selected!(values, U16),
        Storage::I32(values) => selected!(values, I32),
        Storage::U32(values) => selected!(values, U32),
        Storage::I64(values) => selected!(values, I64),
        Storage::U64(values) => selected!(values, U64),
        Storage::F16(values) => selected!(values, F16),
        Storage::BF16(values) => selected!(values, BF16),
        Storage::F32(values) => selected!(values, F32),
        Storage::F64(values) => selected!(values, F64),
    }
}

fn select_many_raw(
    dtype: DType,
    operands: &[TensorData],
    map: &[(usize, usize)],
) -> Result<Storage, MovementExecutionError> {
    macro_rules! selected {
        ($variant:ident) => {{
            let mut output = Vec::with_capacity(map.len());
            for (operand, offset) in map {
                let Storage::$variant(values) = operands[*operand].storage() else {
                    return Err(MovementExecutionError::InvalidGeometry);
                };
                output.push(values[*offset]);
            }
            Storage::$variant(output)
        }};
    }
    Ok(match dtype {
        DType::Bool => selected!(Bool),
        DType::I8 => selected!(I8),
        DType::U8 => selected!(U8),
        DType::I16 => selected!(I16),
        DType::U16 => selected!(U16),
        DType::I32 => selected!(I32),
        DType::U32 => selected!(U32),
        DType::I64 => selected!(I64),
        DType::U64 => selected!(U64),
        DType::F16 => selected!(F16),
        DType::BF16 => selected!(BF16),
        DType::F32 => selected!(F32),
        DType::F64 => selected!(F64),
    })
}

fn scatter_raw(
    base: &Storage,
    updates: &Storage,
    map: &[(usize, usize)],
) -> Result<Storage, MovementExecutionError> {
    macro_rules! scattered {
        ($variant:ident) => {{
            let Storage::$variant(mut output) = base.clone() else {
                return Err(MovementExecutionError::InvalidGeometry);
            };
            let Storage::$variant(updates) = updates else {
                return Err(MovementExecutionError::InvalidGeometry);
            };
            for (destination, source) in map {
                output[*destination] = updates[*source];
            }
            Storage::$variant(output)
        }};
    }
    Ok(match base {
        Storage::Bool(_) => scattered!(Bool),
        Storage::I8(_) => scattered!(I8),
        Storage::U8(_) => scattered!(U8),
        Storage::I16(_) => scattered!(I16),
        Storage::U16(_) => scattered!(U16),
        Storage::I32(_) => scattered!(I32),
        Storage::U32(_) => scattered!(U32),
        Storage::I64(_) => scattered!(I64),
        Storage::U64(_) => scattered!(U64),
        Storage::F16(_) => scattered!(F16),
        Storage::BF16(_) => scattered!(BF16),
        Storage::F32(_) => scattered!(F32),
        Storage::F64(_) => scattered!(F64),
    })
}

fn validate_index_geometry(
    input: &MovementOperand,
    index: &MovementOperand,
    axis: usize,
) -> Result<(), MovementPlanError> {
    input
        .shape
        .numel()
        .and_then(|_| index.shape.numel())
        .map_err(|_| MovementPlanError::Overflow)?;
    if !index.dtype.is_integer()
        || axis >= input.shape.rank()
        || input.shape.rank() != index.shape.rank()
        || input
            .shape
            .dims()
            .iter()
            .zip(index.shape.dims())
            .enumerate()
            .any(|(dim, (input, index))| dim != axis && index > input)
    {
        return Err(MovementPlanError::InvalidGeometry);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend};
    use std::collections::HashMap;

    fn assert_storage_bits_eq(actual: &Storage, expected: &Storage) {
        match (actual, expected) {
            (Storage::F32(actual), Storage::F32(expected)) => assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            ),
            (Storage::F64(actual), Storage::F64(expected)) => assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            ),
            _ => assert_eq!(actual, expected),
        }
    }

    fn plan_result(graph: &Graph, output: NodeId, operands: &[TensorData]) -> TensorData {
        MovementKernelPlan::from_graph(graph, output)
            .unwrap()
            .execute(operands)
            .unwrap()
    }

    #[test]
    fn concat_preserves_raw_storage_for_every_dtype_and_zero_width() {
        let fixtures = vec![
            (Storage::Bool(vec![false]), Storage::Bool(vec![true])),
            (Storage::I8(vec![i8::MIN]), Storage::I8(vec![i8::MAX])),
            (Storage::U8(vec![0]), Storage::U8(vec![u8::MAX])),
            (Storage::I16(vec![i16::MIN]), Storage::I16(vec![i16::MAX])),
            (Storage::U16(vec![0]), Storage::U16(vec![u16::MAX])),
            (Storage::I32(vec![i32::MIN]), Storage::I32(vec![i32::MAX])),
            (Storage::U32(vec![0]), Storage::U32(vec![u32::MAX])),
            (Storage::I64(vec![i64::MIN]), Storage::I64(vec![i64::MAX])),
            (Storage::U64(vec![0]), Storage::U64(vec![u64::MAX])),
            (Storage::F16(vec![0x8000]), Storage::F16(vec![0x7e01])),
            (Storage::BF16(vec![0x8000]), Storage::BF16(vec![0x7fc1])),
            (
                Storage::F32(vec![f32::from_bits(0x8000_0000)]),
                Storage::F32(vec![f32::from_bits(0x7fc0_0001)]),
            ),
            (
                Storage::F64(vec![f64::from_bits(0x8000_0000_0000_0000)]),
                Storage::F64(vec![f64::from_bits(0x7ff8_0000_0000_0001)]),
            ),
        ];
        for (left_storage, right_storage) in fixtures {
            let dtype = left_storage.dtype();
            let left = TensorData::from_storage([1, 1], left_storage).unwrap();
            let right = TensorData::from_storage([1, 1], right_storage).unwrap();
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [1, 1], dtype);
            let rhs = graph.input_dtype("rhs", [1, 1], dtype);
            let output = graph.concat([lhs, rhs], 1).unwrap();
            let result = plan_result(&graph, output, &[left.clone(), right.clone()]);
            let expected = match (left.storage(), right.storage()) {
                (Storage::Bool(a), Storage::Bool(b)) => Storage::Bool([a.as_slice(), b].concat()),
                (Storage::I8(a), Storage::I8(b)) => Storage::I8([a.as_slice(), b].concat()),
                (Storage::U8(a), Storage::U8(b)) => Storage::U8([a.as_slice(), b].concat()),
                (Storage::I16(a), Storage::I16(b)) => Storage::I16([a.as_slice(), b].concat()),
                (Storage::U16(a), Storage::U16(b)) => Storage::U16([a.as_slice(), b].concat()),
                (Storage::I32(a), Storage::I32(b)) => Storage::I32([a.as_slice(), b].concat()),
                (Storage::U32(a), Storage::U32(b)) => Storage::U32([a.as_slice(), b].concat()),
                (Storage::I64(a), Storage::I64(b)) => Storage::I64([a.as_slice(), b].concat()),
                (Storage::U64(a), Storage::U64(b)) => Storage::U64([a.as_slice(), b].concat()),
                (Storage::F16(a), Storage::F16(b)) => Storage::F16([a.as_slice(), b].concat()),
                (Storage::BF16(a), Storage::BF16(b)) => Storage::BF16([a.as_slice(), b].concat()),
                (Storage::F32(a), Storage::F32(b)) => Storage::F32([a.as_slice(), b].concat()),
                (Storage::F64(a), Storage::F64(b)) => Storage::F64([a.as_slice(), b].concat()),
                _ => unreachable!(),
            };
            assert_storage_bits_eq(result.storage(), &expected);
            assert_eq!(result.shape(), &Shape::from([1, 2]));
        }

        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 0], DType::I32);
        let rhs = graph.input_dtype("rhs", [2, 3], DType::I32);
        let output = graph.concat([lhs, rhs], 1).unwrap();
        let empty = TensorData::from_storage([2, 0], Storage::I32(vec![])).unwrap();
        let values =
            TensorData::from_storage([2, 3], Storage::I32(vec![1, 2, 3, 4, 5, 6])).unwrap();
        assert_eq!(
            plan_result(&graph, output, &[empty, values.clone()]),
            values
        );

        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [0, 2], DType::U8);
        let rhs = graph.input_dtype("rhs", [2, 2], DType::U8);
        let output = graph.concat([lhs, rhs], 0).unwrap();
        let empty = TensorData::from_storage([0, 2], Storage::U8(vec![])).unwrap();
        let values = TensorData::from_storage([2, 2], Storage::U8(vec![1, 2, 3, 4])).unwrap();
        assert_eq!(
            plan_result(&graph, output, &[empty, values.clone()]),
            values
        );
    }

    #[test]
    fn mixed_concat_and_every_integer_gather_match_cpu_oracle() {
        let mut concat_graph = Graph::new();
        let lhs = concat_graph.input_dtype("lhs", [1, 2], DType::I8);
        let rhs = concat_graph.input_dtype("rhs", [1, 1], DType::U8);
        let concat = concat_graph.concat([lhs, rhs], 1).unwrap();
        let left = TensorData::from_storage([1, 2], Storage::I8(vec![-2, 3])).unwrap();
        let right = TensorData::from_storage([1, 1], Storage::U8(vec![250])).unwrap();
        let oracle = CpuBackend
            .execute(
                &concat_graph,
                concat,
                &HashMap::from([("lhs".into(), left.clone()), ("rhs".into(), right.clone())]),
            )
            .unwrap();
        assert_eq!(plan_result(&concat_graph, concat, &[left, right]), oracle);

        for dtype in [
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [2, 3], DType::F16);
            let index = graph.input_dtype("index", [2, 2], dtype);
            let output = graph.gather(input, index, 1).unwrap();
            let input_value = TensorData::from_storage(
                [2, 3],
                Storage::F16(vec![0x8000, 0x7e01, 0x3c00, 0x4000, 0x4200, 0x4400]),
            )
            .unwrap();
            let index_value = TensorData::from_scalars(
                [2, 2],
                dtype,
                [Scalar::I(2), Scalar::I(0), Scalar::I(1), Scalar::I(1)],
            )
            .unwrap();
            let oracle = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([
                        ("input".into(), input_value.clone()),
                        ("index".into(), index_value.clone()),
                    ]),
                )
                .unwrap();
            assert_eq!(
                plan_result(&graph, output, &[input_value, index_value]),
                oracle,
                "{dtype:?}"
            );
        }
    }

    #[test]
    fn gather_and_scatter_preflight_indices_and_preserve_row_major_semantics() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1, 3], DType::I32);
        let index = graph.input_dtype("index", [1, 2], DType::I64);
        let gather = graph.gather(input, index, 1).unwrap();
        let plan = MovementKernelPlan::from_graph(&graph, gather).unwrap();
        let input_value = TensorData::from_storage([1, 3], Storage::I32(vec![1, 2, 3])).unwrap();
        for selected in [-1, 3] {
            let index_value =
                TensorData::from_storage([1, 2], Storage::I64(vec![0, selected])).unwrap();
            assert!(matches!(
                plan.execute(&[input_value.clone(), index_value]),
                Err(MovementExecutionError::IndexOutOfBounds {
                    axis: 1,
                    index,
                    dim: 3
                }) if index == selected
            ));
        }
        assert!(matches!(
            plan.execute(std::slice::from_ref(&input_value)),
            Err(MovementExecutionError::OperandCount {
                expected: 2,
                actual: 1
            })
        ));
        let wrong = TensorData::from_storage([1, 2], Storage::I32(vec![0, 1])).unwrap();
        assert!(matches!(
            plan.execute(&[input_value, wrong]),
            Err(MovementExecutionError::OperandDescriptor { position: 1, .. })
        ));

        let mut unsigned_graph = Graph::new();
        let input = unsigned_graph.input_dtype("input", [1, 3], DType::I32);
        let index = unsigned_graph.input_dtype("index", [1, 1], DType::U64);
        let gather = unsigned_graph.gather(input, index, 1).unwrap();
        let unsigned_plan = MovementKernelPlan::from_graph(&unsigned_graph, gather).unwrap();
        assert!(matches!(
            unsigned_plan.execute(&[
                TensorData::from_storage([1, 3], Storage::I32(vec![1, 2, 3])).unwrap(),
                TensorData::from_storage([1, 1], Storage::U64(vec![u64::MAX])).unwrap()
            ]),
            Err(MovementExecutionError::IndexOutOfBounds {
                axis: 1,
                index: i64::MAX,
                dim: 3
            })
        ));

        let mut empty_graph = Graph::new();
        let input = empty_graph.input_dtype("input", [2, 0], DType::I32);
        let index = empty_graph.input_dtype("index", [2, 0], DType::U64);
        let output = empty_graph.gather(input, index, 1).unwrap();
        let empty_input = TensorData::from_storage([2, 0], Storage::I32(vec![])).unwrap();
        let empty_index = TensorData::from_storage([2, 0], Storage::U64(vec![])).unwrap();
        assert_eq!(
            plan_result(&empty_graph, output, &[empty_input, empty_index]).shape(),
            &Shape::from([2, 0])
        );

        let mut scatter_graph = Graph::new();
        let base = scatter_graph.input_dtype("base", [1, 3], DType::F16);
        let index = scatter_graph.input_dtype("index", [1, 3], DType::I32);
        let updates = scatter_graph.input_dtype("updates", [1, 3], DType::F16);
        let output = scatter_graph.scatter(base, index, updates, 1).unwrap();
        let result = plan_result(
            &scatter_graph,
            output,
            &[
                TensorData::from_storage([1, 3], Storage::F16(vec![0x8000, 0x3c00, 0x7e01]))
                    .unwrap(),
                TensorData::from_storage([1, 3], Storage::I32(vec![1, 1, 1])).unwrap(),
                TensorData::from_storage([1, 3], Storage::F16(vec![0x4000, 0x4200, 0x7e55]))
                    .unwrap(),
            ],
        );
        assert_eq!(
            result.storage(),
            &Storage::F16(vec![0x8000, 0x7e55, 0x7e01])
        );

        let mut empty_scatter = Graph::new();
        let base = empty_scatter.input_dtype("base", [2, 0], DType::F32);
        let index = empty_scatter.input_dtype("index", [2, 0], DType::I32);
        let updates = empty_scatter.input_dtype("updates", [2, 0], DType::F32);
        let output = empty_scatter.scatter_add(base, index, updates, 1).unwrap();
        assert_eq!(
            plan_result(
                &empty_scatter,
                output,
                &[
                    TensorData::from_storage([2, 0], Storage::F32(vec![])).unwrap(),
                    TensorData::from_storage([2, 0], Storage::I32(vec![])).unwrap(),
                    TensorData::from_storage([2, 0], Storage::F32(vec![])).unwrap(),
                ]
            )
            .shape(),
            &Shape::from([2, 0])
        );
    }

    #[test]
    fn scatter_add_f32_and_f64_match_cpu_row_major_accumulation() {
        for dtype in [DType::F32, DType::F64] {
            let mut graph = Graph::new();
            let base = graph.input_dtype("base", [1, 2], dtype);
            let index = graph.input_dtype("index", [1, 3], DType::I32);
            let updates = graph.input_dtype("updates", [1, 3], dtype);
            let output = graph.scatter_add(base, index, updates, 1).unwrap();
            let base_value =
                TensorData::from_scalars([1, 2], dtype, [Scalar::F(1.0), Scalar::F(-2.0)]).unwrap();
            let index_value =
                TensorData::from_storage([1, 3], Storage::I32(vec![0, 0, 1])).unwrap();
            let updates_value = TensorData::from_scalars(
                [1, 3],
                dtype,
                [Scalar::F(0.25), Scalar::F(0.5), Scalar::F(4.0)],
            )
            .unwrap();
            let inputs = HashMap::from([
                ("base".into(), base_value.clone()),
                ("index".into(), index_value.clone()),
                ("updates".into(), updates_value.clone()),
            ]);
            let oracle = CpuBackend.execute(&graph, output, &inputs).unwrap();
            assert_eq!(
                plan_result(&graph, output, &[base_value, index_value, updates_value]),
                oracle,
                "{dtype:?}"
            );
        }
    }

    #[test]
    fn pad_plan_canonicalizes_typed_fill_and_geometry_before_execution() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1, 2], DType::F16);
        let output = graph.pad(input, [(1, 0), (0, 2)], Scalar::F(-0.0)).unwrap();
        let plan = MovementKernelPlan::from_graph(&graph, output).unwrap();
        assert_eq!(plan.output_shape, Shape::from([2, 4]));
        let MovementKernelKind::Pad { padding, fill_bits, .. } = &plan.kind else { panic!("pad plan") };
        assert_eq!(padding, vec![(1, 0), (0, 2)]);
        assert_eq!(*fill_bits, 0x8000);
        assert!(plan.validate().is_ok());
    }
}
