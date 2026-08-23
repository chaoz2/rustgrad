use super::Backend;
use crate::{
    BinaryOp, CompareOp, DType, Error, Graph, LogicalOp, NodeId, Op, Result, Scalar, Shape,
    TensorData, UnaryOp,
};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn execute(
        &self,
        graph: &Graph,
        output: NodeId,
        inputs: &HashMap<String, TensorData>,
    ) -> Result<TensorData> {
        graph.node(output)?;
        let mut values: Vec<TensorData> = Vec::with_capacity(output.index() + 1);
        for node in &graph.nodes[..=output.index()] {
            let value = match &node.op {
                Op::Input { name } => {
                    let value = inputs
                        .get(name)
                        .ok_or_else(|| Error::MissingInput(name.clone()))?;
                    if value.shape() != &node.shape {
                        return Err(Error::InputShape {
                            name: name.clone(),
                            expected: node.shape.clone(),
                            actual: value.shape().clone(),
                        });
                    }
                    if value.dtype() != node.dtype {
                        return Err(Error::InputDType {
                            name: name.clone(),
                            expected: node.dtype,
                            actual: value.dtype(),
                        });
                    }
                    value.clone()
                }
                Op::Constant(data) => data.clone(),
                Op::Cast { input, dtype } => values[input.index()].cast(*dtype),
                Op::Unary { op, input } => unary(&values[input.index()], *op)?,
                Op::Binary { op, lhs, rhs } => binary(&values, *lhs, *rhs, &node.shape, *op)?,
                Op::Compare { op, lhs, rhs } => compare(&values, *lhs, *rhs, &node.shape, *op)?,
                Op::Logical { op, lhs, rhs } => logical(&values, *lhs, *rhs, &node.shape, *op)?,
                Op::Select {
                    condition,
                    on_true,
                    on_false,
                } => select(
                    &values,
                    *condition,
                    *on_true,
                    *on_false,
                    &node.shape,
                    node.dtype,
                )?,
                Op::Sum { input, axis } => sum(&values[input.index()], *axis)?,
                Op::SumTo { input, shape } => sum_to(&values[input.index()], shape)?,
                Op::Reshape { input, shape } => TensorData::from_scalars(
                    shape.clone(),
                    values[input.index()].dtype(),
                    (0..values[input.index()].len()).map(|i| values[input.index()].scalar_at(i)),
                )?,
                Op::Permute { input, axes } => permute(&values[input.index()], axes)?,
                Op::Expand { input, shape } => expand(&values[input.index()], shape)?,
                Op::Matmul { lhs, rhs } => matmul(&values[lhs.index()], &values[rhs.index()])?,
            };
            debug_assert_eq!(value.shape(), &node.shape);
            values.push(value);
        }
        values.pop().ok_or(Error::UnknownNode(output))
    }
}

fn binary(
    values: &[TensorData],
    lhs: NodeId,
    rhs: NodeId,
    output_shape: &Shape,
    op: BinaryOp,
) -> Result<TensorData> {
    let lhs = values.get(lhs.index()).ok_or(Error::UnknownNode(lhs))?;
    let rhs = values.get(rhs.index()).ok_or(Error::UnknownNode(rhs))?;
    let output_len = output_shape.numel()?;
    let dtype = lhs.dtype().promote(rhs.dtype());
    let mut data = Vec::with_capacity(output_len);
    for linear in 0..output_len {
        let lhs_offset = broadcast_offset(linear, output_shape, lhs.shape());
        let rhs_offset = broadcast_offset(linear, output_shape, rhs.shape());
        data.push(binary_scalar(
            lhs.scalar_at(lhs_offset),
            rhs.scalar_at(rhs_offset),
            dtype,
            op,
        ));
    }
    TensorData::from_scalars(output_shape.clone(), dtype, data)
}

fn compare(
    values: &[TensorData],
    lhs: NodeId,
    rhs: NodeId,
    output_shape: &Shape,
    op: CompareOp,
) -> Result<TensorData> {
    let lhs = values.get(lhs.index()).ok_or(Error::UnknownNode(lhs))?;
    let rhs = values.get(rhs.index()).ok_or(Error::UnknownNode(rhs))?;
    let data = (0..output_shape.numel()?).map(|linear| {
        let lhs = lhs.scalar_at(broadcast_offset(linear, output_shape, lhs.shape()));
        let rhs = rhs.scalar_at(broadcast_offset(linear, output_shape, rhs.shape()));
        Scalar::Bool(compare_scalar(lhs, rhs, op))
    });
    TensorData::from_scalars(output_shape.clone(), DType::Bool, data)
}

fn compare_scalar(lhs: Scalar, rhs: Scalar, op: CompareOp) -> bool {
    use std::cmp::Ordering;
    let ordering = match (lhs, rhs) {
        (Scalar::F(lhs), rhs) => lhs.partial_cmp(&rhs.as_f64()),
        (lhs, Scalar::F(rhs)) => lhs.as_f64().partial_cmp(&rhs),
        (Scalar::I(lhs), Scalar::I(rhs)) => Some(lhs.cmp(&rhs)),
        (Scalar::U(lhs), Scalar::U(rhs)) => Some(lhs.cmp(&rhs)),
        (Scalar::I(lhs), Scalar::U(rhs)) => {
            if lhs < 0 {
                Some(Ordering::Less)
            } else {
                Some((lhs as u64).cmp(&rhs))
            }
        }
        (Scalar::U(lhs), Scalar::I(rhs)) => {
            if rhs < 0 {
                Some(Ordering::Greater)
            } else {
                Some(lhs.cmp(&(rhs as u64)))
            }
        }
        (Scalar::Bool(lhs), Scalar::Bool(rhs)) => Some(lhs.cmp(&rhs)),
        (Scalar::Bool(lhs), rhs) => Some((lhs as u8 as i64).cmp(&rhs.as_i64())),
        (lhs, Scalar::Bool(rhs)) => Some(lhs.as_i64().cmp(&(rhs as u8 as i64))),
    };
    match op {
        CompareOp::Eq => ordering == Some(Ordering::Equal),
        CompareOp::Ne => ordering != Some(Ordering::Equal),
        CompareOp::Lt => ordering == Some(Ordering::Less),
        CompareOp::Le => matches!(ordering, Some(Ordering::Less | Ordering::Equal)),
        CompareOp::Gt => ordering == Some(Ordering::Greater),
        CompareOp::Ge => matches!(ordering, Some(Ordering::Greater | Ordering::Equal)),
    }
}

fn logical(
    values: &[TensorData],
    lhs: NodeId,
    rhs: Option<NodeId>,
    output_shape: &Shape,
    op: LogicalOp,
) -> Result<TensorData> {
    let lhs = values.get(lhs.index()).ok_or(Error::UnknownNode(lhs))?;
    let rhs = rhs
        .map(|id| values.get(id.index()).ok_or(Error::UnknownNode(id)))
        .transpose()?;
    let data = (0..output_shape.numel()?).map(|linear| {
        let lhs = lhs
            .scalar_at(broadcast_offset(linear, output_shape, lhs.shape()))
            .as_bool();
        let value = match (op, rhs) {
            (LogicalOp::Not, None) => !lhs,
            (LogicalOp::And, Some(rhs)) => {
                lhs && rhs
                    .scalar_at(broadcast_offset(linear, output_shape, rhs.shape()))
                    .as_bool()
            }
            (LogicalOp::Or, Some(rhs)) => {
                lhs || rhs
                    .scalar_at(broadcast_offset(linear, output_shape, rhs.shape()))
                    .as_bool()
            }
            _ => unreachable!("graph validates logical operands"),
        };
        Scalar::Bool(value)
    });
    TensorData::from_scalars(output_shape.clone(), DType::Bool, data)
}

fn select(
    values: &[TensorData],
    condition: NodeId,
    on_true: NodeId,
    on_false: NodeId,
    output_shape: &Shape,
    dtype: DType,
) -> Result<TensorData> {
    let condition = values
        .get(condition.index())
        .ok_or(Error::UnknownNode(condition))?;
    let on_true = values
        .get(on_true.index())
        .ok_or(Error::UnknownNode(on_true))?;
    let on_false = values
        .get(on_false.index())
        .ok_or(Error::UnknownNode(on_false))?;
    let data = (0..output_shape.numel()?).map(|linear| {
        let condition = condition
            .scalar_at(broadcast_offset(linear, output_shape, condition.shape()))
            .as_bool();
        if condition {
            on_true.scalar_at(broadcast_offset(linear, output_shape, on_true.shape()))
        } else {
            on_false.scalar_at(broadcast_offset(linear, output_shape, on_false.shape()))
        }
    });
    TensorData::from_scalars(output_shape.clone(), dtype, data)
}

fn binary_scalar(lhs: Scalar, rhs: Scalar, dtype: DType, op: BinaryOp) -> Scalar {
    if dtype.is_float() {
        let (lhs, rhs) = (lhs.as_f64(), rhs.as_f64());
        return Scalar::F(match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
        });
    }
    if matches!(dtype, DType::Bool) {
        let (lhs, rhs) = (lhs.as_bool(), rhs.as_bool());
        return Scalar::Bool(match op {
            BinaryOp::Add => lhs || rhs,
            BinaryOp::Sub => lhs ^ rhs,
            BinaryOp::Mul => lhs && rhs,
            BinaryOp::Div => lhs && rhs,
        });
    }
    if matches!(dtype.category(), crate::DTypeCategory::Unsigned) {
        let (lhs, rhs) = (lhs.as_u64(), rhs.as_u64());
        return Scalar::U(match op {
            BinaryOp::Add => lhs.wrapping_add(rhs),
            BinaryOp::Sub => lhs.wrapping_sub(rhs),
            BinaryOp::Mul => lhs.wrapping_mul(rhs),
            BinaryOp::Div => lhs / rhs,
        });
    }
    let (lhs, rhs) = (lhs.as_i64(), rhs.as_i64());
    Scalar::I(match op {
        BinaryOp::Add => lhs.wrapping_add(rhs),
        BinaryOp::Sub => lhs.wrapping_sub(rhs),
        BinaryOp::Mul => lhs.wrapping_mul(rhs),
        BinaryOp::Div => lhs / rhs,
    })
}

fn unary(input: &TensorData, op: UnaryOp) -> Result<TensorData> {
    let values = (0..input.len()).map(|index| {
        let value = input.scalar_at(index).as_f64();
        Scalar::F(match op {
            UnaryOp::Neg => -value,
            UnaryOp::Exp => value.exp(),
            UnaryOp::Log => value.ln(),
            UnaryOp::Relu => value.max(0.0),
            UnaryOp::Step => {
                if value > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        })
    });
    TensorData::from_scalars(input.shape().clone(), input.dtype(), values)
}

fn sum(input: &TensorData, axis: usize) -> Result<TensorData> {
    let dims = input.shape().dims();
    let output_shape = input.shape().without_axis(axis).ok_or(Error::InvalidAxis {
        node: NodeId(usize::MAX),
        axis,
        rank: dims.len(),
    })?;
    let outer = Shape::new(dims[..axis].to_vec()).numel()?;
    let inner = Shape::new(dims[axis + 1..].to_vec()).numel()?;
    let axis_len = dims[axis];
    let mut output = vec![Scalar::I(0); outer * inner];
    for o in 0..outer {
        for a in 0..axis_len {
            for i in 0..inner {
                output[o * inner + i] = binary_scalar(
                    output[o * inner + i],
                    input.scalar_at((o * axis_len + a) * inner + i),
                    input.dtype(),
                    BinaryOp::Add,
                );
            }
        }
    }
    TensorData::from_scalars(output_shape, input.dtype(), output)
}

fn expand(input: &TensorData, output_shape: &Shape) -> Result<TensorData> {
    let output: Vec<_> = (0..output_shape.numel()?)
        .map(|linear| input.scalar_at(broadcast_offset(linear, output_shape, input.shape())))
        .collect();
    TensorData::from_scalars(output_shape.clone(), input.dtype(), output)
}

fn sum_to(input: &TensorData, output_shape: &Shape) -> Result<TensorData> {
    let input_shape = input.shape();
    let input_strides = input_shape.contiguous_strides();
    let output_strides = output_shape.contiguous_strides();
    let padding = input_shape.rank() - output_shape.rank();
    let mut output = vec![Scalar::I(0); output_shape.numel()?];
    for linear in 0..input.len() {
        let mut output_offset = 0;
        for (output_axis, output_stride) in output_strides.iter().enumerate() {
            let input_axis = output_axis + padding;
            let coordinate = (linear / input_strides[input_axis]) % input_shape.dims()[input_axis];
            if output_shape.dims()[output_axis] != 1 {
                output_offset += coordinate * output_stride;
            }
        }
        output[output_offset] = binary_scalar(
            output[output_offset],
            input.scalar_at(linear),
            input.dtype(),
            BinaryOp::Add,
        );
    }
    TensorData::from_scalars(output_shape.clone(), input.dtype(), output)
}

fn broadcast_offset(linear: usize, output_shape: &Shape, input_shape: &Shape) -> usize {
    let output_strides = output_shape.contiguous_strides();
    let input_strides = input_shape.contiguous_strides();
    let padding = output_shape.rank() - input_shape.rank();
    output_strides
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis >= padding)
        .map(|(axis, output_stride)| {
            let input_axis = axis - padding;
            if input_shape.dims()[input_axis] == 1 {
                0
            } else {
                let coordinate = (linear / output_stride) % output_shape.dims()[axis];
                coordinate * input_strides[input_axis]
            }
        })
        .sum()
}

fn permute(input: &TensorData, axes: &[usize]) -> Result<TensorData> {
    let output_shape = Shape::new(
        axes.iter()
            .map(|axis| input.shape().dims()[*axis])
            .collect::<Vec<_>>(),
    );
    let output_strides = output_shape.contiguous_strides();
    let input_strides = input.shape().contiguous_strides();
    let mut output = vec![Scalar::I(0); input.len()];
    for (linear, slot) in output.iter_mut().enumerate() {
        let input_offset = axes
            .iter()
            .enumerate()
            .map(|(output_axis, input_axis)| {
                let coordinate =
                    (linear / output_strides[output_axis]) % output_shape.dims()[output_axis];
                coordinate * input_strides[*input_axis]
            })
            .sum::<usize>();
        *slot = input.scalar_at(input_offset);
    }
    TensorData::from_scalars(output_shape, input.dtype(), output)
}

fn matmul(lhs: &TensorData, rhs: &TensorData) -> Result<TensorData> {
    let m = lhs.shape().dims()[0];
    let k = lhs.shape().dims()[1];
    let n = rhs.shape().dims()[1];
    let dtype = lhs.dtype().promote(rhs.dtype());
    let mut output = vec![Scalar::I(0); m * n];
    for row in 0..m {
        for column in 0..n {
            for inner in 0..k {
                let product = binary_scalar(
                    lhs.scalar_at(row * k + inner),
                    rhs.scalar_at(inner * n + column),
                    dtype,
                    BinaryOp::Mul,
                );
                output[row * n + column] =
                    binary_scalar(output[row * n + column], product, dtype, BinaryOp::Add);
            }
        }
    }
    TensorData::from_scalars([m, n], dtype, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
        TensorData::new(shape, values.to_vec()).unwrap()
    }

    #[test]
    fn evaluates_elementwise_and_reduction_graph() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 2]);
        let y = graph.input("y", [2, 2]);
        let product = graph.mul(x, y).unwrap();
        let shifted = graph.add(product, y).unwrap();
        let output = graph.sum(shifted, 1).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([2, 2], &[1., 2., 3., 4.])),
            ("y".into(), data([2, 2], &[5., 6., 7., 8.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, output, &inputs).unwrap(),
            data([2], &[28., 68.])
        );
    }

    #[test]
    fn trace_is_inspectable() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let two = graph.constant(data([2], &[2., 2.]));
        let output = graph.mul(x, two).unwrap();
        assert_eq!(
            graph.trace(output).unwrap().to_string(),
            "%0 = input(\"x\") : [2]\n%1 = constant : [2]\n%2 = mul(%0, %1) : [2]\nreturn %2"
        );
    }

    #[test]
    fn broadcasts_trailing_dimensions_and_scalars() {
        let mut graph = Graph::new();
        let matrix = graph.input("matrix", [2, 3]);
        let row = graph.input("row", [3]);
        let scalar = graph.constant(TensorData::scalar(2.0));
        let shifted = graph.add(matrix, row).unwrap();
        let output = graph.mul(shifted, scalar).unwrap();
        let inputs = HashMap::from([
            ("matrix".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
            ("row".into(), data([3], &[10., 20., 30.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, output, &inputs).unwrap(),
            data([2, 3], &[22., 44., 66., 28., 50., 72.])
        );
    }

    #[test]
    fn reshapes_and_permutes_without_changing_values() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let transposed = graph.permute(input, [1, 0]).unwrap();
        let output = graph.reshape(transposed, [6]).unwrap();
        let inputs = HashMap::from([("x".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.]))]);
        assert_eq!(
            CpuBackend.execute(&graph, output, &inputs).unwrap(),
            data([6], &[1., 4., 2., 5., 3., 6.])
        );
    }

    #[test]
    fn multiplies_rank_two_matrices() {
        let mut graph = Graph::new();
        let lhs = graph.input("lhs", [2, 3]);
        let rhs = graph.input("rhs", [3, 2]);
        let output = graph.matmul(lhs, rhs).unwrap();
        let inputs = HashMap::from([
            ("lhs".into(), data([2, 3], &[1., 2., 3., 4., 5., 6.])),
            ("rhs".into(), data([3, 2], &[7., 8., 9., 10., 11., 12.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, output, &inputs).unwrap(),
            data([2, 2], &[58., 64., 139., 154.])
        );
    }

    #[test]
    fn rejects_invalid_movement_and_matmul_shapes() {
        let mut graph = Graph::new();
        let matrix = graph.input("matrix", [2, 3]);
        let other = graph.input("other", [4, 2]);
        assert!(matches!(
            graph.reshape(matrix, [5]),
            Err(Error::InvalidReshape { .. })
        ));
        assert!(matches!(
            graph.permute(matrix, [0, 0]),
            Err(Error::InvalidPermutation { .. })
        ));
        assert!(matches!(
            graph.matmul(matrix, other),
            Err(Error::InvalidMatmul { .. })
        ));
    }

    #[test]
    fn evaluates_unary_and_binary_alu() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let y = graph.input("y", [3]);
        let quotient = graph.div(x, y).unwrap();
        let shifted = graph.sub(quotient, y).unwrap();
        let negated = graph.neg(shifted).unwrap();
        let output = graph.relu(negated).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([3], &[2., 8., -3.])),
            ("y".into(), data([3], &[2., 4., 1.])),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, output, &inputs).unwrap(),
            data([3], &[1., 2., 4.])
        );
    }

    #[test]
    fn exp_and_log_round_trip_positive_values() {
        let mut graph = Graph::new();
        let x = graph.input("x", [3]);
        let logged = graph.log(x).unwrap();
        let output = graph.exp(logged).unwrap();
        let inputs = HashMap::from([("x".into(), data([3], &[0.5, 1.0, 4.0]))]);
        let actual = CpuBackend.execute(&graph, output, &inputs).unwrap();
        for (actual, expected) in actual.values().iter().zip([0.5, 1.0, 4.0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn promotes_and_executes_mixed_exact_integer_storage() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2], DType::I8);
        let rhs = graph.input_dtype("rhs", [2], DType::U8);
        let output = graph.add(lhs, rhs).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), DType::I16);
        let inputs = HashMap::from([
            (
                "lhs".into(),
                TensorData::from_scalars([2], DType::I8, [Scalar::I(-2), Scalar::I(100)]).unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::from_scalars([2], DType::U8, [Scalar::U(3), Scalar::U(200)]).unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&graph, output, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::I16(vec![1, 300])
        );
    }

    #[test]
    fn cast_nodes_and_input_dtypes_are_checked() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2], DType::I64);
        let output = graph.cast(input, DType::F32).unwrap();
        let inputs = HashMap::from([(
            "x".into(),
            TensorData::from_scalars([2], DType::I64, [Scalar::I(7), Scalar::I(-3)]).unwrap(),
        )]);
        assert_eq!(
            CpuBackend.execute(&graph, output, &inputs).unwrap().dtype(),
            DType::F32
        );
        let wrong = HashMap::from([("x".into(), TensorData::new([2], vec![7.0, -3.0]).unwrap())]);
        assert!(matches!(
            CpuBackend.execute(&graph, output, &wrong),
            Err(Error::InputDType { .. })
        ));
    }

    #[test]
    fn predicates_logic_and_select_broadcast_exact_storage() {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 1], DType::I64);
        let rhs = graph.input_dtype("rhs", [2], DType::U64);
        let condition = graph.lt(lhs, rhs).unwrap();
        assert_eq!(graph.dtype(condition).unwrap(), DType::Bool);
        let selected = graph.select(condition, lhs, rhs).unwrap();
        assert_eq!(graph.shape(selected).unwrap(), &Shape::from([2, 2]));
        assert_eq!(graph.dtype(selected).unwrap(), DType::F64);
        let inputs = HashMap::from([
            (
                "lhs".into(),
                TensorData::from_scalars([2, 1], DType::I64, [Scalar::I(-1), Scalar::I(5)])
                    .unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::from_scalars([2], DType::U64, [Scalar::U(0), Scalar::U(4)]).unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&graph, selected, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::F64(vec![-1.0, -1.0, 0.0, 4.0])
        );

        let mut logical_graph = Graph::new();
        let a = logical_graph.constant(
            TensorData::from_scalars([2], DType::Bool, [Scalar::Bool(true), Scalar::Bool(false)])
                .unwrap(),
        );
        let b = logical_graph.logical_not(a).unwrap();
        let both = logical_graph.logical_and(a, b).unwrap();
        assert_eq!(
            CpuBackend
                .execute(&logical_graph, both, &HashMap::new())
                .unwrap()
                .storage(),
            &crate::Storage::Bool(vec![false, false])
        );
    }

    #[test]
    fn comparisons_define_nan_and_invalid_logical_contracts() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2]);
        let y = graph.input("y", [2]);
        let equal = graph.eq(x, y).unwrap();
        let unequal = graph.ne(x, y).unwrap();
        let inputs = HashMap::from([
            ("x".into(), data([2], &[f32::NAN, 2.0])),
            ("y".into(), data([2], &[f32::NAN, 2.0])),
        ]);
        assert_eq!(
            CpuBackend
                .execute(&graph, equal, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::Bool(vec![false, true])
        );
        assert_eq!(
            CpuBackend
                .execute(&graph, unequal, &inputs)
                .unwrap()
                .storage(),
            &crate::Storage::Bool(vec![true, false])
        );
        assert!(matches!(
            graph.logical_not(x),
            Err(Error::InvalidLogicalDType { .. })
        ));
        assert!(matches!(
            graph.select(x, x, y),
            Err(Error::InvalidLogicalDType { .. })
        ));
        assert!(
            graph
                .trace(equal)
                .unwrap()
                .to_string()
                .contains("eq(%0, %1)")
        );
    }
}
