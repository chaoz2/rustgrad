use super::Backend;
use crate::{BinaryOp, Error, Graph, NodeId, Op, Result, Shape, TensorData, UnaryOp};
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
        let mut values = Vec::with_capacity(output.index() + 1);
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
                    value.clone()
                }
                Op::Constant(data) => data.clone(),
                Op::Unary { op, input } => unary(&values[input.index()], *op)?,
                Op::Binary { op, lhs, rhs } => binary(&values, *lhs, *rhs, &node.shape, *op)?,
                Op::Sum { input, axis } => sum(&values[input.index()], *axis)?,
                Op::SumTo { input, shape } => sum_to(&values[input.index()], shape)?,
                Op::Reshape { input, shape } => {
                    TensorData::new(shape.clone(), values[input.index()].values().to_vec())?
                }
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
    let mut data = Vec::with_capacity(output_len);
    for linear in 0..output_len {
        let lhs_offset = broadcast_offset(linear, output_shape, lhs.shape());
        let rhs_offset = broadcast_offset(linear, output_shape, rhs.shape());
        let lhs = lhs.values()[lhs_offset];
        let rhs = rhs.values()[rhs_offset];
        data.push(match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
        });
    }
    TensorData::new(output_shape.clone(), data)
}

fn unary(input: &TensorData, op: UnaryOp) -> Result<TensorData> {
    let values = input
        .values()
        .iter()
        .map(|value| match op {
            UnaryOp::Neg => -*value,
            UnaryOp::Exp => value.exp(),
            UnaryOp::Log => value.ln(),
            UnaryOp::Relu => value.max(0.0),
            UnaryOp::Step => {
                if *value > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        })
        .collect();
    TensorData::new(input.shape().clone(), values)
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
    let mut output = vec![0.0; outer * inner];
    for o in 0..outer {
        for a in 0..axis_len {
            for i in 0..inner {
                output[o * inner + i] += input.values()[(o * axis_len + a) * inner + i];
            }
        }
    }
    TensorData::new(output_shape, output)
}

fn expand(input: &TensorData, output_shape: &Shape) -> Result<TensorData> {
    let output = (0..output_shape.numel()?)
        .map(|linear| input.values()[broadcast_offset(linear, output_shape, input.shape())])
        .collect();
    TensorData::new(output_shape.clone(), output)
}

fn sum_to(input: &TensorData, output_shape: &Shape) -> Result<TensorData> {
    let input_shape = input.shape();
    let input_strides = input_shape.contiguous_strides();
    let output_strides = output_shape.contiguous_strides();
    let padding = input_shape.rank() - output_shape.rank();
    let mut output = vec![0.0; output_shape.numel()?];
    for linear in 0..input.values().len() {
        let mut output_offset = 0;
        for (output_axis, output_stride) in output_strides.iter().enumerate() {
            let input_axis = output_axis + padding;
            let coordinate = (linear / input_strides[input_axis]) % input_shape.dims()[input_axis];
            if output_shape.dims()[output_axis] != 1 {
                output_offset += coordinate * output_stride;
            }
        }
        output[output_offset] += input.values()[linear];
    }
    TensorData::new(output_shape.clone(), output)
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
    let mut output = vec![0.0; input.values().len()];
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
        *slot = input.values()[input_offset];
    }
    TensorData::new(output_shape, output)
}

fn matmul(lhs: &TensorData, rhs: &TensorData) -> Result<TensorData> {
    let m = lhs.shape().dims()[0];
    let k = lhs.shape().dims()[1];
    let n = rhs.shape().dims()[1];
    let mut output = vec![0.0; m * n];
    for row in 0..m {
        for column in 0..n {
            output[row * n + column] = (0..k)
                .map(|inner| lhs.values()[row * k + inner] * rhs.values()[inner * n + column])
                .sum();
        }
    }
    TensorData::new([m, n], output)
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
}
