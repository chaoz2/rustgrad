use super::{Graph, NodeId};
use crate::{Error, Result, TensorData};

impl Graph {
    /// Reduces multiple axes. Axes refer to the original input rank.
    pub fn sum_axes(
        &mut self,
        input: NodeId,
        axes: impl IntoIterator<Item = usize>,
    ) -> Result<NodeId> {
        let rank = self.node(input)?.shape.rank();
        let mut axes = axes.into_iter().collect::<Vec<_>>();
        axes.sort_unstable();
        if axes.iter().any(|axis| *axis >= rank) || axes.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidReductionAxes {
                node: input,
                axes,
                rank,
            });
        }

        let mut output = input;
        for axis in axes.into_iter().rev() {
            output = self.sum(output, axis)?;
        }
        Ok(output)
    }

    pub fn sum_all(&mut self, input: NodeId) -> Result<NodeId> {
        let rank = self.node(input)?.shape.rank();
        self.sum_axes(input, 0..rank)
    }

    pub fn mean(&mut self, input: NodeId, axis: usize) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let count = *shape.dims().get(axis).ok_or(Error::InvalidAxis {
            node: input,
            axis,
            rank: shape.rank(),
        })?;
        let sum = self.sum(input, axis)?;
        let divisor = self.constant(TensorData::scalar(count as f32));
        self.div(sum, divisor)
    }

    pub fn mean_axes(
        &mut self,
        input: NodeId,
        axes: impl IntoIterator<Item = usize>,
    ) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let mut axes = axes.into_iter().collect::<Vec<_>>();
        axes.sort_unstable();
        if axes.iter().any(|axis| *axis >= shape.rank())
            || axes.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(Error::InvalidReductionAxes {
                node: input,
                axes,
                rank: shape.rank(),
            });
        }
        let count = axes
            .iter()
            .try_fold(1usize, |count, axis| count.checked_mul(shape.dims()[*axis]))
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let sum = self.sum_axes(input, axes)?;
        let divisor = self.constant(TensorData::scalar(count as f32));
        self.div(sum, divisor)
    }

    pub fn mean_all(&mut self, input: NodeId) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let sum = self.sum_all(input)?;
        let divisor = self.constant(TensorData::scalar(shape.numel()? as f32));
        self.div(sum, divisor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, Shape};
    use std::collections::HashMap;

    fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
        TensorData::new(shape, values.to_vec()).unwrap()
    }

    #[test]
    fn multi_axis_sum_and_mean_are_composable_and_differentiable() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 2, 2]);
        let sum = graph.sum_axes(x, [0, 2]).unwrap();
        let mean = graph.mean_all(x).unwrap();
        let dx = graph.grad(mean, x).unwrap();
        let inputs = HashMap::from([(
            "x".into(),
            data([2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]),
        )]);

        assert_eq!(
            CpuBackend.execute(&graph, sum, &inputs).unwrap(),
            data([2], &[14., 22.])
        );
        assert_eq!(
            CpuBackend.execute(&graph, mean, &inputs).unwrap(),
            TensorData::scalar(4.5)
        );
        assert_eq!(
            CpuBackend.execute(&graph, dx, &inputs).unwrap(),
            data([2, 2, 2], &[0.125; 8])
        );
    }

    #[test]
    fn rejects_duplicate_reduction_axes() {
        let mut graph = Graph::new();
        let x = graph.input("x", [2, 3]);
        assert_eq!(
            graph.sum_axes(x, [1, 1]),
            Err(Error::InvalidReductionAxes {
                node: x,
                axes: vec![1, 1],
                rank: 2
            })
        );
    }
}
