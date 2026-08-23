use super::{Graph, NodeId};
use crate::{Result, Shape, TensorData};

impl Graph {
    pub fn full(&mut self, shape: impl Into<Shape>, value: f32) -> Result<NodeId> {
        Ok(self.constant(TensorData::full(shape, value)?))
    }

    pub fn zeros(&mut self, shape: impl Into<Shape>) -> Result<NodeId> {
        Ok(self.constant(TensorData::zeros(shape)?))
    }

    pub fn ones(&mut self, shape: impl Into<Shape>) -> Result<NodeId> {
        Ok(self.constant(TensorData::ones(shape)?))
    }

    pub fn arange(&mut self, start: i64, end: i64, step: i64) -> Result<NodeId> {
        Ok(self.constant(TensorData::arange(start, end, step)?))
    }
}
