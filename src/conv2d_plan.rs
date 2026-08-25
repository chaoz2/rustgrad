//! Immutable CPU-JIT contract for the deliberately narrow static 1x1 NCHW
//! convolution used by the public configured-CIFAR module workflow.

use crate::{DType, Graph, NodeId, Op, Shape, TensorData};
use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
};

/// A validated, serial 1x1 F32 NCHW convolution. This is intentionally not a
/// general convolution plan: every omitted geometry remains a scheduler/native
/// preflight error rather than acquiring accidental semantics.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StaticConv2dPlan {
    pub input: NodeId,
    pub weight: NodeId,
    pub bias: Option<NodeId>,
    pub output: NodeId,
    pub input_shape: Shape,
    pub weight_shape: Shape,
    pub bias_shape: Option<Shape>,
    pub output_shape: Shape,
    pub batch: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub height: usize,
    pub width: usize,
    pub cache_key: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StaticConv2dPlanError {
    NotConv2d,
    Geometry(&'static str),
    DType,
    Overflow,
}

impl fmt::Display for StaticConv2dPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "static 1x1 conv2d plan error: {self:?}")
    }
}
impl std::error::Error for StaticConv2dPlanError {}

impl StaticConv2dPlan {
    pub fn from_graph(graph: &Graph, output: NodeId) -> Result<Self, StaticConv2dPlanError> {
        let Op::Conv2d {
            input,
            weight,
            bias,
            options,
        } = graph
            .op(output)
            .map_err(|_| StaticConv2dPlanError::NotConv2d)?
        else {
            return Err(StaticConv2dPlanError::NotConv2d);
        };
        if options.groups != 1 {
            return Err(StaticConv2dPlanError::Geometry("groups must be one"));
        }
        if options.stride != [1, 1] || options.dilation != [1, 1] || options.padding != [0, 0, 0, 0]
        {
            return Err(StaticConv2dPlanError::Geometry(
                "requires stride/dilation one and zero padding",
            ));
        }
        let input_shape = graph
            .shape(*input)
            .map_err(|_| StaticConv2dPlanError::Geometry("input"))?
            .clone();
        let weight_shape = graph
            .shape(*weight)
            .map_err(|_| StaticConv2dPlanError::Geometry("weight"))?
            .clone();
        let output_shape = graph
            .shape(output)
            .map_err(|_| StaticConv2dPlanError::Geometry("output"))?
            .clone();
        if input_shape.rank() != 4 || weight_shape.rank() != 4 || output_shape.rank() != 4 {
            return Err(StaticConv2dPlanError::Geometry(
                "requires NCHW/OIHW rank four",
            ));
        }
        if weight_shape.dims()[2..] != [1, 1] {
            return Err(StaticConv2dPlanError::Geometry("kernel must be 1x1"));
        }
        let [batch, input_channels, height, width]: [usize; 4] = input_shape
            .dims()
            .try_into()
            .map_err(|_| StaticConv2dPlanError::Geometry("input rank"))?;
        let [output_channels, weight_channels, _, _]: [usize; 4] =
            weight_shape
                .dims()
                .try_into()
                .map_err(|_| StaticConv2dPlanError::Geometry("weight rank"))?;
        if input_channels != weight_channels
            || output_shape.dims() != [batch, output_channels, height, width]
        {
            return Err(StaticConv2dPlanError::Geometry("NCHW output geometry"));
        }
        if graph
            .dtype(*input)
            .map_err(|_| StaticConv2dPlanError::DType)?
            != DType::F32
            || graph
                .dtype(*weight)
                .map_err(|_| StaticConv2dPlanError::DType)?
                != DType::F32
            || graph
                .dtype(output)
                .map_err(|_| StaticConv2dPlanError::DType)?
                != DType::F32
        {
            return Err(StaticConv2dPlanError::DType);
        }
        let bias_shape = if let Some(bias) = bias {
            if graph
                .dtype(*bias)
                .map_err(|_| StaticConv2dPlanError::DType)?
                != DType::F32
            {
                return Err(StaticConv2dPlanError::DType);
            }
            let shape = graph
                .shape(*bias)
                .map_err(|_| StaticConv2dPlanError::Geometry("bias"))?
                .clone();
            if shape.dims() != [output_channels] {
                return Err(StaticConv2dPlanError::Geometry(
                    "bias must be output channels",
                ));
            }
            Some(shape)
        } else {
            None
        };
        input_shape
            .numel()
            .and_then(|_| weight_shape.numel())
            .and_then(|_| output_shape.numel())
            .map_err(|_| StaticConv2dPlanError::Overflow)?;
        let mut plan = Self {
            input: *input,
            weight: *weight,
            bias: *bias,
            output,
            input_shape,
            weight_shape,
            bias_shape,
            output_shape,
            batch,
            input_channels,
            output_channels,
            height,
            width,
            cache_key: 0,
        };
        plan.cache_key = plan.expected_cache_key();
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), StaticConv2dPlanError> {
        if self.input == self.weight
            || self.input == self.output
            || self.weight == self.output
            || self.bias.is_some_and(|bias| {
                bias == self.input || bias == self.weight || bias == self.output
            })
        {
            return Err(StaticConv2dPlanError::Geometry("node identities"));
        }
        if self.input_shape.rank() != 4
            || self.weight_shape.rank() != 4
            || self.output_shape.rank() != 4
            || self.weight_shape.dims()[2..] != [1, 1]
            || self.input_shape.dims() != [self.batch, self.input_channels, self.height, self.width]
            || self.weight_shape.dims() != [self.output_channels, self.input_channels, 1, 1]
            || self.output_shape.dims()
                != [self.batch, self.output_channels, self.height, self.width]
            || self
                .bias_shape
                .as_ref()
                .is_some_and(|shape| shape.dims() != [self.output_channels])
        {
            return Err(StaticConv2dPlanError::Geometry("redundant geometry"));
        }
        self.input_shape
            .numel()
            .and_then(|_| self.weight_shape.numel())
            .and_then(|_| self.output_shape.numel())
            .map_err(|_| StaticConv2dPlanError::Overflow)?;
        if self.cache_key != self.expected_cache_key() {
            return Err(StaticConv2dPlanError::Geometry("cache key"));
        }
        Ok(())
    }

    pub fn abi_nodes(&self) -> Vec<NodeId> {
        let mut nodes = vec![self.input, self.weight];
        if let Some(bias) = self.bias {
            nodes.push(bias);
        }
        nodes.push(self.output);
        nodes
    }

    fn expected_cache_key(&self) -> u64 {
        let mut plan = self.clone();
        plan.cache_key = 0;
        let mut hasher = DefaultHasher::new();
        plan.hash(&mut hasher);
        hasher.finish()
    }

    pub fn execute(
        &self,
        input: &TensorData,
        weight: &TensorData,
        bias: Option<&TensorData>,
    ) -> Result<TensorData, StaticConv2dPlanError> {
        self.validate()?;
        if input.shape() != &self.input_shape
            || weight.shape() != &self.weight_shape
            || bias.map(TensorData::shape) != self.bias_shape.as_ref()
            || input.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || bias.is_some_and(|value| value.dtype() != DType::F32)
        {
            return Err(StaticConv2dPlanError::Geometry("tensor descriptors"));
        }
        let mut values = Vec::with_capacity(
            self.output_shape
                .numel()
                .map_err(|_| StaticConv2dPlanError::Overflow)?,
        );
        for n in 0..self.batch {
            for oc in 0..self.output_channels {
                for y in 0..self.height {
                    for x in 0..self.width {
                        let mut acc = bias.map_or(0.0, |value| value.scalar_at(oc).as_f64() as f32);
                        for ic in 0..self.input_channels {
                            let input_offset =
                                ((n * self.input_channels + ic) * self.height + y) * self.width + x;
                            let weight_offset = oc * self.input_channels + ic;
                            acc += input.scalar_at(input_offset).as_f64() as f32
                                * weight.scalar_at(weight_offset).as_f64() as f32;
                        }
                        values.push(crate::Scalar::F(acc as f64));
                    }
                }
            }
        }
        TensorData::from_scalars(self.output_shape.clone(), DType::F32, values)
            .map_err(|_| StaticConv2dPlanError::DType)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, Conv2dOptions, CpuBackend};
    use std::collections::HashMap;

    fn fixture(options: Conv2dOptions, dtype: DType) -> (Graph, NodeId, NodeId, NodeId, NodeId) {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1, 3, 2, 2], dtype);
        let weight = graph.input_dtype("weight", [2, 3, 1, 1], dtype);
        let bias = graph.input_dtype("bias", [2], dtype);
        let output = graph.conv2d(input, weight, Some(bias), options).unwrap();
        (graph, input, weight, bias, output)
    }

    #[test]
    fn static_1x1_plan_is_deterministic_and_matches_cpu_oracle() {
        let (graph, _input, _weight, _bias, output) = fixture(Conv2dOptions::default(), DType::F32);
        let first = StaticConv2dPlan::from_graph(&graph, output).unwrap();
        let second = StaticConv2dPlan::from_graph(&graph, output).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.output_shape.dims(), &[1, 2, 2, 2]);
        let input_data =
            TensorData::new([1, 3, 2, 2], (1..=12).map(|value| value as f32).collect()).unwrap();
        let weight_data = TensorData::new([2, 3, 1, 1], vec![1., 2., -1., -2., 1., 0.5]).unwrap();
        let bias_data = TensorData::new([2], vec![0.5, -1.]).unwrap();
        let planned = first
            .execute(&input_data, &weight_data, Some(&bias_data))
            .unwrap();
        let cpu = CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([
                    ("input".into(), input_data),
                    ("weight".into(), weight_data),
                    ("bias".into(), bias_data),
                ]),
            )
            .unwrap();
        assert_eq!(planned.storage(), cpu.storage());
    }

    #[test]
    fn static_1x1_plan_rejects_non_contract_geometry_and_dtype() {
        let cases = [
            (
                "stride",
                Conv2dOptions {
                    stride: [2, 1],
                    ..Conv2dOptions::default()
                },
                DType::F32,
            ),
            (
                "padding",
                Conv2dOptions {
                    padding: [1, 0, 0, 0],
                    ..Conv2dOptions::default()
                },
                DType::F32,
            ),
        ];
        for (name, options, dtype) in cases {
            let (graph, _, _, _, output) = fixture(options, dtype);
            assert!(
                StaticConv2dPlan::from_graph(&graph, output).is_err(),
                "{name}"
            );
        }
        let mut grouped = Graph::new();
        let input = grouped.input_dtype("input", [1, 2, 2, 2], DType::F32);
        let weight = grouped.input_dtype("weight", [2, 1, 1, 1], DType::F32);
        let output = grouped
            .conv2d(
                input,
                weight,
                None,
                Conv2dOptions {
                    groups: 2,
                    ..Conv2dOptions::default()
                },
            )
            .unwrap();
        assert!(
            StaticConv2dPlan::from_graph(&grouped, output).is_err(),
            "groups"
        );
        let (graph, _, _, _, output) = fixture(Conv2dOptions::default(), DType::F64);
        assert_eq!(
            StaticConv2dPlan::from_graph(&graph, output),
            Err(StaticConv2dPlanError::DType)
        );
    }
}
