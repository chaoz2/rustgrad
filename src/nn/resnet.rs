//! Checked-in tinygrad ResNet and ResNeXt static module composition.

use super::{
    BatchNorm2d, Conv2d, Linear, Mode, ModeModuleForward, Module, Parameter, PendingModeEffects,
    StateKind, init::InitCursor, norm::visit_source_batch_norm, state::join,
};
use crate::{
    Conv2dOptions, DType, Error, Graph, NodeId, Pool2dOptions, ReduceKind, Result, Shape,
    TensorData,
};

const POINTWISE: Conv2dOptions = Conv2dOptions {
    groups: 1,
    stride: [1; 2],
    dilation: [1; 2],
    padding: [0; 4],
};

/// The five source-defined static ResNet depths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResNetDepth {
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
}

impl ResNetDepth {
    const fn blocks(self) -> [usize; 4] {
        match self {
            Self::ResNet18 => [2, 2, 2, 2],
            Self::ResNet34 => [3, 4, 6, 3],
            Self::ResNet50 => [3, 4, 6, 3],
            Self::ResNet101 => [3, 4, 23, 3],
            Self::ResNet152 => [3, 8, 36, 3],
        }
    }

    const fn bottleneck(self) -> bool {
        !matches!(self, Self::ResNet18 | Self::ResNet34)
    }
}

/// Source-level construction controls for a static ResNet family member.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResNetConfig {
    pub depth: ResNetDepth,
    pub num_classes: Option<usize>,
    pub groups: usize,
    pub width_per_group: usize,
    pub stride_in_1x1: bool,
    pub track_running_stats: bool,
}

/// Standalone construction controls for [`BasicBlock`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BasicBlockConfig {
    pub in_planes: usize,
    pub planes: usize,
    pub stride: usize,
    pub groups: usize,
    pub base_width: usize,
    pub track_running_stats: bool,
}

/// Standalone construction controls for [`Bottleneck`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BottleneckConfig {
    pub in_planes: usize,
    pub planes: usize,
    pub stride: usize,
    pub stride_in_1x1: bool,
    pub groups: usize,
    pub base_width: usize,
    pub track_running_stats: bool,
}

impl Default for ResNetConfig {
    fn default() -> Self {
        Self {
            depth: ResNetDepth::ResNet18,
            num_classes: Some(1000),
            groups: 1,
            width_per_group: 64,
            stride_in_1x1: false,
            track_running_stats: true,
        }
    }
}

/// The source forward's mutually exclusive classifier and feature-only results.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResNetOutput {
    Logits(NodeId),
    Features(Vec<NodeId>),
}

impl ResNetOutput {
    pub const fn logits(&self) -> Option<NodeId> {
        match self {
            Self::Logits(value) => Some(*value),
            Self::Features(_) => None,
        }
    }

    pub fn features(&self) -> Option<&[NodeId]> {
        match self {
            Self::Logits(_) => None,
            Self::Features(values) => Some(values),
        }
    }
}

/// A typed ResNet result with ordered, caller-committed BatchNorm effects.
pub struct ResNetForwardOutput<'a> {
    pub output: ResNetOutput,
    pub pending: PendingModeEffects<'a>,
}

struct PreparedConv {
    weight: TensorData,
    options: Conv2dOptions,
}

impl PreparedConv {
    fn new(
        cursor: &mut InitCursor,
        input: usize,
        output: usize,
        kernel: usize,
        options: Conv2dOptions,
    ) -> Result<Self> {
        if input == 0
            || output == 0
            || kernel == 0
            || options.groups == 0
            || options.stride.contains(&0)
            || options.dilation.contains(&0)
            || input % options.groups != 0
            || output % options.groups != 0
        {
            return Err(Error::InvalidConv2d {
                input: Shape::new([0; 4]),
                weight: Shape::new([0; 4]),
                reason: "invalid ResNet convolution geometry",
            });
        }
        let fan_in = input
            .checked_mul(kernel)
            .and_then(|value| value.checked_mul(kernel))
            .ok_or_else(|| Error::ShapeOverflow(Shape::new([input, kernel, kernel])))?;
        let bound = 1.0 / (fan_in as f32).sqrt();
        Ok(Self {
            weight: cursor.uniform(
                Shape::new([output, input / options.groups, kernel, kernel]),
                -bound,
                bound,
            )?,
            options,
        })
    }

    fn publish(self) -> Conv2d {
        let shape = self.weight.shape().dims();
        Conv2d {
            in_channels: shape[1] * self.options.groups,
            out_channels: shape[0],
            kernel_size: [shape[2], shape[3]],
            weight: Parameter::new(self.weight, true),
            bias: None,
            options: self.options,
        }
    }
}

fn batch_norm(channels: usize, track_running_stats: bool) -> BatchNorm2d {
    BatchNorm2d::new(
        &mut Graph::new(),
        channels,
        1e-5,
        true,
        track_running_stats,
        0.1,
    )
    .expect("ResNet constructor preflighted BatchNorm geometry")
}

fn append_norm<'a>(
    norm: &'a BatchNorm2d,
    graph: &mut Graph,
    input: NodeId,
    mode: Mode,
    pending: &mut PendingModeEffects<'a>,
) -> Result<NodeId> {
    let normalized = norm.forward_mode(graph, input, mode)?;
    pending.append(normalized.pending);
    Ok(normalized.output)
}

struct PreparedBasicBlock {
    conv1: PreparedConv,
    conv2: PreparedConv,
    downsample: Option<PreparedConv>,
    planes: usize,
    track_running_stats: bool,
}

/// Checked-in tinygrad's two-convolution residual block.
pub struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<(Conv2d, BatchNorm2d)>,
}

impl PreparedBasicBlock {
    fn new(
        cursor: &mut InitCursor,
        in_planes: usize,
        planes: usize,
        stride: usize,
        groups: usize,
        base_width: usize,
        track_running_stats: bool,
    ) -> Result<Self> {
        if groups != 1 || base_width != 64 {
            return Err(Error::InvalidConv2d {
                input: Shape::new([0; 4]),
                weight: Shape::new([0; 4]),
                reason: "BasicBlock requires groups=1 and width_per_group=64",
            });
        }
        let conv1 = PreparedConv::new(
            cursor,
            in_planes,
            planes,
            3,
            Conv2dOptions {
                stride: [stride; 2],
                padding: [1; 4],
                ..POINTWISE
            },
        )?;
        let conv2 = PreparedConv::new(
            cursor,
            planes,
            planes,
            3,
            Conv2dOptions {
                padding: [1; 4],
                ..POINTWISE
            },
        )?;
        let downsample = (stride != 1 || in_planes != planes)
            .then(|| {
                PreparedConv::new(
                    cursor,
                    in_planes,
                    planes,
                    1,
                    Conv2dOptions {
                        stride: [stride; 2],
                        ..POINTWISE
                    },
                )
            })
            .transpose()?;
        Ok(Self {
            conv1,
            conv2,
            downsample,
            planes,
            track_running_stats,
        })
    }

    fn publish(self) -> BasicBlock {
        BasicBlock {
            conv1: self.conv1.publish(),
            bn1: batch_norm(self.planes, self.track_running_stats),
            conv2: self.conv2.publish(),
            bn2: batch_norm(self.planes, self.track_running_stats),
            downsample: self.downsample.map(|conv| {
                (
                    conv.publish(),
                    batch_norm(self.planes, self.track_running_stats),
                )
            }),
        }
    }
}

impl BasicBlock {
    /// Constructs one graph-independent source block.
    pub fn new_static(config: BasicBlockConfig, seed: u64) -> Result<Self> {
        let mut cursor = InitCursor::new(seed);
        Ok(PreparedBasicBlock::new(
            &mut cursor,
            config.in_planes,
            config.planes,
            config.stride,
            config.groups,
            config.base_width,
            config.track_running_stats,
        )?
        .publish())
    }

    fn lower<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
        pending: &mut PendingModeEffects<'a>,
    ) -> Result<NodeId> {
        let mut output = self.conv1.forward(graph, input)?;
        output = append_norm(&self.bn1, graph, output, mode, pending)?;
        output = graph.relu(output)?;
        output = self.conv2.forward(graph, output)?;
        output = append_norm(&self.bn2, graph, output, mode, pending)?;
        let residual = if let Some((conv, norm)) = &self.downsample {
            let value = conv.forward(graph, input)?;
            append_norm(norm, graph, value, mode, pending)?
        } else {
            input
        };
        output = graph.add(output, residual)?;
        graph.relu(output)
    }
}

impl Module for BasicBlock {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.conv1.visit(&join(prefix, "conv1"), visitor);
        visit_source_batch_norm(&self.bn1, &join(prefix, "bn1"), visitor);
        self.conv2.visit(&join(prefix, "conv2"), visitor);
        visit_source_batch_norm(&self.bn2, &join(prefix, "bn2"), visitor);
        if let Some((conv, norm)) = &self.downsample {
            conv.visit(&join(prefix, "downsample.0"), visitor);
            visit_source_batch_norm(norm, &join(prefix, "downsample.1"), visitor);
        }
    }
}

impl ModeModuleForward for BasicBlock {
    fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<super::ModeForwardOutput<'a>> {
        let mut candidate = graph.clone();
        let mut pending = PendingModeEffects::empty();
        let output = self.lower(&mut candidate, input, mode, &mut pending)?;
        *graph = candidate;
        Ok(super::ModeForwardOutput { output, pending })
    }
}

struct PreparedBottleneck {
    conv1: PreparedConv,
    conv2: PreparedConv,
    conv3: PreparedConv,
    downsample: Option<PreparedConv>,
    width: usize,
    output: usize,
    track_running_stats: bool,
}

/// Checked-in tinygrad's three-convolution ResNet/ResNeXt bottleneck.
pub struct Bottleneck {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    downsample: Option<(Conv2d, BatchNorm2d)>,
}

impl PreparedBottleneck {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cursor: &mut InitCursor,
        in_planes: usize,
        planes: usize,
        stride: usize,
        stride_in_1x1: bool,
        groups: usize,
        base_width: usize,
        track_running_stats: bool,
    ) -> Result<Self> {
        let scaled = planes as f64 * (base_width as f64 / 64.0);
        if !scaled.is_finite() || scaled > usize::MAX as f64 {
            return Err(Error::ShapeOverflow(Shape::new([planes, base_width])));
        }
        let width = (scaled as usize)
            .checked_mul(groups)
            .ok_or_else(|| Error::ShapeOverflow(Shape::new([planes, groups])))?;
        let output = planes
            .checked_mul(4)
            .ok_or_else(|| Error::ShapeOverflow(Shape::new([planes])))?;
        let conv1 = PreparedConv::new(
            cursor,
            in_planes,
            width,
            1,
            Conv2dOptions {
                stride: [if stride_in_1x1 { stride } else { 1 }; 2],
                ..POINTWISE
            },
        )?;
        let conv2 = PreparedConv::new(
            cursor,
            width,
            width,
            3,
            Conv2dOptions {
                groups,
                stride: [if stride_in_1x1 { 1 } else { stride }; 2],
                padding: [1; 4],
                ..POINTWISE
            },
        )?;
        let conv3 = PreparedConv::new(cursor, width, output, 1, POINTWISE)?;
        let downsample = (stride != 1 || in_planes != output)
            .then(|| {
                PreparedConv::new(
                    cursor,
                    in_planes,
                    output,
                    1,
                    Conv2dOptions {
                        stride: [stride; 2],
                        ..POINTWISE
                    },
                )
            })
            .transpose()?;
        Ok(Self {
            conv1,
            conv2,
            conv3,
            downsample,
            width,
            output,
            track_running_stats,
        })
    }

    fn publish(self) -> Bottleneck {
        Bottleneck {
            conv1: self.conv1.publish(),
            bn1: batch_norm(self.width, self.track_running_stats),
            conv2: self.conv2.publish(),
            bn2: batch_norm(self.width, self.track_running_stats),
            conv3: self.conv3.publish(),
            bn3: batch_norm(self.output, self.track_running_stats),
            downsample: self.downsample.map(|conv| {
                (
                    conv.publish(),
                    batch_norm(self.output, self.track_running_stats),
                )
            }),
        }
    }
}

impl Bottleneck {
    /// Constructs one graph-independent source bottleneck.
    pub fn new_static(config: BottleneckConfig, seed: u64) -> Result<Self> {
        let mut cursor = InitCursor::new(seed);
        Ok(PreparedBottleneck::new(
            &mut cursor,
            config.in_planes,
            config.planes,
            config.stride,
            config.stride_in_1x1,
            config.groups,
            config.base_width,
            config.track_running_stats,
        )?
        .publish())
    }

    fn lower<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
        pending: &mut PendingModeEffects<'a>,
    ) -> Result<NodeId> {
        let mut output = self.conv1.forward(graph, input)?;
        output = append_norm(&self.bn1, graph, output, mode, pending)?;
        output = graph.relu(output)?;
        output = self.conv2.forward(graph, output)?;
        output = append_norm(&self.bn2, graph, output, mode, pending)?;
        output = graph.relu(output)?;
        output = self.conv3.forward(graph, output)?;
        output = append_norm(&self.bn3, graph, output, mode, pending)?;
        let residual = if let Some((conv, norm)) = &self.downsample {
            let value = conv.forward(graph, input)?;
            append_norm(norm, graph, value, mode, pending)?
        } else {
            input
        };
        output = graph.add(output, residual)?;
        graph.relu(output)
    }
}

impl Module for Bottleneck {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.conv1.visit(&join(prefix, "conv1"), visitor);
        visit_source_batch_norm(&self.bn1, &join(prefix, "bn1"), visitor);
        self.conv2.visit(&join(prefix, "conv2"), visitor);
        visit_source_batch_norm(&self.bn2, &join(prefix, "bn2"), visitor);
        self.conv3.visit(&join(prefix, "conv3"), visitor);
        visit_source_batch_norm(&self.bn3, &join(prefix, "bn3"), visitor);
        if let Some((conv, norm)) = &self.downsample {
            conv.visit(&join(prefix, "downsample.0"), visitor);
            visit_source_batch_norm(norm, &join(prefix, "downsample.1"), visitor);
        }
    }
}

impl ModeModuleForward for Bottleneck {
    fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<super::ModeForwardOutput<'a>> {
        let mut candidate = graph.clone();
        let mut pending = PendingModeEffects::empty();
        let output = self.lower(&mut candidate, input, mode, &mut pending)?;
        *graph = candidate;
        Ok(super::ModeForwardOutput { output, pending })
    }
}

enum PreparedBlock {
    Basic(PreparedBasicBlock),
    Bottleneck(PreparedBottleneck),
}

impl PreparedBlock {
    fn publish(self) -> ResNetBlock {
        match self {
            Self::Basic(block) => ResNetBlock::Basic(Box::new(block.publish())),
            Self::Bottleneck(block) => ResNetBlock::Bottleneck(Box::new(block.publish())),
        }
    }
}

/// One stage member in a statically selected ResNet family.
pub enum ResNetBlock {
    Basic(Box<BasicBlock>),
    Bottleneck(Box<Bottleneck>),
}

impl ResNetBlock {
    fn lower<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
        pending: &mut PendingModeEffects<'a>,
    ) -> Result<NodeId> {
        match self {
            Self::Basic(block) => block.lower(graph, input, mode, pending),
            Self::Bottleneck(block) => block.lower(graph, input, mode, pending),
        }
    }
}

impl Module for ResNetBlock {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        match self {
            Self::Basic(block) => block.visit(prefix, visitor),
            Self::Bottleneck(block) => block.visit(prefix, visitor),
        }
    }
}

impl ModeModuleForward for ResNetBlock {
    fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<super::ModeForwardOutput<'a>> {
        let mut candidate = graph.clone();
        let mut pending = PendingModeEffects::empty();
        let output = self.lower(&mut candidate, input, mode, &mut pending)?;
        *graph = candidate;
        Ok(super::ModeForwardOutput { output, pending })
    }
}

struct PreparedResNet {
    config: ResNetConfig,
    stem: PreparedConv,
    layers: [Vec<PreparedBlock>; 4],
    classifier: Option<(TensorData, TensorData)>,
}

/// Checked-in tinygrad's complete static ResNet and ResNeXt-50 family.
pub struct ResNet {
    pub layers: [Vec<ResNetBlock>; 4],
    conv1: Conv2d,
    bn1: BatchNorm2d,
    fc: Option<Linear>,
    config: ResNetConfig,
}

impl PreparedResNet {
    fn new(config: ResNetConfig, seed: u64) -> Result<Self> {
        if !config.depth.bottleneck() && (config.groups != 1 || config.width_per_group != 64) {
            return Err(Error::InvalidConv2d {
                input: Shape::new([0; 4]),
                weight: Shape::new([0; 4]),
                reason: "BasicBlock requires groups=1 and width_per_group=64",
            });
        }
        let mut cursor = InitCursor::new(seed);
        let stem = PreparedConv::new(
            &mut cursor,
            3,
            64,
            7,
            Conv2dOptions {
                stride: [2; 2],
                padding: [3; 4],
                ..POINTWISE
            },
        )?;
        let mut in_planes = 64usize;
        let mut layers: [Vec<PreparedBlock>; 4] = std::array::from_fn(|_| Vec::new());
        for (stage, (&planes, &count)) in [64usize, 128, 256, 512]
            .iter()
            .zip(config.depth.blocks().iter())
            .enumerate()
        {
            for index in 0..count {
                let stride = if stage != 0 && index == 0 { 2 } else { 1 };
                let block = if config.depth.bottleneck() {
                    let block = PreparedBottleneck::new(
                        &mut cursor,
                        in_planes,
                        planes,
                        stride,
                        config.stride_in_1x1,
                        config.groups,
                        config.width_per_group,
                        config.track_running_stats,
                    )?;
                    in_planes = planes
                        .checked_mul(4)
                        .ok_or_else(|| Error::ShapeOverflow(Shape::new([planes])))?;
                    PreparedBlock::Bottleneck(block)
                } else {
                    let block = PreparedBasicBlock::new(
                        &mut cursor,
                        in_planes,
                        planes,
                        stride,
                        config.groups,
                        config.width_per_group,
                        config.track_running_stats,
                    )?;
                    in_planes = planes;
                    PreparedBlock::Basic(block)
                };
                layers[stage].push(block);
            }
        }
        let classifier = config
            .num_classes
            .map(|classes| {
                let bound = 1.0 / (in_planes as f32).sqrt();
                Ok((
                    cursor.uniform(Shape::new([classes, in_planes]), -bound, bound)?,
                    cursor.uniform(Shape::new([classes]), -bound, bound)?,
                ))
            })
            .transpose()?;
        Ok(Self {
            config,
            stem,
            layers,
            classifier,
        })
    }

    fn publish(self) -> ResNet {
        let layers = self.layers.map(|layer| {
            layer
                .into_iter()
                .map(PreparedBlock::publish)
                .collect::<Vec<_>>()
        });
        let fc = self.classifier.map(|(weight, bias)| Linear {
            in_features: weight.shape().dims()[1],
            out_features: weight.shape().dims()[0],
            weight: Parameter::new(weight, true),
            bias: Some(Parameter::new(bias, true)),
        });
        ResNet {
            layers,
            conv1: self.stem.publish(),
            bn1: batch_norm(64, self.config.track_running_stats),
            fc,
            config: self.config,
        }
    }
}

impl ResNet {
    /// Constructs one graph-independent source-shaped family member.
    pub fn new_static(config: ResNetConfig, seed: u64) -> Result<Self> {
        Ok(PreparedResNet::new(config, seed)?.publish())
    }

    pub const fn config(&self) -> ResNetConfig {
        self.config
    }

    fn lower<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<ResNetForwardOutput<'a>> {
        let shape = graph.shape(input)?.clone();
        if shape.rank() != 4 || shape.dims()[1] != 3 {
            return Err(Error::InvalidConv2d {
                input: shape,
                weight: self.conv1.weight.shape()?,
                reason: "ResNet input must be NCHW with three channels",
            });
        }
        let mut pending = PendingModeEffects::empty();
        let mut value = self.conv1.forward(graph, input)?;
        value = append_norm(&self.bn1, graph, value, mode, &mut pending)?;
        value = graph.relu(value)?;
        value = graph.pad(
            value,
            [(0, 0), (0, 0), (1, 1), (1, 1)],
            crate::Scalar::F(0.0),
        )?;
        value = graph.max_pool2d(
            value,
            Pool2dOptions {
                kernel: [3, 3],
                stride: [2, 2],
                dilation: [1, 1],
                padding: [0; 4],
                ceil_mode: false,
                count_include_pad: true,
            },
        )?;
        let mut features = self.fc.is_none().then(Vec::new);
        for layer in &self.layers {
            for block in layer {
                value = block.lower(graph, value, mode, &mut pending)?;
            }
            if let Some(features) = &mut features {
                features.push(value);
            }
        }
        let output = if let Some(fc) = &self.fc {
            value = graph.reduce(value, ReduceKind::Mean, Some(vec![2, 3]), false)?;
            value = graph.cast(value, DType::F32)?;
            ResNetOutput::Logits(fc.forward_source(graph, value)?)
        } else {
            ResNetOutput::Features(features.expect("feature-only configuration"))
        };
        Ok(ResNetForwardOutput { output, pending })
    }

    /// Composes the full forward under an explicit BatchNorm mode.
    pub fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<ResNetForwardOutput<'a>> {
        let mut candidate = graph.clone();
        let output = self.lower(&mut candidate, input, mode)?;
        *graph = candidate;
        Ok(output)
    }

    /// Composes the full forward under the scoped ambient training mode.
    pub fn forward_ambient<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
    ) -> Result<ResNetForwardOutput<'a>> {
        let mode = if crate::TrainingContext::is_training() {
            Mode::Training
        } else {
            Mode::Eval
        };
        self.forward_mode(graph, input, mode)
    }
}

impl Module for ResNet {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.conv1.visit(&join(prefix, "conv1"), visitor);
        visit_source_batch_norm(&self.bn1, &join(prefix, "bn1"), visitor);
        for (stage, layer) in self.layers.iter().enumerate() {
            for (index, block) in layer.iter().enumerate() {
                block.visit(
                    &join(prefix, &format!("layer{}.{index}", stage + 1)),
                    visitor,
                );
            }
        }
        if let Some(fc) = &self.fc {
            fc.visit(&join(prefix, "fc"), visitor);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, Op, TensorData};

    fn zero_parameters(module: &impl Module) {
        let mut parameters = Vec::new();
        module.visit("", &mut |_, parameter, kind| {
            if matches!(kind, StateKind::Parameter) {
                parameters.push(parameter.clone());
            }
        });
        for parameter in parameters {
            let snapshot = parameter.snapshot().unwrap();
            parameter
                .replace(TensorData::zeros(snapshot.shape).unwrap())
                .unwrap();
        }
    }

    #[test]
    fn resnet_family_preserves_source_depth_geometry_names_and_linear() {
        for (depth, expected) in [
            (ResNetDepth::ResNet18, [2, 2, 2, 2]),
            (ResNetDepth::ResNet34, [3, 4, 6, 3]),
            (ResNetDepth::ResNet50, [3, 4, 6, 3]),
            (ResNetDepth::ResNet101, [3, 4, 23, 3]),
            (ResNetDepth::ResNet152, [3, 8, 36, 3]),
        ] {
            assert_eq!(depth.blocks(), expected);
        }
        let model = ResNet::new_static(ResNetConfig::default(), 11).unwrap();
        let state = model.state_dict().unwrap();
        assert!(state.tensors().contains_key("conv1.weight"));
        assert!(state.tensors().contains_key("layer1.0.conv1.weight"));
        assert!(state.tensors().contains_key("layer2.0.downsample.0.weight"));
        assert!(
            state
                .tensors()
                .contains_key("layer2.0.downsample.1.running_mean")
        );
        assert!(state.tensors().contains_key("fc.weight"));
        assert_eq!(
            state.tensors()["fc.weight"].shape(),
            &Shape::new([1000, 512])
        );
        let mut names = Vec::new();
        model.visit("", &mut |name, _, _| names.push(name));
        assert_eq!(
            &names[..6],
            [
                "conv1.weight",
                "bn1.weight",
                "bn1.bias",
                "bn1.num_batches_tracked",
                "bn1.running_mean",
                "bn1.running_var",
            ]
        );
        assert_eq!(&names[names.len() - 2..], ["fc.weight", "fc.bias"]);

        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [0, 3, 32, 32], DType::I32);
        let output = model.forward_mode(&mut graph, input, Mode::Eval).unwrap();
        assert_eq!(
            graph.shape(output.output.logits().unwrap()).unwrap(),
            &Shape::new([0, 1000])
        );
        assert!(
            (0..graph.node_count())
                .all(|index| !matches!(graph.op(NodeId(index)).unwrap(), Op::Matmul { .. }))
        );
    }

    #[test]
    fn resnext_stride_variants_and_feature_only_outputs_are_source_shaped() {
        let v1 = Bottleneck::new_static(
            BottleneckConfig {
                in_planes: 64,
                planes: 64,
                stride: 2,
                stride_in_1x1: true,
                groups: 32,
                base_width: 4,
                track_running_stats: true,
            },
            17,
        )
        .unwrap();
        let v15 = Bottleneck::new_static(
            BottleneckConfig {
                stride_in_1x1: false,
                ..BottleneckConfig {
                    in_planes: 64,
                    planes: 64,
                    stride: 2,
                    stride_in_1x1: true,
                    groups: 32,
                    base_width: 4,
                    track_running_stats: true,
                }
            },
            17,
        )
        .unwrap();
        assert_eq!(v1.conv1.options.stride, [2, 2]);
        assert_eq!(v1.conv2.options.stride, [1, 1]);
        assert_eq!(v15.conv1.options.stride, [1, 1]);
        assert_eq!(v15.conv2.options.stride, [2, 2]);
        assert_eq!(v15.conv2.options.groups, 32);
        assert_eq!(v15.conv2.in_channels, 128);
        assert!(
            BasicBlock::new_static(
                BasicBlockConfig {
                    in_planes: 64,
                    planes: 64,
                    stride: 1,
                    groups: 2,
                    base_width: 64,
                    track_running_stats: true,
                },
                17,
            )
            .is_err()
        );

        let model = ResNet::new_static(
            ResNetConfig {
                depth: ResNetDepth::ResNet50,
                num_classes: None,
                groups: 32,
                width_per_group: 4,
                stride_in_1x1: true,
                ..ResNetConfig::default()
            },
            19,
        )
        .unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [0, 3, 64, 64], DType::F64);
        let output = model.forward_mode(&mut graph, input, Mode::Eval).unwrap();
        let features = output.output.features().unwrap();
        assert_eq!(features.len(), 4);
        assert_eq!(
            graph.shape(features[0]).unwrap(),
            &Shape::new([0, 256, 16, 16])
        );
        assert_eq!(
            graph.shape(features[1]).unwrap(),
            &Shape::new([0, 512, 8, 8])
        );
        assert_eq!(
            graph.shape(features[2]).unwrap(),
            &Shape::new([0, 1024, 4, 4])
        );
        assert_eq!(
            graph.shape(features[3]).unwrap(),
            &Shape::new([0, 2048, 2, 2])
        );
        assert!(output.pending.is_empty());

        let mut training_graph = Graph::new();
        let training_input = training_graph.input("input", [0, 3, 64, 64]);
        let training = model
            .forward_mode(&mut training_graph, training_input, Mode::Training)
            .unwrap();
        assert!(!training.pending.is_empty());
        assert_eq!(training.pending.batchnorm_stat_nodes().len(), 53);
    }

    #[test]
    fn compact_resnet_forward_cpu_vjp_and_atomic_errors() {
        let model = ResNet::new_static(
            ResNetConfig {
                num_classes: Some(2),
                track_running_stats: false,
                ..ResNetConfig::default()
            },
            23,
        )
        .unwrap();
        zero_parameters(&model);
        let mut graph = Graph::new();
        let input = graph.input_dtype_requires_grad("input", [1, 3, 8, 8], DType::F32, true);
        let before = graph.node_count();
        let wrong = graph.input_dtype("wrong", [1, 2, 8, 8], DType::F32);
        let admitted = graph.node_count();
        assert!(model.forward_mode(&mut graph, wrong, Mode::Eval).is_err());
        assert_eq!(graph.node_count(), admitted);
        assert!(admitted > before);
        let output = model
            .forward_mode(&mut graph, input, Mode::Eval)
            .unwrap()
            .output
            .logits()
            .unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([1, 2]));
        let mut bindings = model.input_bindings(&graph).unwrap();
        bindings.insert("input".into(), TensorData::ones([1, 3, 8, 8]).unwrap());
        let realized = CpuBackend.execute(&graph, output, &bindings).unwrap();
        assert_eq!(realized, TensorData::zeros([1, 2]).unwrap());
        let loss = graph.sum_all(output).unwrap();
        let gradient = graph.gradient_default(loss, &[input]).unwrap()[0];
        assert_eq!(graph.shape(gradient).unwrap(), &Shape::new([1, 3, 8, 8]));
    }

    #[test]
    fn malformed_basic_configuration_publishes_no_parameter_identity() {
        let bad = ResNetConfig {
            groups: 2,
            ..ResNetConfig::default()
        };
        assert!(ResNet::new_static(bad, 3).is_err());
        let first = ResNet::new_static(ResNetConfig::default(), 3)
            .unwrap()
            .state_dict()
            .unwrap();
        let second = ResNet::new_static(ResNetConfig::default(), 3)
            .unwrap()
            .state_dict()
            .unwrap();
        assert_eq!(first, second);
    }
}
