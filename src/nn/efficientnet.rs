//! Checked-in tinygrad EfficientNet and mobile inverted bottleneck composition.

use super::{
    BatchNorm2d, Conv2d, Mode, ModeForwardOutput, ModeModuleForward, Module, Parameter,
    PendingModeEffects, StateKind,
    init::{InitCursor, glorot_uniform_bound},
    norm::visit_source_batch_norm,
    state::join,
};
use crate::{Conv2dOptions, Error, Graph, NodeId, Pool2dOptions, Result, Shape, TensorData};

const GLOBAL_PARAMS: [(f64, f64); 10] = [
    (1.0, 1.0),
    (1.0, 1.1),
    (1.1, 1.2),
    (1.2, 1.4),
    (1.4, 1.8),
    (1.6, 2.2),
    (1.8, 2.6),
    (2.0, 3.1),
    (2.2, 3.6),
    (4.3, 5.3),
];

const POINTWISE: Conv2dOptions = Conv2dOptions {
    groups: 1,
    stride: [1; 2],
    dilation: [1; 2],
    padding: [0; 4],
};

#[derive(Clone, Copy, Debug, PartialEq)]
struct BlockSpec {
    repeats: usize,
    kernel: usize,
    stride: [usize; 2],
    expand_ratio: usize,
    input_filters: usize,
    output_filters: usize,
    se_ratio: f64,
}

const DEFAULT_BLOCKS: [BlockSpec; 7] = [
    BlockSpec {
        repeats: 1,
        kernel: 3,
        stride: [1, 1],
        expand_ratio: 1,
        input_filters: 32,
        output_filters: 16,
        se_ratio: 0.25,
    },
    BlockSpec {
        repeats: 2,
        kernel: 3,
        stride: [2, 2],
        expand_ratio: 6,
        input_filters: 16,
        output_filters: 24,
        se_ratio: 0.25,
    },
    BlockSpec {
        repeats: 2,
        kernel: 5,
        stride: [2, 2],
        expand_ratio: 6,
        input_filters: 24,
        output_filters: 40,
        se_ratio: 0.25,
    },
    BlockSpec {
        repeats: 3,
        kernel: 3,
        stride: [2, 2],
        expand_ratio: 6,
        input_filters: 40,
        output_filters: 80,
        se_ratio: 0.25,
    },
    BlockSpec {
        repeats: 3,
        kernel: 5,
        stride: [1, 1],
        expand_ratio: 6,
        input_filters: 80,
        output_filters: 112,
        se_ratio: 0.25,
    },
    BlockSpec {
        repeats: 4,
        kernel: 5,
        stride: [2, 2],
        expand_ratio: 6,
        input_filters: 112,
        output_filters: 192,
        se_ratio: 0.25,
    },
    BlockSpec {
        repeats: 1,
        kernel: 3,
        stride: [1, 1],
        expand_ratio: 6,
        input_filters: 192,
        output_filters: 320,
        se_ratio: 0.25,
    },
];

const FAST_BLOCKS: [BlockSpec; 1] = [BlockSpec {
    repeats: 1,
    kernel: 9,
    stride: [8, 8],
    expand_ratio: 1,
    input_filters: 32,
    output_filters: 320,
    se_ratio: 0.25,
}];

const SMALL_BLOCKS: [BlockSpec; 4] = [
    BlockSpec {
        repeats: 1,
        kernel: 3,
        stride: [2, 2],
        expand_ratio: 1,
        input_filters: 32,
        output_filters: 40,
        se_ratio: 0.25,
    },
    BlockSpec {
        repeats: 1,
        kernel: 3,
        stride: [2, 2],
        expand_ratio: 1,
        input_filters: 40,
        output_filters: 80,
        se_ratio: 0.25,
    },
    BlockSpec {
        repeats: 1,
        kernel: 3,
        stride: [2, 2],
        expand_ratio: 1,
        input_filters: 80,
        output_filters: 192,
        se_ratio: 0.25,
    },
    BlockSpec {
        repeats: 1,
        kernel: 3,
        stride: [2, 2],
        expand_ratio: 1,
        input_filters: 192,
        output_filters: 320,
        se_ratio: 0.25,
    },
];

/// Source-level EfficientNet construction controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EfficientNetConfig {
    pub number: isize,
    pub classes: usize,
    pub has_se: bool,
    pub track_running_stats: bool,
    pub input_channels: usize,
    pub has_fc_output: bool,
}

impl Default for EfficientNetConfig {
    fn default() -> Self {
        Self {
            number: 0,
            classes: 1000,
            has_se: true,
            track_running_stats: true,
            input_channels: 3,
            has_fc_output: true,
        }
    }
}

/// Public construction controls for one source `MBConvBlock`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MBConvBlockConfig {
    pub kernel_size: usize,
    pub strides: [usize; 2],
    pub expand_ratio: usize,
    pub input_filters: usize,
    pub output_filters: usize,
    pub se_ratio: f64,
    pub has_se: bool,
    pub track_running_stats: bool,
}

impl MBConvBlockConfig {
    fn spec(self) -> BlockSpec {
        BlockSpec {
            repeats: 1,
            kernel: self.kernel_size,
            stride: self.strides,
            expand_ratio: self.expand_ratio,
            input_filters: self.input_filters,
            output_filters: self.output_filters,
            se_ratio: self.se_ratio,
        }
    }
}

struct PreparedConv {
    weight: TensorData,
    bias: Option<TensorData>,
    options: Conv2dOptions,
}

impl PreparedConv {
    fn new(
        cursor: &mut InitCursor,
        input: usize,
        output: usize,
        kernel: usize,
        options: Conv2dOptions,
        bias: bool,
    ) -> Result<Self> {
        let group_input = input
            .checked_div(options.groups)
            .ok_or(Error::InvalidConv2d {
                input: Shape::new([0; 4]),
                weight: Shape::new([0; 4]),
                reason: "EfficientNet convolution groups must be positive",
            })?;
        if output == 0
            || kernel == 0
            || options.groups == 0
            || input % options.groups != 0
            || output % options.groups != 0
            || options.stride.contains(&0)
        {
            return Err(Error::InvalidConv2d {
                input: Shape::new([0; 4]),
                weight: Shape::new([output, group_input, kernel, kernel]),
                reason: "invalid EfficientNet convolution geometry",
            });
        }
        Ok(Self {
            weight: cursor.glorot_uniform(Shape::new([output, group_input, kernel, kernel]))?,
            bias: bias.then(|| TensorData::zeros([output])).transpose()?,
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
            bias: self.bias.map(|data| Parameter::new(data, true)),
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
    .expect("EfficientNet constructor preflighted BatchNorm geometry")
}

struct PreparedMBConvBlock {
    config: MBConvBlockConfig,
    expand: Option<PreparedConv>,
    depthwise: PreparedConv,
    se_reduce: Option<PreparedConv>,
    se_expand: Option<PreparedConv>,
    project: PreparedConv,
    track_running_stats: bool,
}

struct BlockGeometry {
    expanded: usize,
    padding: [usize; 4],
    squeezed: usize,
}

fn block_geometry(spec: BlockSpec, has_se: bool) -> Result<BlockGeometry> {
    if spec.kernel == 0
        || spec.stride.contains(&0)
        || spec.expand_ratio == 0
        || spec.input_filters == 0
        || spec.output_filters == 0
        || (has_se && !spec.se_ratio.is_finite())
    {
        return Err(Error::InvalidConv2d {
            input: Shape::new([0; 4]),
            weight: Shape::new([0; 4]),
            reason: "invalid EfficientNet block geometry",
        });
    }
    let expanded = spec
        .expand_ratio
        .checked_mul(spec.input_filters)
        .ok_or_else(|| Error::ShapeOverflow(Shape::new([spec.input_filters])))?;
    let half = (spec.kernel - 1) / 2;
    let padding = if spec.stride == [2, 2] {
        let before = half.checked_sub(1).ok_or(Error::InvalidConv2d {
            input: Shape::new([0; 4]),
            weight: Shape::new([expanded, 1, spec.kernel, spec.kernel]),
            reason: "source stride-two EfficientNet padding would be negative",
        })?;
        [before, half, before, half]
    } else {
        [half; 4]
    };
    let squeezed = if has_se {
        let value = spec.input_filters as f64 * spec.se_ratio;
        if !value.is_finite() || value > usize::MAX as f64 {
            return Err(Error::ShapeOverflow(Shape::new([spec.input_filters])));
        }
        (value as usize).max(1)
    } else {
        1
    };
    Ok(BlockGeometry {
        expanded,
        padding,
        squeezed,
    })
}

/// One checked-in tinygrad mobile inverted bottleneck block.
pub struct MBConvBlock {
    expand: Option<Conv2d>,
    bn0: Option<BatchNorm2d>,
    depthwise: Conv2d,
    bn1: BatchNorm2d,
    se_reduce: Option<Conv2d>,
    se_expand: Option<Conv2d>,
    project: Conv2d,
    bn2: BatchNorm2d,
    config: MBConvBlockConfig,
}

impl PreparedMBConvBlock {
    fn new(
        spec: BlockSpec,
        has_se: bool,
        track_running_stats: bool,
        cursor: &mut InitCursor,
    ) -> Result<Self> {
        let BlockGeometry {
            expanded,
            padding,
            squeezed,
        } = block_geometry(spec, has_se)?;
        let expand = (spec.expand_ratio != 1)
            .then(|| PreparedConv::new(cursor, spec.input_filters, expanded, 1, POINTWISE, false))
            .transpose()?;
        let depthwise = PreparedConv::new(
            cursor,
            expanded,
            expanded,
            spec.kernel,
            Conv2dOptions {
                groups: expanded,
                stride: spec.stride,
                dilation: [1, 1],
                padding,
            },
            false,
        )?;
        let se_reduce = has_se
            .then(|| PreparedConv::new(cursor, expanded, squeezed, 1, POINTWISE, true))
            .transpose()?;
        let se_expand = has_se
            .then(|| PreparedConv::new(cursor, squeezed, expanded, 1, POINTWISE, true))
            .transpose()?;
        let project =
            PreparedConv::new(cursor, expanded, spec.output_filters, 1, POINTWISE, false)?;
        Ok(Self {
            config: MBConvBlockConfig {
                kernel_size: spec.kernel,
                strides: spec.stride,
                expand_ratio: spec.expand_ratio,
                input_filters: spec.input_filters,
                output_filters: spec.output_filters,
                se_ratio: spec.se_ratio,
                has_se,
                track_running_stats,
            },
            expand,
            depthwise,
            se_reduce,
            se_expand,
            project,
            track_running_stats,
        })
    }

    fn publish(self) -> MBConvBlock {
        let expand = self.expand.map(PreparedConv::publish);
        let bn0 = expand
            .as_ref()
            .map(|conv| batch_norm(conv.out_channels, self.track_running_stats));
        let depthwise = self.depthwise.publish();
        let bn1 = batch_norm(depthwise.out_channels, self.track_running_stats);
        let se_reduce = self.se_reduce.map(PreparedConv::publish);
        let se_expand = self.se_expand.map(PreparedConv::publish);
        let project = self.project.publish();
        let bn2 = batch_norm(project.out_channels, self.track_running_stats);
        MBConvBlock {
            expand,
            bn0,
            depthwise,
            bn1,
            se_reduce,
            se_expand,
            project,
            bn2,
            config: self.config,
        }
    }
}

impl MBConvBlock {
    /// Constructs one graph-independent source-shaped block.
    ///
    /// The complete geometry and every host tensor are prepared before any
    /// `Parameter` identity is published. The explicit seed follows the same
    /// Rust module-initialization contract as [`EfficientNet::new_static`].
    pub fn new_static(config: MBConvBlockConfig, seed: u64) -> Result<Self> {
        let mut cursor = InitCursor::new(seed);
        Ok(PreparedMBConvBlock::new(
            config.spec(),
            config.has_se,
            config.track_running_stats,
            &mut cursor,
        )?
        .publish())
    }

    pub const fn config(&self) -> MBConvBlockConfig {
        self.config
    }

    fn lower<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<ModeForwardOutput<'a>> {
        let mut pending = PendingModeEffects::empty();
        let mut value = input;
        if let (Some(expand), Some(norm)) = (&self.expand, &self.bn0) {
            value = expand.forward(graph, value)?;
            let normalized = norm.forward_mode(graph, value, mode)?;
            value = graph.silu(normalized.output)?;
            pending.append(normalized.pending);
        }
        value = self.depthwise.forward(graph, value)?;
        let normalized = self.bn1.forward_mode(graph, value, mode)?;
        value = graph.silu(normalized.output)?;
        pending.append(normalized.pending);
        if let (Some(reduce), Some(expand)) = (&self.se_reduce, &self.se_expand) {
            let shape = graph.shape(value)?.clone();
            let mut squeezed = graph.avg_pool2d(
                value,
                Pool2dOptions {
                    kernel: [shape.dims()[2], shape.dims()[3]],
                    stride: [shape.dims()[2], shape.dims()[3]],
                    dilation: [1, 1],
                    padding: [0; 4],
                    ceil_mode: false,
                    count_include_pad: true,
                },
            )?;
            squeezed = reduce.forward(graph, squeezed)?;
            squeezed = graph.silu(squeezed)?;
            squeezed = expand.forward(graph, squeezed)?;
            squeezed = graph.sigmoid(squeezed)?;
            value = graph.mul(value, squeezed)?;
        }
        value = self.project.forward(graph, value)?;
        let normalized = self.bn2.forward_mode(graph, value, mode)?;
        value = normalized.output;
        pending.append(normalized.pending);
        if graph.shape(value)? == graph.shape(input)? {
            value = graph.add(value, input)?;
        }
        Ok(ModeForwardOutput {
            output: value,
            pending,
        })
    }
}

impl Module for MBConvBlock {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        if let Some(expand) = &self.expand {
            visitor(
                join(prefix, "_expand_conv"),
                &expand.weight,
                StateKind::Parameter,
            );
        }
        if let Some(norm) = &self.bn0 {
            visit_source_batch_norm(norm, &join(prefix, "_bn0"), visitor);
        }
        visitor(
            join(prefix, "_depthwise_conv"),
            &self.depthwise.weight,
            StateKind::Parameter,
        );
        visit_source_batch_norm(&self.bn1, &join(prefix, "_bn1"), visitor);
        if let Some(reduce) = &self.se_reduce {
            visitor(
                join(prefix, "_se_reduce"),
                &reduce.weight,
                StateKind::Parameter,
            );
            visitor(
                join(prefix, "_se_reduce_bias"),
                reduce.bias.as_ref().expect("configured SE bias"),
                StateKind::Parameter,
            );
        }
        if let Some(expand) = &self.se_expand {
            visitor(
                join(prefix, "_se_expand"),
                &expand.weight,
                StateKind::Parameter,
            );
            visitor(
                join(prefix, "_se_expand_bias"),
                expand.bias.as_ref().expect("configured SE bias"),
                StateKind::Parameter,
            );
        }
        visitor(
            join(prefix, "_project_conv"),
            &self.project.weight,
            StateKind::Parameter,
        );
        visit_source_batch_norm(&self.bn2, &join(prefix, "_bn2"), visitor);
    }
}

impl ModeModuleForward for MBConvBlock {
    fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<ModeForwardOutput<'a>> {
        let mut candidate = graph.clone();
        let output = self.lower(&mut candidate, input, mode)?;
        *graph = candidate;
        Ok(output)
    }
}

struct PreparedEfficientNet {
    config: EfficientNetConfig,
    stem: PreparedConv,
    blocks: Vec<PreparedMBConvBlock>,
    head: PreparedConv,
    classifier: Option<(TensorData, TensorData)>,
}

/// Checked-in tinygrad's static EfficientNet family.
///
/// Construction uses an explicit deterministic host seed. That is RustGrad's
/// graph-independent module contract, rather than tinygrad's ambient tensor
/// RNG. The seed is consumed by one shared initializer cursor only after the
/// entire model descriptor has been validated. Forward is RNG-free and any
/// training BatchNorm updates remain explicit ordered pending effects.
pub struct EfficientNet {
    pub blocks: Vec<MBConvBlock>,
    stem: Conv2d,
    bn0: BatchNorm2d,
    head: Conv2d,
    bn1: BatchNorm2d,
    classifier: Option<(Parameter, Parameter)>,
    config: EfficientNetConfig,
}

fn round_filters(filters: usize, multiplier: f64) -> Result<usize> {
    let scaled = filters as f64 * multiplier;
    if !scaled.is_finite() || scaled > usize::MAX as f64 {
        return Err(Error::ShapeOverflow(Shape::new([filters])));
    }
    let mut rounded = (((scaled + 4.0) as usize) / 8 * 8).max(8);
    if (rounded as f64) < 0.9 * scaled {
        rounded = rounded
            .checked_add(8)
            .ok_or_else(|| Error::ShapeOverflow(Shape::new([filters])))?;
    }
    Ok(rounded)
}

impl PreparedEfficientNet {
    fn new(config: EfficientNetConfig, seed: u64) -> Result<Self> {
        let index = usize::try_from(config.number.max(0)).map_err(|_| Error::InvalidRandom {
            reason: "EfficientNet number is outside the checked source family",
        })?;
        let &(width, depth) = GLOBAL_PARAMS.get(index).ok_or(Error::InvalidRandom {
            reason: "EfficientNet number is outside the checked source family",
        })?;
        let stem_channels = round_filters(32, width)?;
        let head_input = round_filters(320, width)?;
        let head_output = round_filters(1280, width)?;
        if config.has_fc_output {
            glorot_uniform_bound(&Shape::new([head_output, config.classes]))?;
        }
        let mut cursor = InitCursor::new(seed);
        let stem = PreparedConv::new(
            &mut cursor,
            config.input_channels,
            stem_channels,
            3,
            Conv2dOptions {
                groups: 1,
                stride: [2, 2],
                dilation: [1, 1],
                padding: [0, 1, 0, 1],
            },
            false,
        )?;
        let source_specs: &[BlockSpec] = match config.number {
            -1 => &SMALL_BLOCKS,
            -2 => &FAST_BLOCKS,
            _ => &DEFAULT_BLOCKS,
        };
        let mut blocks = Vec::new();
        for source in source_specs {
            let input_filters = round_filters(source.input_filters, width)?;
            let output_filters = round_filters(source.output_filters, width)?;
            let repeats = (source.repeats as f64 * depth).ceil() as usize;
            for repeat in 0..repeats {
                let spec = BlockSpec {
                    repeats: 1,
                    input_filters: if repeat == 0 {
                        input_filters
                    } else {
                        output_filters
                    },
                    output_filters,
                    stride: if repeat == 0 { source.stride } else { [1, 1] },
                    ..*source
                };
                blocks.push(PreparedMBConvBlock::new(
                    spec,
                    config.has_se,
                    config.track_running_stats,
                    &mut cursor,
                )?);
            }
        }
        let head = PreparedConv::new(&mut cursor, head_input, head_output, 1, POINTWISE, false)?;
        let classifier = config
            .has_fc_output
            .then(|| {
                Ok((
                    cursor.glorot_uniform(Shape::new([head_output, config.classes]))?,
                    TensorData::zeros([config.classes])?,
                ))
            })
            .transpose()?;
        Ok(Self {
            config,
            stem,
            blocks,
            head,
            classifier,
        })
    }

    fn publish(self) -> EfficientNet {
        let stem = self.stem.publish();
        let bn0 = batch_norm(stem.out_channels, self.config.track_running_stats);
        let blocks = self
            .blocks
            .into_iter()
            .map(PreparedMBConvBlock::publish)
            .collect();
        let head = self.head.publish();
        let bn1 = batch_norm(head.out_channels, self.config.track_running_stats);
        let classifier = self
            .classifier
            .map(|(weight, bias)| (Parameter::new(weight, true), Parameter::new(bias, true)));
        EfficientNet {
            blocks,
            stem,
            bn0,
            head,
            bn1,
            classifier,
            config: self.config,
        }
    }
}

impl EfficientNet {
    pub fn new_static(config: EfficientNetConfig, seed: u64) -> Result<Self> {
        Ok(PreparedEfficientNet::new(config, seed)?.publish())
    }

    pub const fn config(&self) -> EfficientNetConfig {
        self.config
    }

    fn lower<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<ModeForwardOutput<'a>> {
        let input_shape = graph.shape(input)?.clone();
        if input_shape.rank() != 4 || input_shape.dims()[1] != self.config.input_channels {
            return Err(Error::InvalidConv2d {
                input: input_shape,
                weight: self.stem.weight.shape()?,
                reason: "EfficientNet input must be NCHW with configured channels",
            });
        }
        let mut pending = PendingModeEffects::empty();
        let mut value = self.stem.forward(graph, input)?;
        let normalized = self.bn0.forward_mode(graph, value, mode)?;
        value = graph.silu(normalized.output)?;
        pending.append(normalized.pending);
        for block in &self.blocks {
            let block_output = block.lower(graph, value, mode)?;
            value = block_output.output;
            pending.append(block_output.pending);
        }
        value = self.head.forward(graph, value)?;
        let normalized = self.bn1.forward_mode(graph, value, mode)?;
        value = graph.silu(normalized.output)?;
        pending.append(normalized.pending);
        let shape = graph.shape(value)?.clone();
        value = graph.avg_pool2d(
            value,
            Pool2dOptions {
                kernel: [shape.dims()[2], shape.dims()[3]],
                stride: [shape.dims()[2], shape.dims()[3]],
                dilation: [1, 1],
                padding: [0; 4],
                ceil_mode: false,
                count_include_pad: true,
            },
        )?;
        value = graph.reshape(value, Shape::new([shape.dims()[0], shape.dims()[1]]))?;
        if let Some((weight, bias)) = &self.classifier {
            let weight = weight.bind(graph)?;
            let bias = bias.bind(graph)?;
            value = graph.linear(value, weight, Some(bias), None)?;
        }
        Ok(ModeForwardOutput {
            output: value,
            pending,
        })
    }
}

impl Module for EfficientNet {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        visitor(
            join(prefix, "_conv_stem"),
            &self.stem.weight,
            StateKind::Parameter,
        );
        visit_source_batch_norm(&self.bn0, &join(prefix, "_bn0"), visitor);
        for (index, block) in self.blocks.iter().enumerate() {
            block.visit(&join(prefix, &format!("_blocks.{index}")), visitor);
        }
        visitor(
            join(prefix, "_conv_head"),
            &self.head.weight,
            StateKind::Parameter,
        );
        visit_source_batch_norm(&self.bn1, &join(prefix, "_bn1"), visitor);
        if let Some((weight, bias)) = &self.classifier {
            visitor(join(prefix, "_fc"), weight, StateKind::Parameter);
            visitor(join(prefix, "_fc_bias"), bias, StateKind::Parameter);
        }
    }
}

impl ModeModuleForward for EfficientNet {
    fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<ModeForwardOutput<'a>> {
        let mut candidate = graph.clone();
        let output = self.lower(&mut candidate, input, mode)?;
        *graph = candidate;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Backend, CapturedReplayExecutor, CapturedReplayOptions, CpuBackend, DType, Op,
        TrainingContext,
    };
    use std::collections::BTreeMap;

    fn fast_config() -> EfficientNetConfig {
        EfficientNetConfig {
            number: -2,
            classes: 3,
            input_channels: 1,
            ..EfficientNetConfig::default()
        }
    }

    fn zero_parameters(module: &impl Module) {
        let mut parameters = Vec::new();
        module.visit("", &mut |_, parameter, kind| {
            if matches!(kind, StateKind::Parameter) {
                parameters.push(parameter.clone());
            }
        });
        for parameter in parameters {
            parameter
                .replace(TensorData::zeros(parameter.shape().unwrap()).unwrap())
                .unwrap();
        }
    }

    #[test]
    fn efficientnet_preserves_source_geometry_state_and_source_linear() {
        let block_config = crate::MBConvBlockConfig {
            kernel_size: 3,
            strides: [1, 1],
            expand_ratio: 1,
            input_filters: 2,
            output_filters: 2,
            se_ratio: 0.25,
            has_se: true,
            track_running_stats: true,
        };
        let public_block = crate::MBConvBlock::new_static(block_config, 5).unwrap();
        assert_eq!(public_block.config(), block_config);
        let public_names = public_block
            .state_dict()
            .unwrap()
            .tensors()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert!(public_names.iter().any(|name| name == "_depthwise_conv"));
        assert!(public_names.iter().any(|name| name == "_se_reduce_bias"));
        assert!(
            crate::MBConvBlock::new_static(
                crate::MBConvBlockConfig {
                    se_ratio: f64::NAN,
                    ..block_config
                },
                5
            )
            .is_err()
        );
        assert_eq!(
            crate::MBConvBlock::new_static(block_config, 5)
                .unwrap()
                .state_dict()
                .unwrap(),
            public_block.state_dict().unwrap()
        );
        assert!(crate::EfficientNet::new_static(fast_config(), 5).is_ok());

        let b0 = EfficientNet::new_static(EfficientNetConfig::default(), 7).unwrap();
        assert_eq!(b0.blocks.len(), 16);
        zero_parameters(&b0.blocks[2]);
        let mut residual_graph = Graph::new();
        let residual_input = residual_graph.input("residual", [1, 24, 2, 2]);
        let residual_output = b0.blocks[2]
            .forward_mode(&mut residual_graph, residual_input, Mode::Eval)
            .unwrap()
            .output;
        let residual_data = TensorData::ones([1, 24, 2, 2]).unwrap();
        let mut residual_bindings = b0.blocks[2].input_bindings(&residual_graph).unwrap();
        residual_bindings.insert("residual".into(), residual_data.clone());
        assert_eq!(
            CpuBackend
                .execute(&residual_graph, residual_output, &residual_bindings)
                .unwrap(),
            residual_data
        );
        let fast = EfficientNet::new_static(fast_config(), 7).unwrap();
        assert_eq!(fast.blocks.len(), 1);
        assert_eq!(fast.blocks[0].depthwise.options.stride, [8, 8]);
        assert_eq!(fast.blocks[0].depthwise.options.padding, [4; 4]);
        assert_eq!(fast.blocks[0].depthwise.options.groups, 32);
        assert_eq!(
            fast.classifier.as_ref().unwrap().0.shape().unwrap(),
            Shape::new([1280, 3])
        );

        let mut names = Vec::new();
        fast.visit("", &mut |name, _, _| names.push(name));
        assert_eq!(names.first().unwrap(), "_conv_stem");
        assert!(names.iter().any(|name| name == "_blocks.0._depthwise_conv"));
        assert!(
            names
                .iter()
                .any(|name| name == "_blocks.0._bn1.num_batches_tracked")
        );
        assert!(names.iter().any(|name| name == "_blocks.0._se_reduce"));
        assert!(names.iter().any(|name| name == "_blocks.0._se_expand_bias"));
        assert_eq!(names[names.len() - 2..], ["_fc", "_fc_bias"]);

        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [1, 1, 8, 8], DType::I16);
        let output = fast
            .forward_mode(&mut graph, input, Mode::Eval)
            .unwrap()
            .output;
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([1, 3]));
        assert!(
            (0..graph.node_count())
                .all(|index| !matches!(graph.op(NodeId(index)).unwrap(), Op::Matmul { .. }))
        );
    }

    #[test]
    fn efficientnet_fast_path_runs_cpu_capture_vjp_and_empty_batch() {
        let model = EfficientNet::new_static(fast_config(), 11).unwrap();
        zero_parameters(&model);
        model
            .classifier
            .as_ref()
            .unwrap()
            .1
            .replace(TensorData::new([3], vec![1.0, -2.0, 3.0]).unwrap())
            .unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype_requires_grad("input", [1, 1, 8, 8], DType::F32, true);
        let output = model.forward_mode(&mut graph, input, Mode::Eval).unwrap();
        assert!(output.pending.is_empty());
        let loss = graph.sum_all(output.output).unwrap();
        let gradient = graph.gradient_default(loss, &[input]).unwrap()[0];
        let mut bindings = model.input_bindings(&graph).unwrap();
        bindings.insert("input".into(), TensorData::ones([1, 1, 8, 8]).unwrap());
        assert_eq!(
            CpuBackend
                .execute(&graph, output.output, &bindings)
                .unwrap(),
            TensorData::new([1, 3], vec![1.0, -2.0, 3.0]).unwrap()
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &bindings).unwrap(),
            TensorData::zeros([1, 1, 8, 8]).unwrap()
        );
        let schedule = crate::schedule(&graph, output.output).unwrap();
        let capture =
            crate::CapturedSchedule::capture(&graph, &schedule, &[output.output]).unwrap();
        let replay = CapturedReplayExecutor::default()
            .replay(
                &capture,
                &bindings.into_iter().collect::<BTreeMap<_, _>>(),
                CapturedReplayOptions::default(),
            )
            .unwrap();
        assert_eq!(
            replay.outputs,
            vec![TensorData::new([1, 3], vec![1.0, -2.0, 3.0]).unwrap()]
        );

        let mut empty = Graph::new();
        let input = empty.input("input", [0, 1, 8, 8]);
        let output = model
            .forward_mode(&mut empty, input, Mode::Eval)
            .unwrap()
            .output;
        assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 3]));
        let mut bindings = model.input_bindings(&empty).unwrap();
        bindings.insert("input".into(), TensorData::zeros([0, 1, 8, 8]).unwrap());
        assert_eq!(
            CpuBackend.execute(&empty, output, &bindings).unwrap(),
            TensorData::new([0, 3], Vec::<f32>::new()).unwrap()
        );
    }

    #[test]
    fn efficientnet_mode_effects_and_failures_are_ordered_and_atomic() {
        let model = EfficientNet::new_static(fast_config(), 13).unwrap();
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 1, 8, 8]);
        let training = model
            .forward_mode(&mut graph, input, Mode::Training)
            .unwrap();
        assert_eq!(training.pending.batchnorm_stat_nodes().len(), 4);
        let mut ambient_graph = Graph::new();
        let ambient_input = ambient_graph.input("ambient", [2, 1, 8, 8]);
        let _training = TrainingContext::training();
        let ambient = model
            .forward_ambient(&mut ambient_graph, ambient_input)
            .unwrap();
        assert_eq!(ambient.pending.batchnorm_stat_nodes().len(), 4);
        assert!((0..ambient_graph.node_count()).all(|index| {
            !matches!(ambient_graph.op(NodeId(index)).unwrap(), Op::Random { .. })
        }));

        let mut malformed = Graph::new();
        let wrong = malformed.input("wrong", [1, 2, 8, 8]);
        let before = malformed.node_count();
        assert!(
            model
                .forward_mode(&mut malformed, wrong, Mode::Eval)
                .is_err()
        );
        assert_eq!(malformed.node_count(), before);
        let tiny = malformed.input("tiny", [1, 1, 1, 1]);
        let before = malformed.node_count();
        assert!(
            model
                .forward_mode(&mut malformed, tiny, Mode::Eval)
                .is_err()
        );
        assert_eq!(malformed.node_count(), before);

        assert!(
            EfficientNet::new_static(
                EfficientNetConfig {
                    number: 10,
                    ..fast_config()
                },
                1
            )
            .is_err()
        );
        let zero_channel_model = EfficientNet::new_static(
            EfficientNetConfig {
                input_channels: 0,
                ..fast_config()
            },
            1,
        )
        .unwrap();
        let mut zero_channel_graph = Graph::new();
        let zero_channel_input = zero_channel_graph.input("zero", [1, 0, 8, 8]);
        let zero_channel_output = zero_channel_model
            .forward_mode(&mut zero_channel_graph, zero_channel_input, Mode::Eval)
            .unwrap()
            .output;
        assert_eq!(
            zero_channel_graph.shape(zero_channel_output).unwrap(),
            &Shape::new([1, 3])
        );
        let expected = EfficientNet::new_static(fast_config(), 17)
            .unwrap()
            .state_dict()
            .unwrap();
        assert!(
            EfficientNet::new_static(
                EfficientNetConfig {
                    classes: usize::MAX,
                    ..fast_config()
                },
                17
            )
            .is_err()
        );
        let retry = EfficientNet::new_static(fast_config(), 17)
            .unwrap()
            .state_dict()
            .unwrap();
        assert_eq!(retry, expected);
    }
}
