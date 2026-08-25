//! Convolution and transpose-convolution modules.

use super::{Module, ModuleForward, Parameter, StateKind, init::uniform, state::join};
use crate::{Error, Graph, NodeId, Result, Shape};

/// Normalized 1D convolution geometry. Padding is `(before, after)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv1dOptions {
    pub groups: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: (usize, usize),
}
impl Default for Conv1dOptions {
    fn default() -> Self {
        Self {
            groups: 1,
            stride: 1,
            dilation: 1,
            padding: (0, 0),
        }
    }
}

/// A graph-composed 2D convolution module with tinygrad-compatible OIHW
/// parameter layout and fan-in uniform initialization.
pub struct Conv2d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: [usize; 2],
    pub options: crate::Conv2dOptions,
}
impl Conv2d {
    /// Creates graph-independent host parameters for static module workflows.
    pub fn new_static(
        in_channels: usize,
        out_channels: usize,
        kernel_size: [usize; 2],
        options: crate::Conv2dOptions,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        Self::new_impl(in_channels, out_channels, kernel_size, options, bias, seed)
    }

    /// Legacy construction spelling retained for source compatibility.
    pub fn new(
        _graph: &mut Graph,
        in_channels: usize,
        out_channels: usize,
        kernel_size: [usize; 2],
        options: crate::Conv2dOptions,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        Self::new_static(in_channels, out_channels, kernel_size, options, bias, seed)
    }

    fn new_impl(
        in_channels: usize,
        out_channels: usize,
        kernel_size: [usize; 2],
        options: crate::Conv2dOptions,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        if in_channels == 0
            || out_channels == 0
            || kernel_size.contains(&0)
            || options.groups == 0
            || in_channels % options.groups != 0
            || out_channels % options.groups != 0
        {
            return Err(Error::InvalidConv2d {
                input: Shape::new([0; 4]),
                weight: Shape::new([0; 4]),
                reason: "invalid convolution module channel, group, or kernel geometry",
            });
        }
        let fan_in = (in_channels / options.groups)
            .checked_mul(kernel_size[0])
            .and_then(|x| x.checked_mul(kernel_size[1]))
            .ok_or_else(|| Error::ShapeOverflow(Shape::new([in_channels, out_channels])))?;
        let bound = 1.0 / (fan_in as f32).sqrt();
        Ok(Self {
            weight: Parameter::new(
                uniform(
                    Shape::new([
                        out_channels,
                        in_channels / options.groups,
                        kernel_size[0],
                        kernel_size[1],
                    ]),
                    -bound,
                    bound,
                    seed,
                )?,
                true,
            ),
            bias: bias.then(|| {
                Parameter::new(
                    uniform(
                        Shape::new([out_channels]),
                        -bound,
                        bound,
                        seed.wrapping_add(1),
                    )
                    .expect("validated shape"),
                    true,
                )
            }),
            in_channels,
            out_channels,
            kernel_size,
            options,
        })
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        if graph.shape(input)?.rank() != 4 || graph.shape(input)?.dims()[1] != self.in_channels {
            return Err(Error::InvalidConv2d {
                input: graph.shape(input)?.clone(),
                weight: self.weight.shape()?,
                reason: "Conv2d input must be NCHW with the configured channels",
            });
        }
        let weight = self.weight.bind(graph)?;
        let bias = self.bias.as_ref().map(|b| b.bind(graph)).transpose()?;
        graph.conv2d(input, weight, bias, self.options)
    }
}
impl Module for Conv2d {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        v(join(p, "weight"), &self.weight, StateKind::Parameter);
        if let Some(b) = &self.bias {
            v(join(p, "bias"), b, StateKind::Parameter);
        }
    }
}
impl ModuleForward for Conv2d {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

/// Tinygrad-layout IOHW transpose convolution module.
pub struct ConvTranspose2d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: [usize; 2],
    pub options: crate::ConvTranspose2dOptions,
}
impl ConvTranspose2d {
    /// Creates graph-independent host parameters for static module workflows.
    pub fn new_static(
        in_channels: usize,
        out_channels: usize,
        kernel_size: [usize; 2],
        options: crate::ConvTranspose2dOptions,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        Self::new_impl(in_channels, out_channels, kernel_size, options, bias, seed)
    }

    /// Legacy construction spelling retained for source compatibility.
    pub fn new(
        _graph: &mut Graph,
        in_channels: usize,
        out_channels: usize,
        kernel_size: [usize; 2],
        options: crate::ConvTranspose2dOptions,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        Self::new_static(in_channels, out_channels, kernel_size, options, bias, seed)
    }

    fn new_impl(
        in_channels: usize,
        out_channels: usize,
        kernel_size: [usize; 2],
        options: crate::ConvTranspose2dOptions,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        if in_channels == 0
            || out_channels == 0
            || kernel_size.contains(&0)
            || options.groups == 0
            || in_channels % options.groups != 0
            || out_channels % options.groups != 0
        {
            return Err(Error::InvalidConv2d {
                input: Shape::new([0; 4]),
                weight: Shape::new([0; 4]),
                reason: "invalid transpose convolution module geometry",
            });
        }
        let bound = 1.0
            / (in_channels
                .checked_mul(kernel_size[0])
                .and_then(|x| x.checked_mul(kernel_size[1]))
                .ok_or_else(|| Error::ShapeOverflow(Shape::new([in_channels, out_channels])))?
                as f32)
                .sqrt();
        Ok(Self {
            weight: Parameter::new(
                uniform(
                    Shape::new([
                        in_channels,
                        out_channels / options.groups,
                        kernel_size[0],
                        kernel_size[1],
                    ]),
                    -bound,
                    bound,
                    seed,
                )?,
                true,
            ),
            bias: bias.then(|| {
                Parameter::new(
                    uniform(
                        Shape::new([out_channels]),
                        -bound,
                        bound,
                        seed.wrapping_add(1),
                    )
                    .expect("validated shape"),
                    true,
                )
            }),
            in_channels,
            out_channels,
            kernel_size,
            options,
        })
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        if graph.shape(input)?.rank() != 4 || graph.shape(input)?.dims()[1] != self.in_channels {
            return Err(Error::InvalidConv2d {
                input: graph.shape(input)?.clone(),
                weight: self.weight.shape()?,
                reason: "ConvTranspose2d input must be NCHW with the configured channels",
            });
        }
        let weight = self.weight.bind(graph)?;
        let bias = self.bias.as_ref().map(|x| x.bind(graph)).transpose()?;
        graph.conv_transpose2d(input, weight, bias, self.options)
    }
}
impl Module for ConvTranspose2d {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        v(join(p, "weight"), &self.weight, StateKind::Parameter);
        if let Some(x) = &self.bias {
            v(join(p, "bias"), x, StateKind::Parameter)
        }
    }
}
impl ModuleForward for ConvTranspose2d {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

/// Tinygrad-layout IOK transpose convolution lowered through the 2D core.
pub struct ConvTranspose1d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub options: crate::ConvTranspose1dOptions,
}
impl ConvTranspose1d {
    /// Creates graph-independent host parameters for static module workflows.
    pub fn new_static(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        options: crate::ConvTranspose1dOptions,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        Self::new_impl(in_channels, out_channels, kernel_size, options, bias, seed)
    }

    /// Legacy construction spelling retained for source compatibility.
    pub fn new(
        _graph: &mut Graph,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        options: crate::ConvTranspose1dOptions,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        Self::new_static(in_channels, out_channels, kernel_size, options, bias, seed)
    }

    fn new_impl(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        options: crate::ConvTranspose1dOptions,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        if in_channels == 0
            || out_channels == 0
            || kernel_size == 0
            || options.groups == 0
            || options.stride == 0
            || options.dilation == 0
            || in_channels % options.groups != 0
            || out_channels % options.groups != 0
            || options.output_padding >= options.stride
        {
            return Err(Error::InvalidConv2d {
                input: Shape::new([0; 4]),
                weight: Shape::new([0; 4]),
                reason: "invalid ConvTranspose1d module geometry",
            });
        }
        let bound = 1.0
            / (in_channels
                .checked_mul(kernel_size)
                .ok_or_else(|| Error::ShapeOverflow(Shape::new([in_channels, out_channels])))?
                as f32)
                .sqrt();
        Ok(Self {
            weight: Parameter::new(
                uniform(
                    Shape::new([in_channels, out_channels / options.groups, kernel_size]),
                    -bound,
                    bound,
                    seed,
                )?,
                true,
            ),
            bias: bias.then(|| {
                Parameter::new(
                    uniform(
                        Shape::new([out_channels]),
                        -bound,
                        bound,
                        seed.wrapping_add(1),
                    )
                    .expect("validated shape"),
                    true,
                )
            }),
            in_channels,
            out_channels,
            kernel_size,
            options,
        })
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        if graph.shape(input)?.rank() != 3 || graph.shape(input)?.dims()[1] != self.in_channels {
            return Err(Error::InvalidConv2d {
                input: graph.shape(input)?.clone(),
                weight: self.weight.shape()?,
                reason: "ConvTranspose1d input must be NCL with the configured channels",
            });
        }
        let weight = self.weight.bind(graph)?;
        let bias = self.bias.as_ref().map(|x| x.bind(graph)).transpose()?;
        graph.conv_transpose1d(input, weight, bias, self.options)
    }
}
impl Module for ConvTranspose1d {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        v(join(p, "weight"), &self.weight, StateKind::Parameter);
        if let Some(x) = &self.bias {
            v(join(p, "bias"), x, StateKind::Parameter)
        }
    }
}
impl ModuleForward for ConvTranspose1d {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

/// A 1D convolution lowered through the existing typed 2D convolution node.
pub struct Conv1d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub options: Conv1dOptions,
}
impl Conv1d {
    /// Creates graph-independent host parameters for static module workflows.
    pub fn new_static(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        options: Conv1dOptions,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        Self::new_impl(in_channels, out_channels, kernel_size, options, bias, seed)
    }

    /// Legacy construction spelling retained for source compatibility.
    pub fn new(
        _graph: &mut Graph,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        options: Conv1dOptions,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        Self::new_static(in_channels, out_channels, kernel_size, options, bias, seed)
    }

    fn new_impl(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        options: Conv1dOptions,
        bias: bool,
        seed: u64,
    ) -> Result<Self> {
        if in_channels == 0
            || out_channels == 0
            || kernel_size == 0
            || options.groups == 0
            || options.stride == 0
            || options.dilation == 0
            || in_channels % options.groups != 0
            || out_channels % options.groups != 0
        {
            return Err(Error::InvalidConv2d {
                input: Shape::new([0; 4]),
                weight: Shape::new([0; 4]),
                reason: "invalid Conv1d module geometry",
            });
        }
        let fan_in = (in_channels / options.groups)
            .checked_mul(kernel_size)
            .ok_or_else(|| Error::ShapeOverflow(Shape::new([in_channels, out_channels])))?;
        let bound = 1.0 / (fan_in as f32).sqrt();
        Ok(Self {
            weight: Parameter::new(
                uniform(
                    Shape::new([out_channels, in_channels / options.groups, kernel_size]),
                    -bound,
                    bound,
                    seed,
                )?,
                true,
            ),
            bias: bias.then(|| {
                Parameter::new(
                    uniform(
                        Shape::new([out_channels]),
                        -bound,
                        bound,
                        seed.wrapping_add(1),
                    )
                    .expect("validated shape"),
                    true,
                )
            }),
            in_channels,
            out_channels,
            kernel_size,
            options,
        })
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        let shape = graph.shape(input)?.clone();
        if shape.rank() != 3 || shape.dims()[1] != self.in_channels {
            return Err(Error::InvalidConv2d {
                input: shape,
                weight: self.weight.shape()?,
                reason: "Conv1d input must be NCL with the configured channels",
            });
        }
        let x = graph.reshape(
            input,
            Shape::new([shape.dims()[0], self.in_channels, 1, shape.dims()[2]]),
        )?;
        let weight = self.weight.bind(graph)?;
        let weight = graph.reshape(
            weight,
            Shape::new([
                self.out_channels,
                self.in_channels / self.options.groups,
                1,
                self.kernel_size,
            ]),
        )?;
        let bias = self.bias.as_ref().map(|b| b.bind(graph)).transpose()?;
        let y = graph.conv2d(
            x,
            weight,
            bias,
            crate::Conv2dOptions {
                groups: self.options.groups,
                stride: [1, self.options.stride],
                dilation: [1, self.options.dilation],
                padding: [0, 0, self.options.padding.0, self.options.padding.1],
            },
        )?;
        let out = graph.shape(y)?.clone();
        graph.reshape(y, Shape::new([out.dims()[0], out.dims()[1], out.dims()[3]]))
    }
}
impl Module for Conv1d {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        v(join(p, "weight"), &self.weight, StateKind::Parameter);
        if let Some(b) = &self.bias {
            v(join(p, "bias"), b, StateKind::Parameter);
        }
    }
}
impl ModuleForward for Conv1d {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}
