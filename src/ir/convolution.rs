use super::{Graph, NodeId, ReductionDType, source_lub};
use crate::{DType, Error, Result, Scalar, Shape};
use std::fmt;
use std::num::NonZeroUsize;

/// Validated trailing-spatial convolution window geometry.
///
/// Padding is stored in spatial-axis order as signed `(before, after)` pairs.
/// Negative values crop before the window is formed. The type owns one exact
/// kernel/stride/dilation/padding tuple; rank-specific adapters do not create a
/// second convolution taxonomy.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SpatialWindow {
    kernel: Vec<usize>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<(i64, i64)>,
}

/// Descriptor-only errors produced while normalizing convolution geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpatialWindowError {
    RankMismatch,
    NonPositive,
}

impl fmt::Display for SpatialWindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reason = match self {
            Self::RankMismatch => "kernel, stride, dilation, and padding ranks must match",
            Self::NonPositive => "kernel, stride, and dilation entries must be positive",
        };
        f.write_str(reason)
    }
}

impl std::error::Error for SpatialWindowError {}

impl SpatialWindow {
    pub fn new(
        kernel: impl Into<Vec<usize>>,
        stride: impl Into<Vec<usize>>,
        dilation: impl Into<Vec<usize>>,
        padding: impl Into<Vec<(i64, i64)>>,
    ) -> std::result::Result<Self, SpatialWindowError> {
        let kernel = kernel.into();
        let stride = stride.into();
        let dilation = dilation.into();
        let padding = padding.into();
        if stride.len() != kernel.len()
            || dilation.len() != kernel.len()
            || padding.len() != kernel.len()
        {
            return Err(SpatialWindowError::RankMismatch);
        }
        if kernel
            .iter()
            .chain(&stride)
            .chain(&dilation)
            .any(|&value| value == 0)
        {
            return Err(SpatialWindowError::NonPositive);
        }
        Ok(Self {
            kernel,
            stride,
            dilation,
            padding,
        })
    }

    pub fn rank(&self) -> usize {
        self.kernel.len()
    }

    pub fn kernel(&self) -> &[usize] {
        &self.kernel
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn dilation(&self) -> &[usize] {
        &self.dilation
    }

    pub fn padding(&self) -> &[(i64, i64)] {
        &self.padding
    }
}

/// Normalized convolution contract shared by every spatial rank.
///
/// The optional dtype is the caller's only reduction policy input. The graph
/// planner derives the exact product LUB and resulting [`ReductionDType`] from
/// authoritative operand nodes, so callers cannot forge redundant dtype state.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ConvolutionSpec {
    window: SpatialWindow,
    groups: NonZeroUsize,
    dtype: Option<DType>,
}

impl ConvolutionSpec {
    pub fn new(window: SpatialWindow, groups: NonZeroUsize, dtype: Option<DType>) -> Self {
        Self {
            window,
            groups,
            dtype,
        }
    }

    pub fn window(&self) -> &SpatialWindow {
        &self.window
    }

    pub fn groups(&self) -> NonZeroUsize {
        self.groups
    }

    pub fn requested_dtype(&self) -> Option<DType> {
        self.dtype
    }
}

/// Rank-generic source contract for transposed convolution.
///
/// The ordinary [`SpatialWindow`] owns the kernel, positive stride/dilation,
/// and signed asymmetric source padding. `output_padding` is separately signed
/// because tinygrad folds it into the trailing transformed convolution pad; it
/// is not restricted to be smaller than stride. Groups are nonzero by type.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TransposedConvolutionSpec {
    window: SpatialWindow,
    output_padding: Vec<i64>,
    groups: NonZeroUsize,
}

impl TransposedConvolutionSpec {
    pub fn new(
        window: SpatialWindow,
        output_padding: impl Into<Vec<i64>>,
        groups: NonZeroUsize,
    ) -> std::result::Result<Self, SpatialWindowError> {
        let output_padding = output_padding.into();
        if output_padding.len() != window.rank() {
            return Err(SpatialWindowError::RankMismatch);
        }
        Ok(Self {
            window,
            output_padding,
            groups,
        })
    }

    pub fn window(&self) -> &SpatialWindow {
        &self.window
    }

    pub fn output_padding(&self) -> &[i64] {
        &self.output_padding
    }

    pub fn groups(&self) -> NonZeroUsize {
        self.groups
    }
}

#[derive(Clone, Debug)]
struct ConvolutionPlan {
    spec: ConvolutionSpec,
    reduction: ReductionDType,
    output_shape: Shape,
    output_dtype: DType,
}

#[derive(Clone, Debug)]
struct TransposedConvolutionPlan {
    spec: TransposedConvolutionSpec,
    transformed_padding: Vec<(i64, i64)>,
    output_shape: Shape,
    output_dtype: DType,
}

fn invalid(input: &Shape, weight: &Shape, reason: &'static str) -> Error {
    Error::InvalidConvolution {
        input: input.clone(),
        weight: weight.clone(),
        reason,
    }
}

fn checked_bytes(shape: &Shape, dtype: DType) -> Result<()> {
    shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    Ok(())
}

fn signed_padded_extent(
    extent: usize,
    padding: (i64, i64),
    input: &Shape,
    weight: &Shape,
) -> Result<usize> {
    let (before, after) = padding;
    let extent = extent as i128;
    let start = (-(before as i128)).max(0);
    let end = (extent + after as i128).min(extent);
    if end < 0 || start > end {
        return Err(invalid(
            input,
            weight,
            "signed padding crops beyond spatial axis",
        ));
    }
    let retained = usize::try_from(end - start).map_err(|_| Error::ShapeOverflow(input.clone()))?;
    let before = usize::try_from(before.max(0)).map_err(|_| Error::ShapeOverflow(input.clone()))?;
    let after = usize::try_from(after.max(0)).map_err(|_| Error::ShapeOverflow(input.clone()))?;
    retained
        .checked_add(before)
        .and_then(|value| value.checked_add(after))
        .ok_or_else(|| Error::ShapeOverflow(input.clone()))
}

fn output_spatial(input: &Shape, weight: &Shape, window: &SpatialWindow) -> Result<Vec<usize>> {
    let spatial = &input.dims()[input.rank() - window.rank()..];
    spatial
        .iter()
        .enumerate()
        .map(|(axis, &input_extent)| {
            let padded = signed_padded_extent(input_extent, window.padding[axis], input, weight)?;
            let kernel_extent = window.dilation[axis]
                .checked_mul(window.kernel[axis] - 1)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| Error::ShapeOverflow(weight.clone()))?;
            if padded < kernel_extent {
                return Err(invalid(input, weight, "kernel exceeds padded input"));
            }
            Ok((padded - kernel_extent) / window.stride[axis] + 1)
        })
        .collect()
}

fn convolution_plan(
    graph: &Graph,
    input: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
    spec: ConvolutionSpec,
) -> Result<ConvolutionPlan> {
    let input_node = graph.node(input)?;
    let weight_node = graph.node(weight)?;
    let input_shape = input_node.shape.clone();
    let weight_shape = weight_node.shape.clone();
    let spatial_rank = spec.window.rank();
    if input_shape.rank() != spatial_rank + 2 || weight_shape.rank() != input_shape.rank() {
        return Err(invalid(
            &input_shape,
            &weight_shape,
            "input and weight ranks must equal spatial rank plus two",
        ));
    }
    if &weight_shape.dims()[2..] != spec.window.kernel() {
        return Err(invalid(
            &input_shape,
            &weight_shape,
            "weight kernel does not match spatial window",
        ));
    }
    let groups = spec.groups.get();
    let output_channels = weight_shape.dims()[0];
    let input_channels_per_group = weight_shape.dims()[1];
    if output_channels % groups != 0
        || input_channels_per_group
            .checked_mul(groups)
            .ok_or_else(|| Error::ShapeOverflow(weight_shape.clone()))?
            != input_shape.dims()[1]
    {
        return Err(invalid(
            &input_shape,
            &weight_shape,
            "channel/group geometry",
        ));
    }
    if let Some(bias) = bias {
        let bias_node = graph.node(bias)?;
        if bias_node.shape != Shape::from([output_channels]) {
            return Err(invalid(
                &input_shape,
                &weight_shape,
                "bias must be [output_channels]",
            ));
        }
    }
    let spatial = output_spatial(&input_shape, &weight_shape, &spec.window)?;
    let mut output_dims = vec![input_shape.dims()[0], output_channels];
    output_dims.extend(spatial);
    let output_shape = Shape::new(output_dims);
    let product_dtype = source_lub(input_node.dtype, weight_node.dtype);
    let reduction = spec
        .dtype
        .map(|dtype| ReductionDType::new(dtype, dtype))
        .unwrap_or_else(|| ReductionDType::sum_default(product_dtype));
    let mut output_dtype = reduction.output;
    if let Some(bias) = bias {
        output_dtype = source_lub(output_dtype, graph.dtype(bias)?);
    }
    checked_bytes(&input_shape, input_node.dtype)?;
    checked_bytes(&weight_shape, weight_node.dtype)?;
    checked_bytes(&input_shape, product_dtype)?;
    checked_bytes(&weight_shape, product_dtype)?;
    checked_bytes(&output_shape, output_dtype)?;
    Ok(ConvolutionPlan {
        spec,
        reduction,
        output_shape,
        output_dtype,
    })
}

/// Creates tinygrad's compact repeated/shrink/reshape/permute spatial windows.
fn lower_spatial_window(
    graph: &mut Graph,
    input: NodeId,
    weight_shape: &Shape,
    window: &SpatialWindow,
) -> Result<NodeId> {
    let source = graph.shape(input)?.clone();
    let prefix = source.rank() - window.rank();
    let mut padding = vec![(0, 0); prefix];
    padding.extend_from_slice(window.padding());
    let padded = graph.pad_signed(input, padding, Scalar::F(0.0))?;
    let padded_shape = graph.shape(padded)?.clone();
    let input_spatial = &padded_shape.dims()[prefix..];

    let mut output = Vec::with_capacity(window.rank());
    let mut expanded_extents = Vec::with_capacity(window.rank());
    let mut repeats = vec![1isize; padded_shape.rank()];
    for (axis, &extent) in input_spatial.iter().enumerate() {
        let dilation_span = window.dilation[axis]
            .checked_mul(window.kernel[axis] - 1)
            .ok_or_else(|| Error::ShapeOverflow(padded_shape.clone()))?;
        if extent <= dilation_span {
            return Err(Error::InvalidConvolution {
                input: source.clone(),
                weight: weight_shape.clone(),
                reason: "kernel exceeds padded input",
            });
        }
        let count = (extent - dilation_span).div_ceil(window.stride[axis]);
        let scaled = count
            .checked_mul(window.stride[axis])
            .ok_or_else(|| Error::ShapeOverflow(padded_shape.clone()))?;
        let factor = scaled
            .saturating_sub(window.dilation[axis])
            .div_ceil(extent)
            .max(1);
        let expanded = extent
            .checked_mul(factor)
            .and_then(|value| value.checked_add(window.dilation[axis]))
            .ok_or_else(|| Error::ShapeOverflow(padded_shape.clone()))?;
        let repeated_target = window.kernel[axis]
            .checked_mul(expanded)
            .ok_or_else(|| Error::ShapeOverflow(padded_shape.clone()))?;
        repeats[prefix + axis] = isize::try_from(repeated_target.div_ceil(extent))
            .map_err(|_| Error::ShapeOverflow(padded_shape.clone()))?;
        output.push(count);
        expanded_extents.push(expanded);
    }

    // Keep the exact checked-in tinygrad `_pool` structure. Projected-index
    // lowering now authenticates the periodic repeat/reshape chain directly,
    // so convolution windows need no eager Concat storage owners between the
    // padded source and their reduction consumer.
    let mut node = graph.repeat(padded, &repeats)?;
    let mut bounds = padded_shape.dims()[..prefix]
        .iter()
        .map(|&extent| (0, extent))
        .collect::<Vec<_>>();
    bounds.extend(
        window
            .kernel
            .iter()
            .zip(&expanded_extents)
            .map(|(&kernel, &expanded)| (0, kernel * expanded)),
    );
    node = graph.shrink(node, bounds)?;

    let mut shape = padded_shape.dims()[..prefix].to_vec();
    for (&kernel, &expanded) in window.kernel.iter().zip(&expanded_extents) {
        shape.extend([kernel, expanded]);
    }
    node = graph.reshape(node, Shape::new(shape))?;

    let mut bounds = padded_shape.dims()[..prefix]
        .iter()
        .map(|&extent| (0, extent))
        .collect::<Vec<_>>();
    for (axis, &kernel) in window.kernel.iter().enumerate() {
        bounds.extend([(0, kernel), (0, output[axis] * window.stride[axis])]);
    }
    node = graph.shrink(node, bounds)?;
    let mut shape = padded_shape.dims()[..prefix].to_vec();
    for (axis, &kernel) in window.kernel.iter().enumerate() {
        shape.extend([kernel, output[axis], window.stride[axis]]);
    }
    node = graph.reshape(node, Shape::new(shape))?;

    let mut bounds = padded_shape.dims()[..prefix]
        .iter()
        .map(|&extent| (0, extent))
        .collect::<Vec<_>>();
    for (axis, &kernel) in window.kernel.iter().enumerate() {
        bounds.extend([(0, kernel), (0, output[axis]), (0, 1)]);
    }
    node = graph.shrink(node, bounds)?;
    let mut shape = padded_shape.dims()[..prefix].to_vec();
    for (axis, &kernel) in window.kernel.iter().enumerate() {
        shape.extend([kernel, output[axis]]);
    }
    node = graph.reshape(node, Shape::new(shape))?;

    let mut permutation = (0..prefix).collect::<Vec<_>>();
    permutation.extend((0..window.rank()).map(|axis| prefix + axis * 2 + 1));
    permutation.extend((0..window.rank()).map(|axis| prefix + axis * 2));
    graph.permute(node, permutation)
}

fn lower_convolution(
    graph: &mut Graph,
    input: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
    plan: &ConvolutionPlan,
) -> Result<NodeId> {
    let input_shape = graph.shape(input)?.clone();
    let weight_shape = graph.shape(weight)?.clone();
    let spatial_rank = plan.spec.window.rank();
    let groups = plan.spec.groups.get();
    let batch = input_shape.dims()[0];
    let input_channels = weight_shape.dims()[1];
    let output_channels = weight_shape.dims()[0];
    let output_channels_per_group = output_channels / groups;
    let output_spatial = &plan.output_shape.dims()[2..];
    let kernel = plan.spec.window.kernel();

    let windows = lower_spatial_window(graph, input, &weight_shape, &plan.spec.window)?;
    let mut window_shape = vec![batch, groups, input_channels, 1];
    window_shape.extend_from_slice(output_spatial);
    window_shape.extend_from_slice(kernel);
    let windows = graph.reshape(windows, Shape::new(window_shape.clone()))?;
    window_shape[3] = output_channels_per_group;
    let windows = graph.expand(windows, Shape::new(window_shape))?;
    let mut order = vec![0, 1, 3];
    order.extend((0..spatial_rank).map(|axis| 4 + axis));
    order.push(2);
    order.extend((0..spatial_rank).map(|axis| 4 + spatial_rank + axis));
    let windows = graph.permute(windows, order)?;

    let mut weight_view = vec![1, groups, output_channels_per_group];
    weight_view.extend(std::iter::repeat_n(1, spatial_rank));
    weight_view.push(input_channels);
    weight_view.extend_from_slice(kernel);
    let weight = graph.reshape(weight, Shape::new(weight_view))?;
    let product = graph.mul(windows, weight)?;
    let axes = (1..=spatial_rank + 1)
        .map(|axis| -(axis as isize))
        .collect::<Vec<_>>();
    let reduced = graph.reduce_with_dtypes(
        product,
        crate::ReduceKind::Sum,
        Some(axes),
        true,
        plan.reduction,
    )?;
    let output = graph.reshape(reduced, plan.output_shape.clone())?;
    if let Some(bias) = bias {
        let mut bias_shape = vec![1, output_channels];
        bias_shape.extend(std::iter::repeat_n(1, spatial_rank));
        let bias = graph.reshape(bias, Shape::new(bias_shape))?;
        graph.add(output, bias)
    } else {
        Ok(output)
    }
}

fn checked_i64(value: i128, shape: &Shape) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::ShapeOverflow(shape.clone()))
}

fn transposed_convolution_plan(
    graph: &Graph,
    input: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
    spec: TransposedConvolutionSpec,
) -> Result<TransposedConvolutionPlan> {
    let input_node = graph.node(input)?;
    let weight_node = graph.node(weight)?;
    let input_shape = input_node.shape.clone();
    let weight_shape = weight_node.shape.clone();
    let spatial_rank = spec.window.rank();
    if input_shape.rank() != spatial_rank + 2 || weight_shape.rank() != input_shape.rank() {
        return Err(invalid(
            &input_shape,
            &weight_shape,
            "input and weight ranks must equal spatial rank plus two",
        ));
    }
    if &weight_shape.dims()[2..] != spec.window.kernel() {
        return Err(invalid(
            &input_shape,
            &weight_shape,
            "weight kernel does not match spatial window",
        ));
    }
    let groups = spec.groups.get();
    let input_channels = input_shape.dims()[1];
    if input_channels != weight_shape.dims()[0] || input_channels % groups != 0 {
        return Err(invalid(
            &input_shape,
            &weight_shape,
            "channel/group geometry",
        ));
    }
    let output_channels = weight_shape.dims()[1]
        .checked_mul(groups)
        .ok_or_else(|| Error::ShapeOverflow(weight_shape.clone()))?;
    if let Some(bias) = bias
        && graph.node(bias)?.shape != Shape::from([output_channels])
    {
        return Err(invalid(
            &input_shape,
            &weight_shape,
            "bias must be [output_channels]",
        ));
    }

    let mut transformed_padding = Vec::with_capacity(spatial_rank);
    let mut output_dims = vec![input_shape.dims()[0], output_channels];
    for axis in 0..spatial_rank {
        let kernel_span = (spec.window.kernel[axis] as i128 - 1)
            .checked_mul(spec.window.dilation[axis] as i128)
            .ok_or_else(|| Error::ShapeOverflow(weight_shape.clone()))?;
        let (before, after) = spec.window.padding[axis];
        let transformed_before = kernel_span
            .checked_sub(before as i128)
            .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
        let transformed_after = kernel_span
            .checked_sub(after as i128)
            .and_then(|value| value.checked_add(spec.output_padding[axis] as i128))
            .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
        transformed_padding.push((
            checked_i64(transformed_before, &input_shape)?,
            checked_i64(transformed_after, &input_shape)?,
        ));

        let extent = input_shape.dims()[axis + 2];
        let upsampled = extent
            .checked_mul(spec.window.stride[axis])
            .and_then(|value| value.checked_sub(spec.window.stride[axis] - 1))
            .ok_or_else(|| Error::ShapeOverflow(input_shape.clone()))?;
        let padded = signed_padded_extent(
            upsampled,
            transformed_padding[axis],
            &input_shape,
            &weight_shape,
        )?;
        let kernel_extent = spec.window.dilation[axis]
            .checked_mul(spec.window.kernel[axis] - 1)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| Error::ShapeOverflow(weight_shape.clone()))?;
        if padded < kernel_extent {
            return Err(invalid(
                &input_shape,
                &weight_shape,
                "kernel exceeds transformed padded input",
            ));
        }
        output_dims.push(padded - kernel_extent + 1);
    }
    let output_shape = Shape::new(output_dims);
    let product_dtype = source_lub(input_node.dtype, weight_node.dtype);
    let mut output_dtype = ReductionDType::sum_default(product_dtype).output;
    if let Some(bias) = bias {
        output_dtype = source_lub(output_dtype, graph.dtype(bias)?);
    }
    checked_bytes(&input_shape, input_node.dtype)?;
    checked_bytes(&weight_shape, weight_node.dtype)?;
    checked_bytes(&output_shape, output_dtype)?;
    Ok(TransposedConvolutionPlan {
        spec,
        transformed_padding,
        output_shape,
        output_dtype,
    })
}

/// Inserts `stride - 1` source-typed zero lanes between adjacent spatial
/// samples, matching tinygrad's reshape/pad/reshape/shrink construction.
fn lower_transposed_stride(graph: &mut Graph, input: NodeId, stride: &[usize]) -> Result<NodeId> {
    if stride.iter().all(|&value| value == 1) {
        return Ok(input);
    }
    let source = graph.shape(input)?.clone();
    let mut interleaved = source.dims()[..2].to_vec();
    for &extent in &source.dims()[2..] {
        interleaved.extend([extent, 1]);
    }
    let reshaped = graph.reshape(input, Shape::new(interleaved))?;
    let mut padding = vec![(0, 0); 2];
    for &step in stride {
        let after = i64::try_from(step - 1).map_err(|_| Error::ShapeOverflow(source.clone()))?;
        padding.extend([(0, 0), (0, after)]);
    }
    let padded = graph.pad_signed(reshaped, padding, Scalar::F(0.0))?;
    let mut expanded = source.dims()[..2].to_vec();
    for (&extent, &step) in source.dims()[2..].iter().zip(stride) {
        expanded.push(
            extent
                .checked_mul(step)
                .ok_or_else(|| Error::ShapeOverflow(source.clone()))?,
        );
    }
    let expanded = graph.reshape(padded, Shape::new(expanded))?;
    let mut bounds = source.dims()[..2]
        .iter()
        .map(|&extent| (0, extent))
        .collect::<Vec<_>>();
    for (&extent, &step) in source.dims()[2..].iter().zip(stride) {
        let end = extent
            .checked_mul(step)
            .and_then(|value| value.checked_sub(step - 1))
            .ok_or_else(|| Error::ShapeOverflow(source.clone()))?;
        bounds.push((0, end));
    }
    graph.shrink(expanded, bounds)
}

fn lower_transposed_convolution(
    graph: &mut Graph,
    input: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
    plan: &TransposedConvolutionPlan,
) -> Result<NodeId> {
    let input_shape = graph.shape(input)?.clone();
    let weight_shape = graph.shape(weight)?.clone();
    let spatial_rank = plan.spec.window.rank();
    let groups = plan.spec.groups.get();
    let input_channels_per_group = input_shape.dims()[1] / groups;
    let output_channels_per_group = weight_shape.dims()[1];
    let output_channels = output_channels_per_group
        .checked_mul(groups)
        .ok_or_else(|| Error::ShapeOverflow(weight_shape.clone()))?;

    let mut grouped_weight = vec![groups, input_channels_per_group, output_channels_per_group];
    grouped_weight.extend_from_slice(plan.spec.window.kernel());
    let weight = graph.reshape(weight, Shape::new(grouped_weight))?;
    let mut order = vec![0, 2, 1];
    order.extend(3..spatial_rank + 3);
    let weight = graph.permute(weight, order)?;
    let flip_axes = (3..spatial_rank + 3)
        .map(|axis| isize::try_from(axis).map_err(|_| Error::ShapeOverflow(weight_shape.clone())))
        .collect::<Result<Vec<_>>>()?;
    let weight = graph.flip(weight, flip_axes)?;
    let mut transformed_weight = vec![output_channels, input_channels_per_group];
    transformed_weight.extend_from_slice(plan.spec.window.kernel());
    let weight = graph.reshape(weight, Shape::new(transformed_weight))?;

    let input = lower_transposed_stride(graph, input, plan.spec.window.stride())?;
    let window = SpatialWindow::new(
        plan.spec.window.kernel().to_vec(),
        vec![1; spatial_rank],
        plan.spec.window.dilation().to_vec(),
        plan.transformed_padding.clone(),
    )
    .map_err(|_| {
        invalid(
            &input_shape,
            &weight_shape,
            "invalid transformed spatial window",
        )
    })?;
    graph.convolution(
        input,
        weight,
        bias,
        ConvolutionSpec::new(window, plan.spec.groups, None),
    )
}

impl Graph {
    /// Rank-generic convolution lowered entirely through ordinary movement,
    /// multiplication, typed reduction, and optional bias addition nodes.
    pub fn convolution(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        spec: ConvolutionSpec,
    ) -> Result<NodeId> {
        let plan = convolution_plan(self, input, weight, bias, spec)?;
        let mut rehearsal = self.clone();
        let rehearsed = lower_convolution(&mut rehearsal, input, weight, bias, &plan)?;
        debug_assert_eq!(rehearsal.shape(rehearsed).ok(), Some(&plan.output_shape));
        debug_assert_eq!(rehearsal.dtype(rehearsed).ok(), Some(plan.output_dtype));
        let output = lower_convolution(self, input, weight, bias, &plan)?;
        debug_assert_eq!(self.shape(output).ok(), Some(&plan.output_shape));
        debug_assert_eq!(self.dtype(output).ok(), Some(plan.output_dtype));
        Ok(output)
    }

    /// Rank-generic transposed convolution lowered through ordinary movement,
    /// multiplication, typed reduction, and optional bias addition nodes.
    pub fn transposed_convolution(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        spec: TransposedConvolutionSpec,
    ) -> Result<NodeId> {
        let plan = transposed_convolution_plan(self, input, weight, bias, spec)?;
        let mut rehearsal = self.clone();
        let rehearsed = lower_transposed_convolution(&mut rehearsal, input, weight, bias, &plan)?;
        debug_assert_eq!(rehearsal.shape(rehearsed).ok(), Some(&plan.output_shape));
        debug_assert_eq!(rehearsal.dtype(rehearsed).ok(), Some(plan.output_dtype));
        let output = lower_transposed_convolution(self, input, weight, bias, &plan)?;
        debug_assert_eq!(self.shape(output).ok(), Some(&plan.output_shape));
        debug_assert_eq!(self.dtype(output).ok(), Some(plan.output_dtype));
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, Op, TensorData};
    use std::collections::{BTreeMap, HashMap};

    fn data(shape: impl Into<Shape>, values: &[f32]) -> TensorData {
        TensorData::new(shape, values.to_vec()).unwrap()
    }

    fn spec(
        kernel: &[usize],
        stride: &[usize],
        dilation: &[usize],
        padding: &[(i64, i64)],
        groups: usize,
        dtype: Option<DType>,
    ) -> ConvolutionSpec {
        ConvolutionSpec::new(
            SpatialWindow::new(kernel, stride, dilation, padding).unwrap(),
            NonZeroUsize::new(groups).unwrap(),
            dtype,
        )
    }

    fn transposed_spec(
        kernel: &[usize],
        stride: &[usize],
        dilation: &[usize],
        padding: &[(i64, i64)],
        output_padding: &[i64],
        groups: usize,
    ) -> TransposedConvolutionSpec {
        TransposedConvolutionSpec::new(
            SpatialWindow::new(kernel, stride, dilation, padding).unwrap(),
            output_padding,
            NonZeroUsize::new(groups).unwrap(),
        )
        .unwrap()
    }

    fn execute(
        graph: &Graph,
        output: NodeId,
        input: TensorData,
        weight: TensorData,
        bias: Option<TensorData>,
    ) -> TensorData {
        let mut values = HashMap::from([("input".into(), input), ("weight".into(), weight)]);
        if let Some(bias) = bias {
            values.insert("bias".into(), bias);
        }
        CpuBackend.execute(graph, output, &values).unwrap()
    }

    fn assert_sum_contract(graph: &Graph, output: NodeId, accumulator: DType, output_dtype: DType) {
        let sums = (0..graph.node_count())
            .filter_map(|index| {
                let node = NodeId::from_index(index);
                matches!(
                    graph.op(node).ok()?,
                    Op::Reduce {
                        kind: crate::ReduceKind::Sum,
                        ..
                    }
                )
                .then(|| graph.dtype(node).unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(sums, vec![accumulator]);
        assert_eq!(graph.dtype(output).unwrap(), output_dtype);
    }

    #[test]
    fn spatial_window_and_rank_generic_values_cover_zero_one_two_and_three_dimensions() {
        let zero_window = SpatialWindow::new([], [], [], []).unwrap();
        assert_eq!(zero_window.rank(), 0);
        assert_eq!(
            SpatialWindow::new([1], [1, 1], [1], [(0, 0)]),
            Err(SpatialWindowError::RankMismatch)
        );
        assert_eq!(
            SpatialWindow::new([1], [0], [1], [(0, 0)]),
            Err(SpatialWindowError::NonPositive)
        );

        let mut zero = Graph::new();
        let input = zero.input("input", [1, 2]);
        let weight = zero.input("weight", [2, 2]);
        let output = zero
            .convolution(
                input,
                weight,
                None,
                ConvolutionSpec::new(zero_window, NonZeroUsize::new(1).unwrap(), None),
            )
            .unwrap();
        assert_eq!(zero.shape(output).unwrap(), &Shape::from([1, 2]));
        assert_eq!(
            execute(
                &zero,
                output,
                data([1, 2], &[1., 2.]),
                data([2, 2], &[1., 1., 2., 3.]),
                None,
            ),
            data([1, 2], &[3., 8.])
        );
        let squared = zero.square(output).unwrap();
        let loss = zero.sum_all(squared).unwrap();
        let first = zero.grad(loss, input).unwrap();
        let second_loss = zero.sum_all(first).unwrap();
        let second = zero.grad(second_loss, input).unwrap();
        assert_eq!(zero.shape(second).unwrap(), &Shape::from([1, 2]));

        let mut one = Graph::new();
        let input = one.input("input", [1, 1, 4]);
        let weight = one.input("weight", [1, 1, 2]);
        let output = one
            .convolution(
                input,
                weight,
                None,
                spec(&[2], &[1], &[1], &[(0, 0)], 1, None),
            )
            .unwrap();
        assert_eq!(one.shape(output).unwrap(), &Shape::from([1, 1, 3]));
        assert_eq!(
            execute(
                &one,
                output,
                data([1, 1, 4], &[1., 2., 3., 4.]),
                data([1, 1, 2], &[1., 1.]),
                None,
            ),
            data([1, 1, 3], &[3., 5., 7.])
        );

        let mut two = Graph::new();
        let input = two.input("input", [1, 1, 3, 3]);
        let weight = two.input("weight", [1, 1, 2, 2]);
        let output = two
            .convolution(
                input,
                weight,
                None,
                spec(&[2, 2], &[1, 1], &[1, 1], &[(0, 0), (0, 0)], 1, None),
            )
            .unwrap();
        assert_eq!(two.shape(output).unwrap(), &Shape::from([1, 1, 2, 2]));

        let mut three = Graph::new();
        let input = three.input("input", [1, 1, 2, 2, 2]);
        let weight = three.input("weight", [1, 1, 2, 1, 1]);
        let output = three
            .convolution(
                input,
                weight,
                None,
                spec(&[2, 1, 1], &[1, 1, 1], &[1, 1, 1], &[(0, 0); 3], 1, None),
            )
            .unwrap();
        assert_eq!(three.shape(output).unwrap(), &Shape::from([1, 1, 1, 2, 2]));
        assert_eq!(
            execute(
                &three,
                output,
                data([1, 1, 2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]),
                data([1, 1, 2, 1, 1], &[1., 1.]),
                None,
            ),
            data([1, 1, 1, 2, 2], &[6., 8., 10., 12.])
        );
    }

    #[test]
    fn signed_grouped_depthwise_bias_and_dtype_contracts_are_explicit() {
        let mut cropped = Graph::new();
        let input = cropped.input("input", [1, 1, 4]);
        let weight = cropped.input("weight", [1, 1, 2]);
        let output = cropped
            .convolution(
                input,
                weight,
                None,
                spec(&[2], &[1], &[1], &[(-1, 1)], 1, None),
            )
            .unwrap();
        assert_eq!(
            execute(
                &cropped,
                output,
                data([1, 1, 4], &[1., 2., 3., 4.]),
                data([1, 1, 2], &[1., 1.]),
                None,
            ),
            data([1, 1, 3], &[5., 7., 4.])
        );

        let mut anisotropic = Graph::new();
        let input = anisotropic.input("input", [1, 1, 5, 6]);
        let weight = anisotropic.input("weight", [1, 1, 2, 2]);
        let output = anisotropic
            .convolution(
                input,
                weight,
                None,
                spec(&[2, 2], &[2, 1], &[1, 2], &[(-1, 2), (1, -1)], 1, None),
            )
            .unwrap();
        assert_eq!(
            anisotropic.shape(output).unwrap(),
            &Shape::from([1, 1, 3, 4])
        );

        let mut grouped = Graph::new();
        let input = grouped.input("input", [1, 2, 2]);
        let weight = grouped.input("weight", [2, 1, 1]);
        let bias = grouped.input("bias", [2]);
        let output = grouped
            .convolution(
                input,
                weight,
                Some(bias),
                spec(&[1], &[1], &[1], &[(0, 0)], 2, None),
            )
            .unwrap();
        assert_eq!(
            execute(
                &grouped,
                output,
                data([1, 2, 2], &[1., 2., 3., 4.]),
                data([2, 1, 1], &[2., 3.]),
                Some(data([2], &[1., -1.])),
            ),
            data([1, 2, 2], &[3., 5., 8., 11.])
        );

        let mut dtypes = Graph::new();
        let input = dtypes.input_dtype("input", [1, 1, 2], DType::F64);
        let weight = dtypes.input_dtype("weight", [1, 1, 2], DType::F64);
        let default = dtypes
            .convolution(
                input,
                weight,
                None,
                spec(&[2], &[1], &[1], &[(0, 0)], 1, None),
            )
            .unwrap();
        assert_eq!(dtypes.dtype(default).unwrap(), DType::F64);
        assert_sum_contract(&dtypes, default, DType::F64, DType::F64);
        let explicit = dtypes
            .convolution(
                input,
                weight,
                None,
                spec(&[2], &[1], &[1], &[(0, 0)], 1, Some(DType::F32)),
            )
            .unwrap();
        assert_eq!(dtypes.dtype(explicit).unwrap(), DType::F32);
        assert!((0..dtypes.node_count()).any(|index| {
            let node = NodeId::from_index(index);
            matches!(
                dtypes.op(node),
                Ok(Op::Reduce {
                    kind: crate::ReduceKind::Sum,
                    ..
                })
            ) && dtypes.dtype(node) == Ok(DType::F32)
        }));

        let mut narrow = Graph::new();
        let input = narrow.input_dtype("input", [1, 1, 2], DType::F16);
        let weight = narrow.input_dtype("weight", [1, 1, 2], DType::F16);
        let output = narrow
            .convolution(
                input,
                weight,
                None,
                spec(&[2], &[1], &[1], &[(0, 0)], 1, None),
            )
            .unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), DType::F16);
        assert_sum_contract(&narrow, output, DType::F32, DType::F16);
        let trace = narrow.trace(output).unwrap().to_string();
        assert!(trace.contains("F32"));
        assert!(trace.contains("F16"));
    }

    #[test]
    fn source_sum_dtype_matrix_is_derived_from_authoritative_operands() {
        for (dtype, accumulator, output_dtype) in [
            (DType::Bool, DType::I32, DType::I32),
            (DType::I8, DType::I32, DType::I32),
            (DType::F16, DType::F32, DType::F16),
            (DType::BF16, DType::F32, DType::BF16),
            (DType::F8E4M3, DType::F32, DType::F8E4M3),
            (DType::F8E5M2, DType::F32, DType::F8E5M2),
            (DType::F8E4M3FNUZ, DType::F32, DType::F8E4M3FNUZ),
            (DType::F8E5M2FNUZ, DType::F32, DType::F8E5M2FNUZ),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [1, 1, 2], dtype);
            let weight = graph.input_dtype("weight", [1, 1, 2], dtype);
            let output = graph
                .convolution(
                    input,
                    weight,
                    None,
                    spec(&[2], &[1], &[1], &[(0, 0)], 1, None),
                )
                .unwrap();
            assert_sum_contract(&graph, output, accumulator, output_dtype);
        }
    }

    #[test]
    fn transposed_convolution_values_cover_zero_one_two_and_three_spatial_ranks() {
        assert_eq!(
            TransposedConvolutionSpec::new(
                SpatialWindow::new([1], [1], [1], [(0, 0)]).unwrap(),
                [],
                NonZeroUsize::new(1).unwrap(),
            ),
            Err(SpatialWindowError::RankMismatch)
        );

        let mut zero = Graph::new();
        let input = zero.input("input", [1, 2]);
        let weight = zero.input("weight", [2, 2]);
        let output = zero
            .transposed_convolution(
                input,
                weight,
                None,
                transposed_spec(&[], &[], &[], &[], &[], 1),
            )
            .unwrap();
        assert_eq!(zero.shape(output).unwrap(), &Shape::from([1, 2]));
        assert_eq!(
            execute(
                &zero,
                output,
                data([1, 2], &[1., 2.]),
                data([2, 2], &[1., 2., 3., 4.]),
                None,
            ),
            data([1, 2], &[7., 10.])
        );

        let mut one = Graph::new();
        let input = one.input("input", [1, 1, 2]);
        let weight = one.input("weight", [1, 1, 2]);
        let output = one
            .transposed_convolution(
                input,
                weight,
                None,
                transposed_spec(&[2], &[2], &[1], &[(0, 0)], &[1], 1),
            )
            .unwrap();
        assert_eq!(one.shape(output).unwrap(), &Shape::from([1, 1, 5]));
        assert_eq!(
            execute(
                &one,
                output,
                data([1, 1, 2], &[1., 2.]),
                data([1, 1, 2], &[1., 1.]),
                None,
            ),
            data([1, 1, 5], &[1., 1., 2., 2., 0.])
        );

        let mut two = Graph::new();
        let input = two.input("input", [1, 1, 2, 2]);
        let weight = two.input("weight", [1, 1, 2, 2]);
        let output = two
            .transposed_convolution(
                input,
                weight,
                None,
                transposed_spec(&[2, 2], &[1, 1], &[1, 1], &[(0, 0); 2], &[0, 0], 1),
            )
            .unwrap();
        assert_eq!(two.shape(output).unwrap(), &Shape::from([1, 1, 3, 3]));
        assert_eq!(
            execute(
                &two,
                output,
                data([1, 1, 2, 2], &[1., 2., 3., 4.]),
                data([1, 1, 2, 2], &[1.; 4]),
                None,
            ),
            data([1, 1, 3, 3], &[1., 3., 2., 4., 10., 6., 3., 7., 4.])
        );

        let mut three = Graph::new();
        let input = three.input("input", [1, 1, 2, 2, 2]);
        let weight = three.input("weight", [1, 1, 1, 1, 1]);
        let output = three
            .transposed_convolution(
                input,
                weight,
                None,
                transposed_spec(&[1, 1, 1], &[1, 1, 1], &[1, 1, 1], &[(0, 0); 3], &[0; 3], 1),
            )
            .unwrap();
        assert_eq!(three.shape(output).unwrap(), &Shape::from([1, 1, 2, 2, 2]));
        assert_eq!(
            execute(
                &three,
                output,
                data([1, 1, 2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]),
                data([1, 1, 1, 1, 1], &[2.]),
                None,
            ),
            data([1, 1, 2, 2, 2], &[2., 4., 6., 8., 10., 12., 14., 16.])
        );
    }

    #[test]
    fn transposed_convolution_signed_geometry_groups_bias_and_dtype_are_source_derived() {
        let mut geometry = Graph::new();
        let input = geometry.input("input", [1, 1, 2, 3]);
        let weight = geometry.input("weight", [1, 1, 2, 2]);
        let output = geometry
            .transposed_convolution(
                input,
                weight,
                None,
                transposed_spec(&[2, 2], &[2, 1], &[1, 2], &[(-1, 0), (1, -1)], &[3, -1], 1),
            )
            .unwrap();
        assert_eq!(geometry.shape(output).unwrap(), &Shape::from([1, 1, 8, 4]));

        let mut grouped = Graph::new();
        let input = grouped.input("input", [1, 2, 2]);
        let weight = grouped.input("weight", [2, 1, 1]);
        let bias = grouped.input("bias", [2]);
        let output = grouped
            .transposed_convolution(
                input,
                weight,
                Some(bias),
                transposed_spec(&[1], &[1], &[1], &[(0, 0)], &[0], 2),
            )
            .unwrap();
        assert_eq!(
            execute(
                &grouped,
                output,
                data([1, 2, 2], &[1., 2., 3., 4.]),
                data([2, 1, 1], &[2., 3.]),
                Some(data([2], &[1., -1.])),
            ),
            data([1, 2, 2], &[3., 5., 8., 11.])
        );

        for (dtype, accumulator, output_dtype) in [
            (DType::I8, DType::I32, DType::I32),
            (DType::F16, DType::F32, DType::F16),
            (DType::BF16, DType::F32, DType::BF16),
            (DType::F64, DType::F64, DType::F64),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [1, 1, 2], dtype);
            let weight = graph.input_dtype("weight", [1, 1, 2], dtype);
            let output = graph
                .transposed_convolution(
                    input,
                    weight,
                    None,
                    transposed_spec(&[2], &[1], &[1], &[(0, 0)], &[0], 1),
                )
                .unwrap();
            assert_sum_contract(&graph, output, accumulator, output_dtype);
        }
    }

    #[test]
    fn transposed_convolution_is_atomic_compositional_and_higher_order_differentiable() {
        let mut graph = Graph::new();
        let input = graph.input("input", [1, 1, 2]);
        let weight = graph.input("weight", [1, 1, 2]);
        let output = graph
            .transposed_convolution(
                input,
                weight,
                None,
                transposed_spec(&[2], &[2], &[1], &[(0, 0)], &[1], 1),
            )
            .unwrap();
        assert!((0..graph.node_count()).all(|index| {
            !matches!(
                graph.op(NodeId::from_index(index)).unwrap(),
                Op::ConvTranspose2d { .. }
                    | Op::ConvTranspose2dGrad { .. }
                    | Op::ConvTranspose2dGradVjp { .. }
            )
        }));
        let squared = graph.square(output).unwrap();
        let loss = graph.sum_all(squared).unwrap();
        let first = graph.grad(loss, input).unwrap();
        let second_loss = graph.sum_all(first).unwrap();
        let second = graph.grad(second_loss, input).unwrap();
        assert_eq!(graph.shape(second).unwrap(), &Shape::from([1, 1, 2]));

        let mut invalid = Graph::new();
        let input = invalid.input("input", [1, 2, 3]);
        let weight = invalid.input("weight", [1, 1, 2]);
        let before = invalid.node_count();
        assert!(matches!(
            invalid.transposed_convolution(
                input,
                weight,
                None,
                transposed_spec(&[2], &[1], &[1], &[(0, 0)], &[0], 1),
            ),
            Err(Error::InvalidConvolution {
                reason: "channel/group geometry",
                ..
            })
        ));
        assert_eq!(invalid.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("input", [1, 1, usize::MAX]);
        let weight = overflow.input("weight", [1, 1, 1]);
        let before = overflow.node_count();
        assert!(matches!(
            overflow.transposed_convolution(
                input,
                weight,
                None,
                transposed_spec(&[1], &[2], &[1], &[(0, 0)], &[0], 1),
            ),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), before);

        let mut zero_batch = Graph::new();
        let input = zero_batch.input("input", [0, 1, 2]);
        let weight = zero_batch.input("weight", [1, 1, 1]);
        let output = zero_batch
            .transposed_convolution(
                input,
                weight,
                None,
                transposed_spec(&[1], &[1], &[1], &[(0, 0)], &[0], 1),
            )
            .unwrap();
        assert_eq!(zero_batch.shape(output).unwrap(), &Shape::from([0, 1, 2]));

        // As in tinygrad's checked-in `_pool`, an effective kernel cannot be
        // formed from an unpadded zero spatial axis. Rejection is descriptor
        // only and therefore publishes no partial stride/window chain.
        let mut zero_spatial = Graph::new();
        let input = zero_spatial.input("input", [1, 1, 0]);
        let weight = zero_spatial.input("weight", [1, 1, 1]);
        let before = zero_spatial.node_count();
        assert!(matches!(
            zero_spatial.transposed_convolution(
                input,
                weight,
                None,
                transposed_spec(&[1], &[1], &[1], &[(0, 0)], &[0], 1),
            ),
            Err(Error::InvalidConvolution {
                reason: "kernel exceeds transformed padded input",
                ..
            })
        ));
        assert_eq!(zero_spatial.node_count(), before);
    }

    #[test]
    fn transposed_convolution_schedule_capture_interpreter_and_strict_native_share_results() {
        let mut graph = Graph::new();
        let input = graph.input("input", [1, 1, 2]);
        let weight = graph.input("weight", [1, 1, 2]);
        let output = graph
            .transposed_convolution(
                input,
                weight,
                None,
                transposed_spec(&[2], &[2], &[1], &[(0, 0)], &[1], 1),
            )
            .unwrap();
        let schedule = crate::schedule_many(&graph, &[output]).unwrap();
        assert!(schedule.items.iter().all(|item| item.boundary.is_none()));
        assert!(
            schedule
                .items
                .iter()
                .all(|item| { !matches!(item.kernel.operation(), crate::Operation::Conv2d(_)) })
        );
        assert!(
            schedule
                .items
                .iter()
                .all(|item| crate::CpuJit::render(&item.kernel).is_ok())
        );
        let capture = crate::CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        let encoded = capture.to_bytes().unwrap();
        let capture = crate::CapturedSchedule::from_bytes(&encoded).unwrap();
        assert_eq!(capture.to_bytes().unwrap(), encoded);
        let bindings = BTreeMap::from([
            ("input".into(), data([1, 1, 2], &[1., 2.])),
            ("weight".into(), data([1, 1, 2], &[1., 1.])),
        ]);
        let executor = crate::CapturedReplayExecutor::default();
        let interpreter = executor
            .replay(&capture, &bindings, crate::CapturedReplayOptions::default())
            .unwrap();
        let native = executor
            .replay(
                &capture,
                &bindings,
                crate::CapturedReplayOptions {
                    backend: crate::CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(
            interpreter.outputs[0].storage(),
            native.outputs[0].storage()
        );
        assert_eq!(native.outputs[0], data([1, 1, 5], &[1., 1., 2., 2., 0.]));
    }

    #[test]
    fn compositional_convolution_has_no_forward_conv_node_and_supports_higher_order_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("input", [1, 1, 2]);
        let weight = graph.input("weight", [1, 1, 1]);
        let output = graph
            .convolution(
                input,
                weight,
                None,
                spec(&[1], &[1], &[1], &[(0, 0)], 1, None),
            )
            .unwrap();
        assert!((0..graph.node_count()).all(|index| {
            !matches!(
                graph.op(NodeId::from_index(index)).unwrap(),
                Op::Conv2d { .. }
            )
        }));
        let squared = graph.square(output).unwrap();
        let loss = graph.sum_all(squared).unwrap();
        let first = graph.grad(loss, input).unwrap();
        let first_sum = graph.sum_all(first).unwrap();
        let second = graph.grad(first_sum, input).unwrap();
        assert_eq!(graph.shape(second).unwrap(), &Shape::from([1, 1, 2]));
        assert!(
            !graph
                .trace(second)
                .unwrap()
                .to_string()
                .contains("conv2d_grad")
        );
    }

    #[test]
    fn invalid_late_geometry_and_overflow_publish_nothing() {
        let mut mismatch = Graph::new();
        let input = mismatch.input("input", [1, 1, 3]);
        let weight = mismatch.input("weight", [1, 1, 2]);
        let before = mismatch.node_count();
        let wrong = spec(&[1], &[1], &[1], &[(0, 0)], 1, None);
        assert!(matches!(
            mismatch.convolution(input, weight, None, wrong),
            Err(Error::InvalidConvolution {
                input: actual_input,
                weight: actual_weight,
                reason: "weight kernel does not match spatial window",
            }) if actual_input == Shape::from([1, 1, 3])
                && actual_weight == Shape::from([1, 1, 2])
        ));
        assert_eq!(mismatch.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("input", [usize::MAX, 1, 2]);
        let weight = overflow.input("weight", [1, 1, 1]);
        let before = overflow.node_count();
        let valid = spec(&[1], &[1], &[1], &[(0, 0)], 1, None);
        assert!(matches!(
            overflow.convolution(input, weight, None, valid),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn compositional_convolution_schedules_and_captures_without_legacy_conv_operation() {
        let mut graph = Graph::new();
        let input = graph.input("input", [1, 1, 3]);
        let weight = graph.input("weight", [1, 1, 2]);
        let output = graph
            .convolution(
                input,
                weight,
                None,
                spec(&[2], &[1], &[1], &[(0, 0)], 1, None),
            )
            .unwrap();
        let schedule = crate::schedule_many(&graph, &[output]).unwrap();
        assert!(schedule.items.iter().all(|item| item.boundary.is_none()));
        assert!((0..graph.node_count()).all(|index| {
            !matches!(graph.op(NodeId::from_index(index)), Ok(Op::Concat { .. }))
        }));
        assert!(schedule.items.iter().any(|item| {
            item.kernel.topological().is_ok_and(|nodes| {
                nodes
                    .into_iter()
                    .filter(crate::projected_index::ProjectedIndexPlan::is_projected)
                    .filter_map(|index| {
                        crate::projected_index::ProjectedIndexPlan::from_index(&index).ok()
                    })
                    .any(|plan| plan.buffer == input.index() as u64)
            })
        }));
        assert!(
            schedule
                .items
                .iter()
                .all(|item| !matches!(item.kernel.operation(), crate::Operation::Conv2d(_)))
        );
        let captured = crate::CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        assert!(!captured.requested.is_empty());
    }

    #[test]
    fn padded_stride_two_k7_windows_project_directly_from_the_source_image() {
        let mut graph = Graph::new();
        let input = graph.input("input", [1, 3, 224, 224]);
        let weight = graph.input("weight", [64, 3, 7, 7]);
        let output = graph
            .conv2d(
                input,
                weight,
                None,
                crate::Conv2dOptions {
                    groups: 1,
                    stride: [2, 2],
                    dilation: [1, 1],
                    padding: [3; 4],
                },
            )
            .unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        schedule.validate().unwrap();
        assert!(schedule.items.iter().all(|item| item.boundary.is_none()));
        assert!((0..graph.node_count()).all(|index| {
            !matches!(graph.op(NodeId::from_index(index)), Ok(Op::Concat { .. }))
        }));
        let (projection_index, projection) = schedule
            .items
            .iter()
            .flat_map(|item| item.kernel.topological().unwrap())
            .filter(crate::projected_index::ProjectedIndexPlan::is_predicated)
            .find_map(|index| {
                let plan = crate::projected_index::ProjectedIndexPlan::from_index(&index).ok()?;
                (plan.buffer == input.index() as u64
                    && plan.elements == 3 * 224 * 224
                    && plan.output_elements == 64 * 112 * 112 * 3 * 7 * 7)
                    .then_some((index, plan))
            })
            .expect("padded stem window must project from the source image");
        let mut projected_occurrences = 0usize;
        let mut pending = projection_index.sources()[1..].iter().collect::<Vec<_>>();
        while let Some(node) = pending.pop() {
            projected_occurrences += 1;
            pending.extend(node.sources());
        }
        assert!(projected_occurrences < crate::projected_index::MAX_PROJECTED_INDEX_NODES);
        assert!(projection.is_guarded());
        assert!(!projection.valid(0).unwrap());
        let captured = crate::CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        assert_eq!(captured.requested, vec![output.index() as u64]);
    }

    #[test]
    fn rank_eight_window_projection_compacts_before_reduction_lowering() {
        let mut graph = Graph::new();
        let input = graph.input("input", [1, 1, 8, 8]);
        let weight = graph.input("weight", [32, 1, 3, 3]);
        let output = graph
            .conv2d(
                input,
                weight,
                None,
                crate::Conv2dOptions {
                    groups: 1,
                    stride: [2, 2],
                    dilation: [1, 1],
                    padding: [0, 1, 0, 1],
                },
            )
            .unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        schedule.validate().unwrap();
        let projected = schedule
            .items
            .iter()
            .flat_map(|item| item.kernel.topological().unwrap())
            .filter(crate::projected_index::ProjectedIndexPlan::is_projected)
            .collect::<Vec<_>>();
        assert!(!projected.is_empty());
        assert!(projected.iter().all(|index| {
            crate::projected_index::ProjectedIndexPlan::from_index(index).is_ok()
        }));
        let captured = crate::CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        assert_eq!(captured.requested, vec![output.index() as u64]);
    }
}
