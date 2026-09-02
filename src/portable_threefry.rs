//! Checked backend-neutral projection for live packed-U64 Threefry2x32.

use crate::{DType, NodeId, ScheduleInputBinding, Shape, ThreefryValue};
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PortableThreefryError {
    InvalidPlan(String),
    InvalidBinding(String),
    Unsupported(&'static str),
    Overflow,
}

impl fmt::Display for PortableThreefryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(reason) => write!(f, "invalid Threefry payload: {reason}"),
            Self::InvalidBinding(reason) => write!(f, "invalid Threefry binding: {reason}"),
            Self::Unsupported(reason) => write!(f, "unsupported Threefry: {reason}"),
            Self::Overflow => f.write_str("Threefry geometry overflow"),
        }
    }
}

impl std::error::Error for PortableThreefryError {}

/// One checked output-coordinate contribution to a broadcast input address.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PortableThreefryAxis {
    pub(crate) output_stride: usize,
    pub(crate) output_extent: usize,
    pub(crate) input_stride: usize,
}

/// One canonical first-use input in the dependency-bearing pointer ABI.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PortableThreefryInput {
    pub(crate) node: NodeId,
    pub(crate) shape: Shape,
    pub(crate) elements: usize,
    pub(crate) abi_index: usize,
    axes: Vec<PortableThreefryAxis>,
}

/// Fully checked common projection for static accelerator Threefry kernels.
///
/// `ThreefryValue` remains the sole operation taxonomy. This view proves its
/// exact dense U64 ABI and right-aligned broadcast geometry once; renderers
/// only spell the shared 2x32 program in their storage language.
pub(crate) struct PortableThreefry<'a> {
    value: &'a ThreefryValue,
    inputs: Vec<PortableThreefryInput>,
    elements: usize,
}

impl<'a> PortableThreefry<'a> {
    pub(crate) fn new(value: &'a ThreefryValue) -> Result<Self, PortableThreefryError> {
        value
            .validate()
            .map_err(|error| PortableThreefryError::InvalidPlan(error.to_string()))?;
        let elements = value
            .output_shape
            .numel()
            .map_err(|_| PortableThreefryError::Overflow)?;
        if elements > u32::MAX as usize {
            return Err(PortableThreefryError::Unsupported(
                "portable Threefry requires a 32-bit output domain",
            ));
        }
        let inputs = value
            .input_operands()
            .enumerate()
            .map(|(abi_index, (node, shape))| {
                project_input(node, shape, &value.output_shape, abi_index)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            value,
            inputs,
            elements,
        })
    }

    pub(crate) fn value(&self) -> &'a ThreefryValue {
        self.value
    }

    pub(crate) fn inputs(&self) -> &[PortableThreefryInput] {
        &self.inputs
    }

    pub(crate) fn elements(&self) -> usize {
        self.elements
    }

    pub(crate) fn output_abi_index(&self) -> usize {
        self.inputs.len()
    }

    pub(crate) fn input(&self, node: NodeId) -> &PortableThreefryInput {
        self.inputs
            .iter()
            .find(|input| input.node == node)
            .expect("validated Threefry input belongs to canonical ABI")
    }

    pub(crate) fn validate_schedule_bindings(
        &self,
        bindings: &[ScheduleInputBinding],
    ) -> Result<(), PortableThreefryError> {
        if bindings.len() != self.inputs.len() {
            return Err(PortableThreefryError::InvalidBinding(
                "Threefry input count does not match its deduplicated ABI".into(),
            ));
        }
        for (binding, input) in bindings.iter().zip(&self.inputs) {
            let bytes = input
                .elements
                .checked_mul(DType::U64.itemsize())
                .ok_or(PortableThreefryError::Overflow)?;
            if binding.abi_index != input.abi_index
                || binding.input_node != input.node
                || binding.desc.id != input.node.index() as u64
                || binding.desc.shape != input.shape
                || binding.desc.dtype != DType::U64
                || binding.desc.bytes != bytes
                || !binding.desc.read_only
                || binding.desc.view.is_some()
            {
                return Err(PortableThreefryError::InvalidBinding(format!(
                    "Threefry input {} is not its exact dense U64 descriptor",
                    input.abi_index
                )));
            }
        }
        Ok(())
    }
}

fn project_input(
    node: NodeId,
    shape: &Shape,
    output: &Shape,
    abi_index: usize,
) -> Result<PortableThreefryInput, PortableThreefryError> {
    let elements = shape.numel().map_err(|_| PortableThreefryError::Overflow)?;
    if elements > u32::MAX as usize {
        return Err(PortableThreefryError::Unsupported(
            "portable Threefry requires 32-bit input indexing",
        ));
    }
    let rank_delta = output
        .rank()
        .checked_sub(shape.rank())
        .ok_or_else(|| PortableThreefryError::InvalidPlan("input rank exceeds output".into()))?;
    let output_is_empty = output
        .numel()
        .map_err(|_| PortableThreefryError::Overflow)?
        == 0;
    let output_strides = if output_is_empty {
        Vec::new()
    } else {
        dense_strides(output.dims())?
    };
    let input_strides = if output_is_empty {
        Vec::new()
    } else {
        dense_strides(shape.dims())?
    };
    let mut axes = Vec::new();
    for (output_axis, &output_extent) in output.dims().iter().enumerate() {
        let Some(input_axis) = output_axis.checked_sub(rank_delta) else {
            continue;
        };
        let input_extent = shape.dims()[input_axis];
        if input_extent != 1 && input_extent != output_extent {
            return Err(PortableThreefryError::InvalidPlan(
                "input shape does not broadcast to output".into(),
            ));
        }
        if !output_is_empty && input_extent != 1 {
            axes.push(PortableThreefryAxis {
                output_stride: output_strides[output_axis],
                output_extent,
                input_stride: input_strides[input_axis],
            });
        }
    }
    Ok(PortableThreefryInput {
        node,
        shape: shape.clone(),
        elements,
        abi_index,
        axes,
    })
}

fn dense_strides(dims: &[usize]) -> Result<Vec<usize>, PortableThreefryError> {
    let mut stride = 1usize;
    let mut strides = vec![0; dims.len()];
    for (axis, extent) in dims.iter().enumerate().rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(*extent)
            .ok_or(PortableThreefryError::Overflow)?;
    }
    Ok(strides)
}

/// Backend syntax hooks for the shared exact 20-round program.
pub(crate) trait PortableThreefryDialect {
    fn begin(&self, plan: &PortableThreefry<'_>) -> Vec<String>;
    fn load(&self, input: &PortableThreefryInput, name: &str) -> Vec<String>;
    fn initialize(&self) -> Vec<String>;
    fn mix(&self, rotation: u32) -> Vec<String>;
    fn inject(&self, injection: usize) -> Vec<String>;
    fn store(&self, plan: &PortableThreefry<'_>) -> Vec<String>;
}

/// Emits one work item's source-literal packed-U64 Threefry2x32 program.
pub(crate) fn emit_portable_threefry_body(
    plan: &PortableThreefry<'_>,
    dialect: &impl PortableThreefryDialect,
) -> Vec<String> {
    let mut lines = dialect.begin(plan);
    lines.extend(dialect.load(plan.input(plan.value.counter), "counter"));
    lines.extend(dialect.load(plan.input(plan.value.key), "key"));
    lines.extend(dialect.initialize());
    for (_round, rotation, injection) in crate::random::threefry_rounds() {
        lines.extend(dialect.mix(rotation));
        if let Some(injection) = injection {
            lines.extend(dialect.inject(injection));
        }
    }
    lines.extend(dialect.store(plan));
    lines
}

/// Shared C-family spelling used by OpenCL C and Metal Shading Language.
pub(crate) struct CLikePortableThreefryDialect;

impl PortableThreefryDialect for CLikePortableThreefryDialect {
    fn begin(&self, _plan: &PortableThreefry<'_>) -> Vec<String> {
        Vec::new()
    }

    fn load(&self, input: &PortableThreefryInput, name: &str) -> Vec<String> {
        let offset = c_offset(input);
        vec![
            format!(
                "  const ulong rg_{name}_word = b{}[{offset}];",
                input.abi_index
            ),
            format!("  const uint rg_{name}0 = (uint)rg_{name}_word;"),
            format!("  const uint rg_{name}1 = (uint)(rg_{name}_word >> 32u);"),
        ]
    }

    fn initialize(&self) -> Vec<String> {
        vec![
            format!(
                "  const uint rg_key2 = rg_key0 ^ rg_key1 ^ 0x{:08x}u;",
                crate::random::THREEFRY_PARITY
            ),
            "  uint rg_x0 = rg_counter0 + rg_key0;".into(),
            "  uint rg_x1 = rg_counter1 + rg_key1;".into(),
        ]
    }

    fn mix(&self, rotation: u32) -> Vec<String> {
        vec![
            "  rg_x0 += rg_x1;".into(),
            format!(
                "  rg_x1 = ((rg_x1 << {rotation}u) | (rg_x1 >> {}u)) ^ rg_x0;",
                32 - rotation
            ),
        ]
    }

    fn inject(&self, injection: usize) -> Vec<String> {
        vec![
            format!("  rg_x0 += {};", c_key(injection % 3)),
            format!("  rg_x1 += {} + {injection}u;", c_key((injection + 1) % 3)),
        ]
    }

    fn store(&self, plan: &PortableThreefry<'_>) -> Vec<String> {
        vec![format!(
            "  b{}[gid] = (ulong)rg_x0 | ((ulong)rg_x1 << 32u);",
            plan.output_abi_index()
        )]
    }
}

fn c_key(index: usize) -> &'static str {
    match index {
        0 => "rg_key0",
        1 => "rg_key1",
        2 => "rg_key2",
        _ => unreachable!("Threefry key index modulo three"),
    }
}

fn c_offset(input: &PortableThreefryInput) -> String {
    if input.axes.is_empty() {
        return "0ul".into();
    }
    input
        .axes
        .iter()
        .map(|axis| {
            format!(
                "((gid / {}ul) % {}ul) * {}ul",
                axis.output_stride, axis.output_extent, axis.input_stride
            )
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

/// WGSL spelling over raw packed-u32 storage. Each logical U64 occupies two
/// adjacent words, so native shader-u64 support is unnecessary.
pub(crate) struct WgslPortableThreefryDialect;

impl PortableThreefryDialect for WgslPortableThreefryDialect {
    fn begin(&self, _plan: &PortableThreefry<'_>) -> Vec<String> {
        Vec::new()
    }

    fn load(&self, input: &PortableThreefryInput, name: &str) -> Vec<String> {
        let offset = wgsl_offset(input);
        vec![
            format!("  let rg_{name}_offset: u32 = ({offset}) * 2u;"),
            format!(
                "  let rg_{name}0: u32 = b{}[rg_{name}_offset];",
                input.abi_index
            ),
            format!(
                "  let rg_{name}1: u32 = b{}[rg_{name}_offset + 1u];",
                input.abi_index
            ),
        ]
    }

    fn initialize(&self) -> Vec<String> {
        vec![
            format!(
                "  let rg_key2: u32 = rg_key0 ^ rg_key1 ^ 0x{:08x}u;",
                crate::random::THREEFRY_PARITY
            ),
            "  var rg_x0: u32 = rg_counter0 + rg_key0;".into(),
            "  var rg_x1: u32 = rg_counter1 + rg_key1;".into(),
        ]
    }

    fn mix(&self, rotation: u32) -> Vec<String> {
        vec![
            "  rg_x0 = rg_x0 + rg_x1;".into(),
            format!(
                "  rg_x1 = ((rg_x1 << {rotation}u) | (rg_x1 >> {}u)) ^ rg_x0;",
                32 - rotation
            ),
        ]
    }

    fn inject(&self, injection: usize) -> Vec<String> {
        vec![
            format!("  rg_x0 = rg_x0 + {};", wgsl_key(injection % 3)),
            format!(
                "  rg_x1 = rg_x1 + {} + {injection}u;",
                wgsl_key((injection + 1) % 3)
            ),
        ]
    }

    fn store(&self, plan: &PortableThreefry<'_>) -> Vec<String> {
        vec![
            "  let rg_output_offset: u32 = gid * 2u;".into(),
            format!("  b{}[rg_output_offset] = rg_x0;", plan.output_abi_index()),
            format!(
                "  b{}[rg_output_offset + 1u] = rg_x1;",
                plan.output_abi_index()
            ),
        ]
    }
}

fn wgsl_key(index: usize) -> &'static str {
    c_key(index)
}

fn wgsl_offset(input: &PortableThreefryInput) -> String {
    if input.axes.is_empty() {
        return "0u".into();
    }
    input
        .axes
        .iter()
        .map(|axis| {
            format!(
                "((gid / {}u) % {}u) * {}u",
                axis.output_stride, axis.output_extent, axis.input_stride
            )
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeId;

    #[test]
    fn projection_authenticates_broadcast_and_alias_abi() {
        let value = ThreefryValue {
            counter: NodeId::from_index(1),
            key: NodeId::from_index(2),
            counter_shape: Shape::new([2, 1, 3]),
            key_shape: Shape::new([1, 4, 1]),
            output: NodeId::from_index(3),
            output_shape: Shape::new([2, 4, 3]),
        };
        let plan = PortableThreefry::new(&value).unwrap();
        assert_eq!(plan.elements(), 24);
        assert_eq!(plan.inputs().len(), 2);
        assert_eq!(plan.inputs()[0].axes.len(), 2);
        assert_eq!(plan.inputs()[1].axes.len(), 1);
        let source = emit_portable_threefry_body(&plan, &CLikePortableThreefryDialect).join("\n");
        assert_eq!(source.matches("rg_x1 = ((rg_x1 <<").count(), 20);
        assert_eq!(source.matches("rg_x1 += rg_key").count(), 5);

        let aliased = ThreefryValue {
            key: value.counter,
            key_shape: value.counter_shape.clone(),
            output_shape: value.counter_shape.clone(),
            ..value
        };
        assert_eq!(PortableThreefry::new(&aliased).unwrap().inputs().len(), 1);

        let empty = ThreefryValue {
            counter: NodeId::from_index(4),
            key: NodeId::from_index(5),
            counter_shape: Shape::new([2, 0, 3]),
            key_shape: Shape::new([1, 0, 1]),
            output: NodeId::from_index(6),
            output_shape: Shape::new([2, 0, 3]),
        };
        let empty = PortableThreefry::new(&empty).unwrap();
        let source = emit_portable_threefry_body(&empty, &CLikePortableThreefryDialect).join("\n");
        assert!(!source.contains("/ 0ul"));
        assert!(!source.contains("% 0ul"));

        let beyond_wgsl_u32 = (u32::MAX as usize).checked_add(1).unwrap();
        let extreme_empty = ThreefryValue {
            counter: NodeId::from_index(7),
            key: NodeId::from_index(8),
            counter_shape: Shape::new([0, beyond_wgsl_u32]),
            key_shape: Shape::new([0, beyond_wgsl_u32]),
            output: NodeId::from_index(9),
            output_shape: Shape::new([0, beyond_wgsl_u32]),
        };
        let extreme_empty = PortableThreefry::new(&extreme_empty).unwrap();
        assert!(
            extreme_empty
                .inputs()
                .iter()
                .all(|input| input.axes.is_empty())
        );
        let source =
            emit_portable_threefry_body(&extreme_empty, &WgslPortableThreefryDialect).join("\n");
        assert!(!source.contains("4294967296u"));
    }
}
