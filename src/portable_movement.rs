//! Shared source-program coordination for portable dense materialization.

use crate::movement_plan::{PortableDenseMaterialization, PortableDenseRegion};

/// Backend syntax hooks for one output-driven dense materialization program.
///
/// The checked projection owns source order, hyperrectangles, strides, fill
/// bytes, and exact storage width. Dialects only spell those terms in their
/// target language.
pub(crate) trait PortableDenseMaterializationDialect {
    fn begin(&self, plan: &PortableDenseMaterialization<'_>) -> Vec<String>;
    fn select_region(
        &self,
        plan: &PortableDenseMaterialization<'_>,
        region: &PortableDenseRegion,
        first: bool,
    ) -> Vec<String>;
    fn store(&self, plan: &PortableDenseMaterialization<'_>) -> Vec<String>;
}

/// Emits one work item's complete source-selection and raw-storage program.
pub(crate) fn emit_portable_dense_materialization_body(
    plan: &PortableDenseMaterialization<'_>,
    dialect: &impl PortableDenseMaterializationDialect,
) -> Vec<String> {
    if plan.elements() == 0 {
        return Vec::new();
    }
    let mut lines = dialect.begin(plan);
    let mut first = true;
    for region in plan.regions().iter().filter(|region| !region.is_empty()) {
        lines.extend(dialect.select_region(plan, region, first));
        first = false;
    }
    lines.extend(dialect.store(plan));
    lines
}

/// Shared C-family spelling used by OpenCL C and Metal Shading Language.
pub(crate) struct CLikePortableDenseDialect {
    pub(crate) input_address: &'static str,
    pub(crate) output_address: &'static str,
}

fn c_like_coordinate(axis: &crate::movement_plan::PortableDenseAxis) -> String {
    format!(
        "((gid / {}ul) % {}ul)",
        axis.output_divisor, axis.output_dimension
    )
}

impl PortableDenseMaterializationDialect for CLikePortableDenseDialect {
    fn begin(&self, _: &PortableDenseMaterialization<'_>) -> Vec<String> {
        vec![
            "  uint rg_input = 0xffffffffu;".into(),
            "  ulong rg_source = 0ul;".into(),
        ]
    }

    fn select_region(
        &self,
        _: &PortableDenseMaterialization<'_>,
        region: &PortableDenseRegion,
        first: bool,
    ) -> Vec<String> {
        let condition = if region.axes.is_empty() {
            "true".into()
        } else {
            region
                .axes
                .iter()
                .map(|axis| {
                    let coordinate = c_like_coordinate(axis);
                    format!(
                        "({coordinate} >= {}ul && {coordinate} < {}ul)",
                        axis.output_start,
                        axis.output_start + axis.length
                    )
                })
                .collect::<Vec<_>>()
                .join(" && ")
        };
        let source = region
            .axes
            .iter()
            .filter(|axis| axis.length != 0 && axis.source_stride != 0)
            .map(|axis| {
                format!(
                    "({}-{}ul)*{}ul",
                    c_like_coordinate(axis),
                    axis.output_start,
                    axis.source_stride
                )
            })
            .collect::<Vec<_>>();
        let source = if source.is_empty() {
            "0ul".into()
        } else {
            source.join(" + ")
        };
        vec![format!(
            "  {} ({condition}) {{ rg_input = {}u; rg_source = {source}; }}",
            if first { "if" } else { "else if" },
            region.input_abi
        )]
    }

    fn store(&self, plan: &PortableDenseMaterialization<'_>) -> Vec<String> {
        let output = plan.inputs().len();
        let fill = plan.fill_bytes();
        let mut lines = Vec::new();
        for lane in 0..plan.width() {
            lines.push(format!("  uchar rg_byte_{lane} = (uchar)0;"));
            for input in 0..plan.inputs().len() {
                lines.push(format!(
                    "  {} (rg_input == {input}u) rg_byte_{lane} = (({} const uchar*)b{input})[rg_source*{}ul+{}ul];",
                    if input == 0 { "if" } else { "else if" },
                    self.input_address,
                    plan.width(),
                    lane
                ));
            }
            if let Some(bytes) = fill {
                lines.push(format!(
                    "  else rg_byte_{lane} = (uchar)0x{:02x}u;",
                    bytes[lane]
                ));
            }
            lines.push(format!(
                "  (({} uchar*)b{output})[gid*{}ul+{}ul] = rg_byte_{lane};",
                self.output_address,
                plan.width(),
                lane
            ));
        }
        lines
    }
}

pub(crate) struct WgslPortableDenseDialect;

fn wgsl_coordinate(axis: &crate::movement_plan::PortableDenseAxis) -> String {
    format!(
        "((gid / {}u) % {}u)",
        axis.output_divisor, axis.output_dimension
    )
}

impl PortableDenseMaterializationDialect for WgslPortableDenseDialect {
    fn begin(&self, _: &PortableDenseMaterialization<'_>) -> Vec<String> {
        vec![
            "  var rg_input: u32 = 0xffffffffu;".into(),
            "  var rg_source: u32 = 0u;".into(),
        ]
    }

    fn select_region(
        &self,
        _: &PortableDenseMaterialization<'_>,
        region: &PortableDenseRegion,
        first: bool,
    ) -> Vec<String> {
        let condition = if region.axes.is_empty() {
            "true".into()
        } else {
            region
                .axes
                .iter()
                .map(|axis| {
                    let coordinate = wgsl_coordinate(axis);
                    format!(
                        "({coordinate} >= {}u && {coordinate} < {}u)",
                        axis.output_start,
                        axis.output_start + axis.length
                    )
                })
                .collect::<Vec<_>>()
                .join(" && ")
        };
        let source = region
            .axes
            .iter()
            .filter(|axis| axis.length != 0 && axis.source_stride != 0)
            .map(|axis| {
                format!(
                    "({}-{}u)*{}u",
                    wgsl_coordinate(axis),
                    axis.output_start,
                    axis.source_stride
                )
            })
            .collect::<Vec<_>>();
        let source = if source.is_empty() {
            "0u".into()
        } else {
            source.join(" + ")
        };
        vec![format!(
            "  {} ({condition}) {{ rg_input = {}u; rg_source = {source}; }}",
            if first { "if" } else { "else if" },
            region.input_abi
        )]
    }

    fn store(&self, plan: &PortableDenseMaterialization<'_>) -> Vec<String> {
        let output = plan.inputs().len();
        let fill = plan.fill_bytes();
        let mut lines = Vec::new();
        for lane in 0..plan.width() {
            lines.push(format!("  var rg_byte_{lane}: u32 = 0u;"));
            for input in 0..plan.inputs().len() {
                lines.push(format!(
                    "  {} (rg_input == {input}u) {{ let rg_read_{lane}: u32 = rg_source*{}u+{}u; rg_byte_{lane} = (b{input}[rg_read_{lane} >> 2u] >> ((rg_read_{lane} & 3u)*8u)) & 255u; }}",
                    if input == 0 { "if" } else { "else if" },
                    plan.width(),
                    lane
                ));
            }
            if let Some(bytes) = fill {
                lines.push(format!(
                    "  else {{ rg_byte_{lane} = 0x{:02x}u; }}",
                    bytes[lane]
                ));
            }
            lines.extend([
                format!(
                    "  let rg_write_{lane}: u32 = gid*{}u+{}u;",
                    plan.width(),
                    lane
                ),
                format!(
                    "  let rg_shift_{lane}: u32 = (rg_write_{lane} & 3u)*8u;"
                ),
                format!(
                    "  atomicAnd(&b{output}[rg_write_{lane} >> 2u], ~(255u << rg_shift_{lane}));"
                ),
                format!(
                    "  atomicOr(&b{output}[rg_write_{lane} >> 2u], rg_byte_{lane} << rg_shift_{lane});"
                ),
            ]);
        }
        lines
    }
}
