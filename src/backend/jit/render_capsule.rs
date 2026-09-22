//! Fail-soft durable ownership for authenticated native render results.
//!
//! Capsules are private cache sidecars. Their key is an exact, versioned
//! pre-render recipe; their body retains the complete rendered program. A
//! missing or invalid sidecar is only a cache miss and never changes the
//! authoritative renderer's errors.

use super::{
    CpuJitBackend, NativeScheduleLayout, NativeStoreGroup, PreparedZeroDomainEntry,
    RenderedScheduleEntry, RenderedScheduleModule, validate_native_layout,
};
use crate::cpu_jit::{
    ABI_VERSION, BufferAbi, KernelAbi, KernelPointerAbi, NativeMatmulLayouts,
    NativeMatmulOperandLayout, NativeOutputInitialization, QuantizedBufferAbi, RenderedC,
};
use crate::{GgmlType, QuantizedBufferDesc, ScheduleItem, Shape, VectorPlan};
use std::{collections::BTreeMap, fs};

#[cfg(test)]
use std::path::PathBuf;

const MAGIC: &[u8; 4] = b"RGRC";
const VERSION: u8 = 1;
const MAX_BYTES: usize = 128 << 20;
const MAX_ENTRIES: usize = 1 << 20;
const MAX_STRING_BYTES: usize = 64 << 20;

struct Writer(Vec<u8>);

impl Writer {
    fn new() -> Self {
        Self(Vec::new())
    }

    fn bytes(&mut self, bytes: &[u8]) -> Option<()> {
        (self.0.len().checked_add(bytes.len())? <= MAX_BYTES).then_some(())?;
        self.0.try_reserve(bytes.len()).ok()?;
        self.0.extend_from_slice(bytes);
        Some(())
    }

    fn u8(&mut self, value: u8) -> Option<()> {
        self.bytes(&[value])
    }

    fn bool(&mut self, value: bool) -> Option<()> {
        self.u8(u8::from(value))
    }

    fn u32(&mut self, value: u32) -> Option<()> {
        self.bytes(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> Option<()> {
        self.bytes(&value.to_le_bytes())
    }

    fn usize(&mut self, value: usize) -> Option<()> {
        self.u64(u64::try_from(value).ok()?)
    }

    fn count(&mut self, value: usize) -> Option<()> {
        (value <= MAX_ENTRIES).then_some(())?;
        self.u32(u32::try_from(value).ok()?)
    }

    fn string(&mut self, value: &str) -> Option<()> {
        (value.len() <= MAX_STRING_BYTES).then_some(())?;
        self.u32(u32::try_from(value.len()).ok()?)?;
        self.bytes(value.as_bytes())
    }

    fn blob(&mut self, value: &[u8]) -> Option<()> {
        (value.len() <= MAX_STRING_BYTES).then_some(())?;
        self.u32(u32::try_from(value.len()).ok()?)?;
        self.bytes(value)
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let end = self.position.checked_add(count)?;
        let bytes = self.bytes.get(self.position..end)?;
        self.position = end;
        Some(bytes)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn bool(&mut self) -> Option<bool> {
        match self.u8()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn usize(&mut self) -> Option<usize> {
        usize::try_from(self.u64()?).ok()
    }

    fn count(&mut self) -> Option<usize> {
        let count = usize::try_from(self.u32()?).ok()?;
        (count <= MAX_ENTRIES).then_some(count)
    }

    fn string(&mut self) -> Option<String> {
        let count = usize::try_from(self.u32()?).ok()?;
        (count <= MAX_STRING_BYTES).then_some(())?;
        std::str::from_utf8(self.take(count)?)
            .ok()
            .map(str::to_owned)
    }

    fn blob(&mut self) -> Option<&'a [u8]> {
        let count = usize::try_from(self.u32()?).ok()?;
        (count <= MAX_STRING_BYTES).then_some(())?;
        self.take(count)
    }

    fn done(&self) -> bool {
        self.position == self.bytes.len()
    }
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn write_output_initialization(
    writer: &mut Writer,
    initialization: NativeOutputInitialization,
) -> Option<()> {
    writer.u8(match initialization {
        NativeOutputInitialization::NeedsZero => 0,
        NativeOutputInitialization::FullyOverwritten => 1,
    })
}

fn read_output_initialization(reader: &mut Reader<'_>) -> Option<NativeOutputInitialization> {
    match reader.u8()? {
        0 => Some(NativeOutputInitialization::NeedsZero),
        1 => Some(NativeOutputInitialization::FullyOverwritten),
        _ => None,
    }
}

fn write_layout(writer: &mut Writer, layout: &NativeScheduleLayout) -> Option<()> {
    match layout.matmul {
        None => writer.u8(0)?,
        Some(layouts) => {
            writer.u8(1)?;
            writer.u8(u8::from(
                layouts.lhs == NativeMatmulOperandLayout::Transpose2d,
            ))?;
            writer.u8(u8::from(
                layouts.rhs == NativeMatmulOperandLayout::Transpose2d,
            ))?;
        }
    }
    writer.count(layout.retained_matmul_sources.len())?;
    for (source, retained) in &layout.retained_matmul_sources {
        writer.u64(*source)?;
        writer.u64(*retained)?;
    }
    match layout.elided_output_source {
        Some(source) => {
            writer.u8(1)?;
            writer.u64(source)
        }
        None => writer.u8(0),
    }
}

fn read_layout(reader: &mut Reader<'_>) -> Option<NativeScheduleLayout> {
    let matmul = match reader.u8()? {
        0 => None,
        1 => Some(NativeMatmulLayouts {
            lhs: match reader.u8()? {
                0 => NativeMatmulOperandLayout::Dense,
                1 => NativeMatmulOperandLayout::Transpose2d,
                _ => return None,
            },
            rhs: match reader.u8()? {
                0 => NativeMatmulOperandLayout::Dense,
                1 => NativeMatmulOperandLayout::Transpose2d,
                _ => return None,
            },
        }),
        _ => return None,
    };
    let mut retained_matmul_sources = BTreeMap::new();
    for _ in 0..reader.count()? {
        if retained_matmul_sources
            .insert(reader.u64()?, reader.u64()?)
            .is_some()
        {
            return None;
        }
    }
    let elided_output_source = match reader.u8()? {
        0 => None,
        1 => Some(reader.u64()?),
        _ => return None,
    };
    Some(NativeScheduleLayout {
        matmul,
        retained_matmul_sources,
        elided_output_source,
    })
}

fn write_vector(writer: &mut Writer, vector: &VectorPlan) -> Option<()> {
    writer.usize(vector.lanes)?;
    writer.bool(vector.enabled)?;
    writer.string(&vector.reason)
}

fn read_vector(reader: &mut Reader<'_>) -> Option<VectorPlan> {
    Some(VectorPlan {
        lanes: reader.usize()?,
        enabled: reader.bool()?,
        reason: reader.string()?,
    })
}

fn write_shape(writer: &mut Writer, shape: &Shape) -> Option<()> {
    writer.count(shape.rank())?;
    for dimension in shape.dims() {
        writer.usize(*dimension)?;
    }
    Some(())
}

fn read_shape(reader: &mut Reader<'_>) -> Option<Shape> {
    let count = reader.count()?;
    let mut dimensions = Vec::new();
    dimensions.try_reserve_exact(count).ok()?;
    for _ in 0..count {
        dimensions.push(reader.usize()?);
    }
    Some(Shape::new(dimensions))
}

fn write_quantized_desc(writer: &mut Writer, desc: &QuantizedBufferDesc) -> Option<()> {
    writer.u32(desc.ggml_type.raw())?;
    write_shape(writer, &desc.logical_shape)?;
    writer.usize(desc.block_elements)?;
    writer.usize(desc.block_bytes)?;
    writer.usize(desc.bytes)?;
    writer.usize(desc.alignment)?;
    writer.u64(desc.identity)
}

fn read_quantized_desc(reader: &mut Reader<'_>) -> Option<QuantizedBufferDesc> {
    let desc = QuantizedBufferDesc {
        ggml_type: GgmlType::from_raw(reader.u32()?)?,
        logical_shape: read_shape(reader)?,
        block_elements: reader.usize()?,
        block_bytes: reader.usize()?,
        bytes: reader.usize()?,
        alignment: reader.usize()?,
        identity: reader.u64()?,
    };
    desc.validate_metadata().ok()?;
    Some(desc)
}

fn write_abi(writer: &mut Writer, abi: &KernelAbi) -> Option<()> {
    writer.u32(abi.version)?;
    writer.count(abi.buffers.len())?;
    for buffer in &abi.buffers {
        writer.u64(buffer.id)?;
        writer.u8(crate::uop::artifact::dtype_tag(buffer.dtype))?;
        writer.usize(buffer.elements)?;
        writer.bool(buffer.mutable)?;
    }
    writer.count(abi.quantized_buffers.len())?;
    for buffer in &abi.quantized_buffers {
        writer.u64(buffer.id)?;
        write_quantized_desc(writer, &buffer.desc)?;
    }
    writer.count(abi.pointer_order.len())?;
    for pointer in &abi.pointer_order {
        match pointer {
            KernelPointerAbi::Dense(ordinal) => {
                writer.u8(0)?;
                writer.usize(*ordinal)?;
            }
            KernelPointerAbi::Quantized(ordinal) => {
                writer.u8(1)?;
                writer.usize(*ordinal)?;
            }
        }
    }
    writer.usize(abi.symbol_count)
}

fn read_abi(reader: &mut Reader<'_>) -> Option<KernelAbi> {
    let version = reader.u32()?;
    let buffer_count = reader.count()?;
    let mut buffers = Vec::new();
    buffers.try_reserve_exact(buffer_count).ok()?;
    for _ in 0..buffer_count {
        buffers.push(BufferAbi {
            id: reader.u64()?,
            dtype: crate::uop::artifact::dtype(reader.u8()?).ok()?,
            elements: reader.usize()?,
            mutable: reader.bool()?,
        });
    }
    let quantized_count = reader.count()?;
    let mut quantized_buffers = Vec::new();
    quantized_buffers.try_reserve_exact(quantized_count).ok()?;
    for _ in 0..quantized_count {
        quantized_buffers.push(QuantizedBufferAbi {
            id: reader.u64()?,
            desc: read_quantized_desc(reader)?,
        });
    }
    let pointer_count = reader.count()?;
    let mut pointer_order = Vec::new();
    pointer_order.try_reserve_exact(pointer_count).ok()?;
    for _ in 0..pointer_count {
        let tag = reader.u8()?;
        let ordinal = reader.usize()?;
        pointer_order.push(match tag {
            0 => KernelPointerAbi::Dense(ordinal),
            1 => KernelPointerAbi::Quantized(ordinal),
            _ => return None,
        });
    }
    Some(KernelAbi {
        version,
        buffers,
        quantized_buffers,
        pointer_order,
        symbol_count: reader.usize()?,
    })
}

fn write_rendered(writer: &mut Writer, rendered: &RenderedC) -> Option<()> {
    writer.string(&rendered.source)?;
    writer.count(rendered.source_map.len())?;
    for (node, line) in &rendered.source_map {
        writer.usize(*node)?;
        writer.usize(*line)?;
    }
    write_abi(writer, &rendered.abi)?;
    writer.string(&rendered.cache_key)
}

fn read_rendered(reader: &mut Reader<'_>) -> Option<RenderedC> {
    let source = reader.string()?;
    let source_map_count = reader.count()?;
    let mut source_map = BTreeMap::new();
    for _ in 0..source_map_count {
        if source_map
            .insert(reader.usize()?, reader.usize()?)
            .is_some()
        {
            return None;
        }
    }
    Some(RenderedC {
        source,
        source_map,
        abi: read_abi(reader)?,
        cache_key: reader.string()?,
    })
}

fn write_module(writer: &mut Writer, module: &RenderedScheduleModule) -> Option<()> {
    writer.count(module.entries.len())?;
    for entry in &module.entries {
        writer.count(entry.logical_indices.len())?;
        for logical in &entry.logical_indices {
            writer.usize(*logical)?;
        }
        writer.count(entry.native_layouts.len())?;
        for layout in &entry.native_layouts {
            write_layout(writer, layout)?;
        }
        write_vector(writer, &entry.vector)?;
        write_rendered(writer, &entry.rendered)?;
        writer.string(&entry.native_cache_key)?;
        write_output_initialization(writer, entry.output_initialization)?;
    }
    writer.count(module.zero_domains.len())?;
    for entry in &module.zero_domains {
        writer.usize(entry.logical_index)?;
    }
    Some(())
}

fn read_module(reader: &mut Reader<'_>) -> Option<RenderedScheduleModule> {
    let entry_count = reader.count()?;
    let mut entries = Vec::new();
    entries.try_reserve_exact(entry_count).ok()?;
    for _ in 0..entry_count {
        let logical_count = reader.count()?;
        let mut logical_indices = Vec::new();
        logical_indices.try_reserve_exact(logical_count).ok()?;
        for _ in 0..logical_count {
            logical_indices.push(reader.usize()?);
        }
        let layout_count = reader.count()?;
        let mut native_layouts = Vec::new();
        native_layouts.try_reserve_exact(layout_count).ok()?;
        for _ in 0..layout_count {
            native_layouts.push(read_layout(reader)?);
        }
        entries.push(RenderedScheduleEntry {
            logical_indices,
            native_layouts,
            vector: read_vector(reader)?,
            rendered: read_rendered(reader)?,
            native_cache_key: reader.string()?,
            output_initialization: read_output_initialization(reader)?,
        });
    }
    let zero_count = reader.count()?;
    let mut zero_domains = Vec::new();
    zero_domains.try_reserve_exact(zero_count).ok()?;
    for _ in 0..zero_count {
        zero_domains.push(PreparedZeroDomainEntry {
            logical_index: reader.usize()?,
        });
    }
    Some(RenderedScheduleModule {
        entries,
        zero_domains,
        render_wall_time: std::time::Duration::ZERO,
    })
}

fn recipe(
    backend: &CpuJitBackend,
    program_index: usize,
    program_count: usize,
    items: &[ScheduleItem],
    layouts: &[NativeScheduleLayout],
    store_groups: &[NativeStoreGroup],
) -> Option<Vec<u8>> {
    if items.len() != layouts.len() {
        return None;
    }
    let mut writer = Writer::new();
    writer.blob(b"rustgrad-native-render-recipe-v2")?;
    writer.string(&crate::cpu_jit::native_render_capsule_environment())?;
    writer.bool(backend.vectorized)?;
    writer.usize(program_index)?;
    writer.usize(program_count)?;
    writer.count(items.len())?;
    for (item, layout) in items.iter().zip(layouts) {
        // The schedule cache key is the stable, versioned identity of the
        // complete item, including its canonical kernel artifact, bindings,
        // descriptors, and boundary. Retain that identity rather than the
        // potentially very large artifact bytes a second time: large
        // training programs otherwise exceed the bounded recipe codec before
        // their rendered capsule can be admitted.
        writer.u64(item.cache_key)?;
        write_layout(&mut writer, layout)?;
        write_output_initialization(
            &mut writer,
            crate::cpu_jit::native_output_initialization(&item.kernel),
        )?;
    }
    writer.count(store_groups.len())?;
    for group in store_groups {
        writer.count(group.members.len())?;
        for member in &group.members {
            writer.usize(member.logical_index)?;
            writer.u64(member.output_buffer)?;
        }
        // The private fused root is derived solely from the already encoded
        // member kernels and this ordered membership. Bind its exact renderer
        // ABI without requiring that derived multi-store DAG to have a second
        // ordinary schedule-artifact representation.
        write_abi(
            &mut writer,
            &crate::cpu_jit::native_store_group_abi(&group.kernel).ok()?,
        )?;
        write_output_initialization(&mut writer, group.output_initialization)?;
    }
    Some(writer.0)
}

fn authenticates_rendered_payload(rendered: &RenderedC) -> bool {
    let source_line_count = rendered.source.lines().count();
    if rendered.source.is_empty()
        || rendered.cache_key.len() != 16
        || !rendered
            .cache_key
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || rendered.abi.version != ABI_VERSION
        || rendered.abi.symbol_count != 0
        || rendered.abi.pointer_order.len()
            != rendered
                .abi
                .buffers
                .len()
                .saturating_add(rendered.abi.quantized_buffers.len())
        || rendered
            .source_map
            .values()
            .any(|line| *line == 0 || *line > source_line_count)
    {
        return false;
    }
    let dense_ids = rendered
        .abi
        .buffers
        .iter()
        .map(|buffer| buffer.id)
        .collect::<std::collections::BTreeSet<_>>();
    let quantized_ids = rendered
        .abi
        .quantized_buffers
        .iter()
        .map(|buffer| buffer.id)
        .collect::<std::collections::BTreeSet<_>>();
    if dense_ids.len() != rendered.abi.buffers.len()
        || quantized_ids.len() != rendered.abi.quantized_buffers.len()
    {
        return false;
    }
    let mut dense_ordinals = std::collections::BTreeSet::new();
    let mut quantized_ordinals = std::collections::BTreeSet::new();
    for pointer in &rendered.abi.pointer_order {
        match *pointer {
            KernelPointerAbi::Dense(ordinal)
                if ordinal < rendered.abi.buffers.len() && dense_ordinals.insert(ordinal) => {}
            KernelPointerAbi::Quantized(ordinal)
                if ordinal < rendered.abi.quantized_buffers.len()
                    && quantized_ordinals.insert(ordinal) => {}
            _ => return false,
        }
    }
    dense_ordinals.len() == rendered.abi.buffers.len()
        && quantized_ordinals.len() == rendered.abi.quantized_buffers.len()
}

fn authenticates_item_abi(item: &ScheduleItem, abi: &KernelAbi) -> bool {
    let dense_inputs = item
        .ordered_inputs()
        .iter()
        .map(|binding| {
            Some(BufferAbi {
                id: binding.desc.id,
                dtype: binding.desc.dtype,
                elements: binding
                    .desc
                    .bytes
                    .checked_div(binding.desc.dtype.itemsize())?,
                mutable: false,
            })
        })
        .collect::<Option<Vec<_>>>();
    let Some(mut dense_inputs) = dense_inputs else {
        return false;
    };
    let Ok(output_elements) = item.primary_output().shape.numel() else {
        return false;
    };
    dense_inputs.push(BufferAbi {
        id: item.primary_output().id,
        dtype: item.primary_output().dtype,
        elements: output_elements,
        mutable: true,
    });
    let quantized = item
        .ordered_quantized_inputs()
        .iter()
        .map(|binding| QuantizedBufferAbi {
            id: binding.input_node.index() as u64,
            desc: binding.desc.clone(),
        })
        .collect::<Vec<_>>();
    let mut inputs = item
        .ordered_inputs()
        .iter()
        .enumerate()
        .map(|(ordinal, binding)| (binding.abi_index, KernelPointerAbi::Dense(ordinal)))
        .chain(
            item.ordered_quantized_inputs()
                .iter()
                .enumerate()
                .map(|(ordinal, binding)| {
                    (binding.abi_index, KernelPointerAbi::Quantized(ordinal))
                }),
        )
        .collect::<Vec<_>>();
    inputs.sort_by_key(|(index, _)| *index);
    let mut pointer_order = inputs
        .into_iter()
        .map(|(_, pointer)| pointer)
        .collect::<Vec<_>>();
    pointer_order.push(KernelPointerAbi::Dense(dense_inputs.len() - 1));
    abi.buffers == dense_inputs
        && abi.quantized_buffers == quantized
        && abi.pointer_order == pointer_order
}

fn authenticates_store_group_abi(
    group: &NativeStoreGroup,
    items: &[ScheduleItem],
    abi: &KernelAbi,
) -> bool {
    if !crate::cpu_jit::native_store_group_abi(&group.kernel).is_ok_and(|expected| expected == *abi)
        || !abi.quantized_buffers.is_empty()
        || abi.pointer_order
            != (0..abi.buffers.len())
                .map(KernelPointerAbi::Dense)
                .collect::<Vec<_>>()
        || abi.buffers.len() < group.members.len()
    {
        return false;
    }
    abi.buffers[abi.buffers.len() - group.members.len()..]
        .iter()
        .zip(&group.members)
        .all(|(buffer, member)| {
            items.get(member.logical_index).is_some_and(|item| {
                item.primary_output().shape.numel().ok() == Some(buffer.elements)
                    && item.primary_output().id == buffer.id
                    && buffer.id == member.output_buffer
                    && item.primary_output().dtype == crate::DType::F32
                    && buffer.dtype == crate::DType::F32
                    && buffer.mutable
            })
        })
        && abi.buffers[..abi.buffers.len() - group.members.len()]
            .iter()
            .all(|buffer| !buffer.mutable)
}

fn authenticate_module(
    backend: &CpuJitBackend,
    module: &RenderedScheduleModule,
    items: &[ScheduleItem],
    layouts: &[NativeScheduleLayout],
    store_groups: &[NativeStoreGroup],
) -> bool {
    if items.len() != layouts.len() {
        return false;
    }
    let mut group_anchors = std::collections::BTreeSet::new();
    let mut group_members = std::collections::BTreeSet::new();
    for group in store_groups {
        let Some(anchor) = group.members.last().map(|member| member.logical_index) else {
            return false;
        };
        if group.members.len() < 2
            || !group_anchors.insert(anchor)
            || group
                .members
                .windows(2)
                .any(|pair| pair[1].logical_index <= pair[0].logical_index)
        {
            return false;
        }
        for member in &group.members {
            let Some((item, layout)) = items
                .get(member.logical_index)
                .zip(layouts.get(member.logical_index))
            else {
                return false;
            };
            if !group_members.insert(member.logical_index)
                || item.primary_output().id != member.output_buffer
                || item.primary_output().shape.numel().ok() == Some(0)
                || validate_native_layout(item, layout).is_err()
            {
                return false;
            }
        }
    }
    let groups = store_groups
        .iter()
        .filter_map(|group| {
            group
                .members
                .last()
                .map(|member| (member.logical_index, group))
        })
        .collect::<BTreeMap<_, _>>();
    let grouped = group_members;
    let mut expected_zero = Vec::new();
    for (index, (item, layout)) in items.iter().zip(layouts).enumerate() {
        let Ok(elements) = item.primary_output().shape.numel() else {
            return false;
        };
        if elements == 0 {
            if grouped.contains(&index)
                || validate_native_layout(item, layout).is_err()
                || backend.validate_zero_domain_schedule_item(item).is_err()
            {
                return false;
            }
            expected_zero.push(index);
        }
    }
    if module
        .zero_domains
        .iter()
        .map(|entry| entry.logical_index)
        .ne(expected_zero)
    {
        return false;
    }
    let mut entries = module.entries.iter();
    for (index, (item, layout)) in items.iter().zip(layouts).enumerate() {
        let Ok(elements) = item.primary_output().shape.numel() else {
            return false;
        };
        if elements == 0 || grouped.contains(&index) && !groups.contains_key(&index) {
            continue;
        }
        let Some(entry) = entries.next() else {
            return false;
        };
        if !authenticates_rendered_payload(&entry.rendered) {
            return false;
        }
        if let Some(group) = groups.get(&index) {
            let logical_indices = group
                .members
                .iter()
                .map(|member| member.logical_index)
                .collect::<Vec<_>>();
            let native_layouts = logical_indices
                .iter()
                .filter_map(|logical| layouts.get(*logical).cloned())
                .collect::<Vec<_>>();
            let ordered_members = logical_indices
                .iter()
                .filter_map(|logical| items.get(*logical))
                .map(|item| format!("{:016x}", item.cache_key))
                .collect::<Vec<_>>()
                .join("-");
            let ordered_outputs = group
                .members
                .iter()
                .map(|member| format!("{:016x}", member.output_buffer))
                .collect::<Vec<_>>()
                .join("-");
            if entry.logical_indices != logical_indices
                || entry.native_layouts != native_layouts
                || entry.vector
                    != (VectorPlan {
                        lanes: 1,
                        enabled: false,
                        reason: "private native store group is scalar".into(),
                    })
                || entry.output_initialization != group.output_initialization
                || crate::cpu_jit::native_store_group_cache_key(&entry.rendered.source)
                    != entry.rendered.cache_key
                || !authenticates_store_group_abi(group, items, &entry.rendered.abi)
                || entry.native_cache_key
                    != format!(
                        "{}-native-store-group-{ordered_members}-{ordered_outputs}",
                        entry.rendered.cache_key
                    )
            {
                return false;
            }
        } else {
            let expected_vector = if backend.vectorized {
                match crate::CpuJit::vector_plan(&item.kernel) {
                    Ok(vector) => vector,
                    Err(_) => return false,
                }
            } else {
                VectorPlan {
                    lanes: 1,
                    enabled: false,
                    reason: "scalar policy disabled vector lanes".into(),
                }
            };
            if entry.logical_indices != [index]
                || entry.native_layouts != [layout.clone()]
                || entry.vector != expected_vector
                || entry.output_initialization
                    != crate::cpu_jit::native_output_initialization(&item.kernel)
                || entry.native_cache_key
                    != format!(
                        "{}-schedule-{:016x}",
                        entry.rendered.cache_key, item.cache_key
                    )
                || validate_native_layout(item, layout).is_err()
                || backend
                    .validate_rendered_schedule_item(item, &entry.rendered)
                    .is_err()
                || !crate::cpu_jit::native_rendered_cache_key(
                    &item.kernel,
                    backend.vectorized,
                    &entry.rendered.source,
                )
                .is_ok_and(|expected| expected == entry.rendered.cache_key)
                || !authenticates_item_abi(item, &entry.rendered.abi)
            {
                return false;
            }
        }
    }
    entries.next().is_none()
}

fn decode(bytes: &[u8], expected_recipe: &[u8]) -> Option<RenderedScheduleModule> {
    if bytes.len() < MAGIC.len() + 1 + 8 || bytes.len() > MAX_BYTES {
        return None;
    }
    let (payload, checksum_bytes) = bytes.split_at(bytes.len().checked_sub(8)?);
    let expected_checksum = u64::from_le_bytes(checksum_bytes.try_into().ok()?);
    if checksum(payload) != expected_checksum {
        return None;
    }
    let mut reader = Reader::new(payload);
    (reader.take(4)? == MAGIC).then_some(())?;
    (reader.u8()? == VERSION).then_some(())?;
    (reader.blob()? == expected_recipe).then_some(())?;
    let module = read_module(&mut reader)?;
    reader.done().then_some(module)
}

fn encode(recipe: &[u8], module: &RenderedScheduleModule) -> Option<Vec<u8>> {
    let mut writer = Writer::new();
    writer.bytes(MAGIC)?;
    writer.u8(VERSION)?;
    writer.blob(recipe)?;
    write_module(&mut writer, module)?;
    let checksum = checksum(&writer.0);
    writer.u64(checksum)?;
    Some(writer.0)
}

pub(super) struct CapsuleRecipe {
    bytes: Vec<u8>,
    identity: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CapsuleLoadStatus {
    Hit,
    RecipeUnavailable,
    FileUnavailable,
    DecodeRejected,
    AuthenticationRejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CapsuleStoreStatus {
    NotAttempted,
    RecipeUnavailable,
    Stored,
    EncodeRejected,
    FilesystemRejected,
}

pub(super) fn capsule_recipe(
    backend: &CpuJitBackend,
    program_index: usize,
    program_count: usize,
    items: &[ScheduleItem],
    layouts: &[NativeScheduleLayout],
    store_groups: &[NativeStoreGroup],
) -> Option<CapsuleRecipe> {
    let bytes = recipe(
        backend,
        program_index,
        program_count,
        items,
        layouts,
        store_groups,
    )?;
    Some(CapsuleRecipe {
        identity: checksum(&bytes),
        bytes,
    })
}

pub(super) fn load_capsule(
    backend: &CpuJitBackend,
    recipe: &CapsuleRecipe,
    items: &[ScheduleItem],
    layouts: &[NativeScheduleLayout],
    store_groups: &[NativeStoreGroup],
) -> Result<RenderedScheduleModule, CapsuleLoadStatus> {
    let path = crate::cpu_jit::native_render_capsule_path(recipe.identity);
    let bytes = crate::file_io::read_file_bytes_bounded(path, MAX_BYTES)
        .map_err(|_| CapsuleLoadStatus::FileUnavailable)?;
    let module = decode(&bytes, &recipe.bytes).ok_or(CapsuleLoadStatus::DecodeRejected)?;
    authenticate_module(backend, &module, items, layouts, store_groups)
        .then_some(module)
        .ok_or(CapsuleLoadStatus::AuthenticationRejected)
}

pub(super) fn store_capsule(
    recipe: &CapsuleRecipe,
    module: &RenderedScheduleModule,
) -> CapsuleStoreStatus {
    let Some(bytes) = encode(&recipe.bytes, module) else {
        return CapsuleStoreStatus::EncodeRejected;
    };
    let path = crate::cpu_jit::native_render_capsule_path(recipe.identity);
    let Some(parent) = path.parent() else {
        return CapsuleStoreStatus::FilesystemRejected;
    };
    if fs::create_dir_all(parent).is_err() {
        return CapsuleStoreStatus::FilesystemRejected;
    }
    match crate::file_io::replace_file_bytes_atomically(path, &bytes) {
        Ok(()) => CapsuleStoreStatus::Stored,
        Err(_) => CapsuleStoreStatus::FilesystemRejected,
    }
}

#[cfg(test)]
pub(super) fn remove_capsule(recipe: &CapsuleRecipe) {
    let _ = fs::remove_file(crate::cpu_jit::native_render_capsule_path(recipe.identity));
}

#[cfg(test)]
pub(super) fn capsule_path(recipe: &CapsuleRecipe) -> PathBuf {
    crate::cpu_jit::native_render_capsule_path(recipe.identity)
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) enum CapsuleMutation {
    LogicalInventory,
    Layout,
    Vector,
    Abi,
    SourceMap,
    Source,
    RenderedCacheKey,
    NativeCacheKey,
    OutputInitialization,
    StoreGroupInputId,
    StoreGroupInputDType,
    StoreGroupInputElements,
}

#[cfg(test)]
pub(super) fn mutate_capsule_with_valid_checksum(
    recipe: &CapsuleRecipe,
    mutation: CapsuleMutation,
) {
    let path = capsule_path(recipe);
    let bytes = fs::read(&path).unwrap();
    let mut module = decode(&bytes, &recipe.bytes).unwrap();
    let store_group_mutation = matches!(
        mutation,
        CapsuleMutation::StoreGroupInputId
            | CapsuleMutation::StoreGroupInputDType
            | CapsuleMutation::StoreGroupInputElements
    );
    let entry = if store_group_mutation {
        module
            .entries
            .iter_mut()
            .find(|entry| entry.logical_indices.len() > 1)
            .unwrap()
    } else {
        module.entries.first_mut().unwrap()
    };
    match mutation {
        CapsuleMutation::LogicalInventory => entry.logical_indices.push(usize::MAX),
        CapsuleMutation::Layout => entry.native_layouts[0].elided_output_source = Some(u64::MAX),
        CapsuleMutation::Vector => entry.vector.lanes = entry.vector.lanes.saturating_add(1),
        CapsuleMutation::Abi => entry.rendered.abi.symbol_count = 1,
        CapsuleMutation::SourceMap => {
            entry.rendered.source_map.insert(usize::MAX, usize::MAX);
        }
        CapsuleMutation::Source => entry.rendered.source.push_str("/* stale */\n"),
        CapsuleMutation::RenderedCacheKey => {
            let replacement = if entry.rendered.cache_key.starts_with('0') {
                "1"
            } else {
                "0"
            };
            entry.rendered.cache_key.replace_range(..1, replacement);
        }
        CapsuleMutation::NativeCacheKey => entry.native_cache_key.push_str("-stale"),
        CapsuleMutation::OutputInitialization => {
            entry.output_initialization = match entry.output_initialization {
                NativeOutputInitialization::NeedsZero => {
                    NativeOutputInitialization::FullyOverwritten
                }
                NativeOutputInitialization::FullyOverwritten => {
                    NativeOutputInitialization::NeedsZero
                }
            }
        }
        CapsuleMutation::StoreGroupInputId => {
            entry
                .rendered
                .abi
                .buffers
                .iter_mut()
                .find(|buffer| !buffer.mutable)
                .unwrap()
                .id ^= 1;
        }
        CapsuleMutation::StoreGroupInputDType => {
            entry
                .rendered
                .abi
                .buffers
                .iter_mut()
                .find(|buffer| !buffer.mutable)
                .unwrap()
                .dtype = crate::DType::F64;
        }
        CapsuleMutation::StoreGroupInputElements => {
            let input = entry
                .rendered
                .abi
                .buffers
                .iter_mut()
                .find(|buffer| !buffer.mutable)
                .unwrap();
            input.elements = input.elements.saturating_add(1);
        }
    }
    fs::write(path, encode(&recipe.bytes, &module).unwrap()).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendered_payload_accepts_vector_source_lines_and_rejects_invalid_lines() {
        let mut graph = crate::Graph::new();
        let input = graph.input_dtype("source_map_input", [8], crate::DType::F32);
        let output = graph.square(input).unwrap();
        let kernel = crate::lower_graph_elementwise(&graph, output).unwrap();
        let mut rendered = crate::CpuJit::render_vectorized(&kernel).unwrap();
        assert!(!rendered.source_map.is_empty());
        assert!(
            rendered
                .source_map
                .values()
                .all(|line| { *line != 0 && *line <= rendered.source.lines().count() })
        );
        assert!(authenticates_rendered_payload(&rendered));

        *rendered.source_map.values_mut().next().unwrap() = 0;
        assert!(!authenticates_rendered_payload(&rendered));
    }

    #[test]
    fn capsule_checksum_rejects_every_single_byte_mutation() {
        let module = RenderedScheduleModule {
            entries: Vec::new(),
            zero_domains: vec![PreparedZeroDomainEntry { logical_index: 3 }],
            render_wall_time: std::time::Duration::from_secs(1),
        };
        let recipe = b"bounded-recipe";
        let bytes = encode(recipe, &module).unwrap();
        let decoded = decode(&bytes, recipe).unwrap();
        assert_eq!(decoded.zero_domains[0].logical_index, 3);
        assert!(decoded.render_wall_time.is_zero());
        for index in 0..bytes.len() {
            let mut malformed = bytes.clone();
            malformed[index] ^= 1;
            assert!(decode(&malformed, recipe).is_none());
        }
        assert!(decode(&bytes, b"different-recipe").is_none());
    }
}
