use super::{
    JitBackendError, JitKernel, JitScheduleDispatcher, NativeScheduleLayout, RenderedScheduleEntry,
    RenderedScheduleModule,
};
use crate::{CpuJitBackend, ScheduleItem, VectorPlan};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};

#[derive(Clone)]
pub(crate) struct NativeStoreGroupMember {
    pub(crate) logical_index: usize,
    pub(crate) output_buffer: u64,
}

#[derive(Clone)]
pub(crate) struct NativeStoreGroup {
    pub(crate) members: Vec<NativeStoreGroupMember>,
    pub(crate) kernel: crate::UOp,
    pub(crate) output_initialization: crate::cpu_jit::NativeOutputInitialization,
}

pub(crate) struct PreparedNativeStoreGroupMember {
    pub(crate) logical_index: usize,
    pub(crate) output_buffer: u64,
    pub(crate) schedule_cache_key: u64,
    pub(crate) layout: NativeScheduleLayout,
}

pub(crate) struct PreparedNativeStoreGroup {
    pub(crate) members: Vec<PreparedNativeStoreGroupMember>,
    pub(super) cache_hit: bool,
    pub(super) output_initialization: crate::cpu_jit::NativeOutputInitialization,
    pub(super) kernel: Arc<JitKernel>,
    pub(super) dispatcher: Arc<JitScheduleDispatcher>,
}

pub(crate) fn render_schedule_module_entries(
    backend: &CpuJitBackend,
    items: &[ScheduleItem],
    layouts: Vec<NativeScheduleLayout>,
    store_groups: &[NativeStoreGroup],
) -> Result<RenderedScheduleModule, JitBackendError> {
    if items.len() != layouts.len() {
        return Err(JitBackendError::Binding(
            "native schedule module layout count mismatch".into(),
        ));
    }
    let started = Instant::now();
    let mut groups_by_anchor = BTreeMap::new();
    let mut grouped_members = BTreeSet::new();
    for group in store_groups {
        if group.members.len() < 2 {
            return Err(JitBackendError::Binding(
                "native store group requires multiple members".into(),
            ));
        }
        let anchor = group
            .members
            .last()
            .map(|member| member.logical_index)
            .ok_or_else(|| JitBackendError::Binding("empty native store group".into()))?;
        if group
            .members
            .windows(2)
            .any(|pair| pair[1].logical_index <= pair[0].logical_index)
            || group
                .members
                .iter()
                .any(|member| member.logical_index >= items.len())
            || groups_by_anchor.insert(anchor, group).is_some()
            || group
                .members
                .iter()
                .any(|member| !grouped_members.insert(member.logical_index))
        {
            return Err(JitBackendError::Binding(
                "native store group inventory mismatch".into(),
            ));
        }
        for member in &group.members {
            let item = &items[member.logical_index];
            if item.primary_output().id != member.output_buffer {
                return Err(JitBackendError::Binding(
                    "native store group output identity mismatch".into(),
                ));
            }
            super::validate_native_layout(item, &layouts[member.logical_index])?;
        }
    }
    let mut entries = Vec::with_capacity(items.len());
    for (index, (item, layout)) in items.iter().zip(layouts).enumerate() {
        if grouped_members.contains(&index) {
            let Some(group) = groups_by_anchor.get(&index) else {
                continue;
            };
            let (rendered, output_initialization) =
                crate::cpu_jit::render_native_store_group(&group.kernel)
                    .map_err(|error| JitBackendError::Unsupported(error.to_string()))?;
            if output_initialization != group.output_initialization {
                return Err(JitBackendError::Binding(
                    "native store group output policy changed".into(),
                ));
            }
            let ordered_members = group
                .members
                .iter()
                .map(|member| format!("{:016x}", items[member.logical_index].cache_key))
                .collect::<Vec<_>>()
                .join("-");
            let ordered_outputs = group
                .members
                .iter()
                .map(|member| format!("{:016x}", member.output_buffer))
                .collect::<Vec<_>>()
                .join("-");
            entries.push(RenderedScheduleEntry {
                logical_indices: group
                    .members
                    .iter()
                    .map(|member| member.logical_index)
                    .collect(),
                vector: VectorPlan {
                    lanes: 1,
                    enabled: false,
                    reason: "private native store group is scalar".into(),
                },
                native_cache_key: format!(
                    "{}-native-store-group-{ordered_members}-{ordered_outputs}",
                    rendered.cache_key,
                ),
                output_initialization,
                rendered,
            });
        } else {
            super::validate_native_layout(item, &layout)?;
            let (vector, rendered, _) = backend.render_schedule_kernel(item, &layout)?;
            backend.validate_rendered_schedule_item(item, &rendered)?;
            let native_cache_key =
                format!("{}-schedule-{:016x}", rendered.cache_key, item.cache_key);
            entries.push(RenderedScheduleEntry {
                logical_indices: vec![index],
                vector,
                rendered,
                native_cache_key,
                output_initialization: crate::cpu_jit::native_output_initialization(&item.kernel),
            });
        }
    }
    Ok(RenderedScheduleModule {
        entries,
        render_wall_time: started.elapsed(),
    })
}
