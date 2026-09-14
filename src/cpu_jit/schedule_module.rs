use super::{
    C11_COMPILER_COMMAND, C11_COMPILER_FLAGS, C11LocalHelper, COMPILE_SEQUENCE, JitError,
    JitScheduleModuleLoad, NativeCompilerProcessKind, NativeCompilerProcessObservation, RenderedC,
    cache_dir, native_cache_key, run_compiler,
};
use std::{
    fs,
    ops::Range,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::Ordering,
    thread,
    time::{Duration, Instant},
};

// The protected Transformer's 682-entry main suffix dominates cold preparation
// while its 252-entry next-largest suffix does not. Keep small modules as one
// translation unit and split only that measured oversized class.
const CHUNKED_MODULE_ENTRY_THRESHOLD: u64 = 512;
const C11_TRANSLATION_UNIT_FLAGS: &[&str] = &[
    "-std=c11",
    "-O2",
    "-ffp-contract=off",
    "-fPIC",
    "-Werror",
    "-c",
];
const C11_LINK_FLAGS: &[&str] = &["-shared", "-Werror"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeScheduleModuleBuildMode {
    Combined,
    Chunked,
}

impl NativeScheduleModuleBuildMode {
    pub(crate) const fn for_unique_rendered_entry_count(entry_count: u64) -> Option<Self> {
        match entry_count {
            0 => None,
            1..=CHUNKED_MODULE_ENTRY_THRESHOLD => Some(Self::Combined),
            _ => Some(Self::Chunked),
        }
    }

    pub(crate) const fn process_inventory(self) -> (u64, u64, u64) {
        match self {
            Self::Combined => (1, 0, 0),
            Self::Chunked => (0, 2, 1),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ScheduleModuleChunk {
    ordinal: usize,
    entries: Range<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ScheduleModuleBuildPlan {
    mode: NativeScheduleModuleBuildMode,
    chunks: Box<[ScheduleModuleChunk]>,
}

impl ScheduleModuleBuildPlan {
    fn new(rendered: &[RenderedC]) -> Result<Self, JitError> {
        if rendered.is_empty() {
            return Err(JitError::InvalidBuffer(
                "empty native schedule module has no build plan".into(),
            ));
        }
        let entry_count = u64::try_from(rendered.len())
            .map_err(|_| JitError::Io("native schedule entry count overflowed".into()))?;
        let mode = NativeScheduleModuleBuildMode::for_unique_rendered_entry_count(entry_count)
            .expect("nonempty native schedule module has a build mode");
        let ranges = match mode {
            NativeScheduleModuleBuildMode::Combined => std::iter::once(0..rendered.len()).collect(),
            NativeScheduleModuleBuildMode::Chunked => {
                let split = balanced_split(rendered)?;
                vec![0..split, split..rendered.len()]
            }
        };
        Ok(Self {
            mode,
            chunks: ranges
                .into_iter()
                .enumerate()
                .map(|(ordinal, entries)| ScheduleModuleChunk { ordinal, entries })
                .collect(),
        })
    }
}

fn balanced_split(rendered: &[RenderedC]) -> Result<usize, JitError> {
    let total = rendered.iter().try_fold(0usize, |total, entry| {
        total
            .checked_add(entry.source.len())
            .ok_or_else(|| JitError::Io("native schedule source size overflowed".into()))
    })?;
    let mut prefix = 0usize;
    let mut best = (usize::MAX, 1usize);
    for split in 1..rendered.len() {
        prefix = prefix
            .checked_add(rendered[split - 1].source.len())
            .ok_or_else(|| JitError::Io("native schedule source size overflowed".into()))?;
        let suffix = total
            .checked_sub(prefix)
            .ok_or_else(|| JitError::Io("native schedule source size underflowed".into()))?;
        let imbalance = prefix.abs_diff(suffix);
        if imbalance < best.0 {
            best = (imbalance, split);
        }
    }
    Ok(best.1)
}

type TranslationUnitThreadResult =
    thread::Result<Result<NativeCompilerProcessObservation, JitError>>;

struct JoinedTranslationUnit {
    ordinal: usize,
    result: TranslationUnitThreadResult,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ScheduleModuleBuildEvidence {
    combined_compile_link_count: usize,
    object_compile_count: usize,
    linker_invocation_count: usize,
    compiler_process_wall_time: Duration,
    compiler_process_total_wall_time: Duration,
    linker_process_wall_time: Duration,
}

impl ScheduleModuleBuildEvidence {
    fn new(
        mode: NativeScheduleModuleBuildMode,
        observations: &[NativeCompilerProcessObservation],
    ) -> Result<Self, JitError> {
        let combined_compile_link_count = observations
            .iter()
            .filter(|observation| observation.kind == NativeCompilerProcessKind::Combined)
            .count();
        let object_compile_count = observations
            .iter()
            .filter(|observation| matches!(observation.kind, NativeCompilerProcessKind::Object(_)))
            .count();
        let object_ordinals_are_ordered = observations
            .iter()
            .filter_map(|observation| match observation.kind {
                NativeCompilerProcessKind::Object(ordinal) => Some(ordinal),
                NativeCompilerProcessKind::Combined | NativeCompilerProcessKind::Link => None,
            })
            .eq(0..object_compile_count);
        let linker = observations
            .iter()
            .filter(|observation| observation.kind == NativeCompilerProcessKind::Link)
            .collect::<Vec<_>>();
        let process_inventory = (
            u64::try_from(combined_compile_link_count)
                .map_err(|_| JitError::Io("combined compiler count overflowed".into()))?,
            u64::try_from(object_compile_count)
                .map_err(|_| JitError::Io("object compiler count overflowed".into()))?,
            u64::try_from(linker.len())
                .map_err(|_| JitError::Io("linker count overflowed".into()))?,
        );
        if mode.process_inventory() != process_inventory {
            return Err(JitError::Io(
                "native schedule module compiler evidence differs".into(),
            ));
        }
        if !object_ordinals_are_ordered {
            return Err(JitError::Io(
                "native schedule object compiler ordinal differs".into(),
            ));
        }
        let compiler_process_intervals = observations
            .iter()
            .map(|observation| (observation.process_started, observation.process_finished))
            .collect::<Vec<_>>();
        let (compiler_process_total_wall_time, compiler_process_wall_time) =
            process_wall_times(&compiler_process_intervals)?;
        Ok(Self {
            combined_compile_link_count,
            object_compile_count,
            linker_invocation_count: linker.len(),
            compiler_process_wall_time,
            compiler_process_total_wall_time,
            linker_process_wall_time: linker
                .first()
                .map(|linker| {
                    linker
                        .process_finished
                        .duration_since(linker.process_started)
                })
                .unwrap_or(Duration::ZERO),
        })
    }
}

fn process_wall_times(intervals: &[(Instant, Instant)]) -> Result<(Duration, Duration), JitError> {
    let total = intervals
        .iter()
        .try_fold(Duration::ZERO, |total, (start, end)| {
            total
                .checked_add(end.duration_since(*start))
                .ok_or_else(|| JitError::Io("compiler-process duration overflowed".into()))
        })?;
    let mut sorted = intervals.to_vec();
    sorted.sort_by_key(|(start, _)| *start);
    let mut union = Duration::ZERO;
    let mut current: Option<(Instant, Instant)> = None;
    for (start, end) in sorted {
        match current {
            Some((range_start, range_end)) if start <= range_end => {
                current = Some((range_start, range_end.max(end)));
            }
            Some((range_start, range_end)) => {
                union = union
                    .checked_add(range_end.duration_since(range_start))
                    .ok_or_else(|| JitError::Io("compiler-process duration overflowed".into()))?;
                current = Some((start, end));
            }
            None => current = Some((start, end)),
        }
    }
    if let Some((start, end)) = current {
        union = union
            .checked_add(end.duration_since(start))
            .ok_or_else(|| JitError::Io("compiler-process duration overflowed".into()))?;
    }
    Ok((total, union))
}

struct TemporaryFiles(Vec<PathBuf>);

impl Drop for TemporaryFiles {
    fn drop(&mut self) {
        for path in &self.0 {
            let _ = fs::remove_file(path);
        }
    }
}

pub(super) fn schedule_module_entry_symbol(index: usize) -> String {
    format!("rustgrad_schedule_entry_{index:08x}")
}

pub(super) fn schedule_module_local_symbol(index: usize, symbol: &str) -> String {
    format!("rustgrad_schedule_{index:08x}_{symbol}")
}

fn render_schedule_module_chunk(
    rendered: &[RenderedC],
    entries: Range<usize>,
    include_dispatcher: bool,
) -> String {
    let mut source = String::new();
    for index in entries {
        let entry = &rendered[index];
        source.push_str(&format!(
            "#define rustgrad_kernel {}\n",
            schedule_module_entry_symbol(index)
        ));
        for helper in C11LocalHelper::ALL {
            let symbol = helper.name();
            source.push_str(&format!(
                "#define {symbol} {}\n",
                schedule_module_local_symbol(index, symbol)
            ));
        }
        source.push_str(&format!(
            "#line 1 \"{}\"\n",
            schedule_module_entry_symbol(index)
        ));
        source.push_str(&entry.source);
        source.push_str("\n#undef rustgrad_kernel\n");
        for helper in C11LocalHelper::ALL {
            let symbol = helper.name();
            source.push_str(&format!("#undef {symbol}\n"));
        }
    }
    if include_dispatcher {
        source.push_str(
            "#include <stddef.h>\n#include <stdint.h>\n#include <string.h>\n\
typedef int (*rg_schedule_entry_fn)(void **,const int64_t *,uint64_t *);\n\
typedef struct { size_t extent; size_t divisor; size_t stride; size_t reversed; } rg_schedule_axis;\n\
enum { RG_SCHEDULE_COPY=0, RG_SCHEDULE_AFFINE=1 };\n\
typedef struct { size_t kind; const void *source; void *target; size_t width; size_t elements; size_t offset; const rg_schedule_axis *axes; size_t axis_count; } rg_schedule_action;\n\
typedef struct { rg_schedule_entry_fn entry; void **buffers; const int64_t *symbols; const rg_schedule_action *actions; size_t action_count; } rg_schedule_call;\n\
static int rg_schedule_apply(const rg_schedule_action *action){\n\
  if(action->kind==RG_SCHEDULE_COPY){\n\
    if(action->elements!=0)memmove(action->target,action->source,action->elements*action->width);\n\
    return 0;\n\
  }\n\
  if(action->kind!=RG_SCHEDULE_AFFINE)return 4;\n\
  for(size_t logical=0;logical<action->elements;logical++){\n\
    size_t physical=action->offset;\n\
    for(size_t axis=0;axis<action->axis_count;axis++){\n\
      const rg_schedule_axis *term=&action->axes[axis];\n\
      size_t coordinate=(logical/term->divisor)%term->extent;\n\
      if(term->reversed)coordinate=term->extent-1-coordinate;\n\
      physical+=coordinate*term->stride;\n\
    }\n\
    memcpy((unsigned char*)action->target+logical*action->width,(const unsigned char*)action->source+physical*action->width,action->width);\n\
  }\n\
  return 0;\n\
}\n\
int rustgrad_schedule_dispatch(rg_schedule_call *calls,size_t count,uint64_t *failure){\n\
  for(size_t i=0;i<count;i++){\n\
    for(size_t action=0;action<calls[i].action_count;action++){\n\
      int status=rg_schedule_apply(&calls[i].actions[action]);\n\
      if(status!=0){failure[0]=(uint64_t)i;failure[1]=UINT64_MAX;failure[2]=0;return status;}\n\
    }\n\
    uint64_t local[2]={UINT64_MAX,0};\n\
    int status=calls[i].entry(calls[i].buffers,calls[i].symbols,local);\n\
    if(status!=0){failure[0]=(uint64_t)i;failure[1]=local[0];failure[2]=local[1];return status;}\n\
  }\n\
  return 0;\n\
}\n",
        );
    }
    source
}

#[cfg(test)]
pub(super) fn render_schedule_module_source(rendered: &[RenderedC]) -> String {
    render_schedule_module_chunk(rendered, 0..rendered.len(), true)
}

pub(super) fn schedule_module_manifest(rendered: &[RenderedC]) -> String {
    let mut manifest = format!("rustgrad-c11-schedule-module-v4\u{1f}{}", rendered.len());
    for helper in C11LocalHelper::ALL {
        manifest.push('\u{1f}');
        manifest.push_str(helper.name());
    }
    for (index, entry) in rendered.iter().enumerate() {
        manifest.push('\u{1f}');
        manifest.push_str(&schedule_module_entry_symbol(index));
        manifest.push('\u{1f}');
        manifest.push_str(&entry.cache_key);
    }
    manifest
}

pub(crate) fn schedule_module_cache_key(rendered: &[RenderedC]) -> String {
    native_cache_key("schedule-module-v4", &schedule_module_manifest(rendered))
}

fn compile_translation_unit(
    ordinal: usize,
    source: &Path,
    object: &Path,
) -> Result<NativeCompilerProcessObservation, JitError> {
    run_checked_compiler(
        Command::new(C11_COMPILER_COMMAND)
            .args(C11_TRANSLATION_UNIT_FLAGS)
            .arg("-o")
            .arg(object)
            .arg(source),
        NativeCompilerProcessKind::Object(ordinal),
    )
}

fn compile_combined_module(
    source: &Path,
    temporary: &Path,
) -> Result<NativeCompilerProcessObservation, JitError> {
    run_checked_compiler(
        Command::new(C11_COMPILER_COMMAND)
            .args(C11_COMPILER_FLAGS)
            .arg("-o")
            .arg(temporary)
            .arg(source),
        NativeCompilerProcessKind::Combined,
    )
}

fn run_checked_compiler(
    command: &mut Command,
    kind: NativeCompilerProcessKind,
) -> Result<NativeCompilerProcessObservation, JitError> {
    let (output, observation) =
        run_compiler(command, kind).map_err(|error| JitError::Compiler {
            status: None,
            stderr: error.to_string(),
        })?;
    if !output.status.success() {
        return Err(JitError::Compiler {
            status: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(8192)
                .collect(),
        });
    }
    Ok(observation)
}

fn compile_translation_units(
    sources: &[PathBuf],
    objects: &[PathBuf],
) -> Result<Vec<NativeCompilerProcessObservation>, JitError> {
    if sources.len() != objects.len() || sources.is_empty() {
        return Err(JitError::Io(
            "native schedule translation-unit inventory differs".into(),
        ));
    }
    debug_assert!(sources.len() > 1);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(sources.len());
        let mut spawn_failure = None;
        for (ordinal, (source, object)) in sources.iter().zip(objects).enumerate() {
            match thread::Builder::new()
                .name(format!("rustgrad-native-chunk-{ordinal}"))
                .spawn_scoped(scope, move || {
                    compile_translation_unit(ordinal, source, object)
                }) {
                Ok(handle) => handles.push((ordinal, handle)),
                Err(error) => {
                    spawn_failure = Some((
                        ordinal,
                        JitError::Io(format!(
                            "native schedule compiler worker spawn failed: {error}"
                        )),
                    ));
                    break;
                }
            }
        }
        let mut joined = handles
            .into_iter()
            .map(|(ordinal, handle)| JoinedTranslationUnit {
                ordinal,
                result: handle.join(),
            })
            .collect::<Vec<_>>();
        if let Some((ordinal, error)) = spawn_failure {
            joined.push(JoinedTranslationUnit {
                ordinal,
                result: Ok(Err(error)),
            });
        }
        canonical_translation_unit_results(joined)
    })
}

fn canonical_translation_unit_results(
    mut joined: Vec<JoinedTranslationUnit>,
) -> Result<Vec<NativeCompilerProcessObservation>, JitError> {
    joined.sort_by_key(|joined| joined.ordinal);
    let mut observations = Vec::with_capacity(joined.len());
    for joined in joined {
        match joined.result {
            Ok(Ok(observation)) => observations.push(observation),
            Ok(Err(error)) => return Err(error),
            Err(_) => {
                return Err(JitError::Io(format!(
                    "native schedule compiler worker {} panicked",
                    joined.ordinal
                )));
            }
        }
    }
    Ok(observations)
}

fn link_schedule_module(
    objects: &[PathBuf],
    temporary: &Path,
) -> Result<NativeCompilerProcessObservation, JitError> {
    run_checked_compiler(
        Command::new(C11_COMPILER_COMMAND)
            .args(C11_LINK_FLAGS)
            .arg("-o")
            .arg(temporary)
            .args(objects),
        NativeCompilerProcessKind::Link,
    )
}

/// Compiles helper-isolated C translation units concurrently and links them
/// into the one content-addressed schedule library consumed by replay.
pub(super) fn compile_cached_schedule_module_under_gate(
    rendered: &[RenderedC],
) -> Result<(PathBuf, JitScheduleModuleLoad), JitError> {
    let cache_key = schedule_module_cache_key(rendered);
    let directory = cache_dir();
    fs::create_dir_all(&directory).map_err(|error| JitError::Io(error.to_string()))?;
    let library = directory.join(format!(
        "{cache_key}.{}",
        if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        }
    ));
    match fs::symlink_metadata(&library) {
        Ok(metadata) if metadata.file_type().is_file() => {
            return Ok((
                library,
                JitScheduleModuleLoad {
                    durable_cache_hit: true,
                    combined_compile_link_count: 0,
                    object_compile_count: 0,
                    linker_invocation_count: 0,
                    compiler_invocation_count: 0,
                    compiler_process_wall_time: Duration::ZERO,
                    compiler_process_total_wall_time: Duration::ZERO,
                    linker_process_wall_time: Duration::ZERO,
                    module_load_wall_time: Duration::ZERO,
                    compiler_process_observations: Vec::new(),
                },
            ));
        }
        Ok(_) => {
            return Err(JitError::Io(format!(
                "CPU JIT cache entry is not a regular file: {}",
                library.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(JitError::Io(error.to_string())),
    }

    let plan = ScheduleModuleBuildPlan::new(rendered)?;
    let sequence = COMPILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let stem = format!(".{cache_key}-{}-{sequence}", std::process::id());
    let sources = plan
        .chunks
        .iter()
        .map(|chunk| directory.join(format!("{stem}-{:02}.c", chunk.ordinal)))
        .collect::<Vec<_>>();
    let objects = plan
        .chunks
        .iter()
        .map(|chunk| directory.join(format!("{stem}-{:02}.o", chunk.ordinal)))
        .collect::<Vec<_>>();
    let temporary = directory.join(format!("{stem}.tmp"));
    let _temporary_files = TemporaryFiles(
        sources
            .iter()
            .chain(&objects)
            .chain(std::iter::once(&temporary))
            .cloned()
            .collect(),
    );
    for (chunk, source) in plan.chunks.iter().zip(&sources) {
        fs::write(
            source,
            render_schedule_module_chunk(
                rendered,
                chunk.entries.clone(),
                chunk.ordinal + 1 == plan.chunks.len(),
            ),
        )
        .map_err(|error| JitError::Io(error.to_string()))?;
    }
    let observations = if plan.chunks.len() == 1 {
        vec![compile_combined_module(&sources[0], &temporary)?]
    } else {
        let mut observations = compile_translation_units(&sources, &objects)?;
        observations.push(link_schedule_module(&objects, &temporary)?);
        observations
    };
    let evidence = ScheduleModuleBuildEvidence::new(plan.mode, &observations)?;
    fs::File::open(&temporary)
        .and_then(|file| file.sync_all())
        .map_err(|error| JitError::Io(error.to_string()))?;
    match fs::rename(&temporary, &library) {
        Ok(()) => {}
        Err(error) => match fs::symlink_metadata(&library) {
            Ok(metadata) if metadata.file_type().is_file() => {}
            _ => return Err(JitError::Io(error.to_string())),
        },
    }
    let compiler_invocation_count = evidence
        .combined_compile_link_count
        .checked_add(evidence.object_compile_count)
        .and_then(|count| count.checked_add(evidence.linker_invocation_count))
        .ok_or_else(|| JitError::Io("compiler invocation count overflowed".into()))?;
    Ok((
        library,
        JitScheduleModuleLoad {
            durable_cache_hit: false,
            combined_compile_link_count: evidence.combined_compile_link_count,
            object_compile_count: evidence.object_compile_count,
            linker_invocation_count: evidence.linker_invocation_count,
            compiler_invocation_count,
            compiler_process_wall_time: evidence.compiler_process_wall_time,
            compiler_process_total_wall_time: evidence.compiler_process_total_wall_time,
            linker_process_wall_time: evidence.linker_process_wall_time,
            module_load_wall_time: Duration::ZERO,
            compiler_process_observations: observations,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_jit::{ABI_VERSION, KernelAbi};
    use std::collections::BTreeMap;

    fn rendered(source_bytes: usize, key: &str) -> RenderedC {
        RenderedC {
            source: "x".repeat(source_bytes),
            source_map: BTreeMap::new(),
            abi: KernelAbi {
                version: ABI_VERSION,
                buffers: Vec::new(),
                quantized_buffers: Vec::new(),
                pointer_order: Vec::new(),
                symbol_count: 0,
            },
            cache_key: key.into(),
        }
    }

    fn observation(
        kind: NativeCompilerProcessKind,
        start: Instant,
        nanos: u64,
    ) -> NativeCompilerProcessObservation {
        NativeCompilerProcessObservation {
            kind,
            permit_requested: start,
            process_started: start,
            process_finished: start + Duration::from_nanos(nanos),
        }
    }

    #[test]
    fn oversized_schedule_plan_is_balanced_contiguous_and_deterministic() {
        let rendered = (0..513)
            .map(|index| rendered(if index < 300 { 2 } else { 3 }, &index.to_string()))
            .collect::<Vec<_>>();
        let plan = ScheduleModuleBuildPlan::new(&rendered).unwrap();
        assert_eq!(plan, ScheduleModuleBuildPlan::new(&rendered).unwrap());
        assert_eq!(plan.chunks.len(), 2);
        assert_eq!(plan.chunks[0].entries.start, 0);
        assert_eq!(plan.chunks[0].entries.end, plan.chunks[1].entries.start);
        assert_eq!(plan.chunks[1].entries.end, rendered.len());
        let weights = plan
            .chunks
            .iter()
            .map(|chunk| {
                rendered[chunk.entries.clone()]
                    .iter()
                    .map(|entry| entry.source.len())
                    .sum::<usize>()
            })
            .collect::<Vec<_>>();
        assert!(weights[0].abs_diff(weights[1]) <= 3);
    }

    #[test]
    fn maintained_native_program_sizes_split_only_the_dominant_main_suffix() {
        let chunks = [682, 146, 252, 38].map(|entries| {
            ScheduleModuleBuildPlan::new(
                &(0..entries)
                    .map(|index| rendered(1, &index.to_string()))
                    .collect::<Vec<_>>(),
            )
            .unwrap()
            .chunks
            .len()
        });
        assert_eq!(chunks, [2, 1, 1, 1]);
        assert_eq!(chunks.into_iter().sum::<usize>(), 5);
        assert_eq!(
            NativeScheduleModuleBuildMode::for_unique_rendered_entry_count(512),
            Some(NativeScheduleModuleBuildMode::Combined)
        );
        assert_eq!(
            NativeScheduleModuleBuildMode::for_unique_rendered_entry_count(513),
            Some(NativeScheduleModuleBuildMode::Chunked)
        );
    }

    #[test]
    fn chunk_sources_preserve_global_symbols_and_define_one_dispatcher() {
        let rendered = (0..513)
            .map(|index| rendered(1, &index.to_string()))
            .collect::<Vec<_>>();
        let plan = ScheduleModuleBuildPlan::new(&rendered).unwrap();
        let sources = plan
            .chunks
            .iter()
            .map(|chunk| {
                render_schedule_module_chunk(
                    &rendered,
                    chunk.entries.clone(),
                    chunk.ordinal + 1 == plan.chunks.len(),
                )
            })
            .collect::<Vec<_>>();
        assert!(sources[0].contains(&schedule_module_entry_symbol(0)));
        assert!(!sources[0].contains("int rustgrad_schedule_dispatch("));
        assert!(sources[1].contains(&schedule_module_entry_symbol(512)));
        assert!(sources[1].contains("int rustgrad_schedule_dispatch("));
    }

    #[test]
    fn direct_build_evidence_is_one_combined_process() {
        let start = Instant::now();
        let observations = [observation(NativeCompilerProcessKind::Combined, start, 2)];
        let evidence = ScheduleModuleBuildEvidence::new(
            NativeScheduleModuleBuildMode::Combined,
            &observations,
        )
        .unwrap();
        assert_eq!(evidence.combined_compile_link_count, 1);
        assert_eq!(evidence.object_compile_count, 0);
        assert_eq!(evidence.linker_invocation_count, 0);
        assert_eq!(
            evidence.compiler_process_total_wall_time,
            evidence.compiler_process_wall_time
        );
        assert_eq!(evidence.linker_process_wall_time, Duration::ZERO);
        assert!(
            ScheduleModuleBuildEvidence::new(NativeScheduleModuleBuildMode::Chunked, &observations)
                .is_err()
        );
    }

    #[test]
    fn temporary_build_files_are_removed_by_owned_cleanup() {
        let sequence = COMPILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rustgrad-native-schedule-cleanup-{}-{sequence}",
            std::process::id()
        ));
        fs::write(&path, b"temporary").unwrap();
        {
            let _temporary_files = TemporaryFiles(vec![path.clone()]);
            assert!(path.is_file());
        }
        assert!(!path.exists());
    }

    #[test]
    fn build_evidence_distinguishes_cumulative_and_effective_compiler_wall() {
        let start = Instant::now();
        let observations = [
            observation(NativeCompilerProcessKind::Object(0), start, 3),
            observation(NativeCompilerProcessKind::Object(1), start, 2),
            observation(
                NativeCompilerProcessKind::Link,
                start + Duration::from_nanos(3),
                1,
            ),
        ];
        let evidence =
            ScheduleModuleBuildEvidence::new(NativeScheduleModuleBuildMode::Chunked, &observations)
                .unwrap();
        assert_eq!(evidence.combined_compile_link_count, 0);
        assert_eq!(evidence.object_compile_count, 2);
        assert_eq!(evidence.linker_invocation_count, 1);
        assert_eq!(
            evidence.compiler_process_total_wall_time,
            Duration::from_nanos(6)
        );
        assert_eq!(evidence.compiler_process_wall_time, Duration::from_nanos(4));
        assert_eq!(evidence.linker_process_wall_time, Duration::from_nanos(1));
    }

    #[test]
    fn worker_results_choose_lowest_ordinal_failure_after_collection() {
        let start = Instant::now();
        let compile_error = JitError::Compiler {
            status: Some(9),
            stderr: "first".into(),
        };
        let result = canonical_translation_unit_results(vec![
            JoinedTranslationUnit {
                ordinal: 2,
                result: Err(Box::new("later panic") as Box<dyn std::any::Any + Send>),
            },
            JoinedTranslationUnit {
                ordinal: 1,
                result: Ok(Err(JitError::Io("later error".into()))),
            },
            JoinedTranslationUnit {
                ordinal: 0,
                result: Ok(Err(compile_error.clone())),
            },
            JoinedTranslationUnit {
                ordinal: 3,
                result: Ok(Ok(observation(
                    NativeCompilerProcessKind::Object(3),
                    start,
                    1,
                ))),
            },
        ]);
        assert_eq!(result, Err(compile_error));

        let result = canonical_translation_unit_results(vec![
            JoinedTranslationUnit {
                ordinal: 1,
                result: Ok(Err(JitError::Io("later error".into()))),
            },
            JoinedTranslationUnit {
                ordinal: 0,
                result: Err(Box::new("first panic") as Box<dyn std::any::Any + Send>),
            },
        ]);
        assert_eq!(
            result,
            Err(JitError::Io(
                "native schedule compiler worker 0 panicked".into()
            ))
        );
    }
}
