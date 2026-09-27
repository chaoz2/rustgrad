#!/usr/bin/env python3
"""Validate and compare same-runner native Transformer training evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import statistics
import sys
from typing import Any

FORMAT_VERSION = 2
EVIDENCE_KIND = "native_cpu_transformer_training_same_runner_comparison"
SHA_RE = re.compile(r"[0-9a-f]{40}")
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
ORDER = (
    ("baseline", 1),
    ("candidate", 1),
    ("candidate", 2),
    ("baseline", 2),
)
SAMPLE_PARTITIONS = (
    "total_wall_time",
    "executor_wall_time",
    "native_dispatcher_wall_time",
    "executor_host_wall_time",
    "recurrent_overhead_wall_time",
)
PROGRAM_FIELDS = (
    "capture_identity",
    "native_identity",
    "vectorized",
    "native_item_count",
    "executed_native_item_count",
    "module_dispatch_count",
    "module_dispatched_native_item_count",
    "schedule_cache_keys",
    "traffic",
)
SCOREBOARD_PROGRAM_FIELDS = (
    "capture_identity",
    "native_identity",
    "vectorized",
    "execution_plan_identity",
    "logical_schedule_item_count",
    "recurrent_state_count",
    "peak_logical_temporary_allocation_count",
    "peak_logical_temporary_bytes",
    "native_item_count",
    "cache_hit_count",
    "cache_miss_count",
    "rendered_entry_count",
    "rendered_source_bytes",
    "loaded_module_count",
    "referenced_module_count",
    "unique_rendered_entry_count",
    "shared_prefix_entry_count",
    "shared_prefix_source_bytes",
    "shared_prefix_source_program_index",
    "shared_prefix_source_native_identity",
    "durable_artifact_cache_hit_count",
    "durable_artifact_cache_miss_count",
    "compiler_invocation_count",
    "combined_compile_link_count",
    "object_compile_count",
    "linker_invocation_count",
    "dispatch_segmentation",
)
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_BINARY_BYTES = 512 * 1024 * 1024
MEASUREMENT_TARGETS = {
    "steady-replay": "example:compiled_transformer_train_resume",
    "warm-resume": "example:compiled_transformer_scale_evidence",
}


class EvidenceError(ValueError):
    pass


def load_preparation_evidence_module() -> Any:
    path = pathlib.Path(__file__).resolve().with_name(
        "native_training_preparation_evidence.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rustgrad_native_training_preparation_evidence", path
    )
    require(spec is not None and spec.loader is not None, "preparation validator is absent")
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


PREPARATION_EVIDENCE = load_preparation_evidence_module()


def load_larger_evidence_module() -> Any:
    path = pathlib.Path(__file__).resolve().with_name(
        "check-larger-native-transformer-evidence.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rustgrad_larger_native_transformer_evidence", path
    )
    require(spec is not None and spec.loader is not None, "larger evidence validator is absent")
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


LARGER_EVIDENCE = load_larger_evidence_module()


def exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def exact_str(value: Any, label: str) -> str:
    require(type(value) is str and value != "", f"{label} must be a nonempty string")
    return value


def read_bounded(path: pathlib.Path) -> bytes:
    require(path.is_absolute(), f"evidence path must be absolute: {path}")
    require(not path.is_symlink() and path.is_file(), f"evidence must be a regular file: {path}")
    size = path.stat().st_size
    require(0 < size <= MAX_FILE_BYTES, f"evidence file size is invalid: {path}")
    return path.read_bytes()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"invalid JSON in {label}: {error}") from error
    require(type(value) is dict, f"{label} must contain a JSON object")
    return value


def load_json(path: pathlib.Path) -> tuple[dict[str, Any], bytes]:
    raw = read_bounded(path)
    value = decode_json(raw, path.name)
    return value, raw


def duration_ns(value: Any, label: str) -> int:
    try:
        return PREPARATION_EVIDENCE.duration_ns(value, label)
    except PREPARATION_EVIDENCE.PreparationEvidenceError as error:
        raise EvidenceError(str(error)) from error


def parse_provenance(
    raw: bytes,
    expected_sha: str,
    binary_digest: str,
    measurement_mode: str,
) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceError("provenance is not UTF-8") from error
    sections: dict[str, list[str]] = {"root": []}
    current = "root"
    for line in text.splitlines():
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            require(current not in sections, f"duplicate provenance section {current}")
            sections[current] = []
        else:
            sections[current].append(line)
    require(set(sections) == {"root", "cpu_hardware", "rustc", "cargo", "c_compiler"}, "provenance sections differ")

    def key_values(lines: list[str], label: str) -> dict[str, str]:
        values: dict[str, str] = {}
        for line in lines:
            if line == "":
                continue
            require("=" in line, f"{label} contains a non-key line")
            key, value = line.split("=", 1)
            require(key and value and key not in values, f"{label} key is empty or duplicated")
            values[key] = value
        return values

    root = key_values(sections["root"], "provenance root")
    common_root = {
        "schema_version",
        "git_sha",
        "prebuilt_binary_source_sha",
        "prebuilt_binary_sha256",
        "cargo_profile",
        "temporary_cache",
        "runner_os",
        "runner_arch",
        "runner_image_os",
        "runner_image_version",
        "cpu_hardware_policy",
    }
    mode_root = {
        "steady-replay": {
            "steady_measurement_file",
            "steady_measurement_cache_scope",
            "steady_measurement_warmup_windows",
            "steady_measurement_measured_windows",
        },
        "warm-resume": {
            "cargo_build_jobs",
            "workflow_timeout_minutes",
            "workload",
            "warm_resume",
            "timing_policy",
        },
    }
    expected_root = common_root | mode_root[measurement_mode]
    require(set(root) == expected_root, "comparison provenance root fields differ")
    require(root["schema_version"] == "1", "provenance schema differs")
    require(root["git_sha"] == expected_sha, "provenance revision differs")
    require(root["prebuilt_binary_source_sha"] == expected_sha, "prebuilt source revision differs")
    require(root["prebuilt_binary_sha256"] == binary_digest, "prebuilt binary digest differs")
    require(root["cargo_profile"] == "release", "provenance profile differs")
    require(root["temporary_cache"] == "fresh_sha_scoped", "cache provenance differs")
    if measurement_mode == "steady-replay":
        require(root["steady_measurement_file"] == "native-cpu-training-steady-replays.json", "steady filename differs")
        require(root["steady_measurement_cache_scope"] == "same_process_warm_cache", "steady cache scope differs")
        require(root["steady_measurement_warmup_windows"] == "1", "warmup window count differs")
        require(root["steady_measurement_measured_windows"] == "32", "measured window count differs")
    else:
        require(root["cargo_build_jobs"] == "2", "warm build job count differs")
        require(root["workflow_timeout_minutes"] == "75", "warm workflow timeout differs")
        require(
            root["workload"]
            == "batch4_time8_vocab16_embedding8_heads2_ff32_blocks2_replays6",
            "warm workload differs",
        )
        require(
            root["warm_resume"]
            == "portable_rgab_replay3_to6_fresh_executor_same_temporary_cache",
            "warm resume policy differs",
        )
        require(root["timing_policy"] == "observational_no_threshold", "warm timing policy differs")
    require(root["cpu_hardware_policy"] == "required_normalized_lscpu_v1", "CPU provenance policy differs")
    cpu = key_values(sections["cpu_hardware"], "CPU provenance")
    require(
        set(cpu)
        == {
            "cpu_model_name",
            "cpu_socket_count",
            "cpu_cores_per_socket",
            "cpu_threads_per_core",
            "cpu_logical_count",
            "cpu_vendor_id",
            "cpu_family",
            "cpu_model",
            "cpu_stepping",
            "cpu_numa_node_count",
        },
        "CPU provenance fields differ",
    )
    for section in ("rustc", "cargo", "c_compiler"):
        while sections[section] and sections[section][0] == "":
            sections[section].pop(0)
        while sections[section] and sections[section][-1] == "":
            sections[section].pop()
        require(any(line for line in sections[section]), f"{section} provenance is empty")
    stable = {
        "runner": {key: root[key] for key in ("runner_os", "runner_arch", "runner_image_os", "runner_image_version")},
        "cpu_hardware": cpu,
        "rustc": sections["rustc"],
        "cargo": sections["cargo"],
        "c_compiler": sections["c_compiler"],
    }
    return stable


def project_program(value: Any, label: str) -> dict[str, Any]:
    require(type(value) is dict, f"{label} must be an object")
    require(set(PROGRAM_FIELDS).issubset(value), f"{label} fields are absent")
    result = dict(value)
    exact_int(result["capture_identity"], f"{label}.capture_identity")
    exact_int(result["native_identity"], f"{label}.native_identity")
    require(type(result["vectorized"]) is bool and result["vectorized"], f"{label} must be vectorized")
    native = exact_int(result["native_item_count"], f"{label}.native_item_count", minimum=1)
    executed = exact_int(result["executed_native_item_count"], f"{label}.executed_native_item_count", minimum=1)
    dispatched = exact_int(result["module_dispatched_native_item_count"], f"{label}.module_dispatched_native_item_count", minimum=1)
    require(executed <= native and dispatched == executed, f"{label} native inventory differs")
    exact_int(result["module_dispatch_count"], f"{label}.module_dispatch_count", minimum=1)
    require(type(result["schedule_cache_keys"]) is list and len(result["schedule_cache_keys"]) == native, f"{label} cache inventory differs")
    traffic = result["traffic"]
    require(type(traffic) is dict, f"{label}.traffic must be an object")
    for field in traffic:
        exact_int(traffic[field], f"{label}.traffic.{field}")
    require(traffic.get("external_input_import_count") == 0, f"{label} imported external inputs")
    require(traffic.get("external_input_import_bytes") == 0, f"{label} imported external bytes")
    return result


def validate_sidecar(value: dict[str, Any], expected_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    require(value.get("format_version") == 1, "steady evidence format differs")
    require(value.get("evidence_kind") == "native_cpu_compiled_transformer_steady_replay", "steady evidence kind differs")
    provenance = value.get("provenance")
    require(type(provenance) is dict, "steady provenance is absent")
    require(provenance.get("git_sha") == expected_sha and provenance.get("cargo_profile") == "release", "steady provenance differs")
    expected_counts = {
        "warmup_window_count": 1,
        "warmup_replay_count": 3,
        "measured_window_count": 32,
        "measured_replay_count": 96,
        "validated_replay_count": 99,
        "validated_valid_token_count": 363,
        "measured_valid_token_count": 352,
        "compiled_graph_build_count": 1,
    }
    for field, expected in expected_counts.items():
        require(value.get(field) == expected and type(value.get(field)) is int, f"steady {field} differs")
    dropout_blocks_per_replay = exact_int(
        value.get("dropout_blocks_per_replay"),
        "steady dropout blocks per replay",
        minimum=1,
    )
    preparation = value.get("preparation")
    require(type(preparation) is dict, "steady preparation is absent")
    require(preparation.get("cache_scope") == "same_process_warm_cache", "steady preparation cache scope differs")
    for field, expected in {
        "render_capsule_hit_count": 4,
        "render_capsule_miss_count": 0,
        "local_render_job_count": 0,
        "max_parallel_render_job_count": 0,
        "compiler_process_count": 0,
    }.items():
        require(preparation.get(field) == expected and type(preparation.get(field)) is int, f"steady preparation {field} differs")
    duration_ns(preparation.get("wall_time"), "steady preparation.wall_time")
    schedule = value.get("captured_schedule")
    require(
        type(schedule) is dict
        and schedule.get("base_bits") == 1028443341
        and schedule.get("gamma_bits") == 1056964608
        and schedule.get("milestones") == [1],
        "steady captured rate schedule differs",
    )

    accumulation = project_program(value.get("accumulation_only_program"), "accumulation_only_program")
    commit = project_program(value.get("optimizer_commit_program"), "optimizer_commit_program")
    require(accumulation["capture_identity"] != commit["capture_identity"], "steady program captures alias")
    samples: list[tuple[str, dict[str, Any]]] = []
    for role, expected_count, expected_microbatches in (
        ("accumulation_only_samples", 64, {1, 2}),
        ("optimizer_commit_samples", 32, {3}),
    ):
        entries = value.get(role)
        require(type(entries) is list and len(entries) == expected_count, f"{role} count differs")
        for entry in entries:
            require(type(entry) is dict, f"{role} entry must be an object")
            microbatch = exact_int(entry.get("microbatch_ordinal"), f"{role}.microbatch_ordinal", minimum=1)
            require(microbatch in expected_microbatches, f"{role} microbatch role differs")
            samples.append((role, entry))
    samples.sort(key=lambda item: exact_int(item[1].get("measurement_ordinal"), "measurement_ordinal", minimum=1))
    require(len(samples) == 96, "steady sample inventory differs")
    for index, (_, sample) in enumerate(samples, start=1):
        expected_microbatch = (index - 1) % 3 + 1
        expected_window = (index - 1) // 3 + 1
        require(exact_int(sample.get("measurement_ordinal"), "steady measurement ordinal", minimum=1) == index, "steady measurement order differs")
        require(exact_int(sample.get("window_ordinal"), "steady window ordinal", minimum=1) == expected_window, "steady window order differs")
        require(exact_int(sample.get("microbatch_ordinal"), "steady microbatch ordinal", minimum=1) == expected_microbatch, "steady microbatch order differs")
        require(exact_int(sample.get("replay_step"), "steady replay step", minimum=1) == 6 + index, "steady replay progress differs")
        require(exact_int(sample.get("successful_invocation"), "steady successful invocation", minimum=1) == 3 + index, "steady success progress differs")
        require(exact_int(sample.get("valid_token_count"), "steady valid token count", minimum=1) == (5, 3, 3)[expected_microbatch - 1], "steady token weight differs")
        durations = {field: duration_ns(sample.get(field), f"sample {index}.{field}") for field in SAMPLE_PARTITIONS}
        require(durations["native_dispatcher_wall_time"] + durations["executor_host_wall_time"] == durations["executor_wall_time"], "steady executor partition differs")
        require(durations["executor_wall_time"] + durations["recurrent_overhead_wall_time"] == durations["total_wall_time"], "steady total partition differs")

    checkpoints = {}
    for field, replay, optimizer, accumulation_index in (
        ("starting_checkpoint", 3, 1, 0),
        ("warmup_checkpoint", 6, 2, 0),
        ("final_checkpoint", 102, 34, 0),
    ):
        checkpoint = value.get(field)
        require(type(checkpoint) is dict, f"steady {field} is absent")
        require(exact_int(checkpoint.get("replay_step"), f"steady {field}.replay_step") == replay, f"steady {field} replay differs")
        require(exact_int(checkpoint.get("optimizer_step"), f"steady {field}.optimizer_step") == optimizer, f"steady {field} optimizer progress differs")
        require(exact_int(checkpoint.get("accumulation_index"), f"steady {field}.accumulation_index") == accumulation_index, f"steady {field} accumulation progress differs")
        checkpoints[field] = checkpoint
    require(checkpoints["starting_checkpoint"].get("capture_identity") == commit["capture_identity"], "steady main checkpoint capture differs")
    require(checkpoints["starting_checkpoint"].get("accumulation_capture_identity") == accumulation["capture_identity"], "steady accumulation checkpoint capture differs")
    for field in ("warmup_checkpoint", "final_checkpoint"):
        require(checkpoints[field].get("capture_identity") == checkpoints["starting_checkpoint"].get("capture_identity"), f"steady {field} main capture differs")
        require(checkpoints[field].get("accumulation_capture_identity") == checkpoints["starting_checkpoint"].get("accumulation_capture_identity"), f"steady {field} accumulation capture differs")
    start_dropout = exact_int(checkpoints["starting_checkpoint"].get("dropout_block_counter"), "starting dropout")
    warmup_dropout = exact_int(checkpoints["warmup_checkpoint"].get("dropout_block_counter"), "warmup dropout")
    final_dropout = exact_int(checkpoints["final_checkpoint"].get("dropout_block_counter"), "final dropout")
    require(warmup_dropout - start_dropout == dropout_blocks_per_replay * 3, "steady warmup dropout progress differs")
    require(final_dropout - start_dropout == dropout_blocks_per_replay * 99, "steady dropout progress differs")
    exact_int(value.get("final_checkpoint_byte_count"), "steady final checkpoint byte count", minimum=1)

    stable = copy.deepcopy(value)
    stable["provenance"].pop("git_sha")
    stable["preparation"].pop("wall_time")
    for role in ("accumulation_only_samples", "optimizer_commit_samples"):
        for sample in stable[role]:
            for field in SAMPLE_PARTITIONS:
                sample.pop(field)
    return stable, {"samples": samples}


def scoreboard_program(value: Any, label: str) -> dict[str, Any]:
    require(type(value) is dict, f"scoreboard {label} is absent")
    optional = {
        "recurrent_state_count",
        "shared_prefix_source_program_index",
        "shared_prefix_source_native_identity",
    }
    missing = [field for field in SCOREBOARD_PROGRAM_FIELDS if field not in value and field not in optional]
    require(not missing, f"scoreboard {label} fields are absent: {missing}")
    projected = copy.deepcopy(value)
    projected.pop("preparation_timing", None)
    require(projected["vectorized"] is True, f"scoreboard {label} is not vectorized")
    native = exact_int(projected["native_item_count"], f"scoreboard {label}.native_item_count", minimum=1)
    hits = exact_int(projected["cache_hit_count"], f"scoreboard {label}.cache_hit_count")
    misses = exact_int(projected["cache_miss_count"], f"scoreboard {label}.cache_miss_count")
    require(hits + misses == native, f"scoreboard {label} cache inventory differs")
    return projected


def validate_scoreboard(value: dict[str, Any]) -> dict[str, Any]:
    format_version = exact_int(value.get("format_version"), "scoreboard format")
    require(format_version in (24, 25, 26), "scoreboard format differs")
    require(exact_int(value.get("initial_replay_step"), "scoreboard initial replay") == 0, "scoreboard initial replay differs")
    require(exact_int(value.get("successful_replay_count"), "scoreboard replay count") == 3, "scoreboard replay count differs")
    require(exact_int(value.get("fallback_count"), "scoreboard fallback count") == 0, "scoreboard fallback is nonzero")
    cold_prepare = {
        "prepare_max_parallel_render_job_count": 2,
        "prepare_render_capsule_hit_count": 0,
        "prepare_render_capsule_miss_count": 4,
        "prepare_local_render_job_count": 4,
        "prepare_compiler_process_count": 6,
        "prepare_max_parallel_compiler_process_count": 2,
    }
    for field, expected in cold_prepare.items():
        require(value.get(field) == expected and type(value.get(field)) is int, f"scoreboard cold {field} differs")
    raw_programs = {
        role: value.get(role)
        for role in ("main", "accumulation", "partial_flush", "zero_grad", "evaluation")
    }
    try:
        PREPARATION_EVIDENCE.validate_preparation_finalization(
            value, format_version, raw_programs
        )
    except PREPARATION_EVIDENCE.PreparationEvidenceError as error:
        raise EvidenceError(str(error)) from error
    programs = {
        role: scoreboard_program(raw_programs[role], role)
        for role in ("main", "accumulation", "partial_flush", "zero_grad")
    }
    require(value.get("evaluation") is None, "scoreboard unexpectedly includes evaluation")
    checkpoint = value.get("checkpoint")
    require(type(checkpoint) is dict, "scoreboard checkpoint is absent")
    require(exact_int(checkpoint.get("replay_step"), "scoreboard checkpoint replay") == 3, "scoreboard checkpoint progress differs")
    require(exact_int(checkpoint.get("capture_identity"), "scoreboard checkpoint capture") == programs["main"]["capture_identity"], "scoreboard checkpoint capture differs")
    exact_int(checkpoint.get("byte_count"), "scoreboard checkpoint.byte_count", minimum=1)
    for field in ("main_replay_traffic", "accumulation_replay_traffic"):
        traffic = value.get(field)
        require(type(traffic) is dict, f"scoreboard {field} is absent")
        require(traffic.get("external_input_import_count") == 0 and traffic.get("external_input_import_bytes") == 0, f"scoreboard {field} imported inputs")
        for key, entry in traffic.items():
            exact_int(entry, f"scoreboard {field}.{key}")
    stable = copy.deepcopy(value)
    stable.pop("format_version")
    stable.pop("prepare_finalization", None)
    for field in (
        "compile_wall_time",
        "prepare_wall_time",
        "prepare_runtime_overhead_wall_time",
        "prepare_parallel_module_overlap_wall_time",
        "prepare_parallel_render_overlap_wall_time",
        "prepare_compiler_process_overlap_wall_time",
        "prepare_compiler_critical_tail",
        "main_replay_executor_wall_time",
        "main_replay_native_dispatcher_wall_time",
        "main_replay_executor_host_wall_time",
        "main_replay_recurrent_overhead_wall_time",
        "first_replay_wall_time",
        "steady_replay_total_wall_time",
        "steady_replay_wall_time",
        "steady_microbatches_per_second",
        "measured_peak_host_memory_bytes",
    ):
        stable.pop(field, None)
    for role, program in programs.items():
        stable[role] = program
    stable["checkpoint"].pop("wall_time", None)
    compile_phases = stable.get("compile_phases")
    require(type(compile_phases) is dict, "scoreboard compile phases are absent")
    compile_phases.pop("residual_wall_time", None)
    for phase in ("objective_forward", "autograd", "optimizer_lowering"):
        require(type(compile_phases.get(phase)) is dict, f"scoreboard {phase} compile phase is absent")
        compile_phases[phase].pop("wall_time", None)
    for phase in ("main_capture", "accumulation_capture", "partial_flush", "zero_grad"):
        capture = compile_phases.get(phase)
        require(type(capture) is dict, f"scoreboard {phase} capture phase is absent")
        capture.pop("wall_time", None)
        recurrent = capture.get("recurrent_capture")
        require(type(recurrent) is dict, f"scoreboard {phase} recurrent capture is absent")
        for field in (
            "alias_planning_wall_time",
            "final_schedule_wall_time",
            "pure_capture_binding_wall_time",
            "effect_assembly_sealing_wall_time",
            "recurrent_authentication_wall_time",
            "cursor_projection_wall_time",
            "residual_wall_time",
        ):
            recurrent.pop(field, None)
    compiler_timings = stable.get("prepare_compiler_process_timings")
    require(type(compiler_timings) is list, "scoreboard compiler timing inventory is absent")
    require(
        len(compiler_timings) == cold_prepare["prepare_compiler_process_count"],
        "scoreboard compiler timing inventory differs",
    )
    for timing in compiler_timings:
        require(type(timing) is dict, "scoreboard compiler timing entry differs")
        for field in ("permit_request_offset", "permit_wait", "process_wall_time"):
            timing.pop(field, None)
    step_phases = stable.get("step_phases")
    require(type(step_phases) is dict, "scoreboard step phases are absent")
    first = step_phases.get("first")
    require(type(first) is dict, "scoreboard first step phase is absent")
    require(first.get("phase") == "accumulation_only", "scoreboard first step phase differs")
    for field in (
        "total_wall_time",
        "executor_wall_time",
        "native_dispatcher_wall_time",
        "executor_host_wall_time",
        "recurrent_overhead_wall_time",
    ):
        first.pop(field, None)
    for role in ("warm_accumulation_only", "warm_optimizer_commit"):
        require(role in step_phases, f"scoreboard {role} presence is absent")
        warm = step_phases.get(role)
        require(type(warm) is dict, f"scoreboard {role} step phase differs")
        for field in (
            "total_wall_time",
            "steps_per_second",
            "executor_total_wall_time",
            "native_dispatcher_total_wall_time",
            "executor_host_total_wall_time",
            "recurrent_overhead_total_wall_time",
        ):
            warm.pop(field, None)
        for field in (
            "wall_time",
            "executor_wall_time",
            "native_dispatcher_wall_time",
            "executor_host_wall_time",
            "recurrent_overhead_wall_time",
        ):
            summary = warm.get(field)
            require(type(summary) is dict, f"scoreboard {role}.{field} summary differs")
            sample_count = exact_int(
                summary.get("sample_count"),
                f"scoreboard {role}.{field}.sample_count",
                minimum=1,
            )
            require(sample_count == 1, f"scoreboard {role}.{field} sample count differs")
            warm[field] = {"sample_count": sample_count}
    return stable


def nearest_rank(values: list[int], percentile: float) -> int:
    require(values, "timing sample set is empty")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def timing_summary(samples: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    by_role: dict[str, list[dict[str, Any]]] = {"accumulation": [], "commit": []}
    for role, sample in samples:
        destination = "accumulation" if role == "accumulation_only_samples" else "commit"
        by_role[destination].append(sample)
    result: dict[str, Any] = {}
    for role, entries in by_role.items():
        result[role] = {"sample_count": len(entries), "partitions": {}}
        for field in SAMPLE_PARTITIONS:
            values = [duration_ns(entry[field], f"{role}.{field}") for entry in entries]
            result[role]["partitions"][field] = {
                "median_ns": int(statistics.median(values)),
                "nearest_rank_p95_ns": nearest_rank(values, 0.95),
            }
    return result


def binary_digest(path: pathlib.Path) -> str:
    require(path.is_absolute() and not path.is_symlink() and path.is_file(), f"prebuilt binary is not a regular file: {path}")
    require(os.access(path, os.X_OK), f"prebuilt binary is not executable: {path}")
    size = path.stat().st_size
    require(0 < size <= MAX_BINARY_BYTES, f"prebuilt binary size is invalid: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_build_manifest(
    path: pathlib.Path,
    baseline_sha: str,
    candidate_sha: str,
    binary_digests: dict[str, str],
    measurement_mode: str,
) -> tuple[dict[str, Any], bytes]:
    value, raw = load_json(path)
    require(
        set(value)
        == {
            "format_version",
            "evidence_kind",
            "build_profile",
            "cargo_locked",
            "cargo_incremental",
            "cargo_build_jobs",
            "measurement_mode",
            "cargo_target",
            "rustflags",
            "toolchain",
            "target_triple",
            "revisions",
        },
        "build manifest fields differ",
    )
    require(value["format_version"] == 2 and value["evidence_kind"] == "native_cpu_transformer_training_comparison_builds", "build manifest kind differs")
    require(value["build_profile"] == "release" and value["cargo_locked"] is True, "build mode differs")
    require(value["cargo_incremental"] == "0" and value["cargo_build_jobs"] == "2", "build environment differs")
    require(value["measurement_mode"] == measurement_mode, "build measurement mode differs")
    require(value["cargo_target"] == MEASUREMENT_TARGETS[measurement_mode], "build target differs")
    require(value["rustflags"] == "-D warnings", "build rustflags differ")
    exact_str(value["toolchain"], "build toolchain")
    exact_str(value["target_triple"], "build target triple")
    revisions = value["revisions"]
    require(type(revisions) is dict and set(revisions) == {"baseline", "candidate"}, "build revision roles differ")
    for role, expected_sha in (("baseline", baseline_sha), ("candidate", candidate_sha)):
        revision = revisions[role]
        require(type(revision) is dict and set(revision) == {"source_sha", "executable_sha256"}, f"build manifest {role} fields differ")
        require(revision["source_sha"] == expected_sha, f"build manifest {role} revision differs")
        require(revision["executable_sha256"] == binary_digests[role], f"build manifest {role} binary differs")
    return value, raw


WARM_EVIDENCE_FILES = {
    "scoreboard": "native-cpu-training-scoreboard.json",
    "objective": "objective-evidence.json",
    "provenance": "provenance.txt",
    "resume_bundle": "replay-3-resume.rgab",
    "module_checkpoint": "replay-3-module-checkpoint.safetensors",
    "checksums": "sha256.txt",
}


def validate_warm_checksums(raw: bytes, raw_files: dict[str, bytes]) -> None:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise EvidenceError("warm checksum manifest is not UTF-8") from error
    expected = [
        ("scoreboard", WARM_EVIDENCE_FILES["scoreboard"]),
        ("objective", WARM_EVIDENCE_FILES["objective"]),
        ("provenance", WARM_EVIDENCE_FILES["provenance"]),
        ("resume_bundle", WARM_EVIDENCE_FILES["resume_bundle"]),
        ("module_checkpoint", WARM_EVIDENCE_FILES["module_checkpoint"]),
    ]
    require(len(lines) == len(expected), "warm checksum manifest count differs")
    for line, (key, filename) in zip(lines, expected):
        require(
            line == f"{sha256_bytes(raw_files[key])}  {filename}",
            f"warm checksum differs for {filename}",
        )


def warm_timing_ratio(candidate: int, baseline: int) -> float | None:
    """Zero-work cache hits have valid timings but no baseline ratio."""
    exact_int(candidate, "candidate warm timing")
    exact_int(baseline, "baseline warm timing")
    return candidate / baseline if baseline else None


def compare(arguments: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    baseline_sha = arguments.baseline_sha
    candidate_sha = arguments.candidate_sha
    measurement_mode = arguments.measurement_mode
    require(SHA_RE.fullmatch(baseline_sha) is not None, "baseline SHA must be lowercase and full")
    require(SHA_RE.fullmatch(candidate_sha) is not None, "candidate SHA must be lowercase and full")
    require(measurement_mode in MEASUREMENT_TARGETS, "measurement mode differs")
    root = pathlib.Path(arguments.root).resolve()
    require(root.is_dir() and not root.is_symlink(), "comparison root must be a directory")
    binaries = {
        "baseline": pathlib.Path(arguments.baseline_binary).resolve(),
        "candidate": pathlib.Path(arguments.candidate_binary).resolve(),
    }
    binary_digests = {role: binary_digest(path) for role, path in binaries.items()}
    build_manifest, build_raw = validate_build_manifest(
        pathlib.Path(arguments.build_manifest).resolve(),
        baseline_sha,
        candidate_sha,
        binary_digests,
        measurement_mode,
    )
    reasons: list[str] = []
    trial_records: list[dict[str, Any]] = []
    stable_scoreboard: dict[str, Any] | None = None
    stable_sidecar: dict[str, Any] | None = None
    stable_warm_objective: dict[str, Any] | None = None
    stable_warm_artifacts: dict[str, str] | None = None
    stable_machine: dict[str, Any] | None = None
    aggregate_inputs: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    for ordinal, (role, repetition) in enumerate(ORDER, start=1):
        expected_sha = baseline_sha if role == "baseline" else candidate_sha
        directory = root / f"trial-{ordinal:02d}-{role}"
        if measurement_mode == "steady-replay":
            files = {
                "scoreboard": directory / "native-cpu-training-scoreboard.json",
                "steady": directory / "native-cpu-training-steady-replays.json",
                "provenance": directory / "provenance.txt",
            }
        else:
            files = {
                key: directory / filename
                for key, filename in WARM_EVIDENCE_FILES.items()
            }
        trial_record: dict[str, Any] = {
            "ordinal": ordinal,
            "revision_role": role,
            "repetition": repetition,
            "source_sha": expected_sha,
            "binary_sha256": binary_digests[role],
            "status": "invalid",
            "files": {},
        }
        raw_files: dict[str, bytes] = {}
        file_errors: list[str] = []
        for name, path in files.items():
            try:
                raw = read_bounded(path.resolve())
                raw_files[name] = raw
                trial_record["files"][name] = {
                    "sha256": sha256_bytes(raw),
                    "byte_count": len(raw),
                }
            except EvidenceError as error:
                file_errors.append(f"{name}: {error}")
        if file_errors:
            reason = "; ".join(file_errors)
            trial_record["invalid_reason"] = reason
            trial_records.append(trial_record)
            reasons.append(f"trial {ordinal}: {reason}")
            continue
        try:
            machine = parse_provenance(
                raw_files["provenance"],
                expected_sha,
                binary_digests[role],
                measurement_mode,
            )
            if measurement_mode == "steady-replay":
                scoreboard = decode_json(raw_files["scoreboard"], files["scoreboard"].name)
                steady = decode_json(raw_files["steady"], files["steady"].name)
                scoreboard_projection = validate_scoreboard(scoreboard)
                sidecar_projection, timing = validate_sidecar(steady, expected_sha)
                require(
                    scoreboard_projection["checkpoint"]["capture_identity"]
                    == sidecar_projection["starting_checkpoint"]["capture_identity"],
                    "cold scoreboard and steady checkpoint captures differ",
                )
                require(
                    scoreboard_projection["checkpoint"]["byte_count"]
                    == sidecar_projection["final_checkpoint_byte_count"],
                    "cold scoreboard and steady checkpoint byte counts differ",
                )
                if stable_scoreboard is None:
                    stable_scoreboard = scoreboard_projection
                    stable_sidecar = sidecar_projection
                else:
                    if scoreboard_projection != stable_scoreboard:
                        reasons.append(f"trial {ordinal} stable scoreboard facts differ")
                    if sidecar_projection != stable_sidecar:
                        reasons.append(f"trial {ordinal} stable steady-workload facts differ")
                summary = timing_summary(timing["samples"])
            else:
                validate_warm_checksums(raw_files["checksums"], raw_files)
                try:
                    validated = LARGER_EVIDENCE.validate_larger_evidence(
                        files["objective"],
                        files["scoreboard"],
                        expected_sha,
                        files["resume_bundle"],
                        files["module_checkpoint"],
                        pathlib.Path(__file__).resolve().with_name(
                            "native_training_preparation_evidence.py"
                        ),
                        comparison_projection=True,
                    )
                except (LARGER_EVIDENCE.LargerEvidenceError, OSError, json.JSONDecodeError) as error:
                    raise EvidenceError(str(error)) from error
                warm_artifacts = {
                    key: sha256_bytes(raw_files[key])
                    for key in ("resume_bundle", "module_checkpoint")
                }
                if stable_scoreboard is None:
                    stable_scoreboard = validated["stable_scoreboard"]
                    stable_warm_objective = validated["stable_objective"]
                    stable_warm_artifacts = warm_artifacts
                else:
                    if validated["stable_scoreboard"] != stable_scoreboard:
                        reasons.append(f"trial {ordinal} stable scoreboard facts differ")
                    if validated["stable_objective"] != stable_warm_objective:
                        reasons.append(f"trial {ordinal} stable warm objective facts differ")
                    if warm_artifacts != stable_warm_artifacts:
                        reasons.append(f"trial {ordinal} warm checkpoint artifacts differ")
                summary = validated["warm_timing"]
            if stable_machine is None:
                stable_machine = machine
            elif machine != stable_machine:
                reasons.append(f"trial {ordinal} runner or toolchain provenance differs")
            aggregate_inputs[role].append(summary)
            trial_record["status"] = "valid"
            trial_record["timing"] = summary
        except EvidenceError as error:
            trial_record["invalid_reason"] = str(error)
            reasons.append(f"trial {ordinal}: {error}")
        trial_records.append(trial_record)
    if stable_machine is not None:
        if build_manifest["toolchain"].splitlines() != stable_machine["rustc"]:
            reasons.append("build and measurement Rust toolchain provenance differs")
        expected_host = f"host: {build_manifest['target_triple']}"
        if expected_host not in stable_machine["rustc"]:
            reasons.append("build target and measurement Rust host differ")

    def aggregate(role: str) -> dict[str, Any]:
        summaries = aggregate_inputs[role]
        require(len(summaries) == 2, f"{role} does not have two valid trials")
        if measurement_mode == "warm-resume":
            fields = set(summaries[0])
            require(
                all(set(summary) == fields for summary in summaries),
                f"{role} warm timing fields differ",
            )
            return {
                "timings": {
                    field: {
                        "trial_ns": [summary[field] for summary in summaries],
                        "median_of_trials_ns": int(
                            statistics.median(summary[field] for summary in summaries)
                        ),
                    }
                    for field in sorted(fields)
                }
            }
        result: dict[str, Any] = {}
        for phase in ("accumulation", "commit"):
            result[phase] = {"partitions": {}}
            for partition in SAMPLE_PARTITIONS:
                medians = [entry[phase]["partitions"][partition]["median_ns"] for entry in summaries]
                p95s = [entry[phase]["partitions"][partition]["nearest_rank_p95_ns"] for entry in summaries]
                result[phase]["partitions"][partition] = {
                    "trial_median_ns": medians,
                    "median_of_trial_medians_ns": int(statistics.median(medians)),
                    "trial_nearest_rank_p95_ns": p95s,
                    "median_of_trial_p95_ns": int(statistics.median(p95s)),
                }
        return result

    aggregates: dict[str, Any] | None = None
    ratios: dict[str, Any] | None = None
    if not reasons:
        try:
            aggregates = {role: aggregate(role) for role in ("baseline", "candidate")}
            ratios = {}
            if measurement_mode == "warm-resume":
                ratios["warm_resume"] = {"candidate_over_baseline": {}}
                for field, baseline_timing in aggregates["baseline"]["timings"].items():
                    baseline = baseline_timing["median_of_trials_ns"]
                    candidate = aggregates["candidate"]["timings"][field][
                        "median_of_trials_ns"
                    ]
                    # Warm cache hits legitimately perform no rendering or
                    # compilation. Preserve those zero samples; their ratio
                    # is undefined, not evidence that the trial is invalid.
                    ratios["warm_resume"]["candidate_over_baseline"][field] = warm_timing_ratio(
                        candidate, baseline
                    )
            else:
                for phase in ("accumulation", "commit"):
                    ratios[phase] = {"candidate_over_baseline": {}}
                    for partition in SAMPLE_PARTITIONS:
                        baseline = aggregates["baseline"][phase]["partitions"][partition]["median_of_trial_medians_ns"]
                        candidate = aggregates["candidate"][phase]["partitions"][partition]["median_of_trial_medians_ns"]
                        require(baseline != 0, f"{phase} {partition} baseline timing is zero")
                        ratios[phase]["candidate_over_baseline"][partition] = candidate / baseline
        except EvidenceError as error:
            reasons.append(str(error))
    manifest = {
        "format_version": FORMAT_VERSION,
        "evidence_kind": EVIDENCE_KIND,
        "status": "comparable" if not reasons else "incomparable",
        "interpretation": "observational same-runner timing evidence; no speed threshold or speedup claim",
        "build_provenance_scope": "the workflow binds each exact checkout, build invocation, and executable digest; the digest authenticates measured bytes but is not independent proof of source correspondence",
        "reasons": reasons,
        "baseline_sha": baseline_sha,
        "candidate_sha": candidate_sha,
        "measurement_mode": measurement_mode,
        "trial_order": [role for role, _ in ORDER],
        "workflow": {
            "repository": os.environ.get("GITHUB_REPOSITORY", "unknown"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "unknown"),
            "job": os.environ.get("GITHUB_JOB", "unknown"),
        },
        "build_manifest": {
            "sha256": sha256_bytes(build_raw),
            "facts": build_manifest,
        },
        "binary_sha256": binary_digests,
        "runner_and_toolchain": stable_machine,
        "trials": trial_records,
        "aggregates": aggregates,
        "observational_ratios": ratios,
    }
    return manifest, not reasons


def write_manifest(path: pathlib.Path, value: dict[str, Any]) -> None:
    require(path.is_absolute(), "comparison manifest path must be absolute")
    require(not path.exists() and not path.is_symlink(), "comparison manifest already exists")
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(len(raw) <= MAX_FILE_BYTES, "comparison manifest is too large")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--measurement-mode", choices=MEASUREMENT_TARGETS, default="steady-replay")
    parser.add_argument("--baseline-binary", required=True)
    parser.add_argument("--candidate-binary", required=True)
    parser.add_argument("--build-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    output = pathlib.Path(arguments.output).resolve()
    try:
        manifest, comparable = compare(arguments)
    except EvidenceError as error:
        manifest = {
            "format_version": FORMAT_VERSION,
            "evidence_kind": EVIDENCE_KIND,
            "status": "incomparable",
            "interpretation": "observational same-runner timing evidence; no speed threshold or speedup claim",
            "build_provenance_scope": "the workflow binds each exact checkout, build invocation, and executable digest; the digest authenticates measured bytes but is not independent proof of source correspondence",
            "reasons": [str(error)],
            "baseline_sha": arguments.baseline_sha,
            "candidate_sha": arguments.candidate_sha,
            "measurement_mode": arguments.measurement_mode,
            "trials": [],
        }
        comparable = False
    try:
        write_manifest(output, manifest)
    except EvidenceError as error:
        print(f"comparison manifest was not written: {error}", file=sys.stderr)
        return 2
    if not comparable:
        for reason in manifest["reasons"]:
            print(f"incomparable: {reason}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["observational_ratios"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
