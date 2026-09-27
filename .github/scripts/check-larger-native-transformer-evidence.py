#!/usr/bin/env python3
"""Validate larger native Transformer evidence for recording and comparison."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import pathlib
import sys
from typing import Any


class LargerEvidenceError(ValueError):
    pass


def _duration_ns(value: Any, label: str) -> int:
    if type(value) is not dict or set(value) != {"secs", "nanos"}:
        raise LargerEvidenceError(f"{label} duration differs")
    seconds = value["secs"]
    nanos = value["nanos"]
    if (
        type(seconds) is not int
        or type(nanos) is not int
        or not 0 <= seconds <= (1 << 64) - 1
        or not 0 <= nanos < 1_000_000_000
    ):
        raise LargerEvidenceError(f"{label} duration differs")
    return seconds * 1_000_000_000 + nanos


def _timing_shape(value: Any) -> Any:
    if type(value) is dict and set(value) == {"secs", "nanos"}:
        _duration_ns(value, "timing shape")
        return "duration"
    if type(value) is dict:
        return {key: _timing_shape(entry) for key, entry in value.items()}
    if type(value) is list:
        return [_timing_shape(entry) for entry in value]
    return value


def _stable_scoreboard(value: dict[str, Any]) -> dict[str, Any]:
    stable = copy.deepcopy(value)
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
    for role in ("main", "accumulation", "partial_flush", "zero_grad", "evaluation"):
        program = stable.get(role)
        if type(program) is not dict:
            raise LargerEvidenceError(f"larger Transformer {role} program is absent")
        program.pop("preparation_timing", None)
    checkpoint = stable.get("checkpoint")
    if type(checkpoint) is not dict:
        raise LargerEvidenceError("larger Transformer checkpoint is absent")
    checkpoint.pop("wall_time", None)
    compile_phases = stable.get("compile_phases")
    if type(compile_phases) is not dict:
        raise LargerEvidenceError("larger Transformer compile phases are absent")
    compile_phases.pop("residual_wall_time", None)
    for phase in ("objective_forward", "autograd", "optimizer_lowering"):
        compile_phases[phase].pop("wall_time", None)
    for phase in (
        "main_capture",
        "accumulation_capture",
        "partial_flush",
        "zero_grad",
        "evaluation",
    ):
        capture = compile_phases[phase]
        capture.pop("wall_time", None)
        recurrent = capture.get("recurrent_capture")
        if recurrent is None:
            continue
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
    if type(compiler_timings) is not list:
        raise LargerEvidenceError("larger Transformer compiler timings are absent")
    for timing in compiler_timings:
        if type(timing) is not dict:
            raise LargerEvidenceError("larger Transformer compiler timing differs")
        for field in ("permit_request_offset", "permit_wait", "process_wall_time"):
            timing.pop(field, None)
    step_phases = stable.get("step_phases")
    if type(step_phases) is not dict:
        raise LargerEvidenceError("larger Transformer step phases are absent")
    first = step_phases.get("first")
    if type(first) is not dict:
        raise LargerEvidenceError("larger Transformer first step phase is absent")
    for field in (
        "total_wall_time",
        "executor_wall_time",
        "native_dispatcher_wall_time",
        "executor_host_wall_time",
        "recurrent_overhead_wall_time",
    ):
        first.pop(field, None)
    for role, expected_samples in (
        ("warm_accumulation_only", 2),
        ("warm_optimizer_commit", 3),
    ):
        phase = step_phases.get(role)
        if type(phase) is not dict:
            raise LargerEvidenceError(f"larger Transformer {role} phase is absent")
        for field in (
            "total_wall_time",
            "steps_per_second",
            "executor_total_wall_time",
            "native_dispatcher_total_wall_time",
            "executor_host_total_wall_time",
            "recurrent_overhead_total_wall_time",
        ):
            phase.pop(field, None)
        for field in (
            "wall_time",
            "executor_wall_time",
            "native_dispatcher_wall_time",
            "executor_host_wall_time",
            "recurrent_overhead_wall_time",
        ):
            summary = phase.get(field)
            if type(summary) is not dict or summary.get("sample_count") != expected_samples:
                raise LargerEvidenceError(
                    f"larger Transformer {role}.{field} sample count differs"
                )
            phase[field] = {"sample_count": expected_samples}
    return stable


def _warm_timing_projection(warm: dict[str, Any]) -> dict[str, int]:
    result = {
        field: warm[field]
        for field in (
            "artifact_decode_wall_time_ns",
            "owner_restore_wall_time_ns",
            "preparation_wall_time_ns",
        )
    }
    preparation = warm["preparation"]
    for field, value in preparation.items():
        if field.endswith("_wall_time_ns"):
            result[f"preparation.{field}"] = value
    for role in ("main", "accumulation", "partial_flush", "zero_grad", "evaluation"):
        for field, value in preparation[role].items():
            if field.endswith("_wall_time_ns"):
                result[f"preparation.{role}.{field}"] = value

    def collect_durations(prefix: str, value: Any) -> None:
        if type(value) is dict and set(value) == {"secs", "nanos"}:
            result[prefix] = _duration_ns(value, prefix)
            return
        if type(value) is dict:
            for key, entry in value.items():
                collect_durations(f"{prefix}.{key}", entry)

    collect_durations("preparation.prepare_finalization", preparation["prepare_finalization"])
    for phase in preparation.get("capsule_phases", []):
        for field, value in phase.items():
            if field.endswith("_wall_time_ns"):
                result[f"preparation.capsule.{phase['program_index']}.{field}"] = value
    if any(type(value) is not int or value < 0 for value in result.values()):
        raise LargerEvidenceError("larger Transformer warm timing projection differs")
    return result


def validate_larger_evidence(
    objective_path: pathlib.Path,
    scoreboard_path: pathlib.Path,
    expected_sha: str,
    resume_bundle_path: pathlib.Path,
    module_checkpoint_path: pathlib.Path,
    preparation_evidence_path: pathlib.Path,
    *,
    comparison_projection: bool = False,
) -> dict[str, Any]:
    def load_preparation_evidence_module(path):
        spec = importlib.util.spec_from_file_location(
            "rustgrad_native_training_preparation_evidence", path
        )
        if spec is None or spec.loader is None:
            raise LargerEvidenceError("native preparation evidence validator is absent")
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        return module

    PREPARATION_EVIDENCE = load_preparation_evidence_module(
        preparation_evidence_path.resolve()
    )

    try:
        objective = json.loads(objective_path.read_text(encoding="utf-8"))
        scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise LargerEvidenceError("larger Transformer evidence is not UTF-8") from error
    if not isinstance(objective, dict):
        raise LargerEvidenceError("larger Transformer objective must be an object")
    if not isinstance(scoreboard, dict):
        raise LargerEvidenceError("larger Transformer scoreboard must be an object")
    scoreboard_main = scoreboard.get("main")
    scoreboard_evaluation = scoreboard.get("evaluation")
    def same_float(left, right):
        return (
            type(left) in (int, float)
            and type(right) in (int, float)
            and math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)
        )

    def phase_sample_count(report):
        if not isinstance(report, dict):
            return None
        wall_time = report.get("wall_time")
        return wall_time.get("sample_count") if isinstance(wall_time, dict) else None

    def duration_nanos(value):
        try:
            return PREPARATION_EVIDENCE.duration_ns(value, "larger Transformer")
        except PREPARATION_EVIDENCE.PreparationEvidenceError:
            return None

    if objective.get("schema_version") not in (8, 9) or objective.get("git_sha") != expected_sha:
        raise LargerEvidenceError("larger Transformer objective provenance is invalid")
    expected_workload = {
        "batch": 4,
        "time": 8,
        "vocabulary": 16,
        "embedding": 8,
        "heads": 2,
        "feed_forward": 32,
        "blocks": 2,
        "compile_count": 1,
        "gradient_accumulation_steps": 2,
        "replays": 6,
    }
    if objective.get("workload") != expected_workload:
        raise LargerEvidenceError("larger Transformer workload identity is invalid")

    objective_facts = objective.get("objective")
    if not isinstance(objective_facts, dict):
        raise LargerEvidenceError("larger Transformer objective facts are absent")
    trajectory = objective_facts.get("evaluation_trajectory")
    expected_frontiers = [(0, 0, 0), (2, 1, 0), (4, 2, 0), (6, 3, 0)]
    if not isinstance(trajectory, list) or len(trajectory) != len(expected_frontiers):
        raise LargerEvidenceError("larger Transformer evaluation trajectory is invalid")
    trajectory_losses = []
    for point, expected in zip(trajectory, expected_frontiers):
        if not isinstance(point, dict):
            raise LargerEvidenceError("larger Transformer evaluation point is invalid")
        if (
            (point.get("replay"), point.get("optimizer_step"), point.get("accumulation_index"))
            != expected
            or point.get("checkpoint_state_neutral") is not True
        ):
            raise LargerEvidenceError("larger Transformer evaluation frontier is invalid")
        loss = point.get("token_mean_loss")
        if type(loss) not in (int, float) or not math.isfinite(loss):
            raise LargerEvidenceError("larger Transformer evaluation loss is non-finite")
        samples = point.get("samples")
        if not isinstance(samples, list) or len(samples) != 2:
            raise LargerEvidenceError("larger Transformer evaluation samples are invalid")
        weighted_loss = 0.0
        total_weight = 0
        for sample, batch_replay, valid_tokens in zip(samples, [1, 2], [22, 16]):
            if not isinstance(sample, dict):
                raise LargerEvidenceError("larger Transformer evaluation sample is invalid")
            sample_loss = sample.get("token_mean_loss")
            if (
                sample.get("batch_replay") != batch_replay
                or not isinstance(scoreboard_evaluation, dict)
                or sample.get("capture_identity") != scoreboard_evaluation.get("capture_identity")
                or sample.get("valid_token_count") != valid_tokens
                or type(sample_loss) not in (int, float)
                or not math.isfinite(sample_loss)
                or sample.get("native_fallback_count") != 0
                or type(sample.get("executed_native_item_count")) is not int
                or sample["executed_native_item_count"] <= 0
                or type(sample.get("module_dispatch_count")) is not int
                or sample["module_dispatch_count"] <= 0
            ):
                raise LargerEvidenceError("larger Transformer evaluation sample contract is invalid")
            weighted_loss += sample_loss * valid_tokens
            total_weight += valid_tokens
        if total_weight != 38 or not same_float(loss, weighted_loss / total_weight):
            raise LargerEvidenceError("larger Transformer weighted evaluation mean is invalid")
        trajectory_losses.append(loss)
    initial_loss = objective_facts.get("initial_token_mean_loss")
    final_loss = objective_facts.get("final_token_mean_loss")
    if (
        type(initial_loss) not in (int, float)
        or type(final_loss) not in (int, float)
        or not math.isfinite(initial_loss)
        or not math.isfinite(final_loss)
        or not same_float(initial_loss, trajectory_losses[0])
        or not same_float(final_loss, trajectory_losses[-1])
        or objective_facts.get("decreased") is not True
        or not final_loss < initial_loss
    ):
        raise LargerEvidenceError("larger Transformer objective trajectory does not decrease")

    progress = objective.get("progress")
    if not isinstance(progress, dict):
        raise LargerEvidenceError("larger Transformer progress evidence is absent")
    expected_replays = [
        {
            "replay": replay,
            "optimizer_step": replay // 2,
            "accumulation_index": replay % 2,
            "did_update": replay % 2 == 0,
        }
        for replay in range(1, 7)
    ]
    if progress.get("replays") != expected_replays:
        raise LargerEvidenceError("larger Transformer replay progression is invalid")
    pending = progress.get("pending_resume_checkpoint")
    terminal = progress.get("terminal_scoreboard_checkpoint")
    if (
        not isinstance(pending, dict)
        or pending.get("replay") != 3
        or pending.get("optimizer_step") != 1
        or pending.get("accumulation_index") != 1
        or type(pending.get("bytes")) is not int
        or pending["bytes"] <= 0
        or type(pending.get("module_checkpoint_bytes")) is not int
        or pending["module_checkpoint_bytes"] <= 0
        or pending["module_checkpoint_bytes"] != module_checkpoint_path.stat().st_size
    ):
        raise LargerEvidenceError("larger Transformer pending checkpoint evidence is invalid")
    if (
        not isinstance(terminal, dict)
        or terminal.get("replay") != 6
        or terminal.get("optimizer_step") != 3
        or terminal.get("accumulation_index") != 0
        or type(terminal.get("bytes")) is not int
        or terminal["bytes"] <= 0
        or progress.get("final_replay") != 6
        or progress.get("final_optimizer_step") != 3
        or progress.get("final_accumulation_index") != 0
        or progress.get("exact_resume") is not True
    ):
        raise LargerEvidenceError("larger Transformer terminal checkpoint evidence is invalid")

    native = objective.get("native")
    if not isinstance(native, dict):
        raise LargerEvidenceError("larger Transformer native evidence is absent")
    if scoreboard.get("format_version") != 26 or scoreboard.get("initial_replay_step") != 0:
        raise LargerEvidenceError("larger Transformer scoreboard identity is invalid")
    preparation_programs = {
        role: scoreboard.get(role)
        for role in ["main", "accumulation", "partial_flush", "zero_grad", "evaluation"]
    }
    if any(not isinstance(program, dict) for program in preparation_programs.values()):
        raise LargerEvidenceError("larger Transformer preparation program inventory is invalid")
    try:
        PREPARATION_EVIDENCE.validate_preparation_finalization(
            scoreboard, 26, preparation_programs
        )
    except PREPARATION_EVIDENCE.PreparationEvidenceError as error:
        raise LargerEvidenceError(f"larger Transformer preparation evidence is invalid: {error}") from error
    compile_phases = scoreboard.get("compile_phases")
    compile_phase_names = {
        "compile_count",
        "objective_forward",
        "autograd",
        "optimizer_lowering",
        "main_capture",
        "accumulation_capture",
        "partial_flush",
        "zero_grad",
        "evaluation",
        "residual_wall_time",
    }
    if (
        not isinstance(compile_phases, dict)
        or set(compile_phases) != compile_phase_names
        or compile_phases.get("compile_count") != 1
    ):
        raise LargerEvidenceError("larger Transformer compile-phase evidence is invalid")

    compile_phase_total_ns = 0
    graph_node_counts = []
    for phase_name in ["objective_forward", "autograd", "optimizer_lowering"]:
        phase = compile_phases.get(phase_name)
        if (
            not isinstance(phase, dict)
            or set(phase) != {"wall_time", "graph_node_count"}
            or type(phase.get("graph_node_count")) is not int
            or phase["graph_node_count"] <= 0
            or duration_nanos(phase.get("wall_time")) is None
        ):
            raise LargerEvidenceError(f"larger Transformer {phase_name} compile phase is invalid")
        graph_node_counts.append(phase["graph_node_count"])
        compile_phase_total_ns += duration_nanos(phase["wall_time"])
    if graph_node_counts != sorted(graph_node_counts):
        raise LargerEvidenceError("larger Transformer compiled graph inventories are not monotonic")

    capture_programs = {
        "main_capture": scoreboard_main,
        "accumulation_capture": scoreboard.get("accumulation"),
        "partial_flush": scoreboard.get("partial_flush"),
        "zero_grad": scoreboard.get("zero_grad"),
        "evaluation": scoreboard_evaluation,
    }
    for phase_name, program in capture_programs.items():
        phase = compile_phases.get(phase_name)
        expected_phase_fields = {"wall_time", "logical_schedule_item_count"}
        if phase_name != "evaluation":
            expected_phase_fields.add("recurrent_capture")
        if (
            not isinstance(phase, dict)
            or set(phase) != expected_phase_fields
            or not isinstance(program, dict)
            or type(phase.get("logical_schedule_item_count")) is not int
            or phase["logical_schedule_item_count"] <= 0
            or phase["logical_schedule_item_count"]
            != program.get("logical_schedule_item_count")
            or duration_nanos(phase.get("wall_time")) is None
        ):
            raise LargerEvidenceError(f"larger Transformer {phase_name} compile phase is invalid")
        compile_phase_total_ns += duration_nanos(phase["wall_time"])

    recurrent_phase_names = ["main_capture", "accumulation_capture", "partial_flush", "zero_grad"]
    expected_preview_counts = [2, 3, 3, 1]
    recurrent_state_counts = []
    recurrent_fields = {
        "alias_planning_wall_time",
        "preview_schedule_count",
        "final_schedule_wall_time",
        "pure_capture_binding_wall_time",
        "effect_assembly_sealing_wall_time",
        "recurrent_authentication_wall_time",
        "residual_wall_time",
        "recurrent_state_count",
    }
    for ordinal, (phase_name, expected_previews) in enumerate(
        zip(recurrent_phase_names, expected_preview_counts)
    ):
        phase = compile_phases[phase_name]
        recurrent = phase.get("recurrent_capture")
        expected_fields = set(recurrent_fields)
        if ordinal != 0:
            expected_fields.add("cursor_projection_wall_time")
        if not isinstance(recurrent, dict) or set(recurrent) != expected_fields:
            raise LargerEvidenceError(f"larger Transformer {phase_name} recurrent capture is invalid")
        if (
            type(recurrent.get("preview_schedule_count")) is not int
            or recurrent["preview_schedule_count"] != expected_previews
        ):
            raise LargerEvidenceError(f"larger Transformer {phase_name} preview inventory differs")
        state_count = recurrent.get("recurrent_state_count")
        if type(state_count) is not int or state_count <= 0:
            raise LargerEvidenceError(f"larger Transformer {phase_name} state inventory is invalid")
        program_state_count = capture_programs[phase_name].get("recurrent_state_count")
        if type(program_state_count) is not int or program_state_count != state_count:
            raise LargerEvidenceError(f"larger Transformer {phase_name} program state inventory differs")
        recurrent_state_counts.append(state_count)
        timing_fields = [
            "alias_planning_wall_time",
            "final_schedule_wall_time",
            "pure_capture_binding_wall_time",
            "effect_assembly_sealing_wall_time",
            "recurrent_authentication_wall_time",
            "residual_wall_time",
        ]
        if ordinal != 0:
            timing_fields.append("cursor_projection_wall_time")
        timings = [duration_nanos(recurrent.get(field)) for field in timing_fields]
        if any(value is None for value in timings):
            raise LargerEvidenceError(f"larger Transformer {phase_name} timing is invalid")
        timing_total = sum(timings)
        if timing_total != duration_nanos(phase["wall_time"]):
            raise LargerEvidenceError(f"larger Transformer {phase_name} timing does not balance")
    if recurrent_state_counts[0] != scoreboard.get("recurrent_logical_state_count"):
        raise LargerEvidenceError("larger Transformer main recurrent state inventory differs")
    if recurrent_state_counts[1] != recurrent_state_counts[0]:
        raise LargerEvidenceError("larger Transformer accumulation recurrent state inventory differs")
    if not (recurrent_state_counts[3] <= recurrent_state_counts[2] <= recurrent_state_counts[1]):
        raise LargerEvidenceError("larger Transformer auxiliary recurrent state inventory differs")
    if "recurrent_state_count" in scoreboard_evaluation:
        raise LargerEvidenceError("larger Transformer evaluation claims recurrent state inventory")

    residual_compile_ns = duration_nanos(compile_phases.get("residual_wall_time"))
    compile_wall_ns = duration_nanos(scoreboard.get("compile_wall_time"))
    if (
        residual_compile_ns is None
        or compile_wall_ns is None
        or compile_phase_total_ns + residual_compile_ns != compile_wall_ns
    ):
        raise LargerEvidenceError("larger Transformer compile-phase timing does not balance")
    scoreboard_checkpoint = scoreboard.get("checkpoint")
    if (
        not isinstance(scoreboard_checkpoint, dict)
        or not isinstance(scoreboard_main, dict)
        or scoreboard_checkpoint.get("replay_step") != 6
        or scoreboard_checkpoint.get("byte_count") != terminal["bytes"]
        or scoreboard_checkpoint.get("capture_identity") != scoreboard_main.get("capture_identity")
    ):
        raise LargerEvidenceError("larger Transformer scoreboard checkpoint is invalid")
    expected_native = {
        "fallback_count": scoreboard.get("fallback_count"),
        "successful_replay_count": scoreboard.get("successful_replay_count"),
        "recurrent_state_count": scoreboard.get("recurrent_logical_state_count"),
        "recurrent_state_bytes": scoreboard.get("recurrent_logical_state_bytes"),
    }
    if native != expected_native:
        raise LargerEvidenceError("larger Transformer native evidence differs from scoreboard")
    if (
        native["fallback_count"] != 0
        or native["successful_replay_count"] != 6
        or type(native["recurrent_state_count"]) is not int
        or native["recurrent_state_count"] <= 0
        or type(native["recurrent_state_bytes"]) is not int
        or native["recurrent_state_bytes"] <= 0
    ):
        raise LargerEvidenceError("larger Transformer native replay inventory is invalid")
    for program_name in ["main", "accumulation", "partial_flush", "zero_grad", "evaluation"]:
        program = scoreboard.get(program_name)
        if (
            not isinstance(program, dict)
            or program.get("vectorized") is not True
            or type(program.get("native_item_count")) is not int
            or program["native_item_count"] <= 0
        ):
            raise LargerEvidenceError(f"larger Transformer {program_name} program inventory is invalid")
    step_phases = scoreboard.get("step_phases")
    if (
        not isinstance(step_phases, dict)
        or not isinstance(step_phases.get("first"), dict)
        or step_phases["first"].get("phase") != "accumulation_only"
        or not isinstance(step_phases.get("warm_accumulation_only"), dict)
        or phase_sample_count(step_phases["warm_accumulation_only"]) != 2
        or not isinstance(step_phases.get("warm_optimizer_commit"), dict)
        or phase_sample_count(step_phases["warm_optimizer_commit"]) != 3
    ):
        raise LargerEvidenceError("larger Transformer scoreboard replay census is invalid")

    warm = objective.get("warm_resume")
    if not isinstance(warm, dict):
        raise LargerEvidenceError("larger Transformer warm resume evidence is absent")
    if (
        warm.get("replay_from") != 3
        or warm.get("replay_to") != 6
        or warm.get("program_count") != 5
        or warm.get("loaded_module_count") != 5
        or warm.get("durable_artifact_cache_hit_count") != 5
        or warm.get("durable_artifact_cache_miss_count") != 0
        or warm.get("render_capsule_hit_count") != 5
        or warm.get("render_capsule_miss_count") != 0
        or warm.get("local_render_job_count") != 0
        or warm.get("compiler_invocation_count") != 0
        or warm.get("linker_invocation_count") != 0
        or warm.get("fallback_count") != 0
        or warm.get("module_visit_count") != 37
        or warm.get("canonical_state_count") != 36
    ):
        raise LargerEvidenceError("larger Transformer warm resume inventory is invalid")
    preparation = warm.get("preparation")
    preparation_roles = [
        "main",
        "accumulation",
        "partial_flush",
        "zero_grad",
        "evaluation",
    ]
    preparation_totals = {
        "summed_program_wall_time_ns",
        "runtime_overhead_wall_time_ns",
        "module_overlap_wall_time_ns",
        "render_overlap_wall_time_ns",
        "effective_render_wall_time_ns",
        "effective_render_fraction",
    }
    preparation_details = {"prepare_finalization"}
    if objective["schema_version"] == 9:
        preparation_details.add("capsule_phases")
    if (
        not isinstance(preparation, dict)
        or set(preparation)
        != set(preparation_roles) | preparation_totals | preparation_details
    ):
        raise LargerEvidenceError("larger Transformer warm preparation evidence is invalid")
    phase_fields = {
        "total_wall_time_ns",
        "layout_wall_time_ns",
        "render_wall_time_ns",
        "compiler_wall_time_ns",
        "load_wall_time_ns",
        "residual_wall_time_ns",
    }
    program_total = 0
    render_total = 0
    for role in preparation_roles:
        phases = preparation.get(role)
        if not isinstance(phases, dict) or set(phases) != phase_fields:
            raise LargerEvidenceError(f"larger Transformer warm {role} phases are invalid")
        if any(type(phases.get(field)) is not int or phases[field] < 0 for field in phase_fields):
            raise LargerEvidenceError(f"larger Transformer warm {role} timing is invalid")
        accounted = sum(
            phases[field]
            for field in [
                "layout_wall_time_ns",
                "render_wall_time_ns",
                "compiler_wall_time_ns",
                "load_wall_time_ns",
                "residual_wall_time_ns",
            ]
        )
        if phases["total_wall_time_ns"] != accounted:
            raise LargerEvidenceError(f"larger Transformer warm {role} timing does not balance")
        if phases["compiler_wall_time_ns"] != 0:
            raise LargerEvidenceError(f"larger Transformer warm {role} unexpectedly compiled")
        program_total += phases["total_wall_time_ns"]
        render_total += phases["render_wall_time_ns"]
    for field in preparation_totals - {"effective_render_fraction"}:
        if type(preparation.get(field)) is not int or preparation[field] < 0:
            raise LargerEvidenceError(f"larger Transformer warm preparation {field} is invalid")
    if preparation["summed_program_wall_time_ns"] != program_total:
        raise LargerEvidenceError("larger Transformer warm program timing sum is invalid")
    if (
        preparation["module_overlap_wall_time_ns"]
        + preparation["render_overlap_wall_time_ns"]
        > program_total
    ):
        raise LargerEvidenceError("larger Transformer warm overlap exceeds program timing")
    whole_preparation = warm.get("preparation_wall_time_ns")
    if type(whole_preparation) is not int or whole_preparation <= 0:
        raise LargerEvidenceError("larger Transformer warm whole preparation timing is invalid")
    def benchmark_duration(nanos):
        return {
            "secs": nanos // 1_000_000_000,
            "nanos": nanos % 1_000_000_000,
        }

    warm_programs = {
        role: {
            "preparation_timing": {
                "total": benchmark_duration(preparation[role]["total_wall_time_ns"]),
                "render": benchmark_duration(preparation[role]["render_wall_time_ns"]),
            }
        }
        for role in preparation_roles
    }
    warm_partition = {
        "prepare_finalization": preparation["prepare_finalization"],
        "prepare_wall_time": benchmark_duration(whole_preparation),
        "prepare_runtime_overhead_wall_time": benchmark_duration(
            preparation["runtime_overhead_wall_time_ns"]
        ),
        "prepare_parallel_module_overlap_wall_time": benchmark_duration(
            preparation["module_overlap_wall_time_ns"]
        ),
        "prepare_parallel_render_overlap_wall_time": benchmark_duration(
            preparation["render_overlap_wall_time_ns"]
        ),
    }
    try:
        PREPARATION_EVIDENCE.validate_preparation_finalization(
            warm_partition, 26, warm_programs
        )
    except PREPARATION_EVIDENCE.PreparationEvidenceError as error:
        raise LargerEvidenceError(
            f"larger Transformer warm preparation finalization is invalid: {error}"
        ) from error
    reconstructed_preparation = (
        preparation["runtime_overhead_wall_time_ns"]
        + program_total
        - preparation["module_overlap_wall_time_ns"]
        - preparation["render_overlap_wall_time_ns"]
    )
    if reconstructed_preparation != whole_preparation:
        raise LargerEvidenceError("larger Transformer warm whole preparation timing does not balance")
    effective_render = render_total - preparation["render_overlap_wall_time_ns"]
    if (
        effective_render < 0
        or preparation["effective_render_wall_time_ns"] != effective_render
    ):
        raise LargerEvidenceError("larger Transformer warm effective render timing is invalid")
    if objective["schema_version"] == 9:
        phases = preparation.get("capsule_phases")
        fields = ("recipe_wall_time_ns", "file_read_wall_time_ns",
                  "decode_wall_time_ns", "authentication_wall_time_ns")
        if type(phases) is not list or len(phases) != 5:
            raise LargerEvidenceError("larger Transformer capsule phase inventory differs")
        total = 0
        for index, phase in enumerate(phases):
            if (type(phase) is not dict or set(phase) != {"program_index", *fields}
                    or type(phase.get("program_index")) is not int
                    or phase["program_index"] != index):
                raise LargerEvidenceError("larger Transformer capsule phase order differs")
            for field in fields:
                value = phase[field]
                if type(value) is not int or not 0 <= value <= 2**64 - 1:
                    raise LargerEvidenceError("larger Transformer capsule phase timing is invalid")
                total += value
        enclosing = duration_nanos(preparation["prepare_finalization"].get(
            "render_batch_orchestration_wall_time"))
        if enclosing is None or total > enclosing:
            raise LargerEvidenceError("larger Transformer capsule phases exceed batch")
    render_fraction = preparation.get("effective_render_fraction")
    if (
        not isinstance(render_fraction, (int, float))
        or isinstance(render_fraction, bool)
        or not math.isfinite(render_fraction)
        or not same_float(render_fraction, effective_render / whole_preparation)
        or not 0.0 <= render_fraction <= 1.0
    ):
        raise LargerEvidenceError("larger Transformer warm effective render fraction is invalid")
    for field in [
        "resume_bundle_bytes",
        "module_checkpoint_bytes",
    ]:
        if type(warm.get(field)) is not int or warm[field] <= 0:
            raise LargerEvidenceError(f"larger Transformer warm resume {field} is invalid")
    if warm["resume_bundle_bytes"] != resume_bundle_path.stat().st_size:
        raise LargerEvidenceError("larger Transformer warm resume bundle byte count is invalid")
    if warm["module_checkpoint_bytes"] != module_checkpoint_path.stat().st_size:
        raise LargerEvidenceError("larger Transformer module checkpoint byte count is invalid")
    for field in [
        "artifact_decode_wall_time_ns",
        "owner_restore_wall_time_ns",
        "preparation_wall_time_ns",
    ]:
        if type(warm.get(field)) is not int or warm[field] < 0:
            raise LargerEvidenceError(f"larger Transformer warm resume {field} is invalid")
    for field in [
        "artifact_checkpoint_authenticated",
        "topology_authenticated",
        "different_initialization",
        "fresh_executor",
        "exact_continuation",
        "evaluation_state_neutral",
        "target_owned_module_published",
    ]:
        if warm.get(field) is not True:
            raise LargerEvidenceError(f"larger Transformer warm resume {field} is invalid")
    for field in ["capture_identity", "evaluation_capture_identity"]:
        if type(warm.get(field)) is not int or warm[field] <= 0:
            raise LargerEvidenceError(f"larger Transformer warm resume {field} is invalid")
    if (
        warm["capture_identity"] != scoreboard_main.get("capture_identity")
        or warm["evaluation_capture_identity"]
        != scoreboard["evaluation"].get("capture_identity")
    ):
        raise LargerEvidenceError("larger Transformer warm resume identities differ from scoreboard")
    probe = objective.get("gradient_probe")
    if not isinstance(probe, dict):
        raise LargerEvidenceError("larger Transformer gradient probe is absent")
    if (
        probe.get("replay") != 1
        or probe.get("accumulation_index") != 1
        or probe.get("did_update") is not False
        or not same_float(probe.get("dropout_probability", math.nan), 0.0)
        or probe.get("valid_token_count") != 22
    ):
        raise LargerEvidenceError("larger Transformer gradient replay contract is invalid")
    if probe.get("active_parameter_count") != 35 or probe.get("active_coordinate_count") != 1888:
        raise LargerEvidenceError("larger Transformer gradient inventory is invalid")
    if probe.get("tied_output_head_is_canonical_alias") is not True:
        raise LargerEvidenceError("larger Transformer tied output ownership is invalid")
    if probe.get("frozen_position_is_absent") is not True:
        raise LargerEvidenceError("larger Transformer frozen-position ownership is invalid")
    if probe.get("native_preparation_fallback_count") != 0:
        raise LargerEvidenceError("larger Transformer gradient probe prepared fallback")
    if probe.get("native_replay_fallback_count") != 0:
        raise LargerEvidenceError("larger Transformer gradient probe executed fallback")
    projections = probe.get("projections")
    if (
        not isinstance(projections, list)
        or any(not isinstance(item, dict) for item in projections)
        or [item.get("direction_id") for item in projections] != [
        "alternating_dense_v1",
        "seven_phase_dense_v1",
        ]
    ):
        raise LargerEvidenceError("larger Transformer gradient directions are invalid")
    for projection in projections:
        numbers = [
            projection.get("epsilon"),
            projection.get("expected_projection"),
            projection.get("actual_projection"),
            projection.get("absolute_error"),
            projection.get("tolerance"),
        ]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in numbers):
            raise LargerEvidenceError("larger Transformer gradient projection is non-finite")
        if not same_float(projection["epsilon"], 0.04):
            raise LargerEvidenceError("larger Transformer gradient epsilon policy is invalid")
        expected_error = abs(projection["actual_projection"] - projection["expected_projection"])
        expected_tolerance = 0.03 * max(
            1.0,
            abs(projection["actual_projection"]),
            abs(projection["expected_projection"]),
        )
        if not same_float(projection["absolute_error"], expected_error):
            raise LargerEvidenceError("larger Transformer gradient error is not authenticated")
        if not same_float(projection["tolerance"], expected_tolerance):
            raise LargerEvidenceError("larger Transformer gradient tolerance is not authenticated")
        if projection.get("mutation_sensitive") is not True:
            raise LargerEvidenceError("larger Transformer gradient projection is not mutation-sensitive")
        if abs(projection["expected_projection"]) <= projection["tolerance"]:
            raise LargerEvidenceError("larger Transformer expected projection is not informative")
        if abs(projection["actual_projection"]) <= projection["tolerance"]:
            raise LargerEvidenceError("larger Transformer actual projection is not informative")
        if expected_error > expected_tolerance and not same_float(expected_error, expected_tolerance):
            raise LargerEvidenceError("larger Transformer gradient projection exceeds tolerance")

    if not comparison_projection:
        return {
            "objective": objective,
            "scoreboard": scoreboard,
            "warm_resume": warm,
        }

    stable_objective = copy.deepcopy(objective)
    stable_objective.pop("git_sha")
    stable_warm = stable_objective["warm_resume"]
    for field in (
        "artifact_decode_wall_time_ns",
        "owner_restore_wall_time_ns",
        "preparation_wall_time_ns",
    ):
        stable_warm.pop(field)
    stable_preparation = stable_warm["preparation"]
    stable_warm["preparation"] = {
        "roles": {
            role: sorted(stable_preparation[role])
            for role in ("main", "accumulation", "partial_flush", "zero_grad", "evaluation")
        },
        "prepare_finalization": _timing_shape(
            stable_preparation["prepare_finalization"]
        ),
    }
    return {
        "stable_objective": stable_objective,
        "stable_scoreboard": _stable_scoreboard(scoreboard),
        "warm_timing": _warm_timing_projection(warm),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", required=True)
    parser.add_argument("--scoreboard", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--resume-bundle", required=True)
    parser.add_argument("--module-checkpoint", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    script_dir = pathlib.Path(__file__).resolve().parent
    try:
        validate_larger_evidence(
            pathlib.Path(arguments.objective).resolve(),
            pathlib.Path(arguments.scoreboard).resolve(),
            arguments.expected_sha,
            pathlib.Path(arguments.resume_bundle).resolve(),
            pathlib.Path(arguments.module_checkpoint).resolve(),
            script_dir / "native_training_preparation_evidence.py",
        )
    except (LargerEvidenceError, OSError, json.JSONDecodeError) as error:
        print(f"larger Transformer evidence is invalid: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
