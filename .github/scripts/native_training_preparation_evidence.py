"""Shared validation for native-training preparation timing evidence."""

from __future__ import annotations

from typing import Any, Mapping

MAX_DURATION_SECS = (1 << 64) - 1
MAX_DURATION_NS = MAX_DURATION_SECS * 1_000_000_000 + 999_999_999


class PreparationEvidenceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationEvidenceError(message)


def duration_ns(value: Any, label: str) -> int:
    require(
        type(value) is dict and set(value) == {"secs", "nanos"},
        f"{label} duration differs",
    )
    secs = value["secs"]
    nanos = value["nanos"]
    require(type(secs) is int and secs >= 0, f"{label}.secs must be a nonnegative integer")
    require(type(nanos) is int and nanos >= 0, f"{label}.nanos must be a nonnegative integer")
    require(secs <= MAX_DURATION_SECS, f"{label}.secs is out of range")
    require(nanos < 1_000_000_000, f"{label}.nanos is out of range")
    return secs * 1_000_000_000 + nanos


def checked_duration_sum(values: list[int], label: str) -> int:
    total = sum(values)
    require(total <= MAX_DURATION_NS, f"{label} duration overflows")
    return total


def validate_preparation_finalization(
    scoreboard: Mapping[str, Any],
    format_version: int,
    programs: Mapping[str, Any],
) -> None:
    """Validate the v24/v25 preparation timing wire contract and partitions."""

    if format_version == 24:
        require(
            "prepare_finalization" not in scoreboard,
            "scoreboard v24 unexpectedly has preparation finalization",
        )
        return
    require(format_version == 25, "scoreboard preparation format differs")

    finalization = scoreboard.get("prepare_finalization")
    require(type(finalization) is dict, "scoreboard preparation finalization is absent")
    require(type(programs.get("main")) is dict, "scoreboard main program is absent")
    present_roles = [role for role, program in programs.items() if type(program) is dict]
    expected_fields = {
        "instrumented_wall_time",
        "outer_remainder_wall_time",
        "bootstrap_wall_time",
        "report_input_assembly_wall_time",
        "unattributed_wall_time",
        *present_roles,
    }
    require(
        set(finalization) == expected_fields,
        "scoreboard preparation finalization fields differ",
    )

    measured_program_finalization = 0
    for role in present_roles:
        phase = finalization[role]
        require(type(phase) is dict, f"scoreboard {role} finalization differs")
        expected_phase_fields = {
            "pre_layout_admission_wall_time",
            "workspace_construction_wall_time",
        }
        if role != "evaluation":
            expected_phase_fields.add("recurrent_finalization_wall_time")
        require(
            set(phase) == expected_phase_fields,
            f"scoreboard {role} finalization fields differ",
        )
        measured_program_finalization = checked_duration_sum(
            [
                measured_program_finalization,
                *[
                    duration_ns(phase[field], f"scoreboard {role}.{field}")
                    for field in sorted(expected_phase_fields)
                ],
            ],
            "scoreboard program finalization",
        )

    prepare_wall_time = duration_ns(
        scoreboard.get("prepare_wall_time"), "scoreboard prepare_wall_time"
    )
    prepare_overhead = duration_ns(
        scoreboard.get("prepare_runtime_overhead_wall_time"),
        "scoreboard prepare_runtime_overhead_wall_time",
    )
    instrumented = duration_ns(
        finalization["instrumented_wall_time"],
        "scoreboard finalization.instrumented_wall_time",
    )
    outer_remainder = duration_ns(
        finalization["outer_remainder_wall_time"],
        "scoreboard finalization.outer_remainder_wall_time",
    )
    bootstrap = duration_ns(
        finalization["bootstrap_wall_time"],
        "scoreboard finalization.bootstrap_wall_time",
    )
    assembly = duration_ns(
        finalization["report_input_assembly_wall_time"],
        "scoreboard finalization.report_input_assembly_wall_time",
    )
    unattributed = duration_ns(
        finalization["unattributed_wall_time"],
        "scoreboard finalization.unattributed_wall_time",
    )
    require(
        checked_duration_sum(
            [instrumented, outer_remainder], "scoreboard whole preparation"
        )
        == prepare_wall_time,
        "scoreboard instrumented preparation does not partition caller time",
    )
    require(
        checked_duration_sum(
            [
                outer_remainder,
                bootstrap,
                measured_program_finalization,
                assembly,
                unattributed,
            ],
            "scoreboard preparation overhead",
        )
        == prepare_overhead,
        "scoreboard finalization does not partition preparation overhead",
    )

    program_totals = []
    for role, program in programs.items():
        if type(program) is not dict:
            continue
        timing = program.get("preparation_timing")
        require(type(timing) is dict, f"scoreboard {role} preparation timing is absent")
        program_totals.append(
            duration_ns(
                timing.get("total"),
                f"scoreboard {role}.preparation_timing.total",
            )
        )
    program_total = checked_duration_sum(
        program_totals, "scoreboard program preparation"
    )
    module_overlap = duration_ns(
        scoreboard.get("prepare_parallel_module_overlap_wall_time"),
        "scoreboard prepare_parallel_module_overlap_wall_time",
    )
    render_overlap = duration_ns(
        scoreboard.get("prepare_parallel_render_overlap_wall_time"),
        "scoreboard prepare_parallel_render_overlap_wall_time",
    )
    require(
        module_overlap + render_overlap <= program_total,
        "scoreboard preparation overlap exceeds program time",
    )
    effective_program = program_total - module_overlap - render_overlap
    require(
        checked_duration_sum(
            [effective_program, prepare_overhead], "scoreboard preparation partition"
        )
        == prepare_wall_time,
        "scoreboard program and overhead preparation do not partition caller time",
    )
    require(
        checked_duration_sum(
            [
                effective_program,
                bootstrap,
                measured_program_finalization,
                assembly,
                unattributed,
            ],
            "scoreboard instrumented preparation",
        )
        == instrumented,
        "scoreboard finalization does not partition instrumented time",
    )
