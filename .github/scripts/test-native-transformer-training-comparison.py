#!/usr/bin/env python3
"""Synthetic protected-CI checks for the comparison evidence validator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import stat
import tempfile
from types import SimpleNamespace

SCRIPT = pathlib.Path(__file__).with_name("check-native-transformer-training-comparison.py")
SPEC = importlib.util.spec_from_file_location("native_training_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)

BASELINE_SHA = "1" * 40
CANDIDATE_SHA = "2" * 40


def duration(nanos: int) -> dict[str, int]:
    return {"secs": nanos // 1_000_000_000, "nanos": nanos % 1_000_000_000}


def traffic() -> dict[str, int]:
    return {
        "external_input_import_count": 0,
        "external_input_import_bytes": 0,
        "borrowed_recurrent_input_bytes": 4,
        "borrowed_recurrent_output_bytes": 4,
        "retained_recurrent_state_count": 0,
        "retained_recurrent_state_bytes": 0,
        "replaced_recurrent_state_count": 1,
        "replaced_recurrent_state_bytes": 4,
        "materialized_egress_count": 1,
        "materialized_egress_bytes": 4,
    }


def steady_program(capture_identity: int) -> dict[str, object]:
    return {
        "capture_identity": capture_identity,
        "native_identity": capture_identity + 10,
        "vectorized": True,
        "native_item_count": 2,
        "executed_native_item_count": 1,
        "module_dispatch_count": 1,
        "module_dispatched_native_item_count": 1,
        "schedule_cache_keys": [capture_identity, capture_identity + 1],
        "traffic": traffic(),
    }


def checkpoint(replay: int, optimizer: int, dropout: int) -> dict[str, object]:
    return {
        "capture_identity": 1,
        "accumulation_capture_identity": 2,
        "replay_step": replay,
        "optimizer_step": optimizer,
        "accumulation_index": 0,
        "dropout_block_counter": dropout,
    }


def sidecar(source_sha: str) -> dict[str, object]:
    accumulation = []
    commits = []
    for ordinal in range(1, 97):
        microbatch = (ordinal - 1) % 3 + 1
        sample = {
            "measurement_ordinal": ordinal,
            "window_ordinal": (ordinal - 1) // 3 + 1,
            "microbatch_ordinal": microbatch,
            "replay_step": 6 + ordinal,
            "successful_invocation": 3 + ordinal,
            "valid_token_count": (5, 3, 3)[microbatch - 1],
            "total_wall_time": duration(5 + ordinal),
            "executor_wall_time": duration(4 + ordinal),
            "native_dispatcher_wall_time": duration(2),
            "executor_host_wall_time": duration(2 + ordinal),
            "recurrent_overhead_wall_time": duration(1),
        }
        (commits if microbatch == 3 else accumulation).append(sample)
    return {
        "format_version": 1,
        "evidence_kind": "native_cpu_compiled_transformer_steady_replay",
        "provenance": {"git_sha": source_sha, "cargo_profile": "release"},
        "warmup_window_count": 1,
        "warmup_replay_count": 3,
        "measured_window_count": 32,
        "measured_replay_count": 96,
        "validated_replay_count": 99,
        "validated_valid_token_count": 363,
        "measured_valid_token_count": 352,
        "compiled_graph_build_count": 1,
        "dropout_blocks_per_replay": 42,
        "final_checkpoint_byte_count": 13740,
        "preparation": {
            "cache_scope": "same_process_warm_cache",
            "wall_time": duration(10),
            "render_capsule_hit_count": 4,
            "render_capsule_miss_count": 0,
            "local_render_job_count": 0,
            "max_parallel_render_job_count": 0,
            "compiler_process_count": 0,
        },
        "captured_schedule": {
            "base_bits": 1028443341,
            "gamma_bits": 1056964608,
            "milestones": [1],
        },
        "starting_checkpoint": checkpoint(3, 1, 126),
        "warmup_checkpoint": checkpoint(6, 2, 252),
        "final_checkpoint": checkpoint(102, 34, 4284),
        "accumulation_only_program": steady_program(2),
        "optimizer_commit_program": steady_program(1),
        "accumulation_only_samples": accumulation,
        "optimizer_commit_samples": commits,
    }


def scoreboard_program(capture_identity: int) -> dict[str, object]:
    return {
        "capture_identity": capture_identity,
        "native_identity": capture_identity + 10,
        "vectorized": True,
        "execution_plan_identity": capture_identity + 20,
        "logical_schedule_item_count": 2,
        "recurrent_state_count": 1,
        "peak_logical_temporary_allocation_count": 1,
        "peak_logical_temporary_bytes": 4,
        "native_item_count": 2,
        "cache_hit_count": 0,
        "cache_miss_count": 2,
        "rendered_entry_count": 1,
        "rendered_source_bytes": 16,
        "loaded_module_count": 1,
        "referenced_module_count": 1,
        "unique_rendered_entry_count": 1,
        "shared_prefix_entry_count": 0,
        "shared_prefix_source_bytes": 0,
        "shared_prefix_source_program_index": None,
        "shared_prefix_source_native_identity": None,
        "durable_artifact_cache_hit_count": 0,
        "durable_artifact_cache_miss_count": 1,
        "compiler_invocation_count": 1,
        "combined_compile_link_count": 1,
        "object_compile_count": 1,
        "linker_invocation_count": 0,
        "dispatch_segmentation": {"segment_count": 1},
        "preparation_timing": {"total_wall_time": duration(10)},
    }


def recurrent_capture(preview_count: int, state_count: int, cursor: bool) -> dict[str, object]:
    value = {
        "alias_planning_wall_time": duration(1),
        "preview_schedule_count": preview_count,
        "final_schedule_wall_time": duration(1),
        "pure_capture_binding_wall_time": duration(1),
        "effect_assembly_sealing_wall_time": duration(1),
        "recurrent_authentication_wall_time": duration(1),
        "residual_wall_time": duration(1),
        "recurrent_state_count": state_count,
    }
    if cursor:
        value["cursor_projection_wall_time"] = duration(1)
    return value


def warm_step_phase() -> dict[str, object]:
    def summary(nanos: int) -> dict[str, object]:
        return {
            "sample_count": 1,
            "min": duration(nanos),
            "nearest_rank_p50": duration(nanos),
            "nearest_rank_p95": duration(nanos),
            "max": duration(nanos),
        }

    return {
        "total_wall_time": duration(7),
        "wall_time": summary(7),
        "steps_per_second": 1_000_000_000 / 7,
        "executor_total_wall_time": duration(3),
        "executor_wall_time": summary(3),
        "native_dispatcher_total_wall_time": duration(1),
        "native_dispatcher_wall_time": summary(1),
        "executor_host_total_wall_time": duration(2),
        "executor_host_wall_time": summary(2),
        "recurrent_overhead_total_wall_time": duration(4),
        "recurrent_overhead_wall_time": summary(4),
    }


def scoreboard() -> dict[str, object]:
    programs = {
        "main": scoreboard_program(1),
        "accumulation": scoreboard_program(2),
        "partial_flush": scoreboard_program(3),
        "zero_grad": scoreboard_program(4),
    }
    return {
        "format_version": 24,
        "prepare_max_parallel_render_job_count": 2,
        "prepare_render_capsule_hit_count": 0,
        "prepare_render_capsule_miss_count": 4,
        "prepare_local_render_job_count": 4,
        "prepare_compiler_process_count": 6,
        "prepare_max_parallel_compiler_process_count": 2,
        "compile_phases": {
            "compile_count": 1,
            "objective_forward": {"wall_time": duration(1), "graph_node_count": 1},
            "autograd": {"wall_time": duration(1), "graph_node_count": 2},
            "optimizer_lowering": {"wall_time": duration(1), "graph_node_count": 3},
            "main_capture": {
                "wall_time": duration(1),
                "logical_schedule_item_count": 2,
                "recurrent_capture": recurrent_capture(2, 1, False),
            },
            "accumulation_capture": {
                "wall_time": duration(1),
                "logical_schedule_item_count": 2,
                "recurrent_capture": recurrent_capture(3, 1, True),
            },
            "partial_flush": {
                "wall_time": duration(1),
                "logical_schedule_item_count": 2,
                "recurrent_capture": recurrent_capture(3, 1, True),
            },
            "zero_grad": {
                "wall_time": duration(1),
                "logical_schedule_item_count": 2,
                "recurrent_capture": recurrent_capture(1, 1, True),
            },
            "residual_wall_time": duration(1),
        },
        "prepare_compiler_process_timings": [
            {
                "program_index": index,
                "native_identity": index + 10,
                "process": {"kind": "object", "ordinal": index},
                "rendered_source_bytes": 16,
                "permit_request_offset": duration(1),
                "permit_wait": duration(1),
                "process_wall_time": duration(1),
            }
            for index in range(6)
        ],
        "prepare_compiler_critical_tail": {
            "program_index": 3,
            "native_identity": 13,
            "process": {"kind": "combined"},
            "finish_offset": duration(1),
            "post_main_tail": duration(1),
        },
        "step_phases": {
            "first": {
                "phase": "accumulation_only",
                "total_wall_time": duration(7),
                "executor_wall_time": duration(3),
                "native_dispatcher_wall_time": duration(1),
                "executor_host_wall_time": duration(2),
                "recurrent_overhead_wall_time": duration(4),
            },
            "warm_accumulation_only": warm_step_phase(),
            "warm_optimizer_commit": warm_step_phase(),
        },
        "initial_replay_step": 0,
        "successful_replay_count": 3,
        **programs,
        "evaluation": None,
        "recurrent_logical_state_count": 1,
        "recurrent_logical_state_bytes": 4,
        "main_replay_traffic": traffic(),
        "main_replay_executed_native_item_count": 1,
        "accumulation_replay_traffic": traffic(),
        "accumulation_replay_executed_native_item_count": 1,
        "schedule_cache_keys": [1, 2],
        "accumulation_schedule_cache_keys": [2, 3],
        "checkpoint": {
            "capture_identity": 1,
            "replay_step": 3,
            "byte_count": 13740,
            "wall_time": duration(1),
        },
        "fallback_count": 0,
        "kernel_launch_count": None,
        "host_to_device": None,
        "device_to_host": None,
    }


def provenance(source_sha: str, binary_digest: str) -> str:
    return f"""schema_version=1
git_sha={source_sha}
prebuilt_binary_source_sha={source_sha}
prebuilt_binary_sha256={binary_digest}
cargo_profile=release
temporary_cache=fresh_sha_scoped
steady_measurement_file=native-cpu-training-steady-replays.json
steady_measurement_cache_scope=same_process_warm_cache
steady_measurement_warmup_windows=1
steady_measurement_measured_windows=32
runner_os=Linux
runner_arch=X64
runner_image_os=ubuntu24
runner_image_version=synthetic
cpu_hardware_policy=required_normalized_lscpu_v1

[cpu_hardware]
cpu_model_name=synthetic
cpu_socket_count=1
cpu_cores_per_socket=4
cpu_threads_per_core=2
cpu_logical_count=8
cpu_vendor_id=synthetic
cpu_family=1
cpu_model=1
cpu_stepping=1
cpu_numa_node_count=1

[rustc]
rustc synthetic
host: x86_64-unknown-linux-gnu

[cargo]
cargo synthetic

[c_compiler]
cc synthetic
"""


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def fixture(root: pathlib.Path) -> SimpleNamespace:
    binaries = {}
    digests = {}
    for role in ("baseline", "candidate"):
        path = root / f"{role}-binary"
        path.write_bytes(f"synthetic-{role}".encode())
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        binaries[role] = path
        digests[role] = hashlib.sha256(path.read_bytes()).hexdigest()
    for ordinal, (role, _) in enumerate(VALIDATOR.ORDER, start=1):
        source_sha = BASELINE_SHA if role == "baseline" else CANDIDATE_SHA
        directory = root / f"trial-{ordinal:02d}-{role}"
        directory.mkdir()
        write_json(directory / "native-cpu-training-scoreboard.json", scoreboard())
        write_json(directory / "native-cpu-training-steady-replays.json", sidecar(source_sha))
        (directory / "provenance.txt").write_text(provenance(source_sha, digests[role]), encoding="utf-8")
    build_manifest = root / "build-manifest.json"
    write_json(
        build_manifest,
        {
            "format_version": 1,
            "evidence_kind": "native_cpu_transformer_training_comparison_builds",
            "build_profile": "release",
            "cargo_locked": True,
            "cargo_incremental": "0",
            "cargo_build_jobs": "2",
            "cargo_target": "example:compiled_transformer_train_resume",
            "rustflags": "-D warnings",
            "toolchain": "rustc synthetic\nhost: x86_64-unknown-linux-gnu",
            "target_triple": "x86_64-unknown-linux-gnu",
            "revisions": {
                role: {
                    "source_sha": BASELINE_SHA if role == "baseline" else CANDIDATE_SHA,
                    "executable_sha256": digests[role],
                }
                for role in ("baseline", "candidate")
            },
        },
    )
    return SimpleNamespace(
        root=str(root),
        baseline_sha=BASELINE_SHA,
        candidate_sha=CANDIDATE_SHA,
        baseline_binary=str(binaries["baseline"]),
        candidate_binary=str(binaries["candidate"]),
        build_manifest=str(build_manifest),
        output=str(root / "comparison.json"),
    )


def run_case(mutator=None, expected: bool = True, invalid_trial: int | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="rustgrad-native-comparison-") as temporary:
        root = pathlib.Path(temporary).resolve()
        arguments = fixture(root)
        if mutator is not None:
            mutator(root)
        try:
            manifest, comparable = VALIDATOR.compare(arguments)
        except VALIDATOR.EvidenceError:
            assert expected is False
            return
        assert comparable is expected
        assert manifest["status"] == ("comparable" if expected else "incomparable")
        if invalid_trial is not None:
            trial = manifest["trials"][invalid_trial - 1]
            assert trial["status"] == "invalid"
            assert trial["invalid_reason"]
            assert set(trial["files"]) == {"scoreboard", "steady", "provenance"}


def main() -> None:
    run_case()

    def change_timing_only(root: pathlib.Path) -> None:
        path = root / "trial-02-candidate/native-cpu-training-scoreboard.json"
        value = json.loads(path.read_text())
        first = value["step_phases"]["first"]
        first["native_dispatcher_wall_time"] = duration(100_000_000)
        first["executor_host_wall_time"] = duration(200_000_000)
        first["executor_wall_time"] = duration(300_000_000)
        first["recurrent_overhead_wall_time"] = duration(700_000_000)
        first["total_wall_time"] = duration(1_000_000_000)
        warm = value["step_phases"]["warm_optimizer_commit"]
        warm["steps_per_second"] = 1.0
        for field, nanos in (
            ("wall_time", 1_000_000_000),
            ("executor_wall_time", 300_000_000),
            ("native_dispatcher_wall_time", 100_000_000),
            ("executor_host_wall_time", 200_000_000),
            ("recurrent_overhead_wall_time", 700_000_000),
        ):
            for summary_field in ("min", "nearest_rank_p50", "nearest_rank_p95", "max"):
                warm[field][summary_field] = duration(nanos)
        warm["total_wall_time"] = duration(1_000_000_000)
        warm["executor_total_wall_time"] = duration(300_000_000)
        warm["native_dispatcher_total_wall_time"] = duration(100_000_000)
        warm["executor_host_total_wall_time"] = duration(200_000_000)
        warm["recurrent_overhead_total_wall_time"] = duration(700_000_000)
        value["prepare_compiler_critical_tail"] = {
            "program_index": 1,
            "native_identity": 11,
            "process": {"kind": "object", "ordinal": 1},
            "finish_offset": duration(2),
            "post_main_tail": duration(1),
        }
        write_json(path, value)
        path = root / "trial-02-candidate/native-cpu-training-steady-replays.json"
        value = json.loads(path.read_text())
        sample = value["accumulation_only_samples"][0]
        sample["native_dispatcher_wall_time"] = duration(10)
        sample["executor_host_wall_time"] = duration(20)
        sample["executor_wall_time"] = duration(30)
        sample["recurrent_overhead_wall_time"] = duration(40)
        sample["total_wall_time"] = duration(70)
        write_json(path, value)

    run_case(change_timing_only)

    def change_phase(root: pathlib.Path) -> None:
        path = root / "trial-02-candidate/native-cpu-training-scoreboard.json"
        value = json.loads(path.read_text())
        value["step_phases"]["first"]["phase"] = "optimizer_commit"
        write_json(path, value)

    run_case(change_phase, expected=False, invalid_trial=2)

    def change_sample_count(root: pathlib.Path) -> None:
        path = root / "trial-02-candidate/native-cpu-training-scoreboard.json"
        value = json.loads(path.read_text())
        warm = value["step_phases"]["warm_optimizer_commit"]
        for field in (
            "wall_time",
            "executor_wall_time",
            "native_dispatcher_wall_time",
            "executor_host_wall_time",
            "recurrent_overhead_wall_time",
        ):
            warm[field]["sample_count"] = 2
        write_json(path, value)

    run_case(change_sample_count, expected=False, invalid_trial=2)

    def remove_warm_phase(root: pathlib.Path) -> None:
        path = root / "trial-02-candidate/native-cpu-training-scoreboard.json"
        value = json.loads(path.read_text())
        value["step_phases"]["warm_optimizer_commit"] = None
        write_json(path, value)

    run_case(remove_warm_phase, expected=False, invalid_trial=2)

    def change_traffic(root: pathlib.Path) -> None:
        path = root / "trial-02-candidate/native-cpu-training-steady-replays.json"
        value = json.loads(path.read_text())
        value["optimizer_commit_program"]["traffic"]["materialized_egress_bytes"] += 4
        write_json(path, value)

    run_case(change_traffic, expected=False)

    def change_stable_identity(root: pathlib.Path) -> None:
        path = root / "trial-02-candidate/native-cpu-training-steady-replays.json"
        value = json.loads(path.read_text())
        value["optimizer_commit_program"]["native_identity"] += 1
        write_json(path, value)

    run_case(change_stable_identity, expected=False)

    def change_compiler_identity(root: pathlib.Path) -> None:
        path = root / "trial-02-candidate/native-cpu-training-scoreboard.json"
        value = json.loads(path.read_text())
        value["prepare_compiler_process_timings"][0]["native_identity"] += 1
        write_json(path, value)

    run_case(change_compiler_identity, expected=False)

    def change_compiler_source_bytes(root: pathlib.Path) -> None:
        path = root / "trial-02-candidate/native-cpu-training-scoreboard.json"
        value = json.loads(path.read_text())
        value["prepare_compiler_process_timings"][0]["rendered_source_bytes"] += 1
        write_json(path, value)

    run_case(change_compiler_source_bytes, expected=False)

    def break_partition(root: pathlib.Path) -> None:
        path = root / "trial-03-candidate/native-cpu-training-steady-replays.json"
        value = json.loads(path.read_text())
        value["optimizer_commit_samples"][0]["total_wall_time"]["nanos"] = 1_000_000_000
        write_json(path, value)

    run_case(break_partition, expected=False, invalid_trial=3)

    def relabel_binary(root: pathlib.Path) -> None:
        path = root / "trial-04-baseline/provenance.txt"
        text = path.read_text()
        path.write_text(text.replace("prebuilt_binary_sha256=", "prebuilt_binary_sha256=" + "0" * 64 + "\nignored="), encoding="utf-8")

    run_case(relabel_binary, expected=False)

    def relabel_build(root: pathlib.Path) -> None:
        path = root / "build-manifest.json"
        value = json.loads(path.read_text())
        value["revisions"]["candidate"]["source_sha"] = BASELINE_SHA
        write_json(path, value)

    run_case(relabel_build, expected=False)


if __name__ == "__main__":
    main()
