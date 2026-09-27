"""Independent synthetic warm-resume evidence for the protected comparison suite.

These files model evidence, not executable checkpoints. The comparator promises
byte equality and declared training invariants; runtime checkpoint authentication
belongs to the workload that emits the evidence.
"""

import copy
import hashlib
import json
import pathlib
import tempfile


ROLES = ("main", "accumulation", "partial_flush", "zero_grad", "evaluation")


def documents(support, sha):
    duration = support.duration
    board = support.scoreboard(26)
    partition, programs = support.warm_preparation_partition()
    board.update(partition)
    board.update(programs)
    board["evaluation"].pop("recurrent_state_count")
    board["successful_replay_count"] = 6
    board["checkpoint"].update(replay_step=6, byte_count=12)
    phases = board["compile_phases"]
    phases["evaluation"] = {"wall_time": duration(1), "logical_schedule_item_count": 2}
    for role in ("main_capture", "accumulation_capture", "partial_flush", "zero_grad"):
        phases[role]["wall_time"] = duration(6 if role == "main_capture" else 7)
    board["compile_wall_time"] = duration(32)
    for role, count in (("warm_accumulation_only", 2), ("warm_optimizer_commit", 3)):
        for value in board["step_phases"][role].values():
            if isinstance(value, dict) and "sample_count" in value:
                value["sample_count"] = count

    warm = {
        "replay_from": 3, "replay_to": 6, "program_count": 5,
        "loaded_module_count": 5, "durable_artifact_cache_hit_count": 5,
        "durable_artifact_cache_miss_count": 0, "render_capsule_hit_count": 5,
        "render_capsule_miss_count": 0, "local_render_job_count": 0,
        "compiler_invocation_count": 0, "linker_invocation_count": 0,
        "fallback_count": 0, "module_visit_count": 37, "canonical_state_count": 36,
        "resume_bundle_bytes": 12, "module_checkpoint_bytes": 12,
        "artifact_decode_wall_time_ns": 20, "owner_restore_wall_time_ns": 30,
        "preparation_wall_time_ns": 110, "capture_identity": 1,
        "evaluation_capture_identity": 5,
    }
    for field in (
        "artifact_checkpoint_authenticated", "topology_authenticated",
        "different_initialization", "fresh_executor", "exact_continuation",
        "evaluation_state_neutral", "target_owned_module_published",
    ):
        warm[field] = True
    warm["preparation"] = {
        role: {
            "total_wall_time_ns": 10, "layout_wall_time_ns": 2,
            "render_wall_time_ns": 2, "compiler_wall_time_ns": 0,
            "load_wall_time_ns": 3, "residual_wall_time_ns": 3,
        }
        for role in ROLES
    }
    warm["preparation"].update({
        "summed_program_wall_time_ns": 50, "runtime_overhead_wall_time_ns": 70,
        "module_overlap_wall_time_ns": 10, "render_overlap_wall_time_ns": 0,
        "effective_render_wall_time_ns": 10, "effective_render_fraction": 10 / 110,
        "prepare_finalization": copy.deepcopy(partition["prepare_finalization"]),
    })
    trajectory = []
    for replay, loss in ((0, 3.0), (2, 2.9), (4, 2.8), (6, 2.7)):
        trajectory.append({
            "replay": replay, "optimizer_step": replay // 2, "accumulation_index": 0,
            "checkpoint_state_neutral": True, "token_mean_loss": loss,
            "samples": [
                {"batch_replay": index, "capture_identity": 5, "valid_token_count": tokens,
                 "token_mean_loss": loss, "native_fallback_count": 0,
                 "executed_native_item_count": 2, "module_dispatch_count": 1}
                for index, tokens in ((1, 22), (2, 16))
            ],
        })
    objective = {
        "schema_version": 8, "git_sha": sha,
        "workload": {
            "batch": 4, "time": 8, "vocabulary": 16, "embedding": 8,
            "heads": 2, "feed_forward": 32, "blocks": 2, "compile_count": 1,
            "gradient_accumulation_steps": 2, "replays": 6,
        },
        "objective": {
            "initial_token_mean_loss": 3.0, "final_token_mean_loss": 2.7,
            "decreased": True, "evaluation_trajectory": trajectory,
        },
        "progress": {
            "replays": [
                {"replay": replay, "optimizer_step": replay // 2,
                 "accumulation_index": replay % 2, "did_update": replay % 2 == 0}
                for replay in range(1, 7)
            ],
            "pending_resume_checkpoint": {
                "replay": 3, "optimizer_step": 1, "accumulation_index": 1,
                "bytes": 12, "module_checkpoint_bytes": 12,
            },
            "terminal_scoreboard_checkpoint": {
                "replay": 6, "optimizer_step": 3, "accumulation_index": 0, "bytes": 12,
            },
            "final_replay": 6, "final_optimizer_step": 3,
            "final_accumulation_index": 0, "exact_resume": True,
        },
        "native": {"fallback_count": 0, "successful_replay_count": 6,
                   "recurrent_state_count": 1, "recurrent_state_bytes": 4},
        "warm_resume": warm,
        "gradient_probe": {
            "replay": 1, "accumulation_index": 1, "did_update": False,
            "dropout_probability": 0.0, "valid_token_count": 22,
            "active_parameter_count": 35, "active_coordinate_count": 1888,
            "tied_output_head_is_canonical_alias": True, "frozen_position_is_absent": True,
            "native_preparation_fallback_count": 0, "native_replay_fallback_count": 0,
            "projections": [
                {"direction_id": direction, "epsilon": 0.04, "expected_projection": 1.0,
                 "actual_projection": 1.0, "absolute_error": 0.0,
                 "tolerance": 0.03, "mutation_sensitive": True}
                for direction in ("alternating_dense_v1", "seven_phase_dense_v1")
            ],
        },
    }
    return objective, board


def checksums(support, directory):
    lines = []
    for key, filename in support.VALIDATOR.WARM_EVIDENCE_FILES.items():
        if key != "checksums":
            digest = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            lines.append(f"{digest}  {filename}\n")
    (directory / "sha256.txt").write_text("".join(lines), encoding="utf-8")


def fixture(support, root):
    arguments = support.fixture(root)
    arguments.measurement_mode = "warm-resume"
    manifest_path = pathlib.Path(arguments.build_manifest)
    manifest = json.loads(manifest_path.read_text())
    manifest.update(measurement_mode="warm-resume", cargo_target="example:compiled_transformer_scale_evidence")
    support.write_json(manifest_path, manifest)
    for ordinal, (role, _) in enumerate(support.VALIDATOR.ORDER, start=1):
        directory = root / f"trial-{ordinal:02d}-{role}"
        sha = support.BASELINE_SHA if role == "baseline" else support.CANDIDATE_SHA
        objective, board = documents(support, sha)
        support.write_json(directory / "objective-evidence.json", objective)
        support.write_json(directory / "native-cpu-training-scoreboard.json", board)
        for name in ("replay-3-resume.rgab", "replay-3-module-checkpoint.safetensors"):
            (directory / name).write_bytes(b"synthetic-12")
        provenance_path = directory / "provenance.txt"
        lines = [line for line in provenance_path.read_text().splitlines()
                 if not line.startswith("steady_measurement_")]
        lines[1:1] = [
            "cargo_build_jobs=2", "workflow_timeout_minutes=75",
            "workload=batch4_time8_vocab16_embedding8_heads2_ff32_blocks2_replays6",
            "warm_resume=portable_rgab_replay3_to6_fresh_executor_same_temporary_cache",
            "timing_policy=observational_no_threshold",
        ]
        provenance_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        checksums(support, directory)
    return arguments


def check(support):
    cases = (
        "valid", "timing", "checkpoint", "capture", "cache-miss", "hardware", "missing",
        "malformed-projection", "malformed-dropout", "malformed-duration",
    )
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="rustgrad-warm-comparison-") as temporary:
            root = pathlib.Path(temporary).resolve()
            arguments = fixture(support, root)
            trial = root / "trial-02-candidate"
            objective_path = trial / "objective-evidence.json"
            objective = json.loads(objective_path.read_text())
            if case == "timing":
                objective["warm_resume"]["artifact_decode_wall_time_ns"] += 10
            elif case == "capture":
                objective["warm_resume"]["capture_identity"] += 1
            elif case == "cache-miss":
                objective["warm_resume"]["render_capsule_miss_count"] = 1
            elif case == "checkpoint":
                (trial / "replay-3-resume.rgab").write_bytes(b"different-12")
            elif case == "hardware":
                path = trial / "provenance.txt"
                path.write_text(path.read_text().replace("cpu_model_name=synthetic", "cpu_model_name=different"))
            elif case == "malformed-projection":
                objective["gradient_probe"]["projections"][0] = None
            elif case == "malformed-dropout":
                objective["gradient_probe"]["dropout_probability"] = "0.0"
            elif case == "malformed-duration":
                path = trial / "native-cpu-training-scoreboard.json"
                board = json.loads(path.read_text())
                board["compile_phases"]["main_capture"]["recurrent_capture"][
                    "alias_planning_wall_time"
                ] = None
                support.write_json(path, board)
            support.write_json(objective_path, objective)
            checksums(support, trial)
            if case == "missing":
                objective_path.unlink()
            manifest, comparable = support.VALIDATOR.compare(arguments)
            assert comparable is (case in ("valid", "timing")), (case, manifest)
            assert len(manifest["trials"]) == 4
            if case in ("capture", "cache-miss", "missing") or case.startswith("malformed-"):
                assert manifest["trials"][1]["status"] == "invalid", case
                assert manifest["trials"][1]["files"], case
