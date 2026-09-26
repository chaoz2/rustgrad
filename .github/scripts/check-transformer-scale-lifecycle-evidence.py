#!/usr/bin/env python3
"""Fail-closed validation for the protected larger Transformer lifecycle receipt."""

import json
import math
import pathlib
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def exact_keys(value: dict, expected: set[str], label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == expected, f"{label} fields differ: {sorted(value)}")


def main() -> None:
    require(len(sys.argv) == 2, "usage: check-transformer-scale-lifecycle-evidence.py FILE")
    path = pathlib.Path(sys.argv[1])
    raw = path.read_bytes()
    require(raw.endswith(b"\n") and raw.count(b"\n") == 1, "evidence is not compact canonical JSON")
    evidence = json.loads(raw)
    exact_keys(
        evidence,
        {
            "schema_version",
            "workload",
            "native",
            "replay_progress",
            "evaluation_trajectory",
            "pending_resume",
            "continuation",
        },
        "evidence",
    )
    require(evidence["schema_version"] == 1, "unsupported lifecycle evidence version")

    workload = evidence["workload"]
    exact_keys(
        workload,
        {
            "batch",
            "time",
            "vocabulary",
            "embedding",
            "heads",
            "feed_forward",
            "blocks",
            "accumulation_steps",
            "replays",
        },
        "workload",
    )
    require(
        workload
        == {
            "batch": 4,
            "time": 8,
            "vocabulary": 16,
            "embedding": 8,
            "heads": 2,
            "feed_forward": 32,
            "blocks": 2,
            "accumulation_steps": 2,
            "replays": 6,
        },
        "workload facts differ",
    )

    native = evidence["native"]
    exact_keys(
        native,
        {
            "compile_count",
            "capture_identity",
            "evaluation_capture_identity",
            "prepared_roles",
            "initial_prepared_role_count",
            "initial_fallback_count",
            "uninterrupted_replay_fallback_count",
            "restored_prepared_role_count",
            "restored_fallback_count",
            "restored_replay_fallback_count",
        },
        "native",
    )
    require(native["compile_count"] == 1, "compile count differs")
    require(native["capture_identity"] > 0, "training capture identity is absent")
    require(native["evaluation_capture_identity"] > 0, "evaluation capture identity is absent")
    require(native["capture_identity"] != native["evaluation_capture_identity"], "capture identities alias")
    require(
        native["prepared_roles"]
        == ["main", "accumulation", "partial_flush", "zero_grad", "evaluation"],
        "prepared role names or order differ",
    )
    require(native["initial_prepared_role_count"] == 5, "initial role inventory differs")
    require(native["initial_fallback_count"] == 0, "initial preparation used fallback")
    require(native["uninterrupted_replay_fallback_count"] == 0, "uninterrupted replay used fallback")
    require(native["restored_prepared_role_count"] == 5, "restored role inventory differs")
    require(native["restored_fallback_count"] == 0, "restored preparation used fallback")
    require(native["restored_replay_fallback_count"] == 0, "restored replay used fallback")

    progress = evidence["replay_progress"]
    require(isinstance(progress, list) and len(progress) == 6, "replay inventory differs")
    progress_keys = {"replay", "optimizer_step", "accumulation_index", "did_update"}
    for replay, item in enumerate(progress, 1):
        exact_keys(item, progress_keys, f"replay {replay}")
        require(
            item
            == {
                "replay": replay,
                "optimizer_step": replay // 2,
                "accumulation_index": replay % 2,
                "did_update": replay % 2 == 0,
            },
            f"replay {replay} progress differs",
        )

    trajectory = evidence["evaluation_trajectory"]
    require(isinstance(trajectory, list) and len(trajectory) == 4, "evaluation inventory differs")
    frontier_keys = {
        "replay",
        "optimizer_step",
        "accumulation_index",
        "token_mean_loss",
        "checkpoint_state_neutral",
        "samples",
    }
    sample_keys = {
        "batch_replay",
        "capture_identity",
        "token_mean_loss",
        "valid_token_count",
        "native_fallback_count",
        "executed_native_item_count",
        "module_dispatch_count",
    }
    for frontier, replay, optimizer_step in zip(trajectory, (0, 2, 4, 6), (0, 1, 2, 3)):
        exact_keys(frontier, frontier_keys, f"evaluation r{replay}")
        require(frontier["replay"] == replay, f"evaluation r{replay} replay differs")
        require(frontier["optimizer_step"] == optimizer_step, f"evaluation r{replay} step differs")
        require(frontier["accumulation_index"] == 0, f"evaluation r{replay} is pending")
        require(frontier["checkpoint_state_neutral"] is True, f"evaluation r{replay} mutated state")
        require(math.isfinite(frontier["token_mean_loss"]), f"evaluation r{replay} loss is nonfinite")
        samples = frontier["samples"]
        require(isinstance(samples, list) and len(samples) == 2, f"evaluation r{replay} samples differ")
        for sample, batch_replay, tokens in zip(samples, (1, 2), (22, 16)):
            exact_keys(sample, sample_keys, f"evaluation r{replay} sample {batch_replay}")
            require(sample["batch_replay"] == batch_replay, "evaluation sample batch differs")
            require(sample["capture_identity"] == native["evaluation_capture_identity"], "evaluation capture identity differs")
            require(math.isfinite(sample["token_mean_loss"]), "evaluation sample loss is nonfinite")
            require(sample["valid_token_count"] == tokens, "evaluation token weighting differs")
            require(sample["native_fallback_count"] == 0, "evaluation used fallback")
            require(sample["executed_native_item_count"] > 0, "evaluation did no native work")
            require(sample["module_dispatch_count"] > 0, "evaluation dispatched no module")
        recomputed = (
            samples[0]["token_mean_loss"] * 22.0 + samples[1]["token_mean_loss"] * 16.0
        ) / 38.0
        require(recomputed == frontier["token_mean_loss"], f"evaluation r{replay} weighting differs")
    require(
        trajectory[-1]["token_mean_loss"] < trajectory[0]["token_mean_loss"],
        "final evaluation loss did not decrease",
    )

    pending = evidence["pending_resume"]
    exact_keys(
        pending,
        {
            "replay",
            "optimizer_step",
            "accumulation_index",
            "checkpoint_bytes",
            "module_checkpoint_bytes",
            "resume_bundle_bytes",
            "resume_bundle_checksum_fnv1a64",
            "exact_encoded_round_trip",
        },
        "pending resume",
    )
    require((pending["replay"], pending["optimizer_step"], pending["accumulation_index"]) == (3, 1, 1), "pending frontier differs")
    require(pending["checkpoint_bytes"] > 0, "pending checkpoint is empty")
    require(pending["module_checkpoint_bytes"] > 0, "pending module checkpoint is empty")
    require(0 < pending["resume_bundle_bytes"] <= 256 * 1024 * 1024, "resume bundle bound differs")
    require(0 <= pending["resume_bundle_checksum_fnv1a64"] < 2**64, "resume checksum differs")
    require(pending["exact_encoded_round_trip"] is True, "encoded RGAB round trip was not proved")

    continuation = evidence["continuation"]
    exact_keys(
        continuation,
        {
            "replay_from",
            "replay_to",
            "exact_loss_and_checkpoint_continuation",
            "different_initialization",
            "evaluation_state_neutral",
            "tied_output_head_is_canonical",
            "frozen_position_state_preserved",
            "target_owned_module_published",
            "module_visit_count",
            "canonical_state_count",
        },
        "continuation",
    )
    require((continuation["replay_from"], continuation["replay_to"]) == (4, 6), "continuation range differs")
    for field in (
        "exact_loss_and_checkpoint_continuation",
        "different_initialization",
        "evaluation_state_neutral",
        "tied_output_head_is_canonical",
        "frozen_position_state_preserved",
        "target_owned_module_published",
    ):
        require(continuation[field] is True, f"continuation fact {field} is false")
    require(continuation["module_visit_count"] == 37, "module state inventory differs")
    require(continuation["canonical_state_count"] == 36, "canonical state inventory differs")


if __name__ == "__main__":
    main()
