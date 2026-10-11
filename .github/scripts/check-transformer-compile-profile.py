#!/usr/bin/env python3
"""Validate compile-only workload facts and a separate Callgrind observation."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re

WORKLOAD = {
    "batch": 4, "time": 8, "vocabulary": 16, "embedding": 8,
    "heads": 2, "feed_forward": 32, "blocks": 2, "accumulation_steps": 2,
}
ROLES = ["main", "accumulation", "partial_flush", "zero_grad", "evaluation"]


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def validate_compile_only(value: object) -> dict:
    require(type(value) is dict, "compile evidence must be an object")
    expected = {
        "schema_version": 1,
        "evidence_kind": "compiled_transformer_compile_only",
        "workload": WORKLOAD,
        "compile_count": 1,
        "initial_replay_step": 0,
        "attached_roles": ROLES,
        "target_prepared": False,
        "replayed": False,
    }
    fields = set(expected) | {
        "capture_identity", "evaluation_capture_identity", "recurrent_state_count",
        "recurrent_state_bytes", "main_schedule_item_count",
    }
    require(set(value) == fields, "compile evidence fields differ")
    for name, item in expected.items():
        # JSON distinguishes Bool from integer counters; Python equality does not.
        require(type(value[name]) is type(item) and value[name] == item,
                f"compile evidence {name} differs")
    require(all(type(value["workload"][name]) is int for name in WORKLOAD),
            "workload dimensions must be integers")
    for name in fields - set(expected):
        require(type(value[name]) is int and 0 < value[name] < 2**64,
                f"compile evidence {name} is invalid")
    require(value["capture_identity"] != value["evaluation_capture_identity"],
            "training and evaluation identities must differ")
    return value


def validate_callgrind(text: str, command: str) -> int:
    lines = text.splitlines()
    require(lines[:1] == ["# callgrind format"], "Callgrind header differs")
    for prefix, expected in (("events:", "Ir"), ("cmd:", command)):
        values = [line[len(prefix):].strip() for line in lines if line.startswith(prefix)]
        require(values == [expected], f"Callgrind {prefix} differs")
    summaries = [line.removeprefix("summary:").strip()
                 for line in lines if line.startswith("summary:")]
    require(len(summaries) == 1 and re.fullmatch(r"[0-9]+", summaries[0]) is not None,
            "Callgrind instruction summary is missing or ambiguous")
    instructions = int(summaries[0])
    require(0 < instructions < 2**64, "Callgrind instruction summary is invalid")
    require(any(line.startswith("fn=") for line in lines),
            "Callgrind function inventory is absent")
    return instructions


def read_bounded(path: pathlib.Path, limit: int) -> bytes:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path.name}")
    require(0 < path.stat().st_size <= limit, f"file exceeds bound: {path.name}")
    data = path.read_bytes()
    require(0 < len(data) <= limit, f"file exceeds bound: {path.name}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    compile_command = commands.add_parser("compile")
    compile_command.add_argument("evidence", type=pathlib.Path)
    profile_command = commands.add_parser("profile")
    profile_command.add_argument("--root", type=pathlib.Path, required=True)
    profile_command.add_argument("--expected-sha", required=True)
    profile_command.add_argument("--binary", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.mode == "compile":
        validate_compile_only(json.loads(read_bounded(args.evidence, 1 << 20)))
        return
    require(re.fullmatch(r"[0-9a-f]{40}", args.expected_sha) is not None,
            "expected SHA must be a full lowercase commit")
    require(args.binary.is_absolute(), "binary must have an absolute path")
    evidence = read_bounded(args.root / "compile-only.json", 1 << 20)
    validate_compile_only(json.loads(evidence))
    profile = read_bounded(args.root / "callgrind.out", 64 << 20)
    instructions = validate_callgrind(profile.decode("utf-8"), f"{args.binary} compile-only")
    binary = read_bounded(args.binary, 512 << 20)
    versions = read_bounded(args.root / "provenance.txt", 1 << 20)
    for name in ("inclusive.txt", "exclusive.txt"):
        read_bounded(args.root / name, 64 << 20)
    manifest = {
        "schema_version": 1,
        "evidence_kind": "compiled_transformer_callgrind_profile",
        "source_sha": args.expected_sha,
        "binary_sha256": hashlib.sha256(binary).hexdigest(),
        "compile_evidence_sha256": hashlib.sha256(evidence).hexdigest(),
        "profile_sha256": hashlib.sha256(profile).hexdigest(),
        "provenance_sha256": hashlib.sha256(versions).hexdigest(),
        "instruction_reads": instructions,
        "scope": "initialization_compile_inspection_and_teardown",
        "trace_children": False,
        "timing_policy": "instrumented_instruction_counts_not_wall_clock_benchmark",
        "workload": WORKLOAD,
    }
    with (args.root / "profile-manifest.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
