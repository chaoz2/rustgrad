#!/usr/bin/env python3
"""Bind exact comparison revisions to the just-built release executables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess

SHA_RE = re.compile(r"[0-9a-f]{40}")
TARGETS = {
    "steady-replay": "compiled_transformer_train_resume",
    "warm-resume": "compiled_transformer_scale_evidence",
}


def sha256(path: pathlib.Path) -> str:
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"build executable is not an absolute executable regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--baseline-binary", required=True)
    parser.add_argument("--candidate-binary", required=True)
    parser.add_argument("--measurement-mode", choices=TARGETS, default="steady-replay")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    for label, value in (("baseline", arguments.baseline_sha), ("candidate", arguments.candidate_sha)):
        if SHA_RE.fullmatch(value) is None:
            raise SystemExit(f"{label} SHA must be lowercase and full")
    output = pathlib.Path(arguments.output).resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit("build manifest output already exists")
    binaries = {
        "baseline": pathlib.Path(arguments.baseline_binary).resolve(),
        "candidate": pathlib.Path(arguments.candidate_binary).resolve(),
    }
    rustc = subprocess.run(
        ["rustc", "-Vv"], check=True, capture_output=True, text=True
    ).stdout.rstrip()
    host = next(
        (line.removeprefix("host: ") for line in rustc.splitlines() if line.startswith("host: ")),
        None,
    )
    if not host:
        raise SystemExit("rustc host triple is unavailable")
    value = {
        "format_version": 2,
        "evidence_kind": "native_cpu_transformer_training_comparison_builds",
        "build_profile": "release",
        "cargo_locked": True,
        "cargo_incremental": os.environ.get("CARGO_INCREMENTAL"),
        "cargo_build_jobs": os.environ.get("CARGO_BUILD_JOBS"),
        "measurement_mode": arguments.measurement_mode,
        "cargo_target": f"example:{TARGETS[arguments.measurement_mode]}",
        "rustflags": os.environ.get("RUSTFLAGS"),
        "toolchain": rustc,
        "target_triple": host,
        "revisions": {
            role: {
                "source_sha": arguments.baseline_sha if role == "baseline" else arguments.candidate_sha,
                "executable_sha256": sha256(path),
            }
            for role, path in binaries.items()
        },
    }
    if (
        value["cargo_incremental"] != "0"
        or value["cargo_build_jobs"] != "2"
        or value["rustflags"] != "-D warnings"
    ):
        raise SystemExit("comparison build environment differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
