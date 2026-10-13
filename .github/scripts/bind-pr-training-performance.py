#!/usr/bin/env python3
"""Bind optional unprivileged PR measurements to exact sources and run identity.

This sidecar is provenance, not a numerical validator or an automatic merge
decision. Reviewers must inspect the source, raw trials and comparison admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
from typing import Any

WORKFLOW = ".github/workflows/pr-training-performance.yml"
SHA = re.compile(r"[0-9a-f]{40}")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_EVIDENCE_FILES = 64
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
HARNESS_FILES = (
    WORKFLOW,
    ".github/scripts/run-native-transformer-training-comparison.sh",
    ".github/scripts/bind-pr-training-performance.py",
    ".github/scripts/check-native-transformer-training-comparison.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(value: Any, label: str) -> str:
    require(type(value) is str and SHA.fullmatch(value) is not None, f"invalid {label} SHA")
    return value


def positive_id(value: Any, label: str) -> int:
    require(type(value) is str and re.fullmatch(r"[1-9][0-9]*", value) is not None,
            f"invalid {label}")
    return int(value)


def git(source: pathlib.Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def source_binding(event: dict[str, Any], env: dict[str, str], workspace: pathlib.Path) -> dict[str, Any]:
    require(env["GITHUB_EVENT_NAME"] == "pull_request", "ordinary pull_request context required")
    require(event["action"] in {"opened", "reopened", "synchronize", "labeled", "ready_for_review"},
            "unsupported PR event action")
    if event["action"] == "labeled":
        require(event["label"]["name"] == "performance-evidence", "unrelated label event")
    pr = event["pull_request"]
    number = event["number"]
    require(type(number) is int and number > 0 and type(pr["number"]) is int and
            pr["number"] == number, "invalid PR number")
    require(pr["state"] == "open" and pr["draft"] is False, "open non-draft PR required")
    require(any(label["name"] == "performance-evidence" for label in pr["labels"]),
            "performance-evidence label required")
    repository = env["GITHUB_REPOSITORY"]
    require(REPOSITORY.fullmatch(repository) is not None, "invalid repository name")
    require(event["repository"]["full_name"] == pr["base"]["repo"]["full_name"] == repository,
            "base repository differs")
    require(pr["base"]["ref"] == "main", "main base required")
    require(env["GITHUB_REF"] == f"refs/pull/{number}/merge", "tested PR merge ref required")
    baseline = sha(pr["base"]["sha"], "base")
    head = sha(pr["head"]["sha"], "authored head")
    candidate = sha(env["GITHUB_SHA"], "tested merge")
    require(env["BASELINE_SHA"] == baseline and env["CANDIDATE_SHA"] == candidate,
            "comparison revisions differ from event")
    require(env["MEASUREMENT_MODE"] == "warm-resume", "fixed PR warm-resume workload required")
    require(baseline != candidate and head != candidate, "synthetic merge revision required")
    candidate_source = workspace / "candidate-source"
    baseline_source = workspace / "baseline-source"
    require(git(candidate_source, "rev-parse", "HEAD") == candidate, "candidate checkout differs")
    require(git(baseline_source, "rev-parse", "HEAD") == baseline, "baseline checkout differs")
    parents = git(candidate_source, "show", "-s", "--format=%P", candidate).split()
    require(parents == [baseline, head], "tested merge parents differ from event base/head")
    require(not git(candidate_source, "status", "--porcelain") and
            not git(baseline_source, "status", "--porcelain"), "source checkout is dirty")
    workflow_sha = sha(env["GITHUB_WORKFLOW_SHA"], "workflow source")
    workflow_ref = env["GITHUB_WORKFLOW_REF"]
    require(workflow_sha == candidate and
            workflow_ref == f"{repository}/{WORKFLOW}@{env['GITHUB_REF']}",
            "workflow must come from the exact tested merge")
    require(env["GITHUB_JOB"] == "compare", "comparison job required")
    workflow_bytes = subprocess.check_output(
        ["git", "-C", str(candidate_source), "show", f"{workflow_sha}:{WORKFLOW}"]
    )
    attempt = positive_id(env["GITHUB_RUN_ATTEMPT"], "run attempt")
    return {
        "format_version": 1,
        "evidence_kind": "pr_native_training_source_binding",
        "repository": repository,
        "pr_number": number,
        "event_action": event["action"],
        "baseline_sha": baseline,
        "authored_head_sha": head,
        "tested_merge_sha": candidate,
        "tested_merge_parents": parents,
        "head_repository": pr["head"]["repo"]["full_name"],
        "workflow_ref": workflow_ref,
        "workflow_sha": workflow_sha,
        "workflow_source": digest(workflow_bytes),
        "tested_harness_files": {
            name: digest(read_metadata(candidate_source / name)) for name in HARNESS_FILES
        },
        "run_id": positive_id(env["GITHUB_RUN_ID"], "run ID"),
        "run_attempt": attempt,
        "job": env["GITHUB_JOB"],
        "measurement_mode": "warm-resume",
        "observation_role": "initial_attempt" if attempt == 1 else "diagnostic_rerun",
        "selection_policy": "retain first observation for exact base/head/merge, including failures; later runs diagnostic",
        "unique_primary_run_enforced": False,
    }


def read_metadata(path: pathlib.Path) -> bytes:
    require(not path.is_symlink() and path.is_file(), f"metadata must be a regular file: {path.name}")
    require(0 < path.stat().st_size <= MAX_METADATA_BYTES, f"metadata size invalid: {path.name}")
    return path.read_bytes()


def digest(raw: bytes) -> dict[str, Any]:
    return {"byte_count": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def raw_inventory(root: pathlib.Path) -> dict[str, Any]:
    paths = []
    for path in root.rglob("*"):
        require(not path.is_symlink(), "evidence cannot contain symlinks")
        paths.append(path)
        require(len(paths) <= MAX_EVIDENCE_FILES, "evidence inventory exceeds file budget")
    files = {}
    total = 0
    for path in sorted(paths):
        if path.is_dir() or path == root / "pr-evidence.json":
            continue
        require(path.is_file() and path.stat().st_size <= MAX_METADATA_BYTES, "raw file size/type invalid")
        with path.open("rb") as source:
            raw = source.read(MAX_METADATA_BYTES + 1)
        require(len(raw) <= MAX_METADATA_BYTES, "raw file exceeds byte budget")
        total += len(raw)
        require(total <= MAX_EVIDENCE_BYTES, "evidence inventory exceeds byte budget")
        files[path.relative_to(root).as_posix()] = digest(raw)
    return files


def evidence_binding(binding: dict[str, Any], root: pathlib.Path, admission_outcome: str) -> dict[str, Any]:
    require(admission_outcome in {"success", "failure", "skipped", "cancelled"}, "unknown comparison step outcome")
    files = raw_inventory(root)
    decoded = {}
    for name in ("source-binding.json", "build-manifest.json", "comparison.json"):
        path = root / name
        if path.exists() or path.is_symlink():
            raw = read_metadata(path)
            value = json.loads(raw)
            require(type(value) is dict, f"metadata must be an object: {name}")
            files[name] = digest(raw)
            decoded[name] = value
    require(decoded.get("source-binding.json") == binding, "source binding changed")
    build = decoded.get("build-manifest.json")
    if build is not None:
        require(build["measurement_mode"] == binding["measurement_mode"] and
                build["revisions"]["baseline"]["source_sha"] == binding["baseline_sha"] and
                build["revisions"]["candidate"]["source_sha"] == binding["tested_merge_sha"],
                "build source binding differs")
    comparison = decoded.get("comparison.json")
    status = "incomplete"
    if comparison is not None:
        require(comparison["baseline_sha"] == binding["baseline_sha"] and
                comparison["candidate_sha"] == binding["tested_merge_sha"] and
                comparison["measurement_mode"] == binding["measurement_mode"],
                "comparison source binding differs")
        workflow = comparison.get("workflow")
        expected_workflow = {
            "repository": binding["repository"], "run_id": str(binding["run_id"]),
            "run_attempt": str(binding["run_attempt"]), "job": binding["job"],
        }
        status = comparison["status"]
        require(status in {"comparable", "incomparable"}, "unknown comparison status")
        if status == "incomparable":
            require(type(comparison["reasons"]) is list and comparison["reasons"] and
                    all(type(reason) is str and reason for reason in comparison["reasons"]),
                    "incomparable evidence requires reasons")
        if workflow is None and status == "incomparable":
            # The existing checker emits a reduced manifest on early failure.
            # Preserve it, but never authenticate absent run provenance.
            status = "incomplete"
        else:
            require(workflow == expected_workflow, "comparison workflow identity differs")
        if status == "comparable":
            require(admission_outcome == "success", "failed admission cannot authenticate comparable evidence")
            require(build is not None and comparison["reasons"] == [] and
                    comparison["build_manifest"]["sha256"] == files["build-manifest.json"]["sha256"],
                    "comparable evidence has no matching build manifest")
            require(read_metadata(root / "measurement-failed.txt").strip() == b"0",
                    "failed trial cannot be comparable")
    return {
        "format_version": 1,
        "evidence_kind": "pr_native_training_evidence_binding",
        "source": binding,
        "comparison_status": status,
        "comparison_step_outcome": admission_outcome,
        "files": files,
        "performance_threshold": None,
        "automatic_merge_authorization": False,
        "trust_scope": "PR-authored observations; source review and independent raw-evidence verification required",
    }


def write_new(path: pathlib.Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("source", "evidence"))
    arguments = parser.parse_args()
    workspace = pathlib.Path(os.environ["GITHUB_WORKSPACE"])
    event = json.loads(read_metadata(pathlib.Path(os.environ["GITHUB_EVENT_PATH"])))
    binding = source_binding(event, dict(os.environ), workspace)
    root = workspace / "native-transformer-training-comparison"
    if arguments.stage == "source":
        root.mkdir(exist_ok=True)
        write_new(root / "source-binding.json", binding)
    else:
        write_new(root / "pr-evidence.json", evidence_binding(binding, root, os.environ["COMPARISON_ADMISSION_OUTCOME"]))


if __name__ == "__main__":
    main()
