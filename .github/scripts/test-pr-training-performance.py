#!/usr/bin/env python3
"""Hosted-CI admission, provenance, and fixed-recipe fixtures; no Rust build."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile

SCRIPTS = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pr_binding", SCRIPTS / "bind-pr-training-performance.py")
assert spec is not None and spec.loader is not None
BINDER = importlib.util.module_from_spec(spec)
sys.dont_write_bytecode = True
spec.loader.exec_module(BINDER)


def rejected(call) -> None:
    try:
        call()
    except (ValueError, KeyError, subprocess.CalledProcessError):
        return
    raise AssertionError("malformed binding was accepted")


def write_json(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def source_fixture(workspace: pathlib.Path):
    candidate = workspace / "candidate-source"
    candidate.mkdir()
    BINDER.git(candidate, "init", "-q")
    BINDER.git(candidate, "config", "user.name", "Fixture")
    BINDER.git(candidate, "config", "user.email", "fixture@example.invalid")
    value = candidate / "value"
    value.write_text("base\n")
    for name in BINDER.HARNESS_FILES:
        path = candidate / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture harness\n")
    BINDER.git(candidate, "add", ".")
    BINDER.git(candidate, "commit", "-qm", "baseline")
    baseline = BINDER.git(candidate, "rev-parse", "HEAD")
    value.write_text("head\n")
    BINDER.git(candidate, "commit", "-qam", "candidate")
    head = BINDER.git(candidate, "rev-parse", "HEAD")
    tree = BINDER.git(candidate, "rev-parse", "HEAD^{tree}")
    merge = BINDER.git(candidate, "commit-tree", tree, "-p", baseline, "-p", head, "-m", "merge")
    BINDER.git(candidate, "checkout", "-q", merge)
    subprocess.run(["git", "clone", "-q", str(candidate), str(workspace / "baseline-source")], check=True)
    BINDER.git(workspace / "baseline-source", "checkout", "-q", baseline)
    repository = "owner/rustgrad"
    event = {
        "action": "labeled", "label": {"name": "performance-evidence"}, "number": 7,
        "repository": {"full_name": repository},
        "pull_request": {
            "number": 7, "state": "open", "draft": False,
            "labels": [{"name": "performance-evidence"}],
            "base": {"sha": baseline, "ref": "main", "repo": {"full_name": repository}},
            "head": {"sha": head, "repo": {"full_name": "fork/rustgrad"}},
        },
    }
    env = {
        "GITHUB_EVENT_NAME": "pull_request", "GITHUB_REPOSITORY": repository,
        "GITHUB_REF": "refs/pull/7/merge", "GITHUB_SHA": merge,
        "BASELINE_SHA": baseline, "CANDIDATE_SHA": merge, "MEASUREMENT_MODE": "warm-resume",
        "GITHUB_WORKFLOW_SHA": merge,
        "GITHUB_WORKFLOW_REF": f"{repository}/{BINDER.WORKFLOW}@refs/pull/7/merge",
        "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_JOB": "compare",
    }
    return event, env


def check_sources(workspace: pathlib.Path):
    event, env = source_fixture(workspace)
    binding = BINDER.source_binding(event, env, workspace)
    assert binding["tested_merge_parents"] == [env["BASELINE_SHA"], event["pull_request"]["head"]["sha"]]
    assert binding["observation_role"] == "initial_attempt" and not binding["unique_primary_run_enforced"]
    assert binding["workflow_source"]["sha256"] == hashlib.sha256(b"fixture harness\n").hexdigest()
    assert set(binding["tested_harness_files"]) == set(BINDER.HARNESS_FILES)
    another_run = dict(env); another_run["GITHUB_RUN_ID"] = "124"
    assert BINDER.source_binding(event, another_run, workspace)["unique_primary_run_enforced"] is False
    for action in ("opened", "reopened", "synchronize", "ready_for_review"):
        changed = copy.deepcopy(event); changed["action"] = action
        assert BINDER.source_binding(changed, env, workspace)["event_action"] == action
    for key, wrong in {
        "GITHUB_EVENT_NAME": "pull_request_target", "GITHUB_REPOSITORY": "other/repo",
        "GITHUB_REF": "refs/heads/main", "GITHUB_SHA": env["BASELINE_SHA"],
        "BASELINE_SHA": "0" * 40, "CANDIDATE_SHA": "0" * 40,
        "MEASUREMENT_MODE": "steady-replay",
        "GITHUB_WORKFLOW_REF": "other/repo/workflow@main", "GITHUB_RUN_ID": "0",
        "GITHUB_RUN_ATTEMPT": "01",
        "GITHUB_JOB": "unrelated",
        "GITHUB_WORKFLOW_SHA": env["BASELINE_SHA"],
    }.items():
        changed = dict(env); changed[key] = wrong
        rejected(lambda: BINDER.source_binding(event, changed, workspace))
    mutations = (
        lambda e: e.update(action="closed"),
        lambda e: e["label"].update(name="unrelated"),
        lambda e: e["pull_request"].update(draft=True),
        lambda e: e["pull_request"].update(state="closed"),
        lambda e: e["pull_request"].update(labels=[]),
        lambda e: e["pull_request"]["base"].update(ref="feature"),
        lambda e: e["pull_request"]["head"].update(sha="0" * 40),
        lambda e: e["pull_request"]["base"].update(sha=e["pull_request"]["head"]["sha"]),
        lambda e: e.update(number=True),
    )
    for mutate in mutations:
        changed = copy.deepcopy(event); mutate(changed)
        rejected(lambda: BINDER.source_binding(changed, env, workspace))
    candidate = workspace / "candidate-source"
    (candidate / "dirty").write_text("untracked")
    rejected(lambda: BINDER.source_binding(event, env, workspace))
    (candidate / "dirty").unlink()
    baseline_source = workspace / "baseline-source"
    (baseline_source / "dirty").write_text("untracked")
    rejected(lambda: BINDER.source_binding(event, env, workspace))
    (baseline_source / "dirty").unlink()
    BINDER.git(baseline_source, "checkout", "-q", event["pull_request"]["head"]["sha"])
    rejected(lambda: BINDER.source_binding(event, env, workspace))
    BINDER.git(baseline_source, "checkout", "-q", env["BASELINE_SHA"])
    wrong_ref = dict(env); wrong_ref["GITHUB_WORKFLOW_REF"] = f"{env['GITHUB_REPOSITORY']}/{BINDER.WORKFLOW}@refs/heads/main"
    rejected(lambda: BINDER.source_binding(event, wrong_ref, workspace))
    # An ordinary authored commit, not the tested two-parent merge, is rejected.
    wrong = dict(env); wrong["GITHUB_SHA"] = wrong["CANDIDATE_SHA"] = event["pull_request"]["head"]["sha"]
    BINDER.git(candidate, "checkout", "-q", wrong["GITHUB_SHA"])
    rejected(lambda: BINDER.source_binding(event, wrong, workspace))
    BINDER.git(candidate, "checkout", "-q", env["GITHUB_SHA"])
    tree = BINDER.git(candidate, "rev-parse", "HEAD^{tree}")
    extra = BINDER.git(candidate, "commit-tree", tree, "-p", env["BASELINE_SHA"], "-m", "extra")
    head = event["pull_request"]["head"]["sha"]
    for parents in ([head, env["BASELINE_SHA"]], [env["BASELINE_SHA"]],
                    [env["BASELINE_SHA"], head, extra]):
        args = [argument for parent in parents for argument in ("-p", parent)]
        malformed = BINDER.git(candidate, "commit-tree", tree, *args, "-m", "malformed merge")
        BINDER.git(candidate, "checkout", "-q", malformed)
        changed = dict(env); changed["GITHUB_SHA"] = changed["CANDIDATE_SHA"] = malformed
        rejected(lambda: BINDER.source_binding(event, changed, workspace))
    BINDER.git(candidate, "checkout", "-q", env["GITHUB_SHA"])
    rerun = dict(env); rerun["GITHUB_RUN_ATTEMPT"] = "2"
    assert BINDER.source_binding(event, rerun, workspace)["observation_role"] == "diagnostic_rerun"
    evidence = workspace / "native-transformer-training-comparison"
    evidence.mkdir()
    write_json(evidence / "event-context.json", {"evidence_kind": "unverified_pr_event_context"})
    event_path = workspace / "event.json"
    write_json(event_path, event)
    cli_env = {**os.environ, **env, "GITHUB_EVENT_PATH": str(event_path), "GITHUB_WORKSPACE": str(workspace), "COMPARISON_ADMISSION_OUTCOME": "skipped"}
    script = SCRIPTS / "bind-pr-training-performance.py"
    subprocess.run([sys.executable, str(script), "source"], env=cli_env, check=True)
    assert json.loads((evidence / "source-binding.json").read_text()) == binding
    subprocess.run([sys.executable, str(script), "evidence"], env=cli_env, check=True)
    assert json.loads((evidence / "pr-evidence.json").read_text())["comparison_status"] == "incomplete"
    return binding


def check_evidence(binding, root: pathlib.Path):
    root.mkdir()
    write_json(root / "source-binding.json", binding)
    partial = BINDER.evidence_binding(binding, root, "success")
    assert partial["comparison_status"] == "incomplete" and not partial["automatic_merge_authorization"]
    trial = root / "trial-01-baseline"
    trial.mkdir()
    (trial / "partial.txt").write_bytes(b"failed before checkpoint")
    (trial / "empty.txt").write_bytes(b"")
    (trial / "pr-evidence.json").write_text("nested raw file")
    partial = BINDER.evidence_binding(binding, root, "success")
    assert partial["files"]["trial-01-baseline/partial.txt"] == BINDER.digest(b"failed before checkpoint")
    assert partial["files"]["trial-01-baseline/empty.txt"] == BINDER.digest(b"")
    assert "trial-01-baseline/pr-evidence.json" in partial["files"]
    build = {
        "measurement_mode": "warm-resume",
        "revisions": {"baseline": {"source_sha": binding["baseline_sha"]},
                      "candidate": {"source_sha": binding["tested_merge_sha"]}},
    }
    write_json(root / "build-manifest.json", build)
    comparison = {
        "baseline_sha": binding["baseline_sha"], "candidate_sha": binding["tested_merge_sha"],
        "measurement_mode": "warm-resume", "status": "comparable", "reasons": [],
        "build_manifest": {"sha256": hashlib.sha256((root / "build-manifest.json").read_bytes()).hexdigest()},
        "workflow": {"repository": binding["repository"], "run_id": str(binding["run_id"]),
                     "run_attempt": str(binding["run_attempt"]), "job": binding["job"]},
    }
    write_json(root / "comparison.json", comparison)
    for field in ("repository", "run_id", "run_attempt", "job"):
        changed = copy.deepcopy(comparison); changed["workflow"][field] = "different"
        write_json(root / "comparison.json", changed)
        rejected(lambda: BINDER.evidence_binding(binding, root, "success"))
    write_json(root / "comparison.json", comparison)
    (root / "measurement-failed.txt").write_text("0\n")
    admitted = BINDER.evidence_binding(binding, root, "success")
    assert admitted["comparison_status"] == "comparable" and admitted["performance_threshold"] is None
    rejected(lambda: BINDER.evidence_binding(binding, root, "failure"))
    for name, record in admitted["files"].items():
        raw = (root / name).read_bytes()
        assert record == {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
    for field, wrong in (("candidate_sha", "0" * 40), ("status", "success"),
                         ("reasons", ["schema differs"]), ("measurement_mode", "steady-replay"),
                         ("build_manifest", {"sha256": "0" * 64})):
        changed = dict(comparison); changed[field] = wrong
        write_json(root / "comparison.json", changed)
        rejected(lambda: BINDER.evidence_binding(binding, root, "success"))
    write_json(root / "comparison.json", comparison)
    (root / "measurement-failed.txt").write_text("1\n")
    rejected(lambda: BINDER.evidence_binding(binding, root, "success"))
    comparison["status"] = "incomparable"; comparison["reasons"] = ["trial failed"]
    write_json(root / "comparison.json", comparison)
    assert BINDER.evidence_binding(binding, root, "success")["comparison_status"] == "incomparable"
    failed = copy.deepcopy(comparison); del failed["workflow"]
    write_json(root / "comparison.json", failed)
    assert BINDER.evidence_binding(binding, root, "success")["comparison_status"] == "incomplete"
    failed["reasons"] = []
    write_json(root / "comparison.json", failed)
    rejected(lambda: BINDER.evidence_binding(binding, root, "success"))
    write_json(root / "comparison.json", comparison)
    changed = dict(binding); changed["run_id"] += 1
    rejected(lambda: BINDER.evidence_binding(changed, root, "success"))
    (root / "comparison.json").unlink()
    (root / "comparison.json").symlink_to(root / "build-manifest.json")
    rejected(lambda: BINDER.evidence_binding(binding, root, "success"))
    (root / "comparison.json").unlink()
    (trial / "partial.txt").write_bytes(b"changed failed trial")
    assert BINDER.evidence_binding(binding, root, "success")["files"]["trial-01-baseline/partial.txt"] != partial["files"]["trial-01-baseline/partial.txt"]
    previous_budget = BINDER.MAX_EVIDENCE_FILES
    BINDER.MAX_EVIDENCE_FILES = 1
    rejected(lambda: BINDER.evidence_binding(binding, root, "success"))
    BINDER.MAX_EVIDENCE_FILES = previous_budget
    BINDER.write_new(root / "pr-evidence.json", partial)
    try:
        BINDER.write_new(root / "pr-evidence.json", partial)
    except FileExistsError:
        pass
    else:
        raise AssertionError("evidence overwrote an existing observation")


def check_recipe(root: pathlib.Path, mode: str, failed_trial: str | None):
    (root / "baseline-source").mkdir()
    scripts = root / "candidate-source/.github/scripts"
    scripts.mkdir(parents=True)
    # Failing trial A must not hide or skip B, B, A. No Cargo/native execution.
    prefix = "RUSTGRAD_SCALE_EVIDENCE" if mode == "warm-resume" else "RUSTGRAD_NATIVE_SCOREBOARD"
    recorder = "record-larger-native-transformer-evidence.sh" if mode == "warm-resume" else "record-native-transformer-release-scoreboard.sh"
    (scripts / recorder).write_text(
        'set -euo pipefail\n'
        f'mkdir -- "${{{prefix}_OUTPUT_DIR}}"\n'
        f'basename "${{{prefix}_OUTPUT_DIR}}" >> "$RUNNER_TEMP/order"\n'
        f'printf "%s\\n%s\\n%s\\n" "${{{prefix}_MEASURED_SHA}}" "${{{prefix}_BINARY}}" "$PWD" > "${{{prefix}_OUTPUT_DIR}}/binding.txt"\n'
        f'[[ "${{{prefix}_OUTPUT_DIR}}" != *"$FAILED_TRIAL" ]]\n')
    evidence = root / "native-transformer-training-comparison"
    evidence.mkdir()
    env = {**os.environ, "GITHUB_WORKSPACE": str(root), "RUNNER_TEMP": str(root),
           "BASELINE_SHA": "1" * 40, "CANDIDATE_SHA": "2" * 40, "MEASUREMENT_MODE": mode,
           "FAILED_TRIAL": failed_trial or "no-trial-fails"}
    script = SCRIPTS / "run-native-transformer-training-comparison.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(["bash", str(script), "record"], env=env)
    expected = 0 if failed_trial is None else 1
    assert result.returncode == expected and (evidence / "measurement-failed.txt").read_text() == f"{expected}\n"
    assert (root / "order").read_text().splitlines() == [
        "trial-01-baseline", "trial-02-candidate", "trial-03-candidate", "trial-04-baseline"]
    example = "compiled_transformer_scale_evidence" if mode == "warm-resume" else "compiled_transformer_train_resume"
    for ordinal, role in ((1, "baseline"), (2, "candidate"), (3, "candidate"), (4, "baseline")):
        assert (evidence / f"trial-{ordinal:02d}-{role}/binding.txt").read_text().splitlines() == [
            env["BASELINE_SHA" if role == "baseline" else "CANDIDATE_SHA"],
            str(root / f"native-transformer-comparison-build/{role}/release/examples/{example}"),
            str(root / f"{role}-source"),
        ]
    (evidence / "comparison.json").write_text('{"status":"comparable"}')
    assert subprocess.run(["bash", str(script), "validate"], env=env).returncode != 0
    assert (evidence / "comparison.json").read_text() == '{"status":"comparable"}'


def check_workflow_policy():
    workflows = SCRIPTS.parent / "workflows"
    pr = (workflows / "pr-training-performance.yml").read_text()
    manual = (workflows / "native-transformer-training-comparison.yml").read_text()
    assert "  pull_request:\n" in pr and "pull_request_target:" not in pr
    assert "branches: [main]" in pr and "contents: read" in pr and "performance-evidence" in pr
    assert "environment:" not in pr and "secrets." not in pr and "rust-cache" not in pr
    assert "runs-on: ubuntu-24.04" in pr and "timeout-minutes: 75" in pr
    assert "cancel-in-progress: false" in pr
    assert "group: pr-training-performance-${{ github.event.pull_request.number }}-${{ github.run_id }}" in pr
    for name in ("PR_NUMBER", "BASELINE_SHA", "AUTHORED_HEAD_SHA", "CANDIDATE_SHA"):
        assert f"ENVELOPE_{name}: ${{{{ github." in pr
    assert pr.count("persist-credentials: false") == 2
    assert "ref: ${{ github.sha }}" in pr and "${{ github.run_attempt }}" in pr
    assert "${{ github.event.pull_request.head.sha }}" in pr and "if: always()" in pr
    assert pr.index("Retain event context before checkout") < pr.index("Check out exact tested merge")
    assert "Retain stage outcomes independently of checkout\n        id: finalize\n        timeout-minutes: 3\n        if: always()" in pr
    for stage in ("candidate", "baseline", "source", "toolchain", "fixtures", "build", "record", "admission", "binding"):
        assert f"${{{{ steps.{stage}.outcome }}}}" in pr
    assert 'if [[ "${GITHUB_SHA:-}" != "$CANDIDATE_SHA" ]]' in manual
    assert "environment: native-transformer-scale-evidence" in manual
    assert "test-pr-training-performance.py" in manual
    assert "path: ${{ steps.finalize.outputs.upload_path }}" in pr
    budgets = [int(line.split(":", 1)[1]) for line in pr.splitlines() if "timeout-minutes:" in line]
    assert budgets[0] == 75 and sum(budgets[1:]) <= 73
    for text in (pr, manual):
        for stage in ("build", "record", "validate"):
            assert text.count(f"run-native-transformer-training-comparison.sh {stage}") == 1
        for line in text.splitlines():
            if "uses:" in line:
                assert len(line.rsplit("@", 1)[1]) == 40, "action is not full-SHA pinned"


def check_build_failure(root: pathlib.Path, failed_role: str):
    event, env = source_fixture(root)
    evidence = root / "native-transformer-training-comparison"
    evidence.mkdir()
    # Event provenance survives a failed first or second build unchanged.
    write_json(evidence / "event-context.json", {"pr_number": event["number"]})
    before = (evidence / "event-context.json").read_bytes()
    fake_bin = root / "fake-bin"
    fake_bin.mkdir()
    cargo = fake_bin / "cargo"
    cargo.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'if [[ "$*" == *baseline-source/Cargo.toml* ]]; then role=baseline; else role=candidate; fi\n'
        'printf "%s\\n" "$role" >> "$RUNNER_TEMP/build-order"\n'
        '[[ "$role" != "$FAILED_ROLE" ]]\n')
    cargo.chmod(0o755)
    execution_env = {**os.environ, **env, "GITHUB_WORKSPACE": str(root), "RUNNER_TEMP": str(root),
                     "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"], "FAILED_ROLE": failed_role}
    script = SCRIPTS / "run-native-transformer-training-comparison.sh"
    assert subprocess.run(["bash", str(script), "build"], env=execution_env).returncode == 1
    expected = ["baseline"] if failed_role == "baseline" else ["baseline", "candidate"]
    assert (root / "build-order").read_text().splitlines() == expected
    assert (evidence / "event-context.json").read_bytes() == before
    assert not (evidence / "build-manifest.json").exists()


def check_upload_envelope(workspace: pathlib.Path, violation: str | None):
    workflow = (SCRIPTS.parent / "workflows/pr-training-performance.yml").read_text()
    block = workflow.split("- name: Retain stage outcomes independently of checkout", 1)[1]
    body = block.split("python3 - <<'PY'\n", 1)[1].split("          PY", 1)[0]
    code = "\n".join(line.removeprefix("          ") for line in body.splitlines())
    root = workspace / "native-transformer-training-comparison"
    root.mkdir()
    (root / "event-context.json").write_text('{"unverified":true}')
    (root / "empty.txt").write_bytes(b"")
    if violation == "symlink":
        (root / "link").symlink_to(workspace)
    elif violation in ("ancestor-symlink", "intermediate-symlink"):
        (workspace / "outside").mkdir()
        (workspace / "outside/payload").write_text("must not upload")
        (root / "linked-directory").symlink_to(workspace / "outside", target_is_directory=True)
    elif violation == "root-symlink":
        root.rename(workspace / "real-evidence")
        root.symlink_to(workspace / "real-evidence", target_is_directory=True)
    elif violation == "nested-regular":
        (root / "nested").mkdir()
        (root / "nested/payload").write_text("owned payload")
    elif violation == "hardlink":
        (workspace / "outside-file").write_text("must not upload")
        os.link(workspace / "outside-file", root / "hardlink")
    elif violation == "forged-sidecar":
        (root / "pr-evidence.json").write_text('{"comparison_status":"comparable"}')
    elif violation == "entries":
        for index in range(65):
            (root / f"extra-{index}").write_bytes(b"")
    elif violation == "bytes":
        with (root / "too-large").open("wb") as output:
            output.truncate(8 * 1024 * 1024 + 1)
    elif violation == "total":
        for index in range(9):
            with (root / f"large-{index}").open("wb") as output:
                output.truncate(8 * 1024 * 1024)
    elif violation == "reserved":
        (root / "job-outcomes.json").write_text("forged envelope")
    elif violation == "reserved-directory":
        (root / "job-outcomes.json").mkdir()
        (root / "job-outcomes.json/payload").write_text("forged envelope tree")
    env = {**os.environ, "GITHUB_WORKSPACE": str(workspace), "RUNNER_TEMP": str(workspace),
           "GITHUB_OUTPUT": str(workspace / "outputs"), "GITHUB_REPOSITORY": "owner/rustgrad",
           "ENVELOPE_PR_NUMBER": "7", "ENVELOPE_BASELINE_SHA": "1" * 40, "ENVELOPE_AUTHORED_HEAD_SHA": "2" * 40,
           "ENVELOPE_CANDIDATE_SHA": "3" * 40, "GITHUB_WORKFLOW_SHA": "3" * 40,
           "PR_NUMBER": "999", "BASELINE_SHA": "0" * 40, "AUTHORED_HEAD_SHA": "0" * 40,
           "CANDIDATE_SHA": "0" * 40,
           "GITHUB_WORKFLOW_REF": "owner/rustgrad/.github/workflows/pr-training-performance.yml@refs/pull/7/merge",
           "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_JOB": "compare"}
    for name in ("CANDIDATE", "BASELINE", "SOURCE", "TOOLCHAIN", "FIXTURES", "BUILD", "RECORD", "ADMISSION", "BINDING"):
        env[name] = "failure" if name == "CANDIDATE" else "skipped"
    if violation == "intermediate-symlink":
        # Model a descendant yielded before an ancestor was replaced. Keep the
        # real descriptor-relative opener; do not mock the protection itself.
        code = (
            "import pathlib\n"
            "original_rglob = pathlib.Path.rglob\n"
            "def adversarial_descendant(self, pattern):\n"
            "    if self.name == 'native-transformer-training-comparison':\n"
            "        return iter([self / 'linked-directory' / 'payload'])\n"
            "    return original_rglob(self, pattern)\n"
            "pathlib.Path.rglob = adversarial_descendant\n"
        ) + code
    result = subprocess.run([sys.executable, "-c", code], env=env)
    upload = pathlib.Path((workspace / "outputs").read_text().strip().removeprefix("upload_path="))
    envelope = json.loads((upload / "job-outcomes.json").read_text())
    assert envelope["context"]["GITHUB_RUN_ID"] == "123" and envelope["context"]["CANDIDATE_SHA"] == "3" * 40
    assert envelope["source_authentication"] == "skipped" and envelope["observation_role"] == "initial_attempt"
    if violation not in (None, "nested-regular"):
        assert result.returncode == 1 and envelope["retention_error"] is not None
        assert list(upload.iterdir()) == [upload / "job-outcomes.json"] and envelope["files"] == {}
    else:
        assert result.returncode == 0 and envelope["retention_error"] is None
        assert envelope["files"]["empty.txt"] == BINDER.digest(b"")
        for name, record in envelope["files"].items():
            assert BINDER.digest((upload / name).read_bytes()) == record


def main():
    check_workflow_policy()
    with tempfile.TemporaryDirectory(prefix="rustgrad-pr-performance-") as temporary:
        root = pathlib.Path(temporary)
        binding = check_sources(root)
        check_evidence(binding, root / "evidence")
    for mode in ("warm-resume", "steady-replay"):
        for failed in (None, "trial-01-baseline", "trial-02-candidate", "trial-03-candidate", "trial-04-baseline"):
            with tempfile.TemporaryDirectory(prefix="rustgrad-pr-recipe-") as temporary:
                check_recipe(pathlib.Path(temporary), mode, failed)
    for role in ("baseline", "candidate"):
        with tempfile.TemporaryDirectory(prefix="rustgrad-pr-build-failure-") as temporary:
            check_build_failure(pathlib.Path(temporary), role)
    for violation in (None, "nested-regular", "symlink", "ancestor-symlink", "intermediate-symlink", "root-symlink", "hardlink", "forged-sidecar", "entries", "bytes", "total", "reserved", "reserved-directory"):
        with tempfile.TemporaryDirectory(prefix="rustgrad-pr-upload-") as temporary:
            check_upload_envelope(pathlib.Path(temporary), violation)
    print("PR source/run binding, partial evidence, fixed ABBA and workflow policy fixtures passed")


if __name__ == "__main__":
    main()
