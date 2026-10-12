#!/usr/bin/env python3
"""Synthetic profile admission tests; no Rust builds or profiler required."""

import copy
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    "compile_profile", pathlib.Path(__file__).with_name("check-transformer-compile-profile.py")
)
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)


class CompileProfileTests(unittest.TestCase):
    def evidence(self):
        return {
            "schema_version": 1,
            "evidence_kind": "compiled_transformer_compile_only",
            "workload": copy.deepcopy(CHECK.WORKLOAD),
            "compile_count": 1, "initial_replay_step": 0,
            "attached_roles": list(CHECK.ROLES),
            "target_prepared": False, "replayed": False,
            "capture_identity": 123, "evaluation_capture_identity": 456,
            "recurrent_state_count": 145, "recurrent_state_bytes": 30244,
            "main_schedule_item_count": 785,
        }

    def test_compile_contract(self):
        self.assertEqual(CHECK.validate_compile_only(self.evidence()), self.evidence())
        for field, bad in (("compile_count", True), ("compile_count", 2),
                           ("target_prepared", True), ("replayed", True),
                           ("initial_replay_step", 1), ("capture_identity", 0),
                           ("evaluation_capture_identity", 123),
                           ("recurrent_state_bytes", -1),
                           ("main_schedule_item_count", 2**64)):
            with self.subTest(field=field, bad=bad):
                value = self.evidence()
                value[field] = bad
                with self.assertRaises(ValueError):
                    CHECK.validate_compile_only(value)
        for mutate in (
            lambda d: d.pop("attached_roles"),
            lambda d: d.update(extra=True),
            lambda d: d["attached_roles"].reverse(),
            lambda d: d["workload"].update(batch=True),
            lambda d: d["workload"].update(time=32),
        ):
            value = self.evidence()
            mutate(value)
            with self.assertRaises(ValueError):
                CHECK.validate_compile_only(value)

    def test_profile_contract(self):
        text = "# callgrind format\ncmd: /bin/profile compile-only\nevents: Ir\nsummary: 234\nfn=(1) compile\n1 234\n"
        self.assertEqual(CHECK.validate_callgrind(text, "/bin/profile compile-only"), 234)
        for broken in (
            text.replace("summary: 234", "summary: 0"),
            text.replace("summary: 234", "summary: invalid"),
            text.replace("summary: 234", f"summary: {2**64}"),
            text + "summary: 234\n",
            text.replace("events: Ir", "events: Ir Dr"),
            text.replace("compile-only", "lifecycle-only"),
            text.replace("# callgrind format", "# other format"),
            text.replace("fn=(1) compile\n", ""),
        ):
            with self.assertRaises(ValueError):
                CHECK.validate_callgrind(broken, "/bin/profile compile-only")

    def test_manifest_cli_and_file_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"fixture executable bytes")
            evidence = root / "compile-only.json"
            evidence.write_text(json.dumps(self.evidence()), encoding="utf-8")
            (root / "callgrind.out").write_text(
                f"# callgrind format\ncmd: {binary} compile-only\nevents: Ir\nsummary: 234\nfn=(1) compile\n1 234\n",
                encoding="utf-8",
            )
            for name in ("provenance.txt", "inclusive.txt", "exclusive.txt"):
                (root / name).write_text("fixture\n", encoding="utf-8")
            script = pathlib.Path(CHECK.__file__)
            subprocess.run([sys.executable, str(script), "compile", str(evidence)], check=True)
            command = [sys.executable, str(script), "profile", "--root", str(root),
                       "--expected-sha", "1" * 40, "--binary", str(binary)]
            subprocess.run(command, check=True)
            manifest_path = root / "profile-manifest.json"
            original = manifest_path.read_bytes()
            manifest = json.loads(original)
            self.assertEqual(manifest["source_sha"], "1" * 40)
            self.assertEqual(manifest["instruction_reads"], 234)
            self.assertEqual(manifest["binary_sha256"], hashlib.sha256(binary.read_bytes()).hexdigest())
            self.assertEqual(manifest["scope"], "initialization_compile_inspection_and_teardown")
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(manifest_path.read_bytes(), original)
            for path, bound in ((binary, 2), (root / "missing", 100)):
                with self.assertRaises(ValueError):
                    CHECK.read_bounded(path, bound)
            link = root / "link"
            link.symlink_to(binary)
            with self.assertRaises(ValueError):
                CHECK.read_bounded(link, 100)


if __name__ == "__main__":
    unittest.main()
