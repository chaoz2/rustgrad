#!/usr/bin/env python3
"""Remote acceptance: real corpus input and exact cross-process continuation.

Run only after CI builds the example. No timing or generalization claim.
"""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    executable, fixture, receipt = map(Path, sys.argv[1:])
    executable = executable.resolve()
    corpus = fixture.read_bytes()
    batch_tokens = 32  # Independently pinned B=2, T=16 contract.
    weights = [min(batch_tokens, len(corpus) - 1 - start)
               for start in range(0, len(corpus) - 1, batch_tokens)]
    assert len(weights) > 1 and 0 < weights[-1] < batch_tokens

    with tempfile.TemporaryDirectory(prefix="rustgrad-corpus-") as directory:
        root = Path(directory)

        def run(*args):
            result = subprocess.run([str(executable), *map(str, args)],
                                    capture_output=True, text=True, check=True)
            value = json.loads(result.stdout)
            assert value["format_version"] == 1
            assert value["fallback_count"] == 0
            if args[0] != "evaluate":
                for key in ("initial_training_probe_loss", "final_training_probe_loss"):
                    assert math.isfinite(value[key]) and value[key] > 0
            return value

        full_file, pending_file, resumed_file = [root / name for name in
                                                ("full.safetensors", "pending.safetensors", "resumed.safetensors")]
        full = run("train", fixture, full_file, 6)
        pending = run("train", fixture, pending_file, 3)
        # An independent corpus has unequal full/tail weights and is not saved
        # into the training envelope. Evaluate twice across process boundaries.
        evaluation_file = root / "evaluation.txt"
        evaluation_bytes = b"A separate byte sequence for evaluation.\n" * 4
        evaluation_file.write_bytes(evaluation_bytes)
        pending_bytes = pending_file.read_bytes()
        evaluated = run("evaluate", fixture, pending_file, evaluation_file)
        assert run("evaluate", fixture, pending_file, evaluation_file) == evaluated
        eval_weights = [min(batch_tokens, len(evaluation_bytes) - 1 - start)
                        for start in range(0, len(evaluation_bytes) - 1, batch_tokens)]
        assert len(eval_weights) > 4  # Full evaluation must not use the four-batch probe limit.
        assert evaluated["mode"] == "evaluate"
        assert evaluated["valid_tokens_per_batch"] == eval_weights
        assert evaluated["valid_tokens"] == len(evaluation_bytes) - 1
        losses = evaluated["batch_losses"]
        assert len(losses) == len(eval_weights)
        assert all(math.isfinite(loss) and loss > 0 for loss in losses)
        expected_mean = sum(loss * weight for loss, weight in zip(losses, eval_weights)) / sum(eval_weights)
        assert evaluated["mean_token_loss"] == expected_mean
        assert evaluated["restored_without_graph_rebuild"] is True
        assert evaluated["training_state_unchanged"] is True
        assert pending_file.read_bytes() == pending_bytes
        assert evaluation_file.read_bytes() == evaluation_bytes
        # Continuing the evaluated pending checkpoint must still equal the
        # uninterrupted run, including dropout and pending-gradient state.
        resumed = run("resume", fixture, pending_file, resumed_file, 3)
        for value, first, end, restored in ((full, 0, 6, False),
                                             (pending, 0, 3, False),
                                             (resumed, 3, 6, True)):
            assert value["first_replay"] == first
            assert value["completed_replays"] == end
            assert value["optimizer_steps"] == end // 2
            assert value["pending_microbatches"] == end % 2
            assert value["resumed_without_graph_rebuild"] is restored
            assert value["valid_tokens_per_microbatch"] == [weights[i % len(weights)] for i in range(first, end)]
        assert full["initial_training_probe_loss"] == pending["initial_training_probe_loss"]
        assert pending["final_training_probe_loss"] == resumed["initial_training_probe_loss"]
        assert full["final_training_probe_loss"] == resumed["final_training_probe_loss"]
        assert full["final_training_probe_loss"] < full["initial_training_probe_loss"]
        full_bytes = full_file.read_bytes()
        assert full_bytes == resumed_file.read_bytes(), "complete corpus/program/module/optimizer frontier differs"

        def reject(arguments, message):
            output = root / "rejected.safetensors"
            result = subprocess.run([str(executable), *map(str, arguments), str(output), "1"],
                                    capture_output=True, text=True)
            assert result.returncode != 0 and message in result.stderr, result.stderr
            assert not output.exists(), "rejected input published a checkpoint"

        changed = root / "changed.txt"
        changed.write_bytes(bytes([corpus[0] ^ 1]) + corpus[1:])
        reject(["resume", changed, pending_file], "corpus differs")
        for content in (b"", b"x"):
            changed.write_bytes(content)
            reject(["train", changed], "corpus must contain")
            rejected_eval = subprocess.run(
                [str(executable), "evaluate", str(fixture), str(pending_file), str(changed)],
                capture_output=True, text=True)
            assert rejected_eval.returncode != 0 and "corpus must contain" in rejected_eval.stderr
            assert pending_file.read_bytes() == pending_bytes
        # Existing outputs must survive refusal, even when the invocation
        # would otherwise be a valid continuation.
        original = pending_file.read_bytes()
        overwrite = subprocess.run(
            [str(executable), "resume", str(fixture), str(pending_file), str(pending_file), "1"],
            capture_output=True, text=True)
        assert overwrite.returncode != 0 and "output already exists" in overwrite.stderr
        assert pending_file.read_bytes() == original
        receipt.write_text(json.dumps({
            "format_version": 1, "full": full, "pending": pending, "resumed": resumed,
            "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
            "checkpoint_bytes": len(full_bytes),
            "checkpoint_sha256": hashlib.sha256(full_bytes).hexdigest(),
            "exact_cross_process_resume": True,
            "changed_and_short_corpus_rejected": True,
            "existing_checkpoint_preserved": True,
            "evaluation": evaluated,
            "evaluation_corpus_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
