# Train on a local byte corpus

`compiled_corpus_train` connects a caller-owned text or binary file to the
compiled AdamW training runtime. It trains a small, randomly initialized,
two-block causal Transformer on next-byte prediction.

## Training and resume

The example takes one of two argument forms:

```text
train CORPUS OUTPUT_CHECKPOINT MICRO_BATCHES
resume CORPUS INPUT_CHECKPOINT OUTPUT_CHECKPOINT MICRO_BATCHES
```

Use a new output path for each invocation. `MICRO_BATCHES` counts replay calls;
two microbatches produce one AdamW update. An odd count saves the pending
accumulation window rather than flushing it early.

Fresh training compiles forward, masked loss, backward, clipping, dropout, and
AdamW updates once. Resume restores the saved executable captures and complete
state into a differently initialized model; it does not rebuild autograd or
scheduling. Execution uses strict-native CPU with no interpreter fallback.

## What the checkpoint contains

- Exact corpus bytes and the versioned batching contract.
- Executable training and evaluation captures.
- Parameters, frozen positions, tied-weight topology, and optimizer state.
- Replay progress, dropout state, and any pending accumulated gradients.

The file is a bounded safetensors envelope around the existing authenticated
training bundle. Saving uses the existing atomic file writer. A different
corpus is rejected before the executable bundle is decoded or prepared.
Checkpoints contain your input data: treat them as private.

## Data and reporting

Inputs are 2 bytes to 1 MiB. Bytes are token IDs directly, with a vocabulary of
256; there is no tokenizer or Unicode interpretation. Batches contain two
16-token sequences. Contiguous windows cycle deterministically, and the last
window is right-padded. Ignored targets determine both the attention mask and
the exact token weighting across accumulation.

Each invocation prints JSON with replay and optimizer progress, valid-token
weights, fallback count, and initial/final loss on a fixed training-data probe.
That probe is not a held-out metric or a generalization claim.

## Acceptance and limits

Remote CI compares six uninterrupted microbatches with three followed by a
separate-process three-microbatch resume. It requires identical final checkpoint
bytes and probe losses, a decreasing loss on the checked-in fixture, correct
partial-window progress, and rejection of changed or too-short data without
output publication.

This is a fixed-size CPU training example, not a large-corpus loader,
tokenizer, language-model quality benchmark, Metal acceptance, or throughput
claim. Local builds and tests are not required to inspect its implementation.
