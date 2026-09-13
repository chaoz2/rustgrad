#!/usr/bin/env python3
"""Generate the checked two-block Transformer PyTorch reference fixture.

Run with the repository's offline PyTorch environment:

    PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.11 \
      tests/fixtures/generate_two_block_pytorch_frontier.py \
      tests/fixtures/two_block_pytorch_frontier.json

The generator mirrors only RustGrad's deterministic parameter initializer and
compiled Threefry dropout stream. Forward, loss, gradients, clipping, moments,
and the AdamW successor are computed by PyTorch on CPU.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from collections import OrderedDict
from pathlib import Path

import torch


SEED = 0x5678
BATCH = 2
TIME = 3
VOCAB = 5
EMBEDDING = 4
HEADS = 2
HEAD_SIZE = EMBEDDING // HEADS
FEED_FORWARD = 8
DROPOUT = 0.25
DROPOUT_KEY = (0x12345678, 0x9ABCDEF0)
MAX_GRADIENT_NORM = 1.0e-4
LEARNING_RATE = 1.0e-3
BETAS = (0.9, 0.999)
EPSILON = 1.0e-8
POLICY_MAX_GRADIENT_NORM = 0.25
POLICY_WEIGHT_DECAY = 0.01
POLICY_LOSS_SCALE = 128.0
POLICY_IGNORE_INDEX = -100
POLICY_FROZEN_PARAMETER = "positions.weight"
# Key bias shifts every key score equally for one query/head and is therefore a
# topology-derived softmax gauge direction, independent of observed values.
POLICY_ANALYTIC_GAUGE_NULL_PARAMETERS = ("first.key.1", "second.key.1")
POLICY_VALID_LENGTHS = ((3, 2), (2, 1), (3, 0))
MASKS = (
    (
        True, True, True, True, True, True, False, False, False,
        True, False, False, False, True, True, False, True, True,
    ),
    (
        True, True, True, True, True, True, True, True, True,
        True, False, False, False, False, False, False, False, False,
    ),
)
LOSS_MASKS = (
    (1.0, 1.0, 0.0, 1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
)
TOKENS = (0, 1, 2, 3, 4, 1)
TARGETS = (1, 3, 4, 2, 0, 4)
POSITIONS = (0, 1, 2, 0, 1, 2)
THREEFRY_PARITY = 0x1BD11BDA
THREEFRY_ROTATIONS = (13, 15, 26, 6, 17, 29, 16, 24)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def splitmix(seed: int) -> int:
    mask = (1 << 64) - 1
    value = (seed + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def uniform(shape: tuple[int, ...], low: float, high: float, seed: int) -> torch.Tensor:
    low = f32(low)
    high = f32(high)
    width = f32(high - low)
    values = []
    for index in range(math.prod(shape)):
        unit = f32(float(splitmix(seed + index) >> 40) / float(1 << 24))
        values.append(f32(low + f32(width * unit)))
    return torch.tensor(values, dtype=torch.float32).reshape(shape).requires_grad_()


def embedding(shape: tuple[int, int], seed: int) -> torch.Tensor:
    bound = f32(math.sqrt(f32(6.0 / f32(float(sum(shape))))))
    return uniform(shape, -bound, bound, seed)


def projection(params: OrderedDict[str, torch.Tensor], prefix: str, shape: tuple[int, int], seed: int) -> None:
    bound = f32(1.0 / f32(math.sqrt(f32(float(math.prod(shape))))))
    params[f"{prefix}.0"] = uniform(shape, -bound, bound, seed)
    params[f"{prefix}.1"] = torch.zeros(shape[1], dtype=torch.float32, requires_grad=True)


def layer_norm_parameter_names(prefix: str) -> tuple[str, str]:
    """Return the exact RustGrad module-traversal names for LayerNorm state."""
    if prefix.endswith((".ln1", ".ln2")):
        return f"{prefix}.0", f"{prefix}.1"
    return f"{prefix}.weight", f"{prefix}.bias"


def layer_norm_parameters(params: OrderedDict[str, torch.Tensor], prefix: str) -> None:
    weight, bias = layer_norm_parameter_names(prefix)
    params[weight] = torch.ones(EMBEDDING, dtype=torch.float32, requires_grad=True)
    params[bias] = torch.zeros(EMBEDDING, dtype=torch.float32, requires_grad=True)


def make_parameters() -> OrderedDict[str, torch.Tensor]:
    params: OrderedDict[str, torch.Tensor] = OrderedDict()
    params["tokens.weight"] = embedding((VOCAB, EMBEDDING), SEED)
    params["positions.weight"] = embedding((TIME, EMBEDDING), SEED + 1)
    for block, seed in (("first", SEED + 2), ("second", SEED + 3)):
        projection(params, f"{block}.query", (EMBEDDING, EMBEDDING), seed)
        projection(params, f"{block}.key", (EMBEDDING, EMBEDDING), seed + 1)
        projection(params, f"{block}.value", (EMBEDDING, EMBEDDING), seed + 2)
        projection(params, f"{block}.out", (EMBEDDING, EMBEDDING), seed + 3)
        projection(params, f"{block}.ff1", (EMBEDDING, FEED_FORWARD), seed + 4)
        params[f"{block}.ff1.1"] = torch.tensor(
            (2.0, -2.0, 2.25, -2.25, 2.5, -2.5, 2.75, -2.75),
            dtype=torch.float32,
            requires_grad=True,
        )
        projection(params, f"{block}.ff2", (FEED_FORWARD, EMBEDDING), seed + 5)
        layer_norm_parameters(params, f"{block}.ln1")
        layer_norm_parameters(params, f"{block}.ln2")
    layer_norm_parameters(params, "norm")
    params = OrderedDict(sorted(params.items()))
    assert len(params) == 36
    assert sum(parameter.numel() for parameter in params.values()) == 384
    return params


def rotate_left(value: int, amount: int) -> int:
    return ((value << amount) | (value >> (32 - amount))) & 0xFFFFFFFF


def threefry(counter: int) -> tuple[int, int]:
    keys = (DROPOUT_KEY[0], DROPOUT_KEY[1], DROPOUT_KEY[0] ^ DROPOUT_KEY[1] ^ THREEFRY_PARITY)
    x0 = ((counter & 0xFFFFFFFF) + keys[0]) & 0xFFFFFFFF
    x1 = (((counter >> 32) + keys[1]) & 0xFFFFFFFF)
    for round_index in range(20):
        x0 = (x0 + x1) & 0xFFFFFFFF
        x1 = rotate_left(x1, THREEFRY_ROTATIONS[round_index % 8]) ^ x0
        if round_index % 4 == 3:
            injection = round_index // 4 + 1
            x0 = (x0 + keys[injection % 3]) & 0xFFFFFFFF
            x1 = (x1 + keys[(injection + 1) % 3] + injection) & 0xFFFFFFFF
    return x0, x1


def compiled_dropout_mask(shape: tuple[int, ...], start: int) -> tuple[torch.Tensor, int]:
    words: list[int] = []
    blocks = (math.prod(shape) + 1) // 2
    for offset in range(start, start + blocks):
        words.extend(threefry(offset))
    keep = []
    for word in words[: math.prod(shape)]:
        bits = (word & 0x007FFFFF) | 0x3F800000
        unit = f32(struct.unpack("<f", struct.pack("<I", bits))[0] - 1.0)
        keep.append(unit >= DROPOUT)
    return torch.tensor(keep, dtype=torch.bool).reshape(shape), start + blocks


def replay_masks(replay: int) -> list[torch.Tensor]:
    start = (replay - 1) * 84
    masks = []
    for shape in (
        (BATCH, HEADS, TIME, TIME),
        (BATCH, TIME, EMBEDDING),
        (BATCH, TIME, EMBEDDING),
        (BATCH, HEADS, TIME, TIME),
        (BATCH, TIME, EMBEDDING),
        (BATCH, TIME, EMBEDDING),
    ):
        mask, start = compiled_dropout_mask(shape, start)
        masks.append(mask)
    assert start == replay * 84
    return masks


def layer_norm(value: torch.Tensor, params: OrderedDict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    mean = value.mean(dim=-1, keepdim=True)
    centered = value - mean
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    normalized = centered / torch.sqrt(variance + torch.tensor(1.0e-5, dtype=torch.float32))
    weight, bias = layer_norm_parameter_names(prefix)
    return normalized * params[weight] + params[bias]


def linear(value: torch.Tensor, params: OrderedDict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return value @ params[f"{prefix}.0"] + params[f"{prefix}.1"]


def apply_causal_attention_mask(caller: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    effective = caller.expand(BATCH, HEADS, TIME, TIME) & torch.tril(
        torch.ones(TIME, TIME, dtype=torch.bool)
    )
    return effective, effective.any(dim=-1, keepdim=True)


def combined_attention_mask(replay: int) -> tuple[torch.Tensor, torch.Tensor]:
    caller = torch.tensor(MASKS[(replay - 1) % len(MASKS)], dtype=torch.bool).reshape(
        BATCH, 1, TIME, TIME
    )
    return apply_causal_attention_mask(caller)


def apply_dropout(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, value, torch.zeros((), dtype=torch.float32)) / f32(1.0 - DROPOUT)


def block_forward(
    value: torch.Tensor,
    params: OrderedDict[str, torch.Tensor],
    prefix: str,
    effective_mask: torch.Tensor,
    row_valid: torch.Tensor,
    masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = layer_norm(value, params, f"{prefix}.ln1")
    heads = []
    for projection_name in ("query", "key", "value"):
        projected = linear(normalized, params, f"{prefix}.{projection_name}")
        heads.append(projected.reshape(BATCH, TIME, HEADS, HEAD_SIZE).permute(0, 2, 1, 3))
    query, key, projected_value = heads
    scores = (query @ key.transpose(-1, -2)) / f32(math.sqrt(float(HEAD_SIZE)))
    safe_mask = effective_mask | (~row_valid & torch.nn.functional.one_hot(
        torch.zeros(BATCH, HEADS, TIME, dtype=torch.int64), TIME
    ).to(torch.bool))
    probabilities = torch.softmax(scores.masked_fill(~safe_mask, -math.inf), dim=-1)
    probabilities = torch.where(row_valid, probabilities, torch.zeros((), dtype=torch.float32))
    dropped_probabilities = apply_dropout(probabilities, masks[0])
    attended = dropped_probabilities @ projected_value
    attended = attended.permute(0, 2, 1, 3).contiguous().reshape(BATCH, TIME, EMBEDDING)
    attended = linear(attended, params, f"{prefix}.out")
    residual = value + apply_dropout(attended, masks[1])
    ff_input = linear(layer_norm(residual, params, f"{prefix}.ln2"), params, f"{prefix}.ff1")
    ff_output = linear(torch.relu(ff_input), params, f"{prefix}.ff2")
    return residual + apply_dropout(ff_output, masks[2]), probabilities, ff_input


def forward_inputs(
    params: OrderedDict[str, torch.Tensor],
    replay: int,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    effective_mask: torch.Tensor,
    row_valid: torch.Tensor,
    ignore_index: int | None,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    masks = replay_masks(replay)
    positions = torch.tensor(POSITIONS, dtype=torch.int64).reshape(BATCH, TIME)
    value = params["tokens.weight"][tokens] + params["positions.weight"][positions]
    value, first_probabilities, first_relu_input = block_forward(
        value, params, "first", effective_mask, row_valid, masks[:3]
    )
    value, second_probabilities, second_relu_input = block_forward(
        value, params, "second", effective_mask, row_valid, masks[3:]
    )
    value = layer_norm(value, params, "norm")
    logits = value @ params["tokens.weight"].transpose(0, 1)
    if ignore_index is None:
        gather_targets = targets
    else:
        gather_targets = torch.where(targets == ignore_index, 0, targets)
    token_losses = -torch.log_softmax(logits.reshape(-1, VOCAB), dim=-1).gather(
        1, gather_targets.reshape(-1, 1)
    ).reshape(BATCH, TIME) + 1.0
    if ignore_index is not None:
        token_losses = torch.where(
            targets == ignore_index,
            torch.zeros((), dtype=torch.float32),
            token_losses,
        )
    numerator = (token_losses * loss_mask).sum()
    valid_token_count = loss_mask.sum()
    return {
        "logits": logits,
        "token_losses": token_losses,
        "loss": numerator / valid_token_count,
        "numerator": numerator,
        "valid_token_count": valid_token_count,
        "attention_probabilities": [first_probabilities, second_probabilities],
        "relu_inputs": [first_relu_input, second_relu_input],
        "dropout_masks": masks,
        "effective_attention_mask": effective_mask,
        "row_valid": row_valid,
    }


def forward(params: OrderedDict[str, torch.Tensor], replay: int) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    effective_mask, row_valid = combined_attention_mask(replay)
    return forward_inputs(
        params,
        replay,
        torch.tensor(TOKENS, dtype=torch.int64).reshape(BATCH, TIME),
        torch.tensor(TARGETS, dtype=torch.int64).reshape(BATCH, TIME),
        torch.tensor(
            LOSS_MASKS[(replay - 1) % len(LOSS_MASKS)], dtype=torch.float32
        ).reshape(BATCH, TIME),
        effective_mask,
        row_valid,
        None,
    )


def policy_batch(replay: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_lengths = POLICY_VALID_LENGTHS[(replay - 1) % len(POLICY_VALID_LENGTHS)]
    tokens = list(TOKENS)
    targets = list(TARGETS)
    loss_mask = []
    for batch, valid_length in enumerate(valid_lengths):
        for time in range(TIME):
            keep = time < valid_length
            loss_mask.append(1.0 if keep else 0.0)
            if not keep:
                lane = batch * TIME + time
                tokens[lane] = 0
                targets[lane] = POLICY_IGNORE_INDEX
    target_tensor = torch.tensor(targets, dtype=torch.int64).reshape(BATCH, TIME)
    caller = (target_tensor != POLICY_IGNORE_INDEX).reshape(BATCH, 1, 1, TIME)
    return (
        torch.tensor(tokens, dtype=torch.int64).reshape(BATCH, TIME),
        target_tensor,
        torch.tensor(loss_mask, dtype=torch.float32).reshape(BATCH, TIME),
        caller,
    )


def policy_forward(
    params: OrderedDict[str, torch.Tensor], replay: int
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    tokens, targets, loss_mask, caller = policy_batch(replay)
    effective_mask, row_valid = apply_causal_attention_mask(caller)
    return forward_inputs(
        params,
        replay,
        tokens,
        targets,
        loss_mask,
        effective_mask,
        row_valid,
        POLICY_IGNORE_INDEX,
    )


def tensor(value: torch.Tensor) -> dict[str, object]:
    detached = value.detach().cpu().contiguous()
    values = detached.reshape(-1).tolist()
    if detached.dtype == torch.bool:
        return {"shape": list(detached.shape), "bool_values": values}
    return {"shape": list(detached.shape), "values": values}


def tensor_map(values: OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor]) -> dict[str, object]:
    return {name: tensor(value) for name, value in sorted(values.items())}


def replay_fixture(
    params: OrderedDict[str, torch.Tensor], replay: int
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, object]]:
    result = forward(params, replay)
    gradients = torch.autograd.grad(result["numerator"], tuple(params.values()))
    gradient_map = OrderedDict(zip(params.keys(), gradients))
    expected_token_count = 5 if replay % 2 == 1 else 4
    assert int(result["valid_token_count"].item()) == expected_token_count
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(
        torch.count_nonzero(relu_input).item() > 0 for relu_input in result["relu_inputs"]
    )
    for probabilities in result["attention_probabilities"]:
        invalid_rows = ~result["row_valid"].expand_as(probabilities)
        assert torch.count_nonzero(probabilities[invalid_rows]).item() == 0
    mask_index = (replay - 1) % len(MASKS)
    fixture = {
        "replay": replay,
        "tokens": list(TOKENS),
        "targets": list(TARGETS),
        "attention_keep_mask": list(MASKS[mask_index]),
        "loss_mask": list(LOSS_MASKS[mask_index]),
        "effective_attention_mask": tensor(result["effective_attention_mask"]),
        "row_valid": tensor(result["row_valid"]),
        "dropout_masks": [tensor(mask) for mask in result["dropout_masks"]],
        "attention_probabilities": [
            tensor(probabilities) for probabilities in result["attention_probabilities"]
        ],
        "logits": tensor(result["logits"]),
        "token_losses": tensor(result["token_losses"]),
        "token_mean_loss": float(result["loss"].item()),
        "valid_token_count": int(result["valid_token_count"].item()),
        "numerator_gradients": tensor_map(gradient_map),
    }
    return gradient_map, fixture


def policy_replay_fixture(
    params: OrderedDict[str, torch.Tensor], replay: int
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, object], float]:
    result = policy_forward(params, replay)
    active_params = OrderedDict(
        (name, parameter)
        for name, parameter in params.items()
        if name != POLICY_FROZEN_PARAMETER
    )
    gradients = torch.autograd.grad(result["numerator"], tuple(active_params.values()))
    gradient_map = OrderedDict(zip(active_params.keys(), gradients))
    tokens, targets, loss_mask, caller = policy_batch(replay)
    expected_token_count = sum(
        POLICY_VALID_LENGTHS[(replay - 1) % len(POLICY_VALID_LENGTHS)]
    )
    assert int(result["valid_token_count"].item()) == expected_token_count
    assert len(gradient_map) == 35
    assert sum(gradient.numel() for gradient in gradient_map.values()) == 372
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(
        torch.count_nonzero(relu_input).item() > 0 for relu_input in result["relu_inputs"]
    )
    for probabilities in result["attention_probabilities"]:
        invalid_rows = ~result["row_valid"].expand_as(probabilities)
        assert torch.count_nonzero(probabilities[invalid_rows]).item() == 0
    if replay % len(POLICY_VALID_LENGTHS) == 0:
        assert not bool(result["row_valid"][1].any())
    fixture = {
        "replay": replay,
        "tokens": tokens.reshape(-1).tolist(),
        "targets": targets.reshape(-1).tolist(),
        "attention_keep_mask": caller.reshape(-1).tolist(),
        "loss_mask": loss_mask.reshape(-1).tolist(),
        "effective_attention_mask": tensor(result["effective_attention_mask"]),
        "row_valid": tensor(result["row_valid"]),
        "dropout_masks": [tensor(mask) for mask in result["dropout_masks"]],
        "attention_probabilities": [
            tensor(probabilities) for probabilities in result["attention_probabilities"]
        ],
        "logits": tensor(result["logits"]),
        "token_losses": tensor(result["token_losses"]),
        "token_mean_loss": float(result["loss"].item()),
        "valid_token_count": int(result["valid_token_count"].item()),
        "numerator_gradients": tensor_map(gradient_map),
    }
    return gradient_map, fixture, float(result["numerator"].item())


def adamw_window(
    params: OrderedDict[str, torch.Tensor],
    first_moments: OrderedDict[str, torch.Tensor],
    second_moments: OrderedDict[str, torch.Tensor],
    numerator_gradients: list[OrderedDict[str, torch.Tensor]],
    optimizer_step: int,
    valid_token_count: int,
    *,
    max_gradient_norm: float = MAX_GRADIENT_NORM,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = 0.0,
) -> tuple[
    OrderedDict[str, torch.Tensor],
    OrderedDict[str, torch.Tensor],
    OrderedDict[str, torch.Tensor],
    dict[str, object],
]:
    averaged = OrderedDict(
        (
            name,
            sum(
                (gradients[name] for gradients in numerator_gradients),
                start=torch.zeros_like(params[name]),
            )
            / f32(float(valid_token_count)),
        )
        for name in params
    )
    gradient_vector = torch.cat([gradient.reshape(-1) for gradient in averaged.values()])
    pre_clip_norm = torch.sqrt((gradient_vector * gradient_vector).sum())
    max_gradient_norm = torch.tensor(f32(max_gradient_norm), dtype=torch.float32)
    learning_rate = torch.tensor(f32(learning_rate), dtype=torch.float32)
    weight_decay = torch.tensor(f32(weight_decay), dtype=torch.float32)
    clip_scale = max_gradient_norm / torch.maximum(
        pre_clip_norm, max_gradient_norm
    )
    clipped = OrderedDict((name, gradient * clip_scale) for name, gradient in averaged.items())
    next_first_moments = OrderedDict(
        (
            name,
            f32(BETAS[0]) * first_moments[name] + f32(1.0 - BETAS[0]) * gradient,
        )
        for name, gradient in clipped.items()
    )
    next_second_moments = OrderedDict(
        (
            name,
            f32(BETAS[1]) * second_moments[name]
            + f32(1.0 - BETAS[1]) * gradient * gradient,
        )
        for name, gradient in clipped.items()
    )
    first_correction = f32(1.0 - BETAS[0] ** optimizer_step)
    second_correction = f32(1.0 - BETAS[1] ** optimizer_step)
    successors = OrderedDict()
    decay_factor = f32(1.0) - learning_rate * weight_decay
    for name, parameter in params.items():
        first = next_first_moments[name] / first_correction
        second = next_second_moments[name] / second_correction
        if weight_decay.item() == 0.0:
            successor = parameter - LEARNING_RATE * first / (torch.sqrt(second) + EPSILON)
        else:
            normalized = first / (torch.sqrt(second) + EPSILON)
            successor = parameter * decay_factor - learning_rate * normalized
        successors[name] = successor.detach().requires_grad_()

    fixture = {
        "optimizer_step": optimizer_step,
        "valid_token_count": valid_token_count,
        "pre_clip_norm": float(pre_clip_norm.item()),
        "clip_scale": float(clip_scale.item()),
        "first_moments": tensor_map(next_first_moments),
        "second_moments": tensor_map(next_second_moments),
        "parameter_successors": tensor_map(successors),
    }
    return successors, next_first_moments, next_second_moments, fixture


def generate_policy_frontier() -> dict[str, object]:
    params = make_parameters()
    frozen_parameter = params[POLICY_FROZEN_PARAMETER].detach().clone()
    active_params = OrderedDict(
        (name, parameter)
        for name, parameter in params.items()
        if name != POLICY_FROZEN_PARAMETER
    )
    initial_parameters = OrderedDict(
        (name, parameter.detach().clone()) for name, parameter in active_params.items()
    )
    assert all(
        name in active_params
        and torch.count_nonzero(initial_parameters[name]).item() == 0
        for name in POLICY_ANALYTIC_GAUGE_NULL_PARAMETERS
    )
    assert (
        sum(initial_parameters[name].numel() for name in POLICY_ANALYTIC_GAUGE_NULL_PARAMETERS)
        == 8
    )
    first_moments = OrderedDict(
        (name, torch.zeros_like(parameter)) for name, parameter in active_params.items()
    )
    second_moments = OrderedDict(
        (name, torch.zeros_like(parameter)) for name, parameter in active_params.items()
    )
    replays = []
    gradients_by_replay = []
    numerators = []

    def record_replay(replay: int) -> None:
        gradients, fixture, numerator = policy_replay_fixture(params, replay)
        gradients_by_replay.append(gradients)
        replays.append(fixture)
        numerators.append(numerator)

    def commit_window(
        replay_start: int,
        replay_end: int,
        optimizer_step: int,
        valid_token_count: int,
        learning_rate: float,
    ) -> dict[str, object]:
        nonlocal params, active_params, first_moments, second_moments
        prior_active = active_params
        next_active, first_moments, second_moments, window = adamw_window(
            active_params,
            first_moments,
            second_moments,
            gradients_by_replay[replay_start - 1 : replay_end],
            optimizer_step,
            valid_token_count,
            max_gradient_norm=POLICY_MAX_GRADIENT_NORM,
            learning_rate=learning_rate,
            weight_decay=POLICY_WEIGHT_DECAY,
        )
        assert prior_active.keys() == next_active.keys()
        params = OrderedDict(
            (
                name,
                frozen_parameter.detach().clone().requires_grad_()
                if name == POLICY_FROZEN_PARAMETER
                else next_active[name],
            )
            for name in params
        )
        active_params = OrderedDict(
            (name, parameter)
            for name, parameter in params.items()
            if name != POLICY_FROZEN_PARAMETER
        )
        assert torch.equal(params[POLICY_FROZEN_PARAMETER], frozen_parameter)
        window_numerator = f32(0.0)
        for numerator in numerators[replay_start - 1 : replay_end]:
            window_numerator = f32(window_numerator + numerator)
        window.update(
            {
                "learning_rate": learning_rate,
                "mean_loss": f32(window_numerator / f32(float(valid_token_count))),
                "microbatch_count": replay_end - replay_start + 1,
            }
        )
        return window

    for replay in range(1, 4):
        record_replay(replay)
    first_commit = commit_window(1, 3, 1, 11, 1.0e-3)

    record_replay(4)
    pending_checkpoint = {
        "replay_step": 4,
        "optimizer_step": 1,
        "accumulation_index": 1,
        "valid_token_count": 5,
        "dropout_counter": 4 * 84,
        "loss_numerator": numerators[3],
    }
    for replay in (5, 6):
        record_replay(replay)
    second_commit = commit_window(4, 6, 2, 11, 5.0e-4)

    record_replay(7)
    _, _, _, partial_flush = adamw_window(
        active_params,
        first_moments,
        second_moments,
        [gradients_by_replay[6]],
        3,
        5,
        max_gradient_norm=POLICY_MAX_GRADIENT_NORM,
        learning_rate=5.0e-4,
        weight_decay=POLICY_WEIGHT_DECAY,
    )
    partial_flush.update(
        {
            "learning_rate": 5.0e-4,
            "mean_loss": f32(numerators[6] / f32(5.0)),
            "microbatch_count": 1,
        }
    )
    assert torch.equal(params[POLICY_FROZEN_PARAMETER], frozen_parameter)
    return {
        "rustgrad_base": "0b9bae3c69e072e050e7f9c08daf8184c0630133",
        "weight_decay": POLICY_WEIGHT_DECAY,
        "loss_scale": POLICY_LOSS_SCALE,
        "accumulation_steps": 3,
        "max_gradient_norm": POLICY_MAX_GRADIENT_NORM,
        "ignore_index": POLICY_IGNORE_INDEX,
        "learning_rates": [1.0e-3, 5.0e-4],
        "active_parameter_count": len(active_params),
        "active_coordinate_count": sum(parameter.numel() for parameter in active_params.values()),
        "analytic_gauge_null_parameters": list(POLICY_ANALYTIC_GAUGE_NULL_PARAMETERS),
        "frozen_parameter_name": POLICY_FROZEN_PARAMETER,
        "frozen_parameter": tensor(frozen_parameter),
        "initial_parameters": tensor_map(initial_parameters),
        "replays": replays,
        "commits": [first_commit, second_commit],
        "pending_checkpoint": pending_checkpoint,
        "partial_flush": partial_flush,
    }


def main(output: Path) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    params = make_parameters()
    initial_parameters = OrderedDict(
        (name, parameter.detach().clone()) for name, parameter in params.items()
    )
    first_moments = OrderedDict(
        (name, torch.zeros_like(parameter)) for name, parameter in params.items()
    )
    second_moments = OrderedDict(
        (name, torch.zeros_like(parameter)) for name, parameter in params.items()
    )
    replays = []
    windows = []
    for optimizer_step in (1, 2):
        numerator_gradients = []
        for replay in (2 * optimizer_step - 1, 2 * optimizer_step):
            gradient_map, fixture = replay_fixture(params, replay)
            numerator_gradients.append(gradient_map)
            replays.append(fixture)
        params, first_moments, second_moments, window = adamw_window(
            params,
            first_moments,
            second_moments,
            numerator_gradients,
            optimizer_step,
            9,
        )
        windows.append(window)

    replay_five_gradients, replay_five = replay_fixture(params, 5)
    replays.append(replay_five)
    _, _, _, partial_flush = adamw_window(
        params,
        first_moments,
        second_moments,
        [replay_five_gradients],
        3,
        5,
    )

    fixture = {
        "provenance": {
            "generator": "tests/fixtures/generate_two_block_pytorch_frontier.py",
            "rustgrad_base": "dcba9ad2a311746caa340708054f32efea11e444",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "model_seed": SEED,
            "dropout_key": list(DROPOUT_KEY),
        },
        "traversal_name_count": 37,
        "canonical_parameter_count": 36,
        "canonical_coordinate_count": 384,
        "tied_names": ["tokens.weight", "lm_head.weight"],
        "initial_parameters": tensor_map(initial_parameters),
        "replays": replays,
        "windows": windows,
        "partial_flush": partial_flush,
        "policy_frontier": generate_policy_frontier(),
    }
    output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} OUTPUT.json")
    main(Path(sys.argv[1]))
