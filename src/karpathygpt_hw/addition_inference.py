"""Cached autoregressive inference for the trained long-addition KDA model."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from karpathygpt_hw.kda_mlx import (
    KDAAdditionModel,
    KimiDeltaAttention,
    addition_aligned_operand_indices,
    addition_column_position_encoding,
    gated_delta_metal_with_final_state,
)
from karpathygpt_hw.long_addition import DIGIT_TOKEN_OFFSET, TOKEN_TO_ID, TOKENS


@dataclass
class AttentionCache:
    state: mx.array
    q_history: mx.array
    k_history: mx.array
    v_history: mx.array


@dataclass(frozen=True)
class GeneratedAddition:
    answer: str
    reversed_answer: str
    trailing_space_correct: bool
    elapsed_seconds: float


def _attention_inputs(
    attention: KimiDeltaAttention,
    x: mx.array,
    q_projected: mx.array,
    k_projected: mx.array,
    v_projected: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    batch, steps, _ = x.shape
    shape = (batch, steps, attention.num_heads, attention.head_dim)
    q = attention.q_conv(q_projected).reshape(shape)
    k = attention.k_conv(k_projected).reshape(shape)
    v = attention.v_conv(v_projected).reshape(shape)
    q = q * mx.rsqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    q = q * attention.scale
    k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)

    decay_logits = attention.decay_up(attention.decay_down(x)).reshape(shape)
    decay_logits = decay_logits + attention.decay_bias
    log_decay = attention.gate_lower_bound * mx.sigmoid(
        mx.exp(attention.A_log)[None, None, :, None] * decay_logits
    )
    retention = mx.exp(log_decay)
    beta = mx.sigmoid(attention.beta_proj(x))
    return q, k, v, retention, beta


def _attention_output(
    attention: KimiDeltaAttention,
    x: mx.array,
    recurrence: mx.array,
) -> mx.array:
    batch, steps, _ = x.shape
    shape = (batch, steps, attention.num_heads, attention.head_dim)
    out = attention.output_norm(recurrence)
    out = out * mx.sigmoid(attention.output_gate(x).reshape(shape))
    return attention.out_proj(out.reshape(batch, steps, attention.dims))


def _prefill_attention(
    attention: KimiDeltaAttention,
    x: mx.array,
) -> tuple[mx.array, AttentionCache]:
    q_projected = attention.q_proj(x)
    k_projected = attention.k_proj(x)
    v_projected = attention.v_proj(x)
    q, k, v, retention, beta = _attention_inputs(
        attention, x, q_projected, k_projected, v_projected
    )
    recurrence, state = gated_delta_metal_with_final_state(q, k, v, retention, beta)
    history_steps = attention.q_conv.kernel_size - 1
    cache = AttentionCache(
        state=state,
        q_history=q_projected[:, -history_steps:],
        k_history=k_projected[:, -history_steps:],
        v_history=v_projected[:, -history_steps:],
    )
    return _attention_output(attention, x, recurrence), cache


def _conv_step(
    projected: mx.array,
    history: mx.array,
    convolution: object,
) -> tuple[mx.array, mx.array]:
    window = mx.concatenate((history, projected), axis=1)
    convolved = convolution(window)[:, -1:]
    return convolved, window[:, 1:]


def _decode_attention_step(
    attention: KimiDeltaAttention,
    x: mx.array,
    cache: AttentionCache,
) -> tuple[mx.array, AttentionCache]:
    q_projected = attention.q_proj(x)
    k_projected = attention.k_proj(x)
    v_projected = attention.v_proj(x)
    q_convolved, q_history = _conv_step(q_projected, cache.q_history, attention.q_conv)
    k_convolved, k_history = _conv_step(k_projected, cache.k_history, attention.k_conv)
    v_convolved, v_history = _conv_step(v_projected, cache.v_history, attention.v_conv)

    batch = x.shape[0]
    shape = (batch, 1, attention.num_heads, attention.head_dim)
    q = q_convolved.reshape(shape)
    k = k_convolved.reshape(shape)
    v = v_convolved.reshape(shape)
    q = q * mx.rsqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    q = q * attention.scale
    k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)

    decay_logits = attention.decay_up(attention.decay_down(x)).reshape(shape)
    decay_logits = decay_logits + attention.decay_bias
    log_decay = attention.gate_lower_bound * mx.sigmoid(
        mx.exp(attention.A_log)[None, None, :, None] * decay_logits
    )
    retention = mx.exp(log_decay)[:, 0]
    beta = mx.sigmoid(attention.beta_proj(x))[:, 0]
    q_t = q[:, 0].astype(mx.float32)
    k_t = k[:, 0].astype(mx.float32)
    v_t = v[:, 0].astype(mx.float32)

    state = cache.state * retention.astype(mx.float32)[:, :, :, None]
    predicted_value = mx.einsum("bhk,bhkv->bhv", k_t, state)
    residual = v_t - predicted_value
    state = state + mx.einsum("bhk,bhv->bhkv", beta.astype(mx.float32)[:, :, None] * k_t, residual)
    recurrence = mx.einsum("bhk,bhkv->bhv", q_t, state)[:, None]
    recurrence = recurrence.astype(v.dtype)
    new_cache = AttentionCache(state, q_history, k_history, v_history)
    return _attention_output(attention, x, recurrence), new_cache


def _prepared_embeddings(
    model: KDAAdditionModel,
    token_ids: mx.array,
) -> tuple[mx.array, mx.array]:
    raw = model.embedding(token_ids)
    prepared = raw
    if model.position_encoding == "addition_column_context":
        a_indices, b_indices, context_mask = addition_aligned_operand_indices(token_ids.shape[1])
        operand_a = mx.take(raw, a_indices, axis=1)
        operand_b = mx.take(raw, b_indices, axis=1)
        context = model.operand_a_projection(operand_a)
        context = context + model.operand_b_projection(operand_b)
        prepared = prepared + mx.where(context_mask[None, :, None], context, 0)
    if model.position_encoding == "addition_column":
        prepared = prepared + addition_column_position_encoding(token_ids.shape[1], model.dims)
    return raw, prepared


def _eval_cache(logits: mx.array, caches: list[AttentionCache]) -> None:
    arrays: list[mx.array] = [logits]
    for cache in caches:
        arrays.extend((cache.state, cache.q_history, cache.k_history, cache.v_history))
    mx.eval(*arrays)


def generate_addition(
    model: KDAAdditionModel,
    operand_a: str,
    operand_b: str,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> GeneratedAddition:
    """Generate an exact-width addition answer using recurrent KDA caches."""

    if not operand_a.isdigit() or not operand_b.isdigit():
        raise ValueError("operands must contain decimal digits only")
    digits = max(len(operand_a), len(operand_b))
    if digits < 1:
        raise ValueError("operands cannot be empty")
    operand_a = operand_a.zfill(digits)
    operand_b = operand_b.zfill(digits)
    prompt = f"{operand_a}+{operand_b}="
    token_values = [TOKEN_TO_ID[token] for token in prompt]
    token_values.extend([DIGIT_TOKEN_OFFSET] * (digits + 1))
    full_ids = mx.array(np.asarray([token_values], dtype=np.int32))

    model.eval()
    raw_embeddings, prepared_embeddings = _prepared_embeddings(model, full_ids)
    embedding_delta = prepared_embeddings - raw_embeddings
    prefix_steps = 2 * digits + 2
    x = prepared_embeddings[:, :prefix_steps]
    caches: list[AttentionCache] = []
    for block in model.blocks:
        attention_output, cache = _prefill_attention(block.attention, block.norm1(x))
        x = x + attention_output
        x = x + block.feed_forward(block.norm2(x))
        caches.append(cache)
    logits = model.lm_head(model.final_norm(x[:, -1:]))[:, 0]
    _eval_cache(logits, caches)

    started = time.perf_counter()
    answer_tokens: list[int] = []
    next_token = int(np.asarray(mx.argmax(logits, axis=-1))[0])
    answer_tokens.append(next_token)
    if progress is not None:
        progress(1, digits + 1)

    answer_input_start = 2 * digits + 2
    for answer_column in range(1, digits + 1):
        absolute_position = answer_input_start + answer_column - 1
        token = mx.array([[next_token]], dtype=mx.int32)
        x = model.embedding(token)
        x = x + embedding_delta[:, absolute_position : absolute_position + 1]
        new_caches: list[AttentionCache] = []
        for block, cache in zip(model.blocks, caches, strict=True):
            attention_output, cache = _decode_attention_step(block.attention, block.norm1(x), cache)
            x = x + attention_output
            x = x + block.feed_forward(block.norm2(x))
            new_caches.append(cache)
        caches = new_caches
        logits = model.lm_head(model.final_norm(x))[:, 0]
        _eval_cache(logits, caches)
        next_token = int(np.asarray(mx.argmax(logits, axis=-1))[0])
        answer_tokens.append(next_token)
        if progress is not None and (answer_column == digits or answer_column % 64 == 0):
            progress(answer_column + 1, digits + 1)

    final_position = 3 * digits + 2
    final_token = mx.array([[next_token]], dtype=mx.int32)
    x = model.embedding(final_token)
    x = x + embedding_delta[:, final_position : final_position + 1]
    new_caches = []
    for block, cache in zip(model.blocks, caches, strict=True):
        attention_output, cache = _decode_attention_step(block.attention, block.norm1(x), cache)
        x = x + attention_output
        x = x + block.feed_forward(block.norm2(x))
        new_caches.append(cache)
    trailing_logits = model.lm_head(model.final_norm(x))[:, 0]
    mx.eval(trailing_logits)
    trailing_token = int(np.asarray(mx.argmax(trailing_logits, axis=-1))[0])

    try:
        reversed_answer = "".join(TOKENS[token] for token in answer_tokens)
    except IndexError as error:
        raise RuntimeError("model emitted an out-of-vocabulary token") from error
    if not reversed_answer.isdigit():
        raise RuntimeError(f"model emitted a non-digit answer: {reversed_answer!r}")
    answer = reversed_answer[::-1].lstrip("0") or "0"
    return GeneratedAddition(
        answer=answer,
        reversed_answer=reversed_answer,
        trailing_space_correct=trailing_token == TOKEN_TO_ID[" "],
        elapsed_seconds=time.perf_counter() - started,
    )
