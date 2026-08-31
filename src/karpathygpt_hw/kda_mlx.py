"""Educational Kimi K3-style Delta Attention for MLX on Apple silicon.

The differentiable reference follows Equation 1 of the Kimi K3 technical
report. The Metal path supplies both a fused inference kernel and a custom VJP
for training. The VJP saves one recurrent-state checkpoint per 16-token chunk,
recomputes that chunk on-chip, and walks it backward without materializing a
state for every token in global memory.

Sources:
  https://github.com/MoonshotAI/Kimi-K3
  https://github.com/sgl-project/sglang-jax/tree/main/python/sgl_jax/srt/kernels/kda
"""

from __future__ import annotations

import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten


def gated_delta_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    retention: mx.array,
    beta: mx.array,
) -> mx.array:
    """Differentiable KDA recurrence.

    All inputs use ``[batch, time, heads, head_dim]`` except ``beta``, which
    uses ``[batch, time, heads]``. ``retention`` is ``exp(log_decay)``.
    """

    batch, steps, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    state = mx.zeros((batch, heads, key_dim, value_dim), dtype=mx.float32)
    outputs = []

    q32 = q.astype(mx.float32)
    k32 = k.astype(mx.float32)
    v32 = v.astype(mx.float32)
    retention32 = retention.astype(mx.float32)
    beta32 = beta.astype(mx.float32)

    for position in range(steps):
        q_t = q32[:, position]
        k_t = k32[:, position]
        v_t = v32[:, position]

        # S' = Diag(alpha_t) S_{t-1}
        state = state * retention32[:, position, :, :, None]

        # S_t = S' + beta_t k_t (v_t - k_t^T S')^T
        predicted_value = mx.einsum("bhk,bhkv->bhv", k_t, state)
        residual = v_t - predicted_value
        state = state + mx.einsum(
            "bhk,bhv->bhkv",
            beta32[:, position, :, None] * k_t,
            residual,
        )

        # o_t = S_t^T q_t
        outputs.append(mx.einsum("bhk,bhkv->bhv", q_t, state))

    return mx.stack(outputs, axis=1).astype(v.dtype)


_METAL_KERNELS: dict[tuple[int, int, int, str], Any] = {}
_TRAINING_FORWARD_KERNELS: dict[tuple[int, int, int, str], Any] = {}
_TRAINING_BACKWARD_KERNELS: dict[tuple[int, int, int, str], Any] = {}
_TRAINING_CHUNK_SIZE = 16


def _metal_kernel(steps: int, heads: int, head_dim: int, dtype: mx.Dtype) -> Any:
    if head_dim > 32:
        raise ValueError("The educational Metal kernel supports head_dim <= 32")

    key = (steps, heads, head_dim, str(dtype))
    if key in _METAL_KERNELS:
        return _METAL_KERNELS[key]

    source = f"""
        uint row = thread_position_in_threadgroup.x;
        uint batch_head = threadgroup_position_in_grid.x;
        uint batch_id = batch_head / {heads};
        uint head_id = batch_head - batch_id * {heads};

        threadgroup float contributions[{head_dim * head_dim}];
        threadgroup float predicted[{head_dim}];
        float state_row[{head_dim}];

        for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
            state_row[value_col] = 0.0f;
        }}

        for (uint position = 0; position < {steps}; ++position) {{
            uint base = ((batch_id * {steps} + position) * {heads} + head_id)
                        * {head_dim};
            float keep = float(retention[base + row]);
            float key_value = float(k[base + row]);
            float query_value = float(q[base + row]);

            for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                state_row[value_col] *= keep;
                contributions[row * {head_dim} + value_col] =
                    key_value * state_row[value_col];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (row == 0) {{
                for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                    float total = 0.0f;
                    for (uint key_row = 0; key_row < {head_dim}; ++key_row) {{
                        total += contributions[key_row * {head_dim} + value_col];
                    }}
                    predicted[value_col] = total;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float step_size = float(
                beta[(batch_id * {steps} + position) * {heads} + head_id]
            );
            for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                float error = float(v[base + value_col]) - predicted[value_col];
                state_row[value_col] += step_size * key_value * error;
                contributions[row * {head_dim} + value_col] =
                    query_value * state_row[value_col];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (row == 0) {{
                for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                    float total = 0.0f;
                    for (uint key_row = 0; key_row < {head_dim}; ++key_row) {{
                        total += contributions[key_row * {head_dim} + value_col];
                    }}
                    out[base + value_col] = T(total);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    """

    kernel = mx.fast.metal_kernel(
        name=f"educational_kda_t{steps}_h{heads}_d{head_dim}",
        input_names=["q", "k", "v", "retention", "beta"],
        output_names=["out"],
        source=source,
    )
    _METAL_KERNELS[key] = kernel
    return kernel


def gated_delta_metal(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    retention: mx.array,
    beta: mx.array,
) -> mx.array:
    """Fused forward-only Metal implementation of the recurrent KDA core."""

    if not mx.metal.is_available():
        return gated_delta_reference(q, k, v, retention, beta)
    if q.shape != k.shape or q.shape != v.shape or q.shape != retention.shape:
        raise ValueError("q, k, v, and retention must have identical shapes")

    batch, steps, heads, head_dim = q.shape
    if beta.shape != (batch, steps, heads):
        raise ValueError("beta must have shape [batch, time, heads]")

    kernel = _metal_kernel(steps, heads, head_dim, q.dtype)
    return kernel(
        inputs=[q, k, v, retention, beta],
        template=[("T", q.dtype)],
        grid=(batch * heads * head_dim, 1, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


def gated_delta_metal_with_final_state(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    retention: mx.array,
    beta: mx.array,
) -> tuple[mx.array, mx.array]:
    """Run fused KDA prefill and return its final recurrent state.

    The training-forward kernel already records a state checkpoint every 16
    tokens. Replaying only its final partial chunk recovers the exact final
    state without materializing one state per sequence position. This is used
    by cached autoregressive inference; it does not change the training path.
    """

    if q.shape != k.shape or q.shape != v.shape or q.shape != retention.shape:
        raise ValueError("q, k, v, and retention must have identical shapes")
    batch, steps, heads, head_dim = q.shape
    if steps < 1:
        raise ValueError("prefill requires at least one sequence step")
    if beta.shape != (batch, steps, heads):
        raise ValueError("beta must have shape [batch, time, heads]")

    if mx.metal.is_available() and head_dim <= 16:
        out, checkpoints = _gated_delta_training_forward(q, k, v, retention, beta)
        chunk_start = ((steps - 1) // _TRAINING_CHUNK_SIZE) * _TRAINING_CHUNK_SIZE
        state = checkpoints[:, -1]
    else:
        out = gated_delta_reference(q, k, v, retention, beta)
        chunk_start = 0
        state = mx.zeros((batch, heads, head_dim, head_dim), dtype=mx.float32)

    k32 = k.astype(mx.float32)
    v32 = v.astype(mx.float32)
    retention32 = retention.astype(mx.float32)
    beta32 = beta.astype(mx.float32)
    for position in range(chunk_start, steps):
        k_t = k32[:, position]
        v_t = v32[:, position]
        state = state * retention32[:, position, :, :, None]
        predicted_value = mx.einsum("bhk,bhkv->bhv", k_t, state)
        residual = v_t - predicted_value
        state = state + mx.einsum(
            "bhk,bhv->bhkv",
            beta32[:, position, :, None] * k_t,
            residual,
        )
    return out, state


def _training_forward_kernel(
    steps: int,
    heads: int,
    head_dim: int,
    dtype: mx.Dtype,
) -> Any:
    if head_dim > 16:
        raise ValueError("The training Metal kernel currently supports head_dim <= 16")

    key = (steps, heads, head_dim, str(dtype))
    if key in _TRAINING_FORWARD_KERNELS:
        return _TRAINING_FORWARD_KERNELS[key]

    chunk_size = _TRAINING_CHUNK_SIZE
    num_chunks = (steps + chunk_size - 1) // chunk_size
    source = f"""
        uint row = thread_position_in_threadgroup.x;
        uint batch_head = threadgroup_position_in_grid.x;
        uint batch_id = batch_head / {heads};
        uint head_id = batch_head - batch_id * {heads};

        threadgroup float contributions[{head_dim * head_dim}];
        threadgroup float predicted[{head_dim}];
        float state_row[{head_dim}];

        for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
            state_row[value_col] = 0.0f;
        }}

        for (uint position = 0; position < {steps}; ++position) {{
            uint chunk_id = position / {chunk_size};
            if ((position % {chunk_size}) == 0) {{
                uint checkpoint_base =
                    ((((batch_id * {num_chunks} + chunk_id) * {heads} + head_id)
                    * {head_dim} + row) * {head_dim});
                for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                    checkpoints[checkpoint_base + value_col] = state_row[value_col];
                }}
            }}

            uint base = ((batch_id * {steps} + position) * {heads} + head_id)
                        * {head_dim};
            float keep = float(retention[base + row]);
            float key_value = float(k[base + row]);
            float query_value = float(q[base + row]);

            for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                state_row[value_col] *= keep;
                contributions[row * {head_dim} + value_col] =
                    key_value * state_row[value_col];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (row == 0) {{
                for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                    float total = 0.0f;
                    for (uint key_row = 0; key_row < {head_dim}; ++key_row) {{
                        total += contributions[key_row * {head_dim} + value_col];
                    }}
                    predicted[value_col] = total;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float step_size = float(
                beta[(batch_id * {steps} + position) * {heads} + head_id]
            );
            for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                float error = float(v[base + value_col]) - predicted[value_col];
                state_row[value_col] += step_size * key_value * error;
                contributions[row * {head_dim} + value_col] =
                    query_value * state_row[value_col];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (row == 0) {{
                for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                    float total = 0.0f;
                    for (uint key_row = 0; key_row < {head_dim}; ++key_row) {{
                        total += contributions[key_row * {head_dim} + value_col];
                    }}
                    out[base + value_col] = T(total);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    """

    kernel = mx.fast.metal_kernel(
        name=f"educational_kda_train_fwd_t{steps}_h{heads}_d{head_dim}",
        input_names=["q", "k", "v", "retention", "beta"],
        output_names=["out", "checkpoints"],
        source=source,
    )
    _TRAINING_FORWARD_KERNELS[key] = kernel
    return kernel


def _training_backward_kernel(
    steps: int,
    heads: int,
    head_dim: int,
    dtype: mx.Dtype,
) -> Any:
    if head_dim > 16:
        raise ValueError("The training Metal kernel currently supports head_dim <= 16")

    key = (steps, heads, head_dim, str(dtype))
    if key in _TRAINING_BACKWARD_KERNELS:
        return _TRAINING_BACKWARD_KERNELS[key]

    chunk_size = _TRAINING_CHUNK_SIZE
    num_chunks = (steps + chunk_size - 1) // chunk_size
    source = f"""
        uint row = thread_position_in_threadgroup.x;
        uint batch_head = threadgroup_position_in_grid.x;
        uint batch_id = batch_head / {heads};
        uint head_id = batch_head - batch_id * {heads};

        threadgroup float contributions[{head_dim * head_dim}];
        threadgroup float saved_states[{chunk_size * head_dim * head_dim}];
        threadgroup float predicted[{head_dim}];
        threadgroup float residual[{head_dim}];
        threadgroup float grad_residual[{head_dim}];
        threadgroup float scalar_parts[{head_dim}];

        float state_row[{head_dim}];
        float grad_state_row[{head_dim}];
        float previous_state_row[{head_dim}];
        float decayed_state_row[{head_dim}];

        for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
            grad_state_row[value_col] = 0.0f;
        }}

        for (int chunk_id = {num_chunks - 1}; chunk_id >= 0; --chunk_id) {{
            uint chunk_start = uint(chunk_id) * {chunk_size};
            uint valid = min(uint({chunk_size}), uint({steps}) - chunk_start);
            uint checkpoint_base =
                ((((batch_id * {num_chunks} + uint(chunk_id)) * {heads} + head_id)
                * {head_dim} + row) * {head_dim});

            for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                state_row[value_col] = float(checkpoints[checkpoint_base + value_col]);
            }}

            // Recompute this chunk and retain its states in threadgroup memory.
            for (uint local_position = 0; local_position < valid; ++local_position) {{
                uint position = chunk_start + local_position;
                uint base = ((batch_id * {steps} + position) * {heads} + head_id)
                            * {head_dim};
                float keep = float(retention[base + row]);
                float key_value = float(k[base + row]);

                for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                    state_row[value_col] *= keep;
                    contributions[row * {head_dim} + value_col] =
                        key_value * state_row[value_col];
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (row == 0) {{
                    for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                        float total = 0.0f;
                        for (uint key_row = 0; key_row < {head_dim}; ++key_row) {{
                            total += contributions[key_row * {head_dim} + value_col];
                        }}
                        predicted[value_col] = total;
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

                residual[row] = float(v[base + row]) - predicted[row];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float step_size = float(
                    beta[(batch_id * {steps} + position) * {heads} + head_id]
                );
                for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                    state_row[value_col] += step_size * key_value * residual[value_col];
                    uint saved_index =
                        ((local_position * {head_dim} + row) * {head_dim}) + value_col;
                    saved_states[saved_index] = state_row[value_col];
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}

            // Reverse-mode differentiation through the recomputed chunk.
            for (int local_position = int(valid) - 1; local_position >= 0;
                 --local_position) {{
                uint position = chunk_start + uint(local_position);
                uint base = ((batch_id * {steps} + position) * {heads} + head_id)
                            * {head_dim};
                float keep = float(retention[base + row]);
                float key_value = float(k[base + row]);
                float query_value = float(q[base + row]);
                float step_size = float(
                    beta[(batch_id * {steps} + position) * {heads} + head_id]
                );

                float query_gradient = 0.0f;
                for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                    uint saved_index =
                        ((uint(local_position) * {head_dim} + row) * {head_dim})
                        + value_col;
                    state_row[value_col] = saved_states[saved_index];
                    if (local_position == 0) {{
                        previous_state_row[value_col] =
                            float(checkpoints[checkpoint_base + value_col]);
                    }} else {{
                        uint previous_index =
                            (((uint(local_position) - 1) * {head_dim} + row)
                            * {head_dim}) + value_col;
                        previous_state_row[value_col] = saved_states[previous_index];
                    }}
                    decayed_state_row[value_col] =
                        keep * previous_state_row[value_col];

                    float output_gradient = float(cotangent[base + value_col]);
                    query_gradient += state_row[value_col] * output_gradient;
                    grad_state_row[value_col] += query_value * output_gradient;
                    contributions[row * {head_dim} + value_col] =
                        key_value * decayed_state_row[value_col];
                }}
                q_grad[base + row] = T(query_gradient);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (row == 0) {{
                    for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                        float total = 0.0f;
                        for (uint key_row = 0; key_row < {head_dim}; ++key_row) {{
                            total += contributions[key_row * {head_dim} + value_col];
                        }}
                        predicted[value_col] = total;
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                residual[row] = float(v[base + row]) - predicted[row];
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // d(residual) = beta * G^T k
                for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                    contributions[row * {head_dim} + value_col] =
                        grad_state_row[value_col] * key_value;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (row == 0) {{
                    for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                        float total = 0.0f;
                        for (uint key_row = 0; key_row < {head_dim}; ++key_row) {{
                            total += contributions[key_row * {head_dim} + value_col];
                        }}
                        grad_residual[value_col] = step_size * total;
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float direct_key_gradient = 0.0f;
                float indirect_key_gradient = 0.0f;
                float beta_part = 0.0f;
                float retention_gradient = 0.0f;
                for (uint value_col = 0; value_col < {head_dim}; ++value_col) {{
                    direct_key_gradient +=
                        step_size * grad_state_row[value_col] * residual[value_col];
                    indirect_key_gradient +=
                        decayed_state_row[value_col] * grad_residual[value_col];
                    beta_part += grad_state_row[value_col] * residual[value_col];

                    float grad_decayed =
                        grad_state_row[value_col] - key_value * grad_residual[value_col];
                    retention_gradient +=
                        grad_decayed * previous_state_row[value_col];
                    grad_state_row[value_col] = keep * grad_decayed;
                }}
                k_grad[base + row] = T(direct_key_gradient - indirect_key_gradient);
                v_grad[base + row] = T(grad_residual[row]);
                retention_grad[base + row] = T(retention_gradient);

                scalar_parts[row] = key_value * beta_part;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (row == 0) {{
                    float beta_gradient = 0.0f;
                    for (uint key_row = 0; key_row < {head_dim}; ++key_row) {{
                        beta_gradient += scalar_parts[key_row];
                    }}
                    beta_grad[(batch_id * {steps} + position) * {heads} + head_id] =
                        T(beta_gradient);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
        }}
    """

    kernel = mx.fast.metal_kernel(
        name=f"educational_kda_train_bwd_t{steps}_h{heads}_d{head_dim}",
        input_names=[
            "q",
            "k",
            "v",
            "retention",
            "beta",
            "checkpoints",
            "cotangent",
        ],
        output_names=["q_grad", "k_grad", "v_grad", "retention_grad", "beta_grad"],
        source=source,
    )
    _TRAINING_BACKWARD_KERNELS[key] = kernel
    return kernel


def _gated_delta_training_forward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    retention: mx.array,
    beta: mx.array,
) -> tuple[mx.array, mx.array]:
    batch, steps, heads, head_dim = q.shape
    num_chunks = (steps + _TRAINING_CHUNK_SIZE - 1) // _TRAINING_CHUNK_SIZE
    kernel = _training_forward_kernel(steps, heads, head_dim, q.dtype)
    out, checkpoints = kernel(
        inputs=[q, k, v, retention, beta],
        template=[("T", q.dtype)],
        grid=(batch * heads * head_dim, 1, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[q.shape, (batch, num_chunks, heads, head_dim, head_dim)],
        output_dtypes=[q.dtype, mx.float32],
    )
    return out, checkpoints


def _gated_delta_training_backward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    retention: mx.array,
    beta: mx.array,
    checkpoints: mx.array,
    cotangent: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    batch, steps, heads, head_dim = q.shape
    kernel = _training_backward_kernel(steps, heads, head_dim, q.dtype)
    gradients = kernel(
        inputs=[q, k, v, retention, beta, checkpoints, cotangent],
        template=[("T", q.dtype)],
        grid=(batch * heads * head_dim, 1, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[q.shape, k.shape, v.shape, retention.shape, beta.shape],
        output_dtypes=[q.dtype, k.dtype, v.dtype, retention.dtype, beta.dtype],
    )
    return tuple(gradients)


@mx.custom_function
def _gated_delta_metal_custom_vjp(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    retention: mx.array,
    beta: mx.array,
) -> tuple[mx.array, mx.array]:
    return _gated_delta_training_forward(q, k, v, retention, beta)


@_gated_delta_metal_custom_vjp.vjp
def _gated_delta_metal_vjp(
    primals: tuple[mx.array, ...],
    cotangents: tuple[mx.array, ...],
    outputs: tuple[mx.array, ...],
) -> tuple[mx.array, ...]:
    q, k, v, retention, beta = primals
    output_cotangent, _ = cotangents
    _, checkpoints = outputs
    return _gated_delta_training_backward(
        q,
        k,
        v,
        retention,
        beta,
        checkpoints,
        output_cotangent,
    )


def gated_delta_metal_trainable(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    retention: mx.array,
    beta: mx.array,
) -> mx.array:
    """Fused Metal KDA recurrence with a checkpointed custom backward pass."""

    if not mx.metal.is_available():
        return gated_delta_reference(q, k, v, retention, beta)
    if q.shape != k.shape or q.shape != v.shape or q.shape != retention.shape:
        raise ValueError("q, k, v, and retention must have identical shapes")
    batch, steps, heads, _ = q.shape
    if beta.shape != (batch, steps, heads):
        raise ValueError("beta must have shape [batch, time, heads]")
    out, _ = _gated_delta_metal_custom_vjp(q, k, v, retention, beta)
    return out


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise kernel-4 causal convolution used on KDA's Q/K/V paths."""

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        padded = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        return nn.silu(self.conv(padded))


class KimiDeltaAttention(nn.Module):
    """Kimi K3-style KDA layer with bounded decay and full-rank output gate."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        *,
        short_conv_kernel: int = 4,
        gate_rank: int | None = None,
        gate_lower_bound: float = -5.0,
        decay_half_life_range: tuple[float, float] | None = None,
        use_fused_kernel: bool = False,
    ):
        super().__init__()
        if dims % num_heads:
            raise ValueError("dims must be divisible by num_heads")

        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = self.head_dim**-0.5
        self.gate_lower_bound = gate_lower_bound
        self.use_fused_kernel = use_fused_kernel
        gate_rank = gate_rank or self.head_dim

        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.v_proj = nn.Linear(dims, dims, bias=False)
        self.q_conv = CausalDepthwiseConv1d(dims, short_conv_kernel)
        self.k_conv = CausalDepthwiseConv1d(dims, short_conv_kernel)
        self.v_conv = CausalDepthwiseConv1d(dims, short_conv_kernel)

        self.decay_down = nn.Linear(dims, gate_rank, bias=False)
        self.decay_up = nn.Linear(gate_rank, dims, bias=False)
        self.beta_proj = nn.Linear(dims, num_heads, bias=False)
        self.output_gate = nn.Linear(dims, dims, bias=False)
        self.output_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.out_proj = nn.Linear(dims, dims, bias=False)

        # A_log=0 matches K3. By default, a bias of -4 gives the tiny toy model
        # a roughly seven-token half-life. Long algorithmic tasks can instead
        # initialize a geometric spread of memory timescales across channels.
        self.A_log = mx.zeros((num_heads,))
        if decay_half_life_range is None:
            self.decay_bias = mx.full((num_heads, self.head_dim), -4.0)
        else:
            minimum, maximum = decay_half_life_range
            if minimum <= 0 or maximum < minimum:
                raise ValueError("decay_half_life_range must contain positive increasing values")
            channel_count = num_heads * self.head_dim
            if channel_count == 1:
                half_lives = [minimum]
            else:
                ratio = (maximum / minimum) ** (1.0 / (channel_count - 1))
                half_lives = [minimum * ratio**index for index in range(channel_count)]
            # exp(-5 * sigmoid(bias)) ** half_life = 0.5
            probabilities = [math.log(2.0) / (5.0 * value) for value in half_lives]
            if any(probability >= 1.0 for probability in probabilities):
                raise ValueError("decay half-lives must be greater than log(2) / 5")
            biases = [math.log(value / (1.0 - value)) for value in probabilities]
            self.decay_bias = mx.array(biases).reshape(num_heads, self.head_dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch, steps, _ = x.shape
        shape = (batch, steps, self.num_heads, self.head_dim)

        q = self.q_conv(self.q_proj(x)).reshape(shape)
        k = self.k_conv(self.k_proj(x)).reshape(shape)
        v = self.v_conv(self.v_proj(x)).reshape(shape)

        # K3 uses L2-normalized Q/K; only Q receives the 1/sqrt(d) scale.
        q = q * mx.rsqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6) * self.scale
        k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)

        decay_logits = self.decay_up(self.decay_down(x)).reshape(shape)
        decay_logits = decay_logits + self.decay_bias
        log_decay = self.gate_lower_bound * mx.sigmoid(
            mx.exp(self.A_log)[None, None, :, None] * decay_logits
        )
        retention = mx.exp(log_decay)
        beta = mx.sigmoid(self.beta_proj(x))

        if self.use_fused_kernel:
            recurrence = gated_delta_metal_trainable if self.training else gated_delta_metal
        else:
            recurrence = gated_delta_reference
        out = recurrence(q, k, v, retention, beta)
        out = self.output_norm(out)
        out = out * mx.sigmoid(self.output_gate(x).reshape(shape))
        return self.out_proj(out.reshape(batch, steps, self.dims))


class CausalSelfAttention(nn.Module):
    """Fused MLX softmax attention baseline with matching model dimensions."""

    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiHeadAttention(dims, num_heads)

    def __call__(self, x: mx.array) -> mx.array:
        return self.attention(x, x, x, mask="causal")


class FeedForward(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.up = nn.Linear(dims, 4 * dims)
        self.down = nn.Linear(4 * dims, dims)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.gelu(self.up(x)))


class KDABlock(nn.Module):
    """A Transformer block whose communication layer is KDA, not softmax MHA."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        *,
        decay_half_life_range: tuple[float, float] | None = None,
        use_fused_kernel: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims)
        self.attention = KimiDeltaAttention(
            dims,
            num_heads,
            decay_half_life_range=decay_half_life_range,
            use_fused_kernel=use_fused_kernel,
        )
        self.norm2 = nn.LayerNorm(dims)
        self.feed_forward = FeedForward(dims)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class KDAAdditionModel(nn.Module):
    """Decoder-only long-addition model using KDA in every block.

    K3 itself is hybrid (three KDA layers followed by one global MLA layer).
    The default all-KDA path uses NoPE, as K3 does, because the recurrent state
    and short convolution encode sequence order. Optional addition-specific
    modes are kept separate so the general reference path remains unchanged.
    """

    def __init__(
        self,
        vocab_size: int = 13,
        dims: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        *,
        decay_half_life_range: tuple[float, float] | None = None,
        use_fused_kernel: bool = False,
        position_encoding: str = "none",
    ):
        super().__init__()
        if position_encoding not in {
            "none",
            "addition_column",
            "addition_column_context",
        }:
            raise ValueError(
                "position_encoding must be 'none', 'addition_column', or 'addition_column_context'"
            )
        self.dims = dims
        self.position_encoding = position_encoding
        self.embedding = nn.Embedding(vocab_size, dims)
        if position_encoding == "addition_column_context":
            self.operand_a_projection = nn.Linear(dims, dims, bias=False)
            self.operand_b_projection = nn.Linear(dims, dims, bias=False)
        self.blocks = [
            KDABlock(
                dims,
                num_heads,
                decay_half_life_range=decay_half_life_range,
                use_fused_kernel=use_fused_kernel,
            )
            for _ in range(num_layers)
        ]
        self.final_norm = nn.LayerNorm(dims)
        self.lm_head = nn.Linear(dims, vocab_size)

    def __call__(self, token_ids: mx.array) -> mx.array:
        x = self.embedding(token_ids)
        if self.position_encoding == "addition_column_context":
            a_indices, b_indices, context_mask = addition_aligned_operand_indices(
                token_ids.shape[1]
            )
            operand_a = mx.take(x, a_indices, axis=1)
            operand_b = mx.take(x, b_indices, axis=1)
            context = self.operand_a_projection(operand_a)
            context = context + self.operand_b_projection(operand_b)
            x = x + mx.where(context_mask[None, :, None], context, 0)
        if self.position_encoding == "addition_column":
            x = x + addition_column_position_encoding(token_ids.shape[1], self.dims)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


def addition_column_position_ids(sequence_steps: int) -> mx.array:
    """Return arithmetic-column IDs for a shifted long-addition input.

    The model input has ``A+B=reverse(sum)`` layout and length ``3 * n + 3``.
    Operand digits with the same distance from the least-significant end share
    an ID. On the answer side, the hidden state that predicts answer column
    ``j`` receives ID ``j``; this accounts for the one-token language-model
    shift. The signal is deterministic, so it extends beyond trained widths
    instead of learning a finite absolute-position table.
    """

    if sequence_steps < 6 or (sequence_steps - 3) % 3:
        raise ValueError("addition-column positions require an input length of 3 * digits + 3")
    digits = (sequence_steps - 3) // 3
    positions = mx.arange(sequence_steps, dtype=mx.int32)
    operand_a = digits - 1 - positions
    operand_b = 2 * digits - positions
    answer_query = positions - (2 * digits + 1)
    delimiter = mx.full((sequence_steps,), digits + 1, dtype=mx.int32)
    return mx.where(
        positions < digits,
        operand_a,
        mx.where(
            (positions > digits) & (positions <= 2 * digits),
            operand_b,
            mx.where(positions >= 2 * digits + 1, answer_query, delimiter),
        ),
    )


def addition_aligned_operand_indices(
    sequence_steps: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Map answer-query positions to the matching A and B digit positions.

    The returned mask is true only for the ``digits`` answer queries that have
    an operand pair. The final carry and trailing-space queries remain entirely
    autoregressive and must be handled by the KDA recurrence.
    """

    if sequence_steps < 6 or (sequence_steps - 3) % 3:
        raise ValueError("aligned addition context requires an input length of 3 * digits + 3")
    digits = (sequence_steps - 3) // 3
    positions = mx.arange(sequence_steps, dtype=mx.int32)
    answer_column = positions - (2 * digits + 1)
    context_mask = (answer_column >= 0) & (answer_column < digits)
    operand_a = mx.clip(digits - 1 - answer_column, 0, digits - 1)
    operand_b = mx.clip(2 * digits - answer_column, digits + 1, 2 * digits)
    return operand_a, operand_b, context_mask


def addition_column_position_encoding(sequence_steps: int, dims: int) -> mx.array:
    """Create fixed sinusoidal features for arithmetic-column IDs."""

    if dims < 2:
        raise ValueError("addition-column positions require dims >= 2")
    column_ids = addition_column_position_ids(sequence_steps).astype(mx.float32)
    frequencies = mx.exp(-math.log(10_000.0) * mx.arange(0, dims, 2, dtype=mx.float32) / dims)
    angles = column_ids[:, None] * frequencies[None, :]
    encoding = mx.stack((mx.sin(angles), mx.cos(angles)), axis=-1).reshape(sequence_steps, -1)
    return encoding[None, :, :dims]


@dataclass(frozen=True)
class AttentionBenchmark:
    sequence_length: int
    softmax_mha_ms: float
    kimi_delta_ms: float
    measured_speedup: float
    idealized_core_speedup: float
    attention_matrix_reduction: float


@dataclass(frozen=True)
class KDATrainingBenchmark:
    sequence_length: int
    autograd_reference_ms: float
    fused_metal_ms: float
    training_core_speedup: float
    checkpoint_state_reduction: float


@dataclass(frozen=True)
class KDALayerTrainingBenchmark:
    sequence_length: int
    batch_size: int
    autograd_reference_step_ms: float
    fused_metal_step_ms: float
    full_layer_training_speedup: float


def _benchmark_call(
    function: Any,
    *,
    warmup: int,
    iterations: int,
    repeats: int = 5,
) -> float:
    for _ in range(warmup):
        mx.eval(function())
    samples = []
    for _ in range(repeats):
        mx.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            mx.eval(function())
        mx.synchronize()
        samples.append(1_000 * (time.perf_counter() - started) / iterations)
    return statistics.median(samples)


def benchmark_attention(
    sequence_lengths: tuple[int, ...] = (12, 64, 256, 1_024, 2_048, 4_096),
    *,
    batch_size: int = 16,
    dims: int = 64,
    num_heads: int = 4,
    warmup: int = 5,
    iterations: int = 20,
) -> list[dict[str, float | int]]:
    """Benchmark complete attention layers on the current Apple GPU.

    The MHA baseline uses MLX's fused causal SDPA. KDA includes Q/K/V
    projections, short convolutions, decay/write/output gates, normalization,
    the fused recurrent core, and its output projection.
    """

    if not mx.metal.is_available():
        raise RuntimeError("The fused benchmark requires an Apple Metal GPU")

    softmax_attention = CausalSelfAttention(dims, num_heads)
    kimi_delta = KimiDeltaAttention(
        dims,
        num_heads,
        use_fused_kernel=True,
    )
    softmax_attention.eval()
    kimi_delta.eval()
    mx.eval(softmax_attention.parameters(), kimi_delta.parameters())
    head_dim = dims // num_heads
    rows = []

    for length in sequence_lengths:
        x = mx.random.normal((batch_size, length, dims))
        mx.eval(x)
        local_iterations = min(iterations, 10) if length >= 2_048 else iterations
        mha_ms = _benchmark_call(
            lambda inputs=x: softmax_attention(inputs),
            warmup=warmup,
            iterations=local_iterations,
        )
        kda_ms = _benchmark_call(
            lambda inputs=x: kimi_delta(inputs),
            warmup=warmup,
            iterations=local_iterations,
        )
        row = AttentionBenchmark(
            sequence_length=length,
            softmax_mha_ms=mha_ms,
            kimi_delta_ms=kda_ms,
            measured_speedup=mha_ms / kda_ms,
            # Attention core work: ~2*T^2*d versus ~4*T*d^2.
            idealized_core_speedup=length / (2 * head_dim),
            # Materialized score matrix entries versus recurrent-state entries.
            attention_matrix_reduction=(length / head_dim) ** 2,
        )
        rows.append(asdict(row))

    return rows


def benchmark_kda_training(
    sequence_lengths: tuple[int, ...] = (12, 64, 256, 1_024),
    *,
    batch_size: int = 16,
    dims: int = 64,
    num_heads: int = 4,
    warmup: int = 3,
    iterations: int = 10,
) -> list[dict[str, float | int]]:
    """Benchmark forward+backward for the KDA recurrence on Apple Metal.

    This compares the previous MLX-autograd recurrence against the custom
    checkpointed Metal VJP. It isolates the recurrent training core rather
    than comparing full KDA and softmax-attention layers.
    """

    if not mx.metal.is_available():
        raise RuntimeError("The fused training benchmark requires an Apple Metal GPU")
    if dims % num_heads:
        raise ValueError("dims must be divisible by num_heads")
    head_dim = dims // num_heads
    if head_dim > 16:
        raise ValueError("The training Metal kernel currently supports head_dim <= 16")

    rows = []
    for length in sequence_lengths:
        shape = (batch_size, length, num_heads, head_dim)
        q = mx.random.normal(shape) * 0.1
        k = mx.random.normal(shape) * 0.1
        v = mx.random.normal(shape) * 0.1
        retention = mx.full(shape, 0.98)
        beta = mx.full((batch_size, length, num_heads), 0.4)
        cotangent = mx.random.normal(shape)
        mx.eval(q, k, v, retention, beta, cotangent)

        def reference_loss(
            q: mx.array,
            k: mx.array,
            v: mx.array,
            retention: mx.array,
            beta: mx.array,
            weights: mx.array = cotangent,
        ) -> mx.array:
            return mx.mean(gated_delta_reference(q, k, v, retention, beta) * weights)

        def fused_loss(
            q: mx.array,
            k: mx.array,
            v: mx.array,
            retention: mx.array,
            beta: mx.array,
            weights: mx.array = cotangent,
        ) -> mx.array:
            return mx.mean(gated_delta_metal_trainable(q, k, v, retention, beta) * weights)

        reference_step = mx.compile(mx.value_and_grad(reference_loss, argnums=(0, 1, 2, 3, 4)))
        fused_step = mx.compile(mx.value_and_grad(fused_loss, argnums=(0, 1, 2, 3, 4)))
        arguments = (q, k, v, retention, beta)
        # Force graph and Metal compilation outside the timed region.
        mx.eval(reference_step(*arguments), fused_step(*arguments))
        local_iterations = min(iterations, 5) if length >= 256 else iterations
        reference_ms = _benchmark_call(
            lambda step=reference_step, inputs=arguments: step(*inputs),
            warmup=warmup,
            iterations=local_iterations,
        )
        fused_ms = _benchmark_call(
            lambda step=fused_step, inputs=arguments: step(*inputs),
            warmup=warmup,
            iterations=local_iterations,
        )
        num_chunks = (length + _TRAINING_CHUNK_SIZE - 1) // _TRAINING_CHUNK_SIZE
        row = KDATrainingBenchmark(
            sequence_length=length,
            autograd_reference_ms=reference_ms,
            fused_metal_ms=fused_ms,
            training_core_speedup=reference_ms / fused_ms,
            checkpoint_state_reduction=length / num_chunks,
        )
        rows.append(asdict(row))

    return rows


def benchmark_kda_layer_training(
    sequence_lengths: tuple[int, ...] = (12, 64, 256, 1_024),
    *,
    batch_size: int = 4,
    dims: int = 64,
    num_heads: int = 4,
    warmup: int = 2,
    iterations: int = 5,
) -> list[dict[str, float | int]]:
    """Benchmark an optimizer-inclusive KDA attention-layer training step.

    Each timing includes projections, causal depthwise convolutions, decay and
    output gates, normalization, the recurrence forward/backward, and an AdamW
    parameter update. Compilation and the first Metal kernel build are excluded.
    This complements :func:`benchmark_kda_training`, which isolates the recurrent
    core and intentionally excludes the optimizer and surrounding layer.
    """

    if not mx.metal.is_available():
        raise RuntimeError("The fused training benchmark requires an Apple Metal GPU")
    if dims % num_heads:
        raise ValueError("dims must be divisible by num_heads")
    if dims // num_heads > 16:
        raise ValueError("The training Metal kernel currently supports head_dim <= 16")

    rows = []
    for length in sequence_lengths:
        reference = KimiDeltaAttention(dims, num_heads, use_fused_kernel=False)
        fused = KimiDeltaAttention(dims, num_heads, use_fused_kernel=True)
        fused.update(reference.parameters())
        reference.train()
        fused.train()
        reference_optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)
        fused_optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)
        inputs = mx.random.normal((batch_size, length, dims)) * 0.1
        targets = mx.random.normal((batch_size, length, dims)) * 0.1
        mx.eval(reference.parameters(), fused.parameters(), inputs, targets)

        def layer_loss(
            model: KimiDeltaAttention,
            x: mx.array,
            target: mx.array,
        ) -> mx.array:
            difference = model(x) - target
            return mx.mean(difference * difference)

        reference_value_and_grad = nn.value_and_grad(reference, layer_loss)
        fused_value_and_grad = nn.value_and_grad(fused, layer_loss)

        def reference_step(
            x: mx.array,
            target: mx.array,
            value_and_grad: Any = reference_value_and_grad,
            model: KimiDeltaAttention = reference,
            optimizer: Any = reference_optimizer,
        ) -> mx.array:
            loss, gradients = value_and_grad(model, x, target)
            optimizer.update(model, gradients)
            return loss

        def fused_step(
            x: mx.array,
            target: mx.array,
            value_and_grad: Any = fused_value_and_grad,
            model: KimiDeltaAttention = fused,
            optimizer: Any = fused_optimizer,
        ) -> mx.array:
            loss, gradients = value_and_grad(model, x, target)
            optimizer.update(model, gradients)
            return loss

        compiled_reference_step = mx.compile(
            reference_step,
            inputs=[reference.state, reference_optimizer.state],
            outputs=[reference.state, reference_optimizer.state],
        )
        compiled_fused_step = mx.compile(
            fused_step,
            inputs=[fused.state, fused_optimizer.state],
            outputs=[fused.state, fused_optimizer.state],
        )

        def call_reference(
            compiled_step: Any = compiled_reference_step,
            x: mx.array = inputs,
            target: mx.array = targets,
            model: KimiDeltaAttention = reference,
            optimizer: Any = reference_optimizer,
        ) -> tuple[Any, ...]:
            loss = compiled_step(x, target)
            return loss, model.state, optimizer.state

        def call_fused(
            compiled_step: Any = compiled_fused_step,
            x: mx.array = inputs,
            target: mx.array = targets,
            model: KimiDeltaAttention = fused,
            optimizer: Any = fused_optimizer,
        ) -> tuple[Any, ...]:
            loss = compiled_step(x, target)
            return loss, model.state, optimizer.state

        # Trigger graph/Metal compilation and optimizer-state creation before
        # entering the timed region.
        mx.eval(call_reference(), call_fused())
        local_iterations = min(iterations, 2) if length >= 1_024 else iterations
        reference_ms = _benchmark_call(
            call_reference,
            warmup=warmup,
            iterations=local_iterations,
            repeats=3,
        )
        fused_ms = _benchmark_call(
            call_fused,
            warmup=warmup,
            iterations=local_iterations,
            repeats=3,
        )
        row = KDALayerTrainingBenchmark(
            sequence_length=length,
            batch_size=batch_size,
            autograd_reference_step_ms=reference_ms,
            fused_metal_step_ms=fused_ms,
            full_layer_training_speedup=reference_ms / fused_ms,
        )
        rows.append(asdict(row))

    return rows


def platform_report() -> dict[str, str]:
    """Return the MLX/Metal facts used by the notebook's feasibility cell."""

    return {
        "machine": platform.machine(),
        "mlx_version": getattr(mx, "__version__", "installed"),
        "metal_available": str(mx.metal.is_available()),
        "default_device": str(mx.default_device()),
        "training_path": "MLX checkpointed fused Metal custom VJP",
        "inference_path": "MLX fused custom Metal kernel",
    }


def parameter_count(model: nn.Module) -> int:
    return sum(array.size for _, array in tree_flatten(model.parameters()))


def masked_addition_loss(
    model: KDAAdditionModel,
    inputs: mx.array,
    targets: mx.array,
) -> mx.array:
    """Cross-entropy over answer tokens; ``-1`` targets are ignored."""

    logits = model(inputs)
    mask = targets != -1
    safe_targets = mx.where(mask, targets, 0)
    losses = nn.losses.cross_entropy(logits, safe_targets, reduction="none")
    return mx.sum(mx.where(mask, losses, 0)) / mx.sum(mask)


def expected_speedup_table() -> list[dict[str, str]]:
    """Published Kimi Linear targets, kept separate from local microbenchmarks."""

    return [
        {
            "context": "4K",
            "published_speedup": "about 1x",
            "scope": "full-model speed on MMLU-Pro",
        },
        {
            "context": "128K",
            "published_speedup": "3.98x",
            "scope": "full-model speed on RULER",
        },
        {
            "context": "1M",
            "published_speedup": "6.3x",
            "scope": "time per output token versus MLA",
        },
    ]
