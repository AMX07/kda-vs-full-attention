"""A small, reusable decoder-only Transformer.

This is the core model extracted from ``notebooks/gpt_hw-2.ipynb`` and
refactored to avoid notebook globals.  It deliberately excludes datasets,
tokenizers, optimizer setup, logging, and training loops so that it is a good
starting point for attention experiments.

Default attention is full causal softmax attention.  To experiment with a new
attention architecture, replace ``CausalSelfAttention`` in ``TransformerBlock``
while keeping the model, loss, and generation interfaces unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig:
    """Architecture settings for :class:`DecoderOnlyTransformer`."""

    vocab_size: int
    max_seq_len: int
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.max_seq_len <= 0:
            raise ValueError("vocab_size and max_seq_len must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")


class CausalSelfAttention(nn.Module):
    """Parallel multi-head, full-context causal softmax attention."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # The notebook used three separate projections. Keeping them separate
        # makes Q, K, and V easy to inspect or change in later experiments.
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.output_projection = nn.Linear(config.n_embd, config.n_embd)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        # ``True`` means a query position may attend to that key position.
        causal_mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal attention to ``x`` with shape ``(batch, time, embed)``."""
        batch_size, time, embed_dim = x.shape
        if time > self.causal_mask.size(0):
            raise ValueError(
                f"sequence length {time} exceeds max_seq_len {self.causal_mask.size(0)}"
            )

        # (B, T, C) -> (B, H, T, D), where C = H * D.
        q = self.query(x).view(batch_size, time, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, time, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, time, self.n_head, self.head_dim).transpose(1, 2)

        # One score for every query/key pair: (B, H, T, T).
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores.masked_fill(~self.causal_mask[:time, :time], float("-inf"))
        attention = F.softmax(scores, dim=-1)
        attention = self.attention_dropout(attention)

        # Weighted sum of values: (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D).
        output = attention @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, time, embed_dim)
        return self.residual_dropout(self.output_projection(output))


class FeedForward(nn.Module):
    """Token-wise computation part of a Transformer block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),  # Faithful to the notebook; GELU is a common upgrade.
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )   

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm residual block: attention (communication), then MLP (compute)."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only Transformer with next-token cross-entropy loss."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.n_embd)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(
        self, token_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Return ``(logits, loss)``; loss is ``None`` when targets are omitted."""
        batch_size, time = token_ids.shape
        if time > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {time} exceeds max_seq_len {self.config.max_seq_len}"
            )

        positions = torch.arange(time, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * time, self.config.vocab_size),
                targets.reshape(batch_size * time),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Greedily append tokens, cropping context to the configured length."""
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids
