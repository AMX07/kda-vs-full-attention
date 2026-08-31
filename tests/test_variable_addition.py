from __future__ import annotations

import numpy as np
import torch

from karpathygpt_hw.variable_addition import (
    IGNORE_INDEX,
    AdditionExample,
    AdditionLayout,
    AdditionTransformer,
    _sample_full_carry_pair,
    _sample_layout_pair,
    carry_chain_length,
    decode,
    example_layout,
    format_row,
    make_batch,
    masked_loss,
    pair_split,
    valid_layouts,
)


def test_format_is_unpadded_and_reverses_only_the_answer() -> None:
    assert format_row(AdditionExample(7, 23)) == "7+23=03 "
    assert format_row(AdditionExample(12, 3)) == "12+3=51 "
    assert format_row(AdditionExample(0, 0)) == "0+0=0 "


def test_normal_answer_format_is_unpadded_and_not_reversed() -> None:
    assert format_row(AdditionExample(7, 23), "normal") == "7+23=30 "
    assert format_row(AdditionExample(12, 3), "normal") == "12+3=15 "
    assert format_row(AdditionExample(0, 0), "normal") == "0+0=0 "


def test_batch_has_no_padding_and_masks_only_the_prompt() -> None:
    examples = (AdditionExample(7, 23), AdditionExample(8, 34))
    batch = make_batch(examples, device=torch.device("cpu"))
    assert batch.layout == AdditionLayout(1, 2, 2)
    assert batch.inputs.shape == (2, batch.layout.model_input_length)
    assert decode(batch.inputs[0].tolist()) == "7+23=03"
    assert torch.all(batch.targets[:, : batch.layout.prompt_length - 1] == IGNORE_INDEX)
    assert torch.all(batch.targets[:, batch.layout.prompt_length - 1 :] != IGNORE_INDEX)


def test_normal_answer_batch_has_no_padding_and_masks_only_the_prompt() -> None:
    examples = (AdditionExample(7, 23), AdditionExample(8, 34))
    batch = make_batch(examples, device=torch.device("cpu"), answer_order="normal")
    assert batch.layout == AdditionLayout(1, 2, 2)
    assert decode(batch.inputs[0].tolist()) == "7+23=30"
    assert torch.all(batch.targets[:, : batch.layout.prompt_length - 1] == IGNORE_INDEX)
    assert torch.all(batch.targets[:, batch.layout.prompt_length - 1 :] != IGNORE_INDEX)


def test_layout_sampler_honors_lengths_and_split() -> None:
    rng = np.random.default_rng(123)
    layout = AdditionLayout(2, 3, 4)
    examples = [_sample_layout_pair(rng, layout, split="test", split_seed=4242) for _ in range(100)]
    assert all(example_layout(example) == layout for example in examples)
    assert all(pair_split(example.a, example.b, 4242) == "test" for example in examples)


def test_full_carry_sampler_produces_worst_case_chain() -> None:
    rng = np.random.default_rng(123)
    for digits in range(1, 6):
        example = _sample_full_carry_pair(
            rng,
            digits,
            split="train",
            split_seed=4242,
        )
        assert example_layout(example) == AdditionLayout(digits, digits, digits + 1)
        assert carry_chain_length(example.a, example.b) == digits


def test_valid_layouts_are_genuinely_variable() -> None:
    layouts = valid_layouts(3)
    assert AdditionLayout(1, 1, 1) in layouts
    assert AdditionLayout(1, 1, 2) in layouts
    assert AdditionLayout(3, 3, 4) in layouts
    assert len({layout.model_input_length for layout in layouts}) > 1


def test_original_architecture_forward_and_loss() -> None:
    model = AdditionTransformer(block_size=12, dims=64, num_heads=4, num_layers=4)
    batch = make_batch(
        (AdditionExample(123, 456), AdditionExample(321, 678)),
        device=torch.device("cpu"),
    )
    logits = model(batch.inputs)
    assert logits.shape == (*batch.inputs.shape, 13)
    loss = masked_loss(logits, batch.targets)
    loss.backward()
    assert torch.isfinite(loss)


def test_dense_attention_is_causal() -> None:
    torch.manual_seed(123)
    model = AdditionTransformer(block_size=12, dims=64, num_heads=4, num_layers=4)
    model.eval()
    first = torch.randint(0, 13, (1, 12))
    second = first.clone()
    second[:, 6:] = (second[:, 6:] + 1) % 13
    logits = model(torch.cat((first, second), dim=0))
    torch.testing.assert_close(logits[0, :6], logits[1, :6])
