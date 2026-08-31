import numpy as np
import pytest

pytest.importorskip("mlx.core")

from karpathygpt_hw.long_addition import (  # noqa: E402
    TOKENS,
    TrainConfig,
    decode_tokens,
    make_addition_batch,
    parse_schedule,
    sequence_length,
)


def _complete_rows(inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.concatenate([inputs, targets[:, -1:]], axis=1)


def test_1364_digit_format_fits_4096_characters() -> None:
    assert sequence_length(1_364) == 4_095
    assert sequence_length(1_364) + 1 == 4_096


def test_generated_rows_encode_correct_addition() -> None:
    rng = np.random.default_rng(4_242)
    batch = make_addition_batch(
        rng,
        batch_size=32,
        digits=8,
        carry_heavy_fraction=0.5,
    )
    rows = _complete_rows(batch.inputs, batch.targets)

    for row in rows:
        text = decode_tokens(row)
        problem, reversed_answer = text.rstrip().split("=")
        a, b = problem.split("+")
        answer = reversed_answer[::-1]
        assert len(a) == len(b) == 8
        assert len(answer) == 9
        assert int(a) + int(b) == int(answer)

    assert np.all(batch.targets[:, :17] == -1)
    assert batch.targets.shape == (32, sequence_length(8))
    assert len(TOKENS) == 13


def test_carry_heavy_rows_carry_through_every_column() -> None:
    rng = np.random.default_rng(4_243)
    batch = make_addition_batch(
        rng,
        batch_size=16,
        digits=12,
        carry_heavy_fraction=1.0,
    )
    rows = _complete_rows(batch.inputs, batch.targets)
    for row in rows:
        text = decode_tokens(row)
        problem, reversed_answer = text.rstrip().split("=")
        a, b = problem.split("+")
        answer = reversed_answer[::-1]
        carry = 0
        for a_digit, b_digit in zip(reversed(a), reversed(b), strict=True):
            total = int(a_digit) + int(b_digit) + carry
            assert total >= 10
            carry = total // 10
        assert answer[0] == "1"


def test_active_digits_are_left_zero_padded() -> None:
    rng = np.random.default_rng(4_244)
    batch = make_addition_batch(
        rng,
        batch_size=8,
        digits=16,
        active_digits=3,
        carry_heavy_fraction=0.0,
    )
    rows = _complete_rows(batch.inputs, batch.targets)
    for row in rows:
        text = decode_tokens(row)
        problem = text.split("=")[0]
        a, b = problem.split("+")
        assert a[:13] == b[:13] == "0" * 13


def test_schedule_parser() -> None:
    stages = parse_schedule("3:20,8:40,1364:2")
    assert [(stage.digits, stage.steps) for stage in stages] == [
        (3, 20),
        (8, 40),
        (1_364, 2),
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_schedule("8:10,3:10")
    with pytest.raises(ValueError, match="at most 1,364"):
        parse_schedule("1365:1")


def test_layout_rehearsal_mode_is_configurable(tmp_path) -> None:
    config = TrainConfig(
        schedule=parse_schedule("3:20,8:20"),
        output_dir=tmp_path,
        rehearsal_mode="layout",
    )
    assert config.rehearsal_mode == "layout"

    uniform_config = TrainConfig(
        schedule=parse_schedule("16:20"),
        output_dir=tmp_path,
        rehearsal_mode="uniform_layout",
    )
    assert uniform_config.rehearsal_mode == "uniform_layout"
