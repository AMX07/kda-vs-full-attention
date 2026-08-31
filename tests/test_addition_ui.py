import pytest

pytest.importorskip("mlx.core")

from karpathygpt_hw.addition_ui import (  # noqa: E402
    ExpressionError,
    parse_addition_expression,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("12 + 34", ("12", "34")),
        ("(10^16 - 1) + (10^16 - 1)", ("9" * 16, "9" * 16)),
        ("(2 * 10^3) + 7", ("2000", "7")),
    ],
)
def test_parse_addition_expression(expression: str, expected: tuple[str, str]) -> None:
    assert parse_addition_expression(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "10^1365 + 1",
        "(10^1364) + 1",
        "-1 + 2",
        "1 / 2 + 3",
        "__import__('os') + 1",
        "1 * 2",
    ],
)
def test_parse_addition_expression_rejects_unsafe_or_out_of_range_input(
    expression: str,
) -> None:
    with pytest.raises(ExpressionError):
        parse_addition_expression(expression)
