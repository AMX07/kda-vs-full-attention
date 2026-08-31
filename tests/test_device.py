from karpathygpt_hw.device import preferred_device


def test_preferred_device_is_known_value() -> None:
    assert preferred_device() in {"cuda", "mps", "cpu"}
