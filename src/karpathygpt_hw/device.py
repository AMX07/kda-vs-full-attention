from __future__ import annotations

import platform
import sys


def preferred_device() -> str:
    """Return the best available PyTorch device for this machine."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_report() -> dict[str, str]:
    import torch

    device = preferred_device()
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": device,
        "cuda_available": str(torch.cuda.is_available()),
        "mps_available": str(torch.backends.mps.is_available()),
        "mps_built": str(torch.backends.mps.is_built()),
    }
    if torch.cuda.is_available():
        report["cuda_device"] = torch.cuda.get_device_name(0)
    return report


def main() -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Karpathy GPT HW Environment")
    table.add_column("Key")
    table.add_column("Value")

    for key, value in device_report().items():
        table.add_row(key, value)

    Console().print(table)


if __name__ == "__main__":
    main()
