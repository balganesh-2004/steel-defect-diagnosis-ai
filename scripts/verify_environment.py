"""Verify the Phase 1 Python environment for the steel defect project."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))


REQUIRED_PACKAGES = {
    "torch": "PyTorch",
    "torchvision": "torchvision",
    "cv2": "OpenCV",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "sklearn": "scikit-learn",
    "matplotlib": "Matplotlib",
    "PIL": "Pillow",
}


def package_version(module_name: str) -> str:
    """Return an installed package version or a clear missing message."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return "NOT INSTALLED"

    return getattr(module, "__version__", "installed, version unavailable")


def main() -> None:
    print("Steel Defect Diagnosis - Environment Verification")
    print("=" * 56)
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print()

    for module_name, display_name in REQUIRED_PACKAGES.items():
        print(f"{display_name}: {package_version(module_name)}")

    try:
        import torch
    except ImportError:
        print("\nCUDA: unavailable because PyTorch is not installed")
        return

    print()
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version reported by PyTorch: {torch.version.cuda}")
        print(f"GPU device count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
