"""Check that the class Needle checkout is installed at the canonical LOCAL path.

Run from anywhere:
    python scripts/check_needle.py

Checks imports only. Does not compile course extensions or test TTT gradients.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    checkout = repo / "third_party" / "needle"
    package_parent = checkout / "python"
    package_dir = package_parent / "needle"

    if not checkout.is_dir():
        print("Needle checkout not found.")
        print(f"Copy your entire course source folder to: {checkout}")
        print("Instructions: docs/needle_setup.md")
        return 1

    if not package_dir.is_dir():
        print(f"Found checkout at: {checkout}")
        print(f"Cannot find typical course package: {package_dir}")
        print("Your course version may have another layout. Inspect it and update the")
        print("import path following docs/needle_setup.md; do not move files blindly.")
        return 1

    sys.path.insert(0, str(package_parent))
    try:
        module = importlib.import_module("needle")
    except Exception as exc:
        print(f"Needle import failed: {type(exc).__name__}: {exc}")
        print("The checkout exists, but the course may require a compiled extension.")
        print("Follow its own build instructions; see docs/needle_setup.md.")
        return 1

    location = getattr(module, "__file__", None)
    if not location:
        print("Needle imported but has no identifiable source path; inspect manually.")
        return 1
    resolved = Path(location).resolve()
    if not resolved.is_relative_to(checkout.resolve()):
        print(f"Needle imported from an unexpected location: {resolved}")
        print(f"Expected it under: {checkout}")
        return 1

    print("Needle class checkout found and imported successfully.")
    print(f"Checkout: {checkout}")
    print(f"Imported module: {resolved}")
    print("Next: implement and test the project operations in src/needlettt/backend/needle_ops.py")
    print("Note: this does not verify a CUDA build or the training pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
