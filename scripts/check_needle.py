"""Verify a locally copied class Needle checkout without redistributing it.

    python scripts/check_needle.py

Optional --checkout PATH exists for diagnostics and CI tests; teammates should
use the canonical root/third_party/needle checkout location.
This checks imports only, not native extensions, gradients or model training.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def check_checkout(checkout: Path) -> int:
    checkout = checkout.resolve()
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
        print("Your course version may use a different layout. Inspect the original")
        print("package structure; don't move course files just to satisfy this check.")
        return 1

    sys.path.insert(0, str(package_parent))
    try:
        module = importlib.import_module("needle")
    except Exception as exc:
        print(f"Needle import failed: {type(exc).__name__}: {exc}")
        print("The course may require compilation or additional dependencies.")
        print("Follow its original setup instructions; see docs/needle_setup.md.")
        return 1
    finally:
        sys.path.remove(str(package_parent))

    location = getattr(module, "__file__", None)
    if not location:
        print("Needle imported but has no identifiable source path; inspect manually.")
        return 1
    resolved = Path(location).resolve()
    if not resolved.is_relative_to(checkout):
        print(f"Needle imported from an unexpected location: {resolved}")
        print(f"Expected it under: {checkout}")
        return 1

    print("Needle class checkout found and imported successfully.")
    print(f"Checkout: {checkout}")
    print(f"Imported module: {resolved}")
    print("Next: implement and test project operators in src/needlettt/backend/needle_ops.py")
    print("Import success alone does NOT verify a CUDA build or Needle gradients.")
    return 0


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Check local course Needle source import")
    parser.add_argument(
        "--checkout",
        type=Path,
        default=repo / "third_party" / "needle",
        help="Diagnostic override; normal checkout is third_party/needle/",
    )
    args = parser.parse_args()
    return check_checkout(args.checkout)


if __name__ == "__main__":
    raise SystemExit(main())
