"""The diagnostic must work without including restricted course Needle code in CI."""

from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/check_needle.py"


def run_check(checkout: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--checkout", str(checkout)],
        check=False, capture_output=True, text=True,
    )


def test_needle_setup_missing_checkout_is_actionable(tmp_path: Path):
    result = run_check(tmp_path / "not-present")
    assert result.returncode == 1
    assert "Copy your entire course source folder" in result.stdout
    assert "docs/needle_setup.md" in result.stdout


def test_needle_setup_wrong_directory_layout_is_actionable(tmp_path: Path):
    checkout = tmp_path / "needle"
    checkout.mkdir()
    (checkout / "python").mkdir()
    result = run_check(checkout)
    assert result.returncode == 1
    assert "Cannot find typical course package" in result.stdout


def test_needle_setup_accepts_a_synthetic_local_package(tmp_path: Path):
    checkout = tmp_path / "needle"
    package = checkout / "python" / "needle"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Synthetic import-only test, not course source."""')
    result = run_check(checkout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "imported successfully" in result.stdout
    assert str(package.resolve()) in result.stdout
