"""Download verified raw inputs for DEE-6 without preparing training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Source:
    filename: str
    url: str
    size: int
    sha256: str


AMAZON_REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
AMAZON_BASE = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/"
    + AMAZON_REVISION
    + "/"
)
MOVIELENS = Source(
    "ml-1m.zip",
    "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    5_917_549,
    "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
)
AMAZON_SOURCES = {
    "amazon_video_games": (
        Source(
            "Video_Games.csv",
            AMAZON_BASE + "benchmark/5core/rating_only/Video_Games.csv",
            47_244_051,
            "2fb8fd6fb8b93a1a555e1aa4d46e3327e2bf345e9fb02a3bea184a9bce2ba4fa",
        ),
        Source(
            "meta_Video_Games.jsonl",
            AMAZON_BASE + "raw/meta_categories/meta_Video_Games.jsonl",
            437_375_785,
            "3ad94dfaa8dfed23c5f420aa9ddec415d00f2b8ec9f2d796f487b4b2c4d256de",
        ),
    ),
    "amazon_grocery": (
        Source(
            "Grocery_and_Gourmet_Food.csv",
            AMAZON_BASE + "benchmark/5core/rating_only/Grocery_and_Gourmet_Food.csv",
            229_036_007,
            "193ad9f17377666565dae8f5500e1355ca5182534557798e0bf024aa2b991595",
        ),
        Source(
            "meta_Grocery_and_Gourmet_Food.jsonl",
            AMAZON_BASE + "raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl",
            1_382_105_675,
            "ed9b24ea707f7010b5cb321462a8e896ad1fcd59620fbc55f05084c3da59842e",
        ),
    ),
}
MOVIELENS_MEMBERS = ("ml-1m/ratings.dat", "ml-1m/movies.dat", "ml-1m/README")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(source: Source, directory: Path) -> Path:
    """Verify existing files or download to a temporary file before publishing."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source.filename
    if target.exists():
        if (
            target.is_file()
            and target.stat().st_size == source.size
            and _file_sha256(target) == source.sha256
        ):
            print(f"Already verified: {target}", flush=True)
            return target
        raise FileExistsError(f"Existing file does not match the expected source: {target}")

    print(f"Downloading {source.filename} ({source.size / 1_000_000:.1f} MB)...", flush=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{source.filename}.", suffix=".part", dir=directory
    )
    temporary = Path(name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            request = Request(source.url, headers={"User-Agent": "NeedleTTT-dataset-setup/1.0"})
            with urlopen(request, timeout=120) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
        if size != source.size or digest.hexdigest() != source.sha256:
            raise ValueError(f"Download failed size or SHA-256 verification: {source.filename}")
        if target.exists():
            if (
                target.is_file()
                and target.stat().st_size == source.size
                and _file_sha256(target) == source.sha256
            ):
                return target
            raise FileExistsError(f"Another file appeared during download: {target}")
        os.replace(temporary, target)
        print(f"Verified: {target}", flush=True)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def _extract_movielens(archive_path: Path, destination: Path) -> None:
    """Extract only named files and verify existing files before reusing them."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in MOVIELENS_MEMBERS:
            info = archive.getinfo(member)
            target = destination / Path(member).name
            if target.exists():
                crc = 0
                if not target.is_file():
                    raise FileExistsError(f"Expected a file: {target}")
                with target.open("rb") as existing:
                    for block in iter(lambda: existing.read(1024 * 1024), b""):
                        crc = zlib.crc32(block, crc)
                if target.stat().st_size != info.file_size or crc != info.CRC:
                    raise FileExistsError(f"Existing MovieLens file differs: {target}")
                continue
            descriptor, name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".part", dir=destination
            )
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, "wb") as output, archive.open(info) as source:
                    shutil.copyfileobj(source, output)
                if target.exists():
                    raise FileExistsError(f"Another file appeared during extraction: {target}")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)


def prepare_raw_inputs(
    data_root: Path, local_root: Path, datasets: tuple[str, ...]
) -> dict[str, str]:
    """Put Amazon source files in Drive and MovieLens in a private local runtime."""
    # Preserve the user-facing shortcut path in output; resolve separately for
    # containment checks so a shortcut cannot hide a shared Drive location.
    data_root = Path(os.path.abspath(data_root.expanduser()))
    local_root = Path(os.path.abspath(local_root.expanduser()))
    if not (data_root / "raw").is_dir() or not (data_root / "processed").is_dir():
        raise FileNotFoundError("Expected existing shared data/raw and data/processed folders")
    resolved_local = local_root.resolve()
    if resolved_local.is_relative_to(data_root.resolve().parent) or resolved_local.is_relative_to(
        Path("/content/drive")
    ):
        raise ValueError("MovieLens must use a local runtime folder outside the shared Drive")
    unknown = set(datasets) - {"movielens_1m", *AMAZON_SOURCES}
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(sorted(unknown))}")

    result: dict[str, str] = {}
    for dataset in datasets:
        if dataset == "movielens_1m":
            destination = local_root / dataset
            archive = _download(MOVIELENS, destination)
            _extract_movielens(archive, destination)
        else:
            destination = data_root / "raw" / dataset
            for source in AMAZON_SOURCES[dataset]:
                _download(source, destination)
        result[dataset] = str(destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, required=True, help="Existing shared NeedleTTT/data"
    )
    parser.add_argument(
        "--local-root", type=Path, required=True, help="Temporary local runtime directory"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("movielens_1m", *AMAZON_SOURCES),
        default=["movielens_1m", *AMAZON_SOURCES],
    )
    args = parser.parse_args()
    locations = prepare_raw_inputs(args.data_root, args.local_root, tuple(args.datasets))
    print("\nVerified raw inputs:")
    print(json.dumps(locations, indent=2))
    print("No train/validation/test splits or processed datasets were created.")


if __name__ == "__main__":
    main()
