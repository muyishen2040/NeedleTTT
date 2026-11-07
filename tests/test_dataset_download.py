import hashlib
import zipfile
from pathlib import Path

import pytest

from needlettt.data import download


def source(path: Path) -> download.Source:
    data = path.read_bytes()
    return download.Source(path.name, path.as_uri(), len(data), hashlib.sha256(data).hexdigest())


def test_prepare_places_sources_in_separate_roots_and_can_be_repeated(tmp_path, monkeypatch):
    data_root = tmp_path / "shared" / "data"
    (data_root / "raw").mkdir(parents=True)
    (data_root / "processed").mkdir()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    archive = upstream / "ml-1m.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("ml-1m/ratings.dat", "1::2::5::10\n")
        z.writestr("ml-1m/movies.dat", "2::Example::Drama\n")
        z.writestr("ml-1m/README", "Example license\n")
        z.writestr("../escape.txt", "not extracted")
    video = upstream / "Video_Games.csv"
    video.write_text("user_id,parent_asin,rating,timestamp\nu,a,5,1\n", encoding="utf-8")
    grocery = upstream / "Grocery_and_Gourmet_Food.csv"
    grocery.write_text("user_id,parent_asin,rating,timestamp\nu,b,4,2\n", encoding="utf-8")
    metadata = upstream / "metadata.jsonl"
    metadata.write_text('{"parent_asin":"a"}\n', encoding="utf-8")
    monkeypatch.setattr(download, "MOVIELENS", source(archive))
    monkeypatch.setattr(
        download,
        "AMAZON_SOURCES",
        {
            "amazon_video_games": (source(video), source(metadata)),
            "amazon_grocery": (source(grocery), source(metadata)),
        },
    )
    local = tmp_path / "runtime"
    datasets = ("movielens_1m", "amazon_video_games", "amazon_grocery")

    paths = download.prepare_raw_inputs(data_root, local, datasets)
    assert paths == download.prepare_raw_inputs(data_root, local, datasets)
    assert (local / "movielens_1m" / "ratings.dat").read_text() == "1::2::5::10\n"
    assert (
        data_root / "raw" / "amazon_video_games" / video.name
    ).read_bytes() == video.read_bytes()
    assert (
        data_root / "raw" / "amazon_grocery" / grocery.name
    ).read_bytes() == grocery.read_bytes()
    assert not (data_root / "raw" / "movielens_1m").exists()
    assert not (tmp_path / "escape.txt").exists()
    assert list((data_root / "processed").iterdir()) == []


def test_corrupt_existing_and_incomplete_downloads_do_not_overwrite(tmp_path):
    upstream = tmp_path / "source.csv"
    upstream.write_bytes(b"correct content")
    item = source(upstream)
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    existing = target_dir / item.filename
    existing.write_bytes(b"preserve this")
    with pytest.raises(FileExistsError):
        download._download(item, target_dir)
    assert existing.read_bytes() == b"preserve this"
    existing.unlink()
    wrong_hash = download.Source(item.filename, item.url, item.size, "0" * 64)
    with pytest.raises(ValueError, match="verification"):
        download._download(wrong_hash, target_dir)
    assert not existing.exists()
    assert list(target_dir.iterdir()) == []


def test_rejects_missing_shared_folders_and_movielens_inside_shared_drive(tmp_path):
    data = tmp_path / "NeedleTTT" / "data"
    with pytest.raises(FileNotFoundError):
        download.prepare_raw_inputs(data, tmp_path / "runtime", ("movielens_1m",))
    (data / "raw").mkdir(parents=True)
    (data / "processed").mkdir()
    with pytest.raises(ValueError, match="outside the shared Drive"):
        download.prepare_raw_inputs(data, data / "raw", ("movielens_1m",))
