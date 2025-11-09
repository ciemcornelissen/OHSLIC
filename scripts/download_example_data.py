#!/usr/bin/env python3
"""Fetch the public OHSLIC sample capture and expand it into the data directory."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_URL = "https://cloud.ilabt.imec.be/index.php/s/a8JCqB6jQqeG95m/download"
DEFAULT_DEST = Path("data")
ARCHIVE_NAME = "ohslic_sample.zip"
FOLDER_NAME = "OHSLIC_hyperspectral_example"
CHUNK_SIZE = 1024 * 1024


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    temp_path = destination

    with temp_path.open("wb") as handle:
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=temp_path.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))

    return temp_path


def extract(archive_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(target_dir)

    extracted_root = target_dir / FOLDER_NAME
    if extracted_root.exists():
        for item in extracted_root.iterdir():
            destination = target_dir / item.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), destination)
        shutil.rmtree(extracted_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Source URL for the zipped sample capture",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Directory to unpack the sample into",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Skip deleting the downloaded archive",
    )
    args = parser.parse_args()

    target_dir = args.dest.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    archive_path = target_dir / ARCHIVE_NAME
    download(args.url, archive_path)
    extract(archive_path, target_dir)

    if not args.keep_archive:
        archive_path.unlink(missing_ok=True)

    print(f"Sample data available in {target_dir}")


if __name__ == "__main__":
    main()
