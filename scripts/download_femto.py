#!/usr/bin/env python3
"""Download and safely extract the NASA FEMTO/PRONOSTIA dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
ARCHIVE_URL = "https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip"
BUFFER_SIZE = 1024 * 1024
DISK_MARGIN = 128 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_archive(url: str, destination: Path) -> tuple[int, str, str | None]:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        expected_size_header = response.headers.get("Content-Length")
        if expected_size_header is None:
            raise RuntimeError("dataset server did not report the archive size")
        expected_size = int(expected_size_header)
        etag = response.headers.get("ETag")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_size:
        return expected_size, _sha256(destination), etag

    partial_path = destination.with_suffix(destination.suffix + ".part")
    existing_size = partial_path.stat().st_size if partial_path.exists() else 0
    if existing_size > expected_size:
        raise RuntimeError(f"partial download is larger than expected: {partial_path}")
    free_bytes = shutil.disk_usage(destination.parent).free
    if free_bytes < expected_size - existing_size + expected_size + DISK_MARGIN:
        raise OSError(
            "insufficient free disk space for the download and nested archive; "
            f"at least {(expected_size - existing_size + expected_size + DISK_MARGIN) / (1024**3):.1f} GiB "
            "is needed at this stage"
        )

    digest = hashlib.sha256()
    if existing_size:
        with partial_path.open("rb") as existing:
            for block in iter(lambda: existing.read(BUFFER_SIZE), b""):
                digest.update(block)

    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"dataset download failed: HTTP {error.code}") from error

    with response:
        if response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            expected_prefix = f"bytes {existing_size}-"
            if not content_range.startswith(expected_prefix):
                raise RuntimeError(f"unexpected range response from dataset server: {content_range!r}")
            write_mode = "ab"
        elif response.status == 200:
            existing_size = 0
            digest = hashlib.sha256()
            write_mode = "wb"
        else:
            raise RuntimeError(f"unexpected HTTP status while downloading dataset: {response.status}")

        copied = existing_size
        with partial_path.open(write_mode) as output:
            while block := response.read(BUFFER_SIZE):
                output.write(block)
                digest.update(block)
                copied += len(block)
                if copied % (64 * 1024 * 1024) < BUFFER_SIZE:
                    print(f"Downloaded {copied / (1024**3):.2f} / {expected_size / (1024**3):.2f} GiB")

    if copied != expected_size:
        raise RuntimeError(
            f"incomplete dataset download ({copied} of {expected_size} bytes); "
            f"rerun the command to resume from {partial_path}"
        )
    os.replace(partial_path, destination)
    return expected_size, digest.hexdigest(), etag


def safe_extract_zip(archive_path: Path, destination: Path) -> int:
    """Extract a ZIP while rejecting traversal, links, and insufficient disk space."""
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        uncompressed_size = sum(member.file_size for member in members)
        free_bytes = shutil.disk_usage(destination).free
        if free_bytes < uncompressed_size + DISK_MARGIN:
            raise OSError(
                f"insufficient free disk space to extract {archive_path.name}; "
                f"need at least {(uncompressed_size + DISK_MARGIN) / (1024**3):.1f} GiB"
            )

        for member in members:
            member_path = PurePosixPath(member.filename)
            if (
                member_path.is_absolute()
                or "\\" in member.filename
                or PureWindowsPath(member.filename).drive
                or ".." in member_path.parts
            ):
                raise ValueError(f"unsafe path in dataset archive: {member.filename!r}")
            target = (destination / Path(*member_path.parts)).resolve()
            if not target.is_relative_to(resolved_destination):
                raise ValueError(f"archive entry escapes extraction directory: {member.filename!r}")
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise ValueError(f"symbolic links are not allowed in dataset archives: {member.filename!r}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=BUFFER_SIZE)
    return uncompressed_size


def _extract_nested_archives(destination: Path) -> None:
    for _ in range(3):
        nested_archives = sorted(destination.rglob("*.zip"))
        if not nested_archives:
            return
        for nested_archive in nested_archives:
            nested_destination = nested_archive.parent / nested_archive.stem
            if nested_destination.exists():
                raise FileExistsError(f"nested archive extraction target already exists: {nested_destination}")
            safe_extract_zip(nested_archive, nested_destination)
            nested_archive.unlink()
    if any(destination.rglob("*.zip")):
        raise ValueError("dataset contains more nested ZIP levels than supported")


def _publish_dataset(staging_dir: Path, dataset_dir: Path) -> None:
    try:
        os.replace(staging_dir, dataset_dir)
    except OSError as error:
        if error.errno != errno.EPERM:
            raise
        try:
            shutil.copytree(staging_dir, dataset_dir)
        except OSError:
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            raise


def _copy_inner_archive(outer_archive: Path, inner_archive: Path) -> None:
    with zipfile.ZipFile(outer_archive) as archive:
        candidates = [
            member
            for member in archive.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".zip")
        ]
        if len(candidates) != 1:
            raise ValueError(
                "expected one nested ZIP in the NASA download; "
                f"found {len(candidates)}"
            )
        member = candidates[0]
        free_bytes = shutil.disk_usage(inner_archive.parent).free
        if free_bytes < member.file_size + DISK_MARGIN:
            raise OSError(
                "insufficient free disk space for the nested dataset archive; "
                f"need at least {(member.file_size + DISK_MARGIN) / (1024**3):.1f} GiB"
            )
        with archive.open(member) as source, inner_archive.open("wb") as output:
            shutil.copyfileobj(source, output, length=BUFFER_SIZE)
        if inner_archive.stat().st_size != member.file_size:
            raise RuntimeError("nested dataset archive has an unexpected size")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="root directory for downloaded archives and extracted raw data",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="keep the two downloaded ZIP files after successful extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    download_dir = data_root / "downloads"
    raw_dir = data_root / "raw"
    dataset_dir = raw_dir / "femto"
    outer_archive = download_dir / "femto_nasa_outer.zip"
    inner_archive = download_dir / "FEMTOBearingDataSet.zip"
    if dataset_dir.exists():
        if any(dataset_dir.rglob("acc_*.csv")):
            print(f"Dataset already extracted: {dataset_dir}")
            return
        raise FileExistsError(
            f"dataset destination already exists but has no acceleration samples: {dataset_dir}"
        )

    download_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official NASA FEMTO/PRONOSTIA archive: {ARCHIVE_URL}")
    archive_size, archive_sha256, etag = _download_archive(ARCHIVE_URL, outer_archive)
    print(f"Downloaded {archive_size / (1024**3):.2f} GiB; SHA-256: {archive_sha256}")

    _copy_inner_archive(outer_archive, inner_archive)
    with tempfile.TemporaryDirectory(prefix=".femto-staging-", dir=raw_dir) as temporary_dir:
        staging_dir = Path(temporary_dir) / "femto"
        safe_extract_zip(inner_archive, staging_dir)
        _extract_nested_archives(staging_dir)
        if not any(staging_dir.rglob("acc_*.csv")):
            raise ValueError(
                "archive extracted successfully but contains no acc_*.csv samples; "
                "inspect the dataset layout before training"
            )
        (staging_dir / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "dataset": "NASA FEMTO/PRONOSTIA",
                    "source_url": ARCHIVE_URL,
                    "source_etag": etag,
                    "downloaded_archive_bytes": archive_size,
                    "downloaded_archive_sha256": archive_sha256,
                    "downloaded_at_utc": datetime.now(UTC).isoformat(),
                    "citation": (
                        "FEMTO Bearing Data Set, NASA Prognostics Data Repository, NASA Ames "
                        "Research Center; PRONOSTIA experimental platform citation is provided "
                        "in the project README."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _publish_dataset(staging_dir, dataset_dir)

    if not args.keep_archive:
        outer_archive.unlink(missing_ok=True)
        inner_archive.unlink(missing_ok=True)
    print(f"Extracted dataset: {dataset_dir}")


if __name__ == "__main__":
    main()
