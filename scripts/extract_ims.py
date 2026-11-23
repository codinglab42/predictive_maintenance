#!/usr/bin/env python3
"""Extract the NASA IMS ZIP -> 7z -> RAR archive into the raw data directory."""

from __future__ import annotations

import argparse
import errno
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import sys
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import libarchive
import py7zr

SHARED_ROOT = Path("/mnt/shared/predictive-maintenance")
DEFAULT_ARCHIVE = SHARED_ROOT / "downloads" / "ims_bearings.zip"
DEFAULT_OUTPUT = SHARED_ROOT / "raw" / "ims"
EXPECTED_SERIES = {"1st_test", "2nd_test", "3rd_test"}


def _safe_member_path(root: Path, member_name: str) -> Path:
    posix_path = PurePosixPath(member_name)
    windows_path = PureWindowsPath(member_name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError(f"unsafe path in IMS archive: {member_name!r}")
    root_resolved = root.resolve()
    destination = (root / Path(*posix_path.parts)).resolve()
    if not destination.is_relative_to(root_resolved):
        raise ValueError(f"unsafe path in IMS archive: {member_name!r}")
    return destination


def _normalized_member_name(series_name: str, member_name: str) -> str:
    parts = PurePosixPath(member_name).parts
    expected_archive_root = "4th_test" if series_name == "3rd_test" else series_name
    if not parts or parts[0] != expected_archive_root:
        raise ValueError(
            f"unexpected {series_name} path in IMS RAR archive: {member_name!r}"
        )
    return str(PurePosixPath(series_name, *parts[1:]))


def _extract_rar(archive_path: Path, output_dir: Path, series_name: str) -> None:
    with libarchive.file_reader(str(archive_path)) as archive:
        for entry in archive:
            member_name = _normalized_member_name(series_name, entry.pathname)
            destination = _safe_member_path(output_dir, member_name)
            if entry.isdir:
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not entry.isreg:
                raise ValueError(f"unsupported special file in IMS archive: {entry.pathname!r}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as target:
                for block in entry.get_blocks():
                    target.write(block)


def _publish_directory(staging_dir: Path, output_dir: Path) -> None:
    try:
        os.replace(staging_dir, output_dir)
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
            raise
        shutil.copytree(staging_dir, output_dir)


def _extract_series_archives(rar_archives: dict[str, Path], output_dir: Path) -> None:
    if set(rar_archives) != EXPECTED_SERIES:
        raise ValueError(
            "IMS source must contain exactly 1st_test.rar, 2nd_test.rar, and 3rd_test.rar; "
            f"found {', '.join(sorted(rar_archives)) or 'none'}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ims-extract-", dir=output_dir.parent) as temp_name:
        staging_dir = Path(temp_name) / "dataset"
        staging_dir.mkdir()
        for series_name in sorted(EXPECTED_SERIES):
            print(f"Extracting {series_name}...", file=sys.stderr, flush=True)
            _extract_rar(rar_archives[series_name], staging_dir, series_name)

        extracted_series = {
            path.name
            for path in staging_dir.iterdir()
            if path.is_dir() and any(path.rglob("*"))
        }
        if not EXPECTED_SERIES.issubset(extracted_series):
            raise ValueError(
                "IMS RAR archives did not produce all three expected test directories"
            )
        _publish_directory(staging_dir, output_dir)


def extract_ims_archive(archive_path: Path, output_dir: Path) -> None:
    if not archive_path.exists():
        raise FileNotFoundError(f"NASA IMS archive not found: {archive_path}")
    if output_dir.exists():
        existing_series = {
            path.name
            for path in output_dir.iterdir()
            if path.is_dir() and any(path.rglob("*"))
        }
        if EXPECTED_SERIES.issubset(existing_series):
            print(f"NASA IMS data already extracted: {output_dir}")
            return
        raise FileExistsError(
            f"IMS output directory exists but is incomplete; inspect it before retrying: {output_dir}"
        )

    if archive_path.is_dir():
        rar_archives = {
            path.stem: path
            for path in archive_path.rglob("*.rar")
            if path.is_file()
        }
        _extract_series_archives(rar_archives, output_dir)
        print(f"Extracted NASA IMS series: {output_dir}")
        return
    if not archive_path.is_file():
        raise ValueError(f"IMS archive must be a ZIP file or RAR directory: {archive_path}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ims-extract-", dir=output_dir.parent) as temp_name:
        temporary_root = Path(temp_name)
        nested_7z = temporary_root / "IMS.7z"
        nested_contents = temporary_root / "nested"
        nested_contents.mkdir()

        with ZipFile(archive_path) as outer_archive:
            archive_members = [
                member
                for member in outer_archive.namelist()
                if PurePosixPath(member).name.casefold() == "ims.7z"
            ]
            if len(archive_members) != 1:
                raise ValueError("official IMS ZIP must contain exactly one IMS.7z archive")
            with outer_archive.open(archive_members[0]) as source, nested_7z.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)

        with py7zr.SevenZipFile(nested_7z, mode="r") as nested_archive:
            members = nested_archive.getnames()
            for member in members:
                _safe_member_path(nested_contents, member)
            nested_archive.extractall(path=nested_contents)

        rar_archives = {
            path.stem: path
            for path in nested_contents.rglob("*.rar")
            if path.is_file()
        }
        _extract_series_archives(rar_archives, output_dir)

    print(f"Extracted NASA IMS series: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="official IMS ZIP, or a directory containing its extracted RAR files",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    extract_ims_archive(args.archive.expanduser().resolve(), args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
