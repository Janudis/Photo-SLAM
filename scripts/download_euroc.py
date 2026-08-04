#!/usr/bin/env python3

from __future__ import annotations

import io
import re
import time
import urllib.error
import urllib.request
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from email.message import Message
from pathlib import Path


ETH_BITSTREAM_BASE = (
    "https://www.research-collection.ethz.ch/server/api/core/bitstreams"
)
OUTPUT_ROOT = Path(__file__).resolve().parent / "data" / "EuRoC"
BLOCK_SIZE = 16 * 1024 * 1024
COPY_SIZE = 8 * 1024 * 1024
MAX_RETRIES = 5


@dataclass(frozen=True)
class Sequence:
    name: str
    bitstream_id: str
    archive_member: str

    @property
    def url(self) -> str:
        return f"{ETH_BITSTREAM_BASE}/{self.bitstream_id}/content"


SEQUENCES = (
    Sequence(
        "MH_01_easy",
        "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe",
        "machine_hall/MH_01_easy/MH_01_easy.zip",
    ),
    Sequence(
        "MH_02_easy",
        "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe",
        "machine_hall/MH_02_easy/MH_02_easy.zip",
    ),
    Sequence(
        "V1_01_easy",
        "02ecda9a-298f-498b-970c-b7c44334d880",
        "vicon_room1/V1_01_easy/V1_01_easy.zip",
    ),
    Sequence(
        "V2_01_easy",
        "ea12bc01-3677-4b4c-853d-87c7870b8c44",
        "vicon_room2/V2_01_easy/V2_01_easy.zip",
    ),
)


class HttpRangeReader(io.RawIOBase):
    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self.position = 0
        self.size = self._read_remote_size()
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")

        if position < 0:
            raise ValueError("Cannot seek before the beginning of the archive")
        self.position = min(position, self.size)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)

        output = bytearray()
        while size > 0:
            block_index, block_offset = divmod(self.position, BLOCK_SIZE)
            block = self._read_block(block_index)
            count = min(size, len(block) - block_offset)
            if count <= 0:
                break
            output.extend(block[block_offset : block_offset + count])
            self.position += count
            size -= count
        return bytes(output)

    def _read_remote_size(self) -> int:
        _, headers = self._request_range(0, 0)
        content_range = headers.get("Content-Range", "")
        match = re.search(r"/(\d+)$", content_range)
        if match is None:
            raise RuntimeError(
                f"ETH server did not report archive size for {self.url}: "
                f"{content_range!r}"
            )
        return int(match.group(1))

    def _read_block(self, block_index: int) -> bytes:
        cached = self.cache.pop(block_index, None)
        if cached is not None:
            self.cache[block_index] = cached
            return cached

        start = block_index * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, self.size) - 1
        data, _ = self._request_range(start, end)
        expected_size = end - start + 1
        if len(data) != expected_size:
            raise RuntimeError(
                f"Incomplete HTTP range {start}-{end}: "
                f"received {len(data)} of {expected_size} bytes"
            )

        self.cache[block_index] = data
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        return data

    def _request_range(
        self,
        start: int,
        end: int,
    ) -> tuple[bytes, Message]:
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            request = urllib.request.Request(
                self.url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "Photo-SLAM EuRoC downloader",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise RuntimeError(
                            f"ETH server ignored byte range {start}-{end}: "
                            f"HTTP {response.status}"
                        )
                    return response.read(), response.headers
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                last_error = error
                if attempt < MAX_RETRIES:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"Failed to download byte range {start}-{end} after "
            f"{MAX_RETRIES} attempts"
        ) from last_error


def sequence_is_ready(sequence_dir: Path) -> bool:
    camera_csv = sequence_dir / "mav0" / "cam0" / "data.csv"
    camera_data = sequence_dir / "mav0" / "cam0" / "data"
    return camera_csv.is_file() and camera_data.is_dir() and any(camera_data.iterdir())


def download_inner_archive(sequence: Sequence, archive_path: Path) -> None:
    partial_path = archive_path.with_suffix(".zip.part")
    partial_path.unlink(missing_ok=True)

    print(f"[EuRoC] Reading official ETH archive for {sequence.name}...")
    reader = HttpRangeReader(sequence.url)
    with zipfile.ZipFile(reader) as outer_archive:
        try:
            info = outer_archive.getinfo(sequence.archive_member)
        except KeyError as error:
            raise RuntimeError(
                f"{sequence.archive_member} is missing from the ETH archive"
            ) from error

        downloaded = 0
        next_report = 256 * 1024 * 1024
        with outer_archive.open(info) as source, partial_path.open("wb") as target:
            while True:
                chunk = source.read(COPY_SIZE)
                if not chunk:
                    break
                target.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    percent = 100.0 * downloaded / info.file_size
                    print(
                        f"[EuRoC] {sequence.name}: "
                        f"{downloaded / (1024**3):.2f}/"
                        f"{info.file_size / (1024**3):.2f} GiB "
                        f"({percent:.1f}%)"
                    )
                    next_report += 256 * 1024 * 1024

    if not zipfile.is_zipfile(partial_path):
        raise RuntimeError(f"Downloaded file is not a valid ZIP: {partial_path}")
    partial_path.replace(archive_path)


def extract_sequence(archive_path: Path, sequence_dir: Path) -> None:
    sequence_dir.mkdir(parents=True, exist_ok=True)
    resolved_root = sequence_dir.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (sequence_dir / member.filename).resolve()
            if not destination.is_relative_to(resolved_root):
                raise RuntimeError(
                    f"Unsafe path in {archive_path.name}: {member.filename}"
                )
        archive.extractall(sequence_dir)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for sequence in SEQUENCES:
        sequence_dir = OUTPUT_ROOT / sequence.name
        archive_path = OUTPUT_ROOT / f"{sequence.name}.zip"

        if sequence_is_ready(sequence_dir):
            print(f"[EuRoC] {sequence.name} already extracted; skipping.")
            continue

        if not zipfile.is_zipfile(archive_path):
            download_inner_archive(sequence, archive_path)

        print(f"[EuRoC] Extracting {archive_path.name}...")
        extract_sequence(archive_path, sequence_dir)
        if not sequence_is_ready(sequence_dir):
            raise RuntimeError(
                f"{sequence.name} extraction did not produce mav0/cam0 data"
            )
        print(f"[EuRoC] Ready: {sequence_dir}")

    print(f"[EuRoC] All sequences are available under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
