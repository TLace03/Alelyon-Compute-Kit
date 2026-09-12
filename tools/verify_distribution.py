"""Read-only exact archive verification against the reviewed staged source.

No archive extraction, package import, DLL loading, compiler, or device calls.
Hashes bind bytes; they do not authenticate the producer or establish training.
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import stat
import sys
import tarfile
import zipfile

import _build_backend as backend


def _regular_archive(path, expected):
    path = Path(path)
    if (path.name != expected or path.is_symlink() or not path.is_file()
            or not 0 < path.stat().st_size <= backend.MAX_TOTAL):
        raise backend.PackageRefused("archive-path-or-size")
    return path


def _budget(data, total):
    if len(data) > backend.MAX_FILE or total + len(data) > backend.MAX_TOTAL:
        raise backend.PackageRefused("archive-size-budget")
    return total + len(data)


def read_wheel(path):
    path = _regular_archive(path, f"{backend.NAME}-{backend.VERSION}-{backend.TAG}.whl")
    out, total = {}, 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not 1 <= len(members) <= backend.MAX_FILES:
            raise backend.PackageRefused("archive-entry-budget")
        for member in members:
            name = backend.safe_name(member.filename)
            if (name in out or member.is_dir() or member.flag_bits & 1
                    or stat.S_IFMT(member.external_attr >> 16) not in (0, stat.S_IFREG)):
                raise backend.PackageRefused("archive-duplicate-or-nonregular")
            if member.file_size > min(backend.MAX_FILE, backend.MAX_TOTAL - total):
                raise backend.PackageRefused("archive-entry-size")
            with archive.open(member) as stream:
                data = stream.read(backend.MAX_FILE + 1)
            if len(data) != member.file_size:
                raise backend.PackageRefused("archive-entry-size")
            total = _budget(data, total)
            out[name] = data
    return out


class _BoundedReader:
    def __init__(self, stream):
        self.stream, self.remaining = stream, backend.MAX_TOTAL + backend.MAX_FILES * 2048 + 10240

    def read(self, size=-1):
        data = self.stream.read(min(size, self.remaining + 1) if size >= 0 else self.remaining + 1)
        self.remaining -= len(data)
        if self.remaining < 0:
            raise backend.PackageRefused("sdist-stream-size")
        return data


def read_sdist(path):
    path = _regular_archive(path, f"{backend.NAME}-{backend.VERSION}.tar.gz")
    prefix = f"{backend.NAME}-{backend.VERSION}/"
    out, total = {}, 0
    with gzip.open(path, "rb") as compressed:
        bounded = _BoundedReader(compressed)
        # Parse this builder's USTAR framing directly. TarFile's streaming
        # reader can buffer bytes beyond the end marker, hiding trailing data
        # from a subsequent read of the underlying stream.
        while True:
            header = bounded.read(512)
            if len(header) != 512:
                raise backend.PackageRefused("sdist-truncated-header")
            if header == b"\0" * 512:
                if bounded.read(512) != b"\0" * 512:
                    raise backend.PackageRefused("sdist-end-marker")
                break
            member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            backend.safe_name(member.name)
            if (not member.name.startswith(prefix) or member.type != tarfile.REGTYPE
                    or len(out) >= backend.MAX_FILES
                    or not 0 <= member.size <= min(backend.MAX_FILE, backend.MAX_TOTAL - total)):
                raise backend.PackageRefused("sdist-entry")
            name = member.name[len(prefix):]
            backend.safe_name(name)
            if name in out:
                raise backend.PackageRefused("sdist-duplicate")
            data = bounded.read(member.size)
            padding = bounded.read((-member.size) % 512)
            if len(data) != member.size or len(padding) != (-member.size) % 512 or padding.strip(b"\0"):
                raise backend.PackageRefused("sdist-entry-size-or-padding")
            total = _budget(data, total)
            out[name] = data
        while True:
            data = bounded.read(65536)
            if not data:
                break
            if data.strip(b"\0"):
                raise backend.PackageRefused("sdist-trailing-data")
    return out


def verify_distribution(wheel, sdist, *, source_root=backend.ROOT):
    source = backend.sources(source_root)
    library = backend.native(source_root, source)
    expected_wheel = backend.wheel_files(source, library)
    expected_sdist = {**source, "PKG-INFO": backend.metadata(source)}
    actual_wheel, actual_sdist = read_wheel(wheel), read_sdist(sdist)
    if actual_wheel.keys() != expected_wheel.keys():
        raise backend.PackageRefused("wheel-entry-set")
    if actual_sdist.keys() != expected_sdist.keys():
        raise backend.PackageRefused("sdist-entry-set")
    if actual_wheel != expected_wheel:
        raise backend.PackageRefused("wheel-byte-mismatch")
    if actual_sdist != expected_sdist:
        raise backend.PackageRefused("sdist-byte-mismatch")
    if backend.sources(source_root) != source or backend.native(source_root, source) != library:
        raise backend.PackageRefused("verification-source-drift")
    return {"status": "PASS", "version": backend.VERSION,
            "wheel_files": len(actual_wheel), "sdist_files": len(actual_sdist),
            "wheel_sha256": backend.digest(Path(wheel).read_bytes()),
            "sdist_sha256": backend.digest(Path(sdist).read_bytes()),
            "training_readiness": "UNMEASURED", "performance": "UNMEASURED"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_distribution(args.wheel, args.sdist)
    except (backend.PackageRefused, OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(backend.canonical(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
