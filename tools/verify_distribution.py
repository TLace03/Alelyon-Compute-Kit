"""Verify the exact reviewed contents of Alelyon Compute Kit distributions.

This tool reads a wheel and source distribution without extracting, importing,
building, installing, or executing either artifact.  The allowlists are tied to
the 0.1.0a1 package surface and must be reviewed when that surface changes.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
import gzip
import hashlib
from html.parser import HTMLParser
import io
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
from typing import Mapping, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DIST_NAME = "alelyon-ai"
ARCHIVE_NAME = "alelyon_ai"
VERSION = "0.1.0a1"
PYTHON_REQUIRES = ">=3.10"
DIST_INFO = f"{ARCHIVE_NAME}-{VERSION}.dist-info"
SDIST_ROOT = f"{ARCHIVE_NAME}-{VERSION}"
EGG_INFO = f"src/{ARCHIVE_NAME}.egg-info"

MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_ENTRY_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_ENTRIES = 128
MAX_PATH_CHARS = 240
MAX_TAR_STREAM_BYTES = MAX_TOTAL_BYTES + MAX_ENTRIES * 2048 + 10240

PACKAGE_SOURCE_FILES = (
    "src/alelyon_compute_kit/__init__.py",
    "src/alelyon_compute_kit/backends.py",
    "src/alelyon_compute_kit/capabilities.py",
    "src/alelyon_compute_kit/py.typed",
)

WHEEL_SOURCE_MAP = {
    "alelyon_compute_kit/__init__.py": "src/alelyon_compute_kit/__init__.py",
    "alelyon_compute_kit/backends.py": "src/alelyon_compute_kit/backends.py",
    "alelyon_compute_kit/capabilities.py": "src/alelyon_compute_kit/capabilities.py",
    "alelyon_compute_kit/py.typed": "src/alelyon_compute_kit/py.typed",
    f"{DIST_INFO}/licenses/LICENSE": "LICENSE",
}

WHEEL_FILES = frozenset({
    *WHEEL_SOURCE_MAP,
    f"{DIST_INFO}/METADATA",
    f"{DIST_INFO}/WHEEL",
    f"{DIST_INFO}/top_level.txt",
    f"{DIST_INFO}/RECORD",
})

SDIST_SOURCE_FILES = (
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "pyproject.toml",
    "docs/ARCHITECTURE.md",
    "docs/RELEASING.md",
    *PACKAGE_SOURCE_FILES,
    "tests/test_backends.py",
    "tests/test_capabilities.py",
    "tests/test_distribution.py",
    "tools/verify_distribution.py",
)

SDIST_GENERATED_FILES = (
    "PKG-INFO",
    "setup.cfg",
    f"{EGG_INFO}/PKG-INFO",
    f"{EGG_INFO}/SOURCES.txt",
    f"{EGG_INFO}/dependency_links.txt",
    f"{EGG_INFO}/requires.txt",
    f"{EGG_INFO}/top_level.txt",
)

SDIST_FILES = frozenset((*SDIST_SOURCE_FILES, *SDIST_GENERATED_FILES))
SDIST_SOURCES_LIST = frozenset({
    *SDIST_SOURCE_FILES,
    f"{EGG_INFO}/PKG-INFO",
    f"{EGG_INFO}/SOURCES.txt",
    f"{EGG_INFO}/dependency_links.txt",
    f"{EGG_INFO}/requires.txt",
    f"{EGG_INFO}/top_level.txt",
})

EXPECTED_REQUIRES_DIST = frozenset({
    'build>=1.2; extra == "dev"',
    'pytest>=8; extra == "dev"',
})


class DistributionError(ValueError):
    """A bounded, stable refusal from distribution verification."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class DistributionReport:
    wheel_entries: int
    sdist_files: int


def _refuse(code: str) -> None:
    raise DistributionError(code)


def _archive_path(path: Path, expected_name: str, label: str) -> Path:
    path = Path(path)
    if path.name != expected_name:
        _refuse(f"{label}-filename-mismatch")
    if path.is_symlink() or not path.is_file():
        _refuse(f"{label}-not-regular-file")
    try:
        size = path.stat().st_size
    except OSError:
        _refuse(f"{label}-unreadable")
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        _refuse(f"{label}-archive-size")
    return path


def _safe_parts(name: str, label: str) -> tuple[str, ...]:
    if (type(name) is not str or not name or len(name) > MAX_PATH_CHARS
            or "\\" in name or "\x00" in name or name.startswith("/")):
        _refuse(f"{label}-unsafe-path")
    stripped = name[:-1] if name.endswith("/") else name
    parts = stripped.split("/")
    if not stripped or any(part in ("", ".", "..") for part in parts):
        _refuse(f"{label}-unsafe-path")
    # PurePosixPath is a second independent normalization check; drive-like
    # names remain outside the reviewed allowlist but are rejected here too.
    parsed = PurePosixPath(stripped)
    if parsed.is_absolute() or tuple(parsed.parts) != tuple(parts) or ":" in parts[0]:
        _refuse(f"{label}-unsafe-path")
    return tuple(parts)


def _check_entry_budget(size: int, total: int, label: str) -> int:
    if type(size) is not int or size < 0 or size > MAX_ENTRY_BYTES:
        _refuse(f"{label}-entry-size")
    total += size
    if total > MAX_TOTAL_BYTES:
        _refuse(f"{label}-total-size")
    return total


def _source_bytes(source_root: Path, relative: str) -> bytes:
    path = source_root / relative
    if path.is_symlink() or not path.is_file():
        _refuse("source-file-missing-or-linked")
    try:
        size = path.stat().st_size
        if size > MAX_ENTRY_BYTES:
            _refuse("source-file-size")
        data = path.read_bytes()
    except OSError:
        _refuse("source-file-unreadable")
    if len(data) != size:
        _refuse("source-file-size-changed")
    return data


def _read_wheel(path: Path) -> dict[str, bytes]:
    path = _archive_path(
        path, f"{ARCHIVE_NAME}-{VERSION}-py3-none-any.whl", "wheel"
    )
    try:
        with path.open("rb") as stream:
            if stream.read(4) != b"PK\x03\x04":
                _refuse("wheel-malformed")
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if not infos or len(infos) > MAX_ENTRIES:
                _refuse("wheel-entry-count")
            names: list[str] = []
            total = 0
            for info in infos:
                _safe_parts(info.filename, "wheel")
                if info.is_dir() or info.flag_bits & 1:
                    _refuse("wheel-non-regular-entry")
                mode = info.external_attr >> 16
                kind = stat.S_IFMT(mode)
                if kind not in (0, stat.S_IFREG):
                    _refuse("wheel-non-regular-entry")
                total = _check_entry_budget(info.file_size, total, "wheel")
                names.append(info.filename)
            if len(set(names)) != len(names):
                _refuse("wheel-duplicate-entry")
            if set(names) != WHEEL_FILES:
                _refuse("wheel-entry-set-mismatch")
            result = {}
            for info in infos:
                try:
                    data = archive.read(info)
                except (OSError, RuntimeError, zipfile.BadZipFile):
                    _refuse("wheel-entry-unreadable")
                if len(data) != info.file_size:
                    _refuse("wheel-entry-size-mismatch")
                result[info.filename] = data
            return result
    except DistributionError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile):
        _refuse("wheel-malformed")


def _sdist_directories() -> frozenset[str]:
    directories = set()
    for filename in SDIST_FILES:
        parts = filename.split("/")[:-1]
        for count in range(1, len(parts) + 1):
            directories.add("/".join(parts[:count]))
    return frozenset(directories)


class _BoundedReader:
    """Count decompressed bytes before tar parsing can traverse past the cap."""

    def __init__(self, stream, limit: int) -> None:
        self._stream = stream
        self._limit = limit
        self._read = 0

    def read(self, size: int = -1) -> bytes:
        remaining = self._limit - self._read
        request = remaining + 1 if size < 0 else min(size, remaining + 1)
        data = self._stream.read(request)
        self._read += len(data)
        if self._read > self._limit:
            _refuse("sdist-stream-size")
        return data


def _read_sdist(path: Path) -> dict[str, bytes]:
    path = _archive_path(path, f"{ARCHIVE_NAME}-{VERSION}.tar.gz", "sdist")
    allowed_directories = _sdist_directories()
    try:
        with path.open("rb") as stream:
            if stream.read(2) != b"\x1f\x8b":
                _refuse("sdist-compression-mismatch")
        with path.open("rb") as compressed:
            with gzip.GzipFile(fileobj=compressed, mode="rb") as uncompressed:
                bounded = _BoundedReader(uncompressed, MAX_TAR_STREAM_BYTES)
                with tarfile.open(fileobj=bounded, mode="r|") as archive:
                    seen = set()
                    file_members = set()
                    result = {}
                    total = 0
                    root_seen = False
                    count = 0
                    for member in archive:
                        count += 1
                        if count > MAX_ENTRIES:
                            _refuse("sdist-entry-count")
                        parts = _safe_parts(member.name, "sdist")
                        normalized = "/".join(parts)
                        if normalized in seen:
                            _refuse("sdist-duplicate-entry")
                        seen.add(normalized)
                        if parts[0] != SDIST_ROOT:
                            _refuse("sdist-root-mismatch")
                        relative = "/".join(parts[1:])
                        if not relative:
                            if not member.isdir() or root_seen:
                                _refuse("sdist-root-entry")
                            root_seen = True
                            continue
                        if member.isdir():
                            if member.size != 0 or relative not in allowed_directories:
                                _refuse("sdist-entry-set-mismatch")
                            continue
                        if not member.isfile():
                            _refuse("sdist-non-regular-entry")
                        total = _check_entry_budget(member.size, total, "sdist")
                        file_members.add(relative)
                        try:
                            stream = archive.extractfile(member)
                            data = None if stream is None else stream.read(MAX_ENTRY_BYTES + 1)
                        except (OSError, tarfile.TarError):
                            _refuse("sdist-entry-unreadable")
                        if data is None or len(data) != member.size:
                            _refuse("sdist-entry-size-mismatch")
                        result[relative] = data
                    if not count:
                        _refuse("sdist-entry-count")
                    if not root_seen:
                        _refuse("sdist-root-entry")
                    if file_members != SDIST_FILES:
                        _refuse("sdist-entry-set-mismatch")
                while True:
                    trailing = bounded.read(65536)
                    if not trailing:
                        break
                    if trailing.strip(b"\0"):
                        _refuse("sdist-trailing-data")
                return result
    except DistributionError:
        raise
    except (OSError, EOFError, tarfile.TarError):
        _refuse("sdist-malformed")


def _one_header(message, name: str, expected: str, code: str) -> None:
    values = message.get_all(name, [])
    if len(values) != 1 or str(values[0]) != expected:
        _refuse(code)


def _verify_metadata(data: bytes, label: str) -> None:
    try:
        message = BytesParser(policy=policy.default).parsebytes(data)
    except (UnicodeError, ValueError):
        _refuse(f"{label}-metadata-malformed")
    if message.defects or message.is_multipart():
        _refuse(f"{label}-metadata-malformed")
    _one_header(message, "Metadata-Version", "2.4", f"{label}-metadata-version")
    _one_header(message, "Name", DIST_NAME, f"{label}-name-mismatch")
    _one_header(message, "Version", VERSION, f"{label}-version-mismatch")
    _one_header(message, "Requires-Python", PYTHON_REQUIRES,
                f"{label}-python-requirement-mismatch")
    _one_header(message, "License-Expression", "MIT", f"{label}-license-mismatch")
    _one_header(message, "License-File", "LICENSE", f"{label}-license-file-mismatch")
    if message.get_all("Provides-Extra", []) != ["dev"]:
        _refuse(f"{label}-extras-mismatch")
    requirements = message.get_all("Requires-Dist", [])
    if len(requirements) != len(EXPECTED_REQUIRES_DIST) or set(requirements) != EXPECTED_REQUIRES_DIST:
        _refuse(f"{label}-dependencies-mismatch")


def _verify_wheel_metadata(files: Mapping[str, bytes]) -> None:
    _verify_metadata(files[f"{DIST_INFO}/METADATA"], "wheel")
    try:
        wheel = BytesParser(policy=policy.default).parsebytes(files[f"{DIST_INFO}/WHEEL"])
    except (UnicodeError, ValueError):
        _refuse("wheel-control-malformed")
    if wheel.defects or wheel.is_multipart():
        _refuse("wheel-control-malformed")
    expected_names = {"Wheel-Version", "Generator", "Root-Is-Purelib", "Tag"}
    if set(wheel.keys()) != expected_names:
        _refuse("wheel-control-fields")
    _one_header(wheel, "Wheel-Version", "1.0", "wheel-control-version")
    _one_header(wheel, "Root-Is-Purelib", "true", "wheel-not-pure")
    _one_header(wheel, "Tag", "py3-none-any", "wheel-tag-mismatch")
    generators = wheel.get_all("Generator", [])
    if len(generators) != 1 or not str(generators[0]).strip():
        _refuse("wheel-generator-missing")
    if files[f"{DIST_INFO}/top_level.txt"] != b"alelyon_compute_kit\n":
        _refuse("wheel-top-level-mismatch")


def _record_digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


def _verify_record(files: Mapping[str, bytes]) -> None:
    record_name = f"{DIST_INFO}/RECORD"
    try:
        text = files[record_name].decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeError, csv.Error):
        _refuse("wheel-record-malformed")
    records: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            _refuse("wheel-record-malformed")
        name, digest, size = row
        _safe_parts(name, "wheel-record")
        if name in records:
            _refuse("wheel-record-duplicate")
        records[name] = digest, size
    if set(records) != set(files):
        _refuse("wheel-record-entry-set-mismatch")
    for name, data in files.items():
        digest, size = records[name]
        if name == record_name:
            if digest or size:
                _refuse("wheel-record-self-hash")
        elif digest != _record_digest(data) or size != str(len(data)):
            _refuse("wheel-record-integrity")


def _nonempty_lines(data: bytes, code: str) -> tuple[str, ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeError:
        _refuse(code)
    if "\x00" in text:
        _refuse(code)
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def _normalized_text_bytes(data: bytes, code: str) -> bytes:
    normalized = data.replace(b"\r\n", b"\n")
    if b"\r" in normalized:
        _refuse(code)
    try:
        normalized.decode("utf-8")
    except UnicodeError:
        _refuse(code)
    return normalized


class _ReadmeHtmlLinks(HTMLParser):
    """Collect explicitly declared HTML link and image destinations."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.destinations: list[str] = []

    def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del tag
        for name, value in attrs:
            if name.lower() in {"href", "src"} and value is not None:
                self.destinations.append(value.strip())

    def handle_startendtag(
            self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def _verify_readme_links(readme: str) -> None:
    """Admit HTTPS destinations in the bounded syntax this verifier checks.

    This is intentionally not a complete Markdown parser.  It checks ordinary
    inline Markdown destinations, one-line or next-line reference definitions,
    and HTML ``href``/``src`` attributes.  The reviewed README must use one of
    those forms so every link destination remains visible to this gate.
    """
    destinations: list[str] = []
    inline = re.compile(r"\]\(\s*(?:<([^>\r\n]+)>|([^\s)]+))")
    destinations.extend(match.group(1) or match.group(2)
                        for match in inline.finditer(readme))
    reference = re.compile(
        r"(?m)^[ \t]{0,3}\[[^\]\r\n]+\]:[ \t]*"
        r"(?:\r?\n[ \t]+)?(?:<([^>\r\n]+)>|([^\s]+))"
    )
    destinations.extend(match.group(1) or match.group(2)
                        for match in reference.finditer(readme))
    html = _ReadmeHtmlLinks()
    html.feed(readme)
    html.close()
    destinations.extend(html.destinations)
    if any(not destination.startswith("https://") for destination in destinations):
        _refuse("readme-relative-link")


def _verify_sdist_metadata(files: Mapping[str, bytes], wheel_metadata: bytes) -> None:
    package_info = files["PKG-INFO"]
    if package_info != files[f"{EGG_INFO}/PKG-INFO"] or package_info != wheel_metadata:
        _refuse("distribution-metadata-byte-mismatch")
    _verify_metadata(package_info, "sdist")
    try:
        expected_description = files["README.md"].decode("utf-8").replace("\r\n", "\n")
    except UnicodeError:
        _refuse("readme-encoding")
    description = BytesParser(policy=policy.default).parsebytes(package_info).get_payload(decode=True)
    if not isinstance(description, bytes) or description.replace(b"\r\n", b"\n") != expected_description.encode("utf-8"):
        _refuse("distribution-description-mismatch")
    _verify_readme_links(expected_description)
    normalized_metadata = _normalized_text_bytes(
        package_info, "distribution-metadata-text-malformed"
    )
    if b"\n\n" not in normalized_metadata:
        _refuse("distribution-metadata-body-missing")
    description = normalized_metadata.split(b"\n\n", 1)[1]
    readme = _normalized_text_bytes(files["README.md"], "readme-text-malformed")
    if description != readme:
        _refuse("distribution-readme-metadata-mismatch")
    sources = _nonempty_lines(files[f"{EGG_INFO}/SOURCES.txt"], "sdist-sources-malformed")
    if len(sources) != len(set(sources)):
        _refuse("sdist-sources-duplicate")
    for name in sources:
        _safe_parts(name, "sdist-sources")
    if set(sources) != SDIST_SOURCES_LIST:
        _refuse("sdist-sources-mismatch")
    if files[f"{EGG_INFO}/top_level.txt"] != b"alelyon_compute_kit\n":
        _refuse("sdist-top-level-mismatch")
    if files[f"{EGG_INFO}/dependency_links.txt"].strip():
        _refuse("sdist-dependency-links-mismatch")
    requires = _nonempty_lines(files[f"{EGG_INFO}/requires.txt"],
                               "sdist-requires-malformed")
    if requires != ("[dev]", "build>=1.2", "pytest>=8"):
        _refuse("sdist-dependencies-mismatch")
    setup = _nonempty_lines(files["setup.cfg"], "sdist-setup-malformed")
    if setup != ("[egg_info]", "tag_build =", "tag_date = 0"):
        _refuse("sdist-setup-mismatch")


def _verify_source_contract(wheel_files: Mapping[str, bytes],
                            sdist_files: Mapping[str, bytes]) -> None:
    try:
        module = ast.parse(
            wheel_files["alelyon_compute_kit/__init__.py"].decode("utf-8")
        )
    except (UnicodeError, SyntaxError):
        _refuse("package-version-malformed")
    versions = []
    for node in module.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "__version__"
                   for target in targets):
                value = node.value
                versions.append(
                    value.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                    else None
                )
    if versions != [VERSION]:
        _refuse("package-version-mismatch")

    try:
        lines = sdist_files["pyproject.toml"].decode("utf-8").splitlines()
    except UnicodeError:
        _refuse("project-metadata-malformed")
    project_lines = []
    inside = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inside = stripped == "[project]"
            continue
        if inside and stripped and not stripped.startswith("#"):
            project_lines.append(stripped)
    required = {
        "name": f'name = "{DIST_NAME}"',
        "version": f'version = "{VERSION}"',
        "requires-python": f'requires-python = "{PYTHON_REQUIRES}"',
        "dependencies": "dependencies = []",
    }
    for key, expected in required.items():
        declarations = [line for line in project_lines
                        if line.split("=", 1)[0].strip() == key]
        if declarations != [expected]:
            _refuse("project-metadata-mismatch")


def verify_distribution(wheel: Path, sdist: Path, *,
                        source_root: Path = ROOT) -> DistributionReport:
    """Verify exact contents, source bytes, metadata and wheel RECORD hashes."""
    source_root = Path(source_root)
    wheel_files = _read_wheel(Path(wheel))
    sdist_files = _read_sdist(Path(sdist))

    for archived, source in WHEEL_SOURCE_MAP.items():
        if wheel_files[archived] != _source_bytes(source_root, source):
            _refuse("wheel-source-byte-mismatch")
    for relative in SDIST_SOURCE_FILES:
        if sdist_files[relative] != _source_bytes(source_root, relative):
            _refuse("sdist-source-byte-mismatch")

    _verify_wheel_metadata(wheel_files)
    _verify_record(wheel_files)
    _verify_sdist_metadata(sdist_files, wheel_files[f"{DIST_INFO}/METADATA"])
    _verify_source_contract(wheel_files, sdist_files)
    return DistributionReport(len(wheel_files), len(sdist_files))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path,
                        help="reviewed wheel to verify without extracting")
    parser.add_argument("--sdist", required=True, type=Path,
                        help="reviewed .tar.gz source distribution")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = verify_distribution(args.wheel, args.sdist)
    except DistributionError as error:
        print(f"REFUSED: {error.code}", file=sys.stderr)
        return 1
    print(
        f"PASS: {DIST_NAME} {VERSION}; "
        f"wheel_files={report.wheel_entries}; sdist_files={report.sdist_files}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
