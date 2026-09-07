"""Host-only distribution verifier tests built from bounded fake archives."""

from __future__ import annotations

import csv
import gzip
import importlib.util
import io
from pathlib import Path
import shutil
import stat
import sys
import tarfile
import warnings
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "verify_distribution.py"
SPEC = importlib.util.spec_from_file_location("verify_distribution", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
distribution = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = distribution
SPEC.loader.exec_module(distribution)


METADATA = (
    "Metadata-Version: 2.4\n"
    "Name: alelyon-ai\n"
    "Version: 0.1.0a1\n"
    "Summary: Synthetic verifier fixture\n"
    "License-Expression: MIT\n"
    "License-File: LICENSE\n"
    "Requires-Python: >=3.10\n"
    "Provides-Extra: dev\n"
    'Requires-Dist: build>=1.2; extra == "dev"\n'
    'Requires-Dist: pytest>=8; extra == "dev"\n'
    "\n"
).encode() + (ROOT / "README.md").read_bytes().replace(b"\r\n", b"\n")

WHEEL_CONTROL = (
    "Wheel-Version: 1.0\n"
    "Generator: verifier-test\n"
    "Root-Is-Purelib: true\n"
    "Tag: py3-none-any\n"
    "\n"
).encode()


def _wheel_files(metadata: bytes = METADATA) -> dict[str, bytes]:
    files = {
        archived: (ROOT / source).read_bytes()
        for archived, source in distribution.WHEEL_SOURCE_MAP.items()
    }
    files.update({
        f"{distribution.DIST_INFO}/METADATA": metadata,
        f"{distribution.DIST_INFO}/WHEEL": WHEEL_CONTROL,
        f"{distribution.DIST_INFO}/top_level.txt": b"alelyon_compute_kit\n",
    })
    return files


def _record(files: dict[str, bytes]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name, data in sorted(files.items()):
        writer.writerow((name, distribution._record_digest(data), str(len(data))))
    writer.writerow((f"{distribution.DIST_INFO}/RECORD", "", ""))
    return stream.getvalue().encode()


def _write_wheel(path: Path, files: dict[str, bytes], *,
                 extra_entries: tuple[tuple[str, bytes], ...] = ()) -> Path:
    payloads = dict(files)
    payloads[f"{distribution.DIST_INFO}/RECORD"] = _record(payloads)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for name, data in extra_entries:
                archive.writestr(name, data)
    return path


def _sdist_files(metadata: bytes = METADATA) -> dict[str, bytes]:
    files = {
        relative: (ROOT / relative).read_bytes()
        for relative in distribution.SDIST_SOURCE_FILES
    }
    files.update({
        "PKG-INFO": metadata,
        "setup.cfg": b"[egg_info]\ntag_build =\ntag_date = 0\n",
        f"{distribution.EGG_INFO}/PKG-INFO": metadata,
        f"{distribution.EGG_INFO}/SOURCES.txt": (
            "\n".join(sorted(distribution.SDIST_SOURCES_LIST)) + "\n"
        ).encode(),
        f"{distribution.EGG_INFO}/dependency_links.txt": b"\n",
        f"{distribution.EGG_INFO}/requires.txt": (
            b"\n[dev]\nbuild>=1.2\npytest>=8\n"
        ),
        f"{distribution.EGG_INFO}/top_level.txt": b"alelyon_compute_kit\n",
    })
    return files


def _tar_file(name: str, data: bytes, *, kind: bytes = tarfile.REGTYPE,
              linkname: str = "") -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname
    info.size = len(data) if kind == tarfile.REGTYPE else 0
    info.mode = 0o644
    return info, data


def _write_sdist(path: Path, files: dict[str, bytes], *,
                 extras: tuple[tuple[tarfile.TarInfo, bytes], ...] = ()) -> Path:
    root = distribution.SDIST_ROOT
    directories = sorted(distribution._sdist_directories(),
                         key=lambda value: (value.count("/"), value))
    with tarfile.open(path, "w:gz") as archive:
        root_info = tarfile.TarInfo(root)
        root_info.type = tarfile.DIRTYPE
        root_info.mode = 0o755
        archive.addfile(root_info)
        for directory in directories:
            info = tarfile.TarInfo(f"{root}/{directory}")
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            archive.addfile(info)
        for relative, data in files.items():
            info, payload = _tar_file(f"{root}/{relative}", data)
            archive.addfile(info, io.BytesIO(payload))
        for info, payload in extras:
            archive.addfile(info, io.BytesIO(payload) if info.isfile() else None)
    return path


@pytest.fixture
def artifacts(tmp_path: Path) -> tuple[Path, Path]:
    wheel = _write_wheel(
        tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}-py3-none-any.whl",
        _wheel_files(),
    )
    sdist = _write_sdist(
        tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}.tar.gz",
        _sdist_files(),
    )
    return wheel, sdist


def _refused(wheel: Path, sdist: Path, code: str) -> None:
    with pytest.raises(distribution.DistributionError, match=f"^{code}$"):
        distribution.verify_distribution(wheel, sdist, source_root=ROOT)


def test_exact_reviewed_wheel_and_sdist_pass(artifacts: tuple[Path, Path]) -> None:
    report = distribution.verify_distribution(*artifacts, source_root=ROOT)
    assert report.wheel_entries == len(distribution.WHEEL_FILES) == 9
    assert report.sdist_files == len(distribution.SDIST_FILES) == 21
    assert "MANIFEST.in" in distribution.SDIST_FILES
    assert "docs/RELEASING.md" in distribution.SDIST_FILES
    assert not any(path.startswith(".github/") for path in distribution.SDIST_FILES)
    assert "tools/verify_distribution.py" in distribution.SDIST_FILES
    assert "tests/test_distribution.py" in distribution.SDIST_FILES


def test_published_description_must_match_reviewed_readme(tmp_path: Path) -> None:
    metadata = METADATA + b"Unexpected description text\n"
    wheel = _write_wheel(
        tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}-py3-none-any.whl",
        _wheel_files(metadata),
    )
    sdist = _write_sdist(
        tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}.tar.gz",
        _sdist_files(metadata),
    )
    _refused(wheel, sdist, "distribution-description-mismatch")


@pytest.mark.parametrize("addition", [
    "[Inline](LICENSE)",
    "[Reference][local]\n\n[local]: docs/ARCHITECTURE.md",
    "[Reference][local]\n\n[local]:\n  docs/ARCHITECTURE.md",
    '<a href="docs/ARCHITECTURE.md">Architecture</a>',
    '<img src="assets/logo.png" alt="Logo">',
])
def test_source_consistent_readme_with_relative_links_refuses(
        tmp_path: Path, addition: str) -> None:
    source = tmp_path / "source"
    for name in distribution.SDIST_SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    readme = (source / "README.md").read_bytes() + f"\n{addition}\n".encode()
    (source / "README.md").write_bytes(readme)
    metadata = METADATA.split(b"\n\n", 1)[0] + b"\n\n" + readme.replace(b"\r\n", b"\n")
    files = _sdist_files(metadata)
    files["README.md"] = readme
    wheel = _write_wheel(
        tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}-py3-none-any.whl",
        _wheel_files(metadata),
    )
    sdist = _write_sdist(
        tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}.tar.gz", files,
    )
    with pytest.raises(distribution.DistributionError, match="^readme-relative-link$"):
        distribution.verify_distribution(wheel, sdist, source_root=source)


@pytest.mark.parametrize("addition", [
    "[Inline](https://example.test/LICENSE)",
    "[Reference][remote]\n\n[remote]: https://example.test/architecture",
    "[Reference][remote]\n\n[remote]:\n  https://example.test/architecture",
    '<a href="https://example.test/architecture">Architecture</a>',
    '<img src="https://example.test/logo.png" alt="Logo">',
])
def test_source_consistent_readme_with_absolute_https_links_passes(
        tmp_path: Path, addition: str) -> None:
    source = tmp_path / "source"
    for name in distribution.SDIST_SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    readme = (source / "README.md").read_bytes() + f"\n{addition}\n".encode()
    (source / "README.md").write_bytes(readme)
    metadata = METADATA.split(b"\n\n", 1)[0] + b"\n\n" + readme.replace(b"\r\n", b"\n")
    files = _sdist_files(metadata)
    files["README.md"] = readme
    wheel = _write_wheel(
        tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}-py3-none-any.whl",
        _wheel_files(metadata),
    )
    sdist = _write_sdist(
        tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}.tar.gz", files,
    )
    assert distribution.verify_distribution(
        wheel, sdist, source_root=source,
    ).sdist_files == len(distribution.SDIST_FILES)


def test_cli_reports_bounded_pass_and_refusal(
        artifacts: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    wheel, sdist = artifacts
    assert distribution.main(["--wheel", str(wheel), "--sdist", str(sdist)]) == 0
    output = capsys.readouterr()
    assert output.err == "" and output.out.startswith("PASS: alelyon-ai 0.1.0a1;")
    bad = wheel.with_name("renamed.whl")
    shutil.copyfile(wheel, bad)
    assert distribution.main(["--wheel", str(bad), "--sdist", str(sdist)]) == 1
    output = capsys.readouterr()
    assert output.out == "" and output.err == "REFUSED: wheel-filename-mismatch\n"


def test_verification_never_extracts(
        artifacts: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("archive extraction is forbidden")
    monkeypatch.setattr(zipfile.ZipFile, "extract", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extract", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extractall", forbidden)
    assert distribution.verify_distribution(*artifacts, source_root=ROOT).wheel_entries == 9


def test_wheel_duplicate_path_and_escape_refuse(tmp_path: Path) -> None:
    wheel_name = f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}-py3-none-any.whl"
    files = _wheel_files()
    existing = next(iter(files))
    duplicate = _write_wheel(tmp_path / wheel_name, files,
                             extra_entries=((existing, files[existing]),))
    sdist = _write_sdist(tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}.tar.gz",
                         _sdist_files())
    _refused(duplicate, sdist, "wheel-duplicate-entry")
    escaped_dir = tmp_path / "escaped"
    escaped_dir.mkdir()
    escaped = _write_wheel(escaped_dir / wheel_name, files,
                           extra_entries=(("../outside.py", b"x"),))
    _refused(escaped, sdist, "wheel-unsafe-path")


def test_wheel_link_and_extra_file_refuse(tmp_path: Path) -> None:
    name = f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}-py3-none-any.whl"
    sdist = _write_sdist(tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}.tar.gz",
                         _sdist_files())
    extra = _write_wheel(tmp_path / name, _wheel_files(),
                         extra_entries=(("unexpected.py", b"x"),))
    _refused(extra, sdist, "wheel-entry-set-mismatch")

    link_dir = tmp_path / "link"
    link_dir.mkdir()
    payloads = _wheel_files()
    payloads[f"{distribution.DIST_INFO}/RECORD"] = _record(payloads)
    linked = link_dir / name
    with zipfile.ZipFile(linked, "w") as archive:
        for path, data in payloads.items():
            archive.writestr(path, data)
        info = zipfile.ZipInfo("link-entry")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, b"target")
    _refused(linked, sdist, "wheel-non-regular-entry")


def test_wheel_missing_file_and_source_byte_drift_refuse(tmp_path: Path) -> None:
    wheel_name = f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}-py3-none-any.whl"
    sdist = _write_sdist(tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}.tar.gz",
                         _sdist_files())
    files = _wheel_files()
    files.pop("alelyon_compute_kit/py.typed")
    _refused(_write_wheel(tmp_path / wheel_name, files), sdist,
             "wheel-entry-set-mismatch")
    drift_dir = tmp_path / "drift"
    drift_dir.mkdir()
    files = _wheel_files()
    files["alelyon_compute_kit/backends.py"] += b"# altered\n"
    _refused(_write_wheel(drift_dir / wheel_name, files), sdist,
             "wheel-source-byte-mismatch")


@pytest.mark.parametrize(("header", "replacement", "code"), [
    (b"Name: alelyon-ai", b"Name: other", "wheel-name-mismatch"),
    (b"Version: 0.1.0a1", b"Version: 9", "wheel-version-mismatch"),
    (b"Requires-Python: >=3.10", b"Requires-Python: >=3.12",
     "wheel-python-requirement-mismatch"),
])
def test_wheel_identity_and_python_metadata_refuse(
        tmp_path: Path, header: bytes, replacement: bytes, code: str) -> None:
    wheel = _write_wheel(
        tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}-py3-none-any.whl",
        _wheel_files(METADATA.replace(header, replacement)),
    )
    sdist = _write_sdist(
        tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}.tar.gz",
        _sdist_files(),
    )
    _refused(wheel, sdist, code)


def test_runtime_dependency_and_wheel_tag_refuse(tmp_path: Path) -> None:
    sdist = _write_sdist(tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}.tar.gz",
                         _sdist_files())
    wheel_name = f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}-py3-none-any.whl"
    metadata = METADATA.replace(
        b"\n\n", b"\nRequires-Dist: torch>=2\n\n", 1
    )
    _refused(_write_wheel(tmp_path / wheel_name, _wheel_files(metadata)), sdist,
             "wheel-dependencies-mismatch")

    tag_dir = tmp_path / "tag"
    tag_dir.mkdir()
    files = _wheel_files()
    files[f"{distribution.DIST_INFO}/WHEEL"] = WHEEL_CONTROL.replace(
        b"py3-none-any", b"cp312-cp312-win_amd64"
    )
    _refused(_write_wheel(tag_dir / wheel_name, files), sdist,
             "wheel-tag-mismatch")


def test_record_hash_size_missing_and_duplicate_refuse(
        tmp_path: Path, artifacts: tuple[Path, Path]) -> None:
    _, sdist = artifacts
    wheel_name = f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}-py3-none-any.whl"
    files = _wheel_files()
    record = _record(files)
    files[f"{distribution.DIST_INFO}/RECORD"] = record

    hash_dir = tmp_path / "hash"
    hash_dir.mkdir()
    with zipfile.ZipFile(hash_dir / wheel_name, "w") as archive:
        for name, data in files.items():
            if name == "alelyon_compute_kit/__init__.py":
                data += b"# after RECORD\n"
            archive.writestr(name, data)
    _refused(hash_dir / wheel_name, sdist, "wheel-source-byte-mismatch")

    integrity_dir = tmp_path / "record-integrity"
    integrity_dir.mkdir()
    rows = list(csv.reader(io.StringIO(record.decode(), newline="")))
    for row in rows:
        if row[0] == "alelyon_compute_kit/__init__.py":
            row[1] = "sha256=" + "A" * 43
            row[2] = str(int(row[2]) + 1)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    clean_files = _wheel_files()
    clean_files[f"{distribution.DIST_INFO}/RECORD"] = stream.getvalue().encode()
    with zipfile.ZipFile(integrity_dir / wheel_name, "w") as archive:
        for name, data in clean_files.items():
            archive.writestr(name, data)
    _refused(integrity_dir / wheel_name, sdist, "wheel-record-integrity")

    missing_dir = tmp_path / "missing-record"
    missing_dir.mkdir()
    bad_record = record.replace(b"alelyon_compute_kit/py.typed,", b"absent/py.typed,")
    files[f"{distribution.DIST_INFO}/RECORD"] = bad_record
    with zipfile.ZipFile(missing_dir / wheel_name, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    _refused(missing_dir / wheel_name, sdist, "wheel-record-entry-set-mismatch")

    duplicate_dir = tmp_path / "duplicate-record"
    duplicate_dir.mkdir()
    files[f"{distribution.DIST_INFO}/RECORD"] = record + record.splitlines(keepends=True)[0]
    with zipfile.ZipFile(duplicate_dir / wheel_name, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    _refused(duplicate_dir / wheel_name, sdist, "wheel-record-duplicate")


def test_sdist_duplicate_escape_link_and_extra_refuse(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}-py3-none-any.whl",
        _wheel_files(),
    )
    sdist_name = f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}.tar.gz"
    root = distribution.SDIST_ROOT
    duplicate = _tar_file(f"{root}/README.md", b"duplicate")
    _refused(wheel, _write_sdist(tmp_path / sdist_name, _sdist_files(), extras=(duplicate,)),
             "sdist-duplicate-entry")

    escape_dir = tmp_path / "escape"
    escape_dir.mkdir()
    escaped = _tar_file(f"{root}/../outside", b"x")
    _refused(wheel, _write_sdist(escape_dir / sdist_name, _sdist_files(), extras=(escaped,)),
             "sdist-unsafe-path")

    link_dir = tmp_path / "link"
    link_dir.mkdir()
    linked = _tar_file(f"{root}/linked", b"", kind=tarfile.SYMTYPE,
                       linkname="../../outside")
    _refused(wheel, _write_sdist(link_dir / sdist_name, _sdist_files(), extras=(linked,)),
             "sdist-non-regular-entry")

    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()
    extra = _tar_file(f"{root}/unexpected.txt", b"x")
    _refused(wheel, _write_sdist(extra_dir / sdist_name, _sdist_files(), extras=(extra,)),
             "sdist-entry-set-mismatch")


def test_sdist_missing_file_source_drift_and_sources_manifest_refuse(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}-py3-none-any.whl",
        _wheel_files(),
    )
    sdist_name = f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}.tar.gz"
    files = _sdist_files()
    files.pop("tools/verify_distribution.py")
    _refused(wheel, _write_sdist(tmp_path / sdist_name, files),
             "sdist-entry-set-mismatch")

    drift_dir = tmp_path / "drift"
    drift_dir.mkdir()
    files = _sdist_files()
    files["tests/test_distribution.py"] += b"# changed\n"
    _refused(wheel, _write_sdist(drift_dir / sdist_name, files),
             "sdist-source-byte-mismatch")

    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    files = _sdist_files()
    files[f"{distribution.EGG_INFO}/SOURCES.txt"] += b"unreviewed.txt\n"
    _refused(wheel, _write_sdist(sources_dir / sdist_name, files),
             "sdist-sources-mismatch")


def test_sdist_metadata_must_match_wheel_and_egg_info(
        tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}-py3-none-any.whl",
        _wheel_files(),
    )
    files = _sdist_files()
    files[f"{distribution.EGG_INFO}/PKG-INFO"] += b"changed"
    sdist = _write_sdist(
        tmp_path / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}.tar.gz", files
    )
    _refused(wheel, sdist, "distribution-metadata-byte-mismatch")


def test_archive_entry_budgets_are_enforced_before_reading(
        artifacts: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    wheel, sdist = artifacts
    monkeypatch.setattr(distribution, "MAX_ENTRY_BYTES", 8)
    _refused(wheel, sdist, "wheel-entry-size")


def test_compressed_tar_traversal_is_bounded_before_full_enumeration(
        tmp_path: Path, artifacts: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch) -> None:
    wheel, sdist = artifacts
    monkeypatch.setattr(distribution, "MAX_TAR_STREAM_BYTES", 1024)
    _refused(wheel, sdist, "sdist-stream-size")
    monkeypatch.setattr(distribution, "MAX_TAR_STREAM_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(distribution, "MAX_ENTRIES", 5)
    with pytest.raises(distribution.DistributionError,
                       match="^sdist-entry-count$"):
        distribution._read_sdist(sdist)

    monkeypatch.setattr(distribution, "MAX_ENTRIES", 128)
    large_dir = tmp_path / "compressed-large"
    large_dir.mkdir()
    root = distribution.SDIST_ROOT
    oversized = _tar_file(
        f"{root}/oversized.bin", b"\0" * (distribution.MAX_ENTRY_BYTES + 1)
    )
    large = _write_sdist(
        large_dir / f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}.tar.gz",
        _sdist_files(), extras=(oversized,),
    )
    assert large.stat().st_size < distribution.MAX_ARCHIVE_BYTES
    _refused(wheel, large, "sdist-entry-size")


def test_malformed_archives_refuse_with_bounded_codes(
        tmp_path: Path, artifacts: tuple[Path, Path]) -> None:
    wheel, sdist = artifacts
    malformed_wheel_dir = tmp_path / "malformed-wheel"
    malformed_wheel_dir.mkdir()
    bad_wheel = malformed_wheel_dir / (
        f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}-py3-none-any.whl"
    )
    bad_wheel.write_bytes(b"not a zip")
    _refused(bad_wheel, sdist, "wheel-malformed")
    malformed_sdist_dir = tmp_path / "malformed-sdist"
    malformed_sdist_dir.mkdir()
    bad_sdist = malformed_sdist_dir / (
        f"{distribution.ARCHIVE_NAME}-{distribution.VERSION}.tar.gz"
    )
    bad_sdist.write_bytes(gzip.compress(b"not a tar"))
    _refused(wheel, bad_sdist, "sdist-malformed")


def test_runtime_and_project_version_contracts_are_independently_bound() -> None:
    wheel_files = _wheel_files()
    sdist_files = _sdist_files()
    distribution._verify_source_contract(wheel_files, sdist_files)
    wheel_files["alelyon_compute_kit/__init__.py"] = wheel_files[
        "alelyon_compute_kit/__init__.py"
    ].replace(b'__version__ = "0.1.0a1"', b'__version__ = "9"')
    with pytest.raises(distribution.DistributionError,
                       match="^package-version-mismatch$"):
        distribution._verify_source_contract(wheel_files, sdist_files)
    wheel_files = _wheel_files()
    sdist_files["pyproject.toml"] = sdist_files["pyproject.toml"].replace(
        b'dependencies = []', b'dependencies = ["torch"]'
    )
    with pytest.raises(distribution.DistributionError,
                       match="^project-metadata-mismatch$"):
        distribution._verify_source_contract(wheel_files, sdist_files)
