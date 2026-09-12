"""Bounded, stdlib-only packaging backend. Never compiles or loads native code.

The source manifest is a reviewed file boundary, not an authenticated signature.
The explicit native builder supplies a source-bound DLL receipt. Installing an
sdist without that separate build refuses; pip never launches Cargo here.
"""
from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tarfile
import zipfile

VERSION = "0.1.0a2"
NAME = "alelyon_ai"
TAG = "py3-none-win_amd64"
INFO = f"{NAME}-{VERSION}.dist-info"
ROOT = Path(__file__).resolve().parents[1]
KIT = "src/alelyon_compute_kit/_kit"
DLL = f"{KIT}/python/_lib/alelyon_compute_kit.dll"
RECEIPT = f"{KIT}/python/_lib/BUILD.json"
MAX_FILE = 32 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024
MAX_FILES = 256
RUST_VERSION = "1.97.1"
# The exporter normalizes Cargo.lock line endings; notices bind that exact form.
LOCK_SHA256 = "f7ce6708fcb5ead952c826cb635bf4af74f831cd9f65218300c6cbf325b578cf"


class PackageRefused(ValueError):
    """Named refusal; no device, compiler, or network fallback."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise PackageRefused("duplicate-json-key")
        out[key] = value
    return out


def read_json(data):
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise PackageRefused("nonfinite-json")
        return number

    try:
        return json.loads(data, object_pairs_hook=_pairs,
                          parse_float=finite_float,
                          parse_constant=lambda _: (_ for _ in ()).throw(PackageRefused("nonfinite-json")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PackageRefused("malformed-json") from exc


def safe_name(name):
    if (type(name) is not str or not name or len(name) > 240 or "\\" in name
            or ":" in name or "\x00" in name or name.startswith("/")):
        raise PackageRefused("unsafe-path")
    if any(part in ("", ".", "..") for part in name.split("/")):
        raise PackageRefused("unsafe-path")
    return name


def read_file(root, name, *, max_bytes=MAX_FILE):
    safe_name(name)
    root = Path(root).resolve()
    path = root
    for part in name.split("/"):
        path = path / part
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise PackageRefused("linked-source-path")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise PackageRefused("source-missing")
    size = path.stat().st_size
    if size > max_bytes:
        raise PackageRefused("source-size")
    data = path.read_bytes()
    if len(data) != size:
        raise PackageRefused("source-changed")
    return data


def sources(root=ROOT):
    raw = read_file(root, "SOURCE_MANIFEST.json", max_bytes=128 * 1024)
    manifest = read_json(raw)
    if (type(manifest) is not dict or set(manifest) != {"schema", "version", "files"}
            or type(manifest["schema"]) is not int or manifest["schema"] != 1
            or manifest["version"] != VERSION or type(manifest["files"]) is not dict
            or not 1 <= len(manifest["files"]) <= MAX_FILES):
        raise PackageRefused("source-manifest-schema")
    out, total = {}, len(raw)
    for name, sha in manifest["files"].items():
        safe_name(name)
        if (name in {"SOURCE_MANIFEST.json", DLL, RECEIPT} or type(sha) is not str
                or not re.fullmatch("[0-9a-f]{64}", sha)):
            raise PackageRefused("source-manifest-entry")
        data = read_file(root, name, max_bytes=min(MAX_FILE, MAX_TOTAL - total))
        if digest(data) != sha:
            raise PackageRefused("source-digest-mismatch")
        out[name] = data
        total += len(data)
    for name in ("LICENSE", "THIRD-PARTY-NOTICES.txt", "README.md", "pyproject.toml",
                 "tools/_build_backend.py", "src/alelyon_compute_kit/__init__.py",
                 f"{KIT}/Cargo.lock", f"{KIT}/UPSTREAM.json"):
        if name not in out:
            raise PackageRefused("required-source-missing")
    out["SOURCE_MANIFEST.json"] = raw
    if digest(out[f"{KIT}/Cargo.lock"]) != LOCK_SHA256:
        raise PackageRefused("dependency-lock-needs-notice-review")
    validate_payload(out)
    return out


def validate_payload(source):
    upstream = read_json(source[f"{KIT}/UPSTREAM.json"])
    if (type(upstream) is not dict or type(upstream.get("files")) is not dict
            or not upstream["files"] or upstream.get("payload_root") != KIT):
        raise PackageRefused("upstream-schema")
    actual = {name[len(KIT) + 1:] for name in source if name.startswith(KIT + "/")}
    expected = set(upstream["files"]) | {"UPSTREAM.json", "__init__.py"}
    if actual != expected:
        raise PackageRefused("upstream-file-set")
    for name, sha in upstream["files"].items():
        safe_name(name)
        if digest(source[f"{KIT}/{name}"]) != sha:
            raise PackageRefused("upstream-input-digest")
    manifest = read_json(source[f"{KIT}/kernels/MANIFEST.json"])
    kernels = manifest.get("kernels") if type(manifest) is dict else None
    if type(kernels) is not dict or {f"kernels/{name}" for name in kernels} != {
            name for name in upstream["files"] if name.endswith(".spv")}:
        raise PackageRefused("shader-manifest-set")
    for name, entry in kernels.items():
        if type(entry) is not dict or type(entry.get("source")) is not str:
            raise PackageRefused("shader-manifest-schema")
        source_name = "kernels/" + safe_name(entry["source"])
        if source_name not in upstream["files"]:
            raise PackageRefused("shader-source-missing")
        data, text = source[f"{KIT}/kernels/{name}"], source[f"{KIT}/{source_name}"]
        if (type(entry.get("spirv_bytes")) is not int or entry["spirv_bytes"] != len(data)
                or entry.get("spirv_sha256") != digest(data) or entry.get("source_sha256") != digest(text)):
            raise PackageRefused("shader-input-digest")


def validate_pe(data):
    if not 256 <= len(data) <= MAX_FILE or data[:2] != b"MZ":
        raise PackageRefused("native-not-pe")
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if offset < 64 or offset + 26 > len(data) or data[offset:offset + 4] != b"PE\0\0":
        raise PackageRefused("native-pe-header")
    machine, sections = struct.unpack_from("<HH", data, offset + 4)
    optional_size, flags = struct.unpack_from("<HH", data, offset + 20)
    if (machine != 0x8664 or not 1 <= sections <= 96 or not flags & 0x2000
            or optional_size < 112 or offset + 24 + optional_size + sections * 40 > len(data)
            or struct.unpack_from("<H", data, offset + 24)[0] != 0x20B):
        raise PackageRefused("native-target-mismatch")


def native(root, source):
    data, raw = read_file(root, DLL), read_file(root, RECEIPT)
    validate_pe(data)
    record = read_json(raw)
    expected = {"schema", "version", "target", "profile", "features", "rustc",
                "source_manifest_sha256", "library_sha256", "library_bytes"}
    if (type(record) is not dict or set(record) != expected
            or type(record["schema"]) is not int or record["schema"] != 1
            or record["version"] != VERSION or record["target"] != "x86_64-pc-windows-msvc"
            or record["profile"] != "release" or record["features"] != []
            or type(record["rustc"]) is not str
            or not record["rustc"].startswith(f"rustc {RUST_VERSION} ")
            or record["source_manifest_sha256"] != digest(source["SOURCE_MANIFEST.json"])
            or record["library_sha256"] != digest(data)
            or type(record["library_bytes"]) is not int or record["library_bytes"] != len(data)):
        raise PackageRefused("native-receipt-mismatch")
    return {DLL: data, RECEIPT: raw}


def metadata(source):
    return (f"Metadata-Version: 2.4\nName: alelyon-ai\nVersion: {VERSION}\n"
            "Summary: Experimental vendor-neutral compute SDK with explicit native execution\n"
            "Requires-Python: >=3.10\nLicense-Expression: Apache-2.0\n"
            "License-File: LICENSE\nLicense-File: THIRD-PARTY-NOTICES.txt\n"
            "Provides-Extra: numpy\nRequires-Dist: numpy>=1.26; extra == \"numpy\"\n"
            "Provides-Extra: torch\nRequires-Dist: numpy>=1.26; extra == \"torch\"\n"
            "Requires-Dist: torch>=2.9; extra == \"torch\"\n"
            "Provides-Extra: dev\nRequires-Dist: build>=1.2; extra == \"dev\"\n"
            "Requires-Dist: pytest>=8; extra == \"dev\"\n"
            "Project-URL: Repository, https://github.com/TLace03/Alelyon-Compute-Kit\n"
            "Description-Content-Type: text/markdown\n\n").encode() + source["README.md"]


def wheel_files(source, library):
    files = {name[4:]: data for name, data in {**source, **library}.items() if name.startswith("src/")}
    files[f"{INFO}/METADATA"] = metadata(source)
    files[f"{INFO}/WHEEL"] = (f"Wheel-Version: 1.0\nGenerator: alelyon-bounded-backend\n"
                               f"Root-Is-Purelib: false\nTag: {TAG}\n").encode()
    for name in ("LICENSE", "THIRD-PARTY-NOTICES.txt"):
        files[f"{INFO}/licenses/{name}"] = source[name]
    files[f"{INFO}/SOURCE_MANIFEST.json"] = source["SOURCE_MANIFEST.json"]
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, data in sorted(files.items()):
        sha = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + sha, len(data)))
    writer.writerow((f"{INFO}/RECORD", "", ""))
    files[f"{INFO}/RECORD"] = record.getvalue().encode()
    return files


def _exclusive(path):
    path = Path(path)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise PackageRefused("output-directory-invalid")
    return path.open("xb")


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    del config_settings, metadata_directory
    source = sources()
    files = wheel_files(source, native(ROOT, source))
    name = f"{NAME}-{VERSION}-{TAG}.whl"
    with _exclusive(Path(wheel_directory) / name) as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative, data in sorted(files.items()):
                info = zipfile.ZipInfo(relative, (2026, 1, 1, 0, 0, 0))
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)
    if sources() != source:
        raise PackageRefused("source-drift-after-wheel")
    return name


def build_sdist(sdist_directory, config_settings=None):
    del config_settings
    source = sources()
    files = {**source, "PKG-INFO": metadata(source)}
    name = f"{NAME}-{VERSION}.tar.gz"
    with _exclusive(Path(sdist_directory) / name) as stream:
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                for relative, data in sorted(files.items()):
                    entry = tarfile.TarInfo(f"{NAME}-{VERSION}/{relative}")
                    entry.size, entry.mode, entry.mtime = len(data), 0o644, 0
                    archive.addfile(entry, io.BytesIO(data))
    if sources() != source:
        raise PackageRefused("source-drift-after-sdist")
    return name


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_sdist(config_settings=None):
    return []
