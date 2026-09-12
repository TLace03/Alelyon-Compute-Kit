"""Explicit Windows native build. Never called by the packaging backend/import.

Requires pinned Rust, a Vulkan loader at execution time, and the MSVC build
toolchain. Compiles committed SPIR-V inputs without a shader compiler/download.
No device is opened here. Native feature measurements are a separate gate.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess

import _build_backend as backend


def build(root, target):
    root, target = Path(root).resolve(), Path(target).resolve()
    source = backend.sources(root)
    destination = root / Path(backend.DLL).parent
    if destination.exists() or target.exists():
        raise backend.PackageRefused("native-output-exists")
    overrides = ("RUSTFLAGS", "RUSTC", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER")
    if (platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}
            or any(os.environ.get(key) for key in overrides)
            or any(value and key.startswith(("CARGO_BUILD_", "CARGO_PROFILE_", "CARGO_TARGET_"))
                   for key, value in os.environ.items())):
        raise backend.PackageRefused("native-build-platform-or-overrides")
    cargo_home = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo")))
    config_roots = [cargo_home, *((root / backend.KIT).parents), root / backend.KIT]
    for candidate in config_roots:
        directory = candidate if candidate == cargo_home else candidate / ".cargo"
        if any((directory / name).exists() for name in ("config", "config.toml")):
            raise backend.PackageRefused("external-cargo-config")
    version = subprocess.run(["rustc", "+" + backend.RUST_VERSION, "--version"],
                             check=True, text=True, capture_output=True).stdout.strip()
    if not version.startswith(f"rustc {backend.RUST_VERSION} "):
        raise backend.PackageRefused("rust-version-mismatch")
    # A fresh target prevents accidentally packaging an earlier sabotaged or
    # debug library. No --features flag and no default features are enabled.
    target.mkdir(parents=True, exist_ok=False)
    subprocess.run(["cargo", "+" + backend.RUST_VERSION, "build", "--locked", "--release",
                    "--lib", "--no-default-features", "--target", "x86_64-pc-windows-msvc",
                    "--target-dir", str(target)], cwd=root / backend.KIT, check=True)
    library = backend.read_file(target, "x86_64-pc-windows-msvc/release/alelyon_compute_kit.dll")
    backend.validate_pe(library)
    if backend.sources(root) != source:
        raise backend.PackageRefused("source-drift-during-native-build")
    receipt = {"schema": 1, "version": backend.VERSION, "target": "x86_64-pc-windows-msvc",
               "profile": "release", "features": [], "rustc": version,
               "source_manifest_sha256": backend.digest(source["SOURCE_MANIFEST.json"]),
               "library_sha256": backend.digest(library), "library_bytes": len(library)}
    destination.mkdir(parents=True, exist_ok=False)
    with (root / backend.DLL).open("xb") as stream:
        stream.write(library)
    with (root / backend.RECEIPT).open("xb") as stream:
        stream.write(backend.canonical(receipt))
    backend.native(root, source)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", required=True, type=Path, help="absent Cargo target directory")
    args = parser.parse_args(argv)
    build(backend.ROOT, args.target_dir)
    print("BUILT: source-bound Windows AMD64 release DLL; device behavior UNMEASURED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
