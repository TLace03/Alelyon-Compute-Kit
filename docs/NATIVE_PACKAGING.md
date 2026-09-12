# Experimental native package boundary

Version 0.1.0a2 supplies a Windows AMD64 native library through the explicit
`alelyon_compute_kit.ack` API. Root imports retain the capability declaration and
registry API and do not load a DLL, discover devices, import NumPy/Torch, or
register hardware. Install the `numpy` extra to use `vq` and `expert_bank`.
The `torch` extra supplies the optional Python composition dependencies; it does
not include a PrivateUse1 extension or claim end-to-end training coverage.

The wheel carries the reviewed Rust/Cargo/SPIR-V/GLSL source payload and its
normalized source digests. `packed2_ops` is a Rust planner; its presence does not
declare an exported packed2 C operator. VQ schema 1 is an optional explicit C
entry point. Grouped 4-bit codebook indices do not establish model quality,
training throughput, or a trained trillion-parameter model.

This repository is a generated public boundary. Native source changes originate
in the reviewed source exporter; `SOURCE_MANIFEST.json` binds each staged source
file. Hashes establish byte identity, not independent producer authentication.
The upstream shader manifest is projected to shipped modules and each module's
exact GLSL/SPIR-V input digests are checked before staging. No shader compiler or
private application Runtime is needed to build this source tree.

The source distribution intentionally contains no DLL. The stdlib-only PEP 517
backend refuses wheel creation without a separate native build receipt. It never
invokes Cargo, a compiler, a driver, or a network client. On unsupported platforms
an sdist fallback refuses rather than compiling implicitly.

On Windows AMD64 with MSVC and Rust 1.97.1 installed, use fresh output paths:

```powershell
python tools/build_native.py --target-dir C:/temp/ack-native-new
python -m build --no-isolation --sdist --wheel --outdir dist
python tools/verify_distribution.py --wheel dist/alelyon_ai-0.1.0a2-py3-none-win_amd64.whl --sdist dist/alelyon_ai-0.1.0a2.tar.gz
python -m twine check --strict dist/alelyon_ai-0.1.0a2-py3-none-win_amd64.whl dist/alelyon_ai-0.1.0a2.tar.gz
```

Use both `--sdist --wheel`: the frontend's default wheel-from-sdist rebuild has
no separately built DLL and therefore refuses. Build failures leave incomplete
outputs for inspection; use a new stage/target instead of overwriting them.
The native receipt binds the exact source manifest, pinned compiler, target,
plain release feature selection, DLL bytes and SHA-256. It is declared build
provenance. The archive verifier checks the complete file sets, source bytes,
metadata, wheel RECORD, native receipt and PE target without executing artifacts.

The release workflow first tests declaration APIs on Linux/Windows, then builds
and checks the Windows wheel. A fresh virtual environment outside the checkout
checks inert import plus ABI/schema identity without constructing a device.
Real bounded device operations and native memory/refusal checks require separate
hardware evidence. A workflow success alone is not training readiness or a claim
of support for every GPU. The publisher receives only the verified artifacts,
uses the `pypi` environment, and publishes only on explicit dispatch from `main`.
