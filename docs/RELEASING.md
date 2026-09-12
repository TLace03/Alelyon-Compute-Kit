# Releasing alelyon-ai

The `0.1.0a3` candidate adds the experimental Windows AMD64 Vulkan library and
vector-coded expert primitives. Earlier `0.1.0a0` and `0.1.0a1` packages were
MIT-licensed declaration-only releases. The current source and new release use
Apache-2.0 with dependency notices. Publication does not establish universal
device support, complete model training or performance superiority.

## Trusted publisher

Publication uses manual `.github/workflows/release.yml`, with no push, tag,
pull-request or scheduled release trigger. The default `publish` input is false.

| Field | Value |
|---|---|
| PyPI project | `alelyon-ai` |
| GitHub owner | `TLace03` |
| Repository | `Alelyon-Compute-Kit` |
| Workflow | `release.yml` |
| Environment | `pypi` |

The GitHub environment and PyPI trusted-publisher registration are external
configuration. The publish job receives `id-token: write`; it has no source
checkout or compilation step and downloads the exact verified build artifacts.
No persistent PyPI API token is needed.

## Validate before publication

Review the exact main commit, source manifest, license notices and version-bound
metadata. Follow [native packaging](NATIVE_PACKAGING.md) for local artifact
checks. Then run:

```console
gh workflow run release.yml --repo TLace03/Alelyon-Compute-Kit --ref main -f expected_version=0.1.0a3 -f publish=false
```

The four source-test jobs cover Windows and Ubuntu with Python 3.10 and 3.14.
The native build job uses Windows, Python 3.12, locked Cargo dependencies and
the explicitly pinned Rust toolchain. It builds a plain release DLL in a fresh
target, produces one platform wheel and one source-only sdist, verifies exact
archive/source/native identities, runs strict Twine checks and installs the wheel
in a fresh environment outside the checkout. The installed check queries native
ABI/schema without constructing a device. Hosted CI is not actual GPU evidence.

Require successful source/build/clean-install jobs and a skipped publish job.
Review the `verified-distributions` artifact, retained for seven days. Refusals,
missing jobs, unavailable runners or skipped measurements do not become passes.
Source or artifact changes require fresh acceptance.

## Publish and verify

After the dry run passes and the target version is confirmed absent from PyPI:

```console
gh workflow run release.yml --repo TLace03/Alelyon-Compute-Kit --ref main -f expected_version=0.1.0a3 -f publish=true
```

Publication requires both the explicit Boolean input and `refs/heads/main`.
The workflow refuses a source-version mismatch. It never uses `skip-existing`;
immutable duplicate PyPI filenames must fail visibly. Do not delete or replace
a release to disguise a failed run.

After upload, compare PyPI filenames and SHA-256 digests with that run's verified
artifacts. Record the workflow run and exact commit. A green earlier run or a
published project page does not establish acceptance of the current artifact.

## Maintainer changes

Update the project version, package version, build/verifier version pins, fixtures
and release documentation together. Regenerate `SOURCE_MANIFEST.json` through
the reviewed private staging boundary after changing public source. The manifest
describes actual source bytes; do not carry an old digest across an edit.

Workflow actions retain immutable official commit pins. Review release notes,
runtime compatibility and permissions when changing them. Packaging and Rust
pins are explicit in the workflow/build source; changing locked dependencies
also requires updating the reviewed dependency/license boundary. Installation
must continue to refuse missing native artifacts without implicit compilation
or downloads.
