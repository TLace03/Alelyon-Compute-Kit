# Releasing `alelyon-ai`

The `0.1.0a1` release is a declaration-only alpha. It contains the Python
capability and backend registry. It does not contain a native runtime, execute
GPU kernels, train a model, or replace CUDA. Publishing it does not establish
native availability, numerical correctness, performance, or training readiness.

Publication is manual through `.github/workflows/release.yml`. The workflow has
no push, tag, pull-request, or scheduled trigger. Its default run tests and
verifies artifacts without publishing them.

## Trusted publisher configuration

Configure PyPI trusted publishing with this exact tuple:

| Field | Value |
|---|---|
| PyPI project | `alelyon-ai` |
| GitHub owner | `TLace03` |
| GitHub repository | `Alelyon-Compute-Kit` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

Create the GitHub environment named `pypi` before the first publication and
apply the repository's intended environment protection rules. The environment
and PyPI trusted-publisher registration are external configuration. Inspect the
actual release job to verify that its OIDC identity matches this tuple.

Do not create or paste a PyPI API token. The publish job receives only the
`id-token: write` permission needed by PyPI trusted publishing. It has no source
checkout, Python setup, build step, or private import. It downloads the exact
artifact produced after the test and verification jobs.

## Dry run

1. Review the source on the `main` branch. Confirm that `pyproject.toml` and
   `alelyon_compute_kit.__version__` both remain `0.1.0a1`.
2. Open **Actions**, select **Publish Python distribution**, and choose
   **Run workflow** on `main`.
3. Leave `publish` false and set `expected_version` to the version in the
   source. The input has no default: the workflow refuses any value that does
   not equal `alelyon_compute_kit.__version__`, so this states what you mean
   rather than accepting a value somebody wrote once.
4. Require all four source-test matrix jobs and the Ubuntu Python 3.12 build job
   to pass. The publish job must be skipped.
5. Review the uploaded `verified-distributions` artifact. It is retained for
   seven days and must contain exactly one wheel and one source distribution
   for the version under release. The workflow resolves both from `dist/` and
   refuses anything other than exactly one of each, so a stale artifact is a
   failure there rather than an ambiguity here.

The test matrix covers Ubuntu and Windows with Python 3.10 and 3.14. The build
job uses pinned packaging tools, invokes the repository's bounded artifact
verifier with explicit wheel and sdist paths, runs strict `twine check`, and
installs the wheel into a fresh virtual environment outside the checkout using
Python isolated mode. It then runs `pip check` and imports the installed package.

## Publish

After the dry run and trusted-publisher configuration are reviewed, manually
run the same workflow on `main` with:

- `publish`: true
- `expected_version`: `0.1.0a1`

The publish job is gated by the Boolean input and the exact `refs/heads/main`
ref. A run from another branch still executes the test and build evidence but
cannot enter the PyPI environment or publish.

PyPI distribution filenames are immutable. The workflow does not use
`skip-existing`; a duplicate `0.1.0a1` publication must fail visibly. Do not
delete or replace a release to make a rerun appear successful.

## Version changes

For a future release, update and review all version-bound locations together:

- `[project].version` in `pyproject.toml`;
- `alelyon_compute_kit.__version__`;
- the version enforced by `tools/verify_distribution.py`;
- its verifier fixtures;
- this document.

`release.yml` is no longer on that list. It used to hard-compare
`expected_version` against a literal and name each artifact by filename, which
made the input a formality and put a manual edit of the workflow on the
critical path of every release. It now compares the input against the SOURCE
version and resolves the artifacts from `dist/`, so it is version-agnostic.
The verifier's own pin is deliberately kept: it is an independent second
statement of the expected version, and it is what would catch a wrong version
in the source that the workflow's own comparison cannot see.

Run a non-publishing workflow first. A version-input mismatch, failed source
test, artifact-verifier refusal, metadata warning, failed clean install, or
dependency conflict prevents artifact upload to the publish job.

## Pinned workflow dependencies

The workflow references official actions by immutable commit SHA. These refs
were resolved from the corresponding official GitHub repositories on
2026-09-06:

| Action | Reviewed ref | Commit |
|---|---|---|
| `actions/checkout` | `v6` (`v6.1.0`) | `d23441a48e516b6c34aea4fa41551a30e30af803` |
| `actions/setup-python` | `v6` (`v6.3.0`) | `ece7cb06caefa5fff74198d8649806c4678c61a1` |
| `actions/upload-artifact` | `v6` (`v6.0.0`) | `b7c566a772e6b6bfb58ed0dc250532a479d7789f` |
| `actions/download-artifact` | `v7` (`v7.0.0`) | `37930b1c2abaa49bbe596cd826c3c89aef350131` |
| `pypa/gh-action-pypi-publish` | `release/v1` | `dc37677b2e1c63e2034f94d8a5b11f265b73ba33` |

Before changing an action, resolve its new official ref, review its release
notes and permissions, replace the SHA, and update this table in the same
change.

The four JavaScript actions above declare the Node.js 24 runtime. The selected
artifact majors are `upload-artifact` v6 and `download-artifact` v7 because the
official v5 and v6 manifests, respectively, still declare Node.js 20. These
Node.js 24 actions require Actions Runner 2.327.1 or later on a self-hosted
runner. This workflow uses GitHub-hosted runners. The checkout and setup jobs
retain only repository `contents: read`; the PyPI job alone receives
`id-token: write`.

Source review does not establish CI runner availability, installation success
or publication. Inspect each workflow run and its retained artifacts; verify
the published version and file digests on PyPI after the upload completes.
