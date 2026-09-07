# Releasing `alelyon-ai`

The `0.1.0a0` release is a declaration-only alpha. It contains the Python
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
   `alelyon_compute_kit.__version__` both remain `0.1.0a0`.
2. Open **Actions**, select **Publish Python distribution**, and choose
   **Run workflow** on `main`.
3. Leave `publish` false and set `expected_version` to `0.1.0a0`.
4. Require all four source-test matrix jobs and the Ubuntu Python 3.12 build job
   to pass. The publish job must be skipped.
5. Review the uploaded `verified-distributions` artifact. It is retained for
   seven days and must contain exactly:
   - `alelyon_ai-0.1.0a0-py3-none-any.whl`
   - `alelyon_ai-0.1.0a0.tar.gz`

The test matrix covers Ubuntu and Windows with Python 3.10 and 3.14. The build
job uses pinned packaging tools, invokes the repository's bounded artifact
verifier with explicit wheel and sdist paths, runs strict `twine check`, and
installs the wheel into a fresh virtual environment outside the checkout using
Python isolated mode. It then runs `pip check` and imports the installed package.

## Publish

After the dry run and trusted-publisher configuration are reviewed, manually
run the same workflow on `main` with:

- `publish`: true
- `expected_version`: `0.1.0a0`

The publish job is gated by the Boolean input and the exact `refs/heads/main`
ref. A run from another branch still executes the test and build evidence but
cannot enter the PyPI environment or publish.

PyPI distribution filenames are immutable. The workflow does not use
`skip-existing`; a duplicate `0.1.0a0` publication must fail visibly. Do not
delete or replace a release to make a rerun appear successful.

## Version changes

For a future release, update and review all version-bound locations together:

- `[project].version` in `pyproject.toml`;
- `alelyon_compute_kit.__version__`;
- the version and archive names enforced by `tools/verify_distribution.py`;
- its verifier fixtures;
- `expected_version` and artifact paths in `release.yml`;
- this document.

Run a non-publishing workflow first. A version-input mismatch, failed source
test, artifact-verifier refusal, metadata warning, failed clean install, or
dependency conflict prevents artifact upload to the publish job.

## Pinned workflow dependencies

The workflow references official actions by immutable commit SHA. These refs
were resolved from the corresponding official GitHub repositories on
2026-09-06:

| Action | Reviewed ref | Commit |
|---|---|---|
| `actions/checkout` | `v4` | `11d5960a326750d5838078e36cf38b85af677262` |
| `actions/setup-python` | `v5` | `a26af69be951a213d495a4c3e4e4022e16d87065` |
| `actions/upload-artifact` | `v4` | `ea165f8d65b6e75b540449e92b4886f43607fa02` |
| `actions/download-artifact` | `v4` | `d3f86a106a0bac45b974a628896c90dbdf5c8093` |
| `pypa/gh-action-pypi-publish` | `release/v1` | `dc37677b2e1c63e2034f94d8a5b11f265b73ba33` |

Before changing an action, resolve its new official ref, review its release
notes and permissions, replace the SHA, and update this table in the same
change.

Source review does not establish CI runner availability, installation success
or publication. Inspect each workflow run and its retained artifacts; verify
the published version and file digests on PyPI after the upload completes.
