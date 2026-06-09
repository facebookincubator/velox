# Upgrading Dependencies in CI/CD Pipelines

CI jobs run inside pre-built Docker images from `ghcr.io/facebookincubator/velox-dev:*`.
These images are rebuilt by `.github/workflows/docker.yml` whenever setup scripts or
Dockerfiles change on `main`.

If a dependency upgrade introduces breaking API changes, the code on `main` won't compile
against the new version until you land the fix. To avoid blocking other contributors' CI
during this window, Velox uses **staging images** — a second set of Docker images that
carry the new dependency while production images stay on the old version.

## How staging works

Each Docker image target in `docker-bake.hcl` has a corresponding `-staging` target
(e.g., `adapters-staging-amd64`). Staging targets pass `VELOX_STAGING=true` as a build
arg. When this arg is set, `scripts/setup-versions.sh` sources
`scripts/setup-staging-versions.sh` at the end, overriding whichever version variables
are listed there. Production targets don't set the arg, so they ignore the file.

To check what a staging build will produce:

```bash
# Print the resolved staging build config (no actual build)
docker buildx bake adapters-staging-amd64 --print

# Compare against production
docker buildx bake adapters-amd64 --print
```

You can also test locally without Docker:

```bash
# Production versions
source scripts/setup-versions.sh
echo $DUCKDB_VERSION   # v0.8.1

# Staging versions
VELOX_STAGING=true source scripts/setup-versions.sh
echo $DUCKDB_VERSION   # v1.4.4
```

## Upgrade process

The upgrade is split across three PRs so that production CI is never broken.

### PR1 — Stage the new version

Add the new version to `scripts/setup-staging-versions.sh`:

```bash
# scripts/setup-staging-versions.sh
DUCKDB_VERSION="v1.4.4"
DUCKDB_GIT_COMMIT_HASH="6ddac80"
```

If the install function in `scripts/setup-common.sh` has hardcoded values that differ
between versions (commit hashes, cmake flags), extract them into variables in
`scripts/setup-versions.sh` so the staging file can override them.

When this PR merges, `docker.yml` rebuilds all images. Production images still use the
old version from `setup-versions.sh`. Staging images pick up the new version from
`setup-staging-versions.sh`.

### PR2 — Code changes against staging images

Point CI at the staging images and make the code work:

1. In `.github/workflows/linux-build-base.yml`, change the container tags:

   ```yaml
   # before
   container: ghcr.io/facebookincubator/velox-dev:adapters
   # after
   container: ghcr.io/facebookincubator/velox-dev:adapters-staging
   ```

2. Update `CMake/resolve_dependency_modules/<dep>.cmake` (version, SHA256, patches).
3. Fix C++ API compatibility issues and update tests.

This PR can take as long as it needs — production CI is unaffected.

### PR3 — Promote to production

Move the version into production and clean up:

1. In `scripts/setup-versions.sh`, bump the version and any related variables to match
   what's in the staging file.
2. Clear `scripts/setup-staging-versions.sh` (remove all overrides, keep the header).
3. Revert `container:` references in `linux-build-base.yml` back to standard tags
   (e.g., `:adapters`).

When this merges, production images rebuild with the new dependency, and CI points at
them directly.

## Key files

- `scripts/setup-versions.sh` — production dependency versions
- `scripts/setup-staging-versions.sh` — staging overrides (empty when no upgrade is active)
- `scripts/setup-common.sh` — `install_*` functions that use the version variables
- `docker-bake.hcl` — staging targets with `args = { VELOX_STAGING = "true" }`
- `.github/workflows/docker.yml` — builds and pushes all images on merge to `main`
- `.github/workflows/linux-build-base.yml` — `container:` references used by CI jobs
- `CMake/resolve_dependency_modules/<dep>.cmake` — FetchContent version for bundled and macOS builds
