# CI Workflows

Velox CI validates builds and tests across Linux (GCC and optionally Clang) and macOS (Apple Clang). Linux workflows run full unit test suites in Docker containers on 32-core runners; macOS workflows verify compilation only. Fuzzer workflows stress-test functions and operators with randomized inputs.

### Why Docker containers?

Velox has a [large dependency footprint](../../CMake/resolve_dependency_modules/README.md) — over 20 third-party libraries. Installing and version-pinning these on bare runners would be fragile and slow. Instead, the `docker.yml` workflow builds pre-configured Docker images (`ghcr.io/facebookincubator/velox-dev`) with all dependencies pre-installed. CI build jobs run inside these containers, ensuring reproducible environments and fast startup. The images come in several variants:

- **`centos9`** — Base CentOS Stream 9 environment used by fuzzer workflows
- **`adapters`** — Full environment with cloud storage adapters (S3, GCS, ABFS, HDFS) and GPU support
- **`ubuntu`** — Standard Ubuntu 22.04 build environment
- **`fedora`** — Fedora environment for testing against different system package versions (e.g., system-provided gRPC, Arrow, Thrift)
- **`presto-java`** / **`spark-server`** — Images with Presto/Spark servers for fuzzer reference result verification

The `centos9` and `adapters` images are multi-architecture (amd64 + arm64); the `ubuntu` and `fedora` images are amd64-only. Images are rebuilt automatically when Dockerfiles or setup scripts change on `main`. On macOS, dependencies are installed directly via `setup-macos.sh` using Homebrew since Docker is not used.

For current build times and performance trends, see the [CI performance metrics](https://github.com/facebookincubator/velox/actions/metrics/performance).

| Workflow | File | Triggers | Purpose |
|----------|------|----------|---------|
| Linux Build using GCC | `linux-build.yml` + `linux-build-base.yml` | push to main, PRs | Main build & test (3 configs) |
| macOS Build | `macos.yml` | push, PRs | Compilation check (debug + release) |
| Breeze Linux Build | `breeze.yml` | push to main, PRs | Tracing module with sanitizers |
| Fuzzer Jobs | `scheduled.yml` | PRs, push to main, daily cron, manual | Randomized correctness testing |
| Run Checks | `preliminary_checks.yml` | PRs | Formatting, linting, PR title |
| Build Impact Analysis | `build-impact.yml` + `build-impact-comment.yml` | push to main, PRs | CMake dependency impact |
| CI Failure Comment | `ci-failure-comment.yml` | workflow_run (on Linux Build / Fuzzer failure) | AI-powered failure analysis on PRs |
| Claude PR Assistants | `claude.yml` + `claude-review.yml` | PR comments (`@claude`) | AI code review |
| Build Pyvelox Wheels | `build_pyvelox.yml` | manual | Python wheel packaging |
| Docker Images | `docker.yml` | push to main, manual | Multi-arch Docker images |
| Weekly Date Tag | `tag.yml` | weekly cron, manual | Version tagging |
| Update Documentation | `docs.yml` | push (docs changes), PRs | Sphinx docs + GitHub Pages |
| Collect Build Metrics | `build-metrics.yml` | PRs, daily, manual | Binary size tracking (disabled) |
| Ubuntu Bundled Deps | `ubuntu-bundled-deps.yml` | nightly, manual, PRs (dep scripts) | Build-from-source validation |

## Core Build & Test

### Linux Build using GCC (`linux-build.yml` + `linux-build-base.yml`)

The main CI workflow for Velox. Triggered on pushes to `main` and on pull requests when relevant files change (source code, CMake files, setup scripts, or the workflow files themselves). The entry point `linux-build.yml` delegates to the reusable `linux-build-base.yml` template, which builds and tests three configurations in parallel on 32-core Ubuntu runners:

- **Linux adapters release** — Release build using the `velox-dev:adapters` Docker image. Enables cloud storage adapters (S3, GCS, ABFS, HDFS), Parquet, Arrow, geospatial functions, and GPU support (WAVE, cuDF). Tests run with `ctest -j 24` and a 900-second timeout. When cuDF-related files change, a separate cuDF test job runs on a `4-core-ubuntu-gpu-t4` GPU runner.
- **Ubuntu debug** — Debug build using the `velox-dev:ubuntu-22.04` Docker image. Enables benchmarks, examples, Arrow, geospatial, Parquet, shared library (`VELOX_BUILD_SHARED=ON`), and mono library modes. Tests run with `ctest -j 24` and a 1800-second timeout.
- **Fedora debug** — Debug build using the `velox-dev:fedora` Docker image. Validates compilation compatibility with Fedora's system packages including system-provided gRPC and Arrow/Thrift shared libraries. This configuration focuses on compiler and OS compatibility rather than test coverage — it does not have a separate test status job.

All configurations use ccache for build acceleration (persisted via Apache infrastructure stash). The adapters and ubuntu-debug configurations include a flaky test retry mechanism: if any tests fail on the first run, they are automatically retried with `ctest --rerun-failed`. If the retry passes, the tests are marked as flaky; if it fails again, the specific failed test case names are extracted and reported.

#### BUILD / TEST status jobs and `continue-on-error`

Each configuration has separate BUILD and TEST status jobs that appear as independent checks on PRs. This design solves a specific problem: developers need to immediately see whether a CI failure is a build failure (absolute blocker) or a test failure (could be flaky, needs investigation).

The obvious approach — splitting build and test into separate GitHub Actions jobs — is impractical for Velox because the build artifacts are too large to transfer between jobs. Test binaries are ~3GB each (8 grouped binaries totaling ~24GB), plus the shared `libvelox.so` mono library. We tried multiple compression strategies in PR #16938 (tar+gzip, pigz, zstd, direct upload) and all either exceeded the test runtime or were incompatible with the container environment. Stripping debug symbols would defeat the purpose of a debug build. The sequential overhead of packaging, uploading, downloading, and extracting artifacts negates any parallelism gains.

Instead, the test step uses `continue-on-error: true` so the main job always completes, and lightweight status jobs read the step outcomes to provide separate pass/fail signals. The main job (which contains the full test output) appears green even when tests fail, but the status jobs compensate in two ways: (1) failed test names, gtest case names, and failure details (assertion messages, compiler errors) are forwarded to the status jobs via job output variables and displayed directly in the status job logs and as `::error::` annotations, so developers clicking the red check see the specific failures immediately; (2) the `ci-failure-comment.yml` workflow uses Claude to analyze the full logs and post a rich diagnostic comment directly on the PR.

Status jobs handle cancelled runs gracefully — when a job is cancelled (e.g., superseded by a newer push), status jobs exit cleanly without false "build failed" reports.

### macOS Build (`macos.yml`)

Builds Velox on macOS 15 (ARM64/Apple Silicon) with both debug and release configurations. Triggered on pushes to any branch and on pull requests when relevant files change. Uses the Ninja build generator and ccache for faster builds. Tests are currently disabled on macOS — the workflow focuses on ensuring compilation compatibility with Apple's toolchain rather than full test coverage. Dependencies are installed via the `setup-macos.sh` script.

### Breeze Linux Build (`breeze.yml`)

Experimental workflow for the Breeze tracing/profiling module and Perfetto integration. Builds two configurations: a debug build with Address Sanitizer (ASAN) and Undefined Behavior Sanitizer (UBSAN) enabled for memory safety testing, and a RelWithDebInfo build with CUDA support for the Breeze module. Unlike the main Linux build, Breeze uses `VELOX_DEPENDENCY_SOURCE=BUNDLED` to build dependencies from source rather than using system packages. Tests run with `ctest -j 8` under sanitizers.

## Fuzzing (`scheduled.yml`)

A comprehensive fuzzing suite that tests Velox functions and operators against random inputs to catch correctness issues. Triggered on pull requests, pushes to `main`, a daily cron schedule (`0 3 * * *` UTC), and manual `workflow_dispatch`. The daily and main-push triggers are particularly important as they catch regressions that may not be exercised by PR-level test coverage.

The workflow first compiles all fuzzer binaries in a shared `compile` job, then runs 12+ independent fuzzer targets in parallel:

- **Presto Fuzzer** and **Spark Fuzzer** — Test Presto and Spark SQL function implementations with random inputs and verify results against reference implementations (DuckDB for Presto, Spark for Spark functions).
- **Aggregation Fuzzers** — Test aggregate functions with random grouping keys and inputs, with Presto as source of truth.
- **Join Fuzzer** — Tests hash join, merge join, and nested loop join operators with random schemas and data.
- **Exchange Fuzzer** — Tests the data exchange/shuffle operator.
- **Window Fuzzer** — Tests window function implementations.
- **Writer Fuzzer** — Tests file format writers (Parquet, DWRF).
- **RowNumber and TopNRowNumber Fuzzers** — Test row numbering operators.
- **Table Evolution Fuzzer** — Tests schema evolution scenarios.
- **Memory Arbitration Fuzzer** — Tests memory management under pressure.
- **Spatial Join Fuzzer** — Tests geospatial join operations.
- **Cache Fuzzer** — Tests caching infrastructure.

The workflow also includes bias fuzzers that focus specifically on newly added or recently updated functions, and a signature check job that validates function signatures match expected interfaces.

## PR Checks & Feedback

### Run Checks (`preliminary_checks.yml`)

Runs early validation on pull requests before the heavier build workflows. Executes `pre-commit run --all-files` to check code formatting (clang-format), linting (yamllint, zizmor), license headers, and other code quality rules. Also validates the PR title against the conventional commits format (`type(scope): description`), which is required for all PRs.

### Build Impact Analysis (`build-impact.yml` + `build-impact-comment.yml`)

Analyzes which build targets are affected by the files changed in a PR using the CMake dependency graph. On pushes to `main`, the workflow generates and caches a fresh dependency graph. On PRs, it uses a two-path strategy: a fast path that reuses the cached graph from `main`, and a slow path that regenerates the graph when CMake files change. The companion `build-impact-comment.yml` workflow posts the analysis results as a PR comment, updating the existing comment if one already exists.

### CI Failure Comment (`ci-failure-comment.yml`)

Analyzes CI failures and posts diagnostic comments on PRs. Triggered via `workflow_run` when the "Linux Build using GCC" or "Fuzzer Jobs" workflow completes with a failure. The workflow finds the associated PR, downloads failure metadata artifacts uploaded by the status jobs in `linux-build-base.yml` (or, for fuzzer failures, queries the GitHub API for failed job names), then uses Claude to analyze the failures. Claude fetches the full job logs via the GitHub API, reads the PR diff, searches open issues for known failures, and checks recent main branch CI runs for pre-existing flaky tests. The resulting PR comment includes specific failure details (assertion errors, compiler diagnostics), correlation analysis with the PR changes, links to known issues, and recommended fixes. Also supports manual triggering via `workflow_dispatch` with a run ID and PR number.

Uses the `workflow_run` pattern because fork PRs have read-only tokens and cannot post comments directly.

### Claude PR Assistants (`claude.yml` + `claude-review.yml`)

Two AI-powered code review workflows. The newer `claude.yml` uses a skill-based approach triggered by `@claude` mentions in PR comments, restricted to an authorized user allowlist. The legacy `claude-review.yml` supports `/claude-review` (full code review) and `/claude-query` (targeted questions) commands, parsing PR diffs and generating detailed review feedback with file-by-file analysis, risk assessment, and testing recommendations.

## Packaging & Release

### Build Pyvelox Wheels (`build_pyvelox.yml`)

Builds Python wheel packages for PyVelox across multiple Python versions (3.10–3.13) and platforms (Ubuntu, macOS Intel, macOS ARM). Uses `cibuildwheel` for cross-platform wheel building. Triggered via `workflow_dispatch` and on PRs that modify the workflow file. Optionally publishes to PyPI on success.

### Docker Images (`docker.yml`)

Builds and pushes the Docker images described in the [Why Docker containers?](#why-docker-containers) section above. Uses Docker Buildx with BuildKit for multi-arch support (amd64 + arm64) and registry-based layer caching. Triggered on pushes to `main` when Dockerfiles or setup scripts change, and on manual dispatch.

### Weekly Date Tag (`tag.yml`)

Automatically creates weekly date-based version tags (e.g., `v2026.04.03.00`) every Friday at 09:23 UTC. Checks CI status before tagging to ensure only successful builds are released. Also supports manual triggering via `workflow_dispatch`.

## Documentation & Metrics

### Update Documentation (`docs.yml`)

Builds Sphinx documentation when files under `velox/docs/` change on pushes or pull requests. Publishes to GitHub Pages only on pushes to the official `facebookincubator/velox` repository. Includes PyVelox Python API documentation. On pull requests, validates that documentation builds without errors but does not deploy.

### Collect Build Metrics (`build-metrics.yml`)

Measures binary sizes across four build configurations (debug/release x shared/static) and was designed to upload metrics to conbench for performance regression tracking. Currently disabled because the conbench service is unavailable. Triggered on PRs, daily schedule, and manual dispatch.

## Dependency Testing

### Ubuntu Bundled Dependencies (`ubuntu-bundled-deps.yml`)

Tests that Velox can be built entirely from source on a plain Ubuntu system without pre-installed dependencies (except ICU). Uses `VELOX_DEPENDENCY_SOURCE=BUNDLED` to build all dependencies from source, validating the bundled dependency resolution scripts. Runs on a 16-core runner (no Docker container) with 16 build threads. Includes a comprehensive set of feature flags: benchmarks, examples, Arrow, geospatial, Parquet, mono library, shared library, FAISS, and remote functions. Tests run with `ctest -j 8`. Runs nightly at 5 AM UTC, on manual dispatch, and on PRs that modify dependency scripts. Only runs on the official `facebookincubator/velox` repository, not forks.

## Architecture Notes

### Fork PR Permissions

GitHub restricts fork PR tokens to read-only for security. Workflows that need to post PR comments use the **`workflow_run` pattern**: the main workflow uploads results as artifacts, and a separate `workflow_run`-triggered workflow downloads them and posts comments using the base repo's write permissions.

This pattern is used by:
- `build-impact-comment.yml` (posts build impact analysis)
- `ci-failure-comment.yml` (posts CI failure analysis)

### Build Caching

All build workflows use ccache for compiler output caching, persisted across runs using Apache infrastructure stash actions. Cache keys include the platform, build type, and compiler to avoid cache collisions.

### Concurrency

Most workflows use concurrency groups keyed on `workflow + repository + branch/SHA` to automatically cancel in-progress runs when new commits are pushed to the same branch, avoiding wasted CI resources.
