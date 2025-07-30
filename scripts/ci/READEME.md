This README provides an overview of the various CI helpers in this directory.

## CMake Rules and Linting

Velox uses ast-grep for CMake linting with the following rules:

### Core Rules
- `avoid-directory-wide-functions`: :warning: Encourages using target-specific functions
- `cmake-minimum-version-error`: :x: Ensures minimum CMake version of 3.10
- `deprecated-commands`: :x: Prevents use of outdated CMake commands
- `no-hardcoded-absolute-paths`: :x: Prevents hard-coding paths for portability
- `use-find-package`: :x: Enforces use of find_package() over manual library detection
- `variable-instead-of-targets`: :warning: Discourages variables in place of target names

### Test Files
The `scripts/ci/cmake-rules-test/` directory contains test cases for each rule with valid and invalid examples.

The actual linting is done through a pre-commit hook. To add or update existing rules:

- Install ast-grep via `cargo install --locked ast-grep` or `uv tool install ast-grep-cli`
- Install the tree-sitter cli via `cargo install --locked tree-sitter-cli`
- Build the [tree-sitter CMake parser](https://github.com/uyha/tree-sitter-cmake)
  with `tree-sitter build tree-sitter-cmake/`
- `ln -s path/to/cmake.so scripts/ci/cmake.so`
- Modify rules **and** tests
- Run `ast-grep test` in `scripts/ci/` to validate your changes

## Benchmark Tools

Velox includes several tools for benchmarking and performance analysis:

### Benchmark Runner (`benchmark-runner.py`)
- Runs benchmarks and collects performance metrics
- Compares benchmark results between baseline and contender builds
- Uploads results to Conbench for tracking

### Build Metrics (`bm-report/build-metrics.py`)
- Tracks build artifact sizes and compile times
- Analyzes ninja build logs for performance insights
- Generates interactive reports with compile/link time analytics

### Benchmark Alerts (`benchmark-alert.py`)
- Analyzes benchmark results and detects regressions
- Posts GitHub check results and PR comments
- Uses z-score thresholds to identify significant performance changes

## Signature Testing

The `signature.py` tool manages API compatibility checking:

- Exports function signatures for Velox components
- Compares signatures between versions to detect breaking changes
- Creates bias functions for fuzzing-based compatibility tests
- Integrates with GitHub Actions for automated compatibility checking
