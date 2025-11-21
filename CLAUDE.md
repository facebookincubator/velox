# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Velox is a composable execution engine distributed as an open source C++ library providing reusable, extensible, and high-performance data processing components. It is NOT meant for end-users directly - it's a library for developers building compute engines.

Velox does NOT provide:
- SQL parser
- Dataframe layer
- Query optimizer

Velox DOES provide:
- Type system supporting scalar, complex, and nested types
- Arrow-compatible columnar memory layout (Vector)
- Vectorized expression evaluation engine
- Vectorized functions (Presto and Spark semantics)
- Relational operators (scans, joins, aggregations, etc.)
- Extensible I/O connectors and file formats (ORC/DWRF, Parquet)
- Resource management primitives

## Build Commands

### Initial Setup

Dependencies are managed via setup scripts. Set `DEPENDENCY_DIR` (default: `deps-download`) and `INSTALL_PREFIX` (default: `deps-install` on macOS, `/usr/local` on Linux):

```bash
# macOS
./scripts/setup-macos.sh

# Ubuntu 20.04+
./scripts/setup-ubuntu.sh

# CentOS 9 Stream
./scripts/setup-centos9.sh
```

For cloud storage adapters (S3, GCS, Azure):
```bash
./scripts/setup-centos9.sh install_adapters  # Also works for macOS/Ubuntu
```

### Build Targets

```bash
make                  # Build release version (same as 'make release')
make debug            # Build with debugging symbols (_build/debug)
make release          # Build optimized version (_build/release)
make unittest         # Build debug + run all tests via ctest
make clean            # Delete all build artifacts

# Specialized builds
make minimal          # Minimal build (fewer components)
make minimal_debug    # Minimal with debug symbols
make dwio             # Minimal + DWIO (file format readers/writers)
```

Build parallelism: Set `NUM_THREADS` (default: number of cores)
```bash
NUM_THREADS=8 make release
```

### CMake Configuration

The Makefile wraps CMake. Key variables:
- `BUILD_DIR`: Build directory name (default: `release` or `debug`)
- `BUILD_TYPE`: CMake build type (`Release` or `Debug`)
- `VELOX_BUILD_TESTING`: Build tests (default: `ON`)
- `VELOX_BUILD_MINIMAL`: Minimal build (default: `OFF`)
- `TREAT_WARNINGS_AS_ERRORS`: Treat warnings as errors (default: `1`)

Pass extra CMake flags:
```bash
make cmake EXTRA_CMAKE_FLAGS="-DVELOX_BUILD_TESTING=OFF"
make build
```

## Testing

### Running Tests

Tests use Google Test (gtest). Built test executables are in `_build/{debug|release}/velox/`.

```bash
# Run all tests
make unittest

# Run all tests with ctest directly (from build directory)
cd _build/debug && ctest -j $(nproc) -VV --output-on-failure

# Run a specific test executable
_build/debug/velox/exec/tests/velox_aggregation_test

# Run specific test(s) with gtest filters
_build/debug/velox/exec/tests/velox_aggregation_test --gtest_filter="AggregationTest.singleKey"
_build/debug/velox/exec/tests/velox_aggregation_test --gtest_filter="AggregationTest.*"

# Run with verbose logging
_build/debug/velox/exec/tests/velox_aggregation_test --logtostderr=1 --minloglevel=0
```

### Fuzzing

Expression fuzzer for testing functions:
```bash
make fuzzertest  # Runs with default seed and duration

# Manual fuzzing
_build/debug/velox/expression/fuzzer/velox_expression_fuzzer_test \
  --only <function-name> \
  --duration_sec 60 \
  --logtostderr=1 \
  --enable_variadic_signatures \
  --velox_fuzzer_enable_complex_types
```

### Test Structure

Tests inherit from test utilities in `velox/exec/tests/utils/`:
- `OperatorTestBase`: Base for operator tests
- `AssertQueryBuilder`: Build and execute queries with assertions
- `PlanBuilder`: Build query plans programmatically
- `HiveConnectorTestBase`: Base for Hive connector tests

## Code Style & Formatting

Velox uses `pre-commit` for automated code quality checks. Install via `pipx` or `uv tool`:

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Run manually on staged files
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run clang-format
pre-commit run clang-tidy --hook-stage=manual  # Requires compile_commands.json
```

Hooks run automatically on `git commit` and `git push`. Use `--no-verify` to skip temporarily (CI will still check).

### Key Style Guidelines

**Naming Conventions:**
- `PascalCase`: Types, classes, structs, enums, file names
- `camelCase`: Functions, variables, non-type template parameters
- `camelCase_`: Private/protected member variables
- `snake_case`: Namespaces, build targets
- `UPPER_SNAKE_CASE`: Macros
- `kPascalCase`: Static constants, enumerators
- `testing` prefix: Test-only methods (e.g., `testingFoo()`)
- `debug` prefix: Query configs for debugging only

**Assertions:**
- Use `VELOX_CHECK_*` for internal errors, `VELOX_USER_CHECK_*` for user errors
- `VELOX_CHECK_LT(idx, size)` preferred over `VELOX_CHECK(idx < size)` (includes values in error)
- `VELOX_FAIL()`, `VELOX_UNREACHABLE()`, `VELOX_NYI()` for specific cases

**Namespaces:**
- All code in `facebook::velox` namespace
- Use nested namespace definition: `namespace facebook::velox::core {`
- Never use `using namespace` in headers
- In .cpp files, use `using` declarations after includes

**Headers:**
- Use `#pragma once` for include guards
- Include full paths relative to repo root
- Forward-declare when possible to minimize dependencies
- Put large implementation blocks in `-inl.h` files

**See `CODING_STYLE.md` for complete guidelines.**

## Architecture

### Component Structure

Velox is organized into logical components:

**Core Type & Data:**
- `velox/type/`: Type system (scalar, complex, nested types)
- `velox/vector/`: Columnar memory layout (Flat, Dictionary, Constant, RLE encodings)
- `velox/buffer/`: Memory buffers and management

**Expression Evaluation:**
- `velox/expression/`: Vectorized expression evaluation engine
- `velox/functions/`: Function implementations
  - `velox/functions/prestosql/`: Presto SQL functions
  - `velox/functions/sparksql/`: Spark SQL functions
  - `velox/functions/lib/`: Shared function utilities

**Query Execution:**
- `velox/core/`: Core execution primitives (PlanNode, QueryCtx, QueryConfig)
- `velox/exec/`: Operator implementations (scans, joins, aggregations, filters, etc.)
- `velox/exec/prefixsort/`: Specialized sorting implementations

**I/O & Connectors:**
- `velox/dwio/`: Data Warehouse I/O
  - `velox/dwio/dwrf/`: DWRF format (ORC variant)
  - `velox/dwio/parquet/`: Parquet format
  - `velox/dwio/nimble/`: Nimble format
- `velox/connectors/`: Extensible connector interface
  - `velox/connectors/hive/`: Hive connector

**Common Utilities:**
- `velox/common/`: Base utilities (exceptions, checks, memory)
- `velox/common/memory/`: Memory management and arenas
- `velox/common/file/`: File system abstractions

**Testing Utilities:**
- `velox/exec/tests/utils/`: Test helpers (AssertQueryBuilder, PlanBuilder, OperatorTestBase)
- `velox/common/base/tests/`: Common test utilities (GTestUtils)
- `velox/dwio/common/tests/utils/`: DWIO test utilities (BatchMaker)

### Key Design Patterns

**Extensibility Points:**
Velox allows custom implementations of:
1. Custom types
2. Simple and vectorized functions
3. Aggregate functions
4. Window functions
5. Operators
6. File formats
7. Storage adapters
8. Network serializers

**Vector Encodings:**
The Vector system supports multiple encodings for efficiency:
- Flat: Standard columnar layout
- Dictionary: Index-based encoding for repeated values
- Constant: Single value repeated
- Lazy: Deferred materialization

**Memory Management:**
Velox uses custom memory management via `MemoryPool` and `MemoryAllocator` for:
- Fine-grained tracking
- Memory arbitration
- Spilling support

## Contributing

### PR Title Format

Follow [conventional commits](https://www.conventionalcommits.org/):

```
<type>[(optional scope)]: <description>
```

**Types:** `feat`, `fix`, `perf`, `build`, `test`, `docs`, `refactor`, `misc`

**Scopes:** `vector`, `type`, `expr`, `operator`, `memory`, `dwio`, `parquet`, `dwrf`, `filesystem`, `connector`, `hive`, `function`, `aggregate`, etc.

**Examples:**
- `feat(type): Add IPPREFIX`
- `fix(expr): Prevent unnecessary flatmap to map conversion`
- `perf(aggregate): Optimize hash table resizing`

Add `!` for breaking changes: `fix(expr)!: Change signature of evalSimplified`

### PR Body Requirements

- Summary focusing on *what* and *why* (not *how*)
- Wrap lines at 80 characters
- Add `BREAKING CHANGE:` footer for API changes
- Link issues: `Fixes #1234` (auto-closes) or `Part of #1234` (links only)

### CI Requirements

- Do NOT ignore red CI signals, even if they seem preexisting
- If a test fails unrelated to your change:
  - Search for existing issue with test name in title
  - Add comment with link to your failed CI job
  - If no issue exists, create "Broken CI \<test_name\>" issue and tag component maintainers
- Tag reviewers ONLY after all CI signals are green

### Adding Functions

1. Read https://facebookincubator.github.io/velox/develop/scalar-functions.html
2. Use PR title: `Add xxx [Presto|Spark] function`
3. Link to official Presto/Spark documentation in PR
4. Use Presto/Spark to verify semantics (Spark 3.5 with ANSI OFF)
5. Add comprehensive tests covering all signatures and edge cases
6. Add documentation to `velox/docs/functions/*.rst` (alphabetically)
7. Run fuzzer on the new function (see Testing section above)

**Example PRs:**
- #1000: Add sha256 Presto function
- #313: Add sin, cos, tan, cosh and tanh Presto functions

## Documentation

**CRITICAL: Documentation updates are REQUIRED for:**
- **New features** (type=`feat`)
- **Breaking changes** (marked with `!` in PR title)
- **New functions** (Presto/Spark functions)
- **New operators or significant behavior changes**
- **API changes that affect library users**

### Documentation Structure

Documentation is written in ReStructuredText (.rst) format and built with Sphinx:

```
velox/docs/
├── develop/                    # Developer guides
│   ├── scalar-functions.rst   # How to add functions
│   ├── aggregate-functions.rst
│   ├── vectors.rst
│   ├── expression-evaluation.rst
│   ├── joins.rst
│   └── ...
├── functions/                  # Function reference documentation
│   ├── presto/                # Presto function docs
│   │   ├── string.rst
│   │   ├── array.rst
│   │   ├── map.rst
│   │   └── ...
│   └── spark/                 # Spark function docs
│       ├── string.rst
│       ├── array.rst
│       └── ...
├── bindings/python/           # Python API docs
├── monitoring.rst             # Metrics and monitoring
├── configs.rst                # Configuration options
└── index.rst                  # Main documentation index
```

### When Adding Functions

Function documentation is **MANDATORY** when adding new functions:

1. Add function documentation to the appropriate .rst file:
   - Presto functions: `velox/docs/functions/presto/<category>.rst`
   - Spark functions: `velox/docs/functions/spark/<category>.rst`

2. Functions must be listed in **alphabetical order** within their category

3. Use Sphinx directive format:
   ```rst
   .. function:: function_name(arg1, arg2) -> return_type

       Brief description of what the function does.

       Optional detailed explanation, edge cases, examples.

       Example: ::

           SELECT function_name('example', 123); -- result
   ```

4. Document all signatures if the function is overloaded

5. Include any important notes about behavior, NULL handling, or edge cases

### When Adding Features or Breaking Changes

1. **Update existing documentation** if the feature modifies existing behavior:
   - Update relevant guide in `velox/docs/develop/`
   - Update affected sections in other documentation

2. **Create new documentation** for new components:
   - Add new .rst file in appropriate directory
   - Reference it from `index.rst` or parent documentation

3. **Document breaking changes** explicitly:
   - Describe what changed and why
   - Provide migration guidance
   - Update API documentation with new signatures

### Building Documentation Locally

```bash
# Install documentation dependencies
uv sync --extra docs

# Navigate to docs directory
cd velox/docs

# Clean previous builds
make clean

# Build HTML documentation
make html

# View built documentation
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

The documentation will be built in `velox/docs/_build/html/`.

### Documentation CI

- Documentation builds automatically on PR creation if `velox/docs/**` files are modified
- CI validates that documentation builds without errors
- Merged documentation is automatically published to https://facebookincubator.github.io/velox

### Documentation Best Practices

1. **Be concise but complete**: Explain what, why, and when to use features
2. **Include examples**: Code examples help users understand usage
3. **Document edge cases**: NULL handling, empty inputs, overflow behavior, etc.
4. **Use consistent terminology**: Match terminology used in code and other docs
5. **Cross-reference**: Link to related functions, concepts, or guides using Sphinx references
6. **Keep it updated**: Documentation that falls behind code is worse than no documentation

### Checklist for PRs with Documentation

Before submitting a PR that adds features or breaking changes:

- [ ] All new functions documented in appropriate .rst files (alphabetically ordered)
- [ ] Breaking changes explained with migration guidance
- [ ] New configuration options added to `configs.rst`
- [ ] Developer guides updated if adding new patterns or utilities
- [ ] Documentation builds successfully locally (`make html` in `velox/docs/`)
- [ ] Examples provided for new functionality
- [ ] Edge cases and special behaviors documented

## Python Bindings

Build and test Python bindings:

```bash
make python-build  # Creates venv and builds PyVelox
make python-test   # Runs Python unit tests

# Manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install pyarrow scikit_build_core setuptools_scm[toml]
pip install --no-build-isolation -Ccmake.build-type=Debug -v .
```

## Compiler Requirements

**Minimum versions:**
- GCC 11 (Linux)
- Clang 15 (Linux/macOS)

**Required CPU instruction sets:**
- bmi, bmi2, f16c
- avx, avx2, sse (Intel) or Neon/Neon64 (ARM)

**Using Clang on Linux:**
```bash
export USE_CLANG=true
./scripts/setup-ubuntu.sh  # or setup-centos9.sh
export CC=/usr/bin/clang-15
export CXX=/usr/bin/clang++-15
make
```
