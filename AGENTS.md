# AGENTS.md - Velox

## Project Overview

Velox is a composable, high-performance C++ execution engine for data management systems. It provides reusable data processing components covering batch, interactive, stream processing, and AI/ML workloads. Velox takes a **fully optimized query plan as input** and performs the described computation -- it does not provide a SQL parser, dataframe layer, or query optimizer.


**Open source**: Apache 2.0 license, https://github.com/facebookincubator/velox

## Language and Build System

Velox is a **C++20 project** built with **Buck2** internally and **CMake** for open-source builds.

## Build Commands

**macOS:**
```bash
./scripts/setup-macos.sh
make
```

**Ubuntu (20.04+):**
```bash
./scripts/setup-ubuntu.sh
make
```

**CentOS 9:**
```bash
./scripts/setup-centos9.sh
make
```

**With Cloud Storage Adapters (S3, GCS, ABFS):**
```bash
./scripts/setup-<platform>.sh
./scripts/setup-<platform>.sh install_adapters  # or install_aws, install_gcs, install_abfs
make
```

### Build Targets

- `make` or `make release` - Build optimized release version
- `make debug` - Build with debugging symbols
- `make unittest` - Build debug and run all tests with ctest
- `make minimal` - Minimal build (expression evaluation + Presto functions only)
- `make dwio` - Minimal build with file format support (ORC/DWRF, Parquet)
- `make clean` - Delete all build artifacts
- Build outputs: `_build/release/` or `_build/debug/`

### Running Tests

**All tests:**
```bash
make unittest  # Builds debug and runs all tests
# OR
cd _build/debug && ctest -j $(nproc) -VV --output-on-failure
```

**Single test:**
```bash
cd _build/debug
ctest -R <test_name> -VV --output-on-failure
# OR run the test binary directly:
./velox/<component>/tests/<test_binary_name>
```

**Fuzzer test:**
```bash
make fuzzertest  # Uses fixed seed for reproducibility
# OR with custom parameters:
_build/debug/velox/expression/fuzzer/velox_expression_fuzzer_test \
  --seed 123456 --duration_sec 60 --logtostderr=1
```

### Environment Variables

- `DEPENDENCY_DIR` - Where to download dependencies (default: deps-download)
- `INSTALL_PREFIX` - Where to install dependencies (default: deps-install on macOS, /usr/local on Linux)
- `BUILD_THREADS` - Parallelism for dependency builds (default: number of cores)
- `NUM_THREADS` - Parallelism for Velox build (default: number of processors)
- `CPU_TARGET` - Target CPU architecture, e.g., "avx", "sse" (default: "avx")
- `TREAT_WARNINGS_AS_ERRORS` - Set to 0 to disable -Werror (default: 1)
- `USE_CLANG` - Set to true before setup script to use Clang 15 instead of GCC on Linux

**Important:** Set `INSTALL_PREFIX` in your shell config (e.g., `~/.zshrc`) to reuse dependencies across builds.

### CMake Build Options

Key CMake flags that can be passed via `EXTRA_CMAKE_FLAGS`:

- `-DVELOX_BUILD_TESTING=OFF` - Disable tests
- `-DVELOX_BUILD_MINIMAL=ON` - Minimal build
- `-DVELOX_ENABLE_S3=ON` - Enable S3 connector
- `-DVELOX_ENABLE_GCS=ON` - Enable GCS connector
- `-DVELOX_ENABLE_HDFS=ON` - Enable HDFS connector
- `-DVELOX_ENABLE_PARQUET=ON` - Enable Parquet support (default: ON)
- `-DVELOX_ENABLE_GEO=ON` - Enable geospatial functions

### Pre-commit Hooks

```bash
pre-commit install  # Enable hooks for commits and pushes
pre-commit run --all-files  # Run on all files manually
pre-commit run clang-format  # Run specific hook
pre-commit run --hook-stage=manual clang-tidy  # Run clang-tidy (requires compile_commands.json)
```

Other languages: CUDA (experimental GPU support), Python (PyVelox pybind11 bindings), Thrift (remote function framework).

## Directory Structure

### Core Engine

| Directory | Purpose |
|-----------|---------|
| `core/` | Core abstractions: PlanNode, QueryCtx, Expressions, query config |
| `type/` | Type system: all SQL-compatible types, timezone support |
| `vector/` | Columnar data representation (Arrow-compatible): BaseVector, FlatVector, DictionaryVector, ConstantVector, LazyVector, DecodedVector |
| `buffer/` | Buffer management primitives |
| `expression/` | Expression evaluation engine: Expr, CastExpr, special forms (AND/OR/IF/SWITCH/TRY/COALESCE), function signatures |
| `exec/` | Execution engine: relational operators (scan, filter, project, joins, aggregation, sort, exchange, spill), Driver, Task |
| `parse/` | Expression parsing (untyped expression IR) |

### Functions

| Directory | Purpose |
|-----------|---------|
| `functions/prestosql/` | Presto-compatible scalar, aggregate, and window functions |
| `functions/sparksql/` | Spark-compatible function implementations |
| `functions/lib/` | Shared function utilities |
| `functions/remote/` | Remote function execution via Thrift |
| `functions/facebook/` | **Meta-internal** function extensions (not open-sourced) |

### Data I/O

| Directory | Purpose |
|-----------|---------|
| `connectors/` | Connector interface for extensible data sources/sinks |
| `connectors/hive/` | Hive connector (including Iceberg support) |
| `connectors/tpch/` | TPC-H connector |
| `dwio/common/` | Common data warehouse I/O infrastructure |
| `dwio/dwrf/` | DWRF (ORC variant) reader/writer |
| `dwio/parquet/` | Parquet reader/writer |
| `dwio/orc/` | ORC reader |
| `dwio/text/` | Text format reader/writer |

### Infrastructure

| Directory | Purpose |
|-----------|---------|
| `common/memory/` | Memory pools, allocators, arbitration, HashStringAllocator, ByteStream |
| `common/base/` | Core exceptions (VeloxException), stats, async primitives |
| `common/caching/` | Data caching infrastructure |
| `serializers/` | Network serialization: PrestoPage, Spark UnsafeRow, CompactRow |
| `row/` | Row-wise serialization formats |
| `flag_definitions/` | Centralized gflag definitions |

### Other

| Directory | Purpose |
|-----------|---------|
| `python/` | PyVelox: pybind11 bindings for types, vectors, plan building, Arrow interop |
| `duckdb/` | Embedded DuckDB used as reference database for correctness testing |
| `benchmarks/` | Benchmark infrastructure (expression, TPC-H) |
| `examples/` | Example programs demonstrating Velox APIs |
| `tpch/`, `tpcds/` | TPC-H and TPC-DS benchmark data generation |
| `experimental/wave/` | GPU execution via CUDA with JIT compilation |
| `experimental/cudf/` | NVIDIA RAPIDS cuDF integration |
| `experimental/breeze/` | Portable data-parallel algorithms (CUDA, HIP, OpenCL, SYCL, Metal, OpenMP) |
| `docs/` | Sphinx documentation source |

## Coding Style

See `CODING_STYLE.md` for the complete guide.

### Naming Conventions

- **PascalCase**: Types (classes, structs, enums, type aliases), file names.
- **camelCase**: Functions, member and local variables.
- **camelCase_**: Private and protected member variables.
- **snake_case**: Namespace names and build targets.
- **UPPER_SNAKE_CASE**: Macros.
- **kPascalCase**: Static constants and enumerators.
- **testing** prefix: Test-only class methods (`obj.testingFoo()`).

### Comments

- Every file, class, non-trivial method, and member variable should be commented.
- `///` for public API documentation in headers (Doxygen-style).
- `//` for implementation comments in .cpp files and within code blocks.
- Comments should be full English sentences with capital letter and period.
- Capture information that **couldn't be represented as code**.

### Type Aliases

- `XxxPtr` = `std::shared_ptr<Xxx>` (e.g., TypePtr, VectorPtr, PlanNodePtr)
- Prefer value types > `std::optional` > `std::unique_ptr` in that order.

### Variables and Constants

- One variable per line. Initialize at declaration. Declare in smallest scope.
- Use uniform initialization: `size_t size{0};`
- Use `nullptr` for null pointers, `0` for zero values.
- Use `'` for large numbers: `1'000'000`.
- File-level constants in anonymous namespace. Prefer `enum class` over `enum`.
- Use `constexpr std::string_view` for string constants (never `std::string`).

### Function Arguments

- Pass objects as `const T&`. Use `std::string_view` over `const std::string&`.
- For output parameters: non-const ref if not nullable, raw pointer if nullable.
- Use `std::optional` instead of `folly::Optional` or `boost::optional`.

### Namespaces

- All code in `facebook::velox` namespace. Use nested namespace definitions.
- Never put `using namespace` or `using` declarations in headers.
- Do not use `using namespace std;`.

## Exception and Check Macros

| Macro | Use Case | Caught by TRY? |
|-------|----------|----------------|
| `VELOX_CHECK_*` | Internal runtime validation | No |
| `VELOX_USER_CHECK_*` | User input validation | Yes |
| `VELOX_FAIL()` | Throw internal runtime error | No |
| `VELOX_USER_FAIL()` | Throw user error | Yes |
| `VELOX_UNREACHABLE()` | Code that should never execute | No |
| `VELOX_NYI()` | Unimplemented features | No |

Prefer specific comparison macros (`VELOX_CHECK_LT(idx, size)`) over generic checks (`VELOX_CHECK(idx < size)`), as the former includes both values in the error message.

## Function Development

### SimpleFunction API

Template structs with `call()` / `callNullable()` / `callNullFree()` / `callAscii()` methods:

```cpp
template <typename TExec>
struct MyFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  bool call(out_type<Varchar>& result, const arg_type<Varchar>& input) {
    // implementation
    return true; // non-null result
  }
};

// Registration
registerFunction<MyFunction, Varchar, Varchar>({"my_function"});
```

- Always `reserve()` output before emitting entries for complex types.
- Use `atIndex()` for O(1) access on MapView/ArrayView.

### VectorFunction API

Classes inheriting `VectorFunction` for batch-optimized operations. The expression framework peels encodings (Dictionary, Constant) before calling -- do **NOT** handle constant vectors manually.

## Testing

- **Framework**: Google Test (gtest/gmock) via `cpp_unittest` Buck rule.
- **Location**: `velox/<component>/tests/<Component>Test.cpp`
- **Key test utilities**:
  - `FunctionBaseTest` -- for function tests (provides `evaluate()`, `VectorMaker`)
  - `PlanBuilder` + `AssertQueryBuilder` -- for operator/integration tests
  - `OperatorTestBase` -- for execution operator tests
- **Fuzz testing**: Extensive fuzzer infrastructure in `exec/fuzzer/`, `expression/fuzzer/`, `vector/fuzzer/`
- **DuckDB reference**: Used as correctness oracle for fuzz testing


## Preferred Data Structures

- Use `folly::F14FastMap` / `folly::F14FastSet` instead of `std::unordered_map` / `std::unordered_set`.
- Use `folly::Synchronized<T>` for lock-protected objects.
- Use `fmt::format` for string formatting.
