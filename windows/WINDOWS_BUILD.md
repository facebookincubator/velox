# Velox Windows Build Guide

**Last Updated:** November 21, 2025  
**Compiler:** MSVC 19.40+ (Visual Studio 2022)  
**Build System:** CMake 3.21+ with vcpkg  


## Overview

This guide documents the changes and adaptations required to build Velox on Windows with MSVC. The Windows port successfully builds 71+ static libraries with full support for execution engine, Parquet, Arrow integration, Azure Blob File System (ABFS), and Presto/Spark functions.

## Dependencies (via vcpkg)

All dependencies are automatically installed via vcpkg based on `vcpkg.json`:

### Core Dependencies
- **arrow** - Apache Arrow for columnar data processing
- **boost** - C++ libraries collection
- **fmt** - Modern formatting library
- **gflags** - Command-line flags library
- **glog** - Google logging library
- **re2** - Regular expression library
- **protobuf** (3.21.12) - Protocol buffers
- **folly** - Facebook Open-source Library
- **xsimd** - SIMD wrapper library

### Compression Libraries
- **zlib**, **lz4**, **zstd**, **snappy**, **lzo**

### Azure Storage
- **azure-storage-blobs-cpp** - Azure Blob Storage client
- **azure-storage-files-datalake-cpp** - Azure Data Lake Storage client
- **azure-identity-cpp** - Azure authentication

### Other
- **openssl**, **icu**, **simdjson**, **double-conversion**, **libevent**, **libsodium**, **nlohmann-json**, **pybind11**

---

## Build Configuration

### Enabled Features
- `VELOX_BUILD_STATIC=ON` - Build static libraries
- `VELOX_ENABLE_EXEC=ON` - Execution engine
- `VELOX_ENABLE_EXPRESSION=ON` - Expression evaluation
- `VELOX_ENABLE_AGGREGATES=ON` - Aggregation functions
- `VELOX_ENABLE_PARQUET=ON` - Parquet file format support
- `VELOX_ENABLE_HIVE_CONNECTOR=ON` - Hive connector
- `VELOX_ENABLE_ARROW=ON` - Arrow integration
- `VELOX_ENABLE_ABFS=ON` - Azure Blob File System support
- `VELOX_ENABLE_PRESTO_FUNCTIONS=ON` - Presto SQL functions
- `VELOX_ENABLE_SPARK_FUNCTIONS=ON` - Spark SQL functions
- `VELOX_ENABLE_TPCH_CONNECTOR=ON` - TPC-H benchmark connector

### Disabled Features
- `VELOX_BUILD_TESTING=OFF` - Unit tests disabled
- `VELOX_BUILD_SHARED=OFF` - No shared libraries
- `VELOX_ENABLE_S3=OFF` - AWS S3 disabled
- `VELOX_ENABLE_GCS=OFF` - Google Cloud Storage disabled
- `VELOX_ENABLE_HDFS=OFF` - HDFS disabled
- `VELOX_ENABLE_SUBSTRAIT=OFF` - Substrait disabled
- `VELOX_ENABLE_EXAMPLES=OFF` - Examples disabled
- `VELOX_ENABLE_BENCHMARKS=OFF` - Benchmarks disabled

---

## Build Instructions

### Prerequisites
1. **Visual Studio 2022** with C++ development tools
2. **CMake 3.20+**
3. **vcpkg** - Package manager for C++ libraries
4. **WinFlexBison** - Auto-installed by build script if missing

### Recommended: Use the Automated Build Script

The easiest way to build Velox on Windows is to use the provided build script, which automatically handles all configuration including vcpkg detection, WinFlexBison installation, and environment variable setup:

```powershell
# Standard production build (recommended - Release mode)
.\windows\build-windows.ps1

# Or with options:
.\windows\build-windows.ps1 -BuildType Release  # Release build (default)
.\windows\build-windows.ps1 -BuildType Debug    # Debug build
.\windows\build-windows.ps1 -CleanBuild         # Clean rebuild
.\windows\build-windows.ps1 -WithTests          # Include tests
.\windows\build-windows.ps1 -Minimal            # Minimal build

# Combined options:
.\windows\build-windows.ps1 -BuildType Debug -CleanBuild -WithTests
```

**What the script does:**
- Auto-detects or installs vcpkg
- Auto-installs WinFlexBison via winget if missing
- Sets up `BISON_PKGDATADIR` environment variable automatically
- Configures CMake with all required flags
- Enables PDB generation for debugging (both Debug and Release builds)
- Builds with optimal parallelism
- Output: `build\**\{Debug|Release}\*.lib` and `*.pdb`

### Manual Build (Advanced)

If you prefer to configure manually:

```powershell
# 1. Ensure vcpkg and WinFlexBison are installed
winget install winflexbison

# 2. Configure CMake (the build script sets these automatically)
cmake -S . -B build `
  -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake `
  -DVELOX_BUILD_STATIC=ON `
  -DVELOX_BUILD_TESTING=OFF `
  -DVELOX_ENABLE_PARQUET=ON `
  -DVELOX_ENABLE_ARROW=ON `
  -DVELOX_ENABLE_ABFS=ON `
  -DVELOX_ENABLE_EXEC=ON `
  -DVELOX_ENABLE_AGGREGATES=ON `
  -DVELOX_ENABLE_HIVE_CONNECTOR=ON `
  -DVELOX_ENABLE_PRESTO_FUNCTIONS=ON `
  -DVELOX_ENABLE_SPARK_FUNCTIONS=ON

# 3. Build
cmake --build build --config Release -- /m:2
```

**Note:** Manual builds require setting `BISON_PKGDATADIR` environment variable to the WinFlexBison data directory. The build script handles this automatically.

---
- velox_serialization
- velox_test_util
- velox_encode

### DWIO (Data Warehouse I/O) Libraries
- velox_dwio_common
- velox_dwio_common_exception
- velox_dwio_common_encryption
- velox_dwio_catalog_fbhive
- velox_dwio_parquet_reader
- velox_dwio_parquet_writer
- velox_dwio_arrow_parquet_writer_util_lib

### Type System
- velox_type_signature
- velox_type_calculation
- velox_type_tz
- velox_type_fbhive

### Functions
- velox_functions_prestosql_impl
- velox_functions_json
- velox_functions_aggregates
- velox_functions_window
- velox_functions_lib
- velox_functions_lib_date_time_formatter
- velox_functions_util
- velox_function_registry
- velox_is_null_functions
- velox_coverage_util

### Expression Processing
- velox_expression_functions
- velox_signature_parser

### Vector Operations
- velox_vector_fuzzer
- velox_vector_fuzzer_util
- velox_constrained_vector_generator
- velox_constrained_input_generators
- velox_arrow_bridge

### Connectors & Serializers
- velox_connector
- velox_fuzzer_connector
- velox_tpch_connector
- velox_presto_serializer
- velox_presto_types

### Hive Connector Components
- velox_hive_config
- velox_hive_filtered_splitreader
- velox_hive_iceberg_splitreader

### File Systems
- velox_abfs (Azure Blob File System)
- velox_gcs (Google Cloud Storage stub)
- velox_hdfs (HDFS stub)
- velox_s3fs (S3 File System stub)

### TPC-H Benchmark
- velox_tpch_gen
- dbgen

### External Dependencies
- velox_external_md5
- velox_external_date
- simdjson

### Other
- velox_window
- velox_row_fast
- velox_caching
- velox_flag_definitions
- velox_status
- velox_id_map

## Critical Changes Applied

### 1. SIMD Type Conversions (SimdUtil-inl.h)
**Problem:** MSVC treats `__m256i` as a struct, not a primitive type, causing reinterpret_cast errors  
**Solution:** Replace reinterpret_cast with explicit xsimd::batch constructor

```cpp
// Before (causes error on MSVC)
return reinterpret_cast<typename xsimd::batch<T, A>::register_type>(
    _mm256_i64gather_epi64(...));

// After (works on MSVC)
auto result = _mm256_i64gather_epi64(
    reinterpret_cast<const long long*>(base), vindex, kScale);
return xsimd::batch<T, A>(result);
```

### 2. Missing Return After VELOX_UNREACHABLE (HashTable.cpp)
**Problem:** MSVC requires explicit return values even after `VELOX_UNREACHABLE()` macro  
**Solution:** Add explicit return statement with explanatory comment

```cpp
template <>
int32_t HashTable<true>::listNullKeyRows(
    NullKeyRowsIterator*, int32_t, char**) {
  VELOX_UNREACHABLE();
  return 0;  // MSVC requires explicit return even after VELOX_UNREACHABLE
}
```

### 3. DWIO Parquet Reader/Writer Fixes
**Problem:** Multiple MSVC-specific compilation errors in Parquet components  
**Fixes Applied:**
- Fixed const member initialization in FileSink.cpp
- Added missing return statements after unreachable code paths
- Fixed type conversion issues with SIMD operations
- Corrected ADL (Argument-Dependent Lookup) for uint128_t operations

### 4. TypeParser Build (WinFlexBison)
**Problem:** Bison couldn't find m4sugar/m4sugar.m4  
**Solution:** Set `BISON_PKGDATADIR` environment variable and added `YY_NO_UNISTD_H` definition

```cmake
# In CMakeLists.txt
if(WIN32)
  add_definitions(-DYY_NO_UNISTD_H)
endif()
```

### 4. TypeParser Build (WinFlexBison)
**Problem:** Bison couldn't find m4sugar/m4sugar.m4  
**Solution:** Set `BISON_PKGDATADIR` environment variable and added `YY_NO_UNISTD_H` definition

```cmake
# In CMakeLists.txt
if(WIN32)
  add_definitions(-DYY_NO_UNISTD_H)
endif()
```

### 5. int128_t Type Support
**Problem:** Windows doesn't have native `__uint128_t`, MSVC stricter with int128_t operations  
**Solutions:**
- Manual uint128_t arithmetic with explicit casts
- `static_cast<uint64_t>(value % 10)` for conversions
- Template type constraints with `std::remove_cv_t` and `if constexpr`
- Explicit namespace qualification for `facebook::velox::functions::checkedPlus`

### 5. int128_t Type Support
**Problem:** Windows doesn't have native `__uint128_t`, MSVC stricter with int128_t operations  
**Solutions:**
- Manual uint128_t arithmetic with explicit casts
- `static_cast<uint64_t>(value % 10)` for conversions
- Template type constraints with `std::remove_cv_t` and `if constexpr`
- Explicit namespace qualification for `facebook::velox::functions::checkedPlus`

### 6. Folly Namespace Changes
**Problem:** `folly::io::Codec` doesn't exist on Windows  
**Solution:** Changed to `folly::compression::Codec` throughout codebase

### 6. Folly Namespace Changes
**Problem:** `folly::io::Codec` doesn't exist on Windows  
**Solution:** Changed to `folly::compression::Codec` throughout codebase

### 7. Designated Initializers
**Problem:** C++20 feature not available in C++17 mode  
**Solution:** Converted to traditional initialization:
```cpp
// Before (C++20)
CompareFlags flags = {.nullHandlingMode = ...};

// After (C++17)
CompareFlags flags;
flags.nullHandlingMode = ...;
```

### 7. Designated Initializers
**Problem:** C++20 feature not available in C++17 mode  
**Solution:** Converted to traditional initialization:
```cpp
// Before (C++20)
CompareFlags flags = {.nullHandlingMode = ...};

// After (C++17)
CompareFlags flags;
flags.nullHandlingMode = ...;
```

### 8. Zero-Sized Arrays
**Problem:** MSVC rejects arrays with size 0  
**Solution:** `T array[size == 0 ? 1 : size]`

### 8. Zero-Sized Arrays
**Problem:** MSVC rejects arrays with size 0  
**Solution:** `T array[size == 0 ? 1 : size]`

### 9. Template Keyword in Dependent Contexts
**Problem:** MSVC parsing issues with `template` keyword  
**Solution:** Conditional compilation with `#ifdef _MSC_VER`

### 9. Template Keyword in Dependent Contexts
**Problem:** MSVC parsing issues with `template` keyword  
**Solution:** Conditional compilation with `#ifdef _MSC_VER`

### 10. Missing Return Statements
**Problem:** MSVC requires explicit returns after `[[noreturn]]` functions  
**Solution:** Added `return 0;` or `return Status::OK();` after `VELOX_UNREACHABLE()`, etc.

### 10. Missing Return Statements
**Problem:** MSVC requires explicit returns after `[[noreturn]]` functions  
**Solution:** Added `return 0;` or `return Status::OK();` after `VELOX_UNREACHABLE()`, etc.

### 11. Math Constants
**Problem:** `M_E` and `M_PI` not defined on Windows  
**Solution:**
```cpp
#ifndef M_E
#define M_E 2.71828182845904523536
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
```

### 11. Math Constants
**Problem:** `M_E` and `M_PI` not defined on Windows  
**Solution:**
```cpp
#ifndef M_E
#define M_E 2.71828182845904523536
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
```

### 12. Large Object Files
**Problem:** Too many sections in object files  
**Solution:** Added `/bigobj` flag for specific targets

### 12. Large Object Files
**Problem:** Too many sections in object files  
**Solution:** Added `/bigobj` flag for specific targets

### 13. Const Correctness
**Problem:** `const char*` vs `char*` mismatches  
**Solution:** Changed to `static constexpr const char*` where appropriate

### 13. Const Correctness
**Problem:** `const char*` vs `char*` mismatches  
**Solution:** Changed to `static constexpr const char*` where appropriate

## Files Modified (Key Files)

## Files Modified (Key Files)

### SIMD and Performance
- `velox/common/base/SimdUtil-inl.h` - Fixed SIMD type conversions for MSVC

### Execution Engine
- `velox/exec/HashTable.cpp` - Added return statements after VELOX_UNREACHABLE

### DWIO (Parquet Support)
- `velox/dwio/parquet/writer/arrow/FileSink.cpp` - Fixed const member initialization
- `velox/dwio/parquet/reader/BitPackDecoder.cpp` - Fixed SIMD type issues
- `velox/dwio/parquet/reader/StringColumnReader.cpp` - Fixed toAppend ADL
- `velox/dwio/common/SelectiveColumnReader.cpp` - Added uint128_t toAppend overloads
- `velox/dwio/common/DirectDecoder.cpp` - Fixed ZigZag::decode for uint128_t

### Core Type System
- `velox/type/Type.h`
- `velox/type/DecimalUtil.h`
- `velox/type/SimpleFunctionApi.h`
- `velox/type/parser/CMakeLists.txt`

### Functions
- `velox/functions/prestosql/DecimalFunctions.cpp`
- `velox/functions/prestosql/Sequence.cpp`
- `velox/functions/prestosql/Arithmetic.h`
- `velox/functions/prestosql/ArrayFunctions.h`
- `velox/functions/prestosql/ArraySort.cpp`
- `velox/functions/prestosql/Comparisons.cpp`
- `velox/functions/prestosql/types/JsonType.cpp`
- `velox/functions/lib/aggregates/MinMaxAggregateBase.cpp`
- `velox/functions/lib/aggregates/SimpleNumericAggregate.h`

### Serialization
- `velox/serializers/PrestoSerializer.cpp`
- `velox/serializers/RowSerializer.h`

### Expression & Vector
- `velox/expression/VectorReaders.h`
- `velox/expression/ComplexViewTypes.h`
- `velox/core/SimpleFunctionMetadata.h`
- `velox/vector/fuzzer/VectorFuzzer.cpp`

### Common
- `velox/common/base/Scratch.h`
- `velox/common/fuzzer/Utils.h`
- `velox/row/UnsafeRow.h`
- `velox/connectors/Connector.h`

### External
- `velox/external/md5/md5.cpp`

### CMake Configuration
- `velox/functions/prestosql/CMakeLists.txt`
- `velox/functions/prestosql/registration/CMakeLists.txt`
- `velox/type/parser/CMakeLists.txt`

## Known Limitations

1. **BISON_PKGDATADIR Required:** Must set environment variable before each build session
   - **Workaround:** Add to system environment variables permanently via Windows Settings
   - Location varies based on WinFlexBison installation path

2. **Minimal Build Only:** Currently configured with `VELOX_BUILD_MINIMAL=ON`
   - Full build configuration not tested on Windows
   - Some advanced features may not be available

3. **Testing Disabled:** `VELOX_BUILD_TESTING=OFF` to reduce complexity
   - Unit tests are not built or run
   - Functional correctness must be validated through integration testing

4. **Build Configurations:** 
   - **Release build:** Use `build-windows.ps1` (default) or `-BuildType Release`
     - Produces single `velox.lib` via `VELOX_MONO_LIBRARY=ON`
   - **Debug build:** Use `build-windows.ps1 -BuildType Debug`
     - Produces individual `velox_*.lib` files via `VELOX_MONO_LIBRARY=OFF`
     - Mono library disabled to avoid MSVC 4GB static library size limit (LNK1248)
     - Generates 70+ separate .lib files in `build/velox/*/Debug/` directories
   - Both configurations include PDB symbols for debugging

5. **Partial Cloud Storage Support:**
   - Azure Blob File System (ABFS) enabled and built
   - AWS S3 disabled (`VELOX_ENABLE_S3=OFF`)
   - Google Cloud Storage disabled (`VELOX_ENABLE_GCS=OFF`)
   - HDFS disabled (`VELOX_ENABLE_HDFS=OFF`)

6. **MSVC-Specific Code Patterns:**
   - Some code uses `#ifdef _MSC_VER` for Windows-specific implementations
   - May behave differently than Linux/GCC builds
   - SIMD operations use xsimd library wrappers instead of direct intrinsics

## Performance Notes

- Parallel compilation with `/m:2` (adjust based on available CPU cores)
- Can use `/m` without number to use all available cores (may cause high memory usage)
- Total build time: ~20-30 minutes on modern hardware (varies by machine)
- Memory usage: Peak ~8-12GB during parallel compilation
- Recommend at least 16GB RAM for comfortable build experience

## Verification

All libraries can be found in:
```
C:\Git\Velox\build\**\Release\*.lib (Release build)
C:\Git\Velox\build\**\Release\*.pdb (Release symbols)
C:\Git\Velox\build\**\Debug\*.lib (Debug build)
C:\Git\Velox\build\**\Debug\*.pdb (Debug symbols)
```

Total: **71 .lib files** + **71 .pdb files** (per configuration)

### Library Verification Command

```powershell
# Count libraries in output directory
Get-ChildItem C:\Git\Velox\windows-nuget\lib -Filter "*.lib" | Measure-Object

# List all library names
Get-ChildItem C:\Git\Velox\windows-nuget\lib -Filter "*.lib" | Select-Object Name
```

## Next Steps

1. **Testing:** Enable testing framework and verify library functionality
   - Set `VELOX_BUILD_TESTING=ON`
   - Build and run unit tests to validate Windows-specific fixes

2. **Full Build:** Attempt build without `VELOX_BUILD_MINIMAL=ON`
   - May reveal additional components that need Windows fixes
   - Test additional connectors and features

3. **Cloud Storage Integration:**
   - Test ABFS (Azure) integration with real storage accounts
   - Consider enabling S3 and GCS if needed

4. **Debug Configuration:** Verify Debug build works correctly
   - May need additional `/bigobj` flags
   - Could reveal different MSVC warnings/errors

5. **Environment Setup:** Add permanent environment variables
   - Add `BISON_PKGDATADIR` to Windows system environment
   - Consider creating setup script for development machines

6. **Integration Testing:**
   - Link built libraries with downstream projects
   - Verify ABI compatibility and runtime behavior
   - Test Parquet read/write operations
   - Validate expression evaluation and query execution

7. **Performance Benchmarking:**
   - Compare Windows build performance vs Linux
   - Profile SIMD operations to ensure optimal code generation
   - Test with TPC-H benchmark connector

8. **Documentation:**
   - Document any discovered runtime issues
   - Create troubleshooting guide for common Windows build problems
   - Share findings with Velox community

## Troubleshooting

### Common Issues

**Issue:** Bison m4sugar/m4sugar.m4 not found  
**Solution:** Set `BISON_PKGDATADIR` environment variable to WinFlexBison data directory

**Issue:** vcpkg dependencies not found  
**Solution:** Ensure `CMAKE_TOOLCHAIN_FILE` points to vcpkg's buildsystems/vcpkg.cmake

**Issue:** Out of memory during compilation  
**Solution:** Reduce parallel jobs (`/m:1` instead of `/m:2`) or add more RAM

**Issue:** SIMD-related compilation errors  
**Solution:** Ensure xsimd is installed via vcpkg and AVX2 support is available

**Issue:** Linker errors about missing symbols  
**Solution:** Check that all dependency libraries are built in same configuration (Release/Debug)

---

## Windows-Specific Changes Summary

The following changes were required to make Velox compatible with Windows/MSVC:

### 1. Memory Management
- **File:** `velox/common/memory/windows/PosixMemoryCompat.h`
- **Change:** Implement POSIX memory API wrappers (`posix_memalign`, `posix_aligned_alloc`, `posix_free`) using Windows `_aligned_malloc` and `_aligned_free`
- **Reason:** Windows requires aligned memory to be freed with `_aligned_free` instead of `free`

### 2. SIMD Type Conversions
- **File:** `velox/common/base/SimdUtil-inl.h`
- **Change:** Use explicit `xsimd::batch` constructor instead of `reinterpret_cast` for SIMD types
- **Reason:** MSVC treats `__m256i` as struct, not primitive type

### 3. Return Statements After [[noreturn]]
- **Files:** `HashTable.cpp`, various Parquet files
- **Change:** Add explicit return statements after `VELOX_UNREACHABLE()` and similar macros
- **Reason:** MSVC control flow analysis requires returns even after noreturn functions

### 4. Bison/Flex Integration
- **File:** `velox/type/parser/CMakeLists.txt`
- **Change:** Add `-DYY_NO_UNISTD_H` definition, set `BISON_PKGDATADIR` environment variable
- **Reason:** Windows uses WinFlexBison, doesn't have `unistd.h`

### 5. Math Constants
- **Files:** Various math-related files
- **Change:** Define `M_PI`, `M_E` with `#ifndef` guards
- **Reason:** Windows doesn't define POSIX math constants by default

### 6. int128_t Operations
- **Files:** `DecimalUtil.h`, `DecimalFunctions.cpp`, `DirectDecoder.cpp`, `SelectiveColumnReader.cpp`
- **Change:** Add explicit casts, template constraints, function overloads, namespace qualification
- **Reason:** Windows lacks native `__int128_t`, MSVC stricter with custom int128_t type

### 7. Folly Namespace
- **Files:** Compression-related files
- **Change:** Use `folly::compression::Codec` instead of `folly::io::Codec`
- **Reason:** Windows Folly uses different namespace organization

### 8. Designated Initializers
- **Files:** `Comparisons.cpp`, `ArraySort.cpp`
- **Change:** Convert C++20 designated initializers to traditional member-wise initialization
- **Reason:** MSVC in C++17 mode doesn't support designated initializers

### 9. Zero-Sized Arrays
- **Files:** `Scratch.h`, `VectorReaders.h`
- **Change:** Use `T array[size == 0 ? 1 : size]` instead of `T array[size]`
- **Reason:** MSVC rejects zero-sized arrays at compile time

### 10. Template Keyword
- **Files:** `ComplexViewTypes.h`, `VectorReaders.h`
- **Change:** Conditionally omit `template` keyword with `#ifdef _MSC_VER`
- **Reason:** MSVC has different parsing rules for dependent name lookup

### 11. Large Object Files
- **Files:** CMakeLists.txt
- **Change:** Add `/bigobj` compiler flag to large targets
- **Reason:** MSVC has limit on number of sections in object files

### 12. Const Correctness
- **Files:** `JsonType.cpp`, string handling code
- **Change:** Use `static constexpr const char*` for string constants
- **Reason:** MSVC stricter about `const char*` vs `char*` conversions

### 13. ADL for Custom Types
- **Files:** `StringColumnReader.cpp`, `SelectiveColumnReader.cpp`
- **Change:** Add explicit function overloads in correct namespaces
- **Reason:** MSVC requires explicit overloads for Argument-Dependent Lookup

### 14. Static Libraries Only
- **Change:** Build only static libraries (`.lib`), not shared (`.dll`)
- **Reason:** Avoid complexity of DLL export/import declarations

---


## Appendix: CMake Configuration Summary

```cmake
# Core Build Settings
VELOX_BUILD_MINIMAL=ON
VELOX_BUILD_STATIC=ON
VELOX_BUILD_SHARED=OFF
CMAKE_BUILD_TYPE=Release

# Features Enabled
VELOX_ENABLE_EXEC=ON
VELOX_ENABLE_EXPRESSION=ON
VELOX_ENABLE_AGGREGATES=ON
VELOX_ENABLE_PARQUET=ON
VELOX_ENABLE_HIVE_CONNECTOR=ON
VELOX_ENABLE_ABFS=ON
VELOX_ENABLE_ARROW=ON
VELOX_ENABLE_PRESTO_FUNCTIONS=ON
VELOX_ENABLE_SPARK_FUNCTIONS=ON
VELOX_ENABLE_TPCH_CONNECTOR=ON

# Features Disabled
VELOX_BUILD_TESTING=OFF
VELOX_BUILD_TEST_UTILS=OFF
VELOX_ENABLE_S3=OFF
VELOX_ENABLE_GCS=OFF
VELOX_ENABLE_HDFS=OFF
VELOX_ENABLE_SUBSTRAIT=OFF
VELOX_ENABLE_REMOTE_FUNCTIONS=OFF
VELOX_ENABLE_EXAMPLES=OFF
VELOX_ENABLE_BENCHMARKS=OFF
```
