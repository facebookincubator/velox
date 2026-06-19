# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.28)
message(STATUS "Building using CMake version: ${CMAKE_VERSION}")
set(CMAKE_CXX_STANDARD 20 CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE CACHE BOOL "" FORCE)
set(CMAKE_CXX_EXTENSIONS ON CACHE BOOL "" FORCE) # Big Int is an extension
message(STATUS "CXX standard: ${CMAKE_CXX_STANDARD}")

# Disable vcpkg manifest warnings in MSBuild
set(VCPKG_APPLOCAL_DEPS OFF)

# Use dynamic runtime (/MD) - can be overridden by build scripts
if(NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# The policy allows us to change options without caching.
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# Sets new behavior for CMP0135, which controls how timestamps are extracted
# when using ExternalProject_Add():
# https://cmake.org/cmake/help/latest/policy/CMP0135.html
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()

# Set the project name.
project(velox)

set(VELOX_TARGET_IS_ARM64 FALSE)
foreach(
  _velox_target_arch
  "${CMAKE_SYSTEM_PROCESSOR}"
  "${CMAKE_GENERATOR_PLATFORM}"
  "${CMAKE_VS_PLATFORM_NAME}"
  "${VCPKG_TARGET_ARCHITECTURE}"
)
  string(TOLOWER "${_velox_target_arch}" _velox_target_arch_lower)
  if(_velox_target_arch_lower MATCHES "^(arm64|aarch64)$")
    set(VELOX_TARGET_IS_ARM64 TRUE)
  endif()
endforeach()

# Force static linking on Windows after project initialization
if(WIN32)
  set(BUILD_SHARED_LIBS OFF)
  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_USE_STATIC_RUNTIME OFF)  # Use dynamic runtime (/MD)
  # Note: Do NOT define PROTOBUF_USE_DLLS - protobuf headers check
  # defined(PROTOBUF_USE_DLLS), not its value, so even =0 enables dllimport
  # Prevent Windows min/max macros from conflicting with std::min/std::max
  add_compile_definitions(NOMINMAX)
  # Static linking for Arrow and Parquet (prevent __declspec(dllimport))
  add_compile_definitions(ARROW_STATIC PARQUET_STATIC)
  
  # Increase stack size for Windows executables.
  # Debug builds need 32MB due to larger stack frames from no optimizations,
  # debug iterators, and security checks (/GS) with deep recursion (e.g.,
  # HiveTypeParser). Release builds use 8MB.
  # Use per-config linker flags for VS multi-config generators.
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /STACK:33554432")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /STACK:8388608")
  set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /STACK:8388608")
  set(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL "${CMAKE_EXE_LINKER_FLAGS_MINSIZEREL} /STACK:8388608")
endif()

# Workaround for Boost 1.89+ incompatibility on Windows - must be set before any targets
if(WIN32)
  add_compile_definitions(BOOST_MOVE_USE_STANDARD_LIBRARY_MOVE)
endif()

# Disable vcpkg manifest mode warnings in MSBuild
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_GLOBAL_VcpkgEnabled false)

# Defines the target Java version to ensure compatibility with Java-specific
# algorithms, such as double-to-string conversion. Defaults to 17 if not
# specified.
set(VELOX_JAVA_VERSION 17 CACHE STRING "Target Java version")
message(STATUS "Building for Java version: ${VELOX_JAVA_VERSION}")

find_package(Java)

if(Java_FOUND)
  if(NOT Java_VERSION_MAJOR EQUAL VELOX_JAVA_VERSION)
    message(
      WARNING
      "Installed Java version (${Java_VERSION_MAJOR}) does not match "
      "VELOX_JAVA_VERSION (${VELOX_JAVA_VERSION})!"
    )
  endif()
  message(STATUS "Java found: ${Java_VERSION}")
else()
  message(STATUS "Java not found - some features may be disabled")
endif()

# If we are in an active conda env disable search in system paths and add env to
# prefix path
if(DEFINED ENV{CONDA_PREFIX})
  if(NOT DEFINED ENV{VELOX_DEPENDENCY_SOURCE} OR "$ENV{VELOX_DEPENDENCY_SOURCE}" STREQUAL "CONDA")
    message(STATUS "Using Conda environment: $ENV{CONDA_PREFIX}")
    set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    # Override in case it was set to CONDA
    set(ENV{VELOX_DEPENDENCY_SOURCE} AUTO)
  elseif(
    DEFINED ENV{VELOX_DEPENDENCY_SOURCE}
    AND NOT "$ENV{VELOX_DEPENDENCY_SOURCE}" STREQUAL "CONDA"
  )
    message(STATUS "Overriding Conda environment: $ENV{CONDA_PREFIX}")
  endif()
endif()

if(DEFINED ENV{INSTALL_PREFIX})
  message(STATUS "Dependency install directory set to: $ENV{INSTALL_PREFIX}")
  list(APPEND CMAKE_PREFIX_PATH "$ENV{INSTALL_PREFIX}")
endif()

list(
  PREPEND
  CMAKE_MODULE_PATH
  "${PROJECT_SOURCE_DIR}/CMake"
  "${PROJECT_SOURCE_DIR}/CMake/third-party"
)

# Include our ThirdPartyToolchain dependencies macros
include(ResolveDependency)
include(VeloxUtils)
include(CMakeDependentOption)

velox_set_with_default(VELOX_DEPENDENCY_SOURCE_DEFAULT VELOX_DEPENDENCY_SOURCE AUTO)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Add all options below
option(
  VELOX_BUILD_TESTING
  "Enable Velox tests. This will enable all other build options automatically."
  ON
)
option(
  VELOX_BUILD_MINIMAL
  "Build a minimal set of components only. This will override other build options."
  OFF
)
option(
  VELOX_BUILD_MINIMAL_WITH_DWIO
  "Build a minimal set of components, including DWIO (file format readers/writers).
   This will override other build options."
  OFF
)
option(VELOX_MONO_LIBRARY "Build single unified library." ON)
option(ENABLE_ALL_WARNINGS "Enable -Wall and -Wextra compiler warnings." ON)
option(VELOX_BUILD_SHARED "Build Velox as shared libraries." OFF)
option(VELOX_BUILD_CMAKE_PACKAGE "Build CMake package for Velox." OFF)
option(VELOX_SKIP_WAVE_BRANCH_KERNEL_TEST "Disable Wave branch kernel test." OFF)
# While it's possible to build both in one go we currently want to build either
# static or shared.
cmake_dependent_option(
  VELOX_BUILD_STATIC
  "Build Velox as static libraries."
  ON
  "NOT VELOX_BUILD_SHARED"
  OFF
)

if(VELOX_BUILD_SHARED AND NOT VELOX_MONO_LIBRARY)
  # The large number of targets currently in use within Velox make a shared
  # build when not using the mono library brittle and labor intensive
  set(VELOX_MONO_LIBRARY ON)
endif()

if(VELOX_BUILD_SHARED)
  message(
    WARNING
    "When building Velox as a shared library it's recommended to build against a shared build of folly to avoid issues with linking of gflags. "
    "This is currently NOT being enforced so user discretion is advised."
  )
endif()

# option() always creates a BOOL variable so we have to use a normal cache
# variable with STRING type for this option.
#
# * AUTO: Try SYSTEM first fall back to BUNDLED.
# * SYSTEM: Use installed dependencies via find_package.
# * BUNDLED: Build dependencies from source.
set(
  VELOX_DEPENDENCY_SOURCE
  ${VELOX_DEPENDENCY_SOURCE_DEFAULT}
  CACHE STRING
  "Default source for all dependencies with source builds enabled: AUTO SYSTEM BUNDLED."
)
set(
  VELOX_GFLAGS_TYPE
  "shared"
  CACHE STRING
  "Specify whether to find the gflags package as a shared or static package"
)
option(VELOX_ENABLE_EXEC "Build exec." ON)
option(VELOX_ENABLE_AGGREGATES "Build aggregates." ON)
option(VELOX_ENABLE_HIVE_CONNECTOR "Build Hive connector." ON)
option(VELOX_ENABLE_TPCH_CONNECTOR "Build TPC-H connector." ON)
option(VELOX_ENABLE_TPCDS_CONNECTOR "Build TPC-DS connector." ON)
option(VELOX_ENABLE_PRESTO_FUNCTIONS "Build Presto SQL functions." ON)
option(VELOX_ENABLE_SPARK_FUNCTIONS "Build Spark SQL functions." ON)
option(VELOX_ENABLE_ICEBERG_FUNCTIONS "Build Iceberg functions." ON)
option(VELOX_ENABLE_EXPRESSION "Build expression." ON)
option(
  VELOX_ENABLE_EXAMPLES
  "Build examples. This will enable VELOX_ENABLE_EXPRESSION automatically."
  OFF
)
option(VELOX_ENABLE_BENCHMARKS "Enable Velox top level benchmarks." OFF)
option(VELOX_ENABLE_BENCHMARKS_BASIC "Enable Velox basic benchmarks." OFF)
option(VELOX_ENABLE_S3 "Build S3 Connector" OFF)
option(VELOX_ENABLE_GCS "Build GCS Connector" OFF)
option(VELOX_ENABLE_ABFS "Build Abfs Connector" OFF)
option(VELOX_ENABLE_HDFS "Build Hdfs Connector" OFF)
option(VELOX_ENABLE_PARQUET "Enable Parquet support" ON)
option(VELOX_ENABLE_ARROW "Enable Arrow support" OFF)
option(VELOX_ENABLE_GEO "Enable Geospatial support" ON)
option(VELOX_ENABLE_REMOTE_FUNCTIONS "Enable remote function support" OFF)
option(VELOX_ENABLE_CCACHE "Use ccache if installed." ON)
option(VELOX_ENABLE_COMPRESSION_LZ4 "Enable Lz4 compression support." OFF)

option(VELOX_BUILD_TEST_UTILS "Builds Velox test utilities" OFF)
option(VELOX_BUILD_VECTOR_TEST_UTILS "Builds Velox vector test utilities" OFF)
option(VELOX_BUILD_PYTHON_PACKAGE "Builds Velox Python bindings" OFF)
option(VELOX_PYTHON_LEGACY_ONLY "Build only the pyvelox module" OFF)
option(VELOX_BUILD_RUNNER "Builds velox runner" ON)
option(
  VELOX_ENABLE_INT64_BUILD_PARTITION_BOUND
  "make buildPartitionBounds_ a vector int64 instead of int32 to avoid integer overflow when the hashtable has billions of records"
  OFF
)
option(VELOX_SIMDJSON_SKIPUTF8VALIDATION "Skip simdjson utf8 validation in JSON parsing" OFF)
option(VELOX_ENABLE_FAISS "Build faiss vector search support" OFF)
option(VELOX_ENABLE_TORCHWAVE "Build TorchWave experimental subsystem." OFF)

# Explicitly force compilers to generate colored output. Compilers usually do
# this by default if they detect the output is a terminal, but this assumption
# is broken if you use ninja.
option(VELOX_FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." OFF)

# Enable the address and undefined behavior sanitizers. They can run together.
option(
  VELOX_ENABLE_ASAN_UBSAN_SANITIZERS
  "Builds velox with Clang ASAN and UBSAN sanitizer flags"
  OFF
)

if(${VELOX_BUILD_MINIMAL} OR ${VELOX_BUILD_MINIMAL_WITH_DWIO})
  # Enable and disable components for velox base build
  set(VELOX_BUILD_TESTING OFF)
  set(VELOX_ENABLE_PRESTO_FUNCTIONS ON)
  set(VELOX_ENABLE_EXPRESSION ON)
  set(VELOX_ENABLE_EXEC OFF)
  set(VELOX_ENABLE_AGGREGATES OFF)
  set(VELOX_ENABLE_HIVE_CONNECTOR OFF)
  set(VELOX_ENABLE_TPCH_CONNECTOR OFF)
  set(VELOX_ENABLE_TPCDS_CONNECTOR OFF)
  set(VELOX_ENABLE_SPARK_FUNCTIONS OFF)
  set(VELOX_ENABLE_EXAMPLES OFF)
  set(VELOX_ENABLE_S3 OFF)
  set(VELOX_ENABLE_GCS OFF)
  set(VELOX_ENABLE_ABFS OFF)
else()
  if(VELOX_BUILD_CMAKE_PACKAGE)
    message(
      FATAL_ERROR
      "VELOX_BUILD_CMAKE_PACKAGE is only available with "
      "VELOX_BUILD_MINIMAL=ON or VELOX_BUILD_MINIMAL_WITH_DWIO=ON for now."
    )
  endif()
endif()

if(${VELOX_ENABLE_BENCHMARKS})
  set(VELOX_ENABLE_BENCHMARKS_BASIC ON)
endif()

if(VELOX_ENABLE_BENCHMARKS_BASIC)
  set(VELOX_BUILD_TEST_UTILS ON)
endif()

if(VELOX_ENABLE_TORCHWAVE)
  set(VELOX_ENABLE_WAVE ON)
  if(DEFINED ENV{TORCH_DIR})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{TORCH_DIR}")
  endif()
  find_package(Torch REQUIRED)
  message(STATUS "Found Torch: ${TORCH_INSTALL_PREFIX}")
endif()

if(VELOX_ENABLE_CUDF)
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message(FATAL_ERROR "cuDF requires GCC. Found ${CMAKE_CXX_COMPILER_ID}.")
  elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.3)
    message(FATAL_ERROR "cuDF requires GCC >= 13.3. Found GCC ${CMAKE_CXX_COMPILER_VERSION}.")
  endif()
  message(STATUS "Building curl from source to satisfy cuDF curl version requirement")
  set(CURL_SOURCE BUNDLED)
  velox_resolve_dependency(CURL)
endif()

if(VELOX_BUILD_TESTING OR VELOX_BUILD_TEST_UTILS)
  # cuDF bundles curl since it needs a specific version.
  # Use bundled or system curl otherwise.
  if(NOT VELOX_ENABLE_CUDF)
    velox_set_source(CURL)
    velox_resolve_dependency(CURL)
  endif()
  set(VELOX_ENABLE_DUCKDB ON)
  set(VELOX_ENABLE_PARSE ON)
endif()

if(${VELOX_BUILD_TESTING} OR ${VELOX_BUILD_MINIMAL_WITH_DWIO} OR ${VELOX_ENABLE_HIVE_CONNECTOR})
  set(VELOX_ENABLE_COMPRESSION_LZ4 ON)
endif()

if(${VELOX_ENABLE_EXAMPLES})
  set(VELOX_ENABLE_EXPRESSION ON)
endif()

if(${VELOX_BUILD_PYTHON_PACKAGE})
  set(VELOX_ENABLE_PRESTO_FUNCTIONS ON)
  set(VELOX_ENABLE_DUCKDB ON)
  set(VELOX_ENABLE_EXPRESSION ON)
  set(VELOX_ENABLE_PARSE ON)
  set(VELOX_ENABLE_EXEC ON)
  set(VELOX_ENABLE_AGGREGATES ON)
  set(VELOX_ENABLE_SPARK_FUNCTIONS ON)

  if(NOT VELOX_PYTHON_LEGACY_ONLY)
    set(VELOX_BUILD_RUNNER OFF)
    set(VELOX_BUILD_TEST_UTILS ON) # required for velox_exec_test_lib
  endif()
endif()

if(${VELOX_ENABLE_DUCKDB})
  velox_set_source(DuckDB)
  velox_resolve_dependency(DuckDB)
endif()

if(VELOX_ENABLE_FAISS)
  velox_set_source(faiss)
  velox_resolve_dependency(faiss)
  if(NOT TARGET FAISS::faiss)
    add_library(FAISS::faiss ALIAS faiss)
  endif()
endif()

if(DEFINED ENV{INSTALL_PREFIX})
  # Allow installed package headers to be picked up before brew/system package
  # headers. We set this after DuckDB bundling since DuckDB uses its own
  # dependencies.
  include_directories(BEFORE "$ENV{INSTALL_PREFIX}/include")
endif()

# We look for OpenSSL here to cache the result enforce the version across our
# dependencies.
# Force OpenSSL static linking on Windows
if(WIN32)
  set(OPENSSL_USE_STATIC_LIBS ON)
  set(OPENSSL_MSVC_STATIC_RUNTIME ON)
endif()
find_package(OpenSSL REQUIRED)

if(
  VELOX_ENABLE_CCACHE
  AND NOT CMAKE_C_COMPILER_LAUNCHER
  AND NOT CMAKE_CXX_COMPILER_LAUNCHER
  AND NOT CMAKE_CUDA_COMPILER_LAUNCHER
)
  find_program(CCACHE_FOUND ccache)

  if(CCACHE_FOUND)
    message(STATUS "Using ccache: ${CCACHE_FOUND}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_FOUND})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_FOUND})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_FOUND})
    # keep comments as they might matter to the compiler
    set(ENV{CCACHE_COMMENTS} "1")
  endif()
endif()

if(${VELOX_FORCE_COLORED_OUTPUT})
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
  elseif(
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
    OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"
  )
    add_compile_options(-fcolor-diagnostics)
  endif()
endif()

if(VELOX_ENABLE_S3)
  # Set AWS_ROOT_DIR if you have a custom install location of AWS SDK CPP.
  if(AWSSDK_ROOT_DIR)
    list(APPEND CMAKE_PREFIX_PATH ${AWSSDK_ROOT_DIR})
  endif()
  find_package(AWSSDK 1.11.654 REQUIRED COMPONENTS s3;identity-management)
  add_definitions(-DVELOX_ENABLE_S3)
endif()

if(VELOX_ENABLE_ABFS)
  # Set AZURESDK_ROOT_DIR if you have a custom install location of Azure Storage
  # SDK CPP.
  if(AZURESDK_ROOT_DIR)
    list(APPEND CMAKE_PREFIX_PATH ${AZURESDK_ROOT_DIR})
  endif()
  # files-datalake is built on blobs
  find_package(azure-storage-files-datalake-cpp CONFIG REQUIRED)
  find_package(azure-identity-cpp CONFIG REQUIRED)
  add_definitions(-DVELOX_ENABLE_ABFS)
endif()

if(VELOX_ENABLE_HDFS)
  add_definitions(-DVELOX_ENABLE_HDFS)
  # JVM libhdfs requires arrow dependency.
  set(VELOX_ENABLE_ARROW ON)
endif()

if(VELOX_ENABLE_PARQUET)
  add_definitions(-DVELOX_ENABLE_PARQUET)
  # Native Parquet reader requires Arrow Parquet writer, which are
  # included in Arrow.
  set(VELOX_ENABLE_ARROW ON)
endif()

# make buildPartitionBounds_ a vector int64 instead of int32 to avoid integer
# overflow
if(${VELOX_ENABLE_INT64_BUILD_PARTITION_BOUND})
  add_compile_definitions(VELOX_ENABLE_INT64_BUILD_PARTITION_BOUND)
endif()

# MacOSX enables two-level namespace by default:
# http://mirror.informatimago.com/next/developer.apple.com/releasenotes/DeveloperTools/TwoLevelNamespaces.html
# Enables -flat_namespace so type_info can be deduplicated across .so boundaries
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  add_link_options("-Wl,-flat_namespace")
endif()

if(UNIX AND NOT APPLE)
  # linker flags, -export-dynamic for rtti
  add_link_options("-Wl,-export-dynamic")
endif()

# Required so velox code can be used in a dynamic library
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# For C++20 support we need GNU GCC11 (or later versions) or Clang/AppleClang 15
# (or later versions) to get support for the used features.
if(
  NOT
    (
      (
        "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
        AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 11
      )
      OR
        (
          (
            "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
            OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"
          )
          AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15
        )
      OR
        (
          "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC"
        )
    )
)
  message(
    FATAL_ERROR
    "Unsupported compiler ${CMAKE_CXX_COMPILER_ID} with version ${CMAKE_CXX_COMPILER_VERSION} found."
  )
endif()

if(WIN32)
  # Windows-specific compiler flags for MSVC
  if(VELOX_TARGET_IS_ARM64)
    set(SCRIPT_CXX_FLAGS "/std:c++20")
  else()
    set(SCRIPT_CXX_FLAGS "/arch:AVX2 /std:c++20")
  endif()
  message("Setting CMAKE_CXX_FLAGS=${SCRIPT_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SCRIPT_CXX_FLAGS}")
else()
  execute_process(
    COMMAND
      bash -c
      "( source ${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup-helper-functions.sh && echo -n $(get_cxx_flags $ENV{CPU_TARGET}))"
    OUTPUT_VARIABLE SCRIPT_CXX_FLAGS
    RESULT_VARIABLE COMMAND_STATUS
  )

  if(COMMAND_STATUS EQUAL "1")
    message(FATAL_ERROR "Unable to determine compiler flags!")
  endif()
  message("Setting CMAKE_CXX_FLAGS=${SCRIPT_CXX_FLAGS}")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SCRIPT_CXX_FLAGS}")
endif()
if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D USE_VELOX_COMMON_BASE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D HAS_UNCAUGHT_EXCEPTIONS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D GLOG_USE_GLOG_EXPORT")
  # Prevent glog conflict with Windows ERROR macro from wingdi.h
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D GLOG_NO_ABBREVIATED_SEVERITIES")
  # Workaround for Boost 1.89.0 intrusive/detail/mpl.hpp bug on Windows
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D BOOST_MOVE_USE_STANDARD_LIBRARY_MOVE")
  # NOTE: BOOST_MOVE_USE_STANDARD_LIBRARY_TYPE_TRAITS causes compilation errors in C++20
  # because Boost Intrusive still tries to use boost::move_detail::is_unsigned
  # which is not defined when this macro is set.
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D BOOST_MOVE_USE_STANDARD_LIBRARY_TYPE_TRAITS")
  # Boost workarounds for C++20 compatibility
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D BOOST_NO_CXX98_FUNCTION_BASE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D BOOST_USE_STD_UNARY_FUNCTION")
  # Disable all warnings for cleaner build output
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D USE_VELOX_COMMON_BASE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D HAS_UNCAUGHT_EXCEPTIONS")
endif()
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsigned-char")
endif()

# Under Ninja, we are able to designate certain targets large enough to require
# restricted parallelism.
if("${MAX_HIGH_MEM_JOBS}")
  set_property(GLOBAL PROPERTY JOB_POOLS "high_memory_pool=${MAX_HIGH_MEM_JOBS}")
else()
  set_property(GLOBAL PROPERTY JOB_POOLS high_memory_pool=1000)
endif()

if("${MAX_LINK_JOBS}")
  set_property(GLOBAL APPEND PROPERTY JOB_POOLS "link_job_pool=${MAX_LINK_JOBS}")
  set(CMAKE_JOB_POOL_LINK link_job_pool)
endif()

if(ENABLE_ALL_WARNINGS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(
      KNOWN_COMPILER_SPECIFIC_WARNINGS
      "-Wno-range-loop-analysis \
         -Wno-mismatched-tags \
         -Wno-nullability-completeness"
    )
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(
      KNOWN_COMPILER_SPECIFIC_WARNINGS
      "-Wno-implicit-fallthrough \
         -Wno-class-memaccess \
         -Wno-comment \
         -Wno-int-in-bool-context \
         -Wno-redundant-move \
         -Wno-array-bounds \
         -Wno-maybe-uninitialized \
         -Wno-unused-result \
         -Wno-format-overflow \
         -Wno-strict-aliasing"
    )
    # Avoid compiler bug for GCC 12.2.1
    # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
    if(CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL "12.2.1")
      string(APPEND KNOWN_COMPILER_SPECIFIC_WARNINGS " -Wno-restrict")
    endif()
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "14.0.0")
      string(APPEND KNOWN_COMPILER_SPECIFIC_WARNINGS " -Wno-error=template-id-cdtor")
      string(APPEND KNOWN_COMPILER_SPECIFIC_WARNINGS " -Wno-overloaded-virtual")
      string(APPEND KNOWN_COMPILER_SPECIFIC_WARNINGS " -Wno-error=tautological-compare")
    endif()
    # Avoid compiler bug for GCC 13.2.0
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "13.2.0")
      string(APPEND KNOWN_COMPILER_SPECIFIC_WARNINGS " -Wno-error=overloaded-virtual")
      string(APPEND KNOWN_COMPILER_SPECIFIC_WARNINGS " -Wno-error=stringop-overflow")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # MSVC warning settings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
  endif()

  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(
      KNOWN_WARNINGS
      "-Wno-unused \
         -Wno-unused-parameter \
         -Wno-sign-compare \
         -Wno-ignored-qualifiers \
         ${KNOWN_COMPILER_SPECIFIC_WARNINGS}"
    )

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra ${KNOWN_WARNINGS}")
  endif()
endif()

if(VELOX_ENABLE_WAVE OR VELOX_ENABLE_CUDF)
  enable_language(CUDA)
  # Determine CUDA_ARCHITECTURES automatically.
  cmake_policy(SET CMP0104 NEW)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # it will fail later in the build otherwise
    message(FATAL_ERROR "-DCMAKE_CUDA_ARCHITECTURES= must be set")
  endif()
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(VELOX_CUDA_DEBUG_FLAGS "$<$<COMPILE_LANGUAGE:CUDA>:-G>")
  endif()
  find_package(CUDAToolkit REQUIRED)
  # Suppress deprecated GPU targets warning for CUDA 12.8.x
  if(
    CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8.0
    AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0.0
  )
    string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets")
  endif()
  if(VELOX_ENABLE_CUDF)
    foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
      if(arch LESS 70)
        message(
          FATAL_ERROR
          "CUDA architecture ${arch} is below 70. CUDF requires Volta (SM 70) or newer GPUs."
        )
      endif()
    endforeach()
    set(VELOX_ENABLE_ARROW ON)
    velox_set_source(cudf)
    velox_resolve_dependency(cudf)
    if(TARGET aws-cpp-sdk-core)
      # Fix for AWS SDK CPP using hardcoded system curl instead of soft link to curl
      get_target_property(override_curl_lib aws-cpp-sdk-core INTERFACE_LINK_LIBRARIES)
      list(REMOVE_ITEM override_curl_lib "/usr/lib64/libcurl.so")
      list(APPEND override_curl_lib "\$<LINK_ONLY:CURL::libcurl>")
      set_target_properties(
        aws-cpp-sdk-core
        PROPERTIES INTERFACE_LINK_LIBRARIES "${override_curl_lib}"
      )
    endif()
  endif()
endif()

# Set after the test of the CUDA compiler. Otherwise, the test fails with
# -latomic not found because it is added right after the compiler exe.
# Only link libatomic if compiler-rt is not available (compiler-rt provides atomic implementation)
velox_configure_clang_atomic_linker_flags()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  execute_process(
    COMMAND brew --prefix icu4c
    RESULT_VARIABLE BREW_ICU4C
    OUTPUT_VARIABLE BREW_ICU4C_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(BREW_ICU4C EQUAL 0 AND EXISTS "${BREW_ICU4C_PREFIX}")
    message(STATUS "Found icu4c installed by Homebrew at ${BREW_ICU4C_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${BREW_ICU4C_PREFIX}")
  else()
    list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/icu4c")
  endif()
endif()

velox_set_source(ICU)
set(VELOX_ICU_COMPONENTS data i18n io uc)
# The 'tu' (tools utility) component is only available when ICU is built with tools
# feature. For ARM64 cross-compilation, tu is not built for the target. On
# Windows the vcpkg static ICU port does not ship icutu either, and Velox never
# links ICU::tu, so the component is not required there.
if(NOT VELOX_TARGET_IS_ARM64 AND NOT WIN32)
  list(APPEND VELOX_ICU_COMPONENTS tu)
endif()
velox_resolve_dependency(
  ICU
  COMPONENTS ${VELOX_ICU_COMPONENTS}
)

set(
  BOOST_COMPONENTS
  atomic
  context
  date_time
  filesystem
  program_options
  regex
  thread
)
set(
  BOOST_OPTIONAL_COMPONENTS
  system # Boost.System is header only since Boost 1.89
)

velox_set_source(Boost)
velox_resolve_dependency(
  Boost
  1.77.0
  MODULE
  COMPONENTS ${BOOST_COMPONENTS}
  OPTIONAL_COMPONENTS ${BOOST_OPTIONAL_COMPONENTS}
)
if(NOT TARGET Boost::system)
  add_library(Boost::system ALIAS Boost::headers)
endif()

# Workaround for Boost 1.89+ incompatibility with boost::intrusive on Windows
if(WIN32 AND Boost_VERSION_STRING VERSION_GREATER_EQUAL "1.89.0")
  add_compile_definitions(BOOST_MOVE_USE_STANDARD_LIBRARY_MOVE)
endif()

velox_set_source(gflags)

if(WIN32)
  velox_resolve_dependency(gflags)
else()
  velox_resolve_dependency(gflags COMPONENTS ${VELOX_GFLAGS_TYPE})
endif()

if(NOT TARGET gflags::gflags)
  # This is a bit convoluted, but we want to be able to use gflags::gflags as a
  # target even when velox is built as a subproject which uses
  # `find_package(gflags)` which does not create a globally imported target that
  # we can ALIAS.
  add_library(gflags::gflags ALIAS gflags)
endif()

if(${gflags_SOURCE} STREQUAL "BUNDLED")
  # Allow explicit glog_SOURCE override (e.g. when system glog is built
  # against the same gflags version as BUNDLED).
  if(DEFINED ENV{glog_SOURCE})
    set(glog_SOURCE $ENV{glog_SOURCE})
  else()
    set(glog_SOURCE BUNDLED)
  endif()
else()
  set(glog_SOURCE SYSTEM)
endif()
velox_resolve_dependency(glog)

velox_set_source(fmt)
velox_resolve_dependency(fmt 9.0.0)

if(VELOX_ENABLE_COMPRESSION_LZ4)
  find_package(lz4 REQUIRED)
endif()

if(${VELOX_BUILD_MINIMAL_WITH_DWIO} OR ${VELOX_ENABLE_HIVE_CONNECTOR})
  # DWIO needs all sorts of stream compression libraries.
  #
  # TODO: make these optional and pluggable.
  find_package(ZLIB REQUIRED)
  find_package(zstd REQUIRED)
  find_package(Snappy REQUIRED)
  # Ensure zstd::zstd target exists - handle different zstd package configurations

  # On Windows, fix IMPORTED_IMPLIB for DLL targets.
  # Use VCPKG_INSTALLED_DIR/VCPKG_TARGET_TRIPLET for the base prefix path.
  # Do NOT use CMAKE_PREFIX_PATH[0] because the vcpkg toolchain reverses path
  # order for Debug builds, making [0] = .../debug instead of the root prefix,
  # which causes double "debug" in paths like .../debug/debug/lib/zstd.lib.
  if(WIN32)
    set(_vcpkg_prefix "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
    if(TARGET Snappy::snappy)
      set_target_properties(Snappy::snappy PROPERTIES
        IMPORTED_IMPLIB_DEBUG "${_vcpkg_prefix}/debug/lib/snappy.lib"
        IMPORTED_IMPLIB_RELEASE "${_vcpkg_prefix}/lib/snappy.lib"
        IMPORTED_IMPLIB_MINSIZEREL "${_vcpkg_prefix}/lib/snappy.lib"
        IMPORTED_IMPLIB_RELWITHDEBINFO "${_vcpkg_prefix}/lib/snappy.lib"
      )
    endif()
    if(TARGET zstd::zstd OR TARGET zstd::libzstd_shared)
      set(_zstd_target zstd::zstd)
      if(NOT TARGET zstd::zstd)
        set(_zstd_target zstd::libzstd_shared)
      endif()
      set_target_properties(${_zstd_target} PROPERTIES
        IMPORTED_IMPLIB_DEBUG "${_vcpkg_prefix}/debug/lib/zstd.lib"
        IMPORTED_IMPLIB_RELEASE "${_vcpkg_prefix}/lib/zstd.lib"
        IMPORTED_IMPLIB_MINSIZEREL "${_vcpkg_prefix}/lib/zstd.lib"
        IMPORTED_IMPLIB_RELWITHDEBINFO "${_vcpkg_prefix}/lib/zstd.lib"
      )
    endif()
    if(TARGET lz4::lz4)
      set_target_properties(lz4::lz4 PROPERTIES
        IMPORTED_IMPLIB_DEBUG "${_vcpkg_prefix}/debug/lib/lz4d.lib"
        IMPORTED_IMPLIB_RELEASE "${_vcpkg_prefix}/lib/lz4.lib"
        IMPORTED_IMPLIB_MINSIZEREL "${_vcpkg_prefix}/lib/lz4.lib"
        IMPORTED_IMPLIB_RELWITHDEBINFO "${_vcpkg_prefix}/lib/lz4.lib"
      )
    endif()
    if(TARGET lzo2::lzo2)
      set_target_properties(lzo2::lzo2 PROPERTIES
        IMPORTED_IMPLIB_DEBUG "${_vcpkg_prefix}/debug/lib/lzo2.lib"
        IMPORTED_IMPLIB_RELEASE "${_vcpkg_prefix}/lib/lzo2.lib"
        IMPORTED_IMPLIB_MINSIZEREL "${_vcpkg_prefix}/lib/lzo2.lib"
        IMPORTED_IMPLIB_RELWITHDEBINFO "${_vcpkg_prefix}/lib/lzo2.lib"
      )
    endif()
  endif()

  if(NOT TARGET zstd::zstd)
    if(TARGET zstd::libzstd_static)
      add_library(zstd::zstd ALIAS zstd::libzstd_static)
    elseif(TARGET zstd::libzstd_shared)
      add_library(zstd::zstd ALIAS zstd::libzstd_shared)
    else()
      # Fallback: use Findzstd.cmake to create the target
      include(Findzstd)
    endif()
  endif()
endif()

velox_set_source(re2)
velox_resolve_dependency(re2)

if(${VELOX_BUILD_PYTHON_PACKAGE})
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
  velox_set_source(pybind11)
  velox_resolve_dependency(pybind11 2.10.0)
endif()

# DWIO (ORC/DWRF) depends on protobuf.
if(${VELOX_BUILD_MINIMAL_WITH_DWIO} OR ${VELOX_ENABLE_HIVE_CONNECTOR} OR VELOX_ENABLE_GCS)
  # Locate or build protobuf.
  velox_set_source(Protobuf)
  # Force static protobuf on Windows
  if(WIN32)
    set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "")
    set(protobuf_MSVC_STATIC_RUNTIME ON CACHE BOOL "")
    # Ensure PROTOBUF_USE_DLLS is not defined (use static linking)
    # protobuf headers check defined(PROTOBUF_USE_DLLS), not its value
    velox_resolve_dependency(Protobuf CONFIG REQUIRED)
  else()
    velox_resolve_dependency(Protobuf 3.21.7 REQUIRED)
  endif()
  include_directories(${Protobuf_INCLUDE_DIRS})
endif()

velox_set_source(simdjson)
if(WIN32)
  velox_resolve_dependency(simdjson)
else()
  velox_resolve_dependency(simdjson 4.1.0)
endif()

velox_set_source(FastFloat)
velox_resolve_dependency(FastFloat)

# stemmer (Snowball) is built from source via a Unix Makefile and is not
# available on Windows; the word_stem Presto function is disabled there
# (see velox/functions/prestosql/registration/StringFunctionsRegistration.cpp).
if(NOT WIN32)
  velox_set_source(stemmer)
  velox_resolve_dependency(stemmer)
endif()

# On Windows, manually include libevent targets to work around broken LibeventConfig.cmake
if(WIN32)
  set(_LIBEVENT_PREFIX "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}")
  if(EXISTS "${_LIBEVENT_PREFIX}/share/libevent/LibeventTargets-shared.cmake")
    # Set the import prefix before including the targets
    set(_IMPORT_PREFIX "${_LIBEVENT_PREFIX}")
    include("${_LIBEVENT_PREFIX}/share/libevent/LibeventTargets-shared.cmake")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      include("${_LIBEVENT_PREFIX}/share/libevent/LibeventTargets-shared-debug.cmake")
    else()
      include("${_LIBEVENT_PREFIX}/share/libevent/LibeventTargets-shared-release.cmake")
    endif()
    unset(_IMPORT_PREFIX)
    # Clear the import check for libevent targets to work around vcpkg folly packaging bug
    set(_cmake_import_check_files_for_libevent::core "")
    set(_cmake_import_check_files_for_libevent::extra "")
  endif()
  unset(_LIBEVENT_PREFIX)
endif()

velox_set_source(folly)
velox_resolve_dependency(folly)

# Fix folly's INTERFACE_COMPILE_OPTIONS on MSVC: it exports /std:c++17 which
# overrides our /std:c++20, breaking spaceship operators and std::span.
if(MSVC AND TARGET Folly::folly)
  get_target_property(_folly_opts Folly::folly INTERFACE_COMPILE_OPTIONS)
  if(_folly_opts)
    list(FILTER _folly_opts EXCLUDE REGEX "/std:c\\+\\+17")
    set_target_properties(Folly::folly PROPERTIES INTERFACE_COMPILE_OPTIONS "${_folly_opts}")
  endif()
  get_target_property(_folly_opts Folly::folly_deps INTERFACE_COMPILE_OPTIONS)
  if(_folly_opts)
    list(FILTER _folly_opts EXCLUDE REGEX "/std:c\\+\\+17")
    set_target_properties(Folly::folly_deps PROPERTIES INTERFACE_COMPILE_OPTIONS "${_folly_opts}")
  endif()
endif()

if(${VELOX_BUILD_TESTING})
  # Spark query runner depends on absl, gRPC, c-ares.
  # These are not needed on Windows (Spark runner is Linux-only) and gRPC's
  # FetchContent download fails in OneBranch network-isolated containers.
  if(NOT WIN32)
    velox_set_source(absl)
    velox_resolve_dependency(absl)

    # 'gRPC_CARES_PROVIDER' is set as 'package', which means c-ares library needs
    # to be installed on the system, instead of being built by gRPC.
    velox_set_source(c-ares)
    velox_resolve_dependency(c-ares)

    velox_set_source(gRPC)
    velox_resolve_dependency(gRPC)
  endif()
endif()

if(VELOX_ENABLE_REMOTE_FUNCTIONS OR VELOX_ENABLE_PARQUET)
  # TODO: Move this to use resolve_dependency(). For some reason, FBThrift
  # requires clients to explicitly install fizz and wangle.
  find_package(fizz CONFIG REQUIRED)
  find_package(wangle CONFIG REQUIRED)
  find_package(FBThrift CONFIG REQUIRED)

  # Workaround for a missing edge in the system-installed FBThrift's
  # exported CMake graph. At the pinned FB_OS_VERSION, fbthrift's own
  # `target_link_libraries(thriftmetadata PUBLIC ...)` block does not
  # declare an xxhash dep on `thriftmetadata` — the conditional
  # `if(Xxhash_FOUND) target_link_libraries(thriftmetadata PUBLIC
  # ${Xxhash_LIBRARY})` block was added later in fbthrift mainline.
  # Meanwhile `libthriftmetadata.a` itself does call XXH3_64bits(),
  # so the binary needs xxhash but the exported FBThriftTargets.cmake
  # doesn't declare it. Symptom (only with VELOX_BUILD_SHARED=ON +
  # VELOX_MONO_LIBRARY=ON, where libvelox.so doesn't drag xxhash in
  # incidentally): `undefined reference to symbol 'XXH3_64bits'` on
  # any test executable that links FBThrift through velox.
  # Tracking: https://github.com/facebook/fbthrift/issues/651,
  # https://github.com/facebookincubator/velox/issues/13153.
  # Remove once FB_OS_VERSION is bumped past the commit that adds the
  # `if(Xxhash_FOUND)` block AND the velox-dev container is rebuilt
  # against that fixed fbthrift release.
  if(TARGET FBThrift::thriftmetadata)
    find_library(VELOX_XXHASH_LIB xxhash REQUIRED)
    set_property(
      TARGET FBThrift::thriftmetadata
      APPEND
      PROPERTY INTERFACE_LINK_LIBRARIES "${VELOX_XXHASH_LIB}"
    )
  endif()
endif()

if(VELOX_ENABLE_GCS)
  velox_set_source(google_cloud_cpp_storage)
  velox_resolve_dependency(google_cloud_cpp_storage CONFIG 2.22.0 REQUIRED)
  add_definitions(-DVELOX_ENABLE_GCS)
endif()

# GCC needs to link a library to enable std::filesystem.
# GCC needs to enable co-routines. We only support compiler that
# support coroutines.
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  # Find Threads library
  find_package(Threads REQUIRED)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fcoroutines>)
  # Explicitly enable folly coroutines when compiler supports them.
  # This is required for folly::coro::AsyncGenerator to work correctly.
  add_compile_definitions(FOLLY_HAS_COROUTINES=1)
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "AppleClang")
  # Handle Apple toolchain Clang change of name of variable.
  # Remove when upgrading to the latest FBOS version where this has been addressed.
  add_compile_options(-D_LIBCPP_HAS_NO_ASAN)
endif()

if(VELOX_BUILD_TESTING AND NOT VELOX_ENABLE_DUCKDB)
  message(
    FATAL_ERROR
    "Unit tests require duckDB to be enabled (VELOX_ENABLE_DUCKDB=ON or VELOX_BUILD_TESTING=OFF)"
  )
endif()

set(VELOX_DISABLE_GOOGLETEST OFF)
if(NOT VELOX_BUILD_TEST_UTILS AND NOT VELOX_BUILD_TESTING)
  set(VELOX_DISABLE_GOOGLETEST ON)
  add_definitions(-DVELOX_DISABLE_GOOGLETEST)
endif()

# On macOS, search Homebrew for keg-only versions of Bison and Flex. Xcode does
# not provide new enough versions for us to use.
if(CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
  execute_process(
    COMMAND brew --prefix bison
    RESULT_VARIABLE BREW_BISON
    OUTPUT_VARIABLE BREW_BISON_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(BREW_BISON EQUAL 0 AND EXISTS "${BREW_BISON_PREFIX}")
    message(STATUS "Found Bison keg installed by Homebrew at ${BREW_BISON_PREFIX}")
    set(BISON_EXECUTABLE "${BREW_BISON_PREFIX}/bin/bison")
  endif()

  execute_process(
    COMMAND brew --prefix flex
    RESULT_VARIABLE BREW_FLEX
    OUTPUT_VARIABLE BREW_FLEX_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(BREW_FLEX EQUAL 0 AND EXISTS "${BREW_FLEX_PREFIX}")
    message(STATUS "Found Flex keg installed by Homebrew at ${BREW_FLEX_PREFIX}")
    set(FLEX_EXECUTABLE "${BREW_FLEX_PREFIX}/bin/flex")
    set(FLEX_INCLUDE_DIR "${BREW_FLEX_PREFIX}/include")
  endif()
endif()
find_package(BISON 3.0.4 QUIET)
find_package(FLEX 2.5.13 QUIET)
if(NOT BISON_FOUND OR NOT FLEX_FOUND)
  message(STATUS "Bison/Flex not found - will use pre-generated parser files")
endif()

# On Windows, set BISON_PKGDATADIR automatically if using WinFlexBison from WinGet
if(WIN32 AND BISON_EXECUTABLE)
  get_filename_component(BISON_BIN_DIR "${BISON_EXECUTABLE}" DIRECTORY)
  # Normalize path separators
  file(TO_CMAKE_PATH "${BISON_BIN_DIR}" BISON_BIN_DIR)
  string(FIND "${BISON_BIN_DIR}" "WinGet/Links" _is_bison_winget_link)
  if(_is_bison_winget_link GREATER -1)
    # Find the actual WinFlexBison package directory
    string(REPLACE "/Links" "/Packages" _winget_bison_packages_dir "${BISON_BIN_DIR}")
    file(GLOB _winget_bison_dirs "${_winget_bison_packages_dir}/WinFlexBison*")
    if(_winget_bison_dirs)
      list(GET _winget_bison_dirs 0 _winget_bison_dir)
      if(EXISTS "${_winget_bison_dir}/data")
        # Convert to native Windows path for environment variable
        file(TO_NATIVE_PATH "${_winget_bison_dir}/data" _bison_data_path)
        # Create a wrapper script that sets BISON_PKGDATADIR
        set(_bison_wrapper "${CMAKE_BINARY_DIR}/bison_wrapper.bat")
        file(WRITE "${_bison_wrapper}" "@echo off\nset BISON_PKGDATADIR=${_bison_data_path}\n\"${BISON_EXECUTABLE}\" %*\n")
        set(BISON_EXECUTABLE "${_bison_wrapper}" CACHE FILEPATH "Bison executable with PKGDATADIR wrapper" FORCE)
        message(STATUS "Auto-configured BISON_PKGDATADIR: ${_bison_data_path}")
        message(STATUS "Created bison wrapper: ${_bison_wrapper}")
      endif()
    endif()
  endif()
endif()

# On Windows, FLEX_INCLUDE_DIRS may not be set properly by FindFLEX.cmake
# We need to find FlexLexer.h which is typically in the same directory as win_flex.exe
if(WIN32 AND FLEX_FOUND)
  message(STATUS "FLEX_EXECUTABLE from find_package: ${FLEX_EXECUTABLE}")
  
  if(NOT FLEX_INCLUDE_DIRS)
    # If FLEX_EXECUTABLE is set, use it to find FlexLexer.h
    if(FLEX_EXECUTABLE)
      get_filename_component(FLEX_BIN_DIR "${FLEX_EXECUTABLE}" DIRECTORY)
      message(STATUS "FLEX_BIN_DIR: ${FLEX_BIN_DIR}")
      
      # On Windows with WinGet, FLEX_EXECUTABLE may be a symlink in the Links directory
      # CMake's REALPATH doesn't resolve Windows symlinks properly, so we check for WinGet pattern
      string(FIND "${FLEX_BIN_DIR}" "WinGet/Links" _is_winget_link)
      if(_is_winget_link GREATER -1 AND WIN32)
        # Try to find the actual WinFlexBison package directory
        string(REPLACE "/Links" "/Packages" _winget_packages_dir "${FLEX_BIN_DIR}")
        file(GLOB _winget_flex_dirs "${_winget_packages_dir}/WinFlexBison*")
        if(_winget_flex_dirs)
          list(GET _winget_flex_dirs 0 _winget_flex_dir)
          message(STATUS "Found WinGet WinFlexBison package at: ${_winget_flex_dir}")
          if(EXISTS "${_winget_flex_dir}/FlexLexer.h")
            set(FLEX_INCLUDE_DIRS "${_winget_flex_dir}")
            message(STATUS "Found FlexLexer.h in WinGet package: ${FLEX_INCLUDE_DIRS}")
          endif()
        endif()
      endif()
      
      # If not found yet, check common locations relative to flex executable
      if(NOT FLEX_INCLUDE_DIRS)
        set(_flex_include_candidates
          "${FLEX_BIN_DIR}"
          "${FLEX_BIN_DIR}/.."
          "${FLEX_BIN_DIR}/../include"
        )
        
        foreach(_dir ${_flex_include_candidates})
          if(EXISTS "${_dir}/FlexLexer.h")
            get_filename_component(FLEX_INCLUDE_DIRS "${_dir}" ABSOLUTE)
            message(STATUS "Found FlexLexer.h at: ${FLEX_INCLUDE_DIRS}")
            break()
          endif()
        endforeach()
      endif()
    endif()
    
    # If still not found, search in WinFlexBison winget installation
    if(NOT FLEX_INCLUDE_DIRS AND DEFINED ENV{BISON_PKGDATADIR})
      get_filename_component(_bison_pkg_dir "$ENV{BISON_PKGDATADIR}" DIRECTORY)
      if(EXISTS "${_bison_pkg_dir}/FlexLexer.h")
        set(FLEX_INCLUDE_DIRS "${_bison_pkg_dir}")
        message(STATUS "Found FlexLexer.h via BISON_PKGDATADIR: ${FLEX_INCLUDE_DIRS}")
      endif()
    endif()
    
    if(NOT FLEX_INCLUDE_DIRS)
      message(FATAL_ERROR "Could not find FlexLexer.h. Please set FLEX_INCLUDE_DIRS manually.")
    endif()
  endif()
endif()

# Fallback for ARM64 cross-compilation: find FlexLexer.h when FLEX is disabled
# Pre-generated parser files still need this header
if(WIN32 AND NOT FLEX_FOUND AND NOT FLEX_INCLUDE_DIRS)
  # Search common WinFlexBison locations
  set(_flex_search_paths
    "C:/winflexbison"
    "C:/winflexbison/data"
  )
  # Also check relative to FLEX_EXECUTABLE if it was passed via cmake
  if(FLEX_EXECUTABLE)
    get_filename_component(_flex_exe_dir "${FLEX_EXECUTABLE}" DIRECTORY)
    list(APPEND _flex_search_paths "${_flex_exe_dir}")
  endif()
  # Check WinGet installation
  file(GLOB _winget_flex_dirs "$ENV{LOCALAPPDATA}/Microsoft/WinGet/Packages/WinFlexBison*")
  foreach(_d ${_winget_flex_dirs})
    list(APPEND _flex_search_paths "${_d}" "${_d}/data")
  endforeach()
  foreach(_dir ${_flex_search_paths})
    if(EXISTS "${_dir}/FlexLexer.h")
      set(FLEX_INCLUDE_DIRS "${_dir}")
      message(STATUS "Found FlexLexer.h (fallback): ${FLEX_INCLUDE_DIRS}")
      include_directories(${FLEX_INCLUDE_DIRS})
      break()
    endif()
  endforeach()
  if(NOT FLEX_INCLUDE_DIRS)
    message(WARNING "FlexLexer.h not found - parser compilation may fail")
  endif()
endif()

# Ensure FLEX_INCLUDE_DIRS is on the include path even when FindFLEX is disabled
# (e.g., ARM64 cross-compilation passes FLEX_INCLUDE_DIRS via cmake -D)
if(WIN32 AND NOT FLEX_FOUND AND FLEX_INCLUDE_DIRS)
  include_directories(${FLEX_INCLUDE_DIRS})
  message(STATUS "Added FLEX_INCLUDE_DIRS to global includes: ${FLEX_INCLUDE_DIRS}")
endif()

find_package(double-conversion 3.1.5 REQUIRED)

include_directories(SYSTEM velox)
include_directories(SYSTEM velox/external)

if(NOT VELOX_DISABLE_GOOGLETEST)
  velox_set_source(GTest)
  velox_resolve_dependency(GTest)
  set(VELOX_GTEST_INCUDE_DIR "${gtest_SOURCE_DIR}/include" PARENT_SCOPE)
  foreach(tgt gtest gtest_main gmock gmock_main)
    if(NOT TARGET ${tgt} AND TARGET GTest::${tgt})
      add_library(${tgt} INTERFACE IMPORTED)
      target_link_libraries(${tgt} INTERFACE GTest::${tgt})
    endif()
  endforeach(tgt)
endif()

velox_set_source(xsimd)
velox_resolve_dependency(xsimd 13.2.0)

# ARM64 Windows: xsimd 13.2.0's NEON backend is incompatible with MSVC
# (all NEON types are __n128 typedef, breaking template specialization).
# Use the emulated (scalar) backend instead.
if(VELOX_TARGET_IS_ARM64 AND WIN32)
  # Disable NEON detection so xsimd doesn't try the broken NEON backend
  add_compile_options(/U__ARM_NEON)
  # Suppress C4716: functions that throw but have no return (MSVC false positive)
  add_compile_options(/wd4716)
  # Reduce object file size to fit within 4GB static lib limit
  add_compile_options(/Gy /Gw)
  add_compile_definitions(XSIMD_WITH_EMULATED=1)
  # Force-include a header that sets XSIMD_DEFAULT_ARCH (angle brackets
  # in the type make command-line -D unreliable)
  set(_xsimd_default_arch_header "${CMAKE_BINARY_DIR}/xsimd_default_arch_override.h")
  file(WRITE "${_xsimd_default_arch_header}"
    "#pragma once\n"
    "#define XSIMD_DEFAULT_ARCH xsimd::emulated<128>\n"
  )
  add_compile_options("/FI${_xsimd_default_arch_header}")
  # Patch xsimd_config.hpp to include XSIMD_WITH_EMULATED in the supported
  # architecture check, so batch/api headers are not skipped.
  set(_xsimd_config "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/xsimd/config/xsimd_config.hpp")
  if(EXISTS "${_xsimd_config}")
    file(READ "${_xsimd_config}" _xsimd_config_content)
    string(REPLACE
      "!XSIMD_WITH_NEON && !XSIMD_WITH_NEON64 && !XSIMD_WITH_SVE && !XSIMD_WITH_RVV && !XSIMD_WITH_WASM"
      "!XSIMD_WITH_NEON && !XSIMD_WITH_NEON64 && !XSIMD_WITH_SVE && !XSIMD_WITH_RVV && !XSIMD_WITH_WASM && !XSIMD_WITH_EMULATED"
      _xsimd_config_content "${_xsimd_config_content}")
    file(WRITE "${_xsimd_config}" "${_xsimd_config_content}")
  endif()
  # Patch Arrow's bit_util.h: __popcnt/__popcnt64 are x86-only intrinsics.
  # On ARM64 MSVC, use _CountOneBits/_CountOneBits64 instead.
  set(_arrow_bit_util "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/arrow/util/bit_util.h")
  if(EXISTS "${_arrow_bit_util}")
    file(READ "${_arrow_bit_util}" _arrow_bit_util_content)
    string(REPLACE
      "#define ARROW_POPCOUNT64 __popcnt64\n#define ARROW_POPCOUNT32 __popcnt"
      "#if defined(_M_ARM64)\n#define ARROW_POPCOUNT64 _CountOneBits64\n#define ARROW_POPCOUNT32(x) ((int)_CountOneBits((unsigned long)(x)))\n#else\n#define ARROW_POPCOUNT64 __popcnt64\n#define ARROW_POPCOUNT32 __popcnt\n#endif"
      _arrow_bit_util_content "${_arrow_bit_util_content}")
    file(WRITE "${_arrow_bit_util}" "${_arrow_bit_util_content}")
  endif()
endif()

if(VELOX_BUILD_TESTING)
  set(BUILD_TESTING ON)
  include(CTest) # include after project() but before add_subdirectory()
endif()

include_directories(.)

# MSVC: Large translation units need /bigobj to exceed COFF section limit
if(MSVC)
  add_compile_options(/bigobj)
  # MSVC reports C4716 ("must return a value") as an error for functions that
  # only ever throw via VELOX_UNSUPPORTED/VELOX_FAIL/VELOX_UNREACHABLE because it
  # does not always honor the [[noreturn]] on the throw helper across template
  # instantiations. The core library masks this via /w; test targets compile at
  # /W3, so suppress it globally here. POSIX builds are unaffected.
  add_compile_options(/wd4716)
endif()

# On Windows, add vcpkg include directory globally to ensure all targets
# (including standalone non-mono targets like velox_profiler) can find
# vcpkg-installed headers (folly, fmt, etc.)
if(WIN32 AND DEFINED VCPKG_TARGET_TRIPLET)
  include_directories(SYSTEM "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include")
endif()

# Install all headers for vcpkg packaging
# Always install headers (both mono and non-mono builds)
install(DIRECTORY velox/
        DESTINATION include/velox
        FILES_MATCHING PATTERN "*.h"
        PATTERN "test" EXCLUDE
        PATTERN "tests" EXCLUDE
        PATTERN "benchmarks" EXCLUDE
        PATTERN "examples" EXCLUDE
        PATTERN "fuzzer" EXCLUDE
        PATTERN "tool" EXCLUDE)

if(VELOX_MONO_LIBRARY)
  # Generate and install CMake config files
  include(CMakePackageConfigHelpers)

  # Create velox-windows-config.cmake
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/windows/vcpkg-port/velox-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/velox-windows-config.cmake"
    INSTALL_DESTINATION lib/cmake/velox-windows
  )

  # Install the config file
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/velox-windows-config.cmake"
          DESTINATION lib/cmake/velox-windows)

  # Install the main library file
  # The velox mono-library is built as a CMake object library, so we install the .lib file manually
  install(FILES "${CMAKE_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE}/velox.lib"
          DESTINATION lib)
endif()

# TODO: Include all other installation files. For now just making sure this
# generates an installable makefile.
install(FILES velox/type/Type.h DESTINATION "include/velox")

# Adding this down here prevents warnings in dependencies from stopping the
# build
if("${TREAT_WARNINGS_AS_ERRORS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

# Enable the ASAN and UBSAN sanitzers for Clang 20 and above.
if(
  VELOX_ENABLE_ASAN_UBSAN_SANITIZERS
  AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 20
)
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -O1 -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -DVELOX_ENABLE_ASAN_UBSAN_SANITIZERS"
  )
  # Skip alignment errors and member access check for now.
  # https://github.com/facebookincubator/velox/issues/15811
  # https://github.com/facebookincubator/velox/issues/15810
  set(
    CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined -fno-sanitize=alignment -fno-sanitize=vptr"
  )
endif()

message("FINAL CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
message("FINAL CMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}")

if(VELOX_ENABLE_ARROW)
  # Force using vcpkg Arrow on Windows instead of building from source
  if(WIN32)
    set(Arrow_SOURCE "SYSTEM")
    # Find utf8proc first to prevent Arrow from creating duplicate target
    find_package(utf8proc CONFIG QUIET)
  endif()
  velox_set_source(Arrow)
  velox_resolve_dependency(Arrow)
endif()

if(VELOX_ENABLE_GEO)
  velox_set_source(geos)
  velox_resolve_dependency(geos)
endif()

if(VELOX_ENABLE_GEO)
  list(PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake/resolve_dependency_modules/s2geometry)
  velox_set_source(s2geometry)
  velox_resolve_dependency(s2geometry)
endif()

add_subdirectory(velox)

