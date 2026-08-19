# Copyright (c) Meta Platforms, Inc. and affiliates.
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

# Nimble is reached only through velox/dwio/CMakeLists.txt when
# VELOX_ENABLE_NIMBLE=ON. FlatBuffers, OpenZL and the vendored FSST are all
# resolved by the top-level Velox CMakeLists before this point, so this file
# only configures what is specific to Nimble.

# FsstEncoding.h includes <fsst.h>, and DuckDB vendors a header of the same name.
# Resolved BUNDLED, DuckDB contributes its copy to Velox targets through the link
# interface, which CMake emits after a target's own include directories. Adding
# the vendored copy at directory scope therefore puts it first for every target
# in this subtree, including those that only pull FsstEncoding.h in transitively
# and never link fsst. Relies on the fsst target exporting this path as a plain
# -I: while it was SYSTEM, CMake de-duplicated the two and kept the -isystem,
# which is searched after every -I.
include_directories(BEFORE "${CMAKE_SOURCE_DIR}/velox/external/fsst")

# Nimble code expects an upper case suffix to the generated file.
set(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS "--filename-suffix" "Generated")

# Velox only resolves Abseil for its own test targets, but Nimble's field reader
# needs absl::flat_hash_map unconditionally.
if(NOT TARGET absl::flat_hash_map)
  velox_set_source(absl)
  velox_resolve_dependency(absl)
endif()

# Nimble's OSS build has no access to the Meta-internal compressor. This is read
# by widely included headers (compression/CompressionPolicy.h,
# encodings/legacy/EncodingSelectionPolicy.h), so every translation unit that
# sees those headers must agree on it.
add_compile_definitions(DISABLE_META_INTERNAL_COMPRESSOR=1)

option(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS "Enable experimental encodings" OFF)
if(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
  add_compile_definitions(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
endif()

if(VELOX_MONO_LIBRARY AND TARGET velox)
  # add_compile_definitions() above sets a directory property, which only
  # reaches targets created in this directory tree. Under VELOX_MONO_LIBRARY
  # Nimble's sources are folded into the single `velox` target, created
  # elsewhere, so the definitions have to be applied to it explicitly.
  target_compile_definitions(velox PUBLIC DISABLE_META_INTERNAL_COMPRESSOR=1)
  if(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
    target_compile_definitions(velox PUBLIC NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
  endif()
endif()

# Nimble's tests follow Velox's testing switch. The separate variable is kept so
# a build can opt into Nimble's tests alone.
if(NOT DEFINED NIMBLE_BUILD_TESTING)
  set(NIMBLE_BUILD_TESTING ${VELOX_BUILD_TESTING})
endif()

# Ordered by dependency: common is the root, and the tablet, index and velox
# layers reference each other's targets at declaration time.
if(VELOX_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
add_subdirectory(common)
add_subdirectory(compression)
add_subdirectory(encodings)
add_subdirectory(index)
add_subdirectory(tablet)
add_subdirectory(velox)
add_subdirectory(writer)
add_subdirectory(serializer)
add_subdirectory(tools)

if(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
  add_subdirectory(ML_ID_Compression)
endif()

if(NIMBLE_BUILD_TESTING)
  add_subdirectory(fuzzer)
endif()
