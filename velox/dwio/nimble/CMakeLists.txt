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

# Nimble has two entry points: the Velox build reaches it through
# velox/dwio/CMakeLists.txt when VELOX_ENABLE_NIMBLE=ON, and the standalone
# Nimble repository builds Velox as a submodule and adds this directory
# directly. Dependency resolution and option handling therefore live here, so
# both entry points configure Nimble identically.

# FlatBuffers is not a Velox dependency; Nimble is its only consumer.
find_package(flatbuffers QUIET)
set(NIMBLE_NEED_BUNDLED_BUILD_FLAT_BUFFERS FALSE)
if(flatbuffers_FOUND)
  if(flatbuffers_VERSION VERSION_LESS "22.9.4")
    set(NIMBLE_NEED_BUNDLED_BUILD_FLAT_BUFFERS TRUE)
  endif()
else()
  # Fallback to old FlatBuffers (< 2.0.0), which used a capitalised name.
  find_package(Flatbuffers QUIET)
  if(NOT Flatbuffers_FOUND)
    message(
      FATAL_ERROR
      "Nimble requires FlatBuffers, which was not found. Install it (the "
      "scripts/setup-*.sh scripts do this) or reconfigure with "
      "-DVELOX_ENABLE_NIMBLE=OFF."
    )
  endif()
  set(NIMBLE_NEED_BUNDLED_BUILD_FLAT_BUFFERS TRUE)
endif()
if(NIMBLE_NEED_BUNDLED_BUILD_FLAT_BUFFERS)
  # Old FlatBuffers (< 22.9.4) doesn't provide build_flatbuffers(). So we will
  # use the bundled one. We can remove CMake/BuildFlatBuffers once we drop
  # support for FlatBuffers < 22.9.4.
  include(BuildFlatBuffers)
endif()
# Nimble code expects an upper case suffix to the generated file.
set(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS "--filename-suffix" "Generated")

# Velox only resolves Abseil for its own test targets, but Nimble's field reader
# needs absl::flat_hash_map unconditionally.
if(NOT TARGET absl::flat_hash_map)
  velox_set_source(absl)
  velox_resolve_dependency(absl)
endif()

# The standalone Nimble build vendors OpenZL as a git submodule and has already
# added it by this point; the Velox build fetches it instead.
if(NOT TARGET openzl_cpp)
  velox_set_source(openzl)
  velox_resolve_dependency(openzl)
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

# Nimble builds its own tests even when the surrounding Velox build has testing
# disabled: the standalone repo sets VELOX_BUILD_MINIMAL_WITH_DWIO=ON, which
# forces VELOX_BUILD_TESTING=OFF, yet still needs `make unittest` to work.
if(NOT DEFINED NIMBLE_BUILD_TESTING)
  set(NIMBLE_BUILD_TESTING ${VELOX_BUILD_TESTING})
endif()

# Ordered by dependency: common is the root, and the tablet, index and velox
# layers reference each other's targets at declaration time.
add_subdirectory(benchmarks)
add_subdirectory(common)
add_subdirectory(compression)
add_subdirectory(encodings)
add_subdirectory(index)
add_subdirectory(tablet)
add_subdirectory(velox)
add_subdirectory(writer)
add_subdirectory(serializer)
add_subdirectory(tools)

if(NIMBLE_BUILD_TESTING)
  add_subdirectory(fuzzer)
endif()
