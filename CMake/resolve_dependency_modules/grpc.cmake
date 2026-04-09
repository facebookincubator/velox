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
include_guard(GLOBAL)

velox_set_source(absl)
velox_resolve_dependency(absl CONFIG REQUIRED)

# 1.48.x does not build codegen against Protobuf 6+ (absl::string_view APIs,
# header moves, protobuf::util::Status). Use a release aligned with Protobuf
# 25+ for conda/system protobuf package builds.
set(VELOX_GRPC_BUILD_VERSION 1.66.2)
set(
  VELOX_GRPC_BUILD_SHA256_CHECKSUM
  1343e2d0c4cbd36cbfbbe4c7305a5529a7a044212c57b9dbfd929a6ceda285f4
)
string(
  CONCAT
  VELOX_GRPC_SOURCE_URL
  "https://github.com/grpc/grpc/archive/refs/tags/"
  "v${VELOX_GRPC_BUILD_VERSION}.tar.gz"
)

velox_resolve_dependency_url(GRPC)

message(STATUS "Building gRPC from source")

FetchContent_Declare(
  gRPC
  URL ${VELOX_GRPC_SOURCE_URL}
  URL_HASH ${VELOX_GRPC_BUILD_SHA256_CHECKSUM}
  OVERRIDE_FIND_PACKAGE
  EXCLUDE_FROM_ALL
)

# We need to specify CACHE explicitly even when we have
# set(CMAKE_POLICY_DEFAULT_CMP0077 NEW). Because gRPC doesn't use option(). gRPC
# uses set(... CACHE). So CMP0077 isn't affected.
set(gRPC_ABSL_PROVIDER "package" CACHE STRING "Provider of absl library")
set(gRPC_ZLIB_PROVIDER "package" CACHE STRING "Provider of zlib library")
set(gRPC_CARES_PROVIDER "package" CACHE STRING "Provider of c-ares library")
set(gRPC_RE2_PROVIDER "package" CACHE STRING "Provider of re2 library")
set(gRPC_SSL_PROVIDER "package" CACHE STRING "Provider of ssl library")
set(gRPC_PROTOBUF_PROVIDER "package" CACHE STRING "Provider of protobuf library")
set(gRPC_INSTALL ON CACHE BOOL "Generate installation target")
FetchContent_MakeAvailable(gRPC)

# Velox builds with strict warnings; newer GCC (e.g. 14+) treats gRPC's
# GPR_ATTRIBUTE_ALWAYS_INLINE_FUNCTION without `inline` as -Wattributes, and
# flags some chttp2 slice writes as -Wstringop-overflow. gRPC 1.66 also hits
# Abseil deprecation warnings with newer conda Abseil (-Wdeprecated-declarations).
# Those are third-party noise when -Werror is inherited from the parent project.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  foreach(_velox_grpc_tgt IN ITEMS grpc "grpc++" gpr)
    if(TARGET ${_velox_grpc_tgt})
      target_compile_options(
        ${_velox_grpc_tgt}
        PRIVATE
        -Wno-error=attributes
        -Wno-error=stringop-overflow
        -Wno-error=deprecated-declarations
      )
    endif()
  endforeach()
endif()

# With high -j, gRPC can compile gpr / codegen before Abseil or protobuf headers
# are available in the build interface. Serialize against those providers.
if(TARGET gpr)
  foreach(
    _velox_grpc_absl
    absl::base
    absl::strings
    absl::str_format
    absl::status
    absl::time
    absl::synchronization
    absl::flat_hash_map
    absl::optional
    absl::variant
  )
    if(TARGET ${_velox_grpc_absl})
      add_dependencies(gpr ${_velox_grpc_absl})
    endif()
  endforeach()
endif()
if(TARGET grpc_plugin_support)
  if(TARGET protobuf::libprotobuf)
    add_dependencies(grpc_plugin_support protobuf::libprotobuf)
  elseif(TARGET libprotobuf)
    add_dependencies(grpc_plugin_support libprotobuf)
  endif()
endif()

add_library(gRPC::grpc ALIAS grpc)
add_library(gRPC::grpc++ ALIAS grpc++)
add_executable(gRPC::grpc_cpp_plugin ALIAS grpc_cpp_plugin)
