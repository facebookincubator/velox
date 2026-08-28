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
if(NIMBLE_BUILD_TESTING)
  add_subdirectory(tests)
endif()

add_library(
  nimble_common
  Buffer.cpp
  Checksum.cpp
  DataTypeDispatch.h
  Exceptions.cpp
  FeatureGate.cpp
  FixedBitArray.cpp
  MetricsLogger.cpp
  NimbleException.cpp
  Types.cpp
  Varint.cpp
  Zigzag.h
  Vector.h
  Varint.h
  Types.h
  StatsUtil.h
  NimbleException.h
  MetricsLogger.h
  FixedBitArray.h
  FeatureGate.h
  Exceptions.h
  ExceptionHelper.h
  Constants.h
  ChunkHeader.h
  Checksum.h
  Buffer.h
  BitEncoder.h
)

target_link_libraries(nimble_common velox_common_base velox_memory velox_exception Folly::folly)

# Moved from velox/CMakeLists.txt when the nimble/velox/ folder was flattened.

add_library(nimble_velox_common SchemaUtils.cpp SchemaUtils.h)
target_link_libraries(nimble_velox_common nimble_common velox_type Folly::folly fmt::fmt)

add_library(
  nimble_velox_shared_dictionary_config
  SharedDictionaryConfig.cpp
  SharedDictionaryConfig.h
)
target_link_libraries(
  nimble_velox_shared_dictionary_config
  nimble_common
  nimble_encodings
  nimble_tablet_reader
  velox_type
)

add_library(nimble_velox_schema SchemaTypes.cpp SchemaTypes.h)
target_link_libraries(nimble_velox_schema nimble_common Folly::folly)

add_library(nimble_velox_schema_reader SchemaReader.cpp SchemaReader.h)
target_link_libraries(nimble_velox_schema_reader nimble_velox_schema nimble_common Folly::folly)

build_flatbuffers(
  "${CMAKE_CURRENT_SOURCE_DIR}/Schema.fbs"
  ""
  nimble_velox_schema_schema_fb
  ""
  "${CMAKE_CURRENT_BINARY_DIR}"
  ""
  ""
)
add_library(nimble_velox_schema_fb INTERFACE)
# build_flatbuffers() assumes that we include generated .h files by
# #include "SchemaGenerated.h" but we want to use #include
# "velox/dwio/nimble/common/SchemaGenerated.h". So we need
# ${PROJECT_BINARY_DIR} here.
target_include_directories(
  nimble_velox_schema_fb
  INTERFACE ${PROJECT_BINARY_DIR} ${FLATBUFFERS_INCLUDE_DIR}
)
add_dependencies(nimble_velox_schema_fb nimble_velox_schema_schema_fb)

build_flatbuffers(
  "${CMAKE_CURRENT_SOURCE_DIR}/Metadata.fbs"
  ""
  nimble_velox_metadata_schema_fb
  ""
  "${CMAKE_CURRENT_BINARY_DIR}"
  ""
  ""
)
add_library(nimble_velox_metadata_fb INTERFACE)
target_include_directories(
  nimble_velox_metadata_fb
  INTERFACE ${PROJECT_BINARY_DIR} ${FLATBUFFERS_INCLUDE_DIR}
)
add_dependencies(nimble_velox_metadata_fb nimble_velox_metadata_schema_fb)

build_flatbuffers(
  "${CMAKE_CURRENT_SOURCE_DIR}/Stats.fbs"
  ""
  nimble_velox_stats_schema_fb
  ""
  "${CMAKE_CURRENT_BINARY_DIR}"
  ""
  ""
)
add_library(nimble_velox_stats_fb INTERFACE)
target_include_directories(
  nimble_velox_stats_fb
  INTERFACE ${PROJECT_BINARY_DIR} ${FLATBUFFERS_INCLUDE_DIR}
)
add_dependencies(nimble_velox_stats_fb nimble_velox_stats_schema_fb)

add_library(nimble_velox_schema_serialization SchemaSerialization.cpp SchemaSerialization.h)
target_link_libraries(
  nimble_velox_schema_serialization
  nimble_velox_schema_reader
  nimble_velox_schema_builder
  nimble_velox_schema_fb
)

add_library(nimble_velox_schema_builder SchemaBuilder.cpp SchemaBuilder.h)
target_link_libraries(
  nimble_velox_schema_builder
  nimble_velox_schema_reader
  nimble_velox_schema
  nimble_common
  Folly::folly
)
