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
add_library(nimble_velox_common SchemaUtils.cpp)
target_link_libraries(
  nimble_velox_common
  nimble_common
  velox_type
  Folly::folly
  fmt::fmt
)

add_library(nimble_velox_schema SchemaTypes.cpp)
target_link_libraries(nimble_velox_schema nimble_common Folly::folly)

add_library(nimble_velox_schema_reader SchemaReader.cpp)
target_link_libraries(
  nimble_velox_schema_reader
  nimble_velox_schema
  nimble_common
  Folly::folly
)

add_library(nimble_velox_schema_builder SchemaBuilder.cpp)
target_link_libraries(
  nimble_velox_schema_builder
  nimble_velox_schema_reader
  nimble_velox_schema
  nimble_common
  Folly::folly
)

add_library(nimble_velox_stream_data StreamData.cpp)
target_link_libraries(
  nimble_velox_stream_data
  nimble_velox_schema_builder
  nimble_common
)

add_library(nimble_velox_field_reader FieldReader.cpp)
target_link_libraries(
  nimble_velox_field_reader
  nimble_encodings_legacy
  nimble_velox_schema_reader
  nimble_common
  Folly::folly
  absl::flat_hash_map
  protobuf::libprotobuf
)

add_library(nimble_velox_layout_planner LayoutPlanner.cpp)
target_link_libraries(
  nimble_velox_layout_planner
  nimble_velox_schema_reader
  nimble_tablet_writer
)

add_library(
  nimble_velox_field_writer
  BufferGrowthPolicy.cpp
  DeduplicationUtils.cpp
  FieldWriter.cpp
)
target_link_libraries(
  nimble_velox_field_writer
  nimble_column_stats
  nimble_column_stats_utils
  nimble_velox_schema
  nimble_velox_stream_data
  nimble_velox_schema_builder
  velox_test_util
  Folly::folly
)

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
# "velox/dwio/nimble/velox/SchemaGenerated.h". So we need
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
  INTERFACE ${FLATBUFFERS_INCLUDE_DIR}
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
  INTERFACE ${FLATBUFFERS_INCLUDE_DIR}
)
add_dependencies(nimble_velox_stats_fb nimble_velox_stats_schema_fb)

add_library(nimble_velox_schema_serialization SchemaSerialization.cpp)
target_link_libraries(
  nimble_velox_schema_serialization
  nimble_velox_schema_reader
  nimble_velox_schema_builder
  nimble_velox_schema_fb
)

add_library(
  nimble_velox_reader
  ChunkedStream.cpp
  ChunkedStreamDecoder.cpp
  StreamLabels.cpp
  VeloxReader.cpp
)
target_link_libraries(
  nimble_velox_reader
  nimble_velox_common
  nimble_velox_schema
  nimble_velox_schema_serialization
  nimble_velox_schema_fb
  nimble_velox_metadata_fb
  nimble_velox_field_reader
  nimble_tablet_reader
  nimble_common
  nimble_encodings_legacy
  velox_common_base
  Folly::folly
)

add_library(raw_size_utils RawSizeUtils.cpp DecodedVectorManager.cpp)
target_link_libraries(
  raw_size_utils
  nimble_common
  velox_vector
  velox_dwio_common
)

add_subdirectory(index)
add_subdirectory(selective)
add_subdirectory(stats)
add_subdirectory(tests)
