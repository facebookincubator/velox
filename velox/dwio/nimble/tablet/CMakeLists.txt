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

build_flatbuffers(
  "${CMAKE_CURRENT_SOURCE_DIR}/Footer.fbs"
  ""
  nimble_footer_schema_fb
  ""
  "${CMAKE_CURRENT_BINARY_DIR}"
  ""
  ""
)
add_library(nimble_footer_fb INTERFACE)
# build_flatbuffers() assumes that we include generated .h files by
# #include "FooterGenerated.h" but we want to use #include
# "velox/dwio/nimble/tablet/FooterGenerated.h". So we need
# ${PROJECT_BINARY_DIR} here.
target_include_directories(
  nimble_footer_fb
  INTERFACE ${PROJECT_BINARY_DIR} ${FLATBUFFERS_INCLUDE_DIR}
)
add_dependencies(nimble_footer_fb nimble_footer_schema_fb)

build_flatbuffers(
  "${CMAKE_CURRENT_SOURCE_DIR}/ClusterIndex.fbs"
  ""
  nimble_cluster_index_schema_fb
  ""
  "${CMAKE_CURRENT_BINARY_DIR}"
  ""
  ""
)
add_library(nimble_cluster_index_fb INTERFACE)
target_include_directories(
  nimble_cluster_index_fb
  INTERFACE ${PROJECT_BINARY_DIR} ${FLATBUFFERS_INCLUDE_DIR}
)
add_dependencies(nimble_cluster_index_fb nimble_cluster_index_schema_fb)

build_flatbuffers(
  "${CMAKE_CURRENT_SOURCE_DIR}/Index.fbs"
  ""
  nimble_index_schema_fb
  ""
  "${CMAKE_CURRENT_BINARY_DIR}"
  ""
  ""
)
add_library(nimble_index_fb INTERFACE)
target_include_directories(
  nimble_index_fb
  INTERFACE ${PROJECT_BINARY_DIR} ${FLATBUFFERS_INCLUDE_DIR}
)
add_dependencies(nimble_index_fb nimble_index_schema_fb)

build_flatbuffers(
  "${CMAKE_CURRENT_SOURCE_DIR}/ChunkStats.fbs"
  ""
  nimble_chunk_stats_schema_fb
  ""
  "${CMAKE_CURRENT_BINARY_DIR}"
  ""
  ""
)
add_library(nimble_chunk_stats_fb INTERFACE)
target_include_directories(
  nimble_chunk_stats_fb
  INTERFACE ${PROJECT_BINARY_DIR} ${FLATBUFFERS_INCLUDE_DIR}
)
add_dependencies(nimble_chunk_stats_fb nimble_chunk_stats_schema_fb)

build_flatbuffers(
  "${CMAKE_CURRENT_SOURCE_DIR}/BloomFilter.fbs"
  ""
  nimble_bloom_filter_schema_fb
  ""
  "${CMAKE_CURRENT_BINARY_DIR}"
  ""
  ""
)
add_library(nimble_bloom_filter_fb INTERFACE)
target_include_directories(
  nimble_bloom_filter_fb
  INTERFACE ${PROJECT_BINARY_DIR} ${FLATBUFFERS_INCLUDE_DIR}
)
add_dependencies(nimble_bloom_filter_fb nimble_bloom_filter_schema_fb)

build_flatbuffers(
  "${CMAKE_CURRENT_SOURCE_DIR}/Features.fbs"
  ""
  nimble_features_schema_fb
  ""
  "${CMAKE_CURRENT_BINARY_DIR}"
  ""
  ""
)
add_library(nimble_features_fb INTERFACE)
target_include_directories(
  nimble_features_fb
  INTERFACE ${PROJECT_BINARY_DIR} ${FLATBUFFERS_INCLUDE_DIR}
)
add_dependencies(nimble_features_fb nimble_features_schema_fb)

build_flatbuffers(
  "${CMAKE_CURRENT_SOURCE_DIR}/HashIndex.fbs"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  nimble_hash_index_schema_fb
  ""
  "${CMAKE_CURRENT_BINARY_DIR}"
  ""
  ""
)
add_library(nimble_hash_index_fb INTERFACE)
target_include_directories(
  nimble_hash_index_fb
  INTERFACE ${PROJECT_BINARY_DIR} ${FLATBUFFERS_INCLUDE_DIR}
)
add_dependencies(nimble_hash_index_fb nimble_hash_index_schema_fb)

build_flatbuffers(
  "${CMAKE_CURRENT_SOURCE_DIR}/SortedIndex.fbs"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  nimble_sorted_index_schema_fb
  ""
  "${CMAKE_CURRENT_BINARY_DIR}"
  ""
  ""
)
add_library(nimble_sorted_index_fb INTERFACE)
target_include_directories(
  nimble_sorted_index_fb
  INTERFACE ${PROJECT_BINARY_DIR} ${FLATBUFFERS_INCLUDE_DIR}
)
add_dependencies(nimble_sorted_index_fb nimble_sorted_index_schema_fb)

add_library(
  nimble_tablet_common
  Compression.cpp
  FileFeatures.cpp
  MetadataBuffer.cpp
  MetadataInput.cpp
  Postscript.cpp
)
target_link_libraries(
  nimble_tablet_common
  nimble_features_fb
  nimble_footer_fb
  nimble_common
  velox_caching
  velox_file
  velox_memory
  Folly::folly
)

add_library(nimble_data_input DataInput.cpp)
target_link_libraries(
  nimble_data_input
  nimble_common
  velox_memory
  velox_file
  velox_time
  Folly::folly
)

add_library(
  nimble_tablet_reader
  FileLayout.cpp
  StripeGroup.cpp
  TabletReader.cpp
)
target_link_libraries(
  nimble_tablet_reader
  PUBLIC
    nimble_encodings
    nimble_index
    nimble_tablet_common
    nimble_cluster_index_fb
    nimble_chunk_stats_fb
    nimble_hash_index_fb
)

add_library(nimble_tablet_reader_cache TabletReaderCache.cpp)
target_link_libraries(
  nimble_tablet_reader_cache
  PUBLIC
    nimble_tablet_reader
    nimble_velox_schema_reader
    velox_caching
    velox_memory
    velox_type
  PRIVATE
    nimble_common
    nimble_velox_common
    nimble_velox_schema_serialization
    glog::glog
    Folly::folly
)

add_library(nimble_tablet_writer ChunkStatsWriter.cpp TabletWriter.cpp)
target_link_libraries(
  nimble_tablet_writer
  nimble_encodings
  nimble_tablet_common
  nimble_encodings
  nimble_cluster_index_fb
  nimble_chunk_stats_fb
)

if(NIMBLE_BUILD_TESTING)
  add_subdirectory(tests)
endif()
