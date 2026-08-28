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
add_library(
  nimble_index
  BloomFilter.cpp
  ChunkStats.cpp
  ChunkStatsGroup.cpp
  ClusterIndex.cpp
  ClusterIndexFactory.cpp
  ClusterIndexWriter.cpp
  DenseIndexFactory.cpp
  DenseIndexRegistry.cpp
  HashIndex.cpp
  HashIndexWriter.cpp
  IndexConfig.cpp
  IndexFilter.cpp
  IndexKeyEncoder.cpp
  KeyChunkDecoder.cpp
  KeyEncoding.cpp
  SortedIndex.cpp
  SortedIndexWriter.cpp
  IndexLookup.cpp
  IndexWriter.cpp
  SortOrder.cpp
  SortedIndexWriter.h
  SortedIndexConfig.h
  SortedIndex.h
  SortOrder.h
  KeyEncoding.h
  KeyChunkDecoder.h
  IndexWriter.h
  IndexTypes.h
  IndexSerialization.h
  IndexLookup.h
  IndexKeyEncoder.h
  IndexFilter.h
  IndexFactoryRegistry.h
  IndexConstants.h
  IndexConfig.h
  HashIndexWriter.h
  HashIndexUtils.h
  HashIndexConfig.h
  HashIndex.h
  DenseIndexRegistry.h
  DenseIndexFactory.h
  ClusterIndexWriter.h
  ClusterIndexFactory.h
  ClusterIndexConfig.h
  ClusterIndex.h
  ChunkStatsGroup.h
  ChunkStats.h
  BloomFilter.h
)

target_link_libraries(
  nimble_index
  nimble_common
  nimble_compression
  nimble_encodings
  nimble_tablet_common
  nimble_cluster_index_fb
  nimble_chunk_stats_fb
  nimble_hash_index_fb
  nimble_index_fb
  nimble_bloom_filter_fb
  nimble_sorted_index_fb
  velox_core
  velox_type
  velox_vector
  nimble_velox_common
  velox_key_encoder
  fmt::fmt
  velox_dwio_common
  velox_memory
  Folly::folly
)

# Moved from velox/index/CMakeLists.txt when the nimble/velox/ folder was
# flattened.
add_library(nimble_index_projector NimbleIndexProjector.cpp NimbleIndexProjector.h)
target_link_libraries(
  nimble_index_projector
  nimble_data_input
  nimble_index
  nimble_stream_data_writer
  nimble_stream_slicer
  nimble_tablet_reader
  nimble_tablet_reader_cache
  nimble_velox_common
  velox_caching
  velox_time
  velox_key_encoder
  velox_type
  Folly::folly
)

if(NIMBLE_BUILD_TESTING)
  add_subdirectory(tests)
endif()
