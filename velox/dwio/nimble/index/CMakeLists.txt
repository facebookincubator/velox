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

add_subdirectory(tests)
