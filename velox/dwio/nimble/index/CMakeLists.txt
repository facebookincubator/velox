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
  HierarchicalKeyReader.cpp
  BlockedBloomFilter.cpp
  BloomFilter.cpp
  ChunkStats.cpp
  ChunkStatsGroup.cpp
  ClusterIndex.cpp
  ClusterIndexBase.cpp
  ClusterIndexFactory.cpp
  ClusterIndexWriter.cpp
  ClusterIndexWriterBase.cpp
  DenseIndexFactory.cpp
  DenseIndexRegistry.cpp
  HashIndex.cpp
  HashIndexWriter.cpp
  HierarchicalClusterIndex.cpp
  HierarchicalClusterIndexWriter.cpp
  IndexConfig.cpp
  IndexFilter.cpp
  IndexKeyEncoder.cpp
  KeyChunkBuilder.cpp
  KeyChunkDecoder.cpp
  KeyReader.cpp
  KeyEncoding.cpp
  SortedIndex.cpp
  SortedIndexWriter.cpp
  IndexLookup.cpp
  IndexWriter.cpp
  SortOrder.cpp
  VectorIndexWriter.h
  VectorIndexConfig.h
  SortedIndexWriter.h
  HierarchicalClusterIndex.h
  HierarchicalClusterIndexWriter.h
  HierarchicalKeyFormat.h
  HierarchicalKeyReader.h
  SortedIndexConfig.h
  SortedIndex.h
  SortOrder.h
  KeyEncoding.h
  KeyChunkBuilder.h
  KeyChunkDecoder.h
  KeyReader.h
  KeyCursor.h
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
  ClusterIndexWriterBase.h
  ClusterIndexFactory.h
  ClusterIndexConfig.h
  ClusterIndex.h
  ClusterIndexBase.h
  ChunkStatsGroup.h
  ChunkStats.h
  BlockedBloomFilter.h
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

add_library(
  nimble_vector_index
  VectorIndex.cpp
  VectorIndexWriter.cpp
  VectorIndexUtility.cpp
  VectorIndex.h
  VectorIndexUtility.h
)
target_link_libraries(
  nimble_vector_index
  nimble_index
  nimble_common
  nimble_tablet_common
  nimble_vector_index_fb
  velox_buffer
  velox_memory
  velox_type
  velox_vector
  FAISS::faiss
  Folly::folly
  OpenMP::OpenMP_CXX
)

if(NIMBLE_BUILD_TESTING)
  add_subdirectory(tests)
endif()
