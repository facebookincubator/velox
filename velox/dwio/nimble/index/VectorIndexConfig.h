/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstdint>
#include <string>

namespace facebook::nimble {

/// Caps index sections at 1 GiB, within MetadataSection's uint32_t size limit.
inline constexpr uint64_t kMaxVectorIndexSizeBytes{1ULL << 30};

/// Distance metric for vector similarity search.
enum class VectorDistanceMetric : uint8_t {
  /// Squared Euclidean distance.
  kL2,
  /// Cosine similarity over normalized vectors.
  kCosine,
  /// Inner-product similarity over unnormalized vectors.
  kDotProduct,
};

/// Vector index types.
/// IVF partitioning with configurable quantization or sub-index acceleration.
enum class VectorIndexType : uint8_t {
  /// IVF + brute-force scan (no compression).
  kIvfFlat,
  /// IVF + 8-bit scalar quantization.
  kIvfSq8,
  /// IVF + product quantization.
  kIvfPq,
  /// IVF + RaBitQ binary quantization.
  kIvfRaBitQ,
  /// HNSW graph with SQ8 quantization.
  kHnswSq8,
};

/// Configuration for building a vector search index during file writes.
struct VectorIndexConfig {
  /// Selects a direct child of the top-level row with type ARRAY<REAL>.
  /// Nested columns are not supported.
  std::string columnName;

  /// Sets the number of elements in each vector.
  uint32_t dimensions{0};

  /// Selects the distance metric used for training and search.
  VectorDistanceMetric metric{VectorDistanceMetric::kL2};

  /// Selects the FAISS index implementation.
  VectorIndexType indexType{VectorIndexType::kIvfSq8};

  /// Sets the IVF partition count. Zero derives it from the number of vectors.
  uint32_t numPartitions{0};

  /// Sets the number of PQ sub-quantizers.
  uint32_t pqSubQuantizers{8};

  /// Sets the number of bits per PQ code.
  uint8_t pqBits{8};

  /// Sets the number of bidirectional HNSW links per node.
  uint32_t hnswNumConnections{32};

  /// Rejects input when buffered vectors exceed this many bytes.
  uint64_t maxBufferedVectorSizeBytes{kMaxVectorIndexSizeBytes};

  /// Rejects close() when the serialized index exceeds this many bytes.
  uint64_t maxIndexSizeBytes{kMaxVectorIndexSizeBytes};
};

} // namespace facebook::nimble
