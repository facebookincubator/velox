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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <folly/container/F14Map.h>

#include "velox/dwio/nimble/index/IndexLookup.h"
#include "velox/dwio/nimble/index/VectorIndexConfig.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

namespace faiss {
struct Index;
} // namespace faiss

namespace facebook::nimble {
class MetadataInput;
} // namespace facebook::nimble

namespace facebook::nimble::index {

/// Reads a FAISS-based vector similarity search index from a Nimble file.
///
/// The FAISS index is immutable after construction. Concurrent searches use
/// request-local parameters and synchronize FAISS global statistics when
/// necessary.
class VectorIndex {
 public:
  /// Configures one nearest-neighbor query.
  struct SearchConfig {
    /// Supplies a query with the same dimensionality as the indexed vectors.
    std::vector<float> queryVector;

    /// Sets the maximum number of nearest neighbors to return.
    uint32_t numNeighbors{10};

    /// Sets the number of IVF partitions to probe. IVF first assigns the query
    /// to its nearest partitions, then searches vectors only within them.
    /// Higher values generally improve recall at the cost of more work.
    uint32_t numProbes{32};

    /// Sets the HNSW candidate-list size used while traversing the graph.
    /// Higher values generally improve recall at the cost of more work.
    uint32_t hnswSearchDepth{32};
  };

  /// Identifies one nearest-neighbor match and its metric-specific score.
  struct SearchResult {
    /// Identifies the zero-based row in the file.
    int64_t rowId{0};

    /// Contains squared Euclidean distance for L2, where smaller is better.
    /// Contains similarity for cosine and dot product, where larger is better.
    float score{0};
  };

  /// Describes the logical vector index stored in a Nimble file.
  struct Metadata {
    /// Identifies the indexed top-level column.
    std::string columnName;

    /// Defines the required width of indexed and query vectors.
    uint32_t dimensions{0};

    /// Determines whether search scores are distances or similarities.
    VectorDistanceMetric metric{VectorDistanceMetric::kL2};

    /// Selects the concrete FAISS implementation to validate and query.
    VectorIndexType indexType{VectorIndexType::kIvfSq8};

    /// Bounds the valid row labels returned by FAISS.
    uint64_t numVectors{0};
  };

  /// Deserializes and validates one FAISS index.
  static std::shared_ptr<const VectorIndex> create(
      Metadata metadata,
      std::string_view serializedIndex);

  ~VectorIndex();

  /// Searches for nearest neighbors in score order. Row IDs are not
  /// numerically ordered.
  std::vector<SearchResult> search(const SearchConfig& config) const;

  /// Returns the column name this index was built on.
  const std::string& columnName() const;

  /// Returns the vector dimensionality.
  uint32_t dimensions() const;

  /// Returns the distance metric.
  VectorDistanceMetric metric() const;

  /// Returns the index type.
  VectorIndexType indexType() const;

  /// Returns the number of indexed vectors.
  uint64_t numVectors() const;

 private:
  // Constructs and validates one eagerly deserialized FAISS index.
  VectorIndex(Metadata metadata, std::string_view serializedIndex);

  // Identifies the indexed top-level column.
  const std::string columnName_;

  // Defines the required width of search queries.
  const uint32_t dimensions_;

  // Determines how FAISS scores candidate vectors.
  const VectorDistanceMetric metric_;

  // Identifies the serialized FAISS index implementation.
  const VectorIndexType indexType_;

  // Bounds valid row IDs returned by FAISS.
  const uint64_t numVectors_;

  // Remains immutable so concurrent searches only read shared state.
  const std::unique_ptr<faiss::Index> faissIndex_;
};

/// Describes the vector indexes in a Nimble file without loading their data.
///
/// Call load() to materialize only the index needed for a query.
class VectorIndexDirectory {
 public:
  /// Parses a vector-index directory and captures options used to create its
  /// index data input on the first load.
  static VectorIndexDirectory create(
      Section directorySection,
      const IndexLookup::Options& options);

  /// Returns the number of indexes described by the directory.
  size_t numIndexes() const;

  /// Returns whether the directory describes an index for the column.
  bool contains(std::string_view columnName) const;

  /// Loads an immutable index that callers may retain and reuse.
  std::shared_ptr<const VectorIndex> load(std::string_view columnName) const;

 private:
  struct InputState;

  struct Entry {
    // Defines the index and validates its serialized FAISS representation.
    VectorIndex::Metadata metadata;

    // Locates the serialized FAISS representation in the Nimble file.
    MetadataSection indexSection;
  };

  VectorIndexDirectory(
      folly::F14FastMap<std::string, Entry> entries,
      std::shared_ptr<InputState> inputState);

  // Keys descriptors by top-level column name for direct lookup.
  folly::F14FastMap<std::string, Entry> entries_;

  // Lazily creates and owns the index-specific metadata input.
  std::shared_ptr<InputState> inputState_;
};

} // namespace facebook::nimble::index
