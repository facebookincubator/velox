/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <folly/Function.h>

#include "velox/common/vectorindex/Embeddings.h"
#include "velox/common/vectorindex/LocalVectorIndexOptions.h"
#include "velox/common/vectorindex/VectorIndexKind.h"
#include "velox/common/vectorindex/VectorIndexMetric.h"

namespace faiss {
struct Index;
} // namespace faiss

namespace facebook::velox::common::vectorindex {

/// A local index: a FAISS index over a group of embeddings, with each
/// embedding's id as its FAISS id. A vector index keeps one local index per
/// group of nearby embeddings; a search looks only in the groups nearest to
/// the query. VectorIndexKind names the kind of FAISS
/// index. The local index is the plain FAISS serialization, as
/// faiss::write_index writes it, so faiss::read_index rebuilds it with no
/// container around it.
///
/// Ids are not required to be distinct. An id may not be kPaddingId: building
/// or reading a local index with that id throws a user error.
class LocalVectorIndex {
 public:
  /// The id search() writes in the entries it has no result for, as FAISS
  /// does. No embedding may have it.
  static constexpr int64_t kPaddingId{-1};

  /// Builds a local index of 'kind' over 'embeddings', with 'ids' giving each
  /// embedding's id, on the calling thread. Throws a user error for an id of
  /// kPaddingId. The same input gives the same bytes. The embeddings must be
  /// normalized for 'metric' (see
  /// EmbeddingNormalizer). Passes the index's bytes to 'write' in order, a
  /// chunk at a time, so the caller can place them without an intermediate
  /// copy.
  ///
  /// 'ivf.nlist' defaults to 4 * sqrt(number of embeddings), between 8 and
  /// 16384. The build reduces it, even when set, to give each list at least 39
  /// embeddings, the fewest FAISS recommends for training a list; a group of
  /// fewer than 39 embeddings gets one list. 'ivf.nprobe' defaults to a tenth
  /// of the lists, rounded up, and is capped at the number of lists. 'hnsw.M',
  /// 'hnsw.efConstruction' and 'hnsw.efSearch' default to 32, 200 and 64.
  static void build(
      VectorIndexKind kind,
      VectorIndexMetric metric,
      const LocalVectorIndexOptions& options,
      const Embeddings& embeddings,
      std::span<const int64_t> ids,
      folly::FunctionRef<void(std::string_view)> write);

  /// Reads a local index that build() wrote for 'metric'. The index refers to
  /// its embeddings, codes and graph in 'bytes' rather than copying them, so
  /// 'bytes' must outlive it. 'searchOptions', search settings only (see
  /// LocalVectorIndexOptions::parseSearch), override the ones recorded in the
  /// index: 'ivf.nprobe' for IVF types, capped at the number of lists, and
  /// 'hnsw.efSearch' for kHnswSq. Throws a user error if 'bytes' is not such
  /// a local index, if an embedding has kPaddingId, or for a setting that does
  /// not apply to its kind. A local index has at least one embedding.
  LocalVectorIndex(
      std::string_view bytes,
      VectorIndexMetric metric,
      const LocalVectorIndexOptions& searchOptions);

  ~LocalVectorIndex();

  VectorIndexKind kind() const;

  int32_t dimension() const;

  /// The number of embeddings.
  int64_t size() const;

  /// Finds the 'k' embeddings nearest to each of 'queries', which must be
  /// normalized for the metric (see EmbeddingNormalizer). Writes each query's
  /// embeddings nearest first: their ids to 'ids' and their scores to
  /// 'scores', 'k' entries per query, in query order. For kCosine, the score is
  /// the cosine similarity, where larger is nearer. A query with fewer than
  /// 'k' results, from a small index or an approximate search, gets
  /// kPaddingId in the remaining entries. 'k' must be at least 1.
  void search(
      const Embeddings& queries,
      int64_t k,
      std::span<int64_t> ids,
      std::span<float> scores) const;

 private:
  std::unique_ptr<faiss::Index> index_;
  VectorIndexKind kind_;
};

} // namespace facebook::velox::common::vectorindex
