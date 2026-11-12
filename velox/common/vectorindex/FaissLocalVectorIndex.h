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
#include <span>
#include <string_view>
#include <vector>

#include <folly/Function.h>

#include "velox/common/vectorindex/CodecOptions.h"
#include "velox/common/vectorindex/Embeddings.h"
#include "velox/common/vectorindex/LocalVectorIndex.h"
#include "velox/common/vectorindex/LocalVectorIndexOptions.h"
#include "velox/common/vectorindex/VectorIndexKind.h"
#include "velox/common/vectorindex/VectorIndexMetric.h"

namespace faiss {
struct Index;
} // namespace faiss

namespace facebook::velox::common::vectorindex {

/// The local indexes of the "faiss" format: a FAISS index over a group of
/// embeddings, with each embedding's id as its FAISS id. VectorIndexKind names
/// the kind of FAISS index. The local index is the plain FAISS serialization,
/// as faiss::write_index writes it, so faiss::read_index rebuilds it with no
/// container around it.
///
///   auto formats = std::make_shared<LocalVectorIndexFormats>();
///   formats->insert(
///       std::string(FaissLocalVectorIndex::kFormat),
///       FaissLocalVectorIndex::format());
///   queryCtx->setRegistry(LocalVectorIndexFormat::kRegistryKey, formats);
///
/// Ids are not required to be distinct. An id may not be kPaddingId: building
/// or reading a local index with that id throws a user error.
class FaissLocalVectorIndex : public LocalVectorIndex {
 public:
  /// The name an application registers the format under.
  static constexpr std::string_view kFormat{"faiss"};

  /// Returns the format. Its search settings are those of
  /// LocalVectorIndexOptions::parseSearch.
  static std::shared_ptr<LocalVectorIndexFormat> format();

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
  /// fewer than 39 embeddings gets one list. 'ivf.nprobe' defaults to all the
  /// lists and is capped at the number of lists. 'hnsw.M',
  /// 'hnsw.efConstruction' and 'hnsw.efSearch' default to 32, 200 and 64.
  static void build(
      VectorIndexKind kind,
      VectorIndexMetric metric,
      const LocalVectorIndexOptions& options,
      const Embeddings& embeddings,
      std::span<const int64_t> ids,
      folly::FunctionRef<void(std::string_view)> write);

  /// Names of the runtime stats trainCentroids() records on the operator that
  /// calls it, when it runs k-means.
  struct TrainStats {
    /// The number of embeddings it trains on.
    static constexpr std::string_view kEmbeddings{"trainEmbeddings"};

    /// The number of threads it trains on.
    static constexpr std::string_view kThreads{"trainThreads"};

    /// The number of levels it trains in: 1, or 2 for more clusters than the
    /// codec options' maxFlatClusters.
    static constexpr std::string_view kLevels{"trainLevels"};

    /// Its wall time, in nanoseconds.
    static constexpr std::string_view kWallNanos{"trainWallNanos"};
  };

  /// Clusters 'embeddings' with k-means on 'numThreads' threads and returns the
  /// centroids, one after another, normalized for 'metric'. The embeddings
  /// must be normalized for 'metric' (see EmbeddingNormalizer). Returns
  /// min('numClusters', number of embeddings) centroids; with no more
  /// embeddings than clusters, the centroids are the embeddings. The same
  /// embeddings in the same order give the same centroids on the same number
  /// of threads. For kCosine the k-means is spherical: an embedding goes to the
  /// centroid with the largest inner product, and each centroid is the
  /// normalized mean of its embeddings.
  ///
  /// More than 'options.maxFlatClusters' clusters train in two levels: about
  /// sqrt('numClusters') coarse clusters, then within each a share of
  /// 'numClusters' in proportion to its embeddings. That cuts the work by about
  /// sqrt('numClusters') / 2.
  static std::vector<float> trainCentroids(
      VectorIndexMetric metric,
      int64_t numClusters,
      const Embeddings& embeddings,
      const CodecOptions& options,
      int32_t numThreads);

  /// Reads a local index that build() wrote for 'metric'. The index refers to
  /// its embeddings, codes and graph in 'bytes' rather than copying them, so
  /// 'bytes' must outlive it. 'searchOptions', search settings only (see
  /// LocalVectorIndexOptions::parseSearch), override the ones recorded in the
  /// index: 'ivf.nprobe' for IVF types, capped at the number of lists, and
  /// 'hnsw.efSearch' for kHnswSq. Throws a user error if 'bytes' is not such
  /// a local index, if an embedding has kPaddingId, or for a setting that does
  /// not apply to its kind. A local index has at least one embedding.
  FaissLocalVectorIndex(
      std::string_view bytes,
      VectorIndexMetric metric,
      const LocalVectorIndexOptions& searchOptions);

  ~FaissLocalVectorIndex() override;

  /// Returns the number of embeddings in the local index 'bytes', which
  /// build() wrote, from its header, without reading the rest. Throws a user
  /// error if 'bytes' does not start as a local index does.
  static int64_t size(std::string_view bytes);

  VectorIndexKind kind() const;

  int32_t dimension() const override;

  int64_t size() const override;

  void search(
      const Embeddings& queries,
      int64_t k,
      std::span<int64_t> ids,
      std::span<float> scores) const override;

 private:
  std::unique_ptr<faiss::Index> index_;
  VectorIndexKind kind_;
};

} // namespace facebook::velox::common::vectorindex
