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

#include <folly/Function.h>

#include "velox/common/vectorindex/Embeddings.h"
#include "velox/common/vectorindex/LocalVectorIndex.h"
#include "velox/common/vectorindex/LocalVectorIndexAccelerator.h"
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
  /// kPaddingId. The same input gives the same bytes, unless an accelerator
  /// (see LocalVectorIndexAccelerator) builds the index. The embeddings must be
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
  /// not apply to its kind. A local index has at least one embedding. With an
  /// accelerator set, the index also gets a device copy for search() to use.
  FaissLocalVectorIndex(
      std::string_view bytes,
      VectorIndexMetric metric,
      const LocalVectorIndexOptions& searchOptions);

  /// Reads a local index as above, giving it a device copy from 'accelerator'
  /// rather than the one setLocalVectorIndexAccelerator() set. A null
  /// 'accelerator' keeps the index on the CPU.
  FaissLocalVectorIndex(
      std::string_view bytes,
      VectorIndexMetric metric,
      const LocalVectorIndexOptions& searchOptions,
      const std::shared_ptr<const LocalVectorIndexAccelerator>& accelerator);

  ~FaissLocalVectorIndex() override;

  /// Returns the number of embeddings in the local index 'bytes', which
  /// build() wrote, from its header, without reading the rest. Throws a user
  /// error if 'bytes' does not start as a local index does.
  static int64_t size(std::string_view bytes);

  VectorIndexKind kind() const;

  int32_t dimension() const override;

  int64_t size() const override;

  /// True if search() runs on an accelerator's device copy.
  bool accelerated() const {
    return accelerated_ != nullptr;
  }

  /// The device copy, or null if the index is searched on the CPU.
  const LocalVectorIndexAccelerator::Index* deviceIndex() const {
    return accelerated_.get();
  }

  void search(
      const Embeddings& queries,
      int64_t k,
      std::span<int64_t> ids,
      std::span<float> scores) const override;

 private:
  std::unique_ptr<faiss::Index> index_;
  VectorIndexKind kind_;
  // Refers to 'index_', so it is declared after it to be destroyed first.
  std::unique_ptr<LocalVectorIndexAccelerator::Index> accelerated_;
};

} // namespace facebook::velox::common::vectorindex
