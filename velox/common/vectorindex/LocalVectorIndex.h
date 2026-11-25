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
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "velox/common/ScopedRegistry.h"
#include "velox/common/vectorindex/Embeddings.h"
#include "velox/common/vectorindex/VectorIndexMetric.h"

namespace facebook::velox::common::vectorindex {

/// A local index: an index over a group of embeddings, each with an id, that
/// finds the embeddings nearest to a query. A vector index keeps one local
/// index per group of nearby embeddings; a search looks only in the groups
/// nearest to the query.
///
/// A format decides how a local index is encoded (see LocalVectorIndexFormat):
///
///   auto reader = format->makeReader(VectorIndexMetric::kCosine, {});
///   auto index = reader->open(bytes);
///   index->search(queries, k, ids, scores);
class LocalVectorIndex {
 public:
  /// The id search() writes in the entries it has no result for. No embedding
  /// may have it.
  static constexpr int64_t kPaddingId{-1};

  /// Search settings, as key-value pairs, that override the ones recorded in a
  /// local index. Their keys and values are the format's.
  using SearchOptions = std::vector<std::pair<std::string, std::string>>;

  virtual ~LocalVectorIndex() = default;

  /// The length of the embeddings.
  virtual int32_t dimension() const = 0;

  /// The number of embeddings. At least 1.
  virtual int64_t size() const = 0;

  /// Finds the 'k' embeddings nearest to each of 'queries', which must be
  /// normalized for the metric (see EmbeddingNormalizer). Writes each query's
  /// embeddings nearest first: their ids to 'ids' and their scores to
  /// 'scores', 'k' entries per query, in query order. For kCosine, the score is
  /// the cosine similarity, where larger is nearer. A query with fewer than
  /// 'k' results, from a small index or an approximate search, gets
  /// kPaddingId in the remaining entries. Ids are not required to be
  /// distinct. 'k' must be at least 1.
  virtual void search(
      const Embeddings& queries,
      int64_t k,
      std::span<int64_t> ids,
      std::span<float> scores) const = 0;
};

/// Reads local indexes of one format, built for one metric, with one set of
/// search settings.
class LocalVectorIndexReader {
 public:
  virtual ~LocalVectorIndexReader() = default;

  /// Reads 'bytes', a local index. The caller keeps 'bytes' alive while the
  /// index lives, so the index may refer to them rather than copy them. Throws
  /// a user error if 'bytes' are not a local index of the reader's format and
  /// metric.
  virtual std::unique_ptr<LocalVectorIndex> open(
      std::string_view bytes) const = 0;
};

/// A format of local indexes: how their bytes are read. An application
/// registers the formats a query may read in a LocalVectorIndexFormats
/// registry, set on the query's QueryCtx under kRegistryKey; a vector search
/// join reads its indexes with the format it names.
class LocalVectorIndexFormat {
 public:
  /// The key of the query's LocalVectorIndexFormats registry on its QueryCtx.
  static constexpr std::string_view kRegistryKey{"localVectorIndexFormats"};

  virtual ~LocalVectorIndexFormat() = default;

  /// Returns a reader of local indexes built for 'metric' that searches them
  /// with 'searchOptions'. Throws a user error for a search setting this
  /// format does not support.
  virtual std::unique_ptr<LocalVectorIndexReader> makeReader(
      VectorIndexMetric metric,
      const LocalVectorIndex::SearchOptions& searchOptions) const = 0;
};

/// The local vector index formats of a query, by name.
class LocalVectorIndexFormats
    : public ScopedRegistry<std::string, LocalVectorIndexFormat> {
 public:
  using ScopedRegistry::ScopedRegistry;

  /// Returns a reader of local indexes in the format named 'format'. Throws a
  /// user error if no such format is registered. See
  /// LocalVectorIndexFormat::makeReader for 'metric' and 'searchOptions'.
  std::unique_ptr<LocalVectorIndexReader> makeReader(
      const std::string& format,
      VectorIndexMetric metric,
      const LocalVectorIndex::SearchOptions& searchOptions) const;
};

} // namespace facebook::velox::common::vectorindex
