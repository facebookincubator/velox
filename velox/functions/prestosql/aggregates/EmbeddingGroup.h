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

#include <span>
#include <vector>

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/common/vectorindex/Embeddings.h"
#include "velox/common/vectorindex/VectorIndexMetric.h"
#include "velox/exec/SimpleAggregateAdapter.h"

namespace facebook::velox::aggregate::prestosql {

/// The embeddings of an aggregation group and their ids, as the vector index
/// aggregates accumulate them: checked, normalized for the group's metric and
/// stored in chunks of about 64KB. A growing group never moves its embeddings,
/// and over-allocates by at most one chunk.
///
///   EmbeddingGroup group(allocator);
///   group.add(embedding, id, VectorIndexMetric::kCosine);
///   const auto contents = group.gather();
///   FaissLocalVectorIndex::build(
///       ..., contents.embeddings(), contents.ids, ...);
///
/// Invariants:
/// - Every embedding has dimension() values, and dimension() is positive once
///   there is one.
/// - No value is NaN or infinite.
/// - There is an id per embedding, in the same order.
class EmbeddingGroup {
 public:
  explicit EmbeddingGroup(HashStringAllocator* allocator);

  /// Adds 'embedding' with 'id', normalized for 'metric'. Throws a user error
  /// for an empty embedding, one whose length differs from the earlier ones,
  /// or one with a null, NaN or infinite element.
  void add(
      const exec::arg_type<Array<float>>& embedding,
      int64_t id,
      common::vectorindex::VectorIndexMetric metric);

  /// Adds the ids and embeddings of an intermediate result, which
  /// writeIntermediate() wrote, as they are. They may come from a merge
  /// companion function, so they are checked as add() checks them, and for an
  /// id per embedding.
  void addIntermediate(
      const exec::arg_type<Array<int64_t>>& ids,
      const exec::arg_type<Array<float>>& embeddings);

  /// Writes the ids and the embeddings, one after another.
  void writeIntermediate(
      exec::out_type<Array<int64_t>>& ids,
      exec::out_type<Array<float>>& embeddings) const;

  /// The ids and embeddings of a group, each in contiguous memory.
  struct Contents {
    std::vector<int64_t, StlAllocator<int64_t>> ids;
    std::vector<float, StlAllocator<float>> values;
    int32_t dimension;

    common::vectorindex::Embeddings embeddings() const {
      return common::vectorindex::Embeddings(values, dimension);
    }
  };

  /// Returns a copy of the ids and embeddings, in the order they were added.
  /// Requires at least one embedding.
  Contents gather() const;

  /// Like gather(), but in the order of the ids. Embeddings with equal ids keep
  /// their order.
  Contents gatherSortedById() const;

 private:
  void setDimension(int64_t size);

  // Returns room for one more embedding at the end of the last chunk, adding a
  // chunk when it is full.
  std::span<float> appendEmbedding();

  // Returns the embedding at 'index', in the order they were added.
  std::span<const float> embeddingAt(int64_t index) const;

  HashStringAllocator* const allocator_;
  // The length of every embedding; 0 until the first one arrives.
  int32_t dimension_{0};
  // The embeddings a chunk holds; set with the dimension.
  int32_t embeddingsPerChunk_{0};
  // An id per embedding, in the order they were added.
  std::vector<int64_t, StlAllocator<int64_t>> ids_;
  // The embeddings, normalized for the group's metric, in the order they were
  // added. Each chunk has capacity for embeddingsPerChunk_ embeddings, and
  // all but the last are full.
  std::vector<
      std::vector<float, StlAllocator<float>>,
      StlAllocator<std::vector<float, StlAllocator<float>>>>
      chunks_;
};

} // namespace facebook::velox::aggregate::prestosql
