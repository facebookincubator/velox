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
/// stored one after another.
///
///   EmbeddingGroup group(allocator);
///   group.add(embedding, id, VectorIndexMetric::kCosine);
///   FaissLocalVectorIndex::build(..., group.embeddings(), group.ids(), ...);
///
/// Invariants:
/// - Every embedding has dimension() values, and dimension() is positive once
///   there is one.
/// - No value is NaN or infinite.
/// - ids() has an id per embedding, in the same order.
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

  /// Puts the embeddings in the order of their ids, in place. Embeddings with
  /// equal ids keep their order.
  void sortById();

  std::span<const int64_t> ids() const {
    return ids_;
  }

  /// The embeddings. Requires at least one.
  common::vectorindex::Embeddings embeddings() const {
    return common::vectorindex::Embeddings(values_, dimension_);
  }

 private:
  void setDimension(int64_t size);

  // The length of every embedding; 0 until the first one arrives.
  int32_t dimension_{0};
  // An id per embedding, in the order of 'values_'.
  std::vector<int64_t, StlAllocator<int64_t>> ids_;
  // The embeddings, normalized for the group's metric, one after another.
  std::vector<float, StlAllocator<float>> values_;
};

} // namespace facebook::velox::aggregate::prestosql
