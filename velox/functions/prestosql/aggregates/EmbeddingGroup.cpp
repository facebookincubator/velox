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
#include "velox/functions/prestosql/aggregates/EmbeddingGroup.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::aggregate::prestosql {

using common::vectorindex::EmbeddingNormalizer;
using common::vectorindex::VectorIndexMetric;

namespace {

void checkFinite(float value) {
  VELOX_USER_CHECK(
      std::isfinite(value), "Embedding must not contain NaN or infinity");
}

} // namespace

EmbeddingGroup::EmbeddingGroup(HashStringAllocator* allocator)
    : ids_{StlAllocator<int64_t>(allocator)},
      values_{StlAllocator<float>(allocator)} {}

void EmbeddingGroup::add(
    const exec::arg_type<Array<float>>& embedding,
    int64_t id,
    VectorIndexMetric metric) {
  setDimension(embedding.size());
  for (auto value : embedding) {
    VELOX_USER_CHECK(value.has_value(), "Embedding must not contain nulls");
    checkFinite(value.value());
    values_.push_back(value.value());
  }
  EmbeddingNormalizer::normalize(metric, std::span(values_).last(dimension_));
  ids_.push_back(id);
}

void EmbeddingGroup::addIntermediate(
    const exec::arg_type<Array<int64_t>>& ids,
    const exec::arg_type<Array<float>>& embeddings) {
  VELOX_USER_CHECK_GT(
      ids.size(),
      0,
      "Intermediate vector index result must include an embedding");
  VELOX_USER_CHECK_EQ(
      embeddings.size() % ids.size(),
      0,
      "Intermediate vector index result must have the same number of values per embedding");
  setDimension(embeddings.size() / ids.size());
  for (auto id : ids) {
    VELOX_USER_CHECK(
        id.has_value(), "Intermediate vector index ids must not be null");
    ids_.push_back(id.value());
  }
  for (auto value : embeddings) {
    VELOX_USER_CHECK(
        value.has_value(),
        "Intermediate vector index embeddings must not contain nulls");
    checkFinite(value.value());
    values_.push_back(value.value());
  }
}

void EmbeddingGroup::writeIntermediate(
    exec::out_type<Array<int64_t>>& ids,
    exec::out_type<Array<float>>& embeddings) const {
  VELOX_USER_CHECK_LE(
      values_.size(),
      std::numeric_limits<vector_size_t>::max(),
      "Too many embedding values in a vector index group");
  for (auto id : ids_) {
    ids.add_item() = id;
  }
  for (auto value : values_) {
    embeddings.add_item() = value;
  }
}

void EmbeddingGroup::setDimension(int64_t size) {
  VELOX_USER_CHECK_GT(size, 0, "Embedding must not be empty");
  if (dimension_ == 0) {
    dimension_ = size;
    return;
  }
  VELOX_USER_CHECK_EQ(
      size, dimension_, "All embeddings must have the same length");
}

void EmbeddingGroup::sortById() {
  // order[i] is the position of the embedding that belongs at position i.
  std::vector<int64_t, StlAllocator<int64_t>> order(
      ids_.size(), ids_.get_allocator());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](auto lhs, auto rhs) {
    return ids_[lhs] < ids_[rhs];
  });
  // Moves the embeddings along each cycle of the permutation, setting
  // order[i] to i once position i holds its embedding.
  std::vector<float> embedding(dimension_);
  const auto valuesAt = [&](int64_t position) {
    return values_.begin() + position * dimension_;
  };
  for (int64_t start = 0; start < order.size(); ++start) {
    if (order[start] == start) {
      continue;
    }
    const auto id = ids_[start];
    std::copy_n(valuesAt(start), dimension_, embedding.begin());
    auto position = start;
    while (order[position] != start) {
      const auto source = order[position];
      ids_[position] = ids_[source];
      std::copy_n(valuesAt(source), dimension_, valuesAt(position));
      order[position] = position;
      position = source;
    }
    ids_[position] = id;
    std::copy_n(embedding.begin(), dimension_, valuesAt(position));
    order[position] = position;
  }
}

} // namespace facebook::velox::aggregate::prestosql
