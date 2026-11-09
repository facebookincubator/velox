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
    : allocator_{allocator},
      ids_{StlAllocator<int64_t>(allocator)},
      chunks_{
          StlAllocator<std::vector<float, StlAllocator<float>>>(allocator)} {}

void EmbeddingGroup::add(
    const exec::arg_type<Array<float>>& embedding,
    int64_t id,
    VectorIndexMetric metric) {
  setDimension(embedding.size());
  const auto slot = appendEmbedding();
  size_t i = 0;
  for (auto value : embedding) {
    VELOX_USER_CHECK(value.has_value(), "Embedding must not contain nulls");
    checkFinite(value.value());
    slot[i++] = value.value();
  }
  EmbeddingNormalizer::normalize(metric, slot);
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
  std::span<float> slot;
  size_t i = 0;
  for (auto value : embeddings) {
    VELOX_USER_CHECK(
        value.has_value(),
        "Intermediate vector index embeddings must not contain nulls");
    checkFinite(value.value());
    if (i == slot.size()) {
      slot = appendEmbedding();
      i = 0;
    }
    slot[i++] = value.value();
  }
}

void EmbeddingGroup::writeIntermediate(
    exec::out_type<Array<int64_t>>& ids,
    exec::out_type<Array<float>>& embeddings) const {
  VELOX_USER_CHECK_LE(
      ids_.size() * dimension_,
      std::numeric_limits<vector_size_t>::max(),
      "Too many embedding values in a vector index group");
  for (auto id : ids_) {
    ids.add_item() = id;
  }
  for (const auto& chunk : chunks_) {
    for (auto value : chunk) {
      embeddings.add_item() = value;
    }
  }
}

namespace {

// Returns empty contents with room for 'numEmbeddings' embeddings.
EmbeddingGroup::Contents makeContents(
    HashStringAllocator* allocator,
    int32_t dimension,
    size_t numEmbeddings) {
  VELOX_CHECK_GT(numEmbeddings, 0);
  EmbeddingGroup::Contents contents{
      .ids = std::vector<int64_t, StlAllocator<int64_t>>(
          StlAllocator<int64_t>(allocator)),
      .values = std::vector<float, StlAllocator<float>>(
          StlAllocator<float>(allocator)),
      .dimension = dimension,
  };
  contents.ids.reserve(numEmbeddings);
  contents.values.reserve(numEmbeddings * dimension);
  return contents;
}

} // namespace

EmbeddingGroup::Contents EmbeddingGroup::gather() const {
  auto contents = makeContents(allocator_, dimension_, ids_.size());
  contents.ids.assign(ids_.begin(), ids_.end());
  for (const auto& chunk : chunks_) {
    contents.values.insert(contents.values.end(), chunk.begin(), chunk.end());
  }
  return contents;
}

EmbeddingGroup::Contents EmbeddingGroup::gatherSortedById() const {
  auto contents = makeContents(allocator_, dimension_, ids_.size());
  // order[i] is the position, in the order of adding, of the i-th embedding.
  std::vector<int64_t, StlAllocator<int64_t>> order(
      ids_.size(), StlAllocator<int64_t>(allocator_));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](auto lhs, auto rhs) {
    return ids_[lhs] < ids_[rhs];
  });
  for (auto position : order) {
    contents.ids.push_back(ids_[position]);
    const auto embedding = embeddingAt(position);
    contents.values.insert(
        contents.values.end(), embedding.begin(), embedding.end());
  }
  return contents;
}

void EmbeddingGroup::setDimension(int64_t size) {
  VELOX_USER_CHECK_GT(size, 0, "Embedding must not be empty");
  if (dimension_ == 0) {
    constexpr int64_t kChunkBytes{64 << 10};
    dimension_ = size;
    embeddingsPerChunk_ = std::max<int64_t>(
        1, kChunkBytes / (static_cast<int64_t>(sizeof(float)) * size));
    return;
  }
  VELOX_USER_CHECK_EQ(
      size, dimension_, "All embeddings must have the same length");
}

std::span<float> EmbeddingGroup::appendEmbedding() {
  if (chunks_.empty() ||
      chunks_.back().size() ==
          static_cast<size_t>(embeddingsPerChunk_) * dimension_) {
    auto& chunk = chunks_.emplace_back(StlAllocator<float>(allocator_));
    chunk.reserve(static_cast<size_t>(embeddingsPerChunk_) * dimension_);
  }
  auto& chunk = chunks_.back();
  chunk.resize(chunk.size() + dimension_);
  return std::span(chunk).last(dimension_);
}

std::span<const float> EmbeddingGroup::embeddingAt(int64_t index) const {
  const auto& chunk = chunks_[index / embeddingsPerChunk_];
  return std::span(chunk).subspan(
      (index % embeddingsPerChunk_) * dimension_, dimension_);
}

} // namespace facebook::velox::aggregate::prestosql
