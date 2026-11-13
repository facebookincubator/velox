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

#include "velox/exec/VectorSearchJoinEmbeddings.h"

#include <cmath>

#include "velox/common/vectorindex/Embeddings.h"

namespace facebook::velox::exec {

VectorSearchJoinEmbeddings::VectorSearchJoinEmbeddings(
    const BaseVector& column,
    common::vectorindex::VectorIndexMetric metric)
    : metric_{metric}, embeddings_{column} {
  nulls_ = embeddings_.nulls();
  // A null constant embedding has no array vector under it.
  if (embeddings_.isConstantMapping() && embeddings_.isNullAt(0)) {
    return;
  }
  arrays_ = embeddings_.base()->asChecked<ArrayVector>();
  elements_.decode(*arrays_->elements());
  // A nulls buffer often has no nulls set. One pass over it saves checking
  // each element for null.
  bool mayHaveNulls = elements_.mayHaveNulls();
  if (mayHaveNulls && elements_.isIdentityMapping()) {
    mayHaveNulls = !bits::isAllSet(elements_.nulls(), 0, elements_.size());
  }
  // A constant, or a run of rows of a dictionary, refers to one array, checked
  // once.
  vector_size_t lastIndex{-1};
  for (vector_size_t row = 0; row < embeddings_.size(); ++row) {
    if (embeddings_.isNullAt(row)) {
      continue;
    }
    const auto index = embeddings_.index(row);
    if (index == lastIndex) {
      continue;
    }
    lastIndex = index;
    const auto size = arrays_->sizeAt(index);
    if (dimension_ == 0) {
      VELOX_USER_CHECK_GT(size, 0, "Query embedding must not be empty");
      dimension_ = size;
    }
    VELOX_USER_CHECK_EQ(
        size, dimension_, "Query embeddings must have the same length");
    const auto offset = arrays_->offsetAt(index);
    for (auto i = offset; i < offset + size; ++i) {
      VELOX_USER_CHECK(
          !mayHaveNulls || !elements_.isNullAt(i),
          "Query embedding must not contain nulls");
      VELOX_USER_CHECK(
          std::isfinite(elements_.valueAt<float>(i)),
          "Query embedding must not contain NaN or infinity");
    }
  }
}

void VectorSearchJoinEmbeddings::copyNormalized(
    vector_size_t row,
    std::span<float> query) const {
  const auto offset = arrays_->offsetAt(embeddings_.index(row));
  if (elements_.isIdentityMapping()) {
    std::copy_n(elements_.data<float>() + offset, query.size(), query.data());
  } else {
    for (size_t i = 0; i < query.size(); ++i) {
      query[i] = elements_.valueAt<float>(offset + i);
    }
  }
  common::vectorindex::EmbeddingNormalizer::normalize(metric_, query);
}

} // namespace facebook::velox::exec
