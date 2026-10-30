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

#include "velox/common/vectorindex/Embeddings.h"

#include <faiss/utils/distances.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::vectorindex {

Embeddings::Embeddings(std::span<const float> values, int32_t dimension)
    : values_{values}, dimension_{dimension} {
  VELOX_CHECK_GT(dimension_, 0);
  VELOX_CHECK(!values_.empty());
  VELOX_CHECK_EQ(values_.size() % dimension_, 0);
}

void EmbeddingNormalizer::normalize(
    VectorIndexMetric metric,
    std::span<float> embedding) {
  switch (metric) {
    case VectorIndexMetric::kCosine:
      faiss::fvec_renorm_L2(embedding.size(), 1, embedding.data());
      return;
  }
  VELOX_UNREACHABLE();
}

} // namespace facebook::velox::common::vectorindex
