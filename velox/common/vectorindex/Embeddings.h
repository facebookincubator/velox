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
#include <span>

#include "velox/common/vectorindex/VectorIndexMetric.h"

namespace facebook::velox::common::vectorindex {

/// Embeddings of the same dimension, stored one after another. Does not own
/// the values.
class Embeddings {
 public:
  /// 'values' holds at least one embedding of 'dimension' floats.
  Embeddings(std::span<const float> values, int32_t dimension);

  /// 'size' embeddings, at least one, of 'dimension' floats each, starting at
  /// 'values'.
  Embeddings(const float* values, int64_t size, int32_t dimension)
      : Embeddings(
            std::span<const float>(values, size * dimension),
            dimension) {}

  const float* data() const {
    return values_.data();
  }

  int32_t dimension() const {
    return dimension_;
  }

  /// The number of embeddings.
  int64_t size() const {
    return values_.size() / dimension_;
  }

 private:
  const std::span<const float> values_;
  const int32_t dimension_;
};

/// Normalizes embeddings for comparison under a metric.
class EmbeddingNormalizer {
 public:
  /// Normalizes 'embedding' in place for 'metric'. kCosine scales it to length
  /// 1 and leaves a zero embedding as it is.
  ///
  /// Normalize each embedding once: normalizing it again can change the last
  /// bits of its values.
  static void normalize(VectorIndexMetric metric, std::span<float> embedding);
};

} // namespace facebook::velox::common::vectorindex
