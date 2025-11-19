/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <folly/Random.h>
#include <random>
#include "velox/common/base/BitUtil.h"
#include "velox/functions/lib/aggregates/ValueList.h"
#include "velox/type/FloatingPointUtil.h"
#include "velox/type/StringView.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::aggregate::prestosql {

struct ReservoirSampleAccumulator {
  VectorPtr samples;
  VectorPtr initialSamples;
  int64_t initialSeenCount = 0;
  int64_t processedCount = 0;
  int32_t maxSampleSize = -1;

  explicit ReservoirSampleAccumulator() = default;

  bool isInitialized() const {
    return maxSampleSize >= 0;
  }

  int32_t initialSampleCount() const {
    return initialSamples ? initialSamples->size() : 0;
  }

  vector_size_t sampleCount() const {
    return samples ? samples->size() : 0;
  }

  void addValueToReservoir(
      const DecodedVector& decodedVector,
      vector_size_t row,
      const TypePtr& type,
      memory::MemoryPool* pool) {
    if (processedCount <= maxSampleSize) {
      appendValue(decodedVector, row, type, pool);
      return;
    }

    int64_t replaceIndex = -1;
    if (!shouldReplaceSample(decodedVector, row, replaceIndex)) {
      return;
    }

    replaceValueAt(decodedVector, row, replaceIndex);
  }

 private:
  bool shouldReplaceSample(
      const DecodedVector& decodedVector,
      vector_size_t row,
      int64_t& replaceIndex) const {
    folly::ThreadLocalPRNG rng;
    std::uniform_int_distribution<int64_t> dist(0, processedCount - 1);
    int64_t randomPos = dist(rng);

    if (randomPos >= maxSampleSize) {
      return false;
    }

    replaceIndex = randomPos;
    return true;
  }

  void appendValue(
      const DecodedVector& decodedVector,
      vector_size_t row,
      const TypePtr& type,
      memory::MemoryPool* pool) {
    if (!samples) {
      samples = BaseVector::create(type, 0, pool);
    }
    auto currentSize = samples->size();
    samples->resize(currentSize + 1);
    samples->copy(decodedVector.base(), currentSize, row, 1);
  }

  void replaceValueAt(
      const DecodedVector& decodedVector,
      vector_size_t row,
      int64_t replaceIndex) {
    VELOX_DCHECK_GE(replaceIndex, 0);
    VELOX_DCHECK_LT(replaceIndex, samples->size());

    samples->copy(decodedVector.base(), replaceIndex, row, 1);
  }
};

} // namespace facebook::velox::aggregate::prestosql
