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

#include <random>
#include "velox/common/base/BitUtil.h"
#include "velox/functions/lib/aggregates/ValueList.h"
#include "velox/type/FloatingPointUtil.h"
#include "velox/type/StringView.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::aggregate::prestosql {

struct ReservoirSampleAccumulator {
  ValueList samples;
  ValueList initialSamples;
  int64_t initialSeenCount = 0;
  int64_t processedCount = 0;
  int32_t maxSampleSize = -1;

  explicit ReservoirSampleAccumulator() {}

  bool isInitialized() const {
    return maxSampleSize >= 0;
  }

  void addValueToReservoir(
      const DecodedVector& decodedVector,
      vector_size_t row,
      HashStringAllocator* allocator) {
    if (samples.size() < maxSampleSize) {
      // LOG(INFO) << "[46] value : " << decodedVector.valueAt<>(row);
      samples.appendValue(decodedVector, row, allocator);
      return;
    }

    int64_t replaceIndex = -1;
    if (!shouldReplaceSample(decodedVector, row, replaceIndex)) {
      return;
    }

    replaceSampleAt(decodedVector, row, replaceIndex, allocator);
  }

  bool shouldReplaceSample(
      const DecodedVector& decodedVector,
      vector_size_t row,
      int64_t& replaceIndex) {
    uint64_t valueHash = computeValueHash(decodedVector, row);
    uint64_t seed =
        bits::hashMix(valueHash, static_cast<uint64_t>(processedCount));

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist(1, processedCount);
    int64_t randomPos = dist(rng);

    if (randomPos > maxSampleSize) {
      return false;
    }

    replaceIndex = randomPos - 1;
    return true;
  }

  void replaceSampleAt(
      const DecodedVector& newValue,
      vector_size_t newValueRow,
      int64_t replaceIndex,
      HashStringAllocator* allocator) {
    auto sampleSize = samples.size();
    auto tempVector = BaseVector::create(
        newValue.base()->type(), sampleSize, allocator->pool());

    ValueListReader reader(samples);
    for (vector_size_t i = 0; i < sampleSize; i++) {
      reader.next(*tempVector, i);
    }

    tempVector->copy(newValue.base(), replaceIndex, newValueRow, 1);

    DecodedVector decodedTemp(*tempVector);
    ValueList newSamples;
    for (vector_size_t i = 0; i < sampleSize; i++) {
      newSamples.appendValue(decodedTemp, i, allocator);
    }

    samples.free(allocator);
    samples = std::move(newSamples);
  }

  template <TypeKind kind>
  uint64_t computeValueHashTyped(
      const DecodedVector& decodedVector,
      vector_size_t row) const {
    using T = typename TypeTraits<kind>::NativeType;
    if constexpr (
        kind == TypeKind::TINYINT || kind == TypeKind::SMALLINT ||
        kind == TypeKind::INTEGER || kind == TypeKind::BIGINT) {
      return static_cast<uint64_t>(decodedVector.valueAt<T>(row));
    } else if constexpr (kind == TypeKind::REAL) {
      return util::floating_point::NaNAwareHash<float>()(
          decodedVector.valueAt<T>(row));
    } else if constexpr (kind == TypeKind::DOUBLE) {
      return util::floating_point::NaNAwareHash<double>()(
          decodedVector.valueAt<T>(row));
    } else if constexpr (
        kind == TypeKind::VARCHAR || kind == TypeKind::VARBINARY) {
      return folly::hasher<StringView>()(decodedVector.valueAt<T>(row));
    } else if constexpr (kind == TypeKind::TIMESTAMP) {
      return decodedVector.valueAt<T>(row).toMillis();
    } else if constexpr (
        kind == TypeKind::ROW || kind == TypeKind::ARRAY ||
        kind == TypeKind::MAP) {
      return decodedVector.base()->hashValueAt(decodedVector.index(row));
    } else {
      VELOX_UNSUPPORTED(
          "Unsupported type for reservoir sampling: {}",
          decodedVector.base()->type()->toString());
    }
  }

  uint64_t computeValueHash(
      const DecodedVector& decodedVector,
      vector_size_t row) const {
    VELOX_DCHECK_LT(row, decodedVector.size(), "Row index out of bounds");

    if (decodedVector.isNullAt(row)) {
      return 0;
    }

    auto typeKind = decodedVector.base()->typeKind();

    if (typeKind == TypeKind::BOOLEAN) {
      return decodedVector.valueAt<bool>(row) ? 1 : 0;
    }

    return VELOX_DYNAMIC_TYPE_DISPATCH(
        computeValueHashTyped, typeKind, decodedVector, row);
  }
};

} // namespace facebook::velox::aggregate::prestosql
