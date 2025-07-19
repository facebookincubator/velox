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

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/aggregates/sfm/SfmSketch.h"
#include "velox/functions/prestosql/types/SfmSketchType.h"

namespace facebook::velox::functions {

// Helper function to create empty SfmSketch.
std::string createEmptySfmSketch(
    double epsilon,
    std::optional<int64_t> buckets = std::nullopt,
    std::optional<int64_t> precison = std::nullopt);

template <typename T>
struct SfmSketchCardinality {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  FOLLY_ALWAYS_INLINE void call(
      int64_t& result,
      const arg_type<SfmSketch>& input) {
    using SfmSketch = facebook::velox::functions::aggregate::SfmSketch;
    auto pool = memory::memoryManager()->addLeafPool();
    HashStringAllocator allocator(pool.get());
    result = SfmSketch::deserialize(
                 reinterpret_cast<const char*>(input.data()), &allocator)
                 .cardinality();
  }
};

template <typename T>
struct NoisyEmptyApproxSetSfm {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<SfmSketch>& result,
      const double& epsilon) {
    auto serialized = createEmptySfmSketch(epsilon);
    result.resize(serialized.size());
    // SfmSketch is a binary type, so we can just copy the serialized data.
    memcpy(result.data(), serialized.data(), serialized.size());
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<SfmSketch>& result,
      const double& epsilon,
      const int64_t& buckets) {
    auto serialized = createEmptySfmSketch(epsilon, buckets);
    result.resize(serialized.size());
    memcpy(result.data(), serialized.data(), serialized.size());
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<SfmSketch>& result,
      const double& epsilon,
      const int64_t& buckets,
      const int64_t& precision) {
    auto serialized = createEmptySfmSketch(epsilon, buckets, precision);
    result.resize(serialized.size());
    memcpy(result.data(), serialized.data(), serialized.size());
  }
};

template <typename T>
struct mergeSfmSketchArray {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<SfmSketch>& result,
      const arg_type<Array<SfmSketch>>& input) {
    if (input.size() == 0) {
      return false;
    }

    auto pool = memory::memoryManager()->addLeafPool();
    HashStringAllocator allocator(pool.get());

    // Find the first non-null sketch to start with.
    int firstValidIndex = -1;
    for (auto i = 0; i < input.size(); ++i) {
      if (input[i].has_value()) {
        firstValidIndex = i;
        break;
      }
    }

    if (firstValidIndex == -1) {
      return false;
    }

    using SfmSketch = facebook::velox::functions::aggregate::SfmSketch;
    // Deserialize the first valid sketch.
    const auto& firstValidData = input[firstValidIndex].value();
    auto mergedSketch = SfmSketch::deserialize(
        reinterpret_cast<const char*>(firstValidData.data()), &allocator);

    // Merge all remaining valid sketches.
    for (auto i = firstValidIndex + 1; i < input.size(); ++i) {
      if (!input[i].has_value()) {
        continue;
      }
      const auto& currentSketchData = input[i].value();
      auto currentSketch = SfmSketch::deserialize(
          reinterpret_cast<const char*>(currentSketchData.data()), &allocator);
      mergedSketch.mergeWith(currentSketch);
    }

    // Serialize the merged sketch back to result.
    auto serializedSize = mergedSketch.serializedSize();
    result.resize(serializedSize);
    mergedSketch.serialize(result.data());

    return true;
  }
};

} // namespace facebook::velox::functions
