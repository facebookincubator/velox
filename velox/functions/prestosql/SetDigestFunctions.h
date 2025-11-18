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

#include "velox/common/memory/Memory.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/SetDigest.h"

namespace facebook::velox::functions {

/// Scalar function: cardinality(setdigest) -> bigint
/// Returns the estimated cardinality of the set digest.
template <typename T>
struct CardinalitySetDigestFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<int64_t>& result,
      const arg_type<Varbinary>& input) {
    auto pool = memory::memoryManager()->addLeafPool();
    HashStringAllocator allocator(pool.get());

    SetDigest digest(&allocator);
    digest.deserialize(input.data(), input.size());

    result = digest.cardinality();
    return true;
  }
};

/// Scalar function: intersection_cardinality(setdigest, setdigest) -> bigint
/// Returns the intersection cardinality between two set digests.
/// - If both digests are exact: returns exact intersection count
/// - If either digest is approximate: returns estimated intersection using
/// Jaccard index
template <typename T>
struct IntersectionCardinalityFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<int64_t>& result,
      const arg_type<Varbinary>& input1,
      const arg_type<Varbinary>& input2) {
    auto pool = memory::memoryManager()->addLeafPool();
    HashStringAllocator allocator(pool.get());

    SetDigest digest1(&allocator);
    digest1.deserialize(input1.data(), input1.size());

    SetDigest digest2(&allocator);
    digest2.deserialize(input2.data(), input2.size());

    // If both are exact, use exact intersection
    if (digest1.isExact() && digest2.isExact()) {
      result = SetDigest::exactIntersectionCardinality(digest1, digest2);
      return true;
    }

    // Otherwise, estimate using Jaccard index (matching Java behavior)
    int64_t cardinality1 = digest1.cardinality();
    int64_t cardinality2 = digest2.cardinality();
    double jaccard = SetDigest::jaccardIndex(digest1, digest2);
    digest1.mergeWith(digest2);
    int64_t estimatedIntersection =
        std::llround(jaccard * digest1.cardinality());

    // When one set is much smaller and approaches being a subset of the other,
    // the computed cardinality may exceed the smaller set's cardinality.
    // In this case, the smaller cardinality is a better estimate.
    result =
        std::min(estimatedIntersection, std::min(cardinality1, cardinality2));
    return true;
  }
};

/// Scalar function: jaccard_index(setdigest, setdigest) -> double
/// Returns the Jaccard index (similarity coefficient) between two set digests.
/// The Jaccard index is a value in [0, 1] where:
/// - 1.0 means the sets are identical
/// - 0.0 means the sets are disjoint (no overlap)
/// Uses MinHash estimation for efficient computation.
template <typename T>
struct JaccardIndexFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<double>& result,
      const arg_type<Varbinary>& input1,
      const arg_type<Varbinary>& input2) {
    auto pool = memory::memoryManager()->addLeafPool();
    HashStringAllocator allocator(pool.get());

    SetDigest digest1(&allocator);
    digest1.deserialize(input1.data(), input1.size());

    SetDigest digest2(&allocator);
    digest2.deserialize(input2.data(), input2.size());

    result = SetDigest::jaccardIndex(digest1, digest2);
    return true;
  }
};

/// Scalar function: hash_counts(setdigest) -> map(bigint, smallint)
/// Returns a map of hash values to their occurrence counts.
/// This exposes the MinHash component of the SetDigest, showing which
/// hash values were tracked and how many times each value was added.
/// Only available when the digest is exact (cardinality < maxHashes).
template <typename T>
struct HashCountsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Map<int64_t, int16_t>>& result,
      const arg_type<Varbinary>& input) {
    auto pool = memory::memoryManager()->addLeafPool();
    HashStringAllocator allocator(pool.get());

    SetDigest digest(&allocator);
    digest.deserialize(input.data(), input.size());

    auto hashCounts = digest.getHashCounts();
    result.reserve(hashCounts.size());

    for (const auto& [key, value] : hashCounts) {
      auto item = result.add_item();
      std::get<0>(item) = key;
      std::get<1>(item) = value;
    }

    return true;
  }
};

} // namespace facebook::velox::functions
