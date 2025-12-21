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
#include "velox/core/QueryConfig.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/SetDigest.h"

namespace facebook::velox::functions {

/// Scalar function: cardinality(setdigest) -> bigint
/// Returns the estimated cardinality of the set digest.
template <typename T>
struct CardinalitySetDigestFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  std::unique_ptr<HashStringAllocator> allocator_;

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      memory::MemoryPool* pool,
      const arg_type<Varbinary>* /*input*/) {
    VELOX_CHECK_NOT_NULL(
        pool, "cardinality requires MemoryPool for HashStringAllocator");
    allocator_ = std::make_unique<HashStringAllocator>(pool);
  }

  FOLLY_ALWAYS_INLINE Status
  call(out_type<int64_t>& result, const arg_type<Varbinary>& input) {
    SetDigest<int64_t> digest(allocator_.get());
    VELOX_RETURN_NOT_OK(digest.deserialize(input.data(), input.size()));
    result = digest.cardinality();
    return Status::OK();
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

  std::unique_ptr<HashStringAllocator> allocator_;

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      memory::MemoryPool* pool,
      const arg_type<Varbinary>* /*input1*/,
      const arg_type<Varbinary>* /*input2*/) {
    VELOX_CHECK_NOT_NULL(
        pool,
        "intersection_cardinality requires MemoryPool for HashStringAllocator");
    allocator_ = std::make_unique<HashStringAllocator>(pool);
  }

  FOLLY_ALWAYS_INLINE Status call(
      out_type<int64_t>& result,
      const arg_type<Varbinary>& input1,
      const arg_type<Varbinary>& input2) {
    using SetDigestType = SetDigest<int64_t>;

    SetDigestType digest1(allocator_.get());
    VELOX_RETURN_NOT_OK(digest1.deserialize(input1.data(), input1.size()));

    SetDigestType digest2(allocator_.get());
    VELOX_RETURN_NOT_OK(digest2.deserialize(input2.data(), input2.size()));

    // If both are exact, use exact intersection
    if (digest1.isExact() && digest2.isExact()) {
      result = SetDigestType::exactIntersectionCardinality(digest1, digest2);
      return Status::OK();
    }

    // Otherwise, estimate using Jaccard index (matching Java behavior)
    int64_t cardinality1 = digest1.cardinality();
    int64_t cardinality2 = digest2.cardinality();
    double jaccard = SetDigestType::jaccardIndex(digest1, digest2);
    digest1.mergeWith(digest2);
    int64_t estimatedIntersection =
        std::llround(jaccard * digest1.cardinality());

    // When one set is much smaller and approaches being a subset of the other,
    // the computed cardinality may exceed the smaller set's cardinality.
    // In this case, the smaller cardinality is a better estimate.
    result =
        std::min(estimatedIntersection, std::min(cardinality1, cardinality2));
    return Status::OK();
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

  std::unique_ptr<HashStringAllocator> allocator_;

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      memory::MemoryPool* pool,
      const arg_type<Varbinary>* /*input1*/,
      const arg_type<Varbinary>* /*input2*/) {
    VELOX_CHECK_NOT_NULL(
        pool, "jaccard_index requires MemoryPool for HashStringAllocator");
    allocator_ = std::make_unique<HashStringAllocator>(pool);
  }

  FOLLY_ALWAYS_INLINE Status call(
      out_type<double>& result,
      const arg_type<Varbinary>& input1,
      const arg_type<Varbinary>& input2) {
    using SetDigestType = SetDigest<int64_t>;

    SetDigestType digest1(allocator_.get());
    VELOX_RETURN_NOT_OK(digest1.deserialize(input1.data(), input1.size()));

    SetDigestType digest2(allocator_.get());
    VELOX_RETURN_NOT_OK(digest2.deserialize(input2.data(), input2.size()));

    result = SetDigestType::jaccardIndex(digest1, digest2);
    return Status::OK();
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

  std::unique_ptr<HashStringAllocator> allocator_;

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      memory::MemoryPool* pool,
      const arg_type<Varbinary>* /*input*/) {
    VELOX_CHECK_NOT_NULL(
        pool, "hash_counts requires MemoryPool for HashStringAllocator");
    allocator_ = std::make_unique<HashStringAllocator>(pool);
  }

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Map<int64_t, int16_t>>& result,
      const arg_type<Varbinary>& input) {
    SetDigest<int64_t> digest(allocator_.get());
    VELOX_RETURN_NOT_OK(digest.deserialize(input.data(), input.size()));

    auto hashCounts = digest.getHashCounts();
    result.reserve(hashCounts.size());

    for (const auto& [key, value] : hashCounts) {
      auto item = result.add_item();
      std::get<0>(item) = key;
      std::get<1>(item) = value;
    }

    return Status::OK();
  }
};

} // namespace facebook::velox::functions
