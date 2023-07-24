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
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::aggregate::prestosql {

/// A list of non-null values of complex type stored in possibly non-contiguous
/// memory allocated via HashStringAllocator. Provides index-based access to
/// individual elements. Allows removing elements from the end of the list.
///
/// Designed to be used together with F14FastSet or F14FastMap.
///
/// Used in aggregations that deduplicate complex values, e.g. set_agg and
/// set_union.
class AddressableNonNullValueList {
 public:
  struct Hash {
    size_t operator()(HashStringAllocator::Position position) const {
      return AddressableNonNullValueList::readHash(position);
    }
  };

  struct EqualTo {
    const TypePtr& type;

    bool operator()(
        HashStringAllocator::Position left,
        HashStringAllocator::Position right) const {
      return AddressableNonNullValueList::equalTo(left, right, type);
    }
  };

  /// Append a non-null value to the end of the list. Returns 'index' that can
  /// be used to access the value later.
  HashStringAllocator::Position append(
      const DecodedVector& decoded,
      vector_size_t index,
      HashStringAllocator* allocator);

  /// Removes last element. 'position' must be a value returned from the latest
  /// call to 'append'.
  void removeLast(HashStringAllocator::Position position) {
    currentPosition_ = position;
    --size_;
  }

  /// Returns number of elements in the list.
  int32_t size() const {
    return size_;
  }

  /// Returns true if elements at 'left' and 'right' are equal.
  static bool equalTo(
      HashStringAllocator::Position left,
      HashStringAllocator::Position right,
      const TypePtr& type);

  /// Returns the hash of the specified element.
  static uint64_t readHash(HashStringAllocator::Position position);

  /// Copies the specified element to 'result[index]'.
  static void read(
      HashStringAllocator::Position position,
      BaseVector& result,
      vector_size_t index);

  void free(HashStringAllocator& allocator) {
    if (size_ > 0) {
      allocator.free(firstHeader_);
    }
  }

 private:
  // Memory allocation (potentially multi-part).
  HashStringAllocator::Header* firstHeader_{nullptr};
  HashStringAllocator::Position currentPosition_{nullptr, nullptr};

  // Number of values added.
  uint32_t size_{0};
};

} // namespace facebook::velox::aggregate::prestosql
