/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <string_view>
#include <utility>

namespace facebook::nimble::index {

/// Orders the encoded keys of a cluster index whose keys do not sort bytewise,
/// such as keys that end in a fixed-width suffix sorted in descending order.
/// A cluster index factory supplies one through
/// ClusterIndexFactory::keyComparator(); a null comparator means bytewise
/// order. Implementations must be thread-safe and outlive every index that
/// uses them.
class KeyComparator {
 public:
  virtual ~KeyComparator() = default;

  /// Returns a negative value, zero or a positive value when 'lhs' sorts
  /// before, with or after 'rhs'.
  virtual int compare(std::string_view lhs, std::string_view rhs) const = 0;
};

/// Compares 'lhs' with 'rhs' under 'comparator', or bytewise when it is null.
/// Checks 'comparator' on every call, so loops over keys use withKeyOrder().
inline int compareKeys(
    std::string_view lhs,
    std::string_view rhs,
    const KeyComparator* comparator) {
  return comparator == nullptr ? lhs.compare(rhs)
                               : comparator->compare(lhs, rhs);
}

/// Orders keys bytewise with an inlined compare.
struct BytewiseKeyOrder {
  /// Returns a negative value, zero or a positive value when 'lhs' sorts
  /// before, with or after 'rhs'.
  int compare(std::string_view lhs, std::string_view rhs) const {
    return lhs.compare(rhs);
  }

  /// Returns whether 'lhs' sorts before 'rhs', for standard algorithms such as
  /// std::lower_bound.
  bool operator()(std::string_view lhs, std::string_view rhs) const {
    return lhs < rhs;
  }
};

/// Orders keys under a KeyComparator, with the same methods as
/// BytewiseKeyOrder.
struct ComparatorKeyOrder {
  /// Returns a negative value, zero or a positive value when 'lhs' sorts
  /// before, with or after 'rhs'.
  int compare(std::string_view lhs, std::string_view rhs) const {
    return comparator->compare(lhs, rhs);
  }

  /// Returns whether 'lhs' sorts before 'rhs', for standard algorithms such as
  /// std::lower_bound.
  bool operator()(std::string_view lhs, std::string_view rhs) const {
    return comparator->compare(lhs, rhs) < 0;
  }

  /// Order of the keys. Never null.
  const KeyComparator* comparator;
};

/// Calls 'func' with the key order of 'comparator' and returns its result:
/// BytewiseKeyOrder when 'comparator' is null, ComparatorKeyOrder otherwise.
/// Checks 'comparator' once per call, so a loop over keys in 'func' compares
/// each key without a null check, and inlines the bytewise compare.
template <typename Func>
auto withKeyOrder(const KeyComparator* comparator, Func&& func) {
  if (comparator == nullptr) {
    return std::forward<Func>(func)(BytewiseKeyOrder{});
  }
  return std::forward<Func>(func)(ComparatorKeyOrder{comparator});
}

} // namespace facebook::nimble::index
