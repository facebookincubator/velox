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

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {

/// Bits needed to hold every one of `keys`: the smallest w for which every key
/// is below 1 << w. Zero when every key is zero. Pairs with RadixSort, whose
/// pass count follows the width of the keys rather than their type's width.
template <typename Key>
int significantBits(std::span<const Key> keys) {
  static_assert(
      std::is_unsigned_v<Key>, "significantBits needs unsigned keys.");
  Key combined = 0;
  for (const Key key : keys) {
    combined |= key;
  }
  return std::bit_width(combined);
}

/// Stable least-significant-digit radix sort, with the number of passes set by
/// how wide the keys actually are rather than by the width of their type: a
/// ten-bit key is one pass, not eight. Stability means it substitutes for
/// std::stable_sort with a less-than comparator on the key, without moving
/// any output that depends on the order.
///
/// Holds its scratch buffers, so a caller that sorts repeatedly should keep an
/// instance rather than construct one per call.
///
/// TODO: SubIntSplitEncodingView::radixSortBySource does this same job for its
/// own row type and could adopt this instead.
template <typename Item>
class RadixSort {
 public:
  /// Stably sorts `items` by `keyOf(item)`, which must return an unsigned
  /// integer below 1 << keyBits. A keyBits wider than the keys really are
  /// costs passes and nothing else; one narrower is a caller error and sorts
  /// on the low bits only.
  template <typename KeyFn>
  void sortStable(std::span<Item> items, KeyFn&& keyOf, int keyBits) {
    static_assert(
        std::is_unsigned_v<
            std::remove_cvref_t<std::invoke_result_t<KeyFn&, const Item&>>>,
        "RadixSort needs an unsigned key.");

    const size_t count = items.size();
    if (count < 2 || keyBits <= 0) {
      return;
    }
    NIMBLE_CHECK_LE(
        count,
        size_t{std::numeric_limits<uint32_t>::max()},
        "RadixSort counts bucket offsets in 32 bits.");

    // Narrows toward kMinDigitBits on short inputs, since a wide bucket array
    // costs more to clear than the pass it saves.
    int digitBits = kMaxDigitBits;
    while (digitBits > kMinDigitBits && (size_t{1} << digitBits) > count) {
      --digitBits;
    }
    const int passes = (keyBits + digitBits - 1) / digitBits;
    // Rebalanced across the passes needed, avoiding an oversized final pass.
    digitBits = (keyBits + passes - 1) / passes;
    const size_t buckets = size_t{1} << digitBits;
    const uint64_t digitMask = buckets - 1;

    scratch_.resize(count);
    counts_.assign(buckets, 0u);

    // Starting in the scratch when the pass count is odd lands the final
    // result back in the caller's array.
    Item* source = items.data();
    Item* destination = scratch_.data();
    if (passes % 2 == 1) {
      std::copy(items.begin(), items.end(), scratch_.begin());
      std::swap(source, destination);
    }

    for (int pass = 0; pass < passes; ++pass) {
      const int shift = pass * digitBits;
      if (pass > 0) {
        std::fill(counts_.begin(), counts_.end(), 0u);
      }
      for (size_t i = 0; i < count; ++i) {
        const size_t digit = static_cast<size_t>(
            (static_cast<uint64_t>(keyOf(source[i])) >> shift) & digitMask);
        ++counts_[digit];
      }
      uint32_t offset = 0;
      for (size_t bucket = 0; bucket < buckets; ++bucket) {
        const uint32_t bucketCount = counts_[bucket];
        counts_[bucket] = offset;
        offset += bucketCount;
      }
      // Walked forward, appending to each bucket, to keep equal keys in the
      // order they arrived.
      for (size_t i = 0; i < count; ++i) {
        const size_t digit = static_cast<size_t>(
            (static_cast<uint64_t>(keyOf(source[i])) >> shift) & digitMask);
        destination[counts_[digit]++] = source[i];
      }
      std::swap(source, destination);
    }
  }

 private:
  // A digit no wider than this keeps the count table, and the scatter's write
  // positions, within cache; wider digits save a pass but lose more than that
  // to cache misses.
  static constexpr int kMaxDigitBits = 12;
  // Narrowing past this trades a pass for a table too small to be worth it on
  // any input large enough to reach for a radix sort.
  static constexpr int kMinDigitBits = 8;

  std::vector<Item> scratch_;
  std::vector<uint32_t> counts_;
};

} // namespace facebook::nimble
