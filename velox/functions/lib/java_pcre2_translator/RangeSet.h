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
//
// Originally authored by Oleksii PELYKH for pcre4j; ported from
// org.pcre4j.regex.translate.RangeSet (Java) under Apache-2.0 by the
// same author for inclusion in Velox.
//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace facebook::velox::functions::java_pcre2_translator {

/// Immutable, sorted, disjoint set of Unicode code-point ranges over
/// [0, 0x10FFFF].  Internally stored as a flat `std::vector<int>` of the
/// form `[lo0, hi0, lo1, hi1, ...]` where
/// `lo0 <= hi0 < lo1 <= hi1 < ...`.  All endpoints are inclusive.
class RangeSet {
 public:
  /// The full Unicode code-point space [0, 0x10FFFF].
  static constexpr std::int32_t kMaxCp = 0x10FFFF;

  /// Returns the empty set.
  static const RangeSet& empty();

  /// Returns the set containing every code point [0, kMaxCp].
  static const RangeSet& all();

  /// Creates a set containing the single code point `cp`.
  /// Throws `std::invalid_argument` when `cp` is out of range.
  static RangeSet single(std::int32_t cp);

  /// Creates a set containing the range [lo, hi] inclusive.
  /// Throws `std::invalid_argument` when the range is invalid.
  static RangeSet range(std::int32_t lo, std::int32_t hi);

  /// Creates a set from already sorted [lo, hi] pairs, merging adjacent spans.
  static RangeSet fromSortedPairs(std::vector<std::int32_t> pairs);

  /// Returns the union of this set and `other`.
  RangeSet unionWith(const RangeSet& other) const;

  /// Returns the intersection of this set and `other`.
  RangeSet intersect(const RangeSet& other) const;

  /// Returns the complement of this set within [0, kMaxCp].
  RangeSet complement() const;

  /// Returns `this - other`.
  RangeSet subtract(const RangeSet& other) const;

  /// Returns true iff this set contains no code points.
  bool isEmpty() const {
    return ranges_.empty();
  }

  /// Returns true iff this set contains `cp`.
  bool contains(std::int32_t cp) const;

  /// Emits the content of this set as a PCRE2 character-class body — i.e.
  /// what would appear between `[` and `]`.  Printable ASCII in the
  /// range 0x20–0x7E is emitted literally except for `\`, `]`, `^`, `-`
  /// which are backslash-escaped; all other code points are emitted as
  /// `\x{HH...}`.  Contiguous ranges of two-or-more code points are
  /// emitted as `lo-hi`.
  std::string toPcre2ClassBody() const;

  /// Number of contiguous ranges (for testing).
  int rangeCount() const {
    return static_cast<int>(ranges_.size() / 2);
  }

  /// Equality based on the normalised range vector.
  bool operator==(const RangeSet& other) const {
    return ranges_ == other.ranges_;
  }
  bool operator!=(const RangeSet& other) const {
    return !(*this == other);
  }

 private:
  explicit RangeSet(std::vector<std::int32_t> ranges)
      : ranges_(std::move(ranges)) {}

  /// Merges overlapping/adjacent pairs in `raw` (which must already be
  /// sorted by `lo`) and returns the resulting `RangeSet`.
  static RangeSet normalise(std::vector<std::int32_t>&& raw);

  /// Sorted, non-overlapping, non-adjacent pairs.
  std::vector<std::int32_t> ranges_;
};

} // namespace facebook::velox::functions::java_pcre2_translator
