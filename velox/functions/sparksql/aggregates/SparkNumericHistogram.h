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

#include <cstdint>
#include <string_view>
#include <vector>

#include <folly/Range.h>

#include "velox/common/memory/HashStringAllocator.h"

namespace facebook::velox::functions::aggregate::sparksql {

/// Maintains Spark histogram_numeric bins in operation-history order. Owns all
/// bins and merge scratch through the supplied allocator, which must outlive
/// this object. Empty bins represent a null final result; a known capacity
/// still produces a non-null, eight-byte intermediate. Null intermediate
/// transport is handled by the caller, not by passing an empty payload to
/// mergeSerialized.
class SparkNumericHistogram {
 public:
  /// Keeps a center and its height together, including across allocation
  /// errors.
  struct Bin {
    /// Stores the normalized binary64 center, including non-finite values.
    double x;
    /// Stores the binary64 height without an integral conversion.
    double y;
  };

  /// Starts unbound, with an independent Java Random stream seeded with 31183.
  explicit SparkNumericHistogram(HashStringAllocator* allocator);

  /// Binds numBins in [2, INT32_MAX], or checks agreement with an existing
  /// binding. Does not allocate bins, clear contents, or reset the random
  /// stream.
  void initialize(int32_t numBins);

  /// Adds a non-null, already-normalized value using Spark's primitive midpoint
  /// search, then trims if necessary. Requires a bound capacity.
  void add(double value);

  /// Validates and merges a non-null, versionless Spark payload. Rejects
  /// invalid counts, inexact lengths and unsupported sizes before allocation or
  /// mutation. An empty destination adopts the source capacity and incoming
  /// order. A non-empty destination retains its capacity, stably sorts
  /// destination-before-source using Java Double.compare, then trims. Does not
  /// retain payload storage or import a source random stream.
  void mergeSerialized(std::string_view payload);

  /// Computes the checked, int-sized wire length for an actual bin count,
  /// including the eight-byte header, without allocating storage. Rejects
  /// counts whose output would exceed the StringWriter/Java payload limit.
  /// Used to preflight state growth, payload parsing and serialization; the
  /// count is not the configured numBins capacity.
  static int32_t serializedSize(uint64_t numUsedBins);

  /// Returns the checked, int-sized wire length, including the eight-byte
  /// header. Requires a bound capacity, even when there are no bins.
  int32_t serializedSize() const;

  /// Writes exactly serializedSize() bytes to caller-owned storage. Preserves
  /// stored order and raw double bits; does not sort, trim, or advance the RNG.
  void serialize(char* output) const;

  /// Returns the bound capacity, or zero if no capacity is known yet.
  int32_t numBins() const {
    return numBins_;
  }

  /// Exposes stored order without normalizing centers or combining duplicates.
  /// The view is invalidated by mutation or destruction of this object.
  folly::Range<const Bin*> bins() const {
    return {bins_.data(), bins_.size()};
  }

 private:
  // Couples ownership and alignment for all persistent bin storage.
  using BinVector = std::vector<Bin, AlignedStlAllocator<Bin, alignof(Bin)>>;

  // Removes adjacent pairs using Spark's exact gap scan and arithmetic order.
  void trim();

  // Advances the 48-bit Java Random LCG and returns the requested high bits.
  uint32_t nextRandomBits(uint32_t bits);

  // Combines 26 and 27 bits from consecutive transitions into a binary64 value.
  double nextRandomDouble();

  // Zero denotes an unknown capacity, never a valid initialized histogram.
  int32_t numBins_{0};
  // Retains paired bins in raw insertion or most recent merge order.
  BinVector bins_;
  // Persists across operations but is deliberately absent from the wire format.
  uint64_t randomState_;
};

} // namespace facebook::velox::functions::aggregate::sparksql
