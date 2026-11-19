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

#include <cstdint>

namespace facebook::nimble {

/// Decides when TabletWriter closes a stripe group, against
/// TabletWriter::Options::metadataFlushThreshold. A group flushes once its
/// estimated metadata size, 4 bytes plus the bytes per entry for each entry it
/// stores, reaches the threshold. An entry locates a stream id in a stripe: its
/// offset and size, and with the sparse layout its stripe index.
///
/// The first group is estimated at 13 bytes per entry: 8 for the offset, 4 for
/// the size and 1 for the compression type. That overstates the serialized
/// entries of every layout, so the first group closes below the threshold,
/// unless its streams hold only a few entries each and their fixed cost
/// dominates. Each written group then replaces the bytes per entry with its own
/// serialized bytes per entry, since the groups of a file have about the same
/// ratio. So from the second group on, groups close near the threshold, and
/// each group is serialized once, to write it.
class StripeGroupFlushPolicy {
 public:
  explicit StripeGroupFlushPolicy(uint32_t thresholdBytes);

  /// Returns whether to write the current group, which stores `numEntries`
  /// entries.
  bool shouldFlush(uint64_t numEntries) const;

  /// Records that a written group of `numEntries` entries serialized to
  /// `encodedBytes`, so later groups are estimated at its bytes per entry.
  void onFlush(uint64_t numEntries, uint64_t encodedBytes);

 private:
  // Metadata size, in bytes, at which a group flushes.
  const uint64_t thresholdBytes_;

  // Estimated metadata bytes per entry.
  double estimatedBytesPerEntry_;
};

} // namespace facebook::nimble
