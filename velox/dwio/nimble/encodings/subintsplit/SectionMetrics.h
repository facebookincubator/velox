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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h" // @manual=fbsource//third-party/abseil-cpp:container__flat_hash_map

// Per-section metric collection for the SubIntSplit DP planner.
//
// Deliberately minimal: only the statistics the cost models read, with no HLL,
// no entropy, and no residual-frame tracking. Nimble's Statistics<T> already
// handles those heavier computations for full-stream outer selection; this
// collector handles the 64×64 bit-range grid on a small sample.

namespace facebook::nimble::subintsplit {

enum class MetricFlag : uint32_t {
  None = 0,
  MinMax = 1u << 0,
  RunStats = 1u << 1,
  UniqueCount = 1u << 2,
  DominantValue = 1u << 3,
  All = (1u << 4) - 1,
};
using MetricFlags = uint32_t;

inline constexpr MetricFlags operator|(
    MetricFlag lhs,
    MetricFlag rhs) noexcept {
  return static_cast<MetricFlags>(lhs) | static_cast<MetricFlags>(rhs);
}

inline constexpr MetricFlags operator|(
    MetricFlags lhs,
    MetricFlag rhs) noexcept {
  return lhs | static_cast<MetricFlags>(rhs);
}

inline constexpr bool hasFlag(MetricFlags flags, MetricFlag flag) noexcept {
  return (flags & static_cast<MetricFlags>(flag)) != 0;
}

struct SectionMetrics {
  uint64_t min{0};
  uint64_t max{0};
  uint64_t range{0};

  size_t uniqueCount{0};
  bool uniqueCountCapped{false};

  /// Frequency of the most common value. Used by the MainlyConstant cost model.
  /// Unreliable (and flagged capped) once cardinality exceeds the unique cap.
  size_t dominantCount{0};
  bool dominantCountCapped{false};

  size_t runCount{0};
  double avgRunLength{0.0};
};

/// Frequency statistics over one section's values: how many distinct values it
/// holds and how often the most common one appears.
///
/// Two backends share this interface. Narrow sections index a flat table
/// directly; wider ones fall back to a hash map with a cardinality cap. The DP
/// evaluates every [bitStart, bitEnd] pair, so most candidate sections are
/// narrow even when the stream's active range is wide, and on a production
/// ctr_mbl stream the hash map was 41% of encode time.
class FrequencyCounter {
 public:
  /// Cardinality beyond which unique counting stops and reports `capped`.
  /// Lighter than a full HLL and enough for the cost models, which only branch
  /// on "low cardinality" vs "not".
  static constexpr size_t kUniqueCountCap = 1 << 14;

  /// Sections at most this wide are counted with a direct-indexed table. 16
  /// bits keeps the table at 256KB and needs no cap, since a 16-bit section
  /// cannot exceed 65536 distinct values.
  static constexpr int kDirectIndexBits = 16;

  /// True when `bitWidth` is narrow enough for the direct-indexed table.
  /// `bitWidth` of 0 means unknown, which forces the hash path.
  static bool fitsDirectIndex(int bitWidth) noexcept {
    return bitWidth > 0 && bitWidth <= kDirectIndexBits;
  }

  struct Result {
    size_t uniqueCount{0};
    uint32_t dominantCount{0};
    bool capped{false};
  };

  /// Counts `values` with the direct-indexed table. Only the touched slots are
  /// reset afterwards, so the table stays reusable across the grid's thousands
  /// of section evaluations without ever being cleared in bulk.
  Result countDirect(const std::vector<uint64_t>& values);

  /// Counts `values` with the hash map, stopping unique tracking once
  /// kUniqueCountCap is exceeded.
  Result countHashed(const std::vector<uint64_t>& values);

 private:
  absl::flat_hash_map<uint64_t, uint32_t> frequencies_;

  // Direct-indexed counts, with the list of dirtied slots so resetting is
  // O(values) rather than O(table).
  std::vector<uint32_t> directCounts_;
  std::vector<uint32_t> touchedSlots_;
};

/// Computes the metrics the cost models need for one extracted bit range.
///
/// Holds the frequency counter across calls so the split selector, which calls
/// compute() for every cell of an O(kBits^2) grid, does not reallocate it.
class MetricCollector {
 public:
  /// `bitWidth` is the section's width in bits; pass 0 if unknown. Metrics not
  /// named in `flags` are left at their default.
  SectionMetrics compute(
      const std::vector<uint64_t>& values,
      MetricFlags flags = static_cast<MetricFlags>(MetricFlag::All),
      int bitWidth = 0);

 private:
  FrequencyCounter frequencies_;
};

} // namespace facebook::nimble::subintsplit
