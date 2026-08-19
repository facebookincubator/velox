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
#include <limits>
#include <vector>

#include "absl/container/flat_hash_map.h" // @manual=fbsource//third-party/abseil-cpp:container__flat_hash_map

// Lightweight per-segment metric collection for SubIntSplitEncoding's DP
// planner. Deliberately minimal: only the statistics needed by the cost
// models are computed, with no HLL, no entropy, and no residual-frame
// tracking. Nimble's Statistics<T> class already handles those heavier
// computations for full-stream outer selection; this collector handles the
// 64×64 bit-range grid on a small sample.

namespace facebook::nimble::detail::subintsplit {

enum class MetricFlag : uint32_t {
  None = 0,
  MinMax = 1u << 0, // min, max, range
  RunStats = 1u << 1, // runCount, avgRunLength
  UniqueCount = 1u << 2, // uniqueCount (raw, capped)
  DominantValue = 1u << 3, // dominantCount (most-frequent value's frequency)
  All = (1u << 4) - 1,
};
using MetricFlags = uint32_t;

inline constexpr MetricFlags operator|(MetricFlag a, MetricFlag b) noexcept {
  return static_cast<MetricFlags>(a) | static_cast<MetricFlags>(b);
}
inline constexpr MetricFlags operator|(MetricFlags a, MetricFlag b) noexcept {
  return a | static_cast<MetricFlags>(b);
}
inline constexpr bool hasFlag(MetricFlags flags, MetricFlag f) noexcept {
  return (flags & static_cast<MetricFlags>(f)) != 0;
}

struct SegmentMetrics {
  uint64_t min{0};
  uint64_t max{0};
  uint64_t range{0};

  size_t uniqueCount{0};
  bool uniqueCountCapped{false};

  // Frequency of the most common value. Used by the MainlyConstant cost model.
  // Unreliable (and flagged capped) once cardinality exceeds the unique cap.
  size_t dominantCount{0};
  bool dominantCountCapped{false};

  size_t runCount{0};
  double avgRunLength{0.0};
};

// Single-pass metric collector for extracted bit-range values.
// Uses a frequency map for unique/dominant counting (capped at
// kUniqueCountCap) and a running prev-value comparison for run counting.
//
// The frequency map is a reusable member: the split selector calls compute()
// for every bit-range in an O(kBits^2) grid, so allocating a fresh map per
// call dominated encode time. Clearing and reusing one map preserves its
// capacity across calls and avoids that allocation churn.
class MetricCollector {
 public:
  static constexpr size_t kUniqueCountCap = 1
      << 14; // 16K cap (lighter than full HLL)

  SegmentMetrics compute(
      const std::vector<uint64_t>& values,
      MetricFlags flags = static_cast<MetricFlags>(MetricFlag::All)) {
    const bool doMin = hasFlag(flags, MetricFlag::MinMax);
    const bool doRun = hasFlag(flags, MetricFlag::RunStats);
    const bool doUniq = hasFlag(flags, MetricFlag::UniqueCount);
    const bool doDominant = hasFlag(flags, MetricFlag::DominantValue);
    // Unique count and dominant value share a single frequency map pass.
    const bool doFreq = doUniq || doDominant;

    SegmentMetrics out;
    const size_t n = values.size();
    if (n == 0) {
      return out;
    }

    const uint64_t v0 = values[0];
    if (doMin) {
      out.min = v0;
      out.max = v0;
    }
    if (doRun) {
      out.runCount = 1;
    }

    bool capped = false;
    uint32_t maxCount = 0;
    if (doFreq) {
      freqMap_.clear();
      freqMap_.reserve(std::min(n, kUniqueCountCap));
      freqMap_.emplace(v0, 1u);
      maxCount = 1;
    }

    uint64_t prev = v0;
    for (size_t i = 1; i < n; ++i) {
      const uint64_t v = values[i];
      if (doMin) {
        if (v < out.min) {
          out.min = v;
        }
        if (v > out.max) {
          out.max = v;
        }
      }
      if (doRun && v != prev) {
        ++out.runCount;
      }
      if (doFreq && !capped) {
        auto [it, inserted] = freqMap_.try_emplace(v, 0u);
        const uint32_t count = ++it->second;
        if (count > maxCount) {
          maxCount = count;
        }
        if (inserted && freqMap_.size() > kUniqueCountCap) {
          capped = true;
        }
      }
      prev = v;
    }

    if (doMin) {
      out.range = out.max - out.min;
    }
    if (doRun) {
      out.avgRunLength =
          static_cast<double>(n) / static_cast<double>(out.runCount);
    }
    if (doUniq) {
      out.uniqueCount = capped ? (kUniqueCountCap + 1) : freqMap_.size();
      out.uniqueCountCapped = capped;
    }
    if (doDominant) {
      out.dominantCount = maxCount;
      out.dominantCountCapped = capped;
    }

    return out;
  }

 private:
  // frequency map for unique/dominant counting.
  absl::flat_hash_map<uint64_t, uint32_t> freqMap_;
};

} // namespace facebook::nimble::detail::subintsplit
