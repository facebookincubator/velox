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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace facebook::nimble::mlidc {

// ---------------------------------------------------------------------------
// Order statistics over timing samples (nanoseconds).
//
// Helpers that accept a non-const vector are permitted to reorder it; they use
// std::nth_element rather than a full sort, so successive calls on the same
// vector each remain O(n).
// ---------------------------------------------------------------------------

/// Value at quantile p in [0, 1] by nearest-rank.  Returns 0 for empty input.
inline int64_t percentileOf(std::vector<int64_t>& v, double p) {
  if (v.empty()) {
    return 0;
  }
  const double clamped = std::clamp(p, 0.0, 1.0);
  size_t idx =
      static_cast<size_t>(clamped * static_cast<double>(v.size() - 1));
  idx = std::min(idx, v.size() - 1);
  auto nth = v.begin() + static_cast<std::ptrdiff_t>(idx);
  std::nth_element(v.begin(), nth, v.end());
  return *nth;
}

inline int64_t medianOf(std::vector<int64_t>& v) {
  return percentileOf(v, 0.5);
}

inline int64_t minOf(const std::vector<int64_t>& v) {
  if (v.empty()) {
    return 0;
  }
  return *std::min_element(v.begin(), v.end());
}

// ---------------------------------------------------------------------------
// TimingSummary: median, p90, and min for one timed measurement point.
//
// Prefer order statistics for latency data — a single descheduled iteration
// can dominate a mean, but barely moves the median or p90.
// ---------------------------------------------------------------------------
struct TimingSummary {
  int64_t median_ns{0};
  int64_t p90_ns{0};
  int64_t min_ns{0};
};

inline TimingSummary summarize(std::vector<int64_t>& samples) {
  TimingSummary s;
  s.min_ns = minOf(samples);
  s.median_ns = percentileOf(samples, 0.5);
  s.p90_ns = percentileOf(samples, 0.9);
  return s;
}

// ---------------------------------------------------------------------------
// MomentSummary: mean and Bessel-corrected standard deviation over a
// real-valued sample set.
//
// Used for derived quantities (compression ratios, throughputs) that do not
// suffer from the descheduling outliers that make order statistics preferable
// for raw latency.  stddev is 0 for a single sample.
// ---------------------------------------------------------------------------
struct MomentSummary {
  double mean{0.0};
  double stddev{0.0};
};

inline MomentSummary momentSummary(const std::vector<double>& values) {
  MomentSummary s;
  if (values.empty()) {
    return s;
  }
  s.mean = std::accumulate(values.begin(), values.end(), 0.0) /
      static_cast<double>(values.size());
  if (values.size() > 1) {
    double sqSum = 0.0;
    for (double v : values) {
      sqSum += (v - s.mean) * (v - s.mean);
    }
    s.stddev =
        std::sqrt(sqSum / static_cast<double>(values.size() - 1));
  }
  return s;
}

} // namespace facebook::nimble::mlidc
