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
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace facebook::nimble::mlidc {

struct RowRange {
  size_t begin{0};
  size_t end{0};
  size_t size() const {
    return end > begin ? end - begin : 0;
  }
};
using RowRangeList = std::vector<RowRange>;

struct SelectiveTraceParams {
  double selectivity{0.5};
  double meanRunLength{8.0};
  uint64_t seed{42};
  size_t maxRanges{0};
};

inline RowRangeList makeSelectiveTrace(
    size_t n,
    const SelectiveTraceParams& p) {
  RowRangeList ranges;
  if (n == 0 || p.selectivity <= 0.0)
    return ranges;
  if (p.selectivity >= 1.0) {
    ranges.push_back({0, n});
    return ranges;
  }
  const double meanGapLength = p.meanRunLength * (1.0 / p.selectivity - 1.0);
  std::mt19937_64 rng(p.seed);
  std::geometric_distribution<size_t> runDist(
      1.0 / std::max(1.0, p.meanRunLength));
  std::geometric_distribution<size_t> gapDist(
      1.0 / std::max(1.0, meanGapLength));
  size_t pos = 0;
  while (pos < n) {
    size_t runLen = std::min(runDist(rng) + 1, n - pos);
    ranges.push_back({pos, pos + runLen});
    pos += runLen;
    if (pos >= n)
      break;
    if (p.maxRanges && ranges.size() >= p.maxRanges)
      break;
    size_t gapLen = std::min(gapDist(rng) + 1, n - pos);
    pos += gapLen;
  }
  return ranges;
}

} // namespace facebook::nimble::mlidc
