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
#include <numeric>
#include <random>
#include <vector>

namespace facebook::nimble::mlidc {

struct PointTraceParams {
  size_t streamLength{};
  size_t probes{1u << 16};
  uint64_t seed{42};
  bool ascending{false};
};

struct PointTrace {
  std::vector<size_t> indices;
  size_t distinctIndices{};
  double distinctFraction{};
};

// Uniform point-lookup trace: a shuffled permutation of [0, streamLength),
// truncated (and re-shuffled/appended as needed) to exactly `probes` entries.
inline PointTrace buildPointTrace(const PointTraceParams& p) {
  PointTrace t;
  if (p.streamLength == 0 || p.probes == 0) {
    return t;
  }
  const size_t n = p.streamLength;
  std::mt19937_64 rng(p.seed);
  t.indices.reserve(p.probes);

  std::vector<size_t> perm(n);
  std::iota(perm.begin(), perm.end(), size_t{0});
  std::shuffle(perm.begin(), perm.end(), rng);
  while (t.indices.size() < p.probes) {
    const size_t take = std::min(n, p.probes - t.indices.size());
    t.indices.insert(
        t.indices.end(), perm.begin(), perm.begin() + static_cast<std::ptrdiff_t>(take));
    if (t.indices.size() < p.probes) {
      std::shuffle(perm.begin(), perm.end(), rng);
    }
  }

  if (p.ascending) {
    std::sort(t.indices.begin(), t.indices.end());
  }

  std::vector<size_t> sorted(t.indices);
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  t.distinctIndices = sorted.size();
  t.distinctFraction = static_cast<double>(t.distinctIndices) /
      static_cast<double>(t.indices.size());
  return t;
}

} // namespace facebook::nimble::mlidc
