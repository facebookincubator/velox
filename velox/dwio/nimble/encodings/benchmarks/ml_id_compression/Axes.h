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
#include <vector>

namespace facebook::nimble::mlidc {

inline std::vector<size_t> logSpaced(size_t lo, size_t hi, size_t steps) {
  std::vector<size_t> out;
  if (steps == 0)
    return out;
  out.reserve(steps);
  if (steps == 1 || lo >= hi) {
    out.push_back(lo);
    return out;
  }
  const double lgLo = std::log2(static_cast<double>(lo));
  const double lgHi = std::log2(static_cast<double>(hi));
  for (size_t i = 0; i < steps; ++i) {
    const double f = static_cast<double>(i) / static_cast<double>(steps - 1);
    out.push_back(
        static_cast<size_t>(std::llround(std::exp2(lgLo + f * (lgHi - lgLo)))));
  }
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

inline std::vector<double> linSpaced(double lo, double hi, size_t steps) {
  std::vector<double> out;
  if (steps == 0)
    return out;
  out.reserve(steps);
  if (steps == 1) {
    out.push_back(hi);
    return out;
  }
  for (size_t i = 0; i < steps; ++i) {
    const double f = static_cast<double>(i) / static_cast<double>(steps - 1);
    out.push_back(lo + f * (hi - lo));
  }
  return out;
}

} // namespace facebook::nimble::mlidc
