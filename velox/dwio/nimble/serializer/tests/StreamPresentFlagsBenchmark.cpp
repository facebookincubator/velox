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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "common/init/light.h"
#include "folly/Benchmark.h"

namespace {

struct ClearParams {
  uint32_t numStreamFlags;
  uint32_t presentPercent;
};

struct ClearState {
  std::vector<bool> streamPresentFlags;
  std::vector<uint32_t> presentOffsets;
};

ClearState makeClearState(ClearParams params) {
  const auto numPresentStreams =
      params.numStreamFlags * params.presentPercent / 100;
  ClearState state{
      .streamPresentFlags = std::vector<bool>(params.numStreamFlags, false),
      .presentOffsets = {},
  };
  state.presentOffsets.reserve(numPresentStreams);
  for (uint32_t i = 0; i < numPresentStreams; ++i) {
    state.presentOffsets.emplace_back(
        i * params.numStreamFlags / numPresentStreams);
  }
  return state;
}

void restorePresentFlags(ClearState& state) {
  for (const auto offset : state.presentOffsets) {
    state.streamPresentFlags[offset] = true;
  }
}

void selectiveClear(uint32_t iterations, ClearParams params) {
  auto state = makeClearState(params);
  const auto presentOffsets = state.presentOffsets;
  folly::BenchmarkSuspender suspender;
  for (uint32_t i = 0; i < iterations; ++i) {
    state.presentOffsets = presentOffsets;
    restorePresentFlags(state);
    suspender.dismiss();
    for (const auto offset : state.presentOffsets) {
      state.streamPresentFlags[offset] = false;
    }
    state.presentOffsets.clear();
    folly::doNotOptimizeAway(state);
    suspender.rehire();
  }
}

void denseClear(uint32_t iterations, ClearParams params) {
  auto state = makeClearState(params);
  const auto presentOffsets = state.presentOffsets;
  folly::BenchmarkSuspender suspender;
  for (uint32_t i = 0; i < iterations; ++i) {
    state.presentOffsets = presentOffsets;
    restorePresentFlags(state);
    suspender.dismiss();
    std::fill(
        state.streamPresentFlags.begin(),
        state.streamPresentFlags.end(),
        false);
    state.presentOffsets.clear();
    folly::doNotOptimizeAway(state);
    suspender.rehire();
  }
}

BENCHMARK_NAMED_PARAM(
    selectiveClear,
    Flags100_Present10Pct,
    ClearParams{100, 10})
BENCHMARK_RELATIVE_NAMED_PARAM(
    denseClear,
    Flags100_Present10Pct,
    ClearParams{100, 10})
BENCHMARK_NAMED_PARAM(
    selectiveClear,
    Flags100_Present20Pct,
    ClearParams{100, 20})
BENCHMARK_RELATIVE_NAMED_PARAM(
    denseClear,
    Flags100_Present20Pct,
    ClearParams{100, 20})
BENCHMARK_NAMED_PARAM(
    selectiveClear,
    Flags100_Present40Pct,
    ClearParams{100, 40})
BENCHMARK_RELATIVE_NAMED_PARAM(
    denseClear,
    Flags100_Present40Pct,
    ClearParams{100, 40})
BENCHMARK_NAMED_PARAM(
    selectiveClear,
    Flags100_Present80Pct,
    ClearParams{100, 80})
BENCHMARK_RELATIVE_NAMED_PARAM(
    denseClear,
    Flags100_Present80Pct,
    ClearParams{100, 80})

BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    selectiveClear,
    Flags1K_Present10Pct,
    ClearParams{1'000, 10})
BENCHMARK_RELATIVE_NAMED_PARAM(
    denseClear,
    Flags1K_Present10Pct,
    ClearParams{1'000, 10})
BENCHMARK_NAMED_PARAM(
    selectiveClear,
    Flags1K_Present20Pct,
    ClearParams{1'000, 20})
BENCHMARK_RELATIVE_NAMED_PARAM(
    denseClear,
    Flags1K_Present20Pct,
    ClearParams{1'000, 20})
BENCHMARK_NAMED_PARAM(
    selectiveClear,
    Flags1K_Present40Pct,
    ClearParams{1'000, 40})
BENCHMARK_RELATIVE_NAMED_PARAM(
    denseClear,
    Flags1K_Present40Pct,
    ClearParams{1'000, 40})
BENCHMARK_NAMED_PARAM(
    selectiveClear,
    Flags1K_Present80Pct,
    ClearParams{1'000, 80})
BENCHMARK_RELATIVE_NAMED_PARAM(
    denseClear,
    Flags1K_Present80Pct,
    ClearParams{1'000, 80})

BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    selectiveClear,
    Flags10K_Present10Pct,
    ClearParams{10'000, 10})
BENCHMARK_RELATIVE_NAMED_PARAM(
    denseClear,
    Flags10K_Present10Pct,
    ClearParams{10'000, 10})
BENCHMARK_NAMED_PARAM(
    selectiveClear,
    Flags10K_Present20Pct,
    ClearParams{10'000, 20})
BENCHMARK_RELATIVE_NAMED_PARAM(
    denseClear,
    Flags10K_Present20Pct,
    ClearParams{10'000, 20})
BENCHMARK_NAMED_PARAM(
    selectiveClear,
    Flags10K_Present40Pct,
    ClearParams{10'000, 40})
BENCHMARK_RELATIVE_NAMED_PARAM(
    denseClear,
    Flags10K_Present40Pct,
    ClearParams{10'000, 40})
BENCHMARK_NAMED_PARAM(
    selectiveClear,
    Flags10K_Present80Pct,
    ClearParams{10'000, 80})
BENCHMARK_RELATIVE_NAMED_PARAM(
    denseClear,
    Flags10K_Present80Pct,
    ClearParams{10'000, 80})

} // namespace

int main(int argc, char** argv) {
  facebook::init::InitFacebookLight init{&argc, &argv};
  folly::runBenchmarks();
  return 0;
}
