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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/CachePolicy.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/TimingStats.h"

namespace facebook::nimble::mlidc {

inline void clobber(const void* p) {
  asm volatile("" : : "r"(p) : "memory");
}

struct MeasureSpec {
  size_t iterations{5};
  size_t warmup{2};
};

// Caps iterations for an encoder where every read decompresses the whole
// payload, such as a block codec. The fine-grained range and gather sweeps run
// hundreds of cells, and a full decompress per cell would otherwise dominate
// wall-clock time. Warmup is dropped for the same reason, which makes these
// timings noisier than the rest.
inline MeasureSpec specFor(
    const MeasureSpec& base,
    bool wholePayloadCodec,
    size_t blockCodecIters) {
  if (!wholePayloadCodec) {
    return base;
  }
  MeasureSpec spec = base;
  spec.iterations =
      std::max<size_t>(1, std::min(base.iterations, blockCodecIters));
  spec.warmup = 0;
  return spec;
}

struct MeasureResult {
  TimingSummary time;
  TimingSummary evict;
  size_t iterationsRun{};
};

template <typename Fn>
MeasureResult measure(
    const MeasureSpec& spec,
    CacheController& controller,
    const EvictionTargets& targets,
    Fn&& fn) {
  using Clock = std::chrono::high_resolution_clock;

  for (size_t i = 0; i < spec.warmup; ++i) {
    fn();
    clobber(targets.sink.data());
  }

  std::vector<int64_t> timeSamples;
  std::vector<int64_t> evictSamples;
  timeSamples.reserve(spec.iterations);
  evictSamples.reserve(spec.iterations);

  for (size_t i = 0; i < spec.iterations; ++i) {
    controller.prepare(targets);

    const auto t0 = Clock::now();
    fn();
    const auto t1 = Clock::now();

    clobber(targets.sink.data());
    timeSamples.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    evictSamples.push_back(controller.lastEvictNs());
  }

  MeasureResult r;
  r.time = summarize(timeSamples);
  r.evict = summarize(evictSamples);
  r.iterationsRun = spec.iterations;
  return r;
}

inline TimingSummary measureClockOverhead(size_t iterations) {
  using Clock = std::chrono::high_resolution_clock;
  std::vector<int64_t> samples;
  samples.reserve(iterations);
  for (size_t i = 0; i < iterations; ++i) {
    const auto t0 = Clock::now();
    const auto t1 = Clock::now();
    samples.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
  }
  return summarize(samples);
}

} // namespace facebook::nimble::mlidc
