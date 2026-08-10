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

// Benchmarks for RLE encoding decode (materialize) performance.
//
// Tests materialize throughput with different run length distributions:
// - Constant: single value repeated (1 run covering all rows)
// - Long runs: average run length ~1000
// - Medium runs: average run length ~100
// - Short runs: average run length ~10
// - Tiny runs: average run length ~2 (worst case for RLE)
//
// Usage:
//   buck2 run @mode/opt //velox/dwio/nimble/encodings/tests:rle_benchmark

#include "common/init/light.h"
#include "folly/Benchmark.h"
#include "folly/Random.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

using namespace facebook::nimble;

namespace {

constexpr uint32_t kRowCount = 1'000'000;

std::shared_ptr<facebook::velox::memory::MemoryPool> pool;

// Generate data with a given average run length.
template <typename T>
std::vector<T> generateRleData(uint32_t rowCount, uint32_t avgRunLength) {
  std::vector<T> data(rowCount);
  T value = 0;
  uint32_t i = 0;
  while (i < rowCount) {
    // Randomize run length around the average.
    auto runLen = std::max<uint32_t>(
        1,
        avgRunLength / 2 +
            folly::Random::rand32() % std::max<uint32_t>(1, avgRunLength));
    runLen = std::min(runLen, rowCount - i);
    std::fill(data.begin() + i, data.begin() + i + runLen, value);
    i += runLen;
    ++value;
  }
  return data;
}

// Encode data using RLE and return the encoded buffer.
template <typename T>
std::pair<std::string_view, std::unique_ptr<Buffer>> encodeRle(
    const std::vector<T>& data) {
  auto buffer = std::make_unique<Buffer>(*pool);
  ManualEncodingSelectionPolicyFactory factory{
      {{EncodingType::RLE, 1.0}},
      /*compressionOptions=*/std::nullopt};
  auto policy = std::unique_ptr<EncodingSelectionPolicy<T>>(
      static_cast<EncodingSelectionPolicy<T>*>(
          factory.createPolicy(TypeTraits<T>::dataType).release()));
  auto encoded = EncodingFactory::encode<T>(
      std::move(policy), std::span<const T>(data), *buffer);
  return {encoded, std::move(buffer)};
}

// Pre-encoded data for each run length distribution.
struct EncodedData {
  std::string_view encoded;
  std::unique_ptr<Buffer> buffer;
};

template <typename T>
struct BenchmarkData {
  EncodedData constant;
  EncodedData longRuns; // ~1000
  EncodedData mediumRuns; // ~100
  EncodedData shortRuns; // ~10
  EncodedData tinyRuns; // ~2

  void init() {
    {
      std::vector<T> data(kRowCount, static_cast<T>(42));
      auto [enc, buf] = encodeRle(data);
      constant = {enc, std::move(buf)};
    }
    {
      auto data = generateRleData<T>(kRowCount, 1000);
      auto [enc, buf] = encodeRle(data);
      longRuns = {enc, std::move(buf)};
    }
    {
      auto data = generateRleData<T>(kRowCount, 100);
      auto [enc, buf] = encodeRle(data);
      mediumRuns = {enc, std::move(buf)};
    }
    {
      auto data = generateRleData<T>(kRowCount, 10);
      auto [enc, buf] = encodeRle(data);
      shortRuns = {enc, std::move(buf)};
    }
    {
      auto data = generateRleData<T>(kRowCount, 2);
      auto [enc, buf] = encodeRle(data);
      tinyRuns = {enc, std::move(buf)};
    }
  }
};

std::unique_ptr<BenchmarkData<int32_t>> int32Data;
std::unique_ptr<BenchmarkData<int64_t>> int64Data;

// Benchmark materialize: decode all rows in one call.
template <typename T>
void benchMaterialize(unsigned iters, const EncodedData& data) {
  std::vector<T> output(kRowCount);
  for (unsigned i = 0; i < iters; ++i) {
    auto encoding = EncodingFactory().create(*pool, data.encoded, {});
    encoding->materialize(kRowCount, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}

// Benchmark materialize in batches (simulates batched read path).
template <typename T>
void benchMaterializeBatched(
    unsigned iters,
    const EncodedData& data,
    uint32_t batchSize) {
  std::vector<T> output(batchSize);
  for (unsigned i = 0; i < iters; ++i) {
    auto encoding = EncodingFactory().create(*pool, data.encoded, {});
    uint32_t remaining = kRowCount;
    while (remaining > 0) {
      auto count = std::min(remaining, batchSize);
      encoding->materialize(count, output.data());
      remaining -= count;
    }
    folly::doNotOptimizeAway(output[0]);
  }
}

// Benchmark skip + materialize (simulates selective read).
template <typename T>
void benchSkipAndMaterialize(
    unsigned iters,
    const EncodedData& data,
    uint32_t skipCount,
    uint32_t readCount) {
  std::vector<T> output(readCount);
  for (unsigned i = 0; i < iters; ++i) {
    auto encoding = EncodingFactory().create(*pool, data.encoded, {});
    uint32_t pos = 0;
    while (pos < kRowCount) {
      auto toSkip = std::min(skipCount, kRowCount - pos);
      encoding->skip(toSkip);
      pos += toSkip;
      if (pos >= kRowCount) {
        break;
      }
      auto toRead = std::min(readCount, kRowCount - pos);
      encoding->materialize(toRead, output.data());
      pos += toRead;
    }
    folly::doNotOptimizeAway(output[0]);
  }
}

} // namespace

// --- int32_t materialize (full) ---
BENCHMARK(Int32_Materialize_Constant, n) {
  benchMaterialize<int32_t>(n, int32Data->constant);
}
BENCHMARK_RELATIVE(Int32_Materialize_LongRuns, n) {
  benchMaterialize<int32_t>(n, int32Data->longRuns);
}
BENCHMARK_RELATIVE(Int32_Materialize_MediumRuns, n) {
  benchMaterialize<int32_t>(n, int32Data->mediumRuns);
}
BENCHMARK_RELATIVE(Int32_Materialize_ShortRuns, n) {
  benchMaterialize<int32_t>(n, int32Data->shortRuns);
}
BENCHMARK_RELATIVE(Int32_Materialize_TinyRuns, n) {
  benchMaterialize<int32_t>(n, int32Data->tinyRuns);
}

BENCHMARK_DRAW_LINE();

// --- int64_t materialize (full) ---
BENCHMARK(Int64_Materialize_Constant, n) {
  benchMaterialize<int64_t>(n, int64Data->constant);
}
BENCHMARK_RELATIVE(Int64_Materialize_LongRuns, n) {
  benchMaterialize<int64_t>(n, int64Data->longRuns);
}
BENCHMARK_RELATIVE(Int64_Materialize_MediumRuns, n) {
  benchMaterialize<int64_t>(n, int64Data->mediumRuns);
}
BENCHMARK_RELATIVE(Int64_Materialize_ShortRuns, n) {
  benchMaterialize<int64_t>(n, int64Data->shortRuns);
}
BENCHMARK_RELATIVE(Int64_Materialize_TinyRuns, n) {
  benchMaterialize<int64_t>(n, int64Data->tinyRuns);
}

BENCHMARK_DRAW_LINE();

// --- int64_t materialize batched (batch=1024) ---
BENCHMARK(Int64_Batched1024_Constant, n) {
  benchMaterializeBatched<int64_t>(n, int64Data->constant, 1024);
}
BENCHMARK_RELATIVE(Int64_Batched1024_LongRuns, n) {
  benchMaterializeBatched<int64_t>(n, int64Data->longRuns, 1024);
}
BENCHMARK_RELATIVE(Int64_Batched1024_MediumRuns, n) {
  benchMaterializeBatched<int64_t>(n, int64Data->mediumRuns, 1024);
}
BENCHMARK_RELATIVE(Int64_Batched1024_ShortRuns, n) {
  benchMaterializeBatched<int64_t>(n, int64Data->shortRuns, 1024);
}
BENCHMARK_RELATIVE(Int64_Batched1024_TinyRuns, n) {
  benchMaterializeBatched<int64_t>(n, int64Data->tinyRuns, 1024);
}

BENCHMARK_DRAW_LINE();

// --- int64_t skip + materialize (skip 512, read 512) ---
BENCHMARK(Int64_SkipRead_Constant, n) {
  benchSkipAndMaterialize<int64_t>(n, int64Data->constant, 512, 512);
}
BENCHMARK_RELATIVE(Int64_SkipRead_LongRuns, n) {
  benchSkipAndMaterialize<int64_t>(n, int64Data->longRuns, 512, 512);
}
BENCHMARK_RELATIVE(Int64_SkipRead_MediumRuns, n) {
  benchSkipAndMaterialize<int64_t>(n, int64Data->mediumRuns, 512, 512);
}
BENCHMARK_RELATIVE(Int64_SkipRead_ShortRuns, n) {
  benchSkipAndMaterialize<int64_t>(n, int64Data->shortRuns, 512, 512);
}
BENCHMARK_RELATIVE(Int64_SkipRead_TinyRuns, n) {
  benchSkipAndMaterialize<int64_t>(n, int64Data->tinyRuns, 512, 512);
}

int main(int argc, char** argv) {
  facebook::init::initFacebookLight(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize({});
  pool = facebook::velox::memory::memoryManager()->addLeafPool("rle_benchmark");
  int32Data = std::make_unique<BenchmarkData<int32_t>>();
  int32Data->init();
  int64Data = std::make_unique<BenchmarkData<int64_t>>();
  int64Data->init();
  folly::runBenchmarks();
  // Destroy benchmark data before the memory pool is destroyed at exit.
  int32Data.reset();
  int64Data.reset();
  pool.reset();
  return 0;
}
