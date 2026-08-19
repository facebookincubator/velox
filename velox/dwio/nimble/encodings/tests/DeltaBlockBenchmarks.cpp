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

// Benchmarks for DeltaBlock materialize performance on sorted integer streams.
//
// Usage:
//   buck2 run @mode/opt
//   //velox/dwio/nimble/encodings/tests:delta_block_benchmark

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "common/init/light.h"
#include "folly/Benchmark.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

using namespace facebook::nimble;

namespace {

constexpr uint32_t kRowCount{1'000'000};
constexpr uint32_t kRowCountSmall{100'000};
constexpr uint16_t kDefaultBlockSize{256};

std::shared_ptr<facebook::velox::memory::MemoryPool> pool;

template <typename T>
std::vector<T> generateUnitStep(uint32_t rowCount) {
  std::vector<T> data(rowCount);
  for (uint32_t i = 0; i < rowCount; ++i) {
    data[i] = static_cast<T>(i);
  }
  return data;
}

template <typename T>
std::vector<T> generateSmallGaps(uint32_t rowCount) {
  std::vector<T> data(rowCount);
  uint64_t value{1000};
  for (uint32_t i = 0; i < rowCount; ++i) {
    value += 1 + (i % 7);
    data[i] = static_cast<T>(value);
  }
  return data;
}

template <typename T>
std::vector<T> generatePeriodicLargeGaps(uint32_t rowCount) {
  std::vector<T> data(rowCount);
  uint64_t value{0};
  for (uint32_t i = 0; i < rowCount; ++i) {
    value += (i % 1024 == 0) ? 65'536 : 1;
    data[i] = static_cast<T>(value);
  }
  return data;
}

template <typename T>
std::vector<T> generateConstant(uint32_t rowCount) {
  return std::vector<T>(rowCount, static_cast<T>(42));
}

struct EncodedData {
  std::string_view encoded;
  std::unique_ptr<Buffer> buffer;
  uint32_t rowCount{0};
};

template <typename T>
EncodedData encodeDeltaBlock(
    const std::vector<T>& data,
    uint16_t blockSize = kDefaultBlockSize) {
  auto buffer = std::make_unique<Buffer>(*pool);
  ManualEncodingSelectionPolicyFactory factory{
      {{EncodingType::DeltaBlock, 1.0}},
      /*compressionOptions=*/std::nullopt};
  auto policy = std::unique_ptr<EncodingSelectionPolicy<T>>(
      static_cast<EncodingSelectionPolicy<T>*>(
          factory.createPolicy(TypeTraits<T>::dataType).release()));

  Encoding::Options options;
  options.deltaBlockSize = blockSize;
  auto encoded = EncodingFactory::encode<T>(
      std::move(policy),
      std::span<const T>{data.data(), data.size()},
      *buffer,
      options);

  return {
      .encoded = encoded,
      .buffer = std::move(buffer),
      .rowCount = static_cast<uint32_t>(data.size())};
}

template <typename T>
struct BenchmarkData {
  EncodedData unitStep;
  EncodedData smallGaps;
  EncodedData periodicLargeGaps;
  EncodedData constant;

  void init(uint32_t rowCount) {
    unitStep = encodeDeltaBlock(generateUnitStep<T>(rowCount));
    smallGaps = encodeDeltaBlock(generateSmallGaps<T>(rowCount));
    periodicLargeGaps =
        encodeDeltaBlock(generatePeriodicLargeGaps<T>(rowCount));
    constant = encodeDeltaBlock(generateConstant<T>(rowCount));
  }
};

template <typename T>
struct BlockSizeBenchmarkData {
  EncodedData block64;
  EncodedData block256;
  EncodedData block1024;
  EncodedData block4096;

  void init(uint32_t rowCount) {
    const auto data = generateSmallGaps<T>(rowCount);
    block64 = encodeDeltaBlock(data, 64);
    block256 = encodeDeltaBlock(data, 256);
    block1024 = encodeDeltaBlock(data, 1024);
    block4096 = encodeDeltaBlock(data, 4096);
  }
};

template <typename T>
void benchMaterialize(unsigned iters, const EncodedData& data) {
  std::vector<T> output(data.rowCount);
  for (unsigned i = 0; i < iters; ++i) {
    auto encoding = EncodingFactory().create(*pool, data.encoded, {});
    encoding->materialize(data.rowCount, output.data());
    folly::doNotOptimizeAway(output.back());
  }
}

template <typename T>
void benchMaterializeBatched(
    unsigned iters,
    const EncodedData& data,
    uint32_t batchSize) {
  std::vector<T> output(batchSize);
  for (unsigned i = 0; i < iters; ++i) {
    auto encoding = EncodingFactory().create(*pool, data.encoded, {});
    uint32_t remaining = data.rowCount;
    while (remaining > 0) {
      const auto count = std::min<uint32_t>(remaining, batchSize);
      encoding->materialize(count, output.data());
      remaining -= count;
    }
    folly::doNotOptimizeAway(output[0]);
  }
}

template <typename T>
void benchSkipAndMaterialize(
    unsigned iters,
    const EncodedData& data,
    uint32_t skipCount,
    uint32_t readCount) {
  std::vector<T> output(readCount);
  for (unsigned i = 0; i < iters; ++i) {
    auto encoding = EncodingFactory().create(*pool, data.encoded, {});
    uint32_t position{0};
    while (position < data.rowCount) {
      const auto toSkip =
          std::min<uint32_t>(skipCount, data.rowCount - position);
      encoding->skip(toSkip);
      position += toSkip;
      if (position >= data.rowCount) {
        break;
      }

      const auto toRead =
          std::min<uint32_t>(readCount, data.rowCount - position);
      encoding->materialize(toRead, output.data());
      position += toRead;
    }
    folly::doNotOptimizeAway(output[0]);
  }
}

template <typename T>
void benchViewReadAtSequential(unsigned iters, const EncodedData& data) {
  T output{};
  for (unsigned i = 0; i < iters; ++i) {
    auto view = createEncodingView(data.encoded, pool.get());
    for (uint32_t row = 0; row < data.rowCount; ++row) {
      view->readAt(row, &output);
    }
    folly::doNotOptimizeAway(output);
  }
}

template <typename T>
void benchViewReadAtScattered(
    unsigned iters,
    const EncodedData& data,
    const std::vector<uint32_t>& positions) {
  T output{};
  for (unsigned i = 0; i < iters; ++i) {
    auto view = createEncodingView(data.encoded, pool.get());
    for (const auto position : positions) {
      view->readAt(position, &output);
    }
    folly::doNotOptimizeAway(output);
  }
}

std::vector<uint32_t> generateScatteredPositions(
    uint32_t rowCount,
    uint32_t count) {
  std::vector<uint32_t> positions;
  positions.reserve(count);
  uint64_t value{17};
  for (uint32_t i = 0; i < count; ++i) {
    value = (value * 48'271 + 1) % rowCount;
    positions.push_back(static_cast<uint32_t>(value));
  }
  return positions;
}

std::unique_ptr<BenchmarkData<int32_t>> int32Data;
std::unique_ptr<BenchmarkData<int64_t>> int64Data;
std::unique_ptr<BenchmarkData<int64_t>> int64DataSmall;
std::unique_ptr<BlockSizeBenchmarkData<int64_t>> int64BlockSizeData;
std::vector<uint32_t> scatteredPositions;

} // namespace

BENCHMARK(Int32_Materialize_UnitStep, n) {
  benchMaterialize<int32_t>(n, int32Data->unitStep);
}
BENCHMARK_RELATIVE(Int32_Materialize_SmallGaps, n) {
  benchMaterialize<int32_t>(n, int32Data->smallGaps);
}
BENCHMARK_RELATIVE(Int32_Materialize_PeriodicLargeGaps, n) {
  benchMaterialize<int32_t>(n, int32Data->periodicLargeGaps);
}
BENCHMARK_RELATIVE(Int32_Materialize_Constant, n) {
  benchMaterialize<int32_t>(n, int32Data->constant);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(Int64_Materialize_UnitStep, n) {
  benchMaterialize<int64_t>(n, int64Data->unitStep);
}
BENCHMARK_RELATIVE(Int64_Materialize_SmallGaps, n) {
  benchMaterialize<int64_t>(n, int64Data->smallGaps);
}
BENCHMARK_RELATIVE(Int64_Materialize_PeriodicLargeGaps, n) {
  benchMaterialize<int64_t>(n, int64Data->periodicLargeGaps);
}
BENCHMARK_RELATIVE(Int64_Materialize_Constant, n) {
  benchMaterialize<int64_t>(n, int64Data->constant);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(Int64_Block64_SmallGaps, n) {
  benchMaterialize<int64_t>(n, int64BlockSizeData->block64);
}
BENCHMARK_RELATIVE(Int64_Block256_SmallGaps, n) {
  benchMaterialize<int64_t>(n, int64BlockSizeData->block256);
}
BENCHMARK_RELATIVE(Int64_Block1024_SmallGaps, n) {
  benchMaterialize<int64_t>(n, int64BlockSizeData->block1024);
}
BENCHMARK_RELATIVE(Int64_Block4096_SmallGaps, n) {
  benchMaterialize<int64_t>(n, int64BlockSizeData->block4096);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(Int64_Batched1024_SmallGaps, n) {
  benchMaterializeBatched<int64_t>(n, int64DataSmall->smallGaps, 1024);
}
BENCHMARK_RELATIVE(Int64_Batched1024_PeriodicLargeGaps, n) {
  benchMaterializeBatched<int64_t>(n, int64DataSmall->periodicLargeGaps, 1024);
}
BENCHMARK_RELATIVE(Int64_SkipRead512_SmallGaps, n) {
  benchSkipAndMaterialize<int64_t>(n, int64DataSmall->smallGaps, 512, 512);
}
BENCHMARK_RELATIVE(Int64_SkipRead512_PeriodicLargeGaps, n) {
  benchSkipAndMaterialize<int64_t>(
      n, int64DataSmall->periodicLargeGaps, 512, 512);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(Int64_ViewReadAtSequential_SmallGaps, n) {
  benchViewReadAtSequential<int64_t>(n, int64DataSmall->smallGaps);
}
BENCHMARK_RELATIVE(Int64_ViewReadAtScattered_SmallGaps, n) {
  benchViewReadAtScattered<int64_t>(
      n, int64DataSmall->smallGaps, scatteredPositions);
}
BENCHMARK_RELATIVE(Int64_ViewReadAtScattered_PeriodicLargeGaps, n) {
  benchViewReadAtScattered<int64_t>(
      n, int64DataSmall->periodicLargeGaps, scatteredPositions);
}

int main(int argc, char** argv) {
  facebook::init::initFacebookLight(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize({});
  pool = facebook::velox::memory::memoryManager()->addLeafPool(
      "delta_block_benchmark");

  int32Data = std::make_unique<BenchmarkData<int32_t>>();
  int32Data->init(kRowCount);

  int64Data = std::make_unique<BenchmarkData<int64_t>>();
  int64Data->init(kRowCount);

  int64DataSmall = std::make_unique<BenchmarkData<int64_t>>();
  int64DataSmall->init(kRowCountSmall);

  int64BlockSizeData = std::make_unique<BlockSizeBenchmarkData<int64_t>>();
  int64BlockSizeData->init(kRowCount);
  scatteredPositions =
      generateScatteredPositions(kRowCountSmall, /*count=*/kRowCountSmall);

  folly::runBenchmarks();

  scatteredPositions.clear();
  int64BlockSizeData.reset();
  int64DataSmall.reset();
  int64Data.reset();
  int32Data.reset();
  pool.reset();
  return 0;
}
