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

/// Fuzzes SubIntSplit's row frame on columns that follow a line or a steady
/// step through their rows, with noise, outliers and wrap-around, under the
/// frame off, chosen by price (the default) and forced. Every stream must read
/// back exactly through the encoding and through its EncodingView.
///
/// Configuration via CLI flags:
///   --sis_row_frame_fuzzer_iterations=N  Iterations per type (default: 10)
///   --sis_row_frame_fuzzer_max_rows=N    Maximum rows per generic stream,
///                                        and rows added above the 16,385 a
///                                        frame needs for line and step
///                                        streams (default: 3000)
///   --sis_row_frame_fuzzer_seed=N        Fixed seed, 0=random (default: 42)

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "folly/Random.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/subintsplit/Format.h"
#include "velox/dwio/nimble/encodings/subintsplit/RowFrame.h"
#include "velox/dwio/nimble/encodings/subintsplit/TuningConfig.h"
#include "velox/dwio/nimble/fuzzer/encoding/EncodingFuzzer.h"
#include "velox/dwio/nimble/fuzzer/encoding/SubIntSplitFuzzer.h"

DEFINE_uint32(
    sis_row_frame_fuzzer_iterations,
    10,
    "Number of SubIntSplit row frame fuzzer iterations per type");
DEFINE_uint32(
    sis_row_frame_fuzzer_max_rows,
    3000,
    "Maximum rows per SubIntSplit row frame fuzzer stream");
DEFINE_uint32(
    sis_row_frame_fuzzer_seed,
    42,
    "SubIntSplit row frame fuzzer seed (0 = random)");

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::test;

namespace {

enum class FrameMode { kOff, kPriced, kForced };

// base + slope * row plus up to `noiseBits` random low bits, with rare
// outliers. Unsigned arithmetic, so a steep slope wraps.
template <typename T>
Vector<T> makeLineData(
    velox::memory::MemoryPool& pool,
    std::mt19937& rng,
    uint32_t rowCount) {
  using UnsignedT = std::make_unsigned_t<T>;
  const auto base = static_cast<UnsignedT>(folly::Random::rand64(rng));
  const auto slope = static_cast<UnsignedT>(
      folly::Random::oneIn(4, rng) ? folly::Random::rand64(rng)
                                   : 1 + folly::Random::rand32(rng) % 1'000);
  const uint32_t noiseBits = folly::Random::rand32(rng) % 12;
  const bool outliers = folly::Random::oneIn(3, rng);
  Vector<T> data(&pool);
  data.reserve(rowCount);
  for (uint32_t row = 0; row < rowCount; ++row) {
    auto value = static_cast<UnsignedT>(base + slope * row);
    if (noiseBits > 0) {
      value += static_cast<UnsignedT>(
          folly::Random::rand32(rng) & ((1u << noiseBits) - 1));
    }
    if (outliers && folly::Random::oneIn(200, rng)) {
      value = static_cast<UnsignedT>(folly::Random::rand64(rng));
    }
    data.push_back(static_cast<T>(value));
  }
  return data;
}

// A value that advances by `step` once every run of rows, like a timestamp
// whose runs carry a counter in the low bits (UUIDv7's high half).
template <typename T>
Vector<T> makeStepData(
    velox::memory::MemoryPool& pool,
    std::mt19937& rng,
    uint32_t rowCount) {
  using UnsignedT = std::make_unsigned_t<T>;
  const auto base = static_cast<UnsignedT>(folly::Random::rand64(rng));
  const uint32_t counterBits = folly::Random::rand32(rng) % 13;
  const auto step = static_cast<UnsignedT>(
      (1 + folly::Random::rand32(rng) % 3) << counterBits);
  const uint32_t maxRun = 1 + folly::Random::rand32(rng) % 64;
  Vector<T> data(&pool);
  data.reserve(rowCount);
  UnsignedT current = base;
  while (data.size() < rowCount) {
    const uint32_t run = 1 + folly::Random::rand32(rng) % maxRun;
    for (uint32_t i = 0; i < run && data.size() < rowCount; ++i) {
      const auto counter = counterBits == 0
          ? UnsignedT{0}
          : static_cast<UnsignedT>(i & ((1u << counterBits) - 1));
      data.push_back(static_cast<T>(static_cast<UnsignedT>(current + counter)));
    }
    current = static_cast<UnsignedT>(current + step);
  }
  return data;
}

template <typename T>
void runSubIntSplitRowFrameFuzzer(
    uint32_t iterations,
    uint32_t maxRows,
    uint32_t seed) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Buffer dataBuffer(*pool);
  if (seed == 0) {
    seed = folly::Random::rand32();
  }
  LOG(INFO) << "SubIntSplit row frame fuzzer seed: " << seed
            << " dtype: " << toString(TypeTraits<T>::dataType)
            << " iterations: " << iterations << " maxRows: " << maxRows;
  std::mt19937 rng(seed);

  uint32_t numFramed{0};
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    const uint32_t rowCount = 2 + folly::Random::rand32(rng) % maxRows;
    // Both fits sample growth over kRowFrameMinStrides strides of
    // kRowFrameStride rows and decline shorter streams, so the line and step
    // streams start at that length.
    const uint32_t frameRowCount =
        subintsplit::kRowFrameMinStrides * subintsplit::kRowFrameStride + 1 +
        folly::Random::rand32(rng) % maxRows;
    std::vector<Vector<T>> streams;
    streams.push_back(makeLineData<T>(*pool, rng, frameRowCount));
    streams.push_back(makeLineData<T>(*pool, rng, frameRowCount));
    streams.push_back(makeStepData<T>(*pool, rng, frameRowCount));
    streams.push_back(makeStepData<T>(*pool, rng, frameRowCount));
    streams.push_back(makeMonotonicData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(makeSnowflakeData<T>(*pool, rng, rowCount, &dataBuffer));
    streams.push_back(
        makeMixedRegimeData<T>(*pool, rng, rowCount, &dataBuffer));

    for (const auto& data : streams) {
      for (const auto mode :
           {FrameMode::kOff, FrameMode::kPriced, FrameMode::kForced}) {
        subintsplit::TuningConfig tuning;
        tuning.rowFrame = mode != FrameMode::kOff;
        tuning.rowFrameForceApply = mode == FrameMode::kForced;
        tuning.foldConstantSections = folly::Random::oneIn(2, rng);
        tuning.passThrough = folly::Random::oneIn(2, rng);
        const Encoding::Options options{
            .useVarintRowCount = folly::Random::oneIn(2, rng)};
        SCOPED_TRACE(
            ::testing::Message() << "seed=" << seed << " iter=" << iter
                                 << " rowCount=" << data.size()
                                 << " frameMode=" << static_cast<int>(mode));

        Buffer buffer(*pool);
        const auto encoded = encodeSubIntSplit(data, buffer, options, tuning);
        const bool framed = (subintsplit::streamFlags(
                                 encoded,
                                 EncodingPrefix::prefixSize(
                                     encoded, options.useVarintRowCount)) &
                             subintsplit::kFlagRowFrame) != 0;
        if (mode == FrameMode::kOff) {
          ASSERT_FALSE(framed);
        }
        numFramed += framed;

        verifySubIntSplitReads(rng, *pool, encoded, data, options, tuning);
        verifySubIntSplitView(rng, *pool, encoded, data, options);
        if (::testing::Test::HasFatalFailure()) {
          return;
        }
      }
    }
  }
  LOG(INFO) << "SubIntSplit row frame fuzzer framed streams: " << numFramed;
  // Line and step streams fit a frame, and the forced mode keeps it, so a run
  // with no framed stream did not test the frame.
  EXPECT_GT(numFramed, 0);
}

} // namespace

using SubIntSplitRowFrameTypes =
    ::testing::Types<int32_t, uint32_t, int64_t, uint64_t>;

template <typename T>
class SubIntSplitRowFrameFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(SubIntSplitRowFrameFuzzerTest, SubIntSplitRowFrameTypes);

TYPED_TEST(SubIntSplitRowFrameFuzzerTest, framedStreamsRoundTrip) {
  runSubIntSplitRowFrameFuzzer<TypeParam>(
      FLAGS_sis_row_frame_fuzzer_iterations,
      FLAGS_sis_row_frame_fuzzer_max_rows,
      FLAGS_sis_row_frame_fuzzer_seed);
}

// Defines main() through folly::Init, as NimbleWriterFuzzerTest does, so the
// flags above are parsed; gtest_main would leave them at their defaults.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
