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

/// Fuzzes the EncodingView that createEncodingView returns for SubIntSplit
/// streams, with the delta pre-transform both off and on, and for framed,
/// reordered and transformed streams, against the values that were encoded.
///
/// Configuration via CLI flags:
///   --sis_view_fuzzer_iterations=N  Iterations per type (default: 20)
///   --sis_view_fuzzer_max_rows=N    Maximum rows per iteration (default: 2000)
///   --sis_view_fuzzer_seed=N        Fixed seed, 0=random (default: 42)

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <random>
#include <type_traits>
#include <vector>

#include "folly/Random.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/subintsplit/Format.h"
#include "velox/dwio/nimble/encodings/subintsplit/RowFrame.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionTransform.h"
#include "velox/dwio/nimble/encodings/subintsplit/TuningConfig.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/fuzzer/encoding/EncodingFuzzer.h"
#include "velox/dwio/nimble/fuzzer/encoding/SubIntSplitFuzzer.h"
#include "velox/dwio/nimble/velox/RowRange.h"

DEFINE_uint32(
    sis_view_fuzzer_iterations,
    20,
    "Number of SubIntSplit EncodingView fuzzer iterations per type");
DEFINE_uint32(
    sis_view_fuzzer_max_rows,
    2000,
    "Maximum rows per SubIntSplit EncodingView fuzzer iteration");
DEFINE_uint32(
    sis_view_fuzzer_seed,
    42,
    "SubIntSplit EncodingView fuzzer seed (0 = random)");

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::test;

namespace {

template <typename T>
void expectEqual(const T& expected, const T& actual, uint32_t row) {
  using physicalType = typename TypeTraits<T>::physicalType;
  ASSERT_EQ(
      std::bit_cast<physicalType>(actual),
      std::bit_cast<physicalType>(expected))
      << "Mismatch at row " << row;
}

// Datasets that the delta pre-transform keeps (sorted and monotonic ones, whose
// deltas are small) next to ones it rejects, so each run holds both kinds of
// stream.
template <typename T>
std::vector<Vector<T>> makeDatasets(
    velox::memory::MemoryPool& pool,
    std::mt19937& rng,
    uint32_t rowCount,
    Buffer* buffer) {
  std::vector<Vector<T>> datasets;
  Vector<T> random(&pool);
  random.reserve(rowCount);
  nimble::testing::addRandomData<T>(rng, rowCount, &random, buffer);
  datasets.push_back(std::move(random));
  datasets.push_back(makeSingleValueData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(makeMonotonicData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(makeSortedData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(makeDeltaBlockSortedData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(makeBitStructuredData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(makeSnowflakeData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(makeMixedRegimeData<T>(pool, rng, rowCount, buffer));
  datasets.push_back(
      makeAdversarialBitPatternData<T>(pool, rng, rowCount, buffer));
  for (uint32_t size : {2u, 3u}) {
    datasets.push_back(makeMonotonicData<T>(pool, rng, size, buffer));
  }
  return datasets;
}

std::vector<uint32_t> makeProbeRows(std::mt19937& rng, uint32_t rowCount) {
  const auto probeCount = std::min<uint32_t>(rowCount, 256);
  std::vector<uint32_t> rows;
  rows.reserve(probeCount + 8);
  for (uint32_t i = 0; i < probeCount; ++i) {
    rows.push_back(folly::Random::rand32(rng) % rowCount);
  }
  for (const uint32_t row : {0u, 1u, 255u, 256u, 4095u, 4096u}) {
    if (row < rowCount) {
      rows.push_back(row);
    }
  }
  rows.push_back(rowCount - 1);
  return rows;
}

// Ordered, disjoint row ranges with random gaps and lengths.
std::vector<RowRange> makeRowRanges(std::mt19937& rng, uint32_t rowCount) {
  std::vector<RowRange> ranges;
  uint32_t row = folly::Random::rand32(rng) % std::min<uint32_t>(rowCount, 64);
  while (row < rowCount && ranges.size() < 64) {
    const uint32_t length = 1 + folly::Random::rand32(rng) % 96;
    const uint32_t end = std::min(rowCount, row + length);
    ranges.emplace_back(row, end);
    row = end + 1 + folly::Random::rand32(rng) % 128;
  }
  return ranges;
}

template <typename T>
void verifyView(
    std::mt19937& rng,
    velox::memory::MemoryPool& pool,
    const EncodingView& view,
    const Vector<T>& data) {
  const auto rowCount = static_cast<uint32_t>(data.size());
  ASSERT_EQ(view.rowCount(), rowCount);

  const auto probeRows = makeProbeRows(rng, rowCount);
  for (const auto row : probeRows) {
    T actual;
    view.readAt(row, &actual);
    expectEqual(data[row], actual, row);
  }

  Vector<T> gathered(&pool, probeRows.size());
  view.readAt(probeRows, gathered.data());
  for (size_t i = 0; i < probeRows.size(); ++i) {
    expectEqual(data[probeRows[i]], gathered[i], probeRows[i]);
  }

  const uint32_t offset = folly::Random::rand32(rng) % rowCount;
  const uint32_t length = 1 + folly::Random::rand32(rng) % (rowCount - offset);
  Vector<T> contiguous(&pool, length);
  view.read(offset, length, contiguous.data());
  for (uint32_t i = 0; i < length; ++i) {
    expectEqual(data[offset + i], contiguous[i], offset + i);
  }

  const auto ranges = makeRowRanges(rng, rowCount);
  uint32_t rangeRows{0};
  for (const auto& range : ranges) {
    rangeRows += range.numRows();
  }
  Vector<T> ranged(&pool, rangeRows);
  const auto numRead =
      view.read(ranges, [](uint32_t /*outputIndex*/) {}, ranged.data());
  ASSERT_EQ(numRead, rangeRows);
  uint32_t outputIndex{0};
  for (const auto& range : ranges) {
    for (uint32_t row = range.startRow; row < range.endRow; ++row) {
      expectEqual(data[row], ranged[outputIndex++], row);
    }
  }
}

template <typename EncodingClass>
void runSubIntSplitViewFuzzer(
    uint32_t iterations,
    uint32_t maxRows,
    uint32_t seed) {
  using T = typename EncodingClass::cppDataType;
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  auto dataBuffer = std::make_unique<Buffer>(*pool);

  if (seed == 0) {
    seed = folly::Random::rand32();
  }
  LOG(INFO) << "SubIntSplit EncodingView fuzzer seed: " << seed
            << " dtype: " << toString(TypeTraits<T>::dataType)
            << " iterations: " << iterations << " maxRows: " << maxRows;
  std::mt19937 rng(seed);

  uint32_t numDeltaStreams{0};
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    const auto rowCount = 2 + folly::Random::rand32(rng) % maxRows;
    const bool realNestedSelection = folly::Random::oneIn(2, rng);
    auto datasets = makeDatasets<T>(*pool, rng, rowCount, dataBuffer.get());

    for (const auto& data : datasets) {
      if (data.empty()) {
        continue;
      }
      for (const bool deltaPreTransform : {false, true}) {
        const bool useVarint = folly::Random::oneIn(2, rng);
        SCOPED_TRACE(
            ::testing::Message()
            << "seed=" << seed << " iter=" << iter << " deltaPreTransform="
            << deltaPreTransform << " useVarint=" << useVarint
            << " realNestedSelection=" << realNestedSelection
            << " rowCount=" << data.size());
        Encoding::Options options{.useVarintRowCount = useVarint};
        options.subIntSplitDeltaPreTransform = deltaPreTransform;
        Buffer encodeBuffer(*pool);
        std::string_view serialized;
        try {
          serialized = Encoder<EncodingClass>::encode(
              encodeBuffer,
              data,
              CompressionType::Uncompressed,
              options,
              realNestedSelection);
        } catch (const NimbleUserError& e) {
          if (e.errorCode() == error_code::IncompatibleEncoding) {
            continue;
          }
          throw;
        }
        if (subintsplit::isDeltaStream(
                serialized,
                EncodingPrefix::prefixSize(
                    serialized, options.useVarintRowCount))) {
          ASSERT_TRUE(deltaPreTransform);
          ++numDeltaStreams;
        }

        auto view = createEncodingView(serialized, pool.get(), options);
        ASSERT_NE(view, nullptr);
        verifyView<T>(rng, *pool, *view, data);
      }
    }
  }
  // Monotonic and sorted datasets encode smaller as deltas, so a run that
  // produced no delta stream did not test what this suite is for.
  EXPECT_GT(numDeltaStreams, 0);
  LOG(INFO) << numDeltaStreams << " delta streams";
}

// base + slope * row plus up to 11 random low bits, long enough for a row
// frame (kRowFrameMinStrides strides of kRowFrameStride rows). Unsigned
// arithmetic, so a steep slope wraps.
template <typename T>
Vector<T> makeFramedLineData(
    velox::memory::MemoryPool& pool,
    std::mt19937& rng,
    uint32_t extraRows) {
  using UnsignedT = std::make_unsigned_t<T>;
  const uint32_t rowCount =
      subintsplit::kRowFrameMinStrides * subintsplit::kRowFrameStride + 1 +
      extraRows;
  const auto base = static_cast<UnsignedT>(folly::Random::rand64(rng));
  const auto slope =
      static_cast<UnsignedT>(1 + folly::Random::rand32(rng) % 1'000);
  const uint32_t noiseBits = folly::Random::rand32(rng) % 12;
  Vector<T> data(&pool);
  data.reserve(rowCount);
  for (uint32_t row = 0; row < rowCount; ++row) {
    auto value = static_cast<UnsignedT>(base + slope * row);
    if (noiseBits > 0) {
      value += static_cast<UnsignedT>(
          folly::Random::rand32(rng) & ((1u << noiseBits) - 1));
    }
    data.push_back(static_cast<T>(value));
  }
  return data;
}

// Framed, reordered and transformed streams under random settings, which
// the positional view reads directly, next to delta streams, which it
// decodes once.
template <typename EncodingClass>
void runSubIntSplitTransformedViewFuzzer(
    uint32_t iterations,
    uint32_t maxRows,
    uint32_t seed) {
  using T = typename EncodingClass::cppDataType;
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  auto dataBuffer = std::make_unique<Buffer>(*pool);
  if (seed == 0) {
    seed = folly::Random::rand32();
  }
  LOG(INFO) << "SubIntSplit transformed EncodingView fuzzer seed: " << seed
            << " dtype: " << toString(TypeTraits<T>::dataType)
            << " iterations: " << iterations << " maxRows: " << maxRows;
  std::mt19937 rng(seed);

  uint32_t numFramed{0};
  uint32_t numReordered{0};
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    const auto rowCount = 2 + folly::Random::rand32(rng) % maxRows;
    auto datasets = makeDatasets<T>(*pool, rng, rowCount, dataBuffer.get());
    // The other datasets are too short for a row frame, which only integer
    // streams carry.
    if constexpr (std::is_integral_v<T>) {
      datasets.push_back(
          makeFramedLineData<T>(
              *pool, rng, folly::Random::rand32(rng) % maxRows));
    }
    for (const auto& data : datasets) {
      if (data.empty()) {
        continue;
      }
      subintsplit::TuningConfig tuning;
      tuning.rowFrameForceApply = folly::Random::oneIn(2, rng);
      switch (folly::Random::rand32(3, rng)) {
        case 0:
          break;
        case 1:
          tuning.autoTransform = true;
          break;
        default:
          tuning.transform =
              static_cast<uint8_t>(subintsplit::TransformId::KeyDerived);
          tuning.forceApply = true;
          break;
      }
      Encoding::Options options{
          .useVarintRowCount = folly::Random::oneIn(2, rng)};
      options.subIntSplitDeltaPreTransform = folly::Random::oneIn(4, rng);
      SCOPED_TRACE(
          ::testing::Message()
          << "seed=" << seed << " iter=" << iter << " rowCount=" << data.size()
          << " rowFrameForceApply=" << tuning.rowFrameForceApply
          << " autoTransform=" << tuning.autoTransform
          << " forceApply=" << tuning.forceApply
          << " delta=" << options.subIntSplitDeltaPreTransform);
      Buffer encodeBuffer(*pool);
      const auto serialized =
          encodeSubIntSplit(data, encodeBuffer, options, tuning);
      const auto flags = subintsplit::streamFlags(
          serialized,
          EncodingPrefix::prefixSize(serialized, options.useVarintRowCount));
      numFramed += (flags & subintsplit::kFlagRowFrame) != 0;
      numReordered += EncodingPrefix::encodingType(serialized) ==
              EncodingType::SubIntSplitReordered ||
          (flags & subintsplit::kFlagTransforms) != 0;
      verifySubIntSplitView(rng, *pool, serialized, data, options);
      if (::testing::Test::HasFatalFailure()) {
        return;
      }
    }
  }
  LOG(INFO) << "SubIntSplit transformed EncodingView fuzzer framed streams: "
            << numFramed << "; reordered or transformed: " << numReordered;
  // A run without a framed or a reordered stream did not test the view's
  // positional paths for them.
  if constexpr (std::is_integral_v<T>) {
    EXPECT_GT(numFramed, 0);
  }
  EXPECT_GT(numReordered, 0);
}

} // namespace

using SubIntSplitViewTypes = ::testing::Types<
    SubIntSplitEncoding<int32_t>,
    SubIntSplitEncoding<uint32_t>,
    SubIntSplitEncoding<int64_t>,
    SubIntSplitEncoding<uint64_t>,
    SubIntSplitEncoding<float>,
    SubIntSplitEncoding<double>>;

template <typename E>
class SubIntSplitEncodingViewFuzzerTest : public ::testing::Test {};
TYPED_TEST_SUITE(SubIntSplitEncodingViewFuzzerTest, SubIntSplitViewTypes);

TYPED_TEST(SubIntSplitEncodingViewFuzzerTest, transformedViewReadsMatchValues) {
  runSubIntSplitTransformedViewFuzzer<TypeParam>(
      FLAGS_sis_view_fuzzer_iterations,
      FLAGS_sis_view_fuzzer_max_rows,
      FLAGS_sis_view_fuzzer_seed);
}

TYPED_TEST(SubIntSplitEncodingViewFuzzerTest, viewReadsMatchValues) {
  runSubIntSplitViewFuzzer<TypeParam>(
      FLAGS_sis_view_fuzzer_iterations,
      FLAGS_sis_view_fuzzer_max_rows,
      FLAGS_sis_view_fuzzer_seed);
}

// Defines main() through folly::Init, as NimbleWriterFuzzerTest does, so the
// flags above are parsed; gtest_main would leave them at their defaults.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
