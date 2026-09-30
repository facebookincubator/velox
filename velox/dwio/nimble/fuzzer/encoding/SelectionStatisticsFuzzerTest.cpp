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

/// Differential fuzzer for selection's statistics and its candidate skip.
///
/// Statistics counts distinct integers with a table, a bounded hash map or a
/// radix sort depending on range, length and cardinality. Each stream here is
/// shaped to reach one of those paths, and every statistic selection reads is
/// checked against a plain recount. Selection is then checked to pick exactly
/// what pricing every candidate would pick, so skipping a candidate on its
/// size lower bound never changes the result.
///
/// Configuration via CLI flags:
///   --selection_fuzzer_iterations=N  Iterations per type (default: 20)
///   --selection_fuzzer_seed=N        Fixed seed, 0=random (default: 42)

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <map>
#include <random>
#include <span>
#include <vector>

#include "folly/Random.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

DEFINE_uint32(
    selection_fuzzer_iterations,
    20,
    "Number of selection statistics fuzzer iterations per type");
DEFINE_uint32(
    selection_fuzzer_seed,
    42,
    "Selection statistics fuzzer seed (0 = random)");

using namespace facebook;
using namespace facebook::nimble;

namespace {

// Stream shapes aimed at each way Statistics counts distinct values.
enum class Shape {
  // Range no wider than the rows: dense table.
  kDenseRange,
  // Range under 64Ki values but wider than the rows: range table.
  kTableRange,
  // Wide values, at most 16Ki rows: sort.
  kShortWide,
  // Wide values, more rows but few distinct: bounded hash map.
  kFewWideDistinct,
  // Wide values, more rows and distinct values than the map holds: sort.
  kManyWideDistinct,
  // Long runs of repeated values.
  kRuns,
};

template <typename T>
std::vector<T> makeValues(std::mt19937& rng, Shape shape) {
  using UnsignedT = std::make_unsigned_t<T>;
  auto randomValue = [&]() {
    return static_cast<UnsignedT>(folly::Random::rand64(rng));
  };
  const auto base = randomValue();
  std::vector<T> values;
  auto push = [&](UnsignedT offset) {
    values.push_back(static_cast<T>(static_cast<UnsignedT>(base + offset)));
  };
  // Every narrow type is fully covered by the table paths, so it only ever
  // takes the shapes whose ranges it can hold.
  const uint64_t maxRange = std::numeric_limits<UnsignedT>::max();

  switch (shape) {
    case Shape::kDenseRange: {
      const uint32_t numRows = 1 + folly::Random::rand32(rng) % 5'000;
      const uint64_t range = std::min<uint64_t>(
          maxRange, 1 + folly::Random::rand32(rng) % numRows);
      for (uint32_t i = 0; i < numRows; ++i) {
        push(static_cast<UnsignedT>(folly::Random::rand64(rng) % range));
      }
      break;
    }
    case Shape::kTableRange: {
      const uint32_t numRows = 1'000 + folly::Random::rand32(rng) % 7'000;
      const uint64_t range = std::min<uint64_t>(
          maxRange, numRows + folly::Random::rand32(rng) % 60'000);
      for (uint32_t i = 0; i < numRows; ++i) {
        push(static_cast<UnsignedT>(folly::Random::rand64(rng) % range));
      }
      break;
    }
    case Shape::kShortWide: {
      const uint32_t numRows = 1 + folly::Random::rand32(rng) % 16'384;
      for (uint32_t i = 0; i < numRows; ++i) {
        values.push_back(static_cast<T>(randomValue()));
      }
      break;
    }
    case Shape::kFewWideDistinct: {
      const uint32_t numRows = 16'385 + folly::Random::rand32(rng) % 40'000;
      std::vector<UnsignedT> pool(1 + folly::Random::rand32(rng) % 16'000);
      for (auto& value : pool) {
        value = randomValue();
      }
      for (uint32_t i = 0; i < numRows; ++i) {
        values.push_back(
            static_cast<T>(pool[folly::Random::rand32(rng) % pool.size()]));
      }
      break;
    }
    case Shape::kManyWideDistinct: {
      const uint32_t numRows = 16'385 + folly::Random::rand32(rng) % 40'000;
      for (uint32_t i = 0; i < numRows; ++i) {
        values.push_back(static_cast<T>(randomValue()));
      }
      break;
    }
    case Shape::kRuns: {
      const uint32_t numRows = 1 + folly::Random::rand32(rng) % 30'000;
      while (values.size() < numRows) {
        const auto value = static_cast<T>(randomValue());
        const uint32_t runLength = 1 + folly::Random::rand32(rng) % 64;
        for (uint32_t i = 0; i < runLength && values.size() < numRows; ++i) {
          values.push_back(value);
        }
      }
      break;
    }
  }
  return values;
}

template <typename T>
void verifyStatistics(std::span<const T> values) {
  using UnsignedT = std::make_unsigned_t<T>;
  std::map<T, uint64_t> expectedCounts;
  for (const auto value : values) {
    ++expectedCounts[value];
  }

  {
    const auto statistics = Statistics<T>::create(values);
    EXPECT_EQ(statistics.min(), expectedCounts.begin()->first);
    EXPECT_EQ(statistics.max(), expectedCounts.rbegin()->first);
    EXPECT_EQ(statistics.isConstant(), expectedCounts.size() == 1);

    // Before the unique counts exist, so the bound takes its own pass.
    const auto lowerBound = statistics.distinctLowerBound();
    EXPECT_LE(lowerBound, expectedCounts.size());
    EXPECT_GE(lowerBound, 1);
    const auto rangeBits = std::bit_width(
        static_cast<uint64_t>(static_cast<UnsignedT>(
            static_cast<UnsignedT>(statistics.max()) -
            static_cast<UnsignedT>(statistics.min()))));
    if (rangeBits <= Statistics<T>::kDistinctBoundBits) {
      EXPECT_EQ(lowerBound, expectedCounts.size());
    }

    const auto& uniqueCounts = statistics.uniqueCounts();
    ASSERT_TRUE(uniqueCounts.has_value());
    ASSERT_EQ(uniqueCounts->size(), expectedCounts.size());
    std::map<T, uint64_t> actualCounts;
    for (const auto& [value, count] : uniqueCounts.value()) {
      EXPECT_TRUE(actualCounts.emplace(value, count).second)
          << "Value counted twice: " << value;
    }
    EXPECT_EQ(actualCounts, expectedCounts);
    for (const auto& [value, count] : expectedCounts) {
      EXPECT_EQ(uniqueCounts->at(value), count);
    }
    // Once the counts exist the bound is exact.
    EXPECT_EQ(statistics.distinctLowerBound(), expectedCounts.size());
  }

  const auto statistics = Statistics<T>::create(values);
  uint64_t numRuns{1};
  uint64_t runLength{1};
  uint64_t minRepeat = std::numeric_limits<uint64_t>::max();
  uint64_t maxRepeat{0};
  for (size_t i = 1; i <= values.size(); ++i) {
    if (i == values.size() || values[i] != values[i - 1]) {
      minRepeat = std::min(minRepeat, runLength);
      maxRepeat = std::max(maxRepeat, runLength);
      if (i < values.size()) {
        ++numRuns;
      }
      runLength = 1;
    } else {
      ++runLength;
    }
  }
  EXPECT_EQ(statistics.consecutiveRepeatCount(), numRuns);
  EXPECT_EQ(statistics.minRepeat(), minRepeat);
  EXPECT_EQ(statistics.maxRepeat(), maxRepeat);

  // Rows per 7-bit bucket of their offset from min.
  const size_t numBuckets = sizeof(T) * 8 / 7 + 1;
  std::vector<uint64_t> expectedBuckets(numBuckets, 0);
  const auto minValue = static_cast<UnsignedT>(statistics.min());
  for (const auto value : values) {
    const auto offset = static_cast<uint64_t>(
        static_cast<UnsignedT>(static_cast<UnsignedT>(value) - minValue));
    const size_t bucket = offset == 0
        ? 0
        : std::min<size_t>((std::bit_width(offset) - 1) / 7, numBuckets - 1);
    ++expectedBuckets[bucket];
  }
  EXPECT_EQ(statistics.bucketCounts(), expectedBuckets);

  if constexpr (std::is_unsigned_v<T>) {
    for (const uint16_t blockSize : {uint16_t{0}, uint16_t{1}, uint16_t{97}}) {
      const auto& blocks = statistics.minMaxBlocks(blockSize);
      const size_t step = blockSize == 0 ? values.size() : blockSize;
      ASSERT_EQ(blocks.size(), (values.size() + step - 1) / step);
      for (size_t block = 0; block < blocks.size(); ++block) {
        const auto first = values.begin() + block * step;
        const auto last = values.begin() +
            std::min(values.size(), static_cast<size_t>((block + 1) * step));
        const auto [minIt, maxIt] = std::minmax_element(first, last);
        EXPECT_EQ(blocks[block].count, static_cast<uint64_t>(last - first));
        EXPECT_EQ(blocks[block].min, static_cast<uint64_t>(*minIt));
        EXPECT_EQ(blocks[block].max, static_cast<uint64_t>(*maxIt));
      }
    }
  }
}

// Selection must pick what pricing every candidate would: the skip only drops
// a candidate whose lower bound already costs at least the cheapest so far.
template <typename T>
void verifySelectionUnchangedBySkip(std::span<const T> values) {
  using physicalType = typename TypeTraits<T>::physicalType;
  const std::span<const physicalType> physicalValues{
      reinterpret_cast<const physicalType*>(values.data()), values.size()};
  const Encoding::Options options;
  const auto readFactors =
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();

  float minCost = std::numeric_limits<float>::max();
  EncodingType expected = EncodingType::Trivial;
  {
    const auto statistics = Statistics<physicalType>::create(physicalValues);
    for (const auto& [encodingType, readFactor] : readFactors) {
      const auto estimate = detail::EncodingSizeEstimation<T>::estimateSize(
          encodingType, physicalValues, statistics, options);
      if (!estimate.has_value()) {
        continue;
      }
      const auto lowerBound =
          detail::EncodingSizeEstimation<T>::estimateSizeLowerBound(
              encodingType, physicalValues, statistics, options);
      if (lowerBound.has_value()) {
        EXPECT_LE(lowerBound.value(), estimate.value())
            << "Lower bound above the estimate for " << toString(encodingType);
      }
      const float cost = estimate.value() * readFactor;
      if (cost < minCost) {
        minCost = cost;
        expected = encodingType;
      }
    }
  }

  // A fresh Statistics, so the skip sees the counts not yet built.
  const auto statistics = Statistics<physicalType>::create(physicalValues);
  ManualEncodingSelectionPolicy<T> policy{
      readFactors, /*compressionOptions=*/std::nullopt, std::nullopt};
  const auto selected = policy.select(physicalValues, statistics, options);
  EXPECT_EQ(selected.encodingType, expected)
      << "selected " << toString(selected.encodingType) << ", expected "
      << toString(expected);
}

template <typename T>
void runSelectionStatisticsFuzzer(uint32_t iterations, uint32_t seed) {
  if (seed == 0) {
    seed = folly::Random::rand32();
  }
  LOG(INFO) << "Selection statistics fuzzer seed: " << seed
            << " dtype: " << toString(TypeTraits<T>::dataType)
            << " iterations: " << iterations;
  std::mt19937 rng(seed);
  constexpr Shape kShapes[] = {
      Shape::kDenseRange,
      Shape::kTableRange,
      Shape::kShortWide,
      Shape::kFewWideDistinct,
      Shape::kManyWideDistinct,
      Shape::kRuns,
  };
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    for (const auto shape : kShapes) {
      const auto values = makeValues<T>(rng, shape);
      SCOPED_TRACE(
          ::testing::Message()
          << "seed=" << seed << " iter=" << iter
          << " shape=" << static_cast<int>(shape) << " rows=" << values.size());
      verifyStatistics<T>(values);
      verifySelectionUnchangedBySkip<T>(values);
      if (::testing::Test::HasFatalFailure()) {
        return;
      }
    }
  }
}

} // namespace

template <typename T>
class SelectionStatisticsFuzzerTest : public ::testing::Test {};

using IntegerTypes = ::testing::Types<
    int8_t,
    uint8_t,
    int16_t,
    uint16_t,
    int32_t,
    uint32_t,
    int64_t,
    uint64_t>;
TYPED_TEST_SUITE(SelectionStatisticsFuzzerTest, IntegerTypes);

TYPED_TEST(SelectionStatisticsFuzzerTest, matchesRecountAndFullPricing) {
  runSelectionStatisticsFuzzer<TypeParam>(
      FLAGS_selection_fuzzer_iterations, FLAGS_selection_fuzzer_seed);
}

// Defines main() through folly::Init, as NimbleWriterFuzzerTest does, so the
// flags above are parsed; gtest_main would leave them at their defaults.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
