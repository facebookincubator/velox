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
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <limits>
#include <map>
#include <string_view>
#include <type_traits>
#include <utility>

#include <random>
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

namespace {
// Fixed so the shuffled order is reproducible across runs.
constexpr uint32_t kShuffleSeed = 20240816;
} // namespace

using namespace facebook;

#define INTEGRAL_TYPES(StatisticType)                                    \
  StatisticType<int8_t>, StatisticType<uint8_t>, StatisticType<int16_t>, \
      StatisticType<uint16_t>, StatisticType<int32_t>,                   \
      StatisticType<uint32_t>, StatisticType<int64_t>, StatisticType<uint64_t>

#define NUMERIC_TYPES(StatisticType) \
  INTEGRAL_TYPES(StatisticType), StatisticType<float>, StatisticType<double>

using IntegerTypes = ::testing::Types<INTEGRAL_TYPES(nimble::Statistics)>;
using NumericTypes = ::testing::Types<NUMERIC_TYPES(nimble::Statistics)>;
using BoolTypes = ::testing::Types<nimble::Statistics<bool>>;
using StringTypes =
    ::testing::Types<nimble::Statistics<std::string_view, std::string>>;

TYPED_TEST_CASE(StatisticsNumericTests, NumericTypes);
TYPED_TEST_CASE(StatisticsIntegerTests, IntegerTypes);
TYPED_TEST_CASE(StatisticsBoolTests, BoolTypes);
TYPED_TEST_CASE(StatisticsStringTests, StringTypes);

template <typename C>
class StatisticsNumericTests : public ::testing::Test {};

template <typename C>
class StatisticsIntegerTests : public ::testing::Test {};

template <typename C>
class StatisticsBoolTests : public ::testing::Test {};

template <typename C>
class StatisticsStringTests : public ::testing::Test {};

TYPED_TEST(StatisticsIntegerTests, isNonDecreasing) {
  using ValueType = typename TypeParam::valueType;

  const std::vector<ValueType> sorted{
      static_cast<ValueType>(1),
      static_cast<ValueType>(1),
      static_cast<ValueType>(3),
  };
  const std::vector<ValueType> unsorted{
      static_cast<ValueType>(1),
      static_cast<ValueType>(3),
      static_cast<ValueType>(2),
  };

  EXPECT_TRUE(TypeParam::create(sorted).template isNonDecreasing<ValueType>());
  EXPECT_FALSE(
      TypeParam::create(unsorted).template isNonDecreasing<ValueType>());
  EXPECT_TRUE(
      TypeParam::create(std::span<const ValueType>{})
          .template isNonDecreasing<ValueType>());
}

template <typename SignedType>
void testNaturalAndSignedOrderCachesAreIndependent() {
  using UnsignedType = std::make_unsigned_t<SignedType>;
  const std::vector<UnsignedType> signedSorted{
      static_cast<UnsignedType>(std::numeric_limits<SignedType>::min()),
      static_cast<UnsignedType>(-1),
      static_cast<UnsignedType>(0),
      static_cast<UnsignedType>(std::numeric_limits<SignedType>::max()),
  };
  const auto signedFirst =
      nimble::Statistics<UnsignedType>::create(signedSorted);
  EXPECT_TRUE(signedFirst.template isNonDecreasing<SignedType>());
  EXPECT_FALSE(signedFirst.template isNonDecreasing<UnsignedType>());

  const auto naturalFirst =
      nimble::Statistics<UnsignedType>::create(signedSorted);
  EXPECT_FALSE(naturalFirst.template isNonDecreasing<UnsignedType>());
  EXPECT_TRUE(naturalFirst.template isNonDecreasing<SignedType>());

  const std::vector<UnsignedType> signedUnsorted{
      static_cast<UnsignedType>(-1),
      static_cast<UnsignedType>(-2),
  };
  EXPECT_FALSE(
      nimble::Statistics<UnsignedType>::create(signedUnsorted)
          .template isNonDecreasing<SignedType>());
}

TEST(StatisticsTest, naturalAndSignedOrderCachesAreIndependent) {
  testNaturalAndSignedOrderCachesAreIndependent<int8_t>();
  testNaturalAndSignedOrderCachesAreIndependent<int16_t>();
  testNaturalAndSignedOrderCachesAreIndependent<int32_t>();
  testNaturalAndSignedOrderCachesAreIndependent<int64_t>();
}

TEST(StatisticsTest, runValues) {
  const std::vector<int32_t> data = {1, 1, 2, 2, 1, 3, 3};
  const std::vector<int32_t> expected = {1, 2, 1, 3};
  for (const bool populateRepeatMetricsFirst : {false, true}) {
    SCOPED_TRACE(populateRepeatMetricsFirst);
    const auto statistics = nimble::Statistics<int32_t>::create(data);
    if (populateRepeatMetricsFirst) {
      EXPECT_EQ(statistics.consecutiveRepeatCount(), 4);
      EXPECT_EQ(statistics.minRepeat(), 1);
      EXPECT_EQ(statistics.maxRepeat(), 2);
    }

    const auto& runValues = statistics.runValues();
    EXPECT_EQ(runValues, expected);
    EXPECT_EQ(&statistics.runValues(), &runValues);
    EXPECT_EQ(statistics.consecutiveRepeatCount(), 4);
    EXPECT_EQ(statistics.minRepeat(), 1);
    EXPECT_EQ(statistics.maxRepeat(), 2);
  }

  const std::vector<int32_t> empty;
  EXPECT_TRUE(nimble::Statistics<int32_t>::create(empty).runValues().empty());
}

TEST(StatisticsTest, runLengths) {
  const std::vector<int32_t> data = {1, 1, 2, 2, 1, 3, 3, 3};
  const std::vector<uint32_t> expected = {2, 2, 1, 3};
  for (const bool populateRunValuesFirst : {false, true}) {
    SCOPED_TRACE(populateRunValuesFirst);
    const auto statistics = nimble::Statistics<int32_t>::create(data);
    if (populateRunValuesFirst) {
      EXPECT_EQ(statistics.runValues().size(), expected.size());
    }

    const auto& runLengths = statistics.runLengths();
    EXPECT_EQ(runLengths, expected);
    EXPECT_EQ(&statistics.runLengths(), &runLengths);
    EXPECT_EQ(runLengths.size(), statistics.consecutiveRepeatCount());
  }

  const std::vector<int32_t> empty;
  EXPECT_TRUE(nimble::Statistics<int32_t>::create(empty).runLengths().empty());
}

TEST(StatisticsTest, minMaxBlocks) {
  const std::vector<uint64_t> data = {9, 3, 7, 2, 10, 4, 5};
  const auto statistics = nimble::Statistics<uint64_t>::create(data);
  const auto& blocks = statistics.minMaxBlocks(/*blockSize=*/3);

  ASSERT_EQ(blocks.size(), 3);
  EXPECT_EQ(blocks[0].count, 3);
  EXPECT_EQ(blocks[0].min, 3);
  EXPECT_EQ(blocks[0].max, 9);
  EXPECT_EQ(blocks[1].count, 3);
  EXPECT_EQ(blocks[1].min, 2);
  EXPECT_EQ(blocks[1].max, 10);
  EXPECT_EQ(blocks[2].count, 1);
  EXPECT_EQ(blocks[2].min, 5);
  EXPECT_EQ(blocks[2].max, 5);

  const std::vector<uint64_t> empty;
  EXPECT_TRUE(
      nimble::Statistics<uint64_t>::create(empty).minMaxBlocks().empty());
}

TEST(StatisticsTest, mostFrequent) {
  const std::vector<int32_t> data = {3, 2, 3, 2, 4};
  const auto statistics = nimble::Statistics<int32_t>::create(data);
  const auto& uniqueCounts = statistics.uniqueCounts().value();

  EXPECT_EQ(uniqueCounts.mostFrequent(), std::make_pair(2, uint64_t{2}));
  EXPECT_EQ(uniqueCounts.mostFrequent(), std::make_pair(2, uint64_t{2}));

  const std::vector<int32_t> empty;
  EXPECT_EQ(
      nimble::Statistics<int32_t>::create(empty)
          .uniqueCounts()
          .value()
          .mostFrequent(),
      std::nullopt);
}

TYPED_TEST(StatisticsNumericTests, create) {
  using T = TypeParam;
  using ValueType = typename T::valueType;

  constexpr auto dataSize = 100;
  constexpr auto repeatSize = 25;
  constexpr auto offset = 10;
  static_assert(dataSize % repeatSize == 0);

  ValueType minValue =
      std::is_signed<T>() ? (ValueType)-offset : (ValueType)offset;

  std::vector<ValueType> data;
  data.resize(dataSize);

  // Repeating data (non-consecutive)
  for (auto i = 0; i < data.size(); ++i) {
    data[i] = minValue + (i % repeatSize);
  }

  auto statistics = T::create({data});
  EXPECT_EQ(repeatSize, statistics.uniqueCounts().value().size());
  EXPECT_EQ(minValue, statistics.min());
  EXPECT_EQ(minValue + repeatSize - 1, statistics.max());
  EXPECT_EQ(1, statistics.minRepeat());
  EXPECT_EQ(1, statistics.maxRepeat());
  EXPECT_EQ(dataSize, statistics.consecutiveRepeatCount());

  for (const auto& pair : statistics.uniqueCounts().value()) {
    EXPECT_EQ(dataSize / repeatSize, pair.second);
  }

  // Repeating data (consecutive)
  std::sort(data.begin(), data.end());
  statistics = T::create({data});
  EXPECT_EQ(repeatSize, statistics.uniqueCounts().value().size());
  EXPECT_EQ(minValue, statistics.min());
  EXPECT_EQ(minValue + repeatSize - 1, statistics.max());
  EXPECT_EQ(dataSize / repeatSize, statistics.minRepeat());
  EXPECT_EQ(dataSize / repeatSize, statistics.maxRepeat());
  EXPECT_EQ(repeatSize, statistics.consecutiveRepeatCount());

  for (const auto& pair : statistics.uniqueCounts().value()) {
    EXPECT_EQ(dataSize / repeatSize, pair.second);
  }

  // Unique data
  for (auto i = 0; i < data.size(); ++i) {
    data[i] = minValue + i;
  }

  statistics = T::create({data});
  EXPECT_EQ(dataSize, statistics.uniqueCounts().value().size());
  EXPECT_EQ(minValue, statistics.min());
  EXPECT_EQ(minValue + dataSize - 1, statistics.max());
  EXPECT_EQ(1, statistics.minRepeat());
  EXPECT_EQ(1, statistics.maxRepeat());
  EXPECT_EQ(dataSize, statistics.consecutiveRepeatCount());

  for (const auto& pair : statistics.uniqueCounts().value()) {
    EXPECT_EQ(1, pair.second);
  }

  // Limits
  if (nimble::isFloatingPointType<ValueType>()) {
    data = {
        std::numeric_limits<ValueType>::min(),
        1,
        std::numeric_limits<ValueType>::lowest(),
        std::numeric_limits<ValueType>::max()};
    statistics = T::create({data});
    EXPECT_EQ(4, statistics.uniqueCounts().value().size());
    EXPECT_EQ(std::numeric_limits<ValueType>::lowest(), statistics.min());
    EXPECT_EQ(std::numeric_limits<ValueType>::max(), statistics.max());
    EXPECT_EQ(1, statistics.minRepeat());
    EXPECT_EQ(1, statistics.maxRepeat());
    EXPECT_EQ(4, statistics.consecutiveRepeatCount());

    for (const auto& pair : statistics.uniqueCounts().value()) {
      EXPECT_EQ(1, pair.second);
    }
  } else {
    data = {
        std::numeric_limits<ValueType>::min(),
        1,
        std::numeric_limits<ValueType>::max()};
    statistics = T::create({data});
    EXPECT_EQ(3, statistics.uniqueCounts().value().size());
    EXPECT_EQ(std::numeric_limits<ValueType>::min(), statistics.min());
    EXPECT_EQ(std::numeric_limits<ValueType>::max(), statistics.max());
    EXPECT_EQ(1, statistics.minRepeat());
    EXPECT_EQ(1, statistics.maxRepeat());
    EXPECT_EQ(3, statistics.consecutiveRepeatCount());

    for (const auto& pair : statistics.uniqueCounts().value()) {
      EXPECT_EQ(1, pair.second);
    }
  }
}

TYPED_TEST(StatisticsBoolTests, create) {
  using T = TypeParam;
  constexpr auto trueCount = 100;
  constexpr auto falseCount = 230;

  auto data = std::make_unique<bool[]>(trueCount + falseCount);
  for (auto i = 0; i < trueCount; ++i) {
    data.get()[i] = true;
  }
  for (auto i = 0; i < falseCount; ++i) {
    data.get()[i + trueCount] = false;
  }

  auto statistics =
      T::create(std::span<const bool>(data.get(), trueCount + falseCount));
  uint64_t expectedDistinctValuesCount = 0;
  expectedDistinctValuesCount += trueCount > 0 ? 1 : 0;
  expectedDistinctValuesCount += falseCount > 0 ? 1 : 0;
  EXPECT_EQ(
      expectedDistinctValuesCount, statistics.uniqueCounts().value().size());
  EXPECT_EQ(trueCount, statistics.uniqueCounts().value().at(true));
  EXPECT_EQ(falseCount, statistics.uniqueCounts().value().at(false));
  EXPECT_EQ(std::min(trueCount, falseCount), statistics.minRepeat());
  EXPECT_EQ(std::max(trueCount, falseCount), statistics.maxRepeat());
  EXPECT_EQ(2, statistics.consecutiveRepeatCount());

  std::shuffle(
      data.get(),
      data.get() + trueCount + falseCount,
      std::mt19937{kShuffleSeed});

  statistics =
      T::create(std::span<const bool>(data.get(), trueCount + falseCount));
  EXPECT_EQ(
      expectedDistinctValuesCount, statistics.uniqueCounts().value().size());
  EXPECT_EQ(trueCount, statistics.uniqueCounts().value().at(true));
  EXPECT_EQ(falseCount, statistics.uniqueCounts().value().at(false));
}

TYPED_TEST(StatisticsIntegerTests, uniqueCountsDenseRange) {
  using T = TypeParam;
  using ValueType = typename T::valueType;

  struct Test {
    std::string_view name;
    std::vector<ValueType> data;
    std::vector<std::pair<ValueType, uint64_t>> expectedCounts;
    ValueType expectedMin;
    ValueType expectedMax;
  };

  std::vector<Test> tests;
  if constexpr (std::is_signed_v<ValueType>) {
    tests = {
        {"single value",
         {std::numeric_limits<ValueType>::lowest(),
          std::numeric_limits<ValueType>::lowest(),
          std::numeric_limits<ValueType>::lowest(),
          std::numeric_limits<ValueType>::lowest()},
         {{std::numeric_limits<ValueType>::lowest(), 4}},
         std::numeric_limits<ValueType>::lowest(),
         std::numeric_limits<ValueType>::lowest()},
        {"crosses zero",
         {static_cast<ValueType>(-2),
          static_cast<ValueType>(-1),
          static_cast<ValueType>(-2),
          static_cast<ValueType>(0),
          static_cast<ValueType>(1),
          static_cast<ValueType>(1)},
         {{static_cast<ValueType>(-2), 2},
          {static_cast<ValueType>(-1), 1},
          {static_cast<ValueType>(0), 1},
          {static_cast<ValueType>(1), 2}},
         static_cast<ValueType>(-2),
         static_cast<ValueType>(1)},
        {"near lowest",
         {std::numeric_limits<ValueType>::lowest(),
          static_cast<ValueType>(std::numeric_limits<ValueType>::lowest() + 1),
          std::numeric_limits<ValueType>::lowest()},
         {{std::numeric_limits<ValueType>::lowest(), 2},
          {static_cast<ValueType>(std::numeric_limits<ValueType>::lowest() + 1),
           1}},
         std::numeric_limits<ValueType>::lowest(),
         static_cast<ValueType>(std::numeric_limits<ValueType>::lowest() + 1)},
    };
  } else {
    tests = {
        {"single value",
         {std::numeric_limits<ValueType>::max(),
          std::numeric_limits<ValueType>::max(),
          std::numeric_limits<ValueType>::max(),
          std::numeric_limits<ValueType>::max()},
         {{std::numeric_limits<ValueType>::max(), 4}},
         std::numeric_limits<ValueType>::max(),
         std::numeric_limits<ValueType>::max()},
        {"near max",
         {static_cast<ValueType>(std::numeric_limits<ValueType>::max() - 2),
          static_cast<ValueType>(std::numeric_limits<ValueType>::max() - 1),
          std::numeric_limits<ValueType>::max(),
          static_cast<ValueType>(std::numeric_limits<ValueType>::max() - 2),
          std::numeric_limits<ValueType>::max()},
         {{static_cast<ValueType>(std::numeric_limits<ValueType>::max() - 2),
           2},
          {static_cast<ValueType>(std::numeric_limits<ValueType>::max() - 1),
           1},
          {std::numeric_limits<ValueType>::max(), 2}},
         static_cast<ValueType>(std::numeric_limits<ValueType>::max() - 2),
         std::numeric_limits<ValueType>::max()},
        {"starts at zero",
         {static_cast<ValueType>(0),
          static_cast<ValueType>(1),
          static_cast<ValueType>(0),
          static_cast<ValueType>(2)},
         {{static_cast<ValueType>(0), 2},
          {static_cast<ValueType>(1), 1},
          {static_cast<ValueType>(2), 1}},
         static_cast<ValueType>(0),
         static_cast<ValueType>(2)},
    };
  }

  for (const auto& test : tests) {
    SCOPED_TRACE(test.name);
    const auto statistics = T::create({test.data});
    const auto& uniqueCounts = statistics.uniqueCounts().value();
    EXPECT_EQ(test.expectedCounts.size(), uniqueCounts.size());
    for (const auto& [value, count] : test.expectedCounts) {
      EXPECT_EQ(count, uniqueCounts.at(value));
    }
    EXPECT_EQ(test.expectedMin, statistics.min());
    EXPECT_EQ(test.expectedMax, statistics.max());
  }
}

// Counting picks a table, a hash map or a sort by range and cardinality. Each
// case below lands on a different one, and all must agree with a plain count.
TYPED_TEST(StatisticsIntegerTests, uniqueCountsAcrossCountingStrategies) {
  using T = TypeParam;
  using ValueType = typename T::valueType;
  using UnsignedType = std::make_unsigned_t<ValueType>;

  struct Test {
    std::string_view name;
    uint64_t distinct;
    uint64_t step;
  };
  // Stepped so that the range is wide even where the distinct count is small,
  // and anchored at the type's lowest value so that offsets from min cross
  // zero in the signed types.
  const uint64_t typeMax = std::numeric_limits<UnsignedType>::max();
  const std::vector<Test> tests = {
      {"few distinct over a wide range", 300, 97},
      {"many distinct over a narrow range", 40'000, 1},
      {"many distinct over a wide range", 40'000, typeMax / 40'000},
  };

  std::mt19937 rng{kShuffleSeed};
  for (const auto& test : tests) {
    SCOPED_TRACE(test.name);
    const uint64_t step = std::max<uint64_t>(test.step, 1);
    const uint64_t distinct = std::min(test.distinct, typeMax / step + 1);
    std::vector<ValueType> data;
    std::map<ValueType, uint64_t> expected;
    for (uint64_t i = 0; i < distinct; ++i) {
      const auto value = static_cast<ValueType>(static_cast<UnsignedType>(
          static_cast<uint64_t>(static_cast<UnsignedType>(
              std::numeric_limits<ValueType>::lowest())) +
          i * step));
      // Counts of one to three, so that the counts tie as well as differ.
      const uint64_t repeats = i % 3 + 1;
      for (uint64_t repeat = 0; repeat < repeats; ++repeat) {
        data.push_back(value);
      }
      expected[value] += repeats;
    }
    std::shuffle(data.begin(), data.end(), rng);

    const auto statistics = T::create({data});
    const auto& uniqueCounts = statistics.uniqueCounts().value();
    std::map<ValueType, uint64_t> actual;
    for (const auto& [value, count] : uniqueCounts) {
      EXPECT_TRUE(actual.emplace(value, count).second);
    }
    EXPECT_EQ(expected, actual);
    EXPECT_EQ(expected.size(), uniqueCounts.size());
    for (const auto& [value, count] : expected) {
      EXPECT_EQ(count, uniqueCounts.at(value));
    }
  }
}

// Exact while the offsets from min fit in the bits it tells apart, a lower
// bound past that, and the exact count once the unique counts exist.
TYPED_TEST(StatisticsIntegerTests, distinctLowerBound) {
  using T = TypeParam;
  using ValueType = typename T::valueType;
  using UnsignedType = std::make_unsigned_t<ValueType>;

  std::vector<ValueType> narrow;
  for (int i = 0; i < 1'000; ++i) {
    narrow.push_back(
        static_cast<ValueType>(
            static_cast<UnsignedType>(
                std::numeric_limits<ValueType>::lowest()) +
            static_cast<UnsignedType>(i % 97)));
  }
  const auto narrowStatistics = T::create({narrow});
  EXPECT_EQ(97, narrowStatistics.distinctLowerBound());

  if constexpr (sizeof(ValueType) >= 4) {
    // Distinct values that collide in their low bits: the bound sees 3 of 6.
    const uint64_t high = uint64_t{1}
        << nimble::Statistics<ValueType>::kDistinctBoundBits;
    std::vector<ValueType> wide;
    for (uint64_t i = 0; i < 6; ++i) {
      wide.push_back(static_cast<ValueType>((i % 3) + (i / 3) * high));
    }
    const auto wideStatistics = T::create({wide});
    EXPECT_EQ(3, wideStatistics.distinctLowerBound());
    EXPECT_EQ(6, wideStatistics.uniqueCounts().value().size());
    EXPECT_EQ(6, wideStatistics.distinctLowerBound());
  }
}

template <typename T>
void verifyString(
    std::function<T(std::vector<std::string> data)> genStatisticsType) {
  constexpr auto uniqueStrings = 10;
  constexpr auto maxRepeat = 20;

  std::vector<std::string> data;
  data.reserve(uniqueStrings * maxRepeat);

  uint64_t totalLength = 0;
  uint64_t totalRepeatLength = 0;
  auto currentRepeat = maxRepeat;
  for (auto i = 0; i < uniqueStrings; ++i) {
    for (auto j = 0; j < currentRepeat; ++j) {
      data.emplace_back(i + 1, 'a' + i);
      totalLength += data.back().size();
    }
    totalRepeatLength += i + 1;
    --currentRepeat;
  }

  T statistics = genStatisticsType(data);

  EXPECT_EQ(uniqueStrings, statistics.uniqueCounts().value().size());
  EXPECT_EQ(maxRepeat - uniqueStrings + 1, statistics.minRepeat());
  EXPECT_EQ(maxRepeat, statistics.maxRepeat());
  EXPECT_EQ(uniqueStrings, statistics.consecutiveRepeatCount());
  EXPECT_EQ(totalLength, statistics.totalStringsLength());
  EXPECT_EQ(totalRepeatLength, statistics.totalStringsRepeatLength());
  EXPECT_EQ(
      totalRepeatLength, statistics.uniqueCounts().value().uniqueStringBytes());
  EXPECT_EQ(
      std::string(uniqueStrings, 'a' + uniqueStrings - 1), statistics.max());
  EXPECT_EQ(std::string(1, 'a'), statistics.min());

  currentRepeat = maxRepeat;
  for (auto i = 0; i < statistics.uniqueCounts().value().size(); ++i) {
    EXPECT_EQ(
        statistics.uniqueCounts().value().at(std::string(i + 1, 'a' + i)),
        currentRepeat--);
  }

  std::shuffle(data.begin(), data.end(), std::mt19937{kShuffleSeed});

  statistics = genStatisticsType(data);

  EXPECT_EQ(uniqueStrings, statistics.uniqueCounts().value().size());
  EXPECT_EQ(totalLength, statistics.totalStringsLength());
  EXPECT_EQ(
      std::string(uniqueStrings, 'a' + uniqueStrings - 1), statistics.max());
  EXPECT_EQ(std::string(1, 'a'), statistics.min());

  const auto uniqueCounts7 = statistics.uniqueCounts().value();
  for (auto i = 0; i < statistics.uniqueCounts().value().size(); ++i) {
    EXPECT_EQ(
        statistics.uniqueCounts().value().at(std::string(i + 1, 'a' + i)),
        currentRepeat--);
  }
}

TYPED_TEST(StatisticsStringTests, create) {
  using T = TypeParam;

  constexpr auto uniqueStrings = 10;
  constexpr auto maxRepeat = 20;

  std::vector<std::string> data;
  data.reserve(uniqueStrings * maxRepeat);

  uint64_t totalLength = 0;
  uint64_t totalRepeatLength = 0;
  auto currentRepeat = maxRepeat;
  for (auto i = 0; i < uniqueStrings; ++i) {
    for (auto j = 0; j < currentRepeat; ++j) {
      data.emplace_back(i + 1, 'a' + i);
      totalLength += data.back().size();
    }
    totalRepeatLength += i + 1;
    --currentRepeat;
  }

  T statistics =
      nimble::Statistics<std::string_view, std::string>::create(data);

  EXPECT_EQ(uniqueStrings, statistics.uniqueCounts().value().size());
  EXPECT_EQ(maxRepeat - uniqueStrings + 1, statistics.minRepeat());
  EXPECT_EQ(maxRepeat, statistics.maxRepeat());
  EXPECT_EQ(uniqueStrings, statistics.consecutiveRepeatCount());
  EXPECT_EQ(totalLength, statistics.totalStringsLength());
  EXPECT_EQ(totalRepeatLength, statistics.totalStringsRepeatLength());
  EXPECT_EQ(
      totalRepeatLength, statistics.uniqueCounts().value().uniqueStringBytes());
  EXPECT_EQ(
      std::string(uniqueStrings, 'a' + uniqueStrings - 1), statistics.max());
  EXPECT_EQ(std::string(1, 'a'), statistics.min());

  currentRepeat = maxRepeat;
  for (auto i = 0; i < statistics.uniqueCounts().value().size(); ++i) {
    EXPECT_EQ(
        statistics.uniqueCounts().value().at(std::string(i + 1, 'a' + i)),
        currentRepeat--);
  }

  std::shuffle(data.begin(), data.end(), std::mt19937{kShuffleSeed});

  statistics =
      nimble::Statistics<std::string_view, std::string>::create({data});

  EXPECT_EQ(uniqueStrings, statistics.uniqueCounts().value().size());
  EXPECT_EQ(totalLength, statistics.totalStringsLength());
  EXPECT_EQ(
      std::string(uniqueStrings, 'a' + uniqueStrings - 1), statistics.max());
  EXPECT_EQ(std::string(1, 'a'), statistics.min());

  currentRepeat = maxRepeat;
  for (auto i = 0; i < statistics.uniqueCounts().value().size(); ++i) {
    EXPECT_EQ(
        statistics.uniqueCounts().value().at(std::string(i + 1, 'a' + i)),
        currentRepeat--);
  }
}

TYPED_TEST(StatisticsNumericTests, repeat) {
  using T = TypeParam;
  using ValueType = typename T::valueType;

  struct Test {
    std::vector<ValueType> data;
    uint64_t expectedConsecutiveRepeatCount;
    uint64_t expectedMinRepeat;
    uint64_t expectedMaxRepeat;
    ValueType expectedMin;
    ValueType expectedMax;
  };

  std::vector<Test> tests{
      {{}, 0, 0, 0, 0, 0},
      {{1}, 1, 1, 1, 1, 1},
      {{1, 1}, 1, 2, 2, 1, 1},
      {{1, 2}, 2, 1, 1, 1, 2},
      {{1, 1, 2}, 2, 1, 2, 1, 2},
      {{1, 2, 2}, 2, 1, 2, 1, 2},
      {{1, 2, 1}, 3, 1, 1, 1, 2},
      {{1, 1, 2, 1}, 3, 1, 2, 1, 2},
      {{1, 2, 2, 1}, 3, 1, 2, 1, 2},
      {{1, 2, 1, 1}, 3, 1, 2, 1, 2},
      {{1, 1, 1, 2, 2, 2, 1, 1}, 3, 2, 3, 1, 2},
  };

  for (const auto& test : tests) {
    auto statistics = T::create({test.data});
    EXPECT_EQ(
        test.expectedConsecutiveRepeatCount,
        statistics.consecutiveRepeatCount());
    EXPECT_EQ(test.expectedMinRepeat, statistics.minRepeat());
    EXPECT_EQ(test.expectedMaxRepeat, statistics.maxRepeat());
    EXPECT_EQ(test.expectedMin, statistics.min());
    EXPECT_EQ(test.expectedMax, statistics.max());
  }
}

TYPED_TEST(StatisticsIntegerTests, buckets) {
  using T = TypeParam;
  using ValueType = typename T::valueType;
  using UnsignedValueType = typename std::make_unsigned<ValueType>::type;

  constexpr auto offset = 10;
  constexpr auto repeatSize = 25;
  constexpr auto dataSize = 100;
  ValueType minValue =
      std::is_signed<ValueType>() ? (ValueType)-offset : (ValueType)offset;

  std::vector<ValueType> data;
  data.reserve(dataSize + (sizeof(ValueType) * 8));

  std::array<uint64_t, 10> expectedBuckets{};

  // Repeating data (non-consecutive)
  for (auto i = 0; i < dataSize; ++i) {
    data.push_back(minValue + (i % repeatSize));
  }

  for (auto i = 0; i < sizeof(ValueType) * 8; ++i) {
    auto value = std::numeric_limits<ValueType>::max() >> i;
    if (value >= offset) {
      data.push_back(value);
    }
  }

  for (auto i = 0; i < data.size(); ++i) {
    char buffer[10];
    auto pos = buffer;
    nimble::varint::writeVarint(
        static_cast<UnsignedValueType>(
            static_cast<UnsignedValueType>(data[i]) -
            static_cast<UnsignedValueType>(minValue)),
        &pos);
    ++expectedBuckets[pos - buffer - 1];
  }

  auto statistics = T::create({data});
  auto& buckets = statistics.bucketCounts();
  EXPECT_LE(buckets.size(), expectedBuckets.size());
  EXPECT_GT(buckets.size(), 0);
  for (auto i = 0; i < buckets.size(); ++i) {
    EXPECT_EQ(expectedBuckets[i], buckets[i]) << "index: " << i;
  }
}

// Both are accumulated in blocks, so the reference below walks the rows one
// at a time, over lengths that leave partial blocks and over values that reach
// both ends of the type.
TYPED_TEST(StatisticsIntegerTests, adjacentPairStatsAndMinMaxBlocks) {
  using T = TypeParam;
  using ValueType = typename T::valueType;
  using UnsignedType = std::make_unsigned_t<ValueType>;

  std::mt19937_64 rng{kShuffleSeed};
  for (const size_t size :
       {size_t{1}, size_t{2}, size_t{5'000}, size_t{9'000}}) {
    SCOPED_TRACE(size);
    std::vector<ValueType> data(size);
    for (size_t i = 0; i < size; ++i) {
      // Mostly the extremes, so steps span the whole range.
      const uint64_t draw = rng();
      data[i] = draw % 3 == 0 ? std::numeric_limits<ValueType>::max()
          : draw % 3 == 1     ? std::numeric_limits<ValueType>::lowest()
                              : static_cast<ValueType>(draw >> 8);
    }
    std::vector<UnsignedType> unsignedData(size);
    for (size_t i = 0; i < size; ++i) {
      unsignedData[i] = static_cast<UnsignedType>(data[i]);
    }

    uint64_t nonDecreasingCount{0};
    uint64_t maxIncrease{0};
    uint64_t sumAbsoluteDelta{0};
    for (size_t i = 1; i < size; ++i) {
      const uint64_t previous = static_cast<UnsignedType>(data[i - 1]);
      const uint64_t value = static_cast<UnsignedType>(data[i]);
      if (value >= previous) {
        ++nonDecreasingCount;
        maxIncrease = std::max(maxIncrease, value - previous);
        sumAbsoluteDelta += value - previous;
      } else {
        sumAbsoluteDelta += previous - value;
      }
    }
    const auto statistics = T::create({data});
    const auto& pairs = statistics.adjacentPairStats();
    EXPECT_EQ(nonDecreasingCount, pairs.nonDecreasingCount);
    EXPECT_EQ(maxIncrease, pairs.maxIncrease);
    EXPECT_EQ(sumAbsoluteDelta, pairs.sumAbsoluteDelta);

    constexpr uint16_t kBlockSize{1'024};
    const auto unsignedStatistics =
        nimble::Statistics<UnsignedType>::create({unsignedData});
    const auto& blocks = unsignedStatistics.minMaxBlocks(kBlockSize);
    ASSERT_EQ((size + kBlockSize - 1) / kBlockSize, blocks.size());
    for (size_t block = 0; block < blocks.size(); ++block) {
      const size_t start = block * kBlockSize;
      const size_t end = std::min(size, start + kBlockSize);
      const auto [minIt, maxIt] = std::minmax_element(
          unsignedData.begin() + start, unsignedData.begin() + end);
      EXPECT_EQ(end - start, blocks[block].count);
      EXPECT_EQ(*minIt, blocks[block].min);
      EXPECT_EQ(*maxIt, blocks[block].max);
    }
  }
}

TYPED_TEST(StatisticsIntegerTests, bitFlipProfileConstant) {
  using T = TypeParam;
  using ValueType = typename T::valueType;

  std::vector<ValueType> data(100, ValueType{42});
  auto statistics = T::create({data});
  const auto& profile = statistics.bitFlipProfile();

  EXPECT_EQ(profile.numBits, static_cast<int>(sizeof(ValueType) * 8));
  EXPECT_EQ(profile.variance, 0.0);
  for (int b = 0; b < profile.numBits; ++b) {
    EXPECT_EQ(profile.flipProbability[b], 0.0) << "bit: " << b;
  }
}

TYPED_TEST(StatisticsIntegerTests, bitFlipProfileAlternatingIsFullFlip) {
  using T = TypeParam;
  using ValueType = typename T::valueType;

  std::vector<ValueType> data;
  for (int i = 0; i < 100; ++i) {
    data.push_back(i % 2 == 0 ? ValueType{0} : static_cast<ValueType>(-1));
  }
  auto statistics = T::create({data});
  const auto& profile = statistics.bitFlipProfile();

  for (int b = 0; b < profile.numBits; ++b) {
    EXPECT_EQ(profile.flipProbability[b], 1.0) << "bit: " << b;
  }
  EXPECT_EQ(profile.variance, 0.0);
}

TEST(StatisticsTest, bitFlipProfileIsLazyAndCached) {
  std::vector<uint64_t> data{0, 1, 2, 3, 4, 5};
  auto statistics = nimble::Statistics<uint64_t>::create({data});
  const auto& first = statistics.bitFlipProfile();
  const auto& second = statistics.bitFlipProfile();
  EXPECT_EQ(&first, &second);
  EXPECT_EQ(first.numBits, 64);
}

// Constancy is the only question ConstantEncoding asks, and it used to be
// answered by building a unique-value map. The scan has to stop at the first
// value that differs rather than reading the whole stream.
TEST(StatisticsTests, isConstantDetectsConstantAndNonConstant) {
  const std::vector<int64_t> constant(1'000, 7);
  EXPECT_TRUE(nimble::Statistics<int64_t>::create(constant).isConstant());

  const std::vector<int64_t> single{42};
  EXPECT_TRUE(nimble::Statistics<int64_t>::create(single).isConstant());

  // Differs only in the last position, so a correct scan still reads it all.
  std::vector<int64_t> tail(1'000, 7);
  tail.back() = 8;
  EXPECT_FALSE(nimble::Statistics<int64_t>::create(tail).isConstant());

  // Differs at the second position, which is where the scan should stop.
  std::vector<int64_t> head(1'000, 7);
  head[1] = 8;
  EXPECT_FALSE(nimble::Statistics<int64_t>::create(head).isConstant());
}

// isConstant() must agree with the unique count it replaced, on every type
// ConstantEncoding is selected for.
TEST(StatisticsTests, isConstantAgreesWithUniqueCounts) {
  {
    const std::vector<int64_t> data{5, 5, 5};
    const auto stats = nimble::Statistics<int64_t>::create(data);
    EXPECT_EQ(stats.uniqueCounts().value().size() == 1, stats.isConstant());
  }
  {
    // Floating point runs on the physical representation, so constancy is
    // bit-exact -- which is what the non-ALP encodings require.
    const std::vector<uint64_t> data{
        std::bit_cast<uint64_t>(1.5), std::bit_cast<uint64_t>(1.5)};
    const auto stats = nimble::Statistics<uint64_t>::create(data);
    EXPECT_EQ(stats.uniqueCounts().value().size() == 1, stats.isConstant());
  }
  {
    constexpr bool kAllTrue[]{true, true, true};
    const auto stats =
        nimble::Statistics<bool>::create(std::span<const bool>{kAllTrue});
    EXPECT_EQ(stats.uniqueCounts().value().size() == 1, stats.isConstant());
  }
  {
    const std::vector<std::string_view> data{"abc", "abc"};
    const auto stats = nimble::Statistics<std::string_view>::create(data);
    EXPECT_EQ(stats.uniqueCounts().value().size() == 1, stats.isConstant());
  }
  {
    const std::vector<std::string_view> mixed{"abc", "abd"};
    const auto stats = nimble::Statistics<std::string_view>::create(mixed);
    EXPECT_EQ(stats.uniqueCounts().value().size() == 1, stats.isConstant());
  }
}
