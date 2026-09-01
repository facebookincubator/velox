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
#include <limits>
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
