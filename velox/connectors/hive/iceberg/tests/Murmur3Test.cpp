/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/connectors/hive/iceberg/Murmur3.h"
#include <gtest/gtest.h>
#include "folly/Random.h"
#include "velox/type/TimestampConversion.h"

namespace facebook::velox::connector::hive::iceberg {
class Murmur3HashTest : public ::testing::Test {
 public:
  void SetUp() override {
    rng_.seed(1);
  }

  void TearDown() override {}

  // Little-endian.
  static std::vector<char> toBytes(uint64_t value) {
    std::vector<char> bytes;
    bytes.reserve(sizeof(uint64_t));
    for (int32_t i = 0; i < sizeof(uint64_t); ++i) {
      bytes[i] = static_cast<char>((value >> (8 * i)) & 0xFF);
    }
    return bytes;
  }

  static int32_t parseDateToEpochDays(const std::string& date) {
    if (date.size() != 10 || date[4] != '-' || date[7] != '-') {
      throw std::invalid_argument("Invalid date format, expected YYYY-MM-DD.");
    }

    const auto year = std::stoi(date.substr(0, 4));
    const auto month = std::stoi(date.substr(5, 2));
    const auto day = std::stoi(date.substr(8, 2));

    auto result = util::daysSinceEpochFromDate(year, month, day);
    if (result.hasError()) {
      VELOX_FAIL("Date out of range: {}-{}-{}", year, month, day);
    }
    return result.value();
  }

  static int64_t parseDateTimeToEpochMicroSeconds(const std::string& dateTime) {
    if (dateTime.size() != 19 || dateTime[4] != '-' || dateTime[7] != '-' ||
        dateTime[10] != 'T' || dateTime[13] != ':' || dateTime[16] != ':') {
      VELOX_FAIL("Invalid datetime format, expected YYYY-MM-DDTHH:MM:SS");
    }

    const auto year = std::stoi(dateTime.substr(0, 4));
    const auto month = std::stoi(dateTime.substr(5, 2));
    const auto day = std::stoi(dateTime.substr(8, 2));
    const auto hour = std::stoi(dateTime.substr(11, 2));
    const auto minute = std::stoi(dateTime.substr(14, 2));
    const auto second = std::stoi(dateTime.substr(17, 2));
    auto daysResult = util::daysSinceEpochFromDate(year, month, day);
    if (daysResult.hasError()) {
      VELOX_FAIL("Invalid date: {}-{}-{}", year, month, day);
    }

    const auto microsSinceMidnight = util::fromTime(hour, minute, second, 0);
    const auto timestamp =
        util::fromDatetime(daysResult.value(), microsSinceMidnight);
    return timestamp.getSeconds() * 1'000'000L + timestamp.getNanos();
  }

  template <typename T>
  void
  verifyHashBucket(T input, uint32_t bucketCount, uint32_t expectedBucket) {
    const auto hash = Murmur3_32::hash(input);
    uint32_t actualBucket = (hash & 0X7FFFFFFF) % bucketCount;
    EXPECT_EQ(actualBucket, expectedBucket)
        << "Input: " << input << ", Bucket Count: " << bucketCount
        << ", Hash: " << hash << ", Expected Bucket: " << expectedBucket
        << ", Actual Bucket: " << actualBucket;
  }

 protected:
  folly::Random::DefaultGenerator rng_;
};

TEST_F(Murmur3HashTest, testSpecValues) {
  auto hash = Murmur3_32::hash(34);
  EXPECT_EQ(hash, 2'017'239'379);

  const auto days = parseDateToEpochDays("2017-11-16");
  EXPECT_EQ(days, 17'486);
  hash = Murmur3_32::hash(days);
  EXPECT_EQ(hash, -653'330'422);

  const auto seconds = parseDateTimeToEpochMicroSeconds("2017-11-16T22:31:08");
  EXPECT_EQ(seconds, 1'510'871'468'000'000L);
  hash = Murmur3_32::hash(seconds);
  EXPECT_EQ(hash, -2'047'944'441);

  const auto bytes = new char[4]{0x00, 0x01, 0x02, 0x03};
  hash = Murmur3_32::hash(bytes, 4);
  EXPECT_EQ(hash, -188'683'207);

  hash = Murmur3_32::hash("iceberg");
  EXPECT_EQ(hash, 1'210'000'089);
}

TEST_F(Murmur3HashTest, hashString) {
  const std::vector<std::tuple<StringView, uint32_t, uint32_t>> testCases = {
      {"abcdefg", 5, 4},
      {"abc", 128, 122},
      {"abcde", 64, 54},
      {"测试", 12, 8},
      {"测试raul试测", 16, 1},
      {"", 16, 0}};

  for (const auto& [input, bucketCount, expectedBucket] : testCases) {
    verifyHashBucket(input, bucketCount, expectedBucket);
  }
}

TEST_F(Murmur3HashTest, hashInteger) {
  const std::vector<std::tuple<int32_t, uint32_t, uint32_t>> testCases = {
      {8, 10, 3}, {34, 100, 79}};

  for (const auto& [input, bucketCount, expectedBucket] : testCases) {
    verifyHashBucket(input, bucketCount, expectedBucket);
  }
}

TEST_F(Murmur3HashTest, hashTrue) {
  const auto hash = Murmur3_32::hash(1);
  EXPECT_EQ(hash, 1'392'991'556U);
}

TEST_F(Murmur3HashTest, hashDate) {
  const std::vector<std::tuple<int32_t, uint32_t, uint32_t>> testCases = {
      {parseDateToEpochDays("1970-01-09"), 10, 3},
      {parseDateToEpochDays("1970-02-04"), 100, 79}};

  for (const auto& [input, bucketCount, expectedBucket] : testCases) {
    verifyHashBucket(input, bucketCount, expectedBucket);
  }
}

TEST_F(Murmur3HashTest, hashLong) {
  const std::vector<std::tuple<int32_t, uint32_t, uint32_t>> testCases = {
      {34L, 100, 79}, {0L, 100, 76}, {-34L, 100, 97}, {-1L, 2, 0}};

  for (const auto& [input, bucketCount, expectedBucket] : testCases) {
    verifyHashBucket(input, bucketCount, expectedBucket);
  }
}

TEST_F(Murmur3HashTest, hashDecimal) {
  const std::vector<std::tuple<int128_t, uint32_t, uint32_t>> testCases = {
      {1234L, 64, 56},
      {1230L, 18, 13},
      {12999L, 16, 2},
      {5L, 32, 21},
      {5L, 18, 3}};

  for (const auto& [input, bucketCount, expectedBucket] : testCases) {
    const auto hash = Murmur3_32::hashDecimal(input);
    auto actualBucket = (hash & 0X7FFFFFFF) % bucketCount;
    EXPECT_EQ(actualBucket, expectedBucket);
  }
}

TEST_F(Murmur3HashTest, hashBinary) {
  const std::string s("abc\0\0", 5);
  const std::vector<std::tuple<StringView, uint32_t, uint32_t>> testCases = {
      {StringView("abcdefg"), 12, 10},
      {StringView(s), 18, 13},
      {StringView("abc"), 48, 42},
      {StringView("测试_"), 16, 3}};
  for (const auto& [input, bucketCount, expectedBucket] : testCases) {
    verifyHashBucket(input, bucketCount, expectedBucket);
  }
}

TEST_F(Murmur3HashTest, hashIntegerAndBytes) {
  const auto number = folly::Random::rand32(rng_);
  const auto hashOfInteger = Murmur3_32::hash(number);
  const auto hashOfBytes = Murmur3_32::hash(toBytes(number).data(), 8);
  EXPECT_EQ(hashOfInteger, hashOfBytes);
}

TEST_F(Murmur3HashTest, hashLongAndBytes) {
  const auto number = folly::Random::rand64(rng_);
  const auto hashOfLong = Murmur3_32::hash(number);
  const auto hashOfBytes = Murmur3_32::hash(toBytes(number).data(), 8);
  EXPECT_EQ(hashOfLong, hashOfBytes);
}

} // namespace facebook::velox::connector::hive::iceberg
