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

#include "velox/connectors/hive/iceberg/IcebergPartitionName.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

template <typename T>
std::string toPath(TransformType transform, T value, const TypePtr& type) {
  return IcebergPartitionName::toName(value, type, transform);
}

std::string timestampToPath(const Timestamp& timestamp) {
  return toPath(TransformType::kIdentity, timestamp, TIMESTAMP());
}

std::string testString(
    const std::string& value,
    const TypePtr& typePtr = VARCHAR()) {
  auto identityResult =
      toPath(TransformType::kIdentity, StringView(value), typePtr);
  auto truncateResult =
      toPath(TransformType::kTruncate, StringView(value), typePtr);
  EXPECT_EQ(identityResult, truncateResult);
  return identityResult;
}

std::string testVarbinary(const std::string& value) {
  return testString(value, VARBINARY());
}

std::string testInteger(int32_t value) {
  auto identityResult = toPath(TransformType::kIdentity, value, INTEGER());
  auto bucketResult = toPath(TransformType::kBucket, value, INTEGER());
  auto truncResult = toPath(TransformType::kTruncate, value, INTEGER());
  EXPECT_EQ(identityResult, truncResult);
  EXPECT_EQ(bucketResult, truncResult);
  return truncResult;
}

TEST(IcebergPartitionPathTest, integer) {
  EXPECT_EQ(testInteger(0), "0");
  EXPECT_EQ(testInteger(1), "1");
  EXPECT_EQ(testInteger(100), "100");
  EXPECT_EQ(testInteger(-100), "-100");
  EXPECT_EQ(testInteger(128), "128");
  EXPECT_EQ(testInteger(1024), "1024");
}

TEST(IcebergPartitionPathTest, date) {
  EXPECT_EQ(toPath(TransformType::kIdentity, 18'262, DATE()), "2020-01-01");
  EXPECT_EQ(toPath(TransformType::kIdentity, 0, DATE()), "1970-01-01");
  EXPECT_EQ(toPath(TransformType::kIdentity, -1, DATE()), "1969-12-31");
  EXPECT_EQ(toPath(TransformType::kIdentity, 2'932'897, DATE()), "10000-01-01");
}

TEST(IcebergPartitionPathTest, boolean) {
  EXPECT_EQ(toPath(TransformType::kIdentity, true, BOOLEAN()), "true");
  EXPECT_EQ(toPath(TransformType::kIdentity, false, BOOLEAN()), "false");
}

TEST(IcebergPartitionPathTest, string) {
  EXPECT_EQ(testString("a/b/c=d"), "a/b/c=d");
  EXPECT_EQ(testString(""), "");
  EXPECT_EQ(testString("abc"), "abc");
}

TEST(IcebergPartitionPathTest, varbinary) {
  EXPECT_EQ(testVarbinary("\x48\x65\x6c\x6c\x6f"), "SGVsbG8=");
  EXPECT_EQ(testVarbinary("\x1\x2\x3"), "AQID");
  EXPECT_EQ(testVarbinary(""), "");
}

TEST(IcebergPartitionPathTest, timestamp) {
  EXPECT_EQ(timestampToPath(Timestamp(0, 0)), "1970-01-01T00:00:00");
  EXPECT_EQ(
      timestampToPath(Timestamp(1'609'459'200, 999'000'000)),
      "2021-01-01T00:00:00.999");
  EXPECT_EQ(
      timestampToPath(Timestamp(1'640'995'200, 500'000'000)),
      "2022-01-01T00:00:00.5");
  EXPECT_EQ(
      timestampToPath(Timestamp(-1, 999'000'000)), "1969-12-31T23:59:59.999");
  EXPECT_EQ(
      timestampToPath(Timestamp(253'402'300'800, 100'000'000)),
      "+10000-01-01T00:00:00.1");
  EXPECT_EQ(
      timestampToPath(Timestamp(-62'170'000'000, 0)), "-0001-11-29T19:33:20");
  EXPECT_EQ(
      timestampToPath(Timestamp(-62'167'219'199, 0)), "0000-01-01T00:00:01");
}

TEST(IcebergPartitionPathTest, year) {
  EXPECT_EQ(toPath(TransformType::kYear, 0, INTEGER()), "1970");
  EXPECT_EQ(toPath(TransformType::kYear, 1, INTEGER()), "1971");
  EXPECT_EQ(toPath(TransformType::kYear, 8'030, INTEGER()), "10000");
  EXPECT_EQ(toPath(TransformType::kYear, -1, INTEGER()), "1969");
  EXPECT_EQ(toPath(TransformType::kYear, -50, INTEGER()), "1920");
}

TEST(IcebergPartitionPathTest, month) {
  EXPECT_EQ(toPath(TransformType::kMonth, 0, INTEGER()), "1970-01");
  EXPECT_EQ(toPath(TransformType::kMonth, 1, INTEGER()), "1970-02");
  EXPECT_EQ(toPath(TransformType::kMonth, 11, INTEGER()), "1970-12");
  EXPECT_EQ(toPath(TransformType::kMonth, 612, INTEGER()), "2021-01");
  EXPECT_EQ(toPath(TransformType::kMonth, -1, INTEGER()), "1969-12");
  EXPECT_EQ(toPath(TransformType::kMonth, -13, INTEGER()), "1968-12");
}

TEST(IcebergPartitionPathTest, day) {
  EXPECT_EQ(toPath(TransformType::kDay, 0, DATE()), "1970-01-01");
  EXPECT_EQ(toPath(TransformType::kDay, 1, DATE()), "1970-01-02");
  EXPECT_EQ(toPath(TransformType::kDay, 18'262, DATE()), "2020-01-01");
  EXPECT_EQ(toPath(TransformType::kDay, -1, DATE()), "1969-12-31");
}

TEST(IcebergPartitionPathTest, hour) {
  EXPECT_EQ(toPath(TransformType::kHour, 0, INTEGER()), "1970-01-01-00");
  EXPECT_EQ(toPath(TransformType::kHour, 1, INTEGER()), "1970-01-01-01");
  EXPECT_EQ(toPath(TransformType::kHour, 24, INTEGER()), "1970-01-02-00");
  EXPECT_EQ(toPath(TransformType::kHour, 438'288, INTEGER()), "2020-01-01-00");
  EXPECT_EQ(toPath(TransformType::kHour, -1, INTEGER()), "1969-12-31-23");
}

} // namespace

} // namespace facebook::velox::connector::hive::iceberg
