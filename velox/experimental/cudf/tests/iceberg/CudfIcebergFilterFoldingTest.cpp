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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergFilterTransform.h"

#include "velox/type/TimestampConversion.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {
namespace {

ConstantFilterFold fold(
    const common::Filter& filter,
    const TypePtr& type,
    const std::optional<std::string>& value,
    bool readTimestampAsLocalTime = false) {
  return foldFilterOnConstant(filter, type, value, readTimestampAsLocalTime);
}

} // namespace

TEST(CudfIcebergFilterFoldingTest, scalarValues) {
  const common::BytesValues bytesFilter{{"apples"}, /*nullAllowed=*/false};

  EXPECT_EQ(
      fold(bytesFilter, VARCHAR(), "apples"), ConstantFilterFold::kAlwaysTrue);
  EXPECT_EQ(
      fold(bytesFilter, VARCHAR(), "oranges"),
      ConstantFilterFold::kAlwaysFalse);

  const common::BigintRange integerFilter{5, 10, /*nullAllowed=*/false};
  EXPECT_EQ(
      fold(integerFilter, INTEGER(), "5"), ConstantFilterFold::kAlwaysTrue);
  EXPECT_EQ(
      fold(integerFilter, INTEGER(), "11"), ConstantFilterFold::kAlwaysFalse);
}

TEST(CudfIcebergFilterFoldingTest, dateAsDaysSinceEpoch) {
  const auto days = DATE()->toDays("2025-06-05");
  const common::BigintRange filter{days, days, /*nullAllowed=*/false};

  // Iceberg-native encoding.
  EXPECT_EQ(
      fold(filter, DATE(), std::to_string(days)),
      ConstantFilterFold::kAlwaysTrue);
  // Hive-migrated encoding.
  EXPECT_EQ(
      fold(filter, DATE(), "2025-06-05"), ConstantFilterFold::kAlwaysTrue);
  EXPECT_EQ(
      fold(filter, DATE(), "2025-06-06"), ConstantFilterFold::kAlwaysFalse);
}

TEST(CudfIcebergFilterFoldingTest, nullValue) {
  const common::IsNull isNull;
  EXPECT_EQ(
      fold(isNull, BIGINT(), std::nullopt), ConstantFilterFold::kAlwaysTrue);

  const common::BigintRange rejectsNull{5, 10, /*nullAllowed=*/false};
  EXPECT_EQ(
      fold(rejectsNull, BIGINT(), std::nullopt),
      ConstantFilterFold::kAlwaysFalse);

  const common::BigintRange allowsNull{5, 10, /*nullAllowed=*/true};
  EXPECT_EQ(
      fold(allowsNull, BIGINT(), std::nullopt),
      ConstantFilterFold::kAlwaysTrue);
}

TEST(CudfIcebergFilterFoldingTest, timestampMode) {
  const std::string partitionValue = "2020-01-01 12:34:56";
  const auto utcTimestamp =
      util::fromTimestampString(
          StringView(partitionValue), util::TimestampParseMode::kPrestoCast)
          .value();
  auto localTimestamp = utcTimestamp;
  localTimestamp.toGMT(Timestamp::defaultTimezone());

  const common::TimestampRange filter{
      localTimestamp, localTimestamp, /*nullAllowed=*/false};
  EXPECT_EQ(
      fold(
          filter,
          TIMESTAMP(),
          partitionValue,
          /*readTimestampAsLocalTime=*/true),
      ConstantFilterFold::kAlwaysTrue);
  EXPECT_EQ(
      fold(
          filter,
          TIMESTAMP(),
          partitionValue,
          /*readTimestampAsLocalTime=*/false),
      ConstantFilterFold::kAlwaysFalse);
}

TEST(CudfIcebergFilterFoldingTest, unconvertibleValueFails) {
  const common::BigintRange filter{5, 10, /*nullAllowed=*/false};

  EXPECT_THROW(fold(filter, BIGINT(), "apples"), VeloxUserError);
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
