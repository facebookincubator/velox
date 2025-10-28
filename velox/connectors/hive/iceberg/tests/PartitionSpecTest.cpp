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

#include "velox/connectors/hive/iceberg/PartitionSpec.h"

#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

TEST(PartitionSpecTest, invalidColumnType) {
  std::vector<IcebergPartitionSpec::Field> fields1 = {
      {"col_row",
       ROW({{"a", INTEGER()}}),
       TransformType::kIdentity,
       std::nullopt},
  };
  VELOX_ASSERT_USER_THROW(
      std::make_shared<const IcebergPartitionSpec>(1, fields1),
      "not supported as a partition column");

  std::vector<IcebergPartitionSpec::Field> fields2 = {
      {"col_array", ARRAY(INTEGER()), TransformType::kIdentity, std::nullopt},
  };
  VELOX_ASSERT_USER_THROW(
      std::make_shared<const IcebergPartitionSpec>(1, fields2),
      "not supported as a partition column");

  std::vector<IcebergPartitionSpec::Field> fields3 = {
      {"col_map",
       MAP(VARCHAR(), INTEGER()),
       TransformType::kIdentity,
       std::nullopt},
  };
  VELOX_ASSERT_USER_THROW(
      std::make_shared<const IcebergPartitionSpec>(1, fields3),
      "not supported as a partition column");

  std::vector<IcebergPartitionSpec::Field> fields4 = {
      {"c0",
       TIMESTAMP_WITH_TIME_ZONE(),
       TransformType::kIdentity,
       std::nullopt},
  };
  VELOX_ASSERT_USER_THROW(
      std::make_shared<const IcebergPartitionSpec>(1, fields4),
      "not supported as a partition column");
}

TEST(PartitionSpecTest, invalidMultipleTransforms) {
  std::vector<IcebergPartitionSpec::Field> fields1 = {
      {"c0", VARCHAR(), TransformType::kIdentity, std::nullopt},
      {"c0", VARCHAR(), TransformType::kIdentity, std::nullopt},
  };
  VELOX_ASSERT_USER_THROW(
      std::make_shared<const IcebergPartitionSpec>(1, fields1),
      "Multiple transforms of the same category on a column are not allowed. "
      "Each transform category can appear at most once per column. "
      "Column: 'c0', Category: Identity.");

  std::vector<IcebergPartitionSpec::Field> fields2 = {
      {"c0", VARCHAR(), TransformType::kBucket, 16},
      {"c0", VARCHAR(), TransformType::kBucket, 32},
  };
  VELOX_ASSERT_USER_THROW(
      std::make_shared<const IcebergPartitionSpec>(1, fields2),
      "Multiple transforms of the same category on a column are not allowed. "
      "Each transform category can appear at most once per column. "
      "Column: 'c0', Category: Bucket.");

  std::vector<IcebergPartitionSpec::Field> fields3 = {
      {"c0", VARCHAR(), TransformType::kTruncate, 2},
      {"c0", VARCHAR(), TransformType::kTruncate, 5},
  };
  VELOX_ASSERT_USER_THROW(
      std::make_shared<const IcebergPartitionSpec>(1, fields3),
      "Multiple transforms of the same category on a column are not allowed. "
      "Each transform category can appear at most once per column. "
      "Column: 'c0', Category: Truncate.");

  std::vector<IcebergPartitionSpec::Field> fields4 = {
      {"c0", DATE(), TransformType::kYear, std::nullopt},
      {"c0", DATE(), TransformType::kMonth, std::nullopt},
  };
  VELOX_ASSERT_USER_THROW(
      std::make_shared<const IcebergPartitionSpec>(1, fields4),
      "Multiple transforms of the same category on a column are not allowed. "
      "Each transform category can appear at most once per column. "
      "Column: 'c0', Category: Temporal (Year/Month/Day/Hour).");
}

TEST(PartitionSpecTest, validMultipleTransforms) {
  std::vector<IcebergPartitionSpec::Field> fields1 = {
      {"c0", VARCHAR(), TransformType::kIdentity, std::nullopt},
      {"c0", VARCHAR(), TransformType::kBucket, 16},
      {"c0", VARCHAR(), TransformType::kTruncate, 10},
  };
  auto spec = std::make_shared<const IcebergPartitionSpec>(1, fields1);
  EXPECT_EQ(spec->fields.size(), 3);

  std::vector<IcebergPartitionSpec::Field> fields2 = {
      {"c0", DATE(), TransformType::kYear, std::nullopt},
      {"c0", DATE(), TransformType::kBucket, 16},
      {"c0", DATE(), TransformType::kIdentity, std::nullopt},
  };
  spec = std::make_shared<const IcebergPartitionSpec>(1, fields2);
  EXPECT_EQ(spec->fields.size(), 3);
}

} // namespace

} // namespace facebook::velox::connector::hive::iceberg
