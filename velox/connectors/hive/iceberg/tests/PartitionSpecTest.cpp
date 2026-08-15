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
  auto makeSpec = [](const TypePtr& type) {
    std::vector<IcebergPartitionSpec::Field> fields = {
        {"c0", type, TransformType::kIdentity, std::nullopt},
    };
    return std::make_shared<const IcebergPartitionSpec>(1, fields);
  };

  VELOX_ASSERT_USER_THROW(
      makeSpec(ROW({{"a", INTEGER()}})),
      "Type is not supported as a partition column: ROW");
  VELOX_ASSERT_USER_THROW(
      makeSpec(ARRAY(INTEGER())),
      "Type is not supported as a partition column: ARRAY");
  VELOX_ASSERT_USER_THROW(
      makeSpec(MAP(VARCHAR(), INTEGER())),
      "Type is not supported as a partition column: MAP");
  VELOX_ASSERT_USER_THROW(
      makeSpec(TIMESTAMP_WITH_TIME_ZONE()),
      "Type is not supported as a partition column: TIMESTAMP WITH TIME ZONE");
}

TEST(PartitionSpecTest, invalidMultipleTransforms) {
  {
    std::vector<IcebergPartitionSpec::Field> fields = {
        {"c0", VARCHAR(), TransformType::kIdentity, std::nullopt},
        {"c0", VARCHAR(), TransformType::kIdentity, std::nullopt},
    };
    VELOX_ASSERT_USER_THROW(
        std::make_shared<const IcebergPartitionSpec>(1, fields),
        "Column: 'c0', Category: Identity, Transforms: [identity, identity]");
  }

  {
    std::vector<IcebergPartitionSpec::Field> fields = {
        {"c0", VARCHAR(), TransformType::kBucket, 16},
        {"c0", VARCHAR(), TransformType::kBucket, 32},
    };
    VELOX_ASSERT_USER_THROW(
        std::make_shared<const IcebergPartitionSpec>(1, fields),
        "Column: 'c0', Category: Bucket, Transforms: [bucket, bucket]");
  }

  {
    std::vector<IcebergPartitionSpec::Field> fields = {
        {"c0", VARCHAR(), TransformType::kTruncate, 2},
        {"c0", VARCHAR(), TransformType::kTruncate, 5},
    };
    VELOX_ASSERT_USER_THROW(
        std::make_shared<const IcebergPartitionSpec>(1, fields),
        "Column: 'c0', Category: Truncate, Transforms: [trunc, trunc]");
  }

  {
    std::vector<IcebergPartitionSpec::Field> fields4 = {
        {"c0", TIMESTAMP(), TransformType::kYear, std::nullopt},
        {"c0", TIMESTAMP(), TransformType::kMonth, std::nullopt},
        {"c0", TIMESTAMP(), TransformType::kDay, std::nullopt},
        {"c0", TIMESTAMP(), TransformType::kHour, std::nullopt},
    };
    VELOX_ASSERT_USER_THROW(
        std::make_shared<const IcebergPartitionSpec>(1, fields4),
        "Column: 'c0', Category: Temporal, Transforms: [year, month, day, hour]");
  }
}

TEST(PartitionSpecTest, invalidMultipleTransformsMultipleColumns) {
  std::vector<IcebergPartitionSpec::Field> fields = {
      {"c0", DATE(), TransformType::kYear, std::nullopt},
      {"c0", DATE(), TransformType::kMonth, std::nullopt},
      {"c1", VARCHAR(), TransformType::kBucket, 16},
      {"c1", VARCHAR(), TransformType::kBucket, 32},
  };
  // order may vary due to map iteration.
  VELOX_ASSERT_USER_THROW(
      std::make_shared<const IcebergPartitionSpec>(1, fields),
      "Column: 'c0', Category: Temporal, Transforms: [year, month]");
  VELOX_ASSERT_USER_THROW(
      std::make_shared<const IcebergPartitionSpec>(1, fields),
      "Column: 'c1', Category: Bucket, Transforms: [bucket, bucket]");
}

TEST(PartitionSpecTest, validMultipleTransforms) {
  {
    std::vector<IcebergPartitionSpec::Field> fields = {
        {"c0", VARCHAR(), TransformType::kIdentity, std::nullopt},
        {"c0", VARCHAR(), TransformType::kBucket, 16},
        {"c0", VARCHAR(), TransformType::kTruncate, 10},
    };
    auto spec = std::make_shared<const IcebergPartitionSpec>(1, fields);
    EXPECT_EQ(spec->fields.size(), 3);
  }

  {
    std::vector<IcebergPartitionSpec::Field> fields = {
        {"c0", DATE(), TransformType::kYear, std::nullopt},
        {"c0", DATE(), TransformType::kBucket, 16},
        {"c0", DATE(), TransformType::kIdentity, std::nullopt},
    };
    auto spec = std::make_shared<const IcebergPartitionSpec>(1, fields);
    EXPECT_EQ(spec->fields.size(), 3);
  }
}

} // namespace

} // namespace facebook::velox::connector::hive::iceberg
