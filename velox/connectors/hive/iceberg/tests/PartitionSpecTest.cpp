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
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

class PartitionSpecTest : public ::testing::Test {};

TEST_F(PartitionSpecTest, names) {
  EXPECT_EQ("identity", TransformTypeName::toName(TransformType::kIdentity));
  EXPECT_EQ("hour", TransformTypeName::toName(TransformType::kHour));
  EXPECT_EQ("day", TransformTypeName::toName(TransformType::kDay));
  EXPECT_EQ("month", TransformTypeName::toName(TransformType::kMonth));
  EXPECT_EQ("year", TransformTypeName::toName(TransformType::kYear));
  EXPECT_EQ("bucket", TransformTypeName::toName(TransformType::kBucket));
  EXPECT_EQ("trunc", TransformTypeName::toName(TransformType::kTruncate));
}

TEST_F(PartitionSpecTest, basic) {
  std::vector<IcebergPartitionSpec::Field> fields = {
      {"user_id", BIGINT(), TransformType::kIdentity, std::nullopt},
      {"created_date", DATE(), TransformType::kDay, std::nullopt}};

  IcebergPartitionSpec spec(1, fields);

  EXPECT_EQ(1, spec.specId);
  EXPECT_EQ(2, spec.fields.size());

  EXPECT_EQ("user_id", spec.fields[0].name);
  EXPECT_EQ(BIGINT(), spec.fields[0].type);
  EXPECT_EQ(TransformType::kIdentity, spec.fields[0].transformType);
  EXPECT_FALSE(spec.fields[0].parameter.has_value());

  EXPECT_EQ("created_date", spec.fields[1].name);
  EXPECT_EQ(DATE(), spec.fields[1].type);
  EXPECT_EQ(TransformType::kDay, spec.fields[1].transformType);
  EXPECT_FALSE(spec.fields[1].parameter.has_value());
}

TEST_F(PartitionSpecTest, withParameters) {
  std::vector<IcebergPartitionSpec::Field> fields = {
      {"category", VARCHAR(), TransformType::kBucket, 16},
      {"description", VARCHAR(), TransformType::kTruncate, 100}};

  IcebergPartitionSpec spec(2, fields);

  EXPECT_EQ(2, spec.specId);
  EXPECT_EQ(2, spec.fields.size());

  EXPECT_EQ("category", spec.fields[0].name);
  EXPECT_EQ(VARCHAR(), spec.fields[0].type);
  EXPECT_EQ(TransformType::kBucket, spec.fields[0].transformType);
  EXPECT_TRUE(spec.fields[0].parameter.has_value());
  EXPECT_EQ(16, spec.fields[0].parameter.value());

  EXPECT_EQ("description", spec.fields[1].name);
  EXPECT_EQ(VARCHAR(), spec.fields[1].type);
  EXPECT_EQ(TransformType::kTruncate, spec.fields[1].transformType);
  EXPECT_TRUE(spec.fields[1].parameter.has_value());
  EXPECT_EQ(100, spec.fields[1].parameter.value());
}

TEST_F(PartitionSpecTest, empty) {
  std::vector<IcebergPartitionSpec::Field> fields;
  IcebergPartitionSpec spec(0, fields);

  EXPECT_EQ(0, spec.specId);
  EXPECT_TRUE(spec.fields.empty());
}

TEST_F(PartitionSpecTest, temporal) {
  std::vector<IcebergPartitionSpec::Field> fields = {
      {"event_time", TIMESTAMP(), TransformType::kHour, std::nullopt},
      {"created_at", TIMESTAMP(), TransformType::kMonth, std::nullopt},
      {"birth_year", DATE(), TransformType::kYear, std::nullopt}};

  IcebergPartitionSpec spec(3, fields);

  EXPECT_EQ(3, spec.specId);
  EXPECT_EQ(3, spec.fields.size());

  EXPECT_EQ(TransformType::kHour, spec.fields[0].transformType);
  EXPECT_EQ(TransformType::kMonth, spec.fields[1].transformType);
  EXPECT_EQ(TransformType::kYear, spec.fields[2].transformType);
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
