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

#include "velox/connectors/hive/iceberg/ColumnTransform.h"
#include "velox/connectors/hive/iceberg/Transforms.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive::iceberg;
using namespace facebook::velox::test;

namespace facebook::velox::connector::hive::iceberg::test {

class ColumnTransformTest : public IcebergTestBase {};

TEST_F(ColumnTransformTest, testConstructor) {
  auto transform =
      std::make_shared<IdentityTransform<int32_t>>(INTEGER(), opPool_.get());
  ColumnTransform columnTransform("test_column", transform);

  EXPECT_EQ(columnTransform.columnName(), "test_column");
  EXPECT_EQ(columnTransform.transformName(), "identity");
  EXPECT_EQ(columnTransform.resultType(), INTEGER());
}

TEST_F(ColumnTransformTest, testTransformName) {
  auto identityTransform =
      std::make_shared<IdentityTransform<int32_t>>(INTEGER(), opPool_.get());
  ColumnTransform identityColumnTransform("col1", identityTransform);
  EXPECT_EQ(identityColumnTransform.transformName(), "identity");

  auto bucketTransform =
      std::make_shared<BucketTransform<int32_t>>(16, INTEGER(), opPool_.get());
  ColumnTransform bucketColumnTransform("col2", bucketTransform);
  EXPECT_EQ(bucketColumnTransform.transformName(), "bucket");

  auto truncateTransform = std::make_shared<TruncateTransform<int32_t>>(
      10, INTEGER(), opPool_.get());
  ColumnTransform truncateColumnTransform("col3", truncateTransform);
  EXPECT_EQ(truncateColumnTransform.transformName(), "trunc");

  auto yearTransform = std::make_shared<TemporalTransform<int32_t>>(
      INTEGER(), TransformType::kYear, opPool_.get(), [](int32_t v) {
        return v;
      });
  ColumnTransform yearColumnTransform("col4", yearTransform);
  EXPECT_EQ(yearColumnTransform.transformName(), "year");

  auto monthTransform = std::make_shared<TemporalTransform<int32_t>>(
      INTEGER(), TransformType::kMonth, opPool_.get(), [](int32_t v) {
        return v;
      });
  ColumnTransform monthColumnTransform("col5", monthTransform);
  EXPECT_EQ(monthColumnTransform.transformName(), "month");

  auto dayTransform = std::make_shared<TemporalTransform<int32_t>>(
      INTEGER(), TransformType::kDay, opPool_.get(), [](int32_t v) {
        return v;
      });
  ColumnTransform dayColumnTransform("col6", dayTransform);
  EXPECT_EQ(dayColumnTransform.transformName(), "day");

  auto hourTransform = std::make_shared<TemporalTransform<Timestamp>>(
      INTEGER(), TransformType::kHour, opPool_.get(), [](Timestamp v) {
        return v.getSeconds() / 3600;
      });
  ColumnTransform hourColumnTransform("col7", hourTransform);
  EXPECT_EQ(hourColumnTransform.transformName(), "hour");
}

TEST_F(ColumnTransformTest, testColumnName) {
  auto transform =
      std::make_shared<IdentityTransform<int32_t>>(INTEGER(), opPool_.get());

  ColumnTransform simpleColumnTransform("simple_column", transform);
  EXPECT_EQ(simpleColumnTransform.columnName(), "simple_column");

  ColumnTransform nestedColumnTransform("struct.nested_column", transform);
  EXPECT_EQ(nestedColumnTransform.columnName(), "struct.nested_column");

  ColumnTransform deeplyNestedColumnTransform(
      "struct.nested.deeply_nested", transform);
  EXPECT_EQ(
      deeplyNestedColumnTransform.columnName(), "struct.nested.deeply_nested");
}

TEST_F(ColumnTransformTest, testResultType) {
  auto intTransform =
      std::make_shared<IdentityTransform<int32_t>>(INTEGER(), opPool_.get());
  ColumnTransform intColumnTransform("col_int", intTransform);
  EXPECT_EQ(intColumnTransform.resultType(), INTEGER());

  auto bigintTransform =
      std::make_shared<IdentityTransform<int64_t>>(BIGINT(), opPool_.get());
  ColumnTransform bigintColumnTransform("col_bigint", bigintTransform);
  EXPECT_EQ(bigintColumnTransform.resultType(), BIGINT());

  auto varcharTransform =
      std::make_shared<IdentityTransform<StringView>>(VARCHAR(), opPool_.get());
  ColumnTransform varcharColumnTransform("col_varchar", varcharTransform);
  EXPECT_EQ(varcharColumnTransform.resultType(), VARCHAR());

  auto bucketTransform = std::make_shared<BucketTransform<StringView>>(
      16, VARCHAR(), opPool_.get());
  ColumnTransform bucketColumnTransform("col_bucket", bucketTransform);
  EXPECT_EQ(bucketColumnTransform.resultType(), INTEGER());

  auto yearTransform = std::make_shared<TemporalTransform<int32_t>>(
      DATE(), TransformType::kYear, opPool_.get(), [](int32_t v) { return v; });
  ColumnTransform yearColumnTransform("col_year", yearTransform);
  EXPECT_EQ(yearColumnTransform.resultType(), INTEGER());
}

TEST_F(ColumnTransformTest, testTransformSimpleColumn) {
  auto intVector = makeFlatVector<int32_t>({1, 2, 3, 4, 5});
  auto rowVector = makeRowVector({"col_int"}, {intVector});

  auto transform =
      std::make_shared<IdentityTransform<int32_t>>(INTEGER(), opPool_.get());
  ColumnTransform columnTransform("col_int", transform);

  auto result = columnTransform.transform(rowVector);

  ASSERT_EQ(result->size(), 5);
  ASSERT_EQ(result->type(), INTEGER());

  auto resultVector = result->as<FlatVector<int32_t>>();
  EXPECT_EQ(resultVector->valueAt(0), 1);
  EXPECT_EQ(resultVector->valueAt(1), 2);
  EXPECT_EQ(resultVector->valueAt(2), 3);
  EXPECT_EQ(resultVector->valueAt(3), 4);
  EXPECT_EQ(resultVector->valueAt(4), 5);
}

} // namespace facebook::velox::connector::hive::iceberg::test
