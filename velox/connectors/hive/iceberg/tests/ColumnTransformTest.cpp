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

#include "velox/connectors/hive/iceberg/Transforms.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive::iceberg;
using namespace facebook::velox::test;

namespace facebook::velox::connector::hive::iceberg::test {

class ColumnTransformTest : public IcebergTestBase {};

TEST_F(ColumnTransformTest, testConstructor) {
  auto transform = std::make_shared<IdentityTransform<int32_t>>(
      INTEGER(), "test_column", opPool_.get());

  EXPECT_EQ(transform->sourceColumnName(), "test_column");
  EXPECT_EQ(transform->name(), "identity");
  EXPECT_EQ(transform->resultType(), INTEGER());
}

TEST_F(ColumnTransformTest, testTransformName) {
  auto identityTransform = std::make_shared<IdentityTransform<int32_t>>(
      INTEGER(), "col1", opPool_.get());
  EXPECT_EQ(identityTransform->name(), "identity");

  auto bucketTransform = std::make_shared<BucketTransform<int32_t>>(
      16, INTEGER(), "col2", opPool_.get());
  EXPECT_EQ(bucketTransform->name(), "bucket");

  auto truncateTransform = std::make_shared<TruncateTransform<int32_t>>(
      10, INTEGER(), "col3", opPool_.get());
  EXPECT_EQ(truncateTransform->name(), "trunc");

  auto yearTransform = std::make_shared<TemporalTransform<int32_t>>(
      INTEGER(), TransformType::kYear, "col4", opPool_.get(), [](int32_t v) {
        return v;
      });
  EXPECT_EQ(yearTransform->name(), "year");

  auto monthTransform = std::make_shared<TemporalTransform<int32_t>>(
      INTEGER(), TransformType::kMonth, "col5", opPool_.get(), [](int32_t v) {
        return v;
      });
  EXPECT_EQ(monthTransform->name(), "month");

  auto dayTransform = std::make_shared<TemporalTransform<int32_t>>(
      INTEGER(), TransformType::kDay, "col6", opPool_.get(), [](int32_t v) {
        return v;
      });
  EXPECT_EQ(dayTransform->name(), "day");

  auto hourTransform = std::make_shared<TemporalTransform<Timestamp>>(
      INTEGER(), TransformType::kHour, "col7", opPool_.get(), [](Timestamp v) {
        return v.getSeconds() / 3600;
      });
  EXPECT_EQ(hourTransform->name(), "hour");
}

TEST_F(ColumnTransformTest, testResultType) {
  auto intTransform = std::make_shared<IdentityTransform<int32_t>>(
      INTEGER(), "col_int", opPool_.get());
  EXPECT_EQ(intTransform->resultType(), INTEGER());

  auto bigintTransform = std::make_shared<IdentityTransform<int64_t>>(
      BIGINT(), "col_bigint", opPool_.get());
  EXPECT_EQ(bigintTransform->resultType(), BIGINT());

  auto varcharTransform = std::make_shared<IdentityTransform<StringView>>(
      VARCHAR(), "col_varchar", opPool_.get());
  EXPECT_EQ(varcharTransform->resultType(), VARCHAR());

  auto bucketTransform = std::make_shared<BucketTransform<StringView>>(
      16, VARCHAR(), "col_bucket", opPool_.get());
  EXPECT_EQ(bucketTransform->resultType(), INTEGER());

  auto yearTransform = std::make_shared<TemporalTransform<int32_t>>(
      DATE(), TransformType::kYear, "col_year", opPool_.get(), [](int32_t v) {
        return v;
      });
  EXPECT_EQ(yearTransform->resultType(), INTEGER());
}

TEST_F(ColumnTransformTest, testTransformSimpleColumn) {
  auto intVector = makeFlatVector<int32_t>({1, 2, 3, 4, 5});
  auto rowVector = makeRowVector({"col_int"}, {intVector});

  auto transform = std::make_shared<IdentityTransform<int32_t>>(
      INTEGER(), "col_int", opPool_.get());

  auto result = transform->transform(rowVector, 0);

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
