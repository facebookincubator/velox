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

#include "velox/functions/sparksql/SparkErrors.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::functions::sparksql::test {
namespace {

Status checkPositive(int32_t value) {
  SPARK_USER_RETURN(
      spark_error_code::kDivideByZero,
      value <= 0,
      "value must be positive: {}",
      value);
  return Status::OK();
}

Status checkNotZero(int32_t value) {
  SPARK_USER_RETURN_EQ(
      spark_error_code::kDivideByZero, value, 0, "division by zero");
  return Status::OK();
}

TEST(SparkErrorsTest, sparkUserReturn) {
  ASSERT_TRUE(checkPositive(1).ok());

  const auto status = checkPositive(-1);
  ASSERT_TRUE(status.isUserError());
  ASSERT_EQ(status.errorCode(), "SPARK:DIVIDE_BY_ZERO");
  ASSERT_THAT(
      status.message(), testing::HasSubstr("value must be positive: -1"));
  ASSERT_THAT(status.message(), testing::HasSubstr("value <= 0"));
}

TEST(SparkErrorsTest, sparkUserReturnEq) {
  ASSERT_TRUE(checkNotZero(1).ok());

  const auto status = checkNotZero(0);
  ASSERT_TRUE(status.isUserError());
  ASSERT_EQ(status.errorCode(), "SPARK:DIVIDE_BY_ZERO");
  ASSERT_THAT(
      status.message(), testing::HasSubstr("(0 vs. 0) division by zero"));
  ASSERT_THAT(status.message(), testing::HasSubstr("value == 0"));
}

TEST(SparkErrorsTest, skipErrorDetails) {
  // Under TRY, error details are skipped and no error code is attached.
  ScopedThreadSkipErrorDetails skipErrorDetails(true);
  const auto status = checkPositive(-1);
  ASSERT_TRUE(status.isUserError());
  ASSERT_TRUE(status.message().empty());
  ASSERT_TRUE(status.errorCode().empty());
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
