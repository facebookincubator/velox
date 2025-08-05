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
#include "velox/functions/iceberg/Truncate.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/Expressions.h"
#include "velox/functions/iceberg/tests/IcebergFunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions::iceberg;
using namespace facebook::velox::functions::iceberg::test;

namespace {
class TruncateTest : public IcebergFunctionBaseTest {
 public:
  TruncateTest() {
    options_.parseDecimalAsDouble = false;
  }

 protected:
  template <typename T>
  std::optional<T> truncate(
      std::optional<int32_t> length,
      std::optional<T> value) {
    return evaluateOnce<T>("truncate(c0, c1)", length, value);
  }

  void truncateExpr(
      const VectorPtr& expected,
      const std::vector<VectorPtr>& inputs) {
    VELOX_USER_CHECK_EQ(
        inputs.size(),
        2,
        "At least one input vector is needed for arithmetic function test.");
    std::vector<core::TypedExprPtr> inputExprs;
    inputExprs.reserve(inputs.size());
    inputExprs.emplace_back(
        std::make_shared<core::FieldAccessTypedExpr>(inputs[0]->type(), "c0"));
    inputExprs.emplace_back(
        std::make_shared<core::FieldAccessTypedExpr>(inputs[1]->type(), "c1"));
    auto expr = std::make_shared<const core::CallTypedExpr>(
        expected->type(), std::move(inputExprs), "truncate");
    facebook::velox::test::assertEqualVectors(
        expected, evaluate(expr, makeRowVector(inputs)));
  }
};

TEST_F(TruncateTest, tinyInt) {
  EXPECT_EQ(truncate<int8_t>(10, 0), 0);
  EXPECT_EQ(truncate<int8_t>(10, 1), 0);
  EXPECT_EQ(truncate<int8_t>(10, 5), 0);
  EXPECT_EQ(truncate<int8_t>(10, 9), 0);
  EXPECT_EQ(truncate<int8_t>(10, 10), 10);
  EXPECT_EQ(truncate<int8_t>(10, 11), 10);
  EXPECT_EQ(truncate<int8_t>(10, -1), -10);
  EXPECT_EQ(truncate<int8_t>(10, -5), -10);
  EXPECT_EQ(truncate<int8_t>(10, -10), -10);
  EXPECT_EQ(truncate<int8_t>(10, -11), -20);

  // Different widths.
  EXPECT_EQ(truncate<int8_t>(2, -1), -2);

  // Null handling.
  EXPECT_EQ(truncate<int8_t>(10, std::nullopt), std::nullopt);
}

TEST_F(TruncateTest, smallInt) {
  EXPECT_EQ(truncate<int16_t>(10, 0), 0);
  EXPECT_EQ(truncate<int16_t>(10, 1), 0);
  EXPECT_EQ(truncate<int16_t>(10, 5), 0);
  EXPECT_EQ(truncate<int16_t>(10, 9), 0);
  EXPECT_EQ(truncate<int16_t>(10, 10), 10);
  EXPECT_EQ(truncate<int16_t>(10, 11), 10);
  EXPECT_EQ(truncate<int16_t>(10, -1), -10);
  EXPECT_EQ(truncate<int16_t>(10, -5), -10);
  EXPECT_EQ(truncate<int16_t>(10, -10), -10);
  EXPECT_EQ(truncate<int16_t>(10, -11), -20);

  EXPECT_EQ(truncate<int16_t>(2, -1), -2);
  EXPECT_EQ(truncate<int16_t>(10, std::nullopt), std::nullopt);
}

TEST_F(TruncateTest, integer) {
  EXPECT_EQ(truncate<int32_t>(10, 0), 0);
  EXPECT_EQ(truncate<int32_t>(10, 1), 0);
  EXPECT_EQ(truncate<int32_t>(10, 5), 0);
  EXPECT_EQ(truncate<int32_t>(10, 9), 0);
  EXPECT_EQ(truncate<int32_t>(10, 10), 10);
  EXPECT_EQ(truncate<int32_t>(10, 11), 10);
  EXPECT_EQ(truncate<int32_t>(10, -1), -10);
  EXPECT_EQ(truncate<int32_t>(10, -5), -10);
  EXPECT_EQ(truncate<int32_t>(10, -10), -10);
  EXPECT_EQ(truncate<int32_t>(10, -11), -20);

  // Different widths.
  EXPECT_EQ(truncate<int32_t>(2, -1), -2);
  EXPECT_EQ(truncate<int32_t>(300, 1), 0);

  EXPECT_EQ(truncate<int32_t>(10, std::nullopt), std::nullopt);
}

TEST_F(TruncateTest, bigint) {
  EXPECT_EQ(truncate<int64_t>(10, 0), 0);
  EXPECT_EQ(truncate<int64_t>(10, 1), 0);
  EXPECT_EQ(truncate<int64_t>(10, 5), 0);
  EXPECT_EQ(truncate<int64_t>(10, 9), 0);
  EXPECT_EQ(truncate<int64_t>(10, 10), 10);
  EXPECT_EQ(truncate<int64_t>(10, 11), 10);
  EXPECT_EQ(truncate<int64_t>(10, -1), -10);
  EXPECT_EQ(truncate<int64_t>(10, -5), -10);
  EXPECT_EQ(truncate<int64_t>(10, -10), -10);
  EXPECT_EQ(truncate<int64_t>(10, -11), -20);

  EXPECT_EQ(truncate<int64_t>(2, -1), -2);
  EXPECT_EQ(truncate<int64_t>(10, std::nullopt), std::nullopt);
  VELOX_ASSERT_THROW(
      truncate<int64_t>(0, 34), "Invalid truncate width: 0 (must be > 0)");
  VELOX_ASSERT_THROW(
      truncate<int64_t>(-3, 34), "Invalid truncate width: -3 (must be > 0)");
}

TEST_F(TruncateTest, string) {
  // Basic truncation cases.
  EXPECT_EQ(truncate<std::string>(5, "abcdefg"), "abcde");
  EXPECT_EQ(truncate<std::string>(5, "abc"), "abc");
  EXPECT_EQ(truncate<std::string>(5, "abcde"), "abcde");

  // Empty string handling.
  EXPECT_EQ(truncate<std::string>(10, ""), "");

  // Unicode handling (4-byte characters).
  std::string twoFourByteChars = "\U00010000\U00010000"; // Two 𐀀 characters
  EXPECT_EQ(truncate<std::string>(1, twoFourByteChars), "\U00010000");

  // Mixed character sizes.
  EXPECT_EQ(truncate<std::string>(4, "测试raul试测"), "测试ra");

  // Explicit varchar/char types (treated same as string).
  EXPECT_EQ(truncate<std::string>(4, "测试raul试测"), "测试ra");
}

TEST_F(TruncateTest, unicodeBoundaries) {
  // Japanese characters (3-byte UTF-8).
  std::string japanese = "イロハニホヘト";
  EXPECT_EQ(truncate<std::string>(2, japanese), "イロ");
  EXPECT_EQ(truncate<std::string>(3, japanese), "イロハ");
  EXPECT_EQ(truncate<std::string>(7, japanese), japanese);

  // Chinese characters (3-byte UTF-8).
  EXPECT_EQ(truncate<std::string>(1, "测试"), "测");
}

TEST_F(TruncateTest, binary) {
  truncateExpr(
      makeFlatVector<StringView>(
          {"abc\0\0", "abc", "abc", "测试"}, VARBINARY()),
      {makeFlatVector<int32_t>({10, 3, 3, 6}),
       makeFlatVector<StringView>(
           {"abc\0\0", "abcdefg", "abc", "测试_"}, VARBINARY())});
}

TEST_F(TruncateTest, decimal) {
  truncateExpr(
      makeFlatVector<int64_t>({1230, 1230, 0, -10}, DECIMAL(9, 2)),
      {makeFlatVector<int32_t>({10, 10, 10, 10}),
       makeFlatVector<int64_t>({1234, 1230, 5, -5}, DECIMAL(9, 2))});

  truncateExpr(
      makeConstant<int64_t>(12290, 1, DECIMAL(9, 3)),
      {makeConstant<int32_t>(10, 1),
       makeConstant<int64_t>(12299, 1, DECIMAL(9, 3))});

  truncateExpr(
      makeConstant<int64_t>(3, 1, DECIMAL(5, 2)),
      {makeConstant<int32_t>(3, 1),
       makeConstant<int64_t>(5, 1, DECIMAL(5, 2))});

  truncateExpr(
      makeConstant<int64_t>(123453480, 1, DECIMAL(10, 4)),
      {makeConstant<int32_t>(10, 1),
       makeConstant<int64_t>(123453482, 1, DECIMAL(10, 4))});

  truncateExpr(
      makeConstant<int64_t>(-500, 1, DECIMAL(6, 4)),
      {makeConstant<int32_t>(10, 1),
       makeConstant<int64_t>(-500, 1, DECIMAL(6, 4))});

  truncateExpr(
      makeFlatVector<int128_t>({0, -10}, DECIMAL(38, 2)),
      {makeConstant<int32_t>(10, 2),
       makeFlatVector<int128_t>({5, -5}, DECIMAL(38, 2))});
}
} // namespace
