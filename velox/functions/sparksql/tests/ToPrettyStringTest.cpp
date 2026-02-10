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
#include <string>

#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
class ToPrettyStringTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  std::optional<std::string> toPrettyString(std::optional<T> arg) {
    return evaluateOnce<std::string>("to_pretty_string(c0)", arg);
  }

  template <TypeKind KIND>
  void testDecimalExpr(
      const VectorPtr& expected,
      const std::string& expression,
      const std::vector<VectorPtr>& input) {
    using EvalType = typename velox::TypeTraits<KIND>::NativeType;
    auto result =
        evaluate<SimpleVector<EvalType>>(expression, makeRowVector(input));
    velox::test::assertEqualVectors(expected, result);
  }

  void setQueryTimeZone(const std::string& timeZone) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kSessionTimezone, timeZone},
        {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
    });
  }

  const std::string kNull = "NULL";
};

TEST_F(ToPrettyStringTest, toPrettyStringTest) {
  EXPECT_EQ(toPrettyString<int8_t>(1), "1");
  EXPECT_EQ(toPrettyString<int16_t>(1), "1");
  EXPECT_EQ(toPrettyString<int32_t>(1), "1");
  EXPECT_EQ(toPrettyString<int64_t>(1), "1");
  EXPECT_EQ(toPrettyString<float>(1.0), "1.0");
  EXPECT_EQ(toPrettyString<double>(1.0), "1.0");
  EXPECT_EQ(toPrettyString<bool>(true), "true");
  EXPECT_EQ(toPrettyString<StringView>("str"), "str");
  EXPECT_EQ(
      toPrettyString<StringView>("str_is_not_inline"), "str_is_not_inline");
  EXPECT_EQ(toPrettyString<int8_t>(std::nullopt), kNull);
}

TEST_F(ToPrettyStringTest, date) {
  EXPECT_EQ(
      evaluateOnce<std::string>(
          "to_pretty_string(c0)", DATE(), std::optional<int32_t>(18262)),
      "2020-01-01");
}

TEST_F(ToPrettyStringTest, binary) {
  auto toPrettyStringBinary = [&](const std::optional<std::string>& input) {
    return evaluateOnce<std::string>("to_pretty_string(c0)", VARBINARY(), input)
        .value();
  };

  EXPECT_EQ(toPrettyStringBinary("abcdef"), "[61 62 63 64 65 66]");

  EXPECT_EQ(toPrettyStringBinary("ABC."), "[41 42 43 2E]");
}

TEST_F(ToPrettyStringTest, timestamp) {
  EXPECT_EQ(
      toPrettyString<Timestamp>(Timestamp(946729316, 123)),
      "2000-01-01 12:21:56");

  setQueryTimeZone("America/Los_Angeles");
  EXPECT_EQ(
      toPrettyString<Timestamp>(Timestamp(946729316, 123)),
      "2000-01-01 04:21:56");

  EXPECT_EQ(toPrettyString<Timestamp>(std::nullopt), kNull);
}

TEST_F(ToPrettyStringTest, decimal) {
  testDecimalExpr<TypeKind::VARCHAR>(
      {makeFlatVector<StringView>({"0.123", "0.552", "0.000"})},
      "to_pretty_string(c0)",
      {makeFlatVector<int64_t>({123, 552, 0}, DECIMAL(3, 3))});

  testDecimalExpr<TypeKind::VARCHAR>(
      {makeFlatVector<StringView>(
          {"12345678901234.56789",
           "55555555555555.55555",
           "0.00000",
           StringView(kNull)})},
      "to_pretty_string(c0)",
      {makeNullableFlatVector<int128_t>(
          {1234567890123456789, 5555555555555555555, 0, std::nullopt},
          DECIMAL(19, 5))});
}

} // namespace facebook::velox::functions::sparksql::test
