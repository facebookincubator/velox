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
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace {

class VarbinaryFunctionsTest : public functions::test::FunctionBaseTest {};

TEST_F(VarbinaryFunctionsTest, strpos) {
  auto strpos = [&](const std::string& haystack,
                    const std::string& needle,
                    std::optional<int64_t> instance = std::nullopt) {
    if (instance.has_value()) {
      return evaluateOnce<int64_t>(
          "strpos(c0, c1, c2)",
          {VARBINARY(), VARBINARY(), BIGINT()},
          std::optional(haystack),
          std::optional(needle),
          instance);
    } else {
      return evaluateOnce<int64_t>(
          "strpos(c0, c1)",
          {VARBINARY(), VARBINARY()},
          std::optional(haystack),
          std::optional(needle));
    }
  };

  // Basic tests
  EXPECT_EQ(strpos("hello", "l"), 3);
  EXPECT_EQ(strpos("hello", "o"), 5);
  EXPECT_EQ(strpos("hello", "x"), 0);
  EXPECT_EQ(strpos("hello", ""), 1);
  EXPECT_EQ(strpos("", ""), 1);
  EXPECT_EQ(strpos("", "a"), 0);

  // Find nth instance
  EXPECT_EQ(strpos("hello", "l", 1), 3);
  EXPECT_EQ(strpos("hello", "l", 2), 4);
  EXPECT_EQ(strpos("hello", "l", 3), 0);

  // Binary data with embedded nulls
  std::string binaryWithNull = std::string(
      "ab\x00"
      "cd",
      5);
  std::string nullPattern = std::string("\x00", 1);
  EXPECT_EQ(strpos(binaryWithNull, nullPattern), 3);
  EXPECT_EQ(strpos(binaryWithNull, "cd"), 4);
  EXPECT_EQ(strpos(binaryWithNull, "ab"), 1);

  // Binary patterns (non-ASCII bytes)
  std::string binaryData = std::string("\xDE\xAD\xBE\xEF\xCA\xFE", 6);
  std::string pattern1 = std::string("\xBE\xEF", 2);
  std::string pattern2 = std::string("\xCA\xFE", 2);
  EXPECT_EQ(strpos(binaryData, pattern1), 3);
  EXPECT_EQ(strpos(binaryData, pattern2), 5);
  EXPECT_EQ(strpos(binaryData, std::string("\xFF", 1)), 0);

  // Multi-byte UTF-8 characters (treated as bytes)
  std::string utf8String = "\u4FE1\u5FF5"; // Two 3-byte Chinese characters
  EXPECT_EQ(strpos(utf8String, "\xE5"), 4); // Byte position, not char position
}

TEST_F(VarbinaryFunctionsTest, strposInvalidInstance) {
  VELOX_ASSERT_THROW(
      evaluateOnce<int64_t>(
          "strpos(c0, c1, c2)",
          {VARBINARY(), VARBINARY(), BIGINT()},
          std::optional<std::string>("hello"),
          std::optional<std::string>("l"),
          std::optional<int64_t>(0)),
      "'instance' must be a positive number");

  VELOX_ASSERT_THROW(
      evaluateOnce<int64_t>(
          "strpos(c0, c1, c2)",
          {VARBINARY(), VARBINARY(), BIGINT()},
          std::optional<std::string>("hello"),
          std::optional<std::string>("l"),
          std::optional<int64_t>(-1)),
      "'instance' must be a positive number");
}

TEST_F(VarbinaryFunctionsTest, strposFlatVector) {
  // Test with flat vectors for better coverage
  auto haystack = makeFlatVector<std::string>(
      {"hello world", "abcabc", "test", ""}, VARBINARY());
  auto needle =
      makeFlatVector<std::string>({"world", "bc", "x", ""}, VARBINARY());

  auto result = evaluate("strpos(c0, c1)", makeRowVector({haystack, needle}));
  auto expected = makeFlatVector<int64_t>({7, 2, 0, 1});
  assertEqualVectors(expected, result);
}

TEST_F(VarbinaryFunctionsTest, strrpos) {
  auto strrpos = [&](const std::string& haystack,
                     const std::string& needle,
                     std::optional<int64_t> instance = std::nullopt) {
    if (instance.has_value()) {
      return evaluateOnce<int64_t>(
          "strrpos(c0, c1, c2)",
          {VARBINARY(), VARBINARY(), BIGINT()},
          std::optional(haystack),
          std::optional(needle),
          instance);
    } else {
      return evaluateOnce<int64_t>(
          "strrpos(c0, c1)",
          {VARBINARY(), VARBINARY()},
          std::optional(haystack),
          std::optional(needle));
    }
  };

  // Basic tests - strrpos finds from the end
  EXPECT_EQ(strrpos("hello", "l"), 4);
  EXPECT_EQ(strrpos("hello", "o"), 5);
  EXPECT_EQ(strrpos("hello", "x"), 0);
  EXPECT_EQ(strrpos("hello", ""), 1);
  EXPECT_EQ(strrpos("", ""), 1);
  EXPECT_EQ(strrpos("", "a"), 0);

  // Find nth instance from end
  EXPECT_EQ(strrpos("hello", "l", 1), 4);
  EXPECT_EQ(strrpos("hello", "l", 2), 3);
  EXPECT_EQ(strrpos("hello", "l", 3), 0);

  // Binary data
  std::string binaryData = std::string("\xAB\xCD\xAB\xEF", 4);
  std::string pattern = std::string("\xAB", 1);
  EXPECT_EQ(strrpos(binaryData, pattern), 3);
  EXPECT_EQ(strrpos(binaryData, pattern, 2), 1);
}

TEST_F(VarbinaryFunctionsTest, contains) {
  auto contains = [&](const std::string& haystack, const std::string& needle) {
    return evaluateOnce<bool>(
        "contains(c0, c1)",
        {VARBINARY(), VARBINARY()},
        std::optional(haystack),
        std::optional(needle));
  };

  // Basic tests
  EXPECT_EQ(contains("hello", "ell"), true);
  EXPECT_EQ(contains("hello", "x"), false);
  EXPECT_EQ(contains("hello", ""), true);
  EXPECT_EQ(contains("", ""), true);
  EXPECT_EQ(contains("", "a"), false);

  // Binary data with embedded nulls
  std::string binaryWithNull = std::string(
      "ab\x00"
      "cd",
      5);
  std::string nullPattern = std::string("\x00", 1);
  EXPECT_EQ(contains(binaryWithNull, nullPattern), true);
  EXPECT_EQ(contains(binaryWithNull, "cd"), true);
  EXPECT_EQ(contains(binaryWithNull, "xyz"), false);

  // Binary patterns (non-ASCII bytes)
  std::string binaryData = std::string("\xDE\xAD\xBE\xEF", 4);
  std::string pattern1 = std::string("\xAD\xBE", 2);
  std::string pattern2 = std::string("\xFF\xFF", 2);
  EXPECT_EQ(contains(binaryData, pattern1), true);
  EXPECT_EQ(contains(binaryData, pattern2), false);
}

TEST_F(VarbinaryFunctionsTest, containsFlatVector) {
  auto haystack = makeFlatVector<std::string>(
      {"hello world", "abcabc", "test", ""}, VARBINARY());
  auto needle =
      makeFlatVector<std::string>({"world", "xyz", "est", ""}, VARBINARY());

  auto result = evaluate("contains(c0, c1)", makeRowVector({haystack, needle}));
  auto expected = makeFlatVector<bool>({true, false, true, true});
  assertEqualVectors(expected, result);
}

TEST_F(VarbinaryFunctionsTest, nullHandling) {
  // Test strpos with null values
  {
    auto haystack = makeNullableFlatVector<std::string>(
        {std::nullopt, std::string("test"), std::nullopt}, VARBINARY());
    auto needle = makeNullableFlatVector<std::string>(
        {std::string("test"), std::nullopt, std::nullopt}, VARBINARY());
    auto result = evaluate("strpos(c0, c1)", makeRowVector({haystack, needle}));
    auto expected = makeNullableFlatVector<int64_t>(
        {std::nullopt, std::nullopt, std::nullopt});
    assertEqualVectors(expected, result);
  }

  // Test strpos with null instance parameter
  {
    auto haystack = makeFlatVector<std::string>({"hello"}, VARBINARY());
    auto needle = makeFlatVector<std::string>({"l"}, VARBINARY());
    auto instance = makeNullableFlatVector<int64_t>({std::nullopt}, BIGINT());
    auto result = evaluate(
        "strpos(c0, c1, c2)", makeRowVector({haystack, needle, instance}));
    auto expected = makeNullableFlatVector<int64_t>({std::nullopt});
    assertEqualVectors(expected, result);
  }

  // Test strrpos with null values
  {
    auto haystack = makeNullableFlatVector<std::string>(
        {std::nullopt, std::string("test")}, VARBINARY());
    auto needle = makeNullableFlatVector<std::string>(
        {std::string("test"), std::nullopt}, VARBINARY());
    auto result =
        evaluate("strrpos(c0, c1)", makeRowVector({haystack, needle}));
    auto expected =
        makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt});
    assertEqualVectors(expected, result);
  }

  // Test contains with null values
  {
    auto haystack = makeNullableFlatVector<std::string>(
        {std::nullopt, std::string("test"), std::nullopt}, VARBINARY());
    auto needle = makeNullableFlatVector<std::string>(
        {std::string("test"), std::nullopt, std::nullopt}, VARBINARY());
    auto result =
        evaluate("contains(c0, c1)", makeRowVector({haystack, needle}));
    auto expected = makeNullableFlatVector<bool>(
        {std::nullopt, std::nullopt, std::nullopt});
    assertEqualVectors(expected, result);
  }
}

} // namespace
