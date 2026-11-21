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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class UrlDecodeTest : public SparkFunctionBaseTest {
 protected:
  std::optional<std::string> urlDecode(
      const std::optional<std::string>& input) {
    return evaluateOnce<std::string>("url_decode(c0)", input);
  }

  std::optional<std::string> urlDecode(
      const std::optional<std::string>& input,
      bool failOnError) {
    return evaluateOnce<std::string>(
        fmt::format("url_decode(c0, {})", failOnError), input);
  }
};

TEST_F(UrlDecodeTest, urlDecode) {
  EXPECT_EQ(
      urlDecode("https%3A%2F%2Fspark.apache.org"), "https://spark.apache.org");
  EXPECT_EQ(urlDecode("a+b%20c"), "a b c");
  EXPECT_EQ(urlDecode("%e2%82%ac%C3%A9"), "€é");
  EXPECT_EQ(urlDecode("%+1%-0"), std::string("\x01\x00", 2));
  EXPECT_EQ(urlDecode(""), "");
  EXPECT_EQ(urlDecode(std::nullopt), std::nullopt);

  // Decoded bytes that are not valid UTF-8 become U+FFFD.
  EXPECT_EQ(urlDecode("%C3"), "�");
  EXPECT_EQ(urlDecode("%FF"), "�");
  EXPECT_EQ(urlDecode("%C3x"), "�x");
  EXPECT_EQ(urlDecode("%C3%28"), "�(");
  EXPECT_EQ(urlDecode("%E2%82"), "�");
  EXPECT_EQ(urlDecode("%F0%9F%98"), "�");
  EXPECT_EQ(urlDecode("%C3%C3%A9"), "�é");
  EXPECT_EQ(urlDecode("%C0%AF"), "��");
  EXPECT_EQ(urlDecode("%E0%80%80"), "���");
  EXPECT_EQ(urlDecode("%ED%A0%80"), "�");
  EXPECT_EQ(urlDecode("%F4%90%80%80"), "����");

  VELOX_ASSERT_USER_THROW(
      urlDecode("abc%"), "Incomplete trailing escape (%) pattern");
  VELOX_ASSERT_USER_THROW(
      urlDecode("%2s"), "Illegal hex characters in escape (%) pattern: 2s");
  VELOX_ASSERT_USER_THROW(
      urlDecode("%-1"),
      "Illegal hex characters in escape (%) pattern - negative value");
}

TEST_F(UrlDecodeTest, failOnError) {
  EXPECT_EQ(
      urlDecode("https%3A%2F%2Fspark.apache.org", false),
      "https://spark.apache.org");
  EXPECT_EQ(urlDecode("a+b%20c", false), "a b c");
  EXPECT_EQ(urlDecode("%e2%82%ac%C3%A9", false), "€é");
  EXPECT_EQ(urlDecode("%+1%-0", false), std::string("\x01\x00", 2));
  EXPECT_EQ(urlDecode("%C3x", false), "�x");
  EXPECT_EQ(urlDecode("", false), "");
  EXPECT_EQ(urlDecode(std::nullopt, false), std::nullopt);

  // Incomplete escapes.
  EXPECT_EQ(urlDecode("%", false), std::nullopt);
  EXPECT_EQ(urlDecode("%2", false), std::nullopt);
  EXPECT_EQ(urlDecode("abc%", false), std::nullopt);
  EXPECT_EQ(urlDecode("%%", false), std::nullopt);
  EXPECT_EQ(urlDecode("%C3%", false), std::nullopt);
  EXPECT_EQ(urlDecode("%C3%A9%", false), std::nullopt);

  // Non-hex digits, including whitespace and negative values.
  EXPECT_EQ(urlDecode("%2s", false), std::nullopt);
  EXPECT_EQ(urlDecode("%G0", false), std::nullopt);
  EXPECT_EQ(urlDecode("%0x", false), std::nullopt);
  EXPECT_EQ(urlDecode("%%%", false), std::nullopt);
  EXPECT_EQ(urlDecode("% 1", false), std::nullopt);
  EXPECT_EQ(urlDecode("%1 ", false), std::nullopt);
  EXPECT_EQ(urlDecode("%-1", false), std::nullopt);
  EXPECT_EQ(urlDecode("%-F", false), std::nullopt);

  // A malformed escape inside a multi-byte character.
  EXPECT_EQ(urlDecode("%C3%G9", false), std::nullopt);

  // With failOnError true, the result matches the one-argument form.
  EXPECT_EQ(urlDecode("a+b%20c", true), "a b c");
  EXPECT_EQ(urlDecode("%C3x", true), "�x");
  EXPECT_EQ(urlDecode(std::nullopt, true), std::nullopt);
  VELOX_ASSERT_USER_THROW(
      urlDecode("abc%", true), "Incomplete trailing escape (%) pattern");
  VELOX_ASSERT_USER_THROW(
      urlDecode("%2s", true),
      "Illegal hex characters in escape (%) pattern: 2s");
  VELOX_ASSERT_USER_THROW(
      urlDecode("%C3%G9", true),
      "Illegal hex characters in escape (%) pattern: G9");
  VELOX_ASSERT_USER_THROW(
      urlDecode("%-1", true),
      "Illegal hex characters in escape (%) pattern - negative value");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
