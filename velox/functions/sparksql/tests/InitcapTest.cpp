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
class InitcapTest : public SparkFunctionBaseTest {
 public:
  static std::vector<std::tuple<std::string, std::string>>
  getInitcapUnicodeTestData() {
    return {
        {"àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ", "Àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ"},
        {"αβγδεζηθικλμνξοπρςστυφχψ", "Αβγδεζηθικλμνξοπρςστυφχψ"},
        {"абвгдежзийклмнопрстуфхцчшщъыьэюя",
         "Абвгдежзийклмнопрстуфхцчшщъыьэюя"},
        {"hello world", "Hello World"},
        {"HELLO WORLD", "Hello World"},
        {"1234", "1234"},
        {"", ""},
        {"élève très-intelligent", "Élève Très-intelligent"},
        {"mañana-por_la_tarde!", "Mañana-por_la_tarde!"},
        {"добро-пожаловать.тест", "Добро-пожаловать.тест"},
        {"çalışkan öğrenci@üniversite.tr", "Çalışkan Öğrenci@üniversite.tr"},
        {"emoji😊test🚀case", "Emoji😊test🚀case"},
        {"тест@пример.рф", "Тест@пример.рф"}};
  }

  static std::vector<std::tuple<std::string, std::string>>
  getInitcapAsciiTestData() {
    return {
        {"abcdefg", "Abcdefg"},
        {"ABCDEFG", "Abcdefg"},
        {"a B c D e F g", "A B C D E F G"},
        {"hello world", "Hello World"},
        {"HELLO WORLD", "Hello World"},
        {"1234", "1234"},
        {"1 2 3 4", "1 2 3 4"},
        {"1 2 3 4a", "1 2 3 4a"},
        {"", ""},
        {"urna.Ut@egetdictumplacerat.edu", "Urna.ut@egetdictumplacerat.edu"},
        {"nibh.enim@egestas.ca", "Nibh.enim@egestas.ca"},
        {"in@Donecat.ca", "In@donecat.ca"},
        {"sodales@blanditviverraDonec.ca", "Sodales@blanditviverradonec.ca"},
        {"sociis.natoque.penatibus@vitae.org",
         "Sociis.natoque.penatibus@vitae.org"},
        {"john_doe-123@example-site.com", "John_doe-123@example-site.com"},
        {"MIXED.case-EMAIL_42@domain.NET", "Mixed.case-email_42@domain.net"},
        {"...weird..case@@", "...weird..case@@"},
        {"user-name+filter@sub.mail.org", "User-name+filter@sub.mail.org"},
        {"CAPS_LOCK@DOMAIN.COM", "Caps_lock@domain.com"},
        {"__init__.py@example.dev", "__init__.py@example.dev"}};
  }

 protected:
  std::optional<std::string> initcap(const std::optional<std::string>& str) {
    return evaluateOnce<std::string>("initcap(c0)", str);
  }
};

TEST_F(InitcapTest, initcapUnicode) {
  for (const auto& [inputStr, expected] : getInitcapUnicodeTestData()) {
    EXPECT_EQ(initcap(inputStr).value(), expected);
  }
}

TEST_F(InitcapTest, initcapAscii) {
  for (const auto& [inputStr, expected] : getInitcapAsciiTestData()) {
    EXPECT_EQ(initcap(inputStr).value(), expected);
  }
}

TEST_F(InitcapTest, initcap) {
  const auto initcap = [&](const std::optional<std::string>& value) {
    return evaluateOnce<std::string>("initcap(c0)", value);
  };
  // Unicode only.
  EXPECT_EQ(
      initcap("àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ"),
      "Àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ");
  EXPECT_EQ(initcap("αβγδεζηθικλμνξοπρςστυφχψ"), "Αβγδεζηθικλμνξοπρςστυφχψ");
  // Mix of ascii and unicode.
  EXPECT_EQ(initcap("αβγδεζ world"), "Αβγδεζ World");
  EXPECT_EQ(initcap("αfoo wβ"), "Αfoo Wβ");
  // Ascii only.
  EXPECT_EQ(initcap("hello world"), "Hello World");
  EXPECT_EQ(initcap("HELLO WORLD"), "Hello World");
  EXPECT_EQ(initcap("1234"), "1234");
  EXPECT_EQ(initcap("a b c d"), "A B C D");
  EXPECT_EQ(initcap("abcd"), "Abcd");
  // Numbers.
  EXPECT_EQ(initcap("123"), "123");
  EXPECT_EQ(initcap("1abc"), "1abc");
  // Edge cases.
  EXPECT_EQ(initcap(""), "");
  EXPECT_EQ(initcap(std::nullopt), std::nullopt);

  // Test with spaces other than whitespace
  EXPECT_EQ(initcap("YQ\tY"), "Yq\ty");
  EXPECT_EQ(initcap("YQ\nY"), "Yq\ny");
}

TEST_F(InitcapTest, fullCaseMapping) {
  const auto initcap = [&](const std::optional<std::string>& str) {
    return evaluateOnce<std::string>("initcap(c0)", str);
  };
  EXPECT_EQ(initcap("\u00DF"), "Ss");
  EXPECT_EQ(initcap("stra\u00DFe"), "Stra\u00DFe");
  EXPECT_EQ(initcap("\uFB01"), "Fi");
  EXPECT_EQ(initcap("\uFB03"), "Ffi");
  EXPECT_EQ(initcap("\uFB00"), "Ff");
  EXPECT_EQ(initcap("\uFB06"), "St");
  EXPECT_EQ(initcap("\u0149"), "\u02BCN");
  EXPECT_EQ(initcap("\u01F0"), "J\u030C");
  EXPECT_EQ(initcap("\u0587"), "\u0535\u0582");
  EXPECT_EQ(initcap("\u01C6"), "\u01C5");
  // Only the first letter of a word uses the title case mapping.
  EXPECT_EQ(initcap("\u00DF \uFB01"), "Ss Fi");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
