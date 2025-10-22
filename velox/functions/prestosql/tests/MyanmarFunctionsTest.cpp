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
#include "velox/functions/prestosql/MyanmarFunctions.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox::functions {
namespace {

class MyanmarFunctionsTest : public test::FunctionBaseTest {
 protected:
  void SetUp() override {
    test::FunctionBaseTest::SetUp();
    registerMyanmarFontEncoding("myanmar_font_encoding");
    registerMyanmarNormalizeUnicode("myanmar_normalize_unicode");
  }
};

TEST_F(MyanmarFunctionsTest, myanmarFontEncoding) {
  const auto fontEncoding = [&](std::optional<std::string> value) {
    return evaluateOnce<std::string>("myanmar_font_encoding(c0)", value);
  };

  EXPECT_EQ(std::nullopt, fontEncoding(std::nullopt));
  EXPECT_EQ("unicode", fontEncoding("english string"));
  EXPECT_EQ("zawgyi", fontEncoding("\u1095"));
  EXPECT_EQ(
      "zawgyi", fontEncoding("\u1021\u101E\u1004\u1039\u1038\u1019\u103D"));
  EXPECT_EQ(
      "unicode",
      fontEncoding("\u1000\u103B\u103D\u1014\u103A\u102F\u1015\u103A"));
}

TEST_F(MyanmarFunctionsTest, myanmarNormalizeUnicode) {
  const auto normalize = [&](std::optional<std::string> value) {
    return evaluateOnce<std::string>("myanmar_normalize_unicode(c0)", value);
  };

  EXPECT_EQ(std::nullopt, normalize(std::nullopt));
  EXPECT_EQ("english string", normalize("english string"));
  EXPECT_EQ(
      "\u1021\u101E\u1004\u103A\u1038\u1019\u103E",
      normalize("\u1021\u101E\u1004\u1039\u1038\u1019\u103D"));
  EXPECT_EQ(
      "\u1000\u103B\u103D\u1014\u103A\u102F\u1015\u103A",
      normalize("\u1000\u103B\u103D\u1014\u103A\u102F\u1015\u103A"));
  EXPECT_EQ(
      "\u1000\u103B\u103D\u1014\u103A\u102F\u1015\u103A\n\u1021\u101E\u1004\u103A\u1038\u1019\u103E",
      normalize(
          "\u1000\u103B\u103D\u1014\u103A\u102F\u1015\u103A\n\u1021\u101E\u1004\u1039\u1038\u1019\u103D"));
}

} // namespace
} // namespace facebook::velox::functions
