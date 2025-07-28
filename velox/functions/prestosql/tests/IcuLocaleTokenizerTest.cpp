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

#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::functions::test;

class IcuLocaleTokenizerTest : public FunctionBaseTest {
 protected:
  void assertTokens(
      const std::string& text,
      const std::optional<std::string>& locale,
      const std::vector<std::string>& expectedTokens) {
    std::vector<VectorPtr> inputVectors;
    inputVectors.push_back(makeFlatVector<StringView>({StringView(text)}));

    std::string query = "fb_icu_locale_tokenizer(c0)";
    if (locale.has_value()) {
      inputVectors.push_back(
          makeFlatVector<StringView>({StringView(locale.value())}));
      query = "fb_icu_locale_tokenizer(c0, c1)";
    }

    auto rows = makeRowVector(inputVectors);
    auto result = evaluate(query, rows);

    // Get the underlying ArrayVector, regardless of encoding
    auto arrayVector = result->loadedVector()->as<ArrayVector>();
    ASSERT_NE(arrayVector, nullptr);

    if (expectedTokens.empty()) {
      ASSERT_EQ(0, arrayVector->sizeAt(0));
      return;
    }

    auto elements = arrayVector->elements()->asFlatVector<StringView>();
    auto offset = arrayVector->offsetAt(0);
    auto size = arrayVector->sizeAt(0);

    ASSERT_EQ(expectedTokens.size(), size);
    for (size_t i = 0; i < expectedTokens.size(); i++) {
      ASSERT_EQ(expectedTokens[i], elements->valueAt(offset + i).str());
    }
  }

  void assertNullResult(const std::optional<std::string>& locale) {
    std::vector<VectorPtr> inputVectors;
    inputVectors.push_back(makeNullableFlatVector<StringView>({std::nullopt}));

    std::string query = "fb_icu_locale_tokenizer(c0)";
    if (locale.has_value()) {
      inputVectors.push_back(
          makeFlatVector<StringView>({StringView(locale.value())}));
      query = "fb_icu_locale_tokenizer(c0, c1)";
    }

    auto rows = makeRowVector(inputVectors);
    auto result = evaluate(query, rows);

    // For null inputs, the result should be null
    ASSERT_TRUE(result->isNullAt(0));
  }
};

TEST_F(IcuLocaleTokenizerTest, testFbIcuLocalTokenizer) {
  // Basic test case
  assertTokens(
      "Hello world! How are you?",
      std::nullopt,
      {"Hello", "world", "How", "are", "you"});

  // Tests matching Java implementation
  assertTokens("aaa aaa", "en_US", {"aaa", "aaa"});
  assertTokens("aaa aaa", "en", {"aaa", "aaa"});
  assertTokens("aaa aaa", std::nullopt, {"aaa", "aaa"});
  assertTokens("aaa, aaa!", "en_US", {"aaa", "aaa"});
  assertTokens("Anno 2013", "de", {"Anno", "2013"});
  assertNullResult("en_US");
  assertTokens("", "en_US", {});

  // Thai text
  assertTokens(
      "\u0e14\u0e32\u0e27\u0e24\u0e01\u0e29\u0e4c\u0e16\u0e37\u0e2d\u0e01\u0e33\u0e40\u0e19\u0e34\u0e14\u0e02\u0e36\u0e49\u0e19\u0e08\u0e32\u0e01\u0e40\u0e21\u0e06\u0e42\u0e21\u0e40\u0e25\u0e01\u0e38\u0e25\u0e17\u0e35\u0e48\u0e22\u0e38\u0e1a\u0e15\u0e31\u0e27\u0e42\u0e14\u0e22\u0e21\u0e35\u0e44\u0e2e\u0e42\u0e14\u0e23\u0e40\u0e08\u0e19\u0e40\u0e1b\u0e47\u0e19\u0e2a\u0e48\u0e27\u0e19\u0e1b\u0e23\u0e30\u0e01\u0e2d\u0e1a\u0e2b\u0e25\u0e31\u0e01",
      "th_TH",
      {"\u0e14\u0e32\u0e27\u0e24\u0e01\u0e29\u0e4c",
       "\u0e16\u0e37\u0e2d",
       "\u0e01\u0e33\u0e40\u0e19\u0e34\u0e14",
       "\u0e02\u0e36\u0e49\u0e19",
       "\u0e08\u0e32\u0e01",
       "\u0e40\u0e21\u0e06",
       "\u0e42\u0e21\u0e40\u0e25\u0e01\u0e38\u0e25",
       "\u0e17\u0e35\u0e48",
       "\u0e22\u0e38\u0e1a",
       "\u0e15\u0e31\u0e27",
       "\u0e42\u0e14\u0e22",
       "\u0e21\u0e35",
       "\u0e44\u0e2e\u0e42\u0e14\u0e23\u0e40\u0e08\u0e19",
       "\u0e40\u0e1b\u0e47\u0e19",
       "\u0e2a\u0e48\u0e27\u0e19",
       "\u0e1b\u0e23\u0e30\u0e01\u0e2d\u0e1a",
       "\u0e2b\u0e25\u0e31\u0e01"});

  // Chinese text
  assertTokens(
      "\u6f22\u8a9e\u662f\u8054\u5408\u56fd\u7684\u516d\u79cd\u6b63\u5f0f\u8a9e\u8a00\u548c\u5de5\u4f5c\u8bed\u8a00\u4e4b\u4e00\uff0c",
      "zh",
      {"\u6f22\u8a9e",
       "\u662f",
       "\u8054\u5408",
       "\u56fd",
       "\u7684",
       "\u516d\u79cd",
       "\u6b63\u5f0f",
       "\u8a9e\u8a00",
       "\u548c",
       "\u5de5\u4f5c",
       "\u8bed\u8a00",
       "\u4e4b\u4e00"});

  // Korean text
  assertTokens(
      "\ub300\ud55c\ubbfc\uad6d\uc5d0\uc11c\ub294 \uc0c1\ud669\uc5d0 \ub530\ub77c \ub2e4\uc591\ud55c \uc131\uaca9\uc758 \ubb38\uccb4\ub098 \uad6c\uc5b4\uccb4\ub97c \ud65c\uc6a9\ud558\uace0,",
      "ko_KO",
      {"\ub300\ud55c\ubbfc\uad6d\uc5d0\uc11c\ub294",
       "\uc0c1\ud669\uc5d0",
       "\ub530\ub77c",
       "\ub2e4\uc591\ud55c",
       "\uc131\uaca9\uc758",
       "\ubb38\uccb4\ub098",
       "\uad6c\uc5b4\uccb4\ub97c",
       "\ud65c\uc6a9\ud558\uace0"});
}
