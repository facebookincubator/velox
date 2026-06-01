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
//
// Originally authored by Oleksii PELYKH for pcre4j; ported from
// org.pcre4j.regex.translate.ClassRendererTest (Java) under Apache-2.0 by the
// same author for inclusion in Velox.
//
#include "velox/functions/lib/java_pcre2_translator/ClassRenderer.h"

#include "velox/functions/lib/java_pcre2_translator/ClassBodyParser.h"

#include <gtest/gtest.h>

namespace facebook::velox::functions::java_pcre2_translator::test {

namespace {

std::string render(std::string_view classStr) {
  std::size_t pos = 0;
  auto node = ClassBodyParser::parseClass(classStr, pos);
  return ClassRenderer::render(node);
}

} // namespace

TEST(ClassRenderer, simpleLiterals) {
  EXPECT_EQ(render("[abc]"), "[abc]");
}

TEST(ClassRenderer, simpleRange) {
  EXPECT_EQ(render("[a-z]"), "[a-z]");
}

TEST(ClassRenderer, negatedRange) {
  EXPECT_EQ(render("[^a-z]"), "[^a-z]");
}

TEST(ClassRenderer, nestedUnionFlattens) {
  EXPECT_EQ(render("[abc[def]]"), "[abcdef]");
}

TEST(ClassRenderer, negatedNestedFlattens) {
  const auto result = render("[^a-d[0-9]]");
  EXPECT_EQ(result.find("[["), std::string::npos) << result;
  EXPECT_EQ(result.rfind("[^", 0), 0) << result;
}

TEST(ClassRenderer, intersectionLiteralRange) {
  const auto result = render("[a-c&&b-d]");
  EXPECT_NE(result.find("b"), std::string::npos) << result;
  EXPECT_NE(result.find("c"), std::string::npos) << result;
  EXPECT_EQ(result.find("a"), std::string::npos) << result;
  EXPECT_EQ(result.find("d"), std::string::npos) << result;
}

TEST(ClassRenderer, intersectionDisjoint) {
  EXPECT_EQ(render("[a-c&&d-f]"), "[^\\x{0}-\\x{10FFFF}]");
}

TEST(ClassRenderer, wDashHashEscapesDash) {
  const auto result = render("[\\w-#]");
  EXPECT_NE(result.find("\\w"), std::string::npos) << result;
  EXPECT_NE(result.find("\\-"), std::string::npos) << result;
}

TEST(ClassRenderer, intersectionWithKnownProperty) {
  const auto result = render("[\\d&&[0-3]]");
  EXPECT_NE(result.find("0"), std::string::npos) << result;
  EXPECT_NE(result.find("3"), std::string::npos) << result;
  EXPECT_EQ(result.find("&&"), std::string::npos) << result;
}

TEST(ClassRenderer, intersectionWithJdkExpandableProperty) {
  EXPECT_EQ(render("[\\p{L}&&[a-z]]"), "[a-z]");
}

TEST(ClassRenderer, intersectionWithBracketMappedProperty) {
  EXPECT_EQ(render("[\\p{Alpha}&&[a-z]]"), "[a-z]");
}

TEST(ClassRenderer, intersectionWithJavaAlphabeticProperty) {
  EXPECT_EQ(render("[\\p{javaAlphabetic}&&[a-z]]"), "[a-z]");
}

TEST(ClassRenderer, intersectionWithScriptAlias) {
  const auto result = render("[\\p{sc=Grek}&&\\p{L}]");
  EXPECT_EQ(result.find("&&"), std::string::npos) << result;
  EXPECT_EQ(result.find("&"), std::string::npos) << result;
}

TEST(ClassRenderer, pureIntersectionFallbackWithUnknownProperty) {
  const auto result = render("[\\p{UnknownXyz}&&[a-z]]");
  EXPECT_NE(result.find("\\p{UnknownXyz}"), std::string::npos) << result;
  EXPECT_NE(result.find("&&"), std::string::npos) << result;
  EXPECT_TRUE(result.find("a-z") != std::string::npos ||
              (result.find("a") != std::string::npos && result.find("z") != std::string::npos))
      << result;
}

TEST(ClassRenderer, nestedNegatedIntersection) {
  EXPECT_EQ(render("[^[a-c]&&[d-f]]"), "[\\x{0}-\\x{10FFFF}]");
}

TEST(ClassRenderer, negatedIntersectionOfRanges) {
  EXPECT_EQ(render("[^a-c&&b-d]"), "[\\x{0}-ad-\\x{10FFFF}]");
}

TEST(ClassRenderer, propertyLeafPassesThrough) {
  const auto result = render("[\\d\\w]");
  EXPECT_NE(result.find("\\d"), std::string::npos) << result;
  EXPECT_NE(result.find("\\w"), std::string::npos) << result;
}

TEST(ClassRenderer, multipleIntersectionOperands) {
  EXPECT_EQ(render("[a-m&&m-z&&a-c]"), "[^\\x{0}-\\x{10FFFF}]");
}

TEST(ClassRenderer, nestedNegatedWithUnknownPropertyPreservesNegation) {
  const auto result = render("[abc[^\\p{UnknownXyz}]]");
  EXPECT_NE(result.find("[^"), std::string::npos) << result;
  EXPECT_NE(result.find("\\p{UnknownXyz}"), std::string::npos) << result;
}

} // namespace facebook::velox::functions::java_pcre2_translator::test
