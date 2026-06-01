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
// org.pcre4j.regex.translate.EvaluatorTest (Java) under Apache-2.0 by the
// same author for inclusion in Velox.
//
#include "velox/functions/lib/java_pcre2_translator/Evaluator.h"

#include "velox/functions/lib/java_pcre2_translator/ClassBodyParser.h"
#include "velox/functions/lib/java_pcre2_translator/EvaluationFailedException.h"

#include <gtest/gtest.h>

namespace facebook::velox::functions::java_pcre2_translator::test {

class EvaluatorPosixShorthand
    : public testing::TestWithParam<std::tuple<std::string, int>> {};

TEST_P(EvaluatorPosixShorthand, positivePosixShorthandsContainExpectedCodePoint) {
  auto [token, cp] = GetParam();
  auto rs = Evaluator::toRangeSet(ClassNode(PropertyLeaf(token, false)));
  EXPECT_TRUE(rs.contains(cp)) << token;
}

INSTANTIATE_TEST_SUITE_P(
    Tokens,
    EvaluatorPosixShorthand,
    testing::Values(
        std::make_tuple("\\d", 48),
        std::make_tuple("\\w", 95),
        std::make_tuple("\\s", 32),
        std::make_tuple("\\p{ASCII}", 65),
        std::make_tuple("\\p{Alpha}", 65),
        std::make_tuple("\\p{Alnum}", 48),
        std::make_tuple("\\p{Lower}", 97),
        std::make_tuple("\\p{Upper}", 65),
        std::make_tuple("\\p{Digit}", 48),
        std::make_tuple("\\p{XDigit}", 102),
        std::make_tuple("\\p{Space}", 32),
        std::make_tuple("\\p{Blank}", 9),
        std::make_tuple("\\p{Cntrl}", 0),
        std::make_tuple("\\p{Graph}", 33),
        std::make_tuple("\\p{Print}", 32),
        std::make_tuple("\\p{Punct}", 46)));

TEST(Evaluator, negatedShorthandsComplementCorrectly) {
  auto nd = Evaluator::toRangeSet(ClassNode(PropertyLeaf("\\D", true)));
  EXPECT_TRUE(nd.contains('a'));
  EXPECT_FALSE(nd.contains('0'));

  auto ns = Evaluator::toRangeSet(ClassNode(PropertyLeaf("\\S", true)));
  EXPECT_FALSE(ns.contains(' '));
  EXPECT_TRUE(ns.contains('a'));
}

TEST(Evaluator, unknownPropertyThrowsEvaluationFailed) {
  EXPECT_THROW(
      Evaluator::toRangeSet(ClassNode(PropertyLeaf("\\p{ThisPropertyDoesNotExistXyz}", false))),
      EvaluationFailedException);
}

TEST(Evaluator, unknownPropertyInsideIntersectionThrows) {
  auto inter = ClassNode(Intersection(std::vector<ClassNode>{
      ClassNode(PropertyLeaf("\\p{UnknownXyz}", false)), ClassNode(Range('a', 'z'))}));
  EXPECT_THROW(Evaluator::toRangeSet(inter), EvaluationFailedException);
}

TEST(Evaluator, tryToRangeSetReturnsNullOnFailure) {
  EXPECT_FALSE(Evaluator::tryToRangeSet(ClassNode(PropertyLeaf("\\p{UnknownXyz}", false))).has_value());
}

TEST(Evaluator, tryToRangeSetReturnsRangeSetOnSuccess) {
  auto rs = Evaluator::tryToRangeSet(ClassNode(PropertyLeaf("\\d", false)));
  ASSERT_TRUE(rs.has_value());
  EXPECT_TRUE(rs->contains('5'));
}

TEST(Evaluator, javaAlphabeticIntersectionEvaluates) {
  std::size_t pos = 0;
  auto node = ClassBodyParser::parseClass("[\\p{javaAlphabetic}&&[a-z]]", pos);
  auto rs = Evaluator::toRangeSet(node);
  EXPECT_TRUE(rs.contains('a'));
  EXPECT_FALSE(rs.contains(0x03B1));
  EXPECT_FALSE(rs.contains('&'));
}

TEST(Evaluator, scriptAliasIntersectionEvaluates) {
  std::size_t pos = 0;
  auto node = ClassBodyParser::parseClass("[\\p{sc=Grek}&&\\p{L}]", pos);
  auto rs = Evaluator::toRangeSet(node);
  EXPECT_TRUE(rs.contains(0x03B1));
  EXPECT_FALSE(rs.contains('a'));
  EXPECT_FALSE(rs.contains('&'));
}

TEST(Evaluator, inPrefixIntersectionUsesBlockNotScript) {
  std::size_t pos = 0;
  auto node = ClassBodyParser::parseClass("[\\p{InGreek}&&\\x{1F00}]", pos);
  auto rs = Evaluator::toRangeSet(node);
  EXPECT_FALSE(rs.contains(0x1F00));
}

} // namespace facebook::velox::functions::java_pcre2_translator::test
