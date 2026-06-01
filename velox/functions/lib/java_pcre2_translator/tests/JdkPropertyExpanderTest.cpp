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
// org.pcre4j.regex.translate.JdkPropertyExpanderTest (Java) under Apache-2.0 by the
// same author for inclusion in Velox.
//
#include "velox/functions/lib/java_pcre2_translator/JdkPropertyExpander.h"

#include <gtest/gtest.h>

namespace facebook::velox::functions::java_pcre2_translator::test {

TEST(JdkPropertyExpander, asciiLetterCoverage) {
  auto l = JdkPropertyExpander::expand("\\p{L}");
  ASSERT_TRUE(l.has_value());
  EXPECT_TRUE(l->contains('a'));
  EXPECT_TRUE(l->contains('Z'));
  EXPECT_FALSE(l->contains('0'));
  EXPECT_FALSE(l->contains(' '));
}

TEST(JdkPropertyExpander, greekScript) {
  auto g = JdkPropertyExpander::expand("\\p{Greek}");
  ASSERT_TRUE(g.has_value());
  EXPECT_TRUE(g->contains(0x03B1));
  EXPECT_FALSE(g->contains('a'));
}

TEST(JdkPropertyExpander, negatedProperty) {
  auto notL = JdkPropertyExpander::expand("\\P{L}");
  ASSERT_TRUE(notL.has_value());
  EXPECT_FALSE(notL->contains('a'));
  EXPECT_TRUE(notL->contains('0'));
}

TEST(JdkPropertyExpander, unknownReturnsNull) {
  EXPECT_FALSE(JdkPropertyExpander::expand("\\p{FooBarBaz}").has_value());
}

TEST(JdkPropertyExpander, caches) {
  auto first = JdkPropertyExpander::expand("\\p{L}");
  auto second = JdkPropertyExpander::expand("\\p{L}");
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(*first, *second);
}

TEST(JdkPropertyExpander, greekIntersectionWithLetters) {
  auto letters = JdkPropertyExpander::expand("\\p{L}");
  auto notGreek = JdkPropertyExpander::expand("\\P{Greek}");
  ASSERT_TRUE(letters.has_value());
  ASSERT_TRUE(notGreek.has_value());
  auto lettersNotGreek = letters->intersect(*notGreek);
  EXPECT_TRUE(lettersNotGreek.contains('a'));
  EXPECT_TRUE(lettersNotGreek.contains(0x6000));
  EXPECT_FALSE(lettersNotGreek.contains(0x03B1));
}

TEST(JdkPropertyExpander, leafCategoryLu) {
  auto lu = JdkPropertyExpander::expand("\\p{Lu}");
  ASSERT_TRUE(lu.has_value());
  EXPECT_TRUE(lu->contains('A'));
  EXPECT_FALSE(lu->contains('a'));
  EXPECT_FALSE(lu->contains('0'));
}

TEST(JdkPropertyExpander, combinedCategoryN) {
  auto n = JdkPropertyExpander::expand("\\p{N}");
  ASSERT_TRUE(n.has_value());
  EXPECT_TRUE(n->contains('0'));
  EXPECT_FALSE(n->contains('a'));
}

TEST(JdkPropertyExpander, binaryAlphabeticProperty) {
  auto alphabetic = JdkPropertyExpander::expand("\\p{Alphabetic}");
  ASSERT_TRUE(alphabetic.has_value());
  EXPECT_TRUE(alphabetic->contains('a'));
  EXPECT_TRUE(alphabetic->contains(0x03B1));
  EXPECT_FALSE(alphabetic->contains('0'));
}

TEST(JdkPropertyExpander, scriptShortAlias) {
  auto greek = JdkPropertyExpander::expand("\\p{Grek}");
  ASSERT_TRUE(greek.has_value());
  EXPECT_TRUE(greek->contains(0x03B1));
  EXPECT_FALSE(greek->contains('a'));
}

TEST(JdkPropertyExpander, blockLongAlias) {
  auto basicLatin = JdkPropertyExpander::expand("\\p{Basic_Latin}");
  ASSERT_TRUE(basicLatin.has_value());
  EXPECT_TRUE(basicLatin->contains('A'));
  EXPECT_FALSE(basicLatin->contains(0x03B1));
}

TEST(JdkPropertyExpander, inPrefixUsesBlockNotScript) {
  auto greekBlock = JdkPropertyExpander::expand("\\p{InGreek}");
  ASSERT_TRUE(greekBlock.has_value());
  EXPECT_TRUE(greekBlock->contains(0x03B1));
  EXPECT_FALSE(greekBlock->contains(0x1F00));
}

TEST(JdkPropertyExpander, nonPropertyTokenReturnsNull) {
  EXPECT_FALSE(JdkPropertyExpander::expand("\\d").has_value());
  EXPECT_FALSE(JdkPropertyExpander::expand("\\w").has_value());
}

} // namespace facebook::velox::functions::java_pcre2_translator::test
