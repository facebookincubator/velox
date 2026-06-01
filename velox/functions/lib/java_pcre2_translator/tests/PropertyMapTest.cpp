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
// Ported from org.pcre4j.regex.translate.PropertyMapTest (Java).
//
#include "velox/functions/lib/java_pcre2_translator/PropertyMap.h"

#include <gtest/gtest.h>

namespace facebook::velox::functions::java_pcre2_translator::test {

TEST(PropertyMap, inPrefixStrip) {
  EXPECT_EQ("Greek", PropertyMap::apply("InGreek").value());
}

TEST(PropertyMap, isPrefixStrip) {
  EXPECT_EQ("L", PropertyMap::apply("IsL").value());
}

TEST(PropertyMap, unknownReturnsNullopt) {
  EXPECT_FALSE(PropertyMap::apply("FooBarBaz").has_value());
}

TEST(PropertyMap, l1ExpandsToRange) {
  EXPECT_EQ("[\\x{00}-\\x{FF}]", PropertyMap::apply("L1").value());
}

TEST(PropertyMap, javaLowerCase) {
  EXPECT_EQ("Ll", PropertyMap::apply("javaLowerCase").value());
}

TEST(PropertyMap, highSurrogatesExpandToRange) {
  EXPECT_EQ(
      "[\\x{D800}-\\x{DB7F}]", PropertyMap::apply("InHIGH_SURROGATES").value());
}

TEST(PropertyMap, lowSurrogatesExpandToRange) {
  EXPECT_EQ(
      "[\\x{DC00}-\\x{DFFF}]", PropertyMap::apply("InLOW_SURROGATES").value());
}

TEST(PropertyMap, isAsciiStripsIs) {
  EXPECT_EQ("ASCII", PropertyMap::apply("IsASCII").value());
}

TEST(PropertyMap, javaDefinedMapsToNegatedCn) {
  EXPECT_EQ("\\P{Cn}", PropertyMap::apply("javaDefined").value());
}

} // namespace facebook::velox::functions::java_pcre2_translator::test
