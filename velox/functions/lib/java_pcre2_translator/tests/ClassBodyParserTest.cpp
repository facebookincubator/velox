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
// org.pcre4j.regex.translate.ClassBodyParserTest (Java) under Apache-2.0 by the
// same author for inclusion in Velox.
//
#include "velox/functions/lib/java_pcre2_translator/ClassBodyParser.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace facebook::velox::functions::java_pcre2_translator::test {
namespace {

ClassNode parse(std::string_view classStr) {
  std::size_t pos = 0;
  return ClassBodyParser::parseClass(classStr, pos);
}

} // namespace

TEST(ClassBodyParser, simpleLiterals) {
  auto node = parse("[abc]");
  auto* u = node.getIf<Union>();
  ASSERT_NE(nullptr, u);
  ASSERT_EQ(3, u->children.size());
  EXPECT_EQ(ClassNode(Literal('a')), *u->children[0]);
  EXPECT_EQ(ClassNode(Literal('b')), *u->children[1]);
  EXPECT_EQ(ClassNode(Literal('c')), *u->children[2]);
}

TEST(ClassBodyParser, singleCharClass) {
  EXPECT_EQ(ClassNode(Literal('a')), parse("[a]"));
}

TEST(ClassBodyParser, rangeClass) {
  EXPECT_EQ(ClassNode(Range('a', 'z')), parse("[a-z]"));
}

TEST(ClassBodyParser, negatedRange) {
  auto node = parse("[^a-z]");
  auto* neg = node.getIf<Negated>();
  ASSERT_NE(nullptr, neg);
  EXPECT_EQ(ClassNode(Range('a', 'z')), *neg->child);
}

TEST(ClassBodyParser, nestedClassUnion) {
  auto node = parse("[abc[def]]");
  auto* u = node.getIf<Union>();
  ASSERT_NE(nullptr, u);
  ASSERT_EQ(4, u->children.size());
  EXPECT_EQ(ClassNode(Literal('a')), *u->children[0]);
  EXPECT_TRUE(u->children[3]->is<Union>());
}

TEST(ClassBodyParser, intersection) {
  auto node = parse("[a-c&&d-f]");
  auto* inter = node.getIf<Intersection>();
  ASSERT_NE(nullptr, inter);
  ASSERT_EQ(2, inter->operands.size());
  EXPECT_EQ(ClassNode(Range('a', 'c')), *inter->operands[0]);
  EXPECT_EQ(ClassNode(Range('d', 'f')), *inter->operands[1]);
}

TEST(ClassBodyParser, wDashHashPattern) {
  auto node = parse("[\\w-#]");
  auto* u = node.getIf<Union>();
  ASSERT_NE(nullptr, u);
  ASSERT_EQ(3, u->children.size());
  EXPECT_TRUE(u->children[0]->is<PropertyLeaf>());
  EXPECT_EQ(ClassNode(Literal('-')), *u->children[1]);
  EXPECT_EQ(ClassNode(Literal('#')), *u->children[2]);
}

TEST(ClassBodyParser, shorthandEscapes) {
  auto node = parse("[\\d\\p{L}]");
  auto* u = node.getIf<Union>();
  ASSERT_NE(nullptr, u);
  ASSERT_EQ(2, u->children.size());
  ASSERT_TRUE(u->children[0]->is<PropertyLeaf>());
  EXPECT_EQ("\\d", u->children[0]->getIf<PropertyLeaf>()->pcre2Token);
  EXPECT_TRUE(u->children[1]->is<PropertyLeaf>());
}

TEST(ClassBodyParser, bracketPropertyRewriteParsesAsAst) {
  auto node = parse("[\\p{Alpha}]");
  auto* u = node.getIf<Union>();
  ASSERT_NE(nullptr, u);
  ASSERT_EQ(2, u->children.size());
  EXPECT_EQ(ClassNode(Range('a', 'z')), *u->children[0]);
  EXPECT_EQ(ClassNode(Range('A', 'Z')), *u->children[1]);
}

TEST(ClassBodyParser, negatedBracketPropertyRewriteParsesAsNegatedAst) {
  auto node = parse("[\\P{Alpha}]");
  auto* neg = node.getIf<Negated>();
  ASSERT_NE(nullptr, neg);
  EXPECT_TRUE(neg->child->is<Union>());
}

TEST(ClassBodyParser, quotedBracket) {
  EXPECT_EQ(ClassNode(Literal(']')), parse("[\\Q]\\E]"));
}

TEST(ClassBodyParser, hexEscape) {
  EXPECT_EQ(ClassNode(Literal('A')), parse("[\\x41]"));
}

TEST(ClassBodyParser, unicodeEscape) {
  EXPECT_EQ(ClassNode(Literal('A')), parse("[\\u0041]"));
}

TEST(ClassBodyParser, escapedNonAsciiLiteralConsumesWholeCodePoint) {
  EXPECT_EQ(ClassNode(Range('a', 0x4444)), parse("[a-\\\xE4\x91\x84]"));
}

TEST(ClassBodyParser, multipleIntersectionOperands) {
  auto* inter = parse("[a-m&&m-z&&a-c]").getIf<Intersection>();
  ASSERT_NE(nullptr, inter);
  EXPECT_EQ(3, inter->operands.size());
}

TEST(ClassBodyParser, nestedNegatedClass) {
  auto node = parse("[a-d[^0-9]]");
  auto* u = node.getIf<Union>();
  ASSERT_NE(nullptr, u);
  ASSERT_EQ(2, u->children.size());
  EXPECT_TRUE(u->children[1]->is<Negated>());
}

TEST(ClassBodyParser, intersectionWithNestedClass) {
  EXPECT_TRUE(parse("[[a-m]&&[m-z]]").is<Intersection>());
}

TEST(ClassBodyParser, rangeAtEndOfClass) {
  EXPECT_TRUE(parse("[a\\-]").is<Union>());
}

TEST(ClassBodyParser, unterminatedClassThrows) {
  EXPECT_THROW(parse("[abc"), std::invalid_argument);
}

TEST(ClassBodyParser, unterminatedNegatedClassThrows) {
  EXPECT_THROW(parse("[^abc"), std::invalid_argument);
}

TEST(ClassBodyParser, unterminatedNestedClassThrows) {
  EXPECT_THROW(parse("[a[b-c]"), std::invalid_argument);
}

TEST(ClassBodyParser, incompleteHexEscapeThrows) {
  EXPECT_THROW(parse("[\\x]"), std::invalid_argument);
  EXPECT_THROW(parse("[\\xA]"), std::invalid_argument);
}

TEST(ClassBodyParser, unterminatedHexBraceEscapeThrows) {
  EXPECT_THROW(parse("[\\x{ABC]"), std::invalid_argument);
}

TEST(ClassBodyParser, emptyHexBraceEscapeThrows) {
  EXPECT_THROW(parse("[\\x{}]"), std::invalid_argument);
}

TEST(ClassBodyParser, outOfRangeHexBraceEscapeThrows) {
  EXPECT_THROW(parse("[\\x{110000}]"), std::invalid_argument);
  EXPECT_THROW(parse("[\\x{FFFFFFFFF}]"), std::invalid_argument);
}

TEST(ClassBodyParser, incompleteUnicodeEscapeThrows) {
  EXPECT_THROW(parse("[\\u]"), std::invalid_argument);
  EXPECT_THROW(parse("[\\u00]"), std::invalid_argument);
  EXPECT_THROW(parse("[\\u00A]"), std::invalid_argument);
}

TEST(ClassBodyParser, octalEscapeAcceptsThreeDigits) {
  EXPECT_EQ(ClassNode(Literal(0x41)), parse("[\\0101]"));
}

TEST(ClassBodyParser, octalEscapeStopsAtNonOctalChar) {
  auto node = parse("[\\08]");
  auto* u = node.getIf<Union>();
  ASSERT_NE(nullptr, u);
  ASSERT_EQ(2, u->children.size());
  EXPECT_EQ(ClassNode(Literal(0)), *u->children[0]);
  EXPECT_EQ(ClassNode(Literal('8')), *u->children[1]);
}

TEST(ClassBodyParser, octalEscapeCappedAtFF) {
  auto node = parse("[\\0400]");
  auto* u = node.getIf<Union>();
  ASSERT_NE(nullptr, u);
  ASSERT_EQ(2, u->children.size());
  EXPECT_EQ(ClassNode(Literal(0x20)), *u->children[0]);
  EXPECT_EQ(ClassNode(Literal('0')), *u->children[1]);
}

TEST(ClassBodyParser, controlCharacterEscape) {
  EXPECT_EQ(ClassNode(Literal(0x01)), parse("[\\cA]"));
}

TEST(ClassBodyParser, simpleEscapesProduceLiterals) {
  EXPECT_EQ(ClassNode(Literal(0x07)), parse("[\\a]"));
  EXPECT_EQ(ClassNode(Literal(0x1B)), parse("[\\e]"));
  EXPECT_EQ(ClassNode(Literal('\n')), parse("[\\n]"));
  EXPECT_EQ(ClassNode(Literal('\t')), parse("[\\t]"));
}

TEST(ClassBodyParser, trailingBackslashThrows) {
  EXPECT_THROW(parse("[\\"), std::invalid_argument);
}

} // namespace facebook::velox::functions::java_pcre2_translator::test
