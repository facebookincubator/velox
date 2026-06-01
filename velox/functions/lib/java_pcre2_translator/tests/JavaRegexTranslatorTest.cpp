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
// org.pcre4j.regex.translate.JavaRegexTranslatorTest (Java) under
// Apache-2.0 by the same author for inclusion in Velox.
//
#include "velox/functions/lib/java_pcre2_translator/JavaRegexTranslator.h"

#include <gtest/gtest.h>

namespace facebook::velox::functions::java_pcre2_translator::test {

TEST(JavaRegexTranslator, passthroughForPatternsWithoutProperties) {
  EXPECT_EQ("\\d+", toPcre2Pattern("\\d+"));
  EXPECT_EQ("[a-z]", toPcre2Pattern("[a-z]"));
  EXPECT_EQ("abc", toPcre2Pattern("abc"));
}

TEST(JavaRegexTranslator, rewritesInBlockProperty) {
  EXPECT_EQ("\\p{Greek}", toPcre2Pattern("\\p{InGreek}"));
  EXPECT_EQ("\\P{Greek}", toPcre2Pattern("\\P{InGreek}"));
  EXPECT_EQ("\\p{Greek}", toPcre2Pattern("\\p{blk=Greek}"));
  EXPECT_EQ("\\P{Greek}", toPcre2Pattern("\\P{block=Greek}"));
  EXPECT_EQ("\\p{Basic_Latin}", toPcre2Pattern("\\p{blk=BasicLatin}"));
  EXPECT_EQ("a\\p{Greek}b", toPcre2Pattern("a\\p{InGreek}b"));
  EXPECT_EQ("[\\p{Greek}]", toPcre2Pattern("[\\p{InGreek}]"));
}

TEST(JavaRegexTranslator, rewritesIsScriptProperty) {
  EXPECT_EQ("\\p{L}", toPcre2Pattern("\\p{IsL}"));
  EXPECT_EQ("\\p{LC}", toPcre2Pattern("\\p{IsLC}"));
  EXPECT_EQ("\\p{ASCII}", toPcre2Pattern("\\p{IsASCII}"));
}

TEST(JavaRegexTranslator, rewritesShortAliases) {
  EXPECT_EQ("[\\x{00}-\\x{FF}]", toPcre2Pattern("\\p{L1}"));
}

TEST(JavaRegexTranslator, rewritesJavaProperty) {
  EXPECT_EQ("\\p{Ll}", toPcre2Pattern("\\p{javaLowerCase}"));
}

TEST(JavaRegexTranslator, rewritesUnicodeEscapeSurrogatePairs) {
  EXPECT_EQ("\\x{1f600}", toPcre2Pattern("\\uD83D\\uDE00"));
  EXPECT_THROW(toPcre2Pattern("\\uD83D"), EvaluationFailedException);
}

TEST(JavaRegexTranslator, doesNotRewriteInsideQuotation) {
  EXPECT_EQ("\\Q\\p{InGreek}\\E", toPcre2Pattern("\\Q\\p{InGreek}\\E"));
}

TEST(JavaRegexTranslator, doesNotRewriteEscapedBackslashFollowedByP) {
  EXPECT_THROW(toPcre2Pattern("\\\\p{InGreek}"), EvaluationFailedException);
}

TEST(JavaRegexTranslator, rejectsIllegalQuantifierBody) {
  EXPECT_THROW(toPcre2Pattern("a{^InGreek}"), EvaluationFailedException);
  EXPECT_THROW(toPcre2Pattern("a{}"), EvaluationFailedException);
  EXPECT_THROW(toPcre2Pattern("a{,3}"), EvaluationFailedException);
  EXPECT_THROW(toPcre2Pattern("a{"), EvaluationFailedException);
  EXPECT_THROW(toPcre2Pattern("a{3"), EvaluationFailedException);
}

TEST(JavaRegexTranslator, acceptsValidQuantifiers) {
  EXPECT_EQ("a{3}", toPcre2Pattern("a{3}"));
  EXPECT_EQ("a{3,}", toPcre2Pattern("a{3,}"));
  EXPECT_EQ("a{3,5}", toPcre2Pattern("a{3,5}"));
}

TEST(JavaRegexTranslator, escapeHatchDisablesTranslator) {
  EXPECT_EQ("\\p{Greek}", toPcre2Pattern("\\p{InGreek}"));
}

TEST(JavaRegexTranslator, rewritesSurrogateBlockToRange) {
  EXPECT_EQ("[\\x{D800}-\\x{DB7F}]", toPcre2Pattern("\\p{InHIGH_SURROGATES}"));
  EXPECT_EQ("[\\x{DC00}-\\x{DFFF}]", toPcre2Pattern("\\p{InLOW_SURROGATES}"));
}

TEST(JavaRegexTranslator, reportsSurrogateRangeNeedsRawByteMode) {
  bool needsRawByteMode = false;
  EXPECT_EQ(
      "[\\x{D800}-\\x{DB7F}]",
      toPcre2Pattern("\\p{InHIGH_SURROGATES}", needsRawByteMode));
  EXPECT_TRUE(needsRawByteMode);
}

TEST(JavaRegexTranslator, reportsRawSurrogateBytesNeedRawByteMode) {
  bool needsRawByteMode = false;
  EXPECT_EQ(
      "[\xED\xA0\x80]",
      toPcre2Pattern("[\xED\xA0\x80]", needsRawByteMode));
  EXPECT_TRUE(needsRawByteMode);
}

TEST(JavaRegexTranslator, doesNotReportRawByteModeForSupplementaryScalar) {
  bool needsRawByteMode = true;
  EXPECT_EQ("\\x{1f600}", toPcre2Pattern("\\uD83D\\uDE00", needsRawByteMode));
  EXPECT_FALSE(needsRawByteMode);
}

TEST(JavaRegexTranslator, negatedSurrogateBlockIsNegated) {
  EXPECT_EQ("[^\\x{D800}-\\x{DB7F}]", toPcre2Pattern("\\P{InHIGH_SURROGATES}"));
}

TEST(JavaRegexTranslator, rewritesJavaDefinedAsNegatedUnassigned) {
  EXPECT_EQ("\\P{Cn}", toPcre2Pattern("\\p{javaDefined}"));
}

TEST(JavaRegexTranslator, multipleTokensInOnePattern) {
  EXPECT_EQ("\\p{Greek}\\p{Hiragana}", toPcre2Pattern("\\p{InGreek}\\p{InHiragana}"));
}

TEST(JavaRegexTranslator, nestedUnionFlattens) {
  const auto result = toPcre2Pattern("[abc[def]]");
  EXPECT_EQ(std::string::npos, result.find("[[")) << result;
  EXPECT_EQ("[abcdef]", result);
}

TEST(JavaRegexTranslator, intersectionBecomesRangeSet) {
  EXPECT_EQ("[^\\x{0}-\\x{10FFFF}]", toPcre2Pattern("[a-c&&d-f]"));
}

TEST(JavaRegexTranslator, wDashHashEscapesDash) {
  const auto result = toPcre2Pattern("[\\w-#]");
  EXPECT_NE(std::string::npos, result.find("\\-")) << result;
}

TEST(JavaRegexTranslator, classBodyRewritePreservesOutsidePattern) {
  EXPECT_EQ("a[bc]d", toPcre2Pattern("a[bc]d"));
}

TEST(JavaRegexTranslator, propertyInsideClassRewritten) {
  const auto result = toPcre2Pattern("[\\p{InGreek}]");
  EXPECT_NE(std::string::npos, result.find("\\p{Greek}")) << result;
  EXPECT_EQ(std::string::npos, result.find("\\p{InGreek}")) << result;
}

TEST(JavaRegexTranslator, intersectionWithKnownPropertyEvaluated) {
  const auto result = toPcre2Pattern("[\\d&&[0-3]]");
  EXPECT_EQ(std::string::npos, result.find("&&")) << result;
}

TEST(JavaRegexTranslator, dropsUFlagInModeModifier) {
  EXPECT_EQ("(?i)foo", toPcre2Pattern("(?iu)foo"));
  EXPECT_EQ("(?i)foo", toPcre2Pattern("(?ui)foo"));
  EXPECT_EQ("(?im)foo", toPcre2Pattern("(?ium)foo"));
}

TEST(JavaRegexTranslator, dropsUInScopedGroup) {
  EXPECT_EQ("(?i:foo)", toPcre2Pattern("(?iu:foo)"));
}

TEST(JavaRegexTranslator, dropsDFlag) {
  EXPECT_EQ("(?m)foo", toPcre2Pattern("(?dm)foo"));
}

TEST(JavaRegexTranslator, emptyFlagsRemovedEntirely) {
  EXPECT_EQ("foo", toPcre2Pattern("(?u)foo"));
  EXPECT_EQ("(?:foo)", toPcre2Pattern("(?u:foo)"));
}

TEST(JavaRegexTranslator, preservesNonModeGroups) {
  EXPECT_EQ("(?:foo)", toPcre2Pattern("(?:foo)"));
  EXPECT_EQ("(?=foo)", toPcre2Pattern("(?=foo)"));
  EXPECT_EQ("(?<name>foo)", toPcre2Pattern("(?<name>foo)"));
  EXPECT_EQ("(?#comment)foo", toPcre2Pattern("(?#comment)foo"));
}

TEST(JavaRegexTranslator, handlesOnOffFlagGroup) {
  EXPECT_EQ("(?i-m)foo", toPcre2Pattern("(?iu-mU)foo"));
}

TEST(JavaRegexTranslator, allFlagsDroppedFromOnOff) {
  EXPECT_EQ("foo", toPcre2Pattern("(?u-U)foo"));
}

TEST(JavaRegexTranslator, doesNotTouchInsideClass) {
  EXPECT_EQ("[(?i)]", toPcre2Pattern("[(?i)]"));
}

TEST(JavaRegexTranslator, propertyIntersectionEndToEnd) {
  const auto out = toPcre2Pattern("[\\p{L}&&[\\P{InGreek}]]");
  EXPECT_EQ(std::string::npos, out.find("&&")) << out;
  EXPECT_EQ(std::string::npos, out.find("[[")) << out;
  EXPECT_NE(std::string::npos, out.find("A-Z")) << out;
  EXPECT_NE(std::string::npos, out.find("a-z")) << out;
  EXPECT_EQ(std::string::npos, out.find("\\x{3B1}")) << out;
  EXPECT_EQ(std::string::npos, out.find("\\x{3A9}")) << out;
}

TEST(JavaRegexTranslator, inlineCaseInsensitiveExpandsCasedTopLevelProperty) {
  EXPECT_EQ(
      "(?i)[\\p{Lu}\\p{Ll}\\p{Lt}]",
      toPcre2Pattern("(?i)\\p{Lu}"));
}

TEST(JavaRegexTranslator, inlineCaseInsensitiveExpandsCasedClassProperty) {
  EXPECT_EQ(
      "(?i)[\\p{Lu}\\p{Ll}\\p{Lt}]",
      toPcre2Pattern("(?i)[\\p{Lu}]"));
}

TEST(JavaRegexTranslator, inlineCaseInsensitiveExpandsNegatedCasedClassProperty) {
  EXPECT_EQ(
      "(?i)[^\\p{Lu}\\p{Ll}\\p{Lt}]",
      toPcre2Pattern("(?i)[\\P{Lu}]"));
}

TEST(JavaRegexTranslator, inlineCaseInsensitiveKeepsLiteralAsciiRange) {
  EXPECT_EQ("(?i)[A-Z]", toPcre2Pattern("(?i)[A-Z]"));
}

TEST(JavaRegexTranslator, embeddedFlagsDoNotLeakPastEnclosingGroup) {
  EXPECT_EQ("(a(?i)b)[\\p{Lu}]", toPcre2Pattern("(a(?i)b)[\\p{Lu}]"));
}

TEST(JavaRegexTranslator, longBackreferenceDoesNotOverflow) {
  const auto result = toPcre2Pattern("\\999999999999999999999999999999");
  EXPECT_EQ(0, result.rfind("(*F)", 0)) << result;
}

TEST(JavaRegexTranslator, unicodeCharacterClassIntersectionThrowsInsteadOfAsciiEvaluation) {
  EXPECT_THROW(
      toPcre2Pattern("(?U)[\\d&&\\p{InArabic}]"),
      EvaluationFailedException);
}

TEST(JavaRegexTranslator, escapedBraceIsNotQuantifier) {
  EXPECT_EQ("\\{", toPcre2Pattern("\\{"));
  EXPECT_EQ("a\\{b}", toPcre2Pattern("a\\{b}"));
  EXPECT_EQ("\\{not-a-quantifier}", toPcre2Pattern("\\{not-a-quantifier}"));
}

TEST(JavaRegexTranslator, doubleBackslashThenBraceStillQuantifier) {
  EXPECT_THROW(toPcre2Pattern("\\\\{x}"), EvaluationFailedException);
}

TEST(JavaRegexTranslator, commentsModeIgnoresBracesInLineComments) {
  EXPECT_EQ("(?x)# {\n a", toPcre2Pattern("(?x)# {\n a"));
  EXPECT_EQ("(?x:# {\n a)", toPcre2Pattern("(?x:# {\n a)"));
}

TEST(JavaRegexTranslatorRe2, reusesPropertyAndClassPipeline) {
  EXPECT_EQ("\\p{Greek}", toRe2Pattern("\\p{InGreek}"));
  EXPECT_EQ("[abcdef]", toRe2Pattern("[abc[def]]"));
  EXPECT_EQ("[^\\x{0}-\\x{10FFFF}]", toRe2Pattern("[a-c&&d-f]"));
}

TEST(JavaRegexTranslatorRe2, rewritesJavaNamedCapturingGroups) {
  EXPECT_EQ("(?P<name>foo)", toRe2Pattern("(?<name>foo)"));
  EXPECT_EQ("(a(?P<num>\\d+))", toRe2Pattern("(a(?<num>\\d+))"));
}

TEST(JavaRegexTranslatorRe2, doesNotRewriteNamedGroupLookalikesInQuotesOrClasses) {
  EXPECT_EQ("\\Q(?<name>foo)\\E", toRe2Pattern("\\Q(?<name>foo)\\E"));
  EXPECT_EQ("[(?<name>)]", toRe2Pattern("[(?<name>)]"));
}

TEST(JavaRegexTranslatorRe2, rejectsLookaround) {
  EXPECT_THROW(toRe2Pattern("(?=foo)"), EvaluationFailedException);
  EXPECT_THROW(toRe2Pattern("(?!foo)"), EvaluationFailedException);
  EXPECT_THROW(toRe2Pattern("(?<=foo)"), EvaluationFailedException);
  EXPECT_THROW(toRe2Pattern("(?<!foo)"), EvaluationFailedException);
}

TEST(JavaRegexTranslatorRe2, rejectsBackreferencesOutsideClasses) {
  EXPECT_THROW(toRe2Pattern("(a)\\1"), EvaluationFailedException);
  EXPECT_THROW(toRe2Pattern("(?<n>a)\\k<n>"), EvaluationFailedException);
  EXPECT_NO_THROW(toRe2Pattern("[\\1\\k<n>]"));
}

TEST(JavaRegexTranslatorRe2, rejectsPossessiveQuantifiers) {
  EXPECT_THROW(toRe2Pattern("a*+"), EvaluationFailedException);
  EXPECT_THROW(toRe2Pattern("a?+"), EvaluationFailedException);
  EXPECT_THROW(toRe2Pattern("a++"), EvaluationFailedException);
  EXPECT_THROW(toRe2Pattern("a{1,3}+"), EvaluationFailedException);
}

TEST(JavaRegexTranslatorRe2, rejectsAtomicGroupsAndUnsupportedFlags) {
  EXPECT_THROW(toRe2Pattern("(?>foo)"), EvaluationFailedException);
  EXPECT_THROW(toRe2Pattern("(?U)foo"), EvaluationFailedException);
  EXPECT_THROW(toRe2Pattern("(?d)foo"), EvaluationFailedException);
  EXPECT_THROW(toRe2Pattern("(?c)foo"), EvaluationFailedException);
  EXPECT_THROW(toRe2Pattern("(?id:foo)"), EvaluationFailedException);
  EXPECT_EQ("foo", toRe2Pattern("(?u)foo"));
  EXPECT_EQ("foo", toRe2Pattern("(?-U)foo"));
  EXPECT_EQ("(?i:foo)", toRe2Pattern("(?i-d:foo)"));
  EXPECT_EQ("foo", toRe2Pattern("(?-c)foo"));
}

TEST(JavaRegexTranslatorRe2, rewritesJavaOctalEscapesForRe2) {
  EXPECT_EQ("\\x{a}", toRe2Pattern("\\012"));
}

TEST(JavaRegexTranslatorRe2, translatesCommentsModeForRe2) {
  EXPECT_EQ("abc", toRe2Pattern("(?x)a b c"));
  EXPECT_EQ("abcdef", toRe2Pattern("(?x)abc # comment\ndef"));
  EXPECT_EQ("a b", toRe2Pattern("(?x)a\\ b"));
  EXPECT_EQ("[a]", toRe2Pattern("(?x)[ a]"));
  EXPECT_EQ("[ ]", toRe2Pattern("(?x)[\\ ]"));
  EXPECT_EQ("[a]", toRe2Pattern("(?x)[a# comment\n]"));
  EXPECT_THROW(toRe2Pattern("(?x)[a# comment]"), EvaluationFailedException);
  EXPECT_EQ("(?i:ab)", toRe2Pattern("(?ix:a b)"));
  EXPECT_THROW(toRe2Pattern("(?x)(? <name>a)"), EvaluationFailedException);
  EXPECT_THROW(toRe2Pattern("(?x)(? :a)"), EvaluationFailedException);
}

TEST(JavaRegexTranslatorRe2, unsupportedFeatureLookalikesInQuotesAreLiterals) {
  EXPECT_EQ(
      "\\Q(?=foo)\\1*+(?>x)(?U)\\E",
      toRe2Pattern("\\Q(?=foo)\\1*+(?>x)(?U)\\E"));
}

} // namespace facebook::velox::functions::java_pcre2_translator::test
