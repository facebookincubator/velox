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
#include "velox/functions/lib/java_regex/JavaRegexTranslator.h"

#include <gtest/gtest.h>
#include <re2/re2.h>

namespace facebook::velox::functions {
namespace {

TEST(JavaRegexTranslatorTest, unicodeEscape) {
  // \uXXXX becomes \x{XXXX}; \\u (an escaped backslash + literal u) is left
  // alone; a short/invalid \u sequence is left as-is.
  EXPECT_EQ(normalizeJavaRegexForRe2("\\u0041"), "\\x{0041}");
  EXPECT_EQ(normalizeJavaRegexForRe2("a\\u00e9b"), "a\\x{00e9}b");
  EXPECT_EQ(
      normalizeJavaRegexForRe2("[\\u0041-\\u005A]"), "[\\x{0041}-\\x{005A}]");
  EXPECT_EQ(normalizeJavaRegexForRe2("\\\\u0041"), "\\\\u0041");
  EXPECT_EQ(normalizeJavaRegexForRe2("\\u12"), "\\u12");
  // A high/low surrogate pair folds into the supplementary code point it
  // encodes; RE2 has no '\u' but matches '\x{...}'. Java treats the pair as
  // a single character (e.g. U+1F600).
  EXPECT_EQ(normalizeJavaRegexForRe2("\\uD83D\\uDE00"), "\\x{1F600}");
  EXPECT_EQ(normalizeJavaRegexForRe2("a\\uD83D\\uDE00b"), "a\\x{1F600}b");
  EXPECT_EQ(normalizeJavaRegexForRe2("[\\uD83D\\uDE00]"), "[\\x{1F600}]");
  // A high surrogate not followed by a low surrogate is not folded; each unit
  // is emitted as \x{...} individually (RE2 rejects a lone surrogate, so the
  // caller falls back -- unchanged behavior).
  EXPECT_EQ(normalizeJavaRegexForRe2("\\uD83Dx"), "\\x{D83D}x");
  EXPECT_EQ(normalizeJavaRegexForRe2("\\uD83D\\u0041"), "\\x{D83D}\\x{0041}");
}

TEST(JavaRegexTranslatorTest, whitespaceLeftUntouched) {
  // The translator is deliberately widening-only: it never changes what an
  // already-RE2-compilable pattern matches. '\s' and '\S' compile under RE2
  // today, so they are passed through even though RE2's '\s' omits the
  // vertical tab (U+000B) that java.util.regex includes. Correcting that is a
  // semantic change and belongs in a separate change that can apply it to
  // every regex function at once; doing it here would make '\s' mean one thing
  // in the functions wired to the translator and another in the rest.
  EXPECT_EQ(normalizeJavaRegexForRe2("\\s"), "\\s");
  EXPECT_EQ(normalizeJavaRegexForRe2("\\S"), "\\S");
  EXPECT_EQ(normalizeJavaRegexForRe2("[a\\s]"), "[a\\s]");
  EXPECT_EQ(normalizeJavaRegexForRe2("[^\\S]"), "[^\\S]");
  EXPECT_EQ(normalizeJavaRegexForRe2("[\\s\\S]"), "[\\s\\S]");
  EXPECT_EQ(normalizeJavaRegexForRe2("\\\\s"), "\\\\s");
}

TEST(JavaRegexTranslatorTest, verbatimAndUntouched) {
  // Ordinary patterns pass through unchanged.
  EXPECT_EQ(normalizeJavaRegexForRe2("abc\\d+"), "abc\\d+");
  EXPECT_EQ(normalizeJavaRegexForRe2("[a-z]"), "[a-z]");
  // POSIX classes are copied verbatim and do not corrupt class tracking.
  EXPECT_EQ(
      normalizeJavaRegexForRe2("[[:alpha:]]\\u0041"), "[[:alpha:]]\\x{0041}");
  // \Q...\E contents are copied verbatim (the '[' does not open a class).
  EXPECT_EQ(
      normalizeJavaRegexForRe2("\\Q[\\u0041\\E\\u0041"),
      "\\Q[\\u0041\\E\\x{0041}");
  // RE2-irreducible features are left untouched (RE2 rejects them later).
  EXPECT_EQ(normalizeJavaRegexForRe2("(?=foo)"), "(?=foo)");
  EXPECT_EQ(normalizeJavaRegexForRe2("(a)\\1"), "(a)\\1");
}

TEST(JavaUnicodePropertyTest, blocks) {
  // Java Unicode block properties expand to code-point ranges via the static
  // block table (ICU-free). '^' inside braces and \P both negate.
  EXPECT_EQ(
      expandJavaUnicodePropertiesForRe2("\\p{InGreek}"), "[\\x{370}-\\x{3FF}]");
  EXPECT_EQ(
      expandJavaUnicodePropertiesForRe2("\\p{block=Greek}"),
      "[\\x{370}-\\x{3FF}]");
  EXPECT_EQ(
      expandJavaUnicodePropertiesForRe2("\\P{InGreek}"),
      "[^\\x{370}-\\x{3FF}]");
  EXPECT_EQ(
      expandJavaUnicodePropertiesForRe2("\\p{^InGreek}"),
      "[^\\x{370}-\\x{3FF}]");
  // Inside a class the range is emitted without its own brackets.
  EXPECT_EQ(
      expandJavaUnicodePropertiesForRe2("a[x\\p{InGreek}]c"),
      "a[x\\x{370}-\\x{3FF}]c");
  // Unknown block name is left untouched (RE2 rejects it -> Spark fallback).
  EXPECT_EQ(
      expandJavaUnicodePropertiesForRe2("\\p{InNoSuchBlock123}"),
      "\\p{InNoSuchBlock123}");
}

TEST(JavaUnicodePropertyTest, scriptsCategoriesAndAliases) {
  // 'Is'-prefixed scripts/categories drop the prefix to RE2's native name.
  EXPECT_EQ(expandJavaUnicodePropertiesForRe2("\\p{IsL}"), "\\p{L}");
  EXPECT_EQ(expandJavaUnicodePropertiesForRe2("\\p{IsGreek}"), "\\p{Greek}");
  EXPECT_EQ(expandJavaUnicodePropertiesForRe2("\\P{IsLu}"), "\\P{Lu}");
  // Aliases.
  EXPECT_EQ(
      expandJavaUnicodePropertiesForRe2("\\p{ASCII}"), "[\\x{0}-\\x{7F}]");
  EXPECT_EQ(
      expandJavaUnicodePropertiesForRe2("\\p{IsASCII}"), "[\\x{0}-\\x{7F}]");
  EXPECT_EQ(expandJavaUnicodePropertiesForRe2("\\p{L1}"), "[\\x{0}-\\x{FF}]");
  EXPECT_EQ(
      expandJavaUnicodePropertiesForRe2("\\p{LC}"), "[\\p{Lu}\\p{Ll}\\p{Lt}]");
  EXPECT_EQ(
      expandJavaUnicodePropertiesForRe2("\\p{IsLC}"),
      "[\\p{Lu}\\p{Ll}\\p{Lt}]");
}

TEST(JavaUnicodePropertyTest, verbatimAndUntouched) {
  // RE2-native forms are left as-is (RE2 already supports them).
  EXPECT_EQ(expandJavaUnicodePropertiesForRe2("\\p{L}"), "\\p{L}");
  EXPECT_EQ(expandJavaUnicodePropertiesForRe2("\\p{Greek}"), "\\p{Greek}");
  // ICU-only Java predicates are left for Spark fallback.
  EXPECT_EQ(
      expandJavaUnicodePropertiesForRe2("\\p{javaLowerCase}"),
      "\\p{javaLowerCase}");
  // \Q...\E contents are copied verbatim (the '\p' inside is not expanded).
  EXPECT_EQ(
      expandJavaUnicodePropertiesForRe2("\\Q\\p{InGreek}\\E"),
      "\\Q\\p{InGreek}\\E");
}

namespace {
// Returns true if 'pattern' compiles as a valid RE2 program, i.e. it would be
// offloaded to native rather than falling back to Spark.
bool re2Compiles(const std::string& pattern) {
  RE2 re(pattern, RE2::Quiet);
  return re.ok();
}

// The pattern preparation used by native execution BEFORE this change
// (named-group rewrite only).
//
// NOTE: this is the *scanner* form of the named-group rewrite, not the legacy
// RE2::GlobalReplace("[(][?]<([^>]*)>") in prepareRegexpReplacePattern(). The
// legacy form is not escape-, class- or \Q-aware and corrupts patterns where
// '(?<' is literal text: "\Q(?<g>\E" -> "\Q(?P<g>\E", "[(?<g>]" -> "[(?P<g>]",
// "\(?<g>" -> "\(?P<g>". Those are pre-existing defects that this library
// fixes, so the baseline here deliberately models the corrected rewrite; the
// widening-only invariant below then covers the stages layered on top of it.
std::string prepBefore(const std::string& p) {
  return rewriteJavaNamedGroups(p);
}

// The pattern preparation used by native execution AFTER this change
// (Java->RE2 normalization and ICU-free property expansion on top of the
// named-group rewrite).
std::string prepAfter(const std::string& p) {
  return translateJavaRegexToRe2(p);
}
} // namespace

// Measures how many java.util.regex patterns become offloadable to native RE2
// because of normalizeJavaRegexForRe2(). Emits a before/after report and
// guards against regression. This is a fast, cluster-free coverage proxy:
// "offloadable" == "the prepared pattern compiles in RE2".
struct Case {
  const char* feature;
  std::string pattern;
};

// Representative corpus spanning the features this change targets, plus
// controls that must NOT change (already-supported and RE2-irreducible).
const std::vector<Case>& corpus() {
  static const std::vector<Case> kCorpus = {
      // \uXXXX escapes: rejected by RE2 today, offloadable after.
      {"unicode-escape", "\\u0041"},
      {"unicode-escape", "code=\\u00e9"},
      {"unicode-escape", "[\\u0030-\\u0039]+"},
      {"unicode-escape", "\\u0041\\u0042\\u0043"},
      // UTF-16 surrogate pairs: rejected by RE2 today, offloadable after
      // folding into the supplementary code point (e.g. emoji U+1F600).
      {"unicode-surrogate", "\\uD83D\\uDE00"},
      {"unicode-surrogate", "prefix-\\uD83D\\uDE00-suffix"},
      {"unicode-surrogate", "[\\uD83D\\uDE00\\uD83D\\uDE01]"},
      {"unicode-escape", "\\u00e9t\\u00e9"},
      {"unicode-escape", "[\\u4e00-\\u9fff]"},
      {"unicode-escape", "x\\u0041?y"},
      {"unicode-surrogate", "[\\uD83C\\uDF09]"},
      {"unicode-surrogate", "\\uD83D\\uDE00+"},
      {"unicode-property", "\\p{InCyrillic}"},
      {"unicode-property", "\\p{block=Hiragana}"},
      {"unicode-property", "\\p{IsLatin}+"},
      {"unicode-property", "\\p{L1}"},
      {"unicode-property", "\\p{LC}"},
      {"unicode-property", "[\\p{InGreek}\\p{InCyrillic}]"},
      {"whitespace", "[\\S]+"},
      {"whitespace", "a\\s+b"},
      {"whitespace", "[\\s,;]{2,}"},
      {"whitespace", "\\s+ERROR"},
      {"whitespace", "a\\Sb"},
      {"whitespace", "[^\\S]"},
      {"named-group", "(?<year>\\d{4})-(?<month>\\d{2})"},
      // Quoted literal text: RE2 already accepts this, so the widening-only
      // invariant requires it to survive translation byte-for-byte.
      {"named-group", "\\Q(?<g>\\E"},
      {"plain", "^abc$"},
      {"plain", "[0-9]{2,4}"},
      {"lookaround", "(?<=USD)\\d+"},
      {"lookaround", "(?!foo)bar"},
      {"backref", "(a)\\1"},
      // Java Unicode properties: blocks, script/category renames and aliases
      // become RE2-expressible (ICU-free), offloadable after.
      {"unicode-property", "\\p{InGreek}+"},
      {"unicode-property", "a\\P{InGreek}b"},
      {"unicode-property", "\\p{IsL}\\p{IsGreek}"},
      {"unicode-property", "\\p{ASCII}{3}"},
      {"unicode-property", "\\p{LC}+"},
      // Controls: plain patterns already offloadable (must stay offloadable).
      {"plain", "abc\\d+"},
      {"plain", "[A-Za-z0-9_]{3,10}"},
      {"named-group", "(?<y>\\d{4})-(?<m>\\d{2})"},
      // Controls: RE2-irreducible, must remain NON-offloadable (fall back).
      {"lookaround", "foo(?=bar)"},
      {"backref", "(a)\\1"},
  };
  return kCorpus;
}

TEST(JavaRegexCoverageTest, offloadDeltaOverCorpus) {
  int before = 0;
  int after = 0;
  std::map<std::string, std::pair<int, int>>
      byFeature; // feature -> {before,after}
  for (const auto& c : corpus()) {
    const bool b = re2Compiles(prepBefore(c.pattern));
    const bool a = re2Compiles(prepAfter(c.pattern));
    before += b ? 1 : 0;
    after += a ? 1 : 0;
    byFeature[c.feature].first += b ? 1 : 0;
    byFeature[c.feature].second += a ? 1 : 0;
    // A normalized pattern must never lose offloadability.
    EXPECT_TRUE(!b || a) << "regressed: " << c.pattern;
  }

  std::cout << "\n[RE2 offload coverage] corpus=" << corpus().size()
            << " offloadable before=" << before << " after=" << after << " (+"
            << (after - before) << ")\n";
  for (const auto& [feature, counts] : byFeature) {
    std::cout << "  " << feature << ": " << counts.first << " -> "
              << counts.second << "\n";
  }

  // The whole point of the change: coverage strictly increases.
  EXPECT_GT(after, before);
  // Controls: irreducible features stay non-offloadable (2 of them here).
  EXPECT_FALSE(re2Compiles(prepAfter("foo(?=bar)")));
  EXPECT_FALSE(re2Compiles(prepAfter("(a)\\1")));
}

// The core safety property: the translator is widening-only. Any pattern RE2
// already accepts must come out byte-identical, so no query that offloads
// today can change its results. Everything the translator rewrites is a
// construct RE2 rejects, which today forces a fallback.
TEST(JavaRegexCoverageTest, translationIsWideningOnly) {
  for (const auto& c : corpus()) {
    const auto today = prepBefore(c.pattern);
    if (!re2Compiles(today)) {
      continue;
    }
    EXPECT_EQ(prepAfter(c.pattern), today)
        << "pattern: " << c.pattern << " (" << c.feature << ")";
  }
}

// Named-group rewriting must not corrupt lookbehind, which shares the '(?<'
// prefix but is not a group name.
TEST(JavaRegexTranslatorTest, namedGroups) {
  EXPECT_EQ(rewriteJavaNamedGroups("(?<name>a)"), "(?P<name>a)");
  EXPECT_EQ(rewriteJavaNamedGroups("(?<n1>a)(?<n2>b)"), "(?P<n1>a)(?P<n2>b)");

  // Lookbehind: left verbatim so RE2 rejects it, rather than being mangled
  // into a bogus group name.
  EXPECT_EQ(rewriteJavaNamedGroups("(?<=a)b>c"), "(?<=a)b>c");
  EXPECT_EQ(rewriteJavaNamedGroups("(?<!x)y>z"), "(?<!x)y>z");

  // Not a valid group name, so not a named group.
  EXPECT_EQ(rewriteJavaNamedGroups("(?<a b>x)"), "(?<a b>x)");
  EXPECT_EQ(rewriteJavaNamedGroups("(?<1a>x)"), "(?<1a>x)");
  EXPECT_EQ(rewriteJavaNamedGroups("(?<>x)"), "(?<>x)");

  // '(' is a literal inside a character class, and an escaped '(' is a
  // literal anywhere.
  EXPECT_EQ(rewriteJavaNamedGroups("[(?<a>]"), "[(?<a>]");
  EXPECT_EQ(rewriteJavaNamedGroups("\\(?<a>"), "\\(?<a>");

  // \Q...\E quotes literal text, so a '(?<' inside a quoted run opens no
  // group. Rewriting it would silently change what the pattern matches:
  // '\Q(?<g>\E' matches the literal five characters "(?<g>", which RE2
  // already accepts verbatim.
  EXPECT_EQ(rewriteJavaNamedGroups("\\Q(?<g>\\E"), "\\Q(?<g>\\E");
  EXPECT_EQ(rewriteJavaNamedGroups("\\Q(?<g>"), "\\Q(?<g>");
  EXPECT_EQ(
      rewriteJavaNamedGroups("\\Q(?<g>\\E(?<real>a)"),
      "\\Q(?<g>\\E(?P<real>a)");

  // Already-RE2 spelling is untouched, so translation is idempotent.
  EXPECT_EQ(rewriteJavaNamedGroups("(?P<name>a)"), "(?P<name>a)");
}

// The legacy RE2::GlobalReplace("[(][?]<([^>]*)>") in
// prepareRegexpReplacePattern() is not escape-, class- or \Q-aware, so it
// rewrites '(?<' occurrences that are literal text and changes what the
// pattern matches. These three cases are the ones it gets wrong; pinning them
// keeps the scanner from regressing to the old text-substitution behaviour.
TEST(JavaRegexTranslatorTest, namedGroupRewriteFixesLegacyCorruption) {
  // Quoted literal: matches the characters "(?<g>", not a group.
  EXPECT_EQ(rewriteJavaNamedGroups("\\Q(?<g>\\E"), "\\Q(?<g>\\E");
  // Inside a character class '(' is an ordinary member, not a group opener.
  EXPECT_EQ(rewriteJavaNamedGroups("[(?<g>]"), "[(?<g>]");
  // An escaped '(' is a literal paren.
  EXPECT_EQ(rewriteJavaNamedGroups("\\(?<g>"), "\\(?<g>");

  // RE2 accepts all three unchanged, so the corrupted spellings the legacy
  // path produced were silently matching the wrong text rather than erroring.
  for (const auto& p : {"\\Q(?<g>\\E", "[(?<g>]", "\\(?<g>"}) {
    EXPECT_TRUE(re2Compiles(p)) << p;
    EXPECT_EQ(translateJavaRegexToRe2(p), p) << p;
  }
}

// The full composition must be idempotent: translating an already-translated
// pattern is a no-op. Guards against double application by callers.
TEST(JavaRegexTranslatorTest, idempotent) {
  for (const auto& pattern :
       {"\\u0041",
        "\\s",
        "\\S",
        "[\\S]",
        "(?<n>a)",
        "\\p{InGreek}",
        "\\p{ASCII}",
        "[\\s\\S]",
        "(?<=a)b"}) {
    const auto once = translateJavaRegexToRe2(pattern);
    EXPECT_EQ(translateJavaRegexToRe2(once), once) << "pattern: " << pattern;
  }
}

} // namespace
} // namespace facebook::velox::functions
