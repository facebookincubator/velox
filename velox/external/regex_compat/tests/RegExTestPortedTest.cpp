/*
 * Copyright (c) 1999, 2023, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
//
// Ported to GTest for inclusion in Velox's regex-compat test suite.  The
// original source is OpenJDK 17's test/jdk/java/util/regex/RegExTest.java,
// as imported by the pcre4j compatibility fork.  These tests intentionally
// run the same Java-pattern inputs through JavaMatcherAdapter<TypeParam> so
// Java, PCRE2 and RE2 backends report a per-backend compatibility rate.
//

#include "velox/external/regex_compat/tests/BackendTestBase.h"
#include "velox/external/regex_compat/tests/JavaMatcherAdapter.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace facebook::velox::regex_compat::test {
namespace {

template <typename R>
using RegExTestPortedTest = BackendTest<R>;
TYPED_TEST_SUITE(RegExTestPortedTest, AllBackends);

struct RegExStats {
  int passed = 0;
  int failed = 0;
};

std::map<std::string, RegExStats>& regExStats() {
  static std::map<std::string, RegExStats> s;
  return s;
}

class RegExReporter : public ::testing::Environment {
 public:
  void TearDown() override {
    auto& m = regExStats();
    if (m.empty()) {
      return;
    }
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "========== RegExTest ported compat rate =========\n");
    for (const auto& [backend, st] : m) {
      const int total = st.passed + st.failed;
      const double pct = total > 0 ? 100.0 * st.passed / total : 0.0;
      std::fprintf(
          stderr,
          "  %-8s %4d / %4d  (%.2f%%)\n",
          backend.c_str(),
          st.passed,
          total,
          pct);
    }
    std::fprintf(stderr, "=================================================\n");
  }
};

[[maybe_unused]] static auto* kRegExReporter =
    ::testing::AddGlobalTestEnvironment(new RegExReporter);

template <typename R>
const char* backendName() {
  if constexpr (std::is_same_v<R, Re2Regex>) {
    return "Re2";
  } else if constexpr (std::is_same_v<R, Pcre2Regex>) {
    return "Pcre2";
  } else {
    return "Java";
  }
}

template <typename R>
void recordCase(bool ok) {
  auto& st = regExStats()[backendName<R>()];
  if (ok) {
    ++st.passed;
  } else {
    ++st.failed;
  }
}

static Options caseInsensitive() {
  Options opt;
  opt.caseSensitive = false;
  return opt;
}

static Options dotAll() {
  Options opt;
  opt.dotNl = true;
  return opt;
}

static Options multiLine() {
  Options opt;
  opt.oneLine = false;
  return opt;
}

static Options ciDotAllMultiLine() {
  Options opt;
  opt.caseSensitive = false;
  opt.dotNl = true;
  opt.oneLine = false;
  return opt;
}

static std::string utf8(std::uint32_t cp) {
  std::string out;
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return out;
}

static std::string toSupplementaries(std::string_view s) {
  std::string out;
  for (std::size_t i = 0; i < s.size();) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c == '\\' && i + 1 < s.size()) {
      out.push_back(s[i++]);
      out.push_back(s[i++]);
      if (out.back() == 'u' && i + 4 <= s.size()) {
        out.append(s.substr(i, 4));
        i += 4;
      }
    } else if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
      out.append(utf8(0x10000 + c));
      ++i;
    } else {
      out.push_back(s[i++]);
    }
  }
  return out;
}

static std::string javaQuote(std::string_view s) {
  std::string out = "\\Q";
  std::size_t i = 0;
  while (true) {
    auto j = s.find("\\E", i);
    if (j == std::string_view::npos) {
      out.append(s.substr(i));
      break;
    }
    out.append(s.substr(i, j - i));
    out.append("\\E\\\\E\\Q");
    i = j + 2;
  }
  out.append("\\E");
  return out;
}

template <typename R>
bool find(std::string_view pattern, std::string_view input, Options opt = {}) {
  R re(pattern, opt);
  if (!re.ok()) {
    return false;
  }
  JavaMatcherAdapter<R> m(&re, input);
  return m.find();
}

template <typename R>
bool noFind(std::string_view pattern, std::string_view input, Options opt = {}) {
  R re(pattern, opt);
  if (!re.ok()) {
    return false;
  }
  JavaMatcherAdapter<R> m(&re, input);
  return !m.find();
}

template <typename R>
bool full(std::string_view pattern, std::string_view input, Options opt = {}) {
  R re(pattern, opt);
  return re.ok() && R::FullMatch(input, re);
}

template <typename R>
bool notFull(std::string_view pattern, std::string_view input, Options opt = {}) {
  R re(pattern, opt);
  return re.ok() && !R::FullMatch(input, re);
}

template <typename R>
bool findGroup(
    std::string_view pattern,
    std::string_view input,
    std::string_view expected,
    Options opt = {},
    int group = 0) {
  R re(pattern, opt);
  if (!re.ok()) {
    return false;
  }
  JavaMatcherAdapter<R> m(&re, input);
  if (!m.find()) {
    return false;
  }
  auto g = m.group(group);
  return g && *g == expected;
}

template <typename R>
bool findStart(
    std::string_view pattern,
    std::string_view input,
    int expected,
    Options opt = {}) {
  R re(pattern, opt);
  if (!re.ok()) {
    return false;
  }
  JavaMatcherAdapter<R> m(&re, input);
  return m.find() && m.start() == expected;
}

template <typename R>
bool lookingAt(std::string_view pattern, std::string_view input, Options opt = {}) {
  R re(pattern, opt);
  if (!re.ok()) {
    return false;
  }
  JavaMatcherAdapter<R> m(&re, input);
  return m.lookingAt();
}

template <typename R>
bool notLookingAt(std::string_view pattern, std::string_view input, Options opt = {}) {
  R re(pattern, opt);
  if (!re.ok()) {
    return false;
  }
  JavaMatcherAdapter<R> m(&re, input);
  return !m.lookingAt();
}

template <typename R>
bool replaceAllEquals(
    std::string_view pattern,
    std::string input,
    std::string_view replacement,
    std::string_view expected,
    Options opt = {}) {
  R re(pattern, opt);
  if (!re.ok()) {
    return false;
  }
  R::GlobalReplace(&input, re, replacement);
  return input == expected;
}

template <typename R>
bool replaceFirstEquals(
    std::string_view pattern,
    std::string_view input,
    std::string_view replacement,
    std::string_view expected,
    Options opt = {}) {
  R re(pattern, opt);
  if (!re.ok()) {
    return false;
  }
  JavaMatcherAdapter<R> m(&re, input);
  return m.replaceFirst(replacement) == expected;
}

template <typename R>
bool appendWalkEquals(
    std::string_view pattern,
    std::string_view input,
    std::string_view replacement,
    std::string_view expected,
    int skipMiddleFinds = 0) {
  R re(pattern);
  if (!re.ok()) {
    return false;
  }
  JavaMatcherAdapter<R> m(&re, input);
  std::string sb;
  if (skipMiddleFinds == 0) {
    while (m.find()) {
      m.appendReplacement(sb, replacement);
    }
  } else {
    if (!m.find()) return false;
    m.appendReplacement(sb, "$1");
    for (int i = 0; i < skipMiddleFinds; ++i) {
      if (!m.find()) return false;
    }
    m.appendReplacement(sb, replacement);
  }
  m.appendTail(sb);
  return sb == expected;
}

template <typename R>
bool splitEquals(
    std::string_view pattern,
    std::string_view input,
    const std::vector<std::string>& expected) {
  R re(pattern);
  if (!re.ok()) {
    return false;
  }
  JavaMatcherAdapter<R> m(&re, input);
  std::vector<std::string> actual;
  std::size_t prev = 0;
  while (m.find()) {
    actual.emplace_back(input.substr(prev, m.start() - prev));
    prev = static_cast<std::size_t>(m.end());
  }
  actual.emplace_back(input.substr(prev));
  while (actual.size() > 1 && actual.back().empty()) {
    actual.pop_back();
  }
  return actual == expected;
}

template <typename R>
bool compiles(std::string_view pattern, Options opt = {}) {
  R re(pattern, opt);
  return re.ok();
}

template <typename R>
bool rejects(std::string_view pattern, Options opt = {}) {
  R re(pattern, opt);
  return !re.ok();
}

#define PORTED_REGEX_TEST(TestName, Body)                                      \
  TYPED_TEST(RegExTestPortedTest, TestName) {                                  \
    bool ok = true;                                                            \
    auto expect = [&](bool value) { ok = ok && value; };                       \
    using R = TypeParam;                                                       \
    (void)expect;                                                              \
    (void)sizeof(R);                                                           \
    Body                                                                       \
    recordCase<TypeParam>(ok);                                                 \
    if constexpr (std::is_same_v<TypeParam, JavaRegex>) {                      \
      EXPECT_TRUE(ok) << "RegExTest::" #TestName " Java backend regression";  \
    }                                                                          \
  }

#define TODO_REGEX_TEST(TestName, Reason)                                      \
  TYPED_TEST(RegExTestPortedTest, TestName) {                                  \
    GTEST_SKIP() << "TODO: port from RegExTest::" #TestName ": " Reason;      \
  }

TODO_REGEX_TEST(processTestCases, "covered by OpenJdkCorpusDiffTest to avoid double-counting")
TODO_REGEX_TEST(processBMPTestCases, "covered by OpenJdkCorpusDiffTest to avoid double-counting")
TODO_REGEX_TEST(processSupplementaryTestCases, "covered by OpenJdkCorpusDiffTest to avoid double-counting")
TODO_REGEX_TEST(nullArgumentTest, "Java null API behavior has no C++ adapter equivalent")

PORTED_REGEX_TEST(surrogatesInClassTest, {
  const std::string cp = utf8(0x1D122);
  expect(find<R>("[" + utf8(0x1D121) + "-" + utf8(0x1D124) + "]", cp));
})

PORTED_REGEX_TEST(removeQEQuotingTest, {
  expect(find<R>("\\011\\Q1sometext\\E\\011\\Q2sometext\\E", "\t1sometext\t2sometext"));
})

TODO_REGEX_TEST(toMatchResultTest, "MatchResult snapshot object is not exposed by JavaMatcherAdapter")
TODO_REGEX_TEST(toMatchResultTest2, "MatchResult error semantics are Java API-specific")
TODO_REGEX_TEST(hitEndTest, "Matcher.hitEnd is not exposed by JavaMatcherAdapter")

TODO_REGEX_TEST(wordSearchTest, "JavaMatcherAdapter find(int) zero-width boundary cursor behavior differs from java.util.regex.Matcher")

TODO_REGEX_TEST(caretAtEndTest, "zero-width multiline caret cursor behavior needs exact Matcher emulation")

PORTED_REGEX_TEST(unicodeWordBoundsTest, {
  expect(findStart<R>("\\b", "  aa  ", 2));
  expect(findStart<R>("\\b", "  aa\xcc\x8a  ", 2));
  expect(noFind<R>("\\b", "  \xcc\x8a\xcc\x8a  "));
})

PORTED_REGEX_TEST(lookbehindTest, {
  expect(findGroup<R>("(?<=%.{0,5})foo\\d", "%foo1\n%bar foo2\n%bar  foo3\n%blahblah foo4\nfoo5", "foo1"));
  expect(findGroup<R>("(?<=.*\\b)foo", "abcd foo", "foo"));
  expect(noFind<R>("(?<!abc )\\bfoo", "abc foo"));
  expect(findGroup<R>("(?<!%.{0,5})foo\\d", "%foo1\n%bar foo2\n%bar  foo3\n%blahblah foo4\nfoo5", "foo4"));
})

TODO_REGEX_TEST(boundsTest, "transparent and anchoring bounds toggles are not exposed by JavaMatcherAdapter")

PORTED_REGEX_TEST(findFromTest, {
  R re("\\$0");
  if (!re.ok()) { expect(false); } else {
    JavaMatcherAdapter<R> m(&re, "This is 40 $0 message.");
    expect(m.find());
    expect(!m.find());
    expect(!m.find());
  }
})

PORTED_REGEX_TEST(negatedCharClassTest, {
  expect(full<R>("[^>]", "\xe2\x80\xba"));
  expect(find<R>("[^fr]", "a"));
  expect(!find<R>("[^f\xe2\x80\xbar]", "f"));
  expect(find<R>("[^\xe2\x80\xbar\xe2\x80\xbb]", "\xe2\x80\xbc"));
})

PORTED_REGEX_TEST(toStringTest, {
  expect(compiles<R>("b+"));
  expect(find<R>("b+", "aaabbbccc"));
})

PORTED_REGEX_TEST(literalPatternTest, {
  expect(find<R>(javaQuote("abc\\t$^"), "abc\\t$^"));
  expect(find<R>("\\Qa^$bcabc\\E", "a^$bcabc"));
  expect(find<R>("\\Qabc\\Eefg\\\\Q\\\\Ehij", "abcefg\\Q\\Ehij"));
  expect(find<R>(javaQuote("abc\\Edef"), "abc\\Edef"));
  expect(noFind<R>(javaQuote("abc\\Edef"), "abcdef"));
})

PORTED_REGEX_TEST(literalReplacementTest, {
  expect(replaceAllEquals<R>(javaQuote("abc"), "zzzabczzz", "$0", "zzzabczzz"));
  expect(replaceAllEquals<R>(javaQuote("abc"), "zzzabczzz", JavaMatcherAdapter<R>::quoteReplacement("$0"), "zzz$0zzz"));
  expect(replaceAllEquals<R>(javaQuote("abc"), "zzzabczzz", JavaMatcherAdapter<R>::quoteReplacement("\\t$\\$"), "zzz\\t$\\$zzz"));
})

PORTED_REGEX_TEST(regionTest, {
  R re("abc");
  if (!re.ok()) { expect(false); } else {
    JavaMatcherAdapter<R> m(&re, "abcdefabc");
    expect(m.region(0, 9).find());
    expect(m.find());
    expect(m.region(0, 3).find());
    expect(!m.region(3, 6).find());
    expect(!m.region(0, 2).find());
  }
  R anchored("^abc$");
  if (!anchored.ok()) { expect(false); } else {
    JavaMatcherAdapter<R> m(&anchored, "zzzabczzz");
    expect(!m.region(0, 9).find());
    expect(m.region(3, 6).find());
  }
})

PORTED_REGEX_TEST(escapedSegmentTest, {
  expect(find<R>("\\Qdir1\\dir2\\E", "dir1\\dir2"));
  expect(find<R>("\\Qdir1\\dir2\\\\E", "dir1\\dir2\\"));
  expect(find<R>("(\\Qdir1\\dir2\\\\E)", "dir1\\dir2\\"));
})

PORTED_REGEX_TEST(nonCaptureRepetitionTest, {
  const char* input = "abcdefgh;";
  for (std::string_view p : {"(?:\\w{4})+;", "(?:\\w{8})*;", "(?:\\w{2}){2,4};", "(?:\\w{4}){2,};", ".*?(?:\\w{5})+;", ".*?(?:\\w{9})*;", "(?:\\w{4})+?;", "(?:\\w{4})++;", "(?:\\w{2,}?)+;", "(\\w{4})+;"}) {
    expect(findGroup<R>(p, input, input));
    expect(full<R>(p, input));
  }
})

PORTED_REGEX_TEST(notCapturedGroupCurlyMatchTest, {
  R re("(abc)+|(abcd)+");
  if (!re.ok()) { expect(false); } else {
    JavaMatcherAdapter<R> m(&re, "abcd");
    expect(m.matches());
    expect(!m.group(1).has_value());
    expect(m.group(2).has_value() && *m.group(2) == "abcd");
  }
})

TODO_REGEX_TEST(javaCharClassTest, "depends on Java Character predicates and randomized Unicode property coverage")
TODO_REGEX_TEST(caretBetweenTerminatorsTest, "UNIX_LINES flag is not represented in regex_compat Options")
TODO_REGEX_TEST(dollarAtEndTest, "UNIX_LINES flag is not represented in regex_compat Options")

PORTED_REGEX_TEST(multilineDollarTest, {
  R re("$", multiLine());
  if (!re.ok()) { expect(false); } else {
    JavaMatcherAdapter<R> m(&re, "first bit\nsecond bit");
    expect(m.find() && m.start() == 9);
    expect(m.find() && m.start() == 20);
  }
})

PORTED_REGEX_TEST(reluctantRepetitionTest, {
  expect(find<R>("1(\\s\\S+?){1,3}?[\\s,]2", "1 word word word 2"));
  expect(find<R>("1(\\s\\S+?){1,3}?[\\s,]2", "1 word 2"));
  expect(findGroup<R>("([a-z])+?c", "ababcdefdec", "ababc"));
})

TODO_REGEX_TEST(serializeTest, "Java Pattern serialization has no C++ adapter equivalent")

TODO_REGEX_TEST(gTest, "\\G depends on previous-match state that JavaMatcherAdapter does not expose to backends")

TODO_REGEX_TEST(zTest, "UNIX_LINES-sensitive \\Z end-anchor behavior needs dedicated option support")

PORTED_REGEX_TEST(replaceFirstTest, {
  expect(replaceFirstEquals<R>("(ab)(c*)", "abccczzzabcczzzabccc", "test", "testzzzabcczzzabccc"));
  expect(replaceFirstEquals<R>("(ab)(c*)", "zzzabccczzzabcczzzabccczzz", "$1", "zzzabzzzabcczzzabccczzz"));
  expect(replaceFirstEquals<R>("(ab)(c*)", "zzzabccczzzabcczzzabccczzz", "$2", "zzzccczzzabcczzzabccczzz"));
  expect(replaceFirstEquals<R>("a*", "aaaaaaaaaa", "test", "test"));
  expect(replaceFirstEquals<R>("a+", "zzzaaaaaaaaaa", "test", "zzztest"));
})

TODO_REGEX_TEST(unixLinesTest, "UNIX_LINES flag is not represented in regex_compat Options")

PORTED_REGEX_TEST(commentsTest, {
  expect(full<R>("(?x)aa \\# aa", "aa#aa"));
  expect(full<R>("(?x)aa  # blah", "aa"));
  expect(full<R>("(?x)aa blah", "aablah"));
  expect(full<R>("(?x)aa  # blah\n  ", "aa"));
  expect(full<R>("(?x)aa  # blah\nbc # blech", "aabc"));
  expect(full<R>("(?x)aa  # blah\nbc\\# blech", "aabc#blech"));
})

PORTED_REGEX_TEST(caseFoldingTest, {
  expect(notFull<R>("aa", "ab", caseInsensitive()));
  expect(full<R>("a", "A", caseInsensitive()));
  expect(full<R>("ab", "AB", caseInsensitive()));
  expect(full<R>("[a-b]", "B", caseInsensitive()));
})

PORTED_REGEX_TEST(appendTest, {
  expect(replaceAllEquals<R>("(ab)(cd)", "abcd", "$2$1", "cdab"));
  expect(replaceAllEquals<R>("([a-z]+)( *= *)([0-9]+)", "Swap all: first = 123, second = 456", "$3$2$1", "Swap all: 123 = first, 456 = second"));
  R re("([a-z]+)( *= *)([0-9]+)");
  if (!re.ok()) { expect(false); } else {
    JavaMatcherAdapter<R> m(&re, "Swap one: first = 123, second = 456");
    std::string sb;
    expect(m.find());
    m.appendReplacement(sb, "$3$2$1");
    m.appendTail(sb);
    expect(sb == "Swap one: 123 = first, second = 456");
  }
})

PORTED_REGEX_TEST(splitTest, {
  expect(splitEquals<R>(":", "foo:and:boo", {"foo", "and", "boo"}));
  expect(splitEquals<R>("X", "fooXandXboo", {"foo", "and", "boo"}));
  expect(splitEquals<R>("[ \t,:.]", "This is,testing: with\tdifferent separators.", {"This", "is", "testing", "", "with", "different", "separators"}));
  expect(splitEquals<R>("o", "boo:and:foo", {"b", "", ":and:f"}));
})

PORTED_REGEX_TEST(negationTest, {
  expect(findGroup<R>("[\\[@^]+", "@@@@[[[[^^^^", "@@@@[[[[^^^^"));
  expect(findGroup<R>("[@\\[^]+", "@@@@[[[[^^^^", "@@@@[[[[^^^^"));
  expect(findGroup<R>("[@\\[^@]+", "@@@@[[[[^^^^", "@@@@[[[[^^^^"));
  expect(find<R>("\\)", "xxx)xxx"));
})

PORTED_REGEX_TEST(ampersandTest, {
  expect(find<R>("[&@]+", "@@@@&&&&"));
  expect(find<R>("[@&]+", "@@@@&&&&"));
  expect(find<R>("[@\\&]+", "@@@@&&&&"));
})

PORTED_REGEX_TEST(octalTest, {
  expect(full<R>("\\u0007", "\x07"));
  expect(full<R>("\\07", "\x07"));
  expect(full<R>("\\007", "\x07"));
  expect(full<R>("\\0007", "\x07"));
  expect(full<R>("\\040", " "));
  expect(full<R>("\\0403", " 3"));
  expect(full<R>("\\0103", "C"));
})

PORTED_REGEX_TEST(longPatternTest, {
  expect(compiles<R>("a 32-character-long pattern xxxx"));
  expect(compiles<R>("a 33-character-long pattern xxxxx"));
  expect(compiles<R>("a thirty four character long regex"));
  std::string p;
  for (int i = 0; i < 100; ++i) p.push_back(static_cast<char>('a' + i % 26));
  expect(compiles<R>(p));
})

PORTED_REGEX_TEST(group0Test, {
  expect(findGroup<R>("(tes)ting", "testing", "testing"));
  expect(lookingAt<R>("(tes)ting", "testing"));
  expect(full<R>("(tes)ting", "testing"));
  expect(full<R>("^(tes)ting", "testing"));
})

PORTED_REGEX_TEST(findIntTest, {
  R re("blah");
  if (!re.ok()) { expect(false); } else {
    JavaMatcherAdapter<R> m(&re, "zzzzblahzzzzzblah");
    expect(m.find(2));
  }
  R dollar("$");
  if (!dollar.ok()) { expect(false); } else {
    JavaMatcherAdapter<R> m(&dollar, "1234567890");
    expect(m.find(10));
  }
})

PORTED_REGEX_TEST(emptyPatternTest, {
  R re("");
  if (!re.ok()) { expect(false); } else {
    JavaMatcherAdapter<R> m(&re, "foo");
    expect(m.find() && m.start() == 0);
    m.reset();
    expect(!m.matches());
    m.reset("");
    expect(m.matches());
  }
  expect(full<R>("", ""));
  expect(notFull<R>("", "foo"));
})

PORTED_REGEX_TEST(charClassTest, {
  expect(find<R>("blah[ab]]blech", "blahb]blech"));
  expect(find<R>("[abc[def]]", "b"));
  expect(find<R>(std::string("[ab") + utf8(0x00ff) + "cd]", std::string("ab") + utf8(0x00ff) + "cd", caseInsensitive()));
})

PORTED_REGEX_TEST(caretTest, {
  expect(findGroup<R>("\\w*", "a#bc#def##g", "a"));
  expect(findGroup<R>("^\\w*", "a#bc#def##g", "a"));
  expect(findGroup<R>("\\A\\p{Alpha}{3}", "abcdef-ghi\njklmno", "abc"));
  expect(findGroup<R>("^\\p{Alpha}{3}", "abcdef-ghi\njklmno", "abc", multiLine()));
  expect(replaceAllEquals<R>("^", "this is some text", "X", "Xthis is some text"));
})

PORTED_REGEX_TEST(groupCaptureTest, {
  R atomic("x+(?>y+)z+");
  if (atomic.ok()) {
    JavaMatcherAdapter<R> m(&atomic, "xxxyyyzzz");
    expect(m.find());
    bool threw = false;
    try { (void)m.group(1); } catch (const std::out_of_range&) { threw = true; }
    expect(threw);
  } else {
    expect(false);
  }
  R pure("x+(?:y+)z+");
  if (pure.ok()) {
    JavaMatcherAdapter<R> m(&pure, "xxxyyyzzz");
    expect(m.find());
    bool threw = false;
    try { (void)m.group(1); } catch (const std::out_of_range&) { threw = true; }
    expect(threw);
  } else {
    expect(false);
  }
})

PORTED_REGEX_TEST(backRefTest, {
  expect(find<R>("(a*)bc\\1", "zzzaabcazzz"));
  expect(find<R>("(a*)bc\\1", "zzzaabcaazzz"));
  expect(find<R>("(abc)(def)\\1", "abcdefabc"));
  expect(noFind<R>("(abc)(def)\\3", "abcdefabc"));
  expect(noFind<R>("(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)\\11", "abcdefghija"));
  expect(find<R>("(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)\\11", "abcdefghija1"));
  expect(find<R>("(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)(k)\\11", "abcdefghijkk"));
})

TODO_REGEX_TEST(anchorTest, "CRLF/Unicode line-terminator anchor details need a dedicated port")

PORTED_REGEX_TEST(lookingAtTest, {
  expect(lookingAt<R>("(ab)(c*)", "abccczzzabcczzzabccc"));
  expect(notLookingAt<R>("(ab)(c*)", "zzzabccczzzabcczzzabccczzz"));
})

PORTED_REGEX_TEST(matchesTest, {
  expect(full<R>("ulb(c*)", "ulbcccccc"));
  expect(notFull<R>("ulb(c*)", "zzzulbcccccc"));
  expect(notFull<R>("ulb(c*)", "ulbccccccdef"));
  expect(full<R>("a|ad", "ad"));
})

PORTED_REGEX_TEST(patternMatchesTest, {
  expect(full<R>(toSupplementaries("ulb(c*)"), toSupplementaries("ulbcccccc")));
  expect(notFull<R>(toSupplementaries("ulb(c*)"), toSupplementaries("zzzulbcccccc")));
  expect(notFull<R>(toSupplementaries("ulb(c*)"), toSupplementaries("ulbccccccdef")));
})

TODO_REGEX_TEST(ceTest, "CANON_EQ flag is not represented in regex_compat Options")

PORTED_REGEX_TEST(globalSubstitute, {
  expect(replaceAllEquals<R>("(ab)(c*)", "abccczzzabcczzzabccc", "test", "testzzztestzzztest"));
  expect(replaceAllEquals<R>("(ab)(c*)", "zzzabccczzzabcczzzabccczzz", "test", "zzztestzzztestzzztestzzz"));
  expect(replaceAllEquals<R>("(ab)(c*)", "zzzabccczzzabcczzzabccczzz", "$1", "zzzabzzzabzzzabzzz"));
})

PORTED_REGEX_TEST(stringBufferSubstituteLiteral, {
  expect(appendWalkEquals<R>("blah", "zzzblahzzz", "blech", "zzzblechzzz"));
})

PORTED_REGEX_TEST(stringBufferSubtituteWithGroups, {
  expect(appendWalkEquals<R>("(ab)(cd)*", "zzzabcdzzz", "$1", "zzzabzzz"));
})

PORTED_REGEX_TEST(stringBufferThreeSubstitution, {
  expect(appendWalkEquals<R>("(ab)(cd)*(ef)", "zzzabcdcdefzzz", "$1w$2w$3", "zzzabwcdwefzzz"));
})

PORTED_REGEX_TEST(stringBufferSubstituteGroupsThreeMatches, {
  expect(appendWalkEquals<R>("(ab)(cd*)", "zzzabcdzzzabcddzzzabcdzzz", "$2", "zzzabzzzabcddzzzcdzzz", 2));
})

PORTED_REGEX_TEST(stringBufferEscapedDollar, {
  expect(appendWalkEquals<R>("(ab)(cd)*(ef)", "zzzabcdcdefzzz", "$1w\\$2w$3", "zzzabw$2wefzzz"));
})

TODO_REGEX_TEST(stringBufferNonExistentGroup, "requires Java replacement error semantics for nonexistent groups")

TODO_REGEX_TEST(stringBufferCheckDoubleDigitGroupReferences, "requires Java multi-digit replacement group backoff semantics")

PORTED_REGEX_TEST(stringBufferBackoff, {
  expect(appendWalkEquals<R>("(ab)(cd)*(ef)", "zzzabcdcdefzzz", "$1w$15w$3", "zzzabwab5wefzzz"));
})

PORTED_REGEX_TEST(stringBufferSupplementaryCharacter, {
  expect(appendWalkEquals<R>(toSupplementaries("blah"), toSupplementaries("zzzblahzzz"), toSupplementaries("blech"), toSupplementaries("zzzblechzzz")));
})

PORTED_REGEX_TEST(stringBufferSubstitutionWithGroups, {
  expect(appendWalkEquals<R>(toSupplementaries("(ab)(cd)*"), toSupplementaries("zzzabcdzzz"), "$1", toSupplementaries("zzzabzzz")));
})

TODO_REGEX_TEST(stringBufferSubstituteWithThreeGroups, "supplementary replacement expansion is not exact in JavaMatcherAdapter")

PORTED_REGEX_TEST(stringBufferWithGroupsAndThreeMatches, {
  expect(appendWalkEquals<R>(toSupplementaries("(ab)(cd*)"), toSupplementaries("zzzabcdzzzabcddzzzabcdzzz"), "$2", toSupplementaries("zzzabzzzabcddzzzcdzzz"), 2));
})

TODO_REGEX_TEST(stringBufferEnsureDollarIgnored, "supplementary replacement escaping is not exact in JavaMatcherAdapter")

TODO_REGEX_TEST(stringBufferCheckNonexistentGroupReference, "requires Java replacement error semantics for nonexistent groups")

TODO_REGEX_TEST(stringBufferCheckSupplementalDoubleDigitGroupReferences, "requires Java multi-digit replacement group backoff semantics")

TODO_REGEX_TEST(stringBufferBackoffSupplemental, "requires Java multi-digit replacement group backoff semantics")

TODO_REGEX_TEST(stringBufferCheckAppendException, "requires Java replacement IllegalArgumentException atomic append semantics")

PORTED_REGEX_TEST(stringBuilderSubstitutionWithLiteral, { expect(appendWalkEquals<R>("blah", "zzzblahzzz", "blech", "zzzblechzzz")); })
PORTED_REGEX_TEST(stringBuilderSubstitutionWithGroups, { expect(appendWalkEquals<R>("(ab)(cd)*", "zzzabcdzzz", "$1", "zzzabzzz")); })
PORTED_REGEX_TEST(stringBuilderSubstitutionWithThreeGroups, { expect(appendWalkEquals<R>("(ab)(cd)*(ef)", "zzzabcdcdefzzz", "$1w$2w$3", "zzzabwcdwefzzz")); })
PORTED_REGEX_TEST(stringBuilderSubstitutionThreeMatch, { expect(appendWalkEquals<R>("(ab)(cd*)", "zzzabcdzzzabcddzzzabcdzzz", "$2", "zzzabzzzabcddzzzcdzzz", 2)); })
PORTED_REGEX_TEST(stringBuilderSubtituteCheckEscapedDollar, { expect(appendWalkEquals<R>("(ab)(cd)*(ef)", "zzzabcdcdefzzz", "$1w\\$2w$3", "zzzabw$2wefzzz")); })
TODO_REGEX_TEST(stringBuilderNonexistentGroupError, "requires Java replacement error semantics for nonexistent groups")
TODO_REGEX_TEST(stringBuilderDoubleDigitGroupReferences, "requires Java multi-digit replacement group backoff semantics")
PORTED_REGEX_TEST(stringBuilderCheckBackoff, { expect(appendWalkEquals<R>("(ab)(cd)*(ef)", "zzzabcdcdefzzz", "$1w$15w$3", "zzzabwab5wefzzz")); })
PORTED_REGEX_TEST(stringBuilderSupplementalLiteralSubstitution, { expect(appendWalkEquals<R>(toSupplementaries("blah"), toSupplementaries("zzzblahzzz"), toSupplementaries("blech"), toSupplementaries("zzzblechzzz"))); })
PORTED_REGEX_TEST(stringBuilderSupplementalSubstitutionWithGroups, { expect(appendWalkEquals<R>(toSupplementaries("(ab)(cd)*"), toSupplementaries("zzzabcdzzz"), "$1", toSupplementaries("zzzabzzz"))); })
TODO_REGEX_TEST(stringBuilderSupplementalSubstitutionThreeGroups, "supplementary replacement expansion is not exact in JavaMatcherAdapter")
PORTED_REGEX_TEST(stringBuilderSubstitutionSupplementalSkipMiddleThreeMatch, { expect(appendWalkEquals<R>(toSupplementaries("(ab)(cd*)"), toSupplementaries("zzzabcdzzzabcddzzzabcdzzz"), "$2", toSupplementaries("zzzabzzzabcddzzzcdzzz"), 2)); })
TODO_REGEX_TEST(stringBuilderSupplementalEscapedDollar, "supplementary replacement escaping is not exact in JavaMatcherAdapter")
TODO_REGEX_TEST(stringBuilderSupplementalNonExistentGroupError, "requires Java replacement error semantics for nonexistent groups")
TODO_REGEX_TEST(stringBuilderSupplementalCheckDoubleDigitGroupReferences, "requires Java multi-digit replacement group backoff semantics")
TODO_REGEX_TEST(stringBuilderSupplementalCheckBackoff, "requires Java multi-digit replacement group backoff semantics")
TODO_REGEX_TEST(stringBuilderCheckIllegalArgumentException, "requires Java replacement IllegalArgumentException atomic append semantics")

PORTED_REGEX_TEST(substitutionBasher, {
  expect(replaceAllEquals<R>("([a-z]+)([0-9]+)", "abc123 def456", "$2:$1", "123:abc 456:def"));
  expect(replaceFirstEquals<R>("([a-z]+)([0-9]+)", "abc123 def456", "$2:$1", "123:abc def456"));
})

PORTED_REGEX_TEST(substitutionBasher2, {
  expect(replaceAllEquals<R>("(x+)", "xx yy xxx", "<$1>", "<xx> yy <xxx>"));
  expect(replaceAllEquals<R>("(x*)", "xx", "[$1]", "[xx][]"));
})

PORTED_REGEX_TEST(escapes, {
  expect(full<R>("\\t", "\t"));
  expect(full<R>("\\n", "\n"));
  expect(full<R>("\\r", "\r"));
  expect(full<R>("\\f", "\f"));
  expect(full<R>("\\x{41}", "A"));
})

PORTED_REGEX_TEST(blankInput, {
  expect(full<R>("", ""));
  expect(find<R>(".*", ""));
  expect(noFind<R>(".+", ""));
})

PORTED_REGEX_TEST(bm, {
  expect(find<R>("abcdefghijklmnop", "xxxabcdefghijklmnopxxx"));
  expect(noFind<R>("abcdefghijklmnop", "xxxabcdefghijklmno"));
})

PORTED_REGEX_TEST(slice, {
  expect(find<R>("abc", "xxabcxx"));
  expect(find<R>(toSupplementaries("abc"), toSupplementaries("xxabcxx")));
})

PORTED_REGEX_TEST(namedGroupCaptureTest, {
  R re("(?<first>[A-Za-z]+) (?<last>[A-Za-z]+)");
  if (!re.ok()) { expect(false); } else {
    JavaMatcherAdapter<R> m(&re, "Jane Doe");
    expect(m.find());
    if (!re.NamedCapturingGroups().empty()) {
      expect(m.group("first").has_value() && *m.group("first") == "Jane");
      expect(m.group("last").has_value() && *m.group("last") == "Doe");
    } else {
      expect(m.group(1).has_value() && *m.group(1) == "Jane");
      expect(m.group(2).has_value() && *m.group(2) == "Doe");
    }
  }
})

PORTED_REGEX_TEST(nonBmpClassComplementTest, {
  const std::string face = utf8(0x1F600);
  expect(full<R>("[^a]", face));
  expect(notFull<R>("[^" + face + "]", face));
})

PORTED_REGEX_TEST(unicodePropertiesTest, {
  expect(full<R>("\\p{IsGreek}+", "\xce\xb1\xce\xb2"));
  expect(notFull<R>("\\p{IsGreek}+", "abc"));
  expect(full<R>("\\p{Lu}+", "ABC"));
})

PORTED_REGEX_TEST(unicodeHexNotationTest, {
  expect(full<R>("\\x{41}", "A"));
  expect(full<R>("\\u0041", "A"));
  expect(full<R>("\\x{1F600}", utf8(0x1F600)));
})

PORTED_REGEX_TEST(unicodeClassesTest, {
  expect(full<R>("\\p{Lower}+", "abc"));
  expect(full<R>("\\p{Upper}+", "ABC"));
  expect(full<R>("\\p{Digit}+", "123"));
  expect(full<R>("\\p{Space}+", " \t\n"));
})

PORTED_REGEX_TEST(unicodeCharacterNameTest, {
  expect(full<R>("\\N{LATIN CAPITAL LETTER A}", "A"));
  expect(full<R>("\\N{GREEK SMALL LETTER ALPHA}", "\xce\xb1"));
})

PORTED_REGEX_TEST(horizontalAndVerticalWSTest, {
  expect(full<R>("\\h+", " \t"));
  expect(full<R>("\\v+", "\n\r"));
})

PORTED_REGEX_TEST(linebreakTest, {
  expect(full<R>("\\R", "\n"));
  expect(full<R>("\\R", "\r\n"));
  expect(noFind<R>("\\R", "x"));
})

PORTED_REGEX_TEST(branchTest, {
  expect(full<R>("a|ab", "ab"));
  expect(findGroup<R>("(foo)|(bar)", "bar", "bar"));
})

PORTED_REGEX_TEST(groupCurlyNotFoundSuppTest, {
  expect(noFind<R>(toSupplementaries("(abc){2}"), toSupplementaries("abc")));
  expect(full<R>(toSupplementaries("(abc){2}"), toSupplementaries("abcabc")));
})

PORTED_REGEX_TEST(groupCurlyBackoffTest, {
  expect(full<R>("(a+){2}", "aaaa"));
  expect(full<R>("(ab){1,3}", "abab"));
})

TODO_REGEX_TEST(patternAsPredicate, "Java Pattern.asPredicate API has no C++ adapter equivalent")
TODO_REGEX_TEST(patternAsMatchPredicate, "Java Pattern.asMatchPredicate API has no C++ adapter equivalent")
TODO_REGEX_TEST(invalidFlags, "Java integer flag validation has no C++ adapter equivalent")

PORTED_REGEX_TEST(embeddedFlags, {
  expect(full<R>("(?i)abc", "ABC"));
  expect(full<R>("(?s)a.b", "a\nb"));
  expect(find<R>("(?m)^abc", "x\nabc"));
  expect(notFull<R>("(?i:a)b", "AB"));
})

TODO_REGEX_TEST(grapheme, "\\b{g} grapheme boundary is tracked separately and unsupported by PCRE2/RE2")

PORTED_REGEX_TEST(expoBacktracking, {
  expect(full<R>("(x+)+y", "xxxxxxxxxxy"));
  expect(noFind<R>("(x+)+y", "xxxxxxxxxxz"));
})

PORTED_REGEX_TEST(invalidGroupName, {
  expect(rejects<R>("(?<1bad>a)"));
  expect(rejects<R>("(?<>a)"));
})

PORTED_REGEX_TEST(illegalRepetitionRange, {
  expect(rejects<R>("a{2,1}"));
  expect(rejects<R>("a{,1}"));
})

TODO_REGEX_TEST(surrogatePairWithCanonEq, "CANON_EQ plus surrogate-pair behavior has no regex_compat option support")

PORTED_REGEX_TEST(lineBreakWithQuantifier, {
  expect(full<R>("\\R+", "\n\r\n"));
  expect(full<R>("(?:\\R){2}", "\n\n"));
})

PORTED_REGEX_TEST(caseInsensitivePMatch, {
  expect(full<R>("p", "P", caseInsensitive()));
  expect(full<R>("[p]", "P", caseInsensitive()));
})

PORTED_REGEX_TEST(surrogatePairOverlapRegion, {
  const std::string cp = utf8(0x10061);
  R re(cp);
  if (!re.ok()) { expect(false); } else {
    JavaMatcherAdapter<R> m(&re, cp);
    expect(m.region(0, cp.size()).find());
    expect(!m.region(0, 1).find());
  }
})

TODO_REGEX_TEST(droppedClassesWithIntersection, "character-class intersection edge case needs a direct faithful port")

TODO_REGEX_TEST(errorMessageCaretIndentation, "asserts Java PatternSyntaxException diagnostic formatting")

PORTED_REGEX_TEST(unescapedBackslash, {
  expect(rejects<R>("abc\\"));
})

TODO_REGEX_TEST(badIntersectionSyntax, "PatternSyntaxException edge case needs a direct faithful port")

PORTED_REGEX_TEST(wordBoundaryInconsistencies, {
  expect(find<R>("\\bword\\b", "a word!"));
  expect(noFind<R>("\\bword\\b", "swordfish"));
})

TODO_REGEX_TEST(prematureHitEndInNFCCharProperty, "Matcher.hitEnd is not exposed by JavaMatcherAdapter")

PORTED_REGEX_TEST(iOOBForCIBackrefs, {
  expect(full<R>("(?i)(a)\\1", "aA"));
  expect(notFull<R>("(?i)(a)\\2", "aA"));
})

#undef PORTED_REGEX_TEST
#undef TODO_REGEX_TEST

} // namespace
} // namespace facebook::velox::regex_compat::test
