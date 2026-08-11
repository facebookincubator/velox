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
// Cases ported from pcre4j's `MatcherUnicodeTests.java`.  All offsets are
// translated from Java UTF-16 char offsets (used by pcre4j) to UTF-8 byte
// offsets (used by our backends).
//
//   Å      U+00C5  2 UTF-8 bytes  (C3 85)
//   Ǎ      U+01CD  2 UTF-8 bytes  (C7 8D)
//   •      U+2022  3 UTF-8 bytes  (E2 80 A2)
//   🌍     U+1F30D 4 UTF-8 bytes  (F0 9F 8C 8D)
//   !              1 UTF-8 byte
//

#include "velox/external/regex_compat/tests/BackendTestBase.h"
#include "velox/external/regex_compat/tests/JavaMatcherAdapter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::regex_compat::test {
namespace {

template <typename R>
using UnicodePortedTest = BackendTest<R>;
TYPED_TEST_SUITE(UnicodePortedTest, AllBackends);

TYPED_TEST(UnicodePortedTest, unicodeOneByte) {
  TypeParam re("\xC3\x85"); // Å
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "\xC3\x85");
  EXPECT_TRUE(m.matches());
  EXPECT_EQ(0, m.start());
  EXPECT_EQ(2, m.end());
}

TYPED_TEST(UnicodePortedTest, unicodeTwoBytes) {
  TypeParam re("\xC7\x8D"); // Ǎ
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "\xC7\x8D");
  EXPECT_TRUE(m.matches());
  EXPECT_EQ(0, m.start());
  EXPECT_EQ(2, m.end());
}

TYPED_TEST(UnicodePortedTest, unicodeThreeBytes) {
  TypeParam re("\xE2\x80\xA2"); // •
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "\xE2\x80\xA2");
  EXPECT_TRUE(m.matches());
  EXPECT_EQ(0, m.start());
  EXPECT_EQ(3, m.end());
}

TYPED_TEST(UnicodePortedTest, unicodeFourBytes) {
  TypeParam re("\xF0\x9F\x8C\x8D"); // 🌍 U+1F30D
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, "\xF0\x9F\x8C\x8D");
  EXPECT_TRUE(m.matches());
  EXPECT_EQ(0, m.start());
  EXPECT_EQ(4, m.end());
}

TYPED_TEST(UnicodePortedTest, unicode) {
  // ÅǍ•🌍!
  const char* both = "\xC3\x85\xC7\x8D\xE2\x80\xA2\xF0\x9F\x8C\x8D!";
  TypeParam re(both);
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, both);
  EXPECT_TRUE(m.matches());
  EXPECT_EQ(0, m.start());
  EXPECT_EQ(12, m.end());
  EXPECT_EQ(both, m.group(0).value());
}

// region() in Java uses UTF-16 char offsets; the original test calls
// region(3, 5) to bracket the surrogate pair for 🌍.  In our UTF-8 world
// that's byte range [7, 11).  We rely on JavaRegex's adapter doing the
// UTF-16/UTF-8 conversion internally and pass byte offsets to RE2/PCRE2.
TYPED_TEST(UnicodePortedTest, unicodeRegion) {
  const char* input = "\xC3\x85\xC7\x8D\xE2\x80\xA2\xF0\x9F\x8C\x8D!";
  TypeParam re("\xF0\x9F\x8C\x8D"); // 🌍
  ASSERT_TRUE(re.ok()) << re.error();
  JavaMatcherAdapter<TypeParam> m(&re, input);
  m.region(7, 11);
  EXPECT_TRUE(m.matches());
  EXPECT_EQ(7, m.start());
  EXPECT_EQ(11, m.end());
}

} // namespace
} // namespace facebook::velox::regex_compat::test
