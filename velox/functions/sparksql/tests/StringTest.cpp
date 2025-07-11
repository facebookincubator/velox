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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/type/Type.h"

#include <stdint.h>

namespace facebook::velox::functions::sparksql::test {
namespace {

class StringTest : public SparkFunctionBaseTest {
 protected:
  // This is a five codepoint sequence that renders as a single emoji.
  static constexpr char kWomanFacepalmingLightSkinTone[] =
      "\xF0\x9F\xA4\xA6\xF0\x9F\x8F\xBB\xE2\x80\x8D\xE2\x99\x80\xEF\xB8\x8F";
};

TEST_F(StringTest, ascii) {
  const auto ascii = [&](const std::optional<std::string>& arg) {
    return evaluateOnce<int32_t>("ascii(c0)", arg);
  };
  EXPECT_EQ(ascii(std::string("\0", 1)), 0);
  EXPECT_EQ(ascii(" "), 32);
  EXPECT_EQ(ascii("😋"), 128523);
  EXPECT_EQ(ascii(""), 0);
  EXPECT_EQ(ascii("¥"), 165);
  EXPECT_EQ(ascii("®"), 174);
  EXPECT_EQ(ascii("©"), 169);
  EXPECT_EQ(ascii("VELOX"), 86);
  EXPECT_EQ(ascii("VIP"), 86);
  EXPECT_EQ(ascii("Viod"), 86);
  EXPECT_EQ(ascii("V®"), 86);
  EXPECT_EQ(ascii("ÇÉµABC"), 199);
  EXPECT_EQ(ascii("Ȼ %($)"), 571);
  EXPECT_EQ(ascii("@£Ɇ123"), 64);
  EXPECT_EQ(ascii(std::nullopt), std::nullopt);
}

TEST_F(StringTest, bitLength) {
  const auto bitLength = [&](const std::optional<std::string>& arg) {
    return evaluateOnce<int32_t>("bit_length(c0)", arg);
  };

  EXPECT_EQ(bitLength(""), 0);
  EXPECT_EQ(bitLength(std::string("\0", 1)), 8);
  EXPECT_EQ(bitLength("1"), 8);
  EXPECT_EQ(bitLength("123"), 24);
  EXPECT_EQ(bitLength("😋"), 32);
  // Consists of five codepoints.
  EXPECT_EQ(bitLength(kWomanFacepalmingLightSkinTone), 136);
  EXPECT_EQ(bitLength("\U0001F408"), 32);
}

TEST_F(StringTest, bitLengthVarbinary) {
  const auto bitLength = [&](const std::optional<std::string>& arg) {
    return evaluateOnce<int32_t>("bit_length(c0)", VARBINARY(), arg);
  };

  EXPECT_EQ(bitLength(""), 0);
  EXPECT_EQ(bitLength(std::string("\0", 1)), 8);
  EXPECT_EQ(bitLength("1"), 8);
  EXPECT_EQ(bitLength("123"), 24);
  EXPECT_EQ(bitLength("😋"), 32);
  // Consists of five codepoints.
  EXPECT_EQ(bitLength(kWomanFacepalmingLightSkinTone), 136);
  EXPECT_EQ(bitLength("\U0001F408"), 32);
}

TEST_F(StringTest, chr) {
  const auto chr = [&](std::optional<int64_t> arg) {
    return evaluateOnce<std::string>("chr(c0)", arg);
  };
  EXPECT_EQ(chr(-16), "");
  EXPECT_EQ(chr(0), std::string("\0", 1));
  EXPECT_EQ(chr(0x100), std::string("\0", 1));
  EXPECT_EQ(chr(0x1100), std::string("\0", 1));
  EXPECT_EQ(chr(0x20), "\x20");
  EXPECT_EQ(chr(0x100 + 0x20), "\x20");
  EXPECT_EQ(chr(0x80), "\xC2\x80");
  EXPECT_EQ(chr(0x100 + 0x80), "\xC2\x80");
  EXPECT_EQ(chr(0xFF), "\xC3\xBF");
  EXPECT_EQ(chr(0x100 + 0xFF), "\xC3\xBF");
  EXPECT_EQ(chr(std::nullopt), std::nullopt);
}

TEST_F(StringTest, contains) {
  const auto contains = [&](const std::optional<std::string>& str,
                            const std::optional<std::string>& pattern) {
    return evaluateOnce<bool>("contains(c0, c1)", str, pattern);
  };
  EXPECT_EQ(contains("hello", "ello"), true);
  EXPECT_EQ(contains("hello", "hell"), true);
  EXPECT_EQ(contains("hello", "hello there!"), false);
  EXPECT_EQ(contains("hello there!", "hello"), true);
  EXPECT_EQ(contains("hello there!", ""), true);
  EXPECT_EQ(contains("-- hello there!", std::nullopt), std::nullopt);
  EXPECT_EQ(contains(std::nullopt, "abc"), std::nullopt);
}

TEST_F(StringTest, conv) {
  const auto conv = [&](const std::optional<std::string>& str,
                        const std::optional<int32_t>& fromBase,
                        const std::optional<int32_t>& toBase) {
    return evaluateOnce<std::string>("conv(c0, c1, c2)", str, fromBase, toBase);
  };
  EXPECT_EQ(conv("4", 10, 2), "100");
  EXPECT_EQ(conv("110", 2, 10), "6");
  EXPECT_EQ(conv("15", 10, 16), "F");
  EXPECT_EQ(conv("15", 10, -16), "F");
  EXPECT_EQ(conv("big", 36, 16), "3A48");
  EXPECT_EQ(conv("-15", 10, -16), "-F");
  EXPECT_EQ(conv("-10", 16, -10), "-16");

  // Overflow case.
  EXPECT_EQ(
      conv("-9223372036854775809", 10, -2),
      "-111111111111111111111111111111111111111111111111111111111111111");
  EXPECT_EQ(
      conv("-9223372036854775808", 10, -2),
      "-1000000000000000000000000000000000000000000000000000000000000000");
  EXPECT_EQ(
      conv("9223372036854775808", 10, -2),
      "-1000000000000000000000000000000000000000000000000000000000000000");
  EXPECT_EQ(
      conv("8000000000000000", 16, -2),
      "-1000000000000000000000000000000000000000000000000000000000000000");
  EXPECT_EQ(conv("-1", 10, 16), "FFFFFFFFFFFFFFFF");
  EXPECT_EQ(conv("FFFFFFFFFFFFFFFF", 16, -10), "-1");
  EXPECT_EQ(conv("-FFFFFFFFFFFFFFFF", 16, -10), "-1");
  EXPECT_EQ(conv("-FFFFFFFFFFFFFFFF", 16, 10), "18446744073709551615");
  EXPECT_EQ(conv("-15", 10, 16), "FFFFFFFFFFFFFFF1");
  EXPECT_EQ(conv("9223372036854775807", 36, 16), "FFFFFFFFFFFFFFFF");

  // Leading and trailing spaces.
  EXPECT_EQ(conv("15 ", 10, 16), "F");
  EXPECT_EQ(conv(" 15 ", 10, 16), "F");

  // Invalid characters.
  // Only converts "11".
  EXPECT_EQ(conv("11abc", 10, 16), "B");
  // Only converts "F".
  EXPECT_EQ(conv("FH", 16, 10), "15");
  // Discards followed invalid character even though converting to same base.
  EXPECT_EQ(conv("11abc", 10, 10), "11");
  EXPECT_EQ(conv("FH", 16, 16), "F");
  // Begins with invalid character.
  EXPECT_EQ(conv("HF", 16, 10), "0");
  // All are invalid for binary base.
  EXPECT_EQ(conv("2345", 2, 10), "0");

  // Negative symbol only.
  EXPECT_EQ(conv("-", 10, 16), "0");

  // Null result.
  EXPECT_EQ(conv("", 10, 16), std::nullopt);
  EXPECT_EQ(conv(" ", 10, 16), std::nullopt);
  EXPECT_EQ(conv("", std::nullopt, 16), std::nullopt);
  EXPECT_EQ(conv("", 10, std::nullopt), std::nullopt);
}

TEST_F(StringTest, levenshtein) {
  const auto levenshtein = [&](const std::optional<std::string>& left,
                               const std::optional<std::string>& right,
                               const std::optional<int32_t>& threshold =
                                   std::nullopt) {
    if (threshold.has_value()) {
      return evaluateOnce<int32_t>(
          "levenshtein(c0, c1, c2)", left, right, threshold);
    }
    return evaluateOnce<int32_t>("levenshtein(c0, c1)", left, right);
  };

  EXPECT_EQ(levenshtein("abc", "abc"), 0);
  EXPECT_EQ(levenshtein("kitten", "sitting"), 3);
  EXPECT_EQ(levenshtein("frog", "fog"), 1);
  EXPECT_EQ(levenshtein("", "hello"), 5);
  EXPECT_EQ(levenshtein("hello", ""), 5);
  EXPECT_EQ(levenshtein("hello", "hello"), 0);
  EXPECT_EQ(levenshtein("hello", "olleh"), 4);
  EXPECT_EQ(levenshtein("hello world", "hello"), 6);
  EXPECT_EQ(levenshtein("hello", "hello world"), 6);
  EXPECT_EQ(levenshtein("hello world", "hel wold"), 3);
  EXPECT_EQ(levenshtein("hello world", "hellq wodld"), 2);
  EXPECT_EQ(levenshtein("hello word", "dello world"), 2);
  EXPECT_EQ(levenshtein("  facebook  ", "  facebook  "), 0);
  EXPECT_EQ(levenshtein("hello", std::string(100000, 'h')), 99999);
  EXPECT_EQ(levenshtein(std::string(100000, 'l'), "hello"), 99998);
  EXPECT_EQ(levenshtein(std::string(1000001, 'h'), ""), 1000001);
  EXPECT_EQ(levenshtein("", std::string(1000001, 'h')), 1000001);
  EXPECT_EQ(levenshtein("千世", "fog"), 3);
  EXPECT_EQ(levenshtein("世界千世", "大a界b"), 4);

  EXPECT_EQ(levenshtein("kitten", "sitting", 2), -1);

  EXPECT_EQ(levenshtein("", "", 0), 0);

  EXPECT_EQ(levenshtein("aaapppp", "", 8), 7);
  EXPECT_EQ(levenshtein("aaapppp", "", 7), 7);
  EXPECT_EQ(levenshtein("aaapppp", "", 6), -1);

  EXPECT_EQ(levenshtein("elephant", "hippo", 7), 7);
  EXPECT_EQ(levenshtein("elephant", "hippo", 6), -1);
  EXPECT_EQ(levenshtein("hippo", "elephant", 7), 7);
  EXPECT_EQ(levenshtein("hippo", "elephant", 6), -1);

  EXPECT_EQ(levenshtein("b", "a", 0), -1);
  EXPECT_EQ(levenshtein("a", "b", 0), -1);

  EXPECT_EQ(levenshtein("aa", "aa", 0), 0);
  EXPECT_EQ(levenshtein("aa", "aa", 2), 0);

  EXPECT_EQ(levenshtein("aaa", "bbb", 2), -1);
  EXPECT_EQ(levenshtein("aaa", "bbb", 3), 3);

  EXPECT_EQ(levenshtein("aaaaaa", "b", 10), 6);

  EXPECT_EQ(levenshtein("aaapppp", "b", 8), 7);
  EXPECT_EQ(levenshtein("a", "bbb", 4), 3);

  EXPECT_EQ(levenshtein("aaapppp", "b", 7), 7);
  EXPECT_EQ(levenshtein("a", "bbb", 3), 3);

  EXPECT_EQ(levenshtein("a", "bbb", 2), -1);
  EXPECT_EQ(levenshtein("bbb", "a", 2), -1);
  EXPECT_EQ(levenshtein("aaapppp", "b", 6), -1);

  EXPECT_EQ(levenshtein("a", "bbb", 1), -1);
  EXPECT_EQ(levenshtein("bbb", "a", 1), -1);

  EXPECT_EQ(levenshtein("12345", "1234567", 1), -1);
  EXPECT_EQ(levenshtein("1234567", "12345", 1), -1);

  EXPECT_EQ(levenshtein("千世", "fog", 3), 3);
  EXPECT_EQ(levenshtein("千世", "fog", 2), -1);
  EXPECT_EQ(levenshtein("世界千世", "大a界b", 4), 4);
  EXPECT_EQ(levenshtein("世界千世", "大a界b", 3), -1);
}

TEST_F(StringTest, endsWith) {
  const auto endsWith = [&](const std::optional<std::string>& str,
                            const std::optional<std::string>& pattern) {
    return evaluateOnce<bool>("endsWith(c0, c1)", str, pattern);
  };
  EXPECT_EQ(endsWith("hello", "ello"), true);
  EXPECT_EQ(endsWith("hello", "hell"), false);
  EXPECT_EQ(endsWith("hello", "hello there!"), false);
  EXPECT_EQ(endsWith("hello there!", "hello"), false);
  EXPECT_EQ(endsWith("hello there!", "!"), true);
  EXPECT_EQ(endsWith("hello there!", "there!"), true);
  EXPECT_EQ(endsWith("hello there!", "hello there!"), true);
  EXPECT_EQ(endsWith("hello there!", ""), true);
  EXPECT_EQ(endsWith("hello there!", "hello there"), false);
  EXPECT_EQ(endsWith("-- hello there!", "hello there"), false);
  EXPECT_EQ(endsWith("-- hello there!", std::nullopt), std::nullopt);
  EXPECT_EQ(endsWith(std::nullopt, "abc"), std::nullopt);
}

TEST_F(StringTest, findInSet) {
  const auto findInSet = [&](const std::optional<std::string>& str,
                             const std::optional<std::string>& strArray) {
    return evaluateOnce<int32_t>("find_in_set(c0, c1)", str, strArray);
  };
  EXPECT_EQ(findInSet("ab", "abc,b,ab,c,def"), 3);
  EXPECT_EQ(findInSet("abc", "abc,b,ab,c,def"), 1);
  EXPECT_EQ(findInSet("ab,", "abc,b,ab,c,def"), 0);
  EXPECT_EQ(findInSet("ab", "abc,b,ab,ab,ab"), 3);
  EXPECT_EQ(findInSet("abc", "abc,abc,abc,abc,abc"), 1);
  EXPECT_EQ(findInSet("c", "abc,b,ab,c,def"), 4);
  EXPECT_EQ(findInSet("dfg", "abc,b,ab,c,def"), 0);
  EXPECT_EQ(findInSet("dfg", "dfgdsiaq"), 0);
  EXPECT_EQ(findInSet("dfg", "dfgdsiaq, dshadad"), 0);
  EXPECT_EQ(findInSet("", ""), 1);
  EXPECT_EQ(findInSet("", "123"), 0);
  EXPECT_EQ(findInSet("123", ""), 0);
  EXPECT_EQ(findInSet("", "123,"), 2);
  EXPECT_EQ(findInSet("", ",123"), 1);
  EXPECT_EQ(findInSet("dfg", std::nullopt), std::nullopt);
  EXPECT_EQ(findInSet(std::nullopt, "abc"), std::nullopt);
  EXPECT_EQ(findInSet(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(findInSet("\u0061\u0062", "abc,b,ab,c,def"), 3);
  EXPECT_EQ(findInSet("\u0063", "abc,b,ab,c,def"), 4);
  EXPECT_EQ(findInSet("", "\u002c\u0031\u0032\u0033"), 1);
  EXPECT_EQ(findInSet("123", "\u002c\u0031\u0032\u0033"), 2);
  EXPECT_EQ(findInSet("😊", "🌍,😊"), 2);
  EXPECT_EQ(findInSet("😊", "😊,123"), 1);
  EXPECT_EQ(findInSet("abåæçè", ",abåæçè"), 2);
  EXPECT_EQ(findInSet("abåæçè", "abåæçè,"), 1);
  EXPECT_EQ(findInSet("\u0061\u0062\u00e5\u00e6\u00e7\u00e8", ",abåæçè"), 2);
  EXPECT_EQ(
      findInSet("abåæçè", "\u002c\u0061\u0062\u00e5\u00e6\u00e7\u00e8"), 2);
}

TEST_F(StringTest, instr) {
  const auto instr = [&](const std::optional<std::string>& haystack,
                         const std::optional<std::string>& needle) {
    return evaluateOnce<int32_t>("instr(c0, c1)", haystack, needle);
  };
  EXPECT_EQ(instr("SparkSQL", "SQL"), 6);
  EXPECT_EQ(instr(std::nullopt, "SQL"), std::nullopt);
  EXPECT_EQ(instr("SparkSQL", std::nullopt), std::nullopt);
  EXPECT_EQ(instr("SparkSQL", "Spark"), 1);
  EXPECT_EQ(instr("SQL", "SparkSQL"), 0);
  EXPECT_EQ(instr("", ""), 1);
  EXPECT_EQ(instr("abdef", "g"), 0);
  EXPECT_EQ(instr("", "a"), 0);
  EXPECT_EQ(instr("abdef", ""), 1);
  EXPECT_EQ(instr("abc😋def", "😋"), 4);
  // Offsets are calculated in terms of codepoints, not characters.
  // kWomanFacepalmingLightSkinTone is five codepoints.
  EXPECT_EQ(
      instr(std::string(kWomanFacepalmingLightSkinTone) + "abc😋def", "😋"), 9);
  EXPECT_EQ(
      instr(std::string(kWomanFacepalmingLightSkinTone) + "abc😋def", "def"),
      10);
}

TEST_F(StringTest, left) {
  const auto left = [&](const std::optional<std::string>& str,
                        const std::optional<int32_t>& length) {
    return evaluateOnce<std::string>("left(c0, c1)", str, length);
  };
  EXPECT_EQ(left("example", -2), "");
  EXPECT_EQ(left("example", 0), "");
  EXPECT_EQ(left("example", 2), "ex");
  EXPECT_EQ(left("example", 7), "example");
  EXPECT_EQ(left("example", 20), "example");

  EXPECT_EQ(left("da\u6570\u636Eta", 2), "da");
  EXPECT_EQ(left("da\u6570\u636Eta", 3), "da\u6570");
  EXPECT_EQ(left("da\u6570\u636Eta", 30), "da\u6570\u636Eta");
}

TEST_F(StringTest, lengthString) {
  const auto length = [&](const std::optional<std::string>& arg) {
    return evaluateOnce<int32_t>("length(c0)", arg);
  };
  EXPECT_EQ(length(""), 0);
  EXPECT_EQ(length(std::string("\0", 1)), 1);
  EXPECT_EQ(length("1"), 1);
  EXPECT_EQ(length("😋"), 1);
  EXPECT_EQ(length("😋😋"), 2);
  // Consists of five codepoints.
  EXPECT_EQ(length(kWomanFacepalmingLightSkinTone), 5);
  EXPECT_EQ(length("1234567890abdef"), 15);
}

TEST_F(StringTest, lengthVarbinary) {
  const auto length = [&](const std::optional<std::string>& arg) {
    return evaluateOnce<int32_t>("length(c0)", VARBINARY(), arg);
  };
  EXPECT_EQ(length(""), 0);
  EXPECT_EQ(length(std::string("\0", 1)), 1);
  EXPECT_EQ(length("1"), 1);
  EXPECT_EQ(length("😋"), 4);
  EXPECT_EQ(length(kWomanFacepalmingLightSkinTone), 17);
  EXPECT_EQ(length("1234567890abdef"), 15);
}

TEST_F(StringTest, lpad) {
  const std::string invalidString = "Ψ\xFF\xFFΣΓΔA";
  const std::string invalidPadString = "\xFFΨ\xFF";

  const auto lpad = [&](const std::optional<std::string>& string,
                        const std::optional<int32_t>& size) {
    return evaluateOnce<std::string>("lpad(c0, c1)", string, size);
  };

  const auto lpadWithPadString =
      [&](const std::optional<std::string>& string,
          const std::optional<int32_t>& size,
          const std::optional<std::string>& padString) {
        return evaluateOnce<std::string>(
            "lpad(c0, c1, c2)", string, size, padString);
      };

  EXPECT_EQ("  text", lpad("text", 6));

  // ASCII strings with various values for size and padString
  EXPECT_EQ("xtext", lpadWithPadString("text", 5, "x"));
  EXPECT_EQ("text", lpadWithPadString("text", 4, "x"));
  EXPECT_EQ("xyxtext", lpadWithPadString("text", 7, "xy"));

  // Non-ASCII strings with various values for size and padString
  EXPECT_EQ(
      "\u671B\u671B\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ",
      lpadWithPadString("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 11, "\u671B"));
  EXPECT_EQ(
      "\u5E0C\u671B\u5E0C\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ",
      lpadWithPadString(
          "\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 12, "\u5E0C\u671B"));

  // Empty string
  EXPECT_EQ("aaa", lpadWithPadString("", 3, "a"));

  // Truncating string
  EXPECT_EQ("", lpadWithPadString("abc", 0, "e"));
  EXPECT_EQ("tex", lpadWithPadString("text", 3, "xy"));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 ",
      lpadWithPadString("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 5, "\u671B"));

  // Invalid UTF-8 chars
  EXPECT_EQ("x" + invalidString, lpadWithPadString(invalidString, 8, "x"));
  EXPECT_EQ(
      invalidPadString + "abc", lpadWithPadString("abc", 6, invalidPadString));
}

TEST_F(StringTest, ltrim) {
  const auto ltrim = [&](const std::optional<std::string>& srcStr) {
    return evaluateOnce<std::string>("ltrim(c0)", srcStr);
  };

  const auto ltrimWithTrimStr = [&](const std::optional<std::string>& trimStr,
                                    const std::optional<std::string>& srcStr) {
    return evaluateOnce<std::string>("ltrim(c0, c1)", trimStr, srcStr);
  };

  EXPECT_EQ(ltrim(""), "");
  EXPECT_EQ(ltrim("  data\t "), "data\t ");
  EXPECT_EQ(ltrim("  data\t"), "data\t");
  EXPECT_EQ(ltrim("data\t "), "data\t ");
  EXPECT_EQ(ltrim("data\t"), "data\t");
  EXPECT_EQ(ltrim("  \u6570\u636E\t "), "\u6570\u636E\t ");
  EXPECT_EQ(ltrim("  \u6570\u636E\t"), "\u6570\u636E\t");
  EXPECT_EQ(ltrim("\u6570\u636E\t "), "\u6570\u636E\t ");
  EXPECT_EQ(ltrim("\u6570\u636E\t"), "\u6570\u636E\t");

  EXPECT_EQ(ltrimWithTrimStr("", ""), "");
  EXPECT_EQ(ltrimWithTrimStr("", "srcStr"), "srcStr");
  EXPECT_EQ(ltrimWithTrimStr("trimStr", ""), "");
  EXPECT_EQ(ltrimWithTrimStr("data!egr< >int", "integer data!"), "");
  EXPECT_EQ(ltrimWithTrimStr("int", "integer data!"), "eger data!");
  EXPECT_EQ(ltrimWithTrimStr("!!at", "integer data!"), "integer data!");
  EXPECT_EQ(ltrimWithTrimStr("a", "integer data!"), "integer data!");
  EXPECT_EQ(
      ltrimWithTrimStr(
          "\u6570\u6574!\u6570 \u636E!", "\u6574\u6570 \u6570\u636E!"),
      "");
  EXPECT_EQ(
      ltrimWithTrimStr(" \u6574\u6570 ", "\u6574\u6570 \u6570\u636E!"),
      "\u636E!");
  EXPECT_EQ(
      ltrimWithTrimStr("! \u6570\u636E!", "\u6574\u6570 \u6570\u636E!"),
      "\u6574\u6570 \u6570\u636E!");
  EXPECT_EQ(
      ltrimWithTrimStr("\u6570", "\u6574\u6570 \u6570\u636E!"),
      "\u6574\u6570 \u6570\u636E!");
}

TEST_F(StringTest, md5) {
  const auto md5 = [&](const std::optional<std::string>& arg) {
    return evaluateOnce<std::string>("md5(c0)", VARBINARY(), arg);
  };
  EXPECT_EQ(md5(std::nullopt), std::nullopt);
  EXPECT_EQ(md5(""), "d41d8cd98f00b204e9800998ecf8427e");
  EXPECT_EQ(md5("Infinity"), "eb2ac5b04180d8d6011a016aeb8f75b3");
}

TEST_F(StringTest, overlayVarchar) {
  const auto overlay = [&](const std::optional<std::string>& input,
                           const std::optional<std::string>& replace,
                           const std::optional<int32_t>& pos,
                           const std::optional<int32_t>& len) {
    // overlay is a keyword of DuckDB, use double quote avoid parse error.
    return evaluateOnce<std::string>(
        "\"overlay\"(c0, c1, c2, c3)", input, replace, pos, len);
  };
  EXPECT_EQ(overlay("Spark\u6570\u636ESQL", "_", 6, -1), "Spark_\u636ESQL");
  EXPECT_EQ(
      overlay("Spark\u6570\u636ESQL", "_", 6, 0), "Spark_\u6570\u636ESQL");
  EXPECT_EQ(overlay("Spark\u6570\u636ESQL", "_", -6, 2), "_\u636ESQL");

  EXPECT_EQ(overlay("Spark SQL", "_", 6, -1), "Spark_SQL");
  EXPECT_EQ(overlay("Spark SQL", "CORE", 7, -1), "Spark CORE");
  EXPECT_EQ(overlay("Spark SQL", "ANSI ", 7, 0), "Spark ANSI SQL");
  EXPECT_EQ(overlay("Spark SQL", "tructured", 2, 4), "Structured SQL");

  EXPECT_EQ(overlay("Spark SQL", "##", 10, -1), "Spark SQL##");
  EXPECT_EQ(overlay("Spark SQL", "##", 10, 4), "Spark SQL##");
  EXPECT_EQ(overlay("Spark SQL", "##", 0, -1), "##park SQL");
  EXPECT_EQ(overlay("Spark SQL", "##", 0, 4), "##rk SQL");
  EXPECT_EQ(overlay("Spark SQL", "##", -10, -1), "##park SQL");
  EXPECT_EQ(overlay("Spark SQL", "##", -10, 4), "##rk SQL");
}

TEST_F(StringTest, overlayVarbinary) {
  const auto overlay = [&](const std::optional<std::string>& input,
                           const std::optional<std::string>& replace,
                           const std::optional<int32_t>& pos,
                           const std::optional<int32_t>& len) {
    // overlay is a keyword of DuckDB, use double quote avoid parse error.
    return evaluateOnce<std::string>(
        "\"overlay\"(cast(c0 as varbinary), cast(c1 as varbinary), c2, c3)",
        input,
        replace,
        pos,
        len);
  };
  EXPECT_EQ(overlay("Spark\x65\x20SQL", "_", 6, -1), "Spark_\x20SQL");
  EXPECT_EQ(overlay("Spark\x65\x20SQL", "_", 6, 0), "Spark_\x65\x20SQL");
  EXPECT_EQ(overlay("Spark\x65\x20SQL", "_", -6, 2), "_\x20SQL");

  EXPECT_EQ(overlay("Spark SQL", "_", 6, -1), "Spark_SQL");
  EXPECT_EQ(overlay("Spark SQL", "CORE", 7, -1), "Spark CORE");
  EXPECT_EQ(overlay("Spark SQL", "ANSI ", 7, 0), "Spark ANSI SQL");
  EXPECT_EQ(overlay("Spark SQL", "tructured", 2, 4), "Structured SQL");

  EXPECT_EQ(overlay("Spark SQL", "##", 10, -1), "Spark SQL##");
  EXPECT_EQ(overlay("Spark SQL", "##", 10, 4), "Spark SQL##");
  EXPECT_EQ(overlay("Spark SQL", "##", 0, -1), "##park SQL");
  EXPECT_EQ(overlay("Spark SQL", "##", 0, 4), "##rk SQL");
  EXPECT_EQ(overlay("Spark SQL", "##", -10, -1), "##park SQL");
  EXPECT_EQ(overlay("Spark SQL", "##", -10, 4), "##rk SQL");
}

TEST_F(StringTest, repeat) {
  const auto stringRepeat = [&](const std::optional<std::string>& str,
                                const std::optional<int32_t>& times) {
    return evaluateOnce<std::string>(
        fmt::format("{}(c0, c1)", "repeat"), str, times);
  };

  EXPECT_EQ(stringRepeat("hh", 2), "hhhh");
  EXPECT_EQ(stringRepeat("abab", 0), "");
  EXPECT_EQ(stringRepeat("abab", -1), "");
  EXPECT_EQ(stringRepeat("", 2), "");
  EXPECT_EQ(stringRepeat("123\u6570", 2), "123\u6570123\u6570");
  VELOX_ASSERT_USER_THROW(
      stringRepeat("hh", 524289),
      "Result size must be less than or equal to 1048576");
  VELOX_ASSERT_USER_THROW(
      stringRepeat(std::string(214749, 'l'), 10000),
      "integer overflow: 214749 * 10000");
}

TEST_F(StringTest, replace) {
  const auto replace = [&](const std::optional<std::string>& str,
                           const std::optional<std::string>& replaced) {
    return evaluateOnce<std::string>("replace(c0, c1)", str, replaced);
  };

  const auto replaceWithReplacement =
      [&](const std::optional<std::string>& str,
          const std::optional<std::string>& replaced,
          const std::optional<std::string>& replacement) {
        return evaluateOnce<std::string>(
            "replace(c0, c1, c2)", str, replaced, replacement);
      };
  EXPECT_EQ(replace("aaabaac", "a"), "bc");
  EXPECT_EQ(replace("aaabaac", ""), "aaabaac");
  EXPECT_EQ(replaceWithReplacement("aaabaac", "a", "z"), "zzzbzzc");
  EXPECT_EQ(replaceWithReplacement("aaabaac", "", "z"), "aaabaac");
  EXPECT_EQ(replaceWithReplacement("aaabaac", "a", ""), "bc");
  EXPECT_EQ(replaceWithReplacement("aaabaac", "x", "z"), "aaabaac");
  EXPECT_EQ(replaceWithReplacement("aaabaac", "aaa", "z"), "zbaac");
  EXPECT_EQ(replaceWithReplacement("aaabaac", "a", "xyz"), "xyzxyzxyzbxyzxyzc");
  EXPECT_EQ(replaceWithReplacement("aaabaac", "aaabaac", "z"), "z");
  EXPECT_EQ(
      replaceWithReplacement("123\u6570\u6570\u636E", "\u6570\u636E", "data"),
      "123\u6570data");
}

TEST_F(StringTest, rpad) {
  const std::string invalidString = "Ψ\xFF\xFFΣΓΔA";
  const std::string invalidPadString = "\xFFΨ\xFF";

  const auto rpad = [&](const std::optional<std::string>& string,
                        const std::optional<int32_t>& size) {
    return evaluateOnce<std::string>("rpad(c0, c1)", string, size);
  };

  const auto rpadWithPadString =
      [&](const std::optional<std::string>& string,
          const std::optional<int32_t>& size,
          const std::optional<std::string>& padString) {
        return evaluateOnce<std::string>(
            "rpad(c0, c1, c2)", string, size, padString);
      };

  EXPECT_EQ("text  ", rpad("text", 6));

  // ASCII strings with various values for size and padString
  EXPECT_EQ("textx", rpadWithPadString("text", 5, "x"));
  EXPECT_EQ("text", rpadWithPadString("text", 4, "x"));
  EXPECT_EQ("textxyx", rpadWithPadString("text", 7, "xy"));

  // Non-ASCII strings with various values for size and padString
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B  \u671B\u671B",
      rpadWithPadString("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 11, "\u671B"));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B  \u5E0C\u671B\u5E0C",
      rpadWithPadString(
          "\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 12, "\u5E0C\u671B"));

  // Empty string
  EXPECT_EQ("aaa", rpadWithPadString("", 3, "a"));

  // Truncating string
  EXPECT_EQ("", rpadWithPadString("abc", 0, "e"));
  EXPECT_EQ("tex", rpadWithPadString("text", 3, "xy"));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 ",
      rpadWithPadString("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 5, "\u671B"));

  // Invalid UTF-8 chars
  EXPECT_EQ(invalidString + "x", rpadWithPadString(invalidString, 8, "x"));
  EXPECT_EQ(
      "abc" + invalidPadString, rpadWithPadString("abc", 6, invalidPadString));
}

TEST_F(StringTest, rtrim) {
  const auto rtrim = [&](const std::optional<std::string>& srcStr) {
    return evaluateOnce<std::string>("rtrim(c0)", srcStr);
  };

  const auto rtrimWithTrimStr = [&](const std::optional<std::string>& trimStr,
                                    const std::optional<std::string>& srcStr) {
    return evaluateOnce<std::string>("rtrim(c0, c1)", trimStr, srcStr);
  };
  EXPECT_EQ(rtrim(""), "");
  EXPECT_EQ(rtrim("  data\t "), "  data\t");
  EXPECT_EQ(rtrim("  data\t"), "  data\t");
  EXPECT_EQ(rtrim("data\t "), "data\t");
  EXPECT_EQ(rtrim("data\t"), "data\t");
  EXPECT_EQ(rtrim("  \u6570\u636E\t "), "  \u6570\u636E\t");
  EXPECT_EQ(rtrim("  \u6570\u636E\t"), "  \u6570\u636E\t");
  EXPECT_EQ(rtrim("\u6570\u636E\t "), "\u6570\u636E\t");
  EXPECT_EQ(rtrim("\u6570\u636E\t"), "\u6570\u636E\t");

  EXPECT_EQ(rtrimWithTrimStr("", ""), "");
  EXPECT_EQ(rtrimWithTrimStr("", "srcStr"), "srcStr");
  EXPECT_EQ(rtrimWithTrimStr("trimStr", ""), "");
  EXPECT_EQ(rtrimWithTrimStr("data!egr< >int", "integer data!"), "");
  EXPECT_EQ(rtrimWithTrimStr("int", "integer data!"), "integer data!");
  EXPECT_EQ(rtrimWithTrimStr("!!at", "integer data!"), "integer d");
  EXPECT_EQ(rtrimWithTrimStr("a", "integer data!"), "integer data!");
  EXPECT_EQ(
      rtrimWithTrimStr(
          "\u6570\u6574!\u6570 \u636E!", "\u6574\u6570 \u6570\u636E!"),
      "");
  EXPECT_EQ(
      rtrimWithTrimStr(" \u6574\u6570 ", "\u6574\u6570 \u6570\u636E!"),
      "\u6574\u6570 \u6570\u636E!");
  EXPECT_EQ(
      rtrimWithTrimStr("! \u6570\u636E!", "\u6574\u6570 \u6570\u636E!"),
      "\u6574");
  EXPECT_EQ(
      rtrimWithTrimStr("\u6570", "\u6574\u6570 \u6570\u636E!"),
      "\u6574\u6570 \u6570\u636E!");
}

TEST_F(StringTest, sha1) {
  const auto sha1 = [&](const std::optional<std::string>& arg) {
    return evaluateOnce<std::string>("sha1(c0)", VARBINARY(), arg);
  };

  EXPECT_EQ(sha1(std::nullopt), std::nullopt);
  EXPECT_EQ(sha1(""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
  EXPECT_EQ(sha1("Spark"), "85f5955f4b27a9a4c2aab6ffe5d7189fc298b92c");
  EXPECT_EQ(
      sha1("0123456789abcdefghijklmnopqrstuvwxyz"),
      "a26704c04fc5f10db5aab58468035531cc542485");
}

TEST_F(StringTest, sha2) {
  const auto sha2 = [&](const std::optional<std::string>& str,
                        const std::optional<int32_t>& bitLength) {
    return evaluateOnce<std::string, std::string, int32_t>(
        "sha2(cast(c0 as varbinary), c1)", str, bitLength);
  };

  EXPECT_EQ(sha2("Spark", -1), std::nullopt);
  EXPECT_EQ(sha2("Spark", 1), std::nullopt);
  EXPECT_EQ(
      sha2("", 0),
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(
      sha2("Spark", 0),
      "529bc3b07127ecb7e53a4dcf1991d9152c24537d919178022b2c42657f79a26b");
  EXPECT_EQ(
      sha2("0123456789abcdefghijklmnopqrstuvwxyz", 0),
      "74e7e5bb9d22d6db26bf76946d40fff3ea9f0346b884fd0694920fccfad15e33");
  EXPECT_EQ(
      sha2("", 224),
      "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f");
  EXPECT_EQ(
      sha2("Spark", 224),
      "dbeab94971678d36af2195851c0f7485775a2a7c60073d62fc04549c");
  EXPECT_EQ(
      sha2("0123456789abcdefghijklmnopqrstuvwxyz", 224),
      "e6e4a6be069cc9bead8b6050856d2b26da6b3f7efa0951e5fb3a54dd");
  EXPECT_EQ(
      sha2("", 256),
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(
      sha2("Spark", 256),
      "529bc3b07127ecb7e53a4dcf1991d9152c24537d919178022b2c42657f79a26b");
  EXPECT_EQ(
      sha2("0123456789abcdefghijklmnopqrstuvwxyz", 256),
      "74e7e5bb9d22d6db26bf76946d40fff3ea9f0346b884fd0694920fccfad15e33");
  EXPECT_EQ(
      sha2("", 384),
      "38b060a751ac96384cd9327eb1b1e36a21fdb71114be0743"
      "4c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b");
  EXPECT_EQ(
      sha2("Spark", 384),
      "1e40b8d06c248a1cc32428c22582b6219d072283078fa140"
      "d9ad297ecadf2cabefc341b857ad36226aa8d6d79f2ab67d");
  EXPECT_EQ(
      sha2("0123456789abcdefghijklmnopqrstuvwxyz", 384),
      "ce6d4ea5442bc6c830bea1942d4860db9f7b96f0e9d2c307"
      "3ffe47a0e1166d95612d840ff15e5efdd23c1f273096da32");
  EXPECT_EQ(
      sha2("", 512),
      "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
      "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
  EXPECT_EQ(
      sha2("Spark", 512),
      "44844a586c54c9a212da1dbfe05c5f1705de1af5fda1f0d36297623249b279fd"
      "8f0ccec03f888f4fb13bf7cd83fdad58591c797f81121a23cfdd5e0897795238");
  EXPECT_EQ(
      sha2("0123456789abcdefghijklmnopqrstuvwxyz", 512),
      "95cadc34aa46b9fdef432f62fe5bad8d9f475bfbecf797d5802bb5f2937a85d9"
      "3ce4857a6262b03834c01c610d74cd1215f9a466dc6ad3dd15078e3309a03a6d");
}

TEST_F(StringTest, soundex) {
  const auto soundex = [&](const std::optional<std::string>& input) {
    return evaluateOnce<std::string>("soundex(c0)", input);
  };
  EXPECT_EQ(soundex("ZIN"), "Z500");
  EXPECT_EQ(soundex("SU"), "S000");
  EXPECT_EQ(soundex("zZ"), "Z000");
  EXPECT_EQ(soundex("RAGSSEEESSSVEEWE"), "R221");
  EXPECT_EQ(soundex("Miller"), "M460");
  EXPECT_EQ(soundex("Peterson"), "P362");
  EXPECT_EQ(soundex("Peters"), "P362");
  EXPECT_EQ(soundex("Auerbach"), "A612");
  EXPECT_EQ(soundex("Uhrbach"), "U612");
  EXPECT_EQ(soundex("Moskowitz"), "M232");
  EXPECT_EQ(soundex("Moskovitz"), "M213");
  EXPECT_EQ(soundex("relyheewsgeessg"), "R422");

  EXPECT_EQ(soundex("Robert"), "R163");
  EXPECT_EQ(soundex("Rupert"), "R163");
  EXPECT_EQ(soundex("Rubin"), "R150");

  EXPECT_EQ(soundex("Ashcraft"), "A261");
  EXPECT_EQ(soundex("Ashcroft"), "A261");
  EXPECT_EQ(soundex("Aswcraft"), "A261");

  EXPECT_EQ(soundex("Tymczak"), "T522");

  EXPECT_EQ(soundex("Pfister"), "P236");

  EXPECT_EQ(soundex("Honeyman"), "H555");

  EXPECT_EQ(soundex("Tschüss"), "T220");

  EXPECT_EQ(soundex(""), "");
  EXPECT_EQ(soundex("!!"), "!!");
  EXPECT_EQ(soundex("测试"), "测试");
}

TEST_F(StringTest, startsWith) {
  const auto startsWith = [&](const std::optional<std::string>& str,
                              const std::optional<std::string>& pattern) {
    return evaluateOnce<bool>("startsWith(c0, c1)", str, pattern);
  };

  EXPECT_EQ(startsWith("hello", "ello"), false);
  EXPECT_EQ(startsWith("hello", "hell"), true);
  EXPECT_EQ(startsWith("hello", "hello there!"), false);
  EXPECT_EQ(startsWith("hello there!", "hello"), true);
  EXPECT_EQ(startsWith("-- hello there!", "-"), true);
  EXPECT_EQ(startsWith("-- hello there!", ""), true);
  EXPECT_EQ(startsWith("-- hello there!", std::nullopt), std::nullopt);
  EXPECT_EQ(startsWith(std::nullopt, "abc"), std::nullopt);
}

TEST_F(StringTest, substring) {
  const auto substring = [&](const std::optional<std::string>& str,
                             const std::optional<int32_t>& start) {
    return evaluateOnce<std::string>("substring(c0, c1)", str, start);
  };

  const auto substringWithLength = [&](const std::optional<std::string>& str,
                                       const std::optional<int32_t>& start,
                                       const std::optional<int32_t>& length) {
    return evaluateOnce<std::string>(
        "substring(c0, c1, c2)", str, start, length);
  };

  EXPECT_EQ(substring("example", 0), "example");
  EXPECT_EQ(substring("example", 1), "example");
  EXPECT_EQ(substring("example", 2), "xample");
  EXPECT_EQ(substring("example", 8), "");
  EXPECT_EQ(substring("example", 2147483647), "");
  EXPECT_EQ(substring("example", -2), "le");
  EXPECT_EQ(substring("example", -7), "example");
  EXPECT_EQ(substring("example", -8), "example");
  EXPECT_EQ(substring("example", -9), "example");
  EXPECT_EQ(substring("example", -2147483647), "example");
  EXPECT_EQ(substring("da\u6570\u636Eta", 3), "\u6570\u636Eta");
  EXPECT_EQ(substring("da\u6570\u636Eta", -3), "\u636Eta");

  EXPECT_EQ(substringWithLength("example", 0, 2), "ex");
  EXPECT_EQ(substringWithLength("example", 1, -1), "");
  EXPECT_EQ(substringWithLength("example", 1, 0), "");
  EXPECT_EQ(substringWithLength("example", 1, 2), "ex");
  EXPECT_EQ(substringWithLength("example", 1, 7), "example");
  EXPECT_EQ(substringWithLength("example", 1, 100), "example");
  EXPECT_EQ(substringWithLength("example", 2, 2), "xa");
  EXPECT_EQ(substringWithLength("example", 8, 2), "");
  EXPECT_EQ(substringWithLength("example", -2, 2), "le");
  EXPECT_EQ(substringWithLength("example", -7, 2), "ex");
  EXPECT_EQ(substringWithLength("example", -8, 2), "e");
  EXPECT_EQ(substringWithLength("example", -9, 2), "");
  EXPECT_EQ(substringWithLength("example", -7, 7), "example");
  EXPECT_EQ(substringWithLength("example", -9, 9), "example");
  EXPECT_EQ(substringWithLength("example", 4, 2147483645), "mple");
  EXPECT_EQ(substringWithLength("example", 2147483645, 4), "");
  EXPECT_EQ(substringWithLength("example", -2147483648, 1), "");
  EXPECT_EQ(substringWithLength("da\u6570\u636Eta", 2, 4), "a\u6570\u636Et");
  EXPECT_EQ(substringWithLength("da\u6570\u636Eta", -3, 2), "\u636Et");
}

TEST_F(StringTest, locate) {
  const auto locate = [&](const std::optional<std::string>& substr,
                          const std::optional<std::string>& str,
                          const std::optional<int32_t>& start) {
    return evaluateOnce<int32_t>("locate(c0, c1, c2)", substr, str, start);
  };

  EXPECT_EQ(locate("aa", "aaads", 1), 1);
  EXPECT_EQ(locate("aa", "aaads", 0), 0);
  EXPECT_EQ(locate("aa", "aaads", 2), 2);
  EXPECT_EQ(locate("aa", "aaads", 3), 0);
  EXPECT_EQ(locate("aa", "aaads", -3), 0);
  EXPECT_EQ(locate("de", "aaads", 1), 0);
  EXPECT_EQ(locate("de", "aaads", 2), 0);
  EXPECT_EQ(locate("abc", "abcdddabcabc", 6), 7);
  EXPECT_EQ(locate("", "", 1), 1);
  EXPECT_EQ(locate("", "", 3), 1);
  EXPECT_EQ(locate("", "", -1), 0);
  EXPECT_EQ(locate("", "aaads", 1), 1);
  EXPECT_EQ(locate("", "aaads", 9), 1);
  EXPECT_EQ(locate("", "aaads", -1), 0);
  EXPECT_EQ(locate("aa", "", 1), 0);
  EXPECT_EQ(locate("aa", "", 2), 0);
  EXPECT_EQ(locate("zz", "aaads", std::nullopt), 0);
  EXPECT_EQ(locate("aa", std::nullopt, 1), std::nullopt);
  EXPECT_EQ(locate(std::nullopt, "aaads", 1), std::nullopt);
  EXPECT_EQ(locate(std::nullopt, std::nullopt, -1), std::nullopt);
  EXPECT_EQ(locate(std::nullopt, std::nullopt, std::nullopt), 0);

  EXPECT_EQ(locate("", "\u4FE1\u5FF5,\u7231,\u5E0C\u671B", 10), 1);
  EXPECT_EQ(locate("", "\u4FE1\u5FF5,\u7231,\u5E0C\u671B", -1), 0);
  EXPECT_EQ(locate("\u7231", "\u4FE1\u5FF5,\u7231,\u5E0C\u671B", 1), 4);
  EXPECT_EQ(locate("\u7231", "\u4FE1\u5FF5,\u7231,\u5E0C\u671B", 0), 0);
  EXPECT_EQ(
      locate("\u4FE1", "\u4FE1\u5FF5,\u4FE1\u7231,\u4FE1\u5E0C\u671B", 2), 4);
  EXPECT_EQ(
      locate("\u4FE1", "\u4FE1\u5FF5,\u4FE1\u7231,\u4FE1\u5E0C\u671B", 8), 0);
}

TEST_F(StringTest, substringIndex) {
  const auto substringIndex =
      [&](const std::string& str, const std::string& delim, int32_t count) {
        return evaluateOnce<std::string, std::string, std::string, int32_t>(
            "substring_index(c0, c1, c2)", str, delim, count);
      };
  EXPECT_EQ(substringIndex("www.apache.org", ".", 3), "www.apache.org");
  EXPECT_EQ(substringIndex("www.apache.org", ".", 2), "www.apache");
  EXPECT_EQ(substringIndex("www.apache.org", ".", 1), "www");
  EXPECT_EQ(substringIndex("www.apache.org", ".", 0), "");
  EXPECT_EQ(substringIndex("www.apache.org", ".", -1), "org");
  EXPECT_EQ(substringIndex("www.apache.org", ".", -2), "apache.org");
  EXPECT_EQ(substringIndex("www.apache.org", ".", -3), "www.apache.org");
  // Str is empty string.
  EXPECT_EQ(substringIndex("", ".", 1), "");
  // Empty string delim.
  EXPECT_EQ(substringIndex("www.apache.org", "", 1), "");
  // Delim does not exist in str.
  EXPECT_EQ(substringIndex("www.apache.org", "#", 2), "www.apache.org");
  EXPECT_EQ(substringIndex("www.apache.org", "WW", 1), "www.apache.org");
  // Delim is 2 chars.
  EXPECT_EQ(substringIndex("www||apache||org", "||", 2), "www||apache");
  EXPECT_EQ(substringIndex("www||apache||org", "||", -2), "apache||org");
  // Non ascii chars.
  EXPECT_EQ(substringIndex("大千世界大千世界", "千", 2), "大千世界大");

  // Overlapped delim.
  EXPECT_EQ(substringIndex("||||||", "|||", 3), "||");
  EXPECT_EQ(substringIndex("||||||", "|||", -4), "|||");
  EXPECT_EQ(substringIndex("aaaaa", "aa", 2), "a");
  EXPECT_EQ(substringIndex("aaaaa", "aa", -4), "aaa");
  EXPECT_EQ(substringIndex("aaaaa", "aa", 0), "");
  EXPECT_EQ(substringIndex("aaaaa", "aa", 5), "aaaaa");
  EXPECT_EQ(substringIndex("aaaaa", "aa", -5), "aaaaa");
}

TEST_F(StringTest, translate) {
  const auto testTranslate =
      [&](const std::vector<std::optional<std::string>>& inputs,
          auto& expected) {
        EXPECT_EQ(
            evaluateOnce<std::string>(
                "translate(c0, c1, c2)", inputs[0], inputs[1], inputs[2]),
            expected);
      };

  testTranslate({"ab[cd]", "[]", "##"}, "ab#cd#");
  testTranslate({"ab[cd]", "[]", "#"}, "ab#cd");
  testTranslate({"ab[cd]", "[]", "#@$"}, "ab#cd@");
  testTranslate({"ab[cd]", "[]", "  "}, "ab cd ");
  testTranslate({"ab\u2028", "\u2028", "\u2029"}, "ab\u2029");
  testTranslate({"abcabc", "a", "\u2029"}, "\u2029bc\u2029bc");
  testTranslate({"abc", "", ""}, "abc");
  testTranslate({"translate", "rnlt", "123"}, "1a2s3ae");
  testTranslate({"translate", "rnlt", ""}, "asae");
  testTranslate({"abcd", "aba", "123"}, "12cd");
  // Test null input.
  testTranslate({"abc", std::nullopt, "\u2029"}, std::nullopt);
  testTranslate({"abc", "\u2028", std::nullopt}, std::nullopt);
  testTranslate({std::nullopt, "\u2028", "\u2029"}, std::nullopt);
  // Test corner case: the input may contain invalid UTF-8 characters, which
  // could be generated by functions like url_decode. We need to ensure that
  // these characters do not cause incorrect results or failures.
  std::string invalidUtf8Char(1, (char)203);
  testTranslate({invalidUtf8Char, "1", "a"}, invalidUtf8Char);
  invalidUtf8Char[0] = (char)128;
  testTranslate({invalidUtf8Char, "1", "a"}, invalidUtf8Char);
  testTranslate({invalidUtf8Char, invalidUtf8Char, "a"}, "a");
  std::string data = "1ae4b";
  data[2] = (char)128;
  testTranslate({"1ae4b", "e", invalidUtf8Char}, data);
  testTranslate({data, "3", "a"}, data);
}

TEST_F(StringTest, translateConstantMatch) {
  auto rowType = ROW({{"c0", VARCHAR()}});
  auto exprSet = compileExpression("translate(c0, 'ab', '12')", rowType);

  const auto testTranslate = [&](const auto& input, const auto& expected) {
    auto result = evaluate(*exprSet, makeRowVector({input}));
    velox::test::assertEqualVectors(expected, result);
  };

  // Uses ascii batch as the initial input.
  auto input = makeFlatVector<std::string>({"abcd", "cdab"});
  auto expected = makeFlatVector<std::string>({"12cd", "cd12"});
  testTranslate(input, expected);

  // Uses unicode batch as the next input.
  input = makeFlatVector<std::string>({"abåæçè", "åæçèab"});
  expected = makeFlatVector<std::string>({"12åæçè", "åæçè12"});
  testTranslate(input, expected);
}

TEST_F(StringTest, translateNonconstantMatch) {
  auto rowType = ROW({{"c0", VARCHAR()}, {"c1", VARCHAR()}, {"c2", VARCHAR()}});
  auto exprSet = compileExpression("translate(c0, c1, c2)", rowType);

  const auto testTranslate = [&](const std::vector<VectorPtr>& inputs,
                                 const auto& expected) {
    auto result = evaluate(*exprSet, makeRowVector(inputs));
    velox::test::assertEqualVectors(expected, result);
  };

  // All inputs are ascii encoded.
  auto input = makeFlatVector<std::string>({"abcd", "cdab"});
  auto match = makeFlatVector<std::string>({"ab", "ca"});
  auto replace = makeFlatVector<std::string>({"#", "@$"});
  auto expected = makeFlatVector<std::string>({"#cd", "@d$b"});
  testTranslate({input, match, replace}, expected);

  // Partial inputs are ascii encoded.
  input = makeFlatVector<std::string>({"abcd", "cdab"});
  match = makeFlatVector<std::string>({"ac", "ab"});
  replace = makeFlatVector<std::string>({"åç", "æ"});
  expected = makeFlatVector<std::string>({"åbçd", "cdæ"});
  testTranslate({input, match, replace}, expected);

  // All inputs are unicode encoded.
  input = makeFlatVector<std::string>({"abåæçè", "åæçèac"});
  match = makeFlatVector<std::string>({"aå", "çc"});
  replace = makeFlatVector<std::string>({"åa", "cç"});
  expected = makeFlatVector<std::string>({"åbaæçè", "åæcèaç"});
  testTranslate({input, match, replace}, expected);
}

TEST_F(StringTest, trim) {
  const auto trim = [&](const std::optional<std::string>& srcStr) {
    return evaluateOnce<std::string>("trim(c0)", srcStr);
  };

  const auto trimWithTrimStr = [&](const std::optional<std::string>& trimStr,
                                   const std::optional<std::string>& srcStr) {
    return evaluateOnce<std::string>("trim(c0, c1)", trimStr, srcStr);
  };

  EXPECT_EQ(trim(""), "");
  EXPECT_EQ(trim("  data\t "), "data\t");
  EXPECT_EQ(trim("  data\t"), "data\t");
  EXPECT_EQ(trim("data\t "), "data\t");
  EXPECT_EQ(trim("data\t"), "data\t");
  EXPECT_EQ(trim("  \u6570\u636E\t "), "\u6570\u636E\t");
  EXPECT_EQ(trim("  \u6570\u636E\t"), "\u6570\u636E\t");
  EXPECT_EQ(trim("\u6570\u636E\t "), "\u6570\u636E\t");
  EXPECT_EQ(trim("\u6570\u636E\t"), "\u6570\u636E\t");

  EXPECT_EQ(trimWithTrimStr("", ""), "");
  EXPECT_EQ(trimWithTrimStr("", "srcStr"), "srcStr");
  EXPECT_EQ(trimWithTrimStr("trimWithTrimStrStr", ""), "");
  EXPECT_EQ(trimWithTrimStr("data!egr< >int", "integer data!"), "");
  EXPECT_EQ(trimWithTrimStr("int", "integer data!"), "eger data!");
  EXPECT_EQ(trimWithTrimStr("!!at", "integer data!"), "integer d");
  EXPECT_EQ(trimWithTrimStr("a", "integer data!"), "integer data!");
  EXPECT_EQ(
      trimWithTrimStr(
          "\u6570\u6574!\u6570 \u636E!", "\u6574\u6570 \u6570\u636E!"),
      "");
  EXPECT_EQ(
      trimWithTrimStr(" \u6574\u6570 ", "\u6574\u6570 \u6570\u636E!"),
      "\u636E!");
  EXPECT_EQ(
      trimWithTrimStr("! \u6570\u636E!", "\u6574\u6570 \u6570\u636E!"),
      "\u6574");
  EXPECT_EQ(
      trimWithTrimStr("\u6570", "\u6574\u6570 \u6570\u636E!"),
      "\u6574\u6570 \u6570\u636E!");
}

TEST_F(StringTest, empty2Null) {
  const auto empty2Null = [&](const std::optional<std::string>& a) {
    return evaluateOnce<std::string>("empty2null(c0)", a);
  };

  EXPECT_EQ(empty2Null(""), std::nullopt);
  EXPECT_EQ(empty2Null("abc"), "abc");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
