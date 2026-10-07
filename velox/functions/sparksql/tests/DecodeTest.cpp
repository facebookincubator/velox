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
#include <folly/String.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/Expressions.h"
#include "velox/functions/FunctionRegistry.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

using folly::unhexlify;
using velox::test::assertEqualVectors;

class DecodeTest : public SparkFunctionBaseTest {
 protected:
  std::optional<std::string> decode(
      const std::optional<std::string>& input,
      const std::optional<std::string>& charset) {
    return evaluateOnce<std::string>(
        "decode(c0, c1)", {VARBINARY(), VARCHAR()}, input, charset);
  }

  void testHex(
      const std::string& charset,
      const std::vector<std::pair<std::string, std::string>>& cases) {
    for (const auto& [input, expected] : cases) {
      SCOPED_TRACE(fmt::format("{}: {}", charset, input));
      EXPECT_EQ(decode(unhexlify(input), charset), unhexlify(expected));
    }
  }
};

TEST_F(DecodeTest, registration) {
  registerFunctions("decode_test_");
  for (const auto& name : {"decode", "decode_test_decode"}) {
    const auto signatures = getFunctionSignatures(name);
    ASSERT_EQ(signatures.size(), 1);
    EXPECT_EQ(signatures[0]->toString(), "(varbinary,varchar) -> varchar");
    EXPECT_EQ(
        signatures[0]->constantArguments(), (std::vector<bool>{false, false}));
    EXPECT_EQ(resolveFunction(name, {VARBINARY(), VARCHAR()}), VARCHAR());
    EXPECT_EQ(resolveFunction(name, {VARCHAR(), VARCHAR()}), nullptr);
    EXPECT_EQ(resolveFunction(name, {VARBINARY()}), nullptr);
    EXPECT_EQ(
        resolveFunction(name, {VARBINARY(), VARCHAR(), VARCHAR()}), nullptr);
  }
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"hello"}, VARBINARY()),
       makeFlatVector<std::string>({"UTF-8"})});
  assertEqualVectors(
      makeFlatVector<std::string>({"hello"}),
      evaluate("decode_test_decode(c0, c1)", input));
}

TEST_F(DecodeTest, canonicalNames) {
  const std::vector<std::pair<std::string, std::string>> modes{
      {"US-ASCII", "41"},
      {"ISO-8859-1", "41"},
      {"UTF-8", "41"},
      {"UTF-16BE", "0041"},
      {"UTF-16LE", "4100"},
      {"UTF-16", "0041"},
      {"UTF-32", "00000041"}};
  for (const auto& [name, hex] : modes) {
    auto lower = name;
    auto mixed = name;
    for (size_t i = 0; i < name.size(); ++i) {
      if (name[i] >= 'A' && name[i] <= 'Z') {
        lower[i] += 'a' - 'A';
        if (i % 2 == 0) {
          mixed[i] += 'a' - 'A';
        }
      }
    }
    for (const auto& charset : {name, lower, mixed}) {
      EXPECT_EQ(decode(unhexlify(hex), charset), "A");
      EXPECT_EQ(decode("", charset), "");
      EXPECT_EQ(
          evaluateOnce<std::string>(
              fmt::format("decode(c0, '{}')", charset),
              {VARBINARY()},
              std::optional<std::string>(unhexlify(hex))),
          "A");
    }
  }
}

TEST_F(DecodeTest, singleByteAllValues) {
  std::string input;
  std::string ascii;
  std::string latin1;
  for (int i = 0; i < 256; ++i) {
    input += static_cast<char>(i);
    if (i < 128) {
      ascii += static_cast<char>(i);
      latin1 += static_cast<char>(i);
    } else {
      ascii += "\xef\xbf\xbd";
      // U+0080..U+00FF: independently derive the two UTF-8 octets.
      latin1 += static_cast<char>(0xc0 | (i >> 6));
      latin1 += static_cast<char>(0x80 | (i & 0x3f));
    }
  }
  EXPECT_EQ(decode(input, "US-ASCII"), ascii);
  EXPECT_EQ(decode(input, "ISO-8859-1"), latin1);
  testHex("US-ASCII", {{"007F80FF", "007FEFBFBDEFBFBD"}});
  testHex("ISO-8859-1", {{"007F80FF", "007FC280C3BF"}});
}

TEST_F(DecodeTest, nulls) {
  EXPECT_EQ(decode(std::nullopt, "UTF-8"), std::nullopt);
  EXPECT_EQ(decode("a", std::nullopt), std::nullopt);
  EXPECT_EQ(decode(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(decode(std::nullopt, "invalid"), std::nullopt);
  auto binary = makeNullableFlatVector<std::string>(
      {std::nullopt, std::nullopt, std::nullopt}, VARBINARY());
  auto expected = makeNullableFlatVector<std::string>(
      {std::nullopt, std::nullopt, std::nullopt});
  assertEqualVectors(
      expected, evaluate("decode(c0, 'invalid')", makeRowVector({binary})));
  assertEqualVectors(
      expected,
      evaluate("try(decode(c0, 'invalid'))", makeRowVector({binary})));
}

TEST_F(DecodeTest, invalidNames) {
  const std::vector<std::string> names{
      "",
      "invalid",
      "UTF8",
      "ASCII",
      "latin1",
      "UTF_32",
      "UTF-32BE",
      "UTF-32LE",
      " UTF-8",
      "UTF-8 ",
      "UTF-8x",
      "xUTF-8",
      "UTF-8\n",
      "utf\xc4\xb1-8",
      std::string("UTF-8\0", 6),
      std::string("UTF-8\0junk", 10)};
  for (const auto& name : names) {
    SCOPED_TRACE(folly::hexlify(name));
    for (const auto& bytes : {std::string(), std::string("a")}) {
      VELOX_ASSERT_USER_THROW(decode(bytes, name), "Unsupported charset");
      auto input = makeRowVector(
          {makeFlatVector<std::string>({bytes}, VARBINARY()),
           makeFlatVector<std::string>({name})});
      assertEqualVectors(
          makeNullableFlatVector<std::string>({std::nullopt}),
          evaluate("try(decode(c0, c1))", input));
      // Build a real constant expression without SQL tokenization, which
      // cannot represent every charset string (notably embedded NULs).
      auto literal = std::make_shared<core::CallTypedExpr>(
          VARCHAR(),
          "decode",
          std::make_shared<core::FieldAccessTypedExpr>(VARBINARY(), "c0"),
          std::make_shared<core::ConstantTypedExpr>(VARCHAR(), variant(name)));
      VELOX_ASSERT_USER_THROW(evaluate(literal, input), "Unsupported charset");
      assertEqualVectors(
          makeNullableFlatVector<std::string>({std::nullopt}),
          evaluate(
              std::make_shared<core::CallTypedExpr>(VARCHAR(), "try", literal),
              input));
    }
  }
}

TEST_F(DecodeTest, tryIsRowLocal) {
  auto input = makeRowVector(
      {makeNullableFlatVector<std::string>(
           {"ok", "", unhexlify("ff"), std::nullopt, ""}, VARBINARY()),
       makeFlatVector<std::string>(
           {"UTF-8", "invalid", "UTF-8", "invalid", "UTF-16"})});
  assertEqualVectors(
      makeNullableFlatVector<std::string>(
          {"ok", std::nullopt, "\xef\xbf\xbd", std::nullopt, ""}),
      evaluate("try(decode(c0, c1))", input));
  VELOX_ASSERT_USER_THROW(
      evaluate("decode(c0, c1)", input), "Unsupported charset");
}

TEST_F(DecodeTest, inactiveRows) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"ok", "bad", ""}, VARBINARY()),
       makeFlatVector<std::string>({"UTF-8", "invalid", "invalid"}),
       makeFlatVector<bool>({true, false, false})});
  assertEqualVectors(
      makeFlatVector<std::string>({"ok", "skipped", "skipped"}),
      evaluate("if(c2, decode(c0, c1), 'skipped')", input));
  SelectivityVector rows(3, false);
  rows.setValid(0, true);
  rows.updateBounds();
  auto result = evaluate("decode(c0, c1)", input, rows);
  assertEqualVectors(makeFlatVector<std::string>({"ok", "", ""}), result, rows);
  assertEqualVectors(
      makeFlatVector<std::string>({"ok", "ok", "ok"}),
      evaluate("if(c2, 'ok', if(c2, decode(c0, 'invalid'), 'ok'))", input));
}

// Hex expectations below are frozen JDK 17 replacement-mode decoding followed
// by String.getBytes(UTF_8), not output from the implementation under test.
TEST_F(DecodeTest, utf8Malformed) {
  testHex(
      "UTF-8",
      {{"80", "EFBFBD"},
       {"BF", "EFBFBD"},
       {"C080", "EFBFBDEFBFBD"},
       {"C1BF", "EFBFBDEFBFBD"},
       {"C2", "EFBFBD"},
       {"C241", "EFBFBD41"},
       {"E0", "EFBFBD"},
       {"E08080", "EFBFBDEFBFBDEFBFBD"},
       {"E09F", "EFBFBDEFBFBD"},
       {"E141", "EFBFBD41"},
       {"E18041", "EFBFBD41"},
       {"E180", "EFBFBD"},
       {"E28241", "EFBFBD41"},
       {"EDA080", "EFBFBD"},
       {"EDA0", "EFBFBD"},
       {"F0", "EFBFBD"},
       {"F0808080", "EFBFBDEFBFBDEFBFBDEFBFBD"},
       {"F090", "EFBFBD"},
       {"F09080", "EFBFBD"},
       {"F041", "EFBFBD41"},
       {"F09041", "EFBFBD41"},
       {"F0908041", "EFBFBD41"},
       {"F4908080", "EFBFBDEFBFBDEFBFBDEFBFBD"},
       {"F5808080", "EFBFBDEFBFBDEFBFBDEFBFBD"},
       {"F880808080", "EFBFBDEFBFBDEFBFBDEFBFBDEFBFBD"},
       {"FC8080808080", "EFBFBDEFBFBDEFBFBDEFBFBDEFBFBDEFBFBD"},
       {"FEFF", "EFBFBDEFBFBD"},
       {"41E28242EDA08043FF00", "41EFBFBD42EFBFBD43EFBFBD00"}});
}

TEST_F(DecodeTest, utf8ValidAndBom) {
  for (const auto& hex :
       {"",
        "00",
        "007F",
        "C280DFBF",
        "E0A080ED9FBF",
        "EFBFBF",
        "F0908080F48FBFBF",
        "EFBBBF",
        "41EFBBBF00EFBBBF42"}) {
    const auto input = unhexlify(hex);
    EXPECT_EQ(decode(input, "UTF-8"), input);
  }
}

TEST_F(DecodeTest, utf16ByteOrderAndBom) {
  testHex(
      "UTF-16BE",
      {{"", ""},
       {"FEFF", "EFBBBF"},
       {"FEFF0041FEFF", "EFBBBF41EFBBBF"},
       {"0041000000E9D83DDE00", "4100C3A9F09F9880"}});
  testHex(
      "UTF-16LE",
      {{"", ""},
       {"FFFE", "EFBBBF"},
       {"FFFE4100FFFE", "EFBBBF41EFBBBF"},
       {"41000000E9003DD800DE", "4100C3A9F09F9880"}});
  testHex(
      "UTF-16",
      {{"", ""},
       {"FEFF", ""},
       {"FFFE", ""},
       {"0041", "41"},
       {"FFFE4100", "41"},
       {"FEFF0041", "41"},
       {"0041FEFF", "41EFBBBF"},
       {"FEFFFEFF", "EFBBBF"},
       {"FFFEFFFE", "EFBBBF"},
       {"FFFEFEFF", "EFBFBE"}});
}

TEST_F(DecodeTest, utf16Malformed) {
  const std::vector<std::pair<std::string, std::string>> cases{
      {"00", "EFBFBD"},
      {"004100", "41EFBFBD"},
      {"D800", "EFBFBD"},
      {"DC00", "EFBFBD"},
      {"DFFF", "EFBFBD"},
      {"D8000041", "EFBFBD"},
      {"D80000410042", "EFBFBD42"},
      {"D800D800DC00", "EFBFBDEFBFBD"},
      {"DC000041", "EFBFBD41"},
      {"D800DC00", "F0908080"},
      {"DBFFDFFF", "F48FBFBF"},
      {"DC00DC00", "EFBFBDEFBFBD"},
      {"D800FEFF0041", "EFBFBD41"}};
  testHex("UTF-16BE", cases);
  testHex("UTF-16", cases);
  for (const auto& [hex, expected] : cases) {
    auto littleEndian = unhexlify(hex);
    for (size_t i = 0; i + 1 < littleEndian.size(); i += 2) {
      std::swap(littleEndian[i], littleEndian[i + 1]);
    }
    EXPECT_EQ(decode(littleEndian, "UTF-16LE"), unhexlify(expected));
    EXPECT_EQ(
        decode(unhexlify("FFFE") + littleEndian, "UTF-16"),
        unhexlify(expected));
  }
}

TEST_F(DecodeTest, utf16HighSurrogateOddTail) {
  testHex(
      "UTF-16BE",
      {{"D80000", "EFBFBD"},
       {"0041D800FF", "41EFBFBD"},
       {"D800004100", "EFBFBDEFBFBD"},
       {"D800DC0000", "F0908080EFBFBD"}});
  testHex(
      "UTF-16LE",
      {{"00D800", "EFBFBD"},
       {"410000D8FF", "41EFBFBD"},
       {"00D8410000", "EFBFBDEFBFBD"},
       {"00D800DC00", "F0908080EFBFBD"}});
}

TEST_F(DecodeTest, utf16ReversedMarkerJdk17) {
  // JDK 17 preserves U+FFFE; JDK 8 instead replaces this code unit.
  testHex("UTF-16BE", {{"FFFE", "EFBFBE"}, {"0041FFFE0042", "41EFBFBE42"}});
  testHex("UTF-16LE", {{"FEFF", "EFBFBE"}, {"4100FEFF4200", "41EFBFBE42"}});
}

TEST_F(DecodeTest, utf32ByteOrderAndBom) {
  testHex(
      "UTF-32",
      {{"", ""},
       {"0000FEFF", ""},
       {"FFFE0000", ""},
       {"00000041", "41"},
       {"FFFE000041000000", "41"},
       {"0000FEFF00000041", "41"},
       {"000000410000FEFF", "41EFBBBF"},
       {"0000FEFF0000FEFF", "EFBBBF"},
       {"FFFE0000FFFE0000", "EFBBBF"},
       {"FFFE00000000FEFF", "EFBFBD"},
       {"0000FEFFFFFE0000", "EFBFBD"}});
}

TEST_F(DecodeTest, utf32ScalarsAndTails) {
  testHex(
      "UTF-32",
      {{"000000000000007F00000080000007FF00000800", "007FC280DFBFE0A080"},
       {"0000D7FF0000E0000000FFFE0000FFFF", "ED9FBFEE8080EFBFBEEFBFBF"},
       {"000100000010FFFF", "F0908080F48FBFBF"},
       {"00110000", "EFBFBD"},
       {"7FFFFFFF80000000FFFFFFFF", "EFBFBDEFBFBDEFBFBD"},
       {"00", "EFBFBD"},
       {"0000", "EFBFBD"},
       {"000000", "EFBFBD"},
       {"0000004100", "41EFBFBD"},
       {"000000410000", "41EFBFBD"},
       {"00000041000000", "41EFBFBD"},
       {"FFFE00004100000000", "41EFBFBD"}});
}

TEST_F(DecodeTest, utf32SurrogateComposition) {
  testHex(
      "UTF-32",
      {{"0000D800", "3F"},
       {"0000DC00", "3F"},
       {"0000D8000000DC00", "F0908080"},
       {"0000DBFF0000DFFF", "F48FBFBF"},
       {"0000D80000000041", "3F41"},
       {"0000DC000000D800", "3F3F"},
       {"0000D8000000D8000000DC00", "3FF0908080"},
       {"0000D8000000DC000000DC00", "F09080803F"},
       {"0000D800FFFFFFFF0000DC00", "3FEFBFBD3F"},
       {"0000D800000100000000DC00", "3FF09080803F"},
       {"0000D8000000FEFF0000DC00", "3FEFBBBF3F"},
       {"0000D80000", "3FEFBFBD"},
       {"0000D8000000", "3FEFBFBD"},
       {"0000D800000000", "3FEFBFBD"},
       {"FFFE000000D8000000DC0000", "F0908080"}});
}

TEST_F(DecodeTest, encodings) {
  auto input = makeRowVector(
      {makeNullableFlatVector<std::string>(
           {unhexlify("FFFE4100"),
            unhexlify("0041"),
            "",
            unhexlify("FFFE000041000000"),
            unhexlify("00000041"),
            unhexlify("0000D8000000DC00"),
            std::nullopt},
           VARBINARY()),
       makeFlatVector<std::string>(
           {"UTF-16",
            "UTF-16",
            "UTF-32",
            "UTF-32",
            "UTF-32",
            "UTF-32",
            "UTF-8"})});
  auto expected = makeNullableFlatVector<std::string>(
      {"A", "A", "", "A", "A", unhexlify("F0908080"), std::nullopt});
  testEncodings(
      makeTypedExpr("decode(c0, c1)", asRowType(input->type())),
      input->children(),
      expected);
}

TEST_F(DecodeTest, independentlyEncodedArguments) {
  auto binary = makeFlatVector<std::string>(
      {unhexlify("0041"), "A", unhexlify("4100")}, VARBINARY());
  auto names = makeFlatVector<std::string>({"UTF-16LE", "UTF-16BE", "UTF-8"});
  auto expected = makeFlatVector<std::string>({"A", "A", "A"});
  assertEqualVectors(
      expected,
      evaluate(
          "decode(c0, c1)",
          makeRowVector(
              {binary, wrapInDictionary(makeIndices({1, 2, 0}), 3, names)})));
  assertEqualVectors(
      expected,
      evaluate(
          "decode(c0, c1)",
          makeRowVector(
              {wrapInDictionary(makeIndices({2, 0, 1}), 3, binary), names})));
  auto ascii = makeFlatVector<std::string>({"a", "b", "c"}, VARBINARY());
  auto asciiNames =
      makeFlatVector<std::string>({"UTF-8", "US-ASCII", "ISO-8859-1"});
  assertEqualVectors(
      makeFlatVector<std::string>({"a", "a", "a"}),
      evaluate(
          "decode(c0, c1)",
          makeRowVector(
              {BaseVector::wrapInConstant(3, 0, ascii), asciiNames})));
  assertEqualVectors(
      makeFlatVector<std::string>({"a", "b", "c"}),
      evaluate(
          "decode(c0, c1)",
          makeRowVector(
              {ascii, BaseVector::wrapInConstant(3, 0, asciiNames)})));
}

TEST_F(DecodeTest, reusedExpressions) {
  for (const auto& charset : {"UTF-16", "UTF-32"}) {
    auto expr = compileExpression(
        fmt::format("decode(c0, '{}')", charset), ROW({{"c0", VARBINARY()}}));
    // A BOM-free row must return to BE immediately after an LE row.
    const std::vector<std::string> hexes = std::string(charset) == "UTF-16"
        ? std::vector<
              std::string>{"FFFE4100", "0041", "FEFF0041", "", "FEFF", "D800"}
        : std::vector<std::string>{
              "FFFE000041000000",
              "00000041",
              "0000FEFF00000041",
              "",
              "0000FEFF",
              "0000D800"};
    const std::vector<std::string> expected{
        "A",
        "A",
        "A",
        "",
        "",
        std::string(charset) == "UTF-16" ? unhexlify("EFBFBD") : "?"};
    for (int repeat = 0; repeat < 2; ++repeat) {
      for (size_t i = 0; i < hexes.size(); ++i) {
        auto input = makeRowVector(
            {makeFlatVector<std::string>({unhexlify(hexes[i])}, VARBINARY())});
        assertEqualVectors(
            makeFlatVector<std::string>({expected[i]}), evaluate(*expr, input));
      }
    }
  }
  auto expr = compileExpression(
      "decode(c0, c1)", ROW({{"c0", VARBINARY()}, {"c1", VARCHAR()}}));
  for (const auto& [charset, hex] :
       std::vector<std::pair<std::string, std::string>>{
           {"UTF-16LE", "4100"},
           {"UTF-8", "41"},
           {"UTF-16BE", "0041"},
           {"UTF-16", "FFFE4100"},
           {"UTF-16", "0041"},
           {"UTF-32", "FFFE000041000000"},
           {"UTF-32", "00000041"},
           {"UTF-8", "41"}}) {
    assertEqualVectors(
        makeFlatVector<std::string>({"A"}),
        evaluate(
            *expr,
            makeRowVector(
                {makeFlatVector<std::string>({unhexlify(hex)}, VARBINARY()),
                 makeFlatVector<std::string>({charset})})));
  }
}

TEST_F(DecodeTest, ownedOutputAndBoundaries) {
  for (const auto size : {0, 1, 11, 12, 13, 31, 4096, 65536}) {
    const auto binary = std::string(size, '\xff');
    std::string replacement;
    for (int i = 0; i < size; ++i) {
      replacement += "\xef\xbf\xbd";
    }
    EXPECT_EQ(decode(binary, "US-ASCII"), replacement);
    EXPECT_EQ(decode(binary, "UTF-8"), replacement);
    const auto zeros = std::string(size, '\0');
    EXPECT_EQ(decode(zeros, "ISO-8859-1"), zeros);
  }
  for (const auto& [charset, hex] :
       std::vector<std::pair<std::string, std::string>>{
           {"UTF-16BE", "01000000"},
           {"UTF-16LE", "00010000"},
           {"UTF-16", "01000000"},
           {"UTF-32", "0000010000000000"}}) {
    std::string bytes;
    std::string expected;
    for (int i = 0; i < 4096; ++i) {
      bytes += unhexlify(hex);
      expected += unhexlify("C48000");
    }
    EXPECT_EQ(decode(bytes, charset), expected);
  }
  auto input =
      makeFlatVector<std::string>({std::string(4096, 'a')}, VARBINARY());
  auto result = evaluate("decode(c0, 'UTF-8')", makeRowVector({input}));
  const std::string nextInput(4096, 'b');
  input->set(0, StringView(nextInput));
  auto later = evaluate("decode(c0, 'UTF-8')", makeRowVector({input}));
  input.reset();
  assertEqualVectors(
      makeFlatVector<std::string>({std::string(4096, 'a')}), result);
  assertEqualVectors(
      makeFlatVector<std::string>({std::string(4096, 'b')}), later);
}

TEST_F(DecodeTest, asciiMetadata) {
  auto type = ROW({{"c0", VARBINARY()}, {"c1", VARCHAR()}});
  auto length = compileExpression("length(decode(c0, c1))", type);
  auto lower = compileExpression("lower(decode(c0, c1))", type);
  for (int repeat = 0; repeat < 2; ++repeat) {
    for (const auto& hex : {"0041", "0100", "0041"}) {
      auto binary = makeFlatVector<std::string>({unhexlify(hex)}, VARBINARY());
      auto charset = makeFlatVector<std::string>({"UTF-16BE"});
      binary->computeAndSetIsAscii(SelectivityVector(1));
      charset->computeAndSetIsAscii(SelectivityVector(1));
      auto input = makeRowVector({binary, charset});
      assertEqualVectors(
          makeFlatVector<int32_t>({1}), evaluate(*length, input));
      assertEqualVectors(
          makeFlatVector<std::string>(
              {std::string(hex) == "0100" ? unhexlify("C481") : "a"}),
          evaluate(*lower, input));
    }
  }
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
