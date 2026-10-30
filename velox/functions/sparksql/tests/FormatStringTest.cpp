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

namespace facebook::velox::functions::sparksql::test {
namespace {

class FormatStringTest : public SparkFunctionBaseTest {};

TEST_F(FormatStringTest, stringsAndSequentialArguments) {
  auto strings =
      makeNullableFlatVector<StringView>({"hello", std::nullopt, "world"});
  auto integers = makeFlatVector<int32_t>({1, 2, 3});
  auto result = evaluate<SimpleVector<StringView>>(
      "format_string('%s:%d:%%', c0, c1)", makeRowVector({strings, integers}));
  auto expected =
      makeFlatVector<StringView>({"hello:1:%", "null:2:%", "world:3:%"});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(FormatStringTest, booleanStringFormatting) {
  auto booleans = makeNullableFlatVector<bool>({true, false, std::nullopt});
  auto result = evaluate<SimpleVector<StringView>>(
      "format_string('%s', c0)", makeRowVector({booleans}));
  auto expected = makeFlatVector<StringView>({"true", "false", "null"});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(FormatStringTest, ignoresExtraArguments) {
  auto integers = makeFlatVector<int32_t>({1, 2, 3});
  auto strings = makeFlatVector<StringView>({"unused", "values", "ignored"});
  auto result = evaluate<SimpleVector<StringView>>(
      "format_string('value=%d', c0, c1)", makeRowVector({integers, strings}));
  auto expected = makeFlatVector<StringView>({"value=1", "value=2", "value=3"});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(FormatStringTest, encodedPatternsAndArguments) {
  auto patterns =
      makeFlatVector<StringView>({"%s:%d", "[%s]=%04d", "%s/%+d", "%s:%d"});
  auto strings =
      makeNullableFlatVector<StringView>({"a", "b", std::nullopt, "d"});
  auto integers = makeFlatVector<int32_t>({1, 2, 3, -4});
  auto expected =
      makeFlatVector<StringView>({"a:1", "[b]=0002", "null/+3", "d:-4"});
  auto expression = makeTypedExpr(
      "format_string(c0, c1, c2)",
      ROW({"c0", "c1", "c2"}, {VARCHAR(), VARCHAR(), INTEGER()}));

  testEncodings(expression, {patterns, strings, integers}, expected);

  const auto makeLazy = [&](const VectorPtr& vector) -> VectorPtr {
    return std::make_shared<LazyVector>(
        execCtx_.pool(),
        vector->type(),
        vector->size(),
        std::make_unique<velox::test::SimpleVectorLoader>(
            [vector](auto /*rows*/) { return vector; }));
  };
  auto result = evaluate<SimpleVector<StringView>>(
      expression,
      makeRowVector(
          {makeLazy(patterns), makeLazy(strings), makeLazy(integers)}));
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(FormatStringTest, replacesMalformedUtf8InStringArgument) {
  // Spark formats a %s argument via UTF8String.toString(), which decodes the
  // bytes through java.lang.String and replaces malformed UTF-8 with U+FFFD.
  // Strings built from binary data follow the same behavior. See
  // SELECT hex(format_string('%s', CAST(unhex('FF') AS STRING))).
  // The expected replacement counts match OpenJDK 21's UTF-8 decoder:
  //   FF          -> 1  (isolated invalid byte)
  //   E2 82       -> 1  (truncated 3-byte lead)
  //   E0 80 80    -> 3  (overlong, each byte replaced separately)
  //   E1 80       -> 1  (valid 3-byte prefix, truncated)
  //   E0 80       -> 2  (invalid second byte for E0, then stray continuation)
  //   ED A0 80    -> 1  (surrogate encoding, single replacement)
  //   F4 90 80 80 -> 4  (code point above U+10FFFF, each byte replaced)
  const std::string kReplacement("\xEF\xBF\xBD", 3); // U+FFFD, UTF-8 encoded.
  const std::string kTwo = kReplacement + kReplacement;
  const std::string kThree = kReplacement + kReplacement + kReplacement;
  const std::string kFour = kThree + kReplacement;
  const std::string isolatedInvalidByte("\xFF", 1);
  const std::string truncatedSequence("\xE2\x82", 2);
  const std::string overlongSequence("\xE0\x80\x80", 3);
  const std::string truncatedValidPrefix("\xE1\x80", 2);
  const std::string invalidSecondByte("\xE0\x80", 2);
  const std::string surrogateEncoding("\xED\xA0\x80", 3);
  const std::string aboveMaxCodePoint("\xF4\x90\x80\x80", 4);
  const std::string valid("h\xC3\xA9llo", 6); // Valid UTF-8 text.
  auto strings = makeFlatVector<StringView>(
      {StringView(isolatedInvalidByte),
       StringView(truncatedSequence),
       StringView(overlongSequence),
       StringView(truncatedValidPrefix),
       StringView(invalidSecondByte),
       StringView(surrogateEncoding),
       StringView(aboveMaxCodePoint),
       StringView(valid)});
  auto result = evaluate<SimpleVector<StringView>>(
      "format_string('%s', c0)", makeRowVector({strings}));
  auto expected = makeFlatVector<StringView>(
      {StringView(kReplacement),
       StringView(kReplacement),
       StringView(kThree),
       StringView(kReplacement),
       StringView(kTwo),
       StringView(kReplacement),
       StringView(kFour),
       StringView(valid)});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(FormatStringTest, replacesMalformedUtf8InConstantPattern) {
  const std::string kReplacement("\xEF\xBF\xBD", 3);
  const std::string kTwo = kReplacement + kReplacement;
  // E0 80 decodes to two U+FFFD (invalid second byte, then stray continuation),
  // verifying replacement grouping for a constant pattern.
  const std::string malformedPattern = std::string("\xE0\x80", 2) + "%s";
  auto patterns = makeConstant(StringView(malformedPattern), 3);
  auto strings = makeFlatVector<StringView>(
      {StringView("x"), StringView("y"), StringView("z")});
  auto result = evaluate<SimpleVector<StringView>>(
      "format_string(c0, c1)", makeRowVector({patterns, strings}));
  const std::string expectedX = kTwo + "x";
  const std::string expectedY = kTwo + "y";
  const std::string expectedZ = kTwo + "z";
  auto expected = makeFlatVector<StringView>(
      {StringView(expectedX), StringView(expectedY), StringView(expectedZ)});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(FormatStringTest, replacesMalformedUtf8InNonConstantPattern) {
  // Spark decodes the format pattern through UTF8String.toString() as well, so
  // malformed UTF-8 in the literal portion of the pattern is replaced with
  // U+FFFD before formatting.
  const std::string kReplacement("\xEF\xBF\xBD", 3); // U+FFFD, UTF-8 encoded.
  const std::string malformedPattern = std::string("\xFF", 1) + "%s";
  const std::string validPattern("h\xC3\xA9llo:%s", 9); // Valid UTF-8 literal.
  auto patterns = makeFlatVector<StringView>(
      {StringView(malformedPattern), StringView(validPattern)});
  auto strings = makeFlatVector<StringView>({StringView("x"), StringView("y")});
  auto result = evaluate<SimpleVector<StringView>>(
      "format_string(c0, c1)", makeRowVector({patterns, strings}));
  const std::string expectedMalformed = kReplacement + "x";
  const std::string expectedValid = std::string("h\xC3\xA9llo:", 7) + "y";
  auto expected = makeFlatVector<StringView>(
      {StringView(expectedMalformed), StringView(expectedValid)});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(FormatStringTest, signedIntegralFormatting) {
  auto integers = makeFlatVector<int32_t>({42, -42, 0});
  auto result = evaluate<SimpleVector<StringView>>(
      "format_string('%d|%05d|%-5d|%+d|% d', c0, c0, c0, c0, c0)",
      makeRowVector({integers}));
  auto expected = makeFlatVector<StringView>(
      {"42|00042|42   |+42| 42",
       "-42|-0042|-42  |-42|-42",
       "0|00000|0    |+0| 0"});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(FormatStringTest, unsignedIntegralFormattingPreservesSourceWidth) {
  auto tinyints = makeFlatVector<int8_t>({-1, 127});
  auto smallints = makeFlatVector<int16_t>({-1, 256});
  auto integers = makeFlatVector<int32_t>({-1, 255});
  auto bigints = makeFlatVector<int64_t>({-1, 4096});
  auto result = evaluate<SimpleVector<StringView>>(
      "format_string('%x|%X|%o|%x', c0, c1, c2, c3)",
      makeRowVector({tinyints, smallints, integers, bigints}));
  auto expected = makeFlatVector<StringView>(
      {"ff|FFFF|37777777777|ffffffffffffffff", "7f|100|377|1000"});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(FormatStringTest, nullIntegralArguments) {
  auto integers =
      makeNullableFlatVector<int32_t>({std::nullopt, 42, std::nullopt});
  auto result = evaluate<SimpleVector<StringView>>(
      "format_string('%5d|%X|%o', c0, c0, c0)", makeRowVector({integers}));
  auto expected = makeFlatVector<StringView>(
      {" null|NULL|null", "   42|2A|52", " null|NULL|null"});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(FormatStringTest, nullFormatString) {
  auto formats =
      makeNullableFlatVector<StringView>({std::nullopt, "%d", "value=%d"});
  auto integers = makeFlatVector<int32_t>({1, 2, 3});
  auto result = evaluate<SimpleVector<StringView>>(
      "format_string(c0, c1)", makeRowVector({formats, integers}));
  auto expected =
      makeNullableFlatVector<StringView>({std::nullopt, "2", "value=3"});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(FormatStringTest, nullFormatStringSkipsArguments) {
  auto formats = makeNullableFlatVector<StringView>({std::nullopt, "%s"});
  auto messages = makeFlatVector<StringView>({"%q", "ok"});
  auto result = evaluate<SimpleVector<StringView>>(
      "format_string(c0, format_string(c1))",
      makeRowVector({formats, messages}));

  auto expected = makeNullableFlatVector<StringView>({std::nullopt, "ok"});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(FormatStringTest, preservesRowsInConditionalExpression) {
  auto conditions = makeFlatVector<bool>({true, false, false, true});
  auto integers = makeFlatVector<int32_t>({1, 2, 3, 4});
  auto fallback =
      makeFlatVector<StringView>({"unused", "fallback-2", "fallback-3", "x"});
  auto result = evaluate<SimpleVector<StringView>>(
      "if(c0, format_string('id=%04d', c1), c2)",
      makeRowVector({conditions, integers, fallback}));
  auto expected = makeFlatVector<StringView>(
      {"id=0001", "fallback-2", "fallback-3", "id=0004"});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(FormatStringTest, printfAlias) {
  auto integers = makeFlatVector<int32_t>({1, 42});
  auto result = evaluate<SimpleVector<StringView>>(
      "printf('id=%04d', c0)", makeRowVector({integers}));
  auto expected = makeFlatVector<StringView>({"id=0001", "id=0042"});
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(FormatStringTest, tryHandlesRowLocalErrors) {
  auto formats = makeFlatVector<StringView>({"%d", "%d %d", "%d"});
  auto integers = makeFlatVector<int32_t>({1, 2, 3});
  auto result = evaluate<SimpleVector<StringView>>(
      "try(format_string(c0, c1))", makeRowVector({formats, integers}));
  auto expected = makeNullableFlatVector<StringView>({"1", std::nullopt, "3"});

  velox::test::assertEqualVectors(expected, result);

  auto constantError = evaluate<SimpleVector<StringView>>(
      "try(format_string('%d %d', c0))", makeRowVector({integers}));
  velox::test::assertEqualVectors(
      makeAllNullFlatVector<StringView>(integers->size()), constantError);
}

TEST_F(FormatStringTest, rejectsDecimals) {
  auto shortDecimals = makeFlatVector<int64_t>({125}, DECIMAL(5, 2));
  auto longDecimals = makeFlatVector<int128_t>({125}, DECIMAL(20, 2));

  for (const auto& format : {"%s", "%d", "%x"}) {
    VELOX_ASSERT_USER_THROW(
        evaluate<SimpleVector<StringView>>(
            std::string("format_string('") + format + "', c0)",
            makeRowVector({shortDecimals})),
        "format_string does not support decimal arguments");
    VELOX_ASSERT_USER_THROW(
        evaluate<SimpleVector<StringView>>(
            std::string("format_string('") + format + "', c0)",
            makeRowVector({longDecimals})),
        "format_string does not support decimal arguments");
  }
}

TEST_F(FormatStringTest, rejectsUnsupportedConversionsAndOptions) {
  auto integers = makeFlatVector<int32_t>({1});
  auto doubles = makeFlatVector<double>({1.25});
  auto strings = makeFlatVector<StringView>({"value"});

  VELOX_ASSERT_USER_THROW(
      evaluateOnce<StringView>("format_string()"),
      "format_string requires at least one argument");
  VELOX_ASSERT_USER_THROW(
      evaluate<SimpleVector<StringView>>(
          "format_string(c0)", makeRowVector({integers})),
      "The first argument of format_string must be a varchar: INTEGER");
  VELOX_ASSERT_USER_THROW(
      evaluate<SimpleVector<StringView>>(
          "format_string('%.2f', c0)", makeRowVector({doubles})),
      "Unsupported format conversion character: 'f'");
  VELOX_ASSERT_USER_THROW(
      evaluate<SimpleVector<StringView>>(
          "format_string('%g', c0)", makeRowVector({doubles})),
      "Unsupported format conversion character: 'g'");
  VELOX_ASSERT_USER_THROW(
      evaluate<SimpleVector<StringView>>(
          "format_string('%10s', c0)", makeRowVector({strings})),
      "format_string supports only bare %s");
  VELOX_ASSERT_USER_THROW(
      evaluate<SimpleVector<StringView>>(
          "format_string('%s', c0)", makeRowVector({doubles})),
      "Unsupported type for format_string: DOUBLE");
  VELOX_ASSERT_USER_THROW(
      evaluate<SimpleVector<StringView>>(
          "format_string('%1$d', c0)", makeRowVector({integers})),
      "Unsupported format conversion character: '$'");
  VELOX_ASSERT_USER_THROW(
      evaluate<SimpleVector<StringView>>(
          "format_string('%d', c0)", makeRowVector({doubles})),
      "format_string: %d requires an integral type, got DOUBLE");
}

TEST_F(FormatStringTest, rejectsMalformedFormatsAndMissingArguments) {
  auto integers = makeFlatVector<int32_t>({1});
  VELOX_ASSERT_USER_THROW(
      evaluate<SimpleVector<StringView>>(
          "format_string('%--5d', c0)", makeRowVector({integers})),
      "Duplicate flag in format_string: '-'");
  VELOX_ASSERT_USER_THROW(
      evaluate<SimpleVector<StringView>>(
          "format_string('%99999999d', c0)", makeRowVector({integers})),
      "format_string size exceeds supported maximum for width");
  VELOX_ASSERT_USER_THROW(
      evaluate<SimpleVector<StringView>>(
          "format_string('%.99999999d', c0)", makeRowVector({integers})),
      "format_string size exceeds supported maximum for precision");
  VELOX_ASSERT_USER_THROW(
      evaluate<SimpleVector<StringView>>(
          "format_string('%d %d', c0)", makeRowVector({integers})),
      "Not enough arguments for format_string: specifier '%d'");
  VELOX_ASSERT_USER_THROW(
      evaluate<SimpleVector<StringView>>(
          "format_string('%', c0)", makeRowVector({integers})),
      "Incomplete format specifier in format_string");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
