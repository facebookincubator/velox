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

#include <limits>
#include <optional>
#include <string>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/type/tests/utils/CustomTypesForTesting.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

using facebook::velox::test::assertEqualVectors;

class ToCsvTest : public SparkFunctionBaseTest {
 protected:
  std::optional<std::string> toCsv(
      const VectorPtr& input,
      const std::vector<std::string>& configArgs = {}) {
    std::string expr = "to_csv(c0";
    for (size_t i = 0; i < configArgs.size(); ++i) {
      expr += fmt::format(", '{}'", configArgs[i]);
    }
    expr += ")";
    return evaluateOnce<std::string>(expr, makeRowVector({input}));
  }
};

TEST_F(ToCsvTest, basicStruct) {
  auto input = makeRowVector(
      {"a", "b", "c"},
      {makeFlatVector<int32_t>({1}),
       makeFlatVector<StringView>({"hello"}),
       makeFlatVector<double>({3.14})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "1,hello,3.14");
}

TEST_F(ToCsvTest, nullStruct) {
  auto input = makeRowVector({"a"}, {makeFlatVector<int32_t>({1})});
  auto nullInput = BaseVector::createNullConstant(input->type(), 1, pool());
  auto result = toCsv(nullInput);
  EXPECT_FALSE(result.has_value());
}

TEST_F(ToCsvTest, nonRowInput) {
  auto input = makeFlatVector<int32_t>({1});
  VELOX_ASSERT_THROW(toCsv(input), "to_csv: input must be a ROW, got INTEGER");
}

TEST_F(ToCsvTest, nullFields) {
  auto input = makeRowVector(
      {"a", "b", "c"},
      {makeNullableFlatVector<int32_t>({1}),
       makeNullableFlatVector<StringView>({std::nullopt}),
       makeNullableFlatVector<int32_t>({3})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "1,,3");
}

TEST_F(ToCsvTest, vectorEncodingsAndSelectedRows) {
  auto input = makeRowVector(
      {"a", "b"},
      {makeFlatVector<int32_t>({1, 3, 5}),
       makeFlatVector<StringView>({"x", "y", "z"})});
  input->setNull(1, true);

  {
    SCOPED_TRACE("dictionary encoding with dynamic option and mixed nulls");
    auto dictionaryInput = wrapInDictionary(makeIndices({2, 1, 0, 2}), input);
    auto config =
        makeFlatVector<StringView>({"sep=|", "sep=|", "sep=|", "sep=|"});
    auto result = evaluate<FlatVector<StringView>>(
        "to_csv(c0, c1)", makeRowVector({dictionaryInput, config}));
    auto expected =
        makeNullableFlatVector<StringView>({"5|z", std::nullopt, "1|x", "5|z"});
    assertEqualVectors(expected, result);
  }

  {
    SCOPED_TRACE("constant encoding");
    auto constantInput = BaseVector::wrapInConstant(3, 2, input);
    auto result = evaluate<FlatVector<StringView>>(
        "to_csv(c0)", makeRowVector({constantInput}));
    auto expected = makeFlatVector<StringView>({"5,z", "5,z", "5,z"});
    assertEqualVectors(expected, result);
  }

  {
    SCOPED_TRACE("partial selection with pre-existing result");
    auto dictionaryInput = wrapInDictionary(makeIndices({2, 1, 0, 2}), input);
    auto condition = makeFlatVector<bool>({true, false, true, false});
    auto result = evaluate<FlatVector<StringView>>(
        "if(c1, to_csv(c0), 'skipped')",
        makeRowVector({dictionaryInput, condition}));
    auto expected =
        makeFlatVector<StringView>({"5,z", "skipped", "1,x", "skipped"});
    assertEqualVectors(expected, result);
  }
}

TEST_F(ToCsvTest, customDelimiter) {
  auto input = makeRowVector(
      {"a", "b"}, {makeFlatVector<int32_t>({1}), makeFlatVector<int32_t>({2})});
  auto result = toCsv(input, {"sep=|"});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "1|2");
}

TEST_F(ToCsvTest, dynamicConfig) {
  auto input = makeRowVector(
      {"a", "b"}, {makeFlatVector<int32_t>({1}), makeFlatVector<int32_t>({2})});

  {
    SCOPED_TRACE("case-insensitive dynamic option");
    auto config = makeFlatVector<StringView>({"SeP=|"});
    auto result = evaluateOnce<std::string>(
        "to_csv(c0, c1)", makeRowVector({input, config}));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "1|2");
  }

  {
    SCOPED_TRACE("dynamic option overrides earlier constant");
    auto config = makeFlatVector<StringView>({"sep=;"});
    auto result = evaluateOnce<std::string>(
        "to_csv(c0, 'sep=|', c1)", makeRowVector({input, config}));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "1;2");
  }

  {
    SCOPED_TRACE("constant option overrides earlier dynamic option");
    auto config = makeFlatVector<StringView>({"sep=;"});
    auto result = evaluateOnce<std::string>(
        "to_csv(c0, 'sep=|', c1, 'sep=:')", makeRowVector({input, config}));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "1:2");
  }
}

TEST_F(ToCsvTest, nullConfigIsIgnored) {
  auto input = makeRowVector(
      {"a", "b"}, {makeFlatVector<int32_t>({1}), makeFlatVector<int32_t>({2})});

  {
    SCOPED_TRACE("dynamic null option");
    auto config = makeNullableFlatVector<StringView>({std::nullopt});
    auto result = evaluateOnce<std::string>(
        "to_csv(c0, c1)", makeRowVector({input, config}));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "1,2");
  }

  {
    SCOPED_TRACE("constant null option");
    auto result = evaluateOnce<std::string>(
        "to_csv(c0, 'sep=;', cast(null as varchar), 'sep=|')",
        makeRowVector({input}));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "1|2");
  }
}

TEST_F(ToCsvTest, moreThanFiveOptions) {
  auto input = makeRowVector(
      {"a", "b"}, {makeFlatVector<int32_t>({1}), makeFlatVector<int32_t>({2})});
  auto result = toCsv(
      input,
      {"sep=|",
       "quote=~",
       "escape=#",
       "nullValue=NULL",
       "delimiter=:",
       "sep=;"});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "1;2");
}

TEST_F(ToCsvTest, invalidConfig) {
  auto input = makeRowVector({"a"}, {makeFlatVector<int32_t>({1})});
  VELOX_ASSERT_THROW(
      toCsv(input, {"bad"}), "to_csv: option must use key=value format: bad");
  VELOX_ASSERT_THROW(
      toCsv(input, {"header=true"}), "to_csv: unsupported option: header");
  VELOX_ASSERT_THROW(
      toCsv(input, {"sep=||"}),
      "to_csv: separator must be exactly 1 character, got '||'");
}

TEST_F(ToCsvTest, invalidConfigWithTry) {
  auto input = makeRowVector(
      {"a", "b"},
      {makeFlatVector<int32_t>({1, 3, 5}), makeFlatVector<int32_t>({2, 4, 6})});

  {
    SCOPED_TRACE("dynamic option");
    auto config = makeFlatVector<StringView>({"sep=|", "bad", "sep=;"});
    auto result = evaluate<FlatVector<StringView>>(
        "try(to_csv(c0, c1))", makeRowVector({input, config}));
    auto expected =
        makeNullableFlatVector<StringView>({"1|2", std::nullopt, "5;6"});
    assertEqualVectors(expected, result);
  }

  {
    SCOPED_TRACE("constant option");
    auto result = evaluate<FlatVector<StringView>>(
        "try(to_csv(c0, 'bad'))", makeRowVector({input}));
    auto expected = makeNullableFlatVector<StringView>(
        {std::nullopt, std::nullopt, std::nullopt});
    assertEqualVectors(expected, result);
  }
}

TEST_F(ToCsvTest, customNullValue) {
  auto input = makeRowVector(
      {"a", "b"},
      {makeFlatVector<int32_t>({1}),
       makeNullableFlatVector<int32_t>({std::nullopt})});
  auto result = toCsv(input, {"nullValue=NULL"});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "1,NULL");
}

TEST_F(ToCsvTest, quotingRequired) {
  // String containing delimiter should be quoted.
  auto input = makeRowVector(
      {"a", "b"},
      {makeFlatVector<StringView>({"hello,world"}),
       makeFlatVector<int32_t>({42})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "\"hello,world\",42");
}

TEST_F(ToCsvTest, quotingWithEmbeddedQuote) {
  // Default escape is backslash (Spark's CSVOptions default), so an embedded
  // quote is escaped as \" -- matching Spark, not RFC 4180 doubling.
  auto input =
      makeRowVector({"a"}, {makeFlatVector<StringView>({"say \"hi\""})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "\"say \\\"hi\\\"\"");
}

TEST_F(ToCsvTest, embeddedBackslashAndQuote) {
  // Field a"b preceded by a backslash is quoted (contains a quote); both the
  // literal backslash (the escape char) and the quote are escaped with
  // backslash, matching Spark's UnivocityGenerator: a\"b -> "a\\\"b".
  auto input = makeRowVector({"a"}, {makeFlatVector<StringView>({"a\\\"b"})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "\"a\\\\\\\"b\"");
}

TEST_F(ToCsvTest, backslashOnlyNotQuoted) {
  // A backslash alone does not trigger quoting (no delimiter/quote/newline),
  // so it is emitted literally: a\b -> a\b. Matches Spark.
  auto input = makeRowVector({"a"}, {makeFlatVector<StringView>({"a\\b"})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "a\\b");
}

TEST_F(ToCsvTest, customEscapeChar) {
  // Explicit escape '#': an embedded quote is escaped as #".
  auto input = makeRowVector({"a"}, {makeFlatVector<StringView>({"x\"y"})});
  auto result = toCsv(input, {"escape=#"});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "\"x#\"y\"");
}

TEST_F(ToCsvTest, escapeEqualsQuoteDoubles) {
  // When escape is explicitly set equal to the quote char, embedded quotes are
  // doubled (RFC 4180), matching Spark with escape='"'.
  auto input =
      makeRowVector({"a"}, {makeFlatVector<StringView>({"say \"hi\""})});
  auto result = toCsv(input, {"escape=\""});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "\"say \"\"hi\"\"\"");
}

TEST_F(ToCsvTest, doubleSpecialValues) {
  // NaN/Infinity formatting must match Spark (Java Double.toString): "NaN",
  // "Infinity", "-Infinity".
  auto input = makeRowVector(
      {"a", "b", "c"},
      {makeFlatVector<double>({std::numeric_limits<double>::quiet_NaN()}),
       makeFlatVector<double>({std::numeric_limits<double>::infinity()}),
       makeFlatVector<double>({-std::numeric_limits<double>::infinity()})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "NaN,Infinity,-Infinity");
}

TEST_F(ToCsvTest, doubleFormatting) {
  // Whole-number doubles keep ".0"; out-of-range magnitudes use Java-style
  // scientific notation. Matches Spark.
  auto input = makeRowVector(
      {"a", "b", "c", "d", "e"},
      {makeFlatVector<double>({5.0}),
       makeFlatVector<double>({0.0000001}),
       makeFlatVector<double>({1.0e20}),
       makeFlatVector<double>({0.001}),
       makeFlatVector<double>({1.0e7})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "5.0,1.0E-7,1.0E20,0.001,1.0E7");
}

TEST_F(ToCsvTest, booleanValues) {
  auto input = makeRowVector(
      {"a", "b"},
      {makeFlatVector<bool>({true}), makeFlatVector<bool>({false})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "true,false");
}

TEST_F(ToCsvTest, allNumericTypes) {
  auto input = makeRowVector(
      {"tinyint", "smallint", "int", "bigint"},
      {makeFlatVector<int8_t>({-1}),
       makeFlatVector<int16_t>({256}),
       makeFlatVector<int32_t>({100'000}),
       makeFlatVector<int64_t>({9'876'543'210L})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "-1,256,100000,9876543210");
}

TEST_F(ToCsvTest, realSpecialValues) {
  // Float NaN/Infinity formatting must match Spark (Java Float.toString):
  // "NaN", "Infinity", "-Infinity".
  auto input = makeRowVector(
      {"a", "b", "c"},
      {makeFlatVector<float>({std::numeric_limits<float>::quiet_NaN()}),
       makeFlatVector<float>({std::numeric_limits<float>::infinity()}),
       makeFlatVector<float>({-std::numeric_limits<float>::infinity()})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "NaN,Infinity,-Infinity");
}

TEST_F(ToCsvTest, realFormatting) {
  // Match Java Float.toString without spurious double-precision digits.
  auto input = makeRowVector(
      {"a", "b", "c", "d", "e"},
      {makeFlatVector<float>({3.14f}),
       makeFlatVector<float>({5.0f}),
       makeFlatVector<float>({0.1f}),
       makeFlatVector<float>({1.0e10f}),
       makeFlatVector<float>({1.0e-7f})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "3.14,5.0,0.1,1.0E10,1.0E-7");
}

TEST_F(ToCsvTest, realNegativeZero) {
  // Java's Float.toString(-0.0f) returns "-0.0".
  auto input = makeRowVector({"a"}, {makeFlatVector<float>({-0.0f})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "-0.0");
}

TEST_F(ToCsvTest, doubleNegativeZero) {
  // Java's Double.toString(-0.0) returns "-0.0".
  auto input = makeRowVector({"a"}, {makeFlatVector<double>({-0.0})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "-0.0");
}

TEST_F(ToCsvTest, emptyString) {
  // Spark renders a non-null empty string field as the literal two-quote
  // string "" (emptyValueInWrite default), distinct from a NULL field which
  // uses nullValue (empty by default).
  auto input = makeRowVector(
      {"a", "b"},
      {makeFlatVector<StringView>({""}), makeFlatVector<int32_t>({1})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "\"\",1");
}

TEST_F(ToCsvTest, leadingTrailingWhitespaceTrimmed) {
  // Spark's write defaults trim leading/trailing whitespace
  // (ignoreLeadingWhiteSpace/ignoreTrailingWhiteSpace = true). Internal
  // whitespace is preserved.
  auto input = makeRowVector(
      {"a", "b"},
      {makeFlatVector<StringView>({"  a   b  "}),
       makeFlatVector<int32_t>({1})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "a   b,1");
}

TEST_F(ToCsvTest, whitespaceOnlyBecomesEmptyValue) {
  // A whitespace-only string trims to empty, then renders as "".
  auto input = makeRowVector({"a"}, {makeFlatVector<StringView>({"   "})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "\"\"");
}

TEST_F(ToCsvTest, trailingNewlineTrimmedButQuoted) {
  // The quoting decision uses the raw value (the newline triggers quoting),
  // but the trailing newline is trimmed from the emitted content: "ab\n" ->
  // "ab" (quoted, no newline). Matches Spark's UnivocityGenerator.
  auto input = makeRowVector({"a"}, {makeFlatVector<StringView>({"ab\n"})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "\"ab\"");
}

TEST_F(ToCsvTest, newlineInString) {
  // String containing newline should be quoted.
  auto input =
      makeRowVector({"a"}, {makeFlatVector<StringView>({"line1\nline2"})});
  auto result = toCsv(input);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "\"line1\nline2\"");
}

TEST_F(ToCsvTest, customQuoteChar) {
  // Explicit quote '|': a value containing the configured quote char is quoted,
  // and the embedded quote is escaped with the default escape '\'.
  auto input = makeRowVector({"a"}, {makeFlatVector<StringView>({"x|y"})});
  auto result = toCsv(input, {"quote=|"});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "|x\\|y|");
}

TEST_F(ToCsvTest, unsupportedTypes) {
  {
    SCOPED_TRACE("timestamp");
    auto input =
        makeRowVector({"a"}, {makeFlatVector<Timestamp>({Timestamp(0, 0)})});
    VELOX_ASSERT_THROW(
        toCsv(input), "to_csv: unsupported field type at index 0: TIMESTAMP");
  }

  {
    SCOPED_TRACE("short decimal");
    auto input =
        makeRowVector({"a"}, {makeFlatVector<int64_t>({123}, DECIMAL(5, 2))});
    VELOX_ASSERT_THROW(
        toCsv(input),
        "to_csv: unsupported field type at index 0: DECIMAL(5, 2)");
  }

  {
    SCOPED_TRACE("long decimal");
    auto input = makeRowVector(
        {"a"}, {makeFlatVector<int128_t>({12'345}, DECIMAL(20, 2))});
    VELOX_ASSERT_THROW(
        toCsv(input),
        "to_csv: unsupported field type at index 0: DECIMAL(20, 2)");
  }

  {
    SCOPED_TRACE("date");
    auto input = makeRowVector({"a"}, {makeFlatVector<int32_t>({0}, DATE())});
    VELOX_ASSERT_THROW(
        toCsv(input), "to_csv: unsupported field type at index 0: DATE");
  }

  {
    SCOPED_TRACE("custom logical type");
    auto input = makeRowVector(
        {"a"},
        {makeFlatVector<int64_t>(
            {1}, facebook::velox::test::BIGINT_TYPE_WITH_CUSTOM_COMPARISON())});
    VELOX_ASSERT_THROW(
        toCsv(input),
        "to_csv: unsupported field type at index 0: BIGINT TYPE WITH CUSTOM COMPARISON");
  }
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
