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
#include <cmath>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/Expressions.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

inline core::CallTypedExprPtr createFromCsv(const TypePtr& outputType) {
  std::vector<core::TypedExprPtr> inputs = {
      std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0")};
  return std::make_shared<const core::CallTypedExpr>(
      outputType, std::move(inputs), "from_csv");
}

class FromCsvTest : public SparkFunctionBaseTest {
 protected:
  void testFromCsv(const VectorPtr& input, const VectorPtr& expected) {
    auto expr = createFromCsv(expected->type());
    testEncodings(expr, {input}, expected);
  }
};

// Basic struct with integer and double fields.
TEST_F(FromCsvTest, basicStruct) {
  auto input = makeFlatVector<std::string>({"1,1.1", "2,2.2", "3,3.3"});
  auto expectedA = makeFlatVector<int32_t>({1, 2, 3});
  auto expectedB = makeFlatVector<double>({1.1, 2.2, 3.3});
  auto expected = makeRowVector({"a", "b"}, {expectedA, expectedB});
  testFromCsv(input, expected);

  // Verify RowVector itself is not null for valid input rows.
  auto expr = createFromCsv(expected->type());
  auto result = evaluate(expr, makeRowVector({input}));
  for (auto i = 0; i < 3; ++i) {
    EXPECT_FALSE(result->isNullAt(i));
  }
}

// Basic struct with string fields.
TEST_F(FromCsvTest, stringFields) {
  auto input =
      makeFlatVector<std::string>({"hello,world", "foo,bar", "one,two"});
  auto expectedA = makeFlatVector<std::string>({"hello", "foo", "one"});
  auto expectedB = makeFlatVector<std::string>({"world", "bar", "two"});
  auto expected = makeRowVector({"a", "b"}, {expectedA, expectedB});
  testFromCsv(input, expected);
}

TEST_F(FromCsvTest, realEdgeCases) {
  auto input = makeFlatVector<std::string>(
      {"3.4028236e38", "1e-50", "-1e-50", "NaN", "+NaN", "Inf", "-Inf"});
  auto expr = createFromCsv(ROW({"a"}, {REAL()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* child = result->as<RowVector>()->childAt(0)->asFlatVector<float>();
  EXPECT_TRUE(std::isinf(child->valueAt(0)));
  EXPECT_GT(child->valueAt(0), 0);
  EXPECT_EQ(child->valueAt(1), 0.0f);
  EXPECT_FALSE(std::signbit(child->valueAt(1)));
  EXPECT_EQ(child->valueAt(2), 0.0f);
  EXPECT_TRUE(std::signbit(child->valueAt(2)));
  EXPECT_TRUE(std::isnan(child->valueAt(3)));
  EXPECT_TRUE(std::isnan(child->valueAt(4)));
  EXPECT_TRUE(std::isinf(child->valueAt(5)));
  EXPECT_GT(child->valueAt(5), 0);
  EXPECT_TRUE(std::isinf(child->valueAt(6)));
  EXPECT_LT(child->valueAt(6), 0);
}

// Null input returns null row.
TEST_F(FromCsvTest, nullInput) {
  auto input =
      makeNullableFlatVector<std::string>({std::nullopt, "1,2", std::nullopt});
  auto expectedA =
      makeNullableFlatVector<int32_t>({std::nullopt, 1, std::nullopt});
  auto expectedB =
      makeNullableFlatVector<int32_t>({std::nullopt, 2, std::nullopt});
  auto expected = makeRowVector({"a", "b"}, {expectedA, expectedB});

  auto expr = createFromCsv(expected->type());
  auto result = evaluate(expr, makeRowVector({input}));
  EXPECT_TRUE(result->equalValueAt(expected.get(), 1, 1));

  // The result may be wrapped in a DictionaryVector when the expression engine
  // extends the vector to cover null-input rows, so use BaseVector::isNullAt
  // instead of casting to RowVector.
  for (int i = 0; i < 3; ++i) {
    if (i == 0 || i == 2) {
      EXPECT_TRUE(result->isNullAt(i)) << "row " << i;
    } else {
      EXPECT_FALSE(result->isNullAt(i)) << "row " << i;
    }
  }
}

// Fewer fields than schema — missing fields become null.
TEST_F(FromCsvTest, fewerFields) {
  auto input = makeFlatVector<std::string>({"1", "2"});
  auto expectedA = makeNullableFlatVector<int32_t>({1, 2});
  auto expectedB = makeNullableFlatVector<double>({std::nullopt, std::nullopt});
  auto expected = makeRowVector({"a", "b"}, {expectedA, expectedB});
  testFromCsv(input, expected);
}

// More fields than schema — extra fields are ignored.
TEST_F(FromCsvTest, extraFields) {
  auto input = makeFlatVector<std::string>(
      {"1,2.5,extra", "3,4.5,ignored", "5,6.5,dropped"});
  auto expectedA = makeFlatVector<int32_t>({1, 3, 5});
  auto expectedB = makeFlatVector<double>({2.5, 4.5, 6.5});
  auto expected = makeRowVector({"a", "b"}, {expectedA, expectedB});
  testFromCsv(input, expected);
}

// Type mismatch — non-parsable field becomes null.
TEST_F(FromCsvTest, typeMismatch) {
  auto input = makeFlatVector<std::string>({"abc,1.1", "2,xyz"});
  auto expectedA = makeNullableFlatVector<int32_t>({std::nullopt, 2});
  auto expectedB = makeNullableFlatVector<double>({1.1, std::nullopt});
  auto expected = makeRowVector({"a", "b"}, {expectedA, expectedB});

  auto expr = createFromCsv(expected->type());
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();

  auto* childA = resultRow->childAt(0)->asFlatVector<int32_t>();
  auto* childB = resultRow->childAt(1)->asFlatVector<double>();
  EXPECT_TRUE(childA->isNullAt(0));
  EXPECT_FALSE(childA->isNullAt(1));
  EXPECT_EQ(childA->valueAt(1), 2);
  EXPECT_FALSE(childB->isNullAt(0));
  EXPECT_DOUBLE_EQ(childB->valueAt(0), 1.1);
  EXPECT_TRUE(childB->isNullAt(1));
}

// Boolean fields.
TEST_F(FromCsvTest, booleanFields) {
  auto input = makeFlatVector<std::string>({"true,1", "false,2", "invalid,3"});
  auto expectedA = makeNullableFlatVector<bool>({true, false, std::nullopt});
  auto expectedB = makeFlatVector<int32_t>({1, 2, 3});
  auto expected = makeRowVector({"a", "b"}, {expectedA, expectedB});

  auto expr = createFromCsv(expected->type());
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();

  auto* childA = resultRow->childAt(0)->asFlatVector<bool>();
  EXPECT_FALSE(childA->isNullAt(0));
  EXPECT_TRUE(childA->valueAt(0));
  EXPECT_FALSE(childA->isNullAt(1));
  EXPECT_FALSE(childA->valueAt(1));
  EXPECT_TRUE(childA->isNullAt(2));
}

// Boolean fields — case insensitive (Spark semantics).
TEST_F(FromCsvTest, booleanCaseInsensitive) {
  auto input = makeFlatVector<std::string>(
      {"TRUE,1", "True,2", "FALSE,3", "False,4", "tRuE,5"});
  auto expectedA =
      makeNullableFlatVector<bool>({true, true, false, false, true});
  auto expectedB = makeFlatVector<int32_t>({1, 2, 3, 4, 5});
  auto expected = makeRowVector({"a", "b"}, {expectedA, expectedB});

  auto expr = createFromCsv(expected->type());
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();

  auto* childA = resultRow->childAt(0)->asFlatVector<bool>();
  for (int i = 0; i < 5; ++i) {
    EXPECT_FALSE(childA->isNullAt(i)) << "row " << i;
  }
  EXPECT_TRUE(childA->valueAt(0));
  EXPECT_TRUE(childA->valueAt(1));
  EXPECT_FALSE(childA->valueAt(2));
  EXPECT_FALSE(childA->valueAt(3));
  EXPECT_TRUE(childA->valueAt(4));
}

TEST_F(FromCsvTest, booleanStrictness) {
  auto input = makeFlatVector<std::string>(
      {"0", "1", " true ", " false ", "t", "f", "yes", "no"});
  auto expr = createFromCsv(ROW({"a"}, {BOOLEAN()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto child = result->as<RowVector>()->childAt(0);
  for (vector_size_t row = 0; row < input->size(); ++row) {
    EXPECT_TRUE(child->isNullAt(row));
  }
}

// TinyInt and SmallInt fields.
TEST_F(FromCsvTest, tinyIntSmallInt) {
  auto input = makeFlatVector<std::string>({"127,32767", "-128,-32768", "0,0"});
  auto expectedA = makeFlatVector<int8_t>({127, -128, 0});
  auto expectedB = makeFlatVector<int16_t>({32767, -32768, 0});
  auto expected = makeRowVector({"a", "b"}, {expectedA, expectedB});
  testFromCsv(input, expected);
}

// BigInt fields.
TEST_F(FromCsvTest, bigIntFields) {
  auto input = makeFlatVector<std::string>(
      {"9223372036854775807", "-9223372036854775808", "0"});
  auto expectedA = makeFlatVector<int64_t>(
      {9223372036854775807LL, -9223372036854775807LL - 1, 0});
  auto expected = makeRowVector({"a"}, {expectedA});
  testFromCsv(input, expected);
}

// Float fields.
TEST_F(FromCsvTest, floatFields) {
  auto input =
      makeFlatVector<std::string>({"1.5,2.5", "3.14,0.001", "0.0,99.9"});
  auto expectedA = makeFlatVector<float>({1.5f, 3.14f, 0.0f});
  auto expectedB = makeFlatVector<float>({2.5f, 0.001f, 99.9f});
  auto expected = makeRowVector({"a", "b"}, {expectedA, expectedB});
  testFromCsv(input, expected);
}

// Floating-point parsing uses TextReader's case-insensitive special values.
TEST_F(FromCsvTest, nanAndInfinity) {
  auto input = makeFlatVector<std::string>(
      {"NaN,NaN",
       "+NaN,-NaN",
       "Infinity,-Infinity",
       "+Infinity,+Infinity",
       "Inf,-Inf",
       "inf,-inf",
       "nan,INFINITY",
       " Inf,-Inf ",
       " Infinity, -Infinity ",
       "value,other"});

  auto expr = createFromCsv(ROW({"a", "b"}, {DOUBLE(), DOUBLE()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();

  auto* childA = resultRow->childAt(0)->asFlatVector<double>();
  auto* childB = resultRow->childAt(1)->asFlatVector<double>();
  EXPECT_TRUE(std::isnan(childA->valueAt(0)));
  EXPECT_TRUE(std::isnan(childB->valueAt(0)));
  EXPECT_TRUE(std::isnan(childA->valueAt(1)));
  EXPECT_TRUE(childB->isNullAt(1));
  EXPECT_TRUE(std::isinf(childA->valueAt(2)));
  EXPECT_GT(childA->valueAt(2), 0);
  EXPECT_TRUE(std::isinf(childB->valueAt(2)));
  EXPECT_LT(childB->valueAt(2), 0);
  EXPECT_TRUE(std::isinf(childA->valueAt(3)));
  EXPECT_GT(childA->valueAt(3), 0);
  EXPECT_TRUE(std::isinf(childB->valueAt(3)));
  EXPECT_GT(childB->valueAt(3), 0);
  const auto expectSignedInfinities = [&](vector_size_t row) {
    EXPECT_TRUE(std::isinf(childA->valueAt(row)));
    EXPECT_GT(childA->valueAt(row), 0);
    EXPECT_TRUE(std::isinf(childB->valueAt(row)));
    EXPECT_LT(childB->valueAt(row), 0);
  };
  expectSignedInfinities(4);
  expectSignedInfinities(5);
  EXPECT_TRUE(std::isnan(childA->valueAt(6)));
  EXPECT_TRUE(std::isinf(childB->valueAt(6)));
  EXPECT_GT(childB->valueAt(6), 0);
  expectSignedInfinities(7);
  expectSignedInfinities(8);
  EXPECT_TRUE(childA->isNullAt(9));
  EXPECT_TRUE(childB->isNullAt(9));
}

TEST_F(FromCsvTest, canonicalIntegerParsing) {
  auto input = makeFlatVector<std::string>({"+123", "123.45", "-7.9", "123x"});
  auto expr = createFromCsv(ROW({"a"}, {INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* child = result->as<RowVector>()->childAt(0)->asFlatVector<int32_t>();
  EXPECT_TRUE(child->isNullAt(0));
  EXPECT_EQ(child->valueAt(1), 123);
  EXPECT_EQ(child->valueAt(2), -7);
  EXPECT_TRUE(child->isNullAt(3));
}

TEST_F(FromCsvTest, floatTypeSuffixRejected) {
  auto input = makeFlatVector<std::string>(
      {"1.0f", "2.5d", "3F", "4.25D", "1.0fd", "1.0x"});
  auto expr = createFromCsv(ROW({"a"}, {DOUBLE()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  for (vector_size_t row = 0; row < input->size(); ++row) {
    EXPECT_TRUE(resultRow->childAt(0)->isNullAt(row));
  }
}

// Mixed-type sanity check covering int + bool + string + double.
TEST_F(FromCsvTest, mixedTypes) {
  auto input = makeFlatVector<std::string>(
      {"42,true,hello,3.14", "0,false,world,2.71", "-7,true,test,0.5"});
  auto expected = makeRowVector(
      {"id", "flag", "name", "score"},
      {makeFlatVector<int32_t>({42, 0, -7}),
       makeFlatVector<bool>({true, false, true}),
       makeFlatVector<std::string>({"hello", "world", "test"}),
       makeFlatVector<double>({3.14, 2.71, 0.5})});
  testFromCsv(input, expected);
}

TEST_F(FromCsvTest, hexadecimalFloatingPoint) {
  auto input = makeFlatVector<std::string>({"0x1p2", "-0x1p3"});
  auto expected = makeRowVector({"a"}, {makeFlatVector<double>({4.0, -8.0})});
  testFromCsv(input, expected);
}

// VARCHAR fields preserve leading/trailing whitespace (no trimming).
// Integer fields with whitespace fail to parse → NULL (matching Spark's
// Integer.parseInt which rejects whitespace).
TEST_F(FromCsvTest, varcharPreservesWhitespace) {
  auto input = makeFlatVector<std::string>({"  hi  ,1", "hello , 2 "});
  auto expr = createFromCsv(ROW({"a", "b"}, {VARCHAR(), INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childA = resultRow->childAt(0)->asFlatVector<StringView>();
  auto* childB = resultRow->childAt(1)->as<SimpleVector<int32_t>>();
  // VARCHAR preserves original whitespace.
  EXPECT_EQ(childA->valueAt(0).str(), "  hi  ");
  EXPECT_EQ(childA->valueAt(1).str(), "hello ");
  // INTEGER does NOT trim — "1" parses OK, " 2 " fails → null.
  EXPECT_EQ(childB->valueAt(0), 1);
  EXPECT_TRUE(childB->isNullAt(1));
}

// Malformed leading plus rejected (Spark rejects "+-123", "++123").
TEST_F(FromCsvTest, malformedLeadingPlus) {
  auto input = makeFlatVector<std::string>({"+-123", "++456"});
  auto expr = createFromCsv(ROW({"a"}, {INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  EXPECT_TRUE(resultRow->childAt(0)->isNullAt(0));
  EXPECT_TRUE(resultRow->childAt(0)->isNullAt(1));
}

// Boundary values for INT32.
TEST_F(FromCsvTest, intBoundaryValues) {
  auto input =
      makeFlatVector<std::string>({"2147483647", "-2147483648", "0", "-1"});
  auto expectedA = makeFlatVector<int32_t>({2147483647, -2147483648, 0, -1});
  auto expected = makeRowVector({"a"}, {expectedA});
  testFromCsv(input, expected);
}

// Empty string field for non-varchar types becomes null.
TEST_F(FromCsvTest, emptyField) {
  auto input = makeFlatVector<std::string>({",1", "2,"});
  auto expectedA = makeNullableFlatVector<int32_t>({std::nullopt, 2});
  auto expectedB = makeNullableFlatVector<int32_t>({1, std::nullopt});
  auto expected = makeRowVector({"a", "b"}, {expectedA, expectedB});

  auto expr = createFromCsv(expected->type());
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();

  auto* childA = resultRow->childAt(0)->asFlatVector<int32_t>();
  auto* childB = resultRow->childAt(1)->asFlatVector<int32_t>();
  EXPECT_TRUE(childA->isNullAt(0));
  EXPECT_FALSE(childA->isNullAt(1));
  EXPECT_EQ(childA->valueAt(1), 2);
  EXPECT_FALSE(childB->isNullAt(0));
  EXPECT_EQ(childB->valueAt(0), 1);
  EXPECT_TRUE(childB->isNullAt(1));
}

// Spark's default emptyValue and nullValue are both "", so quoted and
// unquoted empty VARCHAR and VARBINARY fields all map to NULL.
TEST_F(FromCsvTest, emptyStringFieldsAreNull) {
  auto input = makeFlatVector<std::string>({
      R"("","")",
      ",",
      R"("",)",
  });

  auto expr = createFromCsv(ROW({"a", "b"}, {VARCHAR(), VARBINARY()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();

  for (vector_size_t row = 0; row < input->size(); ++row) {
    EXPECT_TRUE(resultRow->childAt(0)->isNullAt(row));
    EXPECT_TRUE(resultRow->childAt(1)->isNullAt(row));
  }
}

// Quoted fields.
TEST_F(FromCsvTest, quotedFields) {
  auto input =
      makeFlatVector<std::string>({R"("hello, world",1)", R"("a\"b",2)"});

  auto expectedA = makeFlatVector<std::string>({"hello, world", "a\"b"});
  auto expectedB = makeFlatVector<int32_t>({1, 2});
  auto expected = makeRowVector({"a", "b"}, {expectedA, expectedB});

  auto expr = createFromCsv(expected->type());
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();

  auto* childA = resultRow->childAt(0)->asFlatVector<StringView>();
  auto* childB = resultRow->childAt(1)->asFlatVector<int32_t>();
  EXPECT_EQ(childA->valueAt(0).str(), "hello, world");
  EXPECT_EQ(childB->valueAt(0), 1);
  // Spark's default backslash escape emits a literal double quote.
  EXPECT_EQ(childA->valueAt(1).str(), "a\"b");
  EXPECT_EQ(childB->valueAt(1), 2);
}

// REAL/DOUBLE trim bytes 0x00 through 0x20, matching Java String.trim.
// Integer fields do not trim.
TEST_F(FromCsvTest, whitespaceTrimming) {
  auto input = makeFlatVector<std::string>(
      {"  1  ,  2.5  ", "  42  ,  3.14  ", "  -7  ,  0.0  "});
  // Integer fields " 1 " etc. have whitespace → NULL (Spark doesn't trim).
  auto expectedA = makeNullableFlatVector<int32_t>(
      {std::nullopt, std::nullopt, std::nullopt});
  // Double fields "  2.5  " etc. are trimmed → parse OK (Java's parseDouble
  // trims).
  auto expectedB = makeFlatVector<double>({2.5, 3.14, 0.0});
  auto expected = makeRowVector({"a", "b"}, {expectedA, expectedB});
  testFromCsv(input, expected);
}

// Java's String.trim removes all leading and trailing code units through
// U+0020, but not non-ASCII Unicode whitespace such as U+00A0.
TEST_F(FromCsvTest, javaFloatWhitespaceTrimming) {
  std::string controlWrapped;
  controlWrapped.push_back('\x01');
  controlWrapped.append("2.5");
  controlWrapped.push_back('\x1f');

  const std::string nonBreakingSpace = "\xc2\xa0";
  auto input = makeFlatVector<std::string>(
      {controlWrapped, nonBreakingSpace + "3.5" + nonBreakingSpace});
  auto expr = createFromCsv(ROW({"a"}, {DOUBLE()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* childA = result->as<RowVector>()->childAt(0)->asFlatVector<double>();
  EXPECT_FALSE(childA->isNullAt(0));
  EXPECT_EQ(childA->valueAt(0), 2.5);
  EXPECT_TRUE(childA->isNullAt(1));
}

// Single field struct.
TEST_F(FromCsvTest, singleField) {
  auto input = makeFlatVector<std::string>({"42", "100", "-7"});
  auto expectedA = makeFlatVector<int32_t>({42, 100, -7});
  auto expected = makeRowVector({"a"}, {expectedA});
  testFromCsv(input, expected);
}

// Integer overflow becomes null.
TEST_F(FromCsvTest, integerOverflow) {
  auto input = makeFlatVector<std::string>({"128", "99999"});
  auto expectedA = makeNullableFlatVector<int8_t>({std::nullopt, std::nullopt});
  auto expected = makeRowVector({"a"}, {expectedA});

  auto expr = createFromCsv(expected->type());
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childA = resultRow->childAt(0)->asFlatVector<int8_t>();
  EXPECT_TRUE(childA->isNullAt(0));
  EXPECT_TRUE(childA->isNullAt(1));
}

// INT32 and BIGINT overflow becomes null.
TEST_F(FromCsvTest, intAndBigintOverflow) {
  // INT32 overflow.
  auto intInput = makeFlatVector<std::string>({"2147483648", "-2147483649"});
  auto intExpr = createFromCsv(ROW({"a"}, {INTEGER()}));
  auto intResult = evaluate(intExpr, makeRowVector({intInput}));
  auto* intRow = intResult->as<RowVector>();
  EXPECT_TRUE(intRow->childAt(0)->isNullAt(0));
  EXPECT_TRUE(intRow->childAt(0)->isNullAt(1));

  // BIGINT overflow.
  auto bigInput = makeFlatVector<std::string>(
      {"9223372036854775808", "-9223372036854775809"});
  auto bigExpr = createFromCsv(ROW({"a"}, {BIGINT()}));
  auto bigResult = evaluate(bigExpr, makeRowVector({bigInput}));
  auto* bigRow = bigResult->as<RowVector>();
  EXPECT_TRUE(bigRow->childAt(0)->isNullAt(0));
  EXPECT_TRUE(bigRow->childAt(0)->isNullAt(1));
}

// Unclosed quoted fields return escape-decoded content after the opening
// quote. A dangling escape is preserved to match TextReader.
TEST_F(FromCsvTest, unclosedQuote) {
  auto input = makeFlatVector<std::string>(
      {R"("unclosed)", R"("unclosed,field)", R"(a,")", "\"a\\", "\"\\"});
  auto expr = createFromCsv(ROW({"a", "b"}, {VARCHAR(), VARCHAR()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childA = resultRow->childAt(0)->asFlatVector<StringView>();
  EXPECT_FALSE(childA->isNullAt(0));
  EXPECT_EQ(childA->valueAt(0).str(), "unclosed");
  EXPECT_TRUE(resultRow->childAt(1)->isNullAt(0));
  EXPECT_FALSE(childA->isNullAt(1));
  EXPECT_EQ(childA->valueAt(1).str(), "unclosed,field");
  EXPECT_TRUE(resultRow->childAt(1)->isNullAt(1));
  EXPECT_EQ(childA->valueAt(2).str(), "a");
  EXPECT_EQ(
      resultRow->childAt(1)->asFlatVector<StringView>()->valueAt(2).str(),
      "\"");
  EXPECT_EQ(childA->valueAt(3).str(), "a\\");
  EXPECT_TRUE(resultRow->childAt(1)->isNullAt(3));
  EXPECT_EQ(childA->valueAt(4).str(), "\\");
  EXPECT_TRUE(resultRow->childAt(1)->isNullAt(4));
}

// Empty input line produces a non-null ROW with all-null fields. Spark's
// from_csv returns a non-null Row(null, null, ...) for empty input — the struct
// itself is not null, only its fields are.
TEST_F(FromCsvTest, emptyInputLine) {
  auto input = makeFlatVector<std::string>({""});
  auto expr = createFromCsv(ROW({"a", "b"}, {INTEGER(), VARCHAR()}));
  auto result = evaluate(expr, makeRowVector({input}));
  // The struct itself is NOT null — Spark returns a live row.
  EXPECT_FALSE(result->isNullAt(0));
  auto* resultRow = result->as<RowVector>();
  // But all child fields are null.
  EXPECT_TRUE(resultRow->childAt(0)->isNullAt(0));
  EXPECT_TRUE(resultRow->childAt(1)->isNullAt(0));
}

// Unquoted empty VARCHAR fields become NULL; non-empty fields are preserved.
TEST_F(FromCsvTest, emptyAndNonEmptyVarcharFields) {
  auto input = makeFlatVector<std::string>({",1", "hello,3"});
  auto expr = createFromCsv(ROW({"s", "i"}, {VARCHAR(), INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childS = resultRow->childAt(0)->asFlatVector<StringView>();
  auto* childI = resultRow->childAt(1)->asFlatVector<int32_t>();
  // Row 0: empty unquoted field → null.
  EXPECT_TRUE(childS->isNullAt(0));
  EXPECT_EQ(childI->valueAt(0), 1);
  // Row 1: non-empty field → preserved.
  EXPECT_FALSE(childS->isNullAt(1));
  EXPECT_EQ(childS->valueAt(1).str(), "hello");
  EXPECT_EQ(childI->valueAt(1), 3);
}

// Whitespace-only VARCHAR fields are NOT null (only truly empty is null).
TEST_F(FromCsvTest, whitespaceVarcharNotNull) {
  auto input = makeFlatVector<std::string>({"  ,1", "   ,2", "a,3"});
  auto expr = createFromCsv(ROW({"s", "i"}, {VARCHAR(), INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childS = resultRow->childAt(0)->asFlatVector<StringView>();
  // Whitespace-only is not empty — preserved as-is.
  EXPECT_FALSE(childS->isNullAt(0));
  EXPECT_EQ(childS->valueAt(0).str(), "  ");
  EXPECT_FALSE(childS->isNullAt(1));
  EXPECT_EQ(childS->valueAt(1).str(), "   ");
  EXPECT_FALSE(childS->isNullAt(2));
  EXPECT_EQ(childS->valueAt(2).str(), "a");
}

// Float overflow returns ±Infinity (matching Java's parseDouble("1e400")).
TEST_F(FromCsvTest, floatOverflow) {
  // Row 3 uses a large-magnitude value whose exponent notation is negative but
  // whose effective magnitude still overflows (10^1000 / 10 = 10^999). It
  // guards against classifying overflow/underflow by exponent sign alone.
  const std::string bigSignificandNegExp = "1" + std::string(1000, '0') + "e-1";
  auto input = makeFlatVector<std::string>(
      {"1e400,-1e400",
       "3.4028236e38,1e-400",
       "+1e309,-1e309",
       bigSignificandNegExp + ",-" + bigSignificandNegExp});
  auto expr = createFromCsv(ROW({"a", "b"}, {DOUBLE(), DOUBLE()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childA = resultRow->childAt(0)->asFlatVector<double>();
  auto* childB = resultRow->childAt(1)->asFlatVector<double>();
  // Row 0: 1e400 → +inf, -1e400 → -inf.
  EXPECT_TRUE(std::isinf(childA->valueAt(0)));
  EXPECT_GT(childA->valueAt(0), 0);
  EXPECT_TRUE(std::isinf(childB->valueAt(0)));
  EXPECT_LT(childB->valueAt(0), 0);
  // Row 1: 3.4028236e38 is beyond FLOAT max but valid DOUBLE.
  EXPECT_FALSE(childA->isNullAt(1));
  EXPECT_DOUBLE_EQ(childA->valueAt(1), 3.4028236e38);
  // 1e-400 underflow → 0.
  EXPECT_FALSE(childB->isNullAt(1));
  EXPECT_DOUBLE_EQ(childB->valueAt(1), 0.0);
  // Row 2: +1e309 → +inf, -1e309 → -inf.
  EXPECT_TRUE(std::isinf(childA->valueAt(2)));
  EXPECT_GT(childA->valueAt(2), 0);
  EXPECT_TRUE(std::isinf(childB->valueAt(2)));
  EXPECT_LT(childB->valueAt(2), 0);
  // Row 3: large significand with "e-1" still overflows → ±inf.
  EXPECT_TRUE(std::isinf(childA->valueAt(3)));
  EXPECT_GT(childA->valueAt(3), 0);
  EXPECT_TRUE(std::isinf(childB->valueAt(3)));
  EXPECT_LT(childB->valueAt(3), 0);
}

// Negative overflow for TINYINT/SMALLINT becomes null.
TEST_F(FromCsvTest, negativeIntegerOverflow) {
  auto input = makeFlatVector<std::string>({"-129,-32769", "-130,-32770"});
  auto expr = createFromCsv(ROW({"a", "b"}, {TINYINT(), SMALLINT()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  EXPECT_TRUE(resultRow->childAt(0)->isNullAt(0));
  EXPECT_TRUE(resultRow->childAt(1)->isNullAt(0));
  EXPECT_TRUE(resultRow->childAt(0)->isNullAt(1));
  EXPECT_TRUE(resultRow->childAt(1)->isNullAt(1));
}

// Partial parse failures (valid prefix + junk) become null.
TEST_F(FromCsvTest, partialParseFail) {
  auto input = makeFlatVector<std::string>({"1x,3.14foo,truee"});
  auto expr =
      createFromCsv(ROW({"a", "b", "c"}, {INTEGER(), DOUBLE(), BOOLEAN()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  // All three fields should be null — parser must consume entire token.
  EXPECT_TRUE(resultRow->childAt(0)->isNullAt(0));
  EXPECT_TRUE(resultRow->childAt(1)->isNullAt(0));
  EXPECT_TRUE(resultRow->childAt(2)->isNullAt(0));
}

// Non-whitespace characters after a closing quote trigger literal fallback.
TEST_F(FromCsvTest, garbageAfterQuote) {
  // Spark/Univocity default unescapedQuoteHandling=STOP_AT_DELIMITER: non-
  // whitespace after a closing quote produces the opening quote, decoded
  // quoted content, closing quote, and raw remainder through the delimiter.
  auto input = makeFlatVector<std::string>(
      {R"("hello"world,1)", R"("a,b"x,2)", R"("a\"b"c,d)"});
  auto expr = createFromCsv(ROW({"a", "b"}, {VARCHAR(), INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childA = resultRow->childAt(0)->asFlatVector<StringView>();
  auto* childB = resultRow->childAt(1)->asFlatVector<int32_t>();
  EXPECT_EQ(childA->valueAt(0).str(), "\"hello\"world");
  EXPECT_EQ(childB->valueAt(0), 1);
  EXPECT_EQ(childA->valueAt(1).str(), "\"a,b\"x");
  EXPECT_EQ(childB->valueAt(1), 2);
  EXPECT_EQ(childA->valueAt(2).str(), "\"a\"b\"c");
  EXPECT_TRUE(childB->isNullAt(2));
}

TEST_F(FromCsvTest, whitespaceAfterClosingQuote) {
  // ASCII whitespace between a closing quote and the next delimiter is
  // skipped; the parsed quoted content is used as the field value.
  auto input = makeFlatVector<std::string>({R"("x" ,y)", "\"a\"\t,b"});
  auto expr = createFromCsv(ROW({"a", "b"}, {VARCHAR(), VARCHAR()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childA = resultRow->childAt(0)->asFlatVector<StringView>();
  auto* childB = resultRow->childAt(1)->asFlatVector<StringView>();
  EXPECT_EQ(childA->valueAt(0).str(), "x");
  EXPECT_EQ(childB->valueAt(0).str(), "y");
  EXPECT_EQ(childA->valueAt(1).str(), "a");
  EXPECT_EQ(childB->valueAt(1).str(), "b");
}

// Row nullness: non-null input always produces non-null row, even if all
// fields fail to parse. Only SQL NULL input produces a null row.
TEST_F(FromCsvTest, rowNullness) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc,xyz", ",", "1,2", "", std::nullopt});
  auto expr = createFromCsv(ROW({"a", "b"}, {INTEGER(), INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  // Non-null input (including empty string): row is never null.
  for (int i = 0; i < 4; ++i) {
    EXPECT_FALSE(resultRow->isNullAt(i)) << "row " << i;
  }
  // SQL NULL input: row is null.
  EXPECT_TRUE(result->isNullAt(4));
}

// Unsupported field type throws.
TEST_F(FromCsvTest, unsupportedFieldType) {
  auto input = makeFlatVector<std::string>({"1,2,3"});
  auto expr = createFromCsv(ROW({"a"}, {ARRAY(INTEGER())}));
  VELOX_ASSERT_THROW(
      evaluate(expr, makeRowVector({input})), "Unsupported field type");
}

// Nested types (ARRAY/MAP/ROW as struct fields) are rejected at plan time
// with a clear message, matching Spark's unsupported-data-type behavior.
TEST_F(FromCsvTest, nestedTypesRejected) {
  auto input = makeFlatVector<std::string>({"1"});
  const std::string expected = "Nested types (ARRAY/MAP/ROW) are not supported";
  VELOX_ASSERT_THROW(
      evaluate(
          createFromCsv(ROW({"a"}, {ARRAY(INTEGER())})),
          makeRowVector({input})),
      expected);
  VELOX_ASSERT_THROW(
      evaluate(
          createFromCsv(ROW({"a"}, {MAP(VARCHAR(), INTEGER())})),
          makeRowVector({input})),
      expected);
  VELOX_ASSERT_THROW(
      evaluate(
          createFromCsv(ROW({"a"}, {ROW({"b"}, {INTEGER()})})),
          makeRowVector({input})),
      expected);
}

// A schema field named `_corrupt_record` is treated as an ordinary positional
// column, unlike Spark's special non-positional corrupt-record handling.
TEST_F(FromCsvTest, corruptRecordColumnIsPositional) {
  auto input = makeFlatVector<std::string>({"1", "bad", "bad,x"});
  auto expr =
      createFromCsv(ROW({"a", "_corrupt_record"}, {INTEGER(), VARCHAR()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* row = result->as<RowVector>();
  auto* aCol = row->childAt(0)->asFlatVector<int32_t>();
  auto* corruptCol = row->childAt(1)->asFlatVector<StringView>();

  // Row 0 is well formed, so both Spark and Velox leave the corrupt field null.
  EXPECT_FALSE(aCol->isNullAt(0));
  EXPECT_EQ(aCol->valueAt(0), 1);
  EXPECT_TRUE(corruptCol->isNullAt(0));

  // Row 1 is malformed. Spark would populate the raw record; Velox has no
  // second positional field and leaves it null.
  EXPECT_TRUE(aCol->isNullAt(1));
  EXPECT_TRUE(corruptCol->isNullAt(1));

  // Row 2 demonstrates the positional divergence: Velox stores the second
  // field, while Spark would store the entire malformed record "bad,x".
  EXPECT_TRUE(aCol->isNullAt(2));
  EXPECT_EQ(corruptCol->valueAt(2).str(), "x");
}

// Whitespace-only input is tokenized normally (no special-casing).
// With default ignoreLeadingWhiteSpace=false and
// ignoreTrailingWhiteSpace=false, the whitespace is preserved as the field
// value.
TEST_F(FromCsvTest, whitespaceOnlyInput) {
  auto input = makeFlatVector<std::string>({"   ", " \t\n ", "\t"});
  auto expr = createFromCsv(ROW({"a", "b"}, {INTEGER(), VARCHAR()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  // Struct rows are NOT null.
  EXPECT_FALSE(result->isNullAt(0));
  EXPECT_FALSE(result->isNullAt(1));
  EXPECT_FALSE(result->isNullAt(2));
  // INTEGER column: whitespace fails integer parsing → null.
  EXPECT_TRUE(resultRow->childAt(0)->isNullAt(0));
  EXPECT_TRUE(resultRow->childAt(0)->isNullAt(1));
  EXPECT_TRUE(resultRow->childAt(0)->isNullAt(2));
  // VARCHAR column: only one field produced (no delimiter in input), so
  // second column is missing → null.
  EXPECT_TRUE(resultRow->childAt(1)->isNullAt(0));
  EXPECT_TRUE(resultRow->childAt(1)->isNullAt(1));
  EXPECT_TRUE(resultRow->childAt(1)->isNullAt(2));
}

// Whitespace-only input with a single VARCHAR column preserves whitespace.
TEST_F(FromCsvTest, whitespaceOnlyVarchar) {
  auto input = makeFlatVector<std::string>({"   ", " \t "});
  auto expr = createFromCsv(ROW({"a"}, {VARCHAR()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  EXPECT_FALSE(result->isNullAt(0));
  EXPECT_FALSE(result->isNullAt(1));
  // VARCHAR field preserves the whitespace (not trimmed, doesn't match
  // nullValue "").
  auto* child = resultRow->childAt(0)->asFlatVector<StringView>();
  EXPECT_FALSE(child->isNullAt(0));
  EXPECT_EQ(child->valueAt(0).str(), "   ");
  EXPECT_FALSE(child->isNullAt(1));
  EXPECT_EQ(child->valueAt(1).str(), " \t ");
}

// Short decimal (precision <= 18) parsing.
TEST_F(FromCsvTest, shortDecimal) {
  auto input =
      makeFlatVector<std::string>({"123.45", "  -99.99  ", "abc", "0.01", ""});
  auto decType = DECIMAL(10, 2);
  auto expr = createFromCsv(ROW({"a"}, {decType}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* child = resultRow->childAt(0)->asFlatVector<int64_t>();
  // "123.45" → 12345 (scaled by 10^2)
  EXPECT_FALSE(child->isNullAt(0));
  EXPECT_EQ(child->valueAt(0), 12345);
  // "  -99.99  " → null (whitespace not trimmed for decimal by default,
  // matching Java's BigDecimal which rejects leading/trailing whitespace).
  EXPECT_TRUE(child->isNullAt(1));
  // "abc" → null (parse failure)
  EXPECT_TRUE(child->isNullAt(2));
  // "0.01" → 1
  EXPECT_FALSE(child->isNullAt(3));
  EXPECT_EQ(child->valueAt(3), 1);
  // "" → null (empty field = nullValue sentinel)
  EXPECT_TRUE(child->isNullAt(4));
}

// Long decimal (precision > 18) parsing.
TEST_F(FromCsvTest, longDecimal) {
  auto input = makeFlatVector<std::string>(
      {"12345678901234567890.12", "  0.00  ", "invalid"});
  auto decType = DECIMAL(30, 2);
  auto expr = createFromCsv(ROW({"a"}, {decType}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* child = resultRow->childAt(0)->asFlatVector<int128_t>();
  // "12345678901234567890.12" → 1234567890123456789012
  EXPECT_FALSE(child->isNullAt(0));
  int128_t expected = int128_t(12345678901234567890ULL) * 100 + 12;
  EXPECT_EQ(child->valueAt(0), expected);
  // "  0.00  " → null (whitespace not trimmed for decimal by default).
  EXPECT_TRUE(child->isNullAt(1));
  // "invalid" → null
  EXPECT_TRUE(child->isNullAt(2));
}

// Values exceeding the declared DECIMAL precision yield NULL (matching
// Spark's Cast semantics: overflow past precision fails the field, but the
// row itself remains present under PERMISSIVE mode).
TEST_F(FromCsvTest, decimalPrecisionOverflow) {
  auto input = makeFlatVector<std::string>({
      "12.34", // fits DECIMAL(4,2)
      "999.99", // 5 digits total → overflows DECIMAL(4,2)
      "-999.99", // negative overflow, same precision
  });
  auto expr = createFromCsv(ROW({"a"}, {DECIMAL(4, 2)}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* child = resultRow->childAt(0)->asFlatVector<int64_t>();
  EXPECT_FALSE(child->isNullAt(0));
  EXPECT_EQ(child->valueAt(0), 1234);
  EXPECT_TRUE(child->isNullAt(1));
  EXPECT_TRUE(child->isNullAt(2));
}

TEST_F(FromCsvTest, canonicalDecimalParsing) {
  auto input = makeFlatVector<std::string>({
      R"("1,234.5",ok)",
      "3E2,ok",
      "3E-2,ok",
      "1e99999999999,ok",
      "malformed,ok",
  });
  auto expr = createFromCsv(ROW({"a", "b"}, {DECIMAL(10, 2), VARCHAR()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* decimal = resultRow->childAt(0)->asFlatVector<int64_t>();
  auto* text = resultRow->childAt(1)->asFlatVector<StringView>();
  EXPECT_TRUE(decimal->isNullAt(0));
  EXPECT_FALSE(decimal->isNullAt(1));
  EXPECT_EQ(decimal->valueAt(1), 30000);
  EXPECT_FALSE(decimal->isNullAt(2));
  EXPECT_EQ(decimal->valueAt(2), 3);
  EXPECT_TRUE(decimal->isNullAt(3));
  EXPECT_TRUE(decimal->isNullAt(4));
  for (vector_size_t row = 0; row < input->size(); ++row) {
    EXPECT_EQ(text->valueAt(row).str(), "ok");
  }
}

TEST_F(FromCsvTest, scientificDecimalRoundsOnce) {
  auto input =
      makeFlatVector<std::string>({"4.5e-2", "0.045", "-4.5e-2", "-0.045"});
  auto expr = createFromCsv(ROW({"a"}, {DECIMAL(10, 1)}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* decimal = result->as<RowVector>()->childAt(0)->asFlatVector<int64_t>();

  for (vector_size_t row = 0; row < input->size(); ++row) {
    EXPECT_FALSE(decimal->isNullAt(row));
    EXPECT_EQ(decimal->valueAt(row), 0);
  }
}

TEST_F(FromCsvTest, overflowingDateAndTimestampAreNull) {
  auto input = makeFlatVector<std::string>({
      "2500000000-01-01,2500000000-01-01 00:00:00,ok",
      "2024-01-01,292278994-04-23 11:46:00,out-of-range",
      "2024-01-01,294248-01-01 00:00:00,wider-than-spark",
  });
  auto expr =
      createFromCsv(ROW({"d", "t", "s"}, {DATE(), TIMESTAMP(), VARCHAR()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  EXPECT_TRUE(resultRow->childAt(0)->isNullAt(0));
  EXPECT_TRUE(resultRow->childAt(1)->isNullAt(0));
  EXPECT_EQ(
      resultRow->childAt(2)->asFlatVector<StringView>()->valueAt(0).str(),
      "ok");
  EXPECT_FALSE(resultRow->childAt(0)->isNullAt(1));
  EXPECT_TRUE(resultRow->childAt(1)->isNullAt(1));
  EXPECT_EQ(
      resultRow->childAt(2)->asFlatVector<StringView>()->valueAt(1).str(),
      "out-of-range");
  EXPECT_FALSE(resultRow->childAt(0)->isNullAt(2));
  EXPECT_FALSE(resultRow->childAt(1)->isNullAt(2));
  EXPECT_EQ(
      resultRow->childAt(2)->asFlatVector<StringView>()->valueAt(2).str(),
      "wider-than-spark");
}

// DATE field parsing (yyyy-MM-dd).
TEST_F(FromCsvTest, dateField) {
  auto input =
      makeFlatVector<std::string>({"2023-01-15", "1970-01-01", "bad-date", ""});
  auto expr = createFromCsv(ROW({"a"}, {DATE()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* child = resultRow->childAt(0)->asFlatVector<int32_t>();
  // 2023-01-15 = days since epoch
  EXPECT_FALSE(child->isNullAt(0));
  EXPECT_EQ(child->valueAt(0), 19372); // 2023-01-15
  // 1970-01-01 = 0
  EXPECT_FALSE(child->isNullAt(1));
  EXPECT_EQ(child->valueAt(1), 0);
  // "bad-date" → null
  EXPECT_TRUE(child->isNullAt(2));
  // "" → null (nullValue sentinel)
  EXPECT_TRUE(child->isNullAt(3));
}

TEST_F(FromCsvTest, prestoDateParsing) {
  auto input = makeFlatVector<std::string>({
      "2024-01-01", // canonical default format.
      "12024-01-01", // wider year accepted.
      "2024-1-1", // single-digit month/day accepted.
      "20240101", // no separators: rejected.
      " 2024-01-01", // surrounding whitespace accepted.
      "2024-01-01 ",
      "2024-01", // partial dates are rejected.
      "2024", // year-only values are rejected.
      "not-a-date", // malformed: rejected.
  });
  auto expr = createFromCsv(ROW({"a"}, {DATE()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* childA = result->as<RowVector>()->childAt(0)->asFlatVector<int32_t>();
  EXPECT_FALSE(childA->isNullAt(0));
  EXPECT_EQ(childA->valueAt(0), 19'723);
  EXPECT_FALSE(childA->isNullAt(1));
  EXPECT_EQ(childA->valueAt(1), 3'672'148);
  EXPECT_FALSE(childA->isNullAt(2));
  EXPECT_EQ(childA->valueAt(2), childA->valueAt(0));
  EXPECT_TRUE(childA->isNullAt(3));
  EXPECT_FALSE(childA->isNullAt(4));
  EXPECT_EQ(childA->valueAt(4), childA->valueAt(0));
  EXPECT_FALSE(childA->isNullAt(5));
  EXPECT_EQ(childA->valueAt(5), childA->valueAt(0));
  EXPECT_TRUE(childA->isNullAt(6));
  EXPECT_TRUE(childA->isNullAt(7));
  EXPECT_TRUE(childA->isNullAt(8));
}

// TIMESTAMP field parsing.
TEST_F(FromCsvTest, timestampField) {
  auto input = makeFlatVector<std::string>(
      {"2023-01-15 10:30:00", "1970-01-01 00:00:00", "bad-ts", ""});
  auto expr = createFromCsv(ROW({"a"}, {TIMESTAMP()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* child = resultRow->childAt(0)->asFlatVector<Timestamp>();
  // 10:30 in America/Los_Angeles (PST) is 18:30 UTC.
  EXPECT_FALSE(child->isNullAt(0));
  EXPECT_EQ(child->valueAt(0), Timestamp(1'673'807'400, 0));
  // TextReader normalizes midnight in Los Angeles to 08:00 UTC.
  EXPECT_FALSE(child->isNullAt(1));
  EXPECT_EQ(child->valueAt(1), Timestamp(28'800, 0));
  // "bad-ts" → null
  EXPECT_TRUE(child->isNullAt(2));
  // "" → null
  EXPECT_TRUE(child->isNullAt(3));
}

TEST_F(FromCsvTest, prestoTimestampParsing) {
  auto input = makeFlatVector<std::string>({
      "2024-01-01 00:00:00", // space separator accepted.
      "2024-01-01", // date only defaults to midnight.
      "2024-01-01 00:00:00.123456789", // truncated to microseconds.
      "2024-01-01T00:00:00", // ISO T separator is rejected.
      "2024-01-01 00:00:00Z", // timezone suffixes are rejected.
      "garbage", // malformed: rejected.
  });
  auto expr = createFromCsv(ROW({"a"}, {TIMESTAMP()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* childA = result->as<RowVector>()->childAt(0)->asFlatVector<Timestamp>();
  const auto expected = Timestamp(1'704'096'000, 0);
  EXPECT_FALSE(childA->isNullAt(0));
  EXPECT_EQ(childA->valueAt(0), expected);
  EXPECT_FALSE(childA->isNullAt(1));
  EXPECT_EQ(childA->valueAt(1), expected);
  EXPECT_FALSE(childA->isNullAt(2));
  EXPECT_EQ(childA->valueAt(2), Timestamp(1'704'096'000, 123'456'000));
  EXPECT_TRUE(childA->isNullAt(3));
  EXPECT_TRUE(childA->isNullAt(4));
  EXPECT_TRUE(childA->isNullAt(5));
}

TEST_F(FromCsvTest, nonexistentDefaultTimezoneTimestampIsNull) {
  auto input = makeFlatVector<std::string>(
      {"2024-03-10 02:30:00", "2024-03-10 03:30:00"});
  auto expr = createFromCsv(ROW({"a"}, {TIMESTAMP()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* child = result->as<RowVector>()->childAt(0)->asFlatVector<Timestamp>();

  EXPECT_TRUE(child->isNullAt(0));
  EXPECT_FALSE(child->isNullAt(1));
  EXPECT_EQ(child->valueAt(1), Timestamp(1'710'066'600, 0));
}

// Backslash escape in quoted fields (Spark default escape='\\').
TEST_F(FromCsvTest, backslashEscapeInQuotedField) {
  auto input = makeFlatVector<std::string>({
      R"("hello \"world\"",1)",
      R"("no escape",2)",
      R"("C:\\temp",x)",
      R"("a\\",b)",
      R"("a\"b,c)",
      R"("a\b",c)",
      R"("\a",b)",
      R"("four\,five",x)",
      R"("line1\nline2",y)",
  });
  auto expr = createFromCsv(ROW({"s", "v"}, {VARCHAR(), VARCHAR()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childS = resultRow->childAt(0)->asFlatVector<StringView>();
  auto* childV = resultRow->childAt(1)->asFlatVector<StringView>();
  // Row 0: backslash-escaped quotes → literal quotes in output.
  EXPECT_EQ(childS->valueAt(0).str(), "hello \"world\"");
  EXPECT_EQ(childV->valueAt(0).str(), "1");
  // Row 1: no escape needed.
  EXPECT_EQ(childS->valueAt(1).str(), "no escape");
  EXPECT_EQ(childV->valueAt(1).str(), "2");
  // Escaped backslashes collapse to one backslash.
  EXPECT_EQ(childS->valueAt(2).str(), "C:\\temp");
  EXPECT_EQ(childV->valueAt(2).str(), "x");
  // An escaped trailing backslash does not escape the closing quote.
  EXPECT_EQ(childS->valueAt(3).str(), "a\\");
  EXPECT_EQ(childV->valueAt(3).str(), "b");
  // Unclosed quoted fields retain already-unescaped content through EOF.
  EXPECT_EQ(childS->valueAt(4).str(), "a\"b,c");
  EXPECT_TRUE(childV->isNullAt(4));
  EXPECT_EQ(childS->valueAt(5).str(), R"(a\b)");
  EXPECT_EQ(childV->valueAt(5).str(), "c");
  EXPECT_EQ(childS->valueAt(6).str(), R"(\a)");
  EXPECT_EQ(childV->valueAt(6).str(), "b");
  EXPECT_EQ(childS->valueAt(7).str(), R"(four\,five)");
  EXPECT_EQ(childV->valueAt(7).str(), "x");
  EXPECT_EQ(childS->valueAt(8).str(), R"(line1\nline2)");
  EXPECT_EQ(childV->valueAt(8).str(), "y");
}

TEST_F(FromCsvTest, backslashEscapeInUnquotedField) {
  auto input =
      makeFlatVector<std::string>({R"(hello\,world,value)", R"(a\q,b)"});
  auto expr = createFromCsv(ROW({"s", "v"}, {VARCHAR(), VARCHAR()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childS = resultRow->childAt(0)->asFlatVector<StringView>();
  auto* childV = resultRow->childAt(1)->asFlatVector<StringView>();

  EXPECT_EQ(childS->valueAt(0).str(), "hello,world");
  EXPECT_EQ(childV->valueAt(0).str(), "value");
  EXPECT_EQ(childS->valueAt(1).str(), "aq");
  EXPECT_EQ(childV->valueAt(1).str(), "b");
}

// VARBINARY uses TextReader's base64 decoding and raw fallback semantics.
TEST_F(FromCsvTest, varbinaryField) {
  auto input = makeFlatVector<std::string>(
      {"aGVsbG8=,d29ybGQ=", "not-base64,also-invalid"});
  auto expr = createFromCsv(ROW({"a", "b"}, {VARBINARY(), VARBINARY()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childA = resultRow->childAt(0)->asFlatVector<StringView>();
  auto* childB = resultRow->childAt(1)->asFlatVector<StringView>();
  // Valid base64 is decoded.
  EXPECT_EQ(childA->valueAt(0).getString(), "hello");
  EXPECT_EQ(childB->valueAt(0).getString(), "world");
  // Invalid base64 is copied as-is for compatibility.
  EXPECT_EQ(childA->valueAt(1).getString(), "not-base64");
  EXPECT_EQ(childB->valueAt(1).getString(), "also-invalid");
}

TEST_F(FromCsvTest, nonInlineVarbinaryValuesAreOwnedByResult) {
  auto input = makeFlatVector<std::string>({
      "VGhpcyBpcyBhIGxvbmdlciB2YWx1ZQ==,"
      "QW5vdGhlciBsb25nZXIgdmFsdWU=",
      "not-valid-base64-value,also-not-valid-base64",
  });
  auto expr = createFromCsv(ROW({"a", "b"}, {VARBINARY(), VARBINARY()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childA = resultRow->childAt(0)->asFlatVector<StringView>();
  auto* childB = resultRow->childAt(1)->asFlatVector<StringView>();

  EXPECT_EQ(childA->valueAt(0).getString(), "This is a longer value");
  EXPECT_EQ(childB->valueAt(0).getString(), "Another longer value");
  EXPECT_EQ(childA->valueAt(1).getString(), "not-valid-base64-value");
  EXPECT_EQ(childB->valueAt(1).getString(), "also-not-valid-base64");
}

// Float underflow returns a correctly-signed zero (matching Java's
// parseDouble). Values whose magnitude is below the smallest subnormal double
// underflow regardless of how the magnitude is spelled (bare fraction,
// leading-zero fraction, or exponent notation).
TEST_F(FromCsvTest, floatUnderflow) {
  // Magnitude far below DBL_MIN, written without an exponent as a bare
  // fraction and with varying leading-zero spellings.
  std::string tinyDot = "." + std::string(400, '0') + "1";
  std::string tinyZeroDot = "0." + std::string(400, '0') + "1";
  std::string tinyMultiZero = "0000." + std::string(400, '0') + "1";
  // Same magnitude, negative sign → negative zero.
  std::string tinyNegMultiZero = "-0000." + std::string(400, '0') + "1";
  // Negative exponent underflow.
  std::string negExpUnderflow = "-1e-400";
  // Tiny fraction with a positive exponent whose effective magnitude still
  // underflows (10^-1001 * 10 = 10^-1000). Guards against classifying
  // overflow/underflow by exponent sign alone.
  std::string tinyFractionPosExp = "0." + std::string(1000, '0') + "1e+1";
  auto input = makeFlatVector<std::string>(
      {tinyDot,
       tinyZeroDot,
       tinyMultiZero,
       tinyNegMultiZero,
       negExpUnderflow,
       tinyFractionPosExp});
  auto expr = createFromCsv(ROW({"a"}, {DOUBLE()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* child = resultRow->childAt(0)->asFlatVector<double>();
  // Positive underflows → +0.0.
  EXPECT_FALSE(child->isNullAt(0));
  EXPECT_EQ(child->valueAt(0), 0.0);
  EXPECT_FALSE(std::signbit(child->valueAt(0)));
  EXPECT_FALSE(child->isNullAt(1));
  EXPECT_EQ(child->valueAt(1), 0.0);
  EXPECT_FALSE(std::signbit(child->valueAt(1)));
  EXPECT_FALSE(child->isNullAt(2));
  EXPECT_EQ(child->valueAt(2), 0.0);
  EXPECT_FALSE(std::signbit(child->valueAt(2)));
  // Negative underflows → -0.0 (matching Java's parseDouble).
  EXPECT_FALSE(child->isNullAt(3));
  EXPECT_EQ(child->valueAt(3), 0.0);
  EXPECT_TRUE(std::signbit(child->valueAt(3)));
  EXPECT_FALSE(child->isNullAt(4));
  EXPECT_EQ(child->valueAt(4), 0.0);
  EXPECT_TRUE(std::signbit(child->valueAt(4)));
  // Tiny fraction with "e+1" still underflows → +0.0.
  EXPECT_FALSE(child->isNullAt(5));
  EXPECT_EQ(child->valueAt(5), 0.0);
  EXPECT_FALSE(std::signbit(child->valueAt(5)));
}

// The canonical floating-point parser rejects malformed sign-only input.
TEST_F(FromCsvTest, barePlusSign) {
  auto input = makeFlatVector<std::string>({"+", "+-1", "++"});
  auto expr = createFromCsv(ROW({"a"}, {DOUBLE()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* child = resultRow->childAt(0)->asFlatVector<double>();
  // All should be null (invalid float).
  EXPECT_TRUE(child->isNullAt(0));
  EXPECT_TRUE(child->isNullAt(1));
  EXPECT_TRUE(child->isNullAt(2));
}

TEST_F(FromCsvTest, delimiterOnlyInput) {
  // A single delimiter with a 2-field schema produces two empty (null) fields.
  auto input = makeFlatVector<StringView>({","_sv, ",,,"_sv});
  auto expr = createFromCsv(ROW({"a", "b"}, {INTEGER(), INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* a = resultRow->childAt(0)->asFlatVector<int32_t>();
  auto* b = resultRow->childAt(1)->asFlatVector<int32_t>();
  // "," → two empty fields → both null.
  EXPECT_TRUE(a->isNullAt(0));
  EXPECT_TRUE(b->isNullAt(0));
  // ",,," → 4 fields but schema has 2, extras ignored; first two are empty →
  // null.
  EXPECT_TRUE(a->isNullAt(1));
  EXPECT_TRUE(b->isNullAt(1));
}

TEST_F(FromCsvTest, integerWithDecimalPoint) {
  auto input = makeFlatVector<StringView>({"123.0"_sv, "45.6"_sv, "7.0"_sv});
  auto expr = createFromCsv(ROW({"a"}, {INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* child = resultRow->childAt(0)->asFlatVector<int32_t>();
  EXPECT_EQ(child->valueAt(0), 123);
  EXPECT_EQ(child->valueAt(1), 45);
  EXPECT_EQ(child->valueAt(2), 7);
}

TEST_F(FromCsvTest, hexIntegerRejected) {
  // Hex prefixed integers should be rejected (not valid for Spark integer
  // parsing).
  auto input = makeFlatVector<StringView>({"0x1A"_sv, "0X10"_sv, "0xff"_sv});
  auto expr = createFromCsv(ROW({"a"}, {INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* child = resultRow->childAt(0)->asFlatVector<int32_t>();
  EXPECT_TRUE(child->isNullAt(0));
  EXPECT_TRUE(child->isNullAt(1));
  EXPECT_TRUE(child->isNullAt(2));
}

// Zero-field ROW schema: should return non-null empty row without parsing.
TEST_F(FromCsvTest, zeroFieldRow) {
  auto input = makeFlatVector<std::string>({"hello,world", "", "a,b,c"});
  auto expr = createFromCsv(ROW({}, {}));
  auto result = evaluate(expr, makeRowVector({input}));
  // All rows should be non-null (input is non-null).
  EXPECT_FALSE(result->isNullAt(0));
  EXPECT_FALSE(result->isNullAt(1));
  EXPECT_FALSE(result->isNullAt(2));
  // Result row has zero children.
  auto* resultRow = result->as<RowVector>();
  EXPECT_EQ(resultRow->childrenSize(), 0);
}

// Zero-field ROW with null input yields null row.
TEST_F(FromCsvTest, zeroFieldRowNullInput) {
  auto input = makeNullableFlatVector<std::string>({std::nullopt});
  auto expr = createFromCsv(ROW({}, {}));
  auto result = evaluate(expr, makeRowVector({input}));
  EXPECT_TRUE(result->isNullAt(0));
}

// Spark's default escape is backslash, not double quote. Doubled quotes only
// collapse when the second quote can also close the field; otherwise
// STOP_AT_DELIMITER preserves the field literally.
TEST_F(FromCsvTest, doubledQuoteWithDefaultEscape) {
  auto input = makeFlatVector<std::string>({
      R"("he""llo",world)",
      R"("abc"",x)",
      R"("a ""quoted"" string",other)",
      R"("""quoted""",other)",
      R"("abc""",x)",
      R"("""",x)",
  });
  auto expr = createFromCsv(ROW({"a", "b"}, {VARCHAR(), VARCHAR()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childA = resultRow->childAt(0)->asFlatVector<StringView>();
  auto* childB = resultRow->childAt(1)->asFlatVector<StringView>();
  EXPECT_EQ(childA->valueAt(0).getString(), R"("he""llo")");
  EXPECT_EQ(childB->valueAt(0).getString(), "world");
  EXPECT_EQ(childA->valueAt(1).getString(), "abc\"");
  EXPECT_EQ(childB->valueAt(1).getString(), "x");
  EXPECT_EQ(childA->valueAt(2).getString(), R"("a ""quoted"" string")");
  EXPECT_EQ(childB->valueAt(2).getString(), "other");
  EXPECT_EQ(childA->valueAt(3).getString(), R"("""quoted""")");
  EXPECT_EQ(childB->valueAt(3).getString(), "other");
  EXPECT_EQ(childA->valueAt(4).getString(), "abc\"\"");
  EXPECT_EQ(childB->valueAt(4).getString(), "x");
  EXPECT_EQ(childA->valueAt(5).getString(), "\"\"");
  EXPECT_EQ(childB->valueAt(5).getString(), "x");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
