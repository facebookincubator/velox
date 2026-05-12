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

  auto expr = createFromCsv(expected->type());
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();

  for (int i = 0; i < 2; ++i) {
    EXPECT_FALSE(resultRow->isNullAt(i));
    auto* childB = resultRow->childAt(1)->asFlatVector<double>();
    EXPECT_TRUE(childB->isNullAt(i)) << "row " << i;
  }
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

// NaN and Infinity for float/double fields — Spark-compatible case sensitivity.
TEST_F(FromCsvTest, nanAndInfinity) {
  // Spark accepts: exact "NaN", "Infinity"/"+Infinity"/"-Infinity" (Java),
  // and "Inf"/"-Inf" (Spark CSV defaults: positiveInf="Inf",
  // negativeInf="-Inf").
  auto input = makeFlatVector<std::string>(
      {"NaN,NaN",
       "Infinity,-Infinity",
       "+Infinity,+Infinity",
       "Inf,-Inf",
       "inf,-inf",
       "nan,INFINITY"});

  auto expr = createFromCsv(ROW({"a", "b"}, {DOUBLE(), DOUBLE()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();

  auto* childA = resultRow->childAt(0)->asFlatVector<double>();
  auto* childB = resultRow->childAt(1)->asFlatVector<double>();
  // Row 0: NaN, NaN — accepted.
  EXPECT_TRUE(std::isnan(childA->valueAt(0)));
  EXPECT_TRUE(std::isnan(childB->valueAt(0)));
  // Row 1: Infinity, -Infinity — accepted (Java parseDouble).
  EXPECT_TRUE(std::isinf(childA->valueAt(1)));
  EXPECT_GT(childA->valueAt(1), 0);
  EXPECT_TRUE(std::isinf(childB->valueAt(1)));
  EXPECT_LT(childB->valueAt(1), 0);
  // Row 2: +Infinity, +Infinity — accepted.
  EXPECT_TRUE(std::isinf(childA->valueAt(2)));
  EXPECT_GT(childA->valueAt(2), 0);
  EXPECT_TRUE(std::isinf(childB->valueAt(2)));
  EXPECT_GT(childB->valueAt(2), 0);
  // Row 3: "Inf", "-Inf" — accepted (Spark CSV defaults).
  EXPECT_TRUE(std::isinf(childA->valueAt(3)));
  EXPECT_GT(childA->valueAt(3), 0);
  EXPECT_TRUE(std::isinf(childB->valueAt(3)));
  EXPECT_LT(childB->valueAt(3), 0);
  // Row 4: "inf", "-inf" — rejected (case-sensitive).
  EXPECT_TRUE(childA->isNullAt(4));
  EXPECT_TRUE(childB->isNullAt(4));
  // Row 5: "nan", "INFINITY" — rejected (case-sensitive).
  EXPECT_TRUE(childA->isNullAt(5));
  EXPECT_TRUE(childB->isNullAt(5));
}

// Leading '+' accepted for integers (Spark compatibility).
TEST_F(FromCsvTest, leadingPlusSign) {
  auto input = makeFlatVector<std::string>({"+123,+456", "+1,+2", "+789,+0"});
  auto expectedA = makeFlatVector<int32_t>({123, 1, 789});
  auto expectedB = makeFlatVector<int32_t>({456, 2, 0});
  auto expected = makeRowVector({"a", "b"}, {expectedA, expectedB});
  testFromCsv(input, expected);
}

// Hex float notation rejected (Spark does not accept 0x1A as number).
TEST_F(FromCsvTest, hexFloatRejected) {
  auto input =
      makeFlatVector<std::string>({"0x1A", "0X1.0p10", "+0x1", "-0x1p2"});

  auto expr = createFromCsv(ROW({"a"}, {DOUBLE()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childA = resultRow->childAt(0)->asFlatVector<double>();
  EXPECT_TRUE(childA->isNullAt(0));
  EXPECT_TRUE(childA->isNullAt(1));
  EXPECT_TRUE(childA->isNullAt(2));
  EXPECT_TRUE(childA->isNullAt(3));
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

// Quoted fields.
TEST_F(FromCsvTest, quotedFields) {
  auto input =
      makeFlatVector<std::string>({R"("hello, world",1)", R"("a""b",2)"});

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
  // RFC 4180: escaped double-quote ("") is unescaped to a single quote.
  EXPECT_EQ(childA->valueAt(1).str(), "a\"b");
  EXPECT_EQ(childB->valueAt(1), 2);
}

// Whitespace trimming: only REAL/DOUBLE trim (matching Java's parseDouble).
// Integer fields with whitespace → NULL (matching Java's parseInt rejection).
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

// Unclosed quoted field — Velox parser keeps the leading quote in the
// output (treats the field as unquoted on missing close-quote). Spark's
// Univocity parser strips the leading quote and accepts the rest as the
// field value. This divergence is tracked under ADO 5199738; until the
// Velox parser is fixed, capture current behavior so the test reflects
// reality and we get a regression signal if behavior changes.
TEST_F(FromCsvTest, unclosedQuote) {
  auto input = makeFlatVector<std::string>({R"("unclosed)"});
  auto expr = createFromCsv(ROW({"a"}, {VARCHAR()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childA = resultRow->childAt(0)->asFlatVector<StringView>();
  EXPECT_FALSE(childA->isNullAt(0));
  // TODO(ADO 5199738): Spark returns "unclosed" (quote stripped). Update
  // expected value to "unclosed" once Velox parser is fixed for parity.
  EXPECT_EQ(childA->valueAt(0).str(), "\"unclosed");
}

// Empty input line produces null ROW (Spark's UnivocityParser returns null for
// empty input).
TEST_F(FromCsvTest, emptyInputLine) {
  auto input = makeFlatVector<std::string>({""});
  auto expr = createFromCsv(ROW({"a", "b"}, {INTEGER(), VARCHAR()}));
  auto result = evaluate(expr, makeRowVector({input}));
  // The struct itself is null for empty input.
  EXPECT_TRUE(result->isNullAt(0));
}

// Empty VARCHAR fields become null (Spark default nullValue="").
TEST_F(FromCsvTest, emptyVarcharIsNull) {
  auto input = makeFlatVector<std::string>({",1", R"("",2)", "hello,3"});
  auto expr = createFromCsv(ROW({"s", "i"}, {VARCHAR(), INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childS = resultRow->childAt(0)->asFlatVector<StringView>();
  auto* childI = resultRow->childAt(1)->asFlatVector<int32_t>();
  // Row 0: empty unquoted field → null.
  EXPECT_TRUE(childS->isNullAt(0));
  EXPECT_EQ(childI->valueAt(0), 1);
  // Row 1: quoted empty field "" → null.
  EXPECT_TRUE(childS->isNullAt(1));
  EXPECT_EQ(childI->valueAt(1), 2);
  // Row 2: non-empty field → preserved.
  EXPECT_FALSE(childS->isNullAt(2));
  EXPECT_EQ(childS->valueAt(2).str(), "hello");
  EXPECT_EQ(childI->valueAt(2), 3);
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
  auto input = makeFlatVector<std::string>(
      {"1e400,-1e400", "3.4028236e38,1e-400", "+1e309,-1e309"});
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
  // 1e-400 underflow → 0.
  EXPECT_FALSE(childB->isNullAt(1));
  EXPECT_DOUBLE_EQ(childB->valueAt(1), 0.0);
  // Row 2: +1e309 → +inf, -1e309 → -inf.
  EXPECT_TRUE(std::isinf(childA->valueAt(2)));
  EXPECT_GT(childA->valueAt(2), 0);
  EXPECT_TRUE(std::isinf(childB->valueAt(2)));
  EXPECT_LT(childB->valueAt(2), 0);
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

// Characters after closing quote are skipped (permissive mode).
TEST_F(FromCsvTest, garbageAfterQuote) {
  auto input = makeFlatVector<std::string>({R"("hello"world,1)"});
  auto expr = createFromCsv(ROW({"a", "b"}, {VARCHAR(), INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childA = resultRow->childAt(0)->asFlatVector<StringView>();
  auto* childB = resultRow->childAt(1)->asFlatVector<int32_t>();
  // Quoted field: content is "hello", trailing "world" is skipped.
  EXPECT_EQ(childA->valueAt(0).str(), "hello");
  EXPECT_EQ(childB->valueAt(0), 1);
}

// Row nullness: non-null non-empty input always produces non-null row, even if
// all fields fail to parse. Empty string input produces null row.
TEST_F(FromCsvTest, rowNullness) {
  auto input = makeFlatVector<std::string>({"abc,xyz", ",", "1,2", ""});
  auto expr = createFromCsv(ROW({"a", "b"}, {INTEGER(), INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  // Non-empty input: row is never null, even when all fields are null.
  for (int i = 0; i < 3; ++i) {
    EXPECT_FALSE(resultRow->isNullAt(i)) << "row " << i;
  }
  // Empty string input: row is null (matches Spark).
  EXPECT_TRUE(result->isNullAt(3));
}

// Unsupported field type throws.
TEST_F(FromCsvTest, unsupportedFieldType) {
  auto input = makeFlatVector<std::string>({"1,2,3"});
  auto expr = createFromCsv(ROW({"a"}, {ARRAY(INTEGER())}));
  VELOX_ASSERT_THROW(
      evaluate(expr, makeRowVector({input})), "Unsupported field type");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
