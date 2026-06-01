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
#include "velox/core/QueryConfig.h"
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

// Unclosed quoted field — Spark's UnivocityParser strips the opening quote
// and returns the rest as the field value.
TEST_F(FromCsvTest, unclosedQuote) {
  auto input = makeFlatVector<std::string>({R"("unclosed)"});
  auto expr = createFromCsv(ROW({"a"}, {VARCHAR()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childA = resultRow->childAt(0)->asFlatVector<StringView>();
  EXPECT_FALSE(childA->isNullAt(0));
  // Spark's UnivocityParser strips the leading quote for unclosed quoted
  // fields.
  EXPECT_EQ(childA->valueAt(0).str(), "unclosed");
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

// Oversized input (>10MB) returns non-null row with all-null fields (DoS
// guard).
TEST_F(FromCsvTest, oversizedInput) {
  // Create a string larger than kMaxCsvLineSize (10 MB).
  std::string largeStr(10 * 1024 * 1024 + 1, 'a');
  auto input = makeFlatVector<std::string>({largeStr});
  auto expr = createFromCsv(ROW({"a", "b"}, {VARCHAR(), INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  // Row itself is NOT null (Spark permissive mode returns row-present).
  EXPECT_FALSE(result->isNullAt(0));
  auto* resultRow = result->as<RowVector>();
  // But all child fields are null.
  EXPECT_TRUE(resultRow->childAt(0)->isNullAt(0));
  EXPECT_TRUE(resultRow->childAt(1)->isNullAt(0));
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
  // "  -99.99  " (whitespace trimmed for decimal) → -9999
  EXPECT_FALSE(child->isNullAt(1));
  EXPECT_EQ(child->valueAt(1), -9999);
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
  // "  0.00  " → 0
  EXPECT_FALSE(child->isNullAt(1));
  EXPECT_EQ(child->valueAt(1), 0);
  // "invalid" → null
  EXPECT_TRUE(child->isNullAt(2));
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

// TIMESTAMP field parsing.
TEST_F(FromCsvTest, timestampField) {
  auto input = makeFlatVector<std::string>(
      {"2023-01-15T10:30:00", "1970-01-01 00:00:00", "bad-ts", ""});
  auto expr = createFromCsv(ROW({"a"}, {TIMESTAMP()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* child = resultRow->childAt(0)->asFlatVector<Timestamp>();
  // "2023-01-15T10:30:00" parses as wall-clock time (no session timezone
  // configured in test → no adjustment).
  EXPECT_FALSE(child->isNullAt(0));
  EXPECT_EQ(child->valueAt(0), Timestamp(1673778600, 0));
  // "1970-01-01 00:00:00" → epoch
  EXPECT_FALSE(child->isNullAt(1));
  EXPECT_EQ(child->valueAt(1), Timestamp(0, 0));
  // "bad-ts" → null
  EXPECT_TRUE(child->isNullAt(2));
  // "" → null
  EXPECT_TRUE(child->isNullAt(3));
}

// Naive timestamp with session timezone: Spark interprets naive timestamps
// (no timezone offset in string) using the session timezone.
TEST_F(FromCsvTest, timestampWithSessionTimezone) {
  // Set session timezone to America/Los_Angeles (UTC-8 in winter).
  queryCtx_->testingOverrideConfigUnsafe({
      {core::QueryConfig::kSessionTimezone, "America/Los_Angeles"},
  });
  auto input = makeFlatVector<std::string>({"2023-01-15T10:30:00"});
  auto expr = createFromCsv(ROW({"a"}, {TIMESTAMP()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* child = resultRow->childAt(0)->asFlatVector<Timestamp>();
  EXPECT_FALSE(child->isNullAt(0));
  // 2023-01-15 10:30:00 PST = 2023-01-15 18:30:00 UTC = epoch 1673807400
  EXPECT_EQ(child->valueAt(0), Timestamp(1673807400, 0));
}

// Backslash escape in quoted fields (Spark default escape='\\').
TEST_F(FromCsvTest, backslashEscapeInQuotedField) {
  // \" inside a quoted field should emit a literal quote.
  auto input = makeFlatVector<std::string>(
      {R"("hello \"world\"",1)", R"("no escape",2)"});
  auto expr = createFromCsv(ROW({"s", "i"}, {VARCHAR(), INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childS = resultRow->childAt(0)->asFlatVector<StringView>();
  auto* childI = resultRow->childAt(1)->asFlatVector<int32_t>();
  // Row 0: backslash-escaped quotes → literal quotes in output.
  EXPECT_EQ(childS->valueAt(0).str(), "hello \"world\"");
  EXPECT_EQ(childI->valueAt(0), 1);
  // Row 1: no escape needed.
  EXPECT_EQ(childS->valueAt(1).str(), "no escape");
  EXPECT_EQ(childI->valueAt(1), 2);
}

// VARBINARY field parsing (raw bytes preserved as-is).
TEST_F(FromCsvTest, varbinaryField) {
  auto input =
      makeFlatVector<std::string>({"hello,world", "binary\x01\x02,data"});
  auto expr = createFromCsv(ROW({"a", "b"}, {VARBINARY(), VARBINARY()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* childA = resultRow->childAt(0)->asFlatVector<StringView>();
  auto* childB = resultRow->childAt(1)->asFlatVector<StringView>();
  // First row: "hello" and "world"
  EXPECT_EQ(childA->valueAt(0).getString(), "hello");
  EXPECT_EQ(childB->valueAt(0).getString(), "world");
  // Second row: "binary\x01\x02" and "data" (binary content preserved)
  EXPECT_EQ(childA->valueAt(1).getString(), std::string("binary\x01\x02"));
  EXPECT_EQ(childB->valueAt(1).getString(), "data");
}

// Float underflow edge cases.
TEST_F(FromCsvTest, floatUnderflow) {
  // ".000...1" pattern (leading dot without 0) should underflow to zero.
  std::string tinyDot = "." + std::string(400, '0') + "1";
  // "0.000...1" pattern (leading 0.) should underflow to zero.
  std::string tinyZeroDot = "0." + std::string(400, '0') + "1";
  auto input = makeFlatVector<std::string>({tinyDot, tinyZeroDot});
  auto expr = createFromCsv(ROW({"a"}, {DOUBLE()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* child = resultRow->childAt(0)->asFlatVector<double>();
  // Both should underflow to 0 (not ±Infinity).
  EXPECT_FALSE(child->isNullAt(0));
  EXPECT_EQ(child->valueAt(0), 0.0);
  EXPECT_FALSE(child->isNullAt(1));
  EXPECT_EQ(child->valueAt(1), 0.0);
}

// parseFloat rejects bare "+" sign (empty after strip).
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
  // "123.0" is not a valid integer — should return NULL.
  auto input = makeFlatVector<StringView>({"123.0"_sv, "45.6"_sv, "7.0"_sv});
  auto expr = createFromCsv(ROW({"a"}, {INTEGER()}));
  auto result = evaluate(expr, makeRowVector({input}));
  auto* resultRow = result->as<RowVector>();
  auto* child = resultRow->childAt(0)->asFlatVector<int32_t>();
  EXPECT_TRUE(child->isNullAt(0));
  EXPECT_TRUE(child->isNullAt(1));
  EXPECT_TRUE(child->isNullAt(2));
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

} // namespace
} // namespace facebook::velox::functions::sparksql::test
