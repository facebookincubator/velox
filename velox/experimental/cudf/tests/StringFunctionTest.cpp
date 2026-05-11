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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

// ---------------------------------------------------------------------------
// Plan-based tests: run a Velox plan through the cudf engine and verify
// results against DuckDB.
// ---------------------------------------------------------------------------
class CudfStringFunctionTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }
};

// ---- LIKE tests -----------------------------------------------------------

TEST_F(CudfStringFunctionTest, likeWildcard) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"hello", "world", "help", "held"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"c0 LIKE '%el%' AS result"})
                  .planNode();

  assertQuery(plan, "SELECT c0 LIKE '%el%' AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, likePrefixPattern) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"apple", "apricot", "banana", "avocado"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"c0 LIKE 'ap%' AS result"})
                  .planNode();

  assertQuery(plan, "SELECT c0 LIKE 'ap%' AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, likeSuffixPattern) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"cat", "bat", "car", "hat"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"c0 LIKE '%at' AS result"})
                  .planNode();

  assertQuery(plan, "SELECT c0 LIKE '%at' AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, likeSingleCharWildcard) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"cat", "cut", "cot", "cart"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"c0 LIKE 'c_t' AS result"})
                  .planNode();

  assertQuery(plan, "SELECT c0 LIKE 'c_t' AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, likeExactMatch) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"exact", "EXACT", "exactly", "ex"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"c0 LIKE 'exact' AS result"})
                  .planNode();

  assertQuery(plan, "SELECT c0 LIKE 'exact' AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, likeNoMatch) {
  auto input =
      makeRowVector({makeFlatVector<std::string>({"abc", "def", "ghi"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"c0 LIKE '%xyz%' AS result"})
                  .planNode();

  assertQuery(plan, "SELECT c0 LIKE '%xyz%' AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, likeEmptyStrings) {
  auto input =
      makeRowVector({makeFlatVector<std::string>({"", "a", "", "bc"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"c0 LIKE '' AS result"})
                  .planNode();

  assertQuery(plan, "SELECT c0 LIKE '' AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, likeInFilter) {
  auto input = makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3, 4}),
       makeFlatVector<std::string>({"hello", "world", "help", "held"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .filter("c1 LIKE 'hel%'")
                  .project({"c0", "c1"})
                  .planNode();

  assertQuery(plan, "SELECT c0, c1 FROM tmp WHERE c1 LIKE 'hel%'");
}

TEST_F(CudfStringFunctionTest, likeWithNulls) {
  auto input = makeRowVector({makeNullableFlatVector<std::string>(
      {"hello", std::nullopt, "help", std::nullopt})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"c0 LIKE '%el%' AS result"})
                  .planNode();

  assertQuery(plan, "SELECT c0 LIKE '%el%' AS result FROM tmp");
}

// Input: {"café", "naïve", "hello"}, pattern: '%fé'
TEST_F(CudfStringFunctionTest, likeUnicode) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"caf\xc3\xa9", "na\xc3\xafve", "hello"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"c0 LIKE '%f\xc3\xa9' AS result"})
                  .planNode();

  assertQuery(plan, "SELECT c0 LIKE '%f\xc3\xa9' AS result FROM tmp");
}

// DuckDB single-char wildcard `_` behaves different from cuDF (and hence from
// the SQL standard) with multi-byte UTF-8 characters, treating each byte as a
// separate character. The first and third rows of this test therefore fail
// comparison with the DuckDB result, even though they match the pattern
// according to the SQL standard. Leaving this test disabled for now, pending
// further investigation.
TEST_F(CudfStringFunctionTest, DISABLED_likeUnicodeSingleCharWildcard) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"caf\xc3\xa9", "cafe", "caf\xc3\xa9s"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"c0 LIKE 'caf_' AS result"})
                  .planNode();

  assertQuery(plan, "SELECT c0 LIKE 'caf_' AS result FROM tmp");
}

// ---- UPPER tests ----------------------------------------------------------

TEST_F(CudfStringFunctionTest, upperBasic) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"hello", "World", "FOO", "bAr"})});
  createDuckDbTable({input});

  auto plan =
      PlanBuilder().values({input}).project({"UPPER(c0) AS result"}).planNode();

  assertQuery(plan, "SELECT UPPER(c0) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, upperAlreadyUpper) {
  auto input =
      makeRowVector({makeFlatVector<std::string>({"ABC", "XYZ", "HELLO"})});
  createDuckDbTable({input});

  auto plan =
      PlanBuilder().values({input}).project({"UPPER(c0) AS result"}).planNode();

  assertQuery(plan, "SELECT UPPER(c0) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, upperEmptyAndSpecial) {
  auto input =
      makeRowVector({makeFlatVector<std::string>({"", "123", "a1b2", "!@#"})});
  createDuckDbTable({input});

  auto plan =
      PlanBuilder().values({input}).project({"UPPER(c0) AS result"}).planNode();

  assertQuery(plan, "SELECT UPPER(c0) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, upperWithNulls) {
  auto input = makeRowVector({makeNullableFlatVector<std::string>(
      {"hello", std::nullopt, "WORLD", std::nullopt})});
  createDuckDbTable({input});

  auto plan =
      PlanBuilder().values({input}).project({"UPPER(c0) AS result"}).planNode();

  assertQuery(plan, "SELECT UPPER(c0) AS result FROM tmp");
}

// Input: {"café", "naïve", "hello"}
TEST_F(CudfStringFunctionTest, upperUnicode) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"caf\xc3\xa9", "na\xc3\xafve", "hello"})});
  createDuckDbTable({input});

  auto plan =
      PlanBuilder().values({input}).project({"UPPER(c0) AS result"}).planNode();

  assertQuery(plan, "SELECT UPPER(c0) AS result FROM tmp");
}

// ---- LOWER tests ----------------------------------------------------------

TEST_F(CudfStringFunctionTest, lowerBasic) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"HELLO", "World", "foo", "bAr"})});
  createDuckDbTable({input});

  auto plan =
      PlanBuilder().values({input}).project({"LOWER(c0) AS result"}).planNode();

  assertQuery(plan, "SELECT LOWER(c0) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, lowerAlreadyLower) {
  auto input =
      makeRowVector({makeFlatVector<std::string>({"abc", "xyz", "hello"})});
  createDuckDbTable({input});

  auto plan =
      PlanBuilder().values({input}).project({"LOWER(c0) AS result"}).planNode();

  assertQuery(plan, "SELECT LOWER(c0) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, lowerEmptyAndSpecial) {
  auto input =
      makeRowVector({makeFlatVector<std::string>({"", "123", "A1B2", "!@#"})});
  createDuckDbTable({input});

  auto plan =
      PlanBuilder().values({input}).project({"LOWER(c0) AS result"}).planNode();

  assertQuery(plan, "SELECT LOWER(c0) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, lowerWithNulls) {
  auto input = makeRowVector({makeNullableFlatVector<std::string>(
      {"HELLO", std::nullopt, "world", std::nullopt})});
  createDuckDbTable({input});

  auto plan =
      PlanBuilder().values({input}).project({"LOWER(c0) AS result"}).planNode();

  assertQuery(plan, "SELECT LOWER(c0) AS result FROM tmp");
}

// Input: {"CAFÉ", "NAÏVE", "HELLO"}
TEST_F(CudfStringFunctionTest, lowerUnicode) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"CAF\xc3\x89", "NA\xc3\x8fVE", "HELLO"})});
  createDuckDbTable({input});

  auto plan =
      PlanBuilder().values({input}).project({"LOWER(c0) AS result"}).planNode();

  assertQuery(plan, "SELECT LOWER(c0) AS result FROM tmp");
}

// ---- CONCAT tests ---------------------------------------------------------

TEST_F(CudfStringFunctionTest, concatTwoColumns) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"hello", "foo", ""}),
       makeFlatVector<std::string>({" world", "bar", "empty"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"concat(c0, c1) AS result"})
                  .planNode();

  assertQuery(plan, "SELECT concat(c0, c1) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, concatThreeColumns) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"a", "d"}),
       makeFlatVector<std::string>({"b", "e"}),
       makeFlatVector<std::string>({"c", "f"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"concat(c0, c1, c2) AS result"})
                  .planNode();

  assertQuery(plan, "SELECT concat(c0, c1, c2) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, concatColumnAndLiteral) {
  auto input = makeRowVector({makeFlatVector<std::string>({"hello", "world"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"concat(c0, '!!!') AS result"})
                  .planNode();

  assertQuery(plan, "SELECT concat(c0, '!!!') AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, concatLiteralAndColumn) {
  auto input = makeRowVector({makeFlatVector<std::string>({"world", "there"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"concat('hello ', c0) AS result"})
                  .planNode();

  assertQuery(plan, "SELECT concat('hello ', c0) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, concatWithEmptyStrings) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"", "a", ""}),
       makeFlatVector<std::string>({"", "", "b"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"concat(c0, c1) AS result"})
                  .planNode();

  assertQuery(plan, "SELECT concat(c0, c1) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, concatMultipleLiterals) {
  auto input = makeRowVector({makeFlatVector<std::string>({"mid"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"concat('start-', c0, '-end') AS result"})
                  .planNode();

  assertQuery(plan, "SELECT concat('start-', c0, '-end') AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, concatWithNullColumn) {
  auto input = makeRowVector(
      {makeNullableFlatVector<std::string>({"hello", std::nullopt, "foo"}),
       makeNullableFlatVector<std::string>({" world", "bar", std::nullopt})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"concat(c0, c1) AS result"})
                  .planNode();

  assertQuery(plan, "SELECT concat(c0, c1) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, concatAllNulls) {
  auto input = makeRowVector(
      {makeNullableFlatVector<std::string>({std::nullopt, std::nullopt}),
       makeNullableFlatVector<std::string>({std::nullopt, std::nullopt})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"concat(c0, c1) AS result"})
                  .planNode();

  assertQuery(plan, "SELECT concat(c0, c1) AS result FROM tmp");
}

// Input: {"hello", "München"} + {" wörld", " Straße"}
TEST_F(CudfStringFunctionTest, concatUnicode) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"hello", "M\xc3\xbcnchen"}),
       makeFlatVector<std::string>(
           {" w\xc3\xb6rld",
            " Stra\xc3\x9f"
            "e"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"concat(c0, c1) AS result"})
                  .planNode();

  assertQuery(plan, "SELECT concat(c0, c1) AS result FROM tmp");
}

// ---------------------------------------------------------------------------
// Direct expression tests: exercise the cudf expression evaluator directly
// via CudfFunctionBaseTest::evaluateOnce.
// ---------------------------------------------------------------------------
class CudfStringExprTest : public cudf_velox::CudfFunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    cudf_velox::registerCudf();
  }

  static void TearDownTestCase() {
    cudf_velox::unregisterCudf();
  }
};

TEST_F(CudfStringExprTest, likeMatch) {
  auto result = evaluateOnce<bool, std::string>("c0 LIKE '%ell%'", "hello");
  EXPECT_TRUE(result.value());
}

TEST_F(CudfStringExprTest, likeNoMatch) {
  auto result = evaluateOnce<bool, std::string>("c0 LIKE '%xyz%'", "hello");
  EXPECT_FALSE(result.value());
}

TEST_F(CudfStringExprTest, likePrefixMatch) {
  auto result = evaluateOnce<bool, std::string>("c0 LIKE 'hel%'", "hello");
  EXPECT_TRUE(result.value());
}

TEST_F(CudfStringExprTest, likeSuffixMatch) {
  auto result = evaluateOnce<bool, std::string>("c0 LIKE '%llo'", "hello");
  EXPECT_TRUE(result.value());
}

TEST_F(CudfStringExprTest, likeExact) {
  auto result = evaluateOnce<bool, std::string>("c0 LIKE 'hello'", "hello");
  EXPECT_TRUE(result.value());
}

TEST_F(CudfStringExprTest, likeSingleChar) {
  auto result = evaluateOnce<bool, std::string>("c0 LIKE 'h_llo'", "hello");
  EXPECT_TRUE(result.value());
}

TEST_F(CudfStringExprTest, likeNullInput) {
  auto result = evaluateOnce<bool, std::string>(
      "c0 LIKE '%el%'", std::optional<std::string>(std::nullopt));
  EXPECT_FALSE(result.has_value());
}

// "café" LIKE '%fé'
TEST_F(CudfStringExprTest, likeUnicode) {
  auto result = evaluateOnce<bool, std::string>(
      "c0 LIKE '%f\xc3\xa9'", std::string("caf\xc3\xa9"));
  EXPECT_TRUE(result.value());
}

// "cafe" (no accent) should NOT match '%fé'
TEST_F(CudfStringExprTest, likeUnicodeNoMatch) {
  auto result = evaluateOnce<bool, std::string>(
      "c0 LIKE '%f\xc3\xa9'", std::string("cafe"));
  EXPECT_FALSE(result.value());
}

TEST_F(CudfStringExprTest, upperBasic) {
  auto result = evaluateOnce<std::string, std::string>("upper(c0)", "hello");
  EXPECT_EQ(result.value(), "HELLO");
}

TEST_F(CudfStringExprTest, upperMixedCase) {
  auto result = evaluateOnce<std::string, std::string>("upper(c0)", "HeLLo");
  EXPECT_EQ(result.value(), "HELLO");
}

TEST_F(CudfStringExprTest, upperEmpty) {
  auto result = evaluateOnce<std::string, std::string>("upper(c0)", "");
  EXPECT_EQ(result.value(), "");
}

TEST_F(CudfStringExprTest, upperNullInput) {
  auto result = evaluateOnce<std::string, std::string>(
      "upper(c0)", std::optional<std::string>(std::nullopt));
  EXPECT_FALSE(result.has_value());
}

// upper("café") -> "CAFÉ"
TEST_F(CudfStringExprTest, upperUnicode) {
  auto result = evaluateOnce<std::string, std::string>(
      "upper(c0)", std::string("caf\xc3\xa9"));
  EXPECT_EQ(result.value(), "CAF\xc3\x89");
}

TEST_F(CudfStringExprTest, lowerBasic) {
  auto result = evaluateOnce<std::string, std::string>("lower(c0)", "HELLO");
  EXPECT_EQ(result.value(), "hello");
}

TEST_F(CudfStringExprTest, lowerMixedCase) {
  auto result = evaluateOnce<std::string, std::string>("lower(c0)", "HeLLo");
  EXPECT_EQ(result.value(), "hello");
}

TEST_F(CudfStringExprTest, lowerEmpty) {
  auto result = evaluateOnce<std::string, std::string>("lower(c0)", "");
  EXPECT_EQ(result.value(), "");
}

TEST_F(CudfStringExprTest, lowerNullInput) {
  auto result = evaluateOnce<std::string, std::string>(
      "lower(c0)", std::optional<std::string>(std::nullopt));
  EXPECT_FALSE(result.has_value());
}

// lower("CAFÉ") -> "café"
TEST_F(CudfStringExprTest, lowerUnicode) {
  auto result = evaluateOnce<std::string, std::string>(
      "lower(c0)", std::string("CAF\xc3\x89"));
  EXPECT_EQ(result.value(), "caf\xc3\xa9");
}

TEST_F(CudfStringExprTest, concatTwoArgs) {
  auto result = evaluateOnce<std::string, std::string, std::string>(
      "concat(c0, c1)", "hello", " world");
  EXPECT_EQ(result.value(), "hello world");
}

TEST_F(CudfStringExprTest, concatWithLiteral) {
  auto result =
      evaluateOnce<std::string, std::string>("concat(c0, '!')", "hello");
  EXPECT_EQ(result.value(), "hello!");
}

TEST_F(CudfStringExprTest, concatLiteralFirst) {
  auto result =
      evaluateOnce<std::string, std::string>("concat('hi ', c0)", "there");
  EXPECT_EQ(result.value(), "hi there");
}

TEST_F(CudfStringExprTest, concatEmptyStrings) {
  auto result = evaluateOnce<std::string, std::string, std::string>(
      "concat(c0, c1)", "", "");
  EXPECT_EQ(result.value(), "");
}

TEST_F(CudfStringExprTest, concatWithNullArg) {
  auto result = evaluateOnce<std::string, std::string, std::string>(
      "concat(c0, c1)",
      std::optional<std::string>("hello"),
      std::optional<std::string>(std::nullopt));
  // Velox CPU, DuckDB, and cudf treat nulls as empty strings here.
  EXPECT_EQ(result.value(), "hello");
}

// concat("hello", " wörld") -> "hello wörld"
TEST_F(CudfStringExprTest, concatUnicode) {
  auto result = evaluateOnce<std::string, std::string, std::string>(
      "concat(c0, c1)", std::string("hello"), std::string(" w\xc3\xb6rld"));
  EXPECT_EQ(result.value(), "hello w\xc3\xb6rld");
}

} // namespace
