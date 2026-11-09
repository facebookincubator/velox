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
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

#include <folly/ScopeGuard.h>

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
    cudf_velox::registerPrestoFunctions("");
  }

  void TearDown() override {
    cudf_velox::unregisterFunctions();
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  // Runs the projections on the GPU, which fails if any stays on the CPU, and
  // compares the result with the CPU.
  void assertProjectMatchesCpu(
      const RowVectorPtr& input,
      const std::vector<std::string>& projections) {
    auto plan = PlanBuilder().values({input}).project(projections).planNode();
    auto gpuResult = AssertQueryBuilder(plan).copyResults(pool());
    cudf_velox::unregisterCudf();
    SCOPE_EXIT {
      cudf_velox::registerCudf();
    };
    auto cpuResult = AssertQueryBuilder(plan).copyResults(pool());
    facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
  }

  bool canRunOnGpu(const std::string& sql) {
    auto queryCtx = core::QueryCtx::create();
    return cudf_velox::canExprRunOnGpu(
        parseExpr(sql, ROW("c0", VARCHAR())), queryCtx.get(), pool());
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

// Input: {"café", "cafe", "cafés"}, pattern: 'caf_'
// Compared with explicit values because DuckDB 0.8.1 matches `_` against a
// single byte rather than a single UTF-8 character.
TEST_F(CudfStringFunctionTest, likeUnicodeSingleCharWildcard) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"caf\xc3\xa9", "cafe", "caf\xc3\xa9s"})});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"c0 LIKE 'caf_' AS result"})
                  .planNode();

  // 'caf_' matches one UTF-8 character, so "café" matches.
  AssertQueryBuilder(plan).assertResults(
      makeRowVector({makeFlatVector<bool>({true, true, false})}));
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

// ---- TRIM and SPLIT tests -------------------------------------------------

// Presto trims only the whitespace code points in stringImpl; similar-looking
// non-whitespace code points and the control byte \x01 stay.
TEST_F(CudfStringFunctionTest, trimFunctions) {
  std::vector<std::optional<std::string>> strings{
      "", std::nullopt, "   ", "\x01 d \x01", "\u2007e\u2007"};
  // All 25 whitespace code points recognized by
  // stringImpl::isUnicodeWhiteSpace.
  for (const auto* whitespace :
       {"\t",     "\n",     "\v",     "\f",     "\r",     "\x1c",   "\x1d",
        "\x1e",   "\x1f",   " ",      "\u1680", "\u2000", "\u2001", "\u2002",
        "\u2003", "\u2004", "\u2005", "\u2006", "\u2008", "\u2009", "\u200a",
        "\u2028", "\u2029", "\u205f", "\u3000"}) {
    strings.push_back(std::string(whitespace) + "x" + whitespace);
  }
  for (const auto* nonWhitespace :
       {"\u00a0", "\u202f", "\u0085", "\u200b", "\ufeff"}) {
    strings.push_back(std::string(nonWhitespace) + "x" + nonWhitespace);
  }
  assertProjectMatchesCpu(
      makeRowVector({makeNullableFlatVector<std::string>(strings)}),
      {"trim(c0)", "ltrim(c0)", "rtrim(c0)"});

  for (const auto* sql :
       {"trim(c0, 'x')", "ltrim(c0, 'x')", "rtrim(c0, 'x')"}) {
    SCOPED_TRACE(sql);
    EXPECT_FALSE(canRunOnGpu(sql));
  }
}

TEST_F(CudfStringFunctionTest, split) {
  auto input = makeRowVector({makeNullableFlatVector<std::string>(
      {"a,b", "a,b,c", "a,,b", ",a,", "", std::nullopt, "abc"})});
  assertProjectMatchesCpu(
      input,
      {"split(c0, ',')",
       "split(c0, ',', 2)",
       "split(c0, ',', 3)",
       "split(c0, ',', cast(2 as integer))"});
}

TEST_F(CudfStringFunctionTest, splitOverlappingDelimiter) {
  auto input = makeRowVector({makeFlatVector<std::string>({"aaa"})});
  assertProjectMatchesCpu(input, {"split(c0, 'aa')"});
}

TEST_F(CudfStringFunctionTest, splitLongStrings) {
  // Keep long strings in their own batch so the average exceeds 120 bytes.
  const std::string part(128, 'a');
  auto input = makeRowVector(
      {makeFlatVector<std::string>({part + ",," + part + "," + part})});
  assertProjectMatchesCpu(
      input, {"split(c0, ',')", "split(c0, ',', 2)", "split(c0, ',', 3)"});
}

TEST_F(CudfStringFunctionTest, splitEligibility) {
  EXPECT_TRUE(canRunOnGpu("split(c0, ',', 2147483647)"));

  // cuDF splits on whitespace for an empty delimiter. Presto returns the input
  // for a limit of 1 and raises below; cuDF takes an int32 limit.
  for (const auto* sql :
       {"split(c0, '')",
        "split(c0, '', 2)",
        "split(c0, c0)",
        "split(c0, ',', 1)",
        "split(c0, ',', 0)",
        "split(c0, ',', -1)",
        "split(c0, ',', 2147483648)"}) {
    SCOPED_TRACE(sql);
    EXPECT_FALSE(canRunOnGpu(sql));
  }
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
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
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
