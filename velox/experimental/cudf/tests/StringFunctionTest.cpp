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
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"abc", "def", "ghi"})});
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

  assertQuery(
      plan, "SELECT c0, c1 FROM tmp WHERE c1 LIKE 'hel%'");
}

// ---- UPPER tests ----------------------------------------------------------

TEST_F(CudfStringFunctionTest, upperBasic) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"hello", "World", "FOO", "bAr"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"UPPER(c0) AS result"})
                  .planNode();

  assertQuery(plan, "SELECT UPPER(c0) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, upperAlreadyUpper) {
  auto input =
      makeRowVector({makeFlatVector<std::string>({"ABC", "XYZ", "HELLO"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"UPPER(c0) AS result"})
                  .planNode();

  assertQuery(plan, "SELECT UPPER(c0) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, upperEmptyAndSpecial) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"", "123", "a1b2", "!@#"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"UPPER(c0) AS result"})
                  .planNode();

  assertQuery(plan, "SELECT UPPER(c0) AS result FROM tmp");
}

// ---- LOWER tests ----------------------------------------------------------

TEST_F(CudfStringFunctionTest, lowerBasic) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"HELLO", "World", "foo", "bAr"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"LOWER(c0) AS result"})
                  .planNode();

  assertQuery(plan, "SELECT LOWER(c0) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, lowerAlreadyLower) {
  auto input =
      makeRowVector({makeFlatVector<std::string>({"abc", "xyz", "hello"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"LOWER(c0) AS result"})
                  .planNode();

  assertQuery(plan, "SELECT LOWER(c0) AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, lowerEmptyAndSpecial) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"", "123", "A1B2", "!@#"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"LOWER(c0) AS result"})
                  .planNode();

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
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"hello", "world"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"concat(c0, '!!!') AS result"})
                  .planNode();

  assertQuery(plan, "SELECT concat(c0, '!!!') AS result FROM tmp");
}

TEST_F(CudfStringFunctionTest, concatLiteralAndColumn) {
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"world", "there"})});
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
  auto input = makeRowVector(
      {makeFlatVector<std::string>({"mid"})});
  createDuckDbTable({input});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"concat('start-', c0, '-end') AS result"})
                  .planNode();

  assertQuery(plan, "SELECT concat('start-', c0, '-end') AS result FROM tmp");
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

} // namespace
