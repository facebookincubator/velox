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
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox;

namespace facebook::velox::cudf_velox {
namespace {

class CudfFilterProjectTest : public CudfFunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::sparksql::registerFunctions("");
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    CudfConfig::getInstance().functionEngine = "spark";
    cudf_velox::registerCudf();
  }

  static void TearDownTestCase() {
    cudf_velox::unregisterCudf();
  }

  CudfFilterProjectTest() {
    options_.parseIntegerAsBigint = false;
  }
};

TEST_F(CudfFilterProjectTest, hashWithSeed) {
  auto input = makeFlatVector<int64_t>({INT64_MAX, INT64_MIN});
  auto data = makeRowVector({input});
  auto hashPlan = PlanBuilder()
                      .setParseOptions(options_)
                      .values({data})
                      .project({"hash_with_seed(42, c0) AS c1"})
                      .planNode();
  auto hashResults = AssertQueryBuilder(hashPlan).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({
          -1604625029,
          -853646085,
      }),
  });
  facebook::velox::test::assertEqualVectors(expected, hashResults);
}

// TODO: Re-enable after https://github.com/rapidsai/cudf/issues/21720.
// cuDF's murmurhash3_x86_32 combines columns via hash_combine(h(col0, seed),
// h(col1, seed)), while Spark instead hashes columns iteratively:
// h(col1, h(col0, seed)).
// These produce different results for multi-column inputs. The cudf hashing
// API only accepts a scalar uint32_t seed (no per-row seed column), so
// Spark's iterative semantics cannot be replicated without a custom CUDA
// kernel. Single-column hash_with_seed works correctly since there is no
// combining step.
TEST_F(CudfFilterProjectTest, DISABLED_hashWithSeedMultiColumns) {
  auto input = makeFlatVector<int64_t>({INT64_MAX, INT64_MIN});
  auto data = makeRowVector({input, input});
  auto hashPlan = PlanBuilder()
                      .setParseOptions(options_)
                      .values({data})
                      .project({"hash_with_seed(42, c0, c1) AS c2"})
                      .planNode();
  auto hashResults = AssertQueryBuilder(hashPlan).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({
          -572895191,
          724095561,
      }),
  });
  facebook::velox::test::assertEqualVectors(expected, hashResults);
}

TEST_F(CudfFilterProjectTest, dateAdd) {
  const auto dateAdd = [&](const std::string& dateStr, int32_t value) {
    return evaluateOnce<int32_t>(
        fmt::format("date_add(c0, {})", value),
        {DATE()},
        std::optional<int32_t>(parseDate(dateStr)));
  };

  // Check simple tests.
  EXPECT_EQ(parseDate("2019-03-01"), dateAdd("2019-03-01", 0));
  EXPECT_EQ(parseDate("2019-03-01"), dateAdd("2019-02-28", 1));

  // Account for the last day of a year-month
  EXPECT_EQ(parseDate("2020-02-29"), dateAdd("2019-01-30", 395));
  EXPECT_EQ(parseDate("2020-02-29"), dateAdd("2019-01-30", 395));
}

TEST_F(CudfFilterProjectTest, startswith) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc", "zabc", std::nullopt, "ab", "", "xyz", "abz"});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"startswith(c0, 'ab') AS c1"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>(
          {true, false, std::nullopt, true, false, false, true}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, startswithNullPattern) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc", "zabc", std::nullopt, "ab", "", "xyz", "abz"});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"startswith(c0, cast(null as varchar)) AS c1"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>({
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
      }),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, startswithEmptyPattern) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc", "zabc", std::nullopt, "ab", "", "xyz", "abz"});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"startswith(c0, '') AS c1"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>(
          {true, true, std::nullopt, true, true, true, true}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, startswithColumnPattern) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc", "zabc", std::nullopt, "ab", "", "xyz", "abz"});
  auto pattern = makeNullableFlatVector<std::string>(
      {"ab", "ab", "ab", std::nullopt, "", "yz", "zz"});
  auto data = makeRowVector({input, pattern});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"startswith(c0, c1) AS c2"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>(
          {true, false, std::nullopt, std::nullopt, true, false, false}),
  });
  auto result = AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfFilterProjectTest, startswithConstantInput) {
  auto pattern =
      makeNullableFlatVector<std::string>({"a", "ab", std::nullopt, "", "abc"});
  auto data = makeRowVector({pattern});
  auto typed = test_utils::parseAndInferTypedExpr(
      "startswith('ab', c0)", data->rowType(), &execCtx_, options_);
  exec::ExprSet exprSet({typed}, &execCtx_, /*enableConstantFolding*/ false);

  auto result = evaluate(exprSet, data);
  auto expected = makeNullableFlatVector<bool>(
      {true, true, std::nullopt, true, false});
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfFilterProjectTest, startswithColumnPatternNullInput) {
  auto input = makeNullableFlatVector<std::string>({std::nullopt});
  auto pattern = makeNullableFlatVector<std::string>({"ab"});
  auto data = makeRowVector({input, pattern});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"startswith(c0, c1) AS c2"})
                  .planNode();

  auto expected =
      makeRowVector({makeNullableFlatVector<bool>({std::nullopt})});
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, startswithColumnPatternNullPattern) {
  auto input = makeNullableFlatVector<std::string>({"ab"});
  auto pattern = makeNullableFlatVector<std::string>({std::nullopt});
  auto data = makeRowVector({input, pattern});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"startswith(c0, c1) AS c2"})
                  .planNode();

  auto expected =
      makeRowVector({makeNullableFlatVector<bool>({std::nullopt})});
  AssertQueryBuilder(plan).assertResults(expected);
}

} // namespace
} // namespace facebook::velox::cudf_velox
