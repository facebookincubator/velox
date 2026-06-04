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
#include "velox/experimental/cudf/expression/SparkFunctions.h"
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
    cudf_velox::registerCudf();
    cudf_velox::registerSparkFunctions("");
  }

  static void TearDownTestCase() {
    cudf_velox::unregisterFunctions();
    cudf_velox::unregisterCudf();
  }

  CudfFilterProjectTest() {
    options_.parseIntegerAsBigint = false;
  }

  template <typename T>
  ArrayVectorPtr makeGetNumericTestVector(
      const TypePtr& arrayType = ARRAY(CppToType<T>::create())) {
    return makeNullableArrayVector<T>(
        {
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)}},
            {{static_cast<T>(4), std::nullopt, static_cast<T>(6)}},
            std::nullopt,
            {{static_cast<T>(7), static_cast<T>(8)}},
        },
        arrayType);
  }

  void assertGetElementTypeMatchesCpu(
      const std::string& label,
      const ArrayVectorPtr& arrays) {
    SCOPED_TRACE(label);

    auto assertMatchesCpu = [&](const std::string& expression,
                                const RowVectorPtr& input) {
      SCOPED_TRACE(expression);
      assertExpressionMatchesCpu(expression, input, input->rowType());
    };

    // Spark get is zero-based and supports all integral index widths. Negative
    // and out-of-bounds indexes return null instead of throwing.
    for (const auto& indices : std::vector<VectorPtr>{
             makeFlatVector<int8_t>({1, -1, 0, 1}),
             makeFlatVector<int16_t>({2, 1, 0, 0}),
             makeFlatVector<int32_t>({0, 1, 0, 1}),
             makeFlatVector<int64_t>({2, -1, 0, 0})}) {
      assertMatchesCpu("get(c0, c1)", makeRowVector({arrays, indices}));
    }

    // Literal indexes use a scalar extraction path distinct from variable
    // index columns, including null and invalid literal indexes.
    auto input = makeRowVector({arrays});
    for (const auto& expression : {
             "get(c0, cast(1 as tinyint))",
             "get(c0, cast(1 as smallint))",
             "get(c0, cast(1 as integer))",
             "get(c0, cast(1 as bigint))",
             "get(c0, cast(null as tinyint))",
             "get(c0, cast(null as smallint))",
             "get(c0, cast(null as integer))",
             "get(c0, cast(null as bigint))",
             "get(c0, cast(-1 as tinyint))",
             "get(c0, cast(-1 as smallint))",
             "get(c0, cast(-1 as integer))",
             "get(c0, cast(-1 as bigint))",
             "get(c0, cast(3 as tinyint))",
             "get(c0, cast(3 as smallint))",
             "get(c0, cast(3 as integer))",
             "get(c0, cast(3 as bigint))",
         }) {
      assertMatchesCpu(expression, input);
    }
  }

  void assertConstantArrayGetMatchesCpu(
      const std::string& label,
      const std::string& arraySql) {
    SCOPED_TRACE(label);

    // Constant arrays are materialized by the cuDF adapter because literals are
    // not supplied as input columns during expression evaluation.
    for (const auto& indices : std::vector<VectorPtr>{
             makeFlatVector<int8_t>({0, 1, 2, -1}),
             makeFlatVector<int16_t>({0, 1, 2, 3}),
             makeFlatVector<int32_t>({0, 1, 2, -1}),
             makeFlatVector<int64_t>({0, 1, 2, 3})}) {
      auto input = makeRowVector({indices});
      auto expression = "get(" + arraySql + ", c0)";
      SCOPED_TRACE(expression);
      assertExpressionMatchesCpu(expression, input, input->rowType());
    }
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

TEST_F(CudfFilterProjectTest, getConstantIndex) {
  auto arrays = makeNullableArrayVector<int32_t>({
      {{10, 20, 30}},
      {{4, 5}},
      std::nullopt,
      {{7, std::nullopt, 9}},
  });
  auto input = makeRowVector({arrays});

  assertExpressionMatchesCpu("get(c0, 1)", input, input->rowType());
  assertExpressionMatchesCpu(
      "get(c0, cast(1 as bigint))", input, input->rowType());
}

TEST_F(CudfFilterProjectTest, getNullConstantIndex) {
  auto arrays = makeNullableArrayVector<int32_t>({
      {{10, 20, 30}},
      {{4, 5}},
      std::nullopt,
  });
  auto input = makeRowVector({arrays});

  assertExpressionMatchesCpu(
      "get(c0, cast(null as integer))", input, input->rowType());
  assertExpressionMatchesCpu(
      "get(c0, cast(null as bigint))", input, input->rowType());
}

TEST_F(CudfFilterProjectTest, getReturnsNullForInvalidIndex) {
  auto arrays = makeArrayVector<int32_t>({{1, 2, 3}});
  auto input = makeRowVector({arrays});

  assertExpressionMatchesCpu("get(c0, -1)", input, input->rowType());
  assertExpressionMatchesCpu("get(c0, 3)", input, input->rowType());
}

TEST_F(CudfFilterProjectTest, getSupportsIntegralIndexTypes) {
  auto arrays = makeNullableArrayVector<int32_t>({
      {{1, 2, 3}},
      {{4, std::nullopt, 6}},
      std::nullopt,
      {{7, 8}},
  });

  auto tinyintIndices = makeFlatVector<int8_t>({1, -1, 0, 1});
  auto tinyintInput = makeRowVector({arrays, tinyintIndices});
  assertExpressionMatchesCpu(
      "get(c0, c1)", tinyintInput, tinyintInput->rowType());

  auto smallintIndices = makeFlatVector<int16_t>({2, 1, 0, 0});
  auto smallintInput = makeRowVector({arrays, smallintIndices});
  assertExpressionMatchesCpu(
      "get(c0, c1)", smallintInput, smallintInput->rowType());

  auto integerIndices = makeFlatVector<int32_t>({0, 1, 0, 1});
  auto integerInput = makeRowVector({arrays, integerIndices});
  assertExpressionMatchesCpu(
      "get(c0, c1)", integerInput, integerInput->rowType());

  auto bigintIndices = makeFlatVector<int64_t>({2, -1, 0, 0});
  auto bigintInput = makeRowVector({arrays, bigintIndices});
  assertExpressionMatchesCpu(
      "get(c0, c1)", bigintInput, bigintInput->rowType());
}

TEST_F(CudfFilterProjectTest, getSupportedScalarElementTypes) {
  // Cover all scalar ARRAY element types supported by Velox-cuDF conversion.
  assertGetElementTypeMatchesCpu("tinyint", makeGetNumericTestVector<int8_t>());
  assertGetElementTypeMatchesCpu(
      "smallint", makeGetNumericTestVector<int16_t>());
  assertGetElementTypeMatchesCpu(
      "integer", makeGetNumericTestVector<int32_t>());
  assertGetElementTypeMatchesCpu("bigint", makeGetNumericTestVector<int64_t>());
  assertGetElementTypeMatchesCpu("real", makeGetNumericTestVector<float>());
  assertGetElementTypeMatchesCpu("double", makeGetNumericTestVector<double>());
  assertGetElementTypeMatchesCpu(
      "date", makeGetNumericTestVector<int32_t>(ARRAY(DATE())));
  assertGetElementTypeMatchesCpu(
      "short decimal",
      makeGetNumericTestVector<int64_t>(ARRAY(DECIMAL(10, 2))));
  assertGetElementTypeMatchesCpu(
      "long decimal",
      makeGetNumericTestVector<int128_t>(ARRAY(DECIMAL(20, 4))));

  auto booleanArrays = makeNullableArrayVector<bool>({
      {{true, false, true}},
      {{false, std::nullopt, true}},
      std::nullopt,
      {{true, true}},
  });
  assertGetElementTypeMatchesCpu("boolean", booleanArrays);

  auto varcharArrays = makeNullableArrayVector<std::string>({
      {{"alpha", "beta", "gamma"}},
      {{"delta", std::nullopt, "zeta"}},
      std::nullopt,
      {{"eta", "theta"}},
  });
  assertGetElementTypeMatchesCpu("varchar", varcharArrays);

  auto varbinaryArrays = makeNullableArrayVector<std::string>(
      {
          {{"alpha", "beta", "gamma"}},
          {{"delta", std::nullopt, "zeta"}},
          std::nullopt,
          {{"eta", "theta"}},
      },
      ARRAY(VARBINARY()));
  assertGetElementTypeMatchesCpu("varbinary", varbinaryArrays);

  auto timestampArrays = makeNullableArrayVector<Timestamp>({
      {{Timestamp(1, 0), Timestamp(2, 0), Timestamp(3, 0)}},
      {{Timestamp(4, 0), std::nullopt, Timestamp(6, 0)}},
      std::nullopt,
      {{Timestamp(7, 0), Timestamp(8, 0)}},
  });
  assertGetElementTypeMatchesCpu("timestamp", timestampArrays);
}

TEST_F(CudfFilterProjectTest, getSupportedComplexElementTypes) {
  // Cover complex ARRAY element types that cuDF list extraction can return.
  using OptionalIntArray = std::optional<std::vector<std::optional<int32_t>>>;
  using OptionalNestedArray = std::optional<std::vector<OptionalIntArray>>;

  std::vector<OptionalNestedArray> nestedArrayData = {
      std::vector<OptionalIntArray>{
          std::vector<std::optional<int32_t>>{1, 2},
          std::vector<std::optional<int32_t>>{3},
          std::vector<std::optional<int32_t>>{4, 5},
      },
      std::vector<OptionalIntArray>{
          std::vector<std::optional<int32_t>>{6},
          std::vector<std::optional<int32_t>>{7, 8},
          std::vector<std::optional<int32_t>>{9},
      },
      std::vector<OptionalIntArray>{
          std::vector<std::optional<int32_t>>{10},
          std::vector<std::optional<int32_t>>{11, 12},
      },
      std::vector<OptionalIntArray>{
          std::vector<std::optional<int32_t>>{13},
          std::vector<std::optional<int32_t>>{14, 15},
      },
  };
  assertGetElementTypeMatchesCpu(
      "array", makeNullableNestedArrayVector<int32_t>(nestedArrayData));

  auto rowArrays = makeArrayOfRowVector(
      ROW({"a", "b"}, {INTEGER(), VARCHAR()}),
      {
          {variant::row({1, "alpha"}),
           variant::row({2, "beta"}),
           variant::row({3, "gamma"})},
          {variant::row({4, "delta"}),
           variant::row({5, "epsilon"}),
           variant::row({6, "zeta"})},
          {variant::row({7, "eta"}), variant::row({8, "theta"})},
          {variant::row({9, "iota"}), variant::row({10, "kappa"})},
      });
  assertGetElementTypeMatchesCpu("row", rowArrays);
}

TEST_F(CudfFilterProjectTest, getConstantArraySupportedTypes) {
  // Constant array literals take a different input-column path from array
  // columns, so repeat the supported element-type matrix here.
  assertConstantArrayGetMatchesCpu(
      "tinyint",
      "array_constructor("
      "cast(1 as tinyint), cast(2 as tinyint), cast(3 as tinyint))");
  assertConstantArrayGetMatchesCpu(
      "smallint",
      "array_constructor("
      "cast(1 as smallint), cast(2 as smallint), cast(3 as smallint))");
  assertConstantArrayGetMatchesCpu(
      "integer",
      "array_constructor("
      "cast(1 as integer), cast(2 as integer), cast(3 as integer))");
  assertConstantArrayGetMatchesCpu(
      "bigint",
      "array_constructor("
      "cast(1 as bigint), cast(2 as bigint), cast(3 as bigint))");
  assertConstantArrayGetMatchesCpu(
      "real",
      "array_constructor("
      "cast(1.25 as real), cast(2.5 as real), cast(3.75 as real))");
  assertConstantArrayGetMatchesCpu(
      "double",
      "array_constructor("
      "cast(1.25 as double), cast(2.5 as double), cast(3.75 as double))");
  assertConstantArrayGetMatchesCpu(
      "boolean", "array_constructor(true, false, true)");
  assertConstantArrayGetMatchesCpu(
      "varchar", "array_constructor('alpha', 'beta', 'gamma')");
  assertConstantArrayGetMatchesCpu(
      "varbinary",
      "array_constructor("
      "cast('alpha' as varbinary), "
      "cast('beta' as varbinary), "
      "cast('gamma' as varbinary))");
  assertConstantArrayGetMatchesCpu(
      "date",
      "array_constructor("
      "DATE '2020-01-01', DATE '2020-01-02', DATE '2020-01-03')");
  assertConstantArrayGetMatchesCpu(
      "timestamp",
      "array_constructor("
      "cast('2020-01-01 00:00:00' as timestamp), "
      "cast('2020-01-02 00:00:00' as timestamp), "
      "cast('2020-01-03 00:00:00' as timestamp))");
  assertConstantArrayGetMatchesCpu(
      "short decimal",
      "array_constructor("
      "cast('1.00' as decimal(10, 2)), "
      "cast('2.00' as decimal(10, 2)), "
      "cast('3.00' as decimal(10, 2)))");
  assertConstantArrayGetMatchesCpu(
      "long decimal",
      "array_constructor("
      "cast('1.0000' as decimal(20, 4)), "
      "cast('2.0000' as decimal(20, 4)), "
      "cast('3.0000' as decimal(20, 4)))");
  assertConstantArrayGetMatchesCpu(
      "array",
      "array_constructor("
      "array_constructor(1, 2), "
      "array_constructor(3), "
      "array_constructor(4, 5))");
  assertConstantArrayGetMatchesCpu(
      "row",
      "array_constructor("
      "row_constructor(1, 'alpha'), "
      "row_constructor(2, 'beta'), "
      "row_constructor(3, 'gamma'))");
}

TEST_F(CudfFilterProjectTest, likeWithEscape) {
  auto input = makeNullableFlatVector<std::string>(
      {"a_c", "abc", "abc_", "a%c", std::nullopt, ""});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"like(c0, '%#_%', '#') AS c1"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>(
          {true, false, true, false, std::nullopt, false}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, likeConstantPattern) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc", "zabc", std::nullopt, "ab", "", "a_c", "a%c"});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"like(c0, 'a%') AS c1"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>(
          {true, false, std::nullopt, true, false, true, true}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, likeNullPattern) {
  auto input = makeNullableFlatVector<std::string>(
      {"a_c", "abc", "abc_", "a%c", std::nullopt, ""});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"like(c0, cast(null as varchar)) AS c1"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>({
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

TEST_F(CudfFilterProjectTest, likeNullEscape) {
  auto input = makeNullableFlatVector<std::string>(
      {"a_c", "abc", "abc_", "a%c", std::nullopt, ""});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"like(c0, '%#_%', cast(null as varchar)) AS c1"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>({
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

TEST_F(CudfFilterProjectTest, likeEmptyEscape) {
  auto input = makeNullableFlatVector<std::string>({"abc", "xyz"});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"like(c0, 'abc', '') AS c1"})
                  .planNode();

  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Escape string must be a single character");
}

TEST_F(CudfFilterProjectTest, likeMultiCharacterEscape) {
  auto input = makeNullableFlatVector<std::string>({"abc", "xyz"});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"like(c0, 'abc', 'oops') AS c1"})
                  .planNode();

  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Escape string must be a single character");
}

TEST_F(CudfFilterProjectTest, likeInvalidEscapeUsage) {
  auto input = makeNullableFlatVector<std::string>({"test", "testo"});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"like(c0, 'test_o', 'o') AS c1"})
                  .planNode();

  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Escape character must be followed by '%', '_' or the escape character itself");
}

TEST_F(CudfFilterProjectTest, tryLikeInvalidEscapeUsage) {
  auto input =
      makeNullableFlatVector<std::string>({"test", "testo", std::nullopt});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"try(like(c0, 'test_o', 'o')) AS c1"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>({std::nullopt, std::nullopt, std::nullopt}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, likeColumnPatternInvalidEscapeUsage) {
  auto input = makeNullableFlatVector<std::string>({"abc"});
  auto pattern = makeNullableFlatVector<std::string>({"a#b"});
  auto data = makeRowVector({input, pattern});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"like(c0, c1, '#') AS c2"})
                  .planNode();

  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Escape character must be followed by '%', '_' or the escape character itself");
}

TEST_F(CudfFilterProjectTest, likeColumnPattern) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc", "zabc", std::nullopt, "ab", "", "a_c", "a%c"});
  auto pattern = makeNullableFlatVector<std::string>(
      {"a%", "%bc", "a%", std::nullopt, "", "a_d", "a%%"});
  auto data = makeRowVector({input, pattern});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"like(c0, c1) AS c2"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>(
          {true, true, std::nullopt, std::nullopt, true, false, true}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, likeColumnPatternWithoutPatternNulls) {
  auto input =
      makeNullableFlatVector<std::string>({"abc", std::nullopt, "a_c", ""});
  auto pattern = makeNullableFlatVector<std::string>({"a%", "%", "a_d", ""});
  auto data = makeRowVector({input, pattern});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"like(c0, c1) AS c2"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>({true, std::nullopt, false, true}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, likeColumnPatternWithEscape) {
  auto input = makeNullableFlatVector<std::string>(
      {"a_c", "abc", "abc_", "a%c", std::nullopt, "a#c"});
  auto pattern = makeNullableFlatVector<std::string>(
      {"%#_%", "%#_%", std::nullopt, "a#%%", "%#_%", "a##c"});
  auto data = makeRowVector({input, pattern});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"like(c0, c1, '#') AS c2"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>(
          {true, false, std::nullopt, true, std::nullopt, true}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, likeColumnPatternWithEscapeAllNullPatterns) {
  auto input =
      makeNullableFlatVector<std::string>({"a_c", "abc", std::nullopt, ""});
  auto pattern = makeNullableFlatVector<std::string>(
      {std::nullopt, std::nullopt, std::nullopt, std::nullopt});
  auto data = makeRowVector({input, pattern});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"like(c0, c1, '#') AS c2"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>(
          {std::nullopt, std::nullopt, std::nullopt, std::nullopt}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, likeConstantInputColumnPattern) {
  auto pattern = makeNullableFlatVector<std::string>(
      {"a%", "%b%", "", std::nullopt, "abc"});
  auto data = makeRowVector({pattern});
  auto typed = test_utils::parseAndInferTypedExpr(
      "like('abc', c0)", data->rowType(), &execCtx_, options_);
  exec::ExprSet exprSet({typed}, &execCtx_, /*enableConstantFolding*/ false);

  auto result = evaluate(exprSet, data);
  auto expected =
      makeNullableFlatVector<bool>({true, true, false, std::nullopt, true});
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfFilterProjectTest, likeConstantNullInputColumnPattern) {
  auto pattern = makeNullableFlatVector<std::string>(
      {"a%", "%b%", std::nullopt, "", "abc"});
  auto data = makeRowVector({pattern});
  auto typed = test_utils::parseAndInferTypedExpr(
      "like(cast(null as varchar), c0)", data->rowType(), &execCtx_, options_);
  exec::ExprSet exprSet({typed}, &execCtx_, /*enableConstantFolding*/ false);

  auto result = evaluate(exprSet, data);
  auto expected = makeNullableFlatVector<bool>(
      {std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfFilterProjectTest, likeConstantInputColumnPatternWithEscape) {
  auto pattern = makeNullableFlatVector<std::string>(
      {"%#_%", "a#_c", "a##c", std::nullopt, "a%"});
  auto data = makeRowVector({pattern});
  auto typed = test_utils::parseAndInferTypedExpr(
      "like('a_c', c0, '#')", data->rowType(), &execCtx_, options_);
  exec::ExprSet exprSet({typed}, &execCtx_, /*enableConstantFolding*/ false);

  auto result = evaluate(exprSet, data);
  auto expected =
      makeNullableFlatVector<bool>({true, true, false, std::nullopt, true});
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfFilterProjectTest, likeConstantNullInputColumnPatternWithEscape) {
  auto pattern = makeNullableFlatVector<std::string>(
      {"%#_%", "a#_c", "a##c", std::nullopt, "a%"});
  auto data = makeRowVector({pattern});
  auto typed = test_utils::parseAndInferTypedExpr(
      "like(cast(null as varchar), c0, '#')",
      data->rowType(),
      &execCtx_,
      options_);
  exec::ExprSet exprSet({typed}, &execCtx_, /*enableConstantFolding*/ false);

  auto result = evaluate(exprSet, data);
  auto expected = makeNullableFlatVector<bool>(
      {std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
  facebook::velox::test::assertEqualVectors(expected, result);
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
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, startswithConstantInput) {
  auto pattern =
      makeNullableFlatVector<std::string>({"a", "ab", std::nullopt, "", "abc"});
  auto data = makeRowVector({pattern});
  auto typed = test_utils::parseAndInferTypedExpr(
      "startswith('ab', c0)", data->rowType(), &execCtx_, options_);
  exec::ExprSet exprSet({typed}, &execCtx_, /*enableConstantFolding*/ false);

  auto result = evaluate(exprSet, data);
  auto expected =
      makeNullableFlatVector<bool>({true, true, std::nullopt, true, false});
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfFilterProjectTest, startswithConstantNullInput) {
  auto pattern =
      makeNullableFlatVector<std::string>({"a", "ab", std::nullopt, "", "abc"});
  auto data = makeRowVector({pattern});
  auto typed = test_utils::parseAndInferTypedExpr(
      "startswith(cast(null as varchar), c0)",
      data->rowType(),
      &execCtx_,
      options_);
  exec::ExprSet exprSet({typed}, &execCtx_, /*enableConstantFolding*/ false);

  auto result = evaluate(exprSet, data);
  auto expected = makeNullableFlatVector<bool>(
      {std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
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

  auto expected = makeRowVector({makeNullableFlatVector<bool>({std::nullopt})});
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

  auto expected = makeRowVector({makeNullableFlatVector<bool>({std::nullopt})});
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, contains) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc", "zabc", std::nullopt, "ab", "", "xyz", "abz"});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"contains(c0, 'ab') AS c1"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>(
          {true, true, std::nullopt, true, false, false, true}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, containsNullPattern) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc", "zabc", std::nullopt, "ab", "", "xyz", "abz"});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"contains(c0, cast(null as varchar)) AS c1"})
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

TEST_F(CudfFilterProjectTest, containsEmptyPattern) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc", "zabc", std::nullopt, "ab", "", "xyz", "abz"});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"contains(c0, '') AS c1"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>(
          {true, true, std::nullopt, true, true, true, true}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, containsColumnPattern) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc", "zabc", std::nullopt, "ab", "", "xyz", "abz"});
  auto pattern = makeNullableFlatVector<std::string>(
      {"bc", "ab", "bc", std::nullopt, "", "yz", "zz"});
  auto data = makeRowVector({input, pattern});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"contains(c0, c1) AS c2"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>(
          {true, true, std::nullopt, std::nullopt, true, true, false}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, containsConstantInput) {
  auto pattern =
      makeNullableFlatVector<std::string>({"b", "ab", std::nullopt, "", "abc"});
  auto data = makeRowVector({pattern});
  auto typed = test_utils::parseAndInferTypedExpr(
      "contains('ab', c0)", data->rowType(), &execCtx_, options_);
  exec::ExprSet exprSet({typed}, &execCtx_, /*enableConstantFolding*/ false);

  auto result = evaluate(exprSet, data);
  auto expected =
      makeNullableFlatVector<bool>({true, true, std::nullopt, true, false});
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfFilterProjectTest, containsConstantNullInput) {
  auto pattern =
      makeNullableFlatVector<std::string>({"b", "ab", std::nullopt, "", "abc"});
  auto data = makeRowVector({pattern});
  auto typed = test_utils::parseAndInferTypedExpr(
      "contains(cast(null as varchar), c0)",
      data->rowType(),
      &execCtx_,
      options_);
  exec::ExprSet exprSet({typed}, &execCtx_, /*enableConstantFolding*/ false);

  auto result = evaluate(exprSet, data);
  auto expected = makeNullableFlatVector<bool>(
      {std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfFilterProjectTest, containsColumnPatternNullInput) {
  auto input = makeNullableFlatVector<std::string>({std::nullopt});
  auto pattern = makeNullableFlatVector<std::string>({"ab"});
  auto data = makeRowVector({input, pattern});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"contains(c0, c1) AS c2"})
                  .planNode();

  auto expected = makeRowVector({makeNullableFlatVector<bool>({std::nullopt})});
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, containsColumnPatternNullPattern) {
  auto input = makeNullableFlatVector<std::string>({"ab"});
  auto pattern = makeNullableFlatVector<std::string>({std::nullopt});
  auto data = makeRowVector({input, pattern});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"contains(c0, c1) AS c2"})
                  .planNode();

  auto expected = makeRowVector({makeNullableFlatVector<bool>({std::nullopt})});
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, endswith) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc", "zabc", std::nullopt, "ab", "", "xyz", "abz"});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"endswith(c0, 'ab') AS c1"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>(
          {false, false, std::nullopt, true, false, false, false}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, endswithNullPattern) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc", "zabc", std::nullopt, "ab", "", "xyz", "abz"});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"endswith(c0, cast(null as varchar)) AS c1"})
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

TEST_F(CudfFilterProjectTest, endswithEmptyPattern) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc", "zabc", std::nullopt, "ab", "", "xyz", "abz"});
  auto data = makeRowVector({input});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"endswith(c0, '') AS c1"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>(
          {true, true, std::nullopt, true, true, true, true}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, endswithColumnPattern) {
  auto input = makeNullableFlatVector<std::string>(
      {"abc", "zabc", std::nullopt, "ab", "", "xyz", "abz"});
  auto pattern = makeNullableFlatVector<std::string>(
      {"bc", "bc", "bc", std::nullopt, "", "yz", "zz"});
  auto data = makeRowVector({input, pattern});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"endswith(c0, c1) AS c2"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<bool>(
          {true, true, std::nullopt, std::nullopt, true, true, false}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, endswithConstantInput) {
  auto pattern =
      makeNullableFlatVector<std::string>({"b", "ab", std::nullopt, "", "abc"});
  auto data = makeRowVector({pattern});
  auto typed = test_utils::parseAndInferTypedExpr(
      "endswith('ab', c0)", data->rowType(), &execCtx_, options_);
  exec::ExprSet exprSet({typed}, &execCtx_, /*enableConstantFolding*/ false);

  auto result = evaluate(exprSet, data);
  auto expected =
      makeNullableFlatVector<bool>({true, true, std::nullopt, true, false});
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfFilterProjectTest, endswithConstantNullInput) {
  auto pattern =
      makeNullableFlatVector<std::string>({"b", "ab", std::nullopt, "", "abc"});
  auto data = makeRowVector({pattern});
  auto typed = test_utils::parseAndInferTypedExpr(
      "endswith(cast(null as varchar), c0)",
      data->rowType(),
      &execCtx_,
      options_);
  exec::ExprSet exprSet({typed}, &execCtx_, /*enableConstantFolding*/ false);

  auto result = evaluate(exprSet, data);
  auto expected = makeNullableFlatVector<bool>(
      {std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfFilterProjectTest, endswithColumnPatternNullInput) {
  auto input = makeNullableFlatVector<std::string>({std::nullopt});
  auto pattern = makeNullableFlatVector<std::string>({"ab"});
  auto data = makeRowVector({input, pattern});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"endswith(c0, c1) AS c2"})
                  .planNode();

  auto expected = makeRowVector({makeNullableFlatVector<bool>({std::nullopt})});
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(CudfFilterProjectTest, endswithColumnPatternNullPattern) {
  auto input = makeNullableFlatVector<std::string>({"ab"});
  auto pattern = makeNullableFlatVector<std::string>({std::nullopt});
  auto data = makeRowVector({input, pattern});

  auto plan = PlanBuilder()
                  .setParseOptions(options_)
                  .values({data})
                  .project({"endswith(c0, c1) AS c2"})
                  .planNode();

  auto expected = makeRowVector({makeNullableFlatVector<bool>({std::nullopt})});
  AssertQueryBuilder(plan).assertResults(expected);
}

// Test unary math functions for Spark
TEST_F(CudfFilterProjectTest, unaryMathFunctions) {
  auto testUnaryFunction =
      [&](std::string expr, double input, double expected) {
        auto valueOpt = evaluateOnce<double, double>(expr, input);
        auto value = valueOpt.value();
        EXPECT_DOUBLE_EQ(value, expected);
      };

  auto testUnaryFunctionInt =
      [&](std::string expr, double input, int64_t expected) {
        auto valueOpt = evaluateOnce<int64_t, double>(expr, input);
        auto value = valueOpt.value();
        EXPECT_EQ(value, expected);
      };

  // Inverse trigonometric functions
  testUnaryFunction("asin(c0)", 0.0, 0.0);
  testUnaryFunction("acos(c0)", 1.0, 0.0);
  testUnaryFunction("atan(c0)", 0.0, 0.0);

  // Hyperbolic functions (Spark-specific)
  testUnaryFunction("sinh(c0)", 0.0, 0.0);
  testUnaryFunction("cosh(c0)", 0.0, 1.0);
  testUnaryFunction("asinh(c0)", 0.0, 0.0);
  testUnaryFunction("acosh(c0)", 1.0, 0.0);
  testUnaryFunction("atanh(c0)", 0.0, 0.0);

  // Exponential and root functions
  testUnaryFunction("exp(c0)", 0.0, 1.0);
  testUnaryFunction("sqrt(c0)", 4.0, 2.0);
  testUnaryFunction("cbrt(c0)", 8.0, 2.0);

  // Rounding functions (Spark-specific)
  testUnaryFunctionInt("ceil(c0)", 3.2, 4);
  testUnaryFunctionInt("floor(c0)", 3.8, 3);
  testUnaryFunction("rint(c0)", 3.5, 4.0);

  // Absolute value
  testUnaryFunction("abs(c0)", -5.5, 5.5);
}

} // namespace
} // namespace facebook::velox::cudf_velox
