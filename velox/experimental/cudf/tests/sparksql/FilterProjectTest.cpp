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

// Test unary math functions for Spark
TEST_F(CudfFilterProjectTest, unaryMathFunctions) {
  // Trigonometric functions
  auto sinValue = evaluateOnce<double, double>("sin(c0)", 0.0);
  EXPECT_DOUBLE_EQ(sinValue.value(), 0.0);
  
  auto cosValue = evaluateOnce<double, double>("cos(c0)", 0.0);
  EXPECT_DOUBLE_EQ(cosValue.value(), 1.0);
  
  auto tanValue = evaluateOnce<double, double>("tan(c0)", 0.0);
  EXPECT_DOUBLE_EQ(tanValue.value(), 0.0);
  
  // Inverse trigonometric functions
  auto asinValue = evaluateOnce<double, double>("asin(c0)", 0.0);
  EXPECT_DOUBLE_EQ(asinValue.value(), 0.0);
  
  auto acosValue = evaluateOnce<double, double>("acos(c0)", 1.0);
  EXPECT_DOUBLE_EQ(acosValue.value(), 0.0);
  
  auto atanValue = evaluateOnce<double, double>("atan(c0)", 0.0);
  EXPECT_DOUBLE_EQ(atanValue.value(), 0.0);
  
  // Hyperbolic functions (Spark-specific)
  auto sinhValue = evaluateOnce<double, double>("sinh(c0)", 0.0);
  EXPECT_DOUBLE_EQ(sinhValue.value(), 0.0);
  
  auto coshValue = evaluateOnce<double, double>("cosh(c0)", 0.0);
  EXPECT_DOUBLE_EQ(coshValue.value(), 1.0);
  
  auto asinhValue = evaluateOnce<double, double>("asinh(c0)", 0.0);
  EXPECT_DOUBLE_EQ(asinhValue.value(), 0.0);
  
  auto acoshValue = evaluateOnce<double, double>("acosh(c0)", 1.0);
  EXPECT_DOUBLE_EQ(acoshValue.value(), 0.0);
  
  auto atanhValue = evaluateOnce<double, double>("atanh(c0)", 0.0);
  EXPECT_DOUBLE_EQ(atanhValue.value(), 0.0);
  
  // Exponential and logarithmic functions
  auto expValue = evaluateOnce<double, double>("exp(c0)", 0.0);
  EXPECT_DOUBLE_EQ(expValue.value(), 1.0);
  
  auto logValue = evaluateOnce<double, double>("log(c0)", 1.0);
  EXPECT_DOUBLE_EQ(logValue.value(), 0.0);
  
  auto sqrtValue = evaluateOnce<double, double>("sqrt(c0)", 4.0);
  EXPECT_DOUBLE_EQ(sqrtValue.value(), 2.0);
  
  auto cbrtValue = evaluateOnce<double, double>("cbrt(c0)", 8.0);
  EXPECT_DOUBLE_EQ(cbrtValue.value(), 2.0);
  
  // Rounding functions (Spark-specific)
  auto ceilValue = evaluateOnce<int64_t, double>("ceil(c0)", 3.2);
  EXPECT_EQ(ceilValue.value(), 4);
  
  auto floorValue = evaluateOnce<int64_t, double>("floor(c0)", 3.8);
  EXPECT_EQ(floorValue.value(), 3);
  
  auto rintValue = evaluateOnce<double, double>("rint(c0)", 3.5);
  EXPECT_DOUBLE_EQ(rintValue.value(), 4.0);
  
  // Absolute value
  auto absValue = evaluateOnce<double, double>("abs(c0)", -5.5);
  EXPECT_DOUBLE_EQ(absValue.value(), 5.5);
}
}

} // namespace
} // namespace facebook::velox::cudf_velox
