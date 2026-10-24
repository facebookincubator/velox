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
#include "velox/experimental/cudf/expression/SparkFunctions.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/type/DecimalUtil.h"

#include <cudf/copying.hpp>

#include <bit>
#include <limits>

namespace facebook::velox::cudf_velox {
namespace {

class LegacyCastTest : public CudfFunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::sparksql::registerFunctions();
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
    registerSparkFunctions("");
  }

  static void TearDownTestCase() {
    unregisterFunctions();
    unregisterCudf();
  }

  core::TypedExprPtr castExpr(const TypePtr& source, const TypePtr& target) {
    return std::make_shared<core::CallTypedExpr>(
        target,
        std::vector<core::TypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(source, "c0")},
        "spark_legacy_cast");
  }

  void assertCastMatchesCpu(const VectorPtr& input, const TypePtr& target) {
    SCOPED_TRACE(input->type()->toString() + " -> " + target->toString());
    auto expr = castExpr(input->type(), target);
    ASSERT_TRUE(canExprRunOnGpu(expr, execCtx_.queryCtx(), pool_.get()));
    auto data = makeRowVector({input});
    exec::ExprSet expressions({expr}, &execCtx_);
    auto expected =
        functions::test::FunctionBaseTest::evaluate(expressions, data);
    auto actual = evaluate(expr, data);
    velox::test::assertEqualVectors(expected, actual);
  }

  void assertCastMatchesSpark(
      const VectorPtr& input,
      const VectorPtr& expected) {
    auto expr = castExpr(input->type(), expected->type());
    SCOPED_TRACE(expr->toString());
    ASSERT_TRUE(canExprRunOnGpu(expr, execCtx_.queryCtx(), pool_.get()));
    velox::test::assertEqualVectors(
        expected, evaluate(expr, makeRowVector({input})));
  }
};

TEST_F(LegacyCastTest, integralToFloatingPoint) {
  auto input = makeNullableFlatVector<int64_t>(
      {0, 1, -1, INT64_MIN, INT64_MAX, std::nullopt});
  assertCastMatchesCpu(input, DOUBLE());
  assertCastMatchesCpu(input, REAL());
}

TEST_F(LegacyCastTest, floatingPointToIntegral) {
  VectorPtr input = makeNullableFlatVector<double>(
      {0.0,
       -0.0,
       1.9,
       -1.9,
       127.0,
       128.0,
       -129.0,
       32'768.0,
       2'147'483'647.0,
       2'147'483'648.0,
       -2'147'483'649.0,
       9'223'372'036'854'775'808.0,
       -9'223'372'036'854'775'808.0,
       std::numeric_limits<double>::infinity(),
       -std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::quiet_NaN(),
       std::nullopt});
  // Expected values follow Spark's Java numeric narrowing, including values
  // equal to the floating-point representation of the integral upper bound.
  for (const auto& expected : std::vector<VectorPtr>{
           makeNullableFlatVector<int8_t>(
               {0,
                0,
                1,
                -1,
                127,
                -128,
                127,
                0,
                -1,
                -1,
                0,
                -1,
                0,
                -1,
                0,
                0,
                std::nullopt}),
           makeNullableFlatVector<int16_t>(
               {0,
                0,
                1,
                -1,
                127,
                128,
                -129,
                -32768,
                -1,
                -1,
                0,
                -1,
                0,
                -1,
                0,
                0,
                std::nullopt}),
           makeNullableFlatVector<int32_t>(
               {0,
                0,
                1,
                -1,
                127,
                128,
                -129,
                32768,
                INT32_MAX,
                INT32_MAX,
                INT32_MIN,
                INT32_MAX,
                INT32_MIN,
                INT32_MAX,
                INT32_MIN,
                0,
                std::nullopt}),
           makeNullableFlatVector<int64_t>(
               {0,
                0,
                1,
                -1,
                127,
                128,
                -129,
                32768,
                2'147'483'647,
                2'147'483'648,
                -2'147'483'649,
                INT64_MAX,
                INT64_MIN,
                INT64_MAX,
                INT64_MIN,
                0,
                std::nullopt})}) {
    assertCastMatchesSpark(input, expected);
  }
  input = makeNullableFlatVector<float>(
      {0,
       -0.0F,
       1.9F,
       -1.9F,
       129.0F,
       -129.0F,
       32'768.0F,
       2'147'483'648.0F,
       -2'147'483'648.0F,
       1e30F,
       -1e30F,
       std::numeric_limits<float>::infinity(),
       -std::numeric_limits<float>::infinity(),
       std::numeric_limits<float>::quiet_NaN(),
       std::nullopt});
  for (const auto& expected : std::vector<VectorPtr>{
           makeNullableFlatVector<int8_t>(
               {0,
                0,
                1,
                -1,
                -127,
                127,
                0,
                -1,
                0,
                -1,
                0,
                -1,
                0,
                0,
                std::nullopt}),
           makeNullableFlatVector<int16_t>(
               {0,
                0,
                1,
                -1,
                129,
                -129,
                -32768,
                -1,
                0,
                -1,
                0,
                -1,
                0,
                0,
                std::nullopt}),
           makeNullableFlatVector<int32_t>(
               {0,
                0,
                1,
                -1,
                129,
                -129,
                32768,
                INT32_MAX,
                INT32_MIN,
                INT32_MAX,
                INT32_MIN,
                INT32_MAX,
                INT32_MIN,
                0,
                std::nullopt}),
           makeNullableFlatVector<int64_t>(
               {0,
                0,
                1,
                -1,
                129,
                -129,
                32768,
                2'147'483'648,
                -2'147'483'648,
                INT64_MAX,
                INT64_MIN,
                INT64_MAX,
                INT64_MIN,
                0,
                std::nullopt})}) {
    assertCastMatchesSpark(input, expected);
  }
}

TEST_F(LegacyCastTest, plainNumericCasts) {
  const std::vector<TypePtr> targets{
      BOOLEAN(), TINYINT(), SMALLINT(), INTEGER(), BIGINT(), REAL(), DOUBLE()};
  const std::vector<VectorPtr> inputs{
      makeNullableFlatVector<bool>({false, true, std::nullopt}),
      makeNullableFlatVector<int8_t>({0, INT8_MIN, INT8_MAX, std::nullopt}),
      makeNullableFlatVector<int16_t>({0, INT16_MIN, INT16_MAX, std::nullopt}),
      makeNullableFlatVector<int32_t>({0, INT32_MIN, INT32_MAX, std::nullopt}),
      makeNullableFlatVector<int64_t>({0, INT64_MIN, INT64_MAX, std::nullopt}),
      makeNullableFlatVector<float>({0, 1.5, -1.5, std::nullopt}),
      makeNullableFlatVector<double>({0, 1.5, -1.5, std::nullopt})};
  for (const auto& input : inputs) {
    for (const auto& target : targets) {
      assertCastMatchesCpu(input, target);
    }
  }
}

TEST_F(LegacyCastTest, integralToDecimal) {
  auto input = makeNullableFlatVector<int64_t>(
      {0, 1, -1, 999, -999, 1'000, INT64_MIN, INT64_MAX, std::nullopt});
  for (const auto& type :
       {DECIMAL(5, 2),
        DECIMAL(15, 4),
        DECIMAL(17, 2),
        DECIMAL(20, 0),
        DECIMAL(38, 19)}) {
    assertCastMatchesCpu(input, type);
    assertCastMatchesCpu(
        makeNullableFlatVector<bool>({true, false, std::nullopt}), type);
    assertCastMatchesCpu(
        makeNullableFlatVector<int8_t>({INT8_MIN, INT8_MAX, std::nullopt}),
        type);
    assertCastMatchesCpu(
        makeNullableFlatVector<int16_t>({INT16_MIN, INT16_MAX, std::nullopt}),
        type);
    assertCastMatchesCpu(
        makeNullableFlatVector<int32_t>({INT32_MIN, INT32_MAX, std::nullopt}),
        type);
  }
}

TEST_F(LegacyCastTest, floatingPointToDecimal) {
  // Exact decimal values from Java 17 BigDecimal.valueOf, as used by Spark
  // 3.5. Input bits keep halfway neighbors and large-integer cases stable.
  const std::vector<std::pair<uint64_t, std::string>> fixtures{
      {0x45308b2a2c280291ULL, "20000000000000000000000000"},
      {0x45308b2a2c280290ULL, "19999999999999998000000000"},
      {0x45308b2a2c280292ULL, "20000000000000006000000000"},
      {0x45308b2a2c28028fULL, "19999999999999993000000000"},
      {0x45308b2a2c280293ULL, "20000000000000010000000000"},
      {0x453408587618e398ULL, "24217927147435764000000000"},
      {0x453408587618e397ULL, "24217927147435760000000000"},
      {0x453408587618e399ULL, "24217927147435770000000000"},
      {0x44b52d02c7e14af6ULL, "99999999999999990000000"},
      {0x44b52d02c7e14af5ULL, "99999999999999970000000"},
      {0x44b52d02c7e14af7ULL, "100000000000000010000000"},
      {0x44b52d02c7e14af4ULL, "99999999999999960000000"},
      {0x44b52d02c7e14af8ULL, "100000000000000030000000"},
      {0x4350000000000000ULL, "18014398509481984"},
      {0x434fffffffffffffULL, "18014398509481982"},
      {0x4350000000000001ULL, "18014398509481988"},
      {0x4360000000000000ULL, "36028797018963968"},
      {0x435fffffffffffffULL, "36028797018963964"},
      {0x4360000000000001ULL, "36028797018963976"},
      {0x4370000000000000ULL, "72057594037927936"},
      {0x436fffffffffffffULL, "72057594037927928"},
      {0x4370000000000001ULL, "72057594037927952"},
      {0x4380000000000000ULL, "144115188075855872"},
      {0x437fffffffffffffULL, "144115188075855856"},
      {0x4380000000000001ULL, "144115188075855904"},
      {0x4390000000000000ULL, "288230376151711740"},
      {0x438fffffffffffffULL, "288230376151711712"},
      {0x4390000000000001ULL, "288230376151711810"},
      {0x43a0000000000000ULL, "576460752303423490"},
      {0x439fffffffffffffULL, "576460752303423420"},
      {0x43a0000000000001ULL, "576460752303423620"},
      {0x43b0000000000000ULL, "1152921504606846980"},
      {0x43afffffffffffffULL, "1152921504606846850"},
      {0x43b0000000000001ULL, "1152921504606847230"},
      {0x43c0000000000000ULL, "2305843009213694000"},
      {0x43bfffffffffffffULL, "2305843009213693700"},
      {0x43c0000000000001ULL, "2305843009213694500"},
      {0x43d0000000000000ULL, "4611686018427387900"},
      {0x43cfffffffffffffULL, "4611686018427387400"},
      {0x43d0000000000001ULL, "4611686018427388900"},
      {0x43e0000000000000ULL, "9223372036854776000"},
      {0x43dfffffffffffffULL, "9223372036854774800"},
      {0x43e0000000000001ULL, "9223372036854778000"},
      {0xc350000000000000ULL, "-18014398509481984"},
      {0xc350000000000001ULL, "-18014398509481988"},
      {0xc34fffffffffffffULL, "-18014398509481982"},
      {0xc3b0000000000000ULL, "-1152921504606846980"},
      {0xc3b0000000000001ULL, "-1152921504606847230"},
      {0xc3afffffffffffffULL, "-1152921504606846850"},
      {0xc3e0000000000000ULL, "-9223372036854776000"},
      {0xc3e0000000000001ULL, "-9223372036854778000"},
      {0xc3dfffffffffffffULL, "-9223372036854774800"},
      {0x3ff3c04ea4a8c155ULL, "1.23445"},
      {0x3ff3c04ea4a8c154ULL, "1.2344499999999998"},
      {0x3ff3c04ea4a8c156ULL, "1.2344500000000003"},
      {0xbff3c04ea4a8c155ULL, "-1.23445"},
      {0xbff3c04ea4a8c156ULL, "-1.2344500000000003"},
      {0xbff3c04ea4a8c154ULL, "-1.2344499999999998"},
      {0x3f0a36e2eb1c432dULL, "0.000050"},
      {0x3f0a36e2eb1c432cULL, "0.000049999999999999996"},
      {0x3f0a36e2eb1c432eULL, "0.00005000000000000001"},
      {0xbf0a36e2eb1c432dULL, "-0.000050"},
      {0xbf0a36e2eb1c432eULL, "-0.00005000000000000001"},
      {0xbf0a36e2eb1c432cULL, "-0.000049999999999999996"},
      {0x42374876e7fffffdULL, "99999999999.99995"},
      {0x42374876e7fffffcULL, "99999999999.99994"},
      {0x42374876e7fffffeULL, "99999999999.99997"},
      {0x3ff3c0ca42d8b4d7ULL, "1.234567891234567"},
      {0x3fbf9add37c1215eULL, "0.12345678912345678"},
      {0xbff3c0ca42d8b4d7ULL, "-1.234567891234567"},
      {0x3fefffffffffffffULL, "0.9999999999999999"},
      {0x43118b54f22aeb03ULL, "1234567890123456.8"},
      {0x0000000000000000ULL, "0.0"},
      {0x8000000000000000ULL, "0.0"},
      {0x380b38fb9daa78e4ULL, "0.000000000000000000000000000000000000010"},
      {0x380b38fb9daa78e3ULL,
       "0.000000000000000000000000000000000000009999999999999998"},
      {0x380b38fb9daa78e5ULL,
       "0.000000000000000000000000000000000000010000000000000001"},
      {0x3bc79ca10c924223ULL, "0.000000000000000000010"},
      {0x3bc79ca10c924222ULL, "0.000000000000000000009999999999999998"},
      {0x3bc79ca10c924224ULL, "0.000000000000000000010000000000000001"},
      {0x47d2ced32a16a1b1ULL, "100000000000000000000000000000000000000"},
      {0x47d2ced32a16a1b0ULL, "99999999999999980000000000000000000000"},
      {0x47d2ced32a16a1b2ULL, "100000000000000020000000000000000000000"},
      {0x3fb99999a0000000ULL, "0.10000000149011612"},
      {0xbfb99999a0000000ULL, "-0.10000000149011612"},
      {0x3fb9999980000000ULL, "0.09999999403953552"},
      {0x3fb99999c0000000ULL, "0.10000000894069672"},
      {0x3ff3c0ca20000000ULL, "1.2345677614212036"},
      {0xbff3c0ca20000000ULL, "-1.2345677614212036"},
      {0x3fefffffe0000000ULL, "0.9999999403953552"},
      {0x41e0000000000000ULL, "2147483648"},
      {0x41dfffffe0000000ULL, "2147483520"},
      {0x47efffffe0000000ULL, "340282346638528860000000000000000000000"},
      {0x3bc79ca100000000ULL, "0.000000000000000000009999999682655225"},
      {0x47d2ced320000000ULL, "99999996802856920000000000000000000000"},
  };
  auto input = makeFlatVector<double>(fixtures.size(), [&](auto row) {
    return std::bit_cast<double>(fixtures[row].first);
  });
  auto reference = makeFlatVector<std::string>(
      fixtures.size(), [&](auto row) { return fixtures[row].second; });

  std::vector<float> realValues;
  std::vector<std::string> realReference;
  for (const auto& [bits, text] : fixtures) {
    const auto value = std::bit_cast<double>(bits);
    if (static_cast<double>(static_cast<float>(value)) == value) {
      realValues.push_back(static_cast<float>(value));
      realReference.push_back(text);
    }
  }

  std::vector<TypePtr> targets{
      DECIMAL(1, 0),
      DECIMAL(5, 2),
      DECIMAL(15, 4),
      DECIMAL(17, 2),
      DECIMAL(18, 18)};
  for (int scale = 0; scale <= 38; ++scale) {
    targets.push_back(DECIMAL(38, scale));
  }

  auto assertReference = [&](const VectorPtr& values, const VectorPtr& texts) {
    for (const auto& target : targets) {
      SCOPED_TRACE(target->toString());
      // Parsing decimal text is exact and does not use CPU floating conversion.
      exec::ExprSet expressions({castExpr(VARCHAR(), target)}, &execCtx_);
      auto expected = functions::test::FunctionBaseTest::evaluate(
          expressions, makeRowVector({texts}));
      assertCastMatchesSpark(values, expected);
    }
  };
  assertReference(input, reference);
  assertReference(
      makeFlatVector<float>(realValues),
      makeFlatVector<std::string>(realReference));
  assertReference(
      makeNullableFlatVector<double>(
          {0.0,
           -0.0,
           std::numeric_limits<double>::denorm_min(),
           std::numeric_limits<double>::max(),
           std::numeric_limits<double>::infinity(),
           -std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::quiet_NaN(),
           std::nullopt}),
      makeNullableFlatVector<std::string>(
          {"0.0",
           "-0.0",
           "0.0",
           "NaN",
           "Infinity",
           "-Infinity",
           "NaN",
           std::nullopt}));
}

TEST_F(LegacyCastTest, decimalRescale) {
  for (const auto& source : {DECIMAL(18, 4), DECIMAL(38, 4), DECIMAL(38, 38)}) {
    const auto maximum =
        DecimalUtil::kPowersOfTen[getDecimalPrecisionScale(*source).first] - 1;
    VectorPtr input;
    if (source->isShortDecimal()) {
      input = makeNullableFlatVector<int64_t>(
          {0,
           1,
           -1,
           12'345,
           -12'345,
           99'995,
           -99'995,
           static_cast<int64_t>(maximum),
           -static_cast<int64_t>(maximum),
           std::nullopt},
          source);
    } else {
      input = makeNullableFlatVector<int128_t>(
          {0,
           1,
           -1,
           12'345,
           -12'345,
           99'995,
           -99'995,
           maximum,
           -maximum,
           std::nullopt},
          source);
    }
    for (const auto& target :
         {DECIMAL(1, 0),
          DECIMAL(5, 3),
          DECIMAL(18, 4),
          DECIMAL(38, 0),
          DECIMAL(38, 19),
          DECIMAL(38, 38)}) {
      assertCastMatchesCpu(input, target);
    }
  }
}

TEST_F(LegacyCastTest, emptyNullAndConstantInputs) {
  for (const auto& target :
       std::vector<TypePtr>{INTEGER(), DECIMAL(15, 4), DECIMAL(38, 19)}) {
    assertCastMatchesCpu(makeFlatVector<double>({}), target);
    assertCastMatchesCpu(
        makeNullableFlatVector<double>({std::nullopt, std::nullopt}), target);
    assertCastMatchesCpu(makeConstant<double>(1.25, 3), target);
  }
}

TEST_F(LegacyCastTest, legacyModeIgnoresSessionAnsi) {
  using functions::sparksql::SparkQueryConfig;
  queryCtx_->testingOverrideConfigUnsafe(
      {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled), "true"}});
  assertCastMatchesCpu(makeFlatVector<double>({1e100, -1e100}), INTEGER());
  assertCastMatchesCpu(makeFlatVector<double>({1e100, -1e100}), DECIMAL(15, 4));
}

TEST_F(LegacyCastTest, unsupportedCasts) {
  for (const auto& types : std::vector<std::pair<TypePtr, TypePtr>>{
           {VARCHAR(), INTEGER()},
           {BIGINT(), TIMESTAMP()},
           {DATE(), INTEGER()},
           {ARRAY(BIGINT()), ARRAY(DOUBLE())}}) {
    EXPECT_FALSE(canExprRunOnGpu(
        castExpr(types.first, types.second), execCtx_.queryCtx(), pool_.get()));
  }
  auto ansi = std::make_shared<core::CallTypedExpr>(
      INTEGER(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "c0")},
      "spark_ansi_cast");
  EXPECT_FALSE(canExprRunOnGpu(ansi, execCtx_.queryCtx(), pool_.get()));

  for (const auto& inputs :
       std::vector<std::vector<core::TypedExprPtr>>{{}, {ansi, ansi}}) {
    auto malformed = std::make_shared<core::CallTypedExpr>(
        INTEGER(), inputs, "spark_legacy_cast");
    EXPECT_FALSE(FunctionExpression::canEvaluate(malformed));
    EXPECT_EQ(
        createCudfFunction("spark_legacy_cast", malformed, pool_.get()),
        nullptr);
  }

  registerSparkFunctions("test_");
  auto prefixed = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")},
      "test_spark_legacy_cast");
  EXPECT_TRUE(FunctionExpression::canEvaluate(prefixed));
  EXPECT_NE(
      createCudfFunction("test_spark_legacy_cast", prefixed, pool_.get()),
      nullptr);
}

TEST_F(LegacyCastTest, constantFloatingDecimalCastFallback) {
  auto constant = std::make_shared<core::ConstantTypedExpr>(
      DOUBLE(), variant(1.2344499999999998));
  auto zero = std::make_shared<core::ConstantTypedExpr>(DOUBLE(), variant(0.0));
  auto foldedInput = std::make_shared<core::CallTypedExpr>(
      DOUBLE(), std::vector<core::TypedExprPtr>{constant, zero}, "add");
  for (const auto& input :
       std::vector<core::TypedExprPtr>{constant, foldedInput}) {
    auto cast = std::make_shared<core::CallTypedExpr>(
        DECIMAL(15, 4),
        std::vector<core::TypedExprPtr>{input},
        "spark_legacy_cast");
    // CPU constant folding uses different floating-to-decimal rounding. These
    // calls must stay off GPU until that optimizer behavior is corrected.
    EXPECT_FALSE(canExprRunOnGpu(cast, execCtx_.queryCtx(), pool_.get()));
    auto comparison = std::make_shared<core::CallTypedExpr>(
        BOOLEAN(),
        std::vector<core::TypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(DECIMAL(15, 4), "c0"),
            cast},
        "decimal_lessthan");
    EXPECT_FALSE(canExprRunOnGpu(comparison, execCtx_.queryCtx(), pool_.get()));
  }
  auto cast = std::make_shared<core::CallTypedExpr>(
      DECIMAL(15, 4),
      std::vector<core::TypedExprPtr>{constant},
      "spark_legacy_cast");
  EXPECT_FALSE(canExprRunOnGpu(cast, nullptr, nullptr));
}

TEST_F(LegacyCastTest, constantFloatingIntegralCastFallback) {
  for (const auto& target :
       std::vector<TypePtr>{TINYINT(), SMALLINT(), INTEGER(), BIGINT()}) {
    for (const auto& source : std::vector<TypePtr>{REAL(), DOUBLE()}) {
      SCOPED_TRACE(source->toString() + " -> " + target->toString());
      const auto upperBound =
          target->isBigint() ? 9'223'372'036'854'775'808.0 : 2'147'483'648.0;
      auto constant = std::make_shared<core::ConstantTypedExpr>(
          source,
          source->isReal() ? variant(static_cast<float>(upperBound))
                           : variant(upperBound));
      auto zero = std::make_shared<core::ConstantTypedExpr>(
          source, source->isReal() ? variant(0.0f) : variant(0.0));
      auto foldedInput = std::make_shared<core::CallTypedExpr>(
          source, std::vector<core::TypedExprPtr>{constant, zero}, "add");
      for (const auto& input :
           std::vector<core::TypedExprPtr>{constant, foldedInput}) {
        auto cast = std::make_shared<core::CallTypedExpr>(
            target,
            std::vector<core::TypedExprPtr>{input},
            "spark_legacy_cast");
        // CPU constant folding does not saturate some integral upper bounds
        // like Spark and the column kernel do.
        EXPECT_FALSE(canExprRunOnGpu(cast, execCtx_.queryCtx(), pool_.get()));
        auto comparison = std::make_shared<core::CallTypedExpr>(
            BOOLEAN(),
            std::vector<core::TypedExprPtr>{
                std::make_shared<core::FieldAccessTypedExpr>(target, "c0"),
                cast},
            "lessthan");
        EXPECT_FALSE(
            canExprRunOnGpu(comparison, execCtx_.queryCtx(), pool_.get()));
      }
      auto cast = std::make_shared<core::CallTypedExpr>(
          target,
          std::vector<core::TypedExprPtr>{constant},
          "spark_legacy_cast");
      EXPECT_FALSE(canExprRunOnGpu(cast, nullptr, nullptr));
    }
  }
}

TEST_F(LegacyCastTest, slicedInput) {
  auto input = makeRowVector({makeFlatVector<double>(
      70,
      [](auto row) { return row + 0.5; },
      [](auto row) { return row % 7 == 0; })});
  auto stream = cudfGlobalStreamPool().get_stream();
  auto resource = get_output_mr();
  auto table = with_arrow::toCudfTable(input, pool_.get(), stream, resource);
  auto slices = cudf::slice(table->view().column(0), {3, 68});
  std::vector<ColumnOrView> columns;
  columns.emplace_back(slices[0]);
  auto expr = castExpr(DOUBLE(), INTEGER());
  auto function = createCudfFunction("spark_legacy_cast", expr, pool_.get());
  ASSERT_NE(function, nullptr);
  auto result = function->eval(columns, stream, resource);
  auto outputType = ROW("c0", INTEGER());
  auto actual = with_arrow::toVeloxColumn(
      cudf::table_view({asView(result)}),
      pool_.get(),
      outputType,
      "",
      stream,
      resource);
  auto expected = makeFlatVector<int32_t>(
      65,
      [](auto row) { return row + 3; },
      [](auto row) { return (row + 3) % 7 == 0; });
  velox::test::assertEqualVectors(expected, actual->childAt(0));
}

} // namespace
} // namespace facebook::velox::cudf_velox
