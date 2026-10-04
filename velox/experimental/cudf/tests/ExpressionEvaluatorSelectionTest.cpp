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
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/JitExpression.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/expression/SparkFunctions.h"
#include "velox/experimental/cudf/functions/GpuSfiExpression.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Type.h"

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::test_utils;

namespace {

class CudfExpressionSelectionTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    facebook::velox::functions::prestosql::registerAllScalarFunctions();
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool("", false);
    queryCtx_ = core::QueryCtx::create();
    execCtx_ = std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get());
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
    cudf_velox::registerPrestoFunctions("");
    rowType_ = ROW({
        {"a", BIGINT()},
        {"b", BIGINT()},
        {"c", INTEGER()},
        {"name", VARCHAR()},
        {"d", DOUBLE()},
        {"date", DATE()},
        {"c", INTEGER()},
    });

    parse::registerTypeResolver();
  }

  void TearDown() override {
    cudf_velox::unregisterFunctions();
    cudf_velox::unregisterCudf();
    execCtx_.reset();
    queryCtx_.reset();
    pool_.reset();
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<core::ExecCtx> execCtx_;
  RowTypePtr rowType_;
};

TEST_F(CudfExpressionSelectionTest, astRoot) {
  auto prevAst = CudfConfig::getInstance().astExpressionEnabled;
  auto prevJit = CudfConfig::getInstance().jitExpressionEnabled;
  SCOPE_EXIT {
    CudfConfig::getInstance().astExpressionEnabled = prevAst;
    CudfConfig::getInstance().jitExpressionEnabled = prevJit;
  };
  CudfConfig::getInstance().astExpressionEnabled = true;
  CudfConfig::getInstance().jitExpressionEnabled = true;
  auto expr =
      optimizeTypedExpr("a + c", rowType_, queryCtx_.get(), execCtx_.get());
  auto cudfExpr = createCudfExpression(
      expr, rowType_, pool_.get(), queryCtx_->queryConfig());
  auto* ast = dynamic_cast<ASTExpression*>(cudfExpr.get());
  auto* jit = dynamic_cast<JitExpression*>(cudfExpr.get());
  ASSERT_TRUE(ast != nullptr || jit != nullptr);
}

TEST_F(CudfExpressionSelectionTest, functionRoot) {
  auto expr = optimizeTypedExpr(
      "lower(name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  auto cudfExpr = createCudfExpression(
      expr, rowType_, pool_.get(), queryCtx_->queryConfig());
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
}

// GPU SFI claims a call node only when a registered function binds the
// argument types exactly: round and truncate register each arity separately,
// a chained AND arrives flattened as one call that the variadic registration
// matches at any arity while still checking the element type, and a null
// literal, which has no element 0 to read, is declined in canEvaluate() so
// another evaluator can take the node rather than create() throwing.
TEST_F(CudfExpressionSelectionTest, gpuSfiCanEvaluate) {
  struct Case {
    std::string sql;
    bool claimed;
  };
  const std::vector<Case> cases = {
      {"bitwise_and(a, b)", true},
      {"d + d", true},
      // Integral arithmetic binds the Checked* structs, as Presto does on the
      // CPU.
      {"a + b", true},
      {"a - b", true},
      {"a * b", true},
      {"a / b", true},
      {"negate(a)", true},
      // The decimal-places argument is INTEGER, while an integer literal
      // parses as BIGINT; a coerced plan carries the cast.
      {"round(d)", true},
      {"round(d, cast(2 as integer))", true},
      {"round(a)", true},
      {"truncate(d)", true},
      {"truncate(d, cast(2 as integer))", true},
      {"a > 1 AND b > 2", true},
      {"a > 1 AND b > 2 AND c > 3", true},
      {"a > 1 AND b > 2 AND c > 3 AND a < 9", true},
      {"a > 1 OR b > 2", true},
      {"a > 1 OR b > 2 OR c > 3", true},
      {"not(a > 1)", true},
      {"bitwise_and(a, cast(null as bigint))", false},
      {"a + cast(null as bigint)", false},
  };
  for (const auto& [sql, claimed] : cases) {
    SCOPED_TRACE(sql);
    auto expr =
        optimizeTypedExpr(sql, rowType_, queryCtx_.get(), execCtx_.get());
    EXPECT_EQ(GpuSfiExpression::canEvaluate(expr), claimed);
  }

  // Shapes the parser would not produce: truncate is floating point only, as
  // in Velox, and the and() pack takes booleans, so a bigint pack of the same
  // shape is not a match.
  struct HandBuilt {
    std::string name;
    TypePtr returnType;
    std::vector<std::string> fields;
  };
  const std::vector<HandBuilt> rejected = {
      {"truncate", BIGINT(), {"a"}},
      {"and", BOOLEAN(), {"a", "b"}},
  };
  for (const auto& [name, returnType, fields] : rejected) {
    SCOPED_TRACE(name);
    std::vector<core::TypedExprPtr> arguments;
    for (const auto& field : fields) {
      arguments.push_back(
          std::make_shared<core::FieldAccessTypedExpr>(
              rowType_->findChild(field), field));
    }
    EXPECT_FALSE(
        GpuSfiExpression::canEvaluate(
            std::make_shared<core::CallTypedExpr>(
                returnType, arguments, name)));
  }
}

// GPU SFI (priority 75) sits between the function tier (50) and AST (100), so
// it evaluates a call nothing else implements, AST takes a call both can
// handle, and roots it never claims, such as a column, a literal or a call with
// a null literal, still run on the GPU through another evaluator. When only
// GPU SFI implements the call and it declines, the operator reports itself
// ineligible at plan time and runs on the CPU.
TEST_F(CudfExpressionSelectionTest, gpuSfiSelection) {
  enum class Evaluator { kGpuSfi, kOther, kNone };
  struct Case {
    std::string sql;
    Evaluator evaluator;
  };
  const std::vector<Case> cases = {
      {"bitwise_and(a, b)", Evaluator::kGpuSfi},
      {"d + d", Evaluator::kOther},
      {"a", Evaluator::kOther},
      {"42", Evaluator::kOther},
      {"a + 1", Evaluator::kOther},
      {"a + cast(null as bigint)", Evaluator::kOther},
      {"bitwise_and(a, cast(null as bigint))", Evaluator::kNone},
  };
  for (const auto& [sql, evaluator] : cases) {
    SCOPED_TRACE(sql);
    auto expr =
        optimizeTypedExpr(sql, rowType_, queryCtx_.get(), execCtx_.get());
    ASSERT_EQ(
        canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()),
        evaluator != Evaluator::kNone);
    if (evaluator == Evaluator::kNone) {
      continue;
    }
    auto cudfExpr = createCudfExpression(
        expr, rowType_, pool_.get(), queryCtx_->queryConfig());
    ASSERT_NE(cudfExpr, nullptr);
    EXPECT_EQ(
        dynamic_cast<GpuSfiExpression*>(cudfExpr.get()) != nullptr,
        evaluator == Evaluator::kGpuSfi);
  }
}

// registerCudf() registers GPU SFI at the configured priority.
TEST_F(CudfExpressionSelectionTest, gpuSfiIsRegisteredAtItsConfiguredPriority) {
  const auto& registry = getCudfExpressionEvaluatorRegistry();
  const auto it = registry.find(kGpuSfiEvaluatorName);
  ASSERT_NE(it, registry.end()) << "GPU SFI evaluator was never registered";
  EXPECT_EQ(
      it->second.priority, CudfConfig::getInstance().gpuSfiExpressionPriority);
}

// DATE is INTEGER underneath, so signature matching must compare logical type
// names, not physical kinds.
TEST_F(CudfExpressionSelectionTest, gpuSfiExtractsDateFields) {
  for (const auto& sql :
       {"year(date)",
        "month(date)",
        "day(date)",
        "quarter(date)",
        "day_of_year(date)",
        "day_of_week(date)"}) {
    SCOPED_TRACE(sql);
    auto expr =
        optimizeTypedExpr(sql, rowType_, queryCtx_.get(), execCtx_.get());
    EXPECT_TRUE(GpuSfiExpression::canEvaluate(expr));
  }

  // An INTEGER argument must not bind to the DATE overload.
  auto onInteger = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c")},
      "year");
  EXPECT_FALSE(GpuSfiExpression::canEvaluate(onInteger));
}

// TIMESTAMP WITH TIME ZONE reaches cuDF as the int64 that packs UTC millis over
// a zone key. The evaluators that would read it as a number decline, so GPU
// SFI, whose kernels unpack it, is the only taker of its functions, and a cast
// stays on the CPU.
TEST_F(
    CudfExpressionSelectionTest,
    timestampWithTimeZoneIsClaimedByGpuSfiAlone) {
  auto rowType = ROW(
      {{"tz0", TIMESTAMP_WITH_TIME_ZONE()},
       {"tz1", TIMESTAMP_WITH_TIME_ZONE()},
       {"name", VARCHAR()},
       {"d", DOUBLE()}});
  for (const auto& sql :
       {"tz0 = tz1",
        "tz0 <> tz1",
        "tz0 < tz1",
        "tz0 <= tz1",
        "tz0 > tz1",
        "tz0 >= tz1",
        "tz0 between tz1 and tz1",
        "year(tz0)",
        "hour(tz0)",
        "second(tz0)",
        "to_unixtime(tz0)",
        "timezone_hour(tz0)",
        "timezone_minute(tz0)",
        "at_timezone(tz0, 'UTC')",
        "from_unixtime(d, 'UTC')",
        "from_unixtime(d, 5, 30)",
        "from_unixtime(d, 'America/Los_Angeles')"}) {
    SCOPED_TRACE(sql);
    auto expr =
        optimizeTypedExpr(sql, rowType, queryCtx_.get(), execCtx_.get());
    EXPECT_FALSE(ASTExpression::canEvaluate(expr));
    EXPECT_FALSE(JitExpression::canEvaluate(expr));
    EXPECT_FALSE(FunctionExpression::canEvaluate(expr));
    EXPECT_TRUE(GpuSfiExpression::canEvaluate(expr));
    EXPECT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
    auto cudfExpr = createCudfExpression(
        expr, rowType, pool_.get(), queryCtx_->queryConfig());
    EXPECT_NE(dynamic_cast<GpuSfiExpression*>(cudfExpr.get()), nullptr);
  }

  for (const auto& sql : {"cast(tz0 as varchar)", "cast(tz0 as timestamp)"}) {
    SCOPED_TRACE(sql);
    auto expr =
        optimizeTypedExpr(sql, rowType, queryCtx_.get(), execCtx_.get());
    EXPECT_FALSE(ASTExpression::canEvaluate(expr));
    EXPECT_FALSE(JitExpression::canEvaluate(expr));
    EXPECT_FALSE(FunctionExpression::canEvaluate(expr));
    EXPECT_FALSE(GpuSfiExpression::canEvaluate(expr));
    EXPECT_FALSE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  }

  // The column itself still reaches the GPU: passing it through reads no bits.
  auto column =
      optimizeTypedExpr("tz0", rowType, queryCtx_.get(), execCtx_.get());
  EXPECT_TRUE(canExprRunOnGpu(column, queryCtx_.get(), pool_.get()));
}

// Which calls depend on the session time zone comes from the registrations:
// a struct that runs initialize() over a TIMESTAMP argument reads the zone
// there. While the zone applies, such a call is GPU SFI's alone, whatever the
// other evaluators would claim.
TEST_F(CudfExpressionSelectionTest, sessionTimeZoneSensitivityIsRegistered) {
  auto rowType = ROW(
      {{"ts", TIMESTAMP()},
       {"date", DATE()},
       {"d", DOUBLE()},
       {"tz0", TIMESTAMP_WITH_TIME_ZONE()}});
  struct Case {
    std::string sql;
    bool dependsOnSessionTimeZone;
  };
  const std::vector<Case> cases{
      {"year(ts)", true},
      {"week(ts)", true},
      {"hour(ts)", true},
      {"minute(ts)", true},
      {"date_trunc('hour', ts)", true},
      {"date_trunc('second', ts)", true},
      {"second(ts)", false},
      {"millisecond(ts)", false},
      {"to_unixtime(ts)", false},
      {"year(date)", false},
      {"from_unixtime(d)", false},
      {"year(tz0)", false},
  };
  for (const auto& testCase : cases) {
    SCOPED_TRACE(testCase.sql);
    auto expr = optimizeTypedExpr(
        testCase.sql, rowType, queryCtx_.get(), execCtx_.get());
    EXPECT_EQ(
        GpuSfiExpression::dependsOnSessionTimeZone(expr),
        testCase.dependsOnSessionTimeZone);
  }

  queryCtx_->testingOverrideConfigUnsafe({
      {core::QueryConfig::kSessionTimezone, "America/Los_Angeles"},
      {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
  });
  for (const auto& testCase : cases) {
    if (!testCase.dependsOnSessionTimeZone) {
      continue;
    }
    SCOPED_TRACE(testCase.sql);
    auto expr = optimizeTypedExpr(
        testCase.sql, rowType, queryCtx_.get(), execCtx_.get());
    auto cudfExpr = createCudfExpression(
        expr, rowType, pool_.get(), queryCtx_->queryConfig());
    EXPECT_NE(dynamic_cast<GpuSfiExpression*>(cudfExpr.get()), nullptr);
  }
}

// Which calls depend on the session time zone is knowledge of the GPU SFI
// registrations, not of the evaluator: with cudf.gpu_sfi_expression_enabled
// off no evaluator honours the zone, and a sensitive call must stay on the CPU
// rather than go to the function tier, which reads TIMESTAMP as UTC. Calls
// that read no zone still run there.
TEST_F(CudfExpressionSelectionTest, sensitiveCallsStayOnTheCpuWithoutGpuSfi) {
  auto& registry = getCudfExpressionEvaluatorRegistry();
  const auto gpuSfi = registry.at(kGpuSfiEvaluatorName);
  auto& config = CudfConfig::getInstance();
  SCOPE_EXIT {
    config.gpuSfiExpressionEnabled = true;
    registry[kGpuSfiEvaluatorName] = gpuSfi;
  };
  // registerCudf() never takes an evaluator out, so the entry an earlier
  // registration left is removed before registering with the flag off.
  registry.erase(kGpuSfiEvaluatorName);
  config.gpuSfiExpressionEnabled = false;
  unregisterCudf();
  registerCudf();
  ASSERT_EQ(registry.count(kGpuSfiEvaluatorName), 0);

  auto rowType =
      ROW({{"ts", TIMESTAMP()}, {"d", DATE()}, {"days", INTERVAL_DAY_TIME()}});
  queryCtx_->testingOverrideConfigUnsafe({
      {core::QueryConfig::kSessionTimezone, "America/Los_Angeles"},
      {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
  });
  for (const auto& sql : {"year(ts)", "date_trunc('day', ts)"}) {
    SCOPED_TRACE(sql);
    auto expr =
        optimizeTypedExpr(sql, rowType, queryCtx_.get(), execCtx_.get());
    EXPECT_FALSE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
    VELOX_ASSERT_THROW(
        createCudfExpression(
            expr, rowType, pool_.get(), queryCtx_->queryConfig()),
        "No cuDF expression evaluator can handle");
  }
  for (const auto& sql : {"year(d)", "d + days", "second(ts)"}) {
    SCOPED_TRACE(sql);
    auto expr =
        optimizeTypedExpr(sql, rowType, queryCtx_.get(), execCtx_.get());
    EXPECT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  }
}

// A unit or a zone name is read once in initialize(), so it binds only as a
// literal; a kernel cannot read a strings column. With a column there no
// evaluator claims the call and it stays on the CPU.
TEST_F(CudfExpressionSelectionTest, unitAndZoneColumnsLeaveTheCallToTheCpu) {
  auto rowType = ROW(
      {{"ts", TIMESTAMP()},
       {"d", DATE()},
       {"tz", TIMESTAMP_WITH_TIME_ZONE()},
       {"unit", VARCHAR()},
       {"dbl", DOUBLE()}});
  for (const auto& sql :
       {"date_trunc(unit, ts)",
        "date_trunc(unit, d)",
        "date_trunc(unit, tz)",
        "at_timezone(tz, unit)",
        "from_unixtime(dbl, unit)"}) {
    SCOPED_TRACE(sql);
    auto expr =
        optimizeTypedExpr(sql, rowType, queryCtx_.get(), execCtx_.get());
    EXPECT_FALSE(GpuSfiExpression::canEvaluate(expr));
    EXPECT_FALSE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  }
}

TEST_F(CudfExpressionSelectionTest, astTopLevelWithFunctionPrecompute) {
  auto prevAst = CudfConfig::getInstance().astExpressionEnabled;
  auto prevJit = CudfConfig::getInstance().jitExpressionEnabled;
  SCOPE_EXIT {
    CudfConfig::getInstance().astExpressionEnabled = prevAst;
    CudfConfig::getInstance().jitExpressionEnabled = prevJit;
  };
  CudfConfig::getInstance().astExpressionEnabled = true;
  CudfConfig::getInstance().jitExpressionEnabled = true;
  auto expr = optimizeTypedExpr(
      "(year(date) > 2020) AND (length(name) < 10)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  auto cudfExpr = createCudfExpression(
      expr, rowType_, pool_.get(), queryCtx_->queryConfig());
  auto* ast = dynamic_cast<ASTExpression*>(cudfExpr.get());
  auto* jit = dynamic_cast<JitExpression*>(cudfExpr.get());
  ASSERT_TRUE(ast != nullptr || jit != nullptr);
}

TEST_F(CudfExpressionSelectionTest, functionTopLevelWithNestedFunction) {
  auto expr = optimizeTypedExpr(
      "lower(substr(name, 1, 5))", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  auto cudfExpr = createCudfExpression(
      expr, rowType_, pool_.get(), queryCtx_->queryConfig());

  // Top level should be Function
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
}

TEST_F(
    CudfExpressionSelectionTest,
    signatureAllowsRowConstructorAndDereference) {
  auto row =
      parseAndInferTypedExpr("row_constructor(a, b)", rowType_, execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(row, queryCtx_.get(), pool_.get()));

  auto firstField = parseAndInferTypedExpr(
      "row_constructor(a, b).c1", rowType_, execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(firstField, queryCtx_.get(), pool_.get()));

  auto secondField = parseAndInferTypedExpr(
      "row_constructor(a, 1).c2", rowType_, execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(secondField, queryCtx_.get(), pool_.get()));

  auto nullLiteralField = parseAndInferTypedExpr(
      "row_constructor(a, cast(null as bigint)).c2", rowType_, execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(nullLiteralField, queryCtx_.get(), pool_.get()));

  auto leadingNullField = parseAndInferTypedExpr(
      "row_constructor(cast(null as bigint), b).c1", rowType_, execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(leadingNullField, queryCtx_.get(), pool_.get()));

  auto nestedField = parseAndInferTypedExpr(
      "row_constructor(row_constructor(a, cast(null as bigint)), b).c1.c2",
      rowType_,
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(nestedField, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, nestedRowDereferenceUsesFunctionEvaluator) {
  auto prevAst = CudfConfig::getInstance().astExpressionEnabled;
  auto prevJit = CudfConfig::getInstance().jitExpressionEnabled;
  SCOPE_EXIT {
    CudfConfig::getInstance().astExpressionEnabled = prevAst;
    CudfConfig::getInstance().jitExpressionEnabled = prevJit;
  };
  CudfConfig::getInstance().astExpressionEnabled = true;
  CudfConfig::getInstance().jitExpressionEnabled = true;

  auto expr = parseAndInferTypedExpr(
      "row_constructor(row_constructor(a, b), cast(null as bigint)).c1.c1",
      rowType_,
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));

  auto cudfExpr = createCudfExpression(
      expr, rowType_, pool_.get(), queryCtx_->queryConfig());
  auto* functionExpr = dynamic_cast<FunctionExpression*>(cudfExpr.get());
  ASSERT_NE(functionExpr, nullptr);
}

TEST_F(
    CudfExpressionSelectionTest,
    signatureAllowsRowConstructorDereferenceByIndex) {
  auto unnamedRowType = ROW({{"", BIGINT()}, {"", BIGINT()}});
  core::TypedExprPtr expr = std::make_shared<core::DereferenceTypedExpr>(
      BIGINT(),
      std::make_shared<core::CallTypedExpr>(
          unnamedRowType,
          std::vector<core::TypedExprPtr>{
              std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "a"),
              std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "b"),
          },
          "row_constructor"),
      1);

  ASSERT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  ASSERT_NE(
      createCudfExpression(
          expr, rowType_, pool_.get(), queryCtx_->queryConfig()),
      nullptr);
}

TEST_F(
    CudfExpressionSelectionTest,
    signatureAllowsRowConstructorDereferenceByName) {
  auto namedRowType = ROW({{"left", BIGINT()}, {"right", BIGINT()}});
  core::TypedExprPtr expr = std::make_shared<core::FieldAccessTypedExpr>(
      BIGINT(),
      std::make_shared<core::CallTypedExpr>(
          namedRowType,
          std::vector<core::TypedExprPtr>{
              std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "a"),
              std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "b"),
          },
          "row_constructor"),
      "right");

  ASSERT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  ASSERT_NE(
      createCudfExpression(
          expr, rowType_, pool_.get(), queryCtx_->queryConfig()),
      nullptr);
}

// Disabled because this test segfaults in CI while building the typed
// not use cudf code.
TEST_F(
    CudfExpressionSelectionTest,
    DISABLED_signatureEnforcesConstantArgsSplit) {
  // OK: delimiter and limit are constants
  auto ok = optimizeTypedExpr(
      "split(name, ',', 3)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  ASSERT_TRUE(canExprRunOnGpu(ok, queryCtx_.get(), pool_.get()));

  // Bad: delimiter is not a constant
  auto bad = optimizeTypedExpr(
      "split(name, name, 3)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  ASSERT_FALSE(canExprRunOnGpu(bad, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, signatureAllowsColumnPatternLike) {
  // OK: pattern is a constant
  auto ok = optimizeTypedExpr(
      "like(name, '%abc%')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok, queryCtx_.get(), pool_.get()));

  // OK: pattern can also come from a column.
  auto okColumn = optimizeTypedExpr(
      "like(name, name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okColumn, queryCtx_.get(), pool_.get()));

  // OK: constant input still works when pattern comes from a column.
  auto okConstantInput = optimizeTypedExpr(
      "like('abc', name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okConstantInput, queryCtx_.get(), pool_.get()));

  // OK: constant null input should also remain on the cuDF path.
  auto okNullInput = optimizeTypedExpr(
      "like(cast(null as varchar), name)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okNullInput, queryCtx_.get(), pool_.get()));

  // OK: escape can be a constant too.
  auto okWithEscape = optimizeTypedExpr(
      "like(name, '%#_%', '#')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okWithEscape, queryCtx_.get(), pool_.get()));

  // OK: pattern column + constant escape is supported.
  auto okColumnWithEscape = optimizeTypedExpr(
      "like(name, name, '#')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(
      canExprRunOnGpu(okColumnWithEscape, queryCtx_.get(), pool_.get()));

  // OK: constant input + pattern column + constant escape is supported.
  auto okConstantInputWithEscape = optimizeTypedExpr(
      "like('a_c', name, '#')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(
      canExprRunOnGpu(okConstantInputWithEscape, queryCtx_.get(), pool_.get()));

  // OK: constant null input + pattern column + constant escape is supported.
  auto okNullInputWithEscape = optimizeTypedExpr(
      "like(cast(null as varchar), name, '#')",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(
      canExprRunOnGpu(okNullInputWithEscape, queryCtx_.get(), pool_.get()));

  // OK: null constants should remain on the cuDF path.
  auto okNullPattern = optimizeTypedExpr(
      "like(name, cast(null as varchar))",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okNullPattern, queryCtx_.get(), pool_.get()));

  auto okNullEscape = optimizeTypedExpr(
      "like(name, '%#_%', cast(null as varchar))",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okNullEscape, queryCtx_.get(), pool_.get()));

  // Bad: escape is not a constant.
  auto badEscape = optimizeTypedExpr(
      "like(name, '%#_%', name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_FALSE(canExprRunOnGpu(badEscape, queryCtx_.get(), pool_.get()));

  // Bad: escape column is still unsupported when pattern comes from a column.
  auto badColumnEscape = optimizeTypedExpr(
      "like(name, name, name)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_FALSE(canExprRunOnGpu(badColumnEscape, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, signatureArityAndConstantsSubstr) {
  // The default parser keeps integer literals as BIGINT, which exercises the
  // Presto-compatible `substr` candidate. The Spark candidate, which takes
  // INTEGER positions, is covered in tests/sparksql.

  // OK: 2-arg substr with constant start
  auto ok2 = optimizeTypedExpr(
      "substr(name, 1)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok2, queryCtx_.get(), pool_.get()));

  // OK: 3-arg substr with constant start and length
  auto ok3 = optimizeTypedExpr(
      "substr(name, 1, 5)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok3, queryCtx_.get(), pool_.get()));

  // Bad: column positions are unsupported for BIGINT.
  auto badBigintStart = optimizeTypedExpr(
      "substr(name, a)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_FALSE(canExprRunOnGpu(badBigintStart, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, signatureArrayAccess) {
  auto arrayRowType = ROW({
      {"arr", ARRAY(INTEGER())},
      {"idx_bigint", BIGINT()},
      {"idx_integer", INTEGER()},
  });

  for (const auto& functionName : {"element_at", "subscript"}) {
    SCOPED_TRACE(functionName);

    auto bigintExpr = parseAndInferTypedExpr(
        std::string(functionName) + "(arr, idx_bigint)",
        arrayRowType,
        execCtx_.get());
    ASSERT_TRUE(canExprRunOnGpu(bigintExpr, queryCtx_.get(), pool_.get()));

    auto integerExpr = parseAndInferTypedExpr(
        std::string(functionName) + "(arr, idx_integer)",
        arrayRowType,
        execCtx_.get());
    ASSERT_TRUE(canExprRunOnGpu(integerExpr, queryCtx_.get(), pool_.get()));
  }
}

TEST_F(CudfExpressionSelectionTest, signatureCastsInDivide) {
  // OK: numeric args are castable to double
  auto ok = optimizeTypedExpr(
      "divide(a, b)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, signatureTypeVariableCoalesce) {
  // OK: same type BIGINT
  auto ok1 = optimizeTypedExpr(
      "coalesce(a, b)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok1, queryCtx_.get(), pool_.get()));

  // OK: VARCHAR with literal
  auto ok2 = optimizeTypedExpr(
      "coalesce(name, 'x')", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok2, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, signatureTypeVariableSwitchIf) {
  // OK: boolean + same type BIGINT
  auto ok1 = optimizeTypedExpr(
      "if(true, a, b)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(ok1, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, switchWithoutElseResultTypes) {
  const std::vector<TypePtr> supported{
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      REAL(),
      DOUBLE(),
      VARCHAR(),
      VARBINARY(),
      TIMESTAMP(),
      DATE(),
      DECIMAL(7, 2),
      DECIMAL(20, 2)};
  const std::vector<TypePtr> unsupported{
      ARRAY(BIGINT()),
      ROW("x", BIGINT()),
      MAP(BIGINT(), BIGINT()),
      UNKNOWN(),
      HUGEINT(),
      INTERVAL_DAY_TIME(),
      INTERVAL_YEAR_MONTH()};
  for (const auto& name : {"switch", "if"}) {
    for (const bool expected : {false, true}) {
      for (const auto& type : expected ? supported : unsupported) {
        SCOPED_TRACE(fmt::format("{}: {}", name, type->toString()));
        auto expr = std::make_shared<core::CallTypedExpr>(
            type,
            std::vector<core::TypedExprPtr>{
                std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "flag"),
                std::make_shared<core::FieldAccessTypedExpr>(type, "value")},
            name);
        EXPECT_EQ(
            canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()), expected);
      }
    }
  }
}

TEST_F(CudfExpressionSelectionTest, switchConstantCondition) {
  for (const auto& sql :
       {"CASE WHEN true THEN a END",
        "CASE WHEN false THEN a END",
        "if(true, a, b)",
        "if(false, a, b)"}) {
    SCOPED_TRACE(sql);
    auto expr = parseAndInferTypedExpr(sql, rowType_, execCtx_.get());
    const auto* call = expr->asUnchecked<core::CallTypedExpr>();
    EXPECT_EQ(createCudfFunction(call->name(), expr, pool_.get()), nullptr);
    auto optimized =
        optimizeTypedExpr(sql, rowType_, queryCtx_.get(), execCtx_.get());
    EXPECT_TRUE(canExprRunOnGpu(optimized, queryCtx_.get(), pool_.get()));
  }
}

TEST_F(CudfExpressionSelectionTest, switchWithElseRetainsNestedTypes) {
  auto rowType = ROW({
      {"flag", BOOLEAN()},
      {"values", ARRAY(BIGINT())},
      {"others", ARRAY(BIGINT())},
      {"pair", ROW("x", BIGINT())},
      {"other_pair", ROW("x", BIGINT())},
  });
  for (const auto& sql :
       {"CASE WHEN flag THEN values ELSE others END",
        "CASE WHEN flag THEN pair ELSE other_pair END",
        "if(flag, values, others)",
        "if(flag, pair, other_pair)"}) {
    SCOPED_TRACE(sql);
    auto expr =
        optimizeTypedExpr(sql, rowType, queryCtx_.get(), execCtx_.get());
    EXPECT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  }
}

TEST_F(CudfExpressionSelectionTest, DISABLED_castAndTryCast) {
  // TODO (dm): This is required for passing of castAndTryCast test but breaks
  // others. This is because ASTExpr agrees to support bad casts. remove after
  // ASTExpr checks cast types
  // CudfConfig::getInstance().astExpressionEnabled = false;

  // OK: cast bigint -> double (supported by cuDF)
  auto okCast = optimizeTypedExpr(
      "cast(a AS double)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okCast, queryCtx_.get(), pool_.get()));

  // OK: try_cast bigint -> double (supported by cuDF)
  auto okTryCast = optimizeTypedExpr(
      "try_cast(a AS double)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okTryCast, queryCtx_.get(), pool_.get()));

  // BAD: cast boolean -> date (expected unsupported by cuDF)
  auto badCast = optimizeTypedExpr(
      "cast(length(name) < 10 AS date)",
      rowType_,
      queryCtx_.get(),
      execCtx_.get());
  ASSERT_FALSE(canExprRunOnGpu(badCast, queryCtx_.get(), pool_.get()));
}

TEST_F(CudfExpressionSelectionTest, constantFoldingStringAllocatesOnCompile) {
  auto optimized = optimizeTypedExpr(
      "lower('ABCDEF')", rowType_, queryCtx_.get(), execCtx_.get());

  ASSERT_TRUE(optimized->isConstantKind());
  auto* constant = optimized->asUnchecked<core::ConstantTypedExpr>();
  auto value = constant->toConstantVector(execCtx_->pool());
  ASSERT_EQ(value->toString(0), "abcdef");
  if (constant->hasValueVector()) {
    ASSERT_EQ(constant->valueVector()->pool(), execCtx_->pool());
  }
}

// ---------------------------------------------------------------------------
// createCudfExpression tests — verify the pure-function compilation API
// and expression optimization.
// ---------------------------------------------------------------------------

TEST_F(CudfExpressionSelectionTest, compilerPureAstNoBoundaries) {
  // A simple arithmetic expression handled entirely by AST should compile
  // successfully.
  auto expr = parseAndInferTypedExpr("a + b", rowType_, execCtx_.get());
  auto result = createCudfExpression(
      expr, rowType_, pool_.get(), queryCtx_->queryConfig());
  ASSERT_NE(result, nullptr);
}

TEST_F(CudfExpressionSelectionTest, compilerFunctionBoundaryInAst) {
  // An expression like "a + b > cardinality(names)" where the top-level
  // comparison is AST but cardinality is only supported as a CudfFunction.
  // The compiler should handle mixed evaluators transparently.
  auto arrayType = ROW({
      {"a", BIGINT()},
      {"b", BIGINT()},
      {"names", ARRAY(VARCHAR())},
  });

  auto expr = parseAndInferTypedExpr(
      "a + b > cardinality(names)", arrayType, execCtx_.get());
  auto result = createCudfExpression(
      expr, arrayType, pool_.get(), queryCtx_->queryConfig());
  ASSERT_NE(result, nullptr);
}

TEST_F(CudfExpressionSelectionTest, compilerOptimizesConstantExpr) {
  // expression::optimize folds constant subtrees; "a + (1 + 2)" optimizes to
  // "a + 3".
  auto expr = parseAndInferTypedExpr("a + (1 + 2)", rowType_, execCtx_.get());

  const auto optimized =
      expression::optimize(expr, queryCtx_.get(), pool_.get());
  ASSERT_NE(optimized, nullptr);

  auto result = createCudfExpression(
      optimized, rowType_, pool_.get(), queryCtx_->queryConfig());
  ASSERT_NE(result, nullptr);

  // The optimized tree should have a constant child for the folded value.
  // It should be "a + 3" which has one FieldAccess child and one Constant.
  bool hasConstant = false;
  for (const auto& child : optimized->inputs()) {
    if (child->isConstantKind()) {
      hasConstant = true;
    }
  }
  EXPECT_TRUE(hasConstant)
      << "Constant folding should produce a constant child in 'a + (1+2)'";
}

TEST_F(CudfExpressionSelectionTest, compilerSimpleExpressionCompiles) {
  auto expr = parseAndInferTypedExpr("a + b", rowType_, execCtx_.get());
  auto result = createCudfExpression(
      expr, rowType_, pool_.get(), queryCtx_->queryConfig());
  ASSERT_NE(result, nullptr);
}

} // namespace
