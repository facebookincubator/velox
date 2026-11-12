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
#include "velox/core/QueryConfig.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/types/JsonType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Type.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <folly/ScopeGuard.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::test_utils;

namespace {

// Keeps a copy to exercise the config lifetime contract without GPU execution.
class SessionConfigFunction : public CudfFunction {
 public:
  explicit SessionConfigFunction(const core::QueryConfig& config)
      : config_(config) {}

  std::string sessionTimezone() const {
    return config_.sessionTimezone();
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>&,
      cuda::stream_ref,
      rmm::device_async_resource_ref) const override {
    VELOX_UNREACHABLE("SessionConfigFunction is only used for construction");
  }

 private:
  const core::QueryConfig config_;
};

// Stand-ins for a GPU simple function's initialize(). Two report a check they
// failed, as one behind the GPU shadow does; two throw, as a callee compiled
// outside it does.
uint8_t initializeFailsUserCheck(
    void* /*instance*/,
    const std::vector<TypePtr>& /*inputTypes*/,
    const core::QueryConfig& /*config*/) {
  return static_cast<uint8_t>(gpu_sfi::ErrorClass::kUserError);
}

uint8_t initializeFailsRuntimeCheck(
    void* /*instance*/,
    const std::vector<TypePtr>& /*inputTypes*/,
    const core::QueryConfig& /*config*/) {
  return static_cast<uint8_t>(gpu_sfi::ErrorClass::kRuntimeError);
}

uint8_t initializeThrowsUserError(
    void* /*instance*/,
    const std::vector<TypePtr>& /*inputTypes*/,
    const core::QueryConfig& /*config*/) {
  VELOX_USER_FAIL("Unknown time zone: Mars/Olympus");
}

uint8_t initializeThrowsRuntimeError(
    void* /*instance*/,
    const std::vector<TypePtr>& /*inputTypes*/,
    const core::QueryConfig& /*config*/) {
  VELOX_FAIL("Instance state is inconsistent");
}

// Stands in for a kernel: every row null.
std::unique_ptr<cudf::column> allNullLaunch(
    const std::vector<gpu_sfi::GpuArgView>& /*arguments*/,
    const gpu_sfi::GpuFunctionInstance& /*instance*/,
    cudf::size_type numRows,
    cudf::data_type outputType,
    uint8_t* /*declinedRows*/,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return cudf::make_fixed_width_column(
      outputType, numRows, cudf::mask_state::ALL_NULL, stream, mr);
}

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

TEST_F(CudfExpressionSelectionTest, functionReceivesSessionConfig) {
  const std::string name = "test_session_config";
  auto functions =
      std::make_shared<std::vector<std::shared_ptr<SessionConfigFunction>>>();
  registerCudfFunction(
      name,
      [functions](
          const std::string&,
          const core::TypedExprPtr&,
          memory::MemoryPool*,
          const core::QueryConfig& config) {
        auto function = std::make_shared<SessionConfigFunction>(config);
        functions->push_back(function);
        return function;
      },
      {exec::FunctionSignatureBuilder()
           .returnType("bigint")
           .argumentType("bigint")
           .build()});

  // Check propagation to a function at both the root and a child node.
  auto child = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "a")},
      name);
  auto expr = std::make_shared<core::CallTypedExpr>(
      BIGINT(), std::vector<core::TypedExprPtr>{child}, name);

  for (const std::string timezone : {"America/Los_Angeles", "Asia/Kolkata"}) {
    SCOPED_TRACE(timezone);
    functions->clear();
    CudfExpressionPtr evaluator;
    {
      auto queryCtx =
          core::QueryCtx::Builder()
              .queryConfig(
                  core::QueryConfig(
                      {{core::QueryConfig::kSessionTimezone, timezone}}))
              .build();
      evaluator = createCudfExpression(
          expr, rowType_, pool_.get(), queryCtx->queryConfig());
    }
    ASSERT_NE(dynamic_cast<FunctionExpression*>(evaluator.get()), nullptr);
    // The session and its QueryConfig have gone out of scope.
    EXPECT_THAT(
        *functions,
        testing::AllOf(
            testing::SizeIs(2),
            testing::Each(
                testing::Pointee(
                    testing::Property(
                        &SessionConfigFunction::sessionTimezone, timezone)))));
  }
}

TEST_F(CudfExpressionSelectionTest, functionEligibilityUsesSessionConfig) {
  const std::string name = "test_session_eligibility";
  registerCudfFunction(
      name,
      [](const std::string&,
         const core::TypedExprPtr&,
         memory::MemoryPool*,
         const core::QueryConfig& config) {
        return std::make_shared<SessionConfigFunction>(config);
      },
      {exec::FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("bigint")
           .build()},
      /*overwrite=*/true,
      [](const core::TypedExprPtr&, const core::QueryConfig& config) {
        return !config.isLegacyCast();
      });
  auto expr = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "a")},
      name);
  auto nested = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(), std::vector<core::TypedExprPtr>{expr}, "not");
  auto legacyQueryCtx =
      core::QueryCtx::Builder()
          .queryConfig(
              core::QueryConfig({{core::QueryConfig::kLegacyCast, "true"}}))
          .build();

  // There are no casts here, so only the registered predicate can reject the
  // expression. Exercise both the root and recursive plan-time checks.
  for (const auto& checked : {expr, nested}) {
    EXPECT_TRUE(canExprRunOnGpu(checked, queryCtx_.get(), pool_.get()));
    EXPECT_FALSE(canExprRunOnGpu(checked, legacyQueryCtx.get(), pool_.get()));
    EXPECT_FALSE(canExprRunOnGpu(checked, legacyQueryCtx.get(), nullptr));
    EXPECT_TRUE(canExprRunOnGpu(checked, nullptr, pool_.get()));
    EXPECT_TRUE(canExprRunOnGpu(checked, nullptr, nullptr));
  }

  auto evaluator = createCudfExpression(
      expr, rowType_, pool_.get(), queryCtx_->queryConfig());
  EXPECT_NE(dynamic_cast<FunctionExpression*>(evaluator.get()), nullptr);
  VELOX_ASSERT_THROW(
      createCudfExpression(
          expr, rowType_, pool_.get(), legacyQueryCtx->queryConfig()),
      "No cuDF expression evaluator can handle");
  EXPECT_EQ(
      createCudfFunction(
          name, expr, pool_.get(), legacyQueryCtx->queryConfig()),
      nullptr);
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

// A GPU simple function whose initialize() fails is handled as Velox's
// SimpleFunctionAdapter handles one: a runtime error fails the call when it is
// compiled, and any other error is raised on every row the function applies
// to. Raising it takes an owner that re-evaluates declined rows on the CPU;
// without one, a nonempty batch raises in its place.
TEST_F(CudfExpressionSelectionTest, gpuSfiInitializeFailures) {
  const gpu_sfi::GpuFunctionSignature signature{
      "bigint",
      {"bigint"},
      /*variadicTail=*/false,
      /*integerVariables=*/{},
      /*variableConstraints=*/{},
      /*argumentKinds=*/{TypeKind::BIGINT},
      /*returnKind=*/TypeKind::BIGINT};
  const auto create = [&](const std::string& name,
                          gpu_sfi::GpuInitializeFn initialize) {
    gpu_sfi::registerGpuKernel(
        {name},
        signature,
        &allNullLaunch,
        gpu_sfi::GpuFunctionInstanceSpec{
            initialize,
            static_cast<int32_t>(sizeof(int64_t)),
            static_cast<int32_t>(alignof(int64_t))});
    return GpuSfiExpression::create(
        std::make_shared<core::CallTypedExpr>(
            BIGINT(),
            std::vector<core::TypedExprPtr>{
                std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "a")},
            name),
        rowType_,
        pool_.get(),
        queryCtx_->queryConfig());
  };

  VELOX_ASSERT_RUNTIME_THROW(
      create("initialize_fails_runtime_check", &initializeFailsRuntimeCheck),
      "GPU simple function failed a runtime check in initialize(): "
      "initialize_fails_runtime_check");
  VELOX_ASSERT_RUNTIME_THROW(
      create("initialize_throws_runtime_error", &initializeThrowsRuntimeError),
      "Instance state is inconsistent");

  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  const auto oneRow = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT64},
      1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  const auto noRows =
      cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64});
  const std::vector<std::pair<std::string, gpu_sfi::GpuInitializeFn>>
      userFailures = {
          {"initialize_fails_user_check", &initializeFailsUserCheck},
          {"initialize_throws_user_error", &initializeThrowsUserError},
      };
  for (const auto& [name, initialize] : userFailures) {
    SCOPED_TRACE(name);
    auto expression = create(name, initialize);
    VELOX_ASSERT_USER_THROW(
        expression->eval({oneRow->view()}, stream, mr),
        "GPU simple function failed a check in initialize(), and its caller "
        "cannot re-evaluate the rows on the CPU to raise the error: " +
            name);
    // Velox raises nothing on a batch with no rows.
    expression->eval({noRows->view()}, stream, mr);
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

TEST_F(CudfExpressionSelectionTest, signatureEnforcesConstantArgsSplit) {
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

TEST_F(CudfExpressionSelectionTest, signatureArrayJoin) {
  auto arrayRowType = ROW({
      {"names", ARRAY(VARCHAR())},
      {"documents", ARRAY(JSON())},
      {"ids", ARRAY(BIGINT())},
      {"sep", VARCHAR()},
  });

  for (const auto* ok : {"array_join(names, ', ')", "array_join(names, '')"}) {
    SCOPED_TRACE(ok);
    auto expr =
        optimizeTypedExpr(ok, arrayRowType, queryCtx_.get(), execCtx_.get());
    ASSERT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  }

  // Stays on the CPU: a separator column or a null separator, the null
  // replacement form, elements that are not plain VARCHAR.
  for (const auto* bad :
       {"array_join(names, sep)",
        "array_join(names, cast(null as varchar))",
        "array_join(names, ', ', 'n/a')",
        "array_join(ids, ', ')",
        "array_join(documents, ', ')"}) {
    SCOPED_TRACE(bad);
    auto expr =
        optimizeTypedExpr(bad, arrayRowType, queryCtx_.get(), execCtx_.get());
    ASSERT_FALSE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  }
}

TEST_F(CudfExpressionSelectionTest, signatureUnixtime) {
  auto unixtimeRowType = ROW({
      {"seconds", DOUBLE()},
      {"ts", TIMESTAMP()},
      {"zoned", TIMESTAMP_WITH_TIME_ZONE()},
      {"count", BIGINT()},
  });

  for (const auto* ok :
       {"from_unixtime(seconds)",
        "to_unixtime(ts)",
        "to_unixtime(from_unixtime(seconds))",
        "from_unixtime(cast(count as double))"}) {
    SCOPED_TRACE(ok);
    auto expr =
        optimizeTypedExpr(ok, unixtimeRowType, queryCtx_.get(), execCtx_.get());
    ASSERT_TRUE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
  }

  // Stays on the CPU: the forms that take or return TIMESTAMP WITH TIME
  // ZONE.
  for (const auto* bad :
       {"from_unixtime(seconds, 'UTC')",
        "from_unixtime(seconds, 8, 0)",
        "to_unixtime(zoned)"}) {
    SCOPED_TRACE(bad);
    auto expr = optimizeTypedExpr(
        bad, unixtimeRowType, queryCtx_.get(), execCtx_.get());
    ASSERT_FALSE(canExprRunOnGpu(expr, queryCtx_.get(), pool_.get()));
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
    EXPECT_EQ(
        createCudfFunction(
            call->name(), expr, pool_.get(), queryCtx_->queryConfig()),
        nullptr);
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

TEST_F(CudfExpressionSelectionTest, castAndTryCast) {
  // OK: cast bigint -> double (supported by cuDF)
  auto okCast = optimizeTypedExpr(
      "cast(a AS double)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okCast, queryCtx_.get(), pool_.get()));

  // OK: try_cast bigint -> double (supported by cuDF)
  auto okTryCast = optimizeTypedExpr(
      "try_cast(a AS double)", rowType_, queryCtx_.get(), execCtx_.get());
  ASSERT_TRUE(canExprRunOnGpu(okTryCast, queryCtx_.get(), pool_.get()));

  // OK: cast bigint -> integer, which the AST does not lower
  auto okIntegerCast = optimizeTypedExpr(
      "cast(a AS integer) = 5",
      rowType_,
      queryCtx_.get(),
      execCtx_.get(),
      {.parseIntegerAsBigint = false, .functionPrefix = ""});
  ASSERT_TRUE(canExprRunOnGpu(okIntegerCast, queryCtx_.get(), pool_.get()));

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
