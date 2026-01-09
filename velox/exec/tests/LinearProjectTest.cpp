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

#include "velox/exec/tests/utils/FeatureGen.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/ProjectSequence.h"
#include "velox/exec/Linear.h"
#include "velox/parse/Expressions.h"
#include "velox/core/Expressions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/core/QueryConfig.h"

namespace facebook::velox::exec {
namespace {
using namespace facebook::velox::exec::test;

class LinearProjectTest : public test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    test::HiveConnectorTestBase::SetUp();
    setupLinearMetadata();
    setupSpecialForms();
  }

  // Runs the plan against the splits twice: once with use_project_sequence=false
  // and once with use_project_sequence=true. Verifies that both runs produce
  // the same results.
  void checkSame(
      const core::PlanNodePtr& plan,
      const std::vector<std::shared_ptr<connector::hive::HiveConnectorSplit>>& splits) {
    // Convert connector splits to exec::Splits
    std::vector<exec::Split> execSplits;
    for (auto& connectorSplit : splits) {
      execSplits.emplace_back(exec::Split(connectorSplit, -1));
    }

    // Run with use_project_sequence = false
    auto resultsWithoutProjectSequence =
        AssertQueryBuilder(plan)
            .splits(execSplits)
            .config(core::QueryConfig::kUseProjectSequence, "false")
            .copyResults(pool_.get());

    // Run with use_project_sequence = true
    auto resultsWithProjectSequence =
        AssertQueryBuilder(plan)
            .splits(execSplits)
            .config(core::QueryConfig::kUseProjectSequence, "true")
            .copyResults(pool_.get());

    // Compare results
    assertEqualResults(
		       std::vector<RowVectorPtr>{resultsWithoutProjectSequence},
		       std::vector<RowVectorPtr>{resultsWithProjectSequence});
  }
};

TEST_F(LinearProjectTest, basic) {
  // Create a test dataset with 10 bigint children
  vector_size_t numRows = 1000;
  std::vector<VectorPtr> children;

  for (int i = 0; i < 10; ++i) {
    auto flat = makeFlatVector<int64_t>(
        numRows,
        [](vector_size_t row) { return row; });

    // Add nulls for children 5-9
    if (i >= 5) {
      // Every i-th row is null
      for (vector_size_t row = 0; row < numRows; row += i) {
        flat->setNull(row, true);
      }
    }

    children.push_back(flat);
  }

  // Create RowVector with default names (c0, c1, ..., c9)
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (int i = 0; i < 10; ++i) {
    names.push_back(fmt::format("c{}", i));
    types.push_back(BIGINT());
  }
  auto rowType = ROW(std::move(names), std::move(types));
  auto rowVector = std::make_shared<RowVector>(
      pool_.get(),
      rowType,
      BufferPtr(nullptr),
      numRows,
      std::move(children));

  // Create plan with PlanBuilder
  auto plan = PlanBuilder()
      .values({rowVector})
      .project({
          "c0 + 1 as c0",
          "c1 + c0 as c1",
          "coalesce(c2, 0) as c2",
          "c3 as c3",
          "c4 as c4",
          "coalesce(c5, 0) as c5",
          "coalesce(c6 + c7, 0) as c6",
          "cast(row_constructor(coalesce(c7, 0), c8, c9) as row<c0 bigint, c0 bigint, c0 bigint>) as c7"
      })
      .project({
          "c0 * 2 as c0",
          "c1 * 2 as c1",
          "coalesce(c2, 0) * 3 as c2",
          "clamp(coalesce(c4, 0), 100, 4000) as c3",
          "c5 + c5 as c4",
          "row_constructor(clamp(c7.c0, 30, 300), clamp(c7.c1, 10, 200), c7.c2) as c5"
      })
      .planNode();

  // Run with use_project_sequence = false
  auto resultsWithoutProjectSequence =
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kUseProjectSequence, "false")
          .copyResults(pool_.get());

  // Run with use_project_sequence = true
  auto resultsWithProjectSequence =
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kUseProjectSequence, "true")
          .copyResults(pool_.get());

  // Compare results
  assertEqualResults(
      std::vector<RowVectorPtr>{resultsWithoutProjectSequence},
      std::vector<RowVectorPtr>{resultsWithProjectSequence});
}


TEST_F(LinearProjectTest, featureBasic) {
  test::FeatureOptions opts;
  opts.rng.seed(1);
  auto vectors = test::makeFeatures(1, 100, opts, pool_.get());

  const auto rowType = vectors[0]->rowType();
  const auto fields = rowType->names();

  auto config = std::make_shared<dwrf::Config>();
  config->set(dwrf::Config::FLATTEN_MAP, true);
  config->set<const std::vector<uint32_t>>(
      dwrf::Config::MAP_FLAT_COLS, {2, 3, 4});

  auto file = TempFilePath::create();
  writeToFile(file->getPath(), vectors, config, rowType);

  auto readSchema = ROW({"uid", "ts", "float_features", "id_list_features"}, {BIGINT(), BIGINT(), opts.floatStruct, opts.idListStruct});

  auto plan =
        PlanBuilder()
            .tableScan(readSchema, {}, "", rowType)
            .filter("uid % 511 < 508")
            .project(
                {"uid",
                 "ts",
                 "row_constructor(coalesce(float_features.10010, 0), coalesce(float_features.10020, 0)) as ff_1",
                 "id_list_features"})
            .project(
                {"uid",
                 "ts",
                 "row_constructor(ff_1.c0 * 2 + 1, clamp(ff_1.c1 + 2, -10, 10))",
                 "row_constructor(array_sum(first_x(id_list_features.200100, 10)), array_intersect(id_list_features.200200, id_list_features.200300)) as id_list_features"})
            .planNode();

  auto split = makeHiveConnectorSplit(file->getPath());
}

TEST_F(LinearProjectTest, constantFolding) {
  // Test that preprocess folds constants in "a + (1 + 2 + 3)" to "a + 6"
  auto rowType = ROW({"a"}, {BIGINT()});

  // Parse the expression "a + (1 + 2 + 3)"
  auto untyped = parse::parseExpr("a + (1 + 2 + 3)", {});
  auto typedExpr = core::Expressions::inferTypes(untyped, rowType, pool());

  #if 0
  // Create a simple project node to create ProjectSequence
  auto projectNode = std::make_shared<core::ProjectNode>(
      "test_project",
      rowType,
      std::vector<std::string>{"result"},
      std::vector<core::TypedExprPtr>{typedExpr},
      std::make_shared<core::ValuesNode>("values", rowType));

  // Create ProjectSequence
  ProjectVector projects = {projectNode};
  auto driverCtx = createDriverCtx();
  ProjectSequence sequence(0, driverCtx.get(), projects);

  // Apply preprocessing
  auto preprocessed = sequence.preprocess(typedExpr);

  // Check that the result has the expected structure: "a + 6"
  ASSERT_EQ(preprocessed->kind(), core::ExprKind::kCall);
  auto call = preprocessed->asUnchecked<core::CallTypedExpr>();
  ASSERT_EQ(call->name(), "plus");
  ASSERT_EQ(call->inputs().size(), 2);

  // First input should be field access to "a"
  ASSERT_EQ(call->inputs()[0]->kind(), core::ExprKind::kFieldAccess);

  // Second input should be constant 6
  ASSERT_EQ(call->inputs()[1]->kind(), core::ExprKind::kConstant);
  auto constant = call->inputs()[1]->asUnchecked<core::ConstantTypedExpr>();
  ASSERT_EQ(constant->value().value<int64_t>(), 6);
#endif
}

} // namespace
} // namespace facebook::velox::exec
