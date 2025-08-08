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

#include "velox/exec/fuzzer/VeloxLocalQueryRunner.h"
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/dwrf/RegisterDwrfWriter.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/ComplexVector.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

class VeloxLocalQueryRunnerTest : public testing::Test {
 protected:
  void SetUp() override {
    facebook::velox::memory::MemoryManager::initialize(
        facebook::velox::memory::MemoryManager::Options{});

    filesystems::registerLocalFileSystem();
    connector::registerConnectorFactory(
        std::make_shared<connector::hive::HiveConnectorFactory>());
    dwrf::registerDwrfWriterFactory();

    Type::registerSerDe();
    core::PlanNode::registerSerDe();
    core::ITypedExpr::registerSerDe();
    parse::registerTypeResolver();

    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions(
        "presto.default.", false, true);
    window::prestosql::registerAllWindowFunctions("presto.default.");

    rootPool_ = memory::memoryManager()->addRootPool("root");
    pool_ = rootPool_->addLeafChild("leaf2");
  }

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> pool_;
};

// This test is disabled by default because it requires a running
// LocalRunnerService To run this test, start the LocalRunnerService first: $ cd
// fbsource/fbcode $ buck run velox/runner/tests:local_runner_service
TEST_F(VeloxLocalQueryRunnerTest, SimpleQuery) {
  // Create a VeloxLocalQueryRunner
  auto queryRunner = std::make_unique<VeloxLocalQueryRunner>(
      rootPool_.get(),
      "http://127.0.0.1:9090",
      std::chrono::milliseconds(5000));

  // Create a simple plan: values -> project
  auto rowType = ROW({"c0", "c1"}, {INTEGER(), VARCHAR()});
  std::vector<RowVectorPtr> values;

  // Create a values node with 3 rows
  auto vector = BaseVector::create(rowType, 3, pool_.get());
  auto flatVector1 =
      vector->as<RowVector>()->childAt(0)->asFlatVector<int32_t>();
  auto flatVector2 =
      vector->as<RowVector>()->childAt(1)->asFlatVector<StringView>();

  flatVector1->set(0, 1);
  flatVector1->set(1, 2);
  flatVector1->set(2, 3);

  flatVector2->set(0, "a");
  flatVector2->set(1, "b");
  flatVector2->set(2, "c");

  values.push_back(std::static_pointer_cast<RowVector>(vector));

  // Build the plan
  auto valuesNode = PlanBuilder().values(values).planNode();
  auto projectionNode =
      PlanBuilder().values(values).project({"c0 + 10"}).planNode();

  // Execute the plan
  auto result = queryRunner->executeAndReturnVector(projectionNode);

  // Check the result
  ASSERT_TRUE(result.first.has_value());
  ASSERT_EQ(result.second, ReferenceQueryErrorCode::kSuccess);
  ASSERT_EQ(result.first->size(), 1);
  ASSERT_EQ(result.first->at(0)->size(), 3);

  // Check the values
  auto resultVector = result.first->at(0);
  auto tmp = resultVector->childAt(0);
  auto tmpString = resultVector->toString();
  auto resultFlatVector = resultVector->childAt(0)->asFlatVector<int64_t>();
  ASSERT_EQ(resultFlatVector->valueAt(0), 11);
  ASSERT_EQ(resultFlatVector->valueAt(1), 12);
  ASSERT_EQ(resultFlatVector->valueAt(2), 13);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::init(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
