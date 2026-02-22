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
#include "velox/exec/tests/utils/VeloxPlanLoader.h"
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/exec/PartitionFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/TpcdsQueryBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

namespace facebook::velox::exec::test {

class VeloxPlanLoaderTest : public HiveConnectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    OperatorTestBase::SetUpTestCase();
  }

  static void TearDownTestCase() {
    OperatorTestBase::TearDownTestCase();
  }

  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    parquet::registerParquetReaderFactory();
    functions::prestosql::registerAllScalarFunctions("presto.default.");
    aggregate::prestosql::registerAllAggregateFunctions("presto.default.");
    parse::registerTypeResolver();

    Type::registerSerDe();
    common::Filter::registerSerDe();
    connector::hive::HiveConnector::registerSerDe();
    core::PlanNode::registerSerDe();
    core::ITypedExpr::registerSerDe();
    registerPartitionFunctionSerDe();
  }

  void TearDown() override {
    parquet::unregisterParquetReaderFactory();
    HiveConnectorTestBase::TearDown();
  }
};

TEST_F(VeloxPlanLoaderTest, resolvePlanDirectoryUsesEnv) {
  const std::string defaultDir = "/tmp/";
  // Test with TPCDS_PLAN_DIR env var.
  std::string resolved =
      VeloxPlanLoader::resolvePlanDirectory(defaultDir, "TPCDS_PLAN_DIR");
  EXPECT_FALSE(resolved.empty());
  const char* envDir = std::getenv("TPCDS_PLAN_DIR");
  if (envDir && *envDir != '\0') {
    EXPECT_EQ(resolved, std::string(envDir));
  } else {
    EXPECT_EQ(resolved, defaultDir);
  }

  // Test with no env var name -> always returns default.
  std::string resolved2 = VeloxPlanLoader::resolvePlanDirectory(defaultDir);
  EXPECT_EQ(resolved2, defaultDir);
}

TEST_F(VeloxPlanLoaderTest, getQueryPlanLoadsAndDeserializes) {
  const char* envDir = std::getenv("TPCDS_PLAN_DIR");
  const std::string planDir =
      envDir && *envDir ? std::string(envDir) : "tpcds_plans";

  VeloxPlanLoader loader(planDir, pool());
  for (int queryId = 1; queryId <= 99; ++queryId) {
    VeloxPlan plan;
    try {
      plan = loader.loadPlanByQueryId(queryId);
    } catch (const std::exception& e) {
      GTEST_SKIP() << "Plan for Q" << queryId
                   << ".json not available: " << e.what();
    }

    ASSERT_TRUE(plan.plan != nullptr);
    EXPECT_GT(plan.plan->id().size(), 0u);
    EXPECT_TRUE(plan.plan->outputType() != nullptr);
    auto tableScanNodes = VeloxPlanLoader::collectTableScanNodes(plan.plan);
    EXPECT_GT(tableScanNodes.size(), 0u);
    std::cout << "TPCDS TableScanNodes for Q" << queryId << ":" << std::endl;
    for (const auto& tableScanNode : tableScanNodes) {
      std::cout << "TableScanNode: " << tableScanNode->toString(true, true)
                << std::endl;
    }
#ifndef NDEBUG
    std::cout << "TPCDS Plan for Q" << queryId << ":" << std::endl
              << plan.plan->toString(true, true) << std::endl;
#endif
  }
}

TEST_F(VeloxPlanLoaderTest, getQueryPlanInvalidIdThrows) {
  const char* envDir = std::getenv("TPCDS_PLAN_DIR");
  const std::string planDir =
      envDir && *envDir ? std::string(envDir) : "tpcds_plans";

  VeloxPlanLoader loader(planDir, pool());
  EXPECT_THROW(loader.loadPlanByQueryId(0), velox::VeloxUserError);
  EXPECT_THROW(loader.loadPlanByQueryId(100), velox::VeloxUserError);
}

/// Runs TPC-DS query 1 using TpcdsQueryBuilder which loads the plan from JSON
/// and populates splits from a data directory. Requires TPCDS_PLAN_DIR and
/// TPCDS_DATA_DIR environment variables to be set.
TEST_F(VeloxPlanLoaderTest, runTpcdsQuery1WithBuilder) {
  const char* envPlanDir = std::getenv("TPCDS_PLAN_DIR");
  if (!envPlanDir || !*envPlanDir) {
    GTEST_SKIP() << "Set TPCDS_PLAN_DIR to run this test";
  }
  const std::string planDir(envPlanDir);

  const char* envDataDir = std::getenv("TPCDS_DATA_DIR");
  if (!envDataDir || !*envDataDir) {
    GTEST_SKIP() << "Set TPCDS_DATA_DIR to run this test";
  }
  const std::string dataDir(envDataDir);

  TpcdsQueryBuilder builder;
  builder.initialize(dataDir);

  auto veloxPlan = builder.getQueryPlan(1, planDir, pool());
  ASSERT_NE(veloxPlan.plan, nullptr);

  std::cout << "TpcdsQueryBuilder: connector ID from plan = '"
            << builder.connectorId() << "'" << std::endl;

  // Print scan nodes and matched data files for debugging.
  auto scanNodes = VeloxPlanLoader::collectTableScanNodes(veloxPlan.plan);
  for (const auto& scan : scanNodes) {
    const auto& tbl = scan->tableHandle()->name();
    auto it = veloxPlan.dataFiles.find(scan->id());
    std::cout << "  TableScan " << scan->id() << " table='" << tbl << "' files="
              << (it != veloxPlan.dataFiles.end()
                      ? std::to_string(it->second.size())
                      : "NONE")
              << std::endl;
  }

  // Verify that we found data files for at least some scan nodes.
  if (veloxPlan.dataFiles.empty()) {
    builder.shutdown();
    GTEST_SKIP() << "No data files matched any TableScan node for query 1. "
                 << "Check that TPCDS_DATA_DIR subdirectory names match the "
                 << "table names in the plan.";
  }

  // Build the query with splits for each TableScan node.
  auto queryBuilder = AssertQueryBuilder(veloxPlan.plan);
  for (const auto& [nodeId, files] : veloxPlan.dataFiles) {
    for (const auto& file : files) {
      queryBuilder.split(nodeId, exec::Split(builder.makeSplit(file)));
    }
  }

  auto result = queryBuilder.copyResults(pool());
  ASSERT_NE(result, nullptr);
  ASSERT_GT(result->size(), 0u) << "TPC-DS query 1 returned no results";

  std::cout << "TPC-DS Q1 returned " << result->size() << " rows." << std::endl;

  builder.shutdown();
}

} // namespace facebook::velox::exec::test
