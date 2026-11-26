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

#include "velox/connectors/tpch/TpchConnector.h"
#include <folly/init/Init.h>
#include "gtest/gtest.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

DECLARE_int32(velox_tpch_text_pool_size_mb);

namespace facebook::velox::connector::tpch {
namespace {

using exec::test::PlanBuilder;
using velox::tpch::Table;

class TpchConnectorTest : public exec::test::OperatorTestBase {
 public:
  const std::string kTpchConnectorId = "test-tpch";

  void SetUp() override {
    FLAGS_velox_tpch_text_pool_size_mb = 10;
    OperatorTestBase::SetUp();
    TpchConnectorFactory factory;
    auto tpchConnector = factory.newConnector(
        kTpchConnectorId,
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>()));
    ConnectorRegistry::global().insert(
        tpchConnector->connectorId(), tpchConnector);
  }

  void TearDown() override {
    ConnectorRegistry::global().erase(kTpchConnectorId);
    OperatorTestBase::TearDown();
  }

  exec::Split makeTpchSplit(size_t totalParts = 1, size_t partNumber = 0)
      const {
    return exec::Split(
        std::make_shared<TpchConnectorSplit>(
            kTpchConnectorId, /*cacheable=*/true, totalParts, partNumber));
  }
};

// Simple scan of first 5 rows of "nation".
TEST_F(TpchConnectorTest, simple) {
  auto plan = PlanBuilder()
                  .tpchTableScan(
                      Table::TBL_NATION,
                      {"n_nationkey", "n_name", "n_regionkey", "n_comment"})
                  .limit(0, 5, false)
                  .planNode();

  auto expected = makeRowVector({
      // n_nationkey
      makeFlatVector<int64_t>({0, 1, 2, 3, 4}),
      // n_name
      makeFlatVector<StringView>({
          "ALGERIA",
          "ARGENTINA",
          "BRAZIL",
          "CANADA",
          "EGYPT",
      }),
      // n_regionkey
      makeFlatVector<int64_t>({0, 1, 1, 1, 4}),
      // n_comment
      makeFlatVector<StringView>({
          "furiously regular requests. platelets affix furious",
          "instructions wake quickly. final deposits haggle. final, silent theodolites ",
          "asymptotes use fluffily quickly bold instructions. slyly bold dependencies sleep carefully pending accounts",
          "ss deposits wake across the pending foxes. packages after the carefully bold requests integrate caref",
          "usly ironic, pending foxes. even, special instructions nag. sly, final foxes detect slyly fluffily ",
      }),
  });
  exec::test::AssertQueryBuilder(plan)
      .split(makeTpchSplit())
      .assertResults(expected);
}

// Extract single column from "nation".
TEST_F(TpchConnectorTest, singleColumn) {
  auto plan =
      PlanBuilder().tpchTableScan(Table::TBL_NATION, {"n_name"}).planNode();

  auto expected = makeRowVector({makeFlatVector<StringView>({
      "ALGERIA",       "ARGENTINA", "BRAZIL", "CANADA",
      "EGYPT",         "ETHIOPIA",  "FRANCE", "GERMANY",
      "INDIA",         "INDONESIA", "IRAN",   "IRAQ",
      "JAPAN",         "JORDAN",    "KENYA",  "MOROCCO",
      "MOZAMBIQUE",    "PERU",      "CHINA",  "ROMANIA",
      "SAUDI ARABIA",  "VIETNAM",   "RUSSIA", "UNITED KINGDOM",
      "UNITED STATES",
  })});
  exec::test::AssertQueryBuilder(plan)
      .split(makeTpchSplit())
      .assertResults(expected);
  EXPECT_EQ("n_name", plan->outputType()->nameOf(0));
}

// Check that aliases are correctly resolved.
TEST_F(TpchConnectorTest, singleColumnWithAlias) {
  const std::string aliasedName = "my_aliased_column_name";

  auto outputType = ROW(aliasedName, VARCHAR());
  auto plan =
      PlanBuilder()
          .startTableScan()
          .outputType(outputType)
          .tableHandle(
              std::make_shared<TpchTableHandle>(
                  kTpchConnectorId, Table::TBL_NATION))
          .assignments({
              {aliasedName, std::make_shared<TpchColumnHandle>("n_name")},
              {"other_name", std::make_shared<TpchColumnHandle>("n_name")},
              {"third_column",
               std::make_shared<TpchColumnHandle>("n_regionkey")},
          })
          .endTableScan()
          .limit(0, 1, false)
          .planNode();

  exec::test::AssertQueryBuilder(plan)
      .split(makeTpchSplit())
      .assertSingleResult(std::string("ALGERIA"));

  EXPECT_EQ(aliasedName, plan->outputType()->nameOf(0));
}

// Aggregation over a larger table.
TEST_F(TpchConnectorTest, simpleAggregation) {
  VELOX_ASSERT_THROW(
      std::make_shared<TpchTableHandle>(
          kTpchConnectorId, Table::TBL_SUPPLIER, -1),
      "Tpch scale factor must be non-negative");

  for (double scaleFactor : {0.01, 1.0, 5.0, 13.0}) {
    SCOPED_TRACE(scaleFactor);
    auto plan = PlanBuilder()
                    .startTableScan()
                    .outputType(ROW({}))
                    .tableHandle(
                        std::make_shared<TpchTableHandle>(
                            kTpchConnectorId, Table::TBL_SUPPLIER, scaleFactor))
                    .endTableScan()
                    .singleAggregation({}, {"count(1)"})
                    .planNode();

    exec::test::AssertQueryBuilder(plan)
        .split(makeTpchSplit())
        .assertSingleResult(
            static_cast<int64_t>(
                velox::tpch::getRowCount(Table::TBL_SUPPLIER, scaleFactor)));
  }
}

TEST_F(TpchConnectorTest, lineitemTinyRowCount) {
  // Lineitem rows are generated per order. Scale factor 0.01 has 60'175 of
  // them.
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(ROW({}))
                  .tableHandle(
                      std::make_shared<TpchTableHandle>(
                          kTpchConnectorId, Table::TBL_LINEITEM, 0.01))
                  .endTableScan()
                  .singleAggregation({}, {"count(1)"})
                  .planNode();

  const size_t numParts = 4;
  std::vector<std::shared_ptr<connector::ConnectorSplit>> splits;
  splits.reserve(numParts);
  for (size_t i = 0; i < numParts; ++i) {
    splits.push_back(
        std::make_shared<TpchConnectorSplit>(
            kTpchConnectorId, /*cacheable=*/true, numParts, i));
  }

  exec::test::AssertQueryBuilder(plan).splits(splits).assertSingleResult(
      int64_t{60'175});

  // A batch smaller than one order's lineitems still covers every order.
  for (int batchRows : {1, 2, 3, 7}) {
    SCOPED_TRACE(batchRows);
    exec::test::AssertQueryBuilder(plan)
        .splits(splits)
        .config(
            core::QueryConfig::kTableScanOutputBatchRowsOverride,
            std::to_string(batchRows))
        .assertSingleResult(int64_t{60'175});
  }
}

TEST_F(TpchConnectorTest, unknownColumn) {
  VELOX_ASSERT_USER_THROW(
      PlanBuilder()
          .tpchTableScan(Table::TBL_NATION, {"does_not_exist"})
          .planNode(),
      "Field not found: does_not_exist");
}

// Ensures that splits broken down using different configurations return the
// same dataset in the end.
TEST_F(TpchConnectorTest, multipleSplits) {
  auto plan = PlanBuilder()
                  .tpchTableScan(
                      Table::TBL_NATION,
                      {"n_nationkey", "n_name", "n_regionkey", "n_comment"})
                  .planNode();

  // Use a full read from a single split to use as the source of truth.
  auto fullResult = exec::test::AssertQueryBuilder(plan)
                        .split(makeTpchSplit())
                        .copyResults(pool());
  size_t nationRowCount = velox::tpch::getRowCount(Table::TBL_NATION, 1);
  EXPECT_EQ(nationRowCount, fullResult->size());

  // Run query with different number of parts, up until `totalParts >
  // nationRowCount` in which cases each split will return one or zero records.
  for (size_t totalParts = 1; totalParts < (nationRowCount + 5); ++totalParts) {
    std::vector<exec::Split> splits;
    splits.reserve(totalParts);

    for (size_t i = 0; i < totalParts; ++i) {
      splits.emplace_back(makeTpchSplit(totalParts, i));
    }

    exec::test::AssertQueryBuilder(plan)
        .splits(std::move(splits))
        .assertResults(fullResult);
  }
}

// Test an equality filter in the TpchConnector.
TEST_F(TpchConnectorTest, filterPushdown) {
  auto plan = PlanBuilder(pool())
                  .tpchTableScan(
                      Table::TBL_NATION,
                      {"n_nationkey", "n_name", "n_regionkey"},
                      1.0,
                      kTpchConnectorId,
                      "n_regionkey = 1")
                  .planNode();

  // Should only return nations with regionkey = 1
  auto expected = makeRowVector({
      // n_nationkey
      makeFlatVector<int64_t>({1, 2, 3, 17, 24}),
      // n_name
      makeFlatVector<StringView>({
          "ARGENTINA",
          "BRAZIL",
          "CANADA",
          "PERU",
          "UNITED STATES",
      }),
      // n_regionkey
      makeFlatVector<int64_t>({1, 1, 1, 1, 1}),
  });
  exec::test::AssertQueryBuilder(plan)
      .split(makeTpchSplit())
      .assertResults(expected);
}

// Test a conjunction of range filters in the TpchConnector.
TEST_F(TpchConnectorTest, complexFilterPushdown) {
  auto plan = PlanBuilder(pool())
                  .tpchTableScan(
                      Table::TBL_NATION,
                      {"n_nationkey", "n_name", "n_regionkey"},
                      1.0,
                      kTpchConnectorId,
                      "n_nationkey < 5 AND n_regionkey > 0")
                  .planNode();

  // Should only return nations with nationkey < 5 AND regionkey > 0
  auto expected = makeRowVector({
      // n_nationkey
      makeFlatVector<int64_t>({1, 2, 3, 4}),
      // n_name
      makeFlatVector<StringView>({
          "ARGENTINA",
          "BRAZIL",
          "CANADA",
          "EGYPT",
      }),
      // n_regionkey
      makeFlatVector<int64_t>({1, 1, 1, 4}),
  });
  exec::test::AssertQueryBuilder(plan)
      .split(makeTpchSplit())
      .assertResults(expected);
}

// Test filtering with LIKE operator
TEST_F(TpchConnectorTest, likeFilterPushdown) {
  auto plan = PlanBuilder(pool())
                  .tpchTableScan(
                      Table::TBL_NATION,
                      {"n_nationkey", "n_name", "n_regionkey"},
                      1.0,
                      kTpchConnectorId,
                      "n_name LIKE 'A%'")
                  .planNode();

  // Should only return nations with names starting with 'A'
  auto expected = makeRowVector({
      // n_nationkey
      makeFlatVector<int64_t>({0, 1}),
      // n_name
      makeFlatVector<StringView>({"ALGERIA", "ARGENTINA"}),
      // n_regionkey
      makeFlatVector<int64_t>({0, 1}),
  });
  exec::test::AssertQueryBuilder(plan)
      .split(makeTpchSplit())
      .assertResults(expected);
}

// Test filtering with IN operator
TEST_F(TpchConnectorTest, inFilterPushdown) {
  auto plan = PlanBuilder(pool())
                  .tpchTableScan(
                      Table::TBL_NATION,
                      {"n_nationkey", "n_name", "n_regionkey"},
                      1.0,
                      kTpchConnectorId,
                      "n_nationkey IN (0, 5, 10, 15, 20)")
                  .planNode();

  // Should only return nations with nationkey in the specified list
  auto expected = makeRowVector({
      // n_nationkey
      makeFlatVector<int64_t>({0, 5, 10, 15, 20}),
      // n_name
      makeFlatVector<StringView>({
          "ALGERIA",
          "ETHIOPIA",
          "IRAN",
          "MOROCCO",
          "SAUDI ARABIA",
      }),
      // n_regionkey
      makeFlatVector<int64_t>({0, 0, 4, 0, 4}),
  });
  exec::test::AssertQueryBuilder(plan)
      .split(makeTpchSplit())
      .assertResults(expected);
}

TEST_F(TpchConnectorTest, namespaceInfer) {
  std::vector<std::pair<double, std::string>> expect = {
      {0.05, "tiny.customer"},
      {1.0, "sf1.customer"},
      {5.0, "sf5.customer"},
      {10.0, "sf10.customer"},
      {100.0, "sf100.customer"},
      {300.0, "sf300.customer"},
      {10000.0, "sf10000.customer"},
  };
  for (const auto& [scaleFactor, name] : expect) {
    SCOPED_TRACE(scaleFactor);
    auto handle = std::make_shared<TpchTableHandle>(
        kTpchConnectorId, Table::TBL_CUSTOMER, scaleFactor);
    EXPECT_EQ(handle->name(), name);
  }
}

// Join nation and region.
TEST_F(TpchConnectorTest, join) {
  PlanBuilder builder;
  core::PlanNodeId nationScanId;
  core::PlanNodeId regionScanId;
  auto plan = builder.newPlan()
                  .tpchTableScan(
                      Table::TBL_NATION, {"n_regionkey"}, 1.0 /*scaleFactor*/)
                  .capturePlanNodeId(nationScanId)
                  .hashJoin(
                      "n_regionkey = r_regionkey",
                      builder.newPlan()
                          .tpchTableScan(
                              Table::TBL_REGION,
                              {"r_regionkey", "r_name"},
                              1.0 /*scaleFactor*/)
                          .capturePlanNodeId(regionScanId)
                          .planNode(),
                      {"r_name"})
                  .singleAggregation({"r_name"}, {"count(1) as nation_cnt"})
                  .planNode();

  exec::test::AssertQueryBuilder(plan)
      .split(nationScanId, makeTpchSplit())
      .split(regionScanId, makeTpchSplit())
      .assertResults(makeRowVector({
          makeFlatVector<StringView>(
              {"AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"}),
          makeConstant<int64_t>(5, 5),
      }));
}

TEST_F(TpchConnectorTest, orderDateCount) {
  auto plan = PlanBuilder()
                  .tpchTableScan(Table::TBL_ORDERS, {"o_orderdate"}, 0.01)
                  .filter("o_orderdate = '1992-01-01'::DATE")
                  .limit(0, 10, false)
                  .planNode();

  // Scale factor 0.01 has 9 orders on that date.
  const auto orderDate = DATE()->toDays("1992-01-01");
  exec::test::AssertQueryBuilder(plan)
      .split(makeTpchSplit())
      .assertResults(makeRowVector({makeConstant(orderDate, 9, DATE())}));
}

TEST_F(TpchConnectorTest, config) {
  std::unordered_map<std::string, std::string> properties = {
      {"property", "value"}};
  TpchConnectorFactory factory;
  auto connector = factory.newConnector(
      kTpchConnectorId,
      std::make_shared<config::ConfigBase>(std::move(properties)));

  const auto& config = connector->connectorConfig();
  auto value = config->get<std::string>("property");
  EXPECT_TRUE(value.has_value());
  EXPECT_EQ(value.value(), "value");
}

} // namespace
} // namespace facebook::velox::connector::tpch

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
