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

#include "velox/exec/IndexLookupJoin.h"
#include "fmt/format.h"
#include "folly/synchronization/EventCount.h"
#include "gmock/gmock.h"
#include "gtest/gtest-matchers.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/IndexLookupJoinTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TestIndexStorageConnector.h"
#include "velox/vector/LazyVector.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::testutil;

namespace facebook::velox::exec::test {
namespace {
struct TestParam {
  bool asyncLookup;
  int32_t numPrefetches;
  bool serialExecution;
  bool hasNullKeys;
  bool needsIndexSplit;

  TestParam(
      bool _asyncLookup,
      int32_t _numPrefetches,
      bool _serialExecution,
      bool _hasNullKeys,
      bool _needsIndexSplit = false)
      : asyncLookup(_asyncLookup),
        numPrefetches(_numPrefetches),
        serialExecution(_serialExecution),
        hasNullKeys(_hasNullKeys),
        needsIndexSplit(_needsIndexSplit) {}

  std::string toString() const {
    return fmt::format(
        "asyncLookup={}, numPrefetches={}, serialExecution={}, hasNullKeys={}, needsIndexSplit={}",
        asyncLookup,
        numPrefetches,
        serialExecution,
        hasNullKeys,
        needsIndexSplit);
  }
};

class IndexLookupJoinTest : public IndexLookupJoinTestBase,
                            public testing::WithParamInterface<TestParam> {
 public:
  static std::vector<TestParam> getTestParams() {
    std::vector<TestParam> testParams;
    for (bool asyncLookup : {false, true}) {
      for (int numPrefetches : {0, 3}) {
        for (bool serialExecution : {false, true}) {
          for (bool hasNullKeys : {false, true}) {
            for (bool needsIndexSplit : {false, true}) {
              // Serial execution doesn't support index split as it requires
              // single-threaded execution which is incompatible with the
              // split-based parallelism used by index lookup join.
              if (serialExecution && needsIndexSplit) {
                continue;
              }
              testParams.emplace_back(
                  asyncLookup,
                  numPrefetches,
                  serialExecution,
                  hasNullKeys,
                  needsIndexSplit);
            }
          }
        }
      }
    }
    return testParams;
  }

 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    core::PlanNode::registerSerDe();
    connector::hive::HiveColumnHandle::registerSerDe();
    Type::registerSerDe();
    core::ITypedExpr::registerSerDe();
    TestIndexConnectorFactory::registerConnector(connectorCpuExecutor_.get());

    keyType_ = ROW({"u0", "u1", "u2"}, {BIGINT(), BIGINT(), BIGINT()});
    valueType_ = ROW({"u3", "u4", "u5"}, {BIGINT(), BIGINT(), VARCHAR()});
    tableType_ = concat(keyType_, valueType_);
    probeType_ = ROW(
        {"t0", "t1", "t2", "t3", "t4", "t5"},
        {BIGINT(), BIGINT(), BIGINT(), BIGINT(), ARRAY(BIGINT()), VARCHAR()});
  }

  void TearDown() override {
    connector::ConnectorRegistry::global().erase(kTestIndexConnectorName);
    HiveConnectorTestBase::TearDown();
  }

  void testSerde(const core::PlanNodePtr& plan) {
    auto serialized = plan->serialize();
    auto copy = ISerializable::deserialize<core::PlanNode>(serialized, pool());
    ASSERT_EQ(plan->toString(true, true), copy->toString(true, true));
  }

  // Makes index table handle with the specified index table and async lookup
  // flag.
  static std::shared_ptr<TestIndexTableHandle> makeIndexTableHandle(
      const std::shared_ptr<TestIndexTable>& indexTable,
      bool asyncLookup,
      bool needsIndexSplit = false) {
    return std::make_shared<TestIndexTableHandle>(
        kTestIndexConnectorName, indexTable, asyncLookup, needsIndexSplit);
  }

  static connector::ColumnHandleMap makeIndexColumnHandles(
      const std::vector<std::string>& names) {
    connector::ColumnHandleMap handles;
    for (const auto& name : names) {
      handles.emplace(name, std::make_shared<TestIndexColumnHandle>(name));
    }

    return handles;
  }

  const std::unique_ptr<folly::CPUThreadPoolExecutor> connectorCpuExecutor_{
      std::make_unique<folly::CPUThreadPoolExecutor>(128)};
};

TEST_F(IndexLookupJoinTest, joinCondition) {
  const auto rowType =
      ROW({"c0", "c1", "c2", "c3", "c4"},
          {BIGINT(), BIGINT(), BIGINT(), ARRAY(BIGINT()), BIGINT()});

  auto inJoinCondition = PlanBuilder::parseIndexJoinCondition(
      "contains(c3, c2)", rowType, pool_.get());
  ASSERT_FALSE(inJoinCondition->isFilter());
  ASSERT_EQ(inJoinCondition->toString(), "ROW[\"c2\"] IN ROW[\"c3\"]");

  auto inFilterCondition = PlanBuilder::parseIndexJoinCondition(
      "contains(ARRAY[1,2], c2)", rowType, pool_.get());
  ASSERT_TRUE(inFilterCondition->isFilter());
  ASSERT_EQ(inFilterCondition->toString(), "ROW[\"c2\"] IN {1, 2}");

  auto betweenFilterCondition = PlanBuilder::parseIndexJoinCondition(
      "c0 between 0 AND 1", rowType, pool_.get());
  ASSERT_TRUE(betweenFilterCondition->isFilter());
  ASSERT_EQ(betweenFilterCondition->toString(), "ROW[\"c0\"] BETWEEN 0 AND 1");

  auto betweenJoinCondition1 = PlanBuilder::parseIndexJoinCondition(
      "c0 between c1 AND c4", rowType, pool_.get());
  ASSERT_FALSE(betweenJoinCondition1->isFilter());
  ASSERT_EQ(
      betweenJoinCondition1->toString(),
      "ROW[\"c0\"] BETWEEN ROW[\"c1\"] AND ROW[\"c4\"]");

  auto betweenJoinCondition2 = PlanBuilder::parseIndexJoinCondition(
      "c0 between 0 AND c1", rowType, pool_.get());
  ASSERT_FALSE(betweenJoinCondition2->isFilter());
  ASSERT_EQ(
      betweenJoinCondition2->toString(),
      "ROW[\"c0\"] BETWEEN 0 AND ROW[\"c1\"]");

  auto betweenJoinCondition3 = PlanBuilder::parseIndexJoinCondition(
      "c0 between c1 AND 0", rowType, pool_.get());
  ASSERT_FALSE(betweenJoinCondition3->isFilter());
  ASSERT_EQ(
      betweenJoinCondition3->toString(),
      "ROW[\"c0\"] BETWEEN ROW[\"c1\"] AND 0");

  auto equalFilterCondition =
      PlanBuilder::parseIndexJoinCondition("c0=1", rowType, pool_.get());
  ASSERT_TRUE(equalFilterCondition->isFilter());
  ASSERT_EQ(equalFilterCondition->toString(), "ROW[\"c0\"] = 1");

  auto equalJoinCondition =
      PlanBuilder::parseIndexJoinCondition("c0=c1", rowType, pool_.get());
  ASSERT_FALSE(equalJoinCondition->isFilter());
  ASSERT_EQ(equalJoinCondition->toString(), "ROW[\"c0\"] = ROW[\"c1\"]");

  auto equalJoinConditionAsFilter =
      PlanBuilder::parseIndexJoinCondition("c0=1", rowType, pool_.get());
  ASSERT_TRUE(equalJoinConditionAsFilter->isFilter());
  ASSERT_EQ(equalJoinConditionAsFilter->toString(), "ROW[\"c0\"] = 1");
}

TEST_P(IndexLookupJoinTest, planNodeAndSerde) {
  TestIndexTableHandle::registerSerDe();

  auto indexConnectorHandle =
      makeIndexTableHandle(nullptr, true, GetParam().needsIndexSplit);

  auto left = makeRowVector(
      {"t0", "t1", "t2", "t3", "t4"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({10, 20, 30}),
       makeFlatVector<int64_t>({10, 30, 20}),
       makeArrayVector<int64_t>(
           3,
           [](auto row) { return row; },
           [](auto /*unused*/, auto index) { return index; }),
       makeArrayVector<int64_t>(
           3,
           [](auto row) { return row; },
           [](auto /*unused*/, auto index) { return index; })});

  auto right = makeRowVector(
      {"u0", "u1", "u2"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({10, 20, 30}),
       makeFlatVector<int64_t>({10, 30, 20})});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto planBuilder = PlanBuilder();
  auto nonIndexTableScan = std::dynamic_pointer_cast<const core::TableScanNode>(
      PlanBuilder::TableScanBuilder(planBuilder)
          .outputType(asRowType(right->type()))
          .endTableScan()
          .planNode());
  VELOX_CHECK_NOT_NULL(nonIndexTableScan);

  auto indexTableScan = std::dynamic_pointer_cast<const core::TableScanNode>(
      PlanBuilder::TableScanBuilder(planBuilder)
          .tableHandle(indexConnectorHandle)
          .outputType(asRowType(right->type()))
          .endTableScan()
          .planNode());
  VELOX_CHECK_NOT_NULL(indexTableScan);

  for (const auto joinType : {core::JoinType::kLeft, core::JoinType::kInner}) {
    auto plan = PlanBuilder(planNodeIdGenerator)
                    .values({left})
                    .startIndexLookupJoin()
                    .leftKeys({"t0"})
                    .rightKeys({"u0"})
                    .indexSource(indexTableScan)
                    .outputLayout({"t0", "u1", "t2", "t1"})
                    .joinType(joinType)
                    .endIndexLookupJoin()
                    .planNode();
    auto indexLookupJoinNode =
        std::dynamic_pointer_cast<const core::IndexLookupJoinNode>(plan);
    ASSERT_EQ(
        indexLookupJoinNode->needsIndexSplit(), GetParam().needsIndexSplit);
    ASSERT_TRUE(indexLookupJoinNode->joinConditions().empty());
    ASSERT_EQ(
        indexLookupJoinNode->lookupSource()->tableHandle()->connectorId(),
        kTestIndexConnectorName);
    testSerde(plan);
  }

  // with in join conditions.
  for (const auto joinType : {core::JoinType::kLeft, core::JoinType::kInner}) {
    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({left})
                    .startIndexLookupJoin()
                    .leftKeys({"t0"})
                    .rightKeys({"u0"})
                    .indexSource(indexTableScan)
                    .joinConditions({"contains(t3, u0)", "contains(t4, u1)"})
                    .outputLayout({"t0", "u1", "t2", "t1"})
                    .joinType(joinType)
                    .endIndexLookupJoin()
                    .planNode();
    auto indexLookupJoinNode =
        std::dynamic_pointer_cast<const core::IndexLookupJoinNode>(plan);
    ASSERT_EQ(
        indexLookupJoinNode->needsIndexSplit(), GetParam().needsIndexSplit);
    ASSERT_EQ(indexLookupJoinNode->joinConditions().size(), 2);
    ASSERT_EQ(
        indexLookupJoinNode->lookupSource()->tableHandle()->connectorId(),
        kTestIndexConnectorName);
    testSerde(plan);
  }

  // with between join conditions.
  for (const auto joinType : {core::JoinType::kLeft, core::JoinType::kInner}) {
    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({left})
                    .startIndexLookupJoin()
                    .leftKeys({"t0"})
                    .rightKeys({"u0"})
                    .indexSource(indexTableScan)
                    .joinConditions(
                        {"u0 between t0 AND t1",
                         "u1 between t1 AND 10",
                         "u1 between 10 AND t1"})
                    .outputLayout({"t0", "u1", "t2", "t1"})
                    .joinType(joinType)
                    .endIndexLookupJoin()
                    .planNode();
    auto indexLookupJoinNode =
        std::dynamic_pointer_cast<const core::IndexLookupJoinNode>(plan);
    ASSERT_EQ(
        indexLookupJoinNode->needsIndexSplit(), GetParam().needsIndexSplit);
    ASSERT_EQ(indexLookupJoinNode->joinConditions().size(), 3);
    ASSERT_EQ(
        indexLookupJoinNode->lookupSource()->tableHandle()->connectorId(),
        kTestIndexConnectorName);
    testSerde(plan);
  }

  // with mix join conditions.
  for (const auto joinType : {core::JoinType::kLeft, core::JoinType::kInner}) {
    auto plan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({left})
            .startIndexLookupJoin()
            .leftKeys({"t0"})
            .rightKeys({"u0"})
            .indexSource(indexTableScan)
            .joinConditions({"contains(t3, u0)", "u1 between 10 AND t1"})
            .outputLayout({"t0", "u1", "t2", "t1"})
            .joinType(joinType)
            .endIndexLookupJoin()
            .planNode();
    auto indexLookupJoinNode =
        std::dynamic_pointer_cast<const core::IndexLookupJoinNode>(plan);
    ASSERT_EQ(
        indexLookupJoinNode->needsIndexSplit(), GetParam().needsIndexSplit);
    ASSERT_EQ(indexLookupJoinNode->joinConditions().size(), 2);
    ASSERT_EQ(
        indexLookupJoinNode->lookupSource()->tableHandle()->connectorId(),
        kTestIndexConnectorName);
    testSerde(plan);
  }

  // with has match column.
  {
    auto plan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({left})
            .startIndexLookupJoin()
            .leftKeys({"t0"})
            .rightKeys({"u0"})
            .indexSource(indexTableScan)
            .joinConditions({"contains(t3, u0)", "u1 between 10 AND t1"})
            .hasMarker(true)
            .outputLayout({"t0", "u1", "t2", "t1", "match"})
            .joinType(core::JoinType::kLeft)
            .endIndexLookupJoin()
            .planNode();
    auto indexLookupJoinNode =
        std::dynamic_pointer_cast<const core::IndexLookupJoinNode>(plan);
    ASSERT_EQ(
        indexLookupJoinNode->needsIndexSplit(), GetParam().needsIndexSplit);
    ASSERT_EQ(indexLookupJoinNode->joinConditions().size(), 2);
    ASSERT_EQ(indexLookupJoinNode->filter(), nullptr);
    ASSERT_EQ(
        indexLookupJoinNode->lookupSource()->tableHandle()->connectorId(),
        kTestIndexConnectorName);
    testSerde(plan);
  }

  // with filter.
  for (const auto joinType : {core::JoinType::kLeft, core::JoinType::kInner}) {
    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({left})
                    .indexLookupJoin(
                        {"t0"},
                        {"u0"},
                        indexTableScan,
                        {},
                        /*filter=*/"t1 % 2 = 0",
                        /*hasMarker=*/false,
                        {"t0", "u1", "t2", "t1"},
                        joinType)
                    .planNode();
    auto indexLookupJoinNode =
        std::dynamic_pointer_cast<const core::IndexLookupJoinNode>(plan);
    ASSERT_TRUE(indexLookupJoinNode->joinConditions().empty());
    ASSERT_NE(indexLookupJoinNode->filter(), nullptr);
    ASSERT_EQ(
        indexLookupJoinNode->needsIndexSplit(), GetParam().needsIndexSplit);
    ASSERT_EQ(
        indexLookupJoinNode->filter()->toString(), "eq(mod(ROW[\"t1\"],2),0)");
    ASSERT_EQ(
        indexLookupJoinNode->lookupSource()->tableHandle()->connectorId(),
        kTestIndexConnectorName);
    testSerde(plan);
  }

  // with join conditions and filter.
  for (const auto joinType : {core::JoinType::kLeft, core::JoinType::kInner}) {
    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({left})
                    .indexLookupJoin(
                        {"t0"},
                        {"u0"},
                        indexTableScan,
                        {"contains(t3, u0)"},
                        /*filter=*/"u1 % 2 = 0 AND t2 > 5",
                        /*hasMarker=*/false,
                        {"t0", "u1", "t2", "t1"},
                        joinType)
                    .planNode();
    auto indexLookupJoinNode =
        std::dynamic_pointer_cast<const core::IndexLookupJoinNode>(plan);
    ASSERT_EQ(indexLookupJoinNode->joinConditions().size(), 1);
    ASSERT_EQ(
        indexLookupJoinNode->needsIndexSplit(), GetParam().needsIndexSplit);
    ASSERT_NE(indexLookupJoinNode->filter(), nullptr);
    ASSERT_EQ(
        indexLookupJoinNode->filter()->toString(),
        "and(eq(mod(ROW[\"u1\"],2),0),gt(ROW[\"t2\"],5))");
    ASSERT_EQ(
        indexLookupJoinNode->lookupSource()->tableHandle()->connectorId(),
        kTestIndexConnectorName);
    testSerde(plan);
  }

  // with filter and marker for left join.
  {
    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({left})
                    .indexLookupJoin(
                        {"t0"},
                        {"u0"},
                        indexTableScan,
                        {"u1 between 10 AND t1"},
                        /*filter=*/"t2 < u2",
                        /*hasMarker=*/true,
                        {"t0", "u1", "t2", "t1", "match"},
                        core::JoinType::kLeft)
                    .planNode();
    auto indexLookupJoinNode =
        std::dynamic_pointer_cast<const core::IndexLookupJoinNode>(plan);
    ASSERT_EQ(indexLookupJoinNode->joinConditions().size(), 1);
    ASSERT_EQ(
        indexLookupJoinNode->needsIndexSplit(), GetParam().needsIndexSplit);
    ASSERT_NE(indexLookupJoinNode->filter(), nullptr);
    ASSERT_EQ(
        indexLookupJoinNode->filter()->toString(),
        "lt(ROW[\"t2\"],ROW[\"u2\"])");
    ASSERT_TRUE(indexLookupJoinNode->hasMarker());
    ASSERT_EQ(
        indexLookupJoinNode->lookupSource()->tableHandle()->connectorId(),
        kTestIndexConnectorName);
    testSerde(plan);
  }

  // with complex filter expression.
  {
    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({left})
                    .startIndexLookupJoin()
                    .leftKeys({"t0"})
                    .rightKeys({"u0"})
                    .indexSource(indexTableScan)
                    .filter("(t1 + u1) * 2 > 100 OR t2 = u2")
                    .outputLayout({"t0", "u1", "t2", "t1"})
                    .joinType(core::JoinType::kInner)
                    .endIndexLookupJoin()
                    .planNode();
    auto indexLookupJoinNode =
        std::dynamic_pointer_cast<const core::IndexLookupJoinNode>(plan);
    ASSERT_TRUE(indexLookupJoinNode->joinConditions().empty());
    ASSERT_EQ(
        indexLookupJoinNode->needsIndexSplit(), GetParam().needsIndexSplit);
    ASSERT_NE(indexLookupJoinNode->filter(), nullptr);
    ASSERT_EQ(
        indexLookupJoinNode->filter()->toString(),
        "or(gt(multiply(plus(ROW[\"t1\"],ROW[\"u1\"]),2),100),eq(ROW[\"t2\"],ROW[\"u2\"]))");
    ASSERT_EQ(
        indexLookupJoinNode->lookupSource()->tableHandle()->connectorId(),
        kTestIndexConnectorName);
    testSerde(plan);
  }

  // with splitOutput.
  for (const auto splitOutput :
       {std::optional<bool>(true),
        std::optional<bool>(false),
        std::optional<bool>(std::nullopt)}) {
    auto plan = PlanBuilder(planNodeIdGenerator)
                    .values({left})
                    .startIndexLookupJoin()
                    .leftKeys({"t0"})
                    .rightKeys({"u0"})
                    .indexSource(indexTableScan)
                    .outputLayout({"t0", "u1", "t2", "t1"})
                    .joinType(core::JoinType::kInner)
                    .splitOutput(splitOutput)
                    .endIndexLookupJoin()
                    .planNode();
    auto indexLookupJoinNode =
        std::dynamic_pointer_cast<const core::IndexLookupJoinNode>(plan);
    ASSERT_EQ(indexLookupJoinNode->splitOutput(), splitOutput);
    testSerde(plan);
  }

  // bad join type.
  {
    VELOX_ASSERT_USER_THROW(
        PlanBuilder(planNodeIdGenerator)
            .values({left})
            .startIndexLookupJoin()
            .leftKeys({"t0"})
            .rightKeys({"u0"})
            .indexSource(indexTableScan)
            .outputLayout({"t0", "u1", "t2", "t1"})
            .joinType(core::JoinType::kFull)
            .endIndexLookupJoin()
            .planNode(),
        "Unsupported index lookup join type FULL");
  }

  // bad table handle.
  {
    VELOX_ASSERT_USER_THROW(
        PlanBuilder(planNodeIdGenerator)
            .values({left})
            .startIndexLookupJoin()
            .leftKeys({"t0"})
            .rightKeys({"u0"})
            .indexSource(nonIndexTableScan)
            .outputLayout({"t0", "u1", "t2", "t1"})
            .endIndexLookupJoin()
            .planNode(),
        "The lookup table handle hive_table from connector test-hive doesn't support index lookup");
  }

  // Non-matched join keys.
  {
    VELOX_ASSERT_THROW(
        PlanBuilder(planNodeIdGenerator)
            .values({left})
            .startIndexLookupJoin()
            .leftKeys({"t0", "t1"})
            .rightKeys({"u0"})
            .indexSource(indexTableScan)
            .joinConditions({"contains(t4, u0)"})
            .outputLayout({"t0", "u1", "t2", "t1"})
            .endIndexLookupJoin()
            .planNode(),
        "The index lookup join node requires same number of join keys on left and right sides");
  }

  // No join keys.
  {
    VELOX_ASSERT_THROW(
        PlanBuilder(planNodeIdGenerator)
            .values({left})
            .startIndexLookupJoin()
            .leftKeys({})
            .rightKeys({})
            .indexSource(indexTableScan)
            .joinConditions({"contains(t4, u0)"})
            .outputLayout({"t0", "u1", "t2", "t1"})
            .endIndexLookupJoin()
            .planNode(),
        "The index lookup join node requires at least one join key");
  }
}

TEST_F(IndexLookupJoinTest, pairedBetweenJoinCondition) {
  TestIndexTableHandle::registerSerDe();
  auto indexConnectorHandle = makeIndexTableHandle(nullptr, true);

  // Probe input shaped to cover all paired BETWEEN scenarios:
  //   sid           BIGINT             - left key
  //   event_types   ARRAY<INTEGER>     - IN list + BETWEEN pairedColumn
  //   regions       ARRAY<VARCHAR>     - second IN list + second pairedColumn
  //   lower_ts      ARRAY<BIGINT>      - BETWEEN lower bound array
  //   upper_ts      ARRAY<BIGINT>      - BETWEEN upper bound array
  //   lower_users   ARRAY<INTEGER>     - second BETWEEN lower bound array
  //   upper_users   ARRAY<INTEGER>     - second BETWEEN upper bound array
  auto left = makeRowVector(
      {"sid",
       "event_types",
       "regions",
       "lower_ts",
       "upper_ts",
       "lower_users",
       "upper_users"},
      {makeFlatVector<int64_t>({1}),
       makeArrayVector<int32_t>(
           1, [](auto) { return 1; }, [](auto, auto idx) { return idx; }),
       makeArrayVector<StringView>(
           1,
           [](auto) { return 1; },
           [](auto, auto /*idx*/) { return StringView("us"); }),
       makeArrayVector<int64_t>(
           1, [](auto) { return 1; }, [](auto, auto idx) { return idx; }),
       makeArrayVector<int64_t>(
           1, [](auto) { return 1; }, [](auto, auto idx) { return idx + 100; }),
       makeArrayVector<int32_t>(
           1, [](auto) { return 1; }, [](auto, auto idx) { return idx; }),
       makeArrayVector<int32_t>(
           1,
           [](auto) { return 1; },
           [](auto, auto idx) { return idx + 100; })});

  auto right = makeRowVector(
      {"u_sid", "event_type", "region", "ts", "ts2", "user_count"},
      {makeFlatVector<int64_t>({1}),
       makeFlatVector<int32_t>({1}),
       makeFlatVector<StringView>({StringView("us")}),
       makeFlatVector<int64_t>({100}),
       makeFlatVector<int64_t>({200}),
       makeFlatVector<int32_t>({50})});

  auto planBuilder = PlanBuilder();
  auto indexTableScan = std::dynamic_pointer_cast<const core::TableScanNode>(
      PlanBuilder::TableScanBuilder(planBuilder)
          .tableHandle(indexConnectorHandle)
          .outputType(asRowType(right->type()))
          .endTableScan()
          .planNode());

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto probeNode = PlanBuilder(planNodeIdGenerator).values({left}).planNode();

  auto eventTypesField = std::make_shared<core::FieldAccessTypedExpr>(
      ARRAY(INTEGER()), "event_types");
  auto lowerTsField =
      std::make_shared<core::FieldAccessTypedExpr>(ARRAY(BIGINT()), "lower_ts");
  auto upperTsField =
      std::make_shared<core::FieldAccessTypedExpr>(ARRAY(BIGINT()), "upper_ts");

  auto inCondition = std::make_shared<core::InIndexLookupCondition>(
      std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "event_type"),
      eventTypesField);

  auto buildPlan = [&](std::vector<core::IndexLookupConditionPtr> conditions) {
    return std::make_shared<core::IndexLookupJoinNode>(
        planNodeIdGenerator->next(),
        core::JoinType::kInner,
        std::vector<core::FieldAccessTypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "sid")},
        std::vector<core::FieldAccessTypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "u_sid")},
        std::move(conditions),
        /*filter=*/nullptr,
        /*hasMarker=*/false,
        probeNode,
        indexTableScan,
        ROW({"sid", "ts"}, {BIGINT(), BIGINT()}));
  };

  // Happy path: paired BETWEEN with a matching IN sibling. Serde round-trips
  // the pairedColumn field.
  {
    auto pairedBetween = std::make_shared<core::BetweenIndexLookupCondition>(
        std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts"),
        lowerTsField,
        upperTsField,
        eventTypesField);
    auto plan = buildPlan({inCondition, pairedBetween});
    ASSERT_EQ(plan->joinConditions().size(), 2);
    const auto between =
        std::dynamic_pointer_cast<const core::BetweenIndexLookupCondition>(
            plan->joinConditions()[1]);
    ASSERT_NE(between->pairedColumn, nullptr);
    ASSERT_EQ(between->pairedColumn->name(), "event_types");
    testSerde(plan);
  }

  // Reject: paired BETWEEN with a scalar (non-ARRAY) lower bound. The
  // BetweenIndexLookupCondition constructor's validate() catches it.
  {
    auto scalarLower =
        std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "sid");
    VELOX_ASSERT_THROW(
        std::make_shared<core::BetweenIndexLookupCondition>(
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts"),
            scalarLower,
            upperTsField,
            eventTypesField),
        "Paired between lower bound must be of type ARRAY");
  }

  // Reject: paired BETWEEN with a constant ARRAY lower bound. Constant bounds
  // are disallowed because they would force the per-row size check to
  // special-case "constant array size vs varying IN-list size"; requiring
  // FieldAccess keeps the invariant uniform across rows.
  {
    auto constantLower = std::make_shared<core::ConstantTypedExpr>(
        ARRAY(BIGINT()),
        velox::Variant::array({
            velox::Variant(int64_t(10)),
            velox::Variant(int64_t(20)),
        }));
    VELOX_ASSERT_THROW(
        std::make_shared<core::BetweenIndexLookupCondition>(
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts"),
            constantLower,
            upperTsField,
            eventTypesField),
        "Paired between lower bound must be a column reference");
  }

  // Reject: paired BETWEEN with a constant ARRAY upper bound. Mirror of the
  // case above — confirms both bounds are checked.
  {
    auto constantUpper = std::make_shared<core::ConstantTypedExpr>(
        ARRAY(BIGINT()),
        velox::Variant::array({
            velox::Variant(int64_t(100)),
            velox::Variant(int64_t(200)),
        }));
    VELOX_ASSERT_THROW(
        std::make_shared<core::BetweenIndexLookupCondition>(
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts"),
            lowerTsField,
            constantUpper,
            eventTypesField),
        "Paired between upper bound must be a column reference");
  }

  // Reject: paired BETWEEN without a matching IN sibling. The
  // IndexLookupJoinNode constructor's cross-condition check catches it.
  {
    auto pairedBetween = std::make_shared<core::BetweenIndexLookupCondition>(
        std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts"),
        lowerTsField,
        upperTsField,
        eventTypesField);
    VELOX_ASSERT_THROW(
        buildPlan({pairedBetween}),
        "must also appear as the list of a sibling IN condition");
  }

  // Happy path: two paired BETWEENs paired with different IN columns. Each
  // BETWEEN's pairedColumn names its own partner, so the validation handles
  // them independently.
  {
    auto regionsField = std::make_shared<core::FieldAccessTypedExpr>(
        ARRAY(VARCHAR()), "regions");
    auto lowerUsersField = std::make_shared<core::FieldAccessTypedExpr>(
        ARRAY(INTEGER()), "lower_users");
    auto upperUsersField = std::make_shared<core::FieldAccessTypedExpr>(
        ARRAY(INTEGER()), "upper_users");
    auto inRegions = std::make_shared<core::InIndexLookupCondition>(
        std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "region"),
        regionsField);
    auto betweenTs = std::make_shared<core::BetweenIndexLookupCondition>(
        std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts"),
        lowerTsField,
        upperTsField,
        eventTypesField);
    auto betweenUsers = std::make_shared<core::BetweenIndexLookupCondition>(
        std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "user_count"),
        lowerUsersField,
        upperUsersField,
        regionsField);
    auto plan = buildPlan({inCondition, inRegions, betweenTs, betweenUsers});
    ASSERT_EQ(plan->joinConditions().size(), 4);
    testSerde(plan);
  }

  // Happy path: two paired BETWEENs sharing the same paired column. One IN
  // condition is enough to satisfy both.
  {
    auto betweenTs = std::make_shared<core::BetweenIndexLookupCondition>(
        std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts"),
        lowerTsField,
        upperTsField,
        eventTypesField);
    auto betweenTs2 = std::make_shared<core::BetweenIndexLookupCondition>(
        std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts2"),
        lowerTsField,
        upperTsField,
        eventTypesField);
    auto plan = buildPlan({inCondition, betweenTs, betweenTs2});
    ASSERT_EQ(plan->joinConditions().size(), 3);
    testSerde(plan);
  }

  // Happy path: a paired BETWEEN and an unpaired BETWEEN in the same join.
  // The unpaired BETWEEN's bounds are scalars; it does not need any IN
  // sibling. The cross-condition check should only fire for the paired one.
  {
    auto pairedBetween = std::make_shared<core::BetweenIndexLookupCondition>(
        std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts"),
        lowerTsField,
        upperTsField,
        eventTypesField);
    auto unpairedBetween = std::make_shared<core::BetweenIndexLookupCondition>(
        std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts2"),
        std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "sid"),
        std::make_shared<core::ConstantTypedExpr>(
            BIGINT(), velox::Variant(int64_t(1000))));
    auto plan = buildPlan({inCondition, pairedBetween, unpairedBetween});
    ASSERT_EQ(plan->joinConditions().size(), 3);
    testSerde(plan);
  }
}

// Exercises the operator's runtime path for paired BETWEEN:
// IndexLookupJoin::addBetweenConditionBound is called with isPaired=true,
// which validates that each bound's probe-input column has type
// ARRAY(indexKeyType). Plan-construction tests above don't reach this code
// path because they stop at validate()/serde without running the operator.
TEST_F(IndexLookupJoinTest, pairedBetweenOperatorRuntime) {
  TestIndexTableHandle::registerSerDe();

  // TestIndexTable requires every column referenced by a join condition's key
  // to live in keyData (beyond the numEqualJoinKeys prefix). `event_type` (IN
  // key) and `ts` (BETWEEN key) go in keyData; valueData carries a placeholder
  // because valueType must have at least one column.
  auto keyData = makeRowVector(
      {"u_sid", "event_type", "ts"},
      {makeFlatVector<int64_t>({1}),
       makeFlatVector<int32_t>({1}),
       makeFlatVector<int64_t>({150})});
  auto valueData = makeRowVector({"_unused"}, {makeFlatVector<int64_t>({0})});
  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/1, keyData, valueData, *pool());
  const auto indexTableHandle =
      makeIndexTableHandle(indexTable, /*asyncLookup=*/true);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto scanOutputType =
      ROW({"u_sid", "event_type", "ts"}, {BIGINT(), INTEGER(), BIGINT()});
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      scanOutputType,
      makeIndexColumnHandles({"u_sid", "event_type", "ts"}));

  auto probeInput = makeRowVector(
      {"sid", "event_types", "lower_ts", "upper_ts"},
      {makeFlatVector<int64_t>({1}),
       makeArrayVector<int32_t>(
           1, [](auto) { return 1; }, [](auto, auto) { return 1; }),
       makeArrayVector<int64_t>(
           1, [](auto) { return 1; }, [](auto, auto) { return 100; }),
       makeArrayVector<int64_t>(
           1, [](auto) { return 1; }, [](auto, auto) { return 200; })});
  auto probeNode =
      PlanBuilder(planNodeIdGenerator).values({probeInput}).planNode();

  auto eventTypesField = std::make_shared<core::FieldAccessTypedExpr>(
      ARRAY(INTEGER()), "event_types");
  std::vector<core::IndexLookupConditionPtr> conditions = {
      std::make_shared<core::InIndexLookupCondition>(
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "event_type"),
          eventTypesField),
      std::make_shared<core::BetweenIndexLookupCondition>(
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts"),
          std::make_shared<core::FieldAccessTypedExpr>(
              ARRAY(BIGINT()), "lower_ts"),
          std::make_shared<core::FieldAccessTypedExpr>(
              ARRAY(BIGINT()), "upper_ts"),
          eventTypesField),
  };

  auto plan = std::make_shared<core::IndexLookupJoinNode>(
      planNodeIdGenerator->next(),
      core::JoinType::kInner,
      std::vector<core::FieldAccessTypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "sid")},
      std::vector<core::FieldAccessTypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "u_sid")},
      std::move(conditions),
      /*filter=*/nullptr,
      /*hasMarker=*/false,
      probeNode,
      indexScanNode,
      ROW({"sid", "ts"}, {BIGINT(), BIGINT()}));

  // Running the plan drives the operator's initLookupInput, which calls
  // addBetweenConditionBound for both `lower_ts` and `upper_ts` with
  // isPaired=true. The test passes if the operator's type check accepts
  // ARRAY(BIGINT) for both bounds against the BIGINT index key `ts`.
  ASSERT_NO_THROW(
      exec::test::AssertQueryBuilder(plan).copyResults(pool_.get()));
}

// Null in any of the IN list / lower / upper probe columns: the row is
// deselected upstream by decodeAndDetectNonNullKeys (every paired-BETWEEN
// probe column is in lookupKeyOrConditionHashers_), so
// validatePairedBetweenSizes never sees it. Each sub-case nulls a different
// column to prove the upstream filter covers all three; if a future refactor
// drops one from the hasher set this test flips red.
TEST_F(IndexLookupJoinTest, pairedBetweenNullRowSkipped) {
  TestIndexTableHandle::registerSerDe();

  auto keyData = makeRowVector(
      {"u_sid", "event_type", "ts"},
      {makeFlatVector<int64_t>({1, 2}),
       makeFlatVector<int32_t>({1, 1}),
       makeFlatVector<int64_t>({150, 150})});
  auto valueData =
      makeRowVector({"_unused"}, {makeFlatVector<int64_t>({0, 0})});
  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/1, keyData, valueData, *pool());
  const auto indexTableHandle =
      makeIndexTableHandle(indexTable, /*asyncLookup=*/true);

  // For each null-column choice, row 0 is valid (sizes 1/1/1) and row 1 has
  // a null in the named column with bound sizes that would mismatch if the
  // size check ever ran on row 1. The runtime should not throw.
  enum class NullColumn { kIn, kLower, kUpper };
  auto runWithNullColumn = [&](NullColumn nullColumn) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        ROW({"u_sid", "event_type", "ts"}, {BIGINT(), INTEGER(), BIGINT()}),
        makeIndexColumnHandles({"u_sid", "event_type", "ts"}));

    // Row 1's non-null arrays are size 2; the would-be IN-list size for that
    // row (if it weren't null) is 1. Asymmetric sizes ensure any branch that
    // reads row 1 hits a mismatch. Use makeArrayVector in the non-null branch
    // to avoid an overload-resolution ambiguity with makeNullableArrayVector
    // when no std::nullopt is present.
    auto eventTypes = nullColumn == NullColumn::kIn
        ? makeNullableArrayVector<int32_t>({{{1}}, std::nullopt})
        : makeArrayVector<int32_t>({{1}, {1}});
    auto lowerTs = nullColumn == NullColumn::kLower
        ? makeNullableArrayVector<int64_t>({{{100}}, std::nullopt})
        : makeArrayVector<int64_t>({{100}, {100, 110}});
    auto upperTs = nullColumn == NullColumn::kUpper
        ? makeNullableArrayVector<int64_t>({{{200}}, std::nullopt})
        : makeArrayVector<int64_t>({{200}, {200, 210}});

    auto probeInput = makeRowVector(
        {"sid", "event_types", "lower_ts", "upper_ts"},
        {makeFlatVector<int64_t>({1, 2}), eventTypes, lowerTs, upperTs});
    auto probeNode =
        PlanBuilder(planNodeIdGenerator).values({probeInput}).planNode();

    auto eventTypesField = std::make_shared<core::FieldAccessTypedExpr>(
        ARRAY(INTEGER()), "event_types");
    std::vector<core::IndexLookupConditionPtr> conditions = {
        std::make_shared<core::InIndexLookupCondition>(
            std::make_shared<core::FieldAccessTypedExpr>(
                INTEGER(), "event_type"),
            eventTypesField),
        std::make_shared<core::BetweenIndexLookupCondition>(
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts"),
            std::make_shared<core::FieldAccessTypedExpr>(
                ARRAY(BIGINT()), "lower_ts"),
            std::make_shared<core::FieldAccessTypedExpr>(
                ARRAY(BIGINT()), "upper_ts"),
            eventTypesField),
    };

    auto plan = std::make_shared<core::IndexLookupJoinNode>(
        planNodeIdGenerator->next(),
        core::JoinType::kInner,
        std::vector<core::FieldAccessTypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "sid")},
        std::vector<core::FieldAccessTypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "u_sid")},
        std::move(conditions),
        /*filter=*/nullptr,
        /*hasMarker=*/false,
        probeNode,
        indexScanNode,
        ROW({"sid", "ts"}, {BIGINT(), BIGINT()}));

    ASSERT_NO_THROW(
        exec::test::AssertQueryBuilder(plan).copyResults(pool_.get()));
  };

  runWithNullColumn(NullColumn::kIn);
  runWithNullColumn(NullColumn::kLower);
  runWithNullColumn(NullColumn::kUpper);
}

// Null rows must not mask a real size mismatch elsewhere in the same batch.
// Row 0 is valid, row 1 has a null IN list (gets filtered upstream), and
// row 2 has non-null arrays whose sizes disagree. The operator must still
// throw and must name row 2 in the error so the offending row is identifiable.
// Guards against a future refactor that accidentally short-circuits the whole
// check when any row in the batch is null.
TEST_F(IndexLookupJoinTest, pairedBetweenNullDoesNotMaskMismatch) {
  TestIndexTableHandle::registerSerDe();

  auto keyData = makeRowVector(
      {"u_sid", "event_type", "ts"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int32_t>({1, 1, 1}),
       makeFlatVector<int64_t>({150, 150, 150})});
  auto valueData =
      makeRowVector({"_unused"}, {makeFlatVector<int64_t>({0, 0, 0})});
  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/1, keyData, valueData, *pool());
  const auto indexTableHandle =
      makeIndexTableHandle(indexTable, /*asyncLookup=*/true);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      ROW({"u_sid", "event_type", "ts"}, {BIGINT(), INTEGER(), BIGINT()}),
      makeIndexColumnHandles({"u_sid", "event_type", "ts"}));

  // Row 0: valid (sizes 1/1/1).
  // Row 1: event_types null — upstream filter drops this row.
  // Row 2: event_types size 2, lower/upper size 1 — must trigger the throw.
  auto probeInput = makeRowVector(
      {"sid", "event_types", "lower_ts", "upper_ts"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeNullableArrayVector<int32_t>({{{1}}, std::nullopt, {{1, 2}}}),
       makeArrayVector<int64_t>({{100}, {100}, {100}}),
       makeArrayVector<int64_t>({{200}, {200}, {200}})});
  auto probeNode =
      PlanBuilder(planNodeIdGenerator).values({probeInput}).planNode();

  auto eventTypesField = std::make_shared<core::FieldAccessTypedExpr>(
      ARRAY(INTEGER()), "event_types");
  std::vector<core::IndexLookupConditionPtr> conditions = {
      std::make_shared<core::InIndexLookupCondition>(
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "event_type"),
          eventTypesField),
      std::make_shared<core::BetweenIndexLookupCondition>(
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts"),
          std::make_shared<core::FieldAccessTypedExpr>(
              ARRAY(BIGINT()), "lower_ts"),
          std::make_shared<core::FieldAccessTypedExpr>(
              ARRAY(BIGINT()), "upper_ts"),
          eventTypesField),
  };

  auto plan = std::make_shared<core::IndexLookupJoinNode>(
      planNodeIdGenerator->next(),
      core::JoinType::kInner,
      std::vector<core::FieldAccessTypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "sid")},
      std::vector<core::FieldAccessTypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "u_sid")},
      std::move(conditions),
      /*filter=*/nullptr,
      /*hasMarker=*/false,
      probeNode,
      indexScanNode,
      ROW({"sid", "ts"}, {BIGINT(), BIGINT()}));

  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan).copyResults(pool_.get()),
      "row 2: IN list size 2 vs lower bound size 1");
}

// Per-row size invariant: the IN list and BETWEEN bound arrays must agree in
// length on every row, because the connector pairs the i-th IN-list element
// with the i-th lower/upper bound. The operator catches a mismatch in
// prepareLookup before dispatching to the connector. Each sub-case targets a
// different shape of mismatch (IN vs lower, IN vs upper, multi-row,
// empty-array) to make sure no direction or batch position slips through.
TEST_F(IndexLookupJoinTest, pairedBetweenSizeMismatchRejected) {
  TestIndexTableHandle::registerSerDe();

  auto keyData = makeRowVector(
      {"u_sid", "event_type", "ts"},
      {makeFlatVector<int64_t>({1, 2}),
       makeFlatVector<int32_t>({1, 1}),
       makeFlatVector<int64_t>({150, 150})});
  auto valueData =
      makeRowVector({"_unused"}, {makeFlatVector<int64_t>({0, 0})});
  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/1, keyData, valueData, *pool());
  const auto indexTableHandle =
      makeIndexTableHandle(indexTable, /*asyncLookup=*/true);

  // Builds a one-row probe + plan for a given (inSize, lowerSize, upperSize)
  // triple. Returns the plan; caller asserts the expected throw.
  auto buildOneRowPlan = [&](vector_size_t inSize,
                             vector_size_t lowerSize,
                             vector_size_t upperSize) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        ROW({"u_sid", "event_type", "ts"}, {BIGINT(), INTEGER(), BIGINT()}),
        makeIndexColumnHandles({"u_sid", "event_type", "ts"}));
    auto probeInput = makeRowVector(
        {"sid", "event_types", "lower_ts", "upper_ts"},
        {makeFlatVector<int64_t>({1}),
         makeArrayVector<int32_t>(
             1,
             [inSize](auto) { return inSize; },
             [](auto, auto idx) { return idx + 1; }),
         makeArrayVector<int64_t>(
             1,
             [lowerSize](auto) { return lowerSize; },
             [](auto, auto) { return 100; }),
         makeArrayVector<int64_t>(
             1,
             [upperSize](auto) { return upperSize; },
             [](auto, auto) { return 200; })});
    auto probeNode =
        PlanBuilder(planNodeIdGenerator).values({probeInput}).planNode();
    auto eventTypesField = std::make_shared<core::FieldAccessTypedExpr>(
        ARRAY(INTEGER()), "event_types");
    std::vector<core::IndexLookupConditionPtr> conditions = {
        std::make_shared<core::InIndexLookupCondition>(
            std::make_shared<core::FieldAccessTypedExpr>(
                INTEGER(), "event_type"),
            eventTypesField),
        std::make_shared<core::BetweenIndexLookupCondition>(
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts"),
            std::make_shared<core::FieldAccessTypedExpr>(
                ARRAY(BIGINT()), "lower_ts"),
            std::make_shared<core::FieldAccessTypedExpr>(
                ARRAY(BIGINT()), "upper_ts"),
            eventTypesField),
    };
    return std::make_shared<core::IndexLookupJoinNode>(
        planNodeIdGenerator->next(),
        core::JoinType::kInner,
        std::vector<core::FieldAccessTypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "sid")},
        std::vector<core::FieldAccessTypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "u_sid")},
        std::move(conditions),
        /*filter=*/nullptr,
        /*hasMarker=*/false,
        probeNode,
        indexScanNode,
        ROW({"sid", "ts"}, {BIGINT(), BIGINT()}));
  };

  // 1. IN list larger than lower bound (IN=2, lower=1, upper=1). Lower is
  //    checked first, so the error reports the lower-bound size mismatch.
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(buildOneRowPlan(2, 1, 1))
          .copyResults(pool_.get()),
      "IN list size 2 vs lower bound size 1");

  // 2. IN list smaller than lower bound (IN=1, lower=2, upper=1). Same lower
  //    check fires, but with the inequality reversed.
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(buildOneRowPlan(1, 2, 1))
          .copyResults(pool_.get()),
      "IN list size 1 vs lower bound size 2");

  // 3. Upper bound mismatches IN, lower matches (IN=1, lower=1, upper=2).
  //    Lower passes, so the failure must come from the upper-bound check.
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(buildOneRowPlan(1, 1, 2))
          .copyResults(pool_.get()),
      "IN list size 1 vs upper bound size 2");

  // 4. IN agrees with lower but not upper (IN=2, lower=2, upper=3). Confirms
  //    we don't only catch when the inequality bracket is "lower != IN".
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(buildOneRowPlan(2, 2, 3))
          .copyResults(pool_.get()),
      "IN list size 2 vs upper bound size 3");

  // 5. Empty IN list with a non-empty bound (IN=0, lower=1, upper=1).
  //    Catches the boundary case where the IN list has no elements.
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(buildOneRowPlan(0, 1, 1))
          .copyResults(pool_.get()),
      "IN list size 0 vs lower bound size 1");

  // 6. Mismatch on the second row of a multi-row batch (row 0 OK, row 1 bad).
  //    Confirms the check iterates all rows, not just the first, and the
  //    error message names the offending row index.
  {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        ROW({"u_sid", "event_type", "ts"}, {BIGINT(), INTEGER(), BIGINT()}),
        makeIndexColumnHandles({"u_sid", "event_type", "ts"}));
    auto probeInput = makeRowVector(
        {"sid", "event_types", "lower_ts", "upper_ts"},
        {makeFlatVector<int64_t>({1, 2}),
         // Row 0: IN size 1; row 1: IN size 2.
         makeArrayVector<int32_t>(
             2,
             [](auto row) { return row == 0 ? 1 : 2; },
             [](auto, auto idx) { return idx + 1; }),
         // Both rows have lower/upper size 1 — row 1 is the mismatch.
         makeArrayVector<int64_t>(
             2, [](auto) { return 1; }, [](auto, auto) { return 100; }),
         makeArrayVector<int64_t>(
             2, [](auto) { return 1; }, [](auto, auto) { return 200; })});
    auto probeNode =
        PlanBuilder(planNodeIdGenerator).values({probeInput}).planNode();
    auto eventTypesField = std::make_shared<core::FieldAccessTypedExpr>(
        ARRAY(INTEGER()), "event_types");
    std::vector<core::IndexLookupConditionPtr> conditions = {
        std::make_shared<core::InIndexLookupCondition>(
            std::make_shared<core::FieldAccessTypedExpr>(
                INTEGER(), "event_type"),
            eventTypesField),
        std::make_shared<core::BetweenIndexLookupCondition>(
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "ts"),
            std::make_shared<core::FieldAccessTypedExpr>(
                ARRAY(BIGINT()), "lower_ts"),
            std::make_shared<core::FieldAccessTypedExpr>(
                ARRAY(BIGINT()), "upper_ts"),
            eventTypesField),
    };
    auto plan = std::make_shared<core::IndexLookupJoinNode>(
        planNodeIdGenerator->next(),
        core::JoinType::kInner,
        std::vector<core::FieldAccessTypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "sid")},
        std::vector<core::FieldAccessTypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "u_sid")},
        std::move(conditions),
        /*filter=*/nullptr,
        /*hasMarker=*/false,
        probeNode,
        indexScanNode,
        ROW({"sid", "ts"}, {BIGINT(), BIGINT()}));
    VELOX_ASSERT_THROW(
        exec::test::AssertQueryBuilder(plan).copyResults(pool_.get()),
        "row 1: IN list size 2 vs lower bound size 1");
  }
}

TEST_P(IndexLookupJoinTest, DISABLED_equalJoin) {
  struct {
    std::vector<int> keyCardinalities;
    int numProbeBatches;
    int numRowsPerProbeBatch;
    int matchPct;
    std::vector<std::string> scanOutputColumns;
    std::vector<std::string> outputColumns;
    core::JoinType joinType;
    std::string duckDbVerifySql;

    std::string debugString() const {
      return fmt::format(
          "keyCardinalities: {}, numProbeBatches: {}, numRowsPerProbeBatch: {}, matchPct: {}, "
          "scanOutputColumns: {}, outputColumns: {}, joinType: {},"
          " duckDbVerifySql: {}",
          folly::join(",", keyCardinalities),
          numProbeBatches,
          numRowsPerProbeBatch,
          matchPct,
          folly::join(",", scanOutputColumns),
          folly::join(",", outputColumns),
          core::JoinTypeName::toName(joinType),
          duckDbVerifySql);
    }
  } testSettings[] = {
      // Inner join.
      // 10% match.
      {{100, 1, 1},
       10,
       100,
       10,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      // 10% match with larger lookup table.
      {{500, 1, 1},
       10,
       100,
       10,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      {{2048, 1, 1},
       10,
       100,
       10,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      // Empty lookup table.
      {{0, 1, 1},
       10,
       100,
       10,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      // No match.
      {{500, 1, 1},
       10,
       100,
       0,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      // 10% matcvelox::h with larger lookup table.
      {{500, 1, 1},
       10,
       100,
       10,
       {"u0", "u1", "u2", "u3"},
       {"t1", "u1", "u2", "u3"},
       core::JoinType::kInner,
       "SELECT t.c1, u.c1, u.c2, u.c3 FROM t, u WHERE t.c0 = u.c0"},
      {{2048, 1, 1},
       10,
       100,
       10,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      // very few (2%) match with larger lookup table.
      {{500, 1, 1},
       10,
       100,
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      {{2048, 1, 1},
       10,
       100,
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      // All matches with larger lookup table.
      {{500, 1, 1},
       10,
       100,
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      {{2048, 1, 1},
       10,
       100,
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      // No probe projection.
      {{500, 1, 1},
       10,
       100,
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      // Probe column reorder in output.
      {{500, 1, 1},
       10,
       100,
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t2", "t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c2, t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      {{2048, 1, 1},
       10,
       100,
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t2", "t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c2, t.c1, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      // Both sides reorder in output.
      {{500, 1, 1},
       10,
       100,
       2,
       {"u1", "u0", "u2", "u3", "u5"},
       {"t2", "u2", "u3", "t1", "u1", "u5"},
       core::JoinType::kInner,
       "SELECT t.c2, u.c2, u.c3, t.c1, u.c1, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      {{2048, 1, 1},
       10,
       100,
       2,
       {"u1", "u0", "u2", "u3", "u5"},
       {"t2", "u2", "u3", "t1", "u1", "u5"},
       core::JoinType::kInner,
       "SELECT t.c2, u.c2, u.c3, t.c1, u.c1, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      // With probe key colums.
      {{500, 1, 1},
       2,
       100,
       2,
       {"u1", "u0", "u2", "u3", "u5"},
       {"t2", "u2", "u3", "t1", "u1", "t0", "u5"},
       core::JoinType::kInner,
       "SELECT t.c2, u.c2, u.c3, t.c1, u.c1, t.c0, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      {{2048, 1, 1},
       2,
       100,
       2,
       {"u1", "u0", "u2", "u3", "u5"},
       {"t2", "u2", "u3", "t1", "u1", "t0", "u5"},
       core::JoinType::kInner,
       "SELECT t.c2, u.c2, u.c3, t.c1, u.c1, t.c0, u.c5 FROM t, u WHERE t.c0 = u.c0"},
      // Project key columns from lookup table.
      {{2048, 1, 1},
       10,
       100,
       50,
       {"u1", "u0", "u2", "u3"},
       {"t2", "u2", "u3", "t1", "u1", "u0"},
       core::JoinType::kInner,
       "SELECT t.c2, u.c2, u.c3, t.c1, u.c1, u.c0 FROM t, u WHERE t.c0 = u.c0"},
      {{100, 1, 1},
       10,
       100,
       50,
       {"u1", "u0", "u2", "u3"},
       {"t2", "u2", "u3", "t1", "u1", "u0"},
       core::JoinType::kInner,
       "SELECT t.c2, u.c2, u.c3, t.c1, u.c1, u.c0 FROM t, u WHERE t.c0 = u.c0"},
      {{2048, 1, 1},
       10,
       2048,
       100,
       {"u0", "u1", "u2", "u3"},
       {"t1", "u1", "u2", "u3"},
       core::JoinType::kInner,
       "SELECT t.c1, u.c1, u.c2, u.c3 FROM t, u WHERE t.c0 = u.c0"},

      // Left join.
      // 10% match.
      {{100, 1, 1},
       10,
       100,
       10,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      // 10% match with larger lookup table.
      {{500, 1, 1},
       10,
       100,
       10,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      {{2048, 1, 1},
       10,
       100,
       10,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      // Empty lookup table.
      {{0, 1, 1},
       10,
       100,
       10,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      // No match.
      {{500, 1, 1},
       10,
       100,
       0,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      // 10% match with larger lookup table.
      {{500, 1, 1},
       10,
       100,
       10,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      // very few (2%) match with larger lookup table.
      {{500, 1, 1},
       10,
       100,
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      {{2048, 1, 1},
       10,
       100,
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      // All matches with larger lookup table.
      {{500, 1, 1},
       10,
       100,
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      {{2048, 1, 1},
       10,
       100,
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      // Probe column reorder in output.
      {{500, 1, 1},
       10,
       100,
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t2", "t1", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c2, t.c1, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      // Lookup column reorder in output.
      {{500, 1, 1},
       10,
       100,
       2,
       {"u1", "u0", "u2", "u5"},
       {"t1", "u2", "u1", "t2", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c1, u.c2, u.c1, t.c2, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      // Both sides reorder in output.
      {{500, 1, 1},
       10,
       100,
       2,
       {"u1", "u0", "u2", "u3", "u5"},
       {"t2", "u2", "u3", "t1", "u1", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c2, u.c2, u.c3, t.c1, u.c1, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      // With probe key colums.
      {{500, 1, 1},
       10,
       100,
       2,
       {"u1", "u0", "u2", "u3", "u5"},
       {"t2", "u2", "u3", "t1", "u1", "t0", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c2, u.c2, u.c3, t.c1, u.c1, t.c0, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      // With lookup key colums.
      {{500, 1, 1},
       10,
       100,
       2,
       {"u1", "u0", "u2", "u3", "u5"},
       {"t2", "u2", "u3", "t1", "u1", "u0"},
       core::JoinType::kLeft,
       "SELECT t.c2, u.c2, u.c3, t.c1, u.c1, u.c0 FROM t LEFT JOIN u ON t.c0 = u.c0"},
      {{2048, 1, 1},
       10,
       2048,
       100,
       {"u0", "u1", "u2", "u3"},
       {"t1", "u1", "u2", "u3"},
       core::JoinType::kLeft,
       "SELECT t.c1, u.c1, u.c2, u.c3 FROM t LEFT JOIN u ON t.c0 = u.c0"}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    IndexTableData tableData;
    generateIndexTableData(testData.keyCardinalities, tableData, pool_);
    auto probeVectors = generateProbeInput(
        testData.numProbeBatches,
        testData.numRowsPerProbeBatch,
        1,
        tableData,
        pool_,
        {"t0", "t1", "t2"},
        GetParam().hasNullKeys,
        {},
        {},
        testData.matchPct);
    std::vector<std::shared_ptr<TempFilePath>> probeFiles =
        createProbeFiles(probeVectors);

    createDuckDbTable("t", probeVectors);
    createDuckDbTable("u", {tableData.tableVectors});

    const auto indexTable = TestIndexTable::create(
        /*numEqualJoinKeys=*/3,
        tableData.keyVectors,
        tableData.valueVectors,
        *pool());
    const auto indexTableHandle = makeIndexTableHandle(
        indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        makeScanOutputType(testData.scanOutputColumns),
        makeIndexColumnHandles(testData.scanOutputColumns));

    auto plan = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0", "t1", "t2"},
        {"u0", "u1", "u2"},
        {},
        /*filter=*/"",
        /*hasMarker=*/false,
        testData.joinType,
        testData.outputColumns);
    runLookupQuery(
        plan,
        probeFiles,
        GetParam().serialExecution,
        GetParam().serialExecution,
        32,
        GetParam().numPrefetches,
        GetParam().needsIndexSplit,
        testData.duckDbVerifySql);
    if (testData.joinType != core::JoinType::kLeft) {
      continue;
    }
    const auto probeScanId = probeScanNodeId_;
    auto planWithMatchColumn = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0", "t1", "t2"},
        {"u0", "u1", "u2"},
        /*joinConditions=*/{},
        /*filter=*/"",
        /*hasMarker=*/true,
        testData.joinType,
        testData.outputColumns);
    verifyResultWithMatchColumn(
        plan,
        probeScanId,
        planWithMatchColumn,
        probeScanNodeId_,
        probeFiles,
        GetParam().needsIndexSplit);
  }
}

TEST_P(IndexLookupJoinTest, betweenJoinCondition) {
  struct {
    std::vector<int> keyCardinalities;
    int numProbeBatches;
    int numProbeRowsPerBatch;
    std::string betweenCondition;
    int betweenMatchPct;
    std::vector<std::string> lookupOutputColumns;
    std::vector<std::string> outputColumns;
    core::JoinType joinType;
    std::string duckDbVerifySql;

    std::string debugString() const {
      return fmt::format(
          "keyCardinalities: {}, numProbeBatches: {}, numProbeRowsPerBatch: {}, betweenCondition: {}, betweenMatchPct: {}, lookupOutputColumns: {}, outputColumns: {}, joinType: {}, duckDbVerifySql: {}",
          folly::join(",", keyCardinalities),
          numProbeBatches,
          numProbeRowsPerBatch,
          betweenCondition,
          betweenMatchPct,
          folly::join(",", lookupOutputColumns),
          folly::join(",", outputColumns),
          core::JoinTypeName::toName(joinType),
          duckDbVerifySql);
    }
  } testSettings[] = {// Inner join.
                      // 10% match.
                      {{50, 1, 10},
                       1,
                       100,
                       "u2 between t2 and t3",
                       10,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      {{50, 1, 10},
                       1,
                       100,
                       "u2 between 0 and t3",
                       10,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN 0 AND t.c3"},
                      {{50, 1, 10},
                       1,
                       100,
                       "u2 between t2 and 1",
                       10,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND 1"},
                      // 10% match with larger lookup table.
                      {{256, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       10,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      {{256, 1, 10},
                       10,
                       100,
                       "u2 between 0 and t3",
                       10,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN 0 AND t.c3"},
                      {{256, 1, 10},
                       10,
                       100,
                       "u2 between t2 and 1",
                       10,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND 1"},
                      // Empty lookup table.
                      {{0, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       10,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      // No match.
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       0,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between 0 and t3",
                       0,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN 0 AND t.c3"},
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and 0",
                       0,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND 0"},
                      // very few (2%) match with larger lookup table.
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       2,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      {{256, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       2,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      // All matches
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       100,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      // All matches with larger lookup table.
                      {{256, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       100,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      {{256, 1, 10},
                       10,
                       100,
                       "u2 between 0 and t3",
                       100,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN 0 AND t.c3"},
                      {{256, 1, 10},
                       10,
                       100,
                       "u2 between t2 and 10",
                       100,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kInner,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND 10"},
                      // No probe projection.
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       10,
                       {"u0", "u1", "u2", "u3", "u5"},
                       {"u1", "u0", "u3", "u5"},
                       core::JoinType::kInner,
                       "SELECT u.c1, u.c0, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      // Probe column reorder in output.
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       10,
                       {"u0", "u1", "u2", "u3", "u5"},
                       {"t2", "t1", "u1", "u3", "u5"},
                       core::JoinType::kInner,
                       "SELECT t.c2, t.c1, u.c1, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      // Both sides reorder in output.
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       10,
                       {"u1", "u0", "u2", "u3", "u5"},
                       {"t2", "u3", "t1", "u1", "u5"},
                       core::JoinType::kInner,
                       "SELECT t.c2, u.c3, t.c1, u.c1, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      // With probe key colums.
                      {{50, 1, 10},
                       2,
                       100,
                       "u2 between t2 and t3",
                       10,
                       {"u1", "u0", "u2", "u3", "u5"},
                       {"t2", "u3", "t1", "u1", "u0", "u5"},
                       core::JoinType::kInner,
                       "SELECT t.c2, u.c3, t.c1, u.c1, u.c0, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},

                      // Left join.
                      // 10% match.
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       10,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between 0 and t3",
                       10,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN 0 AND t.c3"},
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and 1",
                       10,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND 1"},
                      // 10% match with larger lookup table.
                      {{256, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       10,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      {{256, 1, 10},
                       10,
                       100,
                       "u2 between 0 and t3",
                       10,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN 0 AND t.c3"},
                      {{256, 1, 10},
                       10,
                       100,
                       "u2 between t2 and 1",
                       10,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND 1"},
                      // Empty lookup table.
                      {{0, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       10,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      // No match.
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       0,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between 0 and t3",
                       0,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN 0 AND t.c3"},
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and 0",
                       0,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND 0"},
                      // very few (2%) match with larger lookup table.
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       2,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      {{256, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       2,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      // All matches.
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       2,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      // All matches with larger lookup table.
                      {{256, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       2,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      {{256, 1, 10},
                       10,
                       100,
                       "u2 between 0 and t3",
                       2,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN 0 AND t.c3"},
                      {{256, 1, 10},
                       10,
                       100,
                       "u2 between t2 and 10",
                       2,
                       {"u0", "u1", "u2", "u3"},
                       {"t0", "t1", "t2", "t3", "u3", "t5"},
                       core::JoinType::kLeft,
                       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND 10"},
                      // Probe column reorder in output.
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       2,
                       {"u0", "u1", "u2", "u3", "u5"},
                       {"t2", "t1", "u1", "u3", "u5"},
                       core::JoinType::kLeft,
                       "SELECT t.c2, t.c1, u.c1, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      // Both sides reorder in output.
                      {{50, 1, 10},
                       10,
                       100,
                       "u2 between t2 and t3",
                       2,
                       {"u1", "u0", "u2", "u3", "u5"},
                       {"t2", "u3", "t1", "u1", "u5"},
                       core::JoinType::kLeft,
                       "SELECT t.c2, u.c3, t.c1, u.c1, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"},
                      // With probe key colums.
                      {{50, 1, 10},
                       2,
                       100,
                       "u2 between t2 and t3",
                       2,
                       {"u1", "u0", "u2", "u3", "u5"},
                       {"t2", "u3", "t1", "u1", "u0", "u5"},
                       core::JoinType::kLeft,
                       "SELECT t.c2, u.c3, t.c1, u.c1, u.c0, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND u.c2 BETWEEN t.c2 AND t.c3"}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    IndexTableData tableData;
    generateIndexTableData(testData.keyCardinalities, tableData, pool_);
    auto probeVectors = generateProbeInput(
        testData.numProbeBatches,
        testData.numProbeRowsPerBatch,
        1,
        tableData,
        pool_,
        {"t0", "t1"},
        GetParam().hasNullKeys,
        {},
        {{"t2", "t3"}},
        /*equalMatchPct=*/80,
        /*inMatchPct=*/std::nullopt,
        testData.betweenMatchPct);
    std::vector<std::shared_ptr<TempFilePath>> probeFiles =
        createProbeFiles(probeVectors);

    createDuckDbTable("t", probeVectors);
    createDuckDbTable("u", {tableData.tableVectors});

    const auto indexTable = TestIndexTable::create(
        /*numEqualJoinKeys=*/2,
        tableData.keyVectors,
        tableData.valueVectors,
        *pool());
    const auto indexTableHandle = makeIndexTableHandle(
        indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        makeScanOutputType(testData.lookupOutputColumns),
        makeIndexColumnHandles(testData.lookupOutputColumns));

    auto plan = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0", "t1"},
        {"u0", "u1"},
        {testData.betweenCondition},
        /*filter=*/"",
        /*hasMarker=*/false,
        testData.joinType,
        testData.outputColumns);
    runLookupQuery(
        plan,
        probeFiles,
        GetParam().serialExecution,
        GetParam().serialExecution,
        32,
        GetParam().numPrefetches,
        GetParam().needsIndexSplit,
        testData.duckDbVerifySql);
    if (testData.joinType != core::JoinType::kLeft) {
      continue;
    }
    const auto probeScanId = probeScanNodeId_;
    auto planWithMatchColumn = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0", "t1"},
        {"u0", "u1"},
        {testData.betweenCondition},
        /*filter=*/"",
        /*hasMarker=*/true,
        testData.joinType,
        testData.outputColumns);
    verifyResultWithMatchColumn(
        plan,
        probeScanId,
        planWithMatchColumn,
        probeScanNodeId_,
        probeFiles,
        GetParam().needsIndexSplit);
  }
}

TEST_P(IndexLookupJoinTest, inJoinCondition) {
  struct {
    std::vector<int> keyCardinalities;
    int numProbeBatches;
    int numProbeRowsPerBatch;
    std::string inCondition;
    int inMatchPct;
    std::vector<std::string> lookupOutputColumns;
    std::vector<std::string> outputColumns;
    core::JoinType joinType;
    std::string duckDbVerifySql;

    std::string debugString() const {
      return fmt::format(
          "keyCardinalities: {}: numProbeBatches: {}, numProbeRowsPerBatch: {}, inCondition: {}, inMatchPct: {}, lookupOutputColumns: {}, outputColumns: {}, joinType: {}, duckDbVerifySql: {}",
          folly::join(",", keyCardinalities),
          numProbeBatches,
          numProbeRowsPerBatch,
          inCondition,
          inMatchPct,
          folly::join(",", lookupOutputColumns),
          folly::join(",", outputColumns),
          core::JoinTypeName::toName(joinType),
          duckDbVerifySql);
    }
  } testSettings[] = {
      // Inner join.
      // 10% match.
      {{50, 1, 10},
       1,
       100,
       "contains(t4, u2)",
       10,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kInner,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // 10% match with larger lookup table.
      {{256, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       10,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kInner,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // Empty lookup table.
      {{0, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       10,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kInner,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // No match.
      {{50, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       0,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kInner,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // very few (2%) match with larger lookup table.
      {{50, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       2,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kInner,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      {{256, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       2,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kInner,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // All matches
      {{50, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       100,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kInner,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // All matches with larger lookup table.
      {{256, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       100,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kInner,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // No probe projection.
      {{50, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       10,
       {"u0", "u1", "u2", "u3", "u5"},
       {"u1", "u0", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT u.c1, u.c0, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // Probe column reorder in output.
      {{50, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       10,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t2", "t1", "u1", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c2, t.c1, u.c1, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // Both sides reorder in output.
      {{50, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       10,
       {"u1", "u0", "u2", "u3", "u5"},
       {"t2", "u3", "t1", "u1", "u5"},
       core::JoinType::kInner,
       "SELECT t.c2, u.c3, t.c1, u.c1, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // With probe key colums.
      {{50, 1, 10},
       2,
       100,
       "contains(t4, u2)",
       10,
       {"u1", "u0", "u2", "u3", "u5"},
       {"t2", "u3", "t1", "u1", "u0", "u5"},
       core::JoinType::kInner,
       "SELECT t.c2, u.c3, t.c1, u.c1, u.c0, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},

      // Left join.
      // 10% match.
      {{50, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       10,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kLeft,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // 10% match with larger lookup table.
      {{256, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       10,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kLeft,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      {{256, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       10,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kLeft,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // Empty lookup table.
      {{0, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       10,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kLeft,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // No match.
      {{50, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       0,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kLeft,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // very few (2%) match with larger lookup table.
      {{50, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       2,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kLeft,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      {{256, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       2,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kLeft,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // All matches.
      {{50, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       2,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kLeft,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // All matches with larger lookup table.
      {{256, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       2,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t2", "t3", "u3", "t5"},
       core::JoinType::kLeft,
       "SELECT t.c0, t.c1, t.c2, t.c3, u.c3, t.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // Probe column reorder in output.
      {{50, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t2", "t1", "u1", "u3", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c2, t.c1, u.c1, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // Both sides reorder in output.
      {{50, 1, 10},
       10,
       100,
       "contains(t4, u2)",
       2,
       {"u1", "u0", "u2", "u3", "u5"},
       {"t2", "u3", "t1", "u1", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c2, u.c3, t.c1, u.c1, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"},
      // With probe key colums.
      {{50, 1, 10},
       2,
       100,
       "contains(t4, u2)",
       2,
       {"u1", "u0", "u2", "u3", "u5"},
       {"t2", "u3", "t1", "u1", "u0", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c2, u.c3, t.c1, u.c1, u.c0, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND array_contains(t.c4, u.c2)"}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    IndexTableData tableData;
    generateIndexTableData(testData.keyCardinalities, tableData, pool_);
    auto probeVectors = generateProbeInput(
        testData.numProbeBatches,
        testData.numProbeRowsPerBatch,
        1,
        tableData,
        pool_,
        {"t0", "t1"},
        GetParam().hasNullKeys,
        {{"t4"}},
        {},
        /*equalMatchPct=*/80,
        testData.inMatchPct);
    std::vector<std::shared_ptr<TempFilePath>> probeFiles =
        createProbeFiles(probeVectors);

    createDuckDbTable("t", probeVectors);
    createDuckDbTable("u", {tableData.tableVectors});

    const auto indexTable = TestIndexTable::create(
        /*numEqualJoinKeys=*/2,
        tableData.keyVectors,
        tableData.valueVectors,
        *pool());
    const auto indexTableHandle = makeIndexTableHandle(
        indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        makeScanOutputType(testData.lookupOutputColumns),
        makeIndexColumnHandles(testData.lookupOutputColumns));

    auto plan = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0", "t1"},
        {"u0", "u1"},
        {testData.inCondition},
        /*filter=*/"",
        /*hasMarker=*/false,
        testData.joinType,
        testData.outputColumns);
    runLookupQuery(
        plan,
        probeFiles,
        GetParam().serialExecution,
        GetParam().serialExecution,
        32,
        GetParam().numPrefetches,
        GetParam().needsIndexSplit,
        testData.duckDbVerifySql);
    if (testData.joinType != core::JoinType::kLeft) {
      continue;
    }
    const auto probeScanId = probeScanNodeId_;
    auto planWithMatchColumn = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0", "t1"},
        {"u0", "u1"},
        {testData.inCondition},
        /*filter=*/"",
        /*hasMarker=*/true,
        testData.joinType,
        testData.outputColumns);
    verifyResultWithMatchColumn(
        plan,
        probeScanId,
        planWithMatchColumn,
        probeScanNodeId_,
        probeFiles,
        GetParam().needsIndexSplit);
  }
}

TEST_P(IndexLookupJoinTest, prefixKeysEqualJoin) {
  struct {
    std::vector<int> keyCardinalities;
    int numProbeBatches;
    int numRowsPerProbeBatch;
    int matchPct;
    int numKeysToUse; // Number of keys to use from the full key set
    std::vector<std::string> scanOutputColumns;
    std::vector<std::string> outputColumns;
    core::JoinType joinType;
    std::string duckDbVerifySql;

    std::string debugString() const {
      return fmt::format(
          "keyCardinalities: {}, numProbeBatches: {}, numRowsPerProbeBatch: {}, matchPct: {}, numKeysToUse: {}, "
          "scanOutputColumns: {}, outputColumns: {}, joinType: {},"
          " duckDbVerifySql: {}",
          folly::join(",", keyCardinalities),
          numProbeBatches,
          numRowsPerProbeBatch,
          matchPct,
          numKeysToUse,
          folly::join(",", scanOutputColumns),
          folly::join(",", outputColumns),
          core::JoinTypeName::toName(joinType),
          duckDbVerifySql);
    }
  } testSettings[] = {
      // Inner join with 2 out of 3 keys
      {{100, 1, 1},
       5,
       100,
       80,
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t0", "t1", "u0", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c0, t.c1, u.c0, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1"},
      // Left join with 2 out of 3 keys
      {{100, 1, 1},
       5,
       100,
       80,
       2,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t0", "t1", "u0", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c0, t.c1, u.c0, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 "},
      // Inner join with 1 out of 3 keys
      {{100, 1, 1},
       5,
       100,
       80,
       1,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t0", "u0", "u1", "u2", "u3", "u5"},
       core::JoinType::kInner,
       "SELECT t.c0, u.c0, u.c1, u.c2, u.c3, u.c5 FROM t, u WHERE t.c0 = u.c0 "},
      // Left join with 1 out of 3 keys
      {{100, 1, 1},
       5,
       100,
       80,
       1,
       {"u0", "u1", "u2", "u3", "u5"},
       {"t0", "u0", "u1", "u2", "u3", "u5"},
       core::JoinType::kLeft,
       "SELECT t.c0, u.c0, u.c1, u.c2, u.c3, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 "}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    IndexTableData tableData;
    generateIndexTableData(testData.keyCardinalities, tableData, pool_);

    // Generate probe vectors with only the prefix of keys
    std::vector<std::string> probeKeys;
    for (int i = 0; i < testData.numKeysToUse; ++i) {
      probeKeys.push_back(fmt::format("t{}", i));
    }

    auto probeVectors = generateProbeInput(
        testData.numProbeBatches,
        testData.numRowsPerProbeBatch,
        1,
        tableData,
        pool_,
        probeKeys,
        GetParam().hasNullKeys,
        {},
        {},
        testData.matchPct);
    std::vector<std::shared_ptr<TempFilePath>> probeFiles =
        createProbeFiles(probeVectors);

    createDuckDbTable("t", probeVectors);
    createDuckDbTable("u", {tableData.tableVectors});

    const auto indexTable = TestIndexTable::create(
        /*numEqualJoinKeys=*/testData.numKeysToUse,
        tableData.keyVectors,
        tableData.valueVectors,
        *pool());
    const auto indexTableHandle = makeIndexTableHandle(
        indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        makeScanOutputType(testData.scanOutputColumns),
        makeIndexColumnHandles(testData.scanOutputColumns));

    // Create left and right join keys with only the prefix
    std::vector<std::string> leftKeys;
    std::vector<std::string> rightKeys;
    for (int i = 0; i < testData.numKeysToUse; ++i) {
      leftKeys.push_back(fmt::format("t{}", i));
      rightKeys.push_back(fmt::format("u{}", i));
    }

    auto plan = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        leftKeys,
        rightKeys,
        {},
        /*filter=*/"",
        /*hasMarker=*/false,
        testData.joinType,
        testData.outputColumns);
    runLookupQuery(
        plan,
        probeFiles,
        GetParam().serialExecution,
        GetParam().serialExecution,
        32,
        GetParam().numPrefetches,
        GetParam().needsIndexSplit,
        testData.duckDbVerifySql);
    if (testData.joinType != core::JoinType::kLeft) {
      continue;
    }
    const auto probeScanId = probeScanNodeId_;
    auto planWithMatchColumn = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        leftKeys,
        rightKeys,
        {},
        /*filter=*/"",
        /*hasMarker=*/true,
        testData.joinType,
        testData.outputColumns);
    verifyResultWithMatchColumn(
        plan,
        probeScanId,
        planWithMatchColumn,
        probeScanNodeId_,
        probeFiles,
        GetParam().needsIndexSplit);
  }
}

TEST_P(IndexLookupJoinTest, prefixKeysbetweenJoinCondition) {
  struct {
    std::vector<int> keyCardinalities;
    int numProbeBatches;
    int numProbeRowsPerBatch;
    std::string betweenCondition;
    int betweenMatchPct;
    std::vector<std::string> lookupOutputColumns;
    std::vector<std::string> outputColumns;
    core::JoinType joinType;
    std::string duckDbVerifySql;

    std::string debugString() const {
      return fmt::format(
          "keyCardinalities: {}, numProbeBatches: {}, numProbeRowsPerBatch: {}, betweenCondition: {}, betweenMatchPct: {}, "
          "lookupOutputColumns: {}, outputColumns: {}, joinType: {}, duckDbVerifySql: {}",
          folly::join(",", keyCardinalities),
          numProbeBatches,
          numProbeRowsPerBatch,
          betweenCondition,
          betweenMatchPct,
          folly::join(",", lookupOutputColumns),
          folly::join(",", outputColumns),
          core::JoinTypeName::toName(joinType),
          duckDbVerifySql);
    }
  } testSettings[] = {
      {{50, 1, 10},
       5,
       100,
       "u1 between t1 and t3",
       10,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t3", "u1", "u3"},
       core::JoinType::kInner,
       "SELECT t.c0, t.c1, t.c3, u.c1, u.c3 FROM t, u WHERE t.c0 = u.c0 AND u.c1 BETWEEN t.c1 AND t.c3"},
      {{50, 1, 10},
       5,
       100,
       "u1 between t1 and t3",
       10,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t3", "u1", "u3"},
       core::JoinType::kLeft,
       "SELECT t.c0, t.c1, t.c3, u.c1, u.c3 FROM t LEFT JOIN u ON t.c0 = u.c0 AND u.c1 BETWEEN t.c1 AND t.c3"},
      {{50, 1, 10},
       5,
       100,
       "u1 between 0 and t1",
       80,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "u1", "u3"},
       core::JoinType::kInner,
       "SELECT t.c0, t.c1, u.c1, u.c3 FROM t, u WHERE t.c0 = u.c0 AND u.c1 BETWEEN 0 AND t.c1"},
      {{50, 1, 10},
       5,
       100,
       "u1 between 0 and t1",
       80,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "u1", "u3"},
       core::JoinType::kLeft,
       "SELECT t.c0, t.c1, u.c1, u.c3 FROM t LEFT JOIN u ON t.c0 = u.c0 AND u.c1 BETWEEN 0 AND t.c1"}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    IndexTableData tableData;
    generateIndexTableData(testData.keyCardinalities, tableData, pool_);

    auto probeVectors = generateProbeInput(
        testData.numProbeBatches,
        testData.numProbeRowsPerBatch,
        1,
        tableData,
        pool_,
        {"t0"},
        GetParam().hasNullKeys,
        {},
        {{"t1", "t2"}},
        /*equalMatchPct=*/80,
        /*inMatchPct=*/std::nullopt,
        testData.betweenMatchPct);
    std::vector<std::shared_ptr<TempFilePath>> probeFiles =
        createProbeFiles(probeVectors);

    createDuckDbTable("t", probeVectors);
    createDuckDbTable("u", {tableData.tableVectors});

    const auto indexTable = TestIndexTable::create(
        /*numEqualJoinKeys=*/1,
        tableData.keyVectors,
        tableData.valueVectors,
        *pool());
    const auto indexTableHandle = makeIndexTableHandle(
        indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        makeScanOutputType(testData.lookupOutputColumns),
        makeIndexColumnHandles(testData.lookupOutputColumns));

    auto plan = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0"},
        {"u0"},
        {testData.betweenCondition},
        /*filter=*/"",
        /*hasMarker=*/false,
        testData.joinType,
        testData.outputColumns);
    runLookupQuery(
        plan,
        probeFiles,
        GetParam().serialExecution,
        GetParam().serialExecution,
        32,
        GetParam().numPrefetches,
        GetParam().needsIndexSplit,
        testData.duckDbVerifySql);
    if (testData.joinType != core::JoinType::kLeft) {
      continue;
    }
    const auto probeScanId = probeScanNodeId_;
    auto planWithMatchColumn = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0"},
        {"u0"},
        {testData.betweenCondition},
        /*filter=*/"",
        /*hasMarker=*/true,
        testData.joinType,
        testData.outputColumns);
    verifyResultWithMatchColumn(
        plan,
        probeScanId,
        planWithMatchColumn,
        probeScanNodeId_,
        probeFiles,
        GetParam().needsIndexSplit);
  }
}

TEST_P(IndexLookupJoinTest, prefixInJoinCondition) {
  struct {
    std::vector<int> keyCardinalities;
    int numProbeBatches;
    int numProbeRowsPerBatch;
    std::string inCondition;
    int inMatchPct;
    std::vector<std::string> lookupOutputColumns;
    std::vector<std::string> outputColumns;
    core::JoinType joinType;
    std::string duckDbVerifySql;

    std::string debugString() const {
      return fmt::format(
          "keyCardinalities: {}: numProbeBatches: {}, numProbeRowsPerBatch: {}, inCondition: {}, inMatchPct: {}, lookupOutputColumns: {}, outputColumns: {}, joinType: {}, duckDbVerifySql: {}",
          folly::join(",", keyCardinalities),
          numProbeBatches,
          numProbeRowsPerBatch,
          inCondition,
          inMatchPct,
          folly::join(",", lookupOutputColumns),
          folly::join(",", outputColumns),
          core::JoinTypeName::toName(joinType),
          duckDbVerifySql);
    }
  } testSettings[] = {
      // Inner join with prefix keys (c0, c1)
      {{50, 1, 10},
       1,
       100,
       "contains(t4, u1)",
       10,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t4", "u1", "u3"},
       core::JoinType::kInner,
       "SELECT t.c0, t.c1, t.c4, u.c1, u.c3 FROM t, u WHERE t.c0 = u.c0 AND array_contains(t.c4, u.c1)"},
      // Left join with prefix keys (c0, c1)
      {{50, 1, 10},
       1,
       100,
       "contains(t4, u1)",
       10,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t4", "u1", "u3"},
       core::JoinType::kLeft,
       "SELECT t.c0, t.c1, t.c4, u.c1, u.c3 FROM t LEFT JOIN u ON t.c0 = u.c0 AND array_contains(t.c4, u.c1)"},
      // Inner join with prefix keys (c0, c1) - higher match percentage
      {{50, 1, 10},
       10,
       100,
       "contains(t4, u1)",
       80,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t4", "u1", "u3"},
       core::JoinType::kInner,
       "SELECT t.c0, t.c1, t.c4, u.c1, u.c3 FROM t, u WHERE t.c0 = u.c0 AND array_contains(t.c4, u.c1)"},
      // Left join with prefix keys (c0, c1) - higher match percentage
      {{50, 1, 10},
       10,
       100,
       "contains(t4, u1)",
       80,
       {"u0", "u1", "u2", "u3"},
       {"t0", "t1", "t4", "u1", "u3"},
       core::JoinType::kLeft,
       "SELECT t.c0, t.c1, t.c4, u.c1, u.c3 FROM t LEFT JOIN u ON t.c0 = u.c0 AND array_contains(t.c4, u.c1)"}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    IndexTableData tableData;
    generateIndexTableData(testData.keyCardinalities, tableData, pool_);
    auto probeVectors = generateProbeInput(
        testData.numProbeBatches,
        testData.numProbeRowsPerBatch,
        1,
        tableData,
        pool_,
        {"t0"},
        GetParam().hasNullKeys,
        {{"t4"}},
        {},
        /*equalMatchPct=*/80,
        testData.inMatchPct);
    std::vector<std::shared_ptr<TempFilePath>> probeFiles =
        createProbeFiles(probeVectors);

    createDuckDbTable("t", probeVectors);
    createDuckDbTable("u", {tableData.tableVectors});

    const auto indexTable = TestIndexTable::create(
        /*numEqualJoinKeys=*/1,
        tableData.keyVectors,
        tableData.valueVectors,
        *pool());
    const auto indexTableHandle = makeIndexTableHandle(
        indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        makeScanOutputType(testData.lookupOutputColumns),
        makeIndexColumnHandles(testData.lookupOutputColumns));

    auto plan = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0"},
        {"u0"},
        {testData.inCondition},
        /*filter=*/"",
        /*hasMarker=*/false,
        testData.joinType,
        testData.outputColumns);
    runLookupQuery(
        plan,
        probeFiles,
        GetParam().serialExecution,
        GetParam().serialExecution,
        32,
        GetParam().numPrefetches,
        GetParam().needsIndexSplit,
        testData.duckDbVerifySql);
    if (testData.joinType != core::JoinType::kLeft) {
      continue;
    }
    const auto probeScanId = probeScanNodeId_;
    auto planWithMatchColumn = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0"},
        {"u0"},
        {testData.inCondition},
        /*filter=*/"",
        /*hasMarker=*/true,
        testData.joinType,
        testData.outputColumns);
    verifyResultWithMatchColumn(
        plan,
        probeScanId,
        planWithMatchColumn,
        probeScanNodeId_,
        probeFiles,
        GetParam().needsIndexSplit);
  }
}

DEBUG_ONLY_TEST_P(IndexLookupJoinTest, connectorError) {
  IndexTableData tableData;
  generateIndexTableData({100, 1, 1}, tableData, pool_);
  const std::vector<RowVectorPtr> probeVectors = generateProbeInput(
      20,
      100,
      1,
      tableData,
      pool_,
      {"t0", "t1", "t2"},
      GetParam().hasNullKeys,
      {},
      {},
      100);
  std::vector<std::shared_ptr<TempFilePath>> probeFiles =
      createProbeFiles(probeVectors);

  const std::string errorMsg{"injectedError"};
  std::atomic_int lookupCount{0};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::test::TestIndexSource::ResultIterator::syncLookup",
      std::function<void(void*)>([&](void*) {
        // Triggers error in the middle.
        if (lookupCount++ == 10) {
          VELOX_FAIL(errorMsg);
        }
      }));

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/3,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType({"u0", "u1", "u2", "u5"}),
      makeIndexColumnHandles({"u0", "u1", "u2", "u5"}));

  auto plan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0", "t1", "t2"},
      {"u0", "u1", "u2"},
      {},
      /*filter=*/"",
      /*hasMarker=*/false,
      core::JoinType::kInner,
      {"u0", "u1", "u2", "t5"});
  VELOX_ASSERT_THROW(
      runLookupQuery(
          plan,
          probeFiles,
          GetParam().serialExecution,
          GetParam().serialExecution,
          100,
          GetParam().numPrefetches,
          GetParam().needsIndexSplit,
          "SELECT u.c0, u.c1, t.c2, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2"),
      errorMsg);
}

DEBUG_ONLY_TEST_P(IndexLookupJoinTest, prefetch) {
  if (!GetParam().asyncLookup || GetParam().serialExecution) {
    // This test only works for async lookup.
    return;
  }
  IndexTableData tableData;
  generateIndexTableData({100, 1, 1}, tableData, pool_);
  const int numProbeBatches{20};
  ASSERT_GT(numProbeBatches, GetParam().numPrefetches);
  const std::vector<RowVectorPtr> probeVectors = generateProbeInput(
      numProbeBatches,
      100,
      1,
      tableData,
      pool_,
      {"t0", "t1", "t2"},
      GetParam().hasNullKeys,
      {},
      {},
      100);
  std::vector<std::shared_ptr<TempFilePath>> probeFiles =
      createProbeFiles(probeVectors);
  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  std::atomic_int lookupCount{0};
  folly::EventCount asyncLookupWait;
  std::atomic_bool asyncLookupWaitFlag{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::test::TestIndexSource::ResultIterator::asyncLookup",
      std::function<void(void*)>([&](void*) {
        if (++lookupCount > 1 + GetParam().numPrefetches) {
          return;
        }
        asyncLookupWait.await([&] { return !asyncLookupWaitFlag.load(); });
      }));

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/3,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType({"u0", "u1", "u2", "u3", "u5"}),
      makeIndexColumnHandles({"u0", "u1", "u2", "u3", "u5"}));

  auto plan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0", "t1", "t2"},
      {"u0", "u1", "u2"},
      {},
      /*filter=*/"",
      /*hasMarker=*/false,
      core::JoinType::kInner,
      {"u3", "t5"});
  std::thread queryThread([&] {
    runLookupQuery(
        plan,
        probeFiles,
        GetParam().serialExecution,
        GetParam().serialExecution,
        100,
        GetParam().numPrefetches,
        GetParam().needsIndexSplit,
        "SELECT u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2");
  });
  while (lookupCount < 1 + GetParam().numPrefetches) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // NOLINT
  }
  std::this_thread::sleep_for(std::chrono::seconds(1)); // NOLINT
  ASSERT_EQ(lookupCount, 1 + GetParam().numPrefetches);
  asyncLookupWaitFlag = false;
  asyncLookupWait.notifyAll();
  queryThread.join();
}

TEST_P(IndexLookupJoinTest, outputBatchSizeWithInnerJoin) {
  IndexTableData tableData;
  generateIndexTableData({3'000, 1, 1}, tableData, pool_);

  struct {
    int numProbeBatches;
    int numRowsPerProbeBatch;
    int maxBatchRows;
    bool splitOutput;
    int numExpectedOutputBatch;

    std::string debugString() const {
      return fmt::format(
          "numProbeBatches: {}, numRowsPerProbeBatch: {}, maxBatchRows: {}, splitOutput: {}, numExpectedOutputBatch: {}",
          numProbeBatches,
          numRowsPerProbeBatch,
          maxBatchRows,
          splitOutput,
          numExpectedOutputBatch);
    }
  } testSettings[] = {
      {10, 100, 10, false, 10},
      {10, 500, 10, false, 10},
      {10, 1, 200, false, 10},
      {1, 500, 10, false, 1},
      {1, 300, 10, false, 1},
      {1, 500, 200, false, 1},
      {10, 200, 200, false, 10},
      {10, 500, 300, false, 10},
      {10, 50, 1, false, 10},
      {10, 100, 10, true, 100},
      {10, 500, 10, true, 500},
      {10, 1, 200, true, 10},
      {1, 500, 10, true, 50},
      {1, 300, 10, true, 30},
      {1, 500, 200, true, 3},
      {10, 200, 200, true, 10},
      {10, 500, 300, true, 20},
      {10, 50, 1, true, 500}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    const auto probeVectors = generateProbeInput(
        testData.numProbeBatches,
        testData.numRowsPerProbeBatch,
        1,
        tableData,
        pool_,
        {"t0", "t1", "t2"},
        GetParam().hasNullKeys,
        {},
        {},
        /*equalMatchPct=*/100);
    std::vector<std::shared_ptr<TempFilePath>> probeFiles =
        createProbeFiles(probeVectors);

    createDuckDbTable("t", probeVectors);
    createDuckDbTable("u", {tableData.tableVectors});

    const auto indexTable = TestIndexTable::create(
        /*numEqualJoinKeys=*/3,
        tableData.keyVectors,
        tableData.valueVectors,
        *pool());
    const auto indexTableHandle = makeIndexTableHandle(
        indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        makeScanOutputType({"u0", "u1", "u2", "u5"}),
        makeIndexColumnHandles({"u0", "u1", "u2", "u5"}));

    auto plan = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0", "t1", "t2"},
        {"u0", "u1", "u2"},
        {},
        /*filter=*/"",
        /*hasMarker=*/false,
        core::JoinType::kInner,
        {"t4", "u5"});
    AssertQueryBuilder queryBuilder(duckDbQueryRunner_);
    queryBuilder.plan(plan)
        .config(
            core::QueryConfig::kIndexLookupJoinMaxPrefetchBatches,
            std::to_string(GetParam().numPrefetches))
        .config(
            core::QueryConfig::kPreferredOutputBatchRows,
            std::to_string(testData.maxBatchRows))
        .config(
            core::QueryConfig::kPreferredOutputBatchBytes,
            std::to_string(1ULL << 30))
        .config(
            core::QueryConfig::kIndexLookupJoinSplitOutput,
            testData.splitOutput ? "true" : "false")
        .splits(probeScanNodeId_, makeHiveConnectorSplits(probeFiles))
        .serialExecution(GetParam().serialExecution)
        .barrierExecution(GetParam().serialExecution);
    if (GetParam().needsIndexSplit) {
      queryBuilder.split(
          indexScanNodeId_,
          Split(
              std::make_shared<TestIndexConnectorSplit>(
                  kTestIndexConnectorName)));
    }
    const auto task = queryBuilder.assertResults(
        "SELECT t.c4, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2");
    ASSERT_EQ(
        toPlanStats(task->taskStats()).at(joinNodeId_).outputVectors,
        testData.numExpectedOutputBatch);
  }
}

TEST_P(IndexLookupJoinTest, outputBatchSizeWithLeftJoin) {
  IndexTableData tableData;
  generateIndexTableData({3'000, 1, 1}, tableData, pool_);

  struct {
    int numProbeBatches;
    int numRowsPerProbeBatch;
    int maxBatchRows;
    bool splitOutput;
    int numExpectedOutputBatch;

    std::string debugString() const {
      return fmt::format(
          "numProbeBatches: {}, numRowsPerProbeBatch: {}, maxBatchRows: {}, splitOutput: {}, numExpectedOutputBatch: {}",
          numProbeBatches,
          numRowsPerProbeBatch,
          maxBatchRows,
          splitOutput,
          numExpectedOutputBatch);
    }
  } testSettings[] = {
      {10, 100, 10, false, 10},
      {10, 500, 10, false, 10},
      {10, 1, 200, false, 10},
      {1, 500, 10, false, 1},
      {1, 300, 10, false, 1},
      {1, 500, 200, false, 1},
      {10, 200, 200, false, 10},
      {10, 500, 300, false, 10},
      {10, 50, 1, false, 10},
      {10, 100, 10, true, 100},
      {10, 500, 10, true, 500},
      {10, 1, 200, true, 10},
      {1, 500, 10, true, 50},
      {1, 300, 10, true, 30},
      {1, 500, 200, true, 3},
      {10, 200, 200, true, 10},
      {10, 500, 300, true, 20},
      {10, 50, 1, true, 500}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    const auto probeVectors = generateProbeInput(
        testData.numProbeBatches,
        testData.numRowsPerProbeBatch,
        1,
        tableData,
        pool_,
        {"t0", "t1", "t2"},
        GetParam().hasNullKeys,
        {},
        {},
        /*equalMatchPct=*/100);
    std::vector<std::shared_ptr<TempFilePath>> probeFiles =
        createProbeFiles(probeVectors);

    createDuckDbTable("t", probeVectors);
    createDuckDbTable("u", {tableData.tableVectors});

    const auto indexTable = TestIndexTable::create(
        /*numEqualJoinKeys=*/3,
        tableData.keyVectors,
        tableData.valueVectors,
        *pool());
    const auto indexTableHandle = makeIndexTableHandle(
        indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        makeScanOutputType({"u0", "u1", "u2", "u5"}),
        makeIndexColumnHandles({"u0", "u1", "u2", "u5"}));

    auto plan = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0", "t1", "t2"},
        {"u0", "u1", "u2"},
        {},
        /*filter=*/"",
        /*hasMarker=*/false,
        core::JoinType::kLeft,
        {"t4", "u5"});
    AssertQueryBuilder queryBuilder(duckDbQueryRunner_);
    queryBuilder.plan(plan)
        .config(
            core::QueryConfig::kIndexLookupJoinMaxPrefetchBatches,
            std::to_string(GetParam().numPrefetches))
        .config(
            core::QueryConfig::kPreferredOutputBatchRows,
            std::to_string(testData.maxBatchRows))
        .config(
            core::QueryConfig::kPreferredOutputBatchBytes,
            std::to_string(1ULL << 30))
        .config(
            core::QueryConfig::kIndexLookupJoinSplitOutput,
            testData.splitOutput ? "true" : "false")
        .splits(probeScanNodeId_, makeHiveConnectorSplits(probeFiles))
        .serialExecution(GetParam().serialExecution)
        .barrierExecution(GetParam().serialExecution);
    if (GetParam().needsIndexSplit) {
      queryBuilder.split(
          indexScanNodeId_,
          Split(
              std::make_shared<TestIndexConnectorSplit>(
                  kTestIndexConnectorName)));
    }
    const auto task = queryBuilder.assertResults(
        "SELECT t.c4, u.c5 FROM t LEFT JOIN u ON t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2");
    ASSERT_EQ(
        toPlanStats(task->taskStats()).at(joinNodeId_).outputVectors,
        testData.numExpectedOutputBatch);
    const auto probeScanId = probeScanNodeId_;
    auto planWithMatchColumn = makeLookupPlan(
        planNodeIdGenerator,
        indexScanNode,
        {"t0", "t1", "t2"},
        {"u0", "u1", "u2"},
        {},
        /*filter=*/"",
        /*hasMarker=*/true,
        core::JoinType::kLeft,
        {"t4", "u5"});
    verifyResultWithMatchColumn(
        plan,
        probeScanId,
        planWithMatchColumn,
        probeScanNodeId_,
        probeFiles,
        GetParam().needsIndexSplit);
  }
}

TEST_P(IndexLookupJoinTest, splitOutputNodeOverride) {
  IndexTableData tableData;
  generateIndexTableData({3'000, 1, 1}, tableData, pool_);

  const int numProbeBatches = 10;
  const int numRowsPerProbeBatch = 100;
  const int maxBatchRows = 10;

  const auto probeVectors = generateProbeInput(
      numProbeBatches,
      numRowsPerProbeBatch,
      1,
      tableData,
      pool_,
      {"t0", "t1", "t2"},
      GetParam().hasNullKeys,
      {},
      {},
      /*equalMatchPct=*/100);
  std::vector<std::shared_ptr<TempFilePath>> probeFiles =
      createProbeFiles(probeVectors);

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/3,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);

  struct {
    std::optional<bool> nodeSplitOutput;
    bool configSplitOutput;
    bool expectSplit;

    std::string debugString() const {
      return fmt::format(
          "nodeSplitOutput: {}, configSplitOutput: {}, expectSplit: {}",
          nodeSplitOutput.has_value()
              ? (nodeSplitOutput.value() ? "true" : "false")
              : "nullopt",
          configSplitOutput,
          expectSplit);
    }
  } testSettings[] = {
      // Node splitOutput not set, use config.
      {std::nullopt, true, true},
      {std::nullopt, false, false},
      // Node splitOutput=true overrides config.
      {true, true, true},
      {true, false, true},
      // Node splitOutput=false overrides config.
      {false, true, false},
      {false, false, false},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    const auto indexScanNode = makeIndexScanNode(
        planNodeIdGenerator,
        indexTableHandle,
        makeScanOutputType({"u0", "u1", "u2", "u5"}),
        makeIndexColumnHandles({"u0", "u1", "u2", "u5"}));

    core::PlanNodeId joinNodeId;
    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .startTableScan()
                    .outputType(probeType_)
                    .endTableScan()
                    .captureScanNodeId(probeScanNodeId_)
                    .startIndexLookupJoin()
                    .leftKeys({"t0", "t1", "t2"})
                    .rightKeys({"u0", "u1", "u2"})
                    .indexSource(indexScanNode)
                    .outputLayout({"t4", "u5"})
                    .joinType(core::JoinType::kInner)
                    .splitOutput(testData.nodeSplitOutput)
                    .endIndexLookupJoin()
                    .capturePlanNodeId(joinNodeId)
                    .planNode();

    const int expectedNumOutputVectors = testData.expectSplit
        ? ((numRowsPerProbeBatch + maxBatchRows - 1) / maxBatchRows) *
            numProbeBatches
        : numProbeBatches;

    AssertQueryBuilder queryBuilder(duckDbQueryRunner_);
    queryBuilder.plan(plan)
        .config(
            core::QueryConfig::kIndexLookupJoinMaxPrefetchBatches,
            std::to_string(GetParam().numPrefetches))
        .config(
            core::QueryConfig::kPreferredOutputBatchRows,
            std::to_string(maxBatchRows))
        .config(
            core::QueryConfig::kPreferredOutputBatchBytes,
            std::to_string(1ULL << 30))
        .config(
            core::QueryConfig::kIndexLookupJoinSplitOutput,
            testData.configSplitOutput ? "true" : "false")
        .splits(probeScanNodeId_, makeHiveConnectorSplits(probeFiles))
        .serialExecution(GetParam().serialExecution)
        .barrierExecution(GetParam().serialExecution);
    if (GetParam().needsIndexSplit) {
      queryBuilder.split(
          indexScanNodeId_,
          Split(
              std::make_shared<TestIndexConnectorSplit>(
                  kTestIndexConnectorName)));
    }
    const auto task = queryBuilder.assertResults(
        "SELECT t.c4, u.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2");
    ASSERT_EQ(
        toPlanStats(task->taskStats()).at(joinNodeId).outputVectors,
        expectedNumOutputVectors);
  }
}

DEBUG_ONLY_TEST_P(IndexLookupJoinTest, statsSplitter) {
  IndexTableData tableData;
  generateIndexTableData({100, 1, 1}, tableData, pool_);
  const int numProbeBatches{2};
  const int batchSize{100};
  const std::vector<RowVectorPtr> probeVectors = generateProbeInput(
      numProbeBatches,
      batchSize,
      1,
      tableData,
      pool_,
      {"t0", "t1", "t2"},
      GetParam().hasNullKeys,
      {},
      {},
      100);
  std::vector<std::shared_ptr<TempFilePath>> probeFiles =
      createProbeFiles(probeVectors);
  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  // Add a small delay in async lookup to ensure timing stats are captured.
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::test::TestIndexSource::ResultIterator::asyncLookup",
      std::function<void(void*)>([&](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // NOLINT
      }));

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/3,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType({"u0", "u1", "u2", "u3", "u5"}),
      makeIndexColumnHandles({"u0", "u1", "u2", "u3", "u5"}));

  auto plan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0", "t1", "t2"},
      {"u0", "u1", "u2"},
      {},
      /*filter=*/"",
      /*hasMarker=*/false,
      core::JoinType::kInner,
      {"u3", "t5"});
  auto task = runLookupQuery(
      plan,
      probeFiles,
      GetParam().serialExecution,
      GetParam().serialExecution,
      100,
      0,
      GetParam().needsIndexSplit,
      "SELECT u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2");

  auto taskStats = toPlanStats(task->taskStats());

  // Verify that both the IndexLookupJoin node and IndexSource node have stats.
  ASSERT_GT(taskStats.count(joinNodeId_), 0);
  ASSERT_GT(taskStats.count(indexScanNodeId_), 0);

  const auto& joinStats = taskStats.at(joinNodeId_);
  const auto& indexSourceStats = taskStats.at(indexScanNodeId_);

  EXPECT_GT(joinStats.inputRows, 0);
  EXPECT_GT(joinStats.outputRows, 0);

  EXPECT_GT(indexSourceStats.outputRows, 0);
  EXPECT_EQ(indexSourceStats.outputRows, joinStats.outputRows);

  EXPECT_GT(indexSourceStats.inputRows, 0);
  EXPECT_GT(indexSourceStats.inputBytes, 0);
  EXPECT_EQ(indexSourceStats.inputRows, joinStats.inputRows);

  EXPECT_GT(indexSourceStats.addInputTiming.count, 0);

  EXPECT_GT(
      indexSourceStats.customStats.count(
          std::string(IndexLookupJoin::kConnectorLookupWallTime)),
      0);
  EXPECT_GT(
      indexSourceStats.customStats
          .at(std::string(IndexLookupJoin::kConnectorLookupWallTime))
          .sum,
      0);

  // backgroundTiming should be cleared from join stats (moved to IndexSource).
  EXPECT_EQ(joinStats.backgroundTiming.count, 0);
  EXPECT_EQ(joinStats.backgroundTiming.cpuNanos, 0);
  EXPECT_EQ(joinStats.backgroundTiming.wallNanos, 0);
}

/// Verifies that the probe-side scan operator's inputBytes/outputBytes are not
/// inflated with index lookup bytes. IndexLookupJoin uses a custom stat writer
/// in getOutput() to redirect index-side lazy loading stats to separate names,
/// so Driver::processLazyIoStats() only transfers probe-side stats to the scan.
TEST_P(IndexLookupJoinTest, scanStatsNotInflated) {
  IndexTableData tableData;
  generateIndexTableData({100, 1, 1}, tableData, pool_);
  const int numProbeBatches{2};
  const int batchSize{100};
  const std::vector<RowVectorPtr> probeVectors = generateProbeInput(
      numProbeBatches,
      batchSize,
      1,
      tableData,
      pool_,
      {"t0", "t1", "t2"},
      GetParam().hasNullKeys,
      {},
      {},
      100);
  std::vector<std::shared_ptr<TempFilePath>> probeFiles =
      createProbeFiles(probeVectors);
  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/3,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType({"u0", "u1", "u2", "u3", "u5"}),
      makeIndexColumnHandles({"u0", "u1", "u2", "u3", "u5"}));

  auto plan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0", "t1", "t2"},
      {"u0", "u1", "u2"},
      {},
      /*filter=*/"",
      /*hasMarker=*/false,
      core::JoinType::kInner,
      {"u3", "t5"});
  auto task = runLookupQuery(
      plan,
      probeFiles,
      GetParam().serialExecution,
      GetParam().serialExecution,
      100,
      0,
      GetParam().needsIndexSplit,
      "SELECT u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2");

  auto taskStats = toPlanStats(task->taskStats());

  ASSERT_GT(taskStats.count(probeScanNodeId_), 0);
  const auto& scanStats = taskStats.at(probeScanNodeId_);

  EXPECT_GT(scanStats.inputRows, 0);
  EXPECT_GT(scanStats.inputBytes, 0);
  EXPECT_EQ(scanStats.inputBytes, scanStats.outputBytes);

  // Verify that lazy loading stats are NOT present on the join node.
  // Probe-side lazy stats accumulate in every non-scan operator's
  // runtimeStats when lazy vectors are loaded. processLazyIoStats transfers
  // timing deltas to the scan but doesn't erase the accumulated stats.
  // For most operators this is harmless — the stats just sit in runtimeStats.
  // But IndexLookupJoin's splitStats copies combinedStats to create the join
  // node stats, so we must explicitly erase them to avoid exposing residual
  // lazy stats on the join node.
  const auto& joinStats = taskStats.at(joinNodeId_);
  EXPECT_EQ(
      joinStats.customStats.count(std::string(LazyVector::kInputBytes)), 0);
}

/// Verifies that IndexSource stats report rows BEFORE the join filter is
/// applied, while IndexLookupJoin stats report rows AFTER the filter.
DEBUG_ONLY_TEST_P(IndexLookupJoinTest, statsSplitterWithFilter) {
  // Skip serial execution tests for simplicity - the stats behavior is the
  // same.
  if (GetParam().serialExecution || GetParam().hasNullKeys) {
    return;
  }

  IndexTableData tableData;
  generateIndexTableData({100, 1, 1}, tableData, pool_);
  const int numProbeBatches{2};
  const int batchSize{100};
  const std::vector<RowVectorPtr> probeVectors = generateProbeInput(
      numProbeBatches,
      batchSize,
      1,
      tableData,
      pool_,
      {"t0", "t1", "t2"},
      /*hasNullKeys=*/false,
      {},
      {},
      100);
  std::vector<std::shared_ptr<TempFilePath>> probeFiles =
      createProbeFiles(probeVectors);
  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", {tableData.tableVectors});

  // Add a small delay in async lookup to ensure timing stats are captured.
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::test::TestIndexSource::ResultIterator::asyncLookup",
      std::function<void(void*)>([&](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // NOLINT
      }));

  const auto indexTable = TestIndexTable::create(
      /*numEqualJoinKeys=*/3,
      tableData.keyVectors,
      tableData.valueVectors,
      *pool());
  const auto indexTableHandle = makeIndexTableHandle(
      indexTable, GetParam().asyncLookup, GetParam().needsIndexSplit);
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto indexScanNode = makeIndexScanNode(
      planNodeIdGenerator,
      indexTableHandle,
      makeScanOutputType({"u0", "u1", "u2", "u3", "u5"}),
      makeIndexColumnHandles({"u0", "u1", "u2", "u3", "u5"}));

  // Add a filter that should filter out approximately half the rows.
  // The filter "u3 % 2 = 0" will keep only even values of u3.
  auto plan = makeLookupPlan(
      planNodeIdGenerator,
      indexScanNode,
      {"t0", "t1", "t2"},
      {"u0", "u1", "u2"},
      {},
      /*filter=*/"u3 % 2 = 0",
      /*hasMarker=*/false,
      core::JoinType::kInner,
      {"u3", "t5"});
  auto task = runLookupQuery(
      plan,
      probeFiles,
      GetParam().serialExecution,
      GetParam().serialExecution,
      100,
      0,
      GetParam().needsIndexSplit,
      "SELECT u.c3, t.c5 FROM t, u WHERE t.c0 = u.c0 AND t.c1 = u.c1 AND t.c2 = u.c2 AND u.c3 % 2 = 0");

  auto taskStats = toPlanStats(task->taskStats());

  ASSERT_GT(taskStats.count(joinNodeId_), 0);
  ASSERT_GT(taskStats.count(indexScanNodeId_), 0);

  const auto& joinStats = taskStats.at(joinNodeId_);
  const auto& indexSourceStats = taskStats.at(indexScanNodeId_);

  EXPECT_GT(joinStats.inputRows, 0);
  EXPECT_GT(joinStats.outputRows, 0);
  EXPECT_GT(indexSourceStats.outputRows, 0);

  // IndexSource reports rows before the filter; IndexLookupJoin reports after.
  EXPECT_GT(indexSourceStats.outputRows, joinStats.outputRows);
}

} // namespace

VELOX_INSTANTIATE_TEST_SUITE_P(
    IndexLookupJoinTest,
    IndexLookupJoinTest,
    testing::ValuesIn(IndexLookupJoinTest::getTestParams()),
    [](const testing::TestParamInfo<TestParam>& info) {
      return fmt::format(
          "{}_{}prefetches_{}_{}_{}",
          info.param.asyncLookup ? "async" : "sync",
          info.param.numPrefetches,
          info.param.serialExecution ? "serial" : "parallel",
          info.param.hasNullKeys ? "nullKeys" : "noNullKeys",
          info.param.needsIndexSplit ? "needsIndexSplit" : "noSplit");
    });
} // namespace facebook::velox::exec::test
