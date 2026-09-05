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
#include "velox/connectors/adbc/AdbcConnector.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/adbc/tests/AdbcTestDriver.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::adbc {
namespace {

using exec::test::AssertQueryBuilder;
using exec::test::PlanBuilder;

class AdbcConnectorTest : public exec::test::OperatorTestBase {
 protected:
  static constexpr const char* kAdbcConnectorId = "test-adbc";

  void SetUp() override {
    OperatorTestBase::SetUp();
    test::AdbcTestDriverState::instance().reset();
    registerAdbcConnector({});
  }

  void TearDown() override {
    // Drop batch vectors held by the fake driver before the fixture pools
    // that own them are destroyed.
    test::AdbcTestDriverState::instance().reset();
    connector::ConnectorRegistry::global().erase(kAdbcConnectorId);
    OperatorTestBase::TearDown();
  }

  // (Re-)registers the test connector, backed by the fake in-process driver,
  // with the given connector config values.
  void registerAdbcConnector(
      std::unordered_map<std::string, std::string> configValues) {
    connector::ConnectorRegistry::global().erase(kAdbcConnectorId);
    auto connector = std::make_shared<AdbcConnector>(
        kAdbcConnectorId,
        std::make_shared<config::ConfigBase>(std::move(configValues)),
        test::adbcTestDriverInit);
    connector::ConnectorRegistry::global().insert(kAdbcConnectorId, connector);
  }

  exec::Split makeAdbcSplit() const {
    return exec::Split(std::make_shared<AdbcConnectorSplit>(kAdbcConnectorId));
  }

  // Builds a table scan over the test connector. Exactly one of 'tableName'
  // and 'query' must be non-empty.
  core::PlanNodePtr makeScanPlan(
      const RowTypePtr& outputType,
      const std::string& tableName,
      const std::string& query) {
    connector::ColumnHandleMap assignments;
    for (const auto& columnName : outputType->names()) {
      assignments.emplace(
          columnName, std::make_shared<AdbcColumnHandle>(columnName));
    }
    auto tableHandle =
        std::make_shared<AdbcTableHandle>(kAdbcConnectorId, tableName, query);
    return PlanBuilder()
        .startTableScan()
        .outputType(outputType)
        .tableHandle(tableHandle)
        .assignments(assignments)
        .endTableScan()
        .planNode();
  }
};

TEST_F(AdbcConnectorTest, roundTrip) {
  auto batch1 = makeRowVector(
      {"c0", "c1", "c2"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeNullableFlatVector<StringView>({"a", std::nullopt, "c"}),
          makeFlatVector<double>({0.1, 0.2, 0.3}),
      });
  auto batch2 = makeRowVector(
      {"c0", "c1", "c2"},
      {
          makeFlatVector<int64_t>({4, 5}),
          makeNullableFlatVector<StringView>({std::nullopt, "e"}),
          makeFlatVector<double>({0.4, 0.5}),
      });
  auto rowType = asRowType(batch1->type());
  test::AdbcTestDriverState::instance().setResult(
      rowType, {batch1, batch2}, pool());

  auto plan = makeScanPlan(rowType, "my_table", "");
  AssertQueryBuilder(plan)
      .split(makeAdbcSplit())
      .assertResults({batch1, batch2});

  EXPECT_EQ(
      test::AdbcTestDriverState::instance().lastSql(),
      "SELECT \"c0\", \"c1\", \"c2\" FROM my_table");
}

TEST_F(AdbcConnectorTest, identifierQuote) {
  registerAdbcConnector({{AdbcConfig::kIdentifierQuote, "`"}});
  auto batch = makeRowVector(
      {"key", "value"},
      {
          makeFlatVector<int32_t>({1, 2}),
          makeFlatVector<StringView>({"x", "y"}),
      });
  auto rowType = asRowType(batch->type());
  test::AdbcTestDriverState::instance().setResult(rowType, {batch}, pool());

  auto plan = makeScanPlan(rowType, "orders", "");
  AssertQueryBuilder(plan).split(makeAdbcSplit()).assertResults(batch);

  EXPECT_EQ(
      test::AdbcTestDriverState::instance().lastSql(),
      "SELECT `key`, `value` FROM orders");
}

TEST_F(AdbcConnectorTest, queryPassthrough) {
  auto batch = makeRowVector(
      {"total"},
      {
          makeFlatVector<int64_t>({42}),
      });
  auto rowType = asRowType(batch->type());
  test::AdbcTestDriverState::instance().setResult(rowType, {batch}, pool());

  const std::string query =
      "SELECT sum(amount) AS total FROM orders WHERE region = 'EMEA'";
  auto plan = makeScanPlan(rowType, "", query);
  AssertQueryBuilder(plan).split(makeAdbcSplit()).assertResults(batch);

  EXPECT_EQ(test::AdbcTestDriverState::instance().lastSql(), query);
}

TEST_F(AdbcConnectorTest, columnReorderAndCaseInsensitiveMatch) {
  // The driver returns columns in a different order and letter case than the
  // scan output type.
  auto served = makeRowVector(
      {"B", "A"},
      {
          makeFlatVector<StringView>({"x", "y"}),
          makeFlatVector<int64_t>({1, 2}),
      });
  test::AdbcTestDriverState::instance().setResult(
      asRowType(served->type()), {served}, pool());

  auto outputType = ROW({"a", "b"}, {BIGINT(), VARCHAR()});
  auto plan = makeScanPlan(outputType, "t", "");
  auto expected = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({1, 2}),
          makeFlatVector<StringView>({"x", "y"}),
      });
  AssertQueryBuilder(plan).split(makeAdbcSplit()).assertResults(expected);
}

TEST_F(AdbcConnectorTest, zeroColumnScan) {
  // A scan with no projected columns, e.g. count(*), selects a constant.
  auto served = makeRowVector(
      {"one"},
      {
          makeFlatVector<int64_t>({1, 1, 1}),
      });
  test::AdbcTestDriverState::instance().setResult(
      asRowType(served->type()), {served}, pool());

  auto plan =
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({}, {}))
          .tableHandle(
              std::make_shared<AdbcTableHandle>(kAdbcConnectorId, "t", ""))
          .endTableScan()
          .singleAggregation({}, {"count(1)"})
          .planNode();
  auto expected = makeRowVector({makeFlatVector<int64_t>({3})});
  AssertQueryBuilder(plan).split(makeAdbcSplit()).assertResults(expected);

  EXPECT_EQ(test::AdbcTestDriverState::instance().lastSql(), "SELECT 1 FROM t");
}

TEST_F(AdbcConnectorTest, emptyResult) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  test::AdbcTestDriverState::instance().setResult(rowType, {}, pool());

  auto plan = makeScanPlan(rowType, "t", "");
  AssertQueryBuilder(plan).split(makeAdbcSplit()).assertEmptyResults();
}

TEST_F(AdbcConnectorTest, missingColumn) {
  auto served = makeRowVector(
      {"other"},
      {
          makeFlatVector<int64_t>({1}),
      });
  test::AdbcTestDriverState::instance().setResult(
      asRowType(served->type()), {served}, pool());

  auto plan = makeScanPlan(ROW({"c0"}, {BIGINT()}), "t", "");
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).split(makeAdbcSplit()).copyResults(pool()),
      "Column not found in ADBC result set: c0");
}

TEST_F(AdbcConnectorTest, typeMismatch) {
  auto served = makeRowVector(
      {"c0"},
      {
          makeFlatVector<StringView>({"x"}),
      });
  test::AdbcTestDriverState::instance().setResult(
      asRowType(served->type()), {served}, pool());

  auto plan = makeScanPlan(ROW({"c0"}, {BIGINT()}), "t", "");
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).split(makeAdbcSplit()).copyResults(pool()),
      "Column type mismatch in ADBC result set: c0");
}

TEST_F(AdbcConnectorTest, driverError) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  test::AdbcTestDriverState::instance().setResult(rowType, {}, pool());
  test::AdbcTestDriverState::instance().setExecuteError(
      "connection lost to remote database");

  auto plan = makeScanPlan(rowType, "t", "");
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).split(makeAdbcSplit()).copyResults(pool()),
      "connection lost to remote database");
}

TEST_F(AdbcConnectorTest, databaseOptionsPassthrough) {
  registerAdbcConnector({
      {"adbc.option.uri", "mysql://localhost:3306/testdb"},
      {"adbc.option.username", "alice"},
  });
  auto options = test::AdbcTestDriverState::instance().databaseOptions();
  EXPECT_EQ(options.at("uri"), "mysql://localhost:3306/testdb");
  EXPECT_EQ(options.at("username"), "alice");
}

TEST_F(AdbcConnectorTest, tableHandleValidation) {
  VELOX_ASSERT_THROW(
      std::make_shared<AdbcTableHandle>(kAdbcConnectorId, "", ""),
      "Exactly one of table name and query must be set");
  VELOX_ASSERT_THROW(
      std::make_shared<AdbcTableHandle>(kAdbcConnectorId, "t", "SELECT 1"),
      "Exactly one of table name and query must be set");
}

TEST_F(AdbcConnectorTest, serde) {
  AdbcTableHandle tableHandle(kAdbcConnectorId, "my_table", "");
  auto tableHandleCopy = std::dynamic_pointer_cast<const AdbcTableHandle>(
      AdbcTableHandle::create(tableHandle.serialize(), nullptr));
  ASSERT_NE(tableHandleCopy, nullptr);
  EXPECT_EQ(tableHandleCopy->connectorId(), kAdbcConnectorId);
  EXPECT_EQ(tableHandleCopy->tableName(), "my_table");
  EXPECT_EQ(tableHandleCopy->query(), "");

  AdbcTableHandle queryHandle(kAdbcConnectorId, "", "SELECT 1");
  auto queryHandleCopy = std::dynamic_pointer_cast<const AdbcTableHandle>(
      AdbcTableHandle::create(queryHandle.serialize(), nullptr));
  ASSERT_NE(queryHandleCopy, nullptr);
  EXPECT_EQ(queryHandleCopy->tableName(), "");
  EXPECT_EQ(queryHandleCopy->query(), "SELECT 1");

  AdbcColumnHandle columnHandle("c0");
  auto columnHandleCopy = AdbcColumnHandle::create(columnHandle.serialize());
  EXPECT_EQ(columnHandleCopy->name(), "c0");

  AdbcConnectorSplit split(kAdbcConnectorId);
  auto splitCopy = AdbcConnectorSplit::create(split.serialize());
  EXPECT_EQ(splitCopy->connectorId, kAdbcConnectorId);
}

} // namespace
} // namespace facebook::velox::connector::adbc
