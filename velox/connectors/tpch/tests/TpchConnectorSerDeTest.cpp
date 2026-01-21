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

#include <gtest/gtest.h>
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/connectors/tpch/TpchConnectorSplit.h"
#include "velox/core/ITypedExpr.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::tpch::test {
namespace {

class TpchConnectorSerDeTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  TpchConnectorSerDeTest() {
    Type::registerSerDe();
    core::ITypedExpr::registerSerDe();
    TpchConnector::registerSerDe();
  }

  template <typename T>
  static void testSerde(const T& handle) {
    auto str = handle.toString();
    auto obj = handle.serialize();
    auto clone = ISerializable::deserialize<T>(obj);
    ASSERT_EQ(clone->toString(), str);
  }

  static void testSerde(const TpchColumnHandle& handle) {
    auto obj = handle.serialize();
    auto clone = ISerializable::deserialize<TpchColumnHandle>(obj);
    ASSERT_EQ(handle.name(), clone->name());
  }

  static void testSerde(const TpchTableHandle& handle) {
    auto str = handle.toString();
    auto obj = handle.serialize();
    auto pool = memory::memoryManager()->addLeafPool();
    auto clone = ISerializable::deserialize<TpchTableHandle>(obj, pool.get());
    ASSERT_EQ(clone->toString(), str);
    ASSERT_EQ(handle.connectorId(), clone->connectorId());
    ASSERT_EQ(handle.getTable(), clone->getTable());
    ASSERT_EQ(handle.getScaleFactor(), clone->getScaleFactor());
    if (handle.filterExpression()) {
      ASSERT_NE(clone->filterExpression(), nullptr);
      ASSERT_EQ(
          handle.filterExpression()->toString(),
          clone->filterExpression()->toString());
    } else {
      ASSERT_EQ(clone->filterExpression(), nullptr);
    }
  }

  static void testSerde(const TpchConnectorSplit& split) {
    auto str = split.toString();
    auto obj = split.serialize();
    auto clone = ISerializable::deserialize<TpchConnectorSplit>(obj);
    ASSERT_EQ(clone->toString(), str);
    ASSERT_EQ(split.connectorId, clone->connectorId);
    ASSERT_EQ(split.splitWeight, clone->splitWeight);
    ASSERT_EQ(split.cacheable, clone->cacheable);
    ASSERT_EQ(split.totalParts, clone->totalParts);
    ASSERT_EQ(split.partNumber, clone->partNumber);
  }
};

TEST_F(TpchConnectorSerDeTest, tpchColumnHandle) {
  auto handle1 = TpchColumnHandle("n_nationkey");
  testSerde(handle1);

  auto handle2 = TpchColumnHandle("l_orderkey");
  testSerde(handle2);

  auto handle3 = TpchColumnHandle("c_name");
  testSerde(handle3);
}

TEST_F(TpchConnectorSerDeTest, tpchTableHandle) {
  const std::string connectorId = "test-tpch";

  auto handle1 = TpchTableHandle(connectorId, velox::tpch::Table::TBL_NATION);
  testSerde(handle1);

  auto handle2 =
      TpchTableHandle(connectorId, velox::tpch::Table::TBL_LINEITEM, 10.0);
  testSerde(handle2);

  auto handle3 =
      TpchTableHandle(connectorId, velox::tpch::Table::TBL_ORDERS, 0.01);
  testSerde(handle3);

  // Test with filterExpression
  auto filterExpr =
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "n_nationkey");
  auto handle4 = TpchTableHandle(
      connectorId, velox::tpch::Table::TBL_NATION, 1.0, filterExpr);
  testSerde(handle4);

  std::vector<velox::tpch::Table> tables = {
      velox::tpch::Table::TBL_NATION,
      velox::tpch::Table::TBL_REGION,
      velox::tpch::Table::TBL_PART,
      velox::tpch::Table::TBL_SUPPLIER,
      velox::tpch::Table::TBL_PARTSUPP,
      velox::tpch::Table::TBL_CUSTOMER,
      velox::tpch::Table::TBL_ORDERS,
      velox::tpch::Table::TBL_LINEITEM,
  };

  for (auto table : tables) {
    testSerde(TpchTableHandle(connectorId, table, 1.0));
  }

  std::vector<double> scaleFactors = {0.01, 0.1, 1.0, 5.0, 10.0, 100.0, 1000.0};
  for (auto sf : scaleFactors) {
    testSerde(
        TpchTableHandle(connectorId, velox::tpch::Table::TBL_CUSTOMER, sf));
  }
}

TEST_F(TpchConnectorSerDeTest, tpchConnectorSplit) {
  const std::string connectorId = "test-tpch";

  auto split1 = TpchConnectorSplit(connectorId, false, 10, 5);
  testSerde(split1);

  auto split2 = TpchConnectorSplit(connectorId, true, 100, 99);
  testSerde(split2);

  auto split3 = TpchConnectorSplit(connectorId, 1, 0);
  testSerde(split3);
}

} // namespace
} // namespace facebook::velox::connector::tpch::test
