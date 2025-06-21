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
#include <folly/dynamic.h>
#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/connectors/hive/HiveColumnHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveObjectFactory.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/type/Filter.h"
#include "velox/type/tests/TypeTestUtil.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive;
using facebook::velox::connector::LocationHandle;

static constexpr char kConnectorId[] = "hive-test";

TEST(HiveObjectFactoryTest, MakeConnectorSplitDefaults) {
  HiveObjectFactory factory;

  // No options: only filePath/start/length/connectorId are set
  auto splitPtr = factory.makeConnectorSplit(
      kConnectorId, "s3://bucket/path/file.orc", 123, 456);
  auto* split = dynamic_cast<HiveConnectorSplit*>(splitPtr.get());
  ASSERT_NE(split, nullptr);
  EXPECT_EQ(split->filePath(), "s3://bucket/path/file.orc");
  EXPECT_EQ(split->start(), 123);
  EXPECT_EQ(split->length(), 456);
  EXPECT_EQ(split->connectorId(), kConnectorId);

  // Defaults: DWRF format, weight=1, cacheable=false
  EXPECT_EQ(split->fileFormat(), dwio::common::FileFormat::DWRF);
  EXPECT_EQ(split->splitWeight(), 1);
  EXPECT_FALSE(split->cacheable());
}

TEST(HiveObjectFactoryTest, MakeConnectorSplitWithOptions) {
  HiveObjectFactory factory;
  folly::dynamic opts = folly::dynamic::object(
      "fileFormat", static_cast<int>(dwio::common::FileFormat::PARQUET))(
      "splitWeight", 42)("cacheable", true)(
      "infoColumns", folly::dynamic::object("colA", "infoA")("colB", "infoB"))(
      "partitionKeys", folly::dynamic::object("p1", "v1"));

  auto split =
      factory.makeConnectorSplit(kConnectorId, "/tmp/f.p", 0, 10, opts);
  //  auto* split = dynamic_cast<HiveConnectorSplit*>(splitPtr.get());
  ASSERT_NE(split, nullptr);

  EXPECT_EQ(split->fileFormat(), dwio::common::FileFormat::PARQUET);
  EXPECT_EQ(split->splitWeight(), 42);
  EXPECT_TRUE(split->cacheable());

  // infoColumns
  auto info = split->infoColumns();
  EXPECT_EQ(info.at("colA"), "infoA");
  EXPECT_EQ(info.at("colB"), "infoB");

  // partitionKeys
  auto parts = split->partitionKeys();
  ASSERT_EQ(parts.size(), 1);
  EXPECT_EQ(parts.at(0).first, "p1");
  EXPECT_EQ(*parts.at(0).second, "v1");
}

TEST(HiveObjectFactoryTest, MakeTableHandleWithOptions) {
  HiveObjectFactory factory;

  // Build a RowType for data columns: two ints
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), INTEGER()});

  // Options: disable filter pushdown, add subfield filter & remaining filter &
  // tableParameters
  folly::dynamic opts = folly::dynamic::object("filterPushdownEnabled", false)(
      "subfieldFilters",
      folly::dynamic::object("c0", folly::dynamic::array("x", "y")))(
      "remainingFilter",
      facebook::velox::core::test::PlanBuilder().captureExpression("c0 > 5"))(
      "tableParameters", folly::dynamic::object("pA", "vA"));

  auto handlePtr = factory.makeTableHandle(kConnectorId, "tbl", rowType, opts);
  auto* hiveHandle = dynamic_cast<HiveTableHandle*>(handlePtr.get());
  ASSERT_NE(hiveHandle, nullptr);

  EXPECT_FALSE(hiveHandle->filterPushdownEnabled());
  // subfieldFilters round-trip via fromDynamic()
  auto filters = hiveHandle->subfieldFilters();
  ASSERT_TRUE(
      filters.at("c0")[0].at(0).begin() == std::vector<std::string>{"x", "y"});
  // remainingFilter: non-null
  ASSERT_NE(hiveHandle->remainingFilter(), nullptr);

  auto params = hiveHandle->tableParameters();
  EXPECT_EQ(params.at("pA"), "vA");
}

TEST(HiveObjectFactoryTest, MakeColumnHandle) {
  HiveObjectFactory factory;
  // DataType = VARCHAR, HiveType = VARCHAR (serialize via TypeTestUtil)
  folly::dynamic opts = folly::dynamic::object("columnType", "partition_key")(
      "hiveType",
      facebook::velox::test::TypeTestUtil::toTypePtr(VARCHAR())->serialize())(
      "requiredSubfields", folly::dynamic::array("f1", "f2"));

  auto colHandle =
      factory.makeColumnHandle(kConnectorId, "colX", BIGINT(), opts);
  auto* hiveColumnHandle = dynamic_cast<HiveColumnHandle*>(colHandle.get());
  ASSERT_NE(hiveColumnHandle, nullptr);

  EXPECT_EQ(hiveColumnHandle->name(), "colX");
  EXPECT_EQ(
      hiveColumnHandle->columnType(),
      HiveColumnHandle::ColumnType::kPartitionKey);
  EXPECT_EQ(hiveColumnHandle->dataType(), BIGINT());
  EXPECT_EQ(hiveColumnHandle->hiveType(), VARCHAR());
  EXPECT_EQ(
      hiveColumnHandle->requiredSubfields(),
      std::vector<std::string>({"f1", "f2"}));
}

TEST(HiveObjectFactoryTest, MakeLocationHandle) {
  HiveObjectFactory factory;

  // Default: writeDirectory == targetDirectory
  auto locationHandle1 = factory.makeLocationHandle(kConnectorId, "/tmp/out");
  EXPECT_EQ(locationHandle1->targetDirectory(), "/tmp/out");
  EXPECT_EQ(locationHandle1->writeDirectory(), "/tmp/out");
  EXPECT_EQ(locationHandle1->tableType(), LocationHandle::TableType::kNew);

  // Explicit writeDirectory and tableType
  auto locationHandle2 = factory.makeLocationHandle(
      kConnectorId,
      "/tmp/tgt",
      "/tmp/write",
      LocationHandle::TableType::kExisting);
  EXPECT_EQ(locationHandle2->targetDirectory(), "/tmp/tgt");
  EXPECT_EQ(locationHandle2->writeDirectory(), "/tmp/write");
  EXPECT_EQ(locationHandle2->tableType(), LocationHandle::TableType::kExisting);
}
