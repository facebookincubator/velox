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

#include "velox/connectors/hive/iceberg/IcebergTableHandle.h"

#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/dwio/common/ParquetFieldId.h"
#include "velox/type/Type.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::connector::hive::iceberg;

namespace {

// Registers all SerDe entries needed to round-trip IcebergTableHandle and
// IcebergColumnHandle.
void registerAll() {
  Type::registerSerDe();
  HiveColumnHandle::registerSerDe();
  IcebergColumnHandle::registerSerDe();
  HiveTableHandle::registerSerDe();
  IcebergTableHandle::registerSerDe();
}

// Builds a minimal IcebergColumnHandle for use in table handle tests.
IcebergColumnHandlePtr makeIcebergCol(
    const std::string& name,
    const TypePtr& type,
    int32_t fieldId = 1) {
  return std::make_shared<IcebergColumnHandle>(
      name,
      FileColumnHandle::ColumnType::kRegular,
      type,
      parquet::ParquetFieldId{fieldId, {}});
}

// Returns a single-entry dataColumnHandles map for the given column.
std::unordered_map<std::string, IcebergColumnHandlePtr> singleColHandles(
    const std::string& name,
    const TypePtr& type,
    int32_t fieldId = 1) {
  return {{name, makeIcebergCol(name, type, fieldId)}};
}

// Builds a minimal IcebergTableHandle with default Iceberg fields.
std::shared_ptr<IcebergTableHandle> makeMinimal(
    const std::string& connectorId = "test-iceberg",
    const std::string& tableName = "test_table") {
  return std::make_shared<IcebergTableHandle>(
      connectorId,
      tableName,
      /*subfieldFilters=*/common::SubfieldFilters{},
      /*remainingFilter=*/nullptr);
}

} // namespace

// ---------------------------------------------------------------------------
// Field accessors
// ---------------------------------------------------------------------------

TEST(IcebergTableHandleTest, defaultFields) {
  registerAll();

  auto handle = makeMinimal();

  ASSERT_EQ(handle->tableName(), "test_table");
  ASSERT_EQ(handle->name(), "test_table");
  ASSERT_FALSE(handle->isChangelogQuery());
  ASSERT_TRUE(handle->dataColumnHandles().empty());
  ASSERT_TRUE(handle->subfieldFilters().empty());
  ASSERT_EQ(handle->remainingFilter(), nullptr);
  ASSERT_EQ(handle->sampleRate(), 1.0);
  ASSERT_EQ(handle->dataColumns(), nullptr);
  ASSERT_TRUE(handle->dbName().empty());
}

// isChangelogQuery=true with empty dataColumnHandles must throw.
TEST(IcebergTableHandleTest, changelogQueryRequiresDataColumnHandles) {
  registerAll();

  VELOX_ASSERT_THROW(
      std::make_shared<IcebergTableHandle>(
          "test-iceberg",
          "cdc_table",
          common::SubfieldFilters{},
          /*remainingFilter=*/nullptr,
          /*dataColumns=*/nullptr,
          /*indexColumns=*/std::vector<std::string>{},
          /*tableParameters=*/std::unordered_map<std::string, std::string>{},
          /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
          /*sampleRate=*/1.0,
          /*dbName=*/"",
          /*isChangelogQuery=*/true,
          /*dataColumnHandles=*/
          std::unordered_map<std::string, IcebergColumnHandlePtr>{}),
      "dataColumnHandles must not be empty when isChangelogQuery is true");
}

// ---------------------------------------------------------------------------
// toString — single fully-populated object, assert all Iceberg-specific fields
// ---------------------------------------------------------------------------

TEST(IcebergTableHandleTest, toString) {
  registerAll();

  auto dataColumns = ROW({{"c0", BIGINT()}, {"c1", VARCHAR()}});
  auto handle = std::make_shared<IcebergTableHandle>(
      "test-iceberg",
      "cdc_table",
      common::SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      dataColumns,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/
      std::unordered_map<std::string, std::string>{{"format", "parquet"}},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/0.5,
      /*dbName=*/"analytics",
      /*isChangelogQuery=*/true,
      /*dataColumnHandles=*/singleColHandles("id", BIGINT(), /*fieldId=*/3));

  const auto str = handle->toString();
  ASSERT_NE(str.find("cdc_table"), std::string::npos);
  ASSERT_NE(str.find("isChangelogQuery: true"), std::string::npos);
  ASSERT_NE(str.find("dataColumnHandles"), std::string::npos);
  ASSERT_NE(str.find("id"), std::string::npos);
}

// ---------------------------------------------------------------------------
// SerDe round-trip — single fully-populated object, assert all members
// ---------------------------------------------------------------------------

TEST(IcebergTableHandleTest, serde) {
  registerAll();

  auto dataColumns = ROW({{"c0", BIGINT()}, {"c1", VARCHAR()}});
  auto handle = std::make_shared<IcebergTableHandle>(
      "test-iceberg",
      "cdc_table",
      common::SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      dataColumns,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/
      std::unordered_map<std::string, std::string>{{"format", "parquet"}},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/0.1,
      /*dbName=*/"analytics",
      /*isChangelogQuery=*/true,
      /*dataColumnHandles=*/singleColHandles("id", BIGINT(), /*fieldId=*/7));

  auto obj = handle->serialize();
  ASSERT_EQ(obj["name"].asString(), "IcebergTableHandle");
  ASSERT_EQ(obj["connectorId"].asString(), "test-iceberg");
  ASSERT_EQ(obj["tableName"].asString(), "cdc_table");
  ASSERT_TRUE(obj["isChangelogQuery"].asBool());

  auto clone =
      ISerializable::deserialize<IcebergTableHandle>(obj, /*context=*/nullptr);

  ASSERT_EQ(clone->tableName(), "cdc_table");
  ASSERT_EQ(clone->dbName(), "analytics");
  ASSERT_DOUBLE_EQ(clone->sampleRate(), 0.1);
  ASSERT_EQ(clone->tableParameters().at("format"), "parquet");
  ASSERT_NE(clone->dataColumns(), nullptr);
  ASSERT_EQ(clone->dataColumns()->toString(), dataColumns->toString());
  ASSERT_TRUE(clone->isChangelogQuery());
  ASSERT_EQ(clone->dataColumnHandles().size(), 1);
  const auto& col = clone->dataColumnHandles().at("id");
  ASSERT_EQ(col->name(), "id");
  ASSERT_EQ(*col->dataType(), *BIGINT());
  ASSERT_EQ(col->field().fieldId, 7);
}
