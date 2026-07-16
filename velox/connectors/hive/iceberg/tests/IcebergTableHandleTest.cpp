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

TEST(IcebergTableHandleTest, changelogQueryFlag) {
  registerAll();

  auto handle = std::make_shared<IcebergTableHandle>(
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
      /*isChangelogQuery=*/true);

  ASSERT_TRUE(handle->isChangelogQuery());
  ASSERT_TRUE(handle->dataColumnHandles().empty());
}

TEST(IcebergTableHandleTest, dataColumnHandles) {
  registerAll();

  auto colHandle = makeIcebergCol("c0", BIGINT());

  std::unordered_map<std::string, IcebergColumnHandlePtr> dataColHandles{
      {"c0", colHandle}};

  auto handle = std::make_shared<IcebergTableHandle>(
      "test-iceberg",
      "test_table",
      common::SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/1.0,
      /*dbName=*/"",
      /*isChangelogQuery=*/false,
      dataColHandles);

  ASSERT_EQ(handle->dataColumnHandles().size(), 1);
  ASSERT_NE(
      handle->dataColumnHandles().find("c0"),
      handle->dataColumnHandles().end());
}

// ---------------------------------------------------------------------------
// toString
// ---------------------------------------------------------------------------

TEST(IcebergTableHandleTest, toStringDefault) {
  registerAll();

  auto handle = makeMinimal();
  const auto str = handle->toString();

  ASSERT_NE(str.find("test_table"), std::string::npos);
  // Neither changelog flag nor column handles section present.
  ASSERT_EQ(str.find("changelog_query"), std::string::npos);
  ASSERT_EQ(str.find("data_column_handles"), std::string::npos);
}

TEST(IcebergTableHandleTest, toStringChangelogQuery) {
  registerAll();

  auto handle = std::make_shared<IcebergTableHandle>(
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
      /*isChangelogQuery=*/true);

  ASSERT_NE(
      handle->toString().find("changelog_query: true"), std::string::npos);
}

TEST(IcebergTableHandleTest, toStringDataColumnHandles) {
  registerAll();

  std::unordered_map<std::string, IcebergColumnHandlePtr> dataColHandles{
      {"c0", makeIcebergCol("c0", BIGINT())}};
  auto handle = std::make_shared<IcebergTableHandle>(
      "test-iceberg",
      "test_table",
      common::SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/1.0,
      /*dbName=*/"",
      /*isChangelogQuery=*/false,
      /*dataColumnHandles=*/std::move(dataColHandles));

  const auto str = handle->toString();
  ASSERT_NE(str.find("data_column_handles"), std::string::npos);
  ASSERT_NE(str.find("c0"), std::string::npos);
}

// ---------------------------------------------------------------------------
// SerDe round-trips
// ---------------------------------------------------------------------------

TEST(IcebergTableHandleTest, serdeMinimal) {
  registerAll();

  auto handle = makeMinimal("conn-1", "orders");
  auto obj = handle->serialize();

  ASSERT_EQ(obj["name"].asString(), "IcebergTableHandle");
  ASSERT_EQ(obj["connectorId"].asString(), "conn-1");
  ASSERT_EQ(obj["tableName"].asString(), "orders");
  ASSERT_FALSE(obj["isChangelogQuery"].asBool());

  auto clone =
      ISerializable::deserialize<IcebergTableHandle>(obj, /*context=*/nullptr);
  ASSERT_EQ(clone->tableName(), "orders");
  ASSERT_EQ(clone->name(), handle->name());
  ASSERT_FALSE(clone->isChangelogQuery());
  ASSERT_TRUE(clone->dataColumnHandles().empty());
}

TEST(IcebergTableHandleTest, serdeChangelogQuery) {
  registerAll();

  auto handle = std::make_shared<IcebergTableHandle>(
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
      /*isChangelogQuery=*/true);

  auto clone = ISerializable::deserialize<IcebergTableHandle>(
      handle->serialize(), /*context=*/nullptr);

  ASSERT_TRUE(clone->isChangelogQuery());
  ASSERT_EQ(clone->tableName(), "cdc_table");
}

TEST(IcebergTableHandleTest, serdeDataColumnHandles) {
  registerAll();

  std::unordered_map<std::string, IcebergColumnHandlePtr> dataColHandles{
      {"id", makeIcebergCol("id", BIGINT())}};
  auto handle = std::make_shared<IcebergTableHandle>(
      "test-iceberg",
      "test_table",
      common::SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/1.0,
      /*dbName=*/"",
      /*isChangelogQuery=*/false,
      /*dataColumnHandles=*/std::move(dataColHandles));

  auto clone = ISerializable::deserialize<IcebergTableHandle>(
      handle->serialize(), /*context=*/nullptr);

  ASSERT_EQ(clone->dataColumnHandles().size(), 1);
  ASSERT_NE(
      clone->dataColumnHandles().find("id"), clone->dataColumnHandles().end());

  const auto& clonedCol = clone->dataColumnHandles().at("id");
  ASSERT_EQ(clonedCol->name(), "id");
  ASSERT_EQ(*clonedCol->dataType(), *BIGINT());
  ASSERT_EQ(clonedCol->field().fieldId, 1);
}

TEST(IcebergTableHandleTest, serdeDbNameAndTableParameters) {
  registerAll();

  auto handle = std::make_shared<IcebergTableHandle>(
      "test-iceberg",
      "events",
      common::SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/
      std::unordered_map<std::string, std::string>{{"format", "parquet"}},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/1.0,
      /*dbName=*/"analytics");

  auto clone = ISerializable::deserialize<IcebergTableHandle>(
      handle->serialize(), /*context=*/nullptr);

  ASSERT_EQ(clone->dbName(), "analytics");
  ASSERT_EQ(clone->tableName(), "events");
  ASSERT_EQ(clone->tableParameters().at("format"), "parquet");
}

TEST(IcebergTableHandleTest, serdeDataColumns) {
  registerAll();

  auto dataColumns = ROW({{"c0", BIGINT()}, {"c1", VARCHAR()}});

  auto handle = std::make_shared<IcebergTableHandle>(
      "test-iceberg",
      "test_table",
      common::SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      dataColumns);

  auto clone = ISerializable::deserialize<IcebergTableHandle>(
      handle->serialize(), /*context=*/nullptr);

  ASSERT_NE(clone->dataColumns(), nullptr);
  ASSERT_EQ(clone->dataColumns()->toString(), dataColumns->toString());
}

TEST(IcebergTableHandleTest, serdeSampleRate) {
  registerAll();

  auto handle = std::make_shared<IcebergTableHandle>(
      "test-iceberg",
      "sampled_table",
      common::SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/0.1);

  auto clone = ISerializable::deserialize<IcebergTableHandle>(
      handle->serialize(), /*context=*/nullptr);

  ASSERT_DOUBLE_EQ(clone->sampleRate(), 0.1);
}
