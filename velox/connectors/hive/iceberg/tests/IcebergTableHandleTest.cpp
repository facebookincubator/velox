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

// facebook::velox
using facebook::velox::BIGINT;
using facebook::velox::DOUBLE;
using facebook::velox::ISerializable;
using facebook::velox::ROW;
using facebook::velox::Type;
using facebook::velox::TypePtr;
using facebook::velox::VARCHAR;
using facebook::velox::dwio::common::ParquetFieldId;

// facebook::velox::common
using facebook::velox::common::Subfield;
using facebook::velox::common::SubfieldFilters;

// facebook::velox::connector::hive
using facebook::velox::connector::hive::FileColumnHandle;
using facebook::velox::connector::hive::HiveColumnHandle;
using facebook::velox::connector::hive::HiveTableHandle;

// facebook::velox::connector::hive::iceberg
using facebook::velox::connector::hive::iceberg::IcebergColumnHandle;
using facebook::velox::connector::hive::iceberg::IcebergColumnHandlePtr;
using facebook::velox::connector::hive::iceberg::IcebergFieldMetadata;
using facebook::velox::connector::hive::iceberg::IcebergTableHandle;

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
      ParquetFieldId{fieldId, {}});
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
      /*subfieldFilters=*/SubfieldFilters{},
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
          SubfieldFilters{},
          /*remainingFilter=*/nullptr,
          /*dataColumns=*/nullptr,
          /*indexColumns=*/std::vector<std::string>{},
          /*tableParameters=*/std::unordered_map<std::string, std::string>{},
          /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
          /*sampleRate=*/1.0,
          /*dbName=*/"",
          /*dataColumnFieldIds=*/std::vector<int32_t>{},
          /*isChangelogQuery=*/true,
          /*dataColumnHandles=*/
          std::unordered_map<std::string, IcebergColumnHandlePtr>{}),
      "dataColumnHandles must not be empty when isChangelogQuery is true");
}

// ---------------------------------------------------------------------------
// toString — fully-populated IcebergColumnHandle inside dataColumnHandles.
// Covers: nested fieldId children, initialDefaultValue, sorted map order.
// ---------------------------------------------------------------------------

TEST(IcebergTableHandleTest, toString) {
  registerAll();

  // Two data columns to exercise sorted output ("id" < "payload").
  // "id": leaf fieldId=3, no default.
  auto idCol = makeIcebergCol("id", BIGINT(), /*fieldId=*/3);

  // "payload": nested fieldId=10[11, 12], with initialDefaultValue.
  ParquetFieldId nestedField{
      10, {ParquetFieldId{11, {}}, ParquetFieldId{12, {}}}};
  auto payloadCol = std::make_shared<IcebergColumnHandle>(
      "payload",
      FileColumnHandle::ColumnType::kRegular,
      ROW({{"key", BIGINT()}, {"value", VARCHAR()}}),
      nestedField,
      /*requiredSubfields=*/std::vector<Subfield>{},
      /*initialDefaultValue=*/std::optional<std::string>{"{}"}); // default
                                                                 // empty struct

  std::unordered_map<std::string, IcebergColumnHandlePtr> dataColumnHandles = {
      {"id", idCol},
      {"payload", payloadCol},
  };

  auto dataColumns = ROW({{"c0", BIGINT()}, {"c1", VARCHAR()}});
  auto handle = std::make_shared<IcebergTableHandle>(
      "test-iceberg",
      "cdc_table",
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      dataColumns,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/
      std::unordered_map<std::string, std::string>{{"format", "parquet"}},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/0.5,
      /*dbName=*/"analytics",
      /*dataColumnFieldIds=*/std::vector<int32_t>{},
      /*isChangelogQuery=*/true,
      dataColumnHandles);

  // dataColumnHandles_ is sorted by name: "id" < "payload".
  ASSERT_EQ(
      handle->toString(),
      "table: cdc_table"
      ", sample rate: 0.5"
      ", data columns: ROW<c0:BIGINT,c1:VARCHAR>"
      ", table parameters: [format:parquet]"
      ", isChangelogQuery: true"
      ", dataColumnHandles: ["
      "id: IcebergColumnHandle [name: id, columnType: Regular,"
      " dataType: BIGINT, requiredSubfields: [ ], field: 3]"
      ", payload: IcebergColumnHandle [name: payload, columnType: Regular,"
      " dataType: ROW<key:BIGINT,value:VARCHAR>, requiredSubfields: [ ],"
      " field: 10[11, 12], initialDefaultValue: {}]"
      "]");
}

// ---------------------------------------------------------------------------
// SerDe round-trip — single fully-populated object, compare clone vs handle
// ---------------------------------------------------------------------------

TEST(IcebergTableHandleTest, serde) {
  registerAll();

  auto dataColumns = ROW({{"c0", BIGINT()}, {"c1", VARCHAR()}});
  auto handle = std::make_shared<IcebergTableHandle>(
      "test-iceberg",
      "cdc_table",
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      dataColumns,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/
      std::unordered_map<std::string, std::string>{{"format", "parquet"}},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/0.1,
      /*dbName=*/"analytics",
      /*dataColumnFieldIds=*/std::vector<int32_t>{},
      /*isChangelogQuery=*/true,
      /*dataColumnHandles=*/singleColHandles("id", BIGINT(), /*fieldId=*/7));

  auto obj = handle->serialize();
  ASSERT_EQ(obj["name"].asString(), "IcebergTableHandle");
  ASSERT_EQ(obj["connectorId"].asString(), handle->connectorId());
  ASSERT_EQ(obj["tableName"].asString(), handle->tableName());
  ASSERT_EQ(obj["isChangelogQuery"].asBool(), handle->isChangelogQuery());

  auto clone =
      ISerializable::deserialize<IcebergTableHandle>(obj, /*context=*/nullptr);

  ASSERT_EQ(clone->connectorId(), handle->connectorId());
  ASSERT_EQ(clone->tableName(), handle->tableName());
  ASSERT_EQ(clone->dbName(), handle->dbName());
  ASSERT_DOUBLE_EQ(clone->sampleRate(), handle->sampleRate());
  ASSERT_EQ(clone->tableParameters(), handle->tableParameters());
  ASSERT_NE(clone->dataColumns(), nullptr);
  ASSERT_EQ(
      clone->dataColumns()->toString(), handle->dataColumns()->toString());
  ASSERT_EQ(clone->isChangelogQuery(), handle->isChangelogQuery());
  ASSERT_EQ(
      clone->dataColumnHandles().size(), handle->dataColumnHandles().size());
  const auto& col = clone->dataColumnHandles().at("id");
  ASSERT_EQ(col->name(), handle->dataColumnHandles().at("id")->name());
  ASSERT_EQ(
      *col->dataType(), *handle->dataColumnHandles().at("id")->dataType());
  ASSERT_EQ(
      col->field().fieldId,
      handle->dataColumnHandles().at("id")->field().fieldId);
}

// ---------------------------------------------------------------------------
// SerDe round-trips — targeted field combinations, compare clone vs handle
// ---------------------------------------------------------------------------

// Minimal handle: only connectorId, tableName, subfieldFilters,
// remainingFilter. All optional fields must deserialize to their defaults.
TEST(IcebergTableHandleTest, serdeMinimal) {
  registerAll();

  auto handle = makeMinimal();
  auto clone = ISerializable::deserialize<IcebergTableHandle>(
      handle->serialize(), /*context=*/nullptr);

  ASSERT_EQ(clone->connectorId(), handle->connectorId());
  ASSERT_EQ(clone->tableName(), handle->tableName());
  ASSERT_TRUE(clone->subfieldFilters().empty());
  ASSERT_EQ(clone->remainingFilter(), handle->remainingFilter());
  ASSERT_EQ(clone->dataColumns(), handle->dataColumns());
  ASSERT_EQ(clone->indexColumns(), handle->indexColumns());
  ASSERT_EQ(clone->tableParameters(), handle->tableParameters());
  ASSERT_DOUBLE_EQ(clone->sampleRate(), handle->sampleRate());
  ASSERT_EQ(clone->dbName(), handle->dbName());
  ASSERT_EQ(clone->isChangelogQuery(), handle->isChangelogQuery());
  ASSERT_EQ(
      clone->dataColumnHandles().size(), handle->dataColumnHandles().size());
}

// sampleRate < 1.0 is serialized and round-trips correctly.
TEST(IcebergTableHandleTest, serdeSampleRate) {
  registerAll();

  auto handle = std::make_shared<IcebergTableHandle>(
      "c",
      "t",
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/0.25);

  auto clone = ISerializable::deserialize<IcebergTableHandle>(
      handle->serialize(), /*context=*/nullptr);
  ASSERT_DOUBLE_EQ(clone->sampleRate(), handle->sampleRate());
}

// dbName is serialized only when non-empty and round-trips correctly.
TEST(IcebergTableHandleTest, serdeDbName) {
  registerAll();

  auto handle = std::make_shared<IcebergTableHandle>(
      "c",
      "t",
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/1.0,
      /*dbName=*/"warehouse");

  auto obj = handle->serialize();
  ASSERT_TRUE(obj.count("dbName"));
  ASSERT_EQ(obj["dbName"].asString(), handle->dbName());

  auto clone =
      ISerializable::deserialize<IcebergTableHandle>(obj, /*context=*/nullptr);
  ASSERT_EQ(clone->dbName(), handle->dbName());

  // Omitting dbName produces empty string on deserialization.
  auto minimal = makeMinimal();
  auto minimalObj = minimal->serialize();
  ASSERT_EQ(minimalObj.count("dbName"), 0);
  auto cloneNoDb = ISerializable::deserialize<IcebergTableHandle>(
      minimalObj, /*context=*/nullptr);
  ASSERT_EQ(cloneNoDb->dbName(), minimal->dbName());
}

// tableParameters round-trip with multiple entries.
TEST(IcebergTableHandleTest, serdeTableParameters) {
  registerAll();

  const std::unordered_map<std::string, std::string> params = {
      {"format", "parquet"},
      {"write.target-file-size-bytes", "134217728"},
      {"sort-order", "id ASC NULLS LAST"},
  };

  auto handle = std::make_shared<IcebergTableHandle>(
      "c",
      "t",
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/params);

  auto clone = ISerializable::deserialize<IcebergTableHandle>(
      handle->serialize(), /*context=*/nullptr);
  ASSERT_EQ(clone->tableParameters(), handle->tableParameters());
}

// dataColumnFieldIds non-empty round-trip.
TEST(IcebergTableHandleTest, serdeDataColumnFieldIds) {
  registerAll();

  auto dataColumns = ROW({{"c0", BIGINT()}, {"c1", VARCHAR()}});
  const std::vector<int32_t> fieldIds = {10, 20};

  auto handle = std::make_shared<IcebergTableHandle>(
      "c",
      "t",
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      dataColumns,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/1.0,
      /*dbName=*/"",
      /*dataColumnFieldIds=*/fieldIds);

  auto obj = handle->serialize();
  ASSERT_TRUE(obj.count("dataColumnFieldIds"));

  auto clone =
      ISerializable::deserialize<IcebergTableHandle>(obj, /*context=*/nullptr);
  ASSERT_EQ(clone->dataColumnFieldIds(), handle->dataColumnFieldIds());

  // Omitting dataColumnFieldIds produces an empty vector.
  auto minimal = makeMinimal();
  auto minimalObj = minimal->serialize();
  ASSERT_EQ(minimalObj.count("dataColumnFieldIds"), 0);
  auto cloneNoIds = ISerializable::deserialize<IcebergTableHandle>(
      minimalObj, /*context=*/nullptr);
  ASSERT_TRUE(cloneNoIds->dataColumnFieldIds().empty());
}

// indexColumns non-empty round-trip.
TEST(IcebergTableHandleTest, serdeIndexColumns) {
  registerAll();

  const std::vector<std::string> idx = {"id", "event_ts"};
  auto handle = std::make_shared<IcebergTableHandle>(
      "c",
      "t",
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/idx);

  auto clone = ISerializable::deserialize<IcebergTableHandle>(
      handle->serialize(), /*context=*/nullptr);
  ASSERT_EQ(clone->indexColumns(), handle->indexColumns());
}

// dataColumnHandles with nested ParquetFieldId children.
TEST(IcebergTableHandleTest, serdeDataColumnHandlesNested) {
  registerAll();

  // A nested field: fieldId=10 with two children (fieldId=11, fieldId=12).
  const ParquetFieldId nestedField{
      10,
      {
          ParquetFieldId{11, {}},
          ParquetFieldId{12, {}},
      }};

  auto col = std::make_shared<IcebergColumnHandle>(
      "payload",
      FileColumnHandle::ColumnType::kRegular,
      ROW({{"key", BIGINT()}, {"value", VARCHAR()}}),
      nestedField);

  const std::unordered_map<std::string, IcebergColumnHandlePtr>
      dataColumnHandles = {{"payload", col}};

  auto handle = std::make_shared<IcebergTableHandle>(
      "c",
      "t",
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/1.0,
      /*dbName=*/"",
      /*dataColumnFieldIds=*/std::vector<int32_t>{},
      /*isChangelogQuery=*/true,
      dataColumnHandles);

  auto clone = ISerializable::deserialize<IcebergTableHandle>(
      handle->serialize(), /*context=*/nullptr);

  ASSERT_EQ(
      clone->dataColumnHandles().size(), handle->dataColumnHandles().size());
  const auto& src = handle->dataColumnHandles().at("payload");
  const auto& restored = clone->dataColumnHandles().at("payload");
  ASSERT_EQ(restored->field().fieldId, src->field().fieldId);
  ASSERT_EQ(restored->field().children.size(), src->field().children.size());
  ASSERT_EQ(
      restored->field().children[0].fieldId, src->field().children[0].fieldId);
  ASSERT_EQ(
      restored->field().children[1].fieldId, src->field().children[1].fieldId);
}

// dataColumnHandles with initialDefaultValue set.
TEST(IcebergTableHandleTest, serdeDataColumnHandlesInitialDefaultValue) {
  registerAll();

  auto col = std::make_shared<IcebergColumnHandle>(
      "status",
      FileColumnHandle::ColumnType::kRegular,
      VARCHAR(),
      ParquetFieldId{20, {}},
      /*requiredSubfields=*/std::vector<Subfield>{},
      /*initialDefaultValue=*/std::optional<std::string>{"active"});

  const std::unordered_map<std::string, IcebergColumnHandlePtr>
      dataColumnHandles = {{"status", col}};

  auto handle = std::make_shared<IcebergTableHandle>(
      "c",
      "t",
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/1.0,
      /*dbName=*/"",
      /*dataColumnFieldIds=*/std::vector<int32_t>{},
      /*isChangelogQuery=*/true,
      dataColumnHandles);

  auto clone = ISerializable::deserialize<IcebergTableHandle>(
      handle->serialize(), /*context=*/nullptr);

  ASSERT_EQ(
      clone->dataColumnHandles().size(), handle->dataColumnHandles().size());
  const auto& src = handle->dataColumnHandles().at("status");
  const auto& restored = clone->dataColumnHandles().at("status");
  ASSERT_EQ(restored->initialDefaultValue(), src->initialDefaultValue());
}

// Multiple dataColumnHandles entries — all names and fieldIds survive the trip.
TEST(IcebergTableHandleTest, serdeDataColumnHandlesMultiple) {
  registerAll();

  const std::unordered_map<std::string, IcebergColumnHandlePtr>
      dataColumnHandles = {
          {"id", makeIcebergCol("id", BIGINT(), 1)},
          {"name", makeIcebergCol("name", VARCHAR(), 2)},
          {"score", makeIcebergCol("score", DOUBLE(), 3)},
      };

  auto handle = std::make_shared<IcebergTableHandle>(
      "c",
      "t",
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/std::vector<IcebergColumnHandlePtr>{},
      /*sampleRate=*/1.0,
      /*dbName=*/"",
      /*dataColumnFieldIds=*/std::vector<int32_t>{},
      /*isChangelogQuery=*/true,
      dataColumnHandles);

  auto clone = ISerializable::deserialize<IcebergTableHandle>(
      handle->serialize(), /*context=*/nullptr);

  ASSERT_EQ(
      clone->dataColumnHandles().size(), handle->dataColumnHandles().size());
  for (const auto& [name, srcHandle] : handle->dataColumnHandles()) {
    ASSERT_EQ(
        clone->dataColumnHandles().at(name)->field().fieldId,
        srcHandle->field().fieldId);
  }
}

// IcebergColumnHandle icebergMetadata_ round-trip: all V3 attributes and
// recursive children survive serialization.
TEST(IcebergTableHandleTest, serdeIcebergMetadata) {
  registerAll();

  IcebergFieldMetadata childMeta;
  childMeta.required = false;
  childMeta.longType = "LONG";

  IcebergFieldMetadata meta;
  meta.required = true;
  meta.longType = "LONG";
  meta.timestampUnit = "MICROS";
  meta.binaryType = "UUID";
  meta.structType = "VariantStruct";
  meta.length = 16;
  meta.children = {childMeta};

  auto col = std::make_shared<IcebergColumnHandle>(
      "ts",
      FileColumnHandle::ColumnType::kRegular,
      BIGINT(),
      ParquetFieldId{99, {ParquetFieldId{100, {}}}},
      /*requiredSubfields=*/std::vector<Subfield>{},
      /*initialDefaultValue=*/std::nullopt,
      meta);

  auto obj = col->serialize();
  ASSERT_TRUE(obj.count("icebergMetadata"));

  auto restored = ISerializable::deserialize<IcebergColumnHandle>(obj);
  const auto& rm = restored->icebergMetadata();

  ASSERT_EQ(rm.required, meta.required);
  ASSERT_EQ(rm.longType, meta.longType);
  ASSERT_EQ(rm.timestampUnit, meta.timestampUnit);
  ASSERT_EQ(rm.binaryType, meta.binaryType);
  ASSERT_EQ(rm.structType, meta.structType);
  ASSERT_EQ(rm.length, meta.length);
  ASSERT_EQ(rm.children.size(), meta.children.size());
  ASSERT_EQ(rm.children[0].required, meta.children[0].required);
  ASSERT_EQ(rm.children[0].longType, meta.children[0].longType);
}

// Empty icebergMetadata is not written to serialized form.
TEST(IcebergTableHandleTest, serdeIcebergMetadataOmittedWhenEmpty) {
  registerAll();

  auto col = makeIcebergCol("id", BIGINT(), /*fieldId=*/1);
  auto obj = col->serialize();
  // No V3 metadata was set — the key must be absent.
  ASSERT_EQ(obj.count("icebergMetadata"), 0);

  auto restored = ISerializable::deserialize<IcebergColumnHandle>(obj);
  ASSERT_TRUE(restored->icebergMetadata().empty());
  ASSERT_TRUE(restored->icebergMetadata().children.empty());
}

// filterColumnHandles round-trip.
TEST(IcebergTableHandleTest, serdeFilterColumnHandles) {
  registerAll();

  std::vector<IcebergColumnHandlePtr> filterHandles = {
      makeIcebergCol("partition_date", VARCHAR(), 5),
      makeIcebergCol("region_id", BIGINT(), 6),
  };

  auto handle = std::make_shared<IcebergTableHandle>(
      "c",
      "t",
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/filterHandles);

  auto clone = ISerializable::deserialize<IcebergTableHandle>(
      handle->serialize(), /*context=*/nullptr);

  // hiveFilterColumnHandles() returns the base vector of HiveColumnHandlePtr;
  // downcast to IcebergColumnHandle to verify fieldIds.
  const auto& restored = clone->hiveFilterColumnHandles();
  ASSERT_EQ(restored.size(), 2);

  const auto* fh0 = dynamic_cast<const IcebergColumnHandle*>(restored[0].get());
  const auto* fh1 = dynamic_cast<const IcebergColumnHandle*>(restored[1].get());
  ASSERT_NE(fh0, nullptr);
  ASSERT_NE(fh1, nullptr);

  // The order from std::vector<IcebergColumnHandlePtr> is preserved.
  ASSERT_EQ(fh0->name(), "partition_date");
  ASSERT_EQ(fh0->field().fieldId, 5);
  ASSERT_EQ(fh1->name(), "region_id");
  ASSERT_EQ(fh1->field().fieldId, 6);
}

// ---------------------------------------------------------------------------
// Negative tests — malformed / missing Iceberg metadata
// ---------------------------------------------------------------------------

// Deserializing an IcebergTableHandle JSON that is missing the required
// "tableName" key must throw with a clear diagnostic.
TEST(IcebergTableHandleTest, deserializeMissingTableName) {
  registerAll();

  // Build a valid handle, serialize it, then surgically remove "tableName".
  auto obj = makeMinimal()->serialize();
  obj.erase("tableName");

  try {
    ISerializable::deserialize<IcebergTableHandle>(obj, /*context=*/nullptr);
    FAIL() << "Expected std::out_of_range for missing 'tableName'";
  } catch (const std::out_of_range& e) {
    EXPECT_NE(std::string(e.what()).find("tableName"), std::string::npos)
        << "Exception message was: " << e.what();
  }
}

// Deserializing an IcebergColumnHandle JSON that is missing the required
// "field" key (the ParquetFieldId) must throw with a clear diagnostic.
TEST(IcebergTableHandleTest, deserializeMissingFieldKey) {
  registerAll();

  auto col = makeIcebergCol("id", BIGINT(), /*fieldId=*/1);
  auto obj = col->serialize();
  obj.erase("field");

  try {
    ISerializable::deserialize<IcebergColumnHandle>(obj);
    FAIL() << "Expected std::out_of_range for missing 'field'";
  } catch (const std::out_of_range& e) {
    EXPECT_NE(std::string(e.what()).find("field"), std::string::npos)
        << "Exception message was: " << e.what();
  }
}

// Deserializing a ParquetFieldId JSON that is missing the required
// "fieldId" key inside the "field" object must throw with a clear diagnostic.
TEST(IcebergTableHandleTest, deserializeMissingFieldId) {
  registerAll();

  auto col = makeIcebergCol("id", BIGINT(), /*fieldId=*/42);
  auto obj = col->serialize();
  // Remove "fieldId" from the nested "field" object.
  obj["field"].erase("fieldId");

  try {
    ISerializable::deserialize<IcebergColumnHandle>(obj);
    FAIL() << "Expected std::out_of_range for missing 'fieldId'";
  } catch (const std::out_of_range& e) {
    EXPECT_NE(std::string(e.what()).find("fieldId"), std::string::npos)
        << "Exception message was: " << e.what();
  }
}
