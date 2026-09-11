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
// Covers: nested fieldId children, initialDefaultValue, icebergMetadata
// (empty, partial, fully-populated), sorted map order.
// ---------------------------------------------------------------------------

TEST(IcebergTableHandleTest, toString) {
  registerAll();

  // Three data columns to exercise sorted output ("id" < "payload" < "score")
  // and all three icebergMetadata states: partial, empty, fully-populated.

  // "id": partial icebergMetadata (required + longType only).
  IcebergFieldMetadata icebergFieldMetadata;
  icebergFieldMetadata.required = false;
  icebergFieldMetadata.longType = "LONG";
  auto idCol = std::make_shared<IcebergColumnHandle>(
      "id",
      FileColumnHandle::ColumnType::kRegular,
      BIGINT(),
      ParquetFieldId{3, {}},
      /*requiredSubfields=*/std::vector<Subfield>{},
      /*initialDefaultValue=*/std::nullopt,
      icebergFieldMetadata);

  // "payload": nested fieldId=10[11, 12], initialDefaultValue, no metadata.
  ParquetFieldId nestedField{
      10, {ParquetFieldId{11, {}}, ParquetFieldId{12, {}}}};
  auto payloadCol = std::make_shared<IcebergColumnHandle>(
      "payload",
      FileColumnHandle::ColumnType::kRegular,
      ROW({{"key", BIGINT()}, {"value", VARCHAR()}}),
      nestedField,
      /*requiredSubfields=*/std::vector<Subfield>{},
      /*initialDefaultValue=*/std::optional<std::string>{"{}"}); // empty struct

  // "score": fully-populated icebergMetadata, all six attributes set.
  IcebergFieldMetadata fullMeta;
  fullMeta.required = true;
  fullMeta.longType = "LONG";
  fullMeta.timestampUnit = "MICROS";
  fullMeta.binaryType = "UUID";
  fullMeta.structType = "VariantStruct";
  fullMeta.length = 16;
  auto scoreCol = std::make_shared<IcebergColumnHandle>(
      "score",
      FileColumnHandle::ColumnType::kRegular,
      BIGINT(),
      ParquetFieldId{7, {}},
      /*requiredSubfields=*/std::vector<Subfield>{},
      /*initialDefaultValue=*/std::nullopt,
      fullMeta);

  std::unordered_map<std::string, IcebergColumnHandlePtr> dataColumnHandles = {
      {"id", idCol},
      {"payload", payloadCol},
      {"score", scoreCol},
  };

  auto dataColumns =
      ROW({{"c0", BIGINT()}, {"c1", VARCHAR()}, {"c2", BIGINT()}});
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

  // dataColumnHandles_ is sorted by name: "id" < "payload" < "score".
  ASSERT_EQ(
      handle->toString(),
      "table: cdc_table"
      ", sample rate: 0.5"
      ", data columns: ROW<c0:BIGINT,c1:VARCHAR,c2:BIGINT>"
      ", table parameters: [format:parquet]"
      ", isChangelogQuery: true"
      ", dataColumnHandles: ["
      "id: IcebergColumnHandle [name: id, columnType: Regular,"
      " dataType: BIGINT, requiredSubfields: [ ], field: 3,"
      " icebergMetadata: {required: false, longType: LONG}]"
      ", payload: IcebergColumnHandle [name: payload, columnType: Regular,"
      " dataType: ROW<key:BIGINT,value:VARCHAR>, requiredSubfields: [ ],"
      " field: 10[11, 12], initialDefaultValue: {}]"
      ", score: IcebergColumnHandle [name: score, columnType: Regular,"
      " dataType: BIGINT, requiredSubfields: [ ], field: 7,"
      " icebergMetadata: {required: true, longType: LONG,"
      " timestampUnit: MICROS, binaryType: UUID,"
      " structType: VariantStruct, length: 16}]"
      "]");
}

// ---------------------------------------------------------------------------
// SerDe round-trips
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
  ASSERT_EQ(clone->remainingFilter(), nullptr);
  ASSERT_EQ(clone->dataColumns(), nullptr);
  ASSERT_TRUE(clone->dataColumnFieldIds().empty());
  ASSERT_TRUE(clone->indexColumns().empty());
  ASSERT_TRUE(clone->tableParameters().empty());
  ASSERT_DOUBLE_EQ(clone->sampleRate(), 1.0);
  ASSERT_TRUE(clone->dbName().empty());
  ASSERT_FALSE(clone->isChangelogQuery());
  ASSERT_TRUE(clone->dataColumnHandles().empty());
  ASSERT_TRUE(clone->hiveFilterColumnHandles().empty());
}

// Fully-populated round-trip: every serialized field is set to a non-default
// value and verified to survive the serialize/deserialize cycle.
TEST(IcebergTableHandleTest, serdeFullyPopulated) {
  registerAll();

  // dataColumnHandles: leaf + nested field, one with initialDefaultValue.
  const ParquetFieldId nestedField{
      10, {ParquetFieldId{11, {}}, ParquetFieldId{12, {}}}};
  auto payloadCol = std::make_shared<IcebergColumnHandle>(
      "payload",
      FileColumnHandle::ColumnType::kRegular,
      ROW({{"key", BIGINT()}, {"value", VARCHAR()}}),
      nestedField,
      /*requiredSubfields=*/std::vector<Subfield>{},
      /*initialDefaultValue=*/std::optional<std::string>{"active"});

  const std::unordered_map<std::string, IcebergColumnHandlePtr>
      dataColumnHandles = {
          {"id", makeIcebergCol("id", BIGINT(), /*fieldId=*/7)},
          {"payload", payloadCol},
      };

  // filterColumnHandles: two Iceberg-typed handles.
  std::vector<IcebergColumnHandlePtr> filterHandles = {
      makeIcebergCol("partition_date", VARCHAR(), /*fieldId=*/5),
      makeIcebergCol("region_id", BIGINT(), /*fieldId=*/6),
  };

  auto dataColumns = ROW({{"c0", BIGINT()}, {"c1", VARCHAR()}});
  auto handle = std::make_shared<IcebergTableHandle>(
      "test-iceberg",
      "cdc_table",
      SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      dataColumns,
      /*indexColumns=*/std::vector<std::string>{"id", "event_ts"},
      /*tableParameters=*/
      std::unordered_map<std::string, std::string>{
          {"format", "parquet"},
          {"write.target-file-size-bytes", "134217728"},
      },
      /*filterColumnHandles=*/filterHandles,
      /*sampleRate=*/0.1,
      /*dbName=*/"warehouse",
      /*dataColumnFieldIds=*/std::vector<int32_t>{10, 20},
      /*isChangelogQuery=*/true,
      dataColumnHandles);

  auto clone = ISerializable::deserialize<IcebergTableHandle>(
      handle->serialize(), /*context=*/nullptr);

  // HiveTableHandle fields.
  ASSERT_EQ(clone->connectorId(), handle->connectorId());
  ASSERT_EQ(clone->tableName(), handle->tableName());
  ASSERT_EQ(clone->dataColumns()->toString(), dataColumns->toString());
  ASSERT_EQ(clone->dataColumnFieldIds(), handle->dataColumnFieldIds());
  ASSERT_EQ(clone->indexColumns(), handle->indexColumns());
  ASSERT_EQ(clone->tableParameters(), handle->tableParameters());
  ASSERT_DOUBLE_EQ(clone->sampleRate(), handle->sampleRate());
  ASSERT_EQ(clone->dbName(), handle->dbName());

  // filterColumnHandles: concrete type survives deserialization.
  const auto& restoredFilters = clone->hiveFilterColumnHandles();
  ASSERT_EQ(restoredFilters.size(), 2);
  const auto* fh0 =
      dynamic_cast<const IcebergColumnHandle*>(restoredFilters[0].get());
  const auto* fh1 =
      dynamic_cast<const IcebergColumnHandle*>(restoredFilters[1].get());
  ASSERT_NE(fh0, nullptr);
  ASSERT_NE(fh1, nullptr);
  ASSERT_EQ(fh0->name(), "partition_date");
  ASSERT_EQ(fh0->field().fieldId, 5);
  ASSERT_EQ(fh1->name(), "region_id");
  ASSERT_EQ(fh1->field().fieldId, 6);

  // Iceberg-specific fields.
  ASSERT_TRUE(clone->isChangelogQuery());
  ASSERT_EQ(
      clone->dataColumnHandles().size(), handle->dataColumnHandles().size());

  // "id": leaf field.
  const auto& cloneId = clone->dataColumnHandles().at("id");
  ASSERT_EQ(cloneId->field().fieldId, 7);
  ASSERT_EQ(*cloneId->dataType(), *BIGINT());

  // "payload": nested field + initialDefaultValue.
  const auto& clonePayload = clone->dataColumnHandles().at("payload");
  ASSERT_EQ(clonePayload->field().fieldId, nestedField.fieldId);
  ASSERT_EQ(clonePayload->field().children.size(), 2);
  ASSERT_EQ(clonePayload->field().children[0].fieldId, 11);
  ASSERT_EQ(clonePayload->field().children[1].fieldId, 12);
  ASSERT_EQ(
      clonePayload->initialDefaultValue(),
      std::optional<std::string>{"active"});
}

// ---------------------------------------------------------------------------
// IcebergColumnHandle icebergMetadata_ SerDe
// ---------------------------------------------------------------------------

// All V3 attributes and recursive children survive serialization.
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
