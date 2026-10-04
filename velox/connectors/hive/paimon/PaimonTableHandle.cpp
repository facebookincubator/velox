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
#include "velox/connectors/hive/paimon/PaimonTableHandle.h"

#include <unordered_set>

#include "velox/connectors/hive/paimon/PaimonMetadata.h"

namespace facebook::velox::connector::hive::paimon {
namespace {

folly::dynamic serializeIds(const std::vector<int32_t>& ids) {
  folly::dynamic result = folly::dynamic::array;
  for (auto id : ids) {
    result.push_back(id);
  }
  return result;
}

std::vector<int32_t> readIds(const folly::dynamic& array) {
  VELOX_USER_CHECK(array.isArray(), "Paimon field IDs must be an array");
  std::vector<int32_t> result;
  for (const auto& id : array) {
    result.push_back(
        static_cast<int32_t>(paimonInt(
            folly::dynamic::object("id", id),
            "id",
            0,
            std::numeric_limits<int32_t>::max())));
  }
  return result;
}

} // namespace

void PaimonSchema::validate() const {
  VELOX_USER_CHECK_GE(id, 0, "Paimon schemaId out of range");
  VELOX_USER_CHECK_NOT_NULL(rowType, "Paimon schema rowType is required");
  VELOX_USER_CHECK_EQ(
      rowType->size(),
      fieldIds.size(),
      "Paimon schema requires one field ID per column");
  std::unordered_set<int32_t> ids;
  std::unordered_set<std::string> names;
  for (auto i = 0; i < fieldIds.size(); ++i) {
    VELOX_USER_CHECK_GE(fieldIds[i], 0, "Paimon fieldId out of range");
    VELOX_USER_CHECK(
        ids.insert(fieldIds[i]).second, "Duplicate Paimon fieldId");
    VELOX_USER_CHECK(
        !rowType->nameOf(i).empty() && names.insert(rowType->nameOf(i)).second,
        "Empty or duplicate Paimon field name");
  }
}

folly::dynamic PaimonSchema::serialize() const {
  return folly::dynamic::object("schemaId", id)(
      "rowType", rowType->serialize())("fieldIds", serializeIds(fieldIds));
}

PaimonSchema PaimonSchema::create(const folly::dynamic& obj) {
  PaimonSchema schema{
      paimonInt(obj, "schemaId"),
      asRowType(ISerializable::deserialize<Type>(obj["rowType"])),
      readIds(obj["fieldIds"])};
  schema.validate();
  return schema;
}

PaimonColumnHandle::PaimonColumnHandle(
    std::string name,
    int32_t fieldId,
    TypePtr type,
    ColumnType role,
    std::vector<common::Subfield> requiredSubfields)
    : name_(std::move(name)),
      fieldId_(fieldId),
      type_(std::move(type)),
      role_(role),
      requiredSubfields_(std::move(requiredSubfields)) {
  VELOX_USER_CHECK_GE(fieldId_, 0, "Paimon column fieldId out of range");
  VELOX_USER_CHECK_NOT_NULL(type_);
  VELOX_USER_CHECK(!name_.empty(), "Paimon column name must not be empty");
}

folly::dynamic PaimonColumnHandle::serialize() const {
  auto obj = ColumnHandle::serializeBase("PaimonColumnHandle");
  obj["columnName"] = name_;
  obj["fieldId"] = fieldId_;
  obj["type"] = type_->serialize();
  obj["role"] = columnTypeName(role_);
  folly::dynamic fields = folly::dynamic::array;
  for (const auto& field : requiredSubfields_) {
    fields.push_back(field.toString());
  }
  obj["requiredSubfields"] = std::move(fields);
  return obj;
}

ColumnHandlePtr PaimonColumnHandle::create(const folly::dynamic& obj) {
  std::vector<common::Subfield> fields;
  for (const auto& field : obj["requiredSubfields"]) {
    fields.emplace_back(field.asString());
  }
  return std::make_shared<PaimonColumnHandle>(
      obj["columnName"].asString(),
      paimonInt(obj, "fieldId", 0, std::numeric_limits<int32_t>::max()),
      ISerializable::deserialize<Type>(obj["type"]),
      columnTypeFromName(obj["role"].asString()),
      std::move(fields));
}

void PaimonColumnHandle::registerSerDe() {
  DeserializationRegistryForSharedPtr().Register("PaimonColumnHandle", create);
}

PaimonTableHandle::PaimonTableHandle(
    std::string connectorId,
    std::string tableName,
    PaimonTableType tableType,
    int64_t targetSchemaId,
    std::vector<PaimonSchema> schemas,
    std::vector<int32_t> partitionFieldIds,
    std::vector<int32_t> primaryKeyFieldIds,
    common::SubfieldFilters filters,
    core::TypedExprPtr remainingFilter,
    std::unordered_map<std::string, std::string> options)
    : FileTableHandle(std::move(connectorId)),
      tableName_(std::move(tableName)),
      tableType_(tableType),
      targetSchemaId_(targetSchemaId),
      schemas_(std::move(schemas)),
      partitionFieldIds_(std::move(partitionFieldIds)),
      primaryKeyFieldIds_(std::move(primaryKeyFieldIds)),
      filters_(std::move(filters)),
      remainingFilter_(std::move(remainingFilter)),
      options_(std::move(options)) {
  std::unordered_set<int64_t> ids;
  for (const auto& item : schemas_) {
    item.validate();
    VELOX_USER_CHECK(ids.insert(item.id).second, "Duplicate Paimon schemaId");
  }
  const auto& target = targetSchema();
  const auto checkIds = [&](const auto& fields) {
    std::unordered_set<int32_t> unique;
    for (auto id : fields) {
      VELOX_USER_CHECK(
          std::find(target.fieldIds.begin(), target.fieldIds.end(), id) !=
              target.fieldIds.end(),
          "Paimon key fieldId {} is absent from schema",
          id);
      VELOX_USER_CHECK(
          unique.insert(id).second, "Duplicate Paimon key fieldId");
    }
  };
  checkIds(partitionFieldIds_);
  checkIds(primaryKeyFieldIds_);
  VELOX_USER_CHECK_EQ(
      tableType_ == PaimonTableType::kAppendOnly,
      primaryKeyFieldIds_.empty(),
      "Paimon tableType and explicit primary keys disagree");
}

const PaimonSchema& PaimonTableHandle::schema(int64_t id) const {
  for (const auto& item : schemas_) {
    if (item.id == id) {
      return item;
    }
  }
  VELOX_USER_FAIL("Missing Paimon schemaId {} in query schema bundle", id);
}

std::vector<FileColumnHandlePtr> PaimonTableHandle::filterColumnHandles()
    const {
  std::vector<FileColumnHandlePtr> handles;
  const auto& target = targetSchema();
  for (auto i = 0; i < target.fieldIds.size(); ++i) {
    const auto id = target.fieldIds[i];
    const auto isPartition =
        std::find(partitionFieldIds_.begin(), partitionFieldIds_.end(), id) !=
        partitionFieldIds_.end();
    handles.push_back(
        std::make_shared<PaimonColumnHandle>(
            target.rowType->nameOf(i),
            id,
            target.rowType->childAt(i),
            isPartition ? FileColumnHandle::ColumnType::kPartitionKey
                        : FileColumnHandle::ColumnType::kRegular));
  }
  return handles;
}

folly::dynamic PaimonTableHandle::serialize() const {
  auto obj = ConnectorTableHandle::serializeBase("PaimonTableHandle");
  obj["wireVersion"] = 1;
  obj["tableName"] = tableName_;
  obj["tableType"] = paimonTableTypeString(tableType_);
  obj["targetSchemaId"] = targetSchemaId_;
  obj["schemas"] = folly::dynamic::array;
  for (const auto& item : schemas_) {
    obj["schemas"].push_back(item.serialize());
  }
  obj["partitionFieldIds"] = serializeIds(partitionFieldIds_);
  obj["primaryKeyFieldIds"] = serializeIds(primaryKeyFieldIds_);
  obj["options"] = folly::dynamic::object;
  for (const auto& [key, value] : options_) {
    obj["options"][key] = value;
  }
  obj["subfieldFilters"] = folly::dynamic::array;
  for (const auto& [field, filter] : filters_) {
    obj["subfieldFilters"].push_back(
        folly::dynamic::object("subfield", field.toString())(
            "filter", filter->serialize()));
  }
  if (remainingFilter_) {
    obj["remainingFilter"] = remainingFilter_->serialize();
  }
  return obj;
}

ConnectorTableHandlePtr PaimonTableHandle::create(
    const folly::dynamic& obj,
    void* context) {
  paimonInt(obj, "wireVersion", 1, 1);
  std::vector<PaimonSchema> schemas;
  for (const auto& item : obj["schemas"]) {
    schemas.push_back(PaimonSchema::create(item));
  }
  common::SubfieldFilters filters;
  for (const auto& item : obj["subfieldFilters"]) {
    VELOX_USER_CHECK(
        filters
            .emplace(
                common::Subfield(item["subfield"].asString()),
                ISerializable::deserialize<common::Filter>(item["filter"])
                    ->clone())
            .second,
        "Duplicate Paimon subfield filter");
  }
  core::TypedExprPtr remaining;
  if (obj.count("remainingFilter")) {
    remaining = ISerializable::deserialize<core::ITypedExpr>(
        obj["remainingFilter"], context);
  }
  std::unordered_map<std::string, std::string> options;
  for (const auto& [key, value] : obj["options"].items()) {
    options.emplace(key.asString(), value.asString());
  }
  return std::make_shared<PaimonTableHandle>(
      obj["connectorId"].asString(),
      obj["tableName"].asString(),
      paimonTableTypeFromString(obj["tableType"].asString()),
      paimonInt(obj, "targetSchemaId"),
      std::move(schemas),
      readIds(obj["partitionFieldIds"]),
      readIds(obj["primaryKeyFieldIds"]),
      std::move(filters),
      std::move(remaining),
      std::move(options));
}

void PaimonTableHandle::registerSerDe() {
  DeserializationWithContextRegistryForSharedPtr().Register(
      "PaimonTableHandle", create);
}

} // namespace facebook::velox::connector::hive::paimon
