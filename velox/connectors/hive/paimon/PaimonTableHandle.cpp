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

#include <cmath>
#include <unordered_set>

namespace facebook::velox::connector::hive::paimon {
namespace {

RowTypePtr deriveDataColumns(
    const RowTypePtr& rowType,
    const std::vector<std::string>& partitionKeys) {
  VELOX_USER_CHECK_NOT_NULL(rowType, "Missing Paimon table row type");
  if (partitionKeys.empty()) {
    return rowType;
  }
  std::unordered_set<column_index_t> partitions;
  for (const auto& name : partitionKeys) {
    const auto index = rowType->getChildIdxIfExists(name);
    VELOX_USER_CHECK(
        index.has_value(), "Unknown Paimon partition column '{}'", name);
    VELOX_USER_CHECK(
        partitions.insert(*index).second,
        "Duplicate Paimon partition column '{}'",
        name);
  }
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (column_index_t i = 0; i < rowType->size(); ++i) {
    if (!partitions.count(i)) {
      names.push_back(rowType->nameOf(i));
      types.push_back(rowType->childAt(i));
    }
  }
  return ROW(std::move(names), std::move(types));
}

} // namespace

PaimonTableHandle::PaimonTableHandle(
    std::string connectorId,
    std::string tableName,
    int64_t schemaId,
    RowTypePtr rowType,
    std::vector<int32_t> fieldIds,
    PaimonTableType tableType,
    std::vector<std::string> partitionKeys,
    std::unordered_map<std::string, std::string> options,
    common::SubfieldFilters subfieldFilters,
    core::TypedExprPtr remainingFilter,
    std::vector<FileColumnHandlePtr> filterColumnHandles,
    double sampleRate,
    std::string dbName)
    : FileTableHandle(std::move(connectorId)),
      tableName_(std::move(tableName)),
      dbName_(std::move(dbName)),
      schemaId_(schemaId),
      rowType_(std::move(rowType)),
      fieldIds_(std::move(fieldIds)),
      tableType_(tableType),
      partitionKeys_(std::move(partitionKeys)),
      options_(std::move(options)),
      subfieldFilters_(std::move(subfieldFilters)),
      remainingFilter_(std::move(remainingFilter)),
      filterColumnHandles_(std::move(filterColumnHandles)),
      sampleRate_(sampleRate),
      dataColumns_(deriveDataColumns(rowType_, partitionKeys_)) {
  VELOX_USER_CHECK_GE(schemaId_, 0, "Paimon schema ID must be nonnegative");
  VELOX_USER_CHECK_EQ(
      fieldIds_.size(),
      rowType_->size(),
      "Paimon field IDs must be aligned to table columns");
  std::unordered_set<int32_t> ids;
  for (auto id : fieldIds_) {
    VELOX_USER_CHECK_GE(id, 0, "Paimon field ID must be nonnegative");
    VELOX_USER_CHECK(ids.insert(id).second, "Duplicate Paimon field ID {}", id);
  }
  VELOX_USER_CHECK(
      std::isfinite(sampleRate_) && sampleRate_ > 0 && sampleRate_ <= 1,
      "Paimon sample rate must be finite and in (0, 1]");
}

const std::unordered_map<std::string, std::string>&
PaimonTableHandle::tableParameters() const {
  // Native Paimon options are not file-reader parameters such as skip-header.
  static const std::unordered_map<std::string, std::string> kEmpty;
  return kEmpty;
}

std::string PaimonTableHandle::toString() const {
  return fmt::format(
      "PaimonTableHandle[table {}, schemaId {}, tableType {}, dataColumns {}, "
      "sampleRate {}]",
      tableName_,
      schemaId_,
      tableType_,
      dataColumns_->toString(),
      sampleRate_);
}

folly::dynamic PaimonTableHandle::serialize() const {
  folly::dynamic obj = ConnectorTableHandle::serializeBase("PaimonTableHandle");
  obj["version"] = kSerdeVersion;
  obj["tableName"] = tableName_;
  obj["schemaId"] = schemaId_;
  obj["rowType"] = rowType_->serialize();
  obj["fieldIds"] = ISerializable::serialize(fieldIds_);
  obj["tableType"] = paimonTableTypeString(tableType_);
  obj["partitionKeys"] = ISerializable::serialize(partitionKeys_);
  obj["options"] = folly::dynamic::object;
  for (const auto& [key, value] : options_) {
    obj["options"][key] = value;
  }
  folly::dynamic subfieldFilters = folly::dynamic::array;
  for (const auto& [subfield, filter] : subfieldFilters_) {
    VELOX_USER_CHECK_NOT_NULL(filter, "Missing Paimon subfield filter");
    folly::dynamic pair = folly::dynamic::object;
    pair["subfield"] = serializeSubfield(subfield);
    pair["filter"] = filter->serialize();
    subfieldFilters.push_back(pair);
  }
  obj["subfieldFilters"] = subfieldFilters;

  if (remainingFilter_) {
    obj["remainingFilter"] = remainingFilter_->serialize();
  }

  if (sampleRate_ < 1.0) {
    obj["sampleRate"] = sampleRate_;
  }

  if (!filterColumnHandles_.empty()) {
    folly::dynamic filterColumnHandles = folly::dynamic::array;
    for (const auto& handle : filterColumnHandles_) {
      VELOX_USER_CHECK_NOT_NULL(handle, "Missing Paimon filter column handle");
      filterColumnHandles.push_back(handle->serialize());
    }
    obj["filterColumnHandles"] = filterColumnHandles;
  }

  if (!dbName_.empty()) {
    obj["dbName"] = dbName_;
  }

  return obj;
}

ConnectorTableHandlePtr PaimonTableHandle::create(
    const folly::dynamic& obj,
    void* context) {
  VELOX_USER_CHECK(
      obj.at("version").isInt() && obj.at("version").asInt() == kSerdeVersion,
      "Unsupported Paimon table handle wire version");

  core::TypedExprPtr remainingFilter;
  if (auto it = obj.find("remainingFilter");
      it != obj.items().end() && !it->second.isNull()) {
    remainingFilter =
        ISerializable::deserialize<core::ITypedExpr>(it->second, context);
    VELOX_USER_CHECK_NOT_NULL(
        remainingFilter, "Invalid Paimon remaining filter");
  }

  common::SubfieldFilters filters;
  VELOX_USER_CHECK(
      obj["subfieldFilters"].isArray(), "Expected Paimon filter array");
  for (const auto& entry : obj["subfieldFilters"]) {
    auto path = deserializeSubfield(entry["subfield"]);
    auto filter = ISerializable::deserialize<common::Filter>(entry["filter"]);
    VELOX_USER_CHECK_NOT_NULL(filter, "Missing Paimon subfield filter");
    VELOX_USER_CHECK(
        filters.emplace(std::move(path), filter->clone()).second,
        "Duplicate Paimon subfield filter");
  }

  double sampleRate = 1.0;
  if (obj.count("sampleRate")) {
    sampleRate = obj["sampleRate"].asDouble();
  }

  std::vector<FileColumnHandlePtr> filterColumns;
  if (auto it = obj.find("filterColumnHandles"); it != obj.items().end()) {
    VELOX_USER_CHECK(
        it->second.isArray(), "Expected Paimon filter column handle array");
    for (const auto& handle : it->second) {
      filterColumns.push_back(
          ISerializable::deserialize<FileColumnHandle>(handle, context));
    }
  }
  std::unordered_map<std::string, std::string> options;
  for (const auto& [key, value] : obj.at("options").items()) {
    options.emplace(key.asString(), value.asString());
  }

  std::string dbName;
  if (auto it = obj.find("dbName"); it != obj.items().end()) {
    dbName = it->second.asString();
  }

  return std::make_shared<PaimonTableHandle>(
      obj["connectorId"].asString(),
      obj["tableName"].asString(),
      ISerializable::deserialize<int64_t>(obj.at("schemaId")),
      ISerializable::deserialize<RowType>(obj.at("rowType"), context),
      ISerializable::deserialize<std::vector<int32_t>>(obj.at("fieldIds")),
      paimonTableTypeFromString(obj.at("tableType").asString()),
      ISerializable::deserialize<std::vector<std::string>>(
          obj.at("partitionKeys")),
      std::move(options),
      std::move(filters),
      std::move(remainingFilter),
      std::move(filterColumns),
      sampleRate,
      std::move(dbName));
}

void PaimonTableHandle::registerSerDe() {
  DeserializationWithContextRegistryForSharedPtr().Register(
      "PaimonTableHandle", create);
}

} // namespace facebook::velox::connector::hive::paimon
