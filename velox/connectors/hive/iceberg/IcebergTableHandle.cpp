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

namespace facebook::velox::connector::hive::iceberg {

IcebergTableHandle::IcebergTableHandle(
    std::string connectorId,
    const std::string& tableName,
    common::SubfieldFilters subfieldFilters,
    const core::TypedExprPtr& remainingFilter,
    const RowTypePtr& dataColumns,
    std::vector<std::string> indexColumns,
    const std::unordered_map<std::string, std::string>& tableParameters,
    std::vector<IcebergColumnHandlePtr> filterColumnHandles,
    double sampleRate,
    std::string dbName,
    bool isChangelogQuery,
    std::unordered_map<std::string, IcebergColumnHandlePtr> dataColumnHandles)
    : HiveTableHandle(
          std::move(connectorId),
          tableName,
          std::move(subfieldFilters),
          remainingFilter,
          dataColumns,
          std::move(indexColumns),
          tableParameters,
          std::vector<HiveColumnHandlePtr>(
              filterColumnHandles.begin(),
              filterColumnHandles.end()),
          sampleRate,
          std::move(dbName)),
      isChangelogQuery_(isChangelogQuery),
      dataColumnHandles_(std::move(dataColumnHandles)) {
  VELOX_CHECK(
      !isChangelogQuery_ || !dataColumnHandles_.empty(),
      "dataColumnHandles must not be empty when isChangelogQuery is true");
}

std::string IcebergTableHandle::toString() const {
  std::string base = HiveTableHandle::toString();
  if (isChangelogQuery_) {
    base += ", isChangelogQuery: true";
  }
  if (!dataColumnHandles_.empty()) {
    base += ", dataColumnHandles: [";
    bool first = true;
    for (const auto& [name, handle] : dataColumnHandles_) {
      if (!first) {
        base += ", ";
      }
      base += name + ": " + handle->toString();
      first = false;
    }
    base += "]";
  }
  return base;
}

folly::dynamic IcebergTableHandle::serialize() const {
  folly::dynamic obj =
      ConnectorTableHandle::serializeBase("IcebergTableHandle");
  obj["tableName"] = tableName();

  folly::dynamic subfieldFiltersArr = folly::dynamic::array;
  for (const auto& [subfield, filter] : subfieldFilters()) {
    folly::dynamic pair = folly::dynamic::object;
    pair["subfield"] = subfield.toString();
    pair["filter"] = filter->serialize();
    subfieldFiltersArr.push_back(pair);
  }
  obj["subfieldFilters"] = subfieldFiltersArr;

  if (remainingFilter()) {
    obj["remainingFilter"] = remainingFilter()->serialize();
  }

  if (sampleRate() < 1.0) {
    obj["sampleRate"] = sampleRate();
  }

  if (dataColumns()) {
    obj["dataColumns"] = dataColumns()->serialize();
  }

  folly::dynamic tableParams = folly::dynamic::object;
  for (const auto& [key, value] : tableParameters()) {
    tableParams[key] = value;
  }
  obj["tableParameters"] = tableParams;

  if (!hiveFilterColumnHandles().empty()) {
    folly::dynamic handles = folly::dynamic::array;
    for (const auto& handle : hiveFilterColumnHandles()) {
      handles.push_back(handle->serialize());
    }
    obj["filterColumnHandles"] = handles;
  }

  if (!indexColumns().empty()) {
    folly::dynamic cols = folly::dynamic::array;
    for (const auto& col : indexColumns()) {
      cols.push_back(col);
    }
    obj["indexColumns"] = cols;
  }

  if (!dbName().empty()) {
    obj["dbName"] = dbName();
  }

  obj["isChangelogQuery"] = isChangelogQuery_;

  if (!dataColumnHandles_.empty()) {
    folly::dynamic dataColumnHandlesObj = folly::dynamic::array;
    for (const auto& [name, handle] : dataColumnHandles_) {
      folly::dynamic entry = folly::dynamic::object;
      entry["name"] = name;
      entry["handle"] = handle->serialize();
      dataColumnHandlesObj.push_back(entry);
    }
    obj["dataColumnHandles"] = dataColumnHandlesObj;
  }

  return obj;
}

// static
ConnectorTableHandlePtr IcebergTableHandle::create(
    const folly::dynamic& obj,
    void* context) {
  auto connectorId = obj["connectorId"].asString();
  auto tableName = obj["tableName"].asString();

  core::TypedExprPtr remainingFilter;
  if (auto it = obj.find("remainingFilter"); it != obj.items().end()) {
    remainingFilter =
        ISerializable::deserialize<core::ITypedExpr>(it->second, context);
  }

  common::SubfieldFilters subfieldFilters;
  for (const auto& entry : obj["subfieldFilters"]) {
    common::Subfield subfield(entry["subfield"].asString());
    auto filter = ISerializable::deserialize<common::Filter>(entry["filter"]);
    subfieldFilters[common::Subfield(std::move(subfield.path()))] =
        filter->clone();
  }

  double sampleRate = 1.0;
  if (obj.count("sampleRate")) {
    sampleRate = obj["sampleRate"].asDouble();
  }

  RowTypePtr dataColumns;
  if (auto it = obj.find("dataColumns"); it != obj.items().end()) {
    dataColumns = ISerializable::deserialize<RowType>(it->second, context);
  }

  std::unordered_map<std::string, std::string> tableParameters;
  for (const auto& key : obj["tableParameters"].keys()) {
    tableParameters.emplace(
        key.asString(), obj["tableParameters"][key].asString());
  }

  std::vector<IcebergColumnHandlePtr> filterColumnHandles;
  if (auto it = obj.find("filterColumnHandles"); it != obj.items().end()) {
    for (const auto& handle : it->second) {
      filterColumnHandles.push_back(
          ISerializable::deserialize<IcebergColumnHandle>(handle));
    }
  }

  std::vector<std::string> indexColumns;
  if (auto it = obj.find("indexColumns"); it != obj.items().end()) {
    for (const auto& col : it->second) {
      indexColumns.push_back(col.asString());
    }
  }

  std::string dbName;
  if (auto it = obj.find("dbName"); it != obj.items().end()) {
    dbName = it->second.asString();
  }

  bool isChangelogQuery = false;
  if (auto it = obj.find("isChangelogQuery"); it != obj.items().end()) {
    isChangelogQuery = it->second.asBool();
  }

  std::unordered_map<std::string, IcebergColumnHandlePtr> dataColumnHandles;
  if (auto it = obj.find("dataColumnHandles"); it != obj.items().end()) {
    for (const auto& entry : it->second) {
      auto name = entry["name"].asString();
      auto handle =
          ISerializable::deserialize<IcebergColumnHandle>(entry["handle"]);
      dataColumnHandles.emplace(std::move(name), std::move(handle));
    }
  }

  return std::make_shared<const IcebergTableHandle>(
      connectorId,
      tableName,
      std::move(subfieldFilters),
      remainingFilter,
      dataColumns,
      std::move(indexColumns),
      tableParameters,
      std::move(filterColumnHandles),
      sampleRate,
      std::move(dbName),
      isChangelogQuery,
      std::move(dataColumnHandles));
}

// static
void IcebergTableHandle::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("IcebergTableHandle", IcebergTableHandle::create);
}

} // namespace facebook::velox::connector::hive::iceberg
