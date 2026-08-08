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

#include <map>

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
  if (isChangelogQuery_) {
    VELOX_USER_CHECK(
        !dataColumnHandles_.empty(),
        "dataColumnHandles must not be empty when isChangelogQuery is true");
  }
}

std::string IcebergTableHandle::toString() const {
  std::string base = HiveTableHandle::toString();
  if (isChangelogQuery_) {
    base += ", isChangelogQuery: true";
  }
  if (!dataColumnHandles_.empty()) {
    // Sort by name for deterministic output, mirroring HiveTableHandle's
    // treatment of subfieldFilters and tableParameters.
    std::map<std::string, const IcebergColumnHandle*> ordered;
    for (const auto& [name, handle] : dataColumnHandles_) {
      ordered[name] = handle.get();
    }
    base += ", dataColumnHandles: [";
    bool first = true;
    for (const auto& [name, handle] : ordered) {
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
  // Start from the common Hive fields under the "IcebergTableHandle" type name,
  // then append Iceberg-specific fields.
  folly::dynamic obj = serializeHiveFields("IcebergTableHandle");

  obj["isChangelogQuery"] = isChangelogQuery_;

  if (!dataColumnHandles_.empty()) {
    folly::dynamic dataColObj = folly::dynamic::object;
    for (const auto& [name, handle] : dataColumnHandles_) {
      dataColObj[name] = handle->serialize();
    }
    obj["dataColumnHandles"] = dataColObj;
  }

  return obj;
}

// static
ConnectorTableHandlePtr IcebergTableHandle::create(
    const folly::dynamic& obj,
    void* context) {
  // Declare locals for all common Hive fields; the base helper fills them.
  // filterColumnHandles is intentionally left empty here — it is parsed below
  // as IcebergColumnHandlePtr to preserve the concrete type.
  std::string connectorId, tableName, dbName;
  common::SubfieldFilters subfieldFilters;
  core::TypedExprPtr remainingFilter;
  double sampleRate;
  RowTypePtr dataColumns;
  std::unordered_map<std::string, std::string> tableParameters;
  std::vector<HiveColumnHandlePtr> unusedFilterHandles; // consumed by helper
  std::vector<std::string> indexColumns;

  deserializeHiveFields(
      obj,
      context,
      connectorId,
      tableName,
      subfieldFilters,
      remainingFilter,
      sampleRate,
      dataColumns,
      tableParameters,
      unusedFilterHandles,
      indexColumns,
      dbName);

  // Re-read filterColumnHandles as IcebergColumnHandlePtr so the concrete
  // type is preserved and callers never need a dynamic_cast.
  std::vector<IcebergColumnHandlePtr> filterColumnHandles;
  if (auto it = obj.find("filterColumnHandles"); it != obj.items().end()) {
    for (const auto& handle : it->second) {
      filterColumnHandles.push_back(
          ISerializable::deserialize<IcebergColumnHandle>(handle));
    }
  }

  bool isChangelogQuery = false;
  if (auto it = obj.find("isChangelogQuery"); it != obj.items().end()) {
    isChangelogQuery = it->second.asBool();
  }

  std::unordered_map<std::string, IcebergColumnHandlePtr> dataColumnHandles;
  if (auto it = obj.find("dataColumnHandles"); it != obj.items().end()) {
    for (const auto& key : it->second.keys()) {
      auto name = key.asString();
      auto handle =
          ISerializable::deserialize<IcebergColumnHandle>(it->second[key]);
      dataColumnHandles.emplace(std::move(name), std::move(handle));
    }
  }

  return std::make_shared<const IcebergTableHandle>(
      std::move(connectorId),
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
