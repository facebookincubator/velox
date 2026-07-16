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
#include <sstream>

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
    std::vector<int32_t> dataColumnFieldIds,
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
          std::move(dbName),
          std::move(dataColumnFieldIds)),
      isChangelogQuery_(isChangelogQuery),
      dataColumnHandles_(std::move(dataColumnHandles)) {
  if (isChangelogQuery_) {
    VELOX_USER_CHECK(
        !dataColumnHandles_.empty(),
        "dataColumnHandles must not be empty when isChangelogQuery is true");
  }
}

std::string IcebergTableHandle::toString() const {
  std::ostringstream out;
  out << HiveTableHandle::toString();
  if (isChangelogQuery_) {
    out << ", isChangelogQuery: true";
  }
  if (!dataColumnHandles_.empty()) {
    // Sort by name for deterministic output, mirroring HiveTableHandle's
    // treatment of subfieldFilters and tableParameters.
    std::map<std::string, const IcebergColumnHandle*> ordered;
    for (const auto& [name, handle] : dataColumnHandles_) {
      ordered[name] = handle.get();
    }
    out << ", dataColumnHandles: [";
    bool first = true;
    for (const auto& [name, handle] : ordered) {
      if (!first) {
        out << ", ";
      }
      out << name << ": " << handle->toString();
      first = false;
    }
    out << "]";
  }
  return out.str();
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
  std::string connectorId;
  std::string tableName;
  std::string dbName;
  common::SubfieldFilters subfieldFilters;
  core::TypedExprPtr remainingFilter;
  double sampleRate{1.0};
  RowTypePtr dataColumns;
  std::unordered_map<std::string, std::string> tableParameters;
  // deserializeHiveFields dispatches through the SerDe registry using the
  // "name" key in each handle's JSON.  Because IcebergColumnHandle::serialize()
  // writes "name":"IcebergColumnHandle", the registry instantiates
  // IcebergColumnHandle objects, so the dynamic_cast below is always valid.
  std::vector<HiveColumnHandlePtr> hiveFilterHandles;
  std::vector<std::string> indexColumns;
  std::vector<int32_t> dataColumnFieldIds;

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
      hiveFilterHandles,
      indexColumns,
      dbName,
      dataColumnFieldIds);

  // Cast the already-deserialized handles to their concrete IcebergColumnHandle
  // type.  The dynamic_cast is guaranteed to succeed (see comment above).
  std::vector<IcebergColumnHandlePtr> filterColumnHandles;
  filterColumnHandles.reserve(hiveFilterHandles.size());
  for (const auto& h : hiveFilterHandles) {
    auto handle = std::dynamic_pointer_cast<const IcebergColumnHandle>(h);
    VELOX_CHECK_NOT_NULL(
        handle,
        "filterColumnHandle is not an IcebergColumnHandle during deserialization");
    filterColumnHandles.push_back(std::move(handle));
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
      VELOX_CHECK_NOT_NULL(
          handle,
          "dataColumnHandle is not an IcebergColumnHandle during deserialization");
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
      std::move(dataColumnFieldIds),
      isChangelogQuery,
      std::move(dataColumnHandles));
}

// static
void IcebergTableHandle::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("IcebergTableHandle", IcebergTableHandle::create);
}

} // namespace facebook::velox::connector::hive::iceberg
