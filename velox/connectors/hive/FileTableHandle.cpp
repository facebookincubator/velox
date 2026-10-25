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

#include "velox/connectors/hive/FileTableHandle.h"

namespace facebook::velox::connector::hive {

FileTableHandle::FileTableHandle(
    std::string connectorId,
    std::string tableName,
    common::SubfieldFilters subfieldFilters,
    core::TypedExprPtr remainingFilter,
    RowTypePtr dataColumns,
    std::unordered_map<std::string, std::string> tableParameters,
    double sampleRate,
    std::string dbName)
    : ConnectorTableHandle(std::move(connectorId)),
      tableName_(std::move(tableName)),
      subfieldFilters_(std::move(subfieldFilters)),
      remainingFilter_(std::move(remainingFilter)),
      dataColumns_(std::move(dataColumns)),
      tableParameters_(std::move(tableParameters)),
      sampleRate_(sampleRate),
      dbName_(std::move(dbName)) {}

folly::dynamic FileTableHandle::serializeFileFields(
    std::string_view typeName) const {
  folly::dynamic obj = ConnectorTableHandle::serializeBase(typeName);
  obj["tableName"] = tableName_;
  if (sampleRate_ < 1.0) {
    obj["sampleRate"] = sampleRate_;
  }
  if (!dbName_.empty()) {
    obj["dbName"] = dbName_;
  }
  return obj;
}

void FileTableHandle::deserializeFileFields(
    const folly::dynamic& obj,
    std::string& connectorId,
    std::string& tableName,
    double& sampleRate,
    std::string& dbName) {
  connectorId = obj["connectorId"].asString();
  tableName = obj["tableName"].asString();
  sampleRate = 1.0;
  if (obj.count("sampleRate")) {
    sampleRate = obj["sampleRate"].asDouble();
  }
  if (auto it = obj.find("dbName"); it != obj.items().end()) {
    dbName = it->second.asString();
  }
}

} // namespace facebook::velox::connector::hive
