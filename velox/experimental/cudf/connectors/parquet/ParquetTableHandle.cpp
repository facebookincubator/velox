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
#include <string>
#include <vector>

#include "velox/connectors/Connector.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetTableHandle.h"
#include "velox/type/Type.h"

#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox::connector::parquet {

using namespace facebook::velox::connector;

ParquetColumnHandle::ParquetColumnHandle(
    const std::string& name,
    const TypePtr& type,
    const cudf::data_type data_type,
    const std::vector<ParquetColumnHandle>& children)
    : name_(name), type_(type), data_type_(data_type), children_(children) {}

ParquetTableHandle::ParquetTableHandle(
    std::string connectorId,
    const std::string& tableName,
    bool filterPushdownEnabled,
    const RowTypePtr& dataColumns)
    : ConnectorTableHandle(std::move(connectorId)),
      tableName_(tableName),
      filterPushdownEnabled_(filterPushdownEnabled),
      dataColumns_(dataColumns) {}

std::string ParquetTableHandle::toString() const {
  std::stringstream out;
  out << "table: " << tableName_;
  if (dataColumns_) {
    out << ", data columns: " << dataColumns_->toString();
  }
  return out.str();
}

ConnectorTableHandlePtr ParquetTableHandle::create(
    const folly::dynamic& obj,
    void* context) {
  VELOX_NYI("ParquetTableHandle::create() not yet implemented");
}

} // namespace facebook::velox::cudf_velox::connector::parquet
