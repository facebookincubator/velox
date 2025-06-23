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
#pragma once

#include <shared_mutex>
#include <string>

#include "velox/connectors/Connector.h"

namespace facebook::velox::connector {

class ConnectorObjectFactory {
 public:
  virtual ~ConnectorObjectFactory() = default;

  const std::string& connectorName() const {
    return name_;
  }

  virtual std::shared_ptr<ConnectorSplit> makeConnectorSplit(
      const std::string& connectorId,
      const std::string& filePath,
      uint64_t start,
      uint64_t length,
      const folly::dynamic& options = {}) const {
    VELOX_UNSUPPORTED("ConnectorSplit not supported by connector", connectorId);
  }

  virtual std::shared_ptr<connector::ColumnHandle> makeColumnHandle(
      const std::string& connectorId,
      const std::string& name,
      const TypePtr& type,
      const folly::dynamic& options = {}) const {
    VELOX_UNSUPPORTED(
        "connector::ColumnHandle not supported by connector", connectorId);
  }

  virtual std::shared_ptr<ConnectorTableHandle> makeTableHandle(
      const std::string& connectorId,
      const std::string& tableName,
      std::vector<std::shared_ptr<const connector::ColumnHandle>> columnHandles,
      const folly::dynamic& options) const {
    VELOX_UNSUPPORTED(
        "ConnectorTableHandle not supported by connector", connectorId);
  }

  virtual std::shared_ptr<ConnectorInsertTableHandle> makeInsertTableHandle(
      const std::string& connectorId,
      std::vector<std::shared_ptr<const connector::ColumnHandle>> inputColumns,
      std::shared_ptr<const ConnectorLocationHandle> locationHandle,
      const folly::dynamic& options = {}) const {
    VELOX_UNSUPPORTED(
        "ConnectorInsertTableHandle not supported by connector", connectorId);
  }

  virtual std::shared_ptr<ConnectorLocationHandle> makeLocationHandle(
      const std::string& connectorId,
      ConnectorLocationHandle::TableType tableType =
          ConnectorLocationHandle::TableType::kNew,
      const folly::dynamic& options = {}) const {
    VELOX_UNSUPPORTED(
        "ConnectorLocationHandle not supported by connector", connectorId);
  }

 private:
  const std::string name_;
};

} // namespace facebook::velox::connector
