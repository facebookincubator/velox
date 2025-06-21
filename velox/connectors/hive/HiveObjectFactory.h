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

#include "velox/connectors/ConnectorObjectFactory.h"

namespace facebook::velox::connector::hive {

class HiveObjectFactory : public connector::ConnectorObjectFactory {
 public:
  ~HiveObjectFactory() override = default;

  std::shared_ptr<connector::ConnectorSplit> makeConnectorSplit(
      const std::string& connectorId,
      const std::string& filePath,
      uint64_t start,
      uint64_t length,
      const folly::dynamic& options = {}) const override;

  std::unique_ptr<connector::ConnectorColumnHandle> makeColumnHandle(
      const std::string& connectorId,
      const std::string& name,
      const TypePtr& type,
      const folly::dynamic& options = {}) const override;

  std::shared_ptr<ConnectorTableHandle> HiveObjectFactory::makeTableHandle(
      const std::string& connectorId,
      const std::string& tableName,
      std::vector<std::shared_ptr<const ConnectorColumnHandle>> columnHandles,
      const folly::dynamic& options) const override;

  std::shared_ptr<ConnectorInsertTableHandle> makeInsertTableHandle(
      const std::string& connectorId,
      std::vector<std::shared_ptr<const ConnectorColumnHandle>> inputColumns;
      std::shared_ptr<const ConnectorLocationHandle> locationHandle;
      const folly::dynamic& options = {}) const override;

  //  std::shared_ptr<connector::ConnectorInsertTableHandle>
  //  makeInsertTableHandle(
  //      const std::string& connectorId,
  //      const std::vector<std::string>& tableColumnNames,
  //      const std::vector<TypePtr>& tableColumnTypes,
  //      std::shared_ptr<ConnectorLocationHandle> locationHandle,
  //      const std::optional<velox::common::CompressionKind> compressionKind,
  //      const folly::dynamic& options = {}) const override;

  std::shared_ptr<connector::ConnectorLocationHandle> makeLocationHandle(
      const std::string& connectorId,
      connector::ConnectorLocationHandle::TableType tableType =
          connector::ConnectorLocationHandle::TableType::kNew,
      const folly::dynamic& options = {}) const override;
};

} // namespace facebook::velox::connector::hive
