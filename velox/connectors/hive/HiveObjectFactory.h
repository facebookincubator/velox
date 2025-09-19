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
#include "velox/dwio/common/Options.h"

namespace facebook::velox::connector::hive {

class HiveObjectFactory : public connector::ConnectorObjectFactory {
 public:
  HiveObjectFactory(std::shared_ptr<memory::MemoryPool> pool) : pool_(pool) {}

  ~HiveObjectFactory() override = default;

  std::shared_ptr<connector::ConnectorSplit> makeSplit(
      const std::string& connectorId,
      const folly::dynamic& options = {}) const override;

  std::shared_ptr<connector::ColumnHandle> makeColumnHandle(
      const std::string& connectorId,
      const std::string& name,
      const TypePtr& type,
      const folly::dynamic& options = {}) const override;

  std::shared_ptr<ConnectorTableHandle> makeTableHandle(
      const std::string& connectorId,
      const std::string& tableName,
      std::vector<ColumnHandlePtr> columnHandles,
      const folly::dynamic& options) const override;

  std::shared_ptr<ConnectorInsertTableHandle> makeInsertTableHandle(
      const std::string& connectorId,
      std::vector<ColumnHandlePtr> inputColumns,
      std::shared_ptr<const ConnectorLocationHandle> locationHandle,
      const folly::dynamic& options = {}) const override;

  std::shared_ptr<connector::ConnectorLocationHandle> makeLocationHandle(
      const std::string& connectorId,
      connector::ConnectorLocationHandle::TableType tableType =
          connector::ConnectorLocationHandle::TableType::kNew,
      const folly::dynamic& options = {}) const override;

 private:
  std::shared_ptr<memory::MemoryPool> pool_;
  dwio::common::FileFormat defaultFileFormat_{
      dwio::common::FileFormat::PARQUET};
};

} // namespace facebook::velox::connector::hive
