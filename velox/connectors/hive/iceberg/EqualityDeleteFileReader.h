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

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/Reader.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::connector::hive::iceberg {

class IcebergDeleteFile;

using SubfieldFilters =
    std::unordered_map<common::Subfield, std::unique_ptr<common::Filter>>;

class EqualityDeleteFileReader {
 public:
  EqualityDeleteFileReader(
      const IcebergDeleteFile& deleteFile,
      std::shared_ptr<const dwio::common::TypeWithId> baseFileSchema,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const HiveConfig> hiveConfig,
      std::shared_ptr<io::IoStatistics> ioStats,
      dwio::common::RuntimeStatistics& runtimeStats,
      const std::string& connectorId);

  void readDeleteValues(
      SubfieldFilters& subfieldFilters,
      std::vector<core::TypedExprPtr>& typedExpressions);

 private:
  void readSingleColumnDeleteValues(SubfieldFilters& subfieldFilters);

  void readMultipleColumnDeleteValues(
      std::vector<core::TypedExprPtr>& expressionInputs);

  const IcebergDeleteFile& deleteFile_;
  const std::shared_ptr<const dwio::common::TypeWithId> baseFileSchema_;
  const ConnectorQueryCtx* const connectorQueryCtx_;
  const std::shared_ptr<const HiveConfig> hiveConfig_;
  std::shared_ptr<const HiveConnectorSplit> deleteSplit_;

  FileHandleFactory* const fileHandleFactory_;
  folly::Executor* const executor_;
  memory::MemoryPool* const pool_;
  const std::shared_ptr<io::IoStatistics> ioStats_;

  RowTypePtr rowType_;
  std::unique_ptr<dwio::common::RowReader> deleteRowReader_;
  VectorPtr deleteValuesOutput_;
};

} // namespace facebook::velox::connector::hive::iceberg