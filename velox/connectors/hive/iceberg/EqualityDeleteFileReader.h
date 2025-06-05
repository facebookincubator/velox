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
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      const std::shared_ptr<io::IoStatistics> ioStats,
      const std::shared_ptr<filesystems::File::IoStats>& fsStats,
      const std::string& connectorId);

  /// Reads the delete values from the equality delete file, and interprets them
  /// as filters for the base file reader.
  ///
  /// @subfieldFilters The built SubfieldFilter that can be pushed down to the
  /// base file RowReader, when the equality delete file only contains one
  /// single subfield of Iceberg RowId type.
  /// @typedExpressions The built TypedExpr that will be evaluated by the
  /// connector DataSource after rows are read from the base file RowReader.
  void readDeleteValues(
      SubfieldFilters& subfieldFilters,
      std::vector<core::TypedExprPtr>& typedExpressions);

 private:
  void buildDomainFilter(SubfieldFilters& subfieldFilters);

  void buildFilterFunctions(std::vector<core::TypedExprPtr>& expressionInputs);

  // The equality delete file to read
  const IcebergDeleteFile& deleteFile_;
  // The schema of the base file in terms of TypeWithId tree. In addition to the
  // existing fields that were included in the base file ScanSpec, it also
  // contains the extra fields that are in the equality delete file but not
  // in the ScanSpec of the base file
  const std::shared_ptr<const dwio::common::TypeWithId> baseFileSchema_;

  // The cache factory of the file handles, which can be used to return the file
  // handle of the delete file.
  FileHandleFactory* const fileHandleFactory_;
  memory::MemoryPool* const pool_;

  // The split of the equality delete file to be processed by the delete file
  // RowReader.
  std::shared_ptr<const HiveConnectorSplit> deleteSplit_;
  // The RowType of the equality delete file
  RowTypePtr deleteFileRowType_;
  // The RowReader to read the equality delete file
  std::unique_ptr<dwio::common::RowReader> deleteRowReader_;
  // The output vector to hold the delete values
  VectorPtr deleteValuesOutput_;
};

} // namespace facebook::velox::connector::hive::iceberg
