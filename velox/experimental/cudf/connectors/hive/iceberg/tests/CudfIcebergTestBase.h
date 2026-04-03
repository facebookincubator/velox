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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergConnector.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"

namespace facebook::velox::cudf_velox::exec::test {

static const std::string kCudfIcebergConnectorId = "test-cudf-iceberg";

/// Test base for CudfIcebergConnector tests. Registers the
/// CudfIcebergConnectorFactory and provides helpers for creating Iceberg
/// splits, writing parquet data files (via CudfHiveConnectorTestBase), and
/// writing DWRF delete files (via the upstream velox::dwrf::Writer).
class CudfIcebergTestBase : public CudfHiveConnectorTestBase {
 public:
  void SetUp() override;
  void TearDown() override;

  /// Creates HiveIcebergSplits pointing to a data file with optional delete
  /// files and sequence number. Uses the cudf iceberg connector ID.
  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
  makeIcebergSplits(
      const std::string& dataFilePath,
      const std::vector<
          facebook::velox::connector::hive::iceberg::IcebergDeleteFile>&
          deleteFiles = {},
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys = {},
      uint32_t splitCount = 1,
      int64_t dataSequenceNumber = 0);

  /// Writes a DWRF file using the upstream velox::dwrf::Writer. Used for
  /// equality and positional delete files which are read by the upstream
  /// Velox DWRF reader (not cudf).
  void writeDeleteFile(
      const std::string& filePath,
      const std::vector<RowVectorPtr>& vectors);

  uint64_t getFileSize(const std::string& path);

  /// Builds a table scan plan using the cudf iceberg connector.
  facebook::velox::core::PlanNodePtr makeTableScanPlan(
      const RowTypePtr& rowType);
};

} // namespace facebook::velox::cudf_velox::exec::test
