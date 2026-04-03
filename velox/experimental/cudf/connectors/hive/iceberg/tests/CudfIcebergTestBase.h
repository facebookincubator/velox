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
/// splits and writing parquet test data files.
class CudfIcebergTestBase : public CudfHiveConnectorTestBase {
 public:
  void SetUp() override;
  void TearDown() override;

  /// Creates a HiveIcebergSplit pointing to a data file with optional delete
  /// files. Uses the cudf iceberg connector ID.
  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
  makeIcebergSplits(
      const std::string& dataFilePath,
      const std::vector<
          facebook::velox::connector::hive::iceberg::IcebergDeleteFile>&
          deleteFiles = {},
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys = {},
      uint32_t splitCount = 1);
};

} // namespace facebook::velox::cudf_velox::exec::test
