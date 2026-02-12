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

#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/Options.h"
#include "velox/exec/tests/utils/TpcdsPlanFromJson.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::exec::test {

/// Builds TPC-DS query plans from pre-dumped JSON plan files and data files
/// located in a given data directory. Each table's data must be placed in
/// hive-style partitioning: a sub-directory per table name under the data path.
///
/// Example:
///   ls /gds/datasets/tpcds/sf100/
///     store_sales/  customer/  item/  ...
///
///   ls /gds/datasets/tpcds/sf100/store_sales/
///     store_sales_part-00000.parquet  store_sales_part-00001.parquet  ...
///
/// When VELOX_ENABLE_CUDF is defined, CuDF GPU acceleration can be enabled
/// via enableCudf(). This swaps the HiveConnector for CudfHiveConnector and
/// registers cuDF GPU operators.
class TpcdsQueryBuilder {
 public:
  explicit TpcdsQueryBuilder(
      dwio::common::FileFormat format = dwio::common::FileFormat::PARQUET);

  /// Scan dataPath for table subdirectories and collect all data file paths
  /// per table.
  void initialize(const std::string& dataPath);

#ifdef VELOX_ENABLE_CUDF
  /// Enable CuDF connector mode. When enabled:
  ///   - Unregisters the default HiveConnector, registers CudfHiveConnector
  ///     under the same connector ID.
  ///   - Calls cudf_velox::registerCudf() to enable GPU operator replacements.
  /// Must be called after initialize() and before getQueryPlan().
  /// @param ioExecutor Executor for IO operations (from test fixture).
  void enableCudf(folly::Executor* ioExecutor = nullptr);
#endif

  /// Load the deserialized plan from JSON (via TpcdsPlanFromJson) and populate
  /// dataFiles for each TableScan node by matching table names to the
  /// discovered data files in the data path.
  ///
  /// The connector ID is auto-detected from the plan's TableScan nodes (e.g.
  /// Presto plans use "hive"). If no connector is registered under that ID,
  /// a HiveConnector (or CudfHiveConnector if cudf is enabled) is registered
  /// automatically.
  ///
  /// Table name matching is flexible: if the exact name from the plan (e.g.
  /// "tpcds.store_sales") doesn't match a data directory, the part after the
  /// last '.' is tried (e.g. "store_sales").
  ///
  /// @param queryId TPC-DS query number (1..99)
  /// @param planDir Directory containing plan JSON files (q1.json, ...)
  /// @param pool Memory pool for plan deserialization
  TpcdsPlan getQueryPlan(
      int queryId,
      const std::string& planDir,
      memory::MemoryPool* pool);

  /// Create a ConnectorSplit for a given file path using HiveConnectorSplit.
  /// The connector ID used matches the one discovered from the plan (set
  /// during getQueryPlan).
  std::shared_ptr<connector::ConnectorSplit> makeSplit(
      const std::string& filePath) const;

  /// Clean up: if cudf was enabled, calls cudf_velox::unregisterCudf().
  /// Also unregisters any connector that was auto-registered by getQueryPlan.
  void shutdown();

  /// Whether CuDF mode is currently enabled.
  bool isCudfEnabled() const {
    return cudfEnabled_;
  }

  /// Returns the connector ID discovered from the plan.
  const std::string& connectorId() const {
    return connectorId_;
  }

 private:
  bool cudfEnabled_{false};
  dwio::common::FileFormat format_;
  /// Connector ID auto-detected from the plan's TableScan nodes.
  std::string connectorId_;
  /// Whether we registered a connector ourselves (need to unregister on
  /// shutdown).
  bool ownedConnector_{false};

#ifdef VELOX_ENABLE_CUDF
  /// Stored for deferred CudfHiveConnector registration in getQueryPlan.
  folly::Executor* ioExecutor_{nullptr};
#endif

  /// tableName -> list of data file paths
  std::unordered_map<std::string, std::vector<std::string>> tableDataFiles_;

  /// Try to find data files for a table name. First tries exact match,
  /// then strips any schema prefix (e.g. "tpcds.store_sales" -> "store_sales").
  const std::vector<std::string>* findDataFiles(
      const std::string& tableName) const;
};

} // namespace facebook::velox::exec::test
