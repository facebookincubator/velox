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
#include "velox/exec/tests/utils/VeloxPlanLoader.h"

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
/// This class is CuDF-free. To use CudfHiveConnector instead of HiveConnector,
/// subclass and override registerConnector().
class TpcdsQueryBuilder {
 public:
  explicit TpcdsQueryBuilder(
      dwio::common::FileFormat format = dwio::common::FileFormat::PARQUET);

  virtual ~TpcdsQueryBuilder() = default;

  /// Scan dataPath for table subdirectories and collect all data file paths
  /// per table.
  void initialize(const std::string& dataPath);

  /// Load the deserialized plan from JSON (via VeloxPlanLoader) and populate
  /// dataFiles for each TableScan node by matching table names to the
  /// discovered data files in the data path.
  ///
  /// The connector ID is auto-detected from the plan's TableScan nodes (e.g.
  /// Presto plans use "hive"). If no connector is registered under that ID,
  /// registerConnector() is called to register one.
  ///
  /// Table name matching is flexible: if the exact name from the plan (e.g.
  /// "tpcds.store_sales") doesn't match a data directory, the part after the
  /// last '.' is tried (e.g. "store_sales").
  ///
  /// @param queryId TPC-DS query number (1..99)
  /// @param planDir Directory containing plan JSON files (Q1.json, ...)
  /// @param pool Memory pool for plan deserialization
  VeloxPlan getQueryPlan(
      int queryId,
      const std::string& planDir,
      memory::MemoryPool* pool);

  /// Create a ConnectorSplit for a given file path using HiveConnectorSplit.
  /// The connector ID used matches the one discovered from the plan.
  std::shared_ptr<connector::ConnectorSplit> makeSplit(
      const std::string& filePath) const;

  /// Clean up: unregisters any connector that was auto-registered by
  /// getQueryPlan.
  virtual void shutdown();

  /// Returns the connector ID discovered from the plan.
  const std::string& connectorId() const {
    return connectorId_;
  }

 protected:
  /// Called when a connector needs to be registered for the given ID.
  /// Default implementation registers a HiveConnector.
  /// Override in subclasses to register CudfHiveConnector or others.
  /// @param connectorId The connector ID from the plan.
  /// @param ioExecutor Optional IO executor.
  virtual void registerHiveConnector(
      const std::string& connectorId,
      folly::Executor* ioExecutor = nullptr);

  /// Try to find data files for a table name. First tries exact match,
  /// then strips any schema prefix (e.g. "tpcds.store_sales" ->
  /// "store_sales").
  const std::vector<std::string>* findDataFiles(
      const std::string& tableName) const;

  dwio::common::FileFormat format_;
  /// Connector ID auto-detected from the plan's TableScan nodes.
  std::string connectorId_;
  /// Whether we registered a connector ourselves (need to unregister on
  /// shutdown).
  bool ownedConnector_{false};

 private:
  /// tableName -> list of data file paths
  std::unordered_map<std::string, std::vector<std::string>> tableDataFiles_;
};

} // namespace facebook::velox::exec::test
