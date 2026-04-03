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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"

#include "velox/common/io/IoStatistics.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/dwio/common/Statistics.h"

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_connector = ::facebook::velox::connector;
namespace velox_hive = ::facebook::velox::connector::hive;

class CudfIcebergSplitReader;

/// GPU-accelerated Iceberg data source that reads parquet data files via
/// libcudf and handles Iceberg delete semantics.
///
/// This is a thin orchestrator that delegates all per-split reading and delete
/// application to CudfIcebergSplitReader. For each split:
///   - addSplit() creates a CudfIcebergSplitReader and calls prepareSplit().
///   - next() delegates to the split reader's next().
class CudfIcebergDataSource : public velox_connector::DataSource,
                              public NvtxHelper {
 public:
  CudfIcebergDataSource(
      const RowTypePtr& outputType,
      const velox_connector::ConnectorTableHandlePtr& tableHandle,
      const velox_connector::ColumnHandleMap& columnHandles,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const velox_connector::ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
      const std::shared_ptr<const velox_hive::HiveConfig>& hiveConfig);

  ~CudfIcebergDataSource() override;

  void addSplit(
      std::shared_ptr<velox_connector::ConnectorSplit> split) override;

  void addDynamicFilter(
      column_index_t /*outputChannel*/,
      const std::shared_ptr<common::Filter>& /*filter*/) override {
    VELOX_NYI(
        "Dynamic filters not yet implemented by CudfIcebergConnector.");
  }

  std::optional<RowVectorPtr> next(
      uint64_t size,
      velox::ContinueFuture& future) override;

  uint64_t getCompletedRows() override;

  uint64_t getCompletedBytes() override {
    return completedBytes_;
  }

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override;

 private:
  const RowTypePtr outputType_;

  std::shared_ptr<const velox_hive::HiveTableHandle> tableHandle_;

  const std::shared_ptr<CudfHiveConfig> cudfHiveConfig_;
  const std::shared_ptr<const velox_hive::HiveConfig> hiveConfig_;
  FileHandleFactory* const fileHandleFactory_;
  folly::Executor* const executor_;
  const velox_connector::ConnectorQueryCtx* const connectorQueryCtx_;

  memory::MemoryPool* const pool_;

  std::vector<std::string> readColumnNames_;

  std::shared_ptr<io::IoStatistics> ioStatistics_;
  std::shared_ptr<IoStats> ioStats_;

  size_t completedBytes_{0};

  dwio::common::RuntimeStatistics runtimeStats_;

  std::unique_ptr<CudfIcebergSplitReader> splitReader_;
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
