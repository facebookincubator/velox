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

#include "velox/common/base/RandomUtil.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/connectors/Connector.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetConnector.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetTableHandle.h"
#include "velox/expression/Expr.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/logger.hpp>
#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox::connector::parquet {

class ParquetDataSource : public facebook::velox::connector::DataSource {
 public:
  ParquetDataSource(
      const std::shared_ptr<const RowType>& outputType,
      const std::shared_ptr<connector::ConnectorTableHandle>& tableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<connector::ColumnHandle>>& columnHandles,
      velox::memory::MemoryPool* pool,
      const std::shared_ptr<parquetConfig>& parquetConfig);

  void addSplit(std::shared_ptr<ConnectorSplit> split) override;

  void addDynamicFilter(
      column_index_t /*outputChannel*/,
      const std::shared_ptr<common::Filter>& /*filter*/) override {
    VELOX_NYI("Dynamic filters not yet implemented by cudf::ParquetConnector.");
    // parquetConfig_->options().set_filter(filter);
  }

  std::optional<RowVectorPtr> next(uint64_t size, velox::ContinueFuture& future)
      override;
  {
    if (splitReader_->has_next()) {
      auto [tbl, meta] = splitReader_->read_chunk();
      return std::make_optional(to_velox_column(tbl->view(), pool_));
    } else {
      return std::nullopt;
    }
  }

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  uint64_t getCompletedBytes() override {
    return completedBytes_;
  }

  std::unordered_map<std::string, RuntimeCounter> runtimeStats() override {
    // TODO: Which stats do we want to expose here?
    return {};
  }

 protected:
  virtual std::unique_ptr<cudf::io::chunked_parquet_reader> createSplitReader();

  folly::Executor* const executor_;
  const ConnectorQueryCtx* const connectorQueryCtx_;
  const std::shared_ptr<ParquetConfig> parquetConfig_;
  memory::MemoryPool* const pool_;

  std::shared_ptr<ParquetConnectorSplit> split_;
  std::shared_ptr<ParquetTableHandle> parquetTableHandle_;
  std::shared_ptr<common::ScanSpec> scanSpec_;
  VectorPtr output_;
  std::unique_ptr<cudf::io::chunked_parquet_reader> splitReader_;

  // Output type from file reader.  This is different from outputType_ that it
  // contains column names before assignment, and columns that only used in
  // remaining filter.
  RowTypePtr readerOutputType_;

  std::shared_ptr<io::IoStatistics> ioStats_;

 private:
  size_t ParquetTableRowCount_{0};
  std::shared_ptr<ParquetConnectorSplit> currentSplit_;

  size_t completedRows_{0};
  size_t completedBytes_{0};

  void setupRowIdColumn();

  // Clear split_ after split has been fully processed.  Keep readers around to
  // hold adaptation.
  void resetSplit();

  const RowVectorPtr& getEmptyOutput() {
    if (!emptyOutput_) {
      emptyOutput_ = RowVector::createEmpty(outputType_, pool_);
    }
    return emptyOutput_;
  }
  RowVectorPtr emptyOutput_;

  // The row type for the data source output, not including filter-only columns
  const RowTypePtr outputType_;

  dwio::common::RuntimeStatistics runtimeStats_;

  std::shared_ptr<random::RandomSkipTracker> randomSkip_;
};

} // namespace facebook::velox::cudf_velox::connector::parquet
