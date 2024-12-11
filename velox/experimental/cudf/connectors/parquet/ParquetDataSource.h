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
#include "velox/expression/Expr.h"

namespace facebook::velox::cudf_velox::connector::parquet {

class ParquetDataSource : public facebook::velox::connector::DataSource {
 public:
  ParquetDataSource(
      const std::shared_ptr<const RowType>& outputType,
      const std::shared_ptr<connector::ConnectorTableHandle>& tableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<connector::ColumnHandle>>& columnHandles,
      velox::memory::MemoryPool* pool);

  void addSplit(std::shared_ptr<ConnectorSplit> split) override;

  void addDynamicFilter(
      column_index_t /*outputChannel*/,
      const std::shared_ptr<common::Filter>& /*filter*/) override {
    VELOX_NYI("Dynamic filters not supported by ParquetConnector.");
  }

  std::optional<RowVectorPtr> next(uint64_t size, velox::ContinueFuture& future)
      override;

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

  FileHandleFactory* const fileHandleFactory_;
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
  // RowVectorPtr projectOutputColumns(RowVectorPtr vector);

  // velox::Parquet::Table ParquetTable_;
  size_t ParquetTableRowCount_{0};
  std::shared_ptr<ParquetConnectorSplit> currentSplit_;

  size_t completedRows_{0};
  size_t completedBytes_{0};

  void setupRowIdColumn();

  // Evaluates remainingFilter_ on the specified vector. Returns number of rows
  // passed. Populates filterEvalCtx_.selectedIndices and selectedBits if only
  // some rows passed the filter. If none or all rows passed
  // filterEvalCtx_.selectedIndices and selectedBits are not updated.
  vector_size_t evaluateRemainingFilter(RowVectorPtr& rowVector);

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
  core::ExpressionEvaluator* const expressionEvaluator_;

  SubfieldFilters filters_;
  std::shared_ptr<common::MetadataFilter> metadataFilter_;
  std::unique_ptr<exec::ExprSet> remainingFilterExprSet_;

  dwio::common::RuntimeStatistics runtimeStats_;
  std::atomic<uint64_t> totalRemainingFilterTime_{0};

  // Field indices referenced in both remaining filter and output type. These
  // columns need to be materialized eagerly to avoid missing values in output.
  std::vector<column_index_t> multiReferencedFields_;

  std::shared_ptr<random::RandomSkipTracker> randomSkip_;
};

} // namespace facebook::velox::cudf_velox::connector::parquet
