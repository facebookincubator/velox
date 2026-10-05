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

#include <folly/container/F14Set.h>
#include <unordered_set>

#include "velox/common/base/RandomUtil.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/IoStatisticsRuntimeStats.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileConnectorSplit.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/FileScanReader.h"
#include "velox/connectors/hive/FileScanSpec.h"
#include "velox/connectors/hive/FileSplitReader.h"
#include "velox/connectors/hive/FileTableHandle.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::connector::hive {

/// File-specific scan batch event with split metadata. Non-owning fields are
/// valid only for the duration of the scan batch callback.
struct FileScanBatchEvent : public core::ScanBatchEvent {
  /// Table name from the connector table handle.
  std::string_view tableName;
  /// Database / namespace name from the connector table handle.
  std::string_view dbName;
  /// Physical file path for this batch, empty if there is no single source.
  std::string_view filePath;
  /// Non-owning pointer to the current physical file's partition keys.
  /// Null when partition keys are not available.
  const std::unordered_map<std::string, std::optional<std::string>>*
      partitionKeys{nullptr};
  /// File format for this batch, UNKNOWN if there is no single source.
  dwio::common::FileFormat fileFormat{dwio::common::FileFormat::UNKNOWN};
  /// Bytes fetched from storage producing this batch, including any read
  /// amplification from coalescing adjacent regions.
  uint64_t storageReadBytes{0};
};

class FileConfig;

/// Base class for file-based data sources that read from columnar file formats
/// (ORC, Parquet, etc.) through FileScanReader. Provides the common scan
/// pipeline: column resolution, filter extraction, scan spec construction,
/// split reading, remaining filter evaluation, and runtime stats collection.
///
/// Connector-specific data sources (Hive, Paimon, etc.) extend this class to
/// add format-specific behavior like bucket conversion, multi-file splits, or
/// merge-on-read.
class FileDataSource : public DataSource {
 public:
  /// Runtime stat keys for file-based data sources. Data IO stats use the
  /// keys directly (e.g., "storageReadBytes"). Metadata IO stats use the
  /// kMetadataPrefix (e.g., "metadata.storageReadBytes").
  static constexpr std::string_view kMetadataPrefix{"metadata"};
  static constexpr std::string_view kNumPrefetch{io::kNumPrefetch};
  static constexpr std::string_view kPrefetchBytes{io::kPrefetchBytes};
  static constexpr std::string_view kTotalScanTime{io::kTotalScanTime};
  static constexpr std::string_view kOverreadBytes{io::kOverreadBytes};
  static constexpr std::string_view kStorageReadBytes{io::kStorageReadBytes};
  static constexpr std::string_view kNumLocalRead{io::kNumLocalRead};
  static constexpr std::string_view kLocalReadBytes{io::kLocalReadBytes};
  static constexpr std::string_view kNumRamRead{io::kNumRamRead};
  static constexpr std::string_view kRamReadBytes{io::kRamReadBytes};
  static constexpr std::string_view kReadGapBytes{io::kReadGapBytes};

  FileDataSource(
      const RowTypePtr& outputType,
      const connector::ConnectorTableHandlePtr& tableHandle,
      const connector::ColumnHandleMap& assignments,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* ioExecutor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<FileConfig>& fileConfig,
      FileScanOptions options = {});

  FileDataSource(
      std::shared_ptr<const FileScanSpec> fileScanSpec,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* ioExecutor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<FileConfig>& fileConfig);

  ~FileDataSource() override;

  void addSplit(std::shared_ptr<ConnectorSplit> split) override;

  void cancel() override;

  std::optional<RowVectorPtr> next(uint64_t size, velox::ContinueFuture& future)
      override;

  void addDynamicFilter(
      column_index_t outputChannel,
      const std::shared_ptr<common::Filter>& filter) override;

  uint64_t getCompletedBytes() override {
    return dataIoStats_->rawBytesRead();
  }

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  void fireScanBatchCallback(core::ScanBatchEvent event) override;

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override;

  bool allPrefetchIssued() const override {
    return scanReader_ && scanReader_->allPrefetchIssued();
  }

  void setFromDataSource(std::unique_ptr<DataSource> sourceUnique) override;

  int64_t estimatedRowSize() override;

  const common::SubfieldFilters* getFilters() const override {
    return &filters_;
  }

  const ConnectorQueryCtx* testingConnectorQueryCtx() const {
    return connectorQueryCtx_;
  }

 protected:
  /// Creates and prepares a reader for activeSplit_. The default adapter uses
  /// the existing single-file createSplitReader() specialization.
  virtual std::unique_ptr<FileScanReader> createScanReader();

  virtual std::unique_ptr<FileSplitReader> createSplitReader();

  /// Rebuild after changing physical column demands. The MetadataFilter and
  /// extraction transforms are always bound to the resulting ScanSpec.
  void resetScanSpec();

  FileHandleFactory* const fileHandleFactory_;
  folly::Executor* const ioExecutor_;
  const ConnectorQueryCtx* const connectorQueryCtx_;
  const std::shared_ptr<FileConfig> fileConfig_;
  memory::MemoryPool* const pool_;

  const std::shared_ptr<const FileScanSpec> fileScanSpec_;
  std::shared_ptr<FileScanState> fileScanState_;

  std::shared_ptr<ConnectorSplit> activeSplit_;
  // Single-file context for existing createSplitReader() specializations.
  // Logical readers do not need to populate this member.
  std::shared_ptr<FileConnectorSplit> split_;
  FileTableHandlePtr tableHandle_;
  std::shared_ptr<common::ScanSpec> scanSpec_;
  VectorPtr output_;
  std::unique_ptr<FileScanReader> scanReader_;

  /// Output type from file reader. This is different from outputType_ in that
  /// it contains column names before assignment, and columns that are only used
  /// in the remaining filter.
  RowTypePtr readerOutputType_;

  /// Column handles for the partition key columns keyed on partition key column
  /// name.
  std::unordered_map<std::string, FileColumnHandlePtr> partitionKeys_;

  std::shared_ptr<io::IoStatistics> dataIoStats_;
  std::shared_ptr<io::IoStatistics> metadataIoStats_;
  std::shared_ptr<IoStats> ioStats_;

  // Cumulative dataIoStats_->read().sum() as of the last scan batch event, so
  // each event reports only the bytes read since the previous one.
  uint64_t lastEventStorageReadBytes_{0};

  /// Column handles for the split info columns keyed on their column names.
  std::unordered_map<std::string, FileColumnHandlePtr> infoColumns_;
  SpecialColumnNames specialColumns_{};

  /// Subfield pruning info collected from column handles and remaining filter.
  folly::F14FastMap<std::string, std::vector<const common::Subfield*>>
      subfields_;
  common::SubfieldFilters filters_;

  const exec::ExprSet* remainingFilterExprSet() const {
    return remainingFilterExprSet_.get();
  }

  const std::shared_ptr<common::MetadataFilter>& metadataFilter() const {
    return metadataFilter_;
  }

  /// The row type for the data source output, not including filter-only
  /// columns. Subclasses may call this from createSplitReader() to obtain
  /// the output schema when constructing a split reader.
  const RowTypePtr& outputType() const {
    return outputType_;
  }

  // Actual type produced by the reader after extraction pushdown.  Differs
  // from readerOutputType_ when the reader handles extraction natively
  // (e.g., MapKeys -> ARRAY).  Null if no extraction pushdown is active.
  RowTypePtr readerProducedType_;

  // Output columns that have extraction chains.  Maps output column index to
  // the column handle.  These columns are read with schemaType and transformed
  // post-read using the extraction chains.
  folly::F14FastMap<column_index_t, const FileColumnHandle*> extractionColumns_;

  // Completed logical readers only. Active statistics are added as a snapshot.
  std::unordered_map<std::string, RuntimeMetric> readerStats_;

 private:
  void applyDynamicFilters();

  /// Evaluates remainingFilter_ on the specified vector. Returns number of rows
  /// passed. Populates filterEvalCtx_.selectedIndices and selectedBits if only
  /// some rows passed the filter. If none or all rows passed
  /// filterEvalCtx_.selectedIndices and selectedBits are not updated.
  vector_size_t evaluateRemainingFilter(RowVectorPtr& rowVector);

  std::optional<RowVectorPtr> nextImpl(uint64_t size, ContinueFuture& future);

  /// Archives statistics once and releases the active reader and split.
  /// ScanSpec retains adaptation across splits.
  void resetSplit();

  const RowVectorPtr& getEmptyOutput() {
    if (!emptyOutput_) {
      emptyOutput_ = RowVector::createEmpty(outputType_, pool_);
    }
    return emptyOutput_;
  }

  /// The row type for the data source output, not including filter-only
  /// columns.
  const RowTypePtr outputType_;
  core::ExpressionEvaluator* const expressionEvaluator_;

  /// Optional post-processors for each output column, collected from
  /// HiveColumnHandle::postProcessor(). Applied after reading and filtering to
  /// transform column values. Indexed by output column position.
  std::vector<std::function<void(VectorPtr&)>> columnPostProcessors_;
  std::shared_ptr<common::MetadataFilter> metadataFilter_;
  std::unique_ptr<exec::ExprSet> remainingFilterExprSet_;
  RowVectorPtr emptyOutput_;
  std::atomic_uint64_t totalRemainingFilterTime_{0};
  std::atomic_uint64_t totalRemainingFilterCpuTime_{0};
  uint64_t completedRows_{0};
  /// Field indices referenced in both remaining filter and output type. These
  /// columns need to be materialized eagerly to avoid missing values in output.
  std::vector<column_index_t> multiReferencedFields_;

  // Column names referenced by the remaining filter expression.
  folly::F14FastSet<std::string> remainingFilterColumns_;

  std::shared_ptr<random::RandomSkipTracker> randomSkip_;

  // Persist logical dynamic conditions independently of file constants and
  // caches. In particular, partition filters must survive a split transition.
  std::unordered_map<column_index_t, common::FilterPtr> dynamicFilters_;

  /// Reusable memory for remaining filter evaluation.
  VectorPtr filterResult_;
  SelectivityVector filterRows_;
  DecodedVector filterLazyDecoded_;
  SelectivityVector filterLazyBaseRows_;
  exec::FilterEvalCtx filterEvalCtx_;
};

} // namespace facebook::velox::connector::hive
