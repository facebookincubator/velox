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
#include "velox/common/file/FileSystems.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/SplitReader.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::connector::hive {

class HiveConfig;

class HiveDataSource : public DataSource {
 public:
  /// Runtime stat keys for Hive data source.
  static constexpr std::string_view kNumPrefetch{"numPrefetch"};
  static constexpr std::string_view kPrefetchBytes{"prefetchBytes"};
  static constexpr std::string_view kTotalScanTime{"totalScanTime"};
  static constexpr std::string_view kOverreadBytes{"overreadBytes"};
  static constexpr std::string_view kStorageReadBytes{"storageReadBytes"};
  static constexpr std::string_view kNumLocalRead{"numLocalRead"};
  static constexpr std::string_view kLocalReadBytes{"localReadBytes"};
  static constexpr std::string_view kNumRamRead{"numRamRead"};
  static constexpr std::string_view kRamReadBytes{"ramReadBytes"};
  static constexpr std::string_view kNumBucketConversion{"numBucketConversion"};

  HiveDataSource(
      const RowTypePtr& outputType,
      const connector::ConnectorTableHandlePtr& tableHandle,
      const connector::ColumnHandleMap& assignments,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* ioExecutor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<HiveConfig>& hiveConfig);

  void addSplit(std::shared_ptr<ConnectorSplit> split) override;

  std::optional<RowVectorPtr> next(uint64_t size, velox::ContinueFuture& future)
      override;

  void addDynamicFilter(
      column_index_t outputChannel,
      const std::shared_ptr<common::Filter>& filter) override;

  uint64_t getCompletedBytes() override {
    return ioStatistics_->rawBytesRead();
  }

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override;

  bool allPrefetchIssued() const override {
    return splitReader_ && splitReader_->allPrefetchIssued();
  }

  void setFromDataSource(std::unique_ptr<DataSource> sourceUnique) override;

  int64_t estimatedRowSize() override;

  const common::SubfieldFilters* getFilters() const override {
    return &filters_;
  }

  std::shared_ptr<wave::WaveDataSource> toWaveDataSource() override;

  using WaveDelegateHookFunction =
      std::function<std::shared_ptr<wave::WaveDataSource>(
          const HiveTableHandlePtr& hiveTableHandle,
          const std::shared_ptr<common::ScanSpec>& scanSpec,
          const RowTypePtr& readerOutputType,
          std::unordered_map<std::string, HiveColumnHandlePtr>* partitionKeys,
          FileHandleFactory* fileHandleFactory,
          folly::Executor* executor,
          const ConnectorQueryCtx* connectorQueryCtx,
          const std::shared_ptr<HiveConfig>& hiveConfig,
          const std::shared_ptr<io::IoStatistics>& ioStatistics,
          const exec::ExprSet* remainingFilter,
          std::shared_ptr<common::MetadataFilter> metadataFilter)>;

  static WaveDelegateHookFunction waveDelegateHook_;

  static void registerWaveDelegateHook(WaveDelegateHookFunction hook);

  const ConnectorQueryCtx* testingConnectorQueryCtx() const {
    return connectorQueryCtx_;
  }

 protected:
  virtual std::unique_ptr<SplitReader> createSplitReader();

  FileHandleFactory* const fileHandleFactory_;
  folly::Executor* const ioExecutor_;
  const ConnectorQueryCtx* const connectorQueryCtx_;
  const std::shared_ptr<HiveConfig> hiveConfig_;
  memory::MemoryPool* const pool_;

  std::shared_ptr<HiveConnectorSplit> split_;
  HiveTableHandlePtr hiveTableHandle_;
  std::shared_ptr<common::ScanSpec> scanSpec_;
  VectorPtr output_;
  std::unique_ptr<SplitReader> splitReader_;

  // Output type from file reader.  This is different from outputType_ that it
  // contains column names before assignment, and columns that only used in
  // remaining filter.
  RowTypePtr readerOutputType_;

  // Column handles for the partition key columns keyed on partition key column
  // name.
  std::unordered_map<std::string, HiveColumnHandlePtr> partitionKeys_;

  std::shared_ptr<io::IoStatistics> ioStatistics_;
  std::shared_ptr<IoStats> ioStats_;

 private:
  std::vector<column_index_t> setupBucketConversion();

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

  // Add the information from column handle to the corresponding fields in this
  // object.
  void processColumnHandle(const HiveColumnHandlePtr& handle);

  // The row type for the data source output, not including filter-only columns
  const RowTypePtr outputType_;
  core::ExpressionEvaluator* const expressionEvaluator_;

  // Column handles for the Split info columns keyed on their column names.
  std::unordered_map<std::string, HiveColumnHandlePtr> infoColumns_;
  SpecialColumnNames specialColumns_{};
  std::vector<common::Subfield> remainingFilterSubfields_;
  folly::F14FastMap<std::string, std::vector<const common::Subfield*>>
      subfields_;
  // Optional post-processors for each output column, collected from
  // HiveColumnHandle::postProcessor(). Applied after reading and filtering to
  // transform column values. Indexed by output column position.
  std::vector<std::function<void(VectorPtr&)>> columnPostProcessors_;
  common::SubfieldFilters filters_;
  std::shared_ptr<common::MetadataFilter> metadataFilter_;
  std::unique_ptr<exec::ExprSet> remainingFilterExprSet_;
  RowVectorPtr emptyOutput_;
  dwio::common::RuntimeStatistics runtimeStats_;
  std::atomic<uint64_t> totalRemainingFilterTime_{0};
  uint64_t completedRows_ = 0;

  // Field indices referenced in both remaining filter and output type. These
  // columns need to be materialized eagerly to avoid missing values in output.
  std::vector<column_index_t> multiReferencedFields_;

  std::shared_ptr<random::RandomSkipTracker> randomSkip_;

  int64_t numBucketConversion_ = 0;

  // Reusable memory for remaining filter evaluation.
  VectorPtr filterResult_;
  SelectivityVector filterRows_;
  DecodedVector filterLazyDecoded_;
  SelectivityVector filterLazyBaseRows_;
  exec::FilterEvalCtx filterEvalCtx_;

  // Remembers the WaveDataSource. Successive calls to toWaveDataSource() will
  // return the same.
  std::shared_ptr<wave::WaveDataSource> waveDataSource_;
};
} // namespace facebook::velox::connector::hive
