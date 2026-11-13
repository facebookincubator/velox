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

#include "velox/connectors/common/FileDataSource.h"
#include "velox/connectors/hive/TableHandle.h"

namespace facebook::velox::connector::hive {

class HiveConfig;

class HiveDataSource : public connector::FileDataSource {
 public:
  static constexpr std::string_view kNumBucketConversion{"numBucketConversion"};
  static constexpr std::string_view kFileFormat{"fileFormat."};

  HiveDataSource(
      const RowTypePtr& outputType,
      const connector::ConnectorTableHandlePtr& tableHandle,
      const connector::ColumnHandleMap& assignments,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* ioExecutor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<HiveConfig>& hiveConfig);

  /// Also recomputes the allowed buckets when the filter is on a bucket
  /// column.
  void addDynamicFilter(
      column_index_t outputChannel,
      const std::shared_ptr<common::Filter>& filter) override;

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override;

  void setFromDataSource(std::unique_ptr<DataSource> sourceUnique) override;

  std::shared_ptr<wave::WaveDataSource> toWaveDataSource() override;

  using WaveDelegateHookFunction =
      std::function<std::shared_ptr<wave::WaveDataSource>(
          const HiveTableHandlePtr& hiveTableHandle,
          const std::shared_ptr<common::ScanSpec>& scanSpec,
          const RowTypePtr& readerOutputType,
          std::unordered_map<std::string, connector::FileColumnHandlePtr>*
              partitionKeys,
          FileHandleFactory* fileHandleFactory,
          folly::Executor* executor,
          const ConnectorQueryCtx* connectorQueryCtx,
          const std::shared_ptr<HiveConfig>& hiveConfig,
          const std::shared_ptr<io::IoStatistics>& ioStatistics,
          const exec::ExprSet* remainingFilter,
          std::shared_ptr<common::MetadataFilter> metadataFilter)>;

  static WaveDelegateHookFunction waveDelegateHook_;

  static void registerWaveDelegateHook(WaveDelegateHookFunction hook);

 protected:
  std::unique_ptr<connector::FileSplitReader> createSplitReader() override;

  // Skips a split whose table bucket the filters on the bucket columns
  // exclude.
  bool skipSplit() override;

  /// Pre-creation setup: stats tracking, bucket conversion, rowId.
  /// Returns bucket channels (empty if none).
  std::vector<column_index_t> prepareSplit();

 private:
  std::vector<column_index_t> setupBucketConversion();

  // Recomputes 'allowedBuckets_' from the scan spec's filters on the bucket
  // columns. The scan spec holds the static filters and the dynamic filters
  // added so far.
  void updateAllowedBuckets();

  void setupRowIdColumn();

  const std::shared_ptr<HiveConfig> hiveConfig_;

  int64_t numBucketConversion_ = 0;

  // HiveBucketPruning::allowedBuckets for the filters on the bucket columns.
  // std::nullopt when the table is not bucketed.
  std::optional<std::vector<bool>> allowedBuckets_;

  // Tracks the number of splits read per file format.
  std::unordered_map<dwio::common::FileFormat, int64_t> numSplitsByFileFormat_;

  std::shared_ptr<wave::WaveDataSource> waveDataSource_;
};
} // namespace facebook::velox::connector::hive
