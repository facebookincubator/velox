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

#include "velox/connectors/hive/FileSplitReader.h"
#include "velox/connectors/hive/HivePartitionFunction.h"

namespace facebook::velox::connector::hive {

struct HiveConnectorSplit;

/// Hive-specific FileSplitReader that adds bucket conversion support.
///
/// Bucket conversion is needed when a table's bucket count is increased but
/// old partitions still use the original bucket count. In that case, a single
/// file may contain rows for multiple new buckets, and the reader must filter
/// to keep only rows belonging to the target bucket.
class HiveSplitReader : public FileSplitReader {
 public:
  HiveSplitReader(
      const std::shared_ptr<const HiveConnectorSplit>& hiveSplit,
      const FileTableHandlePtr& tableHandle,
      const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const FileConfig>& fileConfig,
      const RowTypePtr& readerOutputType,
      const std::shared_ptr<io::IoStatistics>& ioStatistics,
      const std::shared_ptr<IoStats>& ioStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* ioExecutor,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      const std::unordered_map<std::string, FileColumnHandlePtr>* infoColumns,
      std::vector<column_index_t> bucketChannels = {},
      const common::SubfieldFilters* subfieldFiltersForValidation = nullptr);

  ~HiveSplitReader() override = default;

  void prepareSplit(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      dwio::common::RuntimeStatistics& runtimeStats,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {})
      override;

  uint64_t next(uint64_t size, VectorPtr& output) override;

  const folly::F14FastSet<column_index_t>& bucketChannels() const {
    return bucketChannels_;
  }

 protected:
  void configureBaseReaderOptions() override;

  void configureBaseRowReaderOptions(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      RowTypePtr rowType) override;

  std::vector<BaseVector::CopyRange> bucketConversionRows(
      const RowVector& vector);

  void applyBucketConversion(
      VectorPtr& output,
      const std::vector<BaseVector::CopyRange>& ranges);

  void validateSynthesizedColumnFilters() const;

  const std::shared_ptr<const HiveConnectorSplit> hiveSplit_;

 private:
  std::vector<TypePtr> adaptColumns(
      const RowTypePtr& fileType,
      const RowTypePtr& tableSchema) const override;

  const std::unordered_map<std::string, FileColumnHandlePtr>* infoColumns_{
      nullptr};
  folly::F14FastSet<column_index_t> bucketChannels_;
  std::unique_ptr<HivePartitionFunction> partitionFunction_;
  std::vector<uint32_t> partitions_;
};

} // namespace facebook::velox::connector::hive
