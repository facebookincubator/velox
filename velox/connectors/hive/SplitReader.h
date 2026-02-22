/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HivePartitionFunction.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Reader.h"

namespace facebook::velox {
class BaseVector;
using VectorPtr = std::shared_ptr<BaseVector>;
} // namespace facebook::velox

namespace facebook::velox::common {
class MetadataFilter;
class ScanSpec;
} // namespace facebook::velox::common

namespace facebook::velox::connector {
class ConnectorQueryCtx;
} // namespace facebook::velox::connector

namespace facebook::velox::dwio::common {
struct RuntimeStatistics;
} // namespace facebook::velox::dwio::common

namespace facebook::velox::memory {
class MemoryPool;
}

namespace facebook::velox::connector::hive {

/// Creates a constant vector of size 1 from a string representation of a value.
///
/// Used to materialize partition column values and info columns (e.g., $path,
/// $file_size) when reading Hive and Iceberg tables. Partition values are
/// stored as strings in HiveConnectorSplit::partitionKeys and need to be
/// converted to their appropriate types.
///
/// @param type The target Velox type for the constant vector. Supports all
/// scalar types including primitives, dates, timestamps.
/// @param value The string representation of the value to convert, formatted
/// the same way as CAST(x as VARCHAR). Date values must be formatted using ISO
/// 8601 as YYYY-MM-DD. If nullopt, creates a null constant vector.
/// @param pool Memory pool for allocating the constant vector.
/// @param isLocalTimestamp If true and type is TIMESTAMP, interprets the string
/// value as local time and converts it to GMT. If false, treats the value
/// as already in GMT.
/// @param isDaysSinceEpoch If true and type is DATE, treats the string value as
/// an integer representing days since epoch (used by Iceberg). If false, parses
/// the string as a date string in ISO 8601 format (used by Hive).
///
/// @return A constant vector of size 1 containing the converted value, or a
/// null constant if value is nullopt.
/// @throws VeloxUserError if the string cannot be converted to the target type.
VectorPtr newConstantFromString(
    const TypePtr& type,
    const std::optional<std::string>& value,
    velox::memory::MemoryPool* pool,
    bool isLocalTimestamp,
    bool isDaysSinceEpoch);

struct HiveConnectorSplit;
class HiveTableHandle;
class HiveColumnHandle;
class HiveConfig;

class SplitReader {
 public:
  static std::unique_ptr<SplitReader> create(
      const std::shared_ptr<hive::HiveConnectorSplit>& hiveSplit,
      const std::shared_ptr<const HiveTableHandle>& hiveTableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<const HiveColumnHandle>>* partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      const RowTypePtr& readerOutputType,
      const std::shared_ptr<io::IoStatistics>& ioStatistics,
      const std::shared_ptr<IoStats>& ioStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* ioExecutor,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      const common::SubfieldFilters* subfieldFiltersForValidation = nullptr);

  virtual ~SplitReader() = default;

  void configureReaderOptions(
      std::shared_ptr<random::RandomSkipTracker> randomSkip);

  /// This function is used by different table formats like Iceberg and Hudi to
  /// do additional preparations before reading the split, e.g. Open delete
  /// files or log files, and add column adapatations for metadata columns. It
  /// would be called only once per incoming split
  virtual void prepareSplit(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      dwio::common::RuntimeStatistics& runtimeStats,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {});

  virtual uint64_t next(uint64_t size, VectorPtr& output);

  void resetFilterCaches();

  bool emptySplit() const;

  void resetSplit();

  int64_t estimatedRowSize() const;

  void updateRuntimeStats(dwio::common::RuntimeStatistics& stats) const;

  bool allPrefetchIssued() const;

  void setConnectorQueryCtx(const ConnectorQueryCtx* connectorQueryCtx);

  void setBucketConversion(std::vector<column_index_t> bucketChannels);

  /// Sets the info columns map for synthesized column filter validation.
  /// Must be called before prepareSplit() if synthesized column filter
  /// validation is needed.
  void setInfoColumns(
      const std::unordered_map<
          std::string,
          std::shared_ptr<const HiveColumnHandle>>* infoColumns) {
    infoColumns_ = infoColumns;
  }

  const RowTypePtr& readerOutputType() const {
    return readerOutputType_;
  }

  std::string toString() const;

 protected:
  SplitReader(
      const std::shared_ptr<const hive::HiveConnectorSplit>& hiveSplit,
      const std::shared_ptr<const HiveTableHandle>& hiveTableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<const HiveColumnHandle>>* partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      const RowTypePtr& readerOutputType,
      const std::shared_ptr<io::IoStatistics>& ioStatistics,
      const std::shared_ptr<IoStats>& ioStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      const common::SubfieldFilters* subfieldFiltersForValidation = nullptr);

  /// Create the dwio::common::Reader object baseReader_
  /// read the data file's metadata and schema
  void createReader(
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {});

  // Adjust the scan spec according to the current split, then return the
  // adapted row type.
  RowTypePtr getAdaptedRowType() const;

  // Check if the filters pass on the column statistics.  When delta update is
  // present, the corresonding filter should be disabled before calling this
  // function.
  bool filterOnStats(dwio::common::RuntimeStatistics& runtimeStats) const;

  /// Check if the hiveSplit_ is empty. The split is considered empty when
  ///   1) The data file is missing but the user chooses to ignore it
  ///   2) The file does not contain any rows
  ///   3) The data in the file does not pass the filters. The test is based on
  ///      the file metadata and partition key values
  /// This function needs to be called after baseReader_ is created.
  bool checkIfSplitIsEmpty(dwio::common::RuntimeStatistics& runtimeStats);

  /// Create the dwio::common::RowReader object baseRowReader_, which owns the
  /// ColumnReaders that will be used to read the data
  void createRowReader(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      RowTypePtr rowType,
      std::optional<bool> rowSizeTrackingEnabled);

  const folly::F14FastSet<column_index_t>& bucketChannels() const {
    return bucketChannels_;
  }

  std::vector<BaseVector::CopyRange> bucketConversionRows(
      const RowVector& vector);

  void applyBucketConversion(
      VectorPtr& output,
      const std::vector<BaseVector::CopyRange>& ranges);

  /// Sets a constant partition value on the scanSpec for a partition column.
  /// Converts the partition key string value to the appropriate type and sets
  /// it as a constant value in the scanSpec, so the column will be filled
  /// with this constant value.
  ///
  /// @param spec The scan spec to set the constant value on.
  /// @param partitionKey The name of the partition column.
  void setPartitionValue(
      common::ScanSpec* spec,
      const std::string& partitionKey,
      const std::optional<std::string>& value) const;

  /// Validates synthesized column filters against the split's info column
  /// values. This handles filter-only synthesized columns that are not in the
  /// scanSpec by checking them early before any file I/O.
  /// Throws if any synthesized column filter fails validation.
  void validateSynthesizedColumnFilters() const;

 private:
  /// Different table formats may have different meatadata columns.
  /// This function will be used to update the scanSpec for these columns.
  virtual std::vector<TypePtr> adaptColumns(
      const RowTypePtr& fileType,
      const RowTypePtr& tableSchema) const;

 protected:
  std::shared_ptr<const HiveConnectorSplit> hiveSplit_;
  const std::shared_ptr<const HiveTableHandle> hiveTableHandle_;
  const std::unordered_map<
      std::string,
      std::shared_ptr<const HiveColumnHandle>>* const partitionKeys_;
  // Column handles for synthesized columns (e.g., $path, $file_size).
  // Set via setInfoColumns() and used in validateSynthesizedColumnFilters().
  const std::unordered_map<
      std::string,
      std::shared_ptr<const HiveColumnHandle>>* infoColumns_;
  const ConnectorQueryCtx* connectorQueryCtx_;
  const std::shared_ptr<const HiveConfig> hiveConfig_;

  RowTypePtr readerOutputType_;
  const std::shared_ptr<io::IoStatistics> ioStatistics_;
  const std::shared_ptr<IoStats> ioStats_;
  FileHandleFactory* const fileHandleFactory_;
  folly::Executor* const ioExecutor_;
  memory::MemoryPool* const pool_;

  std::shared_ptr<common::ScanSpec> scanSpec_;
  // Subfield filters from HiveDataSource, includes both original
  // subfieldFilters and filters extracted from remainingFilter. Used to
  // validate synthesized column filters in prepareSplit() and adaptColumns().
  const common::SubfieldFilters* subfieldFiltersForValidation_;
  std::unique_ptr<dwio::common::Reader> baseReader_;
  std::unique_ptr<dwio::common::RowReader> baseRowReader_;
  dwio::common::ReaderOptions baseReaderOpts_;
  dwio::common::RowReaderOptions baseRowReaderOpts_;
  bool emptySplit_;

 private:
  folly::F14FastSet<column_index_t> bucketChannels_;
  std::unique_ptr<HivePartitionFunction> partitionFunction_;
  std::vector<uint32_t> partitions_;
};

} // namespace facebook::velox::connector::hive
