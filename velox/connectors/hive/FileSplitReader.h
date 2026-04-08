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
#include "velox/connectors/hive/FileColumnHandle.h"
#include "velox/connectors/hive/FileConnectorSplit.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/FileTableHandle.h"
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

class FileConfig;

class FileSplitReader {
 public:
  static std::unique_ptr<FileSplitReader> create(
      const std::shared_ptr<const hive::FileConnectorSplit>& fileSplit,
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
      const common::SubfieldFilters* subfieldFiltersForValidation = nullptr);

  virtual ~FileSplitReader() = default;

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

  const RowTypePtr& readerOutputType() const {
    return readerOutputType_;
  }

  std::string toString() const;

 protected:
  FileSplitReader(
      const std::shared_ptr<const hive::FileConnectorSplit>& fileSplit,
      const FileTableHandlePtr& tableHandle,
      const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const FileConfig>& fileConfig,
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

  /// Check if the fileSplit_ is empty. The split is considered empty when
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

  /// Virtual hook called by configureReaderOptions() to set format-specific
  /// reader options on baseReaderOpts_. The base implementation calls the
  /// generic configureReaderOptions() from FileConnectorUtil. Subclasses
  /// (e.g., HiveSplitReader) override to call the Hive-specific version
  /// that also applies serde options.
  virtual void configureBaseReaderOptions();

  /// Virtual hook called by createRowReader() to set format-specific row
  /// reader options on baseRowReaderOpts_. The base implementation calls the
  /// generic configureRowReaderOptions() from FileConnectorUtil. Subclasses
  /// (e.g., HiveSplitReader) override to call the Hive-specific version
  /// that also applies serde parameters.
  virtual void configureBaseRowReaderOptions(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      RowTypePtr rowType);

 private:
  /// Different table formats may have different meatadata columns.
  /// This function will be used to update the scanSpec for these columns.
  virtual std::vector<TypePtr> adaptColumns(
      const RowTypePtr& fileType,
      const RowTypePtr& tableSchema) const;

 protected:
  std::shared_ptr<const FileConnectorSplit> fileSplit_;
  const FileTableHandlePtr tableHandle_;
  const std::unordered_map<std::string, FileColumnHandlePtr>* const
      partitionKeys_;

  const ConnectorQueryCtx* connectorQueryCtx_;
  const std::shared_ptr<const FileConfig> fileConfig_;

  RowTypePtr readerOutputType_;
  const std::shared_ptr<io::IoStatistics> ioStatistics_;
  const std::shared_ptr<IoStats> ioStats_;
  FileHandleFactory* const fileHandleFactory_;
  folly::Executor* const ioExecutor_;
  memory::MemoryPool* const pool_;

  const std::shared_ptr<common::ScanSpec> scanSpec_;
  // Subfield filters from HiveDataSource, includes both original
  // subfieldFilters and filters extracted from remainingFilter. Used by
  // subclasses (e.g., HiveSplitReader) for synthesized column filter
  // validation.
  const common::SubfieldFilters* const subfieldFiltersForValidation_;
  std::unique_ptr<dwio::common::Reader> baseReader_;
  std::unique_ptr<dwio::common::RowReader> baseRowReader_;
  dwio::common::ReaderOptions baseReaderOpts_;
  dwio::common::RowReaderOptions baseRowReaderOpts_;
  bool emptySplit_;
};

} // namespace facebook::velox::connector::hive
