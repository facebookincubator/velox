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
#include "velox/common/base/RandomUtil.h"
#include "velox/common/caching/FileHandle.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/common/FileColumnHandle.h"
#include "velox/connectors/common/FileConnectorSplit.h"
#include "velox/connectors/common/FileTableHandle.h"
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
struct RuntimeStats;
} // namespace facebook::velox::dwio::common

namespace facebook::velox::memory {
class MemoryPool;
}

namespace facebook::velox::connector {

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
      const std::shared_ptr<const connector::FileConnectorSplit>& fileSplit,
      const FileTableHandlePtr& tableHandle,
      const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const FileConfig>& fileConfig,
      const RowTypePtr& readerOutputType,
      const std::shared_ptr<io::IoStatistics>& dataIoStats,
      const std::shared_ptr<io::IoStatistics>& metadataIoStats,
      const std::shared_ptr<IoStats>& ioStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* ioExecutor,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      const common::SubfieldFilters* subfieldFiltersForValidation = nullptr);

  virtual ~FileSplitReader() = default;

  /// Returns false if no row of the file can pass the filters in 'scanSpec',
  /// judging by the filtered columns whose values do not come from the file:
  /// - a partition key with a value is tested on that value;
  /// - a column with a constant value is tested on the constant;
  /// - a partition key without a value, or a column missing from 'fileType',
  ///   is tested as NULL.
  /// Columns read from the file are left to common::testFilters.
  ///
  /// @param fileType The columns of the file.
  /// @param partitionKeys Partition values of the split by column name.
  /// @param partitionKeyHandles Handles of the partition key columns. Must
  /// include every partition key with a value that has a filter.
  /// @param asLocalTime Whether timestamp partition values are in local time.
  static bool testPartitionAndConstantFilters(
      const common::ScanSpec& scanSpec,
      const RowType& fileType,
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys,
      const std::unordered_map<std::string, FileColumnHandlePtr>&
          partitionKeyHandles,
      bool asLocalTime);

  /// Configures options for opening a data file: IO, metadata caching, file
  /// format and schema matching. Takes them from the connector config, the
  /// session and the split.
  ///
  /// @param fileSchema Columns of the table as stored in the file.
  /// @param tableParameters Table properties. Only the null string of text
  /// files is read from them.
  /// @param serdeParameters Serde properties of the split. Parsed into the text
  /// options the reader needs when it opens the file: field, collection and map
  /// key delimiters and the escape character. Applied only on the call that
  /// sets the file format.
  static void configureReaderOptions(
      const std::shared_ptr<const FileConfig>& fileConfig,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const FileConnectorSplit>& fileSplit,
      const RowTypePtr& fileSchema,
      const std::unordered_map<std::string, std::string>& tableParameters,
      const std::unordered_map<std::string, std::string>& serdeParameters,
      dwio::common::ReaderOptions& readerOptions);

  /// Configures options for one scan of an opened file: columns, filters,
  /// requested type and the byte or row range of the split.
  ///
  /// @param requestedType Type to read each file column as, which may differ
  /// from the type stored in the file after schema evolution. Null reads the
  /// file's own types.
  /// @param tableParameters Table properties. Only the number of header lines
  /// of text files to skip is read from them.
  /// @param serdeParameters Serde properties of the split, passed to the row
  /// reader as is for formats that read them per scan, such as SST.
  static void configureRowReaderOptions(
      const std::shared_ptr<const FileConfig>& fileConfig,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const FileConnectorSplit>& fileSplit,
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      const RowTypePtr& requestedType,
      folly::Executor* ioExecutor,
      const std::unordered_map<std::string, std::string>& tableParameters,
      const std::unordered_map<std::string, std::string>& serdeParameters,
      dwio::common::RowReaderOptions& rowReaderOptions);

  /// Completes construction: sets the options this reader opens its file and
  /// scans it with. Must be called once, before prepareSplit().
  ///
  /// @param randomSkip Samples rows when the scan has a sample rate. Null reads
  /// every row.
  /// @param remainingFilterColumns Columns the remaining filter reads.
  void initialize(
      std::shared_ptr<random::RandomSkipTracker> randomSkip,
      const folly::F14FastSet<std::string>& remainingFilterColumns);

  /// This function is used by different table formats like Iceberg and Hudi to
  /// do additional preparations before reading the split, e.g. Open delete
  /// files or log files, and add column adaptations for metadata columns. It
  /// would be called only once per incoming split
  virtual void prepareSplit(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      dwio::common::RuntimeStats& runtimeStats,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {});

  /// Returns true if prepareSplit() found that the split has no rows to read:
  /// its file is missing and missing files are ignored, the file has no rows,
  /// or its partition values or statistics rule out every row.
  bool emptySplit() const;

  /// Reads up to 'size' rows of the split into 'output', applying the scan
  /// spec's filters and the random-skip sampling, if any. Returns the number of
  /// rows scanned in the file, including rows filters or sampling removed, so
  /// 'output' can hold fewer rows. Returns 0 at the end of the split.
  virtual uint64_t next(uint64_t size, VectorPtr& output);

  /// Must be called after a filter in the scan spec changes, such as when a
  /// dynamic filter is added, so the row reader drops state derived from the
  /// old filters.
  void resetFilterCaches();

  /// Returns the file row number the next read starts at, or
  /// dwio::common::RowReader::kAtEnd at the end of the split or when no file is
  /// open yet.
  int64_t nextRowNumber() const;

  /// Returns the estimated size in bytes of an output row, from the file's
  /// statistics, or DataSource::kUnknownRowSize if it can't be estimated.
  int64_t estimatedRowSize() const;

  /// Returns true once the row reader has scheduled all the IO it will do for
  /// the split, so the data source can start prefetching the next split.
  /// Returns false before the row reader is created.
  bool allPrefetchIssued() const;

  /// Adds the row reader's runtime statistics to 'stats'.
  void updateRuntimeStats(dwio::common::RuntimeStats& stats) const;

  /// Releases the split after it has been read or skipped. Keeps the file and
  /// row readers, whose column adaptations the next split reuses.
  void resetSplit();

  /// Rebinds a reader prepared by another data source to the query context of
  /// the data source that adopts it, see FileDataSource::setFromDataSource().
  /// Must be called before reading from a reader that outlives the context it
  /// was prepared with.
  void setConnectorQueryCtx(const ConnectorQueryCtx* connectorQueryCtx);

  /// Returns the schema passed to the underlying row reader.
  ///
  /// For most split readers this returns the same type as readerOutputType_.
  /// IcebergChangelogSplitReader overrides this to return the *changelog*
  /// schema (operation/ordinal/snapshotid/rowdata) so FileDataSource allocates
  /// output vectors with the correct shape, while readerOutputType_ holds the
  /// base-table schema passed to the row reader.  Code inside
  /// FileSplitReader::next() and its base-class helpers must read
  /// readerOutputType_ directly; do NOT replace those reads with this getter
  /// without accounting for the override.
  virtual const RowTypePtr& readerOutputType() const {
    return readerOutputType_;
  }

  std::string toString() const;

 protected:
  FileSplitReader(
      const std::shared_ptr<const connector::FileConnectorSplit>& fileSplit,
      const FileTableHandlePtr& tableHandle,
      const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const FileConfig>& fileConfig,
      const RowTypePtr& readerOutputType,
      const std::shared_ptr<io::IoStatistics>& dataIoStats,
      const std::shared_ptr<io::IoStatistics>& metadataIoStats,
      const std::shared_ptr<IoStats>& ioStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      const common::SubfieldFilters* subfieldFiltersForValidation = nullptr);

  // Create the dwio::common::Reader object baseReader_
  // read the data file's metadata and schema
  void createReader(
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {});

  // Adjust the scan spec according to the current split, then return the
  // adapted row type.
  RowTypePtr getAdaptedRowType() const;

  // Check if the filters pass on the column statistics.  When delta update is
  // present, the corresonding filter should be disabled before calling this
  // function.
  bool filterOnStats(dwio::common::RuntimeStats& runtimeStats) const;

  // Check if the fileSplit_ is empty. The split is considered empty when
  //   1) The data file is missing but the user chooses to ignore it
  //   2) The file does not contain any rows
  //   3) The data in the file does not pass the filters. The test is based on
  //      the file metadata and partition key values
  // This function needs to be called after baseReader_ is created.
  bool checkIfSplitIsEmpty(dwio::common::RuntimeStats& runtimeStats);

  // Create the dwio::common::RowReader object baseRowReader_, which owns the
  // ColumnReaders that will be used to read the data
  void createRowReader(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      RowTypePtr rowType,
      std::optional<bool> rowSizeTrackingEnabled);

  // Sets a constant partition value on the scanSpec for a partition column.
  // Converts the partition key string value to the appropriate type and sets
  // it as a constant value in the scanSpec, so the column will be filled
  // with this constant value.
  //
  // @param spec The scan spec to set the constant value on.
  // @param partitionKey The name of the partition column.
  void setPartitionValue(
      common::ScanSpec* spec,
      const std::string& partitionKey,
      const std::optional<std::string>& value) const;

  // Virtual hook called by initialize() to set format-specific reader options
  // on baseReaderOpts_. The base implementation calls configureReaderOptions()
  // without serde parameters. HiveSplitReader overrides it to pass the serde
  // parameters of its split.
  virtual void configureBaseReaderOptions();

  // Virtual hook called by createRowReader() to set format-specific row
  // reader options on baseRowReaderOpts_. The base implementation calls
  // configureRowReaderOptions() without serde parameters. HiveSplitReader
  // overrides it to pass the serde parameters of its split.
  virtual void configureBaseRowReaderOptions(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      RowTypePtr rowType);

 private:
  // Returns the type to read each column of 'fileType' as: the output type for
  // columns in the file. Sets constant values in the scan spec for columns
  // whose values don't come from the file: partition values, NULL for columns
  // missing from the file, and format-specific columns, such as Hive info
  // columns or Iceberg metadata columns.
  virtual std::vector<TypePtr> adaptColumns(
      const RowTypePtr& fileType,
      const RowTypePtr& tableSchema) const;

 protected:
  const FileTableHandlePtr tableHandle_;
  const std::unordered_map<std::string, FileColumnHandlePtr>* const
      partitionKeys_;
  const std::shared_ptr<const FileConfig> fileConfig_;
  const std::shared_ptr<io::IoStatistics> dataIoStats_;
  const std::shared_ptr<io::IoStatistics> metadataIoStats_;
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

  std::shared_ptr<const FileConnectorSplit> fileSplit_;
  const ConnectorQueryCtx* connectorQueryCtx_;
  // Schema passed to the underlying row reader. Always read this member
  // directly inside FileSplitReader and its subclasses rather than calling
  // readerOutputType(), because IcebergChangelogSplitReader overrides the
  // getter to return the *changelog* schema while this member always holds the
  // type that was actually passed to the row reader.
  RowTypePtr readerOutputType_;
  std::unique_ptr<dwio::common::Reader> baseReader_;
  std::unique_ptr<dwio::common::RowReader> baseRowReader_;
  dwio::common::ReaderOptions baseReaderOpts_;
  dwio::common::RowReaderOptions baseRowReaderOpts_;
  bool emptySplit_;
};

} // namespace facebook::velox::connector
