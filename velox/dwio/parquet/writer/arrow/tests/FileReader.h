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

// Adapted from Apache Arrow.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "arrow/io/caching.h"
#include "arrow/util/type_fwd.h"
#include "velox/dwio/parquet/writer/arrow/Metadata.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"

namespace facebook::velox::parquet::arrow {

class ColumnReader;
class FileMetaData;
class PageIndexReader;
class BloomFilterReader;
class PageReader;
class RowGroupMetaData;

class PARQUET_EXPORT RowGroupReader {
 public:
  // Forward declare a virtual class 'Contents' to aid dependency injection and.
  // More easily create test fixtures An implementation of the Contents class
  // is. Defined in the .cc file.
  struct Contents {
    virtual ~Contents() {}
    virtual std::unique_ptr<PageReader> getColumnPageReader(int i) = 0;
    virtual const RowGroupMetaData* metadata() const = 0;
    virtual const ReaderProperties* properties() const = 0;
  };

  explicit RowGroupReader(std::unique_ptr<Contents> contents);

  // Returns the rowgroup metadata.
  const RowGroupMetaData* metadata() const;

  // Construct a ColumnReader for the indicated row group-relative.
  // Column. Ownership is shared with the RowGroupReader.
  std::shared_ptr<ColumnReader> column(int i);

  // Construct a ColumnReader, trying to enable exposed encoding.
  //
  // For dictionary encoding, currently we only support column chunks that are.
  // Fully dictionary encoded, i.e., all data pages in the column chunk are.
  // Dictionary encoded. If a column chunk uses dictionary encoding but then.
  // Falls back to plain encoding, the encoding will not be exposed.
  //
  // The returned column reader provides an API GetExposedEncoding() for the.
  // Users to check the exposed encoding and determine how to read the batches.
  //
  // \note API EXPERIMENTAL.
  std::shared_ptr<ColumnReader> columnWithExposeEncoding(
      int i,
      ExposedEncoding encodingToExpose);

  std::unique_ptr<PageReader> getColumnPageReader(int i);

 private:
  // Holds a pointer to an instance of Contents implementation.
  std::unique_ptr<Contents> contents_;
};

class PARQUET_EXPORT ParquetFileReader {
 public:
  // Declare a virtual class 'Contents' to aid dependency injection and more.
  // Easily create test fixtures.
  // An implementation of the Contents class is defined in the .cc file.
  struct PARQUET_EXPORT Contents {
    static std::unique_ptr<Contents> open(
        std::shared_ptr<::arrow::io::RandomAccessFile> source,
        const ReaderProperties& props = defaultReaderProperties(),
        std::shared_ptr<FileMetaData> metadata = NULLPTR);

    static ::arrow::Future<std::unique_ptr<Contents>> openAsync(
        std::shared_ptr<::arrow::io::RandomAccessFile> source,
        const ReaderProperties& props = defaultReaderProperties(),
        std::shared_ptr<FileMetaData> metadata = NULLPTR);

    virtual ~Contents() = default;
    // Perform any cleanup associated with the file contents.
    virtual void close() = 0;
    virtual std::shared_ptr<RowGroupReader> getRowGroup(int i) = 0;
    virtual std::shared_ptr<FileMetaData> metadata() const = 0;
    virtual std::shared_ptr<PageIndexReader> getPageIndexReader() = 0;
    virtual BloomFilterReader& getBloomFilterReader() = 0;
  };

  ParquetFileReader();
  ~ParquetFileReader();

  // Create a file reader instance from an Arrow file object. Thread-safety is.
  // The responsibility of the file implementation.
  static std::unique_ptr<ParquetFileReader> open(
      std::shared_ptr<::arrow::io::RandomAccessFile> source,
      const ReaderProperties& props = defaultReaderProperties(),
      std::shared_ptr<FileMetaData> metadata = NULLPTR);

  // API Convenience to open a serialized Parquet file on disk, using Arrow IO.
  // Interfaces.
  static std::unique_ptr<ParquetFileReader> openFile(
      const std::string& path,
      bool memoryMap = false,
      const ReaderProperties& props = defaultReaderProperties(),
      std::shared_ptr<FileMetaData> metadata = NULLPTR);

  // Asynchronously open a file reader from an Arrow file object.
  // Does not throw - all errors are reported through the Future.
  static ::arrow::Future<std::unique_ptr<ParquetFileReader>> openAsync(
      std::shared_ptr<::arrow::io::RandomAccessFile> source,
      const ReaderProperties& props = defaultReaderProperties(),
      std::shared_ptr<FileMetaData> metadata = NULLPTR);

  void open(std::unique_ptr<Contents> contents);
  void close();

  // The RowGroupReader is owned by the FileReader.
  std::shared_ptr<RowGroupReader> rowGroup(int i);

  // Returns the file metadata. Only one instance is ever created.
  std::shared_ptr<FileMetaData> metadata() const;

  /// Returns the PageIndexReader. Only one instance is ever created.
  ///
  /// If the file does not have the page index, nullptr may be returned.
  /// Because it pays to check existence of page index in the file, it.
  /// Is possible to return a non null value even if page index does.
  /// Not exist. It is the caller's responsibility to check the return.
  /// Value and follow-up calls to PageIndexReader.
  ///
  /// WARNING: The returned PageIndexReader must not outlive the.
  /// ParquetFileReader. Initialize GetPageIndexReader() is not thread-safety.
  std::shared_ptr<PageIndexReader> getPageIndexReader();

  /// Returns the BloomFilterReader. Only one instance is ever created.
  ///
  /// WARNING: The returned BloomFilterReader must not outlive the.
  /// ParquetFileReader. Initialize GetBloomFilterReader() is not thread-safety.
  BloomFilterReader& getBloomFilterReader();

  /// Pre-buffer the specified column indices in all row groups.
  ///
  /// Readers can optionally call this to cache the necessary slices.
  /// Of the file in-memory before deserialization. Arrow readers can.
  /// Automatically do this via an option. This is intended to.
  /// Increase performance when reading from high-latency filesystems.
  /// (E.g. Amazon S3).
  ///
  /// After calling this, creating readers for row groups/column.
  /// Indices that were not buffered may fail. Creating multiple.
  /// Readers for the a subset of the buffered regions is.
  /// Acceptable. This may be called again to buffer a different set.
  /// Of row groups/columns.
  ///
  /// If memory usage is a concern, note that data will remain.
  /// Buffered in memory until either \a PreBuffer() is called again,.
  /// Or the reader itself is destructed. Reading - and buffering -.
  /// Only one row group at a time may be useful.
  ///
  /// This method may throw.
  void preBuffer(
      const std::vector<int>& rowGroups,
      const std::vector<int>& columnIndices,
      const ::arrow::io::IOContext& ctx,
      const ::arrow::io::CacheOptions& options);

  /// Wait for the specified row groups and column indices to be pre-buffered.
  ///
  /// After the returned Future completes, reading the specified row.
  /// Groups/columns will not block.
  ///
  /// PreBuffer must be called first. This method does not throw.
  ::arrow::Future<> whenBuffered(
      const std::vector<int>& rowGroups,
      const std::vector<int>& columnIndices) const;

 private:
  // Holds a pointer to an instance of Contents implementation.
  std::unique_ptr<Contents> contents_;
};

// Read only Parquet file metadata.
std::shared_ptr<FileMetaData> PARQUET_EXPORT
readMetaData(const std::shared_ptr<::arrow::io::RandomAccessFile>& source);

/// \brief Scan all values in file. Useful for performance testing.
/// \param[in] columns the column numbers to scan. If empty scans all.
/// \param[in] column_batch_size number of values to read at a time when.
/// Scanning column \param[in] reader a ParquetFileReader instance \return.
/// Number of semantic rows in file.
PARQUET_EXPORT
int64_t scanFileContents(
    std::vector<int> columns,
    const int32_t columnBatchSize,
    ParquetFileReader* reader);

} // namespace facebook::velox::parquet::arrow
