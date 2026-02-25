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

// Adapted from Apache Arrow.

#pragma once

#include <cstdint>
#include <memory>

#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"

namespace arrow {

class Array;
class ChunkedArray;
class RecordBatch;
class Schema;
class Table;

} // namespace arrow

namespace facebook::velox::parquet::arrow {

class FileMetaData;
class ParquetFileWriter;

namespace arrow {

/// \brief Iterative FileWriter class.
///
/// For basic usage, can write a Table at a time, creating one or more row
/// groups per write call.
///
/// For advanced usage, can write column-by-column: Start a new row group or
/// chunk with newRowGroup(), then write column-by-column the whole column
/// chunk.
///
/// If PARQUET:field_id is present as a metadata key on a field, and the
/// corresponding value is a nonnegative integer, then it will be used as the
/// field_id in the Parquet file.
class PARQUET_EXPORT FileWriter {
 public:
  static ::arrow::Status make(
      MemoryPool* pool,
      std::unique_ptr<ParquetFileWriter> writer,
      std::shared_ptr<::arrow::Schema> schema,
      std::shared_ptr<ArrowWriterProperties> arrowProperties,
      std::unique_ptr<FileWriter>* out);

  /// \brief Try to create an Arrow to Parquet file writer.
  ///
  /// \param schema Schema of data that will be passed.
  /// \param pool Memory pool to use.
  /// \param sink Output stream to write Parquet data.
  /// \param properties General Parquet writer properties.
  /// \param arrow_properties Arrow-specific writer properties.
  ///
  /// \since 11.0.0.
  static ::arrow::Result<std::unique_ptr<FileWriter>> open(
      const ::arrow::Schema& schema,
      MemoryPool* pool,
      std::shared_ptr<::arrow::io::OutputStream> sink,
      std::shared_ptr<WriterProperties> properties = defaultWriterProperties(),
      std::shared_ptr<ArrowWriterProperties> arrowProperties =
          defaultArrowWriterProperties());

  ARROW_DEPRECATED(
      "Deprecated in 11.0.0. Use Result-returning variants instead.")
  static ::arrow::Status open(
      const ::arrow::Schema& schema,
      MemoryPool* pool,
      std::shared_ptr<::arrow::io::OutputStream> sink,
      std::shared_ptr<WriterProperties> properties,
      std::unique_ptr<FileWriter>* writer);
  ARROW_DEPRECATED(
      "Deprecated in 11.0.0. Use Result-returning variants instead.")
  static ::arrow::Status open(
      const ::arrow::Schema& schema,
      MemoryPool* pool,
      std::shared_ptr<::arrow::io::OutputStream> sink,
      std::shared_ptr<WriterProperties> properties,
      std::shared_ptr<ArrowWriterProperties> arrowProperties,
      std::unique_ptr<FileWriter>* writer);

  /// Return the Arrow schema to be written to.
  virtual std::shared_ptr<::arrow::Schema> schema() const = 0;

  /// \brief Write a Table to Parquet.
  ///
  /// \param table Arrow table to write.
  /// \param chunk_size Maximum number of rows to write per row group.
  virtual ::arrow::Status writeTable(
      const ::arrow::Table& table,
      int64_t chunkSize = DEFAULT_MAX_ROW_GROUP_LENGTH) = 0;

  /// \brief Start a new row group.
  ///
  /// Returns an error if not all columns have been written.
  ///
  /// \param chunk_size The number of rows in the next row group.
  virtual ::arrow::Status newRowGroup(int64_t chunkSize) = 0;

  /// \brief Write ColumnChunk in row group using an array.
  virtual ::arrow::Status writeColumnChunk(const ::arrow::Array& data) = 0;

  /// \brief Write ColumnChunk in row group using slice of a ChunkedArray.
  virtual ::arrow::Status writeColumnChunk(
      const std::shared_ptr<::arrow::ChunkedArray>& data,
      int64_t offset,
      int64_t size) = 0;

  /// \brief Write ColumnChunk in a row group using a ChunkedArray.
  virtual ::arrow::Status writeColumnChunk(
      const std::shared_ptr<::arrow::ChunkedArray>& data) = 0;

  /// \brief Start a new buffered row group.
  ///
  /// Returns an error if not all columns have been written.
  virtual ::arrow::Status newBufferedRowGroup() = 0;

  /// \brief Write a RecordBatch into the buffered row group.
  ///
  /// Multiple RecordBatches can be written into the same row group through this
  /// method.
  ///
  /// WriterProperties.maxRowGroupLength() is respected and a new
  /// row group will be created if the current row group exceeds the
  /// limit.
  ///
  /// Batches get flushed to the output stream once newBufferedRowGroup()
  /// or close() is called.
  ///
  /// WARNING: If you are writing multiple files in parallel in the same
  /// executor, deadlock may occur if ArrowWriterProperties::useThreads
  /// is set to true to write columns in parallel. Please disable useThreads
  /// option in this case.
  virtual ::arrow::Status writeRecordBatch(
      const ::arrow::RecordBatch& batch) = 0;

  /// \brief Write the footer and close the file.
  virtual ::arrow::Status close() = 0;
  virtual ~FileWriter();

  virtual MemoryPool* memoryPool() const = 0;
  /// \brief Return the file metadata, only available after calling close().
  virtual const std::shared_ptr<FileMetaData> metadata() const = 0;
};

/// \brief Write Parquet file metadata only to indicated Arrow OutputStream.
PARQUET_EXPORT
::arrow::Status writeFileMetaData(
    const FileMetaData& fileMetadata,
    ::arrow::io::OutputStream* sink);

/// \brief Write metadata-only Parquet file to indicated Arrow OutputStream.
PARQUET_EXPORT
::arrow::Status writeMetaDataFile(
    const FileMetaData& fileMetadata,
    ::arrow::io::OutputStream* sink);

/// \brief Write a Table to Parquet.
///
/// This writes one table in a single shot. To write a Parquet file with
/// multiple tables iteratively, see parquet::arrow::FileWriter.
///
/// \param table Arrow table to write.
/// \param pool Memory pool to use.
/// \param sink Output stream to write Parquet data.
/// \param chunk_size Maximum number of rows to write per row group.
/// \param properties General Parquet writer properties.
/// \param arrow_properties Arrow-specific writer properties.
::arrow::Status PARQUET_EXPORT writeTable(
    const ::arrow::Table& table,
    MemoryPool* pool,
    std::shared_ptr<::arrow::io::OutputStream> sink,
    int64_t chunkSize = DEFAULT_MAX_ROW_GROUP_LENGTH,
    std::shared_ptr<WriterProperties> properties = defaultWriterProperties(),
    std::shared_ptr<ArrowWriterProperties> arrowProperties =
        defaultArrowWriterProperties());

} // namespace arrow
} // namespace facebook::velox::parquet::arrow
