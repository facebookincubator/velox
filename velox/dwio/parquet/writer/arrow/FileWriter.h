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
#include <utility>

#include "velox/dwio/parquet/writer/arrow/Metadata.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"

namespace facebook::velox::parquet::arrow {

class ColumnWriter;

// FIXME: copied from reader-internal.cc.
static constexpr uint8_t kParquetMagic[4] = {'P', 'A', 'R', '1'};
static constexpr uint8_t kParquetEMagic[4] = {'P', 'A', 'R', 'E'};

class PARQUET_EXPORT RowGroupWriter {
 public:
  // Forward declare a virtual class 'Contents' to aid dependency injection and.
  // More easily create test fixtures An implementation of the Contents class
  // is. Defined in the .cc file.
  struct Contents {
    virtual ~Contents() = default;
    virtual int numColumns() const = 0;
    virtual int64_t numRows() const = 0;

    // To be used only with ParquetFileWriter::AppendRowGroup.
    virtual ColumnWriter* nextColumn() = 0;
    // To be used only with ParquetFileWriter::AppendBufferedRowGroup.
    virtual ColumnWriter* column(int i) = 0;

    virtual int currentColumn() const = 0;
    virtual void close() = 0;

    /// \brief total uncompressed bytes written by the page writer.
    virtual int64_t totalBytesWritten() const = 0;
    /// \brief total bytes still compressed but not written by the page writer.
    virtual int64_t totalCompressedBytes() const = 0;
    /// \brief total compressed bytes written by the page writer.
    virtual int64_t totalCompressedBytesWritten() const = 0;
    /// \brief estimated size of the values that are not written to a page yet.
    virtual int64_t estimatedBufferedValueBytes() const = 0;

    virtual bool buffered() const = 0;
  };

  explicit RowGroupWriter(std::unique_ptr<Contents> contents);

  /// Construct a columnWriter for the indicated row group-relative column.
  ///
  /// To be used only with ParquetFileWriter::AppendRowGroup.
  /// Ownership is solely within the RowGroupWriter. The columnWriter is only.
  /// Valid until the next call to NextColumn or Close. As the contents are.
  /// Directly written to the sink, once a new column is started, the contents.
  /// Of the previous one cannot be modified anymore.
  ColumnWriter* nextColumn();
  /// Index of currently written column. Equal to -1 if NextColumn()
  /// Has not been called yet.
  int currentColumn();
  void close();

  int numColumns() const;

  /// Construct a columnWriter for the indicated row group column.
  ///
  /// To be used only with ParquetFileWriter::AppendBufferedRowGroup.
  /// Ownership is solely within the RowGroupWriter. The columnWriter is.
  /// Valid until Close. The contents are buffered in memory and written to
  /// sink. On Close.
  ColumnWriter* column(int i);

  /**
   * Number of rows that shall be written as part of this RowGroup.
   */
  int64_t numRows() const;

  /// \brief total uncompressed bytes written by the page writer.
  int64_t totalBytesWritten() const;
  /// \brief total bytes still compressed but not written by the page writer.
  /// It will always return 0 from the SerializedPageWriter.
  int64_t totalCompressedBytes() const;
  /// \brief total compressed bytes written by the page writer.
  int64_t totalCompressedBytesWritten() const;
  /// \brief including compressed bytes in page writer and uncompressed data
  /// value buffer.
  int64_t totalBufferedBytes() const;
  /// Returns whether the current rowGroupWriter is in the buffered mode and is.
  /// Created by calling ParquetFileWriter::AppendBufferedRowGroup.
  bool buffered() const;

 private:
  // Holds a pointer to an instance of Contents implementation.
  std::unique_ptr<Contents> contents_;
};

PARQUET_EXPORT
void writeFileMetaData(
    const FileMetaData& fileMetadata,
    ::arrow::io::OutputStream* sink);

PARQUET_EXPORT
void writeMetaDataFile(
    const FileMetaData& fileMetadata,
    ::arrow::io::OutputStream* sink);

PARQUET_EXPORT
void writeEncryptedFileMetadata(
    const FileMetaData& fileMetadata,
    ArrowOutputStream* sink,
    const std::shared_ptr<Encryptor>& Encryptor,
    bool encryptFooter);

PARQUET_EXPORT
void writeEncryptedFileMetadata(
    const FileMetaData& fileMetadata,
    ::arrow::io::OutputStream* sink,
    const std::shared_ptr<Encryptor>& Encryptor = NULLPTR,
    bool encryptFooter = false);
PARQUET_EXPORT
void writeFileCryptoMetaData(
    const FileCryptoMetaData& cryptoMetadata,
    ::arrow::io::OutputStream* sink);

class PARQUET_EXPORT ParquetFileWriter {
 public:
  // Forward declare a virtual class 'Contents' to aid dependency injection and.
  // More easily create test fixtures An implementation of the Contents class
  // is. Defined in the .cc file.
  struct Contents {
    Contents(
        std::shared_ptr<schema::GroupNode> schema,
        std::shared_ptr<const KeyValueMetadata> keyValueMetadata)
        : schema_(), keyValueMetadata_(std::move(keyValueMetadata)) {
      schema_.init(std::move(schema));
    }
    virtual ~Contents() {}
    // Perform any cleanup associated with the file contents.
    virtual void close() = 0;

    /// \note Deprecated since 1.3.0.
    RowGroupWriter* appendRowGroup(int64_t numRows);

    virtual RowGroupWriter* appendRowGroup() = 0;
    virtual RowGroupWriter* appendBufferedRowGroup() = 0;

    virtual int64_t numRows() const = 0;
    virtual int numColumns() const = 0;
    virtual int numRowGroups() const = 0;

    virtual const std::shared_ptr<WriterProperties>& properties() const = 0;

    const std::shared_ptr<const KeyValueMetadata>& keyValueMetadata() const {
      return keyValueMetadata_;
    }

    virtual void addKeyValueMetadata(
        const std::shared_ptr<const KeyValueMetadata>& keyValueMetadata) = 0;

    // Return const-pointer to make it clear that this object is not to be.
    // Copied.
    const SchemaDescriptor* schema() const {
      return &schema_;
    }

    SchemaDescriptor schema_;

    /// This should be the only place this is stored. Everything else is a
    /// const. Reference.
    std::shared_ptr<const KeyValueMetadata> keyValueMetadata_;

    const std::shared_ptr<FileMetaData>& metadata() const {
      return fileMetadata_;
    }
    std::shared_ptr<FileMetaData> fileMetadata_;
  };

  ParquetFileWriter();
  ~ParquetFileWriter();

  static std::unique_ptr<ParquetFileWriter> open(
      std::shared_ptr<::arrow::io::OutputStream> sink,
      std::shared_ptr<schema::GroupNode> schema,
      std::shared_ptr<WriterProperties> properties = defaultWriterProperties(),
      std::shared_ptr<const KeyValueMetadata> keyValueMetadata = NULLPTR);

  void open(std::unique_ptr<Contents> contents);
  void close();

  // Construct a rowGroupWriter for the indicated number of rows.
  //
  // Ownership is solely within the ParquetFileWriter. The rowGroupWriter is.
  // Only valid until the next call to AppendRowGroup or AppendBufferedRowGroup.
  // Or Close.
  // @param num_rows The number of rows that are stored in the new RowGroup.
  //
  // \deprecated Since 1.3.0.
  RowGroupWriter* appendRowGroup(int64_t numRows);

  /// Construct a rowGroupWriter with an arbitrary number of rows.
  ///
  /// Ownership is solely within the ParquetFileWriter. The rowGroupWriter is.
  /// Only valid until the next call to AppendRowGroup or
  /// AppendBufferedRowGroup. Or Close.
  RowGroupWriter* appendRowGroup();

  /// Construct a rowGroupWriter that buffers all the values until the RowGroup.
  /// Is ready. Use this if you want to write a RowGroup based on a certain
  /// size.
  ///
  /// Ownership is solely within the ParquetFileWriter. The rowGroupWriter is.
  /// Only valid until the next call to AppendRowGroup or
  /// AppendBufferedRowGroup. Or Close.
  RowGroupWriter* appendBufferedRowGroup();

  /// \brief Add key-value metadata to the file.
  /// \param[in] key_value_metadata the metadata to add.
  /// \note This will overwrite any existing metadata with the same key.
  /// \throw ParquetException if Close() has been called.
  void addKeyValueMetadata(
      const std::shared_ptr<const KeyValueMetadata>& keyValueMetadata);

  /// Number of columns.
  ///
  /// This number is fixed during the lifetime of the writer as it is
  /// determined. Via the schema.
  int numColumns() const;

  /// Number of rows in the yet started RowGroups.
  ///
  /// Changes on the addition of a new RowGroup.
  int64_t numRows() const;

  /// Number of started RowGroups.
  int numRowGroups() const;

  /// Configuration passed to the writer, e.g. the used Parquet format version.
  const std::shared_ptr<WriterProperties>& properties() const;

  /// Returns the file schema descriptor.
  const SchemaDescriptor* schema() const;

  /// Returns a column descriptor in schema.
  const ColumnDescriptor* descr(int i) const;

  /// Returns the file custom metadata.
  const std::shared_ptr<const KeyValueMetadata>& keyValueMetadata() const;

  /// Returns the file metadata, only available after calling Close().
  const std::shared_ptr<FileMetaData> metadata() const;

 private:
  // Holds a pointer to an instance of Contents implementation.
  std::unique_ptr<Contents> contents_;
  std::shared_ptr<FileMetaData> fileMetadata_;
};

} // namespace facebook::velox::parquet::arrow
