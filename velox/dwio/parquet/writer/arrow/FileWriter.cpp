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

#include "velox/dwio/parquet/writer/arrow/FileWriter.h"

#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "arrow/util/key_value_metadata.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/writer/arrow/ColumnWriter.h"
#include "velox/dwio/parquet/writer/arrow/EncryptionInternal.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/FileEncryptorInternal.h"
#include "velox/dwio/parquet/writer/arrow/PageIndex.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"

using arrow::MemoryPool;

namespace facebook::velox::parquet::arrow {

using schema::GroupNode;

// ----------------------------------------------------------------------.
// RowGroupWriter public API.

RowGroupWriter::RowGroupWriter(std::unique_ptr<Contents> contents)
    : contents_(std::move(contents)) {}

void RowGroupWriter::close() {
  if (contents_) {
    contents_->close();
  }
}

ColumnWriter* RowGroupWriter::nextColumn() {
  return contents_->nextColumn();
}

ColumnWriter* RowGroupWriter::column(int i) {
  return contents_->column(i);
}

int64_t RowGroupWriter::totalCompressedBytes() const {
  return contents_->totalCompressedBytes();
}

int64_t RowGroupWriter::totalBytesWritten() const {
  return contents_->totalBytesWritten();
}

int64_t RowGroupWriter::totalCompressedBytesWritten() const {
  return contents_->totalCompressedBytesWritten();
}

int64_t RowGroupWriter::totalBufferedBytes() const {
  return contents_->totalCompressedBytes() +
      contents_->totalCompressedBytesWritten() +
      contents_->estimatedBufferedValueBytes();
}

bool RowGroupWriter::buffered() const {
  return contents_->buffered();
}

int RowGroupWriter::currentColumn() {
  return contents_->currentColumn();
}

int RowGroupWriter::numColumns() const {
  return contents_->numColumns();
}

int64_t RowGroupWriter::numRows() const {
  return contents_->numRows();
}

inline void throwRowsMisMatchError(int col, int64_t prev, int64_t curr) {
  std::stringstream ss;
  ss << "Column " << col << " had " << curr << " while previous column had "
     << prev;
  throw ParquetException(ss.str());
}

// ----------------------------------------------------------------------.
// RowGroupSerializer.

// RowGroupWriter::Contents implementation for the Parquet file specification.
class RowGroupSerializer : public RowGroupWriter::Contents {
 public:
  RowGroupSerializer(
      std::shared_ptr<ArrowOutputStream> sink,
      RowGroupMetaDataBuilder* metadata,
      int16_t rowGroupOrdinal,
      const WriterProperties* properties,
      bool bufferedRowGroup = false,
      InternalFileEncryptor* fileEncryptor = nullptr,
      PageIndexBuilder* pageIndexBuilder = nullptr)
      : sink_(std::move(sink)),
        metadata_(metadata),
        properties_(properties),
        totalBytesWritten_(0),
        totalCompressedBytesWritten_(0),
        closed_(false),
        rowGroupOrdinal_(rowGroupOrdinal),
        nextColumnIndex_(0),
        numRows_(0),
        bufferedRowGroup_(bufferedRowGroup),
        fileEncryptor_(fileEncryptor),
        pageIndexBuilder_(pageIndexBuilder) {
    if (bufferedRowGroup) {
      initColumns();
    } else {
      columnWriters_.push_back(nullptr);
    }
  }

  int numColumns() const override {
    return metadata_->numColumns();
  }

  int64_t numRows() const override {
    checkRowsWritten();
    // checkRowsWritten() ensures numRows_ is set correctly.
    return numRows_;
  }

  ColumnWriter* nextColumn() override {
    if (bufferedRowGroup_) {
      throw ParquetException(
          "nextColumn() is not supported when a RowGroup is written by size");
    }

    if (columnWriters_[0]) {
      checkRowsWritten();
    }

    // Throws an error if more columns are being written.
    auto colMeta = metadata_->nextColumnChunk();

    if (columnWriters_[0]) {
      totalBytesWritten_ += columnWriters_[0]->close();
      totalCompressedBytesWritten_ +=
          columnWriters_[0]->totalCompressedBytesWritten();
    }

    const int32_t columnOrdinal = nextColumnIndex_++;
    const auto& path = colMeta->descr()->path();
    auto metaEncryptor = fileEncryptor_
        ? fileEncryptor_->getColumnMetaEncryptor(path->toDotString())
        : nullptr;
    auto dataEncryptor = fileEncryptor_
        ? fileEncryptor_->getColumnDataEncryptor(path->toDotString())
        : nullptr;
    auto ciBuilder = pageIndexBuilder_ && properties_->pageIndexEnabled(path) &&
            properties_->statisticsEnabled(path)
        ? pageIndexBuilder_->getColumnIndexBuilder(columnOrdinal)
        : nullptr;
    auto oiBuilder = pageIndexBuilder_ && properties_->pageIndexEnabled(path)
        ? pageIndexBuilder_->getOffsetIndexBuilder(columnOrdinal)
        : nullptr;
    auto codecOptions = properties_->codecOptions(path)
        ? properties_->codecOptions(path).get()
        : nullptr;

    std::unique_ptr<PageWriter> pager;
    if (!codecOptions) {
      pager = PageWriter::open(
          sink_,
          properties_->compression(path),
          colMeta,
          rowGroupOrdinal_,
          static_cast<int16_t>(columnOrdinal),
          properties_->memoryPool(),
          false,
          metaEncryptor,
          dataEncryptor,
          properties_->pageChecksumEnabled(),
          ciBuilder,
          oiBuilder,
          CodecOptions());
    } else {
      pager = PageWriter::open(
          sink_,
          properties_->compression(path),
          colMeta,
          rowGroupOrdinal_,
          static_cast<int16_t>(columnOrdinal),
          properties_->memoryPool(),
          false,
          metaEncryptor,
          dataEncryptor,
          properties_->pageChecksumEnabled(),
          ciBuilder,
          oiBuilder,
          *codecOptions);
    }
    columnWriters_[0] =
        ColumnWriter::make(colMeta, std::move(pager), properties_);
    return columnWriters_[0].get();
  }

  ColumnWriter* column(int i) override {
    if (!bufferedRowGroup_) {
      throw ParquetException(
          "column() is only supported when a BufferedRowGroup is being written");
    }

    if (i >= 0 && i < static_cast<int>(columnWriters_.size())) {
      return columnWriters_[i].get();
    }
    return nullptr;
  }

  int currentColumn() const override {
    return metadata_->currentColumn();
  }

  int64_t totalCompressedBytes() const override {
    int64_t totalCompressedBytes = 0;
    for (size_t i = 0; i < columnWriters_.size(); i++) {
      if (columnWriters_[i]) {
        totalCompressedBytes += columnWriters_[i]->totalCompressedBytes();
      }
    }
    return totalCompressedBytes;
  }

  int64_t totalBytesWritten() const override {
    if (closed_) {
      return totalBytesWritten_;
    }
    int64_t totalBytesWritten = 0;
    for (size_t i = 0; i < columnWriters_.size(); i++) {
      if (columnWriters_[i]) {
        totalBytesWritten += columnWriters_[i]->totalBytesWritten();
      }
    }
    return totalBytesWritten;
  }

  int64_t totalCompressedBytesWritten() const override {
    if (closed_) {
      return totalCompressedBytesWritten_;
    }
    int64_t totalCompressedBytesWritten = 0;
    for (size_t i = 0; i < columnWriters_.size(); i++) {
      if (columnWriters_[i]) {
        totalCompressedBytesWritten +=
            columnWriters_[i]->totalCompressedBytesWritten();
      }
    }
    return totalCompressedBytesWritten;
  }

  int64_t estimatedBufferedValueBytes() const override {
    if (closed_) {
      return 0;
    }
    int64_t estimatedBufferedValueBytes = 0;
    for (size_t i = 0; i < columnWriters_.size(); i++) {
      if (columnWriters_[i]) {
        estimatedBufferedValueBytes +=
            columnWriters_[i]->estimatedBufferedValueBytes();
      }
    }
    return estimatedBufferedValueBytes;
  }

  bool buffered() const override {
    return bufferedRowGroup_;
  }

  void close() override {
    if (!closed_) {
      closed_ = true;
      checkRowsWritten();

      // Avoid invalid state if ColumnWriter::close() throws internally.
      auto columnWriters = std::move(columnWriters_);
      for (size_t i = 0; i < columnWriters.size(); i++) {
        if (columnWriters[i]) {
          totalBytesWritten_ += columnWriters[i]->close();
          totalCompressedBytesWritten_ +=
              columnWriters[i]->totalCompressedBytesWritten();
        }
      }

      // Ensures all columns have been written.
      metadata_->setNumRows(numRows_);
      metadata_->finish(totalBytesWritten_, rowGroupOrdinal_);
    }
  }

 private:
  std::shared_ptr<ArrowOutputStream> sink_;
  mutable RowGroupMetaDataBuilder* metadata_;
  const WriterProperties* properties_;
  int64_t totalBytesWritten_;
  int64_t totalCompressedBytesWritten_;
  bool closed_;
  int16_t rowGroupOrdinal_;
  int nextColumnIndex_;
  mutable int64_t numRows_;
  bool bufferedRowGroup_;
  InternalFileEncryptor* fileEncryptor_;
  PageIndexBuilder* pageIndexBuilder_;

  void checkRowsWritten() const {
    // Verify when only one column is written at a time.
    if (!bufferedRowGroup_ && columnWriters_.size() > 0 && columnWriters_[0]) {
      int64_t currentColRows = columnWriters_[0]->rowsWritten();
      if (numRows_ == 0) {
        numRows_ = currentColRows;
      } else if (numRows_ != currentColRows) {
        throwRowsMisMatchError(nextColumnIndex_, currentColRows, numRows_);
      }
    } else if (bufferedRowGroup_ && columnWriters_.size() > 0) {
      // When bufferedRowGroup = true.
      VELOX_DCHECK_NOT_NULL(columnWriters_[0]);
      int64_t currentColRows = columnWriters_[0]->rowsWritten();
      for (int i = 1; i < static_cast<int>(columnWriters_.size()); i++) {
        VELOX_DCHECK_NOT_NULL(columnWriters_[i]);
        int64_t currentColRowsI = columnWriters_[i]->rowsWritten();
        if (currentColRows != currentColRowsI) {
          throwRowsMisMatchError(i, currentColRowsI, currentColRows);
        }
      }
      numRows_ = currentColRows;
    }
  }

  void initColumns() {
    for (int i = 0; i < numColumns(); i++) {
      auto colMeta = metadata_->nextColumnChunk();
      const auto& path = colMeta->descr()->path();
      const int32_t columnOrdinal = nextColumnIndex_++;
      auto metaEncryptor = fileEncryptor_
          ? fileEncryptor_->getColumnMetaEncryptor(path->toDotString())
          : nullptr;
      auto dataEncryptor = fileEncryptor_
          ? fileEncryptor_->getColumnDataEncryptor(path->toDotString())
          : nullptr;
      auto ciBuilder = pageIndexBuilder_ && properties_->pageIndexEnabled(path)
          ? pageIndexBuilder_->getColumnIndexBuilder(columnOrdinal)
          : nullptr;
      auto oiBuilder = pageIndexBuilder_ && properties_->pageIndexEnabled(path)
          ? pageIndexBuilder_->getOffsetIndexBuilder(columnOrdinal)
          : nullptr;
      auto codecOptions = properties_->codecOptions(path);

      std::unique_ptr<PageWriter> pager;
      if (!codecOptions) {
        pager = PageWriter::open(
            sink_,
            properties_->compression(path),
            colMeta,
            rowGroupOrdinal_,
            static_cast<int16_t>(columnOrdinal),
            properties_->memoryPool(),
            bufferedRowGroup_,
            metaEncryptor,
            dataEncryptor,
            properties_->pageChecksumEnabled(),
            ciBuilder,
            oiBuilder,
            CodecOptions());
      } else {
        pager = PageWriter::open(
            sink_,
            properties_->compression(path),
            colMeta,
            rowGroupOrdinal_,
            static_cast<int16_t>(columnOrdinal),
            properties_->memoryPool(),
            bufferedRowGroup_,
            metaEncryptor,
            dataEncryptor,
            properties_->pageChecksumEnabled(),
            ciBuilder,
            oiBuilder,
            *codecOptions);
      }
      columnWriters_.push_back(
          ColumnWriter::make(colMeta, std::move(pager), properties_));
    }
  }

  std::vector<std::shared_ptr<ColumnWriter>> columnWriters_;
};

// ----------------------------------------------------------------------
// FileSerializer.

// An implementation of ParquetFileWriter::Contents that deals with the Parquet
// file structure, Thrift serialization, and other internal matters.

class FileSerializer : public ParquetFileWriter::Contents {
 public:
  static std::unique_ptr<ParquetFileWriter::Contents> open(
      std::shared_ptr<ArrowOutputStream> sink,
      std::shared_ptr<GroupNode> schema,
      std::shared_ptr<WriterProperties> properties,
      std::shared_ptr<const KeyValueMetadata> keyValueMetadata) {
    std::unique_ptr<ParquetFileWriter::Contents> result(new FileSerializer(
        std::move(sink),
        std::move(schema),
        std::move(properties),
        std::move(keyValueMetadata)));

    return result;
  }

  void close() override {
    if (isOpen_) {
      // If any functions here raise an exception, we set isOpen_ to be false
      // so that this does not get called again (possibly causing segfault).
      isOpen_ = false;
      if (rowGroupWriter_) {
        numRows_ += rowGroupWriter_->numRows();
        rowGroupWriter_->close();
      }
      rowGroupWriter_.reset();

      writePageIndex();

      // Write magic bytes and metadata.
      auto fileEncryptionProperties = properties_->fileEncryptionProperties();

      if (fileEncryptionProperties == nullptr) { // Non encrypted file.
        fileMetadata_ = metadata_->finish(keyValueMetadata_);
        writeFileMetaData(*fileMetadata_, sink_.get());
      } else { // Encrypted file.
        closeEncryptedFile(fileEncryptionProperties);
      }
    }
  }

  int numColumns() const override {
    return schema_.numColumns();
  }

  int numRowGroups() const override {
    return numRowGroups_;
  }

  int64_t numRows() const override {
    return numRows_;
  }

  const std::shared_ptr<WriterProperties>& properties() const override {
    return properties_;
  }

  RowGroupWriter* appendRowGroup(bool bufferedRowGroup) {
    if (rowGroupWriter_) {
      rowGroupWriter_->close();
    }
    numRowGroups_++;
    auto rgMetadata = metadata_->appendRowGroup();
    if (pageIndexBuilder_) {
      pageIndexBuilder_->appendRowGroup();
    }
    std::unique_ptr<RowGroupWriter::Contents> contents(new RowGroupSerializer(
        sink_,
        rgMetadata,
        static_cast<int16_t>(numRowGroups_ - 1),
        properties_.get(),
        bufferedRowGroup,
        fileEncryptor_.get(),
        pageIndexBuilder_.get()));
    rowGroupWriter_ = std::make_unique<RowGroupWriter>(std::move(contents));
    return rowGroupWriter_.get();
  }

  RowGroupWriter* appendRowGroup() override {
    return appendRowGroup(false);
  }

  RowGroupWriter* appendBufferedRowGroup() override {
    return appendRowGroup(true);
  }

  void addKeyValueMetadata(
      const std::shared_ptr<const KeyValueMetadata>& keyValueMetadata)
      override {
    if (keyValueMetadata_ == nullptr) {
      keyValueMetadata_ = keyValueMetadata;
    } else if (keyValueMetadata != nullptr) {
      keyValueMetadata_ = keyValueMetadata_->Merge(*keyValueMetadata);
    }
  }

  ~FileSerializer() override {
    try {
      FileSerializer::close();
    } catch (...) {
    }
  }

 private:
  FileSerializer(
      std::shared_ptr<ArrowOutputStream> sink,
      std::shared_ptr<GroupNode> schema,
      std::shared_ptr<WriterProperties> properties,
      std::shared_ptr<const KeyValueMetadata> keyValueMetadata)
      : ParquetFileWriter::Contents(
            std::move(schema),
            std::move(keyValueMetadata)),
        sink_(std::move(sink)),
        isOpen_(true),
        properties_(std::move(properties)),
        numRowGroups_(0),
        numRows_(0),
        metadata_(FileMetaDataBuilder::make(&schema_, properties_)) {
    PARQUET_ASSIGN_OR_THROW(int64_t position, sink_->Tell());
    if (position == 0) {
      startFile();
    } else {
      throw ParquetException("Appending to file not implemented.");
    }
  }

  void closeEncryptedFile(FileEncryptionProperties* fileEncryptionProperties) {
    // Encrypted file with encrypted footer.
    if (fileEncryptionProperties->encryptedFooter()) {
      // Encrypted footer.
      fileMetadata_ = metadata_->finish(keyValueMetadata_);

      PARQUET_ASSIGN_OR_THROW(int64_t position, sink_->Tell());
      uint64_t metadataStart = static_cast<uint64_t>(position);
      auto cryptoMetadata = metadata_->getCryptoMetaData();
      writeFileCryptoMetaData(*cryptoMetadata, sink_.get());

      auto footerEncryptor = fileEncryptor_->getFooterEncryptor();
      writeEncryptedFileMetadata(
          *fileMetadata_, sink_.get(), footerEncryptor, true);
      PARQUET_ASSIGN_OR_THROW(position, sink_->Tell());
      uint32_t footerAndCryptoLen =
          static_cast<uint32_t>(position - metadataStart);
      PARQUET_THROW_NOT_OK(
          sink_->Write(reinterpret_cast<uint8_t*>(&footerAndCryptoLen), 4));
      PARQUET_THROW_NOT_OK(sink_->Write(kParquetEMagic, 4));
    } else { // Encrypted file with plaintext footer
      fileMetadata_ = metadata_->finish(keyValueMetadata_);
      auto footerSigningEncryptor = fileEncryptor_->getFooterSigningEncryptor();
      writeEncryptedFileMetadata(
          *fileMetadata_, sink_.get(), footerSigningEncryptor, false);
    }
    if (fileEncryptor_) {
      fileEncryptor_->wipeOutEncryptionKeys();
    }
  }

  void writePageIndex() {
    if (pageIndexBuilder_ != nullptr) {
      if (properties_->fileEncryptionProperties()) {
        throw ParquetException("Encryption is not supported with page index");
      }

      // Serialize page index after all row groups have been written and report
      // location to the file metadata.
      PageIndexLocation pageIndexLocation;
      pageIndexBuilder_->finish();
      pageIndexBuilder_->writeTo(sink_.get(), &pageIndexLocation);
      metadata_->setPageIndexLocation(pageIndexLocation);
    }
  }

  std::shared_ptr<ArrowOutputStream> sink_;
  bool isOpen_;
  const std::shared_ptr<WriterProperties> properties_;
  int numRowGroups_;
  int64_t numRows_;
  std::unique_ptr<FileMetaDataBuilder> metadata_;
  // Only one of the row group writers is active at a time.
  std::unique_ptr<RowGroupWriter> rowGroupWriter_;
  std::unique_ptr<PageIndexBuilder> pageIndexBuilder_;
  std::unique_ptr<InternalFileEncryptor> fileEncryptor_;

  void startFile() {
    auto fileEncryptionProperties = properties_->fileEncryptionProperties();
    if (fileEncryptionProperties == nullptr) {
      // Unencrypted parquet files always start with PAR1.
      PARQUET_THROW_NOT_OK(sink_->Write(kParquetMagic, 4));
    } else {
      // Check that all columns in columnEncryptionProperties exist in the
      // schema.
      auto encryptedColumns = fileEncryptionProperties->encryptedColumns();
      // If columnEncryptionProperties is empty, every column in file schema
      // will be encrypted with footer key.
      if (encryptedColumns.size() != 0) {
        std::vector<std::string> columnPathVec;
        // First, save all column paths in schema.
        for (int i = 0; i < numColumns(); i++) {
          columnPathVec.push_back(schema_.column(i)->path()->toDotString());
        }
        // Check if column exists in schema.
        for (const auto& elem : encryptedColumns) {
          auto it =
              std::find(columnPathVec.begin(), columnPathVec.end(), elem.first);
          if (it == columnPathVec.end()) {
            std::stringstream ss;
            ss << "Encrypted column " + elem.first + " not in file schema";
            throw ParquetException(ss.str());
          }
        }
      }

      fileEncryptor_ = std::make_unique<InternalFileEncryptor>(
          fileEncryptionProperties, properties_->memoryPool());
      if (fileEncryptionProperties->encryptedFooter()) {
        PARQUET_THROW_NOT_OK(sink_->Write(kParquetEMagic, 4));
      } else {
        // Encrypted file with plaintext footer mode.
        PARQUET_THROW_NOT_OK(sink_->Write(kParquetMagic, 4));
      }
    }

    if (properties_->pageIndexEnabled()) {
      pageIndexBuilder_ = PageIndexBuilder::make(&schema_);
    }
  }
};

// ----------------------------------------------------------------------
// ParquetFileWriter public API.

ParquetFileWriter::ParquetFileWriter() {}

ParquetFileWriter::~ParquetFileWriter() {
  try {
    close();
  } catch (...) {
  }
}

std::unique_ptr<ParquetFileWriter> ParquetFileWriter::open(
    std::shared_ptr<::arrow::io::OutputStream> sink,
    std::shared_ptr<GroupNode> schema,
    std::shared_ptr<WriterProperties> properties,
    std::shared_ptr<const KeyValueMetadata> keyValueMetadata) {
  auto contents = FileSerializer::open(
      std::move(sink),
      std::move(schema),
      std::move(properties),
      std::move(keyValueMetadata));
  std::unique_ptr<ParquetFileWriter> result(new ParquetFileWriter());
  result->open(std::move(contents));
  return result;
}

void writeFileMetaData(
    const FileMetaData& fileMetadata,
    ArrowOutputStream* sink) {
  // Write metadata.
  PARQUET_ASSIGN_OR_THROW(int64_t position, sink->Tell());
  uint32_t metadataLen = static_cast<uint32_t>(position);

  fileMetadata.writeTo(sink);
  PARQUET_ASSIGN_OR_THROW(position, sink->Tell());
  metadataLen = static_cast<uint32_t>(position) - metadataLen;

  // Write Footer.
  PARQUET_THROW_NOT_OK(
      sink->Write(reinterpret_cast<uint8_t*>(&metadataLen), 4));
  PARQUET_THROW_NOT_OK(sink->Write(kParquetMagic, 4));
}

void writeMetaDataFile(
    const FileMetaData& fileMetadata,
    ArrowOutputStream* sink) {
  PARQUET_THROW_NOT_OK(sink->Write(kParquetMagic, 4));
  return writeFileMetaData(fileMetadata, sink);
}

void writeEncryptedFileMetadata(
    const FileMetaData& fileMetadata,
    ArrowOutputStream* sink,
    const std::shared_ptr<Encryptor>& encryptor,
    bool encryptFooter) {
  if (encryptFooter) { // Encrypted file with encrypted footer.
    // Encrypt and write to sink.
    fileMetadata.writeTo(sink, encryptor);
  } else { // Encrypted file with plaintext footer mode.
    PARQUET_ASSIGN_OR_THROW(int64_t position, sink->Tell());
    uint32_t metadataLen = static_cast<uint32_t>(position);
    fileMetadata.writeTo(sink, encryptor);
    PARQUET_ASSIGN_OR_THROW(position, sink->Tell());
    metadataLen = static_cast<uint32_t>(position) - metadataLen;

    PARQUET_THROW_NOT_OK(
        sink->Write(reinterpret_cast<uint8_t*>(&metadataLen), 4));
    PARQUET_THROW_NOT_OK(sink->Write(kParquetMagic, 4));
  }
}

void writeFileCryptoMetaData(
    const FileCryptoMetaData& cryptoMetadata,
    ArrowOutputStream* sink) {
  cryptoMetadata.writeTo(sink);
}

const SchemaDescriptor* ParquetFileWriter::schema() const {
  return contents_->schema();
}

const ColumnDescriptor* ParquetFileWriter::descr(int i) const {
  return contents_->schema()->column(i);
}

int ParquetFileWriter::numColumns() const {
  return contents_->numColumns();
}

int64_t ParquetFileWriter::numRows() const {
  return contents_->numRows();
}

int ParquetFileWriter::numRowGroups() const {
  return contents_->numRowGroups();
}

const std::shared_ptr<const KeyValueMetadata>&
ParquetFileWriter::keyValueMetadata() const {
  return contents_->keyValueMetadata();
}

const std::shared_ptr<FileMetaData> ParquetFileWriter::metadata() const {
  return fileMetadata_;
}

void ParquetFileWriter::open(
    std::unique_ptr<ParquetFileWriter::Contents> contents) {
  contents_ = std::move(contents);
}

void ParquetFileWriter::close() {
  if (contents_) {
    contents_->close();
    fileMetadata_ = contents_->metadata();
    contents_.reset();
  }
}

RowGroupWriter* ParquetFileWriter::appendRowGroup() {
  return contents_->appendRowGroup();
}

RowGroupWriter* ParquetFileWriter::appendBufferedRowGroup() {
  return contents_->appendBufferedRowGroup();
}

RowGroupWriter* ParquetFileWriter::appendRowGroup(int64_t numRows) {
  return appendRowGroup();
}

void ParquetFileWriter::addKeyValueMetadata(
    const std::shared_ptr<const KeyValueMetadata>& keyValueMetadata) {
  if (contents_) {
    contents_->addKeyValueMetadata(keyValueMetadata);
  } else {
    throw ParquetException("Cannot add key-value metadata to closed file");
  }
}

const std::shared_ptr<WriterProperties>& ParquetFileWriter::properties() const {
  return contents_->properties();
}

} // namespace facebook::velox::parquet::arrow
