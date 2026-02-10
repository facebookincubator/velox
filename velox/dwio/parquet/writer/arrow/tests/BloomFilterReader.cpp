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

#include "velox/dwio/parquet/writer/arrow/tests/BloomFilterReader.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/Metadata.h"
#include "velox/dwio/parquet/writer/arrow/tests/BloomFilter.h"

namespace facebook::velox::parquet::arrow {

class RowGroupBloomFilterReaderImpl final : public RowGroupBloomFilterReader {
 public:
  RowGroupBloomFilterReaderImpl(
      std::shared_ptr<::arrow::io::RandomAccessFile> input,
      std::shared_ptr<RowGroupMetaData> rowGroupMetadata,
      const ReaderProperties& properties)
      : input_(std::move(input)),
        rowGroupMetadata_(std::move(rowGroupMetadata)),
        properties_(properties) {}

  std::unique_ptr<BloomFilter> getColumnBloomFilter(int i) override;

 private:
  /// The input stream that can perform random access read.
  std::shared_ptr<::arrow::io::RandomAccessFile> input_;

  /// The row group metadata to get column chunk metadata.
  std::shared_ptr<RowGroupMetaData> rowGroupMetadata_;

  /// Reader properties used to deserialize thrift object.
  const ReaderProperties& properties_;
};

std::unique_ptr<BloomFilter>
RowGroupBloomFilterReaderImpl::getColumnBloomFilter(int i) {
  if (i < 0 || i >= rowGroupMetadata_->numColumns()) {
    throw ParquetException("Invalid column index at column ordinal ", i);
  }

  auto colChunk = rowGroupMetadata_->columnChunk(i);
  std::unique_ptr<ColumnCryptoMetaData> cryptoMetadata =
      colChunk->cryptoMetadata();
  if (cryptoMetadata != nullptr) {
    ParquetException::NYI("Cannot read encrypted bloom filter yet");
  }

  auto bloomFilterOffset = colChunk->bloomFilterOffset();
  if (!bloomFilterOffset.has_value()) {
    return nullptr;
  }
  PARQUET_ASSIGN_OR_THROW(auto fileSize, input_->GetSize());
  if (fileSize <= *bloomFilterOffset) {
    throw ParquetException("file size less or equal than bloom offset");
  }
  auto stream = ::arrow::io::RandomAccessFile::GetStream(
      input_, *bloomFilterOffset, fileSize - *bloomFilterOffset);
  auto bloomFilter =
      BlockSplitBloomFilter::deserialize(properties_, stream->get());
  return std::make_unique<BlockSplitBloomFilter>(std::move(bloomFilter));
}

class BloomFilterReaderImpl final : public BloomFilterReader {
 public:
  BloomFilterReaderImpl(
      std::shared_ptr<::arrow::io::RandomAccessFile> input,
      std::shared_ptr<FileMetaData> fileMetadata,
      const ReaderProperties& properties,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor)
      : input_(std::move(input)),
        fileMetadata_(std::move(fileMetadata)),
        properties_(properties) {
    if (fileDecryptor != nullptr) {
      ParquetException::NYI("BloomFilter decryption is not yet supported");
    }
  }

  std::shared_ptr<RowGroupBloomFilterReader> rowGroup(int i) {
    if (i < 0 || i >= fileMetadata_->numRowGroups()) {
      throw ParquetException("Invalid row group ordinal: ", i);
    }

    auto rowGroupMetadata = fileMetadata_->rowGroup(i);
    return std::make_shared<RowGroupBloomFilterReaderImpl>(
        input_, std::move(rowGroupMetadata), properties_);
  }

 private:
  /// The input stream that can perform random read.
  std::shared_ptr<::arrow::io::RandomAccessFile> input_;

  /// The file metadata to get row group metadata.
  std::shared_ptr<FileMetaData> fileMetadata_;

  /// Reader properties used to deserialize thrift object.
  const ReaderProperties& properties_;
};

std::unique_ptr<BloomFilterReader> BloomFilterReader::make(
    std::shared_ptr<::arrow::io::RandomAccessFile> input,
    std::shared_ptr<FileMetaData> fileMetadata,
    const ReaderProperties& properties,
    std::shared_ptr<InternalFileDecryptor> fileDecryptor) {
  return std::make_unique<BloomFilterReaderImpl>(
      std::move(input), fileMetadata, properties, std::move(fileDecryptor));
}

} // namespace facebook::velox::parquet::arrow
