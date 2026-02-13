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

#include "velox/dwio/parquet/writer/arrow/tests/FileReader.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "arrow/io/caching.h"
#include "arrow/io/file.h"
#include "arrow/io/memory.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/future.h"
#include "arrow/util/int_util_overflow.h"
#include "arrow/util/ubsan.h"

#include "velox/dwio/parquet/writer/arrow/EncryptionInternal.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/FileDecryptorInternal.h"
#include "velox/dwio/parquet/writer/arrow/FileWriter.h"
#include "velox/dwio/parquet/writer/arrow/Metadata.h"
#include "velox/dwio/parquet/writer/arrow/PageIndex.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"
#include "velox/dwio/parquet/writer/arrow/tests/BloomFilter.h"
#include "velox/dwio/parquet/writer/arrow/tests/BloomFilterReader.h"
#include "velox/dwio/parquet/writer/arrow/tests/ColumnReader.h"
#include "velox/dwio/parquet/writer/arrow/tests/ColumnScanner.h"

using arrow::internal::AddWithOverflow;

namespace facebook::velox::parquet::arrow {

// PARQUET-978: Minimize footer reads by reading 64 KB from the end of the file.
static constexpr int64_t kDefaultFooterReadSize = 64 * 1024;
static constexpr uint32_t kFooterSize = 8;

// For PARQUET-816.
static constexpr int64_t kMaxDictHeaderSize = 100;

// ----------------------------------------------------------------------.
// RowGroupReader public API.

RowGroupReader::RowGroupReader(std::unique_ptr<Contents> contents)
    : contents_(std::move(contents)) {}

std::shared_ptr<ColumnReader> RowGroupReader::column(int i) {
  if (i >= metadata()->numColumns()) {
    std::stringstream ss;
    ss << "Trying to read column index " << i
       << " but row group metadata has only " << metadata()->numColumns()
       << " columns";
    throw ParquetException(ss.str());
  }
  const ColumnDescriptor* descr = metadata()->schema()->column(i);

  std::unique_ptr<PageReader> PageReader = contents_->getColumnPageReader(i);
  return ColumnReader::make(
      descr,
      std::move(PageReader),
      const_cast<ReaderProperties*>(contents_->properties())->memoryPool());
}

std::shared_ptr<ColumnReader> RowGroupReader::columnWithExposeEncoding(
    int i,
    ExposedEncoding encodingToExpose) {
  std::shared_ptr<ColumnReader> reader = column(i);

  if (encodingToExpose == ExposedEncoding::kDictionary) {
    // Check the encoding_stats to see if all data pages are dictionary encoded.
    std::unique_ptr<ColumnChunkMetaData> col = metadata()->columnChunk(i);
    const std::vector<PageEncodingStats>& encodingStats = col->encodingStats();
    if (encodingStats.empty()) {
      // Some parquet files may have empty encoding_stats. In this case we are.
      // Not sure whether all data pages are dictionary encoded. So we do not.
      // Enable exposing dictionary.
      return reader;
    }
    // The 1st page should be the dictionary page.
    if (encodingStats[0].pageType != PageType::kDictionaryPage ||
        (encodingStats[0].encoding != Encoding::kPlain &&
         encodingStats[0].encoding != Encoding::kPlainDictionary)) {
      return reader;
    }
    // The following pages should be dictionary encoded data pages.
    for (size_t idx = 1; idx < encodingStats.size(); ++idx) {
      if ((encodingStats[idx].encoding != Encoding::kRleDictionary &&
           encodingStats[idx].encoding != Encoding::kPlainDictionary) ||
          (encodingStats[idx].pageType != PageType::kDataPage &&
           encodingStats[idx].pageType != PageType::kDataPageV2)) {
        return reader;
      }
    }
  } else {
    // Exposing other encodings are not supported for now.
    return reader;
  }

  // Set exposed encoding.
  reader->setExposedEncoding(encodingToExpose);
  return reader;
}

std::unique_ptr<PageReader> RowGroupReader::getColumnPageReader(int i) {
  if (i >= metadata()->numColumns()) {
    std::stringstream ss;
    ss << "Trying to read column index " << i
       << " but row group metadata has only " << metadata()->numColumns()
       << " columns";
    throw ParquetException(ss.str());
  }
  return contents_->getColumnPageReader(i);
}

// Returns the rowgroup metadata.
const RowGroupMetaData* RowGroupReader::metadata() const {
  return contents_->metadata();
}

/// Compute the section of the file that should be read for the given.
/// Row group and column chunk.
::arrow::io::ReadRange computeColumnChunkRange(
    FileMetaData* fileMetadata,
    int64_t sourceSize,
    int rowGroupIndex,
    int columnIndex) {
  auto rowGroupMetadata = fileMetadata->rowGroup(rowGroupIndex);
  auto columnMetadata = rowGroupMetadata->columnChunk(columnIndex);

  int64_t colStart = columnMetadata->dataPageOffset();
  if (columnMetadata->hasDictionaryPage() &&
      columnMetadata->dictionaryPageOffset() > 0 &&
      colStart > columnMetadata->dictionaryPageOffset()) {
    colStart = columnMetadata->dictionaryPageOffset();
  }

  int64_t colLength = columnMetadata->totalCompressedSize();
  int64_t colEnd;
  if (colStart < 0 || colLength < 0) {
    throw ParquetException("Invalid column metadata (corrupt file?)");
  }

  if (AddWithOverflow(colStart, colLength, &colEnd) || colEnd > sourceSize) {
    throw ParquetException("Invalid column metadata (corrupt file?)");
  }

  // PARQUET-816 workaround for old files created by older parquet-mr.
  const ApplicationVersion& version = fileMetadata->writerVersion();
  if (version.versionLt(ApplicationVersion::PARQUET_816_FIXED_VERSION())) {
    // The Parquet MR writer had a bug in 1.2.8 and below where it didn't.
    // Include the dictionary page header size in total_compressed_size and.
    // Total_uncompressed_size (see IMPALA-694). We add padding to compensate.
    int64_t bytesRemaining = sourceSize - colEnd;
    int64_t padding = std::min<int64_t>(kMaxDictHeaderSize, bytesRemaining);
    colLength += padding;
  }

  return {colStart, colLength};
}

// RowGroupReader::Contents implementation for the Parquet file specification.
class SerializedRowGroup : public RowGroupReader::Contents {
 public:
  SerializedRowGroup(
      std::shared_ptr<ArrowInputFile> source,
      std::shared_ptr<::arrow::io::internal::ReadRangeCache> cachedSource,
      int64_t sourceSize,
      FileMetaData* fileMetadata,
      int rowGroupNumber,
      const ReaderProperties& props,
      std::shared_ptr<Buffer> prebufferedColumnChunksBitmap,
      std::shared_ptr<InternalFileDecryptor> fileDecryptor = nullptr)
      : source_(std::move(source)),
        cachedSource_(std::move(cachedSource)),
        sourceSize_(sourceSize),
        fileMetadata_(fileMetadata),
        properties_(props),
        rowGroupOrdinal_(rowGroupNumber),
        prebufferedColumnChunksBitmap_(
            std::move(prebufferedColumnChunksBitmap)),
        fileDecryptor_(fileDecryptor) {
    rowGroupMetadata_ = fileMetadata->rowGroup(rowGroupNumber);
  }

  const RowGroupMetaData* metadata() const override {
    return rowGroupMetadata_.get();
  }

  const ReaderProperties* properties() const override {
    return &properties_;
  }

  std::unique_ptr<PageReader> getColumnPageReader(int i) override {
    // Read column chunk from the file.
    auto col = rowGroupMetadata_->columnChunk(i);

    ::arrow::io::ReadRange colRange = computeColumnChunkRange(
        fileMetadata_, sourceSize_, rowGroupOrdinal_, i);
    std::shared_ptr<ArrowInputStream> stream;
    if (cachedSource_ && prebufferedColumnChunksBitmap_ != nullptr &&
        ::arrow::bit_util::GetBit(prebufferedColumnChunksBitmap_->data(), i)) {
      // PARQUET-1698: if read coalescing is enabled, read from pre-buffered.
      // Segments.
      PARQUET_ASSIGN_OR_THROW(auto buffer, cachedSource_->Read(colRange));
      stream = std::make_shared<::arrow::io::BufferReader>(buffer);
    } else {
      stream = properties_.getStream(source_, colRange.offset, colRange.length);
    }

    std::unique_ptr<ColumnCryptoMetaData> cryptoMetadata =
        col->cryptoMetadata();

    // Prior to Arrow 3.0.0, is_compressed was always set to false in column.
    // Headers, even if compression was used. See ARROW-17100.
    bool alwaysCompressed = fileMetadata_->writerVersion().versionLt(
        ApplicationVersion::PARQUET_CPP_10353_FIXED_VERSION());

    // Column is encrypted only if crypto_metadata exists.
    if (!cryptoMetadata) {
      return PageReader::open(
          stream,
          col->numValues(),
          col->compression(),
          properties_,
          alwaysCompressed);
    }

    if (fileDecryptor_ == nullptr) {
      throw ParquetException(
          "RowGroup is noted as encrypted but no file decryptor");
    }

    constexpr auto kEncryptedRowGroupsLimit = 32767;
    if (i > kEncryptedRowGroupsLimit) {
      throw ParquetException(
          "Encrypted files cannot contain more than 32767 row groups");
    }

    // The column is encrypted.
    std::shared_ptr<Decryptor> metaDecryptor;
    std::shared_ptr<Decryptor> dataDecryptor;
    // The column is encrypted with footer key.
    if (cryptoMetadata->encryptedWithFooterKey()) {
      metaDecryptor = fileDecryptor_->getFooterDecryptorForColumnMeta();
      dataDecryptor = fileDecryptor_->getFooterDecryptorForColumnData();
      CryptoContext ctx(
          col->hasDictionaryPage(),
          rowGroupOrdinal_,
          static_cast<int16_t>(i),
          metaDecryptor,
          dataDecryptor);
      return PageReader::open(
          stream,
          col->numValues(),
          col->compression(),
          properties_,
          alwaysCompressed,
          &ctx);
    }

    // The column is encrypted with its own key.
    std::string columnKeyMetadata = cryptoMetadata->keyMetadata();
    const std::string ColumnPath =
        cryptoMetadata->pathInSchema()->toDotString();

    metaDecryptor =
        fileDecryptor_->getColumnMetaDecryptor(ColumnPath, columnKeyMetadata);
    dataDecryptor =
        fileDecryptor_->getColumnDataDecryptor(ColumnPath, columnKeyMetadata);

    CryptoContext ctx(
        col->hasDictionaryPage(),
        rowGroupOrdinal_,
        static_cast<int16_t>(i),
        metaDecryptor,
        dataDecryptor);
    return PageReader::open(
        stream,
        col->numValues(),
        col->compression(),
        properties_,
        alwaysCompressed,
        &ctx);
  }

 private:
  std::shared_ptr<ArrowInputFile> source_;
  // Will be nullptr if PreBuffer() is not called.
  std::shared_ptr<::arrow::io::internal::ReadRangeCache> cachedSource_;
  int64_t sourceSize_;
  FileMetaData* fileMetadata_;
  std::unique_ptr<RowGroupMetaData> rowGroupMetadata_;
  ReaderProperties properties_;
  int rowGroupOrdinal_;
  const std::shared_ptr<Buffer> prebufferedColumnChunksBitmap_;
  std::shared_ptr<InternalFileDecryptor> fileDecryptor_;
};

// ----------------------------------------------------------------------.
// SerializedFile: An implementation of ParquetFileReader::Contents that deals.
// With the Parquet file structure, Thrift deserialization, and other internal.
// Matters.

// This class takes ownership of the provided data source.
class SerializedFile : public ParquetFileReader::Contents {
 public:
  SerializedFile(
      std::shared_ptr<ArrowInputFile> source,
      const ReaderProperties& props = defaultReaderProperties())
      : source_(std::move(source)), properties_(props) {
    PARQUET_ASSIGN_OR_THROW(sourceSize_, source_->GetSize());
  }

  ~SerializedFile() override {
    try {
      close();
    } catch (...) {
    }
  }

  void close() override {
    if (fileDecryptor_)
      fileDecryptor_->wipeOutDecryptionKeys();
  }

  std::shared_ptr<RowGroupReader> getRowGroup(int i) override {
    std::shared_ptr<Buffer> prebufferedColumnChunksBitmap;
    // Avoid updating the bitmap as this function can be called concurrently.
    // The bitmap can only be updated within Prebuffer().
    auto prebufferedColumnChunksIter = prebufferedColumnChunks_.find(i);
    if (prebufferedColumnChunksIter != prebufferedColumnChunks_.end()) {
      prebufferedColumnChunksBitmap = prebufferedColumnChunksIter->second;
    }

    std::unique_ptr<SerializedRowGroup> contents =
        std::make_unique<SerializedRowGroup>(
            source_,
            cachedSource_,
            sourceSize_,
            fileMetadata_.get(),
            i,
            properties_,
            std::move(prebufferedColumnChunksBitmap),
            fileDecryptor_);
    return std::make_shared<RowGroupReader>(std::move(contents));
  }

  std::shared_ptr<FileMetaData> metadata() const override {
    return fileMetadata_;
  }

  std::shared_ptr<PageIndexReader> getPageIndexReader() override {
    if (!fileMetadata_) {
      // Usually this won't happen if user calls one of the static Open()
      // Functions to create a ParquetFileReader instance. But if user calls
      // the. Constructor directly and calls GetPageIndexReader() before Open()
      // then. This could happen.
      throw ParquetException(
          "Cannot call GetPageIndexReader() due to missing file metadata. Did you "
          "forget to call ParquetFileReader::Open() first?");
    }
    if (!pageIndexReader_) {
      pageIndexReader_ = PageIndexReader::make(
          source_.get(), fileMetadata_, properties_, fileDecryptor_);
    }
    return pageIndexReader_;
  }

  BloomFilterReader& getBloomFilterReader() override {
    if (!fileMetadata_) {
      // Usually this won't happen if user calls one of the static Open()
      // Functions to create a ParquetFileReader instance. But if user calls
      // the. constructor directly and calls GetBloomFilterReader() before
      // Open() Then this could happen.
      throw ParquetException(
          "Cannot call GetBloomFilterReader() due to missing file metadata. Did you "
          "forget to call ParquetFileReader::Open() first?");
    }
    if (!bloomFilterReader_) {
      bloomFilterReader_ = BloomFilterReader::make(
          source_, fileMetadata_, properties_, fileDecryptor_);
      if (bloomFilterReader_ == nullptr) {
        throw ParquetException("Cannot create BloomFilterReader");
      }
    }
    return *bloomFilterReader_;
  }

  void setMetadata(std::shared_ptr<FileMetaData> metadata) {
    fileMetadata_ = std::move(metadata);
  }

  void preBuffer(
      const std::vector<int>& rowGroups,
      const std::vector<int>& columnIndices,
      const ::arrow::io::IOContext& ctx,
      const ::arrow::io::CacheOptions& options) {
    cachedSource_ = std::make_shared<::arrow::io::internal::ReadRangeCache>(
        source_, ctx, options);
    std::vector<::arrow::io::ReadRange> ranges;
    prebufferedColumnChunks_.clear();
    for (int row : rowGroups) {
      std::shared_ptr<Buffer>& colBitmap = prebufferedColumnChunks_[row];
      int numCols = fileMetadata_->numColumns();
      PARQUET_THROW_NOT_OK(
          ::arrow::AllocateEmptyBitmap(numCols, properties_.memoryPool())
              .Value(&colBitmap));
      for (int col : columnIndices) {
        ::arrow::bit_util::SetBit(colBitmap->mutable_data(), col);
        ranges.push_back(computeColumnChunkRange(
            fileMetadata_.get(), sourceSize_, row, col));
      }
    }
    PARQUET_THROW_NOT_OK(cachedSource_->Cache(ranges));
  }

  ::arrow::Future<> whenBuffered(
      const std::vector<int>& rowGroups,
      const std::vector<int>& columnIndices) const {
    if (!cachedSource_) {
      return ::arrow::Status::Invalid(
          "Must call PreBuffer before WhenBuffered");
    }
    std::vector<::arrow::io::ReadRange> ranges;
    for (int row : rowGroups) {
      for (int col : columnIndices) {
        ranges.push_back(computeColumnChunkRange(
            fileMetadata_.get(), sourceSize_, row, col));
      }
    }
    return cachedSource_->WaitFor(ranges);
  }

  // Metadata/footer parsing. Divided up to separate sync/async paths, and to.
  // Use exceptions for error handling (with the async path converting to.
  // Future/Status).

  void parseMetaData() {
    int64_t footerReadSize = getFooterReadSize();
    PARQUET_ASSIGN_OR_THROW(
        auto footerBuffer,
        source_->ReadAt(sourceSize_ - footerReadSize, footerReadSize));
    uint32_t metadataLen = parseFooterLength(footerBuffer, footerReadSize);
    int64_t metadataStart = sourceSize_ - kFooterSize - metadataLen;

    std::shared_ptr<::arrow::Buffer> metadataBuffer;
    if (footerReadSize >= (metadataLen + kFooterSize)) {
      metadataBuffer = ::arrow::SliceBuffer(
          footerBuffer,
          footerReadSize - metadataLen - kFooterSize,
          metadataLen);
    } else {
      PARQUET_ASSIGN_OR_THROW(
          metadataBuffer, source_->ReadAt(metadataStart, metadataLen));
    }

    // Parse the footer depending on encryption type.
    const bool isEncryptedFooter =
        memcmp(footerBuffer->data() + footerReadSize - 4, kParquetEMagic, 4) ==
        0;
    if (isEncryptedFooter) {
      // Encrypted file with Encrypted footer.
      const std::pair<int64_t, uint32_t> readSize =
          parseMetaDataOfEncryptedFileWithEncryptedFooter(
              metadataBuffer, metadataLen);
      // Read the actual footer.
      metadataStart = readSize.first;
      metadataLen = readSize.second;
      PARQUET_ASSIGN_OR_THROW(
          metadataBuffer, source_->ReadAt(metadataStart, metadataLen));
      // Fall through.
    }

    const uint32_t readMetadataLen =
        parseUnencryptedFileMetadata(metadataBuffer, metadataLen);
    auto fileDecryptionProperties =
        properties_.fileDecryptionProperties().get();
    if (isEncryptedFooter) {
      // Nothing else to do here.
      return;
    } else if (!fileMetadata_
                    ->isEncryptionAlgorithmSet()) { // Non encrypted file.
      if (fileDecryptionProperties != nullptr) {
        if (!fileDecryptionProperties->plaintextFilesAllowed()) {
          throw ParquetException(
              "Applying decryption properties on plaintext file");
        }
      }
    } else {
      // Encrypted file with plaintext footer mode.
      parseMetaDataOfEncryptedFileWithPlaintextFooter(
          fileDecryptionProperties,
          metadataBuffer,
          metadataLen,
          readMetadataLen);
    }
  }

  // Validate the source size and get the initial read size.
  int64_t getFooterReadSize() {
    if (sourceSize_ == 0) {
      throw ParquetInvalidOrCorruptedFileException(
          "Parquet file size is 0 bytes");
    } else if (sourceSize_ < kFooterSize) {
      throw ParquetInvalidOrCorruptedFileException(
          "Parquet file size is ",
          sourceSize_,
          " bytes, smaller than the minimum file footer (",
          kFooterSize,
          " bytes)");
    }
    return std::min(sourceSize_, kDefaultFooterReadSize);
  }

  // Validate the magic bytes and get the length of the full footer.
  uint32_t parseFooterLength(
      const std::shared_ptr<::arrow::Buffer>& footerBuffer,
      const int64_t footerReadSize) {
    // Check if all bytes are read. Check if last 4 bytes read have the magic.
    // Bits.
    if (footerBuffer->size() != footerReadSize ||
        (memcmp(footerBuffer->data() + footerReadSize - 4, kParquetMagic, 4) !=
             0 &&
         memcmp(footerBuffer->data() + footerReadSize - 4, kParquetEMagic, 4) !=
             0)) {
      throw ParquetInvalidOrCorruptedFileException(
          "Parquet magic bytes not found in footer. Either the file is corrupted or this "
          "is not a parquet file.");
    }
    // Both encrypted/unencrypted footers have the same footer length check.
    uint32_t metadataLen = ::arrow::util::SafeLoadAs<uint32_t>(
        reinterpret_cast<const uint8_t*>(footerBuffer->data()) +
        footerReadSize - kFooterSize);
    if (metadataLen > sourceSize_ - kFooterSize) {
      throw ParquetInvalidOrCorruptedFileException(
          "Parquet file size is ",
          sourceSize_,
          " bytes, smaller than the size reported by footer's (",
          metadataLen,
          "bytes)");
    }
    return metadataLen;
  }

  // Does not throw.
  ::arrow::Future<> parseMetaDataAsync() {
    int64_t footerReadSize;
    BEGIN_PARQUET_CATCH_EXCEPTIONS
    footerReadSize = getFooterReadSize();
    END_PARQUET_CATCH_EXCEPTIONS
    // Assumes this is kept alive externally.
    return source_->ReadAsync(sourceSize_ - footerReadSize, footerReadSize)
        .Then(
            [this, footerReadSize](
                const std::shared_ptr<::arrow::Buffer>& footerBuffer)
                -> ::arrow::Future<> {
              uint32_t metadataLen;
              BEGIN_PARQUET_CATCH_EXCEPTIONS
              metadataLen = parseFooterLength(footerBuffer, footerReadSize);
              END_PARQUET_CATCH_EXCEPTIONS
              int64_t metadataStart = sourceSize_ - kFooterSize - metadataLen;

              std::shared_ptr<::arrow::Buffer> metadataBuffer;
              if (footerReadSize >= (metadataLen + kFooterSize)) {
                metadataBuffer = ::arrow::SliceBuffer(
                    footerBuffer,
                    footerReadSize - metadataLen - kFooterSize,
                    metadataLen);
                return parseMaybeEncryptedMetaDataAsync(
                    footerBuffer,
                    std::move(metadataBuffer),
                    footerReadSize,
                    metadataLen);
              }
              return source_->ReadAsync(metadataStart, metadataLen)
                  .Then([this, footerBuffer, footerReadSize, metadataLen](
                            const std::shared_ptr<::arrow::Buffer>&
                                metadataBuffer) {
                    return parseMaybeEncryptedMetaDataAsync(
                        footerBuffer,
                        metadataBuffer,
                        footerReadSize,
                        metadataLen);
                  });
            });
  }

  // Continuation.
  ::arrow::Future<> parseMaybeEncryptedMetaDataAsync(
      std::shared_ptr<::arrow::Buffer> footerBuffer,
      std::shared_ptr<::arrow::Buffer> metadataBuffer,
      int64_t footerReadSize,
      uint32_t metadataLen) {
    // Parse the footer depending on encryption type.
    const bool isEncryptedFooter =
        memcmp(footerBuffer->data() + footerReadSize - 4, kParquetEMagic, 4) ==
        0;
    if (isEncryptedFooter) {
      // Encrypted file with Encrypted footer.
      std::pair<int64_t, uint32_t> readSize;
      BEGIN_PARQUET_CATCH_EXCEPTIONS
      readSize = parseMetaDataOfEncryptedFileWithEncryptedFooter(
          metadataBuffer, metadataLen);
      END_PARQUET_CATCH_EXCEPTIONS
      // Read the actual footer.
      int64_t metadataStart = readSize.first;
      metadataLen = readSize.second;
      return source_->ReadAsync(metadataStart, metadataLen)
          .Then([this, metadataLen, isEncryptedFooter](
                    const std::shared_ptr<::arrow::Buffer>& metadataBuffer) {
            // Continue and read the file footer.
            return parseMetaDataFinal(
                metadataBuffer, metadataLen, isEncryptedFooter);
          });
    }
    return parseMetaDataFinal(
        std::move(metadataBuffer), metadataLen, isEncryptedFooter);
  }

  // Continuation.
  ::arrow::Status parseMetaDataFinal(
      std::shared_ptr<::arrow::Buffer> metadataBuffer,
      uint32_t metadataLen,
      const bool isEncryptedFooter) {
    BEGIN_PARQUET_CATCH_EXCEPTIONS
    const uint32_t readMetadataLen =
        parseUnencryptedFileMetadata(metadataBuffer, metadataLen);
    auto fileDecryptionProperties =
        properties_.fileDecryptionProperties().get();
    if (isEncryptedFooter) {
      // Nothing else to do here.
      return ::arrow::Status::OK();
    } else if (!fileMetadata_
                    ->isEncryptionAlgorithmSet()) { // Non encrypted file.
      if (fileDecryptionProperties != nullptr) {
        if (!fileDecryptionProperties->plaintextFilesAllowed()) {
          throw ParquetException(
              "Applying decryption properties on plaintext file");
        }
      }
    } else {
      // Encrypted file with plaintext footer mode.
      parseMetaDataOfEncryptedFileWithPlaintextFooter(
          fileDecryptionProperties,
          metadataBuffer,
          metadataLen,
          readMetadataLen);
    }
    END_PARQUET_CATCH_EXCEPTIONS
    return ::arrow::Status::OK();
  }

 private:
  std::shared_ptr<ArrowInputFile> source_;
  std::shared_ptr<::arrow::io::internal::ReadRangeCache> cachedSource_;
  int64_t sourceSize_;
  std::shared_ptr<FileMetaData> fileMetadata_;
  ReaderProperties properties_;
  std::shared_ptr<PageIndexReader> pageIndexReader_;
  std::unique_ptr<BloomFilterReader> bloomFilterReader_;
  // Maps row group ordinal and prebuffer status of its column chunks in the.
  // Form of a bitmap buffer.
  std::unordered_map<int, std::shared_ptr<Buffer>> prebufferedColumnChunks_;
  std::shared_ptr<InternalFileDecryptor> fileDecryptor_;

  // \return The true length of the metadata in bytes.
  uint32_t parseUnencryptedFileMetadata(
      const std::shared_ptr<Buffer>& footerBuffer,
      const uint32_t metadataLen);

  std::string handleAadPrefix(
      FileDecryptionProperties* fileDecryptionProperties,
      EncryptionAlgorithm& algo);

  void parseMetaDataOfEncryptedFileWithPlaintextFooter(
      FileDecryptionProperties* fileDecryptionProperties,
      const std::shared_ptr<Buffer>& metadataBuffer,
      uint32_t metadataLen,
      uint32_t readMetadataLen);

  // \return The position and size of the actual footer.
  std::pair<int64_t, uint32_t> parseMetaDataOfEncryptedFileWithEncryptedFooter(
      const std::shared_ptr<Buffer>& cryptoMetadataBuffer,
      uint32_t footerLen);
};

uint32_t SerializedFile::parseUnencryptedFileMetadata(
    const std::shared_ptr<Buffer>& metadataBuffer,
    const uint32_t metadataLen) {
  if (metadataBuffer->size() != metadataLen) {
    throw ParquetException(
        "Failed reading metadata buffer (requested " +
        std::to_string(metadataLen) + " bytes but got " +
        std::to_string(metadataBuffer->size()) + " bytes)");
  }
  uint32_t readMetadataLen = metadataLen;
  // The encrypted read path falls through to here, so pass in the decryptor.
  fileMetadata_ = FileMetaData::make(
      metadataBuffer->data(), &readMetadataLen, properties_, fileDecryptor_);
  return readMetadataLen;
}

std::pair<int64_t, uint32_t>
SerializedFile::parseMetaDataOfEncryptedFileWithEncryptedFooter(
    const std::shared_ptr<::arrow::Buffer>& cryptoMetadataBuffer,
    // Both metadata & crypto metadata length.
    const uint32_t footerLen) {
  // Encryption with encrypted footer.
  // Check if the footer_buffer contains the entire metadata.
  if (cryptoMetadataBuffer->size() != footerLen) {
    throw ParquetException(
        "Failed reading encrypted metadata buffer (requested " +
        std::to_string(footerLen) + " bytes but got " +
        std::to_string(cryptoMetadataBuffer->size()) + " bytes)");
  }
  auto fileDecryptionProperties = properties_.fileDecryptionProperties().get();
  if (fileDecryptionProperties == nullptr) {
    throw ParquetException(
        "Could not read encrypted metadata, no decryption found in reader's properties");
  }
  uint32_t cryptoMetadataLen = footerLen;
  std::shared_ptr<FileCryptoMetaData> fileCryptoMetadata =
      FileCryptoMetaData::make(
          cryptoMetadataBuffer->data(), &cryptoMetadataLen);
  // Handle AAD prefix.
  EncryptionAlgorithm algo = fileCryptoMetadata->encryptionAlgorithm();
  std::string fileAad = handleAadPrefix(fileDecryptionProperties, algo);
  fileDecryptor_ = std::make_shared<InternalFileDecryptor>(
      fileDecryptionProperties,
      fileAad,
      algo.algorithm,
      fileCryptoMetadata->keyMetadata(),
      properties_.memoryPool());

  int64_t metadataOffset =
      sourceSize_ - kFooterSize - footerLen + cryptoMetadataLen;
  uint32_t metadataLen = footerLen - cryptoMetadataLen;
  return std::make_pair(metadataOffset, metadataLen);
}

void SerializedFile::parseMetaDataOfEncryptedFileWithPlaintextFooter(
    FileDecryptionProperties* fileDecryptionProperties,
    const std::shared_ptr<Buffer>& metadataBuffer,
    uint32_t metadataLen,
    uint32_t readMetadataLen) {
  // Providing decryption properties in plaintext footer mode is not mandatory,.
  // For example when reading by legacy reader.
  if (fileDecryptionProperties != nullptr) {
    EncryptionAlgorithm algo = fileMetadata_->encryptionAlgorithm();
    // Handle AAD prefix.
    std::string fileAad = handleAadPrefix(fileDecryptionProperties, algo);
    fileDecryptor_ = std::make_shared<InternalFileDecryptor>(
        fileDecryptionProperties,
        fileAad,
        algo.algorithm,
        fileMetadata_->footerSigningKeyMetadata(),
        properties_.memoryPool());
    // Set the InternalFileDecryptor in the metadata as well, as it's used.
    // For signature verification and for ColumnChunkMetaData creation.
    fileMetadata_->setFileDecryptor(fileDecryptor_);

    if (fileDecryptionProperties->checkPlaintextFooterIntegrity()) {
      if (metadataLen - readMetadataLen !=
          (encryption::kGcmTagLength + encryption::kNonceLength)) {
        throw ParquetInvalidOrCorruptedFileException(
            "Failed reading metadata for encryption signature (requested ",
            encryption::kGcmTagLength + encryption::kNonceLength,
            " bytes but have ",
            metadataLen - readMetadataLen,
            " bytes)");
      }

      if (!fileMetadata_->verifySignature(
              metadataBuffer->data() + readMetadataLen)) {
        throw ParquetInvalidOrCorruptedFileException(
            "Parquet crypto signature verification failed");
      }
    }
  }
}

std::string SerializedFile::handleAadPrefix(
    FileDecryptionProperties* fileDecryptionProperties,
    EncryptionAlgorithm& algo) {
  std::string aadPrefixInProperties = fileDecryptionProperties->aadPrefix();
  std::string aadPrefix = aadPrefixInProperties;
  bool fileHasAadPrefix = algo.aad.aadPrefix.empty() ? false : true;
  std::string aadPrefixInFile = algo.aad.aadPrefix;

  if (algo.aad.supplyAadPrefix && aadPrefixInProperties.empty()) {
    throw ParquetException(
        "AAD prefix used for file encryption, "
        "but not stored in file and not supplied "
        "in decryption properties");
  }

  if (fileHasAadPrefix) {
    if (!aadPrefixInProperties.empty()) {
      if (aadPrefixInProperties.compare(aadPrefixInFile) != 0) {
        throw ParquetException(
            "AAD Prefix in file and in properties "
            "is not the same");
      }
    }
    aadPrefix = aadPrefixInFile;
    std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier =
        fileDecryptionProperties->aadPrefixVerifier();
    if (aadPrefixVerifier != nullptr)
      aadPrefixVerifier->verify(aadPrefix);
  } else {
    if (!algo.aad.supplyAadPrefix && !aadPrefixInProperties.empty()) {
      throw ParquetException(
          "AAD Prefix set in decryption properties, but was not used "
          "for file encryption");
    }
    std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier =
        fileDecryptionProperties->aadPrefixVerifier();
    if (aadPrefixVerifier != nullptr) {
      throw ParquetException(
          "AAD Prefix Verifier is set, but AAD Prefix not found in file");
    }
  }
  return aadPrefix + algo.aad.aadFileUnique;
}

// ----------------------------------------------------------------------.
// ParquetFileReader public API.

ParquetFileReader::ParquetFileReader() {}

ParquetFileReader::~ParquetFileReader() {
  try {
    close();
  } catch (...) {
  }
}

// Open the file. If no metadata is passed, it is parsed from the footer of.
// The file.
std::unique_ptr<ParquetFileReader::Contents> ParquetFileReader::Contents::open(
    std::shared_ptr<ArrowInputFile> source,
    const ReaderProperties& props,
    std::shared_ptr<FileMetaData> metadata) {
  std::unique_ptr<ParquetFileReader::Contents> result(
      new SerializedFile(std::move(source), props));

  // Access private methods here, but otherwise unavailable.
  SerializedFile* file = static_cast<SerializedFile*>(result.get());

  if (metadata == nullptr) {
    // Validates magic bytes, parses metadata, and initializes the.
    // SchemaDescriptor.
    file->parseMetaData();
  } else {
    file->setMetadata(std::move(metadata));
  }

  return result;
}

::arrow::Future<std::unique_ptr<ParquetFileReader::Contents>>
ParquetFileReader::Contents::openAsync(
    std::shared_ptr<ArrowInputFile> source,
    const ReaderProperties& props,
    std::shared_ptr<FileMetaData> metadata) {
  BEGIN_PARQUET_CATCH_EXCEPTIONS
  std::unique_ptr<ParquetFileReader::Contents> result(
      new SerializedFile(std::move(source), props));
  SerializedFile* file = static_cast<SerializedFile*>(result.get());
  if (metadata == nullptr) {
    // TODO(ARROW-12259): workaround since we have Future<(move-only type)>.
    struct {
      ::arrow::Result<std::unique_ptr<ParquetFileReader::Contents>>
      operator()() {
        return std::move(result);
      }

      std::unique_ptr<ParquetFileReader::Contents> result;
    } Continuation;
    Continuation.result = std::move(result);
    return file->parseMetaDataAsync().Then(std::move(Continuation));
  } else {
    file->setMetadata(std::move(metadata));
    return ::arrow::Future<std::unique_ptr<ParquetFileReader::Contents>>::
        MakeFinished(std::move(result));
  }
  END_PARQUET_CATCH_EXCEPTIONS
}

std::unique_ptr<ParquetFileReader> ParquetFileReader::open(
    std::shared_ptr<::arrow::io::RandomAccessFile> source,
    const ReaderProperties& props,
    std::shared_ptr<FileMetaData> metadata) {
  auto contents =
      SerializedFile::open(std::move(source), props, std::move(metadata));
  std::unique_ptr<ParquetFileReader> result =
      std::make_unique<ParquetFileReader>();
  result->open(std::move(contents));
  return result;
}

std::unique_ptr<ParquetFileReader> ParquetFileReader::openFile(
    const std::string& path,
    bool memoryMap,
    const ReaderProperties& props,
    std::shared_ptr<FileMetaData> metadata) {
  std::shared_ptr<::arrow::io::RandomAccessFile> source;
  if (memoryMap) {
    PARQUET_ASSIGN_OR_THROW(
        source,
        ::arrow::io::MemoryMappedFile::Open(path, ::arrow::io::FileMode::READ));
  } else {
    PARQUET_ASSIGN_OR_THROW(
        source, ::arrow::io::ReadableFile::Open(path, props.memoryPool()));
  }

  return open(std::move(source), props, std::move(metadata));
}

::arrow::Future<std::unique_ptr<ParquetFileReader>>
ParquetFileReader::openAsync(
    std::shared_ptr<::arrow::io::RandomAccessFile> source,
    const ReaderProperties& props,
    std::shared_ptr<FileMetaData> metadata) {
  BEGIN_PARQUET_CATCH_EXCEPTIONS
  auto fut =
      SerializedFile::openAsync(std::move(source), props, std::move(metadata));
  // TODO(ARROW-12259): workaround since we have Future<(move-only type)>.
  auto completed = ::arrow::Future<std::unique_ptr<ParquetFileReader>>::Make();
  fut.AddCallback(
      [fut, completed](
          const ::arrow::Result<std::unique_ptr<ParquetFileReader::Contents>>&
              Contents) mutable {
        if (!Contents.ok()) {
          completed.MarkFinished(Contents.status());
          return;
        }
        std::unique_ptr<ParquetFileReader> result =
            std::make_unique<ParquetFileReader>();
        result->open(fut.MoveResult().MoveValueUnsafe());
        completed.MarkFinished(std::move(result));
      });
  return completed;
  END_PARQUET_CATCH_EXCEPTIONS
}

void ParquetFileReader::open(
    std::unique_ptr<ParquetFileReader::Contents> contents) {
  contents_ = std::move(contents);
}

void ParquetFileReader::close() {
  if (contents_) {
    contents_->close();
  }
}

std::shared_ptr<FileMetaData> ParquetFileReader::metadata() const {
  return contents_->metadata();
}

std::shared_ptr<PageIndexReader> ParquetFileReader::getPageIndexReader() {
  return contents_->getPageIndexReader();
}

BloomFilterReader& ParquetFileReader::getBloomFilterReader() {
  return contents_->getBloomFilterReader();
}

std::shared_ptr<RowGroupReader> ParquetFileReader::rowGroup(int i) {
  if (i >= metadata()->numRowGroups()) {
    std::stringstream ss;
    ss << "Trying to read row group " << i << " but file only has "
       << metadata()->numRowGroups() << " row groups";
    throw ParquetException(ss.str());
  }
  return contents_->getRowGroup(i);
}

void ParquetFileReader::preBuffer(
    const std::vector<int>& rowGroups,
    const std::vector<int>& columnIndices,
    const ::arrow::io::IOContext& ctx,
    const ::arrow::io::CacheOptions& options) {
  // Access private methods here.
  SerializedFile* file =
      ::arrow::internal::checked_cast<SerializedFile*>(contents_.get());
  file->preBuffer(rowGroups, columnIndices, ctx, options);
}

::arrow::Future<> ParquetFileReader::whenBuffered(
    const std::vector<int>& rowGroups,
    const std::vector<int>& columnIndices) const {
  // Access private methods here.
  SerializedFile* file =
      ::arrow::internal::checked_cast<SerializedFile*>(contents_.get());
  return file->whenBuffered(rowGroups, columnIndices);
}

// ----------------------------------------------------------------------.
// File metadata helpers.

std::shared_ptr<FileMetaData> readMetaData(
    const std::shared_ptr<::arrow::io::RandomAccessFile>& source) {
  return ParquetFileReader::open(source)->metadata();
}

// ----------------------------------------------------------------------.
// File Scanner for performance testing.

int64_t scanFileContents(
    std::vector<int> columns,
    const int32_t columnBatchSize,
    ParquetFileReader* reader) {
  std::vector<int16_t> repLevels(columnBatchSize);
  std::vector<int16_t> defLevels(columnBatchSize);

  int numColumns = static_cast<int>(columns.size());

  // Columns are not specified explicitly. Add all columns.
  if (columns.size() == 0) {
    numColumns = reader->metadata()->numColumns();
    columns.resize(numColumns);
    for (int i = 0; i < numColumns; i++) {
      columns[i] = i;
    }
  }
  if (numColumns == 0) {
    // If we still have no columns(none in file), return early. The remainder
    // of. Function expects there to be at least one column.
    return 0;
  }

  std::vector<int64_t> totalRows(numColumns, 0);

  for (int r = 0; r < reader->metadata()->numRowGroups(); ++r) {
    auto groupReader = reader->rowGroup(r);
    int col = 0;
    for (auto i : columns) {
      std::shared_ptr<ColumnReader> colReader = groupReader->column(i);
      size_t valueByteSize =
          getTypeByteSize(colReader->descr()->physicalType());
      std::vector<uint8_t> values(columnBatchSize * valueByteSize);

      int64_t valuesRead = 0;
      while (colReader->hasNext()) {
        int64_t levelsRead = scanAllValues(
            columnBatchSize,
            defLevels.data(),
            repLevels.data(),
            values.data(),
            &valuesRead,
            colReader.get());
        if (colReader->descr()->maxRepetitionLevel() > 0) {
          for (int64_t i = 0; i < levelsRead; i++) {
            if (repLevels[i] == 0) {
              totalRows[col]++;
            }
          }
        } else {
          totalRows[col] += levelsRead;
        }
      }
      col++;
    }
  }

  for (int i = 1; i < numColumns; ++i) {
    if (totalRows[0] != totalRows[i]) {
      throw ParquetException(
          "Parquet error: Total rows among columns do not match");
    }
  }

  return totalRows[0];
}

} // namespace facebook::velox::parquet::arrow
