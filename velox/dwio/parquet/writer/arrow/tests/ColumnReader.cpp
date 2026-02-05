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

#include "velox/dwio/parquet/writer/arrow/tests/ColumnReader.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/array/builder_binary.h"
#include "arrow/array/builder_dict.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/chunked_array.h"
#include "arrow/type.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/compression.h"
#include "arrow/util/crc32.h"
#include "arrow/util/int_util_overflow.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/common/LevelComparison.h"
#include "velox/dwio/parquet/common/LevelConversion.h"
#include "velox/dwio/parquet/writer/arrow/ColumnPage.h"
#include "velox/dwio/parquet/writer/arrow/Encoding.h"
#include "velox/dwio/parquet/writer/arrow/EncryptionInternal.h"
#include "velox/dwio/parquet/writer/arrow/FileDecryptorInternal.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"
#include "velox/dwio/parquet/writer/arrow/Statistics.h"
#include "velox/dwio/parquet/writer/arrow/ThriftInternal.h"

using arrow::MemoryPool;
using arrow::internal::AddWithOverflow;
using arrow::internal::checked_cast;
using arrow::internal::MultiplyWithOverflow;

namespace bit_util = arrow::bit_util;

namespace facebook::velox::parquet::arrow {

fmt::underlying_t<Type::type> formatAs(Type::type type) {
  return fmt::underlying(type);
}

namespace {

// The minimum number of repetition/definition levels to decode at a time, for.
// Better vectorized performance when doing many smaller record reads.
constexpr int64_t kMinLevelBatchSize = 1024;

// Batch size for reading and throwing away values during skip.
// Both RecordReader and the ColumnReader use this for skipping.
constexpr int64_t kSkipScratchBatchSize = 1024;

inline bool hasSpacedValues(const ColumnDescriptor* descr) {
  if (descr->maxRepetitionLevel() > 0) {
    // Repeated+flat case.
    return !descr->schemaNode()->isRequired();
  } else {
    // Non-repeated+nested case.
    // Find if a node forces nulls in the lowest level along the hierarchy.
    const schema::Node* Node = descr->schemaNode().get();
    while (Node) {
      if (Node->isOptional()) {
        return true;
      }
      Node = Node->parent();
    }
    return false;
  }
}

// Throws exception if number_decoded does not match expected.
inline void checkNumberDecoded(int64_t numberDecoded, int64_t expected) {
  if (ARROW_PREDICT_FALSE(numberDecoded != expected)) {
    ParquetException::eofException(
        "Decoded values " + std::to_string(numberDecoded) +
        " does not match expected " + std::to_string(expected));
  }
}
} // namespace

LevelDecoder::LevelDecoder() : numValuesRemaining_(0) {}

LevelDecoder::~LevelDecoder() {}

int LevelDecoder::setData(
    Encoding::type encoding,
    int16_t maxLevel,
    int numBufferedValues,
    const uint8_t* data,
    int32_t dataSize) {
  maxLevel_ = maxLevel;
  int32_t numBytes = 0;
  encoding_ = encoding;
  numValuesRemaining_ = numBufferedValues;
  bitWidth_ = ::arrow::bit_util::Log2(maxLevel + 1);
  switch (encoding) {
    case Encoding::kRle: {
      if (dataSize < 4) {
        throw ParquetException("Received invalid levels (corrupt data page?)");
      }
      numBytes = ::arrow::util::SafeLoadAs<int32_t>(data);
      if (numBytes < 0 || numBytes > dataSize - 4) {
        throw ParquetException(
            "Received invalid number of bytes (corrupt data page?)");
      }
      const uint8_t* decoderData = data + 4;
      if (!rleDecoder_) {
        rleDecoder_ =
            std::make_unique<RleDecoder>(decoderData, numBytes, bitWidth_);
      } else {
        rleDecoder_->Reset(decoderData, numBytes, bitWidth_);
      }
      return 4 + numBytes;
    }
    case Encoding::kBitPacked: {
      int32_t numBits = 0;
      if (MultiplyWithOverflow(numBufferedValues, bitWidth_, &numBits)) {
        throw ParquetException(
            "Number of buffered values too large (corrupt data page?)");
      }
      numBytes = static_cast<int32_t>(::arrow::bit_util::BytesForBits(numBits));
      if (numBytes < 0 || numBytes > dataSize - 4) {
        throw ParquetException(
            "Received invalid number of bytes (corrupt data page?)");
      }
      if (!bitPackedDecoder_) {
        bitPackedDecoder_ = std::make_unique<BitReader>(data, numBytes);
      } else {
        bitPackedDecoder_->Reset(data, numBytes);
      }
      return numBytes;
    }
    default:
      throw ParquetException("Unknown encoding type for levels.");
  }
  return -1;
}

void LevelDecoder::setDataV2(
    int32_t numBytes,
    int16_t maxLevel,
    int numBufferedValues,
    const uint8_t* data) {
  maxLevel_ = maxLevel;
  // Repetition and definition levels always uses RLE encoding.
  // In the DataPageV2 format.
  if (numBytes < 0) {
    throw ParquetException("Invalid page header (corrupt data page?)");
  }
  encoding_ = Encoding::kRle;
  numValuesRemaining_ = numBufferedValues;
  bitWidth_ = ::arrow::bit_util::Log2(maxLevel + 1);

  if (!rleDecoder_) {
    rleDecoder_ = std::make_unique<RleDecoder>(data, numBytes, bitWidth_);
  } else {
    rleDecoder_->Reset(data, numBytes, bitWidth_);
  }
}

int LevelDecoder::decode(int batchSize, int16_t* levels) {
  int numDecoded = 0;

  int numValues = std::min(numValuesRemaining_, batchSize);
  if (encoding_ == Encoding::kRle) {
    numDecoded = rleDecoder_->GetBatch(levels, numValues);
  } else {
    numDecoded = bitPackedDecoder_->GetBatch(bitWidth_, levels, numValues);
  }
  if (numDecoded > 0) {
    MinMax minMax = FindMinMax(levels, numDecoded);
    if (ARROW_PREDICT_FALSE(minMax.min < 0 || minMax.max > maxLevel_)) {
      std::stringstream ss;
      ss << "Malformed levels. min: " << minMax.min << " max: " << minMax.max
         << " out of range.  Max Level: " << maxLevel_;
      throw ParquetException(ss.str());
    }
  }
  numValuesRemaining_ -= numDecoded;
  return numDecoded;
}

namespace {

// Extracts encoded statistics from V1 and V2 data page headers.
template <typename H>
EncodedStatistics extractStatsFromHeader(const H& header) {
  EncodedStatistics pageStatistics;
  if (!header.__isset.statistics) {
    return pageStatistics;
  }
  const facebook::velox::parquet::thrift::Statistics& stats = header.statistics;
  // Use the new V2 min-max statistics over the former one if it is filled.
  if (stats.__isset.max_value || stats.__isset.min_value) {
    // TODO: check if the column_order is TYPE_DEFINED_ORDER.
    if (stats.__isset.max_value) {
      pageStatistics.setMax(stats.max_value);
    }
    if (stats.__isset.min_value) {
      pageStatistics.setMin(stats.min_value);
    }
  } else if (stats.__isset.max || stats.__isset.min) {
    // TODO: check created_by to see if it is corrupted for some types.
    // TODO: check if the sort_order is SIGNED.
    if (stats.__isset.max) {
      pageStatistics.setMax(stats.max);
    }
    if (stats.__isset.min) {
      pageStatistics.setMin(stats.min);
    }
  }
  if (stats.__isset.null_count) {
    pageStatistics.setNullCount(stats.null_count);
  }
  if (stats.__isset.distinct_count) {
    pageStatistics.setDistinctCount(stats.distinct_count);
  }
  return pageStatistics;
}

void checkNumValuesInHeader(int numValues) {
  if (numValues < 0) {
    throw ParquetException("Invalid page header (negative number of values)");
  }
}

// ----------------------------------------------------------------------.
// SerializedPageReader deserializes Thrift metadata and pages that have been.
// Assembled in a serialized stream for storing in a Parquet files.

// This subclass delimits pages appearing in a serialized stream, each preceded.
// By a serialized Thrift facebook::velox::parquet::thrift::PageHeader.
// Indicating the type of each page and the page metadata.
class SerializedPageReader : public PageReader {
 public:
  SerializedPageReader(
      std::shared_ptr<ArrowInputStream> stream,
      int64_t totalNumValues,
      Compression::type Codec,
      const ReaderProperties& properties,
      const CryptoContext* cryptoCtx,
      bool alwaysCompressed)
      : properties_(properties),
        stream_(std::move(stream)),
        decompressionBuffer_(allocateBuffer(properties_.memoryPool(), 0)),
        pageOrdinal_(0),
        seenNumValues_(0),
        totalNumValues_(totalNumValues),
        decryptionBuffer_(allocateBuffer(properties_.memoryPool(), 0)) {
    if (cryptoCtx != nullptr) {
      cryptoCtx_ = *cryptoCtx;
      initDecryption();
    }
    maxPageHeaderSize_ = kDefaultMaxPageHeaderSize;
    decompressor_ = getCodec(Codec);
    alwaysCompressed_ = alwaysCompressed;
  }

  // Implement the PageReader interface.
  //
  // The returned Page contains references that aren't guaranteed to live.
  // Beyond the next call to NextPage(). SerializedPageReader reuses the.
  // Decryption and decompression buffers internally, so if NextPage() is.
  // Called then the content of previous page might be invalidated.
  std::shared_ptr<Page> nextPage() override;

  void setMaxPageHeaderSize(uint32_t size) override {
    maxPageHeaderSize_ = size;
  }

 private:
  void updateDecryption(
      const std::shared_ptr<Decryptor>& Decryptor,
      int8_t moduleType,
      std::string* pageAad);

  void initDecryption();

  std::shared_ptr<Buffer> decompressIfNeeded(
      std::shared_ptr<Buffer> pageBuffer,
      int compressedLen,
      int uncompressedLen,
      int levelsByteLen = 0);

  // Returns true for non-data pages, and if we should skip based on.
  // Data_page_filter_. Performs basic checks on values in the page header.
  // Fills in data_page_statistics.
  bool shouldSkipPage(EncodedStatistics* dataPageStatistics);

  const ReaderProperties properties_;
  std::shared_ptr<ArrowInputStream> stream_;

  facebook::velox::parquet::thrift::PageHeader currentPageHeader_;
  std::shared_ptr<Page> currentPage_;

  // Compression codec to use.
  std::unique_ptr<util::Codec> decompressor_;
  std::shared_ptr<ResizableBuffer> decompressionBuffer_;

  bool alwaysCompressed_;

  // The fields below are used for calculation of AAD (additional authenticated.
  // Data) suffix which is part of the Parquet Modular Encryption. The AAD.
  // Suffix for a parquet module is built internally by concatenating different.
  // Parts some of which include the row group ordinal, column ordinal and page.
  // Ordinal. Please refer to the encryption specification for more details:
  // https://github.com/apache/parquet-format/blob/encryption/Encryption.md#44-additional-authenticated-data

  // The ordinal fields in the context below are used for AAD suffix.
  // Calculation.
  CryptoContext cryptoCtx_;
  int32_t pageOrdinal_; // page ordinal does not count the dictionary page

  // Maximum allowed page size.
  uint32_t maxPageHeaderSize_;

  // Number of values read in data pages so far.
  int64_t seenNumValues_;

  // Number of values in all the data pages.
  int64_t totalNumValues_;

  // Data_page_aad_ and data_page_header_aad_ contain the AAD for data page and.
  // Data page header in a single column respectively. While calculating AAD
  // for. Different pages in a single column the pages AAD is updated by only
  // the. Page ordinal.
  std::string dataPageAad_;
  std::string dataPageHeaderAad_;
  // Encryption.
  std::shared_ptr<ResizableBuffer> decryptionBuffer_;
};

void SerializedPageReader::initDecryption() {
  // Prepare the AAD for quick update later.
  if (cryptoCtx_.dataDecryptor != nullptr) {
    VELOX_DCHECK(!cryptoCtx_.dataDecryptor->fileAad().empty());
    dataPageAad_ = encryption::createModuleAad(
        cryptoCtx_.dataDecryptor->fileAad(),
        encryption::kDataPage,
        cryptoCtx_.rowGroupOrdinal,
        cryptoCtx_.columnOrdinal,
        kNonPageOrdinal);
  }
  if (cryptoCtx_.metaDecryptor != nullptr) {
    VELOX_DCHECK(!cryptoCtx_.metaDecryptor->fileAad().empty());
    dataPageHeaderAad_ = encryption::createModuleAad(
        cryptoCtx_.metaDecryptor->fileAad(),
        encryption::kDataPageHeader,
        cryptoCtx_.rowGroupOrdinal,
        cryptoCtx_.columnOrdinal,
        kNonPageOrdinal);
  }
}

void SerializedPageReader::updateDecryption(
    const std::shared_ptr<Decryptor>& Decryptor,
    int8_t moduleType,
    std::string* pageAad) {
  VELOX_DCHECK_NOT_NULL(Decryptor);
  if (cryptoCtx_.startDecryptWithDictionaryPage) {
    std::string aad = encryption::createModuleAad(
        Decryptor->fileAad(),
        moduleType,
        cryptoCtx_.rowGroupOrdinal,
        cryptoCtx_.columnOrdinal,
        kNonPageOrdinal);
    Decryptor->updateAad(aad);
  } else {
    encryption::quickUpdatePageAad(pageOrdinal_, pageAad);
    Decryptor->updateAad(*pageAad);
  }
}

bool SerializedPageReader::shouldSkipPage(
    EncodedStatistics* dataPageStatistics) {
  const PageType::type pageType = loadenumSafe(&currentPageHeader_.type);
  if (pageType == PageType::kDataPage) {
    const facebook::velox::parquet::thrift::DataPageHeader& header =
        currentPageHeader_.data_page_header;
    checkNumValuesInHeader(header.num_values);
    *dataPageStatistics = extractStatsFromHeader(header);
    seenNumValues_ += header.num_values;
    if (dataPageFilter_) {
      const EncodedStatistics* filterStatistics =
          dataPageStatistics->isSet() ? dataPageStatistics : nullptr;
      dataPageStats dataPageStats(
          filterStatistics,
          header.num_values,
          /*num_rows=*/std::nullopt);
      if (dataPageFilter_(dataPageStats)) {
        return true;
      }
    }
  } else if (pageType == PageType::kDataPageV2) {
    const facebook::velox::parquet::thrift::DataPageHeaderV2& header =
        currentPageHeader_.data_page_header_v2;
    checkNumValuesInHeader(header.num_values);
    if (header.num_rows < 0) {
      throw ParquetException("Invalid page header (negative number of rows)");
    }
    if (header.definition_levels_byte_length < 0 ||
        header.repetition_levels_byte_length < 0) {
      throw ParquetException(
          "Invalid page header (negative levels byte length)");
    }
    *dataPageStatistics = extractStatsFromHeader(header);
    seenNumValues_ += header.num_values;
    if (dataPageFilter_) {
      const EncodedStatistics* filterStatistics =
          dataPageStatistics->isSet() ? dataPageStatistics : nullptr;
      dataPageStats dataPageStats(
          filterStatistics, header.num_values, header.num_rows);
      if (dataPageFilter_(dataPageStats)) {
        return true;
      }
    }
  } else if (pageType == PageType::kDictionaryPage) {
    const facebook::velox::parquet::thrift::DictionaryPageHeader& dictHeader =
        currentPageHeader_.dictionary_page_header;
    checkNumValuesInHeader(dictHeader.num_values);
  } else {
    // We don't know what this page type is. We're allowed to skip non-data.
    // Pages.
    return true;
  }
  return false;
}

std::shared_ptr<Page> SerializedPageReader::nextPage() {
  ThriftDeserializer deserializer(properties_);

  // Loop here because there may be unhandled page types that we skip until.
  // Finding a page that we do know what to do with.
  while (seenNumValues_ < totalNumValues_) {
    uint32_t headerSize = 0;
    uint32_t allowedPageSize = kDefaultPageHeaderSize;

    // Page headers can be very large because of page statistics.
    // We try to deserialize a larger buffer progressively.
    // Until a maximum allowed header limit.
    while (true) {
      PARQUET_ASSIGN_OR_THROW(auto view, stream_->Peek(allowedPageSize));
      if (view.size() == 0) {
        return std::shared_ptr<Page>(nullptr);
      }

      // This gets used, then set by DeserializeThriftMsg.
      headerSize = static_cast<uint32_t>(view.size());
      try {
        if (cryptoCtx_.metaDecryptor != nullptr) {
          updateDecryption(
              cryptoCtx_.metaDecryptor,
              encryption::kDictionaryPageHeader,
              &dataPageHeaderAad_);
        }
        // Reset current page header to avoid unclearing the __isset flag.
        currentPageHeader_ = facebook::velox::parquet::thrift::PageHeader();
        deserializer.deserializeMessage(
            reinterpret_cast<const uint8_t*>(view.data()),
            &headerSize,
            &currentPageHeader_,
            cryptoCtx_.metaDecryptor);
        break;
      } catch (std::exception& e) {
        // Failed to deserialize. Double the allowed page header size and try.
        // Again.
        std::stringstream ss;
        ss << e.what();
        allowedPageSize *= 2;
        if (allowedPageSize > maxPageHeaderSize_) {
          ss << "Deserializing page header failed.\n";
          throw ParquetException(ss.str());
        }
      }
    }
    // Advance the stream offset.
    PARQUET_THROW_NOT_OK(stream_->Advance(headerSize));

    int compressedLen = currentPageHeader_.compressed_page_size;
    int uncompressedLen = currentPageHeader_.uncompressed_page_size;
    if (compressedLen < 0 || uncompressedLen < 0) {
      throw ParquetException("Invalid page header");
    }

    EncodedStatistics dataPageStatistics;
    if (shouldSkipPage(&dataPageStatistics)) {
      PARQUET_THROW_NOT_OK(stream_->Advance(compressedLen));
      continue;
    }

    if (cryptoCtx_.dataDecryptor != nullptr) {
      updateDecryption(
          cryptoCtx_.dataDecryptor, encryption::kDictionaryPage, &dataPageAad_);
    }

    // Read the compressed data page.
    PARQUET_ASSIGN_OR_THROW(auto pageBuffer, stream_->Read(compressedLen));
    if (pageBuffer->size() != compressedLen) {
      std::stringstream ss;
      ss << "Page was smaller (" << pageBuffer->size() << ") than expected ("
         << compressedLen << ")";
      ParquetException::eofException(ss.str());
    }

    const PageType::type pageType = loadenumSafe(&currentPageHeader_.type);

    if (properties_.pageChecksumVerification() &&
        currentPageHeader_.__isset.crc && pageCanUseChecksum(pageType)) {
      // Verify crc.
      uint32_t checksum = ::arrow::internal::crc32(
          /* prev */ 0, pageBuffer->data(), compressedLen);
      if (static_cast<int32_t>(checksum) != currentPageHeader_.crc) {
        throw ParquetException(
            "could not verify page integrity, CRC checksum verification failed for "
            "page_ordinal " +
            std::to_string(pageOrdinal_));
      }
    }

    // Decrypt it if we need to.
    if (cryptoCtx_.dataDecryptor != nullptr) {
      PARQUET_THROW_NOT_OK(decryptionBuffer_->Resize(
          compressedLen - cryptoCtx_.dataDecryptor->ciphertextSizeDelta(),
          /*shrink_to_fit=*/false));
      compressedLen = cryptoCtx_.dataDecryptor->decrypt(
          pageBuffer->data(), compressedLen, decryptionBuffer_->mutable_data());

      pageBuffer = decryptionBuffer_;
    }

    if (pageType == PageType::kDictionaryPage) {
      cryptoCtx_.startDecryptWithDictionaryPage = false;
      const facebook::velox::parquet::thrift::DictionaryPageHeader& dictHeader =
          currentPageHeader_.dictionary_page_header;
      bool isSorted =
          dictHeader.__isset.is_sorted ? dictHeader.is_sorted : false;

      pageBuffer = decompressIfNeeded(
          std::move(pageBuffer), compressedLen, uncompressedLen);

      return std::make_shared<DictionaryPage>(
          pageBuffer,
          dictHeader.num_values,
          loadenumSafe(&dictHeader.encoding),
          isSorted);
    } else if (pageType == PageType::kDataPage) {
      ++pageOrdinal_;
      const facebook::velox::parquet::thrift::DataPageHeader& header =
          currentPageHeader_.data_page_header;
      pageBuffer = decompressIfNeeded(
          std::move(pageBuffer), compressedLen, uncompressedLen);

      return std::make_shared<DataPageV1>(
          pageBuffer,
          header.num_values,
          loadenumSafe(&header.encoding),
          loadenumSafe(&header.definition_level_encoding),
          loadenumSafe(&header.repetition_level_encoding),
          uncompressedLen,
          dataPageStatistics);
    } else if (pageType == PageType::kDataPageV2) {
      ++pageOrdinal_;
      const facebook::velox::parquet::thrift::DataPageHeaderV2& header =
          currentPageHeader_.data_page_header_v2;

      // Arrow prior to 3.0.0 set is_compressed to false but still compressed.
      bool isCompressed =
          (header.__isset.is_compressed ? header.is_compressed : false) ||
          alwaysCompressed_;

      // Uncompress if needed.
      int levelsByteLen;
      if (AddWithOverflow(
              header.definition_levels_byte_length,
              header.repetition_levels_byte_length,
              &levelsByteLen)) {
        throw ParquetException("Levels size too large (corrupt file?)");
      }
      // DecompressIfNeeded doesn't take `is_compressed` into account as.
      // It's page type-agnostic.
      if (isCompressed) {
        pageBuffer = decompressIfNeeded(
            std::move(pageBuffer),
            compressedLen,
            uncompressedLen,
            levelsByteLen);
      }

      return std::make_shared<DataPageV2>(
          pageBuffer,
          header.num_values,
          header.num_nulls,
          header.num_rows,
          loadenumSafe(&header.encoding),
          header.definition_levels_byte_length,
          header.repetition_levels_byte_length,
          uncompressedLen,
          isCompressed,
          dataPageStatistics);
    } else {
      throw ParquetException(
          "Internal error, we have already skipped non-data pages in ShouldSkipPage()");
    }
  }
  return std::shared_ptr<Page>(nullptr);
}

std::shared_ptr<Buffer> SerializedPageReader::decompressIfNeeded(
    std::shared_ptr<Buffer> pageBuffer,
    int compressedLen,
    int uncompressedLen,
    int levelsByteLen) {
  if (decompressor_ == nullptr) {
    return pageBuffer;
  }
  if (compressedLen < levelsByteLen || uncompressedLen < levelsByteLen) {
    throw ParquetException("Invalid page header");
  }

  // Grow the uncompressed buffer if we need to.
  PARQUET_THROW_NOT_OK(
      decompressionBuffer_->Resize(uncompressedLen, /*shrink_to_fit=*/false));

  if (levelsByteLen > 0) {
    // First copy the levels as-is.
    uint8_t* decompressed = decompressionBuffer_->mutable_data();
    memcpy(decompressed, pageBuffer->data(), levelsByteLen);
  }

  // Decompress the values.
  PARQUET_THROW_NOT_OK(decompressor_->decompress(
      compressedLen - levelsByteLen,
      pageBuffer->data() + levelsByteLen,
      uncompressedLen - levelsByteLen,
      decompressionBuffer_->mutable_data() + levelsByteLen));

  return decompressionBuffer_;
}

} // namespace

std::unique_ptr<PageReader> PageReader::open(
    std::shared_ptr<ArrowInputStream> stream,
    int64_t totalNumValues,
    Compression::type Codec,
    const ReaderProperties& properties,
    bool alwaysCompressed,
    const CryptoContext* ctx) {
  return std::unique_ptr<PageReader>(new SerializedPageReader(
      std::move(stream),
      totalNumValues,
      Codec,
      properties,
      ctx,
      alwaysCompressed));
}

std::unique_ptr<PageReader> PageReader::open(
    std::shared_ptr<ArrowInputStream> stream,
    int64_t totalNumValues,
    Compression::type Codec,
    bool alwaysCompressed,
    ::arrow::MemoryPool* pool,
    const CryptoContext* ctx) {
  return std::unique_ptr<PageReader>(new SerializedPageReader(
      std::move(stream),
      totalNumValues,
      Codec,
      ReaderProperties(pool),
      ctx,
      alwaysCompressed));
}

namespace {

// ----------------------------------------------------------------------.
// Impl base class for TypedColumnReader and RecordReader.

// PLAIN_DICTIONARY is deprecated but used to be used as a dictionary index.
// Encoding.
static bool isDictionaryIndexEncoding(const Encoding::type& e) {
  return e == Encoding::kRleDictionary || e == Encoding::kPlainDictionary;
}

template <typename DType>
class ColumnReaderImplBase {
 public:
  using T = typename DType::CType;

  ColumnReaderImplBase(const ColumnDescriptor* descr, ::arrow::MemoryPool* pool)
      : descr_(descr),
        maxDefLevel_(descr->maxDefinitionLevel()),
        maxRepLevel_(descr->maxRepetitionLevel()),
        numBufferedValues_(0),
        numDecodedValues_(0),
        pool_(pool),
        currentDecoder_(nullptr),
        currentEncoding_(Encoding::kUnknown) {}

  virtual ~ColumnReaderImplBase() = default;

 protected:
  // Read up to batch_size values from the current data page into the.
  // Pre-allocated memory T*.
  //
  // @returns: the number of values read into the out buffer.
  int64_t readValues(int64_t batchSize, T* out) {
    int64_t numDecoded =
        currentDecoder_->decode(out, static_cast<int>(batchSize));
    return numDecoded;
  }

  // Read up to batch_size values from the current data page into the.
  // Pre-allocated memory T*, leaving spaces for null entries according.
  // To the def_levels.
  //
  // @returns: the number of values read into the out buffer.
  int64_t readValuesSpaced(
      int64_t batchSize,
      T* out,
      int64_t nullCount,
      uint8_t* validBits,
      int64_t validBitsOffset) {
    return currentDecoder_->decodeSpaced(
        out,
        static_cast<int>(batchSize),
        static_cast<int>(nullCount),
        validBits,
        validBitsOffset);
  }

  // Read multiple definition levels into preallocated memory.
  //
  // Returns the number of decoded definition levels.
  int64_t readDefinitionLevels(int64_t batchSize, int16_t* levels) {
    if (maxDefLevel_ == 0) {
      return 0;
    }
    return definitionLevelDecoder_.decode(static_cast<int>(batchSize), levels);
  }

  bool hasNextInternal() {
    // Either there is no data page available yet, or the data page has been.
    // Exhausted.
    if (numBufferedValues_ == 0 || numDecodedValues_ == numBufferedValues_) {
      if (!readNewPage() || numBufferedValues_ == 0) {
        return false;
      }
    }
    return true;
  }

  // Read multiple repetition levels into preallocated memory.
  // Returns the number of decoded repetition levels.
  int64_t readRepetitionLevels(int64_t batchSize, int16_t* levels) {
    if (maxRepLevel_ == 0) {
      return 0;
    }
    return repetitionLevelDecoder_.decode(static_cast<int>(batchSize), levels);
  }

  // Advance to the next data page.
  bool readNewPage() {
    // Loop until we find the next data page.
    while (true) {
      currentPage_ = pager_->nextPage();
      if (!currentPage_) {
        // EOS.
        return false;
      }

      if (currentPage_->type() == PageType::kDictionaryPage) {
        configureDictionary(
            static_cast<const DictionaryPage*>(currentPage_.get()));
        continue;
      } else if (currentPage_->type() == PageType::kDataPage) {
        const auto page = std::static_pointer_cast<DataPageV1>(currentPage_);
        const int64_t levelsByteSize = initializeLevelDecoders(
            *page,
            page->repetitionLevelEncoding(),
            page->definitionLevelEncoding());
        initializeDataDecoder(*page, levelsByteSize);
        return true;
      } else if (currentPage_->type() == PageType::kDataPageV2) {
        const auto page = std::static_pointer_cast<DataPageV2>(currentPage_);
        int64_t levelsByteSize = initializeLevelDecodersV2(*page);
        initializeDataDecoder(*page, levelsByteSize);
        return true;
      } else {
        // We don't know what this page type is. We're allowed to skip non-data.
        // Pages.
        continue;
      }
    }
    return true;
  }

  void configureDictionary(const DictionaryPage* page) {
    int encoding = static_cast<int>(page->encoding());
    if (page->encoding() == Encoding::kPlainDictionary ||
        page->encoding() == Encoding::kPlain) {
      encoding = static_cast<int>(Encoding::kRleDictionary);
    }

    auto it = decoders_.find(encoding);
    if (it != decoders_.end()) {
      throw ParquetException("Column cannot have more than one dictionary.");
    }

    if (page->encoding() == Encoding::kPlainDictionary ||
        page->encoding() == Encoding::kPlain) {
      auto dictionary = makeTypedDecoder<DType>(Encoding::kPlain, descr_);
      dictionary->setData(page->numValues(), page->data(), page->size());

      // The dictionary is fully decoded during DictionaryDecoder::Init, so the.
      // DictionaryPage buffer is no longer required after this step.
      //
      // TODO(wesm): investigate whether this all-or-nothing decoding of the.
      // Dictionary makes sense and whether performance can be improved.

      std::unique_ptr<DictDecoder<DType>> decoder =
          makeDictDecoder<DType>(descr_, pool_);
      decoder->setDict(dictionary.get());
      decoders_[encoding] = std::unique_ptr<DecoderType>(
          dynamic_cast<DecoderType*>(decoder.release()));
    } else {
      ParquetException::NYI(
          "only plain dictionary encoding has been implemented");
    }

    newDictionary_ = true;
    currentDecoder_ = decoders_[encoding].get();
    VELOX_DCHECK(currentDecoder_);
  }

  // Initialize repetition and definition level decoders on the next data page.

  // If the data page includes repetition and definition levels, we.
  // Initialize the level decoders and return the number of encoded level bytes.
  // The return value helps determine the number of bytes in the encoded data.
  int64_t initializeLevelDecoders(
      const DataPage& page,
      Encoding::type repetitionLevelEncoding,
      Encoding::type definitionLevelEncoding) {
    // Read a data page.
    numBufferedValues_ = page.numValues();

    // Have not decoded any values from the data page yet.
    numDecodedValues_ = 0;

    const uint8_t* buffer = page.data();
    int32_t levelsByteSize = 0;
    int32_t maxSize = page.size();

    // Data page Layout: Repetition Levels - Definition Levels - encoded values.
    // Levels are encoded as rle or bit-packed.
    // Init repetition levels.
    if (maxRepLevel_ > 0) {
      int32_t repLevelsBytes = repetitionLevelDecoder_.setData(
          repetitionLevelEncoding,
          maxRepLevel_,
          static_cast<int>(numBufferedValues_),
          buffer,
          maxSize);
      buffer += repLevelsBytes;
      levelsByteSize += repLevelsBytes;
      maxSize -= repLevelsBytes;
    }
    // TODO figure a way to set max_def_level_ to 0.
    // If the initial value is invalid.

    // Init definition levels.
    if (maxDefLevel_ > 0) {
      int32_t defLevelsBytes = definitionLevelDecoder_.setData(
          definitionLevelEncoding,
          maxDefLevel_,
          static_cast<int>(numBufferedValues_),
          buffer,
          maxSize);
      levelsByteSize += defLevelsBytes;
      maxSize -= defLevelsBytes;
    }

    return levelsByteSize;
  }

  int64_t initializeLevelDecodersV2(const DataPageV2& page) {
    // Read a data page.
    numBufferedValues_ = page.numValues();

    // Have not decoded any values from the data page yet.
    numDecodedValues_ = 0;
    const uint8_t* buffer = page.data();

    const int64_t totalLevelsLength =
        static_cast<int64_t>(page.repetitionLevelsByteLength()) +
        page.definitionLevelsByteLength();

    if (totalLevelsLength > page.size()) {
      throw ParquetException(
          "Data page too small for levels (corrupt header?)");
    }

    if (maxRepLevel_ > 0) {
      repetitionLevelDecoder_.setDataV2(
          page.repetitionLevelsByteLength(),
          maxRepLevel_,
          static_cast<int>(numBufferedValues_),
          buffer);
    }
    // ARROW-17453: Even if max_rep_level_ is 0, there may still be.
    // Repetition level bytes written and/or reported in the header by.
    // some writers (e.g. Athena)
    buffer += page.repetitionLevelsByteLength();

    if (maxDefLevel_ > 0) {
      definitionLevelDecoder_.setDataV2(
          page.definitionLevelsByteLength(),
          maxDefLevel_,
          static_cast<int>(numBufferedValues_),
          buffer);
    }

    return totalLevelsLength;
  }

  // Get a decoder object for this page or create a new decoder if this is the.
  // First page with this encoding.
  void initializeDataDecoder(const DataPage& page, int64_t levelsByteSize) {
    const uint8_t* buffer = page.data() + levelsByteSize;
    const int64_t dataSize = page.size() - levelsByteSize;

    if (dataSize < 0) {
      throw ParquetException("Page smaller than size of encoded levels");
    }

    Encoding::type encoding = page.encoding();

    if (isDictionaryIndexEncoding(encoding)) {
      encoding = Encoding::kRleDictionary;
    }

    auto it = decoders_.find(static_cast<int>(encoding));
    if (it != decoders_.end()) {
      VELOX_DCHECK_NOT_NULL(it->second.get());
      currentDecoder_ = it->second.get();
    } else {
      switch (encoding) {
        case Encoding::kPlain: {
          auto decoder = makeTypedDecoder<DType>(Encoding::kPlain, descr_);
          currentDecoder_ = decoder.get();
          decoders_[static_cast<int>(encoding)] = std::move(decoder);
          break;
        }
        case Encoding::kByteStreamSplit: {
          auto decoder =
              makeTypedDecoder<DType>(Encoding::kByteStreamSplit, descr_);
          currentDecoder_ = decoder.get();
          decoders_[static_cast<int>(encoding)] = std::move(decoder);
          break;
        }
        case Encoding::kRle: {
          auto decoder = makeTypedDecoder<DType>(Encoding::kRle, descr_);
          currentDecoder_ = decoder.get();
          decoders_[static_cast<int>(encoding)] = std::move(decoder);
          break;
        }
        case Encoding::kRleDictionary:
          throw ParquetException("Dictionary page must be before data page.");

        case Encoding::kDeltaBinaryPacked: {
          auto decoder =
              makeTypedDecoder<DType>(Encoding::kDeltaBinaryPacked, descr_);
          currentDecoder_ = decoder.get();
          decoders_[static_cast<int>(encoding)] = std::move(decoder);
          break;
        }
        case Encoding::kDeltaByteArray: {
          auto decoder =
              makeTypedDecoder<DType>(Encoding::kDeltaByteArray, descr_);
          currentDecoder_ = decoder.get();
          decoders_[static_cast<int>(encoding)] = std::move(decoder);
          break;
        }
        case Encoding::kDeltaLengthByteArray: {
          auto decoder =
              makeTypedDecoder<DType>(Encoding::kDeltaLengthByteArray, descr_);
          currentDecoder_ = decoder.get();
          decoders_[static_cast<int>(encoding)] = std::move(decoder);
          break;
        }

        default:
          throw ParquetException("Unknown encoding type.");
      }
    }
    currentEncoding_ = encoding;
    currentDecoder_->setData(
        static_cast<int>(numBufferedValues_),
        buffer,
        static_cast<int>(dataSize));
  }

  int64_t availableValuesCurrentPage() const {
    return numBufferedValues_ - numDecodedValues_;
  }

  const ColumnDescriptor* descr_;
  const int16_t maxDefLevel_;
  const int16_t maxRepLevel_;

  std::unique_ptr<PageReader> pager_;
  std::shared_ptr<Page> currentPage_;

  // Not set if full schema for this field has no optional or repeated elements.
  LevelDecoder definitionLevelDecoder_;

  // Not set for flat schemas.
  LevelDecoder repetitionLevelDecoder_;

  // The total number of values stored in the data page. This is the maximum of.
  // The number of encoded definition levels or encoded values. For.
  // Non-repeated, required columns, this is equal to the number of encoded.
  // Values. For repeated or optional values, there may be fewer data values.
  // Than levels, and this tells you how many encoded levels there are in that.
  // Case.
  int64_t numBufferedValues_;

  // The number of values from the current data page that have been decoded.
  // Into memory.
  int64_t numDecodedValues_;

  ::arrow::MemoryPool* pool_;

  using DecoderType = TypedDecoder<DType>;
  DecoderType* currentDecoder_;
  Encoding::type currentEncoding_;

  /// Flag to signal when a new dictionary has been set, for the benefit of.
  /// DictionaryRecordReader.
  bool newDictionary_;

  // The exposed encoding.
  ExposedEncoding exposedEncoding_ = ExposedEncoding::kNoEncoding;

  // Map of encoding type to the respective decoder object. For example, a.
  // Column chunk's data pages may include both dictionary-encoded and.
  // Plain-encoded data.
  std::unordered_map<int, std::unique_ptr<DecoderType>> decoders_;

  void consumeBufferedValues(int64_t numValues) {
    numDecodedValues_ += numValues;
  }
};

// ----------------------------------------------------------------------.
// TypedColumnReader implementations.

template <typename DType>
class TypedColumnReaderImpl : public TypedColumnReader<DType>,
                              public ColumnReaderImplBase<DType> {
 public:
  using T = typename DType::CType;

  TypedColumnReaderImpl(
      const ColumnDescriptor* descr,
      std::unique_ptr<PageReader> pager,
      ::arrow::MemoryPool* pool)
      : ColumnReaderImplBase<DType>(descr, pool) {
    this->pager_ = std::move(pager);
  }

  bool hasNext() override {
    return this->hasNextInternal();
  }

  int64_t readBatch(
      int64_t batchSize,
      int16_t* defLevels,
      int16_t* repLevels,
      T* values,
      int64_t* valuesRead) override;

  int64_t readBatchSpaced(
      int64_t batchSize,
      int16_t* defLevels,
      int16_t* repLevels,
      T* values,
      uint8_t* validBits,
      int64_t validBitsOffset,
      int64_t* levelsRead,
      int64_t* valuesRead,
      int64_t* nullCount) override;

  int64_t skip(int64_t numValuesToSkip) override;

  Type::type type() const override {
    return this->descr_->physicalType();
  }

  const ColumnDescriptor* descr() const override {
    return this->descr_;
  }

  ExposedEncoding getExposedEncoding() override {
    return this->exposedEncoding_;
  };

  int64_t readBatchWithDictionary(
      int64_t batchSize,
      int16_t* defLevels,
      int16_t* repLevels,
      int32_t* indices,
      int64_t* indicesRead,
      const T** dict,
      int32_t* dictLen) override;

 protected:
  void setExposedEncoding(ExposedEncoding encoding) override {
    this->exposedEncoding_ = encoding;
  }

  // Allocate enough scratch space to accommodate skipping 16-bit levels or any.
  // Value type.
  void initScratchForSkip();

  // Scratch space for reading and throwing away rep/def levels and values when.
  // Skipping.
  std::shared_ptr<ResizableBuffer> scratchForSkip_;

 private:
  // Read dictionary indices. Similar to ReadValues but decode data to.
  // Dictionary indices. This function is called only by.
  // ReadBatchWithDictionary().
  int64_t readDictionaryIndices(int64_t indicesToRead, int32_t* indices) {
    auto decoder = dynamic_cast<DictDecoder<DType>*>(this->currentDecoder_);
    return decoder->decodeIndices(static_cast<int>(indicesToRead), indices);
  }

  // Get dictionary. The dictionary should have been set by SetDict(). The.
  // Dictionary is owned by the internal decoder and is destroyed when the.
  // Reader is destroyed. This function is called only by.
  // ReadBatchWithDictionary() after dictionary is configured.
  void getDictionary(const T** dictionary, int32_t* dictionaryLength) {
    auto decoder = dynamic_cast<DictDecoder<DType>*>(this->currentDecoder_);
    decoder->getDictionary(dictionary, dictionaryLength);
  }

  // Read definition and repetition levels. Also return the number of
  // definition. Levels and number of values to read. This function is called
  // before reading. Values.
  void readLevels(
      int64_t batchSize,
      int16_t* defLevels,
      int16_t* repLevels,
      int64_t* numDefLevels,
      int64_t* valuesToRead) {
    batchSize =
        std::min(batchSize, this->numBufferedValues_ - this->numDecodedValues_);

    // If the field is required and non-repeated, there are no definition
    // levels.
    if (this->maxDefLevel_ > 0 && defLevels != nullptr) {
      *numDefLevels = this->readDefinitionLevels(batchSize, defLevels);
      // TODO(wesm): this tallying of values-to-decode can be performed with.
      // Better cache-efficiency if fused with the level decoding.
      for (int64_t i = 0; i < *numDefLevels; ++i) {
        if (defLevels[i] == this->maxDefLevel_) {
          ++(*valuesToRead);
        }
      }
    } else {
      // Required field, read all values.
      *valuesToRead = batchSize;
    }

    // Not present for non-repeated fields.
    if (this->maxRepLevel_ > 0 && repLevels != nullptr) {
      int64_t numRepLevels = this->readRepetitionLevels(batchSize, repLevels);
      if (defLevels != nullptr && *numDefLevels != numRepLevels) {
        throw ParquetException(
            "Number of decoded rep / def levels did not match");
      }
    }
  }
};

template <typename DType>
int64_t TypedColumnReaderImpl<DType>::readBatchWithDictionary(
    int64_t batchSize,
    int16_t* defLevels,
    int16_t* repLevels,
    int32_t* indices,
    int64_t* indicesRead,
    const T** dict,
    int32_t* dictLen) {
  bool hasDictOutput = dict != nullptr && dictLen != nullptr;
  // Similar logic as ReadValues to get pages.
  if (!hasNext()) {
    *indicesRead = 0;
    if (hasDictOutput) {
      *dict = nullptr;
      *dictLen = 0;
    }
    return 0;
  }

  // Verify the current data page is dictionary encoded.
  if (this->currentEncoding_ != Encoding::kRleDictionary) {
    std::stringstream ss;
    ss << "Data page is not dictionary encoded. Encoding: "
       << encodingToString(this->currentEncoding_);
    throw ParquetException(ss.str());
  }

  // Get dictionary pointer and length.
  if (hasDictOutput) {
    getDictionary(dict, dictLen);
  }

  // Similar logic as ReadValues to get def levels and rep levels.
  int64_t numDefLevels = 0;
  int64_t indicesToRead = 0;
  readLevels(batchSize, defLevels, repLevels, &numDefLevels, &indicesToRead);

  // Read dictionary indices.
  *indicesRead = readDictionaryIndices(indicesToRead, indices);
  int64_t totalIndices = std::max<int64_t>(numDefLevels, *indicesRead);
  // Some callers use a batch size of 0 just to get the dictionary.
  int64_t expectedValues =
      std::min(batchSize, this->numBufferedValues_ - this->numDecodedValues_);
  if (totalIndices == 0 && expectedValues > 0) {
    std::stringstream ss;
    ss << "Read 0 values, expected " << expectedValues;
    ParquetException::eofException(ss.str());
  }
  this->consumeBufferedValues(totalIndices);

  return totalIndices;
}

template <typename DType>
int64_t TypedColumnReaderImpl<DType>::readBatch(
    int64_t batchSize,
    int16_t* defLevels,
    int16_t* repLevels,
    T* values,
    int64_t* valuesRead) {
  // HasNext invokes ReadNewPage.
  if (!hasNext()) {
    *valuesRead = 0;
    return 0;
  }

  // TODO(wesm): keep reading data pages until batch_size is reached, or the.
  // Row group is finished.
  int64_t numDefLevels = 0;
  int64_t valuesToRead = 0;
  readLevels(batchSize, defLevels, repLevels, &numDefLevels, &valuesToRead);

  *valuesRead = this->readValues(valuesToRead, values);
  int64_t totalValues = std::max<int64_t>(numDefLevels, *valuesRead);
  int64_t expectedValues =
      std::min(batchSize, this->numBufferedValues_ - this->numDecodedValues_);
  if (totalValues == 0 && expectedValues > 0) {
    std::stringstream ss;
    ss << "Read 0 values, expected " << expectedValues;
    ParquetException::eofException(ss.str());
  }
  this->consumeBufferedValues(totalValues);

  return totalValues;
}

template <typename DType>
int64_t TypedColumnReaderImpl<DType>::readBatchSpaced(
    int64_t batchSize,
    int16_t* defLevels,
    int16_t* repLevels,
    T* values,
    uint8_t* validBits,
    int64_t validBitsOffset,
    int64_t* levelsRead,
    int64_t* valuesRead,
    int64_t* nullCountOut) {
  // HasNext invokes ReadNewPage.
  if (!hasNext()) {
    *levelsRead = 0;
    *valuesRead = 0;
    *nullCountOut = 0;
    return 0;
  }

  int64_t totalValues;
  // TODO(wesm): keep reading data pages until batch_size is reached, or the.
  // Row group is finished.
  batchSize =
      std::min(batchSize, this->numBufferedValues_ - this->numDecodedValues_);

  // If the field is required and non-repeated, there are no definition levels.
  if (this->maxDefLevel_ > 0) {
    int64_t numDefLevels = this->readDefinitionLevels(batchSize, defLevels);

    // Not present for non-repeated fields.
    if (this->maxRepLevel_ > 0) {
      int64_t numRepLevels = this->readRepetitionLevels(batchSize, repLevels);
      if (numDefLevels != numRepLevels) {
        throw ParquetException(
            "Number of decoded rep / def levels did not match");
      }
    }

    const bool hasSpacedValuesFlag = hasSpacedValues(this->descr_);
    int64_t nullCount = 0;
    if (!hasSpacedValuesFlag) {
      int valuesToRead = 0;
      for (int64_t i = 0; i < numDefLevels; ++i) {
        if (defLevels[i] == this->maxDefLevel_) {
          ++valuesToRead;
        }
      }
      totalValues = this->readValues(valuesToRead, values);
      ::arrow::bit_util::SetBitsTo(
          validBits,
          validBitsOffset,
          /*length=*/totalValues,
          /*bits_are_set=*/true);
      *valuesRead = totalValues;
    } else {
      LevelInfo info;
      info.repeatedAncestorDefLevel = this->maxDefLevel_ - 1;
      info.defLevel = this->maxDefLevel_;
      info.repLevel = this->maxRepLevel_;
      ValidityBitmapInputOutput validityIo;
      validityIo.valuesReadUpperBound = numDefLevels;
      validityIo.validBits = validBits;
      validityIo.validBitsOffset = validBitsOffset;
      validityIo.nullCount = nullCount;
      validityIo.valuesRead = *valuesRead;

      DefLevelsToBitmap(defLevels, numDefLevels, info, &validityIo);
      nullCount = validityIo.nullCount;
      *valuesRead = validityIo.valuesRead;

      totalValues = this->readValuesSpaced(
          *valuesRead,
          values,
          static_cast<int>(nullCount),
          validBits,
          validBitsOffset);
    }
    *levelsRead = numDefLevels;
    *nullCountOut = nullCount;

  } else {
    // Required field, read all values.
    totalValues = this->readValues(batchSize, values);
    ::arrow::bit_util::SetBitsTo(
        validBits,
        validBitsOffset,
        /*length=*/totalValues,
        /*bits_are_set=*/true);
    *nullCountOut = 0;
    *valuesRead = totalValues;
    *levelsRead = totalValues;
  }

  this->consumeBufferedValues(*levelsRead);
  return totalValues;
}

template <typename DType>
void TypedColumnReaderImpl<DType>::initScratchForSkip() {
  if (this->scratchForSkip_ == nullptr) {
    int valueSize = TypeTraits<DType::typeNum>::valueByteSize;
    this->scratchForSkip_ = allocateBuffer(
        this->pool_,
        kSkipScratchBatchSize * std::max<int>(sizeof(int16_t), valueSize));
  }
}

template <typename DType>
int64_t TypedColumnReaderImpl<DType>::skip(int64_t numValuesToSkip) {
  int64_t valuesToSkip = numValuesToSkip;
  // Optimization: Do not call HasNext() when values_to_skip == 0.
  while (valuesToSkip > 0 && hasNext()) {
    // If the number of values to skip is more than the number of undecoded.
    // Values, skip the Page.
    const int64_t availableValues = this->availableValuesCurrentPage();
    if (valuesToSkip >= availableValues) {
      valuesToSkip -= availableValues;
      this->consumeBufferedValues(availableValues);
    } else {
      // We need to read this Page.
      // Jump to the right offset in the Page.
      int64_t valuesRead = 0;
      initScratchForSkip();
      VELOX_DCHECK_NOT_NULL(this->scratchForSkip_);
      do {
        int64_t batchSize = std::min(kSkipScratchBatchSize, valuesToSkip);
        valuesRead = readBatch(
            static_cast<int>(batchSize),
            reinterpret_cast<int16_t*>(this->scratchForSkip_->mutable_data()),
            reinterpret_cast<int16_t*>(this->scratchForSkip_->mutable_data()),
            reinterpret_cast<T*>(this->scratchForSkip_->mutable_data()),
            &valuesRead);
        valuesToSkip -= valuesRead;
      } while (valuesRead > 0 && valuesToSkip > 0);
    }
  }
  return numValuesToSkip - valuesToSkip;
}

} // namespace

// ----------------------------------------------------------------------.
// Dynamic column reader constructor.

std::shared_ptr<ColumnReader> ColumnReader::make(
    const ColumnDescriptor* descr,
    std::unique_ptr<PageReader> pager,
    MemoryPool* pool) {
  switch (descr->physicalType()) {
    case Type::kBoolean:
      return std::make_shared<TypedColumnReaderImpl<BooleanType>>(
          descr, std::move(pager), pool);
    case Type::kInt32:
      return std::make_shared<TypedColumnReaderImpl<Int32Type>>(
          descr, std::move(pager), pool);
    case Type::kInt64:
      return std::make_shared<TypedColumnReaderImpl<Int64Type>>(
          descr, std::move(pager), pool);
    case Type::kInt96:
      return std::make_shared<TypedColumnReaderImpl<Int96Type>>(
          descr, std::move(pager), pool);
    case Type::kFloat:
      return std::make_shared<TypedColumnReaderImpl<FloatType>>(
          descr, std::move(pager), pool);
    case Type::kDouble:
      return std::make_shared<TypedColumnReaderImpl<DoubleType>>(
          descr, std::move(pager), pool);
    case Type::kByteArray:
      return std::make_shared<TypedColumnReaderImpl<ByteArrayType>>(
          descr, std::move(pager), pool);
    case Type::kFixedLenByteArray:
      return std::make_shared<TypedColumnReaderImpl<FLBAType>>(
          descr, std::move(pager), pool);
    default:
      ParquetException::NYI("type reader not implemented");
  }
  // Unreachable code, but suppress compiler warning.
  return std::shared_ptr<ColumnReader>(nullptr);
}

// ----------------------------------------------------------------------.
// RecordReader.

namespace internal {

namespace {

template <typename DType>
class TypedRecordReader : public TypedColumnReaderImpl<DType>,
                          virtual public RecordReader {
 public:
  using T = typename DType::CType;
  using BASE = TypedColumnReaderImpl<DType>;
  TypedRecordReader(
      const ColumnDescriptor* descr,
      LevelInfo leafInfo,
      MemoryPool* pool,
      bool readDenseForNullable)
      // Pager must be set using SetPageReader.
      : BASE(descr, /* pager = */ nullptr, pool) {
    leafInfo_ = leafInfo;
    nullableValues_ = leafInfo_.HasNullableValues();
    atRecordStart_ = true;
    valuesWritten_ = 0;
    nullCount_ = 0;
    valuesCapacity_ = 0;
    levelsWritten_ = 0;
    levelsPosition_ = 0;
    levelsCapacity_ = 0;
    readDenseForNullable_ = readDenseForNullable;
    usesValues_ = !(descr->physicalType() == Type::kByteArray);

    if (usesValues_) {
      values_ = allocateBuffer(pool);
    }
    validBits_ = allocateBuffer(pool);
    defLevels_ = allocateBuffer(pool);
    repLevels_ = allocateBuffer(pool);
    TypedRecordReader::reset();
  }

  // Compute the values capacity in bytes for the given number of elements.
  int64_t bytesForValues(int64_t nitems) const {
    int64_t typeSize = getTypeByteSize(this->descr_->physicalType());
    int64_t bytesForValues = -1;
    if (MultiplyWithOverflow(nitems, typeSize, &bytesForValues)) {
      throw ParquetException("Total size of items too large");
    }
    return bytesForValues;
  }

  int64_t readRecords(int64_t numRecords) override {
    if (numRecords == 0)
      return 0;
    // Delimit records, then read values at the end.
    int64_t recordsRead = 0;

    if (hasValuesToProcess()) {
      recordsRead += readRecordData(numRecords);
    }

    int64_t levelBatchSize = std::max<int64_t>(kMinLevelBatchSize, numRecords);

    // If we are in the middle of a record, we continue until reaching the.
    // Desired number of records or the end of the current record if we've
    // found. Enough records.
    while (!atRecordStart_ || recordsRead < numRecords) {
      // Is there more data to read in this row group?
      if (!this->hasNextInternal()) {
        if (!atRecordStart_) {
          // We ended the row group while inside a record that we haven't seen.
          // The end of yet. So increment the record count for the last record.
          // In the row group.
          ++recordsRead;
          atRecordStart_ = true;
        }
        break;
      }

      /// We perform multiple batch reads until we either exhaust the row group.
      /// Or observe the desired number of records.
      int64_t batchSize =
          std::min(levelBatchSize, this->availableValuesCurrentPage());

      // No more data in column.
      if (batchSize == 0) {
        break;
      }

      if (this->maxDefLevel_ > 0) {
        reserveLevels(batchSize);

        int16_t* defLevels = this->defLevels() + levelsWritten_;
        int16_t* repLevels = this->repLevels() + levelsWritten_;

        // Not present for non-repeated fields.
        int64_t levelsRead = 0;
        if (this->maxRepLevel_ > 0) {
          levelsRead = this->readDefinitionLevels(batchSize, defLevels);
          if (this->readRepetitionLevels(batchSize, repLevels) != levelsRead) {
            throw ParquetException(
                "Number of decoded rep / def levels did not match");
          }
        } else if (this->maxDefLevel_ > 0) {
          levelsRead = this->readDefinitionLevels(batchSize, defLevels);
        }

        // Exhausted column chunk.
        if (levelsRead == 0) {
          break;
        }

        levelsWritten_ += levelsRead;
        recordsRead += readRecordData(numRecords - recordsRead);
      } else {
        // No repetition or definition levels.
        batchSize = std::min(numRecords - recordsRead, batchSize);
        recordsRead += readRecordData(batchSize);
      }
    }

    return recordsRead;
  }

  // Throw away levels from start_levels_position to levels_position_.
  // Will update levels_position_, levels_written_, and levels_capacity_.
  // Accordingly and move the levels to left to fill in the gap.
  // It will resize the buffer without releasing the memory allocation.
  void throwAwayLevels(int64_t startLevelsPosition) {
    VELOX_DCHECK_LE(levelsPosition_, levelsWritten_);
    VELOX_DCHECK_LE(startLevelsPosition, levelsPosition_);
    VELOX_DCHECK_GT(this->maxDefLevel_, 0);
    VELOX_DCHECK_NOT_NULL(defLevels_);

    int64_t gap = levelsPosition_ - startLevelsPosition;
    if (gap == 0)
      return;

    int64_t levelsRemaining = levelsWritten_ - gap;

    auto leftShift = [&](::arrow::ResizableBuffer* buffer) {
      int16_t* data = reinterpret_cast<int16_t*>(buffer->mutable_data());
      std::copy(
          data + levelsPosition_,
          data + levelsWritten_,
          data + startLevelsPosition);
      PARQUET_THROW_NOT_OK(buffer->Resize(
          levelsRemaining * sizeof(int16_t),
          /*shrink_to_fit=*/false));
    };

    leftShift(defLevels_.get());

    if (this->maxRepLevel_ > 0) {
      VELOX_DCHECK_NOT_NULL(repLevels_);
      leftShift(repLevels_.get());
    }

    levelsWritten_ -= gap;
    levelsPosition_ -= gap;
    levelsCapacity_ -= gap;
  }

  // Skip records that we have in our buffer. This function is only for.
  // Non-repeated fields.
  int64_t skipRecordsInBufferNonRepeated(int64_t numRecords) {
    VELOX_DCHECK_EQ(this->maxRepLevel_, 0);
    if (!this->hasValuesToProcess() || numRecords == 0)
      return 0;

    int64_t remainingRecords = levelsWritten_ - levelsPosition_;
    int64_t skippedRecords = std::min(numRecords, remainingRecords);
    int64_t startLevelsPosition = levelsPosition_;
    // Since there is no repetition, number of levels equals number of records.
    levelsPosition_ += skippedRecords;

    // We skipped the levels by incrementing 'levels_position_'. For values.
    // We do not have a buffer, so we need to read them and throw them away.
    // First we need to figure out how many present/not-null values there are.
    std::shared_ptr<::arrow::ResizableBuffer> validBits;
    validBits = allocateBuffer(this->pool_);
    PARQUET_THROW_NOT_OK(validBits->Resize(
        ::arrow::bit_util::BytesForBits(skippedRecords),
        /*shrink_to_fit=*/true));
    ValidityBitmapInputOutput validityIo;
    validityIo.valuesReadUpperBound = skippedRecords;
    validityIo.validBits = validBits->mutable_data();
    validityIo.validBitsOffset = 0;
    DefLevelsToBitmap(
        defLevels() + startLevelsPosition,
        skippedRecords,
        this->leafInfo_,
        &validityIo);
    int64_t valuesToRead = validityIo.valuesRead - validityIo.nullCount;

    // Now that we have figured out number of values to read, we do not need.
    // These levels anymore. We will remove these values from the buffer.
    // This requires shifting the levels in the buffer to left. So this will.
    // Update levels_position_ and levels_written_.
    throwAwayLevels(startLevelsPosition);
    // For values, we do not have them in buffer, so we will read them and.
    // Throw them away.
    readAndThrowAwayValues(valuesToRead);

    // Mark the levels as read in the underlying column reader.
    this->consumeBufferedValues(skippedRecords);

    return skippedRecords;
  }

  // Attempts to skip num_records from the buffer. Will throw away levels.
  // And corresponding values for the records it skipped and consumes them from.
  // The underlying decoder. Will advance levels_position_ and update.
  // At_record_start_.
  // Returns how many records were skipped.
  int64_t delimitAndSkipRecordsInBuffer(int64_t numRecords) {
    if (numRecords == 0)
      return 0;
    // Look at the buffered levels, delimit them based on.
    // (Rep_level == 0), report back how many records are in there, and.
    // Fill in how many not-null values (def_level == max_def_level_).
    // DelimitRecords updates levels_position_.
    int64_t startLevelsPosition = levelsPosition_;
    int64_t valuesSeen = 0;
    int64_t skippedRecords = delimitRecords(numRecords, &valuesSeen);
    readAndThrowAwayValues(valuesSeen);
    // Mark those levels and values as consumed in the underlying page.
    // This must be done before we throw away levels since it updates.
    // Levels_position_ and levels_written_.
    this->consumeBufferedValues(levelsPosition_ - startLevelsPosition);
    // Updated levels_position_ and levels_written_.
    throwAwayLevels(startLevelsPosition);
    return skippedRecords;
  }

  // Skip records for repeated fields. For repeated fields, we are technically.
  // Reading and throwing away the levels and values since we do not know the.
  // Record boundaries in advance. Keep filling the buffer and skipping until
  // we. Reach the desired number of records or we run out of values in the
  // column. Chunk. Returns number of skipped records.
  int64_t skipRecordsRepeated(int64_t numRecords) {
    VELOX_DCHECK_GT(this->maxRepLevel_, 0);
    int64_t skippedRecords = 0;

    // First consume what is in the buffer.
    if (levelsPosition_ < levelsWritten_) {
      // This updates at_record_start_.
      skippedRecords = delimitAndSkipRecordsInBuffer(numRecords);
    }

    int64_t levelBatchSize =
        std::max<int64_t>(kMinLevelBatchSize, numRecords - skippedRecords);

    // If 'at_record_start_' is false, but (skipped_records == num_records), it.
    // Means that for the last record that was counted, we have not seen all.
    // Of its values yet.
    while (!atRecordStart_ || skippedRecords < numRecords) {
      // Is there more data to read in this row group?
      // HasNextInternal() will advance to the next page if necessary.
      if (!this->hasNextInternal()) {
        if (!atRecordStart_) {
          // We ended the row group while inside a record that we haven't seen.
          // The end of yet. So increment the record count for the last record.
          // In the row group.
          ++skippedRecords;
          atRecordStart_ = true;
        }
        break;
      }

      // Read some more levels.
      int64_t batchSize =
          std::min(levelBatchSize, this->availableValuesCurrentPage());
      // No more data in column. This must be an empty page.
      // If we had exhausted the last page, HasNextInternal() must have
      // advanced. To the next page. So there must be available values to
      // process.
      if (batchSize == 0) {
        break;
      }

      // For skipping we will read the levels and append them to the end.
      // Of the def_levels and rep_levels just like for read.
      reserveLevels(batchSize);

      int16_t* defLevels = this->defLevels() + levelsWritten_;
      int16_t* repLevels = this->repLevels() + levelsWritten_;

      int64_t levelsRead = 0;
      levelsRead = this->readDefinitionLevels(batchSize, defLevels);
      if (this->readRepetitionLevels(batchSize, repLevels) != levelsRead) {
        throw ParquetException(
            "Number of decoded rep / def levels did not match");
      }

      levelsWritten_ += levelsRead;
      int64_t remainingRecords = numRecords - skippedRecords;
      // This updates at_record_start_.
      skippedRecords += delimitAndSkipRecordsInBuffer(remainingRecords);
    }

    return skippedRecords;
  }

  // Read 'num_values' values and throw them away.
  // Throws an error if it could not read 'num_values'.
  void readAndThrowAwayValues(int64_t numValues) {
    int64_t valuesLeft = numValues;
    int64_t valuesRead = 0;

    // Allocate enough scratch space to accommodate 16-bit levels or any.
    // Value type.
    this->initScratchForSkip();
    VELOX_DCHECK_NOT_NULL(this->scratchForSkip_);
    do {
      int64_t batchSize = std::min<int64_t>(kSkipScratchBatchSize, valuesLeft);
      valuesRead = this->readValues(
          batchSize,
          reinterpret_cast<T*>(this->scratchForSkip_->mutable_data()));
      valuesLeft -= valuesRead;
    } while (valuesRead > 0 && valuesLeft > 0);
    if (valuesLeft > 0) {
      std::stringstream ss;
      ss << "Could not read and throw away " << numValues << " values";
      throw ParquetException(ss.str());
    }
  }

  int64_t skipRecords(int64_t numRecords) override {
    if (numRecords == 0)
      return 0;

    // Top level required field. Number of records equals to number of levels,.
    // And there is not read-ahead for levels.
    if (this->maxRepLevel_ == 0 && this->maxDefLevel_ == 0) {
      return this->skip(numRecords);
    }
    int64_t skippedRecords = 0;
    if (this->maxRepLevel_ == 0) {
      // Non-repeated optional field.
      // First consume whatever is in the buffer.
      skippedRecords = skipRecordsInBufferNonRepeated(numRecords);

      VELOX_DCHECK_LE(skippedRecords, numRecords);

      // For records that we have not buffered, we will use the column.
      // Reader's Skip to do the remaining Skip. Since the field is not.
      // Repeated number of levels to skip is the same as number of records.
      // To skip.
      skippedRecords += this->skip(numRecords - skippedRecords);
    } else {
      skippedRecords += this->skipRecordsRepeated(numRecords);
    }
    return skippedRecords;
  }

  // We may outwardly have the appearance of having exhausted a column chunk.
  // When in fact we are in the middle of processing the last batch.
  bool hasValuesToProcess() const {
    return levelsPosition_ < levelsWritten_;
  }

  std::shared_ptr<ResizableBuffer> releaseValues() override {
    if (usesValues_) {
      auto result = values_;
      PARQUET_THROW_NOT_OK(result->Resize(
          bytesForValues(valuesWritten_), /*shrink_to_fit=*/true));
      values_ = allocateBuffer(this->pool_);
      valuesCapacity_ = 0;
      return result;
    } else {
      return nullptr;
    }
  }

  std::shared_ptr<ResizableBuffer> releaseIsValid() override {
    if (nullableValues()) {
      auto result = validBits_;
      PARQUET_THROW_NOT_OK(result->Resize(
          ::arrow::bit_util::BytesForBits(valuesWritten_),
          /*shrink_to_fit=*/true));
      validBits_ = allocateBuffer(this->pool_);
      return result;
    } else {
      return nullptr;
    }
  }

  // Process written repetition/definition levels to reach the end of.
  // Records. Only used for repeated fields.
  // Process no more levels than necessary to delimit the indicated.
  // Number of logical records. Updates internal state of RecordReader.
  //
  // \return Number of records delimited.
  int64_t delimitRecords(int64_t numRecords, int64_t* valuesSeen) {
    int64_t valuesToRead = 0;
    int64_t recordsRead = 0;

    const int16_t* defLevels = this->defLevels() + levelsPosition_;
    const int16_t* repLevels = this->repLevels() + levelsPosition_;

    VELOX_DCHECK_GT(this->maxRepLevel_, 0);

    // Count logical records and number of values to read.
    while (levelsPosition_ < levelsWritten_) {
      const int16_t repLevel = *repLevels++;
      if (repLevel == 0) {
        // If at_record_start_ is true, we are seeing the start of a record.
        // For the second time, such as after repeated calls to.
        // DelimitRecords. In this case we must continue until we find.
        // Another record start or exhausting the ColumnChunk.
        if (!atRecordStart_) {
          // We've reached the end of a record; increment the record count.
          ++recordsRead;
          if (recordsRead == numRecords) {
            // We've found the number of records we were looking for. Set.
            // At_record_start_ to true and break.
            atRecordStart_ = true;
            break;
          }
        }
      }
      // We have decided to consume the level at this position; therefore we.
      // Must advance until we find another record boundary.
      atRecordStart_ = false;

      const int16_t defLevel = *defLevels++;
      if (defLevel == this->maxDefLevel_) {
        ++valuesToRead;
      }
      ++levelsPosition_;
    }
    *valuesSeen = valuesToRead;
    return recordsRead;
  }

  void reserve(int64_t capacity) override {
    reserveLevels(capacity);
    reserveValues(capacity);
  }

  int64_t updateCapacity(int64_t capacity, int64_t size, int64_t extraSize) {
    if (extraSize < 0) {
      throw ParquetException("Negative size (corrupt file?)");
    }
    int64_t targetSize = -1;
    if (AddWithOverflow(size, extraSize, &targetSize)) {
      throw ParquetException("Allocation size too large (corrupt file?)");
    }
    if (targetSize >= (1LL << 62)) {
      throw ParquetException("Allocation size too large (corrupt file?)");
    }
    if (capacity >= targetSize) {
      return capacity;
    }
    return ::arrow::bit_util::NextPower2(targetSize);
  }

  void reserveLevels(int64_t extraLevels) {
    if (this->maxDefLevel_ > 0) {
      const int64_t newLevelsCapacity =
          updateCapacity(levelsCapacity_, levelsWritten_, extraLevels);
      if (newLevelsCapacity > levelsCapacity_) {
        constexpr auto kItemSize = static_cast<int64_t>(sizeof(int16_t));
        int64_t capacityInBytes = -1;
        if (MultiplyWithOverflow(
                newLevelsCapacity, kItemSize, &capacityInBytes)) {
          throw ParquetException("Allocation size too large (corrupt file?)");
        }
        PARQUET_THROW_NOT_OK(
            defLevels_->Resize(capacityInBytes, /*shrink_to_fit=*/false));
        if (this->maxRepLevel_ > 0) {
          PARQUET_THROW_NOT_OK(
              repLevels_->Resize(capacityInBytes, /*shrink_to_fit=*/false));
        }
        levelsCapacity_ = newLevelsCapacity;
      }
    }
  }

  void reserveValues(int64_t extraValues) {
    const int64_t newValuesCapacity =
        updateCapacity(valuesCapacity_, valuesWritten_, extraValues);
    if (newValuesCapacity > valuesCapacity_) {
      // XXX(wesm): A hack to avoid memory allocation when reading directly.
      // Into builder classes.
      if (usesValues_) {
        PARQUET_THROW_NOT_OK(values_->Resize(
            bytesForValues(newValuesCapacity),
            /*shrink_to_fit=*/false));
      }
      valuesCapacity_ = newValuesCapacity;
    }
    if (nullableValues() && !readDenseForNullable_) {
      int64_t validBytesNew = ::arrow::bit_util::BytesForBits(valuesCapacity_);
      if (validBits_->size() < validBytesNew) {
        int64_t validBytesOld = ::arrow::bit_util::BytesForBits(valuesWritten_);
        PARQUET_THROW_NOT_OK(
            validBits_->Resize(validBytesNew, /*shrink_to_fit=*/false));

        // Avoid valgrind warnings.
        memset(
            validBits_->mutable_data() + validBytesOld,
            0,
            validBytesNew - validBytesOld);
      }
    }
  }

  void reset() override {
    resetValues();

    if (levelsWritten_ > 0) {
      // Throw away levels from 0 to levels_position_.
      throwAwayLevels(0);
    }

    // Call Finish on the binary builders to reset them.
  }

  void setPageReader(std::unique_ptr<PageReader> reader) override {
    atRecordStart_ = true;
    this->pager_ = std::move(reader);
    resetDecoders();
  }

  bool hasMoreData() const override {
    return this->pager_ != nullptr;
  }

  const ColumnDescriptor* descr() const override {
    return this->descr_;
  }

  // Dictionary decoders must be reset when advancing row groups.
  void resetDecoders() {
    this->decoders_.clear();
  }

  virtual void readValuesSpaced(int64_t valuesWithNulls, int64_t nullCount) {
    uint8_t* validBits = validBits_->mutable_data();
    const int64_t validBitsOffset = valuesWritten_;

    int64_t numDecoded = this->currentDecoder_->decodeSpaced(
        valuesHead<T>(),
        static_cast<int>(valuesWithNulls),
        static_cast<int>(nullCount),
        validBits,
        validBitsOffset);
    checkNumberDecoded(numDecoded, valuesWithNulls);
  }

  virtual void readValuesDense(int64_t valuesToRead) {
    int64_t numDecoded = this->currentDecoder_->decode(
        valuesHead<T>(), static_cast<int>(valuesToRead));
    checkNumberDecoded(numDecoded, valuesToRead);
  }

  // Reads repeated records and returns number of records read. Fills in.
  // Values_to_read and null_count.
  int64_t readRepeatedRecords(
      int64_t numRecords,
      int64_t* valuesToRead,
      int64_t* nullCount) {
    const int64_t startLevelsPosition = levelsPosition_;
    // Note that repeated records may be required or nullable. If they have.
    // An optional parent in the path, they will be nullable, otherwise,.
    // They are required. We use leaf_info_->HasNullableValues() that looks.
    // At repeated_ancestor_def_level to determine if it is required or.
    // Nullable. Even if they are required, we may have to read ahead and.
    // Delimit the records to get the right number of values and they will.
    // Have associated levels.
    int64_t recordsRead = delimitRecords(numRecords, valuesToRead);
    if (!nullableValues() || readDenseForNullable_) {
      readValuesDense(*valuesToRead);
      // Null_count is always 0 for required.
      VELOX_DCHECK_EQ(*nullCount, 0);
    } else {
      readSpacedForOptionalOrRepeated(
          startLevelsPosition, valuesToRead, nullCount);
    }
    return recordsRead;
  }

  // Reads optional records and returns number of records read. Fills in.
  // Values_to_read and null_count.
  int64_t readOptionalRecords(
      int64_t numRecords,
      int64_t* valuesToRead,
      int64_t* nullCount) {
    const int64_t startLevelsPosition = levelsPosition_;
    // No repetition levels, skip delimiting logic. Each level represents a.
    // Null or not null entry.
    int64_t recordsRead =
        std::min<int64_t>(levelsWritten_ - levelsPosition_, numRecords);
    // This is advanced by DelimitRecords for the repeated field case above.
    levelsPosition_ += recordsRead;

    // Optional fields are always nullable.
    if (readDenseForNullable_) {
      readDenseForOptional(startLevelsPosition, valuesToRead);
      // We don't need to update null_count when reading dense. It should be.
      // Already set to 0.
      VELOX_DCHECK_EQ(*nullCount, 0);
    } else {
      readSpacedForOptionalOrRepeated(
          startLevelsPosition, valuesToRead, nullCount);
    }
    return recordsRead;
  }

  // Reads required records and returns number of records read. Fills in.
  // Values_to_read.
  int64_t readRequiredRecords(int64_t numRecords, int64_t* valuesToRead) {
    *valuesToRead = numRecords;
    readValuesDense(*valuesToRead);
    return numRecords;
  }

  // Reads dense for optional records. First it figures out how many values to.
  // Read.
  void readDenseForOptional(
      int64_t startLevelsPosition,
      int64_t* valuesToRead) {
    // Levels_position_ must already be incremented based on number of records.
    // Read.
    VELOX_DCHECK_GE(levelsPosition_, startLevelsPosition);

    // When reading dense we need to figure out number of values to read.
    const int16_t* defLevels = this->defLevels();
    for (int64_t i = startLevelsPosition; i < levelsPosition_; ++i) {
      if (defLevels[i] == this->maxDefLevel_) {
        ++(*valuesToRead);
      }
    }
    readValuesDense(*valuesToRead);
  }

  // Reads spaced for optional or repeated fields.
  void readSpacedForOptionalOrRepeated(
      int64_t startLevelsPosition,
      int64_t* valuesToRead,
      int64_t* nullCount) {
    // Levels_position_ must already be incremented based on number of records.
    // Read.
    VELOX_DCHECK_GE(levelsPosition_, startLevelsPosition);
    ValidityBitmapInputOutput validityIo;
    validityIo.valuesReadUpperBound = levelsPosition_ - startLevelsPosition;
    validityIo.validBits = validBits_->mutable_data();
    validityIo.validBitsOffset = valuesWritten_;

    DefLevelsToBitmap(
        defLevels() + startLevelsPosition,
        levelsPosition_ - startLevelsPosition,
        leafInfo_,
        &validityIo);
    *valuesToRead = validityIo.valuesRead - validityIo.nullCount;
    *nullCount = validityIo.nullCount;
    VELOX_DCHECK_GE(*valuesToRead, 0);
    VELOX_DCHECK_GE(*nullCount, 0);
    readValuesSpaced(validityIo.valuesRead, *nullCount);
  }

  // Return number of logical records read.
  // Updates levels_position_, values_written_, and null_count_.
  int64_t readRecordData(int64_t numRecords) {
    // Conservative upper bound.
    const int64_t possibleNumValues =
        std::max<int64_t>(numRecords, levelsWritten_ - levelsPosition_);
    reserveValues(possibleNumValues);

    const int64_t startLevelsPosition = levelsPosition_;

    // To be updated by the function calls below for each of the repetition.
    // Types.
    int64_t recordsRead = 0;
    int64_t valuesToRead = 0;
    int64_t nullCount = 0;
    if (this->maxRepLevel_ > 0) {
      // Repeated fields may be nullable or not.
      // This call updates levels_position_.
      recordsRead = readRepeatedRecords(numRecords, &valuesToRead, &nullCount);
    } else if (this->maxDefLevel_ > 0) {
      // Non-repeated optional values are always nullable.
      // This call updates levels_position_.
      VELOX_DCHECK(nullableValues());
      recordsRead = readOptionalRecords(numRecords, &valuesToRead, &nullCount);
    } else {
      VELOX_DCHECK(!nullableValues());
      recordsRead = readRequiredRecords(numRecords, &valuesToRead);
      // We don't need to update null_count, since it is 0.
    }

    VELOX_DCHECK_GE(recordsRead, 0);
    VELOX_DCHECK_GE(valuesToRead, 0);
    VELOX_DCHECK_GE(nullCount, 0);

    if (readDenseForNullable_) {
      valuesWritten_ += valuesToRead;
      VELOX_DCHECK_EQ(nullCount, 0);
    } else {
      valuesWritten_ += valuesToRead + nullCount;
      nullCount_ += nullCount;
    }
    // Total values, including null spaces, if any.
    if (this->maxDefLevel_ > 0) {
      // Optional, repeated, or some mix thereof.
      this->consumeBufferedValues(levelsPosition_ - startLevelsPosition);
    } else {
      // Flat, non-repeated.
      this->consumeBufferedValues(valuesToRead);
    }

    return recordsRead;
  }

  void debugPrintState() override {
    const int16_t* defLevels = this->defLevels();
    const int16_t* repLevels = this->repLevels();
    const int64_t totalLevelsRead = levelsPosition_;

    const T* vals = reinterpret_cast<const T*>(this->values());

    if (leafInfo_.defLevel > 0) {
      std::cout << "def levels: ";
      for (int64_t i = 0; i < totalLevelsRead; ++i) {
        std::cout << defLevels[i] << " ";
      }
      std::cout << std::endl;
    }

    if (leafInfo_.repLevel > 0) {
      std::cout << "rep levels: ";
      for (int64_t i = 0; i < totalLevelsRead; ++i) {
        std::cout << repLevels[i] << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "values: ";
    for (int64_t i = 0; i < this->valuesWritten(); ++i) {
      std::cout << vals[i] << " ";
    }
    std::cout << std::endl;
  }

  void resetValues() {
    if (valuesWritten_ > 0) {
      // Resize to 0, but do not shrink to fit.
      if (usesValues_) {
        PARQUET_THROW_NOT_OK(values_->Resize(0, /*shrink_to_fit=*/false));
      }
      PARQUET_THROW_NOT_OK(validBits_->Resize(0, /*shrink_to_fit=*/false));
      valuesWritten_ = 0;
      valuesCapacity_ = 0;
      nullCount_ = 0;
    }
  }

 protected:
  template <typename T>
  T* valuesHead() {
    return reinterpret_cast<T*>(values_->mutable_data()) + valuesWritten_;
  }
  LevelInfo leafInfo_;
};

class FLBARecordReader : public TypedRecordReader<FLBAType>,
                         virtual public BinaryRecordReader {
 public:
  FLBARecordReader(
      const ColumnDescriptor* descr,
      LevelInfo leafInfo,
      ::arrow::MemoryPool* pool,
      bool readDenseForNullable)
      : TypedRecordReader<FLBAType>(
            descr,
            leafInfo,
            pool,
            readDenseForNullable),
        builder_(nullptr) {
    VELOX_DCHECK_EQ(
        static_cast<int>(descr_->physicalType()),
        static_cast<int>(Type::kFixedLenByteArray));
    int byteWidth = descr_->typeLength();
    std::shared_ptr<::arrow::DataType> type =
        ::arrow::fixed_size_binary(byteWidth);
    builder_ =
        std::make_unique<::arrow::FixedSizeBinaryBuilder>(type, this->pool_);
  }

  ::arrow::ArrayVector getBuilderChunks() override {
    std::shared_ptr<::arrow::Array> chunk;
    PARQUET_THROW_NOT_OK(builder_->Finish(&chunk));
    return ::arrow::ArrayVector({chunk});
  }

  void readValuesDense(int64_t valuesToRead) override {
    auto values = valuesHead<FLBA>();
    int64_t numDecoded =
        this->currentDecoder_->decode(values, static_cast<int>(valuesToRead));
    checkNumberDecoded(numDecoded, valuesToRead);

    for (int64_t i = 0; i < numDecoded; i++) {
      PARQUET_THROW_NOT_OK(builder_->Append(values[i].ptr));
    }
    resetValues();
  }

  void readValuesSpaced(int64_t valuesToRead, int64_t nullCount) override {
    uint8_t* validBits = validBits_->mutable_data();
    const int64_t validBitsOffset = valuesWritten_;
    auto values = valuesHead<FLBA>();

    int64_t numDecoded = this->currentDecoder_->decodeSpaced(
        values,
        static_cast<int>(valuesToRead),
        static_cast<int>(nullCount),
        validBits,
        validBitsOffset);
    VELOX_DCHECK_EQ(numDecoded, valuesToRead);

    for (int64_t i = 0; i < numDecoded; i++) {
      if (::arrow::bit_util::GetBit(validBits, validBitsOffset + i)) {
        PARQUET_THROW_NOT_OK(builder_->Append(values[i].ptr));
      } else {
        PARQUET_THROW_NOT_OK(builder_->AppendNull());
      }
    }
    resetValues();
  }

 private:
  std::unique_ptr<::arrow::FixedSizeBinaryBuilder> builder_;
};

class ByteArrayChunkedRecordReader : public TypedRecordReader<ByteArrayType>,
                                     virtual public BinaryRecordReader {
 public:
  ByteArrayChunkedRecordReader(
      const ColumnDescriptor* descr,
      LevelInfo leafInfo,
      ::arrow::MemoryPool* pool,
      bool readDenseForNullable)
      : TypedRecordReader<ByteArrayType>(
            descr,
            leafInfo,
            pool,
            readDenseForNullable) {
    VELOX_DCHECK_EQ(
        static_cast<int>(descr_->physicalType()),
        static_cast<int>(Type::kByteArray));
    accumulator_.Builder = std::make_unique<::arrow::BinaryBuilder>(pool);
  }

  ::arrow::ArrayVector getBuilderChunks() override {
    ::arrow::ArrayVector result = accumulator_.chunks;
    if (result.size() == 0 || accumulator_.Builder->length() > 0) {
      std::shared_ptr<::arrow::Array> lastChunk;
      PARQUET_THROW_NOT_OK(accumulator_.Builder->Finish(&lastChunk));
      result.push_back(std::move(lastChunk));
    }
    accumulator_.chunks = {};
    return result;
  }

  void readValuesDense(int64_t valuesToRead) override {
    int64_t numDecoded = this->currentDecoder_->decodeArrowNonNull(
        static_cast<int>(valuesToRead), &accumulator_);
    checkNumberDecoded(numDecoded, valuesToRead);
    resetValues();
  }

  void readValuesSpaced(int64_t valuesToRead, int64_t nullCount) override {
    int64_t numDecoded = this->currentDecoder_->decodeArrow(
        static_cast<int>(valuesToRead),
        static_cast<int>(nullCount),
        validBits_->mutable_data(),
        valuesWritten_,
        &accumulator_);
    checkNumberDecoded(numDecoded, valuesToRead - nullCount);
    resetValues();
  }

 private:
  // Helper data structure for accumulating builder chunks.
  typename EncodingTraits<ByteArrayType>::Accumulator accumulator_;
};

class ByteArrayDictionaryRecordReader : public TypedRecordReader<ByteArrayType>,
                                        virtual public DictionaryRecordReader {
 public:
  ByteArrayDictionaryRecordReader(
      const ColumnDescriptor* descr,
      LevelInfo leafInfo,
      ::arrow::MemoryPool* pool,
      bool readDenseForNullable)
      : TypedRecordReader<ByteArrayType>(
            descr,
            leafInfo,
            pool,
            readDenseForNullable),
        builder_(pool) {
    this->readDictionary_ = true;
  }

  std::shared_ptr<::arrow::ChunkedArray> getResult() override {
    flushBuilder();
    std::vector<std::shared_ptr<::arrow::Array>> result;
    std::swap(result, resultChunks_);
    return std::make_shared<::arrow::ChunkedArray>(
        std::move(result), builder_.type());
  }

  void flushBuilder() {
    if (builder_.length() > 0) {
      std::shared_ptr<::arrow::Array> chunk;
      PARQUET_THROW_NOT_OK(builder_.Finish(&chunk));
      resultChunks_.emplace_back(std::move(chunk));

      // Also clears the dictionary memo table.
      builder_.Reset();
    }
  }

  void maybeWriteNewDictionary() {
    if (this->newDictionary_) {
      /// If there is a new dictionary, we may need to flush the builder, then.
      /// Insert the new dictionary values.
      flushBuilder();
      builder_.ResetFull();
      auto decoder = dynamic_cast<BinaryDictDecoder*>(this->currentDecoder_);
      decoder->insertDictionary(&builder_);
      this->newDictionary_ = false;
    }
  }

  void readValuesDense(int64_t valuesToRead) override {
    int64_t numDecoded = 0;
    if (currentEncoding_ == Encoding::kRleDictionary) {
      maybeWriteNewDictionary();
      auto decoder = dynamic_cast<BinaryDictDecoder*>(this->currentDecoder_);
      numDecoded =
          decoder->decodeIndices(static_cast<int>(valuesToRead), &builder_);
    } else {
      numDecoded = this->currentDecoder_->decodeArrowNonNull(
          static_cast<int>(valuesToRead), &builder_);

      /// Flush values since they have been copied into the builder.
      resetValues();
    }
    checkNumberDecoded(numDecoded, valuesToRead);
  }

  void readValuesSpaced(int64_t valuesToRead, int64_t nullCount) override {
    VELOX_DEBUG_ONLY int64_t numDecoded = 0;
    if (currentEncoding_ == Encoding::kRleDictionary) {
      maybeWriteNewDictionary();
      auto decoder = dynamic_cast<BinaryDictDecoder*>(this->currentDecoder_);
      numDecoded = decoder->decodeIndicesSpaced(
          static_cast<int>(valuesToRead),
          static_cast<int>(nullCount),
          validBits_->mutable_data(),
          valuesWritten_,
          &builder_);
    } else {
      numDecoded = this->currentDecoder_->decodeArrow(
          static_cast<int>(valuesToRead),
          static_cast<int>(nullCount),
          validBits_->mutable_data(),
          valuesWritten_,
          &builder_);

      /// Flush values since they have been copied into the builder.
      resetValues();
    }
    VELOX_DCHECK_EQ(numDecoded, valuesToRead - nullCount);
  }

 private:
  using BinaryDictDecoder = DictDecoder<ByteArrayType>;

  ::arrow::BinaryDictionary32Builder builder_;
  std::vector<std::shared_ptr<::arrow::Array>> resultChunks_;
};

// TODO(wesm): Implement these to some satisfaction.
template <>
void TypedRecordReader<Int96Type>::debugPrintState() {}

template <>
void TypedRecordReader<ByteArrayType>::debugPrintState() {}

template <>
void TypedRecordReader<FLBAType>::debugPrintState() {}

std::shared_ptr<RecordReader> makeByteArrayRecordReader(
    const ColumnDescriptor* descr,
    LevelInfo leafInfo,
    ::arrow::MemoryPool* pool,
    bool readDictionary,
    bool readDenseForNullable) {
  if (readDictionary) {
    return std::make_shared<ByteArrayDictionaryRecordReader>(
        descr, leafInfo, pool, readDenseForNullable);
  } else {
    return std::make_shared<ByteArrayChunkedRecordReader>(
        descr, leafInfo, pool, readDenseForNullable);
  }
}

} // namespace

std::shared_ptr<RecordReader> RecordReader::make(
    const ColumnDescriptor* descr,
    LevelInfo leafInfo,
    MemoryPool* pool,
    bool readDictionary,
    bool readDenseForNullable) {
  switch (descr->physicalType()) {
    case Type::kBoolean:
      return std::make_shared<TypedRecordReader<BooleanType>>(
          descr, leafInfo, pool, readDenseForNullable);
    case Type::kInt32:
      return std::make_shared<TypedRecordReader<Int32Type>>(
          descr, leafInfo, pool, readDenseForNullable);
    case Type::kInt64:
      return std::make_shared<TypedRecordReader<Int64Type>>(
          descr, leafInfo, pool, readDenseForNullable);
    case Type::kInt96:
      return std::make_shared<TypedRecordReader<Int96Type>>(
          descr, leafInfo, pool, readDenseForNullable);
    case Type::kFloat:
      return std::make_shared<TypedRecordReader<FloatType>>(
          descr, leafInfo, pool, readDenseForNullable);
    case Type::kDouble:
      return std::make_shared<TypedRecordReader<DoubleType>>(
          descr, leafInfo, pool, readDenseForNullable);
    case Type::kByteArray: {
      return makeByteArrayRecordReader(
          descr, leafInfo, pool, readDictionary, readDenseForNullable);
    }
    case Type::kFixedLenByteArray:
      return std::make_shared<FLBARecordReader>(
          descr, leafInfo, pool, readDenseForNullable);
    default: {
      // PARQUET-1481: This can occur if the file is corrupt.
      std::stringstream ss;
      ss << "Invalid physical column type: "
         << static_cast<int>(descr->physicalType());
      throw ParquetException(ss.str());
    }
  }
  // Unreachable code, but suppress compiler warning.
  return nullptr;
}

} // namespace internal
} // namespace facebook::velox::parquet::arrow
