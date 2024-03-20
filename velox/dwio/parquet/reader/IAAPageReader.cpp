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

#include "velox/dwio/parquet/reader/IAAPageReader.h"
#include "velox/dwio/common/BufferUtil.h"

namespace facebook::velox::parquet {

using thrift::Encoding;
using thrift::PageHeader;

void IAAPageReader::preDecompressPage(
    bool& need_pre_decompress,
    int64_t numValues) {
  if (codec_ != common::CompressionKind::CompressionKind_GZIP) {
    need_pre_decompress = false;
    return;
  }
  for (;;) {
    auto dataStart = pageStart_;
    if (chunkSize_ <= pageStart_) {
      // This may happen if seeking to exactly end of row group.
      numRepDefsInPage_ = 0;
      numRowsInPage_ = 0;
      break;
    }
    PageHeader pageHeader = readPageHeader();
    pageStart_ = pageDataStart_ + pageHeader.compressed_page_size;
    switch (pageHeader.type) {
      case thrift::PageType::DATA_PAGE:
        prefetchDataPageV1(pageHeader);
        break;
      case thrift::PageType::DATA_PAGE_V2:
        prefetchDataPageV2(pageHeader);
        break;
      case thrift::PageType::DICTIONARY_PAGE:
        prefetchDictionary(pageHeader);
        continue;
      default:
        break; // ignore INDEX page type and any other custom extensions
    }
    break;
  }
  need_pre_decompress = isWinSizeFit_;
  rowGroupPageInfo_.numValues = numValues;
  rowGroupPageInfo_.visitedRows = 0;
}

void IAAPageReader::prefetchNextPage() {
  if (rowGroupPageInfo_.visitedRows + numRowsInPage_ >=
      rowGroupPageInfo_.numValues) {
    return;
  }
  if (chunkSize_ <= pageStart_) {
    return;
  }
  PageHeader pageHeader = readPageHeader();
  switch (pageHeader.type) {
    case thrift::PageType::DATA_PAGE: {
      dataPageHeader_ = pageHeader;
      VELOX_CHECK(
          pageHeader.type == thrift::PageType::DATA_PAGE &&
          pageHeader.__isset.data_page_header);
      rowGroupPageInfo_.dataPageData =
          readBytes(pageHeader.compressed_page_size, pageBuffer_);
      preDecompressData_ = iaaDecompress(
          rowGroupPageInfo_.dataPageData,
          pageHeader.compressed_page_size,
          pageHeader.uncompressed_page_size,
          rowGroupPageInfo_.uncompressedData,
          dataDecompFuture);
      break;
    }
    case thrift::PageType::DATA_PAGE_V2:
      LOG(WARNING) << "Data Page V2 not support ";
      break;
    case thrift::PageType::DICTIONARY_PAGE:
      LOG(WARNING) << "Wrong path ";
      break;
    default:
      break; // ignore INDEX page type and any other custom extensions
  }
}

bool IAAPageReader::seekToPreDecompPage(int64_t row) {
  bool has_qpl = false;
  if (this->dictDecompFuture.valid()) {
    bool job_success = std::move(this->dictDecompFuture).get() > 0;
    prepareDictionary(dictPageHeader_, job_success);
    preDecompressDict_ = false;
    has_qpl = true;
  }

  if (dataDecompFuture.valid()) {
    bool job_success = std::move(this->dataDecompFuture).get() > 0;
    prepareDataPageV1(dataPageHeader_, row, job_success);
    preDecompressData_ = false;
    has_qpl = true;
  }

  if (has_qpl) {
    if (row == kRepDefOnly || row < rowOfPage_ + numRowsInPage_) {
      return true;
    }
    updateRowInfoAfterPageSkipped();
  }
  return false;
}

void IAAPageReader::prefetchDataPageV1(const thrift::PageHeader& pageHeader) {
  dataPageHeader_ = pageHeader;
  VELOX_CHECK(
      pageHeader.type == thrift::PageType::DATA_PAGE &&
      pageHeader.__isset.data_page_header);

  dataPageData_ = readBytes(pageHeader.compressed_page_size, pageBuffer_);
  preDecompressData_ = iaaDecompress(
      dataPageData_,
      pageHeader.compressed_page_size,
      pageHeader.uncompressed_page_size,
      uncompressedDataV1Data_,
      dataDecompFuture);
  return;
}

void IAAPageReader::prefetchDataPageV2(const thrift::PageHeader& pageHeader) {
  return;
}

void IAAPageReader::prefetchDictionary(const thrift::PageHeader& pageHeader) {
  dictPageHeader_ = pageHeader;
  dictionaryEncoding_ = pageHeader.dictionary_page_header.encoding;
  VELOX_CHECK(
      dictionaryEncoding_ == Encoding::PLAIN_DICTIONARY ||
      dictionaryEncoding_ == Encoding::PLAIN);
  dictPageData_ = readBytes(pageHeader.compressed_page_size, pageBuffer_);

  preDecompressDict_ = iaaDecompress(
      dictPageData_,
      pageHeader.compressed_page_size,
      pageHeader.uncompressed_page_size,
      uncompressedDictData_,
      dictDecompFuture);

  return;
}

const bool IAAPageReader::iaaDecompress(
    const char* pageData,
    uint32_t compressedSize,
    uint32_t uncompressedSize,
    BufferPtr& uncompressedData,
    folly::SemiFuture<uint64_t>& future) {
  dwio::common::ensureCapacity<char>(
      uncompressedData, uncompressedSize, &pool_);
  static constexpr int PARQUET_ZLIB_WINDOW_BITS_4KB = 12;
  future = folly::makeSemiFuture((uint64_t)0);
  if (!isWinSizeFit_) {
    // window size should be 4KB for IAA
    if (PARQUET_ZLIB_WINDOW_BITS_4KB ==
        dwio::common::compression::getZlibWindowBits(
            (const uint8_t*)pageData, uncompressedSize)) {
      isWinSizeFit_ = true;
    } else {
      future = folly::makeSemiFuture((uint64_t)0);
      return true;
    }
  }
  std::unique_ptr<dwio::common::compression::AsyncDecompressor> decompressor =
      dwio::common::compression::createAsyncDecompressor(codec_);
  if (decompressor == nullptr) {
    return true;
  }
  auto decompFuture = decompressor->decompressAsync(
      (const char*)pageData,
      compressedSize,
      (char*)uncompressedData->asMutable<char>(),
      uncompressedSize);
  if (decompFuture.isReady()) {
    auto result = std::move(decompFuture).getTry();
    if (result.hasException()) {
      future = folly::makeSemiFuture((uint64_t)0);
      return true;
    }
  }
  future = std::move(decompFuture);
  return true;
}

void IAAPageReader::seekToPage(int64_t row) {
  this->defineDecoder_.reset();
  this->repeatDecoder_.reset();
  // 'rowOfPage_' is the row number of the first row of the next page.
  this->rowOfPage_ += this->numRowsInPage_;

  if (seekToPreDecompPage(row)) {
    if (isWinSizeFit_) {
      prefetchNextPage();
    }
    rowGroupPageInfo_.visitedRows += numRowsInPage_;
    return;
  }

  for (;;) {
    auto dataStart = pageStart_;
    if (chunkSize_ <= pageStart_) {
      // This may happen if seeking to exactly end of row group.
      numRepDefsInPage_ = 0;
      numRowsInPage_ = 0;
      break;
    }
    PageHeader pageHeader = this->readPageHeader();
    pageStart_ = pageDataStart_ + pageHeader.compressed_page_size;

    switch (pageHeader.type) {
      case thrift::PageType::DATA_PAGE:
        prepareDataPageV1(pageHeader, row);
        break;
      case thrift::PageType::DATA_PAGE_V2:
        prepareDataPageV2(pageHeader, row);
        break;
      case thrift::PageType::DICTIONARY_PAGE:
        if (row == kRepDefOnly) {
          skipBytes(
              pageHeader.compressed_page_size,
              inputStream_.get(),
              bufferStart_,
              bufferEnd_);
          continue;
        }
        prepareDictionary(pageHeader);
        continue;
      default:
        break; // ignore INDEX page type and any other custom extensions
    }
    if (row == kRepDefOnly || row < rowOfPage_ + numRowsInPage_) {
      break;
    }
    this->updateRowInfoAfterPageSkipped();
  }
  if (isWinSizeFit_) {
    prefetchNextPage();
  }
  rowGroupPageInfo_.visitedRows += numRowsInPage_;
}

void IAAPageReader::prepareDataPageV1(
    const PageHeader& pageHeader,
    int64_t row,
    bool job_success) {
  VELOX_CHECK(
      pageHeader.type == thrift::PageType::DATA_PAGE &&
      pageHeader.__isset.data_page_header);
  numRepDefsInPage_ = pageHeader.data_page_header.num_values;
  setPageRowInfo(row == kRepDefOnly);
  if (row != kRepDefOnly && numRowsInPage_ != kRowsUnknown &&
      numRowsInPage_ + rowOfPage_ <= row) {
    dwio::common::skipBytes(
        pageHeader.compressed_page_size,
        inputStream_.get(),
        bufferStart_,
        bufferEnd_);

    return;
  }
  if (job_success) {
    if (rowGroupPageInfo_.visitedRows > 0) {
      BufferPtr tmp = uncompressedDataV1Data_;
      uncompressedDataV1Data_ = rowGroupPageInfo_.uncompressedData;
      rowGroupPageInfo_.uncompressedData = tmp;
    }
    pageData_ = uncompressedDataV1Data_->as<char>();
  } else {
    if (!preDecompressData_) {
      dataPageData_ = readBytes(pageHeader.compressed_page_size, pageBuffer_);
    } else if (rowGroupPageInfo_.visitedRows > 0) {
      dataPageData_ = rowGroupPageInfo_.dataPageData;
    }
    pageData_ = decompressData(
        dataPageData_,
        pageHeader.compressed_page_size,
        pageHeader.uncompressed_page_size);
  }
  auto pageEnd = pageData_ + pageHeader.uncompressed_page_size;
  if (maxRepeat_ > 0) {
    uint32_t repeatLength = readField<int32_t>(pageData_);
    repeatDecoder_ = std::make_unique<::arrow::util::RleDecoder>(
        reinterpret_cast<const uint8_t*>(pageData_),
        repeatLength,
        ::arrow::bit_util::NumRequiredBits(maxRepeat_));

    pageData_ += repeatLength;
  }

  if (maxDefine_ > 0) {
    auto defineLength = readField<uint32_t>(pageData_);
    if (maxDefine_ == 1) {
      defineDecoder_ = std::make_unique<RleBpDecoder>(
          pageData_,
          pageData_ + defineLength,
          ::arrow::bit_util::NumRequiredBits(maxDefine_));
    }
    wideDefineDecoder_ = std::make_unique<::arrow::util::RleDecoder>(
        reinterpret_cast<const uint8_t*>(pageData_),
        defineLength,
        ::arrow::bit_util::NumRequiredBits(maxDefine_));
    pageData_ += defineLength;
  }
  encodedDataSize_ = pageEnd - pageData_;

  encoding_ = pageHeader.data_page_header.encoding;
  if (!hasChunkRepDefs_ && (numRowsInPage_ == kRowsUnknown || maxDefine_ > 1)) {
    readPageDefLevels();
  }

  if (row != kRepDefOnly) {
    makeDecoder();
  }
}

void IAAPageReader::prepareDictionary(
    const PageHeader& pageHeader,
    bool job_success) {
  dictionary_.numValues = pageHeader.dictionary_page_header.num_values;
  dictionaryEncoding_ = pageHeader.dictionary_page_header.encoding;
  dictionary_.sorted = pageHeader.dictionary_page_header.__isset.is_sorted &&
      pageHeader.dictionary_page_header.is_sorted;
  VELOX_CHECK(
      dictionaryEncoding_ == Encoding::PLAIN_DICTIONARY ||
      dictionaryEncoding_ == Encoding::PLAIN);

  if (codec_ != common::CompressionKind::CompressionKind_NONE) {
    if (job_success) {
      pageData_ = uncompressedDictData_->as<char>();
    } else {
      if (!preDecompressDict_) {
        dictPageData_ = readBytes(pageHeader.compressed_page_size, pageBuffer_);
      }
      pageData_ = decompressData(
          dictPageData_,
          pageHeader.compressed_page_size,
          pageHeader.uncompressed_page_size);
    }
  }

  auto parquetType = type_->parquetType_.value();
  switch (parquetType) {
    case thrift::Type::INT32:
    case thrift::Type::INT64:
    case thrift::Type::FLOAT:
    case thrift::Type::DOUBLE: {
      int32_t typeSize = (parquetType == thrift::Type::INT32 ||
                          parquetType == thrift::Type::FLOAT)
          ? sizeof(float)
          : sizeof(double);
      auto numBytes = dictionary_.numValues * typeSize;
      if (type_->type()->isShortDecimal() &&
          parquetType == thrift::Type::INT32) {
        auto veloxTypeLength = type_->type()->cppSizeInBytes();
        auto numVeloxBytes = dictionary_.numValues * veloxTypeLength;
        dictionary_.values =
            AlignedBuffer::allocate<char>(numVeloxBytes, &pool_);
      } else {
        dictionary_.values = AlignedBuffer::allocate<char>(numBytes, &pool_);
      }
      if (pageData_) {
        memcpy(dictionary_.values->asMutable<char>(), pageData_, numBytes);
      } else {
        dwio::common::readBytes(
            numBytes,
            inputStream_.get(),
            dictionary_.values->asMutable<char>(),
            bufferStart_,
            bufferEnd_);
      }
      if (type_->type()->isShortDecimal() &&
          parquetType == thrift::Type::INT32) {
        auto values = dictionary_.values->asMutable<int64_t>();
        auto parquetValues = dictionary_.values->asMutable<int32_t>();
        for (auto i = dictionary_.numValues - 1; i >= 0; --i) {
          // Expand the Parquet type length values to Velox type length.
          // We start from the end to allow in-place expansion.
          values[i] = parquetValues[i];
        }
      }
      break;
    }
    case thrift::Type::BYTE_ARRAY: {
      dictionary_.values =
          AlignedBuffer::allocate<StringView>(dictionary_.numValues, &pool_);
      auto numBytes = pageHeader.uncompressed_page_size;
      auto values = dictionary_.values->asMutable<StringView>();
      dictionary_.strings = AlignedBuffer::allocate<char>(numBytes, &pool_);
      auto strings = dictionary_.strings->asMutable<char>();
      if (pageData_) {
        memcpy(strings, pageData_, numBytes);
      } else {
        dwio::common::readBytes(
            numBytes, inputStream_.get(), strings, bufferStart_, bufferEnd_);
      }
      auto header = strings;
      for (auto i = 0; i < dictionary_.numValues; ++i) {
        auto length = *reinterpret_cast<const int32_t*>(header);
        values[i] = StringView(header + sizeof(int32_t), length);
        header += length + sizeof(int32_t);
      }
      VELOX_CHECK_EQ(header, strings + numBytes);
      break;
    }
    case thrift::Type::FIXED_LEN_BYTE_ARRAY: {
      auto parquetTypeLength = type_->typeLength_;
      auto numParquetBytes = dictionary_.numValues * parquetTypeLength;
      auto veloxTypeLength = type_->type()->cppSizeInBytes();
      auto numVeloxBytes = dictionary_.numValues * veloxTypeLength;
      dictionary_.values = AlignedBuffer::allocate<char>(numVeloxBytes, &pool_);
      auto data = dictionary_.values->asMutable<char>();
      // Read the data bytes.
      if (pageData_) {
        memcpy(data, pageData_, numParquetBytes);
      } else {
        dwio::common::readBytes(
            numParquetBytes,
            inputStream_.get(),
            data,
            bufferStart_,
            bufferEnd_);
      }
      if (type_->type()->isShortDecimal()) {
        // Parquet decimal values have a fixed typeLength_ and are in big-endian
        // layout.
        if (numParquetBytes < numVeloxBytes) {
          auto values = dictionary_.values->asMutable<int64_t>();
          for (auto i = dictionary_.numValues - 1; i >= 0; --i) {
            // Expand the Parquet type length values to Velox type length.
            // We start from the end to allow in-place expansion.
            auto sourceValue = data + (i * parquetTypeLength);
            int64_t value = *sourceValue >= 0 ? 0 : -1;
            memcpy(
                reinterpret_cast<uint8_t*>(&value) + veloxTypeLength -
                    parquetTypeLength,
                sourceValue,
                parquetTypeLength);
            values[i] = value;
          }
        }
        auto values = dictionary_.values->asMutable<int64_t>();
        for (auto i = 0; i < dictionary_.numValues; ++i) {
          values[i] = __builtin_bswap64(values[i]);
        }
        break;
      } else if (type_->type()->isLongDecimal()) {
        // Parquet decimal values have a fixed typeLength_ and are in big-endian
        // layout.
        if (numParquetBytes < numVeloxBytes) {
          auto values = dictionary_.values->asMutable<int128_t>();
          for (auto i = dictionary_.numValues - 1; i >= 0; --i) {
            // Expand the Parquet type length values to Velox type length.
            // We start from the end to allow in-place expansion.
            auto sourceValue = data + (i * parquetTypeLength);
            int128_t value = *sourceValue >= 0 ? 0 : -1;
            memcpy(
                reinterpret_cast<uint8_t*>(&value) + veloxTypeLength -
                    parquetTypeLength,
                sourceValue,
                parquetTypeLength);
            values[i] = value;
          }
        }
        auto values = dictionary_.values->asMutable<int128_t>();
        for (auto i = 0; i < dictionary_.numValues; ++i) {
          values[i] = bits::builtin_bswap128(values[i]);
        }
        break;
      }
      VELOX_UNSUPPORTED(
          "Parquet type {} not supported for dictionary", parquetType);
    }
    case thrift::Type::INT96:
    default:
      VELOX_UNSUPPORTED(
          "Parquet type {} not supported for dictionary", parquetType);
  }
}

IAAPageReader::~IAAPageReader() {
  if (dataDecompFuture.valid()) {
    std::move(dataDecompFuture).get();
  }

  if (dictDecompFuture.valid()) {
    std::move(dictDecompFuture).get();
  }
}
} // namespace facebook::velox::parquet