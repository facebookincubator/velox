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

#include "velox/dwio/parquet/writer/arrow/ColumnWriter.h"

#include <glog/logging.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/buffer_builder.h"
#include "arrow/compute/api.h"
#include "arrow/io/memory.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/bitmap_ops.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/endian.h"
#include "arrow/util/type_traits.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/common/LevelConversion.h"
#include "velox/dwio/parquet/writer/arrow/ColumnPage.h"
#include "velox/dwio/parquet/writer/arrow/Encoding.h"
#include "velox/dwio/parquet/writer/arrow/Encryption.h"
#include "velox/dwio/parquet/writer/arrow/EncryptionInternal.h"
#include "velox/dwio/parquet/writer/arrow/FileEncryptorInternal.h"
#include "velox/dwio/parquet/writer/arrow/Metadata.h"
#include "velox/dwio/parquet/writer/arrow/PageIndex.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/Statistics.h"
#include "velox/dwio/parquet/writer/arrow/ThriftInternal.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"
#include "velox/dwio/parquet/writer/arrow/util/Compression.h"
#include "velox/dwio/parquet/writer/arrow/util/Crc32.h"
#include "velox/dwio/parquet/writer/arrow/util/VisitArrayInline.h"

using arrow::Array;
using arrow::ArrayData;
using arrow::Datum;
using arrow::ResizableBuffer;
using arrow::Result;
using arrow::Status;
using arrow::internal::checked_cast;
using arrow::internal::checked_pointer_cast;

namespace arrow {
fmt::underlying_t<Type::type> formatAs(Type::type type) {
  return fmt::underlying(type);
}
}; // namespace arrow

namespace facebook::velox::parquet::arrow {
using util::CodecOptions;

namespace {

// Visitor that extracts the value buffer from a FlatArray at a given offset.
struct ValueBufferSlicer {
  template <typename T>
  ::arrow::enable_if_base_binary<typename T::TypeClass, Status> visit(
      const T& array,
      std::shared_ptr<Buffer>* buffer) {
    auto data = array.data();
    *buffer = ::arrow::SliceBuffer(
        data->buffers[1],
        data->offset * sizeof(typename T::offset_type),
        data->length * sizeof(typename T::offset_type));
    return Status::OK();
  }

  template <typename T>
  ::arrow::enable_if_fixed_size_binary<typename T::TypeClass, Status> visit(
      const T& array,
      std::shared_ptr<Buffer>* buffer) {
    auto data = array.data();
    *buffer = ::arrow::SliceBuffer(
        data->buffers[1],
        data->offset * array.byte_width(),
        data->length * array.byte_width());
    return Status::OK();
  }

  template <typename T>
  ::arrow::enable_if_t<
      ::arrow::has_c_type<typename T::TypeClass>::value &&
          !std::is_same<BooleanType, typename T::TypeClass>::value,
      Status>
  visit(const T& array, std::shared_ptr<Buffer>* buffer) {
    auto data = array.data();
    *buffer = ::arrow::SliceBuffer(
        data->buffers[1],
        ::arrow::TypeTraits<typename T::TypeClass>::bytes_required(
            data->offset),
        ::arrow::TypeTraits<typename T::TypeClass>::bytes_required(
            data->length));
    return Status::OK();
  }

  Status visit(
      const ::arrow::BooleanArray& array,
      std::shared_ptr<Buffer>* buffer) {
    auto data = array.data();
    if (::arrow::bit_util::IsMultipleOf8(data->offset)) {
      *buffer = ::arrow::SliceBuffer(
          data->buffers[1],
          ::arrow::bit_util::BytesForBits(data->offset),
          ::arrow::bit_util::BytesForBits(data->length));
      return Status::OK();
    }
    PARQUET_ASSIGN_OR_THROW(
        *buffer,
        ::arrow::internal::CopyBitmap(
            pool_, data->buffers[1]->data(), data->offset, data->length));
    return Status::OK();
  }
#define NOT_IMPLEMENTED_VISIT(ArrowTypePrefix)            \
  Status visit(                                           \
      const ::arrow::ArrowTypePrefix##Array& array,       \
      std::shared_ptr<Buffer>* buffer) {                  \
    return Status::NotImplemented(                        \
        "Slicing not implemented for " #ArrowTypePrefix); \
  }

  NOT_IMPLEMENTED_VISIT(Null);
  NOT_IMPLEMENTED_VISIT(Union);
  NOT_IMPLEMENTED_VISIT(List);
  NOT_IMPLEMENTED_VISIT(LargeList);
  NOT_IMPLEMENTED_VISIT(ListView);
  NOT_IMPLEMENTED_VISIT(LargeListView);
  NOT_IMPLEMENTED_VISIT(Struct);
  NOT_IMPLEMENTED_VISIT(FixedSizeList);
  NOT_IMPLEMENTED_VISIT(Dictionary);
  NOT_IMPLEMENTED_VISIT(RunEndEncoded);
  NOT_IMPLEMENTED_VISIT(Extension);
  NOT_IMPLEMENTED_VISIT(BinaryView);
  NOT_IMPLEMENTED_VISIT(StringView);

#undef NOT_IMPLEMENTED_VISIT

  MemoryPool* pool_;
};

LevelInfo computeLevelInfo(const ColumnDescriptor* descr) {
  LevelInfo levelInfo;
  levelInfo.defLevel = descr->maxDefinitionLevel();
  levelInfo.repLevel = descr->maxRepetitionLevel();

  int16_t minSpacedDefLevel = descr->maxDefinitionLevel();
  const schema::Node* node = descr->schemaNode().get();
  while (node != nullptr && !node->isRepeated()) {
    if (node->isOptional()) {
      minSpacedDefLevel--;
    }
    node = node->parent();
  }
  levelInfo.repeatedAncestorDefLevel = minSpacedDefLevel;
  return levelInfo;
}

template <class T>
inline const T* addIfNotNull(const T* base, int64_t offset) {
  if (base != nullptr) {
    return base + offset;
  }
  return nullptr;
}

} // namespace

LevelEncoder::LevelEncoder() {}
LevelEncoder::~LevelEncoder() {}

void LevelEncoder::init(
    Encoding::type encoding,
    int16_t maxLevel,
    int numBufferedValues,
    uint8_t* data,
    int dataSize) {
  bitWidth_ = ::arrow::bit_util::Log2(maxLevel + 1);
  encoding_ = encoding;
  switch (encoding) {
    case Encoding::kRle: {
      rleEncoder_ = std::make_unique<RleEncoder>(data, dataSize, bitWidth_);
      break;
    }
    case Encoding::kBitPacked: {
      int numBytes = static_cast<int>(
          ::arrow::bit_util::BytesForBits(numBufferedValues * bitWidth_));
      bitPackedEncoder_ = std::make_unique<BitWriter>(data, numBytes);
      break;
    }
    default:
      throw ParquetException("Unknown encoding type for levels.");
  }
}

int LevelEncoder::maxBufferSize(
    Encoding::type encoding,
    int16_t maxLevel,
    int numBufferedValues) {
  int bitWidth = ::arrow::bit_util::Log2(maxLevel + 1);
  int numBytes = 0;
  switch (encoding) {
    case Encoding::kRle: {
      // TODO: Due to the way we currently check if the buffer is full enough,
      // we need to have MinBufferSize as head room.
      numBytes = RleEncoder::MaxBufferSize(bitWidth, numBufferedValues) +
          RleEncoder::MinBufferSize(bitWidth);
      break;
    }
    case Encoding::kBitPacked: {
      numBytes = static_cast<int>(
          ::arrow::bit_util::BytesForBits(numBufferedValues * bitWidth));
      break;
    }
    default:
      throw ParquetException("Unknown encoding type for levels.");
  }
  return numBytes;
}

int LevelEncoder::encode(int batchSize, const int16_t* levels) {
  int numEncoded = 0;
  if (!rleEncoder_ && !bitPackedEncoder_) {
    throw ParquetException("Level encoders are not initialized.");
  }

  if (encoding_ == Encoding::kRle) {
    for (int i = 0; i < batchSize; ++i) {
      if (!rleEncoder_->Put(*(levels + i))) {
        break;
      }
      ++numEncoded;
    }
    rleEncoder_->Flush();
    rleLength_ = rleEncoder_->len();
  } else {
    for (int i = 0; i < batchSize; ++i) {
      if (!bitPackedEncoder_->PutValue(*(levels + i), bitWidth_)) {
        break;
      }
      ++numEncoded;
    }
    bitPackedEncoder_->Flush();
  }
  return numEncoded;
}

// ----------------------------------------------------------------------.
// PageWriter implementation.

// This subclass delimits pages appearing in a serialized stream, each preceded
// by a serialized Thrift facebook::velox::parquet::thrift::PageHeader
// indicating the type of each page and the page metadata.
class SerializedPageWriter : public PageWriter {
 public:
  SerializedPageWriter(
      std::shared_ptr<ArrowOutputStream> sink,
      Compression::type codec,
      ColumnChunkMetaDataBuilder* metadata,
      int16_t rowGroupOrdinal,
      int16_t columnChunkOrdinal,
      bool usePageChecksumVerification,
      MemoryPool* pool = ::arrow::default_memory_pool(),
      std::shared_ptr<Encryptor> metaEncryptor = nullptr,
      std::shared_ptr<Encryptor> dataEncryptor = nullptr,
      ColumnIndexBuilder* columnIndexBuilder = nullptr,
      OffsetIndexBuilder* offsetIndexBuilder = nullptr,
      const CodecOptions& codecOptions = CodecOptions{})
      : sink_(std::move(sink)),
        metadata_(metadata),
        pool_(pool),
        numValues_(0),
        dictionaryPageOffset_(0),
        dataPageOffset_(0),
        totalUncompressedSize_(0),
        totalCompressedSize_(0),
        pageOrdinal_(0),
        rowGroupOrdinal_(rowGroupOrdinal),
        columnOrdinal_(columnChunkOrdinal),
        pageChecksumVerification_(usePageChecksumVerification),
        metaEncryptor_(std::move(metaEncryptor)),
        dataEncryptor_(std::move(dataEncryptor)),
        encryptionBuffer_(allocateBuffer(pool, 0)),
        columnIndexBuilder_(columnIndexBuilder),
        offsetIndexBuilder_(offsetIndexBuilder) {
    if (dataEncryptor_ != nullptr || metaEncryptor_ != nullptr) {
      initEncryption();
    }
    compressor_ = getCodec(codec, codecOptions);
    thriftSerializer_ = std::make_unique<ThriftSerializer>();
  }

  int64_t writeDictionaryPage(const DictionaryPage& page) override {
    int64_t uncompressedSize = page.size();
    std::shared_ptr<Buffer> compressedData;
    if (hasCompressor()) {
      auto buffer = std::static_pointer_cast<ResizableBuffer>(
          allocateBuffer(pool_, uncompressedSize));
      compress(*(page.buffer().get()), buffer.get());
      compressedData = std::static_pointer_cast<Buffer>(buffer);
    } else {
      compressedData = page.buffer();
    }

    facebook::velox::parquet::thrift::DictionaryPageHeader dictPageHeader;
    dictPageHeader.__set_num_values(page.numValues());
    dictPageHeader.__set_encoding(toThrift(page.encoding()));
    dictPageHeader.__set_is_sorted(page.isSorted());

    const uint8_t* outputDataBuffer = compressedData->data();
    int32_t outputDataLen = static_cast<int32_t>(compressedData->size());

    if (dataEncryptor_.get()) {
      updateEncryption(encryption::kDictionaryPage);
      PARQUET_THROW_NOT_OK(encryptionBuffer_->Resize(
          dataEncryptor_->ciphertextSizeDelta() + outputDataLen, false));
      outputDataLen = dataEncryptor_->encrypt(
          compressedData->data(),
          outputDataLen,
          encryptionBuffer_->mutable_data());
      outputDataBuffer = encryptionBuffer_->data();
    }

    facebook::velox::parquet::thrift::PageHeader pageHeader;
    pageHeader.__set_type(
        facebook::velox::parquet::thrift::PageType::DICTIONARY_PAGE);
    pageHeader.__set_uncompressed_page_size(
        static_cast<int32_t>(uncompressedSize));
    pageHeader.__set_compressed_page_size(static_cast<int32_t>(outputDataLen));
    pageHeader.__set_dictionary_page_header(dictPageHeader);
    if (pageChecksumVerification_) {
      uint32_t crc32 =
          internal::crc32(/* prev */ 0, outputDataBuffer, outputDataLen);
      pageHeader.__set_crc(static_cast<int32_t>(crc32));
    }

    PARQUET_ASSIGN_OR_THROW(int64_t startPos, sink_->Tell());
    if (dictionaryPageOffset_ == 0) {
      dictionaryPageOffset_ = startPos;
    }

    if (metaEncryptor_) {
      updateEncryption(encryption::kDictionaryPageHeader);
    }
    const int64_t headerSize =
        thriftSerializer_->serialize(&pageHeader, sink_.get(), metaEncryptor_);

    PARQUET_THROW_NOT_OK(sink_->Write(outputDataBuffer, outputDataLen));

    totalUncompressedSize_ += uncompressedSize + headerSize;
    totalCompressedSize_ += outputDataLen + headerSize;
    ++dictEncodingStats_[page.encoding()];
    return uncompressedSize + headerSize;
  }

  void close(bool hasDictionary, bool fallback) override {
    if (metaEncryptor_ != nullptr) {
      updateEncryption(encryption::kColumnMetaData);
    }

    // Serialized page writer does not need to adjust page offsets.
    finishPageIndexes(0);

    // Index_page_offset = -1 since they are not supported.
    metadata_->finish(
        numValues_,
        dictionaryPageOffset_,
        -1,
        dataPageOffset_,
        totalCompressedSize_,
        totalUncompressedSize_,
        hasDictionary,
        fallback,
        dictEncodingStats_,
        dataEncodingStats_,
        metaEncryptor_);
    // Write metadata at end of column chunk.
    metadata_->writeTo(sink_.get());
  }

  /**
   * Compress a buffer.
   */
  void compress(const Buffer& srcBuffer, ResizableBuffer* destBuffer) override {
    VELOX_DCHECK_NOT_NULL(compressor_);

    // Compress the data.
    int64_t maxCompressedSize =
        compressor_->maxCompressedLen(srcBuffer.size(), srcBuffer.data());

    // Use Arrow::Buffer::shrink_to_fit = false.
    // Underlying buffer only keeps growing. Resize to a smaller size does not
    // reallocate.
    PARQUET_THROW_NOT_OK(destBuffer->Resize(maxCompressedSize, false));

    PARQUET_ASSIGN_OR_THROW(
        int64_t compressedSize,
        compressor_->compress(
            srcBuffer.size(),
            srcBuffer.data(),
            maxCompressedSize,
            destBuffer->mutable_data()));
    PARQUET_THROW_NOT_OK(destBuffer->Resize(compressedSize, false));
  }

  int64_t writeDataPage(const DataPage& page) override {
    const int64_t uncompressedSize = page.uncompressedSize();
    std::shared_ptr<Buffer> compressedData = page.buffer();
    const uint8_t* outputDataBuffer = compressedData->data();
    int32_t outputDataLen = static_cast<int32_t>(compressedData->size());

    if (dataEncryptor_.get()) {
      PARQUET_THROW_NOT_OK(encryptionBuffer_->Resize(
          dataEncryptor_->ciphertextSizeDelta() + outputDataLen, false));
      updateEncryption(encryption::kDataPage);
      outputDataLen = dataEncryptor_->encrypt(
          compressedData->data(),
          outputDataLen,
          encryptionBuffer_->mutable_data());
      outputDataBuffer = encryptionBuffer_->data();
    }

    facebook::velox::parquet::thrift::PageHeader pageHeader;
    pageHeader.__set_uncompressed_page_size(
        static_cast<int32_t>(uncompressedSize));
    pageHeader.__set_compressed_page_size(static_cast<int32_t>(outputDataLen));

    if (pageChecksumVerification_) {
      uint32_t crc32 =
          internal::crc32(/* prev */ 0, outputDataBuffer, outputDataLen);
      pageHeader.__set_crc(static_cast<int32_t>(crc32));
    }

    if (page.type() == PageType::kDataPage) {
      const DataPageV1& v1Page = checked_cast<const DataPageV1&>(page);
      setDataPageHeader(pageHeader, v1Page);
    } else if (page.type() == PageType::kDataPageV2) {
      const DataPageV2& v2Page = checked_cast<const DataPageV2&>(page);
      setDataPageV2Header(pageHeader, v2Page);
    } else {
      throw ParquetException("Unexpected page type");
    }

    PARQUET_ASSIGN_OR_THROW(int64_t startPos, sink_->Tell());
    if (pageOrdinal_ == 0) {
      dataPageOffset_ = startPos;
    }

    if (metaEncryptor_) {
      updateEncryption(encryption::kDataPageHeader);
    }
    const int64_t headerSize =
        thriftSerializer_->serialize(&pageHeader, sink_.get(), metaEncryptor_);
    PARQUET_THROW_NOT_OK(sink_->Write(outputDataBuffer, outputDataLen));

    /// Collect page index.
    if (columnIndexBuilder_ != nullptr) {
      columnIndexBuilder_->addPage(page.statistics());
    }
    if (offsetIndexBuilder_ != nullptr) {
      const int64_t compressedSize = outputDataLen + headerSize;
      if (compressedSize > std::numeric_limits<int32_t>::max()) {
        throw ParquetException("Compressed page size overflows to INT32_MAX.");
      }
      if (!page.firstRowIndex().has_value()) {
        throw ParquetException("First row index is not set in data page.");
      }
      /// startPos is a relative offset in the buffered mode. It should be
      /// adjusted via OffsetIndexBuilder::finish() after BufferedPageWriter
      /// has flushed all data pages.
      offsetIndexBuilder_->addPage(
          startPos,
          static_cast<int32_t>(compressedSize),
          *page.firstRowIndex());
    }

    totalUncompressedSize_ += uncompressedSize + headerSize;
    totalCompressedSize_ += outputDataLen + headerSize;
    numValues_ += page.numValues();
    ++dataEncodingStats_[page.encoding()];
    ++pageOrdinal_;
    return uncompressedSize + headerSize;
  }

  void setDataPageHeader(
      facebook::velox::parquet::thrift::PageHeader& pageHeader,
      const DataPageV1& page) {
    facebook::velox::parquet::thrift::DataPageHeader dataPageHeader;
    dataPageHeader.__set_num_values(page.numValues());
    dataPageHeader.__set_encoding(toThrift(page.encoding()));
    dataPageHeader.__set_definition_level_encoding(
        toThrift(page.definitionLevelEncoding()));
    dataPageHeader.__set_repetition_level_encoding(
        toThrift(page.repetitionLevelEncoding()));

    // Write page statistics only when page index is not enabled.
    if (columnIndexBuilder_ == nullptr) {
      dataPageHeader.__set_statistics(toThrift(page.statistics()));
    }

    pageHeader.__set_type(
        facebook::velox::parquet::thrift::PageType::DATA_PAGE);
    pageHeader.__set_data_page_header(dataPageHeader);
  }

  void setDataPageV2Header(
      facebook::velox::parquet::thrift::PageHeader& pageHeader,
      const DataPageV2& page) {
    facebook::velox::parquet::thrift::DataPageHeaderV2 dataPageHeader;
    dataPageHeader.__set_num_values(page.numValues());
    dataPageHeader.__set_num_nulls(page.numNulls());
    dataPageHeader.__set_num_rows(page.numRows());
    dataPageHeader.__set_encoding(toThrift(page.encoding()));

    dataPageHeader.__set_definition_levels_byte_length(
        page.definitionLevelsByteLength());
    dataPageHeader.__set_repetition_levels_byte_length(
        page.repetitionLevelsByteLength());

    dataPageHeader.__set_is_compressed(page.isCompressed());

    // Write page statistics only when page index is not enabled.
    if (columnIndexBuilder_ == nullptr) {
      dataPageHeader.__set_statistics(toThrift(page.statistics()));
    }

    pageHeader.__set_type(
        facebook::velox::parquet::thrift::PageType::DATA_PAGE_V2);
    pageHeader.__set_data_page_header_v2(dataPageHeader);
  }

  /// \brief Finish page index builders and update the stream offset to adjust
  /// page offsets.
  void finishPageIndexes(int64_t finalPosition) {
    if (columnIndexBuilder_ != nullptr) {
      columnIndexBuilder_->finish();
    }
    if (offsetIndexBuilder_ != nullptr) {
      offsetIndexBuilder_->finish(finalPosition);
    }
  }

  bool hasCompressor() override {
    return (compressor_ != nullptr);
  }

  int64_t numValues() {
    return numValues_;
  }

  int64_t dictionaryPageOffset() {
    return dictionaryPageOffset_;
  }

  int64_t dataPageOffset() {
    return dataPageOffset_;
  }

  int64_t totalCompressedSize() {
    return totalCompressedSize_;
  }

  int64_t totalUncompressedSize() {
    return totalUncompressedSize_;
  }

  int64_t totalCompressedBytesWritten() const override {
    return totalCompressedSize_;
  }

  bool pageChecksumVerification() {
    return pageChecksumVerification_;
  }

 private:
  // To allow updateEncryption on close.
  friend class BufferedPageWriter;

  void initEncryption() {
    // Prepare the AAD for quick update later.
    if (dataEncryptor_ != nullptr) {
      dataPageAad_ = encryption::createModuleAad(
          dataEncryptor_->fileAad(),
          encryption::kDataPage,
          rowGroupOrdinal_,
          columnOrdinal_,
          kNonPageOrdinal);
    }
    if (metaEncryptor_ != nullptr) {
      dataPageHeaderAad_ = encryption::createModuleAad(
          metaEncryptor_->fileAad(),
          encryption::kDataPageHeader,
          rowGroupOrdinal_,
          columnOrdinal_,
          kNonPageOrdinal);
    }
  }

  void updateEncryption(int8_t moduleType) {
    switch (moduleType) {
      case encryption::kColumnMetaData: {
        metaEncryptor_->updateAad(
            encryption::createModuleAad(
                metaEncryptor_->fileAad(),
                moduleType,
                rowGroupOrdinal_,
                columnOrdinal_,
                kNonPageOrdinal));
        break;
      }
      case encryption::kDataPage: {
        encryption::quickUpdatePageAad(pageOrdinal_, &dataPageAad_);
        dataEncryptor_->updateAad(dataPageAad_);
        break;
      }
      case encryption::kDataPageHeader: {
        encryption::quickUpdatePageAad(pageOrdinal_, &dataPageHeaderAad_);
        metaEncryptor_->updateAad(dataPageHeaderAad_);
        break;
      }
      case encryption::kDictionaryPageHeader: {
        metaEncryptor_->updateAad(
            encryption::createModuleAad(
                metaEncryptor_->fileAad(),
                moduleType,
                rowGroupOrdinal_,
                columnOrdinal_,
                kNonPageOrdinal));
        break;
      }
      case encryption::kDictionaryPage: {
        dataEncryptor_->updateAad(
            encryption::createModuleAad(
                dataEncryptor_->fileAad(),
                moduleType,
                rowGroupOrdinal_,
                columnOrdinal_,
                kNonPageOrdinal));
        break;
      }
      default:
        throw ParquetException("Unknown module type in updateEncryption");
    }
  }

  std::shared_ptr<ArrowOutputStream> sink_;
  ColumnChunkMetaDataBuilder* metadata_;
  MemoryPool* pool_;
  int64_t numValues_;
  int64_t dictionaryPageOffset_;
  int64_t dataPageOffset_;
  // The uncompressed page size the page writer has already written.
  int64_t totalUncompressedSize_;
  // The compressed page size the page writer has already written.
  // If the column is UNCOMPRESSED, the size would be equal to
  // totalUncompressedSize_.
  int64_t totalCompressedSize_;
  int32_t pageOrdinal_;
  int16_t rowGroupOrdinal_;
  int16_t columnOrdinal_;
  bool pageChecksumVerification_;

  std::unique_ptr<ThriftSerializer> thriftSerializer_;

  // Compression codec to use.
  std::unique_ptr<util::Codec> compressor_;

  std::string dataPageAad_;
  std::string dataPageHeaderAad_;

  std::shared_ptr<Encryptor> metaEncryptor_;
  std::shared_ptr<Encryptor> dataEncryptor_;

  std::shared_ptr<ResizableBuffer> encryptionBuffer_;

  std::map<Encoding::type, int32_t> dictEncodingStats_;
  std::map<Encoding::type, int32_t> dataEncodingStats_;

  ColumnIndexBuilder* columnIndexBuilder_;
  OffsetIndexBuilder* offsetIndexBuilder_;
};

// This implementation of the PageWriter writes to the final sink on close.
class BufferedPageWriter : public PageWriter {
 public:
  BufferedPageWriter(
      std::shared_ptr<ArrowOutputStream> sink,
      Compression::type codec,
      ColumnChunkMetaDataBuilder* metadata,
      int16_t rowGroupOrdinal,
      int16_t currentColumnOrdinal,
      bool usePageChecksumVerification,
      MemoryPool* pool = ::arrow::default_memory_pool(),
      std::shared_ptr<Encryptor> metaEncryptor = nullptr,
      std::shared_ptr<Encryptor> dataEncryptor = nullptr,
      ColumnIndexBuilder* columnIndexBuilder = nullptr,
      OffsetIndexBuilder* offsetIndexBuilder = nullptr,
      const CodecOptions& codecOptions = CodecOptions{})
      : finalSink_(std::move(sink)),
        metadata_(metadata),
        hasDictionaryPages_(false) {
    inMemorySink_ = createOutputStream(pool);
    pager_ = std::make_unique<SerializedPageWriter>(
        inMemorySink_,
        codec,
        metadata,
        rowGroupOrdinal,
        currentColumnOrdinal,
        usePageChecksumVerification,
        pool,
        std::move(metaEncryptor),
        std::move(dataEncryptor),
        columnIndexBuilder,
        offsetIndexBuilder,
        codecOptions);
  }

  int64_t writeDictionaryPage(const DictionaryPage& page) override {
    hasDictionaryPages_ = true;
    return pager_->writeDictionaryPage(page);
  }

  void close(bool hasDictionary, bool fallback) override {
    if (pager_->metaEncryptor_ != nullptr) {
      pager_->updateEncryption(encryption::kColumnMetaData);
    }
    // Index_page_offset = -1 since they are not supported.
    PARQUET_ASSIGN_OR_THROW(int64_t finalPosition, finalSink_->Tell());
    // Dictionary page offset should be 0 iff there are no dictionary pages.
    auto dictionaryPageOffset = hasDictionaryPages_
        ? pager_->dictionaryPageOffset() + finalPosition
        : 0;
    metadata_->finish(
        pager_->numValues(),
        dictionaryPageOffset,
        -1,
        pager_->dataPageOffset() + finalPosition,
        pager_->totalCompressedSize(),
        pager_->totalUncompressedSize(),
        hasDictionary,
        fallback,
        pager_->dictEncodingStats_,
        pager_->dataEncodingStats_,
        pager_->metaEncryptor_);

    // Write metadata at end of column chunk.
    metadata_->writeTo(inMemorySink_.get());

    // Buffered page writer needs to adjust page offsets.
    pager_->finishPageIndexes(finalPosition);

    // Flush everything to the serialized sink.
    PARQUET_ASSIGN_OR_THROW(auto buffer, inMemorySink_->Finish());
    PARQUET_THROW_NOT_OK(finalSink_->Write(buffer));
  }

  int64_t writeDataPage(const DataPage& page) override {
    return pager_->writeDataPage(page);
  }

  void compress(const Buffer& srcBuffer, ResizableBuffer* destBuffer) override {
    pager_->compress(srcBuffer, destBuffer);
  }

  bool hasCompressor() override {
    return pager_->hasCompressor();
  }

  int64_t totalCompressedBytesWritten() const override {
    return pager_->totalCompressedBytesWritten();
  }

 private:
  std::shared_ptr<ArrowOutputStream> finalSink_;
  ColumnChunkMetaDataBuilder* metadata_;
  std::shared_ptr<::arrow::io::BufferOutputStream> inMemorySink_;
  std::unique_ptr<SerializedPageWriter> pager_;
  bool hasDictionaryPages_;
};

std::unique_ptr<PageWriter> PageWriter::open(
    std::shared_ptr<ArrowOutputStream> sink,
    Compression::type codec,
    ColumnChunkMetaDataBuilder* metadata,
    int16_t rowGroupOrdinal,
    int16_t columnChunkOrdinal,
    MemoryPool* pool,
    bool bufferedRowGroup,
    std::shared_ptr<Encryptor> metaEncryptor,
    std::shared_ptr<Encryptor> dataEncryptor,
    bool pageWriteChecksumEnabled,
    ColumnIndexBuilder* columnIndexBuilder,
    OffsetIndexBuilder* offsetIndexBuilder,
    const CodecOptions& codecOptions) {
  if (bufferedRowGroup) {
    return std::unique_ptr<PageWriter>(new BufferedPageWriter(
        std::move(sink),
        codec,
        metadata,
        rowGroupOrdinal,
        columnChunkOrdinal,
        pageWriteChecksumEnabled,
        pool,
        std::move(metaEncryptor),
        std::move(dataEncryptor),
        columnIndexBuilder,
        offsetIndexBuilder,
        codecOptions));
  } else {
    return std::unique_ptr<PageWriter>(new SerializedPageWriter(
        std::move(sink),
        codec,
        metadata,
        rowGroupOrdinal,
        columnChunkOrdinal,
        pageWriteChecksumEnabled,
        pool,
        std::move(metaEncryptor),
        std::move(dataEncryptor),
        columnIndexBuilder,
        offsetIndexBuilder,
        codecOptions));
  }
}

std::unique_ptr<PageWriter> PageWriter::open(
    std::shared_ptr<ArrowOutputStream> sink,
    Compression::type codec,
    int compressionLevel,
    ColumnChunkMetaDataBuilder* metadata,
    int16_t rowGroupOrdinal,
    int16_t columnChunkOrdinal,
    MemoryPool* pool,
    bool bufferedRowGroup,
    std::shared_ptr<Encryptor> metaEncryptor,
    std::shared_ptr<Encryptor> dataEncryptor,
    bool pageWriteChecksumEnabled,
    ColumnIndexBuilder* columnIndexBuilder,
    OffsetIndexBuilder* offsetIndexBuilder) {
  return PageWriter::open(
      sink,
      codec,
      metadata,
      rowGroupOrdinal,
      columnChunkOrdinal,
      pool,
      bufferedRowGroup,
      metaEncryptor,
      dataEncryptor,
      pageWriteChecksumEnabled,
      columnIndexBuilder,
      offsetIndexBuilder,
      CodecOptions{compressionLevel});
}
// ----------------------------------------------------------------------.
// ColumnWriter.

const std::shared_ptr<WriterProperties>& defaultWriterProperties() {
  static std::shared_ptr<WriterProperties> defaultWriterProperties =
      WriterProperties::Builder().build();
  return defaultWriterProperties;
}

class ColumnWriterImpl {
 public:
  ColumnWriterImpl(
      ColumnChunkMetaDataBuilder* metadata,
      std::unique_ptr<PageWriter> pager,
      const bool useDictionary,
      Encoding::type encoding,
      const WriterProperties* properties)
      : metadata_(metadata),
        descr_(metadata->descr()),
        levelInfo_(computeLevelInfo(metadata->descr())),
        pager_(std::move(pager)),
        hasDictionary_(useDictionary),
        encoding_(encoding),
        properties_(properties),
        allocator_(properties->memoryPool()),
        numBufferedValues_(0),
        numBufferedEncodedValues_(0),
        numBufferedNulls_(0),
        numBufferedRows_(0),
        rowsWritten_(0),
        totalBytesWritten_(0),
        totalCompressedBytes_(0),
        closed_(false),
        fallback_(false),
        definitionLevelsSink_(allocator_),
        repetitionLevelsSink_(allocator_) {
    definitionLevelsRle_ = std::static_pointer_cast<ResizableBuffer>(
        allocateBuffer(allocator_, 0));
    repetitionLevelsRle_ = std::static_pointer_cast<ResizableBuffer>(
        allocateBuffer(allocator_, 0));
    uncompressedData_ = std::static_pointer_cast<ResizableBuffer>(
        allocateBuffer(allocator_, 0));

    if (pager_->hasCompressor()) {
      compressorTempBuffer_ = std::static_pointer_cast<ResizableBuffer>(
          allocateBuffer(allocator_, 0));
    }
  }

  virtual ~ColumnWriterImpl() = default;

  int64_t close();

 protected:
  virtual std::shared_ptr<Buffer> getValuesBuffer() = 0;

  // Serializes Dictionary Page if enabled.
  virtual void writeDictionaryPage() = 0;

  // Plain-encoded statistics of the current page.
  virtual EncodedStatistics getPageStatistics() = 0;

  // Plain-encoded statistics of the whole chunk.
  virtual EncodedStatistics getChunkStatistics() = 0;

  // Merges page statistics into chunk statistics, then resets the values.
  virtual void resetPageStatistics() = 0;

  // Adds Data Pages to an in memory buffer in dictionary encoding mode.
  // Serializes the Data Pages in other encoding modes.
  void addDataPage();

  void buildDataPageV1(
      int64_t definitionLevelsRleSize,
      int64_t repetitionLevelsRleSize,
      int64_t uncompressedSize,
      const std::shared_ptr<Buffer>& values);

  void buildDataPageV2(
      int64_t definitionLevelsRleSize,
      int64_t repetitionLevelsRleSize,
      int64_t uncompressedSize,
      const std::shared_ptr<Buffer>& values);

  // Serializes Data Pages.
  void writeDataPage(const DataPage& page) {
    totalBytesWritten_ += pager_->writeDataPage(page);
  }

  // Write multiple definition levels.
  void writeDefinitionLevels(int64_t numLevels, const int16_t* levels) {
    VELOX_DCHECK(!closed_);
    PARQUET_THROW_NOT_OK(
        definitionLevelsSink_.Append(levels, sizeof(int16_t) * numLevels));
  }

  // Write multiple repetition levels.
  void writeRepetitionLevels(int64_t numLevels, const int16_t* levels) {
    VELOX_DCHECK(!closed_);
    PARQUET_THROW_NOT_OK(
        repetitionLevelsSink_.Append(levels, sizeof(int16_t) * numLevels));
  }

  // RLE encode the src_buffer into dest_buffer and return the encoded size.
  int64_t rleEncodeLevels(
      const void* srcBuffer,
      ResizableBuffer* destBuffer,
      int16_t maxLevel,
      bool includeLengthPrefix = true);

  // Serialize the buffered Data Pages.
  void flushBufferedDataPages();

  ColumnChunkMetaDataBuilder* metadata_;
  const ColumnDescriptor* descr_;
  // Scratch buffer if validity bits need to be recalculated.
  std::shared_ptr<ResizableBuffer> bitsBuffer_;
  const LevelInfo levelInfo_;

  std::unique_ptr<PageWriter> pager_;

  bool hasDictionary_;
  Encoding::type encoding_;
  const WriterProperties* properties_;

  LevelEncoder levelEncoder_;

  MemoryPool* allocator_;

  // The total number of values stored in the data page. This is the maximum of
  // the number of encoded definition levels or encoded values. For
  // non-repeated, required columns, this is equal to the number of encoded
  // values. For repeated or optional values, there may be fewer data values
  // than levels, and this tells you how many encoded levels there are in that
  // case.
  int64_t numBufferedValues_;

  // The total number of stored values in the data page. For repeated or
  // optional values, this number may be lower than numBufferedValues_.
  int64_t numBufferedEncodedValues_;

  // The total number of nulls stored in the data page.
  int64_t numBufferedNulls_;

  // Total number of rows buffered in the data page.
  int64_t numBufferedRows_;

  // Total number of rows written with this ColumnWriter.
  int64_t rowsWritten_;

  // Records the total number of uncompressed bytes written by the serializer.
  int64_t totalBytesWritten_;

  // Records the current number of compressed bytes in a column.
  // These bytes are unwritten to `pager_` yet.
  int64_t totalCompressedBytes_;

  // Flag to check if the Writer has been closed.
  bool closed_;

  // Flag to infer if dictionary encoding has fallen back to PLAIN.
  bool fallback_;

  ::arrow::BufferBuilder definitionLevelsSink_;
  ::arrow::BufferBuilder repetitionLevelsSink_;

  std::shared_ptr<ResizableBuffer> definitionLevelsRle_;
  std::shared_ptr<ResizableBuffer> repetitionLevelsRle_;

  std::shared_ptr<ResizableBuffer> uncompressedData_;
  std::shared_ptr<ResizableBuffer> compressorTempBuffer_;

  std::vector<std::unique_ptr<DataPage>> dataPages_;

 private:
  void initSinks() {
    definitionLevelsSink_.Rewind(0);
    repetitionLevelsSink_.Rewind(0);
  }

  // Concatenate the encoded levels and values into one buffer.
  void concatenateBuffers(
      int64_t definitionLevelsRleSize,
      int64_t repetitionLevelsRleSize,
      const std::shared_ptr<Buffer>& values,
      uint8_t* combined) {
    memcpy(combined, repetitionLevelsRle_->data(), repetitionLevelsRleSize);
    combined += repetitionLevelsRleSize;
    memcpy(combined, definitionLevelsRle_->data(), definitionLevelsRleSize);
    combined += definitionLevelsRleSize;
    memcpy(combined, values->data(), values->size());
  }
};

// Return the size of the encoded buffer.
int64_t ColumnWriterImpl::rleEncodeLevels(
    const void* srcBuffer,
    ResizableBuffer* destBuffer,
    int16_t maxLevel,
    bool includeLengthPrefix) {
  // V1 DataPage includes the length of the RLE level as a prefix.
  int32_t prefixSize = includeLengthPrefix ? sizeof(int32_t) : 0;

  // TODO: This only works due to some RLE specifics.
  int64_t rleSize =
      LevelEncoder::maxBufferSize(
          Encoding::kRle, maxLevel, static_cast<int>(numBufferedValues_)) +
      prefixSize;

  // Use Arrow::Buffer::shrink_to_fit = false.
  // Underlying buffer only keeps growing. Resize to a smaller size does not.
  // Reallocate.
  PARQUET_THROW_NOT_OK(destBuffer->Resize(rleSize, false));

  levelEncoder_.init(
      Encoding::kRle,
      maxLevel,
      static_cast<int>(numBufferedValues_),
      destBuffer->mutable_data() + prefixSize,
      static_cast<int>(destBuffer->size() - prefixSize));
  VELOX_DEBUG_ONLY int encoded = levelEncoder_.encode(
      static_cast<int>(numBufferedValues_),
      reinterpret_cast<const int16_t*>(srcBuffer));
  VELOX_DCHECK_EQ(encoded, numBufferedValues_);

  if (includeLengthPrefix) {
    reinterpret_cast<int32_t*>(destBuffer->mutable_data())[0] =
        levelEncoder_.len();
  }

  return levelEncoder_.len() + prefixSize;
}

void ColumnWriterImpl::addDataPage() {
  int64_t definitionLevelsRleSize = 0;
  int64_t repetitionLevelsRleSize = 0;

  std::shared_ptr<Buffer> values = getValuesBuffer();
  bool isV1DataPage =
      properties_->dataPageVersion() == ParquetDataPageVersion::V1;

  if (descr_->maxDefinitionLevel() > 0) {
    definitionLevelsRleSize = rleEncodeLevels(
        definitionLevelsSink_.data(),
        definitionLevelsRle_.get(),
        descr_->maxDefinitionLevel(),
        isV1DataPage);
  }

  if (descr_->maxRepetitionLevel() > 0) {
    repetitionLevelsRleSize = rleEncodeLevels(
        repetitionLevelsSink_.data(),
        repetitionLevelsRle_.get(),
        descr_->maxRepetitionLevel(),
        isV1DataPage);
  }

  int64_t uncompressedSize =
      definitionLevelsRleSize + repetitionLevelsRleSize + values->size();

  if (isV1DataPage) {
    buildDataPageV1(
        definitionLevelsRleSize,
        repetitionLevelsRleSize,
        uncompressedSize,
        values);
  } else {
    buildDataPageV2(
        definitionLevelsRleSize,
        repetitionLevelsRleSize,
        uncompressedSize,
        values);
  }

  // Re-initialize the sinks for next Page.
  initSinks();
  numBufferedValues_ = 0;
  numBufferedEncodedValues_ = 0;
  numBufferedRows_ = 0;
  numBufferedNulls_ = 0;
}

void ColumnWriterImpl::buildDataPageV1(
    int64_t definitionLevelsRleSize,
    int64_t repetitionLevelsRleSize,
    int64_t uncompressedSize,
    const std::shared_ptr<Buffer>& values) {
  // Use Arrow::Buffer::shrink_to_fit = false.
  // Underlying buffer only keeps growing. Resize to a smaller size does not.
  // Reallocate.
  PARQUET_THROW_NOT_OK(uncompressedData_->Resize(uncompressedSize, false));
  concatenateBuffers(
      definitionLevelsRleSize,
      repetitionLevelsRleSize,
      values,
      uncompressedData_->mutable_data());

  EncodedStatistics pageStats = getPageStatistics();
  pageStats.applyStatSizeLimits(properties_->maxStatisticsSize(descr_->path()));
  pageStats.setIsSigned(SortOrder::kSigned == descr_->sortOrder());
  resetPageStatistics();

  std::shared_ptr<Buffer> compressedData;
  if (pager_->hasCompressor()) {
    pager_->compress(*(uncompressedData_.get()), compressorTempBuffer_.get());
    compressedData = compressorTempBuffer_;
  } else {
    compressedData = uncompressedData_;
  }

  int32_t numValues = static_cast<int32_t>(numBufferedValues_);
  int64_t firstRowIndex = rowsWritten_ - numBufferedRows_;

  // Write the page to OutputStream eagerly if there is no dictionary or.
  // If dictionary encoding has fallen back to PLAIN.
  if (hasDictionary_ &&
      !fallback_) { // Save pages until end of dictionary encoding
    PARQUET_ASSIGN_OR_THROW(
        auto compressedDataCopy,
        compressedData->CopySlice(0, compressedData->size(), allocator_));
    std::unique_ptr<DataPage> pagePtr = std::make_unique<DataPageV1>(
        compressedDataCopy,
        numValues,
        encoding_,
        Encoding::kRle,
        Encoding::kRle,
        uncompressedSize,
        pageStats,
        firstRowIndex);
    totalCompressedBytes_ +=
        pagePtr->size() + sizeof(facebook::velox::parquet::thrift::PageHeader);

    dataPages_.push_back(std::move(pagePtr));
  } else { // Eagerly write pages
    DataPageV1 page(
        compressedData,
        numValues,
        encoding_,
        Encoding::kRle,
        Encoding::kRle,
        uncompressedSize,
        pageStats,
        firstRowIndex);
    writeDataPage(page);
  }
}

void ColumnWriterImpl::buildDataPageV2(
    int64_t definitionLevelsRleSize,
    int64_t repetitionLevelsRleSize,
    int64_t uncompressedSize,
    const std::shared_ptr<Buffer>& values) {
  // Compress the values if needed. Repetition and definition levels are
  // uncompressed in V2.
  std::shared_ptr<Buffer> compressedValues;
  if (pager_->hasCompressor()) {
    pager_->compress(*values, compressorTempBuffer_.get());
    compressedValues = compressorTempBuffer_;
  } else {
    compressedValues = values;
  }

  // Concatenate uncompressed levels and the possibly compressed values.
  int64_t combinedSize = definitionLevelsRleSize + repetitionLevelsRleSize +
      compressedValues->size();
  std::shared_ptr<ResizableBuffer> combined =
      allocateBuffer(allocator_, combinedSize);

  concatenateBuffers(
      definitionLevelsRleSize,
      repetitionLevelsRleSize,
      compressedValues,
      combined->mutable_data());

  EncodedStatistics pageStats = getPageStatistics();
  pageStats.applyStatSizeLimits(properties_->maxStatisticsSize(descr_->path()));
  pageStats.setIsSigned(SortOrder::kSigned == descr_->sortOrder());
  resetPageStatistics();

  int32_t numValues = static_cast<int32_t>(numBufferedValues_);
  int32_t nullCount = static_cast<int32_t>(numBufferedNulls_);
  int32_t numRows = static_cast<int32_t>(numBufferedRows_);
  int32_t defLevelsByteLength = static_cast<int32_t>(definitionLevelsRleSize);
  int32_t repLevelsByteLength = static_cast<int32_t>(repetitionLevelsRleSize);
  int64_t firstRowIndex = rowsWritten_ - numBufferedRows_;

  // pageStats.null_count is not set when page_statistics_ is nullptr. It is
  // only used here for safety check.
  VELOX_DCHECK(!pageStats.hasNullCount || pageStats.nullCount == nullCount);

  // Write the page to OutputStream eagerly if there is no dictionary or.
  // If dictionary encoding has fallen back to PLAIN.
  if (hasDictionary_ &&
      !fallback_) { // Save pages until end of dictionary encoding
    PARQUET_ASSIGN_OR_THROW(
        auto dataCopy, combined->CopySlice(0, combined->size(), allocator_));
    std::unique_ptr<DataPage> pagePtr = std::make_unique<DataPageV2>(
        combined,
        numValues,
        nullCount,
        numRows,
        encoding_,
        defLevelsByteLength,
        repLevelsByteLength,
        uncompressedSize,
        pager_->hasCompressor(),
        pageStats,
        firstRowIndex);
    totalCompressedBytes_ +=
        pagePtr->size() + sizeof(facebook::velox::parquet::thrift::PageHeader);
    dataPages_.push_back(std::move(pagePtr));
  } else {
    DataPageV2 page(
        combined,
        numValues,
        nullCount,
        numRows,
        encoding_,
        defLevelsByteLength,
        repLevelsByteLength,
        uncompressedSize,
        pager_->hasCompressor(),
        pageStats,
        firstRowIndex);
    writeDataPage(page);
  }
}

int64_t ColumnWriterImpl::close() {
  if (!closed_) {
    closed_ = true;
    if (hasDictionary_ && !fallback_) {
      writeDictionaryPage();
    }

    flushBufferedDataPages();

    EncodedStatistics chunkStatistics = getChunkStatistics();
    chunkStatistics.applyStatSizeLimits(
        properties_->maxStatisticsSize(descr_->path()));
    chunkStatistics.setIsSigned(SortOrder::kSigned == descr_->sortOrder());

    // Write stats only if the column has at least one row written.
    if (rowsWritten_ > 0 && chunkStatistics.isSet()) {
      metadata_->setStatistics(chunkStatistics);
    }
    pager_->close(hasDictionary_, fallback_);
  }

  return totalBytesWritten_;
}

void ColumnWriterImpl::flushBufferedDataPages() {
  // Write all outstanding data to a new page.
  if (numBufferedValues_ > 0) {
    addDataPage();
  }
  for (const auto& pagePtr : dataPages_) {
    writeDataPage(*pagePtr);
  }
  dataPages_.clear();
  totalCompressedBytes_ = 0;
}

// ----------------------------------------------------------------------.
// TypedColumnWriter.

template <typename Action>
inline void doInBatches(int64_t total, int64_t batchSize, Action&& action) {
  int64_t numBatches = static_cast<int>(total / batchSize);
  for (int round = 0; round < numBatches; round++) {
    action(round * batchSize, batchSize, true);
  }
  // Write the remaining values.
  if (total % batchSize > 0) {
    action(numBatches * batchSize, total % batchSize, true);
  }
}

template <typename Action>
inline void doInBatches(
    const int16_t* defLevels,
    const int16_t* repLevels,
    int64_t numLevels,
    int64_t batchSize,
    Action&& action,
    bool pagesChangeOnRecordBoundaries) {
  if (!pagesChangeOnRecordBoundaries || !repLevels) {
    // If repLevels is null, then we are writing a non-repeated column.
    // In this case, every record contains only one level.
    return doInBatches(numLevels, batchSize, std::forward<Action>(action));
  }

  int64_t offset = 0;
  while (offset < numLevels) {
    int64_t endOffset = std::min(offset + batchSize, numLevels);

    // Find next record boundary (i.e. repLevel = 0).
    while (endOffset < numLevels && repLevels[endOffset] != 0) {
      endOffset++;
    }

    if (endOffset < numLevels) {
      // This is not the last chunk of batch and endOffset is a record
      // boundary. It is a good chance to check the page size.
      action(offset, endOffset - offset, true);
    } else {
      VELOX_DCHECK_EQ(endOffset, numLevels);
      // This is the last chunk of batch, and we do not know whether endOffset
      // is a record boundary. Find the offset to beginning of last record in
      // this chunk, so we can check page size.
      int64_t lastRecordBeginOffset = numLevels - 1;
      while (lastRecordBeginOffset >= offset &&
             repLevels[lastRecordBeginOffset] != 0) {
        lastRecordBeginOffset--;
      }

      if (offset < lastRecordBeginOffset) {
        // We have found the beginning of last record and can check page size.
        action(offset, lastRecordBeginOffset - offset, true);
        offset = lastRecordBeginOffset;
      }

      // There is no record boundary in this chunk and cannot check page size.
      action(offset, endOffset - offset, false);
    }

    offset = endOffset;
  }
}

bool dictionaryDirectWriteSupported(const ::arrow::Array& array) {
  VELOX_DCHECK_EQ(
      static_cast<int>(array.type_id()),
      static_cast<int>(::arrow::Type::DICTIONARY));
  const ::arrow::DictionaryType& dictType =
      static_cast<const ::arrow::DictionaryType&>(*array.type());
  return ::arrow::is_base_binary_like(dictType.value_type()->id());
}

Status convertDictionaryToDense(
    const ::arrow::Array& array,
    MemoryPool* pool,
    std::shared_ptr<::arrow::Array>* out) {
  const ::arrow::DictionaryType& dictType =
      static_cast<const ::arrow::DictionaryType&>(*array.type());

  ::arrow::compute::ExecContext ctx(pool);
  ARROW_ASSIGN_OR_RAISE(
      Datum castOutput,
      ::arrow::compute::Cast(
          array.data(),
          dictType.value_type(),
          ::arrow::compute::CastOptions::Safe(),
          &ctx));
  *out = castOutput.make_array();
  return Status::OK();
}

static inline bool isDictionaryEncoding(Encoding::type encoding) {
  return encoding == Encoding::kPlainDictionary;
}

template <typename DType>
class TypedColumnWriterImpl : public ColumnWriterImpl,
                              public TypedColumnWriter<DType> {
 public:
  using T = typename DType::CType;

  TypedColumnWriterImpl(
      ColumnChunkMetaDataBuilder* metadata,
      std::unique_ptr<PageWriter> pager,
      const bool useDictionary,
      Encoding::type encoding,
      const WriterProperties* properties)
      : ColumnWriterImpl(
            metadata,
            std::move(pager),
            useDictionary,
            encoding,
            properties) {
    currentEncoder_ = makeEncoder(
        DType::typeNum,
        encoding,
        useDictionary,
        descr_,
        properties->memoryPool());

    // We have to dynamic_cast as some compilers don't want to static_cast
    // through virtual inheritance.
    currentValueEncoder_ =
        dynamic_cast<TypedEncoder<DType>*>(currentEncoder_.get());

    // Will be null if not using dictionary, but that's ok.
    currentDictEncoder_ =
        dynamic_cast<DictEncoder<DType>*>(currentEncoder_.get());

    if (properties->statisticsEnabled(descr_->path()) &&
        (SortOrder::kUnknown != descr_->sortOrder())) {
      pageStatistics_ = makeStatistics<DType>(descr_, allocator_);
      chunkStatistics_ = makeStatistics<DType>(descr_, allocator_);
    }
    pagesChangeOnRecordBoundaries_ =
        properties->dataPageVersion() == ParquetDataPageVersion::V2 ||
        properties->pageIndexEnabled(descr_->path());
  }

  int64_t close() override {
    return ColumnWriterImpl::close();
  }

  int64_t writeBatch(
      int64_t numValues,
      const int16_t* defLevels,
      const int16_t* repLevels,
      const T* values) override {
    // We check for DataPage limits only after we have inserted the values. If
    // a user writes a large number of values, the DataPage size can be much
    // above the limit. The purpose of this chunking is to bound this. Even if
    // a user writes large number of values, the chunking will ensure the
    // addDataPage() is called at a reasonable pagesize limit.
    int64_t valueOffset = 0;

    auto writeChunk = [&](int64_t offset, int64_t batchSize, bool checkPage) {
      int64_t valuesToWrite = writeLevels(
          batchSize,
          addIfNotNull(defLevels, offset),
          addIfNotNull(repLevels, offset));

      // PARQUET-780.
      if (valuesToWrite > 0) {
        VELOX_DCHECK_NOT_NULL(values);
      }
      const int64_t numNulls = batchSize - valuesToWrite;
      writeValues(addIfNotNull(values, valueOffset), valuesToWrite, numNulls);
      commitWriteAndCheckPageLimit(
          batchSize, valuesToWrite, numNulls, checkPage);
      valueOffset += valuesToWrite;

      // Dictionary size checked separately from data page size since we
      // circumvent this check when writing ::arrow::DictionaryArray directly.
      checkDictionarySizeLimit();
    };
    doInBatches(
        defLevels,
        repLevels,
        numValues,
        properties_->writeBatchSize(),
        writeChunk,
        pagesChangeOnRecordBoundaries());
    return valueOffset;
  }

  void writeBatchSpaced(
      int64_t numValues,
      const int16_t* defLevels,
      const int16_t* repLevels,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      const T* values) override {
    // Like WriteBatch, but for spaced values.
    int64_t valueOffset = 0;
    auto writeChunk = [&](int64_t offset, int64_t batchSize, bool checkPage) {
      int64_t batchNumValues = 0;
      int64_t batchNumSpacedValues = 0;
      int64_t nullCount;
      maybeCalculateValidityBits(
          addIfNotNull(defLevels, offset),
          batchSize,
          &batchNumValues,
          &batchNumSpacedValues,
          &nullCount);

      writeLevelsSpaced(
          batchSize,
          addIfNotNull(defLevels, offset),
          addIfNotNull(repLevels, offset));
      if (bitsBuffer_ != nullptr) {
        writeValuesSpaced(
            addIfNotNull(values, valueOffset),
            batchNumValues,
            batchNumSpacedValues,
            bitsBuffer_->data(),
            0,
            batchSize,
            nullCount);
      } else {
        writeValuesSpaced(
            addIfNotNull(values, valueOffset),
            batchNumValues,
            batchNumSpacedValues,
            validBits,
            validBitsOffset + valueOffset,
            batchSize,
            nullCount);
      }
      commitWriteAndCheckPageLimit(
          batchSize, batchNumSpacedValues, nullCount, checkPage);
      valueOffset += batchNumSpacedValues;

      // Dictionary size checked separately from data page size since we
      // circumvent this check when writing ::arrow::DictionaryArray directly.
      checkDictionarySizeLimit();
    };
    doInBatches(
        defLevels,
        repLevels,
        numValues,
        properties_->writeBatchSize(),
        writeChunk,
        pagesChangeOnRecordBoundaries());
  }

  Status writeArrow(
      const int16_t* defLevels,
      const int16_t* repLevels,
      int64_t numLevels,
      const ::arrow::Array& leafArray,
      ArrowWriteContext* ctx,
      bool leafFieldNullable) override {
    BEGIN_PARQUET_CATCH_EXCEPTIONS
    // Leaf nulls are canonical when there is only a single null element after
    // a list and it is at the leaf.
    bool singleNullableElement =
        (levelInfo_.defLevel == levelInfo_.repeatedAncestorDefLevel + 1) &&
        leafFieldNullable;
    bool maybeParentNulls =
        levelInfo_.HasNullableValues() && !singleNullableElement;
    if (maybeParentNulls) {
      ARROW_ASSIGN_OR_RAISE(
          bitsBuffer_,
          ::arrow::AllocateResizableBuffer(
              ::arrow::bit_util::BytesForBits(properties_->writeBatchSize()),
              ctx->memoryPool));
      bitsBuffer_->ZeroPadding();
    }

    if (leafArray.type()->id() == ::arrow::Type::DICTIONARY) {
      return writeArrowDictionary(
          defLevels, repLevels, numLevels, leafArray, ctx, maybeParentNulls);
    } else {
      return writeArrowDense(
          defLevels, repLevels, numLevels, leafArray, ctx, maybeParentNulls);
    }
    END_PARQUET_CATCH_EXCEPTIONS
  }

  int64_t estimatedBufferedValueBytes() const override {
    return currentEncoder_->estimatedDataEncodedSize();
  }

 protected:
  std::shared_ptr<Buffer> getValuesBuffer() override {
    return currentEncoder_->flushValues();
  }

  // Internal function to handle direct writing of ::arrow::DictionaryArray,
  // since the standard logic concerning dictionary size limits and fallback to
  // plain encoding is circumvented.
  Status writeArrowDictionary(
      const int16_t* defLevels,
      const int16_t* repLevels,
      int64_t numLevels,
      const ::arrow::Array& array,
      ArrowWriteContext* context,
      bool maybeParentNulls);

  Status writeArrowDense(
      const int16_t* defLevels,
      const int16_t* repLevels,
      int64_t numLevels,
      const ::arrow::Array& array,
      ArrowWriteContext* context,
      bool maybeParentNulls);

  void writeDictionaryPage() override {
    VELOX_DCHECK(currentDictEncoder_);
    std::shared_ptr<ResizableBuffer> buffer = allocateBuffer(
        properties_->memoryPool(), currentDictEncoder_->dictEncodedSize());
    currentDictEncoder_->writeDict(buffer->mutable_data());

    DictionaryPage page(
        buffer,
        currentDictEncoder_->numEntries(),
        properties_->dictionaryPageEncoding());
    totalBytesWritten_ += pager_->writeDictionaryPage(page);
  }

  EncodedStatistics getPageStatistics() override {
    EncodedStatistics result;
    if (pageStatistics_)
      result = pageStatistics_->encode();
    return result;
  }

  EncodedStatistics getChunkStatistics() override {
    EncodedStatistics result;
    if (chunkStatistics_)
      result = chunkStatistics_->encode();
    return result;
  }

  void resetPageStatistics() override {
    if (chunkStatistics_ != nullptr) {
      chunkStatistics_->merge(*pageStatistics_);
      pageStatistics_->reset();
    }
  }

  Type::type type() const override {
    return descr_->physicalType();
  }

  const ColumnDescriptor* descr() const override {
    return descr_;
  }

  int64_t rowsWritten() const override {
    return rowsWritten_;
  }

  int64_t totalCompressedBytes() const override {
    return totalCompressedBytes_;
  }

  int64_t totalBytesWritten() const override {
    return totalBytesWritten_;
  }

  int64_t totalCompressedBytesWritten() const override {
    return pager_->totalCompressedBytesWritten();
  }

  const WriterProperties* properties() override {
    return properties_;
  }

  bool pagesChangeOnRecordBoundaries() const {
    return pagesChangeOnRecordBoundaries_;
  }

 private:
  using ValueEncoderType = typename EncodingTraits<DType>::Encoder;
  using TypedStats = TypedStatistics<DType>;
  std::unique_ptr<Encoder> currentEncoder_;
  // Downcasted observers of currentEncoder_.
  // The downcast is performed once as opposed to at every use since
  // dynamic_cast is so expensive, and static_cast is not available due
  // to virtual inheritance.
  ValueEncoderType* currentValueEncoder_;
  DictEncoder<DType>* currentDictEncoder_;
  std::shared_ptr<TypedStats> pageStatistics_;
  std::shared_ptr<TypedStats> chunkStatistics_;
  bool pagesChangeOnRecordBoundaries_;

  // If writing a sequence of ::arrow::DictionaryArray to the writer, we keep
  // the dictionary passed to DictEncoder<T>::putDictionary so we can check
  // subsequent array chunks to see either if materialization is required (in
  // which case we call back to the dense write path).
  std::shared_ptr<::arrow::Array> preservedDictionary_;

  int64_t writeLevels(
      int64_t numValues,
      const int16_t* defLevels,
      const int16_t* repLevels) {
    int64_t valuesToWrite = 0;
    // If the field is required and non-repeated, there are no definition
    // levels.
    if (descr_->maxDefinitionLevel() > 0) {
      for (int64_t i = 0; i < numValues; ++i) {
        if (defLevels[i] == descr_->maxDefinitionLevel()) {
          ++valuesToWrite;
        }
      }

      writeDefinitionLevels(numValues, defLevels);
    } else {
      // Required field, write all values.
      valuesToWrite = numValues;
    }

    // Not present for non-repeated fields.
    if (descr_->maxRepetitionLevel() > 0) {
      // A row could include more than one value.
      // Count the occasions where we start a new row.
      for (int64_t i = 0; i < numValues; ++i) {
        if (repLevels[i] == 0) {
          rowsWritten_++;
          numBufferedRows_++;
        }
      }

      writeRepetitionLevels(numValues, repLevels);
    } else {
      // Each value is exactly one row.
      rowsWritten_ += numValues;
      numBufferedRows_ += numValues;
    }
    return valuesToWrite;
  }

  // This method will always update the three output parameters,
  // outValuesToWrite, outSpacedValuesToWrite and nullCount.
  // Additionally it will update the validity bitmap if required (i.e. if at
  // least one level of nullable structs directly precede the leaf node).
  void maybeCalculateValidityBits(
      const int16_t* defLevels,
      int64_t batchSize,
      int64_t* outValuesToWrite,
      int64_t* outSpacedValuesToWrite,
      int64_t* nullCount) {
    if (bitsBuffer_ == nullptr) {
      if (levelInfo_.defLevel == 0) {
        // In this case def levels should be null and we only
        // need to output counts which will always be equal to
        // the batch size passed in (max defLevel == 0 indicates
        // there cannot be repeated or null fields).
        VELOX_DCHECK_NULL(defLevels);
        *outValuesToWrite = batchSize;
        *outSpacedValuesToWrite = batchSize;
        *nullCount = 0;
      } else {
        for (int x = 0; x < batchSize; x++) {
          *outValuesToWrite += defLevels[x] == levelInfo_.defLevel ? 1 : 0;
          *outSpacedValuesToWrite +=
              defLevels[x] >= levelInfo_.repeatedAncestorDefLevel ? 1 : 0;
        }
        *nullCount = batchSize - *outValuesToWrite;
      }
      return;
    }
    // Shrink to fit possible causes another allocation, and would only be
    // necessary on the last batch.
    int64_t newBitmapSize = ::arrow::bit_util::BytesForBits(batchSize);
    if (newBitmapSize != bitsBuffer_->size()) {
      PARQUET_THROW_NOT_OK(bitsBuffer_->Resize(newBitmapSize, false));
      bitsBuffer_->ZeroPadding();
    }
    ValidityBitmapInputOutput io;
    io.validBits = bitsBuffer_->mutable_data();
    io.valuesReadUpperBound = batchSize;
    DefLevelsToBitmap(defLevels, batchSize, levelInfo_, &io);
    *outValuesToWrite = io.valuesRead - io.nullCount;
    *outSpacedValuesToWrite = io.valuesRead;
    *nullCount = io.nullCount;
  }

  Result<std::shared_ptr<Array>> maybeReplaceValidity(
      std::shared_ptr<Array> array,
      int64_t newNullCount,
      ::arrow::MemoryPool* memoryPool) {
    if (bitsBuffer_ == nullptr) {
      return array;
    }
    std::vector<std::shared_ptr<Buffer>> buffers = array->data()->buffers;
    if (buffers.empty()) {
      return array;
    }
    buffers[0] = bitsBuffer_;
    // Should be a leaf array.
    VELOX_DCHECK_GT(buffers.size(), 1);
    ValueBufferSlicer slicer{memoryPool};
    if (array->data()->offset > 0) {
      RETURN_NOT_OK(util::visitArrayInline(*array, &slicer, &buffers[1]));
    }
    return ::arrow::MakeArray(
        std::make_shared<ArrayData>(
            array->type(), array->length(), std::move(buffers), newNullCount));
  }

  void writeLevelsSpaced(
      int64_t numLevels,
      const int16_t* defLevels,
      const int16_t* repLevels) {
    // If the field is required and non-repeated, there are no definition
    // levels.
    if (descr_->maxDefinitionLevel() > 0) {
      writeDefinitionLevels(numLevels, defLevels);
    }
    // Not present for non-repeated fields.
    if (descr_->maxRepetitionLevel() > 0) {
      // A row could include more than one value.
      // Count the occasions where we start a new row.
      for (int64_t i = 0; i < numLevels; ++i) {
        if (repLevels[i] == 0) {
          rowsWritten_++;
          numBufferedRows_++;
        }
      }
      writeRepetitionLevels(numLevels, repLevels);
    } else {
      // Each value is exactly one row.
      rowsWritten_ += numLevels;
      numBufferedRows_ += numLevels;
    }
  }

  void commitWriteAndCheckPageLimit(
      int64_t numLevels,
      int64_t numValues,
      int64_t numNulls,
      bool checkPageSize) {
    numBufferedValues_ += numLevels;
    numBufferedEncodedValues_ += numValues;
    numBufferedNulls_ += numNulls;

    if (checkPageSize) {
      const bool sizeLimitExceeded =
          currentEncoder_->estimatedDataEncodedSize() >=
          properties_->dataPagesize();
      const bool rowLimitExceeded = properties_->dataPageRowNumberLimit() > 0 &&
          numBufferedRows_ >= properties_->dataPageRowNumberLimit();
      if (sizeLimitExceeded || rowLimitExceeded) {
        addDataPage();
      }
    }
  }

  void fallbackToPlainEncoding() {
    if (isDictionaryEncoding(currentEncoder_->encoding())) {
      writeDictionaryPage();
      // Serialize the buffered dictionary indices.
      flushBufferedDataPages();
      fallback_ = true;
      // Only PLAIN encoding is supported for fallback in V1.
      currentEncoder_ = makeEncoder(
          DType::typeNum,
          Encoding::kPlain,
          false,
          descr_,
          properties_->memoryPool());
      currentValueEncoder_ =
          dynamic_cast<ValueEncoderType*>(currentEncoder_.get());
      currentDictEncoder_ = nullptr; // not using dict
      encoding_ = Encoding::kPlain;
    }
  }

  // Checks if the Dictionary Page size limit is reached.
  // If the limit is reached, the Dictionary and Data Pages are serialized.
  // The encoding is switched to PLAIN.
  //
  // Only one Dictionary Page is written.
  // Fallback to PLAIN if dictionary page limit is reached.
  void checkDictionarySizeLimit() {
    if (!hasDictionary_ || fallback_) {
      // Either not using dictionary encoding, or we have already fallen back
      // to PLAIN encoding because the size threshold was reached.
      return;
    }

    if (currentDictEncoder_->dictEncodedSize() >=
        properties_->dictionaryPagesizeLimit()) {
      fallbackToPlainEncoding();
    }
  }

  void writeValues(const T* values, int64_t numValues, int64_t numNulls) {
    currentValueEncoder_->put(values, static_cast<int>(numValues));
    if (pageStatistics_ != nullptr) {
      pageStatistics_->update(values, numValues, numNulls);
    }
  }

  /// \brief Write values with spaces and update page statistics accordingly.
  ///
  /// @param values input buffer of values to write, including spaces.
  /// @param numValues number of non-null values in the values buffer.
  /// @param numSpacedValues length of values buffer, including spaces and
  /// does not count some nulls from ancestor (e.g. empty lists).
  /// @param validBits validity bitmap of values buffer, which does not
  /// include some nulls from ancestor (e.g. empty lists).
  /// @param validBitsOffset offset to validBits bitmap.
  /// @param numLevels number of levels to write, including nulls from values
  /// buffer and nulls from ancestor (e.g. empty lists).
  /// @param numNulls number of nulls in the values buffer as well as nulls
  /// from the ancestor (e.g. empty lists).
  void writeValuesSpaced(
      const T* values,
      int64_t numValues,
      int64_t numSpacedValues,
      const uint8_t* validBits,
      int64_t validBitsOffset,
      int64_t numLevels,
      int64_t numNulls) {
    if (numValues != numSpacedValues) {
      currentValueEncoder_->putSpaced(
          values,
          static_cast<int>(numSpacedValues),
          validBits,
          validBitsOffset);
    } else {
      currentValueEncoder_->put(values, static_cast<int>(numValues));
    }
    if (pageStatistics_ != nullptr) {
      pageStatistics_->updateSpaced(
          values,
          validBits,
          validBitsOffset,
          numSpacedValues,
          numValues,
          numNulls);
    }
  }
};

template <typename DType>
Status TypedColumnWriterImpl<DType>::writeArrowDictionary(
    const int16_t* defLevels,
    const int16_t* repLevels,
    int64_t numLevels,
    const ::arrow::Array& array,
    ArrowWriteContext* ctx,
    bool maybeParentNulls) {
  // If this is the first time writing a DictionaryArray, then there's
  // a few possible paths to take:
  //
  // - If dictionary encoding is not enabled, convert to densely.
  //   Encoded and call WriteArrow.
  // - Dictionary encoding enabled.
  //   - If this is the first time this is called, then we call
  //     putDictionary into the encoder and then putIndices on each
  //     chunk. We store the dictionary that was written in
  //     preservedDictionary_ so that subsequent calls to this method
  //     can make sure the dictionary has not changed.
  //   - On subsequent calls, we have to check whether the dictionary
  //     has changed. If it has, then we trigger the varying
  //     dictionary path and materialize each chunk and then call
  //     writeArrow with that.
  auto writeDense = [&] {
    std::shared_ptr<::arrow::Array> denseArray;
    RETURN_NOT_OK(convertDictionaryToDense(
        array, properties_->memoryPool(), &denseArray));
    return writeArrowDense(
        defLevels, repLevels, numLevels, *denseArray, ctx, maybeParentNulls);
  };

  if (!isDictionaryEncoding(currentEncoder_->encoding()) ||
      !dictionaryDirectWriteSupported(array)) {
    // No longer dictionary-encoding for whatever reason, maybe we never were
    // or we decided to stop. Note that writeArrow can be invoked multiple
    // times with both dense and dictionary-encoded versions of the same data
    // without a problem. Any dense data will be hashed to indices until the
    // dictionary page limit is reached, at which everything (dictionary and
    // dense) will fall back to plain encoding.
    return writeDense();
  }

  auto dictEncoder = dynamic_cast<DictEncoder<DType>*>(currentEncoder_.get());
  const auto& data = checked_cast<const ::arrow::DictionaryArray&>(array);
  std::shared_ptr<::arrow::Array> dictionary = data.dictionary();
  std::shared_ptr<::arrow::Array> indices = data.indices();

  auto updateStats = [&](int64_t numChunkLevels,
                         const std::shared_ptr<Array>& chunkIndices) {
    // TODO(PARQUET-2068) This approach may make two copies. First, a copy of
    // the indices array to a (hopefully smaller) referenced indices array.
    // Second, a copy of the values array to a (probably not smaller)
    // referenced values array.
    //
    // Once the MinMax kernel supports all data types we should use that kernel.
    // Instead as it does not make any copies.
    ::arrow::compute::ExecContext execCtx(ctx->memoryPool);
    execCtx.set_use_threads(false);

    std::shared_ptr<::arrow::Array> referencedDictionary;
    PARQUET_ASSIGN_OR_THROW(
        ::arrow::Datum referencedIndices,
        ::arrow::compute::Unique(*chunkIndices, &execCtx));

    // On first run, we might be able to re-use the existing dictionary.
    if (referencedIndices.length() == dictionary->length()) {
      referencedDictionary = dictionary;
    } else {
      PARQUET_ASSIGN_OR_THROW(
          ::arrow::Datum referencedDictionaryDatum,
          ::arrow::compute::Take(
              dictionary,
              referencedIndices,
              ::arrow::compute::TakeOptions::NoBoundsCheck(),
              &execCtx));
      referencedDictionary = referencedDictionaryDatum.make_array();
    }

    int64_t nonNullCount = chunkIndices->length() - chunkIndices->null_count();
    pageStatistics_->incrementNullCount(numChunkLevels - nonNullCount);
    pageStatistics_->incrementNumValues(nonNullCount);
    pageStatistics_->update(*referencedDictionary, false);
  };

  int64_t valueOffset = 0;
  auto writeIndicesChunk =
      [&](int64_t offset, int64_t batchSize, bool checkPage) {
        int64_t batchNumValues = 0;
        int64_t batchNumSpacedValues = 0;
        int64_t nullCount = ::arrow::kUnknownNullCount;
        // Bits is not null for nullable values.  At this point in the code we
        // can't. Determine if the leaf array has the same null values as any
        // parents it. Might have had so we need to recompute it from def
        // levels.
        maybeCalculateValidityBits(
            addIfNotNull(defLevels, offset),
            batchSize,
            &batchNumValues,
            &batchNumSpacedValues,
            &nullCount);
        writeLevelsSpaced(
            batchSize,
            addIfNotNull(defLevels, offset),
            addIfNotNull(repLevels, offset));
        std::shared_ptr<Array> writeableIndices =
            indices->Slice(valueOffset, batchNumSpacedValues);
        if (pageStatistics_) {
          updateStats(batchSize, writeableIndices);
        }
        PARQUET_ASSIGN_OR_THROW(
            writeableIndices,
            maybeReplaceValidity(writeableIndices, nullCount, ctx->memoryPool));
        dictEncoder->putIndices(*writeableIndices);
        commitWriteAndCheckPageLimit(
            batchSize, batchNumValues, nullCount, checkPage);
        valueOffset += batchNumSpacedValues;
      };

  // Handle seeing dictionary for the first time.
  if (!preservedDictionary_) {
    // It's a new dictionary. Call PutDictionary and keep track of it.
    PARQUET_CATCH_NOT_OK(dictEncoder->putDictionary(*dictionary));

    // If there were duplicate value in the dictionary, the encoder's memo
    // table will be out of sync with the indices in the Arrow array. The
    // easiest solution for this uncommon case is to fallback to plain
    // encoding.
    if (dictEncoder->numEntries() != dictionary->length()) {
      PARQUET_CATCH_NOT_OK(fallbackToPlainEncoding());
      return writeDense();
    }

    preservedDictionary_ = dictionary;
  } else if (!dictionary->Equals(*preservedDictionary_)) {
    // Dictionary has changed.
    PARQUET_CATCH_NOT_OK(fallbackToPlainEncoding());
    return writeDense();
  }

  PARQUET_CATCH_NOT_OK(doInBatches(
      defLevels,
      repLevels,
      numLevels,
      properties_->writeBatchSize(),
      writeIndicesChunk,
      pagesChangeOnRecordBoundaries()));
  return Status::OK();
}

// ----------------------------------------------------------------------.
// Direct Arrow write path.

template <typename ParquetType, typename ArrowType, typename Enable = void>
struct SerializeFunctor {
  using ArrowCType = typename ArrowType::c_type;
  using ArrayType = typename ::arrow::TypeTraits<ArrowType>::ArrayType;
  using ParquetCType = typename ParquetType::CType;
  Status
  serialize(const ArrayType& array, ArrowWriteContext*, ParquetCType* out) {
    const ArrowCType* input = array.raw_values();
    if (array.null_count() > 0) {
      for (int i = 0; i < array.length(); i++) {
        out[i] = static_cast<ParquetCType>(input[i]);
      }
    } else {
      std::copy(input, input + array.length(), out);
    }
    return Status::OK();
  }
};

template <typename ParquetType, typename ArrowType>
Status writeArrowSerialize(
    const ::arrow::Array& array,
    int64_t numLevels,
    const int16_t* defLevels,
    const int16_t* repLevels,
    ArrowWriteContext* ctx,
    TypedColumnWriter<ParquetType>* writer,
    bool maybeParentNulls) {
  using ParquetCType = typename ParquetType::CType;
  using ArrayType = typename ::arrow::TypeTraits<ArrowType>::ArrayType;

  ParquetCType* buffer = nullptr;
  PARQUET_THROW_NOT_OK(
      ctx->getScratchData<ParquetCType>(array.length(), &buffer));

  SerializeFunctor<ParquetType, ArrowType> functor;
  RETURN_NOT_OK(
      functor.serialize(checked_cast<const ArrayType&>(array), ctx, buffer));
  bool noNulls =
      writer->descr()->schemaNode()->isRequired() || (array.null_count() == 0);
  if (!maybeParentNulls && noNulls) {
    PARQUET_CATCH_NOT_OK(
        writer->writeBatch(numLevels, defLevels, repLevels, buffer));
  } else {
    PARQUET_CATCH_NOT_OK(writer->writeBatchSpaced(
        numLevels,
        defLevels,
        repLevels,
        array.null_bitmap_data(),
        array.offset(),
        buffer));
  }
  return Status::OK();
}

template <typename ParquetType>
Status writeArrowZeroCopy(
    const ::arrow::Array& array,
    int64_t numLevels,
    const int16_t* defLevels,
    const int16_t* repLevels,
    ArrowWriteContext* ctx,
    TypedColumnWriter<ParquetType>* writer,
    bool maybeParentNulls) {
  using T = typename ParquetType::CType;
  const auto& data = static_cast<const ::arrow::PrimitiveArray&>(array);
  const T* values = nullptr;
  // The values buffer may be null if the array is empty (ARROW-2744).
  if (data.values() != nullptr) {
    values = reinterpret_cast<const T*>(data.values()->data()) + data.offset();
  } else {
    VELOX_DCHECK_EQ(data.length(), 0);
  }
  bool noNulls =
      writer->descr()->schemaNode()->isRequired() || (array.null_count() == 0);

  if (!maybeParentNulls && noNulls) {
    PARQUET_CATCH_NOT_OK(
        writer->writeBatch(numLevels, defLevels, repLevels, values));
  } else {
    PARQUET_CATCH_NOT_OK(writer->writeBatchSpaced(
        numLevels,
        defLevels,
        repLevels,
        data.null_bitmap_data(),
        data.offset(),
        values));
  }
  return Status::OK();
}

#define WRITE_SERIALIZE_CASE(Arrowenum, ArrowType, ParquetType)  \
  case ::arrow::Type::Arrowenum:                                 \
    return writeArrowSerialize<ParquetType, ::arrow::ArrowType>( \
        array, numLevels, defLevels, repLevels, ctx, this, maybeParentNulls);

#define WRITE_ZERO_COPY_CASE(Arrowenum, ArrowType, ParquetType) \
  case ::arrow::Type::Arrowenum:                                \
    return writeArrowZeroCopy<ParquetType>(                     \
        array, numLevels, defLevels, repLevels, ctx, this, maybeParentNulls);

#define ARROW_UNSUPPORTED()                                          \
  std::stringstream ss;                                              \
  ss << "Arrow type " << array.type()->ToString()                    \
     << " cannot be written to Parquet type " << descr_->toString(); \
  return Status::Invalid(ss.str());

// ----------------------------------------------------------------------.
// Write Arrow to BooleanType.

template <>
struct SerializeFunctor<BooleanType, ::arrow::BooleanType> {
  Status
  serialize(const ::arrow::BooleanArray& data, ArrowWriteContext*, bool* out) {
    for (int i = 0; i < data.length(); i++) {
      *out++ = data.Value(i);
    }
    return Status::OK();
  }
};

template <>
Status TypedColumnWriterImpl<BooleanType>::writeArrowDense(
    const int16_t* defLevels,
    const int16_t* repLevels,
    int64_t numLevels,
    const ::arrow::Array& array,
    ArrowWriteContext* ctx,
    bool maybeParentNulls) {
  if (array.type_id() != ::arrow::Type::BOOL) {
    ARROW_UNSUPPORTED();
  }
  return writeArrowSerialize<BooleanType, ::arrow::BooleanType>(
      array, numLevels, defLevels, repLevels, ctx, this, maybeParentNulls);
}

// ----------------------------------------------------------------------.
// Write Arrow types to INT32.

template <>
struct SerializeFunctor<Int32Type, ::arrow::Date64Type> {
  Status serialize(
      const ::arrow::Date64Array& array,
      ArrowWriteContext*,
      int32_t* out) {
    const int64_t* input = array.raw_values();
    for (int i = 0; i < array.length(); i++) {
      *out++ = static_cast<int32_t>(*input++ / 86400000);
    }
    return Status::OK();
  }
};

template <typename ParquetType, typename ArrowType>
struct SerializeFunctor<
    ParquetType,
    ArrowType,
    ::arrow::enable_if_t<
        ::arrow::is_decimal_type<ArrowType>::value&& ::arrow::internal::
            IsOneOf<ParquetType, Int32Type, Int64Type>::value>> {
  using ValueType = typename ParquetType::CType;

  Status serialize(
      const typename ::arrow::TypeTraits<ArrowType>::ArrayType& array,
      ArrowWriteContext* ctx,
      ValueType* out) {
    if (array.null_count() == 0) {
      for (int64_t i = 0; i < array.length(); i++) {
        out[i] = transferValue<ArrowType::kByteWidth>(array.Value(i));
      }
    } else {
      for (int64_t i = 0; i < array.length(); i++) {
        out[i] = array.IsValid(i)
            ? transferValue<ArrowType::kByteWidth>(array.Value(i))
            : 0;
      }
    }

    return Status::OK();
  }

  template <int byteWidth>
  ValueType transferValue(const uint8_t* in) const {
    static_assert(
        byteWidth == 16 || byteWidth == 32,
        "only 16 and 32 byte Decimals supported");
    ValueType value = 0;
    if constexpr (byteWidth == 16) {
      ::arrow::Decimal128 decimalValue(in);
      PARQUET_ASSIGN_OR_THROW(value, decimalValue.ToInteger<ValueType>());
    } else {
      ::arrow::Decimal256 decimalValue(in);
      // Decimal256 does not provide ToInteger, but we are sure it fits in the
      // target integer type.
      value = static_cast<ValueType>(decimalValue.low_bits());
    }
    return value;
  }
};

template <>
struct SerializeFunctor<Int32Type, ::arrow::Time32Type> {
  Status serialize(
      const ::arrow::Time32Array& array,
      ArrowWriteContext*,
      int32_t* out) {
    const int32_t* input = array.raw_values();
    const auto& type = static_cast<const ::arrow::Time32Type&>(*array.type());
    if (type.unit() == ::arrow::TimeUnit::SECOND) {
      for (int i = 0; i < array.length(); i++) {
        out[i] = input[i] * 1000;
      }
    } else {
      std::copy(input, input + array.length(), out);
    }
    return Status::OK();
  }
};

template <>
Status TypedColumnWriterImpl<Int32Type>::writeArrowDense(
    const int16_t* defLevels,
    const int16_t* repLevels,
    int64_t numLevels,
    const ::arrow::Array& array,
    ArrowWriteContext* ctx,
    bool maybeParentNulls) {
  switch (array.type()->id()) {
    case ::arrow::Type::NA: {
      PARQUET_CATCH_NOT_OK(
          writeBatch(numLevels, defLevels, repLevels, nullptr));
    } break;
      WRITE_SERIALIZE_CASE(INT8, Int8Type, Int32Type)
      WRITE_SERIALIZE_CASE(UINT8, UInt8Type, Int32Type)
      WRITE_SERIALIZE_CASE(INT16, Int16Type, Int32Type)
      WRITE_SERIALIZE_CASE(UINT16, UInt16Type, Int32Type)
      WRITE_SERIALIZE_CASE(UINT32, UInt32Type, Int32Type)
      WRITE_ZERO_COPY_CASE(INT32, Int32Type, Int32Type)
      WRITE_ZERO_COPY_CASE(DATE32, Date32Type, Int32Type)
      WRITE_SERIALIZE_CASE(DATE64, Date64Type, Int32Type)
      WRITE_SERIALIZE_CASE(TIME32, Time32Type, Int32Type)
      WRITE_SERIALIZE_CASE(DECIMAL128, Decimal128Type, Int32Type)
      WRITE_SERIALIZE_CASE(DECIMAL256, Decimal256Type, Int32Type)
    default:
      ARROW_UNSUPPORTED()
  }
  return Status::OK();
}

// ----------------------------------------------------------------------.
// Write Arrow to Int64 and Int96.

#define INT96_CONVERT_LOOP(conversionFunction) \
  for (int64_t i = 0; i < array.length(); i++) \
    conversionFunction(input[i], &out[i]);

template <>
struct SerializeFunctor<Int96Type, ::arrow::TimestampType> {
  Status serialize(
      const ::arrow::TimestampArray& array,
      ArrowWriteContext*,
      Int96* out) {
    const int64_t* input = array.raw_values();
    const auto& type =
        static_cast<const ::arrow::TimestampType&>(*array.type());
    switch (type.unit()) {
      case ::arrow::TimeUnit::NANO:
        INT96_CONVERT_LOOP(internal::nanosecondsToImpalaTimestamp);
        break;
      case ::arrow::TimeUnit::MICRO:
        INT96_CONVERT_LOOP(internal::microsecondsToImpalaTimestamp);
        break;
      case ::arrow::TimeUnit::MILLI:
        INT96_CONVERT_LOOP(internal::millisecondsToImpalaTimestamp);
        break;
      case ::arrow::TimeUnit::SECOND:
        INT96_CONVERT_LOOP(internal::secondsToImpalaTimestamp);
        break;
    }
    return Status::OK();
  }
};

#define COERCE_DIVIDE -1
#define COERCE_INVALID 0
#define COERCE_MULTIPLY +1

static std::pair<int, int64_t> kTimestampCoercionFactors[4][4] = {
    // From seconds ...
    {{COERCE_INVALID, 0}, // ... to seconds
     {COERCE_MULTIPLY, 1000}, // ... to millis
     {COERCE_MULTIPLY, 1000000}, // ... to micros
     {COERCE_MULTIPLY, INT64_C(1000000000)}}, // ... to nanos
    // From millis ...
    {{COERCE_INVALID, 0},
     {COERCE_MULTIPLY, 1},
     {COERCE_MULTIPLY, 1000},
     {COERCE_MULTIPLY, 1000000}},
    // From micros ...
    {{COERCE_INVALID, 0},
     {COERCE_DIVIDE, 1000},
     {COERCE_MULTIPLY, 1},
     {COERCE_MULTIPLY, 1000}},
    // From nanos ...
    {{COERCE_INVALID, 0},
     {COERCE_DIVIDE, 1000000},
     {COERCE_DIVIDE, 1000},
     {COERCE_MULTIPLY, 1}}};

template <>
struct SerializeFunctor<Int64Type, ::arrow::TimestampType> {
  Status serialize(
      const ::arrow::TimestampArray& array,
      ArrowWriteContext* ctx,
      int64_t* out) {
    const auto& sourceType =
        static_cast<const ::arrow::TimestampType&>(*array.type());
    auto sourceUnit = sourceType.unit();
    const int64_t* values = array.raw_values();

    ::arrow::TimeUnit::type targetUnit =
        ctx->properties->coerceTimestampsUnit();
    auto targetType = ::arrow::timestamp(targetUnit);
    bool truncationAllowed = ctx->properties->truncatedTimestampsAllowed();

    auto divideBy = [&](const int64_t factor) {
      for (int64_t i = 0; i < array.length(); i++) {
        if (!truncationAllowed && array.IsValid(i) &&
            (values[i] % factor != 0)) {
          return Status::Invalid(
              "Casting from ",
              sourceType.ToString(),
              " to ",
              targetType->ToString(),
              " would lose data: ",
              values[i]);
        }
        out[i] = values[i] / factor;
      }
      return Status::OK();
    };

    auto multiplyBy = [&](const int64_t factor) {
      for (int64_t i = 0; i < array.length(); i++) {
        out[i] = values[i] * factor;
      }
      return Status::OK();
    };

    const auto& coercion =
        kTimestampCoercionFactors[static_cast<int>(sourceUnit)]
                                 [static_cast<int>(targetUnit)];

    // first -> coercion operation; second -> scale factor.
    VELOX_DCHECK_NE(coercion.first, COERCE_INVALID);
    return coercion.first == COERCE_DIVIDE ? divideBy(coercion.second)
                                           : multiplyBy(coercion.second);
  }
};

#undef COERCE_DIVIDE
#undef COERCE_INVALID
#undef COERCE_MULTIPLY

Status writeTimestamps(
    const ::arrow::Array& values,
    int64_t numLevels,
    const int16_t* defLevels,
    const int16_t* repLevels,
    ArrowWriteContext* ctx,
    TypedColumnWriter<Int64Type>* writer,
    bool maybeParentNulls) {
  const auto& sourceType =
      static_cast<const ::arrow::TimestampType&>(*values.type());

  auto writeCoerce = [&](const ArrowWriterProperties* properties) {
    ArrowWriteContext tempCtx = *ctx;
    tempCtx.properties = properties;
    return writeArrowSerialize<Int64Type, ::arrow::TimestampType>(
        values,
        numLevels,
        defLevels,
        repLevels,
        &tempCtx,
        writer,
        maybeParentNulls);
  };

  const ParquetVersion::type version = writer->properties()->version();

  if (ctx->properties->coerceTimestampsEnabled()) {
    // User explicitly requested coercion to specific unit.
    if (sourceType.unit() == ctx->properties->coerceTimestampsUnit()) {
      // No data conversion necessary.
      return writeArrowZeroCopy<Int64Type>(
          values,
          numLevels,
          defLevels,
          repLevels,
          ctx,
          writer,
          maybeParentNulls);
    } else {
      return writeCoerce(ctx->properties);
    }
  } else if (
      (version == ParquetVersion::PARQUET_1_0 ||
       version == ParquetVersion::PARQUET_2_4) &&
      sourceType.unit() == ::arrow::TimeUnit::NANO) {
    // Absent superseding user instructions, when writing Parquet version
    // Files, timestamps in nanoseconds are coerced to microseconds.
    std::shared_ptr<ArrowWriterProperties> properties =
        (ArrowWriterProperties::Builder())
            .coerceTimestamps(::arrow::TimeUnit::MICRO)
            ->disallowTruncatedTimestamps()
            ->build();
    return writeCoerce(properties.get());
  } else if (sourceType.unit() == ::arrow::TimeUnit::SECOND) {
    // To milliseconds.
    std::shared_ptr<ArrowWriterProperties> properties =
        (ArrowWriterProperties::Builder())
            .coerceTimestamps(::arrow::TimeUnit::MILLI)
            ->build();
    return writeCoerce(properties.get());
  } else {
    // No data conversion necessary.
    return writeArrowZeroCopy<Int64Type>(
        values, numLevels, defLevels, repLevels, ctx, writer, maybeParentNulls);
  }
}

template <>
Status TypedColumnWriterImpl<Int64Type>::writeArrowDense(
    const int16_t* defLevels,
    const int16_t* repLevels,
    int64_t numLevels,
    const ::arrow::Array& array,
    ArrowWriteContext* ctx,
    bool maybeParentNulls) {
  switch (array.type()->id()) {
    case ::arrow::Type::TIMESTAMP:
      return writeTimestamps(
          array, numLevels, defLevels, repLevels, ctx, this, maybeParentNulls);
      WRITE_ZERO_COPY_CASE(INT64, Int64Type, Int64Type)
      WRITE_SERIALIZE_CASE(UINT32, UInt32Type, Int64Type)
      WRITE_SERIALIZE_CASE(UINT64, UInt64Type, Int64Type)
      WRITE_ZERO_COPY_CASE(TIME64, Time64Type, Int64Type)
      WRITE_ZERO_COPY_CASE(DURATION, DurationType, Int64Type)
      WRITE_SERIALIZE_CASE(DECIMAL128, Decimal128Type, Int64Type)
      WRITE_SERIALIZE_CASE(DECIMAL256, Decimal256Type, Int64Type)
    default:
      ARROW_UNSUPPORTED();
  }
}

template <>
Status TypedColumnWriterImpl<Int96Type>::writeArrowDense(
    const int16_t* defLevels,
    const int16_t* repLevels,
    int64_t numLevels,
    const ::arrow::Array& array,
    ArrowWriteContext* ctx,
    bool maybeParentNulls) {
  if (array.type_id() != ::arrow::Type::TIMESTAMP) {
    ARROW_UNSUPPORTED();
  }
  return writeArrowSerialize<Int96Type, ::arrow::TimestampType>(
      array, numLevels, defLevels, repLevels, ctx, this, maybeParentNulls);
}

// ----------------------------------------------------------------------.
// Floating point types.

template <>
Status TypedColumnWriterImpl<FloatType>::writeArrowDense(
    const int16_t* defLevels,
    const int16_t* repLevels,
    int64_t numLevels,
    const ::arrow::Array& array,
    ArrowWriteContext* ctx,
    bool maybeParentNulls) {
  if (array.type_id() != ::arrow::Type::FLOAT) {
    ARROW_UNSUPPORTED();
  }
  return writeArrowZeroCopy<FloatType>(
      array, numLevels, defLevels, repLevels, ctx, this, maybeParentNulls);
}

template <>
Status TypedColumnWriterImpl<DoubleType>::writeArrowDense(
    const int16_t* defLevels,
    const int16_t* repLevels,
    int64_t numLevels,
    const ::arrow::Array& array,
    ArrowWriteContext* ctx,
    bool maybeParentNulls) {
  if (array.type_id() != ::arrow::Type::DOUBLE) {
    ARROW_UNSUPPORTED();
  }
  return writeArrowZeroCopy<DoubleType>(
      array, numLevels, defLevels, repLevels, ctx, this, maybeParentNulls);
}

// ----------------------------------------------------------------------.
// Write Arrow to BYTE_ARRAY.

template <>
Status TypedColumnWriterImpl<ByteArrayType>::writeArrowDense(
    const int16_t* defLevels,
    const int16_t* repLevels,
    int64_t numLevels,
    const ::arrow::Array& array,
    ArrowWriteContext* ctx,
    bool maybeParentNulls) {
  if (!::arrow::is_base_binary_like(array.type()->id())) {
    ARROW_UNSUPPORTED();
  }

  int64_t valueOffset = 0;
  auto writeChunk = [&](int64_t offset, int64_t batchSize, bool checkPage) {
    int64_t batchNumValues = 0;
    int64_t batchNumSpacedValues = 0;
    int64_t nullCount = 0;

    maybeCalculateValidityBits(
        addIfNotNull(defLevels, offset),
        batchSize,
        &batchNumValues,
        &batchNumSpacedValues,
        &nullCount);
    writeLevelsSpaced(
        batchSize,
        addIfNotNull(defLevels, offset),
        addIfNotNull(repLevels, offset));
    std::shared_ptr<Array> dataSlice =
        array.Slice(valueOffset, batchNumSpacedValues);
    PARQUET_ASSIGN_OR_THROW(
        dataSlice, maybeReplaceValidity(dataSlice, nullCount, ctx->memoryPool));

    currentEncoder_->put(*dataSlice);
    // Null values in ancestors count as nulls.
    const int64_t nonNull = dataSlice->length() - dataSlice->null_count();
    if (pageStatistics_ != nullptr) {
      pageStatistics_->update(*dataSlice, false);
      pageStatistics_->incrementNullCount(batchSize - nonNull);
      pageStatistics_->incrementNumValues(nonNull);
    }
    commitWriteAndCheckPageLimit(
        batchSize, batchNumValues, batchSize - nonNull, checkPage);
    checkDictionarySizeLimit();
    valueOffset += batchNumSpacedValues;
  };

  PARQUET_CATCH_NOT_OK(doInBatches(
      defLevels,
      repLevels,
      numLevels,
      properties_->writeBatchSize(),
      writeChunk,
      pagesChangeOnRecordBoundaries()));
  return Status::OK();
}

// ----------------------------------------------------------------------.
// Write Arrow to FIXED_LEN_BYTE_ARRAY.

template <typename ParquetType, typename ArrowType>
struct SerializeFunctor<
    ParquetType,
    ArrowType,
    ::arrow::enable_if_t<
        ::arrow::is_fixed_size_binary_type<ArrowType>::value &&
        !::arrow::is_decimal_type<ArrowType>::value>> {
  Status serialize(
      const ::arrow::FixedSizeBinaryArray& array,
      ArrowWriteContext*,
      FLBA* out) {
    if (array.null_count() == 0) {
      // No nulls, just dump the data.
      // Todo(advancedxy): use a writeBatch to avoid this step.
      for (int64_t i = 0; i < array.length(); i++) {
        out[i] = FixedLenByteArray(array.GetValue(i));
      }
    } else {
      for (int64_t i = 0; i < array.length(); i++) {
        if (array.IsValid(i)) {
          out[i] = FixedLenByteArray(array.GetValue(i));
        }
      }
    }
    return Status::OK();
  }
};

// ----------------------------------------------------------------------.
// Write Arrow to Decimal128.

// Requires a custom serializer because decimal in parquet are in big-endian
// format. Thus, a temporary local buffer is required.
template <typename ParquetType, typename ArrowType>
struct SerializeFunctor<
    ParquetType,
    ArrowType,
    ::arrow::enable_if_t<
        ::arrow::is_decimal_type<ArrowType>::value &&
        !::arrow::internal::IsOneOf<ParquetType, Int32Type, Int64Type>::
            value>> {
  Status serialize(
      const typename ::arrow::TypeTraits<ArrowType>::ArrayType& array,
      ArrowWriteContext* ctx,
      FLBA* out) {
    allocateScratch(array, ctx);
    auto decimalOffsetValue = decimalOffset(array);

    if (array.null_count() == 0) {
      for (int64_t i = 0; i < array.length(); i++) {
        out[i] = fixDecimalEndianess<ArrowType::kByteWidth>(
            array.GetValue(i), decimalOffsetValue);
      }
    } else {
      for (int64_t i = 0; i < array.length(); i++) {
        out[i] = array.IsValid(i) ? fixDecimalEndianess<ArrowType::kByteWidth>(
                                        array.GetValue(i), decimalOffsetValue)
                                  : FixedLenByteArray();
      }
    }

    return Status::OK();
  }

  // Parquet's decimals are stored with FixedLength values where the
  // length is proportional to the precision. Arrow's Decimal are always stored
  // with 16 or 32 bytes. Thus the internal FLBA pointer must be adjusted by the
  // offset calculated here.
  int32_t decimalOffset(const Array& array) {
    auto decimalType = checked_pointer_cast<::arrow::DecimalType>(array.type());
    return decimalType->byte_width() -
        ::arrow::DecimalType::DecimalSize(decimalType->precision());
  }

  void allocateScratch(
      const typename ::arrow::TypeTraits<ArrowType>::ArrayType& array,
      ArrowWriteContext* ctx) {
    int64_t nonNullCount = array.length() - array.null_count();
    int64_t size = nonNullCount * ArrowType::kByteWidth;
    scratchBuffer = allocateBuffer(ctx->memoryPool, size);
    scratch = reinterpret_cast<int64_t*>(scratchBuffer->mutable_data());
  }

  template <int byteWidth>
  FixedLenByteArray fixDecimalEndianess(const uint8_t* in, int64_t offset) {
    const auto* u64In = reinterpret_cast<const int64_t*>(in);
    auto out = reinterpret_cast<const uint8_t*>(scratch) + offset;
    static_assert(
        byteWidth == 16 || byteWidth == 32,
        "only 16 and 32 byte Decimals supported");
    if (byteWidth == 32) {
      *scratch++ = ::arrow::bit_util::ToBigEndian(u64In[3]);
      *scratch++ = ::arrow::bit_util::ToBigEndian(u64In[2]);
      *scratch++ = ::arrow::bit_util::ToBigEndian(u64In[1]);
      *scratch++ = ::arrow::bit_util::ToBigEndian(u64In[0]);
    } else {
      *scratch++ = ::arrow::bit_util::ToBigEndian(u64In[1]);
      *scratch++ = ::arrow::bit_util::ToBigEndian(u64In[0]);
    }
    return FixedLenByteArray(out);
  }

  std::shared_ptr<ResizableBuffer> scratchBuffer;
  int64_t* scratch;
};

template <>
Status TypedColumnWriterImpl<FLBAType>::writeArrowDense(
    const int16_t* defLevels,
    const int16_t* repLevels,
    int64_t numLevels,
    const ::arrow::Array& array,
    ArrowWriteContext* ctx,
    bool maybeParentNulls) {
  switch (array.type()->id()) {
    WRITE_SERIALIZE_CASE(FIXED_SIZE_BINARY, FixedSizeBinaryType, FLBAType)
    WRITE_SERIALIZE_CASE(DECIMAL128, Decimal128Type, FLBAType)
    WRITE_SERIALIZE_CASE(DECIMAL256, Decimal256Type, FLBAType)
    default:
      break;
  }
  return Status::OK();
}

// ----------------------------------------------------------------------.
// Dynamic column writer constructor.

std::shared_ptr<ColumnWriter> ColumnWriter::make(
    ColumnChunkMetaDataBuilder* metadata,
    std::unique_ptr<PageWriter> pager,
    const WriterProperties* properties) {
  const ColumnDescriptor* descr = metadata->descr();
  const bool useDictionary = properties->dictionaryEnabled(descr->path()) &&
      descr->physicalType() != Type::kBoolean;
  Encoding::type encoding = properties->encoding(descr->path());

  if (encoding == Encoding::kUnknown) {
    // TODO: Arrow uses RLE by default for boolean columns. Since Velox can't
    // read RLEs yet, we disable this check. Re-enable once Velox's native
    // reader supports RLE.
    // Encoding = (descr->physical_type() == Type::kBoolean &&.
    //            properties->version() != ParquetVersion::PARQUET_1_0)
    //               ? Encoding::RLE.
    //               : Encoding::PLAIN;
    encoding = Encoding::kPlain;
  }
  if (useDictionary) {
    encoding = properties->dictionaryIndexEncoding();
  }
  switch (descr->physicalType()) {
    case Type::kBoolean:
      return std::make_shared<TypedColumnWriterImpl<BooleanType>>(
          metadata, std::move(pager), useDictionary, encoding, properties);
    case Type::kInt32:
      return std::make_shared<TypedColumnWriterImpl<Int32Type>>(
          metadata, std::move(pager), useDictionary, encoding, properties);
    case Type::kInt64:
      return std::make_shared<TypedColumnWriterImpl<Int64Type>>(
          metadata, std::move(pager), useDictionary, encoding, properties);
    case Type::kInt96:
      return std::make_shared<TypedColumnWriterImpl<Int96Type>>(
          metadata, std::move(pager), useDictionary, encoding, properties);
    case Type::kFloat:
      return std::make_shared<TypedColumnWriterImpl<FloatType>>(
          metadata, std::move(pager), useDictionary, encoding, properties);
    case Type::kDouble:
      return std::make_shared<TypedColumnWriterImpl<DoubleType>>(
          metadata, std::move(pager), useDictionary, encoding, properties);
    case Type::kByteArray:
      return std::make_shared<TypedColumnWriterImpl<ByteArrayType>>(
          metadata, std::move(pager), useDictionary, encoding, properties);
    case Type::kFixedLenByteArray:
      return std::make_shared<TypedColumnWriterImpl<FLBAType>>(
          metadata, std::move(pager), useDictionary, encoding, properties);
    default:
      ParquetException::NYI("type reader not implemented");
  }
  // Unreachable code, but suppress compiler warning.
  return std::shared_ptr<ColumnWriter>(nullptr);
}

} // namespace facebook::velox::parquet::arrow
