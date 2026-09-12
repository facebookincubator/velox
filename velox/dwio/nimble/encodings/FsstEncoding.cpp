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
#include "velox/dwio/nimble/encodings/FsstEncoding.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>

#include "folly/ScopeGuard.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

namespace facebook::nimble {

namespace {

// Forwards compression decisions to the shared parent policy.
class DelegatingCompressionPolicy final : public CompressionPolicy {
 public:
  explicit DelegatingCompressionPolicy(
      std::shared_ptr<CompressionPolicy> policy)
      : policy_{std::move(policy)} {}

  CompressionConfig config() const override {
    return policy_->config();
  }

  bool shouldAccept(
      CompressionType compressionType,
      uint64_t uncompressedSize,
      uint64_t compressedSize) const override {
    return policy_->shouldAccept(
        compressionType, uncompressedSize, compressedSize);
  }

 private:
  std::shared_ptr<CompressionPolicy> policy_;
};

class TrivialFallbackSelectionPolicy final
    : public EncodingSelectionPolicy<std::string_view> {
 public:
  TrivialFallbackSelectionPolicy(
      EncodingSelection<std::string_view>& parentSelection,
      std::shared_ptr<CompressionPolicy> compressionPolicy)
      : parentSelection_{parentSelection},
        compressionPolicy_{std::move(compressionPolicy)} {}

  EncodingSelectionResult select(
      std::span<const std::string_view> /* values */,
      const Statistics<std::string_view>& /* statistics */,
      const Encoding::Options& /* options */) override {
    return trivialSelectionResult();
  }

  EncodingSelectionResult selectNullable(
      std::span<const std::string_view> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<std::string_view>& /* statistics */,
      const Encoding::Options& /* options */) override {
    NIMBLE_UNSUPPORTED("Nullable FSST fallback is not supported.");
  }

 private:
  EncodingSelectionResult trivialSelectionResult() const {
    return {
        .encodingType = EncodingType::Trivial,
        .compressionPolicyFactory = [compressionPolicy = compressionPolicy_]() {
          return std::make_unique<DelegatingCompressionPolicy>(
              compressionPolicy);
        }};
  }

  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier nestedEncodingIdentifier,
      DataType nestedDataType) override {
    NIMBLE_CHECK_EQ(
        parentEncodingType,
        EncodingType::Trivial,
        "FSST fallback only supports Trivial nested encodings.");
    NIMBLE_CHECK_EQ(
        nestedDataType,
        DataType::Uint32,
        "Trivial string fallback only supports uint32 length encodings.");
    return parentSelection_.template createNestedPolicy<uint32_t>(
        EncodingType::Trivial, nestedEncodingIdentifier);
  }

  EncodingSelection<std::string_view>& parentSelection_;
  std::shared_ptr<CompressionPolicy> compressionPolicy_;
};

size_t sumLengths(std::span<const size_t> lengths) {
  return std::accumulate(lengths.begin(), lengths.end(), size_t{0});
}

uint32_t readFsstHeaderVarint(std::string_view encoding, size_t& offset) {
  uint32_t value{0};
  for (uint32_t byteIndex = 0; byteIndex < 5; ++byteIndex) {
    NIMBLE_CHECK_LT(offset, encoding.size(), "Truncated FSST header varint.");
    const auto byte = static_cast<uint8_t>(encoding[offset++]);
    if (byteIndex == 4) {
      NIMBLE_CHECK_EQ(byte & 0xf0, 0, "Overlong FSST header varint.");
    }
    value |= static_cast<uint32_t>(byte & 0x7f) << (byteIndex * 7);
    if ((byte & 0x80) == 0) {
      NIMBLE_CHECK_EQ(
          byteIndex + 1,
          varint::varintSize(value),
          "Overlong FSST header varint.");
      return value;
    }
  }
  NIMBLE_FAIL("Overlong FSST header varint.");
}

} // namespace

FsstEncoding::CompressedValues::CompressedValues(
    velox::memory::MemoryPool* pool)
    : compressedBuffer{pool}, compressedLengths{pool}, compressedPtrs{pool} {}

FsstEncoding::Header FsstEncoding::parseHeader(
    std::string_view encoding,
    size_t offset) {
  NIMBLE_CHECK_LE(
      offset, encoding.size(), "FSST header offset exceeds encoding bounds.");

  Header header{};
  const auto symbolTableSize = readFsstHeaderVarint(encoding, offset);
  NIMBLE_CHECK_GT(
      symbolTableSize, 0, "FSST symbol table size must be positive.");
  NIMBLE_CHECK_LE(
      symbolTableSize,
      static_cast<uint32_t>(FSST_MAXHEADER),
      "FSST symbol table size exceeds FSST_MAXHEADER.");
  NIMBLE_CHECK_LE(
      static_cast<size_t>(symbolTableSize),
      encoding.size() - offset,
      "FSST symbol table exceeds encoding bounds.");
  header.symbolTable = encoding.substr(offset, symbolTableSize);
  offset += symbolTableSize;

  const auto lengthsSize = readFsstHeaderVarint(encoding, offset);
  NIMBLE_CHECK_GT(
      lengthsSize, 0, "FSST lengths encoding size must be positive.");
  NIMBLE_CHECK_LE(
      static_cast<size_t>(lengthsSize),
      encoding.size() - offset,
      "FSST lengths encoding exceeds encoding bounds.");
  header.lengths = encoding.substr(offset, lengthsSize);
  offset += lengthsSize;
  header.blob = encoding.substr(offset);
  return header;
}

std::string_view FsstEncoding::validateEncodedPrefix(
    std::string_view encoding,
    const Encoding::Options& options) {
  NIMBLE_CHECK_FILE_GE(
      encoding.size(),
      EncodingPrefix::kRowCountOffset,
      "Truncated FSST encoding prefix.");
  NIMBLE_CHECK_FILE_EQ(
      EncodingPrefix::encodingType(encoding),
      EncodingType::Fsst,
      "Expected FSST encoding.");
  NIMBLE_CHECK_FILE_EQ(
      EncodingPrefix::dataType(encoding),
      DataType::String,
      "FSST encoding must contain string data.");

  if (options.useVarintRowCount) {
    size_t offset = EncodingPrefix::kRowCountOffset;
    readFsstHeaderVarint(encoding, offset);
  } else {
    NIMBLE_CHECK_FILE_GE(
        encoding.size(),
        EncodingPrefix::kFixedPrefixSize,
        "Truncated FSST encoding prefix.");
  }
  return encoding;
}

void FsstEncoding::validateSymbolTable(std::string_view symbolTable) {
  constexpr size_t kSerializedHeaderSize = 17;
  NIMBLE_CHECK_FILE_GE(
      symbolTable.size(),
      kSerializedHeaderSize,
      "Truncated FSST symbol table.");

  const auto zeroTerminated = static_cast<uint8_t>(symbolTable[8]);
  NIMBLE_CHECK_FILE_LE(zeroTerminated, 1, "Invalid FSST zero-terminated flag.");

  uint32_t symbolCount{0};
  uint64_t serializedSize = kSerializedHeaderSize;
  for (uint32_t symbolLength = 1; symbolLength <= 8; ++symbolLength) {
    const auto count = static_cast<uint8_t>(symbolTable[9 + symbolLength - 1]);
    symbolCount += count;
    serializedSize += static_cast<uint64_t>(count) * symbolLength;
  }

  NIMBLE_CHECK_FILE_LE(
      symbolCount, 255, "FSST symbol table contains too many symbols.");
  if (zeroTerminated != 0) {
    const auto oneByteSymbolCount = static_cast<uint8_t>(symbolTable[9]);
    NIMBLE_CHECK_FILE_GT(
        oneByteSymbolCount,
        0,
        "FSST zero-terminated table has no terminator symbol.");
    --serializedSize;
  }
  NIMBLE_CHECK_FILE_EQ(
      serializedSize,
      symbolTable.size(),
      "FSST symbol table histogram does not match its serialized size.");
}

size_t FsstEncoding::validateCompressedLengths(
    std::span<const uint32_t> lengths,
    std::string_view blob,
    size_t blobOffset) {
  NIMBLE_CHECK_FILE_LE(
      blobOffset, blob.size(), "FSST blob position exceeds its bounds.");

  size_t compressedBytes{0};
  for (const auto compressedLength : lengths) {
    const auto currentOffset = blobOffset + compressedBytes;
    NIMBLE_CHECK_FILE_LE(
        static_cast<size_t>(compressedLength),
        blob.size() - currentOffset,
        "FSST compressed length exceeds the remaining blob.");
    if (compressedLength > 0) {
      NIMBLE_CHECK_FILE_NE(
          static_cast<uint8_t>(blob[currentOffset + compressedLength - 1]),
          FSST_ESC,
          "FSST compressed string ends with an incomplete escape code.");
    }
    compressedBytes += compressedLength;
  }
  return compressedBytes;
}

FsstEncoding::CompressedValues FsstEncoding::compressValues(
    std::span<const physicalType> values,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK_NOT_NULL(pool, "Memory pool cannot be null.");
  const auto valueCount = static_cast<uint32_t>(values.size());
  CompressedValues result{pool};

  Vector<size_t> inputLengths{pool, valueCount};
  Vector<const unsigned char*> inputPtrs{pool, valueCount};
  for (uint32_t i = 0; i < valueCount; ++i) {
    inputLengths[i] = values[i].size();
    inputPtrs[i] = reinterpret_cast<const unsigned char*>(values[i].data());
  }
  result.totalInputSize =
      sumLengths({inputLengths.data(), inputLengths.size()});

  auto* encoder = nimble_fsst_create(
      valueCount,
      inputLengths.data(),
      inputPtrs.data(),
      /*zeroTerminated=*/0);
  NIMBLE_CHECK_NOT_NULL(encoder, "FSST encoder creation failed.");
  SCOPE_EXIT {
    nimble_fsst_destroy(encoder);
  };

  // fsst_export only takes a raw pointer, so the caller must provide the
  // library-defined maximum header capacity.
  result.symbolTableBuffer =
      velox::AlignedBuffer::allocate<unsigned char>(FSST_MAXHEADER, pool);
  result.symbolTableData = result.symbolTableBuffer->asMutable<unsigned char>();
  result.symbolTableSize = nimble_fsst_export(encoder, result.symbolTableData);
  NIMBLE_CHECK_LE(
      result.symbolTableSize,
      static_cast<size_t>(FSST_MAXHEADER),
      "FSST exported symbol table exceeded FSST_MAXHEADER.");

  // FSST's documented conservative compression output bound.
  const size_t outputBufSize = 7 + 2 * result.totalInputSize;
  result.compressedBuffer.resize(outputBufSize);
  result.compressedLengths.resize(valueCount);
  result.compressedPtrs.resize(valueCount);

  const auto numCompressed = nimble_fsst_compress(
      encoder,
      valueCount,
      inputLengths.data(),
      inputPtrs.data(),
      outputBufSize,
      result.compressedBuffer.data(),
      result.compressedLengths.data(),
      result.compressedPtrs.data());
  NIMBLE_CHECK_EQ(
      static_cast<uint32_t>(numCompressed),
      valueCount,
      "FSST compression did not compress all strings.");

  result.totalCompressedSize = sumLengths(
      {result.compressedLengths.data(), result.compressedLengths.size()});
  return result;
}

bool FsstEncoding::meetsCompressionTarget(
    uint64_t uncompressedSize,
    uint64_t encodedSize,
    double compressionTargetRatio) {
  return encodedSize <= uncompressedSize * compressionTargetRatio;
}

std::string_view FsstEncoding::encodeCompressedLengths(
    EncodingSelection<physicalType>& selection,
    std::span<const size_t> compressedLengths,
    Buffer& buffer,
    const Encoding::Options& options) {
  Vector<uint32_t> lengths{&buffer.getMemoryPool(), compressedLengths.size()};
  for (size_t i = 0; i < compressedLengths.size(); ++i) {
    lengths[i] = static_cast<uint32_t>(compressedLengths[i]);
  }

  return selection.template encodeNested<uint32_t>(
      EncodingIdentifiers::Fsst::Lengths, {lengths}, buffer, options);
}

std::string_view FsstEncoding::encodeTrivialFallback(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  auto compressionPolicy = std::shared_ptr<CompressionPolicy>(
      selection.compressionPolicy().release());

  auto fallbackPolicy = std::make_unique<TrivialFallbackSelectionPolicy>(
      selection, compressionPolicy);
  EncodingSelection<std::string_view> fallbackSelection{
      fallbackPolicy->select(values, selection.statistics(), options),
      Statistics<std::string_view>{selection.statistics()},
      std::move(fallbackPolicy)};
  return TrivialEncoding<std::string_view>::encode(
      fallbackSelection, values, buffer, options);
}

FsstEncoding::FsstEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<
          std::string_view,
          std::
              string_view>{pool, validateEncodedPrefix(data, options), options},
      stringBufferFactory_{std::move(stringBufferFactory)},
      lengthBuffer_{&pool},
      decompressBuffer_{&pool} {
  const auto header = parseHeader(data, this->dataOffset());
  validateSymbolTable(header.symbolTable);
  const auto bytesConsumed = nimble_fsst_import(
      &decoder_,
      const_cast<unsigned char*>(
          reinterpret_cast<const unsigned char*>(header.symbolTable.data())));
  NIMBLE_CHECK_EQ(
      static_cast<size_t>(bytesConsumed),
      header.symbolTable.size(),
      "FSST symbol table import size mismatch.");

  lengths_ = EncodingFactory().create(
      pool, header.lengths, stringBufferFactory_, options);
  NIMBLE_CHECK_FILE_EQ(
      lengths_->dataType(),
      DataType::Uint32,
      "FSST lengths encoding must contain Uint32 values.");
  NIMBLE_CHECK_FILE(
      !lengths_->isNullable(), "FSST lengths encoding must not be nullable.");
  NIMBLE_CHECK_FILE_EQ(
      lengths_->rowCount(),
      this->rowCount(),
      "FSST lengths row count does not match the parent encoding.");
  blob_ = header.blob;
  validateLengthsAndBlob();
}

std::string_view FsstEncoding::lengthsEncoding(
    std::string_view encoding,
    const Encoding::Options& options) {
  validateEncodedPrefix(encoding, options);
  const auto prefixSize =
      EncodingPrefix::prefixSize(encoding, options.useVarintRowCount);
  const auto header = parseHeader(encoding, prefixSize);
  return header.lengths;
}

void FsstEncoding::captureNestedEncoding(
    std::string_view encoding,
    std::vector<std::optional<const EncodingLayout>>& children,
    const Encoding::Options& options) {
  children.reserve(1);
  children.emplace_back(
      EncodingLayoutCapture::capture(
          lengthsEncoding(encoding, options), options));
}

void FsstEncoding::reset() {
  row_ = 0;
  blobOffset_ = 0;
  lengths_->reset();
  pageUsedBytes_ = 0;
  currentPageIndex_ = 0;
  if (stringPages_.empty()) {
    currentPage_ = nullptr;
    pageCapacityBytes_ = 0;
  } else {
    currentPage_ = stringPages_.front().data;
    pageCapacityBytes_ = stringPages_.front().capacity;
  }
}

void FsstEncoding::validateLengthsAndBlob() {
  constexpr uint32_t kValidationBatchSize = 1024;
  size_t validatedBlobBytes{0};
  uint32_t validatedRows{0};
  while (validatedRows < this->rowCount()) {
    const auto batchSize =
        std::min(kValidationBatchSize, this->rowCount() - validatedRows);
    lengthBuffer_.resize(batchSize);
    lengths_->materialize(batchSize, lengthBuffer_.data());
    validatedBlobBytes += validateCompressedLengths(
        {lengthBuffer_.data(), lengthBuffer_.size()},
        blob_,
        validatedBlobBytes);
    validatedRows += batchSize;
  }
  NIMBLE_CHECK_FILE_EQ(
      validatedBlobBytes,
      blob_.size(),
      "FSST compressed lengths do not match the blob size.");
  lengths_->reset();
  lengthBuffer_.resize(0);
}

void FsstEncoding::checkReadRange(uint32_t rowCount, const char* operation)
    const {
  NIMBLE_CHECK_LE(row_, this->rowCount(), "Invalid FSST encoding position.");
  NIMBLE_CHECK(rowCount <= this->rowCount() - row_, operation);
}

void FsstEncoding::checkFinalBlobPosition() const {
  if (row_ == this->rowCount()) {
    NIMBLE_CHECK_FILE_EQ(
        blobOffset_,
        blob_.size(),
        "FSST compressed lengths do not match the blob size.");
  }
}

void FsstEncoding::skip(uint32_t rowCount) {
  checkReadRange(rowCount, "Skipping past end of FSST encoding.");
  lengthBuffer_.resize(rowCount);
  lengths_->materialize(rowCount, lengthBuffer_.data());
  const auto compressedBytes = validateCompressedLengths(
      {lengthBuffer_.data(), lengthBuffer_.size()}, blob_, blobOffset_);
  row_ += rowCount;
  blobOffset_ += compressedBytes;
  checkFinalBlobPosition();
}

void FsstEncoding::materialize(uint32_t rowCount, void* buffer) {
  checkReadRange(rowCount, "Reading past end of FSST encoding.");
  lengthBuffer_.resize(rowCount);
  lengths_->materialize(rowCount, lengthBuffer_.data());
  validateCompressedLengths(
      {lengthBuffer_.data(), lengthBuffer_.size()}, blob_, blobOffset_);

  auto* output = static_cast<std::string_view*>(buffer);
  for (uint32_t i = 0; i < rowCount; ++i) {
    const auto compressedLen = lengthBuffer_[i];
    output[i] =
        decompressToStringBuffer(blob_.substr(blobOffset_, compressedLen));
    blobOffset_ += compressedLen;
  }
  row_ += rowCount;
  checkFinalBlobPosition();
}

std::string_view FsstEncoding::decompressToStringBuffer(
    std::string_view compressed) {
  if (compressed.empty()) {
    return {};
  }

  NIMBLE_CHECK_FILE_NE(
      static_cast<uint8_t>(compressed.back()),
      FSST_ESC,
      "FSST compressed string ends with an incomplete escape code.");
  NIMBLE_CHECK_FILE_LE(
      compressed.size(),
      std::numeric_limits<uint32_t>::max() / kMaxSymbolLength,
      "FSST decompressed string exceeds the supported size.");
  const size_t maxDecompressedSize = compressed.size() * kMaxSymbolLength;
  decompressBuffer_.resize(maxDecompressedSize);

  const auto decompressedSize = nimble_fsst_decompress(
      &decoder_,
      compressed.size(),
      reinterpret_cast<const unsigned char*>(compressed.data()),
      maxDecompressedSize,
      reinterpret_cast<unsigned char*>(decompressBuffer_.data()));

  NIMBLE_CHECK_GT(
      decompressedSize,
      0,
      "FSST decompression failed for non-empty compressed string.");
  NIMBLE_CHECK_FILE_LE(
      decompressedSize,
      maxDecompressedSize,
      "FSST decompressed string exceeds its output buffer.");

  ensurePage(decompressedSize);

  std::memcpy(
      currentPage_ + pageUsedBytes_,
      decompressBuffer_.data(),
      decompressedSize);
  std::string_view result(currentPage_ + pageUsedBytes_, decompressedSize);
  pageUsedBytes_ += decompressedSize;
  return result;
}

void FsstEncoding::ensurePage(size_t requiredBytes) {
  if (pageCapacityBytes_ - pageUsedBytes_ >= requiredBytes) {
    return;
  }
  const auto pageSize = std::max(kStringPageSize, requiredBytes);
  const size_t nextPageIndex =
      currentPage_ == nullptr ? 0 : currentPageIndex_ + 1;
  if (nextPageIndex < stringPages_.size() &&
      stringPages_[nextPageIndex].capacity >= pageSize) {
    currentPage_ = stringPages_[nextPageIndex].data;
    pageCapacityBytes_ = stringPages_[nextPageIndex].capacity;
  } else {
    StringPageSlot page{
        .data = static_cast<char*>(stringBufferFactory_(pageSize)),
        .capacity = pageSize};
    if (nextPageIndex < stringPages_.size()) {
      stringPages_[nextPageIndex] = page;
    } else {
      stringPages_.push_back(page);
    }
    currentPage_ = page.data;
    pageCapacityBytes_ = page.capacity;
  }
  currentPageIndex_ = nextPageIndex;
  pageUsedBytes_ = 0;
}

std::string_view FsstEncoding::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  {
    auto compressedValues = compressValues(values, &buffer.getMemoryPool());

    Buffer lengthsBuffer{buffer.getMemoryPool()};
    const std::string_view serializedLengths = encodeCompressedLengths(
        selection,
        {compressedValues.compressedLengths.data(),
         compressedValues.compressedLengths.size()},
        lengthsBuffer,
        options);

    const bool useVarint = options.useVarintRowCount;
    const auto valueCount = static_cast<uint32_t>(values.size());
    const uint32_t encodingSize =
        Encoding::serializePrefixSize(valueCount, useVarint) +
        varint::varintSize(compressedValues.symbolTableSize) +
        compressedValues.symbolTableSize +
        varint::varintSize(serializedLengths.size()) +
        serializedLengths.size() + compressedValues.totalCompressedSize;

    if (meetsCompressionTarget(
            compressedValues.totalInputSize,
            encodingSize,
            options.fsstCompressionTargetRatio)) {
      Buffer fsstBuffer{buffer.getMemoryPool()};
      char* reserved = fsstBuffer.reserve(encodingSize);
      char* pos = reserved;
      Encoding::serializePrefix(
          EncodingType::Fsst, DataType::String, valueCount, useVarint, pos);
      encoding::writeVarintString(
          {reinterpret_cast<const char*>(compressedValues.symbolTableData),
           compressedValues.symbolTableSize},
          pos);
      encoding::writeVarintString(serializedLengths, pos);
      for (uint32_t i = 0; i < valueCount; ++i) {
        encoding::writeBytes(
            {reinterpret_cast<const char*>(compressedValues.compressedPtrs[i]),
             compressedValues.compressedLengths[i]},
            pos);
      }

      NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
      std::string_view fsstEncoded{reserved, encodingSize};
      return buffer.writeString(fsstEncoded);
    }
  }

  Buffer trivialBuffer{buffer.getMemoryPool()};
  return buffer.writeString(
      encodeTrivialFallback(selection, values, trivialBuffer, options));
}

std::string_view FsstEncoding::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  validateEncodedPrefix(encoded, options);
  const auto sourceRowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_LE(offset, sourceRowCount);
  NIMBLE_CHECK_LE(length, sourceRowCount - offset);
  NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");

  const auto header = parseHeader(
      encoded, EncodingPrefix::prefixSize(encoded, options.useVarintRowCount));
  validateSymbolTable(header.symbolTable);

  const auto rowEnd = offset + length;
  Vector<uint32_t> materializedLengths{&buffer.getMemoryPool(), rowEnd};
  auto lengthsEncoding = EncodingFactory{}.create(
      buffer.getMemoryPool(),
      header.lengths,
      [](uint32_t /*totalLength*/) -> void* { return nullptr; },
      options);
  NIMBLE_CHECK_FILE_EQ(
      lengthsEncoding->dataType(),
      DataType::Uint32,
      "FSST lengths encoding must contain Uint32 values.");
  NIMBLE_CHECK_FILE(
      !lengthsEncoding->isNullable(),
      "FSST lengths encoding must not be nullable.");
  NIMBLE_CHECK_FILE_EQ(
      lengthsEncoding->rowCount(),
      sourceRowCount,
      "FSST lengths row count does not match the parent encoding.");
  lengthsEncoding->materialize(rowEnd, materializedLengths.data());

  const auto blobOffset = std::accumulate(
      materializedLengths.begin(),
      materializedLengths.begin() + offset,
      size_t{0});
  const auto blobBytes = std::accumulate(
      materializedLengths.begin() + offset,
      materializedLengths.end(),
      size_t{0});
  validateCompressedLengths(
      {materializedLengths.data(), materializedLengths.size()}, header.blob, 0);
  if (rowEnd == sourceRowCount) {
    NIMBLE_CHECK_FILE_EQ(
        blobOffset + blobBytes,
        header.blob.size(),
        "FSST compressed lengths do not match the blob size.");
  }

  auto* pool = &buffer.getMemoryPool();
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  const auto slicedLengths = EncodingFactory::slice(
      header.lengths, offset, length, scopedBuffer.get(), options);

  NIMBLE_CHECK_FILE_LE(
      slicedLengths.size(),
      std::numeric_limits<uint32_t>::max(),
      "Sliced FSST lengths encoding exceeds the supported size.");
  const uint64_t fixedEncodingSize =
      EncodingPrefix::serializedSize(length, options.useVarintRowCount) +
      varint::varintSize(header.symbolTable.size()) +
      header.symbolTable.size() + varint::varintSize(slicedLengths.size()) +
      slicedLengths.size();
  constexpr auto kMaxEncodingSize =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
  NIMBLE_CHECK_FILE_LE(
      fixedEncodingSize,
      kMaxEncodingSize,
      "Sliced FSST encoding exceeds the supported size.");
  NIMBLE_CHECK_FILE_LE(
      blobBytes,
      kMaxEncodingSize - fixedEncodingSize,
      "Sliced FSST encoding exceeds the supported size.");
  const auto encodingSize =
      static_cast<uint32_t>(fixedEncodingSize + blobBytes);
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  EncodingPrefix::serialize(
      EncodingType::Fsst,
      DataType::String,
      length,
      options.useVarintRowCount,
      pos);
  encoding::writeVarintString(header.symbolTable, pos);
  encoding::writeVarintString(slicedLengths, pos);
  encoding::writeBytes(header.blob.substr(blobOffset, blobBytes), pos);
  NIMBLE_CHECK_EQ(
      static_cast<uint64_t>(pos - reserved),
      encodingSize,
      "Encoding size mismatch.");
  return {reserved, encodingSize};
}

uint64_t FsstEncoding::estimateSize(
    uint64_t rowCount,
    const Statistics<std::string_view>& statistics,
    const Encoding::Options& options) {
  const uint64_t estimatedBlobSize = static_cast<uint64_t>(
      statistics.totalStringsLength() * options.fsstCompressionTargetRatio);
  const uint64_t estimatedMaxCompressedLength = static_cast<uint64_t>(
      std::ceil(statistics.max().size() * options.fsstCompressionTargetRatio));
  const uint64_t estimatedLengthsSize =
      FixedBitWidthEncoding<uint32_t>::estimateSize(
          rowCount, 0, estimatedMaxCompressedLength, options);
  return Encoding::serializePrefixSize(rowCount, options.useVarintRowCount) +
      varint::varintSize(kSymbolTableOverhead) + kSymbolTableOverhead +
      varint::varintSize(estimatedLengthsSize) + estimatedLengthsSize +
      estimatedBlobSize;
}

std::string FsstEncoding::debugString(int offset) const {
  return fmt::format(
      "{}FsstEncoding: {} rows", std::string(offset, ' '), rowCount());
}

} // namespace facebook::nimble
