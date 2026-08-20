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

} // namespace

FsstEncoding::CompressedValues::CompressedValues(
    velox::memory::MemoryPool* pool)
    : compressedBuffer{pool}, compressedLengths{pool}, compressedPtrs{pool} {}

FsstEncoding::Header FsstEncoding::parseHeader(const char* pos) {
  Header header{};
  header.symbolTableSize = varint::readVarint32(&pos);
  header.symbolTable = pos;
  pos += header.symbolTableSize;
  header.lengthsSize = varint::readVarint32(&pos);
  header.lengths = pos;
  header.blob = pos + header.lengthsSize;
  return header;
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
    : TypedEncoding<std::string_view, std::string_view>{pool, data, options},
      stringBufferFactory_{std::move(stringBufferFactory)},
      lengthBuffer_{&pool},
      decompressBuffer_{&pool} {
  const auto header = parseHeader(data.data() + this->dataOffset());
  NIMBLE_CHECK_GT(
      header.symbolTableSize, 0, "FSST symbol table size must be positive.");
  const auto bytesConsumed = nimble_fsst_import(
      &decoder_,
      const_cast<unsigned char*>(
          reinterpret_cast<const unsigned char*>(header.symbolTable)));
  NIMBLE_CHECK_EQ(
      static_cast<uint32_t>(bytesConsumed),
      header.symbolTableSize,
      "FSST symbol table import size mismatch.");

  lengths_ = EncodingFactory(options).create(
      pool, {header.lengths, header.lengthsSize}, stringBufferFactory_);
  blob_ = header.blob;
  pos_ = blob_;
}

std::string_view FsstEncoding::lengthsEncoding(
    std::string_view encoding,
    const Encoding::Options& options) {
  NIMBLE_CHECK_GE(
      encoding.size(),
      EncodingPrefix::kRowCountOffset,
      "FSST encoding too small.");
  NIMBLE_CHECK_EQ(
      static_cast<EncodingType>(encoding[EncodingPrefix::kEncodingTypeOffset]),
      EncodingType::Fsst,
      "Expected FSST encoding.");

  const auto prefixSize =
      EncodingPrefix::prefixSize(encoding, options.useVarintRowCount);
  NIMBLE_CHECK_GE(encoding.size(), prefixSize, "FSST encoding too small.");
  const auto header = parseHeader(encoding.data() + prefixSize);
  return {header.lengths, header.lengthsSize};
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
  pos_ = blob_;
  lengths_->reset();
}

void FsstEncoding::skip(uint32_t rowCount) {
  lengthBuffer_.resize(rowCount);
  lengths_->materialize(rowCount, lengthBuffer_.data());
  row_ += rowCount;
  pos_ +=
      std::accumulate(lengthBuffer_.begin(), lengthBuffer_.end(), uint64_t{0});
}

void FsstEncoding::materialize(uint32_t rowCount, void* buffer) {
  lengthBuffer_.resize(rowCount);
  lengths_->materialize(rowCount, lengthBuffer_.data());

  auto* output = static_cast<std::string_view*>(buffer);
  for (uint32_t i = 0; i < rowCount; ++i) {
    const auto compressedLen = lengthBuffer_[i];
    output[i] = decompressToStringBuffer(pos_, compressedLen);
    pos_ += compressedLen;
  }
  row_ += rowCount;
}

std::string_view FsstEncoding::decompressToStringBuffer(
    const char* compressedData,
    uint32_t compressedLength) {
  if (compressedLength == 0) {
    return {};
  }

  const size_t maxDecompressedSize =
      static_cast<size_t>(compressedLength) * kMaxSymbolLength;
  decompressBuffer_.resize(maxDecompressedSize);

  const auto decompressedSize = nimble_fsst_decompress(
      &decoder_,
      compressedLength,
      reinterpret_cast<const unsigned char*>(compressedData),
      maxDecompressedSize,
      reinterpret_cast<unsigned char*>(decompressBuffer_.data()));

  NIMBLE_CHECK_GT(
      decompressedSize,
      0,
      "FSST decompression failed for non-empty compressed string.");

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
  currentPage_ = static_cast<char*>(stringBufferFactory_(pageSize));
  pageCapacityBytes_ = pageSize;
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
  const auto sourceRowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_LE(offset, sourceRowCount);
  NIMBLE_CHECK_LE(length, sourceRowCount - offset);
  NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");

  const auto header = parseHeader(
      encoded.data() +
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount));
  const std::string_view lengths{
      header.lengths, static_cast<size_t>(header.lengthsSize)};

  const auto rowEnd = offset + length;
  Vector<uint32_t> materializedLengths{&buffer.getMemoryPool(), rowEnd};
  EncodingFactory{}
      .create(
          buffer.getMemoryPool(),
          lengths,
          [](uint32_t /*totalLength*/) -> void* { return nullptr; },
          options)
      ->materialize(rowEnd, materializedLengths.data());

  const auto blobOffset = std::accumulate(
      materializedLengths.begin(),
      materializedLengths.begin() + offset,
      uint32_t{0});
  const auto blobBytes = std::accumulate(
      materializedLengths.begin() + offset,
      materializedLengths.end(),
      uint32_t{0});

  auto* pool = &buffer.getMemoryPool();
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  const auto slicedLengths = EncodingFactory::slice(
      lengths, offset, length, scopedBuffer.get(), options);

  const uint32_t encodingSize =
      EncodingPrefix::serializedSize(length, options.useVarintRowCount) +
      varint::varintSize(header.symbolTableSize) + header.symbolTableSize +
      varint::varintSize(slicedLengths.size()) + slicedLengths.size() +
      blobBytes;
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  EncodingPrefix::serialize(
      EncodingType::Fsst,
      DataType::String,
      length,
      options.useVarintRowCount,
      pos);
  encoding::writeVarintString(
      {header.symbolTable, static_cast<size_t>(header.symbolTableSize)}, pos);
  encoding::writeVarintString(slicedLengths, pos);
  encoding::writeBytes({header.blob + blobOffset, blobBytes}, pos);
  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
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
