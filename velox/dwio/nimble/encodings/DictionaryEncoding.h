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
#pragma once

#include <span>
#include "folly/container/F14Map.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/NestedAlpSizeEstimation.h"

// A dictionary encoded stream is comprised of two pieces: a mapping from the
// n unique values in a stream to the integers [0, n) and the vector of indices
// that scatter those uniques back to the original ordering.

namespace facebook::nimble {

/// The layout for a dictionary encoding is:
/// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
/// 4 bytes: alphabet size
/// XX bytes: alphabet encoding bytes
/// YY bytes: indices encoding bytes
template <typename T>
class DictionaryEncoding
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  DictionaryEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});

  ~DictionaryEncoding() override {
    this->releaseBuffer(indicesBuffer_);
  }

  DictionaryEncoding(const DictionaryEncoding&) = delete;
  DictionaryEncoding& operator=(const DictionaryEncoding&) = delete;
  DictionaryEncoding(DictionaryEncoding&&) = delete;
  DictionaryEncoding& operator=(DictionaryEncoding&&) = delete;

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  /// Reads dictionary indices only, without looking up values from the
  /// alphabet. This is used for dictionary-aware readers that return
  /// DictionaryVector. Non-legacy encodings only.
  template <typename V>
  void readIndicesWithVisitor(V& visitor, ReadWithVisitorParams& params);

  void materializeIndices(uint32_t rowCount, uint32_t* buffer) override {
    indicesEncoding_->materialize(rowCount, buffer);
  }

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static uint64_t estimateSize(
      uint64_t rowCount,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options = {}) {
    // Estimate the two nested streams produced by Dictionary:
    //
    //   alphabet: unique values. Floating-point alphabets may use ALP when
    //     nested ALP selection is enabled.
    //   indices: one dictionary index per row, estimated as FixedBitWidth.
    const auto& uniqueCounts = statistics.uniqueCounts().value();
    const uint64_t uniqueCount = uniqueCounts.size();
    const uint64_t indicesEncodingSize =
        FixedBitWidthEncoding<uint32_t>::estimateSize(
            rowCount,
            /*minValue=*/0,
            uniqueCount == 0 ? 0 : uniqueCount - 1,
            options);

    uint64_t alphabetEncodingSize{};
    if constexpr (isStringType<physicalType>()) {
      // Get the total blob size for all (unique) strings.
      alphabetEncodingSize = TrivialEncoding<std::string_view>::estimateSize(
          uniqueCount,
          uniqueCounts.uniqueStringBytes(),
          statistics.min().size(),
          statistics.max().size(),
          options);
    } else {
      alphabetEncodingSize = std::min(
          TrivialEncoding<physicalType>::estimateSize(uniqueCount),
          FixedBitWidthEncoding<physicalType>::estimateSize(
              uniqueCount, statistics.min(), statistics.max(), options));
      // TODO: Consider PFOR after its statistics can be derived from unique
      // values without rebuilding Statistics for every dictionary estimate.
      if constexpr (isFloatingPointType<T>()) {
        if (const auto alpEncodingSize =
                detail::uniqueValuesNestedAlpSize<T>(uniqueCounts, options)) {
          alphabetEncodingSize =
              std::min(alphabetEncodingSize, *alpEncodingSize);
        }
      }
    }
    const uint64_t outerEncodingSize =
        EncodingPrefix::kFixedPrefixSize + sizeof(uint32_t);
    return outerEncodingSize + alphabetEncodingSize + indicesEncodingSize;
  }

  std::string debugString(int offset) const final;

  bool dictionaryEnabled() const override {
    return true;
  }

  uint32_t dictionarySize() const override {
    return alphabet_.size();
  }

  const void* dictionaryEntry(uint32_t index) const override {
    return &alphabet_[index];
  }

  const void* dictionaryEntries() const override {
    return alphabet_.data();
  }

  const Encoding* indicesEncoding() const {
    return indicesEncoding_.get();
  }

 private:
  static std::string_view encodeAlphabet(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> alphabet,
      Buffer& buffer,
      const Encoding::Options& options);

  uint32_t* ensureIndicesBuffer(uint32_t numElements) {
    const auto bytes = numElements * sizeof(uint32_t);
    if (indicesBuffer_ == nullptr || indicesBuffer_->capacity() < bytes) {
      indicesBuffer_ = this->getBuffer(bytes);
    }
    return indicesBuffer_->asMutable<uint32_t>();
  }

  Vector<T> alphabet_;
  std::unique_ptr<Encoding> indicesEncoding_;
  std::unique_ptr<Encoding> alphabetEncoding_;
  velox::BufferPtr indicesBuffer_;
};

//
// End of public API. Implementation follows.
//

template <typename T>
DictionaryEncoding<T>::DictionaryEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options},
      alphabet_{this->pool_} {
  const auto* pos = data.data() + this->dataOffset();
  const uint32_t alphabetSize = encoding::readUint32(pos);
  alphabetEncoding_ = EncodingFactory().create(
      *this->pool_, {pos, alphabetSize}, stringBufferFactory, options);
  const uint32_t alphabetCount = alphabetEncoding_->rowCount();
  alphabet_.resize(alphabetCount);
  alphabetEncoding_->materialize(alphabetCount, alphabet_.data());

  pos += alphabetSize;
  indicesEncoding_ = EncodingFactory().create(
      *this->pool_,
      {pos, static_cast<size_t>(data.data() + data.size() - pos)},
      stringBufferFactory,
      options);
}

template <typename T>
void DictionaryEncoding<T>::reset() {
  indicesEncoding_->reset();
}

template <typename T>
void DictionaryEncoding<T>::skip(uint32_t rowCount) {
  indicesEncoding_->skip(rowCount);
}

template <typename T>
void DictionaryEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  auto* rawIndices = ensureIndicesBuffer(rowCount);
  indicesEncoding_->materialize(rowCount, rawIndices);

  T* output = static_cast<T*>(buffer);
  for (uint32_t i = 0; i < rowCount; ++i) {
    *output = alphabet_[rawIndices[i]];
    ++output;
  }
}

namespace detail {

class DictionaryIndicesHook : public velox::ValueHook {
 public:
  static constexpr bool kSkipNulls = true;

  DictionaryIndicesHook(uint32_t* indices, vector_size_t offset)
      : indices_(indices), offset_(offset) {}

  bool acceptsNulls() const final {
    return false;
  }

  void addValue(vector_size_t i, int64_t value) final {
    indices_[i - offset_] = value;
  }

  void addValues(
      const vector_size_t* rows,
      const int32_t* values,
      vector_size_t size) final {
    for (vector_size_t i = 0; i < size; ++i) {
      indices_[rows[i] - offset_] = values[i];
    }
  }

  void addNull(vector_size_t /*i*/) final {
    NIMBLE_UNREACHABLE(__PRETTY_FUNCTION__);
  }

 private:
  uint32_t* const indices_;
  const vector_size_t offset_;
};

} // namespace detail

template <typename T>
template <typename V>
void DictionaryEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  if constexpr (sizeof(T) < sizeof(uint32_t)) {
    // Although we do not use column reader values buffer in this decoder, the
    // fast path in nested decoders of indices can use it, so need to ensure it
    // can hold the indices.
    visitor.reader().template ensureValuesCapacity<uint32_t>(
        visitor.numRows(), true);
  }
  const auto startRowIndex = visitor.rowIndex();
  const auto numIndices = visitor.numRows() - startRowIndex;
  auto* rawIndices = ensureIndicesBuffer(numIndices);
  velox::common::AlwaysTrue indicesFilter;
  detail::DictionaryIndicesHook indicesHook(rawIndices, startRowIndex);
  auto indicesVisitor = DecoderVisitor<
      int32_t,
      velox::common::AlwaysTrue,
      velox::dwio::common::ExtractToHook<detail::DictionaryIndicesHook>,
      V::dense>(
      indicesFilter,
      &visitor.reader(),
      velox::RowSet(visitor.rows(), visitor.numRows()),
      velox::dwio::common::ExtractToHook<detail::DictionaryIndicesHook>(
          &indicesHook));
  indicesVisitor.setRowIndex(startRowIndex);
  callReadWithVisitor(*indicesEncoding_, indicesVisitor, params);
  detail::readWithVisitorSlow(visitor, params, nullptr, [&] {
    return alphabet_[rawIndices[visitor.rowIndex() - startRowIndex]];
  });
}

/// Reads indices only without looking up values from the alphabet.
/// This is used for dictionary-aware readers that return DictionaryVector.
/// Uses materializeIndices to avoid recursing through the visitor machinery,
/// which would incorrectly call addNumValues from the inner encoding.
template <typename T>
template <typename V>
void DictionaryEncoding<T>::readIndicesWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  NIMBLE_CHECK(this->dictionaryEnabled());
  NIMBLE_CHECK(
      !V::kHasHook, "readIndicesWithVisitor does not support value hooks");
  const auto numReadRows =
      visitor.rowAt(visitor.numRows() - 1) - params.numScanned + 1;
  auto* rawNulls = visitor.reader().rawNullsInReadRange();
  const auto numNonNulls = rawNulls != nullptr
      ? velox::bits::countNonNulls(
            rawNulls, params.numScanned, params.numScanned + numReadRows)
      : numReadRows;

  // Dense fast path: materialize directly into rawValues_, scatter for nulls.
  if (V::dense) {
    NIMBLE_CHECK_EQ(
        visitor.rowAt(visitor.numRows() - 1),
        visitor.rowAt(0) + visitor.numRows() - 1,
        "Dense visitor must have contiguous rows");
    detail::readDenseMaterializedIndices(
        *this, visitor, params, rawNulls, numReadRows, numNonNulls);
    return;
  }

  // Sparse path: materialize all rows into buffer, gather only the
  // requested positions into rawValues_.
  // TODO: Use FBW's random-access fixedBitArray_.get() to avoid
  // materializing unrequested rows. Requires templated
  // materializeIndices — see commented-out API mock-ups in this file
  // and EncodingUtils.h.
  auto* rawIndices = ensureIndicesBuffer(numNonNulls);
  detail::readSparseMaterializedIndices(
      *this,
      visitor,
      params.numScanned,
      params.prepareResultNulls,
      rawNulls,
      numReadRows,
      numNonNulls,
      rawIndices);
}

template <typename T>
std::string_view DictionaryEncoding<T>::encodeAlphabet(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> alphabet,
    Buffer& buffer,
    const Encoding::Options& options) {
  if constexpr (isFloatingPointType<T>()) {
    ScopedVector<T> logicalAlphabet{
        alphabet.size(), &buffer.getMemoryPool(), options.bufferPool};
    for (uint32_t i = 0; i < alphabet.size(); ++i) {
      logicalAlphabet[i] =
          EncodingPhysicalType<T>::asEncodingLogicalType(alphabet[i]);
    }

    auto nestedPolicy = std::unique_ptr<EncodingSelectionPolicy<T>>(
        static_cast<EncodingSelectionPolicy<T>*>(
            selection
                .template createNestedPolicy<T>(
                    selection.encodingType(),
                    EncodingIdentifiers::Dictionary::Alphabet)
                .release()));
    return EncodingFactory::encode<T>(
        std::move(nestedPolicy),
        std::span<const T>{logicalAlphabet.data(), logicalAlphabet.size()},
        buffer,
        options);
  } else {
    return selection.template encodeNested<physicalType>(
        EncodingIdentifiers::Dictionary::Alphabet, {alphabet}, buffer, options);
  }
}

template <typename T>
std::string_view DictionaryEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  const bool useVarint = options.useVarintRowCount;
  const uint32_t valueCount = values.size();
  const uint32_t alphabetCount =
      selection.statistics().uniqueCounts().value().size();
  auto* pool = &buffer.getMemoryPool();

  folly::F14FastMap<physicalType, uint32_t> alphabetMapping;
  alphabetMapping.reserve(alphabetCount);
  ScopedVector<physicalType> alphabet{alphabetCount, pool, options.bufferPool};
  ScopedVector<uint32_t> indices{valueCount, pool, options.bufferPool};

  uint32_t alphabetIndex{0};
  uint32_t indiciesIndex{0};
  auto valuesIt = values.begin();

  // Based on experimentation, there are at least two consistently
  // "good" ordering approaches that result in consistently smaller storage
  // footprints:
  //
  // 1. Sort alphabet values using natural order. This improves alphabet
  // compression, but can result in worse indices compression.
  //
  // 2. Order the alphabet by the first occurrence of each value in the input.
  // Storage savings mostly come from indices being much more delta encoding
  // friendly compared to a random order.
  //
  // Other approaches, like sorting alphabet values by frequency, also
  // occasionally yield more compressible data, but they're not consistent and
  // can result in storage regression compared to the first two.
  //
  // The second approach is used in this implementation because it's essentially
  // free: we don't need to allocate O(m) memory for a sorting buffer and
  // then do O(m log m) sorting. However, the first approach has a slightly
  // better distribution curve for storage savings.

  // Step one: fill alphabet and indices until we get a full alphabet mapping.
  for (; alphabetIndex < alphabetCount; ++valuesIt, ++indiciesIndex) {
    NIMBLE_DCHECK(valuesIt != values.end(), "Values iterator out of bounds.");
    const auto& value = *valuesIt;
    const auto& it = alphabetMapping.find(value);
    if (it == alphabetMapping.end()) {
      alphabetMapping.emplace(value, alphabetIndex);
      alphabet[alphabetIndex] = value;
      indices[indiciesIndex] = alphabetIndex;
      ++alphabetIndex;
    } else {
      indices[indiciesIndex] = it->second;
    }
  }

  // Step two: fill indices for the remaining values using the already fully
  // populated alphabet mapping. In most cases, depending on the data skew, we
  // will hit this step relatively early, and it will boost performance by
  // 20-30%.
  for (; valuesIt != values.end(); ++valuesIt) {
    NIMBLE_DCHECK(
        alphabetMapping.find(*valuesIt) != alphabetMapping.end(),
        "Mapping corruption. Missing alphabet entry.");
    indices[indiciesIndex++] = alphabetMapping.find(*valuesIt)->second;
  }

  NIMBLE_CHECK_EQ(indices.size(), valueCount, "Indices size mismatch.");
  NIMBLE_CHECK_EQ(alphabet.size(), alphabetCount, "Alphabet size mismatch.");

  ScopedEncodingBuffer scopedBuffer{
      &buffer.getMemoryPool(), options.encodingBufferPool};
  std::string_view serializedAlphabet = encodeAlphabet(
      selection,
      {alphabet.data(), alphabet.size()},
      scopedBuffer.get(),
      options);
  std::string_view serializedIndices =
      selection.template encodeNested<uint32_t>(
          EncodingIdentifiers::Dictionary::Indices,
          {indices},
          scopedBuffer.get(),
          options);

  const uint32_t encodingSize =
      Encoding::serializePrefixSize(valueCount, useVarint) + 4 +
      serializedAlphabet.size() + serializedIndices.size();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::Dictionary,
      TypeTraits<T>::dataType,
      valueCount,
      useVarint,
      pos);
  encoding::writeUint32(serializedAlphabet.size(), pos);
  encoding::writeBytes(serializedAlphabet, pos);
  encoding::writeBytes(serializedIndices, pos);
  NIMBLE_DCHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string_view DictionaryEncoding<T>::slice(
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

  const char* pos = encoded.data() +
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
  const uint32_t alphabetSize = encoding::readUint32(pos);
  const std::string_view alphabet{pos, alphabetSize};
  pos += alphabetSize;
  const std::string_view indices{
      pos, static_cast<size_t>(encoded.data() + encoded.size() - pos)};

  auto* pool = &buffer.getMemoryPool();
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  const auto slicedIndices = EncodingFactory::slice(
      indices, offset, length, scopedBuffer.get(), options);

  const uint32_t encodingSize =
      EncodingPrefix::serializedSize(length, options.useVarintRowCount) +
      sizeof(uint32_t) + alphabet.size() + slicedIndices.size();
  char* reserved = buffer.reserve(encodingSize);
  char* writePos = reserved;
  EncodingPrefix::serialize(
      EncodingType::Dictionary,
      TypeTraits<T>::dataType,
      length,
      options.useVarintRowCount,
      writePos);
  encoding::writeUint32(alphabet.size(), writePos);
  encoding::writeBytes(alphabet, writePos);
  encoding::writeBytes(slicedIndices, writePos);
  NIMBLE_CHECK_EQ(writePos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string DictionaryEncoding<T>::debugString(int offset) const {
  std::string log = Encoding::debugString(offset);
  log += fmt::format(
      "\n{}indices child:\n{}",
      std::string(offset, ' '),
      indicesEncoding_->debugString(offset + 2));
  log += fmt::format(
      "\n{}alphabet:\n{}entries={}",
      std::string(offset, ' '),
      std::string(offset + 2, ' '),
      alphabet_.size());
  return log;
}

} // namespace facebook::nimble
