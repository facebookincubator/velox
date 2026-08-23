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
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/legacy/Encoding.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"

// A dictionary encoded stream is comprised of two pieces: a mapping from the
// n unique values in a stream to the integers [0, n) and the vector of indices
// that scatter those uniques back to the original ordering.

namespace facebook::nimble::legacy {

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

  static const int kAlphabetSizeOffset = EncodingPrefix::kFixedPrefixSize;

  DictionaryEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory);

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer);

  std::string debugString(int offset) const final;

 private:
  // Stores pre-loaded alphabet
  Vector<T> alphabet_;

  // Indices are uint32_t.
  std::unique_ptr<Encoding> indicesEncoding_;
  std::unique_ptr<Encoding> alphabetEncoding_;
  // Temporary buffer.
  Vector<uint32_t> indicesBuffer_;
};

//
// End of public API. Implementation follows.
//

template <typename T>
DictionaryEncoding<T>::DictionaryEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory)
    : TypedEncoding<T, physicalType>{pool, data},
      alphabet_{this->pool_},
      indicesBuffer_{this->pool_} {
  const EncodingFactory factory;
  const auto* pos = data.data() + kAlphabetSizeOffset;
  const uint32_t alphabetSize = encoding::readUint32(pos);
  alphabetEncoding_ = factory.create(
      *this->pool_,
      {pos, alphabetSize},
      stringBufferFactory,
      Encoding::Options{});
  const uint32_t alphabetCount = alphabetEncoding_->rowCount();
  alphabet_.resize(alphabetCount);
  alphabetEncoding_->materialize(alphabetCount, alphabet_.data());

  pos += alphabetSize;
  indicesEncoding_ = factory.create(
      *this->pool_,
      {pos, static_cast<size_t>(data.end() - pos)},
      stringBufferFactory,
      Encoding::Options{});
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
  indicesBuffer_.resize(rowCount);
  indicesEncoding_->materialize(rowCount, indicesBuffer_.data());

  T* output = static_cast<T*>(buffer);
  for (uint32_t index : indicesBuffer_) {
    *output = alphabet_[index];
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
  indicesBuffer_.resize(visitor.numRows() - startRowIndex);
  velox::common::AlwaysTrue indicesFilter;
  detail::DictionaryIndicesHook indicesHook(
      indicesBuffer_.data(), startRowIndex);
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
  legacy::callReadWithVisitor(*indicesEncoding_, indicesVisitor, params);
  detail::readWithVisitorSlow(visitor, params, nullptr, [&] {
    const auto index = indicesBuffer_[visitor.rowIndex() - startRowIndex];
    return alphabet_[index];
  });
}

template <typename T>
std::string_view DictionaryEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer) {
  const uint32_t valueCount = values.size();
  const uint32_t alphabetCount =
      selection.statistics().uniqueCounts().value().size();

  folly::F14FastMap<physicalType, uint32_t> alphabetMapping;
  alphabetMapping.reserve(alphabetCount);
  Vector<physicalType> alphabet{&buffer.getMemoryPool()};
  alphabet.resize(alphabetCount);
  Vector<uint32_t> indices{&buffer.getMemoryPool()};
  indices.resize(valueCount);

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

  Buffer tempBuffer{buffer.getMemoryPool()};
  std::string_view serializedAlphabet =
      selection.template encodeNested<physicalType>(
          EncodingIdentifiers::Dictionary::Alphabet, {alphabet}, tempBuffer);
  std::string_view serializedIndices =
      selection.template encodeNested<uint32_t>(
          EncodingIdentifiers::Dictionary::Indices, {indices}, tempBuffer);

  const uint32_t encodingSize = EncodingPrefix::kFixedPrefixSize + 4 +
      serializedAlphabet.size() + serializedIndices.size();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::Dictionary,
      TypeTraits<T>::dataType,
      valueCount,
      false,
      pos);
  encoding::writeUint32(serializedAlphabet.size(), pos);
  encoding::writeBytes(serializedAlphabet, pos);
  encoding::writeBytes(serializedIndices, pos);
  NIMBLE_DCHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
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

} // namespace facebook::nimble::legacy
