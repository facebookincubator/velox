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

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

template <typename T>
class TrivialEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  TrivialEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Trivial);
    const char* pos = data.data() + this->dataOffset_;
    NIMBLE_CHECK_FILE_GE(
        this->remainingBytes(pos), 1, "Truncated Trivial encoding.");
    const auto compressionType =
        static_cast<CompressionType>(encoding::readChar(pos));
    const auto payload = this->decompressPayload(
        compressionType,
        TypeTraits<T>::dataType,
        {pos, this->remainingBytes(pos)});
    values_ = reinterpret_cast<const physicalType*>(payload.data());
    NIMBLE_CHECK_FILE_EQ(
        payload.size(),
        static_cast<size_t>(this->rowCount_) * sizeof(physicalType),
        "Unexpected Trivial view end.");
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    return detail::castFromPhysicalType<T>(values_[index]);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    std::copy(values_ + offset, values_ + offset + length, output);
  }

  const physicalType* values_;
};

template <>
class TrivialEncodingView<bool> final : public TypedEncodingView<bool> {
 public:
  TrivialEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<bool>{data, pool, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Trivial);
    const char* pos = data.data() + this->dataOffset_;
    NIMBLE_CHECK_FILE_GE(
        this->remainingBytes(pos), 1, "Truncated Trivial encoding.");
    const auto compressionType =
        static_cast<CompressionType>(encoding::readChar(pos));
    const auto payload = this->decompressPayload(
        compressionType, DataType::Undefined, {pos, this->remainingBytes(pos)});
    bitmap_ = payload.data();
    NIMBLE_CHECK_FILE_EQ(
        payload.size(),
        static_cast<size_t>(FixedBitArray::bufferSize(this->rowCount_, 1)),
        "Unexpected Trivial bool view end.");
  }

  /// Counts the set bits of the bitmap a word at a time, without reading rows.
  uint32_t count(uint32_t end, bool value) const final {
    this->checkReadRange(end, 0);
    const auto* bytes = reinterpret_cast<const uint8_t*>(bitmap_);
    uint32_t numSet{0};
    uint32_t row{0};
    for (; row + 64 <= end; row += 64) {
      // The bitmap may not be aligned to a word.
      uint64_t word;
      std::memcpy(&word, bytes + row / 8, sizeof(word));
      numSet += std::popcount(word);
    }
    if (row < end) {
      // Reads the bytes of the last word, which holds fewer than 64 rows, and
      // keeps its first rows. The first row of a byte is its lowest bit.
      uint64_t word{0};
      std::memcpy(&word, bytes + row / 8, (end - row + 7) / 8);
      numSet += std::popcount(word & ((1ULL << (end - row)) - 1));
    }
    return value ? numSet : end - numSet;
  }

 private:
  bool readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    return velox::bits::isBitSet(
        reinterpret_cast<const uint8_t*>(bitmap_), index);
  }

  void readPhysical(uint32_t offset, uint32_t length, bool* output)
      const final {
    this->checkReadRange(offset, length);
    const auto* bitmap = reinterpret_cast<const uint8_t*>(bitmap_);
    for (uint32_t i = 0; i < length; ++i) {
      output[i] = velox::bits::isBitSet(bitmap, offset + i);
    }
  }

  const char* bitmap_{nullptr};
};

template <>
class TrivialEncodingView<std::string_view> final
    : public TypedEncodingView<std::string_view> {
 public:
  TrivialEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<std::string_view>{data, pool, options},
        offsets_{this->template getVectorBuffer<uint32_t>()} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Trivial);
    const char* pos = data.data() + this->dataOffset_;
    NIMBLE_CHECK_FILE_GE(
        this->remainingBytes(pos), 1, "Truncated Trivial encoding.");
    const auto compressionType =
        static_cast<CompressionType>(encoding::readChar(pos));
    const auto encodedLengths =
        this->readSizedBytes(pos, "Trivial string lengths");
    const auto payload = this->decompressPayload(
        compressionType, DataType::String, {pos, this->remainingBytes(pos)});
    blob_ = payload.data();
    blobSize_ = payload.size();
    if (options.cacheEncodingView) {
      initializeCached(encodedLengths);
    } else {
      initializeUncached(encodedLengths);
    }
  }

  ~TrivialEncodingView() override {
    this->releaseVectorBuffer(offsets_);
  }

 private:
  // Reads the length of every row once and keeps the offset of every row, so
  // that reads need no stream access.
  void initializeCached(std::string_view encodedLengths) {
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto lengths = EncodingFactory().create(
        *this->pool_, encodedLengths, noStringBufferFactory, this->options_);
    NIMBLE_CHECK_NOT_NULL(lengths);
    offsets_.resize(this->rowCount_ + 1);
    offsets_[0] = 0;
    lengths->materialize(this->rowCount_, offsets_.data() + 1);
    for (uint32_t i = 0; i < this->rowCount_; ++i) {
      offsets_[i + 1] += offsets_[i];
    }
    NIMBLE_CHECK_FILE_EQ(
        offsets_.back(), blobSize_, "Unexpected Trivial string view end.");
  }

  // Keeps the lengths stream as a view and derives the offset of each read from
  // it.
  void initializeUncached(std::string_view encodedLengths) {
    lengths_ = detail::createTypedEncodingView<uint32_t>(
        encodedLengths, this->pool_, this->options_);
    NIMBLE_CHECK_NOT_NULL(lengths_);
    NIMBLE_CHECK_FILE_EQ(
        lengths_->rowCount(),
        this->rowCount_,
        "Trivial string lengths must match the row count.");
  }

  // Returns whether the view holds the offset of every row.
  bool cachesOffsets() const {
    return this->options_.cacheEncodingView;
  }

  std::string_view readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    if (cachesOffsets()) {
      return {blob_ + offsets_[index], offsets_[index + 1] - offsets_[index]};
    }
    return valueAfter(lengths_->sum(0, index), index);
  }

  void readPhysical(uint32_t offset, uint32_t length, std::string_view* output)
      const final {
    this->checkReadRange(offset, length);
    if (cachesOffsets()) {
      for (uint32_t i = 0; i < length; ++i) {
        output[i] = {
            blob_ + offsets_[offset + i],
            offsets_[offset + i + 1] - offsets_[offset + i]};
      }
      return;
    }
    // Sums the lengths before the range once and then walks the range.
    auto begin = lengths_->sum(0, offset);
    for (uint32_t i = 0; i < length; ++i) {
      output[i] = valueAfter(begin, offset + i);
      begin += output[i].size();
    }
  }

  void readPhysicalAt(
      std::span<const uint32_t> indices,
      std::string_view* output) const final {
    if (cachesOffsets()) {
      TypedEncodingView<std::string_view>::readPhysicalAt(indices, output);
      return;
    }
    // Sums the lengths once for all indices, not once for each of them.
    ScopedVector<uint64_t> begins{
        indices.size(), this->pool_, this->options_.bufferPool};
    lengths_->sum(indices, begins.data());
    for (size_t i = 0; i < indices.size(); ++i) {
      output[i] = valueAfter(begins[i], indices[i]);
    }
  }

  // Returns the string of a row that starts 'begin' bytes into the blob.
  std::string_view valueAfter(uint64_t begin, uint32_t row) const {
    const auto size = lengths_->readAt(row);
    NIMBLE_CHECK_LE(
        begin + size, blobSize_, "Trivial string exceeds its payload.");
    return {blob_ + begin, size};
  }

  // The concatenated string bytes, and their size in bytes.
  const char* blob_{nullptr};
  size_t blobSize_{0};
  // Holds the offset of every row. Empty without cacheEncodingView.
  Vector<uint32_t> offsets_;
  // Reads the length of one row at a time. Set only without cacheEncodingView.
  std::unique_ptr<TypedEncodingView<uint32_t>> lengths_;
};

} // namespace facebook::nimble
