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
#include <atomic>
#include <cstdint>
#include <cstring>
#include <optional>
#include <type_traits>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

template <typename T>
class DeltaBlockEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  DeltaBlockEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        blockBases_{this->template getVectorBuffer<physicalType>()},
        bitWidths_{this->template getVectorBuffer<uint8_t>()},
        blockOffsets_{this->template getVectorBuffer<uint32_t>()} {
    static_assert(
        std::is_integral_v<T> && !std::is_same_v<T, bool>,
        "DeltaBlockEncodingView only supports non-bool integral types.");

    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::DeltaBlock);
    const char* pos = data.data() + this->dataOffset_;
    blockSize_ = static_cast<uint16_t>(varint::readVarint32(&pos));
    numBlocks_ = varint::readVarint32(&pos);
    NIMBLE_CHECK_GT(blockSize_, 0);
    NIMBLE_CHECK_EQ(
        numBlocks_, velox::bits::divRoundUp(this->rowCount_, blockSize_));

    blockBases_.resize(numBlocks_);
    for (uint32_t i = 0; i < numBlocks_; ++i) {
      blockBases_[i] =
          DeltaBlockEncoding<T>::fromOrdered(varint::readVarint64(&pos));
    }

    bitWidths_.resize(numBlocks_);
    std::memcpy(bitWidths_.data(), pos, numBlocks_ * sizeof(uint8_t));
    pos += numBlocks_ * sizeof(uint8_t);

    blockOffsets_.resize(numBlocks_);
    for (uint32_t i = 0; i < numBlocks_; ++i) {
      blockOffsets_[i] = varint::readVarint32(&pos);
    }
    packedData_ = pos;
  }

  ~DeltaBlockEncodingView() override {
    this->releaseVectorBuffer(blockOffsets_);
    this->releaseVectorBuffer(bitWidths_);
    this->releaseVectorBuffer(blockBases_);
  }

 private:
  // Monotonic per-instantiation id. A freed view's address can be reused by a
  // new view, so the pointer is not a safe key for the shared thread-local
  // cache; a never-reused id disambiguates instances across their lifetimes.
  static uint64_t nextViewId() {
    static std::atomic_uint64_t counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
  }

  struct ReadCache {
    uint64_t viewId{0};
    const char* packedData{nullptr};
    uint32_t rowCount{0};
    uint16_t blockSize{0};
    std::optional<uint32_t> blockIndex;
    uint32_t position{0};
    uint8_t bitWidth{0};
    uint64_t ordered{0};
    FixedBitArray deltas{};
  };

  uint32_t blockRowCount(uint32_t blockIndex) const {
    return DeltaBlockEncoding<T>::blockRowCount(
        this->rowCount_, blockSize_, blockIndex);
  }

  void resetCacheIfNeeded(ReadCache& cache) const {
    if (cache.viewId == viewId_ && cache.packedData == packedData_ &&
        cache.rowCount == this->rowCount_ && cache.blockSize == blockSize_) {
      return;
    }

    cache = ReadCache{};
    cache.viewId = viewId_;
    cache.packedData = packedData_;
    cache.rowCount = this->rowCount_;
    cache.blockSize = blockSize_;
  }

  void seekCache(ReadCache& cache, uint32_t blockIndex, uint32_t position)
      const {
    cache.blockIndex = blockIndex;
    cache.position = 0;
    cache.bitWidth = bitWidths_[blockIndex];
    cache.ordered = DeltaBlockEncoding<T>::toOrdered(blockBases_[blockIndex]);
    if (cache.bitWidth != 0) {
      cache.deltas = FixedBitArray{
          {packedData_ + blockOffsets_[blockIndex],
           FixedBitArray::bufferSize(
               blockRowCount(blockIndex) - 1, cache.bitWidth)},
          cache.bitWidth};
      for (uint32_t i = 0; i < position; ++i) {
        cache.ordered += cache.deltas.get(i);
      }
    }
    cache.position = position;
  }

  physicalType readWithCache(ReadCache& cache, uint32_t index) const {
    const auto blockIndex = index / blockSize_;
    const auto position = index % blockSize_;
    if (!cache.blockIndex.has_value() ||
        cache.blockIndex.value() != blockIndex || position < cache.position) {
      seekCache(cache, blockIndex, position);
    } else if (position > cache.position) {
      if (cache.bitWidth != 0) {
        for (uint32_t i = cache.position; i < position; ++i) {
          cache.ordered += cache.deltas.get(i);
        }
      }
      cache.position = position;
    }
    return DeltaBlockEncoding<T>::fromOrdered(cache.ordered);
  }

  ReadCache* readCache() const {
    static thread_local ReadCache cache;
    resetCacheIfNeeded(cache);
    return &cache;
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    ReadCache* cache = readCache();
    return readWithCache(*cache, index);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    ReadCache* cache = readCache();
    for (uint32_t i = 0; i < length; ++i) {
      output[i] = readWithCache(*cache, offset + i);
    }
  }

  const uint64_t viewId_{nextViewId()};
  uint16_t blockSize_{0};
  uint32_t numBlocks_{0};
  Vector<physicalType> blockBases_;
  Vector<uint8_t> bitWidths_;
  Vector<uint32_t> blockOffsets_;
  const char* packedData_{nullptr};
};

} // namespace facebook::nimble
