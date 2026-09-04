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

#include <array>
#include "velox/dwio/nimble/encodings/common/Encoding.h"

namespace facebook::nimble::detail {

/// Buffers values from an encoding, reading in pages of BufferSize.
template <typename T, uint16_t BufferSize>
class BufferedEncoding {
 public:
  explicit BufferedEncoding(std::unique_ptr<Encoding> encoding)
      : encoding_{std::move(encoding)} {}

  T nextValue() {
    if (UNLIKELY(bufferPosition_ == BufferSize)) {
      refill();
    }
    ++encodingPosition_;
    return buffer_[bufferPosition_++];
  }

  void reset() {
    bufferPosition_ = BufferSize;
    encodingPosition_ = 0;
    encoding_->reset();
  }

  /// Returns the number of values in the wrapped encoding.
  uint32_t rowCount() const {
    return encoding_->rowCount();
  }

 private:
  FOLLY_ALWAYS_INLINE void refill() {
    auto numRows = std::min<uint32_t>(
        BufferSize, encoding_->rowCount() - encodingPosition_);
    encoding_->materialize(numRows, buffer_.data());
    bufferPosition_ = 0;
  }

  const std::unique_ptr<Encoding> encoding_;
  // Position within the current page buffer (0..BufferSize-1).
  // Reset to BufferSize to trigger a refill on the next read.
  uint16_t bufferPosition_{BufferSize};
  // Number of values consumed from the encoding so far.
  uint32_t encodingPosition_{0};
  std::array<T, BufferSize> buffer_;
};

/// Buffers dictionary indices from an encoding, reading in pages of BufferSize.
/// Exposes nextIndex() for dictionary index access and forwards dictionary
/// metadata to the inner encoding.
template <typename T, uint16_t BufferSize>
class BufferedDictEncoding {
 public:
  explicit BufferedDictEncoding(std::unique_ptr<Encoding> encoding)
      : encoding_{std::move(encoding)} {}

  /// Advances position and returns the next dictionary index.
  uint32_t nextIndex() {
    if (UNLIKELY(bufferPosition_ == BufferSize)) {
      refill();
    }
    ++encodingPosition_;
    return indicesBuffer_[bufferPosition_++];
  }

  void reset() {
    bufferPosition_ = BufferSize;
    encodingPosition_ = 0;
    encoding_->reset();
  }

  uint32_t dictionarySize() const noexcept {
    return encoding_->dictionarySize();
  }

  const void* dictionaryEntries() const noexcept {
    return encoding_->dictionaryEntries();
  }

 private:
  FOLLY_ALWAYS_INLINE void refill() {
    auto numRows = std::min<uint32_t>(
        BufferSize, encoding_->rowCount() - encodingPosition_);
    encoding_->materializeIndices(numRows, indicesBuffer_.data());
    // Zero-fill unused positions so the RLE base class's one-run-ahead
    // read produces a valid index (0) instead of uninitialized garbage.
    // When the last page is partial, positions [numRows, BufferSize) are
    // uninitialized. The base class skip/materialize may read one position
    // ahead of the last consumed value, so these must be safe to read.
    if (numRows < BufferSize) {
      std::fill(
          indicesBuffer_.begin() + numRows, indicesBuffer_.end(), uint32_t{0});
    }
    bufferPosition_ = 0;
  }

  const std::unique_ptr<Encoding> encoding_;
  // Position within the current page buffer (0..BufferSize-1).
  // Reset to BufferSize to trigger a refill on the next read.
  uint16_t bufferPosition_{BufferSize};
  // Number of values consumed from the encoding so far.
  uint32_t encodingPosition_{0};
  std::array<uint32_t, BufferSize> indicesBuffer_;
};

} // namespace facebook::nimble::detail
