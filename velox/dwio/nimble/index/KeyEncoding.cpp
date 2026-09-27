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
#include "velox/dwio/nimble/index/KeyEncoding.h"

#include <algorithm>
#include <cstring>
#include <memory>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

namespace facebook::nimble::index {

// ---------------------------------------------------------------------------
// KeyEncoding
// ---------------------------------------------------------------------------

// static
std::unique_ptr<KeyEncoding> KeyEncoding::create(
    velox::memory::MemoryPool& pool,
    std::string_view encodedData,
    std::function<void*(uint32_t)> stringBufferFactory) {
  auto encoding = EncodingFactory().create(
      pool, encodedData, stringBufferFactory, Encoding::Options{});
  NIMBLE_CHECK_EQ(
      encoding->dataType(),
      DataType::String,
      "Key encoding requires String data type");
  switch (encoding->encodingType()) {
    case EncodingType::Trivial: {
      const uint32_t count = encoding->rowCount();
      std::vector<std::string_view> values(count);
      encoding->materialize(count, values.data());
      return std::make_unique<TrivialKeyEncoding>(std::move(values));
    }
    case EncodingType::Prefix:
      return std::make_unique<PrefixKeyEncoding>(
          encodedData, encoding->rowCount(), encoding->dataOffset());
    default:
      NIMBLE_FAIL(
          "Unsupported key encoding type: {}",
          static_cast<int>(encoding->encodingType()));
  }
}

// ---------------------------------------------------------------------------
// TrivialKeyEncoding
// ---------------------------------------------------------------------------

namespace {

// Walks pre-materialized key views.
class TrivialKeyCursor final : public KeyEncoding::Cursor {
 public:
  TrivialKeyCursor(
      const std::string_view* position,
      const std::string_view* end)
      : position_{position}, end_{end} {}

  bool hasNext() const override {
    return position_ != end_;
  }

  std::string_view next() override {
    NIMBLE_CHECK(hasNext(), "Key cursor advanced past the last row");
    return *position_++;
  }

 private:
  const std::string_view* position_;
  const std::string_view* const end_;
};

} // namespace

TrivialKeyEncoding::TrivialKeyEncoding(std::vector<std::string_view> values)
    : values_{std::move(values)} {}

std::optional<uint32_t> TrivialKeyEncoding::seek(
    std::string_view value,
    bool inclusive) const {
  const auto it = inclusive
      ? std::lower_bound(values_.begin(), values_.end(), value)
      : std::upper_bound(values_.begin(), values_.end(), value);
  if (it == values_.end()) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(std::distance(values_.begin(), it));
}

std::string TrivialKeyEncoding::get(uint32_t row) const {
  NIMBLE_CHECK_LT(row, values_.size());
  return std::string(values_[row]);
}

std::vector<std::string> TrivialKeyEncoding::materialize(
    uint32_t startRow,
    uint32_t count) const {
  NIMBLE_CHECK_LE(startRow, values_.size());
  NIMBLE_CHECK_LE(count, values_.size() - startRow);
  std::vector<std::string> result;
  result.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    result.emplace_back(values_[startRow + i]);
  }
  return result;
}

std::unique_ptr<KeyEncoding::Cursor> TrivialKeyEncoding::cursor(
    uint32_t startRow) const {
  NIMBLE_CHECK_LT(startRow, values_.size());
  return std::make_unique<TrivialKeyCursor>(
      values_.data() + startRow, values_.data() + values_.size());
}

// ---------------------------------------------------------------------------
// PrefixKeyEncoding
// ---------------------------------------------------------------------------

class PrefixKeyEncoding::KeyScratch {
 public:
  // Resizes to 'size' bytes, keeping the leading 'preserved' bytes that the
  // next entry shares with its predecessor.
  char* resize(uint32_t size, uint32_t preserved) {
    if (size > capacity_) {
      const uint32_t newCapacity = std::max<uint32_t>(size, capacity_ * 2);
      // make_unique<char[]>() would zero-initialize bytes that the decoder
      // immediately replaces.
      auto newBuffer = std::make_unique_for_overwrite<char[]>(newCapacity);
      if (preserved > 0) {
        std::memcpy(newBuffer.get(), buffer_.get(), preserved);
      }
      buffer_ = std::move(newBuffer);
      capacity_ = newCapacity;
    }
    size_ = size;
    return buffer_.get();
  }

  void clear() {
    size_ = 0;
  }

  uint32_t size() const {
    return size_;
  }

  std::string_view view() const {
    return std::string_view{buffer_.get(), size_};
  }

 private:
  std::unique_ptr<char[]> buffer_;
  uint32_t capacity_{0};
  uint32_t size_{0};
};

// Decodes prefix-compressed keys one row at a time, carrying the previous
// key forward as the shared prefix instead of restarting from a restart
// point on every row.
class PrefixKeyEncoding::PrefixKeyCursor final : public KeyEncoding::Cursor {
 public:
  // Decodes forward from the restart point enclosing 'startRow', which leaves
  // scratch_ holding the preceding key so the first next() has the right
  // shared prefix to build on.
  PrefixKeyCursor(
      const char* position,
      const char* end,
      uint32_t row,
      uint32_t startRow,
      uint32_t rowCount)
      : position_{position}, end_{end}, row_{row}, rowCount_{rowCount} {
    while (row_ < startRow) {
      decodeEntryAt(position_, end_, row_, scratch_);
    }
  }

  bool hasNext() const override {
    return row_ < rowCount_;
  }

  std::string_view next() override {
    NIMBLE_CHECK(hasNext(), "Key cursor advanced past the last row");
    return decodeEntryAt(position_, end_, row_, scratch_);
  }

 private:
  const char* position_;
  const char* const end_;

  // Holds the preceding key so the next key can reuse its shared prefix. Owned
  // per cursor rather than shared through scanScratch(): a cursor keeps
  // decoding through it between next() calls, so any seek() or get() on the
  // same thread would otherwise clear it mid-walk.
  KeyScratch scratch_;

  uint32_t row_;
  const uint32_t rowCount_;
};

PrefixKeyEncoding::KeyScratch& PrefixKeyEncoding::scanScratch() {
  // PrefixKeyEncoding is immutable and safe for concurrent use. One scratch
  // buffer per thread retains capacity without adding synchronization.
  thread_local KeyScratch scratch;
  scratch.clear();
  return scratch;
}

std::string_view PrefixKeyEncoding::decodeEntryAt(
    const char*& position,
    const char* end,
    uint32_t& row,
    KeyScratch& scratch) {
  const uint32_t sharedPrefixLen = encoding::readUint32(position);
  const uint32_t suffixLen = encoding::readUint32(position);
  NIMBLE_CHECK_FILE_LE(
      sharedPrefixLen,
      scratch.size(),
      "Prefix-compressed key shares more bytes than its predecessor holds");
  // Bounding the suffix against the remaining bytes keeps a corrupt length
  // from reading past the encoded data, and keeps the sum below from wrapping.
  NIMBLE_CHECK_FILE_LE(
      suffixLen,
      static_cast<uint64_t>(end - position),
      "Prefix-compressed key suffix runs past the encoded data");
  const uint32_t fullLen = sharedPrefixLen + suffixLen;
  char* data = scratch.resize(fullLen, sharedPrefixLen);
  if (suffixLen > 0) {
    std::memcpy(data + sharedPrefixLen, position, suffixLen);
    position += suffixLen;
  }
  ++row;
  return std::string_view{data, fullLen};
}

PrefixKeyEncoding::PrefixKeyEncoding(
    std::string_view encodedData,
    uint32_t rowCount,
    uint32_t dataOffset)
    : rowCount_{rowCount},
      restartInterval_{[&] {
        const auto* pos = encodedData.data() + dataOffset;
        return encoding::readUint32(pos);
      }()},
      numRestarts_{velox::bits::divRoundUp(rowCount_, restartInterval_)},
      restartOffsets_{encodedData.data() + dataOffset + sizeof(uint32_t)},
      dataStart_{restartOffsets_ + numRestarts_ * sizeof(uint32_t)},
      dataEnd_{encodedData.data() + encodedData.size()} {}

uint32_t PrefixKeyEncoding::restartOffset(uint32_t restartIndex) const {
  NIMBLE_CHECK_LT(restartIndex, numRestarts_, "Restart index out of bounds");
  const char* offsetPos = restartOffsets_ + (restartIndex * sizeof(uint32_t));
  return encoding::readUint32(offsetPos);
}

std::optional<uint32_t> PrefixKeyEncoding::seek(
    std::string_view targetValue,
    bool inclusive) const {
  auto& scratch = scanScratch();
  const char* pos{nullptr};
  uint32_t row{0};

  // Binary search among restart points to find the block containing the target.
  uint32_t left = 0;
  uint32_t right = numRestarts_;

  while (left < right) {
    const uint32_t mid = left + (right - left) / 2;

    pos = restartPosition(mid);
    row = mid * restartInterval_;
    scratch.clear();
    const auto restartValue = decodeEntryAt(pos, dataEnd_, row, scratch);

    if (restartValue.compare(targetValue) < 0) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  // 'left' is the first restart point whose value >= targetValue. Search from
  // the previous restart point since the target might be within that block but
  // after its restart value.
  if (left > 0) {
    --left;
  }

  pos = restartPosition(left);
  row = left * restartInterval_;
  scratch.clear();

  if (inclusive) {
    while (row < rowCount_) {
      if (decodeEntryAt(pos, dataEnd_, row, scratch) >= targetValue) {
        return row - 1;
      }
    }
  } else {
    while (row < rowCount_) {
      if (decodeEntryAt(pos, dataEnd_, row, scratch) > targetValue) {
        return row - 1;
      }
    }
  }

  return std::nullopt;
}

std::unique_ptr<KeyEncoding::Cursor> PrefixKeyEncoding::cursor(
    uint32_t startRow) const {
  NIMBLE_CHECK_LT(startRow, rowCount_);

  const uint32_t restartIndex = startRow / restartInterval_;
  return std::make_unique<PrefixKeyCursor>(
      restartPosition(restartIndex),
      dataEnd_,
      restartIndex * restartInterval_,
      startRow,
      rowCount_);
}

std::string PrefixKeyEncoding::get(uint32_t row) const {
  NIMBLE_CHECK_LT(row, rowCount_);

  const uint32_t restartIndex = row / restartInterval_;
  const char* pos = restartPosition(restartIndex);
  uint32_t currentRow = restartIndex * restartInterval_;
  auto& scratch = scanScratch();

  while (currentRow <= row) {
    decodeEntryAt(pos, dataEnd_, currentRow, scratch);
  }

  return std::string{scratch.view()};
}

std::vector<std::string> PrefixKeyEncoding::materialize(
    uint32_t startRow,
    uint32_t count) const {
  NIMBLE_CHECK_LE(startRow, rowCount_);
  NIMBLE_CHECK_LE(count, rowCount_ - startRow);

  // An empty range needs no cursor, and asking for one at startRow ==
  // rowCount_ would throw.
  if (count == 0) {
    return {};
  }

  const uint32_t restartIndex = startRow / restartInterval_;
  const char* pos = restartPosition(restartIndex);
  uint32_t currentRow = restartIndex * restartInterval_;
  auto& scratch = scanScratch();

  // Skip to startRow.
  while (currentRow < startRow) {
    decodeEntryAt(pos, dataEnd_, currentRow, scratch);
  }
  std::vector<std::string> result;
  result.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    decodeEntryAt(pos, dataEnd_, currentRow, scratch);
    result.emplace_back(scratch.view());
  }
  return result;
}

} // namespace facebook::nimble::index
