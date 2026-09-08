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
  NIMBLE_CHECK_LE(startRow + count, values_.size());
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

namespace {

// Decodes the entry at 'pos' into 'decoded', which must hold the preceding
// entry's key so the shared prefix can be reused. Advances 'pos' past the
// entry and increments 'row'.
std::string_view
decodeEntryAt(const char*& pos, uint32_t& row, std::string& decoded) {
  const uint32_t sharedPrefixLen = encoding::readUint32(pos);
  const uint32_t suffixLen = encoding::readUint32(pos);
  const uint32_t fullLen = sharedPrefixLen + suffixLen;
  NIMBLE_DCHECK_LE(sharedPrefixLen, decoded.size());
  decoded.resize(fullLen);
  if (suffixLen > 0) {
    std::memcpy(decoded.data() + sharedPrefixLen, pos, suffixLen);
    pos += suffixLen;
  }
  ++row;
  return std::string_view(decoded.data(), fullLen);
}

// Decodes prefix-compressed keys one row at a time, carrying the previous
// key forward as the shared prefix instead of restarting from a restart
// point on every row.
class PrefixKeyCursor final : public KeyEncoding::Cursor {
 public:
  PrefixKeyCursor(
      const char* position,
      std::string decoded,
      uint32_t row,
      uint32_t rowCount)
      : position_{position},
        decoded_{std::move(decoded)},
        row_{row},
        rowCount_{rowCount} {}

  bool hasNext() const override {
    return row_ < rowCount_;
  }

  std::string_view next() override {
    NIMBLE_CHECK(hasNext(), "Key cursor advanced past the last row");
    return decodeEntryAt(position_, row_, decoded_);
  }

 private:
  const char* position_;

  // Key of the row the cursor last returned; the next key is decoded on top
  // of it.
  std::string decoded_;

  uint32_t row_;
  const uint32_t rowCount_;
};

} // namespace

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
      dataStart_{restartOffsets_ + numRestarts_ * sizeof(uint32_t)} {}

uint32_t PrefixKeyEncoding::restartOffset(uint32_t restartIndex) const {
  NIMBLE_CHECK_LT(restartIndex, numRestarts_, "Restart index out of bounds");
  const char* offsetPos = restartOffsets_ + (restartIndex * sizeof(uint32_t));
  return encoding::readUint32(offsetPos);
}

std::optional<uint32_t> PrefixKeyEncoding::seek(
    std::string_view targetValue,
    bool inclusive) const {
  const char* pos = nullptr;
  uint32_t row = 0;
  std::string decoded;

  // Binary search among restart points to find the block containing the target.
  uint32_t left = 0;
  uint32_t right = numRestarts_;

  while (left < right) {
    const uint32_t mid = left + (right - left) / 2;

    pos = restartPosition(mid);
    row = mid * restartInterval_;
    decoded.clear();
    const auto restartValue = decodeEntryAt(pos, row, decoded);

    if (restartValue.compare(targetValue) < 0) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  // 'left' is the first restart point whose value >= targetValue.
  // Search from the previous restart point since the target might be
  // within that block but after its restart value.
  if (left > 0) {
    --left;
  }

  pos = restartPosition(left);
  row = left * restartInterval_;
  decoded.clear();

  if (inclusive) {
    while (row < rowCount_) {
      if (decodeEntryAt(pos, row, decoded) >= targetValue) {
        return row - 1;
      }
    }
  } else {
    while (row < rowCount_) {
      if (decodeEntryAt(pos, row, decoded) > targetValue) {
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
  const char* pos = restartPosition(restartIndex);
  uint32_t currentRow = restartIndex * restartInterval_;
  std::string decoded;

  // Decode up to startRow, leaving 'decoded' holding the preceding key so
  // the first next() has the right shared prefix to build on.
  while (currentRow < startRow) {
    decodeEntryAt(pos, currentRow, decoded);
  }

  return std::make_unique<PrefixKeyCursor>(
      pos, std::move(decoded), currentRow, rowCount_);
}

// Walks its own entries rather than going through cursor(): a one-off read
// should not pay for the cursor's heap allocation, which measured ~19% of
// this call.
std::string PrefixKeyEncoding::get(uint32_t row) const {
  NIMBLE_CHECK_LT(row, rowCount_);

  const uint32_t restartIndex = row / restartInterval_;
  const char* pos = restartPosition(restartIndex);
  uint32_t currentRow = restartIndex * restartInterval_;
  std::string decoded;

  while (currentRow <= row) {
    decodeEntryAt(pos, currentRow, decoded);
  }

  return decoded;
}

std::vector<std::string> PrefixKeyEncoding::materialize(
    uint32_t startRow,
    uint32_t count) const {
  NIMBLE_CHECK_LE(startRow + count, rowCount_);

  // An empty range needs no cursor, and asking for one at startRow ==
  // rowCount_ would throw.
  if (count == 0) {
    return {};
  }

  auto keyCursor = cursor(startRow);
  std::vector<std::string> result;
  result.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    result.emplace_back(keyCursor->next());
  }
  return result;
}

} // namespace facebook::nimble::index
