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
#include "velox/dwio/nimble/index/HierarchicalKeyReader.h"

#include <algorithm>
#include <array>
#include <bit>
#include <functional>
#include <limits>
#include <optional>
#include <utility>

#include "folly/lang/Bits.h"
#include "velox/common/Casts.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/index/HierarchicalKeyFormat.h"

namespace facebook::nimble::index {
namespace {

size_t remainingBytes(const char* position, const char* end) {
  return static_cast<size_t>(
      velox::checkedNotNull(end) - velox::checkedNotNull(position));
}

// Reads the number of distinct values at each of 'numLevels' levels.
std::vector<uint32_t> readLevelValueCounts(
    const char*& position,
    const char* end,
    uint32_t numLevels) {
  NIMBLE_CHECK_FILE_LE(
      numLevels,
      remainingBytes(position, end) / sizeof(uint32_t),
      "Truncated HierarchicalKeyReader metadata.");
  std::vector<uint32_t> levelCounts(numLevels);
  encoding::readUint32s(position, std::span<uint32_t>{levelCounts});
  return levelCounts;
}

// Returns the offset range containing `position`.
uint32_t containingRange(std::span<const uint32_t> offsets, uint32_t position) {
  const auto upper = std::upper_bound(offsets.begin(), offsets.end(), position);
  NIMBLE_CHECK(upper != offsets.begin());
  return static_cast<uint32_t>(upper - offsets.begin() - 1);
}

// Returns the next size-prefixed sequence.
std::string_view readSizedSequence(const char*& position, const char* end) {
  NIMBLE_CHECK_FILE_GE(
      remainingBytes(position, end),
      sizeof(uint32_t),
      "Truncated HierarchicalKeyReader metadata.");
  NIMBLE_CHECK_FILE_GE(
      remainingBytes(position, end) - sizeof(uint32_t),
      encoding::peek<uint32_t>(position),
      "Truncated HierarchicalKeyReader sequence.");
  return encoding::readString(position);
}

} // namespace

std::vector<uint64_t> HierarchicalKeyReader::parseEncodedKey(
    std::string_view key) const {
  NIMBLE_CHECK_EQ(
      key.size(),
      encodedKeyBytes_,
      "HierarchicalKeyReader requires a complete encoded key.");
  std::vector<uint64_t> components(keyValueBytes_.size());
  size_t offset{0};
  for (uint32_t level = 0; level < keyValueBytes_.size(); ++level) {
    NIMBLE_CHECK_EQ(
        key[offset],
        0,
        "HierarchicalKeyReader requires non-null encoded key components.");
    const auto width = keyValueBytes_[level];
    std::array<char, sizeof(uint64_t)> encodedBytes{};
    std::copy_n(key.begin() + offset + 1, width, encodedBytes.end() - width);
    components.at(level) =
        folly::Endian::big(std::bit_cast<uint64_t>(encodedBytes));
    offset += 1 + width;
  }
  return components;
}

std::string HierarchicalKeyReader::encodeKey(
    std::span<const uint64_t> components) const {
  NIMBLE_CHECK_EQ(components.size(), keyValueBytes_.size());
  std::string key(encodedKeyBytes_, '\0');
  size_t offset{0};
  for (uint32_t level = 0; level < components.size(); ++level) {
    const auto width = keyValueBytes_[level];
    const auto encodeValue = [&]<typename T>() {
      const auto encodedValue =
          folly::Endian::big(static_cast<T>(components[level]));
      const auto encodedBytes =
          std::bit_cast<std::array<char, sizeof(T)>>(encodedValue);
      std::copy(
          encodedBytes.begin(),
          encodedBytes.end(),
          key.begin() + static_cast<std::string::difference_type>(offset + 1));
    };
    if (width == sizeof(uint8_t)) {
      encodeValue.template operator()<uint8_t>();
    } else if (width == sizeof(uint16_t)) {
      encodeValue.template operator()<uint16_t>();
    } else if (width == sizeof(uint32_t)) {
      encodeValue.template operator()<uint32_t>();
    } else {
      NIMBLE_CHECK_EQ(width, sizeof(uint64_t));
      encodeValue.template operator()<uint64_t>();
    }
    offset += 1 + width;
  }
  return key;
}

namespace detail {

// Opens a view whose values have type T after checking the sequence's prefix,
// physical type and row count.
template <typename T>
std::unique_ptr<TypedEncodingView<T>> createKeyView(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    uint32_t expectedRows) {
  NIMBLE_CHECK_FILE_GE(
      data.size(),
      EncodingPrefix::kFixedPrefixSize,
      "Truncated HierarchicalKeyReader sequence.");
  NIMBLE_CHECK_FILE_EQ(
      EncodingPrefix::dataType(data),
      TypeTraits<T>::dataType,
      "HierarchicalKeyReader child physical type does not match its level.");
  NIMBLE_CHECK_FILE_EQ(
      EncodingPrefix::readRowCount(data, /*useVarint=*/false),
      expectedRows,
      "HierarchicalKeyReader child row count does not match its directory.");
  return nimble::detail::createTypedEncodingView<T>(data, &pool, {});
}

// Type-erases physical widths so one hierarchy can hold mixed-width sequences.
class KeySequence {
 public:
  explicit KeySequence(EncodingType encodingType)
      : encodingType_{encodingType} {}

  virtual ~KeySequence() = default;

  EncodingType encodingType() const {
    return encodingType_;
  }

  virtual uint32_t rowCount() const = 0;
  virtual uint64_t valueAt(uint32_t row) const = 0;
  // Returns the first row in [begin, end) whose value is at least 'value', or
  // 'end'. Stores the value of a returned row before 'end' in 'found', so a
  // caller that compares it needs no second read. The values in [begin, end)
  // must be non-decreasing.
  virtual uint32_t lowerBound(
      uint32_t begin,
      uint32_t end,
      uint64_t value,
      uint64_t& found) const = 0;

 private:
  const EncodingType encodingType_;
};

template <typename T>
class TypedKeySequence final : public KeySequence {
 public:
  explicit TypedKeySequence(std::unique_ptr<TypedEncodingView<T>> view)
      : KeySequence{velox::checkedNotNull(view.get())->encodingType()},
        view_{std::move(view)},
        rowCount_{view_->rowCount()} {}

  uint32_t rowCount() const override {
    return rowCount_;
  }

  uint64_t valueAt(uint32_t row) const override {
    return view_->readAt(row);
  }

  uint32_t lowerBound(
      uint32_t begin,
      uint32_t end,
      uint64_t value,
      uint64_t& found) const override {
    // A value wider than the level's type exceeds every stored value, so no
    // row is at least it.
    if (value > std::numeric_limits<T>::max()) {
      return end;
    }
    T typedFound;
    const auto row =
        view_->lowerBound(begin, end, static_cast<T>(value), typedFound);
    if (row < end) {
      found = typedFound;
    }
    return row;
  }

 private:
  const std::unique_ptr<TypedEncodingView<T>> view_;
  const uint32_t rowCount_;
};

std::unique_ptr<KeySequence> createKeySequence(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    uint32_t expectedRows,
    uint8_t keyValueBytes) {
  const auto expectedDataType = [&] {
    switch (keyValueBytes) {
      case sizeof(uint8_t):
        return DataType::Uint8;
      case sizeof(uint16_t):
        return DataType::Uint16;
      case sizeof(uint32_t):
        return DataType::Uint32;
      case sizeof(uint64_t):
        return DataType::Uint64;
      default:
        NIMBLE_UNREACHABLE(
            "Unsupported HierarchicalKeyReader key value width: {}",
            keyValueBytes);
    }
  }();
  if (expectedDataType == DataType::Uint8) {
    return std::make_unique<TypedKeySequence<uint8_t>>(
        createKeyView<uint8_t>(pool, data, expectedRows));
  }
  if (expectedDataType == DataType::Uint16) {
    return std::make_unique<TypedKeySequence<uint16_t>>(
        createKeyView<uint16_t>(pool, data, expectedRows));
  }
  if (expectedDataType == DataType::Uint32) {
    return std::make_unique<TypedKeySequence<uint32_t>>(
        createKeyView<uint32_t>(pool, data, expectedRows));
  }
  NIMBLE_CHECK_EQ(expectedDataType, DataType::Uint64);
  return std::make_unique<TypedKeySequence<uint64_t>>(
      createKeyView<uint64_t>(pool, data, expectedRows));
}

// Child offsets of one level: decoded values for a short sequence, the encoded
// sequence otherwise, or the identity when none are stored, as for the leaf row
// offsets of distinct keys.
class ChildOffsets {
 public:
  ChildOffsets() = default;

  explicit ChildOffsets(Vector<uint32_t> decoded)
      : decoded_{std::move(decoded)} {}

  explicit ChildOffsets(std::unique_ptr<TypedEncodingView<uint32_t>> encoded)
      : encoded_{std::move(encoded)} {}

  // Returns entry 'index': the first child of value 'index' at the next level
  // or, for leaf row offsets, its first source row.
  uint32_t at(uint32_t index) const {
    if (encoded_ != nullptr) {
      return encoded_->readAt(index);
    }
    if (!decoded_.has_value()) {
      return index;
    }
    NIMBLE_DCHECK_LT(index, decoded_->size());
    return (*decoded_)[index];
  }

  // Returns the offset range containing `position`.
  uint32_t containingRange(uint32_t position) const {
    if (encoded_ != nullptr) {
      uint32_t found;
      return encoded_->lowerBound(
                 0, encoded_->rowCount(), position + 1, found) -
          1;
    }
    if (!decoded_.has_value()) {
      return position;
    }
    return index::containingRange(
        {decoded_->data(), decoded_->size()}, position);
  }

 private:
  std::optional<Vector<uint32_t>> decoded_;
  std::unique_ptr<TypedEncodingView<uint32_t>> encoded_;
};

// Reads and validates the encoded child offsets of 'numValues' values at one
// level.
ChildOffsets readEncodedOffsets(
    const char*& position,
    const char* end,
    uint32_t numValues,
    uint32_t expectedEnd,
    std::string_view description,
    velox::memory::MemoryPool& pool) {
  // Keep short offset sets decoded. They sit at the upper levels, which every
  // seek reads, so reading them through a view would slow every seek for
  // little memory saved.
  // TODO: Read every offset set through its view once view reads cost no
  // more than a decoded lookup.
  constexpr size_t kMaxDecodedOffsets{64};
  auto encodedOffsets = createKeyView<uint32_t>(
      pool, readSizedSequence(position, end), numValues + 1);
  const auto numOffsets = encodedOffsets->rowCount();
  // Validates in fixed batches, so a compact sequence that claims a huge row
  // count fails without a matching allocation.
  std::array<uint32_t, 1'024> batch;
  uint32_t previous{0};
  for (uint32_t begin = 0; begin < numOffsets; begin += batch.size()) {
    const auto length =
        std::min(static_cast<uint32_t>(batch.size()), numOffsets - begin);
    encodedOffsets->read(begin, length, batch.data());
    for (uint32_t i = 0; i < length; ++i) {
      if (begin + i == 0) {
        NIMBLE_CHECK_FILE_EQ(
            batch[i],
            0,
            "HierarchicalKeyReader offsets must start at zero: {}.",
            description);
      } else {
        NIMBLE_CHECK_FILE_LT(
            previous,
            batch[i],
            "HierarchicalKeyReader offsets contain an empty range: {}.",
            description);
      }
      previous = batch[i];
    }
  }
  NIMBLE_CHECK_FILE_EQ(
      previous,
      expectedEnd,
      "HierarchicalKeyReader has an invalid final offset: {}.",
      description);
  if (numOffsets > kMaxDecodedOffsets) {
    return ChildOffsets{std::move(encodedOffsets)};
  }
  Vector<uint32_t> decodedOffsets{&pool, numOffsets};
  encodedOffsets->read(0, numOffsets, decodedOffsets.data());
  return ChildOffsets{std::move(decodedOffsets)};
}

} // namespace detail

void HierarchicalKeyReader::readKeyValueWidths(
    const char*& position,
    const char* end) {
  NIMBLE_CHECK(keyValueBytes_.empty());
  NIMBLE_CHECK_EQ(encodedKeyBytes_, 0);
  NIMBLE_CHECK_FILE_GE(
      remainingBytes(position, end),
      numLevels_,
      "Truncated HierarchicalKeyReader metadata.");

  keyValueBytes_.reserve(numLevels_);
  for (uint32_t level = 0; level < numLevels_; ++level) {
    const auto width = encoding::read<uint8_t>(position);
    NIMBLE_CHECK_FILE(
        HierarchicalKeyFormat::isSupportedKeyValueBytes(width),
        "Unsupported HierarchicalKeyReader key value width: {}.",
        width);
    keyValueBytes_.push_back(width);
    // Each key value follows a non-null marker byte.
    encodedKeyBytes_ += 1 + width;
  }
}

void HierarchicalKeyReader::readSequences(
    const char*& position,
    const char* end,
    std::span<const uint32_t> levelCounts,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK(
      childOffsets_.empty(), "Hierarchical child offsets are already read.");
  NIMBLE_CHECK(
      levelSequences_.empty(),
      "Hierarchical level sequences are already read.");
  NIMBLE_CHECK_EQ(levelCounts.size(), numLevels_);
  auto& memoryPool = *velox::checkedNotNull(pool);
  childOffsets_.resize(numLevels_);
  levelSequences_.reserve(numLevels_);
  for (uint32_t level = 0; level < numLevels_; ++level) {
    if (level > 0) {
      childOffsets_.at(level - 1) = detail::readEncodedOffsets(
          position,
          end,
          levelCounts[level - 1],
          levelCounts[level],
          "child offsets",
          memoryPool);
    }
    levelSequences_.push_back(
        detail::createKeySequence(
            memoryPool,
            readSizedSequence(position, end),
            levelCounts[level],
            keyValueBytes_.at(level)));
  }
  if (rowCount_ > levelCounts.back()) {
    childOffsets_.back() = detail::readEncodedOffsets(
        position,
        end,
        levelCounts.back(),
        rowCount_,
        "leaf row offsets",
        memoryPool);
  }
}

HierarchicalKeyReader::HierarchicalKeyReader(
    std::string_view encodedData,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK_NOT_NULL(pool);
  NIMBLE_CHECK_FILE_GE(
      encodedData.size(),
      HierarchicalKeyFormat::kHeaderSize,
      "Truncated HierarchicalKeyReader metadata.");
  const char* position = velox::checkedNotNull(encodedData.data());
  const char* const end = position + encodedData.size();
  const auto formatVersion = encoding::read<uint8_t>(position);
  NIMBLE_CHECK_FILE_EQ(
      formatVersion,
      HierarchicalKeyFormat::kVersion,
      "Unsupported HierarchicalKeyReader format version.");
  numLevels_ = encoding::read<uint8_t>(position);
  NIMBLE_CHECK_FILE_GT(numLevels_, 0, "HierarchicalKeyReader needs a level.");
  rowCount_ = encoding::readUint32(position);
  NIMBLE_CHECK_FILE_GT(rowCount_, 0, "HierarchicalKeyReader has no rows.");

  readKeyValueWidths(position, end);

  const auto levelCounts = readLevelValueCounts(position, end, numLevels_);
  NIMBLE_CHECK_FILE_GT(
      levelCounts.front(),
      0,
      "HierarchicalKeyReader root level has no values.");
  for (uint32_t level = 1; level < numLevels_; ++level) {
    NIMBLE_CHECK_FILE_GE(
        levelCounts[level],
        levelCounts[level - 1],
        "HierarchicalKeyReader level value counts must be non-decreasing.");
  }
  // Child offsets hold one more entry than their level's value count.
  NIMBLE_CHECK_FILE_LT(
      levelCounts.back(),
      std::numeric_limits<uint32_t>::max(),
      "HierarchicalKeyReader level value count is too large.");
  NIMBLE_CHECK_FILE_GE(
      rowCount_,
      levelCounts.back(),
      "HierarchicalKeyReader has fewer rows than distinct keys.");

  readSequences(position, end, levelCounts, pool);
  NIMBLE_CHECK_FILE_EQ(
      remainingBytes(position, end),
      0,
      "HierarchicalKeyReader contains trailing data.");
}

EncodingType HierarchicalKeyReader::testingValuesEncodingType(
    uint32_t level) const {
  NIMBLE_CHECK_LT(level, numLevels_, "Level is out of range.");
  return levelSequences_.at(level)->encodingType();
}

std::pair<uint32_t, uint32_t> HierarchicalKeyReader::seekBounds(
    std::span<const uint64_t> target) const {
  uint32_t parentIndex{0};
  for (uint32_t level = 0; level < numLevels_; ++level) {
    const auto& sequence = *levelSequences_.at(level);
    const auto runBegin =
        level == 0 ? 0 : childOffsets_.at(level - 1).at(parentIndex);
    const auto runEnd = level == 0
        ? sequence.rowCount()
        : childOffsets_.at(level - 1).at(parentIndex + 1);
    uint64_t found;
    const auto componentIndex =
        sequence.lowerBound(runBegin, runEnd, target[level], found);
    if (componentIndex == runEnd || found != target[level]) {
      const auto row = startRowOfSubtree(level, componentIndex);
      return {row, row};
    }
    parentIndex = componentIndex;
  }
  // Sorted input makes duplicate keys contiguous. The leaf row offsets store
  // one start row per distinct key plus a rowCount sentinel, so adjacent
  // offsets cover the complete duplicate run. They are the identity when all
  // keys are unique.
  return {
      childOffsets_.back().at(parentIndex),
      childOffsets_.back().at(parentIndex + 1)};
}

std::vector<uint64_t> HierarchicalKeyReader::keyAt(uint32_t row) const {
  NIMBLE_CHECK_LT(row, rowCount_);
  std::vector<uint64_t> components(numLevels_);
  auto componentIndex = childOffsets_.back().containingRange(row);
  for (uint32_t level = numLevels_; level-- > 1;) {
    components[level] = levelSequences_.at(level)->valueAt(componentIndex);
    componentIndex =
        childOffsets_.at(level - 1).containingRange(componentIndex);
  }
  components[0] = levelSequences_.at(0)->valueAt(componentIndex);
  return components;
}

uint32_t HierarchicalKeyReader::startRowOfSubtree(
    uint32_t level,
    uint32_t componentIndex) const {
  for (; level < numLevels_; ++level) {
    componentIndex = childOffsets_.at(level).at(componentIndex);
  }
  return componentIndex;
}

HierarchicalKeyReader::~HierarchicalKeyReader() = default;

std::optional<uint32_t> HierarchicalKeyReader::seek(
    std::string_view value,
    bool inclusive) const {
  const auto key = parseEncodedKey(value);
  const auto [lower, upper] = seekBounds(key);
  const auto row = inclusive ? lower : upper;
  return row == rowCount_ ? std::nullopt : std::optional<uint32_t>{row};
}

std::string HierarchicalKeyReader::get(uint32_t row) const {
  return encodeKey(keyAt(row));
}

namespace {

// Walks rows, rebuilding each key from its hierarchy components. Consecutive
// rows share no decoded state, so this only spares the caller a std::string
// per row.
class HierarchicalKeyReaderCursor final : public KeyCursor {
 public:
  HierarchicalKeyReaderCursor(
      const HierarchicalKeyReader& encoding,
      uint32_t row)
      : encoding_{encoding}, rowCount_{encoding.rowCount()}, row_{row} {}

  bool hasNext() const override {
    return row_ < rowCount_;
  }

  /// Returns a view that remains valid until the next call to next() or until
  /// this cursor is destroyed.
  std::string_view next() override {
    NIMBLE_CHECK(hasNext(), "Key cursor advanced past the last row");
    // Assigning into key_ reuses the buffer it already holds, so only the
    // first row allocates.
    key_ = encoding_.get(row_++);
    return key_;
  }

 private:
  const HierarchicalKeyReader& encoding_;
  const uint32_t rowCount_;
  std::string key_;
  uint32_t row_;
};

} // namespace

std::unique_ptr<KeyCursor> HierarchicalKeyReader::cursor(
    uint32_t startRow) const {
  NIMBLE_CHECK_LT(startRow, rowCount_);
  return std::make_unique<HierarchicalKeyReaderCursor>(*this, startRow);
}

std::vector<std::string> HierarchicalKeyReader::materialize(
    uint32_t startRow,
    uint32_t count) const {
  NIMBLE_CHECK_LE(startRow, rowCount_);
  NIMBLE_CHECK_LE(count, rowCount_ - startRow);
  std::vector<std::string> keys;
  keys.reserve(count);
  for (uint32_t row = startRow; row < startRow + count; ++row) {
    keys.push_back(get(row));
  }
  return keys;
}

uint32_t HierarchicalKeyReader::rowCount() const {
  return rowCount_;
}

uint32_t HierarchicalKeyReader::numLevels() const {
  return numLevels_;
}

std::unique_ptr<KeyReader> createHierarchicalKeyReader(
    std::string_view encodedKeys,
    const std::function<void*(uint32_t)>& /*stringBufferFactory*/,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK_NOT_NULL(pool);
  return std::make_unique<HierarchicalKeyReader>(encodedKeys, pool);
}

} // namespace facebook::nimble::index
