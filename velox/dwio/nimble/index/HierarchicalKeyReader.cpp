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
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/views/EliasFanoEncodingView.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/index/HierarchicalKeyFormat.h"

namespace facebook::nimble::index {
namespace {

size_t remainingBytes(const char* position, const char* end) {
  return static_cast<size_t>(
      velox::checkedNotNull(end) - velox::checkedNotNull(position));
}

// Reads bounded uint32 metadata using the shared encoding primitive.
void readUint32Values(
    const char*& position,
    const char* end,
    size_t count,
    std::vector<uint32_t>& values) {
  NIMBLE_CHECK(values.empty(), "uint32 output is already initialized.");
  NIMBLE_CHECK_LE(
      count,
      remainingBytes(position, end) / sizeof(uint32_t),
      "Truncated HierarchicalKeyReader metadata.");
  values.resize(count);
  encoding::readUint32s(position, std::span<uint32_t>{values});
}

// Validates that offsets cover the expected range monotonically.
void validateOffsets(
    std::span<const uint32_t> offsets,
    uint32_t expectedEnd,
    bool allowEmptyRanges,
    std::string_view description) {
  NIMBLE_CHECK(!offsets.empty());
  NIMBLE_CHECK_EQ(
      offsets.front(),
      0,
      "HierarchicalKeyReader offsets must start at zero: {}.",
      description);
  NIMBLE_CHECK_EQ(
      offsets.back(),
      expectedEnd,
      "HierarchicalKeyReader has an invalid final offset: {}.",
      description);
  for (size_t i = 1; i < offsets.size(); ++i) {
    if (allowEmptyRanges) {
      NIMBLE_CHECK_LE(
          offsets[i - 1],
          offsets[i],
          "HierarchicalKeyReader offsets are not ascending: {}.",
          description);
    } else {
      NIMBLE_CHECK_LT(
          offsets[i - 1],
          offsets[i],
          "HierarchicalKeyReader offsets contain an empty range: {}.",
          description);
    }
  }
}

// Returns the offset range containing `position`.
uint32_t containingRange(std::span<const uint32_t> offsets, uint32_t position) {
  const auto upper = std::upper_bound(offsets.begin(), offsets.end(), position);
  NIMBLE_CHECK(upper != offsets.begin());
  return static_cast<uint32_t>(upper - offsets.begin() - 1);
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
  virtual uint32_t lowerBound(uint64_t value) const = 0;

 private:
  const EncodingType encodingType_;
};

template <typename T>
class TypedKeySequence final : public KeySequence {
 public:
  TypedKeySequence(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      uint32_t expectedRows)
      : KeySequence{EncodingPrefix::encodingType(data)},
        view_{createEncodingView(data, &pool)},
        typedView_{dynamic_cast<const TypedEncodingView<T>*>(
            velox::checkedNotNull(view_.get()))},
        eliasFanoView_{
            encodingType() == EncodingType::EliasFano
                ? dynamic_cast<const EliasFanoEncodingView<T>*>(
                      velox::checkedNotNull(view_.get()))
                : nullptr},
        rowCount_{velox::checkedNotNull(view_.get())->rowCount()} {
    NIMBLE_CHECK(
        encodingType() == EncodingType::Constant ||
            encodingType() == EncodingType::EliasFano ||
            encodingType() == EncodingType::FixedBitWidth ||
            encodingType() == EncodingType::Trivial,
        "Unsupported HierarchicalKeyReader child encoding: {}",
        toString(encodingType()));
    NIMBLE_CHECK_NOT_NULL(typedView_);
    if (encodingType() == EncodingType::EliasFano) {
      NIMBLE_CHECK_NOT_NULL(eliasFanoView_);
    }
    NIMBLE_CHECK_EQ(
        rowCount_,
        expectedRows,
        "HierarchicalKeyReader child row count does not match its directory.");
  }

  uint32_t rowCount() const override {
    return rowCount_;
  }

  uint64_t valueAt(uint32_t row) const override {
    return velox::checkedNotNull(typedView_)->readAt(row);
  }

  uint32_t lowerBound(uint64_t value) const override {
    if (value > std::numeric_limits<T>::max()) {
      return rowCount_;
    }
    const auto typedValue = static_cast<T>(value);
    if (eliasFanoView_ != nullptr) {
      return eliasFanoView_->lowerBound(typedValue);
    }
    uint32_t begin{0};
    uint32_t end{rowCount_};
    while (begin < end) {
      const auto middle = begin + (end - begin) / 2;
      if (velox::checkedNotNull(typedView_)->readAt(middle) < typedValue) {
        begin = middle + 1;
      } else {
        end = middle;
      }
    }
    return begin;
  }

 private:
  const std::unique_ptr<EncodingView> view_;
  const TypedEncodingView<T>* const typedView_;
  const EliasFanoEncodingView<T>* const eliasFanoView_;
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
  NIMBLE_CHECK_EQ(
      EncodingPrefix::dataType(data),
      expectedDataType,
      "HierarchicalKeyReader child physical type does not match its level.");
  if (expectedDataType == DataType::Uint8) {
    return std::make_unique<TypedKeySequence<uint8_t>>(
        pool, data, expectedRows);
  }
  if (expectedDataType == DataType::Uint16) {
    return std::make_unique<TypedKeySequence<uint16_t>>(
        pool, data, expectedRows);
  }
  if (expectedDataType == DataType::Uint32) {
    return std::make_unique<TypedKeySequence<uint32_t>>(
        pool, data, expectedRows);
  }
  NIMBLE_CHECK_EQ(expectedDataType, DataType::Uint64);
  return std::make_unique<TypedKeySequence<uint64_t>>(pool, data, expectedRows);
}

} // namespace detail

void HierarchicalKeyReader::readKeyValueWidths(
    const char*& position,
    const char* end) {
  NIMBLE_CHECK(keyValueBytes_.empty());
  NIMBLE_CHECK_EQ(encodedKeyBytes_, 0);
  NIMBLE_CHECK_GE(
      remainingBytes(position, end),
      numLevels_,
      "Truncated HierarchicalKeyReader metadata.");

  keyValueBytes_.reserve(numLevels_);
  for (uint32_t level = 0; level < numLevels_; ++level) {
    const auto width = encoding::read<uint8_t>(position);
    NIMBLE_CHECK(
        HierarchicalKeyFormat::isSupportedKeyValueBytes(width),
        "Unsupported HierarchicalKeyReader key value width: {}.",
        width);
    keyValueBytes_.push_back(width);
    encodedKeyBytes_ += 1 + width;
  }
}

void HierarchicalKeyReader::readChildOffsets(
    const char*& position,
    const char* end,
    std::span<const uint32_t> levelCounts) {
  NIMBLE_CHECK(
      childOffsets_.empty(), "Hierarchical child offsets are already read.");
  NIMBLE_CHECK_EQ(levelCounts.size(), numLevels_);
  childOffsets_.resize(numLevels_);
  for (uint32_t level = 0; level + 1 < numLevels_; ++level) {
    readUint32Values(
        position,
        end,
        static_cast<size_t>(levelCounts[level]) + 1,
        childOffsets_.at(level));
    validateOffsets(
        childOffsets_.at(level),
        levelCounts[level + 1],
        /*allowEmptyRanges=*/false,
        "child offsets");
  }

  const bool hasDuplicateKeys = rowCount_ > levelCounts.back();
  if (hasDuplicateKeys) {
    readUint32Values(
        position,
        end,
        static_cast<size_t>(levelCounts.back()) + 1,
        childOffsets_.back());
    validateOffsets(
        childOffsets_.back(),
        rowCount_,
        /*allowEmptyRanges=*/false,
        "leaf row offsets");
    return;
  }

  NIMBLE_CHECK_EQ(
      levelCounts.back(),
      rowCount_,
      "HierarchicalKeyReader without leaf row offsets must have one leaf "
      "value per row.");
  childOffsets_.back().resize(static_cast<size_t>(rowCount_) + 1);
  for (size_t row = 0; row < childOffsets_.back().size(); ++row) {
    childOffsets_.back().at(row) = static_cast<uint32_t>(row);
  }
}

void HierarchicalKeyReader::readLevelSequences(
    const char*& position,
    const char* end,
    std::span<const uint32_t> levelCounts,
    size_t numSequences,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK(
      levelSequences_.empty(),
      "Hierarchical level sequences are already read.");
  NIMBLE_CHECK_EQ(levelCounts.size(), numLevels_);
  auto& memoryPool = *velox::checkedNotNull(pool);
  std::vector<uint32_t> sequenceSizes;
  readUint32Values(position, end, numSequences, sequenceSizes);

  levelSequences_.resize(numLevels_);
  size_t sequenceIndex{0};
  const auto createSequence = [&](uint32_t level, uint32_t expectedRows) {
    const auto sequenceSize = sequenceSizes[sequenceIndex++];
    NIMBLE_CHECK_GE(
        remainingBytes(position, end),
        sequenceSize,
        "Truncated HierarchicalKeyReader sequence.");
    const auto* sequenceStart = velox::checkedNotNull(position);
    const std::string_view data{sequenceStart, sequenceSize};
    position = sequenceStart + sequenceSize;
    NIMBLE_CHECK_GE(
        data.size(),
        EncodingPrefix::kFixedPrefixSize,
        "Truncated HierarchicalKeyReader sequence.");
    return detail::createKeySequence(
        memoryPool, data, expectedRows, keyValueBytes_.at(level));
  };

  levelSequences_.at(0).push_back(createSequence(0, levelCounts[0]));
  for (uint32_t level = 1; level < numLevels_; ++level) {
    levelSequences_.at(level).reserve(levelCounts[level - 1]);
    for (uint32_t parent = 0; parent < levelCounts[level - 1]; ++parent) {
      levelSequences_.at(level).push_back(createSequence(
          level,
          childOffsets_.at(level - 1).at(parent + 1) -
              childOffsets_.at(level - 1).at(parent)));
    }
  }
}

HierarchicalKeyReader::HierarchicalKeyReader(
    std::string_view encodedData,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK_NOT_NULL(pool);
  NIMBLE_CHECK_GE(
      encodedData.size(),
      HierarchicalKeyFormat::kHeaderSize,
      "Truncated HierarchicalKeyReader metadata.");
  const char* position = velox::checkedNotNull(encodedData.data());
  const char* const end = position + encodedData.size();
  const auto formatVersion = encoding::read<uint8_t>(position);
  NIMBLE_CHECK_EQ(
      formatVersion,
      HierarchicalKeyFormat::kVersion,
      "Unsupported HierarchicalKeyReader format version.");
  numLevels_ = encoding::read<uint8_t>(position);
  NIMBLE_CHECK_GT(numLevels_, 0, "HierarchicalKeyReader needs a level.");
  rowCount_ = encoding::readUint32(position);
  NIMBLE_CHECK_GT(rowCount_, 0, "HierarchicalKeyReader has no rows.");

  readKeyValueWidths(position, end);

  std::vector<uint32_t> levelCounts;
  readUint32Values(position, end, numLevels_, levelCounts);
  NIMBLE_CHECK_GT(
      levelCounts.front(),
      0,
      "HierarchicalKeyReader root level has no values.");
  size_t numSequences{1};
  for (uint32_t level = 1; level < numLevels_; ++level) {
    NIMBLE_CHECK_GE(
        levelCounts[level],
        levelCounts[level - 1],
        "HierarchicalKeyReader level value counts must be non-decreasing.");
    numSequences += levelCounts[level - 1];
  }
  NIMBLE_CHECK_GE(rowCount_, levelCounts.back());

  readChildOffsets(position, end, levelCounts);
  readLevelSequences(position, end, levelCounts, numSequences, pool);
  NIMBLE_CHECK_EQ(
      remainingBytes(position, end),
      0,
      "HierarchicalKeyReader contains trailing data.");
}

std::vector<EncodingType> HierarchicalKeyReader::testingSequenceEncodingTypes(
    uint32_t level) const {
  NIMBLE_CHECK_LT(level, numLevels_, "Level is out of range.");
  std::vector<EncodingType> encodingTypes;
  encodingTypes.reserve(levelSequences_.at(level).size());
  for (const auto& sequence : levelSequences_.at(level)) {
    encodingTypes.push_back(
        velox::checkedNotNull(sequence.get())->encodingType());
  }
  return encodingTypes;
}

std::pair<uint32_t, uint32_t> HierarchicalKeyReader::seekBounds(
    std::span<const uint64_t> target) const {
  uint32_t parentIndex{0};
  for (uint32_t level = 0; level < numLevels_; ++level) {
    const auto& sequence = *levelSequences_.at(level).at(parentIndex);
    const auto sequenceStart =
        level == 0 ? 0 : childOffsets_.at(level - 1).at(parentIndex);
    const auto sequenceIndex = sequence.lowerBound(target[level]);
    if (sequenceIndex == sequence.rowCount()) {
      const auto sibling = level == 0
          ? sequence.rowCount()
          : childOffsets_.at(level - 1).at(parentIndex + 1);
      const auto row = startRowOfSubtree(level, sibling);
      return {row, row};
    }
    const auto componentIndex = sequenceStart + sequenceIndex;
    if (sequence.valueAt(sequenceIndex) != target[level]) {
      const auto row = startRowOfSubtree(level, componentIndex);
      return {row, row};
    }
    parentIndex = componentIndex;
  }
  // Sorted input makes duplicate keys contiguous. The leaf directory stores
  // one start row per distinct key plus a rowCount sentinel, so adjacent
  // offsets cover the complete duplicate run. It is an identity directory
  // when all keys are unique.
  return {
      childOffsets_.back().at(parentIndex),
      childOffsets_.back().at(parentIndex + 1)};
}

std::vector<uint64_t> HierarchicalKeyReader::keyAt(uint32_t row) const {
  NIMBLE_CHECK_LT(row, rowCount_);
  std::vector<uint64_t> components(numLevels_);
  auto componentIndex = containingRange(childOffsets_.back(), row);
  for (uint32_t level = numLevels_; level-- > 1;) {
    const auto& parentOffsets = childOffsets_.at(level - 1);
    const auto parent = containingRange(parentOffsets, componentIndex);
    components[level] = levelSequences_.at(level).at(parent)->valueAt(
        componentIndex - parentOffsets.at(parent));
    componentIndex = parent;
  }
  components[0] = levelSequences_.at(0).at(0)->valueAt(componentIndex);
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
