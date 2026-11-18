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
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>

#include "velox/common/Casts.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/index/HierarchicalKeyFormat.h"

namespace facebook::nimble::index {
namespace {

// Returns room for 'size' values of 'values', which only grows, so the reads
// that reuse it do not allocate.
template <typename T>
std::span<T> growScratch(std::vector<T>& values, size_t size) {
  if (values.size() < size) {
    values.resize(size);
  }
  return {values.data(), size};
}

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

namespace detail {

// Opens a view whose values have type T after checking the sequence's prefix,
// physical type and row count. With 'cacheEncodingView' the view keeps the
// lookup structures of Encoding::Options::cacheEncodingView.
template <typename T>
std::unique_ptr<TypedEncodingView<T>> createKeyView(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    uint32_t expectedRows,
    bool cacheEncodingView) {
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
  return nimble::detail::createTypedEncodingView<T>(
      data, &pool, {.cacheEncodingView = cacheEncodingView});
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

  // Stores in 'values' the values of as many rows from 'begin', which a scan
  // reads together rather than one by one.
  virtual void read(uint32_t begin, std::span<uint64_t> values) const = 0;

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

// Returns room for 'numValues' values of a level narrower than 64 bits, which a
// read widens. Each thread has its own, so the const reads of a reader that
// threads share stay thread-safe, and it only grows, so the reads do not
// allocate.
template <typename T>
std::span<T> narrowValuesScratch(size_t numValues) {
  thread_local std::vector<T> narrowValues;
  return growScratch(narrowValues, numValues);
}

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

  void read(uint32_t begin, std::span<uint64_t> values) const override {
    const auto count = static_cast<uint32_t>(values.size());
    if constexpr (std::is_same_v<T, uint64_t>) {
      view_->read(begin, count, values.data());
    } else {
      // Reads the values in the width of the level, then widens them with one
      // copy.
      const auto narrowValues = narrowValuesScratch<T>(count);
      view_->read(begin, count, narrowValues.data());
      std::copy_n(narrowValues.begin(), count, values.begin());
    }
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
    uint8_t keyValueBytes,
    bool cacheEncodingView) {
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
        createKeyView<uint8_t>(pool, data, expectedRows, cacheEncodingView));
  }
  if (expectedDataType == DataType::Uint16) {
    return std::make_unique<TypedKeySequence<uint16_t>>(
        createKeyView<uint16_t>(pool, data, expectedRows, cacheEncodingView));
  }
  if (expectedDataType == DataType::Uint32) {
    return std::make_unique<TypedKeySequence<uint32_t>>(
        createKeyView<uint32_t>(pool, data, expectedRows, cacheEncodingView));
  }
  NIMBLE_CHECK_EQ(expectedDataType, DataType::Uint64);
  return std::make_unique<TypedKeySequence<uint64_t>>(
      createKeyView<uint64_t>(pool, data, expectedRows, cacheEncodingView));
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

  // Stores in 'entries' the entries from 'begin', which a scan reads together
  // rather than one by one with at().
  void read(uint32_t begin, std::span<uint32_t> entries) const {
    if (encoded_ != nullptr) {
      encoded_->read(
          begin, static_cast<uint32_t>(entries.size()), entries.data());
      return;
    }
    if (!decoded_.has_value()) {
      std::iota(entries.begin(), entries.end(), begin);
      return;
    }
    NIMBLE_DCHECK_LE(begin + entries.size(), decoded_->size());
    std::copy_n(decoded_->data() + begin, entries.size(), entries.begin());
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
    velox::memory::MemoryPool& pool,
    bool cacheEncodingView) {
  // Keep short offset sets decoded. They sit at the upper levels, which every
  // seek reads, so reading them through a view would slow every seek for
  // little memory saved.
  // TODO: Read every offset set through its view once view reads cost no
  // more than a decoded lookup.
  constexpr size_t kMaxDecodedOffsets{64};
  auto encodedOffsets = createKeyView<uint32_t>(
      pool, readSizedSequence(position, end), numValues + 1, cacheEncodingView);
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

void HierarchicalKeyReader::readLevelWidths(
    const char*& position,
    const char* end) {
  NIMBLE_CHECK(levelWidths_.empty());
  NIMBLE_CHECK_FILE_GE(
      remainingBytes(position, end),
      numLevels_,
      "Truncated HierarchicalKeyReader metadata.");

  levelWidths_.reserve(numLevels_);
  for (uint32_t level = 0; level < numLevels_; ++level) {
    const auto width = encoding::read<uint8_t>(position);
    NIMBLE_CHECK_FILE(
        HierarchicalKeyFormat::isSupportedKeyValueBytes(width),
        "Unsupported HierarchicalKeyReader key value width: {}.",
        width);
    levelWidths_.push_back(width);
  }
}

void HierarchicalKeyReader::readSequences(
    const char*& position,
    const char* end,
    std::span<const uint32_t> levelCounts,
    velox::memory::MemoryPool* pool,
    bool cacheEncodingViews) {
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
          memoryPool,
          cacheEncodingViews);
    }
    levelSequences_.push_back(
        detail::createKeySequence(
            memoryPool,
            readSizedSequence(position, end),
            levelCounts[level],
            levelWidths_.at(level),
            cacheEncodingViews));
  }
  if (rowCount_ > levelCounts.back()) {
    childOffsets_.back() = detail::readEncodedOffsets(
        position,
        end,
        levelCounts.back(),
        rowCount_,
        "leaf row offsets",
        memoryPool,
        cacheEncodingViews);
  }
}

HierarchicalKeyReader::HierarchicalKeyReader(
    std::string_view encodedData,
    bool cacheEncodingViews,
    std::shared_ptr<const HierarchicalKeyCodec> keyCodec,
    velox::memory::MemoryPool* pool)
    : keyCodec_{velox::checkedNotNull(std::move(keyCodec))} {
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

  readLevelWidths(position, end);

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

  readSequences(position, end, levelCounts, pool, cacheEncodingViews);
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

void HierarchicalKeyReader::levelsAt(uint32_t row, std::span<uint64_t> levels)
    const {
  NIMBLE_CHECK_LT(row, rowCount_);
  NIMBLE_CHECK_EQ(levels.size(), numLevels_);
  auto componentIndex = childOffsets_.back().containingRange(row);
  for (uint32_t level = numLevels_; level-- > 1;) {
    levels[level] = levelSequences_.at(level)->valueAt(componentIndex);
    componentIndex =
        childOffsets_.at(level - 1).containingRange(componentIndex);
  }
  levels[0] = levelSequences_.at(0)->valueAt(componentIndex);
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

namespace {

// Returns room for the values of the levels of 'numRows' rows of 'numLevels'
// levels. Each thread has its own, so the const reads of a reader that threads
// share stay thread-safe, and it only grows, so the reads do not allocate.
std::span<uint64_t> levelScratch(uint32_t numRows, uint32_t numLevels) {
  thread_local std::vector<uint64_t> levels;
  return growScratch(levels, static_cast<size_t>(numRows) * numLevels);
}

// The values of one level that the rows of a batch span, from the value that
// holds the first row to the value that holds the last row, which a batch reads
// together.
struct LevelSpan {
  // Returns the index of the current value among the values of the level.
  uint32_t currentLevelIndex() const {
    return startIndex + currentIndex;
  }

  // Returns the end offset of the child range of the current value.
  uint32_t currentChildEndOffset() const {
    return childOffsets[currentIndex];
  }

  // Returns the value that holds the current row.
  uint64_t currentValue() const {
    return levelValues[currentIndex];
  }

  // Index of the first value of the span among the values of the level.
  uint32_t startIndex{0};
  // Index of the last value of the span among the values of the level.
  uint32_t lastIndex{0};
  // Index in 'levelValues' of the value that holds the current row.
  uint32_t currentIndex{0};
  // Values of the level in the span, from 'startIndex' to 'lastIndex'.
  std::span<uint64_t> levelValues;
  // End offset of the child range of each value of the span: the index of the
  // first value of the next level past its children or, for the leaf, the
  // first row past its rows.
  std::span<uint32_t> childOffsets;
};

// Returns room for the spans of 'numLevels' levels, per thread like
// levelScratch().
std::span<LevelSpan> levelSpanScratch(uint32_t numLevels) {
  thread_local std::vector<LevelSpan> spans;
  return growScratch(spans, numLevels);
}

// Returns room for the values of level spans that hold 'numValues' values, per
// thread like levelScratch().
std::span<uint64_t> levelValuesSpanScratch(size_t numValues) {
  thread_local std::vector<uint64_t> levelValues;
  return growScratch(levelValues, numValues);
}

// Returns room for the child end offsets of level spans that hold 'numValues'
// values, per thread like levelScratch().
std::span<uint32_t> childOffsetsSpanScratch(size_t numValues) {
  thread_local std::vector<uint32_t> childOffsets;
  return growScratch(childOffsets, numValues);
}

} // namespace

std::optional<uint32_t> HierarchicalKeyReader::seek(
    std::string_view value,
    bool inclusive) const {
  const auto levels = levelScratch(/*numRows=*/1, numLevels_);
  const bool after = keyCodec_->parseKey(value, levelWidths_, levels);
  const auto [lower, upper] = seekBounds(levels);
  const auto row = inclusive && !after ? lower : upper;
  return row == rowCount_ ? std::nullopt : std::optional<uint32_t>{row};
}

std::string HierarchicalKeyReader::get(uint32_t row) const {
  const auto levels = levelScratch(/*numRows=*/1, numLevels_);
  levelsAt(row, levels);
  return keyCodec_->encodeKey(levels, levelWidths_);
}

void HierarchicalKeyReader::levelsOf(
    uint32_t startRow,
    uint32_t numRows,
    std::span<uint64_t> levels) const {
  NIMBLE_CHECK_LE(startRow, rowCount_);
  NIMBLE_CHECK_LE(numRows, rowCount_ - startRow);
  NIMBLE_CHECK_EQ(levels.size(), static_cast<size_t>(numRows) * numLevels_);
  if (numRows == 0) {
    return;
  }

  // Finds the span of each level from the leaf up: the values that hold the
  // first and the last row.
  const auto spans = levelSpanScratch(numLevels_);
  uint32_t startOffset{startRow};
  uint32_t lastOffset{startRow + numRows - 1};
  size_t numSpanValues{0};
  for (uint32_t level = numLevels_; level-- > 0;) {
    const auto& offsets = childOffsets_[level];
    startOffset = offsets.containingRange(startOffset);
    lastOffset =
        numRows == 1 ? startOffset : offsets.containingRange(lastOffset);
    spans[level].startIndex = startOffset;
    spans[level].lastIndex = lastOffset;
    spans[level].currentIndex = 0;
    numSpanValues += lastOffset - startOffset + 1;
  }

  // Reads the values and the child end offsets of each span with one read
  // each, which costs less than reading them one by one where the level
  // changes, as the walk below visits every value of a span in order.
  const auto levelValues = levelValuesSpanScratch(numSpanValues);
  const auto childOffsets = childOffsetsSpanScratch(numSpanValues);
  size_t offset{0};
  for (uint32_t level = 0; level < numLevels_; ++level) {
    auto& span = spans[level];
    const size_t numValues = span.lastIndex - span.startIndex + 1;
    span.levelValues = levelValues.subspan(offset, numValues);
    span.childOffsets = childOffsets.subspan(offset, numValues);
    levelSequences_[level]->read(span.startIndex, span.levelValues);
    childOffsets_[level].read(span.startIndex + 1, span.childOffsets);
    offset += numValues;
  }

  // Every value has a child, so the next row moves a level only when it moves
  // the level below past the child range of the current value. The current
  // value of each level stays in its span, which only debug builds check: the
  // rows end at the last row, which the last value of every span holds, so the
  // leaf moves at most to the value of the last row, and a level above moves
  // only to the value that holds the moved value of the level below. These hold
  // for any index that loads: readSequences() checks that the child offsets of
  // each level start at zero, increase, and end at the number of values of the
  // next level.
  const uint32_t leaf = numLevels_ - 1;
  auto* output = levels.data();
  for (uint32_t row = startRow; row < startRow + numRows; ++row) {
    if (row >= spans[leaf].currentChildEndOffset()) {
      ++spans[leaf].currentIndex;
      NIMBLE_DCHECK_LT(
          spans[leaf].currentIndex, spans[leaf].levelValues.size());
      for (uint32_t level = leaf; level > 0 &&
           spans[level].currentLevelIndex() >=
               spans[level - 1].currentChildEndOffset();
           --level) {
        ++spans[level - 1].currentIndex;
        NIMBLE_DCHECK_LT(
            spans[level - 1].currentIndex, spans[level - 1].levelValues.size());
      }
    }
    for (uint32_t level = 0; level < numLevels_; ++level) {
      *output++ = spans[level].currentValue();
    }
  }
}

void HierarchicalKeyReader::readKeys(
    uint32_t startRow,
    uint32_t numRows,
    std::vector<std::string_view>& keys,
    const KeyCursor::BufferAllocator& bufferAllocator) const {
  if (numRows == 0) {
    return;
  }
  const auto keySize = keyCodec_->keySize(levelWidths_);
  auto* const buffer = static_cast<char*>(
      bufferAllocator(static_cast<size_t>(numRows) * keySize));
  NIMBLE_CHECK_NOT_NULL(buffer);
  const auto levels = levelScratch(numRows, numLevels_);
  levelsOf(startRow, numRows, levels);
  keyCodec_->encodeKeys(levels, levelWidths_, buffer);
  keys.reserve(keys.size() + numRows);
  for (uint32_t row = 0; row < numRows; ++row) {
    keys.emplace_back(buffer + static_cast<size_t>(row) * keySize, keySize);
  }
}

namespace {

// Walks rows, rebuilding each key from its hierarchy components. next() for one
// key finds the components of its row on its own, so it only spares the caller
// a std::string per row. next() for a batch walks the components of
// consecutive rows together, through readKeys().
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

  void next(
      size_t maxKeys,
      std::vector<std::string_view>& keys,
      const BufferAllocator& bufferAllocator) override {
    const auto numRows =
        static_cast<uint32_t>(std::min<size_t>(maxKeys, rowCount_ - row_));
    encoding_.readKeys(row_, numRows, keys, bufferAllocator);
    row_ += numRows;
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
    bool cacheEncodingViews,
    std::shared_ptr<const HierarchicalKeyCodec> keyCodec,
    velox::memory::MemoryPool* pool) {
  NIMBLE_CHECK_NOT_NULL(pool);
  return std::make_unique<HierarchicalKeyReader>(
      encodedKeys, cacheEncodingViews, std::move(keyCodec), pool);
}

} // namespace facebook::nimble::index
