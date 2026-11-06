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
#include "velox/dwio/nimble/index/HierarchicalClusterIndexFactory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>

#include "folly/lang/Bits.h"
#include "velox/buffer/Buffer.h"
#include "velox/common/Casts.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/HierarchicalClusterIndex.h"
#include "velox/dwio/nimble/index/HierarchicalClusterIndexWriter.h"
#include "velox/dwio/nimble/index/HierarchicalKeyFormat.h"
#include "velox/dwio/nimble/index/IndexConstants.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::nimble::index {

namespace {

// Returns the value of the level of the key column value at 'row' of 'values',
// a column of 'kind', which the writer of the index stores.
uint64_t levelValueAt(
    const velox::DecodedVector& values,
    velox::vector_size_t row,
    velox::TypeKind kind,
    bool ascending) {
  // hierarchicalKeyTypes() admits only the integral kinds below.
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (kind) {
    case velox::TypeKind::BOOLEAN:
      return HierarchicalKeyFormat::normalizeKeyValue(
          values.valueAt<bool>(row), ascending);
    case velox::TypeKind::TINYINT:
      return HierarchicalKeyFormat::normalizeKeyValue(
          values.valueAt<int8_t>(row), ascending);
    case velox::TypeKind::SMALLINT:
      return HierarchicalKeyFormat::normalizeKeyValue(
          values.valueAt<int16_t>(row), ascending);
    case velox::TypeKind::INTEGER:
      return HierarchicalKeyFormat::normalizeKeyValue(
          values.valueAt<int32_t>(row), ascending);
    case velox::TypeKind::BIGINT:
      return HierarchicalKeyFormat::normalizeKeyValue(
          values.valueAt<int64_t>(row), ascending);
    default:
      NIMBLE_UNREACHABLE("Unsupported hierarchical key type: {}.", kind);
  }
}

// Returns the type of each of 'columns' of 'inputType'. Throws unless each is
// an integral column of 'inputType', which a level of the index can hold.
std::vector<velox::TypePtr> hierarchicalKeyTypes(
    const std::vector<std::string>& columns,
    const velox::RowType& inputType) {
  NIMBLE_USER_CHECK(
      !columns.empty(), "Hierarchical index key encoder requires key columns.");
  NIMBLE_USER_CHECK_LE(
      columns.size(),
      std::numeric_limits<uint8_t>::max(),
      "Hierarchical key chunks support at most 255 columns.");
  std::vector<velox::TypePtr> types;
  types.reserve(columns.size());
  for (const auto& column : columns) {
    const auto index = inputType.getChildIdxIfExists(column);
    NIMBLE_USER_CHECK(
        index.has_value(), "Hierarchical key column is missing: '{}'.", column);
    const auto& type = inputType.childAt(*index);
    const auto kind = type->kind();
    NIMBLE_USER_CHECK(
        kind == velox::TypeKind::BOOLEAN || kind == velox::TypeKind::TINYINT ||
            kind == velox::TypeKind::SMALLINT ||
            kind == velox::TypeKind::INTEGER || kind == velox::TypeKind::BIGINT,
        "Hierarchical key column must be integral: '{}', type {}.",
        column,
        kind);
    types.push_back(type);
  }
  return types;
}

// Returns the width in bytes of the level of each key column of 'types'.
std::vector<uint8_t> levelWidths(const std::vector<velox::TypePtr>& types) {
  std::vector<uint8_t> widths;
  widths.reserve(types.size());
  for (const auto& type : types) {
    widths.push_back(static_cast<uint8_t>(type->cppSizeInBytes()));
  }
  return widths;
}

// Returns the largest value of a level 'width' bytes wide.
uint64_t maxLevelValue(uint8_t width) {
  return width == sizeof(uint64_t) ? std::numeric_limits<uint64_t>::max()
                                   : (uint64_t{1} << (8 * width)) - 1;
}

// The key encoder of a hierarchical cluster index, which writes its keys with
// the codec of the index. The key columns of a row give the values of the
// levels, as they give them to the writer of the index. It is not thread-safe:
// its calls reuse a buffer of the values of the levels.
class NimbleHierarchicalIndexKeyEncoder final : public IndexKeyEncoder {
 public:
  NimbleHierarchicalIndexKeyEncoder(
      const std::vector<std::string>& columns,
      const velox::RowType& inputType,
      const std::vector<SortOrder>& sortOrders,
      std::shared_ptr<const HierarchicalKeyCodec> keyCodec,
      velox::memory::MemoryPool* pool)
      : keyCodec_{velox::checkedNotNull(std::move(keyCodec))},
        pool_{velox::checkedNotNull(pool)},
        columns_{columns},
        sortOrders_{sortOrders},
        keyTypes_{hierarchicalKeyTypes(columns, inputType)},
        levelWidths_{levelWidths(keyTypes_)},
        keySize_{keyCodec_->keySize(levelWidths_)} {
    NIMBLE_USER_CHECK_EQ(
        sortOrders_.size(),
        columns_.size(),
        "Hierarchical index key encoder requires a sort order for each key column.");
  }

  void encode(
      const velox::VectorPtr& input,
      std::vector<std::string_view>& encodedKeys,
      const BufferAllocator& bufferAllocator) override {
    const auto& rows = *input->asChecked<velox::RowVector>();
    const auto levels = levelsOf(rows);
    const auto numRows = static_cast<size_t>(rows.size());
    const size_t numBytes = numRows * keySize_;
    auto* output = static_cast<char*>(bufferAllocator(numBytes));
    if (numBytes > 0) {
      NIMBLE_CHECK_NOT_NULL(output, "Hierarchical key allocator returned null");
    }
    keyCodec_->encodeKeys(levels, levelWidths_, output);
    encodedKeys.clear();
    encodedKeys.reserve(numRows);
    for (size_t row = 0; row < numRows; ++row) {
      encodedKeys.emplace_back(output + row * keySize_, keySize_);
    }
  }

  std::vector<velox::serializer::EncodedKeyBounds> encodeIndexBounds(
      const velox::serializer::IndexBounds& indexBounds) override {
    NIMBLE_USER_CHECK(
        indexBounds.validate(),
        "Hierarchical index key encoder requires valid index bounds.");
    std::vector<velox::serializer::EncodedKeyBounds> results(
        indexBounds.numRows());
    if (indexBounds.lowerBound.has_value()) {
      const auto& lowerBound = *indexBounds.lowerBound;
      const auto levels = levelsOf(*lowerBound.bound);
      for (size_t row = 0; row < results.size(); ++row) {
        const auto rowLevels = levelsOfRow(levels, row);
        if (!lowerBound.inclusive) {
          NIMBLE_USER_CHECK(
              increment(rowLevels),
              "Exclusive lower bound of a hierarchical index has no next key.");
        }
        results[row].lowerKey = keyCodec_->encodeKey(rowLevels, levelWidths_);
      }
    }
    if (indexBounds.upperBound.has_value()) {
      const auto& upperBound = *indexBounds.upperBound;
      const auto levels = levelsOf(*upperBound.bound);
      for (size_t row = 0; row < results.size(); ++row) {
        const auto rowLevels = levelsOfRow(levels, row);
        // An inclusive upper bound on the last values of the levels bounds no
        // key.
        if (upperBound.inclusive && !increment(rowLevels)) {
          continue;
        }
        results[row].upperKey = keyCodec_->encodeKey(rowLevels, levelWidths_);
      }
    }
    return results;
  }

 private:
  // Returns room for 'numValues' values of the levels. levels_ only grows, so
  // the calls reuse it.
  uint64_t* ensureLevels(size_t numValues) {
    if (levels_ == nullptr ||
        levels_->capacity() < numValues * sizeof(uint64_t)) {
      levels_ = velox::AlignedBuffer::allocate<uint64_t>(numValues, pool_);
    }
    return levels_->asMutable<uint64_t>();
  }

  // Returns the values of the levels of each row of 'rows', which holds the
  // key columns: those of each row after those of the row before. They stay in
  // levels_ until the next call.
  std::span<uint64_t> levelsOf(const velox::RowVector& rows) {
    const auto& rowType = rows.type()->asRow();
    const auto numLevels = columns_.size();
    const auto numValues = static_cast<size_t>(rows.size()) * numLevels;
    auto* levels = ensureLevels(numValues);
    for (size_t level = 0; level < numLevels; ++level) {
      const auto& column = columns_[level];
      const auto index = rowType.getChildIdxIfExists(column);
      NIMBLE_USER_CHECK(
          index.has_value(),
          "Hierarchical key column is missing: '{}'.",
          column);
      const auto kind = keyTypes_[level]->kind();
      NIMBLE_USER_CHECK_EQ(
          rowType.childAt(*index)->kind(),
          kind,
          "Hierarchical key column changed its type: '{}'.",
          column);
      const velox::DecodedVector values{*rows.childAt(*index)};
      for (velox::vector_size_t row = 0; row < rows.size(); ++row) {
        NIMBLE_USER_CHECK(
            !values.isNullAt(row),
            "Hierarchical key column must not be null: '{}'.",
            column);
        levels[static_cast<size_t>(row) * numLevels + level] =
            levelValueAt(values, row, kind, sortOrders_[level].ascending);
      }
    }
    return {levels, numValues};
  }

  // Returns the values of the levels of 'row' in 'levels', which levelsOf()
  // returned.
  std::span<uint64_t> levelsOfRow(std::span<uint64_t> levels, size_t row)
      const {
    const auto numLevels = columns_.size();
    return levels.subspan(row * numLevels, numLevels);
  }

  // Moves 'levels' to the next values in the order of the index: the last
  // level below the largest value of its width takes its next value, and the
  // levels after it their smallest. Returns false when every level has its
  // largest value.
  bool increment(std::span<uint64_t> levels) const {
    for (auto level = levels.size(); level-- > 0;) {
      if (levels[level] < maxLevelValue(levelWidths_[level])) {
        ++levels[level];
        return true;
      }
      levels[level] = 0;
    }
    return false;
  }

  // Writes the keys of the index.
  const std::shared_ptr<const HierarchicalKeyCodec> keyCodec_;
  // Allocates levels_.
  velox::memory::MemoryPool* const pool_;
  // The key columns, one per level.
  const std::vector<std::string> columns_;
  // The sort order of each key column.
  const std::vector<SortOrder> sortOrders_;
  // The type of each key column.
  const std::vector<velox::TypePtr> keyTypes_;
  // The width in bytes of each level.
  const std::vector<uint8_t> levelWidths_;
  // The size of each key, fixed by the widths of the levels.
  const size_t keySize_;

  // Holds the values of the levels of the rows that the last call encodes,
  // reused across calls.
  velox::BufferPtr levels_;
};

// Nimble's key encoding: for each level, a zero null indicator and the value
// big-endian, which sorts bytewise. It takes only complete keys. The class is
// final, so the compiler resolves its own calls of keySize() and encodeKey()
// without a virtual dispatch, which encodeKeys() relies on for each row.
class NimbleHierarchicalKeyCodec final : public HierarchicalKeyCodec {
 public:
  size_t keySize(std::span<const uint8_t> levelWidths) const override {
    size_t bytes{0};
    for (const auto width : levelWidths) {
      bytes += 1 + width;
    }
    return bytes;
  }

  void encodeKey(
      std::span<const uint64_t> levels,
      std::span<const uint8_t> levelWidths,
      char* key) const override {
    NIMBLE_CHECK_EQ(levels.size(), levelWidths.size());
    for (size_t level = 0; level < levels.size(); ++level) {
      *key++ = '\0';
      const auto value = levels[level];
      switch (levelWidths[level]) {
        case sizeof(uint8_t):
          *key = static_cast<char>(value);
          break;
        case sizeof(uint16_t):
          folly::storeUnaligned<uint16_t>(
              key, folly::Endian::big(static_cast<uint16_t>(value)));
          break;
        case sizeof(uint32_t):
          folly::storeUnaligned<uint32_t>(
              key, folly::Endian::big(static_cast<uint32_t>(value)));
          break;
        case sizeof(uint64_t):
          folly::storeUnaligned<uint64_t>(key, folly::Endian::big(value));
          break;
        default:
          NIMBLE_UNREACHABLE(
              "Unsupported hierarchical key level width: {}",
              levelWidths[level]);
      }
      key += levelWidths[level];
    }
  }

  void encodeKeys(
      std::span<const uint64_t> levels,
      std::span<const uint8_t> levelWidths,
      char* keys) const override {
    const auto numLevels = levelWidths.size();
    NIMBLE_CHECK_GT(numLevels, 0);
    NIMBLE_CHECK_EQ(levels.size() % numLevels, 0);
    const auto size = keySize(levelWidths);
    for (size_t offset = 0; offset < levels.size(); offset += numLevels) {
      encodeKey(levels.subspan(offset, numLevels), levelWidths, keys);
      keys += size;
    }
  }

  bool parseKey(
      std::string_view target,
      std::span<const uint8_t> levelWidths,
      std::span<uint64_t> levels) const override {
    NIMBLE_CHECK_EQ(levels.size(), levelWidths.size());
    NIMBLE_CHECK_EQ(
        target.size(),
        keySize(levelWidths),
        "HierarchicalKeyReader requires a complete encoded key.");
    size_t offset{0};
    for (size_t level = 0; level < levelWidths.size(); ++level) {
      NIMBLE_CHECK_EQ(
          target[offset],
          0,
          "HierarchicalKeyReader requires non-null encoded key components.");
      const auto width = levelWidths[level];
      std::array<char, sizeof(uint64_t)> encodedBytes{};
      std::copy_n(
          target.begin() +
              static_cast<std::string_view::difference_type>(offset + 1),
          width,
          encodedBytes.end() - width);
      levels[level] = folly::Endian::big(std::bit_cast<uint64_t>(encodedBytes));
      offset += 1 + width;
    }
    return false;
  }
};

// Returns the codec of Nimble's key encoding, which every factory of the
// built-in index shares.
const std::shared_ptr<const HierarchicalKeyCodec>& nimbleKeyCodec() {
  static const std::shared_ptr<const HierarchicalKeyCodec> keyCodec =
      std::make_shared<const NimbleHierarchicalKeyCodec>();
  return keyCodec;
}

} // namespace

std::string HierarchicalKeyCodec::encodeKey(
    std::span<const uint64_t> levels,
    std::span<const uint8_t> levelWidths) const {
  std::string key(keySize(levelWidths), '\0');
  encodeKey(levels, levelWidths, key.data());
  return key;
}

void HierarchicalKeyCodec::encodeKeys(
    std::span<const uint64_t> levels,
    std::span<const uint8_t> levelWidths,
    char* keys) const {
  const auto numLevels = levelWidths.size();
  NIMBLE_CHECK_GT(numLevels, 0);
  NIMBLE_CHECK_EQ(levels.size() % numLevels, 0);
  const auto size = keySize(levelWidths);
  for (size_t offset = 0; offset < levels.size(); offset += numLevels) {
    encodeKey(levels.subspan(offset, numLevels), levelWidths, keys);
    keys += size;
  }
}

NimbleHierarchicalClusterIndexFactory::NimbleHierarchicalClusterIndexFactory()
    : NimbleHierarchicalClusterIndexFactory{
          std::string{kHierarchicalClusterIndexName},
          nimbleKeyCodec()} {}

NimbleHierarchicalClusterIndexFactory::NimbleHierarchicalClusterIndexFactory(
    std::string name,
    std::shared_ptr<const HierarchicalKeyCodec> keyCodec)
    : name_{std::move(name)},
      keyCodec_{velox::checkedNotNull(std::move(keyCodec))} {}

std::string_view NimbleHierarchicalClusterIndexFactory::name() const {
  return name_;
}

std::unique_ptr<IndexWriter>
NimbleHierarchicalClusterIndexFactory::createWriter(
    const IndexConfig& config,
    const velox::TypePtr& inputType,
    velox::memory::MemoryPool* pool) const {
  NIMBLE_USER_CHECK_EQ(config.family, IndexFamily::Cluster);
  NIMBLE_USER_CHECK_EQ(config.name, name());
  return HierarchicalClusterIndexWriter::create(
      config, inputType, keyCodec_, pool);
}

std::unique_ptr<IndexKeyEncoder>
NimbleHierarchicalClusterIndexFactory::createKeyEncoder(
    const std::vector<std::string>& columns,
    const velox::RowTypePtr& inputType,
    const std::vector<SortOrder>& sortOrders,
    velox::memory::MemoryPool* pool) const {
  NIMBLE_USER_CHECK_NOT_NULL(
      inputType, "Hierarchical index key encoder requires an input type.");
  return std::make_unique<NimbleHierarchicalIndexKeyEncoder>(
      columns, *inputType, sortOrders, keyCodec_, pool);
}

std::unique_ptr<ClusterIndexBase>
NimbleHierarchicalClusterIndexFactory::createReader(
    Section rootSection,
    velox::memory::MemoryPool* pool,
    const IndexLookup::Options& options) const {
  auto readerOptions = options;
  readerOptions.keyComparator = keyComparator();
  return HierarchicalClusterIndex::create(
      std::move(rootSection), keyCodec_, pool, readerOptions);
}

const KeyComparator* NimbleHierarchicalClusterIndexFactory::keyComparator()
    const {
  return keyCodec_->keyComparator();
}

const std::shared_ptr<const HierarchicalKeyCodec>&
NimbleHierarchicalClusterIndexFactory::testingKeyCodec() const {
  return keyCodec_;
}

} // namespace facebook::nimble::index
