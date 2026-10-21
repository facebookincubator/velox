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

#include "velox/dwio/nimble/index/HierarchicalClusterIndexWriter.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include "folly/ScopeGuard.h"
#include "folly/String.h"
#include "velox/buffer/Buffer.h"
#include "velox/common/Casts.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/EliasFanoEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/index/HierarchicalKeyFormat.h"
#include "velox/dwio/nimble/index/IndexKeyEncoder.h"
#include "velox/dwio/nimble/index/KeyChunkBuilder.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::nimble::index {
namespace {

// Normalized values and serialized payload width for one key column.
struct KeyColumnInput {
  explicit KeyColumnInput(
      std::span<const uint64_t> values,
      uint8_t packedKeyBytes = sizeof(uint64_t))
      : values{values}, packedKeyBytes{packedKeyBytes} {}

  size_t size() const {
    return values.size();
  }

  uint64_t operator[](size_t row) const {
    return values[row];
  }

  std::span<const uint64_t> values;
  uint8_t packedKeyBytes;
};

// Returns explicit sort orders, defaulting every key column to ascending.
std::vector<SortOrder> getSortOrders(const ClusterIndexConfig& config) {
  NIMBLE_USER_CHECK(
      config.sortOrders.empty() ||
          config.sortOrders.size() == config.columns.size(),
      "Cluster index columns and sort orders must have the same size");
  if (!config.sortOrders.empty()) {
    return config.sortOrders;
  }
  return std::vector<SortOrder>(
      config.columns.size(), SortOrder{.ascending = true});
}

// Returns whether adjacent keys satisfy the configured duplicate policy.
bool isKeyOrdered(int32_t comparison, bool noDuplicateKey) {
  return noDuplicateKey ? comparison > 0 : comparison >= 0;
}

// Reports an order violation using normalized key components.
[[noreturn]] void failKeyOrder(
    bool noDuplicateKey,
    size_t currIndex,
    std::span<const uint64_t> currKey,
    size_t prevIndex,
    std::span<const uint64_t> prevKey) {
  NIMBLE_USER_FAIL(
      noDuplicateKey
          ? "Encoded keys must be in strictly ascending order (duplicates are not allowed). "
            "Key at index {} (components: [{}]) is not greater than key at index {} (components: [{}])"
          : "Encoded keys must be in ascending order. "
            "Key at index {} (components: [{}]) is less than key at index {} (components: [{}])",
      currIndex,
      folly::join(", ", currKey),
      prevIndex,
      folly::join(", ", prevKey));
}

class HierarchicalKeyChunkBuilder final : public KeyChunkBuilder {
 public:
  HierarchicalKeyChunkBuilder(
      std::unique_ptr<IndexKeyEncoder> keyEncoder,
      const velox::RowTypePtr& inputType,
      std::span<const velox::column_index_t> keyColumnIndices,
      std::span<const SortOrder> sortOrders,
      bool enforceKeyOrder,
      bool noDuplicateKey,
      velox::memory::MemoryPool* pool)
      : keyEncoder_{std::move(keyEncoder)},
        keyColumnIndices_{keyColumnIndices.begin(), keyColumnIndices.end()},
        sortOrders_{sortOrders.begin(), sortOrders.end()},
        noDuplicateKey_{noDuplicateKey},
        encodedKeyStream_{pool},
        keyBuffer_{*velox::checkedNotNull(pool)},
        childBuffer_{*velox::checkedNotNull(pool)} {
    NIMBLE_CHECK_NOT_NULL(pool);
    NIMBLE_CHECK_NOT_NULL(keyEncoder_);
    NIMBLE_USER_CHECK(
        enforceKeyOrder,
        "Hierarchical key chunks require enforceKeyOrder=true.");
    NIMBLE_CHECK_EQ(keyColumnIndices_.size(), sortOrders_.size());
    NIMBLE_USER_CHECK_LE(
        keyColumnIndices_.size(),
        std::numeric_limits<uint8_t>::max(),
        "Hierarchical key chunks support at most 255 columns.");
    keyColumns_.reserve(keyColumnIndices_.size());
    keyColumnKinds_.reserve(keyColumnIndices_.size());
    keyColumnBytes_.reserve(keyColumnIndices_.size());
    for (const auto columnIndex : keyColumnIndices_) {
      const auto& columnType = inputType->childAt(columnIndex);
      const auto kind = columnType->kind();
      NIMBLE_USER_CHECK(
          kind == velox::TypeKind::BOOLEAN ||
              kind == velox::TypeKind::TINYINT ||
              kind == velox::TypeKind::SMALLINT ||
              kind == velox::TypeKind::INTEGER ||
              kind == velox::TypeKind::BIGINT,
          "Hierarchical key column must be integral: '{}', type {}.",
          inputType->nameOf(columnIndex),
          kind);
      keyColumnKinds_.push_back(kind);
      keyColumnBytes_.push_back(
          static_cast<uint8_t>(columnType->cppSizeInBytes()));
      keyColumns_.emplace_back(pool);
    }
    keyColumnInputs_.reserve(keyColumns_.size());
    index_.levels.resize(keyColumns_.size());
    encodedLevelSequences_.reserve(keyColumns_.size());
  }

  void append(const velox::VectorPtr& input) override {
    const auto* row = input->asChecked<velox::RowVector>();
    const auto newKeyStart = size();
    encodedKeys_.clear();
    encodedKeys_.reserve(input->size());
    keyEncoder_->encode(input, encodedKeys_, [this](size_t bytes) {
      return keyBuffer_.reserve(bytes);
    });
    for (const auto key : encodedKeys_) {
      encodedKeyStream_.push_back(key);
    }
    for (size_t level = 0; level < keyColumnIndices_.size(); ++level) {
      velox::DecodedVector decoded(*row->childAt(keyColumnIndices_.at(level)));
      auto& keyColumn = keyColumns_.at(level);
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          appendKeyColumn,
          keyColumnKinds_.at(level),
          decoded,
          sortOrders_.at(level).ascending,
          input->size(),
          keyColumn);
    }
    validateOrder(newKeyStart);
  }

  size_t size() const override {
    return keyColumns_.at(0).size();
  }

  std::string keyAt(size_t row) const override {
    NIMBLE_CHECK_LT(row, encodedKeyStream_.size());
    return std::string{*(encodedKeyStream_.begin() + row)};
  }

  std::string_view encode(size_t offset, uint32_t count, Buffer& buffer)
      const override {
    NIMBLE_CHECK_GT(count, 0);
    NIMBLE_CHECK_LE(offset + count, size());
    resetScratch();
    SCOPE_EXIT {
      resetScratch();
    };
    for (const auto& keyColumn : keyColumns_) {
      const auto level = keyColumnInputs_.size();
      keyColumnInputs_.emplace_back(
          std::span<const uint64_t>{keyColumn.data(), keyColumn.size()}.subspan(
              offset, count),
          keyColumnBytes_.at(level));
    }
    buildIndex();
    encodeIndex();
    return serializeIndex(buffer);
  }

  void clear() override {
    for (auto& keyColumn : keyColumns_) {
      keyColumn.resize(0);
    }
    encodedKeyStream_.clear();
    keyBuffer_.reset();
    resetScratch();
  }

 private:
  // Levels are ordered from the first key column to the leaf key column.
  struct Index {
    struct Level {
      // Distinct values at this level, concatenated across parent groups.
      std::vector<uint64_t> values;
      // For non-leaf levels, ranges in the next level. For the leaf level,
      // source-row ranges for each complete key.
      std::vector<uint32_t> childOffsets;
    };

    std::vector<Level> levels;

    // Returns whether any complete key appears in multiple source rows.
    bool hasDuplicateKeys() const {
      NIMBLE_CHECK(!levels.empty());
      const auto& leaf = levels.at(levels.size() - 1);
      NIMBLE_CHECK(!leaf.childOffsets.empty());
      return leaf.values.size() !=
          leaf.childOffsets.at(leaf.childOffsets.size() - 1);
    }
  };

  inline static constexpr std::array<EncodingType, 4> kLevelEncodings{
      EncodingType::Constant,
      EncodingType::EliasFano,
      EncodingType::FixedBitWidth,
      EncodingType::Trivial,
  };

  template <typename T>
  static std::optional<uint64_t> estimateSequenceSize(
      std::span<const T> values,
      const Statistics<T>& statistics,
      EncodingType encodingType) {
    const Encoding::Options options{};
    if (encodingType == EncodingType::Constant) {
      return ConstantEncoding<T>::estimateSize(values, statistics, options);
    }
    if (encodingType == EncodingType::EliasFano) {
      return EliasFanoEncoding<T>::estimateSize(values, statistics, options);
    }
    if (encodingType == EncodingType::FixedBitWidth) {
      return FixedBitWidthEncoding<T>::estimateSize(
          values.size(), statistics, options);
    }
    NIMBLE_CHECK_EQ(encodingType, EncodingType::Trivial);
    return TrivialEncoding<T>::estimateSize(values.size());
  }

  template <typename T>
  static EncodingType selectSequenceEncoding(std::span<const T> values) {
    const auto statistics = Statistics<T>::create(values);
    EncodingType selected{EncodingType::Trivial};
    uint64_t selectedBytes{std::numeric_limits<uint64_t>::max()};
    for (uint32_t candidate = 0; candidate < kLevelEncodings.size();
         ++candidate) {
      const auto estimate =
          estimateSequenceSize(values, statistics, kLevelEncodings[candidate]);
      if (estimate.has_value() && estimate.value() < selectedBytes) {
        selectedBytes = estimate.value();
        selected = kLevelEncodings[candidate];
      }
    }
    return selected;
  }

  static EncodingLayout childEncodingLayout(EncodingType encodingType) {
    return {encodingType, {}, CompressionType::Uncompressed};
  }

  template <typename T>
  static std::unique_ptr<EncodingSelectionPolicy<T>> sequencePolicy(
      EncodingType encodingType) {
    return std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
        childEncodingLayout(encodingType),
        CompressionOptions{},
        [](DataType dataType) {
          return ManualEncodingSelectionPolicyFactory{
              ManualEncodingSelectionPolicyFactory::
                  defaultEncodingReadFactors(),
              std::nullopt}
              .createPolicy(dataType);
        });
  }

  template <typename T>
  std::string_view encodeSequence(
      std::span<const T> values,
      EncodingType encodingType) const {
    NIMBLE_CHECK(!values.empty());
    return EncodingFactory::encode<T>(
        sequencePolicy<T>(encodingType), values, childBuffer_);
  }

  template <typename T>
  std::span<const T> physicalValues(std::span<const uint64_t> values) const {
    if constexpr (std::is_same_v<T, uint64_t>) {
      return values;
    }
    const auto bytes = values.size() * sizeof(T);
    if (physicalValuesBuffer_ == nullptr ||
        physicalValuesBuffer_->capacity() < bytes) {
      physicalValuesBuffer_ = velox::AlignedBuffer::allocate<char>(
          bytes, &childBuffer_.getMemoryPool());
    }
    physicalValuesBuffer_->setSize(bytes);
    auto* output = physicalValuesBuffer_->asMutable<T>();
    NIMBLE_CHECK_NOT_NULL(output);
    for (size_t i = 0; i < values.size(); ++i) {
      output[i] = static_cast<T>(values[i]);
    }
    return {output, values.size()};
  }

  template <typename T>
  void encodeIndexLevel(size_t level) const {
    const auto values = physicalValues<T>(index_.levels.at(level).values);
    const std::array<uint32_t, 2> rootOffsets{
        0, static_cast<uint32_t>(values.size())};
    const std::span<const uint32_t> parentOffsets = level == 0
        ? rootOffsets
        : std::span<const uint32_t>{index_.levels.at(level - 1).childOffsets};
    for (uint32_t parent = 0; parent + 1 < parentOffsets.size(); ++parent) {
      const auto begin = parentOffsets[parent];
      const auto end = parentOffsets[parent + 1];
      NIMBLE_CHECK_LT(begin, end);
      const auto sequence = values.subspan(begin, end - begin);
      encodedLevelSequences_.push_back(
          encodeSequence(sequence, selectSequenceEncoding(sequence)));
    }
  }

  // Clears per-encode state while retaining allocated capacity for reuse.
  void resetScratch() const {
    keyColumnInputs_.clear();
    for (auto& level : index_.levels) {
      level.values.clear();
      level.childOffsets.clear();
    }
    childBuffer_.reset();
    encodedLevelSequences_.clear();
  }

  // Returns the first key level that differs from the preceding row. Returns
  // zero for the first row and the level count when the complete key repeats.
  uint32_t getNextLevel(uint32_t row) const {
    if (row == 0) {
      return 0;
    }
    const auto numLevels = static_cast<uint32_t>(keyColumnInputs_.size());
    for (uint32_t level = 0; level < numLevels; ++level) {
      const auto previousValue = keyColumnInputs_.at(level)[row - 1];
      const auto currentValue = keyColumnInputs_.at(level)[row];
      if (currentValue != previousValue) {
        NIMBLE_USER_CHECK_GT(
            currentValue,
            previousValue,
            "HierarchicalKeyReader rows must be non-decreasing.");
        return level;
      }
    }
    return numLevels;
  }

  // Builds distinct level values and their outgoing child or row ranges.
  void buildIndex() const {
    NIMBLE_CHECK(!keyColumnInputs_.empty());
    const auto numLevels = static_cast<uint32_t>(keyColumnInputs_.size());
    const auto numRows = static_cast<uint32_t>(keyColumnInputs_.at(0).size());
    for (auto& level : index_.levels) {
      level.childOffsets.push_back(0);
    }

    for (uint32_t row = 0; row < numRows; ++row) {
      // The first level whose value differs from the previous row starts a new
      // subtree, and every level below it starts a new value too. No differing
      // level means the row repeats the previous key, extending the leaf run.
      for (uint32_t level = getNextLevel(row); level < numLevels; ++level) {
        auto& indexLevel = index_.levels.at(level);
        if (row > 0) {
          const auto childEnd = level + 1 < numLevels
              ? index_.levels.at(level + 1).values.size()
              : row;
          indexLevel.childOffsets.push_back(static_cast<uint32_t>(childEnd));
        }
        indexLevel.values.push_back(keyColumnInputs_.at(level)[row]);
      }
    }

    for (uint32_t level = 0; level < numLevels; ++level) {
      const auto childEnd = level + 1 < numLevels
          ? index_.levels.at(level + 1).values.size()
          : numRows;
      index_.levels.at(level).childOffsets.push_back(
          static_cast<uint32_t>(childEnd));
    }
  }

  // Selects and encodes each parent-group sequence independently.
  void encodeIndex() const {
    const auto numLevels = index_.levels.size();
    NIMBLE_CHECK(encodedLevelSequences_.empty());
    for (size_t level = 0; level < numLevels; ++level) {
      switch (keyColumnBytes_.at(level)) {
        case sizeof(uint8_t):
          encodeIndexLevel<uint8_t>(level);
          break;
        case sizeof(uint16_t):
          encodeIndexLevel<uint16_t>(level);
          break;
        case sizeof(uint32_t):
          encodeIndexLevel<uint32_t>(level);
          break;
        case sizeof(uint64_t):
          encodeIndexLevel<uint64_t>(level);
          break;
        default:
          NIMBLE_UNREACHABLE(
              "Unsupported hierarchical key width: {}",
              keyColumnBytes_.at(level));
      }
    }
  }

  // Serializes the in-memory index and encoded sequences into `buffer`.
  std::string_view serializeIndex(Buffer& buffer) const {
    const auto numLevels = static_cast<uint32_t>(index_.levels.size());
    const auto numRows = static_cast<uint32_t>(keyColumnInputs_.at(0).size());
    const bool hasDuplicateKeys = index_.hasDuplicateKeys();

    uint64_t numOffsets{0};
    for (uint32_t level = 0; level + 1 < numLevels; ++level) {
      numOffsets += index_.levels.at(level).childOffsets.size();
    }
    if (hasDuplicateKeys) {
      // Leaf offsets map each distinct complete key to its source-row range.
      // Without duplicates this mapping is identity and need not be stored.
      numOffsets +=
          index_.levels.at(index_.levels.size() - 1).childOffsets.size();
    }
    // Serialized layout:
    // [version:u8][levels:u8][rowCount:u32]
    // [physicalWidth:u8] * levels
    // [valueCount:u32] * levels
    // [childOrRowOffset:u32] * numOffsets
    // [sequenceSize:u32] * sequences
    // [encodedSequenceBytes] * sequences
    uint64_t encodingBytes = HierarchicalKeyFormat::kHeaderSize +
        numLevels * sizeof(uint8_t) + numLevels * sizeof(uint32_t) +
        numOffsets * sizeof(uint32_t) +
        encodedLevelSequences_.size() * sizeof(uint32_t);
    for (const auto sequence : encodedLevelSequences_) {
      encodingBytes += sequence.size();
    }
    NIMBLE_CHECK_LE(encodingBytes, std::numeric_limits<uint32_t>::max());
    const auto encodedSize = static_cast<uint32_t>(encodingBytes);

    char* const reserved = &*velox::checkedNotNull(buffer.reserve(encodedSize));
    char* position = reserved;
    encoding::write<uint8_t>(HierarchicalKeyFormat::kVersion, position);
    encoding::write<uint8_t>(static_cast<uint8_t>(numLevels), position);
    encoding::writeUint32(numRows, position);
    for (const auto width : keyColumnBytes_) {
      encoding::write<uint8_t>(width, position);
    }
    for (uint32_t level = 0; level < numLevels; ++level) {
      encoding::writeUint32(
          static_cast<uint32_t>(index_.levels.at(level).values.size()),
          position);
    }
    const auto writeOffsets = [&position](std::span<const uint32_t> offsets) {
      for (const auto offset : offsets) {
        encoding::writeUint32(offset, position);
      }
    };
    for (uint32_t level = 0; level + 1 < numLevels; ++level) {
      writeOffsets(index_.levels.at(level).childOffsets);
    }
    if (hasDuplicateKeys) {
      writeOffsets(index_.levels.at(index_.levels.size() - 1).childOffsets);
    }
    for (const auto sequence : encodedLevelSequences_) {
      encoding::writeUint32(static_cast<uint32_t>(sequence.size()), position);
    }
    for (const auto sequence : encodedLevelSequences_) {
      NIMBLE_CHECK(!sequence.empty());
      encoding::writeBytes(sequence, position);
    }
    return {reserved, encodedSize};
  }

  template <typename T>
  static uint64_t normalize(T value, bool ascending) {
    static_assert(std::is_integral_v<T>);
    if constexpr (std::is_same_v<T, bool>) {
      const uint8_t normalized = value ? 2 : 1;
      return ascending ? normalized : static_cast<uint8_t>(~normalized);
    } else {
      using Unsigned = std::make_unsigned_t<T>;
      auto normalized =
          std::bit_cast<Unsigned>(value) ^ (Unsigned{1} << (sizeof(T) * 8 - 1));
      return ascending ? normalized : static_cast<Unsigned>(~normalized);
    }
  }

  template <velox::TypeKind kind>
  static void appendKeyColumn(
      const velox::DecodedVector& decoded,
      bool ascending,
      velox::vector_size_t size,
      Vector<uint64_t>& output) {
    if constexpr (
        kind == velox::TypeKind::BOOLEAN || kind == velox::TypeKind::TINYINT ||
        kind == velox::TypeKind::SMALLINT || kind == velox::TypeKind::INTEGER ||
        kind == velox::TypeKind::BIGINT) {
      using T = typename velox::TypeTraits<kind>::NativeType;
      const auto outputOffset = output.size();
      output.resize(outputOffset + size);
      auto* outputValues = output.data() + outputOffset;
      if (decoded.isConstantMapping()) {
        velox::simd::simdFill(
            outputValues,
            normalize(decoded.valueAt<T>(0), ascending),
            static_cast<uint32_t>(size));
        return;
      }
      if (decoded.isIdentityMapping()) {
        if constexpr (kind == velox::TypeKind::BOOLEAN) {
          const auto* inputValues =
              velox::checkedNotNull(decoded.data<uint64_t>());
          for (velox::vector_size_t row = 0; row < size; ++row) {
            outputValues[row] =
                normalize(velox::bits::isBitSet(inputValues, row), ascending);
          }
        } else {
          const auto* inputValues = velox::checkedNotNull(decoded.data<T>());
          for (velox::vector_size_t row = 0; row < size; ++row) {
            outputValues[row] = normalize(inputValues[row], ascending);
          }
        }
        return;
      }
      for (velox::vector_size_t row = 0; row < size; ++row) {
        outputValues[row] = normalize(decoded.valueAt<T>(row), ascending);
      }
      return;
    }
    NIMBLE_UNREACHABLE("Unsupported hierarchical key type: {}.", kind);
  }

  std::vector<uint64_t> keyColumnsAt(size_t row) const {
    std::vector<uint64_t> keyComponents;
    keyComponents.reserve(keyColumns_.size());
    for (const auto& keyColumn : keyColumns_) {
      NIMBLE_CHECK_LT(row, keyColumn.size());
      keyComponents.emplace_back(keyColumn[row]);
    }
    return keyComponents;
  }

  int32_t compareRowKeys(size_t leftRow, size_t rightRow) const {
    NIMBLE_CHECK_LT(leftRow, size());
    NIMBLE_CHECK_LT(rightRow, size());
    for (const auto& keyColumn : keyColumns_) {
      const auto leftValue = keyColumn[leftRow];
      const auto rightValue = keyColumn[rightRow];
      if (leftValue != rightValue) {
        return leftValue < rightValue ? -1 : 1;
      }
    }
    return 0;
  }

  int32_t compareRowKeys(size_t row, std::span<const uint64_t> keyColumnValues)
      const {
    NIMBLE_CHECK_EQ(keyColumns_.size(), keyColumnValues.size());
    NIMBLE_CHECK_LT(row, size());
    for (size_t level = 0; level < keyColumns_.size(); ++level) {
      const auto rowValue = keyColumns_.at(level)[row];
      const auto keyValue = keyColumnValues[level];
      if (rowValue != keyValue) {
        return rowValue < keyValue ? -1 : 1;
      }
    }
    return 0;
  }

  void validateOrder(size_t newKeyStart) {
    if (newKeyStart == 0 && lastKey_.has_value()) {
      const auto comparison = compareRowKeys(0, lastKey_.value());
      if (!isKeyOrdered(comparison, noDuplicateKey_)) {
        const auto currentKey = keyColumnsAt(0);
        failKeyOrder(noDuplicateKey_, 0, currentKey, 0, lastKey_.value());
      }
    }

    for (auto row = std::max<size_t>(newKeyStart, 1); row < size(); ++row) {
      if (FOLLY_UNLIKELY(
              !isKeyOrdered(compareRowKeys(row, row - 1), noDuplicateKey_))) {
        const auto currentKey = keyColumnsAt(row);
        const auto previousKey = keyColumnsAt(row - 1);
        failKeyOrder(noDuplicateKey_, row, currentKey, row - 1, previousKey);
      }
    }
    lastKey_ = keyColumnsAt(size() - 1);
  }

  const std::unique_ptr<IndexKeyEncoder> keyEncoder_;
  const std::vector<velox::column_index_t> keyColumnIndices_;
  const std::vector<SortOrder> sortOrders_;
  const bool noDuplicateKey_;
  Vector<std::string_view> encodedKeyStream_;
  Buffer keyBuffer_;
  std::vector<std::string_view> encodedKeys_;
  std::vector<velox::TypeKind> keyColumnKinds_;
  std::vector<uint8_t> keyColumnBytes_;
  std::vector<Vector<uint64_t>> keyColumns_;
  mutable std::vector<KeyColumnInput> keyColumnInputs_;
  mutable Index index_;
  mutable Buffer childBuffer_;
  mutable velox::BufferPtr physicalValuesBuffer_;
  mutable std::vector<std::string_view> encodedLevelSequences_;
  std::optional<std::vector<uint64_t>> lastKey_;
};

// Creates the key-chunk builder used by the hierarchical writer.
std::unique_ptr<KeyChunkBuilder> createHierarchicalKeyChunkBuilder(
    std::unique_ptr<IndexKeyEncoder> keyEncoder,
    const velox::RowTypePtr& inputType,
    std::span<const velox::column_index_t> keyColumnIndices,
    std::span<const SortOrder> sortOrders,
    bool enforceKeyOrder,
    bool noDuplicateKey,
    velox::memory::MemoryPool* pool) {
  return std::make_unique<HierarchicalKeyChunkBuilder>(
      std::move(keyEncoder),
      inputType,
      keyColumnIndices,
      sortOrders,
      enforceKeyOrder,
      noDuplicateKey,
      pool);
}

} // namespace

std::unique_ptr<HierarchicalClusterIndexWriter>
HierarchicalClusterIndexWriter::create(
    const IndexConfig& config,
    const velox::TypePtr& inputType,
    velox::memory::MemoryPool* pool) {
  NIMBLE_USER_CHECK_EQ(config.family, IndexFamily::Cluster);
  NIMBLE_CHECK_NOT_NULL(pool, "memory pool must not be null");
  const auto& hierarchicalConfig =
      checkedIndexConfig<ClusterIndexConfig>(config);
  return std::unique_ptr<HierarchicalClusterIndexWriter>(
      new HierarchicalClusterIndexWriter(
          hierarchicalConfig,
          velox::asRowType(inputType),
          getSortOrders(hierarchicalConfig),
          pool));
}

HierarchicalClusterIndexWriter::HierarchicalClusterIndexWriter(
    const ClusterIndexConfig& config,
    const velox::RowTypePtr& inputType,
    std::vector<SortOrder> sortOrders,
    velox::memory::MemoryPool* pool)
    : ClusterIndexWriterBase{
          inputType,
          Options{
              .indexName = config.name,
              .columns = config.columns,
              .sortOrders = sortOrders,
              .maxRowsPerKeyChunk = config.maxRowsPerKeyChunk,
              .keyChunkCompressionType = config.keyChunkCompressionType,
          },
          createHierarchicalKeyChunkBuilder(
              createNimbleIndexKeyEncoder(
                  config.columns,
                  inputType,
                  sortOrders,
                  pool),
              inputType,
              getKeyColumnIndices({config.columns}, inputType),
              sortOrders,
              config.enforceKeyOrder,
              config.noDuplicateKey,
              pool),
          pool} {}

HierarchicalClusterIndexWriter::~HierarchicalClusterIndexWriter() = default;

} // namespace facebook::nimble::index
