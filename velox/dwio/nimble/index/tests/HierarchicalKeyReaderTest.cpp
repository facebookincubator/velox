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
#include "velox/dwio/nimble/index/HierarchicalClusterIndexWriter.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/index/HierarchicalKeyFormat.h"
#include "velox/dwio/nimble/index/IndexKeyEncoder.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble::index {
namespace {

constexpr uint32_t kPackedKeyColumnBytes{1 + sizeof(uint64_t)};

TEST(HierarchicalKeyFormatTest, supportedKeyValueBytes) {
  for (const uint8_t bytes : {uint8_t{1}, uint8_t{2}, uint8_t{4}, uint8_t{8}}) {
    SCOPED_TRACE(static_cast<uint32_t>(bytes));
    EXPECT_TRUE(HierarchicalKeyFormat::isSupportedKeyValueBytes(bytes));
  }
  for (const uint8_t bytes :
       {uint8_t{0}, uint8_t{3}, uint8_t{5}, uint8_t{16}}) {
    SCOPED_TRACE(static_cast<uint32_t>(bytes));
    EXPECT_FALSE(HierarchicalKeyFormat::isSupportedKeyValueBytes(bytes));
  }
}

std::string makeKey(std::span<const uint64_t> components) {
  std::string key(components.size() * kPackedKeyColumnBytes, '\0');
  char* position = key.data();
  for (const auto component : components) {
    *position++ = 0;
    for (int shift = 56; shift >= 0; shift -= 8) {
      *position++ = static_cast<char>(component >> shift);
    }
  }
  return key;
}

std::string makeKey(std::initializer_list<uint64_t> components) {
  return makeKey(
      std::span<const uint64_t>{components.begin(), components.size()});
}

void setUint32(std::string& data, size_t offset, uint32_t value) {
  NIMBLE_CHECK_LE(offset + sizeof(value), data.size());
  std::memcpy(data.data() + offset, &value, sizeof(value));
}

class HierarchicalKeyReaderTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!velox::memory::MemoryManager::testInstance()) {
      velox::memory::MemoryManager::testingSetInstance({});
    }
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool(
        "HierarchicalKeyReaderTest");
    pool_ = rootPool_->addLeafChild("test");
  }

  std::unique_ptr<HierarchicalKeyReader> build(
      const std::vector<std::vector<uint64_t>>& columns) {
    std::vector<std::string> names;
    std::vector<velox::VectorPtr> children;
    names.reserve(columns.size());
    children.reserve(columns.size());
    velox::test::VectorMaker vectorMaker{pool_.get()};
    for (size_t i = 0; i < columns.size(); ++i) {
      names.push_back("key" + std::to_string(i));
      std::vector<int64_t> values;
      values.reserve(columns[i].size());
      for (const auto value : columns[i]) {
        values.push_back(
            std::bit_cast<int64_t>(
                value ^ (uint64_t{1} << (sizeof(uint64_t) * 8 - 1))));
      }
      children.push_back(vectorMaker.flatVector<int64_t>(values));
    }

    return buildInput(
        vectorMaker.rowVector(names, children), /*sortOrders=*/{});
  }

  // Writes 'input' as one hierarchical key chunk and opens it.
  std::unique_ptr<HierarchicalKeyReader> buildInput(
      const velox::RowVectorPtr& input,
      std::vector<SortOrder> sortOrders) {
    const auto& names = asRowType(input->type())->names();
    const auto config = ClusterIndexConfigBuilder{kHierarchicalClusterIndexName}
                            .withKeyColumns(names)
                            .withSortOrders(std::move(sortOrders))
                            .withEnforceKeyOrder(true)
                            .build();
    auto writer = HierarchicalClusterIndexWriter::create(
        *config, input->type(), pool_.get());
    writer->write(input);

    chunkData_.clear();
    const WriteDataFn writeDataFn = [&](const auto& segments) {
      for (const auto segment : segments) {
        chunkData_.append(segment);
      }
      NIMBLE_CHECK_LE(chunkData_.size(), std::numeric_limits<uint32_t>::max());
      return std::pair<uint64_t, uint32_t>{
          0, static_cast<uint32_t>(chunkData_.size())};
    };
    uint32_t nextSectionId{0};
    const CreateMetadataSectionFn createMetadataFn =
        [&](std::string_view metadata) {
          NIMBLE_CHECK_LE(
              metadata.size(), std::numeric_limits<uint32_t>::max());
          return MetadataSection{
              nextSectionId++,
              static_cast<uint32_t>(metadata.size()),
              CompressionType::Uncompressed,
              static_cast<uint32_t>(metadata.size())};
        };
    writer->flush(writeDataFn, createMetadataFn);

    const char* position = chunkData_.data();
    const auto header = readChunkHeader(position);
    NIMBLE_CHECK_EQ(header.compressionType, CompressionType::Uncompressed);
    NIMBLE_CHECK_EQ(
        static_cast<size_t>(position - chunkData_.data()) + header.length,
        chunkData_.size());
    // Readers point into their chunk, so each built chunk stays alive for the
    // rest of the test.
    built_ = *builtChunks_.emplace_back(
        std::make_unique<std::string>(position, header.length));
    return std::make_unique<HierarchicalKeyReader>(built_, pool_.get());
  }

  // Row-major view of the same data, for round-trip expectations.
  static std::vector<std::string> packRows(
      const std::vector<std::vector<uint64_t>>& columns) {
    std::vector<std::string> keys;
    for (size_t row = 0; row < columns.front().size(); ++row) {
      std::vector<uint64_t> components;
      components.reserve(columns.size());
      for (const auto& column : columns) {
        components.push_back(column[row]);
      }
      keys.push_back(makeKey(components));
    }
    return keys;
  }

  // Returns the row seek() should return for 'key' in the sorted 'expected'.
  static std::optional<uint32_t> expectedSeek(
      const std::vector<std::string>& expected,
      const std::string& key,
      bool inclusive) {
    const auto it = inclusive
        ? std::lower_bound(expected.begin(), expected.end(), key)
        : std::upper_bound(expected.begin(), expected.end(), key);
    return it == expected.end()
        ? std::nullopt
        : std::optional<uint32_t>{static_cast<uint32_t>(it - expected.begin())};
  }

  // Checks every read of 'index' against 'expected', the sorted packed keys of
  // its rows: get(), materialize() of sub-ranges, cursors from several rows,
  // and both seek() forms of every key and of keys around them. Each column
  // packs one zero marker byte and big-endian value bytes of equal width.
  static void checkReads(
      const HierarchicalKeyReader& index,
      const std::vector<std::string>& expected) {
    const auto numRows = static_cast<uint32_t>(expected.size());
    ASSERT_EQ(index.rowCount(), numRows);
    const auto checkSeeks = [&](const std::string& key) {
      for (const bool inclusive : {true, false}) {
        EXPECT_EQ(
            index.seek(key, inclusive), expectedSeek(expected, key, inclusive));
      }
    };
    for (uint32_t row = 0; row < numRows; ++row) {
      SCOPED_TRACE(row);
      EXPECT_EQ(index.get(row), expected[row]);
      checkSeeks(expected[row]);
    }

    // Keys that match a stored key up to one column and miss there by one
    // step, which ends each lookup at a different level, plus the smallest and
    // largest possible keys.
    const auto columnBytes = expected[0].size() / index.numLevels();
    ASSERT_EQ(columnBytes * index.numLevels(), expected[0].size());
    std::string smallest = expected[0];
    std::string largest = expected[0];
    for (uint32_t level = 0; level < index.numLevels(); ++level) {
      std::fill_n(
          smallest.begin() + level * columnBytes + 1, columnBytes - 1, '\0');
      std::fill_n(
          largest.begin() + level * columnBytes + 1, columnBytes - 1, '\xff');
    }
    checkSeeks(smallest);
    checkSeeks(largest);
    const auto stride = std::max<uint32_t>(1, numRows / 100);
    for (uint32_t row = 0; row < numRows; row += stride) {
      SCOPED_TRACE(row);
      for (uint32_t level = 0; level < index.numLevels(); ++level) {
        const auto begin = level * columnBytes + 1;
        const auto end = (level + 1) * columnBytes;
        for (const bool up : {true, false}) {
          // Adds or subtracts one from the big-endian bytes [begin, end),
          // leaving the key unchanged when that would overflow.
          auto key = expected[row];
          auto byte = end;
          while (byte > begin && key[byte - 1] == (up ? '\xff' : '\0')) {
            key[--byte] = up ? '\0' : '\xff';
          }
          if (byte == begin) {
            continue;
          }
          key[byte - 1] = static_cast<char>(key[byte - 1] + (up ? 1 : -1));
          checkSeeks(key);
        }
      }
    }

    for (const auto& [start, count] :
         std::vector<std::pair<uint32_t, uint32_t>>{
             {0, numRows},
             {0, 0},
             {numRows, 0},
             {numRows - 1, 1},
             {numRows / 3, numRows / 2 - numRows / 3},
             {numRows / 2, numRows - numRows / 2}}) {
      SCOPED_TRACE(fmt::format("materialize start={} count={}", start, count));
      EXPECT_EQ(
          index.materialize(start, count),
          std::vector<std::string>(
              expected.begin() + start, expected.begin() + start + count));
    }

    for (const auto startRow : {0u, numRows / 2, numRows - 1}) {
      SCOPED_TRACE(fmt::format("cursor startRow={}", startRow));
      auto cursor = index.cursor(startRow);
      for (auto row = startRow; row < numRows; ++row) {
        ASSERT_TRUE(cursor->hasNext());
        EXPECT_EQ(cursor->next(), expected[row]);
      }
      EXPECT_FALSE(cursor->hasNext());
    }
  }

  // Builds sorted keys of type T for each sort order and shape, and checks
  // every read against the keys the index key encoder produces.
  template <typename T>
  void checkTypedKeys() {
    SCOPED_TRACE(fmt::format("keyValueBytes={}", sizeof(T)));
    constexpr SortOrder kAsc{.ascending = true};
    constexpr SortOrder kDesc{.ascending = false};
    const std::vector<std::vector<SortOrder>> orders{
        {kAsc, kAsc, kAsc, kAsc},
        {kDesc, kDesc, kDesc, kDesc},
        {kAsc, kAsc, kDesc, kDesc}};
    struct Shape {
      uint32_t numColumns;
      uint32_t numRows;
    };
    // One row, one level, and several levels whose last column repeats so the
    // leaf row offsets are stored. In the four-column shape, level 2 has a few
    // hundred values, so both its child offsets and the leaf row offsets exceed
    // the 64 entries the reader keeps decoded and are read encoded.
    const std::vector<Shape> shapes{{3, 1}, {1, 200}, {3, 800}, {4, 2'000}};
    // Each column draws from a few values, including the type's extremes and
    // negative values, so parents have several children.
    const std::vector<T> pool{
        std::numeric_limits<T>::min(),
        static_cast<T>(-100),
        static_cast<T>(-1),
        0,
        1,
        static_cast<T>(42),
        static_cast<T>(std::numeric_limits<T>::max() / 2),
        std::numeric_limits<T>::max()};
    velox::test::VectorMaker vectorMaker{pool_.get()};
    std::mt19937 rng{11};
    for (const auto& order : orders) {
      for (const auto& shape : shapes) {
        SCOPED_TRACE(
            fmt::format(
                "columns={} rows={} lastAscending={}",
                shape.numColumns,
                shape.numRows,
                order.back().ascending));
        std::vector<std::vector<T>> rows(
            shape.numRows, std::vector<T>(shape.numColumns));
        for (auto& row : rows) {
          for (auto& value : row) {
            value = pool[rng() % pool.size()];
          }
        }
        std::sort(
            rows.begin(), rows.end(), [&](const auto& lhs, const auto& rhs) {
              for (uint32_t column = 0; column < shape.numColumns; ++column) {
                if (lhs[column] != rhs[column]) {
                  return order[column].ascending == (lhs[column] < rhs[column]);
                }
              }
              return false;
            });
        if (shape.numColumns == 4) {
          // Confirms the shape reaches the encoded offsets it is meant to
          // cover: more than 64 level-2 values and repeated complete keys.
          uint32_t numLevel2Values{1};
          uint32_t numKeys{1};
          for (size_t i = 1; i < rows.size(); ++i) {
            numLevel2Values +=
                std::equal(
                    rows[i].begin(), rows[i].begin() + 3, rows[i - 1].begin())
                ? 0
                : 1;
            numKeys += rows[i] == rows[i - 1] ? 0 : 1;
          }
          ASSERT_GT(numLevel2Values, 64);
          ASSERT_GT(numKeys, 64);
          ASSERT_LT(numKeys, rows.size());
        }
        std::vector<std::string> names;
        std::vector<velox::VectorPtr> children;
        for (uint32_t column = 0; column < shape.numColumns; ++column) {
          names.push_back(fmt::format("key{}", column));
          std::vector<T> values;
          values.reserve(rows.size());
          for (const auto& row : rows) {
            values.push_back(row[column]);
          }
          children.push_back(vectorMaker.flatVector<T>(values));
        }
        const auto input = vectorMaker.rowVector(names, children);
        const std::vector<SortOrder> sortOrders{
            order.begin(), order.begin() + shape.numColumns};

        const auto keyEncoder = createNimbleIndexKeyEncoder(
            names, asRowType(input->type()), sortOrders, pool_.get());
        std::vector<std::string_view> encodedKeys;
        std::vector<std::unique_ptr<char[]>> keyBuffers;
        keyEncoder->encode(input, encodedKeys, [&](size_t size) {
          return keyBuffers.emplace_back(std::make_unique<char[]>(size)).get();
        });
        const std::vector<std::string> expected{
            encodedKeys.begin(), encodedKeys.end()};

        checkReads(*buildInput(input, sortOrders), expected);
      }
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::string chunkData_;
  std::vector<std::unique_ptr<std::string>> builtChunks_;
  std::string_view built_;
};

// Assembles chunks by hand. 'values0_', 'childOffsets_' and 'values1_' encode
// two BIGINT levels: values {1, 2}, where 1 owns child {5} and 2 owns children
// {6, 7}, so three distinct keys.
class HierarchicalKeyReaderChunkTest : public HierarchicalKeyReaderTest {
 protected:
  void SetUp() override {
    HierarchicalKeyReaderTest::SetUp();
    values0_ = encodeTrivial<uint64_t>({1, 2});
    childOffsets_ = encodeTrivial<uint32_t>({0, 1, 3});
    values1_ = encodeTrivial<uint64_t>({5, 6, 7});
  }

  // Encodes 'values' as one Trivial sequence.
  template <typename T>
  std::string encodeTrivial(const std::vector<T>& values) {
    return encode(values, EncodingType::Trivial);
  }

  // Encodes a Constant sequence of 'value' that claims 'rowCount' rows.
  template <typename T>
  std::string encodeConstant(T value, uint32_t rowCount) {
    auto data = encode(std::vector<T>{value}, EncodingType::Constant);
    setUint32(data, EncodingPrefix::kRowCountOffset, rowCount);
    return data;
  }

  // Encodes 'values' as one sequence of 'encodingType'.
  template <typename T>
  std::string encode(const std::vector<T>& values, EncodingType encodingType) {
    auto policy =
        ManualEncodingSelectionPolicyFactory{
            {{encodingType, 1.0}}, /*compressionOptions=*/std::nullopt}
            .createPolicy(TypeTraits<T>::dataType);
    Buffer buffer{*pool_};
    return std::string{EncodingFactory::encode<T>(
        std::unique_ptr<EncodingSelectionPolicy<T>>(
            static_cast<EncodingSelectionPolicy<T>*>(policy.release())),
        std::span<const T>{values},
        buffer)};
  }

  // Assembles a two-level BIGINT chunk with 'levelCounts' values per level from
  // encoded sequences.
  static std::string makeChunk(
      uint32_t rowCount,
      const std::vector<std::string>& sequences,
      std::array<uint32_t, 2> levelCounts = {2, 3}) {
    size_t size{
        HierarchicalKeyFormat::kHeaderSize +
        levelCounts.size() * (sizeof(uint8_t) + sizeof(uint32_t))};
    for (const auto& sequence : sequences) {
      size += sizeof(uint32_t) + sequence.size();
    }
    std::string data(size, '\0');
    char* position = data.data();
    encoding::write<uint8_t>(HierarchicalKeyFormat::kVersion, position);
    encoding::write<uint8_t>(levelCounts.size(), position);
    encoding::writeUint32(rowCount, position);
    for (size_t level = 0; level < levelCounts.size(); ++level) {
      encoding::write<uint8_t>(sizeof(uint64_t), position);
    }
    for (const auto count : levelCounts) {
      encoding::writeUint32(count, position);
    }
    for (const auto& sequence : sequences) {
      encoding::writeString(sequence, position);
    }
    return data;
  }

  std::string values0_;
  std::string childOffsets_;
  std::string values1_;
};

TEST_F(HierarchicalKeyReaderTest, roundTripsFourColumnKeys) {
  const std::vector<std::vector<uint64_t>> columns{
      {32'188, 32'188, 32'188, 32'188, 32'188, 32'188},
      {1, 1, 1, 3, 3, 8},
      {10, 10, 20, 4, 4, 9},
      {7, 9, 1, 8, 11, 2},
  };
  const auto index = build(columns);
  const auto expected = packRows(columns);

  EXPECT_EQ(index->rowCount(), expected.size());
  EXPECT_EQ(index->materialize(0, expected.size()), expected);
  for (uint32_t row = 0; row < expected.size(); ++row) {
    SCOPED_TRACE(row);
    EXPECT_EQ(index->get(row), expected[row]);
    EXPECT_EQ(index->seek(expected[row], /*inclusive=*/true), row);
    // Past the last row there is nothing to return, which seek reports as
    // nullopt rather than rowCount.
    const auto next = index->seek(expected[row], /*inclusive=*/false);
    if (row + 1 == expected.size()) {
      EXPECT_EQ(next, std::nullopt);
    } else {
      EXPECT_EQ(next, row + 1);
    }
  }
}

// The unicorn/Figtable shape: one integer column whose values repeat, so each
// distinct key owns a run of rows rather than a single row.
TEST_F(HierarchicalKeyReaderTest, bracketsDuplicateRunsForSingleColumnKey) {
  const std::vector<std::vector<uint64_t>> columns{
      {10, 10, 10, 20, 30, 30},
  };
  const auto index = build(columns);
  EXPECT_EQ(index->numLevels(), 1);
  EXPECT_EQ(index->rowCount(), 6);
  EXPECT_EQ(index->materialize(0, 6), packRows(columns));

  // seek(inclusive) lands on the run's first row; seek(exclusive) lands past
  // its last, so the pair brackets the whole run.
  const auto ten = makeKey({10});
  EXPECT_EQ(index->seek(ten, /*inclusive=*/true), 0);
  EXPECT_EQ(index->seek(ten, /*inclusive=*/false), 3);

  const auto thirty = makeKey({30});
  EXPECT_EQ(index->seek(thirty, /*inclusive=*/true), 4);
  EXPECT_EQ(index->seek(thirty, /*inclusive=*/false), std::nullopt);

  // A key between two runs resolves to the start of the following run.
  const auto fifteen = makeKey({15});
  EXPECT_EQ(index->seek(fifteen, /*inclusive=*/true), 3);
  EXPECT_EQ(index->seek(fifteen, /*inclusive=*/false), 3);

  EXPECT_EQ(index->seek(makeKey({5}), /*inclusive=*/true), 0);
  EXPECT_EQ(index->seek(makeKey({40}), /*inclusive=*/true), std::nullopt);
}

TEST_F(HierarchicalKeyReaderTest, duplicatesAcrossInnerColumns) {
  const std::vector<std::vector<uint64_t>> columns{
      {7, 7, 7, 9},
      {1, 1, 4, 2},
  };
  const auto index = build(columns);
  EXPECT_EQ(index->materialize(0, 4), packRows(columns));

  const auto sevenOne = makeKey({7, 1});
  EXPECT_EQ(index->seek(sevenOne, /*inclusive=*/true), 0);
  EXPECT_EQ(index->seek(sevenOne, /*inclusive=*/false), 2);
}

TEST_F(HierarchicalKeyReaderTest, cursorWalksRowsFromStart) {
  const std::vector<std::vector<uint64_t>> columns{
      {7, 7, 7, 9},
      {1, 1, 4, 2},
  };
  const auto index = build(columns);
  const auto expected = packRows(columns);

  for (uint32_t startRow = 0; startRow < expected.size(); ++startRow) {
    SCOPED_TRACE(startRow);
    auto cursor = index->cursor(startRow);
    std::vector<std::string> walked;
    while (cursor->hasNext()) {
      walked.emplace_back(cursor->next());
    }
    EXPECT_EQ(
        walked,
        std::vector<std::string>(expected.begin() + startRow, expected.end()));
  }
}

TEST_F(HierarchicalKeyReaderTest, readsLongRunsAndRepeatedKeys) {
  // Two parents with long ascending runs of children. Every fourth child
  // repeats, so the leaf row offsets are stored.
  std::vector<std::vector<uint64_t>> columns(2);
  for (const uint64_t parent : {3, 8}) {
    for (uint64_t child = 0; child < 300; ++child) {
      const uint32_t numRows = child % 4 == 0 ? 2 : 1;
      for (uint32_t row = 0; row < numRows; ++row) {
        columns[0].push_back(parent);
        columns[1].push_back(child * 10 + parent);
      }
    }
  }
  const auto index = build(columns);
  const auto expected = packRows(columns);
  checkReads(*index, expected);

  // Keys between and around stored keys resolve to the next stored key.
  for (const auto& probe :
       {makeKey({0, 0}),
        makeKey({3, 5}),
        makeKey({5, 0}),
        makeKey({8, 3'000}),
        makeKey({9, 0})}) {
    EXPECT_EQ(
        index->seek(probe, /*inclusive=*/true),
        expectedSeek(expected, probe, /*inclusive=*/true));
  }
}

TEST_F(HierarchicalKeyReaderTest, readsEveryWidthAndSortOrder) {
  checkTypedKeys<int8_t>();
  checkTypedKeys<int16_t>();
  checkTypedKeys<int32_t>();
  checkTypedKeys<int64_t>();
}

TEST_F(HierarchicalKeyReaderTest, seeksPartitionedEliasFanoSequences) {
  // Level 1 restarts below its previous value at each of three parents, so its
  // sequence is one long ascending run per parent.
  constexpr uint64_t kRowsPerParent{100};
  std::vector<std::vector<uint64_t>> columns(2);
  for (const uint64_t parent : {1, 4, 9}) {
    for (uint64_t row = 0; row < kRowsPerParent; ++row) {
      columns[0].push_back(parent);
      columns[1].push_back(parent * 7 + row * parent);
    }
  }
  const auto index = build(columns);
  ASSERT_EQ(
      index->testingValuesEncodingType(1), EncodingType::PartitionedEliasFano);
  const auto expected = packRows(columns);
  checkReads(*index, expected);
  for (const auto& probe :
       {makeKey({0, 0}),
        makeKey({1, 8}),
        makeKey({1, 1'000}),
        makeKey({4, 0}),
        makeKey({4, 30}),
        makeKey({9, 64}),
        makeKey({9, 10'000})}) {
    for (const bool inclusive : {true, false}) {
      EXPECT_EQ(
          index->seek(probe, inclusive),
          expectedSeek(expected, probe, inclusive));
    }
  }
}

TEST_F(HierarchicalKeyReaderTest, picksPartitionedEliasFanoForLongRuns) {
  // The same number of ascending level-1 values, first in a few long runs and
  // then in many runs too short to partition.
  constexpr uint64_t kNumRows{600};
  for (const uint64_t rowsPerParent : {200, 4}) {
    SCOPED_TRACE(rowsPerParent);
    std::vector<std::vector<uint64_t>> columns(2);
    for (uint64_t row = 0; row < kNumRows; ++row) {
      columns[0].push_back(row / rowsPerParent);
      columns[1].push_back(1'000 + (row % rowsPerParent) * 37);
    }
    const auto index = build(columns);
    EXPECT_EQ(
        index->testingValuesEncodingType(1) ==
            EncodingType::PartitionedEliasFano,
        rowsPerParent == 200);
    EXPECT_EQ(index->materialize(0, kNumRows), packRows(columns));
  }
}

TEST_F(HierarchicalKeyReaderTest, rejectsMalformedInput) {
  EXPECT_THROW(build({{3, 1}}), NimbleUserError);

  std::vector<std::vector<uint64_t>> tooManyColumns(256, {1});
  NIMBLE_ASSERT_THROW(
      build(tooManyColumns),
      "Hierarchical key chunks support at most 255 columns");

  const auto index = build({{1}});
  NIMBLE_ASSERT_THROW(
      index->seek("", /*inclusive=*/true),
      "HierarchicalKeyReader requires a complete encoded key");
  auto nullKey = makeKey({1});
  nullKey.at(0) = 1;
  NIMBLE_ASSERT_THROW(
      index->seek(nullKey, /*inclusive=*/true),
      "HierarchicalKeyReader requires non-null encoded key components");

  for (const uint8_t version : {uint8_t{0}, uint8_t{2}}) {
    SCOPED_TRACE(static_cast<uint32_t>(version));
    std::string unsupportedVersion{built_};
    unsupportedVersion.at(0) = static_cast<char>(version);
    NIMBLE_ASSERT_FILE_THROW(
        HierarchicalKeyReader(unsupportedVersion, pool_.get()),
        "Unsupported HierarchicalKeyReader format version");
  }
}

TEST_F(HierarchicalKeyReaderTest, rejectsInvalidFormatMetadata) {
  build({{1, 2}, {3, 4}});
  const std::string validPayload{built_};
  constexpr size_t kLevelCountsOffset{
      HierarchicalKeyFormat::kHeaderSize + 2 * sizeof(uint8_t)};

  {
    SCOPED_TRACE("truncated header");
    NIMBLE_ASSERT_FILE_THROW(
        HierarchicalKeyReader(
            std::string_view{validPayload}.substr(
                0, HierarchicalKeyFormat::kHeaderSize - 1),
            pool_.get()),
        "Truncated HierarchicalKeyReader metadata");
  }
  {
    SCOPED_TRACE("zero levels");
    auto malformed = validPayload;
    malformed.at(1) = 0;
    NIMBLE_ASSERT_FILE_THROW(
        HierarchicalKeyReader(malformed, pool_.get()),
        "HierarchicalKeyReader needs a level");
  }
  {
    SCOPED_TRACE("zero rows");
    auto malformed = validPayload;
    setUint32(malformed, HierarchicalKeyFormat::kRowCountOffset, 0);
    NIMBLE_ASSERT_FILE_THROW(
        HierarchicalKeyReader(malformed, pool_.get()),
        "HierarchicalKeyReader has no rows");
  }
  {
    SCOPED_TRACE("unsupported key value width");
    auto malformed = validPayload;
    malformed.at(HierarchicalKeyFormat::kHeaderSize) = 3;
    NIMBLE_ASSERT_FILE_THROW(
        HierarchicalKeyReader(malformed, pool_.get()),
        "Unsupported HierarchicalKeyReader key value width");
  }
  {
    SCOPED_TRACE("empty root level");
    auto malformed = validPayload;
    setUint32(malformed, kLevelCountsOffset, 0);
    NIMBLE_ASSERT_FILE_THROW(
        HierarchicalKeyReader(malformed, pool_.get()),
        "HierarchicalKeyReader root level has no values");
  }
  {
    SCOPED_TRACE("decreasing level counts");
    auto malformed = validPayload;
    setUint32(malformed, kLevelCountsOffset + sizeof(uint32_t), 1);
    NIMBLE_ASSERT_FILE_THROW(
        HierarchicalKeyReader(malformed, pool_.get()),
        "HierarchicalKeyReader level value counts must be non-decreasing");
  }
  {
    SCOPED_TRACE("trailing bytes");
    auto malformed = validPayload;
    malformed.push_back('\0');
    NIMBLE_ASSERT_FILE_THROW(
        HierarchicalKeyReader(malformed, pool_.get()),
        "HierarchicalKeyReader contains trailing data");
  }
}

// Nested encoding payload validation belongs to each encoding's tests. This
// mutates only the hierarchy metadata that HierarchicalKeyReader owns.
TEST_F(HierarchicalKeyReaderChunkTest, readsAssembledChunk) {
  EXPECT_EQ(
      HierarchicalKeyReader(
          makeChunk(3, {values0_, childOffsets_, values1_}), pool_.get())
          .materialize(0, 3),
      (std::vector<std::string>{
          makeKey({1, 5}), makeKey({2, 6}), makeKey({2, 7})}));
}

TEST_F(HierarchicalKeyReaderChunkTest, rejectsInvalidChildOffsets) {
  for (const auto& [name, offsets, message] : {
           std::tuple{
               "nonzero start",
               std::vector<uint32_t>{1, 2, 3},
               "offsets must start at zero: child offsets."},
           std::tuple{
               "empty range",
               std::vector<uint32_t>{0, 3, 3},
               "offsets contain an empty range: child offsets."},
           std::tuple{
               "wrong end",
               std::vector<uint32_t>{0, 1, 2},
               "invalid final offset: child offsets."},
       }) {
    SCOPED_TRACE(name);
    NIMBLE_ASSERT_FILE_THROW(
        HierarchicalKeyReader(
            makeChunk(
                3, {values0_, encodeTrivial<uint32_t>(offsets), values1_}),
            pool_.get()),
        message);
  }
}

TEST_F(HierarchicalKeyReaderChunkTest, rejectsCompactOffsetsWithHugeRowCount) {
  // A few bytes of Constant child offsets claim a billion rows. Validation
  // rejects the second offset without decoding the rest.
  constexpr uint32_t kNumValues{1'000'000'000};
  NIMBLE_ASSERT_FILE_THROW(
      HierarchicalKeyReader(
          makeChunk(
              kNumValues,
              {encodeConstant<uint64_t>(1, kNumValues),
               encodeConstant<uint32_t>(0, kNumValues + 1),
               encodeConstant<uint64_t>(5, kNumValues)},
              {kNumValues, kNumValues}),
          pool_.get()),
      "offsets contain an empty range: child offsets.");
  EXPECT_LT(pool_->peakBytes(), 16 << 20);
}

TEST_F(HierarchicalKeyReaderChunkTest, rejectsInvalidLeafRowOffsets) {
  // Four rows over three distinct keys.
  for (const auto& [name, offsets, message] : {
           std::tuple{
               "nonzero start",
               std::vector<uint32_t>{1, 2, 3, 4},
               "offsets must start at zero: leaf row offsets."},
           std::tuple{
               "empty range",
               std::vector<uint32_t>{0, 2, 2, 4},
               "offsets contain an empty range: leaf row offsets."},
           std::tuple{
               "wrong end",
               std::vector<uint32_t>{0, 1, 2, 3},
               "invalid final offset: leaf row offsets."},
       }) {
    SCOPED_TRACE(name);
    NIMBLE_ASSERT_FILE_THROW(
        HierarchicalKeyReader(
            makeChunk(
                4,
                {values0_,
                 childOffsets_,
                 values1_,
                 encodeTrivial<uint32_t>(offsets)}),
            pool_.get()),
        message);
  }
}

TEST_F(HierarchicalKeyReaderChunkTest, rejectsMissingLeafRowOffsets) {
  // Four rows over three distinct keys need leaf row offsets.
  NIMBLE_ASSERT_FILE_THROW(
      HierarchicalKeyReader(
          makeChunk(4, {values0_, childOffsets_, values1_}), pool_.get()),
      "Truncated HierarchicalKeyReader metadata.");
}

TEST_F(HierarchicalKeyReaderChunkTest, rejectsWrongPhysicalType) {
  NIMBLE_ASSERT_FILE_THROW(
      HierarchicalKeyReader(
          makeChunk(
              3, {values0_, childOffsets_, encodeTrivial<uint32_t>({5, 6, 7})}),
          pool_.get()),
      "HierarchicalKeyReader child physical type does not match its level.");
}

TEST_F(HierarchicalKeyReaderChunkTest, rejectsWrongRowCount) {
  NIMBLE_ASSERT_FILE_THROW(
      HierarchicalKeyReader(
          makeChunk(
              3,
              {values0_, childOffsets_, encodeTrivial<uint64_t>({5, 6, 7, 8})}),
          pool_.get()),
      "HierarchicalKeyReader child row count does not match its directory.");
}

TEST_F(HierarchicalKeyReaderTest, mutatedMetadataPreservesReaderContract) {
  std::vector<uint64_t> values(64);
  for (uint32_t row = 0; row < values.size(); ++row) {
    values[row] = row * 3;
  }
  build({values});
  const std::string validPayload{built_};
  constexpr size_t kSequenceSizeOffset{
      HierarchicalKeyFormat::kHeaderSize + sizeof(uint8_t) + sizeof(uint32_t)};
  constexpr size_t kSequenceDataOffset{kSequenceSizeOffset + sizeof(uint32_t)};
  std::mt19937 random{42};
  uint32_t numAccepted{0};
  uint32_t numRejected{0};

  for (uint32_t iteration = 0; iteration < 64; ++iteration) {
    SCOPED_TRACE(iteration);
    auto mutated = validPayload;
    if (iteration > 0 && iteration % 4 == 0) {
      mutated.resize(random() % (kSequenceDataOffset + 1));
    } else if (iteration > 0) {
      const auto numMutations = 1 + random() % 4;
      for (uint32_t mutation = 0; mutation < numMutations; ++mutation) {
        const auto offset = HierarchicalKeyFormat::kHeaderSize +
            random() %
                (kSequenceSizeOffset - HierarchicalKeyFormat::kHeaderSize);
        const auto mask = static_cast<uint8_t>(1 + random() % 255);
        mutated.at(offset) =
            static_cast<char>(static_cast<uint8_t>(mutated.at(offset)) ^ mask);
      }
    }

    try {
      HierarchicalKeyReader reader{mutated, pool_.get()};
      const auto materialized = reader.materialize(0, reader.rowCount());
      std::vector<std::string> cursorValues;
      auto cursor = reader.cursor(0);
      while (cursor->hasNext()) {
        cursorValues.emplace_back(cursor->next());
      }
      EXPECT_EQ(cursorValues, materialized);
      ++numAccepted;
    } catch (const NimbleException&) {
      ++numRejected;
    }
  }

  EXPECT_GT(numAccepted, 0);
  EXPECT_GT(numRejected, 0);
}

TEST_F(HierarchicalKeyReaderTest, rejectsTruncatedMetadataBeforeAllocation) {
  build({{1, 2}, {3, 4}});
  std::string malformed{built_};
  constexpr size_t kLevelCountsOffset{
      HierarchicalKeyFormat::kHeaderSize + 2 * sizeof(uint8_t)};
  constexpr uint32_t kOversizedCount{1'000'000'000};
  std::memcpy(
      malformed.data() + HierarchicalKeyFormat::kRowCountOffset,
      &kOversizedCount,
      sizeof(kOversizedCount));
  std::memcpy(
      malformed.data() + kLevelCountsOffset,
      &kOversizedCount,
      sizeof(kOversizedCount));
  std::memcpy(
      malformed.data() + kLevelCountsOffset + sizeof(uint32_t),
      &kOversizedCount,
      sizeof(kOversizedCount));

  NIMBLE_ASSERT_FILE_THROW(
      HierarchicalKeyReader(malformed, pool_.get()),
      "HierarchicalKeyReader child row count does not match its directory");
}

TEST_F(HierarchicalKeyReaderTest, rejectsEmptySequence) {
  build({{1}});
  std::string malformed{built_};
  constexpr size_t kSequenceSizeOffset{
      HierarchicalKeyFormat::kHeaderSize + sizeof(uint8_t) + sizeof(uint32_t)};
  const uint32_t emptySequenceSize{0};
  std::memcpy(
      malformed.data() + kSequenceSizeOffset,
      &emptySequenceSize,
      sizeof(emptySequenceSize));

  NIMBLE_ASSERT_THROW(
      HierarchicalKeyReader(malformed, pool_.get()),
      "Truncated HierarchicalKeyReader sequence");
}

TEST_F(HierarchicalKeyReaderTest, rejectsLevelValueCountOverflow) {
  build({{1, 2}, {3, 4}});
  std::string malformed{built_};
  constexpr size_t kLevelCountsOffset{
      HierarchicalKeyFormat::kHeaderSize + 2 * sizeof(uint8_t)};
  constexpr auto kMaxCount = std::numeric_limits<uint32_t>::max();
  setUint32(malformed, HierarchicalKeyFormat::kRowCountOffset, kMaxCount);
  setUint32(malformed, kLevelCountsOffset + sizeof(uint32_t), kMaxCount);

  NIMBLE_ASSERT_FILE_THROW(
      HierarchicalKeyReader(malformed, pool_.get()),
      "HierarchicalKeyReader level value count is too large.");
}

} // namespace
} // namespace facebook::nimble::index
