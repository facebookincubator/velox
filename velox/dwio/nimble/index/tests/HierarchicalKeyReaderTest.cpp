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

#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/index/HierarchicalKeyFormat.h"
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

    const auto input = vectorMaker.rowVector(names, children);
    const auto config = ClusterIndexConfigBuilder{kHierarchicalClusterIndexName}
                            .withKeyColumns(names)
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
    built_ = {position, header.length};
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

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::string chunkData_;
  std::string_view built_;
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

TEST_F(HierarchicalKeyReaderTest, picksEncodingPerLevel) {
  // Each leaf parent has one child, so Constant wins for those sequences.
  const std::vector<std::vector<uint64_t>> columns{
      {1, 2, 3, 4},
      {10, 20, 30, 40},
  };
  const auto index = build(columns);
  EXPECT_EQ(index->numLevels(), 2);
  EXPECT_EQ(
      index->testingSequenceEncodingTypes(1),
      (std::vector<EncodingType>{
          EncodingType::Constant,
          EncodingType::Constant,
          EncodingType::Constant,
          EncodingType::Constant}));
}

TEST_F(HierarchicalKeyReaderTest, picksTrivialForFullWidthValues) {
  const auto index = build({{0, std::numeric_limits<uint64_t>::max()}});
  EXPECT_EQ(
      index->testingSequenceEncodingTypes(0),
      (std::vector<EncodingType>{EncodingType::Trivial}));
}

TEST_F(HierarchicalKeyReaderTest, selectsEncodingPerSequence) {
  const std::vector<std::vector<uint64_t>> columns{
      {1, 2, 2},
      {7, 0, std::numeric_limits<uint64_t>::max()},
  };
  const auto index = build(columns);
  EXPECT_EQ(index->materialize(0, 3), packRows(columns));
  EXPECT_EQ(
      index->testingSequenceEncodingTypes(1),
      (std::vector<EncodingType>{
          EncodingType::Constant, EncodingType::Trivial}));
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

  std::string unsupportedVersion{built_};
  unsupportedVersion.at(0) = 2;
  NIMBLE_ASSERT_THROW(
      HierarchicalKeyReader(unsupportedVersion, pool_.get()),
      "Unsupported HierarchicalKeyReader format version");
}

TEST_F(HierarchicalKeyReaderTest, rejectsInvalidFormatMetadata) {
  build({{1, 2}, {3, 4}});
  const std::string validPayload{built_};
  constexpr size_t kLevelCountsOffset{
      HierarchicalKeyFormat::kHeaderSize + 2 * sizeof(uint8_t)};

  {
    SCOPED_TRACE("truncated header");
    NIMBLE_ASSERT_THROW(
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
    NIMBLE_ASSERT_THROW(
        HierarchicalKeyReader(malformed, pool_.get()),
        "HierarchicalKeyReader needs a level");
  }
  {
    SCOPED_TRACE("zero rows");
    auto malformed = validPayload;
    setUint32(malformed, HierarchicalKeyFormat::kRowCountOffset, 0);
    NIMBLE_ASSERT_THROW(
        HierarchicalKeyReader(malformed, pool_.get()),
        "HierarchicalKeyReader has no rows");
  }
  {
    SCOPED_TRACE("unsupported key value width");
    auto malformed = validPayload;
    malformed.at(HierarchicalKeyFormat::kHeaderSize) = 3;
    NIMBLE_ASSERT_THROW(
        HierarchicalKeyReader(malformed, pool_.get()),
        "Unsupported HierarchicalKeyReader key value width");
  }
  {
    SCOPED_TRACE("empty root level");
    auto malformed = validPayload;
    setUint32(malformed, kLevelCountsOffset, 0);
    NIMBLE_ASSERT_THROW(
        HierarchicalKeyReader(malformed, pool_.get()),
        "HierarchicalKeyReader root level has no values");
  }
  {
    SCOPED_TRACE("decreasing level counts");
    auto malformed = validPayload;
    setUint32(malformed, kLevelCountsOffset + sizeof(uint32_t), 1);
    NIMBLE_ASSERT_THROW(
        HierarchicalKeyReader(malformed, pool_.get()),
        "HierarchicalKeyReader level value counts must be non-decreasing");
  }
  {
    SCOPED_TRACE("trailing bytes");
    auto malformed = validPayload;
    malformed.push_back('\0');
    NIMBLE_ASSERT_THROW(
        HierarchicalKeyReader(malformed, pool_.get()),
        "HierarchicalKeyReader contains trailing data");
  }
}

TEST_F(HierarchicalKeyReaderTest, rejectsInvalidChildOffsets) {
  build({{1, 2}, {3, 4}});
  const std::string validPayload{built_};
  constexpr size_t kChildOffsetsOffset{
      HierarchicalKeyFormat::kHeaderSize + 2 * sizeof(uint8_t) +
      2 * sizeof(uint32_t)};

  for (const auto& [name, index, value, message] : {
           std::tuple{"nonzero start", 0, 1, "offsets must start at zero"},
           std::tuple{"empty range", 1, 0, "offsets contain an empty range"},
           std::tuple{"wrong end", 2, 1, "invalid final offset"},
       }) {
    SCOPED_TRACE(name);
    auto malformed = validPayload;
    setUint32(malformed, kChildOffsetsOffset + index * sizeof(uint32_t), value);
    NIMBLE_ASSERT_THROW(HierarchicalKeyReader(malformed, pool_.get()), message);
  }
}

TEST_F(HierarchicalKeyReaderTest, rejectsInvalidLeafRowOffsets) {
  build({{10, 10, 20}});
  const std::string validPayload{built_};
  constexpr size_t kLeafOffsetsOffset{
      HierarchicalKeyFormat::kHeaderSize + sizeof(uint8_t) + sizeof(uint32_t)};

  for (const auto& [name, index, value, message] : {
           std::tuple{"nonzero start", 0, 1, "offsets must start at zero"},
           std::tuple{"empty range", 1, 0, "offsets contain an empty range"},
           std::tuple{"wrong end", 2, 2, "invalid final offset"},
       }) {
    SCOPED_TRACE(name);
    auto malformed = validPayload;
    setUint32(malformed, kLeafOffsetsOffset + index * sizeof(uint32_t), value);
    NIMBLE_ASSERT_THROW(HierarchicalKeyReader(malformed, pool_.get()), message);
  }
}

// Nested encoding payload validation belongs to each encoding's tests. This
// mutates only the hierarchy metadata that HierarchicalKeyReader owns.
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

  NIMBLE_ASSERT_THROW(
      HierarchicalKeyReader(malformed, pool_.get()),
      "Truncated HierarchicalKeyReader metadata");
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

} // namespace
} // namespace facebook::nimble::index
