/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/connectors/hive/iceberg/DeletionVectorReader.h"

#include <fstream>

#include <gtest/gtest.h>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempFilePath.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive::iceberg;
using namespace facebook::velox::common::testutil;

namespace {

/// Serializes a roaring bitmap in the portable format (no-run variant,
/// cookie = 12346). Supports only array containers (cardinality <= 4096).
/// This is the simplest format the DeletionVectorReader needs to parse.
std::string serializeRoaringBitmapNoRun(const std::vector<int64_t>& positions) {
  if (positions.empty()) {
    // Empty bitmap: cookie + 0 containers.
    std::string data(8, '\0');
    uint32_t cookie = 12346;
    uint32_t numContainers = 0;
    std::memcpy(data.data(), &cookie, 4);
    std::memcpy(data.data() + 4, &numContainers, 4);
    return data;
  }

  // Group positions by high 16 bits.
  std::map<uint16_t, std::vector<uint16_t>> containers;
  for (auto pos : positions) {
    auto key = static_cast<uint16_t>(pos >> 16);
    auto low = static_cast<uint16_t>(pos & 0xFFFF);
    containers[key].push_back(low);
  }

  for (auto& [key, vals] : containers) {
    std::sort(vals.begin(), vals.end());
  }

  uint32_t numContainers = static_cast<uint32_t>(containers.size());

  std::string data;
  // Cookie.
  uint32_t cookie = 12346;
  data.append(reinterpret_cast<const char*>(&cookie), 4);
  // Container count.
  data.append(reinterpret_cast<const char*>(&numContainers), 4);

  // Key-cardinality pairs.
  for (auto& [key, vals] : containers) {
    uint16_t cardMinus1 = static_cast<uint16_t>(vals.size() - 1);
    data.append(reinterpret_cast<const char*>(&key), 2);
    data.append(reinterpret_cast<const char*>(&cardMinus1), 2);
  }

  // Offset section (required for >= 4 containers).
  if (numContainers >= 4) {
    uint32_t offset = 4 + 4 + numContainers * 4 + numContainers * 4;
    for (auto& [key, vals] : containers) {
      data.append(reinterpret_cast<const char*>(&offset), 4);
      offset += static_cast<uint32_t>(vals.size()) * 2;
    }
  }

  // Container data (array containers: sorted uint16 values).
  for (auto& [key, vals] : containers) {
    for (auto v : vals) {
      data.append(reinterpret_cast<const char*>(&v), 2);
    }
  }

  return data;
}

/// Serializes a roaring bitmap in the portable format with run containers
/// (cookie = 12347). All containers are run-encoded.
std::string serializeRoaringBitmapWithRuns(
    const std::vector<
        std::pair<uint16_t, std::vector<std::pair<uint16_t, uint16_t>>>>&
        containerRuns) {
  // containerRuns: vector of (highBitsKey, vector of (start, lengthMinus1)).
  uint32_t numContainers = static_cast<uint32_t>(containerRuns.size());

  // Cookie: low 16 bits = 12347, high 16 bits = numContainers - 1.
  uint32_t cookie = static_cast<uint32_t>(12347) | ((numContainers - 1) << 16);

  std::string data;
  data.append(reinterpret_cast<const char*>(&cookie), 4);

  // Run bitmap: all containers are run containers. ceil(numContainers / 8)
  // bytes.
  uint32_t runBitmapBytes = (numContainers + 7) / 8;
  std::vector<uint8_t> runBitmap(runBitmapBytes, 0xFF);
  data.append(reinterpret_cast<const char*>(runBitmap.data()), runBitmapBytes);

  // Compute cardinality for each container.
  std::vector<uint32_t> cardinalities;
  for (auto& [key, runs] : containerRuns) {
    uint32_t card = 0;
    for (auto& [start, lenMinus1] : runs) {
      card += static_cast<uint32_t>(lenMinus1) + 1;
    }
    cardinalities.push_back(card);
  }

  // Key-cardinality pairs.
  for (size_t i = 0; i < containerRuns.size(); ++i) {
    uint16_t key = containerRuns[i].first;
    uint16_t cardMinus1 = static_cast<uint16_t>(cardinalities[i] - 1);
    data.append(reinterpret_cast<const char*>(&key), 2);
    data.append(reinterpret_cast<const char*>(&cardMinus1), 2);
  }

  // Container data: each run container has numRuns (uint16) followed by
  // (start, lengthMinus1) pairs.
  for (auto& [key, runs] : containerRuns) {
    uint16_t numRuns = static_cast<uint16_t>(runs.size());
    data.append(reinterpret_cast<const char*>(&numRuns), 2);
    for (auto& [start, lenMinus1] : runs) {
      data.append(reinterpret_cast<const char*>(&start), 2);
      data.append(reinterpret_cast<const char*>(&lenMinus1), 2);
    }
  }

  return data;
}

/// Writes binary data to a temp file and returns the path.
std::shared_ptr<TempFilePath> writeDvFile(const std::string& bitmapData) {
  auto tempFile = TempFilePath::create();
  // Write directly via C++ streams since TempFilePath already creates the
  // file and the local filesystem openFileForWrite may not overwrite.
  std::ofstream out(tempFile->getPath(), std::ios::binary | std::ios::trunc);
  VELOX_CHECK(out.good(), "Failed to open temp file for writing");
  out.write(bitmapData.data(), static_cast<std::streamsize>(bitmapData.size()));
  out.close();
  return tempFile;
}

/// Creates an IcebergDeleteFile for a deletion vector.
IcebergDeleteFile makeDvDeleteFile(
    const std::string& filePath,
    uint64_t recordCount,
    uint64_t fileSize,
    uint64_t blobOffset = 0,
    std::optional<uint64_t> blobLength = std::nullopt) {
  std::unordered_map<int32_t, std::string> lowerBounds;
  std::unordered_map<int32_t, std::string> upperBounds;

  lowerBounds[DeletionVectorReader::kDvOffsetFieldId] =
      std::to_string(blobOffset);
  if (blobLength.has_value()) {
    upperBounds[DeletionVectorReader::kDvLengthFieldId] =
        std::to_string(blobLength.value());
  } else {
    upperBounds[DeletionVectorReader::kDvLengthFieldId] =
        std::to_string(fileSize);
  }

  return IcebergDeleteFile(
      FileContent::kDeletionVector,
      filePath,
      dwio::common::FileFormat::DWRF,
      recordCount,
      fileSize,
      {},
      lowerBounds,
      upperBounds);
}

/// Extracts which bits are set in a bitmap buffer.
std::vector<uint64_t> getSetBits(const BufferPtr& bitmap, uint64_t size) {
  auto* raw = bitmap->as<uint8_t>();
  std::vector<uint64_t> result;
  for (uint64_t i = 0; i < size; ++i) {
    if (bits::isBitSet(raw, i)) {
      result.push_back(i);
    }
  }
  return result;
}

} // namespace

class DeletionVectorReaderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    filesystems::registerLocalFileSystem();
    pool_ = memory::memoryManager()->addLeafPool("DeletionVectorReaderTest");
  }

  BufferPtr allocateBitmap(uint64_t numBits) {
    auto numBytes = bits::nbytes(numBits);
    auto buffer = AlignedBuffer::allocate<uint8_t>(numBytes, pool_.get(), 0);
    return buffer;
  }

  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(DeletionVectorReaderTest, basicArrayContainer) {
  // Positions: 0, 5, 10, 99.
  std::vector<int64_t> positions = {0, 5, 10, 99};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile =
      makeDvDeleteFile(tempFile->getPath(), positions.size(), fileSize);

  DeletionVectorReader reader(dvFile, 0, pool_.get());
  EXPECT_FALSE(reader.noMoreData());

  auto bitmap = allocateBitmap(100);
  reader.readDeletePositions(0, 100, bitmap);

  auto setBits = getSetBits(bitmap, 100);
  EXPECT_EQ(setBits, (std::vector<uint64_t>{0, 5, 10, 99}));
  EXPECT_TRUE(reader.noMoreData());
}

TEST_F(DeletionVectorReaderTest, batchRangeFiltering) {
  // Positions: 10, 20, 30, 40, 50.
  std::vector<int64_t> positions = {10, 20, 30, 40, 50};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile =
      makeDvDeleteFile(tempFile->getPath(), positions.size(), fileSize);

  DeletionVectorReader reader(dvFile, 0, pool_.get());

  // First batch: rows 0-24 (should contain positions 10, 20).
  auto bitmap1 = allocateBitmap(25);
  reader.readDeletePositions(0, 25, bitmap1);
  auto bits1 = getSetBits(bitmap1, 25);
  EXPECT_EQ(bits1, (std::vector<uint64_t>{10, 20}));
  EXPECT_FALSE(reader.noMoreData());

  // Second batch: rows 25-49 (should contain positions 30, 40).
  auto bitmap2 = allocateBitmap(25);
  reader.readDeletePositions(25, 25, bitmap2);
  auto bits2 = getSetBits(bitmap2, 25);
  EXPECT_EQ(bits2, (std::vector<uint64_t>{5, 15}));
  EXPECT_FALSE(reader.noMoreData());

  // Third batch: rows 50-74 (should contain position 50).
  auto bitmap3 = allocateBitmap(25);
  reader.readDeletePositions(50, 25, bitmap3);
  auto bits3 = getSetBits(bitmap3, 25);
  EXPECT_EQ(bits3, (std::vector<uint64_t>{0}));
  EXPECT_TRUE(reader.noMoreData());
}

TEST_F(DeletionVectorReaderTest, splitOffset) {
  // Positions: 100, 105, 110.
  std::vector<int64_t> positions = {100, 105, 110};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile =
      makeDvDeleteFile(tempFile->getPath(), positions.size(), fileSize);

  // Split starts at row 100.
  DeletionVectorReader reader(dvFile, 100, pool_.get());

  auto bitmap = allocateBitmap(20);
  reader.readDeletePositions(0, 20, bitmap);

  // Positions 100, 105, 110 relative to splitOffset=100, baseReadOffset=0
  // become bit indices 0, 5, 10.
  auto setBits = getSetBits(bitmap, 20);
  EXPECT_EQ(setBits, (std::vector<uint64_t>{0, 5, 10}));
  EXPECT_TRUE(reader.noMoreData());
}

TEST_F(DeletionVectorReaderTest, splitOffsetWithBaseReadOffset) {
  // Positions: 200, 210, 220.
  std::vector<int64_t> positions = {200, 210, 220};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile =
      makeDvDeleteFile(tempFile->getPath(), positions.size(), fileSize);

  // Split starts at row 100.
  DeletionVectorReader reader(dvFile, 100, pool_.get());

  // First batch: baseReadOffset=100, so file positions [200, 300).
  // Positions 200, 210, 220 are all in range.
  auto bitmap = allocateBitmap(100);
  reader.readDeletePositions(100, 100, bitmap);

  auto setBits = getSetBits(bitmap, 100);
  EXPECT_EQ(setBits, (std::vector<uint64_t>{0, 10, 20}));
  EXPECT_TRUE(reader.noMoreData());
}

TEST_F(DeletionVectorReaderTest, noDeletesInRange) {
  // Positions: 1000, 2000.
  std::vector<int64_t> positions = {1000, 2000};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile =
      makeDvDeleteFile(tempFile->getPath(), positions.size(), fileSize);

  DeletionVectorReader reader(dvFile, 0, pool_.get());

  // Batch covers rows 0-99, no deletions in this range.
  auto bitmap = allocateBitmap(100);
  reader.readDeletePositions(0, 100, bitmap);

  auto setBits = getSetBits(bitmap, 100);
  EXPECT_TRUE(setBits.empty());
  EXPECT_FALSE(reader.noMoreData());
}

TEST_F(DeletionVectorReaderTest, runContainers) {
  // Use run-encoded containers: positions 10-19 and 50-59.
  std::vector<std::pair<uint16_t, std::vector<std::pair<uint16_t, uint16_t>>>>
      containerRuns = {
          {0, {{10, 9}, {50, 9}}},
      };
  auto bitmapData = serializeRoaringBitmapWithRuns(containerRuns);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile = makeDvDeleteFile(tempFile->getPath(), 20, fileSize);

  DeletionVectorReader reader(dvFile, 0, pool_.get());

  auto bitmap = allocateBitmap(100);
  reader.readDeletePositions(0, 100, bitmap);

  auto setBits = getSetBits(bitmap, 100);
  // Expect positions 10-19 and 50-59.
  std::vector<uint64_t> expected;
  for (uint64_t i = 10; i <= 19; ++i) {
    expected.push_back(i);
  }
  for (uint64_t i = 50; i <= 59; ++i) {
    expected.push_back(i);
  }
  EXPECT_EQ(setBits, expected);
  EXPECT_TRUE(reader.noMoreData());
}

TEST_F(DeletionVectorReaderTest, largePositionsMultipleContainers) {
  // Positions spanning two containers: one in container 0 (key=0), one in
  // container 1 (key=1, i.e. pos >= 65536).
  std::vector<int64_t> positions = {5, 100, 65536, 65600};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile =
      makeDvDeleteFile(tempFile->getPath(), positions.size(), fileSize);

  DeletionVectorReader reader(dvFile, 0, pool_.get());

  // Read a batch covering all positions.
  auto bitmap = allocateBitmap(66000);
  reader.readDeletePositions(0, 66000, bitmap);

  auto setBits = getSetBits(bitmap, 66000);
  EXPECT_EQ(setBits, (std::vector<uint64_t>{5, 100, 65536, 65600}));
  EXPECT_TRUE(reader.noMoreData());
}

TEST_F(DeletionVectorReaderTest, blobOffset) {
  // Write a file with some padding before the actual bitmap data.
  std::vector<int64_t> positions = {3, 7, 11};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);

  // Prepend 64 bytes of padding.
  std::string padding(64, 'X');
  std::string fileContent = padding + bitmapData;

  auto tempFile = writeDvFile(fileContent);
  auto fileSize = static_cast<uint64_t>(fileContent.size());

  auto dvFile = makeDvDeleteFile(
      tempFile->getPath(), positions.size(), fileSize, 64, bitmapData.size());

  DeletionVectorReader reader(dvFile, 0, pool_.get());

  auto bitmap = allocateBitmap(20);
  reader.readDeletePositions(0, 20, bitmap);

  auto setBits = getSetBits(bitmap, 20);
  EXPECT_EQ(setBits, (std::vector<uint64_t>{3, 7, 11}));
  EXPECT_TRUE(reader.noMoreData());
}

TEST_F(DeletionVectorReaderTest, constructorRejectsWrongContentType) {
  auto tempFile = TempFilePath::create();
  {
    std::ofstream out(tempFile->getPath(), std::ios::binary | std::ios::trunc);
    out.write("dummy", 5);
  }

  IcebergDeleteFile badFile(
      FileContent::kPositionalDeletes,
      tempFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      5);

  VELOX_ASSERT_THROW(
      DeletionVectorReader(badFile, 0, pool_.get()),
      "Expected deletion vector file");
}

TEST_F(DeletionVectorReaderTest, constructorRejectsEmptyDv) {
  auto tempFile = TempFilePath::create();
  {
    std::ofstream out(tempFile->getPath(), std::ios::binary | std::ios::trunc);
    out.write("dummy", 5);
  }

  IcebergDeleteFile emptyDv(
      FileContent::kDeletionVector,
      tempFile->getPath(),
      dwio::common::FileFormat::DWRF,
      0,
      5);

  VELOX_ASSERT_THROW(
      DeletionVectorReader(emptyDv, 0, pool_.get()), "Empty deletion vector");
}

TEST_F(DeletionVectorReaderTest, noMoreDataAfterAllConsumed) {
  std::vector<int64_t> positions = {0, 1, 2};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile =
      makeDvDeleteFile(tempFile->getPath(), positions.size(), fileSize);

  DeletionVectorReader reader(dvFile, 0, pool_.get());
  EXPECT_FALSE(reader.noMoreData());

  auto bitmap = allocateBitmap(10);
  reader.readDeletePositions(0, 10, bitmap);
  EXPECT_TRUE(reader.noMoreData());

  // Additional reads should be no-ops.
  auto bitmap2 = allocateBitmap(10);
  reader.readDeletePositions(10, 10, bitmap2);
  auto setBits2 = getSetBits(bitmap2, 10);
  EXPECT_TRUE(setBits2.empty());
  EXPECT_TRUE(reader.noMoreData());
}

TEST_F(DeletionVectorReaderTest, singlePosition) {
  std::vector<int64_t> positions = {42};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile =
      makeDvDeleteFile(tempFile->getPath(), positions.size(), fileSize);

  DeletionVectorReader reader(dvFile, 0, pool_.get());

  auto bitmap = allocateBitmap(100);
  reader.readDeletePositions(0, 100, bitmap);

  auto setBits = getSetBits(bitmap, 100);
  EXPECT_EQ(setBits, (std::vector<uint64_t>{42}));
}

TEST_F(DeletionVectorReaderTest, consecutivePositions) {
  // Positions: 0 through 99 (100 consecutive positions).
  std::vector<int64_t> positions;
  positions.reserve(100);
  for (int64_t i = 0; i < 100; ++i) {
    positions.push_back(i);
  }
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile =
      makeDvDeleteFile(tempFile->getPath(), positions.size(), fileSize);

  DeletionVectorReader reader(dvFile, 0, pool_.get());

  auto bitmap = allocateBitmap(100);
  reader.readDeletePositions(0, 100, bitmap);

  auto setBits = getSetBits(bitmap, 100);
  std::vector<uint64_t> expected;
  expected.reserve(100);
  for (uint64_t i = 0; i < 100; ++i) {
    expected.push_back(i);
  }
  EXPECT_EQ(setBits, expected);
}

TEST_F(DeletionVectorReaderTest, invalidBitmapTooSmall) {
  // Write a file that is too small to contain a valid roaring bitmap header.
  std::string tinyData(4, '\0');
  auto tempFile = writeDvFile(tinyData);

  auto dvFile = makeDvDeleteFile(tempFile->getPath(), 1, tinyData.size());

  DeletionVectorReader reader(dvFile, 0, pool_.get());

  auto bitmap = allocateBitmap(10);
  VELOX_ASSERT_THROW(reader.readDeletePositions(0, 10, bitmap), "too small");
}

TEST_F(DeletionVectorReaderTest, invalidBitmapBadCookie) {
  // Write a file with an invalid cookie. Data must be large enough to pass
  // the minimum size check (8 bytes for 64-bit header) so that the cookie
  // validation is reached.
  std::string badData(64, '\0');
  uint32_t badCookie = 99999;
  std::memcpy(badData.data(), &badCookie, 4);
  auto tempFile = writeDvFile(badData);

  auto dvFile = makeDvDeleteFile(tempFile->getPath(), 1, badData.size());

  DeletionVectorReader reader(dvFile, 0, pool_.get());

  auto bitmap = allocateBitmap(10);
  VELOX_ASSERT_THROW(
      reader.readDeletePositions(0, 10, bitmap),
      "Unknown roaring bitmap cookie");
}
