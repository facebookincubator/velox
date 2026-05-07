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

#include "velox/connectors/hive/iceberg/DeletionVectorWriter.h"

#include <fstream>

#include <gtest/gtest.h>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/hive/iceberg/DeletionVectorReader.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive::iceberg;
using namespace facebook::velox::common::testutil;

namespace {

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

class DeletionVectorWriterTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    filesystems::registerLocalFileSystem();
    pool_ = memory::memoryManager()->addLeafPool("DeletionVectorWriterTest");
  }

  BufferPtr allocateBitmap(uint64_t numBits) {
    auto numBytes = bits::nbytes(numBits);
    return AlignedBuffer::allocate<uint8_t>(numBytes, pool_.get(), 0);
  }

  /// Writes serialized bitmap to a temp file, reads it back with
  /// DeletionVectorReader, and verifies the positions match.
  void verifyRoundTrip(
      const std::vector<int64_t>& positions,
      uint64_t batchSize) {
    DeletionVectorWriter writer;
    writer.addDeletedPositions(positions);
    EXPECT_EQ(writer.numPositions(), positions.size());

    auto blobData = writer.serialize();

    auto tempFile = TempFilePath::create();
    {
      std::ofstream out(
          tempFile->getPath(), std::ios::binary | std::ios::trunc);
      out.write(blobData.data(), static_cast<std::streamsize>(blobData.size()));
    }

    auto fileSize = static_cast<uint64_t>(blobData.size());

    // Create IcebergDeleteFile with DV metadata.
    std::unordered_map<int32_t, std::string> lowerBounds;
    std::unordered_map<int32_t, std::string> upperBounds;
    lowerBounds[DeletionVectorReader::kDvOffsetFieldId] = "0";
    upperBounds[DeletionVectorReader::kDvLengthFieldId] =
        std::to_string(fileSize);

    IcebergDeleteFile dvFile(
        FileContent::kDeletionVector,
        tempFile->getPath(),
        dwio::common::FileFormat::DWRF,
        positions.size(),
        fileSize,
        {},
        lowerBounds,
        upperBounds);

    DeletionVectorReader reader(dvFile, 0, pool_.get());

    // Collect all set bits across batches.
    std::vector<uint64_t> allSetBits;
    int64_t maxPos = positions.empty()
        ? 0
        : *std::max_element(positions.begin(), positions.end());
    uint64_t totalRows = static_cast<uint64_t>(maxPos) + batchSize;

    for (uint64_t offset = 0; offset < totalRows; offset += batchSize) {
      auto bitmap = allocateBitmap(batchSize);
      reader.readDeletePositions(offset, batchSize, bitmap);
      auto bits = getSetBits(bitmap, batchSize);
      for (auto b : bits) {
        allSetBits.push_back(offset + b);
      }
    }

    // Sort and deduplicate the expected positions.
    std::vector<int64_t> expected = positions;
    std::sort(expected.begin(), expected.end());
    expected.erase(
        std::unique(expected.begin(), expected.end()), expected.end());

    EXPECT_EQ(allSetBits.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(allSetBits[i], static_cast<uint64_t>(expected[i]));
    }
  }

  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(DeletionVectorWriterTest, emptyBitmap) {
  DeletionVectorWriter writer;
  EXPECT_EQ(writer.numPositions(), 0);

  auto data = writer.serialize();
  // Empty 64-bit bitmap: numGroups=0 as uint64 (8 bytes).
  EXPECT_EQ(data.size(), 8);
}

TEST_F(DeletionVectorWriterTest, singlePosition) {
  verifyRoundTrip({42}, 100);
}

TEST_F(DeletionVectorWriterTest, multiplePositions) {
  verifyRoundTrip({0, 5, 10, 99}, 100);
}

TEST_F(DeletionVectorWriterTest, consecutivePositions) {
  std::vector<int64_t> positions;
  positions.reserve(100);
  for (int64_t i = 0; i < 100; ++i) {
    positions.push_back(i);
  }
  verifyRoundTrip(positions, 200);
}

TEST_F(DeletionVectorWriterTest, multipleContainers) {
  // Positions spanning two containers (key=0 and key=1).
  verifyRoundTrip({5, 100, 65536, 65600}, 70000);
}

TEST_F(DeletionVectorWriterTest, largeCardinalityBitmapContainer) {
  // More than 4096 positions in a single container triggers bitmap container.
  std::vector<int64_t> positions;
  positions.reserve(5000);
  for (int64_t i = 0; i < 5000; ++i) {
    positions.push_back(i * 2); // Even numbers 0..9998.
  }
  verifyRoundTrip(positions, 10100);
}

TEST_F(DeletionVectorWriterTest, duplicatePositions) {
  // addDeletedPosition() does not deduplicate — numPositions() counts all
  // insertions including duplicates. serialize() deduplicates via std::set.
  DeletionVectorWriter writer;
  writer.addDeletedPosition(5);
  writer.addDeletedPosition(5);
  writer.addDeletedPosition(10);
  writer.addDeletedPosition(10);
  writer.addDeletedPosition(10);
  EXPECT_EQ(writer.numPositions(), 5);

  auto data = writer.serialize();

  auto tempFile = TempFilePath::create();
  {
    std::ofstream out(tempFile->getPath(), std::ios::binary | std::ios::trunc);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
  }

  std::unordered_map<int32_t, std::string> lowerBounds;
  std::unordered_map<int32_t, std::string> upperBounds;
  lowerBounds[DeletionVectorReader::kDvOffsetFieldId] = "0";
  upperBounds[DeletionVectorReader::kDvLengthFieldId] =
      std::to_string(data.size());

  IcebergDeleteFile dvFile(
      FileContent::kDeletionVector,
      tempFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2, // Only 2 unique positions.
      data.size(),
      {},
      lowerBounds,
      upperBounds);

  DeletionVectorReader reader(dvFile, 0, pool_.get());

  auto bitmap = allocateBitmap(20);
  reader.readDeletePositions(0, 20, bitmap);

  auto setBits = getSetBits(bitmap, 20);
  EXPECT_EQ(setBits, (std::vector<uint64_t>{5, 10}));
}

TEST_F(DeletionVectorWriterTest, clearPositions) {
  DeletionVectorWriter writer;
  writer.addDeletedPosition(1);
  writer.addDeletedPosition(2);
  EXPECT_EQ(writer.numPositions(), 2);

  writer.clear();
  EXPECT_EQ(writer.numPositions(), 0);

  auto data = writer.serialize();
  EXPECT_EQ(data.size(), 8); // Empty bitmap.
}

TEST_F(DeletionVectorWriterTest, negativePositionRejected) {
  DeletionVectorWriter writer;
  VELOX_ASSERT_THROW(
      writer.addDeletedPosition(-1), "Deleted position must be non-negative");
}

TEST_F(DeletionVectorWriterTest, fourOrMoreContainersWithOffsets) {
  // With >= 4 containers, the roaring format includes an offset section.
  std::vector<int64_t> positions;
  positions.reserve(5);
  for (int i = 0; i < 5; ++i) {
    positions.push_back(static_cast<int64_t>(i) * 65536 + 42);
  }
  verifyRoundTrip(positions, 5 * 65536 + 100);
}

TEST_F(DeletionVectorWriterTest, puffinFileRoundTrip) {
  DeletionVectorWriter writer;
  writer.addDeletedPositions({3, 7, 42, 100});
  auto blobData = writer.serialize();

  auto tempFile = TempFilePath::create();
  auto [blobOffset, blobLength] = writePuffinFile(
      tempFile->getPath(), blobData, "/data/test-data-file.parquet");

  EXPECT_EQ(blobOffset, 4); // After "PUF1" magic.
  EXPECT_EQ(blobLength, blobData.size());

  // Read the blob back from the Puffin file using DeletionVectorReader.
  std::unordered_map<int32_t, std::string> lowerBounds;
  std::unordered_map<int32_t, std::string> upperBounds;
  lowerBounds[DeletionVectorReader::kDvOffsetFieldId] =
      std::to_string(blobOffset);
  upperBounds[DeletionVectorReader::kDvLengthFieldId] =
      std::to_string(blobLength);

  // Get full file size.
  std::ifstream in(tempFile->getPath(), std::ios::binary | std::ios::ate);
  auto fileSize = static_cast<uint64_t>(in.tellg());

  IcebergDeleteFile dvFile(
      FileContent::kDeletionVector,
      tempFile->getPath(),
      dwio::common::FileFormat::DWRF,
      4,
      fileSize,
      {},
      lowerBounds,
      upperBounds);

  DeletionVectorReader reader(dvFile, 0, pool_.get());

  auto bitmap = allocateBitmap(200);
  reader.readDeletePositions(0, 200, bitmap);

  auto setBits = getSetBits(bitmap, 200);
  EXPECT_EQ(setBits, (std::vector<uint64_t>{3, 7, 42, 100}));
}

/// Verifies 64-bit positions (>4 billion) serialize and deserialize correctly.
/// This exercises the Roaring64Bitmap group partitioning for large data files.
TEST_F(DeletionVectorWriterTest, largePositions64Bit) {
  // Positions beyond the 32-bit range.
  std::vector<int64_t> positions = {
      100,
      65'536,
      5'000'000'000LL,
      5'000'000'001LL,
      10'000'000'000LL,
  };
  verifyRoundTrip(positions, 1'024);
}

/// Verifies mixed 32-bit and 64-bit positions in the same bitmap.
TEST_F(DeletionVectorWriterTest, mixed32And64BitPositions) {
  std::vector<int64_t> positions = {
      0,
      1,
      65'535,
      65'536,
      4'294'967'295LL,
      4'294'967'296LL,
      8'589'934'592LL,
  };
  verifyRoundTrip(positions, 2'048);
}
