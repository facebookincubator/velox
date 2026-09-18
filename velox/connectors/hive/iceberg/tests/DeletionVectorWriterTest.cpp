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

#include <folly/json.h>
#include <folly/lang/Bits.h>
#include <gtest/gtest.h>
#include <zlib.h>

#include <cstring>
#include <limits>
#include <random>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/hive/iceberg/DeletionVectorReader.h"
#include "velox/dwio/common/FileSink.h"

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
    // writePuffinFile now writes through dwio::common::FileSink::create, which
    // dispatches by URI scheme to a registered factory. Register the
    // local-filesystem sink so plain temp-file paths land through the same
    // dispatch the production binary uses.
    dwio::common::LocalFileSink::registerFactory();
    pool_ = memory::memoryManager()->addLeafPool("DeletionVectorWriterTest");
  }

  BufferPtr allocateBitmap(uint64_t numBits) {
    auto numBytes = bits::nbytes(numBits);
    return AlignedBuffer::allocate<uint8_t>(numBytes, pool_.get(), 0);
  }

  /// Serializes 'positions', reads the blob back through
  /// DeletionVectorReader, and returns every position the reader recovered.
  ///
  /// Prefer this over verifyRoundTrip for inputs spanning a wide range:
  /// verifyRoundTrip walks the whole [0, maxPos] space one batch at a time,
  /// which is impractical once positions reach into the 2^32 range.
  std::vector<int64_t> roundTrip(const std::vector<int64_t>& positions) {
    DeletionVectorWriter writer;
    writer.addDeletedPositions(positions);
    const auto blobData = writer.serialize();

    auto tempFile = TempFilePath::create();
    {
      std::ofstream out(
          tempFile->getPath(), std::ios::binary | std::ios::trunc);
      out.write(blobData.data(), static_cast<std::streamsize>(blobData.size()));
    }

    IcebergDeleteFile dvFile(
        FileContent::kDeletionVector,
        tempFile->getPath(),
        dwio::common::FileFormat::DWRF,
        writer.numDistinctPositions(),
        static_cast<uint64_t>(blobData.size()),
        /*equalityFieldIds=*/{},
        /*lowerBounds=*/{},
        /*upperBounds=*/{},
        /*dataSequenceNumber=*/0,
        /*contentOffset=*/0,
        /*contentLength=*/static_cast<int64_t>(blobData.size()));

    DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
    return reader.deletedPositions();
  }

  /// Returns 'positions' sorted and de-duplicated — what a round trip through
  /// a roaring bitmap is expected to yield, since the bitmap is a set.
  static std::vector<int64_t> sortedUnique(std::vector<int64_t> positions) {
    std::sort(positions.begin(), positions.end());
    positions.erase(
        std::unique(positions.begin(), positions.end()), positions.end());
    return positions;
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

    DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

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

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

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

// Decodes the Puffin footer of a file written by writePuffinFile and returns
// the parsed footer JSON. Layout per the Puffin spec:
//   Magic Blob... Footer
//   Footer := Magic FooterPayload FooterPayloadSize Flags Magic
// with FooterPayloadSize and Flags each a 4-byte little-endian value. The
// trailer is located from the end of the file so nothing here depends on the
// writer's own offset arithmetic.
namespace {
folly::dynamic readPuffinFooter(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  const std::string bytes(
      (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

  constexpr size_t kMagicSize = 4;
  constexpr size_t kTrailerSize = kMagicSize + 4 + 4;
  EXPECT_GE(bytes.size(), kMagicSize + kTrailerSize);
  EXPECT_EQ(bytes.substr(0, kMagicSize), "PFA1") << "missing leading magic";
  EXPECT_EQ(bytes.substr(bytes.size() - kMagicSize), "PFA1")
      << "missing trailing magic";

  const auto readLittleEndian32 = [&](size_t offset) {
    uint32_t value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return folly::Endian::little(value);
  };

  const size_t flagsOffset = bytes.size() - kMagicSize - 4;
  const size_t sizeOffset = flagsOffset - 4;
  EXPECT_EQ(readLittleEndian32(flagsOffset), 0u)
      << "flags must be 0: an uncompressed footer payload is bit 0";

  const uint32_t payloadSize = readLittleEndian32(sizeOffset);
  const size_t payloadOffset = sizeOffset - payloadSize;
  EXPECT_EQ(bytes.substr(payloadOffset - kMagicSize, kMagicSize), "PFA1")
      << "footer payload must be preceded by magic";

  return folly::parseJson(bytes.substr(payloadOffset, payloadSize));
}
} // namespace

TEST_F(DeletionVectorWriterTest, puffinFooterIsSpecCompliant) {
  // Nothing in our own read path parses the Puffin footer --
  // DeletionVectorReader locates the blob from the manifest's contentOffset --
  // so the footer is written blind. Other Iceberg engines do parse it, and
  // FileMetadataParser.blobMetadataFromJson treats type, fields, snapshot-id,
  // sequence-number, offset and length as required, reading fields as a list
  // of integers. A footer that omits or mistypes any of them makes the whole
  // deletion vector unreadable outside Velox.
  DeletionVectorWriter writer;
  writer.addDeletedPositions({3, 7, 42, 100});
  const auto blobData = writer.serialize();

  auto tempDir = TempDirectoryPath::create();
  const std::string puffinPath =
      std::string(tempDir->getPath()) + "/footer.puffin";
  auto sink = dwio::common::FileSink::create(
      "file:" + puffinPath, {.pool = pool_.get()});
  auto [blobOffset, blobLength] = writePuffinFile(
      *sink,
      *pool_,
      blobData,
      "/data/test-data-file.parquet",
      /*cardinality=*/4);
  sink->close();

  const auto footer = readPuffinFooter(puffinPath);
  ASSERT_TRUE(footer.isObject());
  ASSERT_TRUE(footer["blobs"].isArray());
  ASSERT_EQ(footer["blobs"].size(), 1);
  const auto& blob = footer["blobs"][0];

  EXPECT_EQ(blob["type"].asString(), "deletion-vector-v1");

  // Iceberg's BaseDVFileWriter sets this to the single row-position metadata
  // column ID, as a list of plain integers.
  constexpr int64_t kRowPositionFieldId = 2'147'483'645;
  ASSERT_TRUE(blob["fields"].isArray()) << "fields must be a list";
  ASSERT_EQ(blob["fields"].size(), 1);
  ASSERT_TRUE(blob["fields"][0].isInt())
      << "fields entries must be integers, not objects";
  EXPECT_EQ(blob["fields"][0].asInt(), kRowPositionFieldId);

  // Required by the parser even though a freshly written DV has no snapshot
  // assigned yet; Iceberg writes -1 for both.
  ASSERT_TRUE(blob.count("snapshot-id")) << "snapshot-id is required";
  ASSERT_TRUE(blob.count("sequence-number")) << "sequence-number is required";
  EXPECT_EQ(blob["snapshot-id"].asInt(), -1);
  EXPECT_EQ(blob["sequence-number"].asInt(), -1);

  EXPECT_EQ(blob["offset"].asInt(), static_cast<int64_t>(blobOffset));
  EXPECT_EQ(blob["length"].asInt(), static_cast<int64_t>(blobLength));

  const auto& properties = blob["properties"];
  EXPECT_EQ(
      properties["referenced-data-file"].asString(),
      "/data/test-data-file.parquet");
  EXPECT_EQ(properties["cardinality"].asString(), "4");
}

TEST_F(DeletionVectorWriterTest, puffinFileRoundTrip) {
  DeletionVectorWriter writer;
  writer.addDeletedPositions({3, 7, 42, 100});
  auto blobData = writer.serialize();

  auto tempDir = TempDirectoryPath::create();
  const std::string puffinPath =
      std::string(tempDir->getPath()) + "/test-dv.puffin";
  // FileSink::create dispatches by URI scheme; the registered LocalFileSink
  // factory writes the puffin bytes to the local path.
  auto sink = dwio::common::FileSink::create(
      "file:" + puffinPath, {.pool = pool_.get()});
  VELOX_CHECK_NOT_NULL(sink);
  auto [blobOffset, blobLength] = writePuffinFile(
      *sink,
      *pool_,
      blobData,
      "/data/test-data-file.parquet",
      /*cardinality=*/4);
  sink->close();

  EXPECT_EQ(blobOffset, 4); // After "PFA1" magic.
  // blobLength is the framed deletion-vector-v1 blob: 4B length + 4B magic +
  // bitmap + 4B CRC-32.
  EXPECT_EQ(blobLength, blobData.size() + 12);

  // Read the blob back from the Puffin file using DeletionVectorReader.
  std::unordered_map<int32_t, std::string> lowerBounds;
  std::unordered_map<int32_t, std::string> upperBounds;
  lowerBounds[DeletionVectorReader::kDvOffsetFieldId] =
      std::to_string(blobOffset);
  upperBounds[DeletionVectorReader::kDvLengthFieldId] =
      std::to_string(blobLength);

  // Get full file size.
  std::ifstream in(puffinPath, std::ios::binary | std::ios::ate);
  auto fileSize = static_cast<uint64_t>(in.tellg());

  IcebergDeleteFile dvFile(
      FileContent::kDeletionVector,
      puffinPath,
      dwio::common::FileFormat::DWRF,
      4,
      fileSize,
      {},
      lowerBounds,
      upperBounds);

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

  auto bitmap = allocateBitmap(200);
  reader.readDeletePositions(0, 200, bitmap);

  auto setBits = getSetBits(bitmap, 200);
  EXPECT_EQ(setBits, (std::vector<uint64_t>{3, 7, 42, 100}));
}

// Reads a writePuffinFile output back with NO bounds-map location, so the
// reader must parse the Puffin footer (locateBlobFromPuffinFooter) to find the
// blob. Unlike puffinFileRoundTrip -- which supplies explicit offset/length and
// therefore skips the footer parser -- this exercises the writer's footer
// layout against the reader's backwards footer parse, catching any drift in
// trailer size, magic placement, or payload-size/flags encoding.
TEST_F(DeletionVectorWriterTest, puffinFooterFallbackRoundTrip) {
  DeletionVectorWriter writer;
  writer.addDeletedPositions({3, 7, 42, 100});
  auto blobData = writer.serialize();

  auto tempDir = TempDirectoryPath::create();
  const std::string puffinPath =
      std::string(tempDir->getPath()) + "/test-dv-footer.puffin";
  auto sink = dwio::common::FileSink::create(
      "file:" + puffinPath, {.pool = pool_.get()});
  VELOX_CHECK_NOT_NULL(sink);
  writePuffinFile(
      *sink,
      *pool_,
      blobData,
      "/data/test-data-file.parquet",
      /*cardinality=*/4);
  sink->close();

  std::ifstream in(puffinPath, std::ios::binary | std::ios::ate);
  auto fileSize = static_cast<uint64_t>(in.tellg());

  // Empty bounds maps: with no offset/length the reader falls through to
  // Puffin-footer parsing and selects the single deletion-vector blob.
  IcebergDeleteFile dvFile(
      FileContent::kDeletionVector,
      puffinPath,
      dwio::common::FileFormat::DWRF,
      4,
      fileSize,
      {},
      {},
      {});

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

  auto bitmap = allocateBitmap(200);
  reader.readDeletePositions(0, 200, bitmap);

  auto setBits = getSetBits(bitmap, 200);
  EXPECT_EQ(setBits, (std::vector<uint64_t>{3, 7, 42, 100}));
}

// Verifies the on-disk deletion-vector-v1 blob matches the Iceberg V3 spec
// frame: [length: 4B BE][magic D1 D3 39 64][bitmap][CRC-32: 4B BE], where the
// length and CRC-32 cover the magic + bitmap. This is what makes the DV
// interoperable with spec-compliant Iceberg engines.
TEST_F(DeletionVectorWriterTest, deletionVectorV1FrameLayout) {
  DeletionVectorWriter writer;
  writer.addDeletedPositions({3, 7, 42, 100});
  const auto bitmap = writer.serialize();

  auto tempDir = TempDirectoryPath::create();
  const std::string puffinPath =
      std::string(tempDir->getPath()) + "/frame.puffin";
  auto sink = dwio::common::FileSink::create(
      "file:" + puffinPath, {.pool = pool_.get()});
  const auto [blobOffset, blobLength] = writePuffinFile(
      *sink, *pool_, bitmap, "/data/f.parquet", /*cardinality=*/4);
  sink->close();

  std::ifstream in(puffinPath, std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(in)), {});

  // File starts with the Puffin magic "PFA1".
  EXPECT_EQ(bytes.substr(0, 4), std::string("PFA1"));

  const std::string blob = bytes.substr(blobOffset, blobLength);
  ASSERT_EQ(blob.size(), bitmap.size() + 12);

  auto readBigEndian = [](const char* p) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(p);
    return (static_cast<uint32_t>(bytes[0]) << 24) |
        (static_cast<uint32_t>(bytes[1]) << 16) |
        (static_cast<uint32_t>(bytes[2]) << 8) |
        static_cast<uint32_t>(bytes[3]);
  };

  // [length: 4B BE] covers magic (4) + bitmap.
  const uint32_t magicAndVectorLength = readBigEndian(blob.data());
  EXPECT_EQ(magicAndVectorLength, bitmap.size() + 4);

  // [magic: D1 D3 39 64].
  const std::string magic = blob.substr(4, 4);
  EXPECT_EQ(magic, std::string({'\xD1', '\xD3', '\x39', '\x64'}));

  // [bitmap] matches the writer's serialize() output.
  EXPECT_EQ(blob.substr(8, bitmap.size()), bitmap);

  // [CRC-32: 4B BE] over magic + bitmap. Iceberg stores the standard CRC-32
  // (java.util.zip.CRC32); zlib's crc32 is the reference implementation of that
  // same IEEE 802.3 algorithm.
  const uint32_t storedCrc =
      readBigEndian(blob.data() + 4 + magicAndVectorLength);
  uLong crcState = crc32(0L, Z_NULL, 0);
  crcState = crc32(
      crcState,
      reinterpret_cast<const Bytef*>(blob.data() + 4),
      static_cast<uInt>(magicAndVectorLength));
  const auto expectedCrc = static_cast<uint32_t>(crcState);
  EXPECT_EQ(storedCrc, expectedCrc);
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

/// Verifies that duplicate positions collapse in the cardinality reported for
/// the DV blob. Seeding a writer from an existing deletion vector and then
/// adding overlapping new deletes makes 'positions_' hold duplicates, so
/// 'numPositions' overcounts and only 'numDistinctPositions' matches the
/// cardinality Iceberg expects in the blob metadata.
TEST_F(DeletionVectorWriterTest, numDistinctPositionsIgnoresDuplicates) {
  DeletionVectorWriter writer;
  writer.addDeletedPositions({7, 3, 7, 1, 3, 3});

  EXPECT_EQ(writer.numPositions(), 6);
  EXPECT_EQ(writer.numDistinctPositions(), 3);
}

// Randomized round trips across sparse, dense, and mixed position sets.
// Hand-built fixtures only exercise the container shapes we thought to write
// down; random inputs cross the array/bitset thresholds and 64-bit group
// boundaries in combinations we did not enumerate.
//
// Seeds are fixed so a failure is reproducible, and reported on failure so a
// counterexample can be replayed directly.
TEST_F(DeletionVectorWriterTest, randomSparsePositionsRoundTrip) {
  // Few positions spread over a wide 64-bit range: many container keys across
  // several Roaring64 groups, each holding a small array container.
  constexpr uint64_t kSeed = 0x1234'5678'9ABC'DEF0ULL;
  constexpr int kNumPositions = 2'000;
  constexpr int64_t kRange = int64_t{1} << 34;

  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<int64_t> positionDist(0, kRange - 1);

  std::vector<int64_t> positions;
  positions.reserve(kNumPositions);
  for (int i = 0; i < kNumPositions; ++i) {
    positions.push_back(positionDist(rng));
  }

  EXPECT_EQ(roundTrip(positions), sortedUnique(positions)) << "seed=" << kSeed;
}

TEST_F(DeletionVectorWriterTest, randomDensePositionsRoundTrip) {
  // Enough positions inside one 64K block to exceed the 4096 array-container
  // threshold, so the block serializes as a bitset. Deliberately includes
  // duplicates: the bitmap is a set, so they must collapse.
  constexpr uint64_t kSeed = 0x0FED'CBA9'8765'4321ULL;
  constexpr int kNumDraws = 30'000;
  constexpr int64_t kBlockBase = int64_t{7} << 16;

  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<int64_t> offsetDist(0, 65'535);

  std::vector<int64_t> positions;
  positions.reserve(kNumDraws);
  for (int i = 0; i < kNumDraws; ++i) {
    positions.push_back(kBlockBase + offsetDist(rng));
  }

  const auto expected = sortedUnique(positions);
  ASSERT_GT(expected.size(), 4'096)
      << "draws should exceed the array-container threshold";
  EXPECT_EQ(roundTrip(positions), expected) << "seed=" << kSeed;
}

TEST_F(DeletionVectorWriterTest, randomMixedPositionsRoundTrip) {
  // Dense block, sparse scatter, and a contiguous run in one bitmap, split
  // across two Roaring64 groups. This is the shape most likely to expose an
  // offset or stride error, because the reader must step over containers of
  // different encodings to find the next one.
  constexpr uint64_t kSeed = 0x2468'ACE0'1357'9BDFULL;
  constexpr int64_t kHighGroupBase = int64_t{1} << 32;

  std::mt19937_64 rng(kSeed);
  std::vector<int64_t> positions;

  // Dense block in group 0.
  std::uniform_int_distribution<int64_t> denseDist(0, 65'535);
  for (int i = 0; i < 20'000; ++i) {
    positions.push_back(denseDist(rng));
  }
  // Sparse scatter across group 0's upper blocks.
  std::uniform_int_distribution<int64_t> sparseDist(65'536, (int64_t{1} << 31));
  for (int i = 0; i < 500; ++i) {
    positions.push_back(sparseDist(rng));
  }
  // Contiguous run in group 1.
  for (int64_t i = 0; i < 3'000; ++i) {
    positions.push_back(kHighGroupBase + 1'000 + i);
  }
  // Sparse scatter in group 1.
  for (int i = 0; i < 200; ++i) {
    positions.push_back(kHighGroupBase + sparseDist(rng));
  }

  EXPECT_EQ(roundTrip(positions), sortedUnique(positions)) << "seed=" << kSeed;
}

// The Roaring64 group key is read back as a signed 32-bit int, so a position
// whose high word reaches 2^31 would deserialize as a negative key and be
// rejected by spec-compliant readers. Failing at insert time keeps us from
// writing a blob Iceberg cannot read.
TEST_F(DeletionVectorWriterTest, rejectsPositionsOutsideRepresentableRange) {
  DeletionVectorWriter writer;

  VELOX_ASSERT_THROW(writer.addDeletedPosition(-1), "must be non-negative");
  VELOX_ASSERT_THROW(
      writer.addDeletedPosition(DeletionVectorWriter::kMaxPosition + 1),
      "exceeds the maximum");
  VELOX_ASSERT_THROW(
      writer.addDeletedPosition(std::numeric_limits<int64_t>::max()),
      "exceeds the maximum");

  // None of the rejected positions were recorded.
  EXPECT_EQ(writer.numPositions(), 0);

  // The bound itself is representable and round-trips.
  writer.addDeletedPosition(DeletionVectorWriter::kMaxPosition);
  EXPECT_EQ(
      roundTrip({DeletionVectorWriter::kMaxPosition}),
      std::vector<int64_t>{DeletionVectorWriter::kMaxPosition});
}
