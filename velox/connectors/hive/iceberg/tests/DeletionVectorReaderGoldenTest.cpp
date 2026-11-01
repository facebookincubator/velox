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

// Reads deletion vectors written by Apache Iceberg Java and checks every
// deleted position against what Iceberg Java itself decodes from the same
// bytes. The fixtures and the expected results live in
// examples/deletion_vectors; see manifest.tsv there for how they were made.

#include "velox/connectors/hive/iceberg/DeletionVectorReader.h"

#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <sstream>

#include <folly/Portability.h>
#include <folly/String.h>
#include <gtest/gtest.h>
#include <openssl/sha.h>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

using common::testutil::TempFilePath;

static_assert(folly::kIsLittleEndian);

// One record of examples/deletion_vectors/manifest.tsv. The last four fields
// are what Iceberg Java decoded from the blob.
struct GoldenCase {
  std::string name;
  std::string blobSha256;
  uint64_t cardinality;
  int64_t firstPosition;
  int64_t lastPosition;
  std::string positionsSha256;
};

std::string goldenPath(const std::string& file) {
  return test::getDataFilePath(
      "velox/connectors/hive/iceberg/tests",
      "examples/deletion_vectors/" + file);
}

std::string readFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  VELOX_CHECK(in.is_open(), "Cannot open {}", path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::string sha256Hex(const void* data, size_t size) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(static_cast<const unsigned char*>(data), size, digest);
  return folly::hexlify(
      std::string_view(reinterpret_cast<const char*>(digest), sizeof(digest)));
}

// Same encoding as the generator: positions as little-endian int64.
std::string positionsSha256(const std::vector<int64_t>& positions) {
  return sha256Hex(positions.data(), positions.size() * sizeof(int64_t));
}

std::vector<GoldenCase> loadManifest() {
  std::ifstream in(goldenPath("manifest.tsv"));
  VELOX_CHECK(in.is_open(), "Cannot open the golden manifest");
  std::vector<GoldenCase> cases;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::vector<std::string> fields;
    folly::split('\t', line, fields);
    VELOX_CHECK_EQ(fields.size(), 6, "Malformed manifest record: {}", line);
    cases.push_back(
        {fields[0],
         fields[1],
         folly::to<uint64_t>(fields[2]),
         folly::to<int64_t>(fields[3]),
         folly::to<int64_t>(fields[4]),
         fields[5]});
  }
  return cases;
}

IcebergDeleteFile makeDeleteFile(
    const std::string& path,
    uint64_t cardinality,
    uint64_t fileSize,
    uint64_t blobOffset,
    uint64_t blobLength) {
  return IcebergDeleteFile(
      FileContent::kDeletionVector,
      path,
      dwio::common::FileFormat::DWRF,
      cardinality,
      fileSize,
      /*equalityFieldIds=*/{},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/0,
      /*contentOffset=*/static_cast<int64_t>(blobOffset),
      /*contentLength=*/static_cast<int64_t>(blobLength));
}

std::shared_ptr<TempFilePath> writeTempFile(const std::string& data) {
  auto file = TempFilePath::create();
  std::ofstream out(file->getPath(), std::ios::binary | std::ios::trunc);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  return file;
}

class DeletionVectorReaderGoldenTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    filesystems::registerLocalFileSystem();
    pool_ = memory::memoryManager()->addLeafPool();
    cases_ = loadManifest();
  }

  IcebergDeleteFile goldenDeleteFile(const GoldenCase& golden) const {
    const auto path = goldenPath(golden.name + ".bin");
    const auto size = std::filesystem::file_size(path);
    return makeDeleteFile(path, golden.cardinality, size, 0, size);
  }

  std::vector<int64_t> expandedPositions(const IcebergDeleteFile& file) {
    DeletionVectorReader reader(file, 0, pool_.get(), nullptr);
    return reader.deletedPositions();
  }

  // Reads the deletion vector batch by batch, the way the split reader does,
  // and returns the file positions marked deleted. Batch 'k' covers split rows
  // [k * batchSize, (k + 1) * batchSize). When scanning every batch up to the
  // last deleted row would take too many calls, only the batches holding a
  // deleted row and their neighbours are read; skipping ahead is how the
  // reader sees row groups that a filter pruned.
  std::vector<int64_t> scan(
      const GoldenCase& golden,
      uint64_t splitOffset,
      const std::vector<uint64_t>& batchSizes,
      const std::vector<int64_t>& expected) {
    DeletionVectorReader reader(
        goldenDeleteFile(golden), splitOffset, pool_.get(), nullptr);
    EXPECT_FALSE(reader.noMoreData());

    // Batch sizes cycle. Walk batch boundaries rather than indexes so mixed
    // sizes work.
    constexpr uint64_t kMaxFullScanBatches = 1 << 17;
    constexpr uint64_t kMaxFullScanRows = 1 << 30;
    const uint64_t maxBatch =
        *std::max_element(batchSizes.begin(), batchSizes.end());
    const uint64_t minBatch =
        *std::min_element(batchSizes.begin(), batchSizes.end());
    const uint64_t lastRow =
        golden.lastPosition >= static_cast<int64_t>(splitOffset)
        ? golden.lastPosition - splitOffset
        : 0;
    const bool fullScan = lastRow < kMaxFullScanRows &&
        lastRow / minBatch + 2 <= kMaxFullScanBatches;

    // One spare word after the batch catches writes past the end of the
    // bitmap.
    auto bitmap = AlignedBuffer::allocate<uint64_t>(
        bits::nwords(maxBatch) + 1, pool_.get(), 0);

    std::vector<int64_t> deleted;
    size_t nextExpected = 0;
    size_t sizeIndex = 0;
    uint64_t start = 0;
    while (true) {
      const uint64_t size = batchSizes[sizeIndex++ % batchSizes.size()];
      if (!fullScan) {
        // Jump to the batch before the next deleted row, if it is further on.
        while (nextExpected < expected.size() &&
               static_cast<uint64_t>(expected[nextExpected]) - splitOffset <
                   start) {
          ++nextExpected;
        }
        if (nextExpected < expected.size()) {
          const uint64_t row = expected[nextExpected] - splitOffset;
          if (row >= start + 2 * size) {
            start = row - row % size - size;
          }
        }
      }

      const uint64_t numWords = bits::nwords(size) + 1;
      std::memset(
          bitmap->asMutable<uint64_t>(), 0, numWords * sizeof(uint64_t));
      reader.readDeletePositions(start, size, bitmap);
      const auto* words = bitmap->as<uint64_t>();
      for (uint64_t w = 0; w < numWords; ++w) {
        for (uint64_t word = words[w]; word != 0; word &= word - 1) {
          const uint64_t bit = w * 64 + __builtin_ctzll(word);
          if (bit >= size) {
            ADD_FAILURE() << "Bit " << bit << " set past a batch of " << size
                          << " rows at split row " << start;
            break;
          }
          deleted.push_back(static_cast<int64_t>(splitOffset + start + bit));
        }
      }
      start += size;
      if (start > lastRow) {
        break;
      }
    }
    EXPECT_TRUE(reader.noMoreData());
    return deleted;
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::vector<GoldenCase> cases_;
};

TEST_F(DeletionVectorReaderGoldenTest, manifestCoversEveryFixture) {
  std::set<std::string> listed;
  for (const auto& golden : cases_) {
    EXPECT_TRUE(listed.insert(golden.name).second)
        << "Duplicate manifest record " << golden.name;
  }
  std::set<std::string> onDisk;
  for (const auto& entry : std::filesystem::directory_iterator(
           std::filesystem::path(goldenPath("manifest.tsv")).parent_path())) {
    if (entry.path().extension() == ".bin") {
      onDisk.insert(entry.path().stem().string());
    }
  }
  EXPECT_EQ(listed, onDisk);
  EXPECT_EQ(cases_.size(), 12);
}

TEST_F(DeletionVectorReaderGoldenTest, fixturesHaveNotDrifted) {
  for (const auto& golden : cases_) {
    const auto blob = readFile(goldenPath(golden.name + ".bin"));
    EXPECT_EQ(sha256Hex(blob.data(), blob.size()), golden.blobSha256)
        << golden.name
        << ".bin changed. Regenerate the fixtures and manifest from Iceberg "
           "Java instead of editing them.";
  }
}

TEST_F(DeletionVectorReaderGoldenTest, deletedPositionsMatchIcebergJava) {
  for (const auto& golden : cases_) {
    SCOPED_TRACE(golden.name);
    const auto positions = expandedPositions(goldenDeleteFile(golden));
    ASSERT_EQ(positions.size(), golden.cardinality);
    EXPECT_EQ(positions.front(), golden.firstPosition);
    EXPECT_EQ(positions.back(), golden.lastPosition);
    EXPECT_TRUE(std::is_sorted(positions.begin(), positions.end()));
    EXPECT_EQ(
        std::adjacent_find(positions.begin(), positions.end()),
        positions.end());
    EXPECT_EQ(positionsSha256(positions), golden.positionsSha256);
  }
}

TEST_F(DeletionVectorReaderGoldenTest, batchesMatchIcebergJava) {
  // Sizes that do and do not line up with bitmap bytes, 64-bit words, array
  // containers (4096) and whole containers (65536).
  const std::vector<uint64_t> batchSizes = {
      1, 7, 63, 64, 65, 4'096, 4'097, 65'535, 65'536, 65'549, 1 << 20, 1 << 24};
  for (const auto& golden : cases_) {
    SCOPED_TRACE(golden.name);
    const auto expected = expandedPositions(goldenDeleteFile(golden));
    ASSERT_EQ(positionsSha256(expected), golden.positionsSha256);

    for (uint64_t batchSize : batchSizes) {
      SCOPED_TRACE(fmt::format("batch size {}", batchSize));
      const auto deleted = scan(golden, 0, {batchSize}, expected);
      ASSERT_EQ(deleted.size(), golden.cardinality);
      EXPECT_EQ(positionsSha256(deleted), golden.positionsSha256);
    }
  }
}

TEST_F(DeletionVectorReaderGoldenTest, batchesOfMixedSizesMatchIcebergJava) {
  std::mt19937 rng(20'260'101);
  std::uniform_int_distribution<uint64_t> size(1, 100'000);
  std::vector<uint64_t> batchSizes(997);
  for (auto& batchSize : batchSizes) {
    batchSize = size(rng);
  }
  for (const auto& golden : cases_) {
    SCOPED_TRACE(golden.name);
    const auto expected = expandedPositions(goldenDeleteFile(golden));
    const auto deleted = scan(golden, 0, batchSizes, expected);
    ASSERT_EQ(deleted.size(), golden.cardinality);
    EXPECT_EQ(positionsSha256(deleted), golden.positionsSha256);
  }
}

TEST_F(DeletionVectorReaderGoldenTest, splitsMatchIcebergJava) {
  // A split that starts inside the file sees only the deletes at or after its
  // first row. The offsets fall inside an array container, just past a
  // container boundary, and past the first 2^32-row group.
  const std::vector<uint64_t> splitOffsets = {
      7, 65'539, 131'072, (uint64_t{1} << 32) + 5};
  for (const auto& golden : cases_) {
    SCOPED_TRACE(golden.name);
    const auto all = expandedPositions(goldenDeleteFile(golden));
    ASSERT_EQ(positionsSha256(all), golden.positionsSha256);
    for (uint64_t splitOffset : splitOffsets) {
      SCOPED_TRACE(fmt::format("split offset {}", splitOffset));
      std::vector<int64_t> expected;
      std::copy_if(
          all.begin(),
          all.end(),
          std::back_inserter(expected),
          [&](int64_t position) {
            return position >= static_cast<int64_t>(splitOffset);
          });
      for (uint64_t batchSize : {uint64_t{1'000}, uint64_t{65'549}}) {
        SCOPED_TRACE(fmt::format("batch size {}", batchSize));
        EXPECT_EQ(scan(golden, splitOffset, {batchSize}, expected), expected);
      }
    }
  }
}

TEST_F(DeletionVectorReaderGoldenTest, blobInsideLargerFile) {
  // In a Puffin file the blob sits between the header magic and the footer;
  // the manifest's content offset and length locate it.
  for (const auto& golden : cases_) {
    SCOPED_TRACE(golden.name);
    const auto blob = readFile(goldenPath(golden.name + ".bin"));
    const std::string prefix = "PFA1abc";
    const std::string suffix(16, '\xAB');
    auto file = writeTempFile(prefix + blob + suffix);
    const auto deleteFile = makeDeleteFile(
        file->getPath(),
        golden.cardinality,
        prefix.size() + blob.size() + suffix.size(),
        prefix.size(),
        blob.size());
    EXPECT_EQ(
        positionsSha256(expandedPositions(deleteFile)), golden.positionsSha256);
  }
}

TEST_F(DeletionVectorReaderGoldenTest, corruptBlobsAreRejected) {
  for (const auto& golden : cases_) {
    SCOPED_TRACE(golden.name);
    const auto blob = readFile(goldenPath(golden.name + ".bin"));
    const auto read = [&](const std::string& bytes) {
      auto file = writeTempFile(bytes);
      return expandedPositions(makeDeleteFile(
          file->getPath(), golden.cardinality, bytes.size(), 0, bytes.size()));
    };

    auto truncated = blob.substr(0, blob.size() - 1);
    EXPECT_THROW(read(truncated), VeloxException);

    if (golden.name == "legacy32_runs") {
      // Unframed bitmaps carry no checksum.
      continue;
    }

    auto flippedBitmap = blob;
    flippedBitmap[8 + (blob.size() - 12) / 2] ^= 0x10;
    VELOX_ASSERT_THROW(
        read(flippedBitmap), "Deletion-vector-v1 CRC-32 mismatch.");

    auto flippedCrc = blob;
    flippedCrc.back() ^= 0x01;
    VELOX_ASSERT_THROW(read(flippedCrc), "Deletion-vector-v1 CRC-32 mismatch.");

    VELOX_ASSERT_THROW(
        read(truncated), "Deletion-vector-v1 blob size mismatch.");

    auto longerLength = blob;
    ++longerLength[3];
    VELOX_ASSERT_THROW(
        read(longerLength), "Deletion-vector-v1 blob size mismatch.");
  }
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
