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
#include "velox/connectors/paimon/PaimonDeletionVectorReader.h"

#include <fmt/format.h>
#include <folly/lang/Bits.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "velox/buffer/Buffer.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/hive/iceberg/DeletionVectorWriter.h"

namespace facebook::velox::connector::paimon {
namespace {

using common::testutil::TempFilePath;

// Magic numbers of version 1 and version 2 deletion-vector frames.
constexpr uint32_t kBitmap32Magic = 1'581'511'376;
constexpr uint32_t kBitmap64Magic = 1'681'511'377;

// Cookies of a serialized 32-bit RoaringBitmap without and with run
// containers.
constexpr uint32_t kSerialCookieNoRun = 12'346;
constexpr uint32_t kSerialCookie = 12'347;

// A container that is not a run container is an array of up to
// kMaxArrayValues values, and a bitset of kBitsetBytes above that.
constexpr size_t kMaxArrayValues = 4'096;
constexpr size_t kBitsetBytes = 8'192;

// Number of rows a Roaring container covers.
constexpr int64_t kContainerRows = 65'536;

// First row of Roaring64 group 1.
constexpr int64_t kGroupBoundary = int64_t{1} << 32;

template <typename T>
void appendInt(std::string& data, T value, bool bigEndian) {
  value = bigEndian ? folly::Endian::big(value) : folly::Endian::little(value);
  data.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

// Returns the rows in [first, last].
std::vector<int64_t> rowRange(int64_t first, int64_t last) {
  std::vector<int64_t> rows(last - first + 1);
  std::iota(rows.begin(), rows.end(), first);
  return rows;
}

// Serializes 'positions' as the bitmap of a frame of 'version': a 32-bit
// RoaringBitmap for version 1 and a Roaring64 bitmap for version 2.
std::string serializeBitmap(
    const std::vector<int64_t>& positions,
    int32_t version) {
  hive::iceberg::DeletionVectorWriter writer;
  writer.addDeletedPositions(positions);
  auto bitmap = writer.serialize();
  if (version == 1) {
    // Positions below 2^32 form one Roaring64 group. Dropping the group count
    // and key leaves its 32-bit RoaringBitmap.
    bitmap = bitmap.substr(sizeof(uint64_t) + sizeof(uint32_t));
  }
  return bitmap;
}

// Serializes 'positions', which increase and are below 2^32, as a 32-bit
// RoaringBitmap the way Java RoaringBitmap writes it after runOptimize(),
// which Paimon Java calls before it writes a frame: a container takes the run
// encoding when that is smaller than its array or bitset. DeletionVectorWriter
// writes no run containers.
std::string serializeRunOptimized32(const std::vector<int64_t>& positions) {
  // The positions that share the high 16 bits 'key', and the [first, last]
  // runs they form.
  struct Container {
    uint16_t key;
    std::vector<uint16_t> values;
    std::vector<std::pair<uint16_t, uint16_t>> runs;
    bool isRun{false};
  };
  std::vector<Container> containers;
  for (const auto position : positions) {
    const auto key = static_cast<uint16_t>(position >> 16);
    const auto value = static_cast<uint16_t>(position & 0xFFFF);
    if (containers.empty() || containers.back().key != key) {
      containers.emplace_back().key = key;
    }
    auto& container = containers.back();
    container.values.push_back(value);
    if (!container.runs.empty() && container.runs.back().second + 1 == value) {
      container.runs.back().second = value;
    } else {
      container.runs.emplace_back(value, value);
    }
  }

  const auto runBytes = [](const Container& container) {
    return sizeof(uint16_t) * (1 + 2 * container.runs.size());
  };
  const auto arrayOrBitsetBytes = [](const Container& container) {
    return container.values.size() <= kMaxArrayValues
        ? sizeof(uint16_t) * container.values.size()
        : kBitsetBytes;
  };
  bool hasRunContainers{false};
  for (auto& container : containers) {
    container.isRun = runBytes(container) < arrayOrBitsetBytes(container);
    hasRunContainers = hasRunContainers || container.isRun;
  }

  const auto numContainers = containers.size();
  std::string bitmap;
  if (hasRunContainers) {
    appendInt<uint32_t>(
        bitmap,
        kSerialCookie | (static_cast<uint32_t>(numContainers - 1) << 16),
        /*bigEndian=*/false);
    std::string runFlags((numContainers + 7) / 8, '\0');
    for (size_t i = 0; i < numContainers; ++i) {
      if (containers[i].isRun) {
        runFlags[i / 8] = static_cast<char>(runFlags[i / 8] | (1 << (i % 8)));
      }
    }
    bitmap += runFlags;
  } else {
    appendInt<uint32_t>(bitmap, kSerialCookieNoRun, /*bigEndian=*/false);
    appendInt<uint32_t>(
        bitmap, static_cast<uint32_t>(numContainers), /*bigEndian=*/false);
  }
  for (const auto& container : containers) {
    appendInt<uint16_t>(bitmap, container.key, /*bigEndian=*/false);
    appendInt<uint16_t>(
        bitmap,
        static_cast<uint16_t>(container.values.size() - 1),
        /*bigEndian=*/false);
  }
  // With run containers, the payload offsets are written only from 4
  // containers on.
  if (!hasRunContainers || numContainers >= 4) {
    auto payloadOffset = bitmap.size() + sizeof(uint32_t) * numContainers;
    for (const auto& container : containers) {
      appendInt<uint32_t>(
          bitmap, static_cast<uint32_t>(payloadOffset), /*bigEndian=*/false);
      payloadOffset +=
          container.isRun ? runBytes(container) : arrayOrBitsetBytes(container);
    }
  }
  for (const auto& container : containers) {
    if (container.isRun) {
      appendInt<uint16_t>(
          bitmap,
          static_cast<uint16_t>(container.runs.size()),
          /*bigEndian=*/false);
      for (const auto& [first, last] : container.runs) {
        appendInt<uint16_t>(bitmap, first, /*bigEndian=*/false);
        appendInt<uint16_t>(
            bitmap, static_cast<uint16_t>(last - first), /*bigEndian=*/false);
      }
    } else if (container.values.size() <= kMaxArrayValues) {
      for (const auto value : container.values) {
        appendInt<uint16_t>(bitmap, value, /*bigEndian=*/false);
      }
    } else {
      std::vector<uint64_t> words(kBitsetBytes / sizeof(uint64_t));
      for (const auto value : container.values) {
        bits::setBit(words.data(), value);
      }
      for (const auto word : words) {
        appendInt<uint64_t>(bitmap, word, /*bigEndian=*/false);
      }
    }
  }
  return bitmap;
}

// Serializes 'positions', which increase, as the bitmap of a frame of
// 'version' the way Paimon Java does: a run-optimized 32-bit RoaringBitmap for
// version 1, and for version 2 a Roaring64 bitmap of one such bitmap per group
// of positions that share their high 32 bits.
std::string serializeRunOptimized(
    const std::vector<int64_t>& positions,
    int32_t version) {
  if (version == 1) {
    return serializeRunOptimized32(positions);
  }
  std::vector<std::pair<uint32_t, std::vector<int64_t>>> groups;
  for (const auto position : positions) {
    const auto key = static_cast<uint32_t>(position >> 32);
    if (groups.empty() || groups.back().first != key) {
      groups.emplace_back(key, std::vector<int64_t>{});
    }
    groups.back().second.push_back(position & 0xFFFF'FFFF);
  }
  std::string bitmap;
  appendInt<uint64_t>(bitmap, groups.size(), /*bigEndian=*/false);
  for (const auto& [key, lowPositions] : groups) {
    appendInt<uint32_t>(bitmap, key, /*bigEndian=*/false);
    bitmap += serializeRunOptimized32(lowPositions);
  }
  return bitmap;
}

// Returns random increasing positions for a frame of 'version' in up to 8
// containers. Each container is sparse, dense or made of runs, which gives it
// the array, bitset or run encoding, and its runs may continue into the next
// container. Version 1 positions stay below 2^31 as Paimon Java's do, and
// version 2 containers start next to a Roaring64 group boundary.
std::vector<int64_t> randomPositions(int32_t version, std::mt19937& rng) {
  const auto random = [&](int64_t bound) {
    return static_cast<int64_t>(rng() % bound);
  };
  int64_t containerIndex = version == 1
      ? random(30'000)
      : (1 + random(3)) * kContainerRows - random(3);
  const auto numContainers = 1 + random(8);
  std::vector<int64_t> positions;
  std::vector<uint64_t> selected(bits::nwords(kContainerRows));
  for (int64_t i = 0; i < numContainers; ++i) {
    if (i > 0) {
      containerIndex += 1 + random(2);
    }
    std::fill(selected.begin(), selected.end(), 0);
    switch (random(3)) {
      case 0:
        for (auto numValues = 1 + random(500); numValues > 0; --numValues) {
          bits::setBit(selected.data(), random(kContainerRows));
        }
        break;
      case 1:
        // Selects each row with probability 1/2.
        for (auto& word : selected) {
          const uint64_t highBits = rng();
          word = (highBits << 32) | static_cast<uint32_t>(rng());
        }
        break;
      default:
        for (auto numRuns = 1 + random(16); numRuns > 0; --numRuns) {
          const auto first = random(kContainerRows);
          bits::fillBits(
              selected.data(),
              first,
              std::min(first + 1 + random(3'000), kContainerRows),
              true);
        }
        if (random(2) == 0) {
          bits::fillBits(selected.data(), 0, 1 + random(100), true);
        }
        if (random(2) == 0) {
          bits::fillBits(
              selected.data(),
              kContainerRows - 1 - random(100),
              kContainerRows,
              true);
        }
        break;
    }
    bits::forEachSetBit(selected.data(), 0, kContainerRows, [&](auto row) {
      positions.push_back(containerIndex * kContainerRows + row);
    });
  }
  return positions;
}

// Frames 'bitmap' as Paimon Java does: a big-endian length that counts the
// magic number and the bitmap, the magic number of 'version', the bitmap and a
// CRC. Paimon Java skips the CRC on read, so it is left zero.
std::string frameBitmap(std::string_view bitmap, int32_t version) {
  std::string frame;
  appendInt(
      frame,
      static_cast<uint32_t>(sizeof(uint32_t) + bitmap.size()),
      /*bigEndian=*/true);
  if (version == 1) {
    appendInt(frame, kBitmap32Magic, /*bigEndian=*/true);
  } else {
    appendInt(frame, kBitmap64Magic, /*bigEndian=*/false);
  }
  frame += bitmap;
  appendInt(frame, 0, /*bigEndian=*/true);
  return frame;
}

// Returns the length Paimon records for 'frame' of 'version': the frame length
// field for version 1, which leaves out the length field and the CRC, and the
// whole frame for version 2.
uint64_t deletionFileLength(std::string_view frame, int32_t version) {
  return version == 1 ? frame.size() - 2 * sizeof(uint32_t) : frame.size();
}

class PaimonDeletionVectorReaderTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    filesystems::registerLocalFileSystem();
  }

  // Writes 'data' to a new file and returns its path.
  std::string writeFile(std::string_view data) {
    auto file = TempFilePath::create();
    {
      std::ofstream out(file->getPath(), std::ios::binary);
      out.write(data.data(), static_cast<std::streamsize>(data.size()));
    }
    files_.push_back(file);
    return file->getPath();
  }

  // Writes one index file as Paimon Java does, a version byte and then a frame
  // of 'version' for each entry of 'positions', and returns the deletion file
  // of each frame.
  std::vector<PaimonDeletionFile> writeDeletionVectors(
      const std::vector<std::vector<int64_t>>& positions,
      int32_t version) {
    std::string data(1, '\1');
    std::vector<PaimonDeletionFile> deletionFiles;
    deletionFiles.reserve(positions.size());
    for (const auto& filePositions : positions) {
      const auto frame =
          frameBitmap(serializeBitmap(filePositions, version), version);
      // The path is set once the file is written.
      deletionFiles.emplace_back(
          "",
          data.size(),
          deletionFileLength(frame, version),
          filePositions.size());
      data += frame;
    }
    const auto path = writeFile(data);
    for (auto& deletionFile : deletionFiles) {
      deletionFile.path = path;
    }
    return deletionFiles;
  }

  // Writes an index file that holds the frame of 'bitmap' of 'version' alone
  // and returns its deletion file.
  PaimonDeletionFile writeDeletionVector(
      std::string_view bitmap,
      int32_t version,
      std::optional<uint64_t> cardinality) {
    const auto frame = frameBitmap(bitmap, version);
    return PaimonDeletionFile(
        writeFile(std::string(1, '\1') + frame),
        /*offset=*/1,
        deletionFileLength(frame, version),
        cardinality);
  }

  // Returns the rows among the 'size' rows from row 'begin' that 'reader'
  // marks deleted.
  std::vector<int64_t> readDeletedRows(
      uint64_t begin,
      uint64_t size,
      PaimonDeletionVectorReader& reader) {
    const auto bitmap = AlignedBuffer::allocate<bool>(size, pool_.get(), false);
    reader.readDeletePositions(begin, size, bitmap);
    std::vector<int64_t> rows;
    bits::forEachSetBit(
        bitmap->as<uint64_t>(), 0, static_cast<int32_t>(size), [&](auto row) {
          rows.push_back(static_cast<int64_t>(begin + row));
        });
    return rows;
  }

  // Returns the rows in [begin, end) that 'reader' marks deleted, read in
  // batches of 'batchSize' rows.
  std::vector<int64_t> readDeletedRowsInBatches(
      uint64_t begin,
      uint64_t end,
      uint64_t batchSize,
      PaimonDeletionVectorReader& reader) {
    std::vector<int64_t> rows;
    for (auto batchBegin = begin; batchBegin < end; batchBegin += batchSize) {
      const auto batchRows = readDeletedRows(
          batchBegin, std::min(batchSize, end - batchBegin), reader);
      rows.insert(rows.end(), batchRows.begin(), batchRows.end());
    }
    return rows;
  }

  const std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  const std::shared_ptr<const config::ConfigBase> config_{
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{})};
  std::vector<std::shared_ptr<TempFilePath>> files_;
};

// Paimon packs the deletion vectors of several data files into one index file.
TEST_F(PaimonDeletionVectorReaderTest, frameVersions) {
  for (int32_t version : {1, 2}) {
    SCOPED_TRACE(fmt::format("version {}", version));
    const auto deletionFiles =
        writeDeletionVectors({{1, 3}, {0, 70'000}}, version);
    {
      PaimonDeletionVectorReader reader(
          deletionFiles[0], pool_.get(), config_, /*tokenProvider=*/nullptr);
      EXPECT_THAT(readDeletedRows(0, 8, reader), testing::ElementsAre(1, 3));
    }
    {
      PaimonDeletionVectorReader reader(
          deletionFiles[1], pool_.get(), config_, /*tokenProvider=*/nullptr);
      EXPECT_THAT(
          readDeletedRows(0, 70'001, reader), testing::ElementsAre(0, 70'000));
    }
  }
}

// Paimon Java run-optimizes each bitmap before it writes the frame
// (RoaringBitmap32.serialize, Bitmap64DeletionVector.serializeTo), so runs of
// deleted rows arrive as run containers. A run that crosses a container or a
// Roaring64 group boundary becomes one run per container.
TEST_F(PaimonDeletionVectorReaderTest, runContainers) {
  // Rows 10-19, 100 and 65'530-65'541. The last run spans containers 0 and 1.
  auto rows = rowRange(10, 19);
  rows.push_back(100);
  const auto containerBoundaryRun = rowRange(65'530, 65'541);
  rows.insert(
      rows.end(), containerBoundaryRun.begin(), containerBoundaryRun.end());
  {
    SCOPED_TRACE("version 1");
    PaimonDeletionVectorReader reader(
        writeDeletionVector(
            serializeRunOptimized(rows, /*version=*/1),
            /*version=*/1,
            rows.size()),
        pool_.get(),
        config_,
        /*tokenProvider=*/nullptr);
    EXPECT_THAT(
        readDeletedRowsInBatches(0, 70'000, /*batchSize=*/777, reader),
        testing::ElementsAreArray(rows));
    EXPECT_TRUE(reader.noMoreData());
  }
  {
    SCOPED_TRACE("version 2");
    // Rows 2^32 - 4 to 2^32 + 3 span Roaring64 groups 0 and 1.
    const auto groupBoundaryRun =
        rowRange(kGroupBoundary - 4, kGroupBoundary + 3);
    auto allRows = rows;
    allRows.insert(
        allRows.end(), groupBoundaryRun.begin(), groupBoundaryRun.end());
    PaimonDeletionVectorReader reader(
        writeDeletionVector(
            serializeRunOptimized(allRows, /*version=*/2),
            /*version=*/2,
            allRows.size()),
        pool_.get(),
        config_,
        /*tokenProvider=*/nullptr);
    EXPECT_THAT(
        readDeletedRowsInBatches(0, 70'000, /*batchSize=*/777, reader),
        testing::ElementsAreArray(rows));
    EXPECT_FALSE(reader.noMoreData());
    EXPECT_THAT(
        readDeletedRows(kGroupBoundary - 8, 16, reader),
        testing::ElementsAreArray(groupBoundaryRun));
    EXPECT_TRUE(reader.noMoreData());
  }
}

// More than 4'096 deleted rows in a container make a bitset container. The
// batch size 777 is odd, so the batches start at every bit of a 64-bit word.
TEST_F(PaimonDeletionVectorReaderTest, bitsetContainers) {
  // Every other row of [0, 20'000) and every row of [65'536, 70'536).
  std::vector<int64_t> positions;
  for (int64_t row = 0; row < 20'000; row += 2) {
    positions.push_back(row);
  }
  const auto denseRows = rowRange(65'536, 70'535);
  positions.insert(positions.end(), denseRows.begin(), denseRows.end());
  for (int32_t version : {1, 2}) {
    SCOPED_TRACE(fmt::format("version {}", version));
    PaimonDeletionVectorReader reader(
        writeDeletionVectors({positions}, version)[0],
        pool_.get(),
        config_,
        /*tokenProvider=*/nullptr);
    EXPECT_THAT(
        readDeletedRowsInBatches(0, 80'000, /*batchSize=*/777, reader),
        testing::ElementsAreArray(positions));
    EXPECT_TRUE(reader.noMoreData());
  }
}

// A deletion vector without positions deletes no row. Only a deletion file
// without a cardinality can describe one, since a recorded cardinality must be
// positive.
TEST_F(PaimonDeletionVectorReaderTest, emptyDeletionVector) {
  for (int32_t version : {1, 2}) {
    SCOPED_TRACE(fmt::format("version {}", version));
    PaimonDeletionVectorReader reader(
        writeDeletionVector(
            serializeRunOptimized({}, version),
            version,
            /*cardinality=*/std::nullopt),
        pool_.get(),
        config_,
        /*tokenProvider=*/nullptr);
    EXPECT_FALSE(reader.noMoreData());
    EXPECT_THAT(readDeletedRows(0, 8, reader), testing::IsEmpty());
    EXPECT_TRUE(reader.noMoreData());
  }
}

// Version 2 frames hold positions of 2^32 and above.
TEST_F(PaimonDeletionVectorReaderTest, roaring64Positions) {
  constexpr int64_t kHighPosition = kGroupBoundary + 5;
  const auto deletionFile =
      writeDeletionVectors({{2, kHighPosition}}, /*version=*/2)[0];
  PaimonDeletionVectorReader reader(
      deletionFile, pool_.get(), config_, /*tokenProvider=*/nullptr);
  EXPECT_THAT(readDeletedRows(0, 8, reader), testing::ElementsAre(2));
  EXPECT_FALSE(reader.noMoreData());
  EXPECT_THAT(
      readDeletedRows(kHighPosition - 5, 8, reader),
      testing::ElementsAre(kHighPosition));
  EXPECT_TRUE(reader.noMoreData());
}

// A batch that spans 2^32 reads the end of one Roaring64 group and the start
// of the next.
TEST_F(PaimonDeletionVectorReaderTest, roaring64GroupBoundary) {
  PaimonDeletionVectorReader reader(
      writeDeletionVectors(
          {{kGroupBoundary - 1, kGroupBoundary}}, /*version=*/2)[0],
      pool_.get(),
      config_,
      /*tokenProvider=*/nullptr);
  EXPECT_THAT(
      readDeletedRows(kGroupBoundary - 4, 8, reader),
      testing::ElementsAre(kGroupBoundary - 1, kGroupBoundary));
  EXPECT_TRUE(reader.noMoreData());
}

// Version 1 frames hold a 32-bit RoaringBitmap, whose largest position is
// 2^32 - 1. Paimon Java writes positions up to 2^31 - 1 into them
// (BitmapDeletionVector.checkPosition).
TEST_F(PaimonDeletionVectorReaderTest, version1LargestPositions) {
  constexpr int64_t kJavaMaxPosition = std::numeric_limits<int32_t>::max();
  constexpr int64_t kMaxPosition = std::numeric_limits<uint32_t>::max();
  PaimonDeletionVectorReader reader(
      writeDeletionVectors(
          {{kJavaMaxPosition, kMaxPosition}}, /*version=*/1)[0],
      pool_.get(),
      config_,
      /*tokenProvider=*/nullptr);
  EXPECT_THAT(
      readDeletedRows(kJavaMaxPosition - 7, 8, reader),
      testing::ElementsAre(kJavaMaxPosition));
  EXPECT_FALSE(reader.noMoreData());
  EXPECT_THAT(
      readDeletedRows(kMaxPosition - 7, 8, reader),
      testing::ElementsAre(kMaxPosition));
  EXPECT_TRUE(reader.noMoreData());
}

// Each batch gets the deleted rows of its own range, and noMoreData() turns
// true once a batch passes the last deleted row.
TEST_F(PaimonDeletionVectorReaderTest, batches) {
  const auto deletionFile =
      writeDeletionVectors({{1, 20, 21}}, /*version=*/1)[0];
  PaimonDeletionVectorReader reader(
      deletionFile, pool_.get(), config_, /*tokenProvider=*/nullptr);
  EXPECT_FALSE(reader.noMoreData());
  EXPECT_THAT(readDeletedRows(0, 8, reader), testing::ElementsAre(1));
  EXPECT_FALSE(reader.noMoreData());
  EXPECT_THAT(readDeletedRows(8, 8, reader), testing::IsEmpty());
  EXPECT_FALSE(reader.noMoreData());
  EXPECT_THAT(readDeletedRows(16, 8, reader), testing::ElementsAre(20, 21));
  EXPECT_TRUE(reader.noMoreData());
}

// A batch holds the rows [begin, begin + size), so a deleted row at either end
// of a batch belongs to that batch alone. noMoreData() turns true with the
// batch that holds the last deleted row.
TEST_F(PaimonDeletionVectorReaderTest, batchBoundaries) {
  for (int32_t version : {1, 2}) {
    SCOPED_TRACE(fmt::format("version {}", version));
    PaimonDeletionVectorReader reader(
        writeDeletionVectors({{0, 7, 8, 63, 64, 100}}, version)[0],
        pool_.get(),
        config_,
        /*tokenProvider=*/nullptr);
    EXPECT_THAT(readDeletedRows(0, 1, reader), testing::ElementsAre(0));
    EXPECT_THAT(readDeletedRows(1, 6, reader), testing::IsEmpty());
    EXPECT_THAT(readDeletedRows(7, 2, reader), testing::ElementsAre(7, 8));
    EXPECT_THAT(readDeletedRows(9, 55, reader), testing::ElementsAre(63));
    EXPECT_THAT(readDeletedRows(64, 36, reader), testing::ElementsAre(64));
    EXPECT_FALSE(reader.noMoreData());
    EXPECT_THAT(readDeletedRows(100, 1, reader), testing::ElementsAre(100));
    EXPECT_TRUE(reader.noMoreData());
  }
}

// The split reader skips the rows of the stripes that filters or sampling
// skip, so a batch can start past the end of the previous one. The deleted
// rows in between are never set.
TEST_F(PaimonDeletionVectorReaderTest, skippedRows) {
  for (int32_t version : {1, 2}) {
    SCOPED_TRACE(fmt::format("version {}", version));
    {
      PaimonDeletionVectorReader reader(
          writeDeletionVectors({{5, 50, 51, 52, 1'000, 70'000}}, version)[0],
          pool_.get(),
          config_,
          /*tokenProvider=*/nullptr);
      EXPECT_THAT(readDeletedRows(0, 8, reader), testing::ElementsAre(5));
      // Starts inside the deleted rows 50-52.
      EXPECT_THAT(readDeletedRows(51, 8, reader), testing::ElementsAre(51, 52));
      // Skips row 1'000 and the rest of container 0.
      EXPECT_THAT(
          readDeletedRows(69'996, 8, reader), testing::ElementsAre(70'000));
      EXPECT_TRUE(reader.noMoreData());
    }
    // Skipping past the last deleted row leaves no more data.
    {
      PaimonDeletionVectorReader reader(
          writeDeletionVectors({{5, 1'000}}, version)[0],
          pool_.get(),
          config_,
          /*tokenProvider=*/nullptr);
      EXPECT_THAT(readDeletedRows(0, 8, reader), testing::ElementsAre(5));
      EXPECT_FALSE(reader.noMoreData());
      EXPECT_THAT(readDeletedRows(2'000, 8, reader), testing::IsEmpty());
      EXPECT_TRUE(reader.noMoreData());
    }
  }
}

// Generates random deletion vectors whose containers take every encoding
// Paimon Java writes, reads them in batches of random sizes with random skips,
// and compares each batch with the positions it covers.
TEST_F(PaimonDeletionVectorReaderTest, fuzzer) {
  for (uint32_t seed = 0; seed < 40; ++seed) {
    std::mt19937 rng(seed);
    const auto random = [&](int64_t bound) {
      return static_cast<int64_t>(rng() % bound);
    };
    const auto version = static_cast<int32_t>(1 + random(2));
    const auto positions = randomPositions(version, rng);
    SCOPED_TRACE(
        fmt::format(
            "seed {}, version {}, {} positions",
            seed,
            version,
            positions.size()));
    PaimonDeletionVectorReader reader(
        writeDeletionVector(
            serializeRunOptimized(positions, version),
            version,
            positions.size()),
        pool_.get(),
        config_,
        /*tokenProvider=*/nullptr);

    std::vector<int64_t> deletedRows;
    std::vector<int64_t> expectedRows;
    int64_t row{0};
    while (row <= positions.back()) {
      // Jumps over long gaps, such as the rows before a Roaring64 group, to
      // keep the number of batches small.
      const auto nextPosition =
          *std::lower_bound(positions.begin(), positions.end(), row);
      if (nextPosition - row > 100'000) {
        row = nextPosition - random(1'000);
      }
      if (random(4) == 0) {
        row += random(2'000);
      }
      // Small batches start and end at many bit offsets of a word.
      const auto size = 1 + (random(4) == 0 ? random(64) : random(8'000));
      const auto batchRows = readDeletedRows(row, size, reader);
      deletedRows.insert(deletedRows.end(), batchRows.begin(), batchRows.end());
      expectedRows.insert(
          expectedRows.end(),
          std::lower_bound(positions.begin(), positions.end(), row),
          std::lower_bound(positions.begin(), positions.end(), row + size));
      row += size;
      ASSERT_EQ(reader.noMoreData(), row > positions.back());
    }
    // Compares whole vectors of up to ~260'000 rows, which an element matcher
    // per row would make slow.
    EXPECT_EQ(deletedRows, expectedRows);
  }
}

// Paimon 0.9 records no cardinality for its deletion vectors.
TEST_F(PaimonDeletionVectorReaderTest, withoutCardinality) {
  for (int32_t version : {1, 2}) {
    SCOPED_TRACE(fmt::format("version {}", version));
    auto deletionFile = writeDeletionVectors({{1, 3}}, version)[0];
    deletionFile.cardinality = std::nullopt;
    PaimonDeletionVectorReader reader(
        deletionFile, pool_.get(), config_, /*tokenProvider=*/nullptr);
    EXPECT_THAT(readDeletedRows(0, 8, reader), testing::ElementsAre(1, 3));
  }
}

TEST_F(PaimonDeletionVectorReaderTest, cardinalityMismatch) {
  for (int32_t version : {1, 2}) {
    SCOPED_TRACE(fmt::format("version {}", version));
    auto deletionFile = writeDeletionVectors({{1, 3}}, version)[0];
    deletionFile.cardinality = 3;
    PaimonDeletionVectorReader reader(
        deletionFile, pool_.get(), config_, /*tokenProvider=*/nullptr);
    VELOX_ASSERT_THROW(
        readDeletedRows(0, 8, reader),
        fmt::format(
            "(2 vs. 3) Paimon deletion-vector cardinality mismatch: {}",
            deletionFile.path));
  }
}

// Rejects deletion files beyond what Paimon Java writes before it opens them.
TEST_F(PaimonDeletionVectorReaderTest, sanityLimits) {
  const auto makeReader = [&](uint64_t length, uint64_t cardinality) {
    return PaimonDeletionVectorReader(
        PaimonDeletionFile("deletion.index", /*offset=*/1, length, cardinality),
        pool_.get(),
        config_,
        /*tokenProvider=*/nullptr);
  };
  EXPECT_NO_THROW(makeReader(2'147'483'647, 10'000'000'000));
  VELOX_ASSERT_THROW(
      makeReader(2'147'483'648, 1),
      "Paimon deletion vector length exceeds sanity limit: 2147483648");
  VELOX_ASSERT_THROW(
      makeReader(8, 10'000'000'001),
      "Paimon deletion vector cardinality exceeds sanity limit: 10000000001");
}

// The split reader creates the reader of each data file it opens, and the
// reader opens the index file on its first read.
TEST_F(PaimonDeletionVectorReaderTest, openOnFirstRead) {
  const auto path = TempFilePath::create()->getPath() + ".missing";
  PaimonDeletionVectorReader reader(
      PaimonDeletionFile(
          path, /*offset=*/1, /*length=*/8, /*cardinality=*/std::nullopt),
      pool_.get(),
      config_,
      /*tokenProvider=*/nullptr);
  EXPECT_FALSE(reader.noMoreData());
  VELOX_ASSERT_THROW(
      readDeletedRows(0, 8, reader),
      fmt::format("No such file or directory: {}", path));
}

// The reader checks a frame against its deletion file and the index file
// before it decodes the bitmap.
TEST_F(PaimonDeletionVectorReaderTest, invalidFrames) {
  struct TestParam {
    std::string name;
    // Index file bytes after the version byte.
    std::string frames;
    uint64_t offset;
    uint64_t length;
    std::string error;

    std::string debugString() const {
      return name;
    }
  };

  const auto bitmap32 = serializeBitmap({1}, /*version=*/1);
  const auto bitmap32Frame = frameBitmap(bitmap32, /*version=*/1);
  const auto bitmap32Length = deletionFileLength(bitmap32Frame, /*version=*/1);
  const auto bitmap64Frame =
      frameBitmap(serializeBitmap({1}, /*version=*/2), /*version=*/2);

  // Returns 'frame' with its magic number in the byte order of the other
  // version.
  const auto swapMagicByteOrder = [](std::string frame) {
    std::reverse(
        frame.begin() + sizeof(uint32_t), frame.begin() + 2 * sizeof(uint32_t));
    return frame;
  };

  // A frame of length 8 whose magic number 0x12345678 matches neither version.
  std::string unknownMagic;
  appendInt(unknownMagic, 8, /*bigEndian=*/true);
  appendInt(unknownMagic, 0x12'34'56'78, /*bigEndian=*/true);
  unknownMagic.append(8, '\0');

  // A version 1 frame header whose length of 2 cannot hold the magic number.
  std::string shortFrame;
  appendInt(shortFrame, 2, /*bigEndian=*/true);
  appendInt(shortFrame, kBitmap32Magic, /*bigEndian=*/true);

  // Frames that hold a magic number and no bitmap.
  const auto magicOnly32Frame = frameBitmap("", /*version=*/1);
  const auto magicOnly64Frame = frameBitmap("", /*version=*/2);

  // A version 1 frame whose length leaves out the last 2 bytes of its bitmap,
  // the value of its array container, which the file still holds.
  const auto truncatedBitmapLength = sizeof(uint32_t) + bitmap32.size() - 2;
  std::string truncatedBitmapFrame;
  appendInt(
      truncatedBitmapFrame,
      static_cast<uint32_t>(truncatedBitmapLength),
      /*bigEndian=*/true);
  appendInt(truncatedBitmapFrame, kBitmap32Magic, /*bigEndian=*/true);
  truncatedBitmapFrame += bitmap32;
  appendInt(truncatedBitmapFrame, 0, /*bigEndian=*/true);

  // A version 1 frame of a 32-bit RoaringBitmap with the unknown cookie 1.
  const auto unknownCookieFrame =
      frameBitmap(std::string_view("\1\0\0\0\0\0\0\0", 8), /*version=*/1);

  const std::vector<TestParam> testSettings{
      {
          .name = "length mismatch, version 1",
          .frames = bitmap32Frame,
          .offset = 1,
          .length = bitmap32Length + 1,
          .error =
              "Paimon deletion-vector frame length does not match the deletion file length",
      },
      // A version 2 frame recorded with the version 1 length, which leaves out
      // the length field and the CRC.
      {
          .name = "length mismatch, version 2",
          .frames = bitmap64Frame,
          .offset = 1,
          .length = deletionFileLength(bitmap64Frame, /*version=*/1),
          .error =
              "Paimon deletion-vector frame length does not match the deletion file length",
      },
      {
          .name = "unknown magic number",
          .frames = unknownMagic,
          .offset = 1,
          .length = 8,
          .error =
              "Paimon deletion vector has an unknown magic number: 305419896",
      },
      // Paimon Java writes the version 1 magic number big-endian and the
      // version 2 one little-endian, and accepts neither in the other order.
      // The message shows the magic number read big-endian.
      {
          .name = "version 1 magic number little-endian",
          .frames = swapMagicByteOrder(bitmap32Frame),
          .offset = 1,
          .length = bitmap32Length,
          .error = fmt::format(
              "Paimon deletion vector has an unknown magic number: {}",
              folly::Endian::swap(kBitmap32Magic)),
      },
      {
          .name = "version 2 magic number big-endian",
          .frames = swapMagicByteOrder(bitmap64Frame),
          .offset = 1,
          .length = deletionFileLength(bitmap64Frame, /*version=*/2),
          .error = fmt::format(
              "Paimon deletion vector has an unknown magic number: {}",
              kBitmap64Magic),
      },
      {
          .name = "offset past end of file",
          .frames = bitmap32Frame,
          .offset = bitmap32Frame.size() + 2,
          .length = bitmap32Length,
          .error = fmt::format(
              "Paimon deletion-vector offset {} exceeds file size {}.",
              bitmap32Frame.size() + 2,
              bitmap32Frame.size() + 1),
      },
      {
          .name = "offset at end of file",
          .frames = bitmap32Frame,
          .offset = bitmap32Frame.size() + 1,
          .length = bitmap32Length,
          .error = "Paimon deletion-vector frame header is truncated",
      },
      {
          .name = "truncated frame header",
          .frames = bitmap32Frame.substr(0, 7),
          .offset = 1,
          .length = bitmap32Length,
          .error = "Paimon deletion-vector frame header is truncated",
      },
      {
          .name = "frame shorter than magic number",
          .frames = shortFrame,
          .offset = 1,
          .length = 2,
          .error =
              "Paimon deletion-vector frame is shorter than its magic number",
      },
      {
          .name = "magic number only, version 1",
          .frames = magicOnly32Frame,
          .offset = 1,
          .length = deletionFileLength(magicOnly32Frame, /*version=*/1),
          .error = "32-bit bitmap too small.",
      },
      {
          .name = "magic number only, version 2",
          .frames = magicOnly64Frame,
          .offset = 1,
          .length = deletionFileLength(magicOnly64Frame, /*version=*/2),
          .error = "Truncated Roaring64Bitmap group count.",
      },
      // The file ends one byte before the end of the bitmap, which starts after
      // the version byte, the length and the magic number.
      {
          .name = "bitmap past end of file",
          .frames = bitmap32Frame.substr(0, bitmap32Frame.size() - 5),
          .offset = 1,
          .length = bitmap32Length,
          .error = fmt::format(
              "Paimon deletion-vector range [9, {}) exceeds file size {}.",
              9 + bitmap32.size(),
              bitmap32Frame.size() - 4),
      },
      // The reader decodes the frame alone, not the bytes after it.
      {
          .name = "frame length leaves out the end of the bitmap",
          .frames = truncatedBitmapFrame,
          .offset = 1,
          .length = truncatedBitmapLength,
          .error = "Truncated array container.",
      },
      {
          .name = "corrupt bitmap",
          .frames = unknownCookieFrame,
          .offset = 1,
          .length = deletionFileLength(unknownCookieFrame, /*version=*/1),
          .error = "Unknown roaring bitmap cookie: 1. Expected 12346 or 12347.",
      },
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const auto path = writeFile(std::string(1, '\1') + testData.frames);
    PaimonDeletionVectorReader reader(
        PaimonDeletionFile(
            path,
            testData.offset,
            testData.length,
            /*cardinality=*/std::nullopt),
        pool_.get(),
        config_,
        /*tokenProvider=*/nullptr);
    VELOX_ASSERT_THROW(readDeletedRows(0, 8, reader), testData.error);
  }
}

} // namespace
} // namespace facebook::velox::connector::paimon
