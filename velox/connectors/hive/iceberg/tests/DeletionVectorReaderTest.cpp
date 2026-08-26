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

#include "velox/connectors/hive/iceberg/DeletionVectorFormat.h"

#include <fstream>

#include <folly/json.h>
#include <folly/lang/Bits.h>
#include <gtest/gtest.h>
#include <zlib.h>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempFilePath.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive::iceberg;
using namespace facebook::velox::common::testutil;

namespace {

// Serializes a roaring bitmap in the portable format (no-run variant,
// cookie = 12346). Supports only array containers (cardinality <= 4096).
// This is the simplest format the DeletionVectorReader needs to parse.
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
  constexpr uint32_t cookie = 12346;
  data.append(reinterpret_cast<const char*>(&cookie), 4);
  // Container count.
  data.append(reinterpret_cast<const char*>(&numContainers), 4);

  // Key-cardinality pairs.
  for (auto& [key, vals] : containers) {
    uint16_t cardMinus1 = static_cast<uint16_t>(vals.size() - 1);
    data.append(reinterpret_cast<const char*>(&key), 2);
    data.append(reinterpret_cast<const char*>(&cardMinus1), 2);
  }

  // Offset section (required for > 0 containers)
  if (numContainers > 0) {
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

// Serializes a roaring bitmap in the portable no-run format, encoding any
// block whose cardinality exceeds 4096 as a 1024-word bitset container rather
// than an array container. 'serializeRoaringBitmapNoRun' above always emits
// array containers, so it cannot produce this encoding — which is the one
// Iceberg and DeletionVectorWriter use for dense blocks.
std::string serializeRoaringBitmapWithBitsetContainer(
    const std::vector<int64_t>& positions) {
  constexpr uint32_t kMaxArrayContainerCardinality = 4'096;
  constexpr uint32_t kBitmapContainerBytes = 8'192;
  constexpr uint32_t kCookie = 12'346;

  std::map<uint16_t, std::vector<uint16_t>> containers;
  for (auto position : positions) {
    containers[static_cast<uint16_t>(position >> 16)].push_back(
        static_cast<uint16_t>(position & 0xFFFF));
  }
  for (auto& [key, values] : containers) {
    std::sort(values.begin(), values.end());
  }

  const auto numContainers = static_cast<uint32_t>(containers.size());
  std::string data;
  data.append(reinterpret_cast<const char*>(&kCookie), 4);
  data.append(reinterpret_cast<const char*>(&numContainers), 4);

  for (const auto& [key, values] : containers) {
    const auto cardMinus1 = static_cast<uint16_t>(values.size() - 1);
    data.append(reinterpret_cast<const char*>(&key), 2);
    data.append(reinterpret_cast<const char*>(&cardMinus1), 2);
  }

  uint32_t offset = 4 + 4 + numContainers * 4 + numContainers * 4;
  for (const auto& [key, values] : containers) {
    data.append(reinterpret_cast<const char*>(&offset), 4);
    offset += values.size() <= kMaxArrayContainerCardinality
        ? static_cast<uint32_t>(values.size()) * 2
        : kBitmapContainerBytes;
  }

  for (const auto& [key, values] : containers) {
    if (values.size() <= kMaxArrayContainerCardinality) {
      for (auto value : values) {
        data.append(reinterpret_cast<const char*>(&value), 2);
      }
    } else {
      std::vector<uint64_t> words(1'024, 0);
      for (auto value : values) {
        words[value / 64] |= (1ULL << (value % 64));
      }
      for (auto word : words) {
        data.append(reinterpret_cast<const char*>(&word), 8);
      }
    }
  }
  return data;
}

// Wraps serialized 32-bit roaring bitmaps in the Roaring64 envelope:
// [numGroups: uint64] then, per group, [highBits: uint32][32-bit bitmap].
// Reaching group N+1 requires the reader to advance exactly past group N's
// container data, so multi-group inputs exercise that skip arithmetic.
std::string wrapInRoaring64(
    const std::vector<std::pair<uint32_t, std::string>>& groups) {
  std::string data;
  const auto numGroups = static_cast<uint64_t>(groups.size());
  data.append(reinterpret_cast<const char*>(&numGroups), 8);
  for (const auto& [highBits, bitmap] : groups) {
    data.append(reinterpret_cast<const char*>(&highBits), 4);
    data.append(bitmap);
  }
  return data;
}

// Serializes a roaring bitmap whose containers may each use a different
// encoding, which none of the helpers above can express: the no-run helpers
// pick array vs bitset purely by cardinality, and the run helper makes every
// container run-encoded.
//
// Iceberg's own test suite carries an "all container types" case; this builds
// an equivalent bitmap from scratch so the reader is exercised against array,
// run, and bitset containers coexisting in one blob.
struct ContainerSpec {
  enum class Encoding { kArray, kRun, kBitset };

  uint16_t key{0};
  Encoding encoding{Encoding::kArray};
  // Values within the container's 64K block, sorted. For kRun these are still
  // the expanded values; the runs are derived from them.
  std::vector<uint16_t> values;
};

// Collapses sorted values into (start, lengthMinus1) runs.
std::vector<std::pair<uint16_t, uint16_t>> toRuns(
    const std::vector<uint16_t>& values) {
  std::vector<std::pair<uint16_t, uint16_t>> runs;
  for (size_t i = 0; i < values.size();) {
    size_t j = i;
    while (j + 1 < values.size() && values[j + 1] == values[j] + 1) {
      ++j;
    }
    runs.emplace_back(values[i], static_cast<uint16_t>(j - i));
    i = j + 1;
  }
  return runs;
}

std::string serializeRoaringBitmapMixed(
    const std::vector<ContainerSpec>& containers) {
  constexpr uint32_t kSerialCookieWithRuns = 12'347;
  constexpr uint32_t kRunContainersNoOffsetThreshold = 4;
  constexpr size_t kBitmapContainerBytes = 8'192;

  const auto numContainers = static_cast<uint32_t>(containers.size());
  // This helper always emits the run-bearing cookie and a run bitmap, even
  // when no container is actually run-encoded, so readers always take the
  // "runs present" branch. The offset section must follow the same rule or the
  // two sides disagree on the header size.
  const bool hasRunContainers = true;

  std::string data;
  const uint32_t cookie = kSerialCookieWithRuns | ((numContainers - 1) << 16);
  data.append(reinterpret_cast<const char*>(&cookie), 4);

  // Run bitmap: bit i marks container i as run-encoded, LSB-first.
  const uint32_t runBitmapBytes = (numContainers + 7) / 8;
  std::vector<uint8_t> runBitmap(runBitmapBytes, 0);
  for (uint32_t i = 0; i < numContainers; ++i) {
    if (containers[i].encoding == ContainerSpec::Encoding::kRun) {
      runBitmap[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    }
  }
  data.append(reinterpret_cast<const char*>(runBitmap.data()), runBitmapBytes);

  for (const auto& spec : containers) {
    const auto cardMinus1 = static_cast<uint16_t>(spec.values.size() - 1);
    data.append(reinterpret_cast<const char*>(&spec.key), 2);
    data.append(reinterpret_cast<const char*>(&cardMinus1), 2);
  }

  const auto containerBytes = [&](const ContainerSpec& spec) -> uint32_t {
    switch (spec.encoding) {
      case ContainerSpec::Encoding::kArray:
        return static_cast<uint32_t>(spec.values.size()) * 2;
      case ContainerSpec::Encoding::kBitset:
        return kBitmapContainerBytes;
      case ContainerSpec::Encoding::kRun:
        return 2 + 4 * static_cast<uint32_t>(toRuns(spec.values).size());
    }
    return 0;
  };

  // The offset section is omitted for run-bearing bitmaps below the threshold.
  const bool hasOffsetSection =
      !hasRunContainers || numContainers >= kRunContainersNoOffsetThreshold;
  if (hasOffsetSection) {
    uint32_t offset =
        4 + runBitmapBytes + 4 * numContainers + 4 * numContainers;
    for (const auto& spec : containers) {
      data.append(reinterpret_cast<const char*>(&offset), 4);
      offset += containerBytes(spec);
    }
  }

  for (const auto& spec : containers) {
    switch (spec.encoding) {
      case ContainerSpec::Encoding::kArray:
        for (auto value : spec.values) {
          data.append(reinterpret_cast<const char*>(&value), 2);
        }
        break;
      case ContainerSpec::Encoding::kRun: {
        const auto runs = toRuns(spec.values);
        const auto numRuns = static_cast<uint16_t>(runs.size());
        data.append(reinterpret_cast<const char*>(&numRuns), 2);
        for (auto [start, lengthMinus1] : runs) {
          data.append(reinterpret_cast<const char*>(&start), 2);
          data.append(reinterpret_cast<const char*>(&lengthMinus1), 2);
        }
        break;
      }
      case ContainerSpec::Encoding::kBitset: {
        std::vector<uint64_t> words(1'024, 0);
        for (auto value : spec.values) {
          words[value / 64] |= (1ULL << (value % 64));
        }
        for (auto word : words) {
          data.append(reinterpret_cast<const char*>(&word), 8);
        }
        break;
      }
    }
  }
  return data;
}

// Concatenates the deletion-vector-v1 magic bytes with a serialized roaring
// bitmap. The frame's length prefix and CRC-32 both cover these bytes.
std::string magicAndVectorBytes(const std::string& bitmap) {
  std::string bytes;
  bytes.append(kDeletionVectorMagic, kDeletionVectorMagicSize);
  bytes.append(bitmap);
  return bytes;
}

// Wraps a serialized roaring bitmap in the Iceberg deletion-vector-v1 blob
// frame: [length: 4B BE][magic D1 D3 39 64][bitmap][CRC-32: 4B BE], where the
// length and CRC-32 cover magic + bitmap. 'crc' is the checksum stored in the
// trailing 4 bytes; tests pass it explicitly so they can exercise both the
// spec-compliant CRC and a deliberately wrong one.
std::string frameDeletionVectorV1(const std::string& bitmap, uint32_t crc) {
  const std::string magicAndVector = magicAndVectorBytes(bitmap);

  auto appendBigEndian32 = [](std::string& out, uint32_t value) {
    const char bytes[4] = {
        static_cast<char>((value >> 24) & 0xFF),
        static_cast<char>((value >> 16) & 0xFF),
        static_cast<char>((value >> 8) & 0xFF),
        static_cast<char>(value & 0xFF)};
    out.append(bytes, sizeof(bytes));
  };

  std::string framed;
  framed.reserve(sizeof(uint32_t) + magicAndVector.size() + sizeof(uint32_t));
  appendBigEndian32(framed, static_cast<uint32_t>(magicAndVector.size()));
  framed.append(magicAndVector);
  appendBigEndian32(framed, crc);
  return framed;
}

// Standard CRC-32 (IEEE 802.3, as computed by java.util.zip.CRC32 used by
// Iceberg). zlib's crc32 is the reference implementation of that algorithm.
uint32_t standardCrc32(const std::string& data) {
  uLong crc = crc32(0L, Z_NULL, 0);
  crc = crc32(
      crc,
      reinterpret_cast<const Bytef*>(data.data()),
      static_cast<uInt>(data.size()));
  return static_cast<uint32_t>(crc);
}
// (cookie = 12347). All containers are run-encoded.
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

  // Offset section (required for >= 4)
  constexpr uint32_t kRunContainersNoOffsetThreshold = 4;
  if (numContainers >= kRunContainersNoOffsetThreshold) {
    // First container offset = cookie (4) + runBitmap (runBitmapBytes)
    // + descriptive header (4 * numContainers) + offset header
    // (4 * numContainers).
    uint32_t offset =
        4 + runBitmapBytes + 4 * numContainers + 4 * numContainers;
    for (auto& [key, runs] : containerRuns) {
      data.append(reinterpret_cast<const char*>(&offset), 4);
      // Each run container occupies 2 + 4 * numRuns bytes.
      offset += 2 + 4 * static_cast<uint32_t>(runs.size());
    }
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

// Expands a list of run-encoded containers into the full set of positions
// they represent. Each container is keyed by its high 16 bits and contains
// runs of (start, lengthMinus1)
std::vector<uint64_t> expandRuns(
    const std::vector<
        std::pair<uint16_t, std::vector<std::pair<uint16_t, uint16_t>>>>&
        containerRuns) {
  std::vector<uint64_t> result;
  for (const auto& [key, runs] : containerRuns) {
    const uint64_t base = static_cast<uint64_t>(key) * 65536;
    for (const auto& [start, lengthMinus1] : runs) {
      for (uint64_t i = 0; i <= lengthMinus1; ++i) {
        result.push_back(base + start + i);
      }
    }
  }
  return result;
}

// Writes binary data to a temp file and returns the path.
// Builds a Puffin file around 'blob'. Layout per the Puffin spec:
//   Magic | blob | Magic | footerPayload | payloadSize(4B LE) | flags(4B LE)
//   | Magic
// Built by hand rather than via writePuffinFile so the tests can produce
// footers a spec-compliant writer never would.
constexpr std::string_view kPuffinMagic{"PFA1"};

std::string wrapInPuffinFile(
    const std::string& blob,
    const folly::dynamic& footer,
    uint32_t flags = 0) {
  const auto appendLittleEndian32 = [](std::string& out, uint32_t value) {
    const uint32_t little = folly::Endian::little(value);
    out.append(reinterpret_cast<const char*>(&little), sizeof(little));
  };

  const std::string footerJson = folly::toJson(footer);
  std::string file;
  file.append(kPuffinMagic);
  file.append(blob);
  file.append(kPuffinMagic);
  file.append(footerJson);
  appendLittleEndian32(file, static_cast<uint32_t>(footerJson.size()));
  appendLittleEndian32(file, flags);
  file.append(kPuffinMagic);
  return file;
}

// Builds a Puffin footer describing one deletion-vector-v1 blob.
folly::dynamic makeDvBlobMeta(
    uint64_t blobOffset,
    uint64_t blobLength,
    const std::string& referencedDataFile = "") {
  folly::dynamic blobMeta =
      folly::dynamic::object("type", "deletion-vector-v1")(
          "fields", folly::dynamic::array(2'147'483'645));
  blobMeta["snapshot-id"] = -1;
  blobMeta["sequence-number"] = -1;
  blobMeta["offset"] = blobOffset;
  blobMeta["length"] = blobLength;
  if (!referencedDataFile.empty()) {
    blobMeta["properties"] =
        folly::dynamic::object("referenced-data-file", referencedDataFile);
  }
  return blobMeta;
}

folly::dynamic makeDvFooter(
    uint64_t blobOffset,
    uint64_t blobLength,
    const std::string& referencedDataFile = "") {
  return folly::dynamic::object(
      "blobs",
      folly::dynamic::array(
          makeDvBlobMeta(blobOffset, blobLength, referencedDataFile)));
}

// Creates a DV delete file that carries no blob location at all: no typed
// contentOffset/contentLength and no legacy bounds map. The reader must fall
// back to the Puffin footer.
IcebergDeleteFile makeFooterLocatedDvFile(
    const std::string& filePath,
    uint64_t recordCount,
    uint64_t fileSize,
    const std::string& referencedDataFile = "") {
  IcebergDeleteFile dvFile(
      FileContent::kDeletionVector,
      filePath,
      dwio::common::FileFormat::DWRF,
      recordCount,
      fileSize,
      /*equalityFieldIds=*/{},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/0,
      /*contentOffset=*/0,
      /*contentLength=*/0);
  dvFile.referencedDataFile = referencedDataFile;
  return dvFile;
}

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

// Creates an IcebergDeleteFile for a deletion vector. Uses the typed
/// 'contentOffset' / 'contentLength' fields rather than the legacy bounds-map
/// encoding.
IcebergDeleteFile makeDvDeleteFile(
    const std::string& filePath,
    uint64_t recordCount,
    uint64_t fileSize,
    uint64_t blobOffset = 0,
    std::optional<uint64_t> blobLength = std::nullopt) {
  const int64_t contentLength = blobLength.has_value()
      ? static_cast<int64_t>(blobLength.value())
      : static_cast<int64_t>(fileSize);

  return IcebergDeleteFile(
      FileContent::kDeletionVector,
      filePath,
      dwio::common::FileFormat::DWRF,
      recordCount,
      fileSize,
      /*equalityFieldIds=*/{},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/0,
      /*contentOffset=*/static_cast<int64_t>(blobOffset),
      /*contentLength=*/contentLength);
}

// Creates a DV delete file that locates its blob through the legacy
// bounds-map encoding instead of the typed 'contentOffset'/'contentLength'
// fields. Leaving 'contentLength' at 0 is what selects that fallback.
IcebergDeleteFile makeLegacyBoundsDvFile(
    const std::string& filePath,
    uint64_t recordCount,
    uint64_t fileSize,
    const std::optional<std::string>& blobOffset,
    const std::optional<std::string>& blobLength) {
  std::unordered_map<int32_t, std::string> lowerBounds;
  std::unordered_map<int32_t, std::string> upperBounds;
  if (blobOffset.has_value()) {
    lowerBounds[DeletionVectorReader::kDvOffsetFieldId] = *blobOffset;
  }
  if (blobLength.has_value()) {
    upperBounds[DeletionVectorReader::kDvLengthFieldId] = *blobLength;
  }

  return IcebergDeleteFile(
      FileContent::kDeletionVector,
      filePath,
      dwio::common::FileFormat::DWRF,
      recordCount,
      fileSize,
      /*equalityFieldIds=*/{},
      lowerBounds,
      upperBounds,
      /*dataSequenceNumber=*/0,
      /*contentOffset=*/0,
      /*contentLength=*/0);
}

// Extracts which bits are set in a bitmap buffer.
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

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  EXPECT_FALSE(reader.noMoreData());

  auto bitmap = allocateBitmap(100);
  reader.readDeletePositions(0, 100, bitmap);

  auto setBits = getSetBits(bitmap, 100);
  EXPECT_EQ(setBits, (std::vector<uint64_t>{0, 5, 10, 99}));
  EXPECT_TRUE(reader.noMoreData());
}

// Anchors standardCrc32() to the canonical CRC-32 test vector so the framed-DV
// tests below assert against the true Iceberg (java.util.zip.CRC32) checksum,
// not whatever the reader happens to compute.
TEST_F(DeletionVectorReaderTest, standardCrc32TestVector) {
  EXPECT_EQ(standardCrc32("123456789"), 0xCBF43926u);
}

// A deletion-vector-v1 blob written by Iceberg/Spark stores the standard
// CRC-32 (java.util.zip.CRC32) over magic + bitmap. The reader must validate
// against that same checksum and parse the inner bitmap.
TEST_F(DeletionVectorReaderTest, framedBlobStandardCrc) {
  std::vector<int64_t> positions = {0, 5, 10, 99};
  const auto bitmap = serializeRoaringBitmapNoRun(positions);
  const uint32_t crc = standardCrc32(magicAndVectorBytes(bitmap));
  const auto framed = frameDeletionVectorV1(bitmap, crc);

  auto tempFile = writeDvFile(framed);
  auto dvFile = makeDvDeleteFile(
      tempFile->getPath(), positions.size(), framed.size(), 0, framed.size());

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  auto bitmapBuffer = allocateBitmap(100);
  reader.readDeletePositions(0, 100, bitmapBuffer);

  auto setBits = getSetBits(bitmapBuffer, 100);
  EXPECT_EQ(setBits, (std::vector<uint64_t>{0, 5, 10, 99}));
}

// A frame carrying the old un-inverted checksum (the standard CRC-32 without
// the final one's-complement — the bug this test guards against) must be
// rejected: it is not the CRC-32 Iceberg stores.
TEST_F(DeletionVectorReaderTest, framedBlobRejectsUninvertedCrc) {
  std::vector<int64_t> positions = {0, 5, 10, 99};
  const auto bitmap = serializeRoaringBitmapNoRun(positions);
  const auto magicAndVector = magicAndVectorBytes(bitmap);
  const uint32_t uninvertedCrc = ~standardCrc32(magicAndVector);
  const auto framed = frameDeletionVectorV1(bitmap, uninvertedCrc);

  auto tempFile = writeDvFile(framed);
  auto dvFile = makeDvDeleteFile(
      tempFile->getPath(), positions.size(), framed.size(), 0, framed.size());

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  auto bitmapBuffer = allocateBitmap(100);
  VELOX_ASSERT_THROW(
      reader.readDeletePositions(0, 100, bitmapBuffer),
      "Deletion-vector-v1 CRC-32 mismatch");
}

TEST_F(DeletionVectorReaderTest, batchRangeFiltering) {
  // Positions: 10, 20, 30, 40, 50.
  std::vector<int64_t> positions = {10, 20, 30, 40, 50};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile =
      makeDvDeleteFile(tempFile->getPath(), positions.size(), fileSize);

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

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
  DeletionVectorReader reader(dvFile, 100, pool_.get(), nullptr);

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
  DeletionVectorReader reader(dvFile, 100, pool_.get(), nullptr);

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

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

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

  auto expected = expandRuns(containerRuns);
  auto dvFile =
      makeDvDeleteFile(tempFile->getPath(), expected.size(), fileSize);

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

  auto bitmap = allocateBitmap(100);
  reader.readDeletePositions(0, 100, bitmap);

  auto setBits = getSetBits(bitmap, 100);
  EXPECT_EQ(setBits, expected);
  EXPECT_TRUE(reader.noMoreData());
}

TEST_F(DeletionVectorReaderTest, runContainersWithOffsetHeader) {
  std::vector<std::pair<uint16_t, std::vector<std::pair<uint16_t, uint16_t>>>>
      containerRuns = {
          {0, {{10, 4}}}, // positions 10-14
          {1, {{0, 2}, {100, 1}}}, // positions 65536-65538, 65636-65637
          {2, {{500, 0}}}, // position 131072+500 = 131572
          {3, {{1000, 9}}}, // positions 196608+1000..196608+1009
      };
  auto bitmapData = serializeRoaringBitmapWithRuns(containerRuns);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto expected = expandRuns(containerRuns);
  auto dvFile =
      makeDvDeleteFile(tempFile->getPath(), expected.size(), fileSize);

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

  const uint64_t numRows = 200'000;
  auto bitmap = allocateBitmap(numRows);
  reader.readDeletePositions(0, numRows, bitmap);

  auto setBits = getSetBits(bitmap, numRows);
  EXPECT_EQ(setBits, expected);
  EXPECT_TRUE(reader.noMoreData());
}

TEST_F(DeletionVectorReaderTest, bitsetContainer) {
  // 5000 deletes inside a single 64K block exceeds the 4096 array-container
  // threshold, so the block is stored as a 1024-word bitset. The reader picks
  // the container encoding from the cardinality, so this is the only shape
  // that exercises its bitset branch.
  std::vector<int64_t> positions;
  positions.reserve(5'000);
  for (int64_t i = 0; i < 5'000; ++i) {
    positions.push_back(i * 2);
  }

  auto bitmapData = serializeRoaringBitmapWithBitsetContainer(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto dvFile = makeDvDeleteFile(
      tempFile->getPath(),
      positions.size(),
      static_cast<uint64_t>(bitmapData.size()));

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

  const uint64_t numRows = 10'000;
  auto bitmap = allocateBitmap(numRows);
  reader.readDeletePositions(0, numRows, bitmap);

  std::vector<uint64_t> expected;
  expected.reserve(positions.size());
  for (auto position : positions) {
    expected.push_back(static_cast<uint64_t>(position));
  }
  EXPECT_EQ(getSetBits(bitmap, numRows), expected);
  EXPECT_TRUE(reader.noMoreData());
}

TEST_F(DeletionVectorReaderTest, bitsetContainerMixedWithArrayContainer) {
  // A dense block followed by a sparse one. The reader must advance exactly
  // 8192 bytes past the bitset before reading the array container, so a wrong
  // stride here corrupts the second block rather than failing outright.
  std::vector<int64_t> positions;
  positions.reserve(4'100);
  for (int64_t i = 0; i < 4'100; ++i) {
    positions.push_back(i);
  }
  positions.push_back(65'536 + 7);
  positions.push_back(65'536 + 9);

  auto bitmapData = serializeRoaringBitmapWithBitsetContainer(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto dvFile = makeDvDeleteFile(
      tempFile->getPath(),
      positions.size(),
      static_cast<uint64_t>(bitmapData.size()));

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

  std::vector<int64_t> expected(positions.begin(), positions.end());
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(reader.deletedPositions(), expected);
}

TEST_F(DeletionVectorReaderTest, runContainersAcrossRoaring64Groups) {
  // Run containers wrapped in a Roaring64 envelope. Existing run-container
  // coverage uses the bare 32-bit format, which returns before the 64-bit
  // group loop; only a multi-group input exercises the logic that skips past
  // a run container to locate the next group's header.
  const std::vector<
      std::pair<uint16_t, std::vector<std::pair<uint16_t, uint16_t>>>>
      lowGroupRuns = {{0, {{10, 4}, {100, 2}}}};
  const std::vector<
      std::pair<uint16_t, std::vector<std::pair<uint16_t, uint16_t>>>>
      highGroupRuns = {{0, {{20, 1}}}};

  auto bitmapData = wrapInRoaring64(
      {{0, serializeRoaringBitmapWithRuns(lowGroupRuns)},
       {1, serializeRoaringBitmapWithRuns(highGroupRuns)}});
  auto tempFile = writeDvFile(bitmapData);

  std::vector<int64_t> expected;
  for (auto position : expandRuns(lowGroupRuns)) {
    expected.push_back(static_cast<int64_t>(position));
  }
  // Group 1 positions live at highBits 1, i.e. offset 2^32.
  constexpr int64_t kHighGroupBase = int64_t{1} << 32;
  for (auto position : expandRuns(highGroupRuns)) {
    expected.push_back(kHighGroupBase + static_cast<int64_t>(position));
  }
  std::sort(expected.begin(), expected.end());

  auto dvFile = makeDvDeleteFile(
      tempFile->getPath(),
      expected.size(),
      static_cast<uint64_t>(bitmapData.size()));

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  EXPECT_EQ(reader.deletedPositions(), expected);
}

TEST_F(
    DeletionVectorReaderTest,
    arrayAndBitsetContainersAcrossRoaring64Groups) {
  // The multi-group skip arithmetic differs per container encoding: an array
  // container advances by 2 bytes per value while a bitset always advances by
  // a fixed 8192. Pair a sparse group with a dense one so both strides must be
  // right for the second group's header to be found.
  std::vector<int64_t> denseGroupPositions;
  denseGroupPositions.reserve(4'200);
  for (int64_t i = 0; i < 4'200; ++i) {
    denseGroupPositions.push_back(i);
  }

  auto bitmapData = wrapInRoaring64(
      {{0, serializeRoaringBitmapWithBitsetContainer({3, 11, 65'536 + 4})},
       {1, serializeRoaringBitmapWithBitsetContainer(denseGroupPositions)}});
  auto tempFile = writeDvFile(bitmapData);

  constexpr int64_t kHighGroupBase = int64_t{1} << 32;
  std::vector<int64_t> expected = {3, 11, 65'536 + 4};
  for (auto position : denseGroupPositions) {
    expected.push_back(kHighGroupBase + position);
  }
  std::sort(expected.begin(), expected.end());

  auto dvFile = makeDvDeleteFile(
      tempFile->getPath(),
      expected.size(),
      static_cast<uint64_t>(bitmapData.size()));

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  EXPECT_EQ(reader.deletedPositions(), expected);
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

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

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

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

  auto bitmap = allocateBitmap(20);
  reader.readDeletePositions(0, 20, bitmap);

  auto setBits = getSetBits(bitmap, 20);
  EXPECT_EQ(setBits, (std::vector<uint64_t>{3, 7, 11}));
  EXPECT_TRUE(reader.noMoreData());
}

TEST_F(DeletionVectorReaderTest, locatesBlobFromPuffinFooterWhenOffsetsAbsent) {
  // With no typed contentOffset/contentLength and no legacy bounds map, the
  // reader used to fall back to treating the whole file -- leading magic and
  // footer included -- as the blob, which cannot parse. A Puffin file is
  // self-describing, so its footer is the authoritative fallback.
  const std::vector<int64_t> positions = {3, 7, 42, 100};
  const auto blob = serializeRoaringBitmapNoRun(positions);
  const auto file = wrapInPuffinFile(blob, makeDvFooter(4, blob.size()));
  auto tempFile = writeDvFile(file);

  auto dvFile = makeFooterLocatedDvFile(
      tempFile->getPath(), positions.size(), file.size());

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  EXPECT_EQ(reader.deletedPositions(), positions);
}

TEST_F(DeletionVectorReaderTest, puffinFooterSelectsBlobByReferencedDataFile) {
  // One Puffin file can hold vectors for several data files. The referenced
  // data file, not blob order, decides which one applies to this split.
  const std::vector<int64_t> wantedPositions = {5, 9};
  const std::vector<int64_t> otherPositions = {1, 2, 3};
  const auto otherBlob = serializeRoaringBitmapNoRun(otherPositions);
  const auto wantedBlob = serializeRoaringBitmapNoRun(wantedPositions);

  const uint64_t otherOffset = 4;
  const uint64_t wantedOffset = otherOffset + otherBlob.size();
  folly::dynamic footer = folly::dynamic::object(
      "blobs",
      folly::dynamic::array(
          makeDvBlobMeta(otherOffset, otherBlob.size(), "/data/other.parquet"),
          makeDvBlobMeta(
              wantedOffset, wantedBlob.size(), "/data/wanted.parquet")));

  const auto file = wrapInPuffinFile(otherBlob + wantedBlob, footer);
  auto tempFile = writeDvFile(file);

  auto dvFile = makeFooterLocatedDvFile(
      tempFile->getPath(),
      wantedPositions.size(),
      file.size(),
      "/data/wanted.parquet");

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  EXPECT_EQ(reader.deletedPositions(), wantedPositions);
}

TEST_F(DeletionVectorReaderTest, rejectsAmbiguousPuffinFileWithoutReference) {
  // Two candidate vectors and nothing to choose between them: picking either
  // would silently apply the wrong deletes to the scan.
  const auto blob = serializeRoaringBitmapNoRun({1});
  folly::dynamic footer = folly::dynamic::object(
      "blobs",
      folly::dynamic::array(
          makeDvBlobMeta(4, blob.size()), makeDvBlobMeta(4, blob.size())));

  const auto file = wrapInPuffinFile(blob, footer);
  auto tempFile = writeDvFile(file);
  auto dvFile = makeFooterLocatedDvFile(tempFile->getPath(), 1, file.size());

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  VELOX_ASSERT_THROW(
      reader.deletedPositions(),
      "Puffin file has multiple deletion vectors and no referenced data file");
}

TEST_F(DeletionVectorReaderTest, rejectsPuffinFooterWithNoMatchingBlob) {
  const auto blob = serializeRoaringBitmapNoRun({1});
  const auto file = wrapInPuffinFile(
      blob, makeDvFooter(4, blob.size(), "/data/other.parquet"));
  auto tempFile = writeDvFile(file);
  auto dvFile = makeFooterLocatedDvFile(
      tempFile->getPath(), 1, file.size(), "/data/wanted.parquet");

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  VELOX_ASSERT_THROW(
      reader.deletedPositions(),
      "Puffin footer has no deletion-vector blob for data file");
}

TEST_F(DeletionVectorReaderTest, rejectsPuffinFooterWithNoBlobs) {
  // A structurally valid footer that declares no blobs at all. Iceberg allows
  // an empty Puffin file; for a delete file it means there is nothing to read,
  // which must be an error rather than an empty (silently no-op) vector.
  const auto blob = serializeRoaringBitmapNoRun({1});
  const auto file = wrapInPuffinFile(
      blob, folly::dynamic::object("blobs", folly::dynamic::array()));
  auto tempFile = writeDvFile(file);
  auto dvFile = makeFooterLocatedDvFile(tempFile->getPath(), 1, file.size());

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  VELOX_ASSERT_THROW(
      reader.deletedPositions(),
      "Puffin footer has no deletion-vector blob for data file");
}

TEST_F(DeletionVectorReaderTest, rejectsPuffinFooterWithNegativeOffset) {
  // JSON integers are signed, but the blob location is carried as uint64_t.
  // A negative offset must be rejected where it is read, not allowed to wrap
  // into a huge unsigned value and be reported as an absurd out-of-range read.
  const auto blob = serializeRoaringBitmapNoRun({1});
  auto footer = makeDvFooter(4, blob.size());
  footer["blobs"][0]["offset"] = -8;
  const auto file = wrapInPuffinFile(blob, footer);
  auto tempFile = writeDvFile(file);
  auto dvFile = makeFooterLocatedDvFile(tempFile->getPath(), 1, file.size());

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  VELOX_ASSERT_THROW(
      reader.deletedPositions(), "Puffin blob metadata has a negative offset");
}

TEST_F(DeletionVectorReaderTest, rejectsPuffinFileWithoutTrailingMagic) {
  const auto blob = serializeRoaringBitmapNoRun({1});
  auto file = wrapInPuffinFile(blob, makeDvFooter(4, blob.size()));
  file.replace(file.size() - 4, 4, "XXXX");

  auto tempFile = writeDvFile(file);
  auto dvFile = makeFooterLocatedDvFile(tempFile->getPath(), 1, file.size());

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  VELOX_ASSERT_THROW(
      reader.deletedPositions(),
      "Puffin file does not end with the expected magic");
}

TEST_F(DeletionVectorReaderTest, rejectsPuffinFooterPayloadSizeBeyondFile) {
  // A payload size larger than the file would make the payload offset
  // underflow into a huge unsigned read.
  const auto blob = serializeRoaringBitmapNoRun({1});
  auto file = wrapInPuffinFile(blob, makeDvFooter(4, blob.size()));
  const uint32_t absurdSize = folly::Endian::little(uint32_t{1} << 30);
  file.replace(
      file.size() - 12,
      4,
      std::string(reinterpret_cast<const char*>(&absurdSize), 4));

  auto tempFile = writeDvFile(file);
  auto dvFile = makeFooterLocatedDvFile(tempFile->getPath(), 1, file.size());

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  VELOX_ASSERT_THROW(
      reader.deletedPositions(),
      "Puffin footer payload size 1073741824 does not fit");
}

TEST_F(DeletionVectorReaderTest, rejectsCompressedPuffinFooter) {
  // Bit 0 of the flags marks a compressed footer payload. Deletion vectors are
  // always uncompressed, so reject rather than hand zstd bytes to the parser.
  const auto blob = serializeRoaringBitmapNoRun({1});
  const auto file =
      wrapInPuffinFile(blob, makeDvFooter(4, blob.size()), /*flags=*/1);

  auto tempFile = writeDvFile(file);
  auto dvFile = makeFooterLocatedDvFile(tempFile->getPath(), 1, file.size());

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  VELOX_ASSERT_THROW(
      reader.deletedPositions(),
      "Compressed Puffin footer payloads are not supported");
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
      DeletionVectorReader(badFile, 0, pool_.get(), nullptr),
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
      DeletionVectorReader(emptyDv, 0, pool_.get(), nullptr),
      "Empty deletion vector");
}

TEST_F(DeletionVectorReaderTest, noMoreDataAfterAllConsumed) {
  std::vector<int64_t> positions = {0, 1, 2};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile =
      makeDvDeleteFile(tempFile->getPath(), positions.size(), fileSize);

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
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

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

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

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

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

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

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

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

  auto bitmap = allocateBitmap(10);
  VELOX_ASSERT_THROW(
      reader.readDeletePositions(0, 10, bitmap),
      "Unknown roaring bitmap cookie");
}

TEST_F(DeletionVectorReaderTest, legacyBoundsMapLocatesBlob) {
  // Callers that predate the typed contentOffset/contentLength fields encode
  // the blob's location in the delete file's bounds maps instead. Prefix the
  // bitmap with padding so a wrong offset cannot accidentally still parse.
  const std::vector<int64_t> positions = {2, 9, 70'000};
  const std::string padding(16, '\xAB');
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(padding + bitmapData);

  auto dvFile = makeLegacyBoundsDvFile(
      tempFile->getPath(),
      positions.size(),
      padding.size() + bitmapData.size(),
      std::to_string(padding.size()),
      std::to_string(bitmapData.size()));

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  EXPECT_EQ(reader.deletedPositions(), positions);
}

TEST_F(DeletionVectorReaderTest, legacyBoundsMapRejectsNonNumericOffset) {
  auto bitmapData = serializeRoaringBitmapNoRun({1});
  auto tempFile = writeDvFile(bitmapData);

  auto dvFile = makeLegacyBoundsDvFile(
      tempFile->getPath(),
      1,
      bitmapData.size(),
      "not-a-number",
      std::to_string(bitmapData.size()));

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  VELOX_ASSERT_THROW(
      reader.deletedPositions(), "Failed to parse DV blob offset");
}

TEST_F(DeletionVectorReaderTest, legacyBoundsMapRejectsNonNumericLength) {
  auto bitmapData = serializeRoaringBitmapNoRun({1});
  auto tempFile = writeDvFile(bitmapData);

  auto dvFile = makeLegacyBoundsDvFile(
      tempFile->getPath(), 1, bitmapData.size(), "0", "not-a-number");

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  VELOX_ASSERT_THROW(
      reader.deletedPositions(), "Failed to parse DV blob length");
}

TEST_F(DeletionVectorReaderTest, emptyBitmapYieldsNoDeletes) {
  // A container-less bitmap is structurally valid but selects nothing. The
  // reader must return an empty position list rather than misparse the
  // header, and a subsequent read must leave the delete bitmap untouched.
  auto bitmapData = serializeRoaringBitmapNoRun({});
  auto tempFile = writeDvFile(bitmapData);

  // recordCount must be positive to construct a reader at all, so this also
  // covers metadata that disagrees with the blob's actual contents.
  auto dvFile = makeDvDeleteFile(
      tempFile->getPath(), 1, static_cast<uint64_t>(bitmapData.size()));

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  EXPECT_TRUE(reader.deletedPositions().empty());

  auto bitmap = allocateBitmap(64);
  reader.readDeletePositions(0, 64, bitmap);
  EXPECT_TRUE(getSetBits(bitmap, 64).empty());
}

TEST_F(DeletionVectorReaderTest, emptyGroupInRoaring64IsSkipped) {
  // A Roaring64 group carrying no containers contributes nothing, but the
  // reader must still step over its header to reach the following group.
  auto bitmapData = wrapInRoaring64(
      {{0, serializeRoaringBitmapNoRun({})},
       {1, serializeRoaringBitmapNoRun({6, 12})}});
  auto tempFile = writeDvFile(bitmapData);

  constexpr int64_t kHighGroupBase = int64_t{1} << 32;
  const std::vector<int64_t> expected = {
      kHighGroupBase + 6, kHighGroupBase + 12};

  auto dvFile = makeDvDeleteFile(
      tempFile->getPath(),
      expected.size(),
      static_cast<uint64_t>(bitmapData.size()));

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  EXPECT_EQ(reader.deletedPositions(), expected);
}

TEST_F(DeletionVectorReaderTest, skipsPositionsBeforeRequestedRange) {
  // Reading a batch that starts past earlier deletes must advance the cursor
  // over them instead of mapping them into the batch-relative bitmap. This is
  // the path a split beginning at a nonzero row offset takes.
  const std::vector<int64_t> positions = {1, 50};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);

  auto dvFile = makeDvDeleteFile(
      tempFile->getPath(),
      positions.size(),
      static_cast<uint64_t>(bitmapData.size()));

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);

  // Batch covers absolute rows [10, 60): position 1 is behind it, 50 lands at
  // batch-relative index 40.
  auto bitmap = allocateBitmap(50);
  reader.readDeletePositions(10, 50, bitmap);

  EXPECT_EQ(getSetBits(bitmap, 50), (std::vector<uint64_t>{40}));
}

// The roaring serialization format is defined as little-endian. Readers that
// take a caller-configured buffer have to assert its byte order explicitly,
// because a big-endian-configured buffer would silently decode garbage.
//
// There is no equivalent footgun here — the reader takes raw bytes and
// applies an explicit folly::Endian::little conversion to every field, so the
// interpretation is fixed regardless of caller or host. What is worth pinning
// is the observable behavior that check exists to produce: bytes serialized
// big-endian must be rejected, not silently misread.
//
// They are, though via a different guard than one might expect. The
// byte-swapped cookie (0x3A300000) matches neither serial cookie, so the blob
// is treated as the 64-bit format; the byte-swapped group count is then
// absurd and trips the Roaring64 sanity limit. The assertion below pins that
// whole chain, so a future change that loosens either the cookie check or the
// group limit cannot start silently accepting byte-swapped input.
TEST_F(DeletionVectorReaderTest, rejectsBigEndianSerializedBitmap) {
  auto appendBigEndian32 = [](std::string& out, uint32_t value) {
    const char bytes[4] = {
        static_cast<char>((value >> 24) & 0xFF),
        static_cast<char>((value >> 16) & 0xFF),
        static_cast<char>((value >> 8) & 0xFF),
        static_cast<char>(value & 0xFF)};
    out.append(bytes, sizeof(bytes));
  };

  // The same empty 32-bit bitmap serializeRoaringBitmapNoRun({}) produces,
  // but with both header words written big-endian.
  std::string bigEndianBitmap;
  appendBigEndian32(bigEndianBitmap, 12'346);
  appendBigEndian32(bigEndianBitmap, 0);

  auto tempFile = writeDvFile(bigEndianBitmap);
  auto dvFile = makeDvDeleteFile(
      tempFile->getPath(), 1, static_cast<uint64_t>(bigEndianBitmap.size()));

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  VELOX_ASSERT_THROW(
      reader.deletedPositions(),
      "Roaring64Bitmap group count exceeds sanity limit");
}

TEST_F(DeletionVectorReaderTest, enforcesRoaring64GroupKeyBound) {
  // Iceberg caps the Roaring64 group key at 2147483646 because readers load it
  // back as a signed 32-bit int. A key at or above 2^31 would shift into the
  // sign bit of `key << 32` and surface as a negative row position.
  // DeletionVectorWriter already refuses to produce such a blob; the reader
  // must reject one written by anything else rather than emit garbage.
  const auto readPositions = [&](uint32_t groupKey) {
    auto bitmapData =
        wrapInRoaring64({{groupKey, serializeRoaringBitmapNoRun({5})}});
    auto tempFile = writeDvFile(bitmapData);
    auto dvFile = makeDvDeleteFile(
        tempFile->getPath(), 1, static_cast<uint64_t>(bitmapData.size()));
    DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
    return reader.deletedPositions();
  };

  constexpr uint32_t kMaxGroupKey = 2'147'483'646;
  EXPECT_EQ(
      readPositions(kMaxGroupKey),
      (std::vector<int64_t>{(int64_t{kMaxGroupKey} << 32) + 5}));

  VELOX_ASSERT_THROW(
      readPositions(kMaxGroupKey + 1),
      "Roaring64Bitmap group key exceeds the maximum the Iceberg "
      "deletion-vector format can represent");
}

// The scenarios below mirror the shapes Iceberg's own position-index tests
// cover. The bitmaps are built here rather than taken from anywhere, so they
// exercise the same encodings without depending on external fixtures.

TEST_F(DeletionVectorReaderTest, smallAlternatingValues) {
  // Sparse odd positions in a single array container — the simplest shape a
  // real deletion vector takes.
  const std::vector<int64_t> positions = {1, 3, 5, 7, 9};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);

  auto dvFile = makeDvDeleteFile(
      tempFile->getPath(),
      positions.size(),
      static_cast<uint64_t>(bitmapData.size()));

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  EXPECT_EQ(reader.deletedPositions(), positions);
}

TEST_F(DeletionVectorReaderTest, smallAndLargeValuesPast31Bits) {
  // Two array containers far apart, the second past 2^31. The container key
  // there is 32767, so a reader that sign-extends the 16-bit key or the
  // resulting offset lands on a negative position.
  const std::vector<int64_t> positions = {
      100, 101, 2'147'483'747, 2'147'483'748};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);

  auto dvFile = makeDvDeleteFile(
      tempFile->getPath(),
      positions.size(),
      static_cast<uint64_t>(bitmapData.size()));

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  EXPECT_EQ(reader.deletedPositions(), positions);
}

TEST_F(DeletionVectorReaderTest, allContainerEncodingsInOneBitmap) {
  // Array, run, and bitset containers coexisting in a single bitmap, wrapped
  // in a second Roaring64 group. Each encoding advances the read cursor by a
  // different rule, so the reader has to switch strategies three times and
  // still land on the next group's header.
  //
  // Three containers with runs present also puts this below the threshold at
  // which the offset section is written, exercising the no-offset path.
  std::vector<uint16_t> denseValues;
  denseValues.reserve(5'000);
  for (uint16_t i = 0; i < 5'000; ++i) {
    denseValues.push_back(static_cast<uint16_t>(i * 2));
  }

  const std::vector<ContainerSpec> lowGroup = {
      {/*key=*/0, ContainerSpec::Encoding::kArray, {5, 7}},
      {/*key=*/1, ContainerSpec::Encoding::kRun, {1, 2, 3, 4}},
      {/*key=*/2, ContainerSpec::Encoding::kBitset, denseValues},
  };
  const std::vector<ContainerSpec> highGroup = {
      {/*key=*/0, ContainerSpec::Encoding::kArray, {9}},
  };

  auto bitmapData = wrapInRoaring64(
      {{0, serializeRoaringBitmapMixed(lowGroup)},
       {1, serializeRoaringBitmapMixed(highGroup)}});
  auto tempFile = writeDvFile(bitmapData);

  constexpr int64_t kHighGroupBase = int64_t{1} << 32;
  std::vector<int64_t> expected = {5, 7};
  for (int64_t i = 1; i <= 4; ++i) {
    expected.push_back(65'536 + i);
  }
  for (auto value : denseValues) {
    expected.push_back(131'072 + value);
  }
  expected.push_back(kHighGroupBase + 9);
  std::sort(expected.begin(), expected.end());

  auto dvFile = makeDvDeleteFile(
      tempFile->getPath(),
      expected.size(),
      static_cast<uint64_t>(bitmapData.size()));

  DeletionVectorReader reader(dvFile, 0, pool_.get(), nullptr);
  EXPECT_EQ(reader.deletedPositions(), expected);
}
