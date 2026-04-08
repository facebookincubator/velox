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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfDeletionVectorReader.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempFilePath.h"

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <gtest/gtest.h>

#include <fstream>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox::connector::hive::iceberg;
using namespace facebook::velox::common::testutil;

namespace {

// ---------------------------------------------------------------------------
// Bitmap serializers — identical to upstream DeletionVectorReaderTest.cpp.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// File writer — identical to upstream DeletionVectorReaderTest.cpp.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Cudf-specific helpers for constructing CudfDeletionVectorReader and
// verifying results.
// ---------------------------------------------------------------------------

/// Creates the constructor arguments for CudfDeletionVectorReader.
std::tuple<
    std::string,
    uint64_t,
    std::unordered_map<int32_t, std::string>,
    std::unordered_map<int32_t, std::string>>
makeDvReaderArgs(
    const std::string& filePath,
    uint64_t fileSize,
    uint64_t blobOffset = 0,
    std::optional<uint64_t> blobLength = std::nullopt) {
  std::unordered_map<int32_t, std::string> lowerBounds;
  std::unordered_map<int32_t, std::string> upperBounds;
  lowerBounds[CudfDeletionVectorReader::kDvOffsetFieldId] =
      std::to_string(blobOffset);
  upperBounds[CudfDeletionVectorReader::kDvLengthFieldId] =
      std::to_string(blobLength.value_or(fileSize));
  return {filePath, fileSize, lowerBounds, upperBounds};
}

/// Extracts the remaining row indices from a filtered cudf table.
/// The table must have a single INT64 column containing the original row
/// indices (used to verify which rows survived deletion).
std::vector<int64_t> getSurvivorIndices(const cudf::table& table) {
  VELOX_CHECK_EQ(table.num_columns(), 1);
  auto col = table.view().column(0);
  VELOX_CHECK(col.type().id() == cudf::type_id::INT64);
  std::vector<int64_t> result(col.size());
  cudaMemcpy(
      result.data(),
      col.data<int64_t>(),
      col.size() * sizeof(int64_t),
      cudaMemcpyDeviceToHost);
  return result;
}

/// Creates a single-column cudf table of INT64 values [0, numRows).
/// This serves as both the "data" and the row-index tracker: after filtering,
/// the surviving values tell us exactly which row indices were kept.
std::unique_ptr<cudf::table> makeIndexTable(
    int64_t numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto buf = rmm::device_buffer(numRows * sizeof(int64_t), stream, mr);
  std::vector<int64_t> hostData(numRows);
  std::iota(hostData.begin(), hostData.end(), int64_t{0});
  cudaMemcpy(
      buf.data(),
      hostData.data(),
      numRows * sizeof(int64_t),
      cudaMemcpyHostToDevice);
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT64},
      numRows,
      std::move(buf),
      rmm::device_buffer{},
      0);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

} // namespace

class CudfDeletionVectorReaderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!memory::MemoryManager::testInstance()) {
      memory::MemoryManager::initialize(memory::MemoryManager::Options{});
    }
  }

  void SetUp() override {
    filesystems::registerLocalFileSystem();
    stream_ = cudf::get_default_stream();
    mr_ = rmm::mr::get_current_device_resource_ref();
  }

  rmm::cuda_stream_view stream_;
  rmm::device_async_resource_ref mr_{
      rmm::mr::get_current_device_resource_ref()};
};

TEST_F(CudfDeletionVectorReaderTest, basicArrayContainer) {
  // Positions: 0, 5, 10, 99.
  std::vector<int64_t> positions = {0, 5, 10, 99};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto [path, fsize, lower, upper] =
      makeDvReaderArgs(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(
      std::move(path), fsize, std::move(lower), std::move(upper));
  reader.loadAndInitialize(stream_);

  auto table = makeIndexTable(100, stream_, mr_);
  auto rowMask =
      std::make_shared<rmm::device_buffer>(100 * sizeof(bool), stream_, mr_);
  auto filtered =
      reader.applyDeletionVector(table->view(), 0, rowMask, stream_, mr_);
  stream_.synchronize();

  auto survivors = getSurvivorIndices(*filtered);
  EXPECT_EQ(filtered->num_rows(), 96);
  std::set<int64_t> deleted(positions.begin(), positions.end());
  for (auto v : survivors) {
    EXPECT_EQ(deleted.count(v), 0) << "Position " << v << " should be deleted";
  }
}

TEST_F(CudfDeletionVectorReaderTest, noDeletesInRange) {
  // Positions: 1000, 2000.
  std::vector<int64_t> positions = {1000, 2000};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto [path, fsize, lower, upper] =
      makeDvReaderArgs(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(
      std::move(path), fsize, std::move(lower), std::move(upper));
  reader.loadAndInitialize(stream_);

  // Table has 100 rows; deleted positions 1000/2000 are out of range.
  auto table = makeIndexTable(100, stream_, mr_);
  auto rowMask =
      std::make_shared<rmm::device_buffer>(100 * sizeof(bool), stream_, mr_);
  auto filtered =
      reader.applyDeletionVector(table->view(), 0, rowMask, stream_, mr_);
  stream_.synchronize();

  EXPECT_EQ(filtered->num_rows(), 100);
}

TEST_F(CudfDeletionVectorReaderTest, runContainers) {
  // Use run-encoded containers: positions 10-19 and 50-59.
  std::vector<std::pair<uint16_t, std::vector<std::pair<uint16_t, uint16_t>>>>
      containerRuns = {
          {0, {{10, 9}, {50, 9}}},
      };
  auto bitmapData = serializeRoaringBitmapWithRuns(containerRuns);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto [path, fsize, lower, upper] =
      makeDvReaderArgs(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(
      std::move(path), fsize, std::move(lower), std::move(upper));
  reader.loadAndInitialize(stream_);

  auto table = makeIndexTable(100, stream_, mr_);
  auto rowMask =
      std::make_shared<rmm::device_buffer>(100 * sizeof(bool), stream_, mr_);
  auto filtered =
      reader.applyDeletionVector(table->view(), 0, rowMask, stream_, mr_);
  stream_.synchronize();

  auto survivors = getSurvivorIndices(*filtered);
  // Expect positions 10-19 and 50-59 to be deleted (20 positions).
  EXPECT_EQ(filtered->num_rows(), 80);
  std::set<int64_t> deleted;
  for (int64_t i = 10; i <= 19; ++i) {
    deleted.insert(i);
  }
  for (int64_t i = 50; i <= 59; ++i) {
    deleted.insert(i);
  }
  for (auto v : survivors) {
    EXPECT_EQ(deleted.count(v), 0) << "Position " << v << " should be deleted";
  }
}

TEST_F(CudfDeletionVectorReaderTest, largePositionsMultipleContainers) {
  // Positions spanning two containers: one in container 0 (key=0), one in
  // container 1 (key=1, i.e. pos >= 65536).
  std::vector<int64_t> positions = {5, 100, 65536, 65600};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto [path, fsize, lower, upper] =
      makeDvReaderArgs(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(
      std::move(path), fsize, std::move(lower), std::move(upper));
  reader.loadAndInitialize(stream_);

  auto table = makeIndexTable(66000, stream_, mr_);
  auto rowMask =
      std::make_shared<rmm::device_buffer>(66000 * sizeof(bool), stream_, mr_);
  auto filtered =
      reader.applyDeletionVector(table->view(), 0, rowMask, stream_, mr_);
  stream_.synchronize();

  auto survivors = getSurvivorIndices(*filtered);
  EXPECT_EQ(filtered->num_rows(), 65996);
  std::set<int64_t> deleted(positions.begin(), positions.end());
  for (auto v : survivors) {
    EXPECT_EQ(deleted.count(v), 0) << "Position " << v << " should be deleted";
  }
}

TEST_F(CudfDeletionVectorReaderTest, blobOffset) {
  // Write a file with some padding before the actual bitmap data.
  std::vector<int64_t> positions = {3, 7, 11};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);

  // Prepend 64 bytes of padding.
  std::string padding(64, 'X');
  std::string fileContent = padding + bitmapData;

  auto tempFile = writeDvFile(fileContent);
  auto fileSize = static_cast<uint64_t>(fileContent.size());

  auto [path, fsize, lower, upper] =
      makeDvReaderArgs(tempFile->getPath(), fileSize, 64, bitmapData.size());
  CudfDeletionVectorReader reader(
      std::move(path), fsize, std::move(lower), std::move(upper));
  reader.loadAndInitialize(stream_);

  auto table = makeIndexTable(20, stream_, mr_);
  auto rowMask =
      std::make_shared<rmm::device_buffer>(20 * sizeof(bool), stream_, mr_);
  auto filtered =
      reader.applyDeletionVector(table->view(), 0, rowMask, stream_, mr_);
  stream_.synchronize();

  auto survivors = getSurvivorIndices(*filtered);
  EXPECT_EQ(filtered->num_rows(), 17);
  std::set<int64_t> deleted(positions.begin(), positions.end());
  for (auto v : survivors) {
    EXPECT_EQ(deleted.count(v), 0) << "Position " << v << " should be deleted";
  }
}

TEST_F(CudfDeletionVectorReaderTest, singlePosition) {
  std::vector<int64_t> positions = {42};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto [path, fsize, lower, upper] =
      makeDvReaderArgs(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(
      std::move(path), fsize, std::move(lower), std::move(upper));
  reader.loadAndInitialize(stream_);

  auto table = makeIndexTable(100, stream_, mr_);
  auto rowMask =
      std::make_shared<rmm::device_buffer>(100 * sizeof(bool), stream_, mr_);
  auto filtered =
      reader.applyDeletionVector(table->view(), 0, rowMask, stream_, mr_);
  stream_.synchronize();

  auto survivors = getSurvivorIndices(*filtered);
  EXPECT_EQ(filtered->num_rows(), 99);
  for (auto v : survivors) {
    EXPECT_NE(v, 42) << "Position 42 should be deleted";
  }
}

TEST_F(CudfDeletionVectorReaderTest, consecutivePositions) {
  // Positions: 0 through 99 (100 consecutive positions).
  std::vector<int64_t> positions;
  positions.reserve(100);
  for (int64_t i = 0; i < 100; ++i) {
    positions.push_back(i);
  }
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto [path, fsize, lower, upper] =
      makeDvReaderArgs(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(
      std::move(path), fsize, std::move(lower), std::move(upper));
  reader.loadAndInitialize(stream_);

  // Table has 200 rows; first 100 should be deleted.
  auto table = makeIndexTable(200, stream_, mr_);
  auto rowMask =
      std::make_shared<rmm::device_buffer>(200 * sizeof(bool), stream_, mr_);
  auto filtered =
      reader.applyDeletionVector(table->view(), 0, rowMask, stream_, mr_);
  stream_.synchronize();

  auto survivors = getSurvivorIndices(*filtered);
  EXPECT_EQ(filtered->num_rows(), 100);
  for (auto v : survivors) {
    EXPECT_GE(v, 100) << "Position " << v << " should be deleted";
  }
}

TEST_F(CudfDeletionVectorReaderTest, startRowOffset) {
  // Positions: 100, 105, 110. Apply to a chunk starting at row 100.
  std::vector<int64_t> positions = {100, 105, 110};
  auto bitmapData = serializeRoaringBitmapNoRun(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto [path, fsize, lower, upper] =
      makeDvReaderArgs(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(
      std::move(path), fsize, std::move(lower), std::move(upper));
  reader.loadAndInitialize(stream_);

  // Create a 20-row table; startRow=100 means the reader maps row 0 of this
  // chunk to absolute position 100.
  auto table = makeIndexTable(20, stream_, mr_);
  auto rowMask =
      std::make_shared<rmm::device_buffer>(20 * sizeof(bool), stream_, mr_);
  auto filtered =
      reader.applyDeletionVector(table->view(), 100, rowMask, stream_, mr_);
  stream_.synchronize();

  // Rows at absolute positions 100, 105, 110 correspond to chunk indices
  // 0, 5, 10, which should be deleted.
  EXPECT_EQ(filtered->num_rows(), 17);
  auto survivors = getSurvivorIndices(*filtered);
  std::set<int64_t> deletedChunkIndices = {0, 5, 10};
  for (auto v : survivors) {
    EXPECT_EQ(deletedChunkIndices.count(v), 0)
        << "Chunk index " << v << " should be deleted";
  }
}

// ---------------------------------------------------------------------------
// Upstream tests NOT ported (with rationale):
//
// - batchRangeFiltering: The upstream CPU reader supports incremental
//   readDeletePositions() calls with advancing baseReadOffset. The GPU reader
//   applies the DV to entire chunks via applyDeletionVector(startRow). The
//   startRowOffset test above covers the equivalent behavior.
//
// - splitOffset, splitOffsetWithBaseReadOffset: The upstream CPU reader has a
//   separate splitOffset concept. The GPU reader uses startRow in
//   applyDeletionVector() to achieve the same effect; covered by
//   startRowOffset above.
//
// - constructorRejectsWrongContentType, constructorRejectsEmptyDv: The GPU
//   CudfDeletionVectorReader takes primitive arguments (path, size, bounds
//   maps) rather than an IcebergDeleteFile, so these validations happen at
//   the CudfIcebergSplitReader level, not in the reader constructor.
//
// - noMoreDataAfterAllConsumed: The GPU reader doesn't track a noMoreData()
//   state; it filters each chunk independently via applyDeletionVector().
//
// - invalidBitmapTooSmall, invalidBitmapBadCookie: Bitmap parsing is handled
//   by cuco on the GPU; error handling differs from the CPU implementation.
// ---------------------------------------------------------------------------
