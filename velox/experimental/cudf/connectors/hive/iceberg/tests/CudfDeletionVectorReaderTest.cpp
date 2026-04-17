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

#include "velox/experimental/cudf/connectors/hive/iceberg/tests/CudfDeletionVectorTestUtils.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::cudf_velox::iceberg::test;
using ::facebook::velox::cudf_velox::connector::hive::iceberg::
    CudfDeletionVectorReader;
namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

namespace {

/// Serializes a roaring bitmap in the portable format with run containers
/// (cookie = 12347). All containers are run-encoded.
std::string serializeRoaringBitmapWithRuns(
    const std::vector<
        std::pair<uint16_t, std::vector<std::pair<uint16_t, uint16_t>>>>&
        containerRuns) {
  // containerRuns: vector of (highBitsKey, vector of (start, lengthMinus1)).
  const uint32_t numContainers = static_cast<uint32_t>(containerRuns.size());

  // Cookie: low 16 bits = 12347, high 16 bits = numContainers - 1.
  const uint32_t cookie =
      static_cast<uint32_t>(12347) | ((numContainers - 1) << 16);

  std::string data;
  data.append(reinterpret_cast<const char*>(&cookie), 4);

  // Run bitmap: all containers are run containers. ceil(numContainers / 8)
  // bytes.
  const uint32_t runBitmapBytes = (numContainers + 7) / 8;
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
// Cudf-specific helpers for constructing CudfDeletionVectorReader and
// verifying results.
// ---------------------------------------------------------------------------

/// Extracts which positions from [0, totalRows) were deleted (i.e. are NOT
/// present in the filtered table)
template <typename IndexType>
std::vector<IndexType> getSetBits(
    const cudf::table_view& filtered,
    std::size_t numRows) {
  VELOX_CHECK_EQ(filtered.num_columns(), 1);
  auto index_col = filtered.column(0);
  VELOX_CHECK(index_col.type().id() == cudf::type_to_id<IndexType>());

  std::vector<IndexType> survivors(index_col.size());
  CUDF_CUDA_TRY(cudaMemcpy(
      survivors.data(),
      index_col.data<IndexType>(),
      index_col.size() * sizeof(IndexType),
      cudaMemcpyDeviceToHost));

  std::set<IndexType> unsetBits(survivors.begin(), survivors.end());
  std::vector<IndexType> setBits;
  for (IndexType i = 0; i < numRows; ++i) {
    if (unsetBits.count(i) == 0) {
      setBits.push_back(i);
    }
  }
  return setBits;
}

/// Creates a single-column cudf table of INT64 values [0, numRows).
/// This serves as both the "data" and the row-index tracker: after filtering,
/// the surviving values indicate positions
template <typename IndexType>
std::unique_ptr<cudf::table> makeIndexTable(
    cudf::size_type numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto buffer = rmm::device_buffer(numRows * sizeof(IndexType), stream, mr);
  std::vector<IndexType> hostData(numRows);
  std::iota(hostData.begin(), hostData.end(), IndexType{0});
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      buffer.data(),
      hostData.data(),
      numRows * sizeof(IndexType),
      cudaMemcpyHostToDevice));

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(
      std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_to_id<IndexType>()},
          numRows,
          std::move(buffer),
          rmm::device_buffer{},
          0));

  return std::make_unique<cudf::table>(std::move(cols));
}

/// Creates a boolean mask column of true values.
std::unique_ptr<cudf::column> makeRowMaskColumn(
    cudf::size_type numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto true_scalar = cudf::numeric_scalar<bool>(true, true, stream, mr);
  return cudf::make_column_from_scalar(true_scalar, numRows, stream, mr);
}

/// Applies the deletion vector to the input table and returns the filtered
/// table.
std::unique_ptr<cudf::table> applyDeletionVector(
    CudfDeletionVectorReader& reader,
    const std::unique_ptr<cudf::table>& table,
    int64_t startRow,
    cudf::mutable_column_view const& rowMask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  reader.applyDeletes(rowMask, startRow, table->num_rows(), stream, mr);
  return cudf::apply_boolean_mask(table->view(), rowMask, stream, mr);
}

} // namespace

class CudfDeletionVectorReaderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    filesystems::registerLocalFileSystem();
  }

  rmm::cuda_stream_view stream_{cudf::get_default_stream()};
  rmm::device_async_resource_ref mr_{
      rmm::mr::get_current_device_resource_ref()};
};

TEST_F(CudfDeletionVectorReaderTest, basicArrayContainer) {
  // Positions: 0, 5, 10, 99.
  using IndexType = int64_t;
  std::vector<IndexType> positions = {0, 5, 10, 99};
  auto bitmapData = serializeRoaringBitmapNoRun<IndexType>(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile = makeDvDeleteFile(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(dvFile);

  constexpr auto numRows = 100;
  auto table = makeIndexTable<IndexType>(numRows, stream_, mr_);
  auto rowMask = makeRowMaskColumn(numRows, stream_, mr_);
  auto filtered = applyDeletionVector(
      reader, table, 0, rowMask->mutable_view(), stream_, mr_);
  stream_.synchronize();

  auto deleted = getSetBits<IndexType>(filtered->view(), numRows);
  EXPECT_EQ(deleted, (std::vector<IndexType>{0, 5, 10, 99}));
}

TEST_F(CudfDeletionVectorReaderTest, noDeletesInRange) {
  // Positions: 1000, 2000.
  using IndexType = int64_t;
  std::vector<IndexType> positions = {1000, 2000};
  auto bitmapData = serializeRoaringBitmapNoRun<IndexType>(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile = makeDvDeleteFile(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(dvFile);

  // Table has 100 rows; deleted positions 1000/2000 are out of range.
  constexpr auto numRows = 100;
  auto table = makeIndexTable<IndexType>(numRows, stream_, mr_);
  auto rowMask = makeRowMaskColumn(numRows, stream_, mr_);
  auto filtered = applyDeletionVector(
      reader, table, 0, rowMask->mutable_view(), stream_, mr_);
  stream_.synchronize();

  auto deleted = getSetBits<IndexType>(filtered->view(), numRows);
  EXPECT_TRUE(deleted.empty());
}

TEST_F(CudfDeletionVectorReaderTest, runContainers) {
  // Use run-encoded containers: positions 10-19 and 50-59.
  using IndexType = int64_t;
  std::vector<std::pair<uint16_t, std::vector<std::pair<uint16_t, uint16_t>>>>
      containerRuns = {
          {0, {{10, 9}, {50, 9}}},
      };
  auto bitmapData = serializeRoaringBitmapWithRuns(containerRuns);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile = makeDvDeleteFile(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(dvFile);

  constexpr auto numRows = 100;
  auto table = makeIndexTable<IndexType>(numRows, stream_, mr_);
  auto rowMask = makeRowMaskColumn(numRows, stream_, mr_);
  auto filtered = applyDeletionVector(
      reader, table, 0, rowMask->mutable_view(), stream_, mr_);
  stream_.synchronize();

  auto deleted = getSetBits<IndexType>(filtered->view(), numRows);
  // Expect positions 10-19 and 50-59 to be deleted.
  std::vector<IndexType> expected(20);
  std::iota(expected.begin(), expected.begin() + 10, IndexType{10});
  std::iota(expected.begin() + 10, expected.begin() + 20, IndexType{50});
  EXPECT_EQ(deleted, expected);
}

TEST_F(CudfDeletionVectorReaderTest, largePositionsMultipleContainers) {
  // Positions spanning two containers: one in container 0 (key=0), one in
  // container 1 (key=1, i.e. pos >= 65536).
  using IndexType = int64_t;
  std::vector<IndexType> positions = {5, 100, 65536, 65600};
  auto bitmapData = serializeRoaringBitmapNoRun<IndexType>(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile = makeDvDeleteFile(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(dvFile);

  constexpr auto numRows = 66000;
  auto table = makeIndexTable<IndexType>(numRows, stream_, mr_);
  auto rowMask = makeRowMaskColumn(numRows, stream_, mr_);
  auto filtered = applyDeletionVector(
      reader, table, 0, rowMask->mutable_view(), stream_, mr_);
  stream_.synchronize();

  auto deleted = getSetBits<IndexType>(filtered->view(), numRows);
  EXPECT_EQ(deleted, (std::vector<IndexType>{5, 100, 65536, 65600}));
}

TEST_F(CudfDeletionVectorReaderTest, blobOffset) {
  // Write a file with some padding before the actual bitmap data.
  using IndexType = int64_t;
  std::vector<IndexType> positions = {3, 7, 11};
  auto bitmapData = serializeRoaringBitmapNoRun<IndexType>(positions);

  // Prepend 64 bytes of padding.
  std::string padding(64, 'X');
  std::string fileContent = padding + bitmapData;

  auto tempFile = writeDvFile(fileContent);
  auto fileSize = static_cast<uint64_t>(fileContent.size());

  auto dvFile =
      makeDvDeleteFile(tempFile->getPath(), fileSize, 1, 64, bitmapData.size());
  CudfDeletionVectorReader reader(dvFile);

  constexpr auto numRows = 20;
  auto table = makeIndexTable<IndexType>(numRows, stream_, mr_);
  auto rowMask = makeRowMaskColumn(numRows, stream_, mr_);
  auto filtered = applyDeletionVector(
      reader, table, 0, rowMask->mutable_view(), stream_, mr_);
  stream_.synchronize();

  auto deleted = getSetBits<IndexType>(filtered->view(), numRows);
  EXPECT_EQ(deleted, (std::vector<IndexType>{3, 7, 11}));
}

TEST_F(CudfDeletionVectorReaderTest, singlePosition) {
  using IndexType = int64_t;
  std::vector<IndexType> positions = {42};
  auto bitmapData = serializeRoaringBitmapNoRun<IndexType>(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile = makeDvDeleteFile(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(dvFile);

  constexpr auto numRows = 100;
  auto table = makeIndexTable<IndexType>(numRows, stream_, mr_);
  auto rowMask = makeRowMaskColumn(numRows, stream_, mr_);
  auto filtered = applyDeletionVector(
      reader, table, 0, rowMask->mutable_view(), stream_, mr_);
  stream_.synchronize();

  auto deleted = getSetBits<IndexType>(filtered->view(), numRows);
  EXPECT_EQ(deleted, (std::vector<IndexType>{42}));
}

TEST_F(CudfDeletionVectorReaderTest, consecutivePositions) {
  // Positions: 0 through 99 (100 consecutive positions).
  using IndexType = int64_t;
  std::vector<IndexType> positions(100);
  std::iota(positions.begin(), positions.end(), IndexType{0});
  auto bitmapData = serializeRoaringBitmapNoRun<IndexType>(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile = makeDvDeleteFile(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(dvFile);

  // Table has 200 rows; first 100 should be deleted.
  constexpr std::size_t numRows = 200;
  auto table = makeIndexTable<IndexType>(numRows, stream_, mr_);
  auto rowMask = makeRowMaskColumn(numRows, stream_, mr_);
  auto filtered = applyDeletionVector(
      reader, table, 0, rowMask->mutable_view(), stream_, mr_);
  stream_.synchronize();

  auto deleted = getSetBits<IndexType>(filtered->view(), numRows);
  std::vector<IndexType> expected(100);
  std::iota(expected.begin(), expected.end(), IndexType{0});
  EXPECT_EQ(deleted, expected);
}

TEST_F(CudfDeletionVectorReaderTest, startRowOffset) {
  // Positions: 100, 105, 110. Apply to a chunk starting at row 100.
  using IndexType = int64_t;
  std::vector<IndexType> positions = {100, 105, 110};
  auto bitmapData = serializeRoaringBitmapNoRun<IndexType>(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile = makeDvDeleteFile(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(dvFile);

  // Create a 20-row table; startRow=100 means the reader maps row 0 of this
  // chunk to absolute position 100.
  constexpr auto numRows = 20;
  auto table = makeIndexTable<IndexType>(numRows, stream_, mr_);
  auto rowMask = makeRowMaskColumn(numRows, stream_, mr_);
  auto filtered = applyDeletionVector(
      reader, table, 100, rowMask->mutable_view(), stream_, mr_);
  stream_.synchronize();

  // Rows at absolute positions 100, 105, 110 correspond to chunk indices
  // 0, 5, 10, which should be deleted.
  auto deleted = getSetBits<IndexType>(filtered->view(), numRows);
  EXPECT_EQ(deleted, (std::vector<int64_t>{0, 5, 10}));
}

TEST_F(CudfDeletionVectorReaderTest, largeDeletionVector) {
  // Create a bitmap with positions spanning multiple 16-bit containers to
  // exercise multi-container roaring bitmap paths. Each container must have
  // <= 4096 entries (array container limit for the test serializer).
  // Use 5 containers (high keys 0-4) with ~1024 positions each.
  using IndexType = int64_t;
  std::vector<IndexType> positions;
  positions.reserve(5 * 1024);
  for (IndexType key = 0; key < 5; ++key) {
    IndexType base = key * 65536;
    for (int i = 0; i < 1024; ++i) {
      positions.push_back(base + i * 64); // every 64th value
    }
  }

  auto bitmapData = serializeRoaringBitmapNoRun<IndexType>(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile = makeDvDeleteFile(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(dvFile);

  // Apply to a 2000-row table starting at row 0. Only positions in
  // container 0 (key=0, positions 0,64,128,...,65472) overlap with [0,2000).
  constexpr std::size_t numRows = 2000;
  auto table = makeIndexTable<IndexType>(numRows, stream_, mr_);
  auto rowMask = makeRowMaskColumn(numRows, stream_, mr_);
  auto filtered = applyDeletionVector(
      reader, table, 0, rowMask->mutable_view(), stream_, mr_);
  stream_.synchronize();

  auto deleted = getSetBits<IndexType>(filtered->view(), numRows);
  // Only positions in container 0 overlap [0, 2000): 0, 64, 128, ..., 1984.
  std::vector<IndexType> expected;
  for (IndexType i = 0; i < numRows; i += IndexType{64}) {
    expected.push_back(i);
  }
  EXPECT_EQ(deleted, expected);
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
