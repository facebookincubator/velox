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
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <gtest/gtest.h>

#include <numeric>

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
// Cudf-specific helpers for constructing a deletion mask column and extracting
// set bits.
// ---------------------------------------------------------------------------

/// Creates a boolean cudf column initialized to all-false, representing an
/// empty deletion mask (no rows deleted yet).
std::unique_ptr<cudf::column> makeDeletionColumn(
    cudf::size_type numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto falseScalar = cudf::numeric_scalar<bool>(false, true, stream, mr);
  return cudf::make_column_from_scalar(falseScalar, numRows, stream, mr);
}

/// Returns the indices (in [0, numRows)) where the deletion mask is true.
/// These indices correspond to rows that are marked as deleted.
template <typename IndexType>
std::vector<IndexType> getSetBits(
    const cudf::column_view& deleteMask,
    std::size_t numRows,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_EQ(deleteMask.size(), static_cast<cudf::size_type>(numRows));
  VELOX_CHECK(deleteMask.type().id() == cudf::type_id::BOOL8);

  std::vector<uint8_t> host(numRows);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      host.data(),
      deleteMask.data<bool>(),
      numRows * sizeof(bool),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();

  std::vector<IndexType> setBits;
  setBits.reserve(numRows);
  for (IndexType i = 0; i < numRows; ++i) {
    if (host[i] != 0) {
      setBits.emplace_back(i);
    }
  }
  return setBits;
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
  EXPECT_FALSE(reader.noMoreData());

  constexpr auto numRows = 100;
  auto deleteMask = makeDeletionColumn(numRows, stream_, mr_);
  reader.applyDeletes(deleteMask->mutable_view(), 0, numRows, stream_, mr_);

  auto setBits = getSetBits<IndexType>(deleteMask->view(), numRows, stream_);
  EXPECT_EQ(setBits, positions);
  EXPECT_TRUE(reader.noMoreData());
}

TEST_F(CudfDeletionVectorReaderTest, batchRangeFiltering) {
  // Positions: 10, 20, 30, 40, 50.
  using IndexType = int64_t;
  std::vector<IndexType> positions = {10, 20, 30, 40, 50};
  auto bitmapData = serializeRoaringBitmapNoRun<IndexType>(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile = makeDvDeleteFile(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(dvFile);

  // First batch: rows 0-24 (should contain positions 10, 20).
  auto mask1 = makeDeletionColumn(25, stream_, mr_);
  reader.applyDeletes(mask1->mutable_view(), 0, 25, stream_, mr_);
  auto bits1 = getSetBits<IndexType>(mask1->view(), 25, stream_);
  EXPECT_EQ(bits1, (std::vector<IndexType>{10, 20}));

  // Second batch: rows 25-49 (should contain positions 30, 40).
  // Chunk-local indices for 30 and 40 are 5 and 15.
  auto mask2 = makeDeletionColumn(25, stream_, mr_);
  reader.applyDeletes(mask2->mutable_view(), 25, 25, stream_, mr_);
  auto bits2 = getSetBits<IndexType>(mask2->view(), 25, stream_);
  EXPECT_EQ(bits2, (std::vector<IndexType>{5, 15}));

  // Third batch: rows 50-74 (should contain position 50).
  // Chunk-local index for 50 is 0.
  auto mask3 = makeDeletionColumn(25, stream_, mr_);
  reader.applyDeletes(mask3->mutable_view(), 50, 25, stream_, mr_);
  auto bits3 = getSetBits<IndexType>(mask3->view(), 25, stream_);
  EXPECT_EQ(bits3, (std::vector<IndexType>{0}));
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
  auto deleteMask = makeDeletionColumn(numRows, stream_, mr_);
  reader.applyDeletes(deleteMask->mutable_view(), 0, numRows, stream_, mr_);

  auto setBits = getSetBits<IndexType>(deleteMask->view(), numRows, stream_);
  EXPECT_TRUE(setBits.empty());
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
  auto deleteMask = makeDeletionColumn(numRows, stream_, mr_);
  reader.applyDeletes(deleteMask->mutable_view(), 0, numRows, stream_, mr_);

  auto setBits = getSetBits<IndexType>(deleteMask->view(), numRows, stream_);
  // Expect positions 10-19 and 50-59 to be deleted
  std::vector<IndexType> expected(20);
  std::iota(expected.begin(), expected.begin() + 10, IndexType{10});
  std::iota(expected.begin() + 10, expected.begin() + 20, IndexType{50});
  EXPECT_EQ(setBits, expected);
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
  auto deleteMask = makeDeletionColumn(numRows, stream_, mr_);
  reader.applyDeletes(deleteMask->mutable_view(), 0, numRows, stream_, mr_);

  auto setBits = getSetBits<IndexType>(deleteMask->view(), numRows, stream_);
  EXPECT_EQ(setBits, (std::vector<IndexType>{5, 100, 65536, 65600}));
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
  auto deleteMask = makeDeletionColumn(numRows, stream_, mr_);
  reader.applyDeletes(deleteMask->mutable_view(), 0, numRows, stream_, mr_);

  auto setBits = getSetBits<IndexType>(deleteMask->view(), numRows, stream_);
  EXPECT_EQ(setBits, (std::vector<IndexType>{3, 7, 11}));
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
  auto deleteMask = makeDeletionColumn(numRows, stream_, mr_);
  reader.applyDeletes(deleteMask->mutable_view(), 0, numRows, stream_, mr_);

  auto setBits = getSetBits<IndexType>(deleteMask->view(), numRows, stream_);
  EXPECT_EQ(setBits, (std::vector<IndexType>{42}));
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

  // Chunk covers rows 0-99; all 100 should be deleted.
  constexpr auto numRows = 100;
  auto deleteMask = makeDeletionColumn(numRows, stream_, mr_);
  reader.applyDeletes(deleteMask->mutable_view(), 0, numRows, stream_, mr_);

  auto setBits = getSetBits<IndexType>(deleteMask->view(), numRows, stream_);
  std::vector<IndexType> expected(100);
  std::iota(expected.begin(), expected.end(), IndexType{0});
  EXPECT_EQ(setBits, expected);
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

  // Create a 20-row chunk; startRow=100 means the reader maps row 0 of this
  // chunk to absolute position 100.
  constexpr uint64_t numRows = 20;
  auto deleteMask = makeDeletionColumn(numRows, stream_, mr_);
  reader.applyDeletes(deleteMask->mutable_view(), 100, numRows, stream_, mr_);

  // Absolute positions 100, 105, 110 correspond to chunk indices 0, 5, 10.
  auto setBits = getSetBits<IndexType>(deleteMask->view(), numRows, stream_);
  EXPECT_EQ(setBits, (std::vector<IndexType>{0, 5, 10}));
}

TEST_F(CudfDeletionVectorReaderTest, noMoreDataAfterAllConsumed) {
  // GPU reader loads the bitmap lazily on the first `applyDeletes` call and
  // sets noMoreData() to true thereafter. Subsequent calls re-use the same
  // bitmap but do not reload it.
  using IndexType = int64_t;
  std::vector<IndexType> positions = {0, 1, 2};
  auto bitmapData = serializeRoaringBitmapNoRun<IndexType>(positions);
  auto tempFile = writeDvFile(bitmapData);
  auto fileSize = static_cast<uint64_t>(bitmapData.size());

  auto dvFile = makeDvDeleteFile(tempFile->getPath(), fileSize);
  CudfDeletionVectorReader reader(dvFile);
  EXPECT_FALSE(reader.noMoreData());

  constexpr uint64_t numRows = 10;
  auto mask1 = makeDeletionColumn(numRows, stream_, mr_);
  reader.applyDeletes(mask1->mutable_view(), 0, numRows, stream_, mr_);
  EXPECT_TRUE(reader.noMoreData());

  auto bits1 = getSetBits<IndexType>(mask1->view(), numRows, stream_);
  EXPECT_EQ(bits1, (std::vector<IndexType>{0, 1, 2}));

  // A second call on a chunk past the deleted positions should leave the
  // mask all-false.
  auto mask2 = makeDeletionColumn(numRows, stream_, mr_);
  reader.applyDeletes(mask2->mutable_view(), 10, numRows, stream_, mr_);
  auto bits2 = getSetBits<IndexType>(mask2->view(), numRows, stream_);
  EXPECT_TRUE(bits2.empty());
  EXPECT_TRUE(reader.noMoreData());
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

  // Apply to a 2000-row chunk starting at row 0. Only positions in
  // container 0 (key=0, positions 0,64,128,...,65472) overlap with [0,2000).
  constexpr auto numRows = 2000;
  auto deleteMask = makeDeletionColumn(numRows, stream_, mr_);
  reader.applyDeletes(deleteMask->mutable_view(), 0, numRows, stream_, mr_);

  auto setBits = getSetBits<IndexType>(deleteMask->view(), numRows, stream_);
  // Only positions in container 0 overlap [0, 2000): 0, 64, 128, ..., 1984.
  std::vector<IndexType> expected;
  for (IndexType i = 0; i < numRows; i += IndexType{64}) {
    expected.push_back(i);
  }
  EXPECT_EQ(setBits, expected);
}

// ---------------------------------------------------------------------------
// Upstream tests NOT ported (with rationale):
//
// - splitOffset, splitOffsetWithBaseReadOffset: Upstream's DeletionVectorReader
//   accepts a splitOffset constructor argument that is subtracted from bitmap
//   positions when writing into the output bitmap. The GPU reader stores
//   splitOffset but does not currently use it in applyDeletes(); callers are
//   expected to pass the absolute file row index via the startRow parameter.
//   startRowOffset above covers the equivalent behavior.
//
// - constructorRejectsWrongContentType, constructorRejectsEmptyDv: The GPU
//   CudfDeletionVectorReader applies the same content-type and record-count
//   validations in its constructor (see VELOX_CHECK calls). Dedicated tests
//   for these paths can be added once the death-test infrastructure is wired
//   up here.
//
// - invalidBitmapTooSmall, invalidBitmapBadCookie: Bitmap parsing is handled
//   by cuco on the GPU; error handling differs from the CPU implementation.
// ---------------------------------------------------------------------------
