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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BlockCodecTarget.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/OpenZLBenchTarget.h"

// The lazy arm exists to sit between "holds the column compressed" and "holds
// the column decoded", so the two properties that distinguish it from both
// neighbours are pinned here: a read touches only the blocks it overlaps, and
// what the target holds resident grows with the reads rather than with the
// column. Both are asserted on counters and on byte counts, never on elapsed
// time, so they hold on any machine.

namespace facebook::nimble::mlidc {
namespace {

Vector<int64_t> makeData(uint32_t n) {
  auto& pool = benchmarks::benchmarkPool();
  Vector<int64_t> data{pool.get()};
  data.resize(n);
  uint64_t state = 0x2545F4914F6CDD1DULL;
  for (uint32_t i = 0; i < n; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    data[i] = static_cast<int64_t>((state >> 40) ^ (i % 13));
  }
  return data;
}

constexpr uint32_t kBlockSize = 64;
constexpr uint32_t kRows = 1'000; // 16 blocks, the last holding 40.

std::unique_ptr<BlockLazyTarget<int64_t>> makeLazy(
    const Vector<int64_t>& data,
    uint32_t blockSize = kBlockSize) {
  auto inner = std::make_unique<BlockCompressedTarget<int64_t>>(
      std::make_unique<NimbleBlockCodec<int64_t>>(CompressionType::Zstd),
      blockSize,
      "test");
  inner->encode(data, Encoding::Options{});
  return std::make_unique<BlockLazyTarget<int64_t>>(
      std::move(inner), static_cast<uint32_t>(data.size()), blockSize);
}

// The claim the arm makes about work. k scattered probes decode at most k
// blocks, and never the whole column, which is what separates it from
// +materialize. Probes are spaced so each lands in its own block, so the
// expected count is exactly k until k reaches the block count.
TEST(BlockLazyTargetTest, kProbesTouchMinOfKAndBlockCount) {
  const auto data = makeData(kRows);

  for (const size_t probes : {1u, 4u, 16u}) {
    auto target = makeLazy(data);
    int64_t value{};
    for (size_t i = 0; i < probes; ++i) {
      const auto index = static_cast<uint32_t>(i * kBlockSize);
      target->materializeRange(index, 1, &value);
      ASSERT_EQ(value, data[index]) << "probe " << i;
    }
    EXPECT_EQ(target->numBlockMaterializations(), probes);
    // Only meaningful while the workload really is partial. Probing 16
    // distinct blocks of a 16-block column leaves all 16 resident, and that
    // is the arm working, not failing.
    if (probes < 16u) {
      EXPECT_EQ(target->numCachedBlocks(), probes)
          << "a partial workload must not leave every block resident";
    }
  }

  // Probing every block saturates at the block count rather than exceeding it.
  auto target = makeLazy(data);
  int64_t value{};
  for (uint32_t block = 0; block < 16; ++block) {
    target->materializeRange(block * kBlockSize, 1, &value);
  }
  EXPECT_EQ(target->numBlockMaterializations(), 16u);
}

// Repeating a probe must not decode again: the block is already held, and an
// arm that redecoded would be the bare block arm wearing a different name.
TEST(BlockLazyTargetTest, repeatedProbesInOneBlockDecodeItOnce) {
  const auto data = makeData(kRows);
  auto target = makeLazy(data);

  int64_t value{};
  for (size_t i = 0; i < 64; ++i) {
    target->materializeRange(7, 1, &value);
  }
  EXPECT_EQ(target->numBlockMaterializations(), 1u);
  EXPECT_EQ(value, data[7]);

  // A second block, then back to the first: still one decode each, because the
  // cache is indexed by block number rather than holding only the last one.
  target->materializeRange(500, 1, &value);
  target->materializeRange(7, 1, &value);
  EXPECT_EQ(target->numBlockMaterializations(), 2u);
}

// Resident bytes is the axis this whole change adds, so its shape is pinned:
// it starts at the compressed footprint, rises as reads reach new blocks, and
// never falls while nothing is discarded.
TEST(BlockLazyTargetTest, residentBytesRisesWithTheReadsAndNeverFalls) {
  const auto data = makeData(kRows);
  auto target = makeLazy(data);

  const size_t initial = target->residentBytes();
  EXPECT_GT(initial, 0u);

  size_t previous = initial;
  int64_t value{};
  for (uint32_t block = 0; block < 8; ++block) {
    target->materializeRange(block * kBlockSize, 1, &value);
    const size_t now = target->residentBytes();
    EXPECT_GT(now, previous) << "block " << block << " left nothing resident";
    previous = now;
  }

  // Re-reading what is already held adds nothing.
  target->materializeRange(0, 1, &value);
  EXPECT_EQ(target->residentBytes(), previous);

  // Eight of sixteen blocks decoded, so the cache holds about half a column.
  const size_t cached = previous - initial;
  EXPECT_GE(cached, 8u * kBlockSize * sizeof(int64_t));
  EXPECT_LT(cached, static_cast<size_t>(kRows) * sizeof(int64_t));

  // Discarding returns it to where it started.
  target->discardAccessStructure();
  EXPECT_EQ(target->residentBytes(), initial);
  EXPECT_EQ(target->numCachedBlocks(), 0u);
}

// A bulk read must not become full materialisation by a side door. If
// materializeAll populated the cache, one scan would leave the whole column
// resident and this arm would silently be the +materialize arm.
TEST(BlockLazyTargetTest, materializeAllBypassesTheCache) {
  const auto data = makeData(kRows);
  auto target = makeLazy(data);

  const size_t initial = target->residentBytes();
  std::vector<int64_t> out(kRows);
  target->materializeAll(out.data(), kRows);

  EXPECT_THAT(out, testing::ElementsAreArray(data.data(), kRows));
  EXPECT_EQ(target->numCachedBlocks(), 0u);
  EXPECT_EQ(target->numBlockMaterializations(), 0u);
  EXPECT_EQ(target->residentBytes(), initial);
}

// The lazy arm holds strictly less than the materialised one after a partial
// workload, and that gap is the entire reason the arm exists. Asserted against
// MaterializingTarget over the same inner codec so the comparison is like for
// like.
TEST(BlockLazyTargetTest, holdsFarLessThanFullMaterialisationAfterAFewProbes) {
  const auto data = makeData(kRows);
  auto lazy = makeLazy(data);

  auto innerForFull = std::make_unique<BlockCompressedTarget<int64_t>>(
      std::make_unique<NimbleBlockCodec<int64_t>>(CompressionType::Zstd),
      kBlockSize,
      "test");
  innerForFull->encode(data, Encoding::Options{});
  MaterializingTarget<int64_t> full{std::move(innerForFull), kRows};

  int64_t value{};
  for (uint32_t i = 0; i < 4; ++i) {
    lazy->materializeRange(i * kBlockSize, 1, &value);
    full.materializeRange(i * kBlockSize, 1, &value);
  }

  EXPECT_LT(lazy->residentBytes(), full.residentBytes());

  // The materialised arm holds a whole decoded column on top of its payload;
  // the lazy one holds four blocks of it. The gap is therefore the rows the
  // lazy arm has not decoded, less the per-block index it keeps in order to
  // stay addressable -- one empty vector per block, which is real resident
  // memory and is deliberately counted rather than excused.
  const size_t undecodedBytes = (kRows - 4u * kBlockSize) * sizeof(int64_t);
  const size_t blockIndexBytes =
      (kRows / kBlockSize + 1u) * sizeof(std::vector<int64_t>);
  const size_t gap = full.residentBytes() - lazy->residentBytes();
  EXPECT_GE(gap, undecodedBytes - blockIndexBytes);
  EXPECT_LE(gap, undecodedBytes);
}

// Round trips through every read shape, including ranges that span blocks and
// the short final block, so the cache cannot be returning the right byte count
// with the wrong rows.
TEST(BlockLazyTargetTest, roundTripsAcrossBlockBoundaries) {
  const auto data = makeData(kRows);
  auto target = makeLazy(data);

  const std::vector<std::pair<uint32_t, uint32_t>> ranges{
      {0, 1}, {63, 2}, {64, 64}, {1, 128}, {960, 40}, {999, 1}};
  for (const auto& [begin, count] : ranges) {
    std::vector<int64_t> out(count);
    target->materializeRange(begin, count, out.data());
    for (uint32_t i = 0; i < count; ++i) {
      ASSERT_EQ(out[i], data[begin + i])
          << "range [" << begin << ", " << count << ")";
    }
  }

  // A gather, served from whatever those reads left cached.
  const std::vector<nimble::RowRange> gather{
      {5, 8}, {70, 72}, {500, 504}, {997, 1000}};
  uint32_t total = 0;
  for (const auto& range : gather) {
    total += range.numRows();
  }
  std::vector<int64_t> out(total);
  target->skipThenMaterialize(gather, out.data());
  uint32_t cursor = 0;
  for (const auto& range : gather) {
    const uint32_t begin = range.startRow;
    const uint32_t count = range.numRows();
    for (uint32_t i = 0; i < count; ++i) {
      ASSERT_EQ(out[cursor + i], data[begin + i]);
    }
    cursor += count;
  }
}

// The arm reports what the drivers key their caps and their curves off. It is
// block-addressable, and it has no one-time build: its cost is spread over
// whichever reads happen to miss, which is why the story is told by
// resident_bytes along the ops axis rather than by build_ns.
TEST(BlockLazyTargetTest, reportsBlockPathAndNoBuildPhase) {
  const auto data = makeData(kRows);
  auto target = makeLazy(data);

  EXPECT_EQ(target->readPath(), ReadPath::kBlock);
  EXPECT_FALSE(target->buildsAccessStructure());

  // A column that fits in one block is a whole-payload codec whatever the
  // configured block size said, exactly as the eager block arm reports.
  auto single = makeLazy(data, kRows * 2);
  EXPECT_EQ(single->readPath(), ReadPath::kWholePayload);
}

// The arms the sweep registers, and the property that makes them comparable
// with their bare twins: the same encoded bytes, differing only in what a read
// leaves behind.
TEST(BlockLazyTargetTest, encoderEntries) {
  const auto data = makeData(3'000);

  for (auto& entry : buildZstdBlockEncoders<int64_t>()) {
    if (entry.variant != "block-" + std::to_string(kLazyBlockElementCount)) {
      continue;
    }
    const auto bareName = entry.name;
    auto bareTarget = entry.factory(data, Encoding::Options{});
    auto lazy =
        withBlockLazyMaterialization<int64_t>(entry, kLazyBlockElementCount);
    EXPECT_EQ(lazy.name, bareName + "+lazy");

    auto lazyTarget = lazy.factory(data, Encoding::Options{});
    EXPECT_EQ(lazyTarget->payloadSize(), bareTarget->payloadSize());

    std::vector<int64_t> out(data.size());
    lazyTarget->materializeAll(out.data(), data.size());
    EXPECT_THAT(out, testing::ElementsAreArray(data.data(), data.size()));
  }
}

} // namespace
} // namespace facebook::nimble::mlidc

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  facebook::velox::memory::MemoryManager::initialize({});
  return RUN_ALL_TESTS();
}

#else
int main() {
  return 0;
}
#endif
