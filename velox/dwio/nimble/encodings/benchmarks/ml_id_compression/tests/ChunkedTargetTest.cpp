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
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/OpenZLBenchTarget.h"

// The chunked setting reports every axis over the column's own row numbers, so
// a read that lands on the wrong chunk, or a boundary that drops or repeats a
// row, would shift every number without failing anything. These pin the row
// arithmetic against a column held in the clear.

namespace facebook::nimble::mlidc {
namespace {

Vector<int64_t> makeData(uint32_t n) {
  Vector<int64_t> data{benchmarks::benchmarkPool().get()};
  data.resize(n);
  uint64_t state = 0x2545F4914F6CDD1DULL;
  for (uint32_t i = 0; i < n; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    data[i] = static_cast<int64_t>((state >> 40) ^ (i % 13));
  }
  return data;
}

// Holds its chunk in the clear and counts the calls it serves, so a test can
// assert both the values and how a request was split.
class CopyTarget : public NimbleBenchTargetBase<int64_t> {
 public:
  void encode(const Vector<int64_t>& data, const Encoding::Options&) override {
    values_.assign(data.data(), data.data() + data.size());
  }

  void materializeAll(int64_t* dst, uint32_t n) override {
    std::copy_n(values_.data(), n, dst);
  }

  void materializeRange(uint32_t begin, uint32_t count, int64_t* dst) override {
    ASSERT_LE(begin + count, values_.size());
    std::copy_n(values_.data() + begin, count, dst);
  }

  void skipThenMaterialize(
      std::span<const nimble::RowRange> ranges,
      int64_t* dst) override {
    ++numRangeCalls;
    for (const auto& range : ranges) {
      ASSERT_LE(range.endRow, values_.size());
      std::copy_n(values_.data() + range.startRow, range.numRows(), dst);
      dst += range.numRows();
    }
  }

  size_t payloadSize() const override {
    return values_.size() * sizeof(int64_t);
  }

  size_t residentBytes() const override {
    return payloadSize();
  }

  std::vector<std::span<const std::byte>> internalBuffers() const override {
    return {};
  }

  ReadPath readPath() const override {
    return ReadPath::kIndexed;
  }

  size_t numRangeCalls{0};

 private:
  std::vector<int64_t> values_;
};

constexpr uint32_t kRows = 1'000;
constexpr uint32_t kChunkRows = 128;

std::unique_ptr<ChunkedTarget<int64_t>> makeChunked(
    const Vector<int64_t>& data,
    std::vector<CopyTarget*>* chunks = nullptr) {
  auto target = std::make_unique<ChunkedTarget<int64_t>>(
      [chunks](const Vector<int64_t>& chunk, const Encoding::Options& opts) {
        auto inner = std::make_unique<CopyTarget>();
        inner->encode(chunk, opts);
        if (chunks != nullptr) {
          chunks->push_back(inner.get());
        }
        return std::unique_ptr<NimbleBenchTargetBase<int64_t>>(
            std::move(inner));
      },
      kChunkRows);
  target->encode(data, Encoding::Options{});
  return target;
}

TEST(ChunkedTargetTest, coversTheColumnWithAShortLastChunk) {
  const auto data = makeData(kRows);
  auto target = makeChunked(data);
  EXPECT_EQ(target->numChunks(), 8u);
  EXPECT_EQ(target->payloadSize(), kRows * sizeof(int64_t));

  std::vector<int64_t> all(kRows);
  target->materializeAll(all.data(), kRows);
  EXPECT_THAT(all, testing::ElementsAreArray(data.data(), kRows));
}

TEST(ChunkedTargetTest, rangeAcrossSeveralChunks) {
  const auto data = makeData(kRows);
  auto target = makeChunked(data);

  // Starts mid-chunk, spans two whole chunks, ends mid-chunk.
  constexpr uint32_t kBegin = 100;
  constexpr uint32_t kCount = 400;
  std::vector<int64_t> out(kCount);
  target->materializeRange(kBegin, kCount, out.data());
  EXPECT_THAT(out, testing::ElementsAreArray(data.data() + kBegin, kCount));

  // The last row, alone.
  int64_t last{};
  target->materializeRange(kRows - 1, 1, &last);
  EXPECT_EQ(last, data[kRows - 1]);
}

// A range list is partitioned per chunk: one call per chunk touched, none for
// a chunk the list skips, and a range crossing a boundary is split there.
TEST(ChunkedTargetTest, rangeListIsPartitionedPerChunk) {
  const auto data = makeData(kRows);
  std::vector<CopyTarget*> chunks;
  auto target = makeChunked(data, &chunks);

  const std::vector<nimble::RowRange> ranges{
      {0, 5},
      {120, 140}, // Crosses from chunk 0 into chunk 1.
      {200, 201},
      {600, 900}, // Chunks 4 to 7; chunks 2 and 3 are skipped.
      {990, 1'000},
  };
  std::vector<int64_t> expected;
  for (const auto& range : ranges) {
    expected.insert(
        expected.end(),
        data.data() + range.startRow,
        data.data() + range.endRow);
  }
  std::vector<int64_t> out(expected.size());
  target->skipThenMaterialize(ranges, out.data());
  EXPECT_EQ(out, expected);

  std::vector<size_t> calls;
  for (const auto* chunk : chunks) {
    calls.push_back(chunk->numRangeCalls);
  }
  EXPECT_THAT(calls, testing::ElementsAre(1, 1, 0, 0, 1, 1, 1, 1));
}

// Real arms, chunked at a size that leaves a short last chunk, read back the
// column through every read the drivers issue.
TEST(ChunkedTargetTest, realArmsRoundTrip) {
  constexpr uint32_t kRealRows = 5'000;
  const auto data = makeData(kRealRows);
  std::vector<std::string> arms{
      "FixedBitWidth/view",
      "SIS/realNested+view",
  };
  const std::vector<nimble::RowRange> ranges{
      {3, 9},
      {2'040, 2'060},
      {4'990, 5'000},
  };
  size_t numRangeRows{0};
  for (const auto& range : ranges) {
    numRangeRows += range.numRows();
  }
  size_t numFound{0};
  auto entries = buildDefaultEncoders<int64_t>();
  // A blackbox codec, which the drivers register after the default arms.
  entries.push_back(buildOpenZLEncoder<int64_t>());
  arms.push_back("openzl/auto");
  for (const auto& entry : entries) {
    if (std::find(arms.begin(), arms.end(), entry.name) == arms.end()) {
      continue;
    }
    ++numFound;
    auto chunked = withChunking<int64_t>(entry, 2'048);
    EXPECT_EQ(chunked.name, entry.name);
    auto target = chunked.factory(data, Encoding::Options{});

    std::vector<int64_t> all(kRealRows);
    target->materializeAll(all.data(), kRealRows);
    EXPECT_THAT(all, testing::ElementsAreArray(data.data(), kRealRows))
        << entry.name;

    std::vector<int64_t> gathered(numRangeRows);
    target->skipThenMaterialize(ranges, gathered.data());
    std::vector<int64_t> expected;
    for (const auto& range : ranges) {
      expected.insert(
          expected.end(),
          data.data() + range.startRow,
          data.data() + range.endRow);
    }
    EXPECT_EQ(gathered, expected) << entry.name;
  }
  EXPECT_EQ(numFound, arms.size());
}

TEST(ChunkedTargetTest, zeroChunkRowsLeavesTheEntryAlone) {
  const auto entries = buildDefaultEncoders<int64_t>();
  ASSERT_FALSE(entries.empty());
  auto same = withChunking<int64_t>(entries.front(), 0);
  auto target = same.factory(makeData(100), Encoding::Options{});
  EXPECT_EQ(dynamic_cast<ChunkedTarget<int64_t>*>(target.get()), nullptr);
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
