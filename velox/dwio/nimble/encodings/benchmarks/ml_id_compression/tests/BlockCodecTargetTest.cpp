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

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BlockCodecTarget.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/OpenZLBenchTarget.h"

namespace facebook::nimble::mlidc {
namespace {

// Which codec an instantiation of the typed suite exercises.
enum class Codec { kZstd, kOpenZL };

template <typename T>
std::unique_ptr<BlockCodec<T>> makeCodec(Codec codec) {
  if (codec == Codec::kOpenZL) {
    return std::make_unique<OpenZLBlockCodec<T>>();
  }
  return std::make_unique<NimbleBlockCodec<T>>(CompressionType::Zstd);
}

// Compressible in the low bits and varied in the high ones, so blocks are a
// mix of the accepted and the declined paths rather than all one of them.
template <typename T>
Vector<T> makeData(uint32_t n) {
  auto& pool = benchmarks::benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  uint64_t state = 0x2545F4914F6CDD1DULL;
  for (uint32_t i = 0; i < n; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    if constexpr (std::is_floating_point_v<T>) {
      data[i] = static_cast<T>(i % 97) + static_cast<T>(0.5);
    } else {
      data[i] = static_cast<T>((state >> 40) ^ (i % 13));
    }
  }
  return data;
}

template <typename T>
std::unique_ptr<BlockCompressedTarget<T>>
encodeBlocks(Codec codec, const Vector<T>& data, uint32_t blockSize) {
  auto target = std::make_unique<BlockCompressedTarget<T>>(
      makeCodec<T>(codec), blockSize, "test");
  target->encode(data, Encoding::Options{});
  return target;
}

template <typename T>
void expectRoundTrip(Codec codec, uint32_t n, uint32_t blockSize) {
  const auto data = makeData<T>(n);
  auto target = encodeBlocks<T>(codec, data, blockSize);
  std::vector<T> out(n);
  target->materializeAll(out.data(), n);
  for (uint32_t i = 0; i < n; ++i) {
    ASSERT_EQ(out[i], data[i])
        << "index " << i << " of " << n << ", blockSize " << blockSize;
  }
}

template <typename T>
class BlockCodecTargetTest : public ::testing::Test {};

using ElementTypes =
    ::testing::Types<int32_t, uint32_t, int64_t, uint64_t, float, double>;
TYPED_TEST_SUITE(BlockCodecTargetTest, ElementTypes);

// A partial final block is the obvious place for a block codec to lose data,
// so every shape around a block boundary is round-tripped: shorter than a
// block, exactly one block, one past a block, and a single element.
TYPED_TEST(BlockCodecTargetTest, roundTripPartialFinalBlock) {
  for (const auto codec : {Codec::kZstd, Codec::kOpenZL}) {
    for (const uint32_t n : {1u, 2u, 63u, 64u, 65u, 127u, 128u, 129u, 500u}) {
      expectRoundTrip<TypeParam>(codec, n, 64);
    }
  }
}

// The shipped block sizes, at a length that is not a multiple of any of them.
TYPED_TEST(BlockCodecTargetTest, roundTripAtShippedBlockSizes) {
  for (const auto codec : {Codec::kZstd, Codec::kOpenZL}) {
    for (const uint32_t blockSize : kBlockElementCounts) {
      expectRoundTrip<TypeParam>(codec, 3'000, blockSize);
      expectRoundTrip<TypeParam>(codec, 1, blockSize);
    }
  }
}

// A range that starts inside one block and ends inside the next is where an
// off-by-one in the block bounds shows up: it either drops the elements after
// the boundary or writes them at the wrong offset.
TYPED_TEST(BlockCodecTargetTest, rangeSpanningBlockBoundary) {
  constexpr uint32_t kBlockSize = 64;
  constexpr uint32_t kRows = 200;
  const auto data = makeData<TypeParam>(kRows);

  for (const auto codec : {Codec::kZstd, Codec::kOpenZL}) {
    auto target = encodeBlocks<TypeParam>(codec, data, kBlockSize);
    const std::vector<std::pair<uint32_t, uint32_t>> ranges{
        {60, 10}, // Straddles the 64 boundary.
        {63, 2}, // The two elements either side of it.
        {0, 65}, // From the start across one boundary.
        {126, 40}, // Across two boundaries.
        {190, 10}, // Ends exactly at the end of the short final block.
        {0, kRows}, // Everything.
    };
    for (const auto& [begin, count] : ranges) {
      std::vector<TypeParam> out(count);
      target->materializeRange(begin, count, out.data());
      for (uint32_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], data[begin + i])
            << "range [" << begin << ", " << count << ")";
      }
    }
  }
}

// A point read in the short final block reads a block whose element count is
// not blockSize. Indexing that block as if it were full reads past its end.
TYPED_TEST(BlockCodecTargetTest, pointReadInFinalPartialBlock) {
  constexpr uint32_t kBlockSize = 64;
  constexpr uint32_t kRows = 200; // Final block holds 8 elements.
  const auto data = makeData<TypeParam>(kRows);

  for (const auto codec : {Codec::kZstd, Codec::kOpenZL}) {
    auto target = encodeBlocks<TypeParam>(codec, data, kBlockSize);
    for (uint32_t i = 192; i < kRows; ++i) {
      TypeParam value{};
      target->materializeRange(i, 1, &value);
      ASSERT_EQ(value, data[i]) << "point read at " << i;
    }
    // And one point read in every other block, to catch an index that is only
    // right for the first.
    for (uint32_t i = 0; i < kRows; i += 37) {
      TypeParam value{};
      target->materializeRange(i, 1, &value);
      ASSERT_EQ(value, data[i]) << "point read at " << i;
    }
  }
}

// A gather whose ranges land in several blocks, including two ranges inside
// one block and two ranges in reverse-adjacent blocks.
TYPED_TEST(BlockCodecTargetTest, gatherAcrossBlocks) {
  constexpr uint32_t kBlockSize = 64;
  constexpr uint32_t kRows = 200;
  const auto data = makeData<TypeParam>(kRows);
  const std::vector<nimble::RowRange> ranges{
      {5, 8}, {9, 13}, {70, 72}, {126, 132}, {197, 200}};

  uint32_t total = 0;
  for (const auto& range : ranges) {
    total += range.numRows();
  }

  for (const auto codec : {Codec::kZstd, Codec::kOpenZL}) {
    auto target = encodeBlocks<TypeParam>(codec, data, kBlockSize);
    std::vector<TypeParam> out(total);
    target->skipThenMaterialize(ranges, out.data());
    uint32_t cursor = 0;
    for (const auto& range : ranges) {
      const uint32_t begin = range.startRow;
      const uint32_t count = range.numRows();
      for (uint32_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[cursor + i], data[begin + i]);
      }
      cursor += count;
    }
  }
}

// The property the paper's claim turns on, asserted directly: a read
// decompresses the blocks it overlaps and no others. Without this a target
// that quietly decompressed everything would still pass every round trip
// above, and the arm would measure nothing.
TEST(BlockCodecTargetPropertyTest, decompressesOnlyOverlappedBlocks) {
  constexpr uint32_t kBlockSize = 64;
  constexpr uint32_t kRows = 1'000; // 16 blocks, the last holding 40.
  const auto data = makeData<int64_t>(kRows);
  auto target = encodeBlocks<int64_t>(Codec::kZstd, data, kBlockSize);
  ASSERT_EQ(target->numBlocks(), 16u);

  struct Read {
    uint32_t begin;
    uint32_t count;
    size_t expectedBlocks;
  };
  const std::vector<Read> reads{
      {0, 1, 1}, // Point read at the start.
      {999, 1, 1}, // Point read in the short final block.
      {512, 1, 1}, // Point read in the middle: still one block.
      {63, 2, 2}, // Straddles one boundary.
      {64, 64, 1}, // Exactly one aligned block.
      {0, 128, 2}, // Two aligned blocks.
      {1, 128, 3}, // The same span, misaligned, costs one more.
      {0, kRows, 16}, // Full scan touches every block.
  };

  for (const auto& read : reads) {
    const size_t before = target->numBlockDecodes();
    std::vector<int64_t> out(read.count);
    target->materializeRange(read.begin, read.count, out.data());
    EXPECT_EQ(target->numBlockDecodes() - before, read.expectedBlocks)
        << "read [" << read.begin << ", " << read.count << ")";
  }

  // A point read costs one block whatever the column length is, which is the
  // statement "skip latency scales with block size, not range size".
  const auto longData = makeData<int64_t>(100'000);
  auto longTarget = encodeBlocks<int64_t>(Codec::kZstd, longData, kBlockSize);
  const size_t before = longTarget->numBlockDecodes();
  int64_t value{};
  longTarget->materializeRange(99'999, 1, &value);
  EXPECT_EQ(longTarget->numBlockDecodes() - before, 1u);
  EXPECT_EQ(value, longData[99'999]);
}

// Two ranges in one block cost one decompression within a single gather, and
// nothing is cached across calls: the next call pays for its first block again.
TEST(BlockCodecTargetPropertyTest, gatherReusesOneBlockButNotAcrossCalls) {
  constexpr uint32_t kBlockSize = 64;
  const auto data = makeData<int64_t>(1'000);
  auto target = encodeBlocks<int64_t>(Codec::kZstd, data, kBlockSize);

  const std::vector<nimble::RowRange> sameBlock{{5, 8}, {40, 43}};
  std::vector<int64_t> out(6);
  size_t before = target->numBlockDecodes();
  target->skipThenMaterialize(sameBlock, out.data());
  EXPECT_EQ(target->numBlockDecodes() - before, 1u);

  before = target->numBlockDecodes();
  target->skipThenMaterialize(sameBlock, out.data());
  EXPECT_EQ(target->numBlockDecodes() - before, 1u);
}

// The block directory is part of the stored size, so payloadSize() must carry
// it. A directory omitted here would report a file no reader could address.
TEST(BlockCodecTargetPropertyTest, payloadSizeIncludesBlockDirectory) {
  constexpr uint32_t kBlockSize = 64;
  const auto data = makeData<int64_t>(1'000);
  auto target = encodeBlocks<int64_t>(Codec::kZstd, data, kBlockSize);

  ASSERT_EQ(target->numBlocks(), 16u);
  // 5 bytes per block (a 32-bit start offset and a stored-form byte), one
  // terminating offset, and an 8-byte header of element count and block size.
  EXPECT_EQ(target->metadataBytes(), 16u * 5 + 4 + 8);

  const auto buffers = target->internalBuffers();
  ASSERT_THAT(buffers, testing::SizeIs(2));
  EXPECT_EQ(
      target->payloadSize(), buffers.front().size() + target->metadataBytes());
}

// The arms the drivers pick up: correct names, no collision with openzl/auto,
// and each one round-trips through the EncoderEntry factory the drivers use.
TEST(BlockCodecTargetPropertyTest, encoderEntries) {
  auto zstd = buildZstdBlockEncoders<int64_t>();
  auto openzl = buildOpenZLBlockEncoders<int64_t>();

  std::vector<std::string> names;
  for (const auto& entry : zstd) {
    names.push_back(entry.name);
  }
  for (const auto& entry : openzl) {
    names.push_back(entry.name);
  }
  EXPECT_THAT(
      names,
      testing::ElementsAre(
          "zstd/block-1024",
          "zstd/block-65536",
          "zstd/block-262144",
          "openzl/block-1024",
          "openzl/block-65536",
          "openzl/block-262144"));
  EXPECT_THAT(names, testing::Not(testing::Contains("openzl/auto")));

  const auto data = makeData<int64_t>(3'000);
  for (const auto* entries : {&zstd, &openzl}) {
    for (const auto& entry : *entries) {
      auto target = entry.factory(data, Encoding::Options{});
      std::vector<int64_t> out(data.size());
      target->materializeAll(out.data(), data.size());
      for (uint32_t i = 0; i < data.size(); ++i) {
        ASSERT_EQ(out[i], data[i]) << entry.name << " at " << i;
      }
      EXPECT_GT(target->payloadSize(), 0u);
    }
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
