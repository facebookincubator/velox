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
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <vector>
#include "velox/buffer/BufferPool.h"
#include "velox/common/base/BitUtil.h"
#include "velox/dwio/common/Lemire/BitPacking/bitpackinghelpers.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/tests/EncodingLayoutTestHelper.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

class BlockBitPackingEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  template <typename T>
  nimble::Vector<T> toVector(const std::vector<T>& v) {
    nimble::Vector<T> nv{pool_.get()};
    nv.insert(nv.end(), v.data(), v.data() + v.size());
    return nv;
  }

  template <typename T>
  void roundTrip(const std::vector<T>& input) {
    auto values = toVector(input);
    using Enc = nimble::BlockBitPackingEncoding<T>;
    auto encoding =
        nimble::test::Encoder<Enc>::createEncoding(*buffer_, values, nullptr);

    const auto count = static_cast<uint32_t>(input.size());
    std::vector<T> output(count);
    encoding->materialize(count, output.data());
    ASSERT_EQ(input, output);
  }

  template <typename T>
  void roundTripWithSkip(
      const std::vector<T>& input,
      uint32_t skipRows,
      uint32_t readRows) {
    auto values = toVector(input);
    using Enc = nimble::BlockBitPackingEncoding<T>;
    auto encoding =
        nimble::test::Encoder<Enc>::createEncoding(*buffer_, values, nullptr);

    encoding->skip(skipRows);
    std::vector<T> output(readRows);
    encoding->materialize(readRows, output.data());
    std::vector<T> expected(
        input.begin() + skipRows, input.begin() + skipRows + readRows);
    ASSERT_EQ(expected, output);
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

TEST_F(BlockBitPackingEncodingTest, basicUint32) {
  std::vector<uint32_t> values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  roundTrip(values);
}

TEST_F(BlockBitPackingEncodingTest, basicUint64) {
  std::vector<uint64_t> values = {100, 200, 300, 400, 500};
  roundTrip(values);
}

TEST_F(BlockBitPackingEncodingTest, basicUint8) {
  std::vector<uint8_t> values = {0, 1, 2, 3, 255, 128, 64};
  roundTrip(values);
}

TEST_F(BlockBitPackingEncodingTest, basicUint16) {
  std::vector<uint16_t> values = {0, 100, 200, 300, 65535, 32768};
  roundTrip(values);
}

TEST_F(BlockBitPackingEncodingTest, multipleChunks) {
  constexpr uint32_t chunkSize = 1024;
  std::vector<uint32_t> values(chunkSize * 3);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = i % 256;
  }
  roundTrip(values);
}

TEST_F(BlockBitPackingEncodingTest, variableBitWidthsPerChunk) {
  constexpr uint32_t chunkSize = 1024;
  std::vector<uint32_t> values(chunkSize * 3);
  // Block 0: values [0, 7] — 3 bits
  for (uint32_t i = 0; i < chunkSize; ++i) {
    values[i] = i % 8;
  }
  // Block 1: values [0, 255] — 8 bits
  for (uint32_t i = chunkSize; i < chunkSize * 2; ++i) {
    values[i] = i % 256;
  }
  // Block 2: values [0, 65535] — 16 bits
  for (uint32_t i = chunkSize * 2; i < chunkSize * 3; ++i) {
    values[i] = i % 65536;
  }
  roundTrip(values);
}

TEST_F(BlockBitPackingEncodingTest, baselineSubtraction) {
  constexpr uint32_t chunkSize = 1024;
  std::vector<uint32_t> values(chunkSize * 2);
  // Block 0: values [1000, 1003] — baseline=1000, 2 bits
  for (uint32_t i = 0; i < chunkSize; ++i) {
    values[i] = 1000 + (i % 4);
  }
  // Block 1: values [50000, 50001] — baseline=50000, 1 bit
  for (uint32_t i = chunkSize; i < chunkSize * 2; ++i) {
    values[i] = 50000 + (i % 2);
  }
  roundTrip(values);
}

TEST_F(BlockBitPackingEncodingTest, singleChunk) {
  std::vector<uint32_t> values(100);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = i;
  }
  roundTrip(values);
}

TEST_F(BlockBitPackingEncodingTest, constantChunk) {
  constexpr uint32_t chunkSize = 1024;
  std::vector<uint32_t> values(chunkSize);
  std::fill(values.begin(), values.end(), 42);
  roundTrip(values);
}

TEST_F(BlockBitPackingEncodingTest, mixedConstantAndVariable) {
  constexpr uint32_t chunkSize = 1024;
  std::vector<uint32_t> values(chunkSize * 2);
  // Chunk 0: constant 42
  std::fill(values.begin(), values.begin() + chunkSize, 42);
  // Chunk 1: range [0, 255]
  for (uint32_t i = chunkSize; i < chunkSize * 2; ++i) {
    values[i] = i % 256;
  }
  roundTrip(values);
}

TEST_F(BlockBitPackingEncodingTest, skipAndMaterialize) {
  constexpr uint32_t chunkSize = 1024;
  std::vector<uint32_t> values(chunkSize * 3);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = i;
  }
  // Skip into chunk 1, read across chunk boundary into chunk 2
  roundTripWithSkip(values, chunkSize + 500, chunkSize);
}

TEST_F(BlockBitPackingEncodingTest, skipToChunkBoundary) {
  constexpr uint32_t chunkSize = 1024;
  std::vector<uint32_t> values(chunkSize * 2);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = i % 100;
  }
  roundTripWithSkip(values, chunkSize, chunkSize);
}

TEST_F(BlockBitPackingEncodingTest, resetAndReread) {
  std::vector<uint32_t> values = {10, 20, 30, 40, 50};
  auto nv = toVector(values);
  using Enc = nimble::BlockBitPackingEncoding<uint32_t>;
  auto encoding =
      nimble::test::Encoder<Enc>::createEncoding(*buffer_, nv, nullptr);

  const auto count = static_cast<uint32_t>(values.size());
  std::vector<uint32_t> output1(count);
  encoding->materialize(count, output1.data());
  ASSERT_EQ(values, output1);

  encoding->reset();
  std::vector<uint32_t> output2(count);
  encoding->materialize(count, output2.data());
  ASSERT_EQ(values, output2);
}

TEST_F(BlockBitPackingEncodingTest, bufferPoolReusesMetadataBuffers) {
  using Enc = nimble::BlockBitPackingEncoding<uint32_t>;
  constexpr uint32_t blockSize = Enc::kMaxBlockSize;
  std::vector<uint32_t> input(blockSize * 8);
  for (uint32_t i = 0; i < input.size(); ++i) {
    input[i] = (i / blockSize) * 1'000 + (i % 127);
  }

  auto values = toVector(input);
  const auto encoded = nimble::test::Encoder<Enc>::encode(*buffer_, values);
  const std::vector<char> encodedStorage{encoded.begin(), encoded.end()};

  auto decodePool =
      facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  facebook::velox::BufferPool bufferPool{
      facebook::velox::BufferPool::kDefaultCapacity};
  const nimble::Encoding::Options options{.bufferPool = &bufferPool};
  std::function<void*(uint32_t)> stringBufferFactory;

  auto decodeAndVerify = [&]() {
    Enc encoding{
        *decodePool,
        {encodedStorage.data(), encodedStorage.size()},
        stringBufferFactory,
        options};
    std::vector<uint32_t> output(input.size());
    encoding.materialize(static_cast<uint32_t>(input.size()), output.data());
    EXPECT_EQ(input, output);
  };

  decodeAndVerify();
  const auto numAllocsAfterFirstDecode = decodePool->stats().numAllocs;
  EXPECT_GT(bufferPool.size(), 0);

  decodeAndVerify();
  EXPECT_EQ(decodePool->stats().numAllocs, numAllocsAfterFirstDecode);
}

TEST_F(BlockBitPackingEncodingTest, sizeVsFixedBitWidth) {
  constexpr uint32_t chunkSize = 1024;
  std::vector<uint32_t> values(chunkSize * 4);
  // Chunk 0: narrow range [100, 103]
  for (uint32_t i = 0; i < chunkSize; ++i) {
    values[i] = 100 + (i % 4);
  }
  // Chunk 1: narrow range [200, 201]
  for (uint32_t i = chunkSize; i < chunkSize * 2; ++i) {
    values[i] = 200 + (i % 2);
  }
  // Chunk 2: wide range [0, 65535]
  for (uint32_t i = chunkSize * 2; i < chunkSize * 3; ++i) {
    values[i] = i % 65536;
  }
  // Chunk 3: constant
  for (uint32_t i = chunkSize * 3; i < chunkSize * 4; ++i) {
    values[i] = 999;
  }

  auto nv = toVector(values);

  // Encode with BlockBitPacking
  nimble::Buffer cbpBuffer(*pool_);
  auto cbpEncoded =
      nimble::test::Encoder<nimble::BlockBitPackingEncoding<uint32_t>>::encode(
          cbpBuffer, nv);

  // Encode with FixedBitWidth
  nimble::Buffer fbwBuffer(*pool_);
  auto fbwEncoded =
      nimble::test::Encoder<nimble::FixedBitWidthEncoding<uint32_t>>::encode(
          fbwBuffer, nv);

  EXPECT_LT(cbpEncoded.size(), fbwEncoded.size())
      << "BlockBitPacking should produce smaller output for data with "
         "locally narrow ranges";
}

TEST_F(BlockBitPackingEncodingTest, compressionRoundTrip) {
  constexpr uint32_t chunkSize = 1024;
  std::vector<uint32_t> values(chunkSize * 2);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = i % 100;
  }
  auto nv = toVector(values);
  using Enc = nimble::BlockBitPackingEncoding<uint32_t>;

  // Round-trip works with Zstd compression requested.
  auto encoding = nimble::test::Encoder<Enc>::createEncoding(
      *buffer_, nv, nullptr, nimble::CompressionType::Zstd);
  const auto count = static_cast<uint32_t>(values.size());
  std::vector<uint32_t> output(count);
  encoding->materialize(count, output.data());
  ASSERT_EQ(values, output);
}

TEST_F(BlockBitPackingEncodingTest, slice) {
  using Enc = nimble::BlockBitPackingEncoding<uint32_t>;
  constexpr uint32_t blockSize = Enc::kMaxBlockSize;
  struct Range {
    uint32_t offset;
    uint32_t length;
  };

  std::vector<uint32_t> input(blockSize * 3 + 137);
  for (uint32_t i = 0; i < input.size(); ++i) {
    if (i < blockSize) {
      input[i] = 1000 + i % 4;
    } else if (i < blockSize * 2) {
      input[i] = i % 256;
    } else {
      input[i] = 50000 + i % 17;
    }
  }

  const auto values = toVector(input);
  const auto encoded = nimble::test::Encoder<Enc>::encode(*buffer_, values);

  for (const auto range :
       {Range{/*offset=*/0, /*length=*/128},
        Range{/*offset=*/blockSize - 13, /*length=*/64},
        Range{/*offset=*/blockSize + 17, /*length=*/blockSize + 29},
        Range{/*offset=*/static_cast<uint32_t>(input.size() - 137),
              /*length=*/137}}) {
    SCOPED_TRACE(
        testing::Message() << "offset=" << range.offset
                           << " length=" << range.length);
    nimble::Buffer sliceBuffer{*pool_};
    const auto sliced = nimble::EncodingFactory::slice(
        encoded, range.offset, range.length, sliceBuffer);

    EXPECT_EQ(
        nimble::EncodingPrefix::encodingType(sliced),
        nimble::EncodingType::BlockBitPacking);
    EXPECT_EQ(
        nimble::EncodingPrefix::readRowCount(sliced, /*useVarint=*/false),
        range.length);

    auto encoding = nimble::EncodingFactory{}.create(*pool_, sliced, nullptr);
    std::vector<uint32_t> output(range.length);
    encoding->materialize(range.length, output.data());
    const std::vector<uint32_t> expected{
        input.begin() + range.offset,
        input.begin() + range.offset + range.length};
    EXPECT_EQ(output, expected);
  }
}

TEST_F(BlockBitPackingEncodingTest, rejectsZeroLengthSlice) {
  using Enc = nimble::BlockBitPackingEncoding<uint32_t>;
  constexpr uint32_t blockSize = Enc::kMaxBlockSize;
  std::vector<uint32_t> input(blockSize + 17);
  for (uint32_t i = 0; i < input.size(); ++i) {
    input[i] = 1000 + i % 64;
  }

  const auto values = toVector(input);
  const auto encoded = nimble::test::Encoder<Enc>::encode(*buffer_, values);

  nimble::Buffer sliceBuffer{*pool_};
  NIMBLE_ASSERT_THROW(
      nimble::EncodingFactory::slice(
          encoded,
          /*offset=*/0,
          /*length=*/0,
          sliceBuffer),
      "");
}

TEST_F(BlockBitPackingEncodingTest, sliceRandomRanges) {
  using Enc = nimble::BlockBitPackingEncoding<uint32_t>;
  constexpr uint32_t blockSize = Enc::kMaxBlockSize;
  constexpr uint32_t rowCount = blockSize * 5 + 333;
  constexpr uint32_t iterations = 100;

  std::mt19937 rng{12345};
  std::vector<uint32_t> input(rowCount);
  for (uint32_t block = 0; block * blockSize < rowCount; ++block) {
    const auto begin = block * blockSize;
    const auto end = std::min<uint32_t>(begin + blockSize, rowCount);
    const auto base = static_cast<uint32_t>(rng() % 1'000'000);
    switch (block % 4) {
      case 0:
        std::fill(input.begin() + begin, input.begin() + end, base);
        break;
      case 1:
        for (uint32_t i = begin; i < end; ++i) {
          input[i] = base + static_cast<uint32_t>(rng() % 8);
        }
        break;
      case 2:
        for (uint32_t i = begin; i < end; ++i) {
          input[i] = base + static_cast<uint32_t>(rng() % 1024);
        }
        break;
      default:
        for (uint32_t i = begin; i < end; ++i) {
          input[i] = rng();
        }
        break;
    }
  }

  const auto values = toVector(input);
  const auto encoded = nimble::test::Encoder<Enc>::encode(*buffer_, values);
  auto sourceEncoding =
      nimble::EncodingFactory{}.create(*pool_, encoded, nullptr);
  std::vector<uint32_t> sourceOutput(rowCount);
  sourceEncoding->materialize(rowCount, sourceOutput.data());
  ASSERT_EQ(sourceOutput, input);

  std::uniform_int_distribution<uint32_t> offsetDistribution{0, rowCount};

  for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
    const auto offset = offsetDistribution(rng);
    std::uniform_int_distribution<uint32_t> lengthDistribution{
        0, rowCount - offset};
    const auto length = lengthDistribution(rng);
    SCOPED_TRACE(
        testing::Message() << "iteration=" << iteration << " offset=" << offset
                           << " length=" << length);

    nimble::Buffer sliceBuffer{*pool_};
    const auto sliced =
        nimble::EncodingFactory::slice(encoded, offset, length, sliceBuffer);

    auto encoding = nimble::EncodingFactory{}.create(*pool_, sliced, nullptr);
    std::vector<uint32_t> output(length);
    encoding->materialize(length, output.data());
    const std::vector<uint32_t> expected{
        input.begin() + offset, input.begin() + offset + length};
    ASSERT_EQ(output.size(), expected.size());
    for (uint32_t i = 0; i < length; ++i) {
      ASSERT_EQ(output[i], expected[i]) << "sliceRow=" << i;
    }
  }
}

TEST_F(BlockBitPackingEncodingTest, sliceInterleavedConstantAndPayloadBlocks) {
  using Enc = nimble::BlockBitPackingEncoding<uint32_t>;
  constexpr uint32_t blockSize = Enc::kMaxBlockSize;
  constexpr uint32_t rowCount = blockSize * 5;

  std::vector<uint32_t> input(rowCount);
  for (uint32_t i = 0; i < blockSize; ++i) {
    input[i] = i % 31;
  }
  std::fill(input.begin() + blockSize, input.begin() + blockSize * 2, 7);
  for (uint32_t i = blockSize * 2; i < blockSize * 3; ++i) {
    input[i] = 1000 + i % 8;
  }
  std::fill(input.begin() + blockSize * 3, input.begin() + blockSize * 4, 11);
  for (uint32_t i = blockSize * 4; i < blockSize * 5; ++i) {
    input[i] = 50000 + i % 1024;
  }

  const auto values = toVector(input);
  const auto encoded = nimble::test::Encoder<Enc>::encode(*buffer_, values);

  constexpr uint32_t offset{blockSize + 7};
  constexpr uint32_t length{blockSize * 3 + 19};
  nimble::Buffer sliceBuffer{*pool_};
  const auto sliced =
      nimble::EncodingFactory::slice(encoded, offset, length, sliceBuffer);

  auto encoding = nimble::EncodingFactory{}.create(*pool_, sliced, nullptr);
  std::vector<uint32_t> output(length);
  encoding->materialize(length, output.data());
  const std::vector<uint32_t> expected{
      input.begin() + offset, input.begin() + offset + length};
  EXPECT_EQ(output, expected);
}

TEST_F(BlockBitPackingEncodingTest, sliceCompressedSource) {
  using Enc = nimble::BlockBitPackingEncoding<uint32_t>;
  constexpr uint32_t blockSize = Enc::kMaxBlockSize;

  std::vector<uint32_t> input(blockSize * 2);
  for (uint32_t i = 0; i < input.size(); ++i) {
    input[i] = 7000 + i % 32;
  }

  const auto values = toVector(input);
  const auto encoded = nimble::test::Encoder<Enc>::encode(
      *buffer_, values, nimble::CompressionType::Zstd);

  constexpr uint32_t kOffset{31};
  constexpr uint32_t kLength{blockSize + 7};
  nimble::Buffer sliceBuffer{*pool_};
  const auto sliced =
      nimble::EncodingFactory::slice(encoded, kOffset, kLength, sliceBuffer);

  const auto prefixSize = nimble::EncodingPrefix::prefixSize(
      sliced,
      /*useVarint=*/false);
  EXPECT_EQ(
      static_cast<nimble::CompressionType>(
          static_cast<uint8_t>(sliced[prefixSize])),
      nimble::CompressionType::Uncompressed);

  auto encoding = nimble::EncodingFactory{}.create(*pool_, sliced, nullptr);
  std::vector<uint32_t> output(kLength);
  encoding->materialize(kLength, output.data());
  const std::vector<uint32_t> expected{
      input.begin() + kOffset, input.begin() + kOffset + kLength};
  EXPECT_EQ(output, expected);
}

TEST_F(BlockBitPackingEncodingTest, sliceAlignedPartialRawAndConstantPrefixes) {
  using Enc = nimble::BlockBitPackingEncoding<uint32_t>;
  constexpr uint32_t blockSize = Enc::kMaxBlockSize;

  auto verifySlice = [&](const std::vector<uint32_t>& input) {
    const auto values = toVector(input);
    const auto encoded = nimble::test::Encoder<Enc>::encode(*buffer_, values);

    constexpr uint32_t offset{blockSize};
    constexpr uint32_t length{17};
    nimble::Buffer sliceBuffer{*pool_};
    const auto sliced =
        nimble::EncodingFactory::slice(encoded, offset, length, sliceBuffer);

    auto encoding = nimble::EncodingFactory{}.create(*pool_, sliced, nullptr);
    std::vector<uint32_t> output(length);
    encoding->materialize(length, output.data());
    const std::vector<uint32_t> expected{
        input.begin() + offset, input.begin() + offset + length};
    EXPECT_EQ(output, expected);
  };

  {
    SCOPED_TRACE("raw block prefix");
    std::vector<uint32_t> input(blockSize * 2);
    std::fill(input.begin(), input.begin() + blockSize, 42);
    for (uint32_t i = blockSize; i < input.size(); ++i) {
      input[i] = i - blockSize;
    }
    input[blockSize + 1] = std::numeric_limits<uint32_t>::max();
    verifySlice(input);
  }

  {
    SCOPED_TRACE("constant block prefix");
    std::vector<uint32_t> input(blockSize * 2);
    for (uint32_t i = 0; i < blockSize; ++i) {
      input[i] = i % 31;
    }
    std::fill(input.begin() + blockSize, input.end(), 777);
    verifySlice(input);
  }
}

TEST_F(BlockBitPackingEncodingTest, uint64MultiChunk) {
  constexpr uint32_t chunkSize = 1024;
  std::vector<uint64_t> values(chunkSize * 2);
  for (uint32_t i = 0; i < chunkSize; ++i) {
    values[i] = 1000000ULL + (i % 10);
  }
  for (uint32_t i = chunkSize; i < chunkSize * 2; ++i) {
    values[i] = 9000000ULL + (i % 5);
  }
  roundTrip(values);
}

TEST_F(BlockBitPackingEncodingTest, partialLastChunk) {
  // 1500 rows: chunk 0 = 1024 rows, chunk 1 = 476 rows
  std::vector<uint32_t> values(1500);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = i % 50;
  }
  roundTrip(values);
}

TEST_F(BlockBitPackingEncodingTest, debugString) {
  std::vector<uint32_t> values(2048);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = i % 100;
  }
  auto nv = toVector(values);
  using Enc = nimble::BlockBitPackingEncoding<uint32_t>;
  auto encoding =
      nimble::test::Encoder<Enc>::createEncoding(*buffer_, nv, nullptr);
  auto debugStr = encoding->debugString(0);
  EXPECT_NE(debugStr.find("BlockBitPacking"), std::string::npos);
  EXPECT_NE(debugStr.find("numBlocks=2"), std::string::npos);
}

// Verifies that SparseBool with BlockBitPacking indices produces the same
// null bitmap as SparseBool with Trivial indices. Regression test for a bug
// where CBP indices caused incorrect null bitmaps, breaking filter pushdown
// on sparse nullable columns.
TEST_F(BlockBitPackingEncodingTest, sparseBoolWithCbpIndices) {
  // Simulate a very sparse column: ~100 non-null positions out of 100,000 rows.
  // Mimics the data pattern of first_action_date_ai_active in production
  // (258K non-null out of 2.9B rows, ~11K gap between adjacent non-nulls).
  constexpr uint32_t totalRows = 100000;
  constexpr uint32_t nonNullCount = 100;
  constexpr uint32_t gap = totalRows / nonNullCount;

  nimble::Vector<bool> boolData{pool_.get()};
  boolData.resize(totalRows, false);
  for (uint32_t i = 0; i < nonNullCount; ++i) {
    boolData[i * gap] = true;
  }
  std::span<const bool> boolSpan(boolData.data(), boolData.size());

  // Encode with Trivial indices (baseline).
  nimble::EncodingLayout trivialLayout{
      nimble::EncodingType::SparseBool,
      {},
      nimble::CompressionType::Uncompressed,
      {nimble::EncodingLayout{
          nimble::EncodingType::Trivial,
          {},
          nimble::CompressionType::Uncompressed}}};
  auto trivialPolicy =
      std::make_unique<nimble::ReplayedEncodingSelectionPolicy<bool>>(
          trivialLayout,
          nimble::CompressionOptions{},
          [](nimble::DataType type)
              -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
            UNIQUE_PTR_FACTORY(type, nimble::TrivialNestedPolicy);
          });
  auto trivialEncoded = nimble::EncodingFactory::encode<bool>(
      std::move(trivialPolicy), boolSpan, *buffer_);
  auto trivialEncoding = nimble::EncodingFactory().create(
      *pool_, trivialEncoded, nullptr, nimble::Encoding::Options{});

  // Encode with BlockBitPacking indices.
  nimble::EncodingLayout cbpLayout{
      nimble::EncodingType::SparseBool,
      {},
      nimble::CompressionType::Uncompressed,
      {nimble::EncodingLayout{
          nimble::EncodingType::BlockBitPacking,
          {},
          nimble::CompressionType::Uncompressed,
          {std::nullopt, std::nullopt, std::nullopt}}}};
  nimble::Buffer cbpBuffer(*pool_);
  auto cbpPolicy =
      std::make_unique<nimble::ReplayedEncodingSelectionPolicy<bool>>(
          cbpLayout,
          nimble::CompressionOptions{},
          [](nimble::DataType type)
              -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
            UNIQUE_PTR_FACTORY(type, nimble::TrivialNestedPolicy);
          });
  auto cbpEncoded = nimble::EncodingFactory::encode<bool>(
      std::move(cbpPolicy), boolSpan, cbpBuffer);
  auto cbpEncoding = nimble::EncodingFactory().create(
      *pool_, cbpEncoded, nullptr, nimble::Encoding::Options{});

  // Compare materializeBoolsAsBits output.
  auto* trivialSB =
      dynamic_cast<nimble::SparseBoolEncoding*>(trivialEncoding.get());
  auto* cbpSB = dynamic_cast<nimble::SparseBoolEncoding*>(cbpEncoding.get());
  ASSERT_NE(trivialSB, nullptr);
  ASSERT_NE(cbpSB, nullptr);

  const auto numWords = velox::bits::nwords(totalRows);
  std::vector<uint64_t> trivialBits(numWords, 0);
  std::vector<uint64_t> cbpBits(numWords, 0);

  trivialSB->materializeBoolsAsBits(totalRows, trivialBits.data(), 0);
  cbpSB->materializeBoolsAsBits(totalRows, cbpBits.data(), 0);

  for (uint32_t i = 0; i < totalRows; ++i) {
    bool trivialVal = velox::bits::isBitSet(trivialBits.data(), i);
    bool cbpVal = velox::bits::isBitSet(cbpBits.data(), i);
    ASSERT_EQ(trivialVal, cbpVal)
        << "Mismatch at row " << i << ": trivial=" << trivialVal
        << " cbp=" << cbpVal;
    ASSERT_EQ(trivialVal, boolData[i]) << "Wrong value at row " << i;
  }
}

// Same test but with larger data and irregular gaps to stress chunk boundaries.
TEST_F(BlockBitPackingEncodingTest, sparseBoolWithCbpIndicesLargeIrregular) {
  constexpr uint32_t totalRows = 500000;
  nimble::Vector<bool> boolData{pool_.get()};
  boolData.resize(totalRows, false);

  std::mt19937 rng(42);
  uint32_t pos = 0;
  uint32_t nonNullCount = 0;
  while (pos < totalRows) {
    boolData[pos] = true;
    ++nonNullCount;
    pos += 5000 + (rng() % 20000);
  }
  std::span<const bool> boolSpan(boolData.data(), boolData.size());

  nimble::EncodingLayout trivialLayout{
      nimble::EncodingType::SparseBool,
      {},
      nimble::CompressionType::Uncompressed,
      {nimble::EncodingLayout{
          nimble::EncodingType::Trivial,
          {},
          nimble::CompressionType::Uncompressed}}};
  auto trivialPolicy =
      std::make_unique<nimble::ReplayedEncodingSelectionPolicy<bool>>(
          trivialLayout,
          nimble::CompressionOptions{},
          [](nimble::DataType type)
              -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
            UNIQUE_PTR_FACTORY(type, nimble::TrivialNestedPolicy);
          });
  auto trivialEncoded = nimble::EncodingFactory::encode<bool>(
      std::move(trivialPolicy), boolSpan, *buffer_);
  auto trivialEncoding = nimble::EncodingFactory().create(
      *pool_, trivialEncoded, nullptr, nimble::Encoding::Options{});

  nimble::EncodingLayout cbpLayout{
      nimble::EncodingType::SparseBool,
      {},
      nimble::CompressionType::Uncompressed,
      {nimble::EncodingLayout{
          nimble::EncodingType::BlockBitPacking,
          {},
          nimble::CompressionType::Uncompressed,
          {std::nullopt, std::nullopt, std::nullopt}}}};
  nimble::Buffer cbpBuffer(*pool_);
  auto cbpPolicy =
      std::make_unique<nimble::ReplayedEncodingSelectionPolicy<bool>>(
          cbpLayout,
          nimble::CompressionOptions{},
          [](nimble::DataType type)
              -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
            UNIQUE_PTR_FACTORY(type, nimble::TrivialNestedPolicy);
          });
  auto cbpEncoded = nimble::EncodingFactory::encode<bool>(
      std::move(cbpPolicy), boolSpan, cbpBuffer);
  auto cbpEncoding = nimble::EncodingFactory().create(
      *pool_, cbpEncoded, nullptr, nimble::Encoding::Options{});

  auto* trivialSB =
      dynamic_cast<nimble::SparseBoolEncoding*>(trivialEncoding.get());
  auto* cbpSB = dynamic_cast<nimble::SparseBoolEncoding*>(cbpEncoding.get());
  ASSERT_NE(trivialSB, nullptr);
  ASSERT_NE(cbpSB, nullptr);

  const auto numWords = velox::bits::nwords(totalRows);
  std::vector<uint64_t> trivialBits(numWords, 0);
  std::vector<uint64_t> cbpBits(numWords, 0);

  trivialSB->materializeBoolsAsBits(totalRows, trivialBits.data(), 0);
  cbpSB->materializeBoolsAsBits(totalRows, cbpBits.data(), 0);

  for (uint32_t i = 0; i < totalRows; ++i) {
    bool trivialVal = velox::bits::isBitSet(trivialBits.data(), i);
    bool cbpVal = velox::bits::isBitSet(cbpBits.data(), i);
    ASSERT_EQ(trivialVal, cbpVal)
        << "Mismatch at row " << i << ": trivial=" << trivialVal
        << " cbp=" << cbpVal << " (nonNullCount=" << nonNullCount << ")";
  }
}

// Exercises the skip-encoding path (bw=255 on disk) where packing is not
// beneficial because values span the full type range within a block.
TEST_F(BlockBitPackingEncodingTest, skipEncodingFullRange) {
  constexpr uint32_t blockSize = 1024;
  std::vector<uint32_t> values(blockSize);
  // Range [0, 0xFFFFFFFF] → bw=32 → packedSize == rawSize → skip encoding.
  for (uint32_t i = 0; i < blockSize; ++i) {
    values[i] = i;
  }
  values[0] = 0;
  values[blockSize - 1] = 0xFFFFFFFF;
  roundTrip(values);
}

// Mix of skip-encoded and packed blocks in the same stream.
TEST_F(BlockBitPackingEncodingTest, skipEncodingMixedWithPacked) {
  constexpr uint32_t blockSize = 1024;
  std::vector<uint32_t> values(blockSize * 2);
  // Block 0: full range → skip encoding
  for (uint32_t i = 0; i < blockSize; ++i) {
    values[i] = i * 4194304;
  }
  values[blockSize - 1] = 0xFFFFFFFF;
  // Block 1: narrow range [0, 15] → 4-bit packing
  for (uint32_t i = blockSize; i < blockSize * 2; ++i) {
    values[i] = i % 16;
  }
  roundTrip(values);
}

// uint64 with per-block range > 2^32, exercising wide bit widths on encode.
TEST_F(BlockBitPackingEncodingTest, uint64WideBitWidth) {
  constexpr uint32_t blockSize = 1024;
  std::vector<uint64_t> values(blockSize);
  // Range [0, 5 billion] → bw=33, exceeds 32-bit boundary.
  for (uint32_t i = 0; i < blockSize; ++i) {
    values[i] = static_cast<uint64_t>(i) * 5000000;
  }
  roundTrip(values);
}

// uint64 with partial last block, verifying 64-bit padding on encode.
TEST_F(BlockBitPackingEncodingTest, uint64PartialLastBlock) {
  std::vector<uint64_t> values(1500);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = 1000000ULL + (i % 100);
  }
  roundTrip(values);
}

// uint64 partial last block with wide bit width (bw > 32).
TEST_F(BlockBitPackingEncodingTest, uint64PartialLastBlockWideBitWidth) {
  std::vector<uint64_t> values(1500);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<uint64_t>(i) * 3000000000ULL;
  }
  roundTrip(values);
}

// Every block is constant but with different baselines.
TEST_F(BlockBitPackingEncodingTest, allConstantBlocksDifferentBaselines) {
  constexpr uint32_t blockSize = 1024;
  std::vector<uint32_t> values(blockSize * 3);
  std::fill(values.begin(), values.begin() + blockSize, 100);
  std::fill(values.begin() + blockSize, values.begin() + blockSize * 2, 999999);
  std::fill(values.begin() + blockSize * 2, values.end(), 42);
  roundTrip(values);
}

// ---------------------------------------------------------------------------
// unpackTail coverage: row counts chosen so that rows-per-block is NOT a
// multiple of kGroupSize, forcing the scalar tail path in fullUnpack.
// kGroupSize: uint8→8, uint16→16, uint32/uint64→32.
// ---------------------------------------------------------------------------

// uint32: 1024+17 = 1041 rows. Block 0: 1024 (aligned), block 1: 17 rows.
// 17 % 32 = 17 → Lemire does 0 full groups, tail handles all 17.
TEST_F(BlockBitPackingEncodingTest, unpackTailUint32PureTail) {
  std::vector<uint32_t> values(1024 + 17);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = 100 + (i % 50);
  }
  roundTrip(values);
}

// uint32: 1024+45 = 1069 rows. Block 1: 45 rows.
// 45 / 32 = 1 full group + 13 tail rows.
TEST_F(BlockBitPackingEncodingTest, unpackTailUint32PartialGroup) {
  std::vector<uint32_t> values(1024 + 45);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = i % 1000;
  }
  roundTrip(values);
}

// uint64: 1024+7 rows. Block 1: 7 rows, all tail (7 < 32).
// Exercises 64-bit tail path that Impulse's fullUnpack lacks entirely.
TEST_F(BlockBitPackingEncodingTest, unpackTailUint64PureTail) {
  std::vector<uint64_t> values(1024 + 7);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = 500000ULL + (i % 200);
  }
  roundTrip(values);
}

// uint64 with wide bitWidth (bw > 32) and tail rows.
// 1024+3 rows. Block 1: 3 rows with range requiring ~33 bits.
// Exercises the cross-word branch (bitIdx + bitWidth > 64) in unpackTail.
TEST_F(BlockBitPackingEncodingTest, unpackTailUint64WideBitWidthCrossWord) {
  std::vector<uint64_t> values(1024 + 3);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<uint64_t>(i) * 5000000;
  }
  roundTrip(values);
}

// uint8: 1024+5 rows. Block 1: 5 rows (5 % 8 = 5 tail rows).
TEST_F(BlockBitPackingEncodingTest, unpackTailUint8) {
  std::vector<uint8_t> values(1024 + 5);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = i % 7;
  }
  roundTrip(values);
}

// uint16: 1024+11 rows. Block 1: 11 rows (11 % 16 = 11 tail rows).
TEST_F(BlockBitPackingEncodingTest, unpackTailUint16) {
  std::vector<uint16_t> values(1024 + 11);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = 300 + (i % 100);
  }
  roundTrip(values);
}

// Single row — entire decode is tail, no Lemire groups at all.
TEST_F(BlockBitPackingEncodingTest, unpackTailSingleRow) {
  std::vector<uint32_t> values = {42};
  roundTrip(values);
}

// Exactly kGroupSize-1 rows (31 for uint32) — maximum tail, zero full groups.
TEST_F(BlockBitPackingEncodingTest, unpackTailMaxTailUint32) {
  std::vector<uint32_t> values(31);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = i * 7;
  }
  roundTrip(values);
}

// Verifies that FixedBitArray packing produces a bit layout that fastUnpack32
// can decode correctly. This ensures the two libraries agree on the bit-level
// wire format.
TEST_F(BlockBitPackingEncodingTest, fixedBitArrayLayoutMatchesFastUnpack) {
  constexpr uint32_t kGroupSize = 32;
  std::mt19937 rng(12345);

  for (uint32_t bw = 1; bw <= 32; ++bw) {
    const uint32_t maxVal = (bw == 32) ? 0xFFFFFFFF : ((1u << bw) - 1);
    std::uniform_int_distribution<uint32_t> dist(0, maxVal);

    // Generate 32 random values that fit in bw bits.
    std::vector<uint32_t> values(kGroupSize);
    for (auto& v : values) {
      v = dist(rng);
    }

    // Pack with FixedBitArray.
    const auto bufSize = nimble::FixedBitArray::bufferSize(kGroupSize, bw);
    std::vector<char> packed(bufSize, 0);
    nimble::FixedBitArray fba(packed.data(), bw);
    fba.bulkSet32(0, kGroupSize, values.data());

    // Unpack with fastUnpack32 (same library Impulse uses).
    std::vector<uint32_t> decoded(kGroupSize, 0xDEADBEEF);
    const auto* packedWords = reinterpret_cast<const uint32_t*>(packed.data());
    velox::fastpforlib::fastunpack(packedWords, decoded.data(), bw);

    // Verify every value matches.
    for (uint32_t i = 0; i < kGroupSize; ++i) {
      ASSERT_EQ(values[i], decoded[i])
          << "Mismatch at index " << i << " with bitWidth=" << bw
          << ": packed=" << values[i] << " unpacked=" << decoded[i];
    }
  }
}

// Same test with baseline subtraction: encode with bulkSet32WithBaseline,
// decode with fastUnpack32 + manual baseline add.
TEST_F(BlockBitPackingEncodingTest, fixedBitArrayBaselineMatchesFastUnpack) {
  constexpr uint32_t kGroupSize = 32;
  std::mt19937 rng(67890);

  for (uint32_t bw = 1; bw <= 20; ++bw) {
    const uint32_t maxRange = (bw == 32) ? 0xFFFFFFFF : ((1u << bw) - 1);
    const uint32_t baseline = 1000;
    std::uniform_int_distribution<uint32_t> dist(0, maxRange);

    // Generate values = baseline + residual.
    std::vector<uint32_t> values(kGroupSize);
    for (auto& v : values) {
      v = baseline + dist(rng);
    }

    // Pack with FixedBitArray (fused baseline subtraction).
    const auto bufSize = nimble::FixedBitArray::bufferSize(kGroupSize, bw);
    std::vector<char> packed(bufSize, 0);
    nimble::FixedBitArray fba(packed.data(), bw);
    fba.bulkSetWithBaseline(0, kGroupSize, values.data(), baseline);

    // Unpack with fastUnpack32 + manual baseline add.
    std::vector<uint32_t> decoded(kGroupSize, 0xDEADBEEF);
    const auto* packedWords = reinterpret_cast<const uint32_t*>(packed.data());
    velox::fastpforlib::fastunpack(packedWords, decoded.data(), bw);
    for (auto& v : decoded) {
      v += baseline;
    }

    for (uint32_t i = 0; i < kGroupSize; ++i) {
      ASSERT_EQ(values[i], decoded[i])
          << "Mismatch at index " << i << " with bitWidth=" << bw
          << " baseline=" << baseline << ": original=" << values[i]
          << " decoded=" << decoded[i];
    }
  }
}
