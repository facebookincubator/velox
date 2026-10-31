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

#include "velox/dwio/nimble/encodings/BlockBitPackingV2Encoding.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/SliceEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

namespace {

using Enc = nimble::BlockBitPackingV2Encoding<uint32_t>;

void appendVarint(std::string& output, uint32_t value) {
  char bytes[nimble::varint::maxVarintSizeForBitWidth(32)];
  char* pos = bytes;
  nimble::varint::writeVarint(value, &pos);
  output.append(bytes, pos - bytes);
}

// Writes an uncompressed uint32 BlockBitPackingV2 stream field by field, in the
// layout documented in BlockBitPackingV2Encoding.h.
std::string serializeStream(
    uint32_t rowCount,
    uint32_t blockSize,
    uint32_t numBlocks,
    std::string_view baselines,
    std::string_view bitWidths,
    std::string_view blockOffsets,
    std::string_view payload) {
  std::string encoded(
      nimble::EncodingPrefix::serializedSize(rowCount, /*useVarint=*/false),
      '\0');
  char* pos = encoded.data();
  nimble::EncodingPrefix::serialize(
      nimble::EncodingType::BlockBitPackingV2,
      nimble::TypeTraits<uint32_t>::dataType,
      rowCount,
      /*useVarint=*/false,
      pos);
  encoded.push_back(static_cast<char>(nimble::CompressionType::Uncompressed));
  appendVarint(encoded, blockSize);
  appendVarint(encoded, numBlocks);
  for (const auto stream : {baselines, bitWidths, blockOffsets}) {
    appendVarint(encoded, static_cast<uint32_t>(stream.size()));
    encoded.append(stream);
  }
  encoded.append(payload);
  return encoded;
}

} // namespace

class BlockBitPackingV2EncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  template <typename T>
  nimble::Vector<T> toVector(const std::vector<T>& values) {
    nimble::Vector<T> result{pool_.get()};
    result.insert(result.end(), values.data(), values.data() + values.size());
    return result;
  }

  // Encodes 'values' with Trivial per-block metadata.
  template <typename T>
  std::string_view encode(
      const std::vector<T>& values,
      const nimble::Encoding::Options& options = {},
      nimble::CompressionType compressionType =
          nimble::CompressionType::Uncompressed) {
    return nimble::test::Encoder<nimble::BlockBitPackingV2Encoding<T>>::encode(
        *buffer_, toVector(values), compressionType, options);
  }

  // Encodes 'values' as the Trivial stream encode() writes for a per-block
  // metadata stream under the test selection policy.
  template <typename T>
  std::string trivialStream(const std::vector<T>& values) {
    return std::string{
        nimble::test::Encoder<nimble::TrivialEncoding<T>>::encode(
            *buffer_, toVector(values))};
  }

  std::unique_ptr<nimble::Encoding> create(std::string_view encoded) {
    return nimble::EncodingFactory{}.create(*pool_, encoded, nullptr);
  }

  std::string_view
  slice(std::string_view encoded, uint32_t offset, uint32_t length) {
    return nimble::EncodingFactory::slice(encoded, offset, length, *buffer_);
  }

  template <typename T>
  static std::vector<T> materializeAll(nimble::Encoding& encoding) {
    std::vector<T> output(encoding.rowCount());
    encoding.materialize(encoding.rowCount(), output.data());
    return output;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

TEST_F(BlockBitPackingV2EncodingTest, roundTripsShortLastBlock) {
  // 2500 rows with blockSize 1024 give blocks of 1024, 1024 and 452 rows, with
  // bit widths 3, 4 and 5.
  std::vector<uint32_t> values(2500);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = (i / 1024) * 100'000 + i % (8u << (i / 1024));
  }

  auto encoding = create(encode(values));
  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::BlockBitPackingV2);
  EXPECT_EQ(materializeAll<uint32_t>(*encoding), values);
}

TEST_F(BlockBitPackingV2EncodingTest, roundTripsSingleShortBlock) {
  // Fewer rows than blockSize give a single block that holds all of them.
  const std::vector<uint32_t> values{7000, 7003, 7001, 7002, 7005};

  auto encoding = create(encode(values));
  EXPECT_EQ(materializeAll<uint32_t>(*encoding), values);
}

TEST_F(BlockBitPackingV2EncodingTest, roundTripsIntegerWidths) {
  // 250 rows with blockSize 100 give blocks of 100, 100 and 50 rows.
  const nimble::Encoding::Options options{.blockBitPackingBlockSize = 100};
  const auto expectRoundTrip = [&](auto typeTag) {
    using T = decltype(typeTag);
    SCOPED_TRACE(nimble::toString(nimble::TypeTraits<T>::dataType));
    std::vector<T> values(250);
    for (uint32_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<T>((i / 100) * 50 + i % 13);
    }

    auto encoding = create(encode(values, options));
    EXPECT_EQ(materializeAll<T>(*encoding), values);
  };
  expectRoundTrip(uint8_t{});
  expectRoundTrip(uint16_t{});
  expectRoundTrip(uint32_t{});
  expectRoundTrip(uint64_t{});
  expectRoundTrip(int32_t{});
  expectRoundTrip(int64_t{});
}

TEST_F(BlockBitPackingV2EncodingTest, roundTripsBlockModes) {
  // With blockSize 64: a constant block, a packed block, a raw block that spans
  // the full uint32 range, and a short packed block of 10 rows.
  std::vector<uint32_t> values(64 * 3 + 10);
  for (uint32_t i = 0; i < values.size(); ++i) {
    if (i < 64) {
      values[i] = 42;
    } else if (i < 128) {
      values[i] = 1000 + i % 16;
    } else if (i < 192) {
      values[i] =
          i % 2 == 0 ? 0 : std::numeric_limits<uint32_t>::max() - (i % 7);
    } else {
      values[i] = 500 + i % 3;
    }
  }

  const nimble::Encoding::Options options{.blockBitPackingBlockSize = 64};
  for (const auto compressionType :
       {nimble::CompressionType::Uncompressed, nimble::CompressionType::Zstd}) {
    SCOPED_TRACE(nimble::toString(compressionType));
    auto encoding = create(encode(values, options, compressionType));
    EXPECT_EQ(materializeAll<uint32_t>(*encoding), values);
  }
}

TEST_F(BlockBitPackingV2EncodingTest, skipsAcrossBlocks) {
  // blockSize 8 over 20 rows gives blocks of 8, 8 and 4 rows.
  std::vector<uint32_t> values(20);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = (i / 8) * 100 + i;
  }
  auto encoding = create(encode(values, {.blockBitPackingBlockSize = 8}));

  // Rows [5, 17) end block 0, cover block 1 and start block 2.
  encoding->skip(5);
  std::vector<uint32_t> output(12);
  encoding->materialize(/*rowCount=*/12, output.data());
  EXPECT_EQ(
      output, std::vector<uint32_t>(values.begin() + 5, values.begin() + 17));

  encoding->skip(1);
  std::vector<uint32_t> tail(2);
  encoding->materialize(/*rowCount=*/2, tail.data());
  EXPECT_EQ(tail, std::vector<uint32_t>(values.begin() + 18, values.end()));

  encoding->reset();
  EXPECT_EQ(materializeAll<uint32_t>(*encoding), values);
}

TEST_F(BlockBitPackingV2EncodingTest, writesDocumentedLayout) {
  // blockSize 4 over 7 rows gives two 2-bit packed blocks of 4 and 3 rows. The
  // payload follows the metadata streams directly; no first block row count
  // sits in between.
  const std::vector<uint32_t> values{10, 11, 12, 13, 77, 78, 79};
  constexpr int kBitWidth{2};
  const auto packBlock = [&](uint32_t blockStart, uint32_t numRows) {
    std::string block(
        nimble::FixedBitArray::bufferSize(numRows, kBitWidth), '\0');
    nimble::FixedBitArray bits{block.data(), kBitWidth};
    for (uint32_t row = 0; row < numRows; ++row) {
      bits.set(row, values[blockStart + row] - values[blockStart]);
    }
    return block;
  };
  const auto firstBlock = packBlock(/*blockStart=*/0, /*numRows=*/4);
  const auto secondBlock = packBlock(/*blockStart=*/4, /*numRows=*/3);

  const auto expected = serializeStream(
      /*rowCount=*/7,
      /*blockSize=*/4,
      /*numBlocks=*/2,
      trivialStream<uint32_t>({10, 77}),
      trivialStream<uint8_t>({kBitWidth, kBitWidth}),
      trivialStream<uint32_t>({0, static_cast<uint32_t>(firstBlock.size())}),
      firstBlock + secondBlock);
  EXPECT_EQ(encode(values, {.blockBitPackingBlockSize = 4}), expected);
}

TEST_F(BlockBitPackingV2EncodingTest, rejectsEmptyStream) {
  NIMBLE_ASSERT_THROW(
      encode(std::vector<uint32_t>{}),
      "BlockBitPackingV2 cannot encode empty streams.");
}

TEST_F(BlockBitPackingV2EncodingTest, createsFromBothFactories) {
  std::vector<uint32_t> values(100);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = 500 + i % 9;
  }
  const auto encoded = encode(values, {.blockBitPackingBlockSize = 32});

  const auto expectReads = [&](const nimble::EncodingFactory& factory) {
    auto encoding = factory.create(*pool_, encoded, nullptr);
    EXPECT_EQ(
        encoding->encodingType(), nimble::EncodingType::BlockBitPackingV2);
    EXPECT_EQ(materializeAll<uint32_t>(*encoding), values);
  };
  {
    SCOPED_TRACE("EncodingFactory");
    expectReads(nimble::EncodingFactory{});
  }
  {
    SCOPED_TRACE("legacy::EncodingFactory");
    expectReads(nimble::legacy::EncodingFactory{});
  }
}

TEST_F(BlockBitPackingV2EncodingTest, slice) {
  // Three full blocks and a short last block of 137 rows. A range on block
  // boundaries stays a BlockBitPackingV2; any other range comes back wrapped in
  // a SliceEncoding.
  constexpr uint32_t kBlockSize = Enc::kMaxBlockSize;
  constexpr uint32_t kRows = kBlockSize * 3 + 137;
  std::vector<uint32_t> values(kRows);
  for (uint32_t i = 0; i < kRows; ++i) {
    if (i < kBlockSize) {
      values[i] = 1000 + i % 4;
    } else if (i < kBlockSize * 2) {
      values[i] = i % 256;
    } else {
      values[i] = 50'000 + i % 17;
    }
  }
  const auto encoded = encode(values);

  struct Range {
    uint32_t offset;
    uint32_t length;
    nimble::EncodingType expectedType;
  };
  for (const auto& range : {
           Range{0, 128, nimble::EncodingType::Slice},
           Range{kBlockSize - 13, 64, nimble::EncodingType::Slice},
           Range{kBlockSize + 17, kBlockSize + 29, nimble::EncodingType::Slice},
           Range{
               kBlockSize, kBlockSize, nimble::EncodingType::BlockBitPackingV2},
           Range{kBlockSize * 3, 137, nimble::EncodingType::BlockBitPackingV2},
           Range{
               kBlockSize * 2,
               kBlockSize + 137,
               nimble::EncodingType::BlockBitPackingV2},
           Range{0, kRows, nimble::EncodingType::BlockBitPackingV2},
       }) {
    SCOPED_TRACE(
        testing::Message() << "offset=" << range.offset
                           << " length=" << range.length);
    const auto sliced = slice(encoded, range.offset, range.length);
    EXPECT_EQ(nimble::EncodingPrefix::encodingType(sliced), range.expectedType);
    EXPECT_EQ(
        materializeAll<uint32_t>(*create(sliced)),
        std::vector<uint32_t>(
            values.begin() + range.offset,
            values.begin() + range.offset + range.length));
  }
}

TEST_F(BlockBitPackingV2EncodingTest, sliceKeepsTouchedBlocksWhole) {
  // blockSize 8 over 30 rows. Rows [3, 13) touch blocks 0 and 1, so the slice
  // wraps those two whole blocks, 16 rows, and skips 3 of them when read.
  std::vector<uint32_t> values(30);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = 500 + i;
  }
  const auto sliced = slice(
      encode(values, {.blockBitPackingBlockSize = 8}),
      /*offset=*/3,
      /*length=*/10);
  ASSERT_EQ(
      nimble::EncodingPrefix::encodingType(sliced),
      nimble::EncodingType::Slice);

  const nimble::SliceEncoding<uint32_t> wrapper{*pool_, sliced};
  const auto inner = wrapper.innerEncoding();
  EXPECT_EQ(
      nimble::EncodingPrefix::encodingType(inner),
      nimble::EncodingType::BlockBitPackingV2);
  EXPECT_EQ(create(inner)->rowCount(), uint32_t{16});
}

TEST_F(BlockBitPackingV2EncodingTest, slicesEveryRange) {
  // blockSize 8 over 30 rows: a packed block, a constant block, a raw block and
  // a packed block of 6 rows.
  std::vector<uint32_t> values;
  values.reserve(30);
  for (uint32_t i = 0; i < 8; ++i) {
    values.push_back(1000 + i);
  }
  for (uint32_t i = 0; i < 8; ++i) {
    values.push_back(5);
  }
  for (uint32_t i = 0; i < 8; ++i) {
    values.push_back(i == 3 ? std::numeric_limits<uint32_t>::max() : i * 100);
  }
  for (uint32_t i = 0; i < 6; ++i) {
    values.push_back(2000 + i);
  }
  const auto encoded = encode(values, {.blockBitPackingBlockSize = 8});

  for (uint32_t offset = 0; offset < values.size(); ++offset) {
    for (uint32_t length = 1; offset + length <= values.size(); ++length) {
      ASSERT_EQ(
          materializeAll<uint32_t>(*create(slice(encoded, offset, length))),
          std::vector<uint32_t>(
              values.begin() + offset, values.begin() + offset + length))
          << "offset=" << offset << " length=" << length;
    }
  }
}

TEST_F(BlockBitPackingV2EncodingTest, sliceDecompressesSource) {
  constexpr uint32_t kBlockSize = Enc::kMaxBlockSize;
  std::vector<uint32_t> values(kBlockSize * 2);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = 7000 + i % 32;
  }
  const auto encoded = encode(values, {}, nimble::CompressionType::Zstd);
  const auto compressionType = [](std::string_view stream) {
    return static_cast<nimble::CompressionType>(
        stream[nimble::EncodingPrefix::prefixSize(
            stream, /*useVarint=*/false)]);
  };
  ASSERT_EQ(compressionType(encoded), nimble::CompressionType::Zstd);

  // Block-aligned, so the slice is a plain BlockBitPackingV2 whose compression
  // byte follows the prefix.
  const auto sliced = slice(encoded, kBlockSize, kBlockSize);
  ASSERT_EQ(
      nimble::EncodingPrefix::encodingType(sliced),
      nimble::EncodingType::BlockBitPackingV2);
  EXPECT_EQ(compressionType(sliced), nimble::CompressionType::Uncompressed);
  EXPECT_EQ(
      materializeAll<uint32_t>(*create(sliced)),
      std::vector<uint32_t>(values.begin() + kBlockSize, values.end()));
}

TEST_F(BlockBitPackingV2EncodingTest, estimateSizeOverloadsMatch) {
  std::vector<uint32_t> values(2500);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = (i / 1024) * 100'000 + i % 37;
  }

  const auto spanEstimate =
      Enc::estimateSize(std::span<const uint32_t>{values});
  const auto statisticsEstimate =
      Enc::estimateSize(nimble::Statistics<uint32_t>::create(values));
  ASSERT_TRUE(spanEstimate.has_value());
  ASSERT_TRUE(statisticsEstimate.has_value());
  EXPECT_EQ(spanEstimate.value(), statisticsEstimate.value());
}

TEST_F(BlockBitPackingV2EncodingTest, estimateSizeTracksEncodedSize) {
  std::vector<uint32_t> values(2500);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = (i / 1024) * 100'000 + i % 37;
  }

  // The per-block metadata is estimated rather than encoded, so allow the same
  // 2x band as the other estimate checks.
  const auto estimate =
      Enc::estimateSize(std::span<const uint32_t>{values}).value();
  const auto actual = encode(values).size();
  EXPECT_GT(estimate, actual / 2);
  EXPECT_LT(estimate, actual * 2);
}

TEST_F(BlockBitPackingV2EncodingTest, estimateSizeSkipsTooManyBlocks) {
  // blockSize 1 gives one block per row, so kMaxBlockCount + 1 rows have more
  // blocks than the uint16_t block index can address.
  const std::vector<uint32_t> atLimit(Enc::kMaxBlockCount, 3);
  const std::vector<uint32_t> overLimit(Enc::kMaxBlockCount + 1, 3);

  EXPECT_TRUE(
      Enc::estimateSize(std::span<const uint32_t>{atLimit}, 1).has_value());
  EXPECT_TRUE(
      Enc::estimateSize(nimble::Statistics<uint32_t>::create(atLimit), 1)
          .has_value());
  EXPECT_FALSE(
      Enc::estimateSize(std::span<const uint32_t>{overLimit}, 1).has_value());
  EXPECT_FALSE(
      Enc::estimateSize(nimble::Statistics<uint32_t>::create(overLimit), 1)
          .has_value());
}

TEST_F(BlockBitPackingV2EncodingTest, debugString) {
  std::vector<uint32_t> values(2500);
  for (uint32_t i = 0; i < values.size(); ++i) {
    values[i] = i % 100;
  }

  const auto debugString = create(encode(values))->debugString(0);
  EXPECT_THAT(debugString, testing::HasSubstr("BlockBitPackingV2<"));
  EXPECT_THAT(debugString, testing::HasSubstr("numBlocks=3"));
}
