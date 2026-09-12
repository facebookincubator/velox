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

#include "velox/dwio/nimble/encodings/SliceEncoding.h"

#include <gtest/gtest.h>

#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/EliasFanoEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/SharedDictionaryEncodingTestUtils.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

class SliceEncodingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("SliceEncodingTest");
    pool_ = rootPool_->addLeafChild("SliceEncodingTestLeaf");
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  template <typename T>
  nimble::Vector<T> makeVector(std::initializer_list<T> values) {
    nimble::Vector<T> result{pool_.get()};
    result.insert(result.end(), values.begin(), values.end());
    return result;
  }

  std::unique_ptr<nimble::Encoding> createEncoding(std::string_view encoded) {
    return createEncoding(encoded, nimble::Encoding::Options{});
  }

  // Reads the encoded blob with the given options. Matches the options used
  // at write time -- required when useVarintRowCount is set, otherwise the
  // outer prefix parses wrong and any nested construction cascades.
  std::unique_ptr<nimble::Encoding> createEncoding(
      std::string_view encoded,
      const nimble::Encoding::Options& options) {
    return nimble::EncodingFactory{options}.create(
        *pool_, encoded, [](uint32_t /*totalLength*/) -> void* {
          return nullptr;
        });
  }

  std::string_view
  slice(std::string_view encoded, uint32_t offset, uint32_t length) {
    return slice(encoded, offset, length, nimble::Encoding::Options{});
  }

  // Slices the encoded blob with the given options. Matches the options used
  // at write time -- required when a per-encoding option (e.g.
  // useVarintRowCount) must be honoured through the factory dispatch.
  std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      const nimble::Encoding::Options& options) {
    return nimble::EncodingFactory::slice(
        encoded, offset, length, *buffer_, options);
  }

  template <typename T>
  std::vector<T> materialize(nimble::Encoding& encoding, uint32_t rowCount) {
    nimble::Vector<T> output{pool_.get(), rowCount};
    encoding.materialize(rowCount, output.data());
    return std::vector<T>(output.begin(), output.end());
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

TEST_F(SliceEncodingTest, wrapsWithoutSlicing) {
  const auto values = makeVector<int32_t>({10, 10, 12, 10, 14, 10});
  const auto encoded =
      nimble::test::Encoder<nimble::MainlyConstantEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/1, /*length=*/4, *buffer_, {});

  // The payload still carries every source row, so it is larger than the
  // source rather than smaller: the slice was recorded, not performed.
  EXPECT_GT(wrapped.size(), encoded.size());

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Slice);
  EXPECT_EQ(encoding->dataType(), nimble::DataType::Int32);
  // The row count is the slice length, not the source's.
  EXPECT_EQ(encoding->rowCount(), 4);

  const std::vector<int32_t> expected{10, 12, 10, 14};
  EXPECT_EQ(materialize<int32_t>(*encoding, 4), expected);
}

TEST_F(SliceEncodingTest, wrapsZeroOffset) {
  const auto values = makeVector<int32_t>({10, 11, 12, 13, 14});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/0, /*length=*/3, *buffer_, {});

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->rowCount(), 3);

  const std::vector<int32_t> expected{10, 11, 12};
  EXPECT_EQ(materialize<int32_t>(*encoding, 3), expected);
}

TEST_F(SliceEncodingTest, wrapsFullRange) {
  const auto values = makeVector<int32_t>({10, 11, 12});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/0, /*length=*/3, *buffer_, {});

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->rowCount(), 3);

  const std::vector<int32_t> expected{10, 11, 12};
  EXPECT_EQ(materialize<int32_t>(*encoding, 3), expected);
}

TEST_F(SliceEncodingTest, slicesEliasFano) {
  const auto values = makeVector<uint32_t>({10, 11, 15, 15, 30, 100});
  const auto encoded =
      nimble::test::Encoder<nimble::EliasFanoEncoding<uint32_t>>::encode(
          *buffer_, values);

  auto encoding = createEncoding(slice(encoded, /*offset=*/1, /*length=*/4));
  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::EliasFano);
  EXPECT_EQ(encoding->rowCount(), 4);

  const std::vector<uint32_t> expected{11, 15, 15, 30};
  EXPECT_EQ(materialize<uint32_t>(*encoding, 4), expected);
}

TEST_F(SliceEncodingTest, wrapsRle) {
  const auto values = makeVector<int32_t>({10, 10, 11, 11, 11, 12, 12});
  const auto encoded =
      nimble::test::Encoder<nimble::RLEEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/1, /*length=*/5, *buffer_, {});

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Slice);
  EXPECT_EQ(encoding->rowCount(), 5);

  const std::vector<int32_t> expected{10, 11, 11, 11, 12};
  EXPECT_EQ(materialize<int32_t>(*encoding, 5), expected);
}

TEST_F(SliceEncodingTest, wrapsBoolRle) {
  // Bool is the null-stream case: StreamSlicer reads a sliced bool stream back
  // through skip() + materializeBoolsAsBits() before any consumer sees it, so
  // the wrapper has to honour both against the slice rather than the source.
  const auto values =
      makeVector<bool>({false, false, true, true, true, false, true});
  const auto encoded = nimble::test::Encoder<nimble::RLEEncoding<bool>>::encode(
      *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<bool>::wrap(
      encoded, /*offset=*/2, /*length=*/4, *buffer_, {});

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Slice);
  EXPECT_EQ(encoding->dataType(), nimble::DataType::Bool);
  EXPECT_EQ(encoding->rowCount(), 4);

  uint64_t bits{0};
  encoding->materializeBoolsAsBits(/*rowCount=*/4, &bits, /*begin=*/0);
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 0));
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 1));
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 2));
  EXPECT_FALSE(velox::bits::isBitSet(&bits, 3));
}

// --- Deferred RLE run slicing ---------------------------------------------
//
// A slice that starts or ends mid-run keeps the boundary runs whole and wraps
// the result, instead of trimming the two boundary lengths and re-encoding.

TEST_F(SliceEncodingTest, deferredRleMatchesSourceRows) {
  // 5 runs: 10x3, 11x2, 12x4, 13x1, 14x3 over 13 rows. Sweep every non-empty
  // range so aligned, mid-run, single-run and full-range cases are all covered
  // and each must reproduce the source rows exactly.
  const auto values =
      makeVector<int32_t>({10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 14, 14, 14});
  const auto encoded =
      nimble::test::Encoder<nimble::RLEEncoding<int32_t>>::encode(
          *buffer_, values);

  for (uint32_t offset = 0; offset < values.size(); ++offset) {
    for (uint32_t length = 1; offset + length <= values.size(); ++length) {
      const std::vector<int32_t> expected(
          values.begin() + offset, values.begin() + offset + length);

      auto encoding = createEncoding(slice(encoded, offset, length));
      ASSERT_EQ(encoding->rowCount(), length)
          << "offset=" << offset << " length=" << length;
      nimble::Vector<int32_t> output{pool_.get(), length};
      encoding->materialize(length, output.data());
      ASSERT_EQ(std::vector<int32_t>(output.begin(), output.end()), expected)
          << "offset=" << offset << " length=" << length;
    }
  }
}

TEST_F(SliceEncodingTest, deferredRleWrapsOnlyWhenMidRun) {
  // Runs: 10x3 [0,3), 11x2 [3,5), 12x4 [5,9), 13x1 [9,10). The length-1 run
  // covers the aligned single-run edge case; the 10x3 run covers the mid-run
  // subcases (front only, back only, both boundaries in the same run).
  const auto values =
      makeVector<int32_t>({10, 10, 10, 11, 11, 12, 12, 12, 12, 13});
  const auto encoded =
      nimble::test::Encoder<nimble::RLEEncoding<int32_t>>::encode(
          *buffer_, values);

  struct Case {
    const char* name;
    uint32_t offset;
    uint32_t length;
    nimble::EncodingType expectedType;
  };
  for (const auto& testCase : {
           Case{"alignedSingleRun", 3, 2, nimble::EncodingType::RLE},
           Case{"alignedMultiRun", 3, 6, nimble::EncodingType::RLE},
           Case{"alignedSingleRowRun", 9, 1, nimble::EncodingType::RLE},
           Case{"alignedFullRange", 0, 10, nimble::EncodingType::RLE},
           Case{"midRunAcrossRuns", 4, 3, nimble::EncodingType::Slice},
           Case{
               "midRunInsideSingleRunFrontAndBack",
               1,
               1,
               nimble::EncodingType::Slice},
           Case{
               "midRunInsideSingleRunBackOnly",
               0,
               2,
               nimble::EncodingType::Slice},
           Case{
               "midRunInsideSingleRunFrontOnly",
               1,
               2,
               nimble::EncodingType::Slice},
       }) {
    SCOPED_TRACE(testCase.name);
    auto encoding =
        createEncoding(slice(encoded, testCase.offset, testCase.length));
    EXPECT_EQ(encoding->encodingType(), testCase.expectedType);
    EXPECT_EQ(encoding->rowCount(), testCase.length);
  }
}

TEST_F(SliceEncodingTest, deferredRleHandlesBool) {
  // Bool RLE is the FlatMap in-map and null-stream case, and the one the
  // MainlyConstant isCommon child hits.
  const auto values =
      makeVector<bool>({false, false, true, true, true, false, true, true});
  const auto encoded = nimble::test::Encoder<nimble::RLEEncoding<bool>>::encode(
      *buffer_, values);

  for (uint32_t offset = 0; offset < values.size(); ++offset) {
    for (uint32_t length = 1; offset + length <= values.size(); ++length) {
      auto encoding = createEncoding(slice(encoded, offset, length));
      ASSERT_EQ(encoding->rowCount(), length);

      uint64_t bits{0};
      encoding->materializeBoolsAsBits(length, &bits, /*begin=*/0);
      for (uint32_t i = 0; i < length; ++i) {
        ASSERT_EQ(velox::bits::isBitSet(&bits, i), values[offset + i])
            << "offset=" << offset << " length=" << length << " row=" << i;
      }
    }
  }
}

TEST_F(SliceEncodingTest, boolSkipIsRelativeToSlice) {
  const auto values =
      makeVector<bool>({false, false, true, true, true, false, true});
  const auto encoded = nimble::test::Encoder<nimble::RLEEncoding<bool>>::encode(
      *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<bool>::wrap(
      encoded, /*offset=*/2, /*length=*/4, *buffer_, {});

  auto encoding = createEncoding(wrapped);
  encoding->skip(2);

  uint64_t bits{0};
  encoding->materializeBoolsAsBits(/*rowCount=*/2, &bits, /*begin=*/0);
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 0));
  EXPECT_FALSE(velox::bits::isBitSet(&bits, 1));
}

// --- Deferred BlockBitPacking partial-block slicing ----------------------
//
// A slice that starts or ends inside a block keeps the boundary blocks whole
// and wraps the result, instead of unpack+re-packing the partial rows into
// byte-aligned slots.

// Builds a BlockBitPacking source with several blocks whose bit widths vary --
// two normal blocks, one constant block (bitWidth == 0), and one raw block
// (bitWidth == kRawBlockBitWidth). blockSize is 8 so tests cover both aligned
// and misaligned boundaries across every mode.
nimble::Vector<int32_t> makeBlockBitPackingSource(
    velox::memory::MemoryPool* pool) {
  nimble::Vector<int32_t> values{pool};
  // Block 0: narrow range [1000..1007], bit-packed at bitWidth=3.
  for (int32_t i = 0; i < 8; ++i) {
    values.push_back(1000 + i);
  }
  // Block 1: constant 5 -> bitWidth == 0.
  for (int32_t i = 0; i < 8; ++i) {
    values.push_back(5);
  }
  // Block 2: full 32-bit range -> raw block (bitWidth == 255).
  for (int32_t i = 0; i < 8; ++i) {
    values.push_back(
        i == 3 ? std::numeric_limits<int32_t>::max()
               : (i == 5 ? std::numeric_limits<int32_t>::min() : i * 100));
  }
  // Block 3: narrow again, bit-packed.
  for (int32_t i = 0; i < 8; ++i) {
    values.push_back(2000 + i);
  }
  return values;
}

TEST_F(SliceEncodingTest, deferredBlockBitPackingMatchesSourceRows) {
  const auto values = makeBlockBitPackingSource(pool_.get());
  // blockSize=8, four blocks -- sweep every non-empty range so aligned,
  // partial-front, partial-back and single-block-interior cases are all
  // covered against a raw block, a constant block, and two bit-packed ones.
  nimble::Encoding::Options writeOptions{.blockBitPackingBlockSize = 8};
  const auto encoded =
      nimble::test::Encoder<nimble::BlockBitPackingEncoding<int32_t>>::encode(
          *buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          writeOptions);

  for (uint32_t offset = 0; offset < values.size(); ++offset) {
    for (uint32_t length = 1; offset + length <= values.size(); ++length) {
      const std::vector<int32_t> expected(
          values.begin() + offset, values.begin() + offset + length);

      auto encoding = createEncoding(slice(encoded, offset, length));
      ASSERT_EQ(encoding->rowCount(), length)
          << "offset=" << offset << " length=" << length;
      nimble::Vector<int32_t> output{pool_.get(), length};
      encoding->materialize(length, output.data());
      ASSERT_EQ(std::vector<int32_t>(output.begin(), output.end()), expected)
          << "offset=" << offset << " length=" << length;
    }
  }
}

TEST_F(SliceEncodingTest, deferredBlockBitPackingWrapsOnlyWhenPartial) {
  // Four blocks of 8 rows each. Block-aligned slices come back as a plain
  // BlockBitPacking; any partial boundary -- front, back, both, or a
  // single-block interior -- comes back wrapped in a SliceEncoding.
  const auto values = makeBlockBitPackingSource(pool_.get());
  nimble::Encoding::Options writeOptions{.blockBitPackingBlockSize = 8};
  const auto encoded =
      nimble::test::Encoder<nimble::BlockBitPackingEncoding<int32_t>>::encode(
          *buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          writeOptions);

  struct Case {
    const char* name;
    uint32_t offset;
    uint32_t length;
    nimble::EncodingType expectedType;
  };
  for (const auto& testCase : {
           Case{
               "alignedSingleBlock",
               8,
               8,
               nimble::EncodingType::BlockBitPacking},
           Case{
               "alignedMultiBlock",
               8,
               16,
               nimble::EncodingType::BlockBitPacking},
           Case{
               "alignedFullSource",
               0,
               32,
               nimble::EncodingType::BlockBitPacking},
           Case{"partialFrontOnly", 3, 5, nimble::EncodingType::Slice},
           Case{"partialBackOnly", 0, 5, nimble::EncodingType::Slice},
           Case{"partialAcrossBlocks", 10, 10, nimble::EncodingType::Slice},
           Case{"partialInsideSingleBlock", 1, 3, nimble::EncodingType::Slice},
       }) {
    SCOPED_TRACE(testCase.name);
    auto encoding =
        createEncoding(slice(encoded, testCase.offset, testCase.length));
    EXPECT_EQ(encoding->encodingType(), testCase.expectedType);
    EXPECT_EQ(encoding->rowCount(), testCase.length);
  }
}

TEST_F(SliceEncodingTest, deferredBlockBitPackingWrapperTrimsToLength) {
  const auto values = makeBlockBitPackingSource(pool_.get());
  nimble::Encoding::Options writeOptions{.blockBitPackingBlockSize = 8};
  const auto encoded =
      nimble::test::Encoder<nimble::BlockBitPackingEncoding<int32_t>>::encode(
          *buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          writeOptions);

  // Partial slice starting mid-block: the wrapped BlockBitPacking covers 2
  // full blocks (block 0 + block 1 = 16 rows) and the wrapper trims to
  // length=10.
  auto partial = createEncoding(slice(encoded, 3, 10));
  EXPECT_EQ(partial->rowCount(), 10);

  const std::vector<int32_t> expected(
      values.begin() + 3, values.begin() + 3 + 10);
  nimble::Vector<int32_t> out{pool_.get(), 10};
  partial->materialize(10, out.data());
  EXPECT_EQ(std::vector<int32_t>(out.begin(), out.end()), expected);
}

TEST_F(SliceEncodingTest, deferredBlockBitPackingHandlesInt64) {
  // Same shape as the int32 test above; the write path is templated over
  // the physical type, so exercise the 8-byte payload memcpy too.
  nimble::Vector<int64_t> values{pool_.get()};
  for (int64_t i = 0; i < 8; ++i) {
    values.push_back(int64_t{1'000'000'000} + i); // bit-packed narrow range
  }
  for (int64_t i = 0; i < 8; ++i) {
    values.push_back(int64_t{-42}); // constant, bitWidth == 0
  }
  for (int64_t i = 0; i < 8; ++i) {
    values.push_back(
        i == 3 ? std::numeric_limits<int64_t>::max()
               : (i == 5 ? std::numeric_limits<int64_t>::min()
                         : int64_t{1} << (16 + i))); // raw, bitWidth == 255
  }
  nimble::Encoding::Options writeOptions{.blockBitPackingBlockSize = 8};
  const auto encoded =
      nimble::test::Encoder<nimble::BlockBitPackingEncoding<int64_t>>::encode(
          *buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          writeOptions);

  for (uint32_t offset = 0; offset < values.size(); ++offset) {
    for (uint32_t length = 1; offset + length <= values.size(); ++length) {
      const std::vector<int64_t> expected(
          values.begin() + offset, values.begin() + offset + length);
      auto deferred = createEncoding(slice(encoded, offset, length));
      ASSERT_EQ(deferred->rowCount(), length)
          << "offset=" << offset << " length=" << length;
      nimble::Vector<int64_t> out{pool_.get(), length};
      deferred->materialize(length, out.data());
      ASSERT_EQ(std::vector<int64_t>(out.begin(), out.end()), expected)
          << "offset=" << offset << " length=" << length;
    }
  }
}

TEST_F(SliceEncodingTest, deferredBlockBitPackingHandlesPartialLastBlock) {
  // 30 rows with blockSize=8 -> four blocks, the last with only 6 rows.
  // Sweep exercises blockRowCount(source, source.numBlocks-1) < blockSize
  // -- the partial last block's row count must propagate into the wrapped
  // encoding's rowCount when a slice touches it.
  nimble::Vector<int32_t> values{pool_.get()};
  for (int32_t i = 0; i < 30; ++i) {
    values.push_back(500 + i);
  }
  nimble::Encoding::Options writeOptions{.blockBitPackingBlockSize = 8};
  const auto encoded =
      nimble::test::Encoder<nimble::BlockBitPackingEncoding<int32_t>>::encode(
          *buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          writeOptions);

  for (uint32_t offset = 0; offset < values.size(); ++offset) {
    for (uint32_t length = 1; offset + length <= values.size(); ++length) {
      const std::vector<int32_t> expected(
          values.begin() + offset, values.begin() + offset + length);
      auto deferred = createEncoding(slice(encoded, offset, length));
      ASSERT_EQ(deferred->rowCount(), length)
          << "offset=" << offset << " length=" << length;
      nimble::Vector<int32_t> out{pool_.get(), length};
      deferred->materialize(length, out.data());
      ASSERT_EQ(std::vector<int32_t>(out.begin(), out.end()), expected)
          << "offset=" << offset << " length=" << length;
    }
  }
}

TEST_F(SliceEncodingTest, deferredBlockBitPackingHandlesShortSource) {
  // Source shorter than one block -> source.firstBlockRows < blockSize and
  // source.numBlocks == 1. Verifies the write path propagates a partial
  // firstBlockRows into the emitted encoding's header field verbatim.
  nimble::Vector<int32_t> values{pool_.get()};
  for (int32_t i = 0; i < 6; ++i) {
    values.push_back(7000 + i);
  }
  nimble::Encoding::Options writeOptions{.blockBitPackingBlockSize = 8};
  const auto encoded =
      nimble::test::Encoder<nimble::BlockBitPackingEncoding<int32_t>>::encode(
          *buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          writeOptions);

  for (uint32_t offset = 0; offset < values.size(); ++offset) {
    for (uint32_t length = 1; offset + length <= values.size(); ++length) {
      const std::vector<int32_t> expected(
          values.begin() + offset, values.begin() + offset + length);
      auto deferred = createEncoding(slice(encoded, offset, length));
      ASSERT_EQ(deferred->rowCount(), length)
          << "offset=" << offset << " length=" << length;
      nimble::Vector<int32_t> out{pool_.get(), length};
      deferred->materialize(length, out.data());
      ASSERT_EQ(std::vector<int32_t>(out.begin(), out.end()), expected)
          << "offset=" << offset << " length=" << length;
    }
  }
}

TEST_F(SliceEncodingTest, deferredBlockBitPackingHandlesVarintRowCount) {
  // Production StreamSlicer sets useVarintRowCount=true; the write path
  // stamps the emitted encoding's prefix with the flag, so exercise that.
  const auto values = makeBlockBitPackingSource(pool_.get());
  nimble::Encoding::Options writeOptions{
      .useVarintRowCount = true, .blockBitPackingBlockSize = 8};
  const auto encoded =
      nimble::test::Encoder<nimble::BlockBitPackingEncoding<int32_t>>::encode(
          *buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          writeOptions);

  const nimble::Encoding::Options varintOptions{.useVarintRowCount = true};
  for (uint32_t offset = 0; offset < values.size(); ++offset) {
    for (uint32_t length = 1; offset + length <= values.size(); ++length) {
      const std::vector<int32_t> expected(
          values.begin() + offset, values.begin() + offset + length);
      auto deferred = createEncoding(
          slice(encoded, offset, length, varintOptions), varintOptions);
      ASSERT_EQ(deferred->rowCount(), length)
          << "offset=" << offset << " length=" << length;
      nimble::Vector<int32_t> out{pool_.get(), length};
      deferred->materialize(length, out.data());
      ASSERT_EQ(std::vector<int32_t>(out.begin(), out.end()), expected)
          << "offset=" << offset << " length=" << length;
    }
  }
}

TEST_F(SliceEncodingTest, resetReturnsToSliceStart) {
  const auto values = makeVector<int32_t>({10, 10, 12, 10, 14, 10});
  const auto encoded =
      nimble::test::Encoder<nimble::MainlyConstantEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/1, /*length=*/4, *buffer_, {});

  auto encoding = createEncoding(wrapped);

  // reset() must return to the slice start, not to the source's row zero.
  const std::vector<int32_t> expected{10, 12, 10, 14};
  for (int pass = 0; pass < 2; ++pass) {
    EXPECT_EQ(materialize<int32_t>(*encoding, 4), expected) << "pass " << pass;
    encoding->reset();
  }
}

TEST_F(SliceEncodingTest, skipsWithinSlice) {
  const auto values = makeVector<int32_t>({10, 11, 12, 13, 14, 15});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/2, /*length=*/4, *buffer_, {});

  auto encoding = createEncoding(wrapped);
  encoding->skip(1);

  const std::vector<int32_t> expected{13, 14, 15};
  EXPECT_EQ(materialize<int32_t>(*encoding, 3), expected);
}

// --- Value delta ----------------------------------------------------------
//
// A slice can carry a signed value delta that is added to every materialized
// value at decode time. Zero deltas cost 1 byte on the wire (varint 0) and are
// exercised by the round-trip tests above; the tests below cover non-zero
// deltas, the wrapping behaviour on unsigned physical types, and the rejection
// cases enforced by wrap().

TEST_F(SliceEncodingTest, zeroDeltaTrivialRoundTrip) {
  const auto values = makeVector<int32_t>({10, 11, 12, 13, 14, 15});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/1, /*length=*/4, *buffer_, {}, /*valueDelta=*/0);

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->rowCount(), 4);
  const std::vector<int32_t> expected{11, 12, 13, 14};
  EXPECT_EQ(materialize<int32_t>(*encoding, 4), expected);
}

TEST_F(SliceEncodingTest, zeroDeltaFixedBitWidthRoundTrip) {
  const auto values = makeVector<int32_t>({100, 101, 102, 103, 104});
  const auto encoded =
      nimble::test::Encoder<nimble::FixedBitWidthEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/1, /*length=*/3, *buffer_, {}, /*valueDelta=*/0);

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->rowCount(), 3);
  const std::vector<int32_t> expected{101, 102, 103};
  EXPECT_EQ(materialize<int32_t>(*encoding, 3), expected);
}

TEST_F(SliceEncodingTest, zeroDeltaConstantRoundTrip) {
  const auto values = makeVector<int32_t>({42, 42, 42, 42, 42, 42});
  const auto encoded =
      nimble::test::Encoder<nimble::ConstantEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/2, /*length=*/3, *buffer_, {}, /*valueDelta=*/0);

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->rowCount(), 3);
  const std::vector<int32_t> expected{42, 42, 42};
  EXPECT_EQ(materialize<int32_t>(*encoding, 3), expected);
}

TEST_F(SliceEncodingTest, zeroDeltaCostsOneByte) {
  // A zero delta zigzag-encodes to 0, whose varint is a single byte. Verify
  // the wrapped size matches the pre-existing layout plus that one byte
  // instead of relying on a golden byte pattern.
  const auto values = makeVector<int32_t>({10, 11, 12, 13, 14});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/0, /*length=*/5, *buffer_, {}, /*valueDelta=*/0);

  const auto prefixSize = nimble::EncodingPrefix::serializedSize(
      /*rowCount=*/5, /*useVarint=*/false);
  EXPECT_EQ(
      wrapped.size(),
      prefixSize + sizeof(uint32_t) + /*deltaVarintBytes=*/1 + encoded.size());
}

TEST_F(SliceEncodingTest, positiveDeltaFixedBitWidth) {
  const auto values = makeVector<int32_t>({100, 101, 102, 103, 104});
  const auto encoded =
      nimble::test::Encoder<nimble::FixedBitWidthEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/1, /*length=*/3, *buffer_, {}, /*valueDelta=*/5);

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->rowCount(), 3);
  // Source[1..4) = {101, 102, 103}; +5 = {106, 107, 108}.
  const std::vector<int32_t> expected{106, 107, 108};
  EXPECT_EQ(materialize<int32_t>(*encoding, 3), expected);
}

TEST_F(SliceEncodingTest, positiveDeltaLargeInt32) {
  // Delta near INT32_MAX/2 exercises the widest-shift path that still stays
  // within int32_t range for the chosen source values.
  const auto values = makeVector<int32_t>({0, 1, 2, 3});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  constexpr int64_t kDelta = std::numeric_limits<int32_t>::max() / 2;
  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/0, /*length=*/4, *buffer_, {}, kDelta);

  auto encoding = createEncoding(wrapped);
  const std::vector<int32_t> expected{
      static_cast<int32_t>(0 + kDelta),
      static_cast<int32_t>(1 + kDelta),
      static_cast<int32_t>(2 + kDelta),
      static_cast<int32_t>(3 + kDelta)};
  EXPECT_EQ(materialize<int32_t>(*encoding, 4), expected);
}

TEST_F(SliceEncodingTest, positiveDeltaTrivialUint64) {
  const auto values = makeVector<uint64_t>({10, 20, 30, 40, 50});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<uint64_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<uint64_t>::wrap(
      encoded, /*offset=*/1, /*length=*/3, *buffer_, {}, /*valueDelta=*/1000);

  auto encoding = createEncoding(wrapped);
  const std::vector<uint64_t> expected{1020, 1030, 1040};
  EXPECT_EQ(materialize<uint64_t>(*encoding, 3), expected);
}

TEST_F(SliceEncodingTest, negativeDeltaTrivialInt32) {
  const auto values = makeVector<int32_t>({100, 101, 102, 103});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/0, /*length=*/4, *buffer_, {}, /*valueDelta=*/-3);

  auto encoding = createEncoding(wrapped);
  const std::vector<int32_t> expected{97, 98, 99, 100};
  EXPECT_EQ(materialize<int32_t>(*encoding, 4), expected);
}

TEST_F(SliceEncodingTest, negativeDeltaLargeInt64) {
  const auto values = makeVector<int64_t>({2'000'000, 2'000'001, 2'000'002});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int64_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int64_t>::wrap(
      encoded,
      /*offset=*/0,
      /*length=*/3,
      *buffer_,
      {},
      /*valueDelta=*/-1'000'000);

  auto encoding = createEncoding(wrapped);
  const std::vector<int64_t> expected{1'000'000, 1'000'001, 1'000'002};
  EXPECT_EQ(materialize<int64_t>(*encoding, 3), expected);
}

TEST_F(SliceEncodingTest, uint32DeltaWrapsAsPhysicalAdd) {
  // A negative int64 delta cast to uint32 lands near 2^32, so materialized
  // values wrap modulo 2^32 -- exactly the semantics of physical add on the
  // unsigned physical type.
  const auto values = makeVector<uint32_t>({100, 200, 300});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<uint32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<uint32_t>::wrap(
      encoded, /*offset=*/0, /*length=*/3, *buffer_, {}, /*valueDelta=*/-6);

  auto encoding = createEncoding(wrapped);
  const auto out = materialize<uint32_t>(*encoding, 3);
  // (source[i] + 4294967290) mod 2^32 = source[i] - 6 in unsigned arithmetic.
  const std::vector<uint32_t> expected{
      static_cast<uint32_t>(100 - 6),
      static_cast<uint32_t>(200 - 6),
      static_cast<uint32_t>(300 - 6)};
  EXPECT_EQ(out, expected);
}

TEST_F(SliceEncodingTest, rejectDeltaOnFloat) {
  const auto values = makeVector<float>({1.0f, 2.0f, 3.0f});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<float>>::encode(
          *buffer_, values);

  EXPECT_THROW(
      nimble::SliceEncoding<float>::wrap(
          encoded, /*offset=*/0, /*length=*/3, *buffer_, {}, /*valueDelta=*/5),
      nimble::NimbleInternalError);
}

TEST_F(SliceEncodingTest, rejectDeltaOnBool) {
  const auto values = makeVector<bool>({false, true, false, true, true, false});
  const auto encoded = nimble::test::Encoder<nimble::RLEEncoding<bool>>::encode(
      *buffer_, values);

  EXPECT_THROW(
      nimble::SliceEncoding<bool>::wrap(
          encoded, /*offset=*/0, /*length=*/6, *buffer_, {}, /*valueDelta=*/3),
      nimble::NimbleInternalError);
}

TEST_F(SliceEncodingTest, rejectDeltaOnString) {
  const auto values = [&]() {
    nimble::Vector<std::string_view> result{pool_.get()};
    result.push_back("alpha");
    result.push_back("beta");
    result.push_back("gamma");
    return result;
  }();
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<std::string_view>>::encode(
          *buffer_, values);

  EXPECT_THROW(
      nimble::SliceEncoding<std::string_view>::wrap(
          encoded, /*offset=*/0, /*length=*/3, *buffer_, {}, /*valueDelta=*/1),
      nimble::NimbleInternalError);
}

TEST_F(SliceEncodingTest, wrapAcceptsDeltaOnSharedDictionaryInner) {
  // Write-time push-down declines for SharedDictionary (folding the shift
  // into the shared alphabet or the indices would corrupt reads), but wrap()
  // still accepts the pair: the shift stays on the wire and the read-time
  // materialize loop applies it to the values the alphabet resolves. Prove
  // the write-time acceptance with a synthetic prefix tagged SharedDictionary
  // -- a real SharedDictionaryEncoding pulls in the alphabet plumbing and is
  // exercised by the end-to-end test below.
  const uint32_t prefixSize = nimble::EncodingPrefix::serializedSize(
      /*rowCount=*/0, /*useVarint=*/false);
  char* reserved = buffer_->reserve(prefixSize);
  char* pos = reserved;
  nimble::EncodingPrefix::serialize(
      nimble::EncodingType::SharedDictionary,
      nimble::DataType::Int32,
      /*rowCount=*/0,
      /*useVarint=*/false,
      pos);
  const std::string_view syntheticInner{reserved, prefixSize};

  EXPECT_NO_THROW(
      nimble::SliceEncoding<int32_t>::wrap(
          syntheticInner,
          /*offset=*/0,
          /*length=*/0,
          *buffer_,
          {},
          /*valueDelta=*/1));
  EXPECT_NO_THROW(
      nimble::SliceEncoding<int32_t>::wrap(
          syntheticInner,
          /*offset=*/0,
          /*length=*/0,
          *buffer_,
          {},
          /*valueDelta=*/0));
}

TEST_F(SliceEncodingTest, deltaAppliesOverSharedDictionaryInner) {
  // End-to-end: build a real SharedDictionaryEncoding<int32_t>, wrap it in a
  // SliceEncoding with a non-zero delta, and confirm the materialized values
  // are alphabet[indices[i]] + delta. Push-down cannot fold the delta into
  // the shared alphabet or the indices, so the on-wire delta stays non-zero
  // and the read path picks it up at materialize().
  const std::vector<int32_t> alphabetValues{10, 20, 30, 40};
  const std::vector<uint32_t> indices{0, 1, 2, 3, 2, 1};

  nimble::Encoding::Options options;
  options.sharedDictionaryAlphabet =
      nimble::test::createSharedDictionaryAlphabet<int32_t>(
          alphabetValues, std::span<const nimble::EncodingType>{}, pool_.get());

  const auto encodedSharedDict =
      nimble::SharedDictionaryEncoding<int32_t>::encode(
          indices,
          [](nimble::DataType dataType) {
            nimble::ManualEncodingSelectionPolicyFactory factory{
                {{nimble::EncodingType::FixedBitWidth, 1.0}}, std::nullopt};
            return factory.createPolicy(dataType);
          },
          *buffer_,
          options);

  const int64_t delta = 5;
  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encodedSharedDict,
      /*offset=*/0,
      /*length=*/static_cast<uint32_t>(indices.size()),
      *buffer_,
      options,
      delta);

  auto encoding = createEncoding(wrapped, options);
  const std::vector<int32_t> expected{15, 25, 35, 45, 35, 25};
  EXPECT_EQ(materialize<int32_t>(*encoding, indices.size()), expected);
}

TEST_F(SliceEncodingTest, zeroDeltaOnFloatOK) {
  const auto values = makeVector<float>({1.5f, 2.5f, 3.5f, 4.5f});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<float>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<float>::wrap(
      encoded, /*offset=*/1, /*length=*/2, *buffer_, {}, /*valueDelta=*/0);

  auto encoding = createEncoding(wrapped);
  const std::vector<float> expected{2.5f, 3.5f};
  EXPECT_EQ(materialize<float>(*encoding, 2), expected);
}

TEST_F(SliceEncodingTest, zeroDeltaOnBoolStillWorks) {
  const auto values =
      makeVector<bool>({false, true, true, false, true, true, false});
  const auto encoded = nimble::test::Encoder<nimble::RLEEncoding<bool>>::encode(
      *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<bool>::wrap(
      encoded, /*offset=*/1, /*length=*/4, *buffer_, {}, /*valueDelta=*/0);

  auto encoding = createEncoding(wrapped);
  EXPECT_EQ(encoding->rowCount(), 4);
  uint64_t bits{0};
  encoding->materializeBoolsAsBits(/*rowCount=*/4, &bits, /*begin=*/0);
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 0)); // values[1] = true
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 1)); // values[2] = true
  EXPECT_FALSE(velox::bits::isBitSet(&bits, 2)); // values[3] = false
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 3)); // values[4] = true
}

TEST_F(SliceEncodingTest, resetPreservesDelta) {
  const auto values = makeVector<int32_t>({10, 11, 12, 13, 14});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int32_t>::wrap(
      encoded, /*offset=*/1, /*length=*/3, *buffer_, {}, /*valueDelta=*/7);

  auto encoding = createEncoding(wrapped);

  // reset() rebuilds the inner encoding and re-skips to sliceOffset_, but the
  // delta is a data-member constant and applies on every materialize().
  const std::vector<int32_t> expected{18, 19, 20};
  for (int pass = 0; pass < 2; ++pass) {
    EXPECT_EQ(materialize<int32_t>(*encoding, 3), expected) << "pass " << pass;
    encoding->reset();
  }
}

TEST_F(SliceEncodingTest, rejectDeltaOutOfPhysicalTypeRangeOnRead) {
  // wrap() has no range check on the delta today, so a caller can hand it a
  // value that does not fit in the physical type. The constructor's guard
  // catches it -- otherwise materialize() would silently alias the shift.
  // Use RLE as the inner: it is stable across follow-up push-down work as an
  // encoding that never absorbs a delta, so the out-of-range delta stays on
  // the wire where the reader-side guard can see it.
  const auto values = makeVector<int8_t>({10, 10, 10, 20});
  const auto encoded =
      nimble::test::Encoder<nimble::RLEEncoding<int8_t>>::encode(
          *buffer_, values);

  const auto wrapped = nimble::SliceEncoding<int8_t>::wrap(
      encoded, /*offset=*/0, /*length=*/4, *buffer_, {}, /*valueDelta=*/500);

  EXPECT_THROW(createEncoding(wrapped), nimble::NimbleInternalError);
}
