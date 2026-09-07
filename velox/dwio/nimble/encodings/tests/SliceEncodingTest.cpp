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
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
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
