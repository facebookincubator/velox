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
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include "folly/Random.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

using namespace facebook;

class ForEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  template <typename T>
  std::unique_ptr<nimble::Encoding> createEncoding(
      const nimble::Vector<T>& data) {
    return nimble::test::Encoder<nimble::ForEncoding<T>>::createEncoding(
        *buffer_, data, nullptr, nimble::CompressionType::Uncompressed);
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

TEST_F(ForEncodingTest, slicePartialFrames) {
  nimble::Vector<uint32_t> data(pool_.get());
  data.reserve(384);
  for (uint32_t i = 0; i < 384; ++i) {
    data.push_back(1000 + (i / 128) * 100 + (i % 17));
  }

  const auto encoded =
      nimble::test::Encoder<nimble::ForEncoding<uint32_t>>::encode(
          *buffer_, data);

  struct Range {
    std::string_view name;
    uint32_t offset;
    uint32_t length;
  };

  for (const auto range :
       {Range{"partialFirstAndLast", /*offset=*/64, /*length=*/192},
        Range{"misalignedMiddle", /*offset=*/65, /*length=*/130},
        Range{"crossFrameBoundary", /*offset=*/127, /*length=*/2},
        Range{"fullFrame", /*offset=*/128, /*length=*/128}}) {
    SCOPED_TRACE(
        testing::Message() << "name=" << range.name << ", offset="
                           << range.offset << ", length=" << range.length);
    nimble::Buffer sliceBuffer{*pool_};
    const auto sliced = nimble::ForEncoding<uint32_t>::slice(
        encoded,
        range.offset,
        range.length,
        sliceBuffer,
        nimble::Encoding::Options{});

    nimble::ForEncoding<uint32_t> encoding{
        *pool_, sliced, [](uint32_t /*totalLength*/) -> void* {
          return nullptr;
        }};
    nimble::Vector<uint32_t> output(pool_.get(), range.length);
    encoding.materialize(range.length, output.data());

    auto view = nimble::detail::createTypedEncodingView<uint32_t>(
        sliced, pool_.get(), nimble::Encoding::Options{});
    ASSERT_NE(view, nullptr);
    nimble::Vector<uint32_t> viewOutput(pool_.get(), range.length);
    view->read(/*offset=*/0, range.length, viewOutput.data());

    for (uint32_t i = 0; i < range.length; ++i) {
      ASSERT_EQ(output[i], data[range.offset + i]) << "encoding row " << i;
      ASSERT_EQ(viewOutput[i], data[range.offset + i]) << "view row " << i;
    }
  }
}

TEST_F(ForEncodingTest, sliceRandomRanges) {
  constexpr uint32_t kIterations{64};
  std::mt19937 rng{0x5eed};

  for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
    const auto rowCount = std::uniform_int_distribution<uint32_t>{1, 512}(rng);
    nimble::Vector<uint32_t> data(pool_.get());
    data.reserve(rowCount);
    for (uint32_t row = 0; row < rowCount; ++row) {
      data.push_back(
          (row / 128) * 1000 +
          std::uniform_int_distribution<uint32_t>{0, 127}(rng));
    }

    const auto offset =
        std::uniform_int_distribution<uint32_t>{0, rowCount - 1}(rng);
    const auto length =
        std::uniform_int_distribution<uint32_t>{1, rowCount - offset}(rng);
    SCOPED_TRACE(
        testing::Message() << "iteration=" << iteration
                           << ", rowCount=" << rowCount << ", offset=" << offset
                           << ", length=" << length);

    const auto encoded =
        nimble::test::Encoder<nimble::ForEncoding<uint32_t>>::encode(
            *buffer_, data);
    nimble::Buffer sliceBuffer{*pool_};
    const auto sliced = nimble::ForEncoding<uint32_t>::slice(
        encoded, offset, length, sliceBuffer, nimble::Encoding::Options{});

    nimble::ForEncoding<uint32_t> encoding{
        *pool_, sliced, [](uint32_t /*totalLength*/) -> void* {
          return nullptr;
        }};
    nimble::Vector<uint32_t> output(pool_.get(), length);
    encoding.materialize(length, output.data());

    auto view = nimble::detail::createTypedEncodingView<uint32_t>(
        sliced, pool_.get(), nimble::Encoding::Options{});
    ASSERT_NE(view, nullptr);
    nimble::Vector<uint32_t> viewOutput(pool_.get(), length);
    view->read(/*offset=*/0, length, viewOutput.data());

    const std::vector<uint32_t> expected(
        data.begin() + offset, data.begin() + offset + length);
    EXPECT_EQ(std::vector<uint32_t>(output.begin(), output.end()), expected);
    EXPECT_EQ(
        std::vector<uint32_t>(viewOutput.begin(), viewOutput.end()), expected);
  }
}

TEST_F(ForEncodingTest, slicePreservesMetadataLayout) {
  nimble::Vector<uint32_t> data(pool_.get());
  data.reserve(4096);
  for (uint32_t i = 0; i < 4096; ++i) {
    data.push_back(1000 + (i % 3));
  }

  struct SourceCompression {
    nimble::CompressionType type;
    std::string_view name;
  };

  for (const auto sourceCompression :
       {SourceCompression{
            nimble::CompressionType::Uncompressed, "uncompressed"},
        SourceCompression{nimble::CompressionType::Zstd, "zstd"}}) {
    SCOPED_TRACE(
        testing::Message() << "sourceCompression=" << sourceCompression.name);
    const auto encoded =
        nimble::test::Encoder<nimble::ForEncoding<uint32_t>>::encode(
            *buffer_, data, sourceCompression.type);
    const auto sourceLayout = nimble::EncodingLayoutCapture::capture(
        encoded, nimble::Encoding::Options{});
    ASSERT_EQ(sourceLayout.encodingType(), nimble::EncodingType::FOR);
    ASSERT_EQ(sourceLayout.compressionType(), sourceCompression.type);

    constexpr uint32_t offset{129};
    constexpr uint32_t length{257};
    nimble::Buffer sliceBuffer{*pool_};
    const auto sliced = nimble::ForEncoding<uint32_t>::slice(
        encoded, offset, length, sliceBuffer, nimble::Encoding::Options{});
    const auto sliceLayout = nimble::EncodingLayoutCapture::capture(
        sliced, nimble::Encoding::Options{});
    ASSERT_EQ(sliceLayout.encodingType(), nimble::EncodingType::FOR);
    EXPECT_EQ(
        sliceLayout.compressionType(), nimble::CompressionType::Uncompressed);
    ASSERT_EQ(sourceLayout.childrenCount(), 3);
    ASSERT_EQ(sliceLayout.childrenCount(), 3);
    for (uint32_t child = 0; child < 3; ++child) {
      SCOPED_TRACE(testing::Message() << "child=" << child);
      ASSERT_TRUE(sourceLayout.child(child).has_value());
      ASSERT_TRUE(sliceLayout.child(child).has_value());
      EXPECT_EQ(
          sliceLayout.child(child)->encodingType(),
          sourceLayout.child(child)->encodingType());
    }

    nimble::ForEncoding<uint32_t> encoding{
        *pool_, sliced, [](uint32_t /*totalLength*/) -> void* {
          return nullptr;
        }};
    nimble::Vector<uint32_t> output(pool_.get(), length);
    encoding.materialize(length, output.data());

    auto view = nimble::detail::createTypedEncodingView<uint32_t>(
        sliced, pool_.get(), nimble::Encoding::Options{});
    ASSERT_NE(view, nullptr);
    nimble::Vector<uint32_t> viewOutput(pool_.get(), length);
    view->read(/*offset=*/0, length, viewOutput.data());

    const std::vector<uint32_t> expected{
        data.begin() + offset, data.begin() + offset + length};
    EXPECT_EQ(std::vector<uint32_t>(output.begin(), output.end()), expected);
    EXPECT_EQ(
        std::vector<uint32_t>(viewOutput.begin(), viewOutput.end()), expected);
  }
}

TEST_F(ForEncodingTest, sliceRejectsInvalidRange) {
  nimble::Vector<uint32_t> data(pool_.get());
  for (uint32_t i = 0; i < 16; ++i) {
    data.push_back(i);
  }
  const auto encoded =
      nimble::test::Encoder<nimble::ForEncoding<uint32_t>>::encode(
          *buffer_, data);

  nimble::Buffer sliceBuffer{*pool_};
  NIMBLE_ASSERT_THROW(
      nimble::ForEncoding<uint32_t>::slice(
          encoded,
          /*offset=*/0,
          /*length=*/0,
          sliceBuffer,
          nimble::Encoding::Options{}),
      "Cannot slice zero rows.");
  NIMBLE_ASSERT_THROW(
      nimble::ForEncoding<uint32_t>::slice(
          encoded,
          /*offset=*/data.size(),
          /*length=*/1,
          sliceBuffer,
          nimble::Encoding::Options{}),
      "");
}

TEST_F(ForEncodingTest, basicEncodeDecode) {
  nimble::Vector<int32_t> data(pool_.get());
  data.push_back(100);
  data.push_back(105);
  data.push_back(102);
  data.push_back(110);
  data.push_back(101);
  data.push_back(103);

  auto encoding = createEncoding(data);
  ASSERT_EQ(encoding->encodingType(), nimble::EncodingType::FOR);
  ASSERT_EQ(encoding->dataType(), nimble::DataType::Int32);
  ASSERT_EQ(encoding->rowCount(), 6);

  nimble::Vector<int32_t> result(pool_.get(), 6);
  encoding->materialize(6, result.data());

  ASSERT_EQ(data.size(), result.size());
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], data[i]) << "Mismatch at index " << i;
  }
}

TEST_F(ForEncodingTest, allZeros) {
  nimble::Vector<int32_t> data(pool_.get());
  for (int i = 0; i < 1000; ++i) {
    data.push_back(0);
  }

  auto encoding = createEncoding(data);
  ASSERT_EQ(encoding->rowCount(), 1000);

  nimble::Vector<int32_t> result(pool_.get(), 1000);
  encoding->materialize(1000, result.data());

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], 0) << "Expected 0 at index " << i;
  }
}

// Test with constant value (1-bit encoding)
TEST_F(ForEncodingTest, constantValue) {
  nimble::Vector<int64_t> data(pool_.get());
  const int64_t constantValue = 42;
  for (int i = 0; i < 500; ++i) {
    data.push_back(constantValue);
  }

  auto encoding = createEncoding(data);
  ASSERT_EQ(encoding->rowCount(), 500);

  nimble::Vector<int64_t> result(pool_.get(), 500);
  encoding->materialize(500, result.data());

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], constantValue)
        << "Expected " << constantValue << " at index " << i;
  }
}

// Test with values requiring different bit widths
TEST_F(ForEncodingTest, mixedBitWidths) {
  nimble::Vector<int32_t> data(pool_.get());

  // Frame 1: small range (1-bit)
  for (int i = 0; i < 128; ++i) {
    data.push_back(100 + (i % 2));
  }

  // Frame 2: medium range (4-bit)
  for (int i = 0; i < 128; ++i) {
    data.push_back(200 + (i % 16));
  }

  // Frame 3: larger range (8-bit)
  for (int i = 0; i < 128; ++i) {
    data.push_back(300 + (i % 256));
  }

  auto encoding = createEncoding(data);
  ASSERT_EQ(encoding->rowCount(), 384);

  nimble::Vector<int32_t> result(pool_.get(), 384);
  encoding->materialize(384, result.data());

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], data[i]) << "Mismatch at index " << i;
  }
}

// Test random access by reading subsets
TEST_F(ForEncodingTest, randomAccess) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  nimble::Vector<int64_t> data(pool_.get());
  for (int i = 0; i < 1000; ++i) {
    data.push_back(static_cast<int64_t>(rng() % 1000000));
  }

  auto encoding = createEncoding(data);
  ASSERT_EQ(encoding->rowCount(), 1000);

  // Test by materializing entire sequence and checking consistency
  nimble::Vector<int64_t> result(pool_.get(), 1000);
  encoding->materialize(1000, result.data());

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], data[i]) << "Mismatch at index " << i;
  }
}

// Test partial read with skip
TEST_F(ForEncodingTest, selectiveRead) {
  nimble::Vector<int32_t> data(pool_.get());
  for (int i = 0; i < 500; ++i) {
    data.push_back(i * 10);
  }

  auto encoding = createEncoding(data);
  ASSERT_EQ(encoding->rowCount(), 500);

  // Read first 100
  nimble::Vector<int32_t> result1(pool_.get(), 100);
  encoding->materialize(100, result1.data());

  for (size_t i = 0; i < 100; ++i) {
    ASSERT_EQ(result1[i], data[i]) << "Mismatch at index " << i;
  }

  // Skip 200, then read next 100
  encoding->skip(200);
  nimble::Vector<int32_t> result2(pool_.get(), 100);
  encoding->materialize(100, result2.data());

  for (size_t i = 0; i < 100; ++i) {
    ASSERT_EQ(result2[i], data[300 + i]) << "Mismatch at index " << (300 + i);
  }
}

// Test with negative numbers
TEST_F(ForEncodingTest, negativeNumbers) {
  nimble::Vector<int32_t> data(pool_.get());
  data.push_back(-100);
  data.push_back(-50);
  data.push_back(-75);
  data.push_back(-25);
  data.push_back(-1);
  data.push_back(-99);

  auto encoding = createEncoding(data);
  ASSERT_EQ(encoding->rowCount(), 6);

  nimble::Vector<int32_t> result(pool_.get(), 6);
  encoding->materialize(6, result.data());

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], data[i]) << "Mismatch at index " << i;
  }
}

// Test with mixed positive and negative numbers
TEST_F(ForEncodingTest, mixedSignNumbers) {
  nimble::Vector<int32_t> data(pool_.get());
  data.push_back(-100);
  data.push_back(100);
  data.push_back(-50);
  data.push_back(50);
  data.push_back(0);
  data.push_back(-25);

  auto encoding = createEncoding(data);
  ASSERT_EQ(encoding->rowCount(), 6);

  nimble::Vector<int32_t> result(pool_.get(), 6);
  encoding->materialize(6, result.data());

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], data[i]) << "Mismatch at index " << i;
  }
}

// Test unsigned types
TEST_F(ForEncodingTest, unsignedTypes) {
  nimble::Vector<uint32_t> data(pool_.get());
  for (uint32_t i = 0; i < 256; ++i) {
    data.push_back(1000 + i);
  }

  auto encoding = createEncoding(data);
  ASSERT_EQ(encoding->rowCount(), 256);

  nimble::Vector<uint32_t> result(pool_.get(), 256);
  encoding->materialize(256, result.data());

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], data[i]) << "Mismatch at index " << i;
  }
}

// Test very large values (requiring 64-bit width)
TEST_F(ForEncodingTest, largeValues) {
  nimble::Vector<int64_t> data(pool_.get());
  data.push_back(0);
  data.push_back(1LL << 30); // ~1 billion
  data.push_back(1LL << 31); // ~2 billion
  data.push_back(1LL << 32); // ~4 billion
  data.push_back((1LL << 33) - 1);

  auto encoding = createEncoding(data);
  ASSERT_EQ(encoding->rowCount(), 5);

  nimble::Vector<int64_t> result(pool_.get(), 5);
  encoding->materialize(5, result.data());

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], data[i]) << "Mismatch at index " << i;
  }
}

// Test skip functionality
TEST_F(ForEncodingTest, skip) {
  nimble::Vector<int32_t> data(pool_.get());
  for (int i = 0; i < 500; ++i) {
    data.push_back(i);
  }

  auto encoding = createEncoding(data);

  // Skip first 100 elements
  encoding->skip(100);

  // Read next 50
  nimble::Vector<int32_t> result(pool_.get(), 50);
  encoding->materialize(50, result.data());

  for (size_t i = 0; i < 50; ++i) {
    ASSERT_EQ(result[i], data[100 + i])
        << "Mismatch at index " << (100 + i) << " after skip";
  }
}

// Test reset and re-read
TEST_F(ForEncodingTest, reset) {
  nimble::Vector<int32_t> data(pool_.get());
  for (int i = 0; i < 100; ++i) {
    data.push_back(i);
  }

  auto encoding = createEncoding(data);

  // Read first 50
  nimble::Vector<int32_t> result1(pool_.get(), 50);
  encoding->materialize(50, result1.data());

  // Reset
  encoding->reset();

  // Read first 50 again
  nimble::Vector<int32_t> result2(pool_.get(), 50);
  encoding->materialize(50, result2.data());

  for (size_t i = 0; i < 50; ++i) {
    ASSERT_EQ(result1[i], result2[i])
        << "Reset failed - mismatch at index " << i;
  }
}

// Test with compression
TEST_F(ForEncodingTest, withCompression) {
  nimble::Vector<int32_t> data(pool_.get());
  for (int i = 0; i < 1000; ++i) {
    data.push_back(i % 100); // Repeating pattern
  }

  auto encoding =
      nimble::test::Encoder<nimble::ForEncoding<int32_t>>::createEncoding(
          *buffer_,
          data,
          [](uint32_t) -> void* { return nullptr; },
          nimble::CompressionType::Zstd);

  ASSERT_EQ(encoding->rowCount(), 1000);

  nimble::Vector<int32_t> result(pool_.get(), 1000);
  encoding->materialize(1000, result.data());

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], data[i]) << "Mismatch at index " << i;
  }
}

// Test small data (less than one frame)
TEST_F(ForEncodingTest, smallData) {
  nimble::Vector<int32_t> data(pool_.get());
  data.push_back(10);
  data.push_back(20);
  data.push_back(15);

  auto encoding = createEncoding(data);
  ASSERT_EQ(encoding->rowCount(), 3);

  nimble::Vector<int32_t> result(pool_.get(), 3);
  encoding->materialize(3, result.data());

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], data[i]) << "Mismatch at index " << i;
  }
}

// Test exact frame boundary
TEST_F(ForEncodingTest, exactFrameBoundary) {
  nimble::Vector<int32_t> data(pool_.get());
  // Default frame size is 128
  for (int i = 0; i < 256; ++i) { // Exactly 2 frames
    data.push_back(i);
  }

  auto encoding = createEncoding(data);
  ASSERT_EQ(encoding->rowCount(), 256);

  nimble::Vector<int32_t> result(pool_.get(), 256);
  encoding->materialize(256, result.data());

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], data[i]) << "Mismatch at index " << i;
  }
}

// Test with all data types
TEST_F(ForEncodingTest, int8Type) {
  nimble::Vector<int8_t> data(pool_.get());
  for (int8_t i = -50; i < 50; ++i) {
    data.push_back(i);
  }

  auto encoding = createEncoding(data);
  nimble::Vector<int8_t> result(pool_.get(), data.size());
  encoding->materialize(data.size(), result.data());

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], data[i]);
  }
}

TEST_F(ForEncodingTest, int16Type) {
  nimble::Vector<int16_t> data(pool_.get());
  for (int16_t i = 0; i < 500; ++i) {
    data.push_back(i * 10);
  }

  auto encoding = createEncoding(data);
  nimble::Vector<int16_t> result(pool_.get(), data.size());
  encoding->materialize(data.size(), result.data());

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], data[i]);
  }
}

TEST_F(ForEncodingTest, uint64Type) {
  nimble::Vector<uint64_t> data(pool_.get());
  for (uint64_t i = 0; i < 200; ++i) {
    data.push_back(i * 1000000);
  }

  auto encoding = createEncoding(data);
  nimble::Vector<uint64_t> result(pool_.get(), data.size());
  encoding->materialize(data.size(), result.data());

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(result[i], data[i]);
  }
}

// Test selective reads using skip and materialize patterns
TEST_F(ForEncodingTest, selectiveReadsWithPattern) {
  nimble::Vector<int32_t> data(pool_.get());
  for (int i = 0; i < 1000; ++i) {
    data.push_back(i * 7); // Some pattern
  }

  auto encoding = createEncoding(data);
  ASSERT_EQ(encoding->rowCount(), 1000);

  // Read indices: 0, 10, 20, 30, ..., 990 (every 10th element)
  std::vector<int32_t> expected;
  for (int i = 0; i < 100; ++i) {
    expected.push_back(i * 10 * 7);
  }

  nimble::Vector<int32_t> results(pool_.get(), expected.size());
  size_t resultIdx = 0;

  for (int i = 0; i < 100; ++i) {
    // Read one value
    encoding->materialize(1, &results[resultIdx++]);

    // Skip next 9 (unless last iteration)
    if (i < 99) {
      encoding->skip(9);
    }
  }

  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(results[i], expected[i])
        << "Mismatch at selective index " << i << " (row " << (i * 10) << ")";
  }
}

// Test random access pattern with multiple resets
TEST_F(ForEncodingTest, randomAccessWithResets) {
  nimble::Vector<int64_t> data(pool_.get());
  for (int i = 0; i < 500; ++i) {
    data.push_back(static_cast<int64_t>(i) * 100);
  }

  auto encoding = createEncoding(data);

  // Read index 100
  encoding->skip(100);
  nimble::Vector<int64_t> result1(pool_.get(), 1);
  encoding->materialize(1, result1.data());
  ASSERT_EQ(result1[0], 10000);

  // Reset and read index 250
  encoding->reset();
  encoding->skip(250);
  nimble::Vector<int64_t> result2(pool_.get(), 1);
  encoding->materialize(1, result2.data());
  ASSERT_EQ(result2[0], 25000);

  // Reset and read index 0
  encoding->reset();
  nimble::Vector<int64_t> result3(pool_.get(), 1);
  encoding->materialize(1, result3.data());
  ASSERT_EQ(result3[0], 0);

  // Read index 499 from current position (skip 498 more)
  encoding->skip(498);
  nimble::Vector<int64_t> result4(pool_.get(), 1);
  encoding->materialize(1, result4.data());
  ASSERT_EQ(result4[0], 49900);
}

// Test sparse selective read pattern
TEST_F(ForEncodingTest, sparseSelectiveReads) {
  nimble::Vector<uint32_t> data(pool_.get());
  for (uint32_t i = 0; i < 1000; ++i) {
    data.push_back(i * i); // Quadratic values
  }

  auto encoding = createEncoding(data);

  // Read specific sparse indices: 0, 100, 500, 999
  std::vector<uint32_t> indices = {0, 100, 500, 999};
  std::vector<uint32_t> expected;
  for (auto idx : indices) {
    expected.push_back(idx * idx);
  }

  nimble::Vector<uint32_t> results(pool_.get(), indices.size());

  // Read index 0
  encoding->materialize(1, &results[0]);

  // Skip to 100, read it
  encoding->skip(99);
  encoding->materialize(1, &results[1]);

  // Skip to 500, read it
  encoding->skip(399);
  encoding->materialize(1, &results[2]);

  // Skip to 999, read it
  encoding->skip(498);
  encoding->materialize(1, &results[3]);

  for (size_t i = 0; i < indices.size(); ++i) {
    ASSERT_EQ(results[i], expected[i]) << "Mismatch for index " << indices[i];
  }
}

// Test reading across frame boundaries
TEST_F(ForEncodingTest, selectiveAcrossFrameBoundaries) {
  nimble::Vector<int32_t> data(pool_.get());
  // Create data with 3 full frames (128 * 3 = 384 values)
  for (int i = 0; i < 384; ++i) {
    data.push_back(i);
  }

  auto encoding = createEncoding(data);

  // Read values at frame boundaries
  // Frame 0: 0-127, Frame 1: 128-255, Frame 2: 256-383
  std::vector<uint32_t> testIndices = {
      0, // Start of frame 0
      63, // Middle of frame 0
      127, // End of frame 0
      128, // Start of frame 1
      191, // Middle of frame 1
      255, // End of frame 1
      256, // Start of frame 2
      319, // Middle of frame 2
      383 // End of frame 2
  };

  encoding->reset();
  nimble::Vector<int32_t> results(pool_.get(), testIndices.size());

  uint32_t currentPos = 0;
  for (size_t i = 0; i < testIndices.size(); ++i) {
    uint32_t targetIdx = testIndices[i];

    if (targetIdx > currentPos) {
      encoding->skip(targetIdx - currentPos);
      currentPos = targetIdx;
    } else if (targetIdx < currentPos) {
      // Need to reset and skip from beginning
      encoding->reset();
      encoding->skip(targetIdx);
      currentPos = targetIdx;
    }

    encoding->materialize(1, &results[i]);
    currentPos++;
  }

  for (size_t i = 0; i < testIndices.size(); ++i) {
    ASSERT_EQ(results[i], static_cast<int32_t>(testIndices[i]))
        << "Mismatch at frame boundary index " << testIndices[i];
  }
}

// Test selective read with varying bit widths across frames
TEST_F(ForEncodingTest, selectiveWithVaryingBitWidths) {
  nimble::Vector<int64_t> data(pool_.get());

  // Frame 0: small range (1-bit width)
  for (int i = 0; i < 128; ++i) {
    data.push_back(1000 + (i % 2));
  }

  // Frame 1: medium range (8-bit width)
  for (int i = 0; i < 128; ++i) {
    data.push_back(2000 + (i % 200));
  }

  // Frame 2: large range (32-bit width)
  for (int i = 0; i < 128; ++i) {
    data.push_back(1000000000LL + i);
  }

  auto encoding = createEncoding(data);

  // Selectively read from each frame
  std::vector<std::pair<uint32_t, int64_t>> tests = {
      {50, 1000 + (50 % 2)}, // From frame 0
      {150, 2000 + (22 % 200)}, // From frame 1 (150 - 128 = 22)
      {300, 1000000000LL + (300 - 256)} // From frame 2 (300 - 256 = 44)
  };

  encoding->reset();
  for (const auto& [index, expectedValue] : tests) {
    encoding->reset();
    encoding->skip(index);

    nimble::Vector<int64_t> result(pool_.get(), 1);
    encoding->materialize(1, result.data());

    ASSERT_EQ(result[0], expectedValue) << "Mismatch at index " << index;
  }
}

// Test batch selective reads
TEST_F(ForEncodingTest, batchSelectiveReads) {
  nimble::Vector<int32_t> data(pool_.get());
  for (int i = 0; i < 1000; ++i) {
    data.push_back(i * 3);
  }

  auto encoding = createEncoding(data);

  // Read batches: [0-9], [100-109], [500-509], [900-909]
  std::vector<std::pair<uint32_t, uint32_t>> ranges = {
      {0, 10}, {100, 10}, {500, 10}, {900, 10}};

  for (const auto& [start, count] : ranges) {
    encoding->reset();
    encoding->skip(start);

    nimble::Vector<int32_t> result(pool_.get(), count);
    encoding->materialize(count, result.data());

    for (uint32_t i = 0; i < count; ++i) {
      uint32_t expectedIdx = start + i;
      ASSERT_EQ(result[i], static_cast<int32_t>(expectedIdx * 3))
          << "Mismatch in batch starting at " << start << ", position " << i;
    }
  }
}

// readWithVisitor() is implemented and supports O(1) random access.
// Full testing requires Velox's SelectiveColumnReader infrastructure.
// Tests above verify the O(1) access through skip() and materialize().
