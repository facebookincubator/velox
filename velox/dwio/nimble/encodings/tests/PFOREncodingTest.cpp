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
#include "velox/dwio/nimble/encodings/PFOREncoding.h"

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>

#include "velox/buffer/BufferPool.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace ::facebook;

namespace facebook::nimble::test {

struct PforWireInfo {
  uint8_t baseBitWidth;
  uint32_t numExceptions;
  size_t basePayloadSize;
};

PforWireInfo readPforWireInfo(
    std::string_view encoded,
    bool useVarintRowCount) {
  const auto prefixSize =
      EncodingPrefix::prefixSize(encoded, useVarintRowCount);
  const char* pos = encoded.data() + prefixSize;
  pos += sizeof(uint32_t);
  const auto baseBitWidth = static_cast<uint8_t>(encoding::readChar(pos));
  const auto numExceptions = varint::readVarint32(&pos);
  const auto exceptionPositionsSize = varint::readVarint32(&pos);
  pos += exceptionPositionsSize;
  const auto exceptionValuesSize = varint::readVarint32(&pos);
  pos += exceptionValuesSize;
  return {
      .baseBitWidth = baseBitWidth,
      .numExceptions = numExceptions,
      .basePayloadSize =
          static_cast<size_t>(encoded.data() + encoded.size() - pos),
  };
}

// The Pfor wire format is byte-identical for a signed cpp type and its
// matching unsigned physical type, so the per-type tests run on the unsigned
// physical types directly. Signed-input coverage is exercised end-to-end by
// the fuzzer test below, which routes int32/int64 through the standard
// EncodingFactory::encode<T> path.
template <typename T>
class PFOREncodingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  }

  std::unique_ptr<EncodingSelectionPolicy<T>> createSelectionPolicy() {
    EncodingLayout layout{
        EncodingType::PFOR,
        {},
        CompressionType::Uncompressed,
        {std::nullopt, std::nullopt}};
    return std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
        std::move(layout),
        CompressionOptions{},
        encodingSelectionPolicyCreator_);
  }

  std::unique_ptr<Encoding> encodeAndCreate(const std::vector<T>& values) {
    Buffer buffer{*pool_};
    auto encoded = EncodingFactory::encode<T>(
        createSelectionPolicy(),
        std::span<const T>{values.data(), values.size()},
        buffer);
    encodedStorage_.assign(encoded.begin(), encoded.end());
    return EncodingFactory().create(
        *pool_, {encodedStorage_.data(), encodedStorage_.size()}, nullptr);
  }

  void roundTripAndExpect(const std::vector<T>& values) {
    auto encoding = encodeAndCreate(values);
    EXPECT_EQ(encoding->encodingType(), EncodingType::PFOR);
    EXPECT_EQ(encoding->dataType(), TypeTraits<T>::dataType);
    EXPECT_EQ(encoding->rowCount(), values.size());

    std::vector<T> decoded(values.size());
    encoding->materialize(static_cast<uint32_t>(values.size()), decoded.data());
    for (size_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(decoded[i], values[i]) << "mismatch at index " << i;
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::vector<char> encodedStorage_;
  ManualEncodingSelectionPolicyFactory manualPolicyFactory_;
  EncodingSelectionPolicyCreator encodingSelectionPolicyCreator_ =
      [this](DataType dataType) {
        return manualPolicyFactory_.createPolicy(dataType);
      };
};

using PFORTypes = ::testing::Types<uint32_t, uint64_t, uint8_t, uint16_t>;
TYPED_TEST_SUITE(PFOREncodingTest, PFORTypes);

TYPED_TEST(PFOREncodingTest, singleElement) {
  using T = TypeParam;
  this->roundTripAndExpect({T{42}});
}

TYPED_TEST(PFOREncodingTest, allSameValues) {
  using T = TypeParam;
  std::vector<T> values(64, T{7});
  this->roundTripAndExpect(values);
}

TYPED_TEST(PFOREncodingTest, denseSmallValuesNoExceptions) {
  // Residuals all fit comfortably in the smallest 7-bit bucket; the encoder
  // should pick baseBitWidth=7 and emit zero exceptions.
  using T = TypeParam;
  std::vector<T> values;
  values.reserve(200);
  for (uint32_t i = 0; i < 200; ++i) {
    values.push_back(static_cast<T>(100 + (i % 64)));
  }
  this->roundTripAndExpect(values);
}

TYPED_TEST(PFOREncodingTest, sparseOutliers) {
  // 90% small residuals + 10% large outliers — Pfor's sweet spot. Skip on
  // narrow types where outliers would not be representable.
  using T = TypeParam;
  if constexpr (sizeof(T) < 4) {
    return;
  } else {
    std::vector<T> values;
    values.reserve(500);
    for (uint32_t i = 0; i < 500; ++i) {
      if (i % 10 == 7) {
        values.push_back(static_cast<T>(100000 + i));
      } else {
        values.push_back(static_cast<T>(50 + (i % 50)));
      }
    }
    this->roundTripAndExpect(values);
  }
}

TEST(PFOREncodingBufferPoolTest, reusesExceptionBuffers) {
  using Value = uint64_t;
  using EncodingType = PFOREncoding<Value>;

  auto encodePool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Buffer encodeBuffer{*encodePool};
  std::vector<Value> input;
  input.reserve(1'000);
  for (uint32_t i = 0; i < 1'000; ++i) {
    input.push_back(
        i % 10 == 0 ? std::numeric_limits<Value>::max() - i
                    : static_cast<Value>(i % 64));
  }

  Vector<Value> values{encodePool.get()};
  values.insert(values.end(), input.data(), input.data() + input.size());
  const auto encoded = Encoder<EncodingType>::encode(encodeBuffer, values);
  const std::vector<char> encodedStorage{encoded.begin(), encoded.end()};

  auto decodePool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};
  const Encoding::Options options{.bufferPool = &bufferPool};
  std::function<void*(uint32_t)> stringBufferFactory;

  auto decodeAndVerify = [&]() {
    EncodingType encoding{
        *decodePool,
        {encodedStorage.data(), encodedStorage.size()},
        stringBufferFactory,
        options};
    std::vector<Value> output(input.size());
    encoding.materialize(static_cast<uint32_t>(input.size()), output.data());
    EXPECT_EQ(input, output);
  };

  decodeAndVerify();
  const auto numAllocsAfterFirstDecode = decodePool->stats().numAllocs;
  EXPECT_GT(bufferPool.size(), 0);

  decodeAndVerify();
  EXPECT_EQ(decodePool->stats().numAllocs, numAllocsAfterFirstDecode);
}

TEST(PFOREncodingBufferPoolTest, sliceReturnsScratchBufferToPool) {
  using Value = uint64_t;
  using EncodingType = PFOREncoding<Value>;

  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  Buffer encodeBuffer{*pool};
  std::vector<Value> input;
  input.reserve(300);
  for (uint32_t i = 0; i < 300; ++i) {
    input.push_back(
        i % 10 == 7 ? static_cast<Value>(100000 + i)
                    : static_cast<Value>(50 + (i % 64)));
  }

  Vector<Value> values{pool.get()};
  values.insert(values.end(), input.data(), input.data() + input.size());
  const auto encoded = Encoder<EncodingType>::encode(encodeBuffer, values);
  const std::string encodedStorage{encoded};

  velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};
  const Encoding::Options options{.bufferPool = &bufferPool};
  Buffer sliceBuffer{*pool};
  const auto sliced = EncodingType::slice(
      encodedStorage,
      /*offset=*/17,
      /*length=*/128,
      sliceBuffer,
      options);

  EXPECT_GT(bufferPool.size(), 0);

  EncodingType encoding{*pool, sliced, nullptr, options};
  std::vector<Value> output(128);
  encoding.materialize(static_cast<uint32_t>(output.size()), output.data());
  EXPECT_EQ(
      output, std::vector<Value>(input.begin() + 17, input.begin() + 17 + 128));
}

struct PforBitWidthStorageParam {
  bool fixedBitWidthUseExactBits;
  uint8_t expectedBaseBitWidth;
};

class PforBitWidthStorageTest
    : public ::testing::TestWithParam<PforBitWidthStorageParam> {};

TEST_P(PforBitWidthStorageTest, storesExpectedBasePayloadWidth) {
  constexpr uint32_t kRowCount{130};
  constexpr uint32_t kNumExceptions{13};
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  std::vector<uint32_t> values;
  values.reserve(kRowCount);
  for (uint32_t i = 0; i < kRowCount; ++i) {
    values.push_back(i < kRowCount - kNumExceptions ? i % 64 : 512 + i);
  }

  ManualEncodingSelectionPolicyFactory manualFactory;
  EncodingSelectionPolicyCreator creator = [&manualFactory](DataType dataType) {
    return manualFactory.createPolicy(dataType);
  };
  EncodingLayout layout{
      EncodingType::PFOR,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::nullopt}};
  auto policy = std::make_unique<ReplayedEncodingSelectionPolicy<uint32_t>>(
      std::move(layout), CompressionOptions{}, creator);

  Encoding::Options options;
  options.fixedBitWidthUseExactBits = GetParam().fixedBitWidthUseExactBits;
  Buffer buffer{*pool};
  const auto encoded = EncodingFactory::encode<uint32_t>(
      std::move(policy),
      std::span<const uint32_t>{values.data(), values.size()},
      buffer,
      options);
  const std::string encodedStorage{encoded};

  const auto wireInfo =
      readPforWireInfo(encodedStorage, options.useVarintRowCount);
  EXPECT_EQ(wireInfo.baseBitWidth, GetParam().expectedBaseBitWidth);
  EXPECT_EQ(wireInfo.numExceptions, kNumExceptions);
  EXPECT_EQ(
      wireInfo.basePayloadSize,
      FixedBitArray::bufferSize(
          values.size(), GetParam().expectedBaseBitWidth));

  auto encoding =
      EncodingFactory(options).create(*pool, encodedStorage, nullptr);
  std::vector<uint32_t> decoded(values.size());
  encoding->materialize(static_cast<uint32_t>(decoded.size()), decoded.data());
  EXPECT_EQ(decoded, values);
}

INSTANTIATE_TEST_SUITE_P(
    PforBitWidthStorageModes,
    PforBitWidthStorageTest,
    ::testing::Values(
        PforBitWidthStorageParam{
            .fixedBitWidthUseExactBits = false,
            .expectedBaseBitWidth = 7,
        },
        PforBitWidthStorageParam{
            .fixedBitWidthUseExactBits = true,
            .expectedBaseBitWidth = 6,
        }),
    [](const ::testing::TestParamInfo<PforBitWidthStorageParam>& info) {
      return info.param.fixedBitWidthUseExactBits ? "Exact" : "Rounded";
    });

TYPED_TEST(PFOREncodingTest, residualsAtBitWidthBoundary) {
  // Residuals exactly at the (1<<7)-1 = 127 boundary — verifies the
  // base-mask comparison treats the boundary residual as a fitting value.
  using T = TypeParam;
  std::vector<T> values;
  values.reserve(50);
  for (uint32_t i = 0; i < 50; ++i) {
    values.push_back(static_cast<T>(i * 2 % 128));
  }
  this->roundTripAndExpect(values);
}

TYPED_TEST(PFOREncodingTest, mixedSizes) {
  using T = TypeParam;
  for (uint32_t n : {1u, 7u, 15u, 16u, 17u, 33u, 64u, 100u, 257u, 1024u}) {
    SCOPED_TRACE(fmt::format("n={}", n));
    std::vector<T> values;
    values.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      values.push_back(static_cast<T>((i * 31 + 1) % 256));
    }
    this->roundTripAndExpect(values);
  }
}

TYPED_TEST(PFOREncodingTest, skipAndMaterialize) {
  using T = TypeParam;
  std::vector<T> values;
  values.reserve(300);
  for (uint32_t i = 0; i < 300; ++i) {
    if constexpr (sizeof(T) >= 4) {
      values.push_back(
          static_cast<T>(i % 17 == 0 ? 100000 + i : 50 + (i % 30)));
    } else {
      values.push_back(static_cast<T>(50 + (i % 30)));
    }
  }
  auto encoding = this->encodeAndCreate(values);

  // Materialize a few prefixes interleaved with skips — hits the exception
  // cursor and the bitpacked cursor in the same pass.
  encoding->reset();
  std::vector<T> chunk(50);
  encoding->materialize(50, chunk.data());
  for (uint32_t i = 0; i < 50; ++i) {
    EXPECT_EQ(chunk[i], values[i]);
  }
  encoding->skip(75);
  encoding->materialize(50, chunk.data());
  for (uint32_t i = 0; i < 50; ++i) {
    EXPECT_EQ(chunk[i], values[125 + i]) << "i=" << i;
  }
  // 50 read + 75 skip + 50 read + 75 skip + 50 read == 300 == values.size()
  encoding->skip(75);
  encoding->materialize(50, chunk.data());
  for (uint32_t i = 0; i < 50; ++i) {
    EXPECT_EQ(chunk[i], values[250 + i]) << "i=" << i;
  }

  // Reset must restore all cursors so a second pass produces identical output.
  encoding->reset();
  std::vector<T> all(values.size());
  encoding->materialize(static_cast<uint32_t>(values.size()), all.data());
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(all[i], values[i]) << "after-reset i=" << i;
  }
}

TYPED_TEST(PFOREncodingTest, slice) {
  using T = TypeParam;

  struct Range {
    uint32_t offset;
    uint32_t length;
  };

  std::vector<T> values;
  values.reserve(300);
  for (uint32_t i = 0; i < 300; ++i) {
    if constexpr (sizeof(T) >= 4) {
      values.push_back(
          static_cast<T>(i % 10 == 7 ? 100000 + i : 50 + (i % 64)));
    } else {
      values.push_back(static_cast<T>(50 + (i % 64)));
    }
  }
  const auto valueCount = static_cast<uint32_t>(values.size());

  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
    const Encoding::Options options{.useVarintRowCount = useVarint};
    Buffer buffer{*this->pool_};
    const auto encoded = EncodingFactory::encode<T>(
        this->createSelectionPolicy(),
        std::span<const T>{values.data(), values.size()},
        buffer,
        options);

    for (const auto range :
         {Range{/*offset=*/0, /*length=*/1},
          Range{/*offset=*/1, /*length=*/37},
          Range{/*offset=*/127, /*length=*/64},
          Range{/*offset=*/valueCount - 43, /*length=*/43}}) {
      SCOPED_TRACE(
          fmt::format("offset={}, length={}", range.offset, range.length));
      Buffer sliceBuffer{*this->pool_};
      const auto sliced = PFOREncoding<T>::slice(
          encoded, range.offset, range.length, sliceBuffer, options);

      PFOREncoding<T> encoding{*this->pool_, sliced, nullptr, options};
      EXPECT_EQ(encoding.encodingType(), EncodingType::PFOR);
      EXPECT_EQ(encoding.dataType(), TypeTraits<T>::dataType);
      EXPECT_EQ(encoding.rowCount(), range.length);

      std::vector<T> output(range.length);
      encoding.materialize(range.length, output.data());
      EXPECT_EQ(
          output,
          std::vector<T>(
              values.begin() + range.offset,
              values.begin() + range.offset + range.length));
    }
  }
}

TYPED_TEST(PFOREncodingTest, invalidSliceRange) {
  using T = TypeParam;

  std::vector<T> values;
  values.reserve(16);
  for (uint32_t i = 0; i < 16; ++i) {
    values.push_back(static_cast<T>(i + 1));
  }

  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
    const Encoding::Options options{.useVarintRowCount = useVarint};
    Buffer buffer{*this->pool_};
    const auto encoded = EncodingFactory::encode<T>(
        this->createSelectionPolicy(),
        std::span<const T>{values.data(), values.size()},
        buffer,
        options);

    Buffer sliceBuffer{*this->pool_};
    NIMBLE_ASSERT_THROW(
        PFOREncoding<T>::slice(
            encoded,
            /*offset=*/0,
            /*length=*/0,
            sliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        PFOREncoding<T>::slice(
            encoded,
            /*offset=*/values.size() + 1,
            /*length=*/1,
            sliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        PFOREncoding<T>::slice(
            encoded,
            /*offset=*/values.size() - 1,
            /*length=*/2,
            sliceBuffer,
            options),
        "");
  }
}

TYPED_TEST(PFOREncodingTest, exceptionAtFirstAndLastRow) {
  // Edge case: exceptions land on the first row, the last row, and a
  // middle row. Verifies cursor handling at the bounds.
  using T = TypeParam;
  if constexpr (sizeof(T) < 4) {
    return;
  } else {
    std::vector<T> values;
    values.reserve(100);
    for (uint32_t i = 0; i < 100; ++i) {
      values.push_back(static_cast<T>(10));
    }
    values[0] = static_cast<T>(50000);
    values[50] = static_cast<T>(60000);
    values[99] = static_cast<T>(70000);
    this->roundTripAndExpect(values);
  }
}

TYPED_TEST(PFOREncodingTest, debugString) {
  using T = TypeParam;
  std::vector<T> values;
  values.reserve(8);
  for (uint32_t i = 0; i < 8; ++i) {
    values.push_back(static_cast<T>(i + 1));
  }
  auto encoding = this->encodeAndCreate(values);
  const std::string debug = encoding->debugString();
  EXPECT_NE(debug.find("PFOR"), std::string::npos);
  EXPECT_NE(debug.find("baseBitWidth="), std::string::npos);
  EXPECT_NE(debug.find("numExceptions="), std::string::npos);
}

class PFOREncodingFuzzerTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  }

  // Run the fuzzer against the cpp type T (signed or unsigned) by going
  // through EncodingFactory::encode<T>. Verifies the signed-input wire path,
  // materialization, and skip/materialize interleaving over many seeds.
  template <typename T>
  void runFuzzer(uint32_t seed, uint32_t numIterations) {
    std::mt19937 rng(seed);

    ManualEncodingSelectionPolicyFactory manualFactory;
    EncodingSelectionPolicyCreator creator =
        [&manualFactory](DataType dataType) {
          return manualFactory.createPolicy(dataType);
        };

    for (uint32_t iter = 0; iter < numIterations; ++iter) {
      SCOPED_TRACE(fmt::format("iteration {}", iter));

      std::uniform_int_distribution<uint32_t> sizeDist(1, 1000);
      const uint32_t numValues = sizeDist(rng);

      std::uniform_int_distribution<int32_t> baseDist(0, 64);
      std::uniform_int_distribution<int32_t> outlierDist(0, 1 << 20);
      std::uniform_real_distribution<double> outlierProb(0.0, 1.0);
      std::vector<T> values;
      values.reserve(numValues);
      const T offset = static_cast<T>(1000);
      for (uint32_t i = 0; i < numValues; ++i) {
        const bool isOutlier = outlierProb(rng) < 0.05;
        values.push_back(
            static_cast<T>(
                offset + (isOutlier ? outlierDist(rng) : baseDist(rng))));
      }

      Buffer buffer{*pool_};
      EncodingLayout layout{
          EncodingType::PFOR,
          {},
          CompressionType::Uncompressed,
          {std::nullopt, std::nullopt}};
      auto policy = std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
          std::move(layout), CompressionOptions{}, creator);
      auto encoded = EncodingFactory::encode<T>(
          std::move(policy),
          std::span<const T>{values.data(), values.size()},
          buffer);
      std::vector<char> storage(encoded.begin(), encoded.end());
      auto encoding = EncodingFactory().create(
          *pool_, {storage.data(), storage.size()}, nullptr);

      ASSERT_EQ(encoding->encodingType(), EncodingType::PFOR);
      ASSERT_EQ(encoding->dataType(), TypeTraits<T>::dataType);
      ASSERT_EQ(encoding->rowCount(), values.size());
      std::vector<T> decoded(values.size());
      encoding->materialize(
          static_cast<uint32_t>(values.size()), decoded.data());
      for (size_t i = 0; i < values.size(); ++i) {
        ASSERT_EQ(decoded[i], values[i])
            << "iter=" << iter << " i=" << i << " size=" << numValues;
      }

      std::uniform_int_distribution<uint32_t> offsetDist(0, numValues - 1);
      const uint32_t sliceOffset = offsetDist(rng);
      std::uniform_int_distribution<uint32_t> lengthDist(
          1, numValues - sliceOffset);
      const uint32_t sliceLength = lengthDist(rng);
      Buffer sliceBuffer{*pool_};
      const auto sliced = PFOREncoding<T>::slice(
          {storage.data(), storage.size()},
          sliceOffset,
          sliceLength,
          sliceBuffer);
      PFOREncoding<T> slicedEncoding{*pool_, sliced, nullptr};
      std::vector<T> slicedValues(sliceLength);
      slicedEncoding.materialize(sliceLength, slicedValues.data());
      for (uint32_t i = 0; i < sliceLength; ++i) {
        ASSERT_EQ(slicedValues[i], values[sliceOffset + i])
            << "iter=" << iter << " sliceOffset=" << sliceOffset << " i=" << i
            << " sliceLength=" << sliceLength;
      }

      // Random skip/materialize sequence.
      encoding->reset();
      std::uniform_int_distribution<uint32_t> stepDist(0, numValues);
      uint32_t cursor = 0;
      while (cursor < numValues) {
        const uint32_t remaining = numValues - cursor;
        const uint32_t skipCount = stepDist(rng) % (remaining + 1);
        if (skipCount > 0) {
          encoding->skip(skipCount);
          cursor += skipCount;
        }
        if (cursor >= numValues) {
          break;
        }
        const uint32_t matCount =
            std::max<uint32_t>(1, stepDist(rng) % (numValues - cursor + 1));
        std::vector<T> chunk(matCount);
        encoding->materialize(matCount, chunk.data());
        for (uint32_t i = 0; i < matCount; ++i) {
          ASSERT_EQ(chunk[i], values[cursor + i])
              << "iter=" << iter << " cursor=" << cursor << " i=" << i;
        }
        cursor += matCount;
      }
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(PFOREncodingFuzzerTest, fuzzerInt32) {
  runFuzzer<int32_t>(/*seed=*/12345, /*numIterations=*/30);
}

TEST_F(PFOREncodingFuzzerTest, fuzzerInt64) {
  runFuzzer<int64_t>(/*seed=*/67890, /*numIterations=*/30);
}

TEST_F(PFOREncodingFuzzerTest, fuzzerUint32) {
  runFuzzer<uint32_t>(/*seed=*/24680, /*numIterations=*/30);
}

TEST_F(PFOREncodingFuzzerTest, fuzzerUint64) {
  runFuzzer<uint64_t>(/*seed=*/13579, /*numIterations=*/30);
}

TEST_F(PFOREncodingFuzzerTest, encodeRejectsEmpty) {
  Buffer buffer{*pool_};
  EncodingLayout layout{
      EncodingType::PFOR,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::nullopt}};
  ManualEncodingSelectionPolicyFactory manualFactory;
  EncodingSelectionPolicyCreator creator = [&manualFactory](DataType dataType) {
    return manualFactory.createPolicy(dataType);
  };
  auto policy = std::make_unique<ReplayedEncodingSelectionPolicy<uint32_t>>(
      std::move(layout), CompressionOptions{}, creator);
  std::vector<uint32_t> empty;
  NIMBLE_ASSERT_THROW(
      EncodingFactory::encode<uint32_t>(
          std::move(policy),
          std::span<const uint32_t>{empty.data(), empty.size()},
          buffer),
      "Pfor encoding cannot be used with 0 rows.");
}

} // namespace facebook::nimble::test
