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
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include <string>
#include <type_traits>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

class TrivialEncodingTest : public ::testing::Test {
 protected:
  struct Range {
    const char* name;
    uint32_t offset;
    uint32_t length;
  };

  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("TrivialEncodingTest");
    pool_ = rootPool_->addLeafChild("TrivialEncodingTestLeaf");
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  nimble::Vector<std::string_view> toVector(
      std::initializer_list<std::string_view> values) {
    nimble::Vector<std::string_view> result{pool_.get()};
    result.insert(result.end(), values.begin(), values.end());
    return result;
  }

  template <typename T>
  nimble::Vector<T> makeVector(std::initializer_list<T> values) {
    nimble::Vector<T> result{pool_.get()};
    result.insert(result.end(), values.begin(), values.end());
    return result;
  }

  nimble::CompressionType compressionType(
      std::string_view encoded,
      const nimble::Encoding::Options& options = {}) {
    const char* pos = encoded.data() +
        nimble::EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
    return static_cast<nimble::CompressionType>(
        nimble::encoding::readChar(pos));
  }

  std::unique_ptr<nimble::Encoding> createEncoding(
      const nimble::Vector<std::string_view>& values,
      const nimble::Encoding::Options& options = {}) {
    stringBuffers_.clear();
    return nimble::test::Encoder<nimble::TrivialEncoding<std::string_view>>::
        createEncoding(
            *buffer_,
            values,
            [&](uint32_t totalLength) {
              auto& buffer = stringBuffers_.emplace_back(
                  velox::AlignedBuffer::allocate<char>(
                      totalLength, pool_.get()));
              return buffer->asMutable<void>();
            },
            nimble::CompressionType::Uncompressed,
            options);
  }

  std::function<void*(uint32_t)> stringBufferFactory() {
    return [&](uint32_t totalLength) -> void* {
      auto& buffer = stringBuffers_.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
      return buffer->asMutable<void>();
    };
  }

  static std::vector<Range> ranges(uint32_t valueCount) {
    return {
        Range{/*name=*/"full", /*offset=*/0, /*length=*/valueCount},
        Range{/*name=*/"prefix", /*offset=*/0, /*length=*/128},
        Range{/*name=*/"middle", /*offset=*/2047, /*length=*/257},
        Range{/*name=*/"suffix",
              /*offset=*/valueCount - 128,
              /*length=*/128},
        Range{/*name=*/"singleStart", /*offset=*/0, /*length=*/1},
        Range{/*name=*/"singleMiddle", /*offset=*/2048, /*length=*/1},
        Range{/*name=*/"singleEnd",
              /*offset=*/valueCount - 1,
              /*length=*/1},
    };
  }

  template <typename T>
  nimble::Vector<T> makeRepeatedValues(uint32_t valueCount) {
    nimble::Vector<T> values{pool_.get(), valueCount};
    for (uint32_t i = 0; i < valueCount; ++i) {
      values[i] = static_cast<T>(10 + (i % 4));
    }
    return values;
  }

  nimble::Vector<bool> makeBoolValues(uint32_t valueCount) {
    nimble::Vector<bool> values{pool_.get(), valueCount};
    for (uint32_t i = 0; i < valueCount; ++i) {
      values[i] = (i % 4) == 0;
    }
    return values;
  }

  nimble::Vector<std::string_view> makeStringValues(
      std::vector<std::string>& storage,
      uint32_t valueCount) {
    storage.reserve(valueCount);
    nimble::Vector<std::string_view> values{pool_.get(), valueCount};
    for (uint32_t i = 0; i < valueCount; ++i) {
      storage.push_back(std::string{"value-"} + std::to_string(i % 4));
      values[i] = storage.back();
    }
    return values;
  }

  template <typename T>
  void expectSliceRanges(
      const nimble::Vector<T>& values,
      nimble::DataType dataType) {
    const auto valueCount = static_cast<uint32_t>(values.size());
    for (const auto sourceCompression :
         {nimble::CompressionType::Uncompressed,
          nimble::CompressionType::Zstd}) {
      SCOPED_TRACE(
          ::testing::Message()
          << "sourceCompression=" << static_cast<int>(sourceCompression));
      const auto encoded =
          nimble::test::Encoder<nimble::TrivialEncoding<T>>::encode(
              *buffer_, values, sourceCompression);
      EXPECT_EQ(compressionType(encoded), sourceCompression);

      for (const auto range : ranges(valueCount)) {
        SCOPED_TRACE(
            ::testing::Message()
            << "range=" << range.name << ", offset=" << range.offset
            << ", length=" << range.length);
        const auto sliced = nimble::TrivialEncoding<T>::slice(
            encoded,
            range.offset,
            range.length,
            *buffer_,
            nimble::Encoding::Options{});
        EXPECT_NE(sliced.data(), encoded.data());
        EXPECT_EQ(
            compressionType(sliced), nimble::CompressionType::Uncompressed);

        stringBuffers_.clear();
        auto encoding = nimble::EncodingFactory{}.create(
            *pool_, sliced, stringBufferFactory());
        EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Trivial);
        EXPECT_EQ(encoding->dataType(), dataType);
        EXPECT_EQ(encoding->rowCount(), range.length);

        nimble::Vector<T> output{pool_.get(), range.length};
        encoding->materialize(range.length, output.data());

        const std::vector<T> expected(
            values.begin() + range.offset,
            values.begin() + range.offset + range.length);
        EXPECT_EQ(std::vector<T>(output.begin(), output.end()), expected);
      }
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
  std::vector<velox::BufferPtr> stringBuffers_;
};

TEST_F(TrivialEncodingTest, sliceRangesForSupportedTypes) {
  constexpr uint32_t valueCount{4096};
  expectSliceRanges<int8_t>(
      makeRepeatedValues<int8_t>(valueCount), nimble::DataType::Int8);
  expectSliceRanges<uint8_t>(
      makeRepeatedValues<uint8_t>(valueCount), nimble::DataType::Uint8);
  expectSliceRanges<int16_t>(
      makeRepeatedValues<int16_t>(valueCount), nimble::DataType::Int16);
  expectSliceRanges<uint16_t>(
      makeRepeatedValues<uint16_t>(valueCount), nimble::DataType::Uint16);
  expectSliceRanges<int32_t>(
      makeRepeatedValues<int32_t>(valueCount), nimble::DataType::Int32);
  expectSliceRanges<uint32_t>(
      makeRepeatedValues<uint32_t>(valueCount), nimble::DataType::Uint32);
  expectSliceRanges<int64_t>(
      makeRepeatedValues<int64_t>(valueCount), nimble::DataType::Int64);
  expectSliceRanges<uint64_t>(
      makeRepeatedValues<uint64_t>(valueCount), nimble::DataType::Uint64);
  expectSliceRanges<bool>(makeBoolValues(valueCount), nimble::DataType::Bool);

  std::vector<std::string> stringStorage;
  expectSliceRanges<std::string_view>(
      makeStringValues(stringStorage, valueCount), nimble::DataType::String);
}

TEST_F(TrivialEncodingTest, invalidSliceRange) {
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useVarint=" << useVarint);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};

    const auto uint32Values = makeVector<uint32_t>({10, 20, 30});
    const auto uint32Encoded =
        nimble::test::Encoder<nimble::TrivialEncoding<uint32_t>>::encode(
            *buffer_,
            uint32Values,
            nimble::CompressionType::Uncompressed,
            options);
    nimble::Buffer uint32SliceBuffer{*pool_};
    NIMBLE_ASSERT_THROW(
        nimble::TrivialEncoding<uint32_t>::slice(
            uint32Encoded,
            /*offset=*/0,
            /*length=*/0,
            uint32SliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        nimble::TrivialEncoding<uint32_t>::slice(
            uint32Encoded,
            /*offset=*/4,
            /*length=*/0,
            uint32SliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        nimble::TrivialEncoding<uint32_t>::slice(
            uint32Encoded,
            /*offset=*/2,
            /*length=*/2,
            uint32SliceBuffer,
            options),
        "");

    const auto boolValues = makeBoolValues(/*valueCount=*/3);
    const auto boolEncoded =
        nimble::test::Encoder<nimble::TrivialEncoding<bool>>::encode(
            *buffer_,
            boolValues,
            nimble::CompressionType::Uncompressed,
            options);
    nimble::Buffer boolSliceBuffer{*pool_};
    NIMBLE_ASSERT_THROW(
        nimble::TrivialEncoding<bool>::slice(
            boolEncoded,
            /*offset=*/0,
            /*length=*/0,
            boolSliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        nimble::TrivialEncoding<bool>::slice(
            boolEncoded,
            /*offset=*/4,
            /*length=*/0,
            boolSliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        nimble::TrivialEncoding<bool>::slice(
            boolEncoded,
            /*offset=*/2,
            /*length=*/2,
            boolSliceBuffer,
            options),
        "");

    std::vector<std::string> stringStorage;
    const auto stringValues = makeStringValues(stringStorage, /*valueCount=*/3);
    const auto stringEncoded = nimble::test::
        Encoder<nimble::TrivialEncoding<std::string_view>>::encode(
            *buffer_,
            stringValues,
            nimble::CompressionType::Uncompressed,
            options);
    nimble::Buffer stringSliceBuffer{*pool_};
    NIMBLE_ASSERT_THROW(
        nimble::TrivialEncoding<std::string_view>::slice(
            stringEncoded,
            /*offset=*/0,
            /*length=*/0,
            stringSliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        nimble::TrivialEncoding<std::string_view>::slice(
            stringEncoded,
            /*offset=*/4,
            /*length=*/0,
            stringSliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        nimble::TrivialEncoding<std::string_view>::slice(
            stringEncoded,
            /*offset=*/2,
            /*length=*/2,
            stringSliceBuffer,
            options),
        "");
  }
}
