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
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleCompare.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

#include <vector>

using namespace facebook;

template <typename T>
struct DictionaryTestValues;

template <>
struct DictionaryTestValues<uint32_t> {
  static std::vector<uint32_t> values() {
    return {10, 20, 10, 30, 40, 20, 30};
  }
};

template <>
struct DictionaryTestValues<double> {
  static std::vector<double> values() {
    return {1.25, 2.5, 1.25, -3.75, 8.0, 2.5, -3.75};
  }
};

template <>
struct DictionaryTestValues<std::string_view> {
  static std::vector<std::string_view> values() {
    return {"alpha", "beta", "alpha", "gamma", "delta", "beta", "gamma"};
  }
};

template <typename T>
class DictionaryEncodingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("DictionaryEncodingTest");
    pool_ = rootPool_->addLeafChild("DictionaryEncodingTestLeaf");
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  nimble::Vector<T> makeValues() {
    nimble::Vector<T> values{pool_.get()};
    const auto source = DictionaryTestValues<T>::values();
    for (const auto value : source) {
      values.push_back(value);
    }
    return values;
  }

  void* stringBufferFactory(uint32_t totalLength) {
    auto& stringBuffer = stringBuffers_.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
    return stringBuffer->asMutable<void>();
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
  std::vector<velox::BufferPtr> stringBuffers_;
};

using DictionaryEncodingTypes =
    ::testing::Types<uint32_t, double, std::string_view>;

TYPED_TEST_CASE(DictionaryEncodingTest, DictionaryEncodingTypes);

TYPED_TEST(DictionaryEncodingTest, slicePreservesAlphabetAndMaterializesRange) {
  using DataType = TypeParam;

  struct Range {
    uint32_t offset;
    uint32_t length;
  };

  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useVarint=" << useVarint);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    const auto values = this->makeValues();
    const auto encoded =
        nimble::test::Encoder<nimble::DictionaryEncoding<DataType>>::encode(
            *this->buffer_,
            values,
            nimble::CompressionType::Uncompressed,
            options);

    nimble::DictionaryEncoding<DataType> source{
        *this->pool_,
        encoded,
        [this](uint32_t totalLength) {
          return this->stringBufferFactory(totalLength);
        },
        options};
    const auto sourceDictionarySize = source.dictionarySize();

    for (const auto range :
         {Range{/*offset=*/0, /*length=*/1},
          Range{/*offset=*/1, /*length=*/3},
          Range{/*offset=*/2, /*length=*/4},
          Range{/*offset=*/6, /*length=*/1}}) {
      SCOPED_TRACE(
          testing::Message()
          << "offset=" << range.offset << ", length=" << range.length);
      nimble::Buffer sliceBuffer{*this->pool_};
      const auto sliced = nimble::DictionaryEncoding<DataType>::slice(
          encoded, range.offset, range.length, sliceBuffer, options);
      nimble::DictionaryEncoding<DataType> encoding{
          *this->pool_,
          sliced,
          [this](uint32_t totalLength) {
            return this->stringBufferFactory(totalLength);
          },
          options};

      EXPECT_EQ(encoding.encodingType(), nimble::EncodingType::Dictionary);
      EXPECT_EQ(encoding.dataType(), nimble::TypeTraits<DataType>::dataType);
      EXPECT_EQ(encoding.rowCount(), range.length);
      EXPECT_EQ(encoding.dictionarySize(), sourceDictionarySize);

      nimble::Vector<DataType> result{this->pool_.get(), range.length};
      encoding.materialize(range.length, result.data());
      for (uint32_t i = 0; i < range.length; ++i) {
        EXPECT_TRUE(
            nimble::NimbleCompare<DataType>::equals(
                result[i], values[range.offset + i]));
      }
    }
  }
}

TYPED_TEST(DictionaryEncodingTest, invalidSliceRange) {
  using DataType = TypeParam;

  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useVarint=" << useVarint);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    const auto values = this->makeValues();
    const auto encoded =
        nimble::test::Encoder<nimble::DictionaryEncoding<DataType>>::encode(
            *this->buffer_,
            values,
            nimble::CompressionType::Uncompressed,
            options);

    nimble::Buffer sliceBuffer{*this->pool_};
    NIMBLE_ASSERT_THROW(
        nimble::DictionaryEncoding<DataType>::slice(
            encoded,
            /*offset=*/0,
            /*length=*/0,
            sliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        nimble::DictionaryEncoding<DataType>::slice(
            encoded,
            /*offset=*/values.size() + 1,
            /*length=*/1,
            sliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        nimble::DictionaryEncoding<DataType>::slice(
            encoded,
            /*offset=*/values.size() - 1,
            /*length=*/2,
            sliceBuffer,
            options),
        "");
  }
}
