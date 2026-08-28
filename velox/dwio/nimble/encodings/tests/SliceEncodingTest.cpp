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

#include "velox/dwio/nimble/encodings/EncodingSliceFactory.h"

#include <algorithm>
#include <memory>
#include <random>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
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

  template <typename EncodingType>
  nimble::Vector<typename EncodingType::cppDataType> makeValuesForEncoding() {
    using DataType = typename EncodingType::cppDataType;
    if constexpr (std::is_same_v<
                      EncodingType,
                      nimble::ConstantEncoding<DataType>>) {
      return makeVector<DataType>(
          {static_cast<DataType>(7),
           static_cast<DataType>(7),
           static_cast<DataType>(7),
           static_cast<DataType>(7),
           static_cast<DataType>(7),
           static_cast<DataType>(7)});
    } else if constexpr (std::is_same_v<
                             EncodingType,
                             nimble::MainlyConstantEncoding<DataType>>) {
      return makeVector<DataType>(
          {static_cast<DataType>(10),
           static_cast<DataType>(10),
           static_cast<DataType>(12),
           static_cast<DataType>(10),
           static_cast<DataType>(14),
           static_cast<DataType>(10)});
    } else if constexpr (std::is_same_v<
                             EncodingType,
                             nimble::HuffmanEncoding<DataType>>) {
      return makeVector<DataType>(
          {static_cast<DataType>(10),
           static_cast<DataType>(11),
           static_cast<DataType>(10),
           static_cast<DataType>(12),
           static_cast<DataType>(10),
           static_cast<DataType>(13)});
    } else if constexpr (std::is_same_v<
                             EncodingType,
                             nimble::ALPEncoding<DataType>>) {
      return makeVector<DataType>(
          {static_cast<DataType>(1.25),
           static_cast<DataType>(2.5),
           static_cast<DataType>(3.75),
           static_cast<DataType>(4.0),
           static_cast<DataType>(5.125),
           static_cast<DataType>(6.25)});
    } else {
      return makeVector<DataType>(
          {static_cast<DataType>(10),
           static_cast<DataType>(11),
           static_cast<DataType>(12),
           static_cast<DataType>(13),
           static_cast<DataType>(14),
           static_cast<DataType>(15)});
    }
  }

  template <typename EncodingType>
  nimble::Vector<typename EncodingType::cppDataType>
  makeRandomValuesForEncoding(std::mt19937& rng, uint32_t rowCount) {
    using DataType = typename EncodingType::cppDataType;
    nimble::Vector<DataType> values{pool_.get()};
    values.reserve(rowCount);

    const auto nextIntegerValue = [&] {
      return static_cast<DataType>(
          std::uniform_int_distribution<uint32_t>{0, 1024}(rng));
    };

    if constexpr (std::is_same_v<
                      EncodingType,
                      nimble::ConstantEncoding<DataType>>) {
      const auto value = nextIntegerValue();
      for (uint32_t i = 0; i < rowCount; ++i) {
        values.push_back(value);
      }
    } else if constexpr (std::is_same_v<
                             EncodingType,
                             nimble::MainlyConstantEncoding<DataType>>) {
      const auto commonValue = nextIntegerValue();
      for (uint32_t i = 0; i < rowCount; ++i) {
        values.push_back(i % 5 == 2 ? nextIntegerValue() : commonValue);
      }
    } else if constexpr (std::is_same_v<
                             EncodingType,
                             nimble::HuffmanEncoding<DataType>>) {
      for (uint32_t i = 0; i < rowCount; ++i) {
        values.push_back(
            static_cast<DataType>(
                std::uniform_int_distribution<uint32_t>{0, 7}(rng)));
      }
    } else if constexpr (std::is_same_v<
                             EncodingType,
                             nimble::PFOREncoding<DataType>>) {
      for (uint32_t i = 0; i < rowCount; ++i) {
        values.push_back(
            static_cast<DataType>(
                std::uniform_int_distribution<uint32_t>{0, 15}(rng)));
      }
    } else if constexpr (std::is_same_v<
                             EncodingType,
                             nimble::ALPEncoding<DataType>>) {
      for (uint32_t i = 0; i < rowCount; ++i) {
        values.push_back(
            static_cast<DataType>(
                std::uniform_int_distribution<uint32_t>{0, 4096}(rng) / 4.0));
      }
    } else if constexpr (std::is_same_v<
                             EncodingType,
                             nimble::RLEEncoding<DataType>>) {
      while (values.size() < rowCount) {
        const auto value = nextIntegerValue();
        const auto runLength =
            std::uniform_int_distribution<uint32_t>{1, 8}(rng);
        for (uint32_t i = 0; i < runLength && values.size() < rowCount; ++i) {
          values.push_back(value);
        }
      }
    } else {
      for (uint32_t i = 0; i < rowCount; ++i) {
        values.push_back(nextIntegerValue());
      }
    }

    return values;
  }

  std::unique_ptr<nimble::Encoding> createEncoding(std::string_view encoded) {
    return nimble::EncodingFactory{}.create(
        *pool_, encoded, [&](uint32_t totalLength) {
          auto& buffer = stringBuffers_.emplace_back(
              velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
          return buffer->asMutable<void>();
        });
  }

  std::string_view
  slice(std::string_view encoded, uint32_t offset, uint32_t length) {
    return nimble::EncodingFactory::slice(
        encoded, offset, length, *buffer_, nimble::Encoding::Options{});
  }

  template <typename T>
  std::vector<T>
  stdVector(const nimble::Vector<T>& values, uint32_t offset, uint32_t length) {
    return std::vector<T>(
        values.begin() + offset, values.begin() + offset + length);
  }

  template <typename EncodingType, typename T>
  void expectSliceMaterializes(
      std::string_view name,
      nimble::EncodingType expectedEncodingType,
      const nimble::Vector<T>& values,
      uint32_t offset,
      uint32_t length) {
    SCOPED_TRACE(name);
    const auto encoded =
        nimble::test::Encoder<EncodingType>::encode(*buffer_, values);

    const auto sliced = slice(encoded, offset, length);
    EXPECT_NE(sliced.data(), encoded.data());

    auto encoding = createEncoding(sliced);

    EXPECT_EQ(encoding->encodingType(), expectedEncodingType);
    EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<T>::dataType);
    EXPECT_EQ(encoding->rowCount(), length);

    nimble::Vector<T> output{pool_.get(), length};
    encoding->materialize(length, output.data());

    EXPECT_EQ(
        std::vector<T>(output.begin(), output.end()),
        stdVector(values, offset, length));
  }

  template <typename EncodingType, typename T>
  void expectFactorySliceMaterializes(
      const nimble::Vector<T>& values,
      uint32_t offset,
      uint32_t length) {
    const auto encoded =
        nimble::test::Encoder<EncodingType>::encode(*buffer_, values);
    const auto sliced = slice(encoded, offset, length);
    auto encoding = createEncoding(sliced);

    auto expectedEncodingType =
        nimble::test::Encoder<EncodingType>::encodingType();
    if constexpr (std::is_same_v<
                      EncodingType,
                      nimble::MainlyConstantEncoding<T>>) {
      const auto fullRange = offset == 0 && length == values.size();
      bool onlyCommonRows{true};
      for (uint32_t row = offset; row < offset + length; ++row) {
        onlyCommonRows &= row % 5 != 2;
      }
      if (!fullRange && onlyCommonRows) {
        expectedEncodingType = nimble::EncodingType::Constant;
      }
    }

    EXPECT_EQ(encoding->encodingType(), expectedEncodingType);
    EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<T>::dataType);
    EXPECT_EQ(encoding->rowCount(), length);

    nimble::Vector<T> output{pool_.get(), length};
    encoding->materialize(length, output.data());

    EXPECT_EQ(
        std::vector<T>(output.begin(), output.end()),
        stdVector(values, offset, length));
  }

  template <typename EncodingType>
  void expectBoolSliceMaterializes(
      std::string_view name,
      nimble::EncodingType expectedEncodingType,
      const nimble::Vector<bool>& values,
      uint32_t offset,
      uint32_t length) {
    SCOPED_TRACE(name);
    const auto encoded =
        nimble::test::Encoder<EncodingType>::encode(*buffer_, values);

    const auto sliced = slice(encoded, offset, length);
    EXPECT_NE(sliced.data(), encoded.data());

    auto encoding = createEncoding(sliced);

    EXPECT_EQ(encoding->encodingType(), expectedEncodingType);
    EXPECT_EQ(encoding->dataType(), nimble::DataType::Bool);
    EXPECT_EQ(encoding->rowCount(), length);

    nimble::Vector<bool> output{pool_.get(), length};
    encoding->materialize(length, output.data());

    EXPECT_EQ(
        std::vector<bool>(output.begin(), output.end()),
        stdVector(values, offset, length));
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
  std::vector<velox::BufferPtr> stringBuffers_;
};

template <typename Encoding>
struct SliceEncodingConfig {
  using EncodingType = Encoding;
};

using SliceEncodingTypes = ::testing::Types<
    SliceEncodingConfig<nimble::TrivialEncoding<uint32_t>>,
    SliceEncodingConfig<nimble::DictionaryEncoding<uint32_t>>,
    SliceEncodingConfig<nimble::FixedBitWidthEncoding<uint32_t>>,
    SliceEncodingConfig<nimble::VarintEncoding<uint32_t>>,
    SliceEncodingConfig<nimble::RLEEncoding<uint32_t>>,
    SliceEncodingConfig<nimble::ConstantEncoding<uint32_t>>,
    SliceEncodingConfig<nimble::MainlyConstantEncoding<uint32_t>>,
    SliceEncodingConfig<nimble::DeltaEncoding<uint32_t>>,
    SliceEncodingConfig<nimble::BlockBitPackingEncoding<uint32_t>>,
    SliceEncodingConfig<nimble::PFOREncoding<uint32_t>>,
    SliceEncodingConfig<nimble::SimdForBitpackEncoding<uint32_t>>,
    SliceEncodingConfig<nimble::HuffmanEncoding<uint32_t>>,
    SliceEncodingConfig<nimble::ForEncoding<uint32_t>>,
    SliceEncodingConfig<nimble::ALPEncoding<double>>>;

template <typename Config>
class SliceEncodingTypedTest : public SliceEncodingTest {};

TYPED_TEST_CASE(SliceEncodingTypedTest, SliceEncodingTypes);

TYPED_TEST(SliceEncodingTypedTest, materializesRange) {
  using EncodingType = typename TypeParam::EncodingType;
  const auto values = this->template makeValuesForEncoding<EncodingType>();
  constexpr uint32_t offset{1};
  constexpr uint32_t length{3};

  this->template expectSliceMaterializes<EncodingType>(
      nimble::toString(nimble::test::Encoder<EncodingType>::encodingType()),
      nimble::test::Encoder<EncodingType>::encodingType(),
      values,
      offset,
      length);
}

TYPED_TEST(SliceEncodingTypedTest, rejectsZeroLengthRange) {
  using EncodingType = typename TypeParam::EncodingType;
  const auto values = this->template makeValuesForEncoding<EncodingType>();
  const auto encoded =
      nimble::test::Encoder<EncodingType>::encode(*this->buffer_, values);

  NIMBLE_ASSERT_THROW(this->slice(encoded, /*offset=*/0, /*length=*/0), "");
}

TEST_F(SliceEncodingTest, dictionaryRejectsZeroLengthRange) {
  const auto values = makeVector<uint32_t>({10, 11, 10, 12});
  const auto encoded =
      nimble::test::Encoder<nimble::DictionaryEncoding<uint32_t>>::encode(
          *buffer_, values);

  NIMBLE_ASSERT_THROW(
      slice(encoded, /*offset=*/1, /*length=*/0), "Cannot slice zero rows.");
}

TYPED_TEST(SliceEncodingTypedTest, materializesRandomRanges) {
  using EncodingType = typename TypeParam::EncodingType;
  constexpr uint32_t kIterations{64};
  std::mt19937 rng{
      0x5eed0000u +
      static_cast<uint32_t>(
          nimble::test::Encoder<EncodingType>::encodingType())};

  for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
    SCOPED_TRACE(testing::Message() << "iteration=" << iteration);
    constexpr bool kRequiresAtLeastTwoRows = std::is_same_v<
        EncodingType,
        nimble::HuffmanEncoding<typename EncodingType::cppDataType>>;
    const auto rowCount = std::uniform_int_distribution<uint32_t>{
        kRequiresAtLeastTwoRows ? 2U : 1U, 128}(rng);
    const auto values =
        this->template makeRandomValuesForEncoding<EncodingType>(rng, rowCount);
    const auto offset = std::uniform_int_distribution<uint32_t>{
        0, rowCount - (kRequiresAtLeastTwoRows ? 2U : 1U)}(rng);
    const auto length = std::uniform_int_distribution<uint32_t>{
        kRequiresAtLeastTwoRows ? 2U : 1U, rowCount - offset}(rng);

    SCOPED_TRACE(
        testing::Message() << "rowCount=" << rowCount << ", offset=" << offset
                           << ", length=" << length);
    this->template expectFactorySliceMaterializes<EncodingType>(
        values, offset, length);
  }
}

TEST_F(SliceEncodingTest, materializesNumericRange) {
  const auto values = makeVector<int32_t>({10, 11, 12, 13, 14, 15});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto sliced = slice(encoded, /*offset=*/2, /*length=*/3);
  EXPECT_NE(sliced.data(), encoded.data());
  auto encoding = createEncoding(sliced);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Trivial);
  EXPECT_EQ(encoding->dataType(), nimble::DataType::Int32);
  EXPECT_EQ(encoding->rowCount(), 3);

  nimble::Vector<int32_t> output{pool_.get(), 3};
  encoding->materialize(3, output.data());

  const std::vector<int32_t> expected{12, 13, 14};
  EXPECT_EQ(std::vector<int32_t>(output.begin(), output.end()), expected);
}

TEST_F(SliceEncodingTest, fullRangeCopiesToOutputBuffer) {
  const auto values = makeVector<int32_t>({10, 11, 12, 13});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto sliced = slice(encoded, /*offset=*/0, /*length=*/4);

  EXPECT_NE(sliced.data(), encoded.data());
  EXPECT_EQ(sliced, encoded);
}

TEST_F(SliceEncodingTest, materializesAfterSkip) {
  const auto values = makeVector<int32_t>({10, 11, 12, 13, 14, 15});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto sliced = slice(encoded, /*offset=*/1, /*length=*/4);
  EXPECT_NE(sliced.data(), encoded.data());

  auto encoding = createEncoding(sliced);
  encoding->skip(2);

  nimble::Vector<int32_t> output{pool_.get(), 2};
  encoding->materialize(2, output.data());

  const std::vector<int32_t> expected{13, 14};
  EXPECT_EQ(std::vector<int32_t>(output.begin(), output.end()), expected);
}

TEST_F(SliceEncodingTest, materializesStringRange) {
  const auto values =
      makeVector<std::string_view>({"alpha", "beta", "gamma", "delta"});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<std::string_view>>::encode(
          *buffer_, values);

  const auto sliced = slice(encoded, /*offset=*/1, /*length=*/2);
  EXPECT_NE(sliced.data(), encoded.data());

  EXPECT_NE(sliced.find("beta"), std::string_view::npos);
  EXPECT_NE(sliced.find("gamma"), std::string_view::npos);
  EXPECT_EQ(sliced.find("alpha"), std::string_view::npos);
  EXPECT_EQ(sliced.find("delta"), std::string_view::npos);

  auto encoding = createEncoding(sliced);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Trivial);
  EXPECT_EQ(encoding->dataType(), nimble::DataType::String);
  EXPECT_EQ(encoding->rowCount(), 2);

  nimble::Vector<std::string_view> output{pool_.get(), 2};
  encoding->materialize(2, output.data());

  const std::vector<std::string_view> expected{"beta", "gamma"};
  EXPECT_EQ(
      std::vector<std::string_view>(output.begin(), output.end()), expected);
}

TEST_F(SliceEncodingTest, materializesBoolBits) {
  const auto values = makeVector<bool>({true, false, true, true, false});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<bool>>::encode(
          *buffer_, values);

  const auto sliced = slice(encoded, /*offset=*/1, /*length=*/3);
  EXPECT_NE(sliced.data(), encoded.data());

  auto encoding = createEncoding(sliced);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Trivial);
  EXPECT_EQ(encoding->dataType(), nimble::DataType::Bool);
  EXPECT_EQ(encoding->rowCount(), 3);

  uint64_t bits{0};
  encoding->materializeBoolsAsBits(/*rowCount=*/3, &bits, /*begin=*/0);

  EXPECT_FALSE(velox::bits::isBitSet(&bits, 0));
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 1));
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 2));
}

TEST_F(SliceEncodingTest, materializesConstantRangeWithoutWrapper) {
  const auto values = makeVector<int32_t>({7, 7, 7, 7, 7});
  const auto encoded =
      nimble::test::Encoder<nimble::ConstantEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto sliced = slice(encoded, /*offset=*/2, /*length=*/2);
  EXPECT_NE(sliced.data(), encoded.data());

  auto encoding = createEncoding(sliced);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Constant);
  EXPECT_EQ(encoding->dataType(), nimble::DataType::Int32);
  EXPECT_EQ(encoding->rowCount(), 2);

  nimble::Vector<int32_t> output{pool_.get(), 2};
  encoding->materialize(2, output.data());

  const std::vector<int32_t> expected{7, 7};
  EXPECT_EQ(std::vector<int32_t>(output.begin(), output.end()), expected);
}

TEST_F(SliceEncodingTest, materializesRleRangeWithoutWrapper) {
  const auto values = makeVector<int32_t>({10, 10, 11, 11, 11, 12, 12});
  const auto encoded =
      nimble::test::Encoder<nimble::RLEEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto sliced = slice(encoded, /*offset=*/1, /*length=*/5);
  EXPECT_NE(sliced.data(), encoded.data());

  auto encoding = createEncoding(sliced);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::RLE);
  EXPECT_EQ(encoding->dataType(), nimble::DataType::Int32);
  EXPECT_EQ(encoding->rowCount(), 5);

  nimble::Vector<int32_t> output{pool_.get(), 5};
  encoding->materialize(5, output.data());

  const std::vector<int32_t> expected{10, 11, 11, 11, 12};
  EXPECT_EQ(std::vector<int32_t>(output.begin(), output.end()), expected);
}

TEST_F(SliceEncodingTest, materializesRleBoolRangeWithoutWrapper) {
  const auto values = makeVector<bool>({false, false, true, true, true, false});
  const auto encoded = nimble::test::Encoder<nimble::RLEEncoding<bool>>::encode(
      *buffer_, values);

  const auto sliced = slice(encoded, /*offset=*/2, /*length=*/3);
  EXPECT_NE(sliced.data(), encoded.data());

  auto encoding = createEncoding(sliced);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::RLE);
  EXPECT_EQ(encoding->dataType(), nimble::DataType::Bool);
  EXPECT_EQ(encoding->rowCount(), 3);

  uint64_t bits{0};
  encoding->materializeBoolsAsBits(/*rowCount=*/3, &bits, /*begin=*/0);

  EXPECT_TRUE(velox::bits::isBitSet(&bits, 0));
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 1));
  EXPECT_TRUE(velox::bits::isBitSet(&bits, 2));
}

TEST_F(SliceEncodingTest, materializesFixedBitWidthRangeWithoutWrapper) {
  const auto values = makeVector<int32_t>({10, 11, 12, 13, 14});
  const auto encoded =
      nimble::test::Encoder<nimble::FixedBitWidthEncoding<int32_t>>::encode(
          *buffer_, values);

  const auto sliced = slice(encoded, /*offset=*/1, /*length=*/3);
  EXPECT_NE(sliced.data(), encoded.data());

  auto encoding = createEncoding(sliced);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::FixedBitWidth);
  EXPECT_EQ(encoding->dataType(), nimble::DataType::Int32);
  EXPECT_EQ(encoding->rowCount(), 3);

  nimble::Vector<int32_t> output{pool_.get(), 3};
  encoding->materialize(3, output.data());

  const std::vector<int32_t> expected{11, 12, 13};
  EXPECT_EQ(std::vector<int32_t>(output.begin(), output.end()), expected);
}

TEST_F(SliceEncodingTest, materializesCompressedTrivialRangeWithoutWrapper) {
  nimble::Vector<uint32_t> values{pool_.get()};
  values.resize(1024);
  std::fill(values.begin(), values.end(), 42);
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<uint32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Zstd);

  const auto sliced = slice(encoded, /*offset=*/128, /*length=*/3);
  EXPECT_NE(sliced.data(), encoded.data());

  auto encoding = createEncoding(sliced);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Trivial);
  EXPECT_EQ(encoding->dataType(), nimble::DataType::Uint32);
  EXPECT_EQ(encoding->rowCount(), 3);

  nimble::Vector<uint32_t> output{pool_.get(), 3};
  encoding->materialize(3, output.data());

  const std::vector<uint32_t> expected{42, 42, 42};
  EXPECT_EQ(std::vector<uint32_t>(output.begin(), output.end()), expected);
}

TEST_F(SliceEncodingTest, materializesNativeSliceForBoolEncoding) {
  expectBoolSliceMaterializes<nimble::SparseBoolEncoding>(
      "SparseBool",
      nimble::EncodingType::SparseBool,
      makeVector<bool>({false, true, false, false, true, false}),
      /*offset=*/1,
      /*length=*/3);
}

TEST_F(SliceEncodingTest, materializesNativeSliceForNullableEncoding) {
  const auto values = makeVector<uint32_t>({10, 11, 12, 13, 14, 15});
  const auto nulls = makeVector<bool>({true, true, true, true, true, true});
  const auto encoded =
      nimble::test::Encoder<nimble::NullableEncoding<uint32_t>>::encodeNullable(
          *buffer_, values, nulls);

  const auto sliced = slice(encoded, /*offset=*/2, /*length=*/3);
  EXPECT_NE(sliced.data(), encoded.data());

  auto encoding = createEncoding(sliced);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Nullable);
  EXPECT_EQ(encoding->dataType(), nimble::DataType::Uint32);
  EXPECT_EQ(encoding->rowCount(), 3);

  nimble::Vector<uint32_t> output{pool_.get(), 3};
  encoding->materialize(3, output.data());

  const std::vector<uint32_t> expected{12, 13, 14};
  EXPECT_EQ(std::vector<uint32_t>(output.begin(), output.end()), expected);
}

TEST_F(SliceEncodingTest, rejectsOutOfRangeSlice) {
  const auto values = makeVector<int32_t>({10, 11, 12});
  const auto encoded =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);

  NIMBLE_ASSERT_THROW(slice(encoded, /*offset=*/2, /*length=*/2), "");
}
