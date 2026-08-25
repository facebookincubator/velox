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
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleCompare.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

#include <limits>
#include <type_traits>
#include <vector>

using namespace facebook;

template <typename DataType, bool UseVarint>
struct TestConfig {
  using data_type = DataType;
  static constexpr bool useVarint = UseVarint;
};

#define TC(T) TestConfig<T, false>, TestConfig<T, true>

// Forward declaration
template <typename Config>
class ConstantEncodingTest;

// Helper to prepare values - must be at namespace scope
template <typename T, typename TestClass>
struct ValuesPreparer {
  static std::vector<nimble::Vector<T>> prepareValues(TestClass* test) {
    FAIL() << "unspecialized prepareValues() should not be called";
    return {};
  }
  static std::vector<nimble::Vector<T>> prepareFailureValues(TestClass* test) {
    FAIL() << "unspecialized prepareFailureValues() should not be called";
    return {};
  }
};

template <typename TestClass>
struct ValuesPreparer<double, TestClass> {
  static std::vector<nimble::Vector<double>> prepareValues(TestClass* test) {
    return {
        test->toVector({0.0}),
        test->toVector({0.0, 0.00}),
        test->toVector({-0.0, -0.00}),
        test->toVector({-2.1, -2.1, -2.1, -2.1, -2.1}),
        test->toVector({test->dNaN0, test->dNaN0, test->dNaN0}),
        test->toVector({test->dNaN1, test->dNaN1, test->dNaN1}),
        test->toVector({test->dNaN2, test->dNaN2, test->dNaN2})};
  }
  static std::vector<nimble::Vector<double>> prepareFailureValues(
      TestClass* test) {
    return {
        test->toVector({-0.0, -0.00, -0.0000001}),
        test->toVector({-2.1, -2.1, -2.1, -2.1, -2.2}),
        test->toVector({test->dNaN0, test->dNaN0, test->dNaN1})};
  }
};

template <typename TestClass>
struct ValuesPreparer<float, TestClass> {
  static std::vector<nimble::Vector<float>> prepareValues(TestClass* test) {
    return {
        test->toVector({0.0f}),
        test->toVector({0.0f, 0.00f}),
        test->toVector({-0.0f, -0.00f}),
        test->toVector({-2.1f, -2.1f, -2.1f, -2.1f, -2.1f}),
        test->toVector({test->fNaN0, test->fNaN0, test->fNaN0}),
        test->toVector({test->fNaN1, test->fNaN1, test->fNaN1}),
        test->toVector({test->fNaN2, test->fNaN2, test->fNaN2})};
  }
  static std::vector<nimble::Vector<float>> prepareFailureValues(
      TestClass* test) {
    return {
        test->toVector({-0.0f, -0.00f, -0.0000001f}),
        test->toVector({-2.1f, -2.1f, -2.1f, -2.1f, -2.2f}),
        test->toVector({test->fNaN0, test->fNaN0, test->fNaN2})};
  }
};

template <typename TestClass>
struct ValuesPreparer<int32_t, TestClass> {
  static std::vector<nimble::Vector<int32_t>> prepareValues(TestClass* test) {
    return {test->toVector({1}), test->toVector({3, 3, 3})};
  }
  static std::vector<nimble::Vector<int32_t>> prepareFailureValues(
      TestClass* test) {
    return {test->toVector({3, 2, 3})};
  }
};

template <typename Config>
class ConstantEncodingTest : public ::testing::Test {
 protected:
  // Make helper templates friends so they can access protected members
  template <typename T, typename TestClass>
  friend struct ValuesPreparer;

  void SetUp() override {
    pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  template <typename T>
  nimble::Vector<T> toVector(std::initializer_list<T> l) {
    nimble::Vector<T> v{pool_.get()};
    v.insert(v.end(), l.begin(), l.end());
    return v;
  }

  template <typename T>
  std::vector<nimble::Vector<T>> prepareValues() {
    if constexpr (std::is_same_v<T, double>) {
      const double dNaN0 = std::numeric_limits<double>::quiet_NaN();
      const double dNaN1 = std::numeric_limits<double>::signaling_NaN();
      const double dNaN2 =
          nimble::EncodingPhysicalType<double>::asEncodingLogicalType(
              (nimble::EncodingPhysicalType<double>::asEncodingPhysicalType(
                   dNaN0) |
               0x3));
      return {
          toVector({0.0}),
          toVector({0.0, 0.00}),
          toVector({-0.0, -0.00}),
          toVector({-2.1, -2.1, -2.1, -2.1, -2.1}),
          toVector({dNaN0, dNaN0, dNaN0}),
          toVector({dNaN1, dNaN1, dNaN1}),
          toVector({dNaN2, dNaN2, dNaN2})};
    } else if constexpr (std::is_same_v<T, float>) {
      const float fNaN0 = std::numeric_limits<float>::quiet_NaN();
      const float fNaN1 = std::numeric_limits<float>::signaling_NaN();
      const float fNaN2 =
          nimble::EncodingPhysicalType<float>::asEncodingLogicalType(
              (nimble::EncodingPhysicalType<float>::asEncodingPhysicalType(
                   fNaN0) |
               0x3));
      return {
          toVector({0.0f}),
          toVector({0.0f, 0.00f}),
          toVector({-0.0f, -0.00f}),
          toVector({-2.1f, -2.1f, -2.1f, -2.1f, -2.1f}),
          toVector({fNaN0, fNaN0, fNaN0}),
          toVector({fNaN1, fNaN1, fNaN1}),
          toVector({fNaN2, fNaN2, fNaN2})};
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return {toVector({1}), toVector({3, 3, 3})};
    } else {
      static_assert(!std::is_same_v<T, T>, "Unsupported test type");
    }
  }

  template <typename T>
  std::vector<nimble::Vector<T>> prepareFailureValues() {
    if constexpr (std::is_same_v<T, double>) {
      const double dNaN0 = std::numeric_limits<double>::quiet_NaN();
      const double dNaN1 = std::numeric_limits<double>::signaling_NaN();
      return {
          toVector({-0.0, -0.00, -0.0000001}),
          toVector({-2.1, -2.1, -2.1, -2.1, -2.2}),
          toVector({dNaN0, dNaN0, dNaN1})};
    } else if constexpr (std::is_same_v<T, float>) {
      const float fNaN0 = std::numeric_limits<float>::quiet_NaN();
      const float fNaN2 =
          nimble::EncodingPhysicalType<float>::asEncodingLogicalType(
              (nimble::EncodingPhysicalType<float>::asEncodingPhysicalType(
                   fNaN0) |
               0x3));
      return {
          toVector({-0.0f, -0.00f, -0.0000001f}),
          toVector({-2.1f, -2.1f, -2.1f, -2.1f, -2.2f}),
          toVector({fNaN0, fNaN0, fNaN2})};
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return {toVector({3, 2, 3})};
    } else {
      static_assert(!std::is_same_v<T, T>, "Unsupported test type");
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

#define NUM_TYPES TC(int32_t), TC(double), TC(float)

using TestTypes = ::testing::Types<NUM_TYPES>;

TYPED_TEST_CASE(ConstantEncodingTest, TestTypes);

TYPED_TEST(ConstantEncodingTest, serializeThenDeserialize) {
  using DataType = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto valueGroups = this->template prepareValues<DataType>();
  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  for (const auto& values : valueGroups) {
    auto encoding = nimble::test::Encoder<nimble::ConstantEncoding<DataType>>::
        createEncoding(
            *this->buffer_,
            values,
            stringBufferFactory,
            nimble::CompressionType::Uncompressed,
            options);

    uint32_t rowCount = values.size();
    nimble::Vector<DataType> result(this->pool_.get(), rowCount);
    encoding->materialize(rowCount, result.data());

    EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Constant);
    EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<DataType>::dataType);
    EXPECT_EQ(encoding->rowCount(), rowCount);
    for (uint32_t i = 0; i < rowCount; ++i) {
      EXPECT_TRUE(
          nimble::NimbleCompare<DataType>::equals(result[i], values[i]));
    }
  }
}

TYPED_TEST(ConstantEncodingTest, nonConstantFailure) {
  using DataType = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto valueGroups = this->template prepareFailureValues<DataType>();
  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  for (const auto& values : valueGroups) {
    try {
      nimble::test::Encoder<nimble::ConstantEncoding<DataType>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);
      FAIL() << "ConstantEncodingTest should fail due to non constant data";
    } catch (const nimble::NimbleUserError& e) {
      EXPECT_EQ(nimble::error_code::IncompatibleEncoding, e.errorCode());
      EXPECT_EQ("ConstantEncoding requires constant data.", e.errorMessage());
    }
  }
}

TYPED_TEST(ConstantEncodingTest, slice) {
  using DataType = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  struct Range {
    uint32_t offset;
    uint32_t length;
  };

  const auto values = this->template toVector<DataType>(
      {static_cast<DataType>(7),
       static_cast<DataType>(7),
       static_cast<DataType>(7),
       static_cast<DataType>(7)});
  const auto encoded =
      nimble::test::Encoder<nimble::ConstantEncoding<DataType>>::encode(
          *this->buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options);

  for (const auto range :
       {Range{/*offset=*/0, /*length=*/1},
        Range{/*offset=*/1, /*length=*/2},
        Range{/*offset=*/0, /*length=*/3},
        Range{/*offset=*/3, /*length=*/1}}) {
    SCOPED_TRACE(
        testing::Message() << "offset=" << range.offset
                           << ", length=" << range.length);
    nimble::Buffer sliceBuffer{*this->pool_};
    const auto sliced = nimble::ConstantEncoding<DataType>::slice(
        encoded, range.offset, range.length, sliceBuffer, options);
    nimble::ConstantEncoding<DataType> encoding{
        *this->pool_,
        sliced,
        [](uint32_t /*totalLength*/) -> void* { return nullptr; },
        options};

    EXPECT_EQ(encoding.encodingType(), nimble::EncodingType::Constant);
    EXPECT_EQ(encoding.dataType(), nimble::TypeTraits<DataType>::dataType);
    EXPECT_EQ(encoding.rowCount(), range.length);
    nimble::Vector<DataType> result(this->pool_.get(), range.length);
    encoding.materialize(range.length, result.data());
    for (uint32_t i = 0; i < range.length; ++i) {
      EXPECT_TRUE(
          nimble::NimbleCompare<DataType>::equals(
              result[i], values[range.offset + i]));
    }
  }
}

TYPED_TEST(ConstantEncodingTest, invalidSliceRange) {
  using DataType = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  const auto values = this->template toVector<DataType>(
      {static_cast<DataType>(7),
       static_cast<DataType>(7),
       static_cast<DataType>(7),
       static_cast<DataType>(7)});
  const auto encoded =
      nimble::test::Encoder<nimble::ConstantEncoding<DataType>>::encode(
          *this->buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options);

  nimble::Buffer invalidSliceBuffer{*this->pool_};
  NIMBLE_ASSERT_THROW(
      nimble::ConstantEncoding<DataType>::slice(
          encoded,
          /*offset=*/0,
          /*length=*/0,
          invalidSliceBuffer,
          options),
      "");
  NIMBLE_ASSERT_THROW(
      nimble::ConstantEncoding<DataType>::slice(
          encoded,
          /*offset=*/5,
          /*length=*/0,
          invalidSliceBuffer,
          options),
      "");
  NIMBLE_ASSERT_THROW(
      nimble::ConstantEncoding<DataType>::slice(
          encoded,
          /*offset=*/3,
          /*length=*/2,
          invalidSliceBuffer,
          options),
      "");
}

class ConstantEncodingStringTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("ConstantEncodingTest");
    pool_ = rootPool_->addLeafChild("ConstantEncodingTestLeaf");
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

TEST_F(ConstantEncodingStringTest, slice) {
  struct Range {
    uint32_t offset;
    uint32_t length;
  };

  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useVarint=" << useVarint);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    const std::string value{"constant-value"};
    nimble::Vector<std::string_view> values{pool_.get()};
    for (uint32_t i = 0; i < 4; ++i) {
      values.push_back(value);
    }

    const auto encoded = nimble::test::
        Encoder<nimble::ConstantEncoding<std::string_view>>::encode(
            *buffer_, values, nimble::CompressionType::Uncompressed, options);

    for (const auto range :
         {Range{/*offset=*/1, /*length=*/2},
          Range{/*offset=*/0, /*length=*/3}}) {
      SCOPED_TRACE(
          testing::Message()
          << "offset=" << range.offset << ", length=" << range.length);
      nimble::Buffer sliceBuffer{*pool_};
      const auto sliced = nimble::ConstantEncoding<std::string_view>::slice(
          encoded, range.offset, range.length, sliceBuffer, options);

      std::vector<velox::BufferPtr> stringBuffers;
      const auto stringBufferFactory = [&](uint32_t totalLength) -> void* {
        auto& stringBuffer = stringBuffers.emplace_back(
            velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
        return stringBuffer->asMutable<void>();
      };
      nimble::ConstantEncoding<std::string_view> encoding{
          *pool_, sliced, stringBufferFactory, options};

      EXPECT_EQ(encoding.encodingType(), nimble::EncodingType::Constant);
      EXPECT_EQ(encoding.dataType(), nimble::DataType::String);
      EXPECT_EQ(encoding.rowCount(), range.length);
      nimble::Vector<std::string_view> result(pool_.get(), range.length);
      encoding.materialize(range.length, result.data());
      for (uint32_t i = 0; i < range.length; ++i) {
        EXPECT_EQ(result[i], values[range.offset + i]);
      }
    }
  }
}

TEST_F(ConstantEncodingStringTest, invalidSliceRange) {
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useVarint=" << useVarint);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    const std::string value{"constant-value"};
    nimble::Vector<std::string_view> values{pool_.get()};
    for (uint32_t i = 0; i < 4; ++i) {
      values.push_back(value);
    }

    const auto encoded = nimble::test::
        Encoder<nimble::ConstantEncoding<std::string_view>>::encode(
            *buffer_, values, nimble::CompressionType::Uncompressed, options);

    nimble::Buffer invalidSliceBuffer{*pool_};
    NIMBLE_ASSERT_THROW(
        nimble::ConstantEncoding<std::string_view>::slice(
            encoded,
            /*offset=*/0,
            /*length=*/0,
            invalidSliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        nimble::ConstantEncoding<std::string_view>::slice(
            encoded,
            /*offset=*/3,
            /*length=*/2,
            invalidSliceBuffer,
            options),
        "");
  }
}
