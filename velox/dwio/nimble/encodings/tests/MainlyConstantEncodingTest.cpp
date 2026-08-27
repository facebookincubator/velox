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
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleCompare.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

#include <initializer_list>
#include <limits>
#include <string>
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
class MainlyConstantEncodingTest;

// Helper to prepare values - must be at namespace scope
template <typename T, typename TestClass>
struct MainlyConstantValuesPreparer {
  static std::vector<nimble::Vector<T>> prepareValues(TestClass* test) {
    FAIL() << "unspecialized prepareValues() should not be called";
    return {};
  }
};

template <typename TestClass>
struct MainlyConstantValuesPreparer<double, TestClass> {
  static std::vector<nimble::Vector<double>> prepareValues(TestClass* test) {
    return {
        test->toVector({0.0}),
        test->toVector({0.0, 0.00, 0.12}),
        test->toVector({-2.1, -2.1, -2.3, -2.1, -2.1}),
        test->toVector(
            {test->dNaN0, test->dNaN0, test->dNaN1, test->dNaN2, test->dNaN0})};
  }
};

template <typename TestClass>
struct MainlyConstantValuesPreparer<float, TestClass> {
  static std::vector<nimble::Vector<float>> prepareValues(TestClass* test) {
    return {
        test->toVector({0.0f}),
        test->toVector({0.0f, 0.00f, 0.12f}),
        test->toVector({-2.1f, -2.1f, -2.3f, -2.1f, -2.1f}),
        test->toVector(
            {test->fNaN0, test->fNaN0, test->fNaN1, test->fNaN2, test->fNaN2})};
  }
};

template <typename TestClass>
struct MainlyConstantValuesPreparer<int32_t, TestClass> {
  static std::vector<nimble::Vector<int32_t>> prepareValues(TestClass* test) {
    return {test->toVector({3, 3, 3, 1, 3})};
  }
};

template <typename Config>
class MainlyConstantEncodingTest : public ::testing::Test {
 protected:
  // Make helper templates friends so they can access protected members
  template <typename T, typename TestClass>
  friend struct MainlyConstantValuesPreparer;

  void SetUp() override {
    pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  template <typename T>
  nimble::Vector<T> toVector(std::initializer_list<T> values) {
    nimble::Vector<T> vector{pool_.get()};
    vector.insert(vector.end(), values.begin(), values.end());
    return vector;
  }

  template <typename T>
  std::vector<nimble::Vector<T>> prepareValues() {
    auto toVector =
        [this]<typename ValueType>(std::initializer_list<ValueType> values) {
          nimble::Vector<ValueType> vector{pool_.get()};
          vector.insert(vector.end(), values.begin(), values.end());
          return vector;
        };

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
          toVector({0.0, 0.00, 0.12}),
          toVector({-2.1, -2.1, -2.3, -2.1, -2.1}),
          toVector({dNaN0, dNaN0, dNaN1, dNaN2, dNaN0})};
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
          toVector({0.0f, 0.00f, 0.12f}),
          toVector({-2.1f, -2.1f, -2.3f, -2.1f, -2.1f}),
          toVector({fNaN0, fNaN0, fNaN1, fNaN2, fNaN2})};
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return {toVector({3, 3, 3, 1, 3})};
    } else {
      static_assert(!std::is_same_v<T, T>, "Unsupported test type");
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

#define NUM_TYPES TC(int32_t), TC(double), TC(float)

using TestTypes = ::testing::Types<NUM_TYPES>;

TYPED_TEST_CASE(MainlyConstantEncodingTest, TestTypes);

TYPED_TEST(MainlyConstantEncodingTest, serializeThenDeserialize) {
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
    auto encoding = nimble::test::
        Encoder<nimble::MainlyConstantEncoding<DataType>>::createEncoding(
            *this->buffer_,
            values,
            stringBufferFactory,
            nimble::CompressionType::Uncompressed,
            options);

    uint32_t rowCount = values.size();
    nimble::Vector<DataType> result(this->pool_.get(), rowCount);
    encoding->materialize(rowCount, result.data());

    EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::MainlyConstant);
    EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<DataType>::dataType);
    EXPECT_EQ(encoding->rowCount(), rowCount);
    for (uint32_t i = 0; i < rowCount; ++i) {
      EXPECT_TRUE(
          nimble::NimbleCompare<DataType>::equals(result[i], values[i]));
    }
  }
}

TYPED_TEST(MainlyConstantEncodingTest, slice) {
  using DataType = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  const auto values = this->template toVector<DataType>(
      {static_cast<DataType>(7),
       static_cast<DataType>(7),
       static_cast<DataType>(20),
       static_cast<DataType>(7),
       static_cast<DataType>(40),
       static_cast<DataType>(7),
       static_cast<DataType>(7),
       static_cast<DataType>(70),
       static_cast<DataType>(7),
       static_cast<DataType>(7)});
  const auto encoded =
      nimble::test::Encoder<nimble::MainlyConstantEncoding<DataType>>::encode(
          *this->buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options);

  struct Range {
    uint32_t offset;
    uint32_t length;
  };
  for (const auto range :
       {Range{/*offset=*/0, /*length=*/4},
        Range{/*offset=*/2, /*length=*/6},
        Range{/*offset=*/6, /*length=*/4}}) {
    SCOPED_TRACE(
        testing::Message() << "offset=" << range.offset
                           << ", length=" << range.length);
    nimble::Buffer sliceBuffer{*this->pool_};
    const auto sliced = nimble::MainlyConstantEncoding<DataType>::slice(
        encoded, range.offset, range.length, sliceBuffer, options);
    nimble::MainlyConstantEncoding<DataType> encoding{
        *this->pool_,
        sliced,
        [](uint32_t /*totalLength*/) -> void* { return nullptr; },
        options};

    EXPECT_EQ(encoding.encodingType(), nimble::EncodingType::MainlyConstant);
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

TYPED_TEST(MainlyConstantEncodingTest, sliceAllCommonRowsReturnsConstant) {
  using DataType = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  const auto values = this->template toVector<DataType>(
      {static_cast<DataType>(7),
       static_cast<DataType>(7),
       static_cast<DataType>(20),
       static_cast<DataType>(7),
       static_cast<DataType>(40),
       static_cast<DataType>(7)});
  const auto encoded =
      nimble::test::Encoder<nimble::MainlyConstantEncoding<DataType>>::encode(
          *this->buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options);

  nimble::Buffer sliceBuffer{*this->pool_};
  const auto sliced = nimble::MainlyConstantEncoding<DataType>::slice(
      encoded, /*offset=*/0, /*length=*/2, sliceBuffer, options);
  auto encoding = nimble::EncodingFactory{options}.create(
      *this->pool_, sliced, [](uint32_t /*totalLength*/) -> void* {
        return nullptr;
      });

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Constant);
  EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<DataType>::dataType);
  EXPECT_EQ(encoding->rowCount(), 2);

  nimble::Vector<DataType> result(this->pool_.get(), 2);
  encoding->materialize(2, result.data());
  for (uint32_t i = 0; i < result.size(); ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<DataType>::equals(result[i], values[i]));
  }
}

TYPED_TEST(MainlyConstantEncodingTest, invalidSliceRange) {
  using DataType = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  const auto values = this->template toVector<DataType>(
      {static_cast<DataType>(7),
       static_cast<DataType>(7),
       static_cast<DataType>(20),
       static_cast<DataType>(7)});
  const auto encoded =
      nimble::test::Encoder<nimble::MainlyConstantEncoding<DataType>>::encode(
          *this->buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options);

  nimble::Buffer invalidSliceBuffer{*this->pool_};
  NIMBLE_ASSERT_THROW(
      nimble::MainlyConstantEncoding<DataType>::slice(
          encoded,
          /*offset=*/0,
          /*length=*/0,
          invalidSliceBuffer,
          options),
      "");
  NIMBLE_ASSERT_THROW(
      nimble::MainlyConstantEncoding<DataType>::slice(
          encoded,
          /*offset=*/5,
          /*length=*/0,
          invalidSliceBuffer,
          options),
      "");
  NIMBLE_ASSERT_THROW(
      nimble::MainlyConstantEncoding<DataType>::slice(
          encoded,
          /*offset=*/3,
          /*length=*/2,
          invalidSliceBuffer,
          options),
      "");
}

TYPED_TEST(MainlyConstantEncodingTest, deterministicSerializationOnTie) {
  using DataType = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // 7 and 5 are tied for most-frequent (3 occurrences each). The common value
  // is chosen via std::max_element over an absl::flat_hash_map whose iteration
  // order is seeded per table, so each encode() below sees a differently-seeded
  // map. The tie must be broken by value (deterministically) so that identical
  // input always produces identical serialized bytes.
  nimble::Vector<DataType> values{this->pool_.get()};
  for (const auto value : {7, 7, 7, 5, 5, 5, 1}) {
    values.push_back(static_cast<DataType>(value));
  }
  const uint32_t rowCount = values.size();

  std::vector<velox::BufferPtr> stringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    return stringBuffers
        .emplace_back(
            velox::AlignedBuffer::allocate<char>(
                totalLength, this->pool_.get()))
        ->template asMutable<void>();
  };

  // The tie-data must still round-trip to the original values.
  auto encoding = nimble::test::
      Encoder<nimble::MainlyConstantEncoding<DataType>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);
  nimble::Vector<DataType> result(this->pool_.get(), rowCount);
  encoding->materialize(rowCount, result.data());
  for (uint32_t i = 0; i < rowCount; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<DataType>::equals(result[i], values[i]));
  }

  // Re-encoding the same input must yield byte-identical output regardless of
  // per-run hash-map iteration order.
  std::string expected;
  constexpr int kIterations = 32;
  for (int i = 0; i < kIterations; ++i) {
    nimble::Buffer buffer{*this->pool_};
    const std::string encoded{
        nimble::test::Encoder<nimble::MainlyConstantEncoding<DataType>>::encode(
            buffer, values, nimble::CompressionType::Uncompressed, options)};
    if (i == 0) {
      expected = encoded;
    } else {
      EXPECT_EQ(encoded, expected)
          << "MainlyConstant serialization is not deterministic (iteration "
          << i << ")";
    }
  }
}

TEST(MainlyConstantEncodingV1Test, materializeSparseBoolIsCommon) {
  auto pool = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
  const auto toBoolVector = [&](std::initializer_list<bool> values) {
    nimble::Vector<bool> vector{pool.get()};
    vector.insert(vector.end(), values.begin(), values.end());
    return vector;
  };
  const auto toInt32Vector = [&](std::initializer_list<int32_t> values) {
    nimble::Vector<int32_t> vector{pool.get()};
    vector.insert(vector.end(), values.begin(), values.end());
    return vector;
  };

  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useVarint=" << useVarint);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    nimble::Buffer childBuffer{*pool};
    nimble::Buffer mainlyConstantBuffer{*pool};
    std::vector<velox::BufferPtr> stringBuffers;
    const auto stringBufferFactory = [&](uint32_t totalLength) {
      return stringBuffers
          .emplace_back(
              velox::AlignedBuffer::allocate<char>(totalLength, pool.get()))
          ->template asMutable<void>();
    };

    const auto isCommon = toBoolVector(
        {true, true, false, true, false, true, true, false, true, true});
    const auto otherValues = toInt32Vector({20, 40, 70});
    const auto serializedIsCommon =
        nimble::test::Encoder<nimble::SparseBoolEncoding>::encode(
            childBuffer,
            isCommon,
            nimble::CompressionType::Uncompressed,
            options);
    const auto serializedOtherValues =
        nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
            childBuffer,
            otherValues,
            nimble::CompressionType::Uncompressed,
            options);

    const uint32_t rowCount = isCommon.size();
    const uint32_t encodingSize =
        nimble::EncodingPrefix::serializedSize(rowCount, useVarint) + 8 +
        serializedIsCommon.size() + serializedOtherValues.size() +
        sizeof(uint32_t);
    char* const reserved = mainlyConstantBuffer.reserve(encodingSize);
    char* pos = reserved;
    nimble::EncodingPrefix::serialize(
        nimble::EncodingType::MainlyConstant,
        nimble::DataType::Int32,
        rowCount,
        useVarint,
        pos);
    nimble::encoding::writeString(serializedIsCommon, pos);
    nimble::encoding::writeString(serializedOtherValues, pos);
    nimble::encoding::write<uint32_t>(7, pos);
    ASSERT_EQ(pos - reserved, encodingSize);

    nimble::MainlyConstantEncoding<int32_t> encoding{
        *pool, {reserved, encodingSize}, stringBufferFactory, options};
    nimble::Vector<int32_t> result{pool.get(), rowCount};
    encoding.materialize(rowCount, result.data());
    EXPECT_EQ(
        std::vector<int32_t>(result.begin(), result.end()),
        std::vector<int32_t>({7, 7, 20, 7, 40, 7, 7, 70, 7, 7}));

    encoding.reset();
    encoding.skip(3);
    nimble::Vector<int32_t> partial{pool.get(), 4};
    encoding.materialize(4, partial.data());
    EXPECT_EQ(
        std::vector<int32_t>(partial.begin(), partial.end()),
        std::vector<int32_t>({7, 40, 7, 7}));
  }
}
