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
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
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
class RleEncodingTest;

// Helper to prepare values - must be at namespace scope
template <typename T, typename TestClass>
struct RleValuesPreparer {
  static std::vector<nimble::Vector<T>> prepareValues(TestClass* test) {
    FAIL() << "unspecialized prepareValues() should not be called";
    return {};
  }
};

template <typename TestClass>
struct RleValuesPreparer<double, TestClass> {
  static std::vector<nimble::Vector<double>> prepareValues(TestClass* test) {
    return {
        test->toVector({0.0, -0.0}),
        test->toVector(
            {-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0}),
        test->toVector(
            {-0.0, -0.0, -0.0, +0.0, -0.0, +0.0, -0.0, -0.0, -0.0, -0.0}),
        test->toVector(
            {-2.1, -0.0, -0.0, 3.54, 9.87, -0.0, -0.0, -0.0, -0.0, 10.6}),
        test->toVector(
            {0.00, 1.11, 2.22, 3.33, 4.44, 5.55, 6.66, 7.77, 8.88, 9.99}),
        test->toVector(
            {test->dNaN0,
             test->dNaN0,
             test->dNaN0,
             test->dNaN1,
             test->dNaN1,
             test->dNaN2,
             test->dNaN3,
             test->dNaN3,
             test->dNaN0})};
  }
};

template <typename TestClass>
struct RleValuesPreparer<float, TestClass> {
  static std::vector<nimble::Vector<float>> prepareValues(TestClass* test) {
    return {
        test->toVector({0.0f, -0.0f}),
        test->toVector(
            {-0.0f,
             -0.0f,
             -0.0f,
             -0.0f,
             -0.0f,
             -0.0f,
             -0.0f,
             -0.0f,
             -0.0f,
             -0.0f}),
        test->toVector(
            {-0.0f,
             -0.0f,
             -0.0f,
             +0.0f,
             -0.0f,
             +0.0f,
             -0.0f,
             -0.0f,
             -0.0f,
             -0.0f}),
        test->toVector(
            {-2.1f,
             -0.0f,
             -0.0f,
             3.54f,
             9.87f,
             -0.0f,
             -0.0f,
             -0.0f,
             -0.0f,
             10.6f}),
        test->toVector(
            {0.00f,
             1.11f,
             2.22f,
             3.33f,
             4.44f,
             5.55f,
             6.66f,
             7.77f,
             8.88f,
             9.99f}),
        test->toVector(
            {test->fNaN0,
             test->fNaN0,
             test->fNaN0,
             test->fNaN1,
             test->fNaN1,
             test->fNaN2,
             test->fNaN3,
             test->fNaN3,
             test->fNaN0})};
  }
};

template <typename TestClass>
struct RleValuesPreparer<int32_t, TestClass> {
  static std::vector<nimble::Vector<int32_t>> prepareValues(TestClass* test) {
    return {
        test->toVector({2, 3, 3}),
        test->toVector({1, 2, 2, 3, 3, 3, 4, 4, 4, 4})};
  }
};

template <typename Config>
class RleEncodingTest : public ::testing::Test {
 protected:
  // Make helper templates friends so they can access protected members
  template <typename T, typename TestClass>
  friend struct RleValuesPreparer;

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
    if constexpr (std::is_same_v<T, double>) {
      const double dNaN0 = std::numeric_limits<double>::quiet_NaN();
      const double dNaN1 = std::numeric_limits<double>::signaling_NaN();
      const double dNaN2 =
          nimble::EncodingPhysicalType<double>::asEncodingLogicalType(
              (nimble::EncodingPhysicalType<double>::asEncodingPhysicalType(
                   dNaN0) |
               0x3));
      const double dNaN3 =
          nimble::EncodingPhysicalType<double>::asEncodingLogicalType(
              (nimble::EncodingPhysicalType<double>::asEncodingPhysicalType(
                   dNaN0) |
               0x5));
      return {
          toVector({0.0, -0.0}),
          toVector(
              {-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0}),
          toVector(
              {-0.0, -0.0, -0.0, +0.0, -0.0, +0.0, -0.0, -0.0, -0.0, -0.0}),
          toVector(
              {-2.1, -0.0, -0.0, 3.54, 9.87, -0.0, -0.0, -0.0, -0.0, 10.6}),
          toVector(
              {0.00, 1.11, 2.22, 3.33, 4.44, 5.55, 6.66, 7.77, 8.88, 9.99}),
          toVector(
              {dNaN0, dNaN0, dNaN0, dNaN1, dNaN1, dNaN2, dNaN3, dNaN3, dNaN0})};
    } else if constexpr (std::is_same_v<T, float>) {
      const float fNaN0 = std::numeric_limits<float>::quiet_NaN();
      const float fNaN1 = std::numeric_limits<float>::signaling_NaN();
      const float fNaN2 =
          nimble::EncodingPhysicalType<float>::asEncodingLogicalType(
              (nimble::EncodingPhysicalType<float>::asEncodingPhysicalType(
                   fNaN0) |
               0x3));
      const float fNaN3 =
          nimble::EncodingPhysicalType<float>::asEncodingLogicalType(
              (nimble::EncodingPhysicalType<float>::asEncodingPhysicalType(
                   fNaN0) |
               0x5));
      return {
          toVector({0.0f, -0.0f}),
          toVector(
              {-0.0f,
               -0.0f,
               -0.0f,
               -0.0f,
               -0.0f,
               -0.0f,
               -0.0f,
               -0.0f,
               -0.0f,
               -0.0f}),
          toVector(
              {-0.0f,
               -0.0f,
               -0.0f,
               +0.0f,
               -0.0f,
               +0.0f,
               -0.0f,
               -0.0f,
               -0.0f,
               -0.0f}),
          toVector(
              {-2.1f,
               -0.0f,
               -0.0f,
               3.54f,
               9.87f,
               -0.0f,
               -0.0f,
               -0.0f,
               -0.0f,
               10.6f}),
          toVector(
              {0.00f,
               1.11f,
               2.22f,
               3.33f,
               4.44f,
               5.55f,
               6.66f,
               7.77f,
               8.88f,
               9.99f}),
          toVector(
              {fNaN0, fNaN0, fNaN0, fNaN1, fNaN1, fNaN2, fNaN3, fNaN3, fNaN0})};
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return {toVector({2, 3, 3}), toVector({1, 2, 2, 3, 3, 3, 4, 4, 4, 4})};
    } else {
      static_assert(!std::is_same_v<T, T>, "Unsupported test type");
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

#define NUM_TYPES TC(int32_t), TC(double), TC(float)

using TestTypes = ::testing::Types<NUM_TYPES>;

TYPED_TEST_CASE(RleEncodingTest, TestTypes);

TYPED_TEST(RleEncodingTest, serializeThenDeserialize) {
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
    auto encoding =
        nimble::test::Encoder<nimble::RLEEncoding<DataType>>::createEncoding(
            *this->buffer_,
            values,
            stringBufferFactory,
            nimble::CompressionType::Uncompressed,
            options);
    uint32_t rowCount = values.size();
    nimble::Vector<DataType> result(this->pool_.get(), rowCount);
    encoding->materialize(rowCount, result.data());

    EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::RLE);
    EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<DataType>::dataType);
    EXPECT_EQ(encoding->rowCount(), rowCount);
    for (uint32_t i = 0; i < rowCount; ++i) {
      EXPECT_TRUE(
          nimble::NimbleCompare<DataType>::equals(result[i], values[i]));
    }
  }
}

TYPED_TEST(RleEncodingTest, slice) {
  using DataType = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  const auto values = this->template toVector<DataType>(
      {static_cast<DataType>(1),
       static_cast<DataType>(1),
       static_cast<DataType>(1),
       static_cast<DataType>(2),
       static_cast<DataType>(2),
       static_cast<DataType>(3),
       static_cast<DataType>(3),
       static_cast<DataType>(3),
       static_cast<DataType>(4),
       static_cast<DataType>(4)});
  const auto encoded =
      nimble::test::Encoder<nimble::RLEEncoding<DataType>>::encode(
          *this->buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options);

  struct Range {
    const char* name;
    uint32_t offset;
    uint32_t length;
  };
  for (const auto range :
       {Range{/*name=*/"allRunsNoPartial",
              /*offset=*/0,
              /*length=*/10},
        Range{/*name=*/"singleRunNoPartial",
              /*offset=*/0,
              /*length=*/3},
        Range{/*name=*/"middleRunsNoPartial",
              /*offset=*/3,
              /*length=*/5},
        Range{/*name=*/"lastRunNoPartial", /*offset=*/8, /*length=*/2},
        Range{/*name=*/"firstRunPartialLastExact",
              /*offset=*/1,
              /*length=*/7},
        Range{/*name=*/"firstExactLastRunPartial",
              /*offset=*/3,
              /*length=*/4},
        Range{/*name=*/"bothPartialSameRun", /*offset=*/1, /*length=*/1},
        Range{/*name=*/"bothPartialWithInteriorRun",
              /*offset=*/1,
              /*length=*/5}}) {
    SCOPED_TRACE(
        testing::Message() << "case=" << range.name << ", offset="
                           << range.offset << ", length=" << range.length);
    nimble::Buffer sliceBuffer{*this->pool_};
    const auto sliced = nimble::RLEEncoding<DataType>::slice(
        encoded, range.offset, range.length, sliceBuffer, options);
    nimble::RLEEncoding<DataType> encoding{
        *this->pool_,
        sliced,
        [](uint32_t /*totalLength*/) -> void* { return nullptr; },
        options};

    EXPECT_EQ(encoding.encodingType(), nimble::EncodingType::RLE);
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

TYPED_TEST(RleEncodingTest, invalidSliceRange) {
  using DataType = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  const auto values = this->template toVector<DataType>(
      {static_cast<DataType>(1),
       static_cast<DataType>(1),
       static_cast<DataType>(2),
       static_cast<DataType>(2)});
  const auto encoded =
      nimble::test::Encoder<nimble::RLEEncoding<DataType>>::encode(
          *this->buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options);

  nimble::Buffer invalidSliceBuffer{*this->pool_};
  NIMBLE_ASSERT_THROW(
      nimble::RLEEncoding<DataType>::slice(
          encoded,
          /*offset=*/0,
          /*length=*/0,
          invalidSliceBuffer,
          options),
      "");
  NIMBLE_ASSERT_THROW(
      nimble::RLEEncoding<DataType>::slice(
          encoded,
          /*offset=*/5,
          /*length=*/0,
          invalidSliceBuffer,
          options),
      "");
  NIMBLE_ASSERT_THROW(
      nimble::RLEEncoding<DataType>::slice(
          encoded,
          /*offset=*/3,
          /*length=*/2,
          invalidSliceBuffer,
          options),
      "");
}
