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
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "folly/Random.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/SliceEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

#include <algorithm>
#include <array>
#include <vector>

using namespace facebook;

template <bool UseVarint>
struct SparseBoolTestConfig {
  static constexpr bool useVarint = UseVarint;
};

template <typename Config>
class SparseBoolEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  nimble::Vector<bool> toVector(std::initializer_list<bool> l) {
    nimble::Vector<bool> v{pool_.get()};
    v.insert(v.end(), l.begin(), l.end());
    return v;
  }

  std::unique_ptr<nimble::Encoding> createEncoding(
      const nimble::Vector<bool>& values,
      const nimble::Encoding::Options& options = {}) {
    return nimble::test::Encoder<nimble::SparseBoolEncoding>::createEncoding(
        *buffer_,
        values,
        stringBufferFactory(),
        nimble::CompressionType::Uncompressed,
        options);
  }

  std::function<void*(uint32_t)> stringBufferFactory() {
    return [this](uint32_t totalLength) {
      auto& buf = stringBuffers_.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
      return buf->template asMutable<void>();
    };
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
  std::vector<velox::BufferPtr> stringBuffers_;
};

using TestTypes =
    ::testing::Types<SparseBoolTestConfig<false>, SparseBoolTestConfig<true>>;

TYPED_TEST_CASE(SparseBoolEncodingTest, TestTypes);

TYPED_TEST(SparseBoolEncodingTest, allFalse) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector({false, false, false, false, false});
  auto encoding = this->createEncoding(values, options);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::SparseBool);
  EXPECT_EQ(encoding->dataType(), nimble::DataType::Bool);
  EXPECT_EQ(encoding->rowCount(), 5);

  nimble::Vector<bool> result(this->pool_.get(), 5);
  encoding->materialize(5, result.data());
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(result[i], false) << "index " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, allTrue) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector({true, true, true, true, true});
  auto encoding = this->createEncoding(values, options);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::SparseBool);
  EXPECT_EQ(encoding->rowCount(), 5);

  nimble::Vector<bool> result(this->pool_.get(), 5);
  encoding->materialize(5, result.data());
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(result[i], true) << "index " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, sparseTrue) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector(
      {false, false, true, false, false, false, false, false, false, false});
  auto encoding = this->createEncoding(values, options);

  EXPECT_EQ(encoding->rowCount(), 10);

  nimble::Vector<bool> result(this->pool_.get(), 10);
  encoding->materialize(10, result.data());
  for (uint32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, sparseFalse) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector(
      {true, true, false, true, true, true, true, true, true, true});
  auto encoding = this->createEncoding(values, options);

  EXPECT_EQ(encoding->rowCount(), 10);

  nimble::Vector<bool> result(this->pool_.get(), 10);
  encoding->materialize(10, result.data());
  for (uint32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, materializeSparseIndicesForSparseFalse) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector(
      {true, true, false, true, false, true, true, false, true, true});
  auto encoding = this->createEncoding(values, options);
  auto* sparseBool = dynamic_cast<nimble::SparseBoolEncoding*>(encoding.get());
  ASSERT_NE(sparseBool, nullptr);
  EXPECT_FALSE(sparseBool->sparseValue());

  nimble::Vector<uint32_t> positions{this->pool_.get()};
  EXPECT_EQ(sparseBool->materializeSparseIndices(5, positions), 2);
  EXPECT_EQ(
      std::vector<uint32_t>(positions.begin(), positions.end()),
      std::vector<uint32_t>({2, 4}));

  EXPECT_EQ(sparseBool->materializeSparseIndices(5, positions), 1);
  EXPECT_EQ(
      std::vector<uint32_t>(positions.begin(), positions.end()),
      std::vector<uint32_t>({2}));

  sparseBool->reset();
  EXPECT_EQ(sparseBool->skipSparseIndices(5), 2);
  EXPECT_EQ(sparseBool->materializeSparseIndices(5, positions), 1);
  EXPECT_EQ(
      std::vector<uint32_t>(positions.begin(), positions.end()),
      std::vector<uint32_t>({2}));
}

TYPED_TEST(SparseBoolEncodingTest, materializeSparseIndicesForSparseTrue) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector(
      {false, true, false, false, false, false, true, false, false, false});
  auto encoding = this->createEncoding(values, options);
  auto* sparseBool = dynamic_cast<nimble::SparseBoolEncoding*>(encoding.get());
  ASSERT_NE(sparseBool, nullptr);
  EXPECT_TRUE(sparseBool->sparseValue());

  nimble::Vector<uint32_t> positions{this->pool_.get()};
  EXPECT_EQ(sparseBool->materializeSparseIndices(10, positions), 2);
  EXPECT_EQ(
      std::vector<uint32_t>(positions.begin(), positions.end()),
      std::vector<uint32_t>({1, 6}));

  sparseBool->reset();
  EXPECT_EQ(sparseBool->skipSparseIndices(5), 1);
  EXPECT_EQ(sparseBool->materializeSparseIndices(5, positions), 1);
  EXPECT_EQ(
      std::vector<uint32_t>(positions.begin(), positions.end()),
      std::vector<uint32_t>({1}));
}

TYPED_TEST(SparseBoolEncodingTest, sparseValueReportsPositionPolarity) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto sparseFalseValues = this->toVector(
      {true, true, false, true, true, true, true, true, true, true});
  auto sparseFalseEncoding = this->createEncoding(sparseFalseValues, options);
  auto* sparseFalse =
      dynamic_cast<nimble::SparseBoolEncoding*>(sparseFalseEncoding.get());
  ASSERT_NE(sparseFalse, nullptr);
  EXPECT_FALSE(sparseFalse->sparseValue());

  auto sparseTrueValues = this->toVector(
      {false, false, true, false, false, false, false, false, false, false});
  auto sparseTrueEncoding = this->createEncoding(sparseTrueValues, options);
  auto* sparseTrue =
      dynamic_cast<nimble::SparseBoolEncoding*>(sparseTrueEncoding.get());
  ASSERT_NE(sparseTrue, nullptr);
  EXPECT_TRUE(sparseTrue->sparseValue());
}

TYPED_TEST(SparseBoolEncodingTest, singleElement) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  for (bool val : {true, false}) {
    nimble::Vector<bool> values{this->pool_.get()};
    values.push_back(val);
    auto encoding = this->createEncoding(values, options);

    EXPECT_EQ(encoding->rowCount(), 1);

    nimble::Vector<bool> result(this->pool_.get(), 1);
    encoding->materialize(1, result.data());
    EXPECT_EQ(result[0], val);
  }
}

TYPED_TEST(SparseBoolEncodingTest, skipThenMaterialize) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector(
      {false, false, true, false, true, false, false, true, false, false});
  auto encoding = this->createEncoding(values, options);

  encoding->skip(3);

  nimble::Vector<bool> result(this->pool_.get(), 4);
  encoding->materialize(4, result.data());
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(result[i], values[i + 3]) << "index " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, slice) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  for (const auto& values :
       {this->toVector(
            {false,
             false,
             true,
             false,
             true,
             false,
             false,
             true,
             false,
             false}),
        this->toVector(
            {true, true, false, true, false, true, true, false, true, true})}) {
    const auto encoded =
        nimble::test::Encoder<nimble::SparseBoolEncoding>::encode(
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
          Range{/*offset=*/7, /*length=*/3}}) {
      SCOPED_TRACE(
          testing::Message()
          << "offset=" << range.offset << ", length=" << range.length);
      nimble::Buffer sliceBuffer{*this->pool_};
      const auto sliced = nimble::SparseBoolEncoding::slice(
          encoded, range.offset, range.length, sliceBuffer, options);
      nimble::SparseBoolEncoding encoding{
          *this->pool_,
          sliced,
          [](uint32_t /*totalLength*/) -> void* { return nullptr; },
          options};

      EXPECT_EQ(encoding.encodingType(), nimble::EncodingType::SparseBool);
      EXPECT_EQ(encoding.dataType(), nimble::DataType::Bool);
      EXPECT_EQ(encoding.rowCount(), range.length);
      nimble::Vector<bool> result(this->pool_.get(), range.length);
      encoding.materialize(range.length, result.data());
      for (uint32_t i = 0; i < range.length; ++i) {
        EXPECT_EQ(result[i], values[range.offset + i]) << "index " << i;
      }
    }
  }
}

TYPED_TEST(SparseBoolEncodingTest, sliceAndCount) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  struct TestCase {
    const char* name;
    nimble::Vector<bool> values;
  };
  struct Range {
    uint32_t offset;
    uint32_t length;
  };

  const std::vector<TestCase> testCases{
      {
          .name = "allFalse",
          .values = this->toVector(
              {false,
               false,
               false,
               false,
               false,
               false,
               false,
               false,
               false,
               false}),
      },
      {
          .name = "allTrue",
          .values = this->toVector(
              {true, true, true, true, true, true, true, true, true, true}),
      },
      {
          .name = "sparseTrue",
          .values = this->toVector(
              {false,
               false,
               true,
               false,
               true,
               false,
               false,
               true,
               false,
               false}),
      },
      {
          .name = "sparseFalse",
          .values = this->toVector(
              {true, true, false, true, false, true, true, false, true, true}),
      },
      {
          .name = "alternating",
          .values = this->toVector(
              {true,
               false,
               true,
               false,
               true,
               false,
               true,
               false,
               true,
               false}),
      },
  };
  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    const auto& values = testCase.values;
    const auto encoded =
        nimble::test::Encoder<nimble::SparseBoolEncoding>::encode(
            *this->buffer_,
            values,
            nimble::CompressionType::Uncompressed,
            options);

    for (const auto range :
         {Range{/*offset=*/0, /*length=*/1},
          Range{/*offset=*/0, /*length=*/10},
          Range{/*offset=*/1, /*length=*/3},
          Range{/*offset=*/2, /*length=*/1},
          Range{/*offset=*/5, /*length=*/2},
          Range{/*offset=*/9, /*length=*/1}}) {
      SCOPED_TRACE(
          testing::Message()
          << "offset=" << range.offset << ", length=" << range.length);
      const auto expectedBefore = static_cast<uint32_t>(
          std::count(values.begin(), values.begin() + range.offset, true));
      const auto expected = static_cast<uint32_t>(std::count(
          values.begin() + range.offset,
          values.begin() + range.offset + range.length,
          true));

      nimble::Buffer sliceBuffer{*this->pool_};
      const auto sliceResult = nimble::SparseBoolEncoding::sliceAndCount(
          encoded, range.offset, range.length, sliceBuffer, options);
      EXPECT_EQ(sliceResult.counts.numTrueBeforeRange, expectedBefore);
      EXPECT_EQ(sliceResult.counts.numTrueInRange, expected);

      nimble::SparseBoolEncoding slicedEncoding{
          *this->pool_,
          sliceResult.sliced,
          [](uint32_t /*totalLength*/) -> void* { return nullptr; },
          options};
      EXPECT_EQ(slicedEncoding.rowCount(), range.length);
      nimble::Vector<bool> result(this->pool_.get(), range.length);
      slicedEncoding.materialize(range.length, result.data());
      for (uint32_t i = 0; i < range.length; ++i) {
        EXPECT_EQ(result[i], values[range.offset + i]) << "index " << i;
      }
    }
  }
}

TYPED_TEST(SparseBoolEncodingTest, countTrue) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  for (const auto& values :
       {this->toVector(
            {false,
             false,
             true,
             false,
             true,
             false,
             false,
             true,
             false,
             false}),
        this->toVector(
            {true, true, false, true, false, true, true, false, true, true})}) {
    const auto encoded =
        nimble::test::Encoder<nimble::SparseBoolEncoding>::encode(
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
          Range{/*offset=*/5, /*length=*/2},
          Range{/*offset=*/7, /*length=*/3}}) {
      SCOPED_TRACE(
          testing::Message()
          << "offset=" << range.offset << ", length=" << range.length);
      const auto expected = static_cast<uint32_t>(std::count(
          values.begin() + range.offset,
          values.begin() + range.offset + range.length,
          true));
      const auto expectedBefore = static_cast<uint32_t>(
          std::count(values.begin(), values.begin() + range.offset, true));
      EXPECT_EQ(
          nimble::SparseBoolEncoding::countTrue(
              encoded, range.offset, range.length, this->pool_.get(), options),
          expected);
      nimble::SparseBoolEncoding::RangeCounts counts;
      nimble::SparseBoolEncoding::countTrue(
          encoded,
          range.offset,
          range.length,
          this->pool_.get(),
          counts,
          options);
      EXPECT_EQ(counts.numTrueBeforeRange, expectedBefore);
      EXPECT_EQ(counts.numTrueInRange, expected);
    }
  }
}

TYPED_TEST(SparseBoolEncodingTest, estimateSize) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  const std::array<bool, 10> sparseTrueValues{
      false, false, true, false, true, false, false, true, false, false};
  const auto sparseTrueStats =
      nimble::Statistics<bool>::create(sparseTrueValues);
  EXPECT_EQ(
      nimble::SparseBoolEncoding::estimateSize(
          sparseTrueValues.size(), sparseTrueStats, options),
      nimble::SparseBoolEncoding::estimateSize(
          sparseTrueValues.size(), /*exceptionCount=*/3, options));

  const std::array<bool, 10> sparseFalseValues{
      true, true, false, true, false, true, true, false, true, true};
  const auto sparseFalseStats =
      nimble::Statistics<bool>::create(sparseFalseValues);
  EXPECT_EQ(
      nimble::SparseBoolEncoding::estimateSize(
          sparseFalseValues.size(), sparseFalseStats, options),
      nimble::SparseBoolEncoding::estimateSize(
          sparseFalseValues.size(), /*exceptionCount=*/3, options));
}

TYPED_TEST(SparseBoolEncodingTest, invalidCountTrueRange) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  const auto values = this->toVector(
      {false, false, true, false, true, false, false, true, false, false});
  const auto encoded =
      nimble::test::Encoder<nimble::SparseBoolEncoding>::encode(
          *this->buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options);

  auto expectZeroLengthRange = [&](uint32_t offset) {
    SCOPED_TRACE(testing::Message() << "offset=" << offset);
    NIMBLE_ASSERT_THROW(
        nimble::SparseBoolEncoding::countTrue(
            encoded,
            offset,
            /*length=*/0,
            this->pool_.get(),
            options),
        "Cannot count zero rows.");
    nimble::SparseBoolEncoding::RangeCounts counts;
    NIMBLE_ASSERT_THROW(
        nimble::SparseBoolEncoding::countTrue(
            encoded,
            offset,
            /*length=*/0,
            this->pool_.get(),
            counts,
            options),
        "Cannot count zero rows.");
  };
  expectZeroLengthRange(/*offset=*/0);
  expectZeroLengthRange(/*offset=*/4);

  auto expectInvalidRange = [&](uint32_t offset, uint32_t length) {
    SCOPED_TRACE(
        testing::Message() << "offset=" << offset << ", length=" << length);
    NIMBLE_ASSERT_THROW(
        nimble::SparseBoolEncoding::countTrue(
            encoded, offset, length, this->pool_.get(), options),
        "");
    nimble::SparseBoolEncoding::RangeCounts counts;
    NIMBLE_ASSERT_THROW(
        nimble::SparseBoolEncoding::countTrue(
            encoded, offset, length, this->pool_.get(), counts, options),
        "");
  };
  expectInvalidRange(/*offset=*/11, /*length=*/0);
  expectInvalidRange(/*offset=*/9, /*length=*/2);

  NIMBLE_ASSERT_THROW(
      nimble::SparseBoolEncoding::countTrue(
          encoded,
          /*offset=*/0,
          /*length=*/1,
          /*pool=*/nullptr,
          options),
      "Memory pool cannot be null");
  nimble::SparseBoolEncoding::RangeCounts counts;
  NIMBLE_ASSERT_THROW(
      nimble::SparseBoolEncoding::countTrue(
          encoded,
          /*offset=*/0,
          /*length=*/1,
          /*pool=*/nullptr,
          counts,
          options),
      "Memory pool cannot be null");
}

TYPED_TEST(SparseBoolEncodingTest, invalidSliceRange) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector(
      {false, false, true, false, true, false, false, true, false, false});
  const auto encoded =
      nimble::test::Encoder<nimble::SparseBoolEncoding>::encode(
          *this->buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options);

  nimble::Buffer invalidSliceBuffer{*this->pool_};
  NIMBLE_ASSERT_THROW(
      nimble::SparseBoolEncoding::slice(
          encoded,
          /*offset=*/0,
          /*length=*/0,
          invalidSliceBuffer,
          options),
      "");
  NIMBLE_ASSERT_THROW(
      nimble::SparseBoolEncoding::sliceAndCount(
          encoded,
          /*offset=*/0,
          /*length=*/0,
          invalidSliceBuffer,
          options),
      "Cannot slice zero rows.");
  NIMBLE_ASSERT_THROW(
      nimble::SparseBoolEncoding::slice(
          encoded,
          /*offset=*/11,
          /*length=*/0,
          invalidSliceBuffer,
          options),
      "");
  NIMBLE_ASSERT_THROW(
      nimble::SparseBoolEncoding::sliceAndCount(
          encoded,
          /*offset=*/11,
          /*length=*/0,
          invalidSliceBuffer,
          options),
      "");
  NIMBLE_ASSERT_THROW(
      nimble::SparseBoolEncoding::slice(
          encoded,
          /*offset=*/9,
          /*length=*/2,
          invalidSliceBuffer,
          options),
      "");
  NIMBLE_ASSERT_THROW(
      nimble::SparseBoolEncoding::sliceAndCount(
          encoded,
          /*offset=*/9,
          /*length=*/2,
          invalidSliceBuffer,
          options),
      "");
}

TYPED_TEST(SparseBoolEncodingTest, resetAndRematerialize) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector({true, false, true, false, true});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<bool> result1(this->pool_.get(), 5);
  encoding->materialize(5, result1.data());

  encoding->reset();

  nimble::Vector<bool> result2(this->pool_.get(), 5);
  encoding->materialize(5, result2.data());

  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(result1[i], result2[i]) << "index " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, materializeZeroRows) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector({true, false, true});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<bool> result(this->pool_.get(), 3);
  encoding->materialize(0, result.data());

  encoding->materialize(3, result.data());
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, materializeBoolsAsBits) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector(
      {false, false, true, false, true, false, false, true, false, false});
  auto encoding = this->createEncoding(values, options);

  uint64_t buffer[1] = {0};
  auto* typedEncoding =
      dynamic_cast<nimble::SparseBoolEncoding*>(encoding.get());
  ASSERT_NE(typedEncoding, nullptr);
  typedEncoding->materializeBoolsAsBits(10, buffer, 0);

  for (uint32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(velox::bits::isBitSet(buffer, i), values[i]) << "index " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, materializeBoolsAsBitsWithOffset) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector({true, false, true, false});
  auto encoding = this->createEncoding(values, options);

  uint64_t buffer[1] = {0};
  auto* typedEncoding =
      dynamic_cast<nimble::SparseBoolEncoding*>(encoding.get());
  ASSERT_NE(typedEncoding, nullptr);
  typedEncoding->materializeBoolsAsBits(4, buffer, 3);

  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(velox::bits::isBitSet(buffer, i + 3), values[i]) << "index " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, incrementalMaterialize) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector(
      {true, false, true, false, true, false, true, false, true, false});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<bool> result(this->pool_.get(), 10);
  encoding->materialize(3, result.data());
  encoding->materialize(4, result.data() + 3);
  encoding->materialize(3, result.data() + 7);

  for (uint32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, fuzz) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (int trial = 0; trial < 20; ++trial) {
    uint32_t rowCount = folly::Random::rand32(1, 500, rng);
    double sparsity = folly::Random::randDouble01(rng) * 0.1;

    nimble::Vector<bool> values{this->pool_.get()};
    for (uint32_t i = 0; i < rowCount; ++i) {
      values.push_back(folly::Random::randDouble01(rng) < sparsity);
    }

    auto encoding = this->createEncoding(values, options);

    EXPECT_EQ(encoding->rowCount(), rowCount);

    nimble::Vector<bool> result(this->pool_.get(), rowCount);
    encoding->materialize(rowCount, result.data());

    for (uint32_t i = 0; i < rowCount; ++i) {
      EXPECT_EQ(result[i], values[i]) << "trial " << trial << " index " << i;
    }
  }
}

TYPED_TEST(SparseBoolEncodingTest, withZstdCompression) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector({false, false, true,  false, false, false, false,
                                false, true,  false, false, false, false, false,
                                false, false, false, false, false, false});

  auto encoding =
      nimble::test::Encoder<nimble::SparseBoolEncoding>::createEncoding(
          *this->buffer_,
          values,
          this->stringBufferFactory(),
          nimble::CompressionType::Zstd,
          options);

  nimble::Vector<bool> result(this->pool_.get(), values.size());
  encoding->materialize(values.size(), result.data());
  for (uint32_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, largeDataset) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  nimble::Vector<bool> values{this->pool_.get()};
  for (uint32_t i = 0; i < 10000; ++i) {
    values.push_back(i % 100 == 0);
  }

  auto encoding = this->createEncoding(values, options);
  EXPECT_EQ(encoding->rowCount(), 10000);

  nimble::Vector<bool> result(this->pool_.get(), 10000);
  encoding->materialize(10000, result.data());
  for (uint32_t i = 0; i < 10000; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, alternatingPattern) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  nimble::Vector<bool> values{this->pool_.get()};
  for (uint32_t i = 0; i < 200; ++i) {
    values.push_back(i % 2 == 0);
  }

  auto encoding = this->createEncoding(values, options);

  nimble::Vector<bool> result(this->pool_.get(), 200);
  encoding->materialize(200, result.data());
  for (uint32_t i = 0; i < 200; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, skipAllThenMaterializeRemaining) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector(
      {false, false, true, false, true, false, false, true, false, true});
  auto encoding = this->createEncoding(values, options);

  encoding->skip(8);

  nimble::Vector<bool> result(this->pool_.get(), 2);
  encoding->materialize(2, result.data());
  EXPECT_EQ(result[0], values[8]);
  EXPECT_EQ(result[1], values[9]);
}

// --- Slice end-to-end coverage across position child shapes ---------------
//
// The slice path always emits a SliceEncoding wrapper around the sliced
// position sub-stream. Its inner is a fresh re-emit of the retained
// positions through the source's captured encoding layout -- Trivial in,
// Trivial out; Varint in, Varint out; and so on. When the captured layout
// supports SliceEncoding push-down (Trivial, FixedBitWidth, ...), the
// -offset shift is folded into the inner bytes at write time and the
// on-wire delta is zero. Otherwise the delta rides on the SliceEncoding
// wire and the read-time shift loop applies it.

// Peels the SliceEncoding wrapper the slice path always emits and returns
// its inner encoding type plus the on-wire value delta.
struct SliceInnerInfo {
  nimble::EncodingType outerType;
  nimble::EncodingType innerType;
  int64_t valueDelta;
};

static SliceInnerInfo sparseBoolPositionSliceInfo(
    std::string_view encoded,
    const nimble::Encoding::Options& options,
    velox::memory::MemoryPool& pool) {
  const auto payload = encoded.substr(
      nimble::EncodingPrefix::prefixSize(encoded, options.useVarintRowCount));
  const auto positionEnc = payload.substr(sizeof(uint8_t));
  const auto outerType = nimble::EncodingPrefix::encodingType(positionEnc);
  if (outerType != nimble::EncodingType::Slice) {
    return {.outerType = outerType, .innerType = outerType, .valueDelta = 0};
  }
  // Constructing a SliceEncoding<uint32_t> parses the wire layout through the
  // same code path production readers use; the accessors expose the parsed
  // fields directly.
  nimble::SliceEncoding<uint32_t> slice{pool, positionEnc, nullptr, options};
  return {
      .outerType = outerType,
      .innerType = nimble::EncodingPrefix::encodingType(slice.innerEncoding()),
      .valueDelta = slice.valueDelta(),
  };
}

TYPED_TEST(SparseBoolEncodingTest, sliceWithTrivialPositionsFallback) {
  // Default nested selection makes the sparse positions Trivial. The slice
  // path re-emits the retained positions through the source's captured
  // layout, which produces another Trivial<uint32>, and lets SliceEncoding's
  // Trivial push-down fold the -offset shift into the byte copy -- delta on
  // the wire is zero.
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector(
      {false, false, true, false, true, false, true, true, false, false});
  const auto encoded =
      nimble::test::Encoder<nimble::SparseBoolEncoding>::encode(
          *this->buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options);

  nimble::Buffer sliceBuffer{*this->pool_};
  const auto sliced = nimble::SparseBoolEncoding::slice(
      encoded, /*offset=*/2, /*length=*/6, sliceBuffer, options);
  const auto info = sparseBoolPositionSliceInfo(sliced, options, *this->pool_);
  EXPECT_EQ(info.outerType, nimble::EncodingType::Slice);
  EXPECT_EQ(info.innerType, nimble::EncodingType::Trivial);
  EXPECT_EQ(info.valueDelta, 0);
  nimble::SparseBoolEncoding encoding{
      *this->pool_,
      sliced,
      [](uint32_t /*totalLength*/) -> void* { return nullptr; },
      options};
  ASSERT_EQ(encoding.rowCount(), 6);
  nimble::Vector<bool> result(this->pool_.get(), 6);
  encoding.materialize(6, result.data());
  for (uint32_t i = 0; i < 6; ++i) {
    EXPECT_EQ(result[i], values[2 + i]) << "row " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, sliceWithVarintPositionsFallback) {
  // Force the sparse-positions child to Varint, which has no EncodingView and
  // no SliceEncoding push-down folder. The slice path falls into
  // detail::findSortedPositionSlots' materialize + binary-search branch and
  // re-emits the retained range through the source's captured layout --
  // another Varint<uint32>. SliceEncoding::wrap cannot fold the shift into a
  // Varint inner, so the -offset delta rides on the wire and the read-time
  // shift loop applies it.
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector(
      {false, false, true, false, true, false, true, true, false, false});

  // Replay a SparseBool layout whose sole child (Indices) is Varint. The
  // ReplayedEncodingSelectionPolicy routes the nested encoding through the
  // child's layout instead of the default read-factor selector.
  nimble::ManualEncodingSelectionPolicyFactory manualFactory;
  nimble::EncodingSelectionPolicyCreator creator =
      [&manualFactory](nimble::DataType dataType) {
        return manualFactory.createPolicy(dataType);
      };
  nimble::EncodingLayout varintLayout{
      nimble::EncodingType::Varint, {}, nimble::CompressionType::Uncompressed};
  nimble::EncodingLayout layout{
      nimble::EncodingType::SparseBool,
      {},
      nimble::CompressionType::Uncompressed,
      {varintLayout}};
  auto policy = std::make_unique<nimble::ReplayedEncodingSelectionPolicy<bool>>(
      std::move(layout), nimble::CompressionOptions{}, creator);

  const auto encoded = nimble::EncodingFactory::encode<bool>(
      std::move(policy),
      std::span<const bool>{values.data(), values.size()},
      *this->buffer_,
      options);

  // Verify the source really has a Varint positions child so the fallback
  // branch is exercised.
  const auto sourcePayload = encoded.substr(
      nimble::EncodingPrefix::prefixSize(encoded, options.useVarintRowCount));
  const auto sourcePositionEnc = sourcePayload.substr(sizeof(uint8_t));
  ASSERT_EQ(
      nimble::EncodingPrefix::encodingType(sourcePositionEnc),
      nimble::EncodingType::Varint);

  constexpr uint32_t kOffset = 2;
  constexpr uint32_t kLength = 6;
  nimble::Buffer sliceBuffer{*this->pool_};
  const auto sliced = nimble::SparseBoolEncoding::slice(
      encoded, kOffset, kLength, sliceBuffer, options);
  const auto info = sparseBoolPositionSliceInfo(sliced, options, *this->pool_);
  EXPECT_EQ(info.outerType, nimble::EncodingType::Slice);
  // Captured layout preserves the source's Varint shape.
  EXPECT_EQ(info.innerType, nimble::EncodingType::Varint);
  // Varint has no push-down folder, so the -offset shift lands on the wire.
  EXPECT_EQ(info.valueDelta, -static_cast<int64_t>(kOffset));
  nimble::SparseBoolEncoding encoding{
      *this->pool_,
      sliced,
      [](uint32_t /*totalLength*/) -> void* { return nullptr; },
      options};
  ASSERT_EQ(encoding.rowCount(), kLength);
  nimble::Vector<bool> result(this->pool_.get(), kLength);
  encoding.materialize(kLength, result.data());
  for (uint32_t i = 0; i < kLength; ++i) {
    EXPECT_EQ(result[i], values[kOffset + i]) << "row " << i;
  }
}

TYPED_TEST(SparseBoolEncodingTest, sliceRoundTripAtBoundaries) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};
  auto values = this->toVector(
      {false, false, true, false, true, false, false, true, false, true});
  const auto encoded =
      nimble::test::Encoder<nimble::SparseBoolEncoding>::encode(
          *this->buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options,
          /*realNestedSelection=*/true);

  struct Range {
    const char* name;
    uint32_t offset;
    uint32_t length;
  };
  for (const auto range :
       {Range{"singleRowAtStart", 0, 1},
        Range{"fullRange", 0, 10},
        Range{"singleRowAtEnd", 9, 1},
        Range{"singleRowInside", 4, 1}}) {
    SCOPED_TRACE(range.name);
    nimble::Buffer sliceBuffer{*this->pool_};
    const auto sliced = nimble::SparseBoolEncoding::slice(
        encoded, range.offset, range.length, sliceBuffer, options);
    nimble::SparseBoolEncoding encoding{
        *this->pool_,
        sliced,
        [](uint32_t /*totalLength*/) -> void* { return nullptr; },
        options};
    ASSERT_EQ(encoding.rowCount(), range.length);
    nimble::Vector<bool> result(this->pool_.get(), range.length);
    encoding.materialize(range.length, result.data());
    for (uint32_t i = 0; i < range.length; ++i) {
      EXPECT_EQ(result[i], values[range.offset + i]) << "row " << i;
    }
  }
}

TYPED_TEST(SparseBoolEncodingTest, mainlyConstantSliceUsesIsCommonSparseBool) {
  // MainlyConstantEncodingBase::slice reaches SparseBoolEncoding::sliceAndCount
  // for every projected MainlyConstant stream whose isCommon child is
  // SparseBool. This is the hot path the fast-slice work was designed for.
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Build a MainlyConstant stream: 4096 rows with the common value 42
  // interleaved with rare non-common values every 47 rows.
  nimble::Vector<int32_t> values{this->pool_.get()};
  for (uint32_t row = 0; row < 4096; ++row) {
    values.push_back(row % 47 == 0 ? static_cast<int32_t>(1000 + row) : 42);
  }
  const auto encoded =
      nimble::test::Encoder<nimble::MainlyConstantEncoding<int32_t>>::encode(
          *this->buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options,
          /*realNestedSelection=*/true);

  struct Range {
    uint32_t offset;
    uint32_t length;
  };
  for (const auto range :
       {Range{/*offset=*/0, /*length=*/1},
        Range{/*offset=*/0, /*length=*/4096},
        Range{/*offset=*/46, /*length=*/3},
        Range{/*offset=*/47, /*length=*/1},
        Range{/*offset=*/94, /*length=*/100},
        Range{/*offset=*/1000, /*length=*/1000},
        Range{/*offset=*/4095, /*length=*/1}}) {
    SCOPED_TRACE(
        testing::Message() << "offset=" << range.offset
                           << ", length=" << range.length);
    nimble::Buffer sliceBuffer{*this->pool_};
    const auto sliced = nimble::MainlyConstantEncoding<int32_t>::slice(
        encoded, range.offset, range.length, sliceBuffer, options);
    auto encoding = nimble::EncodingFactory{options}.create(
        *this->pool_, sliced, [](uint32_t /*size*/) -> void* {
          return nullptr;
        });
    ASSERT_EQ(encoding->rowCount(), range.length);
    nimble::Vector<int32_t> result(this->pool_.get(), range.length);
    encoding->materialize(range.length, result.data());
    for (uint32_t i = 0; i < range.length; ++i) {
      EXPECT_EQ(result[i], values[range.offset + i]) << "row " << i;
    }
  }
}
