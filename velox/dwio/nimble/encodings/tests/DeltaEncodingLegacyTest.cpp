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
#include <gtest/gtest.h>
#include <limits>

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/legacy/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

using namespace ::facebook;

namespace facebook::nimble::test {

namespace {

class TestCompressPolicy : public CompressionPolicy {
 public:
  CompressionConfig config() const override {
    return {.compressionType = CompressionType::Uncompressed};
  }

  bool shouldAccept(
      CompressionType /* compressionType */,
      uint64_t /* uncompressedSize */,
      uint64_t /* compressedSize */) const override {
    return true;
  }
};

template <typename T>
class TestTrivialSelectionPolicy : public EncodingSelectionPolicy<T> {
  using physicalType = typename TypeTraits<T>::physicalType;

 public:
  EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    return {
        .encodingType = EncodingType::Trivial,
        .compressionPolicyFactory = []() {
          return std::make_unique<TestCompressPolicy>();
        }};
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    return {.encodingType = EncodingType::Nullable};
  }

  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType /* encodingType */,
      NestedEncodingIdentifier /* identifier */,
      DataType type) override {
    UNIQUE_PTR_FACTORY(type, TestTrivialSelectionPolicy);
  }
};

} // namespace

template <typename T>
class DeltaEncodingLegacyTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    if (!velox::memory::MemoryManager::testInstance()) {
      velox::memory::MemoryManager::initialize({});
    }
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool();
    buffer_ = std::make_unique<Buffer>(*pool_);
  }

  Vector<T> toVector(std::initializer_list<T> l) {
    Vector<T> v{pool_.get()};
    v.insert(v.end(), l.begin(), l.end());
    return v;
  }

  // Encode and decode using the legacy DeltaEncoding.
  std::unique_ptr<Encoding> encodeAndDecode(const Vector<T>& values) {
    using physicalType = typename TypeTraits<T>::physicalType;
    auto physicalValues = std::span<const physicalType>(
        reinterpret_cast<const physicalType*>(values.data()), values.size());
    EncodingSelection<physicalType> selection{
        {.encodingType = EncodingType::Delta,
         .compressionPolicyFactory =
             []() { return std::make_unique<TestCompressPolicy>(); }},
        Statistics<physicalType>::create(physicalValues),
        std::make_unique<TestTrivialSelectionPolicy<T>>()};

    auto encoded =
        legacy::DeltaEncoding<T>::encode(selection, physicalValues, *buffer_);

    std::vector<velox::BufferPtr> stringBuffers;
    auto stringBufferFactory = [&](uint32_t totalLength) {
      auto& buf = stringBuffers.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
      return buf->template asMutable<void>();
    };
    return std::make_unique<legacy::DeltaEncoding<T>>(
        *pool_, encoded, stringBufferFactory);
  }

  // Helper: encode and decode with legacy, and verify roundtrip.
  void verifyRoundtrip(const Vector<T>& values) {
    auto encoding = encodeAndDecode(values);

    EXPECT_EQ(encoding->encodingType(), EncodingType::Delta);
    EXPECT_EQ(encoding->dataType(), TypeTraits<T>::dataType);
    EXPECT_EQ(encoding->rowCount(), values.size());

    Vector<T> result(pool_.get(), values.size());
    encoding->materialize(values.size(), result.data());

    for (uint32_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(result[i], values[i]) << "Mismatch at index " << i;
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<Buffer> buffer_;
};

using IntegerTypes = ::testing::Types<
    int8_t,
    uint8_t,
    int16_t,
    uint16_t,
    int32_t,
    uint32_t,
    int64_t,
    uint64_t>;

NIMBLE_TYPED_TEST_SUITE(DeltaEncodingLegacyTest, IntegerTypes);

// The example from the DeltaEncoding.h header comment.
TYPED_TEST(DeltaEncodingLegacyTest, HeaderExample) {
  auto values = this->toVector({1, 2, 4, 1, 2, 3, 4, 1, 2, 4, 8, 8});
  this->verifyRoundtrip(values);
}

// Constant sequence (all deltas are 0).
TYPED_TEST(DeltaEncodingLegacyTest, ConstantValues) {
  auto values = this->toVector({5, 5, 5, 5, 5});
  this->verifyRoundtrip(values);
}

// Single value.
TYPED_TEST(DeltaEncodingLegacyTest, SingleValue) {
  auto values = this->toVector({42});
  this->verifyRoundtrip(values);
}

// Large deltas within type range.
TYPED_TEST(DeltaEncodingLegacyTest, LargeDeltas) {
  using T = TypeParam;
  T maxVal = std::numeric_limits<T>::max();
  T mid = maxVal / 2;
  auto values = this->toVector({0, mid, maxVal});
  this->verifyRoundtrip(values);
}

// Skip then materialize.
TYPED_TEST(DeltaEncodingLegacyTest, SkipThenMaterialize) {
  auto values = this->toVector({1, 2, 4, 1, 2, 3, 4, 1, 2, 4, 8, 8});
  auto encoding = this->encodeAndDecode(values);

  encoding->skip(4);

  Vector<TypeParam> result(this->pool_.get(), 8);
  encoding->materialize(8, result.data());

  for (uint32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(result[i], values[i + 4]) << "Mismatch at index " << i;
  }
}

// Materialize in chunks.
TYPED_TEST(DeltaEncodingLegacyTest, MaterializeInChunks) {
  auto values = this->toVector({1, 3, 5, 2, 4, 6, 8, 10, 7, 9});
  auto encoding = this->encodeAndDecode(values);

  uint32_t chunks[] = {3, 4, 3};
  uint32_t offset = 0;
  for (auto chunkSize : chunks) {
    Vector<TypeParam> result(this->pool_.get(), chunkSize);
    encoding->materialize(chunkSize, result.data());
    for (uint32_t i = 0; i < chunkSize; ++i) {
      EXPECT_EQ(result[i], values[offset + i]);
    }
    offset += chunkSize;
  }
}

// Reset and re-materialize.
TYPED_TEST(DeltaEncodingLegacyTest, ResetAndRematerialize) {
  auto values = this->toVector({5, 10, 15, 8, 12});
  auto encoding = this->encodeAndDecode(values);

  Vector<TypeParam> result1(this->pool_.get(), 5);
  encoding->materialize(5, result1.data());
  encoding->reset();
  Vector<TypeParam> result2(this->pool_.get(), 5);
  encoding->materialize(5, result2.data());

  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(result1[i], values[i]);
    EXPECT_EQ(result2[i], values[i]);
  }
}

// Large dataset with monotonic increase.
TYPED_TEST(DeltaEncodingLegacyTest, LargeMonotonicDataset) {
  Vector<TypeParam> values(this->pool_.get());
  for (uint32_t i = 0; i < 1000; ++i) {
    values.push_back(static_cast<TypeParam>(i));
  }
  this->verifyRoundtrip(values);
}

// Skip with restatement as the last value in the skipped range.
TYPED_TEST(DeltaEncodingLegacyTest, SkipEndingOnRestatement) {
  // Values: 10 20 30 5 15 25
  // isRestatement: T F F T F F
  // Skip 4 values (ends on the restatement at index 3 = value 5).
  auto values = this->toVector({10, 20, 30, 5, 15, 25});
  auto encoding = this->encodeAndDecode(values);
  encoding->skip(4);

  Vector<TypeParam> result(this->pool_.get(), 2);
  encoding->materialize(2, result.data());
  EXPECT_EQ(result[0], 15);
  EXPECT_EQ(result[1], 25);
}

// Two values where the second is smaller (immediate restatement).
TYPED_TEST(DeltaEncodingLegacyTest, TwoValuesDecreasing) {
  auto values = this->toVector({10, 5});
  this->verifyRoundtrip(values);
}

// All values are identical (constant deltas of 0).
TYPED_TEST(DeltaEncodingLegacyTest, AllIdentical) {
  auto values = this->toVector({7, 7, 7, 7, 7, 7, 7, 7, 7, 7});
  this->verifyRoundtrip(values);
}

// Alternating high-low pattern: every other value causes a restatement.
TYPED_TEST(DeltaEncodingLegacyTest, AlternatingHighLow) {
  using T = TypeParam;
  T hi = std::numeric_limits<T>::max() / 2;
  T lo = 1;
  auto values = this->toVector({lo, hi, lo, hi, lo, hi, lo, hi});
  this->verifyRoundtrip(values);
}

// Skip all rows: skip the entire dataset, then there's nothing to read.
TYPED_TEST(DeltaEncodingLegacyTest, SkipAll) {
  auto values = this->toVector({1, 2, 3, 4, 5});
  auto encoding = this->encodeAndDecode(values);
  encoding->skip(5);
  // No rows remain — nothing to materialize.
  EXPECT_EQ(encoding->rowCount(), 5);
}

// Materialize one row at a time.
TYPED_TEST(DeltaEncodingLegacyTest, MaterializeOneByOne) {
  auto values = this->toVector({3, 1, 4, 1, 5, 9, 2, 6});
  auto encoding = this->encodeAndDecode(values);

  for (uint32_t i = 0; i < values.size(); ++i) {
    Vector<TypeParam> result(this->pool_.get(), 1);
    encoding->materialize(1, result.data());
    EXPECT_EQ(result[0], values[i]) << "Mismatch at index " << i;
  }
}

// Skip interleaved with materialize in small steps.
TYPED_TEST(DeltaEncodingLegacyTest, InterleavedSkipMaterialize) {
  auto values = this->toVector({10, 20, 30, 40, 50, 60, 70, 80, 90, 100});
  auto encoding = this->encodeAndDecode(values);

  // skip 2, read 1, skip 3, read 2, skip 1, read 1
  encoding->skip(2);
  {
    Vector<TypeParam> result(this->pool_.get(), 1);
    encoding->materialize(1, result.data());
    EXPECT_EQ(result[0], 30);
  }
  encoding->skip(3);
  {
    Vector<TypeParam> result(this->pool_.get(), 2);
    encoding->materialize(2, result.data());
    EXPECT_EQ(result[0], 70);
    EXPECT_EQ(result[1], 80);
  }
  encoding->skip(1);
  {
    Vector<TypeParam> result(this->pool_.get(), 1);
    encoding->materialize(1, result.data());
    EXPECT_EQ(result[0], 100);
  }
}

// Consecutive restatements: strictly decreasing sequence.
TYPED_TEST(DeltaEncodingLegacyTest, StrictlyDecreasing) {
  auto values = this->toVector({50, 40, 30, 20, 10, 5, 3, 1});
  this->verifyRoundtrip(values);
}

// Mixed: increasing runs followed by drops (sawtooth pattern).
TYPED_TEST(DeltaEncodingLegacyTest, SawtoothPattern) {
  Vector<TypeParam> values(this->pool_.get());
  for (uint32_t i = 0; i < 100; ++i) {
    values.push_back(static_cast<TypeParam>(i % 10));
  }
  this->verifyRoundtrip(values);
}

// Empty data should throw.
TYPED_TEST(DeltaEncodingLegacyTest, EmptyDataThrows) {
  using physicalType = typename TypeTraits<TypeParam>::physicalType;
  auto physicalValues = std::span<const physicalType>();
  EncodingSelection<physicalType> selection{
      {.encodingType = EncodingType::Delta,
       .compressionPolicyFactory =
           []() { return std::make_unique<TestCompressPolicy>(); }},
      Statistics<physicalType>::create(physicalValues),
      std::make_unique<TestTrivialSelectionPolicy<TypeParam>>()};

  NIMBLE_ASSERT_THROW(
      legacy::DeltaEncoding<TypeParam>::encode(
          selection, physicalValues, *this->buffer_),
      "DeltaEncoding can't be used with 0 rows");
}

} // namespace facebook::nimble::test
