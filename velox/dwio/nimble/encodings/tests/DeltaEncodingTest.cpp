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
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <type_traits>
#include <vector>
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/tests/NimbleCompare.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

template <typename DataType, bool UseVarint>
struct TestConfig {
  using data_type = DataType;
  static constexpr bool useVarint = UseVarint;
};

#define TC(T) TestConfig<T, false>, TestConfig<T, true>

template <typename Config>
class DeltaEncodingTest : public ::testing::Test {
 protected:
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
    if constexpr (std::is_same_v<T, int32_t>) {
      return {
          // Monotonically increasing — all deltas, no restatements.
          toVector({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
          // Non-monotonic — triggers restatements on decrease.
          toVector({1, 2, 4, 1, 2, 3, 4, 1, 2, 4, 8, 8}),
          // Single element.
          toVector({42}),
          // Constant — all same values, deltas are all 0.
          toVector({5, 5, 5, 5, 5}),
          // Signed: negative to positive — triggers zero-crossing restatement.
          toVector({-5, -3, -1, 2, 4, 6}),
          // Signed: all negative, increasing (toward zero).
          toVector({-10, -8, -6, -4, -2}),
          // Signed: decreasing from positive to negative.
          toVector({5, 3, 1, -1, -3, -5}),
          // Large gaps.
          toVector({0, 1000, 2000, 3000, 100, 200, 300}),
      };
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      return {
          // Monotonically increasing.
          toVector<uint32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
          // Non-monotonic.
          toVector<uint32_t>({10, 20, 30, 5, 15, 25}),
          // Single element.
          toVector<uint32_t>({99}),
          // Constant.
          toVector<uint32_t>({7, 7, 7, 7}),
      };
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return {
          // Monotonically increasing.
          toVector<int64_t>({100, 200, 300, 400, 500}),
          // Non-monotonic with restatements.
          toVector<int64_t>({100, 200, 50, 150, 250}),
          // Signed zero-crossing.
          toVector<int64_t>({-100, -50, 0, 50, 100}),
      };
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return {
          toVector<uint64_t>({10, 20, 30, 40, 50}),
          toVector<uint64_t>({100, 200, 50, 75, 300}),
      };
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return {
          toVector<int16_t>({1, 2, 3, 4, 5}),
          toVector<int16_t>({5, 3, 1, -1, -3}),
      };
    } else if constexpr (std::is_same_v<T, uint16_t>) {
      return {
          toVector<uint16_t>({1, 2, 3, 4, 5}),
          toVector<uint16_t>({10, 5, 15, 3, 20}),
      };
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return {
          toVector<int8_t>({1, 2, 3, 4, 5}),
          toVector<int8_t>({-5, -3, -1, 1, 3}),
      };
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      return {
          toVector<uint8_t>({1, 2, 3, 4, 5}),
          toVector<uint8_t>({10, 5, 15, 3, 20}),
      };
    } else {
      static_assert(!std::is_same_v<T, T>, "Unsupported test type");
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

#define NUM_TYPES                                                  \
  TC(int8_t), TC(uint8_t), TC(int16_t), TC(uint16_t), TC(int32_t), \
      TC(uint32_t), TC(int64_t), TC(uint64_t)

using TestTypes = ::testing::Types<NUM_TYPES>;

TYPED_TEST_CASE(DeltaEncodingTest, TestTypes);

TYPED_TEST(DeltaEncodingTest, serializeThenDeserialize) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto valueGroups = this->template prepareValues<D>();
  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  for (const auto& values : valueGroups) {
    auto encoding =
        nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
            *this->buffer_,
            values,
            stringBufferFactory,
            nimble::CompressionType::Uncompressed,
            options);
    uint32_t rowCount = values.size();
    nimble::Vector<D> result(this->pool_.get(), rowCount);
    encoding->materialize(rowCount, result.data());

    EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Delta);
    EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<D>::dataType);
    EXPECT_EQ(encoding->rowCount(), rowCount);
    for (uint32_t i = 0; i < rowCount; ++i) {
      EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]))
          << "Mismatch at index " << i << ": expected " << values[i] << " got "
          << result[i];
    }
  }
}

TYPED_TEST(DeltaEncodingTest, resetAndMaterializeInParts) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto valueGroups = this->template prepareValues<D>();
  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  for (const auto& values : valueGroups) {
    auto encoding =
        nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
            *this->buffer_,
            values,
            stringBufferFactory,
            nimble::CompressionType::Uncompressed,
            options);
    uint32_t rowCount = values.size();

    // Materialize in two halves.
    uint32_t firstHalf = rowCount / 2;
    uint32_t secondHalf = rowCount - firstHalf;
    nimble::Vector<D> result(this->pool_.get(), rowCount);

    encoding->materialize(firstHalf, result.data());
    for (uint32_t i = 0; i < firstHalf; ++i) {
      EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]))
          << "First half mismatch at index " << i;
    }

    encoding->materialize(secondHalf, result.data());
    for (uint32_t i = 0; i < secondHalf; ++i) {
      EXPECT_TRUE(
          nimble::NimbleCompare<D>::equals(result[i], values[firstHalf + i]))
          << "Second half mismatch at index " << i;
    }

    // Reset and read one at a time.
    encoding->reset();
    for (uint32_t i = 0; i < rowCount; ++i) {
      D val;
      encoding->materialize(1, &val);
      EXPECT_TRUE(nimble::NimbleCompare<D>::equals(val, values[i]))
          << "One-at-a-time mismatch at index " << i;
    }
  }
}

TYPED_TEST(DeltaEncodingTest, skipThenMaterialize) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto valueGroups = this->template prepareValues<D>();
  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  for (const auto& values : valueGroups) {
    if (values.size() < 3) {
      continue;
    }
    auto encoding =
        nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
            *this->buffer_,
            values,
            stringBufferFactory,
            nimble::CompressionType::Uncompressed,
            options);
    uint32_t rowCount = values.size();
    uint32_t skipCount = rowCount / 3;
    uint32_t readCount = rowCount - skipCount;

    encoding->skip(skipCount);
    nimble::Vector<D> result(this->pool_.get(), readCount);
    encoding->materialize(readCount, result.data());

    for (uint32_t i = 0; i < readCount; ++i) {
      EXPECT_TRUE(
          nimble::NimbleCompare<D>::equals(result[i], values[skipCount + i]))
          << "Skip+materialize mismatch at index " << i;
    }
  }
}

TYPED_TEST(DeltaEncodingTest, boundaryValues) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Build a vector with min, max, and adjacent values for this type.
  nimble::Vector<D> values{this->pool_.get()};
  values.push_back(std::numeric_limits<D>::min());
  values.push_back(static_cast<D>(std::numeric_limits<D>::min() + 1));
  values.push_back(static_cast<D>(std::numeric_limits<D>::max() - 1));
  values.push_back(std::numeric_limits<D>::max());
  // Drop back to min — forces a restatement.
  values.push_back(std::numeric_limits<D>::min());
  values.push_back(std::numeric_limits<D>::max());

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);
  uint32_t rowCount = values.size();
  nimble::Vector<D> result(this->pool_.get(), rowCount);
  encoding->materialize(rowCount, result.data());

  for (uint32_t i = 0; i < rowCount; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]))
        << "BoundaryValues mismatch at index " << i;
  }
}

TYPED_TEST(DeltaEncodingTest, allZeros) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  nimble::Vector<D> values{this->pool_.get()};
  for (int i = 0; i < 20; ++i) {
    values.push_back(static_cast<D>(0));
  }

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);
  nimble::Vector<D> result(this->pool_.get(), values.size());
  encoding->materialize(values.size(), result.data());

  for (uint32_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(result[i], static_cast<D>(0)) << "AllZeros mismatch at " << i;
  }
}

TYPED_TEST(DeltaEncodingTest, strictlyDecreasing) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Strictly decreasing — every row after the first is a restatement.
  nimble::Vector<D> values{this->pool_.get()};
  constexpr int kCount = 15;
  for (int i = 0; i < kCount; ++i) {
    values.push_back(static_cast<D>(kCount - i));
  }

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);
  nimble::Vector<D> result(this->pool_.get(), kCount);
  encoding->materialize(kCount, result.data());

  for (uint32_t i = 0; i < kCount; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]))
        << "StrictlyDecreasing mismatch at " << i;
  }
}

TYPED_TEST(DeltaEncodingTest, alternatingMinMax) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Alternating min/max — restatements on every row after the first.
  nimble::Vector<D> values{this->pool_.get()};
  constexpr int kCount = 10;
  for (int i = 0; i < kCount; ++i) {
    values.push_back(
        i % 2 == 0 ? std::numeric_limits<D>::min()
                   : std::numeric_limits<D>::max());
  }

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);
  nimble::Vector<D> result(this->pool_.get(), kCount);
  encoding->materialize(kCount, result.data());

  for (uint32_t i = 0; i < kCount; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]))
        << "AlternatingMinMax mismatch at " << i;
  }
}

TYPED_TEST(DeltaEncodingTest, skipZeroRows) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->template toVector<D>({1, 2, 3, 4, 5});

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  // skip(0) should be a no-op.
  encoding->skip(0);

  nimble::Vector<D> result(this->pool_.get(), 5);
  encoding->materialize(5, result.data());

  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]))
        << "SkipZeroRows mismatch at " << i;
  }
}

TYPED_TEST(DeltaEncodingTest, skipAllThenMaterialize) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->template toVector<D>({1, 2, 3, 4, 5});

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  // Skip all rows.
  encoding->skip(5);

  // Reset, skip some, materialize the rest.
  encoding->reset();
  encoding->skip(3);
  nimble::Vector<D> result(this->pool_.get(), 2);
  encoding->materialize(2, result.data());

  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[3 + i]))
        << "SkipAllThenMaterialize mismatch at " << i;
  }
}

TYPED_TEST(DeltaEncodingTest, multipleSkipMaterializeCycles) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  nimble::Vector<D> values{this->pool_.get()};
  for (int i = 0; i < 12; ++i) {
    values.push_back(static_cast<D>(i + 1));
  }

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  // skip(2), read(2), skip(2), read(2), skip(2), read(2)
  for (int cycle = 0; cycle < 3; ++cycle) {
    encoding->skip(2);
    nimble::Vector<D> result(this->pool_.get(), 2);
    encoding->materialize(2, result.data());
    uint32_t offset = cycle * 4 + 2;
    for (uint32_t i = 0; i < 2; ++i) {
      EXPECT_TRUE(
          nimble::NimbleCompare<D>::equals(result[i], values[offset + i]))
          << "MultipleSkipMaterializeCycles mismatch at cycle=" << cycle
          << " i=" << i;
    }
  }
}

TYPED_TEST(DeltaEncodingTest, resetAfterPartialSkipAndMaterialize) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  nimble::Vector<D> values{this->pool_.get()};
  for (int i = 0; i < 10; ++i) {
    values.push_back(static_cast<D>(i * 2 + 1));
  }

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  // Partial skip + materialize.
  encoding->skip(4);
  nimble::Vector<D> partial(this->pool_.get(), 3);
  encoding->materialize(3, partial.data());
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(partial[i], values[4 + i]));
  }

  // Reset and verify full stream reads correctly.
  encoding->reset();
  nimble::Vector<D> full(this->pool_.get(), 10);
  encoding->materialize(10, full.data());
  for (uint32_t i = 0; i < 10; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(full[i], values[i]))
        << "ResetAfterPartialSkipAndMaterialize mismatch at " << i;
  }
}

TYPED_TEST(DeltaEncodingTest, emptyData) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  nimble::Vector<D> values{this->pool_.get()};

  // Encoding 0 rows should throw IncompatibleEncoding.
  EXPECT_THROW(
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::encode(
          *this->buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options),
      nimble::NimbleUserError);
}

TYPED_TEST(DeltaEncodingTest, largeRowCount) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // 100K rows: mostly monotonic with periodic restatements.
  constexpr uint32_t kRowCount = 100'000;
  nimble::Vector<D> values{this->pool_.get()};
  values.reserve(kRowCount);
  for (uint32_t i = 0; i < kRowCount; ++i) {
    // Every 1000 rows, reset to a small value (restatement).
    if (i % 1000 == 0) {
      values.push_back(static_cast<D>(1));
    } else {
      values.push_back(static_cast<D>((i % 1000) + 1));
    }
  }

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);
  EXPECT_EQ(encoding->rowCount(), kRowCount);

  nimble::Vector<D> result(this->pool_.get(), kRowCount);
  encoding->materialize(kRowCount, result.data());
  for (uint32_t i = 0; i < kRowCount; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]))
        << "LargeRowCount mismatch at " << i;
  }
}

TYPED_TEST(DeltaEncodingTest, compressionRoundTrip) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  nimble::Vector<D> values{this->pool_.get()};
  for (int i = 0; i < 50; ++i) {
    values.push_back(static_cast<D>(i * 3 + 1));
  }

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Zstd,
          options);
  EXPECT_EQ(encoding->rowCount(), 50);

  nimble::Vector<D> result(this->pool_.get(), 50);
  encoding->materialize(50, result.data());
  for (uint32_t i = 0; i < 50; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]))
        << "CompressionRoundTrip mismatch at " << i;
  }
}

TYPED_TEST(DeltaEncodingTest, singleRestatementAtEnd) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Monotonically increasing then a single drop at the end.
  nimble::Vector<D> values{this->pool_.get()};
  for (int i = 0; i < 10; ++i) {
    values.push_back(static_cast<D>(i + 1));
  }
  // Drop — triggers a single restatement at the end.
  values.push_back(static_cast<D>(1));

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);
  uint32_t rowCount = values.size();
  nimble::Vector<D> result(this->pool_.get(), rowCount);
  encoding->materialize(rowCount, result.data());

  for (uint32_t i = 0; i < rowCount; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]))
        << "SingleRestatementAtEnd mismatch at " << i;
  }
}

// Verifies round-trip correctness when values increase from negative to
// positive. The negative-to-positive transition must be treated as a
// restatement (not a delta) to avoid signed overflow.
TYPED_TEST(DeltaEncodingTest, zeroCrossingNegativeToPositive) {
  using D = typename TypeParam::data_type;
  if constexpr (nimble::isSignedIntegralType<D>()) {
    const nimble::Encoding::Options options{
        .useVarintRowCount = TypeParam::useVarint};

    auto values = this->template toVector<D>({-3, -1, 2, 5});

    std::vector<velox::BufferPtr> newStringBuffers;
    const auto stringBufferFactory = [&](uint32_t totalLength) {
      auto& buffer = newStringBuffers.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
      return buffer->template asMutable<void>();
    };
    auto encoding =
        nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
            *this->buffer_,
            values,
            stringBufferFactory,
            nimble::CompressionType::Uncompressed,
            options);
    nimble::Vector<D> result(this->pool_.get(), values.size());
    encoding->materialize(values.size(), result.data());

    for (uint32_t i = 0; i < values.size(); ++i) {
      EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]))
          << "Mismatch at index " << i << ": expected " << values[i] << " got "
          << result[i];
    }
  } else {
    GTEST_SUCCEED() << "Zero crossing only applies to signed types";
  }
}

// Verifies round-trip when values cross zero multiple times, mixing
// increasing crossings (restatement) and decreasing crossings (also
// restatement due to decrease).
TYPED_TEST(DeltaEncodingTest, zeroCrossingMultipleCrossings) {
  using D = typename TypeParam::data_type;
  if constexpr (nimble::isSignedIntegralType<D>()) {
    const nimble::Encoding::Options options{
        .useVarintRowCount = TypeParam::useVarint};

    // -5→3 crosses zero upward (restatement), 3→-2 decreases (restatement),
    // -2→7 crosses zero upward (restatement), 7→-1 decreases (restatement),
    // -1→4 crosses zero upward (restatement).
    auto values = this->template toVector<D>({-5, 3, -2, 7, -1, 4});

    std::vector<velox::BufferPtr> newStringBuffers;
    const auto stringBufferFactory = [&](uint32_t totalLength) {
      auto& buffer = newStringBuffers.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
      return buffer->template asMutable<void>();
    };
    auto encoding =
        nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
            *this->buffer_,
            values,
            stringBufferFactory,
            nimble::CompressionType::Uncompressed,
            options);
    nimble::Vector<D> result(this->pool_.get(), values.size());
    encoding->materialize(values.size(), result.data());

    for (uint32_t i = 0; i < values.size(); ++i) {
      EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]))
          << "Mismatch at index " << i << ": expected " << values[i] << " got "
          << result[i];
    }
  } else {
    GTEST_SUCCEED() << "Zero crossing only applies to signed types";
  }
}

// Values that pass through zero itself ({-2, -1, 0, 1, 2}) should NOT trigger
// the crossesZero path, because 0 is neither > 0 nor < 0. All transitions
// are plain deltas.
TYPED_TEST(DeltaEncodingTest, zeroCrossingThroughZeroValue) {
  using D = typename TypeParam::data_type;
  if constexpr (nimble::isSignedIntegralType<D>()) {
    const nimble::Encoding::Options options{
        .useVarintRowCount = TypeParam::useVarint};

    auto values = this->template toVector<D>({-2, -1, 0, 1, 2});

    std::vector<velox::BufferPtr> newStringBuffers;
    const auto stringBufferFactory = [&](uint32_t totalLength) {
      auto& buffer = newStringBuffers.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
      return buffer->template asMutable<void>();
    };
    auto encoding =
        nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
            *this->buffer_,
            values,
            stringBufferFactory,
            nimble::CompressionType::Uncompressed,
            options);
    nimble::Vector<D> result(this->pool_.get(), values.size());
    encoding->materialize(values.size(), result.data());

    for (uint32_t i = 0; i < values.size(); ++i) {
      EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[i]))
          << "Mismatch at index " << i << ": expected " << values[i] << " got "
          << result[i];
    }
  } else {
    GTEST_SUCCEED() << "Zero crossing only applies to signed types";
  }
}

// Verifies that skip correctly handles zero-crossing restatements.
TYPED_TEST(DeltaEncodingTest, zeroCrossingWithSkipAndMaterialize) {
  using D = typename TypeParam::data_type;
  if constexpr (nimble::isSignedIntegralType<D>()) {
    const nimble::Encoding::Options options{
        .useVarintRowCount = TypeParam::useVarint};

    // Skip past the zero crossing at index 2 (-1→2), then read the rest.
    auto values = this->template toVector<D>({-5, -1, 2, 4, 6, 8});

    std::vector<velox::BufferPtr> newStringBuffers;
    const auto stringBufferFactory = [&](uint32_t totalLength) {
      auto& buffer = newStringBuffers.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
      return buffer->template asMutable<void>();
    };
    auto encoding =
        nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
            *this->buffer_,
            values,
            stringBufferFactory,
            nimble::CompressionType::Uncompressed,
            options);

    // Skip past the zero crossing point.
    encoding->skip(3);
    nimble::Vector<D> result(this->pool_.get(), 3);
    encoding->materialize(3, result.data());

    for (uint32_t i = 0; i < 3; ++i) {
      EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[3 + i]))
          << "Mismatch at index " << i << ": expected " << values[3 + i]
          << " got " << result[i];
    }

    // Reset and materialize one-by-one across the crossing.
    encoding->reset();
    for (uint32_t i = 0; i < values.size(); ++i) {
      D val;
      encoding->materialize(1, &val);
      EXPECT_TRUE(nimble::NimbleCompare<D>::equals(val, values[i]))
          << "One-at-a-time mismatch at index " << i;
    }
  } else {
    GTEST_SUCCEED() << "Zero crossing only applies to signed types";
  }
}

// --- Tests targeting the fused reverse scan optimization in skip() ---
// The skip() method uses a single reverse scan to find the last restatement
// position AND count total restatements simultaneously. These tests exercise
// various restatement patterns within the skip range to ensure the fused scan
// produces correct results.

// Skip exactly 1 row (the minimum non-zero skip). The first row is always a
// restatement, so lastRestatement=0, totalRestatements=1, deltasToAccumulate=0.
TYPED_TEST(DeltaEncodingTest, skipSingleRow) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->template toVector<D>({3, 5, 7, 9, 11});

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  encoding->skip(1);
  nimble::Vector<D> result(this->pool_.get(), 4);
  encoding->materialize(4, result.data());

  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[1 + i]))
        << "Mismatch at index " << i;
  }
}

// Skip N-1 rows, then materialize only the last row. Tests that the fused
// reverse scan correctly accumulates all trailing deltas after the last
// restatement.
TYPED_TEST(DeltaEncodingTest, skipToLastRow) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->template toVector<D>({2, 4, 6, 8, 10});

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  encoding->skip(4);
  D lastVal;
  encoding->materialize(1, &lastVal);
  EXPECT_TRUE(nimble::NimbleCompare<D>::equals(lastVal, values[4]));
}

// Skip range contains consecutive restatements (3 back-to-back decreasing
// values). The fused reverse scan must correctly find the LAST restatement
// and count ALL of them.
TYPED_TEST(DeltaEncodingTest, skipConsecutiveRestatements) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Values: [20, 15, 10, 5, 6, 7, 8]
  // isRestatement: [T, T, T, T, F, F, F]
  // Skip 4 rows: lastRestatement=3, totalRestatements=4
  // After skip(4): currentValue_ = 5
  // Then materialize 3: should give {6, 7, 8}
  auto values = this->template toVector<D>({20, 15, 10, 5, 6, 7, 8});

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  encoding->skip(4);
  nimble::Vector<D> result(this->pool_.get(), 3);
  encoding->materialize(3, result.data());

  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[4 + i]))
        << "Mismatch at index " << i;
  }
}

// Skip range ends exactly on a restatement (the last skipped row is a
// decrease). deltasToAccumulate should be 0 since there are no trailing
// deltas after the last restatement in the skip range.
TYPED_TEST(DeltaEncodingTest, skipEndingOnRestatement) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Values: [1, 2, 3, 0, 5, 10]
  // isRestatement: [T, F, F, T, F, F]
  // Skip 4: lastRestatement=3, totalRestatements=2
  // deltasToSkip = 3 - 1 = 2, deltasToAccumulate = 4 - 1 - 3 = 0
  // After skip(4): currentValue_ = 0
  // Then materialize 2: should give {5, 10}
  auto values = this->template toVector<D>({1, 2, 3, 0, 5, 10});

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  encoding->skip(4);
  nimble::Vector<D> result(this->pool_.get(), 2);
  encoding->materialize(2, result.data());

  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[4 + i]))
        << "Mismatch at index " << i;
  }
}

// Skip range has trailing deltas after the last restatement, exercising
// the deltasToAccumulate accumulation in the fused scan.
TYPED_TEST(DeltaEncodingTest, skipWithTrailingDeltasAfterRestatement) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Values: [10, 5, 6, 7, 8, 20, 30]
  // isRestatement: [T, T, F, F, F, F, F]
  // Skip 5: lastRestatement=1, totalRestatements=2
  // deltasToSkip = 1 - 1 = 0
  // deltasToAccumulate = 5 - 1 - 1 = 3 → deltas {1, 1, 1} → sum = 3
  // After skip(5): currentValue_ = 5 + 3 = 8
  // Materialize 2: should give {20, 30}
  auto values = this->template toVector<D>({10, 5, 6, 7, 8, 20, 30});

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  encoding->skip(5);
  nimble::Vector<D> result(this->pool_.get(), 2);
  encoding->materialize(2, result.data());

  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[5 + i]))
        << "Mismatch at index " << i;
  }
}

// Skip all-monotonic range (no restatements except the first element).
// The fused scan finds lastRestatement=0, totalRestatements=1, and must
// accumulate all N-1 deltas.
TYPED_TEST(DeltaEncodingTest, skipAllMonotonicThenMaterialize) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->template toVector<D>({1, 3, 6, 10, 15, 21, 28});

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  // Skip first 5, materialize last 2.
  encoding->skip(5);
  nimble::Vector<D> result(this->pool_.get(), 2);
  encoding->materialize(2, result.data());

  EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[0], values[5]));
  EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[1], values[6]));
}

// Skip all-restatement range (strictly decreasing). Every element is a
// restatement. The fused scan must find lastRestatement = rowCount-1,
// totalRestatements = rowCount, deltasToSkip = 0, deltasToAccumulate = 0.
TYPED_TEST(DeltaEncodingTest, skipAllRestatementsThenMaterialize) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Strictly decreasing skip range, then increasing.
  auto values = this->template toVector<D>({50, 40, 30, 20, 21, 22, 23});

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  encoding->skip(4);
  nimble::Vector<D> result(this->pool_.get(), 3);
  encoding->materialize(3, result.data());

  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[4 + i]))
        << "Mismatch at index " << i;
  }
}

// Multiple restatements at scattered positions in the skip range. The fused
// reverse scan must count restatements at positions 0, 3, 5 while correctly
// identifying position 5 as the last.
TYPED_TEST(DeltaEncodingTest, skipScatteredRestatements) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Values: [10, 12, 14, 8, 10, 3, 5, 7]
  // isRestatement: [T, F, F, T, F, T, F, F]
  // Skip 7: lastRestatement=5, totalRestatements=3
  // deltasToSkip = 5 - 2 = 3 (deltas at indices 1, 2, 4)
  // deltasToAccumulate = 7 - 1 - 5 = 1 (delta at index 6: 5-3=2)
  // After skip(7): currentValue_ = 3 + 2 = 5
  // Materialize 1: should give {7}
  auto values = this->template toVector<D>({10, 12, 14, 8, 10, 3, 5, 7});

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  encoding->skip(7);
  D lastVal;
  encoding->materialize(1, &lastVal);
  EXPECT_TRUE(nimble::NimbleCompare<D>::equals(lastVal, values[7]));
}

// Repeated single-row skip + single-row materialize across restatement
// boundaries. Each skip(1) invokes the fused reverse scan on a 1-element
// window, testing the minimal case repeatedly.
TYPED_TEST(DeltaEncodingTest, alternatingSkip1Materialize1) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Pattern with restatements: [10, 5, 8, 3, 12, 7]
  auto values = this->template toVector<D>({10, 5, 8, 3, 12, 7});

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  // Read every other value: skip(1), materialize(1), repeat.
  for (uint32_t i = 0; i < 3; ++i) {
    encoding->skip(1);
    D val;
    encoding->materialize(1, &val);
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(val, values[2 * i + 1]))
        << "Mismatch at pair " << i << ": expected " << values[2 * i + 1]
        << " got " << val;
  }
}

// Large dataset with periodic restatements: sawtooth of 50 values repeated
// 20 times (1000 values). Exercises the fused scan with many restatements at
// regular intervals and a large deltasToAccumulate.
TYPED_TEST(DeltaEncodingTest, skipLargeSawtoothThenMaterialize) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  constexpr uint32_t kPeriod = 50;
  constexpr uint32_t kRepeat = 20;
  constexpr uint32_t kTotal = kPeriod * kRepeat;

  nimble::Vector<D> values(this->pool_.get());
  values.reserve(kTotal);
  for (uint32_t r = 0; r < kRepeat; ++r) {
    for (uint32_t i = 0; i < kPeriod; ++i) {
      values.push_back(static_cast<D>(i));
    }
  }

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  // Skip 900 rows (18 full periods), materialize remaining 100 (2 periods).
  constexpr uint32_t kSkip = 900;
  constexpr uint32_t kRead = kTotal - kSkip;
  encoding->skip(kSkip);
  nimble::Vector<D> result(this->pool_.get(), kRead);
  encoding->materialize(kRead, result.data());

  for (uint32_t i = 0; i < kRead; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[kSkip + i]))
        << "Mismatch at index " << i;
  }
}

// Two skips of different sizes separated by a materialize. The second skip
// starts from a non-initial position, testing that the fused scan works
// correctly when currentValue_ is already set from a prior materialize.
TYPED_TEST(DeltaEncodingTest, twoSkipsMaterializeBetween) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Values: [5, 10, 15, 3, 8, 13, 1, 6, 11, 16]
  auto values = this->template toVector<D>({5, 10, 15, 3, 8, 13, 1, 6, 11, 16});

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  // Skip 2, materialize 2, skip 3, materialize 3.
  encoding->skip(2);
  nimble::Vector<D> r1(this->pool_.get(), 2);
  encoding->materialize(2, r1.data());
  EXPECT_TRUE(nimble::NimbleCompare<D>::equals(r1[0], values[2]));
  EXPECT_TRUE(nimble::NimbleCompare<D>::equals(r1[1], values[3]));

  encoding->skip(3);
  nimble::Vector<D> r2(this->pool_.get(), 3);
  encoding->materialize(3, r2.data());
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(r2[i], values[7 + i]))
        << "Mismatch at index " << i;
  }
}

// Restatements at both boundaries of the skip range (first and last
// positions). Tests that the fused scan correctly handles boundary
// restatements.
TYPED_TEST(DeltaEncodingTest, skipWithRestatementAtBothBoundaries) {
  using D = typename TypeParam::data_type;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  // Values: [10, 12, 14, 5, 20, 25]
  // isRestatement: [T, F, F, T, F, F]
  // Skip 4: lastRestatement=3, totalRestatements=2
  // After skip: currentValue_ = 5, no trailing deltas
  // Wait — index 3 is the last in the skip range (skip 4 = indices 0..3)
  // deltasToAccumulate = 4 - 1 - 3 = 0
  // Materialize 2: {20, 25}
  auto values = this->template toVector<D>({10, 12, 14, 5, 20, 25});

  std::vector<velox::BufferPtr> newStringBuffers;
  const auto stringBufferFactory = [&](uint32_t totalLength) {
    auto& buffer = newStringBuffers.emplace_back(
        velox::AlignedBuffer::allocate<char>(totalLength, this->pool_.get()));
    return buffer->template asMutable<void>();
  };
  auto encoding =
      nimble::test::Encoder<nimble::DeltaEncoding<D>>::createEncoding(
          *this->buffer_,
          values,
          stringBufferFactory,
          nimble::CompressionType::Uncompressed,
          options);

  encoding->skip(4);
  nimble::Vector<D> result(this->pool_.get(), 2);
  encoding->materialize(2, result.data());

  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_TRUE(nimble::NimbleCompare<D>::equals(result[i], values[4 + i]))
        << "Mismatch at index " << i;
  }
}

// Directly tests internal::computeDeltas to verify that zero-crossing
// transitions produce the correct decomposition into deltas, restatements,
// and isRestatements vectors.
TEST(ComputeDeltasTest, zeroCrossingDecomposition) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();

  // Input: {-3, -1, 2, 5}
  // Transition analysis:
  //   -1 >= -3 (delta, no crossing)        → delta = 2
  //   2 >= -1  (increasing, crosses zero)  → restatement = 2
  //   5 >= 2   (delta, no crossing)        → delta = 3
  const std::array<int32_t, 4> values = {-3, -1, 2, 5};

  nimble::Vector<int32_t> deltas(pool.get());
  nimble::Vector<int32_t> restatements(pool.get());
  nimble::Vector<bool> isRestatements(pool.get());

  nimble::internal::computeDeltas<int32_t>(
      std::span<const int32_t>(values),
      &deltas,
      &restatements,
      &isRestatements);

  ASSERT_EQ(isRestatements.size(), 4);
  EXPECT_TRUE(isRestatements[0]); // first element always restatement
  EXPECT_FALSE(isRestatements[1]); // -1 >= -3, no zero crossing
  EXPECT_TRUE(isRestatements[2]); // 2 >= -1, crosses zero → restatement
  EXPECT_FALSE(isRestatements[3]); // 5 >= 2, no zero crossing

  const nimble::Vector<int32_t> expectedDeltas = [&] {
    nimble::Vector<int32_t> v(pool.get());
    v.push_back(2); // -1 - (-3)
    v.push_back(3); // 5 - 2
    return v;
  }();
  ASSERT_EQ(deltas.size(), expectedDeltas.size());
  for (uint32_t i = 0; i < deltas.size(); ++i) {
    EXPECT_EQ(deltas[i], expectedDeltas[i]) << "delta mismatch at " << i;
  }

  const nimble::Vector<int32_t> expectedRestatements = [&] {
    nimble::Vector<int32_t> v(pool.get());
    v.push_back(-3); // first value
    v.push_back(2); // zero-crossing value
    return v;
  }();
  ASSERT_EQ(restatements.size(), expectedRestatements.size());
  for (uint32_t i = 0; i < restatements.size(); ++i) {
    EXPECT_EQ(restatements[i], expectedRestatements[i])
        << "restatement mismatch at " << i;
  }
}

// Verifies that passing through zero (touching the value 0 exactly) does NOT
// trigger zero-crossing restatements. The crossesZero check requires
// values[i] > 0 && values[i-1] < 0, so transitions involving 0 are plain
// deltas.
TEST(ComputeDeltasTest, throughZeroNoZeroCrossing) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();

  // -1 → 0: isDelta=true, crossesZero=false (0 is not > 0) → delta
  // 0 → 1:  isDelta=true, crossesZero=false (0 is not < 0) → delta
  const std::array<int32_t, 3> values = {-1, 0, 1};

  nimble::Vector<int32_t> deltas(pool.get());
  nimble::Vector<int32_t> restatements(pool.get());
  nimble::Vector<bool> isRestatements(pool.get());

  nimble::internal::computeDeltas<int32_t>(
      std::span<const int32_t>(values),
      &deltas,
      &restatements,
      &isRestatements);

  ASSERT_EQ(isRestatements.size(), 3);
  EXPECT_TRUE(isRestatements[0]); // first element always restatement
  EXPECT_FALSE(isRestatements[1]); // -1 → 0, no crossing
  EXPECT_FALSE(isRestatements[2]); // 0 → 1, no crossing

  // All transitions are deltas (none cross zero).
  ASSERT_EQ(deltas.size(), 2);
  EXPECT_EQ(deltas[0], 1); // 0 - (-1)
  EXPECT_EQ(deltas[1], 1); // 1 - 0

  ASSERT_EQ(restatements.size(), 1);
  EXPECT_EQ(restatements[0], -1); // only the first value
}

// Verifies that unsigned types never trigger zero-crossing logic.
TEST(ComputeDeltasTest, unsignedNoZeroCrossing) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();

  // All increasing — every transition is a delta for unsigned types.
  const std::array<uint32_t, 5> values = {1, 2, 3, 4, 5};

  nimble::Vector<uint32_t> deltas(pool.get());
  nimble::Vector<uint32_t> restatements(pool.get());
  nimble::Vector<bool> isRestatements(pool.get());

  nimble::internal::computeDeltas<uint32_t>(
      std::span<const uint32_t>(values),
      &deltas,
      &restatements,
      &isRestatements);

  ASSERT_EQ(isRestatements.size(), 5);
  EXPECT_TRUE(isRestatements[0]);
  for (uint32_t i = 1; i < 5; ++i) {
    EXPECT_FALSE(isRestatements[i]) << "Unexpected restatement at " << i;
  }

  ASSERT_EQ(deltas.size(), 4);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(deltas[i], 1u) << "delta mismatch at " << i;
  }

  ASSERT_EQ(restatements.size(), 1);
  EXPECT_EQ(restatements[0], 1u);
}
