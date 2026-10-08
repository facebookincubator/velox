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
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "folly/Random.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

template <bool UseVarint>
struct VarintTestConfig {
  static constexpr bool useVarint = UseVarint;
};

template <typename Config>
class VarintEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  nimble::Vector<uint32_t> toVector(std::initializer_list<uint32_t> l) {
    nimble::Vector<uint32_t> v{pool_.get()};
    v.insert(v.end(), l.begin(), l.end());
    return v;
  }

  std::unique_ptr<nimble::Encoding> createEncoding(
      const nimble::Vector<uint32_t>& values,
      const nimble::Encoding::Options& options = {}) {
    return nimble::test::Encoder<nimble::VarintEncoding<uint32_t>>::
        createEncoding(
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
    ::testing::Types<VarintTestConfig<false>, VarintTestConfig<true>>;

TYPED_TEST_CASE(VarintEncodingTest, TestTypes);

TYPED_TEST(VarintEncodingTest, serializeThenDeserialize) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({0, 1, 127, 128, 255, 256, 1000});
  auto encoding = this->createEncoding(values, options);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::Varint);
  EXPECT_EQ(encoding->dataType(), nimble::TypeTraits<uint32_t>::dataType);
  EXPECT_EQ(encoding->rowCount(), 7);

  nimble::Vector<uint32_t> result(this->pool_.get(), 7);
  encoding->materialize(7, result.data());
  for (uint32_t i = 0; i < 7; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(VarintEncodingTest, singleValue) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({42});
  auto encoding = this->createEncoding(values, options);

  EXPECT_EQ(encoding->rowCount(), 1);

  nimble::Vector<uint32_t> result(this->pool_.get(), 1);
  encoding->materialize(1, result.data());
  EXPECT_EQ(result[0], 42u);
}

TYPED_TEST(VarintEncodingTest, allZeros) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({0, 0, 0, 0, 0});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), 5);
  encoding->materialize(5, result.data());
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(result[i], 0u) << "index " << i;
  }
}

TYPED_TEST(VarintEncodingTest, allSameNonZero) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({12345, 12345, 12345, 12345});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), 4);
  encoding->materialize(4, result.data());
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(result[i], 12345u) << "index " << i;
  }
}

TYPED_TEST(VarintEncodingTest, varintBoundaryValues) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({0, 127, 128, 16383, 16384, 2097151, 2097152});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), values.size());
  encoding->materialize(values.size(), result.data());
  for (uint32_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(VarintEncodingTest, largeValues) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  uint32_t maxVal = std::numeric_limits<uint32_t>::max();
  uint32_t halfMax = maxVal / 2;
  auto values = this->toVector({maxVal, halfMax, 0, halfMax, maxVal});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), 5);
  encoding->materialize(5, result.data());
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(VarintEncodingTest, skipThenMaterialize) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({10, 20, 30, 40, 50, 60, 70, 80});
  auto encoding = this->createEncoding(values, options);

  encoding->skip(3);

  nimble::Vector<uint32_t> result(this->pool_.get(), 3);
  encoding->materialize(3, result.data());
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_EQ(result[i], values[i + 3]) << "index " << i;
  }
}

TYPED_TEST(VarintEncodingTest, resetAndRematerialize) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({5, 15, 25, 35, 45});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result1(this->pool_.get(), 5);
  encoding->materialize(5, result1.data());

  encoding->reset();

  nimble::Vector<uint32_t> result2(this->pool_.get(), 5);
  encoding->materialize(5, result2.data());

  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(result1[i], result2[i]) << "index " << i;
  }
}

TYPED_TEST(VarintEncodingTest, incrementalMaterialize) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), 10);
  encoding->materialize(4, result.data());
  encoding->materialize(3, result.data() + 4);
  encoding->materialize(3, result.data() + 7);

  for (uint32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(VarintEncodingTest, materializeZeroRows) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({10, 20, 30});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), 3);
  encoding->materialize(0, result.data());

  encoding->materialize(3, result.data());
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(VarintEncodingTest, fuzz) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (int trial = 0; trial < 20; ++trial) {
    uint32_t rowCount = folly::Random::rand32(1, 500, rng);

    nimble::Vector<uint32_t> values{this->pool_.get()};
    for (uint32_t i = 0; i < rowCount; ++i) {
      values.push_back(folly::Random::rand32(rng));
    }

    auto encoding = this->createEncoding(values, options);

    EXPECT_EQ(encoding->rowCount(), rowCount);

    nimble::Vector<uint32_t> result(this->pool_.get(), rowCount);
    encoding->materialize(rowCount, result.data());

    for (uint32_t i = 0; i < rowCount; ++i) {
      EXPECT_EQ(result[i], values[i]) << "trial " << trial << " index " << i;
    }
  }
}

TYPED_TEST(VarintEncodingTest, largeDataset) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  nimble::Vector<uint32_t> values{this->pool_.get()};
  for (uint32_t i = 0; i < 10000; ++i) {
    values.push_back(i * 7);
  }

  auto encoding = this->createEncoding(values, options);
  EXPECT_EQ(encoding->rowCount(), 10000);

  nimble::Vector<uint32_t> result(this->pool_.get(), 10000);
  encoding->materialize(10000, result.data());
  for (uint32_t i = 0; i < 10000; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(VarintEncodingTest, multiByteVarintValues) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector(
      {0,
       (1u << 7) - 1,
       1u << 7,
       (1u << 14) - 1,
       1u << 14,
       (1u << 21) - 1,
       1u << 21,
       (1u << 28) - 1,
       1u << 28,
       std::numeric_limits<uint32_t>::max()});
  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), values.size());
  encoding->materialize(values.size(), result.data());
  for (uint32_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}

TYPED_TEST(VarintEncodingTest, skipToEnd) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto values = this->toVector({10, 20, 30, 40, 50});
  auto encoding = this->createEncoding(values, options);

  encoding->skip(4);

  nimble::Vector<uint32_t> result(this->pool_.get(), 1);
  encoding->materialize(1, result.data());
  EXPECT_EQ(result[0], 50u);
}

TYPED_TEST(VarintEncodingTest, monotonicallyIncreasing) {
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  nimble::Vector<uint32_t> values{this->pool_.get()};
  uint32_t val = 0;
  for (uint32_t i = 0; i < 100; ++i) {
    val += i;
    values.push_back(val);
  }

  auto encoding = this->createEncoding(values, options);

  nimble::Vector<uint32_t> result(this->pool_.get(), 100);
  encoding->materialize(100, result.data());
  for (uint32_t i = 0; i < 100; ++i) {
    EXPECT_EQ(result[i], values[i]) << "index " << i;
  }
}
