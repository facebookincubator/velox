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
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include <gtest/gtest.h>
#include <array>
#include <span>
#include "folly/Random.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/SentinelEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

// Tests the Encoding API for all nullable Encoding implementations + data
// types.
//
// These encodings generally use the factory themselves to encode their non-null
// values as another encoding. We assume that the other encodings are thoroughly
// tested and conform to the API, so we can use just a single underlying
// encoding implementation for the non-null values (namely, the TrivialEncoding)
// rather than having to test all the numNullableEncodings X numNormalEncodings
// combinations.

using namespace facebook;

template <typename EncodingType, bool UseVarint>
struct TestConfig {
  using encoding_type = EncodingType;
  static constexpr bool useVarint = UseVarint;
};

#define TC(T) TestConfig<T, false>, TestConfig<T, true>

#include <random>

namespace {
// Fixed so the shuffled order is reproducible across runs.
constexpr uint32_t kShuffleSeed = 20240816;
} // namespace

namespace {
enum class NullsPattern {
  None,
  All,
  Random,
};
}

// Config wraps the encoding type and varint flag.
template <typename Config>
class NullableEncodingTest : public ::testing::Test {
 protected:
  using C = typename Config::encoding_type;
  using E = typename C::cppDataType;

  void SetUp() override {
    pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
    util_ = std::make_unique<nimble::testing::Util>(*pool_);
  }

  // Makes a random-length nulls vector with num_nonNulls values set to true
  // (and at least one value set to false) scattered randomly throughout.
  template <typename RNG>
  nimble::Vector<bool>
  makeRandomNulls(RNG&& rng, uint32_t dataSize, NullsPattern pattern) {
    if (pattern == NullsPattern::None) {
      return nimble::Vector<bool>{pool_.get(), dataSize, false};
    } else if (pattern == NullsPattern::All) {
      return nimble::Vector<bool>{pool_.get(), dataSize, true};
    }
    const uint32_t rowCount =
        dataSize + folly::Random::rand32(3 * kMaxRows, std::forward<RNG>(rng));
    nimble::Vector<bool> nulls(pool_.get(), rowCount, false);
    for (uint32_t i = 0; i < dataSize; ++i) {
      nulls[i] = true;
    }
    std::shuffle(nulls.begin(), nulls.end(), std::mt19937{kShuffleSeed});
    return nulls;
  }

  // Each unit test runs on randomized data this many times before
  // we conclude the unit test passed.
  static constexpr int kNumRandomRuns = 20;
  // We want the number of row tested to potentially be large compared to a
  // skip block. When we actually generate data we pick a random length between
  // 1 and this size.
  static constexpr int kMaxRows = 2000;

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
  std::unique_ptr<nimble::testing::Util> util_;
};

#define ALL_TYPES(EncodingName)                               \
  TC(EncodingName<int>), TC(EncodingName<int64_t>),           \
      TC(EncodingName<uint32_t>), TC(EncodingName<uint64_t>), \
      TC(EncodingName<float>), TC(EncodingName<double>),      \
      TC(EncodingName<std::string_view>), TC(EncodingName<bool>)

#define NON_BOOL_TYPES(EncodingName)                          \
  TC(EncodingName<int>), TC(EncodingName<int64_t>),           \
      TC(EncodingName<uint32_t>), TC(EncodingName<uint64_t>), \
      TC(EncodingName<float>), TC(EncodingName<double>),      \
      TC(EncodingName<std::string_view>)

using TestTypes = ::testing::Types<
    ALL_TYPES(nimble::NullableEncoding),
    NON_BOOL_TYPES(nimble::SentinelEncoding)>;

TYPED_TEST_CASE(NullableEncodingTest, TestTypes);

//.Spreads the nonNulls out into a vector of length |nulls|, with a non-null
// placed at each true value in |nulls|. Equivalent to Encoding::Materialize.
template <typename E>
nimble::Vector<E> spreadNullsIntoData(
    velox::memory::MemoryPool& memoryPool,
    std::span<const E> nonNulls,
    std::span<const bool> nulls) {
  nimble::Vector<E> result(&memoryPool);
  auto nonNullsIt = nonNulls.begin();
  for (auto nulls_it = nulls.begin(); nulls_it < nulls.end(); ++nulls_it) {
    if (*nulls_it) {
      result.push_back(*nonNullsIt++);
    } else {
      result.push_back(E());
    }
  }

  return result;
}

TYPED_TEST(NullableEncodingTest, materialize) {
  using E = typename TypeParam::encoding_type::cppDataType;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (int run = 0; run < this->kNumRandomRuns; ++run) {
    const std::vector<nimble::Vector<E>> dataPatterns =
        this->util_->template makeDataPatterns<E>(
            rng, this->kMaxRows, this->buffer_.get());
    for (const auto& data : dataPatterns) {
      for (auto nullPattern :
           {NullsPattern::None, NullsPattern::All, NullsPattern::Random}) {
        const nimble::Vector<bool> nulls = this->makeRandomNulls(
            rng, folly::to<uint32_t>(data.size()), nullPattern);
        // Spreading the data out will help us check correctness more easily.
        const nimble::Vector<E> spreadData =
            spreadNullsIntoData<E>(*this->pool_, data, nulls);

        std::vector<velox::BufferPtr> newStringBuffers;
        const auto stringBufferFactory = [&](uint32_t totalLength) {
          auto& buffer = newStringBuffers.emplace_back(
              velox::AlignedBuffer::allocate<char>(
                  totalLength, this->pool_.get()));
          return buffer->template asMutable<void>();
        };
        auto encoding = nimble::test::Encoder<nimble::NullableEncoding<E>>::
            createNullableEncoding(
                *this->buffer_,
                data,
                nulls,
                stringBufferFactory,
                nimble::CompressionType::Uncompressed,
                options);
        ASSERT_EQ(encoding->dataType(), nimble::TypeTraits<E>::dataType);
        ASSERT_TRUE(encoding->isNullable());
        const uint32_t rowCount = encoding->rowCount();

        nimble::Vector<E> buffer(this->pool_.get(), rowCount);
        encoding->materialize(rowCount, buffer.data());
        for (int i = 0; i < rowCount; ++i) {
          ASSERT_EQ(buffer[i], spreadData[i]);
        }

        encoding->reset();
        const int firstBlock = folly::to<int>(rowCount / 2);
        encoding->materialize(firstBlock, buffer.data());
        for (int i = 0; i < firstBlock; ++i) {
          ASSERT_EQ(buffer[i], spreadData[i]);
        }
        const int secondBlock = rowCount - firstBlock;
        encoding->materialize(secondBlock, buffer.data());
        for (int i = 0; i < secondBlock; ++i) {
          ASSERT_EQ(buffer[i], spreadData[firstBlock + i]);
        }

        encoding->reset();
        for (int i = 0; i < rowCount; ++i) {
          encoding->materialize(1, buffer.data());
          ASSERT_EQ(buffer[0], spreadData[i]);
        }

        encoding->reset();
        int start = 0;
        int len = 0;
        for (int i = 0; i < rowCount; ++i) {
          start += len;
          len += 1;
          if (start + len > rowCount) {
            break;
          }
          encoding->materialize(len, buffer.data());
          for (int j = 0; j < len; ++j) {
            ASSERT_EQ(spreadData[start + j], buffer[j]);
          }
        }

        const uint32_t offset =
            folly::to<uint32_t>(folly::Random::rand32(rng) % data.size());
        const uint32_t length = folly::to<uint32_t>(
            1 + folly::Random::rand32(rng) % (data.size() - offset));
        encoding->reset();
        encoding->skip(offset);
        encoding->materialize(length, buffer.data());
        for (uint32_t i = 0; i < length; ++i) {
          ASSERT_EQ(buffer[i], spreadData[offset + i]);
        }
      }
    }
  }
}

class NullableEncodingSliceTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("NullableEncodingTest");
    pool_ = rootPool_->addLeafChild("NullableEncodingTestLeaf");
  }

  nimble::Vector<int32_t> toInt32Vector(std::initializer_list<int32_t> values) {
    nimble::Vector<int32_t> vector{pool_.get()};
    vector.insert(vector.end(), values.begin(), values.end());
    return vector;
  }

  nimble::Vector<bool> toBoolVector(std::initializer_list<bool> values) {
    nimble::Vector<bool> vector{pool_.get()};
    vector.insert(vector.end(), values.begin(), values.end());
    return vector;
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(NullableEncodingSliceTest, slice) {
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useVarint=" << useVarint);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    nimble::Buffer sourceBuffer{*pool_};
    const auto values = toInt32Vector({10, 20, 30, 40, 50});
    const auto nulls =
        toBoolVector({true, false, true, true, false, true, true});
    const auto expected = spreadNullsIntoData<int32_t>(*pool_, values, nulls);
    const auto encoded = nimble::test::
        Encoder<nimble::NullableEncoding<int32_t>>::encodeNullable(
            sourceBuffer,
            values,
            nulls,
            nimble::CompressionType::Uncompressed,
            options);

    struct Range {
      uint32_t offset;
      uint32_t length;
    };
    for (const auto range :
         {Range{/*offset=*/0, /*length=*/3},
          Range{/*offset=*/1, /*length=*/5},
          Range{/*offset=*/4, /*length=*/3}}) {
      SCOPED_TRACE(
          testing::Message()
          << "offset=" << range.offset << ", length=" << range.length);
      nimble::Buffer sliceBuffer{*pool_};
      const auto sliced = nimble::NullableEncoding<int32_t>::slice(
          encoded, range.offset, range.length, sliceBuffer, options);
      nimble::NullableEncoding<int32_t> encoding{
          *pool_,
          sliced,
          [](uint32_t /*totalLength*/) -> void* { return nullptr; },
          options};

      EXPECT_EQ(encoding.encodingType(), nimble::EncodingType::Nullable);
      EXPECT_EQ(encoding.dataType(), nimble::DataType::Int32);
      EXPECT_EQ(encoding.rowCount(), range.length);
      nimble::Vector<int32_t> result(pool_.get(), range.length);
      encoding.materialize(range.length, result.data());
      for (uint32_t i = 0; i < range.length; ++i) {
        EXPECT_EQ(result[i], expected[range.offset + i]);
      }
    }
  }
}

TEST_F(NullableEncodingSliceTest, slicesAllNullRange) {
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useVarint=" << useVarint);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    nimble::Buffer sourceBuffer{*pool_};
    const auto values = toInt32Vector({10, 20});
    const auto nulls = toBoolVector({true, false, false, true});
    const auto encoded = nimble::test::
        Encoder<nimble::NullableEncoding<int32_t>>::encodeNullable(
            sourceBuffer,
            values,
            nulls,
            nimble::CompressionType::Uncompressed,
            options);

    nimble::Buffer sliceBuffer{*pool_};
    const auto sliced = nimble::NullableEncoding<int32_t>::slice(
        encoded, /*offset=*/1, /*length=*/2, sliceBuffer, options);
    nimble::NullableEncoding<int32_t> encoding{
        *pool_,
        sliced,
        [](uint32_t /*totalLength*/) -> void* { return nullptr; },
        options};

    EXPECT_EQ(encoding.rowCount(), 2);
    EXPECT_EQ(encoding.nullCount(), 2);
    EXPECT_EQ(encoding.nonNulls()->rowCount(), 0);
    nimble::Vector<int32_t> output{pool_.get(), 2};
    nimble::Vector<char> outputNulls{pool_.get(), 2};
    EXPECT_EQ(
        encoding.materializeNullable(
            2, output.data(), [&]() { return outputNulls.data(); }),
        0);
  }
}

TEST_F(NullableEncodingSliceTest, encodeNullableFromSerializedChildren) {
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useVarint=" << useVarint);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    const auto values = toInt32Vector({10, 20, 30});
    const auto nulls = toBoolVector({true, false, true, true});

    nimble::Buffer valueChildBuffer{*pool_};
    const auto serializedValues =
        nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
            valueChildBuffer,
            values,
            nimble::CompressionType::Uncompressed,
            options);
    nimble::Buffer nullChildBuffer{*pool_};
    const auto serializedNulls =
        nimble::test::Encoder<nimble::TrivialEncoding<bool>>::encode(
            nullChildBuffer,
            nulls,
            nimble::CompressionType::Uncompressed,
            options);

    nimble::Buffer outputBuffer{*pool_};
    const auto encoded = nimble::NullableEncoding<int32_t>::encodeNullable(
        static_cast<uint32_t>(nulls.size()),
        serializedValues,
        serializedNulls,
        outputBuffer,
        options);
    nimble::NullableEncoding<int32_t> encoding{
        *pool_,
        encoded,
        [](uint32_t /*totalLength*/) -> void* { return nullptr; },
        options};

    EXPECT_EQ(encoding.rowCount(), nulls.size());
    EXPECT_EQ(encoding.nonNulls()->rowCount(), values.size());
    const auto expected = spreadNullsIntoData<int32_t>(*pool_, values, nulls);
    nimble::Vector<int32_t> output{pool_.get(), nulls.size()};
    encoding.materialize(nulls.size(), output.data());
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(output[i], expected[i]);
    }

    const std::array<uint64_t, 3> physicalInput{{10, 20, 30}};
    nimble::Vector<uint64_t> physicalValues{pool_.get()};
    physicalValues.insert(
        physicalValues.end(), physicalInput.begin(), physicalInput.end());
    nimble::Buffer physicalValueChildBuffer{*pool_};
    const auto serializedPhysicalValues =
        nimble::test::Encoder<nimble::TrivialEncoding<uint64_t>>::encode(
            physicalValueChildBuffer,
            physicalValues,
            nimble::CompressionType::Uncompressed,
            options);

    nimble::Buffer physicalOutputBuffer{*pool_};
    const auto physicalEncoded =
        nimble::NullableEncoding<int64_t>::encodeNullable(
            static_cast<uint32_t>(nulls.size()),
            serializedPhysicalValues,
            serializedNulls,
            physicalOutputBuffer,
            options);
    nimble::NullableEncoding<int64_t> physicalEncoding{
        *pool_,
        physicalEncoded,
        [](uint32_t /*totalLength*/) -> void* { return nullptr; },
        options};
    EXPECT_EQ(physicalEncoding.rowCount(), nulls.size());
    EXPECT_EQ(physicalEncoding.nonNulls()->rowCount(), physicalValues.size());
  }
}

TEST_F(NullableEncodingSliceTest, encodeNullableWithDeltaBlockChild) {
  constexpr uint32_t kRowCount = 256;
  nimble::Vector<int64_t> values{pool_.get()};
  nimble::Vector<bool> notNulls{pool_.get()};
  std::vector<int64_t> expected;
  expected.reserve(kRowCount);

  for (uint32_t row = 0; row < kRowCount; ++row) {
    const bool isNotNull = row % 3 != 0;
    notNulls.push_back(isNotNull);
    if (isNotNull) {
      values.push_back(static_cast<int64_t>(row * 11));
      expected.push_back(static_cast<int64_t>(row * 11));
    } else {
      expected.push_back(0);
    }
  }
  ASSERT_EQ(values.size(), 170);

  nimble::ManualEncodingSelectionPolicyFactory factory{
      {{nimble::EncodingType::DeltaBlock, 1.0}}, std::nullopt};
  auto policy = std::unique_ptr<nimble::EncodingSelectionPolicy<int64_t>>(
      static_cast<nimble::EncodingSelectionPolicy<int64_t>*>(
          factory.createPolicy(nimble::DataType::Int64).release()));
  const nimble::Encoding::Options options{
      .useVarintRowCount = false, .deltaBlockSize = 32};
  nimble::Buffer buffer{*pool_};
  const auto encoded = nimble::EncodingFactory::encodeNullable<int64_t>(
      std::move(policy), values, notNulls, buffer, options);

  const char* pos = encoded.data() +
      nimble::EncodingPrefix::prefixSize(encoded, /*useVarint=*/false);
  const auto valuesSize = nimble::encoding::readUint32(pos);
  const std::string_view serializedValues{pos, valuesSize};
  EXPECT_EQ(
      nimble::EncodingPrefix::encodingType(serializedValues),
      nimble::EncodingType::DeltaBlock);
  EXPECT_EQ(
      nimble::EncodingPrefix::readRowCount(
          serializedValues, /*useVarint=*/false),
      values.size());

  nimble::NullableEncoding<int64_t> encoding{
      *pool_,
      encoded,
      [](uint32_t /*totalLength*/) -> void* { return nullptr; },
      options};
  ASSERT_EQ(encoding.rowCount(), kRowCount);
  ASSERT_EQ(
      encoding.nonNulls()->encodingType(), nimble::EncodingType::DeltaBlock);

  std::vector<int64_t> output(kRowCount);
  encoding.materialize(kRowCount, output.data());
  EXPECT_EQ(output, expected);
}

TEST_F(
    NullableEncodingSliceTest,
    encodeNullableFromSerializedChildrenRejectsInvalidChildTypes) {
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useVarint=" << useVarint);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};

    const auto values = toInt32Vector({10, 20});
    const auto nulls = toBoolVector({true, false, true});
    nimble::Buffer valueChildBuffer{*pool_};
    const auto serializedValues =
        nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
            valueChildBuffer,
            values,
            nimble::CompressionType::Uncompressed,
            options);
    nimble::Buffer nullChildBuffer{*pool_};
    const auto serializedNulls =
        nimble::test::Encoder<nimble::TrivialEncoding<bool>>::encode(
            nullChildBuffer,
            nulls,
            nimble::CompressionType::Uncompressed,
            options);

    nimble::Buffer outputBuffer{*pool_};
    NIMBLE_ASSERT_THROW(
        nimble::NullableEncoding<int32_t>::encodeNullable(
            static_cast<uint32_t>(nulls.size()),
            serializedValues,
            serializedValues,
            outputBuffer,
            options),
        "Nullable null child must be bool.");

    NIMBLE_ASSERT_THROW(
        nimble::NullableEncoding<int64_t>::encodeNullable(
            static_cast<uint32_t>(nulls.size()),
            serializedValues,
            serializedNulls,
            outputBuffer,
            options),
        "Nullable value child data type must match");
  }
}

TEST_F(
    NullableEncodingSliceTest,
    encodeNullableFromSerializedChildrenRejectsInvalidRowCounts) {
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useVarint=" << useVarint);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};

    const auto values = toInt32Vector({10, 20, 30, 40});
    nimble::Buffer valueChildBuffer{*pool_};
    const auto serializedValues =
        nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
            valueChildBuffer,
            values,
            nimble::CompressionType::Uncompressed,
            options);
    const auto nulls = toBoolVector({true, false, true});
    nimble::Buffer nullChildBuffer{*pool_};
    const auto serializedNulls =
        nimble::test::Encoder<nimble::TrivialEncoding<bool>>::encode(
            nullChildBuffer,
            nulls,
            nimble::CompressionType::Uncompressed,
            options);

    nimble::Buffer outputBuffer{*pool_};
    NIMBLE_ASSERT_THROW(
        nimble::test::Encoder<nimble::NullableEncoding<int32_t>>::
            encodeNullable(
                outputBuffer,
                values,
                nulls,
                nimble::CompressionType::Uncompressed,
                options),
        "Nullable value count cannot exceed null count.");

    NIMBLE_ASSERT_THROW(
        nimble::NullableEncoding<int32_t>::encodeNullable(
            static_cast<uint32_t>(nulls.size() + 1),
            serializedValues,
            serializedNulls,
            outputBuffer,
            options),
        "Nullable null child row count must match parent.");
  }
}

TEST_F(NullableEncodingSliceTest, invalidSliceRange) {
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(testing::Message() << "useVarint=" << useVarint);
    const nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    nimble::Buffer sourceBuffer{*pool_};
    const auto values = toInt32Vector({10, 20, 30, 40, 50});
    const auto nulls =
        toBoolVector({true, false, true, true, false, true, true});
    const auto encoded = nimble::test::
        Encoder<nimble::NullableEncoding<int32_t>>::encodeNullable(
            sourceBuffer,
            values,
            nulls,
            nimble::CompressionType::Uncompressed,
            options);

    nimble::Buffer invalidSliceBuffer{*pool_};
    NIMBLE_ASSERT_THROW(
        nimble::NullableEncoding<int32_t>::slice(
            encoded,
            /*offset=*/0,
            /*length=*/0,
            invalidSliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        nimble::NullableEncoding<int32_t>::slice(
            encoded,
            /*offset=*/8,
            /*length=*/0,
            invalidSliceBuffer,
            options),
        "");
    NIMBLE_ASSERT_THROW(
        nimble::NullableEncoding<int32_t>::slice(
            encoded,
            /*offset=*/6,
            /*length=*/2,
            invalidSliceBuffer,
            options),
        "");
  }
}

template <typename T>
void checkOutput(
    size_t index,
    const bool* nulls,
    const T* data,
    const char* actualNulls,
    const T* actualData,
    bool hasNulls) {
  if (nulls[index]) {
    ASSERT_EQ(data[index], actualData[index]) << index;
  }
  if (hasNulls) {
    ASSERT_EQ(
        velox::bits::isBitSet(
            reinterpret_cast<const uint8_t*>(actualNulls), index),
        nulls[index])
        << index;
  }
}

TYPED_TEST(NullableEncodingTest, scatteredMaterialize) {
  using E = typename TypeParam::encoding_type::cppDataType;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (int run = 0; run < this->kNumRandomRuns; ++run) {
    const std::vector<nimble::Vector<E>> dataPatterns =
        this->util_->template makeDataPatterns<E>(
            rng, this->kMaxRows, this->buffer_.get());
    for (const auto& data : dataPatterns) {
      for (auto nullPattern :
           {NullsPattern::None, NullsPattern::All, NullsPattern::Random}) {
        const nimble::Vector<bool> nulls =
            this->makeRandomNulls(rng, data.size(), nullPattern);
        // Spreading the data out will help us check correctness more easily.
        const nimble::Vector<E> spreadData =
            spreadNullsIntoData<E>(*this->pool_, data, nulls);

        std::vector<velox::BufferPtr> newStringBuffers;
        const auto stringBufferFactory = [&](uint32_t totalLength) {
          auto& buffer = newStringBuffers.emplace_back(
              velox::AlignedBuffer::allocate<char>(
                  totalLength, this->pool_.get()));
          return buffer->template asMutable<void>();
        };
        auto encoding = nimble::test::Encoder<nimble::NullableEncoding<E>>::
            createNullableEncoding(
                *this->buffer_,
                data,
                nulls,
                stringBufferFactory,
                nimble::CompressionType::Uncompressed,
                options);
        ASSERT_EQ(encoding->dataType(), nimble::TypeTraits<E>::dataType);
        ASSERT_TRUE(encoding->isNullable());
        const uint32_t rowCount = encoding->rowCount();
        ASSERT_EQ(rowCount, nulls.size());

        int setBits = 0;
        std::vector<int32_t> scatterSizes(rowCount + 1);
        scatterSizes[0] = 0;
        nimble::Vector<bool> scatter(this->pool_.get());
        while (setBits < rowCount) {
          scatter.push_back(folly::Random::rand32(2, rng) ? true : false);
          if (scatter.back()) {
            scatterSizes[++setBits] = scatter.size();
          }
        }

        auto newRowCount = scatter.size();
        auto requiredBytes = velox::bits::nbytes(newRowCount);
        // Note: Internally, some bit implementations use word boundaries to
        // efficiently iterate on bitmaps. If the buffer doesn't end on a word
        // boundary, this leads to ASAN buffer overflow (debug builds). So for
        // now, we are allocating extra 7 bytes to make sure the buffer ends or
        // exceeds a word boundary.
        nimble::Buffer scatterBuffer{*this->pool_, requiredBytes + 7};
        nimble::Buffer nullsBuffer{*this->pool_, requiredBytes + 7};
        auto scatterPtr = scatterBuffer.reserve(requiredBytes);
        auto nullsPtr = nullsBuffer.reserve(requiredBytes);
        memset(scatterPtr, 0, requiredBytes);
        velox::bits::packBitmap(scatter, scatterPtr);

        nimble::Vector<E> buffer(this->pool_.get(), newRowCount);

        auto test = [&encoding, &scatter, &nulls, &spreadData](
                        uint32_t rowCount,
                        E* buffer,
                        void* nullsBitmap,
                        uint32_t scatterCount,
                        void* scatterBitmap,
                        uint32_t scatterOffset = 0,
                        uint32_t expectedOffset = 0) {
          uint32_t expectedRow = 0;
          velox::bits::Bitmap bitmap{
              scatterBitmap, scatterOffset + scatterCount};
          auto nonNullCount = encoding->materializeNullable(
              rowCount,
              buffer,
              [&]() { return nullsBitmap; },
              &bitmap,
              scatterOffset);
          for (int i = 0; i < scatterCount; ++i) {
            auto isSet = false;
            if (scatter[i + scatterOffset]) {
              if (nulls[expectedRow + expectedOffset]) {
                ASSERT_EQ(
                    buffer[i + scatterOffset],
                    spreadData[expectedRow + expectedOffset]);
                isSet = true;
              }
              ++expectedRow;
            }
            if (nonNullCount != scatterCount) {
              ASSERT_EQ(
                  isSet,
                  velox::bits::isBitSet(
                      reinterpret_cast<const uint8_t*>(nullsBitmap),
                      i + scatterOffset));
            }
          }

          ASSERT_EQ(rowCount, expectedRow);
        };

        // Test reading all data
        test(rowCount, buffer.data(), nullsPtr, newRowCount, scatterPtr);

        encoding->reset();
        const int firstBlock = newRowCount / 2;

        auto firstBlockSetBits = velox::bits::countBits(
            reinterpret_cast<const uint64_t*>(scatterPtr), 0, firstBlock);

        // Test reading first half of the data
        test(
            firstBlockSetBits, buffer.data(), nullsPtr, firstBlock, scatterPtr);

        const int secondBlock = newRowCount - firstBlock;

        // Test reading second half of the data
        test(
            velox::bits::countBits(
                reinterpret_cast<const uint64_t*>(scatterPtr),
                firstBlock,
                firstBlock + secondBlock),
            buffer.data(),
            nullsPtr,
            secondBlock,
            scatterPtr,
            /* scatterOffset */ firstBlock,
            /* expectedOffset */ firstBlockSetBits);

        encoding->reset();
        uint32_t expectedRow = 0;
        for (int i = 0; i < rowCount; ++i) {
          // Note: Internally, some bit implementations use word boundaries to
          // efficiently iterate on bitmaps. If the buffer doesn't end on a word
          // boundary, this leads to ASAN buffer overflow (debug builds). So for
          // now, we are using uint64_t as the bitmap to make sure the buffer
          // ends on a word boundary.
          auto scatterStart = scatterSizes[i];
          auto scatterSize = scatterSizes[i + 1] - scatterStart;

          // Test reading one item at a time
          test(
              1,
              buffer.data(),
              nullsPtr,
              scatterSize,
              scatterPtr,
              /* scatterOffset */ scatterStart,
              /* expectedOffset */ expectedRow);

          ++expectedRow;
        }

        encoding->reset();
        expectedRow = 0;
        int start = 0;
        int len = 0;
        while (true) {
          start += len;
          len += 1;
          if (start + len > rowCount) {
            break;
          }
          auto scatterStart = scatterSizes[start];
          auto scatterSize = scatterSizes[start + len] - scatterStart;

          // Test reading different ranges of data
          test(
              len,
              buffer.data(),
              nullsPtr,
              scatterSize,
              scatterPtr,
              /* scatterOffset */ scatterStart,
              /* expectedOffset */ expectedRow);

          expectedRow += len;
        }
      }
    }
  }
}

TYPED_TEST(NullableEncodingTest, materializeNullable) {
  using E = typename TypeParam::encoding_type::cppDataType;
  const nimble::Encoding::Options options{
      .useVarintRowCount = TypeParam::useVarint};

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (auto nullPattern :
       {NullsPattern::None, NullsPattern::All, NullsPattern::Random}) {
    for (int run = 0; run < this->kNumRandomRuns; ++run) {
      const std::vector<nimble::Vector<E>> dataPatterns =
          this->util_->template makeDataPatterns<E>(
              rng, this->kMaxRows, this->buffer_.get());
      for (const auto& data : dataPatterns) {
        const nimble::Vector<bool> nulls =
            this->makeRandomNulls(rng, data.size(), nullPattern);
        const nimble::Vector<E> spreadData =
            spreadNullsIntoData<E>(*this->pool_, data, nulls);

        std::vector<velox::BufferPtr> newStringBuffers;
        const auto stringBufferFactory = [&](uint32_t totalLength) {
          auto& buffer = newStringBuffers.emplace_back(
              velox::AlignedBuffer::allocate<char>(
                  totalLength, this->pool_.get()));
          return buffer->template asMutable<void>();
        };
        auto encoding = nimble::test::Encoder<nimble::NullableEncoding<E>>::
            createNullableEncoding(
                *this->buffer_,
                data,
                nulls,
                stringBufferFactory,
                nimble::CompressionType::Uncompressed,
                options);
        ASSERT_TRUE(encoding->isNullable());
        const uint32_t rowCount = encoding->rowCount();
        nimble::Vector<E> buffer(this->pool_.get(), rowCount);
        nimble::Vector<char> bitmap(this->pool_.get(), rowCount);

        auto nonNullCount = encoding->materializeNullable(
            rowCount, buffer.data(), [&]() { return bitmap.data(); });
        EXPECT_EQ(
            std::accumulate(nulls.data(), nulls.data() + nulls.size(), 0),
            nonNullCount);

        for (int i = 0; i < rowCount; ++i) {
          checkOutput(
              i,
              nulls.data(),
              spreadData.data(),
              bitmap.data(),
              buffer.data(),
              nonNullCount != rowCount);
        }

        encoding->reset();
        const int firstBlock = rowCount / 2;
        nonNullCount = encoding->materializeNullable(
            firstBlock, buffer.data(), [&]() { return bitmap.data(); });
        EXPECT_EQ(
            std::accumulate(nulls.data(), nulls.data() + firstBlock, 0),
            nonNullCount);

        for (int i = 0; i < firstBlock; ++i) {
          checkOutput(
              i,
              nulls.data(),
              spreadData.data(),
              bitmap.data(),
              buffer.data(),
              nonNullCount != firstBlock);
        }
        const int secondBlock = rowCount - firstBlock;
        nonNullCount = encoding->materializeNullable(
            secondBlock, buffer.data(), [&]() { return bitmap.data(); });
        EXPECT_EQ(
            std::accumulate(
                nulls.data() + firstBlock, nulls.data() + rowCount, 0),
            nonNullCount);

        for (int i = 0; i < secondBlock; ++i) {
          checkOutput(
              i,
              nulls.data() + firstBlock,
              spreadData.data() + firstBlock,
              bitmap.data(),
              buffer.data(),
              nonNullCount != secondBlock);
        }

        encoding->reset();
        for (int i = 0; i < rowCount; ++i) {
          nonNullCount = encoding->materializeNullable(
              1, buffer.data(), [&]() { return bitmap.data(); });
          checkOutput(
              0,
              nulls.data() + i,
              spreadData.data() + i,
              bitmap.data(),
              buffer.data(),
              nonNullCount == 0);
        }

        encoding->reset();
        int start = 0;
        int len = 0;
        for (int i = 0; i < rowCount; ++i) {
          start += len;
          len += 1;
          if (start + len > rowCount) {
            break;
          }
          nonNullCount = encoding->materializeNullable(
              len, buffer.data(), [&]() { return bitmap.data(); });
          EXPECT_EQ(
              std::accumulate(
                  nulls.data() + start, nulls.data() + start + len, 0),
              nonNullCount);
          for (int j = 0; j < len; ++j) {
            checkOutput(
                j,
                nulls.data() + start,
                spreadData.data() + start,
                bitmap.data(),
                buffer.data(),
                nonNullCount != len);
          }
        }

        const uint32_t offset = folly::Random::rand32(rng) % data.size();
        const uint32_t length =
            1 + folly::Random::rand32(rng) % (data.size() - offset);
        encoding->reset();
        encoding->skip(offset);
        nonNullCount = encoding->materializeNullable(
            length, buffer.data(), [&]() { return bitmap.data(); });
        EXPECT_EQ(
            std::accumulate(
                nulls.data() + offset, nulls.data() + offset + length, 0),
            nonNullCount);
        for (uint32_t i = 0; i < length; ++i) {
          checkOutput(
              i,
              nulls.data() + offset,
              spreadData.data() + offset,
              bitmap.data(),
              buffer.data(),
              nonNullCount != length);
        }
      }
    }
  }
}
