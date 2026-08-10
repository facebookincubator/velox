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
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <span>
#include <vector>
#include "folly/Random.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingUtils.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

// Tests the Encoding API for all basic Encoding implementations and data types.
//
// Nullable data will be tested via a separate file, as will structured data,
// and any data-type-specific functions (e.g. for there are some Encoding calls
// only valid for bools, and those are testing in BoolEncodingTests.cpp).
//
// The tests are templated so as to cover all data types and encoding types with
// a single code path.

using namespace facebook;

namespace {

class TestCompressPolicy : public nimble::CompressionPolicy {
 public:
  explicit TestCompressPolicy(bool compress, bool useVariableBitWidthCompressor)
      : compress_{compress},
        useVariableBitWidthCompressor_{useVariableBitWidthCompressor} {}

  nimble::CompressionConfig config() const override {
    if (!compress_) {
      return {.compressionType = nimble::CompressionType::Uncompressed};
    }

    nimble::CompressionConfig information{
        .compressionType = nimble::CompressionType::MetaInternal};
    information.parameters.metaInternal.compressionLevel = 9;
    information.parameters.metaInternal.decompressionLevel = 2;
    information.parameters.metaInternal.useVariableBitWidthCompressor =
        useVariableBitWidthCompressor_;
    return information;
  }

  virtual bool shouldAccept(
      nimble::CompressionType /* compressionType */,
      uint64_t /* uncompressedSize */,
      uint64_t /* compressedSize */) const override {
    return true;
  }

 private:
  bool compress_;
  bool useVariableBitWidthCompressor_;
};

std::string makeEncodingData() {
  std::string data(nimble::EncodingPrefix::kFixedPrefixSize, '\0');
  auto* pos = data.data();
  nimble::EncodingPrefix::serialize(
      nimble::EncodingType::Trivial,
      nimble::DataType::Int8,
      /*rowCount=*/0,
      /*useVarint=*/false,
      pos);
  return data;
}

class TestEncoding : public nimble::Encoding {
 public:
  TestEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const Options& options)
      : Encoding(pool, data, options) {}

  velox::BufferPtr testingGetBuffer(size_t bytes) {
    return getBuffer(bytes);
  }

  template <typename V>
  nimble::Vector<V> testingGetVectorBuffer() {
    return getVectorBuffer<V>();
  }

  void testingReleaseVectorBuffer(auto& vector) {
    releaseVectorBuffer(vector);
  }

  void reset() override {}

  void skip(uint32_t /*rowCount*/) override {}

  void materialize(uint32_t /*rowCount*/, void* /*buffer*/) override {}

  uint32_t materializeNullable(
      uint32_t /*rowCount*/,
      void* /*buffer*/,
      std::function<void*()> /*nulls*/,
      const velox::bits::Bitmap* /*scatterOutputBitmap*/,
      uint32_t /*offset*/) override {
    return 0;
  }
};

template <typename T>
class TestTrivialEncodingSelectionPolicy
    : public nimble::EncodingSelectionPolicy<T> {
  using physicalType = typename nimble::TypeTraits<T>::physicalType;

 public:
  explicit TestTrivialEncodingSelectionPolicy(
      bool shouldCompress,
      bool useVariableBitWidthCompressor)
      : shouldCompress_{shouldCompress},
        useVariableBitWidthCompressor_{useVariableBitWidthCompressor} {}

  nimble::EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const nimble::Encoding::Options& /* options */) override {
    return {
        .encodingType = nimble::EncodingType::Trivial,
        .compressionPolicyFactory = [this]() {
          return std::make_unique<TestCompressPolicy>(
              shouldCompress_, useVariableBitWidthCompressor_);
        }};
  }

  nimble::EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const nimble::Encoding::Options& /* options */) override {
    return {
        .encodingType = nimble::EncodingType::Nullable,
    };
  }

  std::unique_ptr<nimble::EncodingSelectionPolicyBase> createImpl(
      nimble::EncodingType /* encodingType */,
      nimble::NestedEncodingIdentifier /* identifier */,
      nimble::DataType type) override {
    UNIQUE_PTR_FACTORY(
        type,
        TestTrivialEncodingSelectionPolicy,
        shouldCompress_,
        useVariableBitWidthCompressor_);
  }

 private:
  bool shouldCompress_;
  bool useVariableBitWidthCompressor_;
};

template <typename Encoding>
struct EncodingTypeTraits;

template <typename T>
struct EncodingTypeTraits<nimble::ConstantEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::Constant;
};

template <typename T>
struct EncodingTypeTraits<nimble::DictionaryEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::Dictionary;
};

template <typename T>
struct EncodingTypeTraits<nimble::FixedBitWidthEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::FixedBitWidth;
};

template <typename T>
struct EncodingTypeTraits<nimble::MainlyConstantEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::MainlyConstant;
};

template <typename T>
struct EncodingTypeTraits<nimble::RLEEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::RLE;
};

template <>
struct EncodingTypeTraits<nimble::SparseBoolEncoding> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::SparseBool;
};

template <typename T>
struct EncodingTypeTraits<nimble::TrivialEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::Trivial;
};

template <typename T>
struct EncodingTypeTraits<nimble::VarintEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::Varint;
};

} // namespace

TEST(EncodingBufferPoolTest, getBufferUsesMinimumCapacity) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};
  const auto data = makeEncodingData();
  TestEncoding encoding{
      *pool,
      data,
      nimble::Encoding::Options{
          .useVarintRowCount = false,
          .bufferPool = &bufferPool,
      }};

  auto smallBuffer = velox::AlignedBuffer::allocate<char>(128, pool.get());
  auto* smallBufferPtr = smallBuffer.get();
  bufferPool.release(std::move(smallBuffer));

  auto allocatedBuffer = encoding.testingGetBuffer(1'024);
  ASSERT_NE(allocatedBuffer, nullptr);
  EXPECT_GE(allocatedBuffer->capacity(), 1'024);
  EXPECT_EQ(bufferPool.size(), 1);

  auto remainingSmallBuffer = bufferPool.get();
  ASSERT_NE(remainingSmallBuffer, nullptr);
  EXPECT_EQ(remainingSmallBuffer.get(), smallBufferPtr);
  bufferPool.release(std::move(remainingSmallBuffer));

  auto largeBuffer = velox::AlignedBuffer::allocate<char>(4'096, pool.get());
  auto* largeBufferPtr = largeBuffer.get();
  bufferPool.release(std::move(largeBuffer));

  auto pooledBuffer = encoding.testingGetBuffer(1'024);
  ASSERT_NE(pooledBuffer, nullptr);
  EXPECT_EQ(pooledBuffer.get(), largeBufferPtr);
  EXPECT_EQ(bufferPool.size(), 1);
}

TEST(EncodingBufferPoolTest, getVectorBufferFromPool) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};
  const auto data = makeEncodingData();
  TestEncoding encoding{
      *pool,
      data,
      nimble::Encoding::Options{
          .useVarintRowCount = false,
          .bufferPool = &bufferPool,
      }};

  auto buffer = velox::AlignedBuffer::allocate<char>(4'096, pool.get());
  auto* bufferPtr = buffer.get();
  bufferPool.release(std::move(buffer));
  ASSERT_EQ(bufferPool.size(), 1);

  auto vector = encoding.testingGetVectorBuffer<int32_t>();
  EXPECT_EQ(vector.testingBuffer().get(), bufferPtr);
  EXPECT_GE(vector.capacity(), 4'096 / sizeof(int32_t));
  EXPECT_EQ(vector.size(), 0);
  EXPECT_EQ(bufferPool.size(), 0);
}

TEST(EncodingBufferPoolTest, getVectorBufferWithoutPool) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  const auto data = makeEncodingData();
  TestEncoding encoding{
      *pool, data, nimble::Encoding::Options{.useVarintRowCount = false}};

  auto vector = encoding.testingGetVectorBuffer<int32_t>();
  EXPECT_EQ(vector.size(), 0);
  EXPECT_EQ(vector.capacity(), 0);
}

TEST(EncodingBufferPoolTest, getVectorBufferFromEmptyPool) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};
  const auto data = makeEncodingData();
  TestEncoding encoding{
      *pool,
      data,
      nimble::Encoding::Options{
          .useVarintRowCount = false,
          .bufferPool = &bufferPool,
      }};

  ASSERT_EQ(bufferPool.size(), 0);

  auto vector = encoding.testingGetVectorBuffer<int32_t>();
  EXPECT_EQ(vector.size(), 0);
  EXPECT_EQ(vector.capacity(), 0);
  EXPECT_EQ(bufferPool.size(), 0);
}

TEST(EncodingBufferPoolTest, releaseVectorBufferToPool) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};
  const auto data = makeEncodingData();
  TestEncoding encoding{
      *pool,
      data,
      nimble::Encoding::Options{
          .useVarintRowCount = false,
          .bufferPool = &bufferPool,
      }};

  auto vector = encoding.testingGetVectorBuffer<int32_t>();
  vector.resize(100);
  auto* bufferPtr = vector.testingBuffer().get();
  ASSERT_EQ(bufferPool.size(), 0);

  encoding.testingReleaseVectorBuffer(vector);
  EXPECT_EQ(bufferPool.size(), 1);

  auto recycled = bufferPool.get();
  ASSERT_NE(recycled, nullptr);
  EXPECT_EQ(recycled.get(), bufferPtr);
}

TEST(EncodingBufferPoolTest, releaseVectorBufferWithoutPool) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  const auto data = makeEncodingData();
  TestEncoding encoding{
      *pool, data, nimble::Encoding::Options{.useVarintRowCount = false}};

  auto vector = encoding.testingGetVectorBuffer<int32_t>();
  vector.resize(100);

  encoding.testingReleaseVectorBuffer(vector);
  EXPECT_NE(vector.testingBuffer(), nullptr);
}

TEST(EncodingBufferPoolTest, getVectorBufferRoundTrip) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};
  const auto data = makeEncodingData();
  TestEncoding encoding{
      *pool,
      data,
      nimble::Encoding::Options{
          .useVarintRowCount = false,
          .bufferPool = &bufferPool,
      }};

  auto vector1 = encoding.testingGetVectorBuffer<int32_t>();
  vector1.resize(100);
  auto* bufferPtr = vector1.testingBuffer().get();

  encoding.testingReleaseVectorBuffer(vector1);
  ASSERT_EQ(bufferPool.size(), 1);

  auto vector2 = encoding.testingGetVectorBuffer<int32_t>();
  EXPECT_EQ(vector2.testingBuffer().get(), bufferPtr);
  EXPECT_EQ(bufferPool.size(), 0);
}

template <typename EncodingType, bool UseVarint>
struct TestConfig {
  using encoding_type = EncodingType;
  static constexpr bool useVarint = UseVarint;
};

#define TC(T) TestConfig<T, false>, TestConfig<T, true>

// Helper template to get encoding type for each encoding class
template <typename Encoding>
struct EncodingTypeGetter;

template <typename T>
struct EncodingTypeGetter<nimble::ConstantEncoding<T>> {
  static constexpr nimble::EncodingType value = nimble::EncodingType::Constant;
};

template <typename T>
struct EncodingTypeGetter<nimble::DictionaryEncoding<T>> {
  static constexpr nimble::EncodingType value =
      nimble::EncodingType::Dictionary;
};

template <typename T>
struct EncodingTypeGetter<nimble::FixedBitWidthEncoding<T>> {
  static constexpr nimble::EncodingType value =
      nimble::EncodingType::FixedBitWidth;
};

template <typename T>
struct EncodingTypeGetter<nimble::FrequencyPartitionEncoding<T>> {
  static constexpr nimble::EncodingType value =
      nimble::EncodingType::FrequencyPartition;
};

template <typename T>
struct EncodingTypeGetter<nimble::MainlyConstantEncoding<T>> {
  static constexpr nimble::EncodingType value =
      nimble::EncodingType::MainlyConstant;
};

template <typename T>
struct EncodingTypeGetter<nimble::RLEEncoding<T>> {
  static constexpr nimble::EncodingType value = nimble::EncodingType::RLE;
};

template <>
struct EncodingTypeGetter<nimble::SparseBoolEncoding> {
  static constexpr nimble::EncodingType value =
      nimble::EncodingType::SparseBool;
};

template <typename T>
struct EncodingTypeGetter<nimble::TrivialEncoding<T>> {
  static constexpr nimble::EncodingType value = nimble::EncodingType::Trivial;
};

template <typename T>
struct EncodingTypeGetter<nimble::VarintEncoding<T>> {
  static constexpr nimble::EncodingType value = nimble::EncodingType::Varint;
};

// C is the encoding type.
template <typename Config>
class EncodingTest : public ::testing::Test {
 protected:
  using C = typename Config::encoding_type;
  using E = typename C::cppDataType;

  void SetUp() override {
    pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*this->pool_);
    util_ = std::make_unique<nimble::testing::Util>(*this->pool_);
  }

  std::unique_ptr<nimble::Encoding> createEncoding(
      const nimble::Vector<E>& values,
      bool compress,
      bool useVariableBitWidthCompressor,
      std::function<void*(uint32_t)> stringBufferFactory) {
    constexpr bool useVarint = Config::useVarint;
    using physicalType = typename nimble::TypeTraits<E>::physicalType;
    auto physicalValues = std::span<const physicalType>(
        reinterpret_cast<const physicalType*>(values.data()), values.size());
    nimble::EncodingSelection<physicalType> selection{
        {.encodingType = EncodingTypeGetter<C>::value,
         .compressionPolicyFactory =
             [compress, useVariableBitWidthCompressor]() {
               return std::make_unique<TestCompressPolicy>(
                   compress, useVariableBitWidthCompressor);
             }},
        nimble::Statistics<physicalType>::create(physicalValues),
        std::make_unique<TestTrivialEncodingSelectionPolicy<E>>(
            compress, useVariableBitWidthCompressor)};

    nimble::Encoding::Options options{.useVarintRowCount = useVarint};
    auto encoded = C::encode(selection, physicalValues, *buffer_, options);
    return std::make_unique<C>(
        *this->pool_, encoded, stringBufferFactory, options);
  }

  // Each unit test runs on randomized data this many times before
  // we conclude the unit test passed.
  static constexpr int32_t kNumRandomRuns = 10;

  // We want the number of row tested to potentially be large compared to a
  // skip block. When we actually generate data we pick a random length between
  // 1 and this size.
  static constexpr int32_t kMaxRows = 2000;

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
  std::unique_ptr<nimble::testing::Util> util_;
};

#define VARINT_TYPES(EncodingName)                            \
  TC(EncodingName<int32_t>), TC(EncodingName<int64_t>),       \
      TC(EncodingName<uint32_t>), TC(EncodingName<uint64_t>), \
      TC(EncodingName<float>), TC(EncodingName<double>)

#define NUMERIC_TYPES(EncodingName)                         \
  VARINT_TYPES(EncodingName), TC(EncodingName<int8_t>),     \
      TC(EncodingName<uint8_t>), TC(EncodingName<int16_t>), \
      TC(EncodingName<uint16_t>)

#define NON_BOOL_TYPES(EncodingName) \
  NUMERIC_TYPES(EncodingName), TC(EncodingName<std::string_view>)

#define ALL_TYPES(EncodingName) \
  NON_BOOL_TYPES(EncodingName), TC(EncodingName<bool>)

using TestTypes = ::testing::Types<
    TC(nimble::SparseBoolEncoding),
    VARINT_TYPES(nimble::VarintEncoding),
    NUMERIC_TYPES(nimble::FixedBitWidthEncoding),
    NON_BOOL_TYPES(nimble::DictionaryEncoding),
    NON_BOOL_TYPES(nimble::MainlyConstantEncoding),
    ALL_TYPES(nimble::TrivialEncoding),
    ALL_TYPES(nimble::RLEEncoding),
    ALL_TYPES(nimble::ConstantEncoding)>;

TYPED_TEST_CASE(EncodingTest, TestTypes);

// Regression test for SparseBoolEncoding null pointer crash when rowCount == 0.
// memset(nullptr, 0, 0) is undefined behavior per the C/C++ standard.
class SparseBoolZeroRowTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  std::function<void*(uint32_t)> makeStringBufferFactory() {
    return [this](uint32_t totalLength) {
      auto& buf = stringBuffers_.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
      return buf->asMutable<void>();
    };
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
  std::vector<velox::BufferPtr> stringBuffers_;
};

TEST_F(SparseBoolZeroRowTest, materializeZeroRows) {
  // Create encoding with sparse true values (few trues among mostly false).
  nimble::Vector<bool> values{pool_.get()};
  for (int i = 0; i < 100; ++i) {
    values.push_back(i == 50);
  }

  auto encoding = facebook::nimble::test::Encoder<nimble::SparseBoolEncoding>::
      createEncoding(*buffer_, values, makeStringBufferFactory());

  // Materializing zero rows with nullptr must not crash.
  encoding->materialize(0, nullptr);

  // Encoding should still work correctly after zero-row materialize.
  nimble::Vector<bool> result(pool_.get(), 100);
  encoding->materialize(100, result.data());
  for (int i = 0; i < 100; ++i) {
    ASSERT_EQ(result[i], values[i]) << "i: " << i;
  }
}

TEST_F(SparseBoolZeroRowTest, materializeZeroRowsMidStream) {
  nimble::Vector<bool> values{pool_.get()};
  for (int i = 0; i < 100; ++i) {
    values.push_back(i == 50);
  }

  auto encoding = facebook::nimble::test::Encoder<nimble::SparseBoolEncoding>::
      createEncoding(*buffer_, values, makeStringBufferFactory());

  // Read some rows, then do a zero-row materialize, then continue reading.
  nimble::Vector<bool> result(pool_.get(), 100);
  encoding->materialize(30, result.data());
  encoding->materialize(0, nullptr);
  encoding->materialize(70, result.data());
  for (int i = 0; i < 70; ++i) {
    ASSERT_EQ(result[i], values[30 + i]) << "i: " << i;
  }
}

TEST_F(SparseBoolZeroRowTest, materializeBoolsAsBitsZeroRows) {
  nimble::Vector<bool> values{pool_.get()};
  for (int i = 0; i < 100; ++i) {
    values.push_back(i == 50);
  }

  auto encoding = facebook::nimble::test::Encoder<nimble::SparseBoolEncoding>::
      createEncoding(*buffer_, values, makeStringBufferFactory());
  auto* sparseBool = dynamic_cast<nimble::SparseBoolEncoding*>(encoding.get());
  ASSERT_NE(sparseBool, nullptr);

  // materializeBoolsAsBits with zero rows and nullptr must not crash.
  sparseBool->materializeBoolsAsBits(0, nullptr, 0);

  // Should still work correctly after the zero-row call.
  uint64_t bits[2] = {};
  sparseBool->materializeBoolsAsBits(100, bits, 0);
  for (int i = 0; i < 100; ++i) {
    ASSERT_EQ(velox::bits::isBitSet(bits, i), values[i]) << "i: " << i;
  }
}

TEST_F(SparseBoolZeroRowTest, materializeZeroRowsSparseValueFalse) {
  // Create encoding with sparse false values (few falses among mostly true).
  nimble::Vector<bool> values{pool_.get()};
  for (int i = 0; i < 100; ++i) {
    values.push_back(i != 50);
  }

  auto encoding = facebook::nimble::test::Encoder<nimble::SparseBoolEncoding>::
      createEncoding(*buffer_, values, makeStringBufferFactory());

  // Materializing zero rows with nullptr must not crash.
  encoding->materialize(0, nullptr);

  nimble::Vector<bool> result(pool_.get(), 100);
  encoding->materialize(100, result.data());
  for (int i = 0; i < 100; ++i) {
    ASSERT_EQ(result[i], values[i]) << "i: " << i;
  }
}

// Verifies materializeBoolsAsBits after skipping to an exact run boundary
// in RLE<bool>. The lazy pattern means copiesRemaining_==0 after the skip,
// and materializeBoolsAsBits must load the next run on demand.
TEST_F(SparseBoolZeroRowTest, rleBoolMaterializeAfterSkip) {
  // 3 runs: [true x 5] [false x 3] [true x 4]
  nimble::Vector<bool> values{pool_.get()};
  for (int i = 0; i < 5; ++i) {
    values.push_back(true);
  }
  for (int i = 0; i < 3; ++i) {
    values.push_back(false);
  }
  for (int i = 0; i < 4; ++i) {
    values.push_back(true);
  }

  auto encoding =
      nimble::test::Encoder<nimble::RLEEncoding<bool>>::createEncoding(
          *buffer_, values, makeStringBufferFactory());

  // Skip exactly past run 1 (5 rows).
  encoding->skip(5);

  // materializeBoolsAsBits for runs 2+3 (7 rows).
  const auto remaining = values.size() - 5;
  uint64_t bits = 0;
  encoding->materializeBoolsAsBits(remaining, &bits, 0);
  for (size_t i = 0; i < remaining; ++i) {
    EXPECT_EQ(velox::bits::isBitSet(&bits, i), values[5 + i]) << "bit " << i;
  }
}

TYPED_TEST(EncodingTest, materialize) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  using E = typename TypeParam::encoding_type::cppDataType;

  for (int run = 0; run < this->kNumRandomRuns; ++run) {
    const std::vector<nimble::Vector<E>> dataPatterns =
        this->util_->template makeDataPatterns<E>(
            rng, this->kMaxRows, this->buffer_.get());
    for (const auto& data : dataPatterns) {
      for (auto compress : {false, true}) {
        for (auto useVariableBitWidthCompressor : {false, true}) {
          SCOPED_TRACE(
              fmt::format(
                  "compress {}, variableBitWidth {}",
                  compress,
                  useVariableBitWidthCompressor));
          const int rowCount = data.size();
          ASSERT_GT(rowCount, 0);
          std::unique_ptr<nimble::Encoding> encoding;
          std::vector<velox::BufferPtr> newStringBuffers;
          const auto stringBufferFactory = [&](uint32_t totalLength) {
            auto& buffer = newStringBuffers.emplace_back(
                velox::AlignedBuffer::allocate<char>(
                    totalLength, this->pool_.get()));
            return buffer->template asMutable<void>();
          };
          try {
            encoding = this->createEncoding(
                data,
                compress,
                useVariableBitWidthCompressor,
                stringBufferFactory);
          } catch (const nimble::NimbleUserError& e) {
            if (e.errorCode() == nimble::error_code::IncompatibleEncoding) {
              continue;
            }
            throw;
          }
          ASSERT_EQ(encoding->dataType(), nimble::TypeTraits<E>::dataType);
          ASSERT_EQ(encoding->rowCount(), rowCount);
          nimble::Vector<E> buffer(this->pool_.get(), rowCount);

          encoding->materialize(rowCount, buffer.data());
          for (int i = 0; i < rowCount; ++i) {
            ASSERT_EQ(buffer[i], data[i]) << "i: " << i;
          }

          encoding->reset();
          const int firstBlock = rowCount / 2;
          encoding->materialize(firstBlock, buffer.data());
          for (int i = 0; i < firstBlock; ++i) {
            ASSERT_EQ(buffer[i], data[i]);
          }
          const int secondBlock = rowCount - firstBlock;
          encoding->materialize(secondBlock, buffer.data());
          for (int i = 0; i < secondBlock; ++i) {
            ASSERT_EQ(buffer[i], data[firstBlock + i]);
          }

          encoding->reset();
          for (int i = 0; i < rowCount; ++i) {
            encoding->materialize(1, buffer.data());
            ASSERT_EQ(buffer[0], data[i]);
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
              ASSERT_EQ(data[start + j], buffer[j]);
            }
          }

          const uint32_t offset = folly::Random::rand32(rng) % data.size();
          const uint32_t length =
              1 + folly::Random::rand32(rng) % (data.size() - offset);
          encoding->reset();
          encoding->skip(offset);
          encoding->materialize(length, buffer.data());
          for (uint32_t i = 0; i < length; ++i) {
            ASSERT_EQ(buffer[i], data[offset + i]);
          }
        }
      }
    }
  }
}

template <typename T>
void checkScatteredOutput(
    bool hasNulls,
    size_t index,
    const bool* scatter,
    const T* actual,
    const T* expected,
    const char* nulls,
    uint32_t& expectedRow) {
  if (scatter[index]) {
    ASSERT_EQ(actual[index], expected[expectedRow]);
    ++expectedRow;
  }

  if (hasNulls) {
    ASSERT_EQ(
        scatter[index],
        velox::bits::isBitSet(reinterpret_cast<const uint8_t*>(nulls), index));
  }
}

TYPED_TEST(EncodingTest, scatteredMaterialize) {
  using E = typename TypeParam::encoding_type::cppDataType;

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (int run = 0; run < this->kNumRandomRuns; ++run) {
    const std::vector<nimble::Vector<E>> dataPatterns =
        this->util_->template makeDataPatterns<E>(
            rng, this->kMaxRows, this->buffer_.get());
    for (const auto& data : dataPatterns) {
      for (auto compress : {false, true}) {
        for (auto useVariableBitWidthCompressor : {false, true}) {
          const int rowCount = data.size();
          ASSERT_GT(rowCount, 0);
          std::unique_ptr<nimble::Encoding> encoding;
          std::vector<velox::BufferPtr> newStringBuffers;
          const auto stringBufferFactory = [&](uint32_t totalLength) {
            auto& buffer = newStringBuffers.emplace_back(
                velox::AlignedBuffer::allocate<char>(
                    totalLength, this->pool_.get()));
            return buffer->template asMutable<void>();
          };
          try {
            encoding = this->createEncoding(
                data,
                compress,
                useVariableBitWidthCompressor,
                stringBufferFactory);
          } catch (const nimble::NimbleUserError& e) {
            if (e.errorCode() == nimble::error_code::IncompatibleEncoding) {
              continue;
            }
            throw;
          }
          ASSERT_EQ(encoding->dataType(), nimble::TypeTraits<E>::dataType);

          int setBits = 0;
          std::vector<int32_t> scatterSizes(rowCount + 1);
          scatterSizes[0] = 0;
          nimble::Vector<bool> scatter(this->pool_.get());
          while (setBits < rowCount) {
            scatter.push_back(folly::Random::oneIn(2, rng));
            if (scatter.back()) {
              scatterSizes[++setBits] = scatter.size();
            }
          }

          auto newRowCount = scatter.size();
          auto requiredBytes = velox::bits::nbytes(newRowCount);
          // Note: Internally, some bit implementations use word boundaries to
          // efficiently iterate on bitmaps. If the buffer doesn't end on a
          // word boundary, this leads to ASAN buffer overflow (debug builds).
          // So for now, we are allocating extra 7 bytes to make sure the
          // buffer ends or exceeds a word boundary.
          nimble::Buffer scatterBuffer{*this->pool_, requiredBytes + 7};
          nimble::Buffer nullsBuffer{*this->pool_, requiredBytes + 7};
          auto scatterPtr = scatterBuffer.reserve(requiredBytes);
          auto nullsPtr = nullsBuffer.reserve(requiredBytes);
          memset(scatterPtr, 0, requiredBytes);
          velox::bits::packBitmap(scatter, scatterPtr);

          nimble::Vector<E> buffer(this->pool_.get(), newRowCount);

          uint32_t expectedRow = 0;
          uint32_t actualRows = 0;
          {
            velox::bits::Bitmap scatterBitmap(
                scatterPtr, static_cast<uint32_t>(newRowCount));
            actualRows = encoding->materializeNullable(
                rowCount,
                buffer.data(),
                [&]() { return nullsPtr; },
                &scatterBitmap);
          }
          ASSERT_EQ(actualRows, rowCount);
          auto hasNulls = actualRows != newRowCount;
          for (int i = 0; i < newRowCount; ++i) {
            checkScatteredOutput(
                hasNulls,
                i,
                scatter.data(),
                buffer.data(),
                data.data(),
                nullsPtr,
                expectedRow);
          }
          EXPECT_EQ(rowCount, expectedRow);

          encoding->reset();
          const int firstBlock = rowCount / 2;
          const int firstScatterSize = scatterSizes[firstBlock];
          expectedRow = 0;
          {
            velox::bits::Bitmap scatterBitmap(scatterPtr, firstScatterSize);
            actualRows = encoding->materializeNullable(
                firstBlock,
                buffer.data(),
                [&]() { return nullsPtr; },
                &scatterBitmap);
          }
          ASSERT_EQ(actualRows, firstBlock);
          hasNulls = actualRows != firstScatterSize;
          for (int i = 0; i < firstScatterSize; ++i) {
            checkScatteredOutput(
                hasNulls,
                i,
                scatter.data(),
                buffer.data(),
                data.data(),
                nullsPtr,
                expectedRow);
          }
          ASSERT_EQ(actualRows, expectedRow);

          const int secondBlock = rowCount - firstBlock;
          const int secondScatterSize = scatter.size() - firstScatterSize;
          expectedRow = actualRows;
          {
            velox::bits::Bitmap scatterBitmap(
                scatterPtr, static_cast<uint32_t>(newRowCount));
            actualRows = encoding->materializeNullable(
                secondBlock,
                buffer.data(),
                [&]() { return nullsPtr; },
                &scatterBitmap,
                firstScatterSize);
          }
          ASSERT_EQ(actualRows, secondBlock);
          hasNulls = actualRows != secondScatterSize;
          auto previousRows = expectedRow;
          for (int i = firstScatterSize; i < newRowCount; ++i) {
            checkScatteredOutput(
                hasNulls,
                i,
                scatter.data(),
                buffer.data(),
                data.data(),
                nullsPtr,
                expectedRow);
          }
          ASSERT_EQ(actualRows, expectedRow - previousRows);
          ASSERT_EQ(rowCount, expectedRow);

          encoding->reset();
          expectedRow = 0;
          for (int i = 0; i < rowCount; ++i) {
            // Note: Internally, some bit implementations use word boundaries
            // to efficiently iterate on bitmaps. If the buffer doesn't end on
            // a word boundary, this leads to ASAN buffer overflow (debug
            // builds). So for now, we are using uint64_t as the bitmap to
            // make sure the buffer ends on a word boundary.
            auto scatterStart = scatterSizes[i];
            auto scatterEnd = scatterSizes[i + 1];
            {
              velox::bits::Bitmap scatterBitmap(scatterPtr, scatterEnd);
              actualRows = encoding->materializeNullable(
                  1,
                  buffer.data(),
                  [&]() { return nullsPtr; },
                  &scatterBitmap,
                  scatterStart);
            }
            ASSERT_EQ(actualRows, 1);
            previousRows = expectedRow;
            for (int j = scatterStart; j < scatterEnd; ++j) {
              checkScatteredOutput(
                  scatterEnd - scatterStart > 1,
                  j,
                  scatter.data(),
                  buffer.data(),
                  data.data(),
                  nullsPtr,
                  expectedRow);
            }
            ASSERT_EQ(actualRows, expectedRow - previousRows);
          }
          ASSERT_EQ(rowCount, expectedRow);

          encoding->reset();
          expectedRow = 0;
          int start = 0;
          int len = 0;
          for (int i = 0; i < rowCount; ++i) {
            start += len;
            len += 1;
            if (start + len > rowCount) {
              break;
            }
            auto scatterStart = scatterSizes[start];
            auto scatterEnd = scatterSizes[start + len];
            {
              velox::bits::Bitmap scatterBitmap(scatterPtr, scatterEnd);
              actualRows = encoding->materializeNullable(
                  len,
                  buffer.data(),
                  [&]() { return nullsPtr; },
                  &scatterBitmap,
                  scatterStart);
            }
            ASSERT_EQ(actualRows, len);
            hasNulls = actualRows != scatterEnd - scatterStart;
            previousRows = expectedRow;
            for (int j = scatterStart; j < scatterEnd; ++j) {
              checkScatteredOutput(
                  hasNulls,
                  j,
                  scatter.data(),
                  buffer.data(),
                  data.data(),
                  nullsPtr,
                  expectedRow);
            }
            ASSERT_EQ(actualRows, expectedRow - previousRows);
          }
        }
      }
    }
  }
}

// Templatized dictionary API tests covering all DictionaryEncoding-compatible
// types: numeric types + string_view + bool.
template <typename T>
class DictionaryApiTypedTest : public ::testing::Test {
 protected:
  using PhysicalType = typename nimble::TypeTraits<T>::physicalType;

  void SetUp() override {
    pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  std::unique_ptr<nimble::Encoding> encodeDictionary(
      const std::vector<T>& values) {
    nimble::Vector<T> vec{pool_.get()};
    for (auto v : values) {
      vec.push_back(v);
    }
    auto physicalValues = std::span<const PhysicalType>(
        reinterpret_cast<const PhysicalType*>(vec.data()), vec.size());
    nimble::EncodingSelection<PhysicalType> selection{
        {.encodingType = nimble::EncodingType::Dictionary},
        nimble::Statistics<PhysicalType>::create(physicalValues),
        std::make_unique<TestTrivialEncodingSelectionPolicy<T>>(false, false)};
    auto encoded = nimble::DictionaryEncoding<T>::encode(
        selection, physicalValues, *buffer_);
    return std::make_unique<nimble::DictionaryEncoding<T>>(
        *pool_, encoded, [this](uint32_t size) {
          stringBuffers_.push_back(
              velox::AlignedBuffer::allocate<char>(size, pool_.get()));
          return stringBuffers_.back()->asMutable<void>();
        });
  }

  std::unique_ptr<nimble::Encoding> encodeNullableDictionary(
      const std::vector<T>& values,
      const nimble::Vector<bool>& nulls) {
    NIMBLE_CHECK_EQ(values.size(), nulls.size());
    const uint32_t rowCount = values.size();

    // Encode non-null values as DictionaryEncoding.
    nimble::Vector<T> nonNullVec{pool_.get()};
    for (size_t i = 0; i < values.size(); ++i) {
      if (nulls[i]) {
        nonNullVec.push_back(values[i]);
      }
    }
    auto nonNullSpan = std::span<const PhysicalType>(
        reinterpret_cast<const PhysicalType*>(nonNullVec.data()),
        nonNullVec.size());
    nimble::Buffer valBuffer{*pool_};
    nimble::EncodingSelection<PhysicalType> valSelection{
        {.encodingType = nimble::EncodingType::Dictionary},
        nimble::Statistics<PhysicalType>::create(nonNullSpan),
        std::make_unique<TestTrivialEncodingSelectionPolicy<T>>(false, false)};
    auto serializedValues = nimble::DictionaryEncoding<T>::encode(
        valSelection, nonNullSpan, valBuffer);

    // Encode nulls as TrivialEncoding<bool>.
    auto nullSpan = std::span<const bool>(nulls.data(), nulls.size());
    nimble::Buffer nullBuffer{*pool_};
    nimble::EncodingSelection<bool> nullSelection{
        {.encodingType = nimble::EncodingType::Trivial},
        nimble::Statistics<bool>::create(nullSpan),
        std::make_unique<TestTrivialEncodingSelectionPolicy<bool>>(
            false, false)};
    auto serializedNulls = nimble::TrivialEncoding<bool>::encode(
        nullSelection, nullSpan, nullBuffer);

    // Assemble the NullableEncoding binary format:
    // [EncodingType:1][DataType:1][rowCount:4] | uint32(valSize) | valBytes |
    // nullBytes
    const uint32_t encodingSize = nimble::EncodingPrefix::kFixedPrefixSize + 4 +
        serializedValues.size() + serializedNulls.size();
    char* reserved = buffer_->reserve(encodingSize);
    char* pos = reserved;
    nimble::encoding::writeChar(
        static_cast<char>(nimble::EncodingType::Nullable), pos);
    nimble::encoding::writeChar(
        static_cast<char>(nimble::TypeTraits<T>::dataType), pos);
    nimble::encoding::writeUint32(rowCount, pos);
    nimble::encoding::writeString(serializedValues, pos);
    nimble::encoding::writeBytes(serializedNulls, pos);

    return nimble::EncodingFactory().create(
        *pool_,
        std::string_view(reserved, encodingSize),
        [this](uint32_t size) {
          stringBuffers_.push_back(
              velox::AlignedBuffer::allocate<char>(size, pool_.get()));
          return stringBuffers_.back()->asMutable<void>();
        });
  }

  /// Creates an Encoding from serialized bytes using EncodingFactory.
  std::unique_ptr<nimble::Encoding> decodeEncoding(std::string_view data) {
    nimble::Encoding::Options options{};
    return nimble::EncodingFactory(options).create(
        *pool_, data, [this](uint32_t size) {
          stringBuffers_.push_back(
              velox::AlignedBuffer::allocate<char>(size, pool_.get()));
          return stringBuffers_.back()->asMutable<void>();
        });
  }

  /// Serializes values as MainlyConstant wrapping Constant into outputBuffer.
  /// Requires exactly 2 distinct values. The most frequent becomes the MC
  /// common value; the other becomes a ConstantEncoding leaf.
  std::string_view serializeMainlyConstantWithConstantOthers(
      const std::vector<T>& values,
      nimble::Buffer& outputBuffer) {
    NIMBLE_CHECK(!values.empty());
    const uint32_t rowCount = values.size();

    std::unordered_map<PhysicalType, uint32_t> counts;
    for (const auto& v : values) {
      ++counts[reinterpret_cast<const PhysicalType&>(v)];
    }
    NIMBLE_CHECK_EQ(counts.size(), 2, "Expected exactly 2 distinct values");

    PhysicalType commonValue{};
    uint32_t maxCount = 0;
    for (const auto& [val, cnt] : counts) {
      if (cnt > maxCount) {
        maxCount = cnt;
        commonValue = val;
      }
    }

    nimble::Vector<bool> isCommon{pool_.get(), rowCount, false};
    nimble::Vector<T> otherVec{pool_.get()};
    for (size_t i = 0; i < values.size(); ++i) {
      if (reinterpret_cast<const PhysicalType&>(values[i]) == commonValue) {
        isCommon[i] = true;
      } else {
        otherVec.push_back(values[i]);
      }
    }

    // Encode isCommon as Trivial<bool>.
    auto isCommonSpan = std::span<const bool>(isCommon.data(), isCommon.size());
    nimble::Buffer isCommonBuf{*pool_};
    nimble::EncodingSelection<bool> isCommonSel{
        {.encodingType = nimble::EncodingType::Trivial},
        nimble::Statistics<bool>::create(isCommonSpan),
        std::make_unique<TestTrivialEncodingSelectionPolicy<bool>>(
            false, false)};
    auto serializedIsCommon = nimble::TrivialEncoding<bool>::encode(
        isCommonSel, isCommonSpan, isCommonBuf);

    // Encode other values as Constant.
    auto otherSpan = std::span<const PhysicalType>(
        reinterpret_cast<const PhysicalType*>(otherVec.data()),
        otherVec.size());
    nimble::Buffer otherBuf{*pool_};
    nimble::EncodingSelection<PhysicalType> otherSel{
        {.encodingType = nimble::EncodingType::Constant},
        nimble::Statistics<PhysicalType>::create(otherSpan),
        std::make_unique<TestTrivialEncodingSelectionPolicy<T>>(false, false)};
    auto serializedOther =
        nimble::ConstantEncoding<T>::encode(otherSel, otherSpan, otherBuf);

    // Assemble MainlyConstant binary.
    uint32_t encodingSize = nimble::Encoding::kPrefixSize + 8 +
        serializedIsCommon.size() + serializedOther.size();
    if constexpr (nimble::isNumericType<PhysicalType>()) {
      encodingSize += sizeof(PhysicalType);
    } else {
      encodingSize += 4 + commonValue.size();
    }
    char* reserved = outputBuffer.reserve(encodingSize);
    char* pos = reserved;
    nimble::encoding::writeChar(
        static_cast<char>(nimble::EncodingType::MainlyConstant), pos);
    nimble::encoding::writeChar(
        static_cast<char>(nimble::TypeTraits<T>::dataType), pos);
    nimble::encoding::writeUint32(rowCount, pos);
    nimble::encoding::writeString(serializedIsCommon, pos);
    nimble::encoding::writeString(serializedOther, pos);
    nimble::encoding::write<PhysicalType>(commonValue, pos);

    return std::string_view(reserved, encodingSize);
  }

  /// Serializes values as MainlyConstant wrapping Dictionary into outputBuffer.
  /// The common value is the most frequent element. Returns a string_view of
  /// the serialized bytes in outputBuffer.
  std::string_view serializeMainlyConstantDictionary(
      const std::vector<T>& values,
      nimble::Buffer& outputBuffer) {
    NIMBLE_CHECK(!values.empty());
    const uint32_t rowCount = values.size();

    // Find the most common value.
    std::unordered_map<PhysicalType, uint32_t> counts;
    for (const auto& v : values) {
      ++counts[reinterpret_cast<const PhysicalType&>(v)];
    }
    PhysicalType commonValue{};
    uint32_t maxCount = 0;
    for (const auto& [val, cnt] : counts) {
      if (cnt > maxCount) {
        maxCount = cnt;
        commonValue = val;
      }
    }

    // Build isCommon bools and other (non-common) values.
    nimble::Vector<bool> isCommon{pool_.get(), rowCount, false};
    nimble::Vector<T> otherVec{pool_.get()};
    for (size_t i = 0; i < values.size(); ++i) {
      if (reinterpret_cast<const PhysicalType&>(values[i]) == commonValue) {
        isCommon[i] = true;
      } else {
        otherVec.push_back(values[i]);
      }
    }

    // Encode isCommon as Trivial<bool>.
    auto isCommonSpan = std::span<const bool>(isCommon.data(), isCommon.size());
    nimble::Buffer isCommonBuf{*pool_};
    nimble::EncodingSelection<bool> isCommonSel{
        {.encodingType = nimble::EncodingType::Trivial},
        nimble::Statistics<bool>::create(isCommonSpan),
        std::make_unique<TestTrivialEncodingSelectionPolicy<bool>>(
            false, false)};
    auto serializedIsCommon = nimble::TrivialEncoding<bool>::encode(
        isCommonSel, isCommonSpan, isCommonBuf);

    // Encode other values as Dictionary.
    auto otherSpan = std::span<const PhysicalType>(
        reinterpret_cast<const PhysicalType*>(otherVec.data()),
        otherVec.size());
    nimble::Buffer otherBuf{*pool_};
    nimble::EncodingSelection<PhysicalType> otherSel{
        {.encodingType = nimble::EncodingType::Dictionary},
        nimble::Statistics<PhysicalType>::create(otherSpan),
        std::make_unique<TestTrivialEncodingSelectionPolicy<T>>(false, false)};
    auto serializedOther =
        nimble::DictionaryEncoding<T>::encode(otherSel, otherSpan, otherBuf);

    // Assemble MainlyConstant binary: prefix | isCommon | otherValues |
    // commonValue
    uint32_t encodingSize = nimble::Encoding::kPrefixSize + 8 +
        serializedIsCommon.size() + serializedOther.size();
    if constexpr (nimble::isNumericType<PhysicalType>()) {
      encodingSize += sizeof(PhysicalType);
    } else {
      encodingSize += 4 + commonValue.size();
    }
    char* reserved = outputBuffer.reserve(encodingSize);
    char* pos = reserved;
    nimble::encoding::writeChar(
        static_cast<char>(nimble::EncodingType::MainlyConstant), pos);
    nimble::encoding::writeChar(
        static_cast<char>(nimble::TypeTraits<T>::dataType), pos);
    nimble::encoding::writeUint32(rowCount, pos);
    nimble::encoding::writeString(serializedIsCommon, pos);
    nimble::encoding::writeString(serializedOther, pos);
    nimble::encoding::write<PhysicalType>(commonValue, pos);

    return std::string_view(reserved, encodingSize);
  }

  /// Encodes data as MainlyConstant wrapping Dictionary for the other-values
  /// child. The common value is the most frequent element.
  std::unique_ptr<nimble::Encoding> encodeMainlyConstantDictionary(
      const std::vector<T>& values) {
    auto serialized = serializeMainlyConstantDictionary(values, *buffer_);
    return decodeEncoding(serialized);
  }

  /// Encodes data as Nullable wrapping MainlyConstant wrapping Dictionary.
  std::unique_ptr<nimble::Encoding> encodeNullableMainlyConstantDictionary(
      const std::vector<T>& values,
      const nimble::Vector<bool>& nulls) {
    NIMBLE_CHECK_EQ(values.size(), nulls.size());
    const uint32_t rowCount = values.size();

    // Collect non-null values.
    std::vector<T> nonNullValues;
    for (size_t i = 0; i < values.size(); ++i) {
      if (nulls[i]) {
        nonNullValues.push_back(values[i]);
      }
    }

    // Serialize MC→Dict for non-null values.
    nimble::Buffer mcBuffer{*pool_};
    auto serializedMC =
        serializeMainlyConstantDictionary(nonNullValues, mcBuffer);

    // Encode nulls as Trivial<bool>.
    auto nullSpan = std::span<const bool>(nulls.data(), nulls.size());
    nimble::Buffer nullBuffer{*pool_};
    nimble::EncodingSelection<bool> nullSelection{
        {.encodingType = nimble::EncodingType::Trivial},
        nimble::Statistics<bool>::create(nullSpan),
        std::make_unique<TestTrivialEncodingSelectionPolicy<bool>>(
            false, false)};
    auto serializedNulls = nimble::TrivialEncoding<bool>::encode(
        nullSelection, nullSpan, nullBuffer);

    // Assemble Nullable: prefix | MC bytes | null bytes
    const uint32_t encodingSize = nimble::Encoding::kPrefixSize + 4 +
        serializedMC.size() + serializedNulls.size();
    char* reserved = buffer_->reserve(encodingSize);
    char* pos = reserved;
    nimble::encoding::writeChar(
        static_cast<char>(nimble::EncodingType::Nullable), pos);
    nimble::encoding::writeChar(
        static_cast<char>(nimble::TypeTraits<T>::dataType), pos);
    nimble::encoding::writeUint32(rowCount, pos);
    nimble::encoding::writeString(serializedMC, pos);
    nimble::encoding::writeBytes(serializedNulls, pos);

    return decodeEncoding(std::string_view(reserved, encodingSize));
  }

  /// Serializes values as RLE wrapping Dictionary into outputBuffer.
  /// Computes runs, encodes run lengths as Trivial<uint32_t>, encodes
  /// run values as Dictionary<T>.
  std::string_view serializeRleDictionary(
      const std::vector<T>& values,
      nimble::Buffer& outputBuffer) {
    NIMBLE_CHECK(!values.empty());
    const uint32_t rowCount = values.size();

    nimble::Vector<uint32_t> runLengths{pool_.get()};
    nimble::Vector<PhysicalType> runValues{pool_.get()};
    auto span = std::span<const PhysicalType>(
        reinterpret_cast<const PhysicalType*>(values.data()), values.size());
    nimble::rle::computeRuns(span, &runLengths, &runValues);

    // Encode run lengths as Trivial<uint32_t>.
    auto runLengthSpan =
        std::span<const uint32_t>(runLengths.data(), runLengths.size());
    nimble::Buffer rlBuf{*pool_};
    nimble::EncodingSelection<uint32_t> rlSel{
        {.encodingType = nimble::EncodingType::Trivial},
        nimble::Statistics<uint32_t>::create(runLengthSpan),
        std::make_unique<TestTrivialEncodingSelectionPolicy<uint32_t>>(
            false, false)};
    auto serializedRunLengths =
        nimble::TrivialEncoding<uint32_t>::encode(rlSel, runLengthSpan, rlBuf);

    // Encode run values as Dictionary<T>.
    auto rvSpan =
        std::span<const PhysicalType>(runValues.data(), runValues.size());
    nimble::Buffer rvBuf{*pool_};
    nimble::EncodingSelection<PhysicalType> rvSel{
        {.encodingType = nimble::EncodingType::Dictionary},
        nimble::Statistics<PhysicalType>::create(rvSpan),
        std::make_unique<TestTrivialEncodingSelectionPolicy<T>>(false, false)};
    auto serializedRunValues =
        nimble::DictionaryEncoding<T>::encode(rvSel, rvSpan, rvBuf);

    // Assemble RLE binary: prefix | writeString(runLengths) |
    // writeBytes(runValues)
    const uint32_t encodingSize = nimble::Encoding::kPrefixSize + 4 +
        serializedRunLengths.size() + serializedRunValues.size();
    char* reserved = outputBuffer.reserve(encodingSize);
    char* pos = reserved;
    nimble::encoding::writeChar(
        static_cast<char>(nimble::EncodingType::RLE), pos);
    nimble::encoding::writeChar(
        static_cast<char>(nimble::TypeTraits<T>::dataType), pos);
    nimble::encoding::writeUint32(rowCount, pos);
    nimble::encoding::writeString(serializedRunLengths, pos);
    nimble::encoding::writeBytes(serializedRunValues, pos);

    return std::string_view(reserved, encodingSize);
  }

  /// Encodes data as RLE wrapping Dictionary.
  std::unique_ptr<nimble::Encoding> encodeRleDictionary(
      const std::vector<T>& values) {
    auto serialized = serializeRleDictionary(values, *buffer_);
    return decodeEncoding(serialized);
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
  std::vector<velox::BufferPtr> stringBuffers_;
  std::vector<std::string> stringPool_;
};

using DictionaryApiTypes = ::testing::Types<
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
    float,
    double,
    std::string_view>;

TYPED_TEST_SUITE(DictionaryApiTypedTest, DictionaryApiTypes);

TYPED_TEST(DictionaryApiTypedTest, basics) {
  using T = TypeParam;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    data = {"alpha", "bravo", "alpha", "charlie", "bravo"};
  } else {
    data = {T(1), T(2), T(1), T(3), T(2)};
  }

  auto encoding = this->encodeDictionary(data);
  EXPECT_TRUE(encoding->dictionaryEnabled());
  EXPECT_EQ(encoding->dictionarySize(), 3);

  const auto* entries = static_cast<const T*>(encoding->dictionaryEntries());
  ASSERT_NE(entries, nullptr);
  std::set<T> actualUniques(entries, entries + 3);

  if constexpr (std::is_same_v<T, std::string_view>) {
    EXPECT_THAT(
        actualUniques,
        ::testing::UnorderedElementsAre("alpha", "bravo", "charlie"));
  } else {
    EXPECT_THAT(
        actualUniques, ::testing::UnorderedElementsAre(T(1), T(2), T(3)));
  }

  for (uint32_t i = 0; i < 3; ++i) {
    const auto* entry = static_cast<const T*>(encoding->dictionaryEntry(i));
    EXPECT_EQ(*entry, entries[i]);
  }
}

TYPED_TEST(DictionaryApiTypedTest, fuzz) {
  using T = TypeParam;
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  constexpr int kNumRuns = 10;
  constexpr int kMaxRows = 300;

  for (int run = 0; run < kNumRuns; ++run) {
    const int numRows = 2 + rng() % kMaxRows;
    const int cardinality = 2 + rng() % 10;

    // For string types, build a pool of unique random strings per run.
    if constexpr (std::is_same_v<T, std::string_view>) {
      this->stringPool_.clear();
      for (int j = 0; j < cardinality; ++j) {
        const int len = rng() % 30;
        std::string s = fmt::format("{}:", j);
        for (int k = 0; k < len; ++k) {
          s += static_cast<char>('a' + rng() % 26);
        }
        this->stringPool_.push_back(std::move(s));
      }
    }

    std::vector<T> data;
    std::set<T> expectedUniques;
    data.reserve(numRows);
    for (int i = 0; i < numRows; ++i) {
      T val;
      if constexpr (std::is_same_v<T, std::string_view>) {
        val = std::string_view(this->stringPool_[rng() % cardinality]);
      } else {
        val = static_cast<T>(rng() % cardinality);
      }
      data.push_back(val);
      expectedUniques.insert(val);
    }

    SCOPED_TRACE(fmt::format("run={} seed={} numRows={}", run, seed, numRows));

    auto encoding = this->encodeDictionary(data);
    ASSERT_TRUE(encoding->dictionaryEnabled());
    ASSERT_EQ(encoding->dictionarySize(), expectedUniques.size());

    const auto* entries = static_cast<const T*>(encoding->dictionaryEntries());
    std::set<T> actualUniques(entries, entries + encoding->dictionarySize());
    EXPECT_EQ(actualUniques, expectedUniques);

    for (uint32_t i = 0; i < encoding->dictionarySize(); ++i) {
      const auto* entry = static_cast<const T*>(encoding->dictionaryEntry(i));
      EXPECT_EQ(*entry, entries[i]);
    }
  }
}

TYPED_TEST(DictionaryApiTypedTest, nullableBasics) {
  using T = TypeParam;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    data = {"alpha", "bravo", "alpha", "charlie", "bravo"};
  } else {
    data = {T(1), T(2), T(1), T(3), T(2)};
  }
  nimble::Vector<bool> nulls{this->pool_.get()};
  for (bool null : {true, true, false, true, true}) {
    nulls.push_back(null);
  }

  auto encoding = this->encodeNullableDictionary(data, nulls);
  ASSERT_TRUE(encoding->isNullable());
  EXPECT_TRUE(encoding->dictionaryEnabled());
  EXPECT_EQ(encoding->dictionarySize(), 3);

  const auto* entries = static_cast<const T*>(encoding->dictionaryEntries());
  ASSERT_NE(entries, nullptr);
  std::set<T> actualUniques(entries, entries + 3);

  if constexpr (std::is_same_v<T, std::string_view>) {
    EXPECT_THAT(
        actualUniques,
        ::testing::UnorderedElementsAre("alpha", "bravo", "charlie"));
  } else {
    EXPECT_THAT(
        actualUniques, ::testing::UnorderedElementsAre(T(1), T(2), T(3)));
  }

  for (uint32_t i = 0; i < 3; ++i) {
    const auto* entry = static_cast<const T*>(encoding->dictionaryEntry(i));
    EXPECT_EQ(*entry, entries[i]);
  }
}

// Verifies that dictionary size excludes values that only appear in
// null-masked rows. "charlie"/T(3) only appears at row 2 which is null,
// so the dictionary should contain only 2 entries.
TYPED_TEST(DictionaryApiTypedTest, nullableAlphabetValue) {
  using T = TypeParam;
  // Row:  0       1       2         3       4
  // Data: alpha   bravo   charlie   alpha   bravo
  // Null: true    true    false     true    true
  // "charlie" at row 2 is the only occurrence and it's masked by null.
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    data = {"alpha", "bravo", "charlie", "alpha", "bravo"};
  } else {
    data = {T(1), T(2), T(3), T(1), T(2)};
  }
  nimble::Vector<bool> nulls{this->pool_.get()};
  for (bool null : {true, true, false, true, true}) {
    nulls.push_back(null);
  }

  auto encoding = this->encodeNullableDictionary(data, nulls);
  ASSERT_TRUE(encoding->isNullable());
  EXPECT_TRUE(encoding->dictionaryEnabled());
  // Only "alpha" and "bravo" (or T(1) and T(2)) are non-null entries.
  EXPECT_EQ(encoding->dictionarySize(), 2);

  const auto* entries = static_cast<const T*>(encoding->dictionaryEntries());
  ASSERT_NE(entries, nullptr);
  std::set<T> actualUniques(entries, entries + encoding->dictionarySize());
  if constexpr (std::is_same_v<T, std::string_view>) {
    EXPECT_THAT(
        actualUniques, ::testing::UnorderedElementsAre("alpha", "bravo"));
  } else {
    EXPECT_THAT(actualUniques, ::testing::UnorderedElementsAre(T(1), T(2)));
  }
}

TYPED_TEST(DictionaryApiTypedTest, nullableFuzz) {
  using T = TypeParam;
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  constexpr int kNumRuns = 10;
  constexpr int kMaxRows = 300;

  for (int run = 0; run < kNumRuns; ++run) {
    const int numRows = 2 + rng() % kMaxRows;
    const int cardinality = 2 + rng() % 10;

    if constexpr (std::is_same_v<T, std::string_view>) {
      this->stringPool_.clear();
      for (int j = 0; j < cardinality; ++j) {
        const int len = rng() % 30;
        std::string s = fmt::format("{}:", j);
        for (int k = 0; k < len; ++k) {
          s += static_cast<char>('a' + rng() % 26);
        }
        this->stringPool_.push_back(std::move(s));
      }
    }

    std::vector<T> data;
    nimble::Vector<bool> nulls{this->pool_.get()};
    std::set<T> expectedUniques;
    data.reserve(numRows);
    nulls.reserve(numRows);
    for (int i = 0; i < numRows; ++i) {
      bool isNull = (rng() % 5 == 0);
      nulls.push_back(!isNull);
      T val;
      if constexpr (std::is_same_v<T, std::string_view>) {
        val = std::string_view(this->stringPool_[rng() % cardinality]);
      } else {
        val = static_cast<T>(rng() % cardinality);
      }
      data.push_back(val);
      if (!isNull) {
        expectedUniques.insert(val);
      }
    }

    if (expectedUniques.empty()) {
      continue;
    }

    SCOPED_TRACE(fmt::format("run={} seed={} numRows={}", run, seed, numRows));

    auto encoding = this->encodeNullableDictionary(data, nulls);
    ASSERT_TRUE(encoding->isNullable());
    ASSERT_TRUE(encoding->dictionaryEnabled());
    ASSERT_EQ(encoding->dictionarySize(), expectedUniques.size());

    const auto* entries = static_cast<const T*>(encoding->dictionaryEntries());
    std::set<T> actualUniques(entries, entries + encoding->dictionarySize());
    EXPECT_EQ(actualUniques, expectedUniques);

    for (uint32_t i = 0; i < encoding->dictionarySize(); ++i) {
      const auto* entry = static_cast<const T*>(encoding->dictionaryEntry(i));
      EXPECT_EQ(*entry, entries[i]);
    }
  }
}

TYPED_TEST(DictionaryApiTypedTest, buildAlphabet) {
  using T = TypeParam;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    data = {"alpha", "bravo", "alpha", "charlie", "bravo"};
  } else {
    data = {T(1), T(2), T(1), T(3), T(2)};
  }

  auto encoding = this->encodeDictionary(data);
  auto alphabet = nimble::buildEncodingDictionaryAlphabet<T>(encoding.get());
  EXPECT_EQ(alphabet.size(), 3);
  std::set<T> actual(alphabet.begin(), alphabet.end());
  if constexpr (std::is_same_v<T, std::string_view>) {
    EXPECT_EQ(actual, (std::set<T>{"alpha", "bravo", "charlie"}));
  } else {
    EXPECT_EQ(actual, (std::set<T>{T(1), T(2), T(3)}));
  }
}

TYPED_TEST(DictionaryApiTypedTest, buildAlphabetNullable) {
  using T = TypeParam;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    data = {"alpha", "bravo", "alpha", "charlie", "bravo"};
  } else {
    data = {T(1), T(2), T(1), T(3), T(2)};
  }
  nimble::Vector<bool> nulls{this->pool_.get()};
  for (bool null : {true, true, false, true, true}) {
    nulls.push_back(null);
  }

  auto encoding = this->encodeNullableDictionary(data, nulls);
  auto alphabet = nimble::buildEncodingDictionaryAlphabet<T>(encoding.get());
  EXPECT_EQ(alphabet.size(), 3);
  std::set<T> actual(alphabet.begin(), alphabet.end());
  if constexpr (std::is_same_v<T, std::string_view>) {
    EXPECT_EQ(actual, (std::set<T>{"alpha", "bravo", "charlie"}));
  } else {
    EXPECT_EQ(actual, (std::set<T>{T(1), T(2), T(3)}));
  }
}

// Verifies that buildEncodingDictionaryAlphabet excludes values that only
// appear in null-masked rows.
TYPED_TEST(DictionaryApiTypedTest, buildAlphabetNullableAlphabetValue) {
  using T = TypeParam;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    data = {"alpha", "bravo", "charlie", "alpha", "bravo"};
  } else {
    data = {T(1), T(2), T(3), T(1), T(2)};
  }
  nimble::Vector<bool> nulls{this->pool_.get()};
  for (bool null : {true, true, false, true, true}) {
    nulls.push_back(null);
  }

  auto encoding = this->encodeNullableDictionary(data, nulls);
  auto alphabet = nimble::buildEncodingDictionaryAlphabet<T>(encoding.get());
  EXPECT_EQ(alphabet.size(), 2);
  std::set<T> actual(alphabet.begin(), alphabet.end());
  if constexpr (std::is_same_v<T, std::string_view>) {
    EXPECT_EQ(actual, (std::set<T>{"alpha", "bravo"}));
  } else {
    EXPECT_EQ(actual, (std::set<T>{T(1), T(2)}));
  }
}

TYPED_TEST(DictionaryApiTypedTest, buildAlphabetFuzz) {
  using T = TypeParam;
  for (int run = 0; run < 20; ++run) {
    auto seed = folly::Random::rand32();
    std::mt19937 rng(seed);
    const auto numRows = 10 + rng() % 200;
    const auto cardinality = 2 + rng() % 10;

    if constexpr (std::is_same_v<T, std::string_view>) {
      this->stringPool_.clear();
      for (int j = 0; j < cardinality; ++j) {
        const int len = rng() % 30;
        std::string s = fmt::format("{}:", j);
        for (int k = 0; k < len; ++k) {
          s += static_cast<char>('a' + rng() % 26);
        }
        this->stringPool_.push_back(std::move(s));
      }
    }

    std::vector<T> data;
    std::set<T> expectedUniques;
    data.reserve(numRows);
    for (int i = 0; i < numRows; ++i) {
      T val;
      if constexpr (std::is_same_v<T, std::string_view>) {
        val = std::string_view(this->stringPool_[rng() % cardinality]);
      } else {
        val = static_cast<T>(rng() % cardinality);
      }
      data.push_back(val);
      expectedUniques.insert(val);
    }

    SCOPED_TRACE(fmt::format("run={} seed={} numRows={}", run, seed, numRows));

    auto encoding = this->encodeDictionary(data);
    auto alphabet = nimble::buildEncodingDictionaryAlphabet<T>(encoding.get());
    EXPECT_EQ(alphabet.size(), expectedUniques.size());
    std::set<T> actual(alphabet.begin(), alphabet.end());
    EXPECT_EQ(actual, expectedUniques);
  }
}

TYPED_TEST(DictionaryApiTypedTest, buildAlphabetNullableFuzz) {
  using T = TypeParam;
  for (int run = 0; run < 20; ++run) {
    auto seed = folly::Random::rand32();
    std::mt19937 rng(seed);
    const auto numRows = 10 + rng() % 200;
    const auto cardinality = 2 + rng() % 10;

    if constexpr (std::is_same_v<T, std::string_view>) {
      this->stringPool_.clear();
      for (int j = 0; j < cardinality; ++j) {
        const int len = rng() % 30;
        std::string s = fmt::format("{}:", j);
        for (int k = 0; k < len; ++k) {
          s += static_cast<char>('a' + rng() % 26);
        }
        this->stringPool_.push_back(std::move(s));
      }
    }

    std::vector<T> data;
    nimble::Vector<bool> nulls{this->pool_.get()};
    std::set<T> expectedUniques;
    data.reserve(numRows);
    nulls.reserve(numRows);
    for (int i = 0; i < numRows; ++i) {
      bool isNull = (rng() % 5 == 0);
      nulls.push_back(!isNull);
      T val;
      if constexpr (std::is_same_v<T, std::string_view>) {
        val = std::string_view(this->stringPool_[rng() % cardinality]);
      } else {
        val = static_cast<T>(rng() % cardinality);
      }
      data.push_back(val);
      if (!isNull) {
        expectedUniques.insert(val);
      }
    }

    if (expectedUniques.empty()) {
      continue;
    }

    SCOPED_TRACE(fmt::format("run={} seed={} numRows={}", run, seed, numRows));

    auto encoding = this->encodeNullableDictionary(data, nulls);
    auto alphabet = nimble::buildEncodingDictionaryAlphabet<T>(encoding.get());
    EXPECT_EQ(alphabet.size(), expectedUniques.size());
    std::set<T> actual(alphabet.begin(), alphabet.end());
    EXPECT_EQ(actual, expectedUniques);
  }
}

TYPED_TEST(DictionaryApiTypedTest, mainlyConstantDictionaryBasics) {
  using T = TypeParam;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo"};
    std::string_view common = "common_value";
    this->stringPool_.push_back(std::string(common));
    data.resize(20, std::string_view(this->stringPool_[2]));
    data[3] = std::string_view(this->stringPool_[0]);
    data[7] = std::string_view(this->stringPool_[1]);
    data[15] = std::string_view(this->stringPool_[0]);
  } else {
    T common = T(99);
    data.resize(20, common);
    data[3] = T(1);
    data[7] = T(2);
    data[15] = T(1);
  }

  auto encoding = this->encodeMainlyConstantDictionary(data);
  EXPECT_TRUE(encoding->dictionaryEnabled());
  EXPECT_EQ(encoding->dictionarySize(), 3);

  std::set<T> entries;
  for (uint32_t i = 0; i < encoding->dictionarySize(); ++i) {
    const auto* entry = static_cast<const T*>(encoding->dictionaryEntry(i));
    entries.insert(*entry);
  }

  if constexpr (std::is_same_v<T, std::string_view>) {
    EXPECT_THAT(
        entries,
        ::testing::UnorderedElementsAre("alpha", "bravo", "common_value"));
  } else {
    EXPECT_THAT(entries, ::testing::UnorderedElementsAre(T(1), T(2), T(99)));
  }
}

TYPED_TEST(DictionaryApiTypedTest, mainlyConstantDictionaryFuzz) {
  using T = TypeParam;
  for (int run = 0; run < 20; ++run) {
    auto seed = folly::Random::rand32();
    std::mt19937 rng(seed);
    const auto numRows = 50 + rng() % 500;
    const auto cardinality = 2 + rng() % 50;
    const int commonPct = 55 + rng() % 40;

    if constexpr (std::is_same_v<T, std::string_view>) {
      this->stringPool_.clear();
      for (int j = 0; j < cardinality; ++j) {
        const int len = 1 + rng() % 20;
        std::string s = fmt::format("{}:", j);
        for (int k = 0; k < len; ++k) {
          s += static_cast<char>('a' + rng() % 26);
        }
        this->stringPool_.push_back(std::move(s));
      }
    }

    T commonValue;
    if constexpr (std::is_same_v<T, std::string_view>) {
      commonValue = std::string_view(this->stringPool_[0]);
    } else {
      commonValue = static_cast<T>(cardinality + 1);
    }

    std::vector<T> data;
    std::set<T> expectedUniques;
    expectedUniques.insert(commonValue);
    data.reserve(numRows);
    for (int i = 0; i < numRows; ++i) {
      if (static_cast<int>(rng() % 100) < commonPct) {
        data.push_back(commonValue);
      } else {
        T val;
        if constexpr (std::is_same_v<T, std::string_view>) {
          val = std::string_view(
              this->stringPool_[1 + rng() % (cardinality - 1)]);
        } else {
          val = static_cast<T>(1 + rng() % cardinality);
        }
        data.push_back(val);
        expectedUniques.insert(val);
      }
    }

    SCOPED_TRACE(fmt::format("run={} seed={} numRows={}", run, seed, numRows));

    auto encoding = this->encodeMainlyConstantDictionary(data);
    ASSERT_TRUE(encoding->dictionaryEnabled());
    ASSERT_EQ(encoding->dictionarySize(), expectedUniques.size());

    std::set<T> actualEntries;
    for (uint32_t i = 0; i < encoding->dictionarySize(); ++i) {
      const auto* entry = static_cast<const T*>(encoding->dictionaryEntry(i));
      actualEntries.insert(*entry);
    }
    EXPECT_EQ(actualEntries, expectedUniques);
  }
}

TYPED_TEST(DictionaryApiTypedTest, buildAlphabetMainlyConstantDictionary) {
  using T = TypeParam;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo"};
    std::string_view common = "common";
    this->stringPool_.push_back(std::string(common));
    data.resize(20, std::string_view(this->stringPool_[2]));
    data[3] = std::string_view(this->stringPool_[0]);
    data[7] = std::string_view(this->stringPool_[1]);
    data[15] = std::string_view(this->stringPool_[0]);
  } else {
    data.resize(20, T(99));
    data[3] = T(1);
    data[7] = T(2);
    data[15] = T(1);
  }

  auto encoding = this->encodeMainlyConstantDictionary(data);
  auto alphabet = nimble::buildEncodingDictionaryAlphabet<T>(encoding.get());
  EXPECT_EQ(alphabet.size(), 3);
  std::set<T> actual(alphabet.begin(), alphabet.end());
  if constexpr (std::is_same_v<T, std::string_view>) {
    EXPECT_EQ(actual, (std::set<T>{"alpha", "bravo", "common"}));
  } else {
    EXPECT_EQ(actual, (std::set<T>{T(1), T(2), T(99)}));
  }
}

TYPED_TEST(DictionaryApiTypedTest, buildAlphabetMainlyConstantDictionaryFuzz) {
  using T = TypeParam;
  for (int run = 0; run < 20; ++run) {
    auto seed = folly::Random::rand32();
    std::mt19937 rng(seed);
    const auto numRows = 50 + rng() % 500;
    const auto cardinality = 2 + rng() % 50;
    const int commonPct = 55 + rng() % 40;

    if constexpr (std::is_same_v<T, std::string_view>) {
      this->stringPool_.clear();
      for (int j = 0; j < cardinality; ++j) {
        const int len = 1 + rng() % 20;
        std::string s = fmt::format("{}:", j);
        for (int k = 0; k < len; ++k) {
          s += static_cast<char>('a' + rng() % 26);
        }
        this->stringPool_.push_back(std::move(s));
      }
    }

    T commonValue;
    if constexpr (std::is_same_v<T, std::string_view>) {
      commonValue = std::string_view(this->stringPool_[0]);
    } else {
      commonValue = static_cast<T>(cardinality + 1);
    }

    std::vector<T> data;
    std::set<T> expectedUniques;
    expectedUniques.insert(commonValue);
    data.reserve(numRows);
    for (int i = 0; i < numRows; ++i) {
      if (static_cast<int>(rng() % 100) < commonPct) {
        data.push_back(commonValue);
      } else {
        T val;
        if constexpr (std::is_same_v<T, std::string_view>) {
          val = std::string_view(
              this->stringPool_[1 + rng() % (cardinality - 1)]);
        } else {
          val = static_cast<T>(1 + rng() % cardinality);
        }
        data.push_back(val);
        expectedUniques.insert(val);
      }
    }

    SCOPED_TRACE(fmt::format("run={} seed={} numRows={}", run, seed, numRows));

    auto encoding = this->encodeMainlyConstantDictionary(data);
    auto alphabet = nimble::buildEncodingDictionaryAlphabet<T>(encoding.get());
    EXPECT_EQ(alphabet.size(), expectedUniques.size());
    std::set<T> actual(alphabet.begin(), alphabet.end());
    EXPECT_EQ(actual, expectedUniques);
  }
}

TYPED_TEST(DictionaryApiTypedTest, nullableMainlyConstantDictionaryBasics) {
  using T = TypeParam;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo"};
    std::string_view common = "common";
    this->stringPool_.push_back(std::string(common));
    data = {
        std::string_view(this->stringPool_[2]),
        std::string_view(this->stringPool_[0]),
        std::string_view(this->stringPool_[2]),
        std::string_view(this->stringPool_[1]),
        std::string_view(this->stringPool_[2]),
        std::string_view(this->stringPool_[2]),
        std::string_view(this->stringPool_[2])};
  } else {
    data = {T(99), T(1), T(99), T(2), T(99), T(99), T(99)};
  }
  nimble::Vector<bool> nulls{this->pool_.get()};
  for (bool n : {true, true, false, true, true, true, true}) {
    nulls.push_back(n);
  }

  auto encoding = this->encodeNullableMainlyConstantDictionary(data, nulls);
  EXPECT_TRUE(encoding->dictionaryEnabled());
  EXPECT_EQ(encoding->dictionarySize(), 3);
}

TYPED_TEST(DictionaryApiTypedTest, materializeIndicesDictionary) {
  using T = TypeParam;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo", "charlie"};
    data = {
        std::string_view(this->stringPool_[0]),
        std::string_view(this->stringPool_[1]),
        std::string_view(this->stringPool_[0]),
        std::string_view(this->stringPool_[2]),
        std::string_view(this->stringPool_[1])};
  } else {
    data = {T(1), T(2), T(1), T(3), T(2)};
  }

  auto encoding = this->encodeDictionary(data);
  ASSERT_TRUE(encoding->dictionaryEnabled());
  const uint32_t rowCount = data.size();

  nimble::Vector<uint32_t> indices{this->pool_.get(), rowCount};
  encoding->materializeIndices(rowCount, indices.data());

  // Every materialized index must be in range and map to the correct value.
  for (uint32_t i = 0; i < rowCount; ++i) {
    ASSERT_LT(indices[i], encoding->dictionarySize());
    const auto* entry =
        static_cast<const T*>(encoding->dictionaryEntry(indices[i]));
    EXPECT_EQ(*entry, data[i]) << "row=" << i;
  }
}

TYPED_TEST(DictionaryApiTypedTest, materializeIndicesDictionaryFuzz) {
  using T = TypeParam;

  for (int run = 0; run < 20; ++run) {
    auto seed = folly::Random::rand32();
    std::mt19937 rng(seed);
    const int numRows = 2 + rng() % 300;
    const int cardinality = 2 + rng() % 10;

    if constexpr (std::is_same_v<T, std::string_view>) {
      this->stringPool_.clear();
      for (int j = 0; j < cardinality; ++j) {
        const int len = rng() % 30;
        std::string s = fmt::format("{}:", j);
        for (int k = 0; k < len; ++k) {
          s += static_cast<char>('a' + rng() % 26);
        }
        this->stringPool_.push_back(std::move(s));
      }
    }

    std::vector<T> data;
    data.reserve(numRows);
    for (int i = 0; i < numRows; ++i) {
      T val;
      if constexpr (std::is_same_v<T, std::string_view>) {
        val = std::string_view(this->stringPool_[rng() % cardinality]);
      } else {
        val = static_cast<T>(rng() % cardinality);
      }
      data.push_back(val);
    }

    SCOPED_TRACE(fmt::format("run={} seed={} numRows={}", run, seed, numRows));

    auto encoding = this->encodeDictionary(data);
    ASSERT_TRUE(encoding->dictionaryEnabled());

    nimble::Vector<uint32_t> indices{
        this->pool_.get(), static_cast<uint32_t>(numRows)};
    encoding->materializeIndices(numRows, indices.data());

    for (int i = 0; i < numRows; ++i) {
      ASSERT_LT(indices[i], encoding->dictionarySize()) << "row=" << i;
      const auto* entry =
          static_cast<const T*>(encoding->dictionaryEntry(indices[i]));
      EXPECT_EQ(*entry, data[i]) << "row=" << i;
    }
  }
}

TYPED_TEST(DictionaryApiTypedTest, materializeIndicesMainlyConstantDictionary) {
  using T = TypeParam;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo"};
    std::string_view common = "common_value";
    this->stringPool_.push_back(std::string(common));
    data.resize(20, std::string_view(this->stringPool_[2]));
    data[3] = std::string_view(this->stringPool_[0]);
    data[7] = std::string_view(this->stringPool_[1]);
    data[15] = std::string_view(this->stringPool_[0]);
  } else {
    T common = T(99);
    data.resize(20, common);
    data[3] = T(1);
    data[7] = T(2);
    data[15] = T(1);
  }

  auto encoding = this->encodeMainlyConstantDictionary(data);
  ASSERT_TRUE(encoding->dictionaryEnabled());
  const uint32_t rowCount = data.size();
  const uint32_t commonValueIndex = encoding->dictionarySize() - 1;

  nimble::Vector<uint32_t> indices{this->pool_.get(), rowCount};
  encoding->materializeIndices(rowCount, indices.data());

  // Build a lookup from alphabet entry value to index.
  std::unordered_map<T, uint32_t> valueToIndex;
  for (uint32_t i = 0; i < encoding->dictionarySize(); ++i) {
    const auto* entry = static_cast<const T*>(encoding->dictionaryEntry(i));
    valueToIndex[*entry] = i;
  }

  for (uint32_t i = 0; i < rowCount; ++i) {
    ASSERT_LT(indices[i], encoding->dictionarySize()) << "row=" << i;
    const auto* entry =
        static_cast<const T*>(encoding->dictionaryEntry(indices[i]));
    EXPECT_EQ(*entry, data[i]) << "row=" << i;
  }

  // Common-value rows should all map to the commonValueIndex.
  T commonValue;
  if constexpr (std::is_same_v<T, std::string_view>) {
    commonValue = std::string_view(this->stringPool_[2]);
  } else {
    commonValue = T(99);
  }
  for (uint32_t i = 0; i < rowCount; ++i) {
    if (data[i] == commonValue) {
      EXPECT_EQ(indices[i], commonValueIndex) << "row=" << i;
    }
  }
}

TYPED_TEST(
    DictionaryApiTypedTest,
    materializeIndicesMainlyConstantDictionaryFuzz) {
  using T = TypeParam;

  for (int run = 0; run < 20; ++run) {
    auto seed = folly::Random::rand32();
    std::mt19937 rng(seed);
    const auto numRows = 50 + rng() % 500;
    const auto cardinality = 2 + rng() % 50;
    const int commonPct = 55 + rng() % 40;

    if constexpr (std::is_same_v<T, std::string_view>) {
      this->stringPool_.clear();
      for (int j = 0; j < cardinality; ++j) {
        const int len = 1 + rng() % 20;
        std::string s = fmt::format("{}:", j);
        for (int k = 0; k < len; ++k) {
          s += static_cast<char>('a' + rng() % 26);
        }
        this->stringPool_.push_back(std::move(s));
      }
    }

    T commonValue;
    if constexpr (std::is_same_v<T, std::string_view>) {
      commonValue = std::string_view(this->stringPool_[0]);
    } else {
      commonValue = static_cast<T>(cardinality + 1);
    }

    std::vector<T> data;
    data.reserve(numRows);
    for (int i = 0; i < numRows; ++i) {
      if (static_cast<int>(rng() % 100) < commonPct) {
        data.push_back(commonValue);
      } else {
        T val;
        if constexpr (std::is_same_v<T, std::string_view>) {
          val = std::string_view(
              this->stringPool_[1 + rng() % (cardinality - 1)]);
        } else {
          val = static_cast<T>(1 + rng() % cardinality);
        }
        data.push_back(val);
      }
    }

    SCOPED_TRACE(fmt::format("run={} seed={} numRows={}", run, seed, numRows));

    auto encoding = this->encodeMainlyConstantDictionary(data);
    ASSERT_TRUE(encoding->dictionaryEnabled());
    const uint32_t commonValueIndex = encoding->dictionarySize() - 1;

    nimble::Vector<uint32_t> indices{
        this->pool_.get(), static_cast<uint32_t>(numRows)};
    encoding->materializeIndices(numRows, indices.data());

    for (int i = 0; i < numRows; ++i) {
      ASSERT_LT(indices[i], encoding->dictionarySize()) << "row=" << i;
      const auto* entry =
          static_cast<const T*>(encoding->dictionaryEntry(indices[i]));
      EXPECT_EQ(*entry, data[i]) << "row=" << i;

      // Common-value rows must map to the last alphabet entry.
      if (data[i] == commonValue) {
        EXPECT_EQ(indices[i], commonValueIndex) << "row=" << i;
      }
    }
  }
}

TYPED_TEST(DictionaryApiTypedTest, rleDictionaryBasics) {
  using T = TypeParam;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo", "charlie"};
    // 5x alpha, 3x bravo, 4x charlie, 3x alpha
    for (int i = 0; i < 5; ++i) {
      data.push_back(std::string_view(this->stringPool_[0]));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(std::string_view(this->stringPool_[1]));
    }
    for (int i = 0; i < 4; ++i) {
      data.push_back(std::string_view(this->stringPool_[2]));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(std::string_view(this->stringPool_[0]));
    }
  } else {
    // 5x 10, 3x 20, 4x 30, 3x 10
    for (int i = 0; i < 5; ++i) {
      data.push_back(T(10));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(T(20));
    }
    for (int i = 0; i < 4; ++i) {
      data.push_back(T(30));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(T(10));
    }
  }

  auto encoding = this->encodeRleDictionary(data);
  if constexpr (!nimble::isStringType<T>()) {
    EXPECT_FALSE(encoding->dictionaryEnabled());
    std::vector<T> result(data.size());
    encoding->materialize(data.size(), result.data());
    EXPECT_EQ(result, data);
    return;
  }
  EXPECT_TRUE(encoding->dictionaryEnabled());
  EXPECT_EQ(encoding->dictionarySize(), 3);

  std::set<T> entries;
  for (uint32_t i = 0; i < encoding->dictionarySize(); ++i) {
    const auto* entry = static_cast<const T*>(encoding->dictionaryEntry(i));
    entries.insert(*entry);
  }

  if constexpr (std::is_same_v<T, std::string_view>) {
    EXPECT_THAT(
        entries, ::testing::UnorderedElementsAre("alpha", "bravo", "charlie"));
  } else {
    EXPECT_THAT(entries, ::testing::UnorderedElementsAre(T(10), T(20), T(30)));
  }

  // Verify materializeIndices round-trips through the dictionary.
  const auto dictSize = encoding->dictionarySize();
  std::vector<uint32_t> indices(data.size());
  encoding->materializeIndices(data.size(), indices.data());

  const auto* allEntries = static_cast<const T*>(encoding->dictionaryEntries());
  for (size_t i = 0; i < data.size(); ++i) {
    SCOPED_TRACE(fmt::format("row {}", i));
    ASSERT_LT(indices[i], dictSize);
    EXPECT_EQ(allEntries[indices[i]], data[i]);
  }
}

// Verifies materializeIndices after skipping past the first run.
// This catches off-by-one bugs where reset() or skip() consume a
// run's index without storing it, causing materializeIndices to
// use a stale index for the current run.
TYPED_TEST(DictionaryApiTypedTest, rleDictionaryMaterializeAfterSkip) {
  using T = TypeParam;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo", "charlie"};
    for (int i = 0; i < 5; ++i) {
      data.push_back(std::string_view(this->stringPool_[0]));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(std::string_view(this->stringPool_[1]));
    }
    for (int i = 0; i < 4; ++i) {
      data.push_back(std::string_view(this->stringPool_[2]));
    }
  } else {
    for (int i = 0; i < 5; ++i) {
      data.push_back(T(10));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(T(20));
    }
    for (int i = 0; i < 4; ++i) {
      data.push_back(T(30));
    }
  }

  auto encoding = this->encodeRleDictionary(data);
  if constexpr (!nimble::isStringType<T>()) {
    EXPECT_FALSE(encoding->dictionaryEnabled());
    encoding->skip(5);
    std::vector<T> result(data.size() - 5);
    encoding->materialize(data.size() - 5, result.data());
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_EQ(result[i], data[5 + i]);
    }
    return;
  }
  ASSERT_TRUE(encoding->dictionaryEnabled());

  // Skip past the first run (5 rows) into the second run.
  encoding->skip(5);

  const auto remaining = data.size() - 5;
  const auto dictSize = encoding->dictionarySize();
  std::vector<uint32_t> indices(remaining);
  encoding->materializeIndices(remaining, indices.data());

  const auto* entries = static_cast<const T*>(encoding->dictionaryEntries());
  for (size_t i = 0; i < remaining; ++i) {
    SCOPED_TRACE(fmt::format("row {}", i));
    ASSERT_LT(indices[i], dictSize);
    EXPECT_EQ(entries[indices[i]], data[5 + i]);
  }
}

// Verifies materialize after skipping exactly to a run boundary in
// non-dict mode. This catches off-by-one bugs where skip() exits with
// copiesRemaining_==0 after consuming a full run, and the subsequent
// materialize produces wrong values from a stale currentValue_.
TYPED_TEST(DictionaryApiTypedTest, rleMaterializeAfterSkip) {
  using T = TypeParam;
  using PhysicalType = typename nimble::TypeTraits<T>::physicalType;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo", "charlie"};
    for (int i = 0; i < 5; ++i) {
      data.push_back(std::string_view(this->stringPool_[0]));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(std::string_view(this->stringPool_[1]));
    }
    for (int i = 0; i < 4; ++i) {
      data.push_back(std::string_view(this->stringPool_[2]));
    }
  } else {
    for (int i = 0; i < 5; ++i) {
      data.push_back(T(10));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(T(20));
    }
    for (int i = 0; i < 4; ++i) {
      data.push_back(T(30));
    }
  }

  auto serialized = this->serializeRleDictionary(data, *this->buffer_);
  auto encoding = this->decodeEncoding(serialized);

  // Skip exactly to the first run boundary (5 rows).
  encoding->skip(5);

  const auto remaining = data.size() - 5;
  std::vector<PhysicalType> materialized(remaining);
  encoding->materialize(remaining, materialized.data());
  for (size_t i = 0; i < remaining; ++i) {
    SCOPED_TRACE(fmt::format("row {}", i));
    EXPECT_EQ(
        materialized[i], reinterpret_cast<const PhysicalType&>(data[5 + i]));
  }
}

// Verifies that back-to-back materialize calls that each exactly exhaust
// a run produce correct values. With the lazy pattern, each call ends with
// copiesRemaining_==0 and the next call must load the next run on demand.
TYPED_TEST(DictionaryApiTypedTest, rleMaterializeExactRunBoundaries) {
  using T = TypeParam;
  using PhysicalType = typename nimble::TypeTraits<T>::physicalType;

  // 3 runs: [A x 5] [B x 3] [C x 4]
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo", "charlie"};
    for (int i = 0; i < 5; ++i) {
      data.push_back(std::string_view(this->stringPool_[0]));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(std::string_view(this->stringPool_[1]));
    }
    for (int i = 0; i < 4; ++i) {
      data.push_back(std::string_view(this->stringPool_[2]));
    }
  } else {
    for (int i = 0; i < 5; ++i) {
      data.push_back(T(10));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(T(20));
    }
    for (int i = 0; i < 4; ++i) {
      data.push_back(T(30));
    }
  }

  auto serialized = this->serializeRleDictionary(data, *this->buffer_);
  auto encoding = this->decodeEncoding(serialized);

  // Materialize exactly run 1 (5 rows).
  std::vector<PhysicalType> mat1(5);
  encoding->materialize(5, mat1.data());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(mat1[i], reinterpret_cast<const PhysicalType&>(data[i]));
  }

  // Materialize exactly run 2 (3 rows).
  std::vector<PhysicalType> mat2(3);
  encoding->materialize(3, mat2.data());
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(mat2[i], reinterpret_cast<const PhysicalType&>(data[5 + i]));
  }

  // Materialize exactly run 3 (4 rows).
  std::vector<PhysicalType> mat3(4);
  encoding->materialize(4, mat3.data());
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(mat3[i], reinterpret_cast<const PhysicalType&>(data[8 + i]));
  }
}

// Verifies skip to exact run boundary followed by materialize across
// multiple skip+materialize cycles. Each skip lands exactly at a run
// boundary (copiesRemaining_==0), and the subsequent materialize must
// load the next run on demand.
TYPED_TEST(DictionaryApiTypedTest, rleSkipAndMaterializeAlternating) {
  using T = TypeParam;
  using PhysicalType = typename nimble::TypeTraits<T>::physicalType;

  // 4 runs: [A x 3] [B x 4] [C x 2] [D x 5]
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo", "charlie", "delta"};
    for (int i = 0; i < 3; ++i) {
      data.push_back(std::string_view(this->stringPool_[0]));
    }
    for (int i = 0; i < 4; ++i) {
      data.push_back(std::string_view(this->stringPool_[1]));
    }
    for (int i = 0; i < 2; ++i) {
      data.push_back(std::string_view(this->stringPool_[2]));
    }
    for (int i = 0; i < 5; ++i) {
      data.push_back(std::string_view(this->stringPool_[3]));
    }
  } else {
    for (int i = 0; i < 3; ++i) {
      data.push_back(T(10));
    }
    for (int i = 0; i < 4; ++i) {
      data.push_back(T(20));
    }
    for (int i = 0; i < 2; ++i) {
      data.push_back(T(30));
    }
    for (int i = 0; i < 5; ++i) {
      data.push_back(T(40));
    }
  }

  auto serialized = this->serializeRleDictionary(data, *this->buffer_);
  auto encoding = this->decodeEncoding(serialized);

  // Skip run 1 (3 rows), materialize run 2 (4 rows).
  encoding->skip(3);
  std::vector<PhysicalType> mat1(4);
  encoding->materialize(4, mat1.data());
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(mat1[i], reinterpret_cast<const PhysicalType&>(data[3 + i]));
  }

  // Skip run 3 (2 rows), materialize run 4 (5 rows).
  encoding->skip(2);
  std::vector<PhysicalType> mat2(5);
  encoding->materialize(5, mat2.data());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(mat2[i], reinterpret_cast<const PhysicalType&>(data[9 + i]));
  }
}

// RLE wrapping a dictionary-enabled inner encoding reports
// dictionaryEnabled()=true for string types before mode is chosen.
// materialize() can still be called (lazily enters flat mode).
TYPED_TEST(DictionaryApiTypedTest, rleDictionaryLazyMaterialize) {
  using T = TypeParam;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo"};
    for (int i = 0; i < 5; ++i) {
      data.push_back(std::string_view(this->stringPool_[0]));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(std::string_view(this->stringPool_[1]));
    }
  } else {
    for (int i = 0; i < 5; ++i) {
      data.push_back(T(10));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(T(20));
    }
  }

  // Serialize as RLE→Dictionary.
  auto serialized = this->serializeRleDictionary(data, *this->buffer_);

  auto encoding = this->decodeEncoding(serialized);

  if constexpr (std::is_same_v<T, std::string_view>) {
    EXPECT_TRUE(encoding->dictionaryEnabled());
  } else {
    EXPECT_FALSE(encoding->dictionaryEnabled());
  }

  using PhysicalType = typename nimble::TypeTraits<T>::physicalType;
  std::vector<PhysicalType> materialized(data.size());
  encoding->materialize(data.size(), materialized.data());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(materialized[i], reinterpret_cast<const PhysicalType&>(data[i]))
        << "row " << i;
  }
}

// Dictionary(alphabet=RLE(runValues=Constant)). With lazy loading, the
// DictionaryEncoding constructor calls materialize() on the alphabet RLE,
// which lazily enters flat mode. No crash.
TYPED_TEST(DictionaryApiTypedTest, nestedDictionaryRleConstantAlphabet) {
  using T = TypeParam;
  using PhysicalType = typename nimble::TypeTraits<T>::physicalType;

  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha"};
    for (int32_t i = 0; i < 8; ++i) {
      data.push_back(std::string_view(this->stringPool_[0]));
    }
  } else {
    for (int32_t i = 0; i < 8; ++i) {
      data.push_back(T(42));
    }
  }
  const auto rowCount = static_cast<uint32_t>(data.size());

  // Alphabet has one entry.
  nimble::Vector<PhysicalType> alphabet{this->pool_.get()};
  alphabet.push_back(reinterpret_cast<const PhysicalType&>(data[0]));
  nimble::Vector<uint32_t> indices{this->pool_.get(), rowCount};
  for (uint32_t i = 0; i < rowCount; ++i) {
    indices[i] = 0;
  }

  // Encode alphabet as RLE(runValues=Constant).
  nimble::Vector<uint32_t> runLengths{this->pool_.get()};
  runLengths.push_back(1);
  auto rlSpan = std::span<const uint32_t>(runLengths.data(), runLengths.size());
  nimble::Buffer rlBuf{*this->pool_};
  nimble::EncodingSelection<uint32_t> rlSel{
      {.encodingType = nimble::EncodingType::Trivial},
      nimble::Statistics<uint32_t>::create(rlSpan),
      std::make_unique<TestTrivialEncodingSelectionPolicy<uint32_t>>(
          false, false)};
  auto serializedRunLengths =
      nimble::TrivialEncoding<uint32_t>::encode(rlSel, rlSpan, rlBuf);

  // Run values as Constant (dictionaryEnabled()=true).
  auto rvSpan = std::span<const PhysicalType>(alphabet.data(), alphabet.size());
  nimble::Buffer rvBuf{*this->pool_};
  nimble::EncodingSelection<PhysicalType> rvSel{
      {.encodingType = nimble::EncodingType::Constant},
      nimble::Statistics<PhysicalType>::create(rvSpan),
      std::make_unique<TestTrivialEncodingSelectionPolicy<T>>(false, false)};
  auto serializedRunValues =
      nimble::ConstantEncoding<T>::encode(rvSel, rvSpan, rvBuf);

  // Assemble alphabet as RLE binary.
  nimble::Buffer alpBuf{*this->pool_};
  const auto alpEncodingSize = static_cast<uint32_t>(
      nimble::Encoding::kPrefixSize + 4 + serializedRunLengths.size() +
      serializedRunValues.size());
  char* alpPos = alpBuf.reserve(alpEncodingSize);
  char* alpWrite = alpPos;
  nimble::encoding::writeChar(
      static_cast<char>(nimble::EncodingType::RLE), alpWrite);
  nimble::encoding::writeChar(
      static_cast<char>(nimble::TypeTraits<T>::dataType), alpWrite);
  nimble::encoding::writeUint32(alphabet.size(), alpWrite);
  nimble::encoding::writeString(serializedRunLengths, alpWrite);
  nimble::encoding::writeBytes(serializedRunValues, alpWrite);
  auto serializedAlphabet = std::string_view(alpPos, alpEncodingSize);

  // Encode indices as Trivial<uint32_t>.
  auto idxSpan = std::span<const uint32_t>(indices.data(), indices.size());
  nimble::Buffer idxBuf{*this->pool_};
  nimble::EncodingSelection<uint32_t> idxSel{
      {.encodingType = nimble::EncodingType::Trivial},
      nimble::Statistics<uint32_t>::create(idxSpan),
      std::make_unique<TestTrivialEncodingSelectionPolicy<uint32_t>>(
          false, false)};
  auto serializedIndices =
      nimble::TrivialEncoding<uint32_t>::encode(idxSel, idxSpan, idxBuf);

  // Assemble outer Dictionary binary.
  nimble::Buffer dictBuf{*this->pool_};
  const auto dictSize = static_cast<uint32_t>(
      nimble::Encoding::kPrefixSize + 4 + serializedAlphabet.size() +
      serializedIndices.size());
  char* dictPos = dictBuf.reserve(dictSize);
  char* dictWrite = dictPos;
  nimble::encoding::writeChar(
      static_cast<char>(nimble::EncodingType::Dictionary), dictWrite);
  nimble::encoding::writeChar(
      static_cast<char>(nimble::TypeTraits<T>::dataType), dictWrite);
  nimble::encoding::writeUint32(rowCount, dictWrite);
  nimble::encoding::writeUint32(
      static_cast<uint32_t>(serializedAlphabet.size()), dictWrite);
  std::memcpy(dictWrite, serializedAlphabet.data(), serializedAlphabet.size());
  dictWrite += serializedAlphabet.size();
  std::memcpy(dictWrite, serializedIndices.data(), serializedIndices.size());
  auto serialized = std::string_view(dictPos, dictSize);

  auto encoding = this->decodeEncoding(serialized);

  EXPECT_TRUE(encoding->dictionaryEnabled());
  EXPECT_EQ(encoding->rowCount(), rowCount);

  std::vector<PhysicalType> materialized(rowCount);
  encoding->materialize(rowCount, materialized.data());
  for (uint32_t i = 0; i < rowCount; ++i) {
    EXPECT_EQ(materialized[i], reinterpret_cast<const PhysicalType&>(data[i]))
        << "row " << i;
  }
}

// Skip then dict read: skip() accumulates pendingSkips_, then
// materializeIndices() drains them on first ensureDictValues().
TYPED_TEST(DictionaryApiTypedTest, skipThenMaterializeIndices) {
  using T = TypeParam;
  using PhysicalType = typename nimble::TypeTraits<T>::physicalType;

  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo", "charlie"};
    for (int i = 0; i < 5; ++i) {
      data.push_back(std::string_view(this->stringPool_[0]));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(std::string_view(this->stringPool_[1]));
    }
    for (int i = 0; i < 4; ++i) {
      data.push_back(std::string_view(this->stringPool_[2]));
    }
  } else {
    for (int i = 0; i < 5; ++i) {
      data.push_back(T(10));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(T(20));
    }
    for (int i = 0; i < 4; ++i) {
      data.push_back(T(30));
    }
  }

  auto encoding = this->encodeRleDictionary(data);
  if constexpr (!nimble::isStringType<T>()) {
    return;
  }
  ASSERT_TRUE(encoding->dictionaryEnabled());

  // Skip past the first run (5 rows).
  encoding->skip(5);

  const auto remaining = static_cast<uint32_t>(data.size() - 5);
  std::vector<uint32_t> indices(remaining);
  encoding->materializeIndices(remaining, indices.data());

  const auto* entries =
      static_cast<const PhysicalType*>(encoding->dictionaryEntries());
  for (uint32_t i = 0; i < remaining; ++i) {
    EXPECT_EQ(
        entries[indices[i]], reinterpret_cast<const PhysicalType&>(data[5 + i]))
        << "row " << i;
  }
}

// Dict metadata is available before mode is chosen (delegates to
// valuesEncoding_), but unsupported if flat mode is used.
TYPED_TEST(DictionaryApiTypedTest, dictMetadataBeforeModeChosen) {
  using T = TypeParam;
  using PhysicalType = typename nimble::TypeTraits<T>::physicalType;

  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo"};
    for (int i = 0; i < 5; ++i) {
      data.push_back(std::string_view(this->stringPool_[0]));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(std::string_view(this->stringPool_[1]));
    }
  } else {
    for (int i = 0; i < 5; ++i) {
      data.push_back(T(10));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(T(20));
    }
  }

  auto encoding = this->encodeRleDictionary(data);
  if constexpr (!nimble::isStringType<T>()) {
    return;
  }

  // Query metadata before any read — delegates to valuesEncoding_.
  ASSERT_TRUE(encoding->dictionaryEnabled());
  EXPECT_EQ(encoding->dictionarySize(), 2);
  EXPECT_NE(encoding->dictionaryEntries(), nullptr);

  // Verify alphabet entries are correct.
  const auto* entries =
      static_cast<const PhysicalType*>(encoding->dictionaryEntries());
  std::set<PhysicalType> alphabetSet(entries, entries + 2);
  EXPECT_EQ(alphabetSet.size(), 2);

  // Commit to flat mode — metadata is no longer available.
  std::vector<PhysicalType> values(data.size());
  encoding->materialize(data.size(), values.data());
  EXPECT_FALSE(encoding->dictionaryEnabled());
  EXPECT_THROW(encoding->dictionarySize(), nimble::NimbleInternalError);
  EXPECT_THROW(encoding->dictionaryEntries(), nimble::NimbleInternalError);
}

// Mode conflict guard: calling flat API then dict API on the same instance
// must fail. The ensureDictValues() check catches the conflict.
TYPED_TEST(DictionaryApiTypedTest, modeConflictFlatThenDict) {
  using T = TypeParam;

  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo"};
    for (int i = 0; i < 4; ++i) {
      data.push_back(std::string_view(this->stringPool_[0]));
    }
    for (int i = 0; i < 4; ++i) {
      data.push_back(std::string_view(this->stringPool_[1]));
    }
  } else {
    for (int i = 0; i < 4; ++i) {
      data.push_back(T(10));
    }
    for (int i = 0; i < 4; ++i) {
      data.push_back(T(20));
    }
  }

  auto encoding = this->encodeRleDictionary(data);
  if constexpr (!nimble::isStringType<T>()) {
    return;
  }

  // Commit to flat mode.
  using PhysicalType = typename nimble::TypeTraits<T>::physicalType;
  std::vector<PhysicalType> values(4);
  encoding->materialize(4, values.data());

  // Attempting dict API after flat mode must fail.
  std::vector<uint32_t> indices(4);
  EXPECT_THROW(
      encoding->materializeIndices(4, indices.data()),
      nimble::NimbleInternalError);
}

TYPED_TEST(DictionaryApiTypedTest, materializeIndicesRleDictionaryFuzz) {
  using T = TypeParam;

  for (int run = 0; run < 20; ++run) {
    auto seed = folly::Random::rand32();
    std::mt19937 rng(seed);
    const auto numRows = 50 + rng() % 500;
    const auto cardinality = 2 + rng() % 10;
    const auto maxRunLength = 1 + rng() % 20;

    SCOPED_TRACE(
        fmt::format(
            "run={} seed={} numRows={} cardinality={} maxRunLength={}",
            run,
            seed,
            numRows,
            cardinality,
            maxRunLength));

    std::vector<T> data;
    data.reserve(numRows);
    std::vector<T> alphabet;
    if constexpr (std::is_same_v<T, std::string_view>) {
      this->stringPool_.clear();
      for (size_t i = 0; i < cardinality; ++i) {
        this->stringPool_.push_back("val_" + std::to_string(i));
      }
      for (size_t i = 0; i < cardinality; ++i) {
        alphabet.push_back(std::string_view(this->stringPool_[i]));
      }
    } else {
      for (size_t i = 0; i < cardinality; ++i) {
        alphabet.push_back(static_cast<T>(i + 1));
      }
    }

    // Build data with runs.
    while (data.size() < static_cast<size_t>(numRows)) {
      auto value = alphabet[rng() % cardinality];
      auto runLen =
          std::min<size_t>(1 + rng() % maxRunLength, numRows - data.size());
      for (size_t j = 0; j < runLen; ++j) {
        data.push_back(value);
      }
    }

    auto encoding = this->encodeRleDictionary(data);
    if constexpr (!nimble::isStringType<T>()) {
      EXPECT_FALSE(encoding->dictionaryEnabled());
      std::vector<T> result(data.size());
      encoding->materialize(data.size(), result.data());
      EXPECT_EQ(result, data);
      continue;
    }
    ASSERT_TRUE(encoding->dictionaryEnabled());

    const auto dictSize = encoding->dictionarySize();
    std::vector<uint32_t> indices(data.size());
    encoding->materializeIndices(data.size(), indices.data());

    const auto* entries = static_cast<const T*>(encoding->dictionaryEntries());
    for (size_t i = 0; i < data.size(); ++i) {
      ASSERT_LT(indices[i], dictSize) << "row " << i;
      EXPECT_EQ(entries[indices[i]], data[i]) << "row " << i;
    }
  }
}

TYPED_TEST(DictionaryApiTypedTest, buildAlphabetRleDictionary) {
  using T = TypeParam;
  std::vector<T> data;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alpha", "bravo", "charlie"};
    for (int i = 0; i < 4; ++i) {
      data.push_back(std::string_view(this->stringPool_[0]));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(std::string_view(this->stringPool_[1]));
    }
    for (int i = 0; i < 2; ++i) {
      data.push_back(std::string_view(this->stringPool_[2]));
    }
  } else {
    for (int i = 0; i < 4; ++i) {
      data.push_back(T(10));
    }
    for (int i = 0; i < 3; ++i) {
      data.push_back(T(20));
    }
    for (int i = 0; i < 2; ++i) {
      data.push_back(T(30));
    }
  }

  auto encoding = this->encodeRleDictionary(data);
  if constexpr (!nimble::isStringType<T>()) {
    EXPECT_FALSE(encoding->dictionaryEnabled());
    std::vector<T> result(data.size());
    encoding->materialize(data.size(), result.data());
    EXPECT_EQ(result, data);
    return;
  }
  auto alphabet = nimble::buildEncodingDictionaryAlphabet<T>(encoding.get());
  EXPECT_EQ(alphabet.size(), 3);

  std::set<T> alphabetSet(alphabet.begin(), alphabet.end());
  if constexpr (std::is_same_v<T, std::string_view>) {
    EXPECT_THAT(
        alphabetSet,
        ::testing::UnorderedElementsAre("alpha", "bravo", "charlie"));
  } else {
    EXPECT_THAT(
        alphabetSet, ::testing::UnorderedElementsAre(T(10), T(20), T(30)));
  }
}

TYPED_TEST(DictionaryApiTypedTest, constantDictionaryBasics) {
  using T = TypeParam;
  T value;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"constant_value"};
    value = std::string_view(this->stringPool_[0]);
  } else {
    value = T(42);
  }

  std::vector<T> data(10, value);
  auto span = std::span<const typename nimble::TypeTraits<T>::physicalType>(
      reinterpret_cast<const typename nimble::TypeTraits<T>::physicalType*>(
          data.data()),
      data.size());
  nimble::Buffer encBuf{*this->pool_};
  nimble::EncodingSelection<typename nimble::TypeTraits<T>::physicalType> sel{
      {.encodingType = nimble::EncodingType::Constant},
      nimble::Statistics<typename nimble::TypeTraits<T>::physicalType>::create(
          span),
      std::make_unique<TestTrivialEncodingSelectionPolicy<T>>(false, false)};
  auto serialized = nimble::ConstantEncoding<T>::encode(sel, span, encBuf);
  auto encoding = this->decodeEncoding(serialized);

  EXPECT_TRUE(encoding->dictionaryEnabled());
  EXPECT_EQ(encoding->dictionarySize(), 1);
  EXPECT_EQ(*static_cast<const T*>(encoding->dictionaryEntry(0)), value);

  // Verify materializeIndices round-trips through the dictionary.
  std::vector<uint32_t> indices(data.size());
  encoding->materializeIndices(data.size(), indices.data());

  const auto* entries = static_cast<const T*>(encoding->dictionaryEntries());
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(indices[i], 0) << "row " << i;
    EXPECT_EQ(entries[indices[i]], value);
  }
}

TYPED_TEST(DictionaryApiTypedTest, buildAlphabetConstant) {
  using T = TypeParam;
  T value;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"alphabet_value"};
    value = std::string_view(this->stringPool_[0]);
  } else {
    value = T(77);
  }

  std::vector<T> data(5, value);
  auto span = std::span<const typename nimble::TypeTraits<T>::physicalType>(
      reinterpret_cast<const typename nimble::TypeTraits<T>::physicalType*>(
          data.data()),
      data.size());
  nimble::Buffer encBuf{*this->pool_};
  nimble::EncodingSelection<typename nimble::TypeTraits<T>::physicalType> sel{
      {.encodingType = nimble::EncodingType::Constant},
      nimble::Statistics<typename nimble::TypeTraits<T>::physicalType>::create(
          span),
      std::make_unique<TestTrivialEncodingSelectionPolicy<T>>(false, false)};
  auto serialized = nimble::ConstantEncoding<T>::encode(sel, span, encBuf);
  auto encoding = this->decodeEncoding(serialized);

  auto alphabet = nimble::buildEncodingDictionaryAlphabet<T>(encoding.get());
  ASSERT_EQ(alphabet.size(), 1);
  EXPECT_EQ(alphabet[0], value);
}

// Tests MC with exactly 2 distinct values (one common, one other). Exercises
// two inner encoding variants: Dictionary (from encodeMainlyConstantDictionary)
// and Constant (from serializeMainlyConstantWithConstantOthers). Both should
// produce dictionarySize == 2.
TYPED_TEST(DictionaryApiTypedTest, mainlyConstantWithSingleOtherValue) {
  using T = TypeParam;
  T common;
  T other;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"common_value", "other_value"};
    common = std::string_view(this->stringPool_[0]);
    other = std::string_view(this->stringPool_[1]);
  } else {
    common = T(100);
    other = T(42);
  }

  // 20 rows: 16 common, 4 other (all same "other" value).
  std::vector<T> data(20, common);
  data[3] = other;
  data[7] = other;
  data[11] = other;
  data[15] = other;

  // Test both inner encoding variants: Dictionary and Constant.
  for (int variant = 0; variant < 2; ++variant) {
    SCOPED_TRACE(
        variant == 0 ? "MC->Dictionary others" : "MC->Constant others");
    // mcBuffer must outlive encoding since the encoding references its data.
    nimble::Buffer mcBuffer{*this->pool_};
    std::unique_ptr<nimble::Encoding> encoding;
    if (variant == 0) {
      encoding = this->encodeMainlyConstantDictionary(data);
    } else {
      auto serialized =
          this->serializeMainlyConstantWithConstantOthers(data, mcBuffer);
      encoding = this->decodeEncoding(serialized);
    }

    ASSERT_TRUE(encoding->dictionaryEnabled());
    // MC alphabet = [innerAlphabet..., commonValue].
    // Inner has 1 entry (single other value), so total = 2.
    EXPECT_EQ(encoding->dictionarySize(), 2);

    std::vector<uint32_t> indices(data.size());
    encoding->materializeIndices(data.size(), indices.data());

    const auto* entries = static_cast<const T*>(encoding->dictionaryEntries());
    for (size_t i = 0; i < data.size(); ++i) {
      ASSERT_LT(indices[i], encoding->dictionarySize()) << "row " << i;
      EXPECT_EQ(entries[indices[i]], data[i]) << "row " << i;
    }
  }
}

TYPED_TEST(DictionaryApiTypedTest, rleWithConstantLeaf) {
  using T = TypeParam;
  using PhysicalType = typename nimble::TypeTraits<T>::physicalType;

  // Build RLE manually with Constant as the run-values child encoding.
  T value;
  if constexpr (std::is_same_v<T, std::string_view>) {
    this->stringPool_ = {"repeated_run_value"};
    value = std::string_view(this->stringPool_[0]);
  } else {
    value = T(55);
  }

  // 3 runs of the same value, different lengths.
  nimble::Vector<uint32_t> runLengths{this->pool_.get()};
  runLengths.push_back(5);
  runLengths.push_back(3);
  runLengths.push_back(7);
  const uint32_t totalRows = 15;

  // Encode run lengths as Trivial.
  auto rlSpan = std::span<const uint32_t>(runLengths.data(), runLengths.size());
  nimble::Buffer rlBuf{*this->pool_};
  nimble::EncodingSelection<uint32_t> rlSel{
      {.encodingType = nimble::EncodingType::Trivial},
      nimble::Statistics<uint32_t>::create(rlSpan),
      std::make_unique<TestTrivialEncodingSelectionPolicy<uint32_t>>(
          false, false)};
  auto serializedRunLengths =
      nimble::TrivialEncoding<uint32_t>::encode(rlSel, rlSpan, rlBuf);

  // Encode run values as Constant (all 3 runs have the same value).
  nimble::Vector<PhysicalType> runValues{this->pool_.get()};
  for (int i = 0; i < 3; ++i) {
    runValues.push_back(reinterpret_cast<const PhysicalType&>(value));
  }
  auto rvSpan =
      std::span<const PhysicalType>(runValues.data(), runValues.size());
  nimble::Buffer rvBuf{*this->pool_};
  nimble::EncodingSelection<PhysicalType> rvSel{
      {.encodingType = nimble::EncodingType::Constant},
      nimble::Statistics<PhysicalType>::create(rvSpan),
      std::make_unique<TestTrivialEncodingSelectionPolicy<T>>(false, false)};
  auto serializedRunValues =
      nimble::ConstantEncoding<T>::encode(rvSel, rvSpan, rvBuf);

  // Assemble RLE binary.
  const uint32_t encodingSize = nimble::Encoding::kPrefixSize + 4 +
      serializedRunLengths.size() + serializedRunValues.size();
  char* reserved = this->buffer_->reserve(encodingSize);
  char* pos = reserved;
  nimble::encoding::writeChar(
      static_cast<char>(nimble::EncodingType::RLE), pos);
  nimble::encoding::writeChar(
      static_cast<char>(nimble::TypeTraits<T>::dataType), pos);
  nimble::encoding::writeUint32(totalRows, pos);
  nimble::encoding::writeString(serializedRunLengths, pos);
  nimble::encoding::writeBytes(serializedRunValues, pos);

  auto encoding =
      this->decodeEncoding(std::string_view(reserved, encodingSize));
  if constexpr (!nimble::isStringType<T>()) {
    EXPECT_FALSE(encoding->dictionaryEnabled());
    std::vector<T> result(totalRows);
    encoding->materialize(totalRows, result.data());
    for (size_t i = 0; i < totalRows; ++i) {
      EXPECT_EQ(result[i], value) << "row " << i;
    }
    return;
  }
  ASSERT_TRUE(encoding->dictionaryEnabled());
  EXPECT_EQ(encoding->dictionarySize(), 1);

  std::vector<uint32_t> indices(totalRows);
  encoding->materializeIndices(totalRows, indices.data());

  const auto* entries = static_cast<const T*>(encoding->dictionaryEntries());
  for (size_t i = 0; i < totalRows; ++i) {
    ASSERT_EQ(indices[i], 0) << "row " << i;
    EXPECT_EQ(entries[0], value);
  }
}

// Fuzz MC→Constant: randomize row count, common-value ratio, and non-common
// positions. Always exactly 2 distinct values so the inner encoding is
// Constant.
TYPED_TEST(DictionaryApiTypedTest, mainlyConstantWithConstantLeafFuzz) {
  using T = TypeParam;

  for (int run = 0; run < 20; ++run) {
    auto seed = folly::Random::rand32();
    std::mt19937 rng(seed);
    const auto numRows = 10 + rng() % 500;
    const int commonPct = 55 + rng() % 40;

    T common;
    T other;
    if constexpr (std::is_same_v<T, std::string_view>) {
      this->stringPool_ = {
          "common_" + std::to_string(run), "other_" + std::to_string(run)};
      common = std::string_view(this->stringPool_[0]);
      other = std::string_view(this->stringPool_[1]);
    } else {
      common = static_cast<T>(100 + run);
      other = static_cast<T>(42 + run);
    }

    std::vector<T> data;
    data.reserve(numRows);
    bool hasOther = false;
    for (size_t i = 0; i < numRows; ++i) {
      if (static_cast<int>(rng() % 100) < commonPct) {
        data.push_back(common);
      } else {
        data.push_back(other);
        hasOther = true;
      }
    }
    // Ensure at least one non-common value so MC is valid.
    if (!hasOther) {
      data[rng() % numRows] = other;
    }

    SCOPED_TRACE(
        fmt::format(
            "run={} seed={} numRows={} commonPct={}",
            run,
            seed,
            numRows,
            commonPct));

    nimble::Buffer mcBuffer{*this->pool_};
    auto serialized =
        this->serializeMainlyConstantWithConstantOthers(data, mcBuffer);
    auto encoding = this->decodeEncoding(serialized);

    ASSERT_TRUE(encoding->dictionaryEnabled());
    ASSERT_EQ(encoding->dictionarySize(), 2);

    std::vector<uint32_t> indices(data.size());
    encoding->materializeIndices(data.size(), indices.data());

    const auto* entries = static_cast<const T*>(encoding->dictionaryEntries());
    for (size_t i = 0; i < data.size(); ++i) {
      ASSERT_LT(indices[i], encoding->dictionarySize()) << "row " << i;
      EXPECT_EQ(entries[indices[i]], data[i]) << "row " << i;
    }
  }
}

// Fuzz RLE→Constant: randomize number of runs and run lengths while
// run values are all constant. Exercises materializeIndices through
// RLE with a Constant values child.
TYPED_TEST(DictionaryApiTypedTest, rleWithConstantLeafFuzz) {
  using T = TypeParam;
  using PhysicalType = typename nimble::TypeTraits<T>::physicalType;

  for (int run = 0; run < 20; ++run) {
    auto seed = folly::Random::rand32();
    std::mt19937 rng(seed);
    const auto numRuns = 1 + rng() % 30;
    const auto maxRunLength = 1 + rng() % 50;

    T value;
    if constexpr (std::is_same_v<T, std::string_view>) {
      this->stringPool_ = {"constant_run_value_" + std::to_string(run)};
      value = std::string_view(this->stringPool_[0]);
    } else {
      value = static_cast<T>(run + 1);
    }

    nimble::Vector<uint32_t> runLengths{this->pool_.get()};
    uint32_t totalRows = 0;
    for (size_t i = 0; i < numRuns; ++i) {
      const uint32_t len = 1 + rng() % maxRunLength;
      runLengths.push_back(len);
      totalRows += len;
    }

    SCOPED_TRACE(
        fmt::format(
            "run={} seed={} numRuns={} totalRows={}",
            run,
            seed,
            numRuns,
            totalRows));

    // Encode run lengths as Trivial.
    auto rlSpan =
        std::span<const uint32_t>(runLengths.data(), runLengths.size());
    nimble::Buffer rlBuf{*this->pool_};
    nimble::EncodingSelection<uint32_t> rlSel{
        {.encodingType = nimble::EncodingType::Trivial},
        nimble::Statistics<uint32_t>::create(rlSpan),
        std::make_unique<TestTrivialEncodingSelectionPolicy<uint32_t>>(
            false, false)};
    auto serializedRunLengths =
        nimble::TrivialEncoding<uint32_t>::encode(rlSel, rlSpan, rlBuf);

    // Encode run values as Constant.
    nimble::Vector<PhysicalType> runValues{this->pool_.get()};
    for (size_t i = 0; i < numRuns; ++i) {
      runValues.push_back(reinterpret_cast<const PhysicalType&>(value));
    }
    auto rvSpan =
        std::span<const PhysicalType>(runValues.data(), runValues.size());
    nimble::Buffer rvBuf{*this->pool_};
    nimble::EncodingSelection<PhysicalType> rvSel{
        {.encodingType = nimble::EncodingType::Constant},
        nimble::Statistics<PhysicalType>::create(rvSpan),
        std::make_unique<TestTrivialEncodingSelectionPolicy<T>>(false, false)};
    auto serializedRunValues =
        nimble::ConstantEncoding<T>::encode(rvSel, rvSpan, rvBuf);

    // Assemble RLE binary.
    const uint32_t encodingSize = nimble::Encoding::kPrefixSize + 4 +
        serializedRunLengths.size() + serializedRunValues.size();
    char* reserved = this->buffer_->reserve(encodingSize);
    char* pos = reserved;
    nimble::encoding::writeChar(
        static_cast<char>(nimble::EncodingType::RLE), pos);
    nimble::encoding::writeChar(
        static_cast<char>(nimble::TypeTraits<T>::dataType), pos);
    nimble::encoding::writeUint32(totalRows, pos);
    nimble::encoding::writeString(serializedRunLengths, pos);
    nimble::encoding::writeBytes(serializedRunValues, pos);

    auto encoding =
        this->decodeEncoding(std::string_view(reserved, encodingSize));
    if constexpr (!nimble::isStringType<T>()) {
      EXPECT_FALSE(encoding->dictionaryEnabled());
      std::vector<T> result(totalRows);
      encoding->materialize(totalRows, result.data());
      for (size_t i = 0; i < totalRows; ++i) {
        EXPECT_EQ(result[i], value) << "row " << i;
      }
      continue;
    }
    ASSERT_TRUE(encoding->dictionaryEnabled());
    ASSERT_EQ(encoding->dictionarySize(), 1);

    std::vector<uint32_t> indices(totalRows);
    encoding->materializeIndices(totalRows, indices.data());

    const auto* entries = static_cast<const T*>(encoding->dictionaryEntries());
    for (size_t i = 0; i < totalRows; ++i) {
      ASSERT_EQ(indices[i], 0) << "row " << i;
      EXPECT_EQ(entries[0], value);
    }
  }
}

// Non-templated test for unsupported encodings and bool dictionary.
class DictionaryApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
  std::vector<velox::BufferPtr> stringBuffers_;
};

// Verifies that every non-dictionary encoding type returns
// dictionaryEnabled()=false and aborts on dictionary-specific methods.
TEST_F(DictionaryApiTest, unsupportedEncodings) {
  auto verifyUnsupported = [](nimble::Encoding& enc, const char* name) {
    SCOPED_TRACE(name);
    EXPECT_FALSE(enc.dictionaryEnabled());
    EXPECT_THROW(enc.dictionarySize(), nimble::NimbleInternalError);
    EXPECT_THROW(enc.dictionaryEntry(0), nimble::NimbleInternalError);
    EXPECT_THROW(enc.dictionaryEntries(), nimble::NimbleInternalError);
  };

  // Trivial<string_view>
  {
    nimble::Vector<std::string_view> vals{pool_.get()};
    vals.push_back("a");
    vals.push_back("b");
    auto span = std::span<const std::string_view>(vals.data(), vals.size());
    nimble::EncodingSelection<std::string_view> sel{
        {.encodingType = nimble::EncodingType::Trivial},
        nimble::Statistics<std::string_view>::create(span),
        std::make_unique<TestTrivialEncodingSelectionPolicy<std::string_view>>(
            false, false)};
    auto encoded =
        nimble::TrivialEncoding<std::string_view>::encode(sel, span, *buffer_);
    auto enc = nimble::EncodingFactory().create(
        *pool_, encoded, [this](uint32_t size) {
          stringBuffers_.push_back(
              velox::AlignedBuffer::allocate<char>(size, pool_.get()));
          return stringBuffers_.back()->asMutable<void>();
        });
    verifyUnsupported(*enc, "Trivial");
  }

  // RLE<uint32_t>
  {
    std::vector<uint32_t> data(10, 5);
    auto span = std::span<const uint32_t>(data);
    nimble::EncodingSelection<uint32_t> sel{
        {.encodingType = nimble::EncodingType::RLE},
        nimble::Statistics<uint32_t>::create(span),
        std::make_unique<TestTrivialEncodingSelectionPolicy<uint32_t>>(
            false, false)};
    auto encoded = nimble::RLEEncoding<uint32_t>::encode(sel, span, *buffer_);
    auto enc = nimble::EncodingFactory().create(*pool_, encoded, nullptr);
    verifyUnsupported(*enc, "RLE");
  }

  // FixedBitWidth<uint32_t>
  {
    std::vector<uint32_t> data = {1, 3, 7, 15};
    auto span = std::span<const uint32_t>(data);
    nimble::EncodingSelection<uint32_t> sel{
        {.encodingType = nimble::EncodingType::FixedBitWidth},
        nimble::Statistics<uint32_t>::create(span),
        std::make_unique<TestTrivialEncodingSelectionPolicy<uint32_t>>(
            false, false)};
    auto encoded =
        nimble::FixedBitWidthEncoding<uint32_t>::encode(sel, span, *buffer_);
    auto enc = nimble::EncodingFactory().create(*pool_, encoded, nullptr);
    verifyUnsupported(*enc, "FixedBitWidth");
  }

  // MainlyConstant<uint32_t>
  {
    std::vector<uint32_t> data(10, 99);
    data.push_back(1);
    auto span = std::span<const uint32_t>(data);
    nimble::EncodingSelection<uint32_t> sel{
        {.encodingType = nimble::EncodingType::MainlyConstant},
        nimble::Statistics<uint32_t>::create(span),
        std::make_unique<TestTrivialEncodingSelectionPolicy<uint32_t>>(
            false, false)};
    auto encoded =
        nimble::MainlyConstantEncoding<uint32_t>::encode(sel, span, *buffer_);
    auto enc = nimble::EncodingFactory().create(*pool_, encoded, nullptr);
    verifyUnsupported(*enc, "MainlyConstant");
  }

  // SparseBool
  {
    nimble::Vector<bool> vals{pool_.get()};
    for (int i = 0; i < 10; ++i) {
      vals.push_back(i != 5);
    }
    auto span = std::span<const bool>(vals.data(), vals.size());
    nimble::EncodingSelection<bool> sel{
        {.encodingType = nimble::EncodingType::SparseBool},
        nimble::Statistics<bool>::create(span),
        std::make_unique<TestTrivialEncodingSelectionPolicy<bool>>(
            false, false)};
    auto encoded = nimble::SparseBoolEncoding::encode(sel, span, *buffer_);
    auto enc = nimble::EncodingFactory().create(*pool_, encoded, nullptr);
    verifyUnsupported(*enc, "SparseBool");
  }

  // Varint<uint64_t>
  {
    std::vector<uint64_t> data = {100, 200, 300};
    auto span = std::span<const uint64_t>(data);
    nimble::EncodingSelection<uint64_t> sel{
        {.encodingType = nimble::EncodingType::Varint},
        nimble::Statistics<uint64_t>::create(span),
        std::make_unique<TestTrivialEncodingSelectionPolicy<uint64_t>>(
            false, false)};
    auto encoded =
        nimble::VarintEncoding<uint64_t>::encode(sel, span, *buffer_);
    auto enc = nimble::EncodingFactory().create(*pool_, encoded, nullptr);
    verifyUnsupported(*enc, "Varint");
  }

  // Delta<uint32_t>
  {
    std::vector<uint32_t> data = {10, 20, 30};
    auto span = std::span<const uint32_t>(data);
    nimble::EncodingSelection<uint32_t> sel{
        {.encodingType = nimble::EncodingType::Delta},
        nimble::Statistics<uint32_t>::create(span),
        std::make_unique<TestTrivialEncodingSelectionPolicy<uint32_t>>(
            false, false)};
    auto encoded = nimble::DeltaEncoding<uint32_t>::encode(sel, span, *buffer_);
    auto enc = nimble::EncodingFactory().create(*pool_, encoded, nullptr);
    verifyUnsupported(*enc, "Delta");
  }
}
