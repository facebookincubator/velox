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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

namespace facebook::nimble {
namespace {

class EncodingRebindTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool();
    buffer_ = std::make_unique<Buffer>(*pool_);
  }

  std::function<void*(uint32_t)> stringBufferFactory() {
    return [this](uint32_t size) {
      auto& buffer = stringBuffers_.emplace_back(
          velox::AlignedBuffer::allocate<char>(size, pool_.get()));
      return buffer->asMutable<void>();
    };
  }

  // Test inputs hold bools as uint8_t, as std::vector<bool> is not used here.
  template <typename T>
  using Input = std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>;

  template <typename T>
  Vector<T> makeVector(const std::vector<Input<T>>& values) {
    Vector<T> vector{pool_.get(), values.size()};
    for (size_t i = 0; i < values.size(); ++i) {
      vector[i] = static_cast<T>(values[i]);
    }
    return vector;
  }

  template <typename T>
  std::vector<Input<T>> materialize(Encoding& encoding) {
    Vector<T> output{pool_.get(), encoding.rowCount()};
    encoding.materialize(encoding.rowCount(), output.data());
    std::vector<Input<T>> values;
    for (const auto value : output) {
      values.push_back(static_cast<Input<T>>(value));
    }
    return values;
  }

  // Encodes `first`, reads one row, rebinds it to each of `rest` and checks
  // that every row of each comes back; the encoding type must be E for every
  // input.
  template <typename E, typename T>
  void expectRebinds(
      const std::vector<Input<T>>& first,
      const std::vector<std::vector<Input<T>>>& rest,
      CompressionType compression = CompressionType::Uncompressed) {
    using Encoder = test::Encoder<E>;
    const auto firstData =
        Encoder::encode(*buffer_, makeVector<T>(first), compression);
    ASSERT_EQ(EncodingPrefix::encodingType(firstData), Encoder::encodingType());
    auto encoding = EncodingFactory().create(
        *pool_, firstData, stringBufferFactory(), Encoding::Options{});
    Vector<T> one{pool_.get(), 1};
    encoding->materialize(1, one.data());
    EXPECT_EQ(static_cast<Input<T>>(one[0]), first[0]);
    for (const auto& values : rest) {
      const auto data =
          Encoder::encode(*buffer_, makeVector<T>(values), compression);
      ASSERT_EQ(EncodingPrefix::encodingType(data), Encoder::encodingType());
      ASSERT_TRUE(encoding->rebind(data, stringBufferFactory()));
      EXPECT_EQ(encoding->rowCount(), values.size());
      EXPECT_EQ(materialize<T>(*encoding), values);
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<Buffer> buffer_;
  std::vector<velox::BufferPtr> stringBuffers_;
};

TEST_F(EncodingRebindTest, constant) {
  expectRebinds<ConstantEncoding<int32_t>, int32_t>(
      {7, 7, 7}, {{9, 9}, {-1, -1, -1, -1}});
  expectRebinds<ConstantEncoding<bool>, bool>({1, 1}, {{0}});
  expectRebinds<ConstantEncoding<double>, double>({1.5}, {{2.5, 2.5}});
  const std::string a = "alpha";
  const std::string b = "beta";
  expectRebinds<ConstantEncoding<std::string_view>, std::string_view>(
      {a, a}, {{b, b, b}});
}

TEST_F(EncodingRebindTest, constantStringBufferLifetime) {
  using Encoder = test::Encoder<ConstantEncoding<std::string_view>>;
  auto encoding = EncodingFactory().create(
      *pool_,
      Encoder::encode(*buffer_, makeVector<std::string_view>({"alpha"})),
      stringBufferFactory());
  std::string data{Encoder::encode(
      *buffer_, makeVector<std::string_view>({"beta", "beta"}))};
  ASSERT_TRUE(encoding->rebind(data, stringBufferFactory()));
  const auto values = materialize<std::string_view>(*encoding);

  std::fill(data.begin(), data.end(), 'x');
  EXPECT_THAT(
      materialize<std::string_view>(*encoding),
      testing::ElementsAre("beta", "beta"));
  ASSERT_TRUE(encoding->rebind(
      Encoder::encode(*buffer_, makeVector<std::string_view>({"gamma"})),
      stringBufferFactory()));
  EXPECT_THAT(
      materialize<std::string_view>(*encoding), testing::ElementsAre("gamma"));
  EXPECT_THAT(values, testing::ElementsAre("beta", "beta"));
}

TEST_F(EncodingRebindTest, trivial) {
  for (const auto compression :
       {CompressionType::Uncompressed, CompressionType::Zstd}) {
    expectRebinds<TrivialEncoding<int64_t>, int64_t>(
        {1, 2, 3}, {{4, 5}, {6, 7, 8, 9}}, compression);
    expectRebinds<TrivialEncoding<float>, float>(
        {1.0f, 2.5f}, {{3.5f, 4.0f, 5.5f}}, compression);
    expectRebinds<TrivialEncoding<bool>, bool>(
        {1, 0, 1}, {{0, 0}, {1}}, compression);
    const std::vector<std::string> strings{"a", "bb", "ccc", "dddd"};
    expectRebinds<TrivialEncoding<std::string_view>, std::string_view>(
        {strings[0], strings[1]},
        {{strings[2], strings[3], strings[0]}, {strings[1]}},
        compression);
  }
}

TEST_F(EncodingRebindTest, fixedBitWidth) {
  expectRebinds<FixedBitWidthEncoding<uint32_t>, uint32_t>(
      {1, 2, 3, 4}, {{100, 200}, {70'000, 1, 2}});
}

TEST_F(EncodingRebindTest, varint) {
  expectRebinds<VarintEncoding<uint32_t>, uint32_t>(
      {1, 300, 70'000}, {{5, 6}, {1u << 30, 0}});
}

TEST_F(EncodingRebindTest, mainlyConstant) {
  expectRebinds<MainlyConstantEncoding<int32_t>, int32_t>(
      {5, 5, 5, 5, 9, 5, 5, 5}, {{1, 1, 1, 2, 1, 1, 1, 1, 1, 3}});
}

TEST_F(EncodingRebindTest, mainlyConstantDictionaryEntries) {
  // Force a Dictionary child so this test does not depend on cost-based
  // selection choosing it for the uncommon values.
  const auto encode = [&](int32_t commonValue,
                          const std::vector<int32_t>& otherValues) {
    const auto isCommon = test::Encoder<TrivialEncoding<bool>>::encode(
        *buffer_, makeVector<bool>({1, 0, 1, 0, 1, 0}));
    const auto dictionary = test::Encoder<DictionaryEncoding<int32_t>>::encode(
        *buffer_, makeVector<int32_t>(otherValues));
    const auto encodingSize = static_cast<uint32_t>(
        EncodingPrefix::serializedSize(6, /*useVarint=*/false) +
        2 * sizeof(uint32_t) + isCommon.size() + dictionary.size() +
        sizeof(int32_t));
    char* const reserved = buffer_->reserve(encodingSize);
    char* pos = reserved;
    EncodingPrefix::serialize(
        EncodingType::MainlyConstant,
        DataType::Int32,
        6,
        /*useVarint=*/false,
        pos);
    encoding::writeString(isCommon, pos);
    encoding::writeString(dictionary, pos);
    encoding::write<int32_t>(commonValue, pos);
    return std::string_view{reserved, encodingSize};
  };

  auto encoding = EncodingFactory().create(
      *pool_, encode(7, {10, 20, 10}), stringBufferFactory());
  ASSERT_TRUE(encoding->dictionaryEnabled());
  ASSERT_EQ(encoding->dictionarySize(), 3);
  const auto* alphabet =
      static_cast<const int32_t*>(encoding->dictionaryEntries());
  EXPECT_THAT(
      std::vector<int32_t>(alphabet, alphabet + encoding->dictionarySize()),
      testing::UnorderedElementsAre(10, 20, 7));

  ASSERT_TRUE(encoding->rebind(encode(9, {30, 40, 30}), stringBufferFactory()));
  ASSERT_TRUE(encoding->dictionaryEnabled());
  ASSERT_EQ(encoding->dictionarySize(), 3);
  alphabet = static_cast<const int32_t*>(encoding->dictionaryEntries());
  EXPECT_THAT(
      std::vector<int32_t>(alphabet, alphabet + encoding->dictionarySize()),
      testing::UnorderedElementsAre(30, 40, 9));
  EXPECT_EQ(
      materialize<int32_t>(*encoding),
      std::vector<int32_t>({9, 30, 9, 40, 9, 30}));
}

TEST_F(EncodingRebindTest, dictionary) {
  expectRebinds<DictionaryEncoding<int64_t>, int64_t>(
      {10, 20, 10, 20, 10}, {{30, 40, 30, 30}, {50, 50, 60}});
  const std::vector<std::string> strings{"x", "yy", "zzz"};
  expectRebinds<DictionaryEncoding<std::string_view>, std::string_view>(
      {strings[0], strings[1], strings[0]},
      {{strings[2], strings[2], strings[1]}});
}

TEST_F(EncodingRebindTest, nullable) {
  using Encoder = test::Encoder<NullableEncoding<int32_t>>;
  const auto first = Encoder::encodeNullable(
      *buffer_, makeVector<int32_t>({1, 2}), makeVector<bool>({1, 0, 1}));
  auto encoding = EncodingFactory().create(
      *pool_, first, stringBufferFactory(), Encoding::Options{});
  const auto second = Encoder::encodeNullable(
      *buffer_,
      makeVector<int32_t>({7, 8, 9}),
      makeVector<bool>({0, 1, 1, 0, 1}));
  ASSERT_TRUE(encoding->rebind(second, stringBufferFactory()));
  EXPECT_EQ(encoding->rowCount(), 5);
  Vector<int32_t> values{pool_.get(), 5};
  // materializeNullable() writes a bitmap with a set bit per non-null row.
  uint64_t notNullBits{0};
  ASSERT_EQ(
      encoding->materializeNullable(
          5, values.data(), [&]() { return &notNullBits; }),
      3);
  std::vector<uint8_t> notNull;
  notNull.reserve(5);
  for (int row = 0; row < 5; ++row) {
    notNull.push_back(velox::bits::isBitSet(&notNullBits, row));
  }
  EXPECT_THAT(notNull, testing::ElementsAre(0, 1, 1, 0, 1));
  EXPECT_THAT(
      (std::vector<int32_t>{values[1], values[2], values[4]}),
      testing::ElementsAre(7, 8, 9));
}

TEST_F(EncodingRebindTest, otherEncodingTypeIsRejected) {
  const auto trivial = test::Encoder<TrivialEncoding<int32_t>>::encode(
      *buffer_, makeVector<int32_t>({1, 2, 3}));
  const auto constant = test::Encoder<ConstantEncoding<int32_t>>::encode(
      *buffer_, makeVector<int32_t>({4, 4}));
  auto encoding = EncodingFactory().create(
      *pool_, trivial, stringBufferFactory(), Encoding::Options{});
  EXPECT_FALSE(encoding->rebind(constant, stringBufferFactory()));
  const auto otherType = test::Encoder<TrivialEncoding<int64_t>>::encode(
      *buffer_, makeVector<int64_t>({1, 2}));
  auto rebuilt = EncodingFactory().create(
      *pool_, trivial, stringBufferFactory(), Encoding::Options{});
  EXPECT_FALSE(rebuilt->rebind(otherType, stringBufferFactory()));
}

// rebind() checks the whole prefix before it reads anything else; nested
// encodings reach it through rebindChild().
TEST_F(EncodingRebindTest, truncatedPrefixIsRejected) {
  const auto trivial = test::Encoder<TrivialEncoding<int32_t>>::encode(
      *buffer_, makeVector<int32_t>({1, 2, 3}));
  auto encoding = EncodingFactory().create(
      *pool_, trivial, stringBufferFactory(), Encoding::Options{});
  NIMBLE_ASSERT_THROW(
      encoding->rebind(trivial.substr(0, 3), stringBufferFactory()),
      "Truncated encoding prefix.");
}

TEST_F(EncodingRebindTest, nestedMismatchRebuildsChild) {
  // A Dictionary whose indices change from one encoding type to another.
  auto encoding = EncodingFactory().create(
      *pool_,
      test::Encoder<DictionaryEncoding<int64_t>>::encode(
          *buffer_, makeVector<int64_t>({1, 2, 1, 2})),
      stringBufferFactory(),
      Encoding::Options{});
  std::vector<int64_t> many;
  many.reserve(1'000);
  for (int64_t i = 0; i < 1'000; ++i) {
    many.push_back(i % 300);
  }
  const auto data = test::Encoder<DictionaryEncoding<int64_t>>::encode(
      *buffer_,
      makeVector<int64_t>(many),
      CompressionType::Uncompressed,
      {},
      true);
  ASSERT_TRUE(encoding->rebind(data, stringBufferFactory()));
  EXPECT_EQ(materialize<int64_t>(*encoding), many);
}

} // namespace
} // namespace facebook::nimble
