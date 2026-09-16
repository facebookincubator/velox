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
#include "velox/dwio/nimble/encodings/ALPRDEncoding.h"

#include <gtest/gtest.h>
#include <bit>
#include <numeric>
#include <random>

#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/tools/EncodingUtilities.h"

namespace facebook::nimble {
namespace {

EncodingLayout makeLayout(EncodingType child = EncodingType::FixedBitWidth) {
  EncodingLayout leaf{child, {}, CompressionType::Uncompressed};
  return {
      EncodingType::ALPRD,
      {},
      CompressionType::Uncompressed,
      {leaf, leaf, leaf, leaf}};
}

template <typename T>
std::unique_ptr<EncodingSelectionPolicy<T>> makePolicy(EncodingLayout layout) {
  return std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
      std::move(layout), std::nullopt, [](DataType type) {
        return ManualEncodingSelectionPolicyFactory(
                   ManualEncodingSelectionPolicyFactory::
                       defaultEncodingReadFactors(),
                   std::nullopt)
            .createPolicy(type);
      });
}

template <typename T, bool UseVarint>
struct Config {
  using Value = T;
  static constexpr bool useVarint = UseVarint;
};

template <typename C>
class ALPRDEncodingTest : public ::testing::Test {
 protected:
  using T = typename C::Value;
  using Physical = typename TypeTraits<T>::physicalType;

  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<Buffer>(*pool_);
  }

  std::string_view encode(
      const std::vector<Physical>& values,
      EncodingLayout layout = makeLayout()) {
    std::vector<T> logical;
    for (const auto value : values) {
      logical.push_back(std::bit_cast<T>(value));
    }
    return EncodingFactory::encode<T>(
        makePolicy<T>(std::move(layout)), logical, *buffer_, options_);
  }

  std::unique_ptr<Encoding> decoder(std::string_view encoded) {
    return EncodingFactory(options_).create(*pool_, encoded, nullptr);
  }

  void check(std::string_view encoded, std::span<const Physical> expected) {
    for (auto useLegacy : {false, true}) {
      SCOPED_TRACE(useLegacy);
      auto reader = useLegacy
          ? legacy::EncodingFactory(options_).create(*pool_, encoded, nullptr)
          : decoder(encoded);
      ASSERT_EQ(reader->rowCount(), expected.size());
      std::vector<Physical> actual(expected.size());
      reader->materialize(actual.size(), actual.data());
      EXPECT_EQ(
          actual, (std::vector<Physical>{expected.begin(), expected.end()}));
    }
  }

  std::vector<Physical> valuesWithLateException(uint32_t count = 131077) {
    const Physical high = sizeof(T) == 8 ? 0x3ff1 : 0x3f81;
    constexpr auto shift = sizeof(T) * 8 - 16;
    const auto mask = (Physical{1} << shift) - 1;
    std::mt19937_64 random(0xA1F0);
    std::vector<Physical> values(count);
    for (auto& value : values) {
      value = (high << shift) | (random() & mask);
    }
    // The final row is outside the evenly spaced training sample.
    values.back() = (Physical{0xc001} << shift) | 17;
    return values;
  }

  // Construct a wire fixture without using the ALPRD writer or metadata code.
  void appendInteger(std::string& bytes, uint64_t value, uint8_t width) {
    for (uint8_t i = 0; i < width; ++i) {
      bytes.push_back(static_cast<char>(value >> (8 * i)));
    }
  }

  void appendVarint(std::string& bytes, uint32_t value) {
    while (value >= 128) {
      bytes.push_back(static_cast<char>((value & 127) | 128));
      value >>= 7;
    }
    bytes.push_back(static_cast<char>(value));
  }

  void appendPrefix(
      std::string& bytes,
      uint8_t encoding,
      DataType type,
      uint32_t count) {
    bytes.push_back(encoding);
    bytes.push_back(static_cast<char>(type));
    if (C::useVarint) {
      appendVarint(bytes, count);
    } else {
      appendInteger(bytes, count, 4);
    }
  }

  template <typename U>
  void appendChild(std::string& bytes, const std::vector<U>& values) {
    std::string child;
    appendPrefix(child, 0, TypeTraits<U>::dataType, values.size());
    child.push_back(0); // Uncompressed Trivial payload.
    for (auto value : values) {
      appendInteger(child, value, sizeof(U));
    }
    appendVarint(bytes, child.size());
    bytes += child;
  }

  std::string fixture(
      const std::vector<uint16_t>& dictionary = {1, 2},
      const std::vector<uint16_t>& codes = {1, 0, 1},
      const std::vector<Physical>& right = {1, 2, 3},
      const std::vector<uint32_t>& positions = {1},
      const std::vector<uint16_t>& exceptions = {3},
      uint8_t rightWidth = sizeof(T) * 8 - 16) {
    std::string bytes;
    appendPrefix(bytes, 26, TypeTraits<T>::dataType, codes.size());
    bytes.push_back(rightWidth);
    bytes.push_back(dictionary.size());
    appendVarint(bytes, positions.size());
    for (auto high : dictionary) {
      appendInteger(bytes, high, 2);
    }
    appendChild(bytes, codes);
    appendChild(bytes, right);
    if (!positions.empty()) {
      appendChild(bytes, positions);
      appendChild(bytes, exceptions);
    }
    return bytes;
  }

  Encoding::Options options_{.useVarintRowCount = C::useVarint};
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<Buffer> buffer_;
};

using Configs = ::testing::Types<
    Config<float, false>,
    Config<float, true>,
    Config<double, false>,
    Config<double, true>>;
TYPED_TEST_SUITE(ALPRDEncodingTest, Configs);

TYPED_TEST(ALPRDEncodingTest, independentWireFixture) {
  using Physical = typename TestFixture::Physical;
  constexpr auto shift = sizeof(Physical) * 8 - 16;
  const std::vector<Physical> expected{
      (Physical{2} << shift) | 1,
      (Physical{3} << shift) | 2,
      (Physical{2} << shift) | 3};
  const auto fixture = this->fixture();
  this->check(fixture, expected);
  const auto layout = EncodingLayoutCapture::capture(fixture, this->options_);
  EXPECT_EQ(layout.encodingType(), EncodingType::ALPRD);
  ASSERT_EQ(layout.childrenCount(), 4);
  EXPECT_EQ(layout.child(3)->encodingType(), EncodingType::Trivial);
  if constexpr (!TypeParam::useVarint) {
    EXPECT_NE(
        tools::getEncodingLabel(fixture).find("ALPRD"), std::string::npos);
  }
}

TYPED_TEST(ALPRDEncodingTest, allBitPatterns) {
  using T = typename TestFixture::T;
  using Physical = typename TestFixture::Physical;
  std::vector<Physical> values{
      std::bit_cast<Physical>(T{0}),
      std::bit_cast<Physical>(-T{0}),
      std::bit_cast<Physical>(std::numeric_limits<T>::infinity()),
      std::bit_cast<Physical>(-std::numeric_limits<T>::infinity()),
      std::bit_cast<Physical>(std::numeric_limits<T>::quiet_NaN()),
      std::bit_cast<Physical>(std::numeric_limits<T>::quiet_NaN()) | 37,
      std::bit_cast<Physical>(std::numeric_limits<T>::signaling_NaN()),
      std::bit_cast<Physical>(std::numeric_limits<T>::denorm_min()),
      std::bit_cast<Physical>(std::numeric_limits<T>::max()),
      1,
      std::numeric_limits<Physical>::max()};
  std::mt19937_64 random(73);
  for (uint32_t i = 0; i < 4099; ++i) {
    values.push_back(random());
  }
  for (auto child : {EncodingType::FixedBitWidth, EncodingType::Trivial}) {
    this->check(this->encode(values, makeLayout(child)), values);
  }
}

TYPED_TEST(ALPRDEncodingTest, dictionarySizesAndExceptionRates) {
  using Physical = typename TestFixture::Physical;
  constexpr uint32_t kRowCount = 521;
  constexpr uint8_t kShift = sizeof(Physical) * 8 - 16;
  for (uint16_t dictionarySize = 1; dictionarySize <= 8; ++dictionarySize) {
    for (uint32_t exceptionStride : {0, 2, 1}) {
      SCOPED_TRACE(
          fmt::format(
              "dictionary={} exceptionStride={}",
              dictionarySize,
              exceptionStride));
      std::vector<uint16_t> dictionary(dictionarySize);
      std::iota(dictionary.begin(), dictionary.end(), 1);
      std::vector<uint16_t> codes(kRowCount);
      std::vector<Physical> right(kRowCount);
      std::vector<Physical> expected(kRowCount);
      std::vector<uint32_t> positions;
      std::vector<uint16_t> exceptions;
      for (uint32_t i = 0; i < kRowCount; ++i) {
        codes[i] = i % dictionarySize;
        right[i] = i;
        auto high = dictionary[codes[i]];
        if (exceptionStride != 0 && i % exceptionStride == 0) {
          high = dictionarySize + 1;
          positions.push_back(i);
          exceptions.push_back(high);
        }
        expected[i] = (static_cast<Physical>(high) << kShift) | right[i];
      }
      const auto encoded =
          this->fixture(dictionary, codes, right, positions, exceptions);
      this->check(encoded, expected);
      const auto sliced = EncodingFactory::slice(
          encoded, 255, 3, *this->buffer_, this->options_);
      this->check(sliced, std::span<const Physical>(expected).subspan(255, 3));
      const auto twiceSliced =
          EncodingFactory::slice(sliced, 1, 1, *this->buffer_, this->options_);
      this->check(
          twiceSliced, std::span<const Physical>(expected).subspan(256, 1));
    }
  }
  // Exercise the opposite split boundary: one high bit and W - 1 low bits.
  const auto encoded = this->fixture(
      {0, 1}, {1, 0, 1}, {1, 2, 3}, {}, {}, sizeof(Physical) * 8 - 1);
  const std::vector<Physical> expected{
      (Physical{1} << (sizeof(Physical) * 8 - 1)) | 1,
      2,
      (Physical{1} << (sizeof(Physical) * 8 - 1)) | 3};
  this->check(encoded, expected);
}

TYPED_TEST(ALPRDEncodingTest, shortAndConstantInputs) {
  using Physical = typename TestFixture::Physical;
  for (uint32_t count :
       {1, 2, 7, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025}) {
    SCOPED_TRACE(count);
    std::vector<Physical> values(
        count, std::bit_cast<Physical>(typename TestFixture::T{1.25}));
    this->check(this->encode(values), values);
    auto layout = makeLayout(EncodingType::Constant);
    this->check(this->encode(values, std::move(layout)), values);
  }
  EXPECT_THROW(this->encode({}), NimbleException);
}

TYPED_TEST(ALPRDEncodingTest, skipResetAndSliceWithLargePositions) {
  using Physical = typename TestFixture::Physical;
  const auto values = this->valuesWithLateException();
  const auto encoded = this->encode(values);
  const auto metadata = detail::alprd::readMetadata(encoded, this->options_);
  ASSERT_GT(metadata.exceptionCount, 0);
  ASSERT_EQ(metadata.parameters.rightBitWidth, sizeof(Physical) * 8 - 16);
  this->check(encoded, values);
  auto reader = this->decoder(encoded);
  std::vector<Physical> actual(values.size());
  uint32_t position = 0;
  while (position < values.size()) {
    const auto count = std::min<uint32_t>(113, values.size() - position);
    reader->materialize(count, actual.data() + position);
    position += count;
  }
  EXPECT_EQ(actual, values);
  reader->reset();
  reader->skip(values.size() - 2);
  std::array<Physical, 2> tail;
  reader->materialize(2, tail.data());
  EXPECT_EQ(tail[0], values[values.size() - 2]);
  EXPECT_EQ(tail[1], values.back());
  reader->reset();
  reader->skip(values.size() - 1);
  reader->skip(1);
  reader->materialize(0, nullptr);
  EXPECT_THROW(reader->skip(1), NimbleException);
  for (const auto& [offset, count] : std::vector<std::pair<uint32_t, uint32_t>>{
           {0, 31},
           {17, 513},
           {65536, 1000},
           {uint32_t(values.size() - 5), 5}}) {
    const auto sliced = EncodingFactory::slice(
        encoded, offset, count, *this->buffer_, this->options_);
    EXPECT_EQ(EncodingPrefix::encodingType(sliced), EncodingType::ALPRD);
    this->check(
        sliced, std::span<const Physical>(values).subspan(offset, count));
  }
}

TYPED_TEST(ALPRDEncodingTest, replayUsesCurrentDictionary) {
  using Physical = typename TestFixture::Physical;
  const auto firstValues = this->valuesWithLateException(4099);
  const auto first = this->encode(firstValues);
  auto secondValues = firstValues;
  for (auto& value : secondValues) {
    value ^= Physical{1} << (sizeof(Physical) * 8 - 1);
  }
  const auto replayed = this->encode(
      secondValues, EncodingLayoutCapture::capture(first, this->options_));
  this->check(replayed, secondValues);
  EXPECT_NE(
      detail::alprd::readMetadata(first, this->options_)
          .parameters.dictionary[0],
      detail::alprd::readMetadata(replayed, this->options_)
          .parameters.dictionary[0]);
}

TYPED_TEST(ALPRDEncodingTest, replayAddsAndRemovesExceptionChildren) {
  const auto withException = this->valuesWithLateException(4099);
  auto withoutException = withException;
  withoutException.back() = withoutException.front();
  const auto first = this->encode(withoutException);
  const auto firstLayout =
      EncodingLayoutCapture::capture(first, this->options_);
  ASSERT_EQ(firstLayout.childrenCount(), 4);
  EXPECT_FALSE(firstLayout.child(2).has_value());
  EXPECT_FALSE(firstLayout.child(3).has_value());
  const auto second = this->encode(withException, firstLayout);
  this->check(second, withException);
  const auto secondLayout =
      EncodingLayoutCapture::capture(second, this->options_);
  ASSERT_EQ(secondLayout.childrenCount(), 4);
  EXPECT_TRUE(secondLayout.child(2).has_value());
  EXPECT_TRUE(secondLayout.child(3).has_value());
  const auto third = this->encode(withoutException, secondLayout);
  this->check(third, withoutException);
  const auto thirdLayout =
      EncodingLayoutCapture::capture(third, this->options_);
  EXPECT_FALSE(thirdLayout.child(2).has_value());
  EXPECT_FALSE(thirdLayout.child(3).has_value());
}

TYPED_TEST(ALPRDEncodingTest, rejectsCorruptMetadataAndValues) {
  using T = typename TestFixture::T;
  using Physical = typename TestFixture::Physical;
  const auto valid = this->fixture();
  for (size_t size = 0; size < valid.size(); ++size) {
    SCOPED_TRACE(size);
    EXPECT_THROW(
        (ALPRDEncoding<T>(
            *this->pool_,
            std::string_view(valid).substr(0, size),
            nullptr,
            this->options_)),
        NimbleException);
  }
  auto corrupt = valid + 'x';
  EXPECT_THROW(this->decoder(corrupt), NimbleException);
  const auto prefix =
      EncodingPrefix::prefixSize(valid, this->options_.useVarintRowCount);
  for (const auto width : {0, int(sizeof(T) * 8), int(sizeof(T) * 8 - 17)}) {
    corrupt = valid;
    corrupt[prefix] = width;
    EXPECT_THROW(this->decoder(corrupt), NimbleException);
  }
  corrupt = valid;
  corrupt[prefix + 1] = 9;
  EXPECT_THROW(this->decoder(corrupt), NimbleException);
  EXPECT_THROW(this->decoder(this->fixture({})), NimbleException);
  EXPECT_THROW(this->decoder(this->fixture({1, 1})), NimbleException);
  EXPECT_THROW(
      this->decoder(this->fixture({1, 2}, {1, 0, 1}, {1, 2})), NimbleException);
  EXPECT_THROW(
      this->decoder(this->fixture({1, 2}, {1, 0, 1}, {1, 2, 3}, {3})),
      NimbleException);
  EXPECT_THROW(
      this->decoder(
          this->fixture({1, 2}, {1, 0, 1}, {1, 2, 3}, {1, 1}, {3, 3})),
      NimbleException);
  EXPECT_THROW(
      this->decoder(
          this->fixture({1, 2}, {1, 0, 1}, {1, 2, 3}, {2, 1}, {3, 3})),
      NimbleException);
  EXPECT_THROW(
      this->decoder(this->fixture({1, 2}, {1, 0, 1}, {1, 2, 3}, {1}, {})),
      NimbleException);
  EXPECT_THROW(
      this->decoder(this->fixture(
          {0, 1}, {1, 0, 1}, {1, 2, 3}, {1}, {2}, sizeof(T) * 8 - 1)),
      NimbleException);
  corrupt = valid;
  corrupt.replace(prefix + 2, 1, std::string(5, static_cast<char>(0x80)));
  EXPECT_THROW(this->decoder(corrupt), NimbleException);
  for (const auto& invalid :
       {this->fixture({1, 2}, {2, 0, 1}),
        this->fixture(
            {1, 2}, {1, 0, 1}, {Physical{1} << (sizeof(T) * 8 - 16), 2, 3})}) {
    auto reader = this->decoder(invalid);
    std::array<Physical, 3> output;
    EXPECT_THROW(reader->materialize(3, output.data()), NimbleException);
  }
}

} // namespace
} // namespace facebook::nimble
