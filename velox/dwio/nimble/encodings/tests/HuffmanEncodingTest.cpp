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
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <numeric>

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

namespace facebook::nimble::test {
namespace {

class HuffmanEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<Buffer>(*pool_);
  }

  template <typename T>
  std::unique_ptr<Encoding> encode(const Vector<T>& values) {
    return Encoder<HuffmanEncoding<T>>::createEncoding(
        *buffer_, values, nullptr);
  }

  template <typename T>
  std::string_view encodeData(const Vector<T>& values) {
    return Encoder<HuffmanEncoding<T>>::encode(*buffer_, values);
  }

  template <typename T>
  void verifyFactoryRoundTrip() {
    Vector<T> values{pool_.get()};
    constexpr std::array<T, 5> alphabet{
        std::numeric_limits<T>::lowest(),
        T{0},
        T{1},
        std::numeric_limits<T>::max(),
        static_cast<T>(std::numeric_limits<T>::max() / 2)};
    for (uint32_t row = 0; row < 1025; ++row) {
      values.push_back(alphabet[row % alphabet.size()]);
    }

    const auto encoded = encodeData(values);
    for (const auto useLegacy : {false, true}) {
      SCOPED_TRACE(fmt::format("legacy={}", useLegacy));
      std::unique_ptr<Encoding> encoding = useLegacy
          ? legacy::EncodingFactory().create(*pool_, encoded, nullptr)
          : EncodingFactory().create(*pool_, encoded, nullptr);
      EXPECT_EQ(encoding->encodingType(), EncodingType::Huffman);
      EXPECT_EQ(encoding->dataType(), TypeTraits<T>::dataType);
      EXPECT_EQ(encoding->rowCount(), values.size());

      Vector<T> result{pool_.get(), values.size()};
      encoding->materialize(
          static_cast<uint32_t>(values.size()), result.data());
      EXPECT_TRUE(std::equal(result.begin(), result.end(), values.begin()));
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<Buffer> buffer_;
};

class TestReader {
 public:
  velox::BufferPtr& nullsInReadRange() {
    return nullsInReadRange_;
  }

  const uint64_t* rawNullsInReadRange() const {
    return nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
  }

  bool returnReaderNulls() const {
    return false;
  }

 private:
  velox::BufferPtr nullsInReadRange_;
};

template <typename T>
class IntegralReadWithVisitor {
 public:
  using DataType = T;
  using Extract = std::nullptr_t;

  explicit IntegralReadWithVisitor(std::vector<vector_size_t> rows)
      : rows_{std::move(rows)} {}

  TestReader& reader() {
    return reader_;
  }

  vector_size_t numRows() const {
    return rows_.size();
  }

  vector_size_t rowAt(vector_size_t index) const {
    return rows_[index];
  }

  vector_size_t currentRow() const {
    return rowAt(rowIndex_);
  }

  void process(T value, bool& atEnd) {
    values_.push_back(value);
    addRowIndex(1);
    atEnd = this->atEnd();
  }

  void processNull(bool& atEnd) {
    addRowIndex(1);
    atEnd = this->atEnd();
  }

  bool allowNulls() const {
    return false;
  }

  void addRowIndex(vector_size_t count) {
    rowIndex_ += count;
  }

  void addNumValues(vector_size_t /*count*/) {}

  bool atEnd() const {
    return rowIndex_ >= rows_.size();
  }

  const std::vector<T>& values() const {
    return values_;
  }

 private:
  TestReader reader_;
  std::vector<vector_size_t> rows_;
  vector_size_t rowIndex_{0};
  std::vector<T> values_;
};

TEST_F(HuffmanEncodingTest, factoryRoundTripAllIntegralTypes) {
  verifyFactoryRoundTrip<int8_t>();
  verifyFactoryRoundTrip<uint8_t>();
  verifyFactoryRoundTrip<int16_t>();
  verifyFactoryRoundTrip<uint16_t>();
  verifyFactoryRoundTrip<int32_t>();
  verifyFactoryRoundTrip<uint32_t>();
  verifyFactoryRoundTrip<int64_t>();
  verifyFactoryRoundTrip<uint64_t>();
}

TEST_F(HuffmanEncodingTest, signedRoundTrip) {
  Vector<int32_t> values{pool_.get()};
  for (uint32_t i = 0; i < 700; ++i) {
    values.push_back(i % 10 == 0 ? -7 : static_cast<int32_t>(i % 4));
  }

  auto encoding = encode(values);
  Vector<int32_t> result{pool_.get(), values.size()};
  encoding->materialize(static_cast<uint32_t>(values.size()), result.data());

  EXPECT_TRUE(std::equal(result.begin(), result.end(), values.begin()));
}

TEST_F(HuffmanEncodingTest, skipAndPartialReadsCrossCheckpoints) {
  Vector<uint64_t> values{pool_.get()};
  for (uint32_t i = 0; i < 900; ++i) {
    values.push_back(i % 17);
  }

  auto encoding = encode(values);
  encoding->skip(250);
  Vector<uint64_t> first{pool_.get(), 40};
  encoding->materialize(static_cast<uint32_t>(first.size()), first.data());
  EXPECT_TRUE(std::equal(first.begin(), first.end(), values.begin() + 250));

  encoding->reset();
  encoding->skip(510);
  Vector<uint64_t> second{pool_.get(), 300};
  encoding->materialize(static_cast<uint32_t>(second.size()), second.data());
  EXPECT_TRUE(std::equal(second.begin(), second.end(), values.begin() + 510));
}

TEST_F(HuffmanEncodingTest, skipRejectsPastEnd) {
  Vector<uint32_t> values{pool_.get()};
  values.push_back(1);
  values.push_back(2);

  auto encoding = encode(values);
  encoding->skip(2);
  NIMBLE_ASSERT_THROW(encoding->skip(1), "(1 vs. 0)");
}

TEST_F(HuffmanEncodingTest, resetAndMaterializeOneAtATime) {
  Vector<int32_t> values{pool_.get()};
  for (uint32_t i = 0; i < 513; ++i) {
    values.push_back(static_cast<int32_t>(i % 7) - 3);
  }

  auto encoding = encode(values);
  for (uint32_t i = 0; i < values.size(); ++i) {
    int32_t result{0};
    encoding->materialize(1, &result);
    EXPECT_EQ(result, values[i]) << "row=" << i;
  }

  encoding->reset();
  Vector<int32_t> result{pool_.get(), values.size()};
  encoding->materialize(static_cast<uint32_t>(values.size()), result.data());
  EXPECT_TRUE(std::equal(result.begin(), result.end(), values.begin()));
}

TEST_F(HuffmanEncodingTest, readWithVisitorDenseAndSparseAcrossCheckpoints) {
  Vector<int64_t> values{pool_.get()};
  for (uint32_t i = 0; i < 1025; ++i) {
    values.push_back(i % 13 == 0 ? -9 : static_cast<int64_t>(i % 5));
  }
  const auto encoded = encodeData(values);

  struct TestCase {
    std::string name;
    std::vector<vector_size_t> rows;
  };
  for (const auto& testCase : std::vector<TestCase>{
           {"dense around checkpoints", {254, 255, 256, 257, 510, 511, 512}},
           {"sparse across checkpoints", {0, 17, 255, 513, 768, 1024}},
       }) {
    SCOPED_TRACE(testCase.name);
    HuffmanEncoding<int64_t> encoding{*pool_, encoded};
    IntegralReadWithVisitor<int64_t> visitor{testCase.rows};
    ReadWithVisitorParams params;
    params.numScanned = 0;

    encoding.readWithVisitor(visitor, params);

    ASSERT_EQ(visitor.values().size(), testCase.rows.size());
    for (size_t i = 0; i < testCase.rows.size(); ++i) {
      EXPECT_EQ(visitor.values()[i], values[testCase.rows[i]]);
    }
  }
}

TEST_F(HuffmanEncodingTest, estimateRejectsFewerThanTwoRows) {
  const std::vector<uint32_t> empty;
  const auto emptyStatistics = Statistics<uint32_t>::create(empty);
  EXPECT_EQ(
      HuffmanEncoding<uint32_t>::estimateSize(empty, emptyStatistics),
      std::nullopt);

  const std::vector<uint32_t> single{7};
  const auto singleStatistics = Statistics<uint32_t>::create(single);
  EXPECT_EQ(
      HuffmanEncoding<uint32_t>::estimateSize(single, singleStatistics),
      std::nullopt);
}

TEST_F(HuffmanEncodingTest, estimateRejectsUnsupportedCardinality) {
  const std::vector<uint32_t> singleValue(100, 7);
  EXPECT_EQ(
      HuffmanEncoding<uint32_t>::estimateSize(
          singleValue, Statistics<uint32_t>::create(singleValue)),
      std::nullopt);

  std::vector<uint32_t> tooManySymbols(
      HuffmanEncoding<uint32_t>::kMaxSymbols + 1);
  std::iota(tooManySymbols.begin(), tooManySymbols.end(), 0);
  EXPECT_EQ(
      HuffmanEncoding<uint32_t>::estimateSize(
          tooManySymbols, Statistics<uint32_t>::create(tooManySymbols)),
      std::nullopt);
}

TEST_F(HuffmanEncodingTest, estimateSizeIncludesCodesAndCheckpoints) {
  std::vector<uint32_t> values;
  values.insert(values.end(), 128, 0);
  values.insert(values.end(), 64, 1);
  values.insert(values.end(), 64, 2);

  const auto estimate = HuffmanEncoding<uint32_t>::estimateSize(
      values, Statistics<uint32_t>::create(values));
  const uint64_t bitstreamBytes = 48 + 4;
  const uint64_t expected =
      EncodingPrefix::serializedSize(256, /*useVarint=*/false) +
      varint::varintSize(3) + 1 + 3 * sizeof(uint32_t) + 3 +
      varint::varintSize(1) + 2 + varint::varintSize(bitstreamBytes) +
      bitstreamBytes;
  EXPECT_EQ(estimate, expected);

  values.push_back(0);
  const auto estimateAcrossCheckpoint = HuffmanEncoding<uint32_t>::estimateSize(
      values, Statistics<uint32_t>::create(values));
  const uint64_t bitstreamBytesAcrossCheckpoint = 65 + 4;
  const uint64_t expectedAcrossCheckpoint =
      EncodingPrefix::serializedSize(257, /*useVarint=*/false) +
      varint::varintSize(3) + 1 + 3 * sizeof(uint32_t) + 3 +
      varint::varintSize(2) + 4 +
      varint::varintSize(bitstreamBytesAcrossCheckpoint) +
      bitstreamBytesAcrossCheckpoint;
  EXPECT_EQ(estimateAcrossCheckpoint, expectedAcrossCheckpoint);
}

} // namespace
} // namespace facebook::nimble::test
